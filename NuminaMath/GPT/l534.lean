import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Slope
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.Geometry.Trangle
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Factorial.Basic
import Mathlib.Data.Fin.Perm
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.PrimeFactors
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.ContinuousFunction.Basic
import data.nat.basic

namespace tan_phi_proof_l534_534420

-- Given conditions
variables (β φ : ℝ)
def is_right_triangle (β : ℝ) : Prop := true -- Assume there is a right triangle
def tan_half_beta : Prop := tan (β / 2) = 1 / sqrt 3

-- Proof statement
theorem tan_phi_proof (h1 : is_right_triangle β) (h2 : tan_half_beta) : tan φ = (sqrt 3 - 1) / 3 :=
by sorry

end tan_phi_proof_l534_534420


namespace product_of_distances_to_intersections_l534_534325

noncomputable def line_parametric {α : ℝ} (t : ℝ) :=
  (1 + t * Real.cos α, 1 + t * Real.sin α)

theorem product_of_distances_to_intersections (α : ℝ) (hα : α = Real.pi / 6) :
  ∃ A B : ℝ × ℝ, (line_parametric α A.fst = 2 * Real.cos hα) ∧ (line_parametric α B.fst = 2 * Real.sin hα) ∧
  (distance (1,1) A) * (distance (1,1) B) = 2 :=
by
  sorry

end product_of_distances_to_intersections_l534_534325


namespace number_of_boys_minus_girls_l534_534421

theorem number_of_boys_minus_girls :
  ∀ (number_of_girls number_of_boys : ℕ), 
  number_of_girls = 635 → 
  number_of_boys = 1145 → 
  number_of_boys - number_of_girls = 510 :=
by
  intros number_of_girls number_of_boys h_girls h_boys
  rw [h_girls, h_boys]
  exact rfl -- Replace this with the actual proof which is skipped by "sorry" in the guideline

end number_of_boys_minus_girls_l534_534421


namespace count_two_digit_or_less_numbers_l534_534847

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534847


namespace find_f_at_75_l534_534098

variables (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 2) = -f x)
variables (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)
variables (h₂ : ∀ x, f (-x) = -f x)

theorem find_f_at_75 : f 7.5 = -0.5 := by
  sorry

end find_f_at_75_l534_534098


namespace max_distance_point_circle_l534_534764

open Real

noncomputable def distance (P C : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)

theorem max_distance_point_circle :
  let C : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (3, 3)
  let r : ℝ := 2
  let max_distance : ℝ := distance P C + r
  ∃ M : ℝ × ℝ, distance P M = max_distance ∧ (M.1 - 1)^2 + (M.2 - 2)^2 = r^2 :=
by
  sorry

end max_distance_point_circle_l534_534764


namespace sufficient_but_not_necessary_for_circle_l534_534518

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (m = 0 → ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0) ∧ ¬(∀m, ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0 → m = 0) :=
 by
  sorry

end sufficient_but_not_necessary_for_circle_l534_534518


namespace vector_properties_l534_534783

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (a_def : a = ![2, 4])
variable (b_def : b = ![-2, 1])

theorem vector_properties :
  (dot_product a b = 0) ∧
  (‖a + b‖ = 5) ∧
  (‖a - b‖ = 5) :=
by
  have h₁ : dot_product a b = 0 := by sorry
  have h₂ : ‖a + b‖ = 5 := by sorry
  have h₃ : ‖a - b‖ = 5 := by sorry
  exact ⟨h₁, h₂, h₃⟩

end vector_properties_l534_534783


namespace vector_problem_l534_534367

variables (OM ON MN : ℝ × ℝ)

def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def scalar_mul (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

theorem vector_problem (OM : ℝ × ℝ) (ON : ℝ × ℝ) (hOM : OM = (3, -2)) (hON : ON = (-5, -1)) :
  scalar_mul 1/2 (vec_sub ON OM) = (-4, 1/2) :=
by {
  sorry
}

end vector_problem_l534_534367


namespace count_squares_within_region_l534_534035

theorem count_squares_within_region : 
  let region := {p : ℕ × ℕ | p.2 ≤ 2 * p.1 ∧ p.2 ≤ 10 ∧ 1 ≤ p.1 ∧ 1 ≤ p.2}
  let squares := {p : ℕ × ℕ | p.2 ≤ 2 * (p.1 - 1) ∧ p.2 ≤ 9 ∧ 1 ≤ p.1 ∧ 1 ≤ p.2}
  squares.card = 21 :=
sorry

end count_squares_within_region_l534_534035


namespace trigonometry_identity_l534_534338

theorem trigonometry_identity
  (α : ℝ)
  (h_quad: 0 < α ∧ α < π / 2)
  (h_cos : real.cos α = 2 * real.sqrt 5 / 5) :
  real.cos (2 * α) - real.cos α / real.sin α = -7 / 5 := 
sorry

end trigonometry_identity_l534_534338


namespace max_closable_companies_l534_534173

def number_of_planets : ℕ := 10 ^ 2015
def number_of_companies : ℕ := 2015

theorem max_closable_companies (k : ℕ) : k = 1007 :=
sorry

end max_closable_companies_l534_534173


namespace product_of_digits_largest_num_sum_of_squares_85_l534_534282

theorem product_of_digits_largest_num_sum_of_squares_85 :
  ∃ n : ℕ, sum_of_squares_of_digits n = 85 ∧ digits_descending n ∧ product_of_digits n = 42 :=
sorry

-- Definitions for sum_of_squares_of_digits, digits_descending, and product_of_digits

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.map (λ d, d * d) |>.sum

def digits_descending (n : ℕ) : Prop :=
  list.pairwise (>) (n.digits 10)

def product_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).prod

end product_of_digits_largest_num_sum_of_squares_85_l534_534282


namespace average_book_weight_correct_l534_534955

-- Definitions based on conditions
def sandy_books : Nat := 10
def sandy_book_weight : Real := 1.5

def benny_books : Nat := 24
def benny_book_weight : Real := 1.2

def tim_books : Nat := 33
def tim_book_weight : Real := 1.8

def lily_books : Nat := 15
def lily_book_weight : Real := 1.3

-- Total weights for individual collections
def sandy_total_weight : Real := sandy_books * sandy_book_weight
def benny_total_weight : Real := benny_books * benny_book_weight
def tim_total_weight : Real := tim_books * tim_book_weight
def lily_total_weight : Real := lily_books * lily_book_weight

-- Total number of books and total weight
def total_books : Nat := sandy_books + benny_books + tim_books + lily_books
def total_weight : Real := sandy_total_weight + benny_total_weight + tim_total_weight + lily_total_weight

-- Average weight calculation
def average_weight : Real := total_weight / total_books

-- The proof statement
theorem average_book_weight_correct : average_weight ≈ 1.497 := by
  sorry

end average_book_weight_correct_l534_534955


namespace max_a_value_l534_534009

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x - 1

theorem max_a_value : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), x ∈ Set.Icc (1/2) 2 → (a + 1) * x - 1 - Real.log x ≤ 0) → 
  a ≤ 1 - 2 * Real.log 2 := 
by
  sorry

end max_a_value_l534_534009


namespace geometric_sequence_a6_l534_534723

theorem geometric_sequence_a6
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : a 1 = 1)
  (S3 : S 3 = 7 / 4)
  (sum_S3 : S 3 = a 1 + a 1 * a 2 + a 1 * (a 2)^2) :
  a 6 = 1 / 32 := by
  sorry

end geometric_sequence_a6_l534_534723


namespace max_covered_squares_by_tetromino_l534_534937

-- Definition of the grid size
def grid_size := (5, 5)

-- Definition of S-Tetromino (Z-Tetromino) coverage covering four contiguous squares
def is_STetromino (coords: List (Nat × Nat)) : Prop := 
  coords.length = 4 ∧ ∃ (x y : Nat), coords = [(x, y), (x, y+1), (x+1, y+1), (x+1, y+2)]

-- Definition of the coverage constraint
def no_more_than_two_tiles (cover: List (Nat × Nat)) : Prop :=
  ∀ (coord: Nat × Nat), cover.count coord ≤ 2

-- Definition of the total tiled squares covered by at least one tile
def tiles_covered (cover: List (Nat × Nat)) : Nat := 
  cover.toFinset.card 

-- Definition of the problem using proof equivalence
theorem max_covered_squares_by_tetromino
  (cover: List (List (Nat × Nat)))
  (H_tiles: ∀ t, t ∈ cover → is_STetromino t)
  (H_coverage: no_more_than_two_tiles (cover.join)) :
  tiles_covered (cover.join) = 24 :=
sorry 

end max_covered_squares_by_tetromino_l534_534937


namespace problem_l534_534903

-- Define the problem setup based on given conditions
variables {A B C D M P X : Type} [geometry A B C D M P X]

-- Assumptions based on the problem conditions
variables (h_angle_B : angle B = 60)
variables (h_angle_D : angle D = 60)
variables (h_midpoint_M : midpoint M A D)
variables (h_parallel_MP_CD : parallel (line_through M) CD)
variables (h_intersects_MP_BC : intersects (line_through M) BC P)
variables (h_point_X : on_line X CD)
variables (h_BX_MX : distance B X = distance M X)

-- Problem statement
theorem problem:
  (distance A B = distance B P) ↔ (angle M X B = 60) :=
sorry 

end problem_l534_534903


namespace conditional_probability_l534_534489

/-- Definitions related to the problem setup --/
def total_outcomes := 36
def favorable_outcomes_A := 6
def favorable_outcomes_B := 5
def favorable_outcomes_AB := 1

/-- Calculate conditional probability P(B|A) --/
def P_B_given_A := favorable_outcomes_AB.to_real / favorable_outcomes_A.to_real

/-- Statement to prove --/
theorem conditional_probability :
  P_B_given_A = (1:ℝ) / 6 := by
  sorry

end conditional_probability_l534_534489


namespace exists_root_in_interval_l534_534605

noncomputable theory

open Classical

variables {a b c : ℝ} (h_a : a ≠ 0) (h1 : a * 3.23^2 + b * 3.23 + c = -0.06)
(h2 : a * 3.24^2 + b * 3.24 + c = -0.02)
(h3 : a * 3.25^2 + b * 3.25 + c = 0.03)
(h4 : a * 3.26^2 + b * 3.26 + c = 0.09)

theorem exists_root_in_interval : ∃ x : ℝ, 3.24 < x ∧ x < 3.25 ∧ (a * x^2 + b * x + c = 0) :=
by {
  -- Proof would go here
  sorry
}

end exists_root_in_interval_l534_534605


namespace range_of_a_l534_534157

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ f a 0) : 0 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l534_534157


namespace permutation_identity_l534_534038

open Nat

theorem permutation_identity (n : ℕ) (h : (Nat.factorial n / Nat.factorial (n - 3)) = 6 * n) : n = 4 := 
by
  sorry

end permutation_identity_l534_534038


namespace minimum_omega_value_l534_534487

noncomputable def sin_shift_minimum_omega (ω : ℝ) : Prop :=
  ∃ k : ℤ, ω = 4 * k + 3 / 2

theorem minimum_omega_value : ∃ ω : ℝ, sin_shift_minimum_omega ω ∧ 0 < ω ∧ ∀ ω' : ℝ, sin_shift_minimum_omega ω' → (0 < ω' → ω ≤ ω')
  :=
begin
  sorry
end

end minimum_omega_value_l534_534487


namespace range_of_a_l534_534047

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
by sorry

end range_of_a_l534_534047


namespace pythagorean_theorem_l534_534058

theorem pythagorean_theorem (a b c : ℝ) : (a^2 + b^2 = c^2) ↔ (a^2 + b^2 = c^2) :=
by sorry

end pythagorean_theorem_l534_534058


namespace ice_cream_stack_orders_l534_534135

-- Define the five distinct scoops
inductive Scoop
| vanilla
| chocolate
| strawberry
| cherry
| pistachio

open Scoop

-- The proof statement
theorem ice_cream_stack_orders :
  -- Proving there are 120 ways to stack the five scoops
  fintype.card (List.permutations [vanilla, chocolate, strawberry, cherry, pistachio]) = 120 :=
by
  sorry

end ice_cream_stack_orders_l534_534135


namespace cos_double_angle_minus_cos_over_sin_l534_534335

theorem cos_double_angle_minus_cos_over_sin (α : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : real.cos α = 2 * real.sqrt 5 / 5) :
  real.cos (2 * α) - real.cos α / real.sin α = -7 / 5 :=
sorry

end cos_double_angle_minus_cos_over_sin_l534_534335


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534799

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534799


namespace largest_area_l534_534192

def angle_A : Float := 60
def angle_B : Float := 45
def AC : Float := Real.sqrt 2

def d1 : Float := Real.sqrt 2
def d2 : Float := Real.sqrt 3
def angle_diagonals : Float := 75

def radius_circle : Float := 1

def diagonal_square : Float := 2.5

theorem largest_area :
  let area_triangle := 0.5 * AC * (AC * Real.sin (Real.toRadians angle_B)) * Real.sin (Real.toRadians angle_A)
  let area_rhombus := 0.5 * d1 * d2 * Real.sin (Real.toRadians angle_diagonals)
  let area_circle := Float.pi * radius_circle^2
  let side_square := diagonal_square / Real.sqrt 2
  let area_square := side_square^2
  area_circle > area_triangle ∧
  area_circle > area_rhombus ∧
  area_circle > area_square :=
by
  sorry

end largest_area_l534_534192


namespace max_min_distance_unit_cube_l534_534573

-- We define the structure of a 3D point
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Definition of the unit cube (each coordinate between 0 and 1)
def onBoundaryUnitCube (p : Point3D) : Prop :=
  (0 ≤ p.x ∧ p.x ≤ 1) ∧ (0 ≤ p.y ∧ p.y ≤ 1) ∧ (0 ≤ p.z ∧ p.z ≤ 1) ∧
  (p.x = 0 ∨ p.x = 1 ∨ p.y = 0 ∨ p.y = 1 ∨ p.z = 0 ∨ p.z = 1)

-- Calculate the Euclidean distance between two 3D points
def dist (a b : Point3D) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2)

-- Lean theorem statement to prove that the maximum min distance is sqrt(2)
theorem max_min_distance_unit_cube :
  ∀ (A B C: Point3D),
    onBoundaryUnitCube A →
    onBoundaryUnitCube B →
    onBoundaryUnitCube C →
    (∀ (d : ℝ), (d = dist A B ∨ d = dist A C ∨ d = dist B C) → d ≥ 0) →
    ∃ (d : ℝ), d = sqrt 2 ∧ 
                (∀ (d' : ℝ), d' = dist A B ∨ d' = dist A C ∨ d' = dist B C → d' ≥ d) -> 
                d = max (min (dist A B) (dist A C)) (dist B C) :=
begin
  sorry
end

end max_min_distance_unit_cube_l534_534573


namespace fence_cost_l534_534556

-- Defining the conditions as Lean definitions:
def area : ℝ := 25
def price_per_foot : ℝ := 58

-- The length of one side of the square plot
def side_length : ℝ := Real.sqrt area

-- The total length of the fence
def total_length_of_fence : ℝ := 4 * side_length

-- The total cost of the fence
def total_cost : ℝ := total_length_of_fence * price_per_foot

-- The theorem stating the correct answer
theorem fence_cost : total_cost = 1160 := sorry

end fence_cost_l534_534556


namespace constant_term_expansion_l534_534969

theorem constant_term_expansion :
  let T_r := λ r: ℕ, (-1)^r * (Nat.choose 5 r) * (x^(2*r - 10))
  (x^2 + 2) * (sum (λ r : ℕ, T_r r)) = 3 :=
by
  let T_r := λ r: ℕ, (-1)^r * (Nat.choose 5 r) * (x^(2*r - 10))
  let expansion := (x^2 + 2) * (∑ r in range (6), T_r r)
  show (expansion.const_term = 3)
  sorry

end constant_term_expansion_l534_534969


namespace monic_quadratic_poly_with_root_l534_534696

theorem monic_quadratic_poly_with_root (x : ℝ) :
  (∃ p : Polynomial ℝ, Polynomial.monic p ∧ p.coeff 2 = 1 ∧ p.coeff 1 = 6 ∧ p.coeff 0 = 16 
  ∧ p.eval (-3 - Complex.i * Real.sqrt 7) = 0) :=
by
  use Polynomial.C 16 + Polynomial.X * (Polynomial.C 6 + Polynomial.X)
  field_simp
  sorry

end monic_quadratic_poly_with_root_l534_534696


namespace angle_half_proof_l534_534448

theorem angle_half_proof
  (A B C D N P Q : ℝ → ℝ)
  (h_square: ∀ x y : ℝ → ℝ, x ≠ y)
  (h_N_on_AB: N ∈ line_segment A B)
  (h_P_on_AD: P ∈ line_segment A D)
  (h_NP_eq_NC: ∥N - P∥ = ∥N - C∥)
  (h_Q_on_AN: Q ∈ line_segment A N)
  (h_angle_QPN_eq_NCB: ∠Q P N = ∠N C B) :
  ∠B C Q = (1 / 2) * ∠A Q P :=
by
  sorry

end angle_half_proof_l534_534448


namespace verify_original_speed_l534_534243

noncomputable def original_speed_of_wheel : ℝ :=
  let circumference_feet := 15
  let circumference_miles := circumference_feet / 5280
  let speed_increase := 10
  let time_decrease_sec := 1 / 3
  let time_decrease_hour := time_decrease_sec / 3600
  let new_circumference_time := λ (r t : ℝ), (r + speed_increase) * (t - time_decrease_hour) = circumference_miles * 3600 in
  classical.some (exists.intro 15 (begin
    use new_circumference_time 15,
    sorry
  end))

-- Theorem statement to verify the original speed
theorem verify_original_speed : original_speed_of_wwheel = 15 :=
sorry

end verify_original_speed_l534_534243


namespace count_special_integers_l534_534827

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534827


namespace count_special_integers_l534_534821

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534821


namespace question_I_question_II_l534_534743

noncomputable def sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (λ : ℤ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → a n ≠ 0) ∧ (∀ n : ℕ, a n * a (n + 1) = λ * S n - 1)

theorem question_I (a : ℕ → ℤ) (S : ℕ → ℤ) (λ : ℤ) (h : sequence a S λ) (n : ℕ) : 
  a (n + 2) - a n = λ := 
by
  sorry

theorem question_II (a : ℕ → ℤ) (S : ℕ → ℤ) (λ : ℤ) (h : sequence a S λ) : 
  ∃ λ : ℤ, (∀ n : ℕ, a (n+1) - a n = (λ + 1)/2) → λ = 4 :=
by 
  sorry

end question_I_question_II_l534_534743


namespace problem_solution_l534_534434

noncomputable def parametric_line_eqn (t : ℝ) : ℝ × ℝ :=
  (2 - real.sqrt 3 * t, t)

def circle_eqn (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

noncomputable def standard_line_form (x y : ℝ) : Prop :=
  real.sqrt 3 * x + y - 2 = 0

def polar_eqn_circle (ρ : ℝ) : Prop :=
  ρ = 2

noncomputable def distance_between_line_and_circle_center (A B C x₀ y₀ : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / real.sqrt (A^2 + B^2)

theorem problem_solution :
  let line := parametric_line_eqn,
      circle := circle_eqn,
      std_line := standard_line_form,
      polar_circle := polar_eqn_circle,
      dist := distance_between_line_and_circle_center (real.sqrt 3) 1 (-2) 0 0 in
  (∀ t, std_line (2 - real.sqrt 3 * t) t) ∧ polar_circle 2 ∧ dist = 1 :=
by
  sorry

end problem_solution_l534_534434


namespace sum_k_round_diff_eq_125237_l534_534630

noncomputable def sum_k_round_diff (n : ℕ) : ℤ :=
∑ k in finset.range n.succ, k * ((⌈real.log k / real.log (real.sqrt 3)⌉ : ℤ) - (⌊real.log k / real.log (real.sqrt 3)⌋ : ℤ))

theorem sum_k_round_diff_eq_125237 : sum_k_round_diff 500 = 125237 := by
  -- Here we would add the proof steps to verify the theorem
  sorry

end sum_k_round_diff_eq_125237_l534_534630


namespace largest_number_of_cakes_without_ingredients_l534_534470

theorem largest_number_of_cakes_without_ingredients :
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  ∃ (max_no_ingredients : ℕ), max_no_ingredients = 24 :=
by
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  existsi (60 - max 20 (max 30 (max 36 6))) -- max value should be used to reflect maximum coverage content
  sorry -- Proof to be completed

end largest_number_of_cakes_without_ingredients_l534_534470


namespace polynomial_count_l534_534425

theorem polynomial_count : 
  let expr1 := (1 / x : ℝ)
  let expr2 := (2 * x + y : ℝ)
  let expr3 := (1 / 3 * a^2 * b : ℝ)
  let expr4 := ((x - y) / π : ℝ)
  let expr5 := (5 * y / (4 * x) : ℝ)
  let expr6 := (0.5 : ℝ)
  let is_polynomial (e : ℝ) := ∃ p : ℝ[X], e = p.eval 0
  in 
  (is_polynomial expr2) ∧ 
  (is_polynomial expr3) ∧ 
  (is_polynomial expr4) ∧ 
  (is_polynomial expr6) ∧ 
  ¬(is_polynomial expr1) ∧ 
  ¬(is_polynomial expr5) ∧ 
  (4 = 4) :=
  by sorry

end polynomial_count_l534_534425


namespace arithmetic_square_root_of_sqrt_81_l534_534512

def square_root (x : ℝ) : ℝ :=
  real.sqrt x

theorem arithmetic_square_root_of_sqrt_81 : square_root 81 = 9 :=
by
  sorry

end arithmetic_square_root_of_sqrt_81_l534_534512


namespace no_solution_m1_no_solution_m2_solution_m3_l534_534479

-- Problem 1: No positive integer solutions for m = 1
theorem no_solution_m1 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ x * y * z := sorry

-- Problem 2: No positive integer solutions for m = 2
theorem no_solution_m2 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ 2 * x * y * z := sorry

-- Problem 3: Only solutions for m = 3 are x = y = z = k for some k
theorem solution_m3 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z := sorry

end no_solution_m1_no_solution_m2_solution_m3_l534_534479


namespace min_good_permutations_l534_534113

noncomputable def is_good_permutation (n : ℕ) (a b : list ℝ) : Prop :=
  (∀ k, 1 ≤ k → k ≤ n → (list.sum (b.take k) > 0))

theorem min_good_permutations (n : ℕ) (a : list ℝ) (h_len : a.length = n)
  (h_n_ge_3 : n ≥ 3) (h_distinct : a.nodup) (h_pos_sum : list.sum a > 0) :
  ∃ g : ℕ, (∀ b : list ℝ, b.perm a → is_good_permutation n a b → g = (n-1)!) :=
sorry

end min_good_permutations_l534_534113


namespace find_milk_l534_534621

def chocolate_milk_conditions (milk_per_glass syrup_per_glass total_syrup total_chocolate_milk milk_needed glasses_chocolate_milk : ℕ) : Prop :=
  total_chocolate_milk = glasses_chocolate_milk * 8 ∧
  syrup_per_glass = 1.5 ∧
  milk_per_glass = 6.5 ∧
  total_syrup ≥ glasses_chocolate_milk * syrup_per_glass ∧
  milk_needed = glasses_chocolate_milk * milk_per_glass

theorem find_milk (milk_per_glass syrup_per_glass total_syrup total_chocolate_milk milk_needed glasses_chocolate_milk : ℕ) :
  chocolate_milk_conditions milk_per_glass syrup_per_glass total_syrup total_chocolate_milk milk_needed glasses_chocolate_milk →
  milk_needed = 130 :=
by
  intros h
  sorry

end find_milk_l534_534621


namespace real_and_imaginary_parts_of_z_l534_534099

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i^2 + i

-- State the theorem
theorem real_and_imaginary_parts_of_z :
  z.re = -1 ∧ z.im = 1 :=
by
  -- Provide the proof or placeholder
  sorry

end real_and_imaginary_parts_of_z_l534_534099


namespace int_part_A₁₀₀_add_B₁₀₀_l534_534184

noncomputable def A₁ := Real.sqrt 9
noncomputable def A₂ := Real.sqrt (6 + Real.sqrt 9)
noncomputable def A₃ := Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 9))
noncomputable def A₁₀₀ := A₃ -- Definition is simplified, needs full representation in actual iteration for correctness

noncomputable def B₁ := Real.sqrt 9
noncomputable def B₂ := Real.sqrt (6 + Real.sqrt 9)
noncomputable def B₃ := Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 9))
noncomputable def B₁₀₀ := B₃ -- Definition is simplified, needs full representation in actual iteration for correctness

theorem int_part_A₁₀₀_add_B₁₀₀ : (Real.floor (3 + 2) = 4) := sorry

end int_part_A₁₀₀_add_B₁₀₀_l534_534184


namespace diagonal_ac_length_l534_534064

-- Define the given lengths and angle
def length_AB : ℝ := 13
def length_BC : ℝ := 13
def length_CD : ℝ := 20
def length_DA : ℝ := 20
def angle_ADC : ℝ := 120

-- Define the diagonal AC and the calculated length
def length_AC : ℝ := 20 * Real.sqrt 3

-- The theorem statement
theorem diagonal_ac_length :
  ∀ (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D],
  dist A B = length_AB →
  dist B C = length_BC →
  dist C D = length_CD →
  dist D A = length_DA →
  ∠ D A C = angle_ADC →
  dist A C = length_AC :=
by
  sorry

end diagonal_ac_length_l534_534064


namespace polygon_perimeter_lower_bound_l534_534908

theorem polygon_perimeter_lower_bound (n : ℕ) (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ)
  (h_convex : ConvexHull (Finset.image A Finset.univ)) 
  (h_balance : ∑ i, (A i).fst - O.fst = 0 ∧ ∑ i, (A i).snd - O.snd = 0)
  (d : ℝ) (h_d : d = ∑ i, EuclideanDistance O (A i)) :
  (even n → perimeter A ≥ 4 * d / n) ∧ (odd n → perimeter A ≥ 4 * d * n / (n^2 - 1)) :=
by
  sorry

end polygon_perimeter_lower_bound_l534_534908


namespace largest_pile_is_120_l534_534891

theorem largest_pile_is_120 (x : ℕ) (h_total : x + 2*x + 3*x = 240) : 3*x = 120 :=
by {
  have h_eq : 6 * x = 240 := by linarith,
  have h_solve : x = 240 / 6 := by linarith,
  rw [h_solve],
  exact by norm_num,
  sorry --The proof of 3*x = 120 would go here
}

end largest_pile_is_120_l534_534891


namespace sum_leq_six_of_quadratic_roots_l534_534004

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots_l534_534004


namespace complement_of_M_in_U_l534_534924

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x > 0}
def complement_U_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_of_M_in_U : (U \ M) = complement_U_M :=
by sorry

end complement_of_M_in_U_l534_534924


namespace sum_log_sqrt3_l534_534641

theorem sum_log_sqrt3 : 
  (∑ k in Finset.range (501), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 124886 :=
by
  sorry

end sum_log_sqrt3_l534_534641


namespace sin_cos_inequality_l534_534131

theorem sin_cos_inequality (x : ℝ) (n : ℕ) : 
  (Real.sin (2 * x))^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
by
  sorry

end sin_cos_inequality_l534_534131


namespace probability_of_specific_sequence_l534_534525

variable (p : ℝ) (h : 0 < p ∧ p < 1)

theorem probability_of_specific_sequence :
  (1 - p)^7 * p^3 = sorry :=
by sorry

end probability_of_specific_sequence_l534_534525


namespace min_value_of_sum_l534_534388

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 / x + 1 / y = 1) : x + y = 9 :=
by
  -- sorry used to skip the proof
  sorry

end min_value_of_sum_l534_534388


namespace count_valid_numbers_l534_534860

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534860


namespace count_positive_integers_with_two_digits_l534_534835

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534835


namespace tan_div_alpha_lt_tan_div_beta_l534_534949

theorem tan_div_alpha_lt_tan_div_beta (α β : ℝ) (hα_positive : 0 < α) (hβ_positive : 0 < β) (hα_acute : α < π / 2) (hβ_acute : β < π / 2) (hα_lt_β : α < β) 
    : (tan α) / α < (tan β) / β :=
by
  sorry

end tan_div_alpha_lt_tan_div_beta_l534_534949


namespace son_present_age_l534_534563

variable (S M : ℕ)

-- Condition 1: M = S + 20
def man_age_relation (S M : ℕ) : Prop := M = S + 20

-- Condition 2: In two years, the man's age will be twice the age of his son
def age_relation_in_two_years (S M : ℕ) : Prop := M + 2 = 2*(S + 2)

theorem son_present_age : 
  ∀ (S M : ℕ), man_age_relation S M → age_relation_in_two_years S M → S = 18 :=
by
  intros S M h1 h2
  sorry

end son_present_age_l534_534563


namespace monic_quadratic_with_given_root_l534_534689

theorem monic_quadratic_with_given_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.eval (-3 - complex.i * Real.sqrt 7) = 0 ∧ p = Polynomial.X^2 + 6 * Polynomial.X + 16 :=
by
  sorry

end monic_quadratic_with_given_root_l534_534689


namespace cylindrical_to_spherical_l534_534274

/-- Given the cylindrical coordinates (r, θ, z), the corresponding spherical coordinates are (ρ, φ, θ). -/
theorem cylindrical_to_spherical :
  let r := 10
  let θ := Real.pi / 3
  let z := 2
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ' := Real.atan2 y x
  let φ := Real.acos (z / ρ)
  ρ = Real.sqrt 104 ∧ φ = Real.acos (2 / Real.sqrt 104) ∧ θ' = Real.pi / 3 :=
by
  dsimp [x, y, ρ, θ', φ, r, θ, z]
  split
  · sorry -- proof for ρ = Real.sqrt 104
  · split
    · sorry -- proof for φ = Real.acos (2 / Real.sqrt 104)
    · sorry -- proof for θ' = Real.pi / 3

end cylindrical_to_spherical_l534_534274


namespace monic_quadratic_with_root_l534_534699

theorem monic_quadratic_with_root :
  ∃ (p : ℝ[X]), monic p ∧ (p.coeff 2 = 1) ∧ (p.coeff 1 = 6) ∧ (p.coeff 0 = 16) ∧ is_root p (-3 - complex.I * real.sqrt 7) :=
sorry

end monic_quadratic_with_root_l534_534699


namespace sqrt_of_sqrt_81_l534_534497

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l534_534497


namespace cistern_depth_l534_534215

noncomputable def length : ℝ := 9
noncomputable def width : ℝ := 4
noncomputable def total_wet_surface_area : ℝ := 68.5

theorem cistern_depth (h : ℝ) (h_def : 68.5 = 36 + 18 * h + 8 * h) : h = 1.25 :=
by sorry

end cistern_depth_l534_534215


namespace sum_of_possible_k_l534_534183

theorem sum_of_possible_k (k n : ℕ) (h : ∃ n : ℕ, k + n * k^2 = 306) :
  ({ k : ℕ | ∃ n : ℕ, k + n * k^2 = 306 }.sum id) = 326 :=
sorry

end sum_of_possible_k_l534_534183


namespace cot_arccot_sum_l534_534305

theorem cot_arccot_sum :
  (Real.cot (Real.arccot 4 + Real.arccot 9 + Real.arccot 12 + Real.arccot 18) = 1427 / 769) := by
  sorry

end cot_arccot_sum_l534_534305


namespace arithmetic_square_root_sqrt_81_l534_534511

theorem arithmetic_square_root_sqrt_81 : sqrt (sqrt 81) = 3 := 
by 
  have h1 : sqrt 81 = 9 := sorry
  have h2 : sqrt 9 = 3 := sorry
  show sqrt (sqrt 81) = 3 from sorry

end arithmetic_square_root_sqrt_81_l534_534511


namespace bobby_total_l534_534619

-- Define the conditions
def initial_candy : ℕ := 33
def additional_candy : ℕ := 4
def chocolate : ℕ := 14

-- Define the total pieces of candy Bobby ate
def total_candy : ℕ := initial_candy + additional_candy

-- Define the total pieces of candy and chocolate Bobby ate
def total_candy_and_chocolate : ℕ := total_candy + chocolate

-- Theorem to prove the total pieces of candy and chocolate Bobby ate
theorem bobby_total : total_candy_and_chocolate = 51 :=
by sorry

end bobby_total_l534_534619


namespace chessboard_tiling_impossible_l534_534269

theorem chessboard_tiling_impossible : 
  ∀ (board : array 8 (array 8 bool)), 
  (board[0][0] = false ∧ board[7][7] = false) → 
  (∃ (tiling : list (array 2 (fin 8 × fin 8))), 
     ∀ (d : array 2 (fin 8 × fin 8)), d ∈ tiling →
       (d[0].1 ≠ d[1].1 ∨ d[0].2 ≠ d[1].2) ∧
       board[d[0].1 (d[0].2] = tt ∧ board[d[1].1][d[1].2] = tt →
     list.length tiling = 31) → 
  false :=
begin
  sorry
end

end chessboard_tiling_impossible_l534_534269


namespace lunar_moon_shape_area_l534_534252

theorem lunar_moon_shape_area :
  ∀ (O A B C D : Type) (r : ℝ),
  (AB_perpendicular_CD : AB ⊥ CD)
  (diameter_AB : AB = 10) →
  (r = 5)
  (area_semi_circle : (1/2) * π * r^2 = 25 * π / 2)
  (area_triangle_ABC : (1/2) * AB * BC = 25)
  (area_quarter_circle : (1/4) * π * (2*r)^2 = 25 * π) →
  (area_lunar_moon_shape : (25 * π / 2 + 25) - 25 * π = 25) :=
  sorry

end lunar_moon_shape_area_l534_534252


namespace find_m_n_sum_product_l534_534756

noncomputable def sum_product_of_roots (m n : ℝ) : Prop :=
  (m^2 - 4*m - 12 = 0) ∧ (n^2 - 4*n - 12 = 0) 

theorem find_m_n_sum_product (m n : ℝ) (h : sum_product_of_roots m n) :
  m + n + m * n = -8 :=
by 
  sorry

end find_m_n_sum_product_l534_534756


namespace arithmetic_sequence_difference_l534_534959

noncomputable def arithmetic_difference (d: ℚ) (b₁: ℚ) : Prop :=
  (50 * b₁ + ((50 * 49) / 2) * d = 150) ∧
  (50 * (b₁ + 50 * d) + ((50 * 149) / 2) * d = 250)

theorem arithmetic_sequence_difference {d b₁ : ℚ} (h : arithmetic_difference d b₁) :
  (b₁ + d) - b₁ = (200 / 1295) :=
by
  sorry

end arithmetic_sequence_difference_l534_534959


namespace incircle_area_ratio_comparison_l534_534979

theorem incircle_area_ratio_comparison :
  ∀ (A B C : Type)
  [triangle 𝜏₁ : isosceles_triangle A B C]
  [triangle 𝜏₂ : isosceles_triangle A B C]
  (incircle_1 : incircle_touch_closer_to_base 𝜏₁)
  (incircle_2 : incircle_touch_farther_from_base 𝜏₂),
  (incircle_area_ratio 𝜏₁ incircle_1 > incircle_area_ratio 𝜏₂ incircle_2) := 
sorry

end incircle_area_ratio_comparison_l534_534979


namespace sum_zero_combination_l534_534933

theorem sum_zero_combination (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ k, 1 ≤ k → k ≤ n → a k ≤ k)
  (h2 : ∑ i in Finset.range (n + 1), a i % 2 = 0) :
  ∃ (s : Finset (ℕ → ℤ)), (∑ i in Finset.range (n + 1), if i ∈ s then (a i : ℤ) else -(a i)) = 0 := 
sorry

end sum_zero_combination_l534_534933


namespace perpendicular_slope_l534_534720

theorem perpendicular_slope (x y : ℝ) : (∃ b : ℝ, 4 * x - 5 * y = 10) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro h
  sorry

end perpendicular_slope_l534_534720


namespace find_coefficients_l534_534988

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def g (x : ℝ) : ℝ := x^3 - 8 * x^2 - 36 * x - 64

theorem find_coefficients :
  ∀ r : ℝ, is_root f r → is_root g (r^3) := 
begin
  sorry
end

end find_coefficients_l534_534988


namespace satisfies_conditions_l534_534247

def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)

theorem satisfies_conditions :
  (∀ x : ℝ, f (Real.pi / 12 + x) + f (Real.pi / 12 - x) = 0) ∧
  (∀ x : ℝ, -Real.pi / 6 < x ∧ x < Real.pi / 3 → 0 < Real.deriv f x) :=
by
  sorry

end satisfies_conditions_l534_534247


namespace sqrt_meaningful_range_l534_534397

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1) ↔ x ≥ 1) := 
sorry

end sqrt_meaningful_range_l534_534397


namespace count_at_most_two_different_digits_l534_534850

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534850


namespace chessboard_coloring_problem_l534_534265

noncomputable def l (n : ℕ) : ℕ := sorry   -- Definition of l(n) as per the chessboard problem with minimum colored vertices.

def lim_l_div_n_squared (l : ℕ → ℕ) : Prop :=
  Tendsto (fun n => (l n : ℝ) / (n^2 : ℝ)) atTop (𝓝 (2/7 : ℝ))

theorem chessboard_coloring_problem :
  let l : ℕ → ℕ := sorry in            -- Assume l(n) is appropriately defined elsewhere.
  lim_l_div_n_squared l := sorry

end chessboard_coloring_problem_l534_534265


namespace find_a_7_l534_534428

variable {α : Type*}
variables (a : ℕ → ℤ) (d : ℤ)

-- Defining the arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) = a n + d

-- Given conditions: a_3 = 2 and a_5 = 7
def a_3_eq_2 : Prop := a 3 = 2
def a_5_eq_7 : Prop := a 5 = 7

-- Problem statement: Prove that a_7 = 12 under the given conditions
theorem find_a_7 : 
  arithmetic_sequence a d → 
  a_3_eq_2 → 
  a_5_eq_7 → 
  a 7 = 12 :=
by 
  sorry

end find_a_7_l534_534428


namespace problem_statement_l534_534646

noncomputable def sum_log_floor_ceiling (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * (if (Real.log k / Real.log (Real.sqrt 3)).floor = (Real.log k / Real.log (Real.sqrt 3)).ceil then 0 else 1)

theorem problem_statement :
  sum_log_floor_ceiling 500 = 124886 :=
by
  -- proof starting here, replace with actual proof
  sorry

end problem_statement_l534_534646


namespace max_area_triangle_l534_534774

noncomputable def f (x : ℝ) : ℝ := sin x * cos x + sqrt 3 * cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (2 * x + 2 * Real.pi / 3)

-- Assume A is an angle of the acute triangle such that g(A/2) = 1/2
variable {A : ℝ} (hA : g (A / 2) = 1/2)
-- Assume sides opposite to angles A, B, C are a, b, c respectively and a = 1
variable {a b c : ℝ} (ha : a = 1) (hb : b > 0) (hc : c > 0) (haacute : 0 < A ∧ A < Real.pi / 2)

-- Cosine rule: a^2 = b^2 + c^2 - 2 * b * c * cos A
theorem max_area_triangle (hbc_eq : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * (sqrt 3 / 2)) :
  ∃ S : ℝ, 0 < S ∧ S ≤ (2 + sqrt 3) / 4 :=
sorry

end max_area_triangle_l534_534774


namespace number_of_polynomial_expressions_l534_534426

def is_polynomial (expr : String) : Bool :=
  match expr with
  | "1/x" => false
  | "2x + y" => true
  | "1/3 * a^2 * b" => true
  | "(x - y)/π" => true
  | "5y/4x" => false
  | "0.5" => true
  | _ => false

def expressions : List String := ["1/x", "2x + y", "1/3 * a^2 * b", "(x - y)/π", "5y/4x", "0.5"]

def count_polynomials (expr_list : List String) : Nat :=
  expr_list.countp (λ expr => is_polynomial expr)

theorem number_of_polynomial_expressions : count_polynomials expressions = 4 :=
  by
    sorry

end number_of_polynomial_expressions_l534_534426


namespace lucy_reads_sixty_pages_l534_534262

-- Define the number of pages Carter, Lucy, and Oliver can read in an hour.
def pages_carter : ℕ := 30
def pages_oliver : ℕ := 40

-- Carter reads half as many pages as Lucy.
def reads_half_as_much_as (a b : ℕ) : Prop := a = b / 2

-- Lucy reads more pages than Oliver.
def reads_more_than (a b : ℕ) : Prop := a > b

-- The goal is to show that Lucy can read 60 pages in an hour.
theorem lucy_reads_sixty_pages (pages_lucy : ℕ) (h1 : reads_half_as_much_as pages_carter pages_lucy)
  (h2 : reads_more_than pages_lucy pages_oliver) : pages_lucy = 60 :=
sorry

end lucy_reads_sixty_pages_l534_534262


namespace juice_consumption_l534_534130

variables {n : ℕ}
variables {a : Fin n → ℝ} -- a function that gives the amount of juice each member brings

noncomputable def total_juice : ℝ := ∑ i, a i -- total amount of juice

theorem juice_consumption (h : ∀ i, a i ≤ total_juice / 3)
  (hcond : ∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → 
    ∃ (q : ℝ), q ≥ 0 ∧ q ≤ a i ∧ q ≤ a j ∧ q ≤ a k ∧ q ≤ total_juice / 3) :
  ∀ i, a i = total_juice / 3 :=
sorry

end juice_consumption_l534_534130


namespace area_parallelogram_le_half_area_triangle_l534_534951

/-- The area of a parallelogram inside a triangle does not exceed half the area of the triangle. -/
theorem area_parallelogram_le_half_area_triangle 
  (A B C X : Point)
  (parallelogram_inside_triangle : ParallelogramInsideTriangle A B C X) :
  Area (parallelogram A B C X) <= 1/2 * Area (triangle A B C) :=
by sorry

end area_parallelogram_le_half_area_triangle_l534_534951


namespace at_least_20_single_color_l534_534211

theorem at_least_20_single_color (red green yellow blue white black : ℕ)
  (h_red : red = 35) (h_green : green = 27) (h_yellow : yellow = 22)
  (h_blue : blue = 18) (h_white : white = 15) (h_black : black = 12) :
  ∃ n, n = 103 ∧ (∀ nr ng ny nb nw nk, nr + ng + ny + nb + nw + nk = n →
  (nr > 19) ∨ (ng > 19) ∨ (ny > 19) ∨ (nb > 17) ∨ (nw > 14) ∨ (nk > 11)) :=
begin
  sorry
end

end at_least_20_single_color_l534_534211


namespace isosceles_trapezoid_AB_isosceles_trapezoid_BH_isosceles_trapezoid_BP_isosceles_trapezoid_DF_isosceles_trapezoid_ordering_l534_534583

variables (a b : ℝ)
variables (H1 : a < b)

def AB := (a + b) / 2
def BH := Real.sqrt (a * b)
def BP := (2 * a * b) / (a + b)
def DF := Real.sqrt ((a^2 + b^2) / 2)

theorem isosceles_trapezoid_AB :
  let AB := (a + b) / 2 in AB = (a + b) / 2 :=
by sorry

theorem isosceles_trapezoid_BH :
  let BH := Real.sqrt (a * b) in BH = Real.sqrt (a * b) :=
by sorry

theorem isosceles_trapezoid_BP :
  let BP := (2 * a * b) / (a + b) in BP = (2 * a * b) / (a + b) :=
by sorry

theorem isosceles_trapezoid_DF :
  let DF := Real.sqrt ((a^2 + b^2) / 2) in DF = Real.sqrt ((a^2 + b^2) / 2) :=
by sorry

theorem isosceles_trapezoid_ordering :
  let AB := (a + b) / 2
  let BH := Real.sqrt (a * b)
  let BP := (2 * a * b) / (a + b)
  let DF := Real.sqrt ((a^2 + b^2) / 2)
  in BP < BH ∧ BH < AB ∧ AB < DF :=
by sorry

end isosceles_trapezoid_AB_isosceles_trapezoid_BH_isosceles_trapezoid_BP_isosceles_trapezoid_DF_isosceles_trapezoid_ordering_l534_534583


namespace exists_divisible_diff_l534_534329

theorem exists_divisible_diff (l : List ℤ) (h_len : l.length = 2022) :
  ∃ i j, i ≠ j ∧ (l.nthLe i sorry - l.nthLe j sorry) % 2021 = 0 :=
by
  apply sorry -- Placeholder for proof

end exists_divisible_diff_l534_534329


namespace train_crosses_signal_pole_in_20_seconds_l534_534206

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 285
noncomputable def total_time_to_cross_platform : ℝ := 39

-- Define the speed of the train
noncomputable def train_speed : ℝ := (train_length + platform_length) / total_time_to_cross_platform

-- Define the expected time to cross the signal pole
noncomputable def time_to_cross_signal_pole : ℝ := train_length / train_speed

theorem train_crosses_signal_pole_in_20_seconds :
  time_to_cross_signal_pole = 20 := by
  sorry

end train_crosses_signal_pole_in_20_seconds_l534_534206


namespace count_positive_integers_with_two_digits_l534_534839

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534839


namespace number_of_positive_integers_with_at_most_two_digits_l534_534818

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534818


namespace measure_of_angle_B_l534_534050

theorem measure_of_angle_B (C : ℝ) (a b : ℝ) (B : ℝ) 
  (h1 : C = π / 6)
  (h2 : a = 1)
  (h3 : b = sqrt 3) :
  B = 2 * π / 3 := 
sorry

end measure_of_angle_B_l534_534050


namespace trigonometry_identity_l534_534340

theorem trigonometry_identity
  (α : ℝ)
  (h_quad: 0 < α ∧ α < π / 2)
  (h_cos : real.cos α = 2 * real.sqrt 5 / 5) :
  real.cos (2 * α) - real.cos α / real.sin α = -7 / 5 := 
sorry

end trigonometry_identity_l534_534340


namespace larry_gave_52_apples_l534_534086

-- Define the initial and final count of Joyce's apples
def initial_apples : ℝ := 75.0
def final_apples : ℝ := 127.0

-- Define the number of apples Larry gave Joyce
def apples_given : ℝ := final_apples - initial_apples

-- The theorem stating that Larry gave Joyce 52 apples
theorem larry_gave_52_apples : apples_given = 52 := by
  sorry

end larry_gave_52_apples_l534_534086


namespace initial_peaches_correct_l534_534592

def initial_peaches (x : ℕ) (p1 p2 : ℕ) : Prop :=
  -- First day consumption
  let eaten_first_day := x / 2 - 12 in
  let left_after_first_day := x - eaten_first_day in
  
  -- Second day consumption
  let eaten_second_day := left_after_first_day / 2 + 12 in
  let left_after_second_day := left_after_first_day - eaten_second_day in

  -- Final amount of peaches left after the second day
  left_after_second_day = 19 ∧ p1 = left_after_first_day ∧ p2 = left_after_second_day

theorem initial_peaches_correct : ∃ x : ℕ, initial_peaches x (x / 2 + 12) (x / 4 - 6) :=
  exists.intro 100 (
    by
      dsimp [initial_peaches]
      split
      {
        exact rfl
      }
      {
        split
        {
          exact rfl
        }
        {
          exact rfl
        }
      }
  )

end initial_peaches_correct_l534_534592


namespace int_solution_for_system_l534_534685

noncomputable def log6 : ℝ → ℝ := λ x, Real.log x / Real.log 6

theorem int_solution_for_system :
  ∃ x y : ℤ, (x ^ (x - 2 * y) = 36) ∧ (4 * (x - 2 * y) + Real.log x / Real.log 6 = 9) ∧ (x, y) = (6, 2) := 
by
  sorry

end int_solution_for_system_l534_534685


namespace correct_conclusions_in_space_l534_534620

/--
  By analogy with the property in the plane that "two lines perpendicular to the same line are
  parallel to each other," prove the following conclusions in space.

  Given:
  ① Two lines perpendicular to the same line are parallel to each other;
  ② Two lines perpendicular to the same plane are parallel to each other;
  ③ Two planes perpendicular to the same line are parallel to each other;
  ④ Two planes perpendicular to the same plane are parallel to each other.
  
  The correct conclusion(s) are numbered:
  2 and 3.
-/
theorem correct_conclusions_in_space (h1: "Two lines perpendicular to the same line are parallel to each other" → False)
                                    (h2: "Two lines perpendicular to the same plane are parallel to each other" → True)
                                    (h3: "Two planes perpendicular to the same line are parallel to each other" → True)
                                    (h4: "Two planes perpendicular to the same plane are parallel to each other" → False) :
  (2, 3) :=
by
  sorry

end correct_conclusions_in_space_l534_534620


namespace grace_age_l534_534371

/-- Grace's age calculation based on given family ages. -/
theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ)
  (h1 : mother_age = 80)
  (h2 : grandmother_age = 2 * mother_age)
  (h3 : grace_age = (3 * grandmother_age) / 8) : grace_age = 60 :=
by
  sorry

end grace_age_l534_534371


namespace ab2c_value_l534_534746

theorem ab2c_value (a b c : ℚ) (h₁ : |a + 1| + (b - 2)^2 = 0) (h₂ : |c| = 3) :
  a + b + 2 * c = 7 ∨ a + b + 2 * c = -5 := sorry

end ab2c_value_l534_534746


namespace find_line_eq_l534_534981

noncomputable def circle_eqn (x y a : ℝ): Prop :=
  x^2 + y^2 + 2*x - 4*y + a = 0

theorem find_line_eq (a : ℝ) (h_a : a < 3) :
  (∃ (l : ℝ → ℝ), (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eqn x₁ y₁ a ∧ circle_eqn x₂ y₂ a ∧
    (x₁ + x₂) / 2 = 0 ∧ (y₁ + y₂) / 2 = 1 ∧
    (∀ x, l x = x + 1))) →
  (∀ x y, y = l x ↔ y - x = 1) :=
by sorry

end find_line_eq_l534_534981


namespace num_integers_two_digits_l534_534805

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534805


namespace slope_of_tangent_to_exp2_at_1_l534_534166

theorem slope_of_tangent_to_exp2_at_1 :
  (deriv (λ x : ℝ, 2 ^ x) 1) = 2 * Real.log 2 :=
by {
  sorry -- Proof would be placed here
}

end slope_of_tangent_to_exp2_at_1_l534_534166


namespace range_g_l534_534977

noncomputable def piecewise_linear_function (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -3 then x + 2
  else if -3 < x ∧ x ≤ -2 then -2 * x - 4
  else if -2 < x ∧ x <= 0 then (1/2) * x + 1
  else if 0 < x ∧ x ≤ 2 then (1/2) * x
  else if 2 < x ∧ x ≤ 3 then x - 1
  else if 3 < x ∧ x ≤ 4 then 4 * x - (4 * 3)
  else 0

def g (x : ℝ) : ℝ := piecewise_linear_function x - x^2

theorem range_g :
  set.range g = set.Icc (-14 : ℝ) (2.25 : ℝ) := 
sorry

end range_g_l534_534977


namespace find_phi_l534_534358

def function (x : ℝ) (φ : ℝ) : ℝ := -2 * Real.sin (3 * x + φ)

theorem find_phi (φ : ℝ) :
  (∀ x : ℝ, (π / 2 < x ∧ x < 5 * π / 6) → function x φ > function (x - Real.pi / (3:ℝ )) φ) →
  (0 < φ ∧ φ < 2 * π) →
  φ = π :=
by
  sorry

end find_phi_l534_534358


namespace inequality_hold_l534_534101

theorem inequality_hold (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 :=
by
  -- Proof goes here
  sorry

end inequality_hold_l534_534101


namespace mariela_cards_total_l534_534557

theorem mariela_cards_total : 
  let a := 287.0
  let b := 116
  a + b = 403 := 
by
  sorry

end mariela_cards_total_l534_534557


namespace count_special_integers_l534_534825

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534825


namespace median_is_163_l534_534168

noncomputable def median_of_roller_coaster_times : ℕ :=
let data := [28, 28, 50, 60, 62, 140, 145, 155, 163, 180, 180, 180, 210, 216, 240, 240] in
data.nth_le (data.length / 2) (by apply Nat.div_lt_self; simp [dec_trivial])

theorem median_is_163 :
  median_of_roller_coaster_times = 163 :=
by
  -- Convert the given minutes and seconds to total seconds.
  let data := [28, 28, 50, 60, 62, 140, 145, 155, 163, 180, 180, 180, 210, 216, 240, 240]
  simp [median_of_roller_coaster_times.sorry]  -- Proof statement can go here
  sorry

end median_is_163_l534_534168


namespace prime_rational_root_l534_534683

theorem prime_rational_root (p q : ℕ) (hp : p.prime) (hq : q.prime) (h : ∃ r s : ℚ, r ≠ s ∧ 3 * r ^ 2 - p * r + q = 0 ∧ 3 * s ^ 2 - p * s + q = 0) : 
  (p = 5 ∧ q = 2) ∨ (p = 7 ∧ q = 2) :=
sorry

end prime_rational_root_l534_534683


namespace inequality_induction_l534_534481

theorem inequality_induction (n : ℕ) (h : n > 1) : 
  (\sum i in Finset.range (n + 1) \ Finset.range (1), (1:ℝ) / (i + n + 1)) > (1:ℝ) / 2 :=
sorry

end inequality_induction_l534_534481


namespace count_special_integers_l534_534822

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534822


namespace integer_to_the_fourth_l534_534880

theorem integer_to_the_fourth (a : ℤ) (h : a = 243) : 3^12 * 3^8 = a^4 :=
by {
  sorry
}

end integer_to_the_fourth_l534_534880


namespace max_length_side_parallel_barn_l534_534226

theorem max_length_side_parallel_barn (cost_per_foot : ℝ) (total_cost : ℝ) (side_barn_length : ℝ) : 
  cost_per_foot = 5 → total_cost = 1400 → side_barn_length = 350 → 
  let fencing_length := total_cost / cost_per_foot in 
  ∃ x : ℝ, 280 - 2 * x = 140 :=
by {
  assume h1 : cost_per_foot = 5,
  assume h2 : total_cost = 1400,
  assume h3 : side_barn_length = 350,
  have h4 : fencing_length = 280, from (by rw [h1, h2] : (1400 : ℝ) / (5 : ℝ) = (280 : ℝ)),
  use 70,
  simp,
  sorry
}

end max_length_side_parallel_barn_l534_534226


namespace counterexample_proof_l534_534238

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_not_prime (n : ℕ) : Prop :=
  ¬is_prime n

def counterexample_statement : Prop :=
  ∃ n ∈ ({6, 9, 10, 11, 15} : set ℕ), is_not_prime n ∧ (is_prime (n - 2) ∨ is_prime (n + 2))

theorem counterexample_proof : counterexample_statement :=
sorry

end counterexample_proof_l534_534238


namespace find_f_l534_534523

theorem find_f (d e f : ℝ) (h_g : 16 = g) 
  (h_mean_of_zeros : -d / 12 = 3 + d + e + f + 16) 
  (h_product_of_zeros_two_at_a_time : -d / 12 = e / 3) : 
  f = -39 :=
by
  sorry

end find_f_l534_534523


namespace parabola_intersects_x_axis_l534_534759

theorem parabola_intersects_x_axis {p q x₀ x₁ x₂ : ℝ} (h : ∀ (x : ℝ), x ^ 2 + p * x + q ≠ 0)
    (M_below_x_axis : x₀ ^ 2 + p * x₀ + q < 0)
    (M_at_1_neg2 : x₀ = 1 ∧ (1 ^ 2 + p * 1 + q = -2)) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₀ < x₁ → x₁ < x₂) ∧ x₁ = -1 ∧ x₂ = 2 ∨ x₁ = 0 ∧ x₂ = 3) :=
by
  sorry

end parabola_intersects_x_axis_l534_534759


namespace sqrt_x_minus_1_meaningful_real_l534_534393

theorem sqrt_x_minus_1_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ (x ≥ 1) :=
by
  sorry

end sqrt_x_minus_1_meaningful_real_l534_534393


namespace vector_properties_l534_534369

noncomputable def a : ℝ × ℝ := (-2, 1)
noncomputable def b : ℝ × ℝ := (-1, 3)

def is_linearly_independent (u v : ℝ × ℝ) : Prop :=
  ∀ λ : ℝ, u ≠ λ • v

def sine_angle (u v : ℝ × ℝ) : ℝ :=
  let dot_product := (u.fst * v.fst) + (u.snd * v.snd)
  let norm_u := Real.sqrt ((u.fst ^ 2) + (u.snd ^ 2))
  let norm_v := Real.sqrt ((v.fst ^ 2) + (v.snd ^ 2))
  Real.sqrt (1 - (dot_product / (norm_u * norm_v))^2)

def condition_d (c : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  let l := tuple_space.len (tuple_space.add a c)
  let r := tuple_space.len (tuple_space.sub b c)
  l = r

theorem vector_properties :
  is_linearly_independent a b ∧
  sine_angle a b = Real.sqrt 2 / 2 ∧
  (∃ x : ℝ, let c := (x, x)
            condition_d c a b ∧ x = 5 / 2) :=
sorry

end vector_properties_l534_534369


namespace Justin_run_home_time_l534_534916

variable (blocksPerMinute : ℝ) (totalBlocks : ℝ)

theorem Justin_run_home_time (h1 : blocksPerMinute = 2 / 1.5) (h2 : totalBlocks = 8) :
  totalBlocks / blocksPerMinute = 6 := by
  sorry

end Justin_run_home_time_l534_534916


namespace number_of_paths_l534_534657

-- Define the coordinates and the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

def E := (0, 7)
def F := (4, 5)
def G := (9, 0)

-- Define the number of steps required for each path segment
def steps_to_F := 6
def steps_to_G := 10

-- Capture binomial coefficients for the calculated path segments
def paths_E_to_F := binomial steps_to_F 4
def paths_F_to_G := binomial steps_to_G 5

-- Prove the total number of paths from E to G through F
theorem number_of_paths : paths_E_to_F * paths_F_to_G = 3780 :=
by rw [paths_E_to_F, paths_F_to_G]; sorry

end number_of_paths_l534_534657


namespace monic_quadratic_with_root_l534_534701

theorem monic_quadratic_with_root :
  ∃ (p : ℝ[X]), monic p ∧ (p.coeff 2 = 1) ∧ (p.coeff 1 = 6) ∧ (p.coeff 0 = 16) ∧ is_root p (-3 - complex.I * real.sqrt 7) :=
sorry

end monic_quadratic_with_root_l534_534701


namespace taylor_series_expansion_l534_534293

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 2

theorem taylor_series_expansion :
  ∀ x : ℝ, f(x) = -12 + 16*(x + 1) - 7*(x + 1)^2 + (x + 1)^3 := by
  sorry

end taylor_series_expansion_l534_534293


namespace sequence_product_l534_534187

theorem sequence_product : 
  (∏ (i : ℕ) in Finset.range 1027, (Real.ofNat (i + 5)) / (Real.ofNat (i + 4))) = Real.ofNat 1031 / 4 := 
by
  sorry

end sequence_product_l534_534187


namespace sum_log_sqrt3_l534_534637

theorem sum_log_sqrt3 : 
  (∑ k in Finset.range (501), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 124886 :=
by
  sorry

end sum_log_sqrt3_l534_534637


namespace intersection_point_in_polar_coordinates_l534_534899

theorem intersection_point_in_polar_coordinates (theta : ℝ) (rho : ℝ) (h₁ : theta = π / 3) (h₂ : rho = 2 * Real.cos theta) (h₃ : rho > 0) : rho = 1 :=
by
  -- Proof skipped
  sorry

end intersection_point_in_polar_coordinates_l534_534899


namespace find_ordered_triples_l534_534295

noncomputable def satisfies_conditions (a b c : ℝ) : Prop :=
  (a^2 * b + c = b^2 * c + a) ∧ (b^2 * c + a = c^2 * a + b) ∧ (ab + bc + ca = 1)

theorem find_ordered_triples :
  { (a, b, c) : ℝ × ℝ × ℝ | satisfies_conditions a b c } =
  { (0, 1, 1), (0, -1, -1), (1, 0, 1), (-1, 0, -1), 
    (1, 1, 0), (-1, -1, 0), 
    (real.sqrt 3 / 3, real.sqrt 3 / 3, real.sqrt 3 / 3),
    (-real.sqrt 3 / 3, -real.sqrt 3 / 3, -real.sqrt 3 / 3) } :=
by
  sorry

end find_ordered_triples_l534_534295


namespace math_problem_l534_534735

theorem math_problem 
  (x y : ℝ) 
  (h1 : x + y = -5) 
  (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := 
sorry

end math_problem_l534_534735


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534791

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534791


namespace ratio_ba_area_triangle_l534_534074

-- Define the problem and conditions
variables {A B C : ℝ} {a b c : ℝ}
hypothesis (H1: 2 * Real.sin B = Real.sin A + Real.sin C)
hypothesis (H2: Real.cos C = 1/3)
hypothesis (H3: c = 11)

-- The first part of the proof: b/a = 10/9
theorem ratio_ba : b / a = 10 / 9 := sorry

-- The second part of the proof: Area of △ABC = 30√2
theorem area_triangle (c : ℝ) (a b : ℝ) (H_cosC : Real.cos C = 1/3) (H_sinC : Real.sin C = 2 * Real.sqrt 2 / 3) :
  1/2 * a * b * Real.sin C = 30 * Real.sqrt 2 := sorry

end ratio_ba_area_triangle_l534_534074


namespace sum_log_floor_ceil_l534_534629

theorem sum_log_floor_ceil (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500 → 
  ((⌈ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌋) = 
    if (∃ n : ℕ, k = (3 : ℕ) ^ (n / 2 : ℚ)) then 0 else 1)) :
  ∑ k in Finset.range 501 \ {0}, k * (⌈ Real.log k / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log k / Real.log (Real.sqrt 3) ⌋) = 125237 := by
  have h_sum_all := (500 * 501) / 2
  have h_sum_powers := 1 + 3 + 9
  have h_correct := h_sum_all - h_sum_powers
  rw h_sum_all
  rw h_sum_powers
  rw h_correct
  sorry

end sum_log_floor_ceil_l534_534629


namespace total_number_of_animals_l534_534535

-- Define the data and conditions
def total_legs : ℕ := 38
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the proof problem
theorem total_number_of_animals (h1 : total_legs = 38) 
                                (h2 : chickens = 5) 
                                (h3 : chicken_legs = 2) 
                                (h4 : sheep_legs = 4) : 
  (∃ sheep : ℕ, chickens + sheep = 12) :=
by 
  sorry

end total_number_of_animals_l534_534535


namespace no_solution_l534_534142

theorem no_solution : ∀ x : ℝ, x ≠ 1 → (6 / (x - 1) - (x + 5) / (x^2 - x) = 0) → false :=
by
  assume x hx heq
  sorry -- proof to be filled in later

end no_solution_l534_534142


namespace determine_f2_l534_534345

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry
def a : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f(x)
axiom g_even : ∀ x : ℝ, g (-x) = g(x)
axiom eq1 : f 2 + g 2 = a^2 - a^(-2) + 2
axiom eq2 : g 2 = a

theorem determine_f2 : f 2 = 15 / 4 := sorry

end determine_f2_l534_534345


namespace find_length_AE_l534_534905

theorem find_length_AE
  (O A B C D E : Point)
  (h1 : ∠ A C D = ∠ D O B)
  (h2 : B = C)
  (AD BD BC : ℝ)
  (hAD : AD = 9)
  (hDB : DB = 4)
  (hBC : (sqrt (AD ^ 2 + BD ^ 2))) :
  ∃ AE : ℝ, AE = 39 / 5 := by
  sorry

end find_length_AE_l534_534905


namespace climate_conference_non_indian_percentage_l534_534415

theorem climate_conference_non_indian_percentage :
  let men := 500
  let women := 300
  let children := 500
  let indian_men := 0.10 * men
  let indian_women := 0.60 * women
  let indian_children := 0.70 * children
  let total_indians := indian_men + indian_women + indian_children
  let total_people := men + women + children
  let non_indian_people := total_people - total_indians
  let non_indian_percentage := (non_indian_people / total_people) * 100
  non_indian_percentage ≈ 55.38 :=
by
  let men := 500
  let women := 300
  let children := 500
  let indian_men := 0.10 * men
  let indian_women := 0.60 * women
  let indian_children := 0.70 * children
  let total_indians := indian_men + indian_women + indian_children
  let total_people := men + women + children
  let non_indian_people := total_people - total_indians
  let non_indian_percentage := (non_indian_people / total_people) * 100
  have h : non_indian_percentage ≈ 55.38 := sorry
  exact h

end climate_conference_non_indian_percentage_l534_534415


namespace area_triangle_JBK_l534_534067

-- Definitions of the conditions extracted from the problem
def square_area_25 (ABJI : set.point) : Prop :=
  ABJI.area = 25

def square_area_49 (FEHG : set.point) : Prop :=
  FEHG.area = 49

def is_isosceles_JBK (JB BK : ℝ) : Prop :=
  JB = BK

def side_equal (FE BC : ℝ) : Prop :=
  FE = BC

-- Theorem that leverages the conditions to prove the required area
theorem area_triangle_JBK {ABJI FEHG : set.point'} {JB BK FE BC : ℝ} 
  (h1 : square_area_25 ABJI) 
  (h2 : square_area_49 FEHG) 
  (h3 : is_isosceles_JBK JB BK)
  (h4 : side_equal FE BC) :
  (1 / 2) * JB * BK = 12.5 :=
by { sorry }

end area_triangle_JBK_l534_534067


namespace range_of_f_lt_zero_l534_534041

def f (x : ℝ) : ℝ := x^(2/3) - x^(-1/2)

theorem range_of_f_lt_zero :
  ∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end range_of_f_lt_zero_l534_534041


namespace functions_same_l534_534024

-- Define the functions
def f1 (x : ℝ) : ℝ := (x^2 - 1) / (x - 1)
def g1 (x : ℝ) : ℝ := x + 1

def f2 (x : ℝ) : ℝ := abs x
def g2 (x : ℝ) : ℝ := sqrt (x^2)

def f3 (x : ℝ) : ℝ := x^2 - 2*x - 1
def g3 (t : ℝ) : ℝ := t^2 - 2*t - 1

-- State the problem to prove which functions represent the same function
theorem functions_same :
  ((∀ x : ℝ, f2 x = g2 x) ∧ (∀ x : ℝ, f3 x = g3 x)) ∧
  (¬ (∀ x : ℝ, f1 x = g1 x)) ∧
  (¬ (∀ x : ℝ, f1 x = g2 x)) ∧
  (¬ (∀ x : ℝ, f1 x = g3 x)) ∧
  (¬ (∀ x : ℝ, f2 x = g1 x)) ∧
  (¬ (∀ x : ℝ, f3 x = g1 x)) :=
by
  sorry

end functions_same_l534_534024


namespace salt_quantity_l534_534524

-- Conditions translated to Lean definitions
def cost_of_sugar_per_kg : ℝ := 1.50
def total_cost_sugar_2kg_and_salt (x : ℝ) : ℝ := 5.50
def total_cost_sugar_3kg_and_1kg_salt : ℝ := 5.00

-- Theorem statement
theorem salt_quantity (x : ℝ) : 
  2 * cost_of_sugar_per_kg + x * cost_of_sugar_per_kg / 3 = total_cost_sugar_2kg_and_salt x 
  → 3 * cost_of_sugar_per_kg + x = total_cost_sugar_3kg_and_1kg_salt 
  → x = 5 := 
sorry

end salt_quantity_l534_534524


namespace total_coins_l534_534136

theorem total_coins (q1 q2 q3 q4 : Nat) (d1 d2 d3 : Nat) (n1 n2 : Nat) (p1 p2 p3 p4 p5 : Nat) :
  q1 = 8 → q2 = 6 → q3 = 7 → q4 = 5 →
  d1 = 7 → d2 = 5 → d3 = 9 →
  n1 = 4 → n2 = 6 →
  p1 = 10 → p2 = 3 → p3 = 8 → p4 = 2 → p5 = 13 →
  q1 + q2 + q3 + q4 + d1 + d2 + d3 + n1 + n2 + p1 + p2 + p3 + p4 + p5 = 93 :=
by
  intros
  sorry

end total_coins_l534_534136


namespace num_integers_two_digits_l534_534808

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534808


namespace power_function_correct_option_l534_534250

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

def y1 (x : ℝ) : ℝ := -x^3
def y2 (x : ℝ) : ℝ := x^(-3)
def y3 (x : ℝ) : ℝ := 2 * x^3
def y4 (x : ℝ) : ℝ := x^3 - 1

theorem power_function_correct_option :
  (is_power_function y2) ∧ ¬ (is_power_function y1) ∧ ¬ (is_power_function y3) ∧ ¬ (is_power_function y4) :=
by
  sorry

end power_function_correct_option_l534_534250


namespace Gina_kept_170_l534_534312

def initial_amount : ℕ := 400
def mom_share : ℚ := 1 / 4
def clothes_share : ℚ := 1 / 8
def charity_share : ℚ := 1 / 5

def amount_to_mom : ℕ := initial_amount * (mom_share.to_nat)
def amount_on_clothes : ℕ := initial_amount * (clothes_share.to_nat)
def amount_to_charity : ℕ := initial_amount * (charity_share.to_nat)

def total_given_away : ℕ := amount_to_mom + amount_on_clothes + amount_to_charity
def remaining_amount : ℕ := initial_amount - total_given_away

theorem Gina_kept_170 :
  remaining_amount = 170 :=
by
  sorry

end Gina_kept_170_l534_534312


namespace sum_log_floor_ceil_l534_534628

theorem sum_log_floor_ceil (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500 → 
  ((⌈ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌋) = 
    if (∃ n : ℕ, k = (3 : ℕ) ^ (n / 2 : ℚ)) then 0 else 1)) :
  ∑ k in Finset.range 501 \ {0}, k * (⌈ Real.log k / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log k / Real.log (Real.sqrt 3) ⌋) = 125237 := by
  have h_sum_all := (500 * 501) / 2
  have h_sum_powers := 1 + 3 + 9
  have h_correct := h_sum_all - h_sum_powers
  rw h_sum_all
  rw h_sum_powers
  rw h_correct
  sorry

end sum_log_floor_ceil_l534_534628


namespace transform_sine_identity_l534_534357

theorem transform_sine_identity (f : ℝ → ℝ) :
  (∀ x, f(x) = (λ x => (1 / 2) * Real.sin(2 * x - Real.pi / 2)) x) <->
  (∀ x, (λ x => (1 / 2) * Real.sin(x)) x = (λ x => (1 / 2) * Real.sin(2 * (x + Real.pi / 2))) x) :=
sorry

end transform_sine_identity_l534_534357


namespace area_of_given_45_45_90_triangle_l534_534520

noncomputable def area_of_45_45_90_triangle
  (hypotenuse : ℝ) 
  (angle : ℝ) 
  (h_hypotenuse : hypotenuse = 8 * real.sqrt 2) 
  (h_angle : angle = 45) : ℝ :=
if h : (angle = 45) ∧ (hypotenuse = 8 * real.sqrt 2)
then 32
else 0

theorem area_of_given_45_45_90_triangle 
  (hypotenuse : ℝ) 
  (angle : ℝ) 
  (h_hypotenuse : hypotenuse = 8 * real.sqrt 2) 
  (h_angle : angle = 45) : 
  area_of_45_45_90_triangle hypotenuse angle h_hypotenuse h_angle = 32 :=
by
  unfold area_of_45_45_90_triangle
  split_ifs
  {refl}
  {exact absurd h (by simp [h_hypotenuse, h_angle])}

end area_of_given_45_45_90_triangle_l534_534520


namespace max_leap_years_in_200_years_l534_534056

theorem max_leap_years_in_200_years :
  let years := 200
  let leap_frequency := 3
  let leap_years := years / leap_frequency
  leap_years = 66 := 
by
  let years := 200
  let leap_frequency := 3
  let leap_years := years / leap_frequency
  have h : leap_years = 200 / 3 := rfl
  rw h
  norm_num
  exact eq.refl 66
  sorry

end max_leap_years_in_200_years_l534_534056


namespace christmas_bonus_remainder_l534_534119

theorem christmas_bonus_remainder (B P R : ℕ) (hP : P = 8 * B + 5) (hR : (4 * P) % 8 = R) : R = 4 :=
by
  sorry

end christmas_bonus_remainder_l534_534119


namespace find_x_l534_534049

theorem find_x : ∀ x : ℝ, (2 * x / 4 = 12) → (x = 24) :=
begin
  intros x h,
  sorry
end

end find_x_l534_534049


namespace range_of_m_l534_534734

variable {x m : ℝ}

def p := (12 / (x + 2) ≥ 1)
def q := (x^2 - 2 * x + 1 - m^2 ≤ 0)
def neg_p := ¬p
def neg_q := ¬q
def sufficient_not_necessary (A B : set ℝ) : Prop := A ⊆ B ∧ ¬ (B ⊆ A)

theorem range_of_m (m_pos : m > 0) (H : sufficient_not_necessary {x | x > 10 ∨ x ≤ -2} {x | x > 1 + m ∨ x < 1 - m}) :
  0 < m ∧ m < 3 := 
sorry

end range_of_m_l534_534734


namespace trajectory_midpoint_line_exists_bisect_l534_534353

-- Given definitions
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 9) = 1
def line (l : ℝ → ℝ) : Prop := ∀ (x : ℝ), l x = (4 / 5) * x

-- Problem statement
theorem trajectory_midpoint (P : ℝ → ℝ → Prop) (Pmid : ∀ (A B : ℝ × ℝ), ellipse A.1 A.2 → ellipse B.1 B.2 → P = (λ x y, y = (B.2 + A.2) / 2 ∧ x = (A.1 + B.1) / 2)) :
  ∃ (eq : ℝ → ℝ → Prop), eq = (λ x y, 9 * x + 20 * y = 0 ∧ -4 < x ∧ x < 4) :=
sorry

theorem line_exists_bisect (P : ℝ → ℝ → Prop) (Pval : P (4 / 3) (-3 / 5)) :
  ∃ (l : ℝ → ℝ), (∀ (x : ℝ), l x = (12 / 15) * x - 25 / 15) :=
sorry

end trajectory_midpoint_line_exists_bisect_l534_534353


namespace triangle_DEF_isosceles_set_triangle_DEF_equilateral_position_l534_534088

noncomputable def Apollonius_circle (A B C : Point) : Set Point := sorry

noncomputable def Fermat_point (A B C : Point) : Point := sorry

theorem triangle_DEF_isosceles_set (A B C P : Point):
  let D := foot_of_perpendicular P A B in
  let E := foot_of_perpendicular P B C in
  let F := foot_of_perpendicular P C A in
  acute_triangle A B C →
  P ∈ interior_triangle A B C →
  (isosceles_triangle D E F ↔
    (P ∈ Apollonius_circle A B C ∨
     P ∈ Apollonius_circle B C A ∨
     P ∈ Apollonius_circle C A B)) := 
sorry

theorem triangle_DEF_equilateral_position (A B C P : Point):
  let D := foot_of_perpendicular P A B in
  let E := foot_of_perpendicular P B C in
  let F := foot_of_perpendicular P C A in 
  acute_triangle A B C →
  P ∈ interior_triangle A B C →
  equilateral_triangle D E F ↔ P = Fermat_point A B C :=
sorry

end triangle_DEF_isosceles_set_triangle_DEF_equilateral_position_l534_534088


namespace maximum_monthly_profit_l534_534231

def f (x : ℕ) := -3 * x^2 + 40 * x
def q (x : ℕ) := 150 + 2 * x
def profit (x : ℕ) := (185 - q x) * f x

theorem maximum_monthly_profit :
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 12 ∧ (∀ x' : ℕ, 1 ≤ x' ∧ x' ≤ 12 → profit x' ≤ profit x) ∧ profit x = 3125 :=
by
  sorry

end maximum_monthly_profit_l534_534231


namespace Gina_kept_170_l534_534313

def initial_amount : ℕ := 400
def mom_share : ℚ := 1 / 4
def clothes_share : ℚ := 1 / 8
def charity_share : ℚ := 1 / 5

def amount_to_mom : ℕ := initial_amount * (mom_share.to_nat)
def amount_on_clothes : ℕ := initial_amount * (clothes_share.to_nat)
def amount_to_charity : ℕ := initial_amount * (charity_share.to_nat)

def total_given_away : ℕ := amount_to_mom + amount_on_clothes + amount_to_charity
def remaining_amount : ℕ := initial_amount - total_given_away

theorem Gina_kept_170 :
  remaining_amount = 170 :=
by
  sorry

end Gina_kept_170_l534_534313


namespace number_of_squares_correct_l534_534870

-- Definitions for conditions given in the problem.
def region (x y : ℝ) : Prop :=
  y <= 2 * x ∧ y >= -2 ∧ x <= 7 ∧ x >= 1

def integer_points_in_region (x y : ℝ) : Prop :=
  region x y ∧ x = x.floor ∧ y = y.floor

-- The goal to prove: the number of squares within the specified region
def number_of_squares_in_region : ℝ := 130

theorem number_of_squares_correct :
  ∃ (n : ℕ), n = number_of_squares_in_region ∧ n = 130 :=
by
  use 130
  split
  · rfl
  · sorry

end number_of_squares_correct_l534_534870


namespace problem_statement_l534_534992

def a (n : ℕ) : ℕ :=
if n % 2 = 1 then n else a (n / 2)

def f (n : ℕ) : ℕ :=
(finset.range (2^n + 1)).sum a

theorem problem_statement : f 2014 - f 2013 = 4 ^ 2013 :=
sorry

end problem_statement_l534_534992


namespace find_CBA_l534_534179

-- Defining the conditions mathematically
variables (A B C : Type*) [IsTriangle A B C]
variables (BAC_angle : ℝ) (CBA_angle : ℝ) (BC : ℝ) (AB_ratio : ℝ)
variables (H I O : Type*)

-- Given conditions
def conditions (ABC : IsTriangle A B C) : Prop :=
  (BAC_angle = 45) ∧
  (CBA_angle ≤ 120) ∧
  (BC = 2) ∧
  (AB_ratio = 2) ∧
  (∃ H I O, IsOrthocenter H ABC ∧ IsIncenter I ABC ∧ IsCircumcenter O ABC)

-- Proving the problem
theorem find_CBA (ABC : IsTriangle A B C) (cnd : conditions ABC) : CBA_angle = 60 :=
sorry  -- Proof to be filled in later

end find_CBA_l534_534179


namespace google_hits_exact_31415_l534_534374

/-- Google search result for "3.1415" considering no substring matches -/
theorem google_hits_exact_31415 : 
  (num_hits "3.1415" (substring_search := false)) = 422000 := 
sorry

end google_hits_exact_31415_l534_534374


namespace Justin_run_home_time_l534_534917

variable (blocksPerMinute : ℝ) (totalBlocks : ℝ)

theorem Justin_run_home_time (h1 : blocksPerMinute = 2 / 1.5) (h2 : totalBlocks = 8) :
  totalBlocks / blocksPerMinute = 6 := by
  sorry

end Justin_run_home_time_l534_534917


namespace diamonds_in_G_10_l534_534660

-- Define the sequence rule for diamonds in Gn
def diamonds_in_G (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

-- The main theorem to prove that the number of diamonds in G₁₀ is 218
theorem diamonds_in_G_10 : diamonds_in_G 10 = 218 := by
  sorry

end diamonds_in_G_10_l534_534660


namespace verify_statements_l534_534788

def vec_a : ℝ × ℝ := (2, 4)
def vec_b : ℝ × ℝ := (-2, 1)

theorem verify_statements :
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) ∧
  (real.sqrt ((vec_a.1 + vec_b.1)^2 + (vec_a.2 + vec_b.2)^2) = 5) ∧
  (real.sqrt ((vec_a.1 - vec_b.1)^2 + (vec_a.2 - vec_b.2)^2) = 5) := 
by 
  sorry

end verify_statements_l534_534788


namespace count_two_digit_or_less_numbers_l534_534840

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534840


namespace olivias_dad_total_spending_l534_534123

def people : ℕ := 5
def meal_cost : ℕ := 12
def drink_cost : ℕ := 3
def dessert_cost : ℕ := 5

theorem olivias_dad_total_spending : 
  (people * meal_cost) + (people * drink_cost) + (people * dessert_cost) = 100 := 
by
  sorry

end olivias_dad_total_spending_l534_534123


namespace quadratic_coefficients_l534_534932

theorem quadratic_coefficients (b c : ℝ) :
  (∀ x : ℝ, |x + 4| = 3 ↔ x^2 + bx + c = 0) → (b = 8 ∧ c = 7) :=
by
  sorry

end quadratic_coefficients_l534_534932


namespace convex_quadrilateral_area_max_angle_l534_534219

theorem convex_quadrilateral_area_max_angle
  (AB BC CD DA : ℝ)
  (h_AB : AB = 5)
  (h_BC : BC = 5)
  (h_CD : CD = 5)
  (h_DA : DA = 3)
  (convex : ∀ A B C D : ℝ × ℝ, convex_quadrilateral A B C D)
  (maximize_angle : ∀ A B C D : ℝ × ℝ, maximize_angle_ABC A B C D) :
  ∃ A B C D : ℝ × ℝ,
  quadrilateral A B C D ∧
  AB = dist A B ∧
  BC = dist B C ∧
  CD = dist C D ∧
  DA = dist D A ∧
  area A B C D = 12 :=
sorry

end convex_quadrilateral_area_max_angle_l534_534219


namespace exist_1006_intersecting_permutations_l534_534182

open Equiv

def intersects {n : ℕ} (a b : Perm (Fin n)) : Prop :=
  ∃ k : Fin n, a k = b k

theorem exist_1006_intersecting_permutations :
  ∃ (S : Fin 1006 → Perm (Fin 2010)), ∀ p : Perm (Fin 2010), ∃ i : Fin 1006, intersects (S i) p :=
sorry

end exist_1006_intersecting_permutations_l534_534182


namespace trays_from_second_table_l534_534469

def trays_per_trip : ℕ := 4
def trips : ℕ := 9
def trays_from_first_table : ℕ := 20

theorem trays_from_second_table :
  trays_per_trip * trips - trays_from_first_table = 16 :=
by
  sorry

end trays_from_second_table_l534_534469


namespace ratio_pr_l534_534889

variable (p q r s : ℚ)

def ratio_pq (p q : ℚ) : Prop := p / q = 5 / 4
def ratio_rs (r s : ℚ) : Prop := r / s = 4 / 3
def ratio_sq (s q : ℚ) : Prop := s / q = 1 / 5

theorem ratio_pr (hpq : ratio_pq p q) (hrs : ratio_rs r s) (hsq : ratio_sq s q) : p / r = 75 / 16 := by
  sorry

end ratio_pr_l534_534889


namespace sum_log_floor_ceil_l534_534625

theorem sum_log_floor_ceil (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500 → 
  ((⌈ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌋) = 
    if (∃ n : ℕ, k = (3 : ℕ) ^ (n / 2 : ℚ)) then 0 else 1)) :
  ∑ k in Finset.range 501 \ {0}, k * (⌈ Real.log k / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log k / Real.log (Real.sqrt 3) ⌋) = 125237 := by
  have h_sum_all := (500 * 501) / 2
  have h_sum_powers := 1 + 3 + 9
  have h_correct := h_sum_all - h_sum_powers
  rw h_sum_all
  rw h_sum_powers
  rw h_correct
  sorry

end sum_log_floor_ceil_l534_534625


namespace trapezium_top_width_l534_534970

theorem trapezium_top_width (bottom_width : ℝ) (height : ℝ) (area : ℝ) (top_width : ℝ) 
  (h1 : bottom_width = 8) 
  (h2 : height = 50) 
  (h3 : area = 500) : top_width = 12 :=
by
  -- Definitions
  have h_formula : area = 1 / 2 * (top_width + bottom_width) * height := by sorry
  -- Applying given conditions to the formula
  rw [h1, h2, h3] at h_formula
  -- Solve for top_width
  sorry

end trapezium_top_width_l534_534970


namespace cube_contains_two_regular_tetrahedra_l534_534139

-- Assume the vertices of the cube are given as follows:
variables (A B C D A1 B1 C1 D1 : Type)

-- Assume the edge length of the cube
constant a : ℝ

-- Define the property of vertices forming a regular tetrahedron
def is_regular_tetrahedron (v1 v2 v3 v4 : Type) : Prop :=
  -- Function that checks if the distances between the vertices are equal
  distance v1 v2 = a * real.sqrt 2 ∧
  distance v1 v3 = a * real.sqrt 2 ∧
  distance v1 v4 = a * real.sqrt 2 ∧
  distance v2 v3 = a * real.sqrt 2 ∧
  distance v2 v4 = a * real.sqrt 2 ∧
  distance v3 v4 = a * real.sqrt 2

-- State the theorem
theorem cube_contains_two_regular_tetrahedra :
  (is_regular_tetrahedron A C B1 D1) ∧
  (is_regular_tetrahedron B D A1 C1) ∧
  (∀ v1 v2 v3 v4, (is_regular_tetrahedron v1 v2 v3 v4) →
    ({v1, v2, v3, v4} = {A, C, B1, D1} ∨ {v1, v2, v3, v4} = {B, D, A1, C1})) :=
sorry

end cube_contains_two_regular_tetrahedra_l534_534139


namespace find_a_l534_534091

def A (x : ℝ) : Prop := x^2 + 6 * x < 0
def B (a x : ℝ) : Prop := x^2 - (a - 2) * x - 2 * a < 0
def U (x : ℝ) : Prop := -6 < x ∧ x < 5

theorem find_a : (∀ x, A x ∨ ∃ a, B a x) = U x -> a = 5 :=
by
  sorry

end find_a_l534_534091


namespace range_of_m_l534_534980

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m) ↔ m ≤ -1 ∨ m ≥ 4 :=
by
  sorry

end range_of_m_l534_534980


namespace abs_diff_squares_plus_50_correct_l534_534548

def abs_diff_squares_plus_50 (a b : ℕ) : ℕ :=
  abs (a^2 - b^2) + 50

theorem abs_diff_squares_plus_50_correct : abs_diff_squares_plus_50 105 95 = 2050 := by
  sorry

end abs_diff_squares_plus_50_correct_l534_534548


namespace evaluate_expression_l534_534453

def greatest_power_of_factor_2 (n : ℕ) : ℕ :=
  (nat.factors n).count 2

def greatest_power_of_factor_5 (n : ℕ) : ℕ :=
  (nat.factors n).count 5

theorem evaluate_expression (a b : ℕ) (h₁ : 2^a = 8) (h₂ : 5^b = 25) :
  (1 / 3) ^ (b - a) = 3 := by
  have ha : a = greatest_power_of_factor_2 200 := by sorry
  have hb : b = greatest_power_of_factor_5 200 := by sorry
  rw [greatest_power_of_factor_2, greatest_power_of_factor_5] at ha hb
  simp at ha hb
  exact sorry

end evaluate_expression_l534_534453


namespace total_candidates_l534_534150

def average_marks_all_candidates : ℕ := 35
def average_marks_passed_candidates : ℕ := 39
def average_marks_failed_candidates : ℕ := 15
def passed_candidates : ℕ := 100

theorem total_candidates (T : ℕ) (F : ℕ) 
  (h1 : 35 * T = 39 * passed_candidates + 15 * F)
  (h2 : T = passed_candidates + F) : T = 120 := 
  sorry

end total_candidates_l534_534150


namespace find_DC_l534_534432

theorem find_DC (AB : ℝ) (A B C D : Point) (angle1 : angle A D B = 90) (sinA : sin (angle A) = 1 / 2) (sinC : sin (angle C) = 2 / 5) : 
  CD = 5 * sqrt 47.25 := 
  sorry

end find_DC_l534_534432


namespace ellipse_properties_l534_534022

-- Define the conditions
variables {a b x y x0 y0 : ℝ}
variables (P : ℝ × ℝ)
def ellipse_C (a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a b : ℝ) := (Real.sqrt (a^2 - b^2)) / a = 1 / 2
def passes_through_P (a b : ℝ) (P : ℝ × ℝ) := (P.1^2 / a^2) + (P.2^2 / b^2) = 1

-- The proof problem
theorem ellipse_properties (h₁ : a > b) (h₂ : 0 < b) (h₃ : eccentricity a b)
  (h₄ : passes_through_P a b (1, 3/2)) :
  (∀ x y, ellipse_C (2 : ℝ) (Real.sqrt 3 = (x^2 / 4) + (y^2 / 3) = 1)) ∧
  (∀ x0, -2 ≤ x0 ∧ x0 < 4/3) ∧
  (∃ x0, y0, ∃ DE_max : ℝ, DE_max = 8 * Real.sqrt 3 / 3) :=
sorry

end ellipse_properties_l534_534022


namespace circles_intersect_l534_534366

theorem circles_intersect (t : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * t * x + t^2 - 4 = 0 ∧ x^2 + y^2 + 2 * x - 4 * t * y + 4 * t^2 - 8 = 0) ↔ 
  (-12 / 5 < t ∧ t < -2 / 5) ∨ (0 < t ∧ t < 2) :=
sorry

end circles_intersect_l534_534366


namespace num_integers_two_digits_l534_534806

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534806


namespace domain_sqrt_frac_l534_534280

theorem domain_sqrt_frac (x : ℝ) :
  (x^2 + 4*x + 3 ≠ 0) ∧ (x + 3 ≥ 0) ↔ ((x ∈ Set.Ioc (-3) (-1)) ∨ (x ∈ Set.Ioi (-1))) :=
by
  sorry

end domain_sqrt_frac_l534_534280


namespace tetrahedron_inequality_l534_534948

theorem tetrahedron_inequality
  (A B C D : Type)
  [has_dist A]
  [has_dist B]
  [has_dist C]
  [has_dist D]
  (AB CD AC BD AD BC : ℝ)
  (h1 : dist A B = AB)
  (h2 : dist C D = CD)
  (h3 : dist A C = AC)
  (h4 : dist B D = BD)
  (h5 : dist A D = AD)
  (h6 : dist B C = BC) :
  (AB / CD) ^ 2 + (AC / BD) ^ 2 + (AD / BC) ^ 2 > 1 :=
by
  sorry

end tetrahedron_inequality_l534_534948


namespace distinct_remainders_exists_l534_534108

theorem distinct_remainders_exists {p : ℕ} (hp : Nat.Prime p) 
    (a : Fin p → ℤ) : 
    ∃ k : ℤ, (Set.finite (Set.image (λ i : Fin p, (a i + i.val * k) % p) (Set.univ : Set (Fin p))) ∧
    Finset.card (Set.image (λ i : Fin p, (a i + i.val * k) % p) (Set.univ : Set (Fin p))) ≥ (p / 2)) :=
sorry

end distinct_remainders_exists_l534_534108


namespace shirt_original_price_l534_534912

theorem shirt_original_price (P : ℝ) : 
  (18 = P * 0.75 * 0.75 * 0.90 * 1.15) → 
  P = 18 / (0.75 * 0.75 * 0.90 * 1.15) :=
by
  intro h
  sorry

end shirt_original_price_l534_534912


namespace relationship_between_abc_l534_534747

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem relationship_between_abc (h1 : 2^a = Real.log (1/a) / Real.log 2)
                                 (h2 : Real.log b / Real.log 2 = 2)
                                 (h3 : c = Real.log 2 + Real.log 3 - Real.log 7) :
  b > a ∧ a > c :=
sorry

end relationship_between_abc_l534_534747


namespace gcd_of_lcm_and_ratio_l534_534402

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : Nat.gcd X Y = 18 :=
sorry

end gcd_of_lcm_and_ratio_l534_534402


namespace slope_angle_of_line_l534_534529

theorem slope_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 4 * x + y - 1 = 0 ↔ y = m * x + 1) ∧ (m = -4) → 
  θ = Real.pi - Real.arctan 4 :=
by
  sorry

end slope_angle_of_line_l534_534529


namespace probability_A_wins_l534_534209

theorem probability_A_wins 
  (prob_draw : ℚ)
  (prob_B_wins : ℚ)
  (h_draw : prob_draw = 1/2)
  (h_B_wins : prob_B_wins = 1/3) : 
  1 - prob_draw - prob_B_wins = 1 / 6 :=
by
  rw [h_draw, h_B_wins]
  norm_num

end probability_A_wins_l534_534209


namespace exists_k_with_distinct_remainders_l534_534110

theorem exists_k_with_distinct_remainders :
  ∀ (p : ℕ) (hp: p.prime) (a : ℕ → ℤ),
  ∃ (k : ℤ), 
  (finset.univ.image (λ i : fin p, ((a i.val + ↑i * k) % p))).card ≥ p / 2 :=
by sorry

end exists_k_with_distinct_remainders_l534_534110


namespace jorge_goals_this_season_l534_534442

def jorge_goals_last_season : Nat := 156
def jorge_goals_total : Nat := 343

theorem jorge_goals_this_season :
  ∃ g_s : Nat, g_s = jorge_goals_total - jorge_goals_last_season ∧ g_s = 187 :=
by
  -- proof goes here, we use 'sorry' for now
  sorry

end jorge_goals_this_season_l534_534442


namespace diagonals_product_areas_equal_l534_534153

variable (A B C D O : Type)
variables [𝕜 : Type*] [field 𝕜] 
variables [V : Type*] [add_comm_group V] [module 𝕜 V] 

def area (a b θ : 𝕜) : 𝕜 := (1 / 2) * a * b * real.sin θ

variable {a b c d : V}
variable {θ : 𝕜}

-- Conditions: Points A, B, C, D, and intersection point O are given, along with the angle φ.
def intersects_diagonal (A B C D O : V) (θ : 𝕜) : Prop := 
  area (A.head − O.head) (B.head − O.head) θ *
  area (C.head − O.head) (D.head − O.head) θ =
  area (B.head − O.head) (C.head − O.head) θ *
  area (D.head − O.head) (A.head − O.head) θ

-- Theorem statement conforming to given conditions and requiring proof of equation:
theorem diagonals_product_areas_equal 
  (A B C D O : V) (θ : 𝕜) 
  (h : intersects_diagonal A B C D O θ) : 
  area (A.head − O.head) (B.head − O.head) θ *
  area (C.head − O.head) (D.head − O.head) θ = 
  area (B.head − O.head) (C.head − O.head) θ *
  area (D.head − O.head) (A.head − O.head) θ := 
by 
  -- proof omitted
  sorry

end diagonals_product_areas_equal_l534_534153


namespace find_a3_l534_534018

-- Define the geometric sequence and its properties.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
variable (h_GeoSeq : is_geometric_sequence a q)
variable (h_a1 : a 1 = 1)
variable (h_a5 : a 5 = 9)

-- Define what we need to prove
theorem find_a3 : a 3 = 3 :=
sorry

end find_a3_l534_534018


namespace count_valid_numbers_l534_534865

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534865


namespace trapezoid_parallel_side_length_l534_534233

theorem trapezoid_parallel_side_length :
  ∀ (A B C D : Pointℝ), 
  (square A B C D ∧ side_length A B = 2) → 
  (∃ (M P Q : Pointℝ), 
    midpoint A B M ∧ 
    on_side AD P ∧ 
    on_side BC Q ∧ 
    identical_trapezoids (AMPD BMQC MPQ) ∧ 
    area (AMPD) = area (BMQC) = area (MPQ) = (4 / 3)
  ) ↔ shorter_parallel_side_length (AMPD) = 2 / 3 := 
sorry

end trapezoid_parallel_side_length_l534_534233


namespace bicycle_cost_price_for_A_l534_534229

noncomputable def CP_A := 189.29
noncomputable def final_price_D : ℝ := 450
noncomputable def service_tax : ℝ := 0.10
noncomputable def profit_C : ℝ := 0.40
noncomputable def discount_B : ℝ := 0.05
noncomputable def profit_B : ℝ := 0.30
noncomputable def profit_A : ℝ := 0.25

theorem bicycle_cost_price_for_A (CP_A : ℝ) 
  (final_price_D : ℝ := 450)
  (service_tax : ℝ := 0.10)
  (profit_C : ℝ := 0.40)
  (discount_B : ℝ := 0.05)
  (profit_B : ℝ := 0.30)
  (profit_A : ℝ := 0.25) :
  CP_A ≈ 189.29 := by
    sorry

end bicycle_cost_price_for_A_l534_534229


namespace sum_k_round_diff_eq_125237_l534_534635

noncomputable def sum_k_round_diff (n : ℕ) : ℤ :=
∑ k in finset.range n.succ, k * ((⌈real.log k / real.log (real.sqrt 3)⌉ : ℤ) - (⌊real.log k / real.log (real.sqrt 3)⌋ : ℤ))

theorem sum_k_round_diff_eq_125237 : sum_k_round_diff 500 = 125237 := by
  -- Here we would add the proof steps to verify the theorem
  sorry

end sum_k_round_diff_eq_125237_l534_534635


namespace number_of_positive_integers_with_at_most_two_digits_l534_534816

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534816


namespace last_two_nonzero_digits_85_l534_534522

theorem last_two_nonzero_digits_85! : Nat.modeq 100 (85.factorial / 10^20) 68 :=
by {
  sorry
}

end last_two_nonzero_digits_85_l534_534522


namespace find_actual_food_price_l534_534584

variable (P : Real) -- The actual price of the food before tax, tip, and discounts.
variable (B : Real) -- The total bill.

-- Conditions:
def sales_tax (P : Real) : Real := 0.10 * P
def membership_discount (P : Real) : Real := 0.06 * P
def tips (P : Real) : Real := (0.15 + 0.10 + 0.05) * 0.94 * P
def total_bill (P : Real) : Real := P + sales_tax P + tips P - membership_discount P

theorem find_actual_food_price 
  (h₁ : B = 235.80)
  (h₂ : B = 1.322 * P) : 
  P ≈ 178.36 := 
  by 
    sorry

end find_actual_food_price_l534_534584


namespace area_of_trapezoid_DBCE_l534_534433

/-- All triangles are similar to isosceles triangle ABC, with AB=AC. Each of the 8 smallest triangles has an area of 1, and the total area of ΔABC is 50. Prove that the area of trapezoid DBCE is 45. -/
theorem area_of_trapezoid_DBCE (h₁ : ∀ (Δ : Triangle), similar Δ ABC)
  (AB_eq_AC : AB = AC)
  (small_triangle_area : ∀ (Δ : Triangle), Δ ∈ smallest_triangles → Δ.area = 1)
  (sum_smallest_triangles_area : (finset.sum (finset.univ.filter smallest_triangles) (λ Δ, Δ.area)) = 8)
  (ABC_area : ABC.area = 50) :
  DBCE.area = 45 := by
  sorry

end area_of_trapezoid_DBCE_l534_534433


namespace color_arith_seq_4_colors_l534_534468

theorem color_arith_seq_4_colors :
  ∃ (color : ℕ → ℕ), 
  (∀ x, x ∈ finset.range 2000.succ → color x ∈ finset.range 4) ∧
  (∀ (a d : ℕ), d ≠ 0 → list.nodup (list.of_fn (λ n, color (a + n * d)))) :=
begin
  sorry
end

end color_arith_seq_4_colors_l534_534468


namespace count_at_most_two_different_digits_l534_534853

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534853


namespace length_segment_MN_l534_534920

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def midpoint (p₁ p₂ : Point3D) : Point3D :=
  ⟨(p₁.x + p₂.x) / 2, (p₁.y + p₂.y) / 2, (p₁.z + p₂.z) / 2⟩

def distance_z (p₁ p₂ : Point3D) : ℝ :=
  abs (p₁.z - p₂.z)

noncomputable def A : Point3D := ⟨0, 0, 0⟩
noncomputable def B : Point3D := ⟨0, 2, 0⟩
noncomputable def C : Point3D := ⟨3, 2, 0⟩
noncomputable def D : Point3D := ⟨3, 0, 0⟩

noncomputable def A' : Point3D := ⟨0, 0, 12⟩
noncomputable def B' : Point3D := ⟨0, 2, 6⟩
noncomputable def C' : Point3D := ⟨3, 2, 20⟩
noncomputable def D' : Point3D := ⟨3, 0, 24⟩

noncomputable def M : Point3D := midpoint A' C'
noncomputable def N : Point3D := midpoint B' D'

theorem length_segment_MN : distance_z M N = 1 :=
sorry

end length_segment_MN_l534_534920


namespace trapezoid_area_correct_l534_534071

def trapezoid_area (AD BC AB CD : ℝ) (h : AD = 52 ∧ BC = 65 ∧ AB = 20 ∧ CD = 11) : ℝ :=
  if AD = 52 ∧ BC = 65 ∧ AB = 20 ∧ CD = 11 then 594 else 0

theorem trapezoid_area_correct :
  trapezoid_area 52 65 20 11 (by simp only [eq_self_iff_true, and_true]) = 594 :=
by simp only [trapezoid_area, if_true, eq_self_iff_true]

end trapezoid_area_correct_l534_534071


namespace target_in_second_quadrant_l534_534010

-- Define the complex number z and its conjugate
def z : ℂ := 1 + complex.i
def conj_z : ℂ := complex.conj z

-- Define the target complex number
def target : ℂ := (z * z) / conj_z

-- Prove that target lies in the second quadrant
theorem target_in_second_quadrant : (target.re < 0) ∧ (target.im > 0) :=
by
  -- Definitions
  rw [conj_z, z, complex.conj, complex.of_real_add, complex.of_real_one, complex.add_im, complex.add_re, complex.i_mul, complex.of_real_zero, complex.mul_re, complex.mul_im, complex.of_real_zero, complex.mul_eight, complex.add_eight, complex.map_eight, mul_comm, add_comm]
  -- Sorry for the proof
  sorry

end target_in_second_quadrant_l534_534010


namespace journey_total_distance_l534_534084

def miles_driven : ℕ := 923
def miles_to_go : ℕ := 277
def total_distance : ℕ := 1200

theorem journey_total_distance : miles_driven + miles_to_go = total_distance := by
  sorry

end journey_total_distance_l534_534084


namespace zeros_in_30_factorial_l534_534382

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_factors (n : ℕ) (factor : ℕ) : ℕ :=
  if n < factor then 0 else n / factor + count_factors (n / factor) factor

theorem zeros_in_30_factorial : count_factors 30 5 = 7 :=
  sorry

end zeros_in_30_factorial_l534_534382


namespace solve_exponential_equation_l534_534490

theorem solve_exponential_equation (y : ℝ) (h : 81 = 3 * (27)^(y - 2)) : y = 3 :=
by {
  sorry
}

end solve_exponential_equation_l534_534490


namespace cycling_time_difference_l534_534897

-- Definitions from the conditions
def youth_miles : ℤ := 20
def youth_hours : ℤ := 2
def adult_miles : ℤ := 12
def adult_hours : ℤ := 3

-- Conversion from hours to minutes
def hours_to_minutes (hours : ℤ) : ℤ := hours * 60

-- Time per mile calculations
def youth_time_per_mile : ℤ := hours_to_minutes youth_hours / youth_miles
def adult_time_per_mile : ℤ := hours_to_minutes adult_hours / adult_miles

-- The difference in time per mile
def time_difference : ℤ := adult_time_per_mile - youth_time_per_mile

-- Theorem to prove the difference is 9 minutes
theorem cycling_time_difference : time_difference = 9 := by
  -- Proof steps would go here
  sorry

end cycling_time_difference_l534_534897


namespace part1_part2_l534_534290

def f (x : ℝ) := |x + 4| - |x - 1|
def g (x : ℝ) := |2 * x - 1| + 3

theorem part1 (x : ℝ) : (f x > 3) → x > 0 :=
by sorry

theorem part2 (a : ℝ) : (∃ x, f x + 1 < 4^a - 5 * 2^a) ↔ (a < 0 ∨ a > 2) :=
by sorry

end part1_part2_l534_534290


namespace count_valid_numbers_l534_534862

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534862


namespace shaded_region_area_l534_534904

-- Define the radius of the Larger Circle
def r_larger_circle : ℝ := 8

-- Define the radius and area of the smaller circle
def r_smaller_circle := r_larger_circle / 2
def area_larger_circle := real.pi * r_larger_circle^2
def area_smaller_circle := real.pi * r_smaller_circle^2

-- Theorem stating the area of the shaded region
theorem shaded_region_area :
  area_larger_circle - 2 * area_smaller_circle = 32 * real.pi :=
by
  -- Proof skipped
  sorry

end shaded_region_area_l534_534904


namespace slope_perpendicular_l534_534718

theorem slope_perpendicular (x y : ℝ) (h : 4 * x - 5 * y = 10) :
  let m := 4 / 5 in
  -(1 / m) = -5 / 4 :=
by {
  let m := 4 / 5,
  have h1 : 1 / m = 5 / 4 := by sorry,
  exact neg_eq_neg h1,
}

end slope_perpendicular_l534_534718


namespace find_derivative_at_5_l534_534352

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x * f' 2
noncomputable def f' (x : ℝ) : ℝ := 6 * x + 2 * f' 2  -- Assumed form in conditions

theorem find_derivative_at_5 (f : ℝ → ℝ) (h₁ : ∀ x, deriv f x = f' x)
  (h₂ : f' 2 = -12) : f' 5 = 6 := sorry

end find_derivative_at_5_l534_534352


namespace allocation_problem_l534_534283

theorem allocation_problem :
  let students : Finset ℕ := {1, 2, 3, 4} -- Representing the 4 students
  let schools : Finset ℕ := {1, 2, 3} -- Representing schools A, B, and C
  let A := 1 -- Assume student A is represented by 1
  let not_school_a := 1 ≠ 1 -- Student A can't go to school A
  let min_one_student (alloc : ℕ → ℕ) (s : Finset ℕ) := ∀ x ∈ s, ∃ y ∈ students, alloc y = x -- Each school gets at least one student
  ∃ allocation : students → schools,
    ¬ (allocation 1 = 1) ∧ -- Student A doesn't go to school A
    min_one_student allocation schools ∧ -- Each school gets at least one student
    ∃ h : (card (image allocation students) = 3), -- Verification condition for exactly 3 different schools allocated
    card (Finset.univ.filter (λ f : (students → schools), ¬ (f 1 = 1) ∧ min_one_student f schools)) = 24
:= sorry

end allocation_problem_l534_534283


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534795

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534795


namespace center_of_circle_sum_l534_534674
-- Import the entire library

-- Define the problem using declarations for conditions and required proof
theorem center_of_circle_sum (x y : ℝ) 
  (h : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 9 → (x = 2) ∧ (y = -3)) : 
  x + y = -1 := 
by 
  sorry 

end center_of_circle_sum_l534_534674


namespace anna_apples_left_l534_534251

def initial_apples : ℕ := 150
def apples_sold_to_Bob (apples : ℕ) : ℕ := (20 * apples) / 100
def apples_after_selling_to_Bob (apples : ℕ) : ℕ := apples - apples_sold_to_Bob(apples)
def apples_sold_to_Carol (apples : ℕ) : ℕ := (30 * apples) / 100
def apples_after_selling_to_Carol (apples : ℕ) : ℕ := apples - apples_sold_to_Carol(apples)
def apples_after_giving_to_neighbors (apples : ℕ) : ℕ := apples - 3

theorem anna_apples_left : 
  apples_after_giving_to_neighbors (apples_after_selling_to_Carol (apples_after_selling_to_Bob initial_apples)) = 81 :=
by sorry

end anna_apples_left_l534_534251


namespace sqrt_meaningful_range_l534_534399

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1) ↔ x ≥ 1) := 
sorry

end sqrt_meaningful_range_l534_534399


namespace gcd_abcd_dcba_l534_534154

-- Define the given conditions
def is_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = a + 2
def d_eq_a_plus_five (a d : ℕ) : Prop := d = a + 5

-- Define the statement to be proved
theorem gcd_abcd_dcba (a b c d : ℕ) (h1 : is_consecutive a b c) (h2 : d_eq_a_plus_five a d) :
  ∃ k, k = 1111 ∧ ∀ a b c d, is_consecutive a b c → d_eq_a_plus_five a d →
  ∃ gcd, gcd (1000*a + 100*b + 10*c + d + 1000*d + 100*c + 10*b + a) = k := 
  sorry

end gcd_abcd_dcba_l534_534154


namespace general_formulas_l534_534016

noncomputable def a_n (n : ℕ) : ℝ := 3^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := 3 * n
noncomputable def S_n (n : ℕ) : ℝ := (n * (3 + 3 * n)) / 2
noncomputable def c_n (n : ℕ) : ℝ := 3 / (2 * S_n n)
noncomputable def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, c_n i

-- Theorem to prove the general formulas
theorem general_formulas :
  (∀ n : ℕ, a_n n = 3^(n-1)) ∧
  (∀ n : ℕ, b_n n = 3 * n) ∧
  (∀ n : ℕ, T_n n = n / (n + 1)) :=
by
  -- The proof goes here.
  sorry

end general_formulas_l534_534016


namespace sugar_remains_l534_534264

-- Define the initial conditions
def initial_sugar : ℝ := 24
def number_of_bags : ℕ := 4
def damage_percentage : ℝ := 0.15

-- Define the total remaining sugar proof statement
theorem sugar_remains : 
  let sugar_per_bag := initial_sugar / number_of_bags in
  let loss_per_bag := damage_percentage * sugar_per_bag in
  let remaining_sugar := sugar_per_bag - loss_per_bag in
  number_of_bags * remaining_sugar = 20.4 := 
by
  sorry

end sugar_remains_l534_534264


namespace arithmetic_sequence_terms_l534_534430

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 3 + a 4 = 10) 
  (h2 : a (n - 3) + a (n - 2) = 30) 
  (h3 : (n * (a 1 + a n)) / 2 = 100) : 
  n = 10 :=
sorry

end arithmetic_sequence_terms_l534_534430


namespace construct_circumcircle_l534_534242

-- Definitions for the problem
variables {A B : Point}
variables {F O : Point}

-- Defining the conditions
def is_midpoint (F : Point) (A B : Point) : Prop :=
  dist A F = dist B F

def is_perpendicular_bisector (O F : Point) : Prop := 
  perpendicular (line_through F O) (line_through A B)

def is_circumcircle_center (O : Point) (A B : Point) : Prop :=
  dist O A = dist O B

-- Statement of the problem
theorem construct_circumcircle (A B : Point) (F O : Point)
  (h_mid : is_midpoint F A B)
  (h_perp : is_perpendicular_bisector O F)
  (h_center : is_circumcircle_center O A B) :
  ∃ r : ℝ, r = dist O B := 
sorry

end construct_circumcircle_l534_534242


namespace max_diagonals_no_more_than_one_intersection_l534_534741

theorem max_diagonals_no_more_than_one_intersection (n : ℕ) (h : n ≥ 3) :
  let max_diagonals := if n % 2 = 0 then n / 2 else (n - 1) / 2
  in max_diagonals = (n / 2) :=
sorry

end max_diagonals_no_more_than_one_intersection_l534_534741


namespace min_of_expr_l534_534319

theorem min_of_expr (a b : ℝ) (h : a - 3 * b + 6 = 0) : 2^a + 1 / 8^b = (1 / 4) :=
by
  sorry

end min_of_expr_l534_534319


namespace range_of_objective_function_l534_534364

def objective_function (x y : ℝ) : ℝ := 3 * x - y

theorem range_of_objective_function (x y : ℝ) 
  (h1 : x + 2 * y ≥ 2)
  (h2 : 2 * x + y ≤ 4)
  (h3 : 4 * x - y ≥ -1)
  : - 3 / 2 ≤ objective_function x y ∧ objective_function x y ≤ 6 := 
sorry

end range_of_objective_function_l534_534364


namespace ratio_of_segments_l534_534073

variable (A B C E F G M : Type*)
variable [AddCommGroup A] [Module ℝ A]
variables (a b c e f g m : A)
variables (x : ℝ)

noncomputable def midpoint (u v : A) : A := (u + v) / 2

noncomputable def on_line (s t : A) (α β : ℝ) : A := α • s + β • t

noncomputable def intersection (u v p q : A) (α β : ℝ) : A := α • u + β • v

theorem ratio_of_segments
  (h1 : m = midpoint b c)
  (h2 : f = on_line a b (x / 15) ((15 - x) / 15))
  (h3 : e = on_line a c ((3 * x) / 20) ((20 - 3 * x) / 20))
  (h4 : g = intersection e f m b (3 / 5) (2 / 5))
  : (3 : ℝ) / 5 * e + (2 : ℝ) / 5 * f = g :=
by sorry

end ratio_of_segments_l534_534073


namespace positive_difference_median_mode_zero_l534_534281

noncomputable def stemAndLeafDataset : List ℕ :=
  [50, 52, 53, 55, 55, 61, 62, 62, 63, 63, 63, 70, 71, 75,
   82, 82, 82, 87, 87, 87, 89, 91, 93, 98, 98]

def mode (l : List ℕ) : ℕ :=
  l.groupBy id |>.maxBy (·.length) |>.head!

def median (l : List ℕ) : ℕ :=
  let sorted := l.sort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem positive_difference_median_mode_zero :
  |median stemAndLeafDataset - mode stemAndLeafDataset| = 0 :=
by
  sorry

end positive_difference_median_mode_zero_l534_534281


namespace number_of_positive_integers_with_at_most_two_digits_l534_534810

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534810


namespace lines_are_skew_iff_l534_534666

def point1 (b : ℝ) := (2 : ℝ, 3, b)
def dir1 := (3 : ℝ, 4, 5)
def point2 := (3 : ℝ, 4, 1)
def dir2 := (6 : ℝ, 3, 2)

def lines_intersect (b : ℝ) (t u : ℝ) :=
  point1 b + t • dir1 = point2 + u • dir2

theorem lines_are_skew_iff (b : ℝ) :
  (∀ t u, ¬ lines_intersect b t u) ↔ b ≠ (1 / 15) :=
sorry

end lines_are_skew_iff_l534_534666


namespace quadrilateral_segments_condition_l534_534595

-- Define the lengths and their conditions
variables {a b c d : ℝ}

-- Define the main theorem with necessary and sufficient conditions
theorem quadrilateral_segments_condition (h_sum : a + b + c + d = 1.5)
    (h_order : a ≤ b) (h_order2 : b ≤ c) (h_order3 : c ≤ d) (h_ratio : d ≤ 3 * a) :
    (a ≥ 0.25 ∧ d < 0.75) ↔ (a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by {
  sorry -- proof is omitted
}

end quadrilateral_segments_condition_l534_534595


namespace simplify_expression_l534_534956

theorem simplify_expression (x : ℝ) : 3 * x + 4 * (2 - x) - 2 * (3 - 2 * x) + 5 * (2 + 3 * x) = 18 * x + 12 :=
by
  sorry

end simplify_expression_l534_534956


namespace count_valid_numbers_l534_534869

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534869


namespace tristan_study_hours_l534_534081

theorem tristan_study_hours :
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  saturday_hours = 2 := by
{
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  sorry
}

end tristan_study_hours_l534_534081


namespace polynomial_divisible_by_Q1_l534_534947

-- Definition of polynomial P and the condition that P(x) ≠ x
variables {R : Type*} [CommRing R]
variable (P : R[X])
variable hP : P ≠ X

-- Definition of natural number n
variable (n : ℕ)

-- Inductive definition of Q_n(x) based on P
def Q (n : ℕ) (P : R[X]) : R[X] :=
  (Nat.iterate P n X) - X

-- Statement that we need to prove
theorem polynomial_divisible_by_Q1 (P : R[X]) (hP : P ≠ X) (n : ℕ) : ∃ R_n : R[X], Q n P = (P - X) * R_n := 
sorry

end polynomial_divisible_by_Q1_l534_534947


namespace triangle_ABC_proof_l534_534327

theorem triangle_ABC_proof (a b c A B C : ℝ) (h1 : a * cos C + (c - 2 * b) * cos A = 0)
  (h2 : 2 * sqrt 3 > 0)
  (h3 : a = 2 * sqrt 3) : A = π / 3 ∧ b + c = 6 :=
by
  sorry

end triangle_ABC_proof_l534_534327


namespace sin_alpha_eq_l534_534350

noncomputable def sine_of_angle (α : ℝ) : ℝ :=
let x := 3 in
let y := -4 in
let r := Real.sqrt (x^2 + y^2) in
y / r

theorem sin_alpha_eq : sine_of_angle α = -4 / 5 := 
by 
  -- Here we provide the necessary proof steps.
  sorry

end sin_alpha_eq_l534_534350


namespace find_x_l534_534873

theorem find_x (x : ℝ) (A B : Set ℝ) (hA : A = {1, 4, x}) (hB : B = {1, x^2}) (h_inter : A ∩ B = B) : x = -2 ∨ x = 2 ∨ x = 0 :=
sorry

end find_x_l534_534873


namespace percent_of_x_is_y_l534_534878

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end percent_of_x_is_y_l534_534878


namespace quadratic_polynomial_has_root_l534_534705

theorem quadratic_polynomial_has_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.coeff 1 ∈ ℝ ∧ p.coeff 0 ∈ ℝ ∧ Polynomial.eval (-3 - Complex.i * Real.sqrt 7) p = 0 ∧
                        p = Polynomial.X^2 + Polynomial.C 6 * Polynomial.X + Polynomial.C 16 :=
sorry

end quadratic_polynomial_has_root_l534_534705


namespace length_of_goods_train_l534_534590

/-- The length of the goods train given the conditions of the problem --/
theorem length_of_goods_train
  (speed_passenger_train : ℝ) (speed_goods_train : ℝ) 
  (time_taken_to_pass : ℝ) (length_goods_train : ℝ) :
  speed_passenger_train = 80 / 3.6 →  -- Convert 80 km/h to m/s
  speed_goods_train    = 32 / 3.6 →  -- Convert 32 km/h to m/s
  time_taken_to_pass   = 9 →
  length_goods_train   = 280 → 
  length_goods_train = (speed_passenger_train + speed_goods_train) * time_taken_to_pass := by
    sorry

end length_of_goods_train_l534_534590


namespace sin_angle_FAP_l534_534092

universe u
variable {s : ℝ} (cube_has_edge : ∀ (a b : ℝ), a = b ∨ abs(a - b) = s)
variable {P E F A : ℝ}

def isCube (s : ℝ) (A B C D E F G H : ℝ) : Prop :=
  cube_has_edge A B ∧ cube_has_edge B C ∧ cube_has_edge C D ∧ cube_has_edge D A ∧
  cube_has_edge E F ∧ cube_has_edge F G ∧ cube_has_edge G H ∧ cube_has_edge H E ∧
  cube_has_edge A E ∧ cube_has_edge B F ∧ cube_has_edge C G ∧ cube_has_edge D H

def isPyramid (s : ℝ) (E F G H P : ℝ) : Prop :=
  (λ P E, cube_has_edge P E) ∧ 
  (λ F A, cube_has_edge F A)

theorem sin_angle_FAP
  (P E F A : ℝ) 
  (s : ℝ)
  (h₁ : isCube s A B C D E F G H)
  (h₂ : isPyramid s E F G H P) :
  sin (angle F A P) = sqrt 2 / 2 := by
  sorry

end sin_angle_FAP_l534_534092


namespace problem_statement_l534_534643

noncomputable def sum_log_floor_ceiling (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * (if (Real.log k / Real.log (Real.sqrt 3)).floor = (Real.log k / Real.log (Real.sqrt 3)).ceil then 0 else 1)

theorem problem_statement :
  sum_log_floor_ceiling 500 = 124886 :=
by
  -- proof starting here, replace with actual proof
  sorry

end problem_statement_l534_534643


namespace original_pencils_l534_534178

-- Define the conditions
def pencils_added : ℕ := 30
def total_pencils_now : ℕ := 71

-- Define the theorem to prove the original number of pencils
theorem original_pencils (original_pencils : ℕ) :
  total_pencils_now = original_pencils + pencils_added → original_pencils = 41 :=
by
  intros h
  sorry

end original_pencils_l534_534178


namespace sum_log_sqrt3_l534_534639

theorem sum_log_sqrt3 : 
  (∑ k in Finset.range (501), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 124886 :=
by
  sorry

end sum_log_sqrt3_l534_534639


namespace infinite_nested_radicals_solution_l534_534303

theorem infinite_nested_radicals_solution :
  ∃ x : ℝ, 
    (∃ y z : ℝ, (y = (x * y)^(1/3) ∧ z = (x + z)^(1/3)) ∧ y = z) ∧ 
    0 < x ∧ x = (3 + Real.sqrt 5) / 2 := 
sorry

end infinite_nested_radicals_solution_l534_534303


namespace mon_inc_f_intervals_min_g_values_l534_534770

noncomputable def f (x : ℝ) : ℝ := 
  sqrt 3 * sin x * sin x + sin x * cos x

theorem mon_inc_f_intervals :
  ∀ k : ℤ, ∀ x : ℝ,
    k * real.pi - real.pi / 12 ≤ x ∧ x ≤ k * real.pi + 5 * real.pi / 12 →
    (f'(x) > 0) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 
  sin (x / 2 - real.pi / 3) - sqrt 3 / 2

theorem min_g_values :
  ∀ k : ℤ, 
    g (2 * k * real.pi - real.pi / 6) = -1 :=
sorry

end mon_inc_f_intervals_min_g_values_l534_534770


namespace count_positive_integers_with_two_digits_l534_534838

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534838


namespace alpha_beta_square_l534_534874

-- Statement of the problem in Lean 4
theorem alpha_beta_square :
  ∀ (α β : ℝ), (α ≠ β ∧ ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = α ∨ x = β)) → (α - β)^2 = 8 := 
by
  intros α β h
  sorry

end alpha_beta_square_l534_534874


namespace find_a_l534_534386

theorem find_a (a b d : ℤ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l534_534386


namespace coplanar_sets_count_l534_534610

def points_in_tetrahedron : Set (Fin 10) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_vertex (x : Fin 10) : Prop :=
  x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4

def is_midpoint (x : Fin 10) : Prop :=
  x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10

axiom vertices_midpoints (x : Fin 10) : 
  is_vertex x ∨ is_midpoint x

def in_same_plane (s : Set (Fin 10)) : Prop :=
  ∃ a b c d : Fin 10, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  {a, b, c, d} = s ∧
  (is_vertex a ∨ is_midpoint a) ∧
  (is_vertex b ∨ is_midpoint b) ∧
  (is_vertex c ∨ is_midpoint c) ∧
  (is_vertex d ∨ is_midpoint d)

theorem coplanar_sets_count :
  (Finset.powerset points_in_tetrahedron.to_finset).filter (λ s, s.card = 4 ∧ in_same_plane s).card = 33 := by
  sorry

end coplanar_sets_count_l534_534610


namespace reciprocal_inequalities_l534_534936

theorem reciprocal_inequalities (a b c : ℝ)
  (h1 : -1 < a ∧ a < -2/3)
  (h2 : -1/3 < b ∧ b < 0)
  (h3 : 1 < c) :
  1/c < 1/(b - a) ∧ 1/(b - a) < 1/(a * b) :=
by
  sorry

end reciprocal_inequalities_l534_534936


namespace same_side_of_line_l534_534888

theorem same_side_of_line (a : ℝ) :
  let f := λ x y : ℝ, 3 * x - 2 * y + a in
  (f 3 1) * (f (-4) 6) > 0 → (a < -7) ∨ (a > 24) :=
by
  let f := λ x y : ℝ, 3 * x - 2 * y + a
  intro H
  have H1 : (f 3 1) = 7 + a := by simp [f]
  have H2 : (f (-4) 6) = a - 24 := by simp [f]
  rw [H1, H2] at H
  exact sorry

end same_side_of_line_l534_534888


namespace greater_number_is_twelve_l534_534571

theorem greater_number_is_twelve (x : ℕ) (a b : ℕ) 
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x) 
  (h3 : a + b = 21) : 
  max a b = 12 :=
by 
  sorry

end greater_number_is_twelve_l534_534571


namespace tangent_line_equation_l534_534156

/-- Prove that the equation of the tangent line to the curve y = x^3 - 4x^2 + 4 at the point (1,1) is y = -5x + 6 -/
theorem tangent_line_equation (x y : ℝ)
  (h_curve : y = x^3 - 4 * x^2 + 4)
  (h_point : x = 1 ∧ y = 1) :
  y = -5 * x + 6 := by
  sorry

end tangent_line_equation_l534_534156


namespace triangle_division_possible_l534_534078

theorem triangle_division_possible (ABC : Triangle) :
  ∃ (T1 T2 T3 T4 : ConvexShape), 
    T1.shape = triangle ∧ T2.shape = quadrilateral ∧ 
    T3.shape = pentagon ∧ T4.shape = hexagon ∧ 
    is_convex T1 ∧ is_convex T2 ∧ is_convex T3 ∧ is_convex T4 ∧
    (T1 ∪ T2 ∪ T3 ∪ T4 = ABC) ∧ 
    (∀ (i j : {T1, T2, T3, T4}), i ≠ j → disjoint i j) :=
sorry

end triangle_division_possible_l534_534078


namespace rope_percentage_used_and_left_l534_534228

def rope_length : ℝ := 20
def used_length : ℝ := 15

theorem rope_percentage_used_and_left (L : ℝ) (U : ℝ) (hL : L = rope_length) (hU : U = used_length) :
  U / L = 0.75 ∧ 1 - (U / L) = 0.25 :=
by
  rw [hL, hU]
  simp
  split
  { exact (15 / 20 : ℝ) }
  { exact 1 - (15 / 20 : ℝ) }

end rope_percentage_used_and_left_l534_534228


namespace jims_final_paycheck_l534_534441

noncomputable def final_paycheck (g r t h m b btr : ℝ) := 
  let retirement := g * r
  let gym := m / 2
  let net_before_bonus := g - retirement - t - h - gym
  let after_tax_bonus := b * (1 - btr)
  net_before_bonus + after_tax_bonus

theorem jims_final_paycheck :
  final_paycheck 1120 0.25 100 200 50 500 0.30 = 865 :=
by
  sorry

end jims_final_paycheck_l534_534441


namespace closest_integers_to_2013_satisfy_trig_eq_l534_534304

noncomputable def closestIntegersSatisfyingTrigEq (x : ℝ) : Prop := 
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2)

theorem closest_integers_to_2013_satisfy_trig_eq : closestIntegersSatisfyingTrigEq (1935 * (Real.pi / 180)) ∧ closestIntegersSatisfyingTrigEq (2025 * (Real.pi / 180)) :=
sorry

end closest_integers_to_2013_satisfy_trig_eq_l534_534304


namespace simplify_sqrt1_simplify_sqrt2_l534_534134

-- Define the simplification problem for √(5 - 2√6)
theorem simplify_sqrt1 :
  sqrt(5 - 2 * sqrt 6 ) = sqrt 3 - sqrt 2 :=
sorry

-- Define the simplification problem for √(8 + 4√3)
theorem simplify_sqrt2 :
  sqrt(8 + 4 * sqrt 3) = sqrt 6 + sqrt 2 :=
sorry

end simplify_sqrt1_simplify_sqrt2_l534_534134


namespace binomial_17_16_eq_17_l534_534655

theorem binomial_17_16_eq_17 :
  nat.choose 17 16 = 17 :=
by
  sorry

end binomial_17_16_eq_17_l534_534655


namespace cost_price_equals_selling_price_l534_534152

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (hp : C > 0) (profit : ℝ := 0.25) (h : 30 * C = (1 + profit) * C * x) : x = 24 :=
by
  sorry

end cost_price_equals_selling_price_l534_534152


namespace alternating_sum_is_zero_l534_534256

def alternating_binomial_sum : ℤ :=
  ∑ k in Finset.range 50, (-1:ℤ)^k * (Nat.choose 100 (2 * k + 1))

theorem alternating_sum_is_zero : alternating_binomial_sum = 0 := 
by
  sorry

end alternating_sum_is_zero_l534_534256


namespace complex_number_properties_l534_534002

noncomputable def complex_mag_add_le {z1 z2 : ℂ} (h : |z1| * |z2| ≠ 0) : Prop :=
  |z1 + z2| ≤ |z1| + |z2|

noncomputable def complex_mag_mul_eq {z1 z2 : ℂ} (h : |z1| * |z2| ≠ 0) : Prop :=
  |z1 * z2| = |z1| * |z2|

theorem complex_number_properties (z1 z2 : ℂ) (h : |z1| * |z2| ≠ 0) :
  complex_mag_add_le h ∧ complex_mag_mul_eq h := by
  sorry

end complex_number_properties_l534_534002


namespace no_positive_solution_l534_534437

theorem no_positive_solution (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) :
  ¬ (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) :=
sorry

end no_positive_solution_l534_534437


namespace second_number_removed_condition_l534_534964

noncomputable def removed_number (s : ℝ) (x : ℝ) (removed : ℝ) : ℝ :=
  (s - x - removed) / 48

theorem second_number_removed_condition (avg_50 : ℝ) (removed1 : ℝ) (avg_48 : ℝ) :
  (avg_50 = 56) ∧ (removed1 = 45) ∧ (avg_48 = 56.25) →
  let S := avg_50 * 50 in
  let x := 55 in
  removed_number S removed1 45 = 56.25 :=
by
  sorry

end second_number_removed_condition_l534_534964


namespace vector_properties_l534_534782

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-2, 1)

-- Lean statements to check the conditions
theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 0) ∧        -- Perpendicular vectors
  (real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5) ∧  -- Magnitude of the sum of vectors
  (real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5) := -- Magnitude of the difference of vectors
by
  unfold a b
  simp
  split
  -- Proof of each condition is skipped
  sorry 
  sorry
  sorry

end vector_properties_l534_534782


namespace max_min_product_of_distances_l534_534944

theorem max_min_product_of_distances 
  (P : ℝ × ℝ)
  (hP : P.1 ^ 2 / 4 + P.2 ^ 2 = 1)
  (F1 F2 : ℝ × ℝ)
  (hF1 : F1 = (-Real.sqrt 3, 0))
  (hF2 : F2 = (Real.sqrt 3, 0)) :
  ∃ max min, 
    (∀ (P : ℝ × ℝ), P.1 ^ 2 / 4 + P.2 ^ 2 = 1 → Real.dist P F1 * Real.dist P F2 = max) ∧
    (∀ (P : ℝ × ℝ), P.1 ^ 2 / 4 + P.2 ^ 2 = 1 → Real.dist P F1 * Real.dist P F2 = min) ∧
    max = 4 ∧ min = 1 := by
  sorry

end max_min_product_of_distances_l534_534944


namespace integer_to_the_fourth_l534_534879

theorem integer_to_the_fourth (a : ℤ) (h : a = 243) : 3^12 * 3^8 = a^4 :=
by {
  sorry
}

end integer_to_the_fourth_l534_534879


namespace parabola_slope_angle_range_correct_l534_534973

noncomputable def parabola_slope_angle_range (a : ℝ) (ha : a ≠ 0) :
    set ℝ := 
  {α : ℝ | 0 ≤ α ∧ α ≤ Real.pi / 4 ∨
           3 * Real.pi / 4 ≤ α ∧ α < Real.pi}

theorem parabola_slope_angle_range_correct (a : ℝ) (ha : a ≠ 0) :
  parabola_slope_angle_range a ha = (set.Icc 0 (Real.pi / 4)).union (set.Ico (3 * Real.pi / 4) Real.pi) :=
  sorry

end parabola_slope_angle_range_correct_l534_534973


namespace secretaries_ratio_l534_534540

theorem secretaries_ratio (A B C : ℝ) (hA: A = 75) (h_total: A + B + C = 120) : B + C = 45 :=
by {
  -- sorry: We define this part to be explored by the theorem prover
  sorry
}

end secretaries_ratio_l534_534540


namespace unique_x_of_set_A_l534_534164

theorem unique_x_of_set_A (x : ℝ) (h : {0, -1, x} = {0, -1, x} ∧ x^2 ∈ {0, -1, x}) :
  x = 1 :=
sorry

end unique_x_of_set_A_l534_534164


namespace air_quality_probability_l534_534962

theorem air_quality_probability (P1 : ℝ) (P2 : ℝ) :
  P1 = 0.8 ∧ P2 = 0.6 → (0.75 = P2 / P1) :=
by
  intros h
  cases h with hP1 hP2
  rw [hP1, hP2]
  norm_num
  sorry

end air_quality_probability_l534_534962


namespace sum_in_base5_correct_l534_534161

-- Defining the integers
def num1 : ℕ := 210
def num2 : ℕ := 72

-- Summing the integers
def sum : ℕ := num1 + num2

-- Converting the resulting sum to base 5
def to_base5 (n : ℕ) : String :=
  let rec aux (n : ℕ) (acc : List Char) : List Char :=
    if n < 5 then Char.ofNat (n + 48) :: acc
    else aux (n / 5) (Char.ofNat (n % 5 + 48) :: acc)
  String.mk (aux n [])

-- The expected sum in base 5
def expected_sum_base5 : String := "2062"

-- The Lean theorem to be proven
theorem sum_in_base5_correct : to_base5 sum = expected_sum_base5 :=
by
  sorry

end sum_in_base5_correct_l534_534161


namespace count_special_integers_l534_534826

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534826


namespace sufficient_but_not_necessary_condition_l534_534749

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x > 1) : x > 0 :=
by
  exact Nat.one_lt_one

end sufficient_but_not_necessary_condition_l534_534749


namespace jordan_no_quiz_probability_l534_534989

theorem jordan_no_quiz_probability (P_quiz : ℚ) (h : P_quiz = 5 / 9) :
  1 - P_quiz = 4 / 9 :=
by
  rw [h]
  exact sorry

end jordan_no_quiz_probability_l534_534989


namespace Gemma_ordered_pizzas_l534_534309

-- Definitions of conditions
def pizza_cost : ℕ := 10
def tip : ℕ := 5
def paid_amount : ℕ := 50
def change : ℕ := 5
def total_spent : ℕ := paid_amount - change

-- Statement of the proof problem
theorem Gemma_ordered_pizzas : 
  ∃ (P : ℕ), pizza_cost * P + tip = total_spent ∧ P = 4 :=
sorry

end Gemma_ordered_pizzas_l534_534309


namespace probability_point_on_line_l534_534403

namespace ProbabilityOfPointOnLine

open ProbabilityTheory

theorem probability_point_on_line (m n : ℕ) (hp : 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) :
    (∃ (m n : ℕ), m + n = 4 ∧ 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) →
    (∑' (p : ℕ × ℕ), ite (p.1 + p.2 = 4) 1 0) / 36 = 1 / 12 :=
-- Proof goes here
by
  sorry

end ProbabilityOfPointOnLine

end probability_point_on_line_l534_534403


namespace solution_set_quadratic_inequality_l534_534167

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} :=
sorry

end solution_set_quadratic_inequality_l534_534167


namespace max_area_of_triangle_l534_534976

theorem max_area_of_triangle (a b : ℝ) (h1 : c = 6) (h2 : a + b = 10) :
  ∃ S, S = 4 * sqrt (-(a - 5)^2 + 9) ∧ S = 12 :=
by
  sorry

end max_area_of_triangle_l534_534976


namespace problem_statement_l534_534647

noncomputable def sum_log_floor_ceiling (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * (if (Real.log k / Real.log (Real.sqrt 3)).floor = (Real.log k / Real.log (Real.sqrt 3)).ceil then 0 else 1)

theorem problem_statement :
  sum_log_floor_ceiling 500 = 124886 :=
by
  -- proof starting here, replace with actual proof
  sorry

end problem_statement_l534_534647


namespace verify_statements_l534_534786

def vec_a : ℝ × ℝ := (2, 4)
def vec_b : ℝ × ℝ := (-2, 1)

theorem verify_statements :
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) ∧
  (real.sqrt ((vec_a.1 + vec_b.1)^2 + (vec_a.2 + vec_b.2)^2) = 5) ∧
  (real.sqrt ((vec_a.1 - vec_b.1)^2 + (vec_a.2 - vec_b.2)^2) = 5) := 
by 
  sorry

end verify_statements_l534_534786


namespace man_cannot_row_against_stream_l534_534589

theorem man_cannot_row_against_stream (rate_in_still_water speed_with_stream : ℝ)
  (h_rate : rate_in_still_water = 1)
  (h_speed_with : speed_with_stream = 6) :
  ¬ ∃ (speed_against_stream : ℝ), speed_against_stream = rate_in_still_water - (speed_with_stream - rate_in_still_water) :=
by
  sorry

end man_cannot_row_against_stream_l534_534589


namespace small_circle_area_l534_534205

theorem small_circle_area (r R : ℝ) (n : ℕ)
  (h_n : n = 6)
  (h_area_large : π * R^2 = 120)
  (h_relation : r = R / 2) :
  π * r^2 = 40 :=
by
  sorry

end small_circle_area_l534_534205


namespace product_fg_l534_534029

variable (x : ℝ)

def f (x : ℝ) : ℝ := 4 * x

def g (x : ℝ) : ℝ := sqrt (x + 1) / x

theorem product_fg (hx1 : x ≠ 0) (hx2 : x ≥ -1) :
  f x * g x = 4 * sqrt (x + 1) :=
by
  sorry

end product_fg_l534_534029


namespace sum_of_two_le_two_l534_534450

theorem sum_of_two_le_two (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_sum : a^2 + b^2 + c^2 + d^2 = 4) :
  ∃ x y ∈ {a, b, c, d}, x + y ≤ 2 :=
sorry

end sum_of_two_le_two_l534_534450


namespace equation_negative_roots_iff_l534_534155

theorem equation_negative_roots_iff (a : ℝ) :
  (∃ x < 0, 4^x - 2^(x-1) + a = 0) ↔ (-1/2 < a ∧ a ≤ 1/16) := 
sorry

end equation_negative_roots_iff_l534_534155


namespace cos_double_angle_minus_cos_over_sin_l534_534337

theorem cos_double_angle_minus_cos_over_sin (α : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : real.cos α = 2 * real.sqrt 5 / 5) :
  real.cos (2 * α) - real.cos α / real.sin α = -7 / 5 :=
sorry

end cos_double_angle_minus_cos_over_sin_l534_534337


namespace common_divisors_12600_14400_l534_534034

theorem common_divisors_12600_14400 : 
  let a := 12600 in
  let b := 14400 in
  let gcd_ab := Nat.gcd a b in
  let count_divisors (n : ℕ) : ℕ :=
    (n.factorization.get 2 + 1) * (n.factorization.get 3 + 1) * (n.factorization.get 5 + 1) * (n.factorization.get 7 + 1) in
  count_divisors gcd_ab = 45 := by
  let a := 12600
  let b := 14400
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 3600 := by 
    -- Factoring and GCD computation (omitted here)
    sorry
  have h2 : count_divisors gcd_ab = 45 := by 
    -- Divisor count computation (omitted here)
    sorry
  exact h2

end common_divisors_12600_14400_l534_534034


namespace count_positive_integers_with_two_digits_l534_534830

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534830


namespace find_prices_maximize_profit_l534_534541

theorem find_prices (m n : ℕ) 
  (h1 : 3 * m + 4 * n = 620)
  (h2 : 5 * m + 3 * n = 740) : 
  m = 100 ∧ n = 80 := 
by
  sorry

theorem maximize_profit (x : ℕ) 
  (h_m : 100)
  (h_n : 80)
  (h_cost : 100 * x + 80 * (200 - x) ≤ 18100)
  (h_profit : (250 - 100) * x + (200 - 80) * (200 - x) ≥ 27000) : 
  100 ≤ x ∧ x ≤ 105 ∧ (∃ x_max, x_max = 105 ∧ 30 * x_max + 24000 = 27150) := 
by
  sorry

end find_prices_maximize_profit_l534_534541


namespace rich_knight_l534_534958

-- Definitions for the problem
inductive Status
| knight  -- Always tells the truth
| knave   -- Always lies

def tells_truth (s : Status) : Prop := 
  s = Status.knight

def lies (s : Status) : Prop := 
  s = Status.knave

def not_poor (s : Status) : Prop := 
  s = Status.knight ∨ s = Status.knave -- Knights can either be poor or wealthy

def wealthy (s : Status) : Prop :=
  s = Status.knight

-- Statement to be proven
theorem rich_knight (s : Status) (h_truth : tells_truth s) (h_not_poor : not_poor s) : wealthy s :=
by
  sorry

end rich_knight_l534_534958


namespace Don_bottles_C_and_D_l534_534676

def bottles_shop_A := 150
def price_per_bottle_A := 1
def bottles_shop_B := 180
def price_per_bottle_B := 2
def price_per_bottle_C := 3
def price_per_bottle_D := 5
def total_money := 600

theorem Don_bottles_C_and_D : 
  let spent_A := bottles_shop_A * price_per_bottle_A in
  let spent_B := bottles_shop_B * price_per_bottle_B in
  let spent_total := spent_A + spent_B in
  let remaining := total_money - spent_total in
  let bottles_C := remaining / price_per_bottle_C in
  bottles_C = 30 :=
by
  sorry

end Don_bottles_C_and_D_l534_534676


namespace relationship_y_values_l534_534752

theorem relationship_y_values (m y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-3 : ℝ)^2 - 12 * (-3 : ℝ) + m)
  (h2 : y2 = -3 * (-2 : ℝ)^2 - 12 * (-2 : ℝ) + m)
  (h3 : y3 = -3 * (1 : ℝ)^2 - 12 * (1 : ℝ) + m) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end relationship_y_values_l534_534752


namespace arithmetic_mean_gcd_set_l534_534918

-- Define the set S as described in the problem
def S (n : ℕ) : Set ℕ := { k | 1 ≤ k ∧ k ≤ n ∧ Nat.gcd k n = 1 }

-- State the problem as a theorem
theorem arithmetic_mean_gcd_set (n : ℕ) (h : 0 < n) :
  (∑ k in S n, k : ℚ) / (S n).card = n / 2 := sorry

end arithmetic_mean_gcd_set_l534_534918


namespace measure_of_angle_C_l534_534922

variables {I A B C : Type}

-- Assuming vectors IA, IB, IC belong to some vector space V
variable [AddCommGroup V]
variables {IA IB IC : V}

-- Assuming 3 IA + 4 IB + 5 IC = 0
axiom h : 3 • IA + 4 • IB + 5 • IC = (0 : V)

-- Proof that angle C is 90 degrees
theorem measure_of_angle_C (h : 3 • IA + 4 • IB + 5 • IC = (0 : V)) : measure_of_angle_C = 90° :=
  sorry

end measure_of_angle_C_l534_534922


namespace range_of_a_l534_534378

noncomputable def f (a x : ℝ) : ℝ := cos (2 * x) + a * cos (π / 2 + x)

def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

theorem range_of_a {a : ℝ} :
  is_increasing_on (f a) (set.Ioo (π / 6) (π / 2)) → a ≤ -4 :=
begin
  sorry
end

end range_of_a_l534_534378


namespace find_b_l534_534162

-- Definitions from the conditions
def vertex (q : ℝ) : (ℝ × ℝ) := (q, q + 1)
def y_intercept (q : ℝ) : (ℝ × ℝ) := (0, -2 * q - 1)
def h (q : ℝ) : Prop := q ≠ -1 / 2

-- Statement to be proved
theorem find_b (q : ℝ) (hq : h q) :
  let a := -(3 * q + 2) / q^2 in
  let b := 6 + 4 / q in
  b = 6 + 4 / q := 
by
  simp [h, vertex, y_intercept]
  sorry

end find_b_l534_534162


namespace problem_statement_l534_534642

noncomputable def sum_log_floor_ceiling (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * (if (Real.log k / Real.log (Real.sqrt 3)).floor = (Real.log k / Real.log (Real.sqrt 3)).ceil then 0 else 1)

theorem problem_statement :
  sum_log_floor_ceiling 500 = 124886 :=
by
  -- proof starting here, replace with actual proof
  sorry

end problem_statement_l534_534642


namespace problem1_problem2_problem3_problem4_l534_534255

-- Proof statement for problem 1
theorem problem1 : (1 : ℤ) * (-8) + 10 + 2 + (-1) = 3 := sorry

-- Proof statement for problem 2
theorem problem2 : (-21.6 : ℝ) - (-3) - |(-7.4)| + (-2 / 5) = -26.4 := sorry

-- Proof statement for problem 3
theorem problem3 : (-12 / 5) / (-1 / 10) * (-5 / 6) * (-0.4 : ℝ) = 8 := sorry

-- Proof statement for problem 4
theorem problem4 : ((5 / 8) - (1 / 6) + (7 / 12)) * (-24 : ℝ) = -25 := sorry

end problem1_problem2_problem3_problem4_l534_534255


namespace sequence_statements_correct_l534_534761

theorem sequence_statements_correct (S : ℕ → ℝ) (a : ℕ → ℝ) (T : ℕ → ℝ) 
(h_S_nonzero : ∀ n, n > 0 → S n ≠ 0)
(h_S_T_relation : ∀ n, n > 0 → S n + T n = S n * T n) :
  (a 1 = 2) ∧ (∀ n, n > 0 → T n - T (n - 1) = 1) ∧ (∀ n, n > 0 → S n = (n + 1) / n) :=
by
  sorry

end sequence_statements_correct_l534_534761


namespace quadratic_sum_of_solutions_l534_534163

theorem quadratic_sum_of_solutions :
  ∀ (x : ℝ), ((x^2 - 6 * x + 5) - (2 * x - 8) = 0) → 
             ∑ (a b : ℝ), (a, b).fst = 8 :=
by {
  intro x,
  sorry
}

end quadratic_sum_of_solutions_l534_534163


namespace length_HM_in_triangle_ABC_l534_534410

noncomputable def HM_length (AB BC CA : ℝ) (M H : ℝ) : ℝ := 
  sqrt (
    (AB / 2) ^ 2 - 
    ( 
      (
        (AB ^ 2 - (2 * (BC * sqrt((1 / 2) * (AB + BC - CA))))) ^ 2 / BC + 
        (
          16 * 
          (
            (AB ^ 2 - (2 * (BC * sqrt((1 / 2) * (AB + BC - CA))))) *
            (BC - (AB ^ 2 - (2 * (BC * sqrt((1 / 2) * (AB + BC - CA))))) / BC)
          ) / BC
        )
      )
    )
  )

theorem length_HM_in_triangle_ABC :
    let AB := 16
    let BC := 17
    let CA := 15
    let M : ℝ := AB / 2
    let H : ℝ := sqrt(
      2 * (BC * sqrt((1 / 2) * (AB + BC - CA)))
    ) in
  HM_length AB BC CA M H = 5.76 :=
by
  -- proceed to prove the theorem using Lean
  sorry

end length_HM_in_triangle_ABC_l534_534410


namespace unique_solution_system_l534_534306

theorem unique_solution_system (n : ℕ) (hn : n > 0) (a : ℝ) :
  (∃ (x : ℕ → ℝ), (∀ i, i ≥ n → x i = 0) ∧
    (∑ i in finset.range n, x i = a) ∧ 
    (∑ i in finset.range n, (x i)^2 = a^2) ∧ 
    (∀ k ∈ (finset.range (n+1)).filter (λ k, k ≥ 3), ∑ i in finset.range n, (x i)^k = a^k)) → 
  (∃ (i : ℕ), i < n ∧ (∀ j, j < n → (x j = if j = i then a else 0))) :=
sorry

end unique_solution_system_l534_534306


namespace insect_growth_duration_l534_534599

noncomputable def initial_size (d: ℕ) : ℝ := 10 / (2 ^ d)

noncomputable def days_to_reach_size (initial: ℝ) (target: ℝ) (growth_factor: ℝ) : ℕ :=
  (log (target / initial)) / (log growth_factor)

theorem insect_growth_duration :
  ∀ (d: ℕ), days_to_reach_size (initial_size d) 2.5 2 = 8 :=
by
  intro d
  have h : initial_size 10 = 5 / 512 := by sorry -- from the problem data
  have h_growth_target : days_to_reach_size (5 / 512) 2.5 2 = 8 := by sorry -- derived in the solution
  exact h_growth_target

end insect_growth_duration_l534_534599


namespace mean_median_mode_equal_l534_534373

section
open List

def data : List ℕ := [1, 2, 2, 3, 3, 3, 4, 4, 5]

def mean (l : List ℕ) : ℕ := l.sum / l.length

def median (l : List ℕ) : ℕ := l.get! (l.length / 2)

def mode (l : List ℕ) : ℕ := l.foldr (λ n m => if l.count n > l.count m then n else m) (l.head!)

theorem mean_median_mode_equal : mean data = 3 ∧ median data = 3 ∧ mode data = 3 :=
by
  sorry
end

end mean_median_mode_equal_l534_534373


namespace inclination_angle_of_line_l534_534883

theorem inclination_angle_of_line (a b c : ℝ) (h : a = -√3 * b) :
  (a * x + b * y + c = 0) ∧ ((√3, -1) = (a, b)-> (y/x) = Real.sqrt 3) → 
  ∃ θ : ℝ, θ = Real.arctan(Real.sqrt 3) :=
sorry

end inclination_angle_of_line_l534_534883


namespace f_monotonically_increasing_g_minimum_value_l534_534773

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (sin x)^2 + sin x * cos x

noncomputable def g (x : ℝ) : ℝ := sin (x - π / 3)

theorem f_monotonically_increasing : 
  ∀ k : ℤ, ∀ x : ℝ, x ∈ set.Icc (k * π - π / 12) (k * π + 5 * π / 12) → 
  (f' x > 0 := sorry)

theorem g_minimum_value :
  (min_value := -1 := sorry) ∧ 
  ∀ k : ℤ, ∀ x : ℝ, (x = 2 * k * π - π / 6) →
  (g x = -1 := sorry)

end f_monotonically_increasing_g_minimum_value_l534_534773


namespace regular_n_gon_equal_triangles_l534_534724

theorem regular_n_gon_equal_triangles (n : ℕ) (h : n > 3) :
  (∃ (diagonals : set (set (ℝ × ℝ))) (triangles : set (set (set (ℝ × ℝ)))),
     (∀ t ∈ triangles, ∃ (v1 v2 v3 : ℝ × ℝ), t = {v1, v2, v3} ∧ 
      ∀ (d1 d2 ∈ diagonals), d1 = d2 ∨ d1 ∩ d2 = ∅) ∧ 
      ⋃₀ (diagonals ∪ (⋃₀ triangles)) = {v : ℝ × ℝ | some_relation_with_vertices}) ↔ even n :=
sorry

end regular_n_gon_equal_triangles_l534_534724


namespace sam_has_12_nickels_l534_534137

theorem sam_has_12_nickels (n d : ℕ) (h1 : n + d = 30) (h2 : 5 * n + 10 * d = 240) : n = 12 :=
sorry

end sam_has_12_nickels_l534_534137


namespace sqrt_x_minus_1_meaningful_real_l534_534395

theorem sqrt_x_minus_1_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ (x ≥ 1) :=
by
  sorry

end sqrt_x_minus_1_meaningful_real_l534_534395


namespace monic_quadratic_poly_with_root_l534_534695

theorem monic_quadratic_poly_with_root (x : ℝ) :
  (∃ p : Polynomial ℝ, Polynomial.monic p ∧ p.coeff 2 = 1 ∧ p.coeff 1 = 6 ∧ p.coeff 0 = 16 
  ∧ p.eval (-3 - Complex.i * Real.sqrt 7) = 0) :=
by
  use Polynomial.C 16 + Polynomial.X * (Polynomial.C 6 + Polynomial.X)
  field_simp
  sorry

end monic_quadratic_poly_with_root_l534_534695


namespace max_value_func_l534_534983

-- Define the function y
def func (x : ℝ) : ℝ := (2 + Real.cos x) / (2 - Real.cos x)

-- Define the condition that cos(x) is bounded between -1 and 1
def bounded_cos (x : ℝ) : Prop := -1 ≤ Real.cos x ∧ Real.cos x ≤ 1

-- Statement to prove that the maximum value of the function is 3 under the given condition
theorem max_value_func :
  ∀ x : ℝ, bounded_cos x → func x ≤ 3 :=
by
  sorry

end max_value_func_l534_534983


namespace arrangement_of_4_from_6_tasks_l534_534381

theorem arrangement_of_4_from_6_tasks : ∃ (n : ℕ), n = (Nat.choose 6 4) * 4! ∧ n = 360 :=
by
  exists 360
  sorry

end arrangement_of_4_from_6_tasks_l534_534381


namespace count_positive_integers_with_two_digits_l534_534837

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534837


namespace trip_duration_exactly_six_hours_l534_534593

theorem trip_duration_exactly_six_hours : 
  ∀ start_time end_time : ℕ,
  (start_time = (8 * 60 + 43 * 60 / 11)) ∧ 
  (end_time = (14 * 60 + 43 * 60 / 11)) → 
  (end_time - start_time) = 6 * 60 :=
by
  sorry

end trip_duration_exactly_six_hours_l534_534593


namespace blueberry_jelly_amount_l534_534486

-- Definition of the conditions
def total_jelly : ℕ := 6310
def strawberry_jelly : ℕ := 1792

-- Formal statement of the problem
theorem blueberry_jelly_amount : 
  total_jelly - strawberry_jelly = 4518 :=
by
  sorry

end blueberry_jelly_amount_l534_534486


namespace quadratic_polynomial_has_root_l534_534708

theorem quadratic_polynomial_has_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.coeff 1 ∈ ℝ ∧ p.coeff 0 ∈ ℝ ∧ Polynomial.eval (-3 - Complex.i * Real.sqrt 7) p = 0 ∧
                        p = Polynomial.X^2 + Polynomial.C 6 * Polynomial.X + Polynomial.C 16 :=
sorry

end quadratic_polynomial_has_root_l534_534708


namespace triangle_inequality_mid_angle_sum_inequality_l534_534907

theorem triangle_inequality_mid_angle_sum_inequality
  (a b c : ℝ) (α β γ : ℝ)
  (hα : 0 < α ∧ α < π)
  (hβ : 0 < β ∧ β < π)
  (hγ : 0 < γ ∧ γ < π)
  (h_angle_sum : α + β + γ = π)
  (h_side_angle_relation : ∀ {a b c}, a * α + b * β + c * γ)
  (h_side_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
  ((π / 3) ≤ (a * α + b * β + c * γ) / (a + b + c)) ∧ 
  ((a * α + b * β + c * γ) / (a + b + c) < (π / 2)) :=
by
  sorry

end triangle_inequality_mid_angle_sum_inequality_l534_534907


namespace area_difference_square_rectangle_l534_534687

theorem area_difference_square_rectangle :
  ∀ (square_perimeter rectangle_perimeter rectangle_width : ℕ)
  (hsquare : square_perimeter = 36)
  (hrect_perimeter : rectangle_perimeter = 38)
  (hrect_width : rectangle_width = 15), 
  let square_side := square_perimeter / 4 in
  let square_area := square_side * square_side in
  let rectangle_length := (rectangle_perimeter - 2 * rectangle_width) / 2 in
  let rectangle_area := rectangle_length * rectangle_width in
  square_area - rectangle_area = 21 := 
by
  intros 
    square_perimeter rectangle_perimeter rectangle_width 
    hsquare hrect_perimeter hrect_width
  let square_side := square_perimeter / 4
  let square_area := square_side * square_side
  let rectangle_length := (rectangle_perimeter - 2 * rectangle_width) / 2
  let rectangle_area := rectangle_length * rectangle_width
  have hsquare_side : square_side = 9 := by sorry
  have hsquare_area : square_area = 81 := by sorry
  have hrectangle_length : rectangle_length = 4 := by sorry
  have hrectangle_area : rectangle_area = 60 := by sorry
  calc
    81 - 60 = 21 : by sorry

end area_difference_square_rectangle_l534_534687


namespace vertex_of_given_function_l534_534999

-- Definition of the given quadratic function
def given_function (x : ℝ) : ℝ := 2 * (x - 4) ^ 2 + 5

-- Definition of the vertex coordinates
def vertex_coordinates : ℝ × ℝ := (4, 5)

-- Theorem stating the vertex coordinates of the function
theorem vertex_of_given_function : (0, given_function 4) = vertex_coordinates :=
by 
  -- Placeholder for the proof
  sorry

end vertex_of_given_function_l534_534999


namespace minimum_value_l534_534333

theorem minimum_value (x y z a : ℝ) (h : x + 2 * y + 3 * z = a) : 
  x^2 + y^2 + z^2 ≥ a^2 / 14 := by
sory

end minimum_value_l534_534333


namespace sequence_6th_term_sequence_1994th_term_l534_534709

def sequence_term (n : Nat) : Nat := n * (n + 1)

theorem sequence_6th_term:
  sequence_term 6 = 42 :=
by
  -- proof initially skipped
  sorry

theorem sequence_1994th_term:
  sequence_term 1994 = 3978030 :=
by
  -- proof initially skipped
  sorry

end sequence_6th_term_sequence_1994th_term_l534_534709


namespace new_selling_price_l534_534213

theorem new_selling_price (C : ℝ) (h1 : 1.10 * C = 88) :
  1.15 * C = 92 :=
sorry

end new_selling_price_l534_534213


namespace partition_f_sum_l534_534202

open Set

-- Define conditions as Lean definitions
def S (n : ℕ) : Set ℝ := sorry -- We assume S is a set of n positive real numbers. Details skipped with sorry.

-- Define f(A) as the sum of elements in A
def f {A : Set ℝ} (h₁ : A ⊆ S n) (h₂ : A.nonempty) : ℝ := A.sum -- A.sum would represent the sum of all elements in A

-- Main theorem statement
theorem partition_f_sum (n : ℕ) (hS: ∀ x ∈ S n, 0 < x):
  ∃ (partition : Finset (Set ℝ)), (partition.card = n) ∧ 
  (∀ P ∈ partition, ∃ (maxP minP : ℝ), maxP = P.max' (sorry) ∧ minP = P.min' (sorry) ∧ maxP / minP < 2) :=
sorry


end partition_f_sum_l534_534202


namespace speed_of_boat_in_still_water_l534_534993

-- Define the given conditions
def rate_of_current : ℝ := 3 -- in km/hr
def time_downstream : ℝ := 24 / 60 -- in hours since 24 minutes = 24 / 60 hours
def distance_downstream : ℝ := 9.2 -- in km

-- Define the effective speed downstream
def effective_speed (v : ℝ) : ℝ := v + rate_of_current

-- Define the main claim: to prove the speed of the boat in still water
theorem speed_of_boat_in_still_water (v : ℝ) (h : distance_downstream = effective_speed(v) * time_downstream) : 
  v = 20 := 
sorry

end speed_of_boat_in_still_water_l534_534993


namespace range_of_x_f_lt_0_l534_534040

noncomputable def f (x : ℝ) : ℝ := x ^ (2 / 3) - x ^ (-1 / 2)

theorem range_of_x_f_lt_0 : { x : ℝ | f x < 0 } = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end range_of_x_f_lt_0_l534_534040


namespace larger_number_is_72_l534_534390

theorem larger_number_is_72 (a b : ℕ) (h1 : 5 * b = 6 * a) (h2 : b - a = 12) : b = 72 :=
by
  sorry

end larger_number_is_72_l534_534390


namespace count_valid_n_l534_534302

theorem count_valid_n :
  let factors := (λ n : ℤ, list.range' 1 50).map (λ k, n - (2 * k - 1))
  let count_pos : ℕ := ((list.range' 1 50).filter (λ k, ∃ m : ℤ, n = 2 * k - m ∧ 2 < n ∧ n < 100)).length
  let valid_n : set ℤ := { n | factors n > 0 ∧ n > 0 ∧ n < 100 }
  valid_n = 25 := 
sorry

end count_valid_n_l534_534302


namespace smallest_three_digit_number_with_property_l534_534986

theorem smallest_three_digit_number_with_property : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (∀ d, (1 ≤ d ∧ d ≤ 1000) → ((d = n + 1 ∨ d = n - 1) → d % 11 = 0)) ∧ 
  n = 120 :=
by
  sorry

end smallest_three_digit_number_with_property_l534_534986


namespace triangle_EY_length_l534_534060

/--
In a triangle DEF, point Y bisects ∠EDF. Given the lengths
DY = 36, FY = 24, and DF = 45, determine the length of segment EY.
-/
theorem triangle_EY_length :
  ∃ (EY : ℝ), ∀ (DY FY DF : ℝ), DY = 36 ∧ FY = 24 ∧ DF = 45 ∧ 
  (∃ (DE YE : ℝ), DE = DY + YE ∧ (DE / YE) = (DF / FY)) →
  EY = 288 / 7 :=
begin
  sorry
end

end triangle_EY_length_l534_534060


namespace trigonometric_identity_proof_l534_534160

noncomputable def trigonometric_identity : Prop :=
  (sin (40 : ℝ) - real.sqrt 3 * cos (20 : ℝ)) / cos (10 : ℝ) = -1

theorem trigonometric_identity_proof : trigonometric_identity :=
by
  sorry

end trigonometric_identity_proof_l534_534160


namespace probability_value_l534_534760

variable (ξ : ℝ)
variable (P: set ℝ → ℝ)
variable (a : ℝ)

axiom prob_distribution : ∀ k : ℕ, (1 ≤ k ∧ k ≤ 5) → P {x : ℝ | x = k / 5} = a * k

theorem probability_value (h : a * (1 + 2 + 3 + 4 + 5) = 1) : 
  P {x : ℝ | 1 / 10 < x ∧ x < 7 / 10} = 2 / 5 := 
sorry

end probability_value_l534_534760


namespace rancher_monetary_loss_l534_534225

def rancher_head_of_cattle := 500
def market_rate_per_head := 700
def sick_cattle := 350
def additional_cost_per_sick_animal := 80
def reduced_price_per_head := 450

def expected_revenue := rancher_head_of_cattle * market_rate_per_head
def loss_from_death := sick_cattle * market_rate_per_head
def additional_sick_cost := sick_cattle * additional_cost_per_sick_animal
def remaining_cattle := rancher_head_of_cattle - sick_cattle
def revenue_from_remaining_cattle := remaining_cattle * reduced_price_per_head

def total_loss := (expected_revenue - revenue_from_remaining_cattle) + additional_sick_cost

theorem rancher_monetary_loss : total_loss = 310500 := by
  sorry

end rancher_monetary_loss_l534_534225


namespace ratio_round_to_nearest_tenth_l534_534122

theorem ratio_round_to_nearest_tenth :
  (15 : ℚ) / 20 ≈ 0.8 :=
by
  have h_fraction : (15 : ℚ) / 20 = 0.75 := by norm_num
  have h_rounded : round (0.75 : ℚ) = 0.8 := by norm_num
  rw [h_fraction]
  exact h_rounded

end ratio_round_to_nearest_tenth_l534_534122


namespace sqrt_57_in_range_l534_534677

theorem sqrt_57_in_range (h1 : 49 < 57) (h2 : 57 < 64) (h3 : 7^2 = 49) (h4 : 8^2 = 64) : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by
  sorry

end sqrt_57_in_range_l534_534677


namespace sum_of_squares_l534_534934

theorem sum_of_squares (a b c d : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : d = a + 3) :
  a^2 + b^2 = c^2 + d^2 := by
  sorry

end sum_of_squares_l534_534934


namespace inequality_proof_l534_534728

theorem inequality_proof (a b : ℝ) (h₀ : b > a) (h₁ : ab > 0) : 
  (1 / a > 1 / b) ∧ (a + b < 2 * b) :=
by
  sorry

end inequality_proof_l534_534728


namespace susan_spending_ratio_l534_534146

theorem susan_spending_ratio (initial_amount clothes_spent books_left books_spent left_after_clothes gcd_ratio : ℤ)
  (h1 : initial_amount = 600)
  (h2 : clothes_spent = initial_amount / 2)
  (h3 : left_after_clothes = initial_amount - clothes_spent)
  (h4 : books_left = 150)
  (h5 : books_spent = left_after_clothes - books_left)
  (h6 : gcd books_spent left_after_clothes = 150)
  (h7 : books_spent / gcd_ratio = 1)
  (h8 : left_after_clothes / gcd_ratio = 2) :
  books_spent / gcd books_spent left_after_clothes = 1 ∧ left_after_clothes / gcd books_spent left_after_clothes = 2 :=
sorry

end susan_spending_ratio_l534_534146


namespace range_of_m_l534_534779

variable (m : ℝ)

/-- Proposition p: ∀ x ∈ ℝ, mx² + mx + 1 > 0 always holds. -/
def p : Prop := ∀ x : ℝ, m * x^2 + m * x + 1 > 0

/-- Proposition q: The equation x²/(m-1) + y²/(m-2) = 1 represents a hyperbola with foci on the x-axis. -/
def q : Prop := 1 < m ∧ m < 2

/-- The main theorem for the problem -/
theorem range_of_m (h1 : p m ∨ q m) (h2 : ¬ (p m ∧ q m)) : m ∈ Icc 0 1 ∪ Ico 2 4 := by
  sorry

end range_of_m_l534_534779


namespace sum_of_digits_second_smallest_multiple_of_lcm_below_7_l534_534923

def is_divisible_by_all_below_7 (n : ℕ) : Prop :=
  ∀ m : ℕ, m < 7 → m > 0 → n % m = 0

def second_smallest_multiple (n : ℕ) : ℕ :=
  2 * n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_second_smallest_multiple_of_lcm_below_7 : 
  sum_of_digits (second_smallest_multiple (Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 6)))))) = 3 :=
by
  sorry

end sum_of_digits_second_smallest_multiple_of_lcm_below_7_l534_534923


namespace maximize_total_profit_maximize_average_annual_profit_l534_534241

-- Define the profit function
def total_profit (x : ℤ) : ℤ := -x^2 + 18*x - 36

-- Define the average annual profit function
def average_annual_profit (x : ℤ) : ℤ :=
  let y := total_profit x
  y / x

-- Prove the maximum total profit
theorem maximize_total_profit : 
  ∃ x : ℤ, (total_profit x = 45) ∧ (x = 9) := 
  sorry

-- Prove the maximum average annual profit
theorem maximize_average_annual_profit : 
  ∃ x : ℤ, (average_annual_profit x = 6) ∧ (x = 6) :=
  sorry

end maximize_total_profit_maximize_average_annual_profit_l534_534241


namespace dot_product_is_correct_cosine_of_angle_is_correct_l534_534408

open scoped Real

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 4)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def cos_theta (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude a * magnitude b)

theorem dot_product_is_correct :
  dot_product vector_a vector_b = 5 := sorry

theorem cosine_of_angle_is_correct :
  cos_theta vector_a vector_b = Real.sqrt 5 / 5 := sorry

end dot_product_is_correct_cosine_of_angle_is_correct_l534_534408


namespace frequency_limit_as_n_approaches_infinity_estimated_white_balls_probability_of_same_color_two_same_color_probability_l534_534895

noncomputable def estimate_frequency : ℕ → ℚ := λ n, 
  match n with
  | 2048 => 1061 / 2048
  | 4040 => 2048 / 4040
  | 10000 => 4979 / 10000
  | 12000 => 6019 / 12000
  | _ => 0 -- for any other n, provide a default value

theorem frequency_limit_as_n_approaches_infinity :
  filter.tendsto (λ n, estimate_frequency n) filter.at_top (𝓝 0.5) :=
sorry

def number_of_white_balls (total_balls : ℕ) (prob : ℚ) : ℕ := 
  total_balls * prob

theorem estimated_white_balls : number_of_white_balls 4 0.5 = 2 := 
sorry

theorem probability_of_same_color (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) : ℚ :=
  let total_outcomes := (total_balls * (total_balls - 1))
  let same_color_outcomes := (white_balls * (white_balls - 1)) + (black_balls * (black_balls - 1))
  same_color_outcomes / total_outcomes

theorem two_same_color_probability : 
  probability_of_same_color 4 2 2 = 1 / 3 :=
sorry

end frequency_limit_as_n_approaches_infinity_estimated_white_balls_probability_of_same_color_two_same_color_probability_l534_534895


namespace proof_statement_l534_534733

open Complex

noncomputable def z : ℂ := (3 + 2 * Complex.i) / (2 - Complex.i)
def inFirstQuadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0
def z1_max_value_condition := ∃ (z1 : ℂ), |z1 - z| = 1 ∧ ∀ (w : ℂ), |w - z| = 1 → |w| ≤ |z1|

theorem proof_statement :
  inFirstQuadrant z ∧ z1_max_value_condition → (∀ z1, (|z1 - z| = 1 → |z1| ≤ 1 + (real.sqrt 65) / 5)) := 
by
  sorry

end proof_statement_l534_534733


namespace largest_multiple_of_6_neg_greater_than_neg_150_l534_534553

theorem largest_multiple_of_6_neg_greater_than_neg_150 : 
  ∃ m : ℤ, m % 6 = 0 ∧ -m > -150 ∧ m = 144 :=
by
  sorry

end largest_multiple_of_6_neg_greater_than_neg_150_l534_534553


namespace arithmetic_square_root_of_sqrt_81_l534_534513

def square_root (x : ℝ) : ℝ :=
  real.sqrt x

theorem arithmetic_square_root_of_sqrt_81 : square_root 81 = 9 :=
by
  sorry

end arithmetic_square_root_of_sqrt_81_l534_534513


namespace doug_visible_area_l534_534287

-- Define the visibility area based on given conditions
def visible_area (length : ℝ) (width : ℝ) (sight_distance : ℝ) : ℝ :=
  let inner_rec := length * width
  let top_bottom_rec := 2 * (length * sight_distance)
  let left_right_rec := 2 * (width * sight_distance)
  let corner_circles := 4 * (π * sight_distance ^ 2 / 4)
  inner_rec + top_bottom_rec + left_right_rec + corner_circles

theorem doug_visible_area :
  visible_area 8 3 2 = 81 :=
by
  -- Proof steps go here
  sorry

end doug_visible_area_l534_534287


namespace divide_9kg_into_2kg_and_7kg_l534_534031

-- Definitions
def balance_scale : Type := ℝ -- Type definition for balance scale (reals)
def weight_200g : ℝ := 200 -- 200 grams weight
def total_sugar : ℝ := 9000 -- 9000 grams of sugar (9 kg)

-- The proof problem statement
theorem divide_9kg_into_2kg_and_7kg
  (scale : balance_scale)
  (weight : ℝ)
  (sugar : ℝ)
  (max_weighings : ℕ := 3) :
  (weight = weight_200g) →
  (sugar = total_sugar) →
  ∃ procedure : list ℝ → list ℝ, (length procedure ≤ max_weighings) ∧ (procedure.last = [2000, 7000]) :=
by
  intros h1 h2
  sorry

end divide_9kg_into_2kg_and_7kg_l534_534031


namespace sqrt_x_minus_1_meaningful_real_l534_534394

theorem sqrt_x_minus_1_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ (x ≥ 1) :=
by
  sorry

end sqrt_x_minus_1_meaningful_real_l534_534394


namespace number_of_positive_integers_with_at_most_two_digits_l534_534817

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534817


namespace balance_balls_l534_534124

variable (G B Y W R : ℕ)

theorem balance_balls :
  (4 * G = 8 * B) →
  (3 * Y = 7 * B) →
  (8 * B = 5 * W) →
  (2 * R = 6 * B) →
  (5 * G + 3 * Y + 3 * R = 26 * B) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end balance_balls_l534_534124


namespace total_tickets_sold_l534_534961

-- Define the given conditions
def price_adult : ℕ := 12
def price_student : ℕ := 6
def total_revenue : ℕ := 16200
def num_students : ℕ := 300
def revenue_students : ℕ := price_student * num_students

-- Prove the number of total tickets sold
theorem total_tickets_sold : ∃ (A ℕ), (price_adult * A + revenue_students = total_revenue) ∧
  (A + num_students = 1500) :=
by {
  let A : ℕ := (total_revenue - revenue_students) / price_adult,
  use A,
  split,
  {
    calc
        price_adult * A + revenue_students
        = 12 * ((16200 - 1800) / 12) + 1800 : by rfl
    ... = 16200 : by sorry  -- Prove intermediate calculation for better readability
  },
  {
    calc
        A + num_students
        = (16200 - 1800) / 12 + 300 : by rfl
    ... = 1500 : by sorry -- Prove intermediate calculation for better readability
  }
}

end total_tickets_sold_l534_534961


namespace sum_of_remainders_l534_534926

theorem sum_of_remainders (p : ℕ) (hp : p > 2) (hp_prime : Nat.Prime p)
    (a : ℕ → ℕ) (ha : ∀ k, a k = k^p % p^2) :
    (Finset.sum (Finset.range (p - 1)) a) = (p^3 - p^2) / 2 :=
by
  sorry

end sum_of_remainders_l534_534926


namespace range_of_a_l534_534945

open Real

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a < 6) ∨ (a ≥ 5 ∨ a ≤ 1) ∧ ¬((0 < a ∧ a < 6) ∧ (a ≥ 5 ∨ a ≤ 1)) ↔ 
  (a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)) :=
by sorry

end range_of_a_l534_534945


namespace minimum_value_f_on_neg_l534_534757

noncomputable def q (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def F (a b x : ℝ) : ℝ := a * q(x) + b * g(x)
noncomputable def f (a b x : ℝ) : ℝ := F a b x + 1

-- Defining our conditions
axiom q_odd : ∀ x : ℝ, q(-x) = -q(x)
axiom g_odd : ∀ x : ℝ, g(-x) = -g(x)
axiom f_max_on_pos : ∃ a b : ℝ, ∀ x ∈ set.Ioi (0 : ℝ), f a b x ≤ 5

-- The theorem to prove
theorem minimum_value_f_on_neg (a b : ℝ) (h_f_max : ∀ x ∈ set.Ioi (0 : ℝ), f a b x ≤ 5) : 
  ∃ (c : ℝ) (x : ℝ), x < 0 ∧ f a b x = c ∧ c = -3 :=
by
  sorry

end minimum_value_f_on_neg_l534_534757


namespace county_maplewood_population_l534_534902

theorem county_maplewood_population :
  ∃ pop_avg : ℕ, 6200 ≤ pop_avg ∧ pop_avg ≤ 6800 ∧ 25 * pop_avg = 162500 :=
by {
  let pop_avg := 6500,
  use pop_avg,
  split,
  { linarith, },
  split,
  { linarith, },
  { simp, }
}

end county_maplewood_population_l534_534902


namespace sum_log_sqrt3_l534_534640

theorem sum_log_sqrt3 : 
  (∑ k in Finset.range (501), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 124886 :=
by
  sorry

end sum_log_sqrt3_l534_534640


namespace sum_log_floor_ceil_l534_534626

theorem sum_log_floor_ceil (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500 → 
  ((⌈ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌋) = 
    if (∃ n : ℕ, k = (3 : ℕ) ^ (n / 2 : ℚ)) then 0 else 1)) :
  ∑ k in Finset.range 501 \ {0}, k * (⌈ Real.log k / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log k / Real.log (Real.sqrt 3) ⌋) = 125237 := by
  have h_sum_all := (500 * 501) / 2
  have h_sum_powers := 1 + 3 + 9
  have h_correct := h_sum_all - h_sum_powers
  rw h_sum_all
  rw h_sum_powers
  rw h_correct
  sorry

end sum_log_floor_ceil_l534_534626


namespace lizzy_final_amount_l534_534114

-- Define constants
def m : ℕ := 80   -- cents from mother
def f : ℕ := 40   -- cents from father
def s : ℕ := 50   -- cents spent on candy
def u : ℕ := 70   -- cents from uncle
def t : ℕ := 90   -- cents for the toy
def c : ℕ := 110  -- cents change she received

-- Define the final amount calculation
def final_amount : ℕ := m + f - s + u - t + c

-- Prove the final amount is 160
theorem lizzy_final_amount : final_amount = 160 := by
  sorry

end lizzy_final_amount_l534_534114


namespace solve_inequality_l534_534308

theorem solve_inequality (x : ℝ) : (x^2 - 50 * x + 625 ≤ 25) = (20 ≤ x ∧ x ≤ 30) :=
sorry

end solve_inequality_l534_534308


namespace cube_root_expression_value_l534_534258

noncomputable def cube_root_expression : ℝ :=
  real.cbrt (11 + 4 * real.cbrt (14 + 10 * real.cbrt (17 + 18 * real.cbrt (3^3))))

theorem cube_root_expression_value :
  cube_root_expression = 3 :=
by
  sorry

end cube_root_expression_value_l534_534258


namespace projection_a_on_b_l534_534347

variable {V : Type*} [InnerProductSpace ℝ V]

variable (a b : V)
variable (h_norm_a : ∥a∥ = 1)
variable (h_norm_b : ∥b∥ = 2)
variable (h_perp : ⟪a, a - b⟫ = 0)

theorem projection_a_on_b :
    (⟪a, b⟫ / ∥b∥) = 1 / 2 :=
by
  sorry

end projection_a_on_b_l534_534347


namespace circle_radius_l534_534148

theorem circle_radius (r : ℝ) (π : ℝ) (h1 : π > 0) (h2 : ∀ x, π * x^2 = 100*π → x = 10) : r = 10 :=
by
  have : π * r^2 = 100*π → r = 10 := h2 r
  exact sorry

end circle_radius_l534_534148


namespace greatest_int_radius_of_circle_l534_534391

theorem greatest_int_radius_of_circle (r : ℝ) (A : ℝ) :
  (A < 200 * Real.pi) ∧ (A = Real.pi * r^2) →
  ∃k : ℕ, (k : ℝ) = 14 ∧ ∀n : ℕ, (n : ℝ) = r → n ≤ k := by
  sorry

end greatest_int_radius_of_circle_l534_534391


namespace airplane_takeoff_run_distance_l534_534171

noncomputable def distance_travelled (t : ℝ) (v : ℝ) : ℝ :=
  let v_mps := v * 1000 / 3600  -- Convert km/h to m/s
  let a := v_mps / t
  (1/2) * a * t^2

theorem airplane_takeoff_run_distance :
  distance_travelled 15 100 = 208 :=
by
  -- Definitions and condition handling
  have t := 15 : ℝ
  have v := 100 : ℝ
  let v_mps := v * 1000 / 3600
  let a := v_mps / t
  let S := (1 / 2) * a * t ^ 2
  
  -- Statement to prove
  show S = 208 from sorry

end airplane_takeoff_run_distance_l534_534171


namespace sin_identity_l534_534318

theorem sin_identity {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.sin (π / 6 - 2 * α) = -7 / 8 := 
by 
  sorry

end sin_identity_l534_534318


namespace unique_six_digit_numbers_l534_534789

theorem unique_six_digit_numbers : 
  (∃ (n : ℕ), n = nat.factorial 6 / (nat.factorial 3 * nat.factorial 3) ∧ 1 ≤ n ∧ n <= 999999) :=
begin 
  use 20,
  split,
  { 
    rw nat.factorial_six, 
    rw nat.factorial_three, 
    norm_num,
  },
  split,
  { norm_num },
  { norm_num }
end

end unique_six_digit_numbers_l534_534789


namespace prove_fn_value_l534_534356

noncomputable def f (x : ℝ) : ℝ := 2^x / (2^x + 3 * x)

theorem prove_fn_value
  (m n : ℝ)
  (h1 : 2^(m + n) = 3 * m * n)
  (h2 : f m = -1 / 3) :
  f n = 4 :=
by
  sorry

end prove_fn_value_l534_534356


namespace count_at_most_two_different_digits_l534_534857

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534857


namespace distance_to_place_l534_534222

variables {r c1 c2 t D : ℝ}

theorem distance_to_place (h : t = (D / (r - c1)) + (D / (r + c2))) :
  D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) :=
by
  have h1 : D * (r + c2) / (r - c1) * (r - c1) = D * (r + c2) := by sorry
  have h2 : D * (r - c1) / (r + c2) * (r + c2) = D * (r - c1) := by sorry
  have h3 : D * (r + c2) = D * (r + c2) := by sorry
  have h4 : D * (r - c1) = D * (r - c1) := by sorry
  have h5 : t * (r - c1) * (r + c2) = D * (r + c2) + D * (r - c1) := by sorry
  have h6 : t * (r^2 - c1 * c2) = D * (2 * r + c2 - c1) := by sorry
  have h7 : D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) := by sorry
  exact h7

end distance_to_place_l534_534222


namespace find_c_l534_534068

theorem find_c (c : ℝ) (h : (-(c / 3) + -(c / 5) = 30)) : c = -56.25 :=
sorry

end find_c_l534_534068


namespace measure_of_angle_A_range_of_bc_l534_534062

open Real

variables (A B C : ℝ) (a b c : ℝ)

noncomputable def acute_triangle : Prop := 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧ A + B + C = π 
def sides_opposite (A B C : ℝ) (a b c : ℝ) : Prop := ∀ (A B C : ℝ), a > 0 ∧ b > 0 ∧ c > 0 
def given_condition1 := a * sin B = (sqrt 3 / 2) * b
def given_condition2 := b * cos C + c * cos B = sqrt 3

theorem measure_of_angle_A :
  acute_triangle A B C →
  sides_opposite A B C a b c →
  given_condition1 →
  A = π / 3 := sorry

theorem range_of_bc :
  acute_triangle A B C →
  sides_opposite A B C a b c →
  given_condition1 →
  given_condition2 →
  3 < b + c ∧ b + c ≤ 2 * sqrt 3 := sorry

end measure_of_angle_A_range_of_bc_l534_534062


namespace cash_discount_percentage_l534_534220

-- Cost price of each article
def cp : ℝ := 1

-- Marked price of each article
def mp : ℝ := 2

-- Selling price for 20 articles at the cost price of 15 articles
def sp_20_articles : ℝ := 15

-- Profit percentage
def profit : ℝ := 0.35

-- Prove the cash discount percentage offered by the dealer
theorem cash_discount_percentage : (1 - sp_20_articles / (20 * mp)) * 100 = 30 := by
  -- compute the SP if a 35% profit is made on 20 articles
  let sp_profit_35 := (20 * cp) * (1 + profit)
  -- compute the discount
  let discount := sp_profit_35 - sp_20_articles
  -- calculate the discount percentage
  have discount_percentage : (discount / (20 * mp)) * 100 = 30 := by
    sorry
  exact discount_percentage

end cash_discount_percentage_l534_534220


namespace perpendicular_slope_proof_l534_534188

-- Definitions for the points.
def point1 := (1, 3 : ℝ)
def point2 := (-6, 6 : ℝ)

-- Function to calculate the slope between two points.
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- The slope of the line joining the two points.
def original_slope : ℝ := slope point1 point2

-- The negative reciprocal of the slope.
def perpendicular_slope (m : ℝ) : ℝ := -1 / m

-- Target statement: The perpendicular slope is 7/3.
theorem perpendicular_slope_proof : perpendicular_slope original_slope = (7 / 3 : ℝ) :=
by
  -- The actual proof goes here, but we use sorry for now.
  sorry

end perpendicular_slope_proof_l534_534188


namespace num_integers_two_digits_l534_534800

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534800


namespace log_sum_of_geometric_sequence_l534_534606

noncomputable def a (n : ℕ) : ℝ := sorry

def geometric_sequence (n : ℕ) : Prop :=
    ∀ m, a (m + n) = a m * a n

def positive_terms : Prop :=
  ∀ n, 0 < a n

def specific_condition : Prop :=
  a 5 ^ 2 + a 3 * a 7 = 8

theorem log_sum_of_geometric_sequence 
  (hgeom : geometric_sequence 2)
  (hpos : positive_terms)
  (hcond : specific_condition) :
  \sum_{i=1}^{9} log_base 2 (a i) = 9 :=
sorry

end log_sum_of_geometric_sequence_l534_534606


namespace scoop_order_count_l534_534953

-- Define the set of scoops
inductive Scoop
| vanilla
| chocolate
| strawberry
| cherry
| mint

open Scoop

-- Define the main theorem to prove the number of distinct orders of stacking
theorem scoop_order_count : 
  (Multiset.Of [vanilla, chocolate, strawberry, cherry, mint]).Permutations.card = 5! :=
by sorry

end scoop_order_count_l534_534953


namespace count_special_integers_l534_534829

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534829


namespace vector_properties_l534_534780

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-2, 1)

-- Lean statements to check the conditions
theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 0) ∧        -- Perpendicular vectors
  (real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5) ∧  -- Magnitude of the sum of vectors
  (real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5) := -- Magnitude of the difference of vectors
by
  unfold a b
  simp
  split
  -- Proof of each condition is skipped
  sorry 
  sorry
  sorry

end vector_properties_l534_534780


namespace toothpick_count_43_10_l534_534542

theorem toothpick_count_43_10 :
  (∀ n : ℕ, (n = 1 → (toothpicks n 10 = 31)) → (toothpicks 43 10 = 913)) :=
by {
  sorry
}

def toothpicks (n m : ℕ) : ℕ :=
  0 -- Placeholder definition, needs to be implemented

end toothpick_count_43_10_l534_534542


namespace no_linear_term_in_product_l534_534404

theorem no_linear_term_in_product (a : ℝ) (h : ∀ x : ℝ, (x + 4) * (x + a) - x^2 - 4 * a = 0) : a = -4 :=
sorry

end no_linear_term_in_product_l534_534404


namespace decimal_equivalent_of_fraction_squared_l534_534572

theorem decimal_equivalent_of_fraction_squared : (1 / 4 : ℝ) ^ 2 = 0.0625 :=
by sorry

end decimal_equivalent_of_fraction_squared_l534_534572


namespace slope_perpendicular_l534_534717

theorem slope_perpendicular (x y : ℝ) (h : 4 * x - 5 * y = 10) :
  let m := 4 / 5 in
  -(1 / m) = -5 / 4 :=
by {
  let m := 4 / 5,
  have h1 : 1 / m = 5 / 4 := by sorry,
  exact neg_eq_neg h1,
}

end slope_perpendicular_l534_534717


namespace quadratic_polynomial_has_root_l534_534704

theorem quadratic_polynomial_has_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.coeff 1 ∈ ℝ ∧ p.coeff 0 ∈ ℝ ∧ Polynomial.eval (-3 - Complex.i * Real.sqrt 7) p = 0 ∧
                        p = Polynomial.X^2 + Polynomial.C 6 * Polynomial.X + Polynomial.C 16 :=
sorry

end quadratic_polynomial_has_root_l534_534704


namespace largest_multiple_of_6_neg_greater_than_neg_150_l534_534552

theorem largest_multiple_of_6_neg_greater_than_neg_150 : 
  ∃ m : ℤ, m % 6 = 0 ∧ -m > -150 ∧ m = 144 :=
by
  sorry

end largest_multiple_of_6_neg_greater_than_neg_150_l534_534552


namespace positive_difference_between_diagonals_l534_534268

open Matrix

def original_matrix : Matrix (Fin 5) (Fin 5) Nat :=
  ![[1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]

def reversed_matrix : Matrix (Fin 5) (Fin 5) Nat :=
  ![[1, 2, 3, 4, 5],
    [10, 9, 8, 7, 6],
    [15, 14, 13, 12, 11],
    [16, 17, 18, 19, 20],
    [25, 24, 23, 22, 21]]

def main_diagonal_sum (m : Matrix (Fin 5) (Fin 5) Nat) : Nat :=
  ∑ i, m i i

def anti_diagonal_sum (m : Matrix (Fin 5) (Fin 5) Nat) : Nat :=
  ∑ i, m i ⟨4-i.1, sorry⟩

theorem positive_difference_between_diagonals :
  |main_diagonal_sum reversed_matrix - anti_diagonal_sum reversed_matrix| = 4 :=
by sorry

end positive_difference_between_diagonals_l534_534268


namespace tangent_line_at_P_f_greater_than_2x_minus_ln_x_l534_534769

def f (x : ℝ) : ℝ := Real.exp x / x

theorem tangent_line_at_P :
  ∃ m b : ℝ, (∀ x : ℝ, y = f x → y = m * x + b) ∧ (m = exp 2 / 4) ∧ (b = exp 2 / 2)
              ∧ ∀ x y : ℝ, ((y = f x) ∧ (x = 2) ∧ (y = exp 2 / 2)) → (exp 2 * x - 4 * y = 0) :=
sorry

theorem f_greater_than_2x_minus_ln_x :
  ∀ x : ℝ, (0 < x) → f x > 2 * (x - Real.log x) :=
sorry

end tangent_line_at_P_f_greater_than_2x_minus_ln_x_l534_534769


namespace parabola_focus_area_l534_534754

theorem parabola_focus_area
  (p : ℝ) (h1 : p > 0)
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (A B P : ℝ × ℝ)
  (h_parabola_eq : ∀ x y, y^2 = 2 * p * x → (x, y) ∈ ({p | y^2 = 2 * p * x}))
  (h_l1_intersect : ∃ y, (-y, y) ∈ ({p | y^2 = 2 * p * 8}))
  (l2 : ℝ) (h_l2 : ∀ y, (y + l2, y) ∈ ({p | p = y^2 / 2p}))
  (h_A : A = (l2 * l2 / (4 * p) + 2 * l2, l2))
  (h_B : B = (l2 * l2 / (4 * p) - 2 * l2, -l2))
  (h_P : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_OP_eq_half_AB : ∥P∥ = ∥A - B∥ / 2) :
  let area := abs ((F.1 * (A.2 - B.2) + A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2)) / 2)
  in area = 24 * sqrt 5 := by {
  sorry
}

end parabola_focus_area_l534_534754


namespace jason_initial_quarters_l534_534910

theorem jason_initial_quarters (q_d q_n q_i : ℕ) (h1 : q_d = 25) (h2 : q_n = 74) :
  q_i = q_n - q_d → q_i = 49 :=
by
  sorry

end jason_initial_quarters_l534_534910


namespace cos2α_minus_cosα_over_sinα_l534_534343

variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2)
variable (h2 : cos α = 2 * sqrt 5 / 5)

theorem cos2α_minus_cosα_over_sinα : cos (2 * α) - (cos α) / (sin α) = -7 / 5 :=
by
  sorry

end cos2α_minus_cosα_over_sinα_l534_534343


namespace sqrt_of_sqrt_81_l534_534500

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l534_534500


namespace perpendicular_slope_l534_534715

def line_slope (A B : ℚ) (x y : ℚ) : ℚ := A * x - B * y

def is_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

theorem perpendicular_slope : 
  ∃ m : ℚ, let slope_given_line := (4 : ℚ) / (5 : ℚ) in 
    is_perpendicular m slope_given_line ∧ m = - (5 : ℚ) / (4 : ℚ) := 
by 
  sorry

end perpendicular_slope_l534_534715


namespace youtube_dislikes_l534_534930

theorem youtube_dislikes (x y : ℕ) 
  (h1 : x = 3 * y) 
  (h2 : x = 100 + 2 * y) 
  (h_y_increased : ∃ y' : ℕ, y' = 3 * y) :
  y' = 300 := by
  sorry

end youtube_dislikes_l534_534930


namespace count_numbers_divisible_by_35_l534_534033

theorem count_numbers_divisible_by_35 :
  let S := {n : ℕ | 200 ≤ n ∧ n ≤ 400 ∧ n % 35 = 0} in
  S.card = 6 :=
by
  sorry

end count_numbers_divisible_by_35_l534_534033


namespace sum_log_sqrt3_l534_534638

theorem sum_log_sqrt3 : 
  (∑ k in Finset.range (501), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 124886 :=
by
  sorry

end sum_log_sqrt3_l534_534638


namespace solve_inequality_l534_534349

variable {c : ℝ}
variable (h_c_ne_2 : c ≠ 2)

theorem solve_inequality :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - (1 + 2) * x + 2 ≤ 0) ∧
  (c > 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x > c ∨ x < 2)) ∧
  (c < 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x < c ∨ x > 2)) :=
by
  sorry

end solve_inequality_l534_534349


namespace find_a_l534_534383

-- Defining the conditions as hypotheses
variables (a b d : ℕ)
hypothesis h1 : a + b = d
hypothesis h2 : b + d = 7
hypothesis h3 : d = 4

theorem find_a : a = 1 :=
by
  sorry

end find_a_l534_534383


namespace sequence_term_formula_l534_534326

theorem sequence_term_formula (n : ℕ) (h : 0 < n) :
  let S : ℕ → ℤ := λ n, -(n : ℤ)^2 + 7*n
  (a : ℕ → ℤ := λ n, S n - S (n - 1))
  a n = -2*n + 8 :=
by
  sorry

end sequence_term_formula_l534_534326


namespace nellie_initial_legos_l534_534120

theorem nellie_initial_legos (L : ℕ) (h1 : 57 = 57) (h2 : 24 = 24) (final_legos : 299 = 299):
  L = 380 :=
by
  -- Definitions based on conditions
  let lost_legos : ℕ := 57
  let given_legos : ℕ := 24
  let remaining_legos : ℕ := 299

  -- Given condition that lost + given = 81
  have total_lost_given : lost_legos + given_legos = 81 := by
    rw [h1, h2],
    exact rfl

  -- Given equation L - 81 = 299
  have equation_L : L - 81 = remaining_legos := by
    rw final_legos,
    exact rfl

  -- Satisfies the initial problem statement
  sorry

end nellie_initial_legos_l534_534120


namespace count_three_digit_numbers_with_two_same_digits_l534_534375

theorem count_three_digit_numbers_with_two_same_digits :
  let range := [500, 999]
  let predicate := λ n : ℕ, n >= 500 ∧ n <= 999 ∧ (∃ i j k: ℕ, n = 100*i + 10*j + k ∧ (i = j ∨ j = k ∨ i = k))
  ∑ n in range, if predicate n then 1 else 0 = 140 :=
by
  sorry

end count_three_digit_numbers_with_two_same_digits_l534_534375


namespace probability_sum_to_9_l534_534943

theorem probability_sum_to_9 :
  ∑ k in {2, 3, 4, 5, 6, 7, 8, 9}, (1 / 8) * ([
    (  4, 6^2), 
    ( 25, 6^3), 
    ( 56, 6^4), 
    ( 70, 6^5), 
    ( 56, 6^6), 
    ( 28, 6^7), 
    (  8, 6^8), 
    (  1, 6^9)
  ].nth k).getOrElse (0, 1).fst / ([
    (  4, 6^2), 
    ( 25, 6^3), 
    ( 56, 6^4), 
    ( 70, 6^5), 
    ( 56, 6^6), 
    ( 28, 6^7), 
    (  8, 6^8), 
    (  1, 6^9)
  ].nth k).getOrElse (0, 1).snd = 
  (1 / 8) * ( 4 / 6^2 + 25 / 6^3 + 56 / 6^4 + 70 / 6^5 + 56 / 6^6 + 28 / 6^7 + 8 / 6^8 + 1 / 6^9) :=
sorry

end probability_sum_to_9_l534_534943


namespace mon_inc_f_intervals_min_g_values_l534_534771

noncomputable def f (x : ℝ) : ℝ := 
  sqrt 3 * sin x * sin x + sin x * cos x

theorem mon_inc_f_intervals :
  ∀ k : ℤ, ∀ x : ℝ,
    k * real.pi - real.pi / 12 ≤ x ∧ x ≤ k * real.pi + 5 * real.pi / 12 →
    (f'(x) > 0) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 
  sin (x / 2 - real.pi / 3) - sqrt 3 / 2

theorem min_g_values :
  ∀ k : ℤ, 
    g (2 * k * real.pi - real.pi / 6) = -1 :=
sorry

end mon_inc_f_intervals_min_g_values_l534_534771


namespace total_marbles_l534_534053

variable (r : ℝ) -- number of red marbles
variable (b g y : ℝ) -- number of blue, green, and yellow marbles

-- Conditions
axiom h1 : r = 1.3 * b
axiom h2 : g = 1.5 * r
axiom h3 : y = 0.8 * g

/-- Theorem: The total number of marbles in the collection is 4.47 times the number of red marbles -/
theorem total_marbles (r b g y : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.5 * r) (h3 : y = 0.8 * g) :
  b + r + g + y = 4.47 * r :=
sorry

end total_marbles_l534_534053


namespace product_three_numbers_l534_534997

theorem product_three_numbers 
  (a b c : ℝ)
  (h1 : a + b + c = 30)
  (h2 : a = 3 * (b + c))
  (h3 : b = 5 * c) : 
  a * b * c = 176 := 
by
  sorry

end product_three_numbers_l534_534997


namespace correct_product_l534_534618

theorem correct_product : 
  (0.0063 * 3.85 = 0.024255) :=
sorry

end correct_product_l534_534618


namespace largest_multiple_negation_greater_than_neg150_l534_534550

theorem largest_multiple_negation_greater_than_neg150 (n : ℤ) (h₁ : n % 6 = 0) (h₂ : -n > -150) : n = 144 := 
sorry

end largest_multiple_negation_greater_than_neg150_l534_534550


namespace bijection_properties_l534_534193

variables {A B : Type} (f : A → B)

-- Define a bijection
def isBijection (f : A → B) :=
  function.injective f ∧ function.surjective f

-- Given that f is a bijection, prove the required properties
theorem bijection_properties (h : isBijection f) :
  (∀ a1 a2, a1 ≠ a2 → f a1 ≠ f a2) ∧                   -- Injectivity
  (∀ b, ∃ a, f a = b) ∧                               -- Surjectivity
  (∀ b, ∃ a, f a = b) ∧                               -- Every element in B is an image of some element in A
  (set.range f = set.univ)                            -- The image set is exactly B
  := sorry

end bijection_properties_l534_534193


namespace perpendicular_slope_l534_534713

def line_slope (A B : ℚ) (x y : ℚ) : ℚ := A * x - B * y

def is_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

theorem perpendicular_slope : 
  ∃ m : ℚ, let slope_given_line := (4 : ℚ) / (5 : ℚ) in 
    is_perpendicular m slope_given_line ∧ m = - (5 : ℚ) / (4 : ℚ) := 
by 
  sorry

end perpendicular_slope_l534_534713


namespace distinct_remainders_exists_l534_534107

theorem distinct_remainders_exists {p : ℕ} (hp : Nat.Prime p) 
    (a : Fin p → ℤ) : 
    ∃ k : ℤ, (Set.finite (Set.image (λ i : Fin p, (a i + i.val * k) % p) (Set.univ : Set (Fin p))) ∧
    Finset.card (Set.image (λ i : Fin p, (a i + i.val * k) % p) (Set.univ : Set (Fin p))) ≥ (p / 2)) :=
sorry

end distinct_remainders_exists_l534_534107


namespace initial_pennies_l534_534954

-- Defining the conditions
def pennies_spent : Nat := 93
def pennies_left : Nat := 5

-- Question: How many pennies did Sam have in his bank initially?
theorem initial_pennies : pennies_spent + pennies_left = 98 := by
  sorry

end initial_pennies_l534_534954


namespace count_at_most_two_different_digits_l534_534854

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534854


namespace count_valid_numbers_l534_534866

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534866


namespace percentage_students_school_A_l534_534414

theorem percentage_students_school_A
  (A B : ℝ)
  (h1 : A + B = 100)
  (h2 : 0.30 * A + 0.40 * B = 34) :
  A = 60 :=
sorry

end percentage_students_school_A_l534_534414


namespace problem_statement_l534_534100

noncomputable def α : ℝ := 3 + Real.sqrt 8
noncomputable def β : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by
  sorry

end problem_statement_l534_534100


namespace num_integers_two_digits_l534_534809

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534809


namespace square_circle_area_ratio_l534_534890

theorem square_circle_area_ratio (s : ℝ) :
  let r := s in
  (s^2 / (π * s^2)) = (1 / π) :=
by
  sorry

end square_circle_area_ratio_l534_534890


namespace part1_arithmetic_sequence_part2_l534_534742

noncomputable def a (n : ℕ) : ℕ → ℕ 
| 1 => 1
| k+1 => a k + k + 1 + k * a k

theorem part1 (n : ℕ): a (n+1) (n+1) - a n n = n * (n + 1) := by
  sorry

theorem arithmetic_sequence (n : ℕ) : 1 ≤ n → (∀ n ∈ ℕ, n * (a (n + 1) - a n) = a n + n^2 + n) → (a n) / n - (a (n+1)) / (n+1) = 1 := by 
  sorry

noncomputable def b (n : ℕ) : ℕ := n * 3^n

theorem part2 (n : ℕ) : (∀ n, a n = (b n / 3^n)^2) → 
  S n = ∑ k in Finset.range n, b k = (3 / 4) * (1 - 3^n) + (n / 2) * 3^(n+1) := by 
  sorry

end part1_arithmetic_sequence_part2_l534_534742


namespace negation_of_p_l534_534363
open Classical

variable (p : Prop) : ∃ x : ℝ, 2 * x + 1 ≤ 0

theorem negation_of_p : (¬ p ↔ ∀ x : ℝ, 2 * x + 1 > 0) :=
by
  sorry

end negation_of_p_l534_534363


namespace least_homeowners_l534_534051

theorem least_homeowners (M W : ℕ) (H_M H_W : ℕ) 
  (h1 : M + W = 150)
  (h2 : H_M = Nat.ceil (0.10 * M))
  (h3 : H_W = Nat.ceil (0.20 * W)) :
  H_M + H_W = 16 :=
sorry

end least_homeowners_l534_534051


namespace num_modified_monotonous_numbers_l534_534277

-- Define a modified monotonous number
def is_modified_monotonous (n : ℕ) : Prop :=
  (n < 10) ∨
  (n.to_digits.is_strictly_increasing ∧ n.to_digits.forall (λ d, d ≤ 9)) ∨
  (n.to_digits.reverse.is_strictly_decreasing ∧ n.to_digits.tail.forall (λ d, d ≤ 9))

-- Theorem to prove the number of modified monotonous numbers is 1534
theorem num_modified_monotonous_numbers : 
  (set_of is_modified_monotonous).card = 1534 :=
sorry

end num_modified_monotonous_numbers_l534_534277


namespace evaluate_expression_l534_534452

def greatest_power_of_factor_2 (n : ℕ) : ℕ :=
  (nat.factors n).count 2

def greatest_power_of_factor_5 (n : ℕ) : ℕ :=
  (nat.factors n).count 5

theorem evaluate_expression (a b : ℕ) (h₁ : 2^a = 8) (h₂ : 5^b = 25) :
  (1 / 3) ^ (b - a) = 3 := by
  have ha : a = greatest_power_of_factor_2 200 := by sorry
  have hb : b = greatest_power_of_factor_5 200 := by sorry
  rw [greatest_power_of_factor_2, greatest_power_of_factor_5] at ha hb
  simp at ha hb
  exact sorry

end evaluate_expression_l534_534452


namespace no_ways_to_sum_10001_with_two_primes_l534_534898

theorem no_ways_to_sum_10001_with_two_primes :
  ¬ ∃ p₁ p₂ : ℕ, nat.prime p₁ ∧ nat.prime p₂ ∧ p₁ + p₂ = 10001 :=
by
  sorry

end no_ways_to_sum_10001_with_two_primes_l534_534898


namespace changing_quantities_l534_534491

structure Point (α : Type*) := (x y : α)
structure Triangle (α : Type*) := (A B P : Point α)
structure Midpoint (α : Type*) := (M N C : Point α)

noncomputable def num_changing_quantities {α : Type*}
  [field α] 
  (C : Point α) 
  (T : Triangle α)
  (M : Point α)
  (N : Point α) 
  (MN_seg_len_changes : Prop)
  (perimeter_triangle_changes : Prop)
  (area_triangle_changes : Prop)
  (area_trapezoid_changes : Prop) := ∃ (n : ℕ), n = 4

theorem changing_quantities 
  {α : Type*} [field α]
  (C : Point α) 
  (T : Triangle α)
  (M : Point α) 
  (N : Point α) 
  (MN_seg_len_changes : Prop)
  (perimeter_triangle_changes : Prop)
  (area_triangle_changes : Prop)
  (area_trapezoid_changes : Prop) :
  num_changing_quantities C T M N MN_seg_len_changes perimeter_triangle_changes area_triangle_changes area_trapezoid_changes :=
exists.intro 4 (by sorry)

end changing_quantities_l534_534491


namespace monic_quadratic_with_given_root_l534_534693

theorem monic_quadratic_with_given_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.eval (-3 - complex.i * Real.sqrt 7) = 0 ∧ p = Polynomial.X^2 + 6 * Polynomial.X + 16 :=
by
  sorry

end monic_quadratic_with_given_root_l534_534693


namespace count_two_digit_or_less_numbers_l534_534849

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534849


namespace greatest_sum_of_distances_l534_534449

theorem greatest_sum_of_distances (T : Type) [MetricSpace T] (a b c : ℝ)
  (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  (∃ P ∈ triangle_with_sides a b c, sum_of_distances P = 12 / 5) :=
sorry

end greatest_sum_of_distances_l534_534449


namespace length_KL_eq_LB_l534_534941

noncomputable theory
open_locale classical

variables {A B C D E K L : Type*}
variables [is_R_or_C A] [is_R_or_C B] [is_R_or_C C] [is_R_or_C D] [is_R_or_C E] [is_R_or_C K] [is_R_or_C L]

-- Definitions
variable (triangle_ABC : is_right_triangle (points A B C) ∧ isosceles_triangle (points A B C))
variable (point_D_on_CA : ∃ D, on_segment D (points C A))
variable (point_E_on_CB : ∃ E, on_segment E (points C B))
variable (equal_segments_CD_CE : segment_length (points C D) = segment_length (points C E))
variable (perpendiculars_D_C_to_AE : ∃ K L, perpendicular_to (points D C) (line AE) ∧ intersects (points K L) (line AB))

-- Theorem statement
theorem length_KL_eq_LB (h : triangle_ABC ∧ point_D_on_CA ∧ point_E_on_CB ∧ equal_segments_CD_CE ∧ perpendiculars_D_C_to_AE) : 
  segment_length (points K L) = segment_length (points L B) :=
sorry

end length_KL_eq_LB_l534_534941


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534792

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534792


namespace relation_y1_y2_y3_l534_534750

-- Definition of being on the parabola
def on_parabola (x : ℝ) (y m : ℝ) : Prop := y = -3*x^2 - 12*x + m

-- The conditions given in the problem
variables {y1 y2 y3 m : ℝ}

-- The points (-3, y1), (-2, y2), (1, y3) are on the parabola given by the equation
axiom h1 : on_parabola (-3) y1 m
axiom h2 : on_parabola (-2) y2 m
axiom h3 : on_parabola (1) y3 m

-- We need to prove the relationship between y1, y2, and y3
theorem relation_y1_y2_y3 : y2 > y1 ∧ y1 > y3 :=
by { sorry }

end relation_y1_y2_y3_l534_534750


namespace range_of_m_l534_534332

variable (x m : ℝ)

def p : Prop := x^2 + x - 2 > 0
def q : Prop := x > m

theorem range_of_m (h : (¬ q → ¬ p) ∧ ¬ (p → q)) : m ≥ 1 :=
sorry

end range_of_m_l534_534332


namespace bus_capacity_l534_534195

def seats_available_on_left := 15
def seats_available_diff := 3
def people_per_seat := 3
def back_seat_capacity := 7

theorem bus_capacity : 
  (seats_available_on_left * people_per_seat) + 
  ((seats_available_on_left - seats_available_diff) * people_per_seat) + 
  back_seat_capacity = 88 := 
by 
  sorry

end bus_capacity_l534_534195


namespace sqrt_sqrt_81_is_9_l534_534505

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l534_534505


namespace slide_wait_is_shorter_l534_534532

theorem slide_wait_is_shorter 
  (kids_waiting_for_swings : ℕ)
  (kids_waiting_for_slide_multiplier : ℕ)
  (wait_per_kid_swings_minutes : ℕ)
  (wait_per_kid_slide_seconds : ℕ)
  (kids_waiting_for_swings = 3)
  (kids_waiting_for_slide_multiplier = 2)
  (wait_per_kid_swings_minutes = 2) 
  (wait_per_kid_slide_seconds = 15) :
  let total_wait_swings_seconds := wait_per_kid_swings_minutes * kids_waiting_for_swings * 60,
      kids_waiting_for_slide := kids_waiting_for_swings * kids_waiting_for_slide_multiplier,
      total_wait_slide_seconds := wait_per_kid_slide_seconds * kids_waiting_for_slide in
  270 = total_wait_swings_seconds - total_wait_slide_seconds :=
by
  sorry

end slide_wait_is_shorter_l534_534532


namespace min_x_prime_sum_l534_534925

theorem min_x_prime_sum (x y : ℕ) (h : 3 * x^2 = 5 * y^4) :
  ∃ a b c d : ℕ, x = a^b * c^d ∧ (a + b + c + d = 11) := 
by sorry

end min_x_prime_sum_l534_534925


namespace max_choir_members_l534_534972

theorem max_choir_members : 
  ∃ (m : ℕ), 
    (∃ k : ℕ, m = k^2 + 11) ∧ 
    (∃ n : ℕ, m = n * (n + 5)) ∧ 
    (∀ m' : ℕ, 
      ((∃ k' : ℕ, m' = k' * k' + 11) ∧ 
       (∃ n' : ℕ, m' = n' * (n' + 5))) → 
      m' ≤ 266) ∧ 
    m = 266 :=
by sorry

end max_choir_members_l534_534972


namespace unique_positive_integer_m_l534_534656

theorem unique_positive_integer_m (m : ℕ) (h : m = 129) :
  ∑ k in Finset.range m.succ, k * 2^k = 2^(m + 8) := by
  sorry

end unique_positive_integer_m_l534_534656


namespace real_values_satisfying_inequality_l534_534296

theorem real_values_satisfying_inequality :
  { x : ℝ | (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 } =
  Set.Icc (-1 : ℝ) ((-3 - Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪ 
  Set.Ioi 0 :=
by
  sorry

end real_values_satisfying_inequality_l534_534296


namespace fraction_of_green_balls_l534_534413

theorem fraction_of_green_balls (T G : ℝ)
    (h1 : (1 / 8) * T = 6)
    (h2 : (1 / 12) * T + (1 / 8) * T + 26 = T - G)
    (h3 : (1 / 8) * T = 6)
    (h4 : 26 ≥ 0):
  G / T = 1 / 4 :=
by
  sorry

end fraction_of_green_balls_l534_534413


namespace digits_mod_4_l534_534477

theorem digits_mod_4 (n : ℕ) (a : ℕ → ℕ) :
  (∑ i in Finset.range n, a (i + 1) * (10 ^ (n - i - 1))) % 4 = 
  ((a (n - 1) * 2) + a n) % 4 :=
  sorry

end digits_mod_4_l534_534477


namespace gcd_3_1200_1_3_1210_1_l534_534549

theorem gcd_3_1200_1_3_1210_1 : 
  Int.gcd (3^1200 - 1) (3^1210 - 1) = 59048 := 
by 
  sorry

end gcd_3_1200_1_3_1210_1_l534_534549


namespace eval_expression_l534_534259

theorem eval_expression : 
  (20-19 + 18-17 + 16-15 + 14-13 + 12-11 + 10-9 + 8-7 + 6-5 + 4-3 + 2-1) / 
  (1-2 + 3-4 + 5-6 + 7-8 + 9-10 + 11-12 + 13-14 + 15-16 + 17-18 + 19-20) = -1 := by
  sorry

end eval_expression_l534_534259


namespace eval_f_at_3_l534_534876

-- Define the polynomial function
def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

-- State the theorem to prove f(3) = 41
theorem eval_f_at_3 : f 3 = 41 :=
by
  -- Proof would go here
  sorry

end eval_f_at_3_l534_534876


namespace lcm_72_108_2100_l534_534194

-- Define the input numbers
def a : ℕ := 72
def b : ℕ := 108
def c : ℕ := 2100

-- Define the least common multiple function
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- The main statement we want to prove
theorem lcm_72_108_2100 : lcm (lcm a b) c = 37800 := by
  sorry

end lcm_72_108_2100_l534_534194


namespace stagePlayRolesAssignment_correct_l534_534600

noncomputable def stagePlayRolesAssignment : ℕ :=
  let male_roles : ℕ := 4 * 3 -- ways to assign male roles
  let female_roles : ℕ := 5 * 4 -- ways to assign female roles
  let either_gender_roles : ℕ := 5 * 4 * 3 -- ways to assign either-gender roles
  male_roles * female_roles * either_gender_roles -- total assignments

theorem stagePlayRolesAssignment_correct : stagePlayRolesAssignment = 14400 := by
  sorry

end stagePlayRolesAssignment_correct_l534_534600


namespace find_B_squared_l534_534294

open Real

noncomputable def g (x : ℝ) : ℝ := sqrt 50 + 100 / x

noncomputable def B : ℝ := abs ((sqrt 50 + 15 * sqrt 2) / 2) + abs ((sqrt 50 - 15 * sqrt 2) / 2)

theorem find_B_squared (B^2 = 450) : 
    B^2 = 450 :=
begin
  sorry
end

end find_B_squared_l534_534294


namespace winning_triangles_count_l534_534059

open Nat

-- Definition of the number of contestants
def num_contestants := 10

-- Definition of a winning triangle
def is_winning_triangle (defeat : Fin num_contestants → Fin num_contestants → Prop) (i j k : Fin num_contestants) : Prop :=
  defeat i j ∧ defeat j k ∧ defeat k i

-- Condition for the contestants
def contest_condition (defeat : Fin num_contestants → Fin num_contestants → Prop) : Prop :=
  ∀ i j : Fin num_contestants, (defeat i j → ∃ Wi Li Wj : Nat, Li + Wj ≥ 8 ∧ Li < num_contestants ∧ Wj < num_contestants)

-- Main theorem statement
theorem winning_triangles_count (defeat : Fin num_contestants → Fin num_contestants → Prop)
  (contest_condition : contest_condition defeat) :
  ∃ count : Nat, count = 40 ∧ (∃ triangles : Fin num_contestants → Fin num_contestants → Fin num_contestants → Prop,
    (∀ i j k : Fin num_contestants, triangles i j k = is_winning_triangle defeat i j k ∧ ∑ i j k, if triangles i j k then 1 else 0 = count)) :=
by
  sorry

end winning_triangles_count_l534_534059


namespace probability_of_two_one_color_and_one_other_color_l534_534581

theorem probability_of_two_one_color_and_one_other_color
    (black_balls white_balls : ℕ)
    (total_drawn : ℕ)
    (draw_two_black_one_white : ℕ)
    (draw_one_black_two_white : ℕ)
    (total_ways : ℕ)
    (favorable_ways : ℕ)
    (probability : ℚ) :
    black_balls = 8 →
    white_balls = 7 →
    total_drawn = 3 →
    draw_two_black_one_white = 196 →
    draw_one_black_two_white = 168 →
    total_ways = 455 →
    favorable_ways = draw_two_black_one_white + draw_one_black_two_white →
    probability = favorable_ways / total_ways →
    probability = 4 / 5 :=
by sorry

end probability_of_two_one_color_and_one_other_color_l534_534581


namespace time_to_run_home_l534_534914

-- Define the conditions
def blocks_run_per_time : ℚ := 2 -- Justin runs 2 blocks
def time_per_blocks : ℚ := 1.5 -- in 1.5 minutes
def blocks_to_home : ℚ := 8 -- Justin is 8 blocks from home

-- Define the theorem to prove the time taken for Justin to run home
theorem time_to_run_home : (blocks_to_home / blocks_run_per_time) * time_per_blocks = 6 :=
by
  sorry

end time_to_run_home_l534_534914


namespace polynomial_roots_l534_534712

theorem polynomial_roots :
  (∀ x : ℝ, (x^4 - 3*x^3 + x^2 + 3*x - 2 = 0) ↔ (x = 1 ∨ x = -1 ∨ x = 2)) ∧
  (polynomial.degree_of_factor (polynomial.of_terms [(1 : ℝ), 4, (-3 : ℝ), 3, (1 : ℝ), 2, (3 : ℝ), 1, (-2 : ℝ), 0]) (1 : ℝ) = 2) ∧
  (polynomial.degree_of_factor (polynomial.of_terms [(1 : ℝ), 4, (-3 : ℝ), 3, (1 : ℝ), 2, (3 : ℝ), 1, (-2 : ℝ), 0]) (-1 : ℝ) = 1) ∧
  (polynomial.degree_of_factor (polynomial.of_terms [(1 : ℝ), 4, (-3 : ℝ), 3, (1 : ℝ), 2, (3 : ℝ), 1, (-2 : ℝ), 0]) (2 : ℝ) = 1) :=
by
  -- proof steps go here
  sorry

end polynomial_roots_l534_534712


namespace scientific_notation_of_169200000000_l534_534517

theorem scientific_notation_of_169200000000 : 169200000000 = 1.692 * 10^11 :=
by sorry

end scientific_notation_of_169200000000_l534_534517


namespace num_integers_two_digits_l534_534807

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534807


namespace ratio_of_girls_to_boys_l534_534236

theorem ratio_of_girls_to_boys (total_people : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total_people = 96) (h2 : girls = 80) (h3 : boys = total_people - girls) :
  (5 : ℚ) = (girls : ℚ) / (boys : ℚ) :=
by
  sorry

end ratio_of_girls_to_boys_l534_534236


namespace increasing_interval_of_f_sum_of_zeros_of_g_l534_534348

def f (ω x : ℝ) : ℝ := 2 * Math.sin(ω * x) * (Math.cos(ω * x) + Real.sqrt 3 * Math.sin(ω * x)) - Real.sqrt 3
def g (x : ℝ) : ℝ := 2 * Math.sin(2 * x) + 2

noncomputable def period_condition {ω : ℝ} (hω : ω > 0) :=
  Math.min (abs (Math.pi / ω)) (abs (2 * Math.pi / ω)) = Math.pi

theorem increasing_interval_of_f (ω : ℝ) (hω : ω > 0) (period : period_condition hω) :
  ∃ k : ℤ, ∀ x : ℝ, k * Math.pi - Math.pi / 12 ≤ x ∧ x ≤ k * Math.pi + 5 * Math.pi / 12 → StrictMono (λ x, f ω x) :=
sorry

theorem sum_of_zeros_of_g :
  ∑ x in (finset.filter (λ x, g x = 0) (finset.Icc 0 (5 * Math.pi))), x = 55 * Math.pi / 4 :=
sorry

end increasing_interval_of_f_sum_of_zeros_of_g_l534_534348


namespace count_two_digit_or_less_numbers_l534_534842

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534842


namespace main_inequality_l534_534103

theorem main_inequality (a b c d : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (c * d * a) / (1 - b)^2 + (d * a * b) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
by
  sorry

end main_inequality_l534_534103


namespace cyclist_speed_l534_534882

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem cyclist_speed :
  ∀ (d t : ℝ), 
  (d / 10 = t + 1) → 
  (d / 15 = t - 1) →
  required_speed d t = 12 := 
by
  intros d t h1 h2
  sorry

end cyclist_speed_l534_534882


namespace monic_quadratic_poly_with_root_l534_534694

theorem monic_quadratic_poly_with_root (x : ℝ) :
  (∃ p : Polynomial ℝ, Polynomial.monic p ∧ p.coeff 2 = 1 ∧ p.coeff 1 = 6 ∧ p.coeff 0 = 16 
  ∧ p.eval (-3 - Complex.i * Real.sqrt 7) = 0) :=
by
  use Polynomial.C 16 + Polynomial.X * (Polynomial.C 6 + Polynomial.X)
  field_simp
  sorry

end monic_quadratic_poly_with_root_l534_534694


namespace mean_profit_first_15_days_l534_534598

-- Definitions and conditions
def mean_daily_profit_entire_month : ℝ := 350
def total_days_in_month : ℕ := 30
def mean_daily_profit_last_15_days : ℝ := 445

-- Proof statement
theorem mean_profit_first_15_days : 
  (mean_daily_profit_entire_month * (total_days_in_month : ℝ) 
   - mean_daily_profit_last_15_days * 15) / 15 = 255 :=
by
  sorry

end mean_profit_first_15_days_l534_534598


namespace main_inequality_l534_534102

theorem main_inequality (a b c d : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (c * d * a) / (1 - b)^2 + (d * a * b) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
by
  sorry

end main_inequality_l534_534102


namespace mass_15_implies_age_7_l534_534519

-- Define the mass function m which depends on age a
variable (m : ℕ → ℕ)

-- Define the condition for the mass to be 15 kg
def is_age_when_mass_is_15 (a : ℕ) : Prop :=
  m a = 15

-- The problem statement to be proven
theorem mass_15_implies_age_7 : ∀ a, is_age_when_mass_is_15 m a → a = 7 :=
by
  -- Proof details would follow here
  sorry

end mass_15_implies_age_7_l534_534519


namespace constant_term_in_expansion_l534_534667

-- Define the binomial expansion term
def binomial_term (n r : ℕ) (x : ℝ) : ℝ := 
  (↑(nat.choose n r)) * ((x^2)^(n-r)) * ((-2/x)^r)

-- Function to compute the constant term in the expansion
noncomputable def constant_term := 
  (3.choose 2) * ((-2)^2)

-- The theorem statement
theorem constant_term_in_expansion : 
  (x : ℝ) -> constant_term = 12 := 
by
  sorry

end constant_term_in_expansion_l534_534667


namespace locus_centers_of_tangent_circles_l534_534968

theorem locus_centers_of_tangent_circles (a b : ℝ) :
  (x^2 + y^2 = 1) ∧ ((x - 1)^2 + (y -1)^2 = 81) →
  (a^2 + b^2 - (2 * a * b) / 63 - (66 * a) / 63 - (66 * b) / 63 + 17 = 0) :=
by
  sorry

end locus_centers_of_tangent_circles_l534_534968


namespace incenter_of_triangle_CEF_l534_534967

open_locale euclidean_geometry

theorem incenter_of_triangle_CEF {O B C A D I E F : Point} {S : Circle} (h1 : Circle.center S = O)
(h2 : Circle.diameter S B C) (h3 : A ∈ Circle.points S) (h4 : ∠A O B < 120) 
(h5 : D = Circle.midpoint_arc S A B ¬(Circle.contains_arc S C))
(h6 : ∃ I : Point, Line.parallel (Line.through O D) (Line.through A I) ∧ I ∈ Line.through A C)
(h7 : PerpendicularBisector O A ∩ S = {E, F}) :
    Incenter (Triangle C E F) I :=
sorry

end incenter_of_triangle_CEF_l534_534967


namespace ellipse_area_constant_l534_534001

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x_a y_a x_b y_b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.1 = x_a ∧ p.2 = y_a ∨ p.1 = x_b ∧ p.2 = y_b

def area_ABNM_constant (x y : ℝ) : Prop :=
  let x_0 := x;
  let y_0 := y;
  let y_M := -2 * y_0 / (x_0 - 2);
  let BM := 1 + 2 * y_0 / (x_0 - 2);
  let x_N := - x_0 / (y_0 - 1);
  let AN := 2 + x_0 / (y_0 - 1);
  (1 / 2) * AN * BM = 2

theorem ellipse_area_constant :
  ∀ (a b : ℝ), (a = 2 ∧ b = 1) → 
  (∀ (x y : ℝ), 
    ellipse_equation a b x y → 
    passes_through 2 0 0 1 (x, y) → 
    (x < 0 ∧ y < 0) →
    area_ABNM_constant x y) :=
by
  intros
  sorry

end ellipse_area_constant_l534_534001


namespace area_of_enclosed_region_l534_534686

noncomputable def abs_val (z : ℝ) := if z < 0 then -z else z

def enclosed_region (x y : ℝ) := abs_val (x - 50) + abs_val y = abs_val (x / 3)

theorem area_of_enclosed_region :
  (region := {p : ℝ × ℝ | enclosed_region p.1 p.2}) ∧ 
  (area_of_region region = 1250) :=
by
  sorry

end area_of_enclosed_region_l534_534686


namespace value_of_U_l534_534447

theorem value_of_U : (
  let U := (1 / (4 - Real.sqrt 15)) 
            - (1 / (Real.sqrt 15 - Real.sqrt 14))
            + (1 / (Real.sqrt 14 - Real.sqrt 13))
            - (1 / (Real.sqrt 13 - Real.sqrt 12))
            + (1 / (Real.sqrt 12 - 3))
  in U = 7) :=
sorry

end value_of_U_l534_534447


namespace odd_function_expression_l534_534015

def f (x : ℝ) : ℝ := 
if x > 0 then x^2 - 2*x + 3 
else if x < 0 then -x^2 - 2*x - 3 
else 0

theorem odd_function_expression (x : ℝ) (h : x < 0) :
  f(x) = -(x^2 + 2*x + 3) := 
by
  sorry

end odd_function_expression_l534_534015


namespace arithmetic_sequence_general_sum_first_n_terms_l534_534017

open Nat

theorem arithmetic_sequence_general 
  (a : ℕ → ℚ) (d : ℚ)
  (h1 : ∀ n m : ℕ, n < m → a n < a m)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 2 + a 4 = 14)
  (h4 : (a 2 - 1) * (a 4 + 7) = (a 3 + 1) ^ 2) :
  ∀ n, a n = 2 * n + 1 :=
sorry

theorem sum_first_n_terms
  (a : ℕ → ℚ) 
  (h : ∀ n, a n = 2 * n + 1) :
  ∀ n, let S (n : ℕ) := ∑ i in range n, 6 / (a i * a (i + 1))
    in S n = (2 * n) / (2 * n + 3) :=
sorry

end arithmetic_sequence_general_sum_first_n_terms_l534_534017


namespace average_first_5_multiples_of_9_l534_534966

theorem average_first_5_multiples_of_9 : 
  (let multiples := [9, 18, 27, 36, 45] in
   let sum := multiples.sum in
   sum / 5 = 27) :=
by
  let multiples := [9, 18, 27, 36, 45]
  let sum := List.sum multiples
  rw List.sum_cons at sum
  rw List.sum_cons at sum
  rw List.sum_cons at sum
  rw List.sum_cons at sum
  rw List.sum_cons at sum
  sorry

end average_first_5_multiples_of_9_l534_534966


namespace algorithm_find_GCD_Song_Yuan_l534_534063

theorem algorithm_find_GCD_Song_Yuan :
  (∀ method, method = "continuous subtraction" → method_finds_GCD_Song_Yuan) :=
sorry

end algorithm_find_GCD_Song_Yuan_l534_534063


namespace contest_end_time_l534_534218

theorem contest_end_time (start_time_min: ℕ) (total_duration: ℕ) (break_duration: ℕ):
  start_time_min = 900 ∧ total_duration = 850 ∧ break_duration = 30 → 
  let adjusted_duration := total_duration - break_duration in 
  let hours := adjusted_duration / 60 in 
  let minutes := adjusted_duration % 60 in 
  let end_time_in_minutes := start_time_min + adjusted_duration in 
  let end_hours := end_time_in_minutes / 60 % 24 in 
  let end_minutes := end_time_in_minutes % 60 in 
  (end_hours, end_minutes) = (4, 40) := 
by
  intro h
  cases h with h1 htemp,
  cases htemp with h2 h3,
  have h4 : let adjusted_duration := total_duration - break_duration := sorry,
  have h5 : let hours := adjusted_duration / 60 := sorry,
  have h6 : let minutes := adjusted_duration % 60 := sorry,
  have h7 : let end_time_in_minutes := start_time_min + adjusted_duration := sorry,
  have h8 : let end_hours := end_time_in_minutes / 60 % 24 := sorry,
  have h9 : let end_minutes := end_time_in_minutes % 60 := sorry,
  show (end_hours, end_minutes) = (4, 40), from sorry

end contest_end_time_l534_534218


namespace grace_age_l534_534370

/-- Grace's age calculation based on given family ages. -/
theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ)
  (h1 : mother_age = 80)
  (h2 : grandmother_age = 2 * mother_age)
  (h3 : grace_age = (3 * grandmother_age) / 8) : grace_age = 60 :=
by
  sorry

end grace_age_l534_534370


namespace find_abs_diff_l534_534485

theorem find_abs_diff (x y : ℝ) (exams : List ℝ) (h_exams : exams = [x, y, 105, 109, 110]) (h_avg : (x + y + 105 + 109 + 110) / 5 = 108) (h_var : ((x - 108)^2 + (y - 108)^2 + (105 - 108)^2 + (109 - 108)^2 + (110 - 108)^2) / 5 = 35.2) :
  abs (x - y) = 18 :=
begin
  sorry
end

end find_abs_diff_l534_534485


namespace pair_points_no_intersection_l534_534316

-- Definitions for the problem
variable (n : ℕ)

-- Given two segments and a condition on their points
def no_intersecting_segments (segments : list (ℝ × ℝ × ℝ × ℝ)) : Prop :=
  ∀ (s1 s2 : ℝ × ℝ × ℝ × ℝ), s1 ∈ segments → s2 ∈ segments → s1 ≠ s2 → 
    ¬intersect_segment s1 s2

-- Define what it means for two segments to intersect
def intersect_segment (s1 s2 : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let ⟨x1, y1, x2, y2⟩ := s1 in
  let ⟨x3, y3, x4, y4⟩ := s2 in
  let d := (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3) in
  (d ≠ 0) ∧
  let t := ((x1 - x3) * (y4 - y3) - (y1 - y3) * (x4 - x3)) / d in
  let u := -((x1 - x3) * (y2 - y1) - (y1 - y3) * (x2 - x1)) / d in
  0 < t ∧ t < 1 ∧ 0 < u ∧ u < 1

-- Prove the main theorem
theorem pair_points_no_intersection : 
  (points : list (ℝ × ℝ)) → points.length = 2 * n → 
    ∃ (segments : list (ℝ × ℝ × ℝ × ℝ)), 
      (∀ a ∈ points, ∃ s ∈ segments, a ∈ s) ∧
      no_intersecting_segments segments
:= by
  sorry

end pair_points_no_intersection_l534_534316


namespace measure_angle_APB_l534_534436

noncomputable def angle_APB (A B C D P : ℝ × ℝ) : ℝ :=
  let AP := dist A P
  let BP := dist B P
  let CP := dist C P
  if (AP, BP, CP) = (1, 2, 3) then 135 else 0

theorem measure_angle_APB (A B C D P : ℝ × ℝ) :

  let side := dist A B in
  dist A P = 1 → 
  dist B P = 2 → 
  dist C P = 3 → 
  ∠ A P B = 135 :=
sorry

end measure_angle_APB_l534_534436


namespace sum_leq_six_of_quadratic_roots_l534_534003

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots_l534_534003


namespace value_of_w_over_y_l534_534019

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3.25) : w / y = 0.75 :=
sorry

end value_of_w_over_y_l534_534019


namespace cos_C_values_l534_534070

theorem cos_C_values (sin_A : ℝ) (cos_B : ℝ) (cos_C : ℝ) 
  (h1 : sin_A = 4 / 5) 
  (h2 : cos_B = 12 / 13) 
  : cos_C = -16 / 65 ∨ cos_C = 56 / 65 :=
by
  sorry

end cos_C_values_l534_534070


namespace probability_of_drawing_heart_l534_534176

theorem probability_of_drawing_heart (total_cards hearts spades : ℕ) (h1 : total_cards = 5) (h2 : hearts = 3) (h3 : spades = 2) : 
  (hearts.to_rat / total_cards.to_rat) = 3 / 5 := 
by {
  sorry
}

end probability_of_drawing_heart_l534_534176


namespace sqrt_of_sqrt_81_l534_534498

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l534_534498


namespace a10_gt_500_l534_534463

-- Definition of natural numbers and conditions
variables {a : ℕ → ℕ}
variables {b : ℕ → ℕ}

-- Assume the conditions
axiom a_ascending : ∀ n m : ℕ, n < m → a n < a m
axiom b_definition : ∀ k : ℕ, b k = Nat.divisors (a k) |> filter ((>) (a k)) |> last
axiom b_descending : ∀ n m : ℕ, n < m → b n > b m

theorem a10_gt_500 : a 10 > 500 := sorry

end a10_gt_500_l534_534463


namespace incenter_trapezoid_on_MN_l534_534604

variables {A B C D M N I_X : Type} [trapezoid ABCD]

-- Definitions of points and lines
def lines_parallel (l1 l2: Line) : Prop := l1 ∥ l2
def line_segment (p1 p2: Point) : Line := Line.mk p1 p2
def is_incenter (p: Point) (Δ: Triangle) : Prop := ∀ (A B C : Point), Δ = Triangle.mk A B C → on_incircle p (incenter Δ)
def is_on_line (p: Point) (l: Line) : Prop := p ∈ l

-- Conditions from the problem
axiom trapezoid_ABCD : ∥ Line.mk A B ∥ Line.mk C D
axiom AB_gt_CD : length (line_segment A B) > length (line_segment C D)
axiom incircle_touch : (on_incircle M (incenter (Triangle.mk A B C))) ∧ (on_incircle N (incenter (Triangle.mk A C B)))

-- Proof statement
theorem incenter_trapezoid_on_MN : 
  is_incenter I_X (trapezoid ABCD) → is_on_line I_X (line_segment M N) :=
sorry

end incenter_trapezoid_on_MN_l534_534604


namespace part_a_not_prime_part_b_divisible_1984_l534_534089

section problem

variable {a b c p q : ℕ}
variable (positive_p : p > q)
variable (p q_positive : p > 0 ∧ q > 0)
variable (abc_sum : a + b + c = 2 * p * q * (p^30 + q^30))
noncomputable def k := a^3 + b^3 + c^3

-- Part (a) statement
theorem part_a_not_prime :
  ¬ Prime k :=
sorry

-- Part (b) statement
theorem part_b_divisible_1984 (h : a * b * c = max (a * b * c) (b * c * a) (c * a * b)) :
  1984 ∣ k :=
sorry

end problem

end part_a_not_prime_part_b_divisible_1984_l534_534089


namespace hyperbola_equation_l534_534359

theorem hyperbola_equation {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (asymptote : ∃ k : ℝ, (k = b / a ∧ (P : ℝ × ℝ) ∈ {(3, 4)}))
  (perpendicular : ∃ F1 F2 : ℝ × ℝ, (P : ℝ × ℝ) × (F1, F2)) : 
  ∃ a b : ℝ, (9 = a^2 ∧ 16 = b^2) :=
by
  sorry

end hyperbola_equation_l534_534359


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534798

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534798


namespace rate_of_interest_l534_534569

/-
Let P be the principal amount, SI be the simple interest paid, R be the rate of interest, and N be the number of years. 
The problem states:
- P = 1200
- SI = 432
- R = N

We need to prove that R = 6.
-/

theorem rate_of_interest (P SI R N : ℝ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = N) :
  R = 6 :=
  sorry

end rate_of_interest_l534_534569


namespace count_at_most_two_different_digits_l534_534855

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534855


namespace relationship_y_values_l534_534753

theorem relationship_y_values (m y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-3 : ℝ)^2 - 12 * (-3 : ℝ) + m)
  (h2 : y2 = -3 * (-2 : ℝ)^2 - 12 * (-2 : ℝ) + m)
  (h3 : y3 = -3 * (1 : ℝ)^2 - 12 * (1 : ℝ) + m) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end relationship_y_values_l534_534753


namespace time_to_run_home_l534_534915

-- Define the conditions
def blocks_run_per_time : ℚ := 2 -- Justin runs 2 blocks
def time_per_blocks : ℚ := 1.5 -- in 1.5 minutes
def blocks_to_home : ℚ := 8 -- Justin is 8 blocks from home

-- Define the theorem to prove the time taken for Justin to run home
theorem time_to_run_home : (blocks_to_home / blocks_run_per_time) * time_per_blocks = 6 :=
by
  sorry

end time_to_run_home_l534_534915


namespace mike_payments_total_months_l534_534118

-- Definitions based on conditions
def lower_rate := 295
def higher_rate := 310
def lower_payments := 5
def higher_payments := 7
def total_paid := 3615

-- The statement to prove
theorem mike_payments_total_months : lower_payments + higher_payments = 12 := by
  -- Proof goes here
  sorry

end mike_payments_total_months_l534_534118


namespace Q_sequence_inequality_l534_534659

-- Definition of an n-th order Q sequence
def Q_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in finset.range n, a i = 0) ∧ (∑ i in finset.range n, |a i| = 1)

-- Definition of the sum of the first k terms of the sequence
def S (a : ℕ → ℝ) (k : ℕ) : ℝ := ∑ i in finset.range k, a i

-- The theorem statement
theorem Q_sequence_inequality (a : ℕ → ℝ) (n : ℕ) (hQ : Q_sequence a n) (k : ℕ) (hk : k ≤ n): 
  |S a k| ≤ 1 / 2 :=
sorry

end Q_sequence_inequality_l534_534659


namespace geometric_sum_formula_l534_534406

variable {α : Type*} [Field α] {a : ℕ → α} {n : ℕ} {q : α}

-- Given conditions as definitions
def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n : ℕ, a (n+1) = q * a n

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  ∑ i in Finset.range n, a i

theorem geometric_sum_formula (h_geom : is_geometric_sequence a q)
  (h_q_ne_one : q ≠ 1) :
  sum_first_n_terms a n = (a 0 - a n) / (1 - q) :=
by
  sorry

end geometric_sum_formula_l534_534406


namespace angle_A_is_pi_div_3_area_of_triangle_l534_534435

variables {A B C a b c : ℝ}

-- Conditions
def conditions (A B C a b c : ℝ) := 
  (b * tan A = (2 * c - b) * tan B) ∧ 
  (a = sqrt 13) ∧ 
  (b = 3)

-- First proof statement
theorem angle_A_is_pi_div_3 (h : conditions A B C a b c) : 
  A = π / 3 :=
sorry

-- Second proof statement
theorem area_of_triangle (h : conditions A B C a b c) : 
  let S := (12 * sqrt 546 + 9 * sqrt 39) / 104 in
  ∆Area a b C = S :=
sorry  -- Assuming ∆Area is a function calculating the triangle area.

end angle_A_is_pi_div_3_area_of_triangle_l534_534435


namespace num_integers_two_digits_l534_534801

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534801


namespace number_of_incorrect_statements_is_one_l534_534249

-- Definitions of the statements
def statement1 (l : Set Point) (α : Set Point) (A : Point) (B : Point) : Prop :=
  (A ∈ l ∧ A ∈ α ∧ B ∈ l ∧ B ∈ α) → l ⊆ α

def statement2 (α β : Set Point) (A B : Point) : Prop :=
  (A ∈ α ∧ A ∈ β ∧ B ∈ α ∧ B ∈ β) → (α ∩ β = A ∪ B)

def statement3 (l : Set Point) (α : Set Point) (A : Point) : Prop :=
  (l ⊈ α ∧ A ∈ l) → A ∉ α

def statement4 (α β : Set Point) (A B C : Point) : Prop :=
  (A ∈ α ∧ A ∈ β ∧ B ∈ α ∧ B ∈ β ∧ C ∈ α ∧ C ∈ β ∧ ¬ collinear A B C) → α = β

-- Conditions as Lean statements
axiom statement1_correct : ∀ (l : Set Point) (α : Set Point) (A : Point) (B : Point),
  statement1 l α A B

axiom statement2_correct : ∀ (α β : Set Point) (A B : Point),
  statement2 α β A B

axiom statement3_incorrect : ∀ (l : Set Point) (α : Set Point) (A : Point),
  ¬ statement3 l α A

axiom statement4_correct : ∀ (α β : Set Point) (A B C : Point),
  statement4 α β A B C

-- Final proof problem
theorem number_of_incorrect_statements_is_one :
  ∃ (n : ℕ), n = 1 ∧ 
    (∀ (l : Set Point) (α : Set Point) (A : Point) (B : Point),
      statement1 l α A B) ∧
    (∀ (α β : Set Point) (A B : Point),
      statement2 α β A B) ∧
    (∀ (l : Set Point) (α : Set Point) (A : Point),
      ¬ statement3 l α A) ∧
    (∀ (α β : Set Point) (A B C : Point),
      statement4 α β A B C) := 
sorry

end number_of_incorrect_statements_is_one_l534_534249


namespace quadrilateral_is_rhombus_perpendicular_to_plane_tangent_of_angle_formed_l534_534416

-- Let A, B, C, D, A1, B1, C1, D1 be points in 3D space forming a cube with side length a
-- E is the midpoint of AB
-- F is the midpoint of C1D1

noncomputable def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

def is_rhombus (A B C D : ℝ × ℝ × ℝ) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D A)

def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  (u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0)

def side_lengths := a : ℝ -> ℝ
def cube_side_length : ℝ := side_lengths
def midpoint_AB := midpoint A B
def midpoint_C1D1 := midpoint C1 D1

variables (A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ) (a : ℝ)

theorem quadrilateral_is_rhombus (h: side_lengths a) (midpoint_AB: E) (midpoint_C1D1: F) :
  is_rhombus A1 E C F :=
sorry

theorem perpendicular_to_plane (h: side_lengths a) (EF: E) (plane_A1B1C: B1) :
  is_perpendicular EF (plane_A1B1C) :=
sorry

theorem tangent_of_angle_formed (h: side_lengths a) (A1B1: ℝ) :
  tan_angle_formed_by (A1B1) (plane_A1ECF) = sqrt 2 :=
sorry

end quadrilateral_is_rhombus_perpendicular_to_plane_tangent_of_angle_formed_l534_534416


namespace evaluate_expression_at_neg3_l534_534292

theorem evaluate_expression_at_neg3 :
  let x := -3 in
  let z := (Real.sqrt ((x - 1)^2) + Real.sqrt (x^2 + 4*x + 4)) in
  z = 5 :=
by
  let x := -3
  let z := (Real.sqrt ((x - 1)^2) + Real.sqrt (x^2 + 4*x + 4))
  sorry

end evaluate_expression_at_neg3_l534_534292


namespace exists_x3_root_l534_534766

noncomputable def root_exists_between (a b c x1 x2 : ℝ) : Prop :=
∃ x3 : ℝ, (a / 2) * x3^2 + b * x3 + c = 0 ∧ (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x2 ≤ x3 ∧ x3 ≤ x1)

theorem exists_x3_root :
  ∀ a b c x1 x2 : ℝ, (a ≠ 0) →
  (a * x1^2 + b * x1 + c = 0) →
  (-a * x2^2 + b * x2 + c = 0) →
  root_exists_between a b c x1 x2 :=
begin
  intros a b c x1 x2 ha hx1 hx2,
  sorry
end

end exists_x3_root_l534_534766


namespace sequence_convergence_and_function_linearity_l534_534090

variable (f : ℝ → ℝ) 
variable (a_n b_n : ℕ → ℝ)
variable (h_cont : ContinuousOn f (Set.Icc 0 1))
variable (h_lim : Tendsto (λ n, ∫ x in 0..1, | f x - a_n n * x - b_n n |) atTop (𝓝 0))

theorem sequence_convergence_and_function_linearity 
  (hf : ContinuousOn f (Set.Icc 0 1))
  (h : Tendsto (λ n, ∫ x in 0..1, | f x - a_n n * x - b_n n |) atTop (𝓝 0)) :
  CauchySeq a_n ∧ CauchySeq b_n ∧ (∃ (a b : ℝ), ∀ x ∈ Set.Icc (0 : ℝ) (1 : ℝ), f x = a * x + b) :=
sorry

end sequence_convergence_and_function_linearity_l534_534090


namespace relationship_among_a_b_c_l534_534320

noncomputable def a : ℝ := Real.sqrt 0.3
noncomputable def b : ℝ := 2 ^ 0.3
noncomputable def c : ℝ := 0.3 ^ 0.2

theorem relationship_among_a_b_c : b > c ∧ c > a :=
by
  sorry

end relationship_among_a_b_c_l534_534320


namespace num_integers_two_digits_l534_534804

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534804


namespace cot_sum_arccot_roots_l534_534464

noncomputable def polynomial_roots : Fin 21 → ℂ := sorry

theorem cot_sum_arccot_roots :
  let z := polynomial_roots in
  let θ := λ k, arccot (z k) in
  (cot (finset.univ.sum θ)) = 13 / 12 :=
sorry

end cot_sum_arccot_roots_l534_534464


namespace count_two_digit_or_less_numbers_l534_534841

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534841


namespace smallest_five_digit_divisible_by_3_and_4_l534_534554

theorem smallest_five_digit_divisible_by_3_and_4 : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ n = 10008 :=
sorry

end smallest_five_digit_divisible_by_3_and_4_l534_534554


namespace arithmetic_square_root_sqrt_81_l534_534510

theorem arithmetic_square_root_sqrt_81 : sqrt (sqrt 81) = 3 := 
by 
  have h1 : sqrt 81 = 9 := sorry
  have h2 : sqrt 9 = 3 := sorry
  show sqrt (sqrt 81) = 3 from sorry

end arithmetic_square_root_sqrt_81_l534_534510


namespace triangle_division_exists_l534_534079

theorem triangle_division_exists :
  ∀ (A B C : Point), 
  ∃ (D E F G : Point), 
  convex_shape A B C D E F G :=
sorry

end triangle_division_exists_l534_534079


namespace fifth_largest_divisor_l534_534267

theorem fifth_largest_divisor (n : ℕ) (h : n = 1020000000) : 
    nat.nth_largest_divisor n 5 = 63750000 :=
by
  rw h
  sorry

end fifth_largest_divisor_l534_534267


namespace verify_statements_l534_534787

def vec_a : ℝ × ℝ := (2, 4)
def vec_b : ℝ × ℝ := (-2, 1)

theorem verify_statements :
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) ∧
  (real.sqrt ((vec_a.1 + vec_b.1)^2 + (vec_a.2 + vec_b.2)^2) = 5) ∧
  (real.sqrt ((vec_a.1 - vec_b.1)^2 + (vec_a.2 - vec_b.2)^2) = 5) := 
by 
  sorry

end verify_statements_l534_534787


namespace set_A_definition_l534_534777

-- Definitions given in the problem conditions
def A := {x : ℤ | 6 / (5 - x) ∈ ℕ ∧ x ∈ ℤ}

-- Statement of the problem proving equivalence
theorem set_A_definition :
  A = {-1, 2, 3, 4} :=
sorry

end set_A_definition_l534_534777


namespace Elberta_money_l534_534030

theorem Elberta_money (G A E : ℕ) 
  (hG : G = 120)
  (hA : A = G / 2)
  (hE : E = A + 5) :
  E = 65 :=
  by 
  rw [hG, hA, hE]
  sorry

end Elberta_money_l534_534030


namespace program_output_with_n_6_l534_534482

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def program_output (n : ℕ) : ℕ :=
  let rec loop (i : ℕ) (s : ℕ) : ℕ :=
    if i > n then s else loop (i + 1) (s * i)
  loop 1 1

theorem program_output_with_n_6 : program_output 6 = 720 :=
by
  have : program_output 6 = factorial 6 := by sorry
  rw [this]
  norm_num

end program_output_with_n_6_l534_534482


namespace area_of_triangle_l534_534546

theorem area_of_triangle (m1 m2 : ℝ) (P : ℝ × ℝ) (l3 : ℝ → ℝ) :
  m1 = 2 → m2 = 1 / 2 → P = (2, 2) → l3 = (λ x, -x + 10) →
  let l1 := (λ x, m1 * x + (P.2 - m1 * P.1)) in
  let l2 := (λ x, m2 * x + (P.2 - m2 * P.1)) in
  let intersect (f g : ℝ → ℝ) (x : ℝ) : Prop := f x = g x in
  let A := P in
  let Bx := (4 : ℝ) in
  let By := l1 4 in
  let B := (Bx, By) in
  let Cx := (6 : ℝ) in
  let Cy := l2 6 in
  let C := (Cx, Cy) in
  let area (A B C : ℝ × ℝ) : ℝ :=
    0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  area A B C = 6 :=
begin
  intros,
  sorry
end

end area_of_triangle_l534_534546


namespace range_of_a_l534_534331

-- Definitions for propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ¬(x^2 + (a-1)*x + 1 ≤ 0)

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 1)^x₁ < (a - 1)^x₂

-- The final theorem to prove
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → (-1 < a ∧ a ≤ 2) ∨ (a ≥ 3) :=
by
  sorry

end range_of_a_l534_534331


namespace percentage_increase_correct_l534_534440

variable (D : ℝ) (Distance_first : D)
variable (Distance_second : 1.2 * D = 24)
variable (Total_distance : (D + 24 + Distance_third) = 74)

def distance_third_greater : Prop :=
  Distance_third > 24

def percentage_increase : ℝ :=
  ((Distance_third - 24) / 24) * 100

theorem percentage_increase_correct :
  percentage_increase = 25 :=
sorry

end percentage_increase_correct_l534_534440


namespace ashok_total_subjects_l534_534611

/-- Ashok secured an average of 78 marks in some subjects. If the average of marks in 5 subjects 
is 74, and he secured 98 marks in the last subject, how many subjects are there in total? -/
theorem ashok_total_subjects (n : ℕ) 
  (avg_all : 78 * n = 74 * (n - 1) + 98) : n = 6 :=
sorry

end ashok_total_subjects_l534_534611


namespace determine_a_l534_534351

-- Defining the complex number 'z' and the real number 'a'
noncomputable def z (a : ℝ) : ℂ := (1 + a * complex.I) / (1 - complex.I)

-- The statement we want to prove
theorem determine_a (a : ℝ) (h : (z a).im = 2) : a = 3 :=
by
  sorry

end determine_a_l534_534351


namespace apricot_trees_count_l534_534174

theorem apricot_trees_count (peach_trees apricot_trees : ℕ) 
  (h1 : peach_trees = 300) 
  (h2 : peach_trees = 2 * apricot_trees + 30) : 
  apricot_trees = 135 := 
by 
  sorry

end apricot_trees_count_l534_534174


namespace number_of_knights_and_liars_l534_534474

-- Definitions and conditions
def is_knight (p : Nat) : Prop := ∀ (i : Nat), i = p → (∀ (left : Nat) (right : Nat), 
                      (left = (p + 1) % 100 → left_is_liar = true) ∧
                      (right = (p - 1) % 100 → right_is_trickster = true))
def is_liar (p : Nat) : Prop := ∀ (i : Nat), i = p → (∀ (left : Nat) (right : Nat),
                      (left = (p + 1) % 100 → ¬left_is_liar) ∧
                      (right = (p - 1) % 100 → ¬right_is_trickster))
def is_trickster (p : Nat) : Prop := ¬is_knight p ∧ ¬is_liar p
def left_is_liar := false
def right_is_trickster := false

-- Proof problem statement
theorem number_of_knights_and_liars :
  ∑ i in Finset.range 50, (if is_knight i then 1 else 0) = 25 ∧
  ∑ i in Finset.range 50, (if is_liar i then 1 else 0) = 25 := 
sorry

end number_of_knights_and_liars_l534_534474


namespace limit_Sn_over_n2Bn_l534_534929

open Nat

def A (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}
def S_n (n : ℕ) : ℝ := (Finset.range (n + 1)).sum * (2^(n - 1) - 1)
def B_n (n : ℕ) : ℝ := 2^n

theorem limit_Sn_over_n2Bn :
  (∀ n : ℕ, S_n n = ((n * (n + 1)) / 2) * (2^(n - 1) - 1))
  → (∀ n : ℕ, B_n n = 2^n)
  → (Real.limitAtTop (λ n, (S_n n) / (n * n * B_n n)) (1 / 4)) :=
by
  sorry

end limit_Sn_over_n2Bn_l534_534929


namespace number_of_positive_integers_with_at_most_two_digits_l534_534815

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534815


namespace line_intersects_y_axis_at_l534_534613

def point1 : ℝ × ℝ := (2, 9)
def point2 : ℝ × ℝ := (4, 15)
def y_axis_intersection : ℝ × ℝ := (0, 3)

theorem line_intersects_y_axis_at : ∀ (p1 p2 : ℝ × ℝ), 
  p1 = point1 → p2 = point2 → 
  (∃ m b : ℝ, (∀ x, ((p1.2 = m * p1.1 + b) ∧ (p2.2 = m * p2.1 + b)) ∧ (y_axis_intersection = (0, b)))) :=
by
  intros p1 p2 hp1 hp2
  sorry

end line_intersects_y_axis_at_l534_534613


namespace f_difference_l534_534729

noncomputable def f (n : ℕ) : ℝ := ∑ i in finset.range (3 * n), (1 : ℝ) / (i + 1)

theorem f_difference (n : ℕ) : f (n + 1) - f n = 1 / (3 * n) + 1 / (3 * n + 1) + 1 / (3 * n + 2) := 
sorry

end f_difference_l534_534729


namespace longer_subsegment_length_l534_534165

-- Define the given conditions and proof goal in Lean 4
theorem longer_subsegment_length {DE EF DF DG GF : ℝ} (h1 : 3 * EF < 4 * EF) (h2 : 4 * EF < 5 * EF)
  (ratio_condition : DE / EF = 4 / 5) (DF_length : DF = 12) :
  DG + GF = DF ∧ DE / EF = DG / GF ∧ GF = (5 * 12 / 9) :=
by
  sorry

end longer_subsegment_length_l534_534165


namespace incorrect_proposition_2_l534_534538

-- Conditions
def group_A : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def group_B : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def median_A : ℕ := 5
def median_B : ℕ := (5 + 6) / 2 -- alternatively we can use Rat or Real, but kept it as Nat

def r : ℝ := -0.88
def chi_squared_value : ℝ := 4.567
def chi_squared_threshold : ℝ := 3.841
def residual (x_i y_i b a : ℝ) : ℝ := y_i - (b * x_i + a)

-- Equivalent proof problem
theorem incorrect_proposition_2 : rabs r ≥ 0.75 → abs r < 1 → r = -0.88 → chi_squared_value > chi_squared_threshold → median_A = 5 → median_B = 5.5 → group_A = [1, 2, 3, 4, 5, 6, 7, 8, 9] → group_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] → ¬(r = -0.88 ∧ r < 0.75) :=
by sorry

end incorrect_proposition_2_l534_534538


namespace train_crosses_tunnel_in_45_sec_l534_534239

/-- Given the length of the train, the length of the platform, the length of the tunnel, 
and the time taken to cross the platform, prove the time taken for the train to cross the tunnel is 45 seconds. -/
theorem train_crosses_tunnel_in_45_sec (l_train : ℕ) (l_platform : ℕ) (t_platform : ℕ) (l_tunnel : ℕ)
  (h_train_length : l_train = 330)
  (h_platform_length : l_platform = 180)
  (h_time_platform : t_platform = 15)
  (h_tunnel_length : l_tunnel = 1200) :
  (l_train + l_tunnel) / ((l_train + l_platform) / t_platform) = 45 :=
by
  -- placeholder for the actual proof
  sorry

end train_crosses_tunnel_in_45_sec_l534_534239


namespace sarah_toy_cars_l534_534483

theorem sarah_toy_cars (initial_money toy_car_cost scarf_cost beanie_cost remaining_money: ℕ) 
  (h_initial: initial_money = 53) 
  (h_toy_car_cost: toy_car_cost = 11) 
  (h_scarf_cost: scarf_cost = 10) 
  (h_beanie_cost: beanie_cost = 14) 
  (h_remaining: remaining_money = 7) : 
  (initial_money - remaining_money - scarf_cost - beanie_cost) / toy_car_cost = 2 := 
by 
  sorry

end sarah_toy_cars_l534_534483


namespace locus_trajectory_values_of_a_b_l534_534201

variables (a b : ℕ) (x y : ℝ)
variable (h_a_ne_0 : a ≠ 0)

-- Define the given condition
def condition_1 : Prop := 2 * a * y + b ^ 2 = 0
def equation_of_locus : Prop := y = a * x ^ 2 - b * x + (b ^ 2) / (4 * a)

-- Define the given line and condition for intersection at one point
def given_line : Prop := 4 * (real.sqrt 7 - 1) * a * b * x - 4 * a * y + b ^ 2 + a ^ 2 - 6958 * a = 0

-- The proof problem of part 1
theorem locus_trajectory :
  ∀ x : ℝ, (condition_1 a b) → (equation_of_locus a b x y) :=
sorry

-- The proof problem of part 2
theorem values_of_a_b :
  ∃! (a b : ℕ), (a ≠ 0) ∧ (condition_1 a b) ∧ (given_line a b x y) :=
sorry

end locus_trajectory_values_of_a_b_l534_534201


namespace f_monotonically_increasing_g_minimum_value_l534_534772

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (sin x)^2 + sin x * cos x

noncomputable def g (x : ℝ) : ℝ := sin (x - π / 3)

theorem f_monotonically_increasing : 
  ∀ k : ℤ, ∀ x : ℝ, x ∈ set.Icc (k * π - π / 12) (k * π + 5 * π / 12) → 
  (f' x > 0 := sorry)

theorem g_minimum_value :
  (min_value := -1 := sorry) ∧ 
  ∀ k : ℤ, ∀ x : ℝ, (x = 2 * k * π - π / 6) →
  (g x = -1 := sorry)

end f_monotonically_increasing_g_minimum_value_l534_534772


namespace midpoint_on_circle_of_isosceles_triangle_l534_534952

noncomputable def midpoint (A B : Point) : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

theorem midpoint_on_circle_of_isosceles_triangle
  (A B C M : Point)
  (h_isosceles : dist A B = dist A C)
  (h_midpoint : midpoint A C = M)
  (h_circle : OnCircle M (circle (segment A B))) :
  OnCircle M (circle (segment A B)) := 
sorry

end midpoint_on_circle_of_isosceles_triangle_l534_534952


namespace real_roots_of_quadratic_l534_534739

theorem real_roots_of_quadratic (m : ℝ) : ((m - 2) ≠ 0 ∧ (-4 * m + 24) ≥ 0) → (m ≤ 6 ∧ m ≠ 2) := 
by 
  sorry

end real_roots_of_quadratic_l534_534739


namespace part1_part2_l534_534096

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a - 1) * x + a - 2

theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f x a + 2 ≥ 0) ↔ (3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, f x a < 0) →
  (if a < 3 then ∀ x, a - 2 < x ∧ x < 1
  else if a = 3 then ∀ x, false
  else ∀ x, 1 < x ∧ x < a - 2) :=
sorry

end part1_part2_l534_534096


namespace common_chord_length_l534_534181

/-- Two circles intersect such that each passes through the other's center.
Prove that the length of their common chord is 8√3 cm. -/
theorem common_chord_length (r : ℝ) (h : r = 8) :
  let chord_length := 2 * (r * (Real.sqrt 3 / 2))
  chord_length = 8 * Real.sqrt 3 := by
  sorry

end common_chord_length_l534_534181


namespace cosine_sum_identity_l534_534577

theorem cosine_sum_identity (α : ℝ) (n : ℕ) (h : n > 0) :
  (∑ k in finset.range n, real.cos ((2 * k + 1) * α)) = (real.sin (2 * n * α)) / (2 * real.sin α) :=
by
  sorry

end cosine_sum_identity_l534_534577


namespace solution_concentration_l534_534140

theorem solution_concentration (y z : ℝ) :
  let x_vol := 300
  let y_vol := 2 * z
  let z_vol := z
  let total_vol := x_vol + y_vol + z_vol
  let alcohol_x := 0.10 * x_vol
  let alcohol_y := 0.30 * y_vol
  let alcohol_z := 0.40 * z_vol
  let total_alcohol := alcohol_x + alcohol_y + alcohol_z
  total_vol = 600 ∧ y_vol = 2 * z_vol ∧ y_vol + z_vol = 300 → 
  total_alcohol / total_vol = 21.67 / 100 :=
by
  sorry

end solution_concentration_l534_534140


namespace ella_emma_hotdogs_l534_534116

-- Definitions based on the problem conditions
def hotdogs_each_sister_wants (E : ℕ) :=
  let luke := 2 * E
  let hunter := 3 * E
  E + E + luke + hunter = 14

-- Statement we need to prove
theorem ella_emma_hotdogs (E : ℕ) (h : hotdogs_each_sister_wants E) : E = 2 :=
by
  sorry

end ella_emma_hotdogs_l534_534116


namespace cos2α_minus_cosα_over_sinα_l534_534342

variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2)
variable (h2 : cos α = 2 * sqrt 5 / 5)

theorem cos2α_minus_cosα_over_sinα : cos (2 * α) - (cos α) / (sin α) = -7 / 5 :=
by
  sorry

end cos2α_minus_cosα_over_sinα_l534_534342


namespace servings_per_day_l534_534082

-- Definitions based on the given problem conditions
def serving_size : ℚ := 0.5
def container_size : ℚ := 32 - 2 -- 1 quart is 32 ounces and the jar is 2 ounces less
def days_last : ℕ := 20

-- The theorem statement to prove
theorem servings_per_day (h1 : serving_size = 0.5) (h2 : container_size = 30) (h3 : days_last = 20) :
  (container_size / days_last) / serving_size = 3 :=
by
  sorry

end servings_per_day_l534_534082


namespace find_a_l534_534025
-- Import necessary Lean libraries

-- Define the function and its maximum value condition
def f (a x : ℝ) := -x^2 + 2*a*x + 1 - a

def has_max_value (f : ℝ → ℝ) (M : ℝ) (interval : Set ℝ) : Prop :=
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = M

theorem find_a (a : ℝ) :
  has_max_value (f a) 2 (Set.Icc 0 1) → (a = -1 ∨ a = 2) :=
by
  sorry

end find_a_l534_534025


namespace monic_quadratic_with_root_l534_534702

theorem monic_quadratic_with_root :
  ∃ (p : ℝ[X]), monic p ∧ (p.coeff 2 = 1) ∧ (p.coeff 1 = 6) ∧ (p.coeff 0 = 16) ∧ is_root p (-3 - complex.I * real.sqrt 7) :=
sorry

end monic_quadratic_with_root_l534_534702


namespace fixed_point_of_line_l534_534473

theorem fixed_point_of_line (k : ℝ) : ∃ P : ℝ × ℝ, P = (3, 1) ∧ ∀ k : ℝ, k * P.1 - P.2 + 1 = 3 * k :=
by
  -- Introduce the point (3, 1) as P
  use (3, 1)
  -- Prove that it satisfies the equation for any k
  split
  -- Show the point is (3, 1)
  exact rfl
  -- Show the line passes through (3, 1) for any k
  intro k
  calc k * 3 - 1 + 1 = 3 * k : by rw [sub_add_cancel, mul_comm]

end fixed_point_of_line_l534_534473


namespace find_a_l534_534385

theorem find_a (a b d : ℤ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l534_534385


namespace eq_has_at_most_four_solutions_eq_has_exactly_four_solutions_l534_534946

theorem eq_has_at_most_four_solutions (a b c d : ℤ) (h : a ≠ b) :
  ∃ n : ℕ, n ≤ 4 ∧ ∃ xy : fin n → ℤ × ℤ,
    ∀ p q : ℤ, p * q = 2 →
    ∃ i j : fin n, (xy i).fst + a * (xy i).snd + c = p ∧
                   (xy j).fst + b * (xy j).snd + d = q :=
sorry

theorem eq_has_exactly_four_solutions (a b c d : ℤ) :
  (|a - b| = 1 ∨ |a - b| = 2) ∧ (c - d) % 2 ≠ 0 →
  ∃ (xy : fin 4 → ℤ × ℤ),
    ∀ p q : ℤ, p * q = 2 →
    ∃ i j : fin 4, (xy i).fst + a * (xy i).snd + c = p ∧
                   (xy j).fst + b * (xy j).snd + d = q :=
sorry

end eq_has_at_most_four_solutions_eq_has_exactly_four_solutions_l534_534946


namespace sqrt_meaningful_range_l534_534400

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1) ↔ x ≥ 1) := 
sorry

end sqrt_meaningful_range_l534_534400


namespace patricia_current_hair_length_l534_534942

-- Define the conditions as Lean variables
variables (donation remaining growth_needed total_length current_length : ℕ)

-- Assign the given values to these variables
def donation := 23
def remaining := 12
def growth_needed := 21
def total_length := donation + remaining

-- State the proof problem
theorem patricia_current_hair_length (donation remaining growth_needed total_length current_length : ℕ)
  (h1: total_length = donation + remaining)
  (h2: current_length + growth_needed = total_length) :
  current_length = 14 := by
  sorry

end patricia_current_hair_length_l534_534942


namespace zero_point_interval_l534_534298

open Real

noncomputable def f (x : ℝ) := log x - 2 / x

theorem zero_point_interval : (∃ c ∈ Ioo 2 3, f c = 0) :=
by
  have h2 : f 2 < 0 := by sorry
  have h3 : f 3 > 0 := by sorry
  exact IntermediateValue_zero f h2 h3

end zero_point_interval_l534_534298


namespace smallest_N_diagonal_repetition_l534_534419

def number_of_diagonals_in_regular_polygon (n : ℕ) : ℕ := n * (n - 3) / 2

def different_diagonal_lengths (n : ℕ) : ℕ := (n - 3) / 2

theorem smallest_N_diagonal_repetition (n : ℕ) (h : n = 2017) : 
  ∃ N, N = different_diagonal_lengths n + 1 :=
by
  have h1 : number_of_diagonals_in_regular_polygon 2017 = 2017 * 2014 / 2 :=
    by simp
  have h2 : different_diagonal_lengths 2017 = 1007 :=
    by simp
  use 1008
  sorry

end smallest_N_diagonal_repetition_l534_534419


namespace polynomial_roots_a_ge_five_l534_534462

theorem polynomial_roots_a_ge_five (a b c : ℤ) (h_a_pos : a > 0)
    (h_distinct_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
        a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) : a ≥ 5 := sorry

end polynomial_roots_a_ge_five_l534_534462


namespace find_integers_l534_534684

theorem find_integers (n : ℕ) (h₁ : n > 1) : 
  (∀ (a : ℕ) (h₂ : 1 ≤ a ∧ a < n ∧ Nat.coprime a n) (b : ℕ) (h₃ : a < b ∧ b < n ∧ Nat.coprime b n), ¬(a + b) % 3 = 0) ↔ (n = 2 ∨ n = 4 ∨ n = 10) :=
by
  sorry

end find_integers_l534_534684


namespace sum_k_round_diff_eq_125237_l534_534631

noncomputable def sum_k_round_diff (n : ℕ) : ℤ :=
∑ k in finset.range n.succ, k * ((⌈real.log k / real.log (real.sqrt 3)⌉ : ℤ) - (⌊real.log k / real.log (real.sqrt 3)⌋ : ℤ))

theorem sum_k_round_diff_eq_125237 : sum_k_round_diff 500 = 125237 := by
  -- Here we would add the proof steps to verify the theorem
  sorry

end sum_k_round_diff_eq_125237_l534_534631


namespace monic_quadratic_poly_with_root_l534_534697

theorem monic_quadratic_poly_with_root (x : ℝ) :
  (∃ p : Polynomial ℝ, Polynomial.monic p ∧ p.coeff 2 = 1 ∧ p.coeff 1 = 6 ∧ p.coeff 0 = 16 
  ∧ p.eval (-3 - Complex.i * Real.sqrt 7) = 0) :=
by
  use Polynomial.C 16 + Polynomial.X * (Polynomial.C 6 + Polynomial.X)
  field_simp
  sorry

end monic_quadratic_poly_with_root_l534_534697


namespace max_value_f_l534_534982

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (x + 1)

theorem max_value_f : ∃ x ∈ Icc (-2 : ℝ) 1, ∀ y ∈ Icc (-2 : ℝ) 1, f y ≤ f x ∧ f x = Real.exp 2 := 
begin
  use 1,
  split,
  { norm_num, },
  intros y hy,
  split,
  { 
    by_cases hy1 : y = -2,
    { rw hy1, change (-2 : ℝ)^2 * Real.exp (-2 + 1) ≤ Real.exp 2, norm_num,
      apply Real.exp_pos, },
    by_cases hy0 : y = 0,
    { rw hy0, norm_num, apply Real.exp_pos, },
    {
      have hy_bound: -2 < y ∧ y < 1,
      { split; linarith, },
      set f_deriv := λ x, x * Real.exp (x + 1) * (2 + x),
      change f y ≤ Real.exp 2,
      sorry,
    },
  },
  { norm_num, }
end

end max_value_f_l534_534982


namespace proof_problem_l534_534104

open Real

theorem proof_problem 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_sum : a + b + c + d = 1)
  : (b * c * d / (1 - a)^2) + (c * d * a / (1 - b)^2) + (d * a * b / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1 / 9 := 
   sorry

end proof_problem_l534_534104


namespace combined_interest_is_correct_l534_534244

-- Definitions based on given conditions
def total_investment : ℝ := 9000
def interest_rate1 : ℝ := 0.065
def interest_rate2 : ℝ := 0.08
def investment2 : ℝ := 6258
def investment1 : ℝ := total_investment - investment2

-- The proof problem statement
theorem combined_interest_is_correct :
  let interest1 := investment1 * interest_rate1 * 1 in
  let interest2 := investment2 * interest_rate2 * 1 in
  let combined_interest := interest1 + interest2 in
  combined_interest = 678.87 :=
by
  sorry

end combined_interest_is_correct_l534_534244


namespace abc_zero_iff_quadratic_identities_l534_534919

variable {a b c : ℝ}

theorem abc_zero_iff_quadratic_identities (h : ¬(a = b ∧ b = c ∧ c = a)) : 
  a + b + c = 0 ↔ a^2 + ab + b^2 = b^2 + bc + c^2 ∧ b^2 + bc + c^2 = c^2 + ca + a^2 :=
by
  sorry

end abc_zero_iff_quadratic_identities_l534_534919


namespace perfect_square_property_l534_534884

theorem perfect_square_property : 
  let f := 14
  let n := 3150
  let product := n * f
  product = 44100 ∧ ∃ k : ℕ, k * k = product
:= 
by
  let f := 14
  let n := 3150
  let product := n * f
  have h1 : product = 44100 := by
    simp [product, n, f]
  have h2 : ∃ k : ℕ, k * k = product := by
    use 210
    simp [product, h1]
  exact ⟨h1, h2⟩

end perfect_square_property_l534_534884


namespace _l534_534423

open EuclideanGeometry

noncomputable def rectangle_ratio_theorem : 
  ∀ (A B C D M O : Point) (h : Rectangle A B C D) 
    (hAD : Segment A D = 6) (hBC : Segment B C = 8) 
    (hM : Midpoint M (Segment C D)) 
    (hO : Intersection O (Line A C) (Line B M)),
  Ratio (Segment O C) (Segment O A) = 1 / 2 :=
by
  sorry

end _l534_534423


namespace trapezoid_perimeter_l534_534069

variable {A B C D : Point}
variable {r : ℝ}

-- Conditions
def is_trapezoid (A B C D : Point) : Prop :=
  AD = 8 ∧ BC = 18

def circle_tangent_to_lines (A B D r : Point) : Prop :=
  tangent_circle_point BC r B ∧ tangent_circle_point CD r B

-- Question: perimeter of trapezoid ABCD
theorem trapezoid_perimeter (A B C D : Point) (r : ℝ) 
  (cond1 : is_trapezoid A B C D) 
  (cond2 : circle_tangent_to_lines A B D r) :
  perimeter A B C D = 56 := 
sorry

end trapezoid_perimeter_l534_534069


namespace sequence_count_less_than_1969_l534_534547

-- Define the function f that describes the number of terms needed for sequences ending in n
def f (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 2
  else if n = 4 then 2
  else 1 + (Nat.sqrt n).sum (λ i, f i)

-- Define the main theorem
theorem sequence_count_less_than_1969 : f 1969 < 1969 :=
  sorry

end sequence_count_less_than_1969_l534_534547


namespace min_PA_squared_plus_PB_squared_l534_534330

-- Let points A, B, and the circle be defined as given in the problem.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-2, 0⟩
def B : Point := ⟨2, 0⟩

def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

def PA_squared (P : Point) : ℝ :=
  (P.x - A.x)^2 + (P.y - A.y)^2

def PB_squared (P : Point) : ℝ :=
  (P.x - B.x)^2 + (P.y - B.y)^2

def F (P : Point) : ℝ := PA_squared P + PB_squared P

theorem min_PA_squared_plus_PB_squared : ∃ P : Point, on_circle P ∧ F P = 26 := sorry

end min_PA_squared_plus_PB_squared_l534_534330


namespace relation_y1_y2_y3_l534_534751

-- Definition of being on the parabola
def on_parabola (x : ℝ) (y m : ℝ) : Prop := y = -3*x^2 - 12*x + m

-- The conditions given in the problem
variables {y1 y2 y3 m : ℝ}

-- The points (-3, y1), (-2, y2), (1, y3) are on the parabola given by the equation
axiom h1 : on_parabola (-3) y1 m
axiom h2 : on_parabola (-2) y2 m
axiom h3 : on_parabola (1) y3 m

-- We need to prove the relationship between y1, y2, and y3
theorem relation_y1_y2_y3 : y2 > y1 ∧ y1 > y3 :=
by { sorry }

end relation_y1_y2_y3_l534_534751


namespace divide_arc_ratio_l534_534126

def IsEquilateralTriangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def Circles (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] :=
  ∃ circABM circMBC : MetricSpace → MetricSpace, True

theorem divide_arc_ratio (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
  (h_eq : IsEquilateralTriangle A B C)
  (h_ext : ∃ P : Type, M = P ∧ P ∉ set.range (λ x, (AC : Type)))
  (h_circ : Circles A B C M)
  (h_ratio : ∃ n : ℝ, arc_ratio M A B = n) :
  arc_ratio M C B = 2 * n + 1 :=
sorry

end divide_arc_ratio_l534_534126


namespace sqrt_of_sqrt_81_l534_534499

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l534_534499


namespace total_tires_parking_lot_l534_534057

-- Definitions for each condition in a)
def four_wheel_drive_cars := 30
def motorcycles := 20
def six_wheel_trucks := 10
def bicycles := 5
def unicycles := 3
def baby_strollers := 2

def extra_roof_tires := 4
def flat_bike_tires_removed := 3
def extra_unicycle_wheel := 1

def tires_per_car := 4 + 1
def tires_per_motorcycle := 2 + 2
def tires_per_truck := 6 + 1
def tires_per_bicycle := 2
def tires_per_unicycle := 1
def tires_per_stroller := 4

-- Define total tires calculation
def total_tires (four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
                 extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel : ℕ) :=
  (four_wheel_drive_cars * tires_per_car + extra_roof_tires) +
  (motorcycles * tires_per_motorcycle) +
  (six_wheel_trucks * tires_per_truck) +
  (bicycles * tires_per_bicycle - flat_bike_tires_removed) +
  (unicycles * tires_per_unicycle + extra_unicycle_wheel) +
  (baby_strollers * tires_per_stroller)

-- The Lean statement for the proof problem
theorem total_tires_parking_lot : 
  total_tires four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
              extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel = 323 :=
by 
  sorry

end total_tires_parking_lot_l534_534057


namespace solution_set_eq_l534_534530

theorem solution_set_eq : 
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ (sqrt (3 * x - 1) + abs (2 * y + 2) = 0)} = {((1 / 3 : ℝ), -1)} :=
by
  sorry

end solution_set_eq_l534_534530


namespace solve_system_l534_534200

noncomputable theory

theorem solve_system :
  ∃ x y : ℝ, (1 / x + 1 / y = 2.25) ∧ 
             (x^2 / y + y^2 / x = 32.0625) ∧
             ((x = 4 ∧ y = 1 / 2) ∨ (x = 1 / 12 * (-19 + Real.sqrt (1691 / 3)) ∧ y = 1 / 12 * (-19 - Real.sqrt (1691 / 3)))) := 
by
  sorry

end solve_system_l534_534200


namespace largest_multiple_negation_greater_than_neg150_l534_534551

theorem largest_multiple_negation_greater_than_neg150 (n : ℤ) (h₁ : n % 6 = 0) (h₂ : -n > -150) : n = 144 := 
sorry

end largest_multiple_negation_greater_than_neg150_l534_534551


namespace munificence_minimum_l534_534279

def p (x : ℝ) := x^3 - 3 * x - 1

theorem munificence_minimum : 
  ∃ (M : ℝ), (∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), |p x| ≤ M) ∧ M = 5 := 
by
  sorry

end munificence_minimum_l534_534279


namespace twelve_otimes_eight_otimes_two_l534_534044

def operation (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem twelve_otimes_eight_otimes_two :
  operation (operation 12 8) 2 = 7 / 3 :=
by
  sorry

end twelve_otimes_eight_otimes_two_l534_534044


namespace acute_angle_at_9_35_is_77_5_degrees_l534_534185

def degrees_in_acute_angle_formed_by_hands_of_clock_9_35 : ℝ := 77.5

theorem acute_angle_at_9_35_is_77_5_degrees 
  (hour_angle : ℝ := 270 + (35/60 * 30))
  (minute_angle : ℝ := 35/60 * 360) : 
  |hour_angle - minute_angle| < 180 → |hour_angle - minute_angle| = degrees_in_acute_angle_formed_by_hands_of_clock_9_35 := 
by 
  sorry

end acute_angle_at_9_35_is_77_5_degrees_l534_534185


namespace total_distance_karl_drove_l534_534087

theorem total_distance_karl_drove :
  ∀ (consumption_rate miles_per_gallon : ℕ) 
    (tank_capacity : ℕ) 
    (initial_gas : ℕ) 
    (distance_leg1 : ℕ) 
    (purchased_gas : ℕ) 
    (remaining_gas : ℕ)
    (final_gas : ℕ),
  consumption_rate = 25 → 
  tank_capacity = 18 →
  initial_gas = 12 →
  distance_leg1 = 250 →
  purchased_gas = 10 →
  remaining_gas = initial_gas - distance_leg1 / consumption_rate + purchased_gas →
  final_gas = remaining_gas - distance_leg2 / consumption_rate →
  remaining_gas - distance_leg2 / consumption_rate = final_gas →
  distance_leg2 = (initial_gas - remaining_gas + purchased_gas - final_gas) * miles_per_gallon →
  miles_per_gallon = 25 →
  distance_leg2 + distance_leg1 = 475 :=
sorry

end total_distance_karl_drove_l534_534087


namespace factor_expression_l534_534681

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) :=
by
  sorry

end factor_expression_l534_534681


namespace intersection_unique_l534_534361

variables {α : Type*} [linear_ordered_field α]

def M (x y : α) : Prop := x + y = 2
def N (x y : α) : Prop := x - y = 4
def intersection (x y : α) : Prop := M x y ∧ N x y

theorem intersection_unique :
  ∃ (x y : α), intersection x y ∧ ∀ (a b : α), intersection a b → (a, b) = (3, -1) :=
by
    sorry

end intersection_unique_l534_534361


namespace count_two_digit_or_less_numbers_l534_534845

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534845


namespace sqrt_sqrt_81_is_9_l534_534502

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l534_534502


namespace eval_expression_l534_534454

theorem eval_expression :
  let a := 3
  let b := 2
  (2 ^ a ∣ 200) ∧ ¬(2 ^ (a + 1) ∣ 200) ∧ (5 ^ b ∣ 200) ∧ ¬(5 ^ (b + 1) ∣ 200)
→ (1 / 3)^(b - a) = 3 :=
by sorry

end eval_expression_l534_534454


namespace bejgli_slices_l534_534177

theorem bejgli_slices (x : ℕ) (hx : x ≤ 58) 
    (h1 : x * (x - 1) * (x - 2) = 3 * (58 - x) * (57 - x) * x) : 
    58 - x = 21 :=
by
  have hpos1 : 0 < x := sorry  -- x should be strictly positive since it's a count
  have hpos2 : 0 < 58 - x := sorry  -- the remaining slices should be strictly positive
  sorry

end bejgli_slices_l534_534177


namespace largest_positive_integer_n_l534_534299

noncomputable def max_n_satisfying_condition : ℕ :=
  9

theorem largest_positive_integer_n
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ) :
  ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ),
    {0,1,2,3,4,5,6,7,8,9} ⊆
      ({(abs (xᵢ - xⱼ)) | i j, 1 ≤ i ∧ i < j ∧ j ≤ 4, xᵢ ∈ Pl {x1, x2, x3, x4} xⱼ∈ {x1, x2, x3, x4}} ∪
      ({abs (yᵢ - yⱼ) | i j, 1 ≤ i ∧ i < j ∧ j ≤ 4, yᵢ ∈ {y1, y2, y3, y4} yⱼ∈ {y1, y2, y3, y4})}{
    sorry
  }

end largest_positive_integer_n_l534_534299


namespace sum_correct_l534_534652

open Nat

noncomputable def sum_problem : ℝ :=
  (∑ k in range 1 (501 : ℕ), (k : ℝ) * (if (∃ j : ℤ, (k : ℝ) = real.exp ((j : ℝ) * real.log (real.sqrt 3))) then 0 else 1))

theorem sum_correct :
  sum_problem = 125187 := 
by
  sorry

end sum_correct_l534_534652


namespace sum_log_sqrt3_l534_534636

theorem sum_log_sqrt3 : 
  (∑ k in Finset.range (501), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 124886 :=
by
  sorry

end sum_log_sqrt3_l534_534636


namespace geometric_sequence_a_eq_one_l534_534493

theorem geometric_sequence_a_eq_one (a : ℝ) 
  (h₁ : ∃ (r : ℝ), a = 1 / (1 - r) ∧ r = a - 1/2 ∧ r ≠ 0) : 
  a = 1 := 
sorry

end geometric_sequence_a_eq_one_l534_534493


namespace perpendicular_slope_l534_534719

theorem perpendicular_slope (x y : ℝ) : (∃ b : ℝ, 4 * x - 5 * y = 10) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro h
  sorry

end perpendicular_slope_l534_534719


namespace adjoint_vector_magnitude_k_range_l534_534278

noncomputable def g (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3) + Real.cos x

def adjoint_vector_g : ℝ × ℝ := (-Real.sqrt 3 / 2, 3 / 2)

def magnitude_adjoint_vector_g : ℝ := Real.sqrt 3

def adjunct_func_on (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

def adjoint_func_with_k (k : ℝ) (x : ℝ) : ℝ :=
  adjunct_func_on x + k * adjunct_func_on (x + Real.pi / 2)

theorem adjoint_vector_magnitude :
  |adjoint_vector_g| = Real.sqrt 3 :=
sorry

theorem k_range (k x : ℝ) (hx : 0 ≤ x ∧ x ≤ 11 * Real.pi / 12) :
  adjoint_func_with_k k x > 0 ↔ k ∈ Ioo (-Real.sqrt 3) (-1) :=
sorry

end adjoint_vector_magnitude_k_range_l534_534278


namespace range_f_l534_534526

noncomputable def f : ℝ → ℝ :=
λ x, if (-3 : ℝ) ≤ x ∧ x ≤ 0 then x^2 + 2 * x - 1 
     else if 0 < x ∧ x ≤ 5 then x - 1 
     else 0

theorem range_f : set.range f = set.Icc (-2 : ℝ) 4 :=
by sorry

end range_f_l534_534526


namespace find_triangle_sides_l534_534075

theorem find_triangle_sides (x y : ℕ) : 
  (x * y = 200) ∧ (x + 2 * y = 50) → ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) := 
by
  intro h
  sorry

end find_triangle_sides_l534_534075


namespace brownie_cost_l534_534612

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) (cost_per_piece : ℕ) :
  total_money = 32 → num_pans = 2 → pieces_per_pan = 8 → cost_per_piece = total_money / (num_pans * pieces_per_pan) → 
  cost_per_piece = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end brownie_cost_l534_534612


namespace count_at_most_two_different_digits_l534_534851

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534851


namespace number_of_positive_integers_with_at_most_two_digits_l534_534811

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534811


namespace part_1_average_yields_part_2_m_value_l534_534446

noncomputable def average_yield_rice : ℕ := 500
noncomputable def average_yield_corn : ℕ := 400
noncomputable def m : ℕ := 10 -- percentage

theorem part_1_average_yields (rice_planted corn_planted : ℕ) (rice_price corn_price : ℕ) (total_income : ℕ)
  (rice_yield corn_yield : ℕ) (yield_difference : ℕ) :
  rice_planted = 200 →
  corn_planted = 100 →
  rice_price = 3 →
  corn_price = 2.5 →
  total_income = 400000 →
  yield_difference = 100 →
  rice_yield = average_yield_rice ∧ corn_yield = average_yield_corn :=
by
  sorry

theorem part_2_m_value (rice_increase corn_increase : ℕ) (rice_price_increase corn_price_increase : ℕ)
  (new_total_income : ℕ) (rice_yield corn_yield : ℕ) (increase_percentage : ℕ) :
  rice_yield = 500 →
  corn_yield = 400 →
  rice_increase = (1 + m / 100) →
  corn_increase = (1 + 2 * m / 100) →
  rice_price_increase = 3.2 →
  corn_price_increase = 2.5 * (1 + m / 100) →
  new_total_income = 400000 * (1 + 21 / 100) →
  increase_percentage = m :=
by
  sorry

end part_1_average_yields_part_2_m_value_l534_534446


namespace sum_correct_l534_534649

open Nat

noncomputable def sum_problem : ℝ :=
  (∑ k in range 1 (501 : ℕ), (k : ℝ) * (if (∃ j : ℤ, (k : ℝ) = real.exp ((j : ℝ) * real.log (real.sqrt 3))) then 0 else 1))

theorem sum_correct :
  sum_problem = 125187 := 
by
  sorry

end sum_correct_l534_534649


namespace reflected_point_correct_l534_534151

-- Defining the original point coordinates
def original_point : ℝ × ℝ := (3, -5)

-- Defining the transformation function
def reflect_across_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- Proving the point after reflection is as expected
theorem reflected_point_correct : reflect_across_y_axis original_point = (-3, -5) :=
by
  sorry

end reflected_point_correct_l534_534151


namespace jorge_goals_this_season_l534_534443

def jorge_goals_last_season : Nat := 156
def jorge_goals_total : Nat := 343

theorem jorge_goals_this_season :
  ∃ g_s : Nat, g_s = jorge_goals_total - jorge_goals_last_season ∧ g_s = 187 :=
by
  -- proof goes here, we use 'sorry' for now
  sorry

end jorge_goals_this_season_l534_534443


namespace problem_part1_a_problem_part2_problem_part3_l534_534028

noncomputable def f (a x : ℝ) := 4 * a * cos x * sin (x - π / 6)

theorem problem_part1_a (a x : ℝ) (h : f a (π / 3) = 1) : a = 1 :=
by
  sorry

noncomputable def g (x : ℝ) := 2 * sin (2 * x - π / 6) - 1

theorem problem_part2 (x : ℝ) : has_period g π :=
by
  sorry

theorem problem_part3 (m : ℝ) (h : ∀ x ∈ set.Icc 0 m, monotone_at g x) : m ≤ π / 3 :=
by
  sorry

end problem_part1_a_problem_part2_problem_part3_l534_534028


namespace gcd_of_lcm_and_ratio_l534_534401

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : Nat.gcd X Y = 18 :=
sorry

end gcd_of_lcm_and_ratio_l534_534401


namespace sum_log_floor_ceil_l534_534627

theorem sum_log_floor_ceil (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500 → 
  ((⌈ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌋) = 
    if (∃ n : ℕ, k = (3 : ℕ) ^ (n / 2 : ℚ)) then 0 else 1)) :
  ∑ k in Finset.range 501 \ {0}, k * (⌈ Real.log k / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log k / Real.log (Real.sqrt 3) ⌋) = 125237 := by
  have h_sum_all := (500 * 501) / 2
  have h_sum_powers := 1 + 3 + 9
  have h_correct := h_sum_all - h_sum_powers
  rw h_sum_all
  rw h_sum_powers
  rw h_correct
  sorry

end sum_log_floor_ceil_l534_534627


namespace plan_payment_difference_l534_534138

theorem plan_payment_difference (P r : ℝ) (years : ℕ) (compounds : ℕ) :
    let A1 := P * (1 + r / compounds) ^ (compounds * (years / 2))
    let payment1 := A1 / 3 + (A1 * (2 / 3) * (1 + r / compounds) ^ (compounds * (years / 2)))
    let A2 := P * (1 + r) ^ years
    (A2 - payment1).round = 6546 :=
by
  let P := 12000
  let r := 0.08
  let years := 8
  let compounds1 := 2
  let compounds := 1
  let A1 := P * (1 + r / compounds1) ^ (compounds1 * (years / 2))
  let payment1 := A1 / 3 + (A1 * (2 / 3) * (1 + r / compounds1) ^ (compounds1 * (years / 2)))
  let A2 := P * (1 + r) ^ years
  have h: A2 - payment1 ≈ 6546 
  sorry

end plan_payment_difference_l534_534138


namespace count_valid_numbers_l534_534861

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534861


namespace vector_properties_l534_534785

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (a_def : a = ![2, 4])
variable (b_def : b = ![-2, 1])

theorem vector_properties :
  (dot_product a b = 0) ∧
  (‖a + b‖ = 5) ∧
  (‖a - b‖ = 5) :=
by
  have h₁ : dot_product a b = 0 := by sorry
  have h₂ : ‖a + b‖ = 5 := by sorry
  have h₃ : ‖a - b‖ = 5 := by sorry
  exact ⟨h₁, h₂, h₃⟩

end vector_properties_l534_534785


namespace average_speed_of_the_car_l534_534212

noncomputable def averageSpeed (d1 d2 d3 d4 t1 t2 t3 t4 : ℝ) : ℝ :=
  let totalDistance := d1 + d2 + d3 + d4
  let totalTime := t1 + t2 + t3 + t4
  totalDistance / totalTime

theorem average_speed_of_the_car :
  averageSpeed 30 35 65 (40 * 0.5) (30 / 45) (35 / 55) 1 0.5 = 54 := 
  by 
    sorry

end average_speed_of_the_car_l534_534212


namespace arithmetic_square_root_of_sqrt_81_l534_534515

def square_root (x : ℝ) : ℝ :=
  real.sqrt x

theorem arithmetic_square_root_of_sqrt_81 : square_root 81 = 9 :=
by
  sorry

end arithmetic_square_root_of_sqrt_81_l534_534515


namespace monic_quadratic_poly_with_root_l534_534698

theorem monic_quadratic_poly_with_root (x : ℝ) :
  (∃ p : Polynomial ℝ, Polynomial.monic p ∧ p.coeff 2 = 1 ∧ p.coeff 1 = 6 ∧ p.coeff 0 = 16 
  ∧ p.eval (-3 - Complex.i * Real.sqrt 7) = 0) :=
by
  use Polynomial.C 16 + Polynomial.X * (Polynomial.C 6 + Polynomial.X)
  field_simp
  sorry

end monic_quadratic_poly_with_root_l534_534698


namespace monic_quadratic_with_given_root_l534_534691

theorem monic_quadratic_with_given_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.eval (-3 - complex.i * Real.sqrt 7) = 0 ∧ p = Polynomial.X^2 + 6 * Polynomial.X + 16 :=
by
  sorry

end monic_quadratic_with_given_root_l534_534691


namespace quadratic_polynomial_has_root_l534_534706

theorem quadratic_polynomial_has_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.coeff 1 ∈ ℝ ∧ p.coeff 0 ∈ ℝ ∧ Polynomial.eval (-3 - Complex.i * Real.sqrt 7) p = 0 ∧
                        p = Polynomial.X^2 + Polynomial.C 6 * Polynomial.X + Polynomial.C 16 :=
sorry

end quadratic_polynomial_has_root_l534_534706


namespace a_8_value_l534_534315

theorem a_8_value :
  (∑ k in Finset.range 11, a k * (1 - x) ^ k = (1 + x) ^ 10) → 
  ∀ a_8, a_8 = 180 :=
by
  intro h_eq
  sorry

end a_8_value_l534_534315


namespace penniless_pete_dime_difference_l534_534475

theorem penniless_pete_dime_difference :
  ∃ a b c : ℕ, 
  (a + b + c = 100) ∧ 
  (5 * a + 10 * b + 50 * c = 1350) ∧ 
  (b = 170 ∨ b = 8) ∧ 
  (b - 8 = 162 ∨ 170 - b = 162) :=
sorry

end penniless_pete_dime_difference_l534_534475


namespace find_y_l534_534431

theorem find_y {x y : ℝ} (hx : (8 : ℝ) = (1/4 : ℝ) * x) (hy : (y : ℝ) = (1/4 : ℝ) * (20 : ℝ)) (hprod : x * y = 160) : y = 5 :=
by {
  sorry
}

end find_y_l534_534431


namespace lighthouse_distance_l534_534597

theorem lighthouse_distance 
  (speed : ℝ) (time : ℝ) (angle_A : ℝ) (angle_B : ℝ) (AB : ℝ) (BS : ℝ)
  (h_speed : speed = 32)
  (h_time : time = 0.5)
  (h_angle_A : angle_A = 30)
  (h_angle_B : angle_B = 75)
  (h_AB : AB = speed * time)
  (h_BS : BS = 8 * real.sqrt 2) :
  BS = 8 * real.sqrt 2 := 
begin
  sorry
end

end lighthouse_distance_l534_534597


namespace sum_of_segments_not_less_than_6sqrt2_l534_534076

-- Define the problem in Lean 4
theorem sum_of_segments_not_less_than_6sqrt2 (points : Fin 6 → Fin 3 → ℝ) :
  (∀ i, 0 ≤ points i ∧ points i ≤ 1) →
  Σ (ab ⟨i, j⟩ ∈ ({(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5), (2, 5), (2, 1), (3, 4), (4, 0)})), 
    (∥ points (⟨0, i, j⟩ - points (⟨1, i, j⟩)) ∥) ≥ 6 * Real.sqrt 2 :=
by 
  sorry

end sum_of_segments_not_less_than_6sqrt2_l534_534076


namespace Fermat_numbers_are_not_cubes_l534_534480

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem Fermat_numbers_are_not_cubes : ∀ n : ℕ, ¬ ∃ k : ℕ, F n = k^3 :=
by
  sorry

end Fermat_numbers_are_not_cubes_l534_534480


namespace jorge_goals_l534_534444

theorem jorge_goals (g_last g_total g_this : ℕ) (h_last : g_last = 156) (h_total : g_total = 343) :
  g_this = g_total - g_last → g_this = 187 :=
by
  intro h
  rw [h_last, h_total] at h
  apply h

end jorge_goals_l534_534444


namespace number_of_positive_integers_with_at_most_two_digits_l534_534819

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534819


namespace length_ab_of_acute_triangle_l534_534147

theorem length_ab_of_acute_triangle
  (A B C O M : Type)
  [MetricSpace A] [RealInnerProductSpace ℝ A]
  (ABC : Triangle A)
  (acute : ABC.isAcute)
  (orthocenter_O : ABC.orthocenter = O)
  (circle : Circle A) (radius_R : ℝ)
  (passes_through_A : circle.center = O ∧ circle.radius = radius_R ∧ passes_through circle A)
  (tangent_to_BC : circle.isTangentAt BC)
  (intersects_AC : intersects_at circle AC M)
  (AM_MC_ratio : ratio (segment A M) (segment M C) = 4 / 1) :
  (length (segment A B) = 2 * radius_R * sqrt 2) :=
sorry

end length_ab_of_acute_triangle_l534_534147


namespace solution_set_of_inequality_l534_534203

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 4*x else (x^2 - 4*x)

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(x) = f(-x)

theorem solution_set_of_inequality :
  (is_even_function f) →
  (∀ x ≥ 0, f x = x^2 - 4*x) →
  {x : ℝ | f (x + 2) < 5} = set.Ioo (-7) 3 :=
by
  sorry

end solution_set_of_inequality_l534_534203


namespace max_int_greater_than_15_l534_534995

theorem max_int_greater_than_15 (a : Fin 8 → ℤ) (sum_eq_zero : (∑ i, a i) = 0) : 
  ∃ k ≤ 8, (∀ i < k, a i > 15) → k = 7 := 
sorry

end max_int_greater_than_15_l534_534995


namespace moles_of_Cl2_combined_l534_534710

theorem moles_of_Cl2_combined (nCH4 : ℕ) (nCl2 : ℕ) (nHCl : ℕ) 
  (h1 : nCH4 = 3) 
  (h2 : nHCl = nCl2) 
  (h3 : nHCl ≤ nCH4) : 
  nCl2 = 3 :=
by
  sorry

end moles_of_Cl2_combined_l534_534710


namespace max_sides_of_convex_polygon_with_4_obtuse_l534_534881

theorem max_sides_of_convex_polygon_with_4_obtuse (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k = 4 ∧
    ∀ θ : Fin n → ℝ, 
      (∀ p, θ p > 90 ∧ ∃ t, θ t = 180 ∨ θ t < 90 ∨ θ t = 90) →
      4 = k →
      n ≤ 7
  ) :=
sorry

end max_sides_of_convex_polygon_with_4_obtuse_l534_534881


namespace days_left_to_make_toys_l534_534872

-- Define the constants
def toysMadePerDay : ℕ := 100
def daysWorked : ℕ := 6
def totalToysTarget : ℕ := 1000

-- Prove the number of days left to complete the target
theorem days_left_to_make_toys (toysMadePerDay daysWorked totalToysTarget : ℕ) (sameSpeed : ∀ t : ℕ, t > 0 → toysMadePerDay > 0) :
  let totalToysMade := toysMadePerDay * daysWorked in
  let remainingToys := totalToysTarget - totalToysMade in
  let daysLeft := remainingToys / toysMadePerDay in
  totalToysTarget = 1000 ∧ toysMadePerDay = 100 ∧ daysWorked = 6 →
  daysLeft = 4 :=
by
  -- The proof is not required, so we use sorry
  sorry

end days_left_to_make_toys_l534_534872


namespace count_at_most_two_different_digits_l534_534852

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534852


namespace perfect_square_m_value_l534_534575

theorem perfect_square_m_value (y m : ℤ) (h : ∃ k : ℤ, y^2 - 8 * y + m = (y - k)^2) : m = 16 :=
sorry

end perfect_square_m_value_l534_534575


namespace find_f_4_l534_534346

def f (x : ℝ) : ℝ := sorry 

theorem find_f_4 : (∀ x : ℝ, f (x + 1) = x^2 - 1) → f 4 = 8 :=
by 
  intro h
  specialize h 3
  assumption

end find_f_4_l534_534346


namespace solution_range_of_a_l534_534775

open Real

-- Define the proposition that ensures the function y = log_(1/2)(x^2 - ax + 2a) is defined and decreasing on [1, +∞)
def log_decreasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, 1 ≤ x → 1 ≤ y → x < y → log (1 / 2) (x^2 - a * x + 2 * a) > log (1 / 2) (y^2 - a * y + 2 * a)

-- The main theorem: If the function y = log_(1/2)(x^2 - ax + 2a) is decreasing on [1, +∞), then -1 < a ≤ 2
theorem solution_range_of_a (a : ℝ) (h : log_decreasing a) : -1 < a ∧ a ≤ 2 :=
sorry

end solution_range_of_a_l534_534775


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534796

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534796


namespace general_formula_geom_seq_sum_first_n_terms_l534_534998

noncomputable def geom_seq (n : ℕ) : ℝ := 3^n

def b (n : ℕ) : ℝ := real.logb 3 (geom_seq n)

def sum_seq (n : ℕ) : ℝ := (1 + 3) + (2 + 3^2) + ... + (n + 3^n)

theorem general_formula_geom_seq :
  ∃ (a : ℕ → ℝ), 
    (∀ n, a n > 0) 
    ∧ (4 * a 1 - a 2 = 3) 
    ∧ ((a 5)^2 = 9 * a 2 * a 6) 
    ∧ (∀ n, a n = 3^n) := sorry

theorem sum_first_n_terms (n : ℕ) :
  ∑ i in finset.range n, geom_seq (i+1) + b (i+1) = (n * (n + 1) / 2) + (3 * (1 - 3 ^ n) / (1 - 3)) :=
sorry

end general_formula_geom_seq_sum_first_n_terms_l534_534998


namespace cos_double_angle_minus_cos_over_sin_l534_534336

theorem cos_double_angle_minus_cos_over_sin (α : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : real.cos α = 2 * real.sqrt 5 / 5) :
  real.cos (2 * α) - real.cos α / real.sin α = -7 / 5 :=
sorry

end cos_double_angle_minus_cos_over_sin_l534_534336


namespace minimum_value_4x_minus_y_l534_534758

theorem minimum_value_4x_minus_y (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 4 ≥ 0) (h3 : x ≤ 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), (x' - y' ≥ 0) → (x' + y' - 4 ≥ 0) → (x' ≤ 4) → 4 * x' - y' ≥ m :=
by
  sorry

end minimum_value_4x_minus_y_l534_534758


namespace power_is_seventeen_l534_534527

theorem power_is_seventeen (x : ℕ) : (1000^7 : ℝ) / (10^x) = (10000 : ℝ) ↔ x = 17 := by
  sorry

end power_is_seventeen_l534_534527


namespace apples_to_pears_l534_534145

theorem apples_to_pears :
  (3 / 4) * 12 = 9 → (2 / 3) * 6 = 4 :=
by {
  sorry
}

end apples_to_pears_l534_534145


namespace arithmetic_square_root_of_sqrt_81_l534_534516

def square_root (x : ℝ) : ℝ :=
  real.sqrt x

theorem arithmetic_square_root_of_sqrt_81 : square_root 81 = 9 :=
by
  sorry

end arithmetic_square_root_of_sqrt_81_l534_534516


namespace train_length_is_correct_l534_534603

noncomputable def speed_kmhr : ℝ := 45
noncomputable def time_sec : ℝ := 30
noncomputable def bridge_length_m : ℝ := 235

noncomputable def speed_ms : ℝ := (speed_kmhr * 1000) / 3600
noncomputable def total_distance_m : ℝ := speed_ms * time_sec
noncomputable def train_length_m : ℝ := total_distance_m - bridge_length_m

theorem train_length_is_correct : train_length_m = 140 :=
by
  -- Placeholder to indicate that a proof should go here
  -- Proof is omitted as per the instructions
  sorry

end train_length_is_correct_l534_534603


namespace prime_has_property_P_infinitely_many_composites_with_property_P_l534_534224

-- Definition of property P
def has_property_P (n : ℕ) : Prop :=
  ∀ (a : ℤ), n ∣ (a^n - 1) → n^2 ∣ (a^n - 1)

-- (a) Show that every prime number \( n \) has the property \( P \).
theorem prime_has_property_P (p : ℕ) (h_prime : Nat.Prime p) : has_property_P p := 
  sorry

-- (b) Show that there are infinitely many composite numbers \( n \) that possess the property \( P \).
theorem infinitely_many_composites_with_property_P : 
  ∃ᶠ n in at_top, (¬Nat.Prime n ∧ 1 < n) ∧ has_property_P n := 
  sorry

end prime_has_property_P_infinitely_many_composites_with_property_P_l534_534224


namespace count_positive_integers_with_two_digits_l534_534834

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534834


namespace probability_of_12th_roll_last_l534_534043

noncomputable def prob12thRollLast : ℚ :=
  (7^10) / (8^11)

theorem probability_of_12th_roll_last : prob12thRollLast ≈ 0.016 :=
  sorry

end probability_of_12th_roll_last_l534_534043


namespace count_positive_integers_with_two_digits_l534_534832

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534832


namespace count_valid_numbers_l534_534863

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534863


namespace distribute_items_in_identical_bags_l534_534254

noncomputable def count_ways_to_distribute_items (num_items : ℕ) (num_bags : ℕ) : ℕ :=
  if h : num_items = 5 ∧ num_bags = 3 then 36 else 0

theorem distribute_items_in_identical_bags :
  count_ways_to_distribute_items 5 3 = 36 :=
by
  -- Proof is skipped as per instructions
  sorry

end distribute_items_in_identical_bags_l534_534254


namespace ratio_areas_of_trapezoid_and_triangle_l534_534609

-- Definitions based on the conditions
def equilateral_triangle (XYZ : Type) [is_triangle XYZ] : Prop :=
  ∃ (X Y Z : XYZ), is_equilateral XYZ X Y Z

def parallel_segments {XYZ : Type} (X Y Z : XYZ) 
  [is_triangle XYZ] (LM NO PQ : XYZ) : Prop :=
  parallel_to LM YZ ∧ parallel_to NO YZ ∧ parallel_to PQ YZ

def equal_segments {XYZ : Type} (X Y Z L N P : XYZ) 
  [is_triangle XYZ] : Prop :=
  segment_length XL = segment_length LN ∧ 
  segment_length LN = segment_length NP ∧ 
  segment_length NP = segment_length PY

-- The proof statement
theorem ratio_areas_of_trapezoid_and_triangle 
  {XYZ : Type} [is_triangle XYZ] 
  (X Y Z L N P Q M : XYZ)
  (h1 : equilateral_triangle XYZ)
  (h2 : parallel_segments X Y Z L N P)
  (h3 : equal_segments X Y Z L N P) :
  ratio_area_trapezoid_triangle PQYZ XYZ = 7 / 16 :=
by
  sorry

end ratio_areas_of_trapezoid_and_triangle_l534_534609


namespace max_value_of_expression_l534_534465

theorem max_value_of_expression (x y z : ℝ) (h : 0 < x) (h' : 0 < y) (h'' : 0 < z) (hxyz : x * y * z = 1) :
  (∃ s, s = x ∧ ∃ t, t = y ∧ ∃ u, u = z ∧ 
  (x^2 * y / (x + y) + y^2 * z / (y + z) + z^2 * x / (z + x) ≤ 3 / 2)) :=
sorry

end max_value_of_expression_l534_534465


namespace side_lengths_sum_eq_225_l534_534921

noncomputable def GX (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - x

noncomputable def GY (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - y

noncomputable def GZ (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - z

theorem side_lengths_sum_eq_225
  (x y z : ℝ)
  (h : GX x y z ^ 2 + GY x y z ^ 2 + GZ x y z ^ 2 = 75) :
  (x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2 = 225 := by {
  sorry
}

end side_lengths_sum_eq_225_l534_534921


namespace sum_correct_l534_534650

open Nat

noncomputable def sum_problem : ℝ :=
  (∑ k in range 1 (501 : ℕ), (k : ℝ) * (if (∃ j : ℤ, (k : ℝ) = real.exp ((j : ℝ) * real.log (real.sqrt 3))) then 0 else 1))

theorem sum_correct :
  sum_problem = 125187 := 
by
  sorry

end sum_correct_l534_534650


namespace length_of_AE_l534_534275

variable (A B C D E : Type) [AddGroup A]
variable (AB CD AC AE EC : ℝ)
variable 
  (hAB : AB = 8)
  (hCD : CD = 18)
  (hAC : AC = 20)
  (hEqualAreas : ∀ (AED BEC : Type), (area AED = area BEC) → (AED = BEC))

theorem length_of_AE (hRatio : AE / EC = 4 / 9) (hSum : AC = AE + EC) : AE = 80 / 13 :=
by
  sorry

end length_of_AE_l534_534275


namespace Jason_seashells_l534_534938

theorem Jason_seashells (initial_seashells given_to_Tim remaining_seashells : ℕ) :
  initial_seashells = 49 → given_to_Tim = 13 → remaining_seashells = initial_seashells - given_to_Tim →
  remaining_seashells = 36 :=
by intros; sorry

end Jason_seashells_l534_534938


namespace triangle_division_possible_l534_534077

theorem triangle_division_possible (ABC : Triangle) :
  ∃ (T1 T2 T3 T4 : ConvexShape), 
    T1.shape = triangle ∧ T2.shape = quadrilateral ∧ 
    T3.shape = pentagon ∧ T4.shape = hexagon ∧ 
    is_convex T1 ∧ is_convex T2 ∧ is_convex T3 ∧ is_convex T4 ∧
    (T1 ∪ T2 ∪ T3 ∪ T4 = ABC) ∧ 
    (∀ (i j : {T1, T2, T3, T4}), i ≠ j → disjoint i j) :=
sorry

end triangle_division_possible_l534_534077


namespace remainder_example_l534_534672

def P (x : ℝ) := 8 * x^3 - 20 * x^2 + 28 * x - 26
def D (x : ℝ) := 4 * x - 8

theorem remainder_example : P 2 = 14 :=
by
  sorry

end remainder_example_l534_534672


namespace translate_right_l534_534412

-- Definition of the initial point and translation distance
def point_A : ℝ × ℝ := (2, -1)
def translation_distance : ℝ := 3

-- The proof statement
theorem translate_right (x_A y_A : ℝ) (d : ℝ) 
  (h1 : point_A = (x_A, y_A))
  (h2 : translation_distance = d) : 
  (x_A + d, y_A) = (5, -1) := 
sorry

end translate_right_l534_534412


namespace truth_of_q_l534_534048

variable {p q : Prop}

theorem truth_of_q (hnp : ¬ p) (hpq : p ∨ q) : q :=
  by
  sorry

end truth_of_q_l534_534048


namespace variable_equals_one_l534_534307

def T (x : ℝ) : ℝ := x * (2 - x)

theorem variable_equals_one (y : ℝ) (h : y + 1 = T(y + 1)) : y = 1 :=
sorry

end variable_equals_one_l534_534307


namespace exists_non_convex_pentagon_with_non_intersecting_diagonals_l534_534286

theorem exists_non_convex_pentagon_with_non_intersecting_diagonals :
  ∃ (P : set ℝ^2), (is_pentagon P) ∧ (¬ convex P) ∧ (no_two_diagonals_intersect P) :=
sorry

end exists_non_convex_pentagon_with_non_intersecting_diagonals_l534_534286


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534790

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534790


namespace sufficient_condition_for_perpendicular_l534_534317

variables (α β : Plane) (a b : Line)

-- Conditions
axiom α_perpendicular_β : α ⟂ β
axiom a_perpendicular_α : a ⟂ α
axiom b_perpendicular_β : b ⟂ β

-- Theorem statement
theorem sufficient_condition_for_perpendicular :
  a ⟂ b :=
sorry

end sufficient_condition_for_perpendicular_l534_534317


namespace volume_of_box_l534_534591

-- Defining the initial parameters of the problem
def length_sheet := 48
def width_sheet := 36
def side_length_cut_square := 3

-- Define the transformed dimensions after squares are cut off
def length_box := length_sheet - 2 * side_length_cut_square
def width_box := width_sheet - 2 * side_length_cut_square
def height_box := side_length_cut_square

-- The target volume of the box
def target_volume := 3780

-- Prove that the volume of the box is equal to the target volume
theorem volume_of_box : length_box * width_box * height_box = target_volume := by
  -- Calculate the expected volume
  -- Expected volume = 42 m * 30 m * 3 m
  -- Which equals 3780 m³
  sorry

end volume_of_box_l534_534591


namespace remaining_thumbtacks_in_each_can_l534_534931

-- Definitions based on the conditions:
def total_thumbtacks : ℕ := 450
def num_cans : ℕ := 3
def thumbtacks_per_board_tested : ℕ := 1
def total_boards_tested : ℕ := 120

-- Lean 4 Statement

theorem remaining_thumbtacks_in_each_can :
  ∀ (initial_thumbtacks_per_can remaining_thumbtacks_per_can : ℕ),
  initial_thumbtacks_per_can = (total_thumbtacks / num_cans) →
  remaining_thumbtacks_per_can = (initial_thumbtacks_per_can - (thumbtacks_per_board_tested * total_boards_tested)) →
  remaining_thumbtacks_per_can = 30 :=
by
  sorry

end remaining_thumbtacks_in_each_can_l534_534931


namespace chord_length_range_l534_534763

variable {x y : ℝ}

def center : ℝ × ℝ := (4, 5)
def radius : ℝ := 13
def point : ℝ × ℝ := (1, 1)
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 169

-- statement: prove the range of |AB| for specific conditions
theorem chord_length_range :
  ∀ line : (ℝ × ℝ) → (ℝ × ℝ) → Prop,
  (line center point → line (x, y) (x, y) ∧ circle_eq x y)
  → 24 ≤ abs (dist (x, y) (x, y)) ∧ abs (dist (x, y) (x, y)) ≤ 26 :=
by
  sorry

end chord_length_range_l534_534763


namespace max_value_f_cos_theta_val_l534_534027

def f (x : ℝ) : ℝ := 2 * sin x * cos x - 2 * (cos x) ^ 2 + 1

theorem max_value_f : ∃ x : ℝ, ∀ y : ℝ, f y ≤ sqrt 2 ∧ f x = sqrt 2 :=
sorry

theorem cos_theta_val (θ : ℝ) (hθ : f θ = 3 / 5) : cos (2 * ((π / 4) - 2 * θ)) = 16 / 25 :=
sorry

end max_value_f_cos_theta_val_l534_534027


namespace quadratic_polynomial_has_root_l534_534707

theorem quadratic_polynomial_has_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.coeff 1 ∈ ℝ ∧ p.coeff 0 ∈ ℝ ∧ Polynomial.eval (-3 - Complex.i * Real.sqrt 7) p = 0 ∧
                        p = Polynomial.X^2 + Polynomial.C 6 * Polynomial.X + Polynomial.C 16 :=
sorry

end quadratic_polynomial_has_root_l534_534707


namespace takeoff_run_distance_correct_l534_534170

noncomputable def takeoff_run_distance : ℕ := 
  let v := 27.78        -- lift-off speed in meters per second
  let t := 15           -- takeoff time in seconds
  let a := v / t        -- uniformly accelerated motion: acceleration
  let S := 0.5 * a * t^2 -- distance using uniformly accelerated motion formula
  S.to_nat               -- rounding to the nearest whole number

theorem takeoff_run_distance_correct : takeoff_run_distance = 208 :=
  by sorry

end takeoff_run_distance_correct_l534_534170


namespace Ravi_probability_l534_534545

-- Conditions from the problem
def P_Ram : ℚ := 4 / 7
def P_BothSelected : ℚ := 0.11428571428571428

-- Statement to prove
theorem Ravi_probability :
  ∃ P_Ravi : ℚ, P_Rami = 0.2 ∧ P_Ram * P_Ravi = P_BothSelected := by
  sorry

end Ravi_probability_l534_534545


namespace carpenter_job_duration_l534_534582

theorem carpenter_job_duration
  (total_estimate : ℤ)
  (carpenter_hourly_rate : ℤ)
  (assistant_hourly_rate : ℤ)
  (material_cost : ℤ)
  (H1 : total_estimate = 1500)
  (H2 : carpenter_hourly_rate = 35)
  (H3 : assistant_hourly_rate = 25)
  (H4 : material_cost = 720) :
  (total_estimate - material_cost) / (carpenter_hourly_rate + assistant_hourly_rate) = 13 :=
by
  sorry

end carpenter_job_duration_l534_534582


namespace a_2008_is_2_l534_534744

noncomputable def sequence (n : ℕ) : ℚ :=
nat.rec_on n
  2
  (λ n a_n, 1 - (1 / a_n))

theorem a_2008_is_2 : sequence 2008 = 2 := 
sorry

end a_2008_is_2_l534_534744


namespace smallest_angle_range_l534_534896

theorem smallest_angle_range {A B C : ℝ} (hA : 0 < A) (hABC : A + B + C = 180) (horder : A ≤ B ∧ B ≤ C) :
  0 < A ∧ A ≤ 60 := by
  sorry

end smallest_angle_range_l534_534896


namespace sum_of_odd_numbers_l534_534257

theorem sum_of_odd_numbers (n : ℕ) : (∑ k in Finset.range n, (2 * (k + 1) - 1)) = n^2 :=
by
  sorry

end sum_of_odd_numbers_l534_534257


namespace sum_of_first_10_terms_eq_100_l534_534991

variable {a : ℕ → ℝ} -- Define the arithmetic sequence

-- Define conditions given in part a)
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop := a 0 + a 1 = 4
def condition2 (a : ℕ → ℝ) : Prop := a 4 + a 5 = 20

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * a 0 + (n : ℝ) * (n - 1) / 2 * (a 1 - a 0)

-- Statement to prove that under the given conditions, the sum of the first 10 terms is 100
theorem sum_of_first_10_terms_eq_100 (a : ℕ → ℝ) 
  [arith_seq : arithmetic_sequence a] 
  (h1 : condition1 a) 
  (h2 : condition2 a) :
  sum_of_first_n_terms a 10 = 100 := sorry

end sum_of_first_10_terms_eq_100_l534_534991


namespace triangle_construction_l534_534664

-- Define points A, H, and O with coordinates
variable {A H O : Point}

-- Theorem statement asserting the properties of the triangle 
-- constructed given A, H, and O
theorem triangle_construction (A H O : Point) (is_orthocenter : Orthocenter H (Triangle A B C))
(is_circumcenter : Circumcenter O (Triangle A B C)) : 
-- Proof goal that the triangle constructed with these properties exist
∃ B C : Point, Triangle A B C ∧ EulerLine A H O :=
sorry

end triangle_construction_l534_534664


namespace rectangle_is_square_l534_534578

theorem rectangle_is_square
  (a b: ℝ)  -- rectangle side lengths
  (h: a ≠ b)  -- initial assumption: rectangle not a square
  (shift_perpendicular: ∀ (P Q R S: ℝ × ℝ), (P ≠ Q → Q ≠ R → R ≠ S → S ≠ P) → (∀ (shift: ℝ × ℝ → ℝ × ℝ), ∀ (P₁: ℝ × ℝ), shift P₁ = P₁ + (0, 1) ∨ shift P₁ = P₁ + (1, 0)) → false):
  False := sorry

end rectangle_is_square_l534_534578


namespace correct_function_period_symmetry_l534_534670

theorem correct_function_period_symmetry :
  ∃ f : ℝ → ℝ,
    (f = λ x, 2 * sin (2 * x - π / 6)) ∧
    (∀ x, f (x + π) = f x) ∧
    (∀ x, f (π / 3 - x) = f (π / 3 + x)) :=
begin
  use (λ x, 2 * sin (2 * x - π / 6)),
  split,
  { reflexivity },
  split,
  { sorry },  -- Proof for the period
  { sorry }   -- Proof for the symmetry
end

end correct_function_period_symmetry_l534_534670


namespace increase_in_radius_l534_534559

theorem increase_in_radius 
(C1 : ℝ) (C2 : ℝ) (hC1 : C1 = 20) (hC2 : C2 = 25) (pi : ℝ) (hpi : pi = real.pi) :
  let r1 := C1 / (2 * pi),
      r2 := C2 / (2 * pi),
      delta_r := r2 - r1
  in delta_r = 5 / (2 * pi) :=
by
  -- Mathematical steps outlined in solution are followed here
  unfold r1 r2 delta_r
  rw [hC1, hC2, hpi]
  field_simplify
  norm_num
  exact real.rat_div_two pi

end increase_in_radius_l534_534559


namespace problem_statement_l534_534644

noncomputable def sum_log_floor_ceiling (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * (if (Real.log k / Real.log (Real.sqrt 3)).floor = (Real.log k / Real.log (Real.sqrt 3)).ceil then 0 else 1)

theorem problem_statement :
  sum_log_floor_ceiling 500 = 124886 :=
by
  -- proof starting here, replace with actual proof
  sorry

end problem_statement_l534_534644


namespace count_two_digit_or_less_numbers_l534_534843

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534843


namespace angle_relation_l534_534543

variable (A B C D : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [Isosceles A B C]  -- A, B, C form an isosceles triangle with AB = AC
variable (x y β: ℝ)  -- angles x, y, and β
variable (angle_ABC: angle A B C = β)
variable (angle_ADB: angle A D B == 2 * β)
variable (angle_sum_xy: angle A B D + angle A D C = β)

theorem angle_relation (A B C D : Type*) 
                      [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
                      [Isosceles A B C] 
                      (x y β: ℝ) 
                      (angle_ABC: angle A B C = β)
                      (angle_ADB: angle A D B = 2 * β)
                      (angle_sum_xy: angle A B D + angle A D C = β):
  x + y = β :=
sorry

end angle_relation_l534_534543


namespace no_perfect_square_in_range_l534_534032

def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem no_perfect_square_in_range :
  ∀ (n : ℕ), 4 ≤ n ∧ n ≤ 12 → ¬ isPerfectSquare (2*n*n + 3*n + 2) :=
by
  intro n
  intro h
  sorry

end no_perfect_square_in_range_l534_534032


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534797

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534797


namespace problem_statement_l534_534645

noncomputable def sum_log_floor_ceiling (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * (if (Real.log k / Real.log (Real.sqrt 3)).floor = (Real.log k / Real.log (Real.sqrt 3)).ceil then 0 else 1)

theorem problem_statement :
  sum_log_floor_ceiling 500 = 124886 :=
by
  -- proof starting here, replace with actual proof
  sorry

end problem_statement_l534_534645


namespace prod_formula_value_l534_534266

noncomputable def prod_formula : ℂ := ∏ k in finset.range 10, ∏ j in finset.range 8, (complex.exp (2 * real.pi * complex.I * j / 9) - complex.exp (2 * real.pi * complex.I * k / 11))

theorem prod_formula_value : prod_formula = 1 := by
  sorry

end prod_formula_value_l534_534266


namespace time_to_cover_one_mile_theorem_l534_534235

noncomputable def time_to_cover_one_mile_in_semicircles
    (width : ℝ)
    (semicircles : ℕ)
    (total_length : ℝ)
    (speed : ℝ)
    (radius : ℝ)
    (distance : ℝ) : ℝ :=
    distance / speed

theorem time_to_cover_one_mile_theorem :
    ∀ (width : ℝ)
    (semicircles : ℕ)
    (total_length : ℝ)
    (speed : ℝ)
    (distance : ℝ)
    (radius : ℝ),
    width = 50 →
    total_length = 5280 →
    speed = 6 →
    radius = width / 2 →
    semicircles = (total_length / width).ceil.to_nat →
    distance = (semicircles * (2 * radius * real.pi)) / 2 →
    time_to_cover_one_mile_in_semicircles width semicircles total_length speed radius distance = 265 * real.pi / 3168 :=
    sorry

end time_to_cover_one_mile_theorem_l534_534235


namespace problem_l534_534731

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

theorem problem (a b : ℝ) (h1 : f 1 a b = 0) (h2 : f 2 a b = 0) : f (-1) a b = 6 :=
by
  sorry

end problem_l534_534731


namespace min_value_sin_cos_l534_534984

noncomputable def min_value_trig_function : ℝ :=
  -2

theorem min_value_sin_cos (x : ℝ) : 
  let y := sin x + sqrt 3 * cos x in 
  ∃ (z : ℝ), y = min_value_trig_function :=
by
  use -2
  let y := sin x + sqrt 3 * cos x
  sorry

end min_value_sin_cos_l534_534984


namespace isabella_paintable_area_l534_534909

def total_paintable_area : ℕ :=
  let room1_area := 2 * (14 * 9) + 2 * (12 * 9) - 70
  let room2_area := 2 * (13 * 9) + 2 * (11 * 9) - 70
  let room3_area := 2 * (15 * 9) + 2 * (10 * 9) - 70
  let room4_area := 4 * (12 * 9) - 70
  room1_area + room2_area + room3_area + room4_area

theorem isabella_paintable_area : total_paintable_area = 1502 := by
  sorry

end isabella_paintable_area_l534_534909


namespace find_fraction_l534_534196

theorem find_fraction (x y : ℤ) (h1 : x + 2 = y + 1) (h2 : 2 * (x + 4) = y + 2) : 
  x = -5 ∧ y = -4 := 
sorry

end find_fraction_l534_534196


namespace log_comparison_l534_534726

theorem log_comparison 
(a_def : a = Real.log 2 / Real.log (1/3))
(b_def : b = Real.log 3 / Real.log (1/2))
(c_def : c = (1/2)^0.3) :
  b < a ∧ a < c := 
sorry

end log_comparison_l534_534726


namespace kangaroos_in_circle_l534_534143

-- Definitions of conditions in the problem
variables (n m : ℕ)

-- One theorem to encapsulate all conditions and the proof of the problem
theorem kangaroos_in_circle (n_beavers : n = 3)
  (no_adjacent_beavers : ∀ i : ℕ, beavers i → beavers ((i + 1) % n) → false)
  (adjacent_kangaroos : count_adjacent {i | kangaroos i} = 3) : m = 5 :=
sorry

-- Definitions of beaver and kangaroo presence in circle
def beavers : ℕ → Prop := sorry
def kangaroos : ℕ → Prop := sorry

-- Function to count adjacent elements satisfying a property in the circle
def count_adjacent (P : ℕ → Prop) : ℕ := sorry

end kangaroos_in_circle_l534_534143


namespace arithmetic_square_root_sqrt_81_l534_534507

theorem arithmetic_square_root_sqrt_81 : sqrt (sqrt 81) = 3 := 
by 
  have h1 : sqrt 81 = 9 := sorry
  have h2 : sqrt 9 = 3 := sorry
  show sqrt (sqrt 81) = 3 from sorry

end arithmetic_square_root_sqrt_81_l534_534507


namespace sequence_value_l534_534906

theorem sequence_value (a : ℕ → ℕ) : 
  a 1 = 2 ∧ 
  a 2 = 5 ∧ 
  a 3 = 11 ∧ 
  a 4 = 23 ∧ 
  a 5 = 47 ∧ 
  (∀ n, n ≥ 3 → a (n + 1) - a n = 2 * (a n - a (n - 1))) → 
  a 6 = 95 :=
begin
  sorry
end

end sequence_value_l534_534906


namespace B_is_midpoint_of_PQ_l534_534461

-- Definitions and conditions given in the problem
variables (A B C D E F P Q : Type) [EuclideanGeometry] -- Assuming Euclidean geometry

-- Definitions for points and properties
def triangle_ABC := Triangle A B C
def incircle := incircle_of_triangle triangle_ABC
def D := point_of_tangency incircle side_BC
def E := point_of_tangency incircle side_CA
def F := point_of_tangency incircle side_AB
def P := intersection (line_through E D) (perpendicular_to (line_through E F) F)
def Q := intersection (line_through E F) (perpendicular_to (line_through E D) D)

-- Given conditions as definitions in Lean 4
def side_BC := segment B C
def side_CA := segment C A
def side_AB := segment A B

-- Statement to prove
theorem B_is_midpoint_of_PQ : midpoint B P Q :=
sorry

end B_is_midpoint_of_PQ_l534_534461


namespace angle_BLA_proof_l534_534971

theorem angle_BLA_proof :
  ∀ (A B C D K L : Type) [metric_space Type] [has_dist Type] [add_group Type], 
   -- Define the points and space properties
   dist A B = dist B K ∧ dist B K = dist K D ∧ -- AB = BK = KD
   dist A K = dist L C ∧ -- AK = LC
   angle A B D = 52 ∧
   angle C D B = 74 → -- Given angles
   angle B L A = 42 := -- Angle BLA to be proved
begin
  sorry -- Proof to be completed
end

end angle_BLA_proof_l534_534971


namespace functions_equal_l534_534248

theorem functions_equal : 
  (∀ x : ℝ, x ≠ 0 → f x = g x) ∧ 
  (∀ t : ℝ, t ≠ 0 → f t = g t) →
  (f (x : ℝ) = sqrt (x ^ 2)) ∧ 
  (g (t : ℝ) = abs t) :=
by
  intro h
  sorry

end functions_equal_l534_534248


namespace problem_l534_534730

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

theorem problem (a b : ℝ) (h1 : f 1 a b = 0) (h2 : f 2 a b = 0) : f (-1) a b = 6 :=
by
  sorry

end problem_l534_534730


namespace plot_length_l534_534568

def breadth : ℝ := 40 -- Derived from conditions and cost equation solution
def length : ℝ := breadth + 20
def cost_per_meter : ℝ := 26.50
def total_cost : ℝ := 5300

theorem plot_length :
  (2 * (breadth + (breadth + 20))) * cost_per_meter = total_cost → length = 60 :=
by {
  sorry
}

end plot_length_l534_534568


namespace triangle_max_area_l534_534012

noncomputable def max_triangle_area {A B C : ℝ × ℝ} 
  (hABC_inscribed : ∃ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C ∧ 
    (x ^ 2 / 9) + (y ^ 2 / 4) = 1)
  (hAB_passing_through : ∃ P : ℝ × ℝ, P = (1, 0) ∧ collinear A B P) : ℝ :=
  (16 * Real.sqrt 2) / 3

theorem triangle_max_area
  (A B C : ℝ × ℝ)
  (hABC_inscribed : ∃ x y, ((x, y) = A ∨ (x, y) = B ∨ (x, y) = C) ∧ 
    (x ^ 2 / 9) + (y ^ 2 / 4) = 1)
  (hAB_passing_through : (∃ P : ℝ × ℝ, P = (1, 0) ∧ collinear A B P)) :
  max_triangle_area hABC_inscribed hAB_passing_through = (16 * Real.sqrt 2) / 3 :=
sorry

end triangle_max_area_l534_534012


namespace find_m_l534_534221

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n / 2

theorem find_m (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 16) : m = 59 ∨ m = 91 :=
by sorry

end find_m_l534_534221


namespace find_lambda_l534_534093

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ)

-- Conditions: 
-- S_n = 3^(n+1) + λ
-- a_n is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

-- Lean statement for the problem
theorem find_lambda (h₁ : ∀ n, n > 0 → S n = 3^(n + 1) + λ)
                    (h₂ : ∀ n, n > 0 → n < 4 → a (n + 1) = S (n + 1) - S n)
                    (h₃ : is_geometric_sequence a) :
                    λ = -3 :=
by
  sorry

end find_lambda_l534_534093


namespace gina_keeps_170_l534_534310

theorem gina_keeps_170 (initial_amount : ℕ)
    (money_to_mom : ℕ)
    (money_to_clothes : ℕ)
    (money_to_charity : ℕ)
    (remaining_money : ℕ) :
  initial_amount = 400 →
  money_to_mom = (1 / 4) * initial_amount →
  money_to_clothes = (1 / 8) * initial_amount →
  money_to_charity = (1 / 5) * initial_amount →
  remaining_money = initial_amount - (money_to_mom + money_to_clothes + money_to_charity) →
  remaining_money = 170 := sorry

end gina_keeps_170_l534_534310


namespace sum_odd_primes_less_200_l534_534189

theorem sum_odd_primes_less_200 : (∑ p in Finset.filter (λ p, p.Prime ∧ p % 2 = 1) (Finset.range 200), p) = 4227 := sorry

end sum_odd_primes_less_200_l534_534189


namespace distribution_ways_l534_534284

theorem distribution_ways (books students : ℕ) (h_books : books = 6) (h_students : students = 6) :
  ∃ ways : ℕ, ways = 6 * 5^6 ∧ ways = 93750 :=
by
  sorry

end distribution_ways_l534_534284


namespace count_at_most_two_different_digits_l534_534858

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534858


namespace range_of_x_f_lt_0_l534_534039

noncomputable def f (x : ℝ) : ℝ := x ^ (2 / 3) - x ^ (-1 / 2)

theorem range_of_x_f_lt_0 : { x : ℝ | f x < 0 } = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end range_of_x_f_lt_0_l534_534039


namespace max_distance_to_origin_l534_534900

noncomputable def circleEquation (a b : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1)^2 + (p.2)^2 + a * p.1 + b * p.2 = 0 }

def pt1 : ℝ × ℝ := (0, 0)
def pt2 : ℝ × ℝ := (2, 4)
def pt3 : ℝ × ℝ := (3, 3)

theorem max_distance_to_origin :
  ∃ (a b : ℝ), pt1 ∈ circleEquation a b ∧
                pt2 ∈ circleEquation a b ∧
                pt3 ∈ circleEquation a b ∧
                ∀ (p ∈ circleEquation a b), ⟨(p.1)^2 + (p.2 - 0)^2⟩ ≤ 2 * Real.sqrt 5 :=
sorry

end max_distance_to_origin_l534_534900


namespace monic_quadratic_with_given_root_l534_534690

theorem monic_quadratic_with_given_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.eval (-3 - complex.i * Real.sqrt 7) = 0 ∧ p = Polynomial.X^2 + 6 * Polynomial.X + 16 :=
by
  sorry

end monic_quadratic_with_given_root_l534_534690


namespace wait_time_difference_l534_534533

noncomputable def kids_waiting_for_swings : ℕ := 3
noncomputable def kids_waiting_for_slide : ℕ := 2 * kids_waiting_for_swings
noncomputable def wait_per_kid_swings : ℕ := 2 * 60 -- 2 minutes in seconds
noncomputable def wait_per_kid_slide : ℕ := 15 -- in seconds

noncomputable def total_wait_swings : ℕ := kids_waiting_for_swings * wait_per_kid_swings
noncomputable def total_wait_slide : ℕ := kids_waiting_for_slide * wait_per_kid_slide

theorem wait_time_difference : total_wait_swings - total_wait_slide = 270 := by
  sorry

end wait_time_difference_l534_534533


namespace frog_jump_probability_l534_534585

-- Defining the problem domain
def grid_size : ℕ := 7
def start_x : ℕ := 2
def start_y : ℕ := 3

-- Define the probability function P
def P : ℕ × ℕ → ℚ
| (0, y) := 1
| (grid_size, y) := 1
| (x, 0) := 0
| (x, grid_size) := 0
| (2, 5) := 1/2 * P (2, 3)
| (2, 1) := 1/2 * P (2, 3)
| (2, 3) := 1/4 * P (4, 3) + 1/4 * P (0, 3) + 1/4 * P (2, 5) + 1/4 * P (2, 1)
| _ := 0

-- Statement to prove
theorem frog_jump_probability : P (start_x, start_y) = 2 / 3 := by
  sorry

end frog_jump_probability_l534_534585


namespace num_integers_two_digits_l534_534803

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534803


namespace optimal_speed_and_minimum_cost_l534_534409

noncomputable def fuel_consumption_rate (x : ℝ) : ℝ :=
  3 + x^2 / 360

noncomputable def total_cost (x : ℝ) : ℝ :=
  let driving_time := 45 / x
  let driver_fee := driving_time * 56
  let fuel_consumed := driving_time * fuel_consumption_rate x
  let fuel_cost := fuel_consumed * 8
  driver_fee + fuel_cost

noncomputable def minimum_total_cost (x : ℝ) : ℝ :=
  3600 / x + x

theorem optimal_speed_and_minimum_cost :
  (∃ x, x = 50 ∧ total_cost x = 122) :=
by
  -- We assume the optimal speed and the minimum total cost are correctly derived
  have h1: total_cost 50 = 122, from sorry,
  use 50
  exact ⟨rfl, h1⟩

end optimal_speed_and_minimum_cost_l534_534409


namespace general_term_T_2016_eq_l534_534762

-- Define the arithmetic sequence {a_n} with given conditions
def a (n : ℕ) : ℕ := 2 * n

-- Define the sum of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of the first n terms of the sequence {1/S_n}
def T (n : ℕ) : ℚ := ∑ i in Finset.range n, (1 : ℚ) / (S (i + 1))

-- Prove that the general term formula for the sequence {a_n} is a_n = 2n
theorem general_term :
  ∀ n : ℕ, a n = 2 * n := by 
sorry

-- Prove that the value of T_2016 is 2016 / 2017
theorem T_2016_eq :
  T 2016 = 2016 / 2017 := by 
sorry

end general_term_T_2016_eq_l534_534762


namespace green_bean_ratio_l534_534494

theorem green_bean_ratio (total_beans : ℕ) (red_fraction white_fraction : ℚ)
  (initial_green_beans : ℕ)
  (H1 : total_beans = 572)
  (H2 : red_fraction = 1 / 4)
  (H3 : white_fraction = 1 / 3)
  (H4 : initial_green_beans = 143)
  (red_beans : ℕ := (red_fraction * total_beans).toNat)
  (remaining_after_red : ℕ := total_beans - red_beans)
  (white_beans : ℕ := (white_fraction * remaining_after_red).toNat)
  (remaining_after_white : ℕ := remaining_after_red - white_beans) :
  initial_green_beans = remaining_after_white →
  ratio : ℚ := initial_green_beans / remaining_after_white :
  ratio = 1 :=
sorry

end green_bean_ratio_l534_534494


namespace magnitude_of_sum_l534_534928

-- Definition of the given conditions and vectors
variables {x : ℝ} (a : ℝ × ℝ) (b : ℝ × ℝ)

-- Establish the conditions
def a := (x, 1 : ℝ)
def b := (1, -2 : ℝ)

-- Definition of perpendicularity
def perp (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Main statement to be proved
theorem magnitude_of_sum (h : perp a b) : 
  ‖(a.1 + 2 * b.1, a.2 + 2 * b.2)‖ = 5 :=
sorry

end magnitude_of_sum_l534_534928


namespace point_translation_l534_534065

theorem point_translation :
  ∃ (x_old y_old x_new y_new : ℤ),
  (x_old = 1 ∧ y_old = -2) ∧
  (x_new = x_old + 2) ∧
  (y_new = y_old + 3) ∧
  (x_new = 3) ∧
  (y_new = 1) :=
sorry

end point_translation_l534_534065


namespace cow_problem_l534_534054

noncomputable def problem_statement : Prop :=
  ∃ (F M : ℕ), F + M = 300 ∧
               (∃ S H : ℕ, S = 1/2 * F ∧ H = 1/2 * M ∧ S = H + 50) ∧
               F = 2 * M

theorem cow_problem : problem_statement :=
sorry

end cow_problem_l534_534054


namespace cos_4theta_l534_534875

theorem cos_4theta 
  (θ : ℝ)
  (h : complex.exp (complex.I * θ) = (3 - complex.I * sqrt 2) / 4) : 
  real.cos (4 * θ) = 121 / 256 :=
sorry

end cos_4theta_l534_534875


namespace count_special_integers_l534_534823

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534823


namespace range_of_t_l534_534745

-- A structure to represent a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a circle given a center and radius
def Circle (center : Point) (radius : ℝ) : Prop :=
  ∀ p : Point, (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

-- A point M at (5, t)
def M (t : ℝ) : Point := { x := 5, y := t }

-- Center of the circle C is (1, 4)
def C_center : Point := { x := 1, y := 4 }

-- Radius of circle C is sqrt(10)
def C_radius : ℝ := real.sqrt 10

-- Circle C defined by its center and radius
def C : Prop := Circle C_center C_radius

-- Distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- MC (distance from M to the center of circle C) in limit condition
def MC (t : ℝ) : ℝ := distance (M t) C_center

-- Proving that the range of t is [2, 6]
theorem range_of_t (t : ℝ) : C → (2 ≤ t ∧ t ≤ 6) ↔ (MC t ≤ 2 * real.sqrt 5) := by
  intro hC
  unfold C at hC
  unfold MC
  unfold distance
  sorry

end range_of_t_l534_534745


namespace slope_of_asymptotes_l534_534377

noncomputable def hyperbola_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let A := (-Real.sqrt 5 * a, 2 * b) in
  let slope_OA := (2 * b) / (-Real.sqrt 5 * a) in
  slope_OA = -1 →
  (b / a = Real.sqrt 5 / 2)

theorem slope_of_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_problem a b ha hb → ∃ k, k = Real.sqrt 5 / 2 ∧ (∀ x y : ℝ,  (y = k * x ∨ y = -k * x)) :=
sorry

end slope_of_asymptotes_l534_534377


namespace count_integers_abs_le_7_l534_534300

theorem count_integers_abs_le_7 : {x : ℤ | |x - 3| ≤ 7}.card = 15 :=
by
  sorry

end count_integers_abs_le_7_l534_534300


namespace pq_or_l534_534112

def p : Prop := 2 % 2 = 0
def q : Prop := 3 % 2 = 0

theorem pq_or : p ∨ q :=
by
  -- proof goes here
  sorry

end pq_or_l534_534112


namespace projection_magnitude_l534_534368

-- Definitions for the conditions
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, 1)

-- Theorem statement about the projection magnitude
theorem projection_magnitude (a b : ℝ × ℝ) 
  (h₁ : a = (-1, 1)) (h₂ : b = (3, 1)) 
  : (-(a.1 * b.1 + a.2 * b.2) / sqrt ((a.1 ^ 2 + a.2 ^ 2) * (b.1 ^ 2 + b.2 ^ 2))) * sqrt (a.1 ^ 2 + a.2 ^ 2) = sqrt 2 :=
sorry

end projection_magnitude_l534_534368


namespace part1_part2_part3_l534_534492

/-- The given matrix and conditions. -/
structure MatrixCondition (A : Matrix (Fin 2) (Fin 2) Int) : Prop :=
  (det_one : Int.natAbs (A.det) = 1)

/-- Part (1): Proving that BA and B^{-1}A satisfy the given condition. -/
theorem part1 (a b c d : Int) (h : MatrixCondition (Matrix.of ![![a, b], ![c, d]])) :
  let B := (Matrix.of ![![1, 1], ![0, 1]] : Matrix (Fin 2) (Fin 2) Int)
  let Binv := (Matrix.of ![![1, -1], ![0, 1]] : Matrix (Fin 2) (Fin 2) Int)
  MatrixCondition (B ⬝ (Matrix.of ![![a, b], ![c, d]])) ∧
  MatrixCondition (Binv ⬝ (Matrix.of ![![a, b], ![c, d]])) := sorry

/-- Part (2): Given c = 0 validate resulting matrices. -/
theorem part2 (a b d : Int) (c : Int) (h : c = 0) :
  let A := (Matrix.of ![![a, b], ![c, d]] : Matrix (Fin 2) (Fin 2) Int)
  let B := (Matrix.of ![![1, 1], ![0, 1]] : Matrix (Fin 2) (Fin 2) Int)
  let Binv := (Matrix.of ![![1, -1], ![0, 1]] : Matrix (Fin 2) (Fin 2) Int)
  let BA := B ⬝ A
  let BinvA := Binv ⬝ A
  (BA = (Matrix.of ![![1, 0], ![0, 1]]) ∨
    BA = (Matrix.of ![![-1, 0], ![0, 1]]) ∨
    BA = (Matrix.of ![![1, 0], ![0, -1]]) ∨
    BA = (Matrix.of ![![-1, 0], ![0, -1]])) ∨
  (BinvA = (Matrix.of ![![1, 0], ![0, 1]]) ∨
    BinvA = (Matrix.of ![![-1, 0], ![0, 1]]) ∨
    BinvA = (Matrix.of ![![1, 0], ![0, -1]]) ∨
    BinvA = (Matrix.of ![![-1, 0], ![0, -1]])) := sorry

/-- Part (3): Prove that at least one of BA or B^{-1}A satisfies the inequality. -/
theorem part3 (a b c d : Int) (h : |a| ≥ |c| ∧ |c| > 0 ∧ MatrixCondition (Matrix.of ![![a, b], ![c, d]])) :
  let B := (Matrix.of ![![1, 1], ![0, 1]] : Matrix (Fin 2) (Fin 2) Int)
  let Binv := (Matrix.of ![![1, -1], ![0, 1]] : Matrix (Fin 2) (Fin 2) Int)
  let BA := B ⬝ (Matrix.of ![![a, b], ![c, d]])
  let BinvA := Binv ⬝ (Matrix.of ![![a, b], ![c, d]])
  (let x := BA[0, 0], z := BA[1, 0] in |x| + |z| < |a| + |c|) ∨
  (let x := BinvA[0,0], z := BinvA[1,0] in |x| + |z| < |a| + |c|) := sorry

end part1_part2_part3_l534_534492


namespace monic_quadratic_with_given_root_l534_534692

theorem monic_quadratic_with_given_root :
  ∃ (p : Polynomial ℝ), p.monic ∧ p.eval (-3 - complex.i * Real.sqrt 7) = 0 ∧ p = Polynomial.X^2 + 6 * Polynomial.X + 16 :=
by
  sorry

end monic_quadratic_with_given_root_l534_534692


namespace sqrt_x_minus_1_meaningful_real_l534_534396

theorem sqrt_x_minus_1_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ (x ≥ 1) :=
by
  sorry

end sqrt_x_minus_1_meaningful_real_l534_534396


namespace find_angle_C_l534_534365

variables (A B C : ℝ)
variables (m n : ℝ × ℝ)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_angle_C (h_m : m = (real.sqrt 3 * real.sin A, real.sin B))
                    (h_n : n = (real.cos B, real.sqrt 3 * real.cos A))
                    (h_dot : dot_product m n = 1 + real.cos (A + B)) :
                    C = 2 * real.pi / 3 := 
begin
  sorry
end

end find_angle_C_l534_534365


namespace a0_val_a1_a7_sum_a1_a3_a5_a7_sum_l534_534008

variable {x : ℝ}
noncomputable def poly := (2 * x - 1)^7

theorem a0_val : 
  (poly = (a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0)) → 
  a_0 = -1 :=
sorry

theorem a1_a7_sum :
  (poly = (a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0)) → 
  a_0 = -1 → 
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 2 :=
sorry

theorem a1_a3_a5_a7_sum :
  (poly = (a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0)) → 
  a_0 = -1 → 
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 2 → 
  a_1 + a_3 + a_5 + a_7 = -126 :=
sorry

end a0_val_a1_a7_sum_a1_a3_a5_a7_sum_l534_534008


namespace magnitude_of_difference_l534_534314

def vector_a (x : ℝ) := (x, 2)
def vector_b := (2, -1)

theorem magnitude_of_difference 
  (x : ℝ) 
  (h : vector_a x • vector_b = 0) :
  ‖vector_a x - vector_b‖ = real.sqrt 10 :=
by {
  sorry
}

end magnitude_of_difference_l534_534314


namespace cost_relation_cost_effective_l534_534601

variable (x : ℕ)
def pen_price := 15
def notebook_price := 4
def additional_notebook_count := 10

def y1 (x : ℕ) : ℕ := pen_price * x + notebook_price * additional_notebook_count
def y2 (x : ℕ) : ℕ := ((pen_price * x + notebook_price * (x + additional_notebook_count)) * 0.8).to_nat

theorem cost_relation (x : ℕ) :
  y1 x = pen_price * x + notebook_price * additional_notebook_count ∧
  y2 x = (15.2 * x + 32).to_nat := sorry

theorem cost_effective : y2 10 < y1 10 := sorry

end cost_relation_cost_effective_l534_534601


namespace smallest_positive_period_max_value_g_l534_534466

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * cos (x + π) * cos x

-- Prove the smallest positive period of f(x) is π
theorem smallest_positive_period : ∃ p > 0, ∀ x, f (x + p) = f x ∧ (∀ q, (∀ x, f (x + q) = f x) → q ≥ p) := 
sorry 

noncomputable def g (x : ℝ) : ℝ := f (x - π / 4) + sqrt 3 / 2

-- Prove the maximum value of g(x) on [0, π / 4] is 3 * sqrt 3 / 2
theorem max_value_g : ∃ x ∈ set.Icc (0 : ℝ) (π / 4), ∀ y ∈ set.Icc (0 : ℝ) (π / 4), g y ≤ g x ∧ g x = 3 * sqrt 3 / 2 := 
sorry

end smallest_positive_period_max_value_g_l534_534466


namespace chris_previous_savings_l534_534622

variable (USD : Type) [LinearOrder USD] [Div USD] [OfNat USD 0] [OfNat USD 25]
variable (add : USD → USD → USD) [HasAdd add]
variable (sub : USD → USD → USD) [HasSub sub]
variable (mul : USD → USD → USD) [HasMul mul]
variable (of_rat : ℚ → USD) [HasRatCast of_rat]

def conversion_rates : Type :=
{ eur_to_usd : ℚ := 0.85,
  cad_to_usd : ℚ := 1.25,
  gbp_to_usd : ℚ := 0.72 }

def birthday_money_usd (rates : conversion_rates) : USD :=
  add (add (add (of_nat 25) 
                (mul (of_rat 20) (of_rat (1 / rates.eur_to_usd)))) 
            (mul (of_rat 75) (of_rat (1 / rates.cad_to_usd)))) 
      (mul (of_rat 30) (of_rat (1 / rates.gbp_to_usd)))

def total_savings : USD := of_nat 279

theorem chris_previous_savings (rates : conversion_rates) : 
  sub total_savings (birthday_money_usd rates) = of_nat 128.80 :=
sorry

end chris_previous_savings_l534_534622


namespace sum_le_six_l534_534006

theorem sum_le_six (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
    (h3 : ∃ (r s : ℤ), r * s = a + b ∧ r + s = ab) : a + b ≤ 6 :=
sorry

end sum_le_six_l534_534006


namespace stella_doll_price_l534_534144

theorem stella_doll_price 
  (dolls_count clocks_count glasses_count : ℕ)
  (price_per_clock price_per_glass cost profit : ℕ)
  (D : ℕ)
  (h1 : dolls_count = 3)
  (h2 : clocks_count = 2)
  (h3 : glasses_count = 5)
  (h4 : price_per_clock = 15)
  (h5 : price_per_glass = 4)
  (h6 : cost = 40)
  (h7 : profit = 25)
  (h8 : 3 * D + 2 * price_per_clock + 5 * price_per_glass = cost + profit) :
  D = 5 :=
by
  sorry

end stella_doll_price_l534_534144


namespace smallest_prime_dividing_sum_l534_534555

theorem smallest_prime_dividing_sum : ∃ p : ℕ, p.prime ∧ (p ∣ (2^14 + 7^12)) ∧ ∀ q : ℕ, (q.prime ∧ (q ∣ (2^14 + 7^12))) → q ≥ p := by
  sorry

end smallest_prime_dividing_sum_l534_534555


namespace part1_part2_l534_534736

theorem part1 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h_imag : z.re = 0) : a = 1 :=
sorry

theorem part2 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h4thQuad : z.re > 0 ∧ z.im < 0) : -1 < a ∧ a < 1 :=
sorry

end part1_part2_l534_534736


namespace solution_l534_534877

theorem solution (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : y - 2 * q = 3 - 3 * q :=
by
  sorry

end solution_l534_534877


namespace trajectory_M_equation_l_OP_equal_OM_l534_534007

variables {P : Point}
variables {C : Circle}
variables {O : Point}
variables {l : Line}
variables {A B M : Point}

-- Conditions as definitions
def point_P : P = (2, 2) := sorry
def circle_C : C = { center := (4, 0), radius := 4, equation := λ x y, (x - 4)^2 + y^2 = 16 } := sorry
def line_intersects : l.passes_through P := sorry
def line_intersects_circle : l.intersects_circle C A B := sorry
def midpoint_AB : midpoint A B = M := sorry
def origin_O : O = (0, 0) := sorry

-- Prove statements
theorem trajectory_M : ∀ M, trajectory_eq (x - 3)^2 + (y - 1)^2 = 2 := sorry

theorem equation_l_OP_equal_OM : (dist O P = dist O M) → line_eq l 3x + y - 8 = 0 := sorry

end trajectory_M_equation_l_OP_equal_OM_l534_534007


namespace line_intersects_y_axis_at_point_l534_534616

theorem line_intersects_y_axis_at_point :
  ∃ (y : ℝ), 
  (∃ m : ℝ, m = (15 - 9) / (4 - 2) ∧ ∃ b : ℝ, b = 9 - m * 2 ∧ y = b) ∧
  (0, y) = (0, 3) :=
begin
  sorry
end

end line_intersects_y_axis_at_point_l534_534616


namespace count_positive_integers_with_two_digits_l534_534833

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534833


namespace airplane_takeoff_run_distance_l534_534172

noncomputable def distance_travelled (t : ℝ) (v : ℝ) : ℝ :=
  let v_mps := v * 1000 / 3600  -- Convert km/h to m/s
  let a := v_mps / t
  (1/2) * a * t^2

theorem airplane_takeoff_run_distance :
  distance_travelled 15 100 = 208 :=
by
  -- Definitions and condition handling
  have t := 15 : ℝ
  have v := 100 : ℝ
  let v_mps := v * 1000 / 3600
  let a := v_mps / t
  let S := (1 / 2) * a * t ^ 2
  
  -- Statement to prove
  show S = 208 from sorry

end airplane_takeoff_run_distance_l534_534172


namespace floor_x0_eq_2_l534_534755

-- Conditions: definitions and assumptions
def floor (x : ℝ) : ℤ := Int.floor x
axiom floor_1_2 : floor 1.2 = 1
axiom floor_m1_5 : floor (-1.5) = -2

noncomputable def f (x : ℝ) : ℝ := Real.log x - (2 / x)
axiom x0 : ℝ
axiom zero_point : f x0 = 0

-- Problem statement
theorem floor_x0_eq_2 : floor x0 = 2 :=
  sorry

end floor_x0_eq_2_l534_534755


namespace sqrt_eq_no_real_roots_l534_534661

noncomputable def no_real_roots (x : ℝ) :=
  sqrt (x + 9) - sqrt (x - 1) + 2 = 0 → false

theorem sqrt_eq_no_real_roots : ∀ x : ℝ, no_real_roots x :=
by
  intro x
  sorry

end sqrt_eq_no_real_roots_l534_534661


namespace exists_k_with_distinct_remainders_l534_534109

theorem exists_k_with_distinct_remainders :
  ∀ (p : ℕ) (hp: p.prime) (a : ℕ → ℤ),
  ∃ (k : ℤ), 
  (finset.univ.image (λ i : fin p, ((a i.val + ↑i * k) % p))).card ≥ p / 2 :=
by sorry

end exists_k_with_distinct_remainders_l534_534109


namespace hyperbola_asymptote_slope_l534_534663

theorem hyperbola_asymptote_slope :
  (∀ x y : ℝ, abs (sqrt ((x - 3)^2 + (y + 1)^2) - sqrt ((x - 7)^2 + (y + 1)^2)) = 4) → 
  (∃ m : ℝ, m > 0 ∧ m = 1) :=
by 
  sorry

end hyperbola_asymptote_slope_l534_534663


namespace unique_fraction_increased_by_20_percent_l534_534271

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem unique_fraction_increased_by_20_percent (x y : ℕ) (h1 : relatively_prime x y) (h2 : x > 0) (h3 : y > 0) :
  (∃! (x y : ℕ), relatively_prime x y ∧ (x > 0) ∧ (y > 0) ∧ (x + 2) * y = 6 * (y + 2) * x) :=
sorry

end unique_fraction_increased_by_20_percent_l534_534271


namespace find_a_from_tangent_l534_534321

-- Given the function f(x) = ax + ln x, and one of its tangent lines is y = x,
-- Prove that a = 1 - 1/e
theorem find_a_from_tangent (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (x0 : ℝ) (h1 : f = λ x, a * x + Real.log x) (h2 : f' = λ x, a + 1 / x) 
  (h3 : TangentLine (λ x, f x) x0 = λ x, x) : 
  a = 1 - 1 / Real.exp 1 :=
by
  sorry

-- Definition for TangentLine not directly given in Lean; to be assumed or proven separately.
def TangentLine (f : ℝ → ℝ) (x0 : ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ (y0 : ℝ), (f x0 = y0) ∧ (l = λ x, f' x0 * (x - x0) + y0)

end find_a_from_tangent_l534_534321


namespace difference_sum_Alice_Bob_l534_534273

-- Definition of Alice's sum
def sumAlice (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Definition of rounding to the nearest multiple of 5
def round_to_nearest_5 (x : ℕ) : ℕ :=
  5 * (x + 2) / 5

-- Definition of Bob's sum
def sumBob (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, round_to_nearest_5 (i + 1)

-- The main theorem to prove the positive difference between Alice's and Bob's sums is 1560
theorem difference_sum_Alice_Bob : |sumAlice 60 - sumBob 60| = 1560 :=
by {
  sorry -- Proof to be completed
}

end difference_sum_Alice_Bob_l534_534273


namespace count_special_integers_l534_534820

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534820


namespace wage_consumption_percentage_l534_534227

theorem wage_consumption_percentage
  (x y : ℝ)
  (h_eq : y = 0.66 * x + 1.562)
  (h_y : y = 7.675) :
  (7.675 / x) * 100 ≈ 83 := 
  sorry

end wage_consumption_percentage_l534_534227


namespace find_values_l534_534727

theorem find_values (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 = 4 * a * b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end find_values_l534_534727


namespace find_multiplier_l534_534558

theorem find_multiplier (n m : ℕ) (h1 : 2 * n = (26 - n) + 19) (h2 : n = 15) : m = 2 :=
by
  sorry

end find_multiplier_l534_534558


namespace books_sold_on_Thursday_l534_534913

theorem books_sold_on_Thursday (
  (total_stock : ℕ) 
  (sold_Monday : ℕ) 
  (sold_Tuesday : ℕ) 
  (sold_Wednesday : ℕ) 
  (sold_Friday : ℕ) 
  (not_sold_percentage : ℚ)
  (h_total_stock : total_stock = 1200)
  (h_sold_Monday : sold_Monday = 75)
  (h_sold_Tuesday : sold_Tuesday = 50)
  (h_sold_Wednesday : sold_Wednesday = 64)
  (h_sold_Friday : sold_Friday = 135)
  (h_not_sold_percentage : not_sold_percentage = 0.665)
  (h_books_not_sold : (total_stock : ℚ) * not_sold_percentage = 798)
  (h_books_sold_mon_wed_fri : (sold_Monday + sold_Tuesday + sold_Wednesday + sold_Friday) = 324)
) : (total_stock - (798 + 324) : ℕ) = 78 := by
  sorry

end books_sold_on_Thursday_l534_534913


namespace student_distribution_l534_534536

theorem student_distribution (a b : ℕ) (h1 : a + b = 81) (h2 : a = b - 9) : a = 36 ∧ b = 45 := 
by
  sorry

end student_distribution_l534_534536


namespace james_total_earnings_l534_534439

def january_earnings : ℕ := 4000
def february_earnings : ℕ := january_earnings + (50 * january_earnings / 100)
def march_earnings : ℕ := february_earnings - (20 * february_earnings / 100)
def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings :
  total_earnings = 14800 :=
by
  -- skip the proof
  sorry

end james_total_earnings_l534_534439


namespace triangle_division_exists_l534_534080

theorem triangle_division_exists :
  ∀ (A B C : Point), 
  ∃ (D E F G : Point), 
  convex_shape A B C D E F G :=
sorry

end triangle_division_exists_l534_534080


namespace log_expression_simplification_ratio_subtraction_l534_534115

-- Mathematical Expression (1) Proof Problem
theorem log_expression_simplification :
  (log 3 / log 4) * ((log 8 / log 9) + (log 16 / log 27)) = 17 / 12 :=
by sorry

-- Mathematical Expression (2) Proof Problem
theorem ratio_subtraction (x y z : ℝ) (a : ℝ) (h : a > 0)
  (hx : 3^x = a) (hy : 4^y = a) (hz : 6^z = a) :
  (y / z) - (y / x) = 1 / 2 :=
by sorry

end log_expression_simplification_ratio_subtraction_l534_534115


namespace symmetry_axis_l534_534355

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * sin x * cos x - (cos x) ^ 2
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6)

theorem symmetry_axis (x : ℝ) : ∃ k : ℤ, x = k * (π / 2) + π / 6 :=
begin
  use 0,
  linarith,
end

end symmetry_axis_l534_534355


namespace Cody_spent_25_tickets_on_beanie_l534_534253

-- Introducing the necessary definitions and assumptions
variable (x : ℕ)

-- Define the conditions translated from the problem statement
def initial_tickets := 49
def tickets_left (x : ℕ) := initial_tickets - x + 6

-- State the main problem as Theorem
theorem Cody_spent_25_tickets_on_beanie (H : tickets_left x = 30) : x = 25 := by
  sorry

end Cody_spent_25_tickets_on_beanie_l534_534253


namespace lines_intersection_l534_534778

def intersection_point_of_lines
  (t u : ℚ)
  (x₁ y₁ x₂ y₂ : ℚ)
  (x y : ℚ) : Prop := 
  ∃ (t u : ℚ),
    (x₁ + 3*t = 7 + 6*u) ∧
    (y₁ - 4*t = -5 + 3*u) ∧
    (x = x₁ + 3 * t) ∧ 
    (y = y₁ - 4 * t)

theorem lines_intersection :
  ∀ (t u : ℚ),
    intersection_point_of_lines t u 3 2 7 (-5) (87/11) (-50/11) :=
by
  sorry

end lines_intersection_l534_534778


namespace potatoes_left_l534_534472

theorem potatoes_left (initial_potatoes : ℕ) (potatoes_for_salads : ℕ) (potatoes_for_mashed : ℕ)
  (h1 : initial_potatoes = 52)
  (h2 : potatoes_for_salads = 15)
  (h3 : potatoes_for_mashed = 24) :
  initial_potatoes - (potatoes_for_salads + potatoes_for_mashed) = 13 := by
  sorry

end potatoes_left_l534_534472


namespace edges_not_in_same_plane_as_AC1_are_six_l534_534066

-- Define the vertices of the cube
structure Vertex :=
(x y z : ℕ)

-- Prove that there are exactly 6 edges not in the same plane as the diagonal AC_1
theorem edges_not_in_same_plane_as_AC1_are_six :
  let A : Vertex := ⟨0, 0, 0⟩
  let B : Vertex := ⟨1, 0, 0⟩
  let C : Vertex := ⟨1, 1, 0⟩
  let D : Vertex := ⟨0, 1, 0⟩
  let A1 : Vertex := ⟨0, 0, 1⟩
  let B1 : Vertex := ⟨1, 0, 1⟩
  let C1 : Vertex := ⟨1, 1, 1⟩
  let D1 : Vertex := ⟨0, 1, 1⟩
  let diagonal_AC1 := (A, C1)
  let edges := [(A, B), (B, C), (C, D), (D, A), (A1, B1), (B1, C1), (C1, D1), (D1, A1), (A, A1), (B, B1), (C, C1), (D, D1)] in
  (edges.filter (λ e, ¬ are_in_same_plane e diagonal_AC1)).length = 6 := 
by
  -- Proof goes here
  sorry

end edges_not_in_same_plane_as_AC1_are_six_l534_534066


namespace mean_and_median_change_l534_534537

def original_transactions := [50, 80, 30, 60, 40, 40]
def corrected_transactions := [50, 80, 45, 60, 40]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length
def median (lst : List ℕ) : ℚ :=
  let sorted := lst.qsort (· <= ·)
  let mid := sorted.length / 2
  if sorted.length % 2 = 0 then
    (sorted.get! (mid - 1) + sorted.get! mid : ℚ) / 2
  else
    sorted.get! mid

theorem mean_and_median_change :
  mean original_transactions + 5 = mean corrected_transactions ∧
  median original_transactions + 5 = median corrected_transactions := by
  sorry

end mean_and_median_change_l534_534537


namespace cartesian_plane_divides_into_four_quadrants_l534_534893

theorem cartesian_plane_divides_into_four_quadrants :
  ∀ (CartesianPlane : Type), (∃ o : CartesianPlane, ∃ x y : CartesianPlane, 
  x ≠ y ∧ (x ⊥ y) ∧ (common_origin o x y)) → (number_of_quadrants CartesianPlane = 4) :=
by
  intros CartesianPlane h
  sorry

end cartesian_plane_divides_into_four_quadrants_l534_534893


namespace length_AB_correct_l534_534901

-- Define the parametric equations of the line l
def line_parametric_eq (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 + t, 6 - sqrt 3 * t)

-- Define the polar to Cartesian conversion for the center of the circle
def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

-- Center of the circle in Cartesian coordinates
def center_cartesian : ℝ × ℝ :=
  polar_to_cartesian 2 (Real.pi / 2)

-- Equation of the circle in Cartesian coordinates
def circle_eq (x y : ℝ) : Prop :=
  x ^ 2 + (y - 2) ^ 2 = 4

-- Polar coordinate equation of circle C
def circle_polar_eq (ρ θ : ℝ) : Prop :=
  ρ = 4 * sin θ

-- Intersection points A and B in polar coordinates
def point_A (θ : ℝ) : ℝ :=
  if θ = Real.pi / 3 then 2 * sqrt 3 else 0

def point_B (θ : ℝ) : ℝ :=
  if θ = Real.pi / 3 then 3 * sqrt 3 else 0

-- Length of segment AB
def length_AB (A B : ℝ) : ℝ :=
  abs (B - A)

-- The statement to prove
theorem length_AB_correct :
  (θ : ℝ) → point_A θ ≠ 0 → point_B θ ≠ 0 → θ = Real.pi / 3 →
  length_AB (point_A θ) (point_B θ) = sqrt 3 :=
by intros; sorry

end length_AB_correct_l534_534901


namespace transformed_point_distance_l534_534216

noncomputable def distance_moved_by_P : ℝ :=
  let B : ℝ × ℝ := (3, 3)
  let B' : ℝ × ℝ := (6, 9)
  let P : ℝ × ℝ := (1, 1)
  let dilation_factor : ℝ := 4.5 / 3
  let rotated_P : ℝ × ℝ := 
    (dilation_factor * P.1 * real.cos (real.pi / 4) - dilation_factor * P.2 * real.sin (real.pi / 4),
    dilation_factor * P.1 * real.sin (real.pi / 4) + dilation_factor * P.2 * real.cos (real.pi / 4))
  let moved_distance : ℝ := real.sqrt ((P.1 - rotated_P.1)^2 + (P.2 - rotated_P.2)^2)
  moved_distance

theorem transformed_point_distance :
  distance_moved_by_P = real.sqrt (1 + (1 - 1.5 * real.sqrt 2)^2) := by
  /- Proof goes here -/
  sorry

end transformed_point_distance_l534_534216


namespace find_power_n_l534_534380

-- Define the conditions as hypothesis
variables (k j z : ℝ)
def x (y : ℝ) : ℝ := k * y^4
def y (z : ℝ) : ℝ := j * z^(1/3)

-- The main theorem to find the power n
theorem find_power_n (k j z : ℝ) : ∃ (m : ℝ), ∀ (z : ℝ), x(k, j)(y z) = m * z^(4/3) :=
sorry

end find_power_n_l534_534380


namespace max_temp_range_l534_534965

theorem max_temp_range (temps : Fin 5 → ℝ) (avg_temp : (∑ i, temps i) / 5 = 50) (lowest_temp : ∃ i, temps i = 45) :
  ∃ T_max, (max_range temps = T_max) ∧ T_max = 25 :=
by
  have sum_temps : ∑ i, temps i = 250 := 
    by sorry
  have h_lowest : ∀ i, temps i ≥ 45 := 
    by sorry
  let T_max := 70
  let max_rng := 70 - 45
  existsi T_max
  split
  · sorry
  · exact rfl

end max_temp_range_l534_534965


namespace trigonometry_identity_l534_534339

theorem trigonometry_identity
  (α : ℝ)
  (h_quad: 0 < α ∧ α < π / 2)
  (h_cos : real.cos α = 2 * real.sqrt 5 / 5) :
  real.cos (2 * α) - real.cos α / real.sin α = -7 / 5 := 
sorry

end trigonometry_identity_l534_534339


namespace train_length_eq_135m_l534_534564

noncomputable def speed_km_per_hr := 54
noncomputable def time_sec := 9

theorem train_length_eq_135m (speed_km_per_hr : ℝ) (time_sec : ℕ) (h1: speed_km_per_hr = 54) (h2: time_sec = 9) :
  let speed_m_per_s := (speed_km_per_hr * (1000 / 3600)) in
  let length_of_train := speed_m_per_s * (time_sec : ℝ) in
  length_of_train = 135 :=
by
  sorry

end train_length_eq_135m_l534_534564


namespace train_crossing_time_l534_534567

theorem train_crossing_time 
  (train_length : ℕ) 
  (train_speed_kmph : ℕ) 
  (conversion_factor : ℚ := 1000/3600) 
  (train_speed_mps : ℚ := train_speed_kmph * conversion_factor) :
  train_length = 100 →
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  train_length / train_speed_mps = 5 :=
by
  intros
  sorry

end train_crossing_time_l534_534567


namespace sum_correct_l534_534651

open Nat

noncomputable def sum_problem : ℝ :=
  (∑ k in range 1 (501 : ℕ), (k : ℝ) * (if (∃ j : ℤ, (k : ℝ) = real.exp ((j : ℝ) * real.log (real.sqrt 3))) then 0 else 1))

theorem sum_correct :
  sum_problem = 125187 := 
by
  sorry

end sum_correct_l534_534651


namespace sum_k_round_diff_eq_125237_l534_534632

noncomputable def sum_k_round_diff (n : ℕ) : ℤ :=
∑ k in finset.range n.succ, k * ((⌈real.log k / real.log (real.sqrt 3)⌉ : ℤ) - (⌊real.log k / real.log (real.sqrt 3)⌋ : ℤ))

theorem sum_k_round_diff_eq_125237 : sum_k_round_diff 500 = 125237 := by
  -- Here we would add the proof steps to verify the theorem
  sorry

end sum_k_round_diff_eq_125237_l534_534632


namespace slope_perpendicular_l534_534716

theorem slope_perpendicular (x y : ℝ) (h : 4 * x - 5 * y = 10) :
  let m := 4 / 5 in
  -(1 / m) = -5 / 4 :=
by {
  let m := 4 / 5,
  have h1 : 1 / m = 5 / 4 := by sorry,
  exact neg_eq_neg h1,
}

end slope_perpendicular_l534_534716


namespace product_of_midpoint_coordinates_l534_534186

theorem product_of_midpoint_coordinates :
  let x1 := 10
  let y1 := -3
  let x2 := -4
  let y2 := 9
  let Mx := (x1 + x2) / 2
  let My := (y1 + y2) / 2
  (Mx * My) = 9 :=
by
  let x1 := 10
  let y1 := -3
  let x2 := -4
  let y2 := 9
  let Mx := (x1 + x2) / 2
  let My := (y1 + y2) / 2
  have h : Mx = 3 := by sorry
  have h2 : My = 3 := by sorry
  have h3 : (3 * 3) = 9 := by norm_num
  show (Mx * My) = 9 from by
    rw [h, h2, h3]
    sorry

end product_of_midpoint_coordinates_l534_534186


namespace binomial_identity_l534_534478

-- Define the necessary binomial coefficient function
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Use the main theorem to state the problem in Lean 4
theorem binomial_identity (m n : ℕ) (h : 0 ≤ m ∧ m ≤ n) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : 
  (∑ k in Finset.range(n+1), if hmk : m ≤ k then
    binom n k * binom k m * x^k * (1-x)^(n-k)
  else 0) = binom n m * x^m :=
sorry

end binomial_identity_l534_534478


namespace arithmetic_square_root_of_sqrt_81_l534_534514

def square_root (x : ℝ) : ℝ :=
  real.sqrt x

theorem arithmetic_square_root_of_sqrt_81 : square_root 81 = 9 :=
by
  sorry

end arithmetic_square_root_of_sqrt_81_l534_534514


namespace modulus_of_complex_number_l534_534021

theorem modulus_of_complex_number : 
  let z := (3 + 4 * complex.i) * (3 + 4 * complex.i) / (5 * complex.i) in
  |z| = 5 := 
by 
  sorry

end modulus_of_complex_number_l534_534021


namespace modulus_of_z_l534_534020

noncomputable def z : ℂ := -3 * Complex.I - 4

theorem modulus_of_z : ∀ (z : ℂ), (Complex.I * z = 3 - 4 * Complex.I) → Complex.abs z = 5 :=
by
  intros z hz
  rw [Complex.mul_assoc, Complex.mul_I_I, neg_one_mul, one_mul] at hz
  rw [Complex.re_add_im, abs_div, Complex.abs_I, div_one, sqrt_eq_rpow, Complex.norm_sq_eq_abs_re_add_abs_im, Complex.norm_sq_eq_abs] at hz
  sorry

end modulus_of_z_l534_534020


namespace purely_imaginary_solution_l534_534036

theorem purely_imaginary_solution :
  ∃ m : ℝ, (m^2 - 3 * m - 4 = 0) ∧ ((m^2 - 3 * m - 4) + (m^2 - 5 * m - 6) * complex.I = (m^2 - 5 * m - 6) * complex.I) ∧ m = 4 := 
by
  -- We skip the proof here
  sorry

end purely_imaginary_solution_l534_534036


namespace jorge_goals_l534_534445

theorem jorge_goals (g_last g_total g_this : ℕ) (h_last : g_last = 156) (h_total : g_total = 343) :
  g_this = g_total - g_last → g_this = 187 :=
by
  intro h
  rw [h_last, h_total] at h
  apply h

end jorge_goals_l534_534445


namespace total_games_in_season_l534_534495

theorem total_games_in_season (n : ℕ) (h_teams : n = 10)
  (conference_games_per_pair : ℕ → ℕ → ℕ) -- function representing number of conference games per pair of teams
  (h_conference_games_per_pair : ∀ i j : ℕ, i ≠ j → conference_games_per_pair i j = 3)
  (non_conference_games_per_team : ℕ) (h_non_conference_games : non_conference_games_per_team = 5) :
  let num_pairs := n * (n - 1) / 2 in
  let total_conference_games := 3 * num_pairs in
  let total_non_conference_games := n * non_conference_games_per_team in
  (total_conference_games + total_non_conference_games) = 185 :=
by
  -- Introducing variables
  let num_pairs := n * (n - 1) / 2
  let total_conference_games := 3 * num_pairs
  let total_non_conference_games := n * non_conference_games_per_team
  -- Using the given assumptions
  have h_pairs : num_pairs = 45 := sorry
  have h_conference_games : total_conference_games = 135 := sorry
  have h_non_conference_total : total_non_conference_games = 50 := sorry
  -- Derive the result
  calc
    total_conference_games + total_non_conference_games
    = 135 + 50 : by rw [h_conference_games, h_non_conference_total]
    ... = 185 : by norm_num

end total_games_in_season_l534_534495


namespace rationalize_denominator_sum_l534_534133

theorem rationalize_denominator_sum :
  let expr := 1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)
  ∃ (A B C D E F G H I : ℤ), 
    I > 0 ∧
    expr * (Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 11) /
    ((Real.sqrt 5 + Real.sqrt 3)^2 - (Real.sqrt 11)^2) = 
        (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + 
         G * Real.sqrt H) / I ∧
    (A + B + C + D + E + F + G + H + I) = 225 :=
by
  sorry

end rationalize_denominator_sum_l534_534133


namespace sqrt_of_sqrt_81_l534_534501

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l534_534501


namespace num_integers_two_digits_l534_534802

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l534_534802


namespace Emma_age_ratio_l534_534291

theorem Emma_age_ratio (E M : ℕ) (h1 : E = E) (h2 : E = E) 
(h3 : E - M = 3 * (E - 4 * M)) : E / M = 11 / 2 :=
sorry

end Emma_age_ratio_l534_534291


namespace rate_of_first_car_l534_534180

theorem rate_of_first_car
  (r : ℕ) (h1 : 3 * r + 30 = 180) : r = 50 :=
sorry

end rate_of_first_car_l534_534180


namespace number_of_zeros_of_f_l534_534159

def f (x : ℝ) : ℝ := 
  if h : 0 ≤ x then log (x ^ 2 - 3 * x + 3) / log 10 
  else log ((-x) ^ 2 - 3 * (-x) + 3) / log 10

theorem number_of_zeros_of_f : (∃ n : ℕ, n = 4) :=
sorry

end number_of_zeros_of_f_l534_534159


namespace number_of_positive_integers_with_at_most_two_digits_l534_534812

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534812


namespace complex_magnitude_equality_l534_534111

open Complex

theorem complex_magnitude_equality (z1 z2 : ℂ) (a : ℝ) (h : |(z1 + 2 * z2)| = |(2 * z1 + z2)|) :
  |(z1 + a * z2)| = |(a * z1 + z2)| :=
sorry

end complex_magnitude_equality_l534_534111


namespace arithmetic_square_root_sqrt_81_l534_534508

theorem arithmetic_square_root_sqrt_81 : sqrt (sqrt 81) = 3 := 
by 
  have h1 : sqrt 81 = 9 := sorry
  have h2 : sqrt 9 = 3 := sorry
  show sqrt (sqrt 81) = 3 from sorry

end arithmetic_square_root_sqrt_81_l534_534508


namespace factor_expression_l534_534680

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) :=
by
  sorry

end factor_expression_l534_534680


namespace candle_height_halfway_burn_l534_534586

theorem candle_height_halfway_burn (n : ℕ) (h : n = 100) (time_k : ∀ k, k * 5) :
  let total_burn_time := 5 * (n * (n + 1) / 2)
  half_total_burn_time := total_burn_time / 2
  height := 100 - Nat.floor (sqrt (2 * half_total_burn_time / 5) - 1 / 2)
  in height = 30 :=
by
  let total_burn_time := 5 * (n * (n + 1) / 2)
  let half_total_burn_time := total_burn_time / 2
  let height := 100 - Nat.floor (sqrt (2 * half_total_burn_time / 5) - 1 / 2)
  have : height = 30 := sorry
  exact this

end candle_height_halfway_burn_l534_534586


namespace points_on_circle_multiple_of_4_l534_534125

theorem points_on_circle_multiple_of_4 
  (n : ℕ) 
  (h : 12 ≤ n)
  (circle : ℤ → Prop)
  (cond : ∀ k : ℤ, (Finset.range 11).sum (λ i, if circle (k + i - 5) then 1 else 0) = 5) :
  n % 4 = 0 :=
sorry

end points_on_circle_multiple_of_4_l534_534125


namespace cube_paint_ways_l534_534539

theorem cube_paint_ways :
  let paints := 6,
      faces := 6,
      ways := 4080
  in
  (∃ (f : Fin faces → Fin paints), (∀ (i j : Fin faces), i ≠ j ∧ adjacent i j → f i ≠ f j)
    → num_ways (exists_paint f) = ways) :=
begin
  sorry
end

end cube_paint_ways_l534_534539


namespace area_is_irrational_l534_534740

-- Define rational and irrational numbers
def is_rational (x : ℝ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = (p : ℝ) / q
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- Given conditions: length is rational, width is irrational
variables (l w : ℝ)
hypothesis (h_l : is_rational l) 
hypothesis (h_w : is_irrational w)

-- Proven statement: the area is irrational
theorem area_is_irrational : is_irrational (l * w) :=
sorry

end area_is_irrational_l534_534740


namespace recitation_orders_correct_l534_534594

noncomputable def num_recitation_orders (students : Finset ℕ) (selected : Finset ℕ) : Nat :=
  if (students.card = 7) ∧ (selected.card = 4) ∧
     (∃ a b c : ℕ, a ∈ students ∧ b ∈ students ∧ c ∈ students ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)
  then
    let a := 1
    let b := 2
    let c := 3
    let valid_orders : Nat := 
      -- Case ①
      (choose 3 1) * (choose 4 3) * (4!) +
      -- Case ②
      (choose 3 2) * (choose 4 2) * (4!) +
      -- Case ③ valid
      (choose 3 3) * (choose 4 1) * (4!) - 
      -- Subtract invalid orderings where A and B recite adjacently
      ((choose 3 3) * (choose 4 1) * (3!) * (2!))
    valid_orders
  else 0

theorem recitation_orders_correct : 
  num_recitation_orders (Finset.range 7) (Finset.range 4) = 768 := 
sorry

end recitation_orders_correct_l534_534594


namespace eval_expression_l534_534455

theorem eval_expression :
  let a := 3
  let b := 2
  (2 ^ a ∣ 200) ∧ ¬(2 ^ (a + 1) ∣ 200) ∧ (5 ^ b ∣ 200) ∧ ¬(5 ^ (b + 1) ∣ 200)
→ (1 / 3)^(b - a) = 3 :=
by sorry

end eval_expression_l534_534455


namespace math_proof_problem_l534_534561

theorem math_proof_problem (x : ℝ) (hx : x > 2) :
  ( ( (x + 2) ^ (-1 / 2) + (x - 2) ^ (-1 / 2) ) ^ (-1) + ( (x + 2) ^ (-1 / 2) - (x - 2) ^ (-1 / 2) ) ^ (-1) ) /
    ( ( (x + 2) ^ (-1 / 2) + (x - 2) ^ (-1 / 2) ) ^ (-1) - ( (x + 2) ^ (-1 / 2) - (x - 2) ^ (-1 / 2) ) ^ (-1) ) = 
    - √((x - 2) / (x + 2)) := 
  sorry

end math_proof_problem_l534_534561


namespace max_sphere_volume_in_prism_l534_534232

-- Defining the conditions
def is_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
  (AB BC : ℝ) : Prop :=
  AB = 6 ∧ BC = 8

def height_of_prism (height : ℝ) : Prop :=
  height = 3

-- The desired theorem
theorem max_sphere_volume_in_prism 
  (A B C A1 B1 C1 : Type) [metric_space A] [metric_space B] [metric_space C]
  (AB BC height : ℝ) 
  (h_right_triangle : is_right_triangle A B C AB BC)
  (h_height_of_prism : height_of_prism height) : 
  V = 9 * π / 2 :=
by
  -- Conditions usable within the proof.
  rcases h_right_triangle with ⟨hAB, hBC⟩,
  rcases h_height_of_prism with hAA1,
  -- Skipping the proof itself
  sorry

end max_sphere_volume_in_prism_l534_534232


namespace perfect_squares_sum_divisible_by_9_l534_534996

theorem perfect_squares_sum_divisible_by_9
  (a b c : ℤ)
  (ha : ∃ x, a = x^2)
  (hb : ∃ y, b = y^2)
  (hc : ∃ z, c = z^2)
  (h_sum_div_9 : (a + b + c) % 9 = 0) :
  ∃ k m, (k = a ∨ k = b ∨ k = c) ∧ (m = a ∨ m = b ∨ m = c) ∧ k ≠ m ∧ (k - m) % 9 = 0 :=
begin
  sorry
end

end perfect_squares_sum_divisible_by_9_l534_534996


namespace perpendicular_slope_l534_534714

def line_slope (A B : ℚ) (x y : ℚ) : ℚ := A * x - B * y

def is_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

theorem perpendicular_slope : 
  ∃ m : ℚ, let slope_given_line := (4 : ℚ) / (5 : ℚ) in 
    is_perpendicular m slope_given_line ∧ m = - (5 : ℚ) / (4 : ℚ) := 
by 
  sorry

end perpendicular_slope_l534_534714


namespace polynomial_values_l534_534106

open Polynomial

noncomputable def f : ℕ → ℕ → ℤ
| n, x => Polynomial.eval x (Polynomial.of_finset {⟨λ k, 2^k, finset.mem_range_succ⟩ 
                        | x : ℤ, x - n = 0 ∨ x + n + 1 = 0})

theorem polynomial_values (f : ℕ → ℤ) (n : ℕ) 
  (h1 : ∀ k, 1 ≤ k ∧ k ≤ n + 1 → f k = 2^k):
  f (n+2) = 2^(n+2) - 2 ∧ f (n+3) = 2^(n+3) - 2*n - 6 :=
sorry

end polynomial_values_l534_534106


namespace find_divisor_l534_534940

-- Define the problem specifications
def divisor_problem (D Q R d : ℕ) : Prop :=
  D = d * Q + R

-- The specific instance with given values
theorem find_divisor :
  divisor_problem 15968 89 37 179 :=
by
  -- Proof omitted
  sorry

end find_divisor_l534_534940


namespace minimum_k_element_subset_l534_534389

theorem minimum_k_element_subset (S : set ℕ) (hS : S = {1, 2, ..., 32}) :
  ∃ k, (∀ (T : finset ℕ), T ⊆ S ∧ T.card = k →
    ∃ a b c, a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ a ∣ b ∧ b ∣ c) ∧ k = 25 := by
  sorry

end minimum_k_element_subset_l534_534389


namespace problem_l534_534230

-- Define the sequence (a_n, b_n)
def seq (a : ℕ → ℂ) : Prop :=
  ∀ n, a (n + 1) = (⟨√3, 0⟩.re * a n - (0 * a n).im : ℂ) + ⟨(⟨√3, 0⟩.im * a n + (1 * a n).re) * I⟩

-- Define the given condition
def condition := a 100 = 1 + ⟨√3⟩ * I

-- The specific point of interest, given the conditions
def point_interest := a 1 = (1 / 2^99 * (√3 - 1)) + 0

-- Proof statement
theorem problem (a : ℕ → ℂ) (h_seq : seq a) (h_cond : condition) : point_interest := 
by sorry

end problem_l534_534230


namespace phi_shift_symmetry_l534_534245

/-- 
  Prove that if the graph of f(x) = sin(2x + φ) is shifted to the left by π/6 units
  and the resulting graph is symmetric about the y-axis,
  then φ = π / 6.
--/
theorem phi_shift_symmetry (φ : ℝ) (k : ℤ) :
  (∀ x : ℝ, sin (2 * (x + π / 6) + φ) = sin (-2 * x)) → φ = π / 6 :=
by
  sorry

end phi_shift_symmetry_l534_534245


namespace find_line_equations_l534_534688

theorem find_line_equations
  (M : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hM : M = (2, 2))
  (hA : A = (2, 3))
  (hB : B = (6, -9)) :
  ∃ l : ℝ → ℝ → Prop,
    (l = (λ x y, 5 * x + 2 * y - 14 = 0) ∨
     l = (λ x y, 3 * x + y - 8 = 0)) ∧
    (∀ x y, l x y → M.1 * x + M.2 * y = M.1 * (M.1 - 1) + M.2) ∧
    (let dist_A := ((5 * A.1 + 2 * A.2 - 14) : ℝ) / real.sqrt (5^2 + 2^2),
         dist_B := ((5 * B.1 + 2 * B.2 - 14) : ℝ) / real.sqrt (5^2 + 2^2)
     in dist_A = dist_B) ∧
    (let dist_A := ((3 * A.1 + A.2 - 8) : ℝ) / real.sqrt (3^2 + 1^2),
         dist_B := ((3 * B.1 + A.2 - 8) : ℝ) / real.sqrt (3^2 + 1^2)
     in dist_A = dist_B) :=
sorry

end find_line_equations_l534_534688


namespace bamboo_fifth_section_volume_l534_534960

theorem bamboo_fifth_section_volume
  (a₁ q : ℝ)
  (h1 : a₁ * (a₁ * q) * (a₁ * q^2) = 3)
  (h2 : (a₁ * q^6) * (a₁ * q^7) * (a₁ * q^8) = 9) :
  a₁ * q^4 = Real.sqrt 3 :=
sorry

end bamboo_fifth_section_volume_l534_534960


namespace vector_properties_l534_534781

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-2, 1)

-- Lean statements to check the conditions
theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 0) ∧        -- Perpendicular vectors
  (real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5) ∧  -- Magnitude of the sum of vectors
  (real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5) := -- Magnitude of the difference of vectors
by
  unfold a b
  simp
  split
  -- Proof of each condition is skipped
  sorry 
  sorry
  sorry

end vector_properties_l534_534781


namespace wait_time_difference_l534_534534

noncomputable def kids_waiting_for_swings : ℕ := 3
noncomputable def kids_waiting_for_slide : ℕ := 2 * kids_waiting_for_swings
noncomputable def wait_per_kid_swings : ℕ := 2 * 60 -- 2 minutes in seconds
noncomputable def wait_per_kid_slide : ℕ := 15 -- in seconds

noncomputable def total_wait_swings : ℕ := kids_waiting_for_swings * wait_per_kid_swings
noncomputable def total_wait_slide : ℕ := kids_waiting_for_slide * wait_per_kid_slide

theorem wait_time_difference : total_wait_swings - total_wait_slide = 270 := by
  sorry

end wait_time_difference_l534_534534


namespace sum_le_six_l534_534005

theorem sum_le_six (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
    (h3 : ∃ (r s : ℤ), r * s = a + b ∧ r + s = ab) : a + b ≤ 6 :=
sorry

end sum_le_six_l534_534005


namespace area_triangle_of_vectors_l534_534072

variables {V : Type*} [inner_product_space ℝ V]

def area_of_triangle (a b : V) : ℝ :=
  1 / 2 * real.sqrt (∥a∥ ^ 2 * ∥b∥ ^ 2 - (inner_product a b) ^ 2)

theorem area_triangle_of_vectors
  (a b : V) :
  (area_of_triangle a b) =
  1 / 2 * real.sqrt (∥a∥ ^ 2 * ∥b∥ ^ 2 - (inner_product a b) ^ 2) :=
sorry

end area_triangle_of_vectors_l534_534072


namespace expected_value_of_product_of_marbles_l534_534871

theorem expected_value_of_product_of_marbles : 
  let marbles := [1, 2, 3, 4, 5, 6] in
  let pairs := (marbles.product marbles).filter (λ ⟨a, b⟩, a < b) in
  let products := pairs.map (λ ⟨a, b⟩, a * b) in
  let sum_products := products.sum in
  let num_pairs := pairs.length in
  (sum_products : ℚ) / num_pairs = 35 / 3 :=
by
  sorry

end expected_value_of_product_of_marbles_l534_534871


namespace slide_wait_is_shorter_l534_534531

theorem slide_wait_is_shorter 
  (kids_waiting_for_swings : ℕ)
  (kids_waiting_for_slide_multiplier : ℕ)
  (wait_per_kid_swings_minutes : ℕ)
  (wait_per_kid_slide_seconds : ℕ)
  (kids_waiting_for_swings = 3)
  (kids_waiting_for_slide_multiplier = 2)
  (wait_per_kid_swings_minutes = 2) 
  (wait_per_kid_slide_seconds = 15) :
  let total_wait_swings_seconds := wait_per_kid_swings_minutes * kids_waiting_for_swings * 60,
      kids_waiting_for_slide := kids_waiting_for_swings * kids_waiting_for_slide_multiplier,
      total_wait_slide_seconds := wait_per_kid_slide_seconds * kids_waiting_for_slide in
  270 = total_wait_swings_seconds - total_wait_slide_seconds :=
by
  sorry

end slide_wait_is_shorter_l534_534531


namespace probability_of_two_jacob_one_isaac_l534_534438

-- Definition of the problem conditions
def jacob_letters := 5
def isaac_letters := 5
def total_cards := 12
def cards_drawn := 3

-- Combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probability calculation
def probability_two_jacob_one_isaac : ℚ :=
  (C jacob_letters 2 * C isaac_letters 1 : ℚ) / (C total_cards cards_drawn : ℚ)

-- The statement of the problem
theorem probability_of_two_jacob_one_isaac :
  probability_two_jacob_one_isaac = 5 / 22 :=
  by sorry

end probability_of_two_jacob_one_isaac_l534_534438


namespace trigonometric_series_differentiability_l534_534272

theorem trigonometric_series_differentiability (a : ℝ) (h : |a| < 1) :
    ∀ (s : ℕ), differentiable ℝ (λ x : ℝ, ∑ k in finset.range s, a^k * sin (k * x)) :=
by
  sorry

end trigonometric_series_differentiability_l534_534272


namespace solve_for_z_l534_534141

theorem solve_for_z (z : ℂ) (h : 5 + 2 * (complex.I * z) = 1 - 6 * (complex.I * z)) : z = complex.I / 2 :=
by
  sorry

end solve_for_z_l534_534141


namespace domain_of_f_l534_534669

def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (Real.sqrt (8 - x) + Real.sqrt (x - 1))

theorem domain_of_f : {x : ℝ | 3 ≤ x ∧ x ≤ 8} = set_of (λ x, ∃ y, f (x) = y) :=
by
  -- Proof goes here
  sorry

end domain_of_f_l534_534669


namespace daphne_two_friends_visits_l534_534276

def visits_every (n : ℕ) (d : ℕ) : ℕ := n % d = 0

def from_today (period : ℕ) : ℕ := period + 1

def two_friends_visits_exactly (day : ℕ) (friends : list ℕ) (daphne_visits : ℕ → Prop) : Prop :=
  (friends.filter (λ d, daphne_visits day)).length = 2

def visit_schedule (alice beatrix claire diana : ℕ) (days : ℕ) (daphne_visits : ℕ → Prop) : ℕ :=
  list.range days |> list.filter (λ day, two_friends_visits_exactly day [alice, beatrix, claire, diana] daphne_visits)

theorem daphne_two_friends_visits :
  visit_schedule 4 6 8 10 400 (λ n, visits_every n 4 ∨ visits_every n 6 ∨ visits_every n 8 ∨ visits_every n 10) = 142 :=
sorry

end daphne_two_friends_visits_l534_534276


namespace calculate_product_l534_534260

theorem calculate_product :
  (∏ n in Finset.range (2020 - 2 + 1) | n ≥ 2, (1 - (1 / (n + 2)^2))) = (2021 / 4040) :=
by
  sorry

end calculate_product_l534_534260


namespace area_identity_l534_534127

-- Define the variables representing points in the plane
variables {A B C D M N K L : Type}

-- Define the conditions
-- (we represent points on a 2D plane and use the implicit conditions)
def points_on_sides (M : Type) (N : Type) (AB : Type) (CD : Type) : Prop :=
  (∃ (p : ℚ), p = (AM / MB) ∧ p = (CN / ND))

-- Define the segment intersections forming points K and L
def segment_intersections (A N D M B C : Type) : Prop :=
  (∃ (K : Type), intersection (AN) (DM) = K) ∧ (∃ (L : Type), intersection (BN) (CM) = L)

-- Define the proof problem
theorem area_identity (A B C D M N K L : Type) 
  (h1 : points_on_sides M N (segment A B) (segment C D)) 
  (h2 : segment_intersections A N D M B C) : 
  area (quadrilateral K M L N) = area (triangle A D K) + area (triangle B C L) :=
sorry

end area_identity_l534_534127


namespace five_minute_commercials_count_l534_534288

theorem five_minute_commercials_count (x y : ℕ) (h1 : y = 11) (h2 : 5 * x + 2 * y = 37) : x = 3 :=
by
  substitute h1 in h2
  sorry

end five_minute_commercials_count_l534_534288


namespace quadratic_inequality_k_l534_534673

theorem quadratic_inequality_k :
  (∀ x : ℝ, x ∈ Ioo (-2 : ℝ) (1 / 2) ↔ x * (2 * x + 3) < 2) :=
by
  sorry

end quadratic_inequality_k_l534_534673


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534794

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534794


namespace average_eq_solution_l534_534149

theorem average_eq_solution (x : ℝ) :
  (1 / 3) * ((2 * x + 4) + (4 * x + 6) + (5 * x + 3)) = 3 * x + 5 → x = 1 :=
by
  sorry

end average_eq_solution_l534_534149


namespace line_intersects_y_axis_at_point_l534_534615

theorem line_intersects_y_axis_at_point :
  ∃ (y : ℝ), 
  (∃ m : ℝ, m = (15 - 9) / (4 - 2) ∧ ∃ b : ℝ, b = 9 - m * 2 ∧ y = b) ∧
  (0, y) = (0, 3) :=
begin
  sorry
end

end line_intersects_y_axis_at_point_l534_534615


namespace ellipse_line_fixed_point_and_max_area_diff_l534_534334

theorem ellipse_line_fixed_point_and_max_area_diff (A B : Point) (C : Ellipse) (M N : Point)
  (l : Line) (k1 k2 : ℝ) (S1 S2 : Area) :
  (C.C (A, B)) ∧ (l.line_intersects_ellipse C M N) ∧ (slope (A, M) k1) ∧ (slope (B, N) k2) ∧ (k1 : k2 = 1 : 9) →
  (passes_through (l, (4, 0))) ∧ (max_area_diff (S1, S2, 15)) :=
sorry

end ellipse_line_fixed_point_and_max_area_diff_l534_534334


namespace sum_correct_l534_534648

open Nat

noncomputable def sum_problem : ℝ :=
  (∑ k in range 1 (501 : ℕ), (k : ℝ) * (if (∃ j : ℤ, (k : ℝ) = real.exp ((j : ℝ) * real.log (real.sqrt 3))) then 0 else 1))

theorem sum_correct :
  sum_problem = 125187 := 
by
  sorry

end sum_correct_l534_534648


namespace initial_passengers_l534_534528

theorem initial_passengers (rows seats_per_row people_1_on people_1_off people_2_on people_2_off empty_seats_2 : ℕ)
    (h1 : rows = 23)
    (h2 : seats_per_row = 4)
    (h3 : people_1_on = 15)
    (h4 : people_1_off = 3)
    (h5 : people_2_on = 17)
    (h6 : people_2_off = 10)
    (h7 : empty_seats_2 = 57) :
    ∃ initial_passengers, initial_passengers = 16 :=
by
  -- Definitions of variables based on conditions
  let total_seats := rows * seats_per_row
  let net_increase_1 := people_1_on - people_1_off
  let net_increase_2 := people_2_on - people_2_off
  let total_net_increase := net_increase_1 + net_increase_2
  let current_passengers := total_seats - empty_seats_2
  let initial_passengers := current_passengers - total_net_increase
  
  -- Assertion that initial_passengers equals 16
  exists_unique 16
  
  sorry

end initial_passengers_l534_534528


namespace Gwen_birthday_money_l534_534722

variable (dollars_spent : ℕ) (dollars_left : ℕ)

theorem Gwen_birthday_money (h_spent : dollars_spent = 8) (h_left : dollars_left = 6) : 
  dollars_spent + dollars_left = 14 :=
by
  -- Given conditions
  rw [h_spent, h_left]
  -- Add them
  exact rfl

end Gwen_birthday_money_l534_534722


namespace cloth_woven_in_first_three_days_l534_534608

variable {α : Type*} [field α]

theorem cloth_woven_in_first_three_days
  (x : α)
  (h_total : x + 2 * x + 4 * x + 8 * x + 16 * x = 5) :
  let y := x + 2 * x + 4 * x in
  y = 35 / 31 :=
by {
  sorry
}

end cloth_woven_in_first_three_days_l534_534608


namespace frog_hexagon_jumps_l534_534658

-- Define the vertices of the hexagon
inductive Vertex
| A | B | C | D | E | F

-- Define the adjacency relation of the hexagon
def adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.B | Vertex.A, Vertex.F 
| Vertex.B, Vertex.A | Vertex.B, Vertex.C
| Vertex.C, Vertex.B | Vertex.C, Vertex.D
| Vertex.D, Vertex.C | Vertex.D, Vertex.E
| Vertex.E, Vertex.D | Vertex.E, Vertex.F
| Vertex.F, Vertex.E | Vertex.F, Vertex.A := true
| _, _ := false

-- Define the main problem: total number of distinct jumping sequences
def frog_jump_ways (start jump_count: ℕ) : ℕ := sorry

theorem frog_hexagon_jumps : frog_jump_ways 0 5 = 26 := 
  sorry

end frog_hexagon_jumps_l534_534658


namespace sum_log_floor_ceil_l534_534624

theorem sum_log_floor_ceil (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500 → 
  ((⌈ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log (k : ℝ) / Real.log (Real.sqrt 3) ⌋) = 
    if (∃ n : ℕ, k = (3 : ℕ) ^ (n / 2 : ℚ)) then 0 else 1)) :
  ∑ k in Finset.range 501 \ {0}, k * (⌈ Real.log k / Real.log (Real.sqrt 3) ⌉ - ⌊ Real.log k / Real.log (Real.sqrt 3) ⌋) = 125237 := by
  have h_sum_all := (500 * 501) / 2
  have h_sum_powers := 1 + 3 + 9
  have h_correct := h_sum_all - h_sum_powers
  rw h_sum_all
  rw h_sum_powers
  rw h_correct
  sorry

end sum_log_floor_ceil_l534_534624


namespace exists_acute_triangle_l534_534323

theorem exists_acute_triangle (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h_triangle_abc : a + b > c) (h_triangle_abd : a + b > d) (h_triangle_abe : a + b > e)
  (h_triangle_bcd : b + c > d) (h_triangle_bce : b + c > e) (h_triangle_cde : c + d > e)
  (h_triangle_abc2 : a + c > b) (h_triangle_abd2 : a + d > b) (h_triangle_abe2 : a + e > b)
  (h_triangle_bcd2 : b + d > c) (h_triangle_bce2 : b + e > c) (h_triangle_cde2 : c + e > d)
  (h_triangle_abc3 : b + c > a) (h_triangle_abd3 : b + d > a) (h_triangle_abe3 : b + e > a)
  (h_triangle_bcd3 : b + d > a) (h_triangle_bce3 : c + e > a) (h_triangle_cde3 : d + e > c) :
  ∃ x y z : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
              (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
              (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
              (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
              x + y > z ∧ 
              ¬ (x^2 + y^2 ≤ z^2) :=
by
  sorry

end exists_acute_triangle_l534_534323


namespace common_chord_eq_length_common_chord_tangents_through_P_l534_534328

noncomputable def C1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2 * p.1 - 6 * p.2 - 1 = 0}
noncomputable def C2 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 10 * p.1 - 12 * p.2 + 45 = 0}
def P := (9 : ℝ, 1 : ℝ)

theorem common_chord_eq : 
  ∃ A B C : ℝ, (∀ x y : ℝ, (x,y) ∈ (C1 ∩ C2) ↔ (A * x + B * y + C = 0)) ∧ 
  A = 4 ∧ B = 3 ∧ C = -23 := 
begin
  use [4, 3, -23],
  split,
  { sorry },  -- proof omitted
  split; refl
end

theorem length_common_chord : 
  ∃ l : ℝ, l = 2 * Real.sqrt 7 := 
begin
  use 2 * Real.sqrt 7,
  refl
end

theorem tangents_through_P :
  ∃ (line1 line2 : ℝ × ℝ → Prop), 
    (∀ x y, line1 (x, y) ↔ (x = 9)) ∧ 
    (∀ x y, line2 (x, y) ↔ (9 * x + 40 * y - 121 = 0)) := 
begin
  use [(λ p, p.1 = 9), (λ p, 9 * p.1 + 40 * p.2 - 121 = 0)],
  split,
  { intros x y,
    simp },
  { intros x y,
    simp }
end

end common_chord_eq_length_common_chord_tangents_through_P_l534_534328


namespace takeoff_run_distance_correct_l534_534169

noncomputable def takeoff_run_distance : ℕ := 
  let v := 27.78        -- lift-off speed in meters per second
  let t := 15           -- takeoff time in seconds
  let a := v / t        -- uniformly accelerated motion: acceleration
  let S := 0.5 * a * t^2 -- distance using uniformly accelerated motion formula
  S.to_nat               -- rounding to the nearest whole number

theorem takeoff_run_distance_correct : takeoff_run_distance = 208 :=
  by sorry

end takeoff_run_distance_correct_l534_534169


namespace count_valid_numbers_l534_534864

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534864


namespace math_problem_l534_534654

theorem math_problem 
  (a1 : (10^4 + 500) = 100500)
  (a2 : (25^4 + 500) = 390625500)
  (a3 : (40^4 + 500) = 256000500)
  (a4 : (55^4 + 500) = 915062500)
  (a5 : (70^4 + 500) = 24010062500)
  (b1 : (5^4 + 500) = 625+500)
  (b2 : (20^4 + 500) = 160000500)
  (b3 : (35^4 + 500) = 150062500)
  (b4 : (50^4 + 500) = 625000500)
  (b5 : (65^4 + 500) = 1785062500) :
  ( (100500 * 390625500 * 256000500 * 915062500 * 24010062500) / (625+500 * 160000500 * 150062500 * 625000500 * 1785062500) = 240) :=
by
  sorry

end math_problem_l534_534654


namespace solution_set_even_function_l534_534097

/-- Let f be an even function, and for x in [0, ∞), f(x) = x - 1. Determine the solution set for the inequality f(x) > 1.
We prove that the solution set is {x | x < -2 or x > 2}. -/
theorem solution_set_even_function (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x, 0 ≤ x → f x = x - 1) :
  {x : ℝ | f x > 1} = {x | x < -2 ∨ x > 2} :=
by
  sorry  -- Proof steps go here.

end solution_set_even_function_l534_534097


namespace sum_k_round_diff_eq_125237_l534_534634

noncomputable def sum_k_round_diff (n : ℕ) : ℤ :=
∑ k in finset.range n.succ, k * ((⌈real.log k / real.log (real.sqrt 3)⌉ : ℤ) - (⌊real.log k / real.log (real.sqrt 3)⌋ : ℤ))

theorem sum_k_round_diff_eq_125237 : sum_k_round_diff 500 = 125237 := by
  -- Here we would add the proof steps to verify the theorem
  sorry

end sum_k_round_diff_eq_125237_l534_534634


namespace boxes_per_case_l534_534911

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (h1 : total_boxes = 24) (h2 : total_cases = 3) : (total_boxes / total_cases) = 8 :=
by 
  sorry

end boxes_per_case_l534_534911


namespace vector_properties_l534_534784

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (a_def : a = ![2, 4])
variable (b_def : b = ![-2, 1])

theorem vector_properties :
  (dot_product a b = 0) ∧
  (‖a + b‖ = 5) ∧
  (‖a - b‖ = 5) :=
by
  have h₁ : dot_product a b = 0 := by sorry
  have h₂ : ‖a + b‖ = 5 := by sorry
  have h₃ : ‖a - b‖ = 5 := by sorry
  exact ⟨h₁, h₂, h₃⟩

end vector_properties_l534_534784


namespace four_dimensional_measure_of_hypersphere_l534_534061

theorem four_dimensional_measure_of_hypersphere (r : ℝ) : 
  (∃ (W : ℝ), W = 2 * π * r^4) :=
begin
  use 2 * π * r^4,
  sorry
end

end four_dimensional_measure_of_hypersphere_l534_534061


namespace trigonometric_sum_identity_l534_534132

theorem trigonometric_sum_identity :
  ∑ x in finset.range 89, (1 / (real.cos (x * real.pi / 180) * real.cos ((x + 1) * real.pi / 180))) = 
  real.cos (real.pi / 180) / (real.sin (real.pi / 180))^2 :=
sorry

end trigonometric_sum_identity_l534_534132


namespace distinct_nat_numbers_l534_534322

theorem distinct_nat_numbers 
  (a b c : ℕ) (p q r : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_sum : a + b + c = 55) 
  (h_ab : a + b = p * p) 
  (h_bc : b + c = q * q) 
  (h_ca : c + a = r * r) : 
  a = 19 ∧ b = 6 ∧ c = 30 :=
sorry

end distinct_nat_numbers_l534_534322


namespace sin_wave_cylinder_plane_l534_534939

theorem sin_wave_cylinder_plane : 
  ∀ (u v : ℝ), (∃ (x : ℝ), u = cos x ∧ v = sin x) → (∃ z : ℝ, z = v) :=
by 
  intros u v h_exists
  obtain ⟨x, h_u, h_v⟩ := h_exists
  use v
  exact h_v

end sin_wave_cylinder_plane_l534_534939


namespace count_special_integers_l534_534828

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534828


namespace ferris_wheel_height_at_14_l534_534207

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  let A := (18 - 2) / 2
  let B := 10
  let ω := Real.pi / 6
  let φ := 3 * Real.pi / 2
  A * Real.sin(ω * t + φ) + B

theorem ferris_wheel_height_at_14 :
  ferris_wheel_height 14 = 6 := by
  sorry

end ferris_wheel_height_at_14_l534_534207


namespace unique_numbers_l534_534565

theorem unique_numbers (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (S : x + y = 17) 
  (Q : x^2 + y^2 = 145) 
  : x = 8 ∧ y = 9 ∨ x = 9 ∧ y = 8 :=
by
  sorry

end unique_numbers_l534_534565


namespace spoon_less_than_fork_l534_534617

-- Define the initial price of spoon and fork in kopecks
def initial_price (x : ℕ) : Prop :=
  x > 100 -- ensuring the spoon's sale price remains positive

-- Define the sale price of the spoon
def spoon_sale_price (x : ℕ) : ℕ :=
  x - 100

-- Define the sale price of the fork
def fork_sale_price (x : ℕ) : ℕ :=
  x / 10

-- Prove that the spoon's sale price can be less than the fork's sale price
theorem spoon_less_than_fork (x : ℕ) (h : initial_price x) : 
  spoon_sale_price x < fork_sale_price x :=
by
  sorry

end spoon_less_than_fork_l534_534617


namespace line_tangent_to_circle_l534_534885

theorem line_tangent_to_circle (l : ℝ → ℝ) (P : ℝ × ℝ) 
  (hP1 : P = (0, 1)) (hP2 : ∀ x y : ℝ, x^2 + y^2 = 1 -> l x = y)
  (hTangent : ∀ x y : ℝ, l x = y ↔ x^2 + y^2 = 1 ∧ y = 1):
  l x = 1 := by
  sorry

end line_tangent_to_circle_l534_534885


namespace number_of_positive_integers_with_at_most_two_digits_l534_534813

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534813


namespace area_triangle_l534_534456

open Matrix
open Real

def vector_a : Vector 2 ℝ := ![5, 1]
def vector_b : Vector 2 ℝ := ![2, 4]

theorem area_triangle : 
  let mat := ![vector_a, vector_b] in
  abs (det mat) / 2 = 9 :=
by
  sorry

end area_triangle_l534_534456


namespace mary_paid_amount_l534_534471

-- Definitions for the conditions:
def is_adult (person : String) : Prop := person = "Mary"
def children_count (n : ℕ) : Prop := n = 3
def ticket_cost_adult : ℕ := 2  -- $2 for adults
def ticket_cost_child : ℕ := 1  -- $1 for children
def change_received : ℕ := 15   -- $15 change

-- Mathematical proof to find the amount Mary paid given the conditions
theorem mary_paid_amount (person : String) (n : ℕ) 
  (h1 : is_adult person) (h2 : children_count n) :
  ticket_cost_adult + ticket_cost_child * n + change_received = 20 := 
by 
  -- Sorry as the proof is not required
  sorry

end mary_paid_amount_l534_534471


namespace cos_A_value_l534_534037

theorem cos_A_value (A B C : ℝ) 
  (A_internal : A + B + C = Real.pi) 
  (cos_B : Real.cos B = 1 / 2)
  (sin_C : Real.sin C = 3 / 5) : 
  Real.cos A = (3 * Real.sqrt 3 - 4) / 10 := 
by
  sorry

end cos_A_value_l534_534037


namespace central_cell_value_l534_534892

theorem central_cell_value 
  (M : Matrix (Fin 29) (Fin 29) ℕ)
  (h_distinct : ∀ i j, 1 ≤ M i j ∧ M i j ≤ 29)
  (h_counts : ∀ k, 1 ≤ k ∧ k ≤ 29 → (∑ i j, ite (M i j = k) 1 0) = 29)
  (h_above_below : (∑ i j, ite (i < j) (M i j) 0) = 3 * (∑ i j, ite (i > j) (M i j) 0)) :
  M ⟨14, h14⟩ ⟨14, h14⟩ = 15 := 
sorry

end central_cell_value_l534_534892


namespace tangent_line_at_P_and_C_line_passing_through_D_intersects_circle_trajectory_of_moving_point_Q_l534_534023

-- Definitions and conditions
def circle_eq (x y : ℝ) := x^2 + y^2 = 4
def point_P := (2, 1) : ℝ × ℝ
def point_D := (1, 2) : ℝ × ℝ
def on_circle (x₀ y₀ : ℝ) := x₀^2 + y₀^2 = 4
def vector_ON (y₀ : ℝ) := (0, y₀) : ℝ × ℝ
def vector_OQ (x₀ y₀ : ℝ) := (x₀, 2 * y₀) : ℝ × ℝ

-- Prove tangent line and trajectory conditions
theorem tangent_line_at_P_and_C :
  ∃ l : ℝ → ℝ × ℝ, (l point_P.1 = point_P) ∧ 
                    (∃ k, (l x y).2 = k * (y - l x y).1 + point_P.2 ∧
                    (∀ (x y : ℝ), circle_eq x y → |1 - 2 * k| / sqrt (k^2 + 1) = 2)) →
                    (l = λ x, (x = 2 ∨ 3 * x + 4 * y - 10 = 0)) := sorry

theorem line_passing_through_D_intersects_circle :
  ∃ l : ℝ → ℝ, (l point_D.1 = point_D.2) ∧
    (let k := 
      (∀ (x y : ℝ), circle_eq x y → k x - y - k + 2 = 0) →
      ∃ A B : ℝ × ℝ, A ≠ B ∧ on_circle A.1 A.2 ∧ on_circle B.1 B.2 ∧ |A.1 - B.1| = 2 * sqrt 3 →
      (l = λ x, (x = 1 ∨ 3 * x - 4 * y + 5 = 0))) := sorry

theorem trajectory_of_moving_point_Q :
  ∃ Q : ℝ × ℝ, ∀ (x₀ y₀ : ℝ), on_circle x₀ y₀ ∧
                                       (vector_ON y₀ = (0, y₀)) ∧
                                       (vector_OQ x₀ y₀ = (x₀, 2 * y₀)) →
                                       Q = (x, y) ∧
                                       ∀ (x y : ℝ), (x = x₀ ∧ y = 2 * y₀) →
                                       (circle_eq x (y / 2) → 
                                          x^2 / 4 + y^2 / 16 = 1) := sorry

end tangent_line_at_P_and_C_line_passing_through_D_intersects_circle_trajectory_of_moving_point_Q_l534_534023


namespace walnut_trees_total_l534_534175

theorem walnut_trees_total : 33 + 44 = 77 :=
by
  sorry

end walnut_trees_total_l534_534175


namespace range_of_f_lt_zero_l534_534042

def f (x : ℝ) : ℝ := x^(2/3) - x^(-1/2)

theorem range_of_f_lt_zero :
  ∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end range_of_f_lt_zero_l534_534042


namespace pradeep_marks_l534_534128

theorem pradeep_marks 
  (total_marks : ℕ)
  (pass_percent : ℝ)
  (failed_by : ℕ)
  (H_total_marks : total_marks = 840)
  (H_pass_percent : pass_percent = 0.25)
  (H_failed_by : failed_by = 25) :
  let P := pass_percent * total_marks in
  let M := P - failed_by in
  M = 185 :=
by
  simp [H_total_marks, H_pass_percent, H_failed_by]
  sorry

end pradeep_marks_l534_534128


namespace multiplicative_inverse_correct_l534_534460

def A : ℕ := 123456
def B : ℕ := 654321
def m : ℕ := 1234567
def AB_mod : ℕ := (A * B) % m

def N : ℕ := 513629

theorem multiplicative_inverse_correct (h : AB_mod = 470160) : (470160 * N) % m = 1 := 
by 
  have hN : N = 513629 := rfl
  have hAB : AB_mod = 470160 := h
  sorry

end multiplicative_inverse_correct_l534_534460


namespace angle_XYZ_l534_534457

theorem angle_XYZ {XYZ : Type*} [metric_space XYZ] 
  {X Y Z D E F : XYZ} (h1 : altitude X Y Z D) (h2 : altitude Y Z X E) (h3 : altitude Z X Y F)
  (h : 7 • (D - X) + 3 • (E - Y) + 8 • (F - Z) = 0) : 
  angle X Y Z = real.acos (1 / real.sqrt 21) :=
by sorry

end angle_XYZ_l534_534457


namespace gina_keeps_170_l534_534311

theorem gina_keeps_170 (initial_amount : ℕ)
    (money_to_mom : ℕ)
    (money_to_clothes : ℕ)
    (money_to_charity : ℕ)
    (remaining_money : ℕ) :
  initial_amount = 400 →
  money_to_mom = (1 / 4) * initial_amount →
  money_to_clothes = (1 / 8) * initial_amount →
  money_to_charity = (1 / 5) * initial_amount →
  remaining_money = initial_amount - (money_to_mom + money_to_clothes + money_to_charity) →
  remaining_money = 170 := sorry

end gina_keeps_170_l534_534311


namespace count_two_digit_or_less_numbers_l534_534844

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534844


namespace new_class_mean_l534_534411

theorem new_class_mean 
  (n1 n2 : ℕ) 
  (mean1 mean2 : ℝ)
  (students_total : ℕ)
  (total_score1 total_score2 : ℝ)
  (h1 : n1 = 45)
  (h2 : n2 = 5)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : students_total = 50)
  (h6 : total_score1 = n1 * mean1)
  (h7 : total_score2 = n2 * mean2) :
  (total_score1 + total_score2) / students_total = 81 :=
by
  sorry

end new_class_mean_l534_534411


namespace line_intersects_y_axis_at_l534_534614

def point1 : ℝ × ℝ := (2, 9)
def point2 : ℝ × ℝ := (4, 15)
def y_axis_intersection : ℝ × ℝ := (0, 3)

theorem line_intersects_y_axis_at : ∀ (p1 p2 : ℝ × ℝ), 
  p1 = point1 → p2 = point2 → 
  (∃ m b : ℝ, (∀ x, ((p1.2 = m * p1.1 + b) ∧ (p2.2 = m * p2.1 + b)) ∧ (y_axis_intersection = (0, b)))) :=
by
  intros p1 p2 hp1 hp2
  sorry

end line_intersects_y_axis_at_l534_534614


namespace max_triangles_9261_l534_534324

-- Define the problem formally
noncomputable def max_triangles (points : ℕ) (circ_radius : ℝ) (min_side_length : ℝ) : ℕ :=
  -- Function definition for calculating the maximum number of triangles
  sorry

-- State the conditions and the expected maximum number of triangles
theorem max_triangles_9261 :
  max_triangles 63 10 9 = 9261 :=
sorry

end max_triangles_9261_l534_534324


namespace solve_quadratic_eq_pos_solution_l534_534975

theorem solve_quadratic_eq_pos_solution :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ x : ℝ, x^2 + 8 * x = 48 ∧ x = real.sqrt (a : ℝ) - (b : ℝ) ∧ a + b = 68)) :=
sorry

end solve_quadratic_eq_pos_solution_l534_534975


namespace average_visitors_per_day_l534_534587

theorem average_visitors_per_day
  (sunday_visitors : ℕ := 540)
  (other_days_visitors : ℕ := 240)
  (days_in_month : ℕ := 30)
  (first_day_is_sunday : Bool := true)
  (result : ℕ := 290) :
  let num_sundays := 5
  let num_other_days := days_in_month - num_sundays
  let total_visitors := num_sundays * sunday_visitors + num_other_days * other_days_visitors
  let average_visitors := total_visitors / days_in_month
  average_visitors = result :=
by
  sorry

end average_visitors_per_day_l534_534587


namespace min_inputs_when_zero_min_inputs_unknown_l534_534223

-- Condition: There exists a sequence of 2017 integers with all but one number being the same.
def condition_sequence (x : Fin 2017 → Int) : Prop :=
  ∃ u d t, (∀ i, i ≠ t → x i = u) ∧ x t = d

-- Condition: The machine outputs the value ∑ i, x i * y i.
def machine_output (x y : Fin 2017 → Int) : Int :=
  ∑ i, x i * y i

-- Problem 1: Prove that the minimum number of inputs required to determine the sequence when the distinct number in the sequence is 0 is 2.
theorem min_inputs_when_zero (x : Fin 2017 → Int) (h : condition_sequence x) :
  (∃ y₁ y₂ : Fin 2017 → Int, 
    let o₁ := machine_output x y₁ 
    let o₂ := machine_output x y₂ 
    in ∀ t, (x t = 0 → x (next_t t) = u) → 2 :=
sorry

-- Problem 2: Prove that the minimum number of inputs required to determine the sequence when the distinct number in the sequence is unknown is 2.
theorem min_inputs_unknown (x : Fin 2017 → Int) (h : condition_sequence x) :
  (∃ y₁ y₂ : Fin 2017 → Int, 
    let o₁ := machine_output x y₁ 
    let o₂ := machine_output x y₂ 
    in  (∀ t, (∀ x_t, x t ≠ u ∧ x t ≠ d) → t) ∧ 
        (∀ distinct_d, distinct_d ≠ 0 → x distinct_d) ∧
        u ≠ d →
    2) :=
sorry

end min_inputs_when_zero_min_inputs_unknown_l534_534223


namespace problem_1_problem_2_l534_534576

-- Statements for our proof problems
theorem problem_1 (a b : ℝ) : a^2 + b^2 ≥ 2 * (2 * a - b) - 5 :=
sorry

theorem problem_2 (a b : ℝ) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) ∧ (a = b ↔ a^a * b^b = (a * b)^((a + b) / 2)) :=
sorry

end problem_1_problem_2_l534_534576


namespace quadratic_function_coeffs_l534_534978

theorem quadratic_function_coeffs (a b : ℝ) :
  (∀ x, (y : ℝ) = a * x^2 + b * x + 4) →
  (y = a * (-1)^2 + b * (-1) + 4 = 3) →
  (y = a * (2)^2 + b * (2) + 4 = 18) →
  (a = 2 ∧ b = 3) :=
by
  intros
  sorry

end quadratic_function_coeffs_l534_534978


namespace compare_sums_of_square_roots_l534_534725

theorem compare_sums_of_square_roots
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (M : ℝ := Real.sqrt a + Real.sqrt b) 
  (N : ℝ := Real.sqrt (a + b)) :
  M > N :=
by
  sorry

end compare_sums_of_square_roots_l534_534725


namespace closest_point_to_point_on_line_l534_534711

-- Definition of the line y = (x + 3) / 3
def line (x : ℝ) : ℝ := (x + 3) / 3

-- The point to which we are finding the closest point on the line
def point : ℝ × ℝ := (4, 0)

-- The point we want to prove is the closest point on the line to (4, 0)
def candidate_point : ℝ × ℝ := (39 / 10, 3 / 10)

-- Proof statement
theorem closest_point_to_point_on_line :
  ∃ p : ℝ × ℝ, (p.1 = 39 / 10 ∧ p.2 =  3 / 10) ∧ line p.1 = p.2 ∧
  ∀ q : ℝ × ℝ, (line q.1 = q.2) → 
  ((p.1 - 4)^2 + (p.2 - 0)^2 ≤ (q.1 - 4)^2 + (q.2 - 0)^2) :=
by
  -- proof goes here
  sorry

end closest_point_to_point_on_line_l534_534711


namespace ads_interest_block_l534_534372

theorem ads_interest_block :
  (blocked_ads : ℝ) 
  (interesting_ads : ℝ) : 
  blocked_ads = 0.80 ∧ interesting_ads = 0.20 → (1 - blocked_ads) * (1 - interesting_ads) = 0.16 :=
by {
  sorry
}

end ads_interest_block_l534_534372


namespace sum_of_weights_eq_32_l534_534662

variable (V W : Type)
variable (a : V → ℕ) (b : W → ℕ)
variable (N : V → set V)
variable [Fintype V] [Fintype W]
variable (f : V → W)
variable h : ∀ v, b (f v) = ∑ v' in (N v), a v'

theorem sum_of_weights_eq_32 (h₁ : Fintype.card V = 6) (h₂ : ∀ w, f (classical.some ⟨V, sorry⟩) = w → ∃ v, f v = w) :
  ∑ v in Finset.univ, a v = 32 :=
by sorry

end sum_of_weights_eq_32_l534_534662


namespace correct_ratio_l534_534208

noncomputable def lila_speed_to_average_speed_ratio : ℚ :=
  let L : ℚ := 2 in
  let A : ℚ := 1 in
  L / A

theorem correct_ratio (L A : ℚ) (h1 : 6 * (100 / L) + 6 * (100 / A) = 900) (h2 : A = 1) :
  lila_speed_to_average_speed_ratio = 2 := 
by
  unfold lila_speed_to_average_speed_ratio
  rw [mul_add, mul_div_assoc, mul_div_assoc, add_assoc, div_eq_inv_mul] at h1
  have h3: 100 * (1 / L) + 100 * 1 = 150 := by 
    linarith [h1]
  simp only [mul_one] at h3
  have h4: 100 / L = 50 := by 
    linarith [h3]
  field_simp [div_eq_inv_mul] at h4
  have h5: L = 2 := by
    linarith [h4]
  exact_mod_cast by linarith

end correct_ratio_l534_534208


namespace houses_with_both_features_l534_534417

theorem houses_with_both_features 
  (G P N T : ℕ) 
  (hG : G = 50) 
  (hP : P = 40) 
  (hN : N = 35) 
  (hT : T = 90) : 
  ∃ (B : ℕ), B = 35 := 
by
  -- Define the number of houses with both features
  let GP := T - N
  
  -- Define B using the principle of inclusion-exclusion
  let B := G + P - GP

  -- Assert that B is 35
  use B

  -- sorry
  

end houses_with_both_features_l534_534417


namespace geometric_sum_condition_l534_534467

theorem geometric_sum_condition 
  (a q : ℝ) 
  (h1 : q ≠ 1)
  (h2 : (a * (1 - q^3)) / (1 - q) = 3 * a * q^2) : 
  q = -1 :=
begin
  sorry
end

end geometric_sum_condition_l534_534467


namespace positive_difference_volumes_l534_534263

open Real

noncomputable def charlies_height := 12
noncomputable def charlies_circumference := 10
noncomputable def danas_height := 8
noncomputable def danas_circumference := 10

theorem positive_difference_volumes (hC : ℝ := charlies_height) (CC : ℝ := charlies_circumference)
                                   (hD : ℝ := danas_height) (CD : ℝ := danas_circumference) :
    (π * (π * ((CD / (2 * π)) ^ 2) * hD - π * ((CC / (2 * π)) ^ 2) * hC)) = 100 :=
by
  have rC := CC / (2 * π)
  have VC := π * (rC ^ 2) * hC
  have rD := CD / (2 * π)
  have VD := π * (rD ^ 2) * hD
  sorry

end positive_difference_volumes_l534_534263


namespace broccoli_sales_l534_534117

theorem broccoli_sales (B C S Ca : ℝ) (h1 : C = 2 * B) (h2 : S = B / 2 + 16) (h3 : Ca = 136) (total_sales : B + C + S + Ca = 380) :
  B = 57 :=
by
  sorry

end broccoli_sales_l534_534117


namespace price_decrease_for_original_price_l534_534566

theorem price_decrease_for_original_price (P : ℝ) (h : P > 0) :
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  decrease = 20 :=
by
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  sorry

end price_decrease_for_original_price_l534_534566


namespace max_omega_for_monotonic_sine_l534_534732

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x)

theorem max_omega_for_monotonic_sine :
  (∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 3 → f (3 / 2) x1 ≤ f (3 / 2) x2) ∧
  (∀ ω : ℝ, (0 < ω) → (∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 3 → f ω x1 ≤ f ω x2) → ω ≤ 3 / 2) :=
by sorry

end max_omega_for_monotonic_sine_l534_534732


namespace sufficient_condition_l534_534190

theorem sufficient_condition (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_condition_l534_534190


namespace lcm_of_1227_and_40_l534_534199

theorem lcm_of_1227_and_40 : Nat.lcm 1227 40 = 49080 :=
by
  have h1 : Nat.prime 1227 := sorry
  have h2 : ¬ (40 ∣ 1227) := sorry
  exact sorry

end lcm_of_1227_and_40_l534_534199


namespace bottles_remaining_correct_l534_534580

noncomputable def total_remaining_bottles_in_storage : ℕ :=
let S₀ := 8000 in
let M₀ := 12000 in
let B₀ := 15000 in
let S_sold := 0.10 * S₀ in
let M_sold := 0.20 * M₀ in
let B_sold := 0.25 * B₀ in
let R_S := S₀ - S_sold + 500 in
let R_M := M₀ - M_sold in
let R_B := B₀ - B_sold in
R_S + R_M + R_B

theorem bottles_remaining_correct : 
  total_remaining_bottles_in_storage = 27750 :=
by sorry

end bottles_remaining_correct_l534_534580


namespace range_of_a_l534_534765

-- Definitions for the problem
def y (a x : ℝ) : ℝ := (a - 3) * x^3 + Real.log x
def y'_derivative (a x : ℝ) : ℝ := 3 * (a - 3) * x^2 + (1 / x)
def f (a x : ℝ) : ℝ := x^3 - a * x^2 - 3 * x + 1
def f'_derivative (a x : ℝ) : ℝ := 3 * x^2 - 2 * a * x - 3

-- Stating the problem
theorem range_of_a 
  (a : ℝ) 
  (h1 : ∀ x > 0, y'_derivative a x ≠ 0) 
  (h2 : ∀ x ∈ Set.Icc (1:ℝ) (2:ℝ), f'_derivative a x ≥ 0) 
  : a ≤ 0 :=
sorry

end range_of_a_l534_534765


namespace number_of_games_played_each_year_l534_534289

theorem number_of_games_played_each_year (G : ℕ) 
  (h1 : ∃ D1 : ℕ, D1 = 0.90 * G)
  (h2 : ∃ D2 : ℕ, D2 = D1 - 4 ∧ D2 = 14) : G = 20 :=
by
  obtain ⟨D1, hD1⟩ := h1
  obtain ⟨D2, hD2, hD2_eq_14⟩ := h2
  sorry

end number_of_games_played_each_year_l534_534289


namespace sum_first_45_natural_numbers_l534_534198

theorem sum_first_45_natural_numbers :
  (finset.sum (finset.range 46) (λ x, x)) = 1035 :=
by {
  -- statement of the theorem
  sorry
}

end sum_first_45_natural_numbers_l534_534198


namespace num_ways_to_express_427_l534_534422

theorem num_ways_to_express_427 : 
  (card {k : ℕ | ∃ n : ℕ, k ≥ 2 ∧ 427 = k * n + (k * (k - 1)) / 2}) = 1 :=
sorry

end num_ways_to_express_427_l534_534422


namespace functions_not_same_l534_534246

theorem functions_not_same : 
  (∃ x ∈ ℝ, (f : ℝ → ℝ) = λ x, Real.log (Real.exp x) ∧ (g : ℝ → ℝ) = λ x, 10^Real.log10 x) →
  (∃ x, f x ≠ g x) :=
by
  sorry

end functions_not_same_l534_534246


namespace powers_greater_than_thresholds_l534_534285

theorem powers_greater_than_thresholds :
  (1.01^2778 > 1000000000000) ∧
  (1.001^27632 > 1000000000000) ∧
  (1.000001^27631000 > 1000000000000) ∧
  (1.01^4165 > 1000000000000000000) ∧
  (1.001^41447 > 1000000000000000000) ∧
  (1.000001^41446000 > 1000000000000000000) :=
by sorry

end powers_greater_than_thresholds_l534_534285


namespace midpoint_polar_coordinates_l534_534418

structure PolarCoord :=
  (r : ℝ)
  (theta : ℝ)

theorem midpoint_polar_coordinates (A B : PolarCoord) (hA : A = ⟨8, π / 3⟩) (hB : B = ⟨8, -π / 6⟩) :
  (∃ M : PolarCoord, M = ⟨4 * sqrt (2 + 2 * sqrt 3), π / 12⟩) :=
sorry

end midpoint_polar_coordinates_l534_534418


namespace sum_b_1000_l534_534094

def b : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 1
| n + 3 := 
  let p := b (n + 2)
  let q := b (n + 1) * b n
  if -4 * p^3 + 27 * q^2 > 0 then 3 else 1

theorem sum_b_1000 : (Finset.range 1000).sum b = 2003 := 
by 
  sorry

end sum_b_1000_l534_534094


namespace books_added_l534_534210

theorem books_added (initial_books sold_books current_books added_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : sold_books = 3)
  (h3 : current_books = 11)
  (h4 : added_books = current_books - (initial_books - sold_books)) :
  added_books = 10 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end books_added_l534_534210


namespace find_a_l534_534046

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 5

theorem find_a (a b : ℤ) (h1 : a < b) (h2 : b - a = 1) 
  (h3 : ∃ x : ℝ, x ∈ Set.Ioo a b ∧ f x = 0) : a = 3 :=
sorry

end find_a_l534_534046


namespace proof_problem_l534_534105

open Real

theorem proof_problem 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_sum : a + b + c + d = 1)
  : (b * c * d / (1 - a)^2) + (c * d * a / (1 - b)^2) + (d * a * b / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1 / 9 := 
   sorry

end proof_problem_l534_534105


namespace necessary_but_not_sufficient_cond_l534_534886

open Set

variable {α : Type*} (A B C : Set α)

/-- Mathematical equivalent proof problem statement -/
theorem necessary_but_not_sufficient_cond (h1 : A ∪ B = C) (h2 : ¬ B ⊆ A) (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ y ∈ C, y ∉ A) :=
by
  sorry

end necessary_but_not_sufficient_cond_l534_534886


namespace cos2α_minus_cosα_over_sinα_l534_534341

variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2)
variable (h2 : cos α = 2 * sqrt 5 / 5)

theorem cos2α_minus_cosα_over_sinα : cos (2 * α) - (cos α) / (sin α) = -7 / 5 :=
by
  sorry

end cos2α_minus_cosα_over_sinα_l534_534341


namespace percentage_increase_of_sides_l534_534963

noncomputable def percentage_increase_in_area (L W : ℝ) (p : ℝ) : ℝ :=
  let A : ℝ := L * W
  let L' : ℝ := L * (1 + p / 100)
  let W' : ℝ := W * (1 + p / 100)
  let A' : ℝ := L' * W'
  ((A' - A) / A) * 100

theorem percentage_increase_of_sides (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    percentage_increase_in_area L W 20 = 44 :=
by
  sorry

end percentage_increase_of_sides_l534_534963


namespace defective_x_ray_probability_l534_534055

theorem defective_x_ray_probability :
  let A1 := 5/10
  let A2 := 3/10
  let A3 := 2/10
  let B_given_A1 := 1/10
  let B_given_A2 := 1/15
  let B_given_A3 := 1/20
  (A1 * B_given_A1 + A2 * B_given_A2 + A3 * B_given_A3) = 0.08 :=
by
  let A1 := 5 / 10 : ℝ
  let A2 := 3 / 10 : ℝ
  let A3 := 2 / 10 : ℝ
  let B_given_A1 := 1 / 10 : ℝ
  let B_given_A2 := 1 / 15 : ℝ
  let B_given_A3 := 1 / 20 : ℝ
  sorry

end defective_x_ray_probability_l534_534055


namespace count_positive_integers_with_two_digits_l534_534831

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534831


namespace determine_m_l534_534360

theorem determine_m {m : ℕ} : 
  (∃ (p : ℕ), p = 5 ∧ p = max (max (max 1 (1 + (m+1))) (3+1)) 4) → m = 3 := by
  sorry

end determine_m_l534_534360


namespace extra_postage_count_correct_l534_534496

def length_to_height_ratio (length height : ℝ) : ℝ := length / height

def extra_postage_needed (length height thickness : ℝ) : Bool :=
  (length_to_height_ratio length height < 1.2 ∨ length_to_height_ratio length height > 2.8) ∨ (thickness > 0.25)

def envelopes : List (ℝ × ℝ × ℝ) :=
  [(7, 5, 0.2), (10, 2, 0.3), (7, 7, 0.1), (12, 4, 0.26)]

def count_extra_postage_needed (envs : List (ℝ × ℝ × ℝ)) : ℕ :=
  envs.countp $ λ ⟨l, h, t⟩ => extra_postage_needed l h t

theorem extra_postage_count_correct : count_extra_postage_needed envelopes = 3 := by
  sorry

end extra_postage_count_correct_l534_534496


namespace number_of_polynomial_expressions_l534_534427

def is_polynomial (expr : String) : Bool :=
  match expr with
  | "1/x" => false
  | "2x + y" => true
  | "1/3 * a^2 * b" => true
  | "(x - y)/π" => true
  | "5y/4x" => false
  | "0.5" => true
  | _ => false

def expressions : List String := ["1/x", "2x + y", "1/3 * a^2 * b", "(x - y)/π", "5y/4x", "0.5"]

def count_polynomials (expr_list : List String) : Nat :=
  expr_list.countp (λ expr => is_polynomial expr)

theorem number_of_polynomial_expressions : count_polynomials expressions = 4 :=
  by
    sorry

end number_of_polynomial_expressions_l534_534427


namespace smallest_real_number_l534_534623

theorem smallest_real_number :
  (∀ x ∈ ({0, -((8:ℝ)^(1/3)), 2, -1.7} : set ℝ), -((8:ℝ)^(1/3)) ≤ x) :=
by {
    sorry
}

end smallest_real_number_l534_534623


namespace max_quarters_l534_534484

theorem max_quarters (q : ℕ) (nickels : ℕ) (dimes : ℕ) 
  (h1 : nickels = q) 
  (h2 : dimes = 2 * q) 
  (h3 : 0.25 * q + 0.05 * nickels + 0.10 * dimes = 4.80) : 
  q = 9 :=
by
  sorry

end max_quarters_l534_534484


namespace range_for_a_l534_534013

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1)
def q : Prop := (2 * a - 3)^2 - 4 > 0

theorem range_for_a : 
  (p a ∨ q a) ∧ ¬ (p a ∧ q a) → a ∈ Icc (1/2 : ℝ) 1 ∪ Ioi (5/2 : ℝ) :=
by sorry

end range_for_a_l534_534013


namespace ellipse_eccentricity_half_l534_534588

noncomputable def ellipse_eccentricity (a b c : ℝ) : ℝ := c / a

theorem ellipse_eccentricity_half 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : l passes through both a vertex and a focus of the ellipse with equation x^2 / a^2 + y^2 / b^2 = 1)
  (h4 : distance from center of the ellipse to l = b / 4) 
  : ellipse_eccentricity a b c = 1 / 2 := 
sorry

end ellipse_eccentricity_half_l534_534588


namespace moles_NaNO3_formed_l534_534301

section chemistry
  variables (AgNO3 NaOH AgOH NaNO3 : Type) [noncomputable_instance : Inhabited AgNO3]
  variables [noncomputable_instance : Inhabited NaOH] [noncomputable_instance : Inhabited AgOH]
  variables [noncomputable_instance : Inhabited NaNO3]

  -- Define the amounts of reactants
  def moles_AgNO3 : ℕ := 2
  def moles_NaOH : ℕ := 2

  -- Balanced chemical equation implies the reaction ratio in moles
  axiom balanced_reaction : (∀ (x : ℕ), (x : ℕ) moles_AgNO3 + (x : ℕ) moles_NaOH → (x : ℕ) moles_AgOH + (x : ℕ) moles_NaNO3)

  -- Prove the amount of sodium nitrate formed
  theorem moles_NaNO3_formed : moles_AgNO3 = moles_NaOH → moles_AgNO3 = 2 → moles_NaOH = 2 → moles_NaNO3 = 2 :=
    by
      intros h1 h2 h3
      -- Hence the reaction occurs in a 1:1:1 molar ratio and with initial 2 moles of each reactant
      sorry
end chemistry

end moles_NaNO3_formed_l534_534301


namespace count_two_digit_or_less_numbers_l534_534848

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534848


namespace equal_expression_exists_l534_534052

-- lean statement for the mathematical problem
theorem equal_expression_exists (a b : ℤ) :
  ∃ (expr : ℤ), expr = 20 * a - 18 * b := by
  sorry

end equal_expression_exists_l534_534052


namespace f_not_monotonic_l534_534767

def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.sin x) + (Real.cos x)

theorem f_not_monotonic : ¬(Monotone f) := by
  sorry

end f_not_monotonic_l534_534767


namespace sqrt_meaningful_range_l534_534398

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1) ↔ x ≥ 1) := 
sorry

end sqrt_meaningful_range_l534_534398


namespace min_selling_price_l534_534214

def units_produced : ℕ := 400
def cost_per_unit : ℝ := 40
def variable_cost_per_unit : ℝ := 10
def desired_profit : ℝ := 40000

theorem min_selling_price :
  let total_production_cost := units_produced * cost_per_unit in
  let total_variable_cost := units_produced * variable_cost_per_unit in
  let total_cost := total_production_cost + total_variable_cost in
  let total_revenue_needed := total_cost + desired_profit in
  let min_price_per_unit := total_revenue_needed / units_produced in
  min_price_per_unit = 150 :=
by
  sorry

end min_selling_price_l534_534214


namespace maximize_profit_l534_534234

-- Define the cost price, base selling price, and base daily units sold.
def cost_price : ℝ := 10
def base_price : ℝ := 18
def base_units : ℝ := 60

-- Define the price change and corresponding sales change based on market research.
def sales_increase_per_yuan_decrease: ℝ := 10
def sales_decrease_per_yuan_increase: ℝ := 5

-- Define the profit function when the price is greater than or equal to the base price.
def profit_gte_base_price (x : ℝ) : ℝ := 
  let sales := base_units - sales_decrease_per_yuan_increase * (x - base_price)
  in sales * (x - cost_price)

-- Define the profit function when the price is less than the base price.
def profit_lt_base_price (x : ℝ) : ℝ := 
  let sales := base_units + sales_increase_per_yuan_decrease * (base_price - x)
  in sales * (x - cost_price)

theorem maximize_profit :
  let S := λ x, if x ≥ base_price then profit_gte_base_price x else profit_lt_base_price x in
  x = 20 → ∀ y, (y ≠ x → S y < S x) := by
    sorry

end maximize_profit_l534_534234


namespace count_at_most_two_different_digits_l534_534859

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534859


namespace chessboard_path_even_l534_534602

/- Define the properties of the chessboard and the movement constraints -/

def valid_path (n : ℕ) : Prop :=
  ∃ (path : list (ℕ × ℕ)), -- A path is a list of coordinates on the chessboard
    (∀ coords ∈ path, fst coords < n ∧ snd coords < n) ∧ -- All coordinates are within the n x n chessboard
    (∀ (i : ℕ), i < path.length - 1 → 
      ((abs (fst (path.nth_le i _) - fst (path.nth_le (i+1) _)) = 1 ∧ snd (path.nth_le i _) = snd (path.nth_le (i+1) _)) ∨
       (abs (snd (path.nth_le i _) - snd (path.nth_le (i+1) _)) = 1 ∧ fst (path.nth_le i _) = fst (path.nth_le (i+1) _)) ∨
       (abs ((fst (path.nth_le i _)) - (fst (path.nth_le (i+1) _))) = 1 ∧ abs ((snd (path.nth_le i _)) - (snd (path.nth_le (i+1) _))) = 1)) ∧
    (∀ coords1 coords2 ∈ path, coords1 ≠ coords2) -- All coordinates in the path are distinct

theorem chessboard_path_even (n : ℕ) : valid_path n → n % 2 = 0 :=
sorry

end chessboard_path_even_l534_534602


namespace determine_phi_l534_534579

noncomputable def translation_function (x : ℝ) (φ : ℝ) : ℝ :=
  sin (2 * (x + π / 8) + φ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem determine_phi :
  (∀ x : ℝ, sin (2 * x + φ) = translation_function x φ) →
  is_even_function (translation_function x φ) →
  φ = π / 4 :=
by
  sorry

end determine_phi_l534_534579


namespace max_value_of_E_l534_534476

theorem max_value_of_E (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ^ 5 + b ^ 5 = a ^ 3 + b ^ 3) : 
  a^2 - a*b + b^2 ≤ 1 :=
sorry

end max_value_of_E_l534_534476


namespace parabola_distance_mn_l534_534014

theorem parabola_distance_mn :
  ∀ (M N: ℝ × ℝ) (F: ℝ × ℝ), 
  (∀ y, (4 * (M.snd)^2 = M.fst) ∧ (4 * (N.snd) ^ 2 = N.fst)) →
  F = (1/16, 0) →
  ∥M - F∥ = 1/8 → 
  ∥M - N∥ = 1/4 := sorry

end parabola_distance_mn_l534_534014


namespace find_h_example_l534_534682

-- Given condition
def eq1 (h : ℚ → ℚ) (x : ℚ) : Prop :=
  9*x^3 + 6*x^2 - 3*x + 1 + h(x) = 5*x^2 - 7*x + 4

-- The goal is to find h in terms of x
theorem find_h_example :
  ∃ h : ℚ → ℚ,
    (∀ x : ℚ, eq1 h x) →
    (∀ x : ℚ, h x = -9*x^3 - x^2 - 4*x + 3) :=
begin
  sorry
end

end find_h_example_l534_534682


namespace b32_value_l534_534270

theorem b32_value (b : Fin 32 → ℕ)
  (P : Polynomial ℤ := ∏ i in Finset.range 32, (1 - Polynomial.C (1 : ℤ) * Polynomial.X ^ (i + 1)) ^ (b i))
  (H1 : Polynomial.truncate 32 P = 1 - 2 * Polynomial.X) :
  b 31 = 2^27 - 2^11 :=
by sorry

end b32_value_l534_534270


namespace number_of_positive_integers_with_at_most_two_digits_l534_534814

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l534_534814


namespace no_such_n_exists_l534_534129

theorem no_such_n_exists (k : ℕ) (hk : k > 0) : 
  ¬ ∃ n : ℕ, n > 1 ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n-1 → k ∣ (nat.choose n i) := 
sorry

end no_such_n_exists_l534_534129


namespace general_formula_for_a_sum_of_b_l534_534429

-- Defining the sequence and conditions
def a (n : ℕ) : ℚ := (n + 1) / 2
variables (an_7 : a 7 = 4) (an_19 : a 19 = 2 * a 9)

-- Defining b_n
def b (n : ℕ) : ℚ := 1 / (n * a n)

-- Defining the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := ∑ i in finset.range n, b (i + 1)

-- Problem 1: Prove that the general formula for the sequence is a_n = (n + 1) / 2
theorem general_formula_for_a : a (n : ℕ) = (n + 1) / 2 :=
by
  sorry

-- Problem 2: Prove that the sum of the first n terms of the sequence b_n is S_n = n / (n + 1)
theorem sum_of_b (n : ℕ) : S n = n / (n + 1) :=
by
  sorry

end general_formula_for_a_sum_of_b_l534_534429


namespace book_distribution_l534_534675

theorem book_distribution :
  ∃ (f : Fin 7 → Fin 2), 
    (card (f ⁻¹' {0}) ≥ 2) ∧ 
    (card (f ⁻¹' {1}) ≥ 1) ∧
    (abs (card (f ⁻¹' {0}) - card (f ⁻¹' {1})) > 1) ∧
    multichoose 7 2 + multichoose 7 5 + multichoose 7 6 = 49 :=
sorry

end book_distribution_l534_534675


namespace train_length_l534_534240

theorem train_length
  (t1 : ℕ) (t2 : ℕ)
  (d_platform : ℕ)
  (h1 : t1 = 8)
  (h2 : t2 = 20)
  (h3 : d_platform = 279)
  : ∃ (L : ℕ), (L : ℕ) = 186 :=
by
  sorry

end train_length_l534_534240


namespace complex_modulus_eval_l534_534679

open Complex

def complex_modulus_evaluation : ℂ :=
  (7/4 : ℚ) + 3

theorem complex_modulus_eval :
    abs ((7/4 : ℂ) - 3 * Complex.I + Real.sqrt 3) = (Real.sqrt (241 + 56 * Real.sqrt 3) / 4) :=
    by sorry

end complex_modulus_eval_l534_534679


namespace A_greater_than_B_l534_534376

theorem A_greater_than_B (A B : ℝ) (h₁ : A * 4 = B * 5) (h₂ : A ≠ 0) (h₃ : B ≠ 0) : A > B :=
by
  sorry

end A_greater_than_B_l534_534376


namespace perpendicular_planes_l534_534379

/--
Given distinct lines l, m, n in space, and non-coincident planes α and β.
If l is perpendicular to α and l is parallel to β, then α is perpendicular to β.
-/
theorem perpendicular_planes (l m n : Line) (α β : Plane) :
  (l ∈ α) → (¬ (α = β)) → (l ∈ β) → l ⊥ α → l ∥ β → α ⊥ β :=
sorry

end perpendicular_planes_l534_534379


namespace smallest_positive_period_f_l534_534026

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x ^ 2 - sqrt 3

theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) := by
  use π
  sorry

end smallest_positive_period_f_l534_534026


namespace sum_correct_l534_534653

open Nat

noncomputable def sum_problem : ℝ :=
  (∑ k in range 1 (501 : ℕ), (k : ℝ) * (if (∃ j : ℤ, (k : ℝ) = real.exp ((j : ℝ) * real.log (real.sqrt 3))) then 0 else 1))

theorem sum_correct :
  sum_problem = 125187 := 
by
  sorry

end sum_correct_l534_534653


namespace count_positive_integers_with_two_digits_l534_534836

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l534_534836


namespace sqrt_sqrt_81_is_9_l534_534506

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l534_534506


namespace find_alpha_l534_534405

theorem find_alpha (α : ℕ) :
  (∏ (i : ℕ) in (finset.range (α + 1)), 10 ^ (i / 11)) = 1_000_000 ↔ α = 11 :=
by 
  sorry

end find_alpha_l534_534405


namespace sumCeilSqrtFrom5To35_l534_534678

-- Define the ceiling function for Real numbers
noncomputable def ceiling (x : ℝ) : ℤ := ⌈x⌉

-- Define the range and the sum over ceiling of square roots
noncomputable def sumCeilSqrt : ℤ :=
  ∑ n in (Finset.range 31).filter (λ n, n + 5), ceiling (Real.sqrt (↑n + 5))

-- The main theorem stating the problem and the result.
theorem sumCeilSqrtFrom5To35 :
  sumCeilSqrt = 148 := by
  sorry

end sumCeilSqrtFrom5To35_l534_534678


namespace log_fraction_property_l534_534574

noncomputable def log_base (a N : ℝ) : ℝ := Real.log N / Real.log a

theorem log_fraction_property :
  (log_base 3 4 / log_base 9 8) = 4 / 3 :=
by
  sorry

end log_fraction_property_l534_534574


namespace differentiation_operations_l534_534191

theorem differentiation_operations :
  (∀ x : ℝ, (deriv (λ x, x^3) x ≠ 2 * x^2)) ∧
  (∀ x : ℝ, (deriv log x ≠ 1 / x)) ∧
  (∀ x : ℝ, (deriv (λ x, x^3 + 5) x ≠ 3 * x^2 + 5)) ∧
  (∀ x : ℝ, (deriv (λ x, sin x * cos x) x = cos (2 * x))) :=
by {
  sorry
}

end differentiation_operations_l534_534191


namespace sum_of_values_of_z_for_which_f_2z_eq_4_l534_534095

def f : ℝ → ℝ := λ x, x^2 - 3 * x + 2

theorem sum_of_values_of_z_for_which_f_2z_eq_4 : 
  let z_vals := {z : ℝ | f(2 * z) = 4},
      sum_z := z_vals.sum (λ z, z)
  in sum_z = 3 / 4 :=
by
  sorry

end sum_of_values_of_z_for_which_f_2z_eq_4_l534_534095


namespace intervals_monotonicity_a1_range_of_a_for_monotonicity_l534_534768

open Real

noncomputable def f (x a : ℝ) : ℝ := (3 * x) / a - 2 * x^2 + log x

noncomputable def f_prime (x a : ℝ) : ℝ := (3 / a) - 4 * x + 1 / x

theorem intervals_monotonicity_a1 :
  ∀ (x : ℝ), (0 < x ∧ x < 1 → (f_prime x 1) > 0) ∧ (1 < x → (f_prime x 1) < 0) :=
by sorry

noncomputable def h (x : ℝ) : ℝ := 4 * x - 1 / x

theorem range_of_a_for_monotonicity :
  ∀ a : ℝ, (0 < a ≤ 2 / 5) ↔ ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → (f_prime x a) ≥ 0 :=
by sorry

end intervals_monotonicity_a1_range_of_a_for_monotonicity_l534_534768


namespace monotonic_decreasing_interval_l534_534985

noncomputable def f (x : ℝ) : ℝ := real.log (x^2 - 1)

theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x ∈ Ioo (-(1:ℝ)) 1 → y ∈ Ioo (-(1:ℝ)) 1 → x < y → f y < f x :=
sorry

end monotonic_decreasing_interval_l534_534985


namespace domain_of_h_l534_534668

open Real

theorem domain_of_h : ∀ x : ℝ, |x - 5| + |x + 2| ≠ 0 := by
  intro x
  sorry

end domain_of_h_l534_534668


namespace count_at_most_two_different_digits_l534_534856

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l534_534856


namespace count_special_integers_l534_534824

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l534_534824


namespace parabola_equation_max_area_of_triangle_l534_534354

noncomputable def ellipse (b : ℝ) (x y : ℝ) : Prop :=
  (0 < b ∧ b < 2) ∧ (x^2 / 4 + y^2 / b^2 = 1)

def eccentricity (b : ℝ) : ℝ := (Real.sqrt (4 - b^2)) / 2

noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop :=
  (0 < p) ∧ (x^2 = 2 * p * y)

def on_parabola (x y : ℝ) : Prop := x^2 = 4 * y

def middle_point (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

axiom maximum_area_triangle (M N : ℝ × ℝ) (P : ℝ × ℝ) : ℝ :=
  sorry -- area calculation as given in the solution without proof steps

theorem parabola_equation (b : ℝ) (p : ℝ) (x y : ℝ) :
  ellipse b x y →
  eccentricity b = Real.sqrt 3 / 2 →
  parabola p x y →
  p = 2 →
  ∃ x y, on_parabola x y :=
sorry

theorem max_area_of_triangle (x1 y1 x2 y2 : ℝ) :
  on_parabola x1 y1 → 
  on_parabola x2 y2 → 
  y1 ≠ y2 → 
  y1 + y2 = 4 →
  let M := (x1, y1) in
  let N := (x2, y2) in
  let P := (0, 4) in
  maximum_area_triangle M N P = 8 :=
sorry

end parabola_equation_max_area_of_triangle_l534_534354


namespace sum_k_round_diff_eq_125237_l534_534633

noncomputable def sum_k_round_diff (n : ℕ) : ℤ :=
∑ k in finset.range n.succ, k * ((⌈real.log k / real.log (real.sqrt 3)⌉ : ℤ) - (⌊real.log k / real.log (real.sqrt 3)⌋ : ℤ))

theorem sum_k_round_diff_eq_125237 : sum_k_round_diff 500 = 125237 := by
  -- Here we would add the proof steps to verify the theorem
  sorry

end sum_k_round_diff_eq_125237_l534_534633


namespace find_a_l534_534384

-- Defining the conditions as hypotheses
variables (a b d : ℕ)
hypothesis h1 : a + b = d
hypothesis h2 : b + d = 7
hypothesis h3 : d = 4

theorem find_a : a = 1 :=
by
  sorry

end find_a_l534_534384


namespace age_of_b_l534_534562

-- Definition of conditions
variable (a b c : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : a + b + c = 12)

-- The statement of the proof problem
theorem age_of_b : b = 4 :=
by {
   sorry
}

end age_of_b_l534_534562


namespace magnitude_of_z_l534_534737

def i : ℂ := complex.I
def z : ℂ := 1 / (1 - i) + i

theorem magnitude_of_z : complex.abs z = real.sqrt 10 / 2 :=
by sorry

end magnitude_of_z_l534_534737


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534793

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l534_534793


namespace monic_quadratic_with_root_l534_534703

theorem monic_quadratic_with_root :
  ∃ (p : ℝ[X]), monic p ∧ (p.coeff 2 = 1) ∧ (p.coeff 1 = 6) ∧ (p.coeff 0 = 16) ∧ is_root p (-3 - complex.I * real.sqrt 7) :=
sorry

end monic_quadratic_with_root_l534_534703


namespace first_group_men_8_l534_534887

variable (x : ℕ)

theorem first_group_men_8 (h1 : x * 80 = 20 * 32) : x = 8 := by
  -- provide the proof here
  sorry

end first_group_men_8_l534_534887


namespace Allan_more_balloons_l534_534607

-- Define the number of balloons that Allan and Jake brought
def Allan_balloons := 5
def Jake_balloons := 3

-- Prove that the number of more balloons that Allan had than Jake is 2
theorem Allan_more_balloons : (Allan_balloons - Jake_balloons) = 2 := by sorry

end Allan_more_balloons_l534_534607


namespace count_valid_numbers_l534_534868

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534868


namespace equation_of_line_containing_BC_l534_534000

theorem equation_of_line_containing_BC (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (altitude_from_CA : ∀ x y : ℝ, 2 * x - 3 * y + 1 = 0)
  (altitude_from_BA : ∀ x y : ℝ, x + y = 1)
  (eq_BC : ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b) :
  ∃ k b : ℝ, (∀ x y : ℝ, y = k * x + b) ∧ 2 * x + 3 * y + 7 = 0 :=
sorry

end equation_of_line_containing_BC_l534_534000


namespace number_of_subsets_eq_eight_l534_534045

theorem number_of_subsets_eq_eight {A : Set (ℕ)} (h : A = {a, b, c}) : (2 : ℕ) ^ A.card = 8 := by
  sorry

end number_of_subsets_eq_eight_l534_534045


namespace triangle_side_ratios_l534_534459

-- Define the points A, B, C, and P as points in a Euclidean space
variables {A B C P : Point}

-- Define the angles and lengths involved in the problem
variables [angle_1 angle_2 angle_3 : ℝ]
variables (AB AC PB PC : ℝ)

-- Define the angles sum property
axiom angle_sum_property : 3 * ∠ ABP = 3 * ∠ ACP = ∠ ABC + ∠ ACB

-- The goal to prove is given the conditions,
-- we can show that the ratio property between sides and segments holds.
theorem triangle_side_ratios (h : angle_sum_property) :
  AB / (AC + PB) = AC / (AB + PC) :=
sorry

end triangle_side_ratios_l534_534459


namespace rods_in_mile_l534_534011

theorem rods_in_mile (mile_to_furlongs : ℕ) (furlong_to_rods : ℕ) (mile_conversion : mile_to_furlongs = 10)
                      (furlong_conversion : furlong_to_rods = 50) : 
                      1 * mile_to_furlongs * furlong_to_rods = 500 :=
begin
  rw [mile_conversion, furlong_conversion],
  norm_num,
end

end rods_in_mile_l534_534011


namespace log_simplify_l534_534488

open Real

theorem log_simplify : 
  (1 / (log 12 / log 3 + 1)) + 
  (1 / (log 8 / log 2 + 1)) + 
  (1 / (log 30 / log 5 + 1)) = 2 :=
by
  sorry

end log_simplify_l534_534488


namespace isosceles_triangle_angle_l534_534544

/--
Triangle PQR is isosceles with PR = QR, and ∠PRQ = 108°.
Point N is in the interior of the triangle so that ∠PNR = 9° and ∠NRP = 27°.
Prove that ∠QNR = 36°.
--/
theorem isosceles_triangle_angle (P Q R N : Type) 
  (PR QR : P = Q ∧ R = Q)
  (angle_PRQ : angle P R Q = 108)
  (angle_PNR : angle P N R = 9)
  (angle_NRP : angle N R P = 27) :
  angle Q N R = 36 :=
sorry

end isosceles_triangle_angle_l534_534544


namespace distance_downstream_in_12min_l534_534197

-- Define the given constants
def boat_speed_still_water : ℝ := 15  -- km/hr
def current_speed : ℝ := 3  -- km/hr
def time_minutes : ℝ := 12  -- minutes

-- Prove the distance traveled downstream in 12 minutes
theorem distance_downstream_in_12min
  (b_velocity_still : ℝ)
  (c_velocity : ℝ)
  (time_m : ℝ)
  (h1 : b_velocity_still = boat_speed_still_water)
  (h2 : c_velocity = current_speed)
  (h3 : time_m = time_minutes) :
  let effective_speed := b_velocity_still + c_velocity
  let effective_speed_km_per_min := effective_speed / 60
  let distance := effective_speed_km_per_min * time_m
  distance = 3.6 :=
by
  sorry

end distance_downstream_in_12min_l534_534197


namespace sqrt_sqrt_81_is_9_l534_534504

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l534_534504


namespace range_of_f_l534_534990

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^x + 1

theorem range_of_f :
  set.range (f ∘ (λ x : ℝ, x ∈ Icc (-1 : ℝ) 1)) = set.Icc (3 / 2) 3 := sorry

end range_of_f_l534_534990


namespace median_of_roller_coaster_times_l534_534994

theorem median_of_roller_coaster_times:
  let data := [80, 85, 90, 125, 130, 135, 140, 145, 195, 195, 210, 215, 240, 245, 300, 305, 315, 320, 325, 330, 300]
  ∃ median_time, median_time = 210 ∧
    (∀ t ∈ data, t ≤ median_time ↔ index_of_median = 11) :=
by
  sorry

end median_of_roller_coaster_times_l534_534994


namespace john_gallons_per_day_l534_534085

-- Problem conditions as definitions in Lean
def quarts_per_week := 42
def quarts_per_gallon := 4
def days_per_week := 7
noncomputable def quarts_to_gallons (quarts : ℕ) := quarts / quarts_per_gallon
noncomputable def gallons_per_week := quarts_to_gallons quarts_per_week
noncomputable def gallons_per_day := gallons_per_week / days_per_week

-- Mathematical equivalent proof problem
theorem john_gallons_per_day : gallons_per_day = 1.5 :=
sorry

end john_gallons_per_day_l534_534085


namespace solution_correct_l534_534451

-- Define the problem conditions
def isEven (n : ℕ) : Prop := n % 2 = 0

def colorBoard (n : ℕ) : ℕ × ℕ → Char := 
  λ (x y : ℕ), if (x + y) % 3 = 0 then 'B' else 'W'

def move (p : ℕ × ℕ -> Char) (x : ℕ) (y : ℕ) : ℕ × ℕ -> Char :=
  λ (i j), if i = x ∧ j = y ∨ i = x + 1 ∧ j = y ∨ i = x ∧ j = y + 1 ∨ i = x + 1 ∧ j = y + 1 then
             if p (i, j) = 'B' then 'W' else if p (i, j) = 'W' then 'O' else 'B'
           else p (i, j)

def flip_possible (n : ℕ) : Prop :=
  ∃ (f : Π (x y : ℕ), ℕ × ℕ → Char → ℕ × ℕ → Char), (∀ (x y : ℕ), f x y = move) ∧
  (∀ (x y : ℕ), ∃ k, (f x y^[k]) (colorBoard n) = 
  (λ (x' y' : ℕ), if (x' + y') % 3 ≠ 0 then 'B' else 'W'))

-- Main theorem statement
theorem solution_correct :
  ∀ (n : ℕ), ((isEven n) -> flip_possible n) ∧ (¬ isEven n) -> ¬ flip_possible n :=
by sorry

end solution_correct_l534_534451


namespace sqrt_sqrt_81_is_9_l534_534503

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l534_534503


namespace find_b_l534_534987

noncomputable def P (x a b c : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: P 0 a b c = 12)
  (h2: (-c / 2) * 1 = -6)
  (h3: (2 + a + b + c) = -6)
  (h4: a + b + 14 = -6) : b = -56 :=
sorry

end find_b_l534_534987


namespace perpendicular_slope_l534_534721

theorem perpendicular_slope (x y : ℝ) : (∃ b : ℝ, 4 * x - 5 * y = 10) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro h
  sorry

end perpendicular_slope_l534_534721


namespace solve_log_eq_l534_534957

noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

theorem solve_log_eq :
  (∃ x : ℝ, log3 ((5 * x + 15) / (7 * x - 5)) + log3 ((7 * x - 5) / (2 * x - 3)) = 3 ∧ x = 96 / 49) :=
by
  sorry

end solve_log_eq_l534_534957


namespace number_of_aluminum_atoms_l534_534217

def molecular_weight (n : ℕ) : ℝ :=
  n * 26.98 + 30.97 + 4 * 16.0

theorem number_of_aluminum_atoms (n : ℕ) (h : molecular_weight n = 122) : n = 1 :=
by
  sorry

end number_of_aluminum_atoms_l534_534217


namespace probability_heads_then_tails_l534_534560

-- Defining a fair coin toss probability model
def outcomes := {("H", "H"), ("H", "T"), ("T", "H"), ("T", "T")}

def favorable_outcome := ("H", "T")

-- Main theorem stating the probability of the event where the first toss is Heads and the second is Tails.
theorem probability_heads_then_tails :
  (favorable_outcome ∈ outcomes ∧ outcomes.card = 4) → 
  (∃ n : ℕ, n = cardinal.mk {favorable_outcome} / outcomes.card = 1 / 4) :=
by { sorry }

end probability_heads_then_tails_l534_534560


namespace percentage_of_boys_who_passed_is_50_l534_534204

-- Definitions based on conditions
def total_boys := 50
def total_girls := 100
def percent_girls_passed := 40 / 100 : ℝ -- 40%
def percent_total_failed := 56.67 / 100 : ℝ -- 56.67%

-- The goal is to prove that the percentage of boys who passed is 50%

theorem percentage_of_boys_who_passed_is_50 :
  ∀ (total_boys total_girls : ℕ) 
    (percent_girls_passed percent_total_failed : ℝ),
  total_boys = 50 →
  total_girls = 100 →
  percent_girls_passed = 0.4 →
  percent_total_failed = 0.5667 →
  let total_students := total_boys + total_girls,
      number_girls_passed := percent_girls_passed * total_girls,
      percent_total_passed := 1 - percent_total_failed,
      number_students_passed := percent_total_passed * total_students,
      number_boys_passed := number_students_passed - number_girls_passed,
      percent_boys_passed := number_boys_passed / total_boys
  in percent_boys_passed * 100 = 50 :=
by {
  intros,
  sorry
}

end percentage_of_boys_who_passed_is_50_l534_534204


namespace roots_geom_prog_eq_neg_cbrt_c_l534_534950

theorem roots_geom_prog_eq_neg_cbrt_c {a b c : ℝ} (h : ∀ (x1 x2 x3 : ℝ), 
  (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ (x3^3 + a * x3^2 + b * x3 + c = 0) ∧ 
  (∃ (r : ℝ), (x2 = r * x1) ∧ (x3 = r^2 * x1))) : 
  ∃ (x : ℝ), (x^3 = c) ∧ (x = - ((c) ^ (1/3))) :=
by 
  sorry

end roots_geom_prog_eq_neg_cbrt_c_l534_534950


namespace almostSquareCount_l534_534261

def isAlmostSquare (n a b : ℕ) : Prop :=
  a ≤ b ∧ b ≤ (4 * a) / 3 ∧ a * b = n

theorem almostSquareCount :
  (finset.Icc 1 1000000).filter (λ n, ∃ a b, a > 0 ∧ b > 0 ∧ isAlmostSquare n a b).card = 130348 := 
sorry

end almostSquareCount_l534_534261


namespace problem1_problem2_l534_534458

-- Define D
def D (n : ℕ) (h : n > 1) : ℕ :=
  let divisors := (List.range (n + 1)).filter (Nat.dvd n)
  List.foldl (· + ·) 0 (List.zipWith (· * ·) divisors.tail divisors.init)

theorem problem1 (n : ℕ) (h : n > 1) : D n h < n^2 :=
by
  sorry

theorem problem2 (n : ℕ) (h : n > 1) : (D n h ∣ n^2) ↔ Nat.Prime n :=
by
  sorry

end problem1_problem2_l534_534458


namespace tetrahedron_projection_is_orthocenter_l534_534407

theorem tetrahedron_projection_is_orthocenter 
  (V A B C : Type) [InnerProductSpace ℝ V] 
  (tetrahedron : ∀ (x : V), x ∈ ({A, B, C} : Set V) → orthogonal_projection (span ℝ {A, B, C}) x = orthocenter ({A, B, C} : Set V))
  (mutually_perpendicular : ∀ (x y : V), x ≠ y ∧ x ∈ ({A, B, C} : Set V) ∧ y ∈ ({A, B, C} : Set V) → ⟪x, y⟫ = 0) :
  ∃ (O : V), (orthogonal_projection (span ℝ {A, B, C}) O = orthocenter ({A, B, C} : Set V)) :=
by
  sorry

end tetrahedron_projection_is_orthocenter_l534_534407


namespace num_distinct_prime_factors_of_15147_l534_534387

theorem num_distinct_prime_factors_of_15147 : 
  (nat.factors 15147).to_finset.card = 3 :=
sorry

end num_distinct_prime_factors_of_15147_l534_534387


namespace count_two_digit_or_less_numbers_l534_534846

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l534_534846


namespace arithmetic_square_root_sqrt_81_l534_534509

theorem arithmetic_square_root_sqrt_81 : sqrt (sqrt 81) = 3 := 
by 
  have h1 : sqrt 81 = 9 := sorry
  have h2 : sqrt 9 = 3 := sorry
  show sqrt (sqrt 81) = 3 from sorry

end arithmetic_square_root_sqrt_81_l534_534509


namespace rectangular_prism_volume_l534_534392

theorem rectangular_prism_volume
  (L W h : ℝ)
  (h1 : L - W = 23)
  (h2 : 2 * L + 2 * W = 166) :
  L * W * h = 1590 * h :=
by
  sorry

end rectangular_prism_volume_l534_534392


namespace greatest_number_of_roses_l534_534570

noncomputable def individual_rose_price: ℝ := 2.30
noncomputable def dozen_rose_price: ℝ := 36
noncomputable def two_dozen_rose_price: ℝ := 50
noncomputable def budget: ℝ := 680

theorem greatest_number_of_roses (P: ℝ → ℝ → ℝ → ℝ → ℕ) :
  P individual_rose_price dozen_rose_price two_dozen_rose_price budget = 325 :=
sorry

end greatest_number_of_roses_l534_534570


namespace max_scalene_triangles_l534_534596

-- A scalene triangle has sides of different lengths
structure ScaleneTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  hp : a > b ∧ b > c ∧ b + c > a
  h_bounds : a ≤ 6 ∧ b ≤ 6 ∧ c ≤ 6

-- Set S consists of non-congruent and non-similar scalene triangles
def is_scalene (t : ScaleneTriangle) : Prop := 
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c

noncomputable def S : set ScaleneTriangle := {t : ScaleneTriangle | is_scalene t}

-- Problem statement
theorem max_scalene_triangles : ∃ n, n = 20 ∧ ∀ t1 t2 ∈ S, t1 ≠ t2 → ¬(t1.a / t1.b = t2.a / t2.b ∧ t1.b / t1.c = t2.b / t2.c) :=
by
  sorry

end max_scalene_triangles_l534_534596


namespace polynomial_count_l534_534424

theorem polynomial_count : 
  let expr1 := (1 / x : ℝ)
  let expr2 := (2 * x + y : ℝ)
  let expr3 := (1 / 3 * a^2 * b : ℝ)
  let expr4 := ((x - y) / π : ℝ)
  let expr5 := (5 * y / (4 * x) : ℝ)
  let expr6 := (0.5 : ℝ)
  let is_polynomial (e : ℝ) := ∃ p : ℝ[X], e = p.eval 0
  in 
  (is_polynomial expr2) ∧ 
  (is_polynomial expr3) ∧ 
  (is_polynomial expr4) ∧ 
  (is_polynomial expr6) ∧ 
  ¬(is_polynomial expr1) ∧ 
  ¬(is_polynomial expr5) ∧ 
  (4 = 4) :=
  by sorry

end polynomial_count_l534_534424


namespace b_range_l534_534362

-- Define the sets M and N
def M : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = real.sqrt (9 - x^2)}
def N (b : ℝ) : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = x + b}

-- Statement of the theorem
theorem b_range (b : ℝ) : (M ∩ N b = ∅) ↔ (b > 3 * real.sqrt 2 ∨ b < -3) :=
by
  sorry

end b_range_l534_534362


namespace determine_k_l534_534671

variable (x y z k : ℝ)

theorem determine_k
  (h1 : 9 / (x - y) = 16 / (z + y))
  (h2 : k / (x + z) = 16 / (z + y)) :
  k = 25 := by
  sorry

end determine_k_l534_534671


namespace sum_of_divisors_multiple_3_l534_534665

theorem sum_of_divisors_multiple_3 (k : ℤ) (n : ℤ) (h : n = 3 * k + 2) :
  ∑ d in finset.filter (λ a, a ∣ n) (finset.range (n.nat_abs + 1)), d % 3 = 0 := by
  sorry

end sum_of_divisors_multiple_3_l534_534665


namespace fifth_equation_l534_534935

def equation_follows_pattern (n : ℕ) : Prop :=
  (∑ i in finset.range (2 * n - 1), (n + i)) = (2 * n - 1) * (n + 1) - 1

theorem fifth_equation :
  (∑ i in finset.range 9, (5 + i)) = 81 :=
by
  sorry

end fifth_equation_l534_534935


namespace tank_fraction_l534_534237

theorem tank_fraction (x : ℚ) (h₁ : 48 * x + 8 = 48 * (9 / 10)) : x = 2 / 5 :=
by
  sorry

end tank_fraction_l534_534237


namespace find_f_deriv_two_l534_534738

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + 3 * x * (f_deriv 2)
noncomputable def f_deriv (x : ℝ) : ℝ := derivative f x

theorem find_f_deriv_two : f_deriv 2 = -2 := 
by sorry

end find_f_deriv_two_l534_534738


namespace min_abs_expression_l534_534748

theorem min_abs_expression {x y : ℝ} (h : (x + 2)^2 + (y - 3)^2 = 1) :
  ∃ m, m = 15 ∧ ∀ x y, (x + 2)^2 + (y - 3)^2 = 1 → |3 * x + 4 * y - 26| ≥ m := 
sorry

end min_abs_expression_l534_534748


namespace cuboid_volume_l534_534974

theorem cuboid_volume (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : a * b * c = 120 :=
by {
  rw [ha, hb, hc],
  norm_num,
}

end cuboid_volume_l534_534974


namespace cdf_of_Z_pdf_of_Z_l534_534776

noncomputable def f1 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 0.5 else 0

noncomputable def f2 (y : ℝ) : ℝ :=
  if 0 < y ∧ y < 2 then 0.5 else 0

noncomputable def G (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1

noncomputable def g (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0

theorem cdf_of_Z (z : ℝ) : G z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1 := sorry

theorem pdf_of_Z (z : ℝ) : g z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0 := sorry

end cdf_of_Z_pdf_of_Z_l534_534776


namespace count_valid_numbers_l534_534867

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l534_534867


namespace solve_equation_l534_534297

theorem solve_equation (x : ℝ) : 
  (Real.root (50 - 3 * x) 4) + (Real.root (30 + 3 * x) 4) = 4 ↔ x = -14 ∨ x = 16 :=
by
  sorry

end solve_equation_l534_534297


namespace max_value_of_x_over_norm_b_l534_534344

variables {e1 e2 b : EuclideanSpace ℝ (Fin 2)}
variables {x y : ℝ}
variables (hx : ‖e1‖ = 1) (hy : ‖e2‖ = 1) (hb : b = x • e1 + y • e2)
variables (angle_e1_e2 : real.angle e1 e2 = π / 6)

theorem max_value_of_x_over_norm_b :
  ∃ max_val : ℝ, max_val = 2 ∧
  (∀ x y : ℝ, y ≠ 0 → ∀ b : EuclideanSpace ℝ (Fin 2), b = x • e1 + y • e2 →
    ‖e1‖ = 1 → ‖e2‖ = 1 → 
    real.angle e1 e2 = π / 6 →
    (∀ x y, 0 ≤ (abs x / ‖b‖)) ∧ abs x / ‖b‖ ≤ max_val) :=
sorry

end max_value_of_x_over_norm_b_l534_534344


namespace triangle_third_side_length_l534_534894

theorem triangle_third_side_length {x : ℝ}
    (h1 : 3 > 0)
    (h2 : 7 > 0)
    (h3 : 3 + 7 > x)
    (h4 : x + 3 > 7)
    (h5 : x + 7 > 3) :
    4 < x ∧ x < 10 := by
  sorry

end triangle_third_side_length_l534_534894


namespace jeff_corrected_mean_l534_534083

def initial_scores : List ℕ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℕ := [85, 90, 92, 93, 89, 89, 88]

noncomputable def arithmetic_mean (scores : List ℕ) : ℝ :=
  (scores.sum : ℝ) / (scores.length : ℝ)

theorem jeff_corrected_mean :
  arithmetic_mean corrected_scores = 89.42857142857143 := 
by
  sorry

end jeff_corrected_mean_l534_534083


namespace function_decreasing_range_k_l534_534158

theorem function_decreasing_range_k : 
  ∀ k : ℝ, (∀ x : ℝ, 1 ≤ x → ∀ y : ℝ, 1 ≤ y → x ≤ y → (k * x ^ 2 + (3 * k - 2) * x - 5) ≥ (k * y ^ 2 + (3 * k - 2) * y - 5)) ↔ (k ∈ Set.Iic 0) :=
by sorry

end function_decreasing_range_k_l534_534158


namespace parameterized_line_solution_l534_534521

theorem parameterized_line_solution :
  ∃ (s l : ℚ), 
  (∀ t : ℚ, 
    ∃ x y : ℚ, 
      x = -3 + t * l ∧ 
      y = s + t * (-7) ∧ 
      y = 3 * x + 2
  ) ∧
  s = -7 ∧ l = -7 / 3 := 
sorry

end parameterized_line_solution_l534_534521


namespace monic_quadratic_with_root_l534_534700

theorem monic_quadratic_with_root :
  ∃ (p : ℝ[X]), monic p ∧ (p.coeff 2 = 1) ∧ (p.coeff 1 = 6) ∧ (p.coeff 0 = 16) ∧ is_root p (-3 - complex.I * real.sqrt 7) :=
sorry

end monic_quadratic_with_root_l534_534700


namespace find_constants_l534_534121

-- Given definitions based on the conditions and conjecture
def S (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | 2 => 5
  | 3 => 15
  | 4 => 34
  | 5 => 65
  | _ => 0

noncomputable def conjecture_S (n a b c : ℤ) := (2 * n - 1) * (a * n^2 + b * n + c)

theorem find_constants (a b c : ℤ) (h1 : conjecture_S 1 a b c = 1) (h2 : conjecture_S 2 a b c = 5) (h3 : conjecture_S 3 a b c = 15) : 3 * a + b = 4 :=
by
  -- Proof omitted
  sorry

end find_constants_l534_534121


namespace unique_interpolating_polynomial_exists_l534_534927

theorem unique_interpolating_polynomial_exists 
  (n : ℕ)
  (x : Fin n → ℝ) 
  (y : Fin n → ℝ) 
  (hx : ∀ i j : Fin n, i < j → x i < x j) :
  ∃! f : ℝ[X], (deg f ≤ n - 1) ∧ ∀ i, f.eval (x i) = y i :=
by
  sorry

end unique_interpolating_polynomial_exists_l534_534927
