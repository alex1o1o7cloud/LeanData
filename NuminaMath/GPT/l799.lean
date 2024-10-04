import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Algebra.Quot
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Convex.Caratheodory
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.NormedSpace.InnerProduct
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GroupTheory.Subgroup
import Mathlib.Init.Data.Int.Basic
import Mathlib.LinearAlgebra.TensorProduct
import Mathlib.MeasureTheory.Integration.Convex
import Mathlib.NumberTheory.GCD
import Mathlib.NumberTheory.GCD.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbSpace
import Mathlib.Probability.ProbabilityBasics
import Mathlib.Tactic
import Mathlib.Topology.ProjectivePlane
import data.real.basic
import data.set.basic

namespace regular_pentagon_of_equal_sides_and_angle_conditions_l799_799975

def is_convex (ABCDE : List Point) : Prop := sorry

def equal_sides (ABCDE : List Point) : Prop := sorry

def angle_condition (A B C D E : Point) : Prop :=
  ∠A ≥ ∠B ∧ ∠B ≥ ∠C ∧ ∠C ≥ ∠D ∧ ∠D ≥ ∠E

theorem regular_pentagon_of_equal_sides_and_angle_conditions
  (A B C D E : Point)
  (h₁ : is_convex [A, B, C, D, E])
  (h₂ : equal_sides [A, B, C, D, E])
  (h₃ : angle_condition A B C D E) :
  ∠A = ∠B ∧ ∠B = ∠C ∧ ∠C = ∠D ∧ ∠D = ∠E := 
sorry

end regular_pentagon_of_equal_sides_and_angle_conditions_l799_799975


namespace searchlight_probability_l799_799154

theorem searchlight_probability (revolutions_per_minute : ℕ) (D : ℝ) (prob : ℝ)
  (h1 : revolutions_per_minute = 4)
  (h2 : prob = 0.6666666666666667) :
  D = (2 / 3) * (60 / revolutions_per_minute) :=
by
  -- To complete the proof, we will use the conditions given.
  sorry

end searchlight_probability_l799_799154


namespace log_sqrt_defined_l799_799285

open Real

-- Define the conditions for the logarithm and square root arguments
def log_condition (x : ℝ) : Prop := 4 * x - 7 > 0
def sqrt_condition (x : ℝ) : Prop := 2 * x - 3 ≥ 0

-- Define the combined condition
def combined_condition (x : ℝ) : Prop := x > 7 / 4

-- The proof statement
theorem log_sqrt_defined (x : ℝ) : combined_condition x ↔ log_condition x ∧ sqrt_condition x :=
by
  -- Work through the equivalence and proof steps
  sorry

end log_sqrt_defined_l799_799285


namespace surface_area_of_cylinder_l799_799846

noncomputable def cylinder_surface_area
    (r : ℝ) (V : ℝ) (S : ℝ) : Prop :=
    r = 1 ∧ V = 2 * Real.pi ∧ S = 6 * Real.pi

theorem surface_area_of_cylinder
    (r : ℝ) (V : ℝ) : ∃ S : ℝ, cylinder_surface_area r V S :=
by
  use 6 * Real.pi
  sorry

end surface_area_of_cylinder_l799_799846


namespace smallest_prime_with_digit_sum_23_l799_799662

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799662


namespace find_base_lengths_l799_799048

noncomputable def isosceles_trapezoid_base_lengths 
  (x y : ℝ) : Prop :=
  let side1 := 3
  let side2 := 5
  let midline_ratio := 5 / 11 in
  let sum_of_bases := x + y = 8 in
  let ratio_of_areas := (x + 4) / (y + 4) = midline_ratio in
  sum_of_bases ∧ ratio_of_areas

theorem find_base_lengths :
  ∃ (x y : ℝ), 
  isosceles_trapezoid_base_lengths x y ∧ 
  x = 1 ∧ y = 7 :=
by 
  sorry

end find_base_lengths_l799_799048


namespace car_initial_speed_l799_799735

theorem car_initial_speed (s t : ℝ) (h₁ : t = 15 * s^2) (h₂ : t = 3) :
  s = (Real.sqrt 2) / 5 :=
by
  sorry

end car_initial_speed_l799_799735


namespace greatest_integer_third_side_l799_799075

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end greatest_integer_third_side_l799_799075


namespace AB_length_in_cyclic_quadrilateral_l799_799484

theorem AB_length_in_cyclic_quadrilateral 
  (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (cyclic : CyclicQuadrilateral A B C D)
  (BC_diameter : diameter (circumcircle A B C D) B C)
  (BC_len : dist B C = 8)
  (BD_len : dist B D = 4 * Real.sqrt 2)
  (angle_ratio : ∃ x, angle D C A = 2 * x ∧ angle A C B = x) :
  dist A B = 2 * (Real.sqrt 6 - Real.sqrt 2) := 
sorry

end AB_length_in_cyclic_quadrilateral_l799_799484


namespace percent_volume_filled_by_cubes_l799_799141

theorem percent_volume_filled_by_cubes : 
  let box_volume := 8 * 6 * 12,
      cube_volume := 4 * 4 * 4,
      max_cubes := (8 / 4) * (6 / 4) * (12 / 4),
      total_cube_volume := max_cubes * cube_volume in
  (total_cube_volume / box_volume) * 100 = 66.67 := 
sorry

end percent_volume_filled_by_cubes_l799_799141


namespace solve_equation_l799_799989

theorem solve_equation :
  ∀ x : ℝ, (5 * x - 5 * x^3 + x^5 = 0) ↔ 
            (x = 0) ∨ 
            (x = sqrt ((5 + sqrt 5) / 2)) ∨ 
            (x = -sqrt ((5 + sqrt 5) / 2)) ∨ 
            (x = sqrt ((5 - sqrt 5) / 2)) ∨ 
            (x = -sqrt ((5 - sqrt 5) / 2)) :=
by
  sorry

end solve_equation_l799_799989


namespace incorrect_conclusion_D_l799_799357

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l799_799357


namespace smallest_prime_with_digit_sum_23_l799_799644

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799644


namespace box_cubes_percentage_l799_799147

-- Define the dimensions of the rectangular box
def box_length: ℝ := 8
def box_width: ℝ := 6
def box_height: ℝ := 12

-- Define the side length of the solid cubes
def cube_side: ℝ := 4

-- Define the expected percentage
def expected_percentage: ℝ := 66.67

-- Define the function to calculate the percentage of the volume of the box taken up by the cubes
noncomputable def volume_percentage_of_cubes (L W H side_length: ℝ) : ℝ :=
  let num_cubes_length := (L / side_length).toNat
  let num_cubes_width := (W / side_length).toNat
  let num_cubes_height := (H / side_length).toNat
  let total_num_cubes := num_cubes_length * num_cubes_width * num_cubes_height
  let volume_of_one_cube := side_length ^ 3
  let total_volume_of_cubes := total_num_cubes * volume_of_one_cube
  let volume_of_box := L * W * H
  (total_volume_of_cubes / volume_of_box) * 100

-- The theorem to be proven
theorem box_cubes_percentage:
  volume_percentage_of_cubes box_length box_width box_height cube_side = expected_percentage :=
sorry

end box_cubes_percentage_l799_799147


namespace coin_flip_probability_l799_799552

def total_outcomes := (2:ℕ)^12
def favorable_outcomes := Nat.choose 12 9

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 55 / 1024 := 
by
  sorry

end coin_flip_probability_l799_799552


namespace billion_in_scientific_notation_l799_799498

theorem billion_in_scientific_notation :
  (10^9 = 1 * 10^9) :=
by
  sorry

end billion_in_scientific_notation_l799_799498


namespace enclosed_area_rounded_l799_799455

def g (x : ℝ) : ℝ := 2 - real.sqrt (4 - x^2)

theorem enclosed_area_rounded (h : ∀ x, -2 ≤ x ∧ x ≤ 2 → g(x) = 2 - real.sqrt (4 - x^2)) :
  real.abs ((2 * real.pi - 4) / 4 - 1.14) < 0.005 :=
sorry

end enclosed_area_rounded_l799_799455


namespace arithmetic_sequence_length_l799_799896

theorem arithmetic_sequence_length :
  ∀ (a d a_n : ℕ), a = 6 → d = 4 → a_n = 154 → ∃ n: ℕ, a_n = a + (n-1) * d ∧ n = 38 :=
by
  intro a d a_n ha hd ha_n
  use 38
  rw [ha, hd, ha_n]
  -- Leaving the proof as an exercise
  sorry

end arithmetic_sequence_length_l799_799896


namespace initial_oranges_l799_799944

open Nat

theorem initial_oranges (initial_oranges: ℕ) (eaten_oranges: ℕ) (stolen_oranges: ℕ) (returned_oranges: ℕ) (current_oranges: ℕ):
  eaten_oranges = 10 → 
  stolen_oranges = (initial_oranges - eaten_oranges) / 2 →
  returned_oranges = 5 →
  current_oranges = 30 →
  initial_oranges - eaten_oranges - stolen_oranges + returned_oranges = current_oranges →
  initial_oranges = 60 :=
by
  sorry

end initial_oranges_l799_799944


namespace number_of_truth_tellers_is_twelve_l799_799235
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799235


namespace simplify_ratio_l799_799514

theorem simplify_ratio (k : ℤ) : (∃ a b : ℤ, ∀ a b, (a = 1 ∧ b = 3) → a / b = 1 / 3) :=
by
  use 1
  use 3
  intro a b h
  cases h with ha hb
  rw [ha, hb]
  exact eq.refl (1 / 3)

end simplify_ratio_l799_799514


namespace smallest_prime_with_digit_sum_23_l799_799609

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799609


namespace projection_matrix_unique_solution_l799_799517

theorem projection_matrix_unique_solution :
  ∃ a c : ℚ, let P := ![
    ![a, 18/45],
    ![c, 27/45]
  ] in P * P = P ∧ (a, c) = (1/5, 2/5) :=
by
  sorry

end projection_matrix_unique_solution_l799_799517


namespace combined_weight_l799_799554

theorem combined_weight (x y z : ℕ) (h1 : x + z = 78) (h2 : x + y = 69) (h3 : y + z = 137) : x + y + z = 142 :=
by
  -- Intermediate steps or any additional lemmas could go here
sorry

end combined_weight_l799_799554


namespace smallest_prime_with_digit_sum_23_l799_799658

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799658


namespace length_of_each_part_l799_799695

-- Conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def parts_count : ℕ := 4

-- Question
theorem length_of_each_part : total_length_in_inches / parts_count = 20 :=
by
  -- leave the proof as a sorry
  sorry

end length_of_each_part_l799_799695


namespace rounding_down_both_fractions_less_sum_l799_799967

theorem rounding_down_both_fractions_less_sum
  (a b c d : ℕ) (h1 : a * d + b * c < c * d)
  (f1 : (2 : ℚ) / 3 = (a : ℚ) / b) 
  (f2 : (5 : ℚ) / 4 = (c : ℚ) / d) :
  a / b + c / d < (23 : ℚ) / 12 := sorry

end rounding_down_both_fractions_less_sum_l799_799967


namespace speaking_order_count_l799_799749

theorem speaking_order_count : 
  let speakers := ['A', 'B', 'C', 'D', 'E', 'F'] in
  let condition := ∀ (perm : List Char), perm ∈ speakers.permutations → 
    ∃ i, perm[i] = 'A' ∧ perm[i+1] = 'B' in
  (condition → speakers.permutations.length / 120) = True :=
by
  let speakers := ['A', 'B', 'C', 'D', 'E', 'F']
  let (speakers_with_ab_block, _) := List.splitAt 2 speakers
  sorry

end speaking_order_count_l799_799749


namespace geometric_sequence_problem_l799_799826

noncomputable def geometric_sequence_satisfies_an : Prop :=
∀ (a : ℕ → ℝ) (q : ℝ),
  (∀ n, a (n + 1) = a n * q) → -- geometric property
  (a 2 + a 5 = 0) →           -- given condition
  let f (n : ℕ) := ∑ k in finset.range n, (1 / a (k + 1)) in
  (f 5 / f 2 = -11)

theorem geometric_sequence_problem : geometric_sequence_satisfies_an :=
by sorry

end geometric_sequence_problem_l799_799826


namespace total_candy_bars_correct_l799_799466

-- Define the number of each type of candy bar.
def snickers : Nat := 3
def marsBars : Nat := 2
def butterfingers : Nat := 7

-- Define the total number of candy bars.
def totalCandyBars : Nat := snickers + marsBars + butterfingers

-- Formulate the theorem about the total number of candy bars.
theorem total_candy_bars_correct : totalCandyBars = 12 :=
sorry

end total_candy_bars_correct_l799_799466


namespace isosceles_side_length_l799_799745

noncomputable def length_of_isosceles_side : ℝ :=
  let s := 2 in
  let b := 1 in
  if h : b = s / 2 then sqrt (7 / 3) else 0

theorem isosceles_side_length (s : ℝ) (b : ℝ) (cond : s = 2) (base_cond : b = s / 2) : 
  length_of_isosceles_side = sqrt (7 / 3) :=
by
  have h : b = s / 2 := base_cond
  simp [length_of_isosceles_side, cond, h]
  sorry

end isosceles_side_length_l799_799745


namespace circle_equation_range_of_a_l799_799823

variables {R : Type*} [linearOrderedField R] 

-- Given conditions
def radius (r : R) : Prop := r = real.sqrt 5

def center_on_line (x y : R) : Prop := x - y + 1 = 0

def intersects_at (x y : R) : Prop := real.sqrt 3 * x - y + 1 - real.sqrt 3 = 0

def distance_MN (d : R) : Prop := d = real.sqrt 17

-- Questions and required proofs
theorem circle_equation (a : R) (x y : R) (C_eqn1 C_eqn2 : Prop) 
  (h_radius : radius (real.sqrt 5)) 
  (h_center_on_line : center_on_line x y) 
  (h_intersects_at : intersects_at x y) 
  (h_distance_MN : distance_MN (real.sqrt 17)) : 
  (C_eqn1 ∨ C_eqn2) := 
sorry

theorem range_of_a (a : R) 
  (h_center_integral : ∃ x y : R, x ∈ (ℤ : set R) ∧ y ∈ (ℤ : set R) ∧ center_on_line x y)
  (h_radius : radius (real.sqrt 5))  
  (h_intersects_at : intersects_at x y) 
  (h_distance_MN : distance_MN (real.sqrt 17)) 
  (h_a_range : 0 ≤ a ∧ a ≤ 5) : 
  (∀ m : R, ∃ l₃_intersects : Prop, l₃_intersects) := 
sorry

end circle_equation_range_of_a_l799_799823


namespace longest_tape_length_l799_799702

theorem longest_tape_length {a b c : ℕ} (h1 : a = 100) (h2 : b = 225) (h3 : c = 780) : 
  Int.gcd (Int.gcd a b) c = 5 := by
  sorry

end longest_tape_length_l799_799702


namespace sin_of_angle_passing_through_point_l799_799852

theorem sin_of_angle_passing_through_point :
  (α : ℝ) → (p : ℝ × ℝ) → 
  (h_cond : p = (4, -3)) → sin α = -3/5 :=
by
  intros α p h_cond
  -- using sorry to skip full proof implementation 
  sorry

end sin_of_angle_passing_through_point_l799_799852


namespace smallest_prime_with_digit_sum_23_l799_799583

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799583


namespace smallest_d_3150_eq_14_l799_799909

noncomputable def smallest_d (n : ℕ) : ℕ :=
  if (∃ k : ℕ, prime k ∧ n * 7 * 2 = k^2) then 14 else sorry

theorem smallest_d_3150_eq_14 : smallest_d 3150 = 14 :=
by {
  simp [smallest_d],
  sorry
}

end smallest_d_3150_eq_14_l799_799909


namespace max_area_projection_unit_cube_l799_799949

theorem max_area_projection_unit_cube : 
  let C : set (ℝ × ℝ × ℝ) := { (x, y, z) | 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 }
  let p : (ℝ × ℝ × ℝ) → set (ℝ × ℝ) := λ (x, y, z), {(x', y') | x' = x ∧ y' = y}
  in ∃ plane : (ℝ × ℝ × ℝ) → set (ℝ × ℝ), plane = p → 
     let projection := λ (s : set (ℝ × ℝ × ℝ)), { (x, y) | ∃ z, (x, y, z) ∈ s }
     in (∃ C_projection : set (ℝ × ℝ), C_projection = projection C → 
           ∃ max_area : ℝ, max_area = sqrt 3 ∧ area C_projection = max_area) :=
sorry

end max_area_projection_unit_cube_l799_799949


namespace number_of_truthful_warriors_l799_799208

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799208


namespace num_valid_numbers_l799_799525

def is_valid_number (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 3000 ∧
  let digits := (n.div 1000) % 10 :: (n.div 100) % 10 :: (n.div 10) % 10 :: (n % 10) :: [] in
  digits.count (λ d => d = 2) = 1 ∧
  (∃ d, d ≠ 2 ∧ d < 10 ∧ digits.count (λ x => x = d) = 2)

theorem num_valid_numbers : {n : ℕ | is_valid_number n}.to_finset.card = 384 :=
  sorry

end num_valid_numbers_l799_799525


namespace problem_arith_seq_l799_799329

variables {a : ℕ → ℝ} (d : ℝ)
def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arith_seq (h_arith : is_arithmetic_sequence a) 
  (h_condition : a 1 + a 6 + a 11 = 3) 
  : a 3 + a 9 = 2 :=
sorry

end problem_arith_seq_l799_799329


namespace smallest_sector_angle_circle_l799_799431

theorem smallest_sector_angle_circle (n : ℕ) (d a_1 : ℕ) (angles : Fin n → ℕ) :
  n = 15 →
  (∀ i, angles i = a_1 + i * d) →
  (∑ i, angles i) = 360 →
  angles 0 = 10 :=
by
  intros h1 h2 h3
  sorry

end smallest_sector_angle_circle_l799_799431


namespace box_cubes_percentage_l799_799150

-- Define the dimensions of the rectangular box
def box_length: ℝ := 8
def box_width: ℝ := 6
def box_height: ℝ := 12

-- Define the side length of the solid cubes
def cube_side: ℝ := 4

-- Define the expected percentage
def expected_percentage: ℝ := 66.67

-- Define the function to calculate the percentage of the volume of the box taken up by the cubes
noncomputable def volume_percentage_of_cubes (L W H side_length: ℝ) : ℝ :=
  let num_cubes_length := (L / side_length).toNat
  let num_cubes_width := (W / side_length).toNat
  let num_cubes_height := (H / side_length).toNat
  let total_num_cubes := num_cubes_length * num_cubes_width * num_cubes_height
  let volume_of_one_cube := side_length ^ 3
  let total_volume_of_cubes := total_num_cubes * volume_of_one_cube
  let volume_of_box := L * W * H
  (total_volume_of_cubes / volume_of_box) * 100

-- The theorem to be proven
theorem box_cubes_percentage:
  volume_percentage_of_cubes box_length box_width box_height cube_side = expected_percentage :=
sorry

end box_cubes_percentage_l799_799150


namespace find_num_3_year_olds_l799_799535

noncomputable def num_4_year_olds := 20
noncomputable def num_5_year_olds := 15
noncomputable def num_6_year_olds := 22
noncomputable def average_class_size := 35
noncomputable def num_students_class1 (num_3_year_olds : ℕ) := num_3_year_olds + num_4_year_olds
noncomputable def num_students_class2 := num_5_year_olds + num_6_year_olds
noncomputable def total_students (num_3_year_olds : ℕ) := num_students_class1 num_3_year_olds + num_students_class2

theorem find_num_3_year_olds (num_3_year_olds : ℕ) : 
  (total_students num_3_year_olds) / 2 = average_class_size → num_3_year_olds = 13 :=
by
  sorry

end find_num_3_year_olds_l799_799535


namespace solid_volume_is_correct_l799_799050

-- Define the side length s
def s : ℝ := 6 * Real.sqrt 2

-- Define the dimensions of the solid
def base_width : ℝ := s
def base_length : ℝ := 2 * s
def upper_edge_length : ℝ := 3 * s
def edge_length : ℝ := s

-- The volume of the solid in question
def volume_of_solid : ℝ := 486 * Real.sqrt 2

-- The statement of the theorem
theorem solid_volume_is_correct : 
  (volume_of_solid = 486 * Real.sqrt 2) :=
sorry

end solid_volume_is_correct_l799_799050


namespace smallest_prime_with_digit_sum_23_l799_799587

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799587


namespace angle_A_is_120_degrees_l799_799939

theorem angle_A_is_120_degrees (a b c : ℝ) (ha : a = 7) (hb : b = 5) (hc : c = 3) :
  ∃ A : ℝ, A = 120 ∧ 
  ∀ B C : ℝ, B > 0 ∧ C > 0 ∧ a = √(b^2 + c^2 - 2*b*c*real.cos A) → angle A = 120 :=  
  sorry

end angle_A_is_120_degrees_l799_799939


namespace bogatyrs_truthful_count_l799_799243

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799243


namespace number_of_elements_unchanged_l799_799539

theorem number_of_elements_unchanged 
  (numbers : List ℝ) 
  (h_avg : (numbers.sum / numbers.length) = 26) 
  (h_new_avg : ((numbers.map (λ x, 5 * x)).sum / numbers.length) = 130) : 
  numbers.length = numbers.length :=
by
  sorry

end number_of_elements_unchanged_l799_799539


namespace monotonic_range_of_a_l799_799912

theorem monotonic_range_of_a (a : ℝ) :
  (a ≥ 9 ∨ a ≤ 3) → 
  ∀ x y : ℝ, (1 ≤ x ∧ x ≤ 4) → (1 ≤ y ∧ y ≤ 4) → x ≤ y → 
  (x^2 + (1-a)*x + 3) ≤ (y^2 + (1-a)*y + 3) :=
by
  intro ha x y hx hy hxy
  sorry

end monotonic_range_of_a_l799_799912


namespace boy_late_time_first_day_l799_799115

constants d t1 t2 t_late : ℝ
constants speed1 speed2 : ℝ
constants early_time : ℝ

axiom ax1 : d = 3
axiom ax2 : t1 = d / 6
axiom ax3 : t2 = d / 12
axiom ax4 : early_time = 8 / 60
axiom ax5 : speed1 = 6
axiom ax6 : speed2 = 12

def Δt := t1 - t2
def total_late_time := Δt + early_time
def total_late_time_in_minutes := total_late_time * 60

theorem boy_late_time_first_day : total_late_time_in_minutes = 23 := by
  sorry

end boy_late_time_first_day_l799_799115


namespace Harvard_attendance_l799_799415

theorem Harvard_attendance:
  (total_applicants : ℕ) (acceptance_rate : ℝ) (attendance_rate : ℝ) 
  (h1 : total_applicants = 20000) 
  (h2 : acceptance_rate = 0.05) 
  (h3 : attendance_rate = 0.9) :
  ∃ (number_attending : ℕ), number_attending = 900 := 
by 
  sorry

end Harvard_attendance_l799_799415


namespace distance_between_intersections_l799_799041

noncomputable def intersection_points := 
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  (⟨x1, 2⟩, ⟨x2, 2⟩)  -- Points C and D

noncomputable def calculate_distance (C D : ℝ × ℝ) : ℝ := 
  Real.abs (C.1 - D.1)

theorem distance_between_intersections :
  let C := (⟨(-1 + Real.sqrt 22) / 3, 2⟩)
  let D := (⟨(-1 - Real.sqrt 22) / 3, 2⟩)
  calculate_distance C D = (2 * Real.sqrt 22) / 3 :=
by sorry

end distance_between_intersections_l799_799041


namespace intersection_A_B_l799_799878

open Set

def A : Set ℝ := {1, 2, 1/2}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_A_B : A ∩ B = { 1 } := by
  sorry

end intersection_A_B_l799_799878


namespace nancy_total_money_l799_799972

def total_money (n_five n_ten n_one : ℕ) : ℕ :=
  (n_five * 5) + (n_ten * 10) + (n_one * 1)

theorem nancy_total_money :
  total_money 9 4 7 = 92 :=
by
  sorry

end nancy_total_money_l799_799972


namespace car_y_start_time_l799_799175

theorem car_y_start_time : 
  ∀ (t m : ℝ), 
  (35 * (t + m) = 294) ∧ (40 * t = 294) → 
  t = 7.35 ∧ m = 1.05 → 
  m * 60 = 63 :=
by
  intros t m h1 h2
  sorry

end car_y_start_time_l799_799175


namespace incorrect_conclusion_l799_799352

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l799_799352


namespace touring_plans_l799_799973

theorem touring_plans :
  let options := {1, 2, 3} -- 1: Chengxiang Road, 2: Haxi Station, 3: Harbin Street
  let total_choices := (options × options × options × options).to_finset.card
  let no_haxi_choices := ({1, 3} × {1, 3} × {1, 3} × {1, 3}).to_finset.card
  total_choices - no_haxi_choices = 65 :=
by
  let options := {1, 2, 3}
  let total_choices := (options × options × options × options).to_finset.card
  let no_haxi_choices := ({1, 3} × {1, 3} × {1, 3} × {1, 3}).to_finset.card
  have h1: total_choices = 81 := by sorry
  have h2: no_haxi_choices = 16 := by sorry
  have h3: 81 - 16 = 65 := by norm_num
  rw [h1, h2]
  exact h3

end touring_plans_l799_799973


namespace Ria_original_savings_l799_799684

variables {R F : ℕ}

def initial_ratio (R F : ℕ) : Prop :=
  R * 3 = F * 5

def withdrawn_amount (R : ℕ) : ℕ :=
  R - 160

def new_ratio (R' F : ℕ) : Prop :=
  R' * 5 = F * 3

theorem Ria_original_savings (initial_ratio: initial_ratio R F)
  (new_ratio: new_ratio (withdrawn_amount R) F) : 
  R = 250 :=
by
  sorry

end Ria_original_savings_l799_799684


namespace arithmetic_sequence_a12_l799_799933

-- Define the arithmetic sequence
noncomputable def a (n : ℕ) : ℝ := -5 / 2 + n * (3 / 2)

-- Given conditions
def cond1 : Prop := a 7 + a 9 = 16
def cond2 : Prop := a 4 = 2

-- Statement to prove
theorem arithmetic_sequence_a12 : cond1 ∧ cond2 → a 12 = 14 :=
by
  intro h
  cases h with h1 h2
  sorry

end arithmetic_sequence_a12_l799_799933


namespace smallest_prime_with_digit_sum_23_l799_799570

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799570


namespace smallest_prime_with_digit_sum_23_l799_799624

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799624


namespace smallest_prime_with_digit_sum_23_l799_799574

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799574


namespace isosceles_triangle_angles_l799_799928

theorem isosceles_triangle_angles (A B C : ℝ)
    (h_iso : A = B ∨ B = C ∨ C = A)
    (h_one_angle : A = 36 ∨ B = 36 ∨ C = 36)
    (h_sum_angles : A + B + C = 180) :
  (A = 36 ∧ B = 36 ∧ C = 108) ∨
  (A = 72 ∧ B = 72 ∧ C = 36) :=
by 
  sorry

end isosceles_triangle_angles_l799_799928


namespace monotonic_decreasing_interval_of_f_l799_799520

noncomputable def f (x : ℝ) : ℝ := real.log (2 * x - x ^ 2)

theorem monotonic_decreasing_interval_of_f :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f x > f (x + 0.001)) :=
begin
  sorry
end

end monotonic_decreasing_interval_of_f_l799_799520


namespace find_difference_l799_799441

theorem find_difference (a b : ℕ) (h1 : Nat.coprime a b) (h2 : a > b) (h3 : (a^3 - b^3) / (a - b)^3 = 50 / 3) :
  a - b = 3 :=
sorry

end find_difference_l799_799441


namespace problem1_eq_7_l799_799485

-- First part of the problem
-- Define the two expressions to be simplified and evaluated
def expr1 (x : ℤ) : ℤ := (4-x)*(2*x + 1) + 3*x*(x-3)

-- Define evaluation of the first expression
theorem problem1_eq_7 : expr1 (-1) = 7 :=
sorry

-- Second part of the problem
-- Define the expression to be simplified and evaluated
def expr2 (x y : ℤ) : ℤ := (x^2 + 4*x*y + 4*y^2 - (9*x^2 - y^2) - 5*y^2) / (-1 / (2 * x))

noncomputable theorem problem2_eq_12 : expr2 1 (1/2) = 12 :=
sorry

end problem1_eq_7_l799_799485


namespace max_digit_d_of_form_7d733e_multiple_of_33_l799_799261

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l799_799261


namespace calc_f_g_h_2_l799_799445

def f (x : ℕ) : ℕ := x + 5
def g (x : ℕ) : ℕ := x^2 - 8
def h (x : ℕ) : ℕ := 2 * x + 1

theorem calc_f_g_h_2 : f (g (h 2)) = 22 := by
  sorry

end calc_f_g_h_2_l799_799445


namespace four_intersections_exist_l799_799883

open_locale classical

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

-- Definitions of points, circles, and intersections
def circle (center : α) (radius : ℝ) := {p : α | dist center p = radius}

noncomputable def intersection_points (c1 c2 : set α) : set α := c1 ∩ c2

-- Main statement
theorem four_intersections_exist
  (X Y A B C D : α)
  (Ω : set α)
  (circ1 circ2 : set α)
  (H1: X ∈ intersection_points circ1 circ2)
  (H2: Y ∈ intersection_points circ1 circ2)
  (H3: A ∈ intersection_points Ω circ1)
  (H4: B ∈ intersection_points Ω circ2)
  (H5: C ∈ Ω)
  (H6: D ∈ Ω)
  (H7: C ∈ line[X, Y])
  (H8: D ∈ line[X, Y]) :
  ∃ P Q R S : α, 
    ∀ (circ : set α), 
      circ = set_of (λ p, dist A p = dist circ1 p ∧ dist B p = dist circ2 p) → 
      ∀ (l : set α), l = line_segment C D → 
      ∀ p q : α, p ≠ q →
        ((p ∈ l ∧ q ∈ l) → (line [A, p] = line [A, q] ∨ line [B, p] = line [B, q])) :=
sorry

end four_intersections_exist_l799_799883


namespace prove_tan_alpha_minus_beta_is_half_prove_tan_beta_is_minus_seventh_prove_2_alpha_minus_beta_l799_799853

variable (α β : ℝ)

-- Given conditions
axiom angle_vertex_origin : α ∈ (0, π) ∧ β ∈ (0, π)
axiom point_on_terminal_side : ∃ P, P = (3, 1)
axiom tan_alpha_minus_beta : 
  tan (α - β) = (sin (2 * (π / 2 - α)) + 4 * (cos α)^2) / (10 * (cos α)^2 + cos (3 * π / 2 - 2 * α))

-- Correct answers to prove
theorem prove_tan_alpha_minus_beta_is_half : tan (α - β) = 1 / 2 := 
  by
  sorry

theorem prove_tan_beta_is_minus_seventh 
  (h1: tan (α - β) = 1 / 2): 
  tan β = - 1 / 7 := 
  by
  sorry

theorem prove_2_alpha_minus_beta 
  (h1: tan (α - β) = 1 / 2) 
  (h2: tan β = - 1 / 7): 
  2 * α - β = - 3 * π / 4 := 
  by
  sorry

end prove_tan_alpha_minus_beta_is_half_prove_tan_beta_is_minus_seventh_prove_2_alpha_minus_beta_l799_799853


namespace larger_segment_of_triangle_l799_799049

theorem larger_segment_of_triangle (a b c : ℕ) (h1: a = 35) (h2: b = 65) (h3: c = 85) :
  ∃ (x : ℕ), ∃ (y : ℕ), 170 * x = 4225 ∧ x ≤ 85 ∧ (85 - x) = 60 :=
by
  use 25
  use y
  split
  { exact 170 * 25 = 4225 }
  sorry

end larger_segment_of_triangle_l799_799049


namespace length_BC_l799_799793

theorem length_BC {A B C : ℝ} (r1 r2 : ℝ) (AB : ℝ) (h1 : r1 = 8) (h2 : r2 = 5) (h3 : AB = r1 + r2) :
  C = B + (65 : ℝ) / 3 :=
by
  -- Problem set-up and solving comes here if needed
  sorry

end length_BC_l799_799793


namespace largest_of_four_numbers_l799_799811

theorem largest_of_four_numbers 
  (a b c d : ℝ) 
  (h1 : a + 5 = b^2 - 1) 
  (h2 : a + 5 = c^2 + 3) 
  (h3 : a + 5 = d - 4) 
  : d > max (max a b) c :=
sorry

end largest_of_four_numbers_l799_799811


namespace ratio_of_shorts_to_pants_is_half_l799_799169

-- Define the parameters
def shirts := 4
def pants := 2 * shirts
def total_clothes := 16

-- Define the number of shorts
def shorts := total_clothes - (shirts + pants)

-- Define the ratio
def ratio := shorts / pants

-- Prove the ratio is 1/2
theorem ratio_of_shorts_to_pants_is_half : ratio = 1 / 2 :=
by
  -- Start the proof, but leave it as sorry
  sorry

end ratio_of_shorts_to_pants_is_half_l799_799169


namespace boat_speed_in_still_water_l799_799051

theorem boat_speed_in_still_water 
  (rate_of_current : ℝ) 
  (time_in_hours : ℝ) 
  (distance_downstream : ℝ)
  (h_rate : rate_of_current = 5) 
  (h_time : time_in_hours = 15 / 60) 
  (h_distance : distance_downstream = 6.25) : 
  ∃ x : ℝ, (distance_downstream = (x + rate_of_current) * time_in_hours) ∧ x = 20 :=
by 
  -- Main theorem statement, proof omitted for brevity.
  sorry

end boat_speed_in_still_water_l799_799051


namespace warriors_truth_tellers_l799_799198

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799198


namespace difference_between_twice_smaller_and_larger_is_three_l799_799015

theorem difference_between_twice_smaller_and_larger_is_three
(S L x : ℕ) 
(h1 : L = 2 * S - x) 
(h2 : S + L = 39)
(h3 : S = 14) : 
2 * S - L = 3 := 
sorry

end difference_between_twice_smaller_and_larger_is_three_l799_799015


namespace people_sharing_cookies_l799_799693

theorem people_sharing_cookies (total_cookies : ℕ) (cookies_per_person : ℕ) (people : ℕ) 
  (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) (h3 : total_cookies = cookies_per_person * people) : 
  people = 6 :=
by
  sorry

end people_sharing_cookies_l799_799693


namespace max_d_77733e_divisible_by_33_l799_799259

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l799_799259


namespace prove_age_difference_l799_799747

variable (A C n : ℝ) -- Ana's age, Carmen's age, and the difference in years

-- Condition 1: Age difference
def age_difference := A = C + n

-- Condition 2: Two years ago, Ana was 4 times as old as Carmen
def age_condition2 := A - 2 = 4 * (C - 2)

-- Condition 3: This year, Ana's age is the cube of Carmen's age
def age_condition3 := A = C^3

-- Proof statement: proving that under these conditions n = 1.875
theorem prove_age_difference :
  (age_difference A C n) → (age_condition2 A C n) → (age_condition3 A C) → n = 1.875 :=
by
  intros h1 h2 h3
  sorry

end prove_age_difference_l799_799747


namespace number_of_truth_tellers_is_twelve_l799_799236
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799236


namespace max_tickets_l799_799493

/-- Given the cost of each ticket and the total amount of money available, 
    prove that the maximum number of tickets that can be purchased is 8. -/
theorem max_tickets (ticket_cost : ℝ) (total_amount : ℝ) (h1 : ticket_cost = 18.75) (h2 : total_amount = 150) :
  (∃ n : ℕ, ticket_cost * n ≤ total_amount ∧ ∀ m : ℕ, ticket_cost * m ≤ total_amount → m ≤ n) ∧
  ∃ n : ℤ, (n : ℤ) = 8 :=
by
  sorry

end max_tickets_l799_799493


namespace arithmetic_sequence_a3a6_l799_799443

theorem arithmetic_sequence_a3a6 (a : ℕ → ℤ)
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_inc : ∀ n, a n < a (n + 1))
  (h_eq : a 3 * a 4 = 45): 
  a 2 * a 5 = 13 := 
sorry

end arithmetic_sequence_a3a6_l799_799443


namespace smallest_prime_digit_sum_23_l799_799616

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799616


namespace truthfulness_count_l799_799224

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799224


namespace Bella_hits_10_points_l799_799482

-- Definitions of the names of the friends and their respective scores
def Adam_score := 18
def Bella_score := 15
def Carlos_score := 12
def Diana_score := 9
def Evan_score := 20
def Fiona_score := 13
def Grace_score := 17

-- Define a proof that Bella hits the region worth 10 points given the conditions
theorem Bella_hits_10_points :
  (∃ a1 a2 b1 b2 c1 c2 d1 d2 e1 e2 f1 f2 g1 g2 : ℕ,
    a1 ≠ a2 ∧ b1 ≠ b2 ∧ c1 ≠ c2 ∧ d1 ≠ d2 ∧ e1 ≠ e2 ∧ f1 ≠ f2 ∧ g1 ≠ g2 ∧
    a1 + a2 = Adam_score ∧
    b1 + b2 = Bella_score ∧
    c1 + c2 = Carlos_score ∧
    d1 + d2 = Diana_score ∧
    e1 + e2 = Evan_score ∧
    f1 + f2 = Fiona_score ∧
    g1 + g2 = Grace_score ∧
    {a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2, g1, g2}.card = 14 ∧ -- Ensuring all points are unique
    (b1 = 10 ∨ b2 = 10)) :=
  sorry

end Bella_hits_10_points_l799_799482


namespace six_moves_to_alternating_l799_799789

-- Definitions based on the conditions in the problem
def coins : Type := Fin 8
def flip (configuration : coins → Bool) (i : coins) : coins → Bool :=
  fun c => if c = i ∨ c = i + 1 % 8 then !configuration c else configuration c

def initial_configuration : coins → Bool := fun _ => true

def alternating_configuration (configuration : coins → Bool) : Prop :=
  ∀ i : coins, configuration i ≠ configuration (i + 1 % 8)

-- Main problem translated into a Lean 4 statement
theorem six_moves_to_alternating :
  ∃ seq : List (coins → Bool → Bool), seq.length = 6 ∧
    alternating_configuration (seq.foldl flip initial_configuration) :=
  sorry

end six_moves_to_alternating_l799_799789


namespace total_cost_of_barbed_wire_l799_799994

theorem total_cost_of_barbed_wire (A : ℕ) (c : ℚ) (wg : ℕ) (gates : ℕ) :
  A = 3136 → c = 1.20 → wg = 1 → gates = 2 →
  let s := Nat.sqrt A,
      P := 4 * s,
      lw := P - (gates * wg),
      total_cost := lw * c
  in total_cost = 266.40 :=
by
  intros hA hc hwg hgates
  let s := Nat.sqrt A
  let P := 4 * s
  let lw := P - (hgates * hwg)
  let total_cost := lw * c
  have hA : A = 3136 := by assumption
  have hc : c = 1.20 := by assumption
  have hwg : wg = 1 := by assumption
  have hgates : gates = 2 := by assumption
  sorry

end total_cost_of_barbed_wire_l799_799994


namespace find_x_conditions_l799_799452

theorem find_x_conditions (a : ℝ) (n : ℕ) (h_n : n > 1) :
  {x : ℝ | (root n (x ^ n - a ^ n) + root n (2 * a ^ n - x ^ n) = a)} =
  if a = 0 then 
    if even n then {0} 
    else set.univ 
  else 
    if odd n then {a * root n 2, a} 
    else {x : ℝ | x = a * root n 2 ∨ x = -a * root n 2 ∨ x = a ∨ x = -a} :=
sorry

end find_x_conditions_l799_799452


namespace sum_of_interior_angles_l799_799915

theorem sum_of_interior_angles (n : ℕ) (h : 4 ≤ n) : (∑ i in (range n).erase (-1), ang i) = (n - 2) * 180 :=
by sorry

end sum_of_interior_angles_l799_799915


namespace correct_functional_relationship_l799_799094

-- Definitions for the problem
def is_function (X Y : Type) (f : X → Y) : Prop :=
  ∀ x y1 y2, f x = y1 ∧ f x = y2 → y1 = y2

-- Conditions translated from the problem
def student_gender_math_score (gender : Type) (score: Type) : Prop := 
  ¬ is_function gender score (λ _ , sorry)

def work_environment_health_condition (work_env : Type) (health : Type) : Prop := 
  ¬ is_function work_env health (λ _ , sorry)

def daughter_height_father_height (daughter_height : Type) (father_height : Type) : Prop := 
  ¬ is_function father_height daughter_height (λ _ , sorry)

def side_length_area (side_length : ℝ) (area : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, f = λ s, (sqrt 3 / 4) * s^2 ∧ is_function side_length area f

-- Main proof problem: prove that side_length_area is the only functional relationship.
theorem correct_functional_relationship : 
  (∃ (gender : Type) (score: Type), student_gender_math_score gender score) ∧
  (∃ (work_env : Type) (health : Type), work_environment_health_condition work_env health) ∧
  (∃ (father_height : Type) (daughter_height : Type), daughter_height_father_height father_height daughter_height) ∧
  (∃ (side_length : ℝ) (area : ℝ), side_length_area side_length area) →
  (∃ (side_length : ℝ) (area : ℝ), side_length_area side_length area) ∧ 
  ¬ (∃ (gender : Type) (score: Type), student_gender_math_score gender score) ∧
  ¬ (∃ (work_env : Type) (health : Type), work_environment_health_condition work_env health) ∧
  ¬ (∃ (father_height : Type) (daughter_height : Type), daughter_height_father_height father_height daughter_height) := 
by sorry

end correct_functional_relationship_l799_799094


namespace prove_inequality_l799_799111

variable (a b c : ℝ)

def problem_conditions : Prop :=
  a = (-0.3)^0 ∧ b = 0.32 ∧ c = 20.3

theorem prove_inequality (h : problem_conditions a b c) : b < a ∧ a < c :=
by {
  rcases h with ⟨ha, hb, hc⟩,
  have h1 : a = 1, from calc
    a = (-0.3)^0 : by rw ha
    ... = 1          : by norm_num,
  have h2 : 0 < b ∧ b < 1, from calc
    0 < b          : by norm_num
    ... < 1       : by rw hb; norm_num,
  have h3 : 1 < c, from calc
    1 < c          : by rw hc; norm_num,
  exact ⟨h2.right, lt_trans (lt_of_le_of_eq le_rfl h1.symm) h3⟩
}

end prove_inequality_l799_799111


namespace smallest_prime_with_digit_sum_23_l799_799566

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799566


namespace magnitude_a_plus_b_l799_799500
open Real

-- Given a and b
def a : ℝ × ℝ := (2, 0)
def b (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

-- Conditions
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

def angle_condition (θ : ℝ) : Prop := θ = π / 3
def b_magnitude_condition (θ : ℝ) : Prop := magnitude (b θ) = 1

-- Proving the result
theorem magnitude_a_plus_b : 
  ∀ (θ : ℝ), angle_condition θ ∧ b_magnitude_condition θ → 
  magnitude (a.1 + b θ.1, a.2 + b θ.2) = sqrt 7 := 
by
  intros θ h_cond
  rw [angle_condition, b_magnitude_condition] at h_cond
  cases h_cond with h_angle h_b_mag
  sorry

end magnitude_a_plus_b_l799_799500


namespace evaluate_expression_l799_799791

theorem evaluate_expression :
  (10 ^ (-4 : ℤ) * 7 ^ 0 / 10 ^ (-5 : ℤ) = 10) :=
by
  sorry

end evaluate_expression_l799_799791


namespace required_shift_to_obtain_f_l799_799862

def omega : ℝ := sorry
def f (x : ℝ) : ℝ := Real.cos (omega * x - π / 6)
def distance_between_symmetry_and_zero_point : ℝ := π / 4

theorem required_shift_to_obtain_f (h1 : omega > 0)
                                  (h2 : distance_between_symmetry_and_zero_point = π / 4) :
  -- The graph of y = f(x) can be obtained by shifting the graph of y = cos(omega * x)
  -- by π / 12 units to the right
  Real.cos (omega * (x - π / 12)) = f x :=
sorry

end required_shift_to_obtain_f_l799_799862


namespace relationship_among_roots_l799_799287

variable {R : Type _} [LinearOrderedField R]

def f (x m n : R) : R :=
  (x - m) * (x - n) + 2

noncomputable def α_β {m n : R} (h : m < n) : R × R :=
  let αβ := Classical.some (Exists.intro_pair
    (((x - α) * (x - β) + 2 = 0) ∧ (α < β)))
  in ⟨αβ.1, αβ.2⟩

theorem relationship_among_roots (m n : R) (h1 : m < n) (α β : R)
  (h2 : f α m n = 0) (h3 : f β m n = 0) (h4 : α < β) :
  m < α ∧ α < β ∧ β < n := by
  sorry

end relationship_among_roots_l799_799287


namespace find_x_l799_799294

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end find_x_l799_799294


namespace exists_subset_S_l799_799451

def is_adjacent (p1 p2 : ℤ × ℤ) : Prop :=
  |p1.1 - p2.1| + |p1.2 - p2.2| = 1

def is_in_S (p : ℤ × ℤ) : Prop :=
  (p.1 + 2 * p.2) % 5 = 0

theorem exists_subset_S :
  ∃ S : set (ℤ × ℤ), (∀ p : ℤ × ℤ, p ∈ S ∨ (∃ q : ℤ × ℤ, is_adjacent p q ∧ q ∈ S)) ∧ 
                      (∀ p : ℤ × ℤ, is_in_S p → p ∈ S) :=
sorry

end exists_subset_S_l799_799451


namespace warriors_truth_tellers_l799_799201

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799201


namespace volume_percentage_correct_l799_799136

-- The dimensions of the rectangular box
def length := 8
def width := 6
def height := 12

-- The side length of the cubes
def cube_edge := 4

-- Calculate the volume of the box
def box_volume : ℕ := length * width * height

-- Calculate how many cubes fit in the box
def cubes_in_length := length / cube_edge
def cubes_in_width := width / cube_edge
def cubes_in_height := height / cube_edge

-- Calculate the volume of the part filled with cubes
def cubes_volume := (cubes_in_length * cube_edge) * (cubes_in_width * cube_edge) * (cubes_in_height * cube_edge)

-- Calculate the ratio of the filled volume to the box volume
def volume_ratio := cubes_volume / box_volume

-- Convert the ratio to a percentage
noncomputable def volume_percentage := (volume_ratio : ℝ) * 100

-- Statement of the problem
theorem volume_percentage_correct : volume_percentage = 66.67 := by
  -- Proof is not required, so we use 'sorry'
  sorry

end volume_percentage_correct_l799_799136


namespace recipe_calls_for_sugar_l799_799970

variable (sugar_already_put_in : ℕ)
variable (sugar_to_add : ℕ)
variable (x : ℕ)

theorem recipe_calls_for_sugar : (sugar_already_put_in = 10) ∧ (sugar_to_add = 1) → x = 11 :=
  by
    intro h
    cases h with h1 h2
    have x_def : x = sugar_already_put_in + sugar_to_add := by sorry
    rw [h1, h2] at x_def
    exact x_def

end recipe_calls_for_sugar_l799_799970


namespace cos_sq_sub_sin_sq_pi_div_12_l799_799279

theorem cos_sq_sub_sin_sq_pi_div_12 : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
by
  sorry

end cos_sq_sub_sin_sq_pi_div_12_l799_799279


namespace probability_prime_is_half_l799_799089

def numbers := {3, 7, 8, 9, 2.5, 11}
def is_prime (n : ℕ) : Prop := sorry -- Place holder for prime definition, normally derived from Library

noncomputable def probability_prime : ℚ :=
  let primes := {3, 7, 11} -- prime numbers among the set
  let total := 6 -- total number of sectors
  let count_primes := 3 -- number of prime sectors
  count_primes / total

theorem probability_prime_is_half : probability_prime = 1/2 := by
  sorry

end probability_prime_is_half_l799_799089


namespace min_value_of_f_l799_799043

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)

theorem min_value_of_f : ∀ x ∈ set.Icc (2:ℝ) (4:ℝ), f x ≥ f 4 := 
by
  sorry

end min_value_of_f_l799_799043


namespace mr_bodhi_sheep_problem_l799_799971

theorem mr_bodhi_sheep_problem : 
  let cows := 20
  let foxes := 15
  let zebras := 3 * foxes
  let cow_equiv := 3
  let fox_equiv := 2
  let zebra_equiv := 5
  let required_weight := 300
  let total_weight := cows * cow_equiv + foxes * fox_equiv + zebras * zebra_equiv
  let excess_weight := total_weight - required_weight
  in excess_weight = 15 :=
by 
  let cows := 20
  let foxes := 15
  let zebras := 3 * foxes
  let cow_equiv := 3
  let fox_equiv := 2
  let zebra_equiv := 5
  let required_weight := 300
  let total_weight := cows * cow_equiv + foxes * fox_equiv + zebras * zebra_equiv
  let excess_weight := total_weight - required_weight
  show excess_weight = 15
  sorry

end mr_bodhi_sheep_problem_l799_799971


namespace truthful_warriors_count_l799_799216

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799216


namespace sandy_spent_on_shorts_l799_799023

variable (amount_on_shirt amount_on_jacket total_amount amount_on_shorts : ℝ)

theorem sandy_spent_on_shorts :
  amount_on_shirt = 12.14 →
  amount_on_jacket = 7.43 →
  total_amount = 33.56 →
  amount_on_shorts = total_amount - amount_on_shirt - amount_on_jacket →
  amount_on_shorts = 13.99 :=
by
  intros h_shirt h_jacket h_total h_computation
  sorry

end sandy_spent_on_shorts_l799_799023


namespace book_pages_total_l799_799692

-- Define the conditions
def pagesPerNight : ℝ := 120.0
def nights : ℝ := 10.0

-- State the theorem to prove
theorem book_pages_total : pagesPerNight * nights = 1200.0 := by
  sorry

end book_pages_total_l799_799692


namespace polynomial_roots_unique_b_c_l799_799192

theorem polynomial_roots_unique_b_c :
    ∀ (r : ℝ), (r ^ 2 - 2 * r - 1 = 0) → (r ^ 5 - 29 * r - 12 = 0) :=
by
    sorry

end polynomial_roots_unique_b_c_l799_799192


namespace domain_f_max_min_f_l799_799858

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log 3 (3 - x) - Real.log 3 (1 + x)

theorem domain_f :
  {x : ℝ | f x = f x} ⊆ Ioo (-1 : ℝ) 3 :=
by sorry

theorem max_min_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  f x = -1 ∨ f x = 2 :=
by sorry

end domain_f_max_min_f_l799_799858


namespace geometry_problem_l799_799401

-- Define the acute-angled triangle ABC
variables {A B C H P K M : Type}
variable [affine_space ℝ]
variables [EuclideanGeometry ℝ]

open EuclideanGeometry

-- Given conditions
variables (ABC_acute : is_acute_triangle A B C)
variables (CH_altitude : is_altitude C H)
variables (P_reflection : is_reflection A BC P)
variables (CH_circumcircle_intersect_K : intersects_again CH (circumcircle A C P) K)
variables (KP_intersects_AB_at_M : intersects KP AB M)

-- To prove
theorem geometry_problem : AC = CM :=
by
  sorry

end geometry_problem_l799_799401


namespace binom_10_0_eq_1_l799_799185

theorem binom_10_0_eq_1 :
  (Nat.choose 10 0) = 1 :=
by
  sorry

end binom_10_0_eq_1_l799_799185


namespace smallest_prime_digit_sum_23_l799_799611

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799611


namespace mean_properties_l799_799995

theorem mean_properties (a b c : ℝ) 
    (h1 : a + b + c = 36) 
    (h2 : a * b * c = 125) 
    (h3 : a * b + b * c + c * a = 93.75) : 
    a^2 + b^2 + c^2 = 1108.5 := 
by 
  sorry

end mean_properties_l799_799995


namespace smallest_prime_with_digit_sum_23_l799_799649

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799649


namespace nature_of_graph_l799_799775

theorem nature_of_graph (a b : ℝ) (g : ℝ → ℝ) (h : g = λ x, a*x^2 + b*x + (a + b^2)) :
  (a > 0 → ∃ x₀, ∀ x, g x₀ ≤ g x) ∧ (a < 0 → ∃ x₀, ∀ x, g x₀ ≥ g x) :=
by
  sorry

end nature_of_graph_l799_799775


namespace smallest_prime_digit_sum_23_l799_799613

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799613


namespace total_drink_volume_l799_799736

-- Define the percentages of the various juices
def grapefruit_percentage : ℝ := 0.20
def lemon_percentage : ℝ := 0.25
def pineapple_percentage : ℝ := 0.10
def mango_percentage : ℝ := 0.15

-- Define the volume of orange juice in ounces
def orange_juice_volume : ℝ := 24

-- State the total percentage of all juices other than orange juice
def non_orange_percentage : ℝ := grapefruit_percentage + lemon_percentage + pineapple_percentage + mango_percentage

-- Calculate the percentage of orange juice
def orange_percentage : ℝ := 1 - non_orange_percentage

-- State that the total volume of the drink is such that 30% of it is 24 ounces
theorem total_drink_volume : ∃ (total_volume : ℝ), (orange_percentage * total_volume = orange_juice_volume) ∧ (total_volume = 80) := by
  use 80
  sorry

end total_drink_volume_l799_799736


namespace find_a_b_and_range_of_c_l799_799343

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 + b * x + c

theorem find_a_b_and_range_of_c (c : ℝ) (h1 : ∀ x, 3 * x^2 - 2 * 3 * x - 9 = 0 → x = -1 ∨ x = 3)
    (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 6 → f x 3 (-9) c < c^2 + 4 * c) : 
    (a = 3 ∧ b = -9) ∧ (c > 6 ∨ c < -9) := by
  sorry

end find_a_b_and_range_of_c_l799_799343


namespace asymptote_equation_l799_799348

variables (m x y : ℝ)

def hyperbola := m * x^2 - y^2 = 1

def focal_distance_three_times_conjugate_axis (b c : ℝ) : Prop :=
  2 * c = 6 * b

noncomputable def asymptote_slope := (1 : ℝ) / (2 * real.sqrt 2)

theorem asymptote_equation
  (h : hyperbola m x y)
  (b c a : ℝ)
  (hb : b = 1)
  (hc : c = 3)
  (hrel : 1/m + 1 = c^2)
  (hasym : 2 * real.sqrt 2 = a)
  (hfocal : focal_distance_three_times_conjugate_axis b c) :
  (∀ x : ℝ, y = asymptote_slope * x) ∨ (∀ x : ℝ, y = - asymptote_slope * x) :=
sorry

end asymptote_equation_l799_799348


namespace interval_division_l799_799798

theorem interval_division (n i : ℕ) (h1 : 0 < n) (h2 : 1 ≤ i) (h3 : i ≤ n) :
  (∃ a b : ℝ, a = 2 * (i - 1) / n ∧ b = 2 * i / n ∧ ∀ x : ℝ, x ∈ set.Icc a b ↔ x ∈ set.Icc (2 * (i - 1) / n) (2 * i / n)) :=
begin
  have a : ℝ := 2 * (i - 1) / n,
  have b : ℝ := 2 * i / n,
  use [a, b],
  split,
  { refl, },
  split,
  { refl, },
  intro x,
  split;
  intro h;
  assumption,
end

end interval_division_l799_799798


namespace tangent_line_to_x_cube_at_P_l799_799511

noncomputable def tangent_line_equation (x : ℝ) : ℝ := x^3

def is_tangent_line (f : ℝ → ℝ) (x₀ y₀ : ℝ) (m : ℝ) :=
  ∀ x, f x = m * (x - x₀) + y₀

theorem tangent_line_to_x_cube_at_P {m : ℝ} {b : ℝ} :
  let f := tangent_line_equation
  in (m = 12) ∧ (b = -16) → is_tangent_line f 2 8 12 :=
by
  intros f m b h
  sorry

end tangent_line_to_x_cube_at_P_l799_799511


namespace index_card_area_l799_799433

theorem index_card_area :
  let width := 5
  let height := 7
  let reduced_height := height - 2
  area_reduced_height = width * reduced_height
  area_reduced_height = 25 :=
by
  sorry

end index_card_area_l799_799433


namespace smallest_prime_with_digit_sum_23_l799_799641

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799641


namespace digits_satisfy_conditions_l799_799548

theorem digits_satisfy_conditions (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) : 
  (let minuend := 100 * a + 10 * b + c in
   let subtrahend := 100 * c + 10 * b + a in
   let result := minuend - subtrahend in
   let units_digit := result % 10 in
   let tens_digit := (result / 10) % 10 in
   units_digit = 2 ∧ tens_digit = 9 ∧ c + 10 > a) ↔ a = 9 ∧ b = 9 ∧ c = 1 := 
by sorry

end digits_satisfy_conditions_l799_799548


namespace box_cubes_percentage_l799_799149

-- Define the dimensions of the rectangular box
def box_length: ℝ := 8
def box_width: ℝ := 6
def box_height: ℝ := 12

-- Define the side length of the solid cubes
def cube_side: ℝ := 4

-- Define the expected percentage
def expected_percentage: ℝ := 66.67

-- Define the function to calculate the percentage of the volume of the box taken up by the cubes
noncomputable def volume_percentage_of_cubes (L W H side_length: ℝ) : ℝ :=
  let num_cubes_length := (L / side_length).toNat
  let num_cubes_width := (W / side_length).toNat
  let num_cubes_height := (H / side_length).toNat
  let total_num_cubes := num_cubes_length * num_cubes_width * num_cubes_height
  let volume_of_one_cube := side_length ^ 3
  let total_volume_of_cubes := total_num_cubes * volume_of_one_cube
  let volume_of_box := L * W * H
  (total_volume_of_cubes / volume_of_box) * 100

-- The theorem to be proven
theorem box_cubes_percentage:
  volume_percentage_of_cubes box_length box_width box_height cube_side = expected_percentage :=
sorry

end box_cubes_percentage_l799_799149


namespace erica_total_earnings_l799_799250

def fishPrice : Nat := 20
def pastCatch : Nat := 80
def todayCatch : Nat := 2 * pastCatch
def pastEarnings := pastCatch * fishPrice
def todayEarnings := todayCatch * fishPrice
def totalEarnings := pastEarnings + todayEarnings

theorem erica_total_earnings : totalEarnings = 4800 := by
  sorry

end erica_total_earnings_l799_799250


namespace prob_A_or_B_l799_799046

variable (P : Set ℕ → ℝ)
variable (A B : Set ℕ)

axiom prob_A : P(A) = 0.4
axiom prob_A_and_B : P(A ∩ B) = 0.25
axiom prob_B : P(B) = 0.65

theorem prob_A_or_B : P(A ∪ B) = 0.8 :=
by
  sorry

end prob_A_or_B_l799_799046


namespace skill_position_players_waiting_l799_799124

def linemen_drink : ℕ := 8
def skill_position_player_drink : ℕ := 6
def num_linemen : ℕ := 12
def num_skill_position_players : ℕ := 10
def cooler_capacity : ℕ := 126

theorem skill_position_players_waiting :
  num_skill_position_players - (cooler_capacity - num_linemen * linemen_drink) / skill_position_player_drink = 5 :=
by
  -- Calculation is needed to be filled in here
  sorry

end skill_position_players_waiting_l799_799124


namespace grape_lollipops_count_l799_799763

theorem grape_lollipops_count : 
  let total_lollipops := 42 in
  let cherry_lollipops := total_lollipops / 2 in
  let other_flavors_lollipops := total_lollipops - cherry_lollipops in
  let grape_lollipops := other_flavors_lollipops / 3 in
  grape_lollipops = 7 :=
by
  sorry

end grape_lollipops_count_l799_799763


namespace binom_10_0_eq_1_l799_799178

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem stating the binomial coefficient we need to prove
theorem binom_10_0_eq_1 : binom 10 0 = 1 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end binom_10_0_eq_1_l799_799178


namespace find_m_n_sum_l799_799061

theorem find_m_n_sum :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (m^2 - n = 32) ∧ 
    (∃ x : ℝ, (x ^ 5 - 10 * x ^ 3 + 20 * x - 40 = 0) ∧ 
      (x = (real.root 5 (m + real.sqrt n) + real.root 5 (m - real.sqrt n)))) ∧
    (m + n = 388) := sorry

end find_m_n_sum_l799_799061


namespace f_odd_or_even_f_increasing_range_l799_799341

-- Define the function f(x) and its derivative
def f (a : ℝ) (x : ℝ) := a * x ^ 2 + (1 / x)
def f' (a : ℝ) (x : ℝ) := (2 * a * x ^ 3 - 1) / x ^ 2

-- Prove that f(x) is odd when a = 0 and neither odd nor even when a ≠ 0
theorem f_odd_or_even (a : ℝ) : 
  (a = 0 → ∀ x : ℝ, f a x = f a (-x)) ∧
  (a ≠ 0 → (∃ x : ℝ, f a x ≠ f a (-x)) ∧ ∃ x : ℝ, f a x + f a (-x) ≠ 0) :=
by sorry

-- Prove the range of a for which f(x) is increasing on (1, +∞)
theorem f_increasing_range (a : ℝ) : 
  (∀ x : ℝ, 1 < x → f' a x ≥ 0) ↔ a ∈ Set.Ici (1 / 2) :=
by sorry

end f_odd_or_even_f_increasing_range_l799_799341


namespace journey_distance_l799_799131

theorem journey_distance:
  ∃ D: ℝ, let t1 := (D / 4) / 20,
              t2 := (D / 4) / 10,
              t3 := (D / 4) / 15,
              t4 := (D / 4) / 30 in
  (t1 + t2 + t3 + t4 = 60) ∧ (D = 960) :=
sorry

end journey_distance_l799_799131


namespace dot_product_solution_1_l799_799885

variable (a b : ℝ × ℝ)
variable (k : ℝ)

def two_a_add_b (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 + b.1, 2 * a.2 + b.2)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_solution_1 :
  let a := (1, -1)
  let b := (-1, 2)
  dot_product (two_a_add_b a b) a = 1 := by
sorry

end dot_product_solution_1_l799_799885


namespace smallest_prime_with_digit_sum_23_l799_799591

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799591


namespace smallest_prime_with_digit_sum_23_l799_799669

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799669


namespace jessica_cut_orchids_l799_799065

-- Define initial conditions
def initial_orchids := 3
def initial_roses := 16

-- Define final conditions after cutting more flowers
def final_orchids := 7
def final_roses := 31

-- Define the condition for the ratio of orchids to roses cut
def orchid_to_rose_ratio := (3, 5)

-- Define the question: how many orchids did Jessica cut?
def orchids_cut : Nat := final_orchids - initial_orchids

-- The proof statement
theorem jessica_cut_orchids (roses_cut : Nat) :
  let O := orchids_cut in
  final_orchids = initial_orchids + O ∧ 
  final_roses = initial_roses + roses_cut ∧
  3 * roses_cut = 5 * O →
  O = 4 :=
by
  sorry

end jessica_cut_orchids_l799_799065


namespace right_triangle_pythagorean_l799_799392

variable {A B C : Type} [metric_space A] (a b c : ℝ)

theorem right_triangle_pythagorean (h_angle_B : ∠ B = 90):
  a^2 + c^2 = b^2 := sorry

end right_triangle_pythagorean_l799_799392


namespace limit_calculation_l799_799174

open Real
open Complex

theorem limit_calculation :
  ∃ L : ℝ, 
  (tendsto (λ x, (2 - 3^(arctan (sqrt x))^2) ^ (2 / sin x)) (nhds 0) (nhds L)) ∧ L = 1 / 9 :=
sorry

end limit_calculation_l799_799174


namespace number_of_truth_tellers_is_twelve_l799_799231
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799231


namespace sectors_ratio_eq_one_sixth_l799_799469

noncomputable def ratio_of_sectors_to_circle_area
  (P : Type) [TopologicalSpace P] [MetricSpace P] [Legged P]
  (X Y R S Q Z : P) 
  (angle_RPS angle_QPS angle_ZPS : ℝ) 
  (hRPS : angle_RPS = 60) 
  (hQPS : angle_QPS = 90) 
  (hZPS : angle_ZPS = 30) : ℝ :=
  let angle_QXZ := real.abs (angle_ZPS - angle_QPS) in
  let angle_RYZ := angle_RPS - angle_ZPS in
  (angle_QXZ + angle_RYZ) / 360

theorem sectors_ratio_eq_one_sixth
  (P : Type) [TopologicalSpace P] [MetricSpace P] [Legged P]
  (X Y R S Q Z : P) 
  (angle_RPS angle_QPS angle_ZPS : ℝ) 
  (hRPS : angle_RPS = 60) 
  (hQPS : angle_QPS = 90) 
  (hZPS : angle_ZPS = 30) : 
  ratio_of_sectors_to_circle_area P X Y R S Q Z angle_RPS angle_QPS angle_ZPS hRPS hQPS hZPS = 1 / 6 := 
sorry

end sectors_ratio_eq_one_sixth_l799_799469


namespace vector_equiv_l799_799471

variables {V : Type} [inner_product_space ℝ V]

-- Definitions of points and vectors
variables (A B C P Q : V)
variables (α β γ : ℝ)
variables (p : V)

-- Conditions
def condition1 : Prop := (∃ k1 k2 k3 : ℝ, p = k1 • A + k2 • B + k3 • C)
def condition2 : Prop := (A + 2 • B + 3 • p = 0)
def condition3 : Prop := (∃ λ : ℝ, Q = A + λ • (B - A))
def condition4 : Prop := (∃ μ : ℝ, p = μ • (Q - C))

-- Given vectors
def given_vector : p = 2 • (Q - C)

-- Theorem to prove
theorem vector_equiv : condition1 A B C P p ∧ condition2 A B C P p ∧ condition3 A B C Q ∧ condition4 C P Q p → (Q - C = 2 • p) :=
sorry

end vector_equiv_l799_799471


namespace symmetric_point_proof_l799_799531

def point := ℝ × ℝ × ℝ

def symmetric_wrt_xOy (p : point) : point :=
  (p.1, p.2, -p.3)

def symmetric_wrt_z_axis (p : point) : point :=
  (-p.1, -p.2, p.3)

theorem symmetric_point_proof :
  let P : point := (1, 1, 1)
  let R₁ : point := symmetric_wrt_xOy P
  let p₂ : point := symmetric_wrt_z_axis R₁
  p₂ = (-1, -1, -1) :=
by
  let P : point := (1, 1, 1)
  let R₁ : point := symmetric_wrt_xOy P
  let p₂ : point := symmetric_wrt_z_axis R₁
  show p₂ = (-1, -1, -1)
  sorry

end symmetric_point_proof_l799_799531


namespace count_successful_sequences_l799_799495

-- Definitions based on problem conditions
def cards := Fin 13
def initial_up (c : cards) : Prop := True  -- Initially, all cards are face up

def neighbor_up (c : cards) (c' : cards) (up : cards → Prop) : Prop :=
  (c'.val + 2) % 13 = c.val ∨ (c'.val - 2) % 13 = c.val

def can_flip (c : cards) (up : cards → Prop) : Prop :=
  up c ∧ ∃ c', neighbor_up c c' up

-- Problem simplification
def is_successful_sequence (flip_seq : List cards) : Prop :=
  flip_seq.length = 12 ∧
  ∀ (c : cards), flip_seq.contains c → can_flip c (λ x, ¬ flip_seq.contains x)

set_option maxRecDepth 10000
set_option maxHeartbeats 500000

theorem count_successful_sequences : 
  (List (Fin 13)).countP is_successful_sequence = 26624 :=
sorry

end count_successful_sequences_l799_799495


namespace intersection_complement_l799_799876

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem intersection_complement : A ∩ (U \ B) = {0, 1} := by
  sorry

end intersection_complement_l799_799876


namespace find_distance_BC_l799_799766

variables {d_AB d_AC d_BC : ℝ}

theorem find_distance_BC
  (h1 : d_AB = d_AC + d_BC - 200)
  (h2 : d_AC = d_AB + d_BC - 300) :
  d_BC = 250 := 
sorry

end find_distance_BC_l799_799766


namespace simplify_expression1_simplify_expression2_l799_799026

theorem simplify_expression1 : 
  (4: ℚ) ^ (-2 : ℤ) + (1 / (6 * (6 : ℚ) ^ (1/3 : ℝ))) + ( ( (3: ℚ) ^ (1/2 : ℚ) + (2: ℚ) ^ (1/2 : ℚ)) / ( (3: ℚ) ^ (1/2 : ℚ) - (2: ℚ) ^ (1/2 : ℚ))) + (1 / (2: ℚ) ) * (1.03 : ℚ) ^ (0 : ℤ) * (- (6: ℚ) ^ (1/2: ℚ)) ^ (3 : ℤ) = 21 :=
by
  sorry

theorem simplify_expression2 :
  (log ( (2: ℚ)) 10)^2 + (log (20 : ℚ)) * (log (5 : ℚ)) + (log (2 : ℚ)) (9 ^ 2 / log ( 3 : ℚ) ^ 2) (log (4 : ℚ)) ^ (3: ℚ) = 5 / 4 :=
by
  sorry

end simplify_expression1_simplify_expression2_l799_799026


namespace perimeter_triangle_ABC_l799_799393

-- Definitions based on problem conditions
variables {A B C X Y O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (m : Midpoint A B)

-- Given conditions
def triangle_ABC (AB : ℝ) (AC BC : ℝ) (C_eq : AC = 0.5 * BC) : Prop :=
  ∠ABC C = 90 ∧ 
  AB = 15 ∧ 
  AC = C_eq

-- Theorem to prove
theorem perimeter_triangle_ABC (AB AC BC : ℝ) (C_eq : AC = 0.5 * BC) 
  (h : triangle_ABC AB AC BC C_eq) : 
  15 + AC + BC = 15 + 9 * Real.sqrt 5 :=
  sorry

end perimeter_triangle_ABC_l799_799393


namespace smallest_prime_digit_sum_23_l799_799617

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799617


namespace tangent_line_at_point_l799_799512

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 4 * x + 2

def point : ℝ × ℝ := (1, -3)

def tangent_line (x y : ℝ) : Prop := 5 * x + y - 2 = 0

theorem tangent_line_at_point : tangent_line 1 (-3) :=
  sorry

end tangent_line_at_point_l799_799512


namespace maximize_triangle_area_l799_799069

section triangle_max_area

variables {P Q A B C : Point}
variables (circle1 circle2 : set Point)
variable (center1 center2 : Point)

-- Assume two circles intersect at points P and Q
axiom intersect_circles : P ∈ circle1 ∧ P ∈ circle2 ∧ Q ∈ circle1 ∧ Q ∈ circle2

-- Assume point A is on the first circle but outside the second circle
axiom A_conditions : A ∈ circle1 ∧ A ∉ circle2

-- The lines AP and AQ intersect the second circle at points B and C respectively
axiom intersections : B ∈ circle2 ∧ C ∈ circle2 ∧ (∃ line1 : set Point, ∃ line2 : set Point, A ∈ line1 ∧ P ∈ line1 ∧ B ∈ line1 ∧ A ∈ line2 ∧ Q ∈ line2 ∧ C ∈ line2)

noncomputable theory

-- The point A must be at the intersection of the first circle and the line joining the centers of the two circles to maximize the area of triangle ABC
theorem maximize_triangle_area : 
  ∃ line_centers : set Point, center1 ∈ line_centers ∧ center2 ∈ line_centers ∧ A ∈ line_centers ∧ A ∈ circle1 ∧ 
  ∀ A_location : Point, (A_location ∈ circle1 ∧ A_location ∈ line_centers) → 
  ∃ max_area_condition : ∀ triangle : (triangle ∋ A B C) → (area triangle) = (area (triangle ∋ A B C)) :=
sorry

end triangle_max_area

end maximize_triangle_area_l799_799069


namespace cube_cross_section_area_ratio_squared_l799_799407

noncomputable def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

noncomputable def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  0.5 * d1 * d2

theorem cube_cross_section_area_ratio_squared {s : ℝ} (h : 0 < s) :
  let A := (0, 0, 0),
      D := (0, s, 0),
      H := (0, s, s),
      E := (0, 0, s),
      G := (s, s, s),
      J := midpoint A D,
      I := midpoint H E,
      AG := distance A G,
      JI := distance J I,
      area_AGJI := area_of_rhombus AG JI,
      area_face := s^2,
      R := area_AGJI / area_face
  in R^2 = 3 / 8 := by
  sorry

end cube_cross_section_area_ratio_squared_l799_799407


namespace smallest_prime_with_digit_sum_23_proof_l799_799678

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799678


namespace range_of_2a_plus_3b_l799_799822

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 :=
  sorry

end range_of_2a_plus_3b_l799_799822


namespace average_length_of_strings_l799_799694

theorem average_length_of_strings :
  let l1 := 2.5
  let l2 := 5.5
  let l3 := 3.5
  let total_length := l1 + l2 + l3
  let average_length := total_length / 3
  average_length = 23 / 6 :=
by
  have h1 : l1 + l2 + l3 = 11.5 := by norm_num
  have h2 : 11.5 / 3 = 23 / 6 := by norm_num
  exact h2

end average_length_of_strings_l799_799694


namespace circumcenter_on_OA_l799_799073

variables {P : Type} [EuclideanGeometry.P2 P]
open EuclideanGeometry

def two_lines_intersect_at_O (O A B C : P) (line1 line2 : P → Prop) : Prop :=
  line1 O ∧ line2 O ∧ ¬line1 A ∧ ¬line2 A ∧
  B ∈ reflection (line1 O) A ∧ C ∈ reflection (line2 O) B ∧ reflection (line1 O) C = A

theorem circumcenter_on_OA {O A B C : P} (line1 line2 : P → Prop)
  (h1 : two_lines_intersect_at_O O A B C line1 line2) :
  lies_on_circumcenter_of_OBC_OA O A B C :=
sorry

end circumcenter_on_OA_l799_799073


namespace smallest_prime_with_digit_sum_23_l799_799585

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799585


namespace complex_z_pow_l799_799708

open Complex

theorem complex_z_pow {z : ℂ} (h : (1 + z) / (1 - z) = (⟨0, 1⟩ : ℂ)) : z ^ 2019 = -⟨0, 1⟩ := by
  sorry

end complex_z_pow_l799_799708


namespace trigonometric_ordering_l799_799821

variable (θ : ℝ)
variable (h₀ : 0 < θ)
variable (h₁ : θ < π / 4) -- π/4 radians is 45 degrees

theorem trigonometric_ordering (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 4) :
  sin θ ^ 2 < cos θ ^ 2 ∧ cos θ ^ 2 < (cos θ / sin θ) ^ 2 :=
by
  sorry

end trigonometric_ordering_l799_799821


namespace bill_difference_l799_799001

theorem bill_difference (mandy_bills : ℕ) (manny_bills : ℕ) 
  (mandy_bill_value : ℕ) (manny_bill_value : ℕ) (target_bill_value : ℕ) 
  (h_mandy : mandy_bills = 3) (h_mandy_val : mandy_bill_value = 20) 
  (h_manny : manny_bills = 2) (h_manny_val : manny_bill_value = 50)
  (h_target : target_bill_value = 10) :
  (manny_bills * manny_bill_value / target_bill_value) - (mandy_bills * mandy_bill_value / target_bill_value) = 4 :=
by
  sorry

end bill_difference_l799_799001


namespace butcher_net_loss_l799_799119

noncomputable def dishonest_butcher (advertised_price actual_price : ℝ) (quantity_sold : ℕ) (fine : ℝ) : ℝ :=
  let dishonest_gain_per_kg := actual_price - advertised_price
  let total_dishonest_gain := dishonest_gain_per_kg * quantity_sold
  fine - total_dishonest_gain

theorem butcher_net_loss 
  (advertised_price : ℝ) 
  (actual_price : ℝ) 
  (quantity_sold : ℕ) 
  (fine : ℝ)
  (h_advertised_price : advertised_price = 3.79)
  (h_actual_price : actual_price = 4.00)
  (h_quantity_sold : quantity_sold = 1800)
  (h_fine : fine = 500) :
  dishonest_butcher advertised_price actual_price quantity_sold fine = 122 := 
by
  simp [dishonest_butcher, h_advertised_price, h_actual_price, h_quantity_sold, h_fine]
  sorry

end butcher_net_loss_l799_799119


namespace intersection_complement_eq_l799_799877

open Set

def U : Set ℕ := {x | 1 ≤ x ∧ x ≤ 8}  -- Which is from the solution set of x^2 - 9x + 8 ≤ 0
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem intersection_complement_eq : (U \ A) ∩ (U \ B) = {4, 8} :=
by
  sorry

end intersection_complement_eq_l799_799877


namespace num_of_valid_arrangements_is_144_l799_799423

axiom A : Type
axiom B : Type
axiom C : Type
axiom D : Type
axiom E : Type
axiom F : Type

noncomputable def num_of_arrangements : ℕ :=
  let poems := [A, B, C, D, E, F]
  let is_valid_arrangement := λ arrangement : List A,
    (B ∈ arrangement) ∧ (D ∈ arrangement) ∧ 
    (arrangement.indexOf B < arrangement.indexOf D) ∧ 
    ¬(A = arrangement.getLast) ∧ ¬(F = arrangement.getLast) ∧ 
    (∀ x, x ≠ arrangement.length - 1 → arrangement.indexOf x ≠ arrangement.indexOf x + 1)
  (sorry : ℕ)

theorem num_of_valid_arrangements_is_144 : num_of_arrangements = 144 := sorry

end num_of_valid_arrangements_is_144_l799_799423


namespace erica_earnings_l799_799249

def price_per_kg : ℝ := 20
def past_catch : ℝ := 80
def catch_today := 2 * past_catch
def total_catch := past_catch + catch_today
def total_earnings := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end erica_earnings_l799_799249


namespace smallest_prime_with_digit_sum_23_l799_799571

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799571


namespace john_profit_l799_799701

theorem john_profit (cp_grinder cp_mobile : ℕ) (loss_pct_grinder profit_pct_mobile : ℕ) 
  (cp_grinder = 15000) (cp_mobile = 8000) (loss_pct_grinder = 2) (profit_pct_mobile = 10) :
  let loss_amount_grinder := (loss_pct_grinder * cp_grinder) / 100,
      sp_grinder := cp_grinder - loss_amount_grinder,
      profit_amount_mobile := (profit_pct_mobile * cp_mobile) / 100,
      sp_mobile := cp_mobile + profit_amount_mobile,
      total_cp := cp_grinder + cp_mobile,
      total_sp := sp_grinder + sp_mobile,
      overall_profit := total_sp - total_cp in
  overall_profit = 500 :=
by
  sorry

end john_profit_l799_799701


namespace carlos_laundry_time_l799_799762

theorem carlos_laundry_time :
  ∀ (wash_time per_load : ℕ) (dry_time : ℕ),
    wash_time = 45 →
    per_load = 2 →
    dry_time = 75 →
    (wash_time * per_load + dry_time) = 165 :=
by
  intros wash_time per_load dry_time h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end carlos_laundry_time_l799_799762


namespace tangent_line_equation_l799_799038

open Real
open TopologicalSpace
open Filter
open Asymptotics

def curve (x : ℝ) : ℝ := x * exp x + 2 * x - 1

def point := (0 : ℝ, -1 : ℝ)

theorem tangent_line_equation : ∃ m b, 
  (∀ x, curve x - (-1) = m * (x - 0)) ∧ m = 3 ∧ b = -1 :=
sorry

end tangent_line_equation_l799_799038


namespace stationery_sales_other_l799_799032

theorem stationery_sales_other (p e n : ℝ) (h_p : p = 25) (h_e : e = 30) (h_n : n = 20) :
    100 - (p + e + n) = 25 :=
by
  sorry

end stationery_sales_other_l799_799032


namespace co_bisects_xy_l799_799927

open EuclideanGeometry

variables {A B C H_a H_b O X Y : Point}

declare_provenance triangle_ABC (ABC acute-scalene)

def circumcenter (ABC : acute_scalene_triangle) : Point := sorry
def altitude (ABC : acute_scalene_triangle) (vertex : Point) : Point := sorry
def symmetric_point (p midpoint : Point) : Point := sorry
def midpoint (p1 p2 : Point) : Point := sorry

axiom h1 : circumcenter ABC = O
axiom h2 : altitude ABC A = H_a
axiom h3 : altitude ABC B = H_b 
axiom h4 : (symmetric_point H_a (midpoint B C)) = X
axiom h5 : (symmetric_point H_b (midpoint C A)) = Y

theorem co_bisects_xy : collinear [C, O, midpoint X Y] := sorry

end co_bisects_xy_l799_799927


namespace digit_positions_in_8008_l799_799522

theorem digit_positions_in_8008 :
  (8008 % 10 = 8) ∧ (8008 / 1000 % 10 = 8) :=
by
  sorry

end digit_positions_in_8008_l799_799522


namespace acute_angle_is_30_degrees_l799_799510

def line_equation (x y : ℝ) : Prop :=
  x - sqrt 3 * y + 2016 = 0

def slope (x y : ℝ) : ℝ :=
  sqrt 3 / 3

noncomputable def angle_with_x_axis (m : ℝ) : ℝ :=
  Real.arctan m

theorem acute_angle_is_30_degrees :
  ∀ x y : ℝ, line_equation x y → (∀ α : ℝ, α = angle_with_x_axis (slope x y) → α = 30) :=
by
  intros x y h α ha
  sorry

end acute_angle_is_30_degrees_l799_799510


namespace angle_b_alpha_equal_l799_799993

noncomputable def angle_formed_by_line_and_plane (line : Type) (plane : Type) : ℝ := sorry

variable (a b : Type) (α : Type)
variable (parallel : a → b → Prop)
variable (angle_a_alpha : angle_formed_by_line_and_plane a α = 50)

-- Given condition:
axiom parallel_ab : parallel a b

-- Our proof statement:
theorem angle_b_alpha_equal (h: parallel a b) : angle_formed_by_line_and_plane b α = 50 :=
sorry

end angle_b_alpha_equal_l799_799993


namespace infinite_chain_resistance_l799_799195

noncomputable def resistance_of_infinite_chain (R₀ : ℝ) : ℝ :=
  (R₀ * (1 + Real.sqrt 5)) / 2

theorem infinite_chain_resistance : resistance_of_infinite_chain 10 = 5 + 5 * Real.sqrt 5 :=
by
  sorry

end infinite_chain_resistance_l799_799195


namespace remainder_approximately_14_l799_799470

def dividend : ℝ := 14698
def quotient : ℝ := 89
def divisor : ℝ := 164.98876404494382
def remainder : ℝ := dividend - (quotient * divisor)

theorem remainder_approximately_14 : abs (remainder - 14) < 1e-10 := 
by
-- using abs since the problem is numerical/approximate
sorry

end remainder_approximately_14_l799_799470


namespace number_of_valid_four_digit_numbers_l799_799371

-- Definition for the problem
def is_valid_four_digit_number (n : ℕ) : Prop :=
  2999 < n ∧ n <= 9999 ∧
  (let d1 := n / 1000,
       d2 := (n / 100) % 10,
       d3 := (n / 10) % 10,
       d4 := n % 10 in
   3 <= d1 ∧ d1 <= 9 ∧
   0 <= d4 ∧ d4 <= 9 ∧
   d2 * d3 > 10)

-- Statement of the problem
theorem number_of_valid_four_digit_numbers : 
  (Finset.range 10000).filter is_valid_four_digit_number).card = 4830 :=
sorry

end number_of_valid_four_digit_numbers_l799_799371


namespace ellipse_equation_correct_range_of_slope_correct_l799_799847

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 2 = 1

noncomputable def range_of_slope (k : ℝ) : Prop :=
  - sqrt 6 / 12 ≤ k ∧ k ≤ sqrt 6 / 12

theorem ellipse_equation_correct :
  ∀ x y : ℝ, ellipse_equation x y ↔ (x^2 / 6 + y^2 / 2 = 1) :=
by sorry

theorem range_of_slope_correct :
  ∀ k : ℝ, range_of_slope k ↔ (- sqrt 6 / 12 ≤ k ∧ k ≤ sqrt 6 / 12) :=
by sorry

end ellipse_equation_correct_range_of_slope_correct_l799_799847


namespace train_crossing_time_l799_799098

-- Define the conditions
def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 250
def speed_of_train_kmph : ℝ := 36

-- Convert speed from kmph to m/s
def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

-- Calculate total distance
def total_distance : ℝ := length_of_train + length_of_bridge

-- Calculate the time to cross the bridge
def time_to_cross (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_crossing_time :
  time_to_cross total_distance speed_of_train_mps = 35 :=
by
  sorry

end train_crossing_time_l799_799098


namespace bogatyrs_truthful_count_l799_799238

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799238


namespace point_in_fourth_quadrant_l799_799815

theorem point_in_fourth_quadrant (a : ℝ) (ha : a > 1) :
  let P := (1 + a, 1 - a) in P.1 > 0 ∧ P.2 < 0 :=
by
  let P := (1 + a, 1 - a)
  show P.1 > 0 ∧ P.2 < 0 from sorry

end point_in_fourth_quadrant_l799_799815


namespace immortal_flea_can_visit_every_natural_l799_799040

theorem immortal_flea_can_visit_every_natural :
  ∀ (k : ℕ), ∃ (jumps : ℕ → ℤ), (∀ n : ℕ, ∃ m : ℕ, jumps m = n) :=
by
  -- proof goes here
  sorry

end immortal_flea_can_visit_every_natural_l799_799040


namespace A_share_correct_l799_799737

noncomputable def investment_shares (x : ℝ) (annual_gain : ℝ) := 
  let A_share := x * 12
  let B_share := (2 * x) * 6
  let C_share := (3 * x) * 4
  let total_share := A_share + B_share + C_share
  let total_ratio := 1 + 1 + 1
  annual_gain / total_ratio

theorem A_share_correct (x : ℝ) (annual_gain : ℝ) (h_gain : annual_gain = 18000) : 
  investment_shares x annual_gain / 3 = 6000 := by
  sorry

end A_share_correct_l799_799737


namespace sum_first_11_terms_eq_neg66_l799_799999

def a_n (n : ℕ) : ℤ := 1 - 2 * n

def S_n (n : ℕ) : ℤ := (a_n 1 + a_n n) * n / 2

def sequence (n : ℕ) : ℤ := S_n n / n

noncomputable def sum_first_m_terms_of_sequence (m : ℕ) : ℤ :=
  (List.range m).map (λ i, sequence (i + 1)).sum

theorem sum_first_11_terms_eq_neg66 :
  sum_first_m_terms_of_sequence 11 = -66 :=
  sorry

end sum_first_11_terms_eq_neg66_l799_799999


namespace problem_solution_l799_799105

open Real

noncomputable def length_and_slope_MP 
    (length_MN : ℝ) 
    (slope_MN : ℝ) 
    (length_NP : ℝ) 
    (slope_NP : ℝ) 
    : (ℝ × ℝ) := sorry

theorem problem_solution :
  length_and_slope_MP 6 14 7 8 = (5.55, 25.9) :=
  sorry

end problem_solution_l799_799105


namespace third_circle_tangent_l799_799947

-- Defining the basic structures and conditions of the problem
variables {A B C : Type*}
variables (triangle_ABC : A × B × C)
variables (m_a m_b m_c : Type*)
variables (circle_Ma circle_Mb circle_Mc incircle : Type*)

-- Assuming the circles with medians as diameters are tangent to the incircle
axiom median_a (t_a : tangent circle_Ma incircle)
axiom median_b (t_b : tangent circle_Mb incircle)

-- The goal is to prove that the third circle is also tangent to the incircle
theorem third_circle_tangent :
  tangent circle_Mc incircle :=
sorry

end third_circle_tangent_l799_799947


namespace cannot_finish_third_l799_799027

variable (P Q R S T U : ℕ)
variable (beats : ℕ → ℕ → Prop)
variable (finishes_after : ℕ → ℕ → Prop)
variable (finishes_before : ℕ → ℕ → Prop)

noncomputable def race_conditions (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) : Prop :=
  beats P Q ∧
  beats P R ∧
  beats Q S ∧
  finishes_after T P ∧
  finishes_before T Q ∧
  finishes_after U R ∧
  beats U T

theorem cannot_finish_third (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) :
  race_conditions P Q R S T U beats finishes_after finishes_before →
  ¬ (finishes_before P T ∧ finishes_before T S ∧ finishes_after P R ∧ finishes_after P S) ∧ ¬ (finishes_before S T ∧ finishes_before T P) :=
sorry

end cannot_finish_third_l799_799027


namespace hyperbola_properties_l799_799515

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := (x^2 / 2) - y^2 = 1

-- Define the focal length of the hyperbola
def focal_length : ℝ := 2 * Real.sqrt 3

-- Define the equation of the asymptotes
def asymptotes (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

-- Statement to prove the focal length and equation of asymptotes of the hyperbola
theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → (focal_length = 2 * Real.sqrt 3 ∧ 
  ∀ x : ℝ, asymptotes x y)) :=
by
  sorry

end hyperbola_properties_l799_799515


namespace shift_graph_sin_l799_799545

theorem shift_graph_sin :
  ∀ (x : ℝ),
  (sin (2 * (x + (π / 4)) + (π / 6))) = (sin (2 * x + (2 * π / 3))) := by
  sorry

end shift_graph_sin_l799_799545


namespace odd_function_sufficient_not_necessary_l799_799109

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (x + ϕ)

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem odd_function_sufficient_not_necessary {ϕ : ℝ} (h : ϕ = Real.pi) :
  is_odd (f x ϕ) ∧ ¬ (∀ k : ℤ, ϕ = k * Real.pi → is_odd (f x ϕ)) :=
by
  sorry

end odd_function_sufficient_not_necessary_l799_799109


namespace probability_order_black_red_white_l799_799395

noncomputable def probability_black_red_white (total_balls : ℕ) (black : ℕ) (red : ℕ) (white : ℕ) : ℚ :=
  (black / total_balls) * ((red) / (total_balls - 1)) * ((white) / (total_balls - 2))

theorem probability_order_black_red_white :
  let total_balls := 15 in
  let black := 6 in
  let red := 5 in
  let white := 4 in
  probability_black_red_white total_balls black red white = 4 / 91 :=
by 
  sorry

end probability_order_black_red_white_l799_799395


namespace max_l_a_l799_799960

noncomputable def f (a x : ℝ) := a * x^2 + 8 * x + 3

def l (a : ℝ) : ℝ := if a < -8 then ... else
                    if a = -8 then (Real.sqrt 5 + 1) / 2 else ...

theorem max_l_a : ∀ a : ℝ, a < 0 → 
  ∃ l_a > 0, ∀ x ∈ Set.Icc 0 l_a, |f a x| ≤ 5 → 
  (a = -8 ∧ l (-8) = (Real.sqrt 5 + 1) / 2) :=
begin
  sorry
end

end max_l_a_l799_799960


namespace solution_set_of_inequality_l799_799276

theorem solution_set_of_inequality (x : ℝ) : 
  (-x^2 + 2*x + 3 ≥ 0) ↔ (x ∈ set.Icc (-1 : ℝ) 3) :=
sorry

end solution_set_of_inequality_l799_799276


namespace alfonso_weeks_to_meet_goals_l799_799740

def weekly_earning_weekdays := 6 * 5
def weekly_earning_weekends := 8 * 2
def total_weekly_earnings := weekly_earning_weekdays + weekly_earning_weekends

def cost_helmet := 340
def cost_gloves := 45
def cost_eyewear := 75
def total_cost := cost_helmet + cost_gloves + cost_eyewear

def current_savings := 40
def miscellaneous_expenses := 20

def additional_needed := total_cost - current_savings + miscellaneous_expenses

def required_weeks := (additional_needed / total_weekly_earnings).ceil

theorem alfonso_weeks_to_meet_goals : required_weeks = 10 := 
by
  -- You can complete the detailed proof steps here
  sorry

end alfonso_weeks_to_meet_goals_l799_799740


namespace rotated_particle_speed_l799_799724

-- Define the initial position function of the particle
def position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 10, 6 * t - 16)

-- Compute the speed before rotation
def speed (t1 t2 : ℝ) : ℝ :=
  let pos1 := position t1
  let pos2 := position t2
  real.sqrt ((pos2.1 - pos1.1)^2 + (pos2.2 - pos1.2)^2)

-- Define the rotation matrix for 45-degree rotation
def rotate_45 (x y : ℝ) : ℝ × ℝ :=
  let sqrt2_inv := real.sqrt 2⁻¹
  (sqrt2_inv * (x - y), sqrt2_inv * (x + y))

-- Compute the rotated speed vector
def rotated_speed_vector : ℝ × ℝ :=
  rotate_45 3 6

-- Compute the speed of the rotated particle
def rotated_speed : ℝ :=
  real.sqrt (rotated_speed_vector.1^2 + rotated_speed_vector.2^2)

-- The theorem to prove
theorem rotated_particle_speed : rotated_speed = 3 * real.sqrt 10 := 
  sorry

end rotated_particle_speed_l799_799724


namespace per_capita_income_ratio_l799_799496

theorem per_capita_income_ratio
  (PL_10 PZ_10 PL_now PZ_now : ℝ)
  (h1 : PZ_10 = 0.4 * PL_10)
  (h2 : PZ_now = 0.8 * PL_now)
  (h3 : PL_now = 3 * PL_10) :
  PZ_now / PZ_10 = 6 := by
  -- Proof to be filled
  sorry

end per_capita_income_ratio_l799_799496


namespace electric_field_intensity_at_arc_center_l799_799744

theorem electric_field_intensity_at_arc_center
  (R : ℝ) (α : ℝ) (q : ℝ) (k : ℝ) :
  R = 50 → α = 30 * (Real.pi / 180) → q = 2 * 10^-6 → k = 9 * 10^9 → 
  E = 71 * 10^3 :=
by
  intros hR hα hq hk
  -- proof would go here
  sorry

end electric_field_intensity_at_arc_center_l799_799744


namespace max_sum_hex_digits_l799_799902

theorem max_sum_hex_digits 
  (a b c : ℕ) (y : ℕ) 
  (h_a : 0 ≤ a ∧ a < 16)
  (h_b : 0 ≤ b ∧ b < 16)
  (h_c : 0 ≤ c ∧ c < 16)
  (h_y : 0 < y ∧ y ≤ 16)
  (h_fraction : (a * 256 + b * 16 + c) * y = 4096) : 
  a + b + c ≤ 1 :=
sorry

end max_sum_hex_digits_l799_799902


namespace Harvard_attendance_l799_799413

theorem Harvard_attendance:
  (total_applicants : ℕ) (acceptance_rate : ℝ) (attendance_rate : ℝ) 
  (h1 : total_applicants = 20000) 
  (h2 : acceptance_rate = 0.05) 
  (h3 : attendance_rate = 0.9) :
  ∃ (number_attending : ℕ), number_attending = 900 := 
by 
  sorry

end Harvard_attendance_l799_799413


namespace find_x_l799_799301

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : Nat.coprime 2 3)
  (h3 : (∀ p : ℕ, Prime p → p ∣ x → p = 2 ∨ p = 3))
  (h4 : Nat.count (λ p, Prime p ∧ p ∣ x) = 3)
  : x = 480 ∨ x = 2016 :=
by sorry

end find_x_l799_799301


namespace smallest_prime_with_digit_sum_23_l799_799599

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799599


namespace solve_lg_eq_l799_799529

theorem solve_lg_eq (x : ℝ) (h : log 10 (3 * x + 4) = 1) : x = 2 := sorry

end solve_lg_eq_l799_799529


namespace smallest_prime_with_digit_sum_23_l799_799659

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799659


namespace purely_imaginary_complex_number_l799_799910

theorem purely_imaginary_complex_number (m : ℝ) :
  let z := complex.mk (m^2 - 1) (m + 1)
  in z.re = 0 ∧ z.im ≠ 0 → m = 1 :=
by
  sorry

end purely_imaginary_complex_number_l799_799910


namespace cricket_target_runs_l799_799935

-- Defining the problem statement with given conditions
theorem cricket_target_runs (run_rate_initial : ℝ) (overs_initial : ℕ) (run_rate_remaining : ℝ) (overs_remaining : ℕ)
  (h_initial : run_rate_initial = 3.2) (h_overs_initial : overs_initial = 10)
  (h_remaining : run_rate_remaining = 6.25) (h_overs_remaining : overs_remaining = 40) :
  let T := run_rate_initial * overs_initial + run_rate_remaining * overs_remaining in
  T = 282 :=
by
  sorry

end cricket_target_runs_l799_799935


namespace percent_volume_filled_by_cubes_l799_799140

theorem percent_volume_filled_by_cubes : 
  let box_volume := 8 * 6 * 12,
      cube_volume := 4 * 4 * 4,
      max_cubes := (8 / 4) * (6 / 4) * (12 / 4),
      total_cube_volume := max_cubes * cube_volume in
  (total_cube_volume / box_volume) * 100 = 66.67 := 
sorry

end percent_volume_filled_by_cubes_l799_799140


namespace range_of_a_l799_799359

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l799_799359


namespace distance_between_intersections_l799_799782

noncomputable def distance_between_curves := 
  (x y : ℝ) → x = y^3 ∧ x + y^3 = 2 → ℝ

theorem distance_between_intersections (a b c : ℕ) : 
  distance_between_curves = 2 * real.sqrt 2 ∧ (a, b, c) = (4, 2, 2) := 
begin
  sorry
end

end distance_between_intersections_l799_799782


namespace series_diverges_l799_799253

noncomputable def f (x : ℝ) : ℝ := 1 / (x * real.log x)

theorem series_diverges :
  (∀ x : ℝ, x ≥ 2 → f x > 0) ∧
  (∀ x : ℝ, x > 1 → continuous_at (f x)) ∧
  (∀ x y : ℝ, x > e → y > e → x < y → f x > f y) →
  ¬ (converges (λ n, 1 / (n * real.log n))) :=
begin
  sorry,
end

end series_diverges_l799_799253


namespace rational_points_exist_l799_799964

-- Definitions
def is_rational_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90 ∧
  is_rational A ∧ is_rational B ∧ is_rational C

def is_rational_point (A B C : ℕ) (P : ℕ) : Prop :=
  let angles_formed := [A, B, C] in
  ∀ X, X ∈ angles_formed → is_rational X

-- Theorem Statement
theorem rational_points_exist (A B C : ℕ) : 
  is_rational_triangle A B C → 
  ∃ P₁ P₂ P₃, P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₃ ≠ P₁ ∧ 
               is_rational_point A B C P₁ ∧ 
               is_rational_point A B C P₂ ∧ 
               is_rational_point A B C P₃ :=
by
  sorry

end rational_points_exist_l799_799964


namespace meeting_time_when_speeds_doubled_l799_799546

noncomputable def meeting_time (x y z : ℝ) : ℝ :=
  2 * 91

theorem meeting_time_when_speeds_doubled
  (x y z : ℝ)
  (h1 : 2 * z * (x + y) = (2 * z - 56) * (2 * x + y))
  (h2 : 2 * z * (x + y) = (2 * z - 65) * (x + 2 * y))
  : meeting_time x y z = 182 := 
sorry

end meeting_time_when_speeds_doubled_l799_799546


namespace pure_imaginary_condition_l799_799331

variable {a : ℝ}

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem pure_imaginary_condition :
  (∃ a : ℝ, z = (a^2 - 1) + (a - 2) * Complex.I) → (a = 1) →
  is_pure_imaginary z ∧ ¬is_pure_imaginary (a = 1) :=
sorry

end pure_imaginary_condition_l799_799331


namespace solution_set_inequality_l799_799905

variable {X : Type} [LinearOrder X] [TopologicalSpace X] [OrderTopology X]
variable (f : X → ℝ)

-- Conditions
def is_even (f : X → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_pos : Prop := ∀ x y, 0 < x → x < y → f y < f x

axiom f_even : is_even f
axiom f_decreasing_on_pos : is_decreasing_on_pos f
axiom f_at_neg3 : f (-3) = 1

theorem solution_set_inequality :
  {x | f x < 1} = { x | x < -3 ∨ x > 3 } :=
by
  sorry

end solution_set_inequality_l799_799905


namespace max_areas_in_disk_l799_799118

noncomputable def max_non_overlapping_areas (n : ℕ) : ℕ := 5 * n + 1

theorem max_areas_in_disk (n : ℕ) : 
  let disk_divided_by_2n_radii_and_two_secant_lines_areas  := (5 * n + 1)
  disk_divided_by_2n_radii_and_two_secant_lines_areas = max_non_overlapping_areas n := by sorry

end max_areas_in_disk_l799_799118


namespace dice_product_probability_is_one_l799_799807

def dice_probability_product_is_one : Prop :=
  ∀ (a b c d e : ℕ), (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 → 
    (a * b * c * d * e) = 1) ∧
  ∃ (p : ℚ), p = (1/6)^5 ∧ p = 1/7776

theorem dice_product_probability_is_one (a b c d e : ℕ) :
  dice_probability_product_is_one :=
by
  sorry

end dice_product_probability_is_one_l799_799807


namespace geometric_series_sum_l799_799754

theorem geometric_series_sum :
  ∑' n : ℕ, (2 : ℝ) * (1 / 4) ^ n = 8 / 3 := by
  sorry

end geometric_series_sum_l799_799754


namespace smallest_prime_digit_sum_23_l799_799618

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799618


namespace smallest_prime_with_digit_sum_23_l799_799576

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799576


namespace duck_goose_difference_l799_799750

theorem duck_goose_difference :
  let d0 := 25 in
  let g0 := 2 * d0 - 10 in
  let darrived := 4 in
  let gleaving := 15 - 5 in
  let ducks_remaining := d0 + darrived in
  let geese_remaining := g0 - gleaving in
  geese_remaining - ducks_remaining = 1 :=
by
  sorry

end duck_goose_difference_l799_799750


namespace smallest_prime_with_digit_sum_23_l799_799629

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799629


namespace smallest_prime_with_digit_sum_23_l799_799573

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799573


namespace find_f_7_l799_799444

noncomputable def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

theorem find_f_7 : f 7 = 17 :=
  by
  -- The proof steps will go here
  sorry

end find_f_7_l799_799444


namespace total_population_of_towns_l799_799405

theorem total_population_of_towns :
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  num_towns * estimated_avg_pop = 95000 :=
by
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  show num_towns * estimated_avg_pop = 95000
  sorry

end total_population_of_towns_l799_799405


namespace AC_is_diameter_l799_799982

variable {P Q A B C D : Type} [inner_product_space ℝ P]

-- Triangles ADP and QAB are similar
variable {ADP QAB : Triangle P}
variable (similarity_ADP_QAB : similar ADP QAB)

-- Quadrilateral ABCD is inscribed in a circle with radius 7
variable (circumradius_ABC_is : ∃ (O : P) (r : ℝ), is_circumradius (Quad.to_Triangle_ABC O) r ∧ r = 7)

-- Goal: Proving AC = 14
theorem AC_is_diameter {AC : ℝ} (circumradius_is_7 : circumcircle ABCD = Legend.circle radius_7.to_int_real) :
  AC = 14 :=
sorry

end AC_is_diameter_l799_799982


namespace right_triangle_area_1080_l799_799924

def right_triangle_area (leg1 leg2 : ℝ) : ℝ :=
  (1 / 2) * leg1 * leg2

theorem right_triangle_area_1080 :
  right_triangle_area 45 48 = 1080 :=
by
  sorry

end right_triangle_area_1080_l799_799924


namespace num_sampled_from_major_C_is_16_l799_799709

-- Define the constants for the number of students in each major and total sampled students
def num_students_A : ℕ := 150
def num_students_B : ℕ := 150
def num_students_C : ℕ := 400
def num_students_D : ℕ := 300
def total_sampled_students : ℕ := 40

-- Define the total number of students
def total_students : ℕ := num_students_A + num_students_B + num_students_C + num_students_D

-- Define the proportion of students in major C
def proportion_C : ℚ := (num_students_C : ℚ) / (total_students : ℚ)

-- Define the number of students to be sampled from major C
def sampled_students_C : ℕ := (total_sampled_students : ℚ * proportion_C).to_nat

-- State the theorem
theorem num_sampled_from_major_C_is_16 : sampled_students_C = 16 :=
by
  sorry

end num_sampled_from_major_C_is_16_l799_799709


namespace area_union_six_triangles_l799_799987

theorem area_union_six_triangles (s : ℕ) (area : ℕ → ℝ) (total_area : ℕ → ℝ) 
  (triangle_area_half : ℝ ) (sum_of_overlap_areas : ℝ) 
  (net_area_union : ℝ) (h1 : s = 4)
  (h2 : area s = (real.sqrt 3 / 4) * s^2)
  (h3 : total_area 6 = 6 * area s)
  (h4 : triangle_area_half =  real.sqrt 3)
  (h5 : sum_of_overlap_areas = 5 * triangle_area_half)
  (h6 : net_area_union = total_area 6 - sum_of_overlap_areas)
  : net_area_union = 19 * (real.sqrt 3) :=
by 
  sorry

end area_union_six_triangles_l799_799987


namespace robert_ate_7_chocolates_l799_799983

-- Define the number of chocolates Nickel ate
def nickel_chocolates : ℕ := 5

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := nickel_chocolates + 2

-- Prove that Robert ate 7 chocolates
theorem robert_ate_7_chocolates : robert_chocolates = 7 := by
    sorry

end robert_ate_7_chocolates_l799_799983


namespace description_of_M_l799_799875

variable {R : Type} [Real R]

def M : Set R := {y | ∃ x : R, y = x^2}

theorem description_of_M :
  M = {y | ∃ x : R, y = x^2} :=
by
  sorry

end description_of_M_l799_799875


namespace mason_father_age_l799_799005

theorem mason_father_age
  (Mason_age : ℕ) 
  (Sydney_age : ℕ) 
  (Father_age : ℕ)
  (h1 : Mason_age = 20)
  (h2 : Sydney_age = 3 * Mason_age)
  (h3 : Father_age = Sydney_age + 6) :
  Father_age = 66 :=
by
  sorry

end mason_father_age_l799_799005


namespace exist_three_rational_points_in_acute_rational_triangle_l799_799965

def is_rational_angle (α : ℝ) : Prop := ∃ (q : ℚ), α = (q : ℝ)

def is_rational_triangle (A B C : ℝ) : Prop :=
  is_rational_angle A ∧ is_rational_angle B ∧ is_rational_angle C

def is_acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90 ∧ A + B + C = 180

def is_rational_point (P : ℝ × ℝ) (triangles : List (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (triangle : ℝ × ℝ × ℝ) ∈ triangles, is_rational_triangle triangle.fst triangle.snd triangle.snd

theorem exist_three_rational_points_in_acute_rational_triangle
  (A B C : ℝ) (h_rat_triangle : is_rational_triangle A B C) (h_acute : is_acute_triangle A B C) :
  ∃ P Q R : ℝ × ℝ, is_rational_point P [] ∧ is_rational_point Q [] ∧ is_rational_point R [] ∧ (P ≠ Q ∧ P ≠ R ∧ Q ≠ R) :=
sorry

end exist_three_rational_points_in_acute_rational_triangle_l799_799965


namespace no_two_digit_number_no_three_digit_number_l799_799986

theorem no_two_digit_number:
  ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 → 10 * b + a ≠ 2 * (10 * a + b) := by
  intros a b ha hb
  have h1 : 10 * b + a = 2 * (10 * a + b) → a * 19 = 8 * b := by
    intro h
    have h2 : 10 * b + a = 20 * a + 2 * b := by rw [h]; ring
    linarith
  push_neg
  intro h
  obtain ⟨k, hk⟩ : a * 19 = 8 * b := h1 h
  sorry

theorem no_three_digit_number:
  ∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 → 0 ≤ c ∧ c ≤ 9 → 100 * c + 10 * b + a ≠ 2 * (100 * a + 10 * b + c) := by
  intros a b c ha hb hc
  have h1 : 100 * c + 10 * b + a = 2 * (100 * a + 10 * b + c) → 98 * c = 199 * a - 10 * b := by
    intro h
    have h2 : 100 * c + 10 * b + a = 200 * a + 20 * b + 2 * c := by rw [h]; ring
    linarith
  push_neg
  intro h
  obtain ⟨k, hk⟩ : 98 * c = 199 * a - 10 * b := h1 h
  sorry

end no_two_digit_number_no_three_digit_number_l799_799986


namespace four_digit_numbers_count_l799_799372

def valid_middle_digit_pairs : ℕ :=
  (list.product [2, 3, 4, 5, 6, 7, 8, 9] [2, 3, 4, 5, 6, 7, 8, 9]).count (λ p, p.1 * p.2 > 10)

theorem four_digit_numbers_count :
  let first_digit_choices := 7
  let valid_pairs := valid_middle_digit_pairs
  let last_digit_choices := 10
  ∑ x in range first_digit_choices, 
  ∑ y in range valid_pairs, 
  ∑ z in range last_digit_choices, 1 = 3990 := 
sorry

end four_digit_numbers_count_l799_799372


namespace solve_system_l799_799029

theorem solve_system (a b c : ℝ) (h₁ : a^2 + 3 * a + 1 = (b + c) / 2)
                                (h₂ : b^2 + 3 * b + 1 = (a + c) / 2)
                                (h₃ : c^2 + 3 * c + 1 = (a + b) / 2) : 
  a = -1 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end solve_system_l799_799029


namespace number_of_truthful_warriors_l799_799210

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799210


namespace num_ways_to_assign_roles_l799_799726

-- Definitions for men, women, and roles
def num_men : ℕ := 6
def num_women : ℕ := 7
def male_roles : ℕ := 3
def female_roles : ℕ := 3
def gender_neutral_roles : ℕ := 2

-- Permutations function
noncomputable def permutations : ℕ → ℕ → ℕ
| n, k := Nat.factorial n / Nat.factorial (n - k)

-- Proof statement
theorem num_ways_to_assign_roles :
  let men_perms := permutations num_men male_roles in
  let women_perms := permutations num_women female_roles in
  let remaining_individuals := num_men + num_women - male_roles - female_roles in
  let neutral_roles_perms := permutations remaining_individuals gender_neutral_roles in
  men_perms * women_perms * neutral_roles_perms = 1058400 :=
by
  sorry

end num_ways_to_assign_roles_l799_799726


namespace smallest_prime_with_digit_sum_23_l799_799589

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799589


namespace range_of_m_l799_799361

def set_A : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) : Set ℝ := { x : ℝ | (2 * m - 1) ≤ x ∧ x ≤ (2 * m + 1) }

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ (-1 / 2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l799_799361


namespace perimeter_relationship_area_relationship_find_y_when_S_eq_80_l799_799849

section RectangleProperties

variable (x y S : ℕ)

-- The length of one side of the rectangle is 8
def length := 8
def width := x

-- Definition of the perimeter
def perimeter := 2 * (length + width)

-- Definition of the area
def area := length * width

-- Theorem (1): Relationship between y and x
theorem perimeter_relationship (h : y = perimeter) : y = 16 + 2 * x := sorry

-- Theorem (2): Relationship between S and x
theorem area_relationship (h : S = area) : S = 8 * x := sorry

-- Theorem (3): When S = 80, find y
theorem find_y_when_S_eq_80 (hS : S = 80) (hS_def : S = 8 * x) (hy_def : y = 16 + 2 * x) : y = 36 :=
  (by
    have hx : x = 10 := by linarith
    rw [hx, hy_def]
    linarith
  )

end RectangleProperties

end perimeter_relationship_area_relationship_find_y_when_S_eq_80_l799_799849


namespace four_digit_numbers_count_l799_799374

def valid_middle_digit_pairs : ℕ :=
  (list.product [2, 3, 4, 5, 6, 7, 8, 9] [2, 3, 4, 5, 6, 7, 8, 9]).count (λ p, p.1 * p.2 > 10)

theorem four_digit_numbers_count :
  let first_digit_choices := 7
  let valid_pairs := valid_middle_digit_pairs
  let last_digit_choices := 10
  ∑ x in range first_digit_choices, 
  ∑ y in range valid_pairs, 
  ∑ z in range last_digit_choices, 1 = 3990 := 
sorry

end four_digit_numbers_count_l799_799374


namespace cos_value_l799_799806

variable (α : ℝ)

-- Conditions
def cond1 : Prop := sin (α + π / 3) + cos (α - π / 2) = -4 * sqrt 3 / 5
def cond2 : Prop := -π / 2 < α ∧ α < 0

-- Theorem statement
theorem cos_value (h1 : cond1 α) (h2 : cond2 α) : cos (α + 2 * π / 3) = 4 / 5 :=
sorry

end cos_value_l799_799806


namespace intersection_A_B_l799_799879

def A := { x : ℝ | -1 < x ∧ x ≤ 3 }
def B := { x : ℝ | 0 < x ∧ x < 10 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 3 } :=
  by sorry

end intersection_A_B_l799_799879


namespace surface_area_of_solid_l799_799161

def root_of_two_cubed := 1.26

def height_A := 2 / 3
def height_B := 1 / 4
def height_C := 1 / 12
def height_D := root_of_two_cubed - (height_A + height_B + height_C)

def top_bottom_surface_area := 2 * (root_of_two_cubed * root_of_two_cubed)
def side_surface_area := 4 * (root_of_two_cubed * root_of_two_cubed)
def front_back_surface_area := 2 * (root_of_two_cubed * 4 * height_A)

def total_surface_area := top_bottom_surface_area + side_surface_area + front_back_surface_area

theorem surface_area_of_solid : total_surface_area = 22.8032 :=
by
  -- skip proof
  sorry

end surface_area_of_solid_l799_799161


namespace max_value_of_function_l799_799518

noncomputable def f (x : ℝ) : ℝ := x^2 * real.exp (x + 1)

theorem max_value_of_function :
  ∃ (x : ℝ), x ∈ set.Icc (-2 : ℝ) 1 ∧ (∀ (y : ℝ), y ∈ set.Icc (-2 : ℝ) 1 → f y ≤ f x) ∧ f x = real.exp 2 :=
begin
  sorry
end

end max_value_of_function_l799_799518


namespace first_term_geometric_sequence_l799_799922

variable {a : ℕ → ℝ} -- Define the geometric sequence a_n
variable (q : ℝ) -- Define the common ratio q which is a real number

-- Conditions given in the problem
def geom_seq_first_term (a : ℕ → ℝ) (q : ℝ) :=
  a 3 = 2 ∧ a 4 = 4 ∧ (∀ n : ℕ, a (n+1) = a n * q)

-- Assert that if these conditions hold, then the first term is 1/2
theorem first_term_geometric_sequence (hq : geom_seq_first_term a q) : a 1 = 1/2 :=
by
  sorry

end first_term_geometric_sequence_l799_799922


namespace div_sqrt_81_by_3_is_3_l799_799683

-- Definitions based on conditions
def sqrt_81 := Nat.sqrt 81
def number_3 := 3

-- Problem statement
theorem div_sqrt_81_by_3_is_3 : sqrt_81 / number_3 = 3 := by
  sorry

end div_sqrt_81_by_3_is_3_l799_799683


namespace smallest_prime_with_digit_sum_23_l799_799604

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799604


namespace num_students_earning_B_l799_799923

open Real

theorem num_students_earning_B (total_students : ℝ) (pA : ℝ) (pB : ℝ) (pC : ℝ) (students_A : ℝ) (students_B : ℝ) (students_C : ℝ) :
  total_students = 31 →
  pA = 0.7 * pB →
  pC = 1.4 * pB →
  students_A = 0.7 * students_B →
  students_C = 1.4 * students_B →
  students_A + students_B + students_C = total_students →
  students_B = 10 :=
by
  intros h_total_students h_pa h_pc h_students_A h_students_C h_total_eq
  sorry

end num_students_earning_B_l799_799923


namespace lamps_eventually_off_lamps_never_all_off_l799_799553

/-- 
Problem (a): There are infinitely many integers n of the form 2^x for which all the lamps will eventually be off 
--/
theorem lamps_eventually_off (x : ℕ) (hx : x ≥ 1) : ∃ n, n = 2^x ∧ ∀ t, finite_time_off_condition n t := sorry

/-- 
Problem (b): There are infinitely many integers n of the form 2^x + 1 for which the lamps will never be all off 
--/
theorem lamps_never_all_off (x : ℕ) (hx : x ≥ 1) : ∃ n, n = 2^x + 1 ∧ infinite_time_on_condition n := sorry

/-- Definition (finite_time_off_condition): in finite time condition, lamps will eventually be off --/
def finite_time_off_condition (n t : ℕ) : Prop := 
  ∀ t_final, exists finite t_final where at t_final all lamps are off -- this needs to be precisely defined as per lamp switch rules

/-- Definition (infinite_time_on_condition): in infinite time condition, lamps cannot be all off --/
def infinite_time_on_condition (n : ℕ) : Prop := 
  ∀ t, not_exists finite t_final where at t_final all lamps off -- this needs to be precisely defined as per lamp switch rules

end lamps_eventually_off_lamps_never_all_off_l799_799553


namespace smallest_prime_with_digit_sum_23_l799_799577

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799577


namespace find_n_l799_799166

-- Define that Amy bought and sold 15n avocados.
def bought_sold_avocados (n : ℕ) := 15 * n

-- Define the profit function.
def calculate_profit (n : ℕ) : ℤ := 
  let total_cost := 10 * n
  let total_earnings := 12 * n
  total_earnings - total_cost

theorem find_n (n : ℕ) (profit : ℤ) (h1 : profit = 100) (h2 : profit = calculate_profit n) : n = 50 := 
by 
  sorry

end find_n_l799_799166


namespace number_on_board_after_hour_l799_799011

def digit_product (n : ℕ) : ℕ :=
  let digits := (n.toString.data.map (λ c, c.toNat - '0'.toNat))
  digits.foldl (λ acc d, acc * d) 1

def next_number (n : ℕ) : ℕ :=
  digit_product n + 12

noncomputable def number_after_minutes (initial_number : ℕ) (minutes : ℕ) : ℕ :=
  Nat.iterate next_number minutes initial_number

theorem number_on_board_after_hour : 
  number_after_minutes 27 60 = 14 :=
by
  sorry

end number_on_board_after_hour_l799_799011


namespace number_of_truthful_warriors_l799_799209

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799209


namespace midpoint_path_ellipse_l799_799869

-- Definitions of points and segments
variables {A B C D F : Type} [linear_ordered_field Type] [affine_space ℝ ℝ Type] [normed_space ℝ ℝ Type]
variables (AB : set Type) (CD : set Type) (F : Type)

-- Assumptions and conditions as Lean definitions and hypotheses
def fixed_AB (AB : set Type) : Prop := AB.fixed -- AG: assume existence of a function that defines fixed segments
def parallel_CD_AB (CD AB : set Type) : Prop := CD.parallel_to AB -- AG: likewise, a parallel relationship function
def inscribed_trapezoid (A B C D: Type) : Prop := Exists t : Type // (A, B, C, D form an inscribed trapezoid in E) -- hinged to exist a quadrilateral

-- The midpoint F of segment CD
def midpoint_F (CD : set Type) (F : Type) : Prop := midpoint CD F

-- The theorem to prove that the path of F is an ellipse
theorem midpoint_path_ellipse 
  (h1: fixed_AB AB) 
  (h2: parallel_CD_AB CD AB) 
  (h3: inscribed_trapezoid A B C D)
  (h4: midpoint_F CD F) : 
  ∃ ellipse : Type, follows_ellipse F ellipse :=
begin
  sorry
end

end midpoint_path_ellipse_l799_799869


namespace coordinates_of_point_B_l799_799317

noncomputable def point_A : (ℝ × ℝ) := (1, -2)
def vector_a : (ℝ × ℝ) := (2, 3)
def vector_ab_length : ℝ := 2 * Real.sqrt 13

theorem coordinates_of_point_B :
  ∃ B : (ℝ × ℝ), (∃ (λ : ℝ), λ > 0 ∧ B = (1 + 2 * λ, -2 + 3 * λ)) ∧ 
                 (Real.sqrt ((2 * λ)^2 + (3 * λ)^2) = 2 * Real.sqrt 13) ∧
                 B = (5, 4) :=
by
  sorry

end coordinates_of_point_B_l799_799317


namespace checkerboard_min_moves_l799_799086

theorem checkerboard_min_moves (m n : ℕ) : 
  ∃ k : ℕ, k = (n / 2).floor + (m / 2).floor ∧ 
  (∀ r c, 0 ≤ r < m → 0 ≤ c < n → 
    (color_flipped k (checkerboard m n) r c = monochrome)) := sorry

end checkerboard_min_moves_l799_799086


namespace grasshopper_jumps_rational_angle_l799_799074

noncomputable def alpha_is_rational (α : ℝ) (jump : ℕ → ℝ × ℝ) : Prop :=
  ∃ k n : ℕ, (n ≠ 0) ∧ (jump n = (0, 0)) ∧ (α = (k : ℝ) / (n : ℝ) * 360)

theorem grasshopper_jumps_rational_angle :
  ∀ (α : ℝ) (jump : ℕ → ℝ × ℝ),
    (∀ n : ℕ, dist (jump (n + 1)) (jump n) = 1) →
    (jump 0 = (0, 0)) →
    (∃ n : ℕ, n ≠ 0 ∧ jump n = (0, 0)) →
    alpha_is_rational α jump :=
by
  intros α jump jumps_eq_1 start_exists returns_to_start
  sorry

end grasshopper_jumps_rational_angle_l799_799074


namespace slope_condition_l799_799330

theorem slope_condition (a : ℝ) (h : (3 - (-a)) / (5 + a) = 1) : a = -4 :=
by
  have h1 : (3 + a) / (5 + a) = 1 := by rw [sub_neg_eq_add] at h; exact h
  have h2 : 3 + a = 5 + a := by exact (mul_eq_mul_right_iff.mp (mul_eq_mul_right_iff.mp h1.symm)).right
  have h3 : 3 = 5 := by exact add_left_cancel h2
  contradiction -- This is not necessary, it's just here to show the contradiction
  sorry

end slope_condition_l799_799330


namespace incorrect_option_D_l799_799351

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l799_799351


namespace hour_hand_angle_after_2_hours_l799_799738

-- Definitions
def hours_to_angle (hours : ℕ) : ℝ :=
  - (hours % 12) * (30 * (Math.PI / 180))

-- Theorem statement
theorem hour_hand_angle_after_2_hours : 
  hours_to_angle 2 = - (2 * 30 * (Math.PI / 180)) :=
by sorry

end hour_hand_angle_after_2_hours_l799_799738


namespace Helen_baked_139_cookies_this_morning_l799_799893

theorem Helen_baked_139_cookies_this_morning 
  (cookies_baked_yesterday : ℕ)
  (total_cookies_baked : ℕ)
  (H : cookies_baked_yesterday = 435)
  (H_total : total_cookies_baked = 574) :
  total_cookies_baked - cookies_baked_yesterday = 139 := 
by 
  rw [H, H_total] 
  norm_num

end Helen_baked_139_cookies_this_morning_l799_799893


namespace cosine_B_in_triangle_l799_799919

variable {A B C a b c : ℝ}

theorem cosine_B_in_triangle (h1 : sin B ^ 2 = sin A * sin C) (h2 : c = 2 * a) :
  cos B = 3 / 4 :=
by
  sorry

end cosine_B_in_triangle_l799_799919


namespace students_attending_Harvard_l799_799418

theorem students_attending_Harvard (total_applicants : ℕ) (perc_accepted : ℝ) (perc_attending : ℝ)
    (h1 : total_applicants = 20000)
    (h2 : perc_accepted = 0.05)
    (h3 : perc_attending = 0.9) :
    total_applicants * perc_accepted * perc_attending = 900 := 
by
    sorry

end students_attending_Harvard_l799_799418


namespace cosine_angle_between_vectors_l799_799886

theorem cosine_angle_between_vectors 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (2, -4)) (hb : b = (-3, -4)) :
  real.cos (real.angle (prod.fst a) (prod.snd a) (prod.fst b) (prod.snd b)) = (real.sqrt 5) / 5 :=
by {
  sorry
}

end cosine_angle_between_vectors_l799_799886


namespace odd_and_increasing_f1_odd_and_increasing_f2_l799_799093

-- Define the functions
def f1 (x : ℝ) : ℝ := x * |x|
def f2 (x : ℝ) : ℝ := x^3

-- Define the odd function property
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

-- Define the increasing function property
def is_increasing (f : ℝ → ℝ) : Prop := ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → f x1 < f x2

-- Lean statement to prove
theorem odd_and_increasing_f1 : is_odd f1 ∧ is_increasing f1 := by
  sorry

theorem odd_and_increasing_f2 : is_odd f2 ∧ is_increasing f2 := by
  sorry

end odd_and_increasing_f1_odd_and_increasing_f2_l799_799093


namespace packing_height_difference_l799_799072

theorem packing_height_difference (r : ℝ) (n : ℕ) (ha : n = 100) (hA : r = 5) :
  let height_A := 2 * r * 10,
      height_B := 10 * (r * sqrt 3)
  in abs (height_A - height_B) = 50 * (2 - sqrt 3) :=
by
  let height_A := 2 * r * 10
  let height_B := 10 * (r * sqrt 3)
  have h_height_diff : abs (height_A - height_B) = abs (2 * r * 10 - 10 * (r * sqrt 3)) := rfl
  exact h_height_diff sorry

end packing_height_difference_l799_799072


namespace minimize_material_l799_799113

theorem minimize_material (π V R h : ℝ) (hV : V > 0) (h_cond : π * R^2 * h = V) :
  R = h / 2 :=
sorry

end minimize_material_l799_799113


namespace hexagon_side_lengths_l799_799190

theorem hexagon_side_lengths (n : ℕ) (h1 : n ≥ 0) (h2 : n ≤ 6) (h3 : 10 * n + 8 * (6 - n) = 56) : n = 4 :=
sorry

end hexagon_side_lengths_l799_799190


namespace bill_difference_l799_799002

theorem bill_difference (mandy_bills : ℕ) (manny_bills : ℕ) 
  (mandy_bill_value : ℕ) (manny_bill_value : ℕ) (target_bill_value : ℕ) 
  (h_mandy : mandy_bills = 3) (h_mandy_val : mandy_bill_value = 20) 
  (h_manny : manny_bills = 2) (h_manny_val : manny_bill_value = 50)
  (h_target : target_bill_value = 10) :
  (manny_bills * manny_bill_value / target_bill_value) - (mandy_bills * mandy_bill_value / target_bill_value) = 4 :=
by
  sorry

end bill_difference_l799_799002


namespace price_of_first_variety_l799_799991

theorem price_of_first_variety
  (p2 : ℝ) (p3 : ℝ) (r : ℝ) (w : ℝ)
  (h1 : p2 = 135)
  (h2 : p3 = 177.5)
  (h3 : r = 154)
  (h4 : w = 4) :
  ∃ p1 : ℝ, 1 * p1 + 1 * p2 + 2 * p3 = w * r ∧ p1 = 126 :=
by {
  sorry
}

end price_of_first_variety_l799_799991


namespace domain_of_sqrt_function_l799_799036

theorem domain_of_sqrt_function :
  {x : ℝ | (1 / (Real.log x / Real.log 2) - 2 ≥ 0) ∧ (x > 0) ∧ (x ≠ 1)} 
  = {x : ℝ | 1 < x ∧ x ≤ Real.sqrt 10} :=
sorry

end domain_of_sqrt_function_l799_799036


namespace probability_odd_divisor_of_15_factorial_l799_799521

-- Define the factorial function
def fact : ℕ → ℕ
  | 0 => 1
  | (n+1) => (n+1) * fact n

-- Probability function for choosing an odd divisor
noncomputable def probability_odd_divisor (n : ℕ) : ℚ :=
  let prime_factors := [(2, 11), (3, 6), (5, 3), (7, 2), (11, 1), (13, 1)]
  let total_factors := prime_factors.foldr (λ p acc => (p.2 + 1) * acc) 1
  let odd_factors := ((prime_factors.filter (λ p => p.1 ≠ 2)).foldr (λ p acc => (p.2 + 1) * acc) 1)
  (odd_factors : ℚ) / (total_factors : ℚ)

-- Statement to prove the probability of an odd divisor
theorem probability_odd_divisor_of_15_factorial :
  probability_odd_divisor 15 = 1 / 12 :=
by
  -- Proof goes here, which is omitted as per the instructions
  sorry

end probability_odd_divisor_of_15_factorial_l799_799521


namespace sum_of_reciprocals_eq_six_l799_799530

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + y = 6 * x * y) (h2 : y = 2 * x) :
  (1 / x) + (1 / y) = 6 := by
  sorry

end sum_of_reciprocals_eq_six_l799_799530


namespace num_valid_functions_l799_799703

theorem num_valid_functions :
  ∃! (f : ℤ → ℝ), 
  (f 1 = 1) ∧ 
  (∀ (m n : ℤ), f m ^ 2 - f n ^ 2 = f (m + n) * f (m - n)) ∧ 
  (∀ n : ℤ, f n = f (n + 2013)) :=
sorry

end num_valid_functions_l799_799703


namespace union_sets_l799_799322

def set_A : Set ℝ := {x | x^3 - 3 * x^2 - x + 3 < 0}
def set_B : Set ℝ := {x | |x + 1 / 2| ≥ 1}

theorem union_sets :
  set_A ∪ set_B = ( {x : ℝ | x < -1} ∪ {x : ℝ | x ≥ 1 / 2} ) :=
by
  sorry

end union_sets_l799_799322


namespace smallest_prime_with_digit_sum_23_l799_799590

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799590


namespace fly_dist_ceiling_eq_sqrt255_l799_799719

noncomputable def fly_distance_from_ceiling : ℝ :=
  let x := 3
  let y := 5
  let d := 17
  let z := Real.sqrt (d^2 - (x^2 + y^2))
  z

theorem fly_dist_ceiling_eq_sqrt255 :
  fly_distance_from_ceiling = Real.sqrt 255 :=
by
  sorry

end fly_dist_ceiling_eq_sqrt255_l799_799719


namespace approx_pi_equals_22_div_7_l799_799503

-- We start by defining the known conditions
def cone_volume (π : ℝ) (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

def base_circumference (π : ℝ) (L : ℝ) : ℝ := L / (2 * π)

def alternate_volume (π : ℝ) (L h : ℝ) : ℝ := L^2 * h / (12 * π)

def approximated_volume (L h : ℝ) : ℝ := (7/264) * L^2 * h

-- We state the theorem to prove
theorem approx_pi_equals_22_div_7 (L h : ℝ) :
  ∃ π : ℝ, (alternate_volume π L h = approximated_volume L h) → π = 22 / 7 :=
by
  sorry

end approx_pi_equals_22_div_7_l799_799503


namespace determine_n_values_l799_799959

def is_friend (n x y : ℕ) : Prop :=
  ∃ (a b : ℕ), (a * x = b * y ∧ a * x % n ≠ 0)

def even_number_of_friends (n : ℕ) : Prop :=
  ∀ x ∈ finset.range (n-1), fintype.card {y // y ≠ x ∧ y ∈ finset.range (n-1) ∧ is_friend n x y} % 2 = 0

theorem determine_n_values : ∀ n : ℕ, (n ≥ 3 ∧ even_number_of_friends n) ↔ ∃ s : ℕ, n = 2 ^ s ∧ s ≥ 1 :=
sorry

end determine_n_values_l799_799959


namespace simplify_expression_l799_799025
-- Import the entire Mathlib library to ensure all necessary lemmas and theorems are available

-- Define the main problem as a theorem
theorem simplify_expression (t : ℝ) : 
  (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end simplify_expression_l799_799025


namespace calculate_suit_pants_cost_l799_799164

def budget := 200
def button_shirt := 30
def suit_coat := 38
def socks := 11
def belt := 18
def shoes := 41
def leftover := 16

theorem calculate_suit_pants_cost : 
  let total_expenses := button_shirt + suit_coat + socks + belt + shoes in
  let total_spent := budget - leftover in
  let suit_pants_cost := total_spent - total_expenses in
  suit_pants_cost = 46 := by
  sorry

end calculate_suit_pants_cost_l799_799164


namespace money_leftover_is_correct_l799_799101

noncomputable def cost_of_bread := 2.25
noncomputable def loaves_of_bread := 3
noncomputable def cost_of_peanut_butter := 2.00
noncomputable def initial_money := 14.00

def total_cost := (loaves_of_bread * cost_of_bread) + cost_of_peanut_butter
def money_leftover := initial_money - total_cost

theorem money_leftover_is_correct :
  money_leftover = 5.25 :=
by
  sorry

end money_leftover_is_correct_l799_799101


namespace sqrt_N_25636_l799_799246

theorem sqrt_N_25636 : 
  let N := (25636 : ℕ)
  √N = N :=
by
  sorry

end sqrt_N_25636_l799_799246


namespace least_number_l799_799270

theorem least_number (n : ℕ) (h1 : n % 38 = 1) (h2 : n % 3 = 1) : n = 115 :=
sorry

end least_number_l799_799270


namespace sequence_constant_gcd_l799_799955

theorem sequence_constant_gcd (a b : ℕ) (h1 : odd a) (h2 : odd b) (h3 : a > 0) (h4 : b > 0) :
  ∃ c, (∀ n ≥ N, fn n = c) ∧ c = gcd a b :=
begin
  -- define the sequence
  let f : ℕ → ℕ,
  intro m,
  cases m,
  exact a,
  cases m,
  exact b,
  exact (nat.greatest_odd_divisor (f m + f (m - 1))),
  --
  sorry
end

end sequence_constant_gcd_l799_799955


namespace students_attending_harvard_is_900_l799_799412

noncomputable def students_attending_harvard : ℕ :=
  let total_applicants := 20000
  let acceptance_rate := 0.05
  let yield_rate := 0.90
  let accepted_students := acceptance_rate * total_applicants
  let attending_students := yield_rate * accepted_students
  attending_students.to_nat

theorem students_attending_harvard_is_900 :
  students_attending_harvard = 900 :=
by
  -- proof will go here
  sorry

end students_attending_harvard_is_900_l799_799412


namespace smallest_prime_with_digit_sum_23_l799_799631

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799631


namespace num_ways_to_assign_roles_l799_799727

-- Definitions for men, women, and roles
def num_men : ℕ := 6
def num_women : ℕ := 7
def male_roles : ℕ := 3
def female_roles : ℕ := 3
def gender_neutral_roles : ℕ := 2

-- Permutations function
noncomputable def permutations : ℕ → ℕ → ℕ
| n, k := Nat.factorial n / Nat.factorial (n - k)

-- Proof statement
theorem num_ways_to_assign_roles :
  let men_perms := permutations num_men male_roles in
  let women_perms := permutations num_women female_roles in
  let remaining_individuals := num_men + num_women - male_roles - female_roles in
  let neutral_roles_perms := permutations remaining_individuals gender_neutral_roles in
  men_perms * women_perms * neutral_roles_perms = 1058400 :=
by
  sorry

end num_ways_to_assign_roles_l799_799727


namespace range_of_m_l799_799812

def P (x : ℝ) : Prop := |(4 - x) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) (h : m > 0) : (∀ x, ¬P x → ¬q x m) → m ≥ 9 :=
by
  intros
  sorry

end range_of_m_l799_799812


namespace angle_POQ_ninety_degrees_l799_799786

variables (A B C D O P Q : Type)
variables [metric_space A] 
variables [metric_space B] 
variables [metric_space C] 
variables [metric_space D] 
variables [metric_space Point]
noncomputable theory

open locale euclidean_geometry

-- Given condition: Diagonals AC and BD of quadrilateral ABCD are equal and meet at point O.
def diagonals_equal_and_meet (A B C D O : Point) : Prop :=
  dist A C = dist B D ∧ is_midpoint O A C ∧ is_midpoint O B D

-- Given condition: The perpendicular bisectors of segments AB and CD meet at point P.
def perpendicular_bisectors_meet_P (A B C D P : Point) : Prop :=
  is_perpendicular_bisector P A B ∧ is_perpendicular_bisector P C D

-- Given condition: The perpendicular bisectors of segments BC and AD meet at point Q.
def perpendicular_bisectors_meet_Q (B C A D Q : Point) : Prop :=
  is_perpendicular_bisector Q B C ∧ is_perpendicular_bisector Q A D

-- Conclusion: ∠POQ = 90°
theorem angle_POQ_ninety_degrees (A B C D O P Q : Point)
  (h1 : diagonals_equal_and_meet A B C D O)
  (h2 : perpendicular_bisectors_meet_P A B C D P)
  (h3 : perpendicular_bisectors_meet_Q B C A D Q) :
  ∠ P O Q = 90 := 
sorry

end angle_POQ_ninety_degrees_l799_799786


namespace four_digit_numbers_with_specific_thousands_digit_l799_799895

theorem four_digit_numbers_with_specific_thousands_digit :
    let count_digits := 1000 in
    let thousands_options := 3 in
    (count_digits * thousands_options) = 3000 :=
by
  sorry

end four_digit_numbers_with_specific_thousands_digit_l799_799895


namespace warriors_truth_tellers_l799_799202

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799202


namespace circle_numbers_max_l799_799059

theorem circle_numbers_max (numbers : Fin 10 → ℕ) (h_sum : (∑ i, numbers i) = 100)
  (h_consec : ∀ i : Fin 10, numbers i + numbers (i + 1) + numbers (i + 2) ≥ 29) :
  ∀ i : Fin 10, numbers i ≤ 13 := by
  sorry

end circle_numbers_max_l799_799059


namespace inequality_proof_l799_799019

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x^2 + y^2 + z^2 = 1) : 
  (xy / z + yz / x + xz / y) ≥ sqrt 3 :=
sorry

end inequality_proof_l799_799019


namespace smallest_prime_with_digit_sum_23_l799_799663

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799663


namespace path_count_from_A_to_B_l799_799492

theorem path_count_from_A_to_B : 
  let A := λ (_ : Type), Prop,
      B := λ (_ : Type), Prop,
      paths := finset (finset ℕ),
      directed_paths := {p ∈ paths | ∀ (x y ∈ p), x ≠ y ∧ x < y} in
  (card (directed_paths)) = 63 :=
by
  -- Definitions as per conditions:
  let A := λ (_ : Type), Prop
  let B := λ (_ : Type), Prop
  let paths := finset (finset ℕ)
  let directed_paths := {p ∈ paths | ∀ (x y ∈ p), x ≠ y ∧ x < y}

  -- State the theorem to be proven
  suffices : card (directed_paths) = 63, from this,
  -- Proof is omitted
  sorry

end path_count_from_A_to_B_l799_799492


namespace binom_10_0_equals_1_l799_799180

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem to prove that binom 10 0 = 1
theorem binom_10_0_equals_1 :
  binom 10 0 = 1 := by
  sorry

end binom_10_0_equals_1_l799_799180


namespace smallest_prime_with_digit_sum_23_l799_799660

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799660


namespace jennifer_remaining_money_l799_799943

noncomputable def money_spent_on_sandwich (initial_money : ℝ) : ℝ :=
  let sandwich_cost := (1/5) * initial_money
  let discount := (10/100) * sandwich_cost
  sandwich_cost - discount

noncomputable def money_spent_on_ticket (initial_money : ℝ) : ℝ :=
  (1/6) * initial_money

noncomputable def money_spent_on_book (initial_money : ℝ) : ℝ :=
  (1/2) * initial_money

noncomputable def money_after_initial_expenses (initial_money : ℝ) (gift : ℝ) : ℝ :=
  initial_money - money_spent_on_sandwich initial_money - money_spent_on_ticket initial_money - money_spent_on_book initial_money + gift

noncomputable def money_spent_on_cosmetics (remaining_money : ℝ) : ℝ :=
  (1/4) * remaining_money

noncomputable def money_after_cosmetics (remaining_money : ℝ) : ℝ :=
  remaining_money - money_spent_on_cosmetics remaining_money

noncomputable def money_spent_on_tshirt (remaining_money : ℝ) : ℝ :=
  let tshirt_cost := (1/3) * remaining_money
  let tax := (5/100) * tshirt_cost
  tshirt_cost + tax

noncomputable def remaining_money (initial_money : ℝ) (gift : ℝ) : ℝ :=
  let after_initial := money_after_initial_expenses initial_money gift
  let after_cosmetics := after_initial - money_spent_on_cosmetics after_initial
  after_cosmetics - money_spent_on_tshirt after_cosmetics

theorem jennifer_remaining_money : remaining_money 90 30 = 21.35 := by
  sorry

end jennifer_remaining_money_l799_799943


namespace part1_find_A_part2_find_c_l799_799324

open Real

variable {a b c A B C : ℝ}

-- Conditions
axiom cond1 : a = opposite A
axiom cond2 : b = opposite B
axiom cond3 : c = opposite C
axiom cond4 : sqrt 3 * a * cos C + a * sin C = sqrt 3 * b 
axiom cond5 : c^2 = 4 * a^2 - 4 * b^2
axiom cond6 : a + b = (3 + sqrt 13) / 2

-- Statement for part (1)
theorem part1_find_A : A = π / 3 :=
sorry

-- Statement for part (2)
theorem part2_find_c : c = 2 :=
sorry

end part1_find_A_part2_find_c_l799_799324


namespace part_I_part_II_part_III_l799_799338

-- Part (I)
theorem part_I (x : ℝ) : 
  2 < x ∧ log 2 (x - 2) + log 2 (x - 4) < log 2 3 ↔ 4 < x ∧ x < 5 :=
sorry

-- Part (II)
theorem part_II (m : ℝ) :
  (∀ x, 3 * m ≤ x ∧ x ≤ 4 * m → log m ((x - m) * (x - 2 * m)) ≤ 1) ↔ 1 / 2 ≤ m ∧ m < 1 :=
sorry

-- Part (III)
theorem part_III (m : ℝ) :
  1 / 2 ≤ m ∧ m < 1 → ∃ α β, α > 5 / 2 * m ∧ β > 5 / 2 * m ∧
  (∀ x, α ≤ x ∧ x ≤ β → log m (x - m) + log m (x - 2 * m) ∈ [log m β, log m α]) → False :=
sorry

end part_I_part_II_part_III_l799_799338


namespace find_x_l799_799296

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end find_x_l799_799296


namespace prime_factors_of_x_l799_799305

theorem prime_factors_of_x (n : ℕ) (h1 : 2^n - 32 = x) (h2 : (nat.prime_factors x).length = 3) (h3 : 3 ∈ nat.prime_factors x) :
  x = 480 ∨ x = 2016 :=
sorry

end prime_factors_of_x_l799_799305


namespace math_problem_l799_799742

theorem math_problem :
  (0.8 ^ (-0.1) < 0.8 ^ (-0.2)) ∧ (log 7 6 > log 8 6) ∧
  ¬(log 2 3.4 < log 2 real.pi) ∧ ¬(1.7 ^ 1.01 < 1.6 ^ 1.01) :=
by
  sorry

end math_problem_l799_799742


namespace max_real_solution_under_100_l799_799017

theorem max_real_solution_under_100 (k a b c r : ℕ) (h0 : ∃ (m n p : ℕ), a = k^m ∧ b = k^n ∧ c = k^p)
  (h1 : r < 100) (h2 : b^2 = 4 * a * c) (h3 : r = b / (2 * a)) : r ≤ 64 :=
sorry

end max_real_solution_under_100_l799_799017


namespace solve_system_of_equations_l799_799486

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ (x = 2) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l799_799486


namespace degree_h_is_3_l799_799904

noncomputable def f (x : ℝ) : ℝ := -5 * x^3 + 6 * x^2 + 2 * x - 8

variable (h : ℝ → ℝ)

axiom h_is_polynomial : polynomial.h (h)
axiom degree_f_plus_h : polynomial.degree (λ x : ℝ, f x + h x) = 2

theorem degree_h_is_3 : polynomial.degree h = 3 := sorry

end degree_h_is_3_l799_799904


namespace monotonically_increasing_l799_799339

noncomputable def f (x : ℝ) : ℝ :=
  sin (4 * x + 3 * Real.pi / 4) + cos (4 * x + 3 * Real.pi / 4)

theorem monotonically_increasing :
  ∀ (x y : ℝ), (π / 8 < x) → (x < 3 * π / 8) → (π / 8 < y) → (y < 3 * π / 8) → (x < y) → f x < f y :=
by
  sorry

end monotonically_increasing_l799_799339


namespace part2_l799_799834

noncomputable def a (n : ℕ) : ℕ := 
if h : n = 1 then 3 else
let n' := n-1 in
3 * (n + 1) * a n' / n

def geometric_sequence (a : ℕ → ℕ) :=
∀ n : ℕ, a (n + 1) / a n = 3

lemma part1 (n : ℕ) : a 1 = 3 ∧ (∀ n : ℕ, n * a (n + 1) = 3 * (n + 1) * (a n))
  → geometric_sequence (λ n, a n / n) :=
sorry

def bn (n : ℕ) (a : ℕ → ℕ) := n^2 / a n

def Tn (n : ℕ) (a : ℕ → ℕ) : ℕ → ℕ
| 0 := 0
| (k+1) := bn (k+1) a + Tn k a

theorem part2 (n : ℕ) : 
  a 1 = 3 ∧ (∀ n : ℕ, n * a (n + 1) = 3 * (n + 1) * (a n)) 
  → Tn n (λ n, if h : n = 1 then 3 else a n) = 
        (3 / 4 : ℚ) - (2 * n + 3) / (4 * 3^n) :=
sorry

end part2_l799_799834


namespace value_of_frac_mul_l799_799901

theorem value_of_frac_mul (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 2 * d) :
  (a * c) / (b * d) = 8 :=
by
  sorry

end value_of_frac_mul_l799_799901


namespace b_arithmetic_seq_value_S_sum_l799_799309

noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := 2 - 1 / a n

def b (n : ℕ) : ℝ :=
1 / (a n - 1)

theorem b_arithmetic_seq : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d := 
sorry

def S (n : ℕ) : ℝ :=
∑ i in Finset.range n, b i

theorem value_S_sum (n : ℕ) : (Finset.range n).sum (λ k, 1 / S (k + 1)) = 2 * n / (n + 1) :=
sorry

end b_arithmetic_seq_value_S_sum_l799_799309


namespace sandy_gain_percent_l799_799481

-- Define the individual costs
def cost_scooter : ℝ := 800
def first_repair : ℝ := 150
def second_repair : ℝ := 75
def third_repair : ℝ := 225
def taxes : ℝ := 50
def maintenance_fee : ℝ := 100

-- Define the selling price
def selling_price : ℝ := 1900

-- Define the total cost
def total_cost : ℝ := cost_scooter + first_repair + second_repair + third_repair + taxes + maintenance_fee

-- Define the gain
def gain : ℝ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℝ := (gain / total_cost) * 100

-- The theorem stating that the gain percent is 35.714%
theorem sandy_gain_percent : gain_percent = 35.714 := by
  sorry

end sandy_gain_percent_l799_799481


namespace number_of_heaps_is_5_l799_799170

variable (bundles : ℕ) (bunches : ℕ) (heaps : ℕ) (total_removed : ℕ)
variable (sheets_per_bunch : ℕ) (sheets_per_bundle : ℕ) (sheets_per_heap : ℕ)

def number_of_heaps (bundles : ℕ) (sheets_per_bundle : ℕ)
                    (bunches : ℕ) (sheets_per_bunch : ℕ)
                    (total_removed : ℕ) (sheets_per_heap : ℕ) :=
  (total_removed - (bundles * sheets_per_bundle + bunches * sheets_per_bunch)) / sheets_per_heap

theorem number_of_heaps_is_5 :
  number_of_heaps 3 2 2 4 114 20 = 5 :=
by
  unfold number_of_heaps
  sorry

end number_of_heaps_is_5_l799_799170


namespace F_at_extreme_point_l799_799446

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * Math.log (x + 1)
noncomputable def F (x a : ℝ) : ℝ := f x a + Math.log (Real.sqrt 2)

theorem F_at_extreme_point (a x₁ x₂ : ℝ) 
  (h_cond : 0 < a ∧ a < 1/2) 
  (h_extreme : x₁ < x₂) 
  (h_F_ext : ∃ x1 x2, x1 < x2 ∧ ( ∀ y, y ≠ x1 ∧ y ≠ x2 → deriv (λ x, F x a) y = 0))
  : F x₂ a > 1/4 := 
sorry

end F_at_extreme_point_l799_799446


namespace smallest_prime_with_digit_sum_23_l799_799651

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799651


namespace moms_took_chocolates_l799_799467

theorem moms_took_chocolates (N : ℕ) (A : ℕ) (M : ℕ) : 
  N = 10 → 
  A = 3 * N →
  A - M = N + 15 →
  M = 5 :=
by
  intros h1 h2 h3
  sorry

end moms_took_chocolates_l799_799467


namespace right_triangle_congruence_l799_799721

theorem right_triangle_congruence
  (ABC : Type) [triangle ABC]
  (A'B'C' : Type) [triangle A'B'C']
  (right_angle_ABC : angle ABC = 90)
  (right_angle_A'B'C' : angle A'B'C' = 90)
  (BC_eq_B'C' : side_length BC = side_length B'C')
  (AC_eq_A'C' : side_length AC = side_length A'C') :
  congruent ABC A'B'C' :=
sorry

end right_triangle_congruence_l799_799721


namespace number_of_truthful_warriors_l799_799212

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799212


namespace rainfall_on_tuesday_is_correct_l799_799697

-- Define the total days in a week
def days_in_week : ℕ := 7

-- Define the average rainfall for the whole week
def avg_rainfall : ℝ := 3.0

-- Define the total rainfall for the week
def total_rainfall : ℝ := avg_rainfall * days_in_week

-- Define a proposition that states rainfall on Tuesday equals 10.5 cm
def rainfall_on_tuesday (T : ℝ) : Prop :=
  T = 10.5

-- Prove that the rainfall on Tuesday is 10.5 cm given the conditions
theorem rainfall_on_tuesday_is_correct : rainfall_on_tuesday (total_rainfall / 2) :=
by
  sorry

end rainfall_on_tuesday_is_correct_l799_799697


namespace smallest_prime_with_digit_sum_23_proof_l799_799681

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799681


namespace truthful_warriors_count_l799_799215

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799215


namespace parallel_line_eq_l799_799799

theorem parallel_line_eq (a b c : ℝ) (p : ℝ × ℝ) (eq1 eq2 : ℝ → ℝ → ℝ)
  (h1 : eq1 = λ x y, x + 2 * y + c)
  (h2 : eq2 = λ x y, x + 2 * y - 2)
  (h3 : p = (2, 0))
  (h4 : eq1 2 0 = 0) : eq2 2 0 = 0 :=
by
  sorry

end parallel_line_eq_l799_799799


namespace sin_alpha_cos_2beta_l799_799380

theorem sin_alpha_cos_2beta :
  ∀ α β : ℝ, 3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2 →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 :=
by
  intros α β h
  sorry

end sin_alpha_cos_2beta_l799_799380


namespace jake_peaches_l799_799942

variable (Steven Jake : ℕ)

-- Conditions
def steven_peaches : Steven = 16 := sorry
def jake_fewer_steven : Jake = Steven - 7 := sorry

-- Proof statement
theorem jake_peaches : Jake = 9 := by
  rw [←jake_fewer_steven, steven_peaches]
  simp
  sorry

end jake_peaches_l799_799942


namespace integer_solutions_l799_799267

theorem integer_solutions (a b c : ℤ) (h₁ : 1 < a) 
    (h₂ : a < b) (h₃ : b < c) 
    (h₄ : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) 
    ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by sorry

end integer_solutions_l799_799267


namespace equation_of_line_perpendicular_to_l_l799_799289

open Real

theorem equation_of_line_perpendicular_to_l
  (a : ℝ) (h_pos : a > 0)
  (h_chord : √(a^2 - 1) = 2)
  (h_center : (3 : ℝ) = a)
  :
  ∃ m : ℝ, (3 + 0 + m = 0) ∧ (∀ b : ℝ, ∃ x y : ℝ, l x y → x + y + m = 0) :=
by
  sorry

end equation_of_line_perpendicular_to_l_l799_799289


namespace circles_tangent_iff_m_eq_2_or_minus5_l799_799547

-- Definitions of the circles' equations and external tangency condition
def circle1 (x y m : ℝ) := (x + 2) ^ 2 + (y - m) ^ 2 = 9
def circle2 (x y m : ℝ) := (x - m) ^ 2 + (y + 1) ^ 2 = 4
def externally_tangent {x1 y1 r1 x2 y2 r2 : ℝ} := (real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = r1 + r2)

-- The centers and radii of the circles from their equations
def center1 (m : ℝ) : ℝ × ℝ := (-2, m)
def center2 (m : ℝ) : ℝ × ℝ := (m, -1)
def radius1 : ℝ := 3
def radius2 : ℝ := 2

-- Proof problem: Prove that for the circles to be externally tangent, m must be 2 or -5
theorem circles_tangent_iff_m_eq_2_or_minus5 (m : ℝ) :
  externally_tangent (center1 m).1 (center1 m).2 radius1 (center2 m).1 (center2 m).2 radius2 ↔ 
  (m = 2 ∨ m = -5) :=
by
  sorry

end circles_tangent_iff_m_eq_2_or_minus5_l799_799547


namespace smallest_prime_with_digit_sum_23_l799_799668

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799668


namespace equivalent_form_l799_799193

variable (x y : ℝ)
variable (hx : x ≠ 0) (hy : y ≠ 0)

theorem equivalent_form :
  x ≠ 0 → y ≠ 0 → 
  (x^(-2) * y^(-1)) / (x^(-4) - y^(-2)) = (x^2 * y) / (y^2 - x^2) :=
by 
  intro hx hy,
  sorry

end equivalent_form_l799_799193


namespace smallest_prime_with_digit_sum_23_l799_799569

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799569


namespace part_I_part_II_l799_799864

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 1

theorem part_I {a : ℝ} (ha : a = 2) :
  { x : ℝ | f x a ≥ 4 - abs (x - 4)} = { x | x ≥ 11 / 2 ∨ x ≤ 1 / 2 } := 
by 
  sorry

theorem part_II {a : ℝ} (h : { x : ℝ | abs (f (2 * x + a) a - 2 * f x a) ≤ 1 } = 
      { x | 1 / 2 ≤ x ∧ x ≤ 1 }) : 
  a = 2 := 
by 
  sorry

end part_I_part_II_l799_799864


namespace perpendicular_planes_l799_799818

theorem perpendicular_planes 
  (m l : Line)
  (α β : Plane) 
  (h1 : m ⊆ β)
  (h2 : m ⊥ α) : 
  α ⊥ β := 
sorry

end perpendicular_planes_l799_799818


namespace binom_10_0_eq_1_l799_799179

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem stating the binomial coefficient we need to prove
theorem binom_10_0_eq_1 : binom 10 0 = 1 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end binom_10_0_eq_1_l799_799179


namespace largest_three_digit_multiple_of_12_and_sum_of_digits_24_l799_799555

def sum_of_digits (n : ℕ) : ℕ :=
  ((n / 100) + ((n / 10) % 10) + (n % 10))

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def largest_three_digit_multiple_of_12_with_digits_sum_24 : ℕ :=
  996

theorem largest_three_digit_multiple_of_12_and_sum_of_digits_24 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ sum_of_digits n = 24 ∧ is_multiple_of_12 n ∧ n = largest_three_digit_multiple_of_12_with_digits_sum_24 :=
by 
  sorry

end largest_three_digit_multiple_of_12_and_sum_of_digits_24_l799_799555


namespace abcd_inequality_l799_799365

theorem abcd_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) :=
sorry

end abcd_inequality_l799_799365


namespace median_range_l799_799425

def triangle_abc (A B C : Type) [HasDist A A] := is_triangle A B C

variable {A B C : Type} [HasDist A A]

theorem median_range (h : triangle_abc A B C) 
(AD : A) (AE : A) (m : ℝ)
(h1 : dist A D = 12) (h2 : dist A E = 13) :
(angle A B C).is_acute ↔ 13 < m ∧ m < 17.05 ∧ 
(angle A B C).is_right ↔ m = 17.05 ∧ 
(angle A B C).is_obtuse ↔ 17.05 < m :=
sorry

end median_range_l799_799425


namespace part1_part2_l799_799336

noncomputable def f (x : ℝ) : ℝ := x / (3 * x + 1)

def a_seq : ℕ → ℝ
| 0     := 1
| (n+1) := f (a_seq n)

def inv_a_seq (n : ℕ) : ℝ := 1 / (a_seq n)

def S (n : ℕ) : ℝ := 
  (Finset.range n).sum (λ k, (a_seq k) * (a_seq (k + 1)))

theorem part1 (n : ℕ) : ∃ d : ℝ, ∀ n : ℕ, inv_a_seq (n+1) - inv_a_seq n = d := by
  sorry

theorem part2 (n : ℕ) : S n = n / (3 * n + 1) := by
  sorry

end part1_part2_l799_799336


namespace range_of_a_l799_799911

noncomputable def is_monotonic (f : ℝ → ℝ) := 
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x > 1 then a ^ x else (4 - a / 2) * x + 2

theorem range_of_a {a : ℝ} :
  is_monotonic (f a) ↔ 4 ≤ a ∧ a < 8 :=
by
  sorry

end range_of_a_l799_799911


namespace exponentiation_rule_l799_799687

theorem exponentiation_rule (a : ℝ) : (a^4) * (a^4) = a^8 :=
by 
  sorry

end exponentiation_rule_l799_799687


namespace smallest_prime_with_digit_sum_23_l799_799643

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799643


namespace employee_pays_180_l799_799153

noncomputable def wholesale_cost := 200
noncomputable def retail_percent_increase := 0.20
noncomputable def employee_discount := 0.25
noncomputable def retail_price := wholesale_cost * (1 + retail_percent_increase)
noncomputable def discount_amount := retail_price * employee_discount
noncomputable def employee_price := retail_price - discount_amount

theorem employee_pays_180 : employee_price = 180 := 
  by
    sorry

end employee_pays_180_l799_799153


namespace rotation_phenomena_l799_799039

/-- 
The rotation of the hour hand fits the definition of rotation since it turns around 
the center of the clock, covering specific angles as time passes.
-/
def is_rotation_of_hour_hand : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The rotation of the Ferris wheel fits the definition of rotation since it turns around 
its central axis, making a complete circle.
-/
def is_rotation_of_ferris_wheel : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The annual decline of the groundwater level does not fit the definition of rotation 
since it is a vertical movement (translation).
-/
def is_not_rotation_of_groundwater_level : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The movement of the robots on the conveyor belt does not fit the definition of rotation 
since it is a linear/translational movement.
-/
def is_not_rotation_of_robots_on_conveyor : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
Proof that the phenomena which belong to rotation are exactly the rotation of the hour hand 
and the rotation of the Ferris wheel.
-/
theorem rotation_phenomena :
  is_rotation_of_hour_hand ∧ 
  is_rotation_of_ferris_wheel ∧ 
  is_not_rotation_of_groundwater_level ∧ 
  is_not_rotation_of_robots_on_conveyor →
  "①②" = "①②" :=
by
  intro h
  sorry

end rotation_phenomena_l799_799039


namespace sam_travel_time_l799_799100

theorem sam_travel_time (d_AC d_CB : ℕ) (v_sam : ℕ) 
  (h1 : d_AC = 600) (h2 : d_CB = 400) (h3 : v_sam = 50) : 
  (d_AC + d_CB) / v_sam = 20 := 
by
  sorry

end sam_travel_time_l799_799100


namespace smallest_prime_with_digit_sum_23_proof_l799_799675

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799675


namespace bogatyrs_truthful_count_l799_799245

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799245


namespace correct_total_syrup_l799_799130

variable (syrup_per_shake syrup_per_cone syrup_per_sundae extra_syrup_per_decoration : ℝ)
variable (num_shakes num_cones num_sundaes : ℕ)
variable (decoration_percentage : ℝ)

-- Define the constants based on the given conditions
def syrup_per_shake := 5.5
def syrup_per_cone := 8.0
def syrup_per_sundae := 4.2
def extra_syrup_per_decoration := 0.3
def num_shakes := 5
def num_cones := 4
def num_sundaes := 3
def decoration_percentage := 0.1

-- Calculate the total amount of syrup used
noncomputable def total_syrup_used : ℝ :=
  let shakes_syrup := num_shakes * syrup_per_shake
  let cones_syrup := num_cones * syrup_per_cone
  let sundaes_syrup := num_sundaes * syrup_per_sundae
  let total_shakes_cones := num_shakes + num_cones
  let extra_decorations := ((decoration_percentage * total_shakes_cones).ceil : ℕ)
  let extra_syrup := extra_decorations * extra_syrup_per_decoration
  shakes_syrup + cones_syrup + sundaes_syrup + extra_syrup

theorem correct_total_syrup : total_syrup_used = 72.4 := by
  sorry

end correct_total_syrup_l799_799130


namespace solve_for_x_l799_799988

theorem solve_for_x (x y z : ℝ) (h1 : x * y = 8 - 3 * x - 2 * y) 
                                  (h2 : y * z = 8 - 2 * y - 3 * z) 
                                  (h3 : x * z = 35 - 5 * x - 3 * z) : 
  x = 8 :=
sorry

end solve_for_x_l799_799988


namespace smallest_prime_with_digit_sum_23_l799_799642

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799642


namespace product_seqFrac_l799_799753

def seqFrac (n : ℕ) : ℚ := (n : ℚ) / (n + 5 : ℚ)

theorem product_seqFrac :
  ((List.range 53).map seqFrac).prod = 1 / 27720 := by
  sorry

end product_seqFrac_l799_799753


namespace number_of_valid_four_digit_numbers_l799_799369

-- Definition for the problem
def is_valid_four_digit_number (n : ℕ) : Prop :=
  2999 < n ∧ n <= 9999 ∧
  (let d1 := n / 1000,
       d2 := (n / 100) % 10,
       d3 := (n / 10) % 10,
       d4 := n % 10 in
   3 <= d1 ∧ d1 <= 9 ∧
   0 <= d4 ∧ d4 <= 9 ∧
   d2 * d3 > 10)

-- Statement of the problem
theorem number_of_valid_four_digit_numbers : 
  (Finset.range 10000).filter is_valid_four_digit_number).card = 4830 :=
sorry

end number_of_valid_four_digit_numbers_l799_799369


namespace sum_of_products_of_roots_l799_799454

open Polynomial

noncomputable def polynomial := 2 * (X : ℚ[X])^4 - 6 * X^3 + 14 * X^2 - 13 * X + 8

theorem sum_of_products_of_roots :
  let roots := Polynomial.roots (polynomial) in
  ab + ac + ad + bc + bd + cd = -7 
  := sorry

end sum_of_products_of_roots_l799_799454


namespace smallest_prime_with_digit_sum_23_proof_l799_799680

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799680


namespace kitty_vacuum_time_l799_799974

theorem kitty_vacuum_time
  (weekly_toys : ℕ := 5)
  (weekly_windows : ℕ := 15)
  (weekly_furniture : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  (weeks : ℕ := 4)
  : (weekly_toys + weekly_windows + weekly_furniture) * weeks < total_cleaning_time ∧ ((total_cleaning_time - ((weekly_toys + weekly_windows + weekly_furniture) * weeks)) / weeks = 20)
  := by
  sorry

end kitty_vacuum_time_l799_799974


namespace clock_strike_time_l799_799698

theorem clock_strike_time (t_12 : ℕ) (h_t_12 : t_12 = 33) : 
  let i_12 := 12 - 1 in
  let t_per_interval := t_12 / i_12 in
  let i_6 := 6 - 1 in
  let t_6 := i_6 * t_per_interval in
  t_6 = 15 :=
by
  sorry

end clock_strike_time_l799_799698


namespace prove_slope_of_line_l799_799288

variable {a m : ℝ} 
variable (h₀ : m ≠ 0) 
variable (h₁ : -((2 * a) / (3 * m)) + (-2) = 2)

def slope_of_line (a m : ℝ) : ℝ := 
  let b := -((2 * a) / (3 * m))
  (-(3 * m) / a)

theorem prove_slope_of_line : slope_of_line a m = 2 :=
  by
    sorry

end prove_slope_of_line_l799_799288


namespace product_of_extremes_l799_799803

theorem product_of_extremes :
  let numbers := {2.8, 2.3, 5, 3, 4.3}
  let smallest := 2.3
  let largest := 5
  let expected_product := 11.5
  expected_product = smallest * largest := 
by
  sorry

end product_of_extremes_l799_799803


namespace sum_sequence_eq_l799_799284

noncomputable def S (n : ℕ) : ℝ := Real.log (1 + n) / Real.log 0.1

theorem sum_sequence_eq :
  (S 99 - S 9) = -1 := by
  sorry

end sum_sequence_eq_l799_799284


namespace smallest_prime_with_digit_sum_23_l799_799648

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799648


namespace rhombus_area_filled_by_fourth_vertex_l799_799542

-- Define the square with side length 1
structure Square :=
(A B C D : Vector ℝ)
(side_length : ℝ)
(nonneg : 0 ≤ side_length)

-- Define the rhombus with three vertices on the sides AB, BC, and AD of the square
structure RhombusOnSquare (S : Square) :=
(K L N : Vector ℝ)
(K_on_AB : K ∈ line_segment S.A S.B)
(L_on_BC : L ∈ line_segment S.B S.C)
(N_on_AD : N ∈ line_segment S.A S.D)
equal_sides : dist K L = dist L N ∧ dist L N = dist N K

-- Define the proof problem: given the conditions above, prove the area covered by the fourth vertex P equals 1
theorem rhombus_area_filled_by_fourth_vertex (S : Square) (R : RhombusOnSquare S) :
  ∃ P : Vector ℝ, (area_filled_by P = 1) :=
sorry

end rhombus_area_filled_by_fourth_vertex_l799_799542


namespace number_of_truthful_warriors_l799_799211

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799211


namespace pastries_eaten_l799_799705

theorem pastries_eaten (total_p: ℕ)
  (hare_fraction: ℚ)
  (dormouse_fraction: ℚ)
  (hare_eaten: ℕ)
  (remaining_after_hare: ℕ)
  (dormouse_eaten: ℕ)
  (final_remaining: ℕ) 
  (hatter_with_left: ℕ) :
  (final_remaining = hatter_with_left) -> hare_fraction = 5 / 16 -> dormouse_fraction = 7 / 11 -> hatter_with_left = 8 -> total_p = 32 -> 
  (total_p = hare_eaten + remaining_after_hare) -> (remaining_after_hare - dormouse_eaten = hatter_with_left) -> (hare_eaten = 10) ∧ (dormouse_eaten = 14) := 
by {
  sorry
}

end pastries_eaten_l799_799705


namespace right_triangle_hypotenuse_consecutive_even_l799_799390

theorem right_triangle_hypotenuse_consecutive_even (x : ℕ) (h : x ≠ 0) :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ ((a, b, c) = (x - 2, x, x + 2) ∨ (a, b, c) = (x, x - 2, x + 2) ∨ (a, b, c) = (x + 2, x, x - 2)) ∧ c = 10 := 
by
  sorry

end right_triangle_hypotenuse_consecutive_even_l799_799390


namespace dimension_commutative_space_l799_799948

variable {n : ℕ}
variable {k : ℕ}
variable {d : Fin k → ℕ}
variable {c : Fin k → ℂ}  -- Complex numbers are used to ensure distinct eigenvalues

-- A is a diagonal matrix with distinct eigenvalues, leading to specific characteristic polynomial
def is_diag_with_char_poly (A : Matrix (Fin n) (Fin n) ℂ)
  (d : Fin k → ℕ) (c : Fin k → ℂ) : Prop :=
  A.isDiag ∧ 
  (∀ i j, (i ≠ j → A i i ≠ A j j) ∧ (i = j → A i i = c (Fin.ofNat' i % k))) ∧
  (∑ i, d i = n)

-- The space V of all n x n matrices B such that AB = BA
def commutative_matrices (A : Matrix (Fin n) (Fin n) ℂ) : Type :=
  { B : Matrix (Fin n) (Fin n) ℂ // A.mul B = B.mul A }

-- Statement of the problem
theorem dimension_commutative_space
  (A : Matrix (Fin n) (Fin n) ℂ)
  (hA : is_diag_with_char_poly A d c) :
  FiniteDimensional.finRank ℂ (commutative_matrices A) = ∑ i, (d i) ^ 2 := sorry

end dimension_commutative_space_l799_799948


namespace sum_of_cubes_l799_799382

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end sum_of_cubes_l799_799382


namespace greatest_third_side_of_triangle_l799_799079

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : ∃ x : ℕ, x < a + b ∧ x = 16 := by
  use 16
  rw [h1, h2]
  split
  · linarith
  · rfl

end greatest_third_side_of_triangle_l799_799079


namespace find_x_l799_799304

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : Nat.coprime 2 3)
  (h3 : (∀ p : ℕ, Prime p → p ∣ x → p = 2 ∨ p = 3))
  (h4 : Nat.count (λ p, Prime p ∧ p ∣ x) = 3)
  : x = 480 ∨ x = 2016 :=
by sorry

end find_x_l799_799304


namespace ball_hits_ground_l799_799508

theorem ball_hits_ground :
  ∃ (t : ℝ), (t = 2) ∧ (-4.9 * t^2 + 5.7 * t + 7 = 0) :=
sorry

end ball_hits_ground_l799_799508


namespace problem_l799_799790

theorem problem : 3^128 + 8^5 / 8^3 = 65 := sorry

end problem_l799_799790


namespace tetrahedron_inscribed_in_cylinder_CD_length_l799_799772

noncomputable def length_CD_possible_values (AB AC CB AD DB : ℝ) (cylinder_cond : Prop) : set ℝ :=
  { sqrt 47 + sqrt 34, |sqrt 47 - sqrt 34| }

theorem tetrahedron_inscribed_in_cylinder_CD_length (AB AC CB AD DB : ℝ) (cylinder_cond : Prop) :
  AB = 2 → AC = 6 → CB = 6 → AD = 7 → DB = 7 → cylinder_cond →
  length_CD_possible_values AB AC CB AD DB cylinder_cond = {sqrt 47 + sqrt 34, |sqrt 47 - sqrt 34|} :=
by
  intros hAB hAC1 hCB hAD hDB hcylinder_cond
  sorry

end tetrahedron_inscribed_in_cylinder_CD_length_l799_799772


namespace convex_inscribed_quadrilateral_l799_799771

noncomputable def distance (A B : ℤ × ℤ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem convex_inscribed_quadrilateral {A B : ℤ × ℤ} (h_distinct: A ≠ B) :
  ∃ X Y : ℤ × ℤ, X ≠ Y ∧ convex (polygon.points [A, X, B, Y]) ∧ inscribed (polygon.points [A, X, B, Y]) ↔ distance A B ≠ 1 :=
sorry

end convex_inscribed_quadrilateral_l799_799771


namespace bobbit_worm_days_l799_799066

variable (initial_fish : ℕ)
variable (fish_added : ℕ)
variable (fish_eaten_per_day : ℕ)
variable (week_days : ℕ)
variable (final_fish : ℕ)
variable (d : ℕ)

theorem bobbit_worm_days (h1 : initial_fish = 60)
                         (h2 : fish_added = 8)
                         (h3 : fish_eaten_per_day = 2)
                         (h4 : week_days = 7)
                         (h5 : final_fish = 26) :
  60 - 2 * d + 8 - 2 * week_days = 26 → d = 14 :=
by {
  sorry
}

end bobbit_worm_days_l799_799066


namespace average_price_l799_799714

theorem average_price (
  (m p_1 p_2 : ℝ)
  (p_1_price : p_1 = 2.40)
  (p_2_price : p_2 = 1.60))
: 
  (2 * m / (m/p_1 + m/p_2) = 1.92) := 
by
  sorry

end average_price_l799_799714


namespace shaded_area_is_9_l799_799281

theorem shaded_area_is_9 
    (a b c d e : ℝ)
    (h1 : a = 2)
    (h2 : b = 2)
    (h3 : c = 2)
    (h4 : d = 1)
    (h5 : e = 3)
    (square : ℝ)
    (h_square : ∃ (S J W L : ℝ), 
        J + W + L = square 
        ∧ S = 2 
        ∧ W = 1 
        ∧ L = 3 
        ∧ square = (5 * 5)) : 
    (shaded : ℝ) : shaded = 9 := 
by
  sorry

end shaded_area_is_9_l799_799281


namespace volume_percentage_l799_799143

-- Definitions of the initial conditions
def box_length : ℝ := 8
def box_width : ℝ := 6
def box_height : ℝ := 12
def cube_side : ℝ := 4

-- Definition for the correct answer
def correct_answer : ℝ := 66.67

-- The Lean 4 statement to express and prove the given problem
theorem volume_percentage :
  let box_volume := box_length * box_width * box_height,
      cube_volume := cube_side ^ 3,
      num_cubes := (box_length / cube_side).to_int * (box_width / cube_side).to_int * (box_height / cube_side).to_int,
      cubes_volume := num_cubes * cube_volume,
      volume_percentage := (cubes_volume / box_volume) * 100 in
  volume_percentage = correct_answer :=
by
  sorry

end volume_percentage_l799_799143


namespace final_result_after_operations_l799_799733

theorem final_result_after_operations : 
  let chosen_number := 121 in
  let multiplied_result := chosen_number * 2 in
  let final_result := multiplied_result - 138 in
  final_result = 104 := by
  sorry

end final_result_after_operations_l799_799733


namespace max_digit_d_of_form_7d733e_multiple_of_33_l799_799263

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l799_799263


namespace number_of_integer_solutions_l799_799954

theorem number_of_integer_solutions :
  let ω := Complex.I in
  { p : Int × Int | abs (p.1 * ω + p.2) = 2 }.card = 4 :=
by
  have h1 : { p : Int × Int | abs (p.1 * ω + p.2) = 2 }.card = { (0, 2), (0, -2), (2, 0), (-2, 0) }.card 
  sorry -- steps to transform and prove the sets equivalence
  have h2 : { (0, 2), (0, -2), (2, 0), (-2, 0) }.card = 4 by decide
  exact h1.trans h2

end number_of_integer_solutions_l799_799954


namespace warriors_truth_tellers_l799_799199

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799199


namespace p_implies_q_l799_799819

def p (x : ℝ) := 0 < x ∧ x < 5
def q (x : ℝ) := -5 < x - 2 ∧ x - 2 < 5

theorem p_implies_q (x : ℝ) (h : p x) : q x :=
  by sorry

end p_implies_q_l799_799819


namespace trig_identity_proof_l799_799377

noncomputable def check_trig_identities (α β : ℝ) : Prop :=
  3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2

theorem trig_identity_proof (α β : ℝ) (h : check_trig_identities α β) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end trig_identity_proof_l799_799377


namespace always_has_two_real_roots_find_m_value_l799_799360

-- Define the quadratic equation
def quadratic_eq (x m : ℝ) : ℝ := x^2 - 4 * x - m * (m + 4)

-- 1. Prove that this equation always has two real roots.
theorem always_has_two_real_roots (m : ℝ) : ∃ (x₁ x₂ : ℝ), quadratic_eq x₁ m = 0 ∧ quadratic_eq x₂ m = 0 := by
  have discriminant := 4 * (m + 2)^2
  have h₀ : discriminant ≥ 0 := by
    exact pow_two_nonneg (m + 2)
  sorry

-- 2. If one root of the equation is three times the other root, find the value of m.
theorem find_m_value (x₁ x₂ m : ℝ) (h₀ : quadratic_eq x₁ m = 0) (h₁ : quadratic_eq x₂ m = 0) (h₂ : x₁ = 3 * x₂) : m = -1 ∨ m = -3 := by
  have sum_of_roots := x₁ + x₂ = 4
  have product_of_roots := x₁ * x₂ = -m * (m + 4)
  sorry

end always_has_two_real_roots_find_m_value_l799_799360


namespace smallest_prime_digit_sum_23_l799_799614

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799614


namespace smallest_prime_with_digit_sum_23_l799_799637

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799637


namespace count_even_digits_in_base9_567_number_of_even_digits_in_base9_567_l799_799272

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def decimal_to_base9 (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    let rec digits (n : ℕ) (acc : list ℕ) : list ℕ :=
      if n = 0 then acc else digits (n / 9) ((n % 9) :: acc)
    digits n []

theorem count_even_digits_in_base9_567 :
  (decimal_to_base9 567).filter is_even = [6, 0] :=
by sorry

theorem number_of_even_digits_in_base9_567 :
  (decimal_to_base9 567).filter is_even).length = 2 :=
by sorry

end count_even_digits_in_base9_567_number_of_even_digits_in_base9_567_l799_799272


namespace geometric_sequence_a6_a8_sum_l799_799419

theorem geometric_sequence_a6_a8_sum 
  (a : ℕ → ℕ) (q : ℕ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 1 + a 3 = 5)
  (h2 : a 2 + a 4 = 10) : 
  a 6 + a 8 = 160 := 
sorry

end geometric_sequence_a6_a8_sum_l799_799419


namespace union_sets_l799_799321

def set_A : Set ℝ := {x | x^3 - 3 * x^2 - x + 3 < 0}
def set_B : Set ℝ := {x | |x + 1 / 2| ≥ 1}

theorem union_sets :
  set_A ∪ set_B = ( {x : ℝ | x < -1} ∪ {x : ℝ | x ≥ 1 / 2} ) :=
by
  sorry

end union_sets_l799_799321


namespace assignment_schemes_count_l799_799024

theorem assignment_schemes_count :
  let n := 5
  let k := 3
  (n.perm k) = 60 := 
by {
  sorry
}

end assignment_schemes_count_l799_799024


namespace smallest_prime_with_digit_sum_23_l799_799633

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799633


namespace total_amount_paid_l799_799706

theorem total_amount_paid :
  let pizzas := 3
  let cost_per_pizza := 8
  let total_cost := pizzas * cost_per_pizza
  total_cost = 24 :=
by
  sorry

end total_amount_paid_l799_799706


namespace mother_age_4times_daughter_l799_799541

-- Conditions
def Y := 12
def M := 42

-- Proof statement: Prove that 2 years ago, the mother's age was 4 times Yujeong's age.
theorem mother_age_4times_daughter (X : ℕ) (hY : Y = 12) (hM : M = 42) : (42 - X) = 4 * (12 - X) :=
by
  intros
  sorry

end mother_age_4times_daughter_l799_799541


namespace count_valid_numbers_l799_799497

-- Define the condition that numbers are in the specified range and congruent to 1 modulo 6.
def valid_number (x : ℤ) : Prop :=
  2 ≤ x ∧ x ≤ 2018 ∧ x % 6 = 1

-- Define the set of such valid numbers.
def valid_sequence : set ℤ := {x : ℤ | valid_number x}

-- Define the problem statement to prove the number of elements satisfying the condition.
theorem count_valid_numbers : 
  (set.finite valid_sequence) ∧ (set.card valid_sequence = 336) :=
begin
  sorry
end

end count_valid_numbers_l799_799497


namespace monotonic_increasing_interval_l799_799044

-- Define the function f
def f (x : ℝ) : ℝ := Real.tan (x + Real.pi / 4)

-- The problem statement in Lean would be proving the following theorem:
theorem monotonic_increasing_interval (k : ℤ) : ∀ x,
  (k:ℝ - 3 * Real.pi / 4 < x ∧ x < k:ℝ + Real.pi / 4) ↔
  f(x) = Real.tan (x + Real.pi / 4) :=
sorry

end monotonic_increasing_interval_l799_799044


namespace find_x_l799_799302

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : Nat.coprime 2 3)
  (h3 : (∀ p : ℕ, Prime p → p ∣ x → p = 2 ∨ p = 3))
  (h4 : Nat.count (λ p, Prime p ∧ p ∣ x) = 3)
  : x = 480 ∨ x = 2016 :=
by sorry

end find_x_l799_799302


namespace ratio_of_areas_of_similar_triangles_l799_799020

theorem ratio_of_areas_of_similar_triangles (a b a1 b1 S S1 : ℝ) (α k : ℝ) :
  S = (1/2) * a * b * (Real.sin α) →
  S1 = (1/2) * a1 * b1 * (Real.sin α) →
  a1 = k * a →
  b1 = k * b →
  S1 / S = k^2 := by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_of_similar_triangles_l799_799020


namespace greatest_integer_third_side_of_triangle_l799_799082

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end greatest_integer_third_side_of_triangle_l799_799082


namespace variance_sqrt3Y_plus_1_l799_799872

open ProbabilityTheory

noncomputable def binomial_var (n : ℕ) (p : ℝ) : MeasureSpace ℝ :=
  sorry -- This represents the binomial random variable, which we assume to exist

theorem variance_sqrt3Y_plus_1 :
  ∀ (p : ℝ), 
    (0 ≤ p ∧ p ≤ 1) →
    (P(X ≥ 1) = 5 / 9) →
    D(√3 * Y + 1) = 2
  :=
by
  assume p hp P_X P_Y h,
  sorry


end variance_sqrt3Y_plus_1_l799_799872


namespace integer_solutions_l799_799266

theorem integer_solutions (a b c : ℤ) (h₁ : 1 < a) 
    (h₂ : a < b) (h₃ : b < c) 
    (h₄ : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) 
    ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by sorry

end integer_solutions_l799_799266


namespace number_of_aquariums_l799_799892

theorem number_of_aquariums (total_animals animals_per_aquarium : ℕ) (h1 : total_animals = 40) (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end number_of_aquariums_l799_799892


namespace smallest_prime_with_digit_sum_23_l799_799636

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799636


namespace incorrect_conclusion_l799_799353

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l799_799353


namespace smallest_prime_with_digit_sum_23_l799_799628

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799628


namespace box_cubes_percentage_l799_799148

-- Define the dimensions of the rectangular box
def box_length: ℝ := 8
def box_width: ℝ := 6
def box_height: ℝ := 12

-- Define the side length of the solid cubes
def cube_side: ℝ := 4

-- Define the expected percentage
def expected_percentage: ℝ := 66.67

-- Define the function to calculate the percentage of the volume of the box taken up by the cubes
noncomputable def volume_percentage_of_cubes (L W H side_length: ℝ) : ℝ :=
  let num_cubes_length := (L / side_length).toNat
  let num_cubes_width := (W / side_length).toNat
  let num_cubes_height := (H / side_length).toNat
  let total_num_cubes := num_cubes_length * num_cubes_width * num_cubes_height
  let volume_of_one_cube := side_length ^ 3
  let total_volume_of_cubes := total_num_cubes * volume_of_one_cube
  let volume_of_box := L * W * H
  (total_volume_of_cubes / volume_of_box) * 100

-- The theorem to be proven
theorem box_cubes_percentage:
  volume_percentage_of_cubes box_length box_width box_height cube_side = expected_percentage :=
sorry

end box_cubes_percentage_l799_799148


namespace truthful_warriors_count_l799_799218

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799218


namespace part1_part2_l799_799810

open Nat

-- Part (I)
theorem part1 (a b : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x + b = 0 → x = 2 ∨ x = 3) :
  a + b = 11 :=
by sorry

-- Part (II)
theorem part2 (c : ℝ) (h2 : ∀ x : ℝ, -x^2 + 6 * x + c ≤ 0) :
  c ≤ -9 :=
by sorry

end part1_part2_l799_799810


namespace binary_to_octal_l799_799187

-- Define the binary number to be converted
def binary_num : Fin 128 := 0b1010101 -- Binary representation of the number

-- Prove that its decimal conversion is equal to 85 and its octal conversion is equal to 125
theorem binary_to_octal (b : Fin 128) (h : b = binary_num) : 
  let decimal_val := 85 in 
  let octal_val := 125 in 
  to_Octal (b.toNat) = octal_val := 
by 
  sorry

end binary_to_octal_l799_799187


namespace smallest_prime_with_digit_sum_23_l799_799602

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799602


namespace smallest_prime_with_digit_sum_23_l799_799588

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799588


namespace rate_of_fencing_per_meter_l799_799502

theorem rate_of_fencing_per_meter (area_hectares : ℝ) (cost_rs : ℝ) (rate_per_meter : ℝ)
  (h_area : area_hectares = 13.86) (h_cost : cost_rs = 6202.75) :
  rate_per_meter = (cost_rs / (2 * Math.pi * (sqrt (138600 / Math.pi)))) :=
by
  have area_sq_m := area_hectares * 10000
  have radius := sqrt (area_sq_m / Math.pi)
  have circumference := 2 * Math.pi * radius
  have rate := cost_rs / circumference
  exact calc 
    rate_per_meter = rate : sorry -- Proof here is skipped

end rate_of_fencing_per_meter_l799_799502


namespace range_of_f_l799_799804

noncomputable def f (x : ℝ) : ℝ :=
  (sin x)^3 + 2 * (sin x)^2 + 5 * sin x + 3 * cos x^2 - 12) / (sin x + 2)

theorem range_of_f :
  ∀ x : ℝ, sin x ≠ -2 → -10.5 <= f x ∧ f x <= 4.5 := by
  sorry

end range_of_f_l799_799804


namespace sum_proof_l799_799054

-- Define the context and assumptions
variables (F S T : ℕ)
axiom sum_of_numbers : F + S + T = 264
axiom first_number_twice_second : F = 2 * S
axiom third_number_one_third_first : T = F / 3
axiom second_number_given : S = 72

-- The theorem to prove the sum is 264 given the conditions
theorem sum_proof : F + S + T = 264 :=
by
  -- Given conditions already imply the theorem, the actual proof follows from these
  sorry

end sum_proof_l799_799054


namespace average_of_five_quantities_l799_799996

theorem average_of_five_quantities (a b c d e : ℝ) 
  (h1 : (a + b + c) / 3 = 4) 
  (h2 : (d + e) / 2 = 33) : 
  ((a + b + c + d + e) / 5) = 15.6 := 
sorry

end average_of_five_quantities_l799_799996


namespace rational_points_exist_l799_799963

-- Definitions
def is_rational_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90 ∧
  is_rational A ∧ is_rational B ∧ is_rational C

def is_rational_point (A B C : ℕ) (P : ℕ) : Prop :=
  let angles_formed := [A, B, C] in
  ∀ X, X ∈ angles_formed → is_rational X

-- Theorem Statement
theorem rational_points_exist (A B C : ℕ) : 
  is_rational_triangle A B C → 
  ∃ P₁ P₂ P₃, P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₃ ≠ P₁ ∧ 
               is_rational_point A B C P₁ ∧ 
               is_rational_point A B C P₂ ∧ 
               is_rational_point A B C P₃ :=
by
  sorry

end rational_points_exist_l799_799963


namespace quadrilateral_BFIG_is_rhombus_l799_799998

/-- The quadrilateral BFIG is a rhombus. -/
theorem quadrilateral_BFIG_is_rhombus (A B C E D F G I : Type) 
    [Triangle ABC] 
    (h1 : angle_bisector A B E)
    (h2 : angle_bisector C D E)
    (h3 : intersects_circumcircle B E A C)
    (h4 : intersects_circumcircle D E B C)
    (h5 : DE_intersects_sides F AB)
    (h6 : DE_intersects_sides G BC)
    (h7 : I_is_incenter ABC) :
    is_rhombus BFIG := sorry

end quadrilateral_BFIG_is_rhombus_l799_799998


namespace f_3_neg3div2_l799_799958

noncomputable def f : ℝ → ℝ :=
sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom symm_f : ∀ t : ℝ, f t = f (1 - t)
axiom restriction_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 → f x = -x^2

theorem f_3_neg3div2 :
  f 3 + f (-3/2) = -1/4 :=
sorry

end f_3_neg3div2_l799_799958


namespace max_value_of_d_l799_799255

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end max_value_of_d_l799_799255


namespace coordinates_of_point_with_respect_to_origin_l799_799506

theorem coordinates_of_point_with_respect_to_origin (P : ℝ × ℝ) (h : P = (-2, 4)) : P = (-2, 4) := 
by 
  exact h

end coordinates_of_point_with_respect_to_origin_l799_799506


namespace find_EQ_l799_799067

open Real

noncomputable def Trapezoid_EFGH (EF FG GH HE EQ QF : ℝ) : Prop :=
  EF = 110 ∧
  FG = 60 ∧
  GH = 23 ∧
  HE = 75 ∧
  EQ + QF = EF ∧
  EQ = 250 / 3

theorem find_EQ (EF FG GH HE EQ QF : ℝ) (h : Trapezoid_EFGH EF FG GH HE EQ QF) :
  EQ = 250 / 3 :=
by
  sorry

end find_EQ_l799_799067


namespace truthfulness_count_l799_799226

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799226


namespace trig_identity_proof_l799_799378

noncomputable def check_trig_identities (α β : ℝ) : Prop :=
  3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2

theorem trig_identity_proof (α β : ℝ) (h : check_trig_identities α β) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end trig_identity_proof_l799_799378


namespace sabrina_cookies_l799_799984

theorem sabrina_cookies :
  let S0 : ℕ := 28
  let S1 : ℕ := S0 - 10
  let S2 : ℕ := S1 + 3 * 10
  let S3 : ℕ := S2 - S2 / 3
  let S4 : ℕ := S3 + 16 / 4
  let S5 : ℕ := S4 - S4 / 2
  S5 = 18 := 
by
  -- begin proof here
  sorry

end sabrina_cookies_l799_799984


namespace find_xy_value_l799_799328

-- Given real numbers x and y satisfying the condition
def satisfies_condition (x y : ℝ) : Prop :=
  x / 2 + 2 * y - 2 = Real.log x + Real.log y

-- The goal is to prove that x^y = sqrt 2
theorem find_xy_value (x y : ℝ) (h : satisfies_condition x y) : x^y = Real.sqrt 2 := by
  sorry

end find_xy_value_l799_799328


namespace polar_equation_of_circle_distance_PQ_is_4_l799_799931

def circle_param_eqs (φ : ℝ) : ℝ × ℝ :=
  (2 - 2 * Real.cos φ, 2 * Real.sin φ)

def polar_eq (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.cos θ

def line_eq (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ - π / 6) = 3 * Real.sqrt 3

def ray_eq (θ : ℝ) : Prop :=
  θ = π / 3

def point_O : ℝ × ℝ := (0, 0)

def point_P : ℝ × ℝ := (2, π / 3)

def point_Q : ℝ × ℝ := (6, π / 3)

def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem polar_equation_of_circle :
  ∀ θ ∈ ℝ, ∃ ρ, polar_eq ρ θ :=
by
  sorry

theorem distance_PQ_is_4 :
  distance point_P point_Q = 4 :=
by
  sorry

end polar_equation_of_circle_distance_PQ_is_4_l799_799931


namespace midpoint_ad_l799_799409

variable {α : Type*} [euclidean_geometry α] {A B C D E F G O : α}

noncomputable def circumcircle (A B C : α) : set α := sorry -- Definition for the circumcircle
noncomputable def midpoint (A B : α) : α := sorry -- Definition for the midpoint

-- Conditions
axiom bc_is_diameter (hbc: segment B C) : is_diameter B C (circumcircle A B C)
axiom D_on_arc (hD: α) : on_circumcircle D (circumcircle A B C) ∧ opposite_side A B C D
axiom perpendiculuar_de_bc (hE : α) : E ∈ perpendicular_line D B ∧ segment E C
axiom perpendicular_df_ba (hF : α) : F ∈ perpendicular_line D B ∧ segment F A
axiom intersection_ef_ad (hG : α) : G ∈ intersection (line_through E F) (line_through A D)

theorem midpoint_ad : midpoint A D = G :=
sorry

end midpoint_ad_l799_799409


namespace double_covers_count_l799_799386

-- Define what it means for a collection of sets to be a double cover
def is_double_cover {X : Type*} (A : Finset (Finset X)) :=
  (∀ x : X, ∃! (B : Finset X), B ∈ A ∧ x ∈ B ∧ ∃! (C : Finset X), C ∈ A ∧ x ∈ C ∧ C ≠ B)

-- Define the number of elements in the set X
variable {X : Type*} [Fintype X]
variable {n : ℕ} (hX : Fintype.card X = n)

-- Define the statement about the number of double covers for k = 3
theorem double_covers_count (A : Fin 3 → Finset X) (hA : is_double_cover (Finset.univ.image A)) :
  (3^n - 3) / 6 = double_cover_count_3 :=
sorry

end double_covers_count_l799_799386


namespace find_f_3_8_l799_799770

-- Define the function f within the context of the given problem
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the given conditions as hypotheses
axiom f_zero : f 0 = 0
axiom f_monotone : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → f(x) ≤ f(y)
axiom f_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f(1 - x) = 1 - f(x)
axiom f_scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f(x / 4) = f(x) / 3

-- The statement we want to prove
theorem find_f_3_8 : f (3 / 8) = 2 / 9 :=
by 
  sorry

end find_f_3_8_l799_799770


namespace volume_percentage_correct_l799_799138

-- The dimensions of the rectangular box
def length := 8
def width := 6
def height := 12

-- The side length of the cubes
def cube_edge := 4

-- Calculate the volume of the box
def box_volume : ℕ := length * width * height

-- Calculate how many cubes fit in the box
def cubes_in_length := length / cube_edge
def cubes_in_width := width / cube_edge
def cubes_in_height := height / cube_edge

-- Calculate the volume of the part filled with cubes
def cubes_volume := (cubes_in_length * cube_edge) * (cubes_in_width * cube_edge) * (cubes_in_height * cube_edge)

-- Calculate the ratio of the filled volume to the box volume
def volume_ratio := cubes_volume / box_volume

-- Convert the ratio to a percentage
noncomputable def volume_percentage := (volume_ratio : ℝ) * 100

-- Statement of the problem
theorem volume_percentage_correct : volume_percentage = 66.67 := by
  -- Proof is not required, so we use 'sorry'
  sorry

end volume_percentage_correct_l799_799138


namespace ratio_of_children_to_adults_l799_799499

theorem ratio_of_children_to_adults (total_people children : ℕ) (h1 : total_people = 120) (h2 : children = 80) :
  (children:ℚ) / (total_people - children) = 2 / 1 :=
by
  have adults := total_people - children
  have h_adults : adults = 40 := by sorry  -- Here, you would prove that adults = 40 based on the conditions
  have ratio := (children:ℚ) / adults
  have h_ratio : ratio = 2 / 1 := by sorry -- Here, you would simplify the ratio 80 / 40 to 2 / 1
  exact h_ratio

end ratio_of_children_to_adults_l799_799499


namespace initial_population_correct_l799_799712

-- Definitions based on conditions
def initial_population (P : ℝ) := P
def population_after_bombardment (P : ℝ) := 0.9 * P
def population_after_fear (P : ℝ) := 0.8 * (population_after_bombardment P)
def final_population := 3240

-- Theorem statement
theorem initial_population_correct (P : ℝ) (h : population_after_fear P = final_population) :
  initial_population P = 4500 :=
sorry

end initial_population_correct_l799_799712


namespace arithmetic_seq_a7_a8_l799_799406

theorem arithmetic_seq_a7_a8 (a : ℕ → ℤ) (d : ℤ) (h₁ : a 1 + a 2 = 4) (h₂ : d = 2) :
  a 7 + a 8 = 28 := by
  sorry

end arithmetic_seq_a7_a8_l799_799406


namespace find_x_l799_799293

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end find_x_l799_799293


namespace semicircle_in_rhombus_l799_799035

noncomputable def rhombus_area (r : ℝ) (angle_CBA : ℝ) : ℝ :=
  let angle_CBA_rad := real.pi / 3 -- converting 60 degrees to radians
  let diagonal_AC := 2 * r * real.csc (angle_CBA_rad / 2)
  (diagonal_AC ^ 2) / (2 * real.sin angle_CBA_rad)

theorem semicircle_in_rhombus (r : ℝ) (angle_CBA : ℝ) (h : angle_CBA = 60)
  (area_rhombus : ℝ) (a b : ℕ) (ha : r = 10) (hb : ∃ n : ℕ, b.prime ∧ area_rhombus = a * real.sqrt b)
  (hab : area_rhombus = 150 * real.sqrt 3) :
  a * b + a + b = 603 :=
begin
  rw [hb, hab],
  sorry
end

end semicircle_in_rhombus_l799_799035


namespace polynomial_degree_l799_799091

-- Define the polynomial of interest
def P (x : ℝ) : ℝ := x^7 * (x + 1/x) * (1 + 3/x + 5/x^2)

-- Statement to prove the degree of the polynomial is 8
theorem polynomial_degree : ∀ x : ℝ, degree (P x) = 8 :=
by sorry

end polynomial_degree_l799_799091


namespace regression_line_values_l799_799052

theorem regression_line_values : 
  let b := 9.4 in
  let x := 6 in
  let y := 65.5 in
  let x̄ := 3.5 in
  let ȳ := 28.5 + m / 4 in
  ∃ (a m : ℝ), ȳ = b * x̄ + a ∧ (b * x + a = 65.5) ∧ (a = 9.1) ∧ (m = 54) :=
by
  sorry

end regression_line_values_l799_799052


namespace smallest_prime_with_digit_sum_23_l799_799666

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799666


namespace smallest_prime_with_digit_sum_23_l799_799564

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799564


namespace radius_circle_D_eq_five_l799_799765

-- Definitions for circles with given radii and tangency conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

noncomputable def circle_C : Circle := ⟨(0, 0), 5⟩
noncomputable def circle_D (rD : ℝ) : Circle := ⟨(4 * rD, 0), 4 * rD⟩
noncomputable def circle_E (rE : ℝ) : Circle := ⟨(5 - rE, rE * 5), rE⟩

-- Prove that the radius of circle D is 5
theorem radius_circle_D_eq_five (rE : ℝ) (rD : ℝ) : circle_D rE = circle_C → rD = 5 := by
  sorry

end radius_circle_D_eq_five_l799_799765


namespace smallest_largest_value_sum_l799_799957

noncomputable def smallest_largest_sum (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) : ℝ :=
let m := -1/3, M := 3 in m + M

theorem smallest_largest_value_sum :
  ∀ (x y z : ℝ), 
  (x + y + z = 5) ∧ (x^2 + y^2 + z^2 = 11) → smallest_largest_sum x y z = 8/3 :=
by
  intros x y z h
  unfold smallest_largest_sum
  sorry

end smallest_largest_value_sum_l799_799957


namespace peanut_cluster_percentage_l799_799758

def chocolates :=
  let typeA := 5
  let typeB := 8
  let typeC := Int.round (typeA * 1.5)
  let typeD := 2 * (typeA + typeB)
  let typeE := Int.round (typeC * 1.2)
  let typeF := typeA + 6
  let typeG := typeB + 6
  let typeH := typeC + 6
  let typeI := 7
  let typeJ := Int.round (1.5 * (typeI + typeF))
  let totalChocolates := 150
  let totalNonPeanutClusters := typeA + typeB + typeC + typeD + typeE + typeF + typeG + typeH + typeI + typeJ
  let peanutClusters := totalChocolates - totalNonPeanutClusters
  let percentage := (peanutClusters : ℚ) / totalChocolates * 100
  percentage

theorem peanut_cluster_percentage :
  chocolates = 13.33 :=
by
  -- This is where the proof would go.
  sorry

end peanut_cluster_percentage_l799_799758


namespace smallest_projected_polygon_sides_l799_799151

structure RegularDodecahedron (V : Type*) :=
  (vertices : Finset V)
  (faces : Finset (Finset V))
  (is_regular_dodecahedron : ∀ f ∈ faces, is_pentagon f ∧ faces.card = 12)

def is_pentagon {V : Type*} (f : Finset V) : Prop :=
  f.card = 5

def orthogonal_projection (V : Type*) (P : RegularDodecahedron V) :=
  Finset V  -- Assume this function projects vertices to some vertices in the plane

def projected_polygon (V : Type*) (P : RegularDodecahedron V) :=
  {n : ℕ // ∃ (proj : Finset V), proj ∈ orthogonal_projection V P ∧ proj.card = n}

theorem smallest_projected_polygon_sides {V : Type*} (P : RegularDodecahedron V) :
  ∃ (n : ℕ), (∃ proj, proj ∈ orthogonal_projection V P ∧ proj.card = n) ∧ n = 6 :=
by
  -- Proof to be filled in
  sorry

end smallest_projected_polygon_sides_l799_799151


namespace x_intercepts_count_l799_799783

def sin_x_intercepts (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  let lower_bound := Int.floor (2000 / Real.pi)
  let upper_bound := Int.floor (20000 / Real.pi)
  upper_bound - lower_bound

theorem x_intercepts_count : sin_x_intercepts (λ x => Real.sin (1 / x)) 0.00005 0.0005 = 5729 := by
  sorry

end x_intercepts_count_l799_799783


namespace apples_handed_out_l799_799504

theorem apples_handed_out 
  (initial_apples : ℕ)
  (pies_made : ℕ)
  (apples_per_pie : ℕ)
  (H : initial_apples = 50)
  (H1 : pies_made = 9)
  (H2 : apples_per_pie = 5) :
  initial_apples - (pies_made * apples_per_pie) = 5 := 
by
  sorry

end apples_handed_out_l799_799504


namespace difference_of_roots_leq_common_difference_l799_799314

theorem difference_of_roots_leq_common_difference 
  (f g : ℝ[X]) (d : ℝ)
  (hf : f.monic) (hg : g.monic)
  (hf_roots : ∀ r1 r2 : ℝ, f.roots = [r1, r2] → abs (r1 - r2) = d)
  (hg_roots : ∀ r1 r2 : ℝ, g.roots = [r1, r2] → abs (r1 - r2) = d)
  (hfg_roots : ∀ r1 r2 : ℝ, (f + g).roots = [r1, r2] → r1 ≠ r2) :
  ∀ r1 r2 : ℝ, (f + g).roots = [r1, r2] → abs (r1 - r2) ≤ d :=
sorry

end difference_of_roots_leq_common_difference_l799_799314


namespace circle_line_distance_l799_799710

theorem circle_line_distance (a : ℝ) :
  (∃ a : ℝ, ∀ x y: ℝ, (x + 1)^2 + (y - 3)^2 = 4 ∧ x + ay + 1 = 0 → 
  a = real.sqrt 2 / 4 ∨ a = -real.sqrt 2 / 4) := sorry

end circle_line_distance_l799_799710


namespace value_of_f_log2_9_l799_799337

noncomputable def f : ℝ → ℝ
| x => if x < 1 then 2^x else f (x - 1)

theorem value_of_f_log2_9 : f (Real.log 9 / Real.log 2) = 9 / 8 := by
  -- proof skipped
  sorry

end value_of_f_log2_9_l799_799337


namespace number_of_ways_to_assign_roles_l799_799729

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 7
  let male_roles := 3
  let female_roles := 3
  let neutral_roles := 2
  let ways_male_roles := men * (men - 1) * (men - 2)
  let ways_female_roles := women * (women - 1) * (women - 2)
  let ways_neutral_roles := (men + women - male_roles - female_roles) * (men + women - male_roles - female_roles - 1)
  ways_male_roles * ways_female_roles * ways_neutral_roles = 1058400 := 
by
  sorry

end number_of_ways_to_assign_roles_l799_799729


namespace rationalize_denominator_l799_799479

theorem rationalize_denominator :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) := by
  sorry

end rationalize_denominator_l799_799479


namespace odd_square_mod_eight_l799_799477

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l799_799477


namespace possible_values_of_A_l799_799062

theorem possible_values_of_A :
  ∃ (A : ℕ), (A ≤ 4 ∧ A < 10) ∧ A = 5 :=
sorry

end possible_values_of_A_l799_799062


namespace circle_tangent_to_x_axis_at_origin_l799_799908

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h : ∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → 
      (x = 0 → y = 0)) :
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l799_799908


namespace range_of_k_inequality_l799_799914

theorem range_of_k_inequality
  (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x ^ 2 + k * x - 3 / 2 < 0) ↔ k ∈ set.Icc (-12 : ℝ) 0 :=
by
  sorry

end range_of_k_inequality_l799_799914


namespace part_a_part_b_l799_799112

-- Definition of moving points and their constraints in Part (a)
noncomputable def moving_points (P Q : ℂ) (t : ℝ) (z1 z2 w1 w2 : ℂ) : Prop :=
let A := z1 + w1 * complex.exp (complex.I * t) in
let B := z2 + w2 * complex.exp (complex.I * t) in
let C := z1 * (1 - complex.exp (complex.I * (π / 3))) + z2 * complex.exp (complex.I * (π / 3)) + 
          complex.exp (complex.I * (π / 3 + t)) * (w2 - w1 + w1 * complex.exp (-complex.I * (π / 3))) in
abs (A - B) = abs (B - C) ∧ abs (C - A) = abs (A - B)

theorem part_a (P Q : ℂ) (t : ℝ) (z1 z2 w1 w2 : ℂ) (h : moving_points P Q t z1 z2 w1 w2) :
  ∃ r : ℝ, ∀ t', let C' := 
    z1 * (1 - complex.exp (complex.I * (π / 3))) + 
    z2 * complex.exp (complex.I * (π / 3)) + 
    complex.exp (complex.I * (π / 3 + t')) * (w2 - w1 + w1 * complex.exp (-complex.I * (π / 3))) in
    abs (C' - P) = r := sorry


-- Definition of point P and its constraints in Part (b)
noncomputable def valid_point (A B C P : ℂ) (side : ℝ) : Prop :=
abs (A - P) = 2 ∧ abs (B - P) = 3 ∧ abs (A - B) = side ∧ abs (B - C) = side ∧ abs (C - A) = side

theorem part_b (A B C P : ℂ) (side : ℝ) (h : valid_point A B C P side) :
  ∃ (C_max : ℝ), C_max = 5 :=
begin
  use 5,
  sorry
end

end part_a_part_b_l799_799112


namespace smallest_prime_with_digit_sum_23_l799_799656

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799656


namespace minimum_value_of_g_l799_799282

-- Define the function g(x)
def g (x : ℝ) : ℝ := (5 * x^2 + 20 * x + 25) / (8 * (1 + x))

-- State the theorem and prove it
theorem minimum_value_of_g : ∃ x ≥ 0, g x = (65 / 16) :=
by
  sorry -- proof goes here

end minimum_value_of_g_l799_799282


namespace carl_max_value_l799_799176

-- Definitions based on problem conditions.
def value_of_six_pound_rock : ℕ := 20
def weight_of_six_pound_rock : ℕ := 6
def value_of_three_pound_rock : ℕ := 9
def weight_of_three_pound_rock : ℕ := 3
def value_of_two_pound_rock : ℕ := 4
def weight_of_two_pound_rock : ℕ := 2
def max_weight_carl_can_carry : ℕ := 24

/-- Proves that Carl can carry rocks worth maximum 80 dollars given the conditions. -/
theorem carl_max_value : ∃ (n m k : ℕ),
    n * weight_of_six_pound_rock + m * weight_of_three_pound_rock + k * weight_of_two_pound_rock ≤ max_weight_carl_can_carry ∧
    n * value_of_six_pound_rock + m * value_of_three_pound_rock + k * value_of_two_pound_rock = 80 :=
by
  sorry

end carl_max_value_l799_799176


namespace smallest_prime_with_digit_sum_23_l799_799578

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799578


namespace parking_cost_savings_l799_799132

theorem parking_cost_savings
  (weekly_rate : ℕ := 10)
  (monthly_rate : ℕ := 24)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12) :
  (weekly_rate * weeks_in_year) - (monthly_rate * months_in_year) = 232 :=
by
  sorry

end parking_cost_savings_l799_799132


namespace smallest_prime_with_digit_sum_23_l799_799645

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799645


namespace large_cube_edge_length_l799_799537

theorem large_cube_edge_length : 
  ∃ L : ℝ, 
    (∀ (N : ℝ), abs (N - 1000) ≤ 0.0000000000002 → 
    ∀ (a : ℝ), a = 0.1 → 
    N * (a^3) = (L^3)) → L = 1 :=
begin
  sorry
end

end large_cube_edge_length_l799_799537


namespace smallest_prime_with_digit_sum_23_l799_799638

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799638


namespace compare_neg_numbers_l799_799767

theorem compare_neg_numbers : - 0.6 > - (2 / 3) := 
by sorry

end compare_neg_numbers_l799_799767


namespace seeds_in_big_garden_is_correct_l799_799247

def total_seeds : ℕ := 41
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 4

def seeds_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden
def seeds_in_big_garden : ℕ := total_seeds - seeds_in_small_gardens

theorem seeds_in_big_garden_is_correct : seeds_in_big_garden = 29 := by
  -- proof goes here
  sorry

end seeds_in_big_garden_is_correct_l799_799247


namespace smallest_prime_with_digit_sum_23_l799_799562

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799562


namespace proof_f0_proof_f2_proof_k_l799_799459

namespace Proof

variable (f : ℝ → ℝ)

-- Conditions
def condition1 := ∀ x y : ℝ, f (x + y) = f x + f y
def condition2 := f (-1) = 2
def condition3 := ∀ x > 0, f x < 0
def condition4 := ∀ t : ℝ, f (t^2 + 3 * t) + f (t + k) ≤ 4 → k ≥ 2

-- Proof goals
theorem proof_f0 (h1 : condition1 f) (h2 : condition2 f) (h3 : condition3 f) : f 0 = 0 :=
by {
  sorry
}

theorem proof_f2 (h1 : condition1 f) (h2 : condition2 f) (h3 : condition3 f) : f 2 = -4 :=
by {
  sorry
}

theorem proof_k (h1 : condition1 f) (h2 : condition2 f) (h3 : condition3 f) (h4 : condition4 f) : k ≥ 2 :=
by {
  sorry
}

end Proof

end proof_f0_proof_f2_proof_k_l799_799459


namespace directrix_of_parabola_l799_799509

theorem directrix_of_parabola (x y : ℝ) : (y^2 = 8*x) → (x = -2) :=
by
  sorry

end directrix_of_parabola_l799_799509


namespace max_diagonals_convex_ngon_l799_799559

theorem max_diagonals_convex_ngon (n : ℕ) (h : n ≥ 3) :
  (∃ d : ℕ, d = if n ≥ 5 then n else n - 2 ∧
    ∀ (diagonals : Fin n → Fin n → Bool), (∃ v, diagonals v (v + (if n ≥ 5 then n else n - 2)) 
    ∧ (∀ a b : Fin n,  (diagonals a b → (∃ c,  c ∈ set.range(to_nat)
    ∧ c = a ∨ c = b ∨ (a ≠ b ∧ c ∉ {a, b}) )))) sorry
 
end max_diagonals_convex_ngon_l799_799559


namespace smallest_prime_with_digit_sum_23_l799_799598

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799598


namespace annual_production_minimum_year_wine_production_ratio_after_years_l799_799116

-- Definitions of beer and wine production
def beer_production (n : ℕ) : ℕ := if n = 0 then 16000 else 16000 / 2^n
def wine_production (n : ℕ) : ℕ := if n = 0 then 1000 else 500 * 2^n

-- Total production definitions
def total_beer_production (n : ℕ) : ℕ := 32000 * (2^n - 1) / 2^n
def total_wine_production (n : ℕ) : ℕ := 1000 * (2^n - 1)

-- Proof problem statements
theorem annual_production_minimum_year : argmin (λ n, beer_production (n + 1) + wine_production (n + 1)) 2019 :=
sorry

theorem wine_production_ratio_after_years : ∀ n >= 6, 
(∑ i in range (n + 1), wine_production (i + 1)) ≥ 2 / 3 * (∑ i in range (n + 1), beer_production (i + 1) + wine_production (i + 1)) :=
sorry

end annual_production_minimum_year_wine_production_ratio_after_years_l799_799116


namespace basketball_player_scores_l799_799114

theorem basketball_player_scores :
  ∃ P : finset ℕ, (∀ (x y z : ℕ), x + y + z = 7 → 3 * x + 2 * y + 4 * z ∈ P) ∧ P.card = 14 :=
by
  -- Proof steps can be skipped
  sorry

end basketball_player_scores_l799_799114


namespace min_abs_y1_minus_4y2_l799_799042

-- Definitions based on conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0)

noncomputable def equation_of_line (k y : ℝ) : ℝ := k * y + 1

-- The Lean theorem statement
theorem min_abs_y1_minus_4y2 {x1 y1 x2 y2 : ℝ} (H1 : parabola x1 y1) (H2 : parabola x2 y2)
    (A_in_first_quadrant : 0 < x1 ∧ 0 < y1)
    (line_through_focus : ∃ k : ℝ, x1 = equation_of_line k y1 ∧ x2 = equation_of_line k y2)
    : |y1 - 4 * y2| = 8 :=
sorry

end min_abs_y1_minus_4y2_l799_799042


namespace tangents_circles_concentric_l799_799985

theorem tangents_circles_concentric (O o : Point) (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0)
(points : ∃ A B a b P Q R S : Point,
  tangent_point (O, r₁) A B P Q ∧
  tangent_point (O, r₁) a b R S ∧
  tangent_point (o, r₂) A B P Q ∧
  tangent_point (o, r₂) a b R S) :
  ∃ C : Point,
    C ∈ line O o ∧
    ∃ C₁ C₂ C₃ : Circle,
      (A ∈ C₁ ∧ B ∈ C₁ ∧ a ∈ C₂ ∧ b ∈ C₂ ∧
       P ∈ C₃ ∧ Q ∈ C₃ ∧ R ∈ C₃ ∧ S ∈ C₃) ∧
      (center C₁ = C ∧ center C₂ = C ∧ center C₃ = C) :=
sorry

end tangents_circles_concentric_l799_799985


namespace smallest_prime_with_digit_sum_23_l799_799626

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799626


namespace limit_calculation_l799_799173

open Real
open Complex

theorem limit_calculation :
  ∃ L : ℝ, 
  (tendsto (λ x, (2 - 3^(arctan (sqrt x))^2) ^ (2 / sin x)) (nhds 0) (nhds L)) ∧ L = 1 / 9 :=
sorry

end limit_calculation_l799_799173


namespace bogatyrs_truthful_count_l799_799244

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799244


namespace range_of_x_l799_799090

theorem range_of_x (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 :=
by
  sorry

end range_of_x_l799_799090


namespace probability_sum_8_l799_799070

namespace ProbabilityProof

def faces := {1, 2, 3, 4, 5, 6}

def validRolls : List (ℤ × ℤ) :=
  [(5, 3), (6, 2)]

def totalRolls : List (ℤ × ℤ) :=
  List.product faces.toList faces.toList

noncomputable def validOutcomes : ℤ :=
  validRolls.length

noncomputable def totalOutcomes : ℤ :=
  totalRolls.filter (λ (r, b) => r > b).length

theorem probability_sum_8 : validOutcomes / totalOutcomes = 1 / 18 := by
  sorry

end ProbabilityProof

end probability_sum_8_l799_799070


namespace largest_angle_of_triangle_l799_799448

variable {v : ℝ}
hypothesis hv : v > 0.5

theorem largest_angle_of_triangle :
  let a := Real.sqrt (2 * v + 3)
  let b := Real.sqrt (2 * v + 7)
  let c := 2 * Real.sqrt (v + 1)
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃ C : ℝ, C = Real.arccos (6 / (2 * Real.sqrt ((2 * v + 3) * (2 * v + 7)))) :=
by
  sorry

end largest_angle_of_triangle_l799_799448


namespace segment_length_3sqrt5_l799_799291

noncomputable def line_equation (x : ℝ) : ℝ := 2 * x - 4

def curve_equation (x y : ℝ) : Prop := y^2 = 4 * x

def segment_length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem segment_length_3sqrt5 :
  let A := (1 : ℝ, -2 : ℝ),
      B := (4 : ℝ, 4 : ℝ) in
  segment_length A B = 3 * real.sqrt 5 :=
by
  sorry

end segment_length_3sqrt5_l799_799291


namespace truthful_warriors_count_l799_799214

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799214


namespace medians_perpendicular_ineq_l799_799978

theorem medians_perpendicular_ineq (A B C : ℝ) (triangle : ¬colinear A B C)
 (medians_perpendicular : ∃ M N : ℝ, is_median A B C M ∧ is_median A C B N ∧ M ⊥ N) : 
   ∃ B C : ℝ, cot B + cot C ≥ 2 / 3 := by
  sorry

end medians_perpendicular_ineq_l799_799978


namespace A_superset_B_l799_799961

open Set

variable (N : Set ℕ)
def A : Set ℕ := {x | ∃ n ∈ N, x = 2 * n}
def B : Set ℕ := {x | ∃ n ∈ N, x = 4 * n}

theorem A_superset_B : A N ⊇ B N :=
by
  -- Proof to be written
  sorry

end A_superset_B_l799_799961


namespace sufficient_not_necessary_condition_l799_799449

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (hx : x = 2)
    (ha : a = (x, 1)) (hb : b = (4, x)) : 
    (∃ k : ℝ, a = (k * b.1, k * b.2)) ∧ (¬ (∀ k : ℝ, a = (k * b.1, k * b.2))) :=
by 
  sorry

end sufficient_not_necessary_condition_l799_799449


namespace cone_surface_area_and_volume_l799_799716

theorem cone_surface_area_and_volume (h : ℝ) (theta : ℝ) (F : ℝ) (V : ℝ) 
  (h_gt_zero : h = 12) (theta_deg : theta = 100.8) 
  (surface_area : F = 56 * Real.pi) (volume : V = 49 * Real.pi) :
  F = 56 * Real.pi ∧ V = 49 * Real.pi :=
by 
  have : h = 12 := by rw h_gt_zero
  have : theta = 100.8 := by rw theta_deg
  have : F = 56 * Real.pi := by rw surface_area
  have : V = 49 * Real.pi := by rw volume
  exact ⟨surface_area, volume⟩
  sorry

end cone_surface_area_and_volume_l799_799716


namespace binom_10_0_equals_1_l799_799182

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem to prove that binom 10 0 = 1
theorem binom_10_0_equals_1 :
  binom 10 0 = 1 := by
  sorry

end binom_10_0_equals_1_l799_799182


namespace manny_has_more_10_bills_than_mandy_l799_799000

theorem manny_has_more_10_bills_than_mandy :
  let mandy_bills_20 := 3
  let manny_bills_50 := 2
  let mandy_total_money := 20 * mandy_bills_20
  let manny_total_money := 50 * manny_bills_50
  let mandy_10_bills := mandy_total_money / 10
  let manny_10_bills := manny_total_money / 10
  mandy_10_bills < manny_10_bills →
  manny_10_bills - mandy_10_bills = 4 := sorry

end manny_has_more_10_bills_than_mandy_l799_799000


namespace part_a_part_b_part_c_l799_799755

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem part_a : ∑ k in Finset.range 6, (2^k) * (binomial_coeff 5 k) = 243 :=
by
  sorry

theorem part_b (n : ℕ) (h : n ≥ 1) : ∑ k in Finset.range (n + 1), (-1)^k * (binomial_coeff n k) = 0 :=
by
  sorry

theorem part_c (n : ℕ) : ∑ k in Finset.range (n + 1), (binomial_coeff n k) = 2^n :=
by
  sorry

end part_a_part_b_part_c_l799_799755


namespace truthfulness_count_l799_799227

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799227


namespace sufficient_but_not_necessary_necessary_but_not_sufficient_l799_799286

def M (x : ℝ) : Prop := (x + 3) * (x - 5) > 0
def P (x : ℝ) (a : ℝ) : Prop := x^2 + (a - 8)*x - 8*a ≤ 0
def I : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x, M x ∧ P x a ↔ x ∈ I) → a = 0 :=
sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, (M x ∧ P x a → x ∈ I) ∧ (∀ x, x ∈ I → M x ∧ P x a)) → a ≤ 3 :=
sorry

end sufficient_but_not_necessary_necessary_but_not_sufficient_l799_799286


namespace chess_tournament_participants_l799_799388

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 := by
  sorry

end chess_tournament_participants_l799_799388


namespace driver_a_driven_more_distance_l799_799103

-- Definitions based on conditions
def initial_distance : ℕ := 787
def speed_a : ℕ := 90
def speed_b : ℕ := 80
def start_difference : ℕ := 1

-- Statement of the problem
theorem driver_a_driven_more_distance :
  let distance_a := speed_a * (start_difference + (initial_distance - speed_a) / (speed_a + speed_b))
  let distance_b := speed_b * ((initial_distance - speed_a) / (speed_a + speed_b))
  distance_a - distance_b = 131 := by
sorry

end driver_a_driven_more_distance_l799_799103


namespace ratio_spaghetti_to_manicotti_l799_799707

-- Definitions of the given conditions
def total_students : ℕ := 800
def spaghetti_preferred : ℕ := 320
def manicotti_preferred : ℕ := 160

-- The theorem statement
theorem ratio_spaghetti_to_manicotti : spaghetti_preferred / manicotti_preferred = 2 :=
by sorry

end ratio_spaghetti_to_manicotti_l799_799707


namespace number_of_valid_mappings_l799_799880

def M := { -1, 0, 1 }
def N := { 2, 3, 4, 5 }

def condition (f : ℤ → ℤ) (x : ℤ) : Prop := 
  (x + f x + x * f x) % 2 = 1

def mappings := {f : ℤ → ℤ // ∀ x ∈ M, condition f x}

theorem number_of_valid_mappings : 
  finset.card {f // (∀ x ∈ M, condition f x) ∧ (∀ x ∉ M, f x = 0)} = 24 := 
sorry

end number_of_valid_mappings_l799_799880


namespace range_of_b_l799_799840

theorem range_of_b (a b : ℝ) (f g : ℝ → ℝ) (e : ℝ)
  (h₀ : 0 < e - 1)
  (h₁ : ∀ x, g(x) = real.exp x)
  (h₂ : ∃ x₀ ∈ Ioo 0 (e-1), ∀ x, f(x) = a * real.log (x + 1) - x - b)
  (h₃ : ∀ x₀ y₀, y₀ = f(x₀) → (y₀ / x₀) = -(1 / e))
  (h₄ : ∀ m, m = 1 → (∃ l, ∀ x, l(x) = real.exp x * (x - m))) :
  b ∈ Ioo 0 (1 - (1 / e)) :=
sorry

end range_of_b_l799_799840


namespace words_to_learn_l799_799809

theorem words_to_learn (total_words : ℕ) (required_percentage : ℕ) (recall_correctly : ℕ) : ℕ :=
  let percentage := required_percentage / 100;
  let required_words := percentage * total_words in
  required_words

def minimum_words_to_learn (total_words : ℕ) (required_percentage : ℕ) : ℕ :=
  words_to_learn total_words required_percentage sorry

example : minimum_words_to_learn 800 90 = 720 := 
by sorry

end words_to_learn_l799_799809


namespace find_x_l799_799300

open Nat

theorem find_x (n : ℕ) (x : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (factors x).nodup)
  (h3 : (factors x).length = 3)
  (h4 : 3 ∈ factors x) :
  x = 480 ∨ x = 2016 := 
sorry

end find_x_l799_799300


namespace fruit_distribution_l799_799096

theorem fruit_distribution : 
  let n := 17
  let k := 5
  combinatorics.choose (n - k + k - 1) (k - 1) = 1820 := 
by
  let n := 17
  let k := 5
  let remaining := n - k 
  have calculation : combinatorics.choose (remaining + k - 1) (k - 1) = 1820 := sorry
  exact calculation

end fruit_distribution_l799_799096


namespace roots_of_equation_l799_799785

theorem roots_of_equation (x : ℝ) (hx : x > 0) :
  (3 * Real.sqrt x + 3 * x ^ (-1 / 2) = 8) ↔
  (x = (8 + 2 * Real.sqrt 7) / 6 ^ 2) ∨ (x = (8 - 2 * Real.sqrt 7) / 6 ^ 2) :=
by
  sorry

end roots_of_equation_l799_799785


namespace new_years_day_2007_monday_l799_799890

theorem new_years_day_2007_monday (h_53_sundays : 53 = 52 + 1) (not_leap_year : ∀ (n : ℕ), 2006 % 4 ≠ 0 ∨ (2006 % 100 = 0 ∧ 2006 % 400 ≠ 0)) :
  "New Year's Day in 2007 is Monday" :=
by
  sorry

end new_years_day_2007_monday_l799_799890


namespace truthful_warriors_count_l799_799221

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799221


namespace volume_percentage_correct_l799_799135

-- The dimensions of the rectangular box
def length := 8
def width := 6
def height := 12

-- The side length of the cubes
def cube_edge := 4

-- Calculate the volume of the box
def box_volume : ℕ := length * width * height

-- Calculate how many cubes fit in the box
def cubes_in_length := length / cube_edge
def cubes_in_width := width / cube_edge
def cubes_in_height := height / cube_edge

-- Calculate the volume of the part filled with cubes
def cubes_volume := (cubes_in_length * cube_edge) * (cubes_in_width * cube_edge) * (cubes_in_height * cube_edge)

-- Calculate the ratio of the filled volume to the box volume
def volume_ratio := cubes_volume / box_volume

-- Convert the ratio to a percentage
noncomputable def volume_percentage := (volume_ratio : ℝ) * 100

-- Statement of the problem
theorem volume_percentage_correct : volume_percentage = 66.67 := by
  -- Proof is not required, so we use 'sorry'
  sorry

end volume_percentage_correct_l799_799135


namespace minnow_distribution_l799_799003

def pondA_minnows : ℕ := 60
def pondA_percentages : (ℚ × ℚ × ℚ × ℚ) := (0.45, 0.225, 0.175, 1 - (0.45 + 0.225 + 0.175))

def pondB_minnows : ℕ := 75
def pondB_percentages : (ℚ × ℚ × ℚ × ℚ) := (0.35, 0.275, 0.20, 1 - (0.35 + 0.275 + 0.20))

def pondC_minnows : ℕ := 80
def pondC_percentages : (ℚ × ℚ × ℚ × ℚ) := (0.375, 0.25, 0.15, 1 - (0.375 + 0.25 + 0.15))

theorem minnow_distribution :
  let red_total := pondA_minnows * pondA_percentages.1 + pondB_minnows * pondB_percentages.1 + pondC_minnows * pondC_percentages.1 in
  let green_total := pondA_minnows * pondA_percentages.2 + pondB_minnows * pondB_percentages.2 + pondC_minnows * pondC_percentages.2 in
  let blue_total := pondA_minnows * pondA_percentages.3 + pondB_minnows * pondB_percentages.3 + pondC_minnows * pondC_percentages.3 in
  let white_total := pondA_minnows * pondA_percentages.4 + pondB_minnows * pondB_percentages.4 + pondC_minnows * pondC_percentages.4 in
  red_total.round = 83 ∧ green_total.round = 54 ∧ blue_total.round = 38 ∧ white_total.round = 40 :=
by
  sorry

end minnow_distribution_l799_799003


namespace problem_statement_l799_799458

noncomputable def f (x b : ℝ) : ℝ :=
if h : 1 ≤ x ∧ x ≤ b then -x + 2 * b else b

def g (x a b : ℝ) : ℝ := f x b + a * x

noncomputable def h (a b : ℝ) : ℝ :=
let g' := g (1 : ℝ) a b,
    max_g := max (max (g 1 a b) (g b a b)) (g 3 a b),
    min_g := min (min (g 1 a b) (g b a b)) (g 3 a b) in
max_g - min_g

noncomputable def d (b : ℝ) : ℝ := min {h a b | a ∈ set.univ}

theorem problem_statement :
  ∀ b, 1 < b ∧ b < 3 → d b = 1 / 2 := 
sorry

end problem_statement_l799_799458


namespace value_of_x_l799_799383

theorem value_of_x :
  ∀ (x : ℕ), 
    x = 225 + 2 * 15 * 9 + 81 → 
    x = 576 := 
by
  intro x h
  sorry

end value_of_x_l799_799383


namespace intersect_condition_l799_799774

noncomputable def intersects_at_four_points (A : ℝ) (b : ℝ) : Prop :=
  let f := λ x, A * x^2
  let g := λ x, 4 * A * x - x^2 + b
  ∀ x1 x2 x3 x4 : ℝ, f x1 = g x1 ∧ f x2 = g x2 ∧ f x3 = g x3 ∧ f x4 = g x4 → x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x1

theorem intersect_condition (A : ℝ) (hA : 0 < A) (b : ℝ) :
  intersects_at_four_points A b ↔ b < (4 * A^2 - 2 * A + 0.25) / A^2 :=
sorry

end intersect_condition_l799_799774


namespace no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l799_799524

theorem no_real_roots_eq_xsq_abs_x_plus_1_eq_0 :
  ¬ ∃ x : ℝ, x^2 + abs x + 1 = 0 :=
by
  sorry

end no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l799_799524


namespace smallest_prime_with_digit_sum_23_l799_799634

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799634


namespace quadratic_function_properties_l799_799871

variable (f : ℝ → ℝ)
variables (x : ℝ) (A_x A_y B_x B_y C_x C_y : ℝ) (L : set ℝ)
variables [L_eq   : L = {x | x ≤ -1 ∨ x ≥ 3}]
variables [A_coords : A_x = -1 ∧ A_y = 0]
variables [B_coords : B_x = 3 ∧ B_y = 0]
variables [C_coords : C_x = 1 ∧ C_y = -8]
variables [range_interval : ∀ x, 0 ≤ x ∧ x ≤ 3 → f(x) ≥ -8 ∧ f(x) ≤ 0]

theorem quadratic_function_properties :
  (∃ a b c : ℝ, f = λ x, 2 * x^2 - 4 * x - 6) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f(x) = -8 ∨ f(x) = 0) ∧
  (∀ x, f(x) ≥ 0 ↔ x ∈ L) :=
  sorry

end quadratic_function_properties_l799_799871


namespace smallest_prime_with_digit_sum_23_l799_799567

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799567


namespace students_attending_Harvard_l799_799416

theorem students_attending_Harvard (total_applicants : ℕ) (perc_accepted : ℝ) (perc_attending : ℝ)
    (h1 : total_applicants = 20000)
    (h2 : perc_accepted = 0.05)
    (h3 : perc_attending = 0.9) :
    total_applicants * perc_accepted * perc_attending = 900 := 
by
    sorry

end students_attending_Harvard_l799_799416


namespace find_factors_of_224_l799_799540

theorem find_factors_of_224 : ∃ (a b c : ℕ), a * b * c = 224 ∧ c = 2 * a ∧ a ≠ b ∧ b ≠ c :=
by
  -- Prove that the factors meeting the criteria exist
  sorry

end find_factors_of_224_l799_799540


namespace tangent_line_at_zero_number_of_zeros_of_f_range_of_a_l799_799863

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x * Real.sin x - Real.cos x

-- Part (1)
theorem tangent_line_at_zero : tangent_line_at (f 0) f 0 = -1 := 
sorry

-- Part (2)
theorem number_of_zeros_of_f : number_of_zeros f = 2 := 
sorry

-- Part (3)
theorem range_of_a (a : ℝ) : (∀ x ∈ Icc 0 (Real.pi / 2), a * x^2 - x * Real.sin x - Real.cos x + a ≥ 0) ↔ 1 ≤ a := 
sorry

end tangent_line_at_zero_number_of_zeros_of_f_range_of_a_l799_799863


namespace find_x_l799_799299

open Nat

theorem find_x (n : ℕ) (x : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (factors x).nodup)
  (h3 : (factors x).length = 3)
  (h4 : 3 ∈ factors x) :
  x = 480 ∨ x = 2016 := 
sorry

end find_x_l799_799299


namespace smallest_prime_with_digit_sum_23_l799_799595

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799595


namespace y_intercept_of_line_l799_799795

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  rw [hx] at h
  simp at h
  exact h

end y_intercept_of_line_l799_799795


namespace mason_father_age_l799_799004

theorem mason_father_age
  (Mason_age : ℕ) 
  (Sydney_age : ℕ) 
  (Father_age : ℕ)
  (h1 : Mason_age = 20)
  (h2 : Sydney_age = 3 * Mason_age)
  (h3 : Father_age = Sydney_age + 6) :
  Father_age = 66 :=
by
  sorry

end mason_father_age_l799_799004


namespace smallest_prime_with_digit_sum_23_l799_799593

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799593


namespace largest_multiple_of_15_with_digits_8_or_0_and_5_digits_l799_799447

noncomputable def n : ℕ := 88080

theorem largest_multiple_of_15_with_digits_8_or_0_and_5_digits :
  (∀ k : ℕ, (k < 10^5) → (k % 15 = 0) → (∀ d : ℕ, d ∈ List.of_fn (fin 5 (λ i, (k / 10^i % 10))) → (d = 8 ∨ d = 0)) → k ≤ n) ∧ (n / 15 = 5872) :=
by
  sorry

end largest_multiple_of_15_with_digits_8_or_0_and_5_digits_l799_799447


namespace binom_10_0_eq_1_l799_799184

theorem binom_10_0_eq_1 :
  (Nat.choose 10 0) = 1 :=
by
  sorry

end binom_10_0_eq_1_l799_799184


namespace parabola_intersection_length_l799_799722

theorem parabola_intersection_length (x1 x2 : ℝ) (y1 y2 : ℝ) (p : ℝ) (h_parabola : p = 2)
  (h_focus : ∃ f : ℝ, f = (1, 0) ∧ (∀ x1 y1 x2 y2 : ℝ, y1^2 = 4*x1 ∧ y2^2 = 4*x2))
  (h_intersection : x1 + x2 = 6) : 
  |P Q| = 8 := by
  sorry

end parabola_intersection_length_l799_799722


namespace pq_eq_major_axis_l799_799110

open Locale Classical Real

noncomputable def ellipse := {
  a : ℝ,  -- semi-major axis
  b : ℝ   -- semi-minor axis
}

variables (a b c : ℝ) (O F P Q : ℝ × ℝ) (CD : set (ℝ × ℝ)) (is_parallel : CD ∩ (tangent P) ≠ ∅) 
  (is_on_major_axis : O = (0, 0)) (F_focus : F = (c, 0)) (P_on_ellipse : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1) 
  (Q_PF_intersect : ∃ Q : ℝ × ℝ, Q ∈ CD ∧ Q ∈ line P F)

theorem pq_eq_major_axis : dist P Q = a := by
  sorry

end pq_eq_major_axis_l799_799110


namespace plane_split_into_8_regions_l799_799186

-- Define the conditions as separate lines in the plane.
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := y = (1 / 2) * x
def line3 (x y : ℝ) : Prop := x = y

-- Define a theorem stating that these lines together split the plane into 8 regions.
theorem plane_split_into_8_regions :
  (∀ (x y : ℝ), line1 x y ∨ line2 x y ∨ line3 x y) →
  -- The plane is split into exactly 8 regions by these lines
  ∃ (regions : ℕ), regions = 8 :=
sorry

end plane_split_into_8_regions_l799_799186


namespace average_last_three_l799_799997

noncomputable def average (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem average_last_three (l : List ℝ) (h₁ : l.length = 7) (h₂ : average l = 62) 
  (h₃ : average (l.take 4) = 58) :
  average (l.drop 4) = 202 / 3 := 
by 
  sorry

end average_last_three_l799_799997


namespace smallest_prime_with_digit_sum_23_l799_799650

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799650


namespace warriors_truth_tellers_l799_799200

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799200


namespace min_value_ineq_l799_799866

noncomputable def function_y (a : ℝ) (x : ℝ) : ℝ := a^(1-x)

theorem min_value_ineq (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_ineq_l799_799866


namespace tangent_line_to_parabola_l799_799513

noncomputable def parabola (x : ℝ) : ℝ := 4 * x^2

def derivative_parabola (x : ℝ) : ℝ := 8 * x

def tangent_line_eq (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_to_parabola (x : ℝ) (hx : x = 1) (hy : parabola x = 4) :
    tangent_line_eq 1 4 :=
by 
  -- Sorry to skip the detailed proof, but it should follow the steps outlined in the solution.
  sorry

end tangent_line_to_parabola_l799_799513


namespace number_of_solutions_of_line_passing_through_vertex_of_parabola_l799_799283

theorem number_of_solutions_of_line_passing_through_vertex_of_parabola :
  ∃ (a_values : Finset ℝ), a_values.card = 2 ∧
  ∀ a ∈ a_values, let y_parabola := 0^2 + 4 * a^2 in let y_line := 2 * 0 + a in y_parabola = y_line :=
by
  sorry

end number_of_solutions_of_line_passing_through_vertex_of_parabola_l799_799283


namespace total_votes_cast_l799_799117

theorem total_votes_cast (V : ℕ) (C R : ℕ) 
  (hC : C = 30 * V / 100) 
  (hR1 : R = C + 4000) 
  (hR2 : R = 70 * V / 100) : 
  V = 10000 :=
by
  sorry

end total_votes_cast_l799_799117


namespace smallest_prime_with_digit_sum_23_l799_799655

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799655


namespace value_of_expr_l799_799442

noncomputable def verify_inequality (x a b c : ℝ) : Prop :=
  (x - a) * (x - b) / (x - c) ≥ 0

theorem value_of_expr (a b c : ℝ) :
  (∀ x : ℝ, verify_inequality x a b c ↔ (x < -6 ∨ abs (x - 30) ≤ 2)) →
  a < b →
  a = 28 →
  b = 32 →
  c = -6 →
  a + 2 * b + 3 * c = 74 := by
  sorry

end value_of_expr_l799_799442


namespace smallest_prime_with_digit_sum_23_l799_799661

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799661


namespace smallest_prime_with_digit_sum_23_l799_799582

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799582


namespace binom_10_0_eq_1_l799_799177

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem stating the binomial coefficient we need to prove
theorem binom_10_0_eq_1 : binom 10 0 = 1 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end binom_10_0_eq_1_l799_799177


namespace martin_bought_more_hens_l799_799969

-- Definitions corresponding to conditions
def initial_hens : ℕ := 10  -- Martin initially has 10 hens
def eggs_in_10_days : ℕ := 80  -- 10 hens lay 80 eggs in 10 days
def total_eggs_in_15_days : ℕ := 300  -- All hens lay 300 eggs in 15 days

-- Prove that Martin bought 15 more hens
theorem martin_bought_more_hens (initial_hens eggs_in_10_days total_eggs_in_15_days : ℕ) : 
  initial_hens = 10 → 
  eggs_in_10_days = 80 → 
  total_eggs_in_15_days = 300 → 
  ∃ hens_bought : ℕ, hens_bought = 15 :=
begin
  sorry
end

end martin_bought_more_hens_l799_799969


namespace smallest_prime_with_digit_sum_23_l799_799664

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799664


namespace greatest_integer_third_side_l799_799076

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end greatest_integer_third_side_l799_799076


namespace planes_intersect_single_line_two_altitudes_intersect_then_others_three_altitudes_intersect_then_fourth_l799_799696

-- Define the structures and necessary hypotheses
structure TrihedralAngle where
  A B C : Point
  SA SB SC: Line
  face_opposite_each_edge: ∀ l : Line, l ∈ {SA, SB, SC} → Plane

-- Define the theorem statement for question (a)
theorem planes_intersect_single_line (T : TrihedralAngle) :
  ∃ L : Line, ∀ (P : Plane), P ∈ {T.face_opposite_each_edge T.SA, T.face_opposite_each_edge T.SB, T.face_opposite_each_edge T.SC} → ∃ l : Line, l = L :=
by sorry

-- Define the structure for a tetrahedral
structure Tetrahedron where
  A B C D : Point
  altitude_lines : ∀ p : Point, Line

-- Define the theorem statement for question (b)
theorem two_altitudes_intersect_then_others (T : Tetrahedron) (h_intersect: ∃ O : Point, Line (T.altitude_lines T.A) ∩ Line (T.altitude_lines T.B) = {O}) :
  ∃ O' : Point, Line (T.altitude_lines T.C) ∩ Line (T.altitude_lines T.D) = {O'} :=
by sorry

-- Define the theorem statement for question (c)
theorem three_altitudes_intersect_then_fourth (T : Tetrahedron) (h_intersect: ∃ O : Point, ∀ P : Point, P ∈ {T.A, T.B, T.C} → O ∈ Line (T.altitude_lines P)) :
  ∃ O' : Point, ∀ Q : Point, Q ∈ {T.A, T.B, T.C, T.D} → O' ∈ Line (T.altitude_lines Q) :=
by sorry

end planes_intersect_single_line_two_altitudes_intersect_then_others_three_altitudes_intersect_then_fourth_l799_799696


namespace equilateral_triangle_DE_FG_sum_eq_one_l799_799068

theorem equilateral_triangle_DE_FG_sum_eq_one
  (ABC : Type) 
  [field ABC]
  (A B C D F E G : ABC) 
  (hAB : dist A B = 1)
  (hD_on_AB : D ∈ segment A B) 
  (hF_on_AB : F ∈ segment A B) 
  (hE_on_AC : E ∈ segment A C) 
  (hG_on_AC : G ∈ segment A C) 
  (hDE_parallel_BC : parallel (line D E) (line B C)) 
  (hFG_parallel_BC : parallel (line F G) (line B C)) 
  (hADF_eq_AEG_area : area (triangle A D F) = area (triangle A E G)) 
  (hDFGE_FBCG_perimeter_eq : perimeter (trapezoid D F G E) = perimeter (trapezoid F B C G)) :
  dist D E + dist F G = 1 :=
sorry

end equilateral_triangle_DE_FG_sum_eq_one_l799_799068


namespace condition_A_iff_condition_B_l799_799920

theorem condition_A_iff_condition_B (A B : ℝ) (h₁ : A < B) (h₂ : cos A ^ 2 > cos B ^ 2) :
  A < B ↔ cos A ^ 2 > cos B ^ 2 :=
begin
  split,
  { intro hA,
    // Proof that A < B implies cos A ^ 2 > cos B ^ 2,
    sorry
  },
  { intro hcos,
    // Proof that cos A ^ 2 > cos B ^ 2 implies A < B,
    sorry
  }
end

end condition_A_iff_condition_B_l799_799920


namespace largest_three_digit_multiple_of_12_with_digit_sum_24_l799_799557

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 12 = 0 ∧ (n.digits 10).sum = 24 ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 12 = 0 ∧ (m.digits 10).sum = 24 → m ≤ n) ∧ n = 888 :=
by {
  sorry -- Proof to be filled in
}

#eval largest_three_digit_multiple_of_12_with_digit_sum_24 -- Should output: ⊤ (True)

end largest_three_digit_multiple_of_12_with_digit_sum_24_l799_799557


namespace probability_same_color_opposite_foot_l799_799490

def total_shoes := 28

def black_pairs := 7
def brown_pairs := 4
def gray_pairs := 2
def red_pair := 1

def total_pairs := black_pairs + brown_pairs + gray_pairs + red_pair

theorem probability_same_color_opposite_foot : 
  (7 + 4 + 2 + 1) * 2 = total_shoes →
  (14 / 28 * (7 / 27) + 8 / 28 * (4 / 27) + 4 / 28 * (2 / 27) + 2 / 28 * (1 / 27)) = (20 / 63) :=
by
  sorry

end probability_same_color_opposite_foot_l799_799490


namespace trapezoid_angles_l799_799937

theorem trapezoid_angles
  (A B C D K : Type)
  [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup D]
  (h_AD_parallel_BC : @Parallel A B C D)
  (h_BK_DK : (BK / KD) = 1 / 2)
  (h_AC_AD_BC : AC = AD - 2 * BC)
  (h_angle_CAD_alpha : ∠ CAD = α) :
  let AKC_angle1 := (π - α) / 2,
      AKC_angle2 := arctan((sin α) / (2 + cos α)),
      AKC_angle3 := (π + α) / 2 - arctan((sin α) / (2 + cos α))
  in (∠ AKC = (AKC_angle1, AKC_angle2, AKC_angle3)) :=
sorry

end trapezoid_angles_l799_799937


namespace expected_score_l799_799126

/-- 
A game has the following rules:
1. There are 5 red balls and 5 yellow balls in a pocket.
2. You draw 5 balls at a time.
3. Scoring rules:
    - All 5 balls same color: 100 points
    - 4 balls same color, 1 different: 50 points
    - Otherwise: 0 points

Prove that the expected score when Zhang draws balls once is 75/7.
-/
theorem expected_score :
  let total_ways := (nat.choose 10 5)
  let prob_100 := (2 / total_ways : ℚ)
  let prob_50 := (25 / total_ways : ℚ)
  let prob_0 := 1 - prob_100 - prob_50
  let e_X := 100 * prob_100 + 50 * prob_50 + 0 * prob_0
  e_X = 75 / 7 := 
by {
  let total_ways := (nat.choose 10 5)
  let prob_100 := (2 / total_ways : ℚ)
  let prob_50 := (25 / total_ways : ℚ)
  let prob_0 := 1 - prob_100 - prob_50
  let e_X := 100 * prob_100 + 50 * prob_50 + 0 * prob_0
  
  -- Calculations
  have h1 : total_ways = 252 := by norm_num,
  have h_prob_100 : prob_100 = 1 / 126 := by norm_num[prob_100, h1],
  have h_prob_50 : prob_50 = 25 / 126 := by norm_num[prob_50, h1],
  have h_prob_0 : prob_0 = 100 / 126 := by norm_num[prob_0, h_prob_100, h_prob_50],

  -- Expected value calculation
  change e_X = 75 / 7,
  change 100 * (1 / 126 : ℚ) + 50 * (25 / 126 : ℚ) with 75 / 7,
  norm_num,
  exact h1
}

end expected_score_l799_799126


namespace total_chickens_l799_799752

theorem total_chickens (coops chickens_per_coop : ℕ) (h1 : coops = 9) (h2 : chickens_per_coop = 60) :
  coops * chickens_per_coop = 540 := by
  sorry

end total_chickens_l799_799752


namespace total_water_l799_799534

variable initial_water : ℚ := 7.75
variable added_water : ℚ := 7

theorem total_water : initial_water + added_water = 14.75 :=
by
  sorry

end total_water_l799_799534


namespace sum_abs_values_of_roots_l799_799805

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 4 * x^3 - 6 * x^2 + 20 * x - 10

-- Define the sum of the absolute values of the roots
def sum_abs_roots : ℝ := 4 + 2 * Real.sqrt 10

-- Lean 4 statement of the problem
theorem sum_abs_values_of_roots :
  (∀ (r : ℝ), IsRoot poly r → ∃ (p : Polynomial ℝ), polynomial.coeff p 0 = sum_abs_roots) :=
sorry

end sum_abs_values_of_roots_l799_799805


namespace product_less_than_one_tenth_l799_799474

-- Define the product A as the product of sequential fractions.
def A : ℝ := (List.prod (List.map (fun n => (2*n-1)/(2*n : ℝ)) (List.range 50.succ )))

theorem product_less_than_one_tenth : A < 1/10 :=
by
  -- We'll prove this statement later.
  sorry

end product_less_than_one_tenth_l799_799474


namespace area_of_HA_HB_HC_le_area_of_ABC_l799_799435

theorem area_of_HA_HB_HC_le_area_of_ABC (A B C H H_A H_B H_C : Type)
  [acute_triangle A B C]
  (H_is_orthocenter : orthocenter A B C H)
  (H_A_is_second_intersection : second_intersection_circumcircle_with_altitude A B C H H_A)
  (H_B_is_second_intersection : second_intersection_circumcircle_with_altitude A B C H H_B)
  (H_C_is_second_intersection : second_intersection_circumcircle_with_altitude A B C H H_C) :
  area (triangle H_A H_B H_C) ≤ area (triangle A B C) :=
by
  sorry

end area_of_HA_HB_HC_le_area_of_ABC_l799_799435


namespace prefer_X_perc_is_zero_l799_799010

-- Definitions based on conditions
def total_employees := 200
def reloc_to_X_perc := 0.30
def reloc_to_Y_perc := 0.70
def prefer_Y_perc := 0.40
def max_happy_employees := 140

-- Problem statement to prove
theorem prefer_X_perc_is_zero : 
  ∀ (total_employees reloc_to_X_perc reloc_to_Y_perc prefer_Y_perc max_happy_employees : ℕ), 
  total_employees = 200 →
  reloc_to_X_perc = 0.30 →
  reloc_to_Y_perc = 0.70 →
  prefer_Y_perc = 0.40 →
  max_happy_employees = 140 →
  (0% : nat) = 0 :=
by sorry

end prefer_X_perc_is_zero_l799_799010


namespace greatest_distance_between_vertices_l799_799157

noncomputable def inner_square_perimeter : ℝ := 24
noncomputable def outer_square_perimeter : ℝ := 32

noncomputable def inner_square_side_length : ℝ := inner_square_perimeter / 4
noncomputable def outer_square_side_length : ℝ := outer_square_perimeter / 4

theorem greatest_distance_between_vertices :
  (∃ s₁ s₂ : ℝ, s₁ = inner_square_side_length ∧ s₂ = outer_square_side_length ∧
    ∀ d : ℝ, d = real.sqrt (real.pow (s₂ / 2) 2 + real.pow s₂ 2) → 
    d = 4 * real.sqrt 5) := 
begin
  use inner_square_side_length,
  use outer_square_side_length,
  split,
  { sorry },
  split,
  { sorry },
  intros d hd,
  sorry
end

end greatest_distance_between_vertices_l799_799157


namespace solve_log_eq_l799_799028

open Real

theorem solve_log_eq (x : ℝ) (h : x ≠ -3) :
  log (x^2 + 1) / log 10 - 2 * log (x + 3) / log 10 + log 2 / log 10 = 0 ↔ x = -1 ∨ x = 7 :=
by
  suffices : log (x^2 + 1) - 2 * log (x + 3) + log 2 = 0 ↔ x = -1 ∨ x = 7
  sorry
  exact this

end solve_log_eq_l799_799028


namespace point_in_second_quadrant_l799_799837

-- Definitions based on the problem conditions
def z1 : ℂ := complex.I
def z2 : ℂ := 1 + complex.I
def z : ℂ := z1 * z2

-- Definition of the second quadrant
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The theorem to be proved
theorem point_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end point_in_second_quadrant_l799_799837


namespace fn_conjecture_l799_799842

theorem fn_conjecture (f : ℕ → ℝ → ℝ) (x : ℝ) (h_pos : x > 0) :
  (f 1 x = x / (Real.sqrt (1 + x^2))) →
  (∀ n, f (n + 1) x = f 1 (f n x)) →
  (∀ n, f n x = x / (Real.sqrt (1 + n * x ^ 2))) := by
  sorry

end fn_conjecture_l799_799842


namespace total_earnings_from_selling_working_games_l799_799006

-- Conditions definition
def total_games : ℕ := 16
def broken_games : ℕ := 8
def working_games : ℕ := total_games - broken_games
def game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

-- Proof problem statement
theorem total_earnings_from_selling_working_games : List.sum game_prices = 68 := by
  sorry

end total_earnings_from_selling_working_games_l799_799006


namespace difference_between_largest_and_smallest_quarters_l799_799430

noncomputable def coin_collection : Prop :=
  ∃ (n d q : ℕ), 
    (n + d + q = 150) ∧ 
    (5 * n + 10 * d + 25 * q = 2000) ∧ 
    (forall (q1 q2 : ℕ), (n + d + q1 = 150) ∧ (5 * n + 10 * d + 25 * q1 = 2000) → 
     (n + d + q2 = 150) ∧ (5 * n + 10 * d + 25 * q2 = 2000) → 
     (q1 = q2))

theorem difference_between_largest_and_smallest_quarters : coin_collection :=
  sorry

end difference_between_largest_and_smallest_quarters_l799_799430


namespace solution_set_of_inequality_l799_799835

noncomputable def is_solution_set (f : ℝ → ℝ) : set ℝ :=
  {x | x > 1 ∨ x < -1 ∨ x = 0}

theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (hf_even : ∀ x, f x = f (-x)) 
  (hf_deriv : ∀ x, 2 * f x + x * (deriv f x) > 6)
  (hf_at_1 : f 1 = 2)
  : {x | x^2 * f x > 3 * x^2 - 1} = is_solution_set f :=
by
  sorry

end solution_set_of_inequality_l799_799835


namespace line_and_max_distance_l799_799850

noncomputable def intersection_point (a b c d : ℝ) : ℝ × ℝ :=
  ( (d - b) / (a - b), (a * d - b * c) / (a - b) )

def line_equation (m x₀ y₀ : ℝ) : ℝ → ℝ := 
  λ x => m * (x - x₀) + y₀

def parallel_line (x₀ y₀ m : ℝ) : ℝ × ℝ × ℝ :=
  (m, 1, -(m * x₀ + y₀))

theorem line_and_max_distance :
  ∃ P : ℝ × ℝ, 
    (P = intersection_point 2 1 (-8) 1 (-2) 1) ∧
  ∃ l₁ l₂ : ℝ × ℝ × ℝ, 
    (l₁ = (4, -3, -6)) ∧
    (l₂ = (3, 2, -13)) :=
by
  sorry

end line_and_max_distance_l799_799850


namespace midpoint_probability_sum_l799_799438

def S : Finset (ℕ × ℕ × ℕ) :=
  Finset.univ.filter (fun (x : ℕ × ℕ × ℕ) => x.1 ≤ 2 ∧ x.2.1 ≤ 3 ∧ x.2.2 ≤ 4)

def valid_midpoint (p1 p2 : ℕ × ℕ × ℕ) : Prop :=
  ((p1.1 + p2.1) % 2 = 0) ∧ ((p1.2.1 + p2.2.1) % 2 = 0) ∧ ((p1.2.2 + p2.2.2) % 2 = 0)

def valid_pairs : Finset ((ℕ × ℕ × ℕ) × (ℕ × ℕ × ℕ)) :=
  (S.product S).filter (fun (pp : (ℕ × ℕ × ℕ) × (ℕ × ℕ × ℕ)) => pp.1 ≠ pp.2 ∧ valid_midpoint pp.1 pp.2)

def probability_valid_midpoint : ℚ :=
  (valid_pairs.card : ℚ) / ((S.card * (S.card - 1) / 2) : ℚ)

theorem midpoint_probability_sum :
  probability_valid_midpoint = 23 / 177 ∧ 23 + 177 = 200 :=
by
  sorry  -- Proof to be filled in

end midpoint_probability_sum_l799_799438


namespace smallest_prime_with_digit_sum_23_l799_799597

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799597


namespace bird_count_l799_799532

def initial_birds : ℕ := 12
def new_birds : ℕ := 8
def total_birds : ℕ := initial_birds + new_birds

theorem bird_count : total_birds = 20 := by
  sorry

end bird_count_l799_799532


namespace smallest_prime_digit_sum_23_l799_799615

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799615


namespace fiona_cleaning_time_l799_799464

theorem fiona_cleaning_time (total_time : ℕ) (lilly_ratio : ℚ) (to_minutes : ℕ → ℕ → ℕ) 
  (lilly_fiona_cleaning_time : total_time = 8) 
  (lilly_cleaning_ratio : lilly_ratio = 1/4) 
  (convert_to_minutes : to_minutes = λ h m, h * m) : 
  ∃ fiona_cleaning_time : ℕ, to_minutes (total_time - total_time * lilly_ratio) 60 = 360 :=
by {
  sorry
}

end fiona_cleaning_time_l799_799464


namespace min_largest_value_in_set_l799_799375

theorem min_largest_value_in_set (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : (8:ℚ) / 19 * a * b ≤ (a - 1) * a / 2): a ≥ 13 :=
by
  sorry

end min_largest_value_in_set_l799_799375


namespace concurrency_of_lines_l799_799426

open EuclideanGeometry

-- Define the main problem given the conditions and the statement to be proven
theorem concurrency_of_lines
  (ABC : Triangle)
  (I : Point)
  (A1 B1 C1 K L : Point)
  (incircle_tangent : is_tangent_incircle ABC I A1 B1 C1)
  (circumcircle_O1 : is_circumcircle (Triangle.mk B C1 B1) K)
  (circumcircle_O2 : is_circumcircle (Triangle.mk C B1 C1) L)
  (K_on_BC : on_line_segment BC K)
  (L_on_BC : on_line_segment BC L) :
  concurrent (Line.mk C1 L) (Line.mk B1 K) (Line.mk A1 I) :=
begin
  sorry
end

end concurrency_of_lines_l799_799426


namespace max_positive_integer_value_l799_799325

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n: ℕ, ∃ q: ℝ, a (n + 1) = a n * q

theorem max_positive_integer_value
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 = 4)
  (h4 : a 1 + a 2 + a 3 = 14) : 
  ∃ n, n ≤ 4 ∧ a n * a (n+1) * a (n+2) > 1 / 9 :=
sorry

end max_positive_integer_value_l799_799325


namespace break_even_price_l799_799031

def initial_investment : ℝ := 10410
def cost_per_game : ℝ := 2.65
def break_even_games : ℝ := 600
def total_cost := initial_investment + (cost_per_game * break_even_games)
def selling_price_per_game := total_cost / break_even_games

theorem break_even_price : selling_price_per_game = 20 :=
by
  sorry

end break_even_price_l799_799031


namespace geom_seq_a6_value_l799_799827

variable {α : Type _} [LinearOrderedField α]

theorem geom_seq_a6_value (a : ℕ → α) (q : α) 
(h_geom : ∀ n, a (n + 1) = a n * q)
(h_cond : a 4 + a 8 = π) : 
a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end geom_seq_a6_value_l799_799827


namespace monotonic_f_iff_l799_799344

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2 * x ^ 2 + 10 else (3 - a) * 3 ^ x

theorem monotonic_f_iff (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 1 ≤ a ∧ a < 3 :=
by
  sorry

end monotonic_f_iff_l799_799344


namespace find_value_at_log2_5_l799_799848

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom periodic_function : ∀ x : ℝ, f (x + 2) = f (x)
axiom on_domain : ∀ x : ℝ, -1 < x ∧ x < 0 → f (x) = 2^x

theorem find_value_at_log2_5 : f (Real.log2 5) = -4/5 := 
by
  sorry

end find_value_at_log2_5_l799_799848


namespace distinct_L_values_l799_799950

def complete_graph (n : ℕ) : Type := sorry -- Definition of the complete graph

def edge_color : Type := sorry -- Types of edges: red, blue, white

def triangle (u v w : Type) : Type := sorry -- Definition of a triangle

def L (u v : Type) (V : set Type) (triangle_has_two_red_sides : triangle u v w → Prop) : set Type :=
  {u, v} ∪ {w | w ∈ V ∧ triangle_has_two_red_sides (u, v, w)}

theorem distinct_L_values (G : complete_graph 2015) (edge_coloring : edge_color) 
  (V : set (vertex G)) (triangle_has_two_red_sides : ∀ (u v w : vertex G), Prop) :
  card {L u v V triangle_has_two_red_sides | (u, v) ∈ edges G} ≥ 120 :=
  sorry

end distinct_L_values_l799_799950


namespace coefficient_x_in_expansion_l799_799505

theorem coefficient_x_in_expansion :
  let p := (x^2 + 3 * x + 2)^5
  ∃ c : ℤ, (c * x + _) = p ∧ c = 240 := 
begin
  sorry
end

end coefficient_x_in_expansion_l799_799505


namespace polyhedron_same_number_edges_l799_799732

theorem polyhedron_same_number_edges (n : ℕ) (V E : ℕ) (a : ℕ → ℕ) (F := 7 * n) (M : ℕ)
  (Euler_formula : V - E + F = 2) :
  ∃ k : ℕ, (3 ≤ k ∧ k ≤ M) ∧ (a k) ≥ n+1 :=
by
  sorry

end polyhedron_same_number_edges_l799_799732


namespace eval_sqrt_fraction_expr_l799_799252

def sqrt_fraction_expr : ℝ := real.sqrt ((9 ^ 9 + 3 ^ 12 : ℝ) / (9 ^ 5 + 3 ^ 13 : ℝ))

theorem eval_sqrt_fraction_expr : sqrt_fraction_expr = 15.3 := 
by sorry

end eval_sqrt_fraction_expr_l799_799252


namespace train_length_l799_799159

noncomputable def length_of_train (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (1000 / 3600)
  speed_m_s * time_s

theorem train_length (speed_km_hr : ℝ) (time_s : ℝ) :
  speed_km_hr = 70 → time_s = 12.857142857142858 → length_of_train speed_km_hr time_s = 250 := 
by
  intros hs ht
  have hs' : length_of_train speed_km_hr time_s = 250, sorry
  exact hs'

end train_length_l799_799159


namespace distance_home_gym_l799_799550

theorem distance_home_gym 
  (v_WangLei v_ElderSister : ℕ)  -- speeds in meters per minute
  (d_meeting : ℕ)                -- distance in meters from the gym to the meeting point
  (t_gym : ℕ)                    -- time in minutes for the older sister to the gym
  (speed_diff : v_ElderSister = v_WangLei + 20)  -- speed difference
  (t_gym_reached : d_meeting / 2 = (25 * (v_WangLei + 20)) - d_meeting): 
  v_WangLei * t_gym = 1500 :=
by
  sorry

end distance_home_gym_l799_799550


namespace arc_length_of_sector_l799_799501

theorem arc_length_of_sector 
  (R : ℝ) (θ : ℝ) (hR : R = Real.pi) (hθ : θ = 2 * Real.pi / 3) : 
  (R * θ = 2 * Real.pi^2 / 3) := 
by
  rw [hR, hθ]
  sorry

end arc_length_of_sector_l799_799501


namespace number_of_truth_tellers_is_twelve_l799_799230
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799230


namespace fiona_cleaning_time_l799_799462

theorem fiona_cleaning_time (total_time : ℝ) (lilly_fraction : ℝ) (fiona_fraction : ℝ) :
  total_time = 8 ∧ lilly_fraction = 1/4 ∧ fiona_fraction = 3/4 → 
  let fiona_time := (total_time * fiona_fraction) * 60 in
  fiona_time = 360 :=
by
  intros h
  let h1 := h.1
  let h2 := h.2.1
  let h3 := h.2.2
  have total_time_correct : total_time = 8 := h1
  have lilly_fraction_correct : lilly_fraction = 1 / 4 := h2
  have fiona_fraction_correct : fiona_fraction = 3 / 4 := h3
  let fiona_time := (total_time * fiona_fraction) * 60
  have fiona_time_eq : fiona_time = (8 * (3/4)) * 60 := by { rw [total_time_correct, fiona_fraction_correct] }
  have fiona_time_eq_simplified : fiona_time = 6 * 60 := by { rw [← mul_assoc, mul_comm (8 : ℝ), ← div_mul_cancel (8 * (3 : ℝ)) (4 : ℝ), div_self (4 : ℝ)] }
  have fiona_time_eq_final : fiona_time = 360 := by { rw [mul_comm, mul_assoc, mul_comm 6 60] }
  exact fiona_time_eq_final

end fiona_cleaning_time_l799_799462


namespace smallest_prime_digit_sum_23_l799_799621

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799621


namespace geometryville_high_schools_l799_799934

theorem geometryville_high_schools :
  ∃ (n : ℕ), 
  (∀ (score : ℕ), score ∈ (list.range (4 * n + 1)) ∧ list.pairwise (<) (list.range (4 * n + 1))) ∧
  (∃ (andrearank : ℕ), andrearank = (2 * n) / 2 + 1) ∧
  (∃ (bethrank carlarank : ℕ), bethrank = 48 ∧ carlarank = 75 ∧ bethrank < andrearank ∧ andrearank < carlarank) ∧
  n = 23 :=
by
  -- This is where the formal proof would go
  sorry

end geometryville_high_schools_l799_799934


namespace power_function_value_l799_799868

-- Given conditions
def f : ℝ → ℝ := fun x => x^(1 / 3)

theorem power_function_value :
  f (Real.log 5 / (Real.log 2 * 8) + Real.log 160 / (Real.log (1 / 2))) = -2 := by
  sorry

end power_function_value_l799_799868


namespace reasonable_inferences_l799_799402

/- Given conditions from experiment -/
def tosses : List ℕ := [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
def heads : List ℕ := [265, 512, 793, 1034, 1306, 1558, 2083, 2598]

/- Frequencies calculated from observed data -/
def frequencies : List ℚ :=
  (List.zip tosses heads).map (λ p => p.2 / p.1)

/- Theorem to prove reasonable inferences based on the above conditions -/
theorem reasonable_inferences (H : frequencies ≈ [0.53, 0.512, 0.529, 0.517, 0.522, 0.519, 0.521, 0.52]) :
  (∃ p, (∀ f ∈ frequencies, abs (f - p) < 0.01) ∧ p = 0.520) ∧ 
  (∀ n ∈ [1558], ¬ (∀ heads x ∈ tosses, x = 3000 → n = heads)) :=
sorry

end reasonable_inferences_l799_799402


namespace smallest_prime_with_digit_sum_23_l799_799572

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799572


namespace quadratic_function_origin_l799_799913

theorem quadratic_function_origin (m : ℝ) : (let f := λ x, m * x^2 + x + m * (m - 2) in f 0 = 0) → (m = 0 ∨ m = 2) :=
begin
  intro h,
  -- proof will be done here but is currently omitted
  sorry
end

end quadratic_function_origin_l799_799913


namespace cost_of_mobile_phone_l799_799981

-- Definitions
def initial_cost_refrigerator := 15000
def selling_price_refrigerator := 14700
def overall_profit := 500

-- The cost of the mobile phone
def cost_mobile_phone := 8000

-- Statement of the theorem
theorem cost_of_mobile_phone (M : ℝ) 
  (h1 : initial_cost_refrigerator = 15000)
  (h2 : selling_price_refrigerator = initial_cost_refrigerator - 0.02 * initial_cost_refrigerator)
  (h3 : 500 = selling_price_refrigerator + 1.10 * M - (initial_cost_refrigerator + M)) :
  M = cost_mobile_phone := by
  sorry

end cost_of_mobile_phone_l799_799981


namespace smallest_prime_with_digit_sum_23_l799_799584

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799584


namespace general_formula_sum_T_n_lt_three_halves_l799_799310

open Real

/-- Given a sequence {a_n} whose sum of the first n terms is S_n,
    which satisfies S_n + 2n = 2a_n, prove that the general term
    of the sequence is a_n = 2^(n+1) - 2. -/
theorem general_formula (a S : ℕ → ℕ) (h : ∀ n, S n + 2 * n = 2 * a n) :
  ∀ n, a n = 2^(n + 1) - 2 :=
sorry

/-- Given the sequence {b_n} such that b_n = log2(a_n + 2), and let T_n be 
    the sum of the first n terms of the sequence {b_n / (a_n + 2)}, prove 
    that T_n < 3/2. -/
theorem sum_T_n_lt_three_halves (a b : ℕ → ℝ) (T : ℕ → ℝ)
  (ha : ∀ n, a n = 2^(n + 1) - 2)
  (hb : ∀ n, b n = log (a n + 2) / log 2)
  (hT : ∀ n, T n = (∑ k in range n, b k / (a k + 2))) :
  ∀ n, T n < 3 / 2 :=
sorry

end general_formula_sum_T_n_lt_three_halves_l799_799310


namespace initial_provisions_last_l799_799127

theorem initial_provisions_last (x : ℕ) (h : 2000 * (x - 20) = 4000 * 10) : x = 40 :=
by sorry

end initial_provisions_last_l799_799127


namespace circle_cut_three_parts_circle_cut_two_parts_l799_799731

-- Part (a)
theorem circle_cut_three_parts (C : Type) [circle C] (O A : C) (r : ℝ)
  (radius_O : radius (C, O) = r) (radius_A : radius (C, A) = r) :
  ∃ (parts : list C), (length parts = 3) ∧ (∃ C' : C, center (C', A)) :=
sorry

-- Part (b)
theorem circle_cut_two_parts (C1 C2 : Type) [circle C1, circle C2] (O A B C : C1) (r : ℝ)
  (circle2_center : center (C2, A)) (radius_equiv : radius (C1, O) = radius (C2, A)) :
  (B = intersection_point (C1, C2)) ∧ (C = other_intersection_point (C1, C2)) →
  ∃ (parts : list C1), (length parts = 2) ∧ (∃ C' : C1, center (C', A)) :=
sorry

end circle_cut_three_parts_circle_cut_two_parts_l799_799731


namespace find_polynomial_Q_l799_799450

-- Define the polynomial conditions and derive Q(x)
def poly_Q (Q : ℝ → ℝ) (x : ℝ) :=
  Q x = Q 0 + Q 1 * x + Q 3 * x^2

def Q_at_neg1 (Q : ℝ → ℝ) :=
  Q (-1) = 2

-- Our goal is to prove that Q(x) is the specific polynomial given these conditions
theorem find_polynomial_Q (Q : ℝ → ℝ) (h1 : poly_Q Q) (h2 : Q_at_neg1 Q) :
  ∀ x : ℝ, Q x = 0.6 * x^2 - 2 * x - 0.6 :=
sorry

end find_polynomial_Q_l799_799450


namespace finite_sets_of_primes_existence_l799_799977

noncomputable def finite_sets_of_primes (k : ℕ) (hk : k > 0) : Prop :=
∃ (S : finset (finset ℕ)), 
  ∀ T ∈ S, (∀ p ∈ T, prime p) ∧ (finset.prod T (λ p, p + k) % finset.prod T id = 0)
  
theorem finite_sets_of_primes_existence : ∀ k : ℕ, k > 0 → ∃ S : finset (finset ℕ), finite_sets_of_primes k hk :=
sorry

end finite_sets_of_primes_existence_l799_799977


namespace incorrect_option_D_l799_799349

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l799_799349


namespace curved_surface_area_of_cone_l799_799102

noncomputable def approx_CSA_cone (r l : ℝ) : ℝ := π * r * l

theorem curved_surface_area_of_cone :
  approx_CSA_cone 14 35 ≈ 1539.38 :=
by
  sorry

end curved_surface_area_of_cone_l799_799102


namespace connected_paper_area_l799_799491

def side_length := 30 -- side of each square paper in cm
def overlap_length := 7 -- overlap length in cm
def num_pieces := 6 -- number of paper pieces

def effective_length (side_length overlap_length : ℕ) := side_length - overlap_length
def total_connected_length (num_pieces : ℕ) (side_length overlap_length : ℕ) :=
  side_length + (num_pieces - 1) * (effective_length side_length overlap_length)

def width := side_length -- width of the connected paper is the side of each square piece of paper

def area (length width : ℕ) := length * width

theorem connected_paper_area : area (total_connected_length num_pieces side_length overlap_length) width = 4350 :=
by
  sorry

end connected_paper_area_l799_799491


namespace tim_has_33_books_l799_799751

-- Define the conditions
def b := 24   -- Benny's initial books
def s := 10   -- Books given to Sandy
def total_books : Nat := 47  -- Total books

-- Define the remaining books after Benny gives to Sandy
def remaining_b : Nat := b - s

-- Define Tim's books
def tim_books : Nat := total_books - remaining_b

-- Prove that Tim has 33 books
theorem tim_has_33_books : tim_books = 33 := by
  -- This is a placeholder for the proof
  sorry

end tim_has_33_books_l799_799751


namespace squaredigital_numbers_l799_799723

def is_squaredigital (n : ℕ) : Prop :=
  n = (nat.digits 10 n).sum ^ 2

theorem squaredigital_numbers :
  {n : ℕ | is_squaredigital n} = {0, 1, 81} :=
by
  sorry

end squaredigital_numbers_l799_799723


namespace stu_books_count_l799_799489

theorem stu_books_count (S : ℕ) (h1 : S + 4 * S = 45) : S = 9 := 
by
  sorry

end stu_books_count_l799_799489


namespace find_constant_c_l799_799387

theorem find_constant_c (c : ℝ) :
  (∃ c : ℝ, (λ x : ℝ, c * x^3 + 23 * x^2 - 3 * c * x + 45) (-7) = 0) → 
  c = 586 / 161 :=
by
  -- This part is intentionally left for proving the theorem in Lean.
  sorry

end find_constant_c_l799_799387


namespace smallest_prime_with_digit_sum_23_proof_l799_799671

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799671


namespace sum_of_parallel_segments_l799_799021

theorem sum_of_parallel_segments (AB CB : ℝ) (hAB : AB = 6) (hCB : CB = 8) :
  let a_k := λ k : ℕ, 10 * (120 - k) / 120 in
  (\sum k in Finset.range 120, 2 * a_k (k+1)) - 10 = 1180 :=
by
  intros
  simp only [a_k]
  sorry

end sum_of_parallel_segments_l799_799021


namespace largest_three_digit_multiple_of_12_and_sum_of_digits_24_l799_799556

def sum_of_digits (n : ℕ) : ℕ :=
  ((n / 100) + ((n / 10) % 10) + (n % 10))

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def largest_three_digit_multiple_of_12_with_digits_sum_24 : ℕ :=
  996

theorem largest_three_digit_multiple_of_12_and_sum_of_digits_24 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ sum_of_digits n = 24 ∧ is_multiple_of_12 n ∧ n = largest_three_digit_multiple_of_12_with_digits_sum_24 :=
by 
  sorry

end largest_three_digit_multiple_of_12_and_sum_of_digits_24_l799_799556


namespace smallest_prime_with_digit_sum_23_l799_799565

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799565


namespace rhombus_diagonals_not_equal_bisect_each_other_l799_799690

theorem rhombus_diagonals_not_equal_bisect_each_other :
    ¬(∀ (R : Type) [AddCommGroup R] [Module ℝ R] (x y : R), is_rhombus x y → (diagonal_length x = diagonal_length y ∧ bisect x y)) :=
by
  sorry

end rhombus_diagonals_not_equal_bisect_each_other_l799_799690


namespace smallest_prime_with_digit_sum_23_proof_l799_799673

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799673


namespace num_female_fox_terriers_l799_799725

def total_dogs : Nat := 2012
def total_female_dogs : Nat := 1110
def total_fox_terriers : Nat := 1506
def male_shih_tzus : Nat := 202

theorem num_female_fox_terriers :
    ∃ (female_fox_terriers: Nat), 
        female_fox_terriers = total_fox_terriers - (total_dogs - total_female_dogs - male_shih_tzus) := by
    sorry

end num_female_fox_terriers_l799_799725


namespace pow_div_mul_pow_eq_l799_799171

theorem pow_div_mul_pow_eq (a b c d : ℕ) (h_a : a = 8) (h_b : b = 5) (h_c : c = 2) (h_d : d = 6) :
  (a^b / a^c) * (4^6) = 2^21 := by
  sorry

end pow_div_mul_pow_eq_l799_799171


namespace total_pages_to_read_l799_799480

theorem total_pages_to_read 
  (total_books : ℕ)
  (pages_per_book : ℕ)
  (books_read_first_month : ℕ)
  (books_remaining_second_month : ℕ) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  books_remaining_second_month = (total_books - books_read_first_month) / 2 →
  ((total_books * pages_per_book) - ((books_read_first_month + books_remaining_second_month) * pages_per_book) = 1000) :=
by
  sorry

end total_pages_to_read_l799_799480


namespace smallest_prime_with_digit_sum_23_l799_799580

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799580


namespace equationD_no_real_roots_l799_799688

-- Definitions of equations
def equationA (x : ℝ) : ℝ := 3 * x^2 - 1
def equationB (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt 3 * x + 3
def equationC (x : ℝ) : ℝ := x^2 - 2 * x - 1
def equationD (x : ℝ) : ℝ := x^2 - x + 2

-- Discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Coefficients of equation D
def aD : ℝ := 1
def bD : ℝ := -1
def cD : ℝ := 2

-- Proving that the discriminant of equation D is negative
theorem equationD_no_real_roots : discriminant aD bD cD < 0 :=
by
  simp [discriminant, aD, bD, cD]
  sorry

end equationD_no_real_roots_l799_799688


namespace fruit_farm_oranges_l799_799125

theorem fruit_farm_oranges 
  (number_oranges_box : ℝ) (number_boxes_day : ℝ)
  (h1 : number_oranges_box = 10.0)
  (h2 : number_boxes_day = 2650.0) :
  number_oranges_box * number_boxes_day = 26500.0 :=
by
  rw [h1, h2]
  norm_num
  sorry

end fruit_farm_oranges_l799_799125


namespace truthful_warriors_count_l799_799219

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799219


namespace max_value_of_d_l799_799256

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end max_value_of_d_l799_799256


namespace fencers_count_l799_799030

theorem fencers_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 :=
sorry

end fencers_count_l799_799030


namespace solution_set_l799_799313

-- Definition of odd function and monotonic increasing function
def odd_fn (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def mono_increasing (f : ℝ → ℝ) := ∀ x y, 0 < x → x < y → x < y → f(x) < f(y)

-- The main theorem statement
theorem solution_set (f : ℝ → ℝ) (h_odd : odd_fn f) (h_mono : mono_increasing f) (h_f1 : f 1 = 0) :
  {x : ℝ | (x - 1) * f x > 0} = {x | x < -1} ∪ {x | x > 1} :=
sorry

end solution_set_l799_799313


namespace bogatyrs_truthful_count_l799_799242

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799242


namespace xy_sum_proof_l799_799962

-- Define the given list of numbers
def original_list := [201, 202, 204, 205, 206, 209, 209, 210, 212]

-- Define the target new average and sum of numbers
def target_average : ℕ := 207
def sum_xy : ℕ := 417

-- Calculate the original sum
def original_sum : ℕ := 201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212

-- The new total sum calculation with x and y included
def new_total_sum := original_sum + sum_xy

-- Number of elements in the new list
def new_num_elements : ℕ := 11

-- Target new sum based on the new average and number of elements
def target_new_sum := target_average * new_num_elements

theorem xy_sum_proof : new_total_sum = target_new_sum := by
  sorry

end xy_sum_proof_l799_799962


namespace laundry_time_l799_799759

theorem laundry_time (wash_time_per_load : ℕ) (num_loads : ℕ) (dry_time : ℕ)
  (h_wash_time: wash_time_per_load = 45) (h_num_loads: num_loads = 2) (h_dry_time: dry_time = 75) 
  : (num_loads * wash_time_per_load + dry_time = 165) :=
by 
  rw [h_wash_time, h_num_loads, h_dry_time]
  exact add_comm _ _ sorry

end laundry_time_l799_799759


namespace students_attending_harvard_is_900_l799_799410

noncomputable def students_attending_harvard : ℕ :=
  let total_applicants := 20000
  let acceptance_rate := 0.05
  let yield_rate := 0.90
  let accepted_students := acceptance_rate * total_applicants
  let attending_students := yield_rate * accepted_students
  attending_students.to_nat

theorem students_attending_harvard_is_900 :
  students_attending_harvard = 900 :=
by
  -- proof will go here
  sorry

end students_attending_harvard_is_900_l799_799410


namespace difference_of_largest_and_smallest_l799_799527
noncomputable def roots_in_arithmetic_progression : Prop :=
  ∃ (a d : ℚ), (roots 81 (-162) 108 (-18) = [a - d, a, a + d])

theorem difference_of_largest_and_smallest :
  roots_in_arithmetic_progression →
  let roots_list := (roots 81 (-162) 108 (-18))
  min roots_list - max roots_list = (2 / 3) :=
by
  sorry

end difference_of_largest_and_smallest_l799_799527


namespace area_triangle_CMN_l799_799932

-- Define basic geometric entities and the given conditions
variable (A B C D M N : ℝ × ℝ)

-- Conditions given in the problem
def is_square (A B C D : ℝ × ℝ) : Prop :=
  (dist A B = 2) ∧ (dist B C = 2) ∧ (dist C D = 2) ∧ (dist D A = 2) ∧
  (dist A C = dist B D = 2 * sqrt 2) ∧
  (dist B D = dist A C = 2 * sqrt 2)

def is_midpoint (X Y Z : ℝ × ℝ) : Prop := 
  (X.1 = (Y.1 + Z.1) / 2) ∧ (X.2 = (Y.2 + Z.2) / 2)

def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C in
  let b := dist A C in
  let c := dist A B in
  a^2 + b^2 = c^2

-- Calculate the distance function
def dist (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem area_triangle_CMN (h1 : is_square A B C D)
    (h2 : is_midpoint M A D)
    (h3 : is_midpoint N B C)
    (h4 : is_right_triangle C M N) :
    let CM := dist C M in
    let MN := dist M N in
    abs ((CM * MN) / 2) = sqrt 3 := 
by
  -- The proof will be filled in here
  sorry

end area_triangle_CMN_l799_799932


namespace greatest_integer_third_side_l799_799077

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end greatest_integer_third_side_l799_799077


namespace selected_probability_l799_799334

def prob (d : ℤ) : ℝ := log (d + 1) - log d

theorem selected_probability (P : ℤ → ℝ) : 
  P(2) = log 3 - log 2 ∧
  2 * P(2) = P(4) + P(5) + P(6) + P(7) + P(8) :=
by
  -- Define P(2) according to the problem
  have P2 : P(2) = prob 2 := sorry
  -- Define the combined probability for set {4, 5, 6, 7, 8}
  have P_set : P(4) + P(5) + P(6) + P(7) + P(8) = prob 4 + prob 5 + prob 6 + prob 7 + prob 8 := sorry
  -- Now equate 2 * P(2) to the combined probability
  have h : 2 * P(2) = prob 4 + prob 5 + prob 6 + prob 7 + prob 8 := sorry
  exact ⟨P2, h⟩

end selected_probability_l799_799334


namespace proof_problem_l799_799825

variable (f : ℝ → ℝ)
variable (a b : ℝ)

-- Given conditions
axiom condition1 : ∀ x, f (real.exp (x - 1)) = 2 * x - 1
axiom condition2 : f a + f b = 0

-- The statement to prove
theorem proof_problem : a * b = 1 / real.exp 1 := by
  sorry

end proof_problem_l799_799825


namespace erica_earnings_l799_799248

def price_per_kg : ℝ := 20
def past_catch : ℝ := 80
def catch_today := 2 * past_catch
def total_catch := past_catch + catch_today
def total_earnings := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end erica_earnings_l799_799248


namespace y_intercept_of_line_l799_799797

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  have h' : y = -(4/7) * x + 4 := sorry
  have h_intercept : x = 0 := sorry
  exact sorry

end y_intercept_of_line_l799_799797


namespace intersection_of_M_and_N_l799_799881

open Set

def M := {0, 1, 2}
def N := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := 
by
  sorry

end intersection_of_M_and_N_l799_799881


namespace fraction_defined_l799_799389

theorem fraction_defined (x : ℝ) : (1 - 2 * x ≠ 0) ↔ (x ≠ 1 / 2) :=
by sorry

end fraction_defined_l799_799389


namespace binomial_even_sum_l799_799056

open Nat

theorem binomial_even_sum (n : ℕ) (hn : 0 < n) :
  (∑ k in Ico 1 (n + 1), choose (2 * n) (2 * k)) = 2^(2 * n - 1) - 1 :=
by
  sorry

end binomial_even_sum_l799_799056


namespace socks_pairs_count_l799_799898

theorem socks_pairs_count (W B Bl R : ℕ) (hW : W = 5) (hB : B = 5) (hBl : Bl = 3) (hR : R = 2) :
  let pairs := W * B + W * Bl + W * R + B * Bl + B * R + Bl * R
  in pairs = 81 :=
by
  intros
  simp only [hW, hB, hBl, hR]
  norm_num
  sorry

end socks_pairs_count_l799_799898


namespace smallest_prime_with_digit_sum_23_l799_799627

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799627


namespace sin_alpha_cos_2beta_l799_799379

theorem sin_alpha_cos_2beta :
  ∀ α β : ℝ, 3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2 →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 :=
by
  intros α β h
  sorry

end sin_alpha_cos_2beta_l799_799379


namespace area_difference_l799_799715

theorem area_difference (r : ℝ) (h : ℝ) (O : Type) 
  (circle_area : ℝ) (triangle_area : ℝ) : 
  r = 3 → h = 6 → 
  circle_area = 9 * Real.pi → 
  triangle_area = 9 → 
  (circle_area - triangle_area) = 9 * (Real.pi - 1) := 
by
  intros hr hh hcircle htriangle
  rw [hr, hh, hcircle, htriangle]
  simp
  rw [sub_div, mul_div_cancel_left]
  exact two_ne_zero

sorry

end area_difference_l799_799715


namespace distinct_exponentiation_values_l799_799855

theorem distinct_exponentiation_values : 
  (∃ v1 v2 v3 v4 v5 : ℕ, 
    v1 = (3 : ℕ)^(3 : ℕ)^(3 : ℕ)^(3 : ℕ) ∧
    v2 = (3 : ℕ)^((3 : ℕ)^(3 : ℕ)^(3 : ℕ)) ∧
    v3 = (3 : ℕ)^(((3 : ℕ)^(3 : ℕ))^(3 : ℕ)) ∧
    v4 = ((3 : ℕ)^(3 : ℕ)^3) ∧
    v5 = ((3 : ℕ)^((3 : ℕ)^(3 : ℕ)^3)) ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5) := 
sorry

end distinct_exponentiation_values_l799_799855


namespace five_circles_common_point_l799_799316

theorem five_circles_common_point (a b c d e : set ℝ) [circle a] [circle b] [circle c] [circle d] [circle e] :
  (∃ p₁, p₁ ∈ a ∧ p₁ ∈ b ∧ p₁ ∈ c ∧ p₁ ∈ d) →
  (∃ p₂, p₂ ∈ a ∧ p₂ ∈ b ∧ p₂ ∈ c ∧ p₂ ∈ e) →
  (∃ p₃, p₃ ∈ a ∧ p₃ ∈ b ∧ p₃ ∈ d ∧ p₃ ∈ e) →
  (∃ p₄, p₄ ∈ a ∧ p₄ ∈ c ∧ p₄ ∈ d ∧ p₄ ∈ e) →
  (∃ p₅, p₅ ∈ b ∧ p₅ ∈ c ∧ p₅ ∈ d ∧ p₅ ∈ e) →
  ∃ p, p ∈ a ∧ p ∈ b ∧ p ∈ c ∧ p ∈ d ∧ p ∈ e :=
by
  sorry

end five_circles_common_point_l799_799316


namespace extreme_values_a_minus1_monotonicity_intervals_inequality_holds_l799_799335

-- Define the function f
def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 - 2 * a * Real.log x + (a - 2) * x

-- (i) Prove the extreme values when a = -1
theorem extreme_values_a_minus1 : 
  (∃ x_max, x_max = 1 ∧ f x_max (-1) = -5 / 2) ∧ 
  (∃ x_min, x_min = 2 ∧ f x_min (-1) = 2 * Real.log 2 - 4) :=
  sorry

-- (ii) Prove monotonicity intervals when a < 0 
theorem monotonicity_intervals (a : ℝ) (h : a < 0) :
  if a < -2 then 
    ((∀ x ∈ (0, 2), 0 < derivative f x a) -- Increasing in (0, 2)
    ∧ (∀ x ∈ (2, -a), derivative f x a < 0) -- Decreasing in (2, -a)
    ∧ (∀ x ∈ (-a, +∞), 0 < derivative f x a)) -- Increasing in (-a, +∞)
  else if a = -2 then 
    (∀ x > 0, 0 < derivative f x a) -- Increasing in (0, +∞)
  else 
    ((∀ x ∈ (0, -a), 0 < derivative f x a) -- Increasing in (0, -a)
    ∧ (∀ x ∈ (-a, 2), derivative f x a < 0) -- Decreasing in (-a, 2)
    ∧ (∀ x ∈ (2, +∞), 0 < derivative f x a)) -- Increasing in (2, +∞)
  :=
  sorry

-- (iii) Prove that a ≤ -1/2 is the necessary condition for the inequality to hold
theorem inequality_holds (a : ℝ) :
  (∀ m n : ℝ, 0 < m ∧ 0 < n ∧ m ≠ n → (f m a - f n a) / (m - n) > a) ↔ (a ≤ - 1 / 2) :=
  sorry

end extreme_values_a_minus1_monotonicity_intervals_inequality_holds_l799_799335


namespace proof_problem_l799_799367

-- Definitions corresponding to the conditions
def time_to_burn_entire_construction (width height : ℕ) (toothpicks_count : ℕ) (burn_time : ℕ) (start_points: list (ℕ × ℕ)) : ℕ :=
  sorry

-- The problem instance based on given conditions
def example_problem := 
  time_to_burn_entire_construction 3 5 38 10 [(0, 0), (1, 0)] = 65

-- Statement to prove the equivalence
theorem proof_problem : example_problem :=
sorry

end proof_problem_l799_799367


namespace max_points_three_teams_l799_799399
theorem max_points_three_teams (teams : ℕ) (games_per_pair : ℕ) (points_win : ℕ) (points_draw : ℕ) (points_loss : ℕ) : 
  teams = 9 ∧ games_per_pair = 4 ∧ points_win = 3 ∧ points_draw = 1 ∧ points_loss = 0 → 
  ∃ (points : ℕ), points = 76 ∧ 
  (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    (∃ (A B C : ℕ), A = points ∧ B = points ∧ C = points)) :=
begin
  intro h,
  -- The proof part is omitted as per the instructions.
  sorry,
end

end max_points_three_teams_l799_799399


namespace find_f_of_neg2_l799_799817

theorem find_f_of_neg2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x + 1) = 9 * x ^ 2 - 6 * x + 5) : f (-2) = 20 :=
by
  sorry

end find_f_of_neg2_l799_799817


namespace union_subgroup_iff_l799_799951

variables {Γ : Type*} [Group Γ] {G H : Subgroup Γ}

theorem union_subgroup_iff : IsSubgroup ↑(G ∪ H) ↔ G ≤ H ∨ H ≤ G :=
sorry

end union_subgroup_iff_l799_799951


namespace number_of_valid_numbers_l799_799368

def is_valid_number (N : ℕ) : Prop :=
  N ≥ 1000 ∧ N < 10000 ∧ ∃ a x : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ x < 1000 ∧ 
  N = 1000 * a + x ∧ x = N / 9

theorem number_of_valid_numbers : ∃ (n : ℕ), n = 7 ∧ ∀ N, is_valid_number N → N < 1000 * (n + 2) := 
sorry

end number_of_valid_numbers_l799_799368


namespace random_event_sum_gt_six_l799_799743

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def selection (s : List ℕ) := s.length = 3 ∧ s ⊆ numbers

def sum_is_greater_than_six (s : List ℕ) : Prop := s.sum > 6

theorem random_event_sum_gt_six :
  ∀ (s : List ℕ), selection s → (sum_is_greater_than_six s ∨ ¬ sum_is_greater_than_six s) := 
by
  intros s h
  -- Proof omitted
  sorry

end random_event_sum_gt_six_l799_799743


namespace solve_for_z_l799_799906

theorem solve_for_z (z : ℂ) (i_val : ℂ) (h1 : i_val = complex.I) (h2 : i_val * z = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
sorry

end solve_for_z_l799_799906


namespace students_attending_harvard_is_900_l799_799411

noncomputable def students_attending_harvard : ℕ :=
  let total_applicants := 20000
  let acceptance_rate := 0.05
  let yield_rate := 0.90
  let accepted_students := acceptance_rate * total_applicants
  let attending_students := yield_rate * accepted_students
  attending_students.to_nat

theorem students_attending_harvard_is_900 :
  students_attending_harvard = 900 :=
by
  -- proof will go here
  sorry

end students_attending_harvard_is_900_l799_799411


namespace sin_x_eq_solution_cos_x_eq_solution_tan_x_eq_solution_cot_x_eq_solution_l799_799097

noncomputable def sin_solution (k : ℤ) : ℝ := (-1 : ℝ)^k * (π / 3) + π * k
noncomputable def cos_solution (k : ℤ) : ℝ := (π / 4) * (1 + -1 ^ k) + 2 * π * k
noncomputable def tan_solution (k : ℤ) : ℝ := Real.arctan 3 + π * k
noncomputable def cot_solution (k : ℤ) : ℝ := (3 * π / 4) + π * k

theorem sin_x_eq_solution (x : ℝ) (k : ℤ) :
  Real.sin x = (√3 / 2) ↔ x = sin_solution k := by sorry

theorem cos_x_eq_solution (x : ℝ) (k : ℤ) :
  Real.cos x = (√2 / 2) ↔ x = cos_solution k := by sorry

theorem tan_x_eq_solution (x : ℝ) (k : ℤ) :
  Real.tan x = 3 ↔ x = tan_solution k := by sorry

theorem cot_x_eq_solution (x : ℝ) (k : ℤ) :
  Real.cot x = -1 ↔ x = cot_solution k := by sorry

end sin_x_eq_solution_cos_x_eq_solution_tan_x_eq_solution_cot_x_eq_solution_l799_799097


namespace ellipse_slope_product_l799_799487

variables {a b x1 y1 x2 y2 : ℝ} (h₁ : a > b) (h₂ : b > 0) (h₃ : (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) ∧ (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2))

theorem ellipse_slope_product : 
  (a > b) → (b > 0) → (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) → 
  (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2) → 
  ( (y1 + y2)/(x1 + x2) ) * ( (y1 - y2)/(x1 - x2) ) = - (b^2 / a^2) :=
by
  intros ha hb hxy1 hxy2
  sorry

end ellipse_slope_product_l799_799487


namespace determine_functions_l799_799191

def f_iter (f : ℤ → ℤ) : ℕ → (ℤ → ℤ)
| 0     := id
| (n+1) := f ∘ (f_iter f n)

theorem determine_functions :
  ∀ f : ℤ → ℤ,
  (∀ a b : ℤ, f_iter f (a^2 + b^2) (a + b) = a * f a + b * f b) →
  (f = 0 ∨ ∀ x : ℤ, f x = x + 1) := by
  sorry

end determine_functions_l799_799191


namespace smallest_prime_with_digit_sum_23_l799_799622

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799622


namespace find_x_l799_799303

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : Nat.coprime 2 3)
  (h3 : (∀ p : ℕ, Prime p → p ∣ x → p = 2 ∨ p = 3))
  (h4 : Nat.count (λ p, Prime p ∧ p ∣ x) = 3)
  : x = 480 ∨ x = 2016 :=
by sorry

end find_x_l799_799303


namespace trig_identity_l799_799814

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
    Real.cos (2 * α) - Real.sin α * Real.cos α = -1 := 
by 
  sorry

end trig_identity_l799_799814


namespace locker_number_l799_799432

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def digits_sum (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2

theorem locker_number (n : ℕ) :
  (10 ≤ n ∧ n < 100) ∧
  (((n % 5 = 0) + is_square n + (digits_sum n = 16) + (n % 10 = 4)) = 3) →
  n = 64 :=
begin
  sorry
end

end locker_number_l799_799432


namespace quadratic_equation_solution_and_sum_l799_799990

theorem quadratic_equation_solution_and_sum :
  ∃ a b : ℝ, a ≥ b ∧ a = 3 + real.sqrt 19 ∧ b = 3 - real.sqrt 19 ∧ a + 3 * b = 12 - 2 * real.sqrt 19 :=
by
  sorry

end quadratic_equation_solution_and_sum_l799_799990


namespace floor_length_l799_799516

theorem floor_length (b l : ℝ)
  (h1 : l = 3 * b)
  (h2 : 3 * b^2 = 484 / 3) :
  l = 22 := 
sorry

end floor_length_l799_799516


namespace min_value_of_vector_difference_is_six_l799_799845

open Real

noncomputable def min_value_of_vector_difference (a b : ℝ) (h_angle : ∀ (a b : ℝ), angle a b = π * (2 / 3)) (h_dot_product : ∀ (a b : ℝ), dot_product a b = -1) : ℝ :=
  sqrt (norm (a - b) ^ 2)

-- Now let's state the theorem with the conditions
theorem min_value_of_vector_difference_is_six 
  (a b : ℝ)
  (h_angle : ∀ (a b : ℝ), angle a b = π * (2 / 3)) 
  (h_dot_product : ∀ (a b : ℝ), dot_product a b = -1) :
  min_value_of_vector_difference a b h_angle h_dot_product = sqrt 6 :=
sorry

end min_value_of_vector_difference_is_six_l799_799845


namespace warriors_truth_tellers_l799_799203

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799203


namespace binary_to_decimal_l799_799188

-- Definitions based on conditions
def binary_number := [1, 1, 0, 1, 0, 1, 1]

-- Statement of the proof problem
theorem binary_to_decimal :
  (list.sum (list.map_with_index (λ i b, b * 2 ^ (6 - i)) binary_number) = 107) :=
by
  sorry

end binary_to_decimal_l799_799188


namespace horner_operations_count_l799_799757

-- Given polynomial and x value
def polynomial := (λ x : ℝ, 9 * x^6 + 3 * x^5 + 4 * x^4 + 6 * x^3 + x^2 + 8 * x + 1)
def x_value : ℝ := 3

-- Property to prove
theorem horner_operations_count : 
  let f := polynomial
  let x := x_value
  (∃ total_ops : ℕ, total_ops = 12) := 
sorry

end horner_operations_count_l799_799757


namespace manuscript_typing_cost_l799_799099

theorem manuscript_typing_cost :
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 :=
by
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  have : cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 := sorry
  exact this

end manuscript_typing_cost_l799_799099


namespace seq_property_l799_799453

theorem seq_property (m : ℤ) (h1 : |m| ≥ 2)
  (a : ℕ → ℤ)
  (h2 : ¬ (a 1 = 0 ∧ a 2 = 0))
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) = a (n + 1) - m * a n)
  (r s : ℕ)
  (h4 : r > s ∧ s ≥ 2)
  (h5 : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| :=
by
  sorry

end seq_property_l799_799453


namespace fraction_subtraction_l799_799682

theorem fraction_subtraction (a b : ℕ) (h₁ : a = 18) (h₂ : b = 14) :
  (↑a / ↑b - ↑b / ↑a) = (32 / 63) := by
  sorry

end fraction_subtraction_l799_799682


namespace minimum_value_of_f_on_interval_l799_799519

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem minimum_value_of_f_on_interval : 
  ∃ m : ℝ, (∀ x ∈ Icc (-2 : ℝ) 2, f x ≥ m) ∧ m = 0 :=
by
  sorry

end minimum_value_of_f_on_interval_l799_799519


namespace verify_quotient_remainder_example_l799_799560

open Polynomial

noncomputable def quotient_remainder_example : Prop :=
  let p := Polynomial.C 8 * Polynomial.X^4 + Polynomial.C 16 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 4 * Polynomial.X + Polynomial.C 9
  let q := Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 5 * Polynomial.X + Polynomial.C 3
  let quotient := Polynomial.C 4 * Polynomial.X^2 - Polynomial.C 2 * Polynomial.X + Polynomial.C 3
  let remainder := Polynomial.C 43 * Polynomial.X + Polynomial.C 36
  p = q * quotient + remainder

theorem verify_quotient_remainder_example : quotient_remainder_example := 
sorry

end verify_quotient_remainder_example_l799_799560


namespace collinear_condition_perpendicular_condition_l799_799711

-- Problem 1: Prove collinearity condition for k = -2
theorem collinear_condition (k : ℝ) : 
  (k - 5) * (-12) - (12 - k) * 6 = 0 ↔ k = -2 :=
sorry

-- Problem 2: Prove perpendicular condition for k = 2 or k = 11
theorem perpendicular_condition (k : ℝ) : 
  (20 + (k - 6) * (7 - k)) = 0 ↔ (k = 2 ∨ k = 11) :=
sorry

end collinear_condition_perpendicular_condition_l799_799711


namespace number_of_truth_tellers_is_twelve_l799_799237
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799237


namespace problem_1_problem_2_problem_3_l799_799167

namespace RingTossGame

-- Definitions of probabilities involved
def P_A_hit := (3 : ℚ) / 4
def P_A_miss := 1 - P_A_hit
def P_B_hit := (1 : ℚ) / 2
def P_B_miss := 1 - P_B_hit

-- Total score distribution for one round
def X_distribution (x : ℚ) : ℚ :=
  match x with
  | 0 => P_A_miss * P_A_miss * P_B_miss
  | 1 => (P_A_hit * P_A_miss + P_A_miss * P_A_hit) * P_B_miss
  | 2 => P_A_hit * P_A_hit * P_B_miss + P_A_miss * P_A_miss * P_B_hit
  | 3 => (P_A_hit * P_A_miss + P_A_miss * P_A_hit) * P_B_hit
  | 4 => P_A_hit * P_A_hit * P_B_hit
  | _ => 0

-- Expected value of the total score
def E_X : ℚ :=
  ∑ x in {0, 1, 2, 3, 4}, x * X_distribution x

-- Probability of hitting target exactly once
def P_hit_exactly_once : ℚ :=
  (P_A_hit * P_A_miss + P_A_miss * P_A_hit) * P_B_miss + P_A_miss * P_A_miss * P_B_hit

-- Probability of scoring 2 or 3 points in exactly 3 out of 5 rounds
def P_score_2_or_3_in_3_of_5_rounds : ℚ :=
  let P_succeed := X_distribution 2 + X_distribution 3
  let P_fail := 1 - P_succeed
  ∑ n in {3}, (nat.choose 5 n : ℚ) * (P_succeed ^ n) * (P_fail ^ (5 - n))

-- Statements of the problems
theorem problem_1 : P_hit_exactly_once = (7 : ℚ) / 32 := sorry

theorem problem_2 : E_X = (5 : ℚ) / 2 := sorry

theorem problem_3 : P_score_2_or_3_in_3_of_5_rounds = (5 : ℚ) / 16 := sorry

end RingTossGame

end problem_1_problem_2_problem_3_l799_799167


namespace cos_100_eq_neg_sqrt_one_minus_a_square_l799_799841

theorem cos_100_eq_neg_sqrt_one_minus_a_square (a : ℝ) (ha : Real.sin 80 = a) : 
  Real.cos 100 = -Real.sqrt(1 - a^2) := 
by
  sorry

end cos_100_eq_neg_sqrt_one_minus_a_square_l799_799841


namespace solve_for_x_l799_799381

variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h : 3 * x^2 + 9 * x * y = x^3 + 3 * x^2 * y)

theorem solve_for_x : x = 3 :=
by
  sorry

end solve_for_x_l799_799381


namespace number_of_truth_tellers_is_twelve_l799_799233
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799233


namespace sum_of_valid_x_values_equals_92_l799_799494

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem sum_of_valid_x_values_equals_92 :
  let total_students := 360
  let valid_x (x : ℕ) := x ≥ 18 ∧ is_factor x total_students ∧ total_students / x ≥ 12
  let valid_values := {x ∈ finset.range (total_students + 1) | valid_x x}
  finset.sum valid_values id = 92 := by
  sorry

end sum_of_valid_x_values_equals_92_l799_799494


namespace vector_decomposition_l799_799691

open Matrix

def x : Vector ℝ := ![3, 1, 8]

def p : Vector ℝ := ![0, 1, 3]

def q : Vector ℝ := ![1, 2, -1]

def r : Vector ℝ := ![2, 0, -1]

theorem vector_decomposition :
  ∃ α β γ : ℝ, x = α • p + β • q + γ • r ∧ α = 3 ∧ β = -1 ∧ γ = 2 := by
  sorry

end vector_decomposition_l799_799691


namespace inequality_holds_range_of_expression_l799_799867

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem inequality_holds (x : ℝ) : f x < |x - 2| + 4 ↔ x ∈ Set.Ioo (-5 : ℝ) 3 := by
  sorry

theorem range_of_expression (m n : ℝ) (h : m + n = 2) (hm : m > 0) (hn : n > 0) :
  (m^2 + 2) / m + (n^2 + 1) / n ∈ Set.Ici ((7 + 2 * Real.sqrt 2) / 2) := by
  sorry

end inequality_holds_range_of_expression_l799_799867


namespace bogatyrs_truthful_count_l799_799240

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799240


namespace series_sum_eq_one_half_l799_799778

-- Define the sequence y
def y : ℕ → ℕ
| 0 := 2     -- We start indexing from 0 to align with common Lean conventions
| (n + 1) := y n ^ 2 - y n

-- Define the sequence sum
def sum_term (k : ℕ) : ℚ := 1 / (y k + 1)

-- The proposition we want to prove
theorem series_sum_eq_one_half : 
  (∑' k, sum_term k) = 1 / 2 :=
sorry

end series_sum_eq_one_half_l799_799778


namespace Harvard_attendance_l799_799414

theorem Harvard_attendance:
  (total_applicants : ℕ) (acceptance_rate : ℝ) (attendance_rate : ℝ) 
  (h1 : total_applicants = 20000) 
  (h2 : acceptance_rate = 0.05) 
  (h3 : attendance_rate = 0.9) :
  ∃ (number_attending : ℕ), number_attending = 900 := 
by 
  sorry

end Harvard_attendance_l799_799414


namespace probability_relationship_l799_799543

def total_outcomes : ℕ := 36

def P1 : ℚ := 1 / total_outcomes
def P2 : ℚ := 2 / total_outcomes
def P3 : ℚ := 3 / total_outcomes

theorem probability_relationship :
  P1 < P2 ∧ P2 < P3 :=
by
  sorry

end probability_relationship_l799_799543


namespace range_of_f_l799_799274

def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  set.range (f) = set.Ioo (±∞) 3 ∪ set.Ioo 3 ∞ :=
sorry

end range_of_f_l799_799274


namespace number_of_truth_tellers_is_twelve_l799_799232
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799232


namespace values_of_a_and_b_l799_799917

theorem values_of_a_and_b (a b : ℝ) :
  (∀ x : ℝ, x ≥ -1 → a * x^2 + b * x + a^2 - 1 ≤ 0) →
  a = 0 ∧ b = -1 :=
sorry

end values_of_a_and_b_l799_799917


namespace y_intercept_of_line_l799_799794

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  rw [hx] at h
  simp at h
  exact h

end y_intercept_of_line_l799_799794


namespace correct_chord_construction_l799_799787

noncomputable theory

variable {O : Point} -- center of the circle
variable {P Q A B M N K F A_1 B_1 : Point}
variable {r : Real} -- radius of the circle
variable {circle : Circle O r}

-- Given conditions
axiom chord_in_circle : Chord O circle A B
axiom radii_subdivide : OP = OQ
axiom segment_subdivision : AM = MN = NB
axiom midpoint_of_chord : Midpoint F A B ∧ Midpoint F M N
axiom midpoint_of_arc : Midpoint K P Q
axiom arc_equal : ArcLength A P = ArcLength B Q
axiom chords_parallel : Parallel A B P Q
axiom points_on_line : A_1 = (Line O A).extending_intersection P Q ∧ B_1 = (Line O B).extending_intersection P Q
axiom segment_length : Length A_1 P = Length P Q ∧ Length Q B_1 = Length P Q
axiom homothety_triangles : Homothety O A B A_1 B_1

-- Proof statement to prove the correctness of the chord construction
theorem correct_chord_construction :
  ∃ (A B M N F K A_1 B_1 : Point), 
    chord_in_circle O circle A B ∧
    radii_subdivide O P Q ∧
    segment_subdivision A M N B ∧
    midpoint_of_chord F A B M N ∧
    midpoint_of_arc K P Q ∧
    arc_equal A P B Q ∧
    chords_parallel A B P Q ∧
    points_on_line A_1 A A_1 P Q B_1 B B_1 P Q ∧
    segment_length A_1 P P Q Q B_1 ∧
    homothety_triangles A B A_1 B_1 :=
sorry

end correct_chord_construction_l799_799787


namespace laundry_time_l799_799760

theorem laundry_time (wash_time_per_load : ℕ) (num_loads : ℕ) (dry_time : ℕ)
  (h_wash_time: wash_time_per_load = 45) (h_num_loads: num_loads = 2) (h_dry_time: dry_time = 75) 
  : (num_loads * wash_time_per_load + dry_time = 165) :=
by 
  rw [h_wash_time, h_num_loads, h_dry_time]
  exact add_comm _ _ sorry

end laundry_time_l799_799760


namespace y_intercept_of_line_l799_799796

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  have h' : y = -(4/7) * x + 4 := sorry
  have h_intercept : x = 0 := sorry
  exact sorry

end y_intercept_of_line_l799_799796


namespace letters_received_per_day_l799_799045

-- Define the conditions
def packages_per_day := 20
def total_pieces_in_six_months := 14400
def days_in_month := 30
def months := 6

-- Calculate total days in six months
def total_days := months * days_in_month

-- Calculate pieces of mail per day
def pieces_per_day := total_pieces_in_six_months / total_days

-- Define the number of letters per day
def letters_per_day := pieces_per_day - packages_per_day

-- Prove that the number of letters per day is 60
theorem letters_received_per_day : letters_per_day = 60 := sorry

end letters_received_per_day_l799_799045


namespace smallest_prime_with_digit_sum_23_l799_799603

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799603


namespace percent_volume_filled_by_cubes_l799_799142

theorem percent_volume_filled_by_cubes : 
  let box_volume := 8 * 6 * 12,
      cube_volume := 4 * 4 * 4,
      max_cubes := (8 / 4) * (6 / 4) * (12 / 4),
      total_cube_volume := max_cubes * cube_volume in
  (total_cube_volume / box_volume) * 100 = 66.67 := 
sorry

end percent_volume_filled_by_cubes_l799_799142


namespace smallest_prime_with_digit_sum_23_proof_l799_799674

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799674


namespace students_attending_Harvard_l799_799417

theorem students_attending_Harvard (total_applicants : ℕ) (perc_accepted : ℝ) (perc_attending : ℝ)
    (h1 : total_applicants = 20000)
    (h2 : perc_accepted = 0.05)
    (h3 : perc_attending = 0.9) :
    total_applicants * perc_accepted * perc_attending = 900 := 
by
    sorry

end students_attending_Harvard_l799_799417


namespace repeating_decimal_fraction_817_numerator_denominator_difference_l799_799800

noncomputable def repeating_decimal_fraction (x : ℕ) : ℚ :=
  if x = 817 then 817 / 999 else 0

theorem repeating_decimal_fraction_817 :
  repeating_decimal_fraction 817 = 817 / 999 :=
by
  rcases eq_refl 817 with rfl
  simp [repeating_decimal_fraction]

theorem numerator_denominator_difference (n d : ℕ) :
  let frac := repeating_decimal_fraction 817
  frac.num = n ∧ frac.denom = d → (d - n = 182) :=
by
  intro frac
  have hfrac : repeating_decimal_fraction 817 = 817 / 999 := repeating_decimal_fraction_817
  rw [←rat.mk_eq_div, hfrac] at frac
  simp only [rat.num_mk, rat.denom_mk, int.coe_nat_eq_coe_nat_iff, int.coe_nat_sub, ne.def, 817, 999] at frac
  exact frac.1 ▸ frac.2 ▸ rfl

end repeating_decimal_fraction_817_numerator_denominator_difference_l799_799800


namespace pyramid_angle_problem_l799_799420

theorem pyramid_angle_problem :
  let A := (0, 0, 0)
  let B := (Real.sqrt 3, 0, 0)
  let C := (Real.sqrt 3, 1, 0)
  let D := (0, 1, 0)
  let P := (0, 0, 2)
  let E := (0, 1/2, 1)
  let AC := (Real.sqrt 3, 1, 0)
  let PB := (Real.sqrt 3, 0, -2)
  let dot_product := (AC.1 * PB.1 + AC.2 * PB.2 + AC.3 * PB.3)
  cos_angle_AC_PB : Real := dot_product / (Real.sqrt ((AC.1) ^ 2 + (AC.2) ^ 2 + (AC.3) ^ 2) * Real.sqrt ((PB.1) ^ 2 + (PB.2) ^ 2 + (PB.3) ^ 2)) = 3 * Real.sqrt 7 / 14
  ∧ ∃ (N_x : Real) (N_z : Real), 
    let N := (N_x, 0, N_z)
    let NE := (N.1 - E.1, N.2 - E.2, N.3 - E.3)
    NE.1 * PA.1 + NE.2 * PA.2 + NE.3 * PA.3 = 0
    ∧ NE.1 * AC.1 + NE.2 * AC.2 + NE.3 * AC.3 = 0
    ∧ N_x = Real.sqrt 3 / 6
    ∧ N_z = 1
    ∧ dist_A (A.1, A.2, A.3) N = 1
    ∧ dist_P (A.1, A.2, A.3) N = Real.sqrt 3 / 6 :=
by
  sorry

end pyramid_angle_problem_l799_799420


namespace sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l799_799057

-- Definitions for vertices of pyramids
variables (A B C D E : ℝ)

-- Assuming E is inside pyramid ABCD
variable (inside : E ∈ convex_hull ℝ {A, B, C, D})

-- Assertion 1
theorem sum_of_edges_not_always_smaller
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : D ≠ E):
  ¬ (abs A - E + abs B - E + abs C - E < abs A - D + abs B - D + abs C - D) :=
sorry

-- Assertion 2
theorem at_least_one_edge_shorter
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A)
  (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D)
  (h7 : D ≠ E):
  abs A - E < abs A - D ∨ abs B - E < abs B - D ∨ abs C - E < abs C - D :=
sorry

end sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l799_799057


namespace acute_triangle_iff_equal_segments_l799_799018

-- Define the structure of a point and a triangle
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)

-- Define the notion of a segment within a triangle
def on_side (P : Point) (X Y : Point) : Prop := 
  -- Condition defining P lies on the line segment XY
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = {x := t * X.x + (1 - t) * Y.x, y := t * X.y + (1 - t) * Y.y}

-- Define the condition of a triangle being acute
def is_acute (Δ : Triangle) : Prop := 
  -- All internal angles being less than 90 degrees is equivalent to this derived condition
  let (A, B, C) := (Δ.A, Δ.B, Δ.C) in
  (dot_product (vector_orthogonal (A - B) (C - B)) < 0) ∧
  (dot_product (vector_orthogonal (B - A) (C - A)) < 0) ∧
  (dot_product (vector_orthogonal (C - A) (B - A)) < 0)

-- Main statement of the theorem
theorem acute_triangle_iff_equal_segments (Δ : Triangle) :
  (is_acute Δ) ↔ ∃ A₁ B₁ C₁ : Point, 
          on_side A₁ Δ.B Δ.C ∧ 
          on_side B₁ Δ.C Δ.A ∧ 
          on_side C₁ Δ.A Δ.B ∧
          dist Δ.A A₁ = dist Δ.B B₁ ∧
          dist Δ.C C₁ = dist Δ.A A₁ :=
sorry

end acute_triangle_iff_equal_segments_l799_799018


namespace smallest_prime_with_digit_sum_23_l799_799568

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799568


namespace three_girls_three_boys_l799_799533

variable (G B : Type)
variable [Fintype G] [Fintype B]
variable [DecidableEq G] [DecidableEq B]

def num_girls : ℕ := 21
def num_boys : ℕ := 20

axiom girls_count : Fintype.card G = num_girls
axiom boys_count : Fintype.card B = num_boys

variable (P : Type)
variable [Fintype P]

def max_problems_solved : ℕ := 6

axiom each_solves_max_6_problems (g : G) (b : B) (p : P) : 
  ∀ g, Fintype.card (g → Option P) ≤ max_problems_solved ∧
  ∀ b, Fintype.card (b → Option P) ≤ max_problems_solved

axiom common_problem_exists (g : G) (b : B) : ∃ p : P, g = b

theorem three_girls_three_boys :
  ∃ p : P, ∃ (sg : Finset G) (sb : Finset B), 
    sg.card ≥ 3 ∧ sb.card ≥ 3 ∧ 
    ∀ g ∈ sg, ∀ b ∈ sb, g = b ∧
    common_problem_exists g b :=
sorry

end three_girls_three_boys_l799_799533


namespace exist_three_rational_points_in_acute_rational_triangle_l799_799966

def is_rational_angle (α : ℝ) : Prop := ∃ (q : ℚ), α = (q : ℝ)

def is_rational_triangle (A B C : ℝ) : Prop :=
  is_rational_angle A ∧ is_rational_angle B ∧ is_rational_angle C

def is_acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90 ∧ A + B + C = 180

def is_rational_point (P : ℝ × ℝ) (triangles : List (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (triangle : ℝ × ℝ × ℝ) ∈ triangles, is_rational_triangle triangle.fst triangle.snd triangle.snd

theorem exist_three_rational_points_in_acute_rational_triangle
  (A B C : ℝ) (h_rat_triangle : is_rational_triangle A B C) (h_acute : is_acute_triangle A B C) :
  ∃ P Q R : ℝ × ℝ, is_rational_point P [] ∧ is_rational_point Q [] ∧ is_rational_point R [] ∧ (P ≠ Q ∧ P ≠ R ∧ Q ≠ R) :=
sorry

end exist_three_rational_points_in_acute_rational_triangle_l799_799966


namespace smallest_prime_with_digit_sum_23_l799_799647

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799647


namespace smallest_square_side_length_l799_799713

theorem smallest_square_side_length
    (rect1 : ℕ × ℕ) (rect1_dim1 : rect1 = (2, 4))
    (rect2 : ℕ × ℕ) (rect2_dim1 : rect2 = (4, 5)) :
    ∃ (s : ℕ), s = 6 ∧ (∀ (x1 y1 x2 y2 : ℕ), x1 ≠ x2 ∨ y1 ≠ y2 ∨ 
    (x1 + rect1.1 ≤ s ∧ y1 + rect1.2 ≤ s) ∧ (x2 + rect2.1 ≤ s ∧ y2 + rect2.2 ≤ s)) :=
begin
    sorry
end

end smallest_square_side_length_l799_799713


namespace carlos_laundry_time_l799_799761

theorem carlos_laundry_time :
  ∀ (wash_time per_load : ℕ) (dry_time : ℕ),
    wash_time = 45 →
    per_load = 2 →
    dry_time = 75 →
    (wash_time * per_load + dry_time) = 165 :=
by
  intros wash_time per_load dry_time h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end carlos_laundry_time_l799_799761


namespace constant_arc_length_l799_799163

variables {α : Type}

structure Point (α : Type) :=
(x : α) (y : α)

structure Triangle (α : Type) :=
(A B C : Point α)

-- Predicate for isosceles triangle
def is_isosceles {α : Type} [LinearOrderedField α] (T : Triangle α) : Prop :=
(λ p q : Point α, (p.x - q.x)^2 + (p.y - q.y)^2) T.A T.B = (λ p q : Point α, (p.x - q.x)^2 + (p.y - q.y)^2) T.B T.C

-- Line segment BD is the height
def is_height {α : Type} [LinearOrderedField α] (T : Triangle α) (D : Point α) : Prop :=
D.x = T.B.x ∧ ((T.A.y - T.B.y) * (T.C.x - T.A.x) = (T.A.x - T.C.x) * (T.B.y - D.y))

-- Circle with radius BD
def circle_radius_height {α : Type} [LinearOrderedField α] (T : Triangle α) (D : Point α) (O : Point α) (r : α) : Prop := 
r = (λ p q : Point α, (p.x - q.x)^2 + (p.y - q.y)^2) T.B D ∧ circle_center O r

-- Prove the length of the arc inside the triangle is constant
theorem constant_arc_length {α : Type} [LinearOrderedField α]
  (T : Triangle α) (D : Point α) (O : Point α) (r : α)
  (h_iso : is_isosceles T) (h_height : is_height T D) (h_circle : circle_radius_height T D O r)
  (B_in_circle: (λ p q : Point α, (p.x - q.x)^2 + (p.y - q.y)^2) T.B O ≤ r) :
  ∃ arc_length : α, arc_in_triangle T O r D arc_length ∧ ∀ B_in_circle, arc_in_triangle T O r D arc_length := sorry

end constant_arc_length_l799_799163


namespace find_x_l799_799297

open Nat

theorem find_x (n : ℕ) (x : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (factors x).nodup)
  (h3 : (factors x).length = 3)
  (h4 : 3 ∈ factors x) :
  x = 480 ∨ x = 2016 := 
sorry

end find_x_l799_799297


namespace quarter_circle_area_ratio_l799_799980

theorem quarter_circle_area_ratio (R : ℝ) (hR : 0 < R) :
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  (2 * (AXC + BYD) / O = 1 / 8) := 
by
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  sorry

end quarter_circle_area_ratio_l799_799980


namespace smallest_prime_with_digit_sum_23_l799_799625

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799625


namespace hyperbola_eccentricity_is_sqrt_three_l799_799347

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : c^2 = 3 * a^2) : ℝ :=
  let e := Real.sqrt(1 + (b^2 / a^2)) in
  e

theorem hyperbola_eccentricity_is_sqrt_three
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : c^2 = 3 * a^2)
  (h4 : (∀ x y : ℝ, (b * x + a * y = 0) → (x - c)^2 + y^2 = 4 * a^2 → ∃ k : ℝ, 2 * k = 2 * b)) :
  hyperbola_eccentricity a b c h1 h2 h3 = Real.sqrt(3) := by
  sorry

end hyperbola_eccentricity_is_sqrt_three_l799_799347


namespace number_of_ordered_quadruples_l799_799784

-- Define the conditions for the ordered quadruples
def satisfies_conditions (a b c d : ℝ) : Prop :=
  a^2 + b^2 + c^2 + d^2 = 4 ∧ 
  (a + b + c + d) * (a^4 + b^4 + c^4 + d^4) = 32

-- Define the problem statement
theorem number_of_ordered_quadruples : 
  { quad : ℝ × ℝ × ℝ × ℝ // satisfies_conditions quad.1 quad.2 quad.2 quad.2 }.card = 2 := 
sorry

end number_of_ordered_quadruples_l799_799784


namespace smallest_prime_with_digit_sum_23_l799_799600

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799600


namespace arithmetic_geometric_relation_l799_799839

variable (a₁ a₂ b₁ b₂ b₃ : ℝ)

-- Conditions
def is_arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ (d : ℝ), -2 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -8

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ (r : ℝ), -2 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -8

-- The problem statement
theorem arithmetic_geometric_relation (h₁ : is_arithmetic_sequence a₁ a₂) (h₂ : is_geometric_sequence b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1 / 2 := by
    sorry

end arithmetic_geometric_relation_l799_799839


namespace volume_percentage_l799_799144

-- Definitions of the initial conditions
def box_length : ℝ := 8
def box_width : ℝ := 6
def box_height : ℝ := 12
def cube_side : ℝ := 4

-- Definition for the correct answer
def correct_answer : ℝ := 66.67

-- The Lean 4 statement to express and prove the given problem
theorem volume_percentage :
  let box_volume := box_length * box_width * box_height,
      cube_volume := cube_side ^ 3,
      num_cubes := (box_length / cube_side).to_int * (box_width / cube_side).to_int * (box_height / cube_side).to_int,
      cubes_volume := num_cubes * cube_volume,
      volume_percentage := (cubes_volume / box_volume) * 100 in
  volume_percentage = correct_answer :=
by
  sorry

end volume_percentage_l799_799144


namespace outbreak_time_l799_799254

noncomputable def f (t : ℝ) : ℝ := 1 / (1 + exp (-0.22 * (3 * t - 40)))

theorem outbreak_time : ∃ t : ℝ, f t = 0.1 ∧ t = 10 :=
by {
  use 10,
  split,
  {
    unfold f,
    have : 1 / (1 + exp (-0.22 * (3 * 10 - 40))) = 0.1,
    { sorry }, -- where the detailed proof showing the calculation would go
    exact this,
  },
  {
    refl,
  }
}

end outbreak_time_l799_799254


namespace digit_swap_difference_l799_799777

theorem digit_swap_difference (a b c : ℕ) : 
  (|(100 * a + 10 * b + c) - (10 * a + 100 * b + c)|) = 90 * |a - b| :=
by {
  sorry
}

example : ∃ (a b c : ℕ), (|(100 * a + 10 * b + c) - (10 * a + 100 * b + c)|) = 90 :=
by {
  use 1, use 2, use 0,  -- Example values
  exact digit_swap_difference 1 2 0
}

end digit_swap_difference_l799_799777


namespace smallest_prime_with_digit_sum_23_l799_799654

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799654


namespace proof_by_contradiction_example_l799_799685

variables {L1 L2 : Type*} [linear_ordered_field L1] [linear_ordered_field L2]
          (T : L1 → L2) (θ₁ θ₂ : L1) 

def interior_angles_not_complementary : Prop :=
  θ₁ + θ₂ ≠ 180

def lines_not_parallel (L1 L2 : Type*) [linear_ordered_field L1] [linear_ordered_field L2] : Prop :=
  ∀ θ₁ θ₂, interior_angles_not_complementary θ₁ θ₂ → ¬ (L1 = L2)

theorem proof_by_contradiction_example :
  lines_not_parallel L1 L2 := 
by
  intros
  sorry

end proof_by_contradiction_example_l799_799685


namespace smallest_prime_with_digit_sum_23_l799_799623

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799623


namespace smallest_prime_with_digit_sum_23_l799_799606

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799606


namespace groupA_cannot_form_set_l799_799741

def is_definite (X : Type) (S : set X) : Prop := ∀ x y ∈ S, x = y
def is_unordered (X : Type) (S : set X) : Prop := true  -- Simplified as the definition of unorderedness for sets
def is_distinct (X : Type) (S : set X) : Prop := ∀ x ∈ S, x ≠ x -> false  -- Ensuring no duplicates (a bit simplified)

-- Definitions of each group in the problem
def groupA (X : Type) : set X := { x : X | x ∈ X }  -- Placeholder, real definition depends on the domain
def groupB (X : Type) : set X := { x : X | x ∈ X }  -- Placeholder for parents of students
def groupC (X : Type) : set X := { x : X | x ∈ X }  -- Placeholder for courses
def groupD (X : Type) : set X := { x : X | x ∈ X }  -- Placeholder for students taller than 1.7m

-- Condition that groupA does not satisfy definiteness
axiom groupA_not_definite : ¬ is_definite ℕ groupA

-- The theorem to prove
theorem groupA_cannot_form_set :
  ¬ ( (is_definite ℕ groupA) ∧ (is_unordered ℕ groupA) ∧ (is_distinct ℕ groupA) ) :=
by
  intro h
  cases h with h_definite h_rest
  exact groupA_not_definite h_definite

end groupA_cannot_form_set_l799_799741


namespace max_side_length_of_squares_l799_799538

variable (l w : ℕ)
variable (h_l : l = 54)
variable (h_w : w = 24)

theorem max_side_length_of_squares : gcd l w = 6 :=
by
  rw [h_l, h_w]
  sorry

end max_side_length_of_squares_l799_799538


namespace sequence_inequality_l799_799461

noncomputable def seq_sum (S : ℕ → ℝ) : Prop :=
  S 1 = 1 ∧ ∀ n ≥ 1, S (n + 1) = (2 + S n)^2 / (4 + S n)

def general_term (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then S 1 else S n - S (n - 1)

theorem sequence_inequality (S : ℕ → ℝ) (n : ℕ) (h_seq : seq_sum S) : general_term S n ≥ 4 / (sqrt (9 * n + 7)) :=
sorry

end sequence_inequality_l799_799461


namespace page_added_twice_l799_799095

theorem page_added_twice (n : ℕ) (sum_with_duplicate : ℕ) (sum_without_duplicate : ℕ) :
  (sum_with_duplicate = 2011) →
  (sum_without_duplicate = ∑ i in range (n + 1), i) →
  (∃ k, k ∈ range (n + 1) ∧ sum_with_duplicate = sum_without_duplicate + k) →
  n = 62 ∧ ∃ k, k = 58 :=
by
  sorry

end page_added_twice_l799_799095


namespace smallest_prime_digit_sum_23_l799_799620

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799620


namespace truthful_warriors_count_l799_799220

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799220


namespace deductive_reasoning_used_l799_799108

section
  variable (Metals : Type) (ElectricalConducts : Metals → Prop)
  variable (is_metal : Metals) (iron : Metals)

  -- All metals conduct electricity
  axiom all_metals_conducts : ∀ x : Metals, ElectricalConducts x

  -- Iron is a metal
  axiom iron_is_metal : is_metal = iron

  -- Therefore, Iron conducts electricity
  theorem deductive_reasoning_used :
    (all_metals_conducts iron) → deductive := 
  by
    sorry
end

end deductive_reasoning_used_l799_799108


namespace division_of_power_l799_799956

theorem division_of_power (m : ℕ) 
  (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end division_of_power_l799_799956


namespace number_of_truth_tellers_is_twelve_l799_799234
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l799_799234


namespace mr_green_expected_potatoes_yield_l799_799008

def garden_steps_width := 25
def garden_steps_length := 30
def step_length_feet := 3
def non_productive_percentage := 0.1
def yield_pounds_per_square_foot := 3 / 4

def productive_area_feet : ℕ := (garden_steps_width * step_length_feet) * (garden_steps_length * step_length_feet) * (1 - non_productive_percentage)

def expected_yield_pounds : ℕ := productive_area_feet * yield_pounds_per_square_foot

theorem mr_green_expected_potatoes_yield :
  expected_yield_pounds = 4556.25 := by
  sorry

end mr_green_expected_potatoes_yield_l799_799008


namespace length_of_segment_AC_l799_799473

/-- Given points A and B on a circle of radius 8 with AB = 10, and point C as the midpoint of the minor arc AB.
    Prove that the length of the line segment AC is equal to √(128 - 16√39). -/
theorem length_of_segment_AC 
  (r : ℝ := 8) 
  (h1 : A : ℝ × ℝ) 
  (h2 : B : ℝ × ℝ) 
  (h3 : dist A B = 10) 
  (O : ℝ × ℝ) 
  (C : ℝ × ℝ) 
  (h4 : (dist O A = r) ∧ (dist O B = r))
  (h5 : midpoint O A B C) : 
  dist A C = real.sqrt (128 - 16 * real.sqrt 39) :=
begin
  sorry
end

end length_of_segment_AC_l799_799473


namespace greatest_third_side_of_triangle_l799_799078

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : ∃ x : ℕ, x < a + b ∧ x = 16 := by
  use 16
  rw [h1, h2]
  split
  · linarith
  · rfl

end greatest_third_side_of_triangle_l799_799078


namespace product_of_D_coordinates_l799_799323

theorem product_of_D_coordinates 
  (M D : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hC : C = (5, 3))
  (hM : M = (3, 7))
  (h_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
  D.1 * D.2 = 11 :=
by
  sorry

end product_of_D_coordinates_l799_799323


namespace area_XVW_l799_799940

-- Declare the necessary definitions and conditions
variables {X Y Z V W : Type}
variables (triangle_XYZ : ∀ {a b c : Type}, (a × b × c) → Prop)
variables (midpoint : ∀ {a b c : Type}, (a × b) → c → Prop)
variables (trisection : ∀ {a b : Type}, (a × b) → Prop)
variables (area : ∀ {a b c : Type}, (a × b × c) → ℝ)

-- Define the conditions given in the problem
noncomputable def triangle_XYZ_condition : Prop := triangle_XYZ ⟨X, Y, Z⟩
noncomputable def W_on_XY : Prop := trisection ⟨X, Y⟩
noncomputable def V_midpoint_YZ : Prop := midpoint ⟨Y, Z⟩ V
noncomputable def area_XYZ : Prop := area ⟨X, Y, Z⟩ = 150

-- Lean statement for the proof problem
theorem area_XVW : W_on_XY ∧ V_midpoint_YZ ∧ area_XYZ → area ⟨X, V, W⟩ = 25 := sorry

end area_XVW_l799_799940


namespace smallest_prime_with_digit_sum_23_l799_799592

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799592


namespace birds_per_cup_l799_799941

theorem birds_per_cup :
  ∀ (C B S T : ℕ) (H1 : C = 2) (H2 : S = 1 / 2 * C) (H3 : T = 21) (H4 : B = 14),
    ((C - S) * B = T) :=
by
  sorry

end birds_per_cup_l799_799941


namespace valid_a_in_A_l799_799874

theorem valid_a_in_A : 
  let A := {x : ℝ | x^2 - 2 < 0} in
  -1 ∈ A := 
by {
  -- Lean proof steps here
  sorry
}

end valid_a_in_A_l799_799874


namespace johns_raw_squat_weight_l799_799945

variable (R : ℝ)

def sleeves_lift := R + 30
def wraps_lift := 1.25 * R
def wraps_more_than_sleeves := wraps_lift R - sleeves_lift R = 120

theorem johns_raw_squat_weight : wraps_more_than_sleeves R → R = 600 :=
by
  intro h
  sorry

end johns_raw_squat_weight_l799_799945


namespace problem1_problem2_l799_799851

variables {ℕ : Type*} [linear_ordered_semiring ℕ]

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, a (n + 1) - a n = a n - a (n - 1)

def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, a k)

theorem problem1
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : ∀ n, 2 * S n = n * (a 1 + a n))
  (h2 : ∀ n ∈ ℕ, S (n + 1) = S n + a (n + 1)) :
  is_arithmetic_sequence a := sorry

def Tn (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, b k)

theorem problem2
  (a S : ℕ → ℕ)
  (b : ℕ → ℕ)
  (T : ℕ → ℕ)
  (h1 : ∀ n, 2 * S n = n * (a 1 + a n))
  (h2 : ∀ n ∈ ℕ, S (n + 1) = S n + a (n + 1))
  (h3 : S 3 = 6)
  (h4 : S 5 = 15)
  (h5 : ∀ n, b n = a n / (n + 1)!)
  (h6 : ∀ n, T n = (finset.range n).sum (λ k, b k)) :
  ∀ n, T n < 1 := sorry

end problem1_problem2_l799_799851


namespace smallest_prime_with_digit_sum_23_proof_l799_799672

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799672


namespace inverse_proportion_m_pos_l799_799312

theorem inverse_proportion_m_pos (x : ℝ) (hx : x > 0) (m : ℝ) (y : ℝ) (h : y = m / x) (hy_decreasing : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 < x2 → (m / x1 > m / x2)) : m > 0 :=
by
  -- Proof would go here
  sorry

example : ∃ m : ℝ, m > 0 :=
begin
  use 1,  -- A possible positive value for m is 1
  norm_num,
end

end inverse_proportion_m_pos_l799_799312


namespace smallest_prime_with_digit_sum_23_l799_799586

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799586


namespace smallest_prime_with_digit_sum_23_l799_799605

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799605


namespace smallest_prime_with_digit_sum_23_l799_799601

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799601


namespace smallest_number_of_soldiers_l799_799808

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

noncomputable def smallest_soldiers :=
  let n := Nat.lcm 10 (Nat.lcm 9 (Nat.lcm 8 (Nat.lcm 7 (Nat.lcm 6 (Nat.lcm 5 (Nat.lcm 4 (Nat.lcm 3 2))))))
  in n - 1

theorem smallest_number_of_soldiers :
  ∃ n : ℕ, 
    (n % 2 = 1) ∧ 
    (n % 3 = 2) ∧ 
    (n % 4 = 3) ∧ 
    (n % 5 = 4) ∧ 
    (n % 6 = 5) ∧ 
    (n % 7 = 6) ∧ 
    (n % 8 = 7) ∧ 
    (n % 9 = 8) ∧ 
    (n % 10 = 9) ∧ 
    (n = 2519) :=
by
  use 2519
  sorry

end smallest_number_of_soldiers_l799_799808


namespace smallest_prime_with_digit_sum_23_proof_l799_799677

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799677


namespace solution_set_l799_799196

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 3 * x - 4

-- Define the inequality
def inequality (x : ℝ) : Prop := quadratic_expr x > 0

-- State the theorem
theorem solution_set : ∀ x : ℝ, inequality x ↔ (x > 1 ∨ x < -4) :=
by
  sorry

end solution_set_l799_799196


namespace total_percent_samplers_is_correct_l799_799699

-- Define the constants
def total_customers : ℝ := 100
def percent_caught : ℝ := 22 / 100
def percent_not_caught_of_samplers : ℝ := 5 / 100

-- Define a function representing total percent of customers who sample candy
noncomputable def percent_samplers : ℝ :=
  let x := 22 / 0.95 in x / total_customers

-- Lean 4 statement to prove
theorem total_percent_samplers_is_correct :
  percent_samplers ≈ 23.16 / 100 :=
by
  sorry

end total_percent_samplers_is_correct_l799_799699


namespace pizzas_served_dinner_eq_6_l799_799152

-- Definitions based on the conditions
def pizzas_served_lunch : Nat := 9
def pizzas_served_today : Nat := 15

-- The theorem to prove the number of pizzas served during dinner
theorem pizzas_served_dinner_eq_6 : pizzas_served_today - pizzas_served_lunch = 6 := by
  sorry

end pizzas_served_dinner_eq_6_l799_799152


namespace major_premise_error_l799_799857

variables {Line : Type} {Plane : Type}
variable (a : Line)
variable (b : Line)
variable (alpha : Plane)

-- Conditions
-- Line $a$ is not contained in plane $\alpha$
axiom not_contained_in_plane : ¬(a ∈ alpha)
-- Line $b$ is contained in plane $\alpha$
axiom contained_in_plane : b ∈ alpha
-- Line $a \parallel \alpha$
axiom parallel_to_plane : ∀ (p : Plane), p = alpha → (∀ (l : Line), l ∈ p → a ∥ l)

-- Conclusion: The major premise that "If a line is parallel to a plane, then the line is parallel to all lines within the plane" is incorrect
theorem major_premise_error : ¬ (a ∥ b) :=
sorry

end major_premise_error_l799_799857


namespace non_congruent_triangles_count_l799_799396

-- Let there be 15 equally spaced points on a circle,
-- and considering triangles formed by connecting 3 of these points.
def num_non_congruent_triangles (n : Nat) : Nat :=
  (if n = 15 then 19 else 0)

theorem non_congruent_triangles_count :
  num_non_congruent_triangles 15 = 19 :=
by
  sorry

end non_congruent_triangles_count_l799_799396


namespace age_problem_l799_799992

theorem age_problem :
  ∃ (x y z : ℕ), 
    x - y = 3 ∧
    z = 2 * x + 2 * y - 3 ∧
    z = x + y + 20 ∧
    x = 13 ∧
    y = 10 ∧
    z = 43 :=
by 
  sorry

end age_problem_l799_799992


namespace num_correct_props_l799_799047

-- Define the original proposition p
def p (x : ℝ) : Prop := (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 2)

-- Define the contrapositive of the original proposition
def contrapositive (x : ℝ) : Prop := (¬ (x ≠ 2)) → ¬ (x^2 - 3 * x + 2 ≠ 0)

-- Define the converse of the original proposition
def converse (x : ℝ) : Prop := (x ≠ 2) → (x^2 - 3 * x + 2 ≠ 0)

-- Define the inverse of the original proposition
def inverse (x : ℝ) : Prop := (¬ (x^2 - 3 * x + 2 ≠ 0)) → ¬ (x ≠ 2)

-- The proof problem
theorem num_correct_props :
  (contrapositive 0) ∧ ¬(converse 1) ∧ ¬(inverse 1) → 1 :=
by
  sorry

end num_correct_props_l799_799047


namespace soda_cost_l799_799084

theorem soda_cost (b s f : ℕ) (h1 : 3 * b + 2 * s + 2 * f = 590) (h2 : 2 * b + 3 * s + f = 610) : s = 140 :=
sorry

end soda_cost_l799_799084


namespace number_of_rows_is_nine_l799_799128

-- Define the conditions as lean statements
def top_row_cans : ℕ := 2
def common_difference : ℕ := 3
def nth_row_cans (n : ℕ) : ℕ := top_row_cans + common_difference * (n - 1)

-- Proof problem statement
theorem number_of_rows_is_nine (n : ℕ) :
  (nth_row_cans n = 25) → n = 9 :=
begin
  sorry
end

end number_of_rows_is_nine_l799_799128


namespace ordered_triples_count_l799_799953

open Finset

noncomputable def satisfies_relation (a b : ℕ) : Prop :=
  (0 < a - b ∧ a - b ≤ 9) ∨ (b - a > 9)

def count_triples (S : Finset ℕ) : ℕ :=
  S.filter (λ x, S.filter (λ y, satisfies_relation x y).filter (λ z, satisfies_relation y z ∧ satisfies_relation z x)).card

theorem ordered_triples_count : 
  let S := (Finset.range 19).map (λ n, n + 1) in
  count_triples S = 855 := 
by
  sorry

end ordered_triples_count_l799_799953


namespace smallest_prime_with_digit_sum_23_proof_l799_799670

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799670


namespace red_card_value_l799_799022

theorem red_card_value (credits : ℕ) (total_cards : ℕ) (blue_card_value : ℕ) (red_cards : ℕ) (blue_cards : ℕ) 
    (condition1 : blue_card_value = 5)
    (condition2 : total_cards = 20)
    (condition3 : credits = 84)
    (condition4 : red_cards = 8)
    (condition5 : blue_cards = total_cards - red_cards) :
  (credits - blue_cards * blue_card_value) / red_cards = 3 :=
by
  sorry

end red_card_value_l799_799022


namespace f_2x_increasing_no_extreme_l799_799342

def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem f_2x_increasing_no_extreme : ∀ (x : ℝ), 
  IsIncreasing (λ x, f (2 * x)) ∧ ¬ ∃ a b : ℝ, a < b ∧ localExtremum (λ x, f (2 * x)) a b :=
sorry

end f_2x_increasing_no_extreme_l799_799342


namespace count_plane_figures_l799_799856

def is_plane_figure (fig : string) : Prop :=
  fig = "angle" ∨ fig = "triangle" ∨ fig = "parallelogram" ∨ fig = "trapezoid"

theorem count_plane_figures :
  (["angle", "triangle", "parallelogram", "trapezoid", "quadrilateral"].filter is_plane_figure).length = 4 :=
by
  sorry

end count_plane_figures_l799_799856


namespace find_a_l799_799315

noncomputable theory
open Real

def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x - 34

theorem find_a (a b c : ℝ) (h1 : ∀ x, -2 ≤ x ∧ x ≤ 3 → 3 * a * x ^ 2 + 2 * b * x + c ≤ 0)
  (h2 : ∀ x, f a b c x ≥ -115) :
  a = 2 :=
sorry

end find_a_l799_799315


namespace lcm_of_denominators_l799_799092

theorem lcm_of_denominators : Nat.lcm (List.foldr Nat.lcm 1 [2, 3, 4, 5, 6, 7]) = 420 :=
by 
  sorry

end lcm_of_denominators_l799_799092


namespace isosceles_trapezoid_min_x_squared_l799_799436

open EuclideanGeometry

-- Define the geometric conditions and the proof goal
theorem isosceles_trapezoid_min_x_squared :
  ∀ (A B C D O : Point) (x : ℝ),
    is_isosceles_trapezoid A B C D ∧
    side_length A B = 100 ∧
    side_length C D = 28 ∧
    side_length A D = x ∧
    side_length B C = x ∧
    circle_tangents_on_diagonal A C O A D B C →
    x^2 = 3200 :=
by
  sorry

end isosceles_trapezoid_min_x_squared_l799_799436


namespace hyperbola_eccentricity_ineq_l799_799327

def hyperbola_eccentricity_range (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h_no_intersection : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 ≠ 1 ∨ 2*x ≠ y) : Prop :=
  let e := Real.sqrt (1 + (b/a)^2) in
    1 < e ∧ e ≤ Real.sqrt 5

theorem hyperbola_eccentricity_ineq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h_no_intersection : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 ≠ 1 ∨ 2*x ≠ y) :
  hyperbola_eccentricity_range a b h1 h2 h_no_intersection :=
by
  sorry

end hyperbola_eccentricity_ineq_l799_799327


namespace number_of_solutions_eq_six_l799_799897

/-- 
The number of ordered pairs (m, n) of positive integers satisfying the equation
6/m + 3/n = 1 is 6.
-/
theorem number_of_solutions_eq_six : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ p ∈ s, (1 < p.1 ∧ 1 < p.2) ∧ 6 / p.1 + 3 / p.2 = 1) ∧ s.card = 6 :=
sorry

end number_of_solutions_eq_six_l799_799897


namespace union_of_sets_l799_799320

noncomputable def setA : Set ℝ := { x : ℝ | x^3 - 3 * x^2 - x + 3 < 0 }
noncomputable def setB : Set ℝ := { x : ℝ | abs (x + 1/2) >= 1 }

theorem union_of_sets:
  setA ∪ setB = (Iio (-1) ∪ Ici (1/2)) :=
by
  sorry

end union_of_sets_l799_799320


namespace same_color_3x3_intersection_l799_799034

theorem same_color_3x3_intersection :
  ∀ (grid : Fin 5 → Fin 41 → bool), ∃ (rows : Finset (Fin 5)) (cols : Finset (Fin 41)), 
  rows.card = 3 ∧ cols.card = 3 ∧ 
  (∃ color : bool, ∀ i ∈ rows, ∀ j ∈ cols, grid i j = color) :=
by
  sorry

end same_color_3x3_intersection_l799_799034


namespace find_dividend_l799_799088

theorem find_dividend (divisor quotient remainder : ℤ) (hdivisor : divisor = 17) (hquotient : quotient = 9) (hremainder : remainder = 5) : 
  (divisor * quotient + remainder = 158) :=
by
  rw [hdivisor, hquotient, hreminder]
  sorry

end find_dividend_l799_799088


namespace markup_percentage_correct_l799_799158

def purchase_price_A : ℝ := 48
def purchase_price_B : ℝ := 36
def purchase_price_C : ℝ := 60

def overhead_A : ℝ := 0.20 * purchase_price_A
def overhead_B : ℝ := 0.15 * purchase_price_B
def overhead_C : ℝ := 0.25 * purchase_price_C

def profit_A : ℝ := 12
def profit_B : ℝ := 8
def profit_C : ℝ := 16

def total_cost_A : ℝ := purchase_price_A + overhead_A + profit_A
def total_cost_B : ℝ := purchase_price_B + overhead_B + profit_B
def total_cost_C : ℝ := purchase_price_C + overhead_C + profit_C

def selling_price_A : ℝ := Real.ceil total_cost_A
def selling_price_B : ℝ := Real.ceil total_cost_B
def selling_price_C : ℝ := Real.ceil total_cost_C

def markup_percentage (purchase_price selling_price : ℝ) : ℝ :=
  ((selling_price - purchase_price) / purchase_price) * 100

def markup_percentage_A : ℝ := Real.ceil (markup_percentage purchase_price_A selling_price_A)
def markup_percentage_B : ℝ := Real.ceil (markup_percentage purchase_price_B selling_price_B)
def markup_percentage_C : ℝ := Real.ceil (markup_percentage purchase_price_C selling_price_C)

theorem markup_percentage_correct :
  markup_percentage_A = 46 ∧ markup_percentage_B = 39 ∧ markup_percentage_C = 52 :=
by
  sorry

end markup_percentage_correct_l799_799158


namespace sin_cos_alpha_beta_l799_799888

theorem sin_cos_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.cos α = Real.sin (2 * β)) :
  Real.sin β ^ 2 + Real.cos α ^ 2 = 3 / 2 := 
by
  sorry

end sin_cos_alpha_beta_l799_799888


namespace smallest_prime_with_digit_sum_23_l799_799563

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l799_799563


namespace bogatyrs_truthful_count_l799_799239

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799239


namespace exists_countable_subset_l799_799439

noncomputable theory

variables {α : Type*} (X : ℝ → ℝ) (P : set ℝ → ℝ) (X_alpha : α → ℝ) (Y : ℕ → ℝ)

def extended_random_variable := ∀ a : α, measurable_space (X_alpha a)

def ess_sup (f : α → ℝ) := supr Y

theorem exists_countable_subset 
  (family_X_alpha : ∀ a : α, extended_random_variable)
  (ess_sup_def : ess_sup X_alpha = supr Y) :
  (P (set_of (λ x : ℝ, supr Y ≤ x)) = 1 → 
   ∀ α, P (set_of (λ x : ℝ, X_alpha α ≤ x)) = 1) :=
sorry

end exists_countable_subset_l799_799439


namespace sequence_divisible_by_sum_l799_799106

theorem sequence_divisible_by_sum (p : ℕ) (hp_prime : p.prime) (hp_odd : p % 2 = 1) : 
  (∃ n : ℕ, let seq := list.range (p - 1) | n + 1 in 
      (seq.map (λ k, (k + n + 1) ^ 2)).sum % (seq.sum + (p - 1) * n) = 0) ↔ p % 6 = 5 := 
sorry

end sequence_divisible_by_sum_l799_799106


namespace right_isosceles_triangle_acute_angle_45_l799_799397

theorem right_isosceles_triangle_acute_angle_45
    (a : ℝ)
    (h_leg_conditions : ∀ b : ℝ, a = b)
    (h_hypotenuse_condition : ∀ c : ℝ, c^2 = 2 * (a * a)) :
    ∃ θ : ℝ, θ = 45 :=
by
    sorry

end right_isosceles_triangle_acute_angle_45_l799_799397


namespace min_abs_sum_x1_x2_l799_799340

open Real

-- Define the function f
def f (x : ℝ) : ℝ := sin (x - π / 3)

-- Main theorem statement proving the minimum value of |x₁ + x₂| is 2π/3
theorem min_abs_sum_x1_x2 (x₁ x₂ : ℝ) (hx : f x₁ + f x₂ = 0)
  (hmono : ∀ x y, x₁ ≤ x ∧ x < y ∧ y ≤ x₂ → f x < f y) :
  |x₁ + x₂| = 2 * π / 3 :=
sorry

end min_abs_sum_x1_x2_l799_799340


namespace leap_day_2040_is_wednesday_l799_799946

/-- Given Leap Day, February 29, 2020, occurred on a Saturday and the total number of days between February 29, 2020, and February 29, 2040, find the day of the week for February 29, 2040. -/
theorem leap_day_2040_is_wednesday :
  let 
    year2020 := 2020,
    year2040 := 2040,
    leap_day_2020 := Nat.succ 0, -- Saturday (assuming 0 = Saturday)
    total_years := year2040 - year2020,
    leap_years := List.countp (λ y, (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)) (List.range' year2020 total_years),
    non_leap_years := total_years - leap_years,
    total_days := non_leap_years * 365 + leap_years * 366,
    day_of_week_2040 := (leap_day_2020 + total_days) % 7
  in 
  day_of_week_2040 = 3 :=   -- Wednesday
by 
  sorry

end leap_day_2040_is_wednesday_l799_799946


namespace smallest_prime_with_digit_sum_23_l799_799575

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799575


namespace quadratic_standard_form_l799_799189

theorem quadratic_standard_form :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = (x + 1) * (3 * x + 4) →
  (∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x
  intro h
  sorry

end quadratic_standard_form_l799_799189


namespace smallest_prime_with_digit_sum_23_l799_799646

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799646


namespace max_apple_trees_l799_799889

-- Definitions for the problem
def pear_trees : ℕ := 4
def distance_between_pears : ℝ := 10
def exact_distance : ℝ := 10
def more_apple_than_pear : Prop := ∀ a_count p_count, a_count > p_count

-- Theorem Statement
theorem max_apple_trees (pear_trees : ℕ) (distance_between_pears : ℝ) (exact_distance : ℝ) (apple_trees : ℕ):
  pear_trees = 4 ∧ distance_between_pears = 10 ∧ exact_distance = 10 ∧ more_apple_than_pear apple_trees pear_trees →
  apple_trees = 12 :=
sorry

end max_apple_trees_l799_799889


namespace fiona_cleaning_time_l799_799463

theorem fiona_cleaning_time (total_time : ℝ) (lilly_fraction : ℝ) (fiona_fraction : ℝ) :
  total_time = 8 ∧ lilly_fraction = 1/4 ∧ fiona_fraction = 3/4 → 
  let fiona_time := (total_time * fiona_fraction) * 60 in
  fiona_time = 360 :=
by
  intros h
  let h1 := h.1
  let h2 := h.2.1
  let h3 := h.2.2
  have total_time_correct : total_time = 8 := h1
  have lilly_fraction_correct : lilly_fraction = 1 / 4 := h2
  have fiona_fraction_correct : fiona_fraction = 3 / 4 := h3
  let fiona_time := (total_time * fiona_fraction) * 60
  have fiona_time_eq : fiona_time = (8 * (3/4)) * 60 := by { rw [total_time_correct, fiona_fraction_correct] }
  have fiona_time_eq_simplified : fiona_time = 6 * 60 := by { rw [← mul_assoc, mul_comm (8 : ℝ), ← div_mul_cancel (8 * (3 : ℝ)) (4 : ℝ), div_self (4 : ℝ)] }
  have fiona_time_eq_final : fiona_time = 360 := by { rw [mul_comm, mul_assoc, mul_comm 6 60] }
  exact fiona_time_eq_final

end fiona_cleaning_time_l799_799463


namespace smallest_prime_with_digit_sum_23_l799_799594

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799594


namespace smallest_prime_digit_sum_23_l799_799612

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799612


namespace area_is_half_lr_l799_799854

variables (r l : ℝ)

-- Definition of the area of the sector
def area_of_sector (r l : ℝ) : ℝ := (1 / 2) * l * r

-- Theorem statement
theorem area_is_half_lr : area_of_sector r l = (1 / 2) * l * r :=
by sorry

end area_is_half_lr_l799_799854


namespace largest_perimeter_triangle_l799_799160

theorem largest_perimeter_triangle :
  ∃ (y : ℤ), 4 < y ∧ y < 20 ∧ 8 + 12 + y = 39 :=
by {
  -- we'll skip the proof steps
  sorry 
}

end largest_perimeter_triangle_l799_799160


namespace necessary_but_not_sufficient_condition_for_odd_cosine_l799_799859

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

theorem necessary_but_not_sufficient_condition_for_odd_cosine (A : ℝ) (ω : ℝ) (φ : ℝ) :
  A > 0 → ω > 0 → (∀ x, A * cos (ω * x + φ) = A * -cos (ω * x + φ) → f x = A * cos (ω * x + φ) → is_odd_function f) ↔ cos φ = 0 :=
begin
  sorry -- Proof required
end

end necessary_but_not_sufficient_condition_for_odd_cosine_l799_799859


namespace limit_transform_l799_799460

theorem limit_transform (f : ℝ → ℝ) (h : differentiable ℝ f) :
  (∃ x : ℝ, x = 1) →  (∀ h : ℝ → ℝ, has_deriv_at f (f' h) 1 x ) → 
  (lim (Δx : ℝ) (hx : 0 < Δx) (δx : ℝ) : Δx x = (1 + Δx x (1 + 3*y )) - f x = f' x) :=
by
  sorry

end limit_transform_l799_799460


namespace multiplication_proof_l799_799172

theorem multiplication_proof : (0.05 : ℝ) * (0.3 : ℝ) * (2 : ℝ) = (0.03 : ℝ) :=
by
  have h1 : (0.05 : ℝ) = 5 * 10^(-2) := by norm_num
  have h2 : (0.3 : ℝ) = 3 * 10^(-1) := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end multiplication_proof_l799_799172


namespace sum_of_roots_combined_eq_five_l799_799278

noncomputable def sum_of_roots_poly1 : ℝ :=
-(-9/3)

noncomputable def sum_of_roots_poly2 : ℝ :=
-(-8/4)

theorem sum_of_roots_combined_eq_five :
  sum_of_roots_poly1 + sum_of_roots_poly2 = 5 :=
by
  sorry

end sum_of_roots_combined_eq_five_l799_799278


namespace min_perimeter_of_polygon_formed_by_zeros_of_Q_is_8_sqrt_2_l799_799952

noncomputable def Q (z : ℂ) : ℂ := z^8 + (2 * Real.sqrt 5 + 8) * z^4 - (2 * Real.sqrt 5 + 9)

theorem min_perimeter_of_polygon_formed_by_zeros_of_Q_is_8_sqrt_2 :
  ∀ (z₀ z₁ z₂ z₃ z₄ z₅ z₆ z₇ : ℂ),
    (Q z₀ = 0) ∧ (Q z₁ = 0) ∧ (Q z₂ = 0) ∧ (Q z₃ = 0) ∧ (Q z₄ = 0) ∧ (Q z₅ = 0) ∧ (Q z₆ = 0) ∧ (Q z₇ = 0) →
    list_perimeter [z₀, z₁, z₂, z₃, z₄, z₅, z₆, z₇] ≥ 8 * Real.sqrt 2 := sorry

end min_perimeter_of_polygon_formed_by_zeros_of_Q_is_8_sqrt_2_l799_799952


namespace polar_eq_parabola_l799_799197

/-- Prove that the curve defined by the polar equation is a parabola. -/
theorem polar_eq_parabola :
  ∀ (r θ : ℝ), r = 1 / (2 * Real.sin θ + Real.cos θ) →
    ∃ (x y : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (x + 2 * y = r^2) :=
by 
  sorry

end polar_eq_parabola_l799_799197


namespace set_of_values_a_l799_799899

theorem set_of_values_a (a : ℝ) : (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ≤ 2) :=
by
  sorry

end set_of_values_a_l799_799899


namespace white_pairs_coincide_l799_799788

theorem white_pairs_coincide
  (red_triangles_half : ℕ)
  (blue_triangles_half : ℕ)
  (white_triangles_half : ℕ)
  (red_pairs : ℕ)
  (blue_pairs : ℕ)
  (red_white_pairs : ℕ)
  (red_triangles_total_half : red_triangles_half = 4)
  (blue_triangles_total_half : blue_triangles_half = 6)
  (white_triangles_total_half : white_triangles_half = 10)
  (red_pairs_total : red_pairs = 3)
  (blue_pairs_total : blue_pairs = 4)
  (red_white_pairs_total : red_white_pairs = 3) :
  ∃ w : ℕ, w = 5 :=
by
  sorry

end white_pairs_coincide_l799_799788


namespace complement_intersection_l799_799887

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 3, 4}

theorem complement_intersection (U : Set ℕ) (A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {1, 2, 3, 4}) :
  (U \ (A ∩ B)) = {4, 5} :=
by
  rw [hU, hA, hB]
  simp only [Set.inter_eq, Set.diff_eq]
  exact sorry

end complement_intersection_l799_799887


namespace minimum_distance_l799_799829

theorem minimum_distance (a b : ℝ) (h : 3 * a + 4 * b = 15) : 
  ∃ m : ℝ, m = min (√((a - 1)^2 + (b + 2)^2)) (4) := 
  sorry

end minimum_distance_l799_799829


namespace equal_roots_a_l799_799037

theorem equal_roots_a {a : ℕ} :
  (a * a - 4 * (a + 3) = 0) → a = 6 := 
sorry

end equal_roots_a_l799_799037


namespace smallest_prime_with_digit_sum_23_l799_799635

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799635


namespace central_cell_value_l799_799929

theorem central_cell_value
    (numbers : Fin 5 → Fin 5 → ℕ)
    (sum_all : (∑ i j, numbers i j) = 200)
    (sum_1x3_rects : ∀ i j, (i < 5) → (j < 3) → (numbers i j + numbers i (j+1) + numbers i (j+2) = 23)) :
    numbers 2 2 = 16 :=
by
  sorry

end central_cell_value_l799_799929


namespace erica_total_earnings_l799_799251

def fishPrice : Nat := 20
def pastCatch : Nat := 80
def todayCatch : Nat := 2 * pastCatch
def pastEarnings := pastCatch * fishPrice
def todayEarnings := todayCatch * fishPrice
def totalEarnings := pastEarnings + todayEarnings

theorem erica_total_earnings : totalEarnings = 4800 := by
  sorry

end erica_total_earnings_l799_799251


namespace find_lambda_l799_799364

noncomputable def collinear {V : Type*} [AddCommGroup V] [Module ℝ V] (a b : V) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = k • a

variables (e1 e2 : V) (λ : ℝ)
variables [AddCommGroup V] [Module ℝ V]

def a := (2 : ℝ) • e1 - e2
def b := e1 + λ • e2

axiom non_collinear_vectors : ¬ ∃ k : ℝ, k ≠ 0 ∧ e2 = k • e1

theorem find_lambda (h_collinear : collinear a b) : λ = -1 / 2 :=
sorry

end find_lambda_l799_799364


namespace smallest_prime_with_digit_sum_23_l799_799608

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799608


namespace truthfulness_count_l799_799222

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799222


namespace arithmetic_sequence_b_sum_a_n_max_D_100_l799_799873

-- Definitions
def a : ℕ → ℝ
| 0     => 1
| (n+1) => a n / 2 + 1 / 2^n

def b (n : ℕ) : ℕ := 2^(n-1) * a n

def d (n : ℕ) : ℝ := sqrt (1 + 1 / (b n)^2 + 1 / (b (n + 1))^2)

def S (n : ℕ) : ℝ := ∑ i in range n, a (i + 1)

def D (n : ℕ) : ℝ := ∑ i in range n, d (i + 1)

--- Statements
theorem arithmetic_sequence_b (n : ℕ) : b n = n :=
sorry

theorem sum_a_n (n : ℕ) : S n = 4 - (2+n)/2^(n-1) :=
sorry

theorem max_D_100 : ⌊D 100⌋ = 100 :=
sorry

end arithmetic_sequence_b_sum_a_n_max_D_100_l799_799873


namespace number_on_board_after_hour_l799_799012

def digit_product (n : ℕ) : ℕ :=
  let digits := (n.toString.data.map (λ c, c.toNat - '0'.toNat))
  digits.foldl (λ acc d, acc * d) 1

def next_number (n : ℕ) : ℕ :=
  digit_product n + 12

noncomputable def number_after_minutes (initial_number : ℕ) (minutes : ℕ) : ℕ :=
  Nat.iterate next_number minutes initial_number

theorem number_on_board_after_hour : 
  number_after_minutes 27 60 = 14 :=
by
  sorry

end number_on_board_after_hour_l799_799012


namespace intersection_A_B_l799_799362

open Set

def universal_set : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℤ := {1, 2, 3}
def complement_B : Set ℤ := {1, 2}
def B : Set ℤ := universal_set \ complement_B

theorem intersection_A_B : A ∩ B = {3} :=
by
  sorry

end intersection_A_B_l799_799362


namespace truthfulness_count_l799_799228

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799228


namespace new_cost_after_decrease_l799_799739

def actual_cost : ℝ := 2400
def decrease_percentage : ℝ := 0.50
def decreased_amount (cost percentage : ℝ) : ℝ := percentage * cost
def new_cost (cost decreased : ℝ) : ℝ := cost - decreased

theorem new_cost_after_decrease :
  new_cost actual_cost (decreased_amount actual_cost decrease_percentage) = 1200 :=
by sorry

end new_cost_after_decrease_l799_799739


namespace number_of_terms_added_l799_799421

def f (n : Nat) : Real :=
  ∑ i in Finset.range (2^n), 1 / (i + 1)

theorem number_of_terms_added (k : Nat) : 
  (2^k) = 
  (Finset.range (2^(k+1))).card - (Finset.range (2^k)).card :=
by
  sorry

end number_of_terms_added_l799_799421


namespace smallest_prime_digit_sum_23_l799_799619

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799619


namespace find_x_l799_799298

open Nat

theorem find_x (n : ℕ) (x : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (factors x).nodup)
  (h3 : (factors x).length = 3)
  (h4 : 3 ∈ factors x) :
  x = 480 ∨ x = 2016 := 
sorry

end find_x_l799_799298


namespace smallest_prime_with_digit_sum_23_l799_799665

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799665


namespace parallel_line_plane_no_common_points_l799_799333

noncomputable def line := Type
noncomputable def plane := Type

variable {l : line}
variable {α : plane}

-- Definitions for parallel lines and planes, and relations between lines and planes
def parallel_to_plane (l : line) (α : plane) : Prop := sorry -- Definition of line parallel to plane
def within_plane (m : line) (α : plane) : Prop := sorry -- Definition of line within plane
def no_common_points (l m : line) : Prop := sorry -- Definition of no common points between lines

theorem parallel_line_plane_no_common_points
  (h₁ : parallel_to_plane l α)
  (l2 : line)
  (h₂ : within_plane l2 α) :
  no_common_points l l2 :=
sorry

end parallel_line_plane_no_common_points_l799_799333


namespace smallest_prime_with_digit_sum_23_l799_799657

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799657


namespace westbound_cyclist_speed_increase_l799_799071

def eastbound_speed : ℕ := 18
def travel_time : ℕ := 6
def total_distance : ℕ := 246

theorem westbound_cyclist_speed_increase (x : ℕ) :
  eastbound_speed * travel_time + (eastbound_speed + x) * travel_time = total_distance →
  x = 5 :=
by
  sorry

end westbound_cyclist_speed_increase_l799_799071


namespace bernoulli_sum_zero_l799_799976

-- Condition: generating function of Bernoulli numbers
def bernoulli_numbers_generating_function (x : ℝ) : ℝ :=
  x / (Real.exp x - 1)

-- Theorem: for k > 1, sum equals zero
theorem bernoulli_sum_zero {k : ℕ} (h : k > 1) : 
  let B : ℕ → ℝ := λ n, sorry -- Assuming B is defined via the generating function properly
  in ∑ p in Finset.range k, B p * Nat.choose k p = 0 :=
by
  sorry

end bernoulli_sum_zero_l799_799976


namespace basketball_tournament_sum_of_squares_eq_l799_799107

theorem basketball_tournament_sum_of_squares_eq (n : ℕ)
  (x y : Fin n → ℕ)
  (hx : ∀ i : Fin n, x i + y i = n - 1)
  (hxy_sum : ∑ i in Finset.univ, x i = ∑ i in Finset.univ, y i) :
  ∑ i in Finset.univ, (x i)^2 = ∑ i in Finset.univ, (y i)^2 := 
by
  sorry

end basketball_tournament_sum_of_squares_eq_l799_799107


namespace conjugate_of_z_squared_l799_799457

open Complex

-- Define the complex number z
def z : ℂ := 1 + I

-- Define the squared complex number z²
def z_squared := z * z

-- Define the conjugate of the squared complex number z²
def conj_z_squared := conj z_squared

-- State the theorem proving the conjugate of z² is 1 - I
theorem conjugate_of_z_squared : conj_z_squared = 1 - I :=
by 
  -- The proof is omitted here
  sorry

end conjugate_of_z_squared_l799_799457


namespace draw_points_worth_two_l799_799921

/-
In a certain football competition, a victory is worth 3 points, a draw is worth some points, and a defeat is worth 0 points. Each team plays 20 matches. A team scored 14 points after 5 games. The team needs to win at least 6 of the remaining matches to reach the 40-point mark by the end of the tournament. Prove that the number of points a draw is worth is 2.
-/

theorem draw_points_worth_two :
  ∃ D, (∀ (victory_points draw_points defeat_points total_matches matches_played points_scored remaining_matches wins_needed target_points),
    victory_points = 3 ∧
    defeat_points = 0 ∧
    total_matches = 20 ∧
    matches_played = 5 ∧
    points_scored = 14 ∧
    remaining_matches = total_matches - matches_played ∧
    wins_needed = 6 ∧
    target_points = 40 ∧
    points_scored + 6 * victory_points + (remaining_matches - wins_needed) * D = target_points ∧
    draw_points = D) →
    D = 2 :=
by
  sorry

end draw_points_worth_two_l799_799921


namespace bogatyrs_truthful_count_l799_799241

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l799_799241


namespace watch_A_accurate_l799_799536

variable (T : ℕ) -- Standard time, represented as natural numbers for simplicity
variable (A B : ℕ) -- Watches A and B, also represented as natural numbers
variable (h1 : A = B + 2) -- Watch A is 2 minutes faster than Watch B
variable (h2 : B = T - 2) -- Watch B is 2 minutes slower than the standard time

theorem watch_A_accurate : A = T :=
by
  -- The proof would go here
  sorry

end watch_A_accurate_l799_799536


namespace minimum_tangent_distance_l799_799734

theorem minimum_tangent_distance (x y : ℝ) 
  (h1 : (x-2)^2 + (y-2)^2 = 1) -- This is not used directly; it suggests M is on the tangent to circle
  (h2 : (x-2)^2 + (y-2)^2 - 1 = x^2 + y^2) -- This constraint comes directly from problem statement
  (h3 : | |x| + |y| |.toReal = |Math.sqrt (4^2 + 4^2) / 2| := by sorry) : -- h3 is the minimum distance condition
  | Math.sqrt (4^2 + 4^2) / 2 | = 7 * Math.sqrt(2) / 8 :=
sorry

end minimum_tangent_distance_l799_799734


namespace smallest_prime_with_digit_sum_23_l799_799640

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799640


namespace maximize_prism_volume_correct_l799_799831

noncomputable def maximize_prism_volume : ℝ :=
  let x := 1 in
  let height := 3.2 - 2 * x in
  let volume := -2*x^3 + 2.2*x^2 + 1.6*x in
  if x = 1 ∧ height = 1.2 ∧ volume = 1.8 then volume else 0

theorem maximize_prism_volume_correct :
  let x := 1 in
  let height := 3.2 - 2 * x in
  let volume := -2*x^3 + 2.2*x^2 + 1.6*x in
  x = 1 ∧ height = 1.2 ∧ volume = 1.8 :=
by
  sorry

end maximize_prism_volume_correct_l799_799831


namespace smallest_prime_with_digit_sum_23_l799_799596

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l799_799596


namespace greatest_integer_third_side_of_triangle_l799_799083

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end greatest_integer_third_side_of_triangle_l799_799083


namespace terminal_zeros_l799_799781

theorem terminal_zeros (a b c d : ℕ) : 
  100 = 2^a * 5^a ∧ 
  3600 = 2^b * 3^c * 5^d ∧ 
  a = 2 ∧ b = 4 ∧ c = 2 ∧ d = 2 →
  ∃ k, (100 * 3600 = 2^(a+b) * 3^c * 5^(a+d)) ∧ 
  k = min (a+b) (a+d) ∧ 
  k = 4 :=
begin
  intros h,
  cases h with ha hbcd,
  cases hbcd with hb hcd,
  cases hcd with hc hd,
  use min (a + b) (a + d),
  have H : 100 * 3600 = 2^(a+b) * 3^c * 5^(a+d),
  { rw [ha, hb] },
  split,
  { exact H },
  split,
  { refl },
  { sorry }
end

end terminal_zeros_l799_799781


namespace tyler_meal_combinations_l799_799549

theorem tyler_meal_combinations :
  let meat_choices := 4 in
  let vegetable_choices := Nat.choose 5 3 in
  let dessert_combo_choices := Nat.choose 4 2 in
  meat_choices * vegetable_choices * dessert_combo_choices = 240 :=
by
  let meat_choices := 4
  let vegetable_choices := Nat.choose 5 3
  let dessert_combo_choices := Nat.choose 4 2
  have h1 : meat_choices = 4 := by rfl
  have h2 : vegetable_choices = Nat.choose 5 3 := by rfl
  have h3 : dessert_combo_choices = Nat.choose 4 2 := by rfl
  calc
    meat_choices * vegetable_choices * dessert_combo_choices
        = 4 * 10 * 6 : by
          rw [h1, h2, h3]
          norm_num
    ... = 240 : by norm_num

end tyler_meal_combinations_l799_799549


namespace variance_transformed_data_l799_799311

variables {X : Type*} [fintype X] {f : X → ℝ}

-- Assume the variance of the original data
noncomputable def DX : ℝ := (∑ x, (f x)^2) / (fintype.card X) - ((∑ x, f x) / (fintype.card X))^2

-- Assume the variance is given as 1/2
axiom var_f : DX = 1 / 2

-- Consider the transformed data
noncomputable def g (x : X) : ℝ := 2 * f x - 5

-- Define the variance of the transformed data
noncomputable def DY : ℝ := (∑ x, (g x)^2) / (fintype.card X) - ((∑ x, g x) / (fintype.card X))^2

-- State the theorem to prove
theorem variance_transformed_data : DY = 2 :=
by
  sorry

end variance_transformed_data_l799_799311


namespace sum_coeff_expansion_x_minus_y_pow_10_l799_799277

theorem sum_coeff_expansion_x_minus_y_pow_10 : 
  let k3 := binomial 10 3
  let k7 := binomial 10 7
  -(k3 + k7) = -240 := by
  sorry

end sum_coeff_expansion_x_minus_y_pow_10_l799_799277


namespace max_digit_d_of_form_7d733e_multiple_of_33_l799_799262

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l799_799262


namespace inequality_solution_set_l799_799528

theorem inequality_solution_set : {x : ℝ | (x - 2) / x < 0} = set.Ioo 0 2 :=
by tidy

end inequality_solution_set_l799_799528


namespace probability_log3_integer_l799_799133

theorem probability_log3_integer :
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let valid_numbers := {m ∈ three_digit_numbers | ∃ k : ℕ, m = 3^k}
  ∃ p : ℚ, p = valid_numbers.card / three_digit_numbers.card ∧ p = 1 / 450 :=
begin
  sorry
end

end probability_log3_integer_l799_799133


namespace solve_for_r_l799_799900

variable (k r : ℝ)

theorem solve_for_r (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := sorry

end solve_for_r_l799_799900


namespace sum_of_powers_of_triples_l799_799053

-- Define the conditions and properties
def power_of_triple (x y z : ℕ) : ℕ :=
  if x ≥ y ∧ y ≥ z then x + z
  else if x ≥ z ∧ z ≥ y then x + y
  else if y ≥ x ∧ x ≥ z then y + z
  else if y ≥ z ∧ z ≥ x then y + x
  else if z ≥ x ∧ x ≥ y then z + y
  else z + x

def is_valid_triple (x y z : ℕ) : Prop := 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9

-- The final statement to be proved
theorem sum_of_powers_of_triples : 
  ( ∑ x in (Finset.range 10) ∑ y in (Finset.range 10) ∑ z in (Finset.range 10),
    if is_valid_triple x y z then power_of_triple x y z else 0 
  ) = 7290 :=
  sorry

end sum_of_powers_of_triples_l799_799053


namespace product_of_first_five_primes_ending_in_3_l799_799273

-- Define the prime numbers with a units digit of 3
def primes_ending_in_3 : List ℕ := [3, 13, 23, 43, 53]

-- Calculate their product
def product_of_primes_ending_in_3 : ℕ := primes_ending_in_3.foldl (*) 1

theorem product_of_first_five_primes_ending_in_3 :
  product_of_primes_ending_in_3 = 2042083 :=
by
  unfold product_of_primes_ending_in_3 primes_ending_in_3
  -- Manually expand and simplify the foldl calculation
  have h1 : 3 * 13 = 39 := by norm_num
  have h2 : 39 * 23 = 897 := by norm_num
  have h3 : 897 * 43 = 38571 := by norm_num
  have h4 : 38571 * 53 = 2042083 := by norm_num
  rw [h1, h2, h3, h4]
  rfl

end product_of_first_five_primes_ending_in_3_l799_799273


namespace union_of_sets_l799_799319

noncomputable def setA : Set ℝ := { x : ℝ | x^3 - 3 * x^2 - x + 3 < 0 }
noncomputable def setB : Set ℝ := { x : ℝ | abs (x + 1/2) >= 1 }

theorem union_of_sets:
  setA ∪ setB = (Iio (-1) ∪ Ici (1/2)) :=
by
  sorry

end union_of_sets_l799_799319


namespace smallest_prime_with_digit_sum_23_proof_l799_799679

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799679


namespace find_abc_integers_l799_799264

theorem find_abc_integers (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) 
(h4 : (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) : (a = 3 ∧ b = 5 ∧ c = 15) ∨ 
(a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end find_abc_integers_l799_799264


namespace girls_joined_l799_799398

theorem girls_joined (initial_girls : ℕ) (boys : ℕ) (girls_more_than_boys : ℕ) (G : ℕ) :
  initial_girls = 632 →
  boys = 410 →
  girls_more_than_boys = 687 →
  initial_girls + G = boys + girls_more_than_boys →
  G = 465 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end girls_joined_l799_799398


namespace alison_rice_steps_l799_799165

theorem alison_rice_steps (n : ℕ) (h : n = 2401) : 
  ∃ steps : ℕ, steps = 17 ∧ (∀ n > 1, 
  let d := Nat.find (λ d, d > 1 ∧ n % d = 0) in 
  ∃ remaining_n : ℕ, remaining_n = n * (d - 1) / d) :=
by 
  use 17
  split
  · rfl
  · contradict
    cases n, sorry

end alison_rice_steps_l799_799165


namespace prime_factors_of_x_l799_799307

theorem prime_factors_of_x (n : ℕ) (h1 : 2^n - 32 = x) (h2 : (nat.prime_factors x).length = 3) (h3 : 3 ∈ nat.prime_factors x) :
  x = 480 ∨ x = 2016 :=
sorry

end prime_factors_of_x_l799_799307


namespace locus_of_X_l799_799830
open EuclideanGeometry

noncomputable def is_locus_X (ABCD : Rectangle) (l : Line) : Prop :=
  ∀ X : Point, (AX + BX = CX + DX) ↔ X ∈ l

theorem locus_of_X (ABCD : Rectangle) (l : Line) (mid_BC_AD : passes_through_midpoints l ABCD B C A D):
  is_locus_X ABCD l :=
sorry

end locus_of_X_l799_799830


namespace hexadecagon_triangle_count_l799_799780

theorem hexadecagon_triangle_count :
  let n := 16 in
  ∃ k : ℕ, (nat.choose n 3 = k) ∧ k = 560 :=
by
  sorry

end hexadecagon_triangle_count_l799_799780


namespace part1_part2_part2_max_l799_799345

-- Define the functions f and g
def f (a x : ℝ) := (1/2) * x^2 + 2 * a * x
def g (a b x : ℝ) := 3 * a^2 * Real.log x + b

-- Define the derivatives of f and g
def f' (a x : ℝ) := x + 2 * a
def g' (a x : ℝ) := 3 * a^2 / x

-- Define the conditions
variable (a : ℝ)
variable (x0 b : ℝ)
variable (h : ∀ x > 0, f a x = g a b x ∧ f' a x = g' a x)

-- Prove b when a = 1
theorem part1 (h1 : a = 1) : b = 5 / 2 := sorry

-- Define b in terms of a and prove its maximum value
noncomputable def b_value (a : ℝ) := (5 / 2) * a^2 - 3 * a^2 * Real.log a

theorem part2 : b_value (exp (1 / 3)) = (3 / 2) * exp (2 / 3) := sorry

theorem part2_max : ∀ a > 0, b_value a ≤ (3 / 2) * exp (2 / 3) := sorry

end part1_part2_part2_max_l799_799345


namespace root_condition_implies_k_value_l799_799376

theorem root_condition_implies_k_value (k : ℝ) : 
  is_root (-1) (λ x, x^2 - k*x + 1) → k = -2 :=
by
  intro h
  have eq : (-1)^2 - k*(-1) + 1 = 0 := h
  have simplify : 1 + k + 1 = 0 := by simp [eq]
  have result : k + 2 = 0 := by linarith
  linarith
  sorry

end root_condition_implies_k_value_l799_799376


namespace digit_in_ten_thousandths_place_of_437_div_128_l799_799087

theorem digit_in_ten_thousandths_place_of_437_div_128 : 
  (∀ d, (437 : ℚ) / 128 = d → (if h : d = 3.4146875 then (finiteDecimalToDigit d 4 h) else 0) = 6) :=
sorry

end digit_in_ten_thousandths_place_of_437_div_128_l799_799087


namespace number_of_valid_four_digit_numbers_l799_799370

-- Definition for the problem
def is_valid_four_digit_number (n : ℕ) : Prop :=
  2999 < n ∧ n <= 9999 ∧
  (let d1 := n / 1000,
       d2 := (n / 100) % 10,
       d3 := (n / 10) % 10,
       d4 := n % 10 in
   3 <= d1 ∧ d1 <= 9 ∧
   0 <= d4 ∧ d4 <= 9 ∧
   d2 * d3 > 10)

-- Statement of the problem
theorem number_of_valid_four_digit_numbers : 
  (Finset.range 10000).filter is_valid_four_digit_number).card = 4830 :=
sorry

end number_of_valid_four_digit_numbers_l799_799370


namespace greatest_integer_third_side_of_triangle_l799_799081

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end greatest_integer_third_side_of_triangle_l799_799081


namespace cone_max_cross_section_area_correct_l799_799717

noncomputable def cone_max_cross_section_area {r l θ a : ℝ}
  (h_cone_radius : r = 5 / 3)
  (h_generatrix : l = 2)
  (h_central_angle : θ = 5 * π / 3)
  (h_trace_range : 0 < a ∧ a ≤ 10 / 3) :
  ℝ :=
  let h := √(4 - a^2 / 4) in
  let S := 1 / 2 * a * h in
  S

theorem cone_max_cross_section_area_correct {r l θ a : ℝ}
  (h_cone_radius : r = 5 / 3)
  (h_generatrix : l = 2)
  (h_central_angle : θ = 5 * π / 3)
  (h_trace_range : 0 < a ∧ a ≤ 10 / 3) :
  cone_max_cross_section_area h_cone_radius h_generatrix h_central_angle h_trace_range ≤ 2 :=
  sorry

end cone_max_cross_section_area_correct_l799_799717


namespace closest_wedge_volume_to_603_l799_799428

-- Define the given conditions
variables (r h : ℝ)
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
def wedge_volume (r h : ℝ) : ℝ := cylinder_volume r h / 2

-- Main statement
theorem closest_wedge_volume_to_603 :
  let r := 6 in
  let h := 6 in
  abs (wedge_volume r h - 603) < abs (wedge_volume r h - 48) ∧
  abs (wedge_volume r h - 603) < abs (wedge_volume r h - 75) ∧
  abs (wedge_volume r h - 603) < abs (wedge_volume r h - 151) ∧
  abs (wedge_volume r h - 603) < abs (wedge_volume r h - 192) := by
  sorry

end closest_wedge_volume_to_603_l799_799428


namespace probability_l799_799394

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def probability_of_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem probability : probability_of_different_colors = 148 / 225 :=
by
  unfold probability_of_different_colors
  sorry

end probability_l799_799394


namespace truthfulness_count_l799_799223

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799223


namespace incorrect_conclusion_l799_799354

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l799_799354


namespace smallest_prime_with_digit_sum_23_l799_799581

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799581


namespace smallest_positive_angle_tangent_identity_l799_799768

theorem smallest_positive_angle_tangent_identity :
  ∃ x : ℝ, x > 0 ∧ x < 180 ∧ tan (2 * x * Real.pi / 180) = (cos (x * Real.pi / 180) + sin (x * Real.pi / 180)) / (cos (x * Real.pi / 180) - sin (x * Real.pi / 180)) ∧ x = 15 :=
by
  sorry

end smallest_positive_angle_tangent_identity_l799_799768


namespace digits_of_2048_in_base_9_l799_799894

def digits_base9 (n : ℕ) : ℕ :=
if n < 9 then 1 else 1 + digits_base9 (n / 9)

theorem digits_of_2048_in_base_9 : digits_base9 2048 = 4 :=
by sorry

end digits_of_2048_in_base_9_l799_799894


namespace ratio_of_screams_to_hours_l799_799007

-- Definitions from conditions
def hours_hired : ℕ := 6
def current_babysitter_rate : ℕ := 16
def new_babysitter_rate : ℕ := 12
def extra_charge_per_scream : ℕ := 3
def cost_difference : ℕ := 18

-- Calculate necessary costs
def current_babysitter_cost : ℕ := current_babysitter_rate * hours_hired
def new_babysitter_base_cost : ℕ := new_babysitter_rate * hours_hired
def new_babysitter_total_cost : ℕ := current_babysitter_cost - cost_difference
def screams_cost : ℕ := new_babysitter_total_cost - new_babysitter_base_cost
def number_of_screams : ℕ := screams_cost / extra_charge_per_scream

-- Theorem to prove the ratio
theorem ratio_of_screams_to_hours : number_of_screams / hours_hired = 1 := by
  sorry

end ratio_of_screams_to_hours_l799_799007


namespace smallest_prime_with_digit_sum_23_l799_799639

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l799_799639


namespace distance_focus_vertex_parabola_l799_799828

theorem distance_focus_vertex_parabola 
  (C : Type*)
  (F : Point)
  (axis_of_symmetry : Line)
  (l : Line)
  (h1 : passes_through l F)
  (h2 : angle_between l axis_of_symmetry = π / 4)
  (h3 : chord_length l C = 4) : 
  distance F vertex = 1/2 :=
sorry

end distance_focus_vertex_parabola_l799_799828


namespace limit_difference_interval_l799_799779

variable (f : ℝ → ℝ)

-- Conditions
def strictly_decreasing_continuous : Prop := 
  (∀ x y : ℝ, x < y → f x > f y) ∧ continuous f

def functional_equation_satisfied : Prop := 
  ∀ x : ℝ, (f (f x))^4 - f (f x) + f x = 1

-- Statement
theorem limit_difference_interval 
  (h1 : strictly_decreasing_continuous f) 
  (h2 : functional_equation_satisfied f) : 
  ∃ l u : ℝ, l = 0 ∧ 
  u = 27/64 + 27/(16*(4^(1/3))) + 3153/(1024*(4^(2/3))) ∧
  ∀ M m, M = lim (λ x : ℝ, f x) at_bot ∧ 
          m = lim (λ x : ℝ, f x) at_top → 
          (M - m) ∈ set.Ioo l u :=
sorry

end limit_difference_interval_l799_799779


namespace binom_10_0_equals_1_l799_799181

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem to prove that binom 10 0 = 1
theorem binom_10_0_equals_1 :
  binom 10 0 = 1 := by
  sorry

end binom_10_0_equals_1_l799_799181


namespace proof_problem_l799_799816

theorem proof_problem
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^2 + b^2 - a * b = 2)
  (h4 : |a^2 - b^2| ≤ 2) :
  (a - b ≤ Real.sqrt (6) / 3) ∧
  (Real.log2 a + Real.log2 b ≤ 1) ∧
  (Real.log2 a + Real.log2 (3 * b) ≥ 2) :=
  by
    sorry

end proof_problem_l799_799816


namespace minimum_value_of_function_l799_799384

theorem minimum_value_of_function (x : ℝ) (h : x * Real.log 2 / Real.log 3 ≥ 1) : 
  ∃ t : ℝ, t = 2^x ∧ t ≥ 3 ∧ ∀ y : ℝ, y = t^2 - 2*t - 3 → y = (t-1)^2 - 4 := 
sorry

end minimum_value_of_function_l799_799384


namespace truthfulness_count_l799_799225

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799225


namespace max_d_77733e_divisible_by_33_l799_799260

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l799_799260


namespace emails_left_in_inbox_l799_799488

noncomputable def initialEmails : ℕ := 400
noncomputable def emailsMovedToTrash (n : ℕ) : ℕ := n / 2
noncomputable def remainingAfterTrash (n : ℕ) : ℕ := n - emailsMovedToTrash(n)
noncomputable def movedToWorkFolder (remaining : ℕ) : ℕ := (40 * remaining) / 100
noncomputable def remainingInInbox (n : ℕ) : ℕ :=
  remainingAfterTrash(n) - movedToWorkFolder(remainingAfterTrash(n))

theorem emails_left_in_inbox : remainingInInbox(initialEmails) = 120 :=
  sorry

end emails_left_in_inbox_l799_799488


namespace integer_solutions_no_solutions_2891_l799_799475

-- Define the main problem statement
-- Prove that if the equation x^3 - 3xy^2 + y^3 = n has a solution in integers x, y, then it has at least three such solutions.
theorem integer_solutions (n : ℕ) (x y : ℤ) (h : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x₁ y₁ x₂ y₂ : ℤ, x₁ ≠ x ∧ y₁ ≠ y ∧ x₂ ≠ x ∧ y₂ ≠ y ∧ 
  x₁^3 - 3 * x₁ * y₁^2 + y₁^3 = n ∧ 
  x₂^3 - 3 * x₂ * y₂^2 + y₂^3 = n := sorry

-- Prove that if n = 2891 then no such integer solutions exist.
theorem no_solutions_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) := sorry

end integer_solutions_no_solutions_2891_l799_799475


namespace enterprise_b_pays_more_in_2015_l799_799168

variable (a b x y : ℝ)
variable (ha2x : a + 2 * x = b)
variable (ha1y : a * (1+y)^2 = b)

theorem enterprise_b_pays_more_in_2015 : b * (1 + y) > b + x := by
  sorry

end enterprise_b_pays_more_in_2015_l799_799168


namespace smallest_prime_with_digit_sum_23_l799_799607

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799607


namespace midpoint_property_l799_799472

variables {A B C N M K : Type}
variables [metric_space A] [metric_space B] [metric_space C] 
variables [metric_space N] [metric_space M] [metric_space K] 

noncomputable def is_midpoint (N : metric_space) (B C : metric_space) : Prop :=
dist N B = dist N C

noncomputable def angle_eq (A B C : Type) (θ : ℝ) : Prop :=
angle A B C = θ

noncomputable def dist_eq (X Y Z : metric_space) : Prop :=
dist X Y = dist Y Z

theorem midpoint_property (h1 : is_midpoint N B C)
    (h2 : angle_eq A C B (π / 3))
    (h3 : dist_eq A M N)
    (h4 : is_midpoint K B M) :
  dist A K = dist K C :=
sorry

end midpoint_property_l799_799472


namespace smallest_prime_with_digit_sum_23_l799_799652

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799652


namespace percent_volume_filled_by_cubes_l799_799139

theorem percent_volume_filled_by_cubes : 
  let box_volume := 8 * 6 * 12,
      cube_volume := 4 * 4 * 4,
      max_cubes := (8 / 4) * (6 / 4) * (12 / 4),
      total_cube_volume := max_cubes * cube_volume in
  (total_cube_volume / box_volume) * 100 = 66.67 := 
sorry

end percent_volume_filled_by_cubes_l799_799139


namespace warriors_truth_tellers_l799_799205

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799205


namespace eval_expr_l799_799907

theorem eval_expr (x y : ℕ) (h1 : x = 2) (h2 : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end eval_expr_l799_799907


namespace distance_from_point_to_xOy_plane_l799_799930

theorem distance_from_point_to_xOy_plane (x y z : ℝ) (h : (x, y, z) = (1, -3, 2)) : abs z = 2 := 
by
  have h₁ : z = 2 := by
    rw [Prod.mk.inj_iff] at h
    exact h.2.2
  rw [h₁]
  norm_num

end distance_from_point_to_xOy_plane_l799_799930


namespace complex_root_problem_l799_799275

theorem complex_root_problem (z : ℂ) :
  z^2 - 3*z = 10 - 6*Complex.I ↔
  z = 5.5 - 0.75 * Complex.I ∨
  z = -2.5 + 0.75 * Complex.I ∨
  z = 3.5 - 1.5 * Complex.I ∨
  z = -0.5 + 1.5 * Complex.I :=
sorry

end complex_root_problem_l799_799275


namespace find_prime_pairs_l799_799268

open Nat

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def valid_prime_pairs (p q : ℕ): Prop :=
  is_prime p ∧ is_prime q ∧ divides p (30 * q - 1) ∧ divides q (30 * p - 1)

theorem find_prime_pairs :
  { (p, q) | valid_prime_pairs p q } = { (7, 11), (11, 7), (59, 61), (61, 59) } :=
sorry

end find_prime_pairs_l799_799268


namespace percent_less_50000_l799_799926

variable (A B C : ℝ) -- Define the given percentages
variable (h1 : A = 0.45) -- 45% of villages have populations from 20,000 to 49,999
variable (h2 : B = 0.30) -- 30% of villages have fewer than 20,000 residents
variable (h3 : C = 0.25) -- 25% of villages have 50,000 or more residents

theorem percent_less_50000 : A + B = 0.75 := by
  sorry

end percent_less_50000_l799_799926


namespace set_intersection_l799_799882

theorem set_intersection (U : Set ℝ) (M N : Set ℝ) (complement_N : Set ℝ) :
  U = Set.univ →
  M = {x : ℝ | x^2 - 2 * x < 0} →
  N = {x : ℝ | x > 1} →
  complement_N = {x : ℝ | x ≤ 1} →
  (M ∩ complement_N) = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by 
  intro hU hM hN hcomplN
  rw [hM, hN, hcomplN, Set.inter_def, Set.mem_set_of_eq]
  sorry

end set_intersection_l799_799882


namespace cost_per_km_of_new_energy_car_calculate_a_and_cost_l799_799009

-- Define the conditions as constants and functions
def tank_capacity_fuel_car := 40
def fuel_price_per_liter := 9
def range_fuel_car (a : ℕ) := 2 * a
def battery_capacity_new_energy_car := 60
def electricity_price_per_kwh := 0.6
def range_new_energy_car (a : ℕ) := a
def cost_per_km_fuel_car (a : ℕ) := (tank_capacity_fuel_car * fuel_price_per_liter) / (range_fuel_car a)
def cost_per_km_new_energy_car (a : ℕ) := (battery_capacity_new_energy_car * electricity_price_per_kwh) / (range_new_energy_car a)

-- Theorem 1: Prove the cost per kilometer of the new energy car
theorem cost_per_km_of_new_energy_car (a : ℕ) : cost_per_km_new_energy_car a = 36 / a := by
  simp [cost_per_km_new_energy_car]
  sorry

-- Theorem 2: Calculate a and cost per kilometer for each car
theorem calculate_a_and_cost (a : ℕ) :
  (cost_per_km_fuel_car a) = (cost_per_km_new_energy_car a) + 0.48
  → a = 300
  → cost_per_km_fuel_car a = 0.6
  → cost_per_km_new_energy_car a = 0.12 := by
  intros h1 h2
  rw [h2] at h1
  simp [cost_per_km_fuel_car, cost_per_km_new_energy_car] at *
  sorry


end cost_per_km_of_new_energy_car_calculate_a_and_cost_l799_799009


namespace incorrect_conclusion_D_l799_799356

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l799_799356


namespace largest_three_digit_multiple_of_12_with_digit_sum_24_l799_799558

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 12 = 0 ∧ (n.digits 10).sum = 24 ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 12 = 0 ∧ (m.digits 10).sum = 24 → m ≤ n) ∧ n = 888 :=
by {
  sorry -- Proof to be filled in
}

#eval largest_three_digit_multiple_of_12_with_digit_sum_24 -- Should output: ⊤ (True)

end largest_three_digit_multiple_of_12_with_digit_sum_24_l799_799558


namespace sin2x_plus_cos2y_range_l799_799813

theorem sin2x_plus_cos2y_range (x y : ℝ) (h : 2 * (sin x)^2 + (cos y)^2 = 1) : 
  ∃ a b : ℝ, 0 ≤ a ∧ a ≤ 1 / 2 ∧ b = 1 - a ∧ (sin x)^2 + (cos y)^2 = b ∧ 1 / 2 ≤ b ∧ b ≤ 1 :=
sorry

end sin2x_plus_cos2y_range_l799_799813


namespace angle_B_shape_triangle_l799_799424

variable {a b c R : ℝ} 

theorem angle_B_shape_triangle 
  (h1 : c > a ∧ c > b)
  (h2 : b = Real.sqrt 3 * R)
  (h3 : b * Real.sin (Real.arcsin (b / (2 * R))) = (a + c) * Real.sin (Real.arcsin (a / (2 * R)))) :
  (Real.arcsin (b / (2 * R)) = Real.pi / 3 ∧ a = c / 2 ∧ Real.arcsin (a / (2 * R)) = Real.pi / 6 ∧ Real.arcsin (c / (2 * R)) = Real.pi / 2) :=
by
  sorry

end angle_B_shape_triangle_l799_799424


namespace battery_replacement_in_month_15th_l799_799891

theorem battery_replacement_in_month_15th : ∀ n : ℕ, (n = 7 * 14 + 2) → (n % 12 = 2) :=
by
  intro n h
  rw h
  norm_num
  sorry -- Placeholder for actual proof

end battery_replacement_in_month_15th_l799_799891


namespace parabola_focus_contribution_l799_799129

structure Parabola :=
  (a : ℝ)
  (focus : ℝ × ℝ)
  (directrix : ℝ)

def C : Parabola := { a := 4, focus := (1, 0), directrix := -1 }

noncomputable def intersection_points (l : ℝ → ℝ) (p : Parabola) : ℝ × ℝ :=
  let k := l in
  let x := classical.some (exists_pair_mem_set_univ $
                  fun x => (k * (x - 1)) ^ 2 = p.a * x) in
  (x, k * (x - 1))

def AF (A : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := abs (A.1 + 1)

def BF (B : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := abs (B.1 + 1)

theorem parabola_focus_contribution : 
  ∀ l : ℝ → ℝ, let A := intersection_points l C, let B := intersection_points l C in
  1 / AF A C.focus + 1 / BF B C.focus = 1 :=
by
  intro l A B
  sorry

end parabola_focus_contribution_l799_799129


namespace linear_regression_prediction_expected_value_X_l799_799792

noncomputable def x_values : List ℕ := [1, 2, 3, 4, 5, 6, 7]
noncomputable def y_values : List ℕ := [29, 33, 36, 44, 48, 52, 59]

def x_mean := (List.sum x_values) / (x_values.length)
def y_mean := (List.sum y_values) / (y_values.length)

def b_hat := (List.sum (List.map (λ i => (x_values[i-1] - x_mean) * (y_values[i-1] - y_mean)) (List.range 1 8))) / 
             (List.sum (List.map (λ i => (x_values[i-1] - x_mean) ^ 2) (List.range 1 8)))

def a_hat := y_mean - b_hat * x_mean

def y_estimated (x : ℕ) := b_hat * x + a_hat

theorem linear_regression_prediction : y_estimated 8 = 63 := sorry

def prob_A := 2 / 3
def prob_B := 2 / 3
def prob_C := 2 / 3
def prob_D := 1 / 3

def X := List.sum [prob_A, prob_B, prob_C, prob_D]

theorem expected_value_X : X = 7 / 3 := sorry

end linear_regression_prediction_expected_value_X_l799_799792


namespace find_CD_l799_799280

theorem find_CD : ∃ (C D : ℝ), (C + D = 5) ∧ ∀ x : ℝ, ((Dx - 23) / (x^2 - 9x + 20) = (C / (x - 4)) + (7 / (x - 5))) := by
  sorry

end find_CD_l799_799280


namespace constant_term_expansion_l799_799903

noncomputable def a : ℝ := ∫ x in -1..1, sqrt (1 - x^2)

theorem constant_term_expansion : 
    (a = (π / 2)) → 
    (∀ x : ℝ, constant_term (pow (add (div (mul a x) π) (div (-1) x)) 6) = - (5 / 2)) :=
begin
  sorry
end

end constant_term_expansion_l799_799903


namespace minimum_disks_needed_l799_799968

-- Definition of the conditions
def disk_capacity : ℝ := 2.88
def file_sizes : List (ℝ × ℕ) := [(1.2, 5), (0.9, 10), (0.6, 8), (0.3, 7)]

/-- 
Theorem: Given the capacity of each disk and the sizes and counts of different files,
we can prove that the minimum number of disks needed to store all the files without 
splitting any file is 14.
-/
theorem minimum_disks_needed (capacity : ℝ) (files : List (ℝ × ℕ)) : 
  capacity = disk_capacity ∧ files = file_sizes → ∃ m : ℕ, m = 14 :=
by
  sorry

end minimum_disks_needed_l799_799968


namespace partition_even_odd_l799_799526

theorem partition_even_odd (R B : Set ℕ) (h_partition : ∀ n, n ∈ R ∨ n ∈ B)
  (h_disjoint : ∀ n, ¬ (n ∈ R ∧ n ∈ B))
  (h_infinite_R : Set.Infinite R) (h_infinite_B : Set.Infinite B)
  (h_sum_2023_R : ∀ s ⊆ R, s.card = 2023 → s.sum id ∈ R)
  (h_sum_2023_B : ∀ s ⊆ B, s.card = 2023 → s.sum id ∈ B) :
  (∀ n ∈ R, n % 2 = 1) ∧ (∀ n ∈ B, n % 2 = 0) ∨ (∀ n ∈ R, n % 2 = 0) ∧ (∀ n ∈ B, n % 2 = 1) :=
sorry

end partition_even_odd_l799_799526


namespace max_value_of_d_l799_799257

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end max_value_of_d_l799_799257


namespace percentage_increase_output_per_hour_with_assistant_l799_799700

-- Definition of variables
variables (B H : ℝ)
-- Definition of outputs without and with assistant
def output_per_hour_without_assistant : ℝ := B / H
def output_per_hour_with_assistant : ℝ := (1.80 * B) / (0.90 * H)
-- Definition of percentage increase
def percentage_increase (old new : ℝ) : ℝ := ((new - old) / old) * 100

-- Theorem statement
theorem percentage_increase_output_per_hour_with_assistant : percentage_increase (output_per_hour_without_assistant B H) (output_per_hour_with_assistant B H) = 100 := 
by sorry

end percentage_increase_output_per_hour_with_assistant_l799_799700


namespace graphs_symmetric_about_y_eq_x_l799_799422

open Function

theorem graphs_symmetric_about_y_eq_x :
  ∀ x, 2^x = log 2 (2^x) ↔ x = x :=
by
  sorry

end graphs_symmetric_about_y_eq_x_l799_799422


namespace radius_of_spheres_in_pyramid_l799_799704

theorem radius_of_spheres_in_pyramid
  (h : ℝ) (a : ℝ) (h_value : h = 5 / 4) (a_value : a = Real.sqrt 15)
  (spheres_count : ℕ) (center_sphere : Prop) 
  (side_spheres : Prop) (fifth_sphere : Prop) 
  (center_sphere_touch : Prop) 
  (side_spheres_touch : Prop) 
  (fifth_sphere_touch : Prop) :
  (∃ R : ℝ, R = 1 / 6) :=
by
  have hd := sqrt 15
  sorry -- proof to be filled in later

end radius_of_spheres_in_pyramid_l799_799704


namespace complex_solution_l799_799290

noncomputable def complex_z (z : ℂ) : Prop :=
  z^2 = -4 ∧ z.im > 0

theorem complex_solution (z : ℂ) (h : complex_z z) : z = 2 * complex.I :=
by
  sorry

end complex_solution_l799_799290


namespace chameleons_impossible_all_white_l799_799468

/--
On Easter Island, there are initial counts of blue (12), white (25), and red (8) chameleons.
When two chameleons of different colors meet, they both change to the third color.
Prove that it is impossible for all chameleons to become white.
--/
theorem chameleons_impossible_all_white :
  let n1 := 12 -- Blue chameleons
  let n2 := 25 -- White chameleons
  let n3 := 8  -- Red chameleons
  (∀ (n1 n2 n3 : ℕ), (n1 + n2 + n3 = 45) → 
   ∀ (k : ℕ), ∃ m1 m2 m3 : ℕ, (m1 - m2) % 3 = (n1 - n2) % 3 ∧ (m1 - m3) % 3 = (n1 - n3) % 3 ∧ 
   (m2 - m3) % 3 = (n2 - n3) % 3) → False := sorry

end chameleons_impossible_all_white_l799_799468


namespace count_correct_statements_l799_799523

theorem count_correct_statements :
  let statement1 := ∀ θ: ℝ, θ < 90 ∧ (θ > 0) → θ < 90 → false /* angle is not necessarily acute */
  let statement2 := ∀ θ: ℝ, (90 < θ ∧ θ < 180) → θ > 0 → false /* obtuse angle not always greater than first quadrant angle */
  let statement3 := ∀ θ1 θ2: ℝ, (90< θ1 ∧  θ1 ≤ 180 ∧ 0 < θ2 ∧  θ2 < 90) → θ1 > θ2 → false /* not always true for second quadrant angle */
  let statement4 := ∀ θ: ℝ, (θ = k * 360 → θ = 0) → false /* angle coinciding with its terminal side is 0 (false as stated) */
  in 0 = 0 := by
  -- conditions are given
  let statement1 := ∀ θ: ℝ, θ < 90 ∧ (θ > 0) → θ < 90 → false
  let statement2 := ∀ θ: ℝ, (90 < θ ∧ θ < 180) → θ > 0 → false
  let statement3 := ∀ θ1 θ2: ℝ, (90< θ1 ∧  θ1 ≤180 ∧ 0< θ2 ∧  θ2 <90) → θ1 > θ2 → false
  let statement4 := ∀ θ: ℝ, (θ = k * 360 → θ = 0) → false

  -- indicating there are 0 valid statements
  exact rfl

end count_correct_statements_l799_799523


namespace rice_mixture_ratio_l799_799427

theorem rice_mixture_ratio (x y z : ℕ) (h : 16 * x + 24 * y + 30 * z = 18 * (x + y + z)) : 
  x = 9 * y + 18 * z :=
by
  sorry

end rice_mixture_ratio_l799_799427


namespace pizza_left_for_Wally_l799_799085

theorem pizza_left_for_Wally (a b c : ℚ) (ha : a = 1/3) (hb : b = 1/6) (hc : c = 1/4) :
  1 - (a + b + c) = 1/4 :=
by
  sorry

end pizza_left_for_Wally_l799_799085


namespace reflection_matrix_determine_l799_799870

theorem reflection_matrix_determine (a b : ℚ)
  (h1 : (a^2 - (3/4) * b) = 1)
  (h2 : (-(3/4) * b + (1/16)) = 1)
  (h3 : (a * b + (1/4) * b) = 0)
  (h4 : (-(3/4) * a - (3/16)) = 0) :
  (a, b) = (1/4, -5/4) := 
sorry

end reflection_matrix_determine_l799_799870


namespace solution_set_of_inequality_l799_799836

noncomputable def is_solution_set (f : ℝ → ℝ) : set ℝ :=
  {x | x > 1 ∨ x < -1 ∨ x = 0}

theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (hf_even : ∀ x, f x = f (-x)) 
  (hf_deriv : ∀ x, 2 * f x + x * (deriv f x) > 6)
  (hf_at_1 : f 1 = 2)
  : {x | x^2 * f x > 3 * x^2 - 1} = is_solution_set f :=
by
  sorry

end solution_set_of_inequality_l799_799836


namespace max_n_value_l799_799429

noncomputable def max_n_avoid_repetition : ℕ :=
sorry

theorem max_n_value : max_n_avoid_repetition = 155 :=
by
  -- Assume factorial reciprocals range from 80 to 99
  -- We show no n-digit segments are repeated in such range while n <= 155
  sorry

end max_n_value_l799_799429


namespace find_abc_integers_l799_799265

theorem find_abc_integers (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) 
(h4 : (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) : (a = 3 ∧ b = 5 ∧ c = 15) ∨ 
(a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end find_abc_integers_l799_799265


namespace volume_percentage_l799_799146

-- Definitions of the initial conditions
def box_length : ℝ := 8
def box_width : ℝ := 6
def box_height : ℝ := 12
def cube_side : ℝ := 4

-- Definition for the correct answer
def correct_answer : ℝ := 66.67

-- The Lean 4 statement to express and prove the given problem
theorem volume_percentage :
  let box_volume := box_length * box_width * box_height,
      cube_volume := cube_side ^ 3,
      num_cubes := (box_length / cube_side).to_int * (box_width / cube_side).to_int * (box_height / cube_side).to_int,
      cubes_volume := num_cubes * cube_volume,
      volume_percentage := (cubes_volume / box_volume) * 100 in
  volume_percentage = correct_answer :=
by
  sorry

end volume_percentage_l799_799146


namespace number_of_truthful_warriors_l799_799206

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799206


namespace prime_iff_sum_four_distinct_products_l799_799434

variable (n : ℕ) (a b c d : ℕ)

theorem prime_iff_sum_four_distinct_products (h : n ≥ 5) :
  (Prime n ↔ ∀ (a b c d : ℕ), n = a + b + c + d → a > 0 → b > 0 → c > 0 → d > 0 → ab ≠ cd) :=
sorry

end prime_iff_sum_four_distinct_products_l799_799434


namespace eval_expression_l799_799756

-- First, define the conditions as Lean definitions
def sqrt_2 : ℝ := real.sqrt 2
def sqrt3_4 : ℝ := real.nth_root 3 4
def sqrt6_32 : ℝ := real.nth_root 6 32
def log_10 (x : ℝ) : ℝ := real.log x / real.log 10
def log_rule (a b : ℝ) : ℝ := real.log a ^ b
def lg_1_over_100 : ℝ := -2
def pow_log_base_3_of_2 : ℝ := 2

-- Now, state the theorem to prove that the expression evaluates to 0
theorem eval_expression : 
  (2^(1/2) * 2^(2/3) * 2^(5/6) + log_10 (1/100) - 3^(real.log 2 / real.log 3)) = 0 :=
by
  -- Use sorry to skip the proof
  sorry

end eval_expression_l799_799756


namespace f_recurrence_l799_799456

noncomputable def f (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem f_recurrence (n : ℕ) : f (n + 1) - f (n - 1) = (3 * Real.sqrt 7 / 14) * f n := 
  sorry

end f_recurrence_l799_799456


namespace not_always_possible_to_reassemble_l799_799016

noncomputable def cut_polyhedron_edges (polyhedron : Type) : set (set Type) := sorry
noncomputable def fold_faces_in_envelope (faces : set (set Type)) : Type := sorry
noncomputable def reassemble_polyhedron (faces : set (set Type)) : Type := sorry

theorem not_always_possible_to_reassemble (polyhedron : Type) :
  polyhedron →
  let faces := cut_polyhedron_edges polyhedron in
  let envelope := fold_faces_in_envelope faces in
  ∃ (vanya_polyhedron : Type), vanya_polyhedron ≠ polyhedron :=
sorry

end not_always_possible_to_reassemble_l799_799016


namespace period_of_f_l799_799843

theorem period_of_f (f : ℤ → ℂ) (h : ∀ x, f(x + 2) = -f(x)) : 
  ∀ x, f(x + 4) = f(x) :=
by
  sorry

end period_of_f_l799_799843


namespace complete_square_solution_l799_799507

noncomputable def x : ℝ := sorry

theorem complete_square_solution :
  ∃ (a' b' : ℕ), (a' = 145 ∧ b' = 7) ∧ (x = real.sqrt a' - b' ∧ a' + b' = 152) 
:=
begin
  -- Condition: The equation has two solutions
  have h1 : ∀ (x : ℝ), (x^2 + 14*x = 96) ↔ (x = real.sqrt 145 - 7 ∨ x = -real.sqrt 145 - 7), sorry,

  -- Condition: a' and b' are positive natural numbers
  let a' := 145,
  let b' := 7,

  -- Positive solution is sqrt(a') - b'
  have h2 : x = real.sqrt a' - b' ∧ a' + b' = 152, sorry,

  use [a', b'],
  split,
  { exact ⟨rfl, rfl⟩ },
  exact h2,
end

end complete_square_solution_l799_799507


namespace dot_path_length_l799_799718

def cube_edge_length : ℝ := 2
def dot_initial_position : ℝ := 0  -- Assume a position for simplicity, the specific movement matters here

-- The main theorem statement proving that the length of the path traced by the dot is 2√2π
theorem dot_path_length :
  let r := (cube_edge_length * Real.sqrt 2) / 2 in
  let total_path := 4 * (r * π / 2) in
  total_path = 2 * Real.sqrt 2 * π :=
by
  sorry

end dot_path_length_l799_799718


namespace binom_10_0_eq_1_l799_799183

theorem binom_10_0_eq_1 :
  (Nat.choose 10 0) = 1 :=
by
  sorry

end binom_10_0_eq_1_l799_799183


namespace evaluate_f_sum_l799_799861

noncomputable def f : ℝ → ℝ :=
λ x, if x < 1 then 1 + Real.log 2 (2 - x)
     else 2^(x - 1)

theorem evaluate_f_sum : f (-2) + f (Real.log2 12) = 9 :=
by {
    -- sorry, skipped proof steps as instructed
    sorry
}

end evaluate_f_sum_l799_799861


namespace football_players_configuration_l799_799120

-- Define the conditions: Each player can pass the ball to exactly 4 other players
def pass_possible (players : Finset ℕ) (passes : ℕ → ℕ → Prop) : Prop :=
  (∀ p ∈ players, ∃ q r s t ∈ players, passes p q ∧ passes p r ∧ passes p s ∧ passes p t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)

-- The statement asserts that there exists a configuration of 6 players where each player can pass to exactly 4 other players without obstruction.
theorem football_players_configuration : 
  ∃ (players : Finset ℕ) (passes : ℕ → ℕ → Prop),
    players.card = 6 ∧ pass_possible players passes := 
sorry

end football_players_configuration_l799_799120


namespace remove_red_balls_l799_799400

theorem remove_red_balls (total_balls : ℕ) (initial_red_percentage : ℝ) (target_red_percentage : ℝ) (total_red : ℕ) (total_blue : ℕ) (remaining_balls : ℕ) (remaining_red_percentage : ℝ):
  total_balls = 600 ∧ initial_red_percentage = 0.7 ∧ target_red_percentage = 0.6 ∧ 
  total_red = 420 ∧ total_blue = 180 ∧ remaining_balls = total_balls - 150 ∧ remaining_red_percentage = 0.6
  → let red_balls_removed := total_red - remaining_red_percentage * remaining_balls in 
    red_balls_removed = 150 :=
begin
  sorry
end

end remove_red_balls_l799_799400


namespace poly_nonzero_coeff_l799_799979

theorem poly_nonzero_coeff (P : Polynomial ℝ) (n : ℕ) (hP : P ≠ 0) (hn : n ≥ 1) :
  (Polynomial.C (x : Polynomial ℝ) + 1)^((n : ℕ) - 1) * P).coeffs.count (≠ 0) ≥ n :=
sorry

end poly_nonzero_coeff_l799_799979


namespace truthful_warriors_count_l799_799217

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l799_799217


namespace skill_position_players_wait_l799_799121

theorem skill_position_players_wait
  (num_linemen : ℕ) (drink_per_linemen : ℕ) 
  (num_skill_players : ℕ) (drink_per_skill_player : ℕ) 
  (total_water : ℕ) : ℕ :=
  (num_linemen = 12) →
  (drink_per_linemen = 8) →
  (num_skill_players = 10) →
  (drink_per_skill_player = 6) →
  (total_water = 126) →
  num_skill_players - (total_water - num_linemen * drink_per_linemen) / drink_per_skill_player = 5 := sorry

end skill_position_players_wait_l799_799121


namespace completing_the_square_step_l799_799686

theorem completing_the_square_step (x : ℝ) : 
  x^2 + 4 * x + 2 = 0 → x^2 + 4 * x = -2 :=
by
  intro h
  sorry

end completing_the_square_step_l799_799686


namespace smallest_prime_with_digit_sum_23_proof_l799_799676

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l799_799676


namespace expected_area_projection_l799_799271

noncomputable def expectedProjectionArea (cube_edge : ℝ) : ℝ := 
  3 / 2

theorem expected_area_projection (h : ∀ (c : ℝ), c = 1) : expectedProjectionArea 1 = 3 / 2 :=
by 
  rw expectedProjectionArea 
  simp
  sorry

end expected_area_projection_l799_799271


namespace quadratic_range_l799_799916

theorem quadratic_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 1) :=
by
  sorry

end quadratic_range_l799_799916


namespace smallest_prime_with_digit_sum_23_l799_799667

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l799_799667


namespace number_on_board_after_60_minutes_l799_799013

-- Define the transformation on the number based on the problem's rule
def transform (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2 + 12

-- Define the sequence of transformations starting from the initial number
def sequence (n : ℕ) : ℕ -> ℕ
  | 0     => n
  | (k+1) => transform (sequence n k)

-- The main theorem to prove
theorem number_on_board_after_60_minutes :
  sequence 27 60 = 14 :=
sorry -- proof omitted

end number_on_board_after_60_minutes_l799_799013


namespace reflection_point_lies_on_planes_l799_799478

noncomputable def tetrahedron := 
  { O : Point, -- center of the circumscribed sphere
    S : Point, -- centroid of the tetrahedron
    vertices : List Point, 
    midpoints : List Point, 
    planes : List Plane
   }

theorem reflection_point_lies_on_planes
  (O S T : Point)
  (vertices midpoints : List Point)
  (planes : List Plane)
  (hO : O = center_of_circumsphere vertices)
  (hS : S = centroid vertices)
  (hReflect : T = reflection O S)
  (hPlanes : ∀ e ∈ edges_from_midpoints_to_opposites vertices, 
              ∃ m ∈ midpoints, 
              plane_through m ⊥ e ∈ planes)
  (hIntersect : ∀ p ∈ planes, intersects_line OS p) :
  ∀ p ∈ planes, lies_on T p := 
sorry

end reflection_point_lies_on_planes_l799_799478


namespace geometric_figure_area_l799_799925

def AX : ℕ := 3
def AB : ℕ := 2
def BY : ℕ := 2
def DY : ℕ := 2
def DX : ℕ := 2
def XY : ℕ := 5
def angle_BAX : ℕ := 90

theorem geometric_figure_area :
  (AX = 3) →
  (AB = 2) →
  (BY = 2) →
  (DY = 2) →
  (DX = 2) →
  (XY = 5) →
  (angle_BAX = 90) →
  let A_ABX := 1 / 2 * AB * AX in
  let A_ABYX := AB * (XY - AX) in
  let A_BYDX := BY * (XY - DX) in
  let A_DXY := 1 / 2 * DY * (AX - BY) in
  A_ABX + A_ABYX + A_BYDX + A_DXY = 14 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  let A_ABX := 1 / 2 * AB * AX
  let A_ABYX := AB * (XY - AX)
  let A_BYDX := BY * (XY - DX)
  let A_DXY := 1 / 2 * DY * (AX - BY)
  sorry

end geometric_figure_area_l799_799925


namespace number_of_truthful_warriors_l799_799207

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799207


namespace sum_even_numbered_terms_l799_799346

variable (n : ℕ)

def a_n (n : ℕ) : ℕ := 2 * 3^(n-1)

def new_sequence (n : ℕ) : ℕ := a_n (2 * n)

def Sn (n : ℕ) : ℕ := (6 * (1 - 9^n)) / (1 - 9)

theorem sum_even_numbered_terms (n : ℕ) : Sn n = 3 * (9^n - 1) / 4 :=
by sorry

end sum_even_numbered_terms_l799_799346


namespace real_z_iff_m_eq_5_or_neg3_nonzero_imag_z_iff_m_not_eq_5_and_neg3_pure_imag_z_iff_m_eq_neg2_third_quadrant_z_iff_m_in_interval_l799_799194

-- Given complex number z expressed as a function of real number m
def z (m : ℝ) : ℂ := (1 + complex.i) * m^2 + (5 - 2 * complex.i) * m + (6 - 15 * complex.i)

-- 1. Prove that z is real if and only if m = 5 or m = -3
theorem real_z_iff_m_eq_5_or_neg3 (m : ℝ) : (∃ (r : ℝ), z m = r) ↔ m = 5 ∨ m = -3 := 
sorry

-- 2. Prove that z has a nonzero imaginary part if and only if m ≠ 5 and m ≠ -3
theorem nonzero_imag_z_iff_m_not_eq_5_and_neg3 (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 5 ∧ m ≠ -3 := 
sorry

-- 3. Prove that z is a pure imaginary number if and only if m = -2
theorem pure_imag_z_iff_m_eq_neg2 (m : ℝ) : (∃ (i : ℝ), z m = complex.i * i) ↔ m = -2 := 
sorry

-- 4. Prove that z is in the third quadrant if and only if -3 < m < -2
theorem third_quadrant_z_iff_m_in_interval (m : ℝ) : ((z m).re < 0 ∧ (z m).im < 0) ↔ -3 < m ∧ m < -2 := 
sorry

end real_z_iff_m_eq_5_or_neg3_nonzero_imag_z_iff_m_not_eq_5_and_neg3_pure_imag_z_iff_m_eq_neg2_third_quadrant_z_iff_m_in_interval_l799_799194


namespace four_digit_numbers_count_l799_799373

def valid_middle_digit_pairs : ℕ :=
  (list.product [2, 3, 4, 5, 6, 7, 8, 9] [2, 3, 4, 5, 6, 7, 8, 9]).count (λ p, p.1 * p.2 > 10)

theorem four_digit_numbers_count :
  let first_digit_choices := 7
  let valid_pairs := valid_middle_digit_pairs
  let last_digit_choices := 10
  ∑ x in range first_digit_choices, 
  ∑ y in range valid_pairs, 
  ∑ z in range last_digit_choices, 1 = 3990 := 
sorry

end four_digit_numbers_count_l799_799373


namespace centroid_coordinates_l799_799162

-- Define the vertices of the tetrahedron
variables {R : Type*} [LinearOrderedField R]
variables (x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ x₄ y₄ z₄ : R)

-- Calculate the Euclidean distances
noncomputable def r12 := (sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2 + (z₂ - z₁) ^ 2))
noncomputable def r13 := (sqrt ((x₃ - x₁) ^ 2 + (y₃ - y₁) ^ 2 + (z₃ - z₁) ^ 2))
noncomputable def r14 := (sqrt ((x₄ - x₁) ^ 2 + (y₄ - y₁) ^ 2 + (z₄ - z₁) ^ 2))
noncomputable def r23 := (sqrt ((x₃ - x₂) ^ 2 + (y₃ - y₂) ^ 2 + (z₃ - z₂) ^ 2))
noncomputable def r24 := (sqrt ((x₄ - x₂) ^ 2 + (y₄ - y₂) ^ 2 + (z₄ - z₂) ^ 2))
noncomputable def r34 := (sqrt ((x₄ - x₃) ^ 2 + (y₄ - y₃) ^ 2 + (z₄ - z₃) ^ 2))

-- Calculate the coordinates of the centroid
noncomputable def x₀ := 
  (r12 x₁ + r13 x₃ + r14 x₄ + r23 x₂ + r24 x₂ + r34 x₃) / 
  (r12 + r13 + r14 + r23 + r24 + r34)

noncomputable def y₀ := 
  (r12 y₁ + r13 y₃ + r14 y₄ + r23 y₂ + r24 y₂ + r34 y₃) / 
  (r12 + r13 + r14 + r23 + r24 + r34)

noncomputable def z₀ := 
  (r12 z₁ + r13 z₃ + r14 z₄ + r23 z₂ + r24 z₂ + r34 z₃) / 
  (r12 + r13 + r14 + r23 + r24 + r34)

theorem centroid_coordinates :
  (x₀, y₀, z₀) = 
  ( (r12 * (x₁ + x₂) / 2 + r13 * (x₁ + x₃) / 2 + r14 * (x₁ + x₄) / 2 + r23 * (x₂ + x₃) / 2 + r24 * (x₂ + x₄) / 2 + r34 * (x₃ + x₄) / 2) / 
    (r12 + r13 + r14 + r23 + r24 + r34),
    (r12 * (y₁ + y₂) / 2 + r13 * (y₁ + y₃) / 2 + r14 * (y₁ + y₄) / 2 + r23 * (y₂ + y₃) / 2 + r24 * (y₂ + y₄) / 2 + r34 * (y₃ + y₄) / 2) / 
    (r12 + r13 + r14 + r23 + r24 + r34),
    (r12 * (z₁ + z₂) / 2 + r13 * (z₁ + z₃) / 2 + r14 * (z₁ + z₄) / 2 + r23 * (z₂ + z₃) / 2 + r24 * (z₂ + z₄) / 2 + r34 * (z₃ + z₄) / 2) / 
    (r12 + r13 + r14 + r23 + r24 + r34) ) :=
sorry

end centroid_coordinates_l799_799162


namespace proof_problem_l799_799440

variables (P : Matrix (Fin 2) (Fin 2) ℝ) (x y : Vector (Fin 2) ℝ)

-- Condition 1: P * x = (1, 4)
def condition1 : Prop :=
  P.mulVec x = ![1, 4]

-- Condition 2: P * y = (3, -2)
def condition2 : Prop :=
  P.mulVec y = ![3, -2]

-- The proof problem
theorem proof_problem (h1 : condition1 P x) (h2 : condition2 P y) :
  P.mulVec (2 • x - y) = ![-1, 10] :=
by
  sorry

end proof_problem_l799_799440


namespace loci_of_X_on_line_l_l799_799134

variables {A B C D X : Point}

-- Defining the rectangle ABCD
def is_rectangle (A B C D : Point) : Prop :=
  ∃ (l1 l2 l3 l4 : Line), 
    (A ∈ l1) ∧ (B ∈ l1) ∧ (B ∈ l2) ∧ (C ∈ l2) ∧
    (C ∈ l3) ∧ (D ∈ l3) ∧ (D ∈ l4) ∧ (A ∈ l4) ∧
    is_perpendicular l1 l2 ∧ is_perpendicular l2 l3 ∧
    is_perpendicular l3 l4 ∧ is_perpendicular l4 l1 ∧
    len_segment A B = len_segment C D ∧
    len_segment B C = len_segment D A

-- Defining the midpoints of BC and AD
def midpoint (P Q : Point) : Point := sorry  -- Assume midpoint function exists

-- Define the line l that passes through midpoints of BC and AD
def line_through_midpoints (B C D A : Point) (is_rect : is_rectangle A B C D) : Line :=
  let M1 := midpoint B C in
  let M2 := midpoint D A in
  line_through M1 M2

-- The locus condition
def locus_condition (A B C D X : Point) : Prop :=
  distance A X + distance B X = distance C X + distance D X

-- Final problem statement
theorem loci_of_X_on_line_l (A B C D : Point) (is_rect : is_rectangle A B C D) :
  ∀ X, locus_condition A B C D X ↔ X ∈ (line_through_midpoints B C D A is_rect) :=
sorry

end loci_of_X_on_line_l_l799_799134


namespace prime_factors_of_x_l799_799308

theorem prime_factors_of_x (n : ℕ) (h1 : 2^n - 32 = x) (h2 : (nat.prime_factors x).length = 3) (h3 : 3 ∈ nat.prime_factors x) :
  x = 480 ∨ x = 2016 :=
sorry

end prime_factors_of_x_l799_799308


namespace lines_AE_BC_pass_through_N_line_MN_passes_through_constant_point_G1_locus_of_midpoint_of_centers_l799_799730

variable {A B M N C D E F O1 O2 G1 H1 H2 : Point}
variable (AMCD BMEF : Square)
variable (k1 k2 : Circle)

-- Given conditions
def point_M_moves_on_segment_AB (A B M : Point) : Prop := sorry
def squares_constructed_same_side (AMCD BMEF : Square) (AB : Line) : Prop := sorry
def centers_of_squares (AMCD BMEF : Square) (O1 O2 : Point) : Prop := sorry
def circumcircles_of_squares (AMCD BMEF : Square) (k1 k2 : Circle) : Prop := sorry
def intersect_at_M_and_N (k1 k2 : Circle) (M N : Point) : Prop := sorry

-- Part (a) to be proved
theorem lines_AE_BC_pass_through_N
    (h1 : point_M_moves_on_segment_AB A B M)
    (h2 : squares_constructed_same_side AMCD BMEF)
    (h3 : centers_of_squares AMCD BMEF O1 O2)
    (h4 : circumcircles_of_squares AMCD BMEF k1 k2)
    (h5 : intersect_at_M_and_N k1 k2 M N) :
    passes_through (Line.mk A E) N ∧ passes_through (Line.mk B C) N :=
by sorry

-- Part (b) to be proved
theorem line_MN_passes_through_constant_point_G1 
    (h1 : point_M_moves_on_segment_AB A B M)
    (h2 : squares_constructed_same_side AMCD BMEF)
    (h3 : centers_of_squares AMCD BMEF O1 O2)
    (h4 : circumcircles_of_squares AMCD BMEF k1 k2)
    (h5 : intersect_at_M_and_N k1 k2 M N) :
    passes_through (Line.mk M N) G1 := 
by sorry

-- Part (c) to be proved
theorem locus_of_midpoint_of_centers
    (h1 : point_M_moves_on_segment_AB A B M)
    (h2 : squares_constructed_same_side AMCD BMEF)
    (h3 : centers_of_squares AMCD BMEF O1 O2)
    (h4 : circumcircles_of_squares AMCD BMEF k1 k2)
    (h5 : intersect_at_M_and_N k1 k2 M N) :
    locus_of_midpoint (O1, O2) = LineSegment.mk H1 H2 := 
by sorry

end lines_AE_BC_pass_through_N_line_MN_passes_through_constant_point_G1_locus_of_midpoint_of_centers_l799_799730


namespace truncated_cone_volume_and_area_l799_799055

theorem truncated_cone_volume_and_area
  (R : ℝ) (r : ℝ) (h : ℝ)
  (R_eq : R = 10)
  (r_eq : r = 3)
  (h_eq : h = 8) :
  let V := (1/3) * π * h * (R^2 + R*r + r^2)
  let l := real.sqrt (h^2 + (R - r)^2)
  let A_lateral := π * (R + r) * l
  let A_total := π * R^2 + π * r^2 + A_lateral in
  V = 370.67 * π ∧ A_total = 109 * π + 13 * π * real.sqrt 113 := 
by
  sorry

end truncated_cone_volume_and_area_l799_799055


namespace sqrt_three_irrational_l799_799689

theorem sqrt_three_irrational :
  (∀ a b : ℤ, b ≠ 0 → (- (5 : ℤ) = a / b) → false)
  ∧ (∀ a b : ℤ, b ≠ 0 → (22 = 7 * a / b) → false)
  ∧ (∀ a b : ℤ, b ≠ 0 → (4 * a / b = 2) → false)
  ∧ (∀ a b : ℤ, b ≠ 0 → (3 * a / b = 3) → false) :=
by
  intros a b b_nonzero ab_eq_5 ab_eq_22 ab_eq_4 sqrt3_eq_frac
  sorry

end sqrt_three_irrational_l799_799689


namespace part1_part2_l799_799824

open Complex

-- Define the given conditions
def z (m : ℝ) : ℂ := m - I

-- Part (1)
theorem part1 (m : ℝ) (h : m = 1) : 
  let z1 := (z m + 2 * I) / (1 - I) in
  complex.abs z1 = 1 := by
  sorry

-- Part (2)
theorem part2 (m : ℝ) (h : ∃ k : ℂ, (conj (z m)) * (1 + 3 * I) = k ∧ k.im = k) : 
  m = 3 := by
  sorry

end part1_part2_l799_799824


namespace incorrect_option_D_l799_799350

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l799_799350


namespace residue_11_pow_2016_mod_19_l799_799561

theorem residue_11_pow_2016_mod_19 : (11^2016) % 19 = 17 := 
sorry

end residue_11_pow_2016_mod_19_l799_799561


namespace AD_vector_equation_l799_799938

variables (A B C D : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
  [AddCommGroup D] [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
variables (AB AC BC : A) (BD DC : B) (AD : C)

-- Conditions
-- 1. \overrightarrow{BD} = 3\overrightarrow{DC}
-- 2. \overrightarrow{D} is on side \overrightarrow{BC}
-- Equivalent to: \overrightarrow{D} divides \overrightarrow{BC} in the ratio 3:1

def vector_equation (AD AB AC : A) (BD : B) : Prop :=
  AD = (1/4 : ℝ) • AB + (3/4 : ℝ) • AC

theorem AD_vector_equation (h1 : BD = (3:ℝ) • DC) (h2 : D = BC) :
  vector_equation AD AB AC BD :=
by
  sorry

end AD_vector_equation_l799_799938


namespace smallest_prime_with_digit_sum_23_l799_799653

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l799_799653


namespace question_1_question_2_question_3_l799_799551

noncomputable def represent_in_base (a : List ℤ) (x : ℝ) : ℝ :=
  a.enum.foldl (λ sum (i, ai) => sum + ai * x^i) 0

theorem question_1 (x : ℝ) (hx : x ≠ 0) :
  represent_in_base [1, -2, 3, -6] x = (1 - 2 * x) * (1 + 3 * x^2) :=
sorry

def a_seq (k : ℕ) : ℝ :=
  if k = 0 then 2
  else /-(for k ∈ ℕ*--*/ from sorry)

def b_n (n : ℕ) : ℝ :=
  represent_in_base ([2] ++ (List.range (3 * n)).map a_seq) 2

theorem question_2 :
  ∃ (p q : ℝ), (∀ n : ℕ, b_n n = p * 8^n + q) ∧ p = 2 / 7 ∧ q = -2 / 7 :=
sorry

noncomputable def C (n k : ℕ) : ℕ := sorry -- Binomial coefficient placeholder

noncomputable def d_n (n : ℕ) : ℝ :=
  represent_in_base (List.range (n + 1)).map (C n) 2

theorem question_3 (t : ℝ) (ht1 : t ≠ 0) (ht2 : t > -1) :
  tendsto (λ n : ℕ, d_n n / d_n (n + 1)) at_top (𝓝 (1 / 3)) :=
sorry

end question_1_question_2_question_3_l799_799551


namespace factorial_divisibility_l799_799058

theorem factorial_divisibility (k n : ℕ) (h_k : 0 < k ∧ k ≤ 2020) (h_n : 1 ≤ n) :
  ¬ (3 ^ ((k - 1) * n + 1) ∣ ((kn)! / n!)^2) :=
sorry

end factorial_divisibility_l799_799058


namespace min_value_l799_799838

theorem min_value (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 2) : 
  (∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ (1/3 * x^3 + y^2 + z = 13/12)) :=
sorry

end min_value_l799_799838


namespace number_on_board_after_60_minutes_l799_799014

-- Define the transformation on the number based on the problem's rule
def transform (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2 + 12

-- Define the sequence of transformations starting from the initial number
def sequence (n : ℕ) : ℕ -> ℕ
  | 0     => n
  | (k+1) => transform (sequence n k)

-- The main theorem to prove
theorem number_on_board_after_60_minutes :
  sequence 27 60 = 14 :=
sorry -- proof omitted

end number_on_board_after_60_minutes_l799_799014


namespace volume_percentage_l799_799145

-- Definitions of the initial conditions
def box_length : ℝ := 8
def box_width : ℝ := 6
def box_height : ℝ := 12
def cube_side : ℝ := 4

-- Definition for the correct answer
def correct_answer : ℝ := 66.67

-- The Lean 4 statement to express and prove the given problem
theorem volume_percentage :
  let box_volume := box_length * box_width * box_height,
      cube_volume := cube_side ^ 3,
      num_cubes := (box_length / cube_side).to_int * (box_width / cube_side).to_int * (box_height / cube_side).to_int,
      cubes_volume := num_cubes * cube_volume,
      volume_percentage := (cubes_volume / box_volume) * 100 in
  volume_percentage = correct_answer :=
by
  sorry

end volume_percentage_l799_799145


namespace range_of_x_l799_799865

theorem range_of_x (f : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = 5 + cos x) (h_init : f 0 = 0) :
  (∀ x, f (1 - x) + f (1 - x^2) < 0 → x < -2 ∨ x > 1) :=
sorry

end range_of_x_l799_799865


namespace water_usage_march_l799_799544

theorem water_usage_march :
  let f : ℝ → ℝ :=
    λ x, if 0 ≤ x ∧ x ≤ 7 then 3 * x
         else if 7 < x ∧ x ≤ 11 then 6 * x - 21
         else if 11 < x ∧ x ≤ 15 then 9 * x - 54
  in f 9 + f 12 + f 10 = 126 →
     10 ∈ ℝ ∧ 10 ≤ 15 ∧ 10 ≥ 0 :=
by {
  sorry
}

end water_usage_march_l799_799544


namespace number_of_ways_to_assign_roles_l799_799728

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 7
  let male_roles := 3
  let female_roles := 3
  let neutral_roles := 2
  let ways_male_roles := men * (men - 1) * (men - 2)
  let ways_female_roles := women * (women - 1) * (women - 2)
  let ways_neutral_roles := (men + women - male_roles - female_roles) * (men + women - male_roles - female_roles - 1)
  ways_male_roles * ways_female_roles * ways_neutral_roles = 1058400 := 
by
  sorry

end number_of_ways_to_assign_roles_l799_799728


namespace angle_x_value_l799_799408

theorem angle_x_value :
  ∀ (A B C D E : Point) (x : Real),
  is_line_segment A B ∧
  is_point C ∧ is_point D ∧ is_point E ∧
  is_perpendicular CD AB ∧
  is_diagonal CE CD ∧
  angle DCE = 60 ∧
  angle ACD = 90 ∧
  angle ACB = 180 ->
  x = 30 :=
by
  intro A B C D E x
  intro h
  sorry

end angle_x_value_l799_799408


namespace more_movies_than_books_l799_799060

-- Conditions
def books_read := 15
def movies_watched := 29

-- Question: How many more movies than books have you watched?
theorem more_movies_than_books : (movies_watched - books_read) = 14 := sorry

end more_movies_than_books_l799_799060


namespace sum_of_x_coords_where_g_eq_2_l799_799773

def segment1 (x : ℝ) : ℝ := 2 * x + 3
def segment2 (x : ℝ) : ℝ := -x
def segment3 (x : ℝ) : ℝ := 2 * x - 3

def g (x : ℝ) : ℝ :=
if x < -1 then segment1 x
else if x ≤ 1 then segment2 x
else segment3 x

theorem sum_of_x_coords_where_g_eq_2 : 
  let x1 := -0.5
  let x3 := 2.5
  g x1 = 2 ∧ g x3 = 2 ∧ x1 + x3 = 2 :=
by
  sorry

end sum_of_x_coords_where_g_eq_2_l799_799773


namespace statement_1_statement_2_statement_3_statement_4_l799_799292

-- Conditions
variable (m n : Line) (α β : Plane)
variable (h_m_perp_α : perpendicular m α)
variable (h_n_in_β : in_plane n β)

-- Prove statements
theorem statement_1 (h_α_parallel_β : parallel α β) : perpendicular m n := sorry
theorem statement_2 (h_α_perp_β : perpendicular α β) : ¬parallel m n := sorry
theorem statement_3 (h_m_parallel_n : parallel m n) : perpendicular α β := sorry
theorem statement_4 (h_m_perp_n : perpendicular m n) : ¬parallel α β := sorry

end statement_1_statement_2_statement_3_statement_4_l799_799292


namespace ratio_of_square_sides_l799_799033

theorem ratio_of_square_sides (areaA areaB areaC : ℕ) (hA : areaA = 25) (hB : areaB = 81) (hC : areaC = 64) :
  let sideA := Int.sqrt areaA,
      sideB := Int.sqrt areaB,
      sideC := Int.sqrt areaC
  in (sideA, sideB, sideC) = (5, 9, 8) :=
by
  -- The theorem requires a proof, hence we put sorry for now.
  sorry

end ratio_of_square_sides_l799_799033


namespace isosceles_triangle_perimeter_two_five_l799_799403

theorem isosceles_triangle_perimeter_two_five :
  ∃ (a b c : ℝ), a = 2 ∧ b = 5 ∧ (a = b ∨ b = c ∨ c = a) ∧ (a + b + c = 12) :=
by
  use 2, 5, 5
  split
  exact rfl
  split
  exact rfl
  split
  right
  left
  exact rfl
  sorry

end isosceles_triangle_perimeter_two_five_l799_799403


namespace range_of_x_l799_799844

-- Define the problem conditions and the conclusion to be proved
theorem range_of_x (f : ℝ → ℝ) (h_inc : ∀ x y, -1 ≤ x → x ≤ 1 → -1 ≤ y → y ≤ 1 → x ≤ y → f x ≤ f y)
  (h_ineq : ∀ x, f (x - 2) < f (1 - x)) :
  ∀ x, 1 ≤ x ∧ x < 3 / 2 :=
by
  sorry

end range_of_x_l799_799844


namespace skill_position_players_wait_l799_799122

theorem skill_position_players_wait
  (num_linemen : ℕ) (drink_per_linemen : ℕ) 
  (num_skill_players : ℕ) (drink_per_skill_player : ℕ) 
  (total_water : ℕ) : ℕ :=
  (num_linemen = 12) →
  (drink_per_linemen = 8) →
  (num_skill_players = 10) →
  (drink_per_skill_player = 6) →
  (total_water = 126) →
  num_skill_players - (total_water - num_linemen * drink_per_linemen) / drink_per_skill_player = 5 := sorry

end skill_position_players_wait_l799_799122


namespace greatest_third_side_of_triangle_l799_799080

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : ∃ x : ℕ, x < a + b ∧ x = 16 := by
  use 16
  rw [h1, h2]
  split
  · linarith
  · rfl

end greatest_third_side_of_triangle_l799_799080


namespace fixed_point_exists_l799_799332

noncomputable def ellipse_equation (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0)
    (h_eccentricity : (e : ℝ) = 1 / 2) : Prop :=
  let C := ∀ (x y : ℝ),
  x^2 / a^2 + y^2 / b^2 = 1 ∧
  x = 1 ∧ y = 3 / 2
  in ∃ (c : ℝ), a = 2 * c ∧ a^2 = 4 * c^2 ∧ b^2 = 3 * c^2 ∧ a^2 = 4 ∧ b^2 = 3

noncomputable def fixed_point_condition (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0)
    (h_eccentricity : (e : ℝ) = 1 / 2) (k m : ℝ)
    (h_tangent : 4*k^2 - m^2 + 3 = 0) : Prop :=
  let T := (-4*k / m, 3 / m) in
  let S := (4, 4*k + m) in
  let A := (1, 0) in
  ∀ (k m : ℝ), 4*k^2 - m^2 + 3 = 0 → ∃ A, A = (1, 0)

theorem fixed_point_exists (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0)
    (h_eccentricity : (e : ℝ) = 1 / 2) (k m : ℝ)
    (h_tangent : 4*k^2 - m^2 + 3 = 0) :
    ellipse_equation a b a_gt_b b_gt_0 h_eccentricity →
    fixed_point_condition a b a_gt_b b_gt_0 h_eccentricity k m h_tangent :=
by
  sorry

end fixed_point_exists_l799_799332


namespace error_in_area_is_41_61_percent_l799_799746

def percentage_error_in_area (x : ℝ) : ℝ :=
  let actual_area := x^2
  let measured_side := 1.19 * x
  let erroneous_area := measured_side^2
  let error_in_area := erroneous_area - actual_area
  (error_in_area / actual_area) * 100

theorem error_in_area_is_41_61_percent (x : ℝ) : percentage_error_in_area x = 41.61 := 
by
  sorry

end error_in_area_is_41_61_percent_l799_799746


namespace smallest_prime_digit_sum_23_l799_799610

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l799_799610


namespace original_speed_of_train_l799_799720

theorem original_speed_of_train (delay_minutes : ℝ) (distance : ℝ) (speed_increase : ℝ) 
  (delay_minutes = 12) (distance = 60) (speed_increase = 15) : 
  ∃ (x : ℝ), x = 39.375 := 
begin
  sorry
end

end original_speed_of_train_l799_799720


namespace find_max_min_ex_l799_799776

noncomputable def M_m (S : Set ℝ) : ℝ × ℝ :=
  (Sup S, Inf S)

theorem find_max_min_ex : 
  M_m { x | ∃ a b : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ 2 ∧ x = 3 / a + b } = (5, 2 * Real.sqrt 3) :=
sorry

end find_max_min_ex_l799_799776


namespace exists_tangent_circle_l799_799884

noncomputable def tangentCircleCenter {p q t : Line} (hpq_parallel : parallel p q) (hA : Point) (hB : Point) (ht_intersect_p : t ∩ p = hA) (ht_intersect_q : t ∩ q = hB) : Point :=
sorry

theorem exists_tangent_circle (p q t : Line) (hpq_parallel : parallel p q) (hA : Point) (hB : Point) (ht_intersect_p : t ∩ p = hA) (ht_intersect_q : t ∩ q = hB) :
  ∃ O : Point, ∃ r : ℝ, (∀ P : Point, P ∈ (angleBisector (angle hA t p)) ∧ P ∈ (angleBisector (angle hB t q)) → P = O) ∧ 
  Circle O r ∈ tangentToLines p q t :=
sorry

end exists_tangent_circle_l799_799884


namespace radius_of_sphere_centered_at_centroid_l799_799832

theorem radius_of_sphere_centered_at_centroid (edge_length : ℝ) (total_curve_length : ℝ) :
  edge_length = 2 * Real.sqrt 6 →
  total_curve_length = 4 * Real.pi →
  ∃ R : ℝ, R = Real.sqrt 5 / 2 :=
by
  intros _ _
  use Real.sqrt 5 / 2
  sorry

end radius_of_sphere_centered_at_centroid_l799_799832


namespace smallest_nonnegative_sum_l799_799936

theorem smallest_nonnegative_sum (n : ℕ) (h : n = 2005) :
  ∃ (f : ℕ → ℤ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → f i = (if i % 2 = 0 then -1 else 1) * (i ^ 2)) ∧ ((finset.range n).sum (λ i, f (i + 1)) = 1) :=
by {
  sorry,
}

end smallest_nonnegative_sum_l799_799936


namespace smallest_prime_with_digit_sum_23_l799_799579

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l799_799579


namespace volume_percentage_correct_l799_799137

-- The dimensions of the rectangular box
def length := 8
def width := 6
def height := 12

-- The side length of the cubes
def cube_edge := 4

-- Calculate the volume of the box
def box_volume : ℕ := length * width * height

-- Calculate how many cubes fit in the box
def cubes_in_length := length / cube_edge
def cubes_in_width := width / cube_edge
def cubes_in_height := height / cube_edge

-- Calculate the volume of the part filled with cubes
def cubes_volume := (cubes_in_length * cube_edge) * (cubes_in_width * cube_edge) * (cubes_in_height * cube_edge)

-- Calculate the ratio of the filled volume to the box volume
def volume_ratio := cubes_volume / box_volume

-- Convert the ratio to a percentage
noncomputable def volume_percentage := (volume_ratio : ℝ) * 100

-- Statement of the problem
theorem volume_percentage_correct : volume_percentage = 66.67 := by
  -- Proof is not required, so we use 'sorry'
  sorry

end volume_percentage_correct_l799_799137


namespace number_of_truthful_warriors_l799_799213

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l799_799213


namespace value_of_expression_l799_799358

variable (p q r s : ℝ)

-- Given condition in a)
def polynomial_function (x : ℝ) := p * x^3 + q * x^2 + r * x + s
def passes_through_point := polynomial_function p q r s (-1) = 4

-- Proof statement in c)
theorem value_of_expression (h : passes_through_point p q r s) : 6 * p - 3 * q + r - 2 * s = -24 := by
  sorry

end value_of_expression_l799_799358


namespace geometric_series_half_l799_799391

theorem geometric_series_half (n : ℕ) (h : n ≥ 1) : 
  ∑ i in Finset.range n, (1 / 2) ^ (i + 1) < 1 :=
sorry

end geometric_series_half_l799_799391


namespace gcd_p4_minus_1_eq_240_l799_799476

theorem gcd_p4_minus_1_eq_240 (p : ℕ) (hp : Prime p) (h_gt_5 : p > 5) :
  gcd (p^4 - 1) 240 = 240 :=
by sorry

end gcd_p4_minus_1_eq_240_l799_799476


namespace equidistant_points_at_distance_l799_799063

variable {r : ℝ} {x_A y_A z_A : ℝ}

theorem equidistant_points_at_distance (x y z : ℝ) :
  (x - x_A)^2 + (y - y_A)^2 + (z - z_A)^2 = r^2 ∧ z = x → 
  (x - x_A)^2 + (y - y_A)^2 = r^2 / 2 :=
by
  intros h
  have h1 := h.1
  have h2 := h.2
  rw h2 at h1
  sorry

end equidistant_points_at_distance_l799_799063


namespace problem_l799_799860

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem problem (h : f 10 = 756) : f 10 = 756 := 
by 
  sorry

end problem_l799_799860


namespace range_of_k_l799_799326

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative_exists (x : ℝ) : ∃ f' : ℝ, (f' = deriv f x)
axiom f_derivative_nonzero (x : ℝ) : deriv f x ≠ 0
axiom f_functional_eq (x : ℝ) : f (f x - 2017 ^ x) = 2017

def g (k : ℝ) (x : ℝ) : ℝ := sin x - cos x - k * x

theorem range_of_k (k : ℝ) :
  (∀ x ∈ Icc (-π / 2) (π / 2), deriv (g k) x ≥ 0) ↔ k ∈ set.Iic (-1) :=
sorry

end range_of_k_l799_799326


namespace george_says_the_number_l799_799483

def students_count_300 : ℕ := 300

def alice_barbara_candice_numbers : list ℕ :=
  list.range' 2 (students_count_300 - 1) 3

def filter_out_first_set (lst : list ℕ) (skip : ℕ) : list ℕ :=
  lst.filter (λ n, (n - skip + 1) % 3 ≠ 0)

def debbie_numbers : list ℕ :=
  filter_out_first_set (list.range' 1 students_count_300 1) 3

def eliza_numbers (debbie_nums : list ℕ) : list ℕ :=
  filter_out_first_set debbie_nums 4

def fatima_numbers (eliza_nums : list ℕ) : list ℕ :=
  filter_out_first_set eliza_nums 5

def george_number (fatima_nums : list ℕ) : option ℕ :=
  list.range' 1 students_count_300 1 \ (alice_barbara_candice_numbers ++ debbie_numbers ++ eliza_numbers ++ fatima_numbers)

theorem george_says_the_number :
  george_number fatima_numbers = some 104 :=
sorry

end george_says_the_number_l799_799483


namespace find_x_l799_799295

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end find_x_l799_799295


namespace smallest_prime_with_digit_sum_23_l799_799632

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799632


namespace max_d_77733e_divisible_by_33_l799_799258

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l799_799258


namespace number_of_eight_digit_multiples_of_9_and_distinct_digits_l799_799802

noncomputable def sum (s : Finset ℕ) : ℕ :=
s.val.sum

theorem number_of_eight_digit_multiples_of_9_and_distinct_digits :
  let unused_digit_sum := 9 in
  let total_sum := 45 in
  let digit_pairs_summing_to_unused_digit_sum := {(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)} in
  let total_pairs := digit_pairs_summing_to_unused_digit_sum.card in
  let permutation_without_zero := 8 * (Nat.factorial 7) in
  let permutation_with_zero := 7 * (Nat.factorial 7) in
  let total_permutations_zero_case := 1 * permutation_without_zero in
  let total_permutations_non_zero_case := 4 * permutation_with_zero in
  total_permutations_zero_case + total_permutations_non_zero_case = 181440 :=
by
  sorry

end number_of_eight_digit_multiples_of_9_and_distinct_digits_l799_799802


namespace celine_gabriel_erasers_ratio_l799_799764

theorem celine_gabriel_erasers_ratio :
  ∀ (C G J : ℕ),
    C = 10 →
    J = 2 * C →
    C + G + J = 35 →
    C / Nat.gcd C G = 2 ∧ G / Nat.gcd C G = 1 :=
by
  intros C G J hC hJ hSum
  have h1 : 10 + G + 20 = 35 := by rwa [←hC, ←hJ]
  have hG : G = 35 - 10 - 20 := by linarith
  rw [hG] at *
  have gcd_cg : Nat.gcd 10 5 = 5 := by norm_num
  split
  · rw [Nat.div_eq_of_eq_mul_right gcd_cg] <|> norm_num
  · rw [Nat.div_eq_of_eq_mul_right gcd_cg] <|> norm_num
  sorry

end celine_gabriel_erasers_ratio_l799_799764


namespace aprons_to_sew_tomorrow_l799_799366

-- Definitions for the conditions
def total_aprons : ℕ := 150
def aprons_sewn : ℕ := 13
def aprons_sewn_today : ℕ := 3 * aprons_sewn
def total_sewn_so_far : ℕ := aprons_sewn + aprons_sewn_today
def remaining_aprons : ℕ := total_aprons - total_sewn_so_far

-- Theorem to state the solution
theorem aprons_to_sew_tomorrow : remaining_aprons / 2 = 49 :=
by
  -- Noncomputable sections have been defined
  have hs : remaining_aprons = 98 := sorry 
  show remaining_aprons / 2 = 49
  rw [hs]
  norm_num

end aprons_to_sew_tomorrow_l799_799366


namespace determine_a_l799_799363

/-- Definition for the set A with conditions given. -/
def setA (a : ℝ) : set (ℝ × ℝ) :=
  {p | |p.1| + |p.2| = a ∧ a > 0}

/-- Definition for the set B with conditions given. -/
def setB : set (ℝ × ℝ) :=
  {p | |p.1 * p.2| + 1 = |p.1| + |p.2|}

/-- The main theorem to be proven. -/
theorem determine_a (a : ℝ) :
  (setA a ∩ setB = {p | is_vertex_of_regular_octagon p}) → a = 2 + real.sqrt 2 :=
sorry

end determine_a_l799_799363


namespace fiona_cleaning_time_l799_799465

theorem fiona_cleaning_time (total_time : ℕ) (lilly_ratio : ℚ) (to_minutes : ℕ → ℕ → ℕ) 
  (lilly_fiona_cleaning_time : total_time = 8) 
  (lilly_cleaning_ratio : lilly_ratio = 1/4) 
  (convert_to_minutes : to_minutes = λ h m, h * m) : 
  ∃ fiona_cleaning_time : ℕ, to_minutes (total_time - total_time * lilly_ratio) 60 = 360 :=
by {
  sorry
}

end fiona_cleaning_time_l799_799465


namespace more_boys_than_girls_l799_799064

theorem more_boys_than_girls (total_people : ℕ) (num_girls : ℕ) (num_boys : ℕ) (more_boys : ℕ) : 
  total_people = 133 ∧ num_girls = 50 ∧ num_boys = total_people - num_girls ∧ more_boys = num_boys - num_girls → more_boys = 33 :=
by 
  sorry

end more_boys_than_girls_l799_799064


namespace mass_percentage_of_Cl_in_NH4Cl_l799_799801

def molar_mass (N_mass H_mass Cl_mass : Float) : Float :=
  N_mass + (H_mass * 4) + Cl_mass

def mass_percentage_Cl (Cl_mass NH4Cl_mass : Float) : Float :=
  (Cl_mass / NH4Cl_mass) * 100

theorem mass_percentage_of_Cl_in_NH4Cl : 
  molar_mass 14.01 1.01 35.45 = 53.50 → 
  mass_percentage_Cl 35.45 53.50 = 66.26 :=
by
  intros h₁
  rw [molar_mass, mass_percentage_Cl] at h₁
  sorry

end mass_percentage_of_Cl_in_NH4Cl_l799_799801


namespace isosceles_triangle_perimeter_two_five_l799_799404

theorem isosceles_triangle_perimeter_two_five :
  ∃ (a b c : ℝ), a = 2 ∧ b = 5 ∧ (a = b ∨ b = c ∨ c = a) ∧ (a + b + c = 12) :=
by
  use 2, 5, 5
  split
  exact rfl
  split
  exact rfl
  split
  right
  left
  exact rfl
  sorry

end isosceles_triangle_perimeter_two_five_l799_799404


namespace A0A5_perpendicular_A3A4_l799_799748

variables {A : Type*} [metric_space A] [normed_group A] [normed_space ℝ A]

-- Definitions for points A0, A1, A2, A3, A4, A5 and distances
variables (A0 A1 A2 A3 A4 A5 : A)

-- Distances
variable (hA0A1 : dist A0 A1 = 1)
variable (hA1A2 : dist A1 A2 = 2)
variable (hA2A3 : dist A2 A3 = 3)
variable (hA3A4 : dist A3 A4 = 4)
variable (hA4A5 : dist A4 A5 = 5)

-- Angles
variable (hAngleA0A1A2 : angle A0 A1 A2 = real.pi / 3)
variable (hAngleA1A2A3 : angle A1 A2 A3 = real.pi / 3)
variable (hAngleA2A3A4 : angle A2 A3 A4 = real.pi / 3)
variable (hAngleA3A4A5 : angle A3 A4 A5 = real.pi / 3)

theorem A0A5_perpendicular_A3A4 :
  (angle A0 A5 A3 = real.pi / 2 ∨ angle A0 A5 A3 = 3 * real.pi / 2) ∧
  (angle A3 A4 A5 = real.pi / 2 ∨ angle A3 A4 A5 = 3 * real.pi / 2) :=
sorry

end A0A5_perpendicular_A3A4_l799_799748


namespace opposite_event_of_hitting_at_least_once_is_missing_both_times_l799_799156

theorem opposite_event_of_hitting_at_least_once_is_missing_both_times
  (A B : Prop) :
  ¬(A ∨ B) ↔ (¬A ∧ ¬B) :=
by
  sorry

end opposite_event_of_hitting_at_least_once_is_missing_both_times_l799_799156


namespace smallest_prime_with_digit_sum_23_l799_799630

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l799_799630


namespace number_of_ways_to_represent_2023_l799_799437

def countWaysToRepresent2023 : ℕ := 403

theorem number_of_ways_to_represent_2023 :
  {a_3 a_2 a_1 a_0 : ℕ // 0 ≤ a_3 ∧ a_3 ≤ 199 ∧
                             0 ≤ a_2 ∧ a_2 ≤ 199 ∧
                             0 ≤ a_1 ∧ a_1 ≤ 199 ∧
                             0 ≤ a_0 ∧ a_0 ≤ 199 ∧
                             2023 = a_3 * 10^3 + a_2 * 10^2 + a_1 * 10 + a_0}.count =
  countWaysToRepresent2023 :=
  by
    sorry

end number_of_ways_to_represent_2023_l799_799437


namespace truthfulness_count_l799_799229

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l799_799229


namespace find_coordinates_of_point_l799_799318

theorem find_coordinates_of_point (P : ℝ × ℝ) (P1 : ℝ × ℝ := (2, -1)) (P2 : ℝ × ℝ := (0, 5)) 
  (h : ∥(P.fst - P1.fst, P.snd - P1.snd)∥ = 2 * ∥(P2.fst - P.fst, P2.snd - P.snd)∥) : 
  P = (-2, 11) := 
by 
  have h1 : P.fst - 2 = -2 * (-P.fst),
  have h2 : P.snd + 1 = -2 * (5 - P.snd),
  sorry

end find_coordinates_of_point_l799_799318


namespace warriors_truth_tellers_l799_799204

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l799_799204


namespace area_triangle_aop_l799_799769

noncomputable def vector3d := (ℝ × ℝ × ℝ)

def distance (p1 p2 : vector3d) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

def cross_product (v1 v2 : vector3d) : vector3d :=
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

def magnitude (v : vector3d) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def area_of_triangle (v1 v2 : vector3d) : ℝ :=
  0.5 * magnitude (cross_product v1 v2)

theorem area_triangle_aop : 
  let A := (0, 0, 0),
      O := (1/(2√3), 1/(2√3), 1/(2√3)),
      P := (1/(2√3), 1/(2√3), 1/√3) in
  area_of_triangle (O - A) (P - A) = sqrt (sqrt (2) / 24) :=
by
  sorry

end area_triangle_aop_l799_799769


namespace meters_above_sea_level_l799_799385

theorem meters_above_sea_level :
  (∀ (x y : Int), (x = -15 → y = 15) → (120 = -x → 120 = y)) → (120 = 120) :=
by
  intros _ _ _ _ _ _ _
  sorry

end meters_above_sea_level_l799_799385


namespace frequency_of_sixth_group_l799_799833

theorem frequency_of_sixth_group :
  ∀ (total_points : ℕ) (f1 f2 f3 f4 : ℕ) (fifth_group_ratio : ℝ),
    total_points = 40 →
    f1 = 10 →
    f2 = 5 →
    f3 = 7 →
    f4 = 6 →
    fifth_group_ratio = 0.1 →
    let remaining_points := total_points - (f1 + f2 + f3 + f4) - total_points * fifth_group_ratio in
    remaining_points = 8 :=
begin
  sorry
end

end frequency_of_sixth_group_l799_799833


namespace multiplication_by_9_l799_799918

theorem multiplication_by_9 (n : ℕ) (h1 : n < 10) : 9 * n = 10 * (n - 1) + (10 - n) := 
sorry

end multiplication_by_9_l799_799918


namespace incorrect_conclusion_D_l799_799355

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l799_799355


namespace shopkeeper_loss_amount_l799_799155

theorem shopkeeper_loss_amount (total_stock_worth : ℝ)
                               (portion_sold_at_profit : ℝ)
                               (portion_sold_at_loss : ℝ)
                               (profit_percentage : ℝ)
                               (loss_percentage : ℝ) :
  total_stock_wworth = 14999.999999999996 →
  portion_sold_at_profit = 0.2 →
  portion_sold_at_loss = 0.8 →
  profit_percentage = 0.10 →
  loss_percentage = 0.05 →
  (total_stock_worth - ((portion_sold_at_profit * total_stock_worth * (1 + profit_percentage)) + 
                        (portion_sold_at_loss * total_stock_worth * (1 - loss_percentage)))) = 300 := 
by 
  sorry

end shopkeeper_loss_amount_l799_799155


namespace cos_double_angle_sum_eq_two_l799_799820

theorem cos_double_angle_sum_eq_two (A B C : ℝ) 
  (h1 : A + B + C = 180 * (Real.pi / 180)) 
  (h2 : (sin A + sin B + sin C) / (cos A + cos B + cos C) = 1) :
  (cos (2 * A) + cos (2 * B) + cos (2 * C)) / (cos A + cos B + cos C) = 2 := 
by
  sorry

end cos_double_angle_sum_eq_two_l799_799820


namespace prime_factors_of_x_l799_799306

theorem prime_factors_of_x (n : ℕ) (h1 : 2^n - 32 = x) (h2 : (nat.prime_factors x).length = 3) (h3 : 3 ∈ nat.prime_factors x) :
  x = 480 ∨ x = 2016 :=
sorry

end prime_factors_of_x_l799_799306


namespace find_solutions_l799_799269

theorem find_solutions (x : ℝ) : (x = -9 ∨ x = -3 ∨ x = 3) →
  (1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0) :=
by {
  sorry
}

end find_solutions_l799_799269


namespace definite_integral_cos8_l799_799104

open Real
open IntervalIntegral

-- Define the function
def integrand (x : ℝ) : ℝ := 2^4 * cos x ^ 8

-- Define the integral and the theorem to prove the equality
theorem definite_integral_cos8 : 
  ∫ x in 0..π, integrand x = 35 * π / 8 :=
sorry

end definite_integral_cos8_l799_799104


namespace skill_position_players_waiting_l799_799123

def linemen_drink : ℕ := 8
def skill_position_player_drink : ℕ := 6
def num_linemen : ℕ := 12
def num_skill_position_players : ℕ := 10
def cooler_capacity : ℕ := 126

theorem skill_position_players_waiting :
  num_skill_position_players - (cooler_capacity - num_linemen * linemen_drink) / skill_position_player_drink = 5 :=
by
  -- Calculation is needed to be filled in here
  sorry

end skill_position_players_waiting_l799_799123
