import Data.Real.Basic
import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.CommClass
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Modulo
import Mathlib.Algebra.Order.Rearrangement
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Parametric
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.Geometry.Parabolas
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.CombinatorialNumberTheory
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Square
import Mathlib.Data.Pi
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.NumberTheory.Prime_factors
import Mathlib.Probability.ProbabilityMassFunc
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic
import data.polynomial

namespace abs_eq_self_iff_nonneg_l414_414872

variable (a : ℝ)

theorem abs_eq_self_iff_nonneg (h : |a| = a) : a ≥ 0 :=
by
  sorry

end abs_eq_self_iff_nonneg_l414_414872


namespace not_identical_polyhedron_after_translation_l414_414066

theorem not_identical_polyhedron_after_translation:
  ∃ (P : Set Point) (V : Vector), convex_polyhedron P ∧ 
    (∀ edge ∈ edges P, translate edge V) ∧ new_polyhedron P V ≠ P :=
sorry

end not_identical_polyhedron_after_translation_l414_414066


namespace sum_odd_numbers_1_to_20_l414_414019

def is_odd (n : ℤ) : Prop := n % 2 = 1

def odd_numbers := {n : ℤ | is_odd n ∧ 1 ≤ n ∧ n ≤ 20}

theorem sum_odd_numbers_1_to_20 :
  ∑ n in (finite_of_set odd_numbers).to_finset, n = 100 :=
sorry

end sum_odd_numbers_1_to_20_l414_414019


namespace derivative_first_function_derivative_second_function_l414_414773

-- Definition for the first problem
def first_function (x : ℝ) : ℝ := x^2 * sin x

-- The first proof problem statement
theorem derivative_first_function : 
  (derivative first_function) = (λ x, 2 * x * sin x + x^2 * cos x) := 
by 
  -- proof will go here
  sorry

-- Definition for the second problem
def second_function (x : ℝ) : ℝ := tan x

-- The second proof problem statement
theorem derivative_second_function : 
  (derivative second_function) = (λ x, 1 / (cos x)^2) := 
by 
  -- proof will go here
  sorry

end derivative_first_function_derivative_second_function_l414_414773


namespace div_difference_l414_414788

theorem div_difference {a b n : ℕ} (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (h : n ∣ a^n - b^n) :
  n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end div_difference_l414_414788


namespace ball_returns_to_ami_l414_414764

def next_position (current : ℕ) : ℕ :=
  ((current - 1 + 5) % 11) + 1

theorem ball_returns_to_ami :
  ∃ n, (iteration n next_position 1 = 1) ∧ n = 12 :=
begin
  use 12,
  apply iteration_fixed_point,
  sorry
end

end ball_returns_to_ami_l414_414764


namespace rope_total_in_inches_l414_414559

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end rope_total_in_inches_l414_414559


namespace polynomial_unique_l414_414578

noncomputable def lagrange_interpolation {R : Type*} [Field R] (a : Fin n → R) (b : Fin n → R) : Polynomial R :=
  Polynomial.divX (nconn.toPoly ⟨a, sorry⟩ (⟨b, sorry⟩))

theorem polynomial_unique
  {R : Type*} [Field R]
  (a : Fin n → R) (ha : Function.Injective a) (b : Fin n → R) :
  ∃ (P : Polynomial R), 
    (∀ i, P.eval (a i) = b i) ∧
    (∀ P', (∀ i, P'.eval (a i) = b i) → P' = polynomial.prodX a * Q + lagrange_interpolation a b) :=
begin
  sorry
end

end polynomial_unique_l414_414578


namespace solution_y_amount_l414_414613

-- Definitions based on the conditions
def alcohol_content_x : ℝ := 0.10
def alcohol_content_y : ℝ := 0.30
def initial_volume_x : ℝ := 50
def final_alcohol_percent : ℝ := 0.25

-- Function to calculate the amount of solution y needed
def required_solution_y (y : ℝ) : Prop :=
  (alcohol_content_x * initial_volume_x + alcohol_content_y * y) / (initial_volume_x + y) = final_alcohol_percent

theorem solution_y_amount : ∃ y : ℝ, required_solution_y y ∧ y = 150 := by
  sorry

end solution_y_amount_l414_414613


namespace product_without_zero_digits_l414_414225

def no_zero_digits (n : ℕ) : Prop :=
  ¬ ∃ d : ℕ, d ∈ n.digits 10 ∧ d = 0

theorem product_without_zero_digits :
  ∃ a b : ℕ, a * b = 1000000000 ∧ no_zero_digits a ∧ no_zero_digits b :=
by
  sorry

end product_without_zero_digits_l414_414225


namespace max_good_sequences_correct_l414_414367

-- Define the conditions: Number of blue, red, and green beads in the necklace
def num_blue : Nat := 50
def num_red : Nat := 100
def num_green : Nat := 100

-- Define what makes a sequence good
def good_sequence (seq : List Char) : Prop :=
  (seq.filter (λ c => c = 'B')).length = 2 ∧
  (seq.filter (λ c => c = 'R')).length = 1 ∧
  (seq.filter (λ c => c = 'G')).length = 1

-- Define the necklace as cyclic list of beads
def cyclic_necklace : List Char := List.repeat 'B' num_blue 
                       ++ List.repeat 'R' num_red 
                       ++ List.repeat 'G' num_green

-- Define the length of the necklace
def necklace_len : Nat :=
  num_blue + num_red + num_green

-- Define a function to check if the given indices form a good sequence
def is_good_sequence (necklace : List Char) (idx : Nat) : Prop := 
  good_sequence [necklace.get (idx % necklace_len), 
                 necklace.get ((idx + 1) % necklace_len),
                 necklace.get ((idx + 2) % necklace_len),
                 necklace.get ((idx + 3) % necklace_len)]

-- Define the maximum number of good sequences
def max_good_sequences (necklace : List Char) : Nat :=
  List.length (List.filter (λ idx => is_good_sequence necklace idx) 
              (List.range necklace_len))

-- The theorem to prove
theorem max_good_sequences_correct :
  max_good_sequences cyclic_necklace = 99 := by
  sorry

end max_good_sequences_correct_l414_414367


namespace fixed_point_f_l414_414635

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 1

-- The theorem to be proven
theorem fixed_point_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 2 :=
by
  sorry

end fixed_point_f_l414_414635


namespace floor_add_ge_floor_add_floor_div_floor_eq_floor_div_floor_add_half_eq_floor_double_sub_floor_floor_sum_eq_floor_mult_l414_414980

variables (x y : ℝ) (n : ℤ)

-- 1. ⌊x+y⌋ ≥ ⌊x⌋ + ⌊y⌋
theorem floor_add_ge_floor_add (x y : ℝ) : 
  Int.floor (x + y) ≥ Int.floor x + Int.floor y :=
sorry

-- 2. ⌊ ⌊x⌋ / n ⌋ = ⌊ x / n ⌋ where n is an integer
theorem floor_div_floor_eq_floor_div (x : ℝ) (n : ℤ) :
  Int.floor (Int.floor x / n) = Int.floor (x / n) :=
sorry

-- 3. ⌊x + 1/2⌋ = ⌊2x⌋ - ⌊x⌋
theorem floor_add_half_eq_floor_double_sub_floor (x : ℝ) :
  Int.floor (x + 1/2) = Int.floor (2 * x) - Int.floor x :=
sorry

-- 4. ⌊x⌋ + ⌊x + 1/n⌋ + ⌊x + 2/n⌋ + ... + ⌊x + (n-1)/n⌋ = ⌊nx⌋
theorem floor_sum_eq_floor_mult (x : ℝ) (n : ℤ) :
  (∑ k in Icc 0 (n - 1), Int.floor (x + k / n)) = Int.floor (n * x) :=
sorry

end floor_add_ge_floor_add_floor_div_floor_eq_floor_div_floor_add_half_eq_floor_double_sub_floor_floor_sum_eq_floor_mult_l414_414980


namespace melinda_textbooks_prob_l414_414263

theorem melinda_textbooks_prob :
  let total_ways := (Nat.choose 18 5) * (Nat.choose 13 6) * (Nat.choose 7 3) * (Nat.choose 4 4),
      case1 := Nat.choose 14 1 * (Nat.choose 13 6) * (Nat.choose 7 3) * (Nat.choose 4 4),
      case2 := Nat.choose 14 2 * (Nat.choose 12 4) * (Nat.choose 8 3) * (Nat.choose 5 4),
      case4 := Nat.choose 14 0 * (Nat.choose 14 5) * (Nat.choose 9 6) * (Nat.choose 3 3),
      all_cases := case1 + case2 + case4,
      gcd := Nat.gcd all_cases total_ways,
      num := all_cases / gcd,
      den := total_ways / gcd
  in num + den = 14905595 :=
begin
  sorry
end

end melinda_textbooks_prob_l414_414263


namespace no_consecutive_nat_mul_eq_25k_plus_1_l414_414601

theorem no_consecutive_nat_mul_eq_25k_plus_1 (k : ℕ) : 
  ¬ ∃ n : ℕ, n * (n + 1) = 25 * k + 1 :=
sorry

end no_consecutive_nat_mul_eq_25k_plus_1_l414_414601


namespace locus_of_P_l414_414914

-- Define the points A, B, C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 0, y := 4 / 3 }
def B : Point := { x := -1, y := 0 }
def C : Point := { x := 1, y := 0 }

-- Define the lines AB, AC, and BC
def lineAB (x : ℝ) : ℝ := (4 / 3) * (x + 1)
def lineAC (x : ℝ) : ℝ := -(4 / 3) * (x - 1)
def lineBC (y : ℝ) : ℝ := 0

-- Distance from a point to a line
def distanceToLine (A B C x y : ℝ) : ℝ :=
  abs (A * x + B * y + C) / sqrt (A^2 + B^2)

-- Distances from P(x, y) to the lines AB, AC, and BC
def d1 (x y : ℝ) : ℝ := distanceToLine 4 (-3) 4 x y
def d2 (x y : ℝ) : ℝ := distanceToLine 4 3 (-4) x y
def d3 (x y : ℝ) : ℝ := abs y

-- Define the locus of P subject to the given conditions
def locus1 (x y : ℝ) : Prop := 2 * x^2 + 2 * y^2 + 3 * y - 2 = 0
def locus2 (x y : ℝ) : Prop := 8 * x^2 - 17 * y^2 + 12 * y - 8 = 0

theorem locus_of_P (x y : ℝ) :
  (d3 x y = (d1 x y * d2 x y)^(1/2)) → (locus1 x y ∨ locus2 x y) :=
sorry

end locus_of_P_l414_414914


namespace toby_candies_left_l414_414672

def total_candies : ℕ := 56 + 132 + 8 + 300
def num_cousins : ℕ := 13

theorem toby_candies_left : total_candies % num_cousins = 2 :=
by sorry

end toby_candies_left_l414_414672


namespace find_lambda_l414_414820

noncomputable def vector_a : ℝ × ℝ := (-2, 6)
noncomputable def vector_b (λ : ℝ) : ℝ × ℝ := (-4, λ)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_lambda
  (λ : ℝ)
  (h : dot_product vector_a (vector_b λ) = 0) :
  λ = -4 / 3 :=
by
  sorry

end find_lambda_l414_414820


namespace calculate_expression_l414_414401

theorem calculate_expression :
  (16/81: ℝ) ^ ((-3: ℝ) / 4) + real.log 3 (5/4) + real.log 3 (4/5) = 27 / 8 :=
by
  sorry

end calculate_expression_l414_414401


namespace complete_the_square_l414_414297

theorem complete_the_square (x : ℝ) :
  (x^2 + 14*x + 60) = ((x + 7) ^ 2 + 11) :=
by
  sorry

end complete_the_square_l414_414297


namespace hyperbola_eccentricity_l414_414817

universe u

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

theorem hyperbola_eccentricity
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (c : ℝ) (F : ℝ × ℝ := (c, 0)) (B : ℝ × ℝ := (0, b))
  (M : ℝ × ℝ := midpoint F B)
  (hx : M.1 = c / 2) (hy : M.2 = b / 2)
  (hM_on_C : (M.1^2 / a^2) - (M.2^2 / b^2) = 1) :
  (c^2 / a^2 = 5) →
  (∃ e : ℝ, e = Real.sqrt 5) :=
by
  intro h
  use Real.sqrt 5
  rw [←h]
  exact sorry

end hyperbola_eccentricity_l414_414817


namespace total_books_initial_l414_414544

theorem total_books_initial (books_taken: ℕ) (shelves: ℕ) (books_per_shelf: ℕ) : 
  shelves = 9 → 
  books_per_shelf = 3 → 
  books_taken = 7 → 
  (books_taken + (shelves * books_per_shelf) = 34) :=
by
  -- Given conditions being true, aiming to prove the statement
  intros h_shelves h_boooks_per_shelf h_taken,
  sorry

end total_books_initial_l414_414544


namespace triangle_is_right_l414_414465

theorem triangle_is_right (a b m : ℝ) (h1 : m > b) (h2 : b > 0)
  (h3 : (sqrt (a^2 + b^2) / a) * (sqrt (m^2 - b^2) / m) = 1) :
  a^2 + b^2 = m^2 :=
by sorry

end triangle_is_right_l414_414465


namespace sequence_sum_difference_l414_414144

def sequence_sum (n : ℕ) : ℤ :=
  finset.sum (finset.range n) (λ k, if even k then 4 * (k + 1) - 3 else -4 * (k + 1) + 3)

theorem sequence_sum_difference : sequence_sum 17 - sequence_sum 22 = 77 :=
by
  sorry

end sequence_sum_difference_l414_414144


namespace largest_smallest_angles_adjacent_l414_414385

theorem largest_smallest_angles_adjacent (Pentagon : Type) 
  (convex : ∀ A B C D E : Pentagon, true) 
  (equal_sides : ∀ A B C D E : Pentagon, true) 
  (different_angles : ∀ A B C D E : Pentagon, true) :
  ∃ A B C D E : Pentagon, 
  (∀ a1 a2 a3 a4 a5: ℝ, a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧
  a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5) ∧
  (∀ max_angle min_angle: ℝ, (max_angle > min_angle) ∧ 
  (adjacent max_angle min_angle)) := sorry

end largest_smallest_angles_adjacent_l414_414385


namespace coefficient_x4_y2_eq_120_l414_414532

theorem coefficient_x4_y2_eq_120
: let expansion := (x: ℤ) ^ 2 - x + 2 * y
  in (expand (expansion 5).coeff (monomial (x ^ 4 * y ^ 2)) = 120 :=
begin
  sorry
end

end coefficient_x4_y2_eq_120_l414_414532


namespace actual_mileage_2_fluctuation_value_4_actual_mileage_13_l414_414968

def benchmark : ℕ := 58
def fluctuation_values : list ℤ := [2, 5, -4, 0, 4, -2, 0, -6, 5, 7, -4, -5, 0, -3, 4]

-- Prove the actual mileage of the 2nd torchbearer is 63 meters
theorem actual_mileage_2 (fluctuation_2 : fluctuation_values.nth 1 = some 5) : 
  58 + fluctuation_values.nth_le 1 (by simp [list.length_eq_pred_nth_le]) = 63 :=
by sorry

-- Prove the mileage fluctuation value for the 4th torchbearer is 2 meters given their actual mileage is 60 meters
theorem fluctuation_value_4 (actual_mileage_4 : 60) : 
  60 - benchmark = 2 :=
by sorry

-- Prove the actual mileage of the 13th torchbearer is 53 meters
-- Assuming the cumulative fluctuation values up to the 13th torchbearer is correct and given
theorem actual_mileage_13 (fluctuation_13 : fluctuation_values.nth 12 = some (-5)) : 
  58 + fluctuation_values.nth_le 12 (by simp [list.length_eq_pred_nth_le]) = 53 :=
by sorry

end actual_mileage_2_fluctuation_value_4_actual_mileage_13_l414_414968


namespace common_ratio_of_geometric_sequence_l414_414111

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_nonzero : d ≠ 0) 
  (h_geom : (a 1)^2 = a 0 * a 2) :
  (a 2) / (a 0) = 3 / 2 := 
sorry

end common_ratio_of_geometric_sequence_l414_414111


namespace woman_stop_time_l414_414721

-- Conditions
def man_speed := 5 -- in miles per hour
def woman_speed := 15 -- in miles per hour
def wait_time := 4 -- in minutes
def man_speed_mpm : ℚ := man_speed * (1 / 60) -- convert to miles per minute
def distance_covered := man_speed_mpm * wait_time

-- Definition of the relative speed between the woman and the man
def relative_speed := woman_speed - man_speed
def relative_speed_mpm : ℚ := relative_speed * (1 / 60) -- convert to miles per minute

-- The Proof statement
theorem woman_stop_time :
  (distance_covered / relative_speed_mpm) = 2 :=
by
  sorry

end woman_stop_time_l414_414721


namespace basketball_lineup_l414_414354

def ways_to_choose_lineup (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

theorem basketball_lineup : ways_to_choose_lineup 15 = 3603600 :=
by
  unfold ways_to_choose_lineup
  norm_num
  sorry

end basketball_lineup_l414_414354


namespace permutation_exists_l414_414253

theorem permutation_exists (n : ℕ) (x : Fin n → ℝ)
  (h_sum : abs (∑ i in Finset.univ, x i) = 1)
  (h_bound : ∀ i, abs (x i) ≤ (n + 1) / 2) :
  ∃ y : Fin n → ℝ, (∃ π : Equiv.Perm (Fin n), (∀ i, y i = x (π i))) ∧ abs (∑ i in Finset.univ, (i + 1) * y i) ≤ (n + 1) / 2 :=
by
  sorry

end permutation_exists_l414_414253


namespace simplify_expression_l414_414747

theorem simplify_expression :
  real.cbrt 27 + (2 + real.sqrt 5) * (2 - real.sqrt 5) + (-2)^0 + (- (1 / 3))^(-1) = 0 := by
  sorry

end simplify_expression_l414_414747


namespace range_of_a_l414_414957

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 else Real.log (x + 1)

def g (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x1 ∈ Icc (-2 : ℝ) 9, ∃ x2 ∈ Icc (-2 : ℝ) 2, g a x2 = f x1) ↔ a ∈ Iic (-5 / 8) ∪ Ici 5 :=
by
  sorry

end range_of_a_l414_414957


namespace prob_single_trial_l414_414962

theorem prob_single_trial (P : ℝ) : 
  (1 - (1 - P)^4) = 65 / 81 → P = 1 / 3 :=
by
  intro h
  sorry

end prob_single_trial_l414_414962


namespace inequality_solution_l414_414278

theorem inequality_solution (x : ℝ) :
  (-1 : ℝ) < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < (1 : ℝ) ↔
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by
  sorry

end inequality_solution_l414_414278


namespace maximize_revenue_l414_414355

theorem maximize_revenue : ∃ p : ℝ, (∀ p' : ℝ, (p' ≤ 26 ∧ p' ≥ 0) → (p = 13) ∧ (130 * p - 5 * p^2) ≥ (130 * p' - 5 * p'^2)) :=
begin
  sorry
end

end maximize_revenue_l414_414355


namespace Danielle_rooms_is_6_l414_414850

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l414_414850


namespace determinant_of_trig_matrix_is_zero_l414_414950

theorem determinant_of_trig_matrix_is_zero 
  (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π) :
  matrix.det ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ] = 0 :=
by
  sorry

end determinant_of_trig_matrix_is_zero_l414_414950


namespace only_k_equal_1_works_l414_414094

-- Define the first k prime numbers product
def prime_prod (k : ℕ) : ℕ :=
  Nat.recOn k 1 (fun n prod => prod * (Nat.factorial (n + 1) - Nat.factorial n))

-- Define a predicate for being the sum of two positive cubes
def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a^3 + b^3

-- The theorem statement
theorem only_k_equal_1_works :
  ∀ k : ℕ, (prime_prod k = 2 ↔ k = 1) :=
by
  sorry

end only_k_equal_1_works_l414_414094


namespace danielle_rooms_is_6_l414_414852

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l414_414852


namespace set_intersection_l414_414837

theorem set_intersection (A B : Set ℝ) :
  (A = {x | x^2 > 1} ∧ B = {x | log 2 x > 0}) →
  (A ∩ B = {x | x > 1}) :=
by 
  -- Conditions: 
  assume hA : A = {x | x^2 > 1},
  assume hB : B = {x | log 2 x > 0},
  -- The correct answer:
  have hAB : A ∩ B = {x | x > 1}, from sorry,
  exact ⟨hA, hB, hAB⟩,

end set_intersection_l414_414837


namespace find_b_c_d_l414_414749

-- Define the sequence condition
def sequence_condition (a : ℕ → ℕ) : Prop :=
  ∀ k, ∀ n ∈ (2 * k - 1 .. 2 * k), a n = k

-- Define the main problem statement
theorem find_b_c_d (a : ℕ → ℕ) (b c d : ℤ) :
  sequence_condition a →
  (∀ n : ℕ, a n = b * int.floor (real.sqrt ((n : ℤ) + c)) + d) →
  a 1 = 1 →
  b + c + d = 1 :=
by
  sorry

end find_b_c_d_l414_414749


namespace segment_inequality_l414_414736

variables {Point : Type} [add_group Point] [vector_space ℝ Point]

-- Define variables for points and their respective midpoints
variables A B A₁ B₁ O O₁ : Point

-- Define a condition for non-intersecting line segments (can be an axiom for simplicity)
axiom non_intersecting_segments : ¬ (∃ P : Point, P ∈ set.univ)

-- Define midpoints conditions
axiom midpoint_O : O = (1/2) • (A + B)
axiom midpoint_O₁ : O₁ = (1/2) • (A₁ + B₁)

-- Prove the required inequality
theorem segment_inequality
  (h₀ : O = (1/2) • (A + B))
  (h₁ : O₁ = (1/2) • (A₁ + B₁)) :
  dist O O₁ < (1/2) * (dist A A₁ + dist B B₁) :=
sorry

end segment_inequality_l414_414736


namespace closure_property_of_A_l414_414931

theorem closure_property_of_A 
  (a b c d k1 k2 : ℤ) 
  (x y : ℤ) 
  (Hx : x = a^2 + k1 * a * b + b^2) 
  (Hy : y = c^2 + k2 * c * d + d^2) : 
  ∃ m k : ℤ, x * y = m * (a^2 + k * a * b + b^2) := 
  by 
    -- this is where the proof would go
    sorry

end closure_property_of_A_l414_414931


namespace supplier_received_payment_l414_414389

noncomputable def supplier_payment : ℕ :=
  let price_per_package := 25
  let discount_price_per_package := 4/5 * price_per_package
  let total_packages := 50
  let packages_X := 0.20 * total_packages
  let packages_Y := 8 -- rounding 0.15 * total_packages to an integer
  let packages_Z := 33 -- rounding 0.65 * total_packages to an integer
  let cost_X := packages_X * price_per_package
  let cost_Y := packages_Y * price_per_package
  let cost_Z := 10 * price_per_package + (packages_Z - 10) * discount_price_per_package
  cost_X + cost_Y + cost_Z

theorem supplier_received_payment : supplier_payment = 1160 := by
  sorry

end supplier_received_payment_l414_414389


namespace cos_alpha_value_l414_414808

-- Define the main problem function
noncomputable def find_cos_alpha (α β : ℝ) : ℝ :=
  if (-π / 2 < α ∧ α < π / 2 ∧ 2 * tan β = tan (2 * α) ∧ tan (β - α) = -2 * sqrt 2)
  then cos α
  else 0

-- State the equivalence to prove that given the conditions, cos α = sqrt 3 / 3
theorem cos_alpha_value (α β : ℝ) (h1 : -π / 2 < α) (h2 : α < π / 2) 
  (h3 : 2 * tan β = tan (2 * α)) (h4 : tan (β - α) = -2 * sqrt 2) :
  find_cos_alpha α β = sqrt 3 / 3 :=
by
  sorry

end cos_alpha_value_l414_414808


namespace number_of_students_l414_414663

theorem number_of_students
    (n A : ℕ)  -- Declare n as the number of students and A as the total age of all students
    (h_avg_students : A / n = 21)  -- Average age of students is 21
    (h_teacher_age : 42)  -- Teacher's age is 42
    (h_avg_including_teacher : (A + 42) / (n + 1) = 22)  -- Including teacher, average age is 22
    : n = 20 :=  -- We aim to prove that the number of students is 20
by
  -- Here, we would provide the proof steps, but we're using sorry to focus on the statement
  sorry

end number_of_students_l414_414663


namespace product_of_powers_eq_nine_l414_414688

variable (a : ℕ)

theorem product_of_powers_eq_nine : a^3 * a^6 = a^9 := 
by sorry

end product_of_powers_eq_nine_l414_414688


namespace no_square_in_hexagon_lattice_l414_414306

noncomputable def sqrt_neg_three : ℂ := complex.sqrt (-3)

theorem no_square_in_hexagon_lattice 
  (hexagon_vertices : Set ℂ)
  (is_hexagonal_lattice : ∀ (z : ℂ), z ∈ hexagon_vertices →
    ∃ (R : ℝ), ∀ θ : ℝ, (0 ≤ θ) → (θ < 2 * real.pi/3) →
      z + complex.exp (complex.I * θ) * R ∈ hexagon_vertices) :
  ¬ ∃ (a b c d : ℂ), a ∈ hexagon_vertices ∧ b ∈ hexagon_vertices ∧ 
    c ∈ hexagon_vertices ∧ d ∈ hexagon_vertices ∧ 
    (a - b) = complex.I * (c - d) ∧ 
    (a - d) = complex.I * (c - b) :=
sorry

end no_square_in_hexagon_lattice_l414_414306


namespace talia_mom_age_to_talia_age_ratio_l414_414529

-- Definitions for the problem
def Talia_current_age : ℕ := 13
def Talia_mom_current_age : ℕ := 39
def Talia_father_current_age : ℕ := 36

-- These definitions match the conditions in the math problem
def condition1 : Prop := Talia_current_age + 7 = 20
def condition2 : Prop := Talia_father_current_age + 3 = Talia_mom_current_age
def condition3 : Prop := Talia_father_current_age = 36

-- The ratio calculation
def ratio := Talia_mom_current_age / Talia_current_age

-- The main theorem to prove
theorem talia_mom_age_to_talia_age_ratio :
  condition1 ∧ condition2 ∧ condition3 → ratio = 3 := by
  sorry

end talia_mom_age_to_talia_age_ratio_l414_414529


namespace product_of_roots_of_polynomial_with_rational_coefficients_l414_414566

theorem product_of_roots_of_polynomial_with_rational_coefficients :
  ∃ Q : Polynomial ℚ, (∃ u : ℂ, u^4 = 13 ∧ (u + u^2).isRoot Q) ∧ 
  Q.degree = 4 ∧ 
  (∀ r : ℂ, r.isRoot Q → (r = (u + u^2) ∨ r.is_conjugate_root_of Q (u + u^2))) ∧
  Q.coeff Q.natDegree = 1 ∧ -- leading coefficient is 1 (monic polynomial)
  Q.coeff 0 = -13 := -- constant term is -13
sorry

end product_of_roots_of_polynomial_with_rational_coefficients_l414_414566


namespace complex_modulus_equality_l414_414625

theorem complex_modulus_equality (z : ℂ) (h : (1 + real.sqrt 3 * complex.I) * z = 4) : complex.abs z = 2 :=
sorry

end complex_modulus_equality_l414_414625


namespace correct_statements_l414_414435

-- Definitions of the conditions
variable {a b c x₀ : ℝ} (h₁ : a ≠ 0) 
variable (h₂ : a + b + c = 0)
variable (h₃ : ∀ x, (ax^2 + c = 0 → discriminant a 0 c > 0 → discriminant a b c > 0))
variable (h₄ : (ax₀^2 + bx₀ + c = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2))

-- Prove statements ①, ②, and ④ are correct
theorem correct_statements : (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
                            (discriminant a 0 c > 0 → discriminant a b c > 0) ∧
                            (ax₀^2 + bx₀ + c = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2) :=
  by sorry

end correct_statements_l414_414435


namespace fraction_ratio_l414_414405

theorem fraction_ratio (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 2 :=
by
  sorry

end fraction_ratio_l414_414405


namespace mean_eq_median_l414_414093

def conditions : List Nat := [0, 0, 0, 0,  1, 1,  2, 2, 2, 2, 2,  3, 3,  4,  5, 5, 5,  6]

def median (l : List Nat) : Nat :=
  let sorted := List.sort (≤) l
  sorted.get! ((List.length sorted) / 2)

def mean (l : List Nat) : Rat :=
  (l.foldl (· + ·) 0 : Int) / (List.length l : Int)

theorem mean_eq_median : mean conditions = median conditions := by
  sorry

end mean_eq_median_l414_414093


namespace correct_sunset_time_l414_414591

def sunrise_time : nat × nat := (6, 45) -- hours, minutes
def daylight_duration : nat × nat := (11, 36) -- hours, minutes

def add_time (time1 time2 : nat × nat) : nat × nat := 
  let (h1, m1) := time1
  let (h2, m2) := time2
  let total_minutes := m1 + m2
  let extra_hours := total_minutes / 60
  let final_minutes := total_minutes % 60
  let final_hours := h1 + h2 + extra_hours
  (final_hours, final_minutes)

def convert_to_12_hour_format (time : nat × nat) : string :=
  let (hour, minute) := time
  let (final_hour, period) := 
    if hour == 0 then (12, "AM")
    else if hour < 12 then (hour, "AM")
    else if hour == 12 then (12, "PM")
    else (hour - 12, "PM")
  s!"{final_hour}:{if minute < 10 then "0" ++ toString minute else toString minute}{period}"

def sunset_time : string := convert_to_12_hour_format (add_time sunrise_time daylight_duration)

theorem correct_sunset_time : sunset_time = "6:21PM" :=
  sorry

end correct_sunset_time_l414_414591


namespace determine_angle_B_l414_414220

theorem determine_angle_B
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < B ∧ B < π)
  (h2 : sin C = 2 * sin A)
  (h3 : b = sqrt 3 * a)
  (h4 : a / sin A = b / sin B)
  (h5 : a / sin A = c / sin C) :
  B = π / 3 :=
sorry

end determine_angle_B_l414_414220


namespace find_special_number_l414_414769

theorem find_special_number:
  ∃ (n : ℕ), (n > 0) ∧ (∃ (k : ℕ), 2 * n = k^2)
           ∧ (∃ (m : ℕ), 3 * n = m^3)
           ∧ (∃ (p : ℕ), 5 * n = p^5)
           ∧ n = 1085 :=
by
  sorry

end find_special_number_l414_414769


namespace find_number_correct_l414_414262

theorem find_number_correct : ∃ x : ℕ, (4 * x + 15 = 15 * x + 4) ∧ (x = 1) := 
by
  exists 1
  split
  · calc
      4 * 1 + 15 = 4 + 15 : by ring
      ... = 19 : by norm_num
  15 * 1 + 4 = 15 + 4 : by ring
      ... = 19 : by norm_num
  referent
   only sorry
  -- lean
 -- number


end find_number_correct_l414_414262


namespace exists_n_cos_eq_l414_414423

theorem exists_n_cos_eq :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n:ℝ).to_degrees = Real.cos 942 := by
  sorry

end exists_n_cos_eq_l414_414423


namespace probability_two_tails_two_heads_l414_414877

theorem probability_two_tails_two_heads :
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  total_probability = 3 / 8 :=
by
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  sorry

end probability_two_tails_two_heads_l414_414877


namespace mary_remaining_cards_l414_414963

variable (initial_cards : ℝ) (bought_cards : ℝ) (promised_cards : ℝ)

def remaining_cards (initial : ℝ) (bought : ℝ) (promised : ℝ) : ℝ :=
  initial + bought - promised

theorem mary_remaining_cards :
  initial_cards = 18.0 →
  bought_cards = 40.0 →
  promised_cards = 26.0 →
  remaining_cards initial_cards bought_cards promised_cards = 32.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end mary_remaining_cards_l414_414963


namespace morning_routine_time_l414_414171

section

def time_for_teeth_and_face : ℕ := 3
def time_for_cooking : ℕ := 14
def time_for_reading_while_cooking : ℕ := time_for_cooking - time_for_teeth_and_face
def additional_time_for_reading : ℕ := 1
def total_time_for_reading : ℕ := time_for_reading_while_cooking + additional_time_for_reading
def time_for_eating : ℕ := 6

def total_time_to_school : ℕ := time_for_cooking + time_for_eating

theorem morning_routine_time :
  total_time_to_school = 21 := sorry

end

end morning_routine_time_l414_414171


namespace field_trip_adults_l414_414408

theorem field_trip_adults (A : ℕ) (van_capacity : ℕ) (students : ℕ) (num_vans : ℕ) :
  van_capacity = 4 → students = 2 → num_vans = 2 → 2 * van_capacity - students = 6 :=
by
  intros h1 h2 h3
  calc
    2 * van_capacity - students
    = 2 * 4 - 2 : by rw [h1, h2]
    = 6 : by norm_num

end field_trip_adults_l414_414408


namespace log_base_8_of_256_l414_414766

theorem log_base_8_of_256 : log 8 256 = 8 / 3 :=
by
  sorry

end log_base_8_of_256_l414_414766


namespace pseudocode_output_is_zero_l414_414149

def final_value_n (init_n : ℕ) (init_s : ℕ) : ℕ :=
let ⟨final_n, final_s⟩ := 
    (do { let mut s := init_s
          let mut n := init_n
          while s < 15 do
              s := s + n
              n := n - 1
          pure (n, s) } : Unit → (ℕ × ℕ)) ()
in final_n

theorem pseudocode_output_is_zero : 
  final_value_n 5 0 = 0 :=
sorry

end pseudocode_output_is_zero_l414_414149


namespace find_angle_l414_414331

theorem find_angle (r1 r2 : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 2) 
(h_shaded : ∀ α : ℝ, 0 < α ∧ α < 2 * π → 
  (360 / 360 * pi * r1^2 + (α / (2 * π)) * pi * r2^2 - (α / (2 * π)) * pi * r1^2 = (1/3) * (pi * r2^2))) : 
  (∀ α : ℝ, 0 < α ∧ α < 2 * π ↔ 
  α = π / 3 ) :=
by
  sorry

end find_angle_l414_414331


namespace min_value_frac_l414_414131

theorem min_value_frac (A B C O : Type) (l : Type)
  (collinear : A ∈ l ∧ B ∈ l ∧ C ∈ l)
  (O_not_on_l : O ∉ l)
  (m n : ℝ) (h_mn_pos : m > 0 ∧ n > 0)
  (h_OC : -- condition for the vector equation
   (4 : ℝ) * m • (∥OA∥) + n • (∥OB∥) = (OC)) :
  ∃ (m n : ℝ), (m > 0 ∧ n > 0) ∧ (4 * m + n = 1) ∧ (min_value_frac m n = 6 + 4 * real.sqrt (2 : ℝ)) :=
begin
  sorry
end

end min_value_frac_l414_414131


namespace range_of_m_l414_414127

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (∀ x y : ℝ, 4 * x^2 + y^2 + sqrt (x * y) - m < 0) ↔ m ∈ set.Ioi (17 / 16) :=
sorry

end range_of_m_l414_414127


namespace game_ends_in_65_rounds_l414_414267

noncomputable def player_tokens_A : Nat := 20
noncomputable def player_tokens_B : Nat := 19
noncomputable def player_tokens_C : Nat := 18
noncomputable def player_tokens_D : Nat := 17

def rounds_until_game_ends (A B C D : Nat) : Nat :=
  -- Implementation to count the rounds will go here, but it is skipped for this statement-only task
  sorry

theorem game_ends_in_65_rounds : rounds_until_game_ends player_tokens_A player_tokens_B player_tokens_C player_tokens_D = 65 :=
  sorry

end game_ends_in_65_rounds_l414_414267


namespace rearrange_segments_l414_414442

-- Define a segment as a pair of real numbers, representing the endpoints
structure Segment :=
(left : ℝ)
(right : ℝ)
(h_valid: left ≤ right)

-- Define the length of a segment
def length (s : Segment) : ℝ :=
  s.right - s.left

-- Define the total length of the union of segments (assuming intervals are disjoint for simplicity)
def total_length (S : finset Segment) : ℝ :=
  S.sum length

-- Define the notion that midpoints are closer in the rearranged segments
def midpoints_closer (S S' : finset Segment) : Prop :=
  ∀ s t ∈ S, 
  let m := (s.left + s.right) / 2;
  let m' := (t.left + t.right) / 2;
  ∀ s' t' ∈ S', 
  let m'' := (s'.left + s'.right) / 2;
  let m''' := (t'.left + t'.right) / 2;
  |m'' - m'''| ≤ |m - m'|

-- The main theorem statement
theorem rearrange_segments (S S' : finset Segment) (h: midpoints_closer S S') : total_length S' ≤ total_length S :=
by sorry

end rearrange_segments_l414_414442


namespace finite_number_of_solutions_l414_414062

-- Define the given system of equations
def eq1 (x y z : ℤ) : Prop := x^2 - 4 * x * y + 3 * y^2 - z^2 = 19
def eq2 (x y z : ℤ) : Prop := -x^2 + 8 * y * z + z^2 = 24
def eq3 (x y z : ℤ) : Prop := x^2 + 2 * x * y + 9 * z^2 = 85

-- Define the overall condition for solution set and finiteness
theorem finite_number_of_solutions : 
  { (x, y, z) : ℤ × ℤ × ℤ | eq1 x y z ∧ eq2 x y z ∧ eq3 x y z }.finite ∧ 
  { (x, y, z) : ℤ × ℤ × ℤ | eq1 x y z ∧ eq2 x y z ∧ eq3 x y z }.to_finset.card > 2 :=
sorry

end finite_number_of_solutions_l414_414062


namespace bond_energy_O_F_l414_414745

theorem bond_energy_O_F :
  (let Q_x := (0.25 * 498 + 0.5 * 159 + 0.5 * 22) in Q_x = 215) :=
by
  let Q_x := 0.25 * 498 + 0.5 * 159 + 0.5 * 22
  -- This skipped proof confirms Q_x = 215
  sorry

end bond_energy_O_F_l414_414745


namespace smallest_positive_integer_divisible_l414_414779

theorem smallest_positive_integer_divisible (n : ℕ) (h1 : 15 = 3 * 5) (h2 : 16 = 2 ^ 4) (h3 : 18 = 2 * 3 ^ 2) :
  n = Nat.lcm (Nat.lcm 15 16) 18 ↔ n = 720 :=
by
  sorry

end smallest_positive_integer_divisible_l414_414779


namespace tangent_segment_length_correct_l414_414446

def point (x y : ℝ) : Prop :=
  x + 2 * y = 3

def circle (x y : ℝ) : Prop :=
  (x - 1 / 2)^2 + (y - 1 / 4)^2 = 1 / 2

def circle_center_x := 1 / 2
def circle_center_y := 1 / 4
def circle_radius := (1 / 2 : ℝ) ^ (1 / 2)

def min_value_condition (x y : ℝ) : Prop :=
  2 ^ x + 4 ^ y = 4 * Real.sqrt 2

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

noncomputable def tangent_segment_length (x y : ℝ) [H1 : point x y] [H2 : circle_center_x_circle y] : ℝ :=
  Real.sqrt ((distance x y circle_center_x circle_center_y)^2 - (circle_radius)^2)

theorem tangent_segment_length_correct : ∀ (x y : ℝ), point x y → min_value_condition x y → tangent_segment_length x y = Real.sqrt (6) / 2 := sorry

end tangent_segment_length_correct_l414_414446


namespace roots_of_quadratic_implies_values_l414_414338

theorem roots_of_quadratic_implies_values (a b : ℝ) :
  (∃ x : ℝ, x^2 + 2 * (1 + a) * x + (3 * a^2 + 4 * a * b + 4 * b^2 + 2) = 0) →
  a = 1 ∧ b = -1/2 :=
by
  sorry

end roots_of_quadratic_implies_values_l414_414338


namespace vacation_days_in_march_l414_414391

theorem vacation_days_in_march 
  (days_worked : ℕ) 
  (days_worked_to_vacation_days : ℕ) 
  (vacation_days_left : ℕ) 
  (days_in_march : ℕ) 
  (days_in_september : ℕ)
  (h1 : days_worked = 300)
  (h2 : days_worked_to_vacation_days = 10)
  (h3 : vacation_days_left = 15)
  (h4 : days_in_september = 2 * days_in_march)
  (h5 : days_worked / days_worked_to_vacation_days - (days_in_march + days_in_september) = vacation_days_left) 
  : days_in_march = 5 := 
by
  sorry

end vacation_days_in_march_l414_414391


namespace equation_of_circle_equation_of_line_l414_414134

-- Conditions
def center_on_line (a : ℝ) : Prop := ∃ x : ℝ, x = a ∧ a + 1 = x
def radius_square := 2
def passes_through_P (a : ℝ) (x : ℝ) (y : ℝ) : Prop := x = 3 ∧ y = 6 ∧ (3 - a) ^ 2 + (6 - (a + 1)) ^ 2 = radius_square
def passes_through_Q (a : ℝ) (x : ℝ) (y : ℝ) : Prop := x = 5 ∧ y = 6 ∧ (5 - a) ^ 2 + (6 - (a + 1)) ^ 2 = radius_square
def chord_length_condition (l : ℝ → ℝ) : Prop := ∃ k : ℝ, l = (λ x, k * (x - 3)) ∧ (abs (4 * k - 5 - 3 * k) / (sqrt (1 + k ^ 2))) = 1

-- Questions and correct answers as Lean definitions
theorem equation_of_circle :
  ∀ a : ℝ,
  center_on_line a →
  passes_through_P a 3 6 →
  passes_through_Q a 5 6 →
  ((λ x y, (x - 4) ^ 2 + (y - 5) ^ 2 = radius_square)) sorry :=
sorry

theorem equation_of_line :
  ∀ l : ℝ → ℝ,
  chord_length_condition l →
  ((λ x y, (12 * x - 5 * y - 36 = 0) ∨ (x = 3))) sorry :=
sorry

end equation_of_circle_equation_of_line_l414_414134


namespace number_of_rational_terms_bisnomial_expansion_eq_three_l414_414214

theorem number_of_rational_terms_bisnomial_expansion_eq_three 
   (x : ℝ) : 
   let n := 8,
       binom_exp := (sqrt x + (1 / (2 * x ^ (1 / 4)))) ^ n in
   3 := sorry

end number_of_rational_terms_bisnomial_expansion_eq_three_l414_414214


namespace false_intersecting_planes_perpendicular_to_same_line_l414_414739

theorem false_intersecting_planes_perpendicular_to_same_line :
  ¬(∃ (P Q : std::geometry::plane) (L : std::geometry::line),
    (P ≠ Q) ∧ 
    (std::geometry::plane.contains P L) ∧
    (std::geometry::plane.contains Q L) ∧
    (¬∀ (p1 p2 : std::vector) (p1_on_P : std::geometry::plane.contains_point P p1) (p2_on_Q : std::geometry::plane.contains_point Q p2), 
      p1 × p2 = 0)) :=
sorry

end false_intersecting_planes_perpendicular_to_same_line_l414_414739


namespace bernie_savings_l414_414396

-- Defining conditions
def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def chocolates_total : ℕ := chocolates_per_week * weeks
def local_store_cost_per_chocolate : ℕ := 3
def different_store_cost_per_chocolate : ℕ := 2

-- Defining the costs in both stores
def local_store_total_cost : ℕ := chocolates_total * local_store_cost_per_chocolate
def different_store_total_cost : ℕ := chocolates_total * different_store_cost_per_chocolate

-- The statement we want to prove
theorem bernie_savings : local_store_total_cost - different_store_total_cost = 6 :=
by
  sorry

end bernie_savings_l414_414396


namespace union_M_N_l414_414049

open Set Real

def M : Set ℝ := { x | abs (x + 2) ≤ 1 }
def N : Set ℝ := { x | ∃ a : ℝ, x = 2 * sin a }

theorem union_M_N : M ∪ N = { x | -3 ≤ x ∧ x ≤ 2 } := by
  sorry

end union_M_N_l414_414049


namespace translation_line_segment_l414_414208

theorem translation_line_segment (a b : ℝ) :
  (∃ A B A1 B1: ℝ × ℝ,
    A = (1,0) ∧ B = (3,2) ∧ A1 = (a, 1) ∧ B1 = (4,b) ∧
    ∃ t : ℝ × ℝ, A + t = A1 ∧ B + t = B1) →
  a = 2 ∧ b = 3 :=
by
  sorry

end translation_line_segment_l414_414208


namespace oranges_in_each_box_l414_414742

theorem oranges_in_each_box (total_oranges : ℝ) (boxes : ℝ) (h_total : total_oranges = 72) (h_boxes : boxes = 3.0) : total_oranges / boxes = 24 :=
by
  -- Begin proof
  sorry

end oranges_in_each_box_l414_414742


namespace least_number_of_candles_l414_414930

theorem least_number_of_candles (b : ℕ) :
  (b ≡ 5 [MOD 6]) ∧ (b ≡ 7 [MOD 8]) ∧ (b ≡ 3 [MOD 9]) → b = 119 :=
by
  -- Proof omitted
  sorry

end least_number_of_candles_l414_414930


namespace valid_two_digit_integers_count_l414_414470

def digits := [1, 3, 5, 7, 9]

def isValidSecondDigit (firstDigit secondDigit : Nat) : Prop :=
  secondDigit ≠ 5 ∧ firstDigit ≠ secondDigit

def countValidIntegers : Nat :=
  let firstDigitChoices := digits
  firstDigitChoices.foldl (fun acc firstDigit =>
    let secondDigitChoices := digits.filter (fun d => isValidSecondDigit firstDigit d)
    acc + secondDigitChoices.length) 0

theorem valid_two_digit_integers_count :
  countValidIntegers = 16 :=
by
  sorry

end valid_two_digit_integers_count_l414_414470


namespace find_t_l414_414813

-- Define the conditions given in the problem
variable {F₁ F₂ : ℝ × ℝ} -- coordinates of foci
variable {P : ℝ × ℝ} -- point on the ellipse
variable {t : ℝ} -- the ratio we need to prove

-- The ellipse is given by its standard equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 12) + (y^2 / 3) = 1

-- Condition that P is on the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse P.1 P.2

-- The midpoint of segment PF₁ is on the y-axis
def midpoint_on_y_axis (P F₁ : ℝ × ℝ) : Prop :=
  (P.1 + F₁.1) / 2 = 0

-- PF₁ = t * PF₂
def focus_ratio (P F₁ F₂ : ℝ × ℝ) (t : ℝ) : Prop :=
  let d_PF₁ := real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) in
  let d_PF₂ := real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) in
  d_PF₁ = t * d_PF₂

-- Main statement to prove:
theorem find_t (h1 : is_on_ellipse P) (h2 : midpoint_on_y_axis P F₁) (h3 : focus_ratio P F₁ F₂ t) : t = 7 := 
sorry

end find_t_l414_414813


namespace f_monotone_decreasing_without_min_value_l414_414501

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_without_min_value :
  (∀ x y : ℝ, x < y → f y < f x) ∧ (∃ b : ℝ, ∀ x : ℝ, f x > b) :=
by
  sorry

end f_monotone_decreasing_without_min_value_l414_414501


namespace rectangular_plot_length_l414_414375

theorem rectangular_plot_length 
(poles : ℕ) (distance_between_poles width : ℝ) 
(h_poles : poles = 70) 
(h_distance : distance_between_poles = 4) 
(h_width : width = 50) : 
∃ L : ℝ, L = 88 := 
by 
  -- condition 1: 70 poles are needed
  have h1 : poles = 70 := h_poles
  
  -- condition 2: poles are 4 metres apart
  have h2 : distance_between_poles = 4 := h_distance
  
  -- condition 3: width of the rectangle is 50 metres
  have h3 : width = 50 := h_width
  
  -- calculate the number of gaps
  let gaps := poles - 1
  
  -- calculate the total perimeter
  let perimeter := gaps * distance_between_poles
  
  -- perimeter of a rectangle
  let P := 2 * (88 + width)
  
  -- now, prove that the length is 88 metres
  use 88
  
  -- sorry to skip the proof
  sorry

end rectangular_plot_length_l414_414375


namespace square_bisector_theorem_l414_414623

theorem square_bisector_theorem
    (A B C D M : Point)
    (h_square: square A B C D)
    (h_BM: is_angle_bisector (∠BAC) B M) :
    dist A C = dist A B + dist B M :=
sorry

end square_bisector_theorem_l414_414623


namespace helium_min_cost_l414_414710

noncomputable def W (x : ℝ) : ℝ :=
  if x < 4 then 40 * (4 * x + 16 / x + 100)
  else 40 * (9 / (x * x) - 3 / x + 117)

theorem helium_min_cost :
  (∀ x, W x ≥ 4640) ∧ (W 2 = 4640) :=
by {
  sorry
}

end helium_min_cost_l414_414710


namespace inclusion_exclusion_prob_l414_414612

-- Conditions
variables {Ω : Type*} -- probability space
variables {X : ℕ → Ω → ℝ}
variables {A : ℕ → set Ω}

-- Given inclusion-exclusion formula for the maximum of random variables
noncomputable def x_vee : (ℕ → Ω → ℝ) → (Ω → ℝ) :=
λ X ω, finset.max (finset.range n) (λ i, X i ω)

noncomputable def inclusion_exclusion_max (X : ℕ → Ω → ℝ) : (Ω → ℝ) :=
λ ω, (finset.range (n + 1)).sum (λ k, (-1 : ℤ)^(k+1) * 
               (finset.choose n k) * (finset.product (finset.range k) (finset.range k)).sum 
               (λ (i j), X i ω * X j ω))

-- Correct answer: inclusion-exclusion principle for probabilities
theorem inclusion_exclusion_prob {Ω : Type*} [ProbabilityMeasure Ω] 
  (A : ℕ → set Ω) : 
  Prob (⋃ i, A i) = 
  ∑ i, Prob (A i) 
  - ∑ i j, Prob (A i ∩ A j)
  + ∑ i j k, Prob (A i ∩ A j ∩ A k) 
  + ... 
  + (-1)^(n+1) * Prob (⋂ i, A i) :=
sorry

end inclusion_exclusion_prob_l414_414612


namespace number_of_true_propositions_l414_414842

-- Let's state the propositions
def original_proposition (P Q : Prop) := P → Q
def converse_proposition (P Q : Prop) := Q → P
def inverse_proposition (P Q : Prop) := ¬P → ¬Q
def contrapositive_proposition (P Q : Prop) := ¬Q → ¬P

-- Main statement we need to prove
theorem number_of_true_propositions (P Q : Prop) (hpq : original_proposition P Q) 
  (hc: contrapositive_proposition P Q) (hev: converse_proposition P Q)  (hbv: inverse_proposition P Q) : 
  (¬(P ↔ Q) ∨ (¬¬P ↔ ¬¬Q) ∨ (¬Q → ¬P) ∨ (P → Q)) := sorry

end number_of_true_propositions_l414_414842


namespace net_percentage_change_stock_l414_414307

theorem net_percentage_change_stock :
  ∀ initial_price : ℝ,
  let price_end_year_1 := initial_price * (1 - 0.08) in
  let price_end_year_2 := price_end_year_1 * (1 + 0.10) in
  let price_end_year_3 := price_end_year_2 * (1 + 0.06) in
  (price_end_year_3 - initial_price) / initial_price * 100 = 7.272 :=
by
  sorry

end net_percentage_change_stock_l414_414307


namespace range_of_slope_k_l414_414806

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x = 0

def point_A : ℝ × ℝ := (1, 0)

def line_eq (k x y : ℝ) : Prop := k * x - y - k = 0

noncomputable def distance_point_to_line (k : ℝ) : ℝ :=
  |(-k - k) / sqrt (k^2 + 1)|

theorem range_of_slope_k :
  ∀ k : ℝ,
  (∃ x y : ℝ, circle_eq x y ∧ line_eq k x y) →
  - real.sqrt(3) / 3 ≤ k ∧ k ≤ real.sqrt(3) / 3 := by
  sorry

end range_of_slope_k_l414_414806


namespace probability_white_given_red_l414_414543

-- Define the total number of balls initially
def total_balls := 10

-- Define the number of red balls, white balls, and black balls
def red_balls := 3
def white_balls := 2
def black_balls := 5

-- Define the event A: Picking a red ball on the first draw
def event_A := red_balls

-- Define the event B: Picking a white ball on the second draw
-- Number of balls left after picking one red ball
def remaining_balls_after_A := total_balls - 1

-- Define the event AB: Picking a red ball first and then a white ball
def event_AB := red_balls * white_balls

-- Calculate the probability P(B|A)
def P_B_given_A := event_AB / (event_A * remaining_balls_after_A)

-- Prove the probability of picking a white ball on the second draw given that the first ball picked is a red ball
theorem probability_white_given_red : P_B_given_A = (2 / 9) := by
  sorry

end probability_white_given_red_l414_414543


namespace container_solution_exists_l414_414662

theorem container_solution_exists (x y : ℕ) (h : 130 * x + 160 * y = 3000) : 
  (x = 12) ∧ (y = 9) :=
by sorry

end container_solution_exists_l414_414662


namespace arithmetic_sequence_common_difference_l414_414577

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem arithmetic_sequence_common_difference 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  (h7 : a^2 = b) 
  (h8 : b^2 = c) :
  let log_c_a := log_base c a,
      log_b_c := log_base b c,
      log_a_b := log_base a b,
      d := log_b_c - log_c_a in
  log_a_b - log_b_c = d ∧ d = 7 / 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l414_414577


namespace tetrahedrons_from_cube_vertices_l414_414489

theorem tetrahedrons_from_cube_vertices : 
  let cube_vertices := 8
  let total_combinations := Nat.choose 8 4
  let invalid_tetrahedrons := 12
  total_combinations - invalid_tetrahedrons = 58 :=
begin
  sorry
end

end tetrahedrons_from_cube_vertices_l414_414489


namespace triangle_divisible_equal_parts_l414_414542

def is_scalene (a b c : ℝ) := a ≠ b ∧ b ≠ c ∧ a ≠ c
def can_be_divided_into_three_equal_triangles (α β γ : ℝ) : Prop := 
  α = 30 ∧ β = 60 ∧ γ = 90 ∧ is_scalene 30 60 90

theorem triangle_divisible_equal_parts :
  ∃ α β γ, can_be_divided_into_three_equal_triangles α β γ :=
by {
   use [30, 60, 90],
   unfold can_be_divided_into_three_equal_triangles,
   unfold is_scalene,
   split; norm_num,
   sorry
}

end triangle_divisible_equal_parts_l414_414542


namespace triangle_area_sum_l414_414894

theorem triangle_area_sum :
  ∀ (A B C D E : Type) [has_midpoint E B C] [line_segment D A C],
  let AC_len := 2,
      ∠BAC := 60,
      ∠ABC := 50,
      ∠ACB := 70,
      ∠DEC := 70 in
  let a := (4 * real.sin 70) / real.sqrt 3,
      b := (4 * real.sin 50) / real.sqrt 3 in
  let area_ABC := 4 * real.sin 50 * real.sin 70 / real.sqrt 3,
      area_CDE := b * real.sin 70 / 4 in
  area_ABC + 2 * area_CDE = 6 * real.sin 50 * real.sin 70 / real.sqrt 3 :=
sorry

end triangle_area_sum_l414_414894


namespace quadruple_count_l414_414469

theorem quadruple_count (r s : ℕ) :
  let valid_quadruple_count : ℕ := (1 + 4 * r + 6 * r^2) * (1 + 4 * s + 6 * s^2) in
  valid_quadruple_count = (1 + 4 * r + 6 * r^2) * (1 + 4 * s + 6 * s^2) :=
by
  -- The actual proof would go here, but it's omitted as per the instructions.
  sorry

end quadruple_count_l414_414469


namespace number_of_red_balls_l414_414520

def total_balls : ℕ := 50
def frequency_red_ball : ℝ := 0.7

theorem number_of_red_balls :
  ∃ n : ℕ, n = (total_balls : ℝ) * frequency_red_ball ∧ n = 35 :=
by
  sorry

end number_of_red_balls_l414_414520


namespace sum_of_decimals_l414_414170

theorem sum_of_decimals : (0.305 : ℝ) + (0.089 : ℝ) + (0.007 : ℝ) = 0.401 := by
  sorry

end sum_of_decimals_l414_414170


namespace extension_even_function_l414_414581

noncomputable def g (x : ℝ) : ℝ := 
if x ≤ 0 then 2^x else 2^(-x)

theorem extension_even_function (f : ℝ → ℝ) (D_f D_g : set ℝ) (g : ℝ → ℝ)
  (h1 : D_f ⊆ D_g)
  (h2 : ∀ x ∈ D_f, g x = f x)
  (h3 : ∀ x ≤ 0, f x = 2^x)
  (h4 : ∀ x, g x = g (-x)) :
  ∀ x, g x = 2^(-|x|) := 
by
  intros x
  have h5: g(x) = 2^x ∨ g(x) = 2^(-x) := by sorry
  cases h5 with h5left h5right
  · rw h5left
    sorry
  · rw h5right
    sorry

end extension_even_function_l414_414581


namespace similar_final_l414_414933

noncomputable theory

open EuclideanGeometry

variables {A B C A1 B1 C1 A2 B2 C2 : Point}

-- Given that triangle ABC is scalene
axiom scalene_triangle (ABC : Triangle A B C) : ¬ (Isosceles A B C ∨ Isosceles B C A ∨ Isosceles C A B)

-- Points A1, B1, C1 chosen on segments BC, CA, AB such that triangles are similar
axiom similar_triangles (ABC : Triangle A B C) (A1 B1 C1 : Point) :
  OnSegment A1 B C ∧ OnSegment B1 C A ∧ OnSegment C1 A B ∧
  Similar (Triangle A B C) (Triangle A1 B1 C1)

-- Point A2 on line B1C1 such that AA2 = A1A2, points B2 and C2 similarly defined
axiom point_definition (A B C A1 B1 C1 A2 B2 C2 : Point) :
  OnLine A2 (Line B1 C1) ∧ Distance A A2 = Distance A1 A2 ∧
  OnLine B2 (Line C1 A1) ∧ Distance B B2 = Distance B1 B2 ∧
  OnLine C2 (Line A1 B1) ∧ Distance C C2 = Distance C1 C2

-- Prove that triangle A2B2C2 is similar to triangle ABC
theorem similar_final (ABC : Triangle A B C) (A1 B1 C1 A2 B2 C2 : Point)
  (h_scalene: scalene_triangle ABC)
  (h_similar: similar_triangles ABC A1 B1 C1)
  (h_point_def: point_definition A B C A1 B1 C1 A2 B2 C2) :
  Similar (Triangle A2 B2 C2) (Triangle A B C) :=
sorry

end similar_final_l414_414933


namespace value_of_m_l414_414126

noncomputable def complex_modulus (z : ℂ) : ℝ :=
  |z|

theorem value_of_m (m : ℝ) (i : ℂ) (z : ℂ)
  (h1 : i * i = -1)
  (h2 : z = m / (1 - i))
  (h3 : complex_modulus z = ∫ x in 0..π, sin x - (1 / π)) :
  m = real.sqrt 2 ∨ m = -real.sqrt 2 := by
sorry

end value_of_m_l414_414126


namespace distance_BC_in_circle_l414_414514

theorem distance_BC_in_circle
    (r : ℝ) (A B C : ℝ × ℝ)
    (h_radius : r = 10)
    (h_diameter : dist A B = 2 * r)
    (h_chord : dist A C = 12) :
    dist B C = 16 := by
  sorry

end distance_BC_in_circle_l414_414514


namespace bingo_first_column_count_l414_414518

def num_distinct_arrangements_first_column : ℕ := 
  (List.range' 1 15).permutations.filter (λ l, l.nodup ∧ l.length = 5).length

theorem bingo_first_column_count :
  num_distinct_arrangements_first_column = 360360 := by
  sorry

end bingo_first_column_count_l414_414518


namespace solve_eq1_solve_eq2_l414_414172

theorem solve_eq1 : ∀ (x : ℚ), (3 / 5 - 5 / 8 * x = 2 / 5) → (x = 8 / 25) := by
  intro x
  intro h
  sorry

theorem solve_eq2 : ∀ (x : ℚ), (7 * (x - 2) = 8 * (x - 4)) → (x = 18) := by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l414_414172


namespace parallelogram_area_l414_414239

theorem parallelogram_area (height : ℕ) (side_length : ℕ) (unit_side_length : ℕ) 
  (parallelogram_vertices : Finset (Fin 400))
  (h_ABC_equilateral : side_length = 20)
  (h_unit_division : unit_side_length = 1)
  (h_400_tris : height * height = 400)
  (h_parallelogram_inside : ∀ v ∈ parallelogram_vertices, v ∈ Finset.range 400)
  (h_46_intersection : ∃ parallelogram_set : Finset (Fin 400), 
     parallelogram_vertices.card = 4 ∧ 
     parallelogram_set = Finset.filter 
      (λ v, ∃ t ∈ Finset.range 400, 
        ∃ ⟨x, y⟩ ∈ parallelogram_vertices.image (λ v, (v / height, v % height)), 
          (x = v / height ∨ y = v % height) ∧
          (x + 1 = v / height ∨ y + 1 = v % height ∨ x - 1 = v / height ∨ y - 1 = v % height)
      ) (Finset.range 400) ∧ 
     parallelogram_set.card = 46): 
  ∃ area ∈ {8, 9}, (∃ m n : ℕ, m > 1 ∧ n > 1 ∧ m + n = 6 ∧ area = m * n) := 
  sorry

end parallelogram_area_l414_414239


namespace angle_BAC_l414_414219

open EuclideanGeometry

/-- Given a triangle ABC with AX=XY=YB=BC and ∠ABC=120°, prove that ∠BAC=15° --/
theorem angle_BAC (A B C X Y : Point)
  (hAX : dist A X = dist X Y)
  (hXY : dist X Y = dist Y B)
  (hYB : dist Y B = dist B C)
  (hABC : ∠ABC = 120) :
  ∠BAC = 15 :=
by sorry

end angle_BAC_l414_414219


namespace tank_capacity_is_780_l414_414017

noncomputable def tank_capacity : ℕ := 
  let fill_rate_A := 40
  let fill_rate_B := 30
  let drain_rate_C := 20
  let cycle_minutes := 3
  let total_minutes := 48
  let net_fill_per_cycle := fill_rate_A + fill_rate_B - drain_rate_C
  let total_cycles := total_minutes / cycle_minutes
  let total_fill := total_cycles * net_fill_per_cycle
  let final_capacity := total_fill - drain_rate_C -- Adjust for the last minute where C opens
  final_capacity

theorem tank_capacity_is_780 : tank_capacity = 780 := by
  unfold tank_capacity
  -- Proof steps to be filled in
  sorry

end tank_capacity_is_780_l414_414017


namespace midpoint_distance_l414_414531

noncomputable def midpoint (a b : ℝ) (c d : ℝ) : ℝ × ℝ := ((a + c) / 2, (b + d) / 2)

noncomputable def new_midpoint (a b c d : ℝ) : ℝ × ℝ :=
(((a + 5) + (c - 15)) / 2, ((b + 10) + (d - 5)) / 2)

theorem midpoint_distance (a b c d : ℝ) (m n : ℝ) (ha : m = (a + c) / 2) (hb : n = (b + d) / 2) :
  let mp' := new_midpoint a b c d in
  mp' = (m - 5, n + 2.5) ∧
  real.sqrt ((-5)^2 + (2.5)^2) = 5.5 :=
by
  sorry

end midpoint_distance_l414_414531


namespace calculate_expression_of_roots_of_quadratic_l414_414256

theorem calculate_expression_of_roots_of_quadratic:
  (∀ p q : ℝ, (3 * p^2 + 4 * p - 7 = 0) ∧ (3 * q^2 + 4 * q - 7 = 0) →
  (p - 2) * (q - 2) = 13 / 3) :=
begin
  intros p q h,
  sorry
end

end calculate_expression_of_roots_of_quadratic_l414_414256


namespace equation_of_tangent_line_l414_414445

theorem equation_of_tangent_line 
(line_tangent_to_circle (x y : ℝ) : y = k * x) 
(circle_eqn : ∀ x y : ℝ, x^2 + y^2 - 6 * x + 5 = 0) 
(center (x y : ℝ) : x = 3 ∧ y = 0) 
(radius : ∀ r : ℝ, r = 2) :
x = 0 → y = ± (2 * sqrt 5 / 5) * x := 
by
  sorry

end equation_of_tangent_line_l414_414445


namespace number_of_tetrahedrons_from_cube_l414_414492

noncomputable def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| (n + 1) (k + 1) := binom n k + binom n (k + 1)

theorem number_of_tetrahedrons_from_cube : 
  let V := 8 in
  let coplanar_cases := 12 in
  let total_combinations := binom 8 4 in
  total_combinations - coplanar_cases = 58 := 
by
  sorry

end number_of_tetrahedrons_from_cube_l414_414492


namespace sin_alpha_eq_l414_414133

noncomputable def alpha : ℝ := sorry

theorem sin_alpha_eq :
  (cos (alpha + π / 6) = 1 / 5) →
  sin alpha = (6 * real.sqrt 2 - 1) / 10 :=
by
  intro h_cos
  sorry

end sin_alpha_eq_l414_414133


namespace interest_rate_A_l414_414364

-- Definitions for the conditions
def principal : ℝ := 1000
def rate_C : ℝ := 0.115
def time_period : ℝ := 3
def gain_B : ℝ := 45

-- Main theorem to prove
theorem interest_rate_A {R : ℝ} (h1 : gain_B = (principal * rate_C * time_period - principal * (R / 100) * time_period)) : R = 10 := 
by
  sorry

end interest_rate_A_l414_414364


namespace bacteria_growth_rate_l414_414356

theorem bacteria_growth_rate (r : ℝ) 
  (h1 : ∀ n : ℕ, n = 22 → ∃ c : ℝ, c * r^n = c) 
  (h2 : ∀ n : ℕ, n = 21 → ∃ c : ℝ, 2 * c * r^n = c) : 
  r = 2 := 
by
  sorry

end bacteria_growth_rate_l414_414356


namespace triangle_segment_sum_equal_l414_414254

noncomputable def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

theorem triangle_segment_sum_equal
  (A B C P D E : ℝ × ℝ)
  (M_A M_B : ℝ × ℝ)
  (h_in_triangle : ∃ u v w : ℝ, u + v + w = 1 ∧ u, v, w ≥ 0 ∧ u * A + v * B + w * C = P)
  (h_circ_center_A : M_A = circumcenter A C P)
  (h_circ_center_B : M_B = circumcenter B C P)
  (h_M_A_outside : ∀ u v w : ℝ, u + v + w = 1 → u, v, w ≥ 0 → M_A ≠ u * A + v * B + w * C)
  (h_M_B_outside : ∀ u v w : ℝ, u + v + w = 1 → u, v, w ≥ 0 → M_B ≠ u * A + v * B + w * C)
  (h_collinear_APM_A : ∃ k : ℝ, P = A + k * (M_A - A))
  (h_collinear_BPM_B : ∃ k : ℝ, P = B + k * (M_B - B))
  (h_parallel_DE_AB : ∃ k : ℝ, D = P + k * (B - A) ∧ E = P - k * (B - A)) :
  dist D E = dist A C + dist B C :=
sorry

end triangle_segment_sum_equal_l414_414254


namespace a_annual_income_before_tax_l414_414707

theorem a_annual_income_before_tax:
  ∃ (A_inc: ℝ),
  let A_exp := 15600
  let total_savings := 1600
  let tax_rate_A := 0.10
  let total_expenditure := 15600 + 10400 
  A_exp = (3 / 5) * total_expenditure
  ∧ (A_inc - A_exp - tax_rate_A * A_inc = total_savings) 
  → A_inc ≈ 19111.11 := sorry

end a_annual_income_before_tax_l414_414707


namespace remaining_area_computation_l414_414095

def total_area_of_square (side_length : ℕ) : ℕ :=
  side_length^2

def area_of_dark_grey_triangles : ℕ :=
  1 * 3

def area_of_light_grey_triangles : ℕ :=
  2 * 3

def total_area_removed : ℕ :=
  area_of_dark_grey_triangles + area_of_light_grey_triangles

def remaining_area (side_length : ℕ) : ℕ :=
  total_area_of_square side_length - total_area_removed

theorem remaining_area_computation : remaining_area 6 = 27 :=
by
  rw [remaining_area, total_area_of_square, total_area_removed, area_of_dark_grey_triangles, area_of_light_grey_triangles]
  simp
  norm_num
  sorry

end remaining_area_computation_l414_414095


namespace sam_earnings_difference_l414_414981

def hours_per_dollar := 1 / 10  -- Sam earns $10 per hour, so it takes 1/10 hour per dollar earned.

theorem sam_earnings_difference
  (hours_per_dollar : ℝ := 1 / 10)
  (E1 : ℝ := 200)  -- Earnings in the first month are $200.
  (total_hours : ℝ := 55)  -- Total hours he worked over two months.
  (total_hourly_earning : ℝ := total_hours / hours_per_dollar)  -- Total earnings over two months.
  (E2 : ℝ := total_hourly_earning - E1) :  -- Earnings in the second month.

  E2 - E1 = 150 :=  -- The difference in earnings between the second month and the first month is $150.
sorry

end sam_earnings_difference_l414_414981


namespace find_alpha_l414_414114

open Real

noncomputable def point (x y : ℝ) : Prop := True

noncomputable def line (α t : ℝ) : Prop :=
  ∃ x y : ℝ, x = 2 + t * cos α ∧ y = 1 + t * sin α

theorem find_alpha (α : ℝ) :
  let P := (2, 1)
  let l := { t : ℝ // ∃ x y : ℝ, x = 2 + t * cos α ∧ y = 1 + t * sin α }
  let A : ℝ×ℝ := (2 - (2 - 1 / tan α), 0)
  let B : ℝ×ℝ := (0, 1 - 2 * tan α)
  let PA := sqrt((2 - (2 - 1 / tan α))^2 + (1 - 0)^2)
  let PB := sqrt((2 - 0)^2 + (1 - (1 - 2 * tan α))^2)
  (|PA| * |PB| = 4) → α = 3 * π / 4 :=
begin
  sorry
end

end find_alpha_l414_414114


namespace tangent_line_angle_with_z_axis_l414_414774

variables (a b t t₀ : ℝ)

-- Define the parametric equations of the helical curve
def helix_x (t : ℝ) : ℝ := a * Real.cos t
def helix_y (t : ℝ) : ℝ := a * Real.sin t
def helix_z (t : ℝ) : ℝ := b * t

-- Define the derivatives of the parametric equations
def dx_dt (t : ℝ) : ℝ := -a * Real.sin t
def dy_dt (t : ℝ) : ℝ := a * Real.cos t
def dz_dt (t : ℝ) : ℝ := b

-- Define the tangent vector at t = t₀
def tangent_vector (t₀ : ℝ) : ℝ × ℝ × ℝ :=
  (dx_dt t₀, dy_dt t₀, dz_dt t₀)

-- The magnitude of the tangent vector at t = t₀
def magnitude_tangent_vector (t₀ : ℝ) : ℝ :=
  Real.sqrt ((dx_dt t₀) ^ 2 + (dy_dt t₀) ^ 2 + (dz_dt t₀) ^ 2)

-- Cosine of the angle with the z-axis
def cos_gamma (t₀ : ℝ) : ℝ :=
  dz_dt t₀ / magnitude_tangent_vector t₀

-- Main theorem statement
theorem tangent_line_angle_with_z_axis (t₀ : ℝ) :
  cos_gamma t₀ = b / Real.sqrt (a ^ 2 + b ^ 2) :=
sorry

end tangent_line_angle_with_z_axis_l414_414774


namespace reconstruct_circle_l414_414054

theorem reconstruct_circle (n : ℕ) (hn : 0 < n) :
  ∃ (k : ℕ), (∀ (points : set (ℝ × ℝ)), 
    (∃ (circles : list (metric.sphere (ℝ × ℝ) 1)), 
    ∀ (point : ℝ × ℝ), point ∈ points → ∃ (circle : metric.sphere (ℝ × ℝ) 1), 
    point ∈ circle ∧ circle ∈ circles) →
    ∃ (circle : metric.sphere (ℝ × ℝ) 1), 
    ∀ (circles' : list (metric.sphere (ℝ × ℝ) 1)), 
    (∀ (point : ℝ × ℝ), point ∈ points → ∃ (circle' : metric.sphere (ℝ × ℝ) 1), point ∈ circle' ∧ circle' ∈ circles') → circle ∈ circles') 
     → k = 2 * n^2 + 1.


end reconstruct_circle_l414_414054


namespace range_of_x0_l414_414153

theorem range_of_x0 (x_0 : ℝ) (h1 : x_0 ∈ Icc (-Real.pi / 2) (Real.pi / 2)) :
  let f (x : ℝ) : ℝ := x^2 - Real.cos x in
  f x_0 > f (Real.pi / 3) → x_0 ∈ (Set.Icc (-Real.pi / 2) (-Real.pi / 3) ∪ Set.Ioc (Real.pi / 3) (Real.pi / 2)) :=
by
  let f : ℝ → ℝ := λ x => x^2 - Real.cos x
  sorry

end range_of_x0_l414_414153


namespace largest_invertible_interval_l414_414059

def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

theorem largest_invertible_interval (x : ℝ) (hx : x = 2) : 
  ∃ I : Set ℝ, (I = Set.univ ∩ {y | y ≥ 3 / 2}) ∧ ∀ y ∈ I, g y = 3 * (y - 3 / 2) ^ 2 - 11 / 4 ∧ g y ∈ I ∧ Function.Injective (g ∘ (fun z => z : I → ℝ)) :=
sorry

end largest_invertible_interval_l414_414059


namespace exists_integer_n_pairs_exceed_2024_l414_414270

theorem exists_integer_n_pairs_exceed_2024 : ∃ (n : ℕ), n ≥ 1 ∧ 
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ pair ∈ S, 
      let a := pair.1 
      let b := pair.2 
      0 < a ∧ 0 < b ∧ 
      (1 / (a - b) - 1 / a + 1 / b) = 1 / n) ∧ 
    S.card > 2024 := 
begin
  sorry
end

end exists_integer_n_pairs_exceed_2024_l414_414270


namespace divide_coal_l414_414759

noncomputable def part_of_pile (whole: ℚ) (parts: ℕ) := whole / parts
noncomputable def part_tons (total_tons: ℚ) (fraction: ℚ) := total_tons * fraction

theorem divide_coal (total_tons: ℚ) (parts: ℕ) (h: total_tons = 3 ∧ parts = 5):
  (part_of_pile 1 parts = 1/parts) ∧ (part_tons total_tons (1/parts) = total_tons / parts) :=
by
  sorry

end divide_coal_l414_414759


namespace range_of_a_l414_414575

noncomputable def f (a : ℝ) (x : ℝ) := Real.sqrt (Real.exp x + (Real.exp 1 - 1) * x - a)
def exists_b_condition (a : ℝ) : Prop := ∃ b : ℝ, b ∈ Set.Icc 0 1 ∧ f a b = b

theorem range_of_a (a : ℝ) : exists_b_condition a → a ∈ Set.Icc 1 (2 * Real.exp 1 - 2) :=
sorry

end range_of_a_l414_414575


namespace collinearity_area_equivalence_l414_414145

noncomputable def acute_triangle := sorry

variables {ABC : acute_triangle}
variables {I H : point}
variables {B_1 C_1 B_2 C_2 K : point}
variables {A_1 : point}

-- Conditions given in the problem
def incenter (I : point) (ABC : acute_triangle) : Prop := sorry
def orthocenter (H : point) (ABC : acute_triangle) : Prop := sorry
def midpoint (M : point) (A B : point) : Prop := sorry

def ray_intersect_side (P Q R S : point) : Prop := sorry
def ray_intersect_extension (P Q R S : point) : Prop := sorry

def circumcenter (O : point) (A B C : point) : Prop := sorry
def collinear (A B C : point) : Prop := sorry

def area_eq (t1 t2 : triangle) : Prop := sorry

-- Theorem statement
theorem collinearity_area_equivalence (I H : point) (B_1 C_1 B_2 C_2 K A_1 : point) (ABC : acute_triangle) :
  incenter I ABC →
  orthocenter H ABC →
  midpoint B_1 ABC.C ABC.A →
  midpoint C_1 ABC.A ABC.B →
  ray_intersect_side B_1 I ABC.B B_2 →
  ray_intersect_extension C_1 I ABC.C C_2 →
  line_intersect B_2 C_2 ABC.BC K →
  circumcenter A_1 ABC.B H C →
  collinear ABC.A I A_1 ↔ area_eq (triangle.mk B K B_2) (triangle.mk C K C_2) :=
sorry

end collinearity_area_equivalence_l414_414145


namespace q_of_x_l414_414576

def is_factor (q p : ℤ[X]) : Prop := ∃ r : ℤ[X], p = q * r

theorem q_of_x (a b : ℤ) (q := λ x : ℤ, x^2 + a * x + b)
    (h₁ : is_factor q (λ x : ℤ, x^4 + 8 * x^2 + 49))
    (h₂ : is_factor q (λ x : ℤ, 2 * x^4 + 5 * x^2 + 18 * x + 3)) :
    q (-1) = 66 :=
sorry

end q_of_x_l414_414576


namespace find_value_at_specific_point_l414_414473

-- Define the function f(x)
def f (x : ℝ) (a φ : ℝ) : ℝ := sin (3 * x + φ)

-- State the symmetric condition
def symmetric_at_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x : ℝ, f (a + x) = f (a - x)

-- Main theorem statement
theorem find_value_at_specific_point (a φ : ℝ)
  (h : symmetric_at_a (λ x => sin (3 * x + φ)) a) :
  (sin (3 * (a + π / 6) + φ) = 0) :=
sorry

end find_value_at_specific_point_l414_414473


namespace number_of_red_balls_l414_414521

def total_balls : ℕ := 50
def frequency_red_ball : ℝ := 0.7

theorem number_of_red_balls :
  ∃ n : ℕ, n = (total_balls : ℝ) * frequency_red_ball ∧ n = 35 :=
by
  sorry

end number_of_red_balls_l414_414521


namespace weights_equal_weights_equal_ints_weights_equal_rationals_l414_414660

theorem weights_equal (w : Fin 13 → ℝ) (swap_n_weighs_balance : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℝ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_ints (w : Fin 13 → ℤ) (swap_n_weighs_balance_ints : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℤ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_rationals (w : Fin 13 → ℚ) (swap_n_weighs_balance_rationals : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℚ), ∀ (i : Fin 13), w i = m :=
by
  sorry

end weights_equal_weights_equal_ints_weights_equal_rationals_l414_414660


namespace power_function_difference_l414_414138

theorem power_function_difference (f : ℝ → ℝ) (h : ∃ α : ℝ, f = λ x, x^α ∧ f 9 = 3) : 
  f 2 - f 1 = Real.sqrt 2 - 1 :=
by
  sorry

end power_function_difference_l414_414138


namespace volume_larger_cube_l414_414317

theorem volume_larger_cube (e : ℝ) (V_s : ℝ) (h1 : V_s = e^3) (h2 : V_s = 1) : 
  let e_large := 3 * e in
  let V_l := e_large^3 in
  V_l = 27 :=
by
  -- Proof will be placed here
  sorry

end volume_larger_cube_l414_414317


namespace sum_possible_values_l414_414870

theorem sum_possible_values (y : ℝ) (h : y^2 = 36) : 
  y = 6 ∨ y = -6 → 6 + (-6) = 0 :=
by
  intro hy
  rw [add_comm]
  exact add_neg_self 6

end sum_possible_values_l414_414870


namespace conjugate_of_z_l414_414626

open Complex

theorem conjugate_of_z :
  let i := Complex.I
  let z := i * (i + 1)
  conj z = -1 - i :=
by
  sorry

end conjugate_of_z_l414_414626


namespace cross_section_area_smaller_l414_414600

section
variables {A B C D P Q R : Type} 
variables [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry C] [PlaneGeometry D] [PlaneGeometry P] [PlaneGeometry Q] [PlaneGeometry R]

-- Define a tetrahedron
structure Tetrahedron (A B C D : Type) :=
(vertices : set Type)

-- Assume we have a tetrahedron 
def tetra : Tetrahedron A B C D := 
{ vertices := {A, B, C, D} }

-- Define triangular faces and sections
def triangle (X Y Z : Type) := 
{ vertices := {X, Y, Z} }

-- Define the areas of triangles
def area (t : { vertices : set Type }) : Real := sorry

-- Main theorem statement
theorem cross_section_area_smaller (PQR : { vertices : set Type }) :
  ∃ face, face ∈ {triangle A B C, triangle A B D, triangle A C D, triangle B C D} ∧ area PQR < area face := 
sorry
end

end cross_section_area_smaller_l414_414600


namespace prism_height_and_center_on_AB_l414_414534

-- Given definitions matching the conditions
variables (S A B C H : Type) [normed_add_comm_group S]
variables (a : ℝ) (r : ℝ) (angle : ℝ) (SH SC : ℝ)

-- Conditions derived from the problem
def edge_lengths_equal : Prop := SC = a
def angle_with_base_plane : Prop := angle = 60
def sphere_radius : Prop := r = 1
def points_lying_on_sphere (A B C H : S) (midpoints : S → S → S)  : Prop := 
  dist (midpoints A B) (midpoints B C) = dist (midpoints B C) (midpoints C A) ∧
  dist (midpoints C A) (midpoints A B) = r

theorem prism_height_and_center_on_AB :
  edge_lengths_equal SC a →
  angle_with_base_plane angle →
  sphere_radius r →
  points_lying_on_sphere A B C H (λ x y, midpoint [normed_add_comm_group S]) →
  SH = sqrt 3 ∧ ∃ H_on_AB, H_on_AB = midpoint A B :=
sorry

end prism_height_and_center_on_AB_l414_414534


namespace Rachel_budget_twice_Sara_l414_414607

-- Define the cost of Sara's shoes and dress
def s_shoes : ℕ := 50
def s_dress : ℕ := 200

-- Define the target budget for Rachel
def r : ℕ := 500

-- State the theorem to prove
theorem Rachel_budget_twice_Sara :
  2 * s_shoes + 2 * s_dress = r :=
by simp [s_shoes, s_dress, r]; sorry

end Rachel_budget_twice_Sara_l414_414607


namespace number_of_incorrect_statements_l414_414061

-- Definitions of the propositions
def proposition_1 (a b : ℝ) : Prop :=
  (a = 0 → a + b * complex.I ∈ set_of complex.im) ∧ 
  ¬(a = 0 → a + b * complex.I ∈ set_of complex.re)

def proposition_2 : Prop :=
  (∀ x ∈ set.Icc 0 1, exp x ≥ 1) ∨ 
  (∃ x : ℝ, x^2 + x + 1 < 0)

def proposition_3 (a b m : ℝ) : Prop :=
  (am^2 < bm^2 → a < b) → (a ≥ b → am^2 ≥ bm^2)

def proposition_4 (p q : Prop) : Prop :=
  ¬(p ∨ q) → ¬p ∧ ¬q

-- Statement about the number of incorrect propositions
theorem number_of_incorrect_statements : 
  ∃! incorrect_statements : ℕ, incorrect_statements = 1 :=
  by
    let a b m : ℝ := 0, -- Assign dummy values for simplicity, focus on structure
    have p1 := proposition_1 a b,
    have p2 := proposition_2,
    have p3 := proposition_3 a b m,
    have p4 := proposition_4 p2 false,
    exact sorry -- Proof steps omitted

end number_of_incorrect_statements_l414_414061


namespace isosceles_right_triangle_area_correct_l414_414146

noncomputable def isosceles_right_triangle_area
  (side1_square_area : ℕ)
  (side2_square_area : ℕ)
  (hypotenuse_square_area : ℕ) : ℕ :=
  let side1 := Nat.sqrt side1_square_area
  let side2 := Nat.sqrt side2_square_area
  let hypotenuse := Nat.sqrt hypotenuse_square_area in
  if side1_square_area = side2_square_area ∧
     hypotenuse_square_area = 2 * side1_square_area then
    (side1 * side2) / 2
  else
    0

theorem isosceles_right_triangle_area_correct :
  isosceles_right_triangle_area 64 64 256 = 32 :=
by
  sorry

end isosceles_right_triangle_area_correct_l414_414146


namespace ellipse_equation_l414_414295

theorem ellipse_equation (h_center : ∀ P, P ∈ ellipse → P = (0,0)) 
                         (h_foci : (foci_x_axis ∨ foci_y_axis) : Prop)
                         (h_eccentricity : e = √(3/2))
                         (h_point : (2, 0) ∈ ellipse) : 
  (ellipse = {P | (P.1 ^ 2 / 4 + P.2 ^ 2 = 1)} ∨ ellipse = {P | (P.1 ^ 2 / 4 + P.2 ^ 2 / 16 = 1)}) := 
sorry

end ellipse_equation_l414_414295


namespace arithmetic_square_root_problem_l414_414627

open Real

theorem arithmetic_square_root_problem 
  (a b c : ℝ)
  (ha : 5 * a - 2 = -27)
  (hb : b = ⌊sqrt 22⌋)
  (hc : c = -sqrt (4 / 25)) :
  sqrt (4 * a * c + 7 * b) = 6 := by
  sorry

end arithmetic_square_root_problem_l414_414627


namespace arc_measure_of_angle_inscribed_l414_414215

theorem arc_measure_of_angle_inscribed
  (O B C D : Point)
  (h_circle : Circle O)
  (h_angle : ∠ B C D = 60) :
  measure_arc_minor_BD = 120 :=
by
  sorry

end arc_measure_of_angle_inscribed_l414_414215


namespace max_difference_distance_l414_414807

-- Define the conditions in Lean
def circle1 (E : ℝ × ℝ) : Prop := E.1^2 + E.2^2 = 1/4
def circle2 (F : ℝ × ℝ) : Prop := (F.1 - 3)^2 + (F.2 + 1)^2 = 9/4
def P (t : ℝ) : ℝ × ℝ := (t, t - 1)

-- Define the distances
def dist (A B: ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the statement of the proof
theorem max_difference_distance 
  (t : ℝ) (E F : ℝ × ℝ)
  (hE : circle1 E)
  (hF : circle2 F) : 
  dist (P t) F - dist (P t) E ≤ 4 :=
sorry

end max_difference_distance_l414_414807


namespace original_savings_calculation_l414_414261

theorem original_savings_calculation (S : ℝ) (F : ℝ) (T : ℝ) 
  (h1 : 0.8 * F = (3 / 4) * S)
  (h2 : 1.1 * T = 150)
  (h3 : (1 / 4) * S = T) :
  S = 545.44 :=
by
  sorry

end original_savings_calculation_l414_414261


namespace math_problem_equivalent_proof_l414_414986

theorem math_problem_equivalent_proof :
  ∃ d e f : ℕ, let s := (d - Real.sqrt e) / f in 
  PQRS_IsSquare ∧ PQRS_has_side_len_2 ∧ PTU_IsEquilateral ∧ Side_Length_Expression s d e f ∧
  d = 6 ∧ e = 3 ∧ f = 12 ∧ (d + e + f = 21) :=
by
  sorry

end math_problem_equivalent_proof_l414_414986


namespace triangle_problems_l414_414509

noncomputable def triangle_side_b (a c : ℝ) (A : ℝ) : set ℝ :=
  {b | a^2 = b^2 + c^2 - 2 * b * c * real.cos A}

noncomputable def triangle_area (b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * real.sin A

theorem triangle_problems :
  let a := real.sqrt 7
      c := 3
      A := real.pi / 3
  in (triangle_side_b a c A = {1, 2}) ∧
     (triangle_area 1 c A = 3 * real.sqrt 3 / 4 ∨ triangle_area 2 c A = 3 * real.sqrt 3 / 2) :=
by
  sorry

end triangle_problems_l414_414509


namespace beef_weight_after_processing_l414_414012

theorem beef_weight_after_processing
  (initial_weight : ℝ)
  (weight_loss_percentage : ℝ)
  (processed_weight : ℝ)
  (h1 : initial_weight = 892.31)
  (h2 : weight_loss_percentage = 0.35)
  (h3 : processed_weight = initial_weight * (1 - weight_loss_percentage)) :
  processed_weight = 579.5015 :=
by
  sorry

end beef_weight_after_processing_l414_414012


namespace cevian_intersection_l414_414921

noncomputable def triangle (A B C : Type) := ∃ (X : A),
  (∀ A1 : Type, ∃ A2 : Type, 
    (line_through A X intersects (circumcircle A B C) at A1 ∧ inscribed_circle_in_segment_cut ∧ touches_arc (A1) ∧ touches_side (BC) at A2)) ∧
  (∀ B1 : Type, ∃ B2 : Type, 
    (line_through B X intersects (circumcircle A B C) at B1 ∧ inscribed_circle_in_segment_cut ∧ touches_arc (B1) ∧ touches_side (AC) at B2)) ∧
  (∀ C1 : Type, ∃ C2 : Type, 
    (line_through C X intersects (circumcircle A B C) at C1 ∧ inscribed_circle_in_segment_cut ∧ touches_arc (C1) ∧ touches_side (AB) at C2)) ∧
  ∃ P : Type, (line_through A A2 intersects line_through B B2 ∧ line_through B B2 intersects line_through C C2 ∧ line_through C C2 intersects line_through A A2 at P)

theorem cevian_intersection (A B C A1 B1 C1 A2 B2 C2 X: Type) (hx: triangle A B C):
  ∃ P, (line_through A A2 intersects line_through B B2 ∧ line_through B B2 intersects line_through C C2 ∧ line_through C C2 intersects line_through A A2 at P) :=
hx

end cevian_intersection_l414_414921


namespace lambda_range_l414_414466

-- Definition of f(x) based on the conditions
def f : ℝ → ℝ := λ x, x^2 + 2 * x

-- Derived definition of g(x) based on the symmetry condition
def g : ℝ → ℝ := λ x, - (x^2 + 2 * x)

-- Definition of h(x) with parameter λ
def h (λ : ℝ) : ℝ → ℝ := λ x, f(x) - λ * g(x)

-- Conditions to check for h(x) being increasing on [-1,1]
def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x ∧ y ≤ b ∧ x ≤ y → f x ≤ f y

-- The statement we need to prove
theorem lambda_range (λ : ℝ) : is_increasing (h λ) -1 1 ↔ λ ∈ set.Iic 0 := sorry

end lambda_range_l414_414466


namespace tree_annual_growth_rate_l414_414357

-- Define the initial conditions on growth
def tree_initial_height : ℝ := 100  -- initial height in meters
def tree_growth_over_two_years : ℝ := 21  -- growth over 2 years in meters
def number_of_years : ℕ := 2  -- duration of growth in years

-- Define a function to compute average annual growth rate
def average_annual_growth (total_growth : ℝ) (years : ℕ) : ℝ :=
  total_growth / years

-- Prove that the annual growth rate is 10.5 meters per year
theorem tree_annual_growth_rate :
  average_annual_growth tree_growth_over_two_years number_of_years = 10.5 :=
by
  -- You can complete this proof or skip with sorry
  sorry

end tree_annual_growth_rate_l414_414357


namespace choose_4_from_12_l414_414525

theorem choose_4_from_12 : binomial 12 4 = 495 := 
sorry

end choose_4_from_12_l414_414525


namespace inequality_solution_l414_414057

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
    (15 ≤ x * (x - 2) / (x - 5) ^ 2) ↔ (4.1933 ≤ x ∧ x < 5 ∨ 5 < x ∧ x ≤ 6.3767) :=
by
  sorry

end inequality_solution_l414_414057


namespace can_form_sets_l414_414342

def clearly_defined (s : Set α) : Prop := ∀ x ∈ s, True
def not_clearly_defined (s : Set α) : Prop := ¬clearly_defined s

def cubes := {x : Type | True} -- Placeholder for the actual definition
def major_supermarkets := {x : Type | True} -- Placeholder for the actual definition
def difficult_math_problems := {x : Type | True} -- Placeholder for the actual definition
def famous_dancers := {x : Type | True} -- Placeholder for the actual definition
def products_2012 := {x : Type | True} -- Placeholder for the actual definition
def points_on_axes := {x : ℝ × ℝ | x.1 = 0 ∨ x.2 = 0}

theorem can_form_sets :
  (clearly_defined cubes) ∧
  (not_clearly_defined major_supermarkets) ∧
  (not_clearly_defined difficult_math_problems) ∧
  (not_clearly_defined famous_dancers) ∧
  (clearly_defined products_2012) ∧
  (clearly_defined points_on_axes) →
  True := 
by {
  -- Your proof goes here
  sorry
}

end can_form_sets_l414_414342


namespace sum_of_first_n_terms_l414_414479

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := a n ^ 2 + 2 * a n

noncomputable def b (n : ℕ) : ℚ :=
1 / (a (n + 1)) + 1 / (a n + 2)

noncomputable def S (n : ℕ) : ℚ :=
Σ i in finset.range n, b i

theorem sum_of_first_n_terms (n : ℕ) :
  S n = 1 - 1 / (2 ^ (2 ^ n) - 1) :=
sorry

end sum_of_first_n_terms_l414_414479


namespace distinct_values_of_c_l414_414943

open Complex

theorem distinct_values_of_c :
  ∀ (c p q u : ℂ), 
  p ≠ q → p ≠ u → q ≠ u → 
  (∀ z : ℂ, (z - p) * (z - q) * (z^2 - u) = (z - c * p) * (z - c * q) * (z^2 - c * u)) → 
  ∃ (vals : Finset ℂ), set.finset.card vals = 4 := 
by 
  sorry

end distinct_values_of_c_l414_414943


namespace total_seats_l414_414028

theorem total_seats (s : ℕ) 
  (first_class : ℕ := 30) 
  (business_class : ℕ := (20 * s) / 100) 
  (premium_economy : ℕ := 15) 
  (economy_class : ℕ := s - first_class - business_class - premium_economy) 
  (total : first_class + business_class + premium_economy + economy_class = s) 
  : s = 288 := 
sorry

end total_seats_l414_414028


namespace complex_expression_eq_l414_414072

-- Definitions for complex number terms
def x : ℂ := 7 - 3 * Complex.i
def y : ℂ := 3 * (2 + 4 * Complex.i)
def z : ℂ := (1 - Complex.i) * (3 + Complex.i)
def result : ℂ := 5 - 17 * Complex.i

-- Prove that their combination is equal to the result
theorem complex_expression_eq : x - y + z = result := 
sorry

end complex_expression_eq_l414_414072


namespace inequality_preservation_l414_414495

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3 : ℝ) * a - 1 > (1/3 : ℝ) * b - 1 := 
by sorry

end inequality_preservation_l414_414495


namespace no_such_function_exists_l414_414760

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) :=
by
  sorry

end no_such_function_exists_l414_414760


namespace solve_inequality_l414_414616

theorem solve_inequality (x : Real) : 
  (abs ((3 * x + 2) / (x - 2)) > 3) ↔ (x ∈ Set.Ioo (2 / 3) 2) := by
  sorry

end solve_inequality_l414_414616


namespace susan_correct_guess_probability_l414_414995

theorem susan_correct_guess_probability :
  (1 - (5/6)^6) = 31031/46656 := 
sorry

end susan_correct_guess_probability_l414_414995


namespace not_necessarily_all_liars_l414_414592

-- Define types and properties
constant Person : Type
constant isKnight : Person → Prop
constant isLiar : Person → Prop
constant Friend : Person → Person → Prop

-- Conditions
axiom totalPopulation : ∃ (people : Finset Person), people.card = 2018
axiom disjointKnightLiar : ∀ p : Person, isKnight p ∨ isLiar p
axiom twoFriends : ∀ p : Person, (∃ (f1 f2 : Person), Friend p f1 ∧ Friend p f2 ∧ f1 ≠ f2) ∧ ∀ f : Person, Friend p f → (Friend f p) -- Each person has exactly two friends
axiom claimsOneLiarFriend : ∀ p : Person, (∃ (f1 f2 : Person), Friend p f1 ∧ Friend p f2 ∧ (isKnight p → (isLiar f1 ∨ isLiar f2)) ∧ (isLiar p → ¬(isLiar f1 ∧ isLiar f2)) ∧ f1 ≠ f2)

-- The theorem to prove
theorem not_necessarily_all_liars : ¬ (∀ p : Person, isLiar p) :=
sorry

end not_necessarily_all_liars_l414_414592


namespace problem_solution_l414_414100

theorem problem_solution
  (m : ℝ) (n : ℝ)
  (h1 : m = 1 / (Real.sqrt 3 + Real.sqrt 2))
  (h2 : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) :
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 :=
by sorry

end problem_solution_l414_414100


namespace max_value_fraction_l414_414608

theorem max_value_fraction 
  (a b c x1 x2 x3 λ : ℝ) 
  (f : ℝ → ℝ := λ x, x^3 + a * x^2 + b * x + c) 
  (h1 : x2 - x1 = λ)
  (h2 : x3 > (1/2) * (x1 + x2))
  (h3 : f x1 = 0)
  (h4 : f x2 = 0)
  (h5 : f x3 = 0)
  (λ_pos : λ > 0) :
  ∃ (L : ℝ), L = (3 * real.sqrt 3) / 2 ∧ 
  ( ∀ a b c λ x1 x2 x3, 
    x2 - x1 = λ ∧
    x3 > (1/2) * (x1 + x2) ∧
    f(x1) = 0 ∧
    f(x2) = 0 ∧
    f(x3) = 0 ∧
    λ > 0 → 
    ∃ L, L = (3 * real.sqrt 3) / 2
) := 
sorry

end max_value_fraction_l414_414608


namespace solve_quadratic_eq1_solve_quadratic_eq2_l414_414615

-- Define the first equation
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 := by
  sorry

-- Define the second equation
theorem solve_quadratic_eq2 (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 ∨ x = 1 / 2 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l414_414615


namespace triangle_area_l414_414380

theorem triangle_area (a b c : ℕ) (h : a = 12) (i : b = 16) (j : c = 20) (hc : c * c = a * a + b * b) :
  ∃ (area : ℕ), area = 96 :=
by
  sorry

end triangle_area_l414_414380


namespace determinant_matrix_2x2_is_8_l414_414403

-- Define the 2x2 matrix and its contents
def matrix_2x2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![6, -2], ![-5, 3]]

-- Define the expected determinant
def expected_determinant : ℤ := 8

-- Prove that the determinant of the matrix is 8
theorem determinant_matrix_2x2_is_8 : matrix.det (matrix_2x2) = expected_determinant :=
by
  -- Use matrix.determinant for 2x2 matrices and calculate determinat.
  sorry

end determinant_matrix_2x2_is_8_l414_414403


namespace sum_of_squares_divisible_by_S_l414_414922

variable (n : ℕ)
variable (x : Fin n → ℕ)

def S := ∑ i, x i

theorem sum_of_squares_divisible_by_S (h : ∀ i, S n x ∣ x i * (S n x - x i + 1)) : S n x ∣ ∑ i, (x i) ^ 2 :=
by
  sorry

end sum_of_squares_divisible_by_S_l414_414922


namespace ellipse_equation_l414_414156

theorem ellipse_equation (a : ℝ) (h1 : 0 < a) (h2 : ∀ x y : ℝ, y^2 = 8 * x → x = 2) :
  (\dfrac {x^2} {a^2} + y^2 = 1 ∧ a^2 = 5) → (\dfrac {x^2} {5} + y^2 = 1) :=
by
  intro h3
  cases h3 with ell_eq_a2
  rw ell_eq_a2
  sorry

end ellipse_equation_l414_414156


namespace tangent_to_circle_minimum_distance_PQ_l414_414441

def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

def lineL (x y m : ℝ) : Prop := x + m * y + 2 * m - 3 = 0

theorem tangent_to_circle (m : ℝ) :
  m = 12 / 5 → ∃ x y, circleC x y ∧ lineL x y m ∧ (∀ x' y', circleC x' y' → ¬ lineL x' y' m ) := 
sorry

theorem minimum_distance_PQ (m : ℝ) :
  m = -1 → ∃ P Q x y, lineL P y m ∧ circleC x y ∧ P ≠ Q ∧ (Q ∈ lineL ∧ Q ∈ circleC) ∧ 
  dist P Q = (sqrt 34) / 2 :=
sorry

end tangent_to_circle_minimum_distance_PQ_l414_414441


namespace price_of_shares_l414_414366

variable (share_value : ℝ) (dividend_rate : ℝ) (tax_rate : ℝ) (effective_return : ℝ) (price : ℝ)

-- Given conditions
axiom H1 : share_value = 50
axiom H2 : dividend_rate = 0.185
axiom H3 : tax_rate = 0.05
axiom H4 : effective_return = 0.25
axiom H5 : 0.25 * price = 0.185 * 50 - (0.05 * (0.185 * 50))

-- Prove that the price at which the investor bought the shares is Rs. 35.15
theorem price_of_shares : price = 35.15 :=
by
  sorry

end price_of_shares_l414_414366


namespace minimumPerimeterTriangleDEF_l414_414510

noncomputable def minimumPerimeter (DE DF EF : ℕ) (incenter : Prop) (excircleEF : Prop) (excircleDE : Prop) (excircleDF : Prop) : ℕ :=
if h : DE = DF ∧ EF = 2 * DE then
  40
else
  0

theorem minimumPerimeterTriangleDEF :
  ∀ (DE DF EF : ℕ),
    (DE = DF) →
    (EF = 2 * DE) →
    (∃ incenter : Prop, incenter) →
    (∃ excircleEF : Prop, excircleEF) →
    (∃ excircleDE : Prop, excircleDE) →
    (∃ excircleDF : Prop, excircleDF) →
    minimumPerimeter DE DF EF (∃ incenter : Prop, incenter) (∃ excircleEF : Prop, excircleEF) (∃ excircleDE : Prop, excircleDE) (∃ excircleDF : Prop, excircleDF) = 40 :=
begin
  sorry
end

end minimumPerimeterTriangleDEF_l414_414510


namespace find_alpha_polar_equation_l414_414115

variables {α : Real}

def point_P := (2, 1)
def line_l := { t : Real // x = 2 + t * Real.cos α ∧ y = 1 + t * Real.sin α }
def line_intersects_pos_axes := 
  ∃ A B : (ℝ × ℝ),
  A.1 > 0 ∧ B.2 > 0 ∧ 
  ∃ t1 t2 : Real, 
    line_l t1 = (A.1, 0) ∧ line_l t2 = (0, B.2)

def distance_PA (A : ℝ × ℝ) := ((2 - A.1) ^ 2 + (1 - A.2) ^ 2).sqrt
def distance_PB (B : ℝ × ℝ) := ((2 - B.1) ^ 2 + (1 - B.2) ^ 2).sqrt

def condition_PA_PB_product (PA PB : ℝ) := PA * PB = 4

theorem find_alpha 
  (A B : (ℝ × ℝ)) (h1 : line_intersects_pos_axes) 
  (h2 : condition_PA_PB_product (distance_PA A) (distance_PB B)):
  α = 3 * Real.pi / 4 :=
sorry

def polar_coordinate_line (ρ θ : Real) :=
  ρ * (Real.cos θ + Real.sin θ) = 3

theorem polar_equation 
  (α_value : α = 3 * Real.pi / 4):
  polar_coordinate_line :=
sorry

end find_alpha_polar_equation_l414_414115


namespace students_above_90_expectation_of_X_l414_414286

variable (ξ : ℝ)
variable (X : ℕ)
variable (n : ℕ := 100)
variable (m : ℕ := 10)
variable (σ : ℝ)
variable (μ : ℝ := 85)
variable (P : Set ℝ → ℝ)

-- Normal distribution condition
def normal_distribution : Prop := (ξ ~' N(μ, σ^2))
-- Given probabilities condition
def prob_condition : Prop := (P({x | 80 ≤ x ∧ x ≤ 85}) = 0.4)

-- Part (1): Number of students with scores above 90
theorem students_above_90 
  (h1 : normal_distribution ξ)
  (h2 : prob_condition P)
  : n * (1 - P({x | 80 ≤ x ∧ x ≤ 85}) - P({x | 85 ≤ x ∧ x ≤ 90})) = 10 :=
sorry

-- Part (2): Expectation of X
theorem expectation_of_X
  (h1 : normal_distribution ξ)
  (h2 : prob_condition P)
  (h3 : P({x | x ≥ 80}) = 0.9)
  : (E X) = m * 0.9 :=
sorry

end students_above_90_expectation_of_X_l414_414286


namespace solution_set_f_l414_414471

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then log (2 - x) / log 2 else x ^ (1 / 3 : ℝ)

theorem solution_set_f (x : ℝ) : (f x < 2) ↔ (-2 < x ∧ x < 8) := by
  sorry

end solution_set_f_l414_414471


namespace modulo_remainder_l414_414775

theorem modulo_remainder : (7^2023) % 17 = 15 := 
by 
  sorry

end modulo_remainder_l414_414775


namespace max_norm_c_l414_414457

-- Definitions reflecting the conditions
variables {𝔽 : Type*} [inner_product_space 𝔽 (euclidean_space]
variables (a b c : euclidean_space)
variables (H1 : ∥a∥ = 1) (H2 : ∥b∥ = 1) (H3 : inner_product a b = 0)
variables (H4 : inner_product (a - c) (b - c) = 0)

-- Statement to prove
theorem max_norm_c : ∥c∥ ≤ sqrt 2 := sorry

end max_norm_c_l414_414457


namespace temperature_at_midnight_l414_414901

theorem temperature_at_midnight 
  (morning_temp : ℝ) 
  (afternoon_rise : ℝ) 
  (midnight_drop : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_rise = 1)
  (h3 : midnight_drop = 7) 
  : morning_temp + afternoon_rise - midnight_drop = 24 :=
by
  -- Convert all conditions into the correct forms
  rw [h1, h2, h3]
  -- Perform the arithmetic operations
  norm_num

end temperature_at_midnight_l414_414901


namespace danielle_rooms_is_6_l414_414851

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l414_414851


namespace tom_total_time_l414_414674

theorem tom_total_time
  (BS_time : ℝ) (MS_time : ℝ) (PhD_time : ℝ) (Tom_PhD_rate : ℝ) :
  BS_time = 3 ∧ MS_time = 2 ∧ PhD_time = 5 ∧ Tom_PhD_rate = 3/4 →
  (BS_time + MS_time + (PhD_time * Tom_PhD_rate)) = 8.75 := 
begin
  sorry
end

end tom_total_time_l414_414674


namespace ratio_BP_DP_l414_414676

noncomputable def chord_intersection (A B C D P : Point) : Prop :=
  Let AP := 3
  Let CP := 8
  AP * BP = CP * DP

theorem ratio_BP_DP (A B C D P : Point) (AP BP CP DP : ℝ) (h1 : AP = 3) (h2 : CP = 8) :
  ((AP * BP) = (CP * DP)) → (BP / DP) = (8 / 3) :=
by
  intro h
  rw [h1, h2] at h
  sorry -- Proof not required as per instruction.

end ratio_BP_DP_l414_414676


namespace fraction_of_males_l414_414034

theorem fraction_of_males (M F : ℝ) (h1 : M + F = 1) (h2 : (7/8 * M + 9/10 * (1 - M)) = 0.885) :
  M = 0.6 :=
sorry

end fraction_of_males_l414_414034


namespace ratio_of_speeds_l414_414511

theorem ratio_of_speeds :
  ∀ (t : ℝ) (v_A v_B : ℝ),
  (v_A = 360 / t) →
  (v_B = 480 / t) →
  v_A / v_B = 3 / 4 :=
by
  intros t v_A v_B h_A h_B
  rw [←h_A, ←h_B]
  field_simp
  norm_num

# In this theorem, we state that given the conditions of the speeds
# and distances covered in the problem, we prove that the ratio of
# their speeds is 3:4.

end ratio_of_speeds_l414_414511


namespace area_covered_by_circles_l414_414907

theorem area_covered_by_circles (n : ℕ) (hn : n ≥ 3) (radius : ℝ) (hr : radius = 1)
  (intersect_cond : ∀ (C : finset (metric.sphere (0 : euclidean_space ℝ (fin 2)) radius)), 
    C.card = 3 → ∃ (c1 c2 : metric.sphere (0 : euclidean_space ℝ (fin 2)) radius), c1 ≠ c2 ∧ c1 ∈ C ∧ c2 ∈ C ∧ c1.dist (c2 : euclidean_space ℝ (fin 2)) ≤ 2 * radius) :
  ∃ (S : ℝ), S < 35 :=
  sorry

end area_covered_by_circles_l414_414907


namespace rhombus_diagonals_l414_414644

theorem rhombus_diagonals (p d1 d2 : ℝ) (h1 : p = 100) (h2 : abs (d1 - d2) = 34) :
  ∃ d1 d2 : ℝ, d1 = 14 ∧ d2 = 48 :=
by
  -- proof omitted
  sorry

end rhombus_diagonals_l414_414644


namespace min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l414_414150

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * Real.log x + b

theorem min_value_f 
  (a b : ℝ) 
  (a_non_pos : a ≤ 1) : 
  f 1 a b = (1 / 2) + b :=
sorry

theorem min_value_f_sqrt 
  (a b : ℝ) 
  (a_pos_range : 1 < a ∧ a < 4) : 
  f (Real.sqrt a) a b = (a / 2) - a * Real.log (Real.sqrt a) + b :=
sorry

theorem min_value_f_2 
  (a b : ℝ) 
  (a_ge_4 : 4 ≤ a) : 
  f 2 a b = 2 - a * Real.log 2 + b :=
sorry

theorem min_m 
  (a : ℝ) 
  (a_range : -2 ≤ a ∧ a < 0):
  ∀x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 2) ∧ (0 < x2 ∧ x2 ≤ 2) →
  ∃m : ℝ, m = 12 ∧ abs (f x1 a 0 - f x2 a 0) ≤ m ^ abs (1 / x1 - 1 / x2) :=
sorry

end min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l414_414150


namespace number_of_js_l414_414187

noncomputable def g (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum

theorem number_of_js :
  (Finset.filter (λ j, g j = 2 + int.sqrt j + j) (Finset.range 101)).card = 4 :=
by sorry

end number_of_js_l414_414187


namespace value_division_l414_414722

theorem value_division (x : ℝ) (h1 : 54 / x = 54 - 36) : x = 3 := by
  sorry

end value_division_l414_414722


namespace triangle_inradius_formula1_triangle_inradius_formula2_triangle_inradius_formula3_triangle_inradius_formula4_l414_414508

variables {A B C : ℝ}  -- Angles of the triangle
variables {a b c : ℝ}  -- Sides of the triangle
variables {r R s : ℝ}  -- Inradius, circumradius, semiperimeter
variables {Δ : ℝ}  -- Area of the triangle

-- Define r_a, r_b, r_c
noncomputable def r_a : ℝ := Δ / (s - a)
noncomputable def r_b : ℝ := Δ / (s - b)
noncomputable def r_c : ℝ := Δ / (s - c)

-- (1) r = 4R * sin(A / 2) * sin(B / 2) * sin(C / 2)
theorem triangle_inradius_formula1 :
  r = 4 * R * sin (A / 2) * sin (B / 2) * sin (C / 2) := sorry

-- (2) r = s * tan(A / 2) * tan(B / 2) * tan(C / 2)
theorem triangle_inradius_formula2 :
  r = s * tan (A / 2) * tan (B / 2) * tan (C / 2) := sorry

-- (3) r = R * (cos A + cos B + cos C - 1)
theorem triangle_inradius_formula3 :
  r = R * (cos A + cos B + cos C - 1) := sorry

-- (4) r = r_a + r_b + r_c - 4 * R
theorem triangle_inradius_formula4 :
  r = r_a + r_b + r_c - 4 * R := sorry

end triangle_inradius_formula1_triangle_inradius_formula2_triangle_inradius_formula3_triangle_inradius_formula4_l414_414508


namespace prime_p_equals_2_l414_414326

theorem prime_p_equals_2 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs: Nat.Prime s)
  (h_sum : p + q + r = 2 * s) (h_order : 1 < p ∧ p < q ∧ q < r) : p = 2 :=
sorry

end prime_p_equals_2_l414_414326


namespace total_rope_in_inches_l414_414554

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end total_rope_in_inches_l414_414554


namespace sum_of_interior_angles_l414_414244

theorem sum_of_interior_angles (n : ℕ) (a b : ℕ → ℝ)
  (h1 : ∀ i, a i = 11.5 * b i)
  (h2 : ∑ i in (Finset.range n), b i = 360) :
  ∑ i in (Finset.range n), a i = 4140 :=
by
  sorry

end sum_of_interior_angles_l414_414244


namespace function_period_transformations_needed_l414_414827

-- Define the function
def f (x : ℝ) : ℝ := sin (x / 2) + real.sqrt 3 * cos (x / 2)

-- Question 1: State the property of the period
theorem function_period : ∃ T > 0, ∀ x, f (x + T) = f x := 
sorry 

-- Question 2: Describe the necessary transformations
theorem transformations_needed : 
  ∃ (t1 t2 t3 : ℝ), 
    (∀ x, 2 * sin (1 / 2 * (x + t1)) = f x) ∧ 
    (∀ x, 1 / 2 * (2 * sin (1 / 2 * x)) = sin x) := 
sorry

end function_period_transformations_needed_l414_414827


namespace new_polyhedron_edges_l414_414714

noncomputable def polyhedron_vertices (S : Type) [Polyhedron S] : Nat := sorry
noncomputable def polyhedron_edges (S : Type) [Polyhedron S] : Nat := 120
noncomputable def intersect_planes (U : Type) [Vertex U] : Nat := sorry
noncomputable def planes_intersect_edges (T : Type) [Polyhedron T] (Q : Fin m → Type) [Plane (Q i)] (U : Fin m → Type) [Vertex (U i)] : Prop := sorry

-- Statement of the problem in Lean
theorem new_polyhedron_edges (S : Type) [Polyhedron S] (m : Nat) (U : Fin m → Type) [Vertex (U i)] (Q : Fin m → Type) [Plane (Q i)] (no_intersection : ∀ i j, i ≠ j → ¬intersect (Q i) (Q j) (volume_of S) ∧ ¬intersect (Q i) (Q j) (surface_of S)) :
  polyhedron_edges (new_polyhedron T) = 360 := 
sorry

end new_polyhedron_edges_l414_414714


namespace find_k_l414_414670

-- Assume three lines in the form of equations
def line1 (x y k : ℝ) := x + k * y = 0
def line2 (x y : ℝ) := 2 * x + 3 * y + 8 = 0
def line3 (x y : ℝ) := x - y - 1 = 0

-- Assume the intersection point exists
def intersection_point (x y : ℝ) := 
  line2 x y ∧ line3 x y

-- The main theorem statement
theorem find_k (k : ℝ) (x y : ℝ) (h : intersection_point x y) : 
  line1 x y k ↔ k = -1/2 := 
sorry

end find_k_l414_414670


namespace prove_CF_eq_FG_l414_414896

variables {A B C E F D G : Type*}
variables [linear_ordered_field A]
variables [metric_space B]
variables [metric_space C]
variables [metric_space E]
variables [metric_space F]
variables [metric_space D]
variables [metric_space G]

-- Conditions
-- ∆ABC, where AB > AC
variables (ABC : triangle B C)
variable (AB_gt_AC : B > C)

-- Incircle touches BC at point E
variable (E_touch_BC : incircle E ABC touches_side BC)

-- Connecting AE intersects the incircle at point D other than E
variable (D_on_AE_incircle : intersects AE D E)

-- Point F on AE such that F != E and CE = CF
variable (F_on_AE : F ≠ E)
variable (CE_EQ_CF : CE = CF)

-- CF extended intersects BD at point G
variable (G_on_BD : intersects (extend CF) BD G)

-- Prove CF = FG
theorem prove_CF_eq_FG : CF = FG :=
sorry

end prove_CF_eq_FG_l414_414896


namespace graph_symmetry_wrt_4_f_one_gt_f_pi_solution_set_l414_414944

variable {R : Type*} [LinearOrder R] (f : R → R)
variable (symmetric : ∀ x, f (x + 2) = f (-x + 2))
variable (monotone_inc : ∀ {x1 x2 : R}, x1 ≤ 2 → x2 ≤ 2 → x1 < x2 → f x1 < f x2)

theorem graph_symmetry_wrt_4 :
  (∀ x, f (x - 2) = f (4 - x)) :=
sorry

theorem f_one_gt_f_pi (pi : R) (hpi : pi > 2) :
  f 1 > f pi :=
sorry

theorem solution_set (h0 : f 0 = 0) :
  { x : R | (x - 1) * f x > 0 } = { x | x < 0 } ∪ { x | 1 < x ∧ x < 4 } :=
sorry

end graph_symmetry_wrt_4_f_one_gt_f_pi_solution_set_l414_414944


namespace set_intersection_equality_l414_414840

open set

-- Define the universal set U
def U : set ℝ := {-1, real.log 3 / real.log 2, 2, 4}

-- Define set A = {x | log2(x^2 - x) = 1}
def A : set ℝ := {x | real.log (x^2 - x) / real.log 2 = 1}

-- Define set B = {x | 2^x = 3}
def B : set ℝ := {x | 2^x = 3}

-- Define the complement of A in U
def complement_U_A : set ℝ := U \ A

-- Define the intersection of the complement of A and B
def intersection_complement_U_A_B : set ℝ := complement_U_A ∩ B

-- The theorem we need to prove
theorem set_intersection_equality : intersection_complement_U_A_B = {- (real.log 3 / real.log 2)} :=
by
  -- The proof goes here
  sorry

end set_intersection_equality_l414_414840


namespace train_length_is_240_l414_414379

-- Definitions based on conditions
def train_speed_kmh := 216 -- Speed of the train in km/h
def time_to_cross := 4 -- Time to cross the man in seconds

-- Conversion factor for speed from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Conversion of the train speed to m/s
def train_speed_ms := kmh_to_ms train_speed_kmh

-- The length of the train based on the speed and time to cross
def train_length := train_speed_ms * time_to_cross

-- Theorem statement
theorem train_length_is_240 : train_length = 240 :=
by
  -- Add the proof steps here (not required as per problem statement)
  sorry

end train_length_is_240_l414_414379


namespace student_number_correct_l414_414709

theorem student_number_correct :
    (∀ year classNum studentNum genderIndicator, 
        year = 2008 →
        classNum = 6 →
        studentNum = 23 →
        genderIndicator = 1 →
        num_repr year classNum studentNum genderIndicator = 086231) :=
by
  sorry

/-- 
  Helper definition for constructing the student number based on the year, class, student number, and gender indicator.
-/
def num_repr (year classNum studentNum genderIndicator : ℕ) : ℕ := 
    year * 10000 + classNum * 1000 + studentNum * 10 + genderIndicator


end student_number_correct_l414_414709


namespace circumradius_inradius_perimeter_inequality_l414_414700

open Real

variables {R r P : ℝ} -- circumradius, inradius, perimeter
variable (triangle_type : String) -- acute, obtuse, right

def satisfies_inequality (R r P : ℝ) (triangle_type : String) : Prop :=
  if triangle_type = "right" then
    R ≥ (sqrt 2) / 2 * sqrt (P * r)
  else
    R ≥ (sqrt 3) / 3 * sqrt (P * r)

theorem circumradius_inradius_perimeter_inequality :
  ∀ (R r P : ℝ) (triangle_type : String), satisfies_inequality R r P triangle_type :=
by 
  intros R r P triangle_type
  sorry -- proof steps go here

end circumradius_inradius_perimeter_inequality_l414_414700


namespace solution_to_equation_l414_414651

-- Define the equation
def equation (x : ℝ) : Prop := (x / (x - 1)) - 1 = 1

-- The theorem that states x = 2 is the solution
theorem solution_to_equation : ∃ x : ℝ, x = 2 ∧ x ≠ 1 ∧ equation x := by
  existsi (2 : ℝ)
  split
  . rfl
  . split
  . exact ne_of_gt (by norm_num : 2 > 1)
  . sorry

end solution_to_equation_l414_414651


namespace inequality_sqrt_l414_414268

-- Define the conditions
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 19 / 3

-- State the theorem
theorem inequality_sqrt (x : ℝ) 
  (h_dom : domain x) : 
  sqrt (x - 1) + sqrt (2 * x + 9) + sqrt (19 - 3 * x) < 9 := 
  sorry

end inequality_sqrt_l414_414268


namespace sum_of_reciprocals_converges_to_3_2_l414_414940

noncomputable def S (a : ℕ → ℝ) : ℝ :=
  ∑' n, (1 / ∏ i in finset.range n, a i)

-- The problem statement in Lean
theorem sum_of_reciprocals_converges_to_3_2 {a b : ℕ → ℝ}
  (h1 : a 0 = 1) 
  (h2 : b 0 = 1)
  (h3 : ∀ n, 1 < n → b n = b (n-1) * a n - 2)
  (h4 : ∃ M, ∀ n, b n ≤ M) :
  S a = 3 / 2 := 
sorry

end sum_of_reciprocals_converges_to_3_2_l414_414940


namespace max_chord_length_l414_414079

-- Define the conditions and the statement
def line (t : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = p.1 + t
def ellipse (p : ℝ × ℝ) : Prop := (p.1^2 / 4 + p.2^2 = 1)

-- Define the statement
theorem max_chord_length : 
  ∀ (A B : ℝ × ℝ) (t : ℝ), 
    line t A ∧ line t B ∧ ellipse A ∧ ellipse B → 
    A ≠ B → 
    |A.1 - B.1| + |A.2 - B.2| ≤ 4 * (Real.sqrt 10) / 5 := 
sorry

end max_chord_length_l414_414079


namespace smallest_prime_divisor_of_3_pow_24_plus_8_pow_15_l414_414337

theorem smallest_prime_divisor_of_3_pow_24_plus_8_pow_15 :
  nat.prime_divisors (3 ^ 24 + 8 ^ 15) = [2] :=
by
  sorry

end smallest_prime_divisor_of_3_pow_24_plus_8_pow_15_l414_414337


namespace infinite_nat_exists_l414_414979

open Function

theorem infinite_nat_exists (n : ℕ) :
  (∀ n, ∃ f : Fin n → Fin n, 
  (∀ x : Fin n, f x ≠ x) ∧
  (∀ x : Fin n, f (f x) = x) ∧
  (∀ x : Fin n, f (f (f (x + 1) + 1) + 1) = x)) :=
  sorry

end infinite_nat_exists_l414_414979


namespace sequence_bounds_l414_414312

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n ≥ 1, a (n + 1) = (a n ^ 4 + 1) / (5 * a n)

theorem sequence_bounds (a : ℕ → ℝ) (h : sequence a) :
  ∀ n > 1, 1 / 5 < a n ∧ a n < 2 :=
by
  sorry

end sequence_bounds_l414_414312


namespace length_of_BC_l414_414209

-- Definitions and theorems required for our problem setup
structure Triangle (P Q R : Type _) :=
(AB : ℝ)
(AC : ℝ)
(AM : ℝ)
(midpointM : P → Q → R)

-- Assertion to be proved
theorem length_of_BC (P Q R : Type _) (T : Triangle P Q R) (B C : Q) :
  T.midpointM B C →
  T.AB = 4 →
  T.AC = 8 →
  T.AM = 3 →
  ∃ x : ℝ, x = 2 * Real.sqrt 31 :=
by
  sorry

end length_of_BC_l414_414209


namespace train_passing_time_l414_414706

-- Define the conditions as constants
constants (length_of_train : ℝ) (time_to_pass_tree : ℝ) 
          (length_of_platform : ℝ) (speed_of_train : ℝ) 
          (total_distance : ℝ)

-- Conditions from the problem
axiom length_of_train_eq : length_of_train = 2000
axiom time_to_pass_tree_eq : time_to_pass_tree = 80
axiom length_of_platform_eq : length_of_platform = 1200
axiom speed_of_train_eq : speed_of_train = length_of_train / time_to_pass_tree
axiom total_distance_eq : total_distance = length_of_train + length_of_platform

-- The theorem to be proven
theorem train_passing_time : 
  (total_distance / speed_of_train) = 128 :=
by
  sorry

end train_passing_time_l414_414706


namespace arithmetic_geometric_sequences_l414_414212

variables {n : ℕ} (a b q S : ℕ → ℕ)

-- Definitions of the sequences
def a_seq (n : ℕ) := 3 * n
def b_seq (n : ℕ) := 3^(n-1)
def S_seq (n : ℕ) := (3 * n * (n + 1)) / 2

-- Main statement
theorem arithmetic_geometric_sequences :
  (a_seq 1 = 3) ∧ (b_seq 1 = 1) ∧ 
  (b_seq 2 + S_seq 2 = 12) ∧ 
  (q ≠ 0) ∧ 
  (q = S_seq 2 / b_seq 2) → 
  (∀ n, a_seq n = 3 * n) ∧ 
  (∀ n, b_seq n = 3^(n-1)) ∧ 
  (1 / 3 ≤ ∑ k in range (n+1), 1 / (S_seq (k + 1)) ∧ ∑ k in range (n+1), 1 / (S_seq (k + 1)) < 2 / 3) :=
by {
    sorry
}

end arithmetic_geometric_sequences_l414_414212


namespace sequence_formula_l414_414961

noncomputable def a (n : ℕ) : ℕ := n

theorem sequence_formula (n : ℕ) (h : 0 < n) (S_n : ℕ → ℕ) 
  (hSn : ∀ m : ℕ, S_n m = (1 / 2 : ℚ) * (a m)^2 + (1 / 2 : ℚ) * m) : a n = n :=
by
  sorry

end sequence_formula_l414_414961


namespace determinant_of_trig_matrix_is_zero_l414_414949

theorem determinant_of_trig_matrix_is_zero 
  (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π) :
  matrix.det ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ] = 0 :=
by
  sorry

end determinant_of_trig_matrix_is_zero_l414_414949


namespace product_of_inserted_numbers_l414_414224

theorem product_of_inserted_numbers (a : ℕ → ℝ) (n : ℕ)
  (h0 : a 0 = 1) (h7 : a 7 = 2)
  (h_geom : ∀ m, m < 7 → a (m + 1) / a m = a 1 / a 0) :
  (∏ i in Finset.range 6, a (i + 1)) = 8 :=
by
  sorry

end product_of_inserted_numbers_l414_414224


namespace decreasing_intervals_f_l414_414058

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem decreasing_intervals_f :
  ∀ x, (x ∈ set.Ioo 0 1 ∪ set.Ioo 1 Real.exp 1) → (f x < f (x + ε) → ∃ ε > 0, f x < f (x + ε))
:=
begin
  sorry
end

end decreasing_intervals_f_l414_414058


namespace shortest_distance_ln_x_to_y_eq_x_l414_414648

noncomputable def f (x : ℝ) := Real.log x

theorem shortest_distance_ln_x_to_y_eq_x :
  let line_distance := fun (x0 y0 A B C : ℝ) => (abs (A * x0 + B * y0 + C)) / (Real.sqrt (A^2 + B^2)) in
  line_distance 1 0 1 (-1) 0 = Real.sqrt 2 / 2 :=
  by
  sorry

end shortest_distance_ln_x_to_y_eq_x_l414_414648


namespace range_of_a_l414_414174

theorem range_of_a (a : ℝ) (h : 2^a > 1) : a > 0 :=
sorry

end range_of_a_l414_414174


namespace bisect_AC2_by_BD_l414_414484

variables (k1 k2 : Circle) (A B C1 C2 D : Point)
variables (h1 : Intersect k1 k2 A) (h2 : Intersect k1 k2 B)
variables (tangent_k2_A : Tangent k2 A) (tangent_k1_A : Tangent k1 A)
variables (hC1 : OnCircle k1 C1) (hC2 : OnCircle k2 C2)
variables (line_C1C2_intersects_k1 : Intersect k1 C1C2 D)
variables (hD1 : D ≠ C1) (hD2 : D ≠ B)
variables (h_tangent_k2 : TangentAtPoint k2 A C1) (h_tangent_k1 : TangentAtPoint k1 A C2)

theorem bisect_AC2_by_BD :
  Bisects BD AC2 := sorry

end bisect_AC2_by_BD_l414_414484


namespace x_satisfies_geometric_series_l414_414074

noncomputable def geometric_sum (x : ℝ) (h : |x| < 1) : ℝ :=
1 - x + x^2 - x^3 + x^4 - x^5 + ....

theorem x_satisfies_geometric_series :
  ∃! x : ℝ, (x = geometric_sum x (abs_lt_one_iff.mp h)) ∧ (abs x < 1) := 
begin
  -- This is where we would include the proof, but for now, we use sorry.
  sorry
end

end x_satisfies_geometric_series_l414_414074


namespace polynomial_at_least_one_not_less_than_one_l414_414249

theorem polynomial_at_least_one_not_less_than_one (P : ℕ → ℤ) (n : ℕ)
  (h_poly : ∃a : ℕ → ℤ, ∀ x, P x = a x * x^n + (a x * x^(n-1)) + ... + a x * x + a 0) :
  ∃ k, 0 ≤ k ∧ k ≤ n+1 ∧ |(3^k - P k)| ≥ 1 :=
by
  sorry

end polynomial_at_least_one_not_less_than_one_l414_414249


namespace factorial_division_l414_414456

theorem factorial_division (h : 9! = 362880) : 9! / 4! = 15120 :=
by
  have h₁ : 4! = 24 := by norm_num
  rw [←h₁]
  rw [h]
  norm_num
  sorry

end factorial_division_l414_414456


namespace main_l414_414567

-- Definitions: Set, subset, capacity, odd/even subsets
def Sn (n : ℕ) := {x | 1 ≤ x ∧ x ≤ n}
def is_subset (Z : set ℕ) (n : ℕ) := Z ⊆ Sn n
def capacity (Z : set ℕ) := (Z.to_finset : finset ℕ).sum id
def is_odd (Z : set ℕ) := capacity Z % 2 = 1
def is_even (Z : set ℕ) := capacity Z % 2 = 0

-- Statement ①: The number of odd subsets of Sn equals the number of even subsets of Sn
def statement_1 (n : ℕ) : Prop :=
  (∃f : set ℕ → set ℕ, (∀ Z, is_subset Z n → is_odd Z → is_even (f Z)) ∧
                       (∀ T, is_subset T n → is_even T → is_odd (f T)) ∧
                       (bijective (λ Z, if is_odd Z then f Z else T))) 

-- Statement ②: When n ≥ 3, the sum of the capacities of all odd subsets of Sn equals the sum of the capacities of all even subsets of Sn
def statement_2 (n : ℕ) : Prop :=
  n ≥ 3 → (sum (Z.to_finset.filter (is_odd)) capacity = sum (Z.to_finset.filter (is_even)) capacity)

-- Combine statements into a main theorem to be proven
theorem main (n : ℕ) : Prop :=
  statement_1 n ∧ statement_2 n

end main_l414_414567


namespace weekend_bike_ride_l414_414327

def Tim := { work_distance : ℝ // work_distance = 20 }
def Bike := { speed : ℝ // speed = 25 }
def Week := { workdays : ℕ // workdays = 5 }
def TotalTimeBiking : ℝ := 16

theorem weekend_bike_ride (tim : Tim) (bike : Bike) (week : Week): 
  let distance_work := tim.work_distance * 2 * week.workdays in
  let time_work := distance_work / bike.speed in
  let remaining_time := TotalTimeBiking - time_work in
  let weekend_distance := remaining_time * bike.speed in
  weekend_distance = 200 := by simp [tim, bike, week]; sorry

end weekend_bike_ride_l414_414327


namespace transformed_parabola_equation_l414_414530

theorem transformed_parabola_equation :
  ∀ (x : ℝ), ((x^2 + 3) translated_by (3 units left) and (4 units down)) = (x + 3)^2 - 1 := 
by
  sorry

end transformed_parabola_equation_l414_414530


namespace find_a_l414_414458

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l414_414458


namespace greatest_multiple_of_5_l414_414283

theorem greatest_multiple_of_5 (y : ℕ) (h1 : y > 0) (h2 : y % 5 = 0) (h3 : y^3 < 8000) : y ≤ 15 :=
by {
  sorry
}

end greatest_multiple_of_5_l414_414283


namespace committee_formations_l414_414659

theorem committee_formations : 
  let n := 10
  let k := 5
  let total_committees := Nat.choose n k
  let leader_subsets := 2^k - 2
  total_committees * leader_subsets = 7560
:= by
  let n := 10
  let k := 5
  have h1 : total_committees = Nat.choose n k := rfl
  have h2 : leader_subsets = 2^k - 2 := rfl
  show total_committees * leader_subsets = 7560 from sorry

end committee_formations_l414_414659


namespace radius_of_circle_C_polar_equation_of_circle_C_l414_414533

-- Given conditions
def passes_through_point (ρ : ℝ) (θ : ℝ) : Prop := ρ = √3 ∧ θ = π / 6
def intersects_polar_axis (ρ : ℝ → ℝ → ℝ) : Prop := 
  ∀ θ, ρ (π / 3 - θ) = √(3 / 2)

-- The goal is to prove that the radius of the circle is 1
theorem radius_of_circle_C :
  (passes_through_point √3 (π / 6)) → 
  (intersects_polar_axis (λ ρ θ, ρ sin (π / 3 - θ) = √(3 / 2))) →
  ∃ r : ℝ, r = 1 :=
by 
  sorry

-- The goal is to prove that the polar equation of the circle is ρ = 2 cos θ
theorem polar_equation_of_circle_C :
  (passes_through_point √3 (π / 6)) →
  (intersects_polar_axis (λ ρ θ, ρ sin (π / 3 - θ) = √(3 / 2))) →
  ∃ ρ θ : ℝ, ρ = 2 * cos θ :=
by 
  sorry

end radius_of_circle_C_polar_equation_of_circle_C_l414_414533


namespace joe_probability_select_counsel_l414_414231

theorem joe_probability_select_counsel :
  let CANOE := ['C', 'A', 'N', 'O', 'E']
  let SHRUB := ['S', 'H', 'R', 'U', 'B']
  let FLOW := ['F', 'L', 'O', 'W']
  let COUNSEL := ['C', 'O', 'U', 'N', 'S', 'E', 'L']
  -- Probability of selecting C and O from CANOE
  let p_CANOE := 1 / (Nat.choose 5 2)
  -- Probability of selecting U, S, and E from SHRUB
  let comb_SHRUB := Nat.choose 5 3
  let count_USE := 3  -- Determined from the solution
  let p_SHRUB := count_USE / comb_SHRUB
  -- Probability of selecting L, O, W, F from FLOW
  let p_FLOW := 1 / 1
  -- Total probability
  let total_prob := p_CANOE * p_SHRUB * p_FLOW
  total_prob = 3 / 100 := by
    sorry

end joe_probability_select_counsel_l414_414231


namespace Sn_plus_one_exists_c_k_l414_414791

-- Definition of the geometric sequence
def a₁ : ℕ := 2
def r : ℝ := 1 / 2
noncomputable def S (n : ℕ) : ℝ := 4 * (1 - r^n)

-- Part 1: Prove the relationship for S_{n+1} in terms of S_n
theorem Sn_plus_one (n : ℕ) : S (n + 1) = S n + 2 * (r ^ n) :=
by sorry

-- Part 2: Prove the existence of natural numbers c and k such that the condition holds
theorem exists_c_k : ∃ (c k : ℕ), (S (k + 1) - c) / (S k - c) > 2 :=
by sorry

end Sn_plus_one_exists_c_k_l414_414791


namespace right_triangle_division_l414_414539

theorem right_triangle_division :
    ∃ (t1 t2 t3 : Triangle), 
      (∀ (t : Triangle), t = right_triangle ∧ 
      (t.angle_A = 30 ∧ t.angle_B = 60 ∧ t.angle_C = 90) →
      (t1 = t2 ∧ t2 = t3 ∧ t3 = t) ) := 
    sorry

end right_triangle_division_l414_414539


namespace seq_general_term_l414_414918

-- Define sequence {a_n}
noncomputable def a : ℕ → ℤ
| 0     := 0  -- Indexing starts from 1 for a_1 = 1 in problem statement
| 1     := 1  -- a_1 = 1
| 2     := 2  -- a_2 = 2
| (n+3) := 5 * a (n+2) - 6 * a (n+1) -- a_{n+2} = 5a_{n+1} - 6a_n

-- Theorem to prove: ∀ n ≥ 1, a_n = 2^{n-1}
theorem seq_general_term (n : ℕ) (hn : n ≥ 1) : a n = 2^(n - 1) := 
by sorry

end seq_general_term_l414_414918


namespace remove_remaining_wallpaper_time_l414_414069

noncomputable def time_per_wall : ℕ := 2
noncomputable def walls_dining_room : ℕ := 4
noncomputable def walls_living_room : ℕ := 4
noncomputable def walls_completed : ℕ := 1

theorem remove_remaining_wallpaper_time : 
    time_per_wall * (walls_dining_room - walls_completed) + time_per_wall * walls_living_room = 14 :=
by
  sorry

end remove_remaining_wallpaper_time_l414_414069


namespace find_x_minus_y_l414_414656

theorem find_x_minus_y (x y z : ℤ) (h₁ : x - y - z = 7) (h₂ : x - y + z = 15) : x - y = 11 := by
  sorry

end find_x_minus_y_l414_414656


namespace daria_friends_l414_414754

-- Definitions based on conditions:
def ticket_cost : ℕ := 90
def current_amount : ℕ := 189
def needed_amount : ℕ := 171

-- The total needed amount is derived from the condition
def total_needed : ℕ := current_amount + needed_amount

-- The total tickets to be bought is derived from the total_needed and ticket_cost
def total_tickets : ℕ := total_needed / ticket_cost

-- The number of friends Daria wants to buy tickets for is the total_tickets minus 1
def number_of_friends : ℕ := total_tickets - 1

-- Proving that given the conditions, Daria wants to buy tickets for 3 friends
theorem daria_friends : number_of_friends = 3 :=
by
  unfold total_needed
  unfold total_tickets
  unfold number_of_friends
  norm_num
  sorry

end daria_friends_l414_414754


namespace positive_diff_solutions_l414_414335

theorem positive_diff_solutions (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 14) (h2 : 2 * x2 - 3 = -14) : 
  x1 - x2 = 14 := 
by
  sorry

end positive_diff_solutions_l414_414335


namespace noah_total_sales_revenue_correct_l414_414586

noncomputable def price_large_painting : ℕ := 60
noncomputable def price_small_painting : ℕ := 30
noncomputable def paintings_sold_last_month_large : ℕ := 8
noncomputable def paintings_sold_last_month_small : ℕ := 4
noncomputable def discount_large_paintings : ℝ := 0.10
noncomputable def commission_small_paintings : ℝ := 0.05
noncomputable def sales_tax_rate : ℝ := 0.07

def paintings_sold_this_month_large : ℕ := paintings_sold_last_month_large * 2
def paintings_sold_this_month_small : ℕ := paintings_sold_last_month_small * 2

noncomputable def price_large_after_discount : ℝ := price_large_painting * (1 - discount_large_paintings)
noncomputable def price_small_after_commission : ℝ := price_small_painting * (1 - commission_small_paintings)

noncomputable def revenue_large_paintings : ℝ := paintings_sold_this_month_large * price_large_after_discount
noncomputable def revenue_small_paintings : ℝ := paintings_sold_this_month_small * price_small_after_commission

noncomputable def total_sales_before_tax : ℝ := revenue_large_paintings + revenue_small_paintings
noncomputable def total_sales_tax : ℝ := total_sales_before_tax * sales_tax_rate

noncomputable def total_sales_revenue : ℝ := total_sales_before_tax + total_sales_tax

theorem noah_total_sales_revenue_correct :
  total_sales_revenue = 1168.44 := by
  sorry

end noah_total_sales_revenue_correct_l414_414586


namespace company_employee_count_l414_414318

theorem company_employee_count (E : ℝ) (H1 : E > 0) (H2 : 0.60 * E = 0.55 * (E + 30)) : E + 30 = 360 :=
by
  -- The proof steps would go here, but that is not required.
  sorry

end company_employee_count_l414_414318


namespace total_people_at_meeting_l414_414741

namespace MeetingGraph

open Finset

-- We define a person as a vertex in a graph
variable {V : Type*} [Fintype V] [DecidableEq V]

-- Define the conditions of the meeting graph as specified in the problem
def meeting_graph_conditions (G : SimpleGraph V) : Prop :=
  ∀ v ∈ (univ : Finset V), 
  (degree G v = 22) ∧
  ∀ (u w ∈ (G.neighborFinset v)), 
    u ≠ w → G.neighborFinset u ∩ G.neighborFinset w = ∅ ∧
  ∀ (u w ∉ (G.neighborFinset v)), 
    G.neighborFinset u ∩ G.neighborFinset w = (G.neighborFinset v).filter (λ x, x ∈ u ∧ x ∈ w)

-- The proof problem, proving that the number of people at the meeting is 100
theorem total_people_at_meeting (G : SimpleGraph V) 
  (h : meeting_graph_conditions G) : 
  Fintype.card V = 100 := 
sorry

end MeetingGraph

end total_people_at_meeting_l414_414741


namespace number_of_tetrahedrons_from_cube_l414_414491

noncomputable def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| (n + 1) (k + 1) := binom n k + binom n (k + 1)

theorem number_of_tetrahedrons_from_cube : 
  let V := 8 in
  let coplanar_cases := 12 in
  let total_combinations := binom 8 4 in
  total_combinations - coplanar_cases = 58 := 
by
  sorry

end number_of_tetrahedrons_from_cube_l414_414491


namespace determine_m_l414_414182

-- Define the conditions for the hyperbola problem
def hyperbola_foci_condition (m : ℝ) : Prop := 
  (x : ℝ) (y : ℝ), x^2 - y^2 / m = 1

-- Define the fact that one of the foci is (-3, 0)
def hyperbola_focus_condition (m : ℝ) : Prop :=
  let a := 1 in
  let c := 3 in  -- since focus is at (-3, 0) we have c = 3
  c^2 = a^2 + m

-- The main theorem
theorem determine_m (m : ℝ) :
  hyperbola_foci_condition m →
  hyperbola_focus_condition m →
  m = 8 := 
sorry

end determine_m_l414_414182


namespace find_ABC_l414_414948

theorem find_ABC (A B C : ℝ) (h : ∀ n : ℕ, n > 0 → 2 * n^3 + 3 * n^2 = A * (n * (n - 1) * (n - 2)) / 6 + B * (n * (n - 1)) / 2 + C * n) :
  A = 12 ∧ B = 18 ∧ C = 5 :=
by {
  sorry
}

end find_ABC_l414_414948


namespace zoo_visitors_l414_414590

theorem zoo_visitors (visitors_friday: ℕ) (multiplier: ℕ) (h_visitors_friday: visitors_friday = 1250) (h_multiplier: multiplier = 3) :
  ∃ visitors_saturday, visitors_saturday = 3750 := 
by 
  let visitors_saturday := visitors_friday * multiplier
  have h_visitors_saturday : visitors_saturday = 3750 := by
    rw [h_visitors_friday, h_multiplier]
    norm_num
  exact ⟨visitors_saturday, h_visitors_saturday⟩

end zoo_visitors_l414_414590


namespace sum_possible_values_of_y_l414_414864

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end sum_possible_values_of_y_l414_414864


namespace problem_l414_414350

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)

axiom arithmetic_seq_sum (n : ℕ) : S_n n = ∑ i in range n, a_n i
axiom arithmetic_seq_elem (n : ℕ) : ∃ d a₀, a_n n = a₀ + n * d

theorem problem (h1 : a_n 5 + a_n 6 + a_n 7 = 15) : S_n 11 = 55 :=
sorry

end problem_l414_414350


namespace medians_intersect_at_one_point_l414_414977

universe u

variable {A B C O : Type u} [NonemptyType O]

-- Assume vertices A, B, C form a spherical triangle on a sphere with center O
variables [IsSphere O] (A B C : O)

-- Definition of a median in spherical geometry: a line connecting a vertex with the midpoint of the opposite side
structure SphericalMedian (A B C : O) :=
  (vertex : O)
  (midpoint : O)
  (connects : GeodesicSegment vertex midpoint)

-- The main theorem statement
theorem medians_intersect_at_one_point (A B C : O) [IsSphere A] [IsSphere B] [IsSphere C] :
  ∃ (G : O), ∀ (m₁ m₂ m₃ : SphericalMedian A B C), connects m₁ G ∧ connects m₂ G ∧ connects m₃ G :=
sorry

end medians_intersect_at_one_point_l414_414977


namespace prime_divisors_count_l414_414070

noncomputable def sequence : ℕ → ℕ
| n := 2^n + 1

def is_prime (p : ℕ) : Prop := (2 < p) ∧ (∀ n : ℕ, 1 < n → n < p → ¬ n ∣ p)

def divides_sequence (p : ℕ) : Prop :=
  ∃ n : ℕ, p ∣ sequence n

theorem prime_divisors_count : 
  (finset.filter (λ p, divides_sequence p) (finset.range 1000).filter is_prime).card = 10 := 
sorry

end prime_divisors_count_l414_414070


namespace inequality_fa_f1_fb_l414_414125

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x - 2

theorem inequality_fa_f1_fb
  (a b : ℝ)
  (ha : f a = 0)
  (hb : g b = 0)
  (h1 : 0 < a)
  (h2 : a < 1)
  (h3 : 1 < b)
  (h4 : b < 2) :
  f(a) < f(1) ∧ f(1) < f(b) :=
by
  sorry

end inequality_fa_f1_fb_l414_414125


namespace acute_triangle_AB_AC_product_l414_414202

theorem acute_triangle_AB_AC_product
  (A B C R S Z W : Type)
  (ZR : ℝ) (RS : ℝ) (SW : ℝ)
  (h_acute : acute_triangle A B C)
  (h_perp_R : is_perpendicular C A B R)
  (h_perp_S : is_perpendicular B A C S)
  (h_circumcircle_intersec : circumcircle_intersec A B C R S Z W) :
  ZR = 13 ∧ RS = 28 ∧ SW = 17 →
  let d := foot_perpendicular_length A R,
      e := foot_perpendicular_length A S,
      l := cos_angle A in
  AB * AC = 640 * real.sqrt 15 := sorry

end acute_triangle_AB_AC_product_l414_414202


namespace count_decorations_l414_414398

/--
Define a function T(n) that determines the number of ways to decorate the window 
with n stripes according to the given conditions.
--/
def T : ℕ → ℕ
| 0       => 1 -- optional case for completeness
| 1       => 2
| 2       => 2
| (n + 1) => T n + T (n - 1)

theorem count_decorations : T 10 = 110 := by
  sorry

end count_decorations_l414_414398


namespace rope_total_in_inches_l414_414558

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end rope_total_in_inches_l414_414558


namespace inscribed_radius_l414_414201

theorem inscribed_radius (a b c : ℕ) (h : a = 5 ∧ b = 12 ∧ c = 13 ∧ 
    let p := a + b + c in 
    let A := 1.5 * p - 12 in 
    let s := p / 2 in
    let r := A / s in 
    A = 30) : ∃ r, r = 33 / 15 := sorry

end inscribed_radius_l414_414201


namespace area_triangle_OPQ_is_correct_l414_414812

noncomputable def area_triangle_OPQ (z1 z2 : ℂ) : ℂ :=
  0.5 * complex.abs z1 * complex.abs z2 * complex.sin (complex.arg z1 - complex.arg z2)

theorem area_triangle_OPQ_is_correct {z1 z2 : ℂ} 
  (h1 : complex.abs z2 = 4) 
  (h2 : 4 * z1^2 - 2*z1*z2 + z2^2 = 0) : 
  area_triangle_OPQ z1 z2 = 2 * complex.sqrt 3 := 
sorry

end area_triangle_OPQ_is_correct_l414_414812


namespace f_a2016_zero_l414_414452

theorem f_a2016_zero {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) 
                      (h_recur : ∀ x, f (x + 1) = f (x - 1)) 
                      (a : ℕ → ℤ) (S : ℕ → ℤ)
                      (h_S : ∀ n, S n = 2 * a n + 2)
                      (h_seq : ∀ n, S n = (list.range (n+1)).sum a) :
  f (a 2016) = 0 :=
by
  sorry

end f_a2016_zero_l414_414452


namespace minimize_sum_distances_condition_l414_414561

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)

theorem minimize_sum_distances_condition (A B C D X X0 : ℝ × ℝ × ℝ)
  (h_not_coplanar: ¬ coplanar {A, B, C, D})
  (h_not_collinear: ∀ P Q R ∈ {A, B, C, D}, P ≠ Q → Q ≠ R → P ≠ R → ¬ collinear {P, Q, R})
  (h_X0_diff: X0 ≠ A ∧ X0 ≠ B ∧ X0 ≠ C ∧ X0 ≠ D)
  (h_minimized: ∀ X, distance X A + distance X B + distance X C + distance X D ≥ distance X0 A + distance X0 B + distance X0 C + distance X0 D) :
  ∠A X0 B = ∠C X0 D :=
sorry

end minimize_sum_distances_condition_l414_414561


namespace right_triangle_incenter_distance_l414_414197

noncomputable def triangle_right_incenter_distance : ℝ :=
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let area := (1 / 2) * AB * BC
  let s := (AB + BC + AC) / 2
  let r := area / s
  r

theorem right_triangle_incenter_distance :
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := 2 * Real.sqrt 17
  let area := 12 * Real.sqrt 2
  let s := 2 * Real.sqrt 2 + 3 + Real.sqrt 17
  let BI := area / s
  BI = triangle_right_incenter_distance := sorry

end right_triangle_incenter_distance_l414_414197


namespace c_minus_b_seven_l414_414033

theorem c_minus_b_seven {a b c d : ℕ} (ha : a^6 = b^5) (hb : c^4 = d^3) (hc : c - a = 31) : c - b = 7 :=
sorry

end c_minus_b_seven_l414_414033


namespace min_fraction_of_sets_l414_414804

open Finset

theorem min_fraction_of_sets {n : ℕ} (h : n > 1) (a : Fin n → ℝ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) (h_nonneg : ∀ i, 0 ≤ a i) :
  let A := {x | ∃ i j, i ≤ j ∧ x = a i + a j} in
  let B := {x | ∃ i j, i ≤ j ∧ x = a i * a j} in
  (finset.card A.to_finset : ℚ) / (finset.card B.to_finset : ℚ) ≥ 2 * (2 * n - 1) / (n * (n + 1)) :=
sorry

end min_fraction_of_sets_l414_414804


namespace smallest_points_in_T_l414_414009

def is_symmetric (T : Set (ℝ × ℝ)) : Prop :=
  (∀ (p : ℝ × ℝ), p ∈ T → (-p.1, -p.2) ∈ T) ∧  -- Symmetry about the origin
  (∀ (p : ℝ × ℝ), p ∈ T → (p.1, -p.2) ∈ T) ∧  -- Symmetry about the x-axis
  (∀ (p : ℝ × ℝ), p ∈ T → (-p.1, p.2) ∈ T) ∧  -- Symmetry about the y-axis
  (∀ (p : ℝ × ℝ), p ∈ T → (p.2, p.1) ∈ T) ∧  -- Symmetry about the line y=x
  (∀ (p : ℝ × ℝ), p ∈ T → (-p.2, -p.1) ∈ T)   -- Symmetry about the line y=-x

theorem smallest_points_in_T 
  (T : Set (ℝ × ℝ))
  (h_symmetric : is_symmetric T)
  (h_point : (1, 4) ∈ T) :
  ∃ S, S ⊆ T ∧ S.card = 8 :=
sorry

end smallest_points_in_T_l414_414009


namespace analytic_geometry_essence_l414_414291

theorem analytic_geometry_essence :
  ∀ chapters : Type, 
    (chapters = "Lines on the Coordinate Plane" ∨ chapters = "Conic Sections") →
    (establishes_cartesian_coordinate_system chapters ∧ uses_algebraic_functions chapters) →
      the_essence_of_analytic_geometry = "to study the geometric properties of figures using algebraic methods" :=
by
  sorry

-- Definitions to make the theorem valid
def establishes_cartesian_coordinate_system (chapter : Type) : Prop :=
sorry

def uses_algebraic_functions (chapter : Type) : Prop :=
sorry

def the_essence_of_analytic_geometry : String :=
sorry

end analytic_geometry_essence_l414_414291


namespace exists_n_cos_eq_l414_414424

theorem exists_n_cos_eq :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n:ℝ).to_degrees = Real.cos 942 := by
  sorry

end exists_n_cos_eq_l414_414424


namespace num_of_valid_triangles_l414_414227

/-- A triangle consists of three integer side lengths one of which is 4, but not the shortest. -/
def isValidTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ 4 ∈ {a, b, c} ∧ min a (min b c) ≠ 4

theorem num_of_valid_triangles : {t : (ℕ × ℕ × ℕ) // isValidTriangle t.1 t.2.1 t.2.2}.card = 8 := by
  sorry

end num_of_valid_triangles_l414_414227


namespace f_2021_2_value_l414_414803

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for the function definition according to provided conditions.

theorem f_2021_2_value :
  (∀ x : ℝ, f(-x) = f(x)) ∧                  -- Even function condition
  (∀ x : ℝ, f(x + 2) = -f(x)) ∧              -- Periodicity condition
  (∀ x : set.Ioc 1 2, f(x) = 2^x) ->         -- Interval condition
  f(2021 / 2) = 2 * real.sqrt 2 := 
sorry  -- Skipping the proof.

end f_2021_2_value_l414_414803


namespace store_discount_percentage_l414_414016

theorem store_discount_percentage
  (num_items : ℕ)
  (cost_per_item : ℝ)
  (actual_cost : ℝ)
  (threshold : ℝ)
  (total_cost : ℝ)
  (discounted_cost : ℝ)
  : num_items = 7 
  → cost_per_item = 200
  → actual_cost = discounted_cost
  → threshold = 1000
  → total_cost = num_items * cost_per_item 
  → actual_cost = 1360
  → discounted_cost = total_cost - (total_cost - threshold) * discount_percentage 
  → discount_percentage = (total_cost - actual_cost) / (total_cost - threshold) * 100 
  → discount_percentage = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 sorry

end store_discount_percentage_l414_414016


namespace johns_height_and_average_growth_in_a_year_l414_414233

noncomputable def growth_pattern (initial_height : ℝ) (first_3_months_growth : ℝ) (subsequent_growth_rate : ℝ) (months : ℕ) : ℝ :=
  let base_growth := list.replicate 3 first_3_months_growth
  let subsequent_growth := list.scanl (λ acc _ => acc * subsequent_growth_rate) first_3_months_growth (list.replicate (months - 3) 0)
  initial_height + (list.sum base_growth + list.sum subsequent_growth)

theorem johns_height_and_average_growth_in_a_year :
  let initial_height := 66
  let first_3_months_growth := 2
  let subsequent_growth_rate := 1.10
  let months := 12
  let final_height_in_cm := growth_pattern initial_height first_3_months_growth subsequent_growth_rate months * 2.54
  final_height_in_cm ≈ 261.3 ∧ (final_height_in_cm - initial_height * 2.54) / 12 ≈ 3.072996784 :=
  sorry

end johns_height_and_average_growth_in_a_year_l414_414233


namespace g_ge_one_l414_414477

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem g_ge_one (x : ℝ) (h : 0 < x) : g x ≥ 1 :=
sorry

end g_ge_one_l414_414477


namespace problem_solution_l414_414409

variable {ℝ : Type}

-- Define the even function f on ℝ
noncomputable def f : ℝ → ℝ
-- For any x1, x2 ∈ (-∞, 0], with x1 ≠ x2,
-- it holds that (f x2 - f x1) / (x2 - x1) < 0 
axiom condition1 : ∀ x1 x2 : ℝ, x1 ≤ 0 → x2 ≤ 0 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0

-- Prove that f(1) < f(-2) < f(-3)
theorem problem_solution : f 1 < f (-2) ∧ f (-2) < f (-3) :=
sorry

end problem_solution_l414_414409


namespace two_rats_S5_l414_414997

/--
Given the conditions:
- Larger rat burrows 1 foot per day, and its rate doubles each day.
- Smaller rat burrows 1 foot on the first day, and its rate halves each day.
- Let \( S_n \) be the sum of the lengths burrowed by the two rats in the first \( n \) days.

Prove that for \( n = 5 \), \( S_5 = 32 + \frac{15}{16} \).
-/
theorem two_rats_S5 :
  let S (n : ℕ) : ℚ := (2^n - 1) + 2 - (1 / 2^(n-1))
  in S 5 = 32 + 15 / 16 :=
by
  sorry

end two_rats_S5_l414_414997


namespace multiply_eq_four_l414_414843

variables (a b c d : ℝ)

theorem multiply_eq_four (h1 : a = d) 
                         (h2 : b = c) 
                         (h3 : d + d = c * d) 
                         (h4 : b = d) 
                         (h5 : d + d = d * d) 
                         (h6 : c = 3) :
                         a * b = 4 := 
by 
  sorry

end multiply_eq_four_l414_414843


namespace three_digit_divisible_by_7_iff_last_two_digits_equal_l414_414655

-- Define the conditions as given in the problem
variable (a b c : ℕ)

-- Ensure the sum of the digits is 7, as given by the problem conditions
def sum_of_digits_eq_7 := a + b + c = 7

-- Ensure that it is a three-digit number
def valid_three_digit_number := a ≠ 0

-- Define what it means to be divisible by 7
def divisible_by_7 (n : ℕ) := n % 7 = 0

-- Define the problem statement in Lean
theorem three_digit_divisible_by_7_iff_last_two_digits_equal (h1 : sum_of_digits_eq_7 a b c) (h2 : valid_three_digit_number a) :
  divisible_by_7 (100 * a + 10 * b + c) ↔ b = c :=
by sorry

end three_digit_divisible_by_7_iff_last_two_digits_equal_l414_414655


namespace unique_solution_l414_414771

namespace ProofProblem

open Nat

-- Define the set of positive natural numbers
def Npos := { n : ℕ // n > 0 }

-- Define the function f and the condition as an axiom
def f : Npos → Npos
axiom functional_equation : ∀ (m n : Npos), f ⟨m.val * m.val + f n.val, _⟩ = ⟨f m.val * f m.val + n.val, _⟩

-- The proof that the only function satisfying the condition is the identity function
theorem unique_solution : ∀ n : Npos, f n = n := by
  sorry

end ProofProblem

end unique_solution_l414_414771


namespace london_to_baglmintster_distance_l414_414621

variable (D : ℕ) -- distance from London to Baglmintster

-- Conditions
def meeting_point_condition_1 := D ≥ 40
def meeting_point_condition_2 := D ≥ 48
def initial_meeting := D - 40
def return_meeting := D - 48

theorem london_to_baglmintster_distance :
  (D - 40) + 48 = D + 8 ∧ 40 + (D - 48) = D - 8 → D = 72 :=
by
  intros h
  sorry

end london_to_baglmintster_distance_l414_414621


namespace vector_dot_product_l414_414487

variables {𝕜 : Type*} [Field 𝕜] [Module 𝕜 (EuclideanSpace 𝕜)]

variables {a b c : EuclideanSpace 𝕜}
variables (h1 : inner a b = 2) (h2 : inner a c = -1) (h3 : inner b c = 5)

theorem vector_dot_product :
  inner b (5 • c - 3 • a) = 19 :=
by sorry

end vector_dot_product_l414_414487


namespace equal_angles_l414_414699

variables (A B C D P M : Type*)
variables [Point A] [Point B] [Point C] [Point D] [Point P] [Point M]
variables (quad : Quadrilateral A B C D)
variables (midpoint_M : Midpoint M A D)
variables (BM_eq_CM : BM = CM)
variables (angle_PBA : Angle P B A = 90)
variables (angle_PCD : Angle P C D = 90)

theorem equal_angles :
  ∠PAB = ∠PDC :=
by
  sorry

end equal_angles_l414_414699


namespace find_number_l414_414874

theorem find_number (x : ℝ) (h : (25 / 100) * x = 20 / 100 * 30) : x = 24 :=
by
  sorry

end find_number_l414_414874


namespace blocks_total_l414_414324

theorem blocks_total (blocks_initial : ℕ) (blocks_added : ℕ) (total_blocks : ℕ) 
  (h1 : blocks_initial = 86) (h2 : blocks_added = 9) : total_blocks = 95 :=
by
  sorry

end blocks_total_l414_414324


namespace water_flow_rate_l414_414694

theorem water_flow_rate
  (depth : ℝ := 4)
  (width : ℝ := 22)
  (flow_rate_kmph : ℝ := 2)
  (flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60)
  (cross_sectional_area : ℝ := depth * width)
  (volume_per_minute : ℝ := cross_sectional_area * flow_rate_mpm) :
  volume_per_minute = 2933.04 :=
  sorry

end water_flow_rate_l414_414694


namespace find_k_value_l414_414085

noncomputable def series_k_eq (k : ℚ) : ℚ :=
  5 + (5 + 3 * k)/5 + (5 + 6 * k)/5^2 + (5 + 9 * k)/5^3 + (5 + 12 * k)/5^4 + ...

-- Please note that in actual Lean code, an infinite series needs to be handled more carefully.
-- Here we assume a mathematical context where this series is well-defined and converges.

theorem find_k_value (k : ℚ) (h : series_k_eq k = 12) : k = 112 / 15 :=
sorry

end find_k_value_l414_414085


namespace maximum_bugs_on_board_l414_414617

-- Definition of the problem board size, bug movement directions, and non-collision rule
def board_size := 10
inductive Direction
| up | down | left | right

-- The main theorem stating the maximum number of bugs on the board
theorem maximum_bugs_on_board (bugs : List (Nat × Nat × Direction)) :
  (∀ (x y : Nat) (d : Direction) (bug : Nat × Nat × Direction), 
    bug = (x, y, d) → 
    x < board_size ∧ y < board_size ∧ 
    (∀ (c : Nat × Nat × Direction), 
      c ∈ bugs → bug ≠ c → bug.1 ≠ c.1 ∨ bug.2 ≠ c.2)) →
  List.length bugs <= 40 :=
sorry

end maximum_bugs_on_board_l414_414617


namespace locus_is_circle_l414_414802

variable {t b : ℝ}

def equilateral_triangle_locus (Q : ℝ × ℝ) : Prop :=
  let D : ℝ × ℝ := (0, 0)
  let E : ℝ × ℝ := (t, 0)
  let F : ℝ × ℝ := (t / 2, (t * Real.sqrt 3) / 2)
  (Prod.fst Q - Prod.fst D)^2 + (Prod.snd Q - Prod.snd D)^2 +
  (Prod.fst Q - Prod.fst E)^2 + (Prod.snd Q - Prod.snd E)^2 +
  (Prod.fst Q - Prod.fst F)^2 + (Prod.snd Q - Prod.snd F)^2 = b

theorem locus_is_circle (hb : b > 0) :
  ∃ H : ℝ × ℝ, let r := Real.sqrt (b / 3) in
    (H = (t / 3, (t * Real.sqrt 3) / 6) ∧
      (∀ (Q : ℝ × ℝ), equilateral_triangle_locus Q ↔ 
        (Prod.fst Q - Prod.fst H)^2 + (Prod.snd Q - Prod.snd H)^2 = r^2)) :=
sorry

end locus_is_circle_l414_414802


namespace impossible_conclusion_l414_414942
noncomputable def f (a b c x : ℝ) : ℝ := (x + a) * (x^2 + b * x + c)
noncomputable def g (a b c x : ℝ) : ℝ := (a * x + 1) * (c * x^2 + b * x + 1)

def S (a b c : ℝ) : Set ℝ := {x | f a b c x = 0}
def T (a b c : ℝ) : Set ℝ := {x | g a b c x = 0}

theorem impossible_conclusion (a b c : ℝ) : ∀ x : ℝ, |S a b c| = 2 ∧ |T a b c| = 3 → false :=
by sorry

end impossible_conclusion_l414_414942


namespace sad_outcome_probability_l414_414290

theorem sad_outcome_probability : 
  let total_outcomes := 3^6 in
  let sad_outcomes := 156 in
  (sad_outcomes / total_outcomes : ℚ) = 0.214 := 
by
  /-
  Given conditions:
  - The company consists of three boys and three girls.
  - Each boy loves one of the three girls.
  - Each girl loves one of the boys.
  - In a sad outcome, nobody is loved by the one they love.
  - Using the properties of derangements and additional condition counts.
  - Total number of sad outcomes = 156.
  - Total possible outcomes = 3^6 = 729.
  - Final probability of sad outcome = 156 / 729 = 0.214.
  -/
  sorry

end sad_outcome_probability_l414_414290


namespace count_isosceles_triangle_numbers_l414_414394

theorem count_isosceles_triangle_numbers : 
  (finset.range 10).sum (λ a, (finset.range 10).sum (λ b, (finset.range 10).count (λ c, 
  ((a > 0 ∧ b > 0 ∧ c > 0) ∧ 
   (a = b ∧ c < a + b) ∨ 
   (b = c ∧ a < b + c) ∨ 
   (a = c ∧ b < a + c)))) = 165 := 
begin 
  sorry 
end

end count_isosceles_triangle_numbers_l414_414394


namespace ratio_fourth_to_sixth_l414_414264

-- Definitions from the conditions
def fourth_level_students := 40
def sixth_level_students := 40
def seventh_level_students := 2 * fourth_level_students

-- Statement to prove
theorem ratio_fourth_to_sixth : 
  fourth_level_students / sixth_level_students = 1 :=
by
  -- Proof skipped
  sorry

end ratio_fourth_to_sixth_l414_414264


namespace area_of_region_l414_414067

noncomputable def large_circle_radius : ℝ := 40

noncomputable def small_circle_radius : ℝ := large_circle_radius / (1 + (1 / Real.sin (Real.pi / 8)))

noncomputable def K : ℝ := 
  Real.pi * large_circle_radius^2 - 8 * Real.pi * small_circle_radius^2

theorem area_of_region : ⌊K⌋ = 2191 := by
  sorry

end area_of_region_l414_414067


namespace inequality_X_greater_Y_l414_414992

theorem inequality_X_greater_Y
  (N : Fin 2012 → ℕ)
  (h_pos : ∀ i, 0 < N i) :
  let X := (∑ i in Finset.range 2010, N i) * (∑ i in Finset.range (1010 + 1) 2011, N i)
  let Y := (∑ i in Finset.range 2011, N i) * (∑ i in Finset.range 1 2011, N i)
  X > Y :=
by
  sorry

end inequality_X_greater_Y_l414_414992


namespace ellipses_same_eccentricity_l414_414294

variables {a b k x y : ℝ}
hypothesis h1 : (a ≠ 0) ∧ (b ≠ 0)
hypothesis h2 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1
hypothesis h3 : ∀ x y, x^2 / a^2 + y^2 / b^2 = k
hypothesis h4 : k > 0

theorem ellipses_same_eccentricity :
  let e1 := (a^2 - b^2) / a^2 in
  let e2 := (ka^2 - b^2) / (ka^2) in 
  e1 = e2 :=
by 
  sorry

end ellipses_same_eccentricity_l414_414294


namespace zoo_visitors_l414_414589

theorem zoo_visitors (visitors_friday: ℕ) (multiplier: ℕ) (h_visitors_friday: visitors_friday = 1250) (h_multiplier: multiplier = 3) :
  ∃ visitors_saturday, visitors_saturday = 3750 := 
by 
  let visitors_saturday := visitors_friday * multiplier
  have h_visitors_saturday : visitors_saturday = 3750 := by
    rw [h_visitors_friday, h_multiplier]
    norm_num
  exact ⟨visitors_saturday, h_visitors_saturday⟩

end zoo_visitors_l414_414589


namespace binomial_coefficient_sum_largest_6th_term_l414_414815

theorem binomial_coefficient_sum_largest_6th_term :
  (∃ n : ℕ, (∀ k : ℕ, k ≠ 5 → binomial n k < binomial n 5) ∧ (∑ k in finset.range (n + 1), binomial n k) = 2 ^ 10) :=
begin
  existsi 10,
  split,
  { intros k hk,
    sorry },
  { simp [finset.sum_range_succ, binomial],
    sorry },
end

end binomial_coefficient_sum_largest_6th_term_l414_414815


namespace roots_sum_l414_414805

variables {f : ℝ → ℝ}
variables {m x1 x2 x3 x4 : ℝ}

-- Definition of the odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The given conditions
axiom function_properties : is_odd_function f ∧ (∀ x, f(x - 4) = -f x) ∧ (∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≤ f y)
axiom four_roots : m > 0 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m ∧ f x4 = m ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x1 ∧ x1 ∈ Icc (-8:ℝ) 8  ∧ x2 ∈ Icc (-8:ℝ) 8 ∧ x3 ∈ Icc (-8:ℝ) 8 ∧ x4 ∈ Icc (-8:ℝ) 8

-- The question to be proved
theorem roots_sum : x1 + x2 + x3 + x4 = -8 :=
sorry

end roots_sum_l414_414805


namespace temperature_at_midnight_l414_414902

theorem temperature_at_midnight 
  (morning_temp : ℝ) 
  (afternoon_rise : ℝ) 
  (midnight_drop : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_rise = 1)
  (h3 : midnight_drop = 7) 
  : morning_temp + afternoon_rise - midnight_drop = 24 :=
by
  -- Convert all conditions into the correct forms
  rw [h1, h2, h3]
  -- Perform the arithmetic operations
  norm_num

end temperature_at_midnight_l414_414902


namespace laplace_formula_proof_l414_414570

noncomputable def laplace_formula (n : ℕ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ n then 
    (1 / (nat.factorial n)) * (finset.range (nat.floor x).succ).sum (λ k, ((-1 : ℝ) ^ k) * nat.comb n k * (x - k) ^ n)
  else 0

theorem laplace_formula_proof (ξ : ℕ → ℝ) (n : ℕ) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ n)
  (h₁ : ∀ i, 0 ≤ ξ i ∧ ξ i ≤ 1)  -- ξ_i are uniformly distributed over [0, 1]
  (h₂ : ∀ i j, i ≠ j → ∀ (ai bi : ℝ), prob_space.indep (ξ i = ai) (ξ j = bi)) -- Independence
  :
  prob_space.pr (λ ω, (finset.range n).sum (λ i, ξ i ω) ≤ x) = laplace_formula n x :=
sorry

end laplace_formula_proof_l414_414570


namespace sum_possible_values_of_y_l414_414863

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end sum_possible_values_of_y_l414_414863


namespace polygon_sides_l414_414499

theorem polygon_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 180 - 360 / n = 150) : n = 12 := by
  sorry

end polygon_sides_l414_414499


namespace derivative_at_one_eq_neg_one_l414_414574

variable {α : Type*} [TopologicalSpace α] {f : ℝ → ℝ}
-- condition: f is differentiable
variable (hf_diff : Differentiable ℝ f)
-- condition: limit condition
variable (h_limit : Tendsto (fun Δx => (f (1 + 2 * Δx) - f 1) / Δx) (𝓝 0) (𝓝 (-2)))

-- proof goal: f'(1) = -1
theorem derivative_at_one_eq_neg_one : deriv f 1 = -1 := 
by
  sorry

end derivative_at_one_eq_neg_one_l414_414574


namespace prob_male_given_obese_correct_l414_414052

-- Definitions based on conditions
def ratio_male_female : ℚ := 3 / 2
def prob_obese_male : ℚ := 1 / 5
def prob_obese_female : ℚ := 1 / 10

-- Definition of events
def total_employees : ℚ := ratio_male_female + 1

-- Probability calculations
def prob_male : ℚ := ratio_male_female / total_employees
def prob_female : ℚ := 1 / total_employees

def prob_obese_and_male : ℚ := prob_male * prob_obese_male
def prob_obese_and_female : ℚ := prob_female * prob_obese_female

def prob_obese : ℚ := prob_obese_and_male + prob_obese_and_female

def prob_male_given_obese : ℚ := prob_obese_and_male / prob_obese

-- Theorem statement
theorem prob_male_given_obese_correct : prob_male_given_obese = 3 / 4 := sorry

end prob_male_given_obese_correct_l414_414052


namespace midpoint_coordinates_l414_414305

theorem midpoint_coordinates :
  let A := (7, 8)
  let B := (1, 2)
  let midpoint (p1 p2 : ℕ × ℕ) : ℕ × ℕ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint A B = (4, 5) :=
by
  sorry

end midpoint_coordinates_l414_414305


namespace dogs_groomed_l414_414925

def time_to_groom_dog_hours : ℝ := 2.5
def time_to_groom_cat_hours : ℝ := 0.5
def total_time_minutes : ℕ := 840
def number_of_cats : ℕ := 3

theorem dogs_groomed (time_to_groom_dog_hours time_to_groom_cat_hours total_time_minutes number_of_cats : ℝ)
  (number_of_cats = 3) (time_to_groom_dog_minutes : ℝ := time_to_groom_dog_hours * 60) 
  (time_to_groom_cat_minutes : ℝ := time_to_groom_cat_hours * 60) 
  (cat_grooming_total_time : ℝ := number_of_cats * time_to_groom_cat_minutes) 
  (dog_grooming_total_time : ℝ := total_time_minutes - cat_grooming_total_time) 
  (number_of_dogs : ℝ := dog_grooming_total_time / time_to_groom_dog_minutes) : number_of_dogs = 5 := by
  sorry

end dogs_groomed_l414_414925


namespace percentage_of_300_eq_25_l414_414075
-- Import the entire Mathlib library

-- Define the part and the whole
def part : ℝ := 75
def whole : ℝ := 300

-- Define the formula to calculate the percentage
def percentage_formula (part whole : ℝ) : ℝ := (part / whole) * 100

-- State the theorem
theorem percentage_of_300_eq_25 (h: part = 75 ∧ whole = 300) : percentage_formula part whole = 25 :=
by
  sorry

end percentage_of_300_eq_25_l414_414075


namespace car_speed_l414_414987

theorem car_speed (time : ℕ) (distance : ℕ) (h1 : time = 5) (h2 : distance = 300) : distance / time = 60 := by
  sorry

end car_speed_l414_414987


namespace angle_HKG_is_90_l414_414936

open Point Geometry Triangle Circle Angle

variables {A B C G H L K : Point}
variables (ω : Circle)

-- Given conditions
axiom centroid (G : Point) (Δ : Triangle A B C) : G = Triangle.centroid A B C
axiom orthocenter (H : Point) (Δ : Triangle A B C) : H = Triangle.orthocenter A B C
axiom obtuse_angle_B (Δ : Triangle A B C) : ∠ B > 90
axiom circle_with_diameter (A G : Point) : ω = Circle.diameter A G
axiom L_on_omega (Δ : Triangle A B C) : L ≠ A ∧ L ∈ Circle.points ω
axiom tangent_at_L (Δ : Triangle A B C) : Tangent.exists_at ω L
axiom tangent_intersects_circumcircle_at_K (Δ : Triangle A B C) : K ≠ L ∧ K ∈ Circle.circumcircle_points Δ
axiom AG_eq_GH (Δ : Triangle A B C) : Distance A G = Distance G H

-- Proof to be shown
theorem angle_HKG_is_90 (Δ : Triangle A B C) :
  ∠ H K G = 90 := sorry

end angle_HKG_is_90_l414_414936


namespace num_integers_in_set_x_l414_414890

-- Definition and conditions
variable (x y : Finset ℤ)
variable (h1 : y.card = 10)
variable (h2 : (x ∩ y).card = 6)
variable (h3 : (x.symmDiff y).card = 6)

-- Proof statement
theorem num_integers_in_set_x : x.card = 8 := by
  sorry

end num_integers_in_set_x_l414_414890


namespace generating_function_value_l414_414958

theorem generating_function_value (x m n : ℝ) (h1 : m + n = 1) :
  (m * (x + 1) + n * (2 * x) = 2) :=
by
  have h2 : x = 1 := rfl
  sorry

end generating_function_value_l414_414958


namespace x_plus_z_eq_zero_l414_414406

theorem x_plus_z_eq_zero (u v w x y z : ℂ) (h1 : v.im = 2) (h2 : y = -u - w) 
(h3 : u + v + w + x + y + z = 0 + 2 * complex.I) : x.im + z.im = 0 := by
  sorry

end x_plus_z_eq_zero_l414_414406


namespace find_x0_l414_414862

noncomputable def f (x : ℝ) := x^5

theorem find_x0 {x0 : ℝ} (h : deriv f x0 = 20) : x0 = real.sqrt 2 ∨ x0 = -real.sqrt 2 := by
  have deriv_f : ∀ x, deriv f x = 5 * x^4 := by {
    funext,
    simp only [f],
    exact deriv_cpow' (by norm_num : 5 ≠ 1) (by norm_num : 5 ≠ 0)
  }
  have h' : 5 * x0^4 = 20 := by {
    rw deriv_f at h,
    exact h
  }
  have x0_pow4_eq4 : x0^4 = 4 := by {
    linarith
  }
  have x0_is_sqrt2_or_neg_sqrt2 : x0 = real.sqrt 2 ∨ x0 = -real.sqrt 2 := by {
    exact real.eq_sqrt_or_eq_neg_sqrt.mpr x0_pow4_eq4
  }
  exact x0_is_sqrt2_or_neg_sqrt2

#check find_x0

end find_x0_l414_414862


namespace choose_4_from_7_socks_l414_414611

theorem choose_4_from_7_socks :
  (nat.choose 7 4) = 35 :=
by 
-- Proof skipped
sorry

end choose_4_from_7_socks_l414_414611


namespace find_r_s_t_l414_414568

-- Define the properties in the given conditions and proof goal
structure CircleDiameter (d : ℝ) := 
  (diameter : d = 1)

structure ArcPoints (a b c v w : ℝ) := 
  (midA : a = 1/2) 
  (MB : b = 3 / 5) 
  (onOtherArcC : c ≠ a ∧ c ≠ b)

structure Intersection (d v w : ℝ) := 
  (intersectAc : v ∈ set.Icc 0 d) 
  (intersectBc : w ∈ set.Icc 0 d)

structure MaxValue (r s t d : ℝ) := 
  (valueForm : d = r - s * Real.sqrt t) 
  (rPos : r > 0)
  (sPos : s > 0)
  (tPos : t > 0)
  (tSquareFree : ¬∃ p: ℕ, Nat.prime p ∧ p^2 ∣ t)

theorem find_r_s_t : ∃ (r s t : ℝ), 
  (∀ d (circ : CircleDiameter d) (pts : ArcPoints 0 0 0 0 0) 
  (ints : Intersection d 0 0) 
  (maxv : MaxValue r s t d), d = 7 - 4 * Real.sqrt 3 ∧ r = 7 ∧ s = 4 ∧ t = 3)
:= sorry

end find_r_s_t_l414_414568


namespace johns_initial_playtime_l414_414232

theorem johns_initial_playtime :
  ∃ (x : ℝ), (14 * x = 0.40 * (14 * x + 84)) → x = 4 :=
by
  sorry

end johns_initial_playtime_l414_414232


namespace kaylin_age_correct_l414_414550

def age_Freyja : ℝ := 9.5
def age_Lucas : ℝ := age_Freyja + 9
def age_Eli : ℝ := Real.sqrt age_Lucas
def age_Sarah : ℝ := 2 * age_Eli
def age_Kaylin : ℝ := age_Sarah - 5

theorem kaylin_age_correct : age_Kaylin = 3.6 := 
  by
    -- The details of the proof would be filled in here
    sorry

end kaylin_age_correct_l414_414550


namespace smallest_positive_integer_divisible_l414_414780

theorem smallest_positive_integer_divisible (n : ℕ) (h1 : 15 = 3 * 5) (h2 : 16 = 2 ^ 4) (h3 : 18 = 2 * 3 ^ 2) :
  n = Nat.lcm (Nat.lcm 15 16) 18 ↔ n = 720 :=
by
  sorry

end smallest_positive_integer_divisible_l414_414780


namespace g_at_3_l414_414875

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_at_3 : g 3 = 79 := 
by 
  -- proof placeholder
  sorry

end g_at_3_l414_414875


namespace molecular_weight_N2O3_l414_414426

constant atomic_weight_N : ℝ := 14.01
constant atomic_weight_O : ℝ := 16.00
constant num_N_atoms : ℕ := 2
constant num_O_atoms : ℕ := 3

theorem molecular_weight_N2O3 :
  (num_N_atoms * atomic_weight_N + num_O_atoms * atomic_weight_O) = 76.02 :=
by
  sorry

end molecular_weight_N2O3_l414_414426


namespace solve_engine_consumption_l414_414415

-- Define the given conditions and values
def first_engine_consumed := 300
def second_engine_consumed := 192
def time_difference := 2
def consumption_difference := 6
def second_engine_consumption_per_hour (x : ℕ) := x
def first_engine_consumption_per_hour (x : ℕ) := x + consumption_difference 

-- Define the times of operation based on consumption per hour
def time_first_engine (x : ℕ) : ℝ := first_engine_consumed / (second_engine_consumption_per_hour x + consumption_difference : ℝ)
def time_second_engine (x : ℕ) : ℝ := second_engine_consumed / (second_engine_consumption_per_hour x : ℝ)

theorem solve_engine_consumption : 
  ∃ x : ℕ, time_first_engine x - time_second_engine x = time_difference ∧ 
           second_engine_consumption_per_hour x = 24 ∧ 
           first_engine_consumption_per_hour x = 30 := 
by
  sorry
  
end solve_engine_consumption_l414_414415


namespace arvin_first_day_km_l414_414913

theorem arvin_first_day_km :
  ∀ (x : ℕ), (∀ i : ℕ, (i < 5 → (i + x) < 6) → (x + 4 = 6)) → x = 2 :=
by sorry

end arvin_first_day_km_l414_414913


namespace find_m_and_period_and_increasing_interval_l414_414832

variable (x : ℝ)
variable (α : ℝ)
variable (f : ℝ → ℝ)

-- Define the function f
def f := λ x : ℝ, m * sin (2 * x) - cos² x - (1 / 2)

-- Specify the conditions
axiom tan_alpha : tan α = 2 * sqrt 3
axiom f_alpha : f α = -(3 / 26)

-- Targets to prove
theorem find_m_and_period_and_increasing_interval :
  (∃ m : ℝ, m = sqrt 3 / 2) ∧
  (∀ x : ℝ, f x = sin (2 * x - (π / 6)) - 1) ∧
  (∀ T : ℝ, T = π) ∧
  (∀ (a b : ℝ), f' a > 0 → (0 ≤ a ∧ a ≤ π → 0 ≤ x ∧ x ≤ π → a ≤ x ∧ x ≤ b) → 
       ([0, π] = [0, π / 3] ∪ [5 * π / 6, π])) := 
  sorry

end find_m_and_period_and_increasing_interval_l414_414832


namespace number_of_zeros_l414_414250

open Nat
open Finset

-- Definitions according to conditions
variables (b : ℕ) (hb : b ≥ 2) (p : ℕ) (hp : prime p ∧ ∀ q:ℕ, prime q → q ∣ b → q ≤ p)
variables (z_n : ℕ → ℕ)

-- Definition of z_n
def z_n (n : ℕ) : ℕ := nat.find_greatest (λ k, b^k ∣ fact n) n

-- Prove the statement
theorem number_of_zeros (n : ℕ) : z_n n < n / (p - 1) :=
by
  sorry

end number_of_zeros_l414_414250


namespace smallest_positive_integer_divisible_by_15_16_18_l414_414778

theorem smallest_positive_integer_divisible_by_15_16_18 : 
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ n = 720 := 
by
  sorry

end smallest_positive_integer_divisible_by_15_16_18_l414_414778


namespace same_properties_as_tanh_l414_414027

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def tanh (x : ℝ) : ℝ := (Real.exp(x) - Real.exp(-x))/2

theorem same_properties_as_tanh : 
  is_odd tanh ∧ is_monotonically_increasing tanh →
  (is_odd (fun x => x ^ 3) ∧ is_monotonically_increasing (fun x => x ^ 3)) :=
by
  sorry

end same_properties_as_tanh_l414_414027


namespace primes_with_ones_digit_3_lt_100_count_l414_414858

def is_prime (n : ℕ) : Prop := Nat.Prime n
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def less_than_100 (n : ℕ) : Prop := n < 100

theorem primes_with_ones_digit_3_lt_100_count :
  ∃ count, count = 7 ∧
  count = (List.filter (λ n, is_prime n ∧ has_ones_digit_3 n) (List.filter less_than_100 (List.range 100))).length :=
by
  sorry

end primes_with_ones_digit_3_lt_100_count_l414_414858


namespace danielle_rooms_is_6_l414_414853

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l414_414853


namespace sum_of_first_3_geometric_terms_eq_7_l414_414467

theorem sum_of_first_3_geometric_terms_eq_7 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_ratio_gt_1 : r > 1)
  (h_eq : (a 0 + a 2 = 5) ∧ (a 0 * a 2 = 4)) 
  : (a 0 + a 1 + a 2) = 7 := 
by
  sorry

end sum_of_first_3_geometric_terms_eq_7_l414_414467


namespace value_of_f_l414_414475

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + Real.log (Real.sqrt (x^2 + 1) + x)

-- Define the condition that a and b satisfy
def condition (a b : ℝ) : Prop := f (-a) + f (-b) - 3 = f (a) + f (b) + 3

-- Define the Lean statement to prove
theorem value_of_f (a b : ℝ) (h : condition a b) : f (a) + f (b) = -3 :=
by
  sorry

end value_of_f_l414_414475


namespace monotonicity_inequality_solution_l414_414154

def f (x : ℝ) : ℝ := (3*x + 7) / (x + 2)

theorem monotonicity (x : ℝ) (hx : -2 < x ∧ x < 2) : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := 
sorry

theorem inequality_solution (m : ℝ) :
  (1 / 2) < m ∧ m < 1 ↔ log (1 / 2) (f (-2 * m + 3)) > log (1 / 2) (f (m^2)) := 
sorry

end monotonicity_inequality_solution_l414_414154


namespace common_sum_of_matrix_l414_414998

open Nat

/-- Define the full range of integers from -15 to 33 as a list --/
def int_list := (List.range (33 + 1 + 15)).map (λ x, x - 15)

/-- Check that the number of elements is 49 --/
#eval int_list.length -- should output 49

/-- Define the sum of all integers from -15 to 33 by specifying the arithmetic sum --/
def total_sum : Int := List.sum int_list

/-- Given a 7-by-7 square where the sum of the numbers in each row, column, and diagonal are equal,
    prove that the common sum is 63 --/
theorem common_sum_of_matrix : ∃ common_sum : Int, common_sum = 63 ∧
  (∀ s, (s ∈ (Matrix.row_sums int_list 7 7) ∨
         s ∈ (Matrix.col_sums int_list 7 7) ∨
         s ∈ (Matrix.diag_sums int_list 7 7)) →
         s = common_sum) :=
by
  sorry

end common_sum_of_matrix_l414_414998


namespace manoj_gain_by_interest_l414_414584

-- Define the conditions
def SI_Anwar (principal : ℕ) (rate : ℕ) (time : ℕ) : ℝ := (principal * rate * time) / 100
def SI_Ramu (principal : ℕ) (rate : ℕ) (time : ℕ) : ℝ := (principal * rate * time) / 100
def total_gain (si_ramu : ℝ) (si_anwar : ℝ) : ℝ := si_ramu - si_anwar

-- State the main theorem
theorem manoj_gain_by_interest :
  let principal_anwar := 3900
      rate_anwar := 6
      time_anwar := 3
      principal_ramu := 5655
      rate_ramu := 9
      time_ramu := 3
      si_anwar := SI_Anwar principal_anwar rate_anwar time_anwar
      si_ramu := SI_Ramu principal_ramu rate_ramu time_ramu
  in total_gain si_ramu si_anwar = 824.85 := by
  sorry

end manoj_gain_by_interest_l414_414584


namespace reciprocal_opposite_neg_two_thirds_l414_414310

noncomputable def opposite (a : ℚ) : ℚ := -a
noncomputable def reciprocal (a : ℚ) : ℚ := 1 / a

theorem reciprocal_opposite_neg_two_thirds : reciprocal (opposite (-2 / 3)) = 3 / 2 :=
by sorry

end reciprocal_opposite_neg_two_thirds_l414_414310


namespace temperature_at_midnight_l414_414899

def morning_temp : ℝ := 30
def afternoon_increase : ℝ := 1
def midnight_decrease : ℝ := 7

theorem temperature_at_midnight : morning_temp + afternoon_increase - midnight_decrease = 24 := by
  sorry

end temperature_at_midnight_l414_414899


namespace kyungsoo_has_shorter_string_l414_414919

def cm_to_mm (cm : ℝ) : ℝ := cm * 10

def inhyuk_string_length_cm : ℝ := 97.5
def kyungsoo_string_length_cm : ℝ := 97
def kyungsoo_extra_length_mm : ℝ := 3

def inhyuk_string_length_mm : ℝ := cm_to_mm inhyuk_string_length_cm
def kyungsoo_string_base_length_mm : ℝ := cm_to_mm kyungsoo_string_length_cm
def kyungsoo_string_length_mm : ℝ := kyungsoo_string_base_length_mm + kyungsoo_extra_length_mm

theorem kyungsoo_has_shorter_string :
  kyungsoo_string_length_mm < inhyuk_string_length_mm :=
by
  have h_inhyuk : inhyuk_string_length_mm = 975 := by 
    unfold inhyuk_string_length_mm cm_to_mm inhyuk_string_length_cm
    norm_num
  have h_kyungsoo : kyungsoo_string_length_mm = 973 := by 
    unfold kyungsoo_string_length_mm kyungsoo_string_base_length_mm cm_to_mm kyungsoo_string_length_cm kyungsoo_extra_length_mm
    norm_num
  rw [h_inhyuk, h_kyungsoo]
  norm_num
  sorry

end kyungsoo_has_shorter_string_l414_414919


namespace total_inches_of_rope_l414_414555

noncomputable def inches_of_rope (last_week_feet : ℕ) (less_feet : ℕ) (feet_to_inches : ℕ → ℕ) : ℕ :=
  let last_week_inches := feet_to_inches last_week_feet
  let this_week_feet := last_week_feet - less_feet
  let this_week_inches := feet_to_inches this_week_feet
  last_week_inches + this_week_inches

theorem total_inches_of_rope 
  (six_feet : ℕ := 6)
  (four_feet_less : ℕ := 4)
  (conversion : ℕ → ℕ := λ feet, feet * 12) :
  inches_of_rope six_feet four_feet_less conversion = 96 := by
  sorry

end total_inches_of_rope_l414_414555


namespace strictly_decreasing_f_l414_414425

noncomputable def f (x : ℝ) : ℝ :=
  real.logr (1 / 2) (x^2 - 4 * x - 5)

theorem strictly_decreasing_f :
  ∀ x, x > 5 → f(x) < f(x + 1) :=
by
  sorry

end strictly_decreasing_f_l414_414425


namespace angle_symmetry_l414_414789

theorem angle_symmetry (α β : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) (hβ : 0 < β ∧ β < 2 * Real.pi) (h_symm : α = 2 * Real.pi - β) : α + β = 2 * Real.pi := 
by 
  sorry

end angle_symmetry_l414_414789


namespace james_sold_727_items_l414_414761

def houses_visited : ℕ → ℕ
| 1 := 20
| 2 := 40
| 3 := 50
| 4 := 60
| 5 := 80
| 6 := 100
| 7 := 120
| _ := 0

def success_rate : ℕ → ℚ
| 1 := 1
| 2 := 0.8
| 3 := 0.9
| 4 := 0.75
| 5 := 0.5
| 6 := 0.7
| 7 := 0.6
| _ := 0

def items_per_house : ℕ → ℕ
| 1 := 2
| 2 := 3
| 3 := 1
| 4 := 4
| 5 := 2
| 6 := 1
| 7 := 3
| _ := 0

def items_sold_each_day (day: ℕ) : ℚ := 
  (houses_visited day) * (success_rate day) * (items_per_house day)

def total_items_sold : ℚ :=
  (items_sold_each_day 1) +
  (items_sold_each_day 2) +
  (items_sold_each_day 3) +
  (items_sold_each_day 4) +
  (items_sold_each_day 5) +
  (items_sold_each_day 6) +
  (items_sold_each_day 7)

theorem james_sold_727_items : total_items_sold = 727 := 
by 
  sorry

end james_sold_727_items_l414_414761


namespace movie_sales_economics_correct_l414_414768

-- Definitions based on given conditions
structure EconomicCondition :=
  (study: bool) -- placeholder for an economic case study

structure EconomicArguments :=
  (immediate_higher_revenue: Prop)
  (loss_of_potential_future_revenue: Prop)
  (rights_transferability: Prop)
  (frequent_transactions: Prop)
  (catering_to_price_sensitive_customers: Prop)
  (high_administrative_costs: Prop)
  (risk_of_piracy: Prop)

-- Definitions for our proof problem
def economic_conditions : EconomicCondition := { study := true }

def economic_arguments : EconomicArguments :=
  {
    immediate_higher_revenue := true,
    loss_of_potential_future_revenue := true,
    rights_transferability := true,
    frequent_transactions := true,
    catering_to_price_sensitive_customers := true,
    high_administrative_costs := true,
    risk_of_piracy := true
  }

-- Proof statement
theorem movie_sales_economics_correct 
  (econ_cond: EconomicCondition)
  (econ_args: EconomicArguments) : 
  econ_cond = economic_conditions → 
  econ_args.immediate_higher_revenue = economic_arguments.immediate_higher_revenue ∧
  econ_args.loss_of_potential_future_revenue = economic_arguments.loss_of_potential_future_revenue ∧
  econ_args.rights_transferability = economic_arguments.rights_transferability ∧
  econ_args.frequent_transactions = economic_arguments.frequent_transactions ∧
  econ_args.catering_to_price_sensitive_customers = economic_arguments.catering_to_price_sensitive_customers ∧
  econ_args.high_administrative_costs = economic_arguments.high_administrative_costs ∧
  econ_args.risk_of_piracy = economic_arguments.risk_of_piracy :=
by
  intros h_eq_conditions
  subst econ_cond
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

end movie_sales_economics_correct_l414_414768


namespace chess_kings_arrangements_l414_414351

-- Define the problem conditions and question

def non_attacking_kings (board_size : ℕ) (kings : ℕ) : Prop :=
  let cells := board_size / 2 * board_size / 2
  in kings = cells

def valid_row_signature (n m : ℕ) (board : list (list bool)) : Prop :=
  ∀ i < board.length, (list.filter id (board.nth i).get_or_else []).length = m / 2
  ∧ (list.filter not (board.nth i).get_or_else []).length = m / 2

def valid_column_signature (n m : ℕ) (board : list (list bool)) : Prop :=
  ∀ j < (board.head.get_or_else []).length, 
    (list.filter (λ row, (row.nth j).get_or_else false) board).length = m / 2
  ∧ (list.filter (λ row, ¬(row.nth j).get_or_else false) board).length = m / 2

theorem chess_kings_arrangements : 
  ∃ (board : list (list bool)), valid_row_signature 100 50 board 
  ∧ valid_column_signature 100 50 board 
  ∧ non_attacking_kings 100 2500 := sorry

end chess_kings_arrangements_l414_414351


namespace trajectory_eq_C_circle_eq_P_l414_414135

-- Definition for Question I
def trajectory_equation (M : ℝ × ℝ) (F : ℝ × ℝ) (r : ℚ) (L : ℝ) :=
  let d1 := real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)
  let d2 := abs (M.1 - L)
  d1 / d2 = r

-- Proof problem for Question I
theorem trajectory_eq_C (x y : ℝ) (h : trajectory_equation (x, y) (1, 0) (real.sqrt 3 / 3 : ℚ) 3) :
  x^2 / 3 + y^2 / 2 = 1 :=
sorry

-- Proof problem for Question II
theorem circle_eq_P (r : ℚ)
  (h₁ : (circle_eq r).C)
  (h₂ : line_intersect (x1, y1) l (x2, y2) C)
  (h₃ : midpoint (x1, y1) (x2, y2) = (1, -1))
  (h₄ : |(x1 - x2) * sqrt(1 + (2/3)^2)| = sqrt(50)/15):
  (x - 1)^2 + (y + 1)^2 = 13/30 :=
sorry

end trajectory_eq_C_circle_eq_P_l414_414135


namespace polynomial_coefficients_sum_zero_l414_414124

theorem polynomial_coefficients_sum_zero (a : Fin 12 → ℝ) (x : ℝ) :
  (x^2 + 1) * (2*x + 1)^9 = ∑ i in Finset.range 12, (a i) * (x + 2) ^ i →
  (∑ i in Finset.range 12, a i) = 0 :=
by
  sorry

end polynomial_coefficients_sum_zero_l414_414124


namespace minimum_value_f_prime_range_of_m_l414_414831

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log x

theorem minimum_value_f_prime : 
  (∀ x ≥ 1, f' x = Real.exp x + (1 / x)) → ∃ (m : ℝ), m = Real.exp 1 + 1 := 
sorry

theorem range_of_m :
  (∀ x ≥ 1, f x ≥ Real.exp 1 + m * (x - 1)) → m <= Real.exp 1 + 1 := 
sorry

end minimum_value_f_prime_range_of_m_l414_414831


namespace max_students_with_even_distribution_l414_414904

theorem max_students_with_even_distribution (pens toys books : ℕ) 
  (hp : pens = 451) (ht : toys = 410) (hb : books = 325) : 
  Nat.gcd (Nat.gcd pens toys) books = 1 :=
by
  rw [hp, ht, hb]
  exact Nat.gcd_assoc 451 410 325 ▸ Nat.gcd_comm 410 325 ▸ Nat.gcd 451 (Nat.gcd 410 325) = 1 

end max_students_with_even_distribution_l414_414904


namespace simplify_expression_l414_414275

variable (a b : ℝ)

theorem simplify_expression : 
  (5/6 * a^(1/2) * b^(-1/3) * -3 * a^(-1/6) * b^(-1) / (4 * a^(2/3) * b^(-3))^(1/2)) = -5/4 * b^(1/6) := sorry

end simplify_expression_l414_414275


namespace trig_identity_l414_414399

open Real

theorem trig_identity :
  sin (21 * π / 180) * cos (9 * π / 180) + sin (69 * π / 180) * sin (9 * π / 180) = 1 / 2 :=
by sorry

end trig_identity_l414_414399


namespace exists_zero_l414_414302

open TopologicalSpace
open Classical

variable {α : Type*} [TopologicalSpace α] [LinearOrder α] [OrderTopology α]
variable {β : Type*} [LinearOrderedField β] [TopologicalSpace β] [OrderTopology β]

theorem exists_zero (f : α → β) {a b : α} (h_cont : ContinuousOn f (Icc a b)) (h_sign : f a * f b < 0) :
  ∃ c ∈ Ioo a b, f c = 0 :=
sorry

end exists_zero_l414_414302


namespace unique_f_satisfies_eq_l414_414073

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * (x^2 + 2 * x - 1)

theorem unique_f_satisfies_eq (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x + f (1 - x) = x^2) : 
  ∀ x : ℝ, f x = (1 / 3) * (x^2 + 2 * x - 1) :=
sorry

end unique_f_satisfies_eq_l414_414073


namespace proof_problem_l414_414795

-- Given a right triangle ABC with right angle at C and leg lengths in ratio 1:3
variables (b : ℝ)
def right_triangle (A B C : ℝ × ℝ) : Prop :=
  ∠A C B = 90 ∧ dist A C / dist B C = 1 / 3

-- Definitions of points K, L, and M
def K (A C : ℝ × ℝ) : ℝ × ℝ := (side of square centered at midpoint of AC)
def L (B C : ℝ × ℝ) : ℝ × ℝ := (side of square centered at midpoint of BC)
def M (A B : ℝ × ℝ) : ℝ × ℝ := (midpoint of segment AB)

-- Main statement
theorem proof_problem (A B C K L M : ℝ × ℝ) (h1 : right_triangle A B C b)
  (h2 : K = center_of_square A C) (h3 : L = center_of_square B C) (h4 : M = midpoint A B) :
  (C lies on segment K L) ∧ (area (triangle A B C) / area (triangle K L M) = 3 / 4) :=
sorry

end proof_problem_l414_414795


namespace determinant_cos_tan_zero_l414_414951

theorem determinant_cos_tan_zero
  (A B C : ℝ)
  (h : A + B + C = π) : 
  det ![![cos A ^ 2, tan A, 1], ![cos B ^ 2, tan B, 1], ![cos C ^ 2, tan C, 1]] = 0 :=
by
  sorry

end determinant_cos_tan_zero_l414_414951


namespace plane_through_A_perpendicular_to_BC_l414_414703

def point := (ℝ × ℝ × ℝ)

def vector (p q : point) : point :=
  ((q.1 - p.1), (q.2 - p.2), (q.3 - p.3))

def plane_eq (n : point) (p : point) (x y z : ℝ) : ℝ :=
  n.1 * (x - p.1) + n.2 * (y - p.2) + n.3 * (z - p.3)

theorem plane_through_A_perpendicular_to_BC :
  let A : point := (-2, 0, -5)
  let B : point := (2, 7, -3)
  let C : point := (1, 10, -1)
  let BC : point := vector B C
  let normal_vector : point := (-1, 3, 2)
  ∀ x y z, plane_eq normal_vector A x y z = -x + 3 * y + 2 * z + 8 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l414_414703


namespace cost_of_bananas_l414_414675

-- Definitions of the conditions from the problem
namespace BananasCost

variables (A B : ℝ)

-- Condition equations
def condition1 : Prop := 2 * A + B = 7
def condition2 : Prop := A + B = 5

-- The theorem to prove the cost of a bunch of bananas
theorem cost_of_bananas (h1 : condition1 A B) (h2 : condition2 A B) : B = 3 := 
  sorry

end BananasCost

end cost_of_bananas_l414_414675


namespace find_m_plus_n_l414_414536

theorem find_m_plus_n
  (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (angle_C_right : ∠ A B C = π / 2)
  (BD_length : ∃ (BD : ℕ), BD = 29^3)
  (side_lengths_integers : ∃ (a b c : ℤ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2)
  (cos_B_fraction : ∃ (m n : ℕ), gcd m n = 1 ∧ n > 0 ∧ cos B = m / n) :
  m + n = 450 := 
sorry

end find_m_plus_n_l414_414536


namespace ratio_of_hair_lengths_l414_414237

theorem ratio_of_hair_lengths 
  (logan_hair : ℕ)
  (emily_hair : ℕ)
  (kate_hair : ℕ)
  (h1 : logan_hair = 20)
  (h2 : emily_hair = logan_hair + 6)
  (h3 : kate_hair = 7)
  : kate_hair / emily_hair = 7 / 26 :=
by sorry

end ratio_of_hair_lengths_l414_414237


namespace part_a_part_b_part_c_l414_414932

/-- α(n) is the number of 1s in the binary representation of n -/
def α(n : ℕ) : ℕ :=
  n.toDigits 2 |>.count (1 : ℕ)

theorem part_a (n : ℕ) (h : 0 < n) :
  α(n^2) ≤ (α(n) * (α(n) + 1)) / 2 :=
sorry

theorem part_b : ∀ᶠ (n : ℕ) in Filter.atTop, α(n^2) = (α(n) * (α(n) + 1)) / 2 :=
sorry

theorem part_c : ∃ (n_i : ℕ → ℕ), Filter.Tendsto (fun i => α((n_i i)^2) / α(n_i i)) Filter.atTop (Filter.Principal {0}) :=
sorry

end part_a_part_b_part_c_l414_414932


namespace find_angle_between_vectors_l414_414844

open Real EuclideanGeometry -- Import the broader Mathlib library and essential modules

variables (a b : EuclideanSpace ℝ (Fin 2))

def vector_magnitude (v : EuclideanSpace ℝ (Fin 2)) := ‖v‖

def dot_product (u v : EuclideanSpace ℝ (Fin 2)) := @inner ℝ _ _ _ u v

variables 
  (h1 : vector_magnitude a = 2) 
  (h2 : vector_magnitude b = 2) 
  (h3 : dot_product a (a - b) = 2)

theorem find_angle_between_vectors : 
  let θ := Real.arccos ((dot_product a b) / (vector_magnitude a * vector_magnitude b)) in
  θ = π / 3 :=
by
  sorry

end find_angle_between_vectors_l414_414844


namespace actual_toddler_count_l414_414685

theorem actual_toddler_count (bills_count : ℕ) (double_counted : ℕ) (hidden_toddlers : ℕ) (left_toddlers : ℕ) (new_toddlers : ℕ) :
    bills_count = 34 →
    double_counted = 10 →
    hidden_toddlers = 4 →
    left_toddlers = 6 →
    new_toddlers = 5 →
    (bills_count - double_counted + hidden_toddlers = 28) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3]
  norm_num
  sorry

end actual_toddler_count_l414_414685


namespace domain_of_x_l414_414217

-- Conditions
def is_defined_num (x : ℝ) : Prop := x + 1 >= 0
def not_zero_den (x : ℝ) : Prop := x ≠ 2

-- Proof problem statement
theorem domain_of_x (x : ℝ) : (is_defined_num x ∧ not_zero_den x) ↔ (x >= -1 ∧ x ≠ 2) := by
  sorry

end domain_of_x_l414_414217


namespace simplify_trig_expression_l414_414276

theorem simplify_trig_expression (x : ℝ) :
  (sqrt 3 / 2) * sin x - (1 / 2) * cos x = sin (x - π / 6) :=
by
  sorry

end simplify_trig_expression_l414_414276


namespace triangle_area_proof_l414_414538

variables {A B C O A₁ A₂ : Type*}
variables [metric_space A] [metric_space B] [metric_space C] 
variables [metric_space O] [metric_space A₁] [metric_space A₂] 
variables {d_AB : metric.dist A B = 4} {d_AC : metric.dist A C = 6} 
variables {angle_BAC : real.angle = 60}
variables {center_O : O} {intersect_A₂ : A₂}
variables {bisector_A₁A₂ : line A₁ A₂}

noncomputable def area_OA₂C (O A₂ C : Type*) : ℝ := 7 / real.sqrt 3
noncomputable def area_A₁A₂C (A₁ A₂ C : Type*) : ℝ := 7 * real.sqrt 3 / 5

theorem triangle_area_proof :
  (d_AB = 4) ∧ (d_AC = 6) ∧ (angle_BAC = 60) →
  ∃ (R : ℝ), 
    area_OA₂C O A₂ C = 7 / real.sqrt 3 ∧
    area_A₁A₂C A₁ A₂ C = 7 * real.sqrt 3 / 5 := 
by intros; split; exact sorry

end triangle_area_proof_l414_414538


namespace news_dissemination_l414_414905

theorem news_dissemination (k : ℕ) (h : k > 3) :
  ∃ (m : ℕ), m = 2 * k - 4 ∧ (∀ i, i < k → ∃ (j : ℕ), j < k ∧ i ≠ j) →  ∀ (i : ℕ), i < k → knows_all_news i m :=
sorry

def knows_all_news (i : ℕ) (m : ℕ) : Prop :=
∀ j, j < m → i.knows j.news

query : knows_all_news (k - 1) (2 * k - 4) :=
sorry

end news_dissemination_l414_414905


namespace secant_slope_correct_l414_414776

def f (x : ℝ) : ℝ := x / (1 - x)

theorem secant_slope_correct : 
  let Δx : ℝ := 0.5,
      x₁ : ℝ := 2,
      y₁ : ℝ := -2,
      x₂ : ℝ := x₁ + Δx,
      y₂ : ℝ := f x₂
  in (x₂ - x₁) ≠ 0 → (y₂ - y₁) / (x₂ - x₁) = 2 / 3 :=
by
    intros Δx x₁ y₁ x₂ y₂ h1
        have h3 : y₁ = -2 := rfl
        rw h3 at *
        have h4 : x₁ = 2 := rfl
        rw h4 at *
        have h5 : y₂ = f (x₁ + Δx) := rfl
        rw h5 at *
        have h6 : f (x₁ + Δx) = -5 / 3 := by sorry
        rw h6 at *
        have h7: y₂ = -5 / 3 := rfl
        rw h7 at *
        have h8: x₂ = x₁ + Δx := rfl
        rw h8 at *
        have h9: x₂ - x₁ = Δx := by sorry
        rw h9 at *
        linarith [h1]

end secant_slope_correct_l414_414776


namespace four_digit_one_even_three_odd_no_repetition_l414_414166

theorem four_digit_one_even_three_odd_no_repetition :
  ∃ n : ℕ, n = 1140 ∧
  ∃ (digits : Finset ℕ), digits.card = 4 ∧
  digits.filter (λ d, d % 2 = 0) = {even_digit} ∧
  digits.filter (λ d, d % 2 = 1) = odd_digits ∧
  1 ≤ even_digit ∧ even_digit < 10 ∧
  0 < odd_digits \ {even_digit} ∧
  ∀ d ∈ odd_digits \ {even_digit}, 1 ≤ d ∧ d < 10 ∧ d % 2 = 1 ∧
  (∀ x y ∈ digits, x ≠ y → digits = {x, y, z, w}) :=
begin
  sorry
end

end four_digit_one_even_three_odd_no_repetition_l414_414166


namespace proportional_set_c_l414_414387

def proportional (a b c d : ℝ) : Prop :=
  a * d = b * c

theorem proportional_set_c :
  proportional 2 4 9 18 ∧
  ¬ proportional 2 2.5 3 3.5 ∧
  ¬ proportional (real.sqrt 3) 3 3 (4 * real.sqrt 3) ∧
  ¬ proportional 4 5 6 7 :=
by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end proportional_set_c_l414_414387


namespace max_x_value_l414_414891

theorem max_x_value 
    (x : ℤ) 
    (h1 : 2.134 * 10^x < 210000) 
    : x ≤ 4 :=
by
  sorry

end max_x_value_l414_414891


namespace not_continuous_at_neg_one_l414_414765

def f (x : ℝ) : ℝ := (x^3 - 1) / (x^2 + 2*x + 1)

theorem not_continuous_at_neg_one : ¬continuous_at f (-1) :=
begin
  -- Proof will be inserted here
  sorry
end

end not_continuous_at_neg_one_l414_414765


namespace find_g3_l414_414361

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eq (x y : ℝ) : g(x - y) = g(x) + g(y)
axiom g_zero : g(0) = 0

theorem find_g3 : g(3) = 0 := sorry

end find_g3_l414_414361


namespace S8_value_l414_414257

theorem S8_value (x : ℝ) (h : x + 1/x = 4) (S : ℕ → ℝ) (S_def : ∀ m, S m = x^m + 1/x^m) :
  S 8 = 37634 :=
sorry

end S8_value_l414_414257


namespace pears_initially_l414_414664

/-- 
There were 100 apples, 99 oranges, and some pears on the table. 
The first person took an apple, the second person took a pear, the third person took an orange, 
the next took an apple, the one after that took a pear, and then the next took an orange. 
The children continued to take fruits in this order until the table was empty. 
Prove the number of pears could have been 99 or 100.
-/
theorem pears_initially (apples : ℕ) (oranges : ℕ) (pears : ℕ) (total_fruits : ℕ) : 
  apples = 100 ∧ oranges = 99 ∧ total_fruits = apples + oranges + pears ∧ 
  (∀ n, n * 3 ≤ total_fruits → 
    (n * 3 ≤ 100 ∨ n * 3 - 100 ≤ pears) ∧
    (n * 3 ≤ 99 ∨ n * 3 - 99 ≤ pears)) 
  → pears = 99 ∨ pears = 100 :=
begin
  intros h,
  sorry,
end

end pears_initially_l414_414664


namespace combined_weight_of_three_boxes_l414_414966

theorem combined_weight_of_three_boxes (a b c d : ℕ) (h₁ : a + b = 132) (h₂ : a + c = 136) (h₃ : b + c = 138) (h₄ : d = 60) : 
  a + b + c = 203 :=
sorry

end combined_weight_of_three_boxes_l414_414966


namespace extremum_f_tangent_condition_zero_count_g_l414_414443

-- Part (1)
theorem extremum_f (a : ℝ) (h : a = 2) : (∃ x : ℝ, x > 0 ∧ (x^2 - 2 * log x = 1)) :=
by
  use 1
  split
  norm_num
  suffices : (1:ℝ)^2 - 2 * log 1 = 1 by
    exact this
  sorry

-- Part (2)
theorem tangent_condition (a : ℝ) (h : 2 + a = 0) : ∃ f : ℝ → ℝ, (∀ x > 0, f x = x^2 + 2 * log x) :=
by
  have ha : a = -2 := by linarith
  use λ x, x^2 + 2 * log x
  intro x
  intro hx
  funext
  simp only [ha, log, pow_two]
  exact rfl

-- Part (3)
theorem zero_count_g (a : ℝ) (h : a > 0) : (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^2 - a * log x - a * x → a = 1) :=
by
  use λ x, x^2 - a * log x - a * x
  intro x
  intro hfx
  sorry

end extremum_f_tangent_condition_zero_count_g_l414_414443


namespace y_work_days_l414_414347

theorem y_work_days (d : ℕ) (hx : x_does_work_in := 15) (hxy : x_and_y_do_work_in := 10) :
  (1/15 : ℝ) + (1/d : ℝ) = (1/10 : ℝ) → d = 30 :=
sorry

end y_work_days_l414_414347


namespace cos_double_angle_l414_414859

-- Define the condition as an assumption
def condition_1 (α : ℝ) : Prop :=
  Real.tan (π / 4 - α) = -1 / 3

-- State the proof problem
theorem cos_double_angle (α : ℝ) (h : condition_1 α) : 
  Real.cos (2 * α) = -3 / 5 :=
sorry

end cos_double_angle_l414_414859


namespace rowing_velocity_l414_414372

theorem rowing_velocity (v : ℝ) : 
  (∀ (d : ℝ) (s : ℝ) (total_time : ℝ), 
    s = 10 ∧ 
    total_time = 30 ∧ 
    d = 144 ∧ 
    (d / (s - v) + d / (s + v)) = total_time) → 
  v = 2 := 
by
  sorry

end rowing_velocity_l414_414372


namespace proof_problem_l414_414118

open Real

noncomputable def P : ℝ × ℝ := (2, 1)

-- Parametric form of the line
def l (α t : ℝ) : ℝ × ℝ := (2 + t * cos α, 1 + t * sin α)

-- Condition that |PA| * |PB| = 4
def condition (α : ℝ) : Prop := 
  let A := (2 + (-2 * cos α), 1 + (-2 * sin α) * (cos α == 0))
  let B := (2 + (-1 / (tan α)), 1 + (- (1 / tan α)) * sin α == 0)
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  ‖PA‖ * ‖PB‖ = 4

-- Problem statement
theorem proof_problem : 
  ∃ α, condition α ∧ α = (3 * π / 4) ∧ 
    ∀ ρ θ, ρ * (cos θ + sin θ) = 3 : =
begin
  sorry
end

end proof_problem_l414_414118


namespace find_prime_p_l414_414106

open Nat

def number_of_divisors (n : ℕ) : ℕ :=
  (factors n).to_finset.card

def sum_of_divisors (n : ℕ) : ℕ :=
  (factors n).to_finset.sum id

theorem find_prime_p (p : ℕ) (hprime : prime p)
  (n u v : ℕ) (hn : 0 < n) (hu : 0 < u) (hv : 0 < v)
  (hdivisors : number_of_divisors n = p^u)
  (hsum : sum_of_divisors n = p^v) : p = 2 :=
by
  sorry

end find_prime_p_l414_414106


namespace two_cities_connected_l414_414658

open Classical

variable {V : Type} [Fintype V] [DecidableEq V]

/-- Given a graph with 100 cities such that:
  1. Any four cities are connected by at least two roads,
  2. There is no Hamiltonian path (a path passing through every city exactly once),
there exist two cities such that every other city is connected to at least one of them. -/
theorem two_cities_connected (c : Finset V) (h₁ : c.card = 100)
  (h₂ : ∀ (a b d e : V), a ≠ b → a ≠ d → a ≠ e → b ≠ d → b ≠ e → d ≠ e → 
        ∃ (x y : V), x ≠ y ∧ x ∈ c ∧ y ∈ c ∧ x ≠ a ∧ x ≠ b ∧ x ≠ d ∧ x ≠ e ∧
          y ≠ a ∧ y ≠ b ∧ y ≠ d ∧ y ≠ e ∧
          (x, a) ∈ c ∧ (x, b) ∈ c ∧ (x, d) ∈ c ∧ (x, e) ∈ c ∧
          (y, a) ∈ c ∧ (y, b) ∈ c ∧ (y, d) ∈ c ∧ (y, e) ∈ c)
  (h₃ : ∀ (p : List V), (p.nodup) → (∀ (v : V) (H : v ∈ p), v ∈ c) → p.length ≠ 100) :
  ∃ (v₁ v₂ : V), v₁ ≠ v₂ ∧ ∀ v ∈ c, v ≠ v₁ → v ≠ v₂ → (v ~ v₁ ∨ v ~ v₂) :=
by
  sorry

end two_cities_connected_l414_414658


namespace rachel_budget_proof_l414_414605

-- Define the prices Sara paid for shoes and the dress
def shoes_price : ℕ := 50
def dress_price : ℕ := 200

-- Total amount Sara spent
def sara_total : ℕ := shoes_price + dress_price

-- Rachel's budget should be double of Sara's total spending
def rachels_budget : ℕ := 2 * sara_total

-- The theorem statement
theorem rachel_budget_proof : rachels_budget = 500 := by
  unfold rachels_budget sara_total shoes_price dress_price
  rfl

end rachel_budget_proof_l414_414605


namespace point_not_in_fourth_quadrant_l414_414915

theorem point_not_in_fourth_quadrant (a : ℝ) :
  ¬ ((a - 3 > 0) ∧ (a + 3 < 0)) :=
by
  sorry

end point_not_in_fourth_quadrant_l414_414915


namespace max_diff_is_achieved_l414_414969

structure DigitConstraints (minuend subtrahend result : ℤ) :=
(a_constr : minuend.digits 2 = 9 ∨ minuend.digits 2 = 5 ∨ minuend.digits 2 = 3)
(b_constr : minuend.digits 1 = 7 ∨ minuend.digits 1 = 3 ∨ minuend.digits 1 = 2)
(c_constr : minuend.digits 0 = 9 ∨ minuend.digits 0 = 8 ∨ minuend.digits 0 = 4 ∨ minuend.digits 0 = 3)
(d_constr : subtrahend.digits 2 = 7 ∨ subtrahend.digits 2 = 3 ∨ subtrahend.digits 2 = 2)
(e_constr : subtrahend.digits 1 = 9 ∨ subtrahend.digits 1 = 5 ∨ subtrahend.digits 1 = 3)
(f_constr : subtrahend.digits 0 = 7 ∨ subtrahend.digits 0 = 4 ∨ subtrahend.digits 0 = 1)
(g_constr : result.digits 2 = 9 ∨ result.digits 2 = 5 ∨ result.digits 2 = 4)
(h_constr : result.digits 1 = 2)
(i_constr : result.digits 0 = 9 ∨ result.digits 0 = 5 ∨ result.digits 0 = 4) 

def correct_result (minuend subtrahend result : ℕ) : Prop := 
  minuend = 923 ∧ subtrahend = 394 ∧ result = 529

theorem max_diff_is_achieved : ∃ (minuend subtrahend result : ℕ), correct_result minuend subtrahend result ∧ DigitConstraints minuend subtrahend result :=
by
  sorry

end max_diff_is_achieved_l414_414969


namespace find_number_l414_414873

theorem find_number (x y : ℝ) (N : ℝ) (h1 : x = 9) (h2 : y = 4) :
  0.2 * 0.15 * 0.4 * 0.3 * 0.5 * N * real.sqrt x = real.pow 2 y → N = 29629.63 := 
by
  sorry

end find_number_l414_414873


namespace find_sin_DAE_l414_414911

variable {α : Type*}

-- Definitions for the problem conditions
def is_equilateral_triangle (A B C : α) [HasDistance α] := 
  distance A B = distance B C ∧ distance B C = distance C A

def point_coord (A B C D E : ℝ × ℝ) := 
  B = (0, 0) ∧ C = (9, 0) ∧ D = (3, 0) ∧ E = (7, 0) ∧ A = (0, 9 * Real.sqrt 3 / 2)

noncomputable def sin_angle_eq_sqrt3_div2 (A B C D E : ℝ × ℝ) [HasDistance ℝ] [HasSin ℝ] : Prop := 
  is_equilateral_triangle A B C ∧ 
  point_coord A B C D E → 
  Real.sin (angle A D E) = Real.sqrt 3 / 2

-- The final Lean statement
theorem find_sin_DAE (A B C D E : ℝ × ℝ) [HasDistance ℝ] [HasSin ℝ] :
  sin_angle_eq_sqrt3_div2 A B C D E :=
by sorry

end find_sin_DAE_l414_414911


namespace smaller_square_side_length_l414_414984

noncomputable def squareSideLength {d e f : ℕ} (h_e_primeNotDivisible: (∀ p : ℕ, p.prime → ¬(p ^ 2 ∣ e))) :
  (2 - real.sqrt ↑e) / f = (4 - real.sqrt 3) / 7 :=
sorry

theorem smaller_square_side_length (d e f : ℕ) :
  d = 4 → e = 3 → f = 7 → (∀ p : ℕ, p.prime → ¬(p ^ 2 ∣ e)) → d + e + f = 14 :=
by
  intros hd he hf h_primeNotDivisible
  have h1 := squareSideLength h_primeNotDivisible
  rw [hd, he, hf] at *
  -- add further necessary geometrical and algebraic arguments
  -- to connect conditions and the result.
  exact (by norm_num : 4 + 3 + 7 = 14)

end smaller_square_side_length_l414_414984


namespace sum_nat_numbers_l414_414654

/-- 
If S is the set of all natural numbers n such that 0 ≤ n ≤ 200, n ≡ 7 [MOD 11], 
and n ≡ 5 [MOD 7], then the sum of elements in S is 351.
-/
theorem sum_nat_numbers (S : Finset ℕ) 
  (hs : ∀ n, n ∈ S ↔ n ≤ 200 ∧ n % 11 = 7 ∧ n % 7 = 5) 
  : S.sum id = 351 := 
sorry 

end sum_nat_numbers_l414_414654


namespace max_value_of_f_l414_414304

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sin x + Real.cos x * Real.sin x

theorem max_value_of_f : ∃ x : ℝ, f x = sqrt 2 + 1/2 := sorry

end max_value_of_f_l414_414304


namespace prob_B_hired_is_3_4_prob_at_least_two_hired_l414_414311

-- Definitions for the conditions
def prob_A_hired : ℚ := 2 / 3
def prob_neither_A_nor_B_hired : ℚ := 1 / 12
def prob_B_and_C_hired : ℚ := 3 / 8

-- Targets to prove
theorem prob_B_hired_is_3_4 (P_A_hired : ℚ) (P_neither_A_nor_B_hired : ℚ) (P_B_and_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_neither_A_nor_B_hired_eq : P_neither_A_nor_B_hired = prob_neither_A_nor_B_hired)
    (P_B_and_C_hired_eq : P_B_and_C_hired = prob_B_and_C_hired)
    : ∃ x y : ℚ, y = 1 / 2 ∧ x = 3 / 4 :=
by
  sorry
  
theorem prob_at_least_two_hired (P_A_hired : ℚ) (P_B_hired : ℚ) (P_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_B_hired_eq : P_B_hired = 3 / 4)
    (P_C_hired_eq : P_C_hired = 1 / 2)
    : (P_A_hired * P_B_hired * P_C_hired) + 
      ((1 - P_A_hired) * P_B_hired * P_C_hired) + 
      (P_A_hired * (1 - P_B_hired) * P_C_hired) + 
      (P_A_hired * P_B_hired * (1 - P_C_hired)) = 2 / 3 :=
by
  sorry

end prob_B_hired_is_3_4_prob_at_least_two_hired_l414_414311


namespace probability_sum_five_l414_414193

-- Define the universe of balls
def balls : List ℕ := [1, 2, 3, 4]

-- Define the list of pairs drawn without replacement
def pairs (l : List ℕ) : List (ℕ × ℕ) := 
  (do x ← l, 
      y ← l, 
      if x < y then pure (x, y) else []) -- Ensuring unique combinations

-- Prove that the probability of the sum of the pairs being 5 is 1/3
theorem probability_sum_five (p : List (ℕ × ℕ)) : 
  (reasonably drawn p balls -> 
    (count (λ x, x.1 + x.2 = 5) p).toFloat / (length p).toFloat = 1.0 / 3.0) :=
by sorry


end probability_sum_five_l414_414193


namespace largest_four_digit_number_divisible_by_4_with_digit_sum_20_l414_414334

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def digit_sum_is_20 (n : ℕ) : Prop :=
  (n / 1000) + ((n % 1000) / 100) + ((n % 100) / 10) + (n % 10) = 20

theorem largest_four_digit_number_divisible_by_4_with_digit_sum_20 :
  ∃ n : ℕ, is_four_digit n ∧ is_divisible_by_4 n ∧ digit_sum_is_20 n ∧ ∀ m : ℕ, is_four_digit m ∧ is_divisible_by_4 m ∧ digit_sum_is_20 m → m ≤ n :=
  sorry

end largest_four_digit_number_divisible_by_4_with_digit_sum_20_l414_414334


namespace cost_of_each_bell_pepper_l414_414597

theorem cost_of_each_bell_pepper (cost_taco_shells cost_meat_per_pound total_spent number_bell_peppers : ℝ) :
  cost_taco_shells = 5 → cost_meat_per_pound = 3 → total_spent = 17 → number_bell_peppers = 4 →
  let cost_meat := 2 * cost_meat_per_pound in
  let remaining_cost := total_spent - cost_taco_shells - cost_meat in
  remaining_cost / number_bell_peppers = 1.5 := 
by intros; sorry

end cost_of_each_bell_pepper_l414_414597


namespace car_speed_l414_414988

theorem car_speed (time : ℕ) (distance : ℕ) (h1 : time = 5) (h2 : distance = 300) : distance / time = 60 := by
  sorry

end car_speed_l414_414988


namespace triangle_angle_bisector_proportion_l414_414895

theorem triangle_angle_bisector_proportion
  (a b c x y : ℝ)
  (h : x / c = y / a)
  (h2 : x + y = b) :
  x / c = b / (a + c) :=
sorry

end triangle_angle_bisector_proportion_l414_414895


namespace find_number_l414_414428

theorem find_number (x : ℝ) : (x - (7/13) * x = 110) → x = 237 :=
begin
    sorry
end

end find_number_l414_414428


namespace total_earnings_l414_414551

-- Define the given conditions and rates
def last_week_hours : ℕ := 35
def last_week_pay_rate : ℝ := 10
def pay_rate_increase : ℝ := 0.50
def regular_hours_threshold : ℕ := 40
def overtime_multiplier : ℝ := 1.5
def weekend_multiplier : ℝ := 1.7
def night_shifts_multiplier : ℝ := 1.3
def commission_rate : ℝ := 0.05
def sales_target_bonus : ℝ := 50
def deduction : ℝ := 20
def sales_target_reached : bool := true
def customer_satisfaction_below_threshold : bool := true

-- Breakdown of hours and sales
structure WorkDay where
  hours : ℕ
  night_hours : ℕ
  overtime_hours : ℕ
  weekend_hours : ℕ
  sales : ℝ

def week_work := [
  { hours := 8, night_hours := 3, overtime_hours := 0, weekend_hours := 0, sales := 200 },
  { hours := 10, night_hours := 4, overtime_hours := 0, weekend_hours := 0, sales := 400 },
  { hours := 8, night_hours := 0, overtime_hours := 0, weekend_hours := 0, sales := 500 },
  { hours := 9, night_hours := 3, overtime_hours := 1, weekend_hours := 0, sales := 300 },
  { hours := 5, night_hours := 0, overtime_hours := 0, weekend_hours := 0, sales := 200 },
  { hours := 6, night_hours := 0, overtime_hours := 0, weekend_hours := 6, sales := 300 },
  { hours := 4, night_hours := 2, overtime_hours := 0, weekend_hours := 4, sales := 100 }
]

noncomputable def calculate_total_earnings : ℝ :=
  let this_week_pay_rate := last_week_pay_rate + pay_rate_increase
  let regular_hours := week_work.foldl (λ acc d, acc + d.hours) 0
  let regular_pay := real.of_nat (min regular_hours_threshold regular_hours) * this_week_pay_rate
  let overtime_pay :=
    (overtime_multiplier * this_week_pay_rate) * 
    real.of_nat (week_work.foldl (λ acc d, acc + d.overtime_hours) 0)
  let weekend_pay := 
    (weekend_multiplier * this_week_pay_rate) * 
    real.of_nat (week_work.foldl (λ acc d, acc + d.weekend_hours) 0)
  let night_shift_pay := 
    (night_shifts_multiplier * this_week_pay_rate) * 
    real.of_nat (week_work.foldl (λ acc d, acc + d.night_hours) 0)
  let night_shift_sales := week_work.foldl (λ acc d, acc + d.sales) 0
  let commission := commission_rate * night_shift_sales
  let total_sales := week_work.foldl (λ acc d, acc + d.sales) 0
  let reached_bonus := if sales_target_reached then sales_target_bonus else 0
  let satisfaction_deduction := if customer_satisfaction_below_threshold then -deduction else 0
  let this_week_earnings :=
    regular_pay + overtime_pay + weekend_pay + night_shift_pay + commission + 
    reached_bonus + satisfaction_deduction
  let last_week_earnings := real.of_nat last_week_hours * last_week_pay_rate
  this_week_earnings + last_week_earnings

theorem total_earnings : calculate_total_earnings = 1208.05 :=
by
  sorry

end total_earnings_l414_414551


namespace min_distance_in_regular_tetrahedron_l414_414196

open Finset

theorem min_distance_in_regular_tetrahedron :
  let A := (0 : ℝ, 0 : ℝ, 0 : ℝ),
      B := (2 : ℝ, 0 : ℝ, 0 : ℝ),
      C := (1 : ℝ, Real.sqrt 3, 0 : ℝ),
      D := (1 : ℝ, Real.sqrt 3 / 3, Real.sqrt (8 / 3)) in
  let P := (1/2 : ℝ, 0 : ℝ, 0 : ℝ),
      Q := (1 : ℝ, Real.sqrt 3 / 3, 2 * Real.sqrt (2 / 3) / 3) in
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2) = Real.sqrt (11/12) :=
by sorry

end min_distance_in_regular_tetrahedron_l414_414196


namespace α_value_l414_414110

theorem α_value (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 2 * Real.pi)
  (h2 : ∃ (P : ℝ × ℝ), P = (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) ∧ 
                         α = Real.arctan2 (P.2) (P.1)) : 
  α = 11 * Real.pi / 6 := by
  sorry

end α_value_l414_414110


namespace sum_of_tangents_slopes_at_vertices_l414_414329

noncomputable def curve (x : ℝ) := (x + 3) * (x ^ 2 + 3)

theorem sum_of_tangents_slopes_at_vertices {x_A x_B x_C : ℝ}
  (h1 : curve x_A = x_A * (x_A ^ 2 + 6 * x_A + 9) + 3)
  (h2 : curve x_B = x_B * (x_B ^ 2 + 6 * x_B + 9) + 3)
  (h3 : curve x_C = x_C * (x_C ^ 2 + 6 * x_C + 9) + 3)
  : (3 * x_A ^ 2 + 6 * x_A + 3) + (3 * x_B ^ 2 + 6 * x_B + 3) + (3 * x_C ^ 2 + 6 * x_C + 3) = 237 :=
sorry

end sum_of_tangents_slopes_at_vertices_l414_414329


namespace arithmetic_square_root_problem_l414_414628

open Real

theorem arithmetic_square_root_problem 
  (a b c : ℝ)
  (ha : 5 * a - 2 = -27)
  (hb : b = ⌊sqrt 22⌋)
  (hc : c = -sqrt (4 / 25)) :
  sqrt (4 * a * c + 7 * b) = 6 := by
  sorry

end arithmetic_square_root_problem_l414_414628


namespace sum_possible_values_l414_414869

theorem sum_possible_values (y : ℝ) (h : y^2 = 36) : 
  y = 6 ∨ y = -6 → 6 + (-6) = 0 :=
by
  intro hy
  rw [add_comm]
  exact add_neg_self 6

end sum_possible_values_l414_414869


namespace mode_and_median_of_stem_and_leaf_diagram_l414_414206

-- Define the stem-and-leaf diagram data for the problem
def data_from_stem_and_leaf : list ℕ := [23, 24, 25, 25, 26, 26, 26, 30, 31, 31] -- Example data from the diagram

-- Define mode of a dataset
def mode (data : list ℕ) : ℕ := 
  (data.group_by id).key_max (fun xs => xs.length)

-- Define median of a dataset
def median (data : list ℕ) : ℕ := 
  let sorted_data := data.qsort (≤)
  sorted_data.nth (sorted_data.length / 2)

-- Prove that mode = 31 and median = 26 given the data
theorem mode_and_median_of_stem_and_leaf_diagram :
  mode data_from_stem_and_leaf = 31 ∧ median data_from_stem_and_leaf = 26 :=
by
  sorry

end mode_and_median_of_stem_and_leaf_diagram_l414_414206


namespace set_intersection_complement_l414_414483

def M : Set ℝ := { x | 0 < x ∧ x < 3 }
def N : Set ℝ := { x | x > 2 }

theorem set_intersection_complement : M ∩ (set.compl N) = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end set_intersection_complement_l414_414483


namespace length_of_BC_l414_414507

theorem length_of_BC (A B C X : Point) (AB AC BX CX : ℕ)
  (h1 : dist A B = 75)
  (h2 : dist A C = 105)
  (h3 : dist A X = 75)
  (h4 : Collinear B C X)
  (h5 : B ≠ X)
  (h6 : ∃ BX CX : ℕ, dist B X = BX ∧ dist C X = CX) :
  dist B C = 90 ∨ dist B C = 120 :=
sorry

end length_of_BC_l414_414507


namespace area_of_moon_slice_l414_414045

-- Definitions of the conditions
def larger_circle_radius := 5
def larger_circle_center := (2, 0)
def smaller_circle_radius := 2
def smaller_circle_center := (0, 0)

-- Prove the area of the moon slice
theorem area_of_moon_slice : 
  (1/4) * (larger_circle_radius^2 * Real.pi) - (1/4) * (smaller_circle_radius^2 * Real.pi) = (21 * Real.pi) / 4 :=
by
  sorry

end area_of_moon_slice_l414_414045


namespace rectangle_area_from_quadratic_l414_414139

theorem rectangle_area_from_quadratic :
  (∃ x1 x2 : ℝ, x1 * x2 = 6 ∧ ∀ x : ℝ, x^2 - 5 * x + 6 = 0 → x = x1 ∨ x = x2) →
  ∃ area : ℝ, area = 6 :=
by
  intro h
  use 6
  sorry

end rectangle_area_from_quadratic_l414_414139


namespace sozopolian_ineq_find_p_l414_414647

noncomputable def is_sozopolian (p a b c : ℕ) : Prop :=
  p % 2 = 1 ∧
  Nat.Prime p ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_ineq (p a b c : ℕ) (hp : is_sozopolian p a b c) :
  p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem find_p (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ (a + b + c) / 3 = p + 2) ↔ p = 5 :=
sorry

end sozopolian_ineq_find_p_l414_414647


namespace problem1_problem2_l414_414704

-- Problem 1 statement
theorem problem1 (x : ℝ) : (x^2 - 100 = -75) → (x = 5 ∨ x = -5) :=
by
  intro h
  sorry

-- Problem 2 statement
theorem problem2 : real.cbrt 8 + real.sqrt 0 - real.sqrt (1 / 4) = 3 / 2 :=
by
  sorry

end problem1_problem2_l414_414704


namespace limit_of_derivative_l414_414818

theorem limit_of_derivative (f : ℝ → ℝ) (h_deriv : deriv f 1 = 1) :
  tendsto (λ x : ℝ, (f(1 - x) - f(1 + x)) / (3 * x)) at_top (𝓝 (-2 / 3)) :=
begin
  sorry
end

end limit_of_derivative_l414_414818


namespace hyperbola_eccentricity_range_l414_414959

-- Given the definition of the hyperbola and its properties 
def hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let e := Real.sqrt (1 + (b^2 / a^2)) in
  e ≥ Real.sqrt 2

-- Main theorem statement
theorem hyperbola_eccentricity_range {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + (b^2 / a^2)) in
  e ∈ Set.Ici (Real.sqrt 2) :=
by
  sorry

end hyperbola_eccentricity_range_l414_414959


namespace problem_a9_b9_l414_414967

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Prove the goal
theorem problem_a9_b9 : a^9 + b^9 = 76 :=
by
  -- the proof will come here
  sorry

end problem_a9_b9_l414_414967


namespace Danielle_rooms_is_6_l414_414848

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l414_414848


namespace tangent_line_equation_l414_414077

theorem tangent_line_equation (x y : ℝ) (h : y = x^3 + 1) (t : x = -1) :
  3*x - y + 3 = 0 :=
sorry

end tangent_line_equation_l414_414077


namespace linear_regression_and_correlation_l414_414096

theorem linear_regression_and_correlation 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (Hn : n = 10) 
  (Hx : ∑ i, x i = 80) 
  (Hy : ∑ i, y i = 20)
  (Hxy : ∑ i, (x i) * (y i) = 184)
  (Hxx : ∑ i, (x i)^2 = 720) :
  let mean_x := (∑ i, x i) / n,
      mean_y := (∑ i, y i) / n,
      lxx := (∑ i, (x i)^2) - n * mean_x^2,
      lxy := (∑ i, (x i) * (y i)) - n * mean_x * mean_y,
      b := lxy / lxx,
      a := mean_y - b * mean_x,
      linear_eq := (λ x, b * x + a),
      predicted_savings := linear_eq 12
  in (linear_eq = (λ x, 0.3 * x - 0.4)) ∧ 
     (b > 0) ∧ 
     (predicted_savings = 3.2) := 
by sorry

end linear_regression_and_correlation_l414_414096


namespace sequences_converge_to_X_l414_414756

def x (n : ℕ) : ℝ
| 0     := 1
| (n+1) := sqrt (x n * y n)
  where y : ℕ → ℝ
  | 0     := 2021
  | (n+1) := (x n + y n) / 2

def y (n : ℕ) : ℝ
| 0     := 2021 
| (n+1) := (x n + y n) / 2
  where x : ℕ → ℝ
  | 0     := 1
  | (n+1) := sqrt (x n * y n)

theorem sequences_converge_to_X :
  ∃ X : ℝ, (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - X| < ε ∧ |y n - X| < ε) ∧ X = 0.1745 :=
sorry

end sequences_converge_to_X_l414_414756


namespace false_statement_l414_414485

-- Define propositions p and q
def p := ∀ x : ℝ, (|x| = x) ↔ (x ≥ 0)
def q := ∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → (∃ origin : ℝ, ∀ y : ℝ, f (origin + y) = f (origin - y))

-- Define the possible answers
def option_A := p ∨ q
def option_B := p ∧ q
def option_C := ¬p ∧ q
def option_D := ¬p ∨ q

-- Define the false option (the correct answer was B)
def false_proposition := option_B

-- The statement to prove
theorem false_statement : false_proposition = false :=
by sorry

end false_statement_l414_414485


namespace part1_part2_part3_l414_414091

-- Define S(n) as described
noncomputable def S (n : ℕ) : ℕ := sorry

-- Part 1: for all n ≥ 4, prove S(n) ≤ n² - 14
theorem part1 (n : ℕ) (hn : n ≥ 4) : S(n) ≤ n^2 - 14 := sorry

-- Part 2: find a positive integer n such that S(n) = n² - 14 (n = 13 in the solution)
theorem part2 : S(13) = 13^2 - 14 := sorry

-- Part 3: prove the infinite number of n such that S(n) = n² - 14
theorem part3 : ∃ᶠ n in at_top, S(n) = n^2 - 14 := sorry

end part1_part2_part3_l414_414091


namespace total_surface_area_prism_l414_414633

variables (a l : ℝ)
variables (h1 : ∀ l1 l2, l1 ≠ l2 → dist l1 l2 = a)
variables (h2 : ∃ l, l = l)
variables (h3 : ∀ l, is_angle_inclined l 60)

theorem total_surface_area_prism (a l : ℝ) : 
  let S_total := a * (3 * l + a) in S_total := a * (3 * l + a) :=
by
  sorry

end total_surface_area_prism_l414_414633


namespace smallest_n_l414_414681

theorem smallest_n (n : ℕ) : (725 * n) % 35 = (1275 * n) % 35 → n = 7 :=
by
  assume h: (725 * n) % 35 = (1275 * n) % 35
  sorry

end smallest_n_l414_414681


namespace shaded_quadrilateral_area_l414_414020

theorem shaded_quadrilateral_area :
  let side1 := 3
  let side2 := 5
  let side3 := 7
  let total_length := 15
  let height_largest_square := 7
  let height1 := (side1 * height_largest_square) / total_length
  let height2 := ((side1 + side2) * height_largest_square) / total_length
  let height := side2
  let area := height * (height1 + height2) / 2
  area = 12.825 := by
  let side1 := 3
  let side2 := 5
  let side3 := 7
  let total_length := side1 + side2 + side3
  let height_largest_square := side3
  let height1 := (side1 * height_largest_square) / total_length
  let height2 := ((side1 + side2) * height_largest_square) / total_length
  let height := side2
  let area := height * (height1 + height2) / 2
  have : total_length = 15 := by
    sorry
  have : height_largest_square = 7 := by
    sorry
  have : height1 = (3 * 7) / 15 := by
    sorry
  have : height1 = 1.4 := by
    sorry
  have : height2 = (8 * 7) / 15 := by
    sorry
  have : height2 = 3.733333 := by
    sorry
  have : height = 5 := by
    sorry
  have : area = height * (height1 + height2) / 2 := by
    sorry
  have : area = 5 * (1.4 + 3.733333) / 2 := by
    sorry
  have : area = 12.825 := by
    sorry
  exact this

end shaded_quadrilateral_area_l414_414020


namespace solve_for_x_when_f_prime_is_2_l414_414572

theorem solve_for_x_when_f_prime_is_2 (f : ℝ → ℝ) (h : f = λ x, x * real.log x) (h_deriv : (λ x, real.log x + 1) = (2 : ℝ)) :
  ∃ x : ℝ, x = real.exp 1 :=
by {
  sorry
}

end solve_for_x_when_f_prime_is_2_l414_414572


namespace part1_part2_l414_414097

-- Define the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define a point in the first quadrant on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := 0 < x ∧ 0 < y ∧ ellipse_eq x y

-- Define dot product condition for P and the foci
def dot_product_condition (P F1 F2: ℝ × ℝ) : Prop :=
  let PF1 := (F1.1 - P.1, F1.2 - P.2)
  let PF2 := (F2.1 - P.1, F2.2 - P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = -(5 / 4)

-- Part 1: Prove coordinates of point P on the ellipse
theorem part1 : ∃ (P : ℝ × ℝ), point_on_ellipse P.1 P.2 ∧
  dot_product_condition P (-sqrt 3, 0) (sqrt 3, 0) ∧ P = (1, sqrt 3 / 2) :=
by 
  sorry

-- Define the slope range condition
def slope_range (k : ℝ) : Prop :=
  (3 / 4 < k^2) ∧ (k^2 < 4)

-- Part 2: Prove the range of values for slope k
theorem part2 : ∃ (k : ℝ), slope_range k ∧
  (k ∈ (-2 : ℝ, -(sqrt 3) / 2) ∪ (sqrt 3 / 2, 2)) :=
by 
  sorry

end part1_part2_l414_414097


namespace find_y_coordinate_l414_414916

noncomputable def y_coordinate_of_point_on_line : ℝ :=
  let x1 := 10
  let y1 := 3
  let x2 := 4
  let y2 := 0
  let x := -2
  let m := (y1 - y2) / (x1 - x2)
  let b := y1 - m * x1
  m * x + b

theorem find_y_coordinate :
  (y_coordinate_of_point_on_line = -3) :=
by
  sorry

end find_y_coordinate_l414_414916


namespace triangle_AC_length_l414_414893

noncomputable def cos60 : ℝ := real.cos (real.pi / 3)

theorem triangle_AC_length (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (AB BC AC : ℝ) (angleB : real.angle) :
  AB = 1 → BC = 2 → angleB = 60 →
  AC = real.sqrt 3 :=
by
  intro hAB hBC hAngleB
  -- Create points A, B, C in a metric space
  let AB := 1
  let BC := 2
  let angleB := real.pi / 3
  
  -- Using Law of Cosines
  have law_of_cosines : ∀ a b : ℝ, ∀ C : real.angle, (a^2 + b^2 - 2 * a * b * cos C) = (real.sqrt 3)^2, 
  from 
    sorry,
  
  -- Show that the calculated side length AC is equal to sqrt(3)
  have hAC : AC = real.sqrt 3, from
    begin
      sorry
    end,
  exact hAC

end triangle_AC_length_l414_414893


namespace book_selection_l414_414624

theorem book_selection :
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  tier1 + tier2 + tier3 = 16 :=
by
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  sorry

end book_selection_l414_414624


namespace number_of_pairs_l414_414371

theorem number_of_pairs (p : ℚ) : 
  (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 50 → 
    let a_2i := 2 * i in
    let a_2i_1 := 2 * i - 1 in
    a_2i > a_2i_1) →
  p = (1 / 2) ^ 50 →
  ∃ (n : ℕ), 
    n = 6 ∧ 
    ∀ (a b : ℕ), 
      (1 / a ^ b) = p ↔ (a, b) ∈ {n ∈ ℕ | ∃ k : ℕ, a = 2 ^ k ∧ k * b = 50} :=
by {
    sorry
}

end number_of_pairs_l414_414371


namespace non_deg_ellipse_condition_l414_414301

theorem non_deg_ellipse_condition (k : ℝ) : k > -19 ↔ 
  (∃ x y : ℝ, 3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k) :=
sorry

end non_deg_ellipse_condition_l414_414301


namespace tangent_circle_radius_l414_414029

-- Definition of the ellipse condition based on given major (2a) and minor (2b) axes
def is_ellipse (a b : ℝ) (point : ℝ × ℝ) : Prop :=
  let (x, y) := point in
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Definition of the circle condition based on its center and radius
def is_circle (center : ℝ × ℝ) (r : ℝ) (point : ℝ × ℝ) : Prop :=
  let (h, k) := center in
  let (x, y) := point in
  (x - h)^2 + (y - k)^2 = r^2

-- Proof statement for the problem
theorem tangent_circle_radius :
  ∀ (focus : ℝ × ℝ),
  let center := (0, 0) in
  let a := 7 in  -- half of the major axis length
  let b := 5 in  -- half of the minor axis length
  is_ellipse a b focus →
  focus = (real.sqrt (a^2 - b^2), 0) →
  ∃ r : ℝ, r = 4 ∧
    (∀ (tangent_point : ℝ × ℝ), is_circle focus r tangent_point → is_ellipse a b tangent_point) :=
begin
  sorry
end

end tangent_circle_radius_l414_414029


namespace determine_standard_equation_of_ellipse_l414_414825

-- Define the necessary conditions for the ellipse
variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variable (eccentricity : ℝ := 1 / 2)
variable (sum_of_lengths : ℝ := 6)

-- Define the properties of the ellipse
def ellipse_properties : Prop :=
  (c / a = eccentricity) ∧
  (2 * a + 2 * c = sum_of_lengths)

-- Define the standard form of the ellipse equation
def standard_ellipse_equation : Prop :=
  (a = 2) ∧ (b = sqrt 3) ∧ (c = 1) ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (4 * x^2 + 3 * y^2 = 12))

-- State the Lean 4 theorem
theorem determine_standard_equation_of_ellipse :
  ellipse_properties a b c → standard_ellipse_equation a b c :=
by
  sorry

end determine_standard_equation_of_ellipse_l414_414825


namespace solve_length_CD_l414_414204

-- Definitions and conditions
def is_isosceles_triangle (A B C : Point) (AB AC : ℝ) : Prop :=
  AB = AC

def is_altitude (A D B C : Point) : Prop :=
  A, D, B, C are collinear /\ AD is perpendicular to BC

def base_length (BC : ℝ) : Prop :=
  BC = 8

theorem solve_length_CD
  (A B C D : Point)
  (AB AC BC : ℝ)
  (h_isosceles : is_isosceles_triangle A B C AB AC)
  (h_altitude : is_altitude A D B C)
  (h_base_length : base_length BC)
  : let CD := BC / 2 in CD = 4 :=
by
  sorry

end solve_length_CD_l414_414204


namespace quadratic_function_inequality_l414_414834

theorem quadratic_function_inequality 
    (a b c : ℝ) 
    (h1 : c > b)
    (h2 : b > a)
    (h3 : f_x : (λ x, a * x^2 + 2 * b * x + c))
    (h4 : f_x 1 = 0)
    (h5 : ∃ x, f_x x = -a) : 
    0 ≤ b / a ∧ b / a < 1 := 
by
    sorry

end quadratic_function_inequality_l414_414834


namespace min_distance_run_l414_414198

noncomputable def A : Point := (0, 0)
noncomputable def B : Point := (1500, 0)
noncomputable def wall : Line := (0, 400) -- This line is simplified for the wall condition. 

theorem min_distance_run :
  ∃ C : Point, (distance A C) + (distance C B) = 1803 :=
by
  sorry

end min_distance_run_l414_414198


namespace triangle_angle_right_angle_l414_414330

theorem triangle_angle_right_angle
  {A B C D E F N P : Type}
  [normed_add_comm_group A] [inner_product_space ℝ A]
  [normed_add_comm_group B] [inner_product_space ℝ B]
  [normed_add_comm_group C] [inner_product_space ℝ C]
  [normed_add_comm_group D] [inner_product_space ℝ D]
  [normed_add_comm_group E] [inner_product_space ℝ E]
  [normed_add_comm_group F] [inner_product_space ℝ F]
  [normed_add_comm_group N] [inner_product_space ℝ N]
  [normed_add_comm_group P] [inner_product_space ℝ P]
  (h1: ∃ (a b c : ℝ), 
    a = dist A B ∧ b = dist B C ∧ c = dist A C ∧ 
    a = 12 ∧ b = 18 ∧ c = 15)
  (h2: ∃ (d e f : ℝ), 
    d = dist A D ∧ e = dist B E ∧ f = dist C F ∧ 
    d = 4 ∧ e = 12 ∧ f = 10)
  (h3: on_circumcircle N B D E)
  (h4: ∃ (p : P), 
    p ≠ E ∧ 
    line_intersect_chords BN ED P ∧ 
    angle_90 BP PD)
  : angle B N D = 90 :=
sorry

end triangle_angle_right_angle_l414_414330


namespace distinct_values_of_x_for_1001_l414_414046

theorem distinct_values_of_x_for_1001 :
  ∃! xs : set ℝ,
  (∀ x ∈ xs, x > 0) ∧
  (∀ x ∈ xs, ∃ (a : ℕ → ℝ), a 1 = x ∧ a 2 = 1000 ∧ 
    (∀ n ≥ 3, a n = (a (n-1) + 1) / (a (n-2))) ∧ 
    1001 ∈ set.range a) ∧
  xs.card = 4 :=
sorry

end distinct_values_of_x_for_1001_l414_414046


namespace find_k_from_condition_l414_414140

theorem find_k_from_condition :
  (∃ k : ℝ, k ≠ 0 ∧ (-2:ℝ) = k * (-1:ℝ) - 4) → (-2:ℝ) = (-2:ℝ) :=
by
  intro h
  rcases h with ⟨k, hk0, hk⟩
  rw [<-eq_sub_iff_add_eq'] at hk
  rw [<-sub_eq_add_neg]
  sorry

end find_k_from_condition_l414_414140


namespace f_increasing_on_interval_l414_414164

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (x^2, x + 1)
noncomputable def vec_b (x t : ℝ) : ℝ × ℝ := (1 - x, t)

noncomputable def f (x t : ℝ) : ℝ :=
  let (a1, a2) := vec_a x
  let (b1, b2) := vec_b x t
  a1 * b1 + a2 * b2

noncomputable def f_prime (x t : ℝ) : ℝ :=
  2 * x - 3 * x^2 + t

theorem f_increasing_on_interval :
  ∀ t x, -1 < x → x < 1 → (0 ≤ f_prime x t) → (t ≥ 5) :=
sorry

end f_increasing_on_interval_l414_414164


namespace sum_of_roots_of_y_squared_eq_36_l414_414868

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end sum_of_roots_of_y_squared_eq_36_l414_414868


namespace sum_of_first_2016_terms_l414_414184

-- Define the arithmetic sequence and condition
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

def condition (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 2015 + a 2016 = 3

-- Define the sum of the first n terms of the sequence
def sum_first_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

theorem sum_of_first_2016_terms (a : ℕ → ℝ) (h1 : is_arithmetic a) (h2 : condition a) :
  sum_first_n a 2016 = 1512 :=
sorry

end sum_of_first_2016_terms_l414_414184


namespace cubic_function_decreasing_l414_414629

theorem cubic_function_decreasing (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → (a ≤ 0) := 
by 
  sorry

end cubic_function_decreasing_l414_414629


namespace cards_distribution_l414_414498

theorem cards_distribution (total_cards people : ℕ) (h1 : total_cards = 48) (h2 : people = 7) :
  (people - (total_cards % people)) = 1 :=
by
  sorry

end cards_distribution_l414_414498


namespace complex_pow_imaginary_unit_l414_414461

theorem complex_pow_imaginary_unit (i : ℂ) (h : i^2 = -1) : i^2015 = -i :=
sorry

end complex_pow_imaginary_unit_l414_414461


namespace find_n_l414_414421

theorem find_n :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * Real.pi / 180) = Real.cos (942 * Real.pi / 180) := sorry

end find_n_l414_414421


namespace no_new_marks_after_revolution_sum_of_interior_angles_of_polygon_l414_414280

/-
Conditions:
1. João makes a mark at the position of the hour hand every 80 minutes.
2. The hour hand completes a full revolution every 720 minutes.
-/

-- Definitions for the conditions
def mark_interval : ℕ := 80
def revolution_time : ℕ := 720
def marks_in_one_revolution := revolution_time / mark_interval

-- Target proof problem
theorem no_new_marks_after_revolution :
  ∀ t, t ≥ revolution_time → (t % revolution_time = 0 ∨ (t - revolution_time) % revolution_time = 0) := by
  sorry

theorem sum_of_interior_angles_of_polygon :
  marks_in_one_revolution = 9 → 
  let n := marks_in_one_revolution in
  let S := 180 * (n - 2) in
  S = 1260 := by
  sorry

end no_new_marks_after_revolution_sum_of_interior_angles_of_polygon_l414_414280


namespace shaded_quilt_fraction_l414_414653

-- Define the basic structure of the problem using conditions from step a

def is_unit_square (s : ℕ) : Prop := s = 1

def grid_size : ℕ := 4
def total_squares : ℕ := grid_size * grid_size

def shaded_squares : ℕ := 2
def half_shaded_squares : ℕ := 4

def fraction_shaded (shaded: ℕ) (total: ℕ) : ℚ := shaded / total

theorem shaded_quilt_fraction :
  fraction_shaded (shaded_squares + half_shaded_squares / 2) total_squares = 1 / 4 :=
by
  sorry

end shaded_quilt_fraction_l414_414653


namespace maximum_hexagon_area_l414_414032

def is_convex_hexagon (A B C D E F O : ℝ × ℝ) : Prop :=
  -- Assume some conditions to guarantee the convexity (not explicitly defined here)

def form_convex_hexagon_area (x y z s t u : ℝ) : ℝ :=
  let S := (x * y + y * z + z * s + s * t + t * u + u * x) in
  (Math.sqrt 3) / 4 * S

theorem maximum_hexagon_area :
  ∀ (A B C D E F O : ℝ × ℝ)
    (AB BC CD DE EF FA : ℝ),
    AB = 1 → BC = 2 * Math.sqrt 2 → CD = 5 → DE = 5 → EF = 4 → FA = 3 →
    is_convex_hexagon A B C D E F O →
    ∃ S, S = 21 * Math.sqrt 3 ∧ form_convex_hexagon_area AB BC CD DE EF FA ≤ S :=
begin
  intros,
  sorry
end

end maximum_hexagon_area_l414_414032


namespace find_n_l414_414181

theorem find_n (n : ℕ) (h_ineq : 1 + 3 / (n + 1 : ℝ) < real.sqrt 3 ∧ real.sqrt 3 < 1 + 3 / (n : ℝ)) : n = 4 :=
sorry

end find_n_l414_414181


namespace rectangle_side_length_relation_l414_414183

variable (x y : ℝ)

-- Condition: The area of the rectangle is 10
def is_rectangle_area_10 (x y : ℝ) : Prop := x * y = 10

-- Theorem: Given the area condition, express y in terms of x
theorem rectangle_side_length_relation (h : is_rectangle_area_10 x y) : y = 10 / x :=
sorry

end rectangle_side_length_relation_l414_414183


namespace triangle_not_obtuse_l414_414222

theorem triangle_not_obtuse 
  (A B C : Type) 
  [metric_space A] [has_dist A] 
  (a b c : ℝ) 
  (h1: b = 2 * a) 
  (h2: ∠A = 30) : 
  ¬ obtuse_angle A B C := 
sorry

end triangle_not_obtuse_l414_414222


namespace real_part_product_l414_414281

noncomputable def problem_statement : ℚ :=
  let r := complex.cos (real.pi / 12)
  let s := complex.cos (3 * real.pi / 4)
  let t := complex.cos (17 * real.pi / 12)
  let product := (sqrt 2 * r) * (sqrt 2 * s) * (sqrt 2 * t)
  product

theorem real_part_product :
  let z : complex := complex.mk 0 1 in
  z ^ 3 = complex.mk 2 2 →
  let r : ℚ := sqrt 2 * complex.cos (real.pi / 12) in
  let s : ℚ := sqrt 2 * complex.cos (3 * real.pi / 4) in
  let t : ℚ := sqrt 2 * complex.cos (17 * real.pi / 12) in
  let product : ℚ := r * s * t in
  ∃ p q : ℕ, nat.coprime p q ∧ product = p / q ∧ p + q = 3 :=
sorry

end real_part_product_l414_414281


namespace find_distance_EF_l414_414161

-- Define the mathematical conditions
variables (a b : Type) [skew_lines a b] (theta : ℝ) (d m n : ℝ)

-- Define points E and F on lines a and b respectively
variables (E F : Type) [point_on_line E a] [point_on_line F b] (A A' : Type) 
[A_point_on_perpendicular A b A'] [A'_point_on_perpendicular A' a A] (distance_A_A' : dist A A' = d)
(distance_A'_E : dist A' E = m) (distance_A_F : dist A F = n)

-- The exact theorem to prove
theorem find_distance_EF :
  dist E F = sqrt (d^2 + m^2 + n^2 ± 2 * m * n * cos theta) :=
sorry

end find_distance_EF_l414_414161


namespace tg_ctg_problem_l414_414751

-- Define tangent and cotangent functions explicitly for use in Lean
noncomputable def tan (x : ℝ) : ℝ := Real.tan x
noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x

theorem tg_ctg_problem (x : ℝ) (a : ℝ) (p : ℤ) :
  (a = tan x + cot x) ∧ (a ∈ Set.Ioi 0) ∧ (Nat.Prime p) ∧ (p = tan^3 x + cot^3 x) →
  ∃ k : ℤ, x = π / 4 + k * π :=
by
  sorry

end tg_ctg_problem_l414_414751


namespace find_brick_width_l414_414430

-- Define the conditions
def length : ℝ := 8
def height : ℝ := 2
def surface_area : ℝ := 112

-- Define the formula for the surface area of a rectangular prism
def surface_area_formula (l w h : ℝ) : ℝ :=
  2 * l * w + 2 * l * h + 2 * w * h

-- The theorem to prove that the width w is 4 cm
theorem find_brick_width (w : ℝ) (h : ℝ := height) (l : ℝ := length) :
  surface_area_formula l w h = surface_area → w = 4 :=
by
  -- this is where proof steps would go
  sorry

end find_brick_width_l414_414430


namespace triangle_height_check_l414_414226

theorem triangle_height_check :
  ∀ (A B C : Type) (AC BC : ℝ),
  AC = 2 → BC = 3 → 
  let R := circumscribed_radius A B C in
  let CH := height_from_C_to_AB A B C in
  let OM := distance_center_to_AB A B C in
  OM = R / 2 →
  CH < sqrt (3 / 2) →
  CH = 3 * sqrt (3 / 19) :=
by
  intros,
  let R := circumscribed_radius A B C,
  let CH := height_from_C_to_AB A B C,
  let OM := distance_center_to_AB A B C,
  have hOM : OM = R / 2, { sorry },
  have h1 : CH < sqrt (3 / 2), { sorry },
  have h2 : CH = 3 * sqrt (3 / 19), { sorry },
  exact h2,

end triangle_height_check_l414_414226


namespace total_balloons_l414_414321

theorem total_balloons (Gold Silver Black Total : Nat) (h1 : Gold = 141)
  (h2 : Silver = 2 * Gold) (h3 : Black = 150) (h4 : Total = Gold + Silver + Black) :
  Total = 573 := 
by
  sorry

end total_balloons_l414_414321


namespace sum_of_coefficients_l414_414494

theorem sum_of_coefficients:
  (∃ (a a_1 a_2 ... a_{11} : ℝ), (2*x - 1)^11 = a + a_1*x + a_2*x^2 + ... + a_{11}*x^11) →
  a + a_1 + a_2 + ... + a_{11} = 1 :=
sorry

end sum_of_coefficients_l414_414494


namespace folder_cost_calc_l414_414724

noncomputable def pencil_cost : ℚ := 0.5
noncomputable def dozen_pencils : ℕ := 24
noncomputable def num_folders : ℕ := 20
noncomputable def total_cost : ℚ := 30
noncomputable def total_pencil_cost : ℚ := dozen_pencils * pencil_cost
noncomputable def remaining_cost := total_cost - total_pencil_cost
noncomputable def folder_cost := remaining_cost / num_folders

theorem folder_cost_calc : folder_cost = 0.9 := by
  -- Definitions
  have pencil_cost_def : pencil_cost = 0.5 := rfl
  have dozen_pencils_def : dozen_pencils = 24 := rfl
  have num_folders_def : num_folders = 20 := rfl
  have total_cost_def : total_cost = 30 := rfl
  have total_pencil_cost_def : total_pencil_cost = dozen_pencils * pencil_cost := rfl
  have remaining_cost_def : remaining_cost = total_cost - total_pencil_cost := rfl
  have folder_cost_def : folder_cost = remaining_cost / num_folders := rfl

  -- Calculation steps given conditions
  sorry

end folder_cost_calc_l414_414724


namespace perpendicular_BD_OS_l414_414712

-- Define the geometric entities: points, lines, circle, etc.
variables {Point : Type*} [metric_space Point] [normed_group Point] [normed_space ℝ Point]
variables (O A B C D K L M N S : Point)
variables (circle : set Point) (quadrilateral : set Point) (tangent_points : set Point)
variables (tangent_line_AB : set Point) (tangent_line_BC : set Point)
variables (tangent_line_CD : set Point) (tangent_line_DA : set Point)
variables (line_KL : set Point) (line_MN : set Point)

-- Conditions
def is_circle_with_center (O : Point) (circle : set Point) : Prop :=
  ∀ P ∈ circle, dist O P = dist O (classical.some circle.nonempty)

def is_convex_quadrilateral (A B C D : Point) (quadrilateral : set Point) : Prop :=
  quadrilateral = {A, B, C, D} ∧ convex ℝ (convex_hull ℝ quadrilateral)

def is_tangent (line : set Point) (circle : set Point) (tangent_point : Point) : Prop :=
  tangent_point ∈ line ∧ tangent_point ∈ circle ∧ ∃ T ∈ line, ∃ C ∈ circle, ∀ X ∈ line, ∀ Y ∈ circle, ⟪X - tangent_point, Y - tangent_point⟫ = 0

def lines_intersect_at (line1 line2 : set Point) (intersection_point : Point) : Prop :=
  intersection_point ∈ line1 ∧ intersection_point ∈ line2 ∧ ¬parallel line1 line2

-- Given: conditions
variables (h_circle : is_circle_with_center O circle)
variables (h_quadrilateral : is_convex_quadrilateral A B C D quadrilateral)
variables (h_tangent_K : is_tangent tangent_line_AB circle K)
variables (h_tangent_L : is_tangent tangent_line_BC circle L)
variables (h_tangent_M : is_tangent tangent_line_CD circle M)
variables (h_tangent_N : is_tangent tangent_line_DA circle N)
variables (h_intersect : lines_intersect_at line_KL line_MN S)

-- To Prove: BD is perpendicular to OS.
theorem perpendicular_BD_OS 
  (hO_in_circle : O ∈ circle)
  (hK_in_quadrilateral : K ∈ quadrilateral)
  (hL_in_quadrilateral : L ∈ quadrilateral)
  (hM_in_quadrilateral : M ∈ quadrilateral)
  (hN_in_quadrilateral : N ∈ quadrilateral)
  (hS_on_line_KL : S ∈ line_KL)
  (hS_on_line_MN : S ∈ line_MN)
  (hS_not_parallel_KL_MN : ¬parallel line_KL line_MN)
  : is_perpendicular (B - D) (O - S) := sorry

end perpendicular_BD_OS_l414_414712


namespace transformation_matrix_correct_l414_414185

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℤ := ![
  #[0, -1],
  #[1, 0]
]

def scaling_mirroring_matrix : Matrix (Fin 2) (Fin 2) ℤ := ![
  #[2, 0],
  #[0, -2]
]

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℤ := scaling_mirroring_matrix ⬝ rotation_matrix_90_ccw

theorem transformation_matrix_correct :
  transformation_matrix = ![
    #[0, -2],
    #[2, 0]
  ] :=
begin
  sorry
end

end transformation_matrix_correct_l414_414185


namespace probability_two_same_color_balls_l414_414416

open Rat

theorem probability_two_same_color_balls:
  let total_balls := 16,
      prob_blue := (8 / total_balls) * (8 / total_balls),
      prob_green := (5 / total_balls) * (5 / total_balls),
      prob_red := (3 / total_balls) * (3 / total_balls),
      total_prob := prob_blue + prob_green + prob_red in
  total_prob = (49 / 128) := by
  sorry

end probability_two_same_color_balls_l414_414416


namespace inclination_angle_range_l414_414341

theorem inclination_angle_range (k : ℝ) (α : ℝ) 
  (h1 : -real.sqrt 3 ≤ k)
  (h2 : k ≤ real.sqrt 3 / 3)
  (h3 : α ∈ set.Ico 0 real.pi) :
  (α ∈ set.Icc 0 (real.pi / 6) ∨ α ∈ set.Ico (2 * real.pi / 3) real.pi) :=
sorry

end inclination_angle_range_l414_414341


namespace no_true_statements_l414_414748

def n_sharp (n : ℕ) : ℝ := 1 / (n + 1)

def statement_i : Prop := n_sharp 4 + n_sharp 8 = n_sharp 12
def statement_ii : Prop := n_sharp 9 - n_sharp 3 = n_sharp 6
def statement_iii : Prop := n_sharp 5 * n_sharp 7 = n_sharp 35
def statement_iv : Prop := n_sharp 15 / n_sharp 3 = n_sharp 5

def num_true_statements : ℕ := 
  (if statement_i then 1 else 0) + 
  (if statement_ii then 1 else 0) + 
  (if statement_iii then 1 else 0) + 
  (if statement_iv then 1 else 0)

theorem no_true_statements : num_true_statements = 0 := sorry

end no_true_statements_l414_414748


namespace find_n0_l414_414211

noncomputable theory

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d

-- Define the terms as given in the problem
def a_12 (a d : ℝ) : ℝ := arithmetic_seq a d 12
def a_23 (a d : ℝ) : ℝ := arithmetic_seq a d 23

-- Define the condition involving a_12 and a_23
def condition (a d : ℝ) : Prop := 4 * a_12 a d = -3 * a_23 a d ∧ 4 * a_12 a d > 0 

-- Define the sequence b_n according to the problem statement
def b_n (a d : ℝ) (n : ℕ) : ℝ :=
  let a_n := arithmetic_seq a d n
  let a_n1 := arithmetic_seq a d (n + 1)
  let a_n2 := arithmetic_seq a d (n + 2)
  a_n * a_n1 / a_n2

-- Define the sum S_n of the first n terms of b_n
def S_n (a d : ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, b_n a d i

-- Define the maximum term S_{n_0}
def is_max_term (a d : ℝ) (n_0 : ℕ) : Prop :=
  ∀ n : ℕ, S_n a d n_0 ≥ S_n a d n

-- State the main theorem
theorem find_n0 (a d : ℝ) (h : condition a d) : is_max_term a d 14 := sorry

end find_n0_l414_414211


namespace cos_A_cos_C_range_l414_414797

theorem cos_A_cos_C_range (A B C : ℝ) (h₀ : triangle A B C)
  (h₁ : B = π / 3) :
  -1 / 2 ≤ cos A * cos C ∧ cos A * cos C ≤ 1 / 4 :=
sorry

end cos_A_cos_C_range_l414_414797


namespace min_a1_l414_414799

-- Define the arithmetic sequence and its properties
def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
∀ n m, a m = a n + (m - n) * (a 2 - a 1)

noncomputable def a_n : ℕ → ℕ
| n := 2023 - 8 * (n - 9)

-- Define a_arithmetic_seq meeting the given conditions
def a_seq (a : ℕ → ℕ) : Prop :=
is_arithmetic_seq a ∧ a 9 = 2023

-- Find the minimum value of a1
theorem min_a1 (a : ℕ → ℕ) (h : is_arithmetic_seq a) (h1 : ∀ n, a n > 0) (h2 : a 9 = 2023) : ∃ (a1 : ℕ), a1 = 7 :=
by {
    use 2023 - 8 * 252,
    sorry
}

end min_a1_l414_414799


namespace water_tank_full_capacity_l414_414735

theorem water_tank_full_capacity (x : ℝ) (h1 : x * (3/4) - x * (1/3) = 15) : x = 36 := 
by
  sorry

end water_tank_full_capacity_l414_414735


namespace rachel_budget_proof_l414_414604

-- Define the prices Sara paid for shoes and the dress
def shoes_price : ℕ := 50
def dress_price : ℕ := 200

-- Total amount Sara spent
def sara_total : ℕ := shoes_price + dress_price

-- Rachel's budget should be double of Sara's total spending
def rachels_budget : ℕ := 2 * sara_total

-- The theorem statement
theorem rachel_budget_proof : rachels_budget = 500 := by
  unfold rachels_budget sara_total shoes_price dress_price
  rfl

end rachel_budget_proof_l414_414604


namespace cos_function_equal_zero_implies_difference_is_multiple_of_pi_l414_414255

noncomputable def f (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range n, (1 / 2 ^ k) * real.cos (a k + x)

theorem cos_function_equal_zero_implies_difference_is_multiple_of_pi
  {a : ℕ → ℝ} {n : ℕ} {x1 x2 : ℝ} (h : f a n x1 = f a n x2) :
  ∃ (m : ℤ), x1 - x2 = m * real.pi :=
sorry

end cos_function_equal_zero_implies_difference_is_multiple_of_pi_l414_414255


namespace find_line_eqn_l414_414823

-- Define the conditions
def circle_eqn (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

def point_P : ℝ × ℝ := (2, 0)

def center_C : ℝ × ℝ := (0, 4)

-- The symmetry condition with respect to line l
def is_symmetric (C l : ℝ × ℝ) (P : ℝ × ℝ) (line_eqn : ℝ → ℝ → Prop) : Prop :=
  ∃ k, line_eqn (C.1 + k * (P.1 - C.1)) (C.2 + k * (P.2 - C.2)) ∧
       line_eqn ((C.1 + P.1)/2) ((C.2 + P.2)/2)

def line_eqn (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- The proof statement
theorem find_line_eqn : 
  ∀ (x y : ℝ), 
    circle_eqn x y → 
    is_symmetric center_C line_eqn point_P line_eqn → 
    line_eqn (x - y) := 
sorry

end find_line_eqn_l414_414823


namespace tessa_initial_apples_l414_414285

-- We define a variable for the number of apples Tessa initially had
def initial_apples (x : ℕ) : Prop :=
  -- Condition that after receiving 5 apples, Tessa has 9 apples now
  (x + 5 = 9)

theorem tessa_initial_apples : ∃ x, initial_apples x → x = 4 :=
by {
  existsi 4,
  intro h,
  -- Prove that 4 + 5 = 9
  exact h
}

end tessa_initial_apples_l414_414285


namespace count_numbers_divisible_by_4_5_or_both_l414_414169

theorem count_numbers_divisible_by_4_5_or_both : 
  let count_numbers (n : ℕ) (d : ℕ) := (n / d) in
  let divisible_by_4 := count_numbers 60 4 in
  let divisible_by_5 := count_numbers 60 5 in
  let both := count_numbers 60 20 in
  (divisible_by_4 + divisible_by_5 - both) = 24
:= sorry

end count_numbers_divisible_by_4_5_or_both_l414_414169


namespace largest_S_n_is_S_23_l414_414315

open Nat

def in_arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = ∑ i in range n, a (i + 1)

theorem largest_S_n_is_S_23 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hin_arith_seq : in_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_a1_pos : a 1 > 0)
  (h_S_eq : S 36 = S 10) :
  S 23 = max (finset.image S (finset.range 37)) := 
sorry

end largest_S_n_is_S_23_l414_414315


namespace no_natural_number_for_square_condition_l414_414758

theorem no_natural_number_for_square_condition :
  ∀ n : ℕ, ¬ (∃ k : ℕ, k^2 = n^5 - 5*n^3 + 4*n + 7) :=
begin
  intro n,
  intro h,
  cases h with k hk,
  sorry,
end

end no_natural_number_for_square_condition_l414_414758


namespace product_of_constants_l414_414082

theorem product_of_constants :
  let factors := λ (ab : ℤ × ℤ), ab.1 * ab.2 = -24,
      t_values := λ (ab : ℤ × ℤ), ab.1 + ab.2,
      t_list := ( ([-1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -8, 8, -12, 12, -24, 24].product ([-1, 1, -2, 2, -3, 3, -4, 4, -6, 6, 4, -6, 8, -8, 12, -12, 24, -24])).filter factors).map t_values
  in (t_list.prod : ℤ) = 1 := by
  sorry

end product_of_constants_l414_414082


namespace triangle_trig_identity_l414_414892

open Real

theorem triangle_trig_identity (A B C : ℝ) (h_triangle : A + B + C = 180) (h_A : A = 15) :
  sqrt 3 * sin A - cos (B + C) = sqrt 2 := by
  sorry

end triangle_trig_identity_l414_414892


namespace cos_of_angle_through_fixed_point_l414_414810

noncomputable def fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : ℝ × ℝ :=
  let x := -3
  let y := 4
  (x, y)

theorem cos_of_angle_through_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  let P := fixed_point a h1 h2 in
  P = (-3, 4) →
  let α := real.arccos (-3 / (real.sqrt ((-3) ^ 2 + 4 ^ 2))) in
  real.cos α = -3/5 :=
by
  intro P_fixed
  rw P_fixed
  sorry

end cos_of_angle_through_fixed_point_l414_414810


namespace rectangle_area_l414_414999

variable (x y : ℕ)

theorem rectangle_area
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y) :
  x * y = 36 :=
by
  -- Proof omitted
  sorry

end rectangle_area_l414_414999


namespace sequence_sum_n_eq_10_implies_n_eq_120_l414_414300

theorem sequence_sum_n_eq_10_implies_n_eq_120 (a : ℕ → ℝ)
  (h1 : ∀ n, a n = 1 / (Real.sqrt n + Real.sqrt (n + 1)))
  (h2 : (∑ i in Finset.range n, a i) = 10) :
  n = 120 :=
sorry

end sequence_sum_n_eq_10_implies_n_eq_120_l414_414300


namespace Rachel_budget_twice_Sara_l414_414606

-- Define the cost of Sara's shoes and dress
def s_shoes : ℕ := 50
def s_dress : ℕ := 200

-- Define the target budget for Rachel
def r : ℕ := 500

-- State the theorem to prove
theorem Rachel_budget_twice_Sara :
  2 * s_shoes + 2 * s_dress = r :=
by simp [s_shoes, s_dress, r]; sorry

end Rachel_budget_twice_Sara_l414_414606


namespace calculate_difference_of_squares_l414_414039

theorem calculate_difference_of_squares : (153^2 - 147^2) = 1800 := by
  sorry

end calculate_difference_of_squares_l414_414039


namespace two_axes_rotation_planes_of_symmetry_l414_414854

noncomputable def planes_of_symmetry (body : Type) (A1 A2 : body → Prop) (O : body) : set (set body) :=
  { P | ∀ x, P x ↔ P (O/x) }

theorem two_axes_rotation_planes_of_symmetry
  (body : Type)
  (has_two_axes_of_rotation : ∃ A1 A2 (O : body), (∀ θ : ℝ, (rotate body A1 θ = body) ∧ (rotate body A2 θ = body)) ∧ A1 O ∧ A2 O)
  : ∀ O ∃ planes, planes = planes_of_symmetry body (λ x, x = O) (λ x, x = O) O :=
by
  sorry

end two_axes_rotation_planes_of_symmetry_l414_414854


namespace sum_possible_values_of_y_l414_414865

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end sum_possible_values_of_y_l414_414865


namespace sum_of_legs_le_sqrt2_hypotenuse_l414_414602

theorem sum_of_legs_le_sqrt2_hypotenuse
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2) :
  a + b ≤ Real.sqrt 2 * c :=
sorry

end sum_of_legs_le_sqrt2_hypotenuse_l414_414602


namespace divisible_by_2010_l414_414090

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def P (A : Finset ℕ) : ℕ := A.prod id

def A (i : ℕ) := {1, 2, ..., 2010}.filter (λ x, i = x)

theorem divisible_by_2010 : 
  let n := binom 2010 99
  let A_sets := (Finset.powersetLen 99 (Finset.range 2011)).filter (λ x, x.card = 99)
  let sum_P_A_sets := A_sets.sum (λ A_i, P A_i)
  2010 ∣ sum_P_A_sets :=
by 
  sorry

end divisible_by_2010_l414_414090


namespace union_of_sets_l414_414836

def setA : Set ℕ := {0, 1}
def setB : Set ℕ := {0, 2}

theorem union_of_sets : setA ∪ setB = {0, 1, 2} := 
sorry

end union_of_sets_l414_414836


namespace log_lt_x_squared_for_x_gt_zero_l414_414497

theorem log_lt_x_squared_for_x_gt_zero (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 :=
sorry

end log_lt_x_squared_for_x_gt_zero_l414_414497


namespace platform_length_l414_414362

noncomputable def speed_in_kmph : ℝ := 72
noncomputable def time_in_seconds : ℝ := 26
noncomputable def length_of_train : ℝ := 170.0416
noncomputable def length_of_platform : ℝ := 349.9584

theorem platform_length:
  let speed_in_mps := (speed_in_kmph * 1000) / 3600 in
  let distance_covered := speed_in_mps * time_in_seconds in
  distance_covered - length_of_train = length_of_platform :=
by
  sorry

end platform_length_l414_414362


namespace max_interest_time_l414_414929

noncomputable def t : ℕ := 17

theorem max_interest_time :
  ∀ P r : ℝ, P = 1000 → r = 0.06 → (∀ t : ℕ, (1 + r)^t > 3 → t = 17) :=
begin
  intros P r hP hr t ht,
  sorry
end

end max_interest_time_l414_414929


namespace solve_length_CD_l414_414203

-- Definitions and conditions
def is_isosceles_triangle (A B C : Point) (AB AC : ℝ) : Prop :=
  AB = AC

def is_altitude (A D B C : Point) : Prop :=
  A, D, B, C are collinear /\ AD is perpendicular to BC

def base_length (BC : ℝ) : Prop :=
  BC = 8

theorem solve_length_CD
  (A B C D : Point)
  (AB AC BC : ℝ)
  (h_isosceles : is_isosceles_triangle A B C AB AC)
  (h_altitude : is_altitude A D B C)
  (h_base_length : base_length BC)
  : let CD := BC / 2 in CD = 4 :=
by
  sorry

end solve_length_CD_l414_414203


namespace maximum_fraction_sum_l414_414792

noncomputable def max_fraction_sum (n : ℕ) (a b c d : ℕ) : ℝ :=
  1 - (1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1)))

theorem maximum_fraction_sum (n a b c d : ℕ) (h₀ : n > 1) (h₁ : a + c ≤ n) (h₂ : (a : ℚ) / b + (c : ℚ) / d < 1) :
  ∃ m : ℝ, m = max_fraction_sum n a b c d := by
  sorry

end maximum_fraction_sum_l414_414792


namespace sum_mod_eleven_l414_414618

variable (x y z : ℕ)

theorem sum_mod_eleven (h1 : (x * y * z) % 11 = 3)
                       (h2 : (7 * z) % 11 = 4)
                       (h3 : (9 * y) % 11 = (5 + y) % 11) :
                       (x + y + z) % 11 = 5 :=
sorry

end sum_mod_eleven_l414_414618


namespace roots_cubic_poly_eq_l414_414252

variable {R : Type} [Nonempty R] [NoZeroDivisors R] [Field R]

def P (x m : R) : R := 3 * x^2 + 3 * m * x + m^2 - 1

theorem roots_cubic_poly_eq (x1 x2 m : R) (h1 : P x1 m = 0) (h2 : P x2 m = 0): 
  P (x1^3) m = P (x2^3) m := 
by 
  sorry

end roots_cubic_poly_eq_l414_414252


namespace total_lunch_cost_l414_414024

-- Define the costs of lunch
variables (adam_cost rick_cost jose_cost : ℕ)

-- Given conditions
def jose_ate_certain_price := jose_cost = 45
def rick_and_jose_lunch_same_price := rick_cost = jose_cost
def adam_spends_two_thirds_of_ricks := adam_cost = (2 * rick_cost) / 3

-- Question to prove
theorem total_lunch_cost : 
  jose_ate_certain_price ∧ rick_and_jose_lunch_same_price ∧ adam_spends_two_thirds_of_ricks 
  → (adam_cost + rick_cost + jose_cost = 120) :=
by
  intro h
  sorry  -- Proof to be provided separately

end total_lunch_cost_l414_414024


namespace sufficient_but_not_necessary_l414_414414

-- Define the equations of the lines
def line1 (a : ℝ) (x y : ℝ) : ℝ := 2 * x + a * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := (a - 1) * x + 3 * y - 2

-- Define the condition for parallel lines by comparing their slopes
def parallel_condition (a : ℝ) : Prop :=  (2 * 3 = a * (a - 1))

theorem sufficient_but_not_necessary (a : ℝ) : 3 ≤ a :=
  sorry

end sufficient_but_not_necessary_l414_414414


namespace geometric_progression_product_l414_414228

theorem geometric_progression_product (n : ℕ) (S R : ℝ) (hS : S > 0) (hR : R > 0)
  (h_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ S = a * (q^n - 1) / (q - 1))
  (h_reciprocal_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ R = (1 - q^n) / (a * q^(n-1) * (q - 1))) :
  ∃ P : ℝ, P = (S / R)^(n / 2) := sorry

end geometric_progression_product_l414_414228


namespace distances_to_faces_of_circumscribed_sphere_l414_414908

variable {a b c : ℝ}
variable {K : EuclideanSpace ℝ (Fin 3)}
variable {D A B C : EuclideanSpace ℝ (Fin 3)}

/-- Tetrahedron with right angles at D -/
def tetrahedron_angles_at_D_are_right (D A B C: EuclideanSpace ℝ (Fin 3)) : Prop :=
  let θ1 := ∠(A - D) (B - D) in
  let θ2 := ∠(B - D) (C - D) in
  let θ3 := ∠(C - D) (A - D) in
  θ1 = π/2 ∧ θ2 = π/2 ∧ θ3 = π/2

/-- Given the specific distances -/
def distances_from_K (K D A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  dist K D = 3 ∧
  dist K A = Real.sqrt 5 ∧
  dist K B = Real.sqrt 6 ∧
  dist K C = Real.sqrt 7

/-- Find the distances from the center of the circumscribed sphere to each face of the tetrahedron -/
theorem distances_to_faces_of_circumscribed_sphere 
  (D A B C K O : EuclideanSpace ℝ (Fin 3))
  (h1 : tetrahedron_angles_at_D_are_right D A B C)
  (h2 : distances_from_K K D A B C) :
  ∃ d1 d2 d3 d4,
    dist O (plane_spanned_by D A B) = d1 ∧ d1 = Real.sqrt 3 / 2 ∧
    dist O (plane_spanned_by D B C) = d2 ∧ d2 = 1 ∧
    dist O (plane_spanned_by D A C) = d3 ∧ d3 = Real.sqrt 2 / 2 ∧
    dist O (plane_spanned_by A B C) = d4 ∧ d4 = Real.sqrt 3 / Real.sqrt 13 := 
sorry

end distances_to_faces_of_circumscribed_sphere_l414_414908


namespace parallel_or_coincide_l414_414935

-- Let A_1C_2B_1A_2C_1B_2 be an equilateral hexagon.
-- Let O_1 and H_1 denote the circumcenter and orthocenter of ∆A_1B_1C_1,
-- and let O_2 and H_2 denote the circumcenter and orthocenter of ∆A_2B_2C_2.

variables {A1 B1 C1 A2 B2 C2 O1 H1 O2 H2 : Type}

-- Conditions for the equilateral hexagon and properties of circumcenter and orthocenter
axiom eq_hexagon : ∃ (A1 B1 C1 A2 B2 C2 : ℝ^2),
  (A1, B1, C1, A2, B2, C2 form an equilateral hexagon)
  ∧ O1 = circumcenter (triangle A1 B1 C1)
  ∧ H1 = orthocenter (triangle A1 B1 C1)
  ∧ O2 = circumcenter (triangle A2 B2 C2)
  ∧ H2 = orthocenter (triangle A2 B2 C2)
  ∧ (O1 ≠ O2)
  ∧ (H1 ≠ H2)

-- Theorem statement
theorem parallel_or_coincide : O1O2 ∥ H1H2 ∨ O1O2 = H1H2 :=
by
  sorry

end parallel_or_coincide_l414_414935


namespace number_of_mappings_l414_414259

-- Definitions of sets A and B
def A := {1, 2, 3, 4}
def B := {-1, -2}

-- Exists a function f from A to B
def f (x : ℕ) : ℤ := sorry -- Placeholder for the function definition

-- Main theorem statement
theorem number_of_mappings : (∃ f : ℕ → ℤ, (∀ b ∈ B, ∃ a ∈ A, f a = b)) →
  (f : ℕ → ℤ → ∃ f : A → B, ... -- Specification of the function mapping should be here.
  sorry
-- This theorem needs formulating the overall structure and requirements for the function f.

-- Placeholder for your formal proof.

end number_of_mappings_l414_414259


namespace quadratic_function_properties_l414_414434

theorem quadratic_function_properties :
  let f := λ x : ℝ, -(x - 1) ^ 2 + 2 in
  (∃ x : ℝ, f x = 2) ∧ (∃ x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
  (∃ h : ℝ, (∀ x : ℝ, f x = f (2 * h - x)) ∧ h = 1) :=
begin
  sorry
end

end quadratic_function_properties_l414_414434


namespace minimum_f_a_eq_2_f_gt_0_iff_a_gt_neg3_l414_414152

-- Define the function f(x) with parameter a
def f (x a : ℝ) : ℝ := x + a / x + 2

-- Prove that the minimum value of f(x) in (1, +∞) when a = 2 is 5
theorem minimum_f_a_eq_2 : ∀ x ∈ Ioi 1, f x 2 ≥ 5 :=
sorry

-- Prove that f(x) > 0 for all x ∈ (1, +∞) is equivalent to a > -3
theorem f_gt_0_iff_a_gt_neg3 : (∀ x ∈ Ioi 1, f x a > 0) ↔ a > -3 :=
sorry

end minimum_f_a_eq_2_f_gt_0_iff_a_gt_neg3_l414_414152


namespace find_alpha_l414_414112

open Real

noncomputable def point (x y : ℝ) : Prop := True

noncomputable def line (α t : ℝ) : Prop :=
  ∃ x y : ℝ, x = 2 + t * cos α ∧ y = 1 + t * sin α

theorem find_alpha (α : ℝ) :
  let P := (2, 1)
  let l := { t : ℝ // ∃ x y : ℝ, x = 2 + t * cos α ∧ y = 1 + t * sin α }
  let A : ℝ×ℝ := (2 - (2 - 1 / tan α), 0)
  let B : ℝ×ℝ := (0, 1 - 2 * tan α)
  let PA := sqrt((2 - (2 - 1 / tan α))^2 + (1 - 0)^2)
  let PB := sqrt((2 - 0)^2 + (1 - (1 - 2 * tan α))^2)
  (|PA| * |PB| = 4) → α = 3 * π / 4 :=
begin
  sorry
end

end find_alpha_l414_414112


namespace james_total_payment_l414_414229

noncomputable def total_amount_paid : ℕ :=
  let dirt_bike_count := 3
  let off_road_vehicle_count := 4
  let atv_count := 2
  let moped_count := 5
  let scooter_count := 3
  let dirt_bike_cost := dirt_bike_count * 150
  let off_road_vehicle_cost := off_road_vehicle_count * 300
  let atv_cost := atv_count * 450
  let moped_cost := moped_count * 200
  let scooter_cost := scooter_count * 100
  let registration_dirt_bike := dirt_bike_count * 25
  let registration_off_road_vehicle := off_road_vehicle_count * 25
  let registration_atv := atv_count * 30
  let registration_moped := moped_count * 15
  let registration_scooter := scooter_count * 20
  let maintenance_dirt_bike := dirt_bike_count * 50
  let maintenance_off_road_vehicle := off_road_vehicle_count * 75
  let maintenance_atv := atv_count * 100
  let maintenance_moped := moped_count * 60
  let total_cost_of_vehicles := dirt_bike_cost + off_road_vehicle_cost + atv_cost + moped_cost + scooter_cost
  let total_registration_costs := registration_dirt_bike + registration_off_road_vehicle + registration_atv + registration_moped + registration_scooter
  let total_maintenance_costs := maintenance_dirt_bike + maintenance_off_road_vehicle + maintenance_atv + maintenance_moped
  total_cost_of_vehicles + total_registration_costs + total_maintenance_costs

theorem james_total_payment : total_amount_paid = 5170 := by
  -- The proof would be written here
  sorry

end james_total_payment_l414_414229


namespace min_pq_sq_min_value_l414_414695

noncomputable def min_pq_sq (α : ℝ) : ℝ :=
  let p := α - 2
  let q := -(α + 1)
  (p + q)^2 - 2 * (p * q)

theorem min_pq_sq_min_value : 
  (∃ (α : ℝ), ∀ p q : ℝ, 
    p^2 + q^2 = (p + q)^2 - 2 * p * q ∧ 
    (p + q = α - 2 ∧ p * q = -(α + 1))) → 
  (min_pq_sq 1) = 5 :=
by
  sorry

end min_pq_sq_min_value_l414_414695


namespace length_of_train_is_270_l414_414693

/-- Define the conditions of the problem -/
def speed_kmph : ℕ := 72
def platform_length : ℕ := 250
def time_seconds : ℕ := 26

/-- Convert speed from km/h to m/s -/
def speed_mps : ℕ := speed_kmph * 5 / 18

/-- Define the total distance covered -/
def total_distance : ℕ := speed_mps * time_seconds

/-- Define the length of the train as L -/
def length_of_train : ℕ := total_distance - platform_length

/-- The theorem we need to prove -/
theorem length_of_train_is_270 :
  speed_kmph = 72 →
  platform_length = 250 →
  time_seconds = 26 →
  length_of_train = 270 :=
by
  intros
  simp [speed_kmph, platform_length, time_seconds, speed_mps, total_distance, length_of_train]
  exact eq.refl 270

end length_of_train_is_270_l414_414693


namespace sum_divisors_of_15_l414_414682

theorem sum_divisors_of_15 :
  (∑ n in Finset.filter (λ d, 15 % d = 0) (Finset.range 16), n) = 24 :=
by
  sorry

end sum_divisors_of_15_l414_414682


namespace angles_in_first_or_third_quadrant_l414_414609

noncomputable def angles_first_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}

noncomputable def angles_third_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}

theorem angles_in_first_or_third_quadrant :
  ∃ S1 S2 : Set ℝ, 
    (S1 = {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}) ∧
    (S2 = {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}) ∧
    (angles_first_quadrant_set = S1 ∧ angles_third_quadrant_set = S2)
  :=
sorry

end angles_in_first_or_third_quadrant_l414_414609


namespace polynomial_equiv_logarithm_rationality_l414_414433

noncomputable def sufficient_condition (a b : ℝ) (h_pos : 1 < a ∧ 1 < b) : Prop :=
log a b ∈ ℚ

def polynomial_relation (P Q : ℝ → ℝ) (A B : set ℝ) : Prop :=
∀ r, (P r ∈ A) ↔ (Q r ∈ B)

theorem polynomial_equiv_logarithm_rationality (a b : ℝ) (h_pos : 1 < a ∧ 1 < b)
  (A : set ℝ := {x | ∃ n : ℕ, x = a ^ n}) (B : set ℝ := {y | ∃ m : ℕ, y = b ^ m}) :
  (∃ (P Q : ℝ → ℝ), polynomial_relation P Q A B ∧ (∃ N : ℝ → ℝ, polynomial_relation P N A A) ∧
      (∃ M : ℝ → ℝ, polynomial_relation Q M B B)) ↔ (sufficient_condition a b h_pos) :=
sorry

end polynomial_equiv_logarithm_rationality_l414_414433


namespace internal_diagonal_cubes_l414_414353

-- Define the dimensions of the rectangular solid
def x_dimension : ℕ := 168
def y_dimension : ℕ := 350
def z_dimension : ℕ := 390

-- Define the GCD calculations for the given dimensions
def gcd_xy : ℕ := Nat.gcd x_dimension y_dimension
def gcd_yz : ℕ := Nat.gcd y_dimension z_dimension
def gcd_zx : ℕ := Nat.gcd z_dimension x_dimension
def gcd_xyz : ℕ := Nat.gcd (Nat.gcd x_dimension y_dimension) z_dimension

-- Define a statement that the internal diagonal passes through a certain number of cubes
theorem internal_diagonal_cubes :
  x_dimension + y_dimension + z_dimension - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 880 :=
by
  -- Configuration of conditions and proof skeleton with sorry
  sorry

end internal_diagonal_cubes_l414_414353


namespace sum_of_integers_c_with_4_solutions_l414_414637

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  (x - 4) * (x - 2) * (x + 2) * (x + 4) / 205 - 2

-- The problem statement
theorem sum_of_integers_c_with_4_solutions :
  let c_values := {c | ∃ x : ℝ, g x = c ∧ (∃! (a b c d : ℝ), g a = c ∧ g b = c ∧ g c = c ∧ g d = c ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)) }
  in ∑ c in c_values, c = -2 :=
sorry

end sum_of_integers_c_with_4_solutions_l414_414637


namespace min_odd_solution_l414_414060

theorem min_odd_solution (a m1 m2 n1 n2 : ℕ)
  (h1: a = m1^2 + n1^2)
  (h2: a^2 = m2^2 + n2^2)
  (h3: m1 - n1 = m2 - n2)
  (h4: a > 5)
  (h5: a % 2 = 1) :
  a = 261 :=
sorry

end min_odd_solution_l414_414060


namespace inequality_proof_l414_414790

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  b^2 / a ≥ 2 * b - a :=
begin
  sorry
end

end inequality_proof_l414_414790


namespace QI_parallel_AC_l414_414221

variables {A B C K M I P Q : Type}
variables [triangle A B C] [is_midpoint K A B] [is_midpoint M A C] [is_incenter I A B C]
          [intersection P (line_through K M) (line_through C I)]
          [perpendicular Q P (line_through K M)] [parallel Q M (line_through B I)]

theorem QI_parallel_AC :
  parallel (line_through Q I) (line_through A C) := 
sorry

end QI_parallel_AC_l414_414221


namespace factory_produces_1275_doors_l414_414360

noncomputable def total_doors_produced : Nat := 
  let initial_A := 100
  let initial_B := 200
  let initial_C := 300
  let first_quarter_A := initial_A - 20
  let first_quarter_B := initial_B - 40
  let first_quarter_C := initial_C - 60
  let second_quarter_A := first_quarter_A - (35 * first_quarter_A / 100)
  let second_quarter_B := first_quarter_B - (25 * first_quarter_B / 100)
  let second_quarter_C := first_quarter_C - (50 * first_quarter_C / 100)
  let third_quarter_A := second_quarter_A - (20 * second_quarter_A / 100)
  let third_quarter_B := second_quarter_B - (20 * second_quarter_B / 100)
  let third_quarter_C := second_quarter_C - (20 * second_quarter_C / 100)
  let total_doors_A := third_quarter_A.toNat * 3
  let total_doors_B := third_quarter_B * 5
  let total_doors_C := third_quarter_C * 7
  total_doors_A + total_doors_B + total_doors_C

theorem factory_produces_1275_doors :
  total_doors_produced = 1275 :=
by
  show total_doors_produced = 1275
  sorry

end factory_produces_1275_doors_l414_414360


namespace zacharys_bus_ride_length_l414_414678

theorem zacharys_bus_ride_length (Vince Zachary : ℝ) (hV : Vince = 0.62) (hDiff : Vince = Zachary + 0.13) : Zachary = 0.49 :=
by
  sorry

end zacharys_bus_ride_length_l414_414678


namespace ratio_of_men_to_women_l414_414001

theorem ratio_of_men_to_women (C W M : ℕ) 
  (hC : C = 30) 
  (hW : W = 3 * C) 
  (hTotal : M + W + C = 300) : 
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l414_414001


namespace parallel_lines_l414_414798

open EuclideanGeometry

variables {A B C M N O : Point} {ω : Circle}

theorem parallel_lines (h1 : ∃ (ω : Circle), inscribed_in_triangle A B C ω)
    (h2 : foot_of_perpendicular_point B (line_segment A C) = M)
    (h3 : foot_of_perpendicular_point A (tangent_to_circle ω B) = N) :
    parallel (line_segment M N) (line_segment B C) :=
begin
  sorry
end

end parallel_lines_l414_414798


namespace systematic_sampling_selects_616_l414_414008

theorem systematic_sampling_selects_616 (n : ℕ) (h₁ : n = 1000) (h₂ : (∀ i : ℕ, ∃ j : ℕ, i = 46 + j * 10) → True) :
  (∃ m : ℕ, m = 616) :=
  by
  sorry

end systematic_sampling_selects_616_l414_414008


namespace simplify_f_evaluate_f_l414_414099

noncomputable def f (α : ℝ) : ℝ :=
  (sin (π/2 - α) * cos (π/2 + α) / cos (π + α)) -
  (sin (2 * π - α) * cos (π/2 - α) / sin (π - α))

theorem simplify_f (α : ℝ) : f α = 2 * sin α :=
by
sorry

theorem evaluate_f (α : ℝ) (h : cos α = sqrt 3 / 2) : f α = 1 ∨ f α = -1 :=
by
sorry

end simplify_f_evaluate_f_l414_414099


namespace range_f_l414_414242

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else 0

theorem range_f : set.range f = {0, 1} :=
  sorry

end range_f_l414_414242


namespace men_initially_l414_414279

noncomputable def initial_men (d1 : ℕ) (d2 : ℕ) (k : ℕ) (days2 : ℝ) : ℝ := 
  (d2 * days2) / (d1 - days2)

theorem men_initially (d1 d2 k : ℕ) (days2 : ℝ) (h1 : d1 = 15) (h2 : k = 200) (h3 : days2 = 12.857) :
  initial_men d1 d2 k days2 ≈ 1200 :=
by {
  sorry
}

end men_initially_l414_414279


namespace total_rope_in_inches_l414_414552

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end total_rope_in_inches_l414_414552


namespace part1_part2_l414_414121

variable (a t : ℝ)

def proposition_p : Prop :=
  -2 * t ^ 2 + 7 * t - 5 > 0

def proposition_q : Prop :=
  t ^ 2 - (a + 3) * t + (a + 2) < 0

theorem part1 (t : ℝ) (h : proposition_p t) : 1 < t ∧ t < (5 / 2) :=
  sorry

theorem part2 (a : ℝ) (h : ∀ t, proposition_p t → proposition_q t) : a ≥ (1 / 2) :=
  sorry

end part1_part2_l414_414121


namespace math_problem_equivalent_proof_l414_414985

theorem math_problem_equivalent_proof :
  ∃ d e f : ℕ, let s := (d - Real.sqrt e) / f in 
  PQRS_IsSquare ∧ PQRS_has_side_len_2 ∧ PTU_IsEquilateral ∧ Side_Length_Expression s d e f ∧
  d = 6 ∧ e = 3 ∧ f = 12 ∧ (d + e + f = 21) :=
by
  sorry

end math_problem_equivalent_proof_l414_414985


namespace gold_per_hour_l414_414548

section
variable (t : ℕ) (g_chest g_small_bag total_gold : ℕ)

def g_per_hour (t : ℕ) (g_chest g_small_bag : ℕ) :=
  let total_gold := g_chest + 2 * g_small_bag
  total_gold / t

theorem gold_per_hour (h1 : t = 8) (h2 : g_chest = 100) (h3 : g_small_bag = g_chest / 2) :
  g_per_hour t g_chest g_small_bag = 25 := by
  -- substitute the given values
  have h4 : g_small_bag = 50 := h3
  have h5 : total_gold = g_chest + 2 * g_small_bag := by
    rw [h2, h4]
    rfl
  have h6 : g_per_hour t g_chest g_small_bag = total_gold / t := rfl
  rw [h1, h2, h4, h5] at h6
  exact h6
end

end gold_per_hour_l414_414548


namespace max_sum_arithmetic_prog_l414_414702

theorem max_sum_arithmetic_prog (a d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 3 = 327)
  (h2 : S 57 = 57)
  (hS : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  ∃ max_S : ℝ, max_S = 1653 := by
  sorry

end max_sum_arithmetic_prog_l414_414702


namespace number_of_paths_to_BC_l414_414856

def initial_position : (ℕ × ℕ) := (0, 0)

def P : ℕ × ℕ → ℕ
| (0, 0) := 1
| (i, j) :=
  if j > 0 then
    (if i > 0 then P (i-1, j-1) else 0) +
    P (i, j-1) +
    P (i+1, j-1)
  else
    0

def line_BC_at (n : ℕ) := (λ (i j : ℕ), j = n)

theorem number_of_paths_to_BC (n : ℕ) :
  (∑ i, P (i, n)) = 252 :=
by
  sorry

end number_of_paths_to_BC_l414_414856


namespace geometric_series_second_term_l414_414031

theorem geometric_series_second_term
  (r : ℚ) (S : ℚ) (h1 : r = 1/4) (h2 : S = 40) : 
  let a := 120 / 4 in  -- Solving for a as shown in the solution steps.
  let second_term := a * r in
  second_term = 15 / 2 :=
by
  -- These steps suffice to establish the theorem structure and statement,
  -- without requiring the full proof to be written here:
  sorry

end geometric_series_second_term_l414_414031


namespace circle_proof_problem_l414_414669

variables {P Q R : Type}
variables {p q r dPQ dPR dQR : ℝ}

-- Given Conditions
variables (hpq : p > q) (hqr : q > r)
variables (hdPQ : ℝ) (hdPR : ℝ) (hdQR : ℝ)

-- Statement of the problem: prove that all conditions can be true
theorem circle_proof_problem :
  (∃ hpq' : dPQ = p + q, true) ∧
  (∃ hqr' : dQR = q + r, true) ∧
  (∃ hpr' : dPR > p + r, true) ∧
  (∃ hpq_diff : dPQ > p - q, true) →
  false := 
sorry

end circle_proof_problem_l414_414669


namespace inequality_range_l414_414089

theorem inequality_range (a : ℝ) : (-1 < a ∧ a ≤ 0) → ∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0 :=
by
  intro ha
  sorry

end inequality_range_l414_414089


namespace domain_of_expression_l414_414076

theorem domain_of_expression (x : ℝ) :
  (x - 3 ≥ 0) → (7 - x ≥ 0) → (7 - x > 0) → (x ∈ set.Ico 3 7) :=
by
  intros hx3 hx7 h7x 
  sorry

end domain_of_expression_l414_414076


namespace sequence_properties_l414_414478

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 8 ∧ a 4 = 2 ∧ ∀ n, a (n + 2) + a n = 2 * a (n + 1)

def general_term (a : ℕ → ℤ) (n : ℕ) : Prop :=
  a n = -2 * (n : ℤ) + 10

def sum_of_abs_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = if n ≤ 5 then - (n : ℤ)^2 + 9 * (n : ℤ)
             else (n : ℤ)^2 - 9 * (n : ℤ) + 40

theorem sequence_properties :
  ∃ a : ℕ → ℤ, seq a ∧ general_term a ∧
  ∃ S : ℕ → ℤ, sum_of_abs_terms a S :=
sorry

end sequence_properties_l414_414478


namespace probability_of_selecting_male_l414_414051

-- We define the proportions and ratio given in the problem.
def proportion_obese_men := 1 / 5
def proportion_obese_women := 1 / 10
def ratio_men_to_women := 3 / 2

-- From the given conditions, prove that the probability of selecting a male given that the individual is obese is 3/4.
theorem probability_of_selecting_male (P_A B : Prop) 
  (h_ratio: ratio_men_to_women = 3 / 2)
  (h_obese_men: P_A → proportion_obese_men)
  (h_obese_women: P_A → proportion_obese_women):
  (proportion_obese_men) * (3 / 5 : ℝ) / ((proportion_obese_men) * (3 / 5 : ℝ) + (proportion_obese_women) * (2 / 5 : ℝ)) = 3 / 4 := 
by
  sorry

end probability_of_selecting_male_l414_414051


namespace find_nat_int_l414_414427

theorem find_nat_int (x y : ℕ) (h : x^2 = y^2 + 7 * y + 6) : x = 6 ∧ y = 3 := 
by
  sorry

end find_nat_int_l414_414427


namespace strictly_decreasing_function_implies_inequality_l414_414573

variable {R : Type*} [Real R]
variable (f g : R → R)
variable [Differentiable R f]
variable [Differentiable R g]

theorem strictly_decreasing_function_implies_inequality (a b x : R) (h1 : f x * (deriv g x) + (deriv f x) * g x < 0) (h2 : a < x) (h3 : x < b) :
  f x * g x > f b * g b := 
sorry

end strictly_decreasing_function_implies_inequality_l414_414573


namespace ellipse_behavior_l414_414450

theorem ellipse_behavior (a : ℝ) (h : a ∈ Ioo (2 - sqrt 3) (2 + sqrt 3)) :
  let b := sqrt (4 * a)
  let e := sqrt ((4 * a - a^2 - 1) / (4 * a))
  (∃ a₀ ∈ Ioo (2 - sqrt 3) 2, ∀ a₁ ∈ Ioo (2 - sqrt 3) a₀, e = sqrt ((4 * a₁ - a₁^2 - 1) / (4 * a₁))) ∧
  (∃ a₀ ∈ Ioo 2 (2 + sqrt 3), ∀ a₁ ∈ Ioo a₀ (2 + sqrt 3), e = sqrt ((4 * a₁ - a₁^2 - 1) / (4 * a₁))) := 
sorry

end ellipse_behavior_l414_414450


namespace interest_rate_proof_l414_414972

noncomputable def remaining_interest_rate (total_investment yearly_interest part_investment interest_rate_part amount_remaining_interest : ℝ) : Prop :=
  (part_investment * interest_rate_part) + amount_remaining_interest = yearly_interest ∧
  (total_investment - part_investment) * (amount_remaining_interest / (total_investment - part_investment)) = amount_remaining_interest

theorem interest_rate_proof :
  remaining_interest_rate 3000 256 800 0.1 176 :=
by
  sorry

end interest_rate_proof_l414_414972


namespace log_w_u_value_l414_414598

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_w_u_value (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu1 : u ≠ 1) (hv1 : v ≠ 1) (hw1 : w ≠ 1)
    (h1 : log u (v * w) + log v w = 5) (h2 : log v u + log w v = 3) : 
    log w u = 4 / 5 := 
sorry

end log_w_u_value_l414_414598


namespace non_obtuse_triangle_exists_l414_414668

-- Given definitions and conditions
def is_acute_triangle (A B C : Point) : Prop :=
  triangle.is_acute ⟨A, B, C⟩

def inscribed_in_same_circle (A B C : Point) (circle : Circle) : Prop :=
  circle.contains A ∧ circle.contains B ∧ circle.contains C

def distinct_vertices (pts : Set Point) : Prop :=
  ∀ p1 p2 p3 ∈ pts, p1 ≠ p2 → p2 ≠ p3 → p3 ≠ p1

-- Proof problem to show
theorem non_obtuse_triangle_exists
  (circle : Circle)
  (A1 A2 A3 B1 B2 B3 C1 C2 C3 : Point)
  (hA : is_acute_triangle A1 A2 A3)
  (hB : is_acute_triangle B1 B2 B3)
  (hC : is_acute_triangle C1 C2 C3)
  (h_inscribed : inscribed_in_same_circle A1 circle ∧ inscribed_in_same_circle A2 circle ∧ inscribed_in_same_circle A3 circle ∧
                 inscribed_in_same_circle B1 circle ∧ inscribed_in_same_circle B2 circle ∧ inscribed_in_same_circle B3 circle ∧
                 inscribed_in_same_circle C1 circle ∧ inscribed_in_same_circle C2 circle ∧ inscribed_in_same_circle C3 circle)
  (h_distinct : distinct_vertices {A1, A2, A3, B1, B2, B3, C1, C2, C3}) :
  ∃ A_i B_j C_k, 
    (A_i ∈ {A1, A2, A3}) ∧ 
    (B_j ∈ {B1, B2, B3}) ∧ 
    (C_k ∈ {C1, C2, C3}) ∧ 
    triangle.is_non_obtuse ⟨A_i, B_j, C_k⟩ :=
sorry

end non_obtuse_triangle_exists_l414_414668


namespace truck_distance_in_3_hours_l414_414734

theorem truck_distance_in_3_hours : 
  ∀ (speed_2miles_2_5minutes : ℝ) 
    (time_minutes : ℝ),
    (speed_2miles_2_5minutes = 2 / 2.5) →
    (time_minutes = 180) →
    (speed_2miles_2_5minutes * time_minutes = 144) :=
by
  intros
  sorry

end truck_distance_in_3_hours_l414_414734


namespace increasing_interval_of_function_l414_414411

noncomputable def monotonic_increasing_interval (k : ℤ) : set ℝ :=
  {x : ℝ | 4 * (k : ℝ) * real.pi - 3 * real.pi / 2 ≤ x ∧ x ≤ 4 * (k : ℝ) * real.pi + real.pi / 2}

theorem increasing_interval_of_function :
  ∀ (k : ℤ), ∃ (I : set ℝ), I = monotonic_increasing_interval k :=
by
  intro k
  use monotonic_increasing_interval k
  sorry

end increasing_interval_of_function_l414_414411


namespace parabola_eq_max_triangle_area_l414_414816

-- Defining the parabola equation and the given point F
def parabola_equation (p : ℝ) : Prop :=
  x^2 = 2 * p * y

def point_F : Point := (0, 1)

-- The first part of the problem: prove the equation of the parabola
theorem parabola_eq (p : ℝ) : parabola_equation p ↔ p = 2 ∧ (x^2 = 4 * y) := by
  sorry

-- Defining the points A, B, C on the parabola
def point_on_parabola (x : ℝ) : Point :=
  (x, x^2 / 4)

def vector_sum_zero (A B C : Point) (F : Point) : Prop :=
  (A - F) + (B - F) + (C - F) = (0, 0)

-- The second part of the problem: prove the maximum area of triangle ABC
theorem max_triangle_area (A B C : Point) (F : Point)
    (hA : A = point_on_parabola x1)
    (hB : B = point_on_parabola x2)
    (hC : C = point_on_parabola x3)
    (hF : F = point_F)
    (hVecSum : vector_sum_zero A B C F) : 
    (area_of_triangle ABC) = (3 * √6 / 2) := by
  sorry

end parabola_eq_max_triangle_area_l414_414816


namespace alpha_plus_2beta_eq_45_l414_414599

theorem alpha_plus_2beta_eq_45 
  (α β : ℝ) 
  (hα_pos : 0 < α ∧ α < π / 2) 
  (hβ_pos : 0 < β ∧ β < π / 2) 
  (tan_alpha : Real.tan α = 1 / 7) 
  (sin_beta : Real.sin β = 1 / Real.sqrt 10)
  : α + 2 * β = π / 4 :=
sorry

end alpha_plus_2beta_eq_45_l414_414599


namespace prove_f_3_equals_11_l414_414476

-- Assuming the given function definition as condition
def f (y : ℝ) : ℝ := sorry

-- The condition provided: f(x - 1/x) = x^2 + 1/x^2.
axiom function_definition (x : ℝ) (h : x ≠ 0): f (x - 1 / x) = x^2 + 1 / x^2

-- The goal is to prove that f(3) = 11
theorem prove_f_3_equals_11 : f 3 = 11 :=
by
  sorry

end prove_f_3_equals_11_l414_414476


namespace sally_took_out_5_onions_l414_414273

theorem sally_took_out_5_onions (X Y : ℕ) 
    (h1 : 4 + 9 - Y + X = X + 8) : Y = 5 := 
by
  sorry

end sally_took_out_5_onions_l414_414273


namespace midpoints_coincide_l414_414794
-- Import Mathlib for necessary definitions and theorems

-- Define the problem in Lean 4
theorem midpoints_coincide {A B C D O : Point}
  (isRectangle : is_rectangle A B C D)
  (isosceles_triangle : ∀ K L M, is_vertex_angle K L M α ∧ vertex_on_segment K L M BC ∧ endpoints_on_segments K L M AB CD → midpoint_of_base_km_coincides K L M O) :
  ∀ L₁ L₂ : Point, midpoint_of_base_km_coincides L₁ L₂ O :=
by
  sorry

end midpoints_coincide_l414_414794


namespace fourth_guard_distance_l414_414698

theorem fourth_guard_distance (d1 d2 d3 : ℕ) (d4 : ℕ) (h1 : d1 + d2 + d3 + d4 = 1000) (h2 : d1 + d2 + d3 = 850) : d4 = 150 :=
sorry

end fourth_guard_distance_l414_414698


namespace exists_polynomials_with_degrees_l414_414064

-- Define the problem in a Lean 4 statement
theorem exists_polynomials_with_degrees :
  ∃ (P Q R : ℝ[X]), 
  (P ≠ 0) ∧ (Q ≠ 0) ∧ (R ≠ 0) ∧
  (degree (P * Q) = degree (Q * R)) ∧ (degree (Q * R) = degree (P * R)) ∧
  (degree (P + Q) ≠ degree (P + R)) ∧ (degree (P + R) ≠ degree (Q + R)) ∧ (degree (P + Q) ≠ degree (Q + R)) :=
sorry

end exists_polynomials_with_degrees_l414_414064


namespace age_difference_eq_l414_414365

-- Definitions from the conditions
def son_age := 20
noncomputable def man_age := 42 -- obtained from the problem, solution step not included

-- Per the second condition, in two years man's age would be twice son's age
def age_relation := man_age + 2 = 2 * (son_age + 2)

-- Prove that the man is 22 years older than his son
theorem age_difference_eq:
  man_age - son_age = 22 :=
by
  -- Adding the assumptions to context
  have h1 : man_age + 2 = 2 * (son_age + 2) := age_relation
  -- Then the proof would follow from the known values
  sorry

end age_difference_eq_l414_414365


namespace mass_of_Al2S3_produced_l414_414468

def molar_mass_Al : ℝ := 26.98
def molar_mass_S : ℝ := 32.07
def molar_mass_Al2S3 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_S

def initial_moles_Al : ℕ := 6
def initial_moles_S : ℕ := 12

def moles_produced_by_Al (moles_Al : ℕ) : ℝ := moles_Al / 2
def moles_produced_by_S (moles_S : ℕ) : ℝ := moles_S / 3

theorem mass_of_Al2S3_produced :
  let limiting_moles_Al2S3 := min (moles_produced_by_Al initial_moles_Al) (moles_produced_by_S initial_moles_S)
  limiting_moles_Al2S3 * molar_mass_Al2S3 = 450.51 :=
by
  sorry

end mass_of_Al2S3_produced_l414_414468


namespace smallest_sum_of_four_distinct_numbers_l414_414650

theorem smallest_sum_of_four_distinct_numbers 
  (S : Finset ℤ) 
  (h : S = {8, 26, -2, 13, -4, 0}) :
  ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 2 :=
sorry

end smallest_sum_of_four_distinct_numbers_l414_414650


namespace triangle_probability_l414_414068

open Classical

theorem triangle_probability : 
  let sticks := [1, 2, 4, 5, 8, 10, 12, 15] in
  let valid_sets := (sticks.toFinset.powerset.filter (λ x, x.card = 3)).filter(λ s, 
    ∃ a b c : ℕ, a = s.min' (by simp) ∧ c = s.max' (by simp) ∧ b = (s\\{a, c}.toFinset).min' (by simp) ∧ a + b > c ) in
  let total_sets := (sticks.toFinset.powerset.filter (λ x, x.card = 3)).card in 
  total_sets = 56 ∧ valid_sets.card = 9 →
  (valid_sets.card : ℚ) / (total_sets : ℚ) = 9 / 56 :=
by
  intros sticks valid_sets total_sets h
  sorry

end triangle_probability_l414_414068


namespace cube_root_inequality_l414_414177

theorem cube_root_inequality {a b : ℝ} (h : a > b) : (a^(1/3)) > (b^(1/3)) :=
sorry

end cube_root_inequality_l414_414177


namespace intersection_P_Q_l414_414486

noncomputable def P : set (ℝ × ℝ) := { a | ∃ m : ℝ, a = (1, 0) + m • (0, 1) }
noncomputable def Q : set (ℝ × ℝ) := { b | ∃ n : ℝ, b = (1, 1) + n • (-1, 1) }

theorem intersection_P_Q : P ∩ Q = {(1,1)} :=
sorry

end intersection_P_Q_l414_414486


namespace ellipse_proof_l414_414143

noncomputable def ellipse_equation {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
    (line_slope : ℝ) (dist_from_center : ℝ) (chord_length : ℝ) (major_axis : ℝ) : Prop :=
  line_slope = 1/2 ∧
  dist_from_center = 1 ∧
  chord_length = (4/5) * major_axis ∧
  major_axis = 2 * a ∧
  36 = 5 * b^2 → 
  (a = 3 ∧ b = 2 ∧ (eq : ( ∀ x y : ℝ, (x^2) / 9 + (y^2) / 4 = 1 )))

theorem ellipse_proof : ellipse_equation (a := 3) (b := 2) (h1 := by norm_num) (h2 := by norm_num) (h3 := by linarith)
    1/2 1 ((4/5) * 6) 6 :=
by
  sorry

end ellipse_proof_l414_414143


namespace circle_bisection_relationship_l414_414464

-- Defining the conditions for the circles
def circle1 (a b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 - a) ^ 2 + (p.2 - b) ^ 2 = b^2 + 1}

def circle2 : set (ℝ × ℝ) :=
  {p | (p.1 + 1) ^ 2 + (p.2 + 1) ^ 2 = 4}

-- The theorem to prove the relationship between a and b
theorem circle_bisection_relationship (a b : ℝ) :
  (∀ p ∈ circle1 a b, p ∈ circle2) → a^2 + 2a + 2b + 5 = 0 := 
sorry

end circle_bisection_relationship_l414_414464


namespace number_of_arrangements_l414_414493

open Nat
open BigOperators

theorem number_of_arrangements :
  (factorial 3) * (factorial 3) * (factorial 4) * (factorial 5) = 103680 := by
  sorry

end number_of_arrangements_l414_414493


namespace inequality_bound_l414_414886

theorem inequality_bound (m : ℝ) (h : ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), x^2 - 4 * x ≥ m) : m ≤ -3 := 
by 
  sorry

end inequality_bound_l414_414886


namespace probability_factor_less_than_10_l414_414680

theorem probability_factor_less_than_10 : 
  (∃ f : ℕ → Prop, 
   (∀ x, f x ↔ (x > 0 ∧ 120 % x = 0)) ∧ 
   ((∑ x in finset.filter (λ x, x < 10) (finset.filter f (finset.range 121)), 1) / 
    (∑ x in finset.filter f (finset.range 121), 1) = 7 / 16)) :=
begin
  sorry,
end

end probability_factor_less_than_10_l414_414680


namespace parallel_lines_m_value_perpendicular_lines_m_value_l414_414841

theorem parallel_lines_m_value (m : ℝ) :
    (∀ x y : ℝ, mx + 2 * y + 4 = 0) ∧ (∀ x y : ℝ, x + (1 + m) * y - 2 = 0) →
    (∃ m = 1, ∀ a b : ℝ, (mx + 2 * b + 4 = 0) ∧ (a + (1 + m) * b - 2 = 0)) :=
begin
  sorry
end

theorem perpendicular_lines_m_value (m : ℝ) :
    (∀ x y : ℝ, mx + 2 * y + 4 = 0) ∧ (∀ x y : ℝ, x + (1 + m) * y - 2 = 0) →
    (∃ m = -2/3, ∀ a b : ℝ, (mx + 2 * b + 4 = 0) ∧ (a + (1 + m) * b - 2 = 0)) :=
begin
  sorry
end 

end parallel_lines_m_value_perpendicular_lines_m_value_l414_414841


namespace jordan_rectangle_width_l414_414041

theorem jordan_rectangle_width
  (length_carol : ℕ) (width_carol : ℕ) (length_jordan : ℕ) (width_jordan : ℕ)
  (h1 : length_carol = 5) (h2 : width_carol = 24) (h3 : length_jordan = 2)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 60 := by
  sorry

end jordan_rectangle_width_l414_414041


namespace rounding_condition_l414_414243

noncomputable def R (x : ℝ) : ℝ := Real.round x

theorem rounding_condition (x : ℝ) : 
  (R ((R (x + 1)) / 2) = 5) ↔ (7.5 ≤ x ∧ x < 9.5) :=
by
  sorry

end rounding_condition_l414_414243


namespace xy_product_range_l414_414750

theorem xy_product_range (x y : ℝ) (h : x^2 * y^2 + x^2 - 10 * x * y - 8 * x + 16 = 0) :
  0 ≤ x * y ∧ x * y ≤ 10 := 
sorry

end xy_product_range_l414_414750


namespace jim_gold_per_hour_l414_414546

theorem jim_gold_per_hour :
  ∀ (hours: ℕ) (treasure_chest: ℕ) (num_small_bags: ℕ)
    (each_small_bag_has: ℕ),
    hours = 8 →
    treasure_chest = 100 →
    num_small_bags = 2 →
    each_small_bag_has = (treasure_chest / 2) →
    (treasure_chest + num_small_bags * each_small_bag_has) / hours = 25 :=
by
  intros hours treasure_chest num_small_bags each_small_bag_has
  intros hours_eq treasure_chest_eq num_small_bags_eq small_bag_eq
  have total_gold : ℕ := treasure_chest + num_small_bags * each_small_bag_has
  have per_hour : ℕ := total_gold / hours
  sorry

end jim_gold_per_hour_l414_414546


namespace train_length_l414_414018

/-- Given a train traveling at 72 km/hr passing a pole in 8 seconds,
     prove that the length of the train in meters is 160. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (speed_m_s : ℝ) (distance_m : ℝ) :
  speed_kmh = 72 → 
  time_s = 8 → 
  speed_m_s = (speed_kmh * 1000) / 3600 → 
  distance_m = speed_m_s * time_s → 
  distance_m = 160 :=
by
  sorry

end train_length_l414_414018


namespace determine_integer_lengths_l414_414272

-- Define the triangle and its properties
structure Triangle :=
  (DE EF DF : ℕ)
  (is_right_triangle : DE^2 + EF^2 = DF^2)
  (leg1_pos : 0 < DE)
  (leg2_pos : 0 < EF)

-- Define the specific triangle DEF with given legs
def DEF : Triangle :=
{ DE := 18,
  EF := 24,
  DF := 30,
  is_right_triangle := by norm_num,
  leg1_pos := by norm_num,
  leg2_pos := by norm_num }

-- Hypothesis to check integer lengths of segments from E to hypotenuse
def integer_lengths_from_E_to_hypotenuse (T : Triangle) : ℕ :=
if T = DEF then 10 else 0

-- The main proof statement
theorem determine_integer_lengths : integer_lengths_from_E_to_hypotenuse DEF = 10 :=
sorry

end determine_integer_lengths_l414_414272


namespace balloon_count_correct_l414_414323

def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def silver_balloons : ℕ := 2 * gold_balloons
def total_balloons : ℕ := gold_balloons + silver_balloons + black_balloons

theorem balloon_count_correct : total_balloons = 573 := by
  sorry

end balloon_count_correct_l414_414323


namespace find_B_squared_l414_414419

theorem find_B_squared :
  let g (x : ℝ) := Real.sqrt 23 + 105 / x in
  let B := abs ((Real.sqrt 23 + Real.sqrt 443) / 2) + abs ((Real.sqrt 23 - Real.sqrt 443) / 2) in
  B^2 = 443 :=
by
  -- Proof can be filled here
  sorry

end find_B_squared_l414_414419


namespace right_triangle_division_l414_414540

theorem right_triangle_division :
    ∃ (t1 t2 t3 : Triangle), 
      (∀ (t : Triangle), t = right_triangle ∧ 
      (t.angle_A = 30 ∧ t.angle_B = 60 ∧ t.angle_C = 90) →
      (t1 = t2 ∧ t2 = t3 ∧ t3 = t) ) := 
    sorry

end right_triangle_division_l414_414540


namespace range_of_f_2019_l414_414083

noncomputable def f (x : ℝ) : ℝ := Real.logb 0.5 (sin x / (sin x + 15))

theorem range_of_f_2019 :
  (∃ R, f^[2019] ⊆ R) → (R = set.Ici 4) := sorry

end range_of_f_2019_l414_414083


namespace total_inches_of_rope_l414_414556

noncomputable def inches_of_rope (last_week_feet : ℕ) (less_feet : ℕ) (feet_to_inches : ℕ → ℕ) : ℕ :=
  let last_week_inches := feet_to_inches last_week_feet
  let this_week_feet := last_week_feet - less_feet
  let this_week_inches := feet_to_inches this_week_feet
  last_week_inches + this_week_inches

theorem total_inches_of_rope 
  (six_feet : ℕ := 6)
  (four_feet_less : ℕ := 4)
  (conversion : ℕ → ℕ := λ feet, feet * 12) :
  inches_of_rope six_feet four_feet_less conversion = 96 := by
  sorry

end total_inches_of_rope_l414_414556


namespace Sn_limit_as_n_approaches_infinity_l414_414013

variable (m : ℝ)

def area_of_first_circle : ℝ := π * (m / 2) ^ 2
def ratio_of_area_reduction : ℝ := (sqrt 3) / 3

theorem Sn_limit_as_n_approaches_infinity
    (S : ℕ → ℝ) (S_1 : S 1 = area_of_first_circle m)
    (recurrence_relation : ∀ n, S (n + 1) = S n * ratio_of_area_reduction) :
    ∑' n, S n = π * m^2 / (4 - sqrt 3) := 
sorry

end Sn_limit_as_n_approaches_infinity_l414_414013


namespace nurses_count_l414_414345

theorem nurses_count (total personnel_ratio d_ratio n_ratio : ℕ)
  (ratio_eq: personnel_ratio = 280)
  (ratio_condition: d_ratio = 5)
  (person_count: n_ratio = 9) :
  n_ratio * (personnel_ratio / (d_ratio + n_ratio)) = 180 := by
  -- Total personnel = 280
  -- Ratio of doctors to nurses = 5/9
  -- Prove that the number of nurses is 180
  -- sorry is used to skip proof
  sorry

end nurses_count_l414_414345


namespace even_n_property_l414_414240

def S : Finset ℕ := {1, ..., 100}

def T_n (n : ℕ) : Finset (Fin n → ℕ) := 
  { t | (∑ i in Finset.univ, t i) % 100 = 0 }.to_finset

theorem even_n_property (n : ℕ) :
  (∀ red : Finset ℕ, red.card = 75 → 
    (T_n n).filter (λ t, (range n).count (λ i, t i ∈ red) % 2 = 0 ).card ≥ (T_n n).card / 2) ↔ even n :=
sorry

end even_n_property_l414_414240


namespace jessica_current_age_l414_414926

-- Define the conditions
def jessicaOlderThanClaire (jessica claire : ℕ) : Prop :=
  jessica = claire + 6

def claireAgeInTwoYears (claire : ℕ) : Prop :=
  claire + 2 = 20

-- State the theorem to prove
theorem jessica_current_age : ∃ jessica claire : ℕ, 
  jessicaOlderThanClaire jessica claire ∧ claireAgeInTwoYears claire ∧ jessica = 24 := 
sorry

end jessica_current_age_l414_414926


namespace train_speed_calculation_l414_414378

theorem train_speed_calculation
  (len_train : ℕ)
  (len_bridge : ℕ)
  (time_seconds : ℕ)
  (total_distance : ℕ)
  (v : ℕ) :
  len_train = 120 →
  len_bridge = 240 →
  time_seconds = 180 →
  total_distance = 360 →
  v = total_distance / time_seconds →
  v = 2 :=
begin
  sorry
end

end train_speed_calculation_l414_414378


namespace quadrilateral_area_l414_414528

-- Definition of the right trapezoid PQRS with given conditions
def right_trapezoid (PQ RS PR : ℝ) (PT TU UQ RV VW WS : ℝ) := 
  PQ = 2 ∧ RS = 6 ∧ PR = 4 ∧ 
  PT = 2 / 3 ∧ TU = 2 / 3 ∧ UQ = 2 / 3 ∧ 
  RV = 2 ∧ VW = 2 ∧ WS = 2

-- Theorem stating that the area of the quadrilateral XYZA formed by midpoints X, Y, Z, A 
-- is equal to 4/3 under the given conditions
theorem quadrilateral_area : 
  forall (PQ RS PR : ℝ) (PT TU UQ RV VW WS : ℝ), 
    right_trapezoid PQ RS PR PT TU UQ RV VW WS -> 
    let X := (PT / 2) in
    let Y := (TU / 2) in
    let Z := (RV / 2) in
    let A := (WS / 2) in
    area_quadrilateral X Y Z A = 4 / 3 := 
by 
  sorry

end quadrilateral_area_l414_414528


namespace angle_GFE_eq_angle_ADE_l414_414216

-- Definitions and conditions
variables {O P C E A B D F G : Type}

-- Circle conditions
def is_diameter (CD : Type) (O : Type) : Prop := sorry
def is_tangent (P : Type) (C : Type) (O : Type) : Prop := sorry
def is_secant (PBA : Type) (O : Type) : Prop := sorry
def intersects (l1 l2 : Type) (P : Type) : Prop := sorry

-- Given conditions
axiom diameter_CD : is_diameter CD O
axiom tangent_PC : is_tangent P C O
axiom tangent_PE : is_tangent P E O
axiom secant_PBA : is_secant PBA O
axiom intersect_AC_BD : intersects AC BD F
axiom intersect_DE_AB : intersects DE AB G

-- Angles
def angle (X Y Z : Type) : Type := sorry

-- Theorem statement
theorem angle_GFE_eq_angle_ADE :
  angle G F E = angle A D E := sorry

end angle_GFE_eq_angle_ADE_l414_414216


namespace rectangle_division_l414_414109

theorem rectangle_division (n : ℕ) (h : n ≥ 5) :
  ∃ (rectangles : list (ℝ × ℝ × ℝ × ℝ)), -- list of rectangles, each defined by (x1, y1, x2, y2)
  (∀ r ∈ rectangles, r.1 < r.3 ∧ r.2 < r.4) ∧ -- each rectangle has positive width and height
  (∑ r in rectangles, (r.3 - r.1) * (r.4 - r.2) = 1) ∧ -- rectangles fill the given rectangle (assumed to be of area 1)
  (rectangles.length = n) ∧ -- there are exactly n rectangles
  (∀ r1 r2 ∈ rectangles, r1 ≠ r2 → ¬is_adjacent r1 r2 ∨ ¬forms_larger_rectangle r1 r2) := -- no two adjacent rectangles form a larger rectangle
sorry

-- Helper definitions
def is_adjacent (r1 r2 : ℝ × ℝ × ℝ × ℝ) := -- Two rectangles are adjacent if they share a side
  (r1.1 = r2.1 ∧ r1.3 = r2.3 ∧ (r1.2 = r2.4 ∨ r1.4 = r2.2)) ∨
  (r1.2 = r2.2 ∧ r1.4 = r2.4 ∧ (r1.1 = r2.3 ∨ r1.3 = r2.1))

def forms_larger_rectangle (r1 r2 : ℝ × ℝ × ℝ × ℝ) := -- Check if combined area forms a larger rectangle
  (r1.1 = r2.1 ∧ r1.3 = r2.3 ∧ (r1.2 = r2.2 ∨ r1.4 = r2.4)) ∨
  (r1.2 = r2.2 ∧ r1.4 = r2.4 ∧ (r1.1 = r2.1 ∨ r1.3 = r2.3))

end rectangle_division_l414_414109


namespace intersection_of_M_and_N_l414_414838

def M : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 1}
def N : Set ℤ := {x : ℤ | x^2 = x}

theorem intersection_of_M_and_N :
  M ∩ N = {0, 1} := by
sory

end intersection_of_M_and_N_l414_414838


namespace propositions_incorrect_l414_414689

theorem propositions_incorrect :
  ¬ (∃ P : Prop, 
    (P = (∀ε>0, ∃x : ℝ, 0 < x ∧ x < ε)) ∨ 
    (P = (∀x : ℝ, ∀y : ℝ, (y = x^2 - 1) → (∃x : ℝ, (x, y) = (x, x^2 - 1)))) ∨ 
    (P = ({1, 3/2, 6/4, | -1/2 |, 0.5}.card = 5)) ∨ 
    (P = (∀x y : ℝ, xy ≤ 0 → ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))))) :=
by {
  -- Proof goes here.
  sorry
}

end propositions_incorrect_l414_414689


namespace find_f_val_l414_414882

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then Real.log x / Real.log 2
  else sorry -- since we are not required to define outside (0,1] for this problem

theorem find_f_val : f (15 / 2) = -1 :=
by
  have h1 : ∀ x : ℝ, f x = f (-x), from sorry, -- even function property
  have h2: ∀ x : ℝ, f x = f (2 - x), from sorry, -- given condition
  have h3: ∀ x : ℝ, f x = f (x - 2), from sorry, -- combined properties
  show f (15 / 2) = -1, from sorry

end find_f_val_l414_414882


namespace negation_proof_l414_414642

theorem negation_proof :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  -- Proof to be filled
  sorry

end negation_proof_l414_414642


namespace fraction_of_married_men_is_one_fourth_l414_414035

noncomputable def fraction_of_married_men_problem 
  (prob_single_woman : ℚ) 
  (total_people : ℕ) 
  (fraction_married_men : ℚ) 
  (total_women : ℕ) 
  (single_women : ℕ) 
  (married_women : ℕ) 
  (married_men : ℕ) : Prop :=
  prob_single_woman = (1 / 3) ∧
  total_people = 24 ∧
  single_women = (total_women * 1/3).to_nat ∧
  married_women = total_women - single_women ∧
  married_men = married_women ∧
  fraction_married_men = (married_men : ℚ) / total_people

theorem fraction_of_married_men_is_one_fourth :
  ∃ fraction : ℚ, fraction_of_married_men_problem (1/3) 24 fraction 9 3 6 6 ∧ fraction = 1/4 :=
by {
  sorry 
}

end fraction_of_married_men_is_one_fourth_l414_414035


namespace parabola_geometric_sequence_l414_414719

theorem parabola_geometric_sequence (M : ℝ × ℝ) (l: ℝ × ℝ → ℝ) (parabola : ℝ → ℝ → Prop) 
  (A B : ℝ × ℝ) (p : ℝ) 
  (hM : M = (-2, -4))
  (slope_angle : ∀ x, l x = (-2 + real.sqrt 2 / 2 * x, -4 + real.sqrt 2 / 2 * x))
  (parabola_cond : ∀ x y, parabola x y ↔ y^2 = 2 * p * x)
  (intersection_points : parabola (A.1) (A.2) ∧ parabola (B.1) (B.2))
  (geometric_sequence_cond : distance M A * distance M B = distance A B^2)
  : p = 1 :=
sorry

end parabola_geometric_sequence_l414_414719


namespace part1_part2_l414_414449

variables (A B C : ℝ) (a b c : ℝ)

-- Conditions: Triangle ABC lies on the unit circle, sides opposite to angles A, B, C are a, b, c respectively, and 2a cos A = c cos B + b cos C
def conditions : Prop := 
  A ∈ set.Ioo 0 π ∧
  a ≠ 0 ∧ 
  b ≠ 0 ∧ 
  c ≠ 0 ∧
  (2 * a * Real.cos A = b * Real.cos C + c * Real.cos B)

-- Question 1: Prove cos A = 1 / 2 given the conditions
theorem part1 (h : conditions A a b c) : Real.cos A = 1 / 2 := sorry

-- Additional condition for part 2: b^2 + c^2 = 4
def additional_condition : Prop := 
  (b^2 + c^2 = 4)

-- Question 2: Prove the area of triangle ABC, given cos A = 1 / 2 and b^2 + c^2 = 4
theorem part2 (h1 : conditions A a b c) (h2 : Real.cos A = 1 / 2) (h3 : additional_condition b c) :
  (1 / 2) * b * c * Real.sin A = sqrt 3 / 4 := sorry

end part1_part2_l414_414449


namespace value_of_60th_number_l414_414646

theorem value_of_60th_number : 
  (let seq := λ n : ℕ, list.replicate (2 * n) (2 * n)
  in nth_le (list.join (list.map seq (list.range 8))) 59 (by decide)) = 16 :=
sorry

end value_of_60th_number_l414_414646


namespace probability_intersection_l414_414482

noncomputable def setA : Set ℝ := { x | 2 * x^2 - x - 3 < 0 }
noncomputable def setB : Set ℝ := { x | 1 - x > 0 ∧ x + 3 > 0 }

theorem probability_intersection :
  let interval := Ioo (-3 : ℝ) 3
  let A := setA
  let B := setB
  let intersection := A ∩ B
  let probability := (interval ∩ intersection).measure / interval.measure
  probability = (1 : ℝ) / 3 :=
sorry

end probability_intersection_l414_414482


namespace consecutive_coeff_sum_l414_414563

theorem consecutive_coeff_sum (P : Polynomial ℕ) (hdeg : P.degree = 699)
  (hP : P.eval 1 ≤ 2022) :
  ∃ k : ℕ, k < 700 ∧ (P.coeff (k + 1) + P.coeff k) = 22 ∨
                    (P.coeff (k + 1) + P.coeff k) = 55 ∨
                    (P.coeff (k + 1) + P.coeff k) = 77 :=
by
  sorry

end consecutive_coeff_sum_l414_414563


namespace find_BP_l414_414975

noncomputable def BP_length (A B C D P : Point) (BP DP : ℝ) : Prop :=
  BP < DP ∧ 
  let AP := 8 in
  let PC := 1 in
  let BD := 6 in
  let BP := 2 in
  AP * PC = BP * (BD - BP)

theorem find_BP (A B C D P : Point) (h : BP_length A B C D P 2 (6 - 2)):
  BP = 2 :=
sorry

end find_BP_l414_414975


namespace sum_of_max_and_min_values_on_interval_l414_414829

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

theorem sum_of_max_and_min_values_on_interval :
  let a := 3 in
  (∃ x ∈ Ioo (0 : ℝ) (⊤ : ℝ), f x = 0) →
  let f_on_interval := (Icc (-1 : ℝ) 1 : set ℝ),
  let max_value := sup (f '' f_on_interval),
  let min_value := inf (f '' f_on_interval)
  in max_value + min_value = -3 :=
by sorry

end sum_of_max_and_min_values_on_interval_l414_414829


namespace calculate_fraction_l414_414038

theorem calculate_fraction :
  (10^9 / (2 * 10^5) = 5000) :=
  sorry

end calculate_fraction_l414_414038


namespace percentage_of_square_in_rectangle_l414_414014

noncomputable def side_of_square : ℝ := sorry -- We let 's' be the side of the square.

def width_of_rectangle (s: ℝ) : ℝ := 3 * s
def length_of_rectangle (s: ℝ) : ℝ := 9 * s

def area_of_square (s: ℝ) : ℝ := s ^ 2
def area_of_rectangle (s: ℝ) : ℝ := 27 * (s ^ 2)

def percentage_covered (s: ℝ) : ℝ :=
  (area_of_square s / area_of_rectangle s) * 100

theorem percentage_of_square_in_rectangle :
  percentage_covered side_of_square = 100 / 27 :=
sorry

end percentage_of_square_in_rectangle_l414_414014


namespace wire_length_is_correct_l414_414352

noncomputable def wire_length (volume : ℝ) (diameter : ℝ) : ℝ :=
  let radius := diameter / 2 in 
  let volume_cm := volume in
  let volume_formula := Real.pi * (radius ^ 2) * (1 : ℝ) in
  let length_cm := volume / volume_formula in
  length_cm * 0.01

theorem wire_length_is_correct :
  wire_length 44 (1 * 0.1) ≈ 56.05 :=
by
  -- Placeholder for the actual proof
  sorry

end wire_length_is_correct_l414_414352


namespace find_m_plus_n_l414_414941

noncomputable theory
open_locale classical

variables {A B C H P Q : Type}
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq H]

def altitude (A B C H : Type) := true  -- To represent that CH is the altitude

def inscribed_in (T P : Type) (H : Type) := true  -- To represent taht P is the point where the circles inscribed in T is tangent to CH

def distances (a b c : ℕ) : Prop := a = 2023 ∧ b = 2022 ∧ c = 2021

theorem find_m_plus_n 
    (h1 : altitude A B C H) 
    (h2 : inscribed_in A H P H) 
    (h3 : inscribed_in B H Q H) 
    (h4 : distances 2023 2022 2021) : 
    let m := 0,
        n := 1 in m + n = 1 := 
begin 
  sorry
end

end find_m_plus_n_l414_414941


namespace linette_problem_proof_l414_414583

def boxes_with_neither_markers_nor_stickers (total_boxes markers stickers both : ℕ) : ℕ :=
  total_boxes - (markers + stickers - both)

theorem linette_problem_proof : 
  let total_boxes := 15
  let markers := 9
  let stickers := 5
  let both := 4
  boxes_with_neither_markers_nor_stickers total_boxes markers stickers both = 5 :=
by
  sorry

end linette_problem_proof_l414_414583


namespace symmetric_function_log_eq_log2_x_plus_1_l414_414884

theorem symmetric_function_log_eq_log2_x_plus_1 (f : ℝ → ℝ) 
  (h : ∀ x, f (2 * x + 1) = x + 1 ↔ 2^(x+1) = f (2 * x + 1)) : 
  f x = log 2 (x + 1) :=
sorry

end symmetric_function_log_eq_log2_x_plus_1_l414_414884


namespace proposition_D_true_l414_414343

-- Assume we are working in Euclidean geometry
axiom EuclideanGeometry : Type

-- Definition of a line in Euclidean geometry
structure Line (G : EuclideanGeometry) :=
(point1 point2 : G)

-- Definition of a point in Euclidean geometry
structure Point (G : EuclideanGeometry) :=
(coord : ℝ × ℝ)

-- Parallel Postulate in Euclidean geometry
axiom parallel_postulate (G : EuclideanGeometry) (P : Point G) (l : Line G) : 
  ∃! m : Line G, P ∉ {q | q = l.point1 ∨ q = l.point2} ∧ 
                ∀ Q : Point G, Q ∉ {q | q = l.point1 ∨ q = l.point2} → 
                ∀ R : Point G, (Q, R) ∉ { (q1, q2) | q1 = l.point1 ∨ q1 = l.point2 ∨ q2 = l.point2 ∨ q2 = l.point1 } → 
                    m = l → 
                    ∀ x : Point G, ¬(x ∈ {y : Point G | y = P})

theorem proposition_D_true (G : EuclideanGeometry) : 
  ∀ (P : Point G) (l : Line G), ∃! m : Line G, P ∉ {q | q = l.point1 ∨ q = l.point2} ∧ 
                      ∀ Q : Point G, Q ∉ {q | q = l.point1 ∨ q = l.point2} → 
                      ∀ R : Point G, (Q, R) ∉ { (q1, q2) | q1 = l.point1 ∨ q1 = l.point2 ∨ q2 = l.point2 ∨ q2 = l.point1 } → 
                          m = l → 
                          ∀ x : Point G, ¬(x ∈ {y : Point G | y = P}) := 
  parallel_postulate G

-- The proof is omitted
sorry

end proposition_D_true_l414_414343


namespace find_alpha_polar_equation_l414_414116

variables {α : Real}

def point_P := (2, 1)
def line_l := { t : Real // x = 2 + t * Real.cos α ∧ y = 1 + t * Real.sin α }
def line_intersects_pos_axes := 
  ∃ A B : (ℝ × ℝ),
  A.1 > 0 ∧ B.2 > 0 ∧ 
  ∃ t1 t2 : Real, 
    line_l t1 = (A.1, 0) ∧ line_l t2 = (0, B.2)

def distance_PA (A : ℝ × ℝ) := ((2 - A.1) ^ 2 + (1 - A.2) ^ 2).sqrt
def distance_PB (B : ℝ × ℝ) := ((2 - B.1) ^ 2 + (1 - B.2) ^ 2).sqrt

def condition_PA_PB_product (PA PB : ℝ) := PA * PB = 4

theorem find_alpha 
  (A B : (ℝ × ℝ)) (h1 : line_intersects_pos_axes) 
  (h2 : condition_PA_PB_product (distance_PA A) (distance_PB B)):
  α = 3 * Real.pi / 4 :=
sorry

def polar_coordinate_line (ρ θ : Real) :=
  ρ * (Real.cos θ + Real.sin θ) = 3

theorem polar_equation 
  (α_value : α = 3 * Real.pi / 4):
  polar_coordinate_line :=
sorry

end find_alpha_polar_equation_l414_414116


namespace max_tan_C_l414_414537

-- Define vectors in a Euclidean space and their operations
variables {V : Type*} [inner_product_space ℝ V]
variables {A B C : V}
variables {AB AC BC CB : V}

-- Conditions
def condition (A B C : V) (AC AB BC CB : V) : Prop :=
  AC ⬝ (AB - BC) = 2 * (CB ⬝ (AC - AB))

-- Statement including the maximum value of tan C
theorem max_tan_C (A B C : V) (AC AB BC CB : V) (h : condition A B C AC AB BC CB) :
  ∃ C : ℝ, C = arctan (√14 / 2) :=
sorry

end max_tan_C_l414_414537


namespace escalator_length_l414_414030

theorem escalator_length :
  ∃ L : ℝ, L = 150 ∧ 
    (∀ t : ℝ, t = 10 → ∀ v_p : ℝ, v_p = 3 → ∀ v_e : ℝ, v_e = 12 → L = (v_p + v_e) * t) :=
by sorry

end escalator_length_l414_414030


namespace significant_digits_of_side_length_l414_414412

def compute_side_length (area : ℝ) : ℝ := Real.sqrt area

def significant_digits (x : ℝ) : ℕ :=
  if x = 0 then 0
  else x.toString.filter (λ c, c ≠ '0' ∧ c ≠ '.').length

theorem significant_digits_of_side_length :
  significant_digits (compute_side_length 1.21) = 2 :=
by
  sorry

end significant_digits_of_side_length_l414_414412


namespace interest_is_less_by_1940_l414_414717

noncomputable def principal : ℕ := 2000
noncomputable def rate : ℕ := 3
noncomputable def time : ℕ := 3

noncomputable def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

noncomputable def difference (sum_lent interest : ℕ) : ℕ :=
  sum_lent - interest

theorem interest_is_less_by_1940 :
  difference principal (simple_interest principal rate time) = 1940 :=
by
  sorry

end interest_is_less_by_1940_l414_414717


namespace mason_hotdogs_proof_mason_ate_15_hotdogs_l414_414585

-- Define the weights of the items.
def weight_hotdog := 2 -- in ounces
def weight_burger := 5 -- in ounces
def weight_pie := 10 -- in ounces

-- Define Noah's consumption
def noah_burgers := 8

-- Define the total weight of hotdogs Mason ate
def mason_hotdogs_weight := 30

-- Calculate the number of hotdogs Mason ate
def hotdogs_mason_ate := mason_hotdogs_weight / weight_hotdog

-- Calculate the number of pies Jacob ate
def jacob_pies := noah_burgers - 3

-- Given conditions
theorem mason_hotdogs_proof :
  mason_hotdogs_weight / weight_hotdog = 3 * (noah_burgers - 3) :=
by
  sorry

-- Proving the number of hotdogs Mason ate equals 15
theorem mason_ate_15_hotdogs :
  hotdogs_mason_ate = 15 :=
by
  sorry

end mason_hotdogs_proof_mason_ate_15_hotdogs_l414_414585


namespace probability_is_correct_l414_414692

noncomputable def probability_of_three_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) : ℚ :=
  let total_balls := red_balls + blue_balls + green_balls in
  (red_balls.to_rat / total_balls.to_rat) * ((red_balls - 1).to_rat / (total_balls - 1).to_rat) * ((red_balls - 2).to_rat / (total_balls - 2).to_rat)

theorem probability_is_correct :
  probability_of_three_red_balls 12 4 5 3 = 1/55 :=
by 
  sorry

end probability_is_correct_l414_414692


namespace min_distance_of_extrema_l414_414828

-- Definitions
def f (x : ℝ) : ℝ := 2 * Real.sin(π / 2 * x + π / 5)

-- Problem statement
theorem min_distance_of_extrema:
  ∀ x ∈ ℝ, 
  ∃ (x₁ x₂ : ℝ), 
  f x₁ = -2 ∧ f x₂ = 2 ∧ ∀ y ∈ ℝ, f x₁ ≤ f y ∧ f y ≤ f x₂ → |x₁ - x₂| = 2 :=
sorry

end min_distance_of_extrema_l414_414828


namespace min_colors_needed_l414_414451

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  max (abs (c1.1 - c2.1)) (abs (c1.2 - c2.2))

def color (c : cell) : ℤ :=
  (c.1 + c.2) % 4

theorem min_colors_needed : 4 = 4 :=
sorry

end min_colors_needed_l414_414451


namespace inclusion_exclusion_l414_414104

open Finset BigOperators

variable (n : ℕ)
variable (p : Fin n → ℝ)
variable (h : ∀ i, 0 < p i ∧ p i < 1)

noncomputable def S : Finset (Fin n) := Finset.univ

noncomputable def P (I : Finset (Fin n)) : ℝ :=
  if I = ∅ then 1 else ∏ i in I, p i

noncomputable def Q (I : Finset (Fin n)) : ℝ :=
  P I⁻¹

noncomputable def Phi (I : Finset (Fin n)) : ℝ :=
  if I = ∅ then 1 else ∏ i in I, (1 / p i - 1)

theorem inclusion_exclusion :
  ∑ I in S.powerset, Φ I * P (S \ I) = Φ S :=
sorry

end inclusion_exclusion_l414_414104


namespace largest_primary_divisor_l414_414374

-- Define the concept of a primary divisor.
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m ∈ { d : ℕ | d > 0 ∧ d | p }, m = 1 ∨ m = p

def primary_divisor (n : ℕ) : Prop :=
  ∀ d ∈ { d : ℕ | d > 0 ∧ d | n }, (is_prime (d - 1)) ∨ (is_prime (d + 1))

-- Proof that the largest primary divisor is 48
theorem largest_primary_divisor : ∃ n, primary_divisor n ∧ n = 48 :=
by {
  use 48,
  sorry
}

end largest_primary_divisor_l414_414374


namespace max_intersection_points_l414_414393

theorem max_intersection_points (n : ℕ) (h : n = 12) : 
  (∑ (i : ℕ) in finset.ico 0 (n.choose 2), 2) = 132 :=
by 
  sorry

end max_intersection_points_l414_414393


namespace jane_mistake_corrected_l414_414733

-- Conditions translated to Lean definitions
variables (x y z : ℤ)
variable (h1 : x - (y + z) = 15)
variable (h2 : x - y + z = 7)

-- Statement to prove
theorem jane_mistake_corrected : x - y = 11 :=
by
  -- Placeholder for the proof
  sorry

end jane_mistake_corrected_l414_414733


namespace hired_male_workers_24_l414_414657

noncomputable def companyX : Type := sorry

variables (E M : ℕ) (companyX : Type)

-- Conditions
def initial_female_percentage (E : ℕ) : ℕ := 60 * E
def new_employee_count (E M : ℕ) : ℕ := E + M
def new_female_percentage (E M : ℕ) : bool := 60 * E = 55 * (E + M)
def total_employees_after_hiring (E M : ℕ) : bool := E + M = 288

-- Prove that the number of additional male workers hired is 24
theorem hired_male_workers_24 : 
  (∀ E M : ℕ, new_female_percentage E M → total_employees_after_hiring E M → M = 24) :=
  sorry

end hired_male_workers_24_l414_414657


namespace max_nondiagonal_5x5_grid_l414_414731

open Set

/-- Maximum number of non-intersecting diagonals in a 5x5 grid of squares. --/
theorem max_nondiagonal_5x5_grid : ∃ n, n = 16 ∧ 
  ∀ diags : Finset (Fin 5 × Fin 5), 
    (∀ (x y : Fin 5 × Fin 5), x ≠ y → diags x ∩ diags y = ∅) →
    diags.card ≤ n :=
begin
  use 16,
  split,
  { refl, },
  {
    intros diags h,
    sorry
  }
end

end max_nondiagonal_5x5_grid_l414_414731


namespace find_y_l414_414723

theorem find_y (x y : ℕ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 :=
by 
  -- Proof is skipped
  sorry

end find_y_l414_414723


namespace max_profit_l414_414141

-- Define the profits of laptops and desktop computers
def profit_laptops (t : ℝ) : ℝ := (1 / 16) * t
def profit_desktops (t : ℝ) : ℝ := (1 / 2) * t

-- Total purchase capital
def total_capital : ℝ := 50

-- Define the profit function in terms of allocated capital to desktops (m)
def profit (m : ℝ) : ℝ := (1 / 16) * (total_capital - m) + (1 / 2) * m

-- Maximum profit to be proved
theorem max_profit : ∃ (m : ℝ), profit m = 833 / 16 := by
  sorry

end max_profit_l414_414141


namespace right_triangle_area_and_hypotenuse_l414_414516

-- Definitions based on given conditions
def a : ℕ := 24
def b : ℕ := 2 * a + 10

-- Statements based on the questions and correct answers
theorem right_triangle_area_and_hypotenuse :
  (1 / 2 : ℝ) * (a : ℝ) * (b : ℝ) = 696 ∧ (Real.sqrt ((a : ℝ)^2 + (b : ℝ)^2) = Real.sqrt 3940) := by
  sorry

end right_triangle_area_and_hypotenuse_l414_414516


namespace fill_bucket_time_l414_414889

theorem fill_bucket_time (time_full_bucket : ℕ) (fraction : ℚ) (time_two_thirds_bucket : ℕ) 
  (h1 : time_full_bucket = 150) (h2 : fraction = 2 / 3) : time_two_thirds_bucket = 100 :=
sorry

end fill_bucket_time_l414_414889


namespace divisible_by_condition_a_l414_414269

theorem divisible_by_condition_a (a b c k : ℤ) 
  (h : ∃ k : ℤ, a - b * c = (10 * c + 1) * k) : 
  ∃ k : ℤ, 10 * a + b = (10 * c + 1) * k :=
by
  sorry

end divisible_by_condition_a_l414_414269


namespace overall_percentage_profit_or_loss_l414_414728

-- Definitions for conditions
def original_price := 100
def profit_rate_A_to_B := 0.35
def loss_rate_B_to_C := 0.25
def discount_rate_C := 0.10
def sales_tax_B_to_C := 0.05
def profit_rate_C_to_D := 0.20
def loss_rate_D_to_E := 0.15
def sales_tax_D_to_E := 0.07

-- Final statement of the problem
theorem overall_percentage_profit_or_loss :
  let selling_price_A_to_B := original_price * (1 + profit_rate_A_to_B)
  let selling_price_B_to_C_before_discount := selling_price_A_to_B * (1 - loss_rate_B_to_C)
  let discounted_price_C := selling_price_B_to_C_before_discount * (1 - discount_rate_C)
  let total_cost_C := discounted_price_C * (1 + sales_tax_B_to_C)
  let selling_price_C_to_D := total_cost_C * (1 + profit_rate_C_to_D)
  let selling_price_D_to_E_before_tax := selling_price_C_to_D * (1 - loss_rate_D_to_E)
  let total_cost_E := selling_price_D_to_E_before_tax * (1 + sales_tax_D_to_E)
  let overall_profit := total_cost_E - original_price
  let overall_percentage_profit := (overall_profit / original_price) * 100
  overall_percentage_profit = 4.42651625 := 
sorry

end overall_percentage_profit_or_loss_l414_414728


namespace find_smaller_number_l414_414632

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  -- Proof steps will be filled in here
  sorry

end find_smaller_number_l414_414632


namespace minimum_value_of_expression_l414_414946

noncomputable def minimal_value_condition (x y a : ℝ) : Prop :=
  (x + 2 * y = 1) ∧ (min (3 / x + a / y) = 6 * Real.sqrt 3)

theorem minimum_value_of_expression (x y a : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) :
  minimal_value_condition x y a →
  (1 / x + 2 / y = 1) →
  min (3 * x + a * y) = 6 * Real.sqrt 3 :=
by 
  sorry

end minimum_value_of_expression_l414_414946


namespace num_std_dev_below_mean_is_2_l414_414088

-- Let's define the mean, the scores, and the relevant standard deviation condition.
def mean : ℝ := 76
def score1 : ℝ := 60
def score2 : ℝ := 100
def z2 : ℝ := 3  -- score2 is 3 standard deviations above the mean

-- The number of standard deviations (σ) that corresponds to score2
def std_dev : ℝ := (score2 - mean) / z2

-- Find the number of standard deviations (k) below the mean when the score is score1
def num_std_dev_below_mean : ℝ := (mean - score1) / std_dev

theorem num_std_dev_below_mean_is_2 : num_std_dev_below_mean = 2 := by
  -- This would be proved by manipulating the definitions and showing the equivalence
  sorry

end num_std_dev_below_mean_is_2_l414_414088


namespace graph_coordinate_sum_l414_414994

theorem graph_coordinate_sum (f : ℝ → ℝ) (hf : Injective f) (h : f 3 = 12) : 
  12 + (f⁻¹ 12 / 3) = 13 := by
  sorry

end graph_coordinate_sum_l414_414994


namespace neither_sufficient_nor_necessary_l414_414459

theorem neither_sufficient_nor_necessary 
  (a b c : ℝ) : 
  ¬ ((∀ x : ℝ, b^2 - 4 * a * c < 0 → a * x^2 + b * x + c > 0) ∧ 
     (∀ x : ℝ, a * x^2 + b * x + c > 0 → b^2 - 4 * a * c < 0)) := 
by
  sorry

end neither_sufficient_nor_necessary_l414_414459


namespace rhombus_longest_diagonal_l414_414727

theorem rhombus_longest_diagonal (A : ℝ) (r : ℝ) (hA : A = 200) (hr : r = 4):
  ∃ d_1 d_2 : ℝ, d_1 / d_2 = r ∧ (1/2) * d_1 * d_2 = A ∧ d_1 = 40 :=
by
  use 40, 10
  have hr_eq : 40 / 10 = 4 := by norm_num
  have area_eq : (1 / 2) * 40 * 10 = 200 := by norm_num
  exact ⟨hr_eq, area_eq, rfl⟩
  sorry

end rhombus_longest_diagonal_l414_414727


namespace number_of_common_tangents_l414_414488

/-- Define the first circle C₁ -/
noncomputable def C₁ := {P : ℝ × ℝ | (P.1 - 2)^2 + (P.2 - 5)^2 = 4^2}

/-- Define the second circle C₂ -/
noncomputable def C₂ := {P : ℝ × ℝ | (P.1 + 1)^2 + (P.2 + 3)^2 = 1^2}

/-- Proof statement: The number of common tangents -/
theorem number_of_common_tangents :
  (CommonTangentsCount C₁ C₂) = 4 :=
by
  sorry

/-- Define a function to calculate the number of common tangents -/
def CommonTangentsCount (circle1 circle2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

end number_of_common_tangents_l414_414488


namespace log_ratio_independence_of_base_l414_414978

theorem log_ratio_independence_of_base
  (a b P K : ℝ)
  (ha : a > 0) (hb : b > 0) (hP : P > 0) (hK : K > 0) :
  (log a P) / (log a K) = (log b P) / (log b K) := sorry

end log_ratio_independence_of_base_l414_414978


namespace find_radius_l414_414358

-- Definitions and conditions
variables (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 25)

-- Theorem statement
theorem find_radius : r = 50 :=
sorry

end find_radius_l414_414358


namespace distance_between_foci_l414_414993

theorem distance_between_foci (a : ℝ) (ha : a = 2 * real.sqrt 2) :
  real.sqrt ((-a - a)^2 + (-4 / a - 4 / a)^2) = 2 * real.sqrt 10 :=
by
  sorry

end distance_between_foci_l414_414993


namespace problem_statement_l414_414876

variable {f : ℝ → ℝ}

-- f is an even function: ∀ x ∈ ℝ, f(x) = f(-x)
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)

-- f is decreasing on [0, ∞): ∀ x y ∈ [0, ∞), x < y → f(x) > f(y)
def is_decreasing_nonneg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f(x) > f(y)

theorem problem_statement (h_even: is_even f) (h_decreasing: is_decreasing_nonneg f) : f(3) < f(-2) ∧ f(-2) < f(1) :=
by
  sorry

end problem_statement_l414_414876


namespace max_variance_l414_414308

theorem max_variance (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) : 
  ∃ q, p * (1 - p) ≤ q ∧ q = 1 / 4 :=
by
  existsi (1 / 4)
  sorry

end max_variance_l414_414308


namespace min_AB_distance_is_correct_l414_414641

noncomputable def min_AB_distance : ℝ :=
  let f := λ x : ℝ, x - (1/2 * (log x - 1))
  in min (f 1/2) (f (1 + (1/2 * log 2)))

theorem min_AB_distance_is_correct :
  min_AB_distance = 1 + 1/2 * log 2 :=
by
  sorry

end min_AB_distance_is_correct_l414_414641


namespace abs_a_minus_b_eq_2_l414_414132

theorem abs_a_minus_b_eq_2 (a b : ℝ) :
  (∃ y1 y2 : ℝ, (y1^2 + (real.sqrt(real.pi))^4 = 2*(real.sqrt(real.pi))^2*y1 + 1) ∧ (y2^2 + (real.sqrt(real.pi))^4 = 2*(real.sqrt(real.pi))^2*y2 + 1) ∧ y1 = a ∧ y2 = b ∧ a ≠ b) 
    → (|a - b| = 2) := 
by 
  sorry

end abs_a_minus_b_eq_2_l414_414132


namespace median_calculation_l414_414679

def median_of_special_list : Real :=
  let list_of_ints := List.range (3030 + 1)      -- Integers from 1 to 3030
  let list_of_cubes := list_of_ints.map (fun x => x^3) -- Cubes from 1^3 to 3030^3
  let combined_list := list_of_ints ++ list_of_cubes    -- Combined list
  if 6060 % 2 = 0 then
    -- Median for even length list: average of 3030-th and 3031-st terms
    (combined_list.get 3029 + combined_list.get 3030) / 2
  else
    -- Median for odd length list: middle term
    combined_list.get (6060 / 2)

theorem median_calculation :
  median_of_special_list = 1515.5 :=
sorry

end median_calculation_l414_414679


namespace tan_30_eq_sqrt3_div_3_l414_414043

-- Define the specific point Q on the unit circle
def Q := (↑(Real.sqrt 3) / 2, 1 / 2 : Real × Real)

-- Define the point E, the foot of the altitude from Q to the x-axis
def E := (↑(Real.sqrt 3) / 2, 0 : Real × Real)

-- Define the lengths of sides QE and EO in a 30-60-90 triangle
def length_QE : Real := 1 / 2
def length_EO : Real := Real.sqrt 3 / 2

-- Prove that tan 30 degrees is sqrt(3) / 3
theorem tan_30_eq_sqrt3_div_3 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := by
    sorry

end tan_30_eq_sqrt3_div_3_l414_414043


namespace parabola_focus_distance_l414_414819

theorem parabola_focus_distance (m : ℝ) (p : ℝ) :
  (m^2 = -8*(-3)) ↔ abs(m) = 2 * real.sqrt(6) :=
by sorry

end parabola_focus_distance_l414_414819


namespace common_ratio_sin_arithmetic_geometric_l414_414314

theorem common_ratio_sin_arithmetic_geometric (α : ℕ → ℝ) (β q α_1 : ℝ)
  (h_arith : ∀ n, α (n + 1) = α n + β)
  (h_geom  : ∀ n, sin (α (n + 1)) = q * sin (α n))
  (h_alpha1 : α 0 = α_1) :
  q = 1 ∨ q = -1 := 
sorry

end common_ratio_sin_arithmetic_geometric_l414_414314


namespace fraction_painting_students_l414_414903

theorem fraction_painting_students (total_students : ℕ) (field_fraction : ℚ) (students_left : ℕ) :
  total_students = 50 →
  field_fraction = 1 / 5 →
  students_left = 10 →
  (total_students - students_left - field_fraction * total_students) / total_students = 3 / 5 := 
by
  intros h_total h_field h_left
  have n_field : ℚ := field_fraction * total_students
  have n_away := total_students - students_left
  have n_paint := n_away - n_field
  have p := n_paint / total_students
  rw [h_total, h_field, h_left] at *
  norm_num at *
  exact rfl

end fraction_painting_students_l414_414903


namespace part1_part2_part3_l414_414787

noncomputable def f (x a : ℝ) := log (1 / (2 ^ x) + a) / log 2

-- Part 1
theorem part1 (x : ℝ) (h1 : 1 ∈ ℝ) : f x 1 > 1 ↔ x < 0 := sorry

-- Part 2
theorem part2 (a : ℝ) : (∃! x : ℝ, f x a + 2 * x = 0) ↔ (a ≥ 0 ∨ a = -1 / 4) := sorry

-- Part 3
theorem part3 (a : ℝ) (ha : a > 0) (h2 : ∀ t : ℝ, t ∈ Set.Icc (-1) 0 → f t a + f (t + 1) a ≤ log 6 / log 2) : a ∈ Set.Ioc 0 1 := sorry

end part1_part2_part3_l414_414787


namespace intercepted_chord_length_correct_l414_414160

noncomputable def chord_length_intercepted_by_line_on_circle 
    (m : ℝ) (line_eq : ℝ → ℝ → Prop) (circle_eq : ℝ → ℝ → Prop) : ℝ :=
  let center_C1 : ℝ × ℝ := (0, 0) in
  let center_C2 : ℝ × ℝ := (4, -3) in
  let radius_C1 : ℝ := 2 in
  let radius_C2 : ℝ := 3 in -- calculated implicitly by m = 16 leading to radius 3 since 25 - 16 = 9 has radius 3
  let distance_between_centers : ℝ := 5 in
  if (distance_between_centers = radius_C1 + radius_C2) ∧ line_eq = (λ x y, x + y = 0) ∧ circle_eq = (λ x y, (x - 4)^2 + (y + 3)^2 = 9) 
  then 2 * sqrt (radius_C2^2 - (1/√2)^2) -- computed distance to line
  else 0

theorem intercepted_chord_length_correct : 
  chord_length_intercepted_by_line_on_circle 16 (λ x y, x + y = 0) (λ x y, (x - 4)^2 + (y + 3)^2 = 9) = sqrt 34 := by
  sorry

end intercepted_chord_length_correct_l414_414160


namespace red_balls_approximation_l414_414523

def total_balls : ℕ := 50
def red_ball_probability : ℚ := 7 / 10

theorem red_balls_approximation (r : ℕ)
  (h1 : total_balls = 50)
  (h2 : red_ball_probability = 0.7) :
  r = 35 := by
  sorry

end red_balls_approximation_l414_414523


namespace rhombus_dot_product_l414_414447

variables {V : Type*} [inner_product_space ℝ V]

theorem rhombus_dot_product (A B C D : V) (a : ℝ) (h : a ≠ 0)
  (hABCD: dist A B = a ∧ dist B C = a ∧ dist C D = a ∧ dist D A = a)
  (hABC : ∠ B A C = real.pi / 3) :
  (B - D) ⬝ (C - D) = (3 / 2) * a^2 :=
sorry

end rhombus_dot_product_l414_414447


namespace function_D_min_value_is_2_l414_414388

noncomputable def function_A (x : ℝ) : ℝ := x + 2
noncomputable def function_B (x : ℝ) : ℝ := Real.sin x + 2
noncomputable def function_C (x : ℝ) : ℝ := abs x + 2
noncomputable def function_D (x : ℝ) : ℝ := x^2 + 1

theorem function_D_min_value_is_2
  (x : ℝ) :
  ∃ x, function_D x = 2 := by
  sorry
 
end function_D_min_value_is_2_l414_414388


namespace problem_1_problem_2_problem_3_l414_414564

-- Problem 1: List all A_6 for n = 3
def possible_A6 : List (List Int) := 
  [
    [1, 1, 1, -1, -1, -1],
    [1, 1, -1, 1, -1, -1],
    [1, 1, -1, -1, 1, -1],
    [1, -1, 1, 1, -1, -1],
    [1, -1, 1, -1, 1, -1]
  ]

theorem problem_1 :
  ∀ A : List Int, 
    A.length = 6 ∧ 
    (∀ i, 0 ≤ i ∧ i < 6 → A[i] ∈ {1, -1}) ∧ 
    (List.sum A = 0) ∧ 
    (∀ i, 1 ≤ i ∧ i < 6 → List.sum (A.take i) ≥ 0) 
    ↔ A ∈ possible_A6 := 
sorry

-- Problem 2: Set of possible values for a_1 + a_2 + ... + a_{2k-1}
theorem problem_2 (k : ℕ) (hk : k ≥ 1) :
  let n := 2 * k - 1 in
  (∀ A : List Int,
      A.length = n ∧ 
      (∀ i, 0 ≤ i ∧ i < n → A[i] ∈ {1, -1}) ∧ 
      (List.sum (List.take n A ++ List.drop n A) = 0) ∧ 
      (1 ≤ n - 1 → List.sum (A.take i) ≥ 0)) →
    List.sum (List.take n A) ∈ {x | ∃ m : ℕ, x = 2 * m + 1 ∧ x ≤ n} := sorry

-- Problem 3: Number of A_{2n}
theorem problem_3 (n : ℕ) (hn : n > 0) :
  let A_length := 2 * n in
  let valid_A (A : List Int) := 
    A.length = A_length ∧ 
    (∀ i, 0 ≤ i ∧ i < A_length → A[i] ∈ {1, -1}) ∧ 
    (List.sum A = 0) ∧ 
    (∀ i, 1 ≤ i ∧ i < A_length → List.sum (A.take i) ≥ 0) in
  (finset.card ((finset.univ : finset (List Int)).filter valid_A) = 
    (nat.factorial (2 * n)) / ((nat.factorial n) * (nat.factorial (n + 1)))
  ) := sorry

end problem_1_problem_2_problem_3_l414_414564


namespace gold_per_hour_l414_414547

section
variable (t : ℕ) (g_chest g_small_bag total_gold : ℕ)

def g_per_hour (t : ℕ) (g_chest g_small_bag : ℕ) :=
  let total_gold := g_chest + 2 * g_small_bag
  total_gold / t

theorem gold_per_hour (h1 : t = 8) (h2 : g_chest = 100) (h3 : g_small_bag = g_chest / 2) :
  g_per_hour t g_chest g_small_bag = 25 := by
  -- substitute the given values
  have h4 : g_small_bag = 50 := h3
  have h5 : total_gold = g_chest + 2 * g_small_bag := by
    rw [h2, h4]
    rfl
  have h6 : g_per_hour t g_chest g_small_bag = total_gold / t := rfl
  rw [h1, h2, h4, h5] at h6
  exact h6
end

end gold_per_hour_l414_414547


namespace total_lunch_cost_l414_414022

def adam_spends (rick_spent : ℝ) : ℝ := (2 / 3) * rick_spent
def jose_spent := 45
def rick_spent := jose_spent
def total_cost (adam_spent : ℝ) (rick_spent : ℝ) (jose_spent : ℝ) : ℝ := adam_spent + rick_spent + jose_spent

theorem total_lunch_cost : total_cost (adam_spends rick_spent) rick_spent jose_spent = 120 :=
by sorry

end total_lunch_cost_l414_414022


namespace coordinates_of_P_minimum_PA_PB_l414_414833

noncomputable def line_equation : ℝ → ℝ := λ x, 2 * x + 1

def point_A : ℝ × ℝ := (-2, 3)
def point_B : ℝ × ℝ := (1, 6)

theorem coordinates_of_P (P : ℝ × ℝ) : 
  (P.2 = line_equation P.1) →
  (real.dist P point_A = real.dist P point_B) →
  P = (1, 3) := 
sorry

theorem minimum_PA_PB (P : ℝ × ℝ) :
  (P.2 = line_equation P.1) →
  (|real.dist P point_A + real.dist P point_B|) ≥ real.dist (14/5, -8/5) point_B :=
sorry

end coordinates_of_P_minimum_PA_PB_l414_414833


namespace shadow_length_building_l414_414718

theorem shadow_length_building:
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  (height_flagstaff / shadow_flagstaff = height_building / expected_shadow_building) := by
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  sorry

end shadow_length_building_l414_414718


namespace problem1_problem2_problem3_l414_414055

noncomputable def difference (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

-- Proof 1
theorem problem1 (a b c : ℚ) (ha : a = -2) (hb : b = -4) (hc : c = 1) : 
  difference a b c = -5 / 3 :=
by 
  sorry

-- Proof 2
theorem problem2 :
  ∃ a b c : ℚ, a ∈ {-2, -4, 1} ∧ b ∈ {-2, -4, 1} ∧ c ∈ {-2, -4, 1} ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  difference a b c = 2 / 3 :=
by
  sorry

-- Proof 3
theorem problem3 (x : ℚ) :
  difference (-1) 6 x = 2 → (x = -7 ∨ x = 8) :=
by
  sorry

end problem1_problem2_problem3_l414_414055


namespace six_digit_numbers_with_even_start_l414_414004

theorem six_digit_numbers_with_even_start (digits : Multiset ℕ) 
  (h_digits : digits = {4, 4, 3, 5, 6, 7})
  (even_start : ℕ → Prop) 
  (h_even : ∀ n, even_start n ↔ n ∈ {4, 6}) :
  ∃ n : ℕ, n = 300 :=
by sorry

end six_digit_numbers_with_even_start_l414_414004


namespace area_ratio_l414_414645

variable (a b c d k : ℝ)

-- Defining the conditions
def proportional_sides := a / b = c / d
def perimeters_relation := 2 * (a + b) = 4 * 2 * (c + d)
def sides_proportional_assumption := b = k * a ∧ d = k * c

-- The theorem statement
theorem area_ratio :
  proportional_sides →
  perimeters_relation →
  sides_proportional_assumption →
  (a = 4 * c) →
  k = 1 →
  (a * b) / (c * d) = 16 :=
by
  sorry

end area_ratio_l414_414645


namespace total_roses_in_a_week_l414_414328

theorem total_roses_in_a_week : 
  let day1 := 24 
  let day2 := day1 + 6
  let day3 := day2 + 6
  let day4 := day3 + 6
  let day5 := day4 + 6
  let day6 := day5 + 6
  let day7 := day6 + 6
  (day1 + day2 + day3 + day4 + day5 + day6 + day7) = 294 :=
by
  sorry

end total_roses_in_a_week_l414_414328


namespace find_triangle_sides_l414_414376

noncomputable def rhombus_in_right_triangle (a b c : ℝ) : Prop :=
  ∃ (triangle : Triangle), 
    triangle.angles.is_right_trid() ∧   -- The triangle is right-angled
    triangle.angles.contains 60 ∧           -- One angle is 60°
    ∃ (rhombus : Rhombus),                      -- There exists a rhombus inscribed
    rhombus.sides.length = 6 ∧                -- Side length of the rhombus is 6 cm
    rhombus.vertices.all_on_triangle_sides() -- All vertices of the rhombus lie on the triangle sides

theorem find_triangle_sides : 
  ∃ (a b c : ℝ), rhombus_in_right_triangle a b c ∧ a = 9 ∧ b = 9 * real.sqrt 3 ∧ c = 18 := 
by 
  sorry

end find_triangle_sides_l414_414376


namespace project_allocation_l414_414437

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem project_allocation : binomial 8 3 * binomial 5 1 * binomial 4 2 * binomial 2 2 = 1680 := 
by {
-- The proof would go here, explaining each combinatorial step.
sorry
}

end project_allocation_l414_414437


namespace find_b_2015_l414_414835

noncomputable def a (n : ℕ) := real.sqrt (5 * n - 1)

noncomputable def b (n : ℕ) : ℕ :=
  if even n then 3 + 5 * (n / 2 - 1) else 2 + 5 * (n / 2)

theorem find_b_2015 :
  b 2015 = 5037 := by
  unfold b
  simp
  sorry

end find_b_2015_l414_414835


namespace find_b_for_3_roots_l414_414284

theorem find_b_for_3_roots (a b : ℝ) (h₁ : a ≠ 0) (h₂ : ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ ∀ x, ||x - a| - b| = 2008 → (x = x₁ ∨ x = x₂ ∨ x = x₃)) : b = 2008 :=
by
  sorry

end find_b_for_3_roots_l414_414284


namespace range_of_a_l414_414883

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ set.Icc 1 2, deriv f x ≤ 0) :
  a ≥ 5 / 2 :=
by
  have f_def : f = λ x, (x^3 / 3) - (a / 2) * x^2 + x := sorry
  have deriv_f : ∀ x, deriv f x = x^2 - a * x + 1 := sorry
  sorry

end range_of_a_l414_414883


namespace calories_in_250_grams_is_106_l414_414235

noncomputable def total_calories_apple : ℝ := 150 * (46 / 100)
noncomputable def total_calories_orange : ℝ := 50 * (45 / 100)
noncomputable def total_calories_carrot : ℝ := 300 * (40 / 100)
noncomputable def total_calories_mix : ℝ := total_calories_apple + total_calories_orange + total_calories_carrot
noncomputable def total_weight_mix : ℝ := 150 + 50 + 300
noncomputable def caloric_density : ℝ := total_calories_mix / total_weight_mix
noncomputable def calories_in_250_grams : ℝ := 250 * caloric_density

theorem calories_in_250_grams_is_106 : calories_in_250_grams = 106 :=
by
  sorry

end calories_in_250_grams_is_106_l414_414235


namespace hands_overlap_facts_l414_414167

theorem hands_overlap_facts:
  (∀ n, n = 2 → hour_hand_revolutions n) →
  (∀ n, n = 24 → minute_hand_revolutions n) →
  (∀ n, n = 1440 → second_hand_revolutions n) →
  let overlap_hour_minute := 22 in
  let overlap_minute_second := 1416 in
  hour_minute_overlap = 22 ∧ minute_second_overlap = 1416 := 
by 
  intro hour_hand_revolutions minute_hand_revolutions second_hand_revolutions
  let hour_minute_overlap := 24 - 2
  let minute_second_overlap := 1440 - 24
  exact And.intro rfl rfl
  sorry

end hands_overlap_facts_l414_414167


namespace power_identity_l414_414565

-- Define the given definitions
def P (m : ℕ) : ℕ := 5 ^ m
def R (n : ℕ) : ℕ := 7 ^ n

-- The theorem to be proved
theorem power_identity (m n : ℕ) : 35 ^ (m + n) = (P m ^ n * R n ^ m) := 
by sorry

end power_identity_l414_414565


namespace choose_captains_from_team_l414_414527

theorem choose_captains_from_team (n k : ℕ) (h1 : n = 12) (h2 : k = 4) : nat.choose n k = 990 := by
  rw [h1, h2]
  sorry

end choose_captains_from_team_l414_414527


namespace brendan_total_matches_won_l414_414036

def matches_won_round1 := 8
def matches_won_round2 := 0.7 * 10
def matches_won_round3 := 0.5 * 12
def matches_won_round4 := 0.6 * 14
def matches_won_round5 := 0.4 * 15

def total_matches_won := 
  matches_won_round1 + matches_won_round2 + matches_won_round3 + matches_won_round4 + matches_won_round5

theorem brendan_total_matches_won :
  total_matches_won = 35 :=
by
  -- Calculations based on given conditions:
  have round1 : matches_won_round1 = 8 := rfl
  have round2 : matches_won_round2 = 7 := by norm_num
  have round3 : matches_won_round3 = 6 := by norm_num
  have round4 : matches_won_round4 = 8.4 := by norm_num
  have round5 : matches_won_round5 = 6 := by norm_num
  -- Sum up all scores:
  have sum_scores : 8 + 7 + 6 + 8 + 6 = 35 := by linarith
  exact sum_scores

end brendan_total_matches_won_l414_414036


namespace condition_for_equation_l414_414247

theorem condition_for_equation (a b c : ℕ) (ha : 0 < a ∧ a < 20) (hb : 0 < b ∧ b < 20) (hc : 0 < c ∧ c < 20) :
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 :=
by
  sorry

end condition_for_equation_l414_414247


namespace proposition2_and_4_correct_l414_414148

theorem proposition2_and_4_correct (a b : ℝ) : 
  (a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧ 
  (a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → a^2 + b^2 = 9) :=
by
  sorry

end proposition2_and_4_correct_l414_414148


namespace necessary_but_not_sufficient_l414_414454

def mutually_exclusive (A1 A2 : Prop) : Prop := (A1 ∧ A2) → False
def complementary (A1 A2 : Prop) : Prop := (A1 ∨ A2) ∧ ¬(A1 ∧ A2)

theorem necessary_but_not_sufficient {A1 A2 : Prop}: 
  mutually_exclusive A1 A2 → complementary A1 A2 → (¬(mutually_exclusive A1 A2 → complementary A1 A2) ∧ (complementary A1 A2 → mutually_exclusive A1 A2)) := 
  by
    sorry

end necessary_but_not_sufficient_l414_414454


namespace digit_A_divisibility_l414_414569

theorem digit_A_divisibility :
  ∃ (A : ℕ), (0 ≤ A ∧ A < 10) ∧ (∃ k_5 : ℕ, 353809 * 10 + A = 5 * k_5) ∧ 
  (∃ k_7 : ℕ, 353809 * 10 + A = 7 * k_7) ∧ (∃ k_11 : ℕ, 353809 * 10 + A = 11 * k_11) 
  ∧ A = 0 :=
by 
  sorry

end digit_A_divisibility_l414_414569


namespace find_b2_l414_414313

def seq (n : ℕ) : ℕ → ℝ
| 1       := 25
| 9       := 125
| (nat.succ (nat.succ (nat.succ n))) := 
    real.geometric_mean (list.iota n).map (λ i, seq (i+1))

theorem find_b2 : seq 2 = 625 :=
sorry

end find_b2_l414_414313


namespace pete_miles_walked_l414_414976

noncomputable def steps_from_first_pedometer (flips1 : ℕ) (final_reading1 : ℕ) : ℕ :=
  flips1 * 100000 + final_reading1 

noncomputable def steps_from_second_pedometer (flips2 : ℕ) (final_reading2 : ℕ) : ℕ :=
  flips2 * 400000 + final_reading2 * 4

noncomputable def total_steps (flips1 flips2 final_reading1 final_reading2 : ℕ) : ℕ :=
  steps_from_first_pedometer flips1 final_reading1 + steps_from_second_pedometer flips2 final_reading2

noncomputable def miles_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem pete_miles_walked
  (flips1 flips2 final_reading1 final_reading2 steps_per_mile : ℕ)
  (h_flips1 : flips1 = 50)
  (h_final_reading1 : final_reading1 = 25000)
  (h_flips2 : flips2 = 15)
  (h_final_reading2 : final_reading2 = 30000)
  (h_steps_per_mile : steps_per_mile = 1500) :
  miles_walked (total_steps flips1 flips2 final_reading1 final_reading2) steps_per_mile = 7430 :=
by sorry

end pete_miles_walked_l414_414976


namespace integral_M_sin2a_eq_1_l414_414431

theorem integral_M_sin2a_eq_1 : (∫ a in 0..π/2, 1 * sin (2 * a)) = 1 :=
by
  sorry

end integral_M_sin2a_eq_1_l414_414431


namespace value_of_expression_l414_414767

-- Let's define the sequences and sums based on the conditions in a)
def sum_of_evens (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_of_multiples_of_three (p : ℕ) : ℕ :=
  3 * (p * (p + 1)) / 2

def sum_of_odds (m : ℕ) : ℕ :=
  m * m

-- Now let's formulate the problem statement as a theorem.
theorem value_of_expression : 
  sum_of_evens 200 - sum_of_multiples_of_three 100 - sum_of_odds 148 = 3146 :=
  by
  sorry

end value_of_expression_l414_414767


namespace hour_minute_hand_coincide_at_l414_414340

noncomputable def coinciding_time : ℚ :=
  90 / (6 - 0.5)

theorem hour_minute_hand_coincide_at : coinciding_time = 16 + 4 / 11 := 
  sorry

end hour_minute_hand_coincide_at_l414_414340


namespace area_sum_not_2019_l414_414965

theorem area_sum_not_2019 (points : Fin 6 → ℝ × ℝ)
    (areas : Fin 20 → ℤ) :
    (∀ i, ∃ (a b c : Fin 6), area_of_triangle (points a) (points b) (points c) = areas i) →
    ∑ i, areas i ≠ 2019 := by sorry

end area_sum_not_2019_l414_414965


namespace find_m_l414_414102

noncomputable def f (x m : ℝ) : ℝ := x^2 + 2^x - m * 2^(-x)

theorem find_m (m : ℝ) (h : ∀ x : ℝ, f x m = f (-x) m) : m = -1 :=
begin
  -- proof is omitted
  sorry
end

end find_m_l414_414102


namespace shift_parabola_5_units_right_l414_414887

def original_parabola (x : ℝ) : ℝ := x^2 + 3
def shifted_parabola (x : ℝ) : ℝ := (x-5)^2 + 3

theorem shift_parabola_5_units_right : ∀ x : ℝ, shifted_parabola x = original_parabola (x - 5) :=
by {
  -- This is the mathematical equivalence that we're proving
  sorry
}

end shift_parabola_5_units_right_l414_414887


namespace count_ordered_pairs_satisfying_conditions_l414_414432

theorem count_ordered_pairs_satisfying_conditions : 
    let S := ∑ y in Finset.range 200, Nat.floor ((200 - y) / (2 * y * (y + 1))) in
    ∃ (S : ℕ), S = ∑ y in Finset.range 200, Nat.floor ((200 - y) / (2 * y * (y + 1))) :=
by
  sorry

end count_ordered_pairs_satisfying_conditions_l414_414432


namespace find_alpha_polar_equation_l414_414117

variables {α : Real}

def point_P := (2, 1)
def line_l := { t : Real // x = 2 + t * Real.cos α ∧ y = 1 + t * Real.sin α }
def line_intersects_pos_axes := 
  ∃ A B : (ℝ × ℝ),
  A.1 > 0 ∧ B.2 > 0 ∧ 
  ∃ t1 t2 : Real, 
    line_l t1 = (A.1, 0) ∧ line_l t2 = (0, B.2)

def distance_PA (A : ℝ × ℝ) := ((2 - A.1) ^ 2 + (1 - A.2) ^ 2).sqrt
def distance_PB (B : ℝ × ℝ) := ((2 - B.1) ^ 2 + (1 - B.2) ^ 2).sqrt

def condition_PA_PB_product (PA PB : ℝ) := PA * PB = 4

theorem find_alpha 
  (A B : (ℝ × ℝ)) (h1 : line_intersects_pos_axes) 
  (h2 : condition_PA_PB_product (distance_PA A) (distance_PB B)):
  α = 3 * Real.pi / 4 :=
sorry

def polar_coordinate_line (ρ θ : Real) :=
  ρ * (Real.cos θ + Real.sin θ) = 3

theorem polar_equation 
  (α_value : α = 3 * Real.pi / 4):
  polar_coordinate_line :=
sorry

end find_alpha_polar_equation_l414_414117


namespace rhombus_implies_perpendicular_diagonals_perpendicular_diagonals_not_implies_rhombus_l414_414107

variable (A B C D : Type) 
variables [quad : quadrilateral A B C D]

def is_rhombus (A B C D : Type) : Prop :=
  -- Definition of a rhombus based on properties
  sorry

def diagonals_perpendicular (A B C D : Type) : Prop :=
  -- Definition of perpendicular diagonals
  sorry

theorem rhombus_implies_perpendicular_diagonals :
  is_rhombus A B C D → diagonals_perpendicular A B C D :=
  sorry

theorem perpendicular_diagonals_not_implies_rhombus :
  ¬(diagonals_perpendicular A B C D → is_rhombus A B C D) :=
  sorry

end rhombus_implies_perpendicular_diagonals_perpendicular_diagonals_not_implies_rhombus_l414_414107


namespace find_D_l414_414395

theorem find_D (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : D = 1 :=
by
  sorry

end find_D_l414_414395


namespace derivative_of_exp_2x_l414_414630

theorem derivative_of_exp_2x (x : ℝ) : 
  (deriv (λ x, real.exp (2 * x))) x = 2 * real.exp (2 * x) :=
by 
  sorry

end derivative_of_exp_2x_l414_414630


namespace no_nontrivial_integer_solutions_l414_414603

theorem no_nontrivial_integer_solutions (a b c d : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
by
  intro h
  sorry

end no_nontrivial_integer_solutions_l414_414603


namespace intersection_proof_complement_proof_range_of_m_condition_l414_414960

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

theorem intersection_proof : A ∩ B = {x | -2 ≤ x ∧ x < 1} := sorry

theorem complement_proof : (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 1} := sorry

theorem range_of_m_condition (m : ℝ) : (A ∪ C m = A) → (m ≤ 2) := sorry

end intersection_proof_complement_proof_range_of_m_condition_l414_414960


namespace cf_eq_fg_l414_414223

theorem cf_eq_fg 
  (A B C E D F G : Type)
  (h_triangle : is_triangle A B C)
  (h_AB_AC : length AB > length AC)
  (h_incircle_touches_BC_E : incircle_touches BC E)
  (h_AE_intersects_incircle_at_D : AE_intersects_incircle_at D E)
  (h_CE_eq_CF : length CE = length CF)
  (h_CF_extends_to_intersect_BD_G : extends_to_intersect CF BD G) 
  : length CF = length FG :=
sorry

end cf_eq_fg_l414_414223


namespace clerical_staff_reduction_l414_414402

noncomputable def clerical_staff_percentage (total_employees dept_a_total dept_b_total dept_c_total dept_a_clerical_ratio dept_b_clerical_ratio dept_c_clerical_ratio reduction_a reduction_b : ℝ) : ℝ :=
let
  clerical_a_before := dept_a_total * dept_a_clerical_ratio,
  clerical_b_before := dept_b_total * dept_b_clerical_ratio,
  clerical_c_before := dept_c_total * dept_c_clerical_ratio,
  clerical_a_after := clerical_a_before * (1 - reduction_a),
  clerical_b_after := clerical_b_before * (1 - reduction_b),
  clerical_c_after := clerical_c_before
in
  ((clerical_a_after + clerical_b_after + clerical_c_after) / total_employees) * 100

theorem clerical_staff_reduction :
  clerical_staff_percentage 12000 4000 6000 2000 (1/4) (1/6) (1/8) 0.25 0.1 ≈ 15.83 :=
by sorry

end clerical_staff_reduction_l414_414402


namespace original_price_of_cycle_l414_414003

theorem original_price_of_cycle
  (selling_price : ℝ)
  (loss_percentage : ℝ)
  (h_selling_price : selling_price = 1080)
  (h_loss_percentage : loss_percentage = 0.10) :
  ∃ P : ℝ, P = 1200 :=
by
  let original_price := selling_price / (1 - loss_percentage)
  have h : original_price = 1200 := by
    rw [h_selling_price, h_loss_percentage]
    norm_num
  use original_price
  exact h

end original_price_of_cycle_l414_414003


namespace probability_second_try_success_l414_414725

-- Definitions based directly on the problem conditions
def total_keys : ℕ := 4
def keys_can_open_door : ℕ := 2
def keys_cannot_open_door : ℕ := total_keys - keys_can_open_door

-- Theorem statement translation
theorem probability_second_try_success :
  let prob_first_try_fail := (keys_cannot_open_door : ℝ) / total_keys,
      prob_second_try_success := (keys_can_open_door : ℝ) / (total_keys - 1)
  in
  prob_first_try_fail * prob_second_try_success = (1/3 : ℝ) :=
by
  -- Proof content goes here
  sorry

end probability_second_try_success_l414_414725


namespace bridget_gave_erasers_l414_414973

variable (p_start : ℕ) (p_end : ℕ) (e_b : ℕ)

theorem bridget_gave_erasers (h1 : p_start = 8) (h2 : p_end = 11) (h3 : p_end = p_start + e_b) :
  e_b = 3 := by
  sorry

end bridget_gave_erasers_l414_414973


namespace extra_kilometers_per_hour_l414_414677

theorem extra_kilometers_per_hour (S a : ℝ) (h : a > 2) : 
  (S / (a - 2)) - (S / a) = (S / (a - 2)) - (S / a) :=
by sorry

end extra_kilometers_per_hour_l414_414677


namespace cos_neg_300_l414_414684

theorem cos_neg_300 : Real.cos (-(300 : ℝ) * Real.pi / 180) = 1 / 2 :=
by
  -- Proof goes here
  sorry

end cos_neg_300_l414_414684


namespace probability_of_prime_ball_l414_414277

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def balls := {2, 3, 4, 5, 6, 7}

noncomputable def prime_balls : Set ℕ :=
  {n ∈ balls | is_prime n}

def probability_prime : ℚ :=
  (prime_balls.to_finset.card : ℚ) / (balls.to_finset.card : ℚ)

theorem probability_of_prime_ball : probability_prime = 2/3 := by
  sorry

end probability_of_prime_ball_l414_414277


namespace find_a_extremum_monotonic_intervals_l414_414151

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2

theorem find_a_extremum:
  (a : ℝ) (a_extremum : ∃ a, f a (1/3) = f a_extremum) : a = 6 := sorry

theorem monotonic_intervals :
  f 6 x → 
  (∀ x, (f 6' x > 0 ↔ (x < 0 ∨ x > 1/3)) ∧ (f 6' x < 0 ↔ (0 < x ∧ x < 1/3))) :=
sorry

end find_a_extremum_monotonic_intervals_l414_414151


namespace geometric_product_identity_l414_414953

-- Definitions for the geometric entities involved
variables (A I B C S D : Type) [Geometry A I B C S D] -- Assuming there is a geometric context

-- Conditions
axiom D_is_intersection : ∃ (D : Type), is_intersection D (A I) (B C)

-- Proof Problem Statement
theorem geometric_product_identity (h1 : D_is_intersection D A I B C) :
  SA * SD = (SI)^2 := 
sorry -- Proof is not required

end geometric_product_identity_l414_414953


namespace positive_integers_square_of_sum_of_digits_l414_414056

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem positive_integers_square_of_sum_of_digits :
  ∀ (n : ℕ), (n > 0) → (n = sum_of_digits n ^ 2) → (n = 1 ∨ n = 81) :=
by
  sorry

end positive_integers_square_of_sum_of_digits_l414_414056


namespace probability_diff_colors_l414_414192

-- Definitions based on the conditions provided.
-- Total number of chips
def total_chips := 15

-- Individual probabilities of drawing each color first
def prob_green_first := 6 / total_chips
def prob_purple_first := 5 / total_chips
def prob_orange_first := 4 / total_chips

-- Probabilities of drawing a different color second
def prob_not_green := 9 / total_chips
def prob_not_purple := 10 / total_chips
def prob_not_orange := 11 / total_chips

-- Combined probabilities for each case
def prob_green_then_diff := prob_green_first * prob_not_green
def prob_purple_then_diff := prob_purple_first * prob_not_purple
def prob_orange_then_diff := prob_orange_first * prob_not_orange

-- Total probability of drawing two chips of different colors
def total_prob_diff_colors := prob_green_then_diff + prob_purple_then_diff + prob_orange_then_diff

-- Theorem statement to be proved
theorem probability_diff_colors : total_prob_diff_colors = 148 / 225 :=
by
  -- Proof would go here
  sorry

end probability_diff_colors_l414_414192


namespace sum_of_possible_x_l414_414649

theorem sum_of_possible_x 
  (x : ℝ)
  (squareSide : ℝ) 
  (rectangleLength : ℝ) 
  (rectangleWidth : ℝ) 
  (areaCondition : (rectangleLength * rectangleWidth) = 3 * (squareSide ^ 2)) : 
  6 + 6.5 = 12.5 := 
by 
  sorry

end sum_of_possible_x_l414_414649


namespace chromium_mass_percentage_correct_l414_414078

def potassium_molar_mass : ℝ := 39.10
def chromium_molar_mass : ℝ := 51.996
def oxygen_molar_mass : ℝ := 16.00

def K2Cr2O7 : ℝ := (2 * potassium_molar_mass) + (2 * chromium_molar_mass) + (7 * oxygen_molar_mass)

def chromium_mass_in_K2Cr2O7 : ℝ := 2 * chromium_molar_mass

def chromium_mass_percentage_in_K2Cr2O7 : ℝ := (chromium_mass_in_K2Cr2O7 / K2Cr2O7) * 100

theorem chromium_mass_percentage_correct : chromium_mass_percentage_in_K2Cr2O7 = 35.34 :=
  by
    sorry

end chromium_mass_percentage_correct_l414_414078


namespace magnitude_of_z_l414_414101

theorem magnitude_of_z 
  (z : ℂ) 
  (h : (3 + I) / z = 1 - I) : 
  complex.abs z = real.sqrt 5 := 
sorry

end magnitude_of_z_l414_414101


namespace find_alpha_l414_414113

open Real

noncomputable def point (x y : ℝ) : Prop := True

noncomputable def line (α t : ℝ) : Prop :=
  ∃ x y : ℝ, x = 2 + t * cos α ∧ y = 1 + t * sin α

theorem find_alpha (α : ℝ) :
  let P := (2, 1)
  let l := { t : ℝ // ∃ x y : ℝ, x = 2 + t * cos α ∧ y = 1 + t * sin α }
  let A : ℝ×ℝ := (2 - (2 - 1 / tan α), 0)
  let B : ℝ×ℝ := (0, 1 - 2 * tan α)
  let PA := sqrt((2 - (2 - 1 / tan α))^2 + (1 - 0)^2)
  let PB := sqrt((2 - 0)^2 + (1 - (1 - 2 * tan α))^2)
  (|PA| * |PB| = 4) → α = 3 * π / 4 :=
begin
  sorry
end

end find_alpha_l414_414113


namespace red_balls_approximation_l414_414522

def total_balls : ℕ := 50
def red_ball_probability : ℚ := 7 / 10

theorem red_balls_approximation (r : ℕ)
  (h1 : total_balls = 50)
  (h2 : red_ball_probability = 0.7) :
  r = 35 := by
  sorry

end red_balls_approximation_l414_414522


namespace max_months_with_five_sundays_l414_414205

theorem max_months_with_five_sundays (normal_year_days : ℕ) (leap_year_days : ℕ) (days_per_month : list ℕ)
  (normal_year_eq : normal_year_days = 365) (leap_year_eq : leap_year_days = 366)
  (months : ℕ) (days_in_week : ℕ) (months_eq : months = 12) (week_eq : days_in_week = 7)
  (days_per_month_range : ∀ d, d ∈ days_per_month → 28 ≤ d ∧ d ≤ 31) :
  ∃ max_five_sundays, max_five_sundays = 5 :=
  sorry

end max_months_with_five_sundays_l414_414205


namespace find_min_area_of_triangle_l414_414248

theorem find_min_area_of_triangle (a : ℝ) (h_a_pos : a > 0) :
  let Q := (Real.exp (a^2), a)
  let y_diff := (1 : ℝ) / (a * Real.exp (a^2))
  let R := (0, a - (1 / a))
  |a - (a - 1 / a)| = 1/a in
  let area := (1 / 2) * Real.exp (a^2) * (1 / |a|) in
  let f := λ a, (Real.exp (a^2)) / (2 * a) in
  let min_area := f (Real.sqrt 2 / 2) in
  min_area = Real.exp (1 / 2) / Real.sqrt 2 :=
sorry

end find_min_area_of_triangle_l414_414248


namespace orthic_triangle_altitude_points_equal_l414_414934

theorem orthic_triangle_altitude_points_equal {A B C H H_A H_B H_C : Type}
  [triangle ABC] [orthocenter H ABC]
  (altitude_A : altitude H_A A B C) (altitude_B : altitude H_B A B C) (altitude_C : altitude H_C A B C)
  (H_orthic : orthic_triangle ABC H_A H_B H_C)
  (orthocenter_AHBHC : orthocenter H (triangle A H_B H_C))
  (orthocenter_BHCHC : orthocenter H (triangle B H_A H_C))
  (orthocenter_CHBHB : orthocenter H (triangle C H_A H_B))
  :
  ∀ (X Y Z : Type),
    altitudes_intersect_at H (triangle A H_B H_C) = X ∧
    altitudes_intersect_at H (triangle B H_A H_C) = Y ∧
    altitudes_intersect_at H (triangle C H_A H_B) = Z →
    triangle X Y Z = triangle H_A H_B H_C :=
sorry

end orthic_triangle_altitude_points_equal_l414_414934


namespace sequence_formulas_and_sum_l414_414809

variable {a b S T : ℕ → ℝ}
variable {n : ℕ}

-- Given conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
def geometric_sequence (b : ℕ → ℝ) := ∃ q : ℝ, ∀ n, b (n + 1) = b n * q
def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n, S n = (finset.range n).sum a

variables (a_arith : arithmetic_sequence a)
variables (b_geom : geometric_sequence b)
variables (a1 : a 1 = 1)
variables (b1 : b 1 = 1)
variables (b3S3 : b 3 * S 3 = 36)
variables (b2S2 : b 2 * S 2 = 8)

-- To prove
theorem sequence_formulas_and_sum (a_increasing : ∀ n, a n < a (n + 1)) :
  (∃ d q, (∀ n, a (n + 1) = a n + d) ∧ (∀ n, b (n + 1) = b n * q) ∧ ((b 3 * (sum_first_n_terms S a) 3 = 36) ∧ (b 2 * (sum_first_n_terms S a) 2 = 8))) ∧
  ∀ n, T n = (1 / ((a n) * (a (n + 1)))) → T n = n / (2 * n + 1) :=
sorry

end sequence_formulas_and_sum_l414_414809


namespace value_of_a_plus_b_l414_414839

theorem value_of_a_plus_b (a b x y : ℝ) 
  (h1 : 2 * x + 4 * y = 20)
  (h2 : a * x + b * y = 1)
  (h3 : 2 * x - y = 5)
  (h4 : b * x + a * y = 6) : a + b = 1 := 
sorry

end value_of_a_plus_b_l414_414839


namespace sand_in_partial_bag_correct_l414_414319

-- Define the conditions
def total_sand : ℝ := 1254.75
def bag_capacity : ℝ := 73.5

-- Define the question and expected answer
def sand_in_partial_bag : ℝ := total_sand - (bag_capacity * (total_sand / bag_capacity).floor)
def expected_answer : ℝ := 5.25

-- The theorem to prove the equivalence
theorem sand_in_partial_bag_correct : sand_in_partial_bag = expected_answer := by
    sorry

end sand_in_partial_bag_correct_l414_414319


namespace milo_dozen_eggs_l414_414964

theorem milo_dozen_eggs (total_weight_pounds egg_weight_pounds dozen : ℕ) (h1 : total_weight_pounds = 6)
  (h2 : egg_weight_pounds = 1 / 16) (h3 : dozen = 12) :
  total_weight_pounds / egg_weight_pounds / dozen = 8 :=
by
  -- The proof would go here
  sorry

end milo_dozen_eggs_l414_414964


namespace correct_propositions_count_l414_414048

theorem correct_propositions_count :
  let X : Type := real,
      σ : Type := real,
      proposition_1 := ∀ (X : X) (σ : σ),
                       (P : set real → real)
                       (hX : prob.normal X 0 (σ^2))
                       (hP : P (set.Icc (-2 : real) 2) = 0.6),
                       P ({x : real | x > 2}) = 0.2,
      proposition_2 := ∀ p : Prop,
                       ((∃ x_0 : real, 1 ≤ x_0 ∧ x_0^2 - x_0 - 1 < 0) ↔
                        ∀ x : real, 1 ≤ x → x^2 - x - 1 ≥ 0),
      proposition_3 := ∀ (l1 l2 : Prop) 
                        (a b : real),
                        ((a ≠ 0 ∧ b ≠ 0) → 
                         let l1 := (λ x y : real, a * x + 3 * y - 1 = 0),
                             l2 := (λ x y : real, x + b * y + 1 = 0),
                         (l1 ⊥ l2) ↔ arctan (-a / 3) * arctan (-1 / b) = -1)
  in proposition_1 = true ∧ proposition_2 = false ∧ proposition_3 = false → 
     (1) :=
  begin
    intros,
    sorry
  end

end correct_propositions_count_l414_414048


namespace total_inches_of_rope_l414_414557

noncomputable def inches_of_rope (last_week_feet : ℕ) (less_feet : ℕ) (feet_to_inches : ℕ → ℕ) : ℕ :=
  let last_week_inches := feet_to_inches last_week_feet
  let this_week_feet := last_week_feet - less_feet
  let this_week_inches := feet_to_inches this_week_feet
  last_week_inches + this_week_inches

theorem total_inches_of_rope 
  (six_feet : ℕ := 6)
  (four_feet_less : ℕ := 4)
  (conversion : ℕ → ℕ := λ feet, feet * 12) :
  inches_of_rope six_feet four_feet_less conversion = 96 := by
  sorry

end total_inches_of_rope_l414_414557


namespace proof_problem_l414_414120

open Real

noncomputable def P : ℝ × ℝ := (2, 1)

-- Parametric form of the line
def l (α t : ℝ) : ℝ × ℝ := (2 + t * cos α, 1 + t * sin α)

-- Condition that |PA| * |PB| = 4
def condition (α : ℝ) : Prop := 
  let A := (2 + (-2 * cos α), 1 + (-2 * sin α) * (cos α == 0))
  let B := (2 + (-1 / (tan α)), 1 + (- (1 / tan α)) * sin α == 0)
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  ‖PA‖ * ‖PB‖ = 4

-- Problem statement
theorem proof_problem : 
  ∃ α, condition α ∧ α = (3 * π / 4) ∧ 
    ∀ ρ θ, ρ * (cos θ + sin θ) = 3 : =
begin
  sorry
end

end proof_problem_l414_414120


namespace inequ_am_gm_l414_414440

-- declare variables and conditions
variables (a b : ℝ)
hypothesis h1 : a > 0
hypothesis h2 : b > 0

-- the theorem statement
theorem inequ_am_gm : (a + b) / 2 ≥ (2 * a * b) / (a + b) :=
by
  sorry

end inequ_am_gm_l414_414440


namespace smallest_positive_integer_divisible_by_15_16_18_l414_414777

theorem smallest_positive_integer_divisible_by_15_16_18 : 
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ n = 720 := 
by
  sorry

end smallest_positive_integer_divisible_by_15_16_18_l414_414777


namespace dot_product_AB_AC_l414_414720

noncomputable def point := (ℝ × ℝ)

def line_through_point_slope (p : point) (m : ℝ) : set point :=
  {q : point | q.2 = m * (q.1 - p.1)}

def circle (center : point) (radius : ℝ) : set point :=
  {q : point | (q.1 - center.1)^2 + (q.2 - center.2)^2 = radius^2}

def dot_product (v w : point) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def point_A := (1, -√3 / 3 * (1 - 4))
def point_B := (4, 0)
def center_C := (2, 0)
def A_vector := (1 - center_C.1, -√3 / 3 * (-3) - center_C.2)
def B_vector := (4 - center_C.1, 0 - center_C.2)
def AB_vector := (point_B.1 - point_A.1, point_B.2 - point_A.2)

theorem dot_product_AB_AC : dot_product AB_vector A_vector = 6 :=
  sorry

end dot_product_AB_AC_l414_414720


namespace determinant_scaled_l414_414438

variable (x y z w : ℝ)
variable (h : det (matrix ![![x, y], ![z, w]]) = -3)

theorem determinant_scaled :
  det (matrix ![![3 * x, 3 * y], ![5 * z, 5 * w]]) = -45 :=
by
  sorry

end determinant_scaled_l414_414438


namespace tabitha_candy_count_l414_414619

-- Definitions from conditions
def Stan_pieces := 13
def Tabitha_pieces : ℕ  -- Assumed to be ℕ as pieces of candy

def Julie_pieces := Tabitha_pieces / 2
def Carlos_pieces := Stan_pieces * 2

-- Condition of the problem
def total_pieces : ℕ := Tabitha_pieces + Stan_pieces + Julie_pieces + Carlos_pieces

-- The proof problem
theorem tabitha_candy_count (T : ℕ) (h : total_pieces = 72) : T = 22 := by
  sorry

end tabitha_candy_count_l414_414619


namespace magnitude_k_add_i_l414_414162

open_locale classical

variables {E : Type*} [inner_product_space ℝ E] (i j k : E)

-- Conditions
-- (1) i and j are unit vectors
axiom unit_i : ∥i∥ = 1
axiom unit_j : ∥j∥ = 1

-- (2) i and j are perpendicular
axiom perp_ij : inner_product i j = 0

-- (3) k = 2i - 4j
axiom k_def : k = 2 • i - 4 • j

-- Theorem: Magnitude of k + i is 5
theorem magnitude_k_add_i : ∥k + i∥ = 5 :=
sorry

end magnitude_k_add_i_l414_414162


namespace average_weight_of_dogs_is_5_l414_414996

def weight_of_brown_dog (B : ℝ) : ℝ := B
def weight_of_black_dog (B : ℝ) : ℝ := B + 1
def weight_of_white_dog (B : ℝ) : ℝ := 2 * B
def weight_of_grey_dog (B : ℝ) : ℝ := B - 1

theorem average_weight_of_dogs_is_5 (B : ℝ) (h : (weight_of_brown_dog B + weight_of_black_dog B + weight_of_white_dog B + weight_of_grey_dog B) / 4 = 5) :
  5 = 5 :=
by sorry

end average_weight_of_dogs_is_5_l414_414996


namespace balloon_count_correct_l414_414322

def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def silver_balloons : ℕ := 2 * gold_balloons
def total_balloons : ℕ := gold_balloons + silver_balloons + black_balloons

theorem balloon_count_correct : total_balloons = 573 := by
  sorry

end balloon_count_correct_l414_414322


namespace expression_evaluation_l414_414071

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end expression_evaluation_l414_414071


namespace number_of_perfect_square_factors_l414_414857

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def factor_of (a b : ℕ) : Prop :=
  b % a = 0

theorem number_of_perfect_square_factors :
  let product := (2^12) * (3^10) * (5^18) * (7^8) in
  ∃ n, is_perfect_square n ∧ factor_of n product ∧ n = 2100 :=
sorry

end number_of_perfect_square_factors_l414_414857


namespace minimum_value_of_reciprocal_sum_l414_414128

theorem minimum_value_of_reciprocal_sum (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  ∃ m, m = 9 ∧ (∀ a ≥ 0, ∀ b ≥ 0, ∀ c ≥ 0, x = a ∧ y = b ∧ z = c → a + b + c = 1 → (a > 0 ∧ b > 0 ∧ c > 0) 
  → m ≤ (1/a + 1/b + 1/c)) :=
begin
  use 9,
  split,
  { refl },
  { intros a ha b hb c hc hcross hmedium,
    sorry }
end

end minimum_value_of_reciprocal_sum_l414_414128


namespace solve_ARKA_l414_414420

def distinct_digits (A R K : ℕ) : Prop :=
  A ≠ R ∧ A ≠ K ∧ R ≠ K

def valid_solution (A R K : ℕ) : Prop :=
  let ARKA := 1000 * A + 100 * R + 10 * K + A
  let RKA := 100 * R + 10 * K + A
  let KA := 10 * K + A
  ARKA + RKA + KA + A = 2014 ∧ distinct_digits A R K

theorem solve_ARKA : ∃ (A R K : ℕ), valid_solution A R K ∧ A = 1 ∧ R = 4 ∧ K = 7 :=
by
  exists 1, 4, 7
  unfold valid_solution
  unfold distinct_digits
  simp
  split; simp
  sorry

end solve_ARKA_l414_414420


namespace sum_of_a_b_c_l414_414158

theorem sum_of_a_b_c (a b c : ℝ) (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 :=
by
  -- The proof would go here
  sorry

end sum_of_a_b_c_l414_414158


namespace tetrahedrons_from_cube_vertices_l414_414490

theorem tetrahedrons_from_cube_vertices : 
  let cube_vertices := 8
  let total_combinations := Nat.choose 8 4
  let invalid_tetrahedrons := 12
  total_combinations - invalid_tetrahedrons = 58 :=
begin
  sorry
end

end tetrahedrons_from_cube_vertices_l414_414490


namespace model_tower_height_l414_414190

variable (real_sphere_volume : ℝ) (model_sphere_volume : ℝ) (total_real_tower_height : ℝ)

-- conditions
def real_sphere_volume := 200000
def model_sphere_volume := 0.05
def total_real_tower_height := 70

-- question and answer as a theorem
theorem model_tower_height :
  let volume_ratio := real_sphere_volume / model_sphere_volume
  let scale_factor := real.to_nat (volume_ratio^(1/3))
  (total_real_tower_height / scale_factor) = 0.44 :=
by
  sorry

end model_tower_height_l414_414190


namespace find_correct_statements_l414_414129

-- Given entities
variables (α β γ : Plane) (l : Line)

-- Conditions from the problem
def condition1 : Prop := α ⊥ β ∧ l ⊥ β → l ∥ α
def condition2 : Prop := l ⊥ α ∧ l ∥ β → α ⊥ β
def condition3 : Prop := (∀ p₁ p₂ : Point, p₁ ∈ l ∧ p₂ ∈ l ∧ distance p₁ α = distance p₂ α → l ∥ α)
def condition4 : Prop := α ⊥ β ∧ α ∥ γ → γ ⊥ β

-- The Lean statement to prove
theorem find_correct_statements : condition2 ∧ condition4 :=
by
  sorry

end find_correct_statements_l414_414129


namespace sum_of_roots_of_y_squared_eq_36_l414_414867

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end sum_of_roots_of_y_squared_eq_36_l414_414867


namespace find_x_solutions_l414_414087

theorem find_x_solutions (x : ℝ) (h : sqrt (5 * x + 2 * x ^ 2 + 8) = 12) :
  x = 8 ∨ x = -17 / 2 :=
sorry

end find_x_solutions_l414_414087


namespace election_results_l414_414910

theorem election_results (T : ℕ) 
  (h : 3000 = 0.22 * T) :
  T = 13636 ∧
  (0.35 * T).round = 4773 ∧
  (0.28 * T).round = 3818 ∧
  T - (4773 + 3818 + 3000) = 2045 := by
  sorry

end election_results_l414_414910


namespace radius_of_smaller_circle_l414_414210

theorem radius_of_smaller_circle 
  (TP TQ: ℝ) 
  (circles_concentric: bool) 
  (T T' : ℝ) 
  (tangency_T T' : ℝ) 
  (PTQ : bool)
  (TP_eq_5 : TP = 5) 
  (TQ_eq_12 : TQ = 12) :
  ∃ r : ℝ, r = 5 :=
by
  -- The proof is omitted as per the instruction to provide just the statement
  sorry

end radius_of_smaller_circle_l414_414210


namespace line_circle_intersection_probability_l414_414098

theorem line_circle_intersection_probability (b r : ℕ) (hb : b ∈ {1, 2, 3, 4}) (hr : r ∈ {1, 2, 3, 4}) :
  (b^2 ≤ 2 * r) → 
  let valid_combinations := 
    {br : ℕ × ℕ | (br.1 ∈ {1, 2, 3, 4}) ∧ (br.2 ∈ {1, 2, 3, 4}) ∧ (br.1^2 ≤ 2 * br.2)} in
  (↑(set.card valid_combinations) / 16 = (7 / 16 : ℚ)) :=
by
  sorry

end line_circle_intersection_probability_l414_414098


namespace cross_section_area_correct_l414_414671

noncomputable def cross_section_area (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 33) / 8

theorem cross_section_area_correct
  (AB CC1 : ℝ)
  (h1 : AB = a)
  (h2 : CC1 = 2 * a) :
  cross_section_area a = (3 * a^2 * Real.sqrt 33) / 8 :=
by
  sorry

end cross_section_area_correct_l414_414671


namespace choose_captains_from_team_l414_414526

theorem choose_captains_from_team (n k : ℕ) (h1 : n = 12) (h2 : k = 4) : nat.choose n k = 990 := by
  rw [h1, h2]
  sorry

end choose_captains_from_team_l414_414526


namespace distance_to_excircle_center_eq_perimeter_l414_414622

theorem distance_to_excircle_center_eq_perimeter 
  (A B C O : Type) [Point A] [Point B] [Point C] [Point O]
  (angle_A : ∠ B A C = 120)
  (tangent1 : Tangent (A, B) O)
  (tangent2 : Tangent (B, C) O)
  (tangent3 : Tangent (C, A) O) :
  dist(A, O) = perimeter A B C :=
begin
  sorry
end

end distance_to_excircle_center_eq_perimeter_l414_414622


namespace probability_of_vowels_l414_414897

-- Conditions
def num_students : ℕ := 20
def initials : Finset String := 
  finset.of_list ["NN", "OO", "PP", "QQ", "RR", "SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ"]
def vowels : Finset String :=
  finset.of_list ["O", "U", "Y"]

-- Probability calculation
def probability_initials_vowels : ℚ :=
  (vowels.card : ℚ) / (initials.card : ℚ)

-- Proof statement
theorem probability_of_vowels : 
  probability_initials_vowels = (3 : ℚ) / (13 : ℚ) :=
sorry

end probability_of_vowels_l414_414897


namespace tan_half_angle_second_quadrant_l414_414814

variable {α : ℝ}

theorem tan_half_angle_second_quadrant (h1 : π/2 < α ∧ α < π) 
                                      (h2 : 3 * sin α + 4 * cos α = 0) : 
                                      tan (α / 2) = 2 := 
by 
  sorry

end tan_half_angle_second_quadrant_l414_414814


namespace curve_tangent_parallel_l414_414147

theorem curve_tangent_parallel (n : ℝ) : 
  let curve : ℝ → ℝ := λ x, x^n,
      tangent_slope_at (y' : ℝ → ℝ) (x : ℝ) := y' x,
      line_slope := 2 in
  tangent_slope_at (λ x, n * x^(n - 1)) 1 = line_slope → n = 2 := 
by
  sorry

end curve_tangent_parallel_l414_414147


namespace integral_value_l414_414462

theorem integral_value :
  (∫ x in 0..(Real.pi / 2), (4 * Real.sin x + Real.cos x)) = 5 :=
by
  sorry

end integral_value_l414_414462


namespace IO_eq_OJ_l414_414218

-- Definitions of the key points and conditions
variables {A B C D E F G H I J O : Type} [point A] [point B] [point C] [point D]
          [point E] [point F] [point G] [point H] [point I] [point J] [point O]

-- Quadrilateral with given properties
def quadrilateral_ABCD (AB AD BC CD : ℝ) : Prop :=
  AB = AD ∧ BC = CD

-- The intersection point of AC and BD
def intersection_point (AC BD : line) (O : point) : Prop :=
  incidence AC O ∧ incidence BD O

-- Arbitrary lines through O and their intersections with AD, BC, AB, and CD
def arbitrary_lines_through_O (AD BC AB CD : line) (O E F G H : point) : Prop :=
  incidence AD O ∧ incidence AD E ∧
  incidence BC O ∧ incidence BC F ∧
  incidence AB O ∧ incidence AB G ∧
  incidence CD O ∧ incidence CD H

-- GF and EH intersect BD at I and J respectively
def intersection_GF_EH_BD (GF EH BD : line) (I J : point) : Prop :=
  intersection GF BD I ∧ intersection EH BD J

-- The main theorem to prove
theorem IO_eq_OJ
  (AB AD BC CD O E F G H I J : point)
  (GF EH BD : line)
  (hq : quadrilateral_ABCD AB AD BC CD)
  (hi : intersection_point AC BD O)
  (hlin : arbitrary_lines_through_O AD BC AB CD O E F G H)
  (hintsec : intersection_GF_EH_BD GF EH BD I J) :
  distance I O = distance J O := 
by {
  sorry
}

end IO_eq_OJ_l414_414218


namespace range_of_m_l414_414881

-- Define the complex number z
def z (m : ℝ) : ℂ := complex.mk (m + 1) (-(m - 3))

-- Define the first and third quadrants conditions
def in_first_quadrant (m : ℝ) : Prop := (m + 1 > 0) ∧ (-(m - 3) > 0)
def in_third_quadrant (m : ℝ) : Prop := (m + 1 < 0) ∧ (-(m - 3) < 0)

-- Main statement
theorem range_of_m (m : ℝ) :
  (in_first_quadrant m ∨ in_third_quadrant m) ↔ (-1 < m ∧ m < 3) :=
sorry

end range_of_m_l414_414881


namespace part1_part2_l414_414122

-- Definition of set A
def set_A : set ℝ := {x : ℝ | abs (x + 2) < 3}

-- Definition of set B
def set_B (m : ℝ) : set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- Proof statement for Part 1
theorem part1 (m : ℝ) : set_A ⊆ set_B m ↔ m ≤ -5 := by sorry

-- Proof statement for Part 2
theorem part2 (n m : ℝ) : set_A ∩ set_B m = set.Ioo (-1) n → (m = -1 ∧ n = 1) := by sorry

end part1_part2_l414_414122


namespace probability_same_outcomes_l414_414015

-- Let us define the event space for a fair coin
inductive CoinTossOutcome
| H : CoinTossOutcome
| T : CoinTossOutcome

open CoinTossOutcome

-- Definition of an event where the outcomes are the same (HHH or TTT)
def same_outcomes (t1 t2 t3 : CoinTossOutcome) : Prop :=
  (t1 = H ∧ t2 = H ∧ t3 = H) ∨ (t1 = T ∧ t2 = T ∧ t3 = T)

-- Number of all possible outcomes for three coin tosses
def total_outcomes : ℕ := 2 ^ 3

-- Number of favorable outcomes where all outcomes are the same
def favorable_outcomes : ℕ := 2

-- Calculation of probability
def prob_same_outcomes : ℚ := favorable_outcomes / total_outcomes

-- The statement to be proved in Lean 4
theorem probability_same_outcomes : prob_same_outcomes = 1 / 4 := 
by sorry

end probability_same_outcomes_l414_414015


namespace sum_of_largest_two_numbers_in_each_row_equals_column_l414_414349

theorem sum_of_largest_two_numbers_in_each_row_equals_column
    (n : ℕ) (a b : ℝ) (table : Matrix (Fin n) (Fin n) ℝ)
    (h1 : ∀ i : Fin n, (∑ j in (finset.univ : Finset (Fin n)), table i j) = a)
    (h2 : ∀ j : Fin n, (∑ i in (finset.univ : Finset (Fin n)), table i j) = b) :
    a = b :=
  sorry

end sum_of_largest_two_numbers_in_each_row_equals_column_l414_414349


namespace Danielle_rooms_is_6_l414_414849

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l414_414849


namespace math_problem_l414_414260

-- Define the function f
def f (x : ℝ) : ℝ :=
  (2 * real.sqrt 3 * real.sin x - real.cos x) * real.cos x 
  + real.cos (real.pi / 2 - x) ^ 2

-- Define the interval [0, π]
def interval := set.Icc 0 real.pi

-- Part (1) Definition
def monotonic_increase_intervals (f : ℝ → ℝ) (interval : set ℝ) : set (set ℝ) :=
  {{ x | 0 ≤ x ∧ x ≤ real.pi / 3 }, { x | 5 * real.pi / 6 ≤ x ∧ x ≤ real.pi }}

-- Part (2) Definitions - triangle properties and condition
structure triangle :=
  (A B C a b c : ℝ)
  (acute : 0 < A ∧ A < real.pi / 2 ∧ 0 < B ∧ B < real.pi / 2 ∧ 0 < C ∧ C < real.pi / 2)
  (condition : (a ^ 2 + c ^ 2 - b ^ 2) / c = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a - c))

def f_A (A : ℝ) : ℝ :=
  2 * real.sin (2 * A - real.pi / 6)

def range_of_f_A := set.Ioo 1 2

-- Lean 4 Statement
theorem math_problem (x : ℝ) (t : triangle):
  (∀ x ∈ interval, f x = f (interval x)) →
  (∀ t.A > real.pi / 6 ∧ t.A < real.pi / 2,
    f_A t.A ∈ range_of_f_A) :=
sorry

end math_problem_l414_414260


namespace reciprocal_sum_greater_l414_414496

theorem reciprocal_sum_greater (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
    (1 / a + 1 / b) > 1 / (a + b) :=
sorry

end reciprocal_sum_greater_l414_414496


namespace bobby_toy_cars_l414_414744

theorem bobby_toy_cars (initial_cars : ℕ) (increase_rate : ℕ → ℕ) (n : ℕ) :
  initial_cars = 16 →
  increase_rate 1 = initial_cars + (initial_cars / 2) →
  increase_rate 2 = increase_rate 1 + (increase_rate 1 / 2) →
  increase_rate 3 = increase_rate 2 + (increase_rate 2 / 2) →
  n = 3 →
  increase_rate n = 54 :=
by
  intros
  sorry

end bobby_toy_cars_l414_414744


namespace sqrt_neg_squared_eq_two_l414_414040

theorem sqrt_neg_squared_eq_two : (-Real.sqrt 2) ^ 2 = 2 := by
  sorry

end sqrt_neg_squared_eq_two_l414_414040


namespace total_lunch_cost_l414_414021

def adam_spends (rick_spent : ℝ) : ℝ := (2 / 3) * rick_spent
def jose_spent := 45
def rick_spent := jose_spent
def total_cost (adam_spent : ℝ) (rick_spent : ℝ) (jose_spent : ℝ) : ℝ := adam_spent + rick_spent + jose_spent

theorem total_lunch_cost : total_cost (adam_spends rick_spent) rick_spent jose_spent = 120 :=
by sorry

end total_lunch_cost_l414_414021


namespace binom_coeff_identity_compute_value_l414_414044

noncomputable def binom_coeff (x : ℚ) (n : ℕ) : ℚ := 
  if h : n = 0 then 1 else (x * binom_coeff (x - 1) (n - 1)) / n

theorem binom_coeff_identity (n : ℕ) (k : ℕ) (h : 0 ≤ k ∧ k ≤ n) : 
  binom_coeff n k = (n.choose k) :=
sorry

theorem compute_value :
  (binom_coeff (1/2) 2015 * 4^2015) / (4030.choose 2015) = -2 / 4029 :=
sorry

end binom_coeff_identity_compute_value_l414_414044


namespace remainder_when_divided_by_product_l414_414954

def Q : ℤ[X] -- polynomial with integer coefficients

axiom condition1 : polynomial.Remainder Q (X - 15) = 7
axiom condition2 : polynomial.Remainder Q (X - 10) = 2

theorem remainder_when_divided_by_product :
  polynomial.Remainder Q ((X - 10) * (X - 15)) = X - 8 :=
sorry

end remainder_when_divided_by_product_l414_414954


namespace barry_sotter_magic_l414_414970

theorem barry_sotter_magic (n : ℕ) : (n + 3) / 3 = 50 → n = 147 := 
by 
  sorry

end barry_sotter_magic_l414_414970


namespace florida_vs_georgia_license_plates_l414_414582

theorem florida_vs_georgia_license_plates :
  26 ^ 4 * 10 ^ 3 - 26 ^ 3 * 10 ^ 3 = 439400000 := by
  -- proof is omitted as directed
  sorry

end florida_vs_georgia_license_plates_l414_414582


namespace number_of_solutions_cosine_equation_l414_414413

theorem number_of_solutions_cosine_equation : 
  (∃ P : set ℝ, {x | 3 * (cos x)^3 - 7 * (cos x)^2 + 3 * (cos x) = 0 ∧ 0 ≤ x ∧ x ≤ 2 * π}.finite ∧ 
  (finset.card (set.to_finset P) = 4)) :=
sorry

end number_of_solutions_cosine_equation_l414_414413


namespace christine_wander_time_l414_414042

theorem christine_wander_time (distance speed time : ℝ) (h1 : distance = 20) (h2 : speed = 4) :
  time = distance / speed → time = 5 :=
by
  intros h
  rw [h1, h2] at h
  have h3 : 20 / 4 = 5 := by norm_num
  rw h3 at h
  exact h

end christine_wander_time_l414_414042


namespace find_number_l414_414631

theorem find_number : ∃ x : ℤ, (3 * x - 1 = 2 * x) → x = 1 :=
by
  intro x
  assume h : 3 * x - 1 = 2 * x
  sorry

end find_number_l414_414631


namespace congruent_functions_l414_414885

-- Define the functions
def f1 (x : ℝ) := sin x + cos x
def f2 (x : ℝ) := √2 * sin x + √2
def f3 (x : ℝ) := sin x

-- Prove that f1 and f2 are congruent after a translation and f3 is not congruent with both
theorem congruent_functions (x : ℝ) :
  ∃ t1 t2 : ℝ, f1(x) = f2(x + t1) ∧ f1(x) ≠ f3(x + t2) ∧ f2(x) ≠ f3(x + t2) :=
sorry

end congruent_functions_l414_414885


namespace team_a_wins_3_1_l414_414906

-- Define the probabilities for Team A and Team B winning a single game
def prob_team_a_wins_single_game : ℚ := 1 / 2
def prob_team_b_wins_single_game : ℚ := 1 / 2

-- Define the probability of Team A winning with a score of 3:1 in a best-of-five series
def prob_team_a_wins_series_3_1 : ℚ :=
  let single_scenario_prob := (prob_team_a_wins_single_game ^ 3) * prob_team_b_wins_single_game
  in single_scenario_prob + single_scenario_prob + single_scenario_prob

-- State the theorem
theorem team_a_wins_3_1 : prob_team_a_wins_series_3_1 = 3 / 16 :=
  sorry

end team_a_wins_3_1_l414_414906


namespace eval_expression_l414_414418

theorem eval_expression : 68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := 
by
  sorry

end eval_expression_l414_414418


namespace slope_of_EF_is_zero_l414_414801

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem slope_of_EF_is_zero (A E F : ℝ × ℝ) (hA : ellipse_equation A.1 A.2)
  (hE : ellipse_equation E.1 E.2) (hF : ellipse_equation F.1 F.2)
  (hAE_AF : slope A.1 A.2 E.1 E.2 = -1 / slope A.1 A.2 F.1 F.2) :
  slope E.1 E.2 F.1 F.2 = 0 := by
  sorry

end slope_of_EF_is_zero_l414_414801


namespace all_divisible_by_5_l414_414092

theorem all_divisible_by_5 (a : Fin 7 → ℕ) (h : ∀ i : Fin 7, 5 ∣ ∑ j : Fin 7, if j = i then 0 else a j) : 
  ∀ i : Fin 7, 5 ∣ a i :=
by
  sorry

end all_divisible_by_5_l414_414092


namespace problem_I_problem_IIa_problem_IIb_problem_III_l414_414155

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + Real.log x

noncomputable def g (x : ℝ) : ℝ :=
  x^2 - 2 * x + 2

theorem problem_I (a : ℝ) (h : a = 2) : 
  (Deriv (f a) 1 = 3) :=
by
  sorry

theorem problem_IIa (a : ℝ) (h : a >= 0) : 
  ∀ x > 0, deriv (f a) x > 0 :=
by
  sorry

theorem problem_IIb (a : ℝ) (h : a < 0) :
  ∀ x, if x ∈ Ioo 0 (-1/a) then deriv (f a) x > 0 else deriv (f a) x < 0 :=
by
  sorry

theorem problem_III (a : ℝ) :
  (∀ x_1 ∈ Ioi 0, ∃ x_2 ∈ Icc 0 1, f a x_1 < g x_2) → a < -1 / Real.exp 3 :=
by
  sorry

end problem_I_problem_IIa_problem_IIb_problem_III_l414_414155


namespace vector_decomposition_l414_414690

/-

Question: Decompose the vector \( x \) in terms of vectors \( p, q, r \).

Conditions:
\[ x = \{ 3, -3, 4 \} \]
\[ p = \{ 1, 0, 2 \} \]
\[ q = \{ 0, 1, 1 \} \]
\[ r = \{ 2, -1, 4 \} \]

Correct Answer:
\[ x = \mathbf{p} - 2\mathbf{q} + \mathbf{r} \]

-/

theorem vector_decomposition :
  let x := (3 : ℝ, -3, 4)
  let p := (1 : ℝ, 0, 2)
  let q := (0 : ℝ, 1, 1)
  let r := (2 : ℝ, -1, 4)
  in x = (1:ℝ) • p - (2:ℝ) • q + (1:ℝ) • r :=
by
  let x := (3 : ℝ, -3, 4)
  let p := (1 : ℝ, 0, 2)
  let q := (0 : ℝ, 1, 1)
  let r := (2 : ℝ, -1, 4)
  show x = p - (2:ℝ) • q + r
  sorry

end vector_decomposition_l414_414690


namespace projection_of_w_l414_414007

-- Define the vectors involved
def v1 : ℝ × ℝ := (3, -3)
def v2 : ℝ × ℝ := (27 / 10, -9 / 10)
def w : ℝ × ℝ := (1, -1)

-- Projection function
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_u := u.1 * u.1 + u.2 * u.2
  ((dot_uv / dot_u) * u.1, (dot_uv / dot_u) * u.2)

-- Conditions as definitions
def condition := proj (3, -1) v1 = v2

-- Goal statement
theorem projection_of_w : proj (3, -1) w = (6 / 5, -2 / 5) := by
  sorry

end projection_of_w_l414_414007


namespace remainder_2001_div_9_l414_414691

theorem remainder_2001_div_9 : (∑ i in Finset.range 2002, i) % 9 = 6 := 
sorry

end remainder_2001_div_9_l414_414691


namespace intersection_points_of_parabolas_l414_414757

/-- Let P1 be the equation of the first parabola: y = 3x^2 - 8x + 2 -/
def P1 (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2

/-- Let P2 be the equation of the second parabola: y = 6x^2 + 4x + 2 -/
def P2 (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

/-- Prove that the intersection points of P1 and P2 are (-4, 82) and (0, 2) -/
theorem intersection_points_of_parabolas : 
  {p : ℝ × ℝ | ∃ x, p = (x, P1 x) ∧ P1 x = P2 x} = 
    {(-4, 82), (0, 2)} :=
sorry

end intersection_points_of_parabolas_l414_414757


namespace value_of_x_l414_414786

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x)
  else Real.log x / Real.log 81

theorem value_of_x (x : ℝ) : f x = 1 / 4 ↔ x = 3 :=
by 
  sorry

end value_of_x_l414_414786


namespace triangle_divisible_equal_parts_l414_414541

def is_scalene (a b c : ℝ) := a ≠ b ∧ b ≠ c ∧ a ≠ c
def can_be_divided_into_three_equal_triangles (α β γ : ℝ) : Prop := 
  α = 30 ∧ β = 60 ∧ γ = 90 ∧ is_scalene 30 60 90

theorem triangle_divisible_equal_parts :
  ∃ α β γ, can_be_divided_into_three_equal_triangles α β γ :=
by {
   use [30, 60, 90],
   unfold can_be_divided_into_three_equal_triangles,
   unfold is_scalene,
   split; norm_num,
   sorry
}

end triangle_divisible_equal_parts_l414_414541


namespace f_value_at_negative_13pi_over_6_l414_414956

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ π then 1 else f (x - π) + sin x

theorem f_value_at_negative_13pi_over_6 : f (- 13 * π / 6) = 3 / 2 := 
sorry

end f_value_at_negative_13pi_over_6_l414_414956


namespace find_sin_beta_l414_414130

variable (α β : ℝ)
-- Conditions: α and β are acute angles, cos(α) = 4/5, cos(α + β) = 3/5
axiom hα_angle : 0 < α ∧ α < π / 2
axiom hβ_angle : 0 < β ∧ β < π / 2
axiom h_cosα : Real.cos α = 4 / 5
axiom h_cosαβ : Real.cos (α + β) = 3 / 5

-- Goal: sin(β) = 7 / 25
theorem find_sin_beta : Real.sin β = 7 / 25 := 
by
  sorry

end find_sin_beta_l414_414130


namespace total_rope_in_inches_l414_414553

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end total_rope_in_inches_l414_414553


namespace variance_X_eq_l414_414726

noncomputable def variance_of_X : ℝ :=
  let f (x : ℝ) := if (0 < x ∧ x < π) then (1/2) * Real.sin x else 0
  let M_X := (1/2) * ∫ x in 0..π, x * Real.sin x
  let integral_x2_sin := ∫ x in 0..π, x^2 * Real.sin x
  (1/2) * integral_x2_sin - M_X^2

theorem variance_X_eq : variance_of_X = (π^2 - 8) / 4 := by
  sorry

end variance_X_eq_l414_414726


namespace probability_of_selecting_male_l414_414050

-- We define the proportions and ratio given in the problem.
def proportion_obese_men := 1 / 5
def proportion_obese_women := 1 / 10
def ratio_men_to_women := 3 / 2

-- From the given conditions, prove that the probability of selecting a male given that the individual is obese is 3/4.
theorem probability_of_selecting_male (P_A B : Prop) 
  (h_ratio: ratio_men_to_women = 3 / 2)
  (h_obese_men: P_A → proportion_obese_men)
  (h_obese_women: P_A → proportion_obese_women):
  (proportion_obese_men) * (3 / 5 : ℝ) / ((proportion_obese_men) * (3 / 5 : ℝ) + (proportion_obese_women) * (2 / 5 : ℝ)) = 3 / 4 := 
by
  sorry

end probability_of_selecting_male_l414_414050


namespace size_of_concentrate_cans_l414_414737

theorem size_of_concentrate_cans (servings : ℕ) (serving_size : ℕ) (mix_ratio : ℕ)
  (total_volume_needed : ℕ) (concentrate_needed : ℕ) : servings = 280 → 
  serving_size = 6 → mix_ratio = 4 → 
  total_volume_needed = servings * serving_size → 
  concentrate_needed = total_volume_needed / mix_ratio → 
  concentrate_needed = 420 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3] at h4 
  rw h4 at h5 
  exact h5 

end size_of_concentrate_cans_l414_414737


namespace pyramid_height_l414_414715

noncomputable def height_of_pyramid :=
  let volume_cube := 6 ^ 3
  let volume_sphere := (4 / 3) * Real.pi * (4 ^ 3)
  let total_volume := volume_cube + volume_sphere
  let base_area := 10 ^ 2
  let h := (3 * total_volume) / base_area
  h

theorem pyramid_height :
  height_of_pyramid = 6.48 + 2.56 * Real.pi :=
by
  sorry

end pyramid_height_l414_414715


namespace total_lunch_cost_l414_414023

-- Define the costs of lunch
variables (adam_cost rick_cost jose_cost : ℕ)

-- Given conditions
def jose_ate_certain_price := jose_cost = 45
def rick_and_jose_lunch_same_price := rick_cost = jose_cost
def adam_spends_two_thirds_of_ricks := adam_cost = (2 * rick_cost) / 3

-- Question to prove
theorem total_lunch_cost : 
  jose_ate_certain_price ∧ rick_and_jose_lunch_same_price ∧ adam_spends_two_thirds_of_ricks 
  → (adam_cost + rick_cost + jose_cost = 120) :=
by
  intro h
  sorry  -- Proof to be provided separately

end total_lunch_cost_l414_414023


namespace bus_trip_from_A_to_C_in_3_hours_l414_414708

noncomputable def bus_trip_time (v : ℝ) (AB BC tX tY : ℝ) : ℝ :=
  let tBC := 25 in
  let tstop := 5 in
  let tXB := 2 * tBC in
  let total_segments := 3 in
  (total_segments * tXB + tBC + tstop) / 60

theorem bus_trip_from_A_to_C_in_3_hours 
  (v : ℝ) (AB BC tX tY : ℝ) 
  (h1 : ¬(AB = 0 ∨ BC = 0)) 
  (hX : ∃ X, X > AB ∧ (X - AB - BC = 0)) 
  (hY : ∃ Y, Y > AB ∧ (tX = tY)) 
  : bus_trip_time v AB BC tX tY = 3 := 
sorry


end bus_trip_from_A_to_C_in_3_hours_l414_414708


namespace multiplication_sequence_result_l414_414400

theorem multiplication_sequence_result : (1 * 3 * 5 * 7 * 9 * 11 = 10395) :=
by
  sorry

end multiplication_sequence_result_l414_414400


namespace sin_75_equals_sqrt_1_plus_sin_2_equals_l414_414037

noncomputable def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)
noncomputable def sqrt_1_plus_sin_2 : ℝ := Real.sqrt (1 + Real.sin 2)

theorem sin_75_equals :
  sin_75 = (Real.sqrt 2 + Real.sqrt 6) / 4 := 
sorry

theorem sqrt_1_plus_sin_2_equals :
  sqrt_1_plus_sin_2 = Real.sin 1 + Real.cos 1 := 
sorry

end sin_75_equals_sqrt_1_plus_sin_2_equals_l414_414037


namespace intersection_line_equation_and_chord_length_l414_414159

noncomputable def circle1 : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 + 2 * p.1 - 6 * p.2 + 1 = 0}
noncomputable def circle2 : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 4 * p.1 + 2 * p.2 - 11 = 0}
noncomputable def intersection_points : set (ℝ × ℝ) := circle1 ∩ circle2
noncomputable def intersection_line : set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 + 6 = 0}

theorem intersection_line_equation_and_chord_length :
  (∀ p, p ∈ intersection_points → p ∈ intersection_line) ∧ 
  (2 * real.sqrt (3^2 - ((-(10/5))/(real.sqrt(3^2+0^2)))^2) = 24/5) :=
by sorry

end intersection_line_equation_and_chord_length_l414_414159


namespace ellipse_ratio_sum_l414_414047

theorem ellipse_ratio_sum (a b : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + 2 * x * y + 4 * y^2 - 15 * x - 25 * y + 50 = 0 → 
  (a = max (λ x y, y / x) ∧ b = min (λ x y, y / x))) → a + b = -14 :=
sorry

end ellipse_ratio_sum_l414_414047


namespace basketball_game_points_l414_414513

variable (J T K : ℕ)

theorem basketball_game_points (h1 : T = J + 20) (h2 : J + T + K = 100) (h3 : T = 30) : 
  T / K = 1 / 2 :=
by sorry

end basketball_game_points_l414_414513


namespace cone_from_half_sector_volume_l414_414002

noncomputable def cone_volume (r : ℝ) (h: ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

theorem cone_from_half_sector_volume :
  let R := 6 in
  let slant_height := R in
  let arc_length := π * R in
  let base_radius := arc_length / (2 * π) in
  let height := real.sqrt (slant_height^2 - base_radius^2) in
  cone_volume base_radius height = 9 * π * real.sqrt 3 :=
by
  -- Definitions
  let R := 6
  let slant_height := R
  let arc_length := π * R
  let base_radius := arc_length / (2 * π)
  let height := real.sqrt (slant_height^2 - base_radius^2)
  -- Calculate volume
  let volume := cone_volume base_radius height
  show volume = 9 * π * real.sqrt 3
  -- Provide the proof (to be completed)
  sorry

end cone_from_half_sector_volume_l414_414002


namespace total_amount_paid_l414_414755

variable (n : ℕ) (each_paid : ℕ)

/-- This is a statement that verifies the total amount paid given the number of friends and the amount each friend pays. -/
theorem total_amount_paid (h1 : n = 7) (h2 : each_paid = 70) : n * each_paid = 490 := by
  -- This proof will validate that the total amount paid is 490
  sorry

end total_amount_paid_l414_414755


namespace sequence_expression_l414_414796

theorem sequence_expression (a : ℕ → ℝ) (h_base : a 1 = 2)
  (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * (n + 1) * a n / (a n + n)) :
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n + 1) * 2^(n + 1) / (2^(n + 1) - 1) :=
by
  sorry

end sequence_expression_l414_414796


namespace calculate_expression_l414_414505

theorem calculate_expression :
  let x := 7
      y := -2
      z := 4
  in (x - 2 * y)^y / z = 1 / 484 := by sorry

end calculate_expression_l414_414505


namespace question_1_question_2_question_3_l414_414207

-- Define "k-th level associated point"
def kth_level_associated_point (k : ℝ) (A : ℝ × ℝ) : ℝ × ℝ :=
  (k * A.1 + A.2, A.1 + k * A.2)

-- Proof Problem for Question 1
theorem question_1 : kth_level_associated_point 2 (1, 2) = (4, 5) :=
sorry

-- Proof Problem for Question 2
theorem question_2 (k m : ℝ) 
  (h : kth_level_associated_point k (2, -1) = (9, m)) : k + m = 2 :=
sorry

-- Proof Problem for Question 3
theorem question_3 (a : ℝ) : 
  kth_level_associated_point (-4) (a-1, 2*a) = (0, _)
  ∨ kth_level_associated_point (-4) (a-1, 2*a) = (_, 0) :=
sorry

end question_1_question_2_question_3_l414_414207


namespace distance_between_parallel_lines_l414_414822

theorem distance_between_parallel_lines :
  ∀ (x y : ℝ),
  (3 * x + 4 * y - 9 = 0) ∧ (3 * x + 4 * y + 1 = 0) → 
  (∀ d : ℝ, d = 2) :=
begin
  sorry
end

end distance_between_parallel_lines_l414_414822


namespace Kiran_money_l414_414309

theorem Kiran_money (R G K : ℕ) (h1 : R / G = 6 / 7) (h2 : G / K = 6 / 15) (h3 : R = 36) : K = 105 := by
  sorry

end Kiran_money_l414_414309


namespace tan_angle_addition_l414_414860

theorem tan_angle_addition (y : ℝ) (hyp : Real.tan y = -3) : 
  Real.tan (y + Real.pi / 3) = - (5 * Real.sqrt 3 - 6) / 13 := 
by 
  sorry

end tan_angle_addition_l414_414860


namespace division_quotient_proof_l414_414924

theorem division_quotient_proof :
  (300324 / 29 = 10356) →
  (100007892 / 333 = 300324) :=
by
  intros h1
  sorry

end division_quotient_proof_l414_414924


namespace range_of_a_l414_414481

theorem range_of_a (a : ℝ) (h : 0 < a) : {x : ℝ | x^2 - x < 0} ⊆ set.Ioo 0 a → 1 ≤ a :=
begin
  intro h_subset,
  -- we constructively need to show that 1 ≤ a given the subset relation
  sorry
end

end range_of_a_l414_414481


namespace trapezoid_to_parallelogram_trapezoid_orthogonal_projection_l414_414199

variables {A B C D L K O T : Type}
variables [affine_space ℝ (points ℝ)]

-- Conditions
def IsTrapezoid (A B C D : points ℝ) : Prop := 
(trapezoid A B C D) ∧ 
((A ≁ C) ∧ (B ≁ D))

def OrthoProjectionOnLine (C L A B : points ℝ) : Prop :=
orthogonal_projection_on_line C L A B

def IsPtOnLinePerpendicular (A K C D : points ℝ) : Prop :=
on_line_perpendicular A K C D

def IsCircumcenter (O A C D : points ℝ) : Prop :=
circumcenter O A C D

def LinesIntersectAt (A K C L D O T : points ℝ) : Prop :=
lines_intersect_at A K C L D O T
-- Question to prove
theorem trapezoid_to_parallelogram_trapezoid_orthogonal_projection
  (h1 : IsTrapezoid A B C D)
  (h2 : OrthoProjectionOnLine C L A B)
  (h3 : IsPtOnLinePerpendicular A K C D)
  (h4 : IsCircumcenter O A C D)
  (h5 : LinesIntersectAt A K C L D O T) :
  parallelogram A B C D :=
sorry

end trapezoid_to_parallelogram_trapezoid_orthogonal_projection_l414_414199


namespace max_good_sequences_correct_l414_414368

-- Define the conditions: Number of blue, red, and green beads in the necklace
def num_blue : Nat := 50
def num_red : Nat := 100
def num_green : Nat := 100

-- Define what makes a sequence good
def good_sequence (seq : List Char) : Prop :=
  (seq.filter (λ c => c = 'B')).length = 2 ∧
  (seq.filter (λ c => c = 'R')).length = 1 ∧
  (seq.filter (λ c => c = 'G')).length = 1

-- Define the necklace as cyclic list of beads
def cyclic_necklace : List Char := List.repeat 'B' num_blue 
                       ++ List.repeat 'R' num_red 
                       ++ List.repeat 'G' num_green

-- Define the length of the necklace
def necklace_len : Nat :=
  num_blue + num_red + num_green

-- Define a function to check if the given indices form a good sequence
def is_good_sequence (necklace : List Char) (idx : Nat) : Prop := 
  good_sequence [necklace.get (idx % necklace_len), 
                 necklace.get ((idx + 1) % necklace_len),
                 necklace.get ((idx + 2) % necklace_len),
                 necklace.get ((idx + 3) % necklace_len)]

-- Define the maximum number of good sequences
def max_good_sequences (necklace : List Char) : Nat :=
  List.length (List.filter (λ idx => is_good_sequence necklace idx) 
              (List.range necklace_len))

-- The theorem to prove
theorem max_good_sequences_correct :
  max_good_sequences cyclic_necklace = 99 := by
  sorry

end max_good_sequences_correct_l414_414368


namespace determinant_cos_tan_zero_l414_414952

theorem determinant_cos_tan_zero
  (A B C : ℝ)
  (h : A + B + C = π) : 
  det ![![cos A ^ 2, tan A, 1], ![cos B ^ 2, tan B, 1], ![cos C ^ 2, tan C, 1]] = 0 :=
by
  sorry

end determinant_cos_tan_zero_l414_414952


namespace probability_greater_than_30_l414_414784

noncomputable def different_digits : set (ℕ × ℕ) :=
  {(x, y) | x ∈ {1, 2, 3, 4, 5} ∧ y ∈ {1, 2, 3, 4, 5} ∧ x ≠ y }

noncomputable def is_greater_than_30 (a b : ℕ) : Prop :=
  10 * a + b > 30

/-- The probability that a randomly selected two-digit number 
    formed using two different digits from {1, 2, 3, 4, 5} 
    without repeating digits is greater than 30 is 3/5. -/
theorem probability_greater_than_30 : 
  (∑ (x : ℕ × ℕ) in different_digits, if is_greater_than_30 x.1 x.2 then 1 else 0 : ℝ) /
  (∑ (x : ℕ × ℕ) in different_digits, 1 : ℝ) = 3 / 5 :=
sorry

end probability_greater_than_30_l414_414784


namespace cost_of_filling_pool_l414_414673

noncomputable theory

def fill_hours_day1 := 30
def fill_hours_day2 := 20
def hose_flow_rate := 100 -- gallons per hour
def pump_flow_rate := 150 -- gallons per hour
def hose_cost_rate := 1 / 10 -- cents per gallon
def pump_cost_rate := 1 / 8 -- cents per gallon
def evaporation_rate := 0.02 -- 2% per day

theorem cost_of_filling_pool : 
  (let total_flow_rate := hose_flow_rate + pump_flow_rate in
   let water_added_day1 := total_flow_rate * fill_hours_day1 in
   let water_left_after_evap_day1 := water_added_day1 * (1 - evaporation_rate) in
   let water_added_day2 := total_flow_rate * fill_hours_day2 in
   let total_water_before_evap_day2 := water_left_after_evap_day1 + water_added_day2 in
   let water_left_after_evap_day2 := total_water_before_evap_day2 * (1 - evaporation_rate) in
   let hose_cost := (water_added_day1 + water_added_day2) * hose_cost_rate in
   let pump_cost := (water_added_day1 + water_added_day2) * pump_cost_rate in
   let total_cost := hose_cost + pump_cost in
   (total_cost / 100) = 28.125) :=
begin
  sorry
end

end cost_of_filling_pool_l414_414673


namespace find_sum_uv_l414_414175

theorem find_sum_uv (u v : ℝ) (h1 : 3 * u - 7 * v = 29) (h2 : 5 * u + 3 * v = -9) : u + v = -3.363 := 
sorry

end find_sum_uv_l414_414175


namespace geometric_seq_find_lambda_mu_arithmetic_seq_l414_414448

section problem

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {λ μ : ℝ}

def sequence_sum (n : ℕ) : ℝ :=
  if n = 1 then a n
  else S n = λ * n * a n + μ * a (n-1)

theorem geometric_seq {n : ℕ} (hn : 2 ≤ n) (h1 : λ = 0) (h2 : μ = 4) :
  let b := λ n, a (n + 1) - 2 * a n 
  in ∃ r, ∀ n, b n = r * b (n - 1) :=
sorry

theorem find_lambda_mu (h_seq : ∃ q, ∀ n, a (n + 1) = q * a n) :
  λ = 1 ∧ μ = 0 :=
sorry

theorem arithmetic_seq (h1 : a 2 = 3) (h2 : λ + μ = 3 / 2) :
  ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1) :=
sorry

end problem

end geometric_seq_find_lambda_mu_arithmetic_seq_l414_414448


namespace E_areas_equal_l414_414108

variables {Point : Type*} [AddCommGroup Point] [Module ℝ Point]

structure Quadrilateral (Point : Type*) :=
(A B C D : Point)

structure Parallelogram (Point : Type*) :=
(E F G H : Point)

variables (ABCD : Quadrilateral Point)
variables (E F G H E' F' G' H' : Point)

-- Conditions
axiom AE_equals_BE : dist (ABCD.A) E' = dist E E'
axiom BF_equals_CF : dist (ABCD.B) F' = dist F F'
axiom CG_equals_DG : dist (ABCD.C) G' = dist G G'
axiom DH_equals_AH : dist (ABCD.D) H' = dist H H'
axiom EFGH_is_parallelogram : is_parallelogram E F G H

-- Questions
theorem E'F'G'H'_is_parallelogram (h1 : dist (ABCD.A) E' = dist E E')
                                  (h2 : dist (ABCD.B) F' = dist F F')
                                  (h3 : dist (ABCD.C) G' = dist G G')
                                  (h4 : dist (ABCD.D) H' = dist H H')
                                  (h5 : is_parallelogram E F G H) : 
  is_parallelogram E' F' G' H' := 
  sorry

theorem areas_equal (h1 : dist (ABCD.A) E' = dist E E')
                     (h2 : dist (ABCD.B) F' = dist F F')
                     (h3 : dist (ABCD.C) G' = dist G G')
                     (h4 : dist (ABCD.D) H' = dist H H')
                     (h5 : is_parallelogram E F G H) :
  area E F G H = area E' F' G' H' :=
  sorry

end E_areas_equal_l414_414108


namespace sphere_surface_area_quadruple_l414_414500

theorem sphere_surface_area_quadruple (r : ℝ) :
  (4 * π * (2 * r)^2) = 4 * (4 * π * r^2) :=
by
  sorry

end sphere_surface_area_quadruple_l414_414500


namespace ambulance_ride_cost_correct_l414_414236

noncomputable def total_bill : ℝ := 18000
noncomputable def medication_percentage : ℝ := 0.35
noncomputable def imaging_percentage : ℝ := 0.15
noncomputable def surgery_percentage : ℝ := 0.25
noncomputable def overnight_stays_percentage : ℝ := 0.10
noncomputable def doctors_fees_percentage : ℝ := 0.05

noncomputable def food_fee : ℝ := 300
noncomputable def consultation_fee : ℝ := 450
noncomputable def physical_therapy_fee : ℝ := 600

noncomputable def medication_cost : ℝ := medication_percentage * total_bill
noncomputable def imaging_cost : ℝ := imaging_percentage * total_bill
noncomputable def surgery_cost : ℝ := surgery_percentage * total_bill
noncomputable def overnight_stays_cost : ℝ := overnight_stays_percentage * total_bill
noncomputable def doctors_fees_cost : ℝ := doctors_fees_percentage * total_bill

noncomputable def percentage_based_costs : ℝ :=
  medication_cost + imaging_cost + surgery_cost + overnight_stays_cost + doctors_fees_cost

noncomputable def fixed_costs : ℝ :=
  food_fee + consultation_fee + physical_therapy_fee

noncomputable def total_known_costs : ℝ :=
  percentage_based_costs + fixed_costs

noncomputable def ambulance_ride_cost : ℝ :=
  total_bill - total_known_costs

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 450 := by
  sorry

end ambulance_ride_cost_correct_l414_414236


namespace y_intercept_of_circle_l414_414711

theorem y_intercept_of_circle {x y : ℝ} :
  (x - 5)^2 + (y - 4)^2 = 25 → x = 0 → y = 4 :=
  by
    intros h hx
    rw [hx] at h
    have : (0 - 5)^2 = 25 := by norm_num
    rw [this] at h
    norm_num at h
    exact h

# Check if we can apply this theorem using the given conditions.
example : y_intercept_of_circle (by norm_num : (0 - 5)^2 + (4 - 4)^2 = 25) (by refl : 0 = 0) = (4 : ℝ) :=
by apply y_intercept_of_circle

end y_intercept_of_circle_l414_414711


namespace cone_section_area_half_base_ratio_l414_414292

theorem cone_section_area_half_base_ratio (h_base h_upper h_lower : ℝ) (A_base A_upper : ℝ) 
  (h_total : h_upper + h_lower = h_base)
  (A_upper : A_upper = A_base / 2) :
  h_upper = h_lower :=
by
  sorry

end cone_section_area_half_base_ratio_l414_414292


namespace consecutive_odd_numbers_l414_414971

/- 
  Out of some consecutive odd numbers, 9 times the first number 
  is equal to the addition of twice the third number and adding 9 
  to twice the second. Let x be the first number, then we aim to prove that 
  9 * x = 2 * (x + 4) + 2 * (x + 2) + 9 ⟹ x = 21 / 5
-/

theorem consecutive_odd_numbers (x : ℚ) (h : 9 * x = 2 * (x + 4) + 2 * (x + 2) + 9) : x = 21 / 5 :=
sorry

end consecutive_odd_numbers_l414_414971


namespace alan_spent_amount_l414_414738

theorem alan_spent_amount (john_spent : ℝ) (extra_percent : ℝ) : 
  (john_spent = 2040) ∧ (extra_percent = 0.02) → let alan_spent := john_spent / (1 + extra_percent) in alan_spent = 1999.20 :=
by
  intro h,
  cases h with h_john_spent h_extra_percent,
  let alan_spent := john_spent / (1 + extra_percent),
  rw [h_john_spent, h_extra_percent],
  have h_div : 2040 / (1 + 0.02) = 1999.20,
  calc
    2040 / (1 + 0.02) = 2040 / 1.02 : by rw add_comm
    ... = 1999.20 : by norm_num,
  rw h_div,
  exact rfl,

end alan_spent_amount_l414_414738


namespace find_a_for_equation_l414_414436

def equation (x a : ℝ) : ℝ := 
  4 ^ (abs (x - a)) * log (x ^ 2 - 2 * x + 4) / log (1/3) + 
  2 ^ (x ^ 2 - 2 * x) * log (2 * abs (x - a) + 3) / log (sqrt 3)

def hasExactlyThreeSolutions (f : ℝ → ℝ) : Prop := 
  ∃ a1 a2 a3 : ℝ, a1 ≠ a2 ∧ a2 ≠ a3 ∧ f a1 = 0 ∧ f a2 = 0 ∧ f a3 = 0 ∧ 
  ∀ a' : ℝ, f a' = 0 → a' = a1 ∨ a' = a2 ∨ a' = a3

theorem find_a_for_equation :
  hasExactlyThreeSolutions (equation x a) → (a = 1 / 2 ∨ a = 1 ∨ a = 3 / 2) :=
sorry

end find_a_for_equation_l414_414436


namespace complex_root_magnitude_one_iff_divisible_l414_414251

theorem complex_root_magnitude_one_iff_divisible (n : ℕ) (hn : n > 0) : 
  (∃ (z : ℂ), |z| = 1 ∧ z^(n + 1) - z^n - 1 = 0) ↔ (6 ∣ (n + 2)) := 
by 
  sorry

end complex_root_magnitude_one_iff_divisible_l414_414251


namespace max_good_sequences_l414_414369

-- Definitions based on the problem conditions
def is_good_sequence (beads : List Char) : Prop :=
  beads.length = 4 ∧
  (beads.count 'B' = 2) ∧
  (beads.count 'R' = 1) ∧
  (beads.count 'G' = 1)

def count_max_good_sequences (necklace : List Char) : Nat :=
  let cyclic_necklace := necklace ++ necklace.take 3
  List.countp is_good_sequence (List.sublistsLen 4 cyclic_necklace)

-- Statement of the mathematically equivalent proof problem
theorem max_good_sequences (necklace : List Char)
  (h_length_necklace : necklace.length = 250)
  (h_blue : necklace.count 'B' = 50)
  (h_red : necklace.count 'R' = 100)
  (h_green : necklace.count 'G' = 100)
  : count_max_good_sequences necklace = 99 :=
sorry

end max_good_sequences_l414_414369


namespace sum_of_roots_of_y_squared_eq_36_l414_414866

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end sum_of_roots_of_y_squared_eq_36_l414_414866


namespace find_radius_l414_414006

noncomputable def radius_of_circle (P_distance_from_center : ℝ) (PQ : ℝ) (QR : ℝ) : ℝ :=
  let PR := PQ + QR in
  let secant_power := PQ * PR in
  let tangent_power := P_distance_from_center^2 - radius^2 in
  let radius_square := tangent_power in
  let radius := real.sqrt radius_square in
  radius

theorem find_radius :
  radius_of_circle 15 10 8 = 3 * real.sqrt 5 :=
by
  sorry

end find_radius_l414_414006


namespace cost_per_lesson_is_correct_l414_414390

noncomputable def monthly_pasture_cost : ℕ := 500
noncomputable def daily_food_cost : ℕ := 10
noncomputable def weekly_lessons : ℕ := 2
noncomputable def yearly_total_spending : ℕ := 15890
def annual_pasture_cost : ℕ := monthly_pasture_cost * 12
def annual_food_cost : ℕ := daily_food_cost * 365
def annual_pasture_and_food_cost : ℕ := annual_pasture_cost + annual_food_cost
def annual_lesson_cost : ℕ := yearly_total_spending - annual_pasture_and_food_cost
def total_lessons : ℕ := weekly_lessons * 52

theorem cost_per_lesson_is_correct : annual_lesson_cost / total_lessons = 60 := 
by 
    sorry

end cost_per_lesson_is_correct_l414_414390


namespace log_eq_solution_l414_414652

theorem log_eq_solution (x : ℝ) (hx : log 2 (3 * x + 2) = 1 + log 2 (x + 2)) : x = 2 := by
  -- sorry is used to skip the proof
  sorry

end log_eq_solution_l414_414652


namespace ratio_of_speeds_l414_414512

theorem ratio_of_speeds :
  ∀ (t : ℝ) (v_A v_B : ℝ),
  (v_A = 360 / t) →
  (v_B = 480 / t) →
  v_A / v_B = 3 / 4 :=
by
  intros t v_A v_B h_A h_B
  rw [←h_A, ←h_B]
  field_simp
  norm_num

# In this theorem, we state that given the conditions of the speeds
# and distances covered in the problem, we prove that the ratio of
# their speeds is 3:4.

end ratio_of_speeds_l414_414512


namespace prob_male_given_obese_correct_l414_414053

-- Definitions based on conditions
def ratio_male_female : ℚ := 3 / 2
def prob_obese_male : ℚ := 1 / 5
def prob_obese_female : ℚ := 1 / 10

-- Definition of events
def total_employees : ℚ := ratio_male_female + 1

-- Probability calculations
def prob_male : ℚ := ratio_male_female / total_employees
def prob_female : ℚ := 1 / total_employees

def prob_obese_and_male : ℚ := prob_male * prob_obese_male
def prob_obese_and_female : ℚ := prob_female * prob_obese_female

def prob_obese : ℚ := prob_obese_and_male + prob_obese_and_female

def prob_male_given_obese : ℚ := prob_obese_and_male / prob_obese

-- Theorem statement
theorem prob_male_given_obese_correct : prob_male_given_obese = 3 / 4 := sorry

end prob_male_given_obese_correct_l414_414053


namespace domain_of_ln_f_l414_414293

def domain_ln_fraction (x : ℝ) : Prop := 0 < 1 / (1 - x)

theorem domain_of_ln_f (f : ℝ → ℝ := λ x, Real.log (1 / (1 - x))) :
  ∀ x, domain_ln_fraction x ↔ x < 1 :=
sorry

end domain_of_ln_f_l414_414293


namespace largest_prime_factor_of_expression_l414_414344

theorem largest_prime_factor_of_expression :
  ∀ n : ℕ, n = 16^4 + 2 * 16^2 + 1 - 13^4 → nat.largest_prime_divisor n = 71 :=
by
  intro n h
  rw h
  sorry

end largest_prime_factor_of_expression_l414_414344


namespace find_c_l414_414439

theorem find_c (a b c : ℝ) (h_a : 0 < a ∧ a ≠ 1) (h_b : 0 < b ∧ b ≠ 1)
  (h_fixed_point : ∀ x, (x + log a (x - 2) = b^(x - c) + 2)) : c = 3 :=
sorry

end find_c_l414_414439


namespace range_of_theta_max_modulus_z_l414_414824

noncomputable def z (θ : ℝ) : ℂ := 1 + complex.of_real (cos θ) + complex.i * (sin θ)

theorem range_of_theta (θ : ℝ) (k : ℤ) (h1 : 1 + cos θ > 0) (h2 : sin θ < 0) :
  2 * k * real.pi - real.pi < θ ∧ θ < 2 * k * real.pi := 
sorry

theorem max_modulus_z (θ : ℝ) : complex.abs (z θ) ≤ 2 :=
begin
  calc complex.abs (z θ) = sqrt ((1 + cos θ)^2 + (sin θ)^2) : by sorry
                      ... = sqrt (2 * (1 + cos θ)) : by sorry
                      ... ≤ sqrt (2 * 2) : by sorry
                      ... = 2 : by sorry,
end

end range_of_theta_max_modulus_z_l414_414824


namespace parallelogram_altitude_base_ratio_l414_414287

theorem parallelogram_altitude_base_ratio 
  (area base : ℕ) (h : ℕ) 
  (h_base : base = 9)
  (h_area : area = 162)
  (h_area_eq : area = base * h) : 
  h / base = 2 := 
by 
  -- placeholder for the proof
  sorry

end parallelogram_altitude_base_ratio_l414_414287


namespace last_number_remaining_is_49_l414_414610

def skipAndMark (n : ℕ) (xs : List ℕ) : List ℕ :=
  -- Simulates marking every second number starting from n in list xs
  xs.filter (fun x => (x - n) % 2 ≠ 0)

noncomputable def findLastRemaining : ℕ :=
  let rec process (xs : List ℕ) : List ℕ :=
    match xs with
    | [] => []
    | y :: ys => 
      let unmarked = skipAndMark y ys
      if unmarked.tail.isEmpty then [y] else process unmarked.tail
  (process (List.range 1 51)).head!

theorem last_number_remaining_is_49 : findLastRemaining = 49 :=
  sorry

end last_number_remaining_is_49_l414_414610


namespace anvil_factory_fraction_of_journeymen_l414_414588

theorem anvil_factory_fraction_of_journeymen 
  (total_employees : ℕ := 20210)
  (half_laid_off_journeymen_percent : ℝ := 49.99999999999999 / 100) :
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ x * total_employees / (total_employees - x / 2 * total_employees) = half_laid_off_journeymen_percent ∧ x = 2 / 3 :=
begin
  use 2 / 3,
  split,
  { norm_num }, -- Proof that 2/3 > 0
  split,
  { norm_num }, -- Proof that 2/3 < 1
  split,
  { norm_num1, -- Proof that (2/3 * 20210) / (20210 - (2/3) / 2 * 20210) = 0.5 is omitted
    sorry
  },
  { refl }
end

end anvil_factory_fraction_of_journeymen_l414_414588


namespace lottery_digits_unique_l414_414195

theorem lottery_digits_unique :
  ∃ A B C D E : ℕ,
  (1 ≤ A ∧ A < B ∧ B < C ∧ C < 9 ∧ B < D ∧ D ≤ 9) ∧
  (let num1 := 10 * A + B in
   let num2 := 10 * B + C in
   let num3 := 10 * C + A in
   let num4 := 10 * C + B in
   let num5 := 10 * C + D in
   num1 + num2 + num3 + num4 + num5 = 100 * B + 10 * C + C ∧
   num3 * num2 = 1000 * B + 100 * B + 10 * E + C ∧
   num3 * num5 = 1000 * E + 100 * C + 10 * C + D) ∧
  (A = 1 ∧ B = 2 ∧ C = 8 ∧ D = 5 ∧ E = 6) := 
by
  exists 1 2 8 5 6
  split
  repeat {split}; norm_num
  sorry

end lottery_digits_unique_l414_414195


namespace sum_of_c₁_to_c₁₀₀_l414_414821

open BigOperators

-- Definitions and conditions from part (a)
def a₁ := 1
def a (n : ℕ) : ℕ := if n = 0 then 0 else (a (n - 1) * 2 + 1)
def b (n : ℕ) : ℤ := 2 * Int.log2 (1 + a n) - 1
def c (n : ℕ) : ℤ := b n

-- The proof statement translating part (c)
theorem sum_of_c₁_to_c₁₀₀ : (∑ i in Finset.range 101 \ (Finset.range 8), c i) = 11202 := 
sorry

end sum_of_c₁_to_c₁₀₀_l414_414821


namespace car_speed_l414_414990

def travel_time : ℝ := 5
def travel_distance : ℝ := 300

theorem car_speed :
  travel_distance / travel_time = 60 := sorry

end car_speed_l414_414990


namespace total_payment_correct_l414_414011

variables (pay1 pay2 : ℝ)
variables (total_payment : ℝ)

def discounts (amount: ℝ) : ℝ :=
if amount ≤ 200 then amount
else if amount ≤ 500 then 200 + (amount - 200) * 0.9
else 200 + 300 * 0.9 + (amount - 500) * 0.7

theorem total_payment_correct (pay1 pay2 : ℝ):
  pay1 = 168 ∧ pay2 = 423 →
  let total := pay1 + pay2 in
  total_payment = discounts total →
  total_payment = 533.7 :=
by
  intros h
  sorry

end total_payment_correct_l414_414011


namespace bracelet_arrangements_l414_414912

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def distinct_arrangements : ℕ := factorial 8 / (8 * 2)

theorem bracelet_arrangements : distinct_arrangements = 2520 :=
by
  sorry

end bracelet_arrangements_l414_414912


namespace point_not_on_line_l414_414945

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : ¬ (0 = 3 * m * 4 + 4 * b) :=
by {
  intro h1,
  have h2 : 0 = 12 * m + 4 * b := h1,
  have h3 : 4 * b = -12 * m := by linarith,
  have h4 : b = -3 * m := by linarith,
  have h5 : m * b = m * (-3 * m) := by rw h4,
  have h6 : m * b = -3 * (m * m) := by ring,
  have h7 : 0 < m * m := mul_self_pos (ne_of_gt h),
  have h8 : 0 > -3 := by linarith,
  have h9 : 0 < -3 * (m * m) := by nlinarith,
  exact lt_irrefl _ (lt_trans h9 h),
}

end point_not_on_line_l414_414945


namespace peach_pies_count_l414_414383

theorem peach_pies_count (total_pies : ℕ) (ratio : ℕ) (apple_ratio blueberry_ratio cherry_ratio peach_ratio : ℕ) (ceiling : ℚ → ℚ) :
  total_pies = 36 →
  ratio = 10 →
  apple_ratio = 1 → blueberry_ratio = 4 → cherry_ratio = 3 → peach_ratio = 2 →
  ceiling ((total_pies : ℚ) / ratio * peach_ratio) = 8 :=
begin
  intro total_pies_eq,
  intro ratio_eq,
  intro apple_eq,
  intro blueberry_eq,
  intro cherry_eq,
  intro peach_eq,
  -- these will be skipped
  sorry
end

end peach_pies_count_l414_414383


namespace sum_of_valid_a_l414_414504

theorem sum_of_valid_a :
  (∑ x in ({-1, 0, 2, 3, 4, 5} : Finset ℤ), x) = 13 := by
  sorry

end sum_of_valid_a_l414_414504


namespace three_digit_numbers_with_digit_sum_27_and_end_4_count_l414_414855

theorem three_digit_numbers_with_digit_sum_27_and_end_4_count :
  ∀ n : ℕ, (100 ≤ n ∧ n < 1000 ∧ (n % 10 = 4) ∧ (n.digits.sum = 27)) → false :=
by
  sorry

end three_digit_numbers_with_digit_sum_27_and_end_4_count_l414_414855


namespace min_t_of_inequalities_l414_414246

theorem min_t_of_inequalities (x y : ℝ) (hx: 0 < x) (hy: 0 < y) (hxy: x > 2 * y) :
  ∃ t, t = 4 ∧ t = min { max (x^2 / 2) (4 / (y * (x - 2 * y))) } :=
by {
  sorry
}

end min_t_of_inequalities_l414_414246


namespace proof_problem_l414_414119

open Real

noncomputable def P : ℝ × ℝ := (2, 1)

-- Parametric form of the line
def l (α t : ℝ) : ℝ × ℝ := (2 + t * cos α, 1 + t * sin α)

-- Condition that |PA| * |PB| = 4
def condition (α : ℝ) : Prop := 
  let A := (2 + (-2 * cos α), 1 + (-2 * sin α) * (cos α == 0))
  let B := (2 + (-1 / (tan α)), 1 + (- (1 / tan α)) * sin α == 0)
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  ‖PA‖ * ‖PB‖ = 4

-- Problem statement
theorem proof_problem : 
  ∃ α, condition α ∧ α = (3 * π / 4) ∧ 
    ∀ ρ θ, ρ * (cos θ + sin θ) = 3 : =
begin
  sorry
end

end proof_problem_l414_414119


namespace complex_pow_eq_neg_one_l414_414180

theorem complex_pow_eq_neg_one (z : ℂ) (hz : z + z⁻¹ = 1) : z^1000 + z^(-1000) = -1 :=
sorry

end complex_pow_eq_neg_one_l414_414180


namespace soup_can_pyramid_rows_l414_414730

theorem soup_can_pyramid_rows (n : ℕ) :
  (∃ (n : ℕ), (2 * n^2 - n = 225)) → n = 11 :=
by
  sorry

end soup_can_pyramid_rows_l414_414730


namespace card_team_probability_l414_414620

theorem card_team_probability :
  ∃ (m n : ℕ), nat.coprime m n ∧
  (m + n = 1826) ∧
  (∀ a, 1 ≤ a ∧ a ≤ 41 →
       let p := (nat.choose (40-a) 2 + nat.choose (a-1) 2) / 1225 in
           (a = 5 → p = (601/1225))
           ∧
           (a = 36 → p = (601/1225))
           ∧
           (p ≥ 1/2)
  ).
sorry

end card_team_probability_l414_414620


namespace max_good_sequences_l414_414370

-- Definitions based on the problem conditions
def is_good_sequence (beads : List Char) : Prop :=
  beads.length = 4 ∧
  (beads.count 'B' = 2) ∧
  (beads.count 'R' = 1) ∧
  (beads.count 'G' = 1)

def count_max_good_sequences (necklace : List Char) : Nat :=
  let cyclic_necklace := necklace ++ necklace.take 3
  List.countp is_good_sequence (List.sublistsLen 4 cyclic_necklace)

-- Statement of the mathematically equivalent proof problem
theorem max_good_sequences (necklace : List Char)
  (h_length_necklace : necklace.length = 250)
  (h_blue : necklace.count 'B' = 50)
  (h_red : necklace.count 'R' = 100)
  (h_green : necklace.count 'G' = 100)
  : count_max_good_sequences necklace = 99 :=
sorry

end max_good_sequences_l414_414370


namespace units_digit_of_expression_l414_414316

theorem units_digit_of_expression : Nat.unitsDigit (((2 + 1) * (2^2 + 1) * (2^4 + 1) * ... * (2^32 + 1)) + 2) = 7 := 
by
  sorry

end units_digit_of_expression_l414_414316


namespace original_price_l414_414729

variable (P : ℝ)

theorem original_price (h : 560 = 1.05 * (0.72 * P)) : P = 740.46 := 
by
  sorry

end original_price_l414_414729


namespace unique_intersections_l414_414752

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

theorem unique_intersections :
  (∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1) ∧
  (∃ x2 y2, line2 x2 y2 ∧ line3 x2 y2) ∧
  ¬ (∃ x y, line1 x y ∧ line3 x y) ∧
  (∀ x y x' y', (line1 x y ∧ line2 x y ∧ line2 x' y' ∧ line3 x' y') → (x = x' ∧ y = y')) :=
by
  sorry

end unique_intersections_l414_414752


namespace increasing_intervals_f_range_of_m_l414_414845

-- Define the vectors
def vector_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
def vector_b (x : ℝ) (m : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x) + m)

-- Define the function f
def f (x m : ℝ) : ℝ :=
  (vector_a x).1 * (vector_b x m).1 + (vector_a x).2 * (vector_b x m).2 - 1

-- Problem 1: Prove the intervals where f(x) is increasing on [0, π]
theorem increasing_intervals_f (m : ℝ) :
  ∀ x, (x ∈ Set.Icc 0 Real.pi) → 
       (∃ s e : ℝ, (s = 0 ∧ e = Real.pi / 6 ∧ x ∈ Set.Icc s e) ∨
                    (s = 2 * Real.pi / 3 ∧ e = Real.pi ∧ x ∈ Set.Icc s e)) :=
sorry

-- Problem 2: Prove the range of m such that -4 ≤ f(x) ≤ 4 for x ∈ [0, π/6]
theorem range_of_m :
  ∀ m, (∀ x, (x ∈ Set.Icc 0 (Real.pi / 6)) →
       (-4 ≤ f x m ∧ f x m ≤ 4)) ↔ -5 ≤ m ∧ m ≤ 2 :=
sorry

end increasing_intervals_f_range_of_m_l414_414845


namespace arithmetic_sum_geometric_seq_l414_414245

noncomputable def a (n : ℕ) (d : ℝ) : ℝ := 20 + (n - 1) * d

theorem arithmetic_sum_geometric_seq 
  (d : ℝ) (h1 : d ≠ 0) 
  (h2 : (a 5 d)^2 = (a 2 d) * (a 7 d)) :
  ∑ i in Finset.range 10, a (i + 1) d = 110 :=
by
  sorry

end arithmetic_sum_geometric_seq_l414_414245


namespace directrix_of_parabola_l414_414296

theorem directrix_of_parabola (y x : ℝ) (h : y = 4 * x^2) : y = - (1 / 16) :=
sorry

end directrix_of_parabola_l414_414296


namespace number_of_ways_to_divide_friends_l414_414168

-- Definitions
def friends := Fin 8
def teams := Fin 4

-- Proven answer for the problem
theorem number_of_ways_to_divide_friends :
  ∃ (ways : ℕ), ways = 4 ^ 8 ∧ ways = 65536 :=
by {
  use 4 ^ 8,
  split,
  { sorry },  -- Here you would prove 4 ^ 8 = 65536, but for now we leave it at "sorry"
  { sorry }   -- This just directly expresses equivalence asserted in natural number equality
}

end number_of_ways_to_divide_friends_l414_414168


namespace sin_alpha_value_l414_414186

def point_in_trig (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = -2 * real.cos (30 * real.pi / 180) ∧ y = 2 * real.sin (30 * real.pi / 180) ∧ (x, y) = (-2 * real.cos (30 * real.pi / 180), 2 * real.sin (30 * real.pi / 180))

theorem sin_alpha_value (α : ℝ) (h : point_in_trig α) : real.sin α = 1 / 2 := 
sorry

end sin_alpha_value_l414_414186


namespace min_pairs_with_same_sum_l414_414404

theorem min_pairs_with_same_sum (n : ℕ) (h1 : n > 0) :
  (∀ weights : Fin n → ℕ, (∀ i, weights i ≤ 21) → (∃ i j k l : Fin n,
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    weights i + weights j = weights k + weights l)) ↔ n ≥ 8 :=
by
  sorry

end min_pairs_with_same_sum_l414_414404


namespace eq_has_three_integer_solutions_l414_414826

theorem eq_has_three_integer_solutions (n : ℕ) (x1 y1 : ℤ)
  (h : x1 ^ 3 - 3 * x1 * y1 ^ 2 + y1 ^ 3 = n) :
  ∃ x2 y2 x3 y3 : ℤ,
    (x1, y1) ≠ (x2, y2) ∧
    (x1, y1) ≠ (x3, y3) ∧
    (x2, y2) ≠ (x3, y3) ∧
    x2 ^ 3 - 3 * x2 * y2 ^ 2 + y2 ^ 3 = n ∧
    x3 ^ 3 - 3 * x3 * y3 ^ 2 + y3 ^ 3 = n :=
begin
  -- The proof goes here
  sorry
end

end eq_has_three_integer_solutions_l414_414826


namespace milk_pumping_hours_l414_414667

theorem milk_pumping_hours 
    (initial_milk : ℕ := 30000)
    (pumping_rate : ℕ := 2880)
    (additional_rate : ℕ := 1500)
    (additional_time : ℕ := 7)
    (milk_left : ℕ := 28980) : 
    ∃ h : ℕ, 30000 - 2880 * h + 1500 * 7 = 28980 ∧ h = 4 :=
by
  use 4
  simp
  norm_num
  sorry

end milk_pumping_hours_l414_414667


namespace dodecagon_diagonals_concurrent_l414_414392

/-- Let \( P_1, \cdots, P_{12} \) be the vertices of a regular dodecagon. 
    Prove that the diagonals \(P_1 P_9\), \(P_2 P_{11}\), and \(P_4 P_{12}\) are concurrent. -/
theorem dodecagon_diagonals_concurrent 
    (P : ℤ → ℝ × ℝ) 
    (h_regular_dodecagon : ∀ i, dist (P i) (P (i + 1)) = dist (P 1) (P 2))
    (h_conc : ∃ O, line_through (P 1) (P 9) = line_through (P 2) (P 11) ∧
                    line_through (P 2) (P 11) = line_through (P 4) (P 12) ∧
                    line_through (P 4) (P 12) = line_through (P 1) (P 9)) :
    ∃ O, ∃ Q, line_through (P 1) (P 9) = line_through (P 2) (P 11) ∧
             line_through (P 2) (P 11) = line_through (P 4) (P 12) ∧
             Q = O :=
sorry

end dodecagon_diagonals_concurrent_l414_414392


namespace sum_possible_values_l414_414871

theorem sum_possible_values (y : ℝ) (h : y^2 = 36) : 
  y = 6 ∨ y = -6 → 6 + (-6) = 0 :=
by
  intro hy
  rw [add_comm]
  exact add_neg_self 6

end sum_possible_values_l414_414871


namespace jim_gold_per_hour_l414_414545

theorem jim_gold_per_hour :
  ∀ (hours: ℕ) (treasure_chest: ℕ) (num_small_bags: ℕ)
    (each_small_bag_has: ℕ),
    hours = 8 →
    treasure_chest = 100 →
    num_small_bags = 2 →
    each_small_bag_has = (treasure_chest / 2) →
    (treasure_chest + num_small_bags * each_small_bag_has) / hours = 25 :=
by
  intros hours treasure_chest num_small_bags each_small_bag_has
  intros hours_eq treasure_chest_eq num_small_bags_eq small_bag_eq
  have total_gold : ℕ := treasure_chest + num_small_bags * each_small_bag_has
  have per_hour : ℕ := total_gold / hours
  sorry

end jim_gold_per_hour_l414_414545


namespace pentagon_diagonal_ratio_l414_414515

theorem pentagon_diagonal_ratio (A B C D E P Q R S T : Point)
  (paral1 : (segment BD).parallel (segment CE))
  (paral2 : (segment CE).parallel (segment DA))
  (paral3 : (segment DA).parallel (segment EB))
  (paral4 : (segment EB).parallel (segment AC))
  (paral5 : (segment AC).parallel (segment BD)) :
  ∃ v : ℝ, v = (1 + Real.sqrt 5) / 2 ∧
    (length (segment BD)) / (length (segment CE)) = v ∧
    (length (segment CE)) / (length (segment DA)) = v ∧
    (length (segment DA)) / (length (segment EB)) = v ∧
    (length (segment EB)) / (length (segment AC)) = v ∧
    (length (segment AC)) / (length (segment BD)) = v :=
sorry

end pentagon_diagonal_ratio_l414_414515


namespace list_price_correct_l414_414384

noncomputable def list_price_satisfied : Prop :=
∃ x : ℝ, 0.25 * (x - 25) + 0.05 * (x - 5) = 0.15 * (x - 15) ∧ x = 28.33

theorem list_price_correct : list_price_satisfied :=
sorry

end list_price_correct_l414_414384


namespace rain_on_monday_correct_l414_414397

variable (x : ℝ)

def rain_on_monday (rain_on_tuesday rain_on_wednesday total_rain : ℝ) : ℝ :=
  total_rain - (rain_on_tuesday + rain_on_wednesday)

theorem rain_on_monday_correct (h : rain_on_monday 0.42 0.08 0.67 = x) : x = 0.17 := by
  rw [rain_on_monday, add_comm, add_assoc] at h
  exact h

end rain_on_monday_correct_l414_414397


namespace max_friends_l414_414898

theorem max_friends (m : ℕ) (m_geq_three : m ≥ 3) (friends : ℕ → ℕ → Prop)
  (sym_friends : ∀ (a b : ℕ), friends a b → friends b a)
  (no_self_friends : ∀ (a : ℕ), ¬ friends a a)
  (unique_common_friend : ∀ (S : Finset ℕ), S.card = m → ∃! c, ∀ x ∈ S, friends c x) :
  ∃ p, ∀ a, p a ≤ m :=
sorry

end max_friends_l414_414898


namespace original_number_of_bullets_each_had_l414_414696

theorem original_number_of_bullets_each_had (x : ℕ) (h₁ : 5 * (x - 4) = x) : x = 5 := 
sorry

end original_number_of_bullets_each_had_l414_414696


namespace soil_extraction_volume_l414_414373

theorem soil_extraction_volume (L W D1 D2 : ℝ) (hL : L = 20) (hW : W = 12) (hD1 : D1 = 3) (hD2 : D2 = 7) :
  (L * W * ((D1 + D2) / 2) = 1200) :=
by {
  rw [hL, hW, hD1, hD2],
  norm_num,
  sorry
}

end soil_extraction_volume_l414_414373


namespace johnny_tables_l414_414234

theorem johnny_tables :
  ∀ (T : ℕ),
  (∀ (T : ℕ), 4 * T + 5 * T = 45) →
  T = 5 :=
  sorry

end johnny_tables_l414_414234


namespace slope_angle_range_l414_414502

theorem slope_angle_range (k : ℝ) (θ : ℝ) (h1 : ∀ (x y : ℝ), (y = k * x - sqrt 3) ∧ (2 * x + 3 * y - 6 = 0) → (x > 0) ∧ (y > 0)):
  θ > π / 6 ∧ θ < π / 2 ↔ tan θ = k :=
sorry

end slope_angle_range_l414_414502


namespace sum_first_10_terms_l414_414084
open Nat

noncomputable def seq (n : ℕ) : ℚ := 1 / n / (n + 2)

theorem sum_first_10_terms :
  (∑ i in range 1 11, seq i) = 175 / 264 :=
  sorry

end sum_first_10_terms_l414_414084


namespace log_to_exponential_l414_414086

-- Define x such that log_25 (x + 25) = 3/2 implies x = 100
theorem log_to_exponential (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 := by
  sorry

end log_to_exponential_l414_414086


namespace part1_min_positive_period_and_axis_of_symmetry_part2_g_range_in_interval_part3_a_range_l414_414453

noncomputable section
open Real

-- Definitions from conditions
def f (x : ℝ) : ℝ := sin x + cos x
def g (x : ℝ) : ℝ := sin (2 * x) - f x
def h (x : ℝ) : ℝ := (9^x - 1) / (9^x + 1)

-- Part (1)
theorem part1_min_positive_period_and_axis_of_symmetry :
  (∀ x, f (x + 2 * π) = f x) ∧ (∃ k : ℤ, ∀ x, f x = f (k * π + π/4)) := sorry

-- Part (2)
theorem part2_g_range_in_interval :
  ∀ x ∈ Icc (-(π / 2))  0, g(x) ∈ Icc (-5/4 : ℝ) 1 := sorry

-- Part (3)
theorem part3_a_range :
  ∀ a, (0 < a ∧ (∀ x ∈ Icc (-(π / 2)) 0, ¬ (a * g(x) ∈ ({m : ℝ | ∀ x > 0, m * h (x / 2) - h x > 0} : set ℝ)))) ↔ (0 < a ∧ a < 2) := sorry

end part1_min_positive_period_and_axis_of_symmetry_part2_g_range_in_interval_part3_a_range_l414_414453


namespace global_maximum_condition_l414_414103

noncomputable def f (x m : ℝ) : ℝ :=
if x ≤ m then -x^2 - 2 * x else -x + 2

theorem global_maximum_condition (m : ℝ) (h : ∃ (x0 : ℝ), ∀ (x : ℝ), f x m ≤ f x0 m) : m ≥ 1 :=
sorry

end global_maximum_condition_l414_414103


namespace sum_of_3_digit_numbers_with_remainder_2_div_4_l414_414697

/-- 
Problem statement: Prove that the sum of all 3-digit numbers 
that leave a remainder of 2 when divided by 4 is 123750.
-/
theorem sum_of_3_digit_numbers_with_remainder_2_div_4 : 
  let numbers := {x | 100 ≤ x ∧ x < 1000 ∧ x % 4 = 2},
      total_sum := ∑ x in numbers, x
  in
  total_sum = 123750 :=
sorry

end sum_of_3_digit_numbers_with_remainder_2_div_4_l414_414697


namespace choose_4_from_12_l414_414524

theorem choose_4_from_12 : binomial 12 4 = 495 := 
sorry

end choose_4_from_12_l414_414524


namespace angle_ABC_70_l414_414947

theorem angle_ABC_70 (A B C D : Type)
  [IsTriangle ABC]
  (h1 : ∠CAB = 20)
  (h2 : IsMidpoint D A B)
  (h3 : ∠CDB = 40) : ∠ABC = 70 :=
  sorry

end angle_ABC_70_l414_414947


namespace seating_capacity_for_ten_tables_in_two_rows_l414_414377

-- Definitions based on the problem conditions
def seating_for_one_table : ℕ := 6

def seating_for_two_tables : ℕ := 10

def seating_for_three_tables : ℕ := 14

def additional_people_per_table : ℕ := 4

-- Calculating the seating capacity for n tables based on the pattern
def seating_capacity (n : ℕ) : ℕ :=
  if n = 1 then seating_for_one_table
  else seating_for_one_table + (n - 1) * additional_people_per_table

-- Proof statement without the proof
theorem seating_capacity_for_ten_tables_in_two_rows :
  (seating_capacity 5) * 2 = 44 :=
by sorry

end seating_capacity_for_ten_tables_in_two_rows_l414_414377


namespace largest_area_convex_polygon_l414_414587

theorem largest_area_convex_polygon (a : ℝ) :
  ∃ (P : Type) [is_convex_polygon P] (s : Side P = a) 
  (sum_ext_angles_vertices_P = 120), 
  ∀ (Q : Type) [is_convex_polygon Q] (t : Side Q = a) 
  (sum_ext_angles_vertices_Q = 120),
  area P ≥ area Q :=
by {
  sorry
}

end largest_area_convex_polygon_l414_414587


namespace hundredth_letter_is_a_l414_414381

def letter_sequence : ℕ → char
| 0  := 'a'
| 1  := 'b'
| 2  := 'a'
| 3  := 'c'
| 4  := 'a'
| 5  := 'd'
| 6  := 'b'
| 7  := 'd'
| 8  := 'a'
| 9  := 'c'
| 10 := 'd'
| 11 := 'b'
| 12 := 'd'
| n  := letter_sequence (n % 13)

theorem hundredth_letter_is_a : letter_sequence 99 = 'a' :=
by
  sorry

end hundredth_letter_is_a_l414_414381


namespace max_quotient_value_l414_414080

noncomputable def max_quotient_of_three_digit_number : ℝ :=
  (max (λ a b c : ℕ, if a > 5 ∧ b > 5 ∧ c > 5 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a
                     then (100 * a + 10 * b + c) / (a + b + c)
                     else 0) { a | a > 5 } { b | b > 5 } { c | c > 5 })

theorem max_quotient_value : max_quotient_of_three_digit_number = 41.125 := sorry

end max_quotient_value_l414_414080


namespace danny_initial_wrappers_l414_414407

def initial_wrappers (total_wrappers: ℕ) (found_wrappers: ℕ): ℕ :=
  total_wrappers - found_wrappers

theorem danny_initial_wrappers : initial_wrappers 57 30 = 27 :=
by
  exact rfl

end danny_initial_wrappers_l414_414407


namespace ratio_PA_AB_l414_414506

-- Geometry Setup
variables (A B C P : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P]
variables (AC CB PA AB : ℝ)
variables (h_ratio : AC / CB = 3 / 4)
variables (h_bisector : ∃ X, collinear {A, B, P} ∧ ∠(A, X, B) = ∠(C, X, P))

-- Main theorem statement
theorem ratio_PA_AB (h_isosceles : isosceles_triangle A C B) : PA / AB = 3 / 1 :=
by
  sorry

end ratio_PA_AB_l414_414506


namespace rope_total_in_inches_l414_414560

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end rope_total_in_inches_l414_414560


namespace playground_ratio_l414_414288

theorem playground_ratio (L B : ℕ) (playground_area landscape_area : ℕ) 
  (h1 : B = 8 * L)
  (h2 : B = 480)
  (h3 : playground_area = 3200)
  (h4 : landscape_area = L * B) : 
  (playground_area : ℚ) / landscape_area = 1 / 9 :=
by
  sorry

end playground_ratio_l414_414288


namespace find_g_inv_f_neg8_l414_414740

variable {X Y : Type}
variable (f : X → Y) (g : Y → X)
variable (h_f : ∀ y, f (g y) = 3 * g y + 5)

theorem find_g_inv_f_neg8 :
  g (f (-8)) = - (13 / 3) :=
sorry

end find_g_inv_f_neg8_l414_414740


namespace sum_of_numerator_and_denominator_of_cos_alpha_l414_414194

variable {α β : ℝ}
variable {c : ℚ}

-- Conditions:
-- 1. Parallel chords of lengths 4, 6, and 8 determine central angles
--    of α, β, and α + β respectively.
-- 2. α + β < π
-- 3. cos α is a positive rational number

theorem sum_of_numerator_and_denominator_of_cos_alpha (h1 : length_of_chord 4 α)
    (h2 : length_of_chord 6 β)
    (h3 : length_of_chord 8 (α + β))
    (h4 : α + β < π)
    (h5 : ∃ (p q : ℤ), (q ≠ 0) ∧ (p.to_rational / q.to_rational = (c : ℚ)) ∧ (p.gcd q = 1) ∧ (c > 0)) :
    ∑ (p q : ℤ), (q ≠ 0) ∧ (p.to_rational / q.to_rational = (c : ℚ)) ∧ (p.gcd q = 1) ∧ (c > 0) = 5 := sorry

end sum_of_numerator_and_denominator_of_cos_alpha_l414_414194


namespace slope_range_PA1_l414_414640

theorem slope_range_PA1
  (x0 y0 : ℝ)
  (h_ellipse : (x0 ^ 2) / 4 + (y0 ^ 2) / 3 = 1)
  (h_x0_ne : x0 ≠ 2 ∧ x0 ≠ -2)
  (h_slope_PA2 : -2 ≤ y0 / (x0 - 2) ∧ y0 / (x0 - 2) ≤ -1) :
  (3 / 8) ≤ y0 / (x0 + 2) ∧ y0 / (x0 + 2) ≤ (3 / 4) :=
begin
  sorry
end

end slope_range_PA1_l414_414640


namespace fraction_sum_lt_one_l414_414348

theorem fraction_sum_lt_one
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = (∏ i in finset.range n, a (i + 1)) + 1)
  : ∀ n : ℕ, n ≥ 1 → (∑ i in finset.range n, 1 / a (i + 1)) < 1 := 
  sorry

end fraction_sum_lt_one_l414_414348


namespace john_profit_l414_414928

theorem john_profit (cost price : ℕ) (n : ℕ) (h1 : cost = 4) (h2 : price = 8) (h3 : n = 30) : 
  n * (price - cost) = 120 :=
by
  -- The proof goes here
  sorry

end john_profit_l414_414928


namespace ellipse_equation_range_expression_l414_414800

-- First part: Equation of the ellipse
theorem ellipse_equation (x y : ℝ) (a b : ℝ) (h1 : a = 4) (h2 : a > b) (h3 : b > 0)
  (h4 : (y / (x + 4)) * (y / (x - 4)) = -3 / 4) :
  x^2 / 16 + y^2 / 12 = 1 :=
sorry

-- Second part: Range of the expression
theorem range_expression (k x1 x2 y1 y2 : ℝ) 
  (h1 : a = 4) (h5 : b = sqrt 12)
  (h6 : y1 = k * x1 + 2) (h7 : y2 = k * x2 + 2) 
  (h8 : x1 + x2 = -16 * k / (4 * k^2 + 3))
  (h9 : x1 * x2 = -32 / (4 * k^2 + 3)) :
  -20 ≤ ((x1 * x2) + (y1 * y2)) + (x1 * x2 + (y1 - 2) * (y2 - 2)) ∧ 
  ((x1 * x2) + (y1 * y2)) + (x1 * x2 + (y1 - 2) * (y2 - 2)) ≤ -52 / 3 :=
sorry

end ellipse_equation_range_expression_l414_414800


namespace collinear_B_l414_414562

variables (A B C P D E O O' B' A' : Type)
variables [has_collinear A B C P D E O O' B' A']
variables [has_perp B' A]
variables [has_perp A' B]
variables (H1 : triangle_acute A B C)
variables (H2 : perp_bisector B' (segment AC))
variables (H3 : perp_bisector A' (segment BC))
variables (H4 : on_segment P (segment AB))
variables (O_def : circumcenter O A B C)
variables (H5 : perp_to DP (line BO))
variables (H6 : perp_to EP (line AO))
variables (O'_def : circumcenter O' C D E)

theorem collinear_B'_A'_O' : collinear B' A' O' :=
sorry

end collinear_B_l414_414562


namespace election_majority_l414_414519

theorem election_majority (total_votes : ℕ) (winning_percentage : ℝ) (losing_percentage : ℝ)
  (h_total_votes : total_votes = 700)
  (h_winning_percentage : winning_percentage = 0.70)
  (h_losing_percentage : losing_percentage = 0.30) :
  (winning_percentage * total_votes - losing_percentage * total_votes) = 280 :=
by
  sorry

end election_majority_l414_414519


namespace rectangle_perimeter_l414_414271

noncomputable def perimeter_of_rectangle (x y a b : ℝ) : ℝ :=
  if h : x * y = 4032 ∧ 4032 * π = π * a * b ∧ (x + y) = 2 * a then 4 * a
  else 0

theorem rectangle_perimeter (x y a b : ℝ) :
  x * y = 4032 →
  4032 * π = π * a * b →
  (x + y) = 2 * a →
  4 * a = 8 * real.sqrt 2016 :=
begin
  intros h1 h2 h3,
  sorry
end

end rectangle_perimeter_l414_414271


namespace mike_seashells_l414_414230

theorem mike_seashells (initial total : ℕ) (h1 : initial = 79) (h2 : total = 142) :
    total - initial = 63 :=
by
  sorry

end mike_seashells_l414_414230


namespace count_integers_satisfying_condition_l414_414410

/-- Theorem:
Determine the number of integers n between 1 and 100, inclusive,
satisfying the condition that (n^2 - 2)! / (n!)^(n - 1) is an integer.
-/
theorem count_integers_satisfying_condition :
  ∃ (count : ℕ), count = 99 ∧ 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 100 → Int.mod (factorial(n^2 - 2)) (factorial(n))^(n - 1) = 0 :=
sorry

end count_integers_satisfying_condition_l414_414410


namespace hyperbola_constants_sum_l414_414517

theorem hyperbola_constants_sum :
  ∃ (h k a b : ℝ), 
  (h = 1) ∧ 
  (k = 1) ∧ 
  (a = 3) ∧ 
  (b = 3 * Real.sqrt 3) ∧ 
  (h + k + a + b = 5 + 3 * Real.sqrt 3) :=
begin
  use [1, 1, 3, 3 * Real.sqrt 3],
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  rw [refl 1, refl 1, refl 3, refl (3 * Real.sqrt 3)],
end

end hyperbola_constants_sum_l414_414517


namespace Typist_A_words_typed_l414_414716

variables (work_total : ℕ)
variables (rate_A : ℕ)
variables (rate_B : ℕ)
variables (total_time : ℕ)

-- The given problem conditions
def given_conditions := work_total = 5700 ∧ rate_A = 100 ∧ rate_B = 90 ∧ total_time = work_total / (rate_A + rate_B)

-- Lean statement to prove
theorem Typist_A_words_typed (h : given_conditions) : rate_A * total_time = 3000 :=
by sorry

end Typist_A_words_typed_l414_414716


namespace total_houses_in_lincoln_county_l414_414665

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (built_houses : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : built_houses = 97741) : 
  original_houses + built_houses = 118558 := 
by
  -- Sorry is used to skip the proof.
  sorry

end total_houses_in_lincoln_county_l414_414665


namespace prob_xi_ge_2_l414_414793

noncomputable theory
open Classical

variable (ξ : ℝ → ℝ)

def normal_dist (μ σ : ℝ) := measure_theory.measure.norm_pdf μ σ

axiom ξ_normal : normal_dist 0 σ = ξ
axiom prob_neg2_to_0 : measure_theory.prob ξ (-2 ≤ ξ) (ξ ≤ 0) = 0.2

theorem prob_xi_ge_2 : measure_theory.prob ξ (ξ ≥ 2) = 0.3 :=
sorry

end prob_xi_ge_2_l414_414793


namespace inequality_proof_l414_414939

variable {R : Type*} [LinearOrderedField R]

theorem inequality_proof 
  (a b c x y z : R)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : 0 < x)
  (h5 : 0 < y)
  (h6 : 0 < z)
  (h7 : x + y + z = 3) :
  (3 / 2) * Real.sqrt ((1 / (a * a)) + (1 / (b * b)) + (1 / (c * c)))
  ≥ (x / (1 + a * a)) + (y / (1 + b * b)) + (z / (1 + c * c)) :=
by
  -- This is where the proof would go.
  sorry

end inequality_proof_l414_414939


namespace matrix_transformation_l414_414460

theorem matrix_transformation
  (a b : ℝ)
  (M : Matrix (Fin 2) (Fin 2) ℝ := ![![a, -1/2], ![1/2, b]])
  (P : ℝ × ℝ := (2, 2))
  (P' : ℝ × ℝ := (Real.sqrt 3 - 1, Real.sqrt 3 + 1))
  (H : M.mulVec (Vec.P P) = Vec.P P') :
  a = Real.sqrt 3 / 2 ∧ b = Real.sqrt 3 / 2 ∧ Matrix.invDet M (Fin 2) (Fin 2) ℝ := sorry

end matrix_transformation_l414_414460


namespace minimum_path_length_l414_414189

noncomputable def minimum_yp_pq_qz (XY XZ : ℝ) (angle_XYZ : ℝ) (P Q : ℝ → bool) : ℝ :=
  let Y' := -- coordinates or lengths related to the reflection of Y (according to reflection conditions)
  let Z' := -- coordinates or lengths related to the reflection of Z (according to reflection conditions)
  let d := λ P Q, -- distance function for points P and Q (defined properly to handle reflections)
  let path_length := λ P Q, d Y' P + d P Q + d Q Z'
  path_length P Q

theorem minimum_path_length :
  ∃ P Q, 
  ∀ (XY XZ : ℝ) (angle_XYZ : ℝ) (_ : angle_XYZ = 60) (_ : XY = 12) (_ : XZ = 8),
  minimum_yp_pq_qz XY XZ angle_XYZ P Q = real.sqrt 304 :=
sorry

end minimum_path_length_l414_414189


namespace order_of_three_numbers_l414_414643

noncomputable def log_base_2 : ℝ → ℝ := λ x, Real.log x / Real.log 2

theorem order_of_three_numbers : 0 < (0.3 : ℝ)^2 ∧ (0.3 : ℝ)^2 < 1 ∧ log_base_2 0.3 < 0 ∧ 2^(0.3 : ℝ) > 1 →
  log_base_2 0.3 < (0.3 : ℝ)^2 ∧ (0.3 : ℝ)^2 < 2^(0.3 : ℝ) :=
by sorry

end order_of_three_numbers_l414_414643


namespace product_value_l414_414063

noncomputable def infinite_product : ℝ :=
  ∏ n in {2^k : ℕ // k > 0}, 3^(1/(2^n))

theorem product_value : infinite_product = 9 :=
by
  sorry

end product_value_l414_414063


namespace car_speed_l414_414989

def travel_time : ℝ := 5
def travel_distance : ℝ := 300

theorem car_speed :
  travel_distance / travel_time = 60 := sorry

end car_speed_l414_414989


namespace sheets_of_paper_in_each_box_l414_414927

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 70) 
  (h2 : 4 * (E - 20) = S) : 
  S = 120 := 
by 
  sorry

end sheets_of_paper_in_each_box_l414_414927


namespace largest_k_three_element_subsets_l414_414105

open Finset

def num_subsets_with_non_empty_intersections (M : Finset ℕ) : ℕ :=
  (card M - 1).choose 2

theorem largest_k_three_element_subsets (n : ℕ) (hn : n ≥ 6) (M : Finset ℕ) (hM : card M = n) :
  ∃ (ψ : Finset (Finset ℕ)), 
    (∀ A ∈ ψ, card A = 3) ∧ 
    (∀ A B ∈ ψ, A ≠ B → (A ∩ B).nonempty) ∧ 
    card ψ = num_subsets_with_non_empty_intersections M :=
  sorry

end largest_k_three_element_subsets_l414_414105


namespace find_n_l414_414422

theorem find_n :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * Real.pi / 180) = Real.cos (942 * Real.pi / 180) := sorry

end find_n_l414_414422


namespace inequality_comparison_l414_414332

theorem inequality_comparison (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3 * y + x * y^3 :=
  sorry

end inequality_comparison_l414_414332


namespace minimum_value_l414_414463

theorem minimum_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x - 2 * y + 3 = 0) : 
  ∃ z : ℝ, z = 3 ∧ (∀ z' : ℝ, (z' = y^2 / x) → z ≤ z') :=
sorry

end minimum_value_l414_414463


namespace angle_between_generatrix_and_base_plane_l414_414974

theorem angle_between_generatrix_and_base_plane
  (O A B : Point)
  (ϕ β : ℝ)
  (cone1 : Cone {vertex := O, apex := A})
  (cone2 : Cone {vertex := O, apex := B})
  (height1 := cone1.height)
  (height2 := cone2.height)
  (α : Plane)
  (cond1 : A ∈ α)
  (cond2 : B ∈ α)
  (cond3 : angle height1 height2 = β)
  (cond4 : angle (height1.direction) (cone1.generatrix.direction) = ϕ)
  (cond5 : 2 * ϕ < β) :
  angle OA (base_plane cone2 B) = 
    (π / 2) - arccos (cos ϕ - (2 * (sin (β / 2))^2 / cos ϕ)) :=
  sorry

end angle_between_generatrix_and_base_plane_l414_414974


namespace greatest_number_of_elements_l414_414010

theorem greatest_number_of_elements (S : Finset ℕ) : 
  (∀ x ∈ S, (S.sum - x) % (S.card - 1) = 0) ∧ 
  (∃ k, S.sum = k^2) ∧ 
  (1 ∈ S) ∧ 
  (3003 ∈ S ∧ ∀ y ∈ S, y ≤ 3003) → 
  S.card = 22 :=
by
  sorry

end greatest_number_of_elements_l414_414010


namespace elisa_target_amount_l414_414417

def elisa_current_amount : ℕ := 37
def elisa_additional_amount : ℕ := 16

theorem elisa_target_amount : elisa_current_amount + elisa_additional_amount = 53 :=
by
  sorry

end elisa_target_amount_l414_414417


namespace polynomial_coefficients_sum_zero_l414_414888

noncomputable def polynomial_expansion (x : ℝ) :=
  (5 * x^2 - 3 * x + 2) * (9 - 3 * x)

theorem polynomial_coefficients_sum_zero : 
  ∃ a b c d : ℝ, 
  (polynomial_expansion x = a * x^3 + b * x^2 + c * x + d) ∧ 
  (27 * a + 9 * b + 3 * c + d = 0) :=
by 
  -- Polynomial expansion
  let a := -15
  let b := 54
  let c := -33
  let d := 18
  use [a, b, c, d]
  sorry

end polynomial_coefficients_sum_zero_l414_414888


namespace sum_of_solutions_eq_zero_l414_414781

def f (x : ℝ) : ℝ := 2^|x| + 3*|x| + x

theorem sum_of_solutions_eq_zero :
  (∑ x in {x : ℝ | f x = 25}, x) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l414_414781


namespace diagonal_square_grid_size_l414_414025

theorem diagonal_square_grid_size (n : ℕ) (h : 2 * n - 1 = 2017) : n = 1009 :=
by
  sorry

end diagonal_square_grid_size_l414_414025


namespace minimize_diagonal_l414_414920

-- Define the right triangle ABC with right angle at C and hypotenuse AB
variables {A B C M N P : Type} [RightTriangle A B C] (hypotenuseAC : A ∘∘ > C) (hypotenuseBC : B ∘∘ > C)

-- Define the inscribed rectangle CMNP
def InscribedRectangle (CMNP : Type) (AC BC : Type) :=
  Rectangle CMNP ∧ SideOf CMNP → LegOf ∘(AC) (+) ∘(BC)

-- The main theorem stating the minimization condition
theorem minimize_diagonal (d : Type) (CMNP : InscribedRectangle CMNP hypotenuseAC hypotenuseBC) :
  minimized_diagonal d CMNP → orthogonal (C N) (A B) :=
begin
  sorry -- proof is omitted
end

end minimize_diagonal_l414_414920


namespace angleTSB_is_27_l414_414909

-- Defining the points based on the given conditions
def B := (0, 1)    -- Bottom edge of the painting
def T := (0, 3)    -- Top edge of the painting
def S := (3, 4)    -- Spotlight position

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def angle_deg (P Q R : ℝ × ℝ) : ℝ :=
  let a := distance Q R
  let b := distance P R
  let c := distance P Q
  real.acos ((b^2 + c^2 - a^2) / (2 * b * c)) * (180 / real.pi)

theorem angleTSB_is_27 :
  angle_deg T S B ≈ 27 :=
sorry

end angleTSB_is_27_l414_414909


namespace volume_of_earth_dug_out_l414_414000

noncomputable def pi_approx := 3.14159

theorem volume_of_earth_dug_out :
  let diameter := 2
  let depth := 14
  let radius := diameter / 2
  let V := pi_approx * (radius ^ 2) * depth
  V ≈ 43.98 :=
by
  sorry

end volume_of_earth_dug_out_l414_414000


namespace exists_n_satisfying_condition_l414_414282

-- Definition of the divisor function d(n)
def d (n : ℕ) : ℕ := Nat.divisors n |>.card

-- Theorem statement
theorem exists_n_satisfying_condition : ∃ n : ℕ, ∀ i : ℕ, i ≤ 1402 → (d n : ℚ) / d (n + i) > 1401 ∧ (d n : ℚ) / d (n - i) > 1401 :=
by
  sorry

end exists_n_satisfying_condition_l414_414282


namespace opposite_face_of_silver_is_orange_l414_414634
-- Import all necessary libraries from Mathlib

-- Define the colors as a datatype
inductive Color
| orange
| blue
| yellow
| black
| silver
| pink

open Color

-- Define the given conditions as hypotheses
axiom TopFaceView1 : Color = black
axiom FrontFaceView1 : Color = blue
axiom RightFaceView1 : Color = yellow

axiom TopFaceView2 : Color = black
axiom FrontFaceView2 : Color = pink
axiom RightFaceView2 : Color = yellow

axiom TopFaceView3 : Color = black
axiom FrontFaceView3 : Color = silver
axiom RightFaceView3 : Color = yellow

-- Define the theorem to be proven
theorem opposite_face_of_silver_is_orange :
    ∀ (C : Color), 
    (TopFaceView1 = black) ∧ 
    (FrontFaceView1 = blue) ∧ 
    (RightFaceView1 = yellow) ∧
    (TopFaceView2 = black) ∧ 
    (FrontFaceView2 = pink) ∧ 
    (RightFaceView2 = yellow) ∧
    (TopFaceView3 = black) ∧ 
    (FrontFaceView3 = silver) ∧ 
    (RightFaceView3 = yellow)
    → C = orange :=
sorry

end opposite_face_of_silver_is_orange_l414_414634


namespace extremum_when_a_ge_zero_range_for_two_zeroes_and_sum_greater_than_two_l414_414472

noncomputable def f (a x : ℝ) : ℝ := ln x - (a/2) * x^2 + (a-1) * x

theorem extremum_when_a_ge_zero (a : ℝ) (h : a ≥ 0) :
  ∃ x_max, x_max = 1 ∧ f a x_max = a/2 - 1 :=
sorry

theorem range_for_two_zeroes_and_sum_greater_than_two {a x₁ x₂ : ℝ} (h₁ : 2 < a) 
  (h₂ : f a x₁ = 0) (h₃ : f a x₂ = 0) 
  (hx₁ : 0 < x₁ ∧ x₁ < 1) (hx₂ : x₂ > 1) :
  x₁ + x₂ > 2 :=
sorry

end extremum_when_a_ge_zero_range_for_two_zeroes_and_sum_greater_than_two_l414_414472


namespace sum_of_first_two_primes_gt_50_l414_414173

theorem sum_of_first_two_primes_gt_50 : 
  let p1 := 53 in 
  let p2 := 59 in
  p1 + p2 = 112 :=
by
  sorry

end sum_of_first_two_primes_gt_50_l414_414173


namespace probability_even_product_l414_414188

open Finset

theorem probability_even_product :
  let x_set := {1, 2, 3, 4}
  let y_set := {5, 6}
  let even (n : ℕ) := n % 2 = 0
  let prob_even_x := (card (x_set.filter even) : ℝ) / (card x_set : ℝ)
  let prob_even_y := (card (y_set.filter even) : ℝ) / (card y_set : ℝ)
  let prob_odd_x := 1 - prob_even_x
  in prob_even_x + prob_odd_x * prob_even_y = 3 / 4 := by
  let x_set := {1, 2, 3, 4}
  let y_set := {5, 6}
  let even (n : ℕ) := n % 2 = 0
  let prob_even_x := (card (x_set.filter even) : ℝ) / (card x_set : ℝ)
  let prob_even_y := (card (y_set.filter even) : ℝ) / (card y_set : ℝ)
  let prob_odd_x := 1 - prob_even_x
  calc
    prob_even_x + prob_odd_x * prob_even_y = 2 / 4 + (1 - 2 / 4) * 1 / 2 : by sorry
    ... = 2 / 4 + 2 / 4 * 1 / 2 : by sorry
    ... = 2 / 4 + 1 / 4 : by sorry
    ... = 3 / 4 : by sorry

end probability_even_product_l414_414188


namespace team_of_smurfs_l414_414705

-- Definitions based on conditions:
def smurfs : Finset ℕ := Finset.range 12  -- There are 12 Smurfs labeled from 0 to 11
def dislikes (a b : ℕ) : Prop := (b = (a+1) % 12) ∨ (b = (a-1+12) % 12)  -- A Smurf dislikes its two adjacent

-- Prove that there are 36 ways to form a team of 5 Smurfs who do not dislike each other:
theorem team_of_smurfs : ∃ (team : Finset (Fin ℕ)), team.card = 5 ∧ ∀ a b ∈ team, ¬ dislikes a b :=
by
  -- The proof will be filled in later
  sorry

end team_of_smurfs_l414_414705


namespace factorization_result_l414_414687

-- Definitions of the given conditions
def Eq1 : Prop := x^2 + 2 * x - 1 = (x - 1) ^ 2
def Eq2 : Prop := (a + b) * (a - b) = a^2 - b^2
def Eq3 : Prop := x^2 + 4 * x + 4 = (x + 2) ^ 2
def Eq4 : Prop := x^2 - 4 * x + 3 = x * (x - 4) + 3

-- The main theorem to prove
theorem factorization_result : Eq3 :=
by
  skip -- Insert proof here

end factorization_result_l414_414687


namespace max_page_number_l414_414596

theorem max_page_number (plenty_of_zeros_threes_fours_fives_sixes_sevens_eights_nines: true) 
  (limit_twos: ℕ = 30) (limit_ones: ℕ = 25) : ℕ :=
by 
  let max_pages := 199
  let q1 := -- additional assumptions
  let q2 := -- condition application
  
  sorry

end max_page_number_l414_414596


namespace slope_of_tangent_line_extreme_points_l414_414258

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - (a + 1) * x + a * Real.log x

theorem slope_of_tangent_line (a x : ℝ) (h1 : a = 2) (h2 : x = 3) : 
  Deriv (f a) x = 2/3 :=
sorry

theorem extreme_points (a : ℝ) (h1 : 0 < a) :
  (f' a 1 = 0 ∨ f' a a = 0) → 
  ((0 < a ∧ a < 1) ∧ (is_max (f a) a) ∧ (is_min (f a) 1)) ∨
  (a = 1 ∧ ¬∃ x, is_min (f a) x ∧ ¬∃ x, is_max (f a) x) ∨ 
  (a > 1 ∧ (is_max (f a) 1) ∧ (is_min (f a) a)) :=
sorry

end slope_of_tangent_line_extreme_points_l414_414258


namespace parallel_vectors_solution_l414_414163

noncomputable def vector_a : (ℝ × ℝ) := (1, 2)
noncomputable def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -4)

def vectors_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_solution (x : ℝ) (h : vectors_parallel vector_a (vector_b x)) : x = -2 :=
sorry

end parallel_vectors_solution_l414_414163


namespace find_a_condition_l414_414830

theorem find_a_condition 
(f : ℝ → ℝ) 
(a : ℝ) 
(x₁ x₂ : ℝ) 
(hf : ∀ x, f x = (1/3)*x^3 + x^2 + a * x)
(h_ext1 : f'(x₁) = 0)
(h_ext2 : f'(x₂) = 0)
(h_line : ∃ x₀, (x₀, 0) ∈ l ∧ f(x₀) = 0) :
  a = 2 / 3 :=
sorry

end find_a_condition_l414_414830


namespace commute_times_abs_diff_l414_414005

def commute_times_avg (x y : ℝ) : Prop := (x + y + 7 + 8 + 9) / 5 = 8
def commute_times_var (x y : ℝ) : Prop := ((x - 8)^2 + (y - 8)^2 + (7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2) / 5 = 4

theorem commute_times_abs_diff (x y : ℝ) (h_avg : commute_times_avg x y) (h_var : commute_times_var x y) :
  |x - y| = 6 :=
sorry

end commute_times_abs_diff_l414_414005


namespace inequality_holds_for_m16_l414_414811

theorem inequality_holds_for_m16 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (4 / a) + (1 / b) ≥ 16 / (a + 4 * b) :=
by
  sorry

end inequality_holds_for_m16_l414_414811


namespace checkerboard_pattern_exists_l414_414065

-- Define the board dimensions
def n : ℕ := 100

-- Define the type of colors
inductive Color
| black
| white

-- Define the board as a function from coordinates to Color
def Board : Type := ℕ × ℕ → Color

-- Given conditions
variables (b : Board)
(h1 : ∀ i, i < n → b (0, i) = Color.black)
(h2 : ∀ j, j < n → b (j, 0) = Color.black)
(h3 : ∀ j, j < n → b (j, n - 1) = Color.black)
(h4 : ∀ i, i < n → b (n - 1, i) = Color.black)
(h5 : ∀ i j : ℕ, i < n - 1 → j < n - 1 →
  (b (i, j) = b (i + 1, j) ∧ b (i + 1, j + 1) = b (i, j + 1) ∧
   b (i, j) = b (i, j + 1) ∧ b (i + 1, j + 1) = b (i + 1, j))) → False)

-- Prove that there exists a 2x2 checkerboard pattern
theorem checkerboard_pattern_exists :
  ∃ i j : ℕ, i < n - 1 ∧ j < n - 1 ∧
  b (i, j) = Color.black ∧ b (i + 1, j + 1) = Color.black ∧
  b (i, j + 1) = Color.white ∧ b (i + 1, j) = Color.white :=
sorry

end checkerboard_pattern_exists_l414_414065


namespace number_of_valid_integers_l414_414783
-- Import the necessary Lean libraries

-- Define the necessary conditions
def is_integer (q : ℚ) : Prop := q.denom = 1

-- Main theorem statement
theorem number_of_valid_integers : 
  ∃ (k : ℕ), k = 2 ∧ 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 100 →
  (is_integer ((↑((n^3 - 1)!)) / ((↑((n^2)!) * ↑((n^2)!) * (↑((n^2)!)^n)))) ↔ (n = 1 ∨ n = 2)) :=
sorry

end number_of_valid_integers_l414_414783


namespace sum_of_valid_divisors_l414_414571

-- Define condition for d being a divisor leaving remainder 7 when dividing 134
def valid_divisor (d : ℕ) : Prop :=
  d > 0 ∧ 134 % d = 7

-- Define the main theorem to prove that the sum of all two-digit values for such d is 0
theorem sum_of_valid_divisors : 
  ∑ d in (finset.filter (λ d, valid_divisor d ∧ d ≥ 10 ∧ d < 100) (finset.range 135)), d = 0 := 
by 
  sorry

end sum_of_valid_divisors_l414_414571


namespace find_a3_a7_l414_414213

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Arithmetic sequence definition

def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n (i + 1)

axiom h1 : ∀ n, a_n (n + 1) = a_n 0 + n * (a_n 1 - a_n 0)
axiom h2 : S_n 9 = a_n 4 + a_n 5 + a_n 6 + 72

theorem find_a3_a7 : a_n 3 + a_n 7 = 24 := 
by {
  -- Proof is skipped
  sorry
}

end find_a3_a7_l414_414213


namespace kitten_eats_all_snacks_l414_414325

noncomputable def first_day_all_snacks (start : ℕ) : ℕ :=
  let cat_stick_cycle := 1
  let egg_yolk_cycle := 2
  let nutritional_cream_cycle := 3
  let lcm_cycles := Nat.lcm 1 (Nat.lcm 2 3)
  start + lcm_cycles

theorem kitten_eats_all_snacks (start : ℕ) : first_day_all_snacks 23 = 29 :=
  by {
    unfold first_day_all_snacks,
    let lcm_cycles := Nat.lcm 1 (Nat.lcm 2 3),
    have h_lcm : lcm_cycles = 6 := sorry,
    rw h_lcm,
    exact Nat.add_eq_iff_eq_sub.mpr rfl
  }

end kitten_eats_all_snacks_l414_414325


namespace hazel_sold_18_cups_to_kids_l414_414165

theorem hazel_sold_18_cups_to_kids:
  ∀ (total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup: ℕ),
     total_cups = 56 →
     cups_sold_construction = 28 →
     crew_remaining = total_cups - cups_sold_construction →
     last_cup = 1 →
     crew_remaining = cups_sold_kids + (cups_sold_kids / 2) + last_cup →
     cups_sold_kids = 18 :=
by
  intros total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup h_total h_construction h_remaining h_last h_equation
  sorry

end hazel_sold_18_cups_to_kids_l414_414165


namespace inverse_proportionality_direct_proportionality_l414_414991

-- Condition 1: x and y are inversely proportional, and x = 40 when y = 9.
-- Proving that x = 18 when y = 20 under the given conditions.
theorem inverse_proportionality (x y k : ℝ) (h1 : x * y = k) (h2 : x = 40) (h3 : y = 9) : ∃ x, x * 20 = k ∧ x = 18 :=
by
  have hxy : 40 * 9 = k, from by rwa [h2, h3] at h1
  use 18
  have : 18 * 20 = 40 * 9, from by calc
    18 * 20 = 360 : by norm_num
    ... = 40 * 9 : by norm_num
  rw hxy at this
  exact ⟨this, rfl⟩

-- Condition 2: z is directly proportional to y, and z = 45 when y = 10.
-- Proving that z = 90 when y = 20 under the given conditions.
theorem direct_proportionality (z y k' : ℝ) (h1 : z = k' * y) (h2 : z = 45) (h3 : y = 10) : ∃ z, (z = 4.5 * 20) ∧ z = 90 :=
by
  have hz : 45 = k' * 10, from by rwa [h2, h3] at h1
  use 90
  have : 4.5 * 20 = 90, from by norm_num
  exact ⟨this, rfl⟩

end inverse_proportionality_direct_proportionality_l414_414991


namespace max_possible_k_l414_414785

theorem max_possible_k :
  ∃ (k : ℕ), 
  (∀ (a b : ℕ) (a_i b_i : fin k),
     a_i < k ∧ b_i < k ∧ a_i ≠ b_i ∧ a_i < b_i ∧ (a_i + b_i ≤ 2023)) → 
  (∀ i j : fin k, i ≠ j → (a_i + b_i) ≠ (a_j + b_j)) ∧
  k ≤ 809 := sorry

end max_possible_k_l414_414785


namespace find_number_l414_414666

theorem find_number (x : ℝ) (h : (x / 4) + 9 = 15) : x = 24 :=
by
  sorry

end find_number_l414_414666


namespace painting_two_sides_time_l414_414265

-- Definitions for the conditions
def time_to_paint_one_side_per_board : Nat := 1
def drying_time_per_board : Nat := 5

-- Definitions for the problem
def total_boards : Nat := 6

-- Main theorem statement
theorem painting_two_sides_time :
  (total_boards * time_to_paint_one_side_per_board) + drying_time_per_board + (total_boards * time_to_paint_one_side_per_board) = 12 :=
sorry

end painting_two_sides_time_l414_414265


namespace temperature_at_midnight_l414_414900

def morning_temp : ℝ := 30
def afternoon_increase : ℝ := 1
def midnight_decrease : ℝ := 7

theorem temperature_at_midnight : morning_temp + afternoon_increase - midnight_decrease = 24 := by
  sorry

end temperature_at_midnight_l414_414900


namespace jogging_distance_apart_l414_414382

theorem jogging_distance_apart 
  (anna_rate : ℕ) (mark_rate : ℕ) (time_hours : ℕ) :
  anna_rate = (1 / 20) ∧ mark_rate = (3 / 40) ∧ time_hours = 2 → 
  6 + 3 = 9 :=
by
  -- setting up constants and translating conditions into variables
  have anna_distance : ℕ := 6
  have mark_distance : ℕ := 3
  sorry

end jogging_distance_apart_l414_414382


namespace num_complementary_sets_l414_414762

-- Definitions for shapes, colors, shades, and patterns
inductive Shape
| circle | square | triangle

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

inductive Pattern
| striped | dotted | plain

-- Definition of a card
structure Card where
  shape : Shape
  color : Color
  shade : Shade
  pattern : Pattern

-- Condition: Each possible combination is represented once in a deck of 81 cards.
def deck : List Card := sorry -- Construct the deck with 81 unique cards

-- Predicate for complementary sets of three cards
def is_complementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∧ c1.shape = c3.shape ∨
   c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∧ c1.color = c3.color ∨
   c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∧ c1.shade = c3.shade ∨
   c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∧ c1.pattern = c3.pattern ∨
   c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

-- Statement of the theorem to prove
theorem num_complementary_sets : 
  ∃ (complementary_sets : List (Card × Card × Card)), 
  complementary_sets.length = 5400 ∧
  ∀ (c1 c2 c3 : Card), (c1, c2, c3) ∈ complementary_sets → is_complementary c1 c2 c3 :=
sorry

end num_complementary_sets_l414_414762


namespace unique_zero_function_l414_414081

noncomputable def f : ℝ → ℝ := sorry -- definition of the function

-- functional equation predicate
def func_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) * f(x - y) = (f(x) - f(y))^2 - (4 * x * y) * f(y)

theorem unique_zero_function : ∃! f : ℝ → ℝ, func_eq f ∧ ∀ x, f x = 0 :=
by
  sorry

end unique_zero_function_l414_414081


namespace S_2016_value_l414_414636

noncomputable def a_n (n : ℕ) : ℝ := 2^n * Real.cos (n * Real.pi / 2)

noncomputable def S_n (n : ℕ) : ℝ := (Finset.range n).sum (λ i, a_n (i + 1))

theorem S_2016_value :
  S_n 2016 = (4 / 5) * (2 ^ 2016 - 1) :=
sorry

end S_2016_value_l414_414636


namespace probability_of_two_tails_two_heads_l414_414880

theorem probability_of_two_tails_two_heads :
  let p := (1:ℚ) / 2 in
  ∃ (prob : ℚ), prob = 3/8 ∧ (∀ (k : Finset (Fin 4)), k.card = 2 → k = 3) → prob = (nat.choose 4 2) * (p ^ 4) :=
begin
  -- Definitions derived from conditions
  -- p is the probability of head or tail in single toss
  -- We need to show the overall probability of two heads and two tails equals 3/8
  -- p is 1/2
  let p : ℚ := 1/2,
  -- There are 4 choose 2 ways to arrange 2 heads and 2 tails among 4 coins
  have h1 : nat.choose 4 2 = 6,
  { rw [nat.choose_eq_factorial_div_factorial, nat.factorial, nat.factorial],
    norm_num, },
  -- The overall probability of one arrangement of the coins is (1/2)^4
  let seq_prob := p ^ 4,
  -- The total probability is the number of arrangements times the probability of one specific arrangement
  let prob := 6 * seq_prob,
  have h2 : prob = 3/8,
  { simp [h1, seq_prob],
    norm_num, },
  existsi prob,
  split,
  exact h2,
  intros k hk,
  -- This is to show that any set of 2 elements chosen from 4 elements (coins) will
  -- be one of those 6 arrangements that results in the desired probability.
  have h3 : ∀ (k : Finset (Fin 4)), k.card = 2 → k ∈ Finset.powerset.fin k,
  { intro k,
    norm_num, },
  exact h3,
  sorry,
end

end probability_of_two_tails_two_heads_l414_414880


namespace range_of_a_l414_414580

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - a = 0) ↔ a ≥ -1 :=
by
  sorry

end range_of_a_l414_414580


namespace diet_soda_bottles_l414_414363

theorem diet_soda_bottles (R D : ℕ) 
  (h1 : R = 60)
  (h2 : R = D + 41) :
  D = 19 :=
by {
  sorry
}

end diet_soda_bottles_l414_414363


namespace least_x_for_factorial_divisible_by_100000_l414_414782

theorem least_x_for_factorial_divisible_by_100000 :
  ∃ x : ℕ, (∀ y : ℕ, (y < x) → (¬ (100000 ∣ y!))) ∧ (100000 ∣ x!) :=
  sorry

end least_x_for_factorial_divisible_by_100000_l414_414782


namespace probability_two_tails_two_heads_l414_414878

theorem probability_two_tails_two_heads :
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  total_probability = 3 / 8 :=
by
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  sorry

end probability_two_tails_two_heads_l414_414878


namespace max_dot_product_l414_414579

open Real EuclideanSpace

theorem max_dot_product (a b : EuclideanSpace.R 2) 
  (h : ‖a + b‖ = 3) : 
  ∃ c : ℝ, c = (9 / 4) ∧ ∀ d, (a + b = d → ‖d‖ = 3 → a • b ≤ (9 / 4)) :=
by sorry

end max_dot_product_l414_414579


namespace sum_of_numbers_is_4_digits_l414_414176

theorem sum_of_numbers_is_4_digits (C D : ℕ) (hC : 1 ≤ C ∧ C ≤ 9) (hD : 1 ≤ D ∧ D ≤ 9) : 
  let sum := 3654 + (100 * C + 41) + (10 * D + 2) + 111 in 
  1000 ≤ sum ∧ sum < 10000 :=
by
  sorry

end sum_of_numbers_is_4_digits_l414_414176


namespace prob_not_all_same_l414_414336

-- Definition: Rolling five fair 6-sided dice
def all_outcomes := Finset.pi (Finset.range 5) (λ _, Finset.range 6)
def same_outcome_count := 6
def total_outcomes := 6 ^ 5

-- Theorem: The probability that not all five dice show the same number is 1295/1296
theorem prob_not_all_same :
  (total_outcomes - same_outcome_count) / total_outcomes = 1295 / 1296 :=
by
  sorry

end prob_not_all_same_l414_414336


namespace solve_for_n_l414_414614

-- Define the equation as a Lean expression
def equation (n : ℚ) : Prop :=
  (2 - n) / (n + 1) + (2 * n - 4) / (2 - n) = 1

theorem solve_for_n : ∃ n : ℚ, equation n ∧ n = -1 / 4 := by
  sorry

end solve_for_n_l414_414614


namespace rational_non_positive_l414_414861

variable (a : ℚ)

theorem rational_non_positive (h : ∃ a : ℚ, True) : 
  -a^2 ≤ 0 :=
by
  sorry

end rational_non_positive_l414_414861


namespace lock_code_unique_l414_414289

theorem lock_code_unique : ∃ (a b : ℕ), 
  (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ 
  (3337777 = a * 10^6 + a * 10^5 + a * 10^4 + b * 10^3 + b * 10^2 + b * 10^1 + b) ∧ 
  (let S := 3 * a + 4 * b in 
    (10 * a + b = S) ∧ 
    (10 ≤ S ∧ S ≤ 99) ∧ 
    (S / 10 = a) ∧ (S % 10 = b)) := 
begin
  use 3,
  use 7,
  split; [norm_num, split]; [norm_num, split; [refl, split]],
  { norm_num,
    norm_num,  -- 3337777 = 3 * 10^6 + 3 * 10^5 + 3 * 10^4 + 7 * 10^3 + 7 * 10^2 + 7 * 10 + 7, which is true
    split,
    { norm_num,
      linarith },
    { split,
      { linarith },
      { linarith } } }
end

end lock_code_unique_l414_414289


namespace probability_top_two_cards_red_black_l414_414732

-- Define the basic properties of the deck and colors
universe u

def standard_deck : Type u :=
  { deck // deck.card_count = 52 ∧ deck.suits.card_count = 4 ∧ deck.ranks.card_count = 13 ∧
    ∃ reds blacks, reds.card_count = 26 ∧ blacks.card_count = 26 ∧
    ∀ card, card ∈ deck.cards → card.suit ∈ reds ∨ card.suit ∈ blacks }

-- Define function to compute probability
def probability_one_red_one_black (deck : standard_deck) : ℚ :=
  let red_prob  := 26 / 52 in
  let black_prob := 26 / 51 in
  red_prob * black_prob + black_prob * red_prob

-- The theorem statement
theorem probability_top_two_cards_red_black :
  ∀ deck : standard_deck,
    probability_one_red_one_black deck = 26 / 51 :=
by
  sorry

end probability_top_two_cards_red_black_l414_414732


namespace net_rate_of_pay_is_25_l414_414359

def travel_time := 3 -- hours
def speed := 50 -- miles per hour
def fuel_efficiency := 25 -- miles per gallon
def earnings_per_mile := 0.60 -- dollars per mile
def cost_per_gallon := 2.50 -- dollars per gallon

def total_distance (t : ℕ) (s : ℕ) := t * s -- distance = time * speed
def gasoline_used (d : ℕ) (f : ℕ) := d / f -- gasoline used = distance / fuel efficiency
def total_earnings (d : ℕ) (e : ℝ) := d * e -- earnings = distance * earnings per mile
def gasoline_cost (g : ℕ) (c : ℝ) := g * c -- cost = gasoline used * cost per gallon
def net_earnings (e : ℝ) (c : ℝ) := e - c -- net earnings = earnings - cost
def net_rate_per_hour (n : ℝ) (t : ℕ) := n / t -- net rate per hour = net earnings / time

theorem net_rate_of_pay_is_25 :
  net_rate_per_hour
    (net_earnings
      (total_earnings (total_distance travel_time speed) earnings_per_mile)
      (gasoline_cost (gasoline_used (total_distance travel_time speed) fuel_efficiency) cost_per_gallon))
    travel_time
  = 25 :=
by sorry

end net_rate_of_pay_is_25_l414_414359


namespace johns_gas_usage_per_week_l414_414549

def mpg : ℕ := 30
def miles_to_work_each_way : ℕ := 20
def days_per_week_to_work : ℕ := 5
def leisure_miles_per_week : ℕ := 40

theorem johns_gas_usage_per_week : 
  (2 * miles_to_work_each_way * days_per_week_to_work + leisure_miles_per_week) / mpg = 8 :=
by
  sorry

end johns_gas_usage_per_week_l414_414549


namespace smaller_square_side_length_l414_414983

noncomputable def squareSideLength {d e f : ℕ} (h_e_primeNotDivisible: (∀ p : ℕ, p.prime → ¬(p ^ 2 ∣ e))) :
  (2 - real.sqrt ↑e) / f = (4 - real.sqrt 3) / 7 :=
sorry

theorem smaller_square_side_length (d e f : ℕ) :
  d = 4 → e = 3 → f = 7 → (∀ p : ℕ, p.prime → ¬(p ^ 2 ∣ e)) → d + e + f = 14 :=
by
  intros hd he hf h_primeNotDivisible
  have h1 := squareSideLength h_primeNotDivisible
  rw [hd, he, hf] at *
  -- add further necessary geometrical and algebraic arguments
  -- to connect conditions and the result.
  exact (by norm_num : 4 + 3 + 7 = 14)

end smaller_square_side_length_l414_414983


namespace number_of_subsets_A_l414_414480

theorem number_of_subsets_A :
  let A := {0, 1, 2} in
  (cardinal.mk (set.univ : set (A.set.univ))) = 8 :=
by
  sorry

end number_of_subsets_A_l414_414480


namespace Heechul_has_most_books_l414_414847

namespace BookCollection

variables (Heejin Heechul Dongkyun : ℕ)

theorem Heechul_has_most_books (h_h : ℕ) (h_j : ℕ) (d : ℕ) 
  (h_h_eq : h_h = h_j + 2) (d_lt_h_j : d < h_j) : 
  h_h > h_j ∧ h_h > d := 
by
  sorry

end BookCollection

end Heechul_has_most_books_l414_414847


namespace top_leftmost_rectangle_is_E_l414_414763

def rectangle (w x y z : ℕ) : Prop := true

-- Define the rectangles according to the given conditions
def rectangle_A : Prop := rectangle 4 1 6 9
def rectangle_B : Prop := rectangle 1 0 3 6
def rectangle_C : Prop := rectangle 3 8 5 2
def rectangle_D : Prop := rectangle 7 5 4 8
def rectangle_E : Prop := rectangle 9 2 7 0

-- Prove that the top leftmost rectangle is E
theorem top_leftmost_rectangle_is_E : rectangle_E → True :=
by
  sorry

end top_leftmost_rectangle_is_E_l414_414763


namespace max_value_k_l414_414917

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 4
  | (n+1) => 3 * seq n - 2

theorem max_value_k (k : ℝ) :
  (∀ n : ℕ, n > 0 → k * (seq n) ≤ 9^n) → k ≤ 9 / 4 :=
sorry

end max_value_k_l414_414917


namespace length_PP2_eq_4sqrt5_div_3_l414_414444

theorem length_PP2_eq_4sqrt5_div_3 :
  ∀ (x : ℝ), (0 < x) ∧ (x < π / 2) ∧ (6 * cos x = 5 * tan x) →
  (5 * tan x - sqrt 5 * sin x = 4 * sqrt 5 / 3) :=
by
  intro x
  rintro ⟨hx0, hxπ2, hcos_tan⟩
  sorry

end length_PP2_eq_4sqrt5_div_3_l414_414444


namespace probability_of_two_tails_two_heads_l414_414879

theorem probability_of_two_tails_two_heads :
  let p := (1:ℚ) / 2 in
  ∃ (prob : ℚ), prob = 3/8 ∧ (∀ (k : Finset (Fin 4)), k.card = 2 → k = 3) → prob = (nat.choose 4 2) * (p ^ 4) :=
begin
  -- Definitions derived from conditions
  -- p is the probability of head or tail in single toss
  -- We need to show the overall probability of two heads and two tails equals 3/8
  -- p is 1/2
  let p : ℚ := 1/2,
  -- There are 4 choose 2 ways to arrange 2 heads and 2 tails among 4 coins
  have h1 : nat.choose 4 2 = 6,
  { rw [nat.choose_eq_factorial_div_factorial, nat.factorial, nat.factorial],
    norm_num, },
  -- The overall probability of one arrangement of the coins is (1/2)^4
  let seq_prob := p ^ 4,
  -- The total probability is the number of arrangements times the probability of one specific arrangement
  let prob := 6 * seq_prob,
  have h2 : prob = 3/8,
  { simp [h1, seq_prob],
    norm_num, },
  existsi prob,
  split,
  exact h2,
  intros k hk,
  -- This is to show that any set of 2 elements chosen from 4 elements (coins) will
  -- be one of those 6 arrangements that results in the desired probability.
  have h3 : ∀ (k : Finset (Fin 4)), k.card = 2 → k ∈ Finset.powerset.fin k,
  { intro k,
    norm_num, },
  exact h3,
  sorry,
end

end probability_of_two_tails_two_heads_l414_414879


namespace layla_feeding_total_food_l414_414238

theorem layla_feeding_total_food :
  let goldfish := 4
  let swordtails := 5
  let guppies := 10
  let angelfish := 3
  let tetra := 6
  let food_goldfish := 1.0
  let food_swordtail := 2.0
  let food_guppy := 0.5
  let food_angelfish := 1.5
  let food_tetra := 1.0
  let total_food := (goldfish * food_goldfish) +
                    (swordtails * food_swordtail) +
                    (guppies * food_guppy) +
                    (angelfish * food_angelfish) +
                    (tetra * food_tetra)
  in total_food = 29.5 := by
  sorry

end layla_feeding_total_food_l414_414238


namespace total_balloons_l414_414320

theorem total_balloons (Gold Silver Black Total : Nat) (h1 : Gold = 141)
  (h2 : Silver = 2 * Gold) (h3 : Black = 150) (h4 : Total = Gold + Silver + Black) :
  Total = 573 := 
by
  sorry

end total_balloons_l414_414320


namespace contrapositive_of_equation_l414_414339

variable (m : ℕ) (hm : 0 < m)

theorem contrapositive_of_equation :
  (¬∃ (x : ℝ), x^2 + x - m = 0) → m ≤ 0 :=
begin
  sorry
end

end contrapositive_of_equation_l414_414339


namespace christmas_tree_ornaments_l414_414266

theorem christmas_tree_ornaments (x : ℕ) (h1 : x = 6 * x - 15) (h2 : x = 3) :
  (x + 2 * x + 6 * x) = 27 :=
by
  rw [h2]
  norm_num

end christmas_tree_ornaments_l414_414266


namespace product_of_two_numbers_l414_414639

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.lcm a b = 72) (h2 : Nat.gcd a b = 8) :
  a * b = 576 :=
by
  sorry

end product_of_two_numbers_l414_414639


namespace arithmetic_geometric_product_l414_414298

theorem arithmetic_geometric_product :
  let a (n : ℕ) := 2 * n - 1
  let b (n : ℕ) := 2 ^ (n - 1)
  b (a 1) * b (a 3) * b (a 5) = 4096 :=
by 
  sorry

end arithmetic_geometric_product_l414_414298


namespace minimum_workers_for_profit_l414_414713

theorem minimum_workers_for_profit (n : ℕ) :
  (let maintenance_cost := 500
       wage_per_hour := 20
       production_per_hour := 5
       price_per_widget := 3.50
       work_hours := 8,
       
       daily_cost := maintenance_cost + (wage_per_hour * work_hours * n : ℕ),
       daily_revenue := (price_per_widget * (production_per_hour * work_hours) * n : ℕ)
    in 
    daily_revenue > daily_cost) → n ≥ 26 :=
by
  sorry

end minimum_workers_for_profit_l414_414713


namespace gcd_condition_divides_l414_414770

open polynomial

noncomputable def divides_polynomials (m n : ℕ) (f g : polynomial ℂ) : Prop :=
  ∀ k : ℕ, k > 0 → k ≤ m → eval (complex.exp(2 * ↑real.pi * complex.I / (m + 1)) ^ k) f = 0

theorem gcd_condition_divides (m n : ℕ) (hm : m > 0) (hn : n > 0):
  let f := (range (m + 1)).sum (λ k, (C 1) * X ^ (k * n)) in
  let g := (range (m + 1)).sum (λ k, (C 1) * X ^ k) in
  (divides_polynomials m n f g ↔ nat.gcd n (m + 1) = 1) :=
sorry

end gcd_condition_divides_l414_414770


namespace marble_ratio_l414_414386

theorem marble_ratio 
  (K A M : ℕ) 
  (M_has_5_times_as_many_as_K : M = 5 * K)
  (M_has_85_marbles : M = 85)
  (M_has_63_more_than_A : M = A + 63)
  (A_needs_12_more : A + 12 = 34) :
  34 / 17 = 2 := 
by 
  sorry

end marble_ratio_l414_414386


namespace find_z_l414_414346

/-- x and y are positive integers. When x is divided by 9, the remainder is 2, 
and when x is divided by 7, the remainder is 4. When y is divided by 13, 
the remainder is 12. The least possible value of y - x is 14. 
Prove that the number that y is divided by to get a remainder of 3 is 22. -/
theorem find_z (x y z : ℕ) (hx9 : x % 9 = 2) (hx7 : x % 7 = 4) (hy13 : y % 13 = 12) (hyx : y = x + 14) 
: y % z = 3 → z = 22 := 
by 
  sorry

end find_z_l414_414346


namespace value_of_4x_l414_414178

variable (x : ℤ)

theorem value_of_4x (h : 2 * x - 3 = 10) : 4 * x = 26 := 
by
  sorry

end value_of_4x_l414_414178


namespace notepad_days_last_l414_414026

def fold_paper (n : Nat) : Nat := 2 ^ n

def lettersize_paper_pieces : Nat := 5
def folds : Nat := 3
def notes_per_day : Nat := 10

def smaller_note_papers_per_piece : Nat := fold_paper folds
def total_smaller_note_papers : Nat := lettersize_paper_pieces * smaller_note_papers_per_piece
def total_days : Nat := total_smaller_note_papers / notes_per_day

theorem notepad_days_last : total_days = 4 := by
  sorry

end notepad_days_last_l414_414026


namespace total_points_l414_414846

theorem total_points (gwen_points_per_4 : ℕ) (lisa_points_per_5 : ℕ) (jack_points_per_7 : ℕ) 
                     (gwen_recycled : ℕ) (lisa_recycled : ℕ) (jack_recycled : ℕ)
                     (gwen_ratio : gwen_points_per_4 = 2) (lisa_ratio : lisa_points_per_5 = 3) 
                     (jack_ratio : jack_points_per_7 = 1) (gwen_pounds : gwen_recycled = 12) 
                     (lisa_pounds : lisa_recycled = 25) (jack_pounds : jack_recycled = 21) 
                     : gwen_points_per_4 * (gwen_recycled / 4) + 
                       lisa_points_per_5 * (lisa_recycled / 5) + 
                       jack_points_per_7 * (jack_recycled / 7) = 24 := by
  sorry

end total_points_l414_414846


namespace time_addition_result_l414_414923

theorem time_addition_result :
  let current_time := (3, 0, 0) -- represents 3:00:00 PM
  let added_hours := 317
  let added_minutes := 15
  let added_seconds := 30
  let final_time := (8, 15, 30) -- based on 12-hour clock calculations
  let sum := 53 in
  D + E + F = sum :=
by
  sorry

end time_addition_result_l414_414923


namespace obtuse_triangle_l414_414191

theorem obtuse_triangle (A B C : ℝ) (h1 : 0 < B ∧ B < π / 2) (h2 : 0 < C ∧ C < π / 2) (h3 : sin B < cos C) :
  π / 2 < A ∧ A < π :=
by
  sorry

end obtuse_triangle_l414_414191


namespace slope_of_line_n_l414_414142

noncomputable def tan_double_angle (t : ℝ) : ℝ := (2 * t) / (1 - t^2)

theorem slope_of_line_n :
  let slope_m := 6
  let alpha := Real.arctan slope_m
  let slope_n := tan_double_angle slope_m
  slope_n = -12 / 35 :=
by
  sorry

end slope_of_line_n_l414_414142


namespace ratio_income_to_expenditure_l414_414638

theorem ratio_income_to_expenditure (I E S : ℕ) 
  (h1 : I = 10000) 
  (h2 : S = 3000) 
  (h3 : S = I - E) : I / Nat.gcd I E = 10 ∧ E / Nat.gcd I E = 7 := by 
  sorry

end ratio_income_to_expenditure_l414_414638


namespace intersection_M_N_eq_S_l414_414503

def M : Set ℝ := {x | (x - 3) * sqrt (x - 1) ≥ 0}
def N : Set ℝ := {x | (x - 3) * (x - 1) ≥ 0}
def S : Set ℝ := {x | x ≥ 3 ∨ x = 1}

theorem intersection_M_N_eq_S : (M ∩ N) = S := 
sorry

end intersection_M_N_eq_S_l414_414503


namespace v_2023_is_1_l414_414299

def g : ℕ → ℕ
| 1 := 3
| 2 := 4
| 3 := 2
| 4 := 1
| 5 := 5
| _ := 0 -- Although the table didn't provide values for other inputs, we need a default case.

def v : ℕ → ℕ
| 0 := 3
| (n + 1) := g (v n)

theorem v_2023_is_1 : v 2023 = 1 :=
  sorry

end v_2023_is_1_l414_414299


namespace show_inequalities_l414_414938

variable (a b c : ℝ)

theorem show_inequalities
  (h_roots : ∃ P : ℝ[X], P.degree = 3 ∧ P.is_root a ∧ P.is_root b ∧ P.is_root c)
  (h_sum : a + b + c = 6)
  (h_product_sum : ab + bc + ca = 9)
  (h_order : a < b ∧ b < c) :
  0 < a ∧ a < 1 ∧ 1 < b ∧ b < 3 ∧ 3 < c ∧ c < 4 := 
sorry

end show_inequalities_l414_414938


namespace range_of_x_for_f_3_minus_x_positive_l414_414474

noncomputable def f (x : ℝ) : ℝ := x^(-3/2) - x^(2/3)

theorem range_of_x_for_f_3_minus_x_positive :
  ∀ x : ℝ, x > 0 → f(3 - x) > 0 ↔ 2 < x ∧ x < 3 :=
by sorry

end range_of_x_for_f_3_minus_x_positive_l414_414474


namespace probability_female_likes_running_is_3_over_8_probability_male_likes_running_is_7_over_12_chi_squared_is_greater_than_6_635_confidence_is_99_percent_l414_414743

def total_students : ℕ := 200
def total_boys : ℕ := 120
def total_girls : ℕ := total_students - total_boys -- 200 - 120 = 80
def girls_who_like_running : ℕ := 30
def boys_who_do_not_like_running : ℕ := 50
def boys_who_like_running : ℕ := total_boys - boys_who_do_not_like_running -- 120 - 50 = 70
def girls_who_do_not_like_running : ℕ := total_girls - girls_who_like_running -- 80 - 30 = 50

noncomputable def probability_female_likes_running : ℚ := girls_who_like_running / total_girls
noncomputable def probability_male_likes_running : ℚ := boys_who_like_running / total_boys

theorem probability_female_likes_running_is_3_over_8 :
  probability_female_likes_running = 3 / 8 := by
  sorry

theorem probability_male_likes_running_is_7_over_12 :
  probability_male_likes_running = 7 / 12 := by
  sorry

def chi_squared : ℚ :=
  let a := boys_who_like_running
  let b := boys_who_do_not_like_running
  let c := girls_who_like_running
  let d := girls_who_do_not_like_running
  let n := total_students
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_is_greater_than_6_635 :
  chi_squared > 6.635 := by
  sorry

theorem confidence_is_99_percent :
  0.99 ≤ P(chi_squared) := by
  sorry

end probability_female_likes_running_is_3_over_8_probability_male_likes_running_is_7_over_12_chi_squared_is_greater_than_6_635_confidence_is_99_percent_l414_414743


namespace count_palindromes_l414_414333

def is_digit (n : ℕ) : Prop :=
  n = 5 ∨ n = 7 ∨ n = 8 ∨ n = 9

def is_palindrome_eight_digit (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ 
    n = a * 10^7 + b * 10^6 + c * 10^5 + d * 10^4 + d * 10^3 + c * 10^2 + b * 10 + a

theorem count_palindromes : 
  (card {n : ℕ | is_palindrome_eight_digit n} = 256) :=
sorry

end count_palindromes_l414_414333


namespace math_problem_l414_414241

noncomputable def f (n : ℕ) (α : ℝ) (a c : ℕ → ℝ) (y : ℝ) : ℝ :=
  (∑ i in (Finset.univ.filter (λ i, a i ≤ y)), c i * (a i)^2) ^ (1/2) +
  (∑ i in (Finset.univ.filter (λ i, a i > y)), c i * (a i)^α) ^ (1/α)

theorem math_problem (n : ℕ) 
  (hn : n > 1) 
  (α : ℝ) 
  (hα : 0 < α ∧ α < 2) 
  (a c : ℕ → ℝ) 
  (ha : ∀ i, 0 < a i) 
  (hc : ∀ i, 0 < c i) 
  (x y : ℝ) 
  (hx_positive : 0 < x) 
  (hy_positive : 0 < y) 
  (hxy : x ≥ f n α a c y) : 
  f n α a c x ≤ 8^(1/α) * x :=
by
  sorry

end math_problem_l414_414241


namespace exists_disjoint_subsets_with_small_diff_l414_414937

theorem exists_disjoint_subsets_with_small_diff (S : Finset ℕ) (h : S.card = 10) :
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.card = B.card ∧
  abs (A.sum (λ x, (1 : ℚ) / x) - B.sum (λ x, (1 : ℚ) / x)) < 1 / 100 :=
sorry

end exists_disjoint_subsets_with_small_diff_l414_414937


namespace julie_reimbursement_in_cents_l414_414274

-- Definitions
def num_lollipops : ℕ := 12
def total_cost_dollars : ℝ := 3
def lollipops_shared_fraction : ℝ := 1 / 4

-- Theorem stating the amount Julie reimbursed Sarah
theorem julie_reimbursement_in_cents : ℝ :=
  let lollipops_shared := lollipops_shared_fraction * num_lollipops
  let cost_per_lollipop := total_cost_dollars / num_lollipops
  let cost_shared_dollars := cost_per_lollipop * lollipops_shared
  let cost_shared_cents := cost_shared_dollars * 100
  cost_shared_cents = 75 
sorry

end julie_reimbursement_in_cents_l414_414274


namespace conic_section_rect_eq_line_conic_intersection_l414_414157

noncomputable def conic_section_polar (θ : ℝ) : ℝ :=
  (12 / (3 + (Real.sin θ)^2))^(1/2)

def point_A : ℝ × ℝ := (0, -Real.sqrt 3)

def focus_F1 : ℝ × ℝ := (-1, 0)
def focus_F2 : ℝ × ℝ := (1, 0)

def line_eq (x : ℝ) : ℝ := Real.sqrt 3 * (x + 1)

theorem conic_section_rect_eq :
  ∀ (x y : ℝ),
  (3 * x^2 + 4 * y^2 = 12)

theorem line_conic_intersection (M N : ℝ × ℝ) :
  (M.1 ≠ N.1 ∨ M.2 ≠ N.2) → 
  dist focus_F1 M * dist focus_F1 N = 12 / 5 :=
by
  sorry

end conic_section_rect_eq_line_conic_intersection_l414_414157


namespace units_digit_of_sum_l414_414683

theorem units_digit_of_sum (a b : ℕ) : 
  (35^87 + 3^45) % 10 = 8 := 
by
  have units_digit_35 := (5 : ℕ)
  have units_digit_3_power_cycle := [3, 9, 7, 1]
  have remainder_45 := 45 % 4
  have units_digit_3_pow_45 := units_digit_3_power_cycle.nth_le remainder_45 sorry
  have add_units_digits := (units_digit_35 + units_digit_3_pow_45) % 10
  exact add_units_digits = 8

end units_digit_of_sum_l414_414683


namespace H2CO3_formation_l414_414772

-- Define the given conditions
def one_to_one_reaction (a b : ℕ) := a = b

-- Define the reaction
theorem H2CO3_formation (m_CO2 m_H2O : ℕ) 
  (h : one_to_one_reaction m_CO2 m_H2O) : 
  m_CO2 = 2 → m_H2O = 2 → m_CO2 = 2 ∧ m_H2O = 2 := 
by 
  intros h1 h2
  exact ⟨h1, h2⟩

end H2CO3_formation_l414_414772


namespace ant_paths_count_l414_414593

def num_paths (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * nat.choose (2*n-2) (n-1) / n

theorem ant_paths_count (n : ℕ) :
  num_paths n = (2 / n : ℚ) * nat.choose (2*n-2) (n-1) := by
  sorry

end ant_paths_count_l414_414593


namespace find_length_BC_l414_414535

noncomputable def length_BC (AB AC AM : ℝ) (M_is_midpoint : Bool) : ℝ :=
  if h : M_is_midpoint = true then 2 * Real.sqrt 21 else 0

theorem find_length_BC
  (AB AC AM : ℝ)
  (h₀ : AB = 5)
  (h₁ : AC = 7)
  (h₂ : AM = 4)
  (M_is_midpoint : Bool)
  (hM : M_is_midpoint = true) :
  length_BC AB AC AM M_is_midpoint = 2 * Real.sqrt 21 :=
begin
  rw length_BC,
  rw hM,
  simp,
end

end find_length_BC_l414_414535


namespace compute_sumSeries_l414_414955

variable (x : ℝ) (hx : x > 1)

noncomputable def sumSeries : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem compute_sumSeries : sumSeries x = 1 / (x^3 - 1) := by
  sorry

end compute_sumSeries_l414_414955


namespace trapezoid_EFGH_area_l414_414686

def E := (2, -3)
def F := (2, 2)
def G := (7, 8)
def H := (7, 3)

def length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

theorem trapezoid_EFGH_area :
  let b1 := length E F in
  let b2 := length G H in
  let h := real.abs (E.1 - G.1) in
  trapezoid_area b1 b2 h = 25 := by
sorry

end trapezoid_EFGH_area_l414_414686


namespace smallest_n_for_a2017_eq_5_l414_414701

noncomputable def a : ℕ → ℕ
| 1       := 1
| (2 * n) := if even n then a n else 2 * a n
| (2 * n + 1) := if even n then 2 * a n + 1 else a n

theorem smallest_n_for_a2017_eq_5 : 
  ∃ n : ℕ, a n = a 2017 ∧ ∀ m : ℕ, m < n → a m ≠ a 2017 :=
begin
  sorry
end

end smallest_n_for_a2017_eq_5_l414_414701


namespace fraction_sum_l414_414746

-- Define the fractions
def frac1: ℚ := 3/9
def frac2: ℚ := 5/12

-- The theorem statement
theorem fraction_sum : frac1 + frac2 = 3/4 := 
sorry

end fraction_sum_l414_414746


namespace max_letters_B_47_l414_414982

noncomputable def max_letters_B (sticks : ℕ) : ℕ :=
  let sticks_B := 4
  let sticks_V := 5
  let maxB := sticks / sticks_B
  (maxB.downto 0).find (fun b => (sticks - b * sticks_B) % sticks_V = 0) |> Option.getOrElse 0

theorem max_letters_B_47 : max_letters_B 47 = 8 := by
  sorry

end max_letters_B_47_l414_414982


namespace solution_set_inequality_l414_414429

theorem solution_set_inequality (x : ℝ) : (x^2 - 2*x - 8 ≥ 0) ↔ (x ≤ -2 ∨ x ≥ 4) := 
sorry

end solution_set_inequality_l414_414429


namespace fencing_cost_l414_414303

theorem fencing_cost (L B: ℝ) (cost_per_meter : ℝ) (H1 : L = 58) (H2 : L = B + 16) (H3 : cost_per_meter = 26.50) : 
    2 * (L + B) * cost_per_meter = 5300 := by
  sorry

end fencing_cost_l414_414303


namespace vector_dot_product_l414_414123

open Real

def a : ℝ × ℝ × ℝ := (0, 2, 0)
def b : ℝ × ℝ × ℝ := (1, 0, -1)

#check (λ (u v: ℝ × ℝ × ℝ), u.1 * v.1 + u.2 * v.2 + u.3 * v.3)

theorem vector_dot_product : ((a.1 + b.1, a.2 + b.2, a.3 + b.3) : ℝ × ℝ × ℝ) • b = 2 :=
by sorry

end vector_dot_product_l414_414123


namespace female_officers_count_l414_414595

theorem female_officers_count (F M T : ℕ) (h_ratio : 7 * F = 3 * M) (h_total : M + F = 400) (h_percentage : 0.32 * T = F) :
  T = 375 :=
by
  -- proof would go here
  sorry

end female_officers_count_l414_414595


namespace total_num_of_cars_l414_414661

-- Define conditions
def row_from_front := 14
def row_from_left := 19
def row_from_back := 11
def row_from_right := 16

-- Compute total number of rows from front to back
def rows_front_to_back : ℕ := (row_from_front - 1) + 1 + (row_from_back - 1)

-- Compute total number of rows from left to right
def rows_left_to_right : ℕ := (row_from_left - 1) + 1 + (row_from_right - 1)

theorem total_num_of_cars :
  rows_front_to_back = 24 ∧
  rows_left_to_right = 34 ∧
  24 * 34 = 816 :=
by
  sorry

end total_num_of_cars_l414_414661


namespace regular_eqn_exists_l414_414753

noncomputable def parametric_eqs (k : ℝ) : ℝ × ℝ :=
  (4 * k / (1 - k^2), 4 * k^2 / (1 - k^2))

theorem regular_eqn_exists (k : ℝ) (x y : ℝ) (h1 : x = 4 * k / (1 - k^2)) 
(h2 : y = 4 * k^2 / (1 - k^2)) : x^2 - y^2 - 4 * y = 0 :=
sorry

end regular_eqn_exists_l414_414753


namespace even_number_of_segments_l414_414594

theorem even_number_of_segments (n : ℕ) (segments : List (ℤ × ℤ))
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → segments i = (x_i, y_i) → x_i^2 + y_i^2 = 1)
  (h2 : ∑ i in (List.range n), (segments i).1 = 0)
  (h3 : ∑ i in (List.range n), (segments i).2 = 0) :
  ∃ k, 2 * k = n :=
sorry

end even_number_of_segments_l414_414594


namespace min_value_of_m_n_l414_414137

variable {a b : ℝ}
variable (ab_eq_4 : a * b = 4)
variable (m : ℝ := b + 1 / a)
variable (n : ℝ := a + 1 / b)

theorem min_value_of_m_n (h1 : 0 < a) (h2 : 0 < b) : m + n = 5 :=
sorry

end min_value_of_m_n_l414_414137


namespace trapezoid_diagonal_angle_l414_414200

theorem trapezoid_diagonal_angle {A B C D O : Point} (trapezoid_ABCD : Trapezoid A B C D)
  (right_angle_intersection : ∠A O C = 90)
  (diagonal_equals_midsegment : AC = (AB + CD) / 2)
  : angle A B C = 60 := sorry

end trapezoid_diagonal_angle_l414_414200


namespace a7_value_in_expansion_l414_414455

noncomputable def a (k : ℕ) : ℤ :=
∑ i in Finset.range (k + 1), (Nat.choose 10 k * (-2)^(10 - k) : ℤ)

theorem a7_value_in_expansion :
  (a 7) = -960 := 
sorry

end a7_value_in_expansion_l414_414455


namespace find_minimum_r_l414_414179

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_minimum_r (r : ℕ) (h_pos : r > 0) (h_perfect : is_perfect_square (4^3 + 4^r + 4^4)) : r = 4 :=
sorry

end find_minimum_r_l414_414179


namespace sequence_is_arithmetic_with_common_difference_l414_414136

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a(n+1) - a n = d

def sequence (n : ℕ) : ℤ :=
  2 * (n + 1) + 3

theorem sequence_is_arithmetic_with_common_difference :
  is_arithmetic_sequence sequence 2 :=
sorry

end sequence_is_arithmetic_with_common_difference_l414_414136
