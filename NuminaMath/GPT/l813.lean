import Mathlib
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Sequence
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinations
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.MeasureTheory.Measure
import Mathlib.Probability.Independence
import Mathlib.SetTheory.Ordinal.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Sorry
import Real
import data.nat.basic

namespace valid_starting_days_count_l813_813304

def is_valid_starting_day (d : ℕ) : Prop :=
  (d % 7 = 3 ∨ d % 7 = 4 ∨ d % 7 = 5)

theorem valid_starting_days_count : 
  (finset.filter is_valid_starting_day (finset.range 7)).card = 3 :=
begin
  sorry
end

end valid_starting_days_count_l813_813304


namespace complex_circle_intersection_l813_813110

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l813_813110


namespace compare_data_sets_l813_813903

-- Definitions of the components of the data sets.
variables {x1 x2 x3 x4 x5 : ℝ}

-- Definitions for conditions
def mean_x := (x1 + x2 + x3 + x4 + x5) / 5
def variance_x := ((x1 - mean_x) ^ 2 + (x2 - mean_x) ^ 2 + (x3 - mean_x) ^ 2 + (x4 - mean_x) ^ 2 + (x5 - mean_x) ^ 2) / 5
def std_dev_x := real.sqrt variance_x
def median_x := x3  -- assuming the list {x1, x2, x3, x4, x5} is sorted

def mean_y := (2 * x1 + 3 + 2 * x2 + 3 + 2 * x3 + 3 + 2 * x4 + 3 + 2 * x5 + 3) / 5
def variance_y := ((2 * x1 + 3 - mean_y) ^ 2 + (2 * x2 + 3 - mean_y) ^ 2 + (2 * x3 + 3 - mean_y) ^ 2 + (2 * x4 + 3 - mean_y) ^ 2 + (2 * x5 + 3 - mean_y) ^ 2) / 5
def std_dev_y := real.sqrt variance_y
def median_y := 2 * x3 + 3

theorem compare_data_sets : mean_x ≠ mean_y ∧ std_dev_x ≠ std_dev_y ∧ median_x ≠ median_y :=
by sorry

end compare_data_sets_l813_813903


namespace constant_term_in_binomial_expansion_l813_813020

noncomputable def binomialExpansionConstantTerm (a : ℝ) : ℝ :=
  (∫ x in 0..2, (2*x - 1)) = a → 
  let b := x + (a / x) in 
  ((b^4).coeff 0) = 24

theorem constant_term_in_binomial_expansion : binomialExpansionConstantTerm 2 := by
  sorry

end constant_term_in_binomial_expansion_l813_813020


namespace negation_example_l813_813573

open Real

theorem negation_example :
  (¬ ∃ x : ℝ, ln x < x^2 - 1) ↔ (∀ x : ℝ, ln x ≥ x^2 - 1) :=
sorry

end negation_example_l813_813573


namespace arithmetic_sequence_extreme_points_log_product_l813_813084

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x^2 + 6 * x - 1

theorem arithmetic_sequence_extreme_points_log_product :
  (∃ a2 a2017 a4032 : ℝ,
    (a2 + a4032 = 2 * a2017) ∧
    (a2 * a4032 = 6) ∧
    (f' a2 = 0) ∧
    (f' a4032 = 0) ∧
    (a2017 = 4 ∧ f' x = x^2 - 8 * x + 6)) →
  log 2 (a2 * a2017 * a4032) = 3 + log 2 3 := sorry

end arithmetic_sequence_extreme_points_log_product_l813_813084


namespace faster_speed_l813_813297

def jogger_distance_slow : ℝ := 30 -- the actual distance jogged at slower speed
def jogger_speed_slow : ℝ := 12 -- the slower speed in km/hr
def jogger_additional_distance : ℝ := 10 -- the additional distance if jogged at faster speed
def jogger_time_slow : ℝ := jogger_distance_slow / jogger_speed_slow -- time taken to jog at slower speed
def jogger_distance_fast : ℝ := jogger_distance_slow + jogger_additional_distance -- distance jogged at faster speed

theorem faster_speed (v : ℝ) :
  jogger_time_slow = jogger_distance_fast / v → v = 16 :=
by
  sorry

end faster_speed_l813_813297


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813694

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813694


namespace find_max_side_length_l813_813792

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813792


namespace max_side_of_triangle_l813_813809

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813809


namespace days_with_equal_sun_tue_l813_813300

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l813_813300


namespace maximum_value_and_monotonic_intervals_tangent_line_equations_l813_813426

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 4*x + 1

theorem maximum_value_and_monotonic_intervals :
  ∃ (m : ℝ), m = 2 ∧ ∀ x : ℝ,
    (f(-2) = 9) ∧
    (f' x = 3*x^2 + 4*x - 4) ∧
    ( ∀ x < -2, f' x > 0 ) ∧
    ( -2 < x ∧ x < 2/3 → f' x < 0 ) ∧
    ( ∀ x > 2/3, f' x > 0 ) :=
by sorry

theorem tangent_line_equations :
  ∀ x : ℝ, 
    (f'(x) = 2 → ( ∃ k : ℝ, k = (-2 + Real.sqrt 22) / 3 ∨ k = (-2 - Real.sqrt 22) / 3) ) ∧
    ( k = (-2 + Real.sqrt(22)) / 3 → ( ∃ y, y - f(k) = 2*(x - k) ) ∨ (\ k = (-2 - Real.sqrt(22)) / 3 → ( ∃ y, y - f(k) = 2*(x - k) )) :=
by sorry

end maximum_value_and_monotonic_intervals_tangent_line_equations_l813_813426


namespace find_e_l813_813577

-- Define the conditions and state the theorem.
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) 
  (h1: ∃ a b c : ℝ, (a + b + c)/3 = -3 ∧ a * b * c = -3 ∧ 3 + d + e + f = -3)
  (h2: Q 0 d e f = 9) : e = -42 :=
by
  sorry

end find_e_l813_813577


namespace inequality_median_bound_l813_813533

noncomputable theory
open_locale big_operators

theorem inequality_median_bound 
  (a : ℝ) 
  (p : ℝ) 
  (hp : p ≥ 1)
  (ξ : ℝ → MeasureTheory.ProbabilityMeasure)
  (hξ_integrable : ∫⁻ x, (|ξ(x)|^p : ℝ≥0∞) < ⊤)
  (μ : ℝ)
  (hμ_median : ∀ E : Set ℝ, 0 < MeasureTheory.ProbabilityMeasure.measure E → 
    (↑μ ∈ E ↔ MeasureTheory.ProbabilityMeasure.measure {x : ℝ | x ≥ μ} ≥ 1/2)) :
  |μ - a| ^ p ≤ 2 * ∫ x, |ξ(x) - a|^p :=
sorry

end inequality_median_bound_l813_813533


namespace solve_cubic_equation_l813_813544

theorem solve_cubic_equation : ∀ x : ℝ, (x^3 - 5*x^2 + 6*x - 2 = 0) → (x = 2) :=
by
  intro x
  intro h
  sorry

end solve_cubic_equation_l813_813544


namespace perimeter_of_similar_triangle_isosceles_l813_813980

noncomputable def isosceles_triangle {α : Type*} [linear_ordered_ring α] :=
  ∃ (a b c : α), a = b ∧ a = 15 ∧ c = 24

noncomputable def similar_triangle {α : Type*} [linear_ordered_ring α] :=
  ∃ (base 𝑙𝑠₁ 𝑙𝑠₂ : α), base = 60 ∧ (∃ (s₁ s₂ : α), s₁ * (60 / 24) = 𝑙𝑠₁ ∧ s₂ * (60 / 24) = 𝑙𝑠₂ ∧ s₁ = 15 ∧ s₂ = 15)

theorem perimeter_of_similar_triangle_isosceles (α : Type*) [linear_ordered_ring α] :
  isosceles_triangle ∧ similar_triangle → ∃ (P : α), P = 135 :=
by
  sorry

end perimeter_of_similar_triangle_isosceles_l813_813980


namespace non_empty_subsets_without_isolated_elements_card_l813_813581

def S := {0, 1, 2, 3, 4, 5}
def has_isolated_element (A : set ℕ) (x : ℕ) : Prop := x ∈ A ∧ (x-1 ∉ A ∧ x+1 ∉ A)

def is_non_empty_subset_without_isolated_elements (A : set ℕ) : Prop :=
  A ≠ ∅ ∧ ∀ x ∈ A, ¬ has_isolated_element A x

theorem non_empty_subsets_without_isolated_elements_card : 
  (∃ A : set ℕ, A ⊆ S ∧ is_non_empty_subset_without_isolated_elements A) :=
  sorry

end non_empty_subsets_without_isolated_elements_card_l813_813581


namespace ellipse_standard_equation_range_reciprocal_distances_l813_813420

theorem ellipse_standard_equation 
    (x y : ℝ)
    (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
    (M : ℝ × ℝ)
    (hF1 : F1 = (0, -Real.sqrt 3))
    (hF2 : F2 = (0, Real.sqrt 3))
    (hM : M = (Real.sqrt 3 / 2, 1))
    (hPassesThroughM : (M.1^2 + M.2^2 / 4 = 1)) :
    (y^2 / 4 + x^2 = 1) :=
sorry

theorem range_reciprocal_distances 
    (P : ℝ × ℝ)
    (F1 : ℝ × ℝ)
    (F2 : ℝ × ℝ)
    (hF1 : F1 = (0, -Real.sqrt 3))
    (hF2 : F2 = (0, Real.sqrt 3))
    (hP_on_ellipse : P.1^2 + P.2^2 / 4 = 1) :
    (1 ≤ 1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 + 1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) 
    ∧ 1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 + 1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) ≤ 4) :=
sorry

end ellipse_standard_equation_range_reciprocal_distances_l813_813420


namespace problem_l813_813295

open Real

variable (f : ℝ → ℝ)
variable (cond1 : ∀ x : ℝ, 0 < x → f (2 * x) = 2 * (f x))
variable (cond2 : ∀ x : ℝ, 1 < x ∧ x ≤ 2 → f x = 2 - x)

theorem problem (f : ℝ → ℝ)
  (cond1 : ∀ x : ℝ, 0 < x → f (2 * x) = 2 * (f x))
  (cond2 : ∀ x : ℝ, 1 < x ∧ x ≤ 2 → f x = 2 - x) :
  (∀ m : ℤ, f (2^m) = 0) ∧
  (∃ y : ℝ, 0 ≤ y → ∃ x : ℝ, 0 < x → f x = y) ∧
  ¬ (∃ n : ℤ, f (2^n + 1) = 9) ∧
  (∀ a b : ℝ, (∃ k : ℤ, a > 0 ∧ b > 0 ∧ a < b ∧ (a, b) ⊆ (2^k, 2^(k+1))) →  (∀ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ ≥ f x₂)) := 
sorry

end problem_l813_813295


namespace all_three_pets_l813_813073

-- Definitions of the given conditions
def total_students : ℕ := 40
def dog_owners : ℕ := 20
def cat_owners : ℕ := 13
def other_pet_owners : ℕ := 8
def no_pets : ℕ := 7

-- Definitions from Venn diagram
def dogs_only : ℕ := 12
def cats_only : ℕ := 3
def other_pets_only : ℕ := 2

-- Intersection variables
variables (a b c d : ℕ)

-- Translated problem
theorem all_three_pets :
  dogs_only + cats_only + other_pets_only + a + b + c + d = total_students - no_pets ∧
  dogs_only + a + c + d = dog_owners ∧
  cats_only + a + b + d = cat_owners ∧
  other_pets_only + b + c + d = other_pet_owners ∧
  d = 2 :=
sorry

end all_three_pets_l813_813073


namespace gcd_mn_eq_one_l813_813260

def m : ℤ := 123^2 + 235^2 - 347^2
def n : ℤ := 122^2 + 234^2 - 348^2

theorem gcd_mn_eq_one : Int.gcd m n = 1 := 
by
  sorry

end gcd_mn_eq_one_l813_813260


namespace max_side_length_is_11_l813_813673

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813673


namespace seating_arrangement_l813_813868

theorem seating_arrangement (x: ℕ) (y: ℕ) (z: ℕ) (h : (6 * x + 8 * y + 9 * z = 58) ∧ (x + y + z ≤ 7)) : z = 4 :=
by {
  sorry,
}

end seating_arrangement_l813_813868


namespace no_extreme_points_in_interval_l813_813240

theorem no_extreme_points_in_interval (a : ℝ) (f : ℝ → ℝ) :
  (∀ x ∈ Ioo (2 : ℝ) 3, f x = (x - a) * Real.exp x) →
  (∀ x ∈ Ioo (2 : ℝ) 3, (f x).deriv ≠ 0) →
  a ∈ Set.Iic 3 ∪ Set.Ici 4 :=
by
  intros h1 h2
  sorry

end no_extreme_points_in_interval_l813_813240


namespace average_of_last_six_numbers_l813_813556

theorem average_of_last_six_numbers
  (numbers : List ℝ) 
  (h_len : numbers.length = 11)
  (h_avg_11 : (numbers.sum / 11) = 9.9)
  (h_avg_6 : (numbers.take 6).sum / 6 = 10.5)
  (h_middle : numbers.nth 6 = some 22.5) :
  ((numbers.drop 5).sum / 6) = 3.9 :=
sorry

end average_of_last_six_numbers_l813_813556


namespace valid_starting_days_count_l813_813303

def is_valid_starting_day (d : ℕ) : Prop :=
  (d % 7 = 3 ∨ d % 7 = 4 ∨ d % 7 = 5)

theorem valid_starting_days_count : 
  (finset.filter is_valid_starting_day (finset.range 7)).card = 3 :=
begin
  sorry
end

end valid_starting_days_count_l813_813303


namespace no_3_by_3_all_purple_l813_813346

theorem no_3_by_3_all_purple (m n : ℕ) (hrel : Nat.coprime m n) (prob : (m * 128 = 127 * n)) : m + n = 255 :=
by
    have total_colorings : ℕ := 2^16
    have disallowed_colorings : ℕ := 4 * 2^7
    have allowed_colorings := total_colorings - disallowed_colorings
    have fraction := allowed_colorings.toRational / total_colorings.toRational
    have m_val : ℕ := 127
    have n_val : ℕ := 128
    have prob_correct : fraction = (m_val : ℚ) / (n_val : ℚ) :=
        by 
            rw [fraction]
            simp [allowed_colorings, total_colorings, m_val, n_val]
    have common_factor := Nat.gcd m_val n_val
    have eq_val : m = 127 := sorry
    have eq_val_₁ : n = 128 := sorry
    exact eq_val.symm ▸ eq_val_₁.symm ▸ rfl

end no_3_by_3_all_purple_l813_813346


namespace measure_gamma_right_angle_l813_813388

noncomputable def angle_measure (α β γ : ℝ) : Prop :=
  (γ ∈ Icc 0 π) ∧ 
  (α + β + γ = π) ∧ 
  (sin α + sin β = (cos α + cos β) * sin γ)
  
theorem measure_gamma_right_angle (α β γ : ℝ) (h : angle_measure α β γ) : 
  γ = π / 2 := 
by
  sorry

end measure_gamma_right_angle_l813_813388


namespace math_problem_l813_813011

def proposition_p (a : ℝ) (h : 1 < a ∧ a < 2) : Prop :=
  ∀ x ∈ set.Icc (0 : ℝ) 1, ∀ y ∈ set.Icc (0 : ℝ) 1, x ≤ y → log a (2 - a*y) ≤ log a (2 - a*x)

def proposition_q (a : ℝ) (h : 1 < a ∧ a < 2) : Prop :=
  ∀ x : ℝ, abs x < 1 → x < a

theorem math_problem (a : ℝ) (h : 1 < a ∧ a < 2) :
  (proposition_p a h ∨ proposition_q a h) :=
sorry

end math_problem_l813_813011


namespace am_gm_inequality_l813_813392

theorem am_gm_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end am_gm_inequality_l813_813392


namespace frac_1_7_correct_l813_813222

-- Define the fraction 1/7
def frac_1_7 : ℚ := 1 / 7

-- Define the decimal approximation 0.142857142857 as a rational number
def dec_approx : ℚ := 142857142857 / 10^12

-- Define the small fractional difference
def small_diff : ℚ := 1 / (7 * 10^12)

-- The theorem to be proven
theorem frac_1_7_correct :
  frac_1_7 = dec_approx + small_diff := 
sorry

end frac_1_7_correct_l813_813222


namespace L_shaped_symmetry_l813_813323

-- Definitions
def original_shape := (long_top : Prop) (short_bottom : Prop)
def shape_C := (short_top : Prop) (long_bottom : Prop)

-- Symmetry condition
def reflection_symmetric (top_1 bottom_1 top_2 bottom_2: Prop) :=
  top_1 = bottom_2 ∧ bottom_1 = top_2

-- The theorem statement
theorem L_shaped_symmetry :
  reflection_symmetric (original_shape long_top short_bottom) (shape_C short_top long_bottom) :=
sorry

end L_shaped_symmetry_l813_813323


namespace mixed_doubles_teams_l813_813201

theorem mixed_doubles_teams (m n : ℕ) (h_m : m = 7) (h_n : n = 5) :
  (∃ (k : ℕ), k = 4) ∧ (m ≥ 2) ∧ (n ≥ 2) →
  ∃ (number_of_combinations : ℕ), number_of_combinations = 2 * Nat.choose 7 2 * Nat.choose 5 2 :=
by
  intros
  sorry

end mixed_doubles_teams_l813_813201


namespace find_angle_B_find_area_triangle_l813_813485

-- Definitions of the conditions
def triangle (A B C : Type) :=  -- This can be expanded based on actual geometry definitions in Lean libraries
  ∃ (a b c : ℝ), -- Sides of the triangle
    a > 0 ∧ b > 0 ∧ c > 0 ∧     -- Sides must be positive
    true -- Placeholder to represent a valid triangle (more can be added)

-- Placeholder for cosine law related proof
axiom cosine_law (a b c : ℝ) (A B C : ℝ) : Prop :=
 -- Assuming this or related geometry proof exists in Lean libraries

-- Problem Statement Part (I)
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (h : 2 * b * cos C = 2 * a - c) : 
  B = π / 3 := 
sorry

-- Problem Statement Part (II)
theorem find_area_triangle (b c : ℝ) (A B : ℝ) (B_eq : B = π / 3) (b_val : b = sqrt 7) (c_val : Rational.c = 2) :
  let a : ℝ := 3 in
  let sin_B : ℝ := sqrt 3 / 2 in
  a * c * sin (B) / 2 = 3 * sqrt 3 / 2 :=
sorry

end find_angle_B_find_area_triangle_l813_813485


namespace real_part_of_product_l813_813404

open Complex

theorem real_part_of_product (α β : ℝ) :
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  Complex.re (z1 * z2) = Real.cos (α + β) :=
by
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  sorry

end real_part_of_product_l813_813404


namespace arctan_identity_l813_813847

theorem arctan_identity : 
  ∀ (x y : ℝ), 
    tan (70 * real.pi / 180) - 2 * tan (35 * real.pi / 180) = tan (20 * real.pi / 180) → 
    real.arctan (tan (70 * real.pi / 180) - 2 * tan (35 * real.pi / 180)) = 20 * real.pi / 180 :=
begin
  intros x y h,
  sorry
end

end arctan_identity_l813_813847


namespace product_of_reflected_midpoint_coords_l813_813194

def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (13, 16)
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def reflect_x (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

theorem product_of_reflected_midpoint_coords :
  let N := midpoint A B,
      N' := reflect_x N
  in N'.1 * N'.2 = -72 :=
by
  sorry

end product_of_reflected_midpoint_coords_l813_813194


namespace intersection_of_circles_l813_813088

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l813_813088


namespace max_side_length_l813_813774

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813774


namespace perpendicular_lines_slope_l813_813067

theorem perpendicular_lines_slope (a : ℝ) :
  let m1 := 2 in
  let m2 := -a / 2 in
  (m1 * m2 = -1) → (a = 1) :=
by
  sorry

end perpendicular_lines_slope_l813_813067


namespace part_I_part_II_l813_813475

namespace ArithmeticGeometricSequences

-- Definitions of sequences and their properties
def a1 : ℕ := 1
def b1 : ℕ := 2
def b (n : ℕ) : ℕ := 2 * 3 ^ (n - 1) -- General term of the geometric sequence

-- Definitions from given conditions
def a (n : ℕ) : ℕ := 3 * n - 2 -- General term of the arithmetic sequence

-- Sum of the first n terms of the geometric sequence
def S (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n - 1

-- Theorem statement
theorem part_I (n : ℕ) : 
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) →
  (a n = 3 * n - 2) ∧ 
  (b n = 2 * 3 ^ (n - 1)) :=
  sorry

theorem part_II (n : ℕ) (m : ℝ) :
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) → 
  (∀ n > 0, S n + a n > m) → 
  (m < 3) :=
  sorry

end ArithmeticGeometricSequences

end part_I_part_II_l813_813475


namespace return_journey_time_l813_813176

-- Define the conditions
def walking_speed : ℕ := 100 -- meters per minute
def walking_time : ℕ := 36 -- minutes
def running_speed : ℕ := 3 -- meters per second

-- Define derived values from conditions
def distance_walked : ℕ := walking_speed * walking_time -- meters
def running_speed_minute : ℕ := running_speed * 60 -- meters per minute

-- Statement of the problem
theorem return_journey_time :
  (distance_walked / running_speed_minute) = 20 := by
  sorry

end return_journey_time_l813_813176


namespace sum_fourth_and_sixth_term_l813_813972

def seq (n : ℕ) : ℚ :=
  if n = 1 then 2 else ((n : ℚ) / (n - 1))^(3: ℚ)

theorem sum_fourth_and_sixth_term :
  seq 4 + seq 6 = (13832 / 3375 : ℚ) := by
sorry

end sum_fourth_and_sixth_term_l813_813972


namespace omega_range_l813_813034

theorem omega_range (ω : ℝ) (h : 0 < ω) :
  (∀ x ∈ set.Ioo (π / 12) (π / 3), (∃ y ∈ set.Ioo (π / 4) (5 * π / 4), f x = sin (ω * x + y)) 
    ∧ ¬(∃ z ∈ set.Ioo (5 * π / 4) (9 * π / 4), f x = sin (ω * x + z))) ↔ ω ∈ set.Ioo (3 / 4) 3 :=
sorry

end omega_range_l813_813034


namespace real_solutions_eq_l813_813875

theorem real_solutions_eq (x y : ℝ) : 
  (x^2 + 2 * x * sin(x * y) + 1 = 0) ↔ 
  (∃ k : ℤ, (x = 1 ∧ y = (2 * k + 1) * π / 2) ∨ 
            (x = -1 ∧ y = (2 * k + 1) * π / 2)) :=
by { sorry }

end real_solutions_eq_l813_813875


namespace steven_sixth_quiz_score_l813_813207

theorem steven_sixth_quiz_score :
  ∃ x : ℕ, (75 + 80 + 85 + 90 + 100 + x) / 6 = 95 ∧ x = 140 :=
by
  sorry

end steven_sixth_quiz_score_l813_813207


namespace original_number_l813_813062

theorem original_number (h : 2.04 / 1.275 = 1.6) : 204 / 12.75 = 16 := 
by
  sorry

end original_number_l813_813062


namespace h_at_2_l813_813160

noncomputable def f (x : ℝ) : ℝ := 3 * x - 4
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (3 * f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_2 : h 2 = 3 * Real.sqrt 6 - 13 := 
by 
  sorry -- We skip the proof steps.

end h_at_2_l813_813160


namespace gain_percent_is_correct_l813_813944

theorem gain_percent_is_correct (gain_in_paise : ℝ) (cost_price_in_rs : ℝ) (conversion_factor : ℝ)
  (gain_percent_formula : ∀ (gain : ℝ) (cost : ℝ), ℝ) : 
  gain_percent_formula (gain_in_paise / conversion_factor) cost_price_in_rs = 1 :=
by
  let gain := gain_in_paise / conversion_factor
  let cost := cost_price_in_rs
  have h : gain_percent_formula gain cost = (gain / cost) * 100 := sorry
  have h2 : gain_percent_formula (70 / 100) 70 = 1 := sorry
  exact h2

end gain_percent_is_correct_l813_813944


namespace sum_distinct_prime_factors_of_expr_l813_813891

theorem sum_distinct_prime_factors_of_expr : 
  ∑ p in {2, 3, 7}, p = 12 :=
by
  -- The proof will be written here.
  sorry

end sum_distinct_prime_factors_of_expr_l813_813891


namespace pizza_slices_left_l813_813192

theorem pizza_slices_left (initial_slices cuts friends total_given : ℕ) (h_slices: initial_slices = 1 * 2^cuts)
  (h_first_group: ∀ f ∈ friends, f = 2) (h_second_group: ∀ f ∉ friends, f = 1) 
  (h_friends: ∑ f in friends, f = total_given) 
  (distribution: total_given = (card friends * 2 + (card (range 3) - card friends) * 1)) 
  (h_card_friends: card friends = 2) : 
  (initial_slices - total_given = 1) :=
sorry

end pizza_slices_left_l813_813192


namespace problem_part1_problem_part2_l813_813482

noncomputable def pointP : (ℝ × ℝ) := (0, sqrt 3)
noncomputable def lineL : (ℝ → ℝ × ℝ) := λ t, (-1/2 * t, sqrt 3 + sqrt 3 / 2 * t)
noncomputable def curveC (φ : ℝ) : (ℝ × ℝ) := (sqrt 5 * cos φ, sqrt 15 * sin φ)

theorem problem_part1 : ∃ t : ℝ, lineL t = pointP := 
by {
  let t := 0
  use t
  unfold lineL pointP
  sorry
}

theorem problem_part2 : ∃ A B : (ℝ × ℝ), (λ (A B : ℝ × ℝ), |A.1 - pointP.1| * |B.1 - pointP.1| = 8) :=
by {
  let t := λ t, lineL t
  let A := t 2
  let B := t (-4)
  use A, B
  sorry
}

end problem_part1_problem_part2_l813_813482


namespace exterior_angle_of_polygon_l813_813068

theorem exterior_angle_of_polygon (n : ℕ) (h₁ : (n - 2) * 180 = 1800) (h₂ : n > 2) :
  360 / n = 30 := by
    sorry

end exterior_angle_of_polygon_l813_813068


namespace find_k_values_for_intersection_l813_813118

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l813_813118


namespace complex_coordinate_l813_813857

open Complex

theorem complex_coordinate (z : ℂ) (h : z = (2 * complex.I) / (1 - complex.I)) : z = -1 + complex.I :=
by
  rw [h]
  sorry

end complex_coordinate_l813_813857


namespace proof_angles_and_ratios_l813_813072

def angles_and_ratios (A B C a b c : ℝ) (h1 : a^2 + b^2 - c^2 = a * b) (h2 : 2 * cos (A / 2)^2 - 2 * sin (B / 2)^2 = sqrt 3 / 2) (h3 : A < B) : Prop :=
  (C = π / 3) ∧ (c / a = sqrt 3)

theorem proof_angles_and_ratios (A B C a b c : ℝ) 
  (h1 : a^2 + b^2 - c^2 = a * b) 
  (h2 : 2 * cos (A / 2)^2 - 2 * sin (B / 2)^2 = sqrt 3 / 2) 
  (h3 : A < B) :
  angles_and_ratios A B C a b c h1 h2 h3 :=
by
  sorry

end proof_angles_and_ratios_l813_813072


namespace maleAssociateFullTenurePercentage_l813_813334

variable (P : ℕ) -- Total number of professors

-- Conditions
def numWomen : ℕ := 7 * P / 10
def numTenured : ℕ := 7 * P / 10
def numAbove60Years : ℕ := 15 * numTenured / 100
def numAssociateFullProfessors : ℕ := 5 * P / 10
def numWomenTenuredOrBoth : ℕ := 9 * P / 10
def numMaleAssociateFullProfessors : ℕ := 8 * numAssociateFullProfessors / 10

-- Derived calculations based on conditions
def numMaleNotWomenNotTenured : ℕ := P - numWomenTenuredOrBoth
def numMaleTotal : ℕ := P - numWomen
def numMaleTenured : ℕ := numMaleTotal * 7 / 10
def percentMaleAssociateFullTenured : ℕ := numMaleTenured * 100 / numMaleAssociateFullProfessors

-- Goal: Prove that 50% of the male associate or full professors are tenured
theorem maleAssociateFullTenurePercentage : percentMaleAssociateFullTenured = 50 :=
by
  sorry

end maleAssociateFullTenurePercentage_l813_813334


namespace num_four_digit_integers_divisible_by_12_and_8_l813_813057

-- Definitions
def four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n < 10000}
def divisible_by_12_and_8 (n : ℕ) := n % 12 = 0 ∧ n % 8 = 0

-- Problem Statement
theorem num_four_digit_integers_divisible_by_12_and_8 :
  {n : ℕ | n ∈ four_digit_numbers ∧ divisible_by_12_and_8 n}.finite.card = 375 :=
by
  -- skip the proof
  sorry

end num_four_digit_integers_divisible_by_12_and_8_l813_813057


namespace difference_max_min_y_l813_813021

noncomputable def a (n : ℕ) : ℝ × ℝ := (Real.cos (n * Real.pi / 3), Real.sin (n * Real.pi / 3))
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def y (θ : ℝ) : ℝ := ∑ n in Finset.range 100, 
  let an := a (n+1) in let bn := b θ in (an.1 + bn.1)^2 + (an.2 + bn.2)^2 

theorem difference_max_min_y (θ : ℝ) : ∀ θ, 
   let max_y := 200 + 2 * Real.sqrt 3, 
       min_y := 200 - 2 * Real.sqrt 3 in 
   y θ = 200 - 2 * Real.sqrt 3 * Real.sin(θ + Real.pi / 3) →
   max_y - min_y = 4 * Real.sqrt 3 :=
sorry

end difference_max_min_y_l813_813021


namespace imaginary_number_condition_fourth_quadrant_condition_l813_813393

-- Part 1: Prove that if \( z \) is purely imaginary, then \( m = 0 \)
theorem imaginary_number_condition (m : ℝ) :
  (m * (m + 2) = 0) ∧ (m^2 + m - 2 ≠ 0) → m = 0 :=
by
  sorry

-- Part 2: Prove that if \( z \) is in the fourth quadrant, then \( 0 < m < 1 \)
theorem fourth_quadrant_condition (m : ℝ) :
  (m * (m + 2) > 0) ∧ (m^2 + m - 2 < 0) → (0 < m ∧ m < 1) :=
by
  sorry

end imaginary_number_condition_fourth_quadrant_condition_l813_813393


namespace max_side_of_triangle_l813_813798

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813798


namespace max_side_length_of_triangle_l813_813757

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813757


namespace intersection_of_circles_l813_813091

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l813_813091


namespace rectangle_hall_length_l813_813627

variable (L B : ℝ)

theorem rectangle_hall_length (h1 : B = (2 / 3) * L) (h2 : L * B = 2400) : L = 60 :=
by sorry

end rectangle_hall_length_l813_813627


namespace percentage_difference_l813_813616

theorem percentage_difference :
    let A := (40 / 100) * ((50 / 100) * 60)
    let B := (50 / 100) * ((60 / 100) * 70)
    (B - A) = 9 :=
by
    sorry

end percentage_difference_l813_813616


namespace angle_bisectors_envelop_parabola_l813_813512

-- Conditions
variable (l : Line) (F : Point)

-- Definitions used in proof
def angle_bisector (A M : Point) (F : Point) : Prop := -- defining the angle bisector property
sorry

def perpendicular (B : Point) (bisector : Prop) : Prop := -- defining perpendicular construction
sorry

def locus_of_points (F : Point) (directrix : Line) (M : Point) : Prop := -- defining locus of points forming a parabola
MB = MF

-- Problem statement
theorem angle_bisectors_envelop_parabola :
  ∀ A : Point, (A ∈ l) → 
  (∃ (angle_bisector (A M : Point) F), 
    (∃ B : Point, (B ∈ l) ∧ perpendicular B (angle_bisector A M F)) → 
    (locus_of_points F l M)) :=
by
  sorry

end angle_bisectors_envelop_parabola_l813_813512


namespace total_cost_l813_813655

-- Define the cost of a neutral pen and a pencil
variables (x y : ℝ)

-- The total cost of buying 5 neutral pens and 3 pencils
theorem total_cost (x y : ℝ) : 5 * x + 3 * y = 5 * x + 3 * y :=
by
  -- The statement is self-evident, hence can be written directly
  sorry

end total_cost_l813_813655


namespace find_m_value_l813_813914

noncomputable def x0 : ℝ := sorry

noncomputable def m : ℝ := x0^3 + 2 * x0^2 + 2

theorem find_m_value :
  (x0^2 + x0 - 1 = 0) → (m = 3) :=
by
  intro h
  have hx : x0 = sorry := sorry
  have hm : m = x0 ^ 3 + 2 * x0^2 + 2 := rfl
  rw [hx] at hm
  sorry

end find_m_value_l813_813914


namespace correct_sample_statement_l813_813250

def population : Type := { ages : List ℕ // ages.length = 1000 }
def individual := ℕ
def sample (pop: population) : Type := { sampled_ages : List ℕ // sampled_ages.length = 100 }

theorem correct_sample_statement :
  ∀ (pop : population) (sample : sample pop), 
    sample.sampled_ages.length = 100 := 
by
  intros,
  sorry

end correct_sample_statement_l813_813250


namespace gcm_10_15_less_than_90_l813_813261

noncomputable def gcd (a b : ℕ) : ℕ := 
  if a = 0 then b else gcd (b % a) a

noncomputable def lcm (a b : ℕ) : ℕ := 
  (a * b) / gcd a b

noncomputable def greatest_common_multiple_less_than (a b n : ℕ) : ℕ :=
  let l := lcm a b
  let multiples_less_than_n := (n-1) / l
  l * multiples_less_than_n

theorem gcm_10_15_less_than_90 : greatest_common_multiple_less_than 10 15 90 = 60 :=
  by 
    sorry

end gcm_10_15_less_than_90_l813_813261


namespace find_base_l813_813961

theorem find_base (b x y : ℝ) (h₁ : b^x * 4^y = 59049) (h₂ : x = 10) (h₃ : x - y = 10) : b = 3 :=
by
  sorry

end find_base_l813_813961


namespace range_of_ω_l813_813429

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ :=
  Real.cos (ω * x + ϕ)

theorem range_of_ω :
  ∀ (ω : ℝ) (ϕ : ℝ),
    (0 < ω) →
    (-π ≤ ϕ) →
    (ϕ ≤ 0) →
    (∀ x, f x ω ϕ = -f (-x) ω ϕ) →
    (∀ x1 x2, (x1 < x2) → (-π/4 ≤ x1 ∧ x1 ≤ 3*π/16) ∧ (-π/4 ≤ x2 ∧ x2 ≤ 3*π/16) → f x1 ω ϕ ≤ f x2 ω ϕ) →
    (0 < ω ∧ ω ≤ 2) :=
by
  sorry

end range_of_ω_l813_813429


namespace derivative_f_l813_813431

def f (x : ℝ) : ℝ := Real.exp (-2 * x)

theorem derivative_f (x : ℝ) : deriv f x = -2 * Real.exp (-2 * x) :=
by
  sorry

end derivative_f_l813_813431


namespace units_digit_2016_pow_2017_add_2017_pow_2016_l813_813265

theorem units_digit_2016_pow_2017_add_2017_pow_2016 :
  (2016 ^ 2017 + 2017 ^ 2016) % 10 = 7 :=
by
  sorry

end units_digit_2016_pow_2017_add_2017_pow_2016_l813_813265


namespace polar_eqn_curve_C_correct_OP_OQ_product_correct_l813_813479

structure CurveParametricEquations (α : ℝ) :=
(x : ℝ) (y : ℝ)

structure LineEquation :=
(k : ℝ) (c : ℝ)

def curve_C_parametric : CurveParametricEquations ℝ :=
{ x := λ α, sqrt 3 + 2 * cos α,
  y := λ α, 2 + 2 * sin α }

def line_l : LineEquation :=
{ k := sqrt 3 / 3,
  c := 0 }

noncomputable def polar_eqn_curve_C (ρ θ : ℝ) :=
  ρ^2 - 2*sqrt 3*ρ*cos θ - 4*ρ*sin θ + 3 = 0

theorem polar_eqn_curve_C_correct (ρ θ : ℝ) :
  polar_eqn_curve_C ρ θ = (ρ^2 - 2*sqrt (3 : ℝ)*ρ*cos θ - 4*ρ*sin θ + 3 = 0) :=
by sorry

noncomputable def OP_OQ_product : ℝ :=
  let θ := π / 6 in
  let ρ_roots := [1, 2] in -- Placeholder: actual solution requires solving quadratic
  ρ_roots.product

theorem OP_OQ_product_correct : OP_OQ_product = 3 :=
by sorry

end polar_eqn_curve_C_correct_OP_OQ_product_correct_l813_813479


namespace final_selling_price_l813_813651

-- Define the conditions as constants
def CP := 750
def loss_percentage := 20 / 100
def sales_tax_percentage := 10 / 100

-- Define the final selling price after loss and adding sales tax
theorem final_selling_price 
  (CP : ℝ) 
  (loss_percentage : ℝ)
  (sales_tax_percentage : ℝ) 
  : 750 = CP ∧ 20 / 100 = loss_percentage ∧ 10 / 100 = sales_tax_percentage → 
    (CP - (loss_percentage * CP) + (sales_tax_percentage * CP) = 675) := 
by
  intros
  sorry

end final_selling_price_l813_813651


namespace range_of_a_l813_813930

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 3 * x + a < 0) ∧ (∀ x : ℝ, 2 * x + 7 > 4 * x - 1) ∧ (∀ x : ℝ, x < 0) → a = 0 := 
by sorry

end range_of_a_l813_813930


namespace percentage_of_students_in_70_to_79_range_l813_813171

def num_students_in_90_to_100 := 5
def num_students_in_80_to_89 := 7
def num_students_in_70_to_79 := 9
def num_students_in_60_to_69 := 4
def num_students_below_60 := 3
def total_students :=
  num_students_in_90_to_100 + num_students_in_80_to_89 +
  num_students_in_70_to_79 + num_students_in_60_to_69 +
  num_students_below_60

theorem percentage_of_students_in_70_to_79_range :
  ((num_students_in_70_to_79 : ℝ) / total_students * 100 ≈ 32.14) := 
sorry

end percentage_of_students_in_70_to_79_range_l813_813171


namespace find_max_side_length_l813_813787

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813787


namespace sum_of_prime_factors_l813_813587

theorem sum_of_prime_factors (x : ℕ) (h1 : x = 2^10 - 1) 
  (h2 : 2^10 - 1 = (2^5 + 1) * (2^5 - 1)) 
  (h3 : 2^5 - 1 = 31) 
  (h4 : 2^5 + 1 = 33) 
  (h5 : 33 = 3 * 11) : 
  (31 + 3 + 11 = 45) := 
  sorry

end sum_of_prime_factors_l813_813587


namespace cost_of_normal_mouse_l813_813519

-- Definitions of the conditions based on the problem statement
def num_days_open_per_week : ℕ := 7 - 3 -- Ned's store is open 4 days a week
def mice_sold_per_day : ℕ := 25 -- He sells 25 left-handed mice per day
def revenue_per_week : ℝ := 15600 -- He makes $15,600 a week
def price_multiplier : ℝ := 1.30 -- Left-handed mice cost 30% more than normal mice

-- Goal: Prove the cost of a normal mouse
theorem cost_of_normal_mouse : 
  let mice_sold_per_week := mice_sold_per_day * num_days_open_per_week
  let revenue_per_mouse := revenue_per_week / mice_sold_per_week
  let cost_of_normal_mouse := revenue_per_mouse / price_multiplier in
  cost_of_normal_mouse = 120 := 
by
  sorry

end cost_of_normal_mouse_l813_813519


namespace best_fit_model_l813_813981

-- Define the coefficients of determination for each model
noncomputable def R2_Model1 : ℝ := 0.75
noncomputable def R2_Model2 : ℝ := 0.90
noncomputable def R2_Model3 : ℝ := 0.45
noncomputable def R2_Model4 : ℝ := 0.65

-- State the theorem 
theorem best_fit_model : 
  R2_Model2 ≥ R2_Model1 ∧ 
  R2_Model2 ≥ R2_Model3 ∧ 
  R2_Model2 ≥ R2_Model4 :=
by
  sorry

end best_fit_model_l813_813981


namespace sum_of_prime_factors_of_2_to_10_minus_1_l813_813588

theorem sum_of_prime_factors_of_2_to_10_minus_1 :
  let n := 2^10 - 1,
      factors := [31, 3, 11] in
  (n = factors.prod) ∧ (factors.all Prime) → factors.sum = 45 :=
by
  let n := 2^10 - 1
  let factors := [31, 3, 11]
  have fact_prod : n = factors.prod := by sorry
  have all_prime : factors.all Prime := by sorry
  have sum_factors : factors.sum = 45 := by sorry
  exact ⟨fact_prod, all_prime, sum_factors⟩

end sum_of_prime_factors_of_2_to_10_minus_1_l813_813588


namespace monthly_installments_l813_813557

theorem monthly_installments (cash_price deposit installment saving : ℕ) (total_paid installments_made : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment = 300 →
  saving = 4000 →
  total_paid = cash_price + saving →
  installments_made = (total_paid - deposit) / installment →
  installments_made = 30 :=
by
  intros h_cash_price h_deposit h_installment h_saving h_total_paid h_installments_made
  sorry

end monthly_installments_l813_813557


namespace total_batteries_correct_l813_813253

-- Definitions of the number of batteries used in each category
def batteries_flashlight : ℕ := 2
def batteries_toys : ℕ := 15
def batteries_controllers : ℕ := 2

-- The total number of batteries used by Tom
def total_batteries : ℕ := batteries_flashlight + batteries_toys + batteries_controllers

-- The proof statement that needs to be proven
theorem total_batteries_correct : total_batteries = 19 := by
  sorry

end total_batteries_correct_l813_813253


namespace problem_statement_l813_813510

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2 - Real.pi / 3)

theorem problem_statement :
  (∀ k : ℤ, x ≠ 2 * k * Real.pi + 5 * Real.pi / 3) ∧
  (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧
  (∀ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 3) → f x > f (x + ε) ∀ ε : ℝ, ε > 0) ∧
  (∀ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x ≤ 2 * k * Real.pi + 4 * Real.pi / 3) ↔ f x ≤ √3) :=
sorry

end problem_statement_l813_813510


namespace midpoint_distances_are_correct_l813_813596

-- Define the conditions
def rectangle_side1 : ℝ := 24
def rectangle_side2 : ℝ := 7

-- Radius of the circle (rectangle is inscribed)
def radius : ℝ := (real.sqrt (rectangle_side1^2 + rectangle_side2^2)) / 2

-- Midpoint distances from the larger arcs to the vertices
def larger_arc_distance1 : ℝ := 15
def larger_arc_distance2 : ℝ := 20

-- The proof problem statement
theorem midpoint_distances_are_correct :
  ∃ (M A B : ℝ), 
    radius = 12.5 ∧
    M = larger_arc_distance1 ∧
    A = 15 ∧
    B = 20 :=
sorry

end midpoint_distances_are_correct_l813_813596


namespace max_side_length_l813_813771

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813771


namespace abs_a_b_l813_813386

def tau (n : ℕ) : ℕ :=
  if h : n > 0 then (finset.filter (λ i, n % i = 0) (finset.range (n+1))).card
  else 0

def S (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum tau

def number_of_odds (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ k, nat.sqrt k ^ 2 = k).card

theorem abs_a_b : |(finset.range 2006).filter (λ n, S n % 2 = 1).card - 
                 (finset.range 2006).filter (λ n, S n % 2 = 0).card| = 1 :=
by
  sorry

end abs_a_b_l813_813386


namespace max_side_of_triangle_l813_813750

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813750


namespace incorrect_statement_C_l813_813037

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem incorrect_statement_C (a b c : ℝ) (x0 : ℝ) (h_local_min : ∀ y, f x0 a b c ≤ f y a b c) :
  ∃ z, z < x0 ∧ ¬ (f z a b c ≤ f (z + ε) a b c) := sorry

end incorrect_statement_C_l813_813037


namespace problem1_probability_l813_813315

noncomputable def poisson_probability (λ k : ℝ) : ℝ := 
  (λ ^ k * Real.exp (-λ)) / (Nat.factorial k)

theorem problem1_probability :
  let n := 1000
  let p := 0.004
  let λ := n * p
  poisson_probability λ 5 = 0.1562 :=
by
  sorry

end problem1_probability_l813_813315


namespace graph_symmetric_about_point_minus_pi_six_zero_f_monotonically_increasing_on_interval_l813_813430

noncomputable def f : ℝ → ℝ := fun x => √3 * cos (π/2 - x) + sin (π/2 + x)

theorem graph_symmetric_about_point_minus_pi_six_zero :
  ∀ x: ℝ, f (-π/6 - x) = - f (π/6 + x) := sorry

theorem f_monotonically_increasing_on_interval :
  ∀ x y: ℝ, x ∈ set.Icc (-2 * π / 3) 0 → y ∈ set.Icc (-2 * π / 3) 0 → x < y → f x < f y := sorry

end graph_symmetric_about_point_minus_pi_six_zero_f_monotonically_increasing_on_interval_l813_813430


namespace find_max_side_length_l813_813790

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813790


namespace find_k_values_for_intersection_l813_813117

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l813_813117


namespace sum_of_first_100_terms_l813_813399

variable (a : ℕ → ℤ)
def a_n (n : ℕ) : ℤ := (-1 : ℤ)^(n-1) * (4 * n - 3)

theorem sum_of_first_100_terms : 
  ∑ i in Finset.range 100, a_n (i + 1) = -200 
:= by sorry

end sum_of_first_100_terms_l813_813399


namespace days_with_equal_sun_tue_l813_813302

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l813_813302


namespace find_max_side_length_l813_813784

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813784


namespace frogs_moving_l813_813648

theorem frogs_moving (initial_frogs tadpoles mature_frogs pond_capacity frogs_to_move : ℕ)
  (h1 : initial_frogs = 5)
  (h2 : tadpoles = 3 * initial_frogs)
  (h3 : mature_frogs = (2 * tadpoles) / 3)
  (h4 : pond_capacity = 8)
  (h5 : frogs_to_move = (initial_frogs + mature_frogs) - pond_capacity) :
  frogs_to_move = 7 :=
by {
  sorry
}

end frogs_moving_l813_813648


namespace evaluate_g_neg_five_l813_813509

def g (x : ℝ) : ℝ :=
if x < 0 then 3 * x - 4 else 7 - 3 * x

theorem evaluate_g_neg_five : g (-5) = -19 :=
by
  simp [g]
  sorry

end evaluate_g_neg_five_l813_813509


namespace function_is_even_l813_813017

variables {R : Type*} [LinearOrderedField R] -- Real numbers domain
variables {V : Type*} [NormedAddCommGroup V] [InnerProductSpace R V] 

def is_even_function {f : R → R} : Prop :=
  ∀ x, f(-x) = f(x)

noncomputable def f (a b : V) (x : R) : R :=
  (inner a a) * x^2 + (inner b b)

theorem function_is_even
  (a b : V)
  (h_a_nonzero : a ≠ 0)
  (h_b_nonzero : b ≠ 0)
  (h_orthogonal : inner a b = 0) :
  is_even_function (f a b) :=
by
  sorry

end function_is_even_l813_813017


namespace smallest_norm_l813_813145

noncomputable def vectorNorm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_norm (v : ℝ × ℝ)
  (h : vectorNorm (v.1 + 4, v.2 + 2) = 10) :
  vectorNorm v >= 10 - 2 * Real.sqrt 5 :=
by
  sorry

end smallest_norm_l813_813145


namespace max_side_length_is_11_l813_813677

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813677


namespace sum_arithmetic_sequence_mod_10_l813_813349

noncomputable def arithmetic_sequence_sum_mod_10 (a d last : ℕ) : ℕ :=
let n := (last - a) / d + 1 in
(n * (2 * a + (n - 1) * d)) / 2 % 10

theorem sum_arithmetic_sequence_mod_10 :
  arithmetic_sequence_sum_mod_10 7 7 91 = 7 := 
sorry

end sum_arithmetic_sequence_mod_10_l813_813349


namespace middle_elementary_students_l813_813971

theorem middle_elementary_students (S S_PS S_MS S_MR : ℕ) 
  (h1 : S = 12000)
  (h2 : S_PS = (15 * S) / 16)
  (h3 : S_MS = S - S_PS)
  (h4 : S_MR + S_MS = (S_PS) / 2) : 
  S_MR = 4875 :=
by
  sorry

end middle_elementary_students_l813_813971


namespace tournament_six_points_l813_813473

theorem tournament_six_points (n : ℕ) (hn : n = 9) : 
  let participants := 2^n in
  let points := 6 in
  participants = 512 → 
  (n = 9) →
  ∃ finishers_with_six_points, finishers_with_six_points = 84 := 
by
  intros participants_eq hn_eq
  have num_participants : 2^9 = 512 := by norm_num
  rw [←participants_eq] at num_participants
  exact ⟨84, rfl⟩

end tournament_six_points_l813_813473


namespace slope_angle_of_line_l813_813406

theorem slope_angle_of_line (M N : ℝ × ℝ) (hM : M = (1, 2)) (hN : N = (0, 1)) : 
  ∃ α : ℝ, tan α = (M.2 - N.2) / (M.1 - N.1) ∧ α = Real.pi / 4 :=
by
  --  We prove the existence of such an α
  use Real.pi / 4
  split
  -- Proving tan(Real.pi / 4) = 1 which is the slope
  { have slope := (2 - 1) / (1 - 0)
    simp [slope, tan_pi_div_four]
  }
  -- Proving the angle is indeed pi / 4
  { simp }

end slope_angle_of_line_l813_813406


namespace graph_passes_fixed_point_l813_813568

theorem graph_passes_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 1 ∧ y = 4 ∧ f x = y
where f : ℝ → ℝ := λ x, 3 + a^(x-1) :=
begin
  sorry
end

end graph_passes_fixed_point_l813_813568


namespace perpendicular_vectors_l813_813965

def vector_a := (2, 0 : ℤ × ℤ)
def vector_b := (1, 1 : ℤ × ℤ)

theorem perpendicular_vectors:
  let v := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) in
  v.1 * vector_b.1 + v.2 * vector_b.2 = 0 :=
by
  sorry

end perpendicular_vectors_l813_813965


namespace nina_weeks_to_afford_game_l813_813520

noncomputable def game_cost : ℝ := 50
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def weekly_allowance : ℝ := 10
noncomputable def saving_rate : ℝ := 0.5

noncomputable def total_cost : ℝ := game_cost + (game_cost * sales_tax_rate)
noncomputable def savings_per_week : ℝ := weekly_allowance * saving_rate
noncomputable def weeks_needed : ℝ := total_cost / savings_per_week

theorem nina_weeks_to_afford_game : weeks_needed = 11 := by
  sorry

end nina_weeks_to_afford_game_l813_813520


namespace algebraic_expression_value_l813_813424

theorem algebraic_expression_value (x : ℝ) (h : 3 / (x^2 + x) - x^2 = 2 + x) :
  2 * x^2 + 2 * x = 2 :=
sorry

end algebraic_expression_value_l813_813424


namespace problem_solution_l813_813338

noncomputable def problem_expr : ℝ :=
  - (1 ^ 2023) + abs (sqrt 3 - 2) - 3 * tan (Real.pi / 3)

theorem problem_solution : problem_expr = 1 - 4 * sqrt 3 := 
by
  sorry

end problem_solution_l813_813338


namespace max_side_length_l813_813776

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813776


namespace rocket_coaster_total_cars_l813_813553

theorem rocket_coaster_total_cars (C_4 C_6 : ℕ) (h1 : C_4 = 9) (h2 : 4 * C_4 + 6 * C_6 = 72) :
  C_4 + C_6 = 15 :=
sorry

end rocket_coaster_total_cars_l813_813553


namespace problem_statement_l813_813121

noncomputable def angleBAC (ABC ACB : ℕ) : ℕ := 180 - ABC - ACB
noncomputable def angleADE (x : ℕ) : ℕ := 180 - x
noncomputable def angleAED (angleADE angleBAC : ℕ) : ℕ := angleADE + x - 60
noncomputable def angleDEB (x : ℕ) : ℕ := 180 - (x - 60)

theorem problem_statement :
  (angleBAC 50 70 = 60) →
  (angleADE 180 = 180 - x) →
  (angleAED angleADE angleBAC = x - 60) →
  (angleDEB x = 120) →
  x = 180 :=
sorry

end problem_statement_l813_813121


namespace value_range_quadratic_l813_813595

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 2 * x

theorem value_range_quadratic :
  ∃ a b, a ≤ b ∧ (∀ x ∈ set.Icc 0 3, a ≤ quadratic_function x ∧ quadratic_function x ≤ b) ∧ a = -1 ∧ b = 3 :=
by
  sorry

end value_range_quadratic_l813_813595


namespace intersecting_circles_l813_813098

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l813_813098


namespace conj_of_z_l813_813031

theorem conj_of_z (z : ℂ) (h : (z / complex.I + 4 = 3 * complex.I)) :
  conj z = -3 + 4 * complex.I := by
  sorry

end conj_of_z_l813_813031


namespace equal_angles_trapezoid_l813_813994

theorem equal_angles_trapezoid
  (A B C D N M P : Type)
  [trapezoid : is_trapezoid A B C D]
  (AB_CD : base1 A B > base2 C D)
  (AC_BD : diagonal A C ∩ diagonal B D = {N})
  (AD_BC : line A D ∩ line B C = {M})
  (circ_ADN_BCN : (circumscribed (triangle A D N)) ∩ (circumscribed (triangle B C N)) = {P, N})
  (P_neq_N : P ≠ N) :
  angle A M P = angle B M N :=
by sorry

end equal_angles_trapezoid_l813_813994


namespace sum_of_prime_factors_l813_813585

theorem sum_of_prime_factors (x : ℕ) (h1 : x = 2^10 - 1) 
  (h2 : 2^10 - 1 = (2^5 + 1) * (2^5 - 1)) 
  (h3 : 2^5 - 1 = 31) 
  (h4 : 2^5 + 1 = 33) 
  (h5 : 33 = 3 * 11) : 
  (31 + 3 + 11 = 45) := 
  sorry

end sum_of_prime_factors_l813_813585


namespace central_angle_of_regular_hexagon_l813_813559

theorem central_angle_of_regular_hexagon:
  ∀ (α : ℝ), 
  (∃ n : ℕ, n = 6 ∧ n * α = 360) →
  α = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l813_813559


namespace domain_of_f_l813_813565

noncomputable def f (x : ℝ) := 2 ^ (Real.sqrt (3 - x)) + 1 / (x - 1)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y = f x) ↔ (x ≤ 3 ∧ x ≠ 1) :=
by
  sorry

end domain_of_f_l813_813565


namespace ratio_of_chips_l813_813491

def bags_total (d a : ℕ) : Prop := d + a = 3

theorem ratio_of_chips (d a : ℕ) (hd : d = 1) (ha : a = 2) (h : bags_total d a) :
  a / d = 2 :=
by
  rw [hd, ha]
  exact rfl

end ratio_of_chips_l813_813491


namespace Mrs_Hilt_pies_l813_813518

theorem Mrs_Hilt_pies :
  ∀ (P A M T : ℝ), P = 16.0 → A = 14.0 → M = 5.0 → T = M * (P + A) → T = 150.0 :=
by
  intros P A M T hP hA hM hT
  rw [hP, hA, hM] at hT
  exact hT

end Mrs_Hilt_pies_l813_813518


namespace ratio_of_volumes_total_surface_area_smaller_cube_l813_813264

-- Definitions using the conditions in (a)
def edge_length_smaller_cube := 4 -- in inches
def edge_length_larger_cube := 24 -- in inches (2 feet converted to inches)

-- Propositions based on the correct answers in (b)
theorem ratio_of_volumes : 
  (edge_length_smaller_cube ^ 3) / (edge_length_larger_cube ^ 3) = 1 / 216 := by
  sorry

theorem total_surface_area_smaller_cube : 
  6 * (edge_length_smaller_cube ^ 2) = 96 := by
  sorry

end ratio_of_volumes_total_surface_area_smaller_cube_l813_813264


namespace line_circle_intersect_equilateral_l813_813436

theorem line_circle_intersect_equilateral (a : ℝ) : 
    (∃ A B: ℝ × ℝ, (A.1 + a * A.2 + 3 = 0 ∧ A.1^2 + A.2^2 = 4) ∧ 
                    (B.1 + a * B.2 + 3 = 0 ∧ B.1^2 + B.2^2 = 4) ∧
                    (A ≠ B) ∧
                    (dist (0, 0) A = dist (0,0) B) ∧ 
                    (∠AOB = real.pi / 3)) → 
    (a = real.sqrt 2 ∨ a = -real.sqrt 2) :=
by sorry

end line_circle_intersect_equilateral_l813_813436


namespace max_geometries_l813_813932

variables {α β : set (set ℝ)}

/-- Given two parallel planes α and β, each containing distinct points, determine the 
   maximum number of lines, planes, and triangular pyramids. -/
theorem max_geometries
  (h_parallel : α ∩ β = ∅)
  (hα : α ≠ ∅)
  (hβ : β ≠ ∅)
  (points_α : ∀ p ∈ α, ∃! q ∈ α, q ≠ p) -- 4 points in α
  (points_β : ∀ p ∈ β, ∃! q ∈ β, q ≠ p) -- 5 points in β
  (h_noncoplanar : ∀ s : finset (sets ℝ), s.card = 4 → ¬∃ p, p ∈ α ∪ β ∧ p ∈ s)
  (h_noncollinear : ∀ s : finset (sets ℝ), s.card = 3 → ¬∃ p, p ∈ α ∪ β ∧ p ∈ s) :
  (nat.choose 9 2 = 36) ∧
  ((nat.choose 4 1) * (nat.choose 5 2) + (nat.choose 4 2) * (nat.choose 5 1) + 2 = 72) ∧
  ((nat.choose 4 3) * (nat.choose 5 1) + (nat.choose 4 2) * (nat.choose 5 2) + (nat.choose 4 1) * (nat.choose 5 3) = 120)
:= sorry

end max_geometries_l813_813932


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813692

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813692


namespace number_of_matches_among_three_players_l813_813974

-- Define the given conditions
variables (n r : ℕ) -- n is the number of participants, r is the number of matches among the 3 players
variables (m : ℕ := 50) -- m is the total number of matches played

-- Given assumptions
def condition1 := m = 50
def condition2 := ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)

-- The target proof
theorem number_of_matches_among_three_players (n r : ℕ) (m : ℕ := 50)
  (h1 : m = 50)
  (h2 : ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)) :
  r = 1 :=
sorry

end number_of_matches_among_three_players_l813_813974


namespace problem_equivalence_l813_813999

open Complex

theorem problem_equivalence
(n : ℕ) (h : 2 ≤ n) (a b : Fin n → ℂ) :
  (∀ z : ℂ, ∑ k, ∥z - a k∥^2 ≤ ∑ k, ∥z - b k∥^2) ↔ 
  (∑ k, a k = ∑ k, b k ∧ ∑ k, ∥a k∥^2 ≤ ∑ k, ∥b k∥^2) :=
sorry

end problem_equivalence_l813_813999


namespace solution_correct_l813_813506

noncomputable def f : ℝ → ℝ :=
λ x, if x < 2 then 2 / (2 - x) else 0

theorem solution_correct :
  (∀ x y : ℝ, f (x * f y) * f y = f (x + y)) ∧
  f 2 = 0 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) →
  ∀ x : ℝ, f x = (if x < 2 then 2 / (2 - x) else 0) :=
by {
  assume h,
  sorry -- Proof goes here.
}

end solution_correct_l813_813506


namespace find_incorrect_statement_l813_813076

def statement_A := ∀ (P Q : Prop), (P → Q) → (¬Q → ¬P)
def statement_B := ∀ (P : Prop), ((¬P) → false) → P
def statement_C := ∀ (shape : Type), (∃ s : shape, true) → false
def statement_D := ∀ (P : ℕ → Prop), P 0 → (∀ n, P n → P (n + 1)) → ∀ n, P n
def statement_E := ∀ {α : Type} (p : Prop), (¬p ∨ p)

theorem find_incorrect_statement : statement_C :=
sorry

end find_incorrect_statement_l813_813076


namespace more_boys_than_girls_l813_813246

theorem more_boys_than_girls : 
  let girls := 28.0
  let boys := 35.0
  boys - girls = 7.0 :=
by
  sorry

end more_boys_than_girls_l813_813246


namespace max_triangle_side_l813_813718

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813718


namespace max_side_of_triangle_l813_813742

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813742


namespace fisherman_daily_earnings_l813_813220

def red_snapper_quantity : Nat := 8
def tuna_quantity : Nat := 14
def red_snapper_cost : Nat := 3
def tuna_cost : Nat := 2

theorem fisherman_daily_earnings
  (rs_qty : Nat := red_snapper_quantity)
  (t_qty : Nat := tuna_quantity)
  (rs_cost : Nat := red_snapper_cost)
  (t_cost : Nat := tuna_cost) :
  rs_qty * rs_cost + t_qty * t_cost = 52 := 
by {
  sorry
}

end fisherman_daily_earnings_l813_813220


namespace max_side_length_l813_813773

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813773


namespace gross_income_increase_l813_813272

variables (X P : ℝ)

def gross_income_no_discount := X * P
def gross_income_discount := 1.25 * X * 0.9 * P
def percentage_increase := ((gross_income_discount X P - gross_income_no_discount X P) / (gross_income_no_discount X P)) * 100

theorem gross_income_increase : percentage_increase X P = 12.5 := by
  sorry

end gross_income_increase_l813_813272


namespace solve_for_A_l813_813570

theorem solve_for_A : ∃ (A : ℕ), A7 = 10 * A + 7 ∧ A7 + 30 = 77 ∧ A = 4 :=
by
  sorry

end solve_for_A_l813_813570


namespace smallest_k_divides_polynomial_l813_813888

noncomputable def is_divisible (f g : polynomial complex) : Prop :=
∃ q, f = q * g

theorem smallest_k_divides_polynomial:
  let f := (polynomial.C (1:complex) * polynomial.X ^ 12 +
            polynomial.C (1:complex) * polynomial.X ^ 11 +
            polynomial.C (1:complex) * polynomial.X ^ 8 +
            polynomial.C (1:complex) * polynomial.X ^ 7 +
            polynomial.C (1:complex) * polynomial.X ^ 6 +
            polynomial.C (1:complex) * polynomial.X ^ 3 +
            polynomial.C (1:complex)) in
  ∃ k : ℕ, k > 0 ∧ is_divisible (polynomial.X ^ k - 1) f ∧ k = 120 :=
begin
  intros f,
  use [120],
  split,
  { linarith, },
  split,
  { sorry, },
  { refl, },
end

end smallest_k_divides_polynomial_l813_813888


namespace find_max_side_length_l813_813794

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813794


namespace coefficients_values_l813_813370

noncomputable def find_coefficients (a b c : ℝ) : Prop :=
∀ (n : ℕ) (h : 0 < n), (n + 3)^2 = a * (n + 2)^2 + b * (n + 1)^2 + c * n^2

theorem coefficients_values :
  ∃ (a b c : ℝ), find_coefficients a b c ∧ a = 3 ∧ b = -3 ∧ c = 1 :=
begin
  use [3, -3, 1],
  split,
  { intros n hn,
    calc (n + 3)^2
        = n^2 + 6 * n + 9 : by ring
    ... = 3 * (n^2 + 4 * n + 4) - 3 * (n^2 + 2 * n + 1) + n^2 : by ring
    ... = 3 * (n + 2)^2 - 3 * (n + 1)^2 + n^2 : by ring },
  repeat { split; refl }
end

end coefficients_values_l813_813370


namespace solution_l813_813004

noncomputable def problem_statement : Prop :=
  ∀ (α β : ℝ), 
  (0 < α ∧ α < π) ∧ (0 < β ∧ β < π) ∧ 
  (Real.tan (α - β) = 0.5) ∧ (Real.tan β = -1 / 7) → 
  2 * α - β = -3 * π / 4

theorem solution : problem_statement :=
  by
    intros α β h_conditions,
    sorry

end solution_l813_813004


namespace prob_four_vertical_faces_same_color_l813_813867

noncomputable def painted_cube_probability : ℚ :=
  let total_arrangements := 3^6
  let suitable_arrangements := 3 + 18 + 6
  suitable_arrangements / total_arrangements

theorem prob_four_vertical_faces_same_color : 
  painted_cube_probability = 1 / 27 := by
  sorry

end prob_four_vertical_faces_same_color_l813_813867


namespace winnie_keeps_balloons_l813_813622

theorem winnie_keeps_balloons :
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  (totalBalloons % friends) = 8 := 
by 
  -- Definitions
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  -- Conclusion
  show totalBalloons % friends = 8
  sorry

end winnie_keeps_balloons_l813_813622


namespace max_side_of_triangle_l813_813745

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813745


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813688

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813688


namespace total_wasted_time_is_10_l813_813167

-- Define the time Martin spends waiting in traffic
def waiting_time : ℕ := 2

-- Define the constant for the multiplier
def multiplier : ℕ := 4

-- Define the time spent trying to get off the freeway
def off_freeway_time : ℕ := waiting_time * multiplier

-- Define the total wasted time
def total_wasted_time : ℕ := waiting_time + off_freeway_time

-- Theorem stating that the total time wasted is 10 hours
theorem total_wasted_time_is_10 : total_wasted_time = 10 :=
by
  sorry

end total_wasted_time_is_10_l813_813167


namespace ellipse_equation_range_reciprocal_distances_l813_813419

-- Definitions of the foci and the point M on the ellipse.
def F1 : ℝ × ℝ := (0, -Real.sqrt 3)
def F2 : ℝ × ℝ := (0, Real.sqrt 3)
def M : ℝ × ℝ := (Real.sqrt 3 / 2, 1)

-- The statement to prove the standard equation of the ellipse C.
theorem ellipse_equation :
  (∃ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b), a^2 = b^2 + (Real.sqrt 3)^2 ∧
    (M.1)^2 / b^2 + (M.2)^2 / a^2 = 1) →
  (a = 2 ∧ b = 1) →
  (∀ (x y : ℝ), (x^2 / b^2 + y^2 / a^2 = 1) ↔ (y^2 / 4 + x^2 = 1)) :=
sorry

-- The statement to prove the range of (1 / |PF1|) + (1 / |PF2|).
theorem range_reciprocal_distances :
  (∀ (P : ℝ × ℝ), (P.1^2 / 1 + P.2^2 / 4 = 1) →
     1 ≤ (1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
          1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) ∧ 
     (1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
      1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) ≤ 4) :=
sorry

end ellipse_equation_range_reciprocal_distances_l813_813419


namespace volume_new_parallelepiped_l813_813597

variables (a b c : ℝ^3)
variable (volume : ℝ)

-- Volume of the initial parallelepiped
-- a · (b × c) = ± 8
axiom volume_initial : abs (a ⋅ (b × c)) = 8

-- Vectors forming new parallelepiped
def v1 := 2 • a + b
def v2 := b + 2 • c
def v3 := c - 5 • a

-- Statement of the problem in Lean: prove the volume of the new parallelepiped is 16
theorem volume_new_parallelepiped : abs ((v1 a b) ⋅ (v2 b c × v3 a c)) = 16 :=
by
  sorry

end volume_new_parallelepiped_l813_813597


namespace lottery_result_probability_l813_813983

noncomputable def binom (n k : Nat) : ℚ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def lottery_probability : ℚ := 
  let total_valid_outcomes :=
    3 * binom 86 3
  let total_possible_outcomes := binom 90 5
  total_valid_outcomes / total_possible_outcomes

theorem lottery_result_probability : 
  lottery_probability ≈ 0.005876 := sorry

end lottery_result_probability_l813_813983


namespace total_valid_documents_after_reprinting_l813_813339

theorem total_valid_documents_after_reprinting :
  let total_documents := 4000
  let invalid_percent := 0.35
  let color_reprint_percent := 0.12
  let bw_reprint_percent := 0.23
  let invalid_documents := invalid_percent * total_documents
  let valid_documents_initially := total_documents - invalid_documents
  let color_reprints := color_reprint_percent * invalid_documents
  let bw_reprints := bw_reprint_percent * invalid_documents
  let total_reprints := color_reprints + bw_reprints
  let total_valid_documents_after_reprinting := valid_documents_initially + total_reprints
  total_valid_documents_after_reprinting = 3090 := by
  let total_documents := 4000
  let invalid_percent := 0.35
  let color_reprint_percent := 0.12
  let bw_reprint_percent := 0.23
  let invalid_documents := invalid_percent * total_documents
  let valid_documents_initially := total_documents - invalid_documents
  let color_reprints := color_reprint_percent * invalid_documents
  let bw_reprints := bw_reprint_percent * invalid_documents
  let total_reprints := color_reprints + bw_reprints
  let total_valid_documents_after_reprinting := valid_documents_initially + total_reprints
  have h1 : invalid_documents = 1400 := by sorry
  have h2 : valid_documents_initially = 2600 := by sorry
  have h3 : color_reprints = 168 := by sorry
  have h4 : bw_reprints = 322 := by sorry
  have h5 : total_reprints = 490 := by sorry
  have h6 : total_valid_documents_after_reprinting = 3090 := by sorry
  exact h6

end total_valid_documents_after_reprinting_l813_813339


namespace chris_newspapers_l813_813846

theorem chris_newspapers (C L : ℕ) 
  (h1 : L = C + 23) 
  (h2 : C + L = 65) : 
  C = 21 := 
by 
  sorry

end chris_newspapers_l813_813846


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813690

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813690


namespace no_extension_to_countably_additive_l813_813535

-- Define the base set Omega and the algebra A
def Omega : Set ℚ := Set.univ

def A : Set (Set ℚ) := 
  {s | ∃ (U : Finset (ℚ × ℚ)), s = ⋃ (p ∈ U), Set.Ioc p.1 p.2 }

-- Define the finitely additive measure mu
def mu (s : Set ℚ) : ℝ :=
  if h : s ∈ A then ∑ p in (Finset.filter (λ (x : ℚ × ℚ), Set.Ioc x.1 x.2 ⊆ s) (Finset.univ : Finset (ℚ × ℚ))), (p.2 - p.1) else 0

-- The main statement asserting that mu cannot be extended to a countably additive measure on sigma(A)
theorem no_extension_to_countably_additive :
  ¬ ∃ (mu' : Measure ℚ), 
    (∀ s ∈ A, mu' s = mu s) ∧ 
    (∀ (s : Set (Set ℚ)), (countable s ∧ s ⊆ A) → mu' (⋃₀ s) = ∑' (t : Set ℚ) in s, mu' t) := 
begin
  sorry
end

end no_extension_to_countably_additive_l813_813535


namespace zero_integers_in_range_such_that_expr_is_perfect_square_l813_813056

theorem zero_integers_in_range_such_that_expr_is_perfect_square :
  (∃ n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n ^ 2 + n + 2 = m ^ 2) → False :=
by sorry

end zero_integers_in_range_such_that_expr_is_perfect_square_l813_813056


namespace max_triangle_side_24_l813_813736

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813736


namespace p_two_pow_k_eq_k_plus_one_p_two_pow_k_plus_one_eq_2k_plus_2_p_divides_p_two_l813_813850

-- Define the sequence b recursively according to the given condition
def b_sequence (n : ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, b (i + 1) = if b i ≤ n then 2 * b i - 1 else 2 * b i - 2 * n

-- Define the period function p(b_0, n)
def p (b₀ n : ℕ) : ℕ :=
  Inf {p | p > 0 ∧ ∃ (b : ℕ → ℕ), b_sequence n b ∧ b 0 = b₀ ∧ b p = b₀}

-- Part (1) Proofs
theorem p_two_pow_k_eq_k_plus_one (k : ℕ) : p 2 (2^k) = k + 1 :=
  sorry

theorem p_two_pow_k_plus_one_eq_2k_plus_2 (k : ℕ) : p 2 (2^k + 1) = 2 * (k + 1) :=
  sorry

-- Part (2) Proof
theorem p_divides_p_two (n b₀ : ℕ) : p b₀ n ∣ p 2 n :=
  sorry

end p_two_pow_k_eq_k_plus_one_p_two_pow_k_plus_one_eq_2k_plus_2_p_divides_p_two_l813_813850


namespace equal_sundays_tuesdays_l813_813307

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l813_813307


namespace min_value_proof_l813_813144

noncomputable def min_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2

theorem min_value_proof :
  ∃ α β : ℝ, min_value_expression α β = 48 := by
  sorry

end min_value_proof_l813_813144


namespace intersecting_circles_unique_point_l813_813105

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l813_813105


namespace set_difference_P_M_l813_813443

open Set

noncomputable def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2009}
noncomputable def P : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2010}

theorem set_difference_P_M : P \ M = {2010} :=
by
  sorry

end set_difference_P_M_l813_813443


namespace ratio_of_sides_in_acute_triangle_l813_813976

theorem ratio_of_sides_in_acute_triangle (ABC : Triangle) (h_acute : ABC.isAcute) (angle_B : ABC.angle B = 60) :
  let m := ABC.longest_side / ABC.shortest_side in 1 ≤ m ∧ m < 2 :=
sorry

end ratio_of_sides_in_acute_triangle_l813_813976


namespace max_triangle_side_24_l813_813728

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813728


namespace trigonometric_identity_l813_813949

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 :=
  sorry

end trigonometric_identity_l813_813949


namespace intersection_distance_l813_813923

open Real

-- Definition of the curve C in standard coordinates
def curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Definition of the line l in parametric form
def line_l (x y t : ℝ) : Prop :=
  x = 1 + t ∧ y = -1 + t

-- The length of the intersection points A and B of curve C and line l
theorem intersection_distance : ∃ t1 t2 : ℝ, (curve_C (1 + t1) (-1 + t1) ∧ curve_C (1 + t2) (-1 + t2)) ∧ (abs (t1 - t2) = 4 * sqrt 6) :=
sorry

end intersection_distance_l813_813923


namespace number_of_students_above_120_l813_813074

def math_scores_distribution : ℝ → ℝ := sorry  -- Define the distribution function according to N(110, 10^2)

def students_above_120 (n : ℕ) : ℕ :=
  let total_students : ℕ := 50 in
  sorry -- Implement the calculation logic here

theorem number_of_students_above_120 :
  let prob : ℝ := 0.34 in
  let total_students : ℕ := 50 in
  students_above_120 total_students = 8 :=
sorry

end number_of_students_above_120_l813_813074


namespace max_triangle_side_24_l813_813729

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813729


namespace Jina_mascots_total_l813_813489

theorem Jina_mascots_total :
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  teddies + bunnies + koala + additional_teddies = 51 :=
by
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  show teddies + bunnies + koala + additional_teddies = 51
  sorry

end Jina_mascots_total_l813_813489


namespace intersecting_circles_unique_point_l813_813102

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l813_813102


namespace sandwich_not_condiment_percentage_l813_813660

theorem sandwich_not_condiment_percentage :
  (total_weight : ℝ) → (condiment_weight : ℝ) →
  total_weight = 150 → condiment_weight = 45 →
  ((total_weight - condiment_weight) / total_weight) * 100 = 70 :=
by
  intros total_weight condiment_weight h_total h_condiment
  sorry

end sandwich_not_condiment_percentage_l813_813660


namespace mia_eggs_per_hour_l813_813516

/-- Define the number of eggs Mia can decorate per hour as a variable of interest -/
def mia_rate : ℕ := 24

/-- Define the rate at which Billy can decorate eggs -/
def billy_rate : ℕ := 10

/-- Define the total number of eggs to be decorated -/
def total_eggs : ℕ := 170

/-- Define the total time in hours they will work together -/
def total_time : ℕ := 5

/-- Define the combined rate of Mia and Billy decorating eggs -/
def combined_rate : ℕ := total_eggs / total_time

/-- Prove that Mia’s decorating rate is 24 eggs per hour given the conditions -/
theorem mia_eggs_per_hour : mia_rate + billy_rate = combined_rate := by
  /-- Calculate the combined rate from total_eggs and total_time -/
  have h : combined_rate = 34 := rfl

  /-- Substitute the defined values for mia_rate, billy_rate, and combined_rate into the equation -/
  show 24 + 10 = 34

  /-- Final step (proof not required) -/
  rfl

end mia_eggs_per_hour_l813_813516


namespace auditorium_rows_l813_813456

noncomputable def rows_in_auditorium : Nat :=
  let class1 := 30
  let class2 := 26
  let condition1 := ∃ row : Nat, row < class1 ∧ ∀ students_per_row : Nat, students_per_row ≤ row 
  let condition2 := ∃ empty_rows : Nat, empty_rows ≥ 3 ∧ ∀ students : Nat, students = class2 - empty_rows
  29

theorem auditorium_rows (n : Nat) (class1 : Nat) (class2 : Nat) (c1 : class1 ≥ n) (c2 : class2 ≤ n - 3)
  : n = 29 :=
by
  sorry

end auditorium_rows_l813_813456


namespace no_such_N_exists_l813_813494

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ n = a ^ b

noncomputable def p : ℕ → ℕ
| 1       := 1
| (n + 1) := if is_perfect_power (n + 1) then (n + 1) * p n else p n

theorem no_such_N_exists :
  ¬ ∃ N : ℕ, ∀ n : ℕ, n > N → p n > 2^n :=
sorry

end no_such_N_exists_l813_813494


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813816

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813816


namespace fisherman_daily_earnings_l813_813221

def red_snapper_quantity : Nat := 8
def tuna_quantity : Nat := 14
def red_snapper_cost : Nat := 3
def tuna_cost : Nat := 2

theorem fisherman_daily_earnings
  (rs_qty : Nat := red_snapper_quantity)
  (t_qty : Nat := tuna_quantity)
  (rs_cost : Nat := red_snapper_cost)
  (t_cost : Nat := tuna_cost) :
  rs_qty * rs_cost + t_qty * t_cost = 52 := 
by {
  sorry
}

end fisherman_daily_earnings_l813_813221


namespace binomial_sum_non_constant_terms_excluding_constant_term_l813_813872

theorem binomial_sum_non_constant_terms_excluding_constant_term : 
  let T (x : ℝ) := (2 / (real.sqrt x) - x) ^ 9 in
  (eval (1 : ℝ) T) - (-5376) = 5377 :=
by
  -- Using the binomial theorem to find the general term and the constant term.
  let T (x : ℝ) := (2 / (real.sqrt x) - x) ^ 9
  let S := (eval (1 : ℝ) T) -- sum of coefficients when x=1
  let C := (-5376) -- constant term
  have h : S - C = 1 - (-5376) := by sorry
  exact h

end binomial_sum_non_constant_terms_excluding_constant_term_l813_813872


namespace probability_red_and_purple_l813_813624

-- Define the basic elements: a list of flower colors
def flower_colors : List String := ["red", "yellow", "white", "purple"]

-- Define all combinations of choosing 2 flowers out of these 4 colors
def combinations : List (String × String) :=
  List.choosePairs flower_colors

-- Define the successful event: choosing both red and purple
def successful_event : String × String := ("red", "purple")
def successful_event_symmetric : String × String := ("purple", "red")

-- Define the probability of the successful event occurring
theorem probability_red_and_purple :
  let total_combinations := List.length combinations in
  let successful_combinations := List.filter (λ x => x = successful_event || x = successful_event_symmetric) combinations in
  ∃ (P : ℚ), 
    P = rat.ofInt (List.length successful_combinations) / total_combinations ∧
    P = 1 / 6 :=
by
  sorry

end probability_red_and_purple_l813_813624


namespace vector_difference_perpendicular_l813_813966

/-- Proof that the vector difference a - b is perpendicular to b given specific vectors a and b -/
theorem vector_difference_perpendicular {a b : ℝ × ℝ} (h_a : a = (2, 0)) (h_b : b = (1, 1)) :
  (a - b) • b = 0 :=
by
  sorry

end vector_difference_perpendicular_l813_813966


namespace find_max_side_length_l813_813789

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813789


namespace vectors_properties_l813_813019

noncomputable def a : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def b : ℝ × ℝ × ℝ := (-1, 0, 2)

theorem vectors_properties : 
  (a.1 + b.1, a.2 + b.2, a.3 + b.3) = (0, 1, 3) ∧ 
  Real.sqrt (a.1^2 + a.2^2 + a.3^2) = Real.sqrt 3 := 
by
  sorry

end vectors_properties_l813_813019


namespace max_side_length_of_triangle_l813_813764

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813764


namespace intersection_of_A_and_B_l813_813908

open Set

theorem intersection_of_A_and_B (A B : Set ℕ) (hA : A = {1, 2, 4}) (hB : B = {2, 4, 6}) : A ∩ B = {2, 4} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  sorry

end intersection_of_A_and_B_l813_813908


namespace pizza_slices_left_for_Phill_l813_813185

theorem pizza_slices_left_for_Phill :
  ∀ (initial_slices : ℕ) (first_cut : ℕ) (second_cut : ℕ) (third_cut : ℕ)
    (slices_given_to_3_friends : ℕ) (slices_given_to_2_friends : ℕ) (slices_left_for_Phill : ℕ),
    initial_slices = 1 →
    first_cut = 2 →
    second_cut = 4 →
    third_cut = 8 →
    slices_given_to_3_friends = 3 →
    slices_given_to_2_friends = 4 →
    slices_left_for_Phill = third_cut - (slices_given_to_3_friends + slices_given_to_2_friends) →
    slices_left_for_Phill = 1 :=
by {
  intros,
  subst_vars,
  simp, -- Simplify the boolean equalities
  -- We assume the steps are correct, so we leave it with sorry for now
  -- The proof should be easy for the given example and conditions.
  sorry,
}

end pizza_slices_left_for_Phill_l813_813185


namespace problem1_problem2_problem3_l813_813915

-- Define the given conditions for the ellipse and hyperbola
def ellipseC1 : Set (ℝ × ℝ) := { p | (p.1 ^ 2) / 4 + (p.2 ^ 2) / 2 = 1 }
def hyperbolaC2 : Set (ℝ × ℝ) := { p | (p.1 ^ 2) / 2 - p.2 ^ 2 = 1 }

-- Define points A and B on the ellipse C1
def A := (-√2 : ℝ, 1 : ℝ)
def B := (√2 : ℝ, -1 : ℝ)

-- Define the given conditions for the dot product
def AQ (Q : ℝ × ℝ) := (Q.fst + √2, Q.snd - 1)
def AP (P : ℝ × ℝ) := (P.fst + √2, P.snd - 1)
def BQ (Q : ℝ × ℝ) := (Q.fst - √2, Q.snd + 1)
def BP (P : ℝ × ℝ) := (P.fst - √2, P.snd + 1)

-- Problem 1: Prove the equation of the ellipse C1
theorem problem1 : ∀ p ∈ ellipseC1, (p.1 ^ 2) / 4 + (p.2 ^ 2) / 2 = 1 := sorry

-- Problem 2: Prove the trajectory equation of point Q
theorem problem2 (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P ∈ ellipseC1) 
  (hAQ : AQ Q ⬝ AP P = 0) 
  (hBQ : BQ Q ⬝ BP P = 0) : 2 * (Q.1 ^ 2) + Q.2 ^ 2 = 5 := sorry

-- Problem 3: Prove the maximum area of ΔABQ
theorem problem3 (Q : ℝ × ℝ)
  (Q₁ : Q.1) (Q₂ : Q.2)
  (h1 : 2 * Q.1 ^ 2 + Q.2 ^ 2 = 5) 
  (h2 : ¬ collinear ℝ (Set.insert A (Set.insert B {Q}))) : 
  area_triangle A B Q = 5 * √2 / 2 := sorry

end problem1_problem2_problem3_l813_813915


namespace value_of_a_f_is_odd_f_increasing_on_interval_l813_813918

def f (x a : ℝ) : ℝ := x + a / x

-- 1. Prove that a such that f(1) = 10 is 9.
theorem value_of_a (a : ℝ) (h : f 1 a = 10) : a = 9 := sorry

-- 2. Prove that f(x) = x + 9 / x is odd.
theorem f_is_odd : ∀ x : ℝ, f (-x) 9 = - (f x 9) := sorry

-- 3. Prove that f(x) = x + 9 / x is increasing on (3, +∞).
theorem f_increasing_on_interval : ∀ x1 x2 : ℝ, 3 < x1 → x1 < x2 → x2 → (f x1 9 < f x2 9) := sorry

end value_of_a_f_is_odd_f_increasing_on_interval_l813_813918


namespace parametric_line_option_d_l813_813377

-- Define the parametric equations as options
def option_a (t : ℝ) : ℝ × ℝ := (sqrt t, 2 * sqrt t)
def option_b (t : ℝ) : ℝ × ℝ := (2 * t + 1, 4 * t + 1)
def option_c (θ : ℝ) : ℝ × ℝ := (cos θ, 2 * sin θ)
def option_d (θ : ℝ) : ℝ × ℝ := (tan θ, 2 * tan θ)

-- Define the line equation y = 2x
def line_eq (x y : ℝ) : Prop := y = 2 * x

-- Theorem to prove option D represents y = 2x
theorem parametric_line_option_d (θ : ℝ) :
  ∃ (x y : ℝ), option_d θ = (x, y) ∧ line_eq x y :=
sorry

end parametric_line_option_d_l813_813377


namespace max_triangle_side_l813_813725

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813725


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813823

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813823


namespace numberOfBookshelves_l813_813335

-- Define the conditions as hypotheses
def numBooks : ℕ := 23
def numMagazines : ℕ := 61
def totalItems : ℕ := 2436

-- Define the number of items per bookshelf
def itemsPerBookshelf : ℕ := numBooks + numMagazines

-- State the theorem to be proven
theorem numberOfBookshelves (bookshelves : ℕ) :
  itemsPerBookshelf * bookshelves = totalItems → 
  bookshelves = 29 :=
by
  -- placeholder for proof
  sorry

end numberOfBookshelves_l813_813335


namespace part_one_part_two_l813_813842

-- First part of the problem
theorem part_one : 
  ( (1/3)⁻¹ - log 2 8 + (0.5⁻² - 2) * (27/8)^(2/3) ) = 9/2 := 
sorry

-- Second part of the problem with the given condition
theorem part_two (α : ℝ) (h : tan α = -2) : 
  ( sin (π + α) + 2 * sin (π / 2 - α) ) / ( sin (-α) + cos (π - α) ) = 4 := 
sorry

end part_one_part_two_l813_813842


namespace max_triangle_side_24_l813_813733

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813733


namespace max_triangle_side_l813_813716

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813716


namespace cone_height_l813_813643

theorem cone_height (V : ℝ) (vertex_angle : ℝ) (base_angle : ℝ) :
  V = 16384 * Real.pi →
  vertex_angle = 90 →
  base_angle = 45 →
  exists (h : ℝ), h = 36.8 :=
by
  intros hvolume hvertex hbase
  sorry

end cone_height_l813_813643


namespace rectangular_prism_diagonal_length_l813_813316

noncomputable def diagonal_length (a b c : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_prism_diagonal_length :
  diagonal_length 15 25 12 = real.sqrt 994 :=
by
  sorry

end rectangular_prism_diagonal_length_l813_813316


namespace vector_parallel_implies_x_4_l813_813935

theorem vector_parallel_implies_x_4 (x : ℝ) : 
  let a := (2, x)
  let b := (1, 2)
  (∃ k : ℝ, a = (k * fst b, k * snd b)) → x = 4 :=
by
  sorry

end vector_parallel_implies_x_4_l813_813935


namespace pacific_ocean_area_rewrite_l813_813228

theorem pacific_ocean_area_rewrite :
  ∀ (area : ℕ), area = 17996800 → 
  (let ten_thousand_unit := (area : ℝ) / 10000 in 
   let rounded_area := if (ten_thousand_unit - ten_thousand_unit.floor) * 10000 >= 5000 
                       then ten_thousand_unit.floor.toNat + 1 
                       else ten_thousand_unit.floor.toNat in
   rounded_area = 1800) := 
begin
  intros area h,
  simp [h],
  sorry
end

end pacific_ocean_area_rewrite_l813_813228


namespace rectangle_area_increase_l813_813628

variable (L B : ℝ)

theorem rectangle_area_increase :
  let L_new := 1.30 * L
  let B_new := 1.45 * B
  let A_original := L * B
  let A_new := L_new * B_new
  let A_increase := A_new - A_original
  let percentage_increase := (A_increase / A_original) * 100
  percentage_increase = 88.5 := by
    sorry

end rectangle_area_increase_l813_813628


namespace max_side_length_l813_813708

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813708


namespace mixed_ratio_depends_on_volumes_l813_813609

variables {V1 V2 : ℝ} {p q : ℝ}

def volume_alcohol_jar1 (V1 p : ℝ) := (2 * p / (2 * p + 3)) * V1
def volume_water_jar1 (V1 p : ℝ) := (3 / (2 * p + 3)) * V1
def volume_alcohol_jar2 (V2 q : ℝ) := (3 * q / (3 * q + 2)) * V2
def volume_water_jar2 (V2 q : ℝ) := (2 / (3 * q + 2)) * V2

def total_volume_alcohol (V1 V2 p q : ℝ) : ℝ := 
  volume_alcohol_jar1 V1 p + volume_alcohol_jar2 V2 q

def total_volume_water (V1 V2 p q : ℝ) : ℝ := 
  volume_water_jar1 V1 p + volume_water_jar2 V2 q

theorem mixed_ratio_depends_on_volumes 
  (V1 V2 p q : ℝ) :
  (total_volume_alcohol V1 V2 p q) /
  (total_volume_water V1 V2 p q) = 
  (2 * p * V1 + 3 * q * V2) /
  (3 * V1 + 2 * V2) → 
  true := sorry

end mixed_ratio_depends_on_volumes_l813_813609


namespace max_side_length_is_11_l813_813682

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813682


namespace triangle_angle_and_area_l813_813969

theorem triangle_angle_and_area (a b c C : ℝ) 
  (h1 : tan (π / 4 - C) = sqrt 3 - 2) 
  (h2 : c = sqrt 7) 
  (h3 : a + b = 5) 
  (h4 : 0 < C ∧ C < π) 
  : C = π / 3 ∧ (1 / 2 * a * b * sin C = 3 * sqrt 3 / 2) := 
sorry

end triangle_angle_and_area_l813_813969


namespace three_digit_numbers_no_5s_8s_l813_813058

theorem three_digit_numbers_no_5s_8s : ∃ (n : ℕ), n = 7 * 8 * 8 ∧ n = 448 :=
by
    exists 7 * 8 * 8
    split
    . rfl
    . norm_num

end three_digit_numbers_no_5s_8s_l813_813058


namespace find_k_values_for_intersection_l813_813116

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l813_813116


namespace point_equidistant_x_axis_y_axis_line_l813_813314

theorem point_equidistant_x_axis_y_axis_line (x y : ℝ) (h1 : abs y = abs x) (h2 : abs (x + y - 2) / Real.sqrt 2 = abs x) :
  x = 1 :=
  sorry

end point_equidistant_x_axis_y_axis_line_l813_813314


namespace base6_sum_correct_l813_813948

theorem base6_sum_correct {S H E : ℕ} (hS : S < 6) (hH : H < 6) (hE : E < 6) 
  (dist : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (rightmost : (E + E) % 6 = S) 
  (second_rightmost : (H + H + if E + E < 6 then 0 else 1) % 6 = E) :
  S + H + E = 11 := 
by sorry

end base6_sum_correct_l813_813948


namespace number_of_routes_of_duration_10_minutes_l813_813211

def M : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := M n + M (n + 1)

theorem number_of_routes_of_duration_10_minutes : M 10 = 34 :=
by {
  -- Proof will go here
  sorry
}

end number_of_routes_of_duration_10_minutes_l813_813211


namespace carolyn_sum_l813_813549

theorem carolyn_sum (n : ℕ) (init_list : list ℕ) (C1 : n = 8)
  (C2 : init_list = [1, 2, 3, 4, 5, 6, 7, 8])
  (C3 : ∃ c1 ∈ init_list, c1 = 3):
  (sum_of_Carolyn (n : ℕ) (init_list : list ℕ) = 9) := by
sorry

end carolyn_sum_l813_813549


namespace largest_number_l813_813331

-- Definitions for the options
def option_A := Real.sqrt 4
def option_B := Real.pi
def option_C := 4
def option_D := (Real.sqrt 5) ^ 2

theorem largest_number :
  option_D > option_A ∧ option_D > option_B ∧ option_D > option_C := by
  -- Proof omitted
  sorry

end largest_number_l813_813331


namespace assign_roles_l813_813047

-- Define the members of the group
inductive Person
| Alice | Bob | Carol

-- Define the roles within the group
inductive Role
| President | Secretary | Treasurer

-- Define the problem
theorem assign_roles :
  ∃ (f : Role → Person), function.bijective f := 
begin
  let people := [Person.Alice, Person.Bob, Person.Carol],
  let roles := [Role.President, Role.Secretary, Role.Treasurer],
  have perm_count: ↥(set.univ : set (finset.perm (fin 3))) = 6, by sorry,
  let f := λ r, people[roles.index_of r],
  exact ⟨f, perm_count⟩,
end

end assign_roles_l813_813047


namespace product_of_valid_c_l813_813378

theorem product_of_valid_c : 
  let valid_c (c : ℕ) := 10 * c < 625
  (∏ c in Finset.filter valid_c (Finset.range 16), c) = 1307674368000 := by
  sorry

end product_of_valid_c_l813_813378


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813814

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813814


namespace heartbeats_in_20_mile_race_l813_813833

-- Define the constants and initial conditions
def heart_rate_constant : ℕ := 160
def heart_rate_increase : ℕ := 5
def distance_total : ℕ := 20
def pace_minutes_per_mile : ℕ := 6

-- Define the total heartbeats function
def total_heartbeats (heart_rate_constant: ℕ) (heart_rate_increase: ℕ)
  (distance_total: ℕ) (pace_minutes_per_mile: ℕ) : ℕ :=
let first_10_miles := 10 * heart_rate_constant in
let additional_miles := λ n, heart_rate_constant + heart_rate_increase * (n - 10) in
let additional_mile_heartbeats := (list.range 10).map (λ n, additional_miles (n + 11)) in
let total_additional_mile_heartbeats := additional_mile_heartbeats.sum in
first_10_miles + total_additional_mile_heartbeats

-- Theorem statement to be proved (with the correct answer)
theorem heartbeats_in_20_mile_race :
  total_heartbeats heart_rate_constant heart_rate_increase distance_total pace_minutes_per_mile = 11475 :=
sorry

end heartbeats_in_20_mile_race_l813_813833


namespace incenter_eccentricity_ratio_l813_813413

-- Definitions related to ellipses, coordinates, and geometry
variables {a b c : ℝ} (e : ℝ)
variable {c_squared_eq : c ^ 2 = a ^ 2 - b ^ 2}
variable {e_eq : e = c / a}
variables (P : ℝ × ℝ) (F1 F2 T I : ℝ × ℝ)
variable {ellipse_cond : (P.1 ^ 2) / (a ^ 2) + (P.2 ^ 2) / (b ^ 2) = 1}
variable {not_vertex : P ≠ (a, 0) ∧ P ≠ (-a, 0)}
variable {foci_def : F1 = (-c, 0) ∧ F2 = (c, 0)}
variable {incenter_criterion : -- need condition here defining I with respect to triangle incenter properties}
variable {intersection_criterion : -- need condition here defining T with respect to intersection of PI and F1F2}

-- Define the statement of the problem
theorem incenter_eccentricity_ratio :
  \[ given 
   ellipse_cond ∧ 
   not_vertex ∧ 
   foci_def ∧ 
   incenter_criterion ∧ 
   intersection_criterion \]
  \[
    \frac(dist T I}{dist I P} = e
  \] := sorry

end incenter_eccentricity_ratio_l813_813413


namespace union_of_A_and_B_l813_813410

-- Define the sets A and B
def A := {x : ℝ | 0 < x ∧ x < 16}
def B := {y : ℝ | -1 < y ∧ y < 4}

-- Prove that A ∪ B = (-1, 16)
theorem union_of_A_and_B : A ∪ B = {z : ℝ | -1 < z ∧ z < 16} :=
by sorry

end union_of_A_and_B_l813_813410


namespace batsman_average_after_12th_innings_l813_813289

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (h1 : 75 = (A + 12)) 
  (h2 : 11 * A + 75 = 12 * (A + 1)) :
  (A + 1) = 64 :=
by 
  sorry

end batsman_average_after_12th_innings_l813_813289


namespace problem_proof_l813_813032

-- Definition of the ellipse with given a and b such that a > b > 0
def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.fst ^ 2 / a ^ 2 + p.snd ^ 2 / b ^ 2 = 1}

-- Given conditions related to points A and B, and their distance
def AB_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : Prop :=
  let c := sqrt (a ^ 2 - b ^ 2) in
  let A := (a, 0) in
  let B := (0, b) in
  dist A B = (sqrt 2 / 2) * (2 * c)

-- Eccentricity of ellipse
def eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : ℝ :=
  let c := sqrt (a ^ 2 - b ^ 2) in
  c / a

-- The equation of the ellipse with given condition
def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : Prop :=
  b = sqrt (a ^ 2 - (2 / 3) * a ^ 2) ∧
  ∃ a' b', a' = 3 * sqrt 3 ∧ b' = 3 ∧
  ∀ x y, x ^ 2 / 27 + y ^ 2 / 9 = 1

-- The equation of the line l passing through point P and satisfying other conditions
def line_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : Prop :=
  let P : ℝ × ℝ := (-2, 1) in
  ∃ M N : ℝ × ℝ, M ≠ N ∧
  (M.fst ^ 2 / 27 + M.snd ^ 2 / 9 = 1) ∧
  (N.fst ^ 2 / 27 + N.snd ^ 2 / 9 = 1) ∧
  (M.fst + N.fst = -4) ∧
  (M.snd + N.snd = 2) ∧
  ∃ l : ℝ × ℝ → Prop, ∀ x y, l (x, y) ↔ 2 * x - 3 * y + 7 = 0

-- Overall proposition to be proven
theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_dist : AB_condition a b ha hb hab) :
  eccentricity a b ha hb hab = sqrt 6 / 3 ∧
  ellipse_equation a b ha hb hab ∧
  line_equation a b ha hb hab :=
by
  sorry

end problem_proof_l813_813032


namespace minimum_black_squares_l813_813181

theorem minimum_black_squares : 
  ∃ (painted : Finset (Fin 4 × Fin 4)), 
  (∀ (p1 p2 p3 : Fin 4 × Fin 4), 
    (true, true) ∈ painted ∨ (true, true) ∈ painted ∨ (true, true) ∈ painted) ∧ 
  painted.card = 8 :=
sorry

end minimum_black_squares_l813_813181


namespace Jina_has_51_mascots_l813_813487

def teddies := 5
def bunnies := 3 * teddies
def koala_bear := 1
def additional_teddies := 2 * bunnies
def total_mascots := teddies + bunnies + koala_bear + additional_teddies

theorem Jina_has_51_mascots : total_mascots = 51 := by
  sorry

end Jina_has_51_mascots_l813_813487


namespace max_set_size_divisible_diff_l813_813401

theorem max_set_size_divisible_diff (S : Finset ℕ) (h1 : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → (5 ∣ (x - y) ∨ 25 ∣ (x - y))) : S.card ≤ 25 :=
sorry

end max_set_size_divisible_diff_l813_813401


namespace trig_identity_l813_813881

theorem trig_identity :
  (sin (20 * real.pi / 180) * cos (10 * real.pi / 180) + cos (160 * real.pi / 180) * cos (110 * real.pi / 180))
  / (sin (24 * real.pi / 180) * cos (6 * real.pi / 180) + cos (156 * real.pi / 180) * cos (106 * real.pi / 180))
  = 1 :=
by
  sorry

end trig_identity_l813_813881


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813812

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813812


namespace correct_propositions_l813_813148

-- Definitions for non-coinciding lines and planes
variables (a b : Line) (α β : Plane)

def prop1 (ha₁ : a ∥ b) (ha₂ : a ⟂ α) : b ⟂ α :=
sorry

def prop2 (ha₁ : a ⟂ b) (ha₂ : a ⟂ α) : b ∥ α :=
sorry

def prop3 (ha₁ : a ⟂ α) (ha₂ : a ⟂ β) : α ∥ β :=
sorry

def prop4 (ha₁ : a ⟂ β) (ha₂ : α ⟂ β) : a ∥ α :=
sorry

theorem correct_propositions (ha₁ : a ∥ b) (ha₂ : a ⟂ α) 
                             (hb₁ : a ⟂ b) (hb₂ : a ⟂ α) 
                             (hc₁ : a ⟂ α) (hc₂ : a ⟂ β) 
                             (hd₁ : a ⟂ β) (hd₂ : α ⟂ β) :
  (prop1 ha₁ ha₂) ∧ (prop3 hc₁ hc₂) :=
begin
  split,
  { exact sorry },
  { exact sorry },
end

end correct_propositions_l813_813148


namespace peculiar_polynomial_p2_eq_4_l813_813398

def is_peculiar (p : ℝ → ℝ) : Prop :=
  ∃ r s : ℝ, p = (λ x, x^2 - (r+s)*x + r*s) ∧
  -- The condition that p(p(x)) = 0 has exactly four real roots
  ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  p(p(a)) = 0 ∧ p(p(b)) = 0 ∧ p(p(c)) = 0 ∧ p(p(d)) = 0

noncomputable def peculiar_polynomial_value : ℝ :=
  if (∃ p : ℝ → ℝ, is_peculiar p ∧ p 2 = 4) then 4 else 0

theorem peculiar_polynomial_p2_eq_4 :
  peculiar_polynomial_value = 4 :=
sorry

end peculiar_polynomial_p2_eq_4_l813_813398


namespace same_function_C_l813_813828

def f_A (x : ℝ) : ℝ := Real.sqrt (x ^ 2)
def g_A (x : ℝ) : ℝ := Real.cbrt (x ^ 3)

def f_B (x : ℝ) : ℝ := 1
def g_B (x : ℝ) : ℝ := if x = 0 then 1 else x ^ 0

def f_C (x : ℝ) : ℝ := 2 * x + 1
def g_C (t : ℝ) : ℝ := 2 * t + 1

def f_D (x : ℝ) : ℝ := x
def g_D (x : ℝ) : ℝ := (Real.sqrt x) ^ 2

theorem same_function_C : (∀ x, (f_C x = g_C x)) := by
  sorry

end same_function_C_l813_813828


namespace max_side_length_l813_813775

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813775


namespace max_side_length_l813_813700

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813700


namespace tiling_no_gaps_sum_reciprocal_l813_813070

theorem tiling_no_gaps_sum_reciprocal (x y z : ℕ) (hx : 3 ≤ x) (hy : 3 ≤ y) (hz : 3 ≤ z) 
    (h : ∃ (α β γ : ℝ), ∠regular_polygon_interior x α ∧ ∠regular_polygon_interior y β ∧ ∠regular_polygon_interior z γ ∧ (α + β + γ = 2 * pi)) :
    (1 / x + 1 / y + 1 / z = 1) :=
by
  sorry

end tiling_no_gaps_sum_reciprocal_l813_813070


namespace original_count_l813_813465

-- Conditions
def original_count_eq (ping_pong_balls shuttlecocks : ℕ) : Prop :=
  ping_pong_balls = shuttlecocks

def removal_count (x : ℕ) : Prop :=
  5 * x - 3 * x = 16

-- Theorem to prove the original number of ping-pong balls and shuttlecocks
theorem original_count (ping_pong_balls shuttlecocks : ℕ) (x : ℕ) (h1 : original_count_eq ping_pong_balls shuttlecocks) (h2 : removal_count x) : ping_pong_balls = 40 ∧ shuttlecocks = 40 :=
  sorry

end original_count_l813_813465


namespace math_proof_problem_l813_813425

noncomputable def validateConclusions (α φ : ℝ) (k : ℤ) (x0 x : ℝ) (p q : Prop) : Prop :=
  let neg1 := ¬(α = Real.pi / 6 → Real.sin α = 1 / 2)
  let neg2 := ∀ x : ℝ, Real.sin x ≤ 1
  let condition3 := φ = Real.pi / 2 + 2 * ↑k * Real.pi ↔ (¬(φ = Real.pi / 2 + k * Real.pi) ∨ (∀ x : ℝ, Real.sin (2 * x + φ) = Real.sin (-2 * x - φ)))
  let neg_p_and_q := (¬(∃ x ∈ Set.Ioo 0 (Real.pi / 2), Real.sin x + Real.cos x = 1 / 2) ∧ p → q ∧ p)

  let correct1 := neg1 = false
  let correct2 := neg2 = (¬ (∃ x0 : ℝ, Real.sin x0 > 1))
  let correct3 := ¬condition3
  let correct4 := ¬p ∧ q

  let number_of_correct_conclusions := (if correct1 then 1 else 0) + 
                                       (if correct2 then 1 else 0) + 
                                       (if correct3 then 0 else 1) + 
                                       (if correct4 then 1 else 0)
  number_of_correct_conclusions = 3

theorem math_proof_problem (α φ : ℝ) (k : ℤ) (x0 x : ℝ) (p q : Prop) :
  validateConclusions α φ k x0 x p q := sorry

end math_proof_problem_l813_813425


namespace max_side_length_l813_813705

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813705


namespace largest_in_n_minus_one_steps_not_less_than_n_minus_one_steps_l813_813002

theorem largest_in_n_minus_one_steps (n : ℕ) (hn : n ≥ 2) (numbers : Fin n → ℝ) : ∃ (compare : nat → nat → ℝ), 
  (∀ i j, i < j → compare i j = max (numbers i) (numbers j)) → sorry :=
sorry

theorem not_less_than_n_minus_one_steps (n : ℕ) (hn : n ≥ 2) (numbers : Fin n → ℝ) : ∀ (k : ℕ), 
  (k < n-1) → ∃ m1 m2, m1 ≠ m2 ∧ m1, m2 <= n-1 ∧ (∀ i < n, compare_steps i ≠) :=
sorry

end largest_in_n_minus_one_steps_not_less_than_n_minus_one_steps_l813_813002


namespace max_side_length_of_triangle_l813_813767

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813767


namespace chewbacca_gum_problem_l813_813845

theorem chewbacca_gum_problem (x : ℝ) :
  (18 - 2 * x) / 24 = 18 / (24 + 3 * x) → x = 1 :=
by
  intro h
  field_simp at h
  have : (18 - 2 * x) * (24 + 3 * x) = 18 * 24 := by linarith [h]
  simp [mul_add, add_mul] at this
  linarith
  sorry -- Completing the proof step is skipped

end chewbacca_gum_problem_l813_813845


namespace particle_position_after_moves_l813_813657

theorem particle_position_after_moves :
  let ω := complex.exp (complex.I * (real.pi / 3))
  let z₀ := complex.of_real 3
  (finset.range 120).foldl (λ z _, ω * z + 8) z₀ = z₀ :=
by
  sorry

end particle_position_after_moves_l813_813657


namespace fourth_month_sale_l813_813296

-- Define the sales for each month
constant sale_1 : ℕ
constant sale_2 : ℕ
constant sale_3 : ℕ
constant sale_4 : ℕ
constant sale_5 : ℕ
constant sale_6 : ℕ

-- Define the known sales and average
def sale_1 := 5420
def sale_2 := 5660
def sale_3 := 6200
def sale_5 := 6500
def sale_6 := 6470
def average_sale := 6100

-- State the theorem to prove
theorem fourth_month_sale : sale_4 = 6350 :=
by
  sorry

end fourth_month_sale_l813_813296


namespace martin_total_waste_is_10_l813_813168

def martinWastesTrafficTime : Nat := 2
def martinWastesFreewayTime : Nat := 4 * martinWastesTrafficTime
def totalTimeWasted : Nat := martinWastesTrafficTime + martinWastesFreewayTime

theorem martin_total_waste_is_10 : totalTimeWasted = 10 := 
by 
  sorry

end martin_total_waste_is_10_l813_813168


namespace find_k_l813_813071

noncomputable def poly (x : ℝ) : ℝ := 2 * x^3 + 9 * x^2 - 117 * x + k

theorem find_k (k : ℝ) (a b : ℝ) :
  (poly a = 0) ∧ (poly b = 0) ∧ (a = b) ∧ (k > 0) →
  k = 217.82 := by
  sorry

end find_k_l813_813071


namespace prime_p_satisfies_condition_l813_813358

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_p_satisfies_condition {p : ℕ} (hp : is_prime p) (hp2_8 : is_prime (p^2 + 8)) : p = 3 :=
sorry

end prime_p_satisfies_condition_l813_813358


namespace mowing_time_closest_to_2_3_hours_l813_813517

def swath_width_inches : ℝ := 30
def overlap_inches : ℝ := 2
def effective_swath_width_feet : ℝ := (swath_width_inches - overlap_inches) / 12
def lawn_length_feet : ℝ := 120
def lawn_width_feet : ℝ := 180
def walking_speed_feet_per_hour : ℝ := 4000

theorem mowing_time_closest_to_2_3_hours :
  let number_of_strips := lawn_width_feet / effective_swath_width_feet in
  let total_distance := number_of_strips.ceil * lawn_length_feet in
  let time_to_mow := total_distance / walking_speed_feet_per_hour in
  abs (time_to_mow - 2.3) < abs (time_to_mow - 2.1) ∧ abs (time_to_mow - 2.3) < abs (time_to_mow - 2.5) ∧
  abs (time_to_mow - 2.3) < abs (time_to_mow - 2.8) ∧ abs (time_to_mow - 2.3) < abs (time_to_mow - 3) :=
by
  sorry

end mowing_time_closest_to_2_3_hours_l813_813517


namespace max_pairs_of_friends_l813_813467

theorem max_pairs_of_friends (n : ℕ) :
  ∃ G : SimpleGraph (Fin (2 * n)),
  (∀ u v : Fin (2 * n), (G.degree u = G.degree v) → Disjoint (G.neighbor_set u) (G.neighbor_set v)) →
  G.edge_finset.card ≤ n * (n + 1) / 2 :=
sorry

end max_pairs_of_friends_l813_813467


namespace checkered_rectangles_containing_one_gray_cell_l813_813447

theorem checkered_rectangles_containing_one_gray_cell 
  (num_gray_cells : ℕ) 
  (num_blue_cells : ℕ) 
  (num_red_cells : ℕ)
  (blue_containing_rectangles : ℕ) 
  (red_containing_rectangles : ℕ) :
  num_gray_cells = 40 →
  num_blue_cells = 36 →
  num_red_cells = 4 →
  blue_containing_rectangles = 4 →
  red_containing_rectangles = 8 →
  num_blue_cells * blue_containing_rectangles + num_red_cells * red_containing_rectangles = 176 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end checkered_rectangles_containing_one_gray_cell_l813_813447


namespace max_marks_l813_813273

theorem max_marks (M : ℕ) (h1 : M * 33 / 100 = 175 + 56) : M = 700 :=
by
  sorry

end max_marks_l813_813273


namespace simplify_expression_l813_813202

variable (y : ℝ)

theorem simplify_expression : (3 * y)^3 + (4 * y) * (y^2) - 2 * y^3 = 29 * y^3 :=
by
  sorry

end simplify_expression_l813_813202


namespace two_a_minus_b_l813_813446

variables (a b : ℝ × ℝ)
variables (m : ℝ)

def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem two_a_minus_b 
  (ha : a = (1, -2))
  (hb : b = (m, 4))
  (h_parallel : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end two_a_minus_b_l813_813446


namespace circle_equation_correct_l813_813213

-- Define the given elements: center and radius
def center : (ℝ × ℝ) := (1, -1)
def radius : ℝ := 2

-- Define the equation of the circle with the given center and radius
def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = radius^2

-- Prove that the equation of the circle holds with the given center and radius
theorem circle_equation_correct : 
  ∀ x y : ℝ, circle_eqn x y ↔ (x - 1)^2 + (y + 1)^2 = 4 := 
by
  sorry

end circle_equation_correct_l813_813213


namespace sum_of_prime_factors_l813_813586

theorem sum_of_prime_factors (x : ℕ) (h1 : x = 2^10 - 1) 
  (h2 : 2^10 - 1 = (2^5 + 1) * (2^5 - 1)) 
  (h3 : 2^5 - 1 = 31) 
  (h4 : 2^5 + 1 = 33) 
  (h5 : 33 = 3 * 11) : 
  (31 + 3 + 11 = 45) := 
  sorry

end sum_of_prime_factors_l813_813586


namespace quadratic_root_value_l813_813952

theorem quadratic_root_value {m : ℝ} (h : m^2 + m - 1 = 0) : 2 * m^2 + 2 * m + 2025 = 2027 :=
sorry

end quadratic_root_value_l813_813952


namespace cone_ratio_l813_813317

theorem cone_ratio (r h : ℕ) (m n : ℕ) (hnpos : 0 < n) (hnnot_square : ∀ p : ℕ, prime p → p ^ 2 ∣ n → false) (eq1 : h ^ 2 = 288 * r ^ 2) (eq2 : h = m * r * n.sqrt) : m + n = 14 := 
by 
  sorry

end cone_ratio_l813_813317


namespace find_x0_l813_813951

theorem find_x0 (f : ℝ → ℝ) (f' : ℝ → ℝ) (x₀ : ℝ) 
  (h1 : f = λ x, x^3 + x - 1) 
  (h2 : f' = λ x, 3*x^2 + 1) 
  (h3 : f' x₀ = 4) : x₀ = 1 ∨ x₀ = -1 :=
sorry

end find_x0_l813_813951


namespace average_string_length_l813_813016

theorem average_string_length :
  let lengths := [1.5, 4.5, 6.0, 3.0] in
  (list.sum lengths) / (list.length lengths) = 3.75 :=
by
  sorry

end average_string_length_l813_813016


namespace females_in_group_l813_813382

theorem females_in_group (F M : ℕ) (h1 : F + M = 20) 
    (h2 : (20 - F) / 20 - (20 - M) / 20 = 0.30000000000000004) : F = 7 := 
by
  sorry

end females_in_group_l813_813382


namespace intersecting_circles_unique_point_l813_813100

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l813_813100


namespace sufficient_but_not_necessary_condition_l813_813630

theorem sufficient_but_not_necessary_condition
  (m : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, 1 ≤ x → f x = 3^(x + m) - 3 * real.sqrt 3)
  (h2 : ∀ x : ℝ, 1 ≤ x → f x ≠ 0):
  m > 1 := sorry

end sufficient_but_not_necessary_condition_l813_813630


namespace card_pairing_probability_l813_813646

theorem card_pairing_probability (m n : ℕ) (hrelprime : Nat.gcd m n = 1) :
  let deck := 48
  let pair_count := 68
  let total_ways := Nat.choose deck 2
  let prob := pair_count * 1128 / total_ways
  m = 17 ∧ n = 282 →
  m + n = 299 :=
by
  intros _ _ _ _ _ _ h
  sorry

end card_pairing_probability_l813_813646


namespace find_max_side_length_l813_813782

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813782


namespace max_side_length_of_triangle_l813_813758

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813758


namespace find_x_for_cos_alpha_l813_813025

theorem find_x_for_cos_alpha (α : ℝ) (x : ℝ) (hx : x^2 = 3) (hα_quad : 90 < α ∧ α < 180)
  (P : ℝ × ℝ) (hP : P = (x, sqrt 5)) (hcos : cos α = (sqrt 2 / 4) * x) :
  x = -√3 :=
by sorry

end find_x_for_cos_alpha_l813_813025


namespace circumcenter_on_angle_bisector_l813_813403

open EuclideanGeometry

theorem circumcenter_on_angle_bisector
  (A B C P Q O : Point)
  (h_triangle_ABC : ¬Collinear {A, B, C})
  (hP : ∃ K, Between B K P ∧ dist B P = dist B A)
  (hQ : ∃ K, Between C K Q ∧ dist C Q = dist C A)
  (hO : Cirumcenter O A P Q) :
  On_Bisector O A B C :=
sorry

end circumcenter_on_angle_bisector_l813_813403


namespace negation_proposition_l813_813572

theorem negation_proposition :
  ¬ (∀ x : ℝ, x ≥ 0 → sin x ≤ 1) ↔ ∃ x : ℝ, x ≥ 0 ∧ sin x > 1 := 
sorry

end negation_proposition_l813_813572


namespace projection_a_on_b_l813_813051

-- Define vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

-- Prove that the projection of a on b is equal to the given value
theorem projection_a_on_b :
  let dot_product := (a.1 * b.1 + a.2 * b.2)
  let mag_a := real.sqrt (a.1^2 + a.2^2)
  let mag_b := real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (mag_a * mag_b)
  let projection_length := mag_a * cos_theta
  projection_length = real.sqrt(65) / 5 :=
by
  -- Here comes the proof steps, which are skipped
  sorry

end projection_a_on_b_l813_813051


namespace max_triangle_side_24_l813_813737

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813737


namespace probability_at_most_two_heads_l813_813279

/-- The probability of getting at most 2 heads when three unbiased coins are tossed is 7/8. -/
theorem probability_at_most_two_heads : 
  let outcomes := { (toss1: Bool, toss2: Bool, toss3: Bool) | true } in
  let favorable := { outcome ∈ outcomes | 
                     let heads_count := outcome.val.count (λ b => b = true) in 
                     heads_count ≤ 2 } in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 7 / 8 :=
sorry

end probability_at_most_two_heads_l813_813279


namespace inscribed_rectangle_exists_l813_813013

-- Assuming we have a triangle ABC with vertices A, B, C
structure Triangle :=
  (A B C : Point)

-- Given perimeter P of a rectangle
constant (P : ℝ)

-- Define point type
structure Point :=
  (x y : ℝ)

-- Define a Rectangle structure with vertices R1, R2, R3, R4
structure Rectangle :=
  (R1 R2 R3 R4 : Point)
  (Perimeter : ℝ)

-- Main theorem statement
theorem inscribed_rectangle_exists (T : Triangle) (R : Rectangle) :
  R.R1 = T.A ∧ 
  (line_through R.R2 R.R3 intersects T.B) ∧ 
  (line_through R.R3 R.R4 intersects T.C) ∧ 
  R.Perimeter = P :=
sorry

end inscribed_rectangle_exists_l813_813013


namespace recoveries_second_day_l813_813326

-- Conditions
variables (R : ℕ) (cases_day1 cases_incr_day2 cases_new_day3 recoveries_day3 total_cases_day3 : ℕ)

-- Specific values from the problem
def cases_day1 : ℕ := 2000
def cases_incr_day2 : ℕ := 500
def cases_new_day3 : ℕ := 1500
def recoveries_day3 : ℕ := 200
def total_cases_day3 : ℕ := 3750

-- The Lean 4 statement for the proof problem
theorem recoveries_second_day : (cases_day1 + cases_incr_day2 - R + cases_new_day3 - recoveries_day3 = total_cases_day3) → R = 50 :=
by
  sorry

end recoveries_second_day_l813_813326


namespace max_side_length_is_11_l813_813671

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813671


namespace abc_range_l813_813005

-- Define the function f(x) as described in the problem:
def f (x : ℝ) : ℝ :=
if 0 < x ∧ x <= 9 
then log 3 x - 1 
else 4 - sqrt x

-- State the theorem for the range of abc given f(a) = f(b) = f(c)
theorem abc_range (a b c : ℝ) (h : a ≠ b ∧ a ≠ c ∧ b ≠ c) 
  (ha : 0 < a ∧ a ≤ 9 ∨ a > 9) (hb : 0 < b ∧ b ≤ 9 ∨ b > 9)
  (hc : 0 < c ∧ c ≤ 9 ∨ c > 9)
  (hfab : f a = f b) (hfbc : f b = f c) : 
  81 < a * b * c ∧ a * b * c < 144 :=
sorry

end abc_range_l813_813005


namespace find_a_l813_813223

noncomputable def f (x a : ℝ) : ℝ := -x^2 + 2 * a * x + 1 - a

theorem find_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x a ≤ 2) ∧ (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x a = 2) →
  (a = -1 ∨ a = 2) :=
begin
  sorry
end

end find_a_l813_813223


namespace cut_one_more_2x2_square_l813_813319

theorem cut_one_more_2x2_square (total_squares remaining_squares: ℕ) : 
    29 * 29 ≥ total_squares + remaining_squares * 4 → 
    total_squares = 99 → 
    ∃ unused_cells, unused_cells ≥ 4 ∧ total_squares + 4 = remaining_squares * 4 → 
    true := 
begin
  sorry
end

end cut_one_more_2x2_square_l813_813319


namespace max_side_length_is_11_l813_813676

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813676


namespace zane_spent_more_on_cookies_l813_813230

theorem zane_spent_more_on_cookies
  (o c : ℕ) -- number of Oreos and cookies
  (pO pC : ℕ) -- price of each Oreo and cookie
  (h_ratio : 9 * o = 4 * c) -- ratio condition
  (h_price_O : pO = 2) -- price of Oreo
  (h_price_C : pC = 3) -- price of cookie
  (h_total : o + c = 65) -- total number of items
  : 3 * c - 2 * o = 95 := 
begin
  sorry
end

end zane_spent_more_on_cookies_l813_813230


namespace max_side_of_triangle_l813_813797

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813797


namespace max_side_length_l813_813772

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813772


namespace centers_circumcircles_parallel_AD_l813_813864

open EuclideanGeometry

theorem centers_circumcircles_parallel_AD
  (A B C D P Q : Point)
  (h1 : cyclic_quad A B C D)
  (h2 : AC ∩ BD = Some P)
  (h3 : Q ∈ BC)
  (h4 : PQ ⊥ AC) :
  let ω1 := circumcenter A P D,
      ω2 := circumcenter B Q D in
  line_through ω1 ω2 ∥ AD :=
by
  sorry

end centers_circumcircles_parallel_AD_l813_813864


namespace general_term_formula_l813_813400

-- Definitions from conditions
def a : ℕ → ℝ
| 0       := 1       -- Lean uses zero-based indexing, hence a_1 is a(0)
| n + 1   := 3 * (Finset.sum (Finset.range (n + 1)) a)

-- General term formula we need to prove
theorem general_term_formula (n : ℕ) : 
  a n = if n = 0 then 1 else 3 * 4^(n-1) :=
sorry

end general_term_formula_l813_813400


namespace integer_divisibility_l813_813963

open Nat

theorem integer_divisibility (n : ℕ) (h1 : ∃ m : ℕ, 2^n - 2 = n * m) : ∃ k : ℕ, 2^((2^n) - 1) - 2 = (2^n - 1) * k := by
  sorry

end integer_divisibility_l813_813963


namespace exists_m_in_range_l813_813066

theorem exists_m_in_range :
  ∃ m : ℝ, 0 ≤ m ∧ m < 1 ∧ ∀ x : ℕ, (x > m ∧ x < 2) ↔ (x = 1) :=
by
  sorry

end exists_m_in_range_l813_813066


namespace product_mod_10_l813_813619

theorem product_mod_10 (a b c : ℕ) (ha : a % 10 = 4) (hb : b % 10 = 5) (hc : c % 10 = 5) :
  (a * b * c) % 10 = 0 :=
sorry

end product_mod_10_l813_813619


namespace expression_value_l813_813626

theorem expression_value (x y z w : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 := 
sorry

end expression_value_l813_813626


namespace max_side_length_of_triangle_l813_813760

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813760


namespace nina_weeks_to_afford_game_l813_813521

noncomputable def game_cost : ℝ := 50
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def weekly_allowance : ℝ := 10
noncomputable def saving_rate : ℝ := 0.5

noncomputable def total_cost : ℝ := game_cost + (game_cost * sales_tax_rate)
noncomputable def savings_per_week : ℝ := weekly_allowance * saving_rate
noncomputable def weeks_needed : ℝ := total_cost / savings_per_week

theorem nina_weeks_to_afford_game : weeks_needed = 11 := by
  sorry

end nina_weeks_to_afford_game_l813_813521


namespace find_min_value_l813_813224

noncomputable def min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :=
  (8 * a + b) / (a * b)

theorem find_min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :
  min_value a b h_a h_b h_slope = 18 :=
sorry

end find_min_value_l813_813224


namespace weight_of_original_piece_of_marble_l813_813663

theorem weight_of_original_piece_of_marble (W : ℝ) 
  (h1 : W > 0)
  (h2 : (0.75 * 0.56 * W) = 105) : 
  W = 250 :=
by
  sorry

end weight_of_original_piece_of_marble_l813_813663


namespace chessboard_trimino_uncovered_cell_l813_813641

theorem chessboard_trimino_uncovered_cell 
  (MATRIX: Type)
  (m n : ℕ)
  [finite m] [finite n]
  [all_1_2_3 : MATRIX → m × n → fin 3]
  (chessboard : MATRIX)
  (covered_by_triminos : (MATRIX → m × n → fin 3) → ∀ M: MATRIX, finite (fin 3))
  (h_condition1 : m = 8 ∧ n = 8)
  (h_condition2 : ∀ chessboard, ∃ ! uncovered_cell, ¬ covered_by_triminos all_1_2_3 uncovered_cell)
  (h_triminos_cover_3cells : ∀ trimino, ∃ cells : (list ((MATRIX → m × n → fin 3))), trimino', fin 3),
    trimino'.1 = fin.mk && trimino'.2 = fin.mk && trimino'.3 = fin.mk
  )
: ∃ cell : fin (64), cell ∈ finset.of_list [(1,1), (1,4), (4,1), (4,4)] :=
by
  sorry

end chessboard_trimino_uncovered_cell_l813_813641


namespace max_side_length_l813_813781

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813781


namespace number_of_best_friends_l813_813175

-- Constants and conditions
def initial_tickets : ℕ := 37
def tickets_per_friend : ℕ := 5
def tickets_left : ℕ := 2

-- Problem statement
theorem number_of_best_friends : (initial_tickets - tickets_left) / tickets_per_friend = 7 :=
by
  sorry

end number_of_best_friends_l813_813175


namespace central_angle_of_regular_hexagon_l813_813560

-- Define the total degrees in a circle
def total_degrees_in_circle : ℝ := 360

-- Define the number of sides in a regular hexagon
def sides_in_hexagon : ℕ := 6

-- Theorems to prove that the central angle of a regular hexagon is 60°
theorem central_angle_of_regular_hexagon :
  total_degrees_in_circle / sides_in_hexagon = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l813_813560


namespace alpha_range_l813_813059

theorem alpha_range (α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) : 
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔ 
  (0 < α ∧ α < Real.pi / 3 ∨ 5 * Real.pi / 3 < α ∧ α < 2 * Real.pi) := 
sorry

end alpha_range_l813_813059


namespace central_angle_of_regular_hexagon_l813_813558

theorem central_angle_of_regular_hexagon:
  ∀ (α : ℝ), 
  (∃ n : ℕ, n = 6 ∧ n * α = 360) →
  α = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l813_813558


namespace incorrect_inequality_l813_813001

theorem incorrect_inequality (m n : ℝ) (a : ℝ) (hmn : m > n) (hm1 : m > 1) (hn1 : n > 1) (ha0 : 0 < a) (ha1 : a < 1) : 
  ¬ (a^m > a^n) :=
sorry

end incorrect_inequality_l813_813001


namespace infinite_product_a_eq_3_div_5_l813_813661

noncomputable def a : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 1 + (a n - 1)^2

theorem infinite_product_a_eq_3_div_5 : 
    (∏ n, a n) = 3 / 5 :=
sorry

end infinite_product_a_eq_3_div_5_l813_813661


namespace pizza_slices_left_l813_813191

theorem pizza_slices_left (initial_slices cuts friends total_given : ℕ) (h_slices: initial_slices = 1 * 2^cuts)
  (h_first_group: ∀ f ∈ friends, f = 2) (h_second_group: ∀ f ∉ friends, f = 1) 
  (h_friends: ∑ f in friends, f = total_given) 
  (distribution: total_given = (card friends * 2 + (card (range 3) - card friends) * 1)) 
  (h_card_friends: card friends = 2) : 
  (initial_slices - total_given = 1) :=
sorry

end pizza_slices_left_l813_813191


namespace find_k_values_for_intersection_l813_813114

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l813_813114


namespace max_triangle_side_l813_813720

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813720


namespace determine_m_first_degree_inequality_l813_813450

theorem determine_m_first_degree_inequality (m : ℝ) (x : ℝ) :
  (m + 1) * x ^ |m| + 2 > 0 → |m| = 1 → m = 1 :=
by
  intro h1 h2
  sorry

end determine_m_first_degree_inequality_l813_813450


namespace four_digit_numbers_divisible_by_nine_l813_813937

theorem four_digit_numbers_divisible_by_nine : 
  (count (λ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 9 = 0) = 1000) :=
by
  sorry

end four_digit_numbers_divisible_by_nine_l813_813937


namespace number_of_primes_between_20_and_30_l813_813448

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k > 1 ∧ k < n → n % k ≠ 0

def primes_between_20_and_30 : ℕ :=
  (finset.filter is_prime (finset.range 31)).filter (λ n, 20 < n ∧ n < 30)
    .card

theorem number_of_primes_between_20_and_30 :
  primes_between_20_and_30 = 2 := by
  sorry

end number_of_primes_between_20_and_30_l813_813448


namespace profit_in_2004_correct_l813_813636

-- We define the conditions as given in the problem
def annual_profit_2002 : ℝ := 10
def annual_growth_rate (p : ℝ) : ℝ := p

-- The expression for the annual profit in 2004 given the above conditions
def annual_profit_2004 (p : ℝ) : ℝ := annual_profit_2002 * (1 + p) * (1 + p)

-- The theorem to prove that the computed annual profit in 2004 matches the expected answer
theorem profit_in_2004_correct (p : ℝ) :
  annual_profit_2004 p = 10 * (1 + p)^2 := 
by 
  sorry

end profit_in_2004_correct_l813_813636


namespace even_func_monotonic_on_negative_interval_l813_813960

variable {α : Type*} [LinearOrderedField α]
variable {f : α → α}

theorem even_func_monotonic_on_negative_interval 
  (h_even : ∀ x : α, f (-x) = f x)
  (h_mon_incr : ∀ x y : α, x < y → (x < 0 ∧ y ≤ 0) → f x < f y) :
  f 2 < f (-3 / 2) :=
sorry

end even_func_monotonic_on_negative_interval_l813_813960


namespace units_digit_of_7_pow_6_pow_5_l813_813862

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l813_813862


namespace sequence_a_sequence_S_l813_813511

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / 3^n

def b (n : ℕ) : ℚ :=
  if hn : n ≠ 0 then n / (a n) else 0

def S (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, b (i + 1)

theorem sequence_a (n : ℕ) (hn : n ≠ 0) :
  (∑ i in Finset.range n, 3^i * a (i + 1)) = n / 3 := sorry

theorem sequence_S (n : ℕ) :
  S n = (2 * n - 1) * 3^(n+1) / 4 + 3 / 4 := sorry

end sequence_a_sequence_S_l813_813511


namespace part1_max_value_part2_a_range_l813_813433

-- Definition of the function
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a * x^2

-- Prove Part 1: If f'(1) = 3, the maximum value of f(x) on [0, 2] is 8
theorem part1_max_value (a : ℝ) (h : deriv (f a) 1 = 3) : ∃ x ∈ set.Icc (0 : ℝ) (2 : ℝ), (f a x) = 8 := sorry

-- Prove Part 2: If f(x) is increasing on [1, 2], then a ∈ (-∞, 3/2]
theorem part2_a_range (a : ℝ) (h : ∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), deriv (f a) x ≥ 0) : a ≤ 3/2 := sorry

end part1_max_value_part2_a_range_l813_813433


namespace find_x_from_angles_l813_813477

theorem find_x_from_angles : ∀ (x : ℝ), (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end find_x_from_angles_l813_813477


namespace equal_sundays_tuesdays_l813_813309

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l813_813309


namespace max_side_length_is_11_l813_813679

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813679


namespace no_real_roots_x2_bx_8_eq_0_l813_813896

theorem no_real_roots_x2_bx_8_eq_0 (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 5 ≠ -3) ↔ (-4 * Real.sqrt 2 < b ∧ b < 4 * Real.sqrt 2) := by
  sorry

end no_real_roots_x2_bx_8_eq_0_l813_813896


namespace time_to_pass_l813_813611

noncomputable def length_train := 150
noncomputable def speed_train1_kmh := 95
noncomputable def speed_train2_kmh := 85

noncomputable def speed_train1_ms := speed_train1_kmh * 1000 / 3600
noncomputable def speed_train2_ms := speed_train2_kmh * 1000 / 3600

noncomputable def relative_speed_ms := speed_train1_ms + speed_train2_ms
noncomputable def total_length := length_train * 2

theorem time_to_pass : total_length / relative_speed_ms = 6 := by
  simp [total_length, relative_speed_ms, speed_train1_ms, speed_train2_ms, length_train, speed_train1_kmh, speed_train2_kmh]
  sorry

end time_to_pass_l813_813611


namespace units_digit_7_pow_6_pow_5_l813_813858

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end units_digit_7_pow_6_pow_5_l813_813858


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813691

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813691


namespace quadruple_application_of_h_l813_813352

-- Define the function as specified in the condition
def h (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem quadruple_application_of_h : h (h (h (h 40))) = 9.536 :=
  by
    sorry

end quadruple_application_of_h_l813_813352


namespace correct_statements_l813_813432

noncomputable def f (x k : ℝ) : ℝ :=
  real.log x - k * real.sin x - 1

theorem correct_statements (x : ℝ) (k : ℝ) (h1 : x ∈ set.Ioc 0 real.pi):
  (∀ k ≥ 0, ∃ x ∈ set.Ioc 0 real.pi, f x k = 0) ∧ 
  (∃ k : ℝ, k ∉ {k | k ≤ -1/real.pi ∨ k ≥ 1/real.pi}, ∃ x ∈ set.Ioc 0 real.pi, ∀ y ∈ set.Ioc 0 real.pi, f y k ≤ f x k) :=
sorry

end correct_statements_l813_813432


namespace rectangle_inscription_l813_813849

theorem rectangle_inscription (AB BC : ℝ) (h_AB : AB = 2) (h_BC : BC = 1)
  (E F : ℝ) (h_triangle : E = F ∧ F = sqrt 3) (x : ℝ) (h_x : x = sqrt 3) :
  x = sqrt 3 :=
sorry

end rectangle_inscription_l813_813849


namespace triangle_problems_l813_813483

noncomputable def triangle_relations (A B C a b c : ℝ) :=
  a = 3 ∧ sin B = sin (2 * A) ∧ 
  ((a > c → 3 < b ∧ b < 3 * Real.sqrt 2) ∧ (b = 2 * a * cos A)) 

theorem triangle_problems (A B C a b c : ℝ) :
  triangle_relations A B C a b c → (b / (cos A) = 6) :=
by
  sorry

end triangle_problems_l813_813483


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813686

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813686


namespace part_a_part_b_part_c_part_d_l813_813196

theorem part_a : (4237 * 27925 ≠ 118275855) :=
by sorry

theorem part_b : (42971064 / 8264 ≠ 5201) :=
by sorry

theorem part_c : (1965^2 ≠ 3761225) :=
by sorry

theorem part_d : (23 ^ 5 ≠ 371293) :=
by sorry

end part_a_part_b_part_c_part_d_l813_813196


namespace gain_percent_is_correct_l813_813943

theorem gain_percent_is_correct (gain_in_paise : ℝ) (cost_price_in_rs : ℝ) (conversion_factor : ℝ)
  (gain_percent_formula : ∀ (gain : ℝ) (cost : ℝ), ℝ) : 
  gain_percent_formula (gain_in_paise / conversion_factor) cost_price_in_rs = 1 :=
by
  let gain := gain_in_paise / conversion_factor
  let cost := cost_price_in_rs
  have h : gain_percent_formula gain cost = (gain / cost) * 100 := sorry
  have h2 : gain_percent_formula (70 / 100) 70 = 1 := sorry
  exact h2

end gain_percent_is_correct_l813_813943


namespace ellipse_eccentricity_l813_813583

noncomputable def complex_roots_of_polynomial (z : ℂ) : Prop :=
  (z - 2) * (z^2 + 3 * z + 5) * (z^2 + 5 * z + 8) = 0

theorem ellipse_eccentricity (roots : Fin 5 → ℂ) 
  (h_roots : ∀ k, complex_roots_of_polynomial (roots k)) :
  ∃ m n : ℕ, Nat.coprime m n ∧ 
  ∑ r in Finset.univ, r ∈ complex_roots_of_polynomial ∧ 
  let xs := map (λ r, r.re) roots in
  let ys := map (λ r, r.im) roots in
  let h := -2 in
  let a_sq := (xs.to_list.max - xs.to_list.min) ^ 2 in
  let b_sq := (ys.to_list.max - ys.to_list.min) ^ 2 in
  let c_sq := a_sq - b_sq in
  let e := c_sq.sqrt / a_sq.sqrt in
  e = Real.sqrt (m / n) ∧
  m∊ Finset.univ ∧
  n∊ Finset.univ :=
begin
  sorry
end

end ellipse_eccentricity_l813_813583


namespace permutation_indices_divisible_by_prime_l813_813208

theorem permutation_indices_divisible_by_prime (a b : Finₓ 2016 → ℕ) (h₁ : ∀ i, 1 ≤ a i ∧ a i ≤ 2016) (h₂ : ∀ i, 1 ≤ b i ∧ b i ≤ 2016) (h₃ : ∀ i ≠ j, a i ≠ a j) (h₄ : ∀ i ≠ j, b i ≠ b j) :
  ∃ i j, i ≠ j ∧ 2017 ∣ (a i * b i) - (a j * b j) :=
sorry

end permutation_indices_divisible_by_prime_l813_813208


namespace max_side_of_triangle_l813_813799

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813799


namespace cycle_exists_l813_813463

-- Definitions given in the problem
variables {City : Type*} {Airline : Type*}
variables (flights : City → City → Airline → Prop)
variables (exists_flight : ∀ (A : City), ∃ (B : City) (a1 a2 a3: Airline), 
                        a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3 ∧ (flights A B a1 ∨ flights A B a2 ∨ flights A B a3))

-- The theorem to be proved
theorem cycle_exists : 
  ∃ (cycle : list (City × Airline)), 
    cycle.head!.fst = cycle.last!.fst ∧ 
    (∀ (i : ℕ), i < cycle.length → i ≠ cycle.length - 1 → 
      cycle.nth_le! i.fst ≠ cycle.nth_le! (i+1).mod cycle.length.fst) ∧
    ∃ (i1 i2 i3 : ℕ), i1 < cycle.length ∧ i2 < cycle.length ∧ i3 < cycle.length ∧ 
      i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧ 
      (cycle[i1].snd = cycle[i3].snd) ∧ (cycle[i2].snd ≠ cycle[i3].snd) ∧ (cycle[i1].snd ≠ cycle[i2].snd) :=
sorry

end cycle_exists_l813_813463


namespace slices_left_for_phill_correct_l813_813187

-- Define the initial conditions about the pizza and the distribution.
def initial_pizza := 1
def slices_after_first_cut := initial_pizza * 2
def slices_after_second_cut := slices_after_first_cut * 2
def slices_after_third_cut := slices_after_second_cut * 2
def total_slices_given_to_two_friends := 2 * 2
def total_slices_given_to_three_friends := 3 * 1
def total_slices_given_out := total_slices_given_to_two_friends + total_slices_given_to_three_friends
def slices_left_for_phill := slices_after_third_cut - total_slices_given_out

-- State the theorem we need to prove.
theorem slices_left_for_phill_correct : slices_left_for_phill = 1 := by sorry

end slices_left_for_phill_correct_l813_813187


namespace equation_solutions_l813_813941

noncomputable def count_solutions (a : ℝ) : ℕ :=
  if 0 < a ∧ a <= 1 ∨ a = Real.exp (1 / Real.exp 1) then 1
  else if 1 < a ∧ a < Real.exp (1 / Real.exp 1) then 2
  else if a > Real.exp (1 / Real.exp 1) then 0
  else 0

theorem equation_solutions (a : ℝ) (h₀ : 0 < a) :
  (∃! x : ℝ, a^x = x) ↔ count_solutions a = 1 ∨ count_solutions a = 2 ∨ count_solutions a = 0 := sorry

end equation_solutions_l813_813941


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813821

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813821


namespace area_of_region_is_four_l813_813372

noncomputable theory
open Topology Classical

-- Conditions of the problem defined as functions and sets
def first_condition (x : ℝ) : Prop := sqrt (1 - x) + 2 * x ≥ 0

def second_condition (x y : ℝ) : Prop := -1 - x^2 ≤ y ∧ y ≤ 2 + sqrt x

-- Define the region of interest
def region (p : ℝ × ℝ) : Prop := 
  let (x, y) := p in 
  first_condition x ∧ second_condition x y

-- Define the bounds of integration
def x_bounds (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Upper and lower bounds for y
def y_upper (x : ℝ) : ℝ := 2 + sqrt x
def y_lower (x : ℝ) : ℝ := -1 - x^2

-- Define the target area as a definite integral
def target_area : ℝ :=
  ∫ x in (0 : ℝ)..1, (y_upper x - y_lower x)

-- The statement of the problem
theorem area_of_region_is_four : target_area = 4 := 
  sorry

end area_of_region_is_four_l813_813372


namespace intersection_eq_l813_813440

def A : Set ℝ := {x | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {x | (-2 < x ∧ x < -π/2) ∨ (π/3 < x ∧ x < π/2)} :=
by
  sorry

end intersection_eq_l813_813440


namespace hp_passes_through_midpoint_bc_l813_813901

variables {A B C H E F K P M : Type} [NonIsoscelesAcuteTriangle A B C] 

-- Assuming the definitions of altitudes and orthocenter and necessary points in a triangle
variables [Altitude BE A C] [Altitude CF B A] [Orthocenter H A B C] 
variables [Intersection K B C F E] [Midpoint M B C] [Perpendicular HP A K]

theorem hp_passes_through_midpoint_bc :
  HP ∩ BC = M :=
sorry

end hp_passes_through_midpoint_bc_l813_813901


namespace max_distance_sum_l813_813534

-- Definitions for triangle and distances
variables {A B C M : Type} [linear_ordered_field A] [decidable_eq A] [geometry_state A B C M]
def altitude (a b c : Type) := sorry -- replace with proper altitude definition

theorem max_distance_sum (a b c m : A) (ha hb hc : A)
    (h_altitude : altitude a b c = ha)
    (h_altitude : altitude b a c = hb)
    (h_altitude : altitude c a b = hc) :
    forall m, distance_to_sides_sum m a b c ≤ max ha (max hb hc) :=
sorry

end max_distance_sum_l813_813534


namespace geom_prog_roots_a_eq_22_l813_813897

theorem geom_prog_roots_a_eq_22 (x1 x2 x3 a : ℝ) :
  (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) → 
  (∃ b q, (x1 = b ∧ x2 = b * q ∧ x3 = b * q^2) ∧ (x1 + x2 + x3 = 11) ∧ (x1 * x2 * x3 = 8) ∧ (x1*x2 + x2*x3 + x3*x1 = a)) → 
  a = 22 :=
sorry

end geom_prog_roots_a_eq_22_l813_813897


namespace intersection_of_circles_l813_813090

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l813_813090


namespace days_with_equal_sun_tue_l813_813299

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l813_813299


namespace probability_of_picking_letter_in_mathematics_l813_813455

theorem probability_of_picking_letter_in_mathematics : 
  let total_unique_letters_in_mathematics := 8 in
  let total_letters_in_alphabet := 26 in
  total_unique_letters_in_mathematics / total_letters_in_alphabet = (4 : ℝ) / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l813_813455


namespace melanie_total_dimes_l813_813515

-- Definitions based on the problem conditions
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def mom_dimes : ℕ := 4

def total_dimes : ℕ := initial_dimes + dad_dimes + mom_dimes

-- Proof statement based on the correct answer
theorem melanie_total_dimes : total_dimes = 19 := by 
  -- Proof here is omitted as per instructions
  sorry

end melanie_total_dimes_l813_813515


namespace slices_left_for_phill_correct_l813_813189

-- Define the initial conditions about the pizza and the distribution.
def initial_pizza := 1
def slices_after_first_cut := initial_pizza * 2
def slices_after_second_cut := slices_after_first_cut * 2
def slices_after_third_cut := slices_after_second_cut * 2
def total_slices_given_to_two_friends := 2 * 2
def total_slices_given_to_three_friends := 3 * 1
def total_slices_given_out := total_slices_given_to_two_friends + total_slices_given_to_three_friends
def slices_left_for_phill := slices_after_third_cut - total_slices_given_out

-- State the theorem we need to prove.
theorem slices_left_for_phill_correct : slices_left_for_phill = 1 := by sorry

end slices_left_for_phill_correct_l813_813189


namespace nextSimultaneousRingingTime_l813_813325

-- Define the intervals
def townHallInterval := 18
def universityTowerInterval := 24
def fireStationInterval := 30

-- Define the start time (in minutes from 00:00)
def startTime := 8 * 60 -- 8:00 AM

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Prove the next simultaneous ringing time
theorem nextSimultaneousRingingTime : 
  let lcmIntervals := lcm (lcm townHallInterval universityTowerInterval) fireStationInterval 
  startTime + lcmIntervals = 14 * 60 := -- 14:00 equals 2:00 PM in minutes
by
  -- You can replace the proof with the actual detailed proof.
  sorry

end nextSimultaneousRingingTime_l813_813325


namespace BE_l813_813277

section geometry_problem
variables {A B C E F E' F' : Point}

-- Assumptions
variable (h_E_on_AC : E ∈ line A C)
variable (h_F_on_AB : F ∈ line A B)
variable (h_BE_parallel_CF : BE ∥ CF)
variable (h_circumcircle_BCE : ∃ω, is_circumcircle ω B C E ∧ ∃F', F' ∈ ω ∧ F' ∈ line A B)
variable (h_circumcircle_BCF : ∃ω', is_circumcircle ω' B C F ∧ ∃E', E' ∈ ω' ∧ E' ∈ line A C)

-- Theorem to Prove
theorem BE'_parallel_CF' :
  BE' ∥ CF' :=
sorry

end geometry_problem

end BE_l813_813277


namespace car_mileage_before_modification_l813_813292

variables (x : ℕ)

theorem car_mileage_before_modification :
  (16 * (5 * x / 4) - 16 * x = 176) → (x = 44) :=
by
  intro h,
  sorry

end car_mileage_before_modification_l813_813292


namespace max_side_length_is_11_l813_813678

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813678


namespace kenneth_joystick_percentage_spent_l813_813992

theorem kenneth_joystick_percentage_spent :
  ∀ (earnings amount_left : ℝ), 
  earnings = 450 ∧ amount_left = 405 → 
  ((earnings - amount_left) / earnings) * 100 = 10 := 
by
  intros earnings amount_left
  intro h
  cases h with h_earnings h_amount_left
  simp [h_earnings, h_amount_left]
  sorry

end kenneth_joystick_percentage_spent_l813_813992


namespace max_side_length_l813_813702

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813702


namespace problem_l813_813907

theorem problem (x y : ℝ) (h : 2^x + 3^y = 4^x + 9^y) :
  1 < 8^x + 27^y ∧ 8^x + 27^y ≤ 2 :=
by
  sorry

end problem_l813_813907


namespace digital_earth_functions_l813_813216

def DigitalEarth : Type := sorry

def function1 (d : DigitalEarth) : Prop :=
  "Joint research by scientists worldwide on the global environment and climate change, disaster prevention."

def function2 (d : DigitalEarth) : Prop :=
  "Providing a global educational interface for learning new knowledge and mastering new technologies."

def function3 (d : DigitalEarth) : Prop :=
  "Making it easier to track and investigate crime patterns and gang activities on a global scale."

def function4 (d : DigitalEarth) : Prop :=
  "Facilitating governments, research institutions, and organizations in addressing population issues and sustainable development."

theorem digital_earth_functions (d : DigitalEarth) : 
  function1 d ∧ function2 d ∧ function3 d ∧ function4 d :=
sorry

end digital_earth_functions_l813_813216


namespace julia_bakes_per_day_l813_813982

theorem julia_bakes_per_day (x : ℕ) (h1 : ∀ x, total_baked = 6 * x)
  (h2 : ∀ x, cakes_eaten = 3) (h3 : ∀ x, total_baked - cakes_eaten = 21) :
  x = 4 :=
by
  -- Sorry, the proof is omitted.
  sorry

end julia_bakes_per_day_l813_813982


namespace max_rectangle_area_l813_813350

theorem max_rectangle_area (l w : ℕ) (h : 3 * l + 5 * w ≤ 50) : (l * w ≤ 35) :=
by sorry

end max_rectangle_area_l813_813350


namespace expression_simplifies_to_one_l813_813541

-- Define x in terms of the given condition
def x : ℚ := (1 / 2) ^ (-1 : ℤ) + (-3) ^ (0 : ℤ)

-- Define the given expression
def expr (x : ℚ) : ℚ := (((x^2 - 1) / (x^2 - 2 * x + 1)) - (1 / (x - 1))) / (3 / (x - 1))

-- Define the theorem stating the equivalence
theorem expression_simplifies_to_one : expr x = 1 := by
  sorry

end expression_simplifies_to_one_l813_813541


namespace problem_A_problem_B_problem_C_problem_D_l813_813266

theorem problem_A : 2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5 := by
  sorry

theorem problem_B : 3 * Real.sqrt 3 * (3 * Real.sqrt 2) ≠ 3 * Real.sqrt 6 := by
  sorry

theorem problem_C : (Real.sqrt 27 / Real.sqrt 3) = 3 := by
  sorry

theorem problem_D : 2 * Real.sqrt 2 - Real.sqrt 2 ≠ 2 := by
  sorry

end problem_A_problem_B_problem_C_problem_D_l813_813266


namespace find_ON_l813_813065

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

theorem find_ON :
  ∀ (M F1 F2 O N : ℝ × ℝ),
    ellipse M.1 M.2 →
    distance M F1 = 2 →
    distance F1 O = 5 ∧ distance F2 O = 5 →
    F1 ≠ F2 →
    N = midpoint M F1 →
    distance O N = 4 :=
by
  sorry

end find_ON_l813_813065


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813820

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813820


namespace find_e_l813_813576

-- Define the conditions and state the theorem.
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) 
  (h1: ∃ a b c : ℝ, (a + b + c)/3 = -3 ∧ a * b * c = -3 ∧ 3 + d + e + f = -3)
  (h2: Q 0 d e f = 9) : e = -42 :=
by
  sorry

end find_e_l813_813576


namespace min_sum_chessboard_labels_l813_813584

theorem min_sum_chessboard_labels :
  ∃ (r : Fin 9 → Fin 9), 
  (∀ (i j : Fin 9), i ≠ j → r i ≠ r j) ∧ 
  ((Finset.univ : Finset (Fin 9)).sum (λ i => 1 / (r i + i.val + 1)) = 1) :=
by
  sorry

end min_sum_chessboard_labels_l813_813584


namespace shaniqua_earnings_l813_813539

noncomputable def shaniqua_total_earnings : ℕ :=
  let haircut_rate := 12
  let style_rate := 25
  let coloring_rate := 35
  let treatment_rate := 50
  let haircuts := 8
  let styles := 5
  let colorings := 10
  let treatments := 6
  (haircuts * haircut_rate) +
  (styles * style_rate) +
  (colorings * coloring_rate) +
  (treatments * treatment_rate)

theorem shaniqua_earnings : shaniqua_total_earnings = 871 := by
  sorry

end shaniqua_earnings_l813_813539


namespace sequence_behavior_l813_813507

theorem sequence_behavior {a : ℝ} (h₀ : 0 < a) (h₁ : a < 1) :
  let x : ℕ → ℝ := λ n, Nat.recOn n a (λ n xn, a ^ xn) in
  (∀ m : ℕ, odd m → x (m + 2) > x m) ∧ (∀ m : ℕ, even m → x (m + 2) < x m) :=
by
  sorry

end sequence_behavior_l813_813507


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813687

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813687


namespace fold_point_area_l813_813009

theorem fold_point_area (A B C P : ℝ × ℝ)
  (h1 : dist A B = 48)
  (h2 : dist A C = 96)
  (h3 : ∠ B = real.pi / 2)
  (h4 : lies_on_altitude B P A C)
  (h5 : folds_properly A B C P) :
  area_of_fold_points (A, B, C) = 771 := sorry

end fold_point_area_l813_813009


namespace sprinkles_remaining_l813_813341

/-- Given that Coleen started with twelve cans of sprinkles and after applying she had 3 less than half as many cans, prove the remaining cans are 3 --/
theorem sprinkles_remaining (initial_sprinkles : ℕ) (h_initial : initial_sprinkles = 12) (h_remaining : ∃ remaining_sprinkles, remaining_sprinkles = initial_sprinkles / 2 - 3) : ∃ remaining_sprinkles, remaining_sprinkles = 3 :=
by
  have half_initial := initial_sprinkles / 2
  have remaining_sprinkles := half_initial - 3
  use remaining_sprinkles
  rw [h_initial, Nat.div_eq_of_lt (by decide : 6 < 12)]
  sorry

end sprinkles_remaining_l813_813341


namespace max_value_of_y_over_x_on_circle_l813_813451

-- Definition of the circle
def circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - sqrt 3)^2 = 3

-- The theorem we need to prove
theorem max_value_of_y_over_x_on_circle :
  ∀ (x y : ℝ), circle x y → (y / x) ≤ sqrt 3 := by
sorry

end max_value_of_y_over_x_on_circle_l813_813451


namespace paper_net_removal_l813_813183

theorem paper_net_removal:
  ∃ net : list (list (option ℕ)),
  -- condition that the original net has 10 squares
  length (filter (λ x, x.is_some) (net.bind id)) = 10 ∧
  -- condition that one square is removed, leaving 9 squares
  length (filter (λ x, x.is_some) (net.bind id)) = 9 := 
sorry

end paper_net_removal_l813_813183


namespace isosceles_triangles_problem_l813_813986

noncomputable def isosceles_triangles_count_in_figure : ℕ :=
  let PQR_isosceles := isosceles_triangle P Q R 
  let bisects_angle := ∠PQR = 60 ∧ angle_bisector PQR S PR
  let ST_parallel_PQ := segment_parallel ST PQ
  let TU_parallel_PS := segment_parallel TU PS
  -- Proving the total count of isosceles triangles
  have six_isosceles_triangles : ∀ (P Q R S T U : Point), 
    PQR_isosceles ∧ bisects_angle ∧ ST_parallel_PQ ∧ TU_parallel_PS → 
    count_isosceles_triangles P Q R S T U = 6 :=
  sorry

theorem isosceles_triangles_problem (P Q R S T U : Point) 
  (H1 : isosceles_triangle P Q R)
  (H2 : ∠ PQR = 60)
  (H3 : segment_bisects_angle PQR S PR)
  (H4 : segment_parallel ST PQ)
  (H5 : segment_parallel TU PS) : 
  count_isosceles_triangles P Q R S T U = 6 :=
by
  apply six_isosceles_triangles
  repeat { constructor; assumption }

end isosceles_triangles_problem_l813_813986


namespace TT_is_translation_or_identity_l813_813480

variable {Plane : Type}
variable {a b c : Plane → Plane}
variable T : Plane → Plane

def reflection (p : Plane → Plane) : Plane → Plane := sorry

def T_def : Plane → Plane := reflection a ∘ reflection b ∘ reflection c

-- We are to prove that T composed with itself results in a translation or identity mapping
theorem TT_is_translation_or_identity :
  let T := reflection a ∘ reflection b ∘ reflection c in
  (T ∘ T) = (λ x, x) ∨
  ∃ v : Plane, (λ x, T (T x)) = (λ x, x + v) :=
sorry

end TT_is_translation_or_identity_l813_813480


namespace range_g_x_l813_813234

theorem range_g_x (a : ℝ) : 
  set.range (λ x : ℝ, if x ≤ 2 then 2^x - a else 0) = set.Ioo (-a) (4 - a) ∪ {4 - a} := 
begin
  sorry
end

end range_g_x_l813_813234


namespace card_stack_partition_l813_813244

theorem card_stack_partition (n k : ℕ) (cards : Multiset ℕ) (h1 : ∀ x ∈ cards, x ∈ Finset.range (n + 1)) (h2 : cards.sum = k * n!) :
  ∃ stacks : List (Multiset ℕ), stacks.length = k ∧ ∀ stack ∈ stacks, stack.sum = n! :=
sorry

end card_stack_partition_l813_813244


namespace range_of_x_l813_813968

theorem range_of_x (x y : ℝ) (h : x - 4 * real.sqrt y = 2 * real.sqrt (x - y)) : x ∈ {0} ∪ set.Icc 4 20 :=
sorry

end range_of_x_l813_813968


namespace total_worth_of_pattay_coins_l813_813528

def num_quarters_and_dimes := 
  ∃ (q d : ℕ), q + d = 25 ∧ (10 * q + 25 * d) - (25 * q + 10 * d) = 150

def coins_value := 15 * 3 + 250 / 100

theorem total_worth_of_pattay_coins : 
  num_quarters_and_dimes ∧ coins_value = 3.70 :=
begin
  sorry
end

end total_worth_of_pattay_coins_l813_813528


namespace isosceles_triangle_JMK_collinear_l813_813493

open EuclideanGeometry

variables {A B C D E H J K F G I M : Point}

/-- Definition: Given an isosceles triangle ABC with base AC, 
  points D and E such that CD = DE,
  H, J, K as the midpoints of DE, AE, BD respectively,
  the line through K parallel to AC intersects AB at I,
  and M is the intersection of IH and GF,
  prove that J, M, and K are collinear. -/
theorem isosceles_triangle_JMK_collinear
  (isosceles_triangle : is_isosceles_triangle A B C)
  (on_AC : D ∈ AC)
  (on_BC : E ∈ BC)
  (equal_segments : segment_length C D = segment_length D E)
  (midpoint_H : is_midpoint H D E)
  (midpoint_J : is_midpoint J A E)
  (midpoint_K : is_midpoint K B D)
  (parallel_line : is_parallel (line_through K I) (line_through A C))
  (intersection_M : is_intersection (line_through I H) (line_through G F) = M) :
  collinear J M K :=
sorry

end isosceles_triangle_JMK_collinear_l813_813493


namespace value_of_y_at_x_8_l813_813954

theorem value_of_y_at_x_8 (k : ℝ) (y : ℝ) (h1 : y = k * 64^(1/3)) (h2 : y = 4 * sqrt 3) : 
  ∃ y, y = k * 8^(1/3) ∧ y = 2 * sqrt 3 :=
by {
  have h3 : 64^(1/3) = 4 := sorry,
  have h4 : k = sqrt 3 := by {
    rw [h3, ←mul_right_inj' (ne_of_lt (sqrt_pos.2 (by norm_num)))],
    sorry,
  },
  have h5 : 8^(1/3) = 2 := sorry,
  use k * 2,
  split,
  { rw h5, },
  { rw h4, simp, }
}

end value_of_y_at_x_8_l813_813954


namespace product_of_segments_l813_813321

noncomputable def square_area_smaller : ℝ := 16
noncomputable def square_area_larger : ℝ := 25

def side_length_smaller : ℝ := real.sqrt square_area_smaller
def side_length_larger : ℝ := real.sqrt square_area_larger

def segment_ratio (x y : ℝ) : Prop := y = 3 * x ∧ x + y = side_length_larger

theorem product_of_segments (x y : ℝ) (h : segment_ratio x y) : x * y = 75 / 16 :=
by sorry

end product_of_segments_l813_813321


namespace part1_f0_k_part2_monotone_solution_set_part3_m_value_l813_813499

variables (a : ℝ) (k : ℝ) (x : ℝ) (m : ℝ)

-- Definitions and assumptions as per the conditions
def f (x : ℝ) : ℝ := k * a^x - a^(-x)
def g (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * m * f x

-- Assuming the given conditions
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom f_odd : ∀ x, f (-x) = - f x

theorem part1_f0_k :
  f 0 = 0 ∧ k = 1 :=
sorry

theorem part2_monotone_solution_set (f_one_pos : f 1 > 0) :
  (∀ x y, x ≤ y → f x ≤ f y) ∧ { x | f (x^2 + 2*x) + f (4 - x^2) > 0 } = { x | x > -2 } :=
sorry

theorem part3_m_value (f_one_eq_three_over_two : f 1 = 3 / 2) (min_g_at_least_one : ∀ x, x ≥ 1 → g x ≥ -2) :
  m = 2 :=
sorry

end part1_f0_k_part2_monotone_solution_set_part3_m_value_l813_813499


namespace even_sum_probability_l813_813130

def possible_outcomes_X : Set ℕ := {2, 5, 7}
def possible_outcomes_Y : Set ℕ := {2, 4, 6}
def possible_outcomes_Z : Set ℕ := {1, 2, 3, 4}

def count_even (n : ℕ) : bool := (n % 2 = 0)

theorem even_sum_probability :
  (∑ x in possible_outcomes_X, ∑ y in possible_outcomes_Y, ∑ z in possible_outcomes_Z, if count_even (x + y + z) then 1 else 0) /
  (∑ x in possible_outcomes_X, ∑ y in possible_outcomes_Y, ∑ z in possible_outcomes_Z, 1) = 1 / 2 :=
  sorry

end even_sum_probability_l813_813130


namespace correct_option_c_l813_813268

theorem correct_option_c (a : ℝ) : (-2 * a) ^ 3 = -8 * a ^ 3 :=
sorry

end correct_option_c_l813_813268


namespace probability_log2_floor_eq_l813_813537

-- Define the interval (0,1)
def interval (a b : ℝ) := {x : ℝ | a < x ∧ x < b}

-- Define a probability measure on the interval (0,1)
noncomputable def uniform (a b : ℝ) : Measure ℝ := sorry

-- Define the log base 2 floor function
def log2_floor (x : ℝ) : ℤ := Int.floor (Real.log x / Real.log 2)

-- Define the event that the log2_floor values of x and y are equal
def event (x y : ℝ) := log2_floor x = log2_floor y

-- Statement of the problem
theorem probability_log2_floor_eq :
  Measure.probability_of (interval 0 1) (λ x => Measure.probability_of (interval 0 1) (λ y => event x y) = 1 / 3 :=
sorry

end probability_log2_floor_eq_l813_813537


namespace find_max_side_length_l813_813788

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813788


namespace evaluate_expression_l813_813239

theorem evaluate_expression : (2014 - 2013) * (2013 - 2012) = 1 := 
by sorry

end evaluate_expression_l813_813239


namespace donors_are_B_and_D_l813_813825

section Donations

variables (A B C D : Prop) -- Whether each student made a donation
variables (hA : (C ∨ D)) -- Condition from A: "At least one of C and D has made a donation."
variables (hB : (¬(D ∧ A))) -- Condition from B: "At most one of D and A has made a donation."
variables (hC : ((A ∧ B) ∨ (B ∧ D) ∨ (A ∧ D))) -- Condition from C: "Among the three of you, at least two have made a donation."
variables (hD : (¬(A ∧ B) ∧ ¬(B ∧ C) ∧ ¬(A ∧ C))) -- Condition from D: "Among the three of you, at most two have made a donation."
variable (hSum : (A + B + C + D = 2)) -- Exactly two students have made a donation

-- Prove that B and D made a donation
theorem donors_are_B_and_D : B ∧ D :=
by sorry

end Donations

end donors_are_B_and_D_l813_813825


namespace intersecting_circles_l813_813094

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l813_813094


namespace max_side_length_of_triangle_l813_813759

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813759


namespace power_function_properties_l813_813925

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^3 - m + 1) * x^(1/2 * (1 - 8 * m - m^2))

theorem power_function_properties :
  (∀ (x : ℝ), x ≠ 0 → ∀ (m : ℝ), (m^3 - m + 1 = 1) ∧ (∀ (x : ℝ), f m x ≠ 0) →
  f 1 x = x^(-4) ∧ (∀ x : ℝ, x < 1/2 ∧ x ≠ -1))
:=
begin
  sorry
end

end power_function_properties_l813_813925


namespace negative_integer_is_C_l813_813829

def A := 3
def B := -1 / 3
def C := -6
def D := -1.5

theorem negative_integer_is_C:
  (C < 0) ∧ (C % 1 = 0) ∧ ¬((A < 0) ∧ (A % 1 = 0)) ∧ ¬((B < 0) ∧ (B % 1 = 0)) ∧ ¬((D < 0) ∧ (D % 1 = 0)) :=
by
  sorry

end negative_integer_is_C_l813_813829


namespace projection_of_diff_in_direction_of_a_l813_813050

-- Definitions of unit vector a and b, and the angle between them is 60 degrees
variables (a b : Vector ℝ) (unit_a : a.norm = 1) (unit_b : b.norm = 1) (theta : Real.anglePOS := π / 3)

-- Desired proof of the projection
theorem projection_of_diff_in_direction_of_a (h1 : a.inner b = (1/2)) : 
  (a - b).inner a = (1/2) :=
by
  have h2 : a.norm^2 = 1 := norm_sq_eq_one_of_norm_eq_one unit_a
  have h3 : b.norm^2 = 1 := norm_sq_eq_one_of_norm_eq_one unit_b
  sorry

end projection_of_diff_in_direction_of_a_l813_813050


namespace find_sum_of_abs_roots_l813_813387

variable {p q r n : ℤ}

theorem find_sum_of_abs_roots (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2024) (h3 : p * q * r = -n) :
  |p| + |q| + |r| = 100 :=
  sorry

end find_sum_of_abs_roots_l813_813387


namespace wifes_raise_l813_813866

variable (D W : ℝ)
variable (h1 : 0.08 * D = 800)
variable (h2 : 1.08 * D - 1.08 * W = 540)

theorem wifes_raise : 0.08 * W = 760 :=
by
  sorry

end wifes_raise_l813_813866


namespace range_of_a_l813_813035

noncomputable def decreasing_function (a : ℝ) : Prop :=
∀ x y : ℝ, x < y → f a x > f a y 

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≥ 0 then -x + 3 * a else a^x

theorem range_of_a (a : ℝ) :
  (decreasing_function a ↔ (1/3 ≤ a ∧ a < 1)) := sorry

end range_of_a_l813_813035


namespace perpendicular_vectors_l813_813964

def vector_a := (2, 0 : ℤ × ℤ)
def vector_b := (1, 1 : ℤ × ℤ)

theorem perpendicular_vectors:
  let v := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) in
  v.1 * vector_b.1 + v.2 * vector_b.2 = 0 :=
by
  sorry

end perpendicular_vectors_l813_813964


namespace cos_angle_VTU_l813_813120

variable (V T X U : Type) [InnerProductSpace ℝ V]
variable (cos_VTX : ℝ)
variable (angle_VTX : ℝ) (angle_VTU : ℝ)

-- Given condition: \( \cos \angle VTX = \frac{4}{5} \)
axiom cos_angle_VTX : cos_VTX = 4 / 5

-- Given condition: \(\angle VTU = 180^\circ - \angle VTX\)
axiom angle_VTU_supplementary : angle_VTU = π - angle_VTX

-- The goal is to show:  \(\cos \angle VTU = - \frac{4}{5} \)
theorem cos_angle_VTU : cos (π - angle_VTX) = - 4 / 5 :=
by
  -- Use the property: \( \cos (180^\circ - x) = - \cos x \)
  calc
    cos (π - angle_VTX) = - cos angle_VTX : by sorry
                                    -- substitution of cos_angle_VTX
                       ... = - (4 / 5) : by rw [cos_angle_VTX]
                       ... = - 4 / 5 : rfl

end cos_angle_VTU_l813_813120


namespace remainder_of_poly_div_l813_813884

noncomputable def p (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x + 1

theorem remainder_of_poly_div (
  x : ℝ
) : (p 2) = 7 :=
by
  dsimp [p]
  norm_num
  sorry

end remainder_of_poly_div_l813_813884


namespace compute_AQ_l813_813198

variable (A B C P Q R : Type) 
variables (d_AQ d_PC d_BP d_CQ : ℝ)
variables [metric_space E]
variables (is_right_triangle : is_right_triangle (△ABC))
variables (is_equilateral_triangle : is_equilateral_triangle (△PQR))

-- Conditions
noncomputable def length_PC : ℝ := 4
noncomputable def length_BP : ℝ := 3
noncomputable def length_CQ : ℝ := 3

theorem compute_AQ (h1 : length_PC = 4)
                   (h2 : length_BP = 3)
                   (h3 : length_CQ = 3) : 
      d_AQ = 3 / 4
:= sorry

end compute_AQ_l813_813198


namespace rational_sqrt_condition_l813_813437

variable (r q n : ℚ)

theorem rational_sqrt_condition
  (h : (1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q))) : 
  ∃ x : ℚ, x^2 = (n - 3) / (n + 1) :=
sorry

end rational_sqrt_condition_l813_813437


namespace rachel_plant_distribution_l813_813536

-- Define types for plants and lamps
inductive Plant 
| basil : Plant
| aloe : Plant
| cactus : Plant

inductive LampColor
| white : LampColor
| red : LampColor

open Plant LampColor

-- Define the set of plants
def plants : Finset Plant := {basil, basil, aloe, cactus}

-- Define the set of lamp colors with multiplicities
def lamps : Finset LampColor := {white, white, red, red}

-- State the theorem
theorem rachel_plant_distribution : 
  ∃ (f : Plant → LampColor), (∀ p ∈ plants, f p ∈ lamps) 
                             ∧ (∀ l ∈ lamps, (plants.filter (λ p, f p = l)).card ≤ 4)
                             ∧ (plants.image f).card = 4
                             ∧ 12 = (number_of_ways_to_distribute_plants 4 2 2) :=
sorry

-- Function to calculate the number of ways to distribute n plants under w white 
-- lamps and r red lamps (assuming they can share lamps).
noncomputable def number_of_ways_to_distribute_plants (n w r : ℕ) : ℕ :=
2 + 4 + 6 -- this is based on the solution steps provided


end rachel_plant_distribution_l813_813536


namespace area_of_parallelogram_l813_813280

open Real

-- Define two vectors p and q
variables (p q : ℝ^3)

-- Define the norm (magnitude) of the vectors p and q
variables (norm_p : ∥p∥ = 1) (norm_q : ∥q∥ = 2)

-- Define the angle between the vectors p and q
variables (angle_pq : ∠(p, q) = π / 6)

-- Define vectors a and b based on p and q
def a : ℝ^3 := p - 4 * q
def b : ℝ^3 := 3 * p + q

-- Define the cross product of a and b
def cross_product_ab : ℝ^3 := a ⬝ b

-- Define the magnitude of the cross product
def magnitude_cross_product_ab : ℝ := ∥cross_product_ab∥

-- Define the condition to check the area of the parallelogram
theorem area_of_parallelogram :
  magnitude_cross_product_ab = 13 :=
sorry

end area_of_parallelogram_l813_813280


namespace f_correct_g_correct_commutativity_additional_conditions_l813_813022

/-
  Definition of functions f and g under given conditions
-/
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := x^2 + (1/2)*x + 1/16

variables (x : ℝ)

/-
  Theorems to prove
-/

theorem f_correct (x : ℝ) : f x = x := 
by 
  -- sorry so as to skip the proof
  sorry

theorem g_correct (x : ℝ) : g x = x^2 + (1/2)*x + 1/16 := 
by 
  -- sorry so as to skip the proof
  sorry

/-
  The commutativity condition f[g(x)] = g[f(x)]
-/
theorem commutativity (x : ℝ) : f (g x) = g (f x) := 
by
  -- sorry so as to skip the proof
  sorry

/-
  Additional conditions on the coefficients
  g(0) = 1/16
  Tangent condition to the x-axis: 4ac = b²
  Tangent condition to the line x: (b-1)^2 = (1/4)a
-/
theorem additional_conditions (a b c : ℝ) (h1 : g 0 = 1/16) (h2 : 4 * a * c = b^2) (h3 : (b - 1)^2 = 1/4 * a) : 
  a = 1 ∧ b = 1/2 ∧ c = 1/16 :=
by 
  -- sorry so as to skip the proof
  sorry

end f_correct_g_correct_commutativity_additional_conditions_l813_813022


namespace participants_with_six_points_l813_813470

theorem participants_with_six_points (n : ℕ) (h : n = 9) : 
  let num_participants := 2^n,
      final_score_count (num_points : ℕ) := 0 ≤ num_points ∧ num_points ≤ n → 
        ∑ k in finset.range (n + 1), if k = 6 then 
          Nat.choose n k else 0 
  in final_score_count 6 = 84 :=
  sorry

end participants_with_six_points_l813_813470


namespace percentage_reduction_price_increase_for_profit_price_increase_max_profit_l813_813638

-- Define the conditions
def original_price : ℝ := 50
def final_price : ℝ := 32
def daily_sales : ℝ := 500
def profit_per_kg : ℝ := 10
def sales_decrease_per_yuan : ℝ := 20
def required_daily_profit : ℝ := 6000
def max_possible_profit : ℝ := 6125

-- Proving the percentage reduction each time
theorem percentage_reduction (a : ℝ) :
  (original_price * (1 - a) ^ 2 = final_price) → (a = 0.2) :=
sorry

-- Proving the price increase per kilogram to ensure a daily profit of 6000 yuan
theorem price_increase_for_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = required_daily_profit) → (x = 5) :=
sorry

-- Proving the price increase per kilogram to maximize daily profit
theorem price_increase_max_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = max_possible_profit) → (x = 7.5) :=
sorry

end percentage_reduction_price_increase_for_profit_price_increase_max_profit_l813_813638


namespace common_point_of_average_l813_813251

variable {x y : Type}
variable {n m : ℕ}

structure RegressionLine where
  linear_eq : (ℝ → ℝ)

structure DataSet where
  data_points : ℕ → (ℝ × ℝ)

def average (s : DataSet) (f : ℝ × ℝ → ℝ) : ℝ :=
  (∑ i in finset.range s.data_points.card, f (s.data_points i)) / s.data_points.card

theorem common_point_of_average (s t : ℝ) (data1 data2 : DataSet) 
  (r1 r2 : RegressionLine)
  (h1 : average data1 (λ p, p.fst) = s)
  (h2 : average data1 (λ p, p.snd) = t)
  (h3 : average data2 (λ p, p.fst) = s)
  (h4 : average data2 (λ p, p.snd) = t)
  (h5 : ∀ p, r1.linear_eq p.fst = p.snd)
  (h6 : ∀ p, r2.linear_eq p.fst = p.snd) :
  (r1.linear_eq s = t) ∧ (r2.linear_eq s = t) :=
sorry

end common_point_of_average_l813_813251


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813689

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813689


namespace polynomial_cubes_l813_813990

theorem polynomial_cubes (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ∃ f : ℤ[X], f.leadingCoeff > 0 ∧
    f.eval (a : ℤ) = (a^3 : ℤ) ∧
    f.eval (b : ℤ) = (b^3 : ℤ) ∧
    f.eval (c : ℤ) = (c^3 : ℤ) :=
by
  let f : ℤ[X] := (a+b+c)*X^2 - (a*b + b*c + c*a)*X + a*b*c
  have hfc : f.eval (a : ℤ) = (a^3 : ℤ), sorry
  have hfb : f.eval (b : ℤ) = (b^3 : ℤ), sorry
  have hfa : f.eval (c : ℤ) = (c^3 : ℤ), sorry
  exact ⟨f, sorry, hfc, hfb, hfa⟩

end polynomial_cubes_l813_813990


namespace exists_subsets_with_equal_sum_and_sum_of_squares_l813_813155

variable (S : Finset ℕ) (hS : ∀ n ∈ S, n < 10^100) (hS_size : S.card = 2000)

theorem exists_subsets_with_equal_sum_and_sum_of_squares :
  ∃ (A B : Finset ℕ), A ⊆ S ∧ B ⊆ S ∧ A.disjoint B ∧
                      A.card = B.card ∧ 
                      A.sum id = B.sum id ∧ 
                      (A.sum (λ x, x * x)) = B.sum (λ x, x * x) :=
sorry

end exists_subsets_with_equal_sum_and_sum_of_squares_l813_813155


namespace solve_x4_minus_16_eq_0_l813_813367

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2*complex.I, -2*complex.I} :=
sorry

end solve_x4_minus_16_eq_0_l813_813367


namespace quadratic_roots_real_and_equal_l813_813356

theorem quadratic_roots_real_and_equal :
  ∀ (a b c : ℝ), a ≠ 0 ∧ a = 1 ∧ b = -4 * real.sqrt 3 ∧ c = 12 →
  let Δ := b^2 - 4 * a * c in
  Δ = 0 → ∃ x : ℝ, (polynomial.X^2 + (b/a) * polynomial.X + (c/a)).eval x = 0 ∧ (polynomial.X^2 + (b/a) * polynomial.X + (c/a)).derivative.eval x = 0 :=
by
  intros a b c h Δ_eq_zero
  sorry

end quadratic_roots_real_and_equal_l813_813356


namespace Rohan_house_rent_l813_813199

variable (salary : ℕ) (foodPercent : ℕ) (entertainPercent : ℕ) (conveyPercent : ℕ) (savings : ℕ)

def houseRentPercent (salary foodPercent entertainPercent conveyPercent savings : ℕ) : ℕ :=
  40 - (savings * 100 / salary)

theorem Rohan_house_rent :
  salary = 12500 → foodPercent = 40 → entertainPercent = 10 → conveyPercent = 10 → savings = 2500 →
  houseRentPercent salary foodPercent entertainPercent conveyPercent savings = 20 :=
by
  intros hsal hfp hep hcp hsav
  simp [houseRentPercent]
  rw [hsal, hfp, hep, hcp, hsav]
  norm_num
  sorry

end Rohan_house_rent_l813_813199


namespace f_expression_l813_813042

def a (n : ℕ) (hn : 0 < n) : ℝ := 1 / (n + 1)^2
def f (n : ℕ) (hn : 0 < n) : ℝ :=
  ∏ i in finset.range n, (1 - a (i + 1) (nat.succ_pos i))

theorem f_expression (n : ℕ) (hn : 0 < n) : f n hn = (n + 2) / (2 * n + 2) := 
  sorry

end f_expression_l813_813042


namespace number_of_true_propositions_is_zero_l813_813830

-- Definitions for the propositions
def prop1 (a b : V) : Prop := 
  collinear a b → (∀ l, (contains l a ∧ contains l b) → parallel l)

def prop2 (a b : V) : Prop := 
  (skew_lines a b) → ¬ coplanar a b

def prop3 (a b c : V) : Prop :=
  (coplanar a b ∧ coplanar b c ∧ coplanar a c) → coplanar a b c

def prop4 (a b c : V) : Prop :=
  ∀ (p : V), ∃ (x y z : ℝ), p = x • a + y • b + z • c

-- Main theorem stating the number of true propositions
theorem number_of_true_propositions_is_zero (a b c : V) : 
  (¬ prop1 a b) ∧ (¬ prop2 a b) ∧ (¬ prop3 a b c) ∧ (¬ prop4 a b c) :=
sorry

end number_of_true_propositions_is_zero_l813_813830


namespace x_gt_neg2_is_necessary_for_prod_lt_0_l813_813359

theorem x_gt_neg2_is_necessary_for_prod_lt_0 (x : Real) :
  (x > -2) ↔ (((x + 2) * (x - 3)) < 0) → (x > -2) :=
by
  sorry

end x_gt_neg2_is_necessary_for_prod_lt_0_l813_813359


namespace max_side_of_triangle_l813_813808

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813808


namespace pizza_slices_left_l813_813190

theorem pizza_slices_left (initial_slices cuts friends total_given : ℕ) (h_slices: initial_slices = 1 * 2^cuts)
  (h_first_group: ∀ f ∈ friends, f = 2) (h_second_group: ∀ f ∉ friends, f = 1) 
  (h_friends: ∑ f in friends, f = total_given) 
  (distribution: total_given = (card friends * 2 + (card (range 3) - card friends) * 1)) 
  (h_card_friends: card friends = 2) : 
  (initial_slices - total_given = 1) :=
sorry

end pizza_slices_left_l813_813190


namespace partition_sum_equal_l813_813348

open Set Finset

def isPermutationOfDigits (n : ℕ) : Prop :=
  ∀ {a b c d e : ℕ}, List.Perm [a, b, c, d, e] [1, 2, 3, 4, 5] → 
  (n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)

theorem partition_sum_equal (S : Finset ℕ) :
  (∀ n ∈ S, isPermutationOfDigits n) →
  ∃ (A B : Finset ℕ), A ∪ B = S ∧ A ∩ B = ∅ ∧ 
  (∑ x in A, x^2) = (∑ x in B, x^2) :=
by
  sorry

end partition_sum_equal_l813_813348


namespace mia_12th_roll_last_is_approximately_027_l813_813957

noncomputable def mia_probability_last_roll_on_12th : ℚ :=
  (5/6) ^ 10 * (1/6)

theorem mia_12th_roll_last_is_approximately_027 : 
  abs (mia_probability_last_roll_on_12th - 0.027) < 0.001 :=
sorry

end mia_12th_roll_last_is_approximately_027_l813_813957


namespace variable_cost_per_book_l813_813662

theorem variable_cost_per_book
  (F : ℝ) (S : ℝ) (N : ℕ) (V : ℝ)
  (fixed_cost : F = 56430) 
  (selling_price_per_book : S = 21.75) 
  (num_books : N = 4180) 
  (production_eq_sales : S * N = F + V * N) :
  V = 8.25 :=
by sorry

end variable_cost_per_book_l813_813662


namespace sum_of_sequence_is_26_l813_813902

noncomputable def sequence (a : ℕ → ℤ) (h : ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = c) : ℕ → ℤ := sorry

theorem sum_of_sequence_is_26 
  (a : ℕ → ℤ)
  (h_const_sum : ∀ n, a n + a (n + 1) + a (n + 2) = c)
  (h_a3 : a 3 = 9)
  (h_a7 : a 7 = -7)
  (h_a98 : a 98 = -1) :
  (∑ i in Finset.range 101, a i) = 26 := 
sorry

end sum_of_sequence_is_26_l813_813902


namespace part_1_part_2_l813_813996

/--  Lean 4 proof statement for the given mathematical problem  --/

-- Define the set S of 10-tuples of non-negative integers summing to 2019
def S : Set (Fin 10 → ℕ) := {t | (∑ i, t i) = 2019}

-- Define the operation on a tuple A in S
def operation (A : Fin 10 → ℕ) (i : Fin 10) : Fin 10 → ℕ :=
  fun j => if i = j then A j - 9 else A j + 1

-- Define reachability between tuples A and B in S
def reachable_from (A B : Fin 10 → ℕ) : Prop :=
  ∃ n : ℕ, (λ x => operation x (Fin.ofNat 0))^[n] A = B

-- Define the smallest integer k such that if the minimum number in A, B ∈ S are both ≥ k 
-- then A → B implies B → A
def smallest_k : ℕ :=
  8

-- Define the number of tuples such that any two distinct tuples A, B that are distinct 
-- A → B does not hold
def num_tuples_k_8 : ℕ :=
  10 ^ 8

theorem part_1 : (∀ A B ∈ S, (∀ i : Fin 10, A i ≥ 8) ∧ (∀ i : Fin 10, B i ≥ 8) →
  (reachable_from A B ↔ reachable_from B A)) := sorry

theorem part_2 : (∀ k = smallest_k, num_tuples_k_8 = 10 ^ 8) := sorry

end part_1_part_2_l813_813996


namespace Jina_has_51_mascots_l813_813488

def teddies := 5
def bunnies := 3 * teddies
def koala_bear := 1
def additional_teddies := 2 * bunnies
def total_mascots := teddies + bunnies + koala_bear + additional_teddies

theorem Jina_has_51_mascots : total_mascots = 51 := by
  sorry

end Jina_has_51_mascots_l813_813488


namespace probability_complement_independent_l813_813026

variables {Ω : Type} [ProbabilitySpace Ω]
variable (A B : Event Ω)

theorem probability_complement_independent (h_independent : Independent A B)
    (h_PA : Probability A = 1 / 2) (h_PB : Probability B = 2 / 3) :
    Probability (¬ (A ∩ B)) = 1 / 6 :=
  sorry

end probability_complement_independent_l813_813026


namespace find_x_for_sin_cos_l813_813369

theorem find_x_for_sin_cos (x : ℝ) (h : 0 ≤ x ∧ x < 2 * Real.pi) : 
  (sin x - cos x = Real.sqrt 2) ↔ x = 3 * Real.pi / 4 :=
by sorry

end find_x_for_sin_cos_l813_813369


namespace problem1_problem2_problem3_l813_813629

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 - 3 * x = 2) : 1 + 2 * x^2 - 6 * x = 5 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : x^2 - 3 * x - 4 = 0) : 1 + 3 * x - x^2 = -3 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) (p q : ℝ) (h1 : x = 1 → p * x^3 + q * x + 1 = 5) (h2 : p + q = 4) (hx : x = -1) : p * x^3 + q * x + 1 = -3 :=
by
  sorry

end problem1_problem2_problem3_l813_813629


namespace max_triangle_side_l813_813712

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813712


namespace bob_walked_when_meet_l813_813180

theorem bob_walked_when_meet 
  (X Y : Type)
  (distance_XY : ℝ)
  (yolanda_rate bob_rate : ℝ)
  (yolanda_start bob_start : ℝ)
  (meet_time : ℝ)
  (distance_walking : ∀ (t : ℝ), yolanda_rate * (t + yolanda_start) + bob_rate * t = distance_XY)
  (distance_XY = 24)
  (yolanda_rate = 3)
  (bob_rate = 4)
  (yolanda_start = 1)
  (meet_time = 3)
: bob_rate * meet_time = 12 := 
sorry

end bob_walked_when_meet_l813_813180


namespace pencils_in_drawer_after_operations_l813_813248

def initial_pencils : ℝ := 2
def pencils_added : ℝ := 3.5
def pencils_removed : ℝ := 1.2

theorem pencils_in_drawer_after_operations : ⌊initial_pencils + pencils_added - pencils_removed⌋ = 4 := by
  sorry

end pencils_in_drawer_after_operations_l813_813248


namespace students_in_first_class_l813_813210

theorem students_in_first_class (x : ℕ)
    (h1 : 30 * x + 60 * 50 = 48.75 * (x + 50)) 
    : x = 30 :=
by
  -- Proof should be written here
  sorry

end students_in_first_class_l813_813210


namespace equation_of_perpendicular_line_l813_813215

theorem equation_of_perpendicular_line :
  ∃ (a b c : ℝ), (5, 3) ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  ∧ (a = 2 ∧ b = 1 ∧ c = -13)
  ∧ (a * 1 + b * (-2) = 0) :=
sorry

end equation_of_perpendicular_line_l813_813215


namespace coeff_x5_in_expansion_l813_813122

theorem coeff_x5_in_expansion : 
  let p := (1 - X^3) * (1 + X)^10 in
  coeff p 5 = 207 := 
by 
  sorry

end coeff_x5_in_expansion_l813_813122


namespace Nina_saves_enough_to_buy_video_game_in_11_weeks_l813_813523

-- Definitions (directly from conditions)
def game_cost : ℕ := 50
def tax_rate : ℚ := 10 / 100
def sales_tax (cost : ℕ) (rate : ℚ) : ℚ := cost * rate
def total_cost (cost : ℕ) (tax : ℚ) : ℚ := cost + tax
def weekly_allowance : ℕ := 10
def savings_rate : ℚ := 1 / 2
def weekly_savings (allowance : ℕ) (rate : ℚ) : ℚ := allowance * rate
def weeks_to_save (total_cost : ℚ) (savings_per_week : ℚ) : ℚ := total_cost / savings_per_week

-- Theorem to prove
theorem Nina_saves_enough_to_buy_video_game_in_11_weeks :
  weeks_to_save
    (total_cost game_cost (sales_tax game_cost tax_rate))
    (weekly_savings weekly_allowance savings_rate) = 11 := by
-- We skip the proof for now, as per instructions
  sorry

end Nina_saves_enough_to_buy_video_game_in_11_weeks_l813_813523


namespace triangle_inequality_and_condition_eq_triangle_l813_813970

theorem triangle_inequality_and_condition_eq_triangle :
  ∀ {A B C : ℝ}, 
  ∠A + ∠B + ∠C = π →
  ( (Real.csc (A / 2) + Real.csc (B / 2) + Real.csc (C / 2))^2 ≥ 
    9 + (Real.cot (A / 2) + Real.cot (B / 2) + Real.cot (C / 2))^2 ) ∧ 
  ( (Real.csc (A / 2) + Real.csc (B / 2) + Real.csc (C / 2))^2 = 
    9 + (Real.cot (A / 2) + Real.cot (B / 2) + Real.cot (C / 2))^2 ↔ equalAngles A B C ) := 
sorry

end triangle_inequality_and_condition_eq_triangle_l813_813970


namespace find_value_l813_813149

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity_condition : ∀ x : ℝ, f (2 + x) = f (-x)
axiom value_at_half : f (1/2) = 1/2

theorem find_value : f (2023 / 2) = 1/2 := by
  sorry

end find_value_l813_813149


namespace area_difference_correct_l813_813942

-- Define the conditions as current in the Lean statement
def radius_larger_circle := 30
def diameter_smaller_circle := 12
def radius_smaller_circle := diameter_smaller_circle / 2

-- Calculate the area of the circles
def area_larger_circle := (Real.pi : ℝ) * (radius_larger_circle ^ 2)
def area_smaller_circle := (Real.pi : ℝ) * (radius_smaller_circle ^ 2)

-- Define the target value which correct answer.
def target_area_difference := 864 * (Real.pi : ℝ)

-- Define the theorem statement
theorem area_difference_correct : (area_larger_circle - area_smaller_circle) = target_area_difference := by
  sorry

end area_difference_correct_l813_813942


namespace no_center_of_symmetry_for_2_abs_x_l813_813267

def has_center_of_symmetry (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, f (c - x) = f (c + x)

theorem no_center_of_symmetry_for_2_abs_x :
  ¬ has_center_of_symmetry (λ x : ℝ, 2^|x|) ∧
  has_center_of_symmetry (λ x : ℝ, 1 / (x + 1)) ∧
  has_center_of_symmetry (λ x : ℝ, x^3) ∧
  has_center_of_symmetry (λ x : ℝ, Real.tan x) :=
by
  sorry

end no_center_of_symmetry_for_2_abs_x_l813_813267


namespace limit_point_seq_a_limit_points_seq_b_limit_points_seq_c_limit_points_seq_d_limit_point_seq_e_limit_points_seq_f_l813_813384

open Set Filter

-- (a) \( x_n = \frac{n+1}{n} \)
theorem limit_point_seq_a : ∀ {ε : ℝ} (hε : ε > 0), ∃ N : ℕ, ∀ n : ℕ, n ≥ N → | ((n + 1) / n : ℝ) - 1 | < ε :=
by sorry

-- (b) \( x_n = (-1)^n \)
theorem limit_points_seq_b : ∀ L, L ∈ {-1, 1} ↔ ∃ (subseq : ℕ → ℕ), StrictMono subseq ∧ Tendsto (λ n, (-1 : ℤ) ^ subseq n) atTop (nhds L) :=
by sorry

-- (c) \( x_n = \sin(n^\circ) \)
theorem limit_points_seq_c : ∀ x, x ∈ (range (λ k : ℤ, sin (k : ℝ))) ↔ ∃ (subseq : ℕ → ℕ), StrictMono subseq ∧ Tendsto (λ n, sin (subseq n : ℝ)) atTop (nhds x) :=
by sorry

-- (d) \( x_n = n^{(-1)^{n}} \)
theorem limit_points_seq_d : ∀ L, L ∈ {0, ∞} ↔ ∃ (subseq : ℕ → ℕ), StrictMono subseq ∧ Tendsto (λ n, (subseq n) ^ ((-1) ^ (subseq n) : ℤ)) atTop (nhds L) :=
by sorry

-- (e) \( x_n = n \)
theorem limit_point_seq_e : ∀ (L : ℝ) (ε > 0), ∃ N : ℕ, ∀ n, n ≥ N → | (n.to_real) - L | < ε → L = ∞ :=
by sorry

-- (f) \( \frac{1}{2} ; \frac{1}{3} ; \frac{2}{3} ; \frac{1}{4} ; \frac{2}{4} ; \frac{3}{4} ; \frac{1}{5} ; \frac{2}{5} ; \frac{3}{5} ; \frac{4}{5} ; \ldots \)
theorem limit_points_seq_f : ∀ x, x ∈ Icc 0 1 ↔ ∃ (subseq : ℕ → ℕ × ℕ), (∀ n, (subseq n).fst < (subseq n).snd) ∧ StrictMono (Prod.snd ∘ subseq) ∧ Tendsto (λ n, (subseq n).fst / (subseq n).snd : ℝ) atTop (nhds x) :=
by sorry

end limit_point_seq_a_limit_points_seq_b_limit_points_seq_c_limit_points_seq_d_limit_point_seq_e_limit_points_seq_f_l813_813384


namespace triangle_angles_l813_813607

theorem triangle_angles (A B C P Q : Point) (hABC : ∠ACB = 90°)
  (hBCircle : is_circumcircle A B C)
  (hABisector : line A P = bisector A B C)
  (hBCond : BP = 2 * PQ) :
  ∠BAC = 30° ∧ ∠ABC = 60° ∧ ∠BCA = 90° :=
sorry

end triangle_angles_l813_813607


namespace factor_polynomials_l813_813873

theorem factor_polynomials (x : ℝ) :
  (x^2 + 4 * x + 3) * (x^2 + 9 * x + 20) + (x^2 + 6 * x - 9) = 
  (x^2 + 6 * x + 6) * (x^2 + 6 * x + 3) :=
sorry

end factor_polynomials_l813_813873


namespace value_of_a2021_l813_813124

noncomputable def sequence : ℕ → ℚ 
| 0       := -2 
| (n + 1) := 1 - (1 / sequence n)

theorem value_of_a2021 : sequence 2020 = 3 / 2 := 
by 
  sorry

end value_of_a2021_l813_813124


namespace sum_of_coefficients_6_l813_813412

theorem sum_of_coefficients_6
  (a : ℕ → ℤ)
  (h : ∀ x : ℤ, (2 - x)^7 = ∑ i in Finset.range 8, a i * (1 + x) ^ i) :
  (∑ i in Finset.range 7, a i) = 129 :=
by
  sorry

end sum_of_coefficients_6_l813_813412


namespace domain_log_base2_l813_813212

noncomputable def domain_of_log_function (x : ℝ) : Set ℝ :=
  {x | x > 4}

theorem domain_log_base2 (x : ℝ) : 
  (∃ x, x - 4 > 0) ->
  domain_of_log_function x = {x | x > 4} :=
by
  intro hx
  unfold domain_of_log_function
  rw Set.ext_iff
  intro y
  exact ⟨id, id⟩

end domain_log_base2_l813_813212


namespace area_of_region_l813_813373

noncomputable def region_area : ℝ :=
  ∫ x in 0..1, (3 + real.sqrt x + x^2)

theorem area_of_region :
  (∫ x in 0..1, (3 + real.sqrt x + x^2)) = 4 :=
by
  sorry

end area_of_region_l813_813373


namespace coeff_x2_term_l813_813877

def poly1 := 3 * x^3 + 4 * x^2 + 5 * x + 6
def poly2 := 7 * x^3 + 8 * x^2 + 9 * x + 10

theorem coeff_x2_term :
  coefficient (poly1 * poly2) x^2 = 93 :=
by
  -- Proof would be provided here
  sorry

end coeff_x2_term_l813_813877


namespace max_triangle_side_24_l813_813735

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813735


namespace possible_to_divide_polygon_with_cut_l813_813989

noncomputable def is_possible_divide_polygon : Prop :=
  ∃ (polygon : Polygon) (cut : Cut),
    (cut.lies_inside polygon) ∧
    (cut.endpoints_reach_boundary polygon) ∧
    (polygon.sides_follow_grid_lines) ∧
    (cut.segments_follow_grid_lines) ∧
    (cut.smaller_segments_half_of_longer) ∧
    (polygon.divided_into_equal_parts_by cut)

theorem possible_to_divide_polygon_with_cut :
  is_possible_divide_polygon :=
sorry

end possible_to_divide_polygon_with_cut_l813_813989


namespace projection_proof_l813_813233

-- Definitions and conditions based on the problem statement.
variable (w : ℝ × ℝ)
def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let a := (u.1 * v.1 + u.2 * v.2) 
  let b := (v.1 * v.1 + v.2 * v.2)
  ((a / b) * v.1, (a / b) * v.2)

def w_value : ℝ × ℝ := (2 / 5, 8 / 5)

-- Given condition
axiom condition : proj (1, 4) w_value = (2 / 5, 8 / 5)

-- Proof problem statement
theorem projection_proof : proj (5, -2) w_value = (-6 / 17, -24 / 17) := 
  sorry

end projection_proof_l813_813233


namespace sum_prime_factors_2_10_minus_1_l813_813591

theorem sum_prime_factors_2_10_minus_1 : 
  let n := 10 
  let number := 2^n - 1 
  let factors := [3, 5, 7, 11] 
  number.prime_factors.sum = 26 :=
by
  sorry

end sum_prime_factors_2_10_minus_1_l813_813591


namespace length_of_first_train_l813_813256

theorem length_of_first_train
  (speed_first_train_km_hr : ℕ)
  (speed_second_train_km_hr : ℕ)
  (cross_time_seconds : ℕ)
  (length_second_train_m : ℕ) :
  speed_first_train_km_hr = 36 →
  speed_second_train_km_hr = 54 →
  cross_time_seconds = 12 →
  length_second_train_m = 80 →
  let speed_first_train_m_s := (speed_first_train_km_hr * 5) / 18 in
  let speed_second_train_m_s := (speed_second_train_km_hr * 5) / 18 in
  let relative_speed_m_s := speed_first_train_m_s + speed_second_train_m_s in
  let total_distance_m := relative_speed_m_s * cross_time_seconds in
  let length_first_train_m := total_distance_m - length_second_train_m in
  length_first_train_m = 220 :=
by
  intros
  sorry

end length_of_first_train_l813_813256


namespace find_x_l813_813445

def vector := ℝ × ℝ

def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def orthogonal (v w : vector) : Prop :=
  dot_product v w = 0

def a : vector := (3, 1)
def c : vector := (0, 2)

noncomputable def b (x : ℝ) : vector := (x, -2)
noncomputable def d (x : ℝ) : vector := let ⟨bx, by⟩ := b(x) in (bx, by - 2) 

theorem find_x (x : ℝ) : orthogonal a (d x) → x = 4 / 3 := by
  sorry

end find_x_l813_813445


namespace sandy_marks_l813_813538

theorem sandy_marks (x : ℕ) 
  (h1 : ∑ i in (finset.range 30), 1 = 30) 
  (h2 : ∑ i in (finset.range 29), if i < 25 then x else -2 = 65): x = 3 :=
by
  sorry

end sandy_marks_l813_813538


namespace total_marks_l813_813135

theorem total_marks (Keith_marks Larry_marks Danny_marks : ℕ)
  (hK : Keith_marks = 3)
  (hL : Larry_marks = 3 * Keith_marks)
  (hD : Danny_marks = Larry_marks + 5) :
  Keith_marks + Larry_marks + Danny_marks = 26 := 
by
  sorry

end total_marks_l813_813135


namespace total_marks_secured_l813_813978

-- Define the conditions
def correct_points_per_question := 4
def wrong_points_per_question := 1
def total_questions := 60
def correct_questions := 40

-- Calculate the remaining incorrect questions
def wrong_questions := total_questions - correct_questions

-- Calculate total marks secured by the student
def total_marks := (correct_questions * correct_points_per_question) - (wrong_questions * wrong_points_per_question)

-- The statement to be proven
theorem total_marks_secured : total_marks = 140 := by
  -- This will be proven in Lean's proof assistant
  sorry

end total_marks_secured_l813_813978


namespace x_eq_one_sufficient_but_not_necessary_for_x_squared_eq_one_l813_813284

theorem x_eq_one_sufficient_but_not_necessary_for_x_squared_eq_one :
  ∀ x : ℝ, (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) := by
  intro x
  split
  · intro h
    rw h
    norm_num
  · intro h
    rw eq_comm at h
    exact sq_eq_one_iff.1 h

end x_eq_one_sufficient_but_not_necessary_for_x_squared_eq_one_l813_813284


namespace grasshopper_max_jumps_l813_813527

/-- On a real number line, the points 1, 2, 3, ..., 11 are marked. A grasshopper starts at point 1, 
then jumps to each of the other 10 marked points in some order so that no point is visited twice, 
before returning to point 1. The maximal length that he could have jumped in total is L, 
and there are N possible ways to achieve this maximum. Prove that L + N = 144060. -/
theorem grasshopper_max_jumps : 
  let points := (list.range 11).map (+1)
  let L := 60
  let N := 144000
  in L + N = 144060 := 
by
  simp [L, N]
  sorry

end grasshopper_max_jumps_l813_813527


namespace max_side_length_is_11_l813_813670

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813670


namespace max_triangle_side_l813_813723

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813723


namespace min_value_geom_seq_l813_813006

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  0 < a 4 ∧ 0 < a 14 ∧ a 4 * a 14 = 8 ∧ 0 < a 7 ∧ 0 < a 11 ∧ a 7 * a 11 = 8

theorem min_value_geom_seq {a : ℕ → ℝ} (h : geom_seq a) :
  2 * a 7 + a 11 = 8 :=
by
  sorry

end min_value_geom_seq_l813_813006


namespace relationship_between_a_and_x_l813_813959

theorem relationship_between_a_and_x 
  (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 + b^3 = 14 * x^3) 
  (h3 : a + b = x) :
  a = (sqrt 165 - 3) / 6 * x ∨ a = (-sqrt 165 - 3) / 6 * x :=
by
  sorry

end relationship_between_a_and_x_l813_813959


namespace integral_eq_exp_integral_eq_one_l813_813547

noncomputable
def y1 (τ : ℝ) (t : ℝ) (y : ℝ → ℝ) : Prop :=
  y τ = ∫ x in (0 : ℝ)..t, y x + 1

theorem integral_eq_exp (y : ℝ → ℝ) : 
  (∀ τ t, y1 τ t y) ↔ (∀ t, y t = Real.exp t) := 
  sorry

noncomputable
def y2 (t : ℝ) (y : ℝ → ℝ) : Prop :=
  ∫ x in (0 : ℝ)..t, y x * Real.sin (t - x) = 1 - Real.cos t

theorem integral_eq_one (y : ℝ → ℝ) : 
  (∀ t, y2 t y) ↔ (∀ t, y t = 1) :=
  sorry

end integral_eq_exp_integral_eq_one_l813_813547


namespace find_varphi_l813_813428

-- Define the problem conditions and the theorem
theorem find_varphi
  (omega : ℝ)
  (varphi : ℝ)
  (h1 : omega > 0)
  (h2 : abs varphi < (Real.pi / 2))
  (h3 : ∀ x1 x2 : ℝ, (2 * sin (omega * x1 + varphi) = 1) →
    (2 * sin (omega * x2 + varphi) = 1) →
    (abs (x2 - x1) = Real.pi / 3))
  (h4 : ∀ x : ℝ, 2 * sin (omega * x + varphi) ≤ 2 * sin (omega * (Real.pi / 12) + varphi)) :
  varphi = Real.pi / 3 :=
sorry

end find_varphi_l813_813428


namespace max_value_of_linear_combination_l813_813409

theorem max_value_of_linear_combination (x y : ℝ) (h : x^2 - 3 * x + 4 * y = 7) : 
  3 * x + 4 * y ≤ 16 :=
sorry

end max_value_of_linear_combination_l813_813409


namespace max_side_length_l813_813698

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813698


namespace value_of_a_l813_813028

variables {R : Type*} [Real R]

-- Conditions
def f (x : R) : R := log x
def g (x : R) : R := -f x

theorem value_of_a :
  ∀ (a : R), g(a) = 1 → a = (1 / exp(1)) := 
by
  sorry

end value_of_a_l813_813028


namespace triangle_ZM_ratio_l813_813987

theorem triangle_ZM_ratio:
  (XY YZ XZ : ℕ) -> XY = 5 -> YZ = 12 -> XZ = 13 ->
  (p : ℝ) -> (M divides XY in ratio 2:3) ->
  ZM = p * sqrt 2 -> p ≈ 3.7734 := by
  sorry

end triangle_ZM_ratio_l813_813987


namespace salt_added_correct_l813_813664

variables (x: ℝ) (y: ℝ) (salt_in_original: ℝ) (volume_after_evaporation: ℝ) (new_volume: ℝ) (salt_in_new_solution: ℝ)

def initial_salt_concentration := 0.20
def water_evaporated_fraction := 1/4
def final_salt_concentration := 1/3
def initial_solution_volume := 74.99999999999997
def added_water_volume := 5

theorem salt_added_correct :
  x = initial_solution_volume →
  x * initial_salt_concentration = salt_in_original →
  x * (1 - water_evaporated_fraction) = volume_after_evaporation →
  volume_after_evaporation + added_water_volume = new_volume →
  new_volume * final_salt_concentration = salt_in_new_solution →
  salt_in_original + y = salt_in_new_solution →
  y = 5.416666666666668 :=
by
  sorry

end salt_added_correct_l813_813664


namespace find_area_of_triangle_ABC_l813_813271

variables {A B C M : Type}
variables [add_comm_group A] [add_comm_group B] [add_comm_group C] [add_comm_group M]
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ M]

-- Conditions
def BC_length (BC : ℝ) : Prop := BC = 10
def CA_length (CA : ℝ) : Prop := CA = 12
def midpoint_M (M CA : ℝ) : Prop := M = CA / 2
def BM_parallel_external_bisector (BM : ℝ → Prop) (external_bisector : ℝ → Prop) : Prop := 
  ∀ x, BM x ↔ external_bisector x

-- Area of Triangle ABC
def area_triangle_ABC (area : ℝ) : Prop := area = 8 * sqrt 14

-- Lean statement equivalent to proving the mathematical problem
theorem find_area_of_triangle_ABC (BC CA M : ℝ) (BM : ℝ → Prop) (external_bisector : ℝ → Prop) :
  BC_length BC → CA_length CA → midpoint_M M CA → BM_parallel_external_bisector BM external_bisector → 
  area_triangle_ABC (8 * sqrt 14) :=
by
  sorry

end find_area_of_triangle_ABC_l813_813271


namespace vector_difference_perpendicular_l813_813967

/-- Proof that the vector difference a - b is perpendicular to b given specific vectors a and b -/
theorem vector_difference_perpendicular {a b : ℝ × ℝ} (h_a : a = (2, 0)) (h_b : b = (1, 1)) :
  (a - b) • b = 0 :=
by
  sorry

end vector_difference_perpendicular_l813_813967


namespace max_side_length_of_triangle_l813_813766

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813766


namespace set_intersection_l813_813046

open Set

/-- Given sets M and N as defined below, we wish to prove that their complements and intersections work as expected. -/
theorem set_intersection (R : Set ℝ)
  (M : Set ℝ := {x | x > 1})
  (N : Set ℝ := {x | abs x ≤ 2})
  (R_universal : R = univ) :
  ((compl M) ∩ N) = Icc (-2 : ℝ) (1 : ℝ) := by
  sorry

end set_intersection_l813_813046


namespace children_left_on_bus_l813_813286

-- Definitions based on the conditions
def initial_children := 43
def children_got_off := 22

-- The theorem we want to prove
theorem children_left_on_bus (initial_children children_got_off : ℕ) : 
  initial_children - children_got_off = 21 :=
by
  sorry

end children_left_on_bus_l813_813286


namespace f_is_odd_l813_813329

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd : is_odd_function f :=
by sorry

end f_is_odd_l813_813329


namespace average_of_remaining_students_l813_813554

-- Conditions
variables (students total_students mark excluded_students avg original_sum excluded_sum remaining_sum remaining_students new_avg : ℕ)

-- Hypotheses
def hypotheses :=
  total_students = 30 ∧
  avg = 80 ∧
  students = 5 ∧
  mark = 30 ∧
  original_sum = avg * total_students ∧
  excluded_sum = students * mark ∧
  remaining_sum = original_sum - excluded_sum ∧
  remaining_students = total_students - students ∧
  new_avg = remaining_sum / remaining_students

-- Claim
theorem average_of_remaining_students : hypotheses → new_avg = 90 :=
by
  intro h
  sorry

end average_of_remaining_students_l813_813554


namespace max_side_of_triangle_l813_813800

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813800


namespace domain_of_sqrt_1_minus_x_l813_813566

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def domain_sqrt (x : ℝ) : ℝ := sqrt (1 - x)

theorem domain_of_sqrt_1_minus_x :
  ∀ x : ℝ, (1 - x >= 0) ↔ (x <= 1) := 
by
  sorry

end domain_of_sqrt_1_minus_x_l813_813566


namespace minimum_a_l813_813416

theorem minimum_a (a b x : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : b - a = 2013) (h₃ : x > 0) (h₄ : x^2 - a * x + b = 0) : a = 93 :=
by
  sorry

end minimum_a_l813_813416


namespace zane_spent_more_on_cookies_l813_813229

theorem zane_spent_more_on_cookies
  (o c : ℕ) -- number of Oreos and cookies
  (pO pC : ℕ) -- price of each Oreo and cookie
  (h_ratio : 9 * o = 4 * c) -- ratio condition
  (h_price_O : pO = 2) -- price of Oreo
  (h_price_C : pC = 3) -- price of cookie
  (h_total : o + c = 65) -- total number of items
  : 3 * c - 2 * o = 95 := 
begin
  sorry
end

end zane_spent_more_on_cookies_l813_813229


namespace max_side_length_of_triangle_l813_813765

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813765


namespace sum_of_prime_factors_of_2_to_10_minus_1_l813_813589

theorem sum_of_prime_factors_of_2_to_10_minus_1 :
  let n := 2^10 - 1,
      factors := [31, 3, 11] in
  (n = factors.prod) ∧ (factors.all Prime) → factors.sum = 45 :=
by
  let n := 2^10 - 1
  let factors := [31, 3, 11]
  have fact_prod : n = factors.prod := by sorry
  have all_prime : factors.all Prime := by sorry
  have sum_factors : factors.sum = 45 := by sorry
  exact ⟨fact_prod, all_prime, sum_factors⟩

end sum_of_prime_factors_of_2_to_10_minus_1_l813_813589


namespace unique_squares_contacted_l813_813634

/-
  The block has dimensions 1 x 2 x 3 cm.
  The faces are marked X, Y, and Z.
  The block starts with face X facing down in contact with the board.
  The block is rotated 90 degrees around one of its edges so that face Y is facing down.
  The block is rotated another 90 degrees around one of its edges so that face Z is facing down.
  The block is rotated three more times by 90 degrees making faces X, Y, and Z face down in that order.
  The board is an 8 x 8 cm square.
-/

def block (dim1 dim2 dim3 : ℕ) := (dim1, dim2, dim3)

def faces := { X := 1 * 2, Y := 1 * 3, Z := 2 * 3 : ℕ }

def initial_position := faces.X

def rotations : ℕ := 5

def board_size := 8 * 8

theorem unique_squares_contacted : ∃ n : ℕ, n = 19 :=
by
  -- Mathematical proof would go here
  sorry

end unique_squares_contacted_l813_813634


namespace peter_initial_money_l813_813182

def cost_potatoes (k: ℕ) (p: ℕ) : ℕ := k * p
def cost_tomatoes (k: ℕ) (p: ℕ) : ℕ := k * p
def cost_cucumbers (k: ℕ) (p: ℕ) : ℕ := k * p
def cost_bananas (k: ℕ) (p: ℕ) : ℕ := k * p

def total_cost := cost_potatoes 6 2 + cost_tomatoes 9 3 + cost_cucumbers 5 4 + cost_bananas 3 5

theorem peter_initial_money : total_cost + 426 = 500 := by
  have h1 : cost_potatoes 6 2 = 12 := by rfl
  have h2 : cost_tomatoes 9 3 = 27 := by rfl
  have h3 : cost_cucumbers 5 4 = 20 := by rfl
  have h4 : cost_bananas 3 5 = 15 := by rfl
  have h_total : total_cost = 12 + 27 + 20 + 15 := by
    unfold total_cost
    rw [h1, h2, h3, h4]
    rfl
  have h_total_sum : total_cost = 74 := by
    rw [h_total]
    norm_num
  show total_cost + 426 = 500 from
    calc 
      total_cost + 426 = 74 + 426 : by rw [h_total_sum]
      ... = 500 : by norm_num

end peter_initial_money_l813_813182


namespace part_one_part_two_l813_813043

variable (a : ℝ)
def A : Set ℝ := {x | -3 < 2 * x + 1 ∧ 2 * x + 1 < 7}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C : Set ℝ := {x | 3 * a - 2 < x ∧ x < a + 1}
def CR (S T : Set ℝ) : Set ℝ := S ∩ Tᶜ

theorem part_one : A ∩ (CR B) = {x : ℝ | -2 < x ∧ x ≤ 2} := by
  sorry

theorem part_two (h : CR (A ∪ B) ⊆ C) : -3 < a ∧ a < -2 / 3 := by
  sorry

end part_one_part_two_l813_813043


namespace positive_diff_largest_prime_factors_204204_l813_813263

theorem positive_diff_largest_prime_factors_204204 :
  let largest_prime_factors (n : ℕ) (a b : ℕ) := (∀ p : ℕ, Prime p ∧ p ∣ n → p ≤ a) ∧ (∀ q : ℕ, Prime q ∧ q ∣ n → q ≤ b) ∧ a ≠ b ∧ a > b in
  let positive_difference (a b : ℕ) := a - b in
  (∃ a b : ℕ, largest_prime_factors 204204 a b ∧ positive_difference a b = 16) :=
by
  sorry

end positive_diff_largest_prime_factors_204204_l813_813263


namespace complex_circle_intersection_l813_813108

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l813_813108


namespace six_digit_count_l813_813940

namespace NumberTheory

def digits := {1, 2, 3, 4, 5, 6, 7, 9}

def six_digit_numbers := digits × digits × digits × digits × digits × digits

theorem six_digit_count : 
  ∃ n : ℕ, n = 262144 ∧ n = Fintype.card six_digit_numbers :=
by
  sorry

end NumberTheory

end six_digit_count_l813_813940


namespace find_abc_sum_l813_813502

-- Define the given equation
def equation (x : ℝ) : Prop :=
  (3 / (x - 3) + 5 / (x - 5) + 17 / (x - 17) + 19 / (x - 19) = x^2 - 11 * x - 4)

-- Define m as the largest real solution to the equation
def largest_real_solution (m : ℝ) : Prop :=
  equation m ∧ ∀ x, equation x → x ≤ m

-- Define m in terms of a, b, c
def m_formula (m a b c : ℝ) : Prop :=
  m = a + real.sqrt (b + real.sqrt c)

-- Final theorem statement
theorem find_abc_sum (a b c m : ℝ) (ha : a = 11) (hb : b = 52) (hc : c = 200)
  (hm : largest_real_solution m) (h_formula : m_formula m a b c) :
  a + b + c = 263 :=
by
  sorry

end find_abc_sum_l813_813502


namespace dot_product_MN_MO_is_8_l813_813422

-- Define the circle O as a set of points (x, y) such that x^2 + y^2 = 9
def is_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the length of the chord MN in the circle
def chord_length (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  (x1 - x2)^2 + (y1 - y2)^2 = 16

-- Define the vector MN and MO
def vector_dot_product (M N O : ℝ × ℝ) : ℝ :=
  let (x1, y1) := M
  let (x2, y2) := N
  let (x0, y0) := O
  let v1 := (x2 - x1, y2 - y1)
  let v2 := (x0 - x1, y0 - y1)
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the origin point O (center of the circle)
def O : ℝ × ℝ := (0, 0)

-- The theorem to prove
theorem dot_product_MN_MO_is_8 (M N : ℝ × ℝ) (hM : is_circle M.1 M.2) (hN : is_circle N.1 N.2) (hMN : chord_length M N) :
  vector_dot_product M N O = 8 :=
sorry

end dot_product_MN_MO_is_8_l813_813422


namespace max_triangle_side_24_l813_813727

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813727


namespace find_vector_result_l813_813934

-- Define the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m: ℝ) : ℝ × ℝ := (-2, m)
def m := -4
def result := 2 • vector_a + 3 • vector_b m

-- State the theorem
theorem find_vector_result : result = (-4, -8) := 
by {
  -- skipping the proof
  sorry
}

end find_vector_result_l813_813934


namespace find_x_l813_813157

def x_condition (x : ℤ) : Prop :=
  (120 ≤ x ∧ x ≤ 150) ∧ (x % 5 = 2) ∧ (x % 6 = 5)

theorem find_x :
  ∃ x : ℤ, x_condition x ∧ x = 137 :=
by
  sorry

end find_x_l813_813157


namespace a_2023_eq_3_5_l813_813580

-- Define the function that generates the sequence based on given rules
def seq (a_n : ℝ) : ℝ := 
  if 0 ≤ a_n ∧ a_n < 0.5 then 2 * a_n 
  else 2 * a_n - 1

-- Define the sequence
noncomputable def a : ℕ → ℝ
| 0     := 2 / 5
| (n+1) := seq (a n)

-- Define the proposition to be proved
theorem a_2023_eq_3_5 : a 2023 = 3 / 5 :=
sorry

end a_2023_eq_3_5_l813_813580


namespace geometric_sequence_common_ratio_l813_813397

open Nat

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n : ℕ, a (n + 1) = a n * q

-- Definition of the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms n + a (n + 1)

theorem geometric_sequence_common_ratio 
    (h1 : a 1 + a 3 = 6)
    (h2 : sum_of_first_n_terms a 4 + a 2 = sum_of_first_n_terms a 3 + 3)
    (hg : is_geometric_sequence a q) :
    q = 1 / 2 :=
sorry

end geometric_sequence_common_ratio_l813_813397


namespace max_side_of_triangle_l813_813749

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813749


namespace fixed_point_for_line_l813_813524

theorem fixed_point_for_line (m : ℝ) :
  let line_eq := (2 * m - 1) * 2 + (m + 3) * (-3) - (m - 11) in
  line_eq = 0 :=
by
  sorry

end fixed_point_for_line_l813_813524


namespace polynomial_roots_ratio_l813_813255

theorem polynomial_roots_ratio (a b c d : ℝ) (h₀ : a ≠ 0) 
    (h₁ : a * 64 + b * 16 + c * 4 + d = 0)
    (h₂ : -a + b - c + d = 0) : 
    (b + c) / a = -13 :=
by {
    sorry
}

end polynomial_roots_ratio_l813_813255


namespace five_points_distance_ratio_ge_two_sin_54_l813_813905

theorem five_points_distance_ratio_ge_two_sin_54
  (points : Fin 5 → ℝ × ℝ)
  (distinct : Function.Injective points) :
  let distances := {d : ℝ | ∃ (i j : Fin 5), i ≠ j ∧ d = dist (points i) (points j)}
  ∃ (max_dist min_dist : ℝ), max_dist ∈ distances ∧ min_dist ∈ distances ∧ max_dist / min_dist ≥ 2 * Real.sin (54 * Real.pi / 180) := by
  sorry

end five_points_distance_ratio_ge_two_sin_54_l813_813905


namespace smallest_norm_l813_813146

noncomputable def vectorNorm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_norm (v : ℝ × ℝ)
  (h : vectorNorm (v.1 + 4, v.2 + 2) = 10) :
  vectorNorm v >= 10 - 2 * Real.sqrt 5 :=
by
  sorry

end smallest_norm_l813_813146


namespace incorrect_statement_B_l813_813332

-- Conditions extracted from the problem description
variables {r : ℝ}
def non_deterministic_relationship : Prop := ∀ x y, ¬ (function.deterministic x y)
def correlation_range : Prop := |r| ≤ 1
def positive_or_negative_correlation : Prop := (r > 0 ∨ r < 0)
def perfect_correlation (x y : ℝ) : Prop := r = 1 ∨ r = -1

-- Statement to be proven
theorem incorrect_statement_B : ¬ (r ∈ set.Ioo (-1 : ℝ) (1 : ℝ)) :=
by
  sorry

end incorrect_statement_B_l813_813332


namespace general_formula_sum_bound_l813_813439

noncomputable def S (n : ℕ) : ℝ := 2 * a n + (-1) ^ n

theorem general_formula (n : ℕ) (hn : n ≥ 1) :
  let a := (fun n : ℕ, (2 / 3) * (2 ^ (n - 2) + (-1) ^ (n - 1))) in
  S n = 2 * a n + (-1) ^ n := sorry

theorem sum_bound (m : ℕ) (hm : m > 4) :
  let a := (fun n : ℕ, (2 / 3) * (2 ^ (n - 2) + (-1) ^ (n - 1))) in
  ∑ (i : ℕ) in Finset.range (m - 4 + 1) + 4, 1 / a i < 7 / 8 := sorry

end general_formula_sum_bound_l813_813439


namespace cross_country_meet_winning_scores_l813_813075

theorem cross_country_meet_winning_scores : 
  (∃ n : ℕ, n = 18) ∧ 
  ∀ (scores : Finset ℕ), 
  (∀ (pos : ℕ), pos ∈ scores → pos ≥ 1 ∧ pos ≤ 12) → 
  (∑ x in scores, x) < 39 → 
  21 ≤ (∑ x in scores, x) → 
  scores.card = 6 → 
  (∃ possible_scores, possible_scores = Finset.range 18 ∧ 
  (∀ score ∈ possible_scores, score ≥ 21 ∧ score ≤ 38)) :=
by 
  sorry

end cross_country_meet_winning_scores_l813_813075


namespace yellow_side_probability_correct_l813_813291

-- Define the problem scenario
structure CardBox where
  total_cards : ℕ := 8
  green_green_cards : ℕ := 4
  green_yellow_cards : ℕ := 2
  yellow_yellow_cards : ℕ := 2

noncomputable def yellow_side_probability 
  (box : CardBox)
  (picked_is_yellow : Bool) : ℚ :=
  if picked_is_yellow then
    let total_yellow_sides := 2 * box.green_yellow_cards + 2 * box.yellow_yellow_cards
    let yellow_yellow_sides := 2 * box.yellow_yellow_cards
    yellow_yellow_sides / total_yellow_sides
  else 0

theorem yellow_side_probability_correct :
  yellow_side_probability {total_cards := 8, green_green_cards := 4, green_yellow_cards := 2, yellow_yellow_cards := 2} true = 2 / 3 :=
by 
  sorry

end yellow_side_probability_correct_l813_813291


namespace parabola_equivalence_l813_813330

theorem parabola_equivalence :
  ∃ (a : ℝ) (h k : ℝ),
    (a = -3 ∧ h = -1 ∧ k = 2) ∧
    ∀ (x : ℝ), (y = -3 * x^2 + 1) → (y = -3 * (x + 1)^2 + 2) :=
sorry

end parabola_equivalence_l813_813330


namespace lambda_value_l813_813921

theorem lambda_value (p : ℝ) (hp : 0 < p) (A B F : ℝ × ℝ) 
  (hF : F = (p / 2, 0))
  (hA : A = (p, 2 * sqrt 2 * (p - p / 2))) 
  (hB : B = (p / 4, 2 * sqrt 2 * (p / 4 - p / 2))) 
  (hline : ∀ x : ℝ, (2 * sqrt 2 * (x - p / 2))^2 = 2 * p * x) : 
  ∃ λ : ℝ, λ = 2 ∨ λ = 1 / 2 :=
begin
  -- skipped proof
  sorry
end

end lambda_value_l813_813921


namespace range_of_a_l813_813040

noncomputable def f (x a : ℝ) : ℝ := x * (a - 1 / (Real.exp x))

theorem range_of_a (a : ℝ) :
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ deriv (λ x, f x a) x1 = 0 ∧ deriv (λ x, f x a) x2 = 0) →
  (-1 / (Real.exp 2) < a ∧ a < 0) :=
sorry

end range_of_a_l813_813040


namespace max_triangle_side_l813_813714

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813714


namespace _l813_813900

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  x^2 = 4 * y

noncomputable def lambda_range (λ : ℝ) : Prop :=
  1/2 ≤ λ ∧ λ ≤ 2/3

noncomputable def slope_range (k : ℝ) : Prop :=
  -Real.sqrt 2 / 4 ≤ k ∧ k ≤ Real.sqrt 2 / 4

noncomputable theorem minimum_product_areas :
  (∀ x y, parabola_equation x y) ∧
  (∀ F k (A B P : ℝ → ℝ), line_through_point_with_slope (0, 1) k (A, B, P)) ∧
  (∀ λ, lambda_range λ) →
  slope_range k ∧
  ∃ P, minimum_triangle_product_area P = 1 :=
sorry

end _l813_813900


namespace find_a_l813_813894

def valid_a (n T : ℕ) (a : ℕ) : Prop :=
  ∀ (a_i : Fin n.succ → ℕ) (h : ∀ i, a_i i > 0),
    (∑ k in Finset.range n, (a * (k+1) + a^2 / 4) / (∑ i in Finset.range (k+1), a_i i)) <
    T^2 * (∑ k in Finset.range n, 1 / a_i k)

theorem find_a (n T : ℕ) (h1 : 2 ≤ n) (h2 : 2 ≤ T) : (valid_a n T 1) ∧ (valid_a n T 2) :=
sorry

end find_a_l813_813894


namespace fraction_of_termite_ridden_homes_collapsing_l813_813525

variable (T : ℕ) -- T represents the total number of homes
variable (termiteRiddenFraction : ℚ := 1/3) -- Fraction of homes that are termite-ridden
variable (termiteRiddenNotCollapsingFraction : ℚ := 1/7) -- Fraction of homes that are termite-ridden but not collapsing

theorem fraction_of_termite_ridden_homes_collapsing :
  termiteRiddenFraction - termiteRiddenNotCollapsingFraction = 4/21 :=
by
  -- Proof goes here
  sorry

end fraction_of_termite_ridden_homes_collapsing_l813_813525


namespace max_triangle_side_24_l813_813734

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813734


namespace enclosed_area_eq_pi_add_16_l813_813615

theorem enclosed_area_eq_pi_add_16 :
  ∀ (x y : ℝ), x^2 + y^2 = 2 * (|x| + |y|) → area (region_enclosed_by_graph x y) = π + 16 :=
by
  sorry

end enclosed_area_eq_pi_add_16_l813_813615


namespace candies_problem_l813_813131

theorem candies_problem (emily jennifer bob : ℕ) (h1 : emily = 6) 
  (h2 : jennifer = 2 * emily) (h3 : jennifer = 3 * bob) : bob = 4 := by
  -- Lean code to skip the proof
  sorry

end candies_problem_l813_813131


namespace period_of_cosine_function_l813_813839

theorem period_of_cosine_function {a b c d : ℝ} (h : ∀ x : ℝ, y = a * cos (b * x + c) + d) 
  (period_condition : ∀ x : ℝ, y x ∈ [-2π, 2π] → y x = y (x + π)) : b = 2 := 
sorry

end period_of_cosine_function_l813_813839


namespace total_ducats_is_160_l813_813892

variable (T : ℤ) (a b c d e : ℤ) -- Variables to represent the amounts taken by the robbers

-- Conditions
axiom h1 : a = 81                                            -- The strongest robber took 81 ducats
axiom h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e    -- Each remaining robber took a different amount
axiom h3 : a + b + c + d + e = T                             -- Total amount of ducats
axiom redistribution : 
  -- Redistribution process leads to each robber having the same amount
  2*b + 2*c + 2*d + 2*e = T ∧
  2*(2*c + 2*d + 2*e) = T ∧
  2*(2*(2*d + 2*e)) = T ∧
  2*(2*(2*(2*e))) = T

-- Proof that verifies the total ducats is 160
theorem total_ducats_is_160 : T = 160 :=
by
  sorry

end total_ducats_is_160_l813_813892


namespace min_valid_subset_card_eq_l813_813582

open Finset

def pairs (n : ℕ) : Finset (ℕ × ℕ) := 
  (range n).product (range n)

def valid_subset (X : Finset (ℕ × ℕ)) (n : ℕ) : Prop :=
  ∀ (seq : ℕ → ℕ), ∃ k, (seq k, seq (k+1)) ∈ X

theorem min_valid_subset_card_eq (n : ℕ) (h : n = 10) : 
  ∃ X : Finset (ℕ × ℕ), valid_subset X n ∧ X.card = 55 := 
by 
  sorry

end min_valid_subset_card_eq_l813_813582


namespace correct_statements_l813_813127

noncomputable def triangle_conditions := 
  { (a b : ℝ) (c A B : ℝ) | 
      (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (A > 0) ∧ (B > 0) ∧ (A < π) ∧ (B < π) ∧ 
      (a = 3 * √3 ∧ b = 3 ∧ B = π / 6) ∨ 
      (A > B) ∨ 
      (c / b < Real.cos A) ∨ 
      (a = √2 ∧ b = 3 ∧ c^2 + a * b = a^2 + b^2) }

theorem correct_statements (A B : ℝ) (sin : ℝ → ℝ) (cos : ℝ → ℝ) (a b c : ℝ) (H : triangle_conditions a b c A B) :
  (A > B → sin A > sin B) ∧ (c / b < cos A → a^2 + c^2 < b^2) := 
by
  sorry

end correct_statements_l813_813127


namespace fisherman_daily_earnings_l813_813219

theorem fisherman_daily_earnings :
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 :=
by
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  show red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52
  sorry

end fisherman_daily_earnings_l813_813219


namespace proof_problem_l813_813018

def p := ∃ x0 : ℝ, tan x0 = 2
def q := ∀ x : ℝ, x^2 + 2 * x + 1 > 0

theorem proof_problem : p ∧ ¬q := by
  -- proof logic will be inserted here
  sorry

end proof_problem_l813_813018


namespace smallest_natural_with_twenty_divisors_l813_813236

theorem smallest_natural_with_twenty_divisors : ∃ n : ℕ, (n.divisors.count = 20) ∧ 
  ∀ m : ℕ, (m.divisors.count = 20) → n ≤ m :=
begin
  sorry
end

end smallest_natural_with_twenty_divisors_l813_813236


namespace ted_speed_l813_813552

variables (T F : ℝ)

-- Ted runs two-thirds as fast as Frank
def condition1 : Prop := T = (2 / 3) * F

-- In two hours, Frank runs eight miles farther than Ted
def condition2 : Prop := 2 * F = 2 * T + 8

-- Prove that Ted runs at a speed of 8 mph
theorem ted_speed (h1 : condition1 T F) (h2 : condition2 T F) : T = 8 :=
by
  sorry

end ted_speed_l813_813552


namespace similar_segments_areas_proportional_to_chords_squares_l813_813195

variables {k k₁ Δ Δ₁ r r₁ a a₁ S S₁ : ℝ}

-- Conditions given in the problem
def similar_segments (r r₁ a a₁ Δ Δ₁ k k₁ : ℝ) :=
  (Δ / Δ₁ = (a^2 / a₁^2) ∧ (Δ / Δ₁ = r^2 / r₁^2)) ∧ (k / k₁ = r^2 / r₁^2)

-- Given the areas of the segments in terms of sectors and triangles
def area_of_segment (k Δ : ℝ) := k - Δ

-- Theorem statement proving the desired relationship
theorem similar_segments_areas_proportional_to_chords_squares
  (h : similar_segments r r₁ a a₁ Δ Δ₁ k k₁) :
  (S = area_of_segment k Δ) → (S₁ = area_of_segment k₁ Δ₁) → (S / S₁ = a^2 / a₁^2) :=
by
  sorry

end similar_segments_areas_proportional_to_chords_squares_l813_813195


namespace sequence_formula_l813_813354

open Int

def floor_function (x : ℝ) : ℤ := floor x

def f (x : ℝ) : ℤ := floor (x * (floor_function x))

noncomputable def a (n : ℕ) : ℕ := n * (n - 1) / 2 + 1

theorem sequence_formula (n : ℕ) (h : 0 < n) : 
  ∀ x : ℝ, (0 ≤ x ∧ x < n) → 
  (f x ∈ finset.range (a n)) :=
sorry

end sequence_formula_l813_813354


namespace value_of_t_f_seven_l813_813503

def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
def f (x : ℝ) : ℝ := 7 - t x

theorem value_of_t_f_seven : t (f 7) = Real.sqrt (37 - 5 * Real.sqrt 37) :=
by
  sorry

end value_of_t_f_seven_l813_813503


namespace max_side_length_l813_813711

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813711


namespace point_translation_properties_l813_813605

theorem point_translation_properties
  (x : ℝ)
  (m : ℝ)
  (h₀ : m > 0)
  (f₀ : y = cos (2 * x + π / 6))
  (P : x = π / 4)
  (P' : x' = x + m)
  (h₁ : y' = cos (2 * x'))
  : y = cos (2 * x + π / 6) → x = π / 4 → y = -1 / 2 → ∃ min_m, min_m = π / 12 := by
  sorry

end point_translation_properties_l813_813605


namespace area_triangle_ADC_l813_813469

noncomputable theory

open Classical
open Real Set

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [ordered_semiring B]
variables {x : ℝ}

def is_right_triangle (A B C : Type) : Prop :=
  ∃ (B : ℝ), ∀ (t : ℝ), 0 ≤ B

def angle_bisector {A B C : Type} (ang : Type) : Prop :=
  ∀ (k : ℝ), 0 ≤ k

def triangle_form (A B C : Type) : Prop := is_right_triangle A B C ∧ angle_bisector A

def length (BD DC : ℝ) : Prop := BD = 4 ∧ DC = 6

theorem area_triangle_ADC :
  (triangle_form A B C) →
  (length 4 6) →
  ∃ k : ℝ, k = (60 * real.sqrt (13) / 13) :=
begin
  intros h₁ h₂,
  sorry
end

end area_triangle_ADC_l813_813469


namespace sum_of_prime_factors_of_2_to_10_minus_1_l813_813590

theorem sum_of_prime_factors_of_2_to_10_minus_1 :
  let n := 2^10 - 1,
      factors := [31, 3, 11] in
  (n = factors.prod) ∧ (factors.all Prime) → factors.sum = 45 :=
by
  let n := 2^10 - 1
  let factors := [31, 3, 11]
  have fact_prod : n = factors.prod := by sorry
  have all_prime : factors.all Prime := by sorry
  have sum_factors : factors.sum = 45 := by sorry
  exact ⟨fact_prod, all_prime, sum_factors⟩

end sum_of_prime_factors_of_2_to_10_minus_1_l813_813590


namespace find_max_side_length_l813_813783

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813783


namespace shortest_distance_l813_813650

-- Define the positions and distances
def herdsman_pos : ℝ × ℝ := (0, 4) -- 4 miles north of the river (position on coordinate plane)
def house_pos : ℝ × ℝ := (-8, 11) -- 8 miles west and 7+4 miles north from the river and herdsman

-- Define the function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the main theorem to state the shortest distance
theorem shortest_distance : distance herdsman_pos house_pos = 17 := by
  sorry

end shortest_distance_l813_813650


namespace complex_circle_intersection_l813_813109

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l813_813109


namespace num_ways_books_distribution_l813_813298

-- Given conditions
def num_copies_type1 : ℕ := 8
def num_copies_type2 : ℕ := 4
def min_books_in_library_type1 : ℕ := 1
def max_books_in_library_type1 : ℕ := 7
def min_books_in_library_type2 : ℕ := 1
def max_books_in_library_type2 : ℕ := 3

-- The proof problem statement
theorem num_ways_books_distribution : 
  (max_books_in_library_type1 - min_books_in_library_type1 + 1) * 
  (max_books_in_library_type2 - min_books_in_library_type2 + 1) = 21 := by
    sorry

end num_ways_books_distribution_l813_813298


namespace smallest_k_divides_l813_813886

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (z : ℂ) : ∀ k : ℕ, (f z ∣ z^42 - 1) ∧ (∀ k' : ℕ, k' < 42 → ¬ (f z ∣ z^k' - 1)) :=
by
  sorry

end smallest_k_divides_l813_813886


namespace max_side_length_of_triangle_l813_813763

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813763


namespace height_of_pole_l813_813324

theorem height_of_pole 
    (A B C D E : ℝ)
    (AC AD DE : ℝ)
    (h_AC : AC = 5)
    (h_AD : AD = 4)
    (h_DE : DE = 3)
    (h_pythagorean : AC = AD + (AC - AD)) :
  let DC = AC - AD in 
  let similarity_ratio := DE / DC in 
  AB = similarity_ratio * AC := 
  by sorry

end height_of_pole_l813_813324


namespace max_side_length_l813_813707

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813707


namespace problem_a4_equals_80_l813_813391

noncomputable def a : ℕ → ℕ
| 1 => 2
| n+1 => 3 * (a n + 1) - 1

theorem problem_a4_equals_80 : a 4 = 80 := 
by sorry

end problem_a4_equals_80_l813_813391


namespace minimum_perimeter_of_triangle_l813_813508

noncomputable def triangle_min_perimeter (a b c : ℕ) (A B C : ℝ) : ℕ :=
if h1 : a^2 = b^2 + c^2 - 2 * b * c * real.cos A ∧ b^2 = a^2 + c^2 - 2 * a * c * real.cos B ∧ c^2 = a^2 + b^2 - 2 * a * b * real.cos C
then a + b + c
else 0

theorem minimum_perimeter_of_triangle (a b c : ℕ) (A B C : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) -- Triangle side lengths are positive integers
  (h2 : A = 3 * B) -- Given angle condition
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * real.cos A) -- Law of cosines for angle A
  (h4 : b^2 = a^2 + c^2 - 2 * a * c * real.cos B) -- Law of cosines for angle B
  (h5 : c^2 = a^2 + b^2 - 2 * a * b * real.cos C) -- Law of cosines for angle C
  : triangle_min_perimeter a b c A B C = 21 :=
begin
  sorry
end

end minimum_perimeter_of_triangle_l813_813508


namespace coleen_sprinkles_l813_813342

theorem coleen_sprinkles : 
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  remaining_sprinkles = 3 :=
by
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  sorry

end coleen_sprinkles_l813_813342


namespace total_burn_time_l813_813336

/-- Define the number of toothpicks in the 3x5 rectangle -/
def num_toothpicks : ℕ := 38

/-- Define the burn time for each toothpick -/
def burn_time_per_toothpick : ℕ := 10

/-- Given conditions for burning -/
structure fire_conditions where
  num_toothpicks : ℕ
  burn_time_per_toothpick : ℕ
  ignited_at_corners : bool -- true indicates fire starts at two adjacent corners

/-- Initialize the conditions -/
def initial_conditions : fire_conditions :=
  { num_toothpicks := num_toothpicks,
    burn_time_per_toothpick := burn_time_per_toothpick,
    ignited_at_corners := true }

/-- Theorem stating the total burn time for the entire structure -/
theorem total_burn_time (conds : fire_conditions) : conds = initial_conditions → ℕ :=
by
  intro h
  exact 65
  sorry

end total_burn_time_l813_813336


namespace illuminate_no_vertices_of_cube_l813_813832

open Real EuclideanGeometry

def spotlight_center : EuclideanSpace ℝ 3 := (0, 0, 0)

def cube_vertices : List (EuclideanSpace ℝ 3) :=
  [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
   (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

noncomputable def illuminates (spotlight : EuclideanSpace ℝ 3) : List (EuclideanSpace ℝ 3) → Bool :=
  fun vertices => List.any vertices (λ v, (spotlight.1 * v.1 ≥ 0) ∧ (spotlight.2 * v.2 ≥ 0) ∧ (spotlight.3 * v.3 ≥ 0))

theorem illuminate_no_vertices_of_cube :
  ∃ orientation : EuclideanSpace ℝ 3, ¬ illuminates orientation cube_vertices :=
by
  sorry

end illuminate_no_vertices_of_cube_l813_813832


namespace find_line_equation_l813_813063

noncomputable def line_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x - 4

theorem find_line_equation :
  ∃ (x₁ y₁ : ℝ), x₁ = Real.sqrt 3 ∧ y₁ = -3 ∧ ∀ x y, (line_equation x y ↔ 
  (y + 3 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3))) :=
sorry

end find_line_equation_l813_813063


namespace union_A_B_range_of_a_l813_813927

noncomputable def A := {x : ℝ | 1 < x ∧ x ≤ 2}
noncomputable def B1 := {y : ℝ | ∃ x : ℝ, x ≤ 0 ∧ y = 2^x + 3/2}
noncomputable def B2 (a : ℝ) := {y : ℝ | ∃ x : ℝ, x <= 0 ∧ y = 2^x + a}

theorem union_A_B (a : ℝ) (a_eq : a = 3/2) : 
  A ∪ B1 = {x : ℝ | 1 < x ∧ x ≤ 5/2} := by
  sorry

theorem range_of_a (a : ℝ) (intersection_empty : A ∩ B2(a) = ∅) :
  a ≥ 2 ∨ a ≤ 0 := by
  sorry

end union_A_B_range_of_a_l813_813927


namespace max_side_length_l813_813701

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813701


namespace sum_inequality_l813_813138

theorem sum_inequality (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (b * (a + b))) + (1 / (c * (b + c))) + (1 / (a * (c + a))) ≥ 3 / 2 :=
sorry

end sum_inequality_l813_813138


namespace angle_bisector_length_formula_l813_813532

variable (a b c : ℝ)
noncomputable def p : ℝ := (a + b + c) / 2

theorem angle_bisector_length_formula : 
  let l_C := (2 * (Real.sqrt(p a b c * (p a b c - a) * b * c))) / (b + c)
  in l_C = (2 * (Real.sqrt((a + b + c) / 2 * ((a + b + c) / 2 - a) * b * c))) / (b + c) :=
by 
  sorry

end angle_bisector_length_formula_l813_813532


namespace carrots_total_l813_813283

theorem carrots_total 
  (picked_1 : Nat) 
  (thrown_out : Nat) 
  (picked_2 : Nat) 
  (total_carrots : Nat) 
  (h_picked1 : picked_1 = 23) 
  (h_thrown_out : thrown_out = 10) 
  (h_picked2 : picked_2 = 47) : 
  total_carrots = 60 := 
by
  sorry

end carrots_total_l813_813283


namespace monotonic_increase_interval_l813_813457

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  real.log (a) (2 * x^2 + x)

theorem monotonic_increase_interval (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : ∀ x, 0 < x ∧ x < 1/2 → f a x > 0) : 
  ∀ x, x ∈ I uSet.Ioo (-∞) (-1/2) → monotonic_increase (f a x)
  :=
sorry

end monotonic_increase_interval_l813_813457


namespace problem_l813_813140

variable (ABCD : Type)
variable [ConvexQuadrilateral ABCD]
variable (A B C D T : ABCD)
variable (AB AD AC : ℝ)
variable (∠ ABT ∠ ADT ∠ BCD : ℝ)

-- Given conditions
variable (hABeqAD : AB = AD)
variable (hT_on_AC : T ∈ interior (closed_segment A C))
variable (h_angle_sum : ∠ ABT + ∠ ADT = ∠ BCD)

-- proof goal
theorem problem : AT + AC ≥ AB + AD :=
by sorry

end problem_l813_813140


namespace max_side_length_l813_813703

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813703


namespace matrix_multiplication_correct_l813_813848

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![2, 0, -1],
    ![1, 3, -2],
    ![0, 2, 4]
  ]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![1, -1, 0],
    ![2, 0, -2],
    ![3, 0, 1]
  ]

def AB_product : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![-1, -2, -1],
    ![1, -1, -8],
    ![16, 0, 0]
  ]

theorem matrix_multiplication_correct :
  A ⬝ B = AB_product :=
by
  sorry

end matrix_multiplication_correct_l813_813848


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813822

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813822


namespace problem_correct_l813_813637

variables (A B : Type) 

def truck_type_B_capacity (x : ℤ) : Prop := 
  let capacity_B := 80 in 
  capacity_B = x -- given that type B trucks can carry 80 boxes

def truck_type_A_capacity (x : ℤ) : Prop :=
  let capacity_A := 100 in 
  capacity_A = x + 20 -- given that type A trucks can carry 20 more boxes than type B trucks

def transport_boxes_equivalence (x y : ℤ) (n m : ℤ) : Prop := 
  (1000 / x = 800 / y) ∧
  (m = 18 - n) ∧ 
  (n + m = 18) ∧
  (100 * n + 80 * (m - 1) + 65 = 1625)

theorem problem_correct
  (x_A x_B : ℤ)
  (m n : ℤ) 
  (hx_B : truck_type_B_capacity x_B)
  (hx_A : truck_type_A_capacity x_A) 
  (htrans : transport_boxes_equivalence x_A x_B n m) :
  x_A = 100 ∧ x_B = 80 ∧ n = 10 ∧ m = 8 :=
by 
  -- statements provided for context.
  sorry

end problem_correct_l813_813637


namespace trig_identity_l813_813205

theorem trig_identity (x z : ℝ) :
  sin x ^ 2 + sin (x + z) ^ 2 - 2 * sin x * sin z * sin (x + z) = sin z ^ 2 :=
by
  sorry

end trig_identity_l813_813205


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813817

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813817


namespace minimum_positive_difference_contains_amounts_of_numbers_on_strips_l813_813530

theorem minimum_positive_difference_contains_amounts_of_numbers_on_strips (a b c d e f : ℕ) 
  (h1 : a + f = 7) (h2 : b + e = 7) (h3 : c + d = 7) :
  ∃ (min_diff : ℕ), min_diff = 1 :=
by {
  -- The problem guarantees the minimum difference given the conditions.
  sorry
}

end minimum_positive_difference_contains_amounts_of_numbers_on_strips_l813_813530


namespace arithmetic_progression_any_real_l813_813380

theorem arithmetic_progression_any_real (y : ℝ) :
  ∃ d : ℝ, (30 + y) - (10 + y) = d ∧ (60 + y) - (30 + y) = d :=
by
  use 20
  split
  {
    calc
      (30 + y) - (10 + y) = 30 - 10 : by ring
      ... = 20 : by ring
  }
  {
    calc
      (60 + y) - (30 + y) = 60 - 30 : by ring
      ... = 30 : by ring
  }
  sorry

end arithmetic_progression_any_real_l813_813380


namespace fixed_point_through_CP_lines_l813_813997

open Set

noncomputable def midpoint {α : Type*} [Field α] (A B : α × α) : α × α :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def is_tangent {α : Type*} [Field α] (A B C : α × α) (O : α × α) : Prop :=
  ∃ P : α × α, is_circle O P ∧ is_tangent_to_circle (Circ PAO)

noncomputable def circ {α : Type*} [Field α] (A B M : α × α) : Set₁ (α × α) :=
  {Q | dist Q O = dist A O}

theorem fixed_point_through_CP_lines {α : Type*} [Field α] 
  (O : α × α) (A B : α × α) (hChord : ¬collinear {A, O, B})
  (hAB : dist A B > 0)
  (hM : let M := midpoint A B in True)
  (C : α × α) (hC_diff : C ≠ A ∧ C ≠ B ∧ is_circle O C)
  (P : α × α) (hTangent_A: is_tangent A (circ A C (midpoint A B)) P)
  (hTangent_B : is_tangent B (circ B C (midpoint A B)) P) :
  ∃ F : α × α, ∀ C, (C ≠ A ∧ C ≠ B ∧ is_circle O C) → (line_through C P ∩ line_through A B = {F}) :=
sorry

end fixed_point_through_CP_lines_l813_813997


namespace length_of_bridge_l813_813666

theorem length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (cross_time_sec : ℕ) (bridge_length: ℕ):
  train_length = 110 →
  train_speed_kmph = 45 →
  cross_time_sec = 30 →
  bridge_length = 265 :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l813_813666


namespace ellipse_equation_range_reciprocal_distances_l813_813418

-- Definitions of the foci and the point M on the ellipse.
def F1 : ℝ × ℝ := (0, -Real.sqrt 3)
def F2 : ℝ × ℝ := (0, Real.sqrt 3)
def M : ℝ × ℝ := (Real.sqrt 3 / 2, 1)

-- The statement to prove the standard equation of the ellipse C.
theorem ellipse_equation :
  (∃ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b), a^2 = b^2 + (Real.sqrt 3)^2 ∧
    (M.1)^2 / b^2 + (M.2)^2 / a^2 = 1) →
  (a = 2 ∧ b = 1) →
  (∀ (x y : ℝ), (x^2 / b^2 + y^2 / a^2 = 1) ↔ (y^2 / 4 + x^2 = 1)) :=
sorry

-- The statement to prove the range of (1 / |PF1|) + (1 / |PF2|).
theorem range_reciprocal_distances :
  (∀ (P : ℝ × ℝ), (P.1^2 / 1 + P.2^2 / 4 = 1) →
     1 ≤ (1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
          1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) ∧ 
     (1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
      1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) ≤ 4) :=
sorry

end ellipse_equation_range_reciprocal_distances_l813_813418


namespace max_side_length_l813_813779

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813779


namespace sqrt_ab_is_integer_l813_813531

theorem sqrt_ab_is_integer
  (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
  (h_eq : a * (b^2 + n^2) = b * (a^2 + n^2)) :
  ∃ k : ℕ, k * k = a * b :=
by
  sorry

end sqrt_ab_is_integer_l813_813531


namespace sprinkles_remaining_l813_813340

/-- Given that Coleen started with twelve cans of sprinkles and after applying she had 3 less than half as many cans, prove the remaining cans are 3 --/
theorem sprinkles_remaining (initial_sprinkles : ℕ) (h_initial : initial_sprinkles = 12) (h_remaining : ∃ remaining_sprinkles, remaining_sprinkles = initial_sprinkles / 2 - 3) : ∃ remaining_sprinkles, remaining_sprinkles = 3 :=
by
  have half_initial := initial_sprinkles / 2
  have remaining_sprinkles := half_initial - 3
  use remaining_sprinkles
  rw [h_initial, Nat.div_eq_of_lt (by decide : 6 < 12)]
  sorry

end sprinkles_remaining_l813_813340


namespace flowers_remaining_along_path_after_events_l813_813826

def total_flowers : ℕ := 30
def total_peonies : ℕ := 15
def total_tulips : ℕ := 15
def unwatered_flowers : ℕ := 10
def tulips_watered_by_sineglazka : ℕ := 10
def tulips_picked_by_neznaika : ℕ := 6
def remaining_flowers : ℕ := 19

theorem flowers_remaining_along_path_after_events :
  total_peonies + total_tulips = total_flowers →
  tulips_watered_by_sineglazka + unwatered_flowers = total_flowers →
  tulips_picked_by_neznaika ≤ total_tulips →
  remaining_flowers = 19 := sorry

end flowers_remaining_along_path_after_events_l813_813826


namespace simplify_expression_l813_813204

theorem simplify_expression (x : ℝ) : 
  (tan(2 * x) + 2 * tan(4 * x) + 4 * tan(8 * x) + 8 * cot(16 * x) = cot(2 * x)) := 
by 
  sorry

end simplify_expression_l813_813204


namespace inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l813_813919

variable (a : ℝ) (x : ℝ)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Part (1)
theorem inequality_solution (a : ℝ) (h1 : 0 < a ∧ a < 1) : (0 ≤ x ∧ x ≤ 2*a / (1 - a^2)) → (f x a ≤ 1) :=
sorry

theorem inequality_solution_b (a : ℝ) (h2 : a ≥ 1) : (0 ≤ x) → (f x a ≤ 1) :=
sorry

-- Part (2)
theorem monotonic_increasing (a : ℝ) (h3 : a ≤ 0) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≤ f x2 a :=
sorry

theorem monotonic_decreasing (a : ℝ) (h4 : a ≥ 1) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≥ f x2 a :=
sorry

end inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l813_813919


namespace sum_c_squared_approx_l813_813351

def c (k : ℕ) : ℝ := k + 1 / (3 * k + 1)

noncomputable def sum_c_squared (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), (c k) ^ 2

theorem sum_c_squared_approx : sum_c_squared 10 ≈ 391.98 :=
by
  sorry

end sum_c_squared_approx_l813_813351


namespace selling_price_per_bowl_l813_813313

-- Definitions based on conditions
def num_bowls : ℕ := 110
def cost_per_bowl : ℝ := 10
def num_sold_bowls : ℕ := 100
def gain_percentage : ℝ := 27.27272727272727 / 100

-- The resulting Lean 4 statement:
theorem selling_price_per_bowl :
  let total_cost := (num_bowls * cost_per_bowl)
  let total_SP := total_cost + (gain_percentage * total_cost)
  let price_per_bowl_sold := total_SP / num_sold_bowls
  price_per_bowl_sold = 14 := 
by
  -- Proof goes here
  sorry

end selling_price_per_bowl_l813_813313


namespace tan_double_angle_l813_813039

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_derivative_def (x : ℝ) : ℝ := 3 * f x

theorem tan_double_angle (x : ℝ) (h : f_derivative_def x = Real.cos x - Real.sin x) : 
  Real.tan (2 * x) = -4 / 3 :=
by
  sorry

end tan_double_angle_l813_813039


namespace minimum_turns_to_exceed_1000000_l813_813988

theorem minimum_turns_to_exceed_1000000 :
  let a : Fin 5 → ℕ := fun n => if n = 0 then 1 else 0
  (∀ n : ℕ, ∃ (b_2 b_3 b_4 b_5 : ℕ),
    a 4 + b_2 ≥ 0 ∧
    a 3 + b_3 ≥ 0 ∧
    a 2 + b_4 ≥ 0 ∧
    a 1 + b_5 ≥ 0 ∧
    b_2 * b_3 * b_4 * b_5 > 1000000 →
    b_2 + b_3 + b_4 + b_5 = n) → 
    ∃ n, n = 127 :=
by
  sorry

end minimum_turns_to_exceed_1000000_l813_813988


namespace book_pairs_count_l813_813947

theorem book_pairs_count :
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let science_fiction_books := 3
  let genres := 4
  let genre_pairs := Nat.choose genres 2
  let books_in_pair := mystery_books * fantasy_books  
  genre_pairs * books_in_pair = 54 :=
by
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let science_fiction_books := 3
  let genres := 4
  let genre_pairs := Nat.choose genres 2 -- 6 ways to choose 2 genres out of 4
  let books_in_pair := mystery_books * fantasy_books -- 9 ways to choose 1 book from each genre
  have genre_pairs_eq : genre_pairs = 6 := by simp [genre_pairs]
  have books_in_pair_eq : books_in_pair = 9 := by simp [books_in_pair]
  have total_pairs := genre_pairs * books_in_pair
  have total_pairs_eq : total_pairs = 6 * 9 := by simp [total_pairs]
  have result : total_pairs = 54 := by simp [total_pairs_eq]
  exact result

end book_pairs_count_l813_813947


namespace original_number_is_7_l813_813880

theorem original_number_is_7 (N : ℕ) (h : ∃ (k : ℤ), N = 12 * k + 7) : N = 7 :=
sorry

end original_number_is_7_l813_813880


namespace max_triangle_side_l813_813721

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813721


namespace sum_of_numbers_l813_813571

theorem sum_of_numbers (a b c : ℕ) (h_order: a ≤ b ∧ b ≤ c) (h_median: b = 10) 
    (h_mean_least: (a + b + c) / 3 = a + 15) (h_mean_greatest: (a + b + c) / 3 = c - 20) :
    a + b + c = 45 :=
  by
  sorry

end sum_of_numbers_l813_813571


namespace paths_A_to_C_l813_813893

theorem paths_A_to_C :
  let paths_AB := 2
  let paths_BD := 3
  let paths_DC := 3
  let paths_AC_direct := 1
  paths_AB * paths_BD * paths_DC + paths_AC_direct = 19 :=
by
  sorry

end paths_A_to_C_l813_813893


namespace children_per_row_l813_813362

theorem children_per_row (total_children rows_per_bus : ℕ) (h1 : total_children = 36) (h2 : rows_per_bus = 9) :
  (total_children / rows_per_bus = 4) :=
by {
  rw [h1, h2],    -- Replace total_children with 36 and rows_per_bus with 9
  norm_num,       -- Simplify the division 36 / 9 = 4
  sorry           -- Proof is omitted
}

end children_per_row_l813_813362


namespace milan_long_distance_bill_l813_813895

theorem milan_long_distance_bill
  (monthly_fee : ℝ := 2)
  (per_minute_cost : ℝ := 0.12)
  (minutes_used : ℕ := 178) :
  ((minutes_used : ℝ) * per_minute_cost + monthly_fee = 23.36) :=
by
  sorry

end milan_long_distance_bill_l813_813895


namespace find_first_coaster_speed_l813_813604

/-- Given conditions --/
variables
  (speed1 : ℕ)  -- speed of the first coaster
  (speed2 speed3 speed4 speed5 : ℕ)  -- speeds of the other four coasters

/-- Given values of known coasters' speeds and average speed condition --/
constants
  (h_speed2 : speed2 = 62)
  (h_speed3 : speed3 = 73)
  (h_speed4 : speed4 = 70)
  (h_speed5 : speed5 = 40)
  (h_avg_speed : (speed1 + speed2 + speed3 + speed4 + speed5) / 5 = 59)

/-- Theorem: Prove that the speed of the first coaster is 50 --/
theorem find_first_coaster_speed : speed1 = 50 :=
by
  sorry

end find_first_coaster_speed_l813_813604


namespace norm_mul_euclidean_division_prime_iff_irreducible_prime_factorization_unique_prime_characterization_l813_813209

-- Define Gaussian integers and their norm
def GaussianInteger : Type := ℤ × ℤ
def norm (a : GaussianInteger) : ℤ := a.1 * a.1 + a.2 * a.2

-- Verification of norm multiplicativity
theorem norm_mul (a b : GaussianInteger) : norm (a.1 * b.1 - a.2 * b.2, a.1 * b.2 + a.2 * b.1) = norm a * norm b := sorry

-- Existence of Euclidean division
theorem euclidean_division (a b : GaussianInteger) : ∃ q r : GaussianInteger, a = (b.1 * q.1 - b.2 * q.2, b.1 * q.2 + b.2 * q.1) + r ∧ norm r < norm b := sorry

-- Prime if and only if irreducible
def is_unit (a : GaussianInteger) : Prop := norm a = 1

def is_prime (p : GaussianInteger) : Prop :=
  ∀ a b : GaussianInteger, p ∣ (a.1 * b.1 - a.2 * b.2, a.1 * b.2 + a.2 * b.1) → (p ∣ a ∨ p ∣ b)

def is_irreducible (p : GaussianInteger) : Prop :=
  ∀ a b : GaussianInteger, p = (a.1 * b.1 - a.2 * b.2, a.1 * b.2 + a.2 * b.1) → (is_unit a ∨ is_unit b)

theorem prime_iff_irreducible (p : GaussianInteger) : is_prime p ↔ is_irreducible p := sorry

-- Uniqueness of prime factorization
theorem prime_factorization_unique (a : GaussianInteger) : ∃! (p : List GaussianInteger), (∀ x : GaussianInteger, x ∣ a ↔ x ∈ p) := sorry

-- Characterization of prime Gaussian integers
def is_prime_gaussian (a : GaussianInteger) : Prop :=
  norm a = 2 ∨ (∃ p : ℤ, p.prime ∧ p % 4 = 1 ∧ norm a = p) ∨ (∃ q : ℤ, q.prime ∧ q % 4 = 3 ∧ norm a = q * q)

theorem prime_characterization (a : GaussianInteger) : is_prime a ↔ is_prime_gaussian a := sorry

end norm_mul_euclidean_division_prime_iff_irreducible_prime_factorization_unique_prime_characterization_l813_813209


namespace Elle_practice_time_l813_813390

variable (x : ℕ)

theorem Elle_practice_time : 
  (5 * x) + (3 * x) = 240 → x = 30 :=
by
  intro h
  sorry

end Elle_practice_time_l813_813390


namespace fraction_replaced_l813_813640

theorem fraction_replaced :
  ∃ x : ℚ, (0.60 * (1 - x) + 0.25 * x = 0.35) ∧ x = 5 / 7 := by
    sorry

end fraction_replaced_l813_813640


namespace length_of_EH_l813_813606

-- Definitions based on the conditions
structure Trapezoid :=
(E F G H : ℝ)
(EF_parallel_GH : True) -- EF parallel to GH
(FG_EQ_GH : FG = 39 ∧ GH = 39) -- FG = GE = 39
(EH_perp_FH : EH ⊥ FH) -- EH perpendicular to FH
(diagonals_intersect_J : ∃ J : ℝ, True) -- Not fully specified, but exists such a point J
(midpoint_K_FH : ∃ K : ℝ, True) -- Midpoint of FH is K
(JK_EQ_13 : JK = 13) -- JK = 13

-- Prove the question
theorem length_of_EH (T : Trapezoid) : 
  ∃ (p q : ℕ), p = 5 ∧ q = 304 ∧ sqrt (p * p * q) = EH := 
by
  sorry -- proof

end length_of_EH_l813_813606


namespace intersection_A_complement_B_is_01_l813_813136

open Set

def U := Set.univ
def A := {x : ℝ | x^2 - 2 * x < 0}
def B := {x : ℝ | x ≥ 1}
def C_U_B := {x : ℝ | x ∉ B}

theorem intersection_A_complement_B_is_01 :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_A_complement_B_is_01_l813_813136


namespace fisherman_daily_earnings_l813_813218

theorem fisherman_daily_earnings :
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 :=
by
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  show red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52
  sorry

end fisherman_daily_earnings_l813_813218


namespace four_digit_numbers_divisible_by_nine_l813_813936

theorem four_digit_numbers_divisible_by_nine : 
  (count (λ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 9 = 0) = 1000) :=
by
  sorry

end four_digit_numbers_divisible_by_nine_l813_813936


namespace initialNumberOfFishIs60_l813_813603

/-- Define the daily fish consumption by the Bobbit worm -/
def dailyConsumption := 2

/-- Define the total days in the first period -/
def firstPeriodDays := 14

/-- Define the total fish consumption in the first period -/
def totalConsumptionFirstPeriod := dailyConsumption * firstPeriodDays

/-- Define the number of fish added after the first period -/
def fishAdded := 8

/-- Define the total days in the second period -/
def secondPeriodDays := 7

/-- Define the total fish consumption in the second period -/
def totalConsumptionSecondPeriod := dailyConsumption * secondPeriodDays

/-- Define the final number of fish after both periods and fish addition -/
def finalFish := 26

/-- Define the initial number of fish -/
def initialFish (F : ℕ) : Prop :=
  F - totalConsumptionFirstPeriod + fishAdded - totalConsumptionSecondPeriod = finalFish

/-- The problem: Prove that the initial number of fish was 60 -/
theorem initialNumberOfFishIs60 : initialFish 60 :=
by
  unfold initialFish
  dsimp
  norm_num
  sorry

end initialNumberOfFishIs60_l813_813603


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813810

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813810


namespace rice_in_each_container_l813_813288

-- Definition of the given conditions
def weight_in_pounds : ℝ := real.sqrt 50
def num_containers : ℝ := 7
def pound_to_ounce : ℝ := 16

-- The target value to prove
def weight_per_container_in_ounces : ℝ := (80 * real.sqrt 2) / 7

-- The main theorem
theorem rice_in_each_container :
  (weight_in_pounds / num_containers) * pound_to_ounce = weight_per_container_in_ounces :=
by sorry

end rice_in_each_container_l813_813288


namespace initial_investment_calculation_l813_813834

theorem initial_investment_calculation :
  ∃ (P : ℝ), (P * (1 + 0.10 / 1)^2 = 4840.000000000001) ∧ P = 4000 :=
by
  use 4000
  split
  calc
    4000 * (1 + 0.10 / 1)^2 = 4000 * (1.10)^2 : by rfl
    ... = 4000 * 1.21 : by norm_num
    ... = 4840 : by linarith
  rfl

end initial_investment_calculation_l813_813834


namespace find_point_P_l813_813405

noncomputable def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  ( (a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2 ).sqrt

theorem find_point_P (z : ℝ) : 
  let A := (1, -2, 1)
  let B := (2, 2, 2)
  let P := (0, 0, z)
  distance A P = distance B P → z = 3 :=
by
  sorry

end find_point_P_l813_813405


namespace union_sets_a_l813_813909

theorem union_sets_a (P S : Set ℝ) (a : ℝ) :
  P = {1, 5, 10} →
  S = {1, 3, a^2 + 1} →
  S ∪ P = {1, 3, 5, 10} →
  a = 2 ∨ a = -2 ∨ a = 3 ∨ a = -3 :=
by
  intros hP hS hUnion 
  sorry

end union_sets_a_l813_813909


namespace number_of_dimes_paid_l813_813177

theorem number_of_dimes_paid (cost_in_dollars : ℕ) (value_of_dime_in_cents : ℕ) (value_of_dollar_in_cents : ℕ) 
  (h_cost : cost_in_dollars = 9) (h_dime : value_of_dime_in_cents = 10) (h_dollar : value_of_dollar_in_cents = 100) : 
  (cost_in_dollars * value_of_dollar_in_cents) / value_of_dime_in_cents = 90 := by
  -- Proof to be provided here
  sorry

end number_of_dimes_paid_l813_813177


namespace max_side_length_is_11_l813_813681

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813681


namespace intersection_PQ_fixed_point_eq_PQ_line_r2_M42_l813_813984

noncomputable def circle_center_eq {r : ℝ} (r_pos : 0 < r) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 :=
begin
  use [r, 0],
  exact eq.refl (r^2),
end

noncomputable def intersection_fixed_point (r a t : ℝ) (cond : 0 < r ∧ r < a) :
  ∃ (x : ℝ), x = r^2 / a :=
begin
  use r^2 / a,
  exact eq.refl (r^2 / a),
end

theorem intersection_PQ_fixed_point (r a : ℝ) (ha : 0 < a) (hr : 0 < r) (h_cond : r < a) :
  ∃ (fx fy : ℝ), fx = r^2 / a ∧ fy = 0 :=
begin
  use [r^2 / a, 0],
  refine ⟨eq.refl _, eq.refl 0⟩,
end

noncomputable def eq_PQ_line (r a t : ℝ) (cond : 0 < r ∧ r < a) :
  ∃ C : ℝ, C = 2 :=
begin
  use 2,
  apply (eq.refl 2),
end

theorem eq_PQ_line_r2_M42 :
  ∃ C : ℝ, C = 2 :=
begin
  exact eq_PQ_line 2 4 2 ⟨by norm_num, by norm_num⟩,
end

end intersection_PQ_fixed_point_eq_PQ_line_r2_M42_l813_813984


namespace solution_set_f_lt_6_l813_813154

theorem solution_set_f_lt_6 
    (f : ℝ → ℝ) 
    (h_monotonic : monotone f ∨ antitone f) 
    (h_func_eq : ∀ x > 0, f (f x - 2 * log x / log 2) = 4)
    : {x : ℝ | 0 < x ∧ f x < 6} = {x | 0 < x ∧ x < 4} := 
by 
  sorry

end solution_set_f_lt_6_l813_813154


namespace max_side_of_triangle_l813_813747

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813747


namespace three_pairwise_intersecting_circles_lines_intersection_l813_813931

theorem three_pairwise_intersecting_circles_lines_intersection
    (C1 C2 C3 : Circle)
    (h1 : intersects C1 C2)
    (h2 : intersects C2 C3)
    (h3 : intersects C1 C3) :
    let L1 := radical_axis C1 C2
    let L2 := radical_axis C2 C3
    let L3 := radical_axis C1 C3
    (∃ P : Point, P ∈ L1 ∧ P ∈ L2 ∧ P ∈ L3) ∨ (L1 ∥ L2 ∧ L1 ∥ L3) :=
begin
    sorry
end

end three_pairwise_intersecting_circles_lines_intersection_l813_813931


namespace units_digit_of_7_pow_6_pow_5_l813_813861

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l813_813861


namespace first_shaded_complete_cycle_seat_190_l813_813642

theorem first_shaded_complete_cycle_seat_190 : 
  ∀ (n : ℕ), (n ≥ 1) → 
  ∃ m : ℕ, 
    ((m ≥ n) ∧ 
    (∀ i : ℕ, (1 ≤ i ∧ i ≤ 12) → 
    ∃ k : ℕ, (k ≤ m ∧ (k * (k + 1) / 2) % 12 = (i - 1) % 12))) ↔ 
  ∃ m : ℕ, (m = 19 ∧ 190 = (m * (m + 1)) / 2) :=
by
  sorry

end first_shaded_complete_cycle_seat_190_l813_813642


namespace units_digit_7_pow_6_pow_5_l813_813860

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end units_digit_7_pow_6_pow_5_l813_813860


namespace geometric_sequence_k_value_l813_813014

theorem geometric_sequence_k_value (a : ℕ → ℝ) (S : ℕ → ℝ) (a1_pos : 0 < a 1)
  (geometric_seq : ∀ n, a (n + 2) = a n * (a 3 / a 1)) (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) (h_Sk : S k = 63) :
  k = 6 := 
by
  sorry

end geometric_sequence_k_value_l813_813014


namespace propositions_validity_l813_813033

theorem propositions_validity (p q : Prop) (a b m x : ℝ) :
  (¬ (p ∨ q) → (¬ p ∧ ¬ q)) ∧
  (∃ x : ℝ, x^2 + 1 > 3 * x ∧ ∀ x : ℝ, x^2 - 1 < 3 * x) → False ∧
  (a < b → ∀ m : ℝ, (am^2 < bm^2)) → False ∧
  ((p → q) ∧ (¬ q → ¬ p) → (¬ p → ¬ q)) :=
by
  sorry

end propositions_validity_l813_813033


namespace part1_part2_l813_813038

noncomputable def f (x : ℝ) : ℝ := |(x - 1)| + |(x + 2)|

theorem part1 (x : ℝ) : f(x) ≥ 4 ↔ x ≤ -7/2 ∨ x ≥ 3/2 :=
by
  sorry

theorem part2 (c : ℝ) : (∀ x : ℝ, f(x) ≥ c) → c ≤ 3 :=
by
  sorry

end part1_part2_l813_813038


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813818

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813818


namespace pizza_slices_left_for_Phill_l813_813184

theorem pizza_slices_left_for_Phill :
  ∀ (initial_slices : ℕ) (first_cut : ℕ) (second_cut : ℕ) (third_cut : ℕ)
    (slices_given_to_3_friends : ℕ) (slices_given_to_2_friends : ℕ) (slices_left_for_Phill : ℕ),
    initial_slices = 1 →
    first_cut = 2 →
    second_cut = 4 →
    third_cut = 8 →
    slices_given_to_3_friends = 3 →
    slices_given_to_2_friends = 4 →
    slices_left_for_Phill = third_cut - (slices_given_to_3_friends + slices_given_to_2_friends) →
    slices_left_for_Phill = 1 :=
by {
  intros,
  subst_vars,
  simp, -- Simplify the boolean equalities
  -- We assume the steps are correct, so we leave it with sorry for now
  -- The proof should be easy for the given example and conditions.
  sorry,
}

end pizza_slices_left_for_Phill_l813_813184


namespace question1_solution_question2_solution_l813_813041

def f (x : ℝ) : ℝ := abs (x - 1)
def g (x a : ℝ) : ℝ := 2 * abs (x - a)

theorem question1_solution (a : ℝ) (x : ℝ) (h : a = 2) : (f x - g x a ≤ x - 3) ↔ (x ≤ 1 ∨ x ≥ 3) := sorry

theorem question2_solution (h : ∀ (m : ℝ), 1 < m → ∃ x : ℝ, f x + g x m ≤ (m^2 + m + 4) / (m - 1)) :
  ∀ a : ℝ, a ∈ Icc ( -2 - 2 * real.sqrt 6) (2 * real.sqrt 6 + 4) := sorry

end question1_solution_question2_solution_l813_813041


namespace general_formula_l813_813007

variable {α : Type} [Field α]

noncomputable def geometric_sequence (a1 q : α) (n : ℕ) : α :=
  if q = 1 then (3 / 2 : α)
  else 6 * (-1 / 2) ^ (n - 1)

def sum_of_first_n_terms (a1 q : α) (n : ℕ) : α :=
  if q = 1 then n * a1
  else a1 * (1 - q ^ n) / (1 - q)

theorem general_formula 
  {a1 q : α}
  (hn : ∀ n, (geometric_sequence a1 q n) = if q = 1 then (3 / 2 : α) else 6 * (-1 / 2) ^ (n - 1))
  (hS3 : sum_of_first_n_terms a1 q 3 = 9 / 2)
  (ha3 : geometric_sequence a1 q 3 = 3 / 2) :
  ∀ n, geometric_sequence a1 q n = if q = 1 then (3 / 2 : α) else 6 * (-1 / 2) ^ (n - 1) :=
sorry

end general_formula_l813_813007


namespace probability_of_cos_ge_half_l813_813658

noncomputable def probability_cos_ge_half : ℝ :=
  let A : set ℝ := {x | x ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ cos (real.pi * x) ≥ (1 / 2 : ℝ)} in
  (∫ x in A, 1 : ℝ) / (∫ x in set.Icc (-1) (1), 1)

theorem probability_of_cos_ge_half : probability_cos_ge_half = 1 / 3 :=
begin
  sorry
end

end probability_of_cos_ge_half_l813_813658


namespace max_triangle_side_24_l813_813731

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813731


namespace max_side_length_is_11_l813_813680

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813680


namespace intersection_of_circles_l813_813085

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l813_813085


namespace carol_ate_12_cakes_l813_813365

-- Definitions for conditions
def cakes_per_day : ℕ := 10
def days_baking : ℕ := 5
def cans_per_cake : ℕ := 2
def cans_for_remaining_cakes : ℕ := 76

-- Total cakes baked by Sara
def total_cakes_baked (cakes_per_day days_baking : ℕ) : ℕ :=
  cakes_per_day * days_baking

-- Remaining cakes based on frosting cans needed
def remaining_cakes (cans_for_remaining_cakes cans_per_cake : ℕ) : ℕ :=
  cans_for_remaining_cakes / cans_per_cake

-- Cakes Carol ate
def cakes_carol_ate (total_cakes remaining_cakes : ℕ) : ℕ :=
  total_cakes - remaining_cakes

-- Theorem statement
theorem carol_ate_12_cakes :
  cakes_carol_ate (total_cakes_baked cakes_per_day days_baking) (remaining_cakes cans_for_remaining_cakes cans_per_cake) = 12 :=
by
  sorry

end carol_ate_12_cakes_l813_813365


namespace initial_passengers_is_350_l813_813123

variable (N : ℕ)

def initial_passengers (N : ℕ) : Prop :=
  let after_first_train := 9 * N / 10
  let after_second_train := 27 * N / 35
  let after_third_train := 108 * N / 175
  after_third_train = 216

theorem initial_passengers_is_350 : initial_passengers 350 := 
  sorry

end initial_passengers_is_350_l813_813123


namespace inequality_sqrt_l813_813411

variable {a b c : ℝ}

-- Condition 1: a, b, c > 0
def positive (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Condition 2: a + b + c = 1
def sum_to_one (a b c : ℝ) : Prop :=
  a + b + c = 1

theorem inequality_sqrt (h1 : positive a b c) (h2 : sum_to_one a b c) :
  sqrt (ab / (ab + c)) + sqrt (bc / (bc + a)) + sqrt (ca / (ca + b)) ≤ 3 / 2 :=
sorry

end inequality_sqrt_l813_813411


namespace angle_quadrant_l813_813452

theorem angle_quadrant (k : ℤ) : 
  let α := k * 180 + 45 in 
  0 ≤ α % 360 ∧ α % 360 < 360 ∧ 
  (0 ≤ α % 360 ∧ α % 360 < 90 ∨ 180 ≤ α % 360 ∧ α % 360 < 270) := 
sorry

end angle_quadrant_l813_813452


namespace katka_perimeter_l813_813492

def perimeter (n : ℕ) : ℕ := 2 * (n + (n+1))

def total_perimeter (n : ℕ) : ℕ :=
  2 * (∑ i in finset.range n, 2*i + 1)

theorem katka_perimeter :
  total_perimeter 20 = 462 := sorry

end katka_perimeter_l813_813492


namespace computer_game_cost_l813_813129

variable (ticket_cost : ℕ := 12)
variable (num_tickets : ℕ := 3)
variable (total_spent : ℕ := 102)

theorem computer_game_cost (C : ℕ) (h : C + num_tickets * ticket_cost = total_spent) : C = 66 :=
by
  -- Proof would go here
  sorry

end computer_game_cost_l813_813129


namespace total_length_of_lines_in_setT_l813_813143

def isInSetT (x y : ℝ) : Prop :=
  abs (abs (abs x - 3) - 1) + abs (abs (abs y - 3) - 1) = 2

def setT : Set (ℝ × ℝ) := { p | isInSetT p.1 p.2 }

theorem total_length_of_lines_in_setT :
  let total_length := 32 * Real.sqrt 2
  ∑ length in setT, length = total_length :=
sorry

end total_length_of_lines_in_setT_l813_813143


namespace sum_of_corners_correct_l813_813599

theorem sum_of_corners_correct :
  let n := 9
  let checkerboard := List.range (n * n) + List.init n _ -- represents a 1-indexed list as per the condition
  let top_left := checkerboard[0]
  let top_right := checkerboard[n - 1]
  let bottom_right := checkerboard[n * n - 1]
  let bottom_left := checkerboard[(n - 1) * n]
  top_left + top_right + bottom_left + bottom_right = 164 := 
by {
  sorry
}

end sum_of_corners_correct_l813_813599


namespace cos_value_given_sin_condition_l813_813003

open Real

theorem cos_value_given_sin_condition (x : ℝ) (h : sin (x + π / 12) = -1/4) : 
  cos (5 * π / 6 - 2 * x) = -7 / 8 :=
sorry -- Proof steps are omitted.

end cos_value_given_sin_condition_l813_813003


namespace g_6_eq_1_l813_813913

variable (f : ℝ → ℝ)

noncomputable def g (x : ℝ) := f x + 1 - x

theorem g_6_eq_1 
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1) :
  g f 6 = 1 :=
by
  sorry

end g_6_eq_1_l813_813913


namespace white_cookies_ratio_l813_813854

noncomputable def initial_white_cookies : ℕ := 80
noncomputable def initial_black_cookies : ℕ := initial_white_cookies + 50
noncomputable def remaining_cookies : ℕ := 85
noncomputable def remaining_black_cookies : ℕ := initial_black_cookies / 2
noncomputable def remaining_white_cookies : ℕ := remaining_cookies - remaining_black_cookies

theorem white_cookies_ratio : 
  let white_cookies_eaten := initial_white_cookies - remaining_white_cookies in
  (white_cookies_eaten : ℚ) / initial_white_cookies = 3 / 4 :=
by
  sorry

end white_cookies_ratio_l813_813854


namespace shift_sin_to_cos_l813_813252

open Real

theorem shift_sin_to_cos:
  ∀ x: ℝ, 3 * cos (2 * x) = 3 * sin (2 * (x + π / 6) - π / 6) :=
by 
  sorry

end shift_sin_to_cos_l813_813252


namespace remainder_of_large_number_l813_813885

theorem remainder_of_large_number (N : ℕ) (hN : N = 123456789012): 
  N % 360 = 108 :=
by
  have h1 : N % 4 = 0 := by 
    sorry
  have h2 : N % 9 = 3 := by 
    sorry
  have h3 : N % 10 = 2 := by
    sorry
  sorry

end remainder_of_large_number_l813_813885


namespace intersecting_circles_l813_813095

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l813_813095


namespace combined_team_statistics_l813_813598

theorem combined_team_statistics
  (avg_weight_A : ℝ) (var_A : ℝ)
  (avg_weight_B : ℝ) (var_B : ℝ)
  (ratio_TeamA_TeamB : ℝ)
  (h1 : avg_weight_A = 60)
  (h2 : var_A = 200)
  (h3 : avg_weight_B = 68)
  (h4 : var_B = 300)
  (h5 : ratio_TeamA_TeamB = 1 / 3) :
  let combined_avg_weight := (1 / (1 + 3)) * 60 + (3 / (1 + 3)) * 68,
      combined_variance := (1 / 4) * (200 + (60 - 66)^2) + (3 / 4) * (300 + (68 - 66)^2) in
  combined_avg_weight = 66 ∧ combined_variance = 287 := by
  sorry

end combined_team_statistics_l813_813598


namespace sin_product_inequality_l813_813899

variable (n : ℕ)
variable (x : Fin n → ℝ)

theorem sin_product_inequality (h : ∀ i, 0 < x i ∧ x i < π) :
  ∏ i, Real.sin (x i) ≤ Real.sin (∑ i in Finset.univ, x i / n) ^ n :=
by
  sorry

end sin_product_inequality_l813_813899


namespace locus_of_Q_l813_813402

variable {R : ℝ} (P A B C Q O : ℝ × ℝ × ℝ)

-- Given conditions
def is_on_sphere (O : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) (R : ℝ) : Prop :=
  (O.1 - P.1)^2 + (O.2 - P.2)^2 + (O.3 - P.3)^2 = R^2

def are_mutually_perpendicular (P A B C : ℝ × ℝ × ℝ) : Prop :=
  ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) + (P.3 - A.3) * (P.3 - B.3) = 0) ∧
  ((P.1 - A.1) * (P.1 - C.1) + (P.2 - A.2) * (P.2 - C.2) + (P.3 - A.3) * (P.3 - C.3) = 0) ∧
  ((P.1 - B.1) * (P.1 - C.1) + (P.2 - B.2) * (P.2 - C.2) + (P.3 - B.3) * (P.3 - C.3) = 0)

-- The problem statement
theorem locus_of_Q (h1 : is_on_sphere O P R)
                   (h2 : is_on_sphere O A R)
                   (h3 : is_on_sphere O B R)
                   (h4 : is_on_sphere O C R)
                   (h5 : are_mutually_perpendicular P A B C) :
  ∃ r : ℝ, ∃ Q : ℝ × ℝ × ℝ, (r = sqrt(3 * R^2 - ((O.1 - P.1)^2 + (O.2 - P.2)^2 + (O.3 - P.3)^2))) ∧
  (∃ c : ℝ × ℝ × ℝ, is_on_sphere c Q r) := sorry

end locus_of_Q_l813_813402


namespace dividend_calculation_l813_813078

theorem dividend_calculation (divisor quotient remainder dividend : ℕ)
  (h1 : divisor = 36)
  (h2 : quotient = 20)
  (h3 : remainder = 5)
  (h4 : dividend = (divisor * quotient) + remainder)
  : dividend = 725 := 
by
  -- We skip the proof here
  sorry

end dividend_calculation_l813_813078


namespace max_side_length_l813_813778

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813778


namespace borek_thermometer_l813_813620

def bořek_temp (b : ℤ) : ℤ :=
  match b with
  | 2   => 11
  | -8  => -4
  | -2  => 5
  | _   => 0 -- Placeholder for other values

theorem borek_thermometer : bořek_temp (-2) = 5 := by
  have h1 : bořek_temp 2 = 11 := rfl
  have h2 : bořek_temp (-8) = -4 := rfl
  -- Here we need some transformations and calculations
  -- Assuming consistency in the function as per given points
  sorry

end borek_thermometer_l813_813620


namespace cakes_served_during_dinner_l813_813659

theorem cakes_served_during_dinner :
  ∀ (t l d : ℕ), l = 6 → t = 15 → d = t - l → d = 9 :=
by
  intros t l d h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  simp
  sorry

end cakes_served_during_dinner_l813_813659


namespace find_x_l813_813312

variable (x : ℕ)

def f (x : ℕ) : ℕ := 2 * x + 5
def g (y : ℕ) : ℕ := 3 * y

theorem find_x (h : g (f x) = 123) : x = 18 :=
by {
  sorry
}

end find_x_l813_813312


namespace max_side_of_triangle_l813_813740

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813740


namespace max_side_of_triangle_l813_813796

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813796


namespace max_side_of_triangle_l813_813804

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813804


namespace expenditure_proof_l813_813232

namespace OreoCookieProblem

variables (O C : ℕ) (CO CC : ℕ → ℕ) (total_items cost_difference : ℤ)

def oreo_count_eq : Prop := O = (4 * (65 : ℤ) / 13)
def cookie_count_eq : Prop := C = (9 * (65 : ℤ) / 13)
def oreo_cost (o : ℕ) : ℕ := o * 2
def cookie_cost (c : ℕ) : ℕ := c * 3
def total_item_condition : Prop := O + C = 65
def ratio_condition : Prop := 9 * O = 4 * C
def cost_difference_condition (o_cost c_cost : ℕ) : Prop := cost_difference = (c_cost - o_cost)

theorem expenditure_proof :
  (O + C = 65) →
  (9 * O = 4 * C) →
  (O = 20) →
  (C = 45) →
  cost_difference = (45 * 3 - 20 * 2) →
  cost_difference = 95 :=
by sorry

end OreoCookieProblem

end expenditure_proof_l813_813232


namespace intersecting_circles_unique_point_l813_813099

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l813_813099


namespace max_side_of_triangle_l813_813801

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813801


namespace expression_equals_neg_one_l813_813871

theorem expression_equals_neg_one (x y : ℝ) (hx : x ≠ 0) (hxy1 : x ≠ 2 * y) (hxy2 : x ≠ -2 * y) :
  (y ≠ x / 2) ∧ (y ≠ -x / 2) ↔ 
  (frac
    (frac x (x + 2 * y) + frac (2 * y) (x - 2 * y))
    (frac (2 * y) (x + 2 * y) - frac x (x - 2 * y)) = -1) :=
sorry

end expression_equals_neg_one_l813_813871


namespace no_special_points_remain_l813_813910

inductive Color
| Red
| Blue

open Color

structure Point :=
  (id : ℕ)
  (color : Color)
  (neighbors : Finset Point)

def is_special (p : Point) : Prop :=
  let diff_colored_neighbors := p.neighbors.filter (λ n, n.color ≠ p.color)
  2 * diff_colored_neighbors.card > p.neighbors.card

noncomputable def recolor (p : Point) : Point :=
  { p with color := if p.color = Red then Blue else Red }

theorem no_special_points_remain (points : Finset Point) (steps : ℕ) : ∃ n ≤ steps, ∀ p ∈ points, ¬ is_special (rec_recolor points n)
    :=
begin
  sorry  -- Proof will be provided here.
end

end no_special_points_remain_l813_813910


namespace pizza_slices_left_for_Phill_l813_813186

theorem pizza_slices_left_for_Phill :
  ∀ (initial_slices : ℕ) (first_cut : ℕ) (second_cut : ℕ) (third_cut : ℕ)
    (slices_given_to_3_friends : ℕ) (slices_given_to_2_friends : ℕ) (slices_left_for_Phill : ℕ),
    initial_slices = 1 →
    first_cut = 2 →
    second_cut = 4 →
    third_cut = 8 →
    slices_given_to_3_friends = 3 →
    slices_given_to_2_friends = 4 →
    slices_left_for_Phill = third_cut - (slices_given_to_3_friends + slices_given_to_2_friends) →
    slices_left_for_Phill = 1 :=
by {
  intros,
  subst_vars,
  simp, -- Simplify the boolean equalities
  -- We assume the steps are correct, so we leave it with sorry for now
  -- The proof should be easy for the given example and conditions.
  sorry,
}

end pizza_slices_left_for_Phill_l813_813186


namespace problem_l813_813926

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
sorry

end problem_l813_813926


namespace magnitude_z_l813_813505

open Complex

theorem magnitude_z
  (z w : ℂ)
  (h1 : abs (2 * z - w) = 25)
  (h2 : abs (z + 2 * w) = 5)
  (h3 : abs (z + w) = 2) : abs z = 9 := 
by 
  sorry

end magnitude_z_l813_813505


namespace max_side_of_triangle_l813_813752

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813752


namespace max_side_length_l813_813699

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813699


namespace multiple_of_remainder_l813_813077

theorem multiple_of_remainder (R V D Q k : ℤ) (h1 : R = 6) (h2 : V = 86) (h3 : D = 5 * Q) 
  (h4 : D = k * R + 2) (h5 : V = D * Q + R) : k = 3 := by
  sorry

end multiple_of_remainder_l813_813077


namespace part1_part2_l813_813396

theorem part1 (m : ℂ) : (m * (m + 2)).re = 0 ∧ (m^2 + m - 2).im ≠ 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (m * (m + 2) > 0 ∧ m^2 + m - 2 < 0) → 0 < m ∧ m < 1 := by
  sorry

end part1_part2_l813_813396


namespace distance_from_P_to_right_directrix_l813_813434

noncomputable def hyperbola : Prop :=
  ∃ (x y : ℝ), (x^2) / 25 - (y^2) / 144 = 1

-- Given conditions
def left_branch_point_P (P : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, P = (x, y) ∧ x < 0

def left_focus_M : (ℝ × ℝ) := (-13, 0)
def right_focus_N : (ℝ × ℝ) := (13, 0)

def distance_between_points (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def distance_P_to_left_focus (P : ℝ × ℝ) : Prop :=
  distance_between_points P left_focus_M = 16

def eccentricity : ℝ :=
  13 / 5

def distance_P_to_right_directrix (P : ℝ × ℝ) (d : ℝ) : Prop :=
  eccentricity = distance_between_points P right_focus_N / d

-- Theorem to prove
theorem distance_from_P_to_right_directrix :
  ∀ (P : ℝ × ℝ), hyperbola ∧ left_branch_point_P P ∧ distance_P_to_left_focus P →
  ∃ d : ℝ, distance_P_to_right_directrix P d ∧ d = 10 :=
sorry

end distance_from_P_to_right_directrix_l813_813434


namespace part1_part2_part3_l813_813933

-- Part 1
theorem part1 (x : ℝ) (h : abs (x + 2) = abs (x - 4)) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h : abs (x + 2) + abs (x - 4) = 8) : x = -3 ∨ x = 5 :=
by
  sorry

-- Part 3
theorem part3 (t : ℝ) :
  let M := -2 - t
  let N := 4 - 3 * t
  (abs M = abs (M - N) → t = 1/2) ∧ 
  (N = 0 → t = 4/3) ∧
  (abs N = abs (N - M) → t = 2) ∧
  (M = N → t = 3) ∧
  (abs (M - N) = abs (2 * M) → t = 8) :=
by
  sorry

end part1_part2_part3_l813_813933


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813695

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813695


namespace eval_expression_l813_813870

theorem eval_expression : 
  (sqrt 5 * 5^(1/2) + 20 / 5 * 2 - 9^(3/2)) = -14 := 
by 
  sorry

end eval_expression_l813_813870


namespace sufficient_not_necessary_condition_l813_813015

variable {m : ℝ}

def p := m < -2
def q := (4 + 4 * m) < 0

theorem sufficient_not_necessary_condition :
  (p → q) ∧ ¬(q → p) :=
by
  sorry

end sufficient_not_necessary_condition_l813_813015


namespace nancy_shirts_l813_813173

theorem nancy_shirts (loads : ℕ) (pieces_per_load : ℕ) (sweaters : ℕ) (shirts : ℕ) 
  (h_loads : loads = 3) (h_pieces_per_load : pieces_per_load = 9) (h_sweaters : sweaters = 8) 
  (h_total : loads * pieces_per_load = shirts + sweaters) : 
  shirts = 19 :=
by 
  rw [h_loads, h_pieces_per_load, h_sweaters] at h_total
  have h : 3 * 9 = shirts + 8 := h_total
  norm_num at h
  exact h

end nancy_shirts_l813_813173


namespace jamal_max_points_l813_813464

theorem jamal_max_points
  (x y : ℕ)
  (h1 : x + y = 40)
  (h2 : 0.25 * x = n)
  (h3 : 0.40 * y = m) 
  : 
  let points := (3 * n) + (2 * m) in 
  points ≤ 32 := 
sorry

end jamal_max_points_l813_813464


namespace smallest_k_divides_polynomial_l813_813889

noncomputable def is_divisible (f g : polynomial complex) : Prop :=
∃ q, f = q * g

theorem smallest_k_divides_polynomial:
  let f := (polynomial.C (1:complex) * polynomial.X ^ 12 +
            polynomial.C (1:complex) * polynomial.X ^ 11 +
            polynomial.C (1:complex) * polynomial.X ^ 8 +
            polynomial.C (1:complex) * polynomial.X ^ 7 +
            polynomial.C (1:complex) * polynomial.X ^ 6 +
            polynomial.C (1:complex) * polynomial.X ^ 3 +
            polynomial.C (1:complex)) in
  ∃ k : ℕ, k > 0 ∧ is_divisible (polynomial.X ^ k - 1) f ∧ k = 120 :=
begin
  intros f,
  use [120],
  split,
  { linarith, },
  split,
  { sorry, },
  { refl, },
end

end smallest_k_divides_polynomial_l813_813889


namespace radius_increase_correct_l813_813562

noncomputable def radius_increase (C1 C2 : ℝ) : ℝ :=
  (C2 - C1) / (2 * Real.pi)

theorem radius_increase_correct (C1 C2 : ℝ) (hC1 : C1 = 30) (hC2 : C2 = 40) : 
  radius_increase C1 C2 = 5 / Real.pi :=
by
  rw [radius_increase, hC1, hC2]
  norm_num
  field_simp
  ring
  simp only [Real.pi_pos]
  norm_num

#eval radius_increase 30 40 -- This should output 5 / Real.pi

end radius_increase_correct_l813_813562


namespace gain_percent_calculation_l813_813946

theorem gain_percent_calculation (gain_paise : ℕ) (cost_price_rupees : ℕ) (rupees_to_paise : ℕ)
  (h_gain_paise : gain_paise = 70)
  (h_cost_price_rupees : cost_price_rupees = 70)
  (h_rupees_to_paise : rupees_to_paise = 100) :
  ((gain_paise / rupees_to_paise) / cost_price_rupees) * 100 = 1 :=
by
  -- Placeholder to indicate the need for proof
  sorry

end gain_percent_calculation_l813_813946


namespace total_guppies_correct_l813_813827

-- Define the initial conditions as variables
def initial_guppies : ℕ := 7
def baby_guppies_1 : ℕ := 3 * 12
def baby_guppies_2 : ℕ := 9

-- Define the total number of guppies
def total_guppies : ℕ := initial_guppies + baby_guppies_1 + baby_guppies_2

-- Theorem: Proving the total number of guppies is 52
theorem total_guppies_correct : total_guppies = 52 :=
by
  sorry

end total_guppies_correct_l813_813827


namespace max_side_length_of_triangle_l813_813756

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813756


namespace cos_690_eq_sqrt3_div_2_l813_813379

theorem cos_690_eq_sqrt3_div_2 : Real.cos (690 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_690_eq_sqrt3_div_2_l813_813379


namespace find_angle_B_find_area_l813_813461

namespace TriangleProof

-- Define the sides and angles of triangle ABC
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define the cosine condition
theorem find_angle_B (h1 : ∀ A B C : ℝ, a * cos B = -(a * cos C) - (b * sin (B + C)))
  (h2 : ∀ A B C : ℝ, cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (h3 : ∀ A B C : ℝ, cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  B = 2 * π / 3 :=
begin
  -- proof required
  sorry
end

-- Define the area condition
theorem find_area (h1 : b = sqrt 13)
  (h2 : a + c = 4)
  (h3 : B = 2 * π / 3) :
  let area := (1 / 2) * a * c * sin B in 
  area = (3 * sqrt 3) / 4 :=
begin
  -- proof required
  sorry
end

end TriangleProof

end find_angle_B_find_area_l813_813461


namespace helga_ratio_l813_813053

variable (a b c d : ℕ)

def helga_shopping (a b c d total_shoes pairs_first_three : ℕ) : Prop :=
  a = 7 ∧
  b = a + 2 ∧
  c = 0 ∧
  a + b + c + d = total_shoes ∧
  pairs_first_three = a + b + c ∧
  total_shoes = 48 ∧
  (d : ℚ) / (pairs_first_three : ℚ) = 2

theorem helga_ratio : helga_shopping 7 9 0 32 48 16 := by
  sorry

end helga_ratio_l813_813053


namespace Ara_height_in_inches_l813_813540

theorem Ara_height_in_inches (Shea_current_height : ℝ) (Shea_growth_percentage : ℝ) (Ara_growth_factor : ℝ) (Shea_growth_amount : ℝ) (Ara_current_height : ℝ) :
  Shea_current_height = 75 →
  Shea_growth_percentage = 0.25 →
  Ara_growth_factor = 1 / 3 →
  Shea_growth_amount = 75 * (1 / (1 + 0.25)) * 0.25 →
  Ara_current_height = 75 * (1 / (1 + 0.25)) + (75 * (1 / (1 + 0.25)) * 0.25) * (1 / 3) →
  Ara_current_height = 65 :=
by sorry

end Ara_height_in_inches_l813_813540


namespace find_e_l813_813575

theorem find_e (d e f : ℤ) (Q : ℤ → ℤ) (hQ : ∀ x, Q x = 3 * x^3 + d * x^2 + e * x + f)
  (mean_zeros_eq_prod_zeros : let zeros := {x // Q x = 0} in
    (∑ x in zeros, x) / 3 = ∏ x in zeros, x)
  (sum_coeff_eq_mean_zeros : 3 + d + e + f = (∑ x in {x // Q x = 0}, x) / 3)
  (y_intercept : Q 0 = 9) :
  e = -42 :=
sorry

end find_e_l813_813575


namespace weight_of_second_new_player_l813_813601

theorem weight_of_second_new_player 
  (total_weight_seven_players : ℕ)
  (average_weight_seven_players : ℕ)
  (total_players_with_new_players : ℕ)
  (average_weight_with_new_players : ℕ)
  (weight_first_new_player : ℕ)
  (W : ℕ) :
  total_weight_seven_players = 7 * average_weight_seven_players →
  total_players_with_new_players = 9 →
  average_weight_with_new_players = 106 →
  weight_first_new_player = 110 →
  (total_weight_seven_players + weight_first_new_player + W) / total_players_with_new_players = average_weight_with_new_players →
  W = 60 := 
by sorry

end weight_of_second_new_player_l813_813601


namespace sum_prime_factors_2_10_minus_1_l813_813593

theorem sum_prime_factors_2_10_minus_1 : 
  let n := 10 
  let number := 2^n - 1 
  let factors := [3, 5, 7, 11] 
  number.prime_factors.sum = 26 :=
by
  sorry

end sum_prime_factors_2_10_minus_1_l813_813593


namespace equal_sundays_tuesdays_l813_813310

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l813_813310


namespace quadrilateral_side_length_l813_813197

variable (A B C D : Type)  -- Define the vertices of the quadrilateral
variable [hasAngle A B C D]  -- Define the angles of the quadrilateral
variable [hasLength A B C D]  -- Define the lengths of the sides of the quadrilateral

theorem quadrilateral_side_length
    (h1: AB = 20)
    (h2: ∠A = 45)
    (h3: AB ∥ CD)
    (h4: is_arithmetic_progression [AB, BC, CD, DA])
    : CD = 14 := sorry

end quadrilateral_side_length_l813_813197


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813685

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813685


namespace Nina_saves_enough_to_buy_video_game_in_11_weeks_l813_813522

-- Definitions (directly from conditions)
def game_cost : ℕ := 50
def tax_rate : ℚ := 10 / 100
def sales_tax (cost : ℕ) (rate : ℚ) : ℚ := cost * rate
def total_cost (cost : ℕ) (tax : ℚ) : ℚ := cost + tax
def weekly_allowance : ℕ := 10
def savings_rate : ℚ := 1 / 2
def weekly_savings (allowance : ℕ) (rate : ℚ) : ℚ := allowance * rate
def weeks_to_save (total_cost : ℚ) (savings_per_week : ℚ) : ℚ := total_cost / savings_per_week

-- Theorem to prove
theorem Nina_saves_enough_to_buy_video_game_in_11_weeks :
  weeks_to_save
    (total_cost game_cost (sales_tax game_cost tax_rate))
    (weekly_savings weekly_allowance savings_rate) = 11 := by
-- We skip the proof for now, as per instructions
  sorry

end Nina_saves_enough_to_buy_video_game_in_11_weeks_l813_813522


namespace correct_operation_l813_813269

theorem correct_operation :
  (sqrt 2) * (sqrt 6) = 2 * (sqrt 3) := 
sorry

end correct_operation_l813_813269


namespace intersecting_circles_l813_813096

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l813_813096


namespace equation_of_plane_l813_813878

noncomputable def points := [( -3, 5, -2), (1, 5, 0), (3, 3, -1)]

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.2 * v2.3 - v1.3 * v2.2,
   v1.3 * v2.1 - v1.1 * v2.3,
   v1.1 * v2.2 - v1.2 * v2.1)

def normal_vector (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let v1 := vector_sub p2 p1
  let v2 := vector_sub p3 p1
  cross_product v1 v2

-- Ensure the normal vector satisfies the form by reducing it
def normalized_vector (v : ℝ × ℝ × ℝ) :=
  let g := Int.gcd_nnnorm (v.1.to_int, Int.gcd_nnnorm (v.2.to_int, v.3.to_int))
  (v.1 / g, v.2 / g, v.3 / g)

def plane_equation (p : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  n.1 * p.1 + n.2 * p.2 + n.3 * p.3

theorem equation_of_plane :
  ∃ (A B C D : ℤ), 
    let n := normalized_vector (normal_vector (-3, 5, -2) (1, 5, 0) (3, 3, -1)) in
    let (A, B, C) := (n.1, n.2, n.3) in
    let D := -plane_equation (-3, 5, -2) (A, B, C) in
    A > 0 ∧ Int.gcd (abs A) (Int.gcd (abs B) (abs C)) = 1 ∧
    (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + D = 0 :=
by
  sorry

end equation_of_plane_l813_813878


namespace scientific_notation_of_taichulight_performance_l813_813081

noncomputable def trillion := 10^12

def convert_to_scientific_notation (x : ℝ) (n : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ x * 10^n = 12.5 * trillion

theorem scientific_notation_of_taichulight_performance :
  ∃ (x : ℝ) (n : ℤ), convert_to_scientific_notation x n ∧ x = 1.25 ∧ n = 13 :=
by
  unfold convert_to_scientific_notation
  use 1.25
  use 13
  sorry

end scientific_notation_of_taichulight_performance_l813_813081


namespace find_max_side_length_l813_813795

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813795


namespace chosen_number_l813_813276

theorem chosen_number :
  ∃ x : ℤ, (x / 9) - 100 = 10 := 
begin
  use 990,
  -- The actual proof would go here, but we add sor, for now
  sorry,
end

end chosen_number_l813_813276


namespace max_side_of_triangle_l813_813803

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813803


namespace number_of_right_triangles_l813_813347

noncomputable def rectangle := Type

def is_rectangle (A B C D P Q R : rectangle) (one_third two_thirds : ℝ) (congruent : list rectangle) : Prop :=
  ∃ (length : ℝ), length = (dist A B) ∧ (dist P Q = length / 3) ∧ (dist Q R = 2 * length / 3) ∧
    (midpoint P Q P) ∧ (midpoint R Q Q) ∧ 
    (congruent_squares : rectangle → rectangle → list rectangle) (congruent_squares A B C D = congruent)

theorem number_of_right_triangles (A B C D P Q R : rectangle) (one_third two_thirds : ℝ) (congruent : list rectangle) :
  is_rectangle A B C D P Q R one_third two_thirds congruent → 
  count_right_triangles {A, P, Q, R, B, C, D} = 12 := 
  by sorry

end number_of_right_triangles_l813_813347


namespace max_triangle_side_l813_813715

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813715


namespace range_of_a_l813_813594

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

theorem range_of_a (a : ℝ)
  (h₁ : ∀ x ∈ Icc (-2 : ℝ) 4, f x a ∈ Icc (f a a) (f 4 a))
  (h₂ : f a a ≤ f 4 a): 
  -2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l813_813594


namespace max_triangle_side_l813_813717

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813717


namespace solve_for_x_l813_813543

theorem solve_for_x (x : ℝ) :
  (2 * x - 30) / 3 = (5 - 3 * x) / 4 + 1 → x = 147 / 17 := 
by
  intro h
  sorry

end solve_for_x_l813_813543


namespace find_distance_l813_813290

noncomputable def is_distance (v t : ℝ) : Prop :=
  let d := v * t in
  (d = (v + 1) * (2 * t / 3)) ∧ 
  (d = (v - 1) * (t + 1))

theorem find_distance : ∃ d : ℝ, ∀ v t : ℝ, is_distance v t → d = 2 :=
begin
  sorry
end

end find_distance_l813_813290


namespace smallest_n_l813_813150

def g (n : ℕ) : ℕ :=
  Nat.find (λ m, n ∣ Nat.factorial m)

theorem smallest_n (n : ℕ) (h : n % 21 = 0) (h_g : g n > 21) : n = 483 := by
  sorry

end smallest_n_l813_813150


namespace max_triangle_side_l813_813719

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813719


namespace log_five_one_over_sqrt_five_l813_813363

theorem log_five_one_over_sqrt_five : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_five_one_over_sqrt_five_l813_813363


namespace equation_of_line_through_point_l813_813375

-- Define the conditions
lemma line_through_point_with_slope_angle (P : ℝ × ℝ) (k : ℝ) (hP : P = (-4, 3)) (hk : k = Real.tan (Real.pi / 4)) :
  (∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ ∀ (x y : ℝ), y - P.2 = k * (x - P.1) → a * x + b * y + c = 0) :=
sorry

-- Prove that the equation of the line through point P=(-4,3) with slope angle 45° is x - y + 7 = 0
theorem equation_of_line_through_point (P : ℝ × ℝ) (k : ℝ) (hP : P = (-4, 3)) (hk : k = Real.tan (Real.pi / 4)) :
  ∃ a b c : ℝ, a * (-4) + b * 3 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 7 :=
begin
  sorry
end

end equation_of_line_through_point_l813_813375


namespace inequality_always_holds_l813_813159

-- Given conditions
variables {a b : ℝ}
hypothesis ha : a > 1
hypothesis hb1 : b < 1
hypothesis hb2 : -1 < b

-- To prove
theorem inequality_always_holds :
  a > b^2 :=
sorry

end inequality_always_holds_l813_813159


namespace probability_multiple_of_three_l813_813287

theorem probability_multiple_of_three : 
  ∀ (bag : Finset ℕ), 
  bag = {1, 2, 3, 4, 5} → 
  (∃ (draw : Finset ℕ), draw.card = 3 ∧ (∑ x in draw, x) % 3 = 0) →
  ∃ (num_valid : ℕ) (total_combinations : ℕ), 
  (total_combinations = 60) ∧ (num_valid = 6) ∧ (num_valid / total_combinations = 1 / 10) :=
begin
  sorry
end

end probability_multiple_of_three_l813_813287


namespace digit_to_make_multiple_of_5_l813_813614

theorem digit_to_make_multiple_of_5 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 9) 
  (N := 71360 + d) : (N % 5 = 0) → (d = 0 ∨ d = 5) :=
by
  sorry

end digit_to_make_multiple_of_5_l813_813614


namespace find_certain_number_l813_813064

noncomputable def certain_number_is_square (n : ℕ) (x : ℕ) : Prop :=
  ∃ (y : ℕ), x * n = y * y

theorem find_certain_number : ∃ x, certain_number_is_square 3 x :=
by 
  use 1
  unfold certain_number_is_square
  use 3
  sorry

end find_certain_number_l813_813064


namespace find_m_l813_813151

variables {V : Type*} [AddCommGroup V] [Module ℝ V] [CrossProduct V]

theorem find_m 
  (m : ℝ) 
  (u v w : V)
  (h1 : u + v + w = 0) 
  (h2 : m • (v × u) + (v × w) + (w × u) = v × u) : 
  m = 3 := 
sorry

end find_m_l813_813151


namespace abs_diff_f_g_gt_2_l813_813153

noncomputable def f (x : ℝ) : ℝ :=
  ∑ i in Finset.range 1010, 1 / (x - 2 * i)

noncomputable def g (x : ℝ) : ℝ :=
  ∑ i in Finset.range 1009, 1 / (x - (2 * i + 1))

theorem abs_diff_f_g_gt_2 {x : ℝ} (hx1 : 0 < x) (hx2 : x < 2018) (hx3 : ¬ (∃ n : ℤ, x = n)) :
  |f x - g x| > 2 :=
sorry

end abs_diff_f_g_gt_2_l813_813153


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813813

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813813


namespace symmetric_line_equation_l813_813214

theorem symmetric_line_equation (l1 l2 l3 : (ℝ × ℝ) → Prop)
  (h1 : ∀ x y : ℝ, l1 (x, y) ↔ 2 * x - y + 1 = 0)
  (h2 : ∀ x y : ℝ, l2 (x, y) ↔ x + y + 2 = 0)
  (h3 : ∀ x y : ℝ, l3 (x, y) ↔ x - 2 * y - 1 = 0) :
  l3 = {
    -- Prove that l3 is the equation of the line symmetric to l1 with respect to l2
    (λ p, ∃ q : ℝ × ℝ, l1 q ∧ exists s : ℝ × ℝ, 
    l2 ((p.1 + q.1) / 2, (p.2 + q.2) / 2) ∧ l3 p)
  } := sorry

end symmetric_line_equation_l813_813214


namespace abc_gcd_triples_l813_813876

theorem abc_gcd_triples (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_eq : a + Nat.gcd a b = b + Nat.gcd b c = c + Nat.gcd c a) :
  ∃ k : ℕ, a = k ∧ b = k ∧ c = k :=
sorry

end abc_gcd_triples_l813_813876


namespace censusSurveys_l813_813333

-- Definitions corresponding to the problem conditions
inductive Survey where
  | TVLifespan
  | ManuscriptReview
  | PollutionInvestigation
  | StudentSizeSurvey

open Survey

-- The aim is to identify which surveys are more suitable for a census.
def suitableForCensus (s : Survey) : Prop :=
  match s with
  | TVLifespan => False  -- Lifespan destruction implies sample survey.
  | ManuscriptReview => True  -- Significant and needs high accuracy, thus census.
  | PollutionInvestigation => False  -- Broad scope implies sample survey.
  | StudentSizeSurvey => True  -- Manageable scope makes census appropriate.

-- The theorem to be formalized.
theorem censusSurveys : (suitableForCensus ManuscriptReview) ∧ (suitableForCensus StudentSizeSurvey) :=
  by sorry

end censusSurveys_l813_813333


namespace find_t_l813_813883

-- Definitions of the given conditions
def a : ℂ := sorry 
def b : ℂ := sorry

-- The values of the moduli
def mod_a : ℝ := 3
def mod_b : ℝ := 5

axiom abs_a : complex.abs a = mod_a
axiom abs_b : complex.abs b = mod_b

-- Define the product ab and set it equal to t - 3i
def t : ℝ := sorry
axiom eq_ab : a * b = t - 3 * complex.i

-- The proof to be completed
theorem find_t : t = 6 * real.sqrt 6 :=
by
  -- Proof steps here (proof is omitted)
  sorry

end find_t_l813_813883


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813811

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813811


namespace diamond_chain_eq_one_l813_813353

def diamond (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

theorem diamond_chain_eq_one : 
  diamond 1 (List.foldr diamond 500 (List.range' 2 499)) = 1 := 
by
  sorry

end diamond_chain_eq_one_l813_813353


namespace distance_light_in_50_years_l813_813564

/-- The distance light travels in one year, given in scientific notation -/
def distance_light_per_year : ℝ := 9.4608 * 10^12

/-- The distance light travels in 50 years is calculated -/
theorem distance_light_in_50_years :
  distance_light_per_year * 50 = 4.7304 * 10^14 :=
by
  -- the proof is not demanded, so we use sorry
  sorry

end distance_light_in_50_years_l813_813564


namespace max_side_length_is_11_l813_813672

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813672


namespace imaginary_part_conjugate_z_l813_813962

theorem imaginary_part_conjugate_z (z : ℂ) (h : (complex.I * z = -((1 + complex.I) / 2))) :
  complex.im (complex.conj z) = -1/2 :=
sorry

end imaginary_part_conjugate_z_l813_813962


namespace cube_perpendicular_pairs_l813_813466

theorem cube_perpendicular_pairs 
  (cube : Type)
  (edges_of_cube : cube → set (fin 12))
  (faces_of_cube : cube → set (fin 6))
  (edges_per_face : ∀ {f : faces_of_cube cube}, set (fin 4))
  (faces_per_edge : ∀ {e : edges_of_cube cube}, set (fin 2))
  (unique_perpendicular_edge_face : ∀ {f : faces_of_cube cube}, ∃! (e : edges_per_face), e ⊥ f)
  (unique_perpendicular_face_face : ∀ {e : edges_of_cube cube}, ∃! (perp_faces : faces_per_edge), perp_faces ⊥ e) : 
  ∃ (p_lp_pairs p_p_pairs : ℕ), p_lp_pairs = 24 ∧ p_p_pairs = 12 :=
  sorry

end cube_perpendicular_pairs_l813_813466


namespace isosceles_right_triangle_inscribed_circle_l813_813979

theorem isosceles_right_triangle_inscribed_circle
  (h r x : ℝ)
  (h_def : h = 2 * r)
  (r_def : r = Real.sqrt 2 / 4)
  (x_def : x = h - r) :
  x = Real.sqrt 2 / 4 :=
by
  sorry

end isosceles_right_triangle_inscribed_circle_l813_813979


namespace max_side_length_l813_813710

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813710


namespace perp_lines_iff_parallel_lines_iff_l813_813049

section
variables {m : ℝ}

def line1 (x y : ℝ) := x + m * y + 6 = 0
def line2 (x y : ℝ) := (m - 2) * x + 3 * y + 2 * m = 0

theorem perp_lines_iff : (∀ {x y : ℝ}, line1 x y → line2 x y → true) ↔ m = 1/2 :=
by sorry

theorem parallel_lines_iff : (∀ {x y : ℝ}, line1 x y ↔ line2 x y → true) ↔ m = -1 :=
by sorry
end

end perp_lines_iff_parallel_lines_iff_l813_813049


namespace vector_u_l813_813381

def proj (v u : Vector ℝ 2) : Vector ℝ 2 :=
  let scalar := (u ⬝ v) / (v ⬝ v)
  scalar • v

theorem vector_u :
  ∃ (u : Vector ℝ 2),
    proj (Vector.ofFn ![3, 1]) u = Vector.ofFn ![45 / 10, 15 / 10] ∧
    proj (Vector.ofFn ![1, 2]) u = Vector.ofFn ![36 / 5, 72 / 5] ∧
    u = Vector.ofFn ![-6 / 5, 93 / 5] :=
by
  sorry

end vector_u_l813_813381


namespace max_area_CDFE_l813_813495

theorem max_area_CDFE :
  ∀ (AB AD : ℝ) (x : ℝ) (AE AF : ℝ),
    AB = 2 → 
    AD = 1 →
    AE = x → 
    AF = x / 2 →
    (1 ≤ x ∧ x ≤ 2) →
    (AE = x ∧ AE = 2 * AF) →
    let area : ℝ := (5 * x / 4) - (3 / 4 * x^2) in
    ∃ (x : ℝ), (1 ≤ x ∧ x ≤ 2) ∧ area = 5 / 16 :=
by
  sorry

end max_area_CDFE_l813_813495


namespace valid_starting_days_count_l813_813306

def is_valid_starting_day (d : ℕ) : Prop :=
  (d % 7 = 3 ∨ d % 7 = 4 ∨ d % 7 = 5)

theorem valid_starting_days_count : 
  (finset.filter is_valid_starting_day (finset.range 7)).card = 3 :=
begin
  sorry
end

end valid_starting_days_count_l813_813306


namespace max_triangle_side_l813_813713

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813713


namespace percentage_of_students_on_trip_l813_813955

-- Define the problem context
variable (total_students : ℕ)
variable (students_more_100 : ℕ)
variable (students_on_trip : ℕ)

-- Define the conditions as per the problem
def condition_1 : Prop := students_more_100 = total_students * 15 / 100
def condition_2 : Prop := students_more_100 = students_on_trip * 25 / 100

-- Define the problem statement
theorem percentage_of_students_on_trip
  (h1 : condition_1 total_students students_more_100)
  (h2 : condition_2 students_more_100 students_on_trip) :
  students_on_trip = total_students * 60 / 100 :=
by
  sorry

end percentage_of_students_on_trip_l813_813955


namespace min_positive_period_f_mono_inc_interval_f_min_value_g_l813_813427

open Real

noncomputable def f(x : ℝ) := 2 * sqrt(3) * sin x * cos x + 1 - 2 * (sin x)^2
noncomputable def g(x : ℝ) := 2 * sin (4 * x + 5 * π / 6)

theorem min_positive_period_f : (∃ T > 0, ∀ x, f(x) = f(x + T)) ↔ T = π := sorry

theorem mono_inc_interval_f (k : ℤ) : ∀ x ∈ Icc (k * π - π / 3) (k * π + π / 6), 
  ∃ c, ∀ y ∈ Icc (k * π - π / 3) x, f(y) ≤ f(y + c) := sorry

theorem min_value_g 
  : ∃ x ∈ Icc 0 (π / 8), ∀ y ∈ Icc 0 (π / 8), g(x) ≤ g(y) ∧ g(x) = -sqrt(3) := sorry

end min_positive_period_f_mono_inc_interval_f_min_value_g_l813_813427


namespace work_together_days_l813_813635

theorem work_together_days :
  let a_rate := 1 / 15
  let b_rate := 1 / 20
  let combined_rate := a_rate + b_rate
  let total_work := 1
  let remaining_work := 0.65
  let completed_work := total_work - remaining_work
  ∃ d : ℝ, combined_rate * d = completed_work ∧ d = 3 :=
by
  let a_rate := (1 / (15:ℝ))
  let b_rate := (1 / (20:ℝ))
  let combined_rate := a_rate + b_rate
  let total_work := 1
  let remaining_work := 0.65
  let completed_work := total_work - remaining_work
  have h1 : ∃ d : ℝ, combined_rate * d = completed_work,
    from ⟨3, by linarith [combined_rate * 3 = completed_work]⟩,
  exact h1

#check work_together_days -- just to verify the theorem is well-formed in the Lean environment

end work_together_days_l813_813635


namespace colorings_exists_l813_813366

/- Definitions and assumptions from the problem -/
def color (k : Int) : Fin 5 :=
if k = 0 then 0 else (k / (5 ^ (Nat.find (fun m => 5^m ∣ k ∧ ¬ 5^(m+1) ∣ k)))) % 5

theorem colorings_exists :
  ∃ (color : Int → Fin 5), 
    ∀ (a b c d : Int), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ color a = color b ∧ color b = color c ∧ color c = color d →
      3 * a - 2 * b ≠ 2 * c - 3 * d := by
  sorry

end colorings_exists_l813_813366


namespace participants_with_six_points_l813_813471

theorem participants_with_six_points (n : ℕ) (h : n = 9) : 
  let num_participants := 2^n,
      final_score_count (num_points : ℕ) := 0 ≤ num_points ∧ num_points ≤ n → 
        ∑ k in finset.range (n + 1), if k = 6 then 
          Nat.choose n k else 0 
  in final_score_count 6 = 84 :=
  sorry

end participants_with_six_points_l813_813471


namespace part1_part2_l813_813395

theorem part1 (m : ℂ) : (m * (m + 2)).re = 0 ∧ (m^2 + m - 2).im ≠ 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (m * (m + 2) > 0 ∧ m^2 + m - 2 < 0) → 0 < m ∧ m < 1 := by
  sorry

end part1_part2_l813_813395


namespace fifty_seventh_pair_is_2_10_l813_813012

def sequence_of_int_pairs : Nat → (Nat × Nat) :=
λ n, 
  let sum := Nat.find (λ k, k * (k - 1) / 2 ≥ n) in
  let previous_sum_count := (sum - 1) * (sum - 2) / 2 in
  let position_in_sum := n - previous_sum_count - 1 in
  (position_in_sum + 1, sum - (position_in_sum + 1))

theorem fifty_seventh_pair_is_2_10 :
  sequence_of_int_pairs 57 = (2, 10) :=
sorry

end fifty_seventh_pair_is_2_10_l813_813012


namespace new_milk_water_ratio_l813_813973

def initial_milk (total_volume : ℕ) (ratio_milk_water : ℕ × ℕ) : ℕ :=
  (ratio_milk_water.1 * total_volume) / (ratio_milk_water.1 + ratio_milk_water.2)

def initial_water (total_volume : ℕ) (ratio_milk_water : ℕ × ℕ) : ℕ :=
  (ratio_milk_water.2 * total_volume) / (ratio_milk_water.1 + ratio_milk_water.2)

def new_ratio (initial_milk : ℕ) (initial_water : ℕ) (added_water : ℕ) : ℕ × ℕ :=
  let new_water := initial_water + added_water in
  (initial_milk, new_water)

theorem new_milk_water_ratio (total_volume initial_ratio_milk_water added_water : ℕ) :
  let initial_milk := initial_milk total_volume initial_ratio_milk_water,
      initial_water := initial_water total_volume initial_ratio_milk_water,
      new_ratio := new_ratio initial_milk initial_water added_water
  in new_ratio = (1, 2) :=
by
  sorry

end new_milk_water_ratio_l813_813973


namespace length_of_XY_l813_813476

theorem length_of_XY (W X Y Z : Type)
  [trapezoid W X Y Z] (h1 : parallel WX ZY) (h2 : perpendicular WY ZY)
  (hYZ : YZ = 15) (htanZ : tan Z = 2) (htanX : tan X = 2.5) :
  XY = 2 * sqrt 261 :=
by
  sorry

end length_of_XY_l813_813476


namespace triangle_DEF_is_right_isosceles_l813_813139

open EuclideanGeometry

noncomputable def triangle.isosceles_right (A B C : Point) : Prop :=
  ∃ (D E F : Point),
    -- Conditions for point D
    (dist A D = dist A B) ∧
    (angle D A B = 90) ∧
    -- Conditions for point E
    (dist A E = dist A C) ∧
    (angle E A C = 90) ∧
    -- Conditions for point F
    (dist F B = dist F C) ∧
    (angle B F C = 90) ∧
    -- Properties to prove: right-isosceles triangle DEF
    (dist D F = dist F E) ∧
    (angle D F E = 90)

theorem triangle_DEF_is_right_isosceles
  (A B C : Point) :
  triangle.isosceles_right A B C :=
begin
  sorry
end

end triangle_DEF_is_right_isosceles_l813_813139


namespace intersecting_circles_l813_813097

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l813_813097


namespace max_side_of_triangle_l813_813744

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813744


namespace complex_circle_intersection_l813_813112

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l813_813112


namespace digit_sum_congruence_l813_813158

noncomputable def S_q (q x : ℕ) : ℕ := sorry -- Definition of the sum of the digits of x in base q

theorem digit_sum_congruence
    (a b b' c m q M : ℕ)
    (ha : 0 < a)
    (hb : 0 < b)
    (hb' : 0 < b')
    (hc : 0 < c)
    (hm : 1 < m)
    (hq : 1 < q)
    (h_diff : |b - b'| ≥ a)
    (hM : ∀ n, n ≥ M → S_q q (a * n + b) ≡ S_q q (a * n + b') + c [MOD m]) :
  ∀ n, 0 < n → S_q q (a * n + b) ≡ S_q q (a * n + b') + c [MOD m] := sorry

end digit_sum_congruence_l813_813158


namespace medians_sum_le_circumradius_l813_813462

-- Definition of the problem
variable (a b c R : ℝ) (m_a m_b m_c : ℝ)

-- Conditions: medians of triangle ABC, and R is the circumradius
def is_median (m : ℝ) (a b c : ℝ) : Prop :=
  m^2 = (2*b^2 + 2*c^2 - a^2) / 4

-- Main theorem to prove
theorem medians_sum_le_circumradius (h_ma : is_median m_a a b c)
  (h_mb : is_median m_b b a c) (h_mc : is_median m_c c a b) 
  (h_R : a^2 + b^2 + c^2 ≤ 9 * R^2) :
  m_a + m_b + m_c ≤ 9 / 2 * R :=
sorry

end medians_sum_le_circumradius_l813_813462


namespace length_of_train_approx_100_l813_813669

noncomputable def length_of_train (t : ℝ) (v_m : ℝ) (v_t : ℝ) : ℝ :=
  let relative_speed_km_hr := v_t - v_m
  let relative_speed_m_s := (relative_speed_km_hr * (1000 / 3600))
  in relative_speed_m_s * t

theorem length_of_train_approx_100 :
  length_of_train 5.999520038396929 3 63 ≈ 100 := by
  sorry

end length_of_train_approx_100_l813_813669


namespace cost_of_eraser_carton_l813_813529

/-- An order consists of 100 cartons, including 20 cartons of pencils costing 6 dollars per carton. 
     The total cost of the order is 360 dollars. Prove that the cost of a carton of erasers is 3 dollars. -/
theorem cost_of_eraser_carton 
  (total_cartons : ℕ) 
  (pencil_cartons : ℕ) 
  (pencil_cost_per_carton : ℚ)
  (eraser_cost_per_carton : ℚ)
  (total_cost : ℚ)
  (total_pencil_cost : ℚ)
  (eraser_cartons : ℕ) 
  (remaining_cost : ℚ)
  (cost_per_eraser_carton : ℚ) 
  (h1 : total_cartons = 100)
  (h2 : pencil_cartons = 20)
  (h3 : pencil_cost_per_carton = 6)
  (h4 : total_cost = 360)
  (h5 : total_pencil_cost = (pencil_cartons * pencil_cost_per_carton))
  (h6 : eraser_cartons = (total_cartons - pencil_cartons))
  (h7 : remaining_cost = (total_cost - total_pencil_cost))
  (h8 : cost_per_eraser_carton = (remaining_cost / eraser_cartons)) 
  : eraser_cost_per_carton = 3 :=
by 
  -- Definition of each step is stated as hypotheses
  have h_total_pencil_cost : total_pencil_cost = (20 * 6) := by rw [h2, h3]; rfl
  have h_eraser_cartons : eraser_cartons = (100 - 20) := by rw [h1, h2]; rfl
  have h_remaining_cost : remaining_cost = (360 - 120) := by rw [h4, h_total_pencil_cost]; rfl
  have h_cost_per_eraser : cost_per_eraser_carton = (240 / 80) := 
    by rw [h8]; rw [h_remaining_cost, h_eraser_cartons]; rfl
  
  -- Final proof by asserting eraser_cost_per_carton is 3 
  have h_final : eraser_cost_per_carton = 3 := by 
    rw [h5, h6, h7, h8, h_total_pencil_cost, h_eraser_cartons, h_remaining_cost, h_cost_per_eraser]; exact rfl
  exact h_final

end cost_of_eraser_carton_l813_813529


namespace largest_sum_of_three_2_digit_numbers_l813_813327

theorem largest_sum_of_three_2_digit_numbers : 
  ∀ (a b c d e f : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
  0 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 ∧ 
  0 ≤ e ∧ e ≤ 9 ∧ 
  0 ≤ f ∧ f ≤ 9 →
  let sum := 10 * (a + b + c) + (d + e + f) in
  sum ≤ 255 ∧ 
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
  x / 10 = a ∧ x % 10 = d ∧ y / 10 = b ∧ y % 10 = e ∧ z / 10 = c ∧ z % 10 = f ∧
  x + y + z = 255 := 
sorry

end largest_sum_of_three_2_digit_numbers_l813_813327


namespace hexagon_coloring_unique_l813_813227

-- Define the coloring of the hexagon using enumeration
inductive Color
  | green
  | blue
  | orange

-- Assume we have a function that represents the coloring of the hexagons
-- with the constraints given in the problem
def is_valid_coloring (coloring : ℕ → ℕ → Color) : Prop :=
  ∀ x y : ℕ, -- For all hexagons
  (coloring x y = Color.green ∧ x = 0 ∧ y = 0) ∨ -- The labeled hexagon G is green
  (coloring x y ≠ coloring (x + 1) y ∧ -- No two hexagons with a common side have the same color
   coloring x y ≠ coloring (x - 1) y ∧ 
   coloring x y ≠ coloring x (y + 1) ∧
   coloring x y ≠ coloring x (y - 1))

-- The problem is to prove there are exactly 2 valid colorings of the hexagon grid
theorem hexagon_coloring_unique :
  ∃ (count : ℕ), count = 2 ∧
  ∀ coloring : (ℕ → ℕ → Color), is_valid_coloring coloring → count = 2 :=
by
  sorry

end hexagon_coloring_unique_l813_813227


namespace complex_circle_intersection_l813_813107

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l813_813107


namespace solve_for_y_l813_813206

theorem solve_for_y (y : ℝ) (h : (↑(30 * y) + (↑(30 * y) + 17) ^ (1 / 3)) ^ (1 / 3) = 17) :
  y = 816 / 5 := 
sorry

end solve_for_y_l813_813206


namespace valid_4_digit_integers_count_l813_813055

def valid_first_two_digits (d1 d2 : ℕ) : Prop :=
  d1 ∈ {1, 4, 5, 7} ∧ d2 ∈ {1, 4, 5, 7} ∧ (d1 + d2) % 2 = 0

def valid_last_two_digits (d3 d4 : ℕ) : Prop :=
  d3 ∈ {5, 7, 8, 9} ∧ d4 ∈ {5, 7, 8, 9} ∧ d3 ≠ d4

noncomputable def count_valid_4_digit_integers : ℕ :=
  let valid_pairs_first_digits := [(1,5), (1,7), (4,4), (4,8), (5,1), (5,5), (7,1), (7,7)]
  let count_valid_pairs_first_digits := valid_pairs_first_digits.length
  let count_valid_pairs_last_digits := 4 * 3
  count_valid_pairs_first_digits * count_valid_pairs_last_digits

theorem valid_4_digit_integers_count : count_valid_4_digit_integers = 96 := 
  sorry

end valid_4_digit_integers_count_l813_813055


namespace problem_solution_l813_813551

def average_pair (a b : ℝ) : ℝ := (a + b) / 2
def average_triple (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem problem_solution :
  average_triple (average_pair 2 4) (average_pair 6 2) (average_pair 3 3) = 10 / 3 :=
by
  sorry

end problem_solution_l813_813551


namespace days_with_equal_sun_tue_l813_813301

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l813_813301


namespace max_side_length_of_triangle_l813_813754

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813754


namespace count_divisible_digits_23n_l813_813385

noncomputable def count_divisible_digits : ℕ :=
  finset.card {n ∈ finset.range 10 | n ≠ 0 ∧ 23 * n % n = 0}

theorem count_divisible_digits_23n : count_divisible_digits = 3 := by
  sorry

end count_divisible_digits_23n_l813_813385


namespace proportional_b_value_l813_813458

theorem proportional_b_value (b : ℚ) : (∃ k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, x + 2 - 3 * b = k * x)) ↔ b = 2 / 3 :=
by
  sorry

end proportional_b_value_l813_813458


namespace intersecting_circles_unique_point_l813_813104

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l813_813104


namespace probability_gk_divisible_by_3_l813_813161

open Set

def g_set : Set ℕ := {3, 5, 7, 9, 11, 13, 8, 12}
def k_set : Set ℕ := {2, 4, 6, 10, 7, 21, 9}

def count_div_by_3 (s : Set ℕ) : ℕ :=
  s.count (λ x, x % 3 = 0)

theorem probability_gk_divisible_by_3 :
  let g_choices := g_set.toFinset.card
  let k_choices := k_set.toFinset.card
  let outcomes := g_choices * k_choices
  let favorable :=
    (count_div_by_3 g_set) * k_choices +
    (k_choices - count_div_by_3 k_set) * (count_div_by_3 k_set)
  (favorable / outcomes) = 9 / 14 :=
begin
  have g_div_3 := count_div_by_3 g_set,
  have k_div_3 := count_div_by_3 k_set,
  let g_choices := (g_set.toFinset.card : ℚ),
  let k_choices := (k_set.toFinset.card : ℚ),
  let outcomes := g_choices * k_choices,
  have favorable := g_div_3 * k_choices + (g_choices - g_div_3) * k_div_3,
  have prob := favorable / outcomes,
  rw [g_choices, k_choices, g_div_3, k_div_3, outcomes],
  norm_num [favorable],
  sorry
end

end probability_gk_divisible_by_3_l813_813161


namespace bees_count_l813_813179

theorem bees_count (x : ℕ) (h1 : (1/5 : ℚ) * x + (1/3 : ℚ) * x + 
    3 * ((1/3 : ℚ) * x - (1/5 : ℚ) * x) + 1 = x) : x = 15 := 
sorry

end bees_count_l813_813179


namespace students_taking_neither_math_nor_physics_l813_813178

theorem students_taking_neither_math_nor_physics (students_total students_math students_physics students_both: ℕ) 
  (h1 : students_total = 150) 
  (h2 : students_math = 85) 
  (h3 : students_physics = 60) 
  (h4 : students_both = 20): 
  students_total - (students_math - students_both + students_physics - students_both + students_both) = 25 :=
by
  rw [h1, h2, h3, h4]
  simp
  sorry

end students_taking_neither_math_nor_physics_l813_813178


namespace intersection_of_circles_l813_813089

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l813_813089


namespace max_side_length_l813_813777

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813777


namespace probability_alpha_in_interval_l813_813444

def vector_of_die_rolls_angle_probability : ℚ := 
  let total_outcomes := 36
  let favorable_pairs := 15
  favorable_pairs / total_outcomes

theorem probability_alpha_in_interval (m n : ℕ)
  (hm : 1 ≤ m ∧ m ≤ 6) (hn : 1 ≤ n ∧ n ≤ 6) :
  (vector_of_die_rolls_angle_probability = 5 / 12) := by
  sorry

end probability_alpha_in_interval_l813_813444


namespace quadratic_root_value_l813_813953

theorem quadratic_root_value {m : ℝ} (h : m^2 + m - 1 = 0) : 2 * m^2 + 2 * m + 2025 = 2027 :=
sorry

end quadratic_root_value_l813_813953


namespace find_t_l813_813882

-- Definitions of the given conditions
def a : ℂ := sorry 
def b : ℂ := sorry

-- The values of the moduli
def mod_a : ℝ := 3
def mod_b : ℝ := 5

axiom abs_a : complex.abs a = mod_a
axiom abs_b : complex.abs b = mod_b

-- Define the product ab and set it equal to t - 3i
def t : ℝ := sorry
axiom eq_ab : a * b = t - 3 * complex.i

-- The proof to be completed
theorem find_t : t = 6 * real.sqrt 6 :=
by
  -- Proof steps here (proof is omitted)
  sorry

end find_t_l813_813882


namespace dot_product_of_OA_and_OB_l813_813985

noncomputable def circle := {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 - 5)^2 = 9}

def line_through_fixed_point (m : ℝ) : set (ℝ × ℝ) := {p | p.2 - 4 = m * (p.1 - 4)}

theorem dot_product_of_OA_and_OB (m : ℝ) (A B : ℝ × ℝ) (hA : A ∈ circle) (hB : B ∈ circle) (h : A ≠ B) :
  A ∈ line_through_fixed_point m ∧ B ∈ line_through_fixed_point m →
    (\vec{O} A).x * (\vec{O} B).x + (\vec{O} A).y * (\vec{O} B).y ∈ {30, 24} :=
begin
  -- Proof omitted
  sorry
end

end dot_product_of_OA_and_OB_l813_813985


namespace perpendicular_vectors_m_value_l813_813052

variables (m : ℝ)

def a : ℝ × ℝ := (m, 4)
def b : ℝ × ℝ := (3, -2)

theorem perpendicular_vectors_m_value
  (h : a.1 * b.1 + a.2 * b.2 = 0) :
  m = 8 / 3 :=
by sorry

end perpendicular_vectors_m_value_l813_813052


namespace fleet_problem_l813_813625

theorem fleet_problem (initial_old : ℕ) (retired_per_year : ℕ) (new_per_year : ℕ) :
  initial_old = 20 → retired_per_year = 5 → new_per_year = 6 →
  ∃ n : ℕ, n = 3 ∧ (initial_old - (n * retired_per_year) < (initial_old - (n * retired_per_year) + n * new_per_year) / 2) :=
by
  intros h1 h2 h3
  use 3
  split
  { refl }
  { sorry }

end fleet_problem_l813_813625


namespace cos_B_eq_find_b_eq_l813_813484

variable (A B C a b c : ℝ)

-- Given conditions
axiom sin_A_plus_C_eq : Real.sin (A + C) = 8 * Real.sin (B / 2) ^ 2
axiom a_plus_c : a + c = 6
axiom area_of_triangle : 1 / 2 * a * c * Real.sin B = 2

-- Proving cos B
theorem cos_B_eq :
  Real.cos B = 15 / 17 :=
sorry

-- Proving b given the area and sides condition
theorem find_b_eq :
  Real.cos B = 15 / 17 → b = 2 :=
sorry

end cos_B_eq_find_b_eq_l813_813484


namespace exists_triangle_with_conditions_l813_813852

-- Define the properties and conditions
variables {α : Type*} [NormedSpace ℝ α] {A B C : α}

-- Given angle β, side b, and the sum a/2 + c
variables (β : ℝ) (b : ℝ) (a : ℝ) (c : ℝ)

-- Define the triangle condition
def is_triangle (A B C : α) : Prop :=
  dist A C = b ∧ ∠ B A C = β ∧ dist A B + (dist B C) / 2 = c + a / 2

-- The theorem statement
theorem exists_triangle_with_conditions : ∃ (A B C : α), is_triangle β b a c A B C :=
sorry

end exists_triangle_with_conditions_l813_813852


namespace max_side_length_of_triangle_l813_813761

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813761


namespace intersecting_circles_unique_point_l813_813101

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l813_813101


namespace range_of_a_l813_813460

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3) ↔ (-1 < a ∧ a < 2) :=
by 
  sorry

end range_of_a_l813_813460


namespace not_identity_element_l813_813147

def T := {x : ℝ // x ≠ 0}

def otimes (x y : T) : T := ⟨x.val / (y.val ^ 2), by { apply div_ne_zero; exact_mod_cast y.property, field_norm }⟩

theorem not_identity_element : ¬ (∀ x : T, otimes x ⟨1, one_ne_zero⟩ = x ∧ otimes ⟨1, one_ne_zero⟩ x = x) :=
by {
  intro h,
  have h1 := h ⟨1, one_ne_zero⟩,
  cases h1,
  simp [otimes] at *,
  sorry
}

end not_identity_element_l813_813147


namespace find_k_values_for_intersection_l813_813119

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l813_813119


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813696

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813696


namespace max_triangle_side_24_l813_813730

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813730


namespace locus_of_G_on_line_l813_813995

-- Define the problem in terms of given conditions
variables {A B C P K G F : Point}
variables (T : Triangle A B C) (PBC_side : P ∈ side BC) (K_incenter_PAB : is_incenter K (Triangle P A B))
variables (F_incircle_tangent_PAC : is_incircle_tangent F (Triangle P A C) BC)
variables (G_on_CK : G ∈ line_segment(C, K)) (FG_parallel_PK : is_parallel FG PK)

-- Goal: Prove the locus of G
theorem locus_of_G_on_line :
  locus G = line XY := 
sorry   -- This skips the proof as instructed

end locus_of_G_on_line_l813_813995


namespace sets_B_C_D_represent_same_function_l813_813831

theorem sets_B_C_D_represent_same_function :
  (∀ x : ℝ, (2 * x = 2 * (x ^ (3 : ℝ) ^ (1 / 3)))) ∧
  (∀ x t : ℝ, (x ^ 2 + x + 3 = t ^ 2 + t + 3)) ∧
  (∀ x : ℝ, (x ^ 2 = (x ^ 4) ^ (1 / 2))) :=
by
  sorry

end sets_B_C_D_represent_same_function_l813_813831


namespace percentage_of_material_A_in_second_solution_l813_813320

theorem percentage_of_material_A_in_second_solution 
  (material_A_first_solution : ℝ)
  (material_B_first_solution : ℝ)
  (material_B_second_solution : ℝ)
  (material_A_mixture : ℝ)
  (percentage_first_solution_in_mixture : ℝ)
  (percentage_second_solution_in_mixture : ℝ)
  (total_mixture: ℝ)
  (hyp1 : material_A_first_solution = 20 / 100)
  (hyp2 : material_B_first_solution = 80 / 100)
  (hyp3 : material_B_second_solution = 70 / 100)
  (hyp4 : material_A_mixture = 22 / 100)
  (hyp5 : percentage_first_solution_in_mixture = 80 / 100)
  (hyp6 : percentage_second_solution_in_mixture = 20 / 100)
  (hyp7 : percentage_first_solution_in_mixture + percentage_second_solution_in_mixture = total_mixture)
  : ∃ (x : ℝ), x = 30 := by
  sorry

end percentage_of_material_A_in_second_solution_l813_813320


namespace max_triangle_side_24_l813_813738

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813738


namespace books_left_after_sale_l813_813991

theorem books_left_after_sale (initial_books sold_books books_left : ℕ)
    (h1 : initial_books = 33)
    (h2 : sold_books = 26)
    (h3 : books_left = initial_books - sold_books) :
    books_left = 7 := by
  sorry

end books_left_after_sale_l813_813991


namespace min_value_frac_l813_813906

theorem min_value_frac (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  \(\frac{1}{x} + \frac{4}{y + 1} = \frac{9}{2}\) :=
sorry

end min_value_frac_l813_813906


namespace altitude_eq_circumcircle_eq_l813_813917

-- Definitions of the points of the triangle
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (2, -2)

-- Problem 1: Equation of the line containing the altitude drawn from point A to side BC
theorem altitude_eq : ∀ (x y : ℝ), (2 * x + y = 2) ↔ is_altitude A B C (x, y) := sorry

-- Problem 2: Equation of the circumcircle of triangle ABC
theorem circumcircle_eq : ∀ (x y : ℝ), ((x + 2)^2 + (y + 2)^2 = 16) ↔ on_circumcircle A B C (x, y) := sorry

end altitude_eq_circumcircle_eq_l813_813917


namespace max_side_length_l813_813704

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813704


namespace ratio_of_numbers_l813_813243

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 124) (h2 : y = 3 * x) : x / Nat.gcd x y = 1 ∧ y / Nat.gcd x y = 3 := by
  sorry

end ratio_of_numbers_l813_813243


namespace max_side_length_l813_813769

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813769


namespace frog_jump_distance_l813_813226

variable (grasshopper_jump frog_jump mouse_jump : ℕ)
variable (H1 : grasshopper_jump = 19)
variable (H2 : grasshopper_jump = frog_jump + 4)
variable (H3 : mouse_jump = frog_jump - 44)

theorem frog_jump_distance : frog_jump = 15 := by
  sorry

end frog_jump_distance_l813_813226


namespace line_passes_through_fixed_point_l813_813497

theorem line_passes_through_fixed_point (O M N : ℝ × ℝ)
  (hx : M.1^2 = 4 * M.2) (hy : N.1^2 = 4 * N.2)
  (dot_product_condition : M.1 * N.1 + M.2 * N.2 = -4) :
  ∃ k : ℝ, ∀ x, y, (y = k * x + 2) → (0, 2) = (0, y) := 
begin
  sorry
end

end line_passes_through_fixed_point_l813_813497


namespace coleen_sprinkles_l813_813343

theorem coleen_sprinkles : 
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  remaining_sprinkles = 3 :=
by
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  sorry

end coleen_sprinkles_l813_813343


namespace sum_prime_factors_2_10_minus_1_l813_813592

theorem sum_prime_factors_2_10_minus_1 : 
  let n := 10 
  let number := 2^n - 1 
  let factors := [3, 5, 7, 11] 
  number.prime_factors.sum = 26 :=
by
  sorry

end sum_prime_factors_2_10_minus_1_l813_813592


namespace cars_meet_time_l813_813254

theorem cars_meet_time (t : ℝ) (highway_length : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ)
  (h1 : highway_length = 105) (h2 : speed_car1 = 15) (h3 : speed_car2 = 20) :
  15 * t + 20 * t = 105 → t = 3 := by
  sorry

end cars_meet_time_l813_813254


namespace largest_n_for_positive_sum_l813_813029

noncomputable def a_n (n : ℕ) : ℝ

axiom arithmetic_sequence (d : ℝ) : ∀ n : ℕ, a_n (n+1) = a_n n + d

axiom first_term_positive : a_n 1 > 0

axiom condition1 : a_n 2003 + a_n 2004 > 0

axiom condition2 : a_n 2003 * a_n 2004 < 0

theorem largest_n_for_positive_sum : ∃ n : ℕ, n = 4006 ∧ 
    let S_n := λ n, n / 2 * (2 * a_n 1 + (n - 1) * d) in
     S_n n > 0 :=
sorry

end largest_n_for_positive_sum_l813_813029


namespace find_point_N_l813_813922

noncomputable def point (α : Type _) := α × α

def on_line (pt : point ℝ) (a b c : ℝ) : Prop := a * pt.1 + b * pt.2 + c = 0

def perpendicular_slope (k1 k2 : ℝ) : Prop := k1 * k2 = -1

def slope (p1 p2 : point ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_point_N :
  ∀ (M N : point ℝ),
  (M = (0, -1)) →
  on_line N 1 (-1) 1 →
  perpendicular_slope (slope M N) (-1/2) →
  N = (2, 3) :=
by
  intros M N M_coord N_on_line MN_perp
  sorry

end find_point_N_l813_813922


namespace scaled_multiplication_l813_813454

theorem scaled_multiplication (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 :=
by
  sorry

end scaled_multiplication_l813_813454


namespace inequality_solution_l813_813569

theorem inequality_solution (x y : ℝ) : y - x < abs x ↔ y < 0 ∨ y < 2 * x :=
by sorry

end inequality_solution_l813_813569


namespace circle_sequence_sum_l813_813318

def radius_relation (r1 r2 r3 : ℝ) : Prop :=
  r3 = (r1 * r2) / ((real.sqrt r1 + real.sqrt r2) ^ 2)

def sum_of_reciprocal_sqrt_radii (S : Finset ℝ) : ℝ :=
  S.sum (λ r, 1 / (real.sqrt r))

theorem circle_sequence_sum :
  let L0 := {80^2, 75^2} : Finset ℝ,
      S := L0 ∪ ... ∪ (L0.image (λ r1, L0.image (λ r2, (r1 * r2) / ((real.sqrt r1 + real.sqrt r2) ^ 2))))
  in sum_of_reciprocal_sqrt_radii S = (1 / 80) + (1 / 75) :=
sorry

end circle_sequence_sum_l813_813318


namespace number_of_4_digit_numbers_divisible_by_9_l813_813938

theorem number_of_4_digit_numbers_divisible_by_9 :
  ∃ n : ℕ, (∀ k : ℕ, k ∈ Finset.range n → 1008 + k * 9 ≤ 9999) ∧
           (1008 + (n - 1) * 9 = 9999) ∧
           n = 1000 :=
by
  sorry

end number_of_4_digit_numbers_divisible_by_9_l813_813938


namespace simplest_quadratic_radical_l813_813270

theorem simplest_quadratic_radical :
  ∀ (A B C D : ℝ), A = sqrt 9 → B = sqrt (1 / 2) → C = sqrt 0.1 → D = sqrt 3 → D = sqrt 3 :=
by
  intros A B C D hA hB hC hD
  exact hD
  sorry

end simplest_quadratic_radical_l813_813270


namespace elmer_saving_percent_l813_813869

theorem elmer_saving_percent (x c : ℝ) (hx : x > 0) (hc : c > 0) :
  let old_car_fuel_efficiency := x
  let new_car_fuel_efficiency := 1.6 * x
  let gasoline_cost := c
  let diesel_cost := 1.25 * c
  let trip_distance := 300
  let old_car_fuel_needed := trip_distance / old_car_fuel_efficiency
  let new_car_fuel_needed := trip_distance / new_car_fuel_efficiency
  let old_car_cost := old_car_fuel_needed * gasoline_cost
  let new_car_cost := new_car_fuel_needed * diesel_cost
  let cost_saving := old_car_cost - new_car_cost
  let percent_saving := (cost_saving / old_car_cost) * 100
  percent_saving = 21.875 :=
by
  sorry

end elmer_saving_percent_l813_813869


namespace bridge_length_l813_813633

theorem bridge_length (train_length : ℕ) (crossing_time : ℕ) (train_speed_kmh : ℕ) :
  train_length = 500 → crossing_time = 45 → train_speed_kmh = 64 → 
  ∃ (bridge_length : ℝ), bridge_length = 300.1 :=
by
  intros h1 h2 h3
  have speed_mps := (train_speed_kmh * 1000) / 3600
  have total_distance := speed_mps * crossing_time
  have bridge_length_calculated := total_distance - train_length
  use bridge_length_calculated
  sorry

end bridge_length_l813_813633


namespace amount_deducted_from_third_l813_813555

theorem amount_deducted_from_third
  (x : ℝ) 
  (h1 : ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 16)) 
  (h2 : (( (x - 9) + ((x + 1) - 8) + ((x + 2) - d) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) ) / 10 = 11.5)) :
  d = 13.5 :=
by
  sorry

end amount_deducted_from_third_l813_813555


namespace calculate_values_l813_813383

/-- Define the special operation --/
def op (m x y : ℝ) : ℝ := (4 * x * y) / (m * x + 3 * y)

/-- Given that op 2 1 2 equals 1, calculate the value of op 2 3 12 --/
theorem calculate_values : 
  op 2 1 2 = 1 ∧ op 2 3 12 = 24 / 7 :=
by
  sorry

end calculate_values_l813_813383


namespace convex_quad_diagonal_triangles_not_twice_angle_l813_813486

theorem convex_quad_diagonal_triangles_not_twice_angle :
  ∀ (ABCD : ConvexQuadrilateral) (diag : Diagonals ABCD),
  (similar (diag.triangle₁) (diag.triangle₂) ∧ 
   similar (diag.triangle₂) (diag.triangle₃) ∧ 
   similar (diag.triangle₃) (diag.triangle₁)) ∧ 
  ¬ similar (diag.triangle₄) (diag.triangle₁) →
  ∀ (θ₄ α : ℝ), acute θ₄ ∧ θ₄ = 2 * α →
  ¬∃ (α₁ α₂ α₃ : ℝ), 
    (angle_in_triangle diag.triangle₁ α₁ ∧
     angle_in_triangle diag.triangle₂ α₂ ∧
     angle_in_triangle diag.triangle₃ α₃) ∧
    (θ₄ = 2 * α₁ ∨ θ₄ = 2 * α₂ ∨ θ₄ = 2 * α₃) := sorry

end convex_quad_diagonal_triangles_not_twice_angle_l813_813486


namespace max_side_of_triangle_l813_813806

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813806


namespace sandwiches_first_cousin_eaten_l813_813200

theorem sandwiches_first_cousin_eaten :
  ∀ (R T : ℕ), (R = 1) → (T = 10) → 
  ∃ C1 C2 B left total_eaten : ℕ, 
  (C1 = 1) ∧ (C2 = 2) ∧ (B = 2) ∧ (left = 3) ∧ 
  (C1 + C2 + B = 5) ∧ (total_eaten = T - left) ∧ 
  (total_eaten - (C1 + C2 + B) = 2) :=
by
  intros R T hR hT
  use 1, 2, 2, 3, 7
  split; [refl, split; [refl, split; [refl, split; [refl, split; [linarith, split; [linarith, linarith]]]]]]


end sandwiches_first_cousin_eaten_l813_813200


namespace find_building_height_l813_813355

theorem find_building_height 
  (crane1_height : ℕ := 228) (building1_height : ℕ := 200)
  (crane2_height : ℕ := 120) (building2_height : ℕ)
  (crane3_height : ℕ := 147) (building3_height : ℕ := 140)
  (avg_percentage : ℝ := 0.13) : building2_height ≈ 98.05 :=
by
  have h_avg : (crane1_height + crane2_height + crane3_height : ℝ) / 
               (building1_height + building2_height + building3_height : ℝ) = 1 + avg_percentage,
    sorry,
  sorry

end find_building_height_l813_813355


namespace solution_interval_l813_813654

noncomputable def f (x : ℝ) : ℝ := x * real.exp x

def F (x : ℝ) : ℝ := f x - (x + 1) * real.exp x - x

theorem solution_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) (-1/2), F x = 0 :=
sorry

end solution_interval_l813_813654


namespace star_polygon_points_eq_24_l813_813977

theorem star_polygon_points_eq_24 (n : ℕ) 
  (A_i B_i : ℕ → ℝ) 
  (h_congruent_A : ∀ i j, A_i i = A_i j) 
  (h_congruent_B : ∀ i j, B_i i = B_i j) 
  (h_angle_difference : ∀ i, A_i i = B_i i - 15) : 
  n = 24 := 
sorry

end star_polygon_points_eq_24_l813_813977


namespace find_length_QJ_l813_813126

variables {P Q R J G H I : Type}
variables [LinearOrderedField P]

noncomputable def PQ : P := 15
noncomputable def PR : P := 17
noncomputable def QR : P := 16
noncomputable def incenter (T: Triangle P Q R) : Point P := J
noncomputable def incircle_touches (T: Triangle P Q R) (s1 s2 s3: Side P) :
  (Point P) × (Point P) × (Point P) := (G, H, I)
noncomputable def length (a b : Point P) : P := sorry

theorem find_length_QJ (PQ_len: length P Q = 15)
                       (PR_len: length P R = 17)
                       (QR_len: length Q R = 16)
                       (J_incenter: incenter (Triangle.mk P Q R) = J)
                       (QR_touch : incircle_touches (Triangle.mk P Q R) (Side.mk Q R) = (G, H, I))
                       (PR_touch : incircle_touches (Triangle.mk P R Q) (Side.mk P R) = (H, J, I))
                       (PQ_touch : incircle_touches (Triangle.mk Q R P) (Side.mk P Q) = (I, G, H)):
  length Q J = 2 * Real.sqrt 23 :=
sorry

end find_length_QJ_l813_813126


namespace Davey_guitars_proof_l813_813838

variable (S : ℝ)

-- Define the number of guitars each person has given S.
def BarbeckGuitars := 2 * S
def DaveyGuitars := 3 * BarbeckGuitars
def JaneGuitars := (BarbeckGuitars + DaveyGuitars) / 2 - 1

-- State the total number of guitars.
def TotalGuitars := S + BarbeckGuitars + DaveyGuitars + JaneGuitars

-- Prove that Davey has 18 guitars given that TotalGuitars = 46.
theorem Davey_guitars_proof (h : TotalGuitars = 46) : DaveyGuitars = 18 :=
by
  sorry

end Davey_guitars_proof_l813_813838


namespace max_side_of_triangle_l813_813753

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813753


namespace problem_I_problem_II_l813_813162

open Set

-- Definitions of the sets A and B, and their intersections would be needed
def A := {x : ℝ | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ 3 * a}

-- (I) When a = 1, find A ∩ B
theorem problem_I : A ∩ (B 1) = {x : ℝ | (2 ≤ x ∧ x ≤ 3) ∨ x = 1} := by
  sorry

-- (II) When A ∩ B = B, find the range of a
theorem problem_II : {a : ℝ | a > 0 ∧ ∀ x, x ∈ B a → x ∈ A} = {a : ℝ | (0 < a ∧ a ≤ 1 / 3) ∨ a ≥ 2} := by
  sorry

end problem_I_problem_II_l813_813162


namespace value_after_increase_l813_813656

def original_number : ℝ := 400
def percentage_increase : ℝ := 0.20

theorem value_after_increase : original_number * (1 + percentage_increase) = 480 := by
  sorry

end value_after_increase_l813_813656


namespace largest_invertible_interval_l813_813879

noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 4

theorem largest_invertible_interval : 
  ∃ (a : ℝ), a ≤ 2 ∧ (∀ x1 x2 ∈ set.Ici a, g x1 = g x2 → x1 = x2) ∧ ([a, ∞) = [3/2, ∞)) := 
sorry

end largest_invertible_interval_l813_813879


namespace max_triangle_side_24_l813_813739

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813739


namespace complex_circle_intersection_l813_813111

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l813_813111


namespace units_digit_of_7_pow_6_pow_5_l813_813863

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l813_813863


namespace intersection_empty_l813_813928

def A : Set ℝ := {x | x > -1 ∧ x ≤ 3}
def B : Set ℝ := {2, 4}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l813_813928


namespace find_f_l813_813024

theorem find_f (f : ℝ → ℝ) :
  (∀ x : ℝ, f(sin x - 1) ≥ (cos x) ^ 2 + 2) →
  (∀ z : ℝ, -2 ≤ z ∧ z ≤ 0 → f(z) ≥ -z^2 - 2*z + 2) :=
by
  intro h
  intros z hz
  sorry

end find_f_l813_813024


namespace max_triangle_side_l813_813724

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813724


namespace max_side_length_of_triangle_l813_813762

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813762


namespace equal_sundays_tuesdays_l813_813308

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l813_813308


namespace height_difference_of_packings_l813_813608

theorem height_difference_of_packings :
  (let d := 12
   let n := 180
   let rowsA := n / 10
   let heightA := rowsA * d
   let height_of_hex_gap := (6 * Real.sqrt 3 : ℝ)
   let gaps := rowsA - 1
   let heightB := gaps * height_of_hex_gap + 2 * (d / 2)
   heightA - heightB) = 204 - 102 * Real.sqrt 3 :=
  sorry

end height_difference_of_packings_l813_813608


namespace area_of_region_is_four_l813_813371

noncomputable theory
open Topology Classical

-- Conditions of the problem defined as functions and sets
def first_condition (x : ℝ) : Prop := sqrt (1 - x) + 2 * x ≥ 0

def second_condition (x y : ℝ) : Prop := -1 - x^2 ≤ y ∧ y ≤ 2 + sqrt x

-- Define the region of interest
def region (p : ℝ × ℝ) : Prop := 
  let (x, y) := p in 
  first_condition x ∧ second_condition x y

-- Define the bounds of integration
def x_bounds (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Upper and lower bounds for y
def y_upper (x : ℝ) : ℝ := 2 + sqrt x
def y_lower (x : ℝ) : ℝ := -1 - x^2

-- Define the target area as a definite integral
def target_area : ℝ :=
  ∫ x in (0 : ℝ)..1, (y_upper x - y_lower x)

-- The statement of the problem
theorem area_of_region_is_four : target_area = 4 := 
  sorry

end area_of_region_is_four_l813_813371


namespace distance_between_nest_and_ditch_is_250_meters_l813_813644

-- Define conditions
def trips : ℕ := 15
def time_hours : ℝ := 1.5
def speed_km_per_hour : ℝ := 5

-- Define the main problem statement to prove
theorem distance_between_nest_and_ditch_is_250_meters :
  let total_distance := speed_km_per_hour * time_hours in
  let distance_one_round_trip := total_distance / trips in
  let distance_nest_to_ditch_km := distance_one_round_trip / 2 in
  let distance_nest_to_ditch_m := distance_nest_to_ditch_km * 1000 in
  distance_nest_to_ditch_m = 250 :=
by
  sorry

end distance_between_nest_and_ditch_is_250_meters_l813_813644


namespace two_unique_sequences_l813_813613

theorem two_unique_sequences (s : ℕ → ℕ) (h1 : ∀ n, s n ∈ {k | 1 ≤ k ∧ k ≤ 98}) 
    (h2 : ∀ n, n < 97 → |s (n + 1) - s n| > 48) : 
  ∃! t : ℕ → ℕ, (∀ n, t n ∈ {k | 1 ≤ k ∧ k ≤ 98}) ∧ (∀ n, n < 97 → |t (n + 1) - t n| > 48) ∧ 
  (s = t ∨ s = (λ n, t (97 - n))) :=
begin
  sorry
end

end two_unique_sequences_l813_813613


namespace arithmetic_sequence_sum_l813_813083

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℕ), a 1 = 2 ∧ a 2 + a 3 = 13 → a 4 + a 5 + a 6 = 42 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_sum_l813_813083


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813819

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813819


namespace total_wasted_time_is_10_l813_813166

-- Define the time Martin spends waiting in traffic
def waiting_time : ℕ := 2

-- Define the constant for the multiplier
def multiplier : ℕ := 4

-- Define the time spent trying to get off the freeway
def off_freeway_time : ℕ := waiting_time * multiplier

-- Define the total wasted time
def total_wasted_time : ℕ := waiting_time + off_freeway_time

-- Theorem stating that the total time wasted is 10 hours
theorem total_wasted_time_is_10 : total_wasted_time = 10 :=
by
  sorry

end total_wasted_time_is_10_l813_813166


namespace variance_transformation_l813_813069

open BigOperators

variable {n : ℕ}
variables {x : Fin n → ℝ}

noncomputable def variance (s : Fin n → ℝ) : ℝ :=
  (∑ i, (s i - (∑ j, s j) / n)^2) / n

theorem variance_transformation (h : variance x = 4) : 
  variance (λ i, 2 * x i + 1) = 16 := 
by
  sorry

end variance_transformation_l813_813069


namespace max_side_of_triangle_exists_max_side_of_elevent_l813_813815

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l813_813815


namespace intersecting_circles_l813_813092

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l813_813092


namespace problem_statement_l813_813423

noncomputable def minimum_distance (θ : ℝ) : ℝ :=
  (real.sqrt 5 / 5) * abs (4 * real.cos θ - 3 * real.sin θ - 13)

theorem problem_statement
  (M : ℝ × ℝ) (P : ℝ × ℝ := (-4, 4))
  (Q : ℝ → ℝ × ℝ := λ θ, (8 * real.cos θ, 3 * real.sin θ))
  (M := λ θ, ((-2 + 4 * real.cos θ), (2 + 3 / 2 * real.sin θ)))
  (d : ℝ → ℝ := λ θ, minimum_distance θ) :
  ∃ θ₀ : ℝ, real.cos θ₀ = 4 / 5 ∧ real.sin θ₀ = -3 / 5 ∧ d θ₀ = 8 * real.sqrt 5 / 5 :=
sorry

end problem_statement_l813_813423


namespace total_money_l813_813652

theorem total_money (total_notes : ℕ) (notes_50 : ℕ) (total_notes_eq : total_notes = 126) (notes_50_eq : notes_50 = 117) :
  let notes_500 := total_notes - notes_50 in
  let amount_50 := notes_50 * 50 in
  let amount_500 := notes_500 * 500 in
  (amount_50 + amount_500) = 10350 :=
by
  intros
  unfold notes_500 amount_50 amount_500
  rw [total_notes_eq, notes_50_eq]
  norm_num
  sorry

end total_money_l813_813652


namespace solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l813_813545

-- Equivalent proof problem for equation (1)
theorem solve_equation_1 (x : ℝ) : (x + 1)^2 - 5 = 0 ↔ x = sqrt 5 - 1 ∨ x = -sqrt 5 - 1 :=
by sorry

-- Equivalent proof problem for equation (2)
theorem solve_equation_2 (x : ℝ) : 2 * x^2 - 4 * x + 1 = 0 ↔ 
  x = (4 + sqrt 8) / 4 ∨ x = (4 - sqrt 8) / 4 :=
by sorry

-- Equivalent proof problem for equation (3)
theorem solve_equation_3 (x : ℝ) : (2 * x + 1) * (x - 3) = -7 ↔ False :=
by sorry

-- Equivalent proof problem for equation (4)
theorem solve_equation_4 (x : ℝ) : 3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 :=
by sorry

end solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l813_813545


namespace max_side_of_triangle_l813_813746

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813746


namespace sum_of_roots_P_l813_813890

noncomputable def P (x : ℂ) : ℂ :=
  (x - 1)^2007 + 2 * (x - 2)^2006 + 3 * (x - 3)^2005 + ∑ (k : ℕ) in finset.range 2004, (k + 3) * (x - (k + 3))^(2008 - (k + 4))

-- The theorem statement for the sum of the roots of the polynomial P
theorem sum_of_roots_P : ∑ (γ : ℂ) in (P.roots), γ = 2005 :=
by
  sorry

end sum_of_roots_P_l813_813890


namespace intersection_of_circles_l813_813087

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l813_813087


namespace carrie_total_sales_l813_813843

theorem carrie_total_sales :
  let tomatoes := 200
  let carrots := 350
  let price_tomato := 1.0
  let price_carrot := 1.50
  (tomatoes * price_tomato + carrots * price_carrot) = 725 := by
  -- let tomatoes := 200
  -- let carrots := 350
  -- let price_tomato := 1.0
  -- let price_carrot := 1.50
  -- show (tomatoes * price_tomato + carrots * price_carrot) = 725
  sorry

end carrie_total_sales_l813_813843


namespace find_missing_figure_l813_813275

theorem find_missing_figure :
  ∃ (x : ℝ), 0.0075 * x = 0.06 ∧ x = 8 :=
begin
  use 8,
  split,
  { norm_num, },
  { refl, }
end

end find_missing_figure_l813_813275


namespace max_triangle_side_24_l813_813732

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813732


namespace martin_total_waste_is_10_l813_813169

def martinWastesTrafficTime : Nat := 2
def martinWastesFreewayTime : Nat := 4 * martinWastesTrafficTime
def totalTimeWasted : Nat := martinWastesTrafficTime + martinWastesFreewayTime

theorem martin_total_waste_is_10 : totalTimeWasted = 10 := 
by 
  sorry

end martin_total_waste_is_10_l813_813169


namespace bounded_f_range_l813_813649

noncomputable def f (a : ℝ) (x : ℝ) := 1 + a*(1/2)^x + (1/4)^x

theorem bounded_f_range (a : ℝ) :
  (∀ x ∈ Icc (-2 : ℝ) (1 : ℝ), abs (f a x) ≤ 3) →
  -4 ≤ a ∧ a ≤ -7/2 :=
begin
  sorry
end

end bounded_f_range_l813_813649


namespace smallest_k_divides_l813_813887

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (z : ℂ) : ∀ k : ℕ, (f z ∣ z^42 - 1) ∧ (∀ k' : ℕ, k' < 42 → ¬ (f z ∣ z^k' - 1)) :=
by
  sorry

end smallest_k_divides_l813_813887


namespace Beth_finishes_first_l813_813836

noncomputable def AndyMowingTime (x y : ℝ) : ℝ :=
  x / y

noncomputable def BethMowingTime (x y : ℝ) : ℝ :=
  x / (3 * y)

noncomputable def CarlosMowingTime (x y : ℝ) : ℝ :=
  x / y

theorem Beth_finishes_first (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  BethMowingTime x y < AndyMowingTime x y ∧ BethMowingTime x y < CarlosMowingTime x y :=
by
  unfold AndyMowingTime BethMowingTime CarlosMowingTime
  apply and.intro
  {
    rw lt_div_iff' h2
    rw div_eq_mul_inv
    simp
    rw lt_mul_iff_one_lt_left h1
    norm_num
  },
  {
    rw lt_div_iff' h2
    rw div_eq_mul_inv
    simp
    rw lt_mul_iff_one_lt_left h1
    norm_num
  }
  sorry

end Beth_finishes_first_l813_813836


namespace nate_search_time_l813_813174

theorem nate_search_time
  (rowsG : Nat) (cars_per_rowG : Nat)
  (rowsH : Nat) (cars_per_rowH : Nat)
  (rowsI : Nat) (cars_per_rowI : Nat)
  (walk_speed : Nat) : Nat :=
  let total_cars : Nat := rowsG * cars_per_rowG + rowsH * cars_per_rowH + rowsI * cars_per_rowI
  let total_minutes : Nat := total_cars / walk_speed
  if total_cars % walk_speed == 0 then total_minutes else total_minutes + 1

/-- Given:
- rows in Section G = 18, cars per row in Section G = 12
- rows in Section H = 25, cars per row in Section H = 10
- rows in Section I = 17, cars per row in Section I = 11
- Nate's walking speed is 8 cars per minute
Prove: Nate took 82 minutes to search the parking lot
-/
example : nate_search_time 18 12 25 10 17 11 8 = 82 := by
  sorry

end nate_search_time_l813_813174


namespace sum_first_2018_terms_l813_813438

noncomputable def a : ℕ → ℝ
| 0       := 3
| (n + 1) := let an := a n in (2 * an + 1) / (an - 1)

theorem sum_first_2018_terms : (∑ i in Finset.range 2018, a i) = 589 := by
  sorry

end sum_first_2018_terms_l813_813438


namespace vector_combination_solution_l813_813048

theorem vector_combination_solution :
  let a := (3 : ℝ, 2 : ℝ)
  let b := (-1 : ℝ, 2 : ℝ)
  let c := (4 : ℝ, 1 : ℝ)
  ∃ m n : ℝ, a = (m * b.1 + n * c.1, m * b.2 + n * c.2) ∧ m = 5 / 9 ∧ n = 8 / 9 :=
by {
  let a := (3 : ℝ, 2 : ℝ)
  let b := (-1 : ℝ, 2 : ℝ)
  let c := (4 : ℝ, 1 : ℝ)
  exact ⟨5 / 9, 8 / 9, by simp [a, b, c]⟩
}

end vector_combination_solution_l813_813048


namespace max_side_length_l813_813706

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813706


namespace gain_percent_calculation_l813_813945

theorem gain_percent_calculation (gain_paise : ℕ) (cost_price_rupees : ℕ) (rupees_to_paise : ℕ)
  (h_gain_paise : gain_paise = 70)
  (h_cost_price_rupees : cost_price_rupees = 70)
  (h_rupees_to_paise : rupees_to_paise = 100) :
  ((gain_paise / rupees_to_paise) / cost_price_rupees) * 100 = 1 :=
by
  -- Placeholder to indicate the need for proof
  sorry

end gain_percent_calculation_l813_813945


namespace explicit_formula_inequality_solution_l813_813920

noncomputable def f (x : ℝ) : ℝ := (x : ℝ) / (x^2 + 1)

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x → y < b → x < y → f x < f y
def f_half_eq_two_fifths : Prop := f (1/2) = 2/5

-- Questions rewritten as goals
theorem explicit_formula :
  odd_function f ∧ increasing_on_interval f (-1) 1 ∧ f_half_eq_two_fifths →
  ∀ x, f x = x / (x^2 + 1) := by 
sorry

theorem inequality_solution :
  odd_function f ∧ increasing_on_interval f (-1) 1 →
  ∀ t, (f (t - 1) + f t < 0) ↔ (0 < t ∧ t < 1/2) := by 
sorry

end explicit_formula_inequality_solution_l813_813920


namespace number_of_zeros_in_T_l813_813238

theorem number_of_zeros_in_T:
  let S_k (k : ℕ) := 2 * (10^k - 1) / 9 in
  let S_2 := 2 * (10^2 - 1) / 9 in
  let S_20 := 2 * (10^20 - 1) / 9 in
  let T := S_20 / S_2 in
  ∃ (n : ℕ), T = 1 + 10^2 + 10^4 + ... + 10^18 ∧ n = 18 :=
sorry

end number_of_zeros_in_T_l813_813238


namespace complex_circle_intersection_l813_813106

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l813_813106


namespace max_side_length_l813_813709

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l813_813709


namespace train_crossing_time_l813_813667

def train_length : ℕ := 320
def train_speed_kmh : ℕ := 72
def kmh_to_ms (v : ℕ) : ℕ := v * 1000 / 3600
def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh
def crossing_time (length : ℕ) (speed : ℕ) : ℕ := length / speed

theorem train_crossing_time : crossing_time train_length train_speed_ms = 16 := 
by {
  sorry
}

end train_crossing_time_l813_813667


namespace distribution_of_X_when_n_is_3_expectation_and_variance_of_Y_l813_813610

theorem distribution_of_X_when_n_is_3 :
  let X := (
    if A_guesses_incorrectly_in_all_3_rounds then -9
    else if A_guesses_correctly_in_1_round then -3
    else if A_guesses_correctly_in_2_rounds then 3
    else if A_guesses_correctly_in_all_3_rounds then 9
    else 0) in
  ∃ P : X -> ℚ, 
    P X = (if X = -9 then 1/8 
           else if X = -3 then 3/8 
           else if X = 3 then 3/8 
           else if X = 9 then 1/8
           else 0) := sorry

theorem expectation_and_variance_of_Y (n : ℕ) (hn : 0 < n) :
  let k := binomial_distribution n (1/2) in
  let Y := 6 * k - 3 * n in
  E Y = 0 ∧ D Y = 9 * n := sorry

end distribution_of_X_when_n_is_3_expectation_and_variance_of_Y_l813_813610


namespace sum_of_a_b_l813_813950

theorem sum_of_a_b (a b : ℝ) (h1 : ∀ x : ℝ, (a * (b * x + a) + b = x))
  (h2 : ∀ y : ℝ, (b * (a * y + b) + a = y)) : a + b = -2 := 
sorry

end sum_of_a_b_l813_813950


namespace minimum_value_of_z_l813_813618

noncomputable def quadratic_expression : ℝ × ℝ → ℝ :=
  λ p, 3 * p.1 ^ 2 + 5 * p.2 ^ 2 + 12 * p.1 - 10 * p.2 + 40

theorem minimum_value_of_z :
  ∃ (x y : ℝ), quadratic_expression (x, y) = 23 :=
by
  use (-2, 1)
  have : quadratic_expression (-2, 1) = 23 := sorry
  exact this

end minimum_value_of_z_l813_813618


namespace max_triangles_in_K4_free_graph_l813_813141

open SimpleGraph

variables {G : SimpleGraph (V : Type)} [DecidableRel G.Adj]

def is_triangle (G : SimpleGraph V) (u v w : V) : Prop :=
  G.Adj u v ∧ G.Adj v w ∧ G.Adj u w

def triangle_free (G : SimpleGraph V) : Prop :=
  ∀ (u v w : V), ¬(is_triangle G u v w)

theorem max_triangles_in_K4_free_graph (G : SimpleGraph V) [Fintype V] (k : ℕ) :
  ¬ (∃ (s : Finset V), G.inducedSubgraph s ∼= (CompleteGraph 4)) →
  Fintype.card V = 3 * k →
  ∃ (count : ℕ), (∀ (triangle_count : ℕ), triangle_count ≤ count) ∧ count = k^3 :=
sorry

end max_triangles_in_K4_free_graph_l813_813141


namespace exists_points_with_small_distance_l813_813282

theorem exists_points_with_small_distance :
  ∃ A B : ℝ × ℝ, (A.2 = A.1^4) ∧ (B.2 = B.1^4 + B.1^2 + B.1 + 1) ∧ 
  (dist A B < 1 / 100) :=
by
  sorry

end exists_points_with_small_distance_l813_813282


namespace sum_of_x_and_y_l813_813958

theorem sum_of_x_and_y (x y : ℝ) (h1 : x + abs x + y = 5) (h2 : x + abs y - y = 6) : x + y = 9 / 5 :=
by
  sorry

end sum_of_x_and_y_l813_813958


namespace distance_between_foci_l813_813225

theorem distance_between_foci :
  let F1 := (2 : ℝ, -3 : ℝ)
  let F2 := (-6 : ℝ, 9 : ℝ)
  sqrt ( (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2 ) = 4 * sqrt 13 := by
    sorry

end distance_between_foci_l813_813225


namespace trapezoid_shorter_base_l813_813975

theorem trapezoid_shorter_base (y : ℝ) (h1 : (117 - y) / 2 = 5) : y = 107 :=
begin
  sorry
end

end trapezoid_shorter_base_l813_813975


namespace john_spends_40_dollars_l813_813133

-- Definitions based on conditions
def cost_per_loot_box : ℝ := 5
def average_value_per_loot_box : ℝ := 3.5
def average_loss : ℝ := 12

-- Prove the amount spent on loot boxes is $40
theorem john_spends_40_dollars :
  ∃ S : ℝ, (S * (cost_per_loot_box - average_value_per_loot_box) / cost_per_loot_box = average_loss) ∧ S = 40 :=
by
  sorry

end john_spends_40_dollars_l813_813133


namespace real_roots_count_l813_813851

theorem real_roots_count (s : Finset ℤ) (h : ∀ x, x ∈ s → 1 ≤ x ∧ x ≤ 10) :
  (∑ b in s, (s.filter (λ c, b^2 - 4 * c ≥ 0)).card) = 60 := by
  sorry

end real_roots_count_l813_813851


namespace find_triples_l813_813368

theorem find_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 = c^2) ∧ (a^3 + b^3 + 1 = (c-1)^3) ↔ (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10) :=
by
  sorry

end find_triples_l813_813368


namespace sum_of_possible_x_values_l813_813835

theorem sum_of_possible_x_values :
  (sum (filter (λ x, 
    (10 + 17 > x) ∧ (100 + 289 < x ^ 2)) (Ico 20 27)) + 
  sum (filter (λ x, 
    (10 + x > 17) ∧ (100 + x ^ 2 < 289) ∧ (x < 14)) (Ico 8 14))) = 224 := 
by
  sorry

end sum_of_possible_x_values_l813_813835


namespace smallest_part_proportional_division_l813_813956

theorem smallest_part_proportional_division (a b c d total : ℕ) (h : a + b + c + d = total) (sum_equals_360 : 360 = total * 15):
  min (4 * 15) (min (5 * 15) (min (7 * 15) (8 * 15))) = 60 :=
by
  -- Defining the proportions and overall total
  let a := 5
  let b := 7
  let c := 4
  let d := 8
  let total_parts := a + b + c + d

  -- Given that the division is proportional
  let part_value := 360 / total_parts

  -- Assert that the smallest part is equal to the smallest proportion times the value of one part
  let smallest_part := c * part_value
  trivial

end smallest_part_proportional_division_l813_813956


namespace carrie_total_sales_l813_813844

theorem carrie_total_sales :
  let tomatoes := 200
  let carrots := 350
  let price_tomato := 1.0
  let price_carrot := 1.50
  (tomatoes * price_tomato + carrots * price_carrot) = 725 := by
  -- let tomatoes := 200
  -- let carrots := 350
  -- let price_tomato := 1.0
  -- let price_carrot := 1.50
  -- show (tomatoes * price_tomato + carrots * price_carrot) = 725
  sorry

end carrie_total_sales_l813_813844


namespace number_of_correct_propositions_l813_813501

-- Definitions corresponding to the conditions
variables (α β γ : Plane) (l : Line)

-- Conditions embedded as definitions
def prop1 := (α ⊥ γ) ∧ (β ⊥ γ) → (α ⊥ β)
def prop2 := (α ⊥ γ) ∧ (β ∥ γ) → (α ⊥ β)
def prop3 := (l ∥ α) ∧ (l ⊥ β) → (α ⊥ β)

-- Statement that proves exactly 2 out of 3 propositions are correct
theorem number_of_correct_propositions : (∀ α β γ l, (prop1 ↔ false) ∨ (prop2 ↔ true) ∨ (prop3 ↔ true) → (prop2 ∧ prop3)) :=
sorry

end number_of_correct_propositions_l813_813501


namespace coin_die_probability_l813_813550

theorem coin_die_probability :
  let outcomes_coin := 2,
      outcomes_die := 8,
      total_outcomes := outcomes_coin * outcomes_die,
      successful_outcome := 1 in
  (successful_outcome / total_outcomes) = (1 / 16) :=
by
  sorry

end coin_die_probability_l813_813550


namespace weight_second_square_l813_813665

noncomputable def side_length_first_square : ℝ := 4
noncomputable def weight_first_square : ℝ := 16
noncomputable def side_length_second_square : ℝ := 6

theorem weight_second_square : 
  let area_first := side_length_first_square^2
      area_second := side_length_second_square^2
      weight_second := weight_first_square * area_second / area_first
  in weight_second = 36 := by
  let area_first := side_length_first_square^2
  let area_second := side_length_second_square^2
  let weight_second := weight_first_square * area_second / area_first
  show weight_second = 36 from sorry

end weight_second_square_l813_813665


namespace distinct_books_distribution_l813_813865

theorem distinct_books_distribution :
  ∃ (x : Finset (Fin 17)), 
    (∀ a ∈ x, a ≠ 0) ∧ x.card = 4 ∧ x.sum (λ b, (b : ℕ)) = 16 ∧ 
    (x.pairwise (≠)) ∧ 
    (x.to_multiset.distinct_permutations.card = 216) := 
sorry

end distinct_books_distribution_l813_813865


namespace simplify_and_evaluate_expression_l813_813203

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate_expression :
  ((x + 1) / (x^2 + 2 * x + 1)) / (1 - (2 / (x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l813_813203


namespace intersection_sets_l813_813441

theorem intersection_sets (A B : Set ℝ) (hA : A = { x | x^2 - 16 < 0 }) (hB : B = { x | x^2 - 4x + 3 > 0 }) :
  A ∩ B = { x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4 } :=
by
  sorry

end intersection_sets_l813_813441


namespace slower_train_pass_time_l813_813258

noncomputable def relative_speed_km_per_hr (v1 v2 : ℕ) : ℕ :=
v1 + v2

noncomputable def relative_speed_m_per_s (v_km_per_hr : ℕ) : ℝ :=
(v_km_per_hr * 5) / 18

noncomputable def time_to_pass (distance_m : ℕ) (speed_m_per_s : ℝ) : ℝ :=
distance_m / speed_m_per_s

theorem slower_train_pass_time
  (length_train1 length_train2 : ℕ)
  (speed_train1_km_per_hr speed_train2_km_per_hr : ℕ)
  (distance_to_cover : ℕ)
  (h1 : length_train1 = 800)
  (h2 : length_train2 = 600)
  (h3 : speed_train1_km_per_hr = 85)
  (h4 : speed_train2_km_per_hr = 65)
  (h5 : distance_to_cover = length_train2) :
  time_to_pass distance_to_cover (relative_speed_m_per_s (relative_speed_km_per_hr speed_train1_km_per_hr speed_train2_km_per_hr)) = 14.4 := 
sorry

end slower_train_pass_time_l813_813258


namespace length_seg_PT_expr_l813_813548

namespace math_proof

def square_side_len := 2
def PT_SU_condition (PT SU : ℝ) : Prop := PT = SU
def folded_condition (PQ T U : ℝ → Prop) (PT SU : ℝ) : Prop := 
  ∃ (RT RU : ℝ), RT = sqrt 2 * (2 - PT) ∧ RU = sqrt 2 * (2 - SU) ∧ 
  (PR = 2 ∧ SR = 2 ∧ diagonal_len = sqrt 8 ∧ 
  (2 * sqrt 2 = 2 + sqrt 2 * (2 - PT)))

theorem length_seg_PT_expr :
  ∃ k m : ℕ, (PT : ℝ) = sqrt (k) - m ∧ k + m = 3 := 
by
  let PT := 1 in
  let k := 2 in
  let m := 1 in
  use [k, m]
  split
  · simp [PT]
  · simp [k, m]
  
end math_proof

end length_seg_PT_expr_l813_813548


namespace min_good_triangles_l813_813079

-- Definition of the problem conditions
def is_good_triangle {α : Type} [linear_ordered_field α] (t : finset (α × α)) (rect_area : α) : Prop :=
  t.card = 3 ∧ t.area ≤ rect_area / 4

def rectangle := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Statement the problem
theorem min_good_triangles (points : finset (ℝ × ℝ)) (h_points : points ⊆ rectangle)
  (h_card : points.card = 5) (h_nocollinear : ∀ t, t ⊆ points → t.card = 3 → ¬ collinear ℝ t) :
  ∃ (good_triangles : finset (finset (ℝ × ℝ))), good_triangles.card ≥ 2 ∧ 
  ∀ t ∈ good_triangles, is_good_triangle t 1 :=
sorry

end min_good_triangles_l813_813079


namespace number_of_small_cubes_l813_813328

-- Definition of the conditions from the problem
def painted_cube (n : ℕ) :=
  6 * (n - 2) * (n - 2) = 54

-- The theorem we need to prove
theorem number_of_small_cubes (n : ℕ) (h : painted_cube n) : n^3 = 125 :=
by
  have h1 : 6 * (n - 2) * (n - 2) = 54 := h
  sorry

end number_of_small_cubes_l813_813328


namespace total_students_correct_l813_813242

def num_boys : ℕ := 272
def num_girls : ℕ := num_boys + 106
def total_students : ℕ := num_boys + num_girls

theorem total_students_correct : total_students = 650 :=
by
  sorry

end total_students_correct_l813_813242


namespace determine_n_l813_813855

theorem determine_n (n : ℕ) (x : ℤ) (h : x^n + (2 + x)^n + (2 - x)^n = 0) : n = 1 :=
sorry

end determine_n_l813_813855


namespace Jina_mascots_total_l813_813490

theorem Jina_mascots_total :
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  teddies + bunnies + koala + additional_teddies = 51 :=
by
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  show teddies + bunnies + koala + additional_teddies = 51
  sorry

end Jina_mascots_total_l813_813490


namespace vector_problems_l813_813912

theorem vector_problems (a b c : ℝ×ℝ) (λ : ℝ) (θ : ℝ):
  a = (1, 3) →
  b = (λ, 3 * λ) →
  |b| = 2 * sqrt 10 →
  λ < 0 →
  c = (c.1, c.2) →
  |c| = sqrt 5 →
  |a + c| = sqrt ((1 + c.1)^2 + (3 + c.2)^2) →
  (a + c) ⬝ (2 * a - 3 * c) = 0 →
  b = (-2, -6) ∧ θ = π / 4 :=
by
  sorry

end vector_problems_l813_813912


namespace intersecting_circles_unique_point_l813_813103

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l813_813103


namespace units_digit_7_pow_6_pow_5_l813_813859

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end units_digit_7_pow_6_pow_5_l813_813859


namespace trigonometric_identity_sin_value_l813_813415

theorem trigonometric_identity (α : ℝ) (k : ℤ) (h₀ : α ≠ k * π / 2) :
  (f: ℝ→ℝ) f(α) = (sin(α - π/2) * cos(3 * π / 2 + α) * tan(π - α)) / (tan(-α - π) * sin(-α - π)) :=
  cos(α) :=
sorry

theorem sin_value (β : ℝ) (h₀ : (π / 2 + β) ≁ k * π / 2) (h₁ : β ∈ IV_quadrant) (h₂ : (cos(π / 2 + β) = -√3 / 3)):
  sin(2 * β + π / 6) = (1 + 2 * √6) / 6 :=
sorry

end trigonometric_identity_sin_value_l813_813415


namespace dealer_purchased_articles_l813_813645

theorem dealer_purchased_articles (profit_percentage : ℝ) (total_cost total_sp : ℝ) (sp_qty purchased_qty : ℕ) 
  (h_profit : profit_percentage = 0.65) 
  (h_total_cost : total_cost = 25) 
  (h_total_sp : total_sp = 33) 
  (h_sp_qty : sp_qty = 12) 
  (purchased_qty = 15 : Prop) : 
  (total_sp / sp_qty) - (total_cost / purchased_qty) = 0.65 * (total_cost / purchased_qty) :=
by 
  sorry

end dealer_purchased_articles_l813_813645


namespace train_cross_pole_time_l813_813668

/-- The speed of the train in km/hr -/
def speed_km_per_hr : ℕ := 52

/-- The length of the train in meters -/
def train_length_m : ℕ := 260

/-- Conversion factor from km/hr to m/s -/
def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

/-- The speed of the train in m/s -/
def speed_m_per_s : ℝ := speed_km_per_hr * km_per_hr_to_m_per_s

/-- Time taken for the train to cross the pole in seconds -/
def time_seconds : ℝ := train_length_m / speed_m_per_s

theorem train_cross_pole_time :
  time_seconds ≈ 18 := sorry

end train_cross_pole_time_l813_813668


namespace find_cost_price_per_meter_l813_813274

noncomputable def cost_price_per_meter
  (total_cloth : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_cloth) / total_cloth

theorem find_cost_price_per_meter :
  cost_price_per_meter 75 4950 15 = 51 :=
by
  unfold cost_price_per_meter
  sorry

end find_cost_price_per_meter_l813_813274


namespace polynomial_identity_l813_813874

theorem polynomial_identity (P : ℝ[X])
  (h₀ : P.eval 0 = 0)
  (h₁ : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) :
  ∀ x : ℝ, P.eval x = x :=
by
  sorry

end polynomial_identity_l813_813874


namespace evaluate_f_at_2_l813_813924

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem evaluate_f_at_2 : f 2 = 259 := 
by
  -- Substitute x = 2 into the polynomial and simplify the expression.
  sorry

end evaluate_f_at_2_l813_813924


namespace max_side_of_triangle_l813_813743

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813743


namespace max_side_length_l813_813780

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813780


namespace line_perpendicular_l813_813459

-- Definitions for the conditions given in the problem
def line1 : ℝ → Prop := λ a x y, a * x + 2 * y + 2 = 0
def line2 : ℝ → ℝ → Prop := λ x y, 3 * x - y - 2 = 0

-- The Lean statement to prove the mathematical equivalence
theorem line_perpendicular (a : ℝ) (x y : ℝ) :
  (∀ x y, line1 a x y ∧ line2 x y) → a = 2 / 3 :=
by sorry

end line_perpendicular_l813_813459


namespace more_straws_than_paper_l813_813600

theorem more_straws_than_paper : 
  ∀ (pieces_of_paper straws : ℕ), 
  pieces_of_paper = 7 → straws = 15 → 
  (straws - pieces_of_paper) = 8 := 
by 
  intros pieces_of_paper straws h_paper h_straws 
  rw [h_paper, h_straws] 
  norm_num

end more_straws_than_paper_l813_813600


namespace area_of_circle_l813_813193

-- Define the points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (14, 9)

-- Define the statement that needs to be proven
theorem area_of_circle (ω : Type*) [euclidean_space ω] (A B : ω) :
  tangent_to_circle_at A intersect x_axis ∧ tangent_to_circle_at B intersect x_axis → 
  area_of_circle ω = 154.73 * real.pi :=
by 
  -- The proof is omitted as per the instructions
  sorry

end area_of_circle_l813_813193


namespace area_of_region_l813_813374

noncomputable def region_area : ℝ :=
  ∫ x in 0..1, (3 + real.sqrt x + x^2)

theorem area_of_region :
  (∫ x in 0..1, (3 + real.sqrt x + x^2)) = 4 :=
by
  sorry

end area_of_region_l813_813374


namespace evaluate_expression_l813_813364

-- Given conditions 
def x := 3
def y := 2

-- Prove that y + y(y^x + x!) evaluates to 30.
theorem evaluate_expression : y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end evaluate_expression_l813_813364


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813693

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813693


namespace determine_k_values_l813_813361

noncomputable def has_one_vertical_asymptote (k : ℝ) : Prop :=
  let g (x : ℝ) := (x^2 - 2*x + k) / (x^2 + 2*x - 8) in
  ((x - 2) = 0 → (x^2 - 2*x + k) = 0) ∧ ((x + 4) ≠ 0) ∨
  ((x + 4) = 0 → (x^2 - 2*x + k) = 0) ∧ ((x - 2) ≠ 0)

theorem determine_k_values :
  ∃ k : ℝ, has_one_vertical_asymptote k :=
begin
  use [0, -24],
  sorry
end

end determine_k_values_l813_813361


namespace degree_f_plus_g_is_3_l813_813500

def f (z : ℂ) : ℂ := a₃ * z^3 + a₂ * z^2 + a₁ * z + a₀
def g (z : ℂ) : ℂ := b₂ * z^2 + b₁ * z + b₀

theorem degree_f_plus_g_is_3
  (a₃ : ℂ) (a₂ : ℂ) (a₁ : ℂ) (a₀ : ℂ)
  (b₂ : ℂ) (b₁ : ℂ) (b₀ : ℂ)
  (h_a₃ : a₃ ≠ 0) :
  polynomial.degree (polynomial.C a₃ * polynomial.X^3 + polynomial.C a₂ * polynomial.X^2 + polynomial.C a₁ * polynomial.X + polynomial.C a₀ +
                     polynomial.C b₂ * polynomial.X^2 + polynomial.C b₁ * polynomial.X + polynomial.C b₀) = 3 :=
by 
  sorry

end degree_f_plus_g_is_3_l813_813500


namespace certain_number_example_l813_813453

theorem certain_number_example (x : ℝ) 
    (h1 : 213 * 16 = 3408)
    (h2 : 0.16 * x = 0.3408) : 
    x = 2.13 := 
by 
  sorry

end certain_number_example_l813_813453


namespace integral_cth_squared_eq_l813_813376

open Real

noncomputable def integral_cth_squared (x : ℝ) : ℝ :=
  ∫ (λ x, (cosh x / sinh x) ^ 2)

theorem integral_cth_squared_eq (x : ℝ) (C : ℝ) :
  integral_cth_squared x = x - coth x + C :=
by
  sorry

end integral_cth_squared_eq_l813_813376


namespace cost_two_cones_l813_813172

-- Definition for the cost of a single ice cream cone
def cost_one_cone : ℕ := 99

-- The theorem to prove the cost of two ice cream cones
theorem cost_two_cones : 2 * cost_one_cone = 198 := 
by 
  sorry

end cost_two_cones_l813_813172


namespace maria_punch_l813_813513

variable (L S W : ℕ)

theorem maria_punch (h1 : S = 3 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 36 :=
by
  sorry

end maria_punch_l813_813513


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813684

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813684


namespace a3_plus_a4_l813_813030

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 3^(n + 1)

theorem a3_plus_a4 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : sum_of_sequence S a) :
  a 3 + a 4 = 216 :=
sorry

end a3_plus_a4_l813_813030


namespace number_of_4_digit_numbers_divisible_by_9_l813_813939

theorem number_of_4_digit_numbers_divisible_by_9 :
  ∃ n : ℕ, (∀ k : ℕ, k ∈ Finset.range n → 1008 + k * 9 ≤ 9999) ∧
           (1008 + (n - 1) * 9 = 9999) ∧
           n = 1000 :=
by
  sorry

end number_of_4_digit_numbers_divisible_by_9_l813_813939


namespace intersection_of_circles_l813_813086

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l813_813086


namespace max_side_of_triangle_l813_813741

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813741


namespace max_side_length_is_11_l813_813675

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813675


namespace perpendiculars_concurrent_iff_perpendiculars_concurrent_l813_813904

theorem perpendiculars_concurrent_iff_perpendiculars_concurrent
  (ABC : Triangle)
  (P Q R : Point) :
  (∃ M : Point, lines_concurrent (perpendicular BC P) (perpendicular AC Q) (perpendicular AB R) M) ↔
  (∃ N : Point, lines_concurrent (perpendicular QR A) (perpendicular PR B) (perpendicular PQ C) N) := 
sorry

end perpendiculars_concurrent_iff_perpendiculars_concurrent_l813_813904


namespace part1_part2_l813_813285

-- Part 1: Proving the calculation result 

theorem part1 : (\sqrt(12) - 3 * \sqrt(\frac(1, 3))) / \sqrt(3) = 1 := 
by
  sorry

-- Part 2: Proving the equation has no solution

theorem part2 : ¬∃ x, (\frac(x-1, x+1) + \frac(4, x^2 - 1) = \frac(x+1, x-1)) := 
by
  sorry

end part1_part2_l813_813285


namespace tangent_sum_identity_l813_813631

theorem tangent_sum_identity (n : ℕ) (h : 0 ≤ n ∧ n ≤ 88) :
  ∑ k in Finset.range (n + 1), 1 / (Real.cos (k : ℕ) * Real.cos ((k + 1) : ℕ)) = 
  Real.tan ((n + 1) * Real.pi / 180) / Real.sin (Real.pi / 180) :=
sorry

end tangent_sum_identity_l813_813631


namespace least_weighings_needed_l813_813617

theorem least_weighings_needed (n : ℕ) (h : n = 13) : ∃ W, W = 8 ∧ 
  (∀ f : finset (fin n × fin n), (∀ p ∈ f, (p.1 ≠ p.2)) ∧ f.card = W 
  → (∃ S : finset ℝ, (∀ x ∈ S, ∃ p ∈ f, x = p.1 + p.2) ∧ S.card = W)) := 
by {
  use 8,
  split,
  repeat { sorry },
}

end least_weighings_needed_l813_813617


namespace overtime_ratio_l813_813653

theorem overtime_ratio (R_regular R_ot : ℝ) (h₁ : R_regular = 3) (h₂ : ∀ hours : ℝ, 0 < hours → hours ≤ 40 → total_regular_pay = 3 * hours) 
  (h₃ : total_received = 192) (h₄ : overtime_hours = 12) (h₅ : total_regular_pay = 40 * 3) (h₆ : total_overtime_pay = total_received - total_regular_pay) 
  (h₇ : R_ot * overtime_hours = total_overtime_pay) : R_ot / R_regular = 2 :=
begin
  -- sorry placeholder for proof
  sorry
end

end overtime_ratio_l813_813653


namespace max_side_length_is_11_l813_813683

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813683


namespace even_number_of_rooks_on_black_squares_l813_813526

theorem even_number_of_rooks_on_black_squares :
  ∀ (rooks : Fin 8 → Fin 8 × Fin 8), (∀ i j, i ≠ j → rooks i ≠ rooks j) →
    ∃ n, n % 2 = 0 ∧ 
    n = Finset.card {ij | ∃ i, rooks i = ij ∧ (ij.1 + ij.2) % 2 = 0} := 
by
  intros rooks no_two_attack
  sorry

end even_number_of_rooks_on_black_squares_l813_813526


namespace range_of_a_l813_813929

-- Definitions of the sets M and N
def M (a : ℝ) : set ℝ := {x | x * (x - a - 1) < 0}
def N : set ℝ := {x | x^2 - 2*x - 3 <= 0}

-- The given condition M ∪ N = N
def M_union_N_eq_N (a : ℝ) : Prop := M a ∪ N = N

-- Prove that the range of a is [-1, 2]
theorem range_of_a (a : ℝ) : M_union_N_eq_N a → -1 ≤ a ∧ a ≤ 2 :=
by
  intro h
  sorry

end range_of_a_l813_813929


namespace calculate_sum_l813_813345

theorem calculate_sum : 
  (∑ k in Finset.range 50, (3 + (k + 1) * 10) / 8^(51 - (k + 1))) = 500 / 7 := 
by
  sorry

end calculate_sum_l813_813345


namespace find_percentage_l813_813639

theorem find_percentage (P : ℝ) (h : P / 100 * 3200 = 0.20 * 650 + 190) : P = 10 :=
by 
  sorry

end find_percentage_l813_813639


namespace A_subset_B_determine_B_when_A_is_given_A_eq_B_if_singleton_l813_813163

/-- Definition of function f(x) based on real parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b

/-- Set A using definition of function f -/
def A (a b : ℝ) : set ℝ := { x : ℝ | x = f a b x }

/-- Set B using definition of function f -/
def B (a b : ℝ) : set ℝ := { x : ℝ | x = f a b (f a b x) }

/-- Prove A ⊆ B for any real a and b -/
theorem A_subset_B (a b : ℝ) : A a b ⊆ B a b := 
by {
  -- Proof omitted
  sorry
}

/-- When A = {-1, 3}, determine B for derived a and b -/
theorem determine_B_when_A_is_given : 
  ∃ (a b : ℝ), A a b = {-1, 3} ∧ B a b = { -1, 3, -real.sqrt 3, real.sqrt 3 } :=
by {
  -- Proof omitted
  sorry
}

/-- Prove that if A has only one element, then A = B -/
theorem A_eq_B_if_singleton (a b : ℝ) (x : ℝ) (h : A a b = {x}) : A a b = B a b :=
by {
  -- Proof omitted
  sorry
}

end A_subset_B_determine_B_when_A_is_given_A_eq_B_if_singleton_l813_813163


namespace arden_cricket_club_members_l813_813082

theorem arden_cricket_club_members (gloves_cost cap_extra_cost total_expenditure : ℕ) 
(gloves_cost_eq : gloves_cost = 6) 
(cap_extra_cost_eq : cap_extra_cost = 8) 
(total_expenditure_eq : total_expenditure = 4140) :
  let cap_cost := gloves_cost + cap_extra_cost
  let total_cost_per_member := 2 * (gloves_cost + cap_cost)
  total_expenditure / total_cost_per_member = 103 :=
by
  have gloves_cost_eq6 : gloves_cost = 6 := gloves_cost_eq
  have cap_extra_cost_eq8 : cap_extra_cost = 8 := cap_extra_cost_eq
  have total_expenditure_val : total_expenditure = 4140 := total_expenditure_eq
  let cap_cost := gloves_cost + cap_extra_cost
  let total_cost_per_member := 2 * (gloves_cost + cap_cost)
  have total_cost_per_member_eq : total_cost_per_member = 40 := by 
    rw [gloves_cost_eq6, cap_extra_cost_eq8]
    simp
  have total_expenditure_divided : total_expenditure / 40 = 103 := by 
    rw total_expenditure_val
    norm_num
  sorry

end arden_cricket_club_members_l813_813082


namespace max_side_length_of_triangle_l813_813755

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l813_813755


namespace all_roads_from_capital_outbound_l813_813281

def RoadNetwork :=
  { cities : Type,
    roads : cities → cities → Prop,
    directed : ∀ a b, roads a b → roads b a → False,
    acyclic : ∀ a, ¬ (roads a a ∨ (∃ b, roads a b ∧ roads b a)) }

def CapitalCity (cities : Type) := cities

def OutboundOnlyCity (network : RoadNetwork) (city : network.cities) : Prop :=
  ¬ ∃ b : network.cities, network.roads b city

def Week (network : RoadNetwork) : Prop :=
  ∀ cit, OutboundOnlyCity network cit → ∃ b, network.directed cit b

theorem all_roads_from_capital_outbound :
  ∀ (network : RoadNetwork) (capital : CapitalCity network.cities),
  (∀ week : Week network, net.directed) ∨ (∀ t, ∃ week t, ∀ r, OutboundOnlyCity week t) ∨ (∀ w, ∃ city, OutboundOnlyCity w net.city): 
  -- Prove that there will come a week when all roads from the capital city will be outbound.
  sorry

end all_roads_from_capital_outbound_l813_813281


namespace max_side_length_l813_813770

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813770


namespace nearest_integer_l813_813262

-- Definition for the expanded terms
def expansion_val (a b : ℝ) (n : ℕ) : ℝ := 
  ∑ k in Finset.range (n + 1), (Nat.choose n k : ℕ) * (a^(n-k)) * (b^k)

-- Define (3 + sqrt(2))^5 and (3 - sqrt(2))^5 using the expansion
def expr1 := expansion_val 3 (Real.sqrt 2) 5
def expr2 := expansion_val 3 (-Real.sqrt 2) 5

-- summing the expressions
def sum_expr := expr1 + expr2

-- defining the calculation for the integer part
def expected_sum : ℝ := 2 * (243 + 540 + 180) -- simplifying the sum given in the solution

-- main theorem to prove
theorem nearest_integer : Int.round (3 + Real.sqrt 2)^5 = 1926 := by
  have expr1 := expansion_val 3 (Real.sqrt 2) 5
  have expr2 := expansion_val 3 (-Real.sqrt 2) 5
  have sum_expr := expr1 + expr2
  have h1 : expected_sum = 1926 := by  -- expected sum is already simplified
    sorry  -- actual proof skipped
  exact Int.round (3 + Real.sqrt 2)^5 = 1926

end nearest_integer_l813_813262


namespace trajectory_ellipse_l813_813578

theorem trajectory_ellipse (x y : ℝ) :
  (real.sqrt ((x - 3) ^ 2 + y ^ 2) / real.abs (x - 25 / 3) = 3 / 5) →
  (x ^ 2 / 25 + y ^ 2 / 16 = 1) :=
sorry

end trajectory_ellipse_l813_813578


namespace avg_rates_inequality_l813_813567

def avg_rate_change (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

theorem avg_rates_inequality :
  let f := λ x : ℝ, 1 / x in
  let k1 := avg_rate_change f 1 2 in
  let k2 := avg_rate_change f 2 3 in
  let k3 := avg_rate_change f 3 4 in
  k1 < k2 ∧ k2 < k3 :=
by
  let f := λ x : ℝ, 1 / x
  let k1 := avg_rate_change f 1 2
  let k2 := avg_rate_change f 2 3
  let k3 := avg_rate_change f 3 4
  have h1 : k1 = -1/2 := sorry
  have h2 : k2 = -1/6 := sorry
  have h3 : k3 = -1/12 := sorry
  rw [h1, h2, h3]
  split
  · exact sorry
  · exact sorry

end avg_rates_inequality_l813_813567


namespace discount_difference_l813_813837

def single_discount (original: ℝ) (discount: ℝ) : ℝ :=
  original * (1 - discount)

def successive_discount (original: ℝ) (first_discount: ℝ) (second_discount: ℝ) : ℝ :=
  original * (1 - first_discount) * (1 - second_discount)

theorem discount_difference : 
  let original := 12000
  let single_disc := 0.30
  let first_disc := 0.20
  let second_disc := 0.10
  single_discount original single_disc - successive_discount original first_disc second_disc = 240 := 
by sorry

end discount_difference_l813_813837


namespace congruent_figures_coincide_l813_813623

theorem congruent_figures_coincide (F1 F2 : Type) 
  [figure F1] [figure F2] 
  (congruent : F1 ≃ F2) :
  ∃ (translations : ℝ^3) (rotations : ℝ^3 × ℝ), 
    move_figure F1 translations rotations = F2 := 
sorry

end congruent_figures_coincide_l813_813623


namespace find_a_l813_813027

theorem find_a (a : ℝ) (h : coeff_x3_in_expansion_eq5 (a + x) (1 - x)^6 = 5) : a = 1 / 2 := sorry

end find_a_l813_813027


namespace P_N_S_collinear_l813_813235

-- Define the structure for points and lines
variable {Point Line : Type*}
variable (A B C D I K L M N P Q R S : Point)
variable (AI BI CI DI PK QL RM : Line)

-- Conditions based on the problem description
def quadrilateral_touches_circle_with_center_I :=
  touches_circle_with_tangents A B C D I K L M N

def point_on_line (P : Point) (L : Line) : Prop :=
  lies_on P L

def PK_meets_BI_at_Q : Prop :=
  intersection PK BI Q

def QL_meets_CI_at_R : Prop :=
  intersection QL CI R

def RM_meets_DI_at_S : Prop :=
  intersection RM DI S

-- Main theorem statement converting the conditions and proving collinearity
theorem P_N_S_collinear
  (h1 : quadrilateral_touches_circle_with_center_I A B C D I K L M N)
  (h2 : point_on_line P AI)
  (h3 : PK_meets_BI_at_Q PK BI Q)
  (h4 : QL_meets_CI_at_R QL CI R)
  (h5 : RM_meets_DI_at_S RM DI S) :
  collinear P N S :=
sorry

end P_N_S_collinear_l813_813235


namespace children_ages_l813_813647

-- Define the ages of the four children
variable (a b c d : ℕ)

-- Define the conditions
axiom h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom h2 : a + b + c + d = 31
axiom h3 : (a - 4) + (b - 4) + (c - 4) + (d - 4) = 16
axiom h4 : (a - 7) + (b - 7) + (c - 7) + (d - 7) = 8
axiom h5 : (a - 11) + (b - 11) + (c - 11) + (d - 11) = 1
noncomputable def ages : ℕ × ℕ × ℕ × ℕ := (12, 10, 6, 3)

-- The theorem to prove
theorem children_ages (h1 : a = 12) (h2 : b = 10) (h3 : c = 6) (h4 : d = 3) : a = 12 ∧ b = 10 ∧ c = 6 ∧ d = 3 :=
by sorry

end children_ages_l813_813647


namespace find_max_side_length_l813_813793

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813793


namespace imaginary_number_condition_fourth_quadrant_condition_l813_813394

-- Part 1: Prove that if \( z \) is purely imaginary, then \( m = 0 \)
theorem imaginary_number_condition (m : ℝ) :
  (m * (m + 2) = 0) ∧ (m^2 + m - 2 ≠ 0) → m = 0 :=
by
  sorry

-- Part 2: Prove that if \( z \) is in the fourth quadrant, then \( 0 < m < 1 \)
theorem fourth_quadrant_condition (m : ℝ) :
  (m * (m + 2) > 0) ∧ (m^2 + m - 2 < 0) → (0 < m ∧ m < 1) :=
by
  sorry

end imaginary_number_condition_fourth_quadrant_condition_l813_813394


namespace number_of_people_who_only_speak_one_language_l813_813468

theorem number_of_people_who_only_speak_one_language :
  let total_people := 40 in
  let speak_latin := 20 in
  let speak_french := 22 in
  let speak_spanish := 15 in
  let speak_none := 5 in
  let speak_latin_french := 8 in
  let speak_latin_spanish := 6 in
  let speak_french_spanish := 4 in
  let speak_all_three := 3 in
  let speak_atleast_one := total_people - speak_none in
  let speak_two_languages := (speak_latin_french + speak_latin_spanish + speak_french_spanish) - 2 * speak_all_three in
  let only_speak_one := speak_atleast_one - speak_two_languages - speak_all_three in
  only_speak_one = 20 :=
by
  -- Definitions and assumptions
  let total_people := 40
  let speak_latin := 20
  let speak_french := 22
  let speak_spanish := 15
  let speak_none := 5
  let speak_latin_french := 8
  let speak_latin_spanish := 6
  let speak_french_spanish := 4
  let speak_all_three := 3
  -- Calculations
  let speak_atleast_one := total_people - speak_none
  let speak_two_languages := (speak_latin_french + speak_latin_spanish + speak_french_spanish) - 2 * speak_all_three
  let only_speak_one := speak_atleast_one - speak_two_languages - speak_all_three
  -- Proof
  have h1 : speak_atleast_one = total_people - speak_none := rfl
  have h2 : speak_two_languages = (speak_latin_french + speak_latin_spanish + speak_french_spanish) - 2 * speak_all_three := rfl
  have h3 : only_speak_one = speak_atleast_one - speak_two_languages - speak_all_three := rfl
  have h4 : only_speak_one = 20 := by
    -- The actual proof logic will go here, but for now we use sorry to skip it.
    sorry
  show only_speak_one = 20 from h4

end number_of_people_who_only_speak_one_language_l813_813468


namespace range_a_p_range_a_pq_l813_813408

variables (a : ℝ) (ρ α : ℝ)

-- Definition of proposition p
def prop_p : Prop := ∀ x : ℝ, x^2 + 1 >= a

-- Definition of proposition q
def prop_q : Prop := (ρ * cos α)^2 - (ρ * sin a)^2 = a + 2 × (1 : ℝ)  -- Represents hyperbola with foci on x-axis

-- Theorem 1: If p is true, then a ≤ 1
theorem range_a_p (h_p : prop_p a) : a <= 1 := by
  sorry

-- Theorem 2: If p and q are true, then -2 < a ≤ 1
theorem range_a_pq (h_p : prop_p a) (h_q : prop_q ρ α a) : -2 < a ∧ a <= 1 := by
  sorry

end range_a_p_range_a_pq_l813_813408


namespace sum_of_products_of_roots_eq_neg3_l813_813152

theorem sum_of_products_of_roots_eq_neg3 {p q r s : ℂ} 
  (h : ∀ {x : ℂ}, 4 * x^4 - 8 * x^3 + 12 * x^2 - 16 * x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) : 
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := 
sorry

end sum_of_products_of_roots_eq_neg3_l813_813152


namespace isosceles_triangle_base_angle_l813_813080

theorem isosceles_triangle_base_angle (α β γ : ℝ) 
  (h_triangle: α + β + γ = 180) 
  (h_isosceles: α = β ∨ α = γ ∨ β = γ) 
  (h_one_angle: α = 80 ∨ β = 80 ∨ γ = 80) : 
  (α = 50 ∨ β = 50 ∨ γ = 50) ∨ (α = 80 ∨ β = 80 ∨ γ = 80) :=
by 
  sorry

end isosceles_triangle_base_angle_l813_813080


namespace part_1_a_part_1_b_part_2_l813_813044

variable (a : ℝ)
def A (a : ℝ) : set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 3}
def B : set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def U : set ℝ := set.univ

theorem part_1_a (h: a = 2) : A a ∪ B = {x | -2 ≤ x ∧ x ≤ 7} := sorry

theorem part_1_b (h: a = 2) : (U \ A a) ∩ B = {x | -2 ≤ x ∧ x < 1} := sorry

theorem part_2 : (A a ⊆ B) ↔ a ∈ set.Icc (-∞ : ℝ) (-4 : ℝ) ∪ set.Icc (-1 : ℝ) (1 / 2) := sorry

end part_1_a_part_1_b_part_2_l813_813044


namespace length_of_AB_l813_813474

variables (AB CD : ℝ)

-- Given conditions
def area_ratio (h : ℝ) : Prop := (1/2 * AB * h) / (1/2 * CD * h) = 4
def sum_condition : Prop := AB + CD = 200

-- The proof problem: proving the length of AB
theorem length_of_AB (h : ℝ) (h_area_ratio : area_ratio AB CD h) 
  (h_sum_condition : sum_condition AB CD) : AB = 160 :=
sorry

end length_of_AB_l813_813474


namespace max_x_plus_2y_l813_813504

theorem max_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x + 2 * y ≤ 3 :=
sorry

end max_x_plus_2y_l813_813504


namespace sum_of_areas_l813_813322

theorem sum_of_areas (s : ℕ) (side1 : ℕ) (side2 : ℕ) (side_sqr : ℕ) (R_area : ℕ) :
  s = 4 → side1 = 2 → side2 = 4 → side_sqr = 2 → s * s = 16 ∧ side1 * side2 = 8 ∧ side_sqr * side_sqr = 4 ∧ R_area = 16 - (8 + 4) →
  let R as ℕ := 4 in
  let m : ℕ := 4 in
  let n : ℕ := 1 in
  gcd m n = 1 → (m + n) = 4 := 
by
  sorry

end sum_of_areas_l813_813322


namespace shift_to_even_function_l813_813249

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def shifted_function (m : ℝ) (x : ℝ) : ℝ :=
  sin (2 * (x + m) + π / 4)

theorem shift_to_even_function : is_even_function (shifted_function (π / 8)) :=
  sorry

end shift_to_even_function_l813_813249


namespace solve_quadratic_roots_l813_813000

theorem solve_quadratic_roots (b c : ℝ) 
  (h : {1, 2} = {x : ℝ | x^2 + b * x + c = 0}) : 
  b = -3 ∧ c = 2 :=
by
  sorry

end solve_quadratic_roots_l813_813000


namespace part1_part2_l813_813414

variable {α : ℝ} (h1 : α ∈ Set.Ioo (π) (3 * π / 2))

def f (α : ℝ) : ℝ := 
  (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) / 
  (tan (-α - π) * sin (-α - π))

theorem part1 (h1 : α ∈ Set.Ioo (π) (3 * π / 2)) :
  f(α) = -cos(α) := by
  sorry

theorem part2 (h2 : cos(α - 3 * π / 2) = 1 / 5) (h1 : α ∈ Set.Ioo (π) (3 * π / 2)) :
  f(2 * α) = -23 / 25 := by
  sorry

end part1_part2_l813_813414


namespace num_diagonals_29_sides_l813_813841

-- Define the number of sides
def n : Nat := 29

-- Calculate the combination (binomial coefficient) of selecting 2 vertices from n vertices
def binom (n k : Nat) : Nat := Nat.choose n k

-- Define the number of diagonals in a polygon with n sides
def num_diagonals (n : Nat) : Nat := binom n 2 - n

-- State the theorem to prove the number of diagonals for a polygon with 29 sides is 377
theorem num_diagonals_29_sides : num_diagonals 29 = 377 :=
by
  sorry

end num_diagonals_29_sides_l813_813841


namespace max_triangle_side_l813_813722

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l813_813722


namespace range_of_a_l813_813045

def A : Set ℝ := { x | x^2 - 3 * x + 2 ≤ 0 }
def B (a : ℝ) : Set ℝ := { x | 1 / (x - 3) < a }

theorem range_of_a (a : ℝ) : A ⊆ B a ↔ a > -1/2 :=
by sorry

end range_of_a_l813_813045


namespace valid_starting_days_count_l813_813305

def is_valid_starting_day (d : ℕ) : Prop :=
  (d % 7 = 3 ∨ d % 7 = 4 ∨ d % 7 = 5)

theorem valid_starting_days_count : 
  (finset.filter is_valid_starting_day (finset.range 7)).card = 3 :=
begin
  sorry
end

end valid_starting_days_count_l813_813305


namespace quadratic_vertex_distance_l813_813010

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 0 3
def B := Point.mk 1 0
def C := Point.mk 4 3

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- The conditions given in the problem
theorem quadratic_vertex_distance :
  ∃ a b c : ℝ, 
  (quadratic a b c A.x = A.y) ∧ 
  (quadratic a b c B.x = B.y) ∧ 
  (quadratic a b c C.x = C.y) ∧ 
  let vertex_x := -b / (2 * a) in 
  let vertex_y := quadratic a b c vertex_x in 
  abs(vertex_y) = 1 :=
by
  sorry

end quadratic_vertex_distance_l813_813010


namespace village_population_80_percent_l813_813632

theorem village_population_80_percent (total_population : ℝ) (percentage : ℝ) (expected_result : ℝ) : 
  total_population = 28800 → 
  percentage = 0.80 →
  expected_result = total_population * percentage → 
  expected_result = 23040 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3 sorry

end village_population_80_percent_l813_813632


namespace angle_SRT_l813_813060

-- Define angles in degrees
def angle_P : ℝ := 50
def angle_Q : ℝ := 60
def angle_R : ℝ := 40

-- Define the problem: Prove that angle SRT is 30 degrees given the above conditions
theorem angle_SRT : 
  (angle_P = 50 ∧ angle_Q = 60 ∧ angle_R = 40) → (∃ angle_SRT : ℝ, angle_SRT = 30) :=
by
  intros h
  sorry

end angle_SRT_l813_813060


namespace number_of_special_two_digit_numbers_l813_813449

def isTwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def reverseNumber (n : ℕ) : ℕ :=
  let t := n / 10
  let u := n % 10
  10 * u + t

def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem number_of_special_two_digit_numbers : 
  ∃ count, count = 2 ∧ count = (Finset.filter (λ n, 
    isTwoDigitNumber n ∧ 
    isPerfectSquare ((n * n) + (reverseNumber n * reverseNumber n))
  ) (Finset.range 100)).card :=
by
  sorry

end number_of_special_two_digit_numbers_l813_813449


namespace cream_cheese_volume_l813_813170

theorem cream_cheese_volume
  (raw_spinach : ℕ)
  (spinach_reduction : ℕ)
  (eggs_volume : ℕ)
  (total_volume : ℕ)
  (cooked_spinach : ℕ)
  (cream_cheese : ℕ) :
  raw_spinach = 40 →
  spinach_reduction = 20 →
  eggs_volume = 4 →
  total_volume = 18 →
  cooked_spinach = raw_spinach * spinach_reduction / 100 →
  cream_cheese = total_volume - cooked_spinach - eggs_volume →
  cream_cheese = 6 :=
by
  intros h_raw_spinach h_spinach_reduction h_eggs_volume h_total_volume h_cooked_spinach h_cream_cheese
  sorry

end cream_cheese_volume_l813_813170


namespace sum_of_number_and_reverse_eq_33_l813_813563

theorem sum_of_number_and_reverse_eq_33 :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (10 * a + b) - (10 * b + a) = 7 * (a - b) →
  (10 * a + b) + (10 * b + a) = 33 :=
by
  intros a b h1 h2 h3 h4 h5
  sorry

end sum_of_number_and_reverse_eq_33_l813_813563


namespace sum_of_corners_is_2004_l813_813259

-- Conditions
def num_elements : ℕ := 221
def total_sum : ℝ := 110721
-- additional conditions: rows and first & last columns are APs

-- Data type for the table
structure RectTable :=
(n m : ℕ) -- dimensions
(a : ℕ → ℕ → ℝ) -- elements

def is_arithmetic_progression (seq : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ i : ℕ, seq (i + 1) = seq i + d

def is_valid_table (t : RectTable) : Prop :=
  (t.n * t.m = num_elements) ∧
  (∑ i j, t.a i j = total_sum) ∧
  (∀ i, is_arithmetic_progression (λ j, t.a i j)) ∧
  (is_arithmetic_progression (λ i, t.a i 1)) ∧
  (is_arithmetic_progression (λ i, t.a i t.m))

-- Define the sum of the four corners
def sum_corners (t : RectTable) : ℝ :=
  t.a 0 0 + t.a 0 (t.m - 1) + t.a (t.n - 1) 0 + t.a (t.n - 1) (t.m - 1)

-- The proof statement
theorem sum_of_corners_is_2004 (t : RectTable) (ht : is_valid_table t) : 
  sum_corners t = 2004 :=
by sorry

end sum_of_corners_is_2004_l813_813259


namespace sum_of_numbers_in_ratio_l813_813898

theorem sum_of_numbers_in_ratio 
  (x : ℕ)
  (h : 5 * x = 560) : 
  2 * x + 3 * x + 4 * x + 5 * x = 1568 := 
by 
  sorry

end sum_of_numbers_in_ratio_l813_813898


namespace ratio_of_areas_l813_813142

theorem ratio_of_areas (a : ℝ) : 
  let A := (0, 0)
  let B := (a, 0)
  let C := (a, a)
  let D := (0, a)
  let E := (2 * a / 3, -a / (3 * √2))
  let F := (a + a / (3 * √2), 2 * a / 3)
  let G := (2 * a / 3, a + a / (3 * √2))
  let H := (-a / (3 * √2), 2 * a / 3)
  let area_ABCD := a^2
  let EF := (√((a / 3 + a / (3 * √2))^2 + (a / (3 * √2))^2))
  let area_EFGH := (EF^2)
  area_EFGH / area_ABCD = (2 + √2) / 9 :=
sorry

end ratio_of_areas_l813_813142


namespace cans_to_collect_l813_813514

theorem cans_to_collect
  (martha_cans : ℕ)
  (diego_half_plus_ten : ℕ)
  (total_cans_required : ℕ)
  (martha_cans_collected : martha_cans = 90)
  (diego_collected : diego_half_plus_ten = (martha_cans / 2) + 10)
  (goal_cans : total_cans_required = 150) :
  total_cans_required - (martha_cans + diego_half_plus_ten) = 5 :=
by
  sorry

end cans_to_collect_l813_813514


namespace total_marks_l813_813134

theorem total_marks (Keith_marks Larry_marks Danny_marks : ℕ)
  (hK : Keith_marks = 3)
  (hL : Larry_marks = 3 * Keith_marks)
  (hD : Danny_marks = Larry_marks + 5) :
  Keith_marks + Larry_marks + Danny_marks = 26 := 
by
  sorry

end total_marks_l813_813134


namespace find_max_side_length_l813_813786

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813786


namespace find_max_side_length_l813_813791

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813791


namespace dot_product_necessary_condition_dot_product_not_sufficient_condition_l813_813417

open Function

theorem dot_product_necessary_condition
  {a b c : ℝ^2} (a_nonzero : a ≠ 0) (h₁ : a•b = a•c) : b ≠ c → a ≠ 0 :=
by sorry

theorem dot_product_not_sufficient_condition
  {a b c : ℝ^2} (a_nonzero : a ≠ 0) (h₁ : a•b = a•c) (h₂ : b = c) : a•b = a•c :=
by sorry

end dot_product_necessary_condition_dot_product_not_sufficient_condition_l813_813417


namespace correct_statements_about_f_l813_813389

def f (x : ℝ) : ℝ := cos (2 * x) - 2 * sqrt 3 * sin x * cos x

theorem correct_statements_about_f :
  (∀ x, f x ≤ 2) ∧ (¬(∀ x, f (π / 6) = f (-π / 6))) ∧ (f (7 * π / 12) = 0) 
    ∧ (¬(∀ x ∈ set.Icc (-π / 6) (π / 3), monotone_on f (set.Icc (-π / 6) (π / 3)))) :=
sorry

end correct_statements_about_f_l813_813389


namespace restaurant_supplies_ratio_l813_813824

theorem restaurant_supplies_ratio 
  (budget : ℕ := 3000)
  (food : ℕ := budget / 3)
  (wages : ℕ := 1250)
  (supplies : ℕ := budget - food - wages) :
  supplies : budget = 1 : 4 :=
by
  sorry

end restaurant_supplies_ratio_l813_813824


namespace sum_powers_divisible_by_13_l813_813237

-- Statement of the problem in Lean
theorem sum_powers_divisible_by_13 (a b p : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : p = 13) :
  (a^1974 + b^1974) % p = 0 := 
by
  sorry

end sum_powers_divisible_by_13_l813_813237


namespace radar_arrangements_l813_813357

-- Define the number of letters in the word RADAR
def total_letters : Nat := 5

-- Define the number of times each letter is repeated
def repetition_R : Nat := 2
def repetition_A : Nat := 2

-- The expected number of unique arrangements
def expected_unique_arrangements : Nat := 30

theorem radar_arrangements :
  (Nat.factorial total_letters) / (Nat.factorial repetition_R * Nat.factorial repetition_A) = expected_unique_arrangements := by
  sorry

end radar_arrangements_l813_813357


namespace central_angle_of_regular_hexagon_l813_813561

-- Define the total degrees in a circle
def total_degrees_in_circle : ℝ := 360

-- Define the number of sides in a regular hexagon
def sides_in_hexagon : ℕ := 6

-- Theorems to prove that the central angle of a regular hexagon is 60°
theorem central_angle_of_regular_hexagon :
  total_degrees_in_circle / sides_in_hexagon = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l813_813561


namespace psychiatrist_is_dushmen_l813_813247

noncomputable def doctor_names := ["Deschamp", "Dubois", "Dushmen"]
noncomputable def specialties := ["therapist", "psychiatrist", "ophthalmologist"]
structure Woman :=
  (name: String) (height: ℝ) (weight: ℝ)
structure Doctor :=
  (name: String) (specialty: String) (wife: Woman) (weight: ℝ)

variables (therapist psychiatrist ophthalmologist: Doctor)
variables (madame_dubois madame_deschamps madame_dushmen: Woman)

axiom therapist_conditions :
  madame_dubois.height = therapist.wife.height + (madame_dubois.height - therapist.height) ∧
  madame_dushmen.weight = therapist.weight ∧
  madame_deschamps.weight = madame_dushmen.weight - 10

axiom dushmen_conditions : 
  psychiatrist.weight = ophthalmologist.weight + 20 ∧
  psychiatrist.name = "Dushmen"

theorem psychiatrist_is_dushmen :
  psychiatrist.name = "Dushmen" :=
by
  sorry

end psychiatrist_is_dushmen_l813_813247


namespace prism_height_l813_813245

theorem prism_height (a h : ℝ) 
  (base_side : a = 10) 
  (total_edge_length : 3 * a + 3 * a + 3 * h = 84) : 
  h = 8 :=
by sorry

end prism_height_l813_813245


namespace sum_products_eq_binom_l813_813993

open Nat

theorem sum_products_eq_binom (n k : ℕ) (hk : k > 0) (hn : n > 0) :
  (∑ (y : Finset (ℕ × ℕ ... × ℕ)) in (Finset.filter (λ y, y.sum = n) (Finset.univ ...)), y.prod) = (n-1).choose (k-1) :=
sorry

end sum_products_eq_binom_l813_813993


namespace slices_left_for_phill_correct_l813_813188

-- Define the initial conditions about the pizza and the distribution.
def initial_pizza := 1
def slices_after_first_cut := initial_pizza * 2
def slices_after_second_cut := slices_after_first_cut * 2
def slices_after_third_cut := slices_after_second_cut * 2
def total_slices_given_to_two_friends := 2 * 2
def total_slices_given_to_three_friends := 3 * 1
def total_slices_given_out := total_slices_given_to_two_friends + total_slices_given_to_three_friends
def slices_left_for_phill := slices_after_third_cut - total_slices_given_out

-- State the theorem we need to prove.
theorem slices_left_for_phill_correct : slices_left_for_phill = 1 := by sorry

end slices_left_for_phill_correct_l813_813188


namespace sum_imaginary_unit_l813_813840

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the sum from i to i^2014
def geometric_sum (n : ℕ) (z : ℂ) : ℂ :=
  (z * (z ^ n - 1)) / (z - 1)

-- State the theorem to be proved
theorem sum_imaginary_unit : geometric_sum 2014 i = -1 + i := by
  sorry

end sum_imaginary_unit_l813_813840


namespace prove_survey_suitable_for_sampling_l813_813621

def Survey (A B C D : Prop) : Prop :=
  (¬ (A ∧ B ∧ C) ∧ D)

def survey_suitable_for_sampling : Prop := 
  let A := false
  let B := false
  let C := false
  let D := true
  Survey A B C D

theorem prove_survey_suitable_for_sampling : survey_suitable_for_sampling :=
  by
    simp [survey_suitable_for_sampling, Survey]
    sorry

end prove_survey_suitable_for_sampling_l813_813621


namespace tournament_six_points_l813_813472

theorem tournament_six_points (n : ℕ) (hn : n = 9) : 
  let participants := 2^n in
  let points := 6 in
  participants = 512 → 
  (n = 9) →
  ∃ finishers_with_six_points, finishers_with_six_points = 84 := 
by
  intros participants_eq hn_eq
  have num_participants : 2^9 = 512 := by norm_num
  rw [←participants_eq] at num_participants
  exact ⟨84, rfl⟩

end tournament_six_points_l813_813472


namespace isosceles_triangle_infinte_solution_l813_813128

theorem isosceles_triangle_infinte_solution (n : ℕ) (h_pos : n > 0) :
  (AB : ℕ) = 3 * n + 6 ∧ (AC : ℕ) = 2 * n + 8 ∧ (BC : ℕ) = 2 * n + 8 ∧ (∠B > ∠C) := 
begin
  sorry
end

end isosceles_triangle_infinte_solution_l813_813128


namespace max_side_of_triangle_l813_813805

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813805


namespace smallest_angle_of_right_triangle_l813_813579

theorem smallest_angle_of_right_triangle (x : ℝ) (h1 : 3 * x + 2 * x = 90) : 2 * x = 36 :=
by 
  have : 5 * x = 90 := h1
  have h2 : x = 90 / 5 := by rw [this, div_eq_mul_one_div, div_self] -- Rationalizing 
  exact mul_eq_add_sub_of_newthing_eq6 h2 -- Rationalizing

end smallest_angle_of_right_triangle_l813_813579


namespace transition_term_l813_813612

theorem transition_term (k : ℕ) : (2 * k + 2) + (2 * k + 3) = (2 * (k + 1) + 1) + (2 * k + 2) :=
by
  sorry

end transition_term_l813_813612


namespace max_side_of_triangle_l813_813807

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813807


namespace find_A_inter_B_find_A_union_complement_B_range_of_a_l813_813442

open Set

variable {U : Type} [TopologicalSpace U] [Preorder U] [LinearOrder U]

def A : Set U := {x | 2 ≤ x ∧ x < 4}
def B : Set U := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : U) : Set U := {x | x < a}

theorem find_A_inter_B :
  A ∩ B = {x | 3 ≤ x ∧ x < 4} := 
sorry

theorem find_A_union_complement_B :
  A ∪ (compl B) = {x | x < 4} := 
sorry

theorem range_of_a (a : U) (h : A ⊆ C a) :
  4 ≤ a :=
sorry

end find_A_inter_B_find_A_union_complement_B_range_of_a_l813_813442


namespace max_side_of_triangle_l813_813751

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813751


namespace larry_cards_after_taken_l813_813137

def larry_initial_cards := 67
def dennis_takes_away := 9
def larry_remaining_cards := 58

theorem larry_cards_after_taken (initial : ℕ) (taken : ℕ) : initial - taken = larry_remaining_cards :=
by
  have h : initial - taken = 58,
  exact dec_trivial, -- This is to ensure the calculation initial - taken == 58 is done automatically
  exact h

end larry_cards_after_taken_l813_813137


namespace power_one_power_two_l813_813337

-- Defining the conditions for the first power
def x1 : ℝ := 0.02
def n1 : ℕ := 30

-- Statement for the first power problem
theorem power_one :
  (1 + x1)^n1 ≈ 1.8114 := by
  sorry

-- Defining the conditions for the second power
def x2 : ℝ := -0.004
def n2 : ℕ := 13

-- Statement for the second power problem
theorem power_two :
  (1 + x2)^n2 ≈ 0.9492 := by
  sorry

end power_one_power_two_l813_813337


namespace geom_sequence_terms_l813_813478

theorem geom_sequence_terms (a_n : ℕ → ℤ) (n : ℕ) 
  (h1 : a_n 1 + a_n n = 82) 
  (h2 : a_n 3 * a_n (n - 2) = 81) 
  (h3 : (range (n + 1)).sum (λ i, a_n i) = 121) : 
  n = 5 :=
sorry

end geom_sequence_terms_l813_813478


namespace solve_f_l813_813916

noncomputable def f (x : ℝ) : ℝ := sorry

-- The first condition: f(x + π) = f(x) + sin x
axiom ax1 : ∀ (x : ℝ), f(x + real.pi) = f(x) + real.sin x

-- The second condition: 0 ≤ x ≤ π → f(x) = 0
axiom ax2 : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ real.pi → f(x) = 0

-- The statement to prove: f(23π/6) = 1/2
theorem solve_f : f (23 * real.pi / 6) = 1 / 2 := by
  sorry

end solve_f_l813_813916


namespace quadrilateral_inequality_l813_813998

theorem quadrilateral_inequality
  (a1 a2 a3 a4 s : ℝ)
  (ha : a1 + a2 + a3 + a4 = 2 * s) :
  (1 / (a1 + s) + 1 / (a2 + s) + 1 / (a3 + s) + 1 / (a4 + s)) ≤
  (2 / 9) * (
    1 / real.sqrt ((s - a1) * (s - a2)) +
    1 / real.sqrt ((s - a1) * (s - a3)) +
    1 / real.sqrt ((s - a1) * (s - a4)) +
    1 / real.sqrt ((s - a2) * (s - a3)) +
    1 / real.sqrt ((s - a2) * (s - a4)) +
    1 / real.sqrt ((s - a3) * (s - a4))) :=
sorry

end quadrilateral_inequality_l813_813998


namespace water_percent_yield_is_90_l813_813360

def sulfuric_acid_moles : ℝ := 3
def sodium_hydroxide_moles : ℝ := 6
def water_experimental_yield : ℝ := 5.4
def water_theoretical_yield : ℝ := 2 * sulfuric_acid_moles

def percent_yield (experimental_yield theoretical_yield : ℝ) : ℝ :=
  (experimental_yield / theoretical_yield) * 100

theorem water_percent_yield_is_90 :
  percent_yield water_experimental_yield water_theoretical_yield = 90 :=
by
  sorry

end water_percent_yield_is_90_l813_813360


namespace equilateral_triangle_of_equal_areas_l813_813217

theorem equilateral_triangle_of_equal_areas (
    (A B C A1 B1 C1 A0 B0 C0 : Point) 
    (σ : Triangle) (circle : Circle) 
    (hA : lies_on A σ) (hB : lies_on B σ) (hC : lies_on C σ)
    (is_median_A : median A B C A1) (is_median_B : median B A C B1) (is_median_C : median C A B C1)
    (extended_median_A : (midpoint A A1) lies_on circle ∧ lies_on A0 circle)
    (extended_median_B : (midpoint B B1) lies_on circle ∧ lies_on B0 circle)
    (extended_median_C : (midpoint C C1) lies_on circle ∧ lies_on C0 circle)
    (equal_areas : area(B C0 A) = area(B0 C A) ∧ area(A0 B C) = area(B0 C A)) :
    
    is_equilateral_triangle A B C :=
sorry

end equilateral_triangle_of_equal_areas_l813_813217


namespace max_side_length_of_integer_triangle_with_perimeter_24_l813_813697

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l813_813697


namespace trapezoid_area_l813_813856

noncomputable def base1 := 4 -- meters
noncomputable def base2 := 7 -- meters
noncomputable def height := 5 -- meters

noncomputable def area_of_trapezoid (b1 b2 h : ℕ) : ℝ :=
  (1 / 2 : ℝ) * (b1 + b2) * h

theorem trapezoid_area : area_of_trapezoid base1 base2 height = 27.5 :=
  by
    sorry

end trapezoid_area_l813_813856


namespace max_side_of_triangle_l813_813748

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l813_813748


namespace max_side_length_is_11_l813_813674

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l813_813674


namespace extra_interest_is_correct_l813_813293

def principal : ℝ := 5000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

def interest1 : ℝ := simple_interest principal rate1 time
def interest2 : ℝ := simple_interest principal rate2 time

def extra_interest : ℝ := interest1 - interest2

theorem extra_interest_is_correct : extra_interest = 600 := by
  sorry

end extra_interest_is_correct_l813_813293


namespace ellipse_standard_equation_range_reciprocal_distances_l813_813421

theorem ellipse_standard_equation 
    (x y : ℝ)
    (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
    (M : ℝ × ℝ)
    (hF1 : F1 = (0, -Real.sqrt 3))
    (hF2 : F2 = (0, Real.sqrt 3))
    (hM : M = (Real.sqrt 3 / 2, 1))
    (hPassesThroughM : (M.1^2 + M.2^2 / 4 = 1)) :
    (y^2 / 4 + x^2 = 1) :=
sorry

theorem range_reciprocal_distances 
    (P : ℝ × ℝ)
    (F1 : ℝ × ℝ)
    (F2 : ℝ × ℝ)
    (hF1 : F1 = (0, -Real.sqrt 3))
    (hF2 : F2 = (0, Real.sqrt 3))
    (hP_on_ellipse : P.1^2 + P.2^2 / 4 = 1) :
    (1 ≤ 1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 + 1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) 
    ∧ 1 / Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 + 1 / Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) ≤ 4) :=
sorry

end ellipse_standard_equation_range_reciprocal_distances_l813_813421


namespace max_triangle_side_24_l813_813726

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l813_813726


namespace max_side_of_triangle_l813_813802

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l813_813802


namespace expenditure_proof_l813_813231

namespace OreoCookieProblem

variables (O C : ℕ) (CO CC : ℕ → ℕ) (total_items cost_difference : ℤ)

def oreo_count_eq : Prop := O = (4 * (65 : ℤ) / 13)
def cookie_count_eq : Prop := C = (9 * (65 : ℤ) / 13)
def oreo_cost (o : ℕ) : ℕ := o * 2
def cookie_cost (c : ℕ) : ℕ := c * 3
def total_item_condition : Prop := O + C = 65
def ratio_condition : Prop := 9 * O = 4 * C
def cost_difference_condition (o_cost c_cost : ℕ) : Prop := cost_difference = (c_cost - o_cost)

theorem expenditure_proof :
  (O + C = 65) →
  (9 * O = 4 * C) →
  (O = 20) →
  (C = 45) →
  cost_difference = (45 * 3 - 20 * 2) →
  cost_difference = 95 :=
by sorry

end OreoCookieProblem

end expenditure_proof_l813_813231


namespace max_area_CDFE_l813_813496

noncomputable def quadrilateral_area (x : ℝ) : ℝ :=
  0.5 * (2 - x) * x + 0.5 * x^2 + 0.5 * (2 - x / 3) * (x / 3)

theorem max_area_CDFE : ∃ x, AE = x ∧ AF = x / 3 ∧ quadrilateral_area x = 22 / 9 :=
begin
  sorry
end

end max_area_CDFE_l813_813496


namespace lucille_remaining_cents_l813_813165

def earnings_per_weed : ℕ → ℕ
| 0 := 4  -- small weed
| 1 := 8  -- medium weed
| 2 := 12 -- large weed
| _ := 0  -- default (should not reach)

def weeds_in_area : ℕ → ℕ × ℕ × ℕ
| 0 := (6, 3, 2)  -- flower bed
| 1 := (10, 2, 2) -- vegetable patch
| 2 := (20 / 2, 10 / 2, 2 / 2) -- half grass
| _ := (0, 0, 0)  -- default (should not reach)

def soda_cost : ℕ := 99
def soda_tax : ℕ := 15

def total_earnings : ℕ :=
  let flower_bed := (weeds_in_area 0).1 * earnings_per_weed 0
                    + (weeds_in_area 0).2 * earnings_per_weed 1
                    + (weeds_in_area 0).3 * earnings_per_weed 2 in
  let vegetable_patch := (weeds_in_area 1).1 * earnings_per_weed 0
                         + (weeds_in_area 1).2 * earnings_per_weed 1
                         + (weeds_in_area 1).3 * earnings_per_weed 2 in
  let half_grass := (weeds_in_area 2).1 * earnings_per_weed 0
                    + (weeds_in_area 2).2 * earnings_per_weed 1
                    + (weeds_in_area 2).3 * earnings_per_weed 2 in
  flower_bed + vegetable_patch + half_grass

def total_soda_cost : ℕ :=
  soda_cost + (soda_cost * soda_tax) / 100

def remaining_cents (earnings : ℕ) (cost : ℕ) : ℕ :=
  earnings - cost

theorem lucille_remaining_cents : remaining_cents total_earnings total_soda_cost = 130 :=
by
  rw [remaining_cents, total_earnings, total_soda_cost]
  -- Compute specific terms if necessary, otherwise:
  sorry

end lucille_remaining_cents_l813_813165


namespace find_k_values_for_intersection_l813_813113

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l813_813113


namespace flag_arrangement_l813_813602

noncomputable def arrange_flags : ℕ :=
  let total_reds := 12
  let total_yellows := 11
  let possible_positions := 13  -- positions around and between 12 red flags
  let valid_arrangements := Nat.choose possible_positions total_yellows * total_reds 
  valid_arrangements

theorem flag_arrangement:
  let N := arrange_flags in
  N % 1000 = 858 :=
by
  sorry

end flag_arrangement_l813_813602


namespace age_of_b_l813_813278

theorem age_of_b (a b : ℕ) 
(h1 : a + 10 = 2 * (b - 10)) 
(h2 : a = b + 4) : 
b = 34 := 
sorry

end age_of_b_l813_813278


namespace percentage_increase_l813_813344

-- Defining the problem constants
def price (P : ℝ) : ℝ := P
def assets_A (A : ℝ) : ℝ := A
def assets_B (B : ℝ) : ℝ := B
def percentage (X : ℝ) : ℝ := X

-- Conditions
axiom price_company_B_double_assets : ∀ (P B: ℝ), price P = 2 * assets_B B
axiom price_seventy_five_percent_combined_assets : ∀ (P A B: ℝ), price P = 0.75 * (assets_A A + assets_B B)
axiom price_percentage_more_than_A : ∀ (P A X: ℝ), price P = assets_A A * (1 + percentage X / 100)

-- Theorem to prove
theorem percentage_increase : ∀ (P A B X : ℝ)
  (h1 : price P = 2 * assets_B B)
  (h2 : price P = 0.75 * (assets_A A + assets_B B))
  (h3 : price P = assets_A A * (1 + percentage X / 100)),
  percentage X = 20 :=
by
  intros P A B X h1 h2 h3
  -- Proof steps would go here
  sorry

end percentage_increase_l813_813344


namespace range_a_l813_813435

-- Define the function P(x, a)
def P (x a : ℝ) : ℝ :=
  (x^2 + (2*a^2 + 2)*x - a^2 + 4*a - 7) / (x^2 + (a^2 + 4*a - 5)*x - a^2 + 4*a - 7)

-- Prove the main theorem
theorem range_a (a : ℝ) :
  (∀ x t : ℝ, t = (x^2 + (2*a^2 + 2)*x - a^2 + 4*a - 7) / (x^2 + (a^2 + 4*a - 5)*x - a^2 + 4*a - 7) → t < 0 → (x = x.min ∨ x = x.max) ) →
  (∑ x in (range x), x.length) < 4 → 
  1 ≤ a ∧ a ≤ 3 := 
sorry

end range_a_l813_813435


namespace min_sum_ab_l813_813407

theorem min_sum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (2 / b) = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_sum_ab_l813_813407


namespace intersecting_circles_l813_813093

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l813_813093


namespace find_k_values_for_intersection_l813_813115

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l813_813115


namespace solution_l813_813156

noncomputable def digit_product (x : ℕ) : ℕ :=
  (toDigits 10 x).prod

theorem solution : ∃ x : ℕ, x > 0 ∧ digit_product x = x^2 - 10 * x - 22 ∧ x = 12 :=
by
  sorry

end solution_l813_813156


namespace total_days_spent_l813_813132

theorem total_days_spent {weeks_to_days : ℕ → ℕ} : 
  (weeks_to_days 3 + weeks_to_days 1) + 
  (weeks_to_days (weeks_to_days 3 + weeks_to_days 2) + 3) + 
  (2 * (weeks_to_days (weeks_to_days 3 + weeks_to_days 2))) + 
  (weeks_to_days 5 - weeks_to_days 1) + 
  (weeks_to_days ((weeks_to_days 5 - weeks_to_days 1) - weeks_to_days 3) + 6) + 
  (weeks_to_days (weeks_to_days 5 - weeks_to_days 1) + 4) = 230 :=
by
  sorry

end total_days_spent_l813_813132


namespace min_value_of_reciprocal_sum_l813_813008

variable (m n : ℝ)

theorem min_value_of_reciprocal_sum (hmn : m * n > 0) (h_line : m + n = 2) :
  (1 / m + 1 / n = 2) :=
sorry

end min_value_of_reciprocal_sum_l813_813008


namespace m_gt_n_l813_813023

noncomputable def m : ℕ := 2015 ^ 2016
noncomputable def n : ℕ := 2016 ^ 2015

theorem m_gt_n : m > n := by
  sorry

end m_gt_n_l813_813023


namespace count_3_digit_numbers_divisible_by_13_l813_813054

theorem count_3_digit_numbers_divisible_by_13 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n < 1000}
      divis_by_13 := {n : ℕ | n % 13 = 0}
      three_digit_divis_by_13 := three_digit_numbers ∩ divis_by_13 in 
  three_digit_divis_by_13.card = 69 :=
by sorry

end count_3_digit_numbers_divisible_by_13_l813_813054


namespace six_digit_number_l813_813311

noncomputable def number_of_digits (N : ℕ) : ℕ := sorry

theorem six_digit_number :
  ∀ (N : ℕ),
    (N % 2020 = 0) ∧
    (∀ a b : ℕ, (a ≠ b ∧ N / 10^a % 10 ≠ N / 10^b % 10)) ∧
    (∀ a b : ℕ, (a ≠ b) → ((N / 10^a % 10 = N / 10^b % 10) -> (N % 2020 ≠ 0))) →
    number_of_digits N = 6 :=
sorry

end six_digit_number_l813_813311


namespace proof_for_W_l813_813125

noncomputable def solve_for_W : ℕ :=
  let T := 8
  let valid_O := [1, 3, 5, 7, 9].filter (λ O, let E := (2 * O) % 10 in E % 2 = 0)
  let (O, E) := match valid_O with
    | [] => (0, 0) -- should never happen, just to exhaust pattern match
    | O::O_tail => (O, (2 * O) % 10) -- choose the first valid pair
  let W := ((F := 1) (I := 5) (V := 5) 2 -- Solutions compliant with the problem
  W

theorem proof_for_W : solve_for_W = 2 :=
by
  sorry

end proof_for_W_l813_813125


namespace divide_circle_identical_triangles_l813_813241

-- Define the conditions
noncomputable def number_of_points := 6000
noncomputable def max_distance := 9
noncomputable def number_of_regions := 2000
noncomputable def num_identical_triangles := 22

-- Define the problem statements
theorem divide_circle (h₁: ∀ (points : Finset ℝ), points.card = number_of_points ∧ (∀ p1 p2 p3 : ℝ, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ ((p1, p2, p3) are_collinear))) :
  ∃ (regions : Finset (Finset ℝ)), regions.card = number_of_regions ∧ (∀ region ∈ regions, region.card = 3 ∧ ∀ (p1 p2 : ℝ), abs (p1 - p2) ≤ max_distance) :=
sorry

theorem identical_triangles :
  ∃ (n : ℕ), n ≥ num_identical_triangles :=
sorry

end divide_circle_identical_triangles_l813_813241


namespace solution_valid_l813_813061

-- Define the conditions
variables (a b c : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variables (h4 : a * b = 24) (h5 : a * c = 40) (h6 : b * c = 60)

noncomputable def determine_abc : ℝ :=
  let abc := a * b * c in abc

theorem solution_valid :
  determine_abc a b c = 240 :=
by
  -- Provide the proof here
  sorry

end solution_valid_l813_813061


namespace decreasing_a_geq_3_l813_813036

def is_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x ≤ y → f y ≤ f x

theorem decreasing_a_geq_3 {a : ℝ} :
  is_decreasing (λ x : ℝ, x^2 - 2*a*x + 6) (set.Iic 3) → a ≥ 3 :=
sorry

end decreasing_a_geq_3_l813_813036


namespace max_side_length_l813_813768

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l813_813768


namespace g_zero_l813_813498

variable (f g h : Polynomial ℤ) -- Assume f, g, h are polynomials over the integers

-- Condition: h(x) = f(x) * g(x)
axiom h_def : h = f * g

-- Condition: The constant term of f(x) is 2
axiom f_const : f.coeff 0 = 2

-- Condition: The constant term of h(x) is -6
axiom h_const : h.coeff 0 = -6

-- Proof statement that g(0) = -3
theorem g_zero : g.coeff 0 = -3 := by
  sorry

end g_zero_l813_813498


namespace value_of_a_l813_813911

theorem value_of_a (a : ℤ) : (∃ x : ℤ, x = 5 ∧ 3 * x - 2 * a = 7) → a = 4 :=
by {
  intro h,
  cases h with x hx,
  cases hx with h1 h2,
  sorry
}

end value_of_a_l813_813911


namespace geometric_sequence_sum_l813_813481

theorem geometric_sequence_sum {a : ℕ → ℝ} (h : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) 
  (h_cond : (1 / (a 2 * a 4)) + (2 / (a 4 * a 4)) + (1 / (a 4 * a 6)) = 81) :
  (1 / a 3) + (1 / a 5) = 9 :=
sorry

end geometric_sequence_sum_l813_813481


namespace trains_meet_in_10_point_57_seconds_l813_813257

/-
Given:
1. Length of the first train: l1 = 100 meters
2. Length of the second train: l2 = 200 meters
3. Initial distance apart: d = 70 meters
4. Speed of the first train: v1 = 54 kilometers per hour
5. Speed of the second train: v2 = 72 kilometers per hour

Prove:
The time t that it takes for the trains to meet is approximately 10.57 seconds.
-/

noncomputable def length_first_train : ℝ := 100
noncomputable def length_second_train : ℝ := 200
noncomputable def initial_distance_apart : ℝ := 70
noncomputable def speed_first_train_kmph : ℝ := 54
noncomputable def speed_second_train_kmph : ℝ := 72

noncomputable def kmph_to_mps (v_kmph : ℝ) : ℝ := v_kmph * 1000 / 3600
noncomputable def speed_first_train_mps : ℝ := kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train_mps : ℝ := kmph_to_mps speed_second_train_kmph

noncomputable def relative_speed : ℝ := speed_first_train_mps + speed_second_train_mps
noncomputable def total_distance : ℝ := length_first_train + length_second_train + initial_distance_apart

noncomputable def time_to_meet : ℝ := total_distance / relative_speed

theorem trains_meet_in_10_point_57_seconds : time_to_meet ≈ 10.57 := sorry

end trains_meet_in_10_point_57_seconds_l813_813257


namespace find_number_of_children_l813_813164

-- Definitions based on conditions
def decorative_spoons : Nat := 2
def new_set_large_spoons : Nat := 10
def new_set_tea_spoons : Nat := 15
def total_spoons : Nat := 39
def spoons_per_child : Nat := 3
def new_set_spoons := new_set_large_spoons + new_set_tea_spoons

-- The main statement to prove the number of children
theorem find_number_of_children (C : Nat) :
  3 * C + decorative_spoons + new_set_spoons = total_spoons → C = 4 :=
by
  -- Proof would go here
  sorry

end find_number_of_children_l813_813164


namespace food_suggestions_ratio_l813_813542

def students_suggested_sushi : ℕ := 297
def students_suggested_mashed_potatoes : ℕ := 144
def students_suggested_bacon : ℕ := 467
def students_suggested_tomatoes : ℕ := 79

theorem food_suggestions_ratio :
  (students_suggested_sushi:nat) : (students_suggested_mashed_potatoes:nat) : (students_suggested_bacon:nat) : (students_suggested_tomatoes:nat) = 297 : 144 : 467 : 79 :=
by
  sorry

end food_suggestions_ratio_l813_813542


namespace find_max_side_length_l813_813785

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l813_813785


namespace radian_to_degree_conversion_l813_813853

theorem radian_to_degree_conversion
: (π : ℝ) = 180 → ((-23 / 12) * π) = -345 :=
by
  sorry

end radian_to_degree_conversion_l813_813853


namespace problem_1_problem_2_l813_813546

noncomputable def problem_1_solution : Set ℝ := {6, -2}
noncomputable def problem_2_solution : Set ℝ := {2 + Real.sqrt 7, 2 - Real.sqrt 7}

theorem problem_1 :
  {x : ℝ | x^2 - 4 * x - 12 = 0} = problem_1_solution :=
by
  sorry

theorem problem_2 :
  {x : ℝ | x^2 - 4 * x - 3 = 0} = problem_2_solution :=
by
  sorry

end problem_1_problem_2_l813_813546


namespace OD_expression_in_terms_of_sine_l813_813294

variables (O A B D : Type) [ordered_ring O] [ordered_ring A] [ordered_ring B] [ordered_ring D]
variables (OA : O) (OB : O) (AB : O)
variables (radius : ℝ) (phi : ℝ) (OD : ℝ)

-- Using given conditions as assumptions

-- Circle centered at O with radius 2
axiom circle_centered_at_O_has_radius_2 : radius = 2
-- Point A lies on the circumference
axiom point_A_on_circumference : (OA = radius)
-- Segment AB is tangent to the circle at A
axiom AB_tangent_at_A : true
-- Angle AOB is phi
axiom angle_AOB_is_phi : true
-- Point D lies on OA such that BD bisects angle ABO
axiom segment_BD_bisects_angle_ABO : true
  
theorem OD_expression_in_terms_of_sine
  (phi : ℝ) (r : ℝ) (h1 : r = 2) 
  : OD = 2 / (1 + sin phi) :=
sorry

end OD_expression_in_terms_of_sine_l813_813294


namespace find_e_l813_813574

theorem find_e (d e f : ℤ) (Q : ℤ → ℤ) (hQ : ∀ x, Q x = 3 * x^3 + d * x^2 + e * x + f)
  (mean_zeros_eq_prod_zeros : let zeros := {x // Q x = 0} in
    (∑ x in zeros, x) / 3 = ∏ x in zeros, x)
  (sum_coeff_eq_mean_zeros : 3 + d + e + f = (∑ x in {x // Q x = 0}, x) / 3)
  (y_intercept : Q 0 = 9) :
  e = -42 :=
sorry

end find_e_l813_813574
