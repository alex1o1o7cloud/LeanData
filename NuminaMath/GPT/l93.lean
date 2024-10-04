import Mathlib
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Interval
import Mathlib.Combinatorics.Pigeonhole
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Geometry.Triangle.IncenterExcenter
import Mathlib.MeasureTheory.Pigeonhole
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.RingTheory.Fractions
import Mathlib.Tactic

namespace line_through_two_points_l93_93275

theorem line_through_two_points (A B : ℝ × ℝ)
  (hA : A = (2, -3))
  (hB : B = (1, 4)) :
  ∃ (m b : ℝ), (∀ x y : ℝ, (y = m * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ m = -7 ∧ b = 11 := by
  sorry

end line_through_two_points_l93_93275


namespace simplify_expression_l93_93860

-- Define the conditions
def sqrt_648 : ℝ := real.sqrt 648
def sqrt_81 : ℝ := real.sqrt 81
def sqrt_294 : ℝ := real.sqrt 294
def sqrt_49 : ℝ := real.sqrt 49

-- Define the given expression
def expr : ℝ := (sqrt_648 / sqrt_81) - (sqrt_294 / sqrt_49)

-- Provide the correct simplified form
def correct_answer : ℝ := 2 * real.sqrt 2 - real.sqrt 42

-- State the theorem to be proved
theorem simplify_expression : expr = correct_answer := by
  sorry

end simplify_expression_l93_93860


namespace find_a_l93_93228

noncomputable def f : ℝ → ℝ := λ x, if x ≤ -1 then -x - 4 else x^2 - 5

theorem find_a (a : ℝ) : f a - 11 = 0 → (a = -15 ∨ a = 4) :=
by
  sorry

end find_a_l93_93228


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93077

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93077


namespace find_matrix_A_l93_93596

-- Let A be a 2x2 matrix such that 
def A (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

theorem find_matrix_A :
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ,
  (A.mulVec ![4, 1] = ![8, 14]) ∧ (A.mulVec ![2, -3] = ![-2, 11]) ∧
  A = ![![2, 1/2], ![-1, -13/3]] :=
by
  sorry

end find_matrix_A_l93_93596


namespace no_solutions_to_inequality_l93_93172

theorem no_solutions_to_inequality (x : ℝ) : ¬(3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  intro h,
  -- Simplify the inequality by dividing each term by 3
  have h_simplified : x^2 + 3 * x + 4 ≤ 0 := by linarith,
  -- Compute the discriminant of the quadratic expression to show it's always positive
  let a := (1 : ℝ),
  let b := (3 : ℝ),
  let c := (4 : ℝ),
  let discriminant := b^2 - 4 * a * c,
  have h_discriminant : discriminant < 0 := by norm_num,
  -- Since discriminant is negative, the quadratic has no real roots, thus x^2 + 3x + 4 > 0
  have h_positive : ∀ x, x^2 + 3 * x + 4 > 0 := 
    by {
      intro x,
      apply (quadratic_not_negative_of_discriminant neg_discriminant).mp,
      exact h_discriminant,
    },
  exact absurd (show x^2 + 3 * x + 4 ≤ 0 from h_simplified) (lt_irrefl 0 (h_positive x)),
}

end no_solutions_to_inequality_l93_93172


namespace range_a_l93_93636

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_a (f_even : ∀ x, f x = f (-x))
  (f_mono_dec : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
  (h : ∀ a : ℝ, f (log 3 a) + f (log (1 / 3) a) ≤ 2 * f 2) :
  ∀ a : ℝ, (0 < a ∧ a ≤ 1 / 9) ∨ (9 ≤ a) := 
by
  sorry

end range_a_l93_93636


namespace price_system_of_equations_l93_93509

-- Define the price of basketball and soccer ball in terms of x and y
variables (x y : ℝ)

-- The conditions stated in the problem
def condition1 : Prop := 3 * x + 4 * y = 330
def condition2 : Prop := x = y - 5

-- The system of equations to be proved
theorem price_system_of_equations : condition1 → condition2 → (3 * x + 4 * y = 330 ∧ x = y - 5) :=
by
  intros h₁ h₂
  exact ⟨h₁, h₂⟩

end price_system_of_equations_l93_93509


namespace nice_positive_integers_count_l93_93842

def is_nice (n m : ℕ) : Prop := n^3 < 5 * m * n ∧ 5 * m * n < n^3 + 100

def count_nice_numbers : ℕ :=
  { n : ℕ | ∃ m : ℕ, is_nice n m }.to_finset.card

theorem nice_positive_integers_count :
  count_nice_numbers = 53 := 
sorry

end nice_positive_integers_count_l93_93842


namespace randy_trips_l93_93374

def trips_per_month
  (initial : ℕ) -- Randy initially had $200 in his piggy bank
  (final : ℕ)   -- Randy had $104 left in his piggy bank after a year
  (spend_per_trip : ℕ) -- Randy spends $2 every time he goes to the store
  (months_in_year : ℕ) -- Number of months in a year, which is 12
  (total_trips_per_year : ℕ) -- Total trips he makes in a year
  (trips_per_month : ℕ) -- Trips to the store every month
  : Prop :=
  initial = 200 ∧ final = 104 ∧ spend_per_trip = 2 ∧ months_in_year = 12 ∧
  total_trips_per_year = (initial - final) / spend_per_trip ∧ 
  trips_per_month = total_trips_per_year / months_in_year ∧
  trips_per_month = 4

theorem randy_trips :
  trips_per_month 200 104 2 12 ((200 - 104) / 2) (48 / 12) :=
by 
  sorry

end randy_trips_l93_93374


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93066

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93066


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93750

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93750


namespace initial_goldfish_correct_l93_93355

-- Define the constants related to the conditions
def weekly_die := 5
def weekly_purchase := 3
def final_goldfish := 4
def weeks := 7

-- Define the initial number of goldfish that we need to prove
def initial_goldfish := 18

-- The proof statement: initial_goldfish - weekly_change * weeks = final_goldfish
theorem initial_goldfish_correct (G : ℕ)
  (h : G - weeks * (weekly_purchase - weekly_die) = final_goldfish) :
  G = initial_goldfish := by
  sorry

end initial_goldfish_correct_l93_93355


namespace smallest_composite_no_prime_factors_less_than_20_l93_93145

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93145


namespace total_field_goals_attempted_l93_93910

namespace WarioFieldGoals

def field_goals := ℕ

-- Conditions
def misses_one_fourth_of_attempts (total_attempts missed_goals : field_goals) : Prop :=
  missed_goals = total_attempts / 4

def twenty_percent_wide_right (missed_goals wide_right_misses : field_goals) : Prop :=
  wide_right_misses = missed_goals / 5  -- 20% is the same as 1/5

def three_wide_right (wide_right_misses : field_goals) : Prop :=
  wide_right_misses = 3

-- Proof we need to show
theorem total_field_goals_attempted : ∃ (total_attempts : field_goals), 
  ∀ (missed_goals wide_right_misses : field_goals),
  misses_one_fourth_of_attempts total_attempts missed_goals ∧
  twenty_percent_wide_right missed_goals wide_right_misses ∧
  three_wide_right wide_right_misses → 
  total_attempts = 60 := sorry

end WarioFieldGoals

end total_field_goals_attempted_l93_93910


namespace value_of_expression_l93_93452

theorem value_of_expression (y : ℕ) (h : y = 50) :
  (y^3 + 3 * y^2 * (2 * y) + 3 * y * (2 * y)^2 + (2 * y)^3) = 3375000 :=
by {
  rw h,
  sorry
}

end value_of_expression_l93_93452


namespace cost_of_fencing_proof_l93_93943

noncomputable def cost_of_fencing (area : ℝ) (ratio : ℝ × ℝ) (cost_per_meter : ℝ) : ℝ :=
  let length := 3 * real.sqrt (area / 6)
  let width := 2 * real.sqrt (area / 6)
  let perimeter := 2 * (length + width)
  perimeter * cost_per_meter

theorem cost_of_fencing_proof : cost_of_fencing 2400 (3, 2) 0.50 = 100 :=
by
  -- simplified calculation using definitions
  sorry

end cost_of_fencing_proof_l93_93943


namespace prob_5_shots_expected_number_shots_l93_93958

variable (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

def prob_exactly_five_shots : ℝ := 6 * p^3 * (1 - p)^2
def expected_shots : ℝ := 3 / p

theorem prob_5_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the probability of exactly 5 shots needed is as calculated
  prob_exactly_five_shots p = 6 * p^3 * (1 - p)^2 :=
by
  sorry

theorem expected_number_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the expected number of shots to hit all targets is as calculated
  expected_shots p = 3 / p :=
by
  sorry

end prob_5_shots_expected_number_shots_l93_93958


namespace count_whole_numbers_between_roots_l93_93758

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93758


namespace salad_cucumbers_l93_93542

theorem salad_cucumbers (c t : ℕ) 
  (h1 : c + t = 280)
  (h2 : t = 3 * c) : c = 70 :=
sorry

end salad_cucumbers_l93_93542


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93089

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93089


namespace max_xy_value_l93_93764

theorem max_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 200) : x * y ≤ 28 :=
begin
  sorry
end

example : ∃ x y : ℕ, 7 * x + 4 * y = 200 ∧ x * y = 28 := 
begin
  use [28, 1],
  split,
  { norm_num, },
  { norm_num, },
end

end max_xy_value_l93_93764


namespace correct_technology_used_l93_93868

-- Define the condition that the program title is "Back to the Dinosaur Era"
def program_title : String := "Back to the Dinosaur Era"

-- Define the condition that the program vividly recreated various dinosaurs and their living environments
def recreated_living_environments : Bool := true

-- Define the options for digital Earth technologies
inductive DigitalEarthTechnology
| InformationSuperhighway
| HighResolutionSatelliteTechnology
| SpatialInformationTechnology
| VisualizationAndVirtualRealityTechnology

-- Define the correct answer
def correct_technology := DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology

-- The proof problem: Prove that given the conditions, the technology used is the correct one
theorem correct_technology_used
  (title : program_title = "Back to the Dinosaur Era")
  (recreated : recreated_living_environments) :
  correct_technology = DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology :=
by
  sorry

end correct_technology_used_l93_93868


namespace car_distances_equal_600_l93_93887

-- Define the variables
def time_R (t : ℝ) := t
def speed_R := 50
def time_P (t : ℝ) := t - 2
def speed_P := speed_R + 10
def distance (t : ℝ) := speed_R * time_R t

-- The Lean theorem statement
theorem car_distances_equal_600 (t : ℝ) (h : time_R t = t) (h1 : speed_R = 50) 
  (h2 : time_P t = t - 2) (h3 : speed_P = speed_R + 10) :
  distance t = 600 :=
by
  -- We would provide the proof here, but for now we use sorry to indicate the proof is omitted.
  sorry

end car_distances_equal_600_l93_93887


namespace intersection_A_B_l93_93203

noncomputable def A : Set ℝ := {x | 9 * x ^ 2 < 1}

noncomputable def B : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 2 * x + 5 / 4}

theorem intersection_A_B :
  (A ∩ B) = {y | y ∈ Set.Ico (1/4 : ℝ) (1/3 : ℝ)} :=
by
  sorry

end intersection_A_B_l93_93203


namespace vector_addition_simplification_l93_93382

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_simplification
  (AB BC AC DC CD : V)
  (h1 : AB + BC = AC)
  (h2 : - DC = CD) :
  AB + BC - AC - DC = CD :=
by
  -- Placeholder for the proof
  sorry

end vector_addition_simplification_l93_93382


namespace correct_propositions_count_l93_93306

noncomputable def manhattan_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

def prop1 (P : ℝ × ℝ) : Prop :=
  manhattan_distance P (0, 0) = 1

def set_of_distance_1 : set (ℝ × ℝ) :=
  { P | prop1 P }

def prop3 (P : ℝ × ℝ) : Prop :=
  manhattan_distance P (-1, 0) = manhattan_distance P (1, 0)

def prop4 (P : ℝ × ℝ) : Prop :=
  |manhattan_distance P (-1, 0) - manhattan_distance P (1, 0)| = 1

theorem correct_propositions_count : 
  (({ P | prop1 P } = {(x, y) | |x| + |y| = 1}) ∧ 
   ({ P | prop3 P } = { (0, y) | y ∈ ℝ }) ∧ 
   ({ P | prop4 P } = { (x, y) | x = 1 ∨ x = -1 })) → 
  3 = 3 := 
by 
  sorry

end correct_propositions_count_l93_93306


namespace relationship_p_q_l93_93608

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem relationship_p_q (x a p q : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1)
  (hp : p = |log_a a (1 + x)|) (hq : q = |log_a a (1 - x)|) : p ≤ q :=
sorry

end relationship_p_q_l93_93608


namespace sum_of_coefficients_l93_93880

noncomputable def polynomial (x : ℝ) : ℝ := x^3 + 3*x^2 - 4*x - 12
noncomputable def simplified_polynomial (x : ℝ) (A B C : ℝ) : ℝ := A*x^2 + B*x + C

theorem sum_of_coefficients : 
  ∃ (A B C D : ℝ), 
    (∀ x ≠ D, simplified_polynomial x A B C = (polynomial x) / (x + 3)) ∧ 
    (A + B + C + D = -6) :=
by
  sorry

end sum_of_coefficients_l93_93880


namespace cos_390_eq_sqrt3_div_2_l93_93497

theorem cos_390_eq_sqrt3_div_2 : real.cos (390 * real.pi / 180) = sqrt 3 / 2 :=
by
  have h1 : 390 * real.pi / 180 = 2 * real.pi + 30 * real.pi / 180 := by sorry
  rw [h1, real.cos_add_two_pi, real.cos]
  have h2 : real.pi / 6 = 30 * real.pi / 180 := by sorry
  rw [h2, real.cos_pi_div_six]
  exact sqrt_three_div_two

lemma real.cos_add_two_pi (x : ℝ) : real.cos (x + 2 * real.pi) = real.cos x := by sorry
lemma real.cos_pi_div_six : real.cos (real.pi / 6) = sqrt 3 / 2 := by sorry
lemma sqrt_three_div_two : sqrt 3 / 2 = sqrt 3 / 2 := by sorry

end cos_390_eq_sqrt3_div_2_l93_93497


namespace find_floor_abs_sum_l93_93863

noncomputable def floor_abs_sum (x : ℕ → ℤ) (S : ℤ) :=
  ∑ n in Finset.range 100, x n

theorem find_floor_abs_sum :
  (∀ n : ℕ, 1 ≤ n → n ≤ 100 → x n + (n + 1) = ∑ n in Finset.range 100, x n + 102) →
  abs (∑ n in Finset.range 100, x n) = 5150 / 99 →
  int.floor (abs (∑ n in Finset.range 100, x n)) = 52 :=
begin
  sorry
end

end find_floor_abs_sum_l93_93863


namespace count_whole_numbers_between_roots_l93_93759

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93759


namespace invariant_line_BK_for_any_M_on_BC_l93_93561

variables {A B C M O K : Type} [Geometry A B C M O K]

-- Assume the necessary conditions hold:
variables (BC_non_collinear : ¬ collinear [B, C, A])
          (point_M_on_BC : M ∈ line_segment B C)
          (O_center_ABM : O = circumcenter A B M)
          (circle_k : ∃ (center : Point), center ∈ line_segment B C ∧ ∀ x, x ∈ circle center A ↔ x = A ∨ x = M)

-- Prove that the line BK is invariant for any choice of point M on segment BC
theorem invariant_line_BK_for_any_M_on_BC :
  BK_invariant (circle_k.center) :=
by
  sorry

end invariant_line_BK_for_any_M_on_BC_l93_93561


namespace number_of_distinct_paths_in_grid_l93_93283

/-- Number of distinct paths from point A to B in a 7x4 grid avoiding two specific forbidden segments,
given that only south and east movements are allowed. -/
theorem number_of_distinct_paths_in_grid :
  let grid_width := 7
  let grid_height := 4
  let forbidden_segments := [(3, 2), (5, 3)] -- Pair (Column, Row) for forbidden segments
  (number_of_distinct_paths grid_width grid_height forbidden_segments) = 64 :=
sorry

noncomputable def number_of_distinct_paths (width height : ℕ) (forbidden : list (ℕ × ℕ)) : ℕ :=
-- Placeholder definition for the number of distinct paths function.
0 -- This would be properly defined with further implementation details.

end number_of_distinct_paths_in_grid_l93_93283


namespace max_power_speed_l93_93399

noncomputable def force (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v)^2) / 2

def power (B S ρ v₀ v : ℝ) : ℝ :=
  force B S ρ v₀ v * v

theorem max_power_speed (B ρ : ℝ) (S : ℝ := 7) (v₀ : ℝ := 6.3) :
  ∃ v, v = 2.1 ∧ (∀ v', power B S ρ v₀ v' ≤ power B S ρ v₀ v) :=
begin
  sorry,
end

end max_power_speed_l93_93399


namespace volume_of_cylinder_is_correct_l93_93189

-- Definitions corresponding to the conditions in the problem
def side_length_square_base := Real.sqrt 2
def lateral_edge_length := Real.sqrt 5
def diagonal_of_square_base := 2
def height_of_pyramid := 2
def diameter_upper_base := 1
def radius_upper_base := 1 / 2
def height_of_cylinder := 1

-- Volume of the cylinder
def volume_of_cylinder := Real.pi * (radius_upper_base ^ 2) * height_of_cylinder

-- Theorem statement to prove the volume of the cylinder
theorem volume_of_cylinder_is_correct : volume_of_cylinder = Real.pi / 4 := by
  sorry

end volume_of_cylinder_is_correct_l93_93189


namespace B_finishes_remaining_work_in_3_days_l93_93505

-- Definitions related to the problem conditions
def work_per_day_A : ℚ := 1 / 4
def work_per_day_B : ℚ := 1 / 10
def work_together_days : ℚ := 2
def total_work : ℚ := 1

-- The statement to prove
theorem B_finishes_remaining_work_in_3_days :
  let combined_work := work_together_days * (work_per_day_A + work_per_day_B) in
  let remaining_work := total_work - combined_work in
  let days_B_to_finish := remaining_work / work_per_day_B in
  days_B_to_finish = 3 :=
by
  -- Proof is not required, hence adding sorry
  sorry

end B_finishes_remaining_work_in_3_days_l93_93505


namespace mia_has_largest_final_value_l93_93564

def daniel_final : ℕ := (12 * 2 - 3 + 5)
def mia_final : ℕ := ((15 - 2) * 2 + 3)
def carlos_final : ℕ := (13 * 2 - 4 + 6)

theorem mia_has_largest_final_value : mia_final > daniel_final ∧ mia_final > carlos_final := by
  sorry

end mia_has_largest_final_value_l93_93564


namespace max_value_of_quadratic_on_interval_l93_93882

theorem max_value_of_quadratic_on_interval :
  ∃ (x : ℝ), (x ∈ set.Icc (-4 : ℝ) 3) ∧ (∀ y ∈ set.Icc (-4 : ℝ) 3, (y^2 + 2*y) ≤ (x^2 + 2*x)) ∧ ((x^2 + 2*x) = 15) :=
by {
  sorry
}

end max_value_of_quadratic_on_interval_l93_93882


namespace odd_number_of_vertices_of_tree_l93_93392

-- Define eccentricity of a vertex
def eccentricity (G : Type) [Finite G] [EdgeGraph G] (v : G) : ℕ :=
  ∑ u in Finset.univ, distance v u

-- Define condition that two vertices have eccentricities that differ by 1
def diff_by_one (G : Type) [Finite G] [EdgeGraph G] (v1 v2 : G) : Prop :=
  eccentricity G v1 = eccentricity G v2 + 1 ∨ eccentricity G v2 = eccentricity G v1 + 1

-- Define the theorem
theorem odd_number_of_vertices_of_tree {G : Type} [Finite G] [Tree G] (v1 v2 : G) 
  (h : diff_by_one G v1 v2) : (Finset.univ.card : ℕ) % 2 = 1 := 
  sorry

end odd_number_of_vertices_of_tree_l93_93392


namespace greatest_imaginary_part_of_2w3_l93_93643

theorem greatest_imaginary_part_of_2w3 :
  let w₁ := (-3 : ℂ)
  let w₂ := (-2*Real.sqrt 2 + (1:ℂ) * Complex.I)
  let w₃ := (-2 + (Real.sqrt 3) * Complex.I)
  let w₄ := (-2 + 2 * Complex.I)
  let w₅ := (3 * Complex.I)
  let imgs :=
    {Complex.imaginaryPart (2 * w₁ ^ 3),
     Complex.imaginaryPart (2 * w₂ ^ 3),
     Complex.imaginaryPart (2 * w₃ ^ 3),
     Complex.imaginaryPart (2 * w₄ ^ 3),
     Complex.imaginaryPart (2 * w₅ ^ 3)}
  hasSup imgs (Complex.imaginaryPart (2 * w₂ ^ 3)) :=
by
  sorry

end greatest_imaginary_part_of_2w3_l93_93643


namespace smallest_composite_no_prime_factors_less_than_20_l93_93097

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93097


namespace statement_A_statement_B_statement_C_statement_D_l93_93294

section Probability

variables (Products : finset ℕ) (qualified defective : ℕ)

def is_first_class (p : ℕ) : Prop := p < 2
def is_second_class (p : ℕ) : Prop := p = 2
def is_defective (p : ℕ) : Prop := p = 3

def event_A (s : finset ℕ) : Prop := ∀ p ∈ s, is_first_class p
def event_B (s : finset ℕ) : Prop := ∃ p₁ p₂ ∈ s, p₁ ≠ p₂ ∧ is_first_class p₁ ∧ is_second_class p₂
def event_C (s : finset ℕ) : Prop := ∃ p₁ p₂ ∈ s, p₁ ≠ p₂ ∧ ((is_defective p₁ ∧ is_first_class p₂) ∨ (is_defective p₁ ∧ is_second_class p₂))
def event_D (s : finset ℕ) : Prop := ∃ p ∈ s, is_first_class p

variables (s : finset ℕ) (h : s.card = 2)

-- Statement A: Events \(A\) and \(B\) are mutually exclusive.
theorem statement_A : ¬(event_A Products s h ∧ event_B Products s h) := 
sorry

-- Statement B: Events \(A\) and \(B\) are independent.
theorem statement_B : ¬(independent (event_A Products) (event_B Products)) := 
sorry

-- Statement C: \(P(C) = \frac{1}{6}\)
theorem statement_C : Prob.event_Q prob_space (λ w, event_C Products (finset.singleton w)) = 1/6 := 
sorry

-- Statement D: \(P(D) = P(A) + P(B) + P(C)\)
theorem statement_D : Prob.event_Q prob_space (λ w, event_D Products (finset.singleton w)) = Prob.event_Q prob_space (λ w, event_A Products (finset.singleton w)) + Prob.event_Q prob_space (λ w, event_B Products (finset.singleton w)) + Prob.event_Q prob_space (λ w, event_C Products (finset.singleton w)) := 
sorry

end Probability

end statement_A_statement_B_statement_C_statement_D_l93_93294


namespace sequence_a_5_l93_93191

noncomputable section

-- Definition of the sequence
def a : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => a (n + 1) + a n

-- Statement to prove that a 4 = 8 (in Lean, the sequence is zero-indexed, so a 4 is a_5)
theorem sequence_a_5 : a 4 = 8 :=
  by
    sorry

end sequence_a_5_l93_93191


namespace sum_of_product_of_tangents_l93_93195

theorem sum_of_product_of_tangents (a_n : ℕ → ℝ) (d : ℝ) (k : ℝ) :
  (∀ n, a_n = n * d + a_n 0) →
  d = π / 9 →
  ∑ i in range 8, a_n i = 6 * π →
  tan (π / 9) = k →
  ∑ i in range 7, tan (a_n i) * tan (a_n (i + 1)) = (11 - 7 * k^2) / (k^2 - 1) :=
begin
  sorry
end

end sum_of_product_of_tangents_l93_93195


namespace smallest_composite_no_prime_factors_less_than_20_l93_93027

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93027


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93086

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93086


namespace fraction_white_surface_area_l93_93986

-- Definitions of the given conditions
def cube_side_length : ℕ := 4
def small_cubes : ℕ := 64
def black_cubes : ℕ := 34
def white_cubes : ℕ := 30
def total_surface_area : ℕ := 6 * cube_side_length^2
def black_faces_exposed : ℕ := 32 
def white_faces_exposed : ℕ := total_surface_area - black_faces_exposed

-- The proof statement
theorem fraction_white_surface_area (cube_side_length_eq : cube_side_length = 4)
                                    (small_cubes_eq : small_cubes = 64)
                                    (black_cubes_eq : black_cubes = 34)
                                    (white_cubes_eq : white_cubes = 30)
                                    (black_faces_eq : black_faces_exposed = 32)
                                    (total_surface_area_eq : total_surface_area = 96)
                                    (white_faces_eq : white_faces_exposed = 64) : 
                                    (white_faces_exposed : ℚ) / (total_surface_area : ℚ) = 2 / 3 :=
by
  sorry

end fraction_white_surface_area_l93_93986


namespace complex_conjugate_product_l93_93349

variable (w : ℂ)
variable (h : Complex.abs w = 15)

theorem complex_conjugate_product : w * Complex.conj w = 225 := by
  sorry

end complex_conjugate_product_l93_93349


namespace ab_cd_not_prime_l93_93327

theorem ab_cd_not_prime (a b c d : ℕ) (ha : a > b) (hb : b > c) (hc : c > d) (hd : d > 0)
  (h : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : ¬ Nat.Prime (a * b + c * d) := 
sorry

end ab_cd_not_prime_l93_93327


namespace probability_exactly_5_shots_expected_number_of_shots_l93_93975

-- Define the problem statement and conditions with Lean definitions
variables (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

-- Part (a): Probability of 5 shots needed
theorem probability_exactly_5_shots : 
  (6 * p^3 * (1 - p)^2) = probability_exactly_5_shots (p : ℝ) :=
sorry

-- Part (b): Expected number of shots needed
theorem expected_number_of_shots :
  (3 / p) = expected_number_of_shots (p : ℝ) :=
sorry

end probability_exactly_5_shots_expected_number_of_shots_l93_93975


namespace smallest_composite_no_prime_factors_less_than_20_l93_93021

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93021


namespace num_whole_numbers_between_l93_93702

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93702


namespace average_budget_decrease_l93_93290

theorem average_budget_decrease (n : ℕ) (ΔT T_new k : ℕ) (A B : ℝ) 
  (h1 : n = 100) 
  (h2 : ΔT = 400) 
  (h3 : T_new = 5400) 
  (h4 : k = 32) 
  (h5 : 100 * A = 5000) 
  (h6 : 132 * B = 5400) :
  A - B ≈ 9.09 := 
by 
  sorry

end average_budget_decrease_l93_93290


namespace jennifer_distance_l93_93817

noncomputable def distance_from_start (north: ℝ) (east: ℝ) : ℝ :=
  real.sqrt (north^2 + east^2)

theorem jennifer_distance: distance_from_start (3 + 5 / real.sqrt 2) (5 / real.sqrt 2) = real.sqrt (34 + 15 * real.sqrt 2) :=
by
  sorry

end jennifer_distance_l93_93817


namespace gasoline_saving_problem_l93_93865

theorem gasoline_saving_problem
  (mix_percent : ℝ := 0.06)
  (reduction_percent : ℝ := 0.15)
  (target_saving : ℝ := 100) :
  ∃ (gasoline_mix_needed : ℝ), gasoline_mix_needed ≈ 500 := by
  sorry

end gasoline_saving_problem_l93_93865


namespace product_in_fourth_quadrant_l93_93803

def complex_number_1 := (1 : ℂ) - (2 : ℂ) * I
def complex_number_2 := (2 : ℂ) + I

def product := complex_number_1 * complex_number_2

theorem product_in_fourth_quadrant :
  product.re > 0 ∧ product.im < 0 :=
by
  -- the proof is omitted
  sorry

end product_in_fourth_quadrant_l93_93803


namespace prime_square_mod_12_l93_93373

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_ne2 : p ≠ 2) (h_ne3 : p ≠ 3) :
    (∃ n : ℤ, p = 6 * n + 1 ∨ p = 6 * n + 5) → (p^2 % 12 = 1) := by
  sorry

end prime_square_mod_12_l93_93373


namespace find_a_find_sin_B_sin_C_l93_93786

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cos_A : ℝ)
variable (√21 : ℝ)
variable (sin_B sin_C : ℝ)

noncomputable def cos_A := 1/2
noncomputable def |√21 := Real.sqrt 21
noncomputable def sin_B := 5 / (2 * √21)
noncomputable def sin_C := 4 / (2 * √21)

theorem find_a (A : ℝ) (b : ℝ) (c : ℝ) (cos_A : ℝ) : a = Real.sqrt 21 :=
by
  -- proof here
  sorry

theorem find_sin_B_sin_C (A : ℝ) (b : ℝ) (c : ℝ) (a : ℝ) (sin_B : ℝ) (sin_C : ℝ) : sin_B * sin_C = 5 / 7 :=
by
  -- proof here
  sorry

end find_a_find_sin_B_sin_C_l93_93786


namespace second_person_avg_pages_per_day_l93_93565

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end second_person_avg_pages_per_day_l93_93565


namespace count_whole_numbers_between_roots_l93_93756

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93756


namespace inequality_has_no_solutions_l93_93177

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end inequality_has_no_solutions_l93_93177


namespace probability_R_given_A_and_B_l93_93946

-- Define the events and their probabilities
variable (R A B : Prop)

-- Given probabilities
variable (P_R : ℝ := 0.5)
variable (P_notR : ℝ := 0.5)
variable (P_B_given_R : ℝ := 0.8)
variable (P_A_given_R_and_B : ℝ := 0.05)
variable (P_A_given_notR : ℝ := 0.9)
variable (P_B_given_notR_and_A : ℝ := 0.02)

-- Define the total probability of event A and B
noncomputable def P_A_and_B : ℝ :=
  (P_R * P_B_given_R * P_A_given_R_and_B) +
  (P_notR * P_A_given_notR * P_B_given_notR_and_A)

-- Define the probability of R and A and B
noncomputable def P_R_and_A_and_B : ℝ :=
  P_R * P_B_given_R * P_A_given_R_and_B

-- Define the conditional probability P(R | A ∩ B)
noncomputable def P_R_given_A_and_B : ℝ :=
  P_R_and_A_and_B / P_A_and_B

-- Prove that the conditional probability is approximately 0.69
theorem probability_R_given_A_and_B : P_R_given_A_and_B R A B P_R P_notR P_B_given_R P_A_given_R_and_B P_A_given_notR P_B_given_notR_and_A ≈ 0.69 :=
sorry

end probability_R_given_A_and_B_l93_93946


namespace prime_divisors_390_num_prime_divisors_390_l93_93671

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

theorem prime_divisors_390 : prime_factors 390 = {2, 3, 5, 13} :=
by {
  sorry
}

theorem num_prime_divisors_390 : (prime_factors 390).to_finset.card = 4 :=
by {
  rw prime_divisors_390,
  simp,
}

end prime_divisors_390_num_prime_divisors_390_l93_93671


namespace smallest_composite_no_prime_factors_less_than_20_l93_93115

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93115


namespace smallest_composite_no_prime_factors_less_20_l93_93041

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93041


namespace probability_spinner_lands_in_shaded_region_l93_93516

theorem probability_spinner_lands_in_shaded_region :
  let total_regions := 4
  let shaded_regions := 3
  (shaded_regions: ℝ) / total_regions = 3 / 4 :=
by
  let total_regions := 4
  let shaded_regions := 3
  sorry

end probability_spinner_lands_in_shaded_region_l93_93516


namespace num_whole_numbers_between_l93_93699

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93699


namespace letter_in_sequence_l93_93949

theorem letter_in_sequence : 
  ∀ (n : ℕ), let word := "математика" in
    word.length = 10 → 
    (word.to_list.nth ((n - 1) % 10) = some 'т') ↔ n ≡ 3 [MOD 10] :=
by
  sorry

end letter_in_sequence_l93_93949


namespace collinear_DMC_l93_93193

variables {A B C O1 O2 D M : Type*}

open_locale classical
noncomputable theory

-- Let A, B, C be points forming the triangle ABC
-- Let O1 be the incenter of triangle ABC
-- Let O2 be the excenter opposite to vertex A
-- Let D be a point on the arc BO2 of the circumcircle of triangle O1O2B such that angle BO2D = 1/2 angle BAC
-- Let M be the midpoint of the arc BC of the circumcircle of triangle ABC that does not contain A

-- Assuming all necessary geometrical constructs and definitions
variables [triangle A B C] [incenter O1 A B C] [excenter O2 A B C]
variables [on_arc D B O2 (circumcircle O1 O2 B)] [angle_eq_half_AAC O2 D B C A]

-- Prove that points D, M, and C are collinear
theorem collinear_DMC (h_mid : midpoint M arc_BC (circumcircle A B C)) :
  collinear {D, M, C} := sorry

end collinear_DMC_l93_93193


namespace least_possible_product_of_consecutive_primes_gt_50_l93_93907

/-- Let p1 and p2 be the smallest consecutive primes greater than 50. --/
def p1 : ℕ := 53
def p2 : ℕ := 59
def is_prime (n : ℕ) : Prop := nat.prime n

theorem least_possible_product_of_consecutive_primes_gt_50 : 
  is_prime p1 ∧ is_prime p2 ∧ p1 > 50 ∧ p2 > 50 ∧ p2 = p1 + nat.find (λ k, k ≥ 2 ∧ is_prime (p1 + k)) → p1 * p2 = 3127 := by
  sorry

end least_possible_product_of_consecutive_primes_gt_50_l93_93907


namespace x_minus_p_expression_l93_93769

theorem x_minus_p_expression (x p : ℝ) (h1 : |x - 5| = p) (h2 : x < 5) : x - p = 5 - 2p := 
by sorry

end x_minus_p_expression_l93_93769


namespace original_number_l93_93462

theorem original_number (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by {
  sorry -- We will skip the actual proof steps here.
}

end original_number_l93_93462


namespace infinite_n_f_n_ge_f_n_plus_1_l93_93604

-- Define f(n)
def f (n : ℕ) : ℕ := (n^2 + n + 1).num_divisors

-- Theorem statement: There are infinitely many n such that f(n) ≥ f(n+1)
theorem infinite_n_f_n_ge_f_n_plus_1 : ∃ᶠ n in at_top, f(n) ≥ f(n + 1) :=
sorry

end infinite_n_f_n_ge_f_n_plus_1_l93_93604


namespace smallest_composite_no_prime_lt_20_l93_93058

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93058


namespace company_B_more_cost_effective_l93_93510

-- Definitions of the conditions
def discount_A := 0.2
def full_price := 10
def area_limit := 200
def discount_B := 0.4

-- Monthly maintenance fees in terms of x
def monthly_fee_A (x : ℕ) : ℝ := full_price * (1 - discount_A) * x
def monthly_fee_B (x : ℕ) : ℝ := full_price * area_limit + full_price * (1 - discount_B) * (x - area_limit)

-- The given landscaping area for comparison
def given_area := 600

-- The monthly fees for the given area
def fee_A := monthly_fee_A given_area
def fee_B := monthly_fee_B given_area

-- Proof problem
theorem company_B_more_cost_effective : fee_B < fee_A :=
by
  unfold fee_A fee_B monthly_fee_A monthly_fee_B full_price discount_A discount_B area_limit given_area
  simp [full_price, discount_A, discount_B, area_limit, given_area]
  linarith
  -- Numeric computation steps can be expressed in actual proofs
  -- 8 * 600 > 6 * 600 + 800
  -- 4800 > 4400

end company_B_more_cost_effective_l93_93510


namespace algorithm_effective_expected_residents_l93_93488

-- Define the parameters and conditions
def mailboxes : Type := Fin 80
def initial_mailbox : mailboxes := 37

-- Part (a): Prove that the algorithm is effective
theorem algorithm_effective (k : mailboxes -> mailboxes) (h_random : random_permutation k) :
  ∃ n, (k^[n] initial_mailbox) = initial_mailbox :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

-- Part (b): Find the expected number of cycles
noncomputable def harmonic_number (n : ℕ) : ℝ :=
  (finset.range (nat.succ n)).sum (λ i, 1 / (i + 1))

theorem expected_residents (n : ℕ) (h_key_count : n = 80) :
  ∑ k in finset.range n, (1 : ℝ) / (k+1) = harmonic_number 80 :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

end algorithm_effective_expected_residents_l93_93488


namespace typist_original_salary_l93_93886

theorem typist_original_salary (x : ℝ) (h : (x * 1.10 * 0.95 = 4180)) : x = 4000 :=
by sorry

end typist_original_salary_l93_93886


namespace min_k_value_l93_93493

-- Definitions for conditions
def num_students : ℕ := 1200

variable (Student : Type) -- Type representing students
variable [fintype Student] -- Ensuring the set of students is finite
variable [decidable_eq Student] -- Ensuring students are decidable equality

def Club : Type := Set Student

variable (clubs : ℕ → Club) -- Function mapping an index to a club
variable (k : ℕ) -- Number of clubs each student is in

-- Conditions
def condition1 : Prop := fintype.card Student = num_students
def condition2 : Prop := ∀ s : Student, fintype.card {c | s ∈ clubs c} = k
def condition3 : Prop := ∀ (s : finset Student), s.card = 23 → ∃ c, ∀ t ∈ s, t ∈ clubs c
def condition4 : Prop := ∀ c, ¬ ∀ s : Student, s ∈ clubs c

-- Theorem statement
theorem min_k_value : condition1 →
                      condition2 →
                      condition3 →
                      condition4 →
                      k = 23 :=
begin
  intros h1 h2 h3 h4,
  sorry -- the proof would go here
end

end min_k_value_l93_93493


namespace smallest_composite_no_prime_factors_less_than_20_l93_93033

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93033


namespace square_of_square_root_l93_93918

theorem square_of_square_root (x : ℝ) (hx : (Real.sqrt x)^2 = 49) : x = 49 :=
by 
  sorry

end square_of_square_root_l93_93918


namespace smallest_composite_no_prime_lt_20_l93_93054

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93054


namespace parallel_vectors_x_value_l93_93252

-- Defining the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Condition for vectors a and b to be parallel
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value : ∃ x, are_parallel a (b x) ∧ x = 6 := by
  sorry

end parallel_vectors_x_value_l93_93252


namespace focus_coincidence_parabola_hyperbola_l93_93635

theorem focus_coincidence_parabola_hyperbola (p : ℝ) (hp : p > 0) :
  (∃ y, ∀ x, x^2 = 2 * p * y) ∧
  (∀ x y, y^2 / 3 - x^2 = 1 → ∃ y_focus, (0, y_focus) ∈ set_of (λ x, y^2 / 3 - x^2 = 1)) →
  p = 4 :=
by 
  sorry

end focus_coincidence_parabola_hyperbola_l93_93635


namespace no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l93_93494

noncomputable def system_discriminant (a b c : ℝ) : ℝ := (b - 1)^2 - 4 * a * c

theorem no_real_solutions_if_discriminant_neg (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c < 0) :
  ¬∃ (x₁ x₂ x₃ : ℝ), (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

theorem one_real_solution_if_discriminant_zero (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c = 0) :
  ∃ (x : ℝ), ∀ (x₁ x₂ x₃ : ℝ), (x₁ = x) ∧ (x₂ = x) ∧ (x₃ = x) ∧
                              (a * x₁^2 + b * x₁ + c = x₂) ∧
                              (a * x₂^2 + b * x₂ + c = x₃) ∧
                              (a * x₃^2 + b * x₃ + c = x₁)  :=
sorry

theorem more_than_one_real_solution_if_discriminant_pos (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c > 0) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

end no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l93_93494


namespace root_interval_exists_l93_93279

def f (x : ℝ) : ℝ := 2^x + x - 2

theorem root_interval_exists:
  ∃ a : ℤ, f (a : ℝ) < 0 ∧ f (a + 1 : ℝ) > 0 := by
  use (0 : ℤ)
  -- Let's check f(0) and f(1):
  -- f(0) = 2^0 + 0 - 2 = -1 < 0
  -- f(1) = 2^1 + 1 - 2 = 1 > 0
  sorry

end root_interval_exists_l93_93279


namespace terminating_decimal_l93_93588

theorem terminating_decimal : (19 : ℝ) / (2^2 * 5^3) = 0.095 := 
by {
  sorry,
}

end terminating_decimal_l93_93588


namespace number_of_whole_numbers_between_cubicroots_l93_93734

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93734


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93745

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93745


namespace no_solution_l93_93171

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end no_solution_l93_93171


namespace complex_number_solution_l93_93613

theorem complex_number_solution (z : ℂ) (h : (1 + (real.sqrt 3 : ℂ) * complex.I) * z = (real.sqrt 3 : ℂ) * complex.I) :
  z = (3 / 4 : ℂ) + (real.sqrt 3 / 4 : ℂ) * complex.I :=
sorry

end complex_number_solution_l93_93613


namespace length_of_train_l93_93932

theorem length_of_train 
  (L V : ℝ) 
  (h1 : L = V * 8) 
  (h2 : L + 279 = V * 20) : 
  L = 186 :=
by
  -- solve using the given conditions
  sorry

end length_of_train_l93_93932


namespace smallest_composite_no_prime_factors_lt_20_l93_93157

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93157


namespace general_term_a_sum_of_b_sequence_l93_93218

noncomputable def S (n : ℕ) : ℝ := (n^2 + n) / 2

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else n

noncomputable def b (n : ℕ) : ℝ := 1 / S n

noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)

theorem general_term_a (n : ℕ) : a n = n :=
begin
  sorry
end

theorem sum_of_b_sequence (n : ℕ) : T n = (2 * n) / (n + 1) :=
begin
  sorry
end

end general_term_a_sum_of_b_sequence_l93_93218


namespace smallest_composite_no_prime_factors_less_than_20_l93_93004

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93004


namespace solution_set_of_inequality_l93_93269

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable (odd_f : ∀ x, f (-x) = -f x)
variable (f_der : ∀ x, HasDerivAt f (f' x) x)
variable (cond : ∀ x, x > 0 → x * f' x + 3 * f x > 0)

theorem solution_set_of_inequality :
  { x : ℝ | x^3 * f x + (2 * x - 1)^3 * f (1 - 2 * x) < 0 } = 
    set.Iio (1/3) ∪ set.Ioi 1 :=
sorry

end solution_set_of_inequality_l93_93269


namespace find_key_effective_expected_residents_to_use_algorithm_l93_93470

-- Define mailboxes and keys
def num_mailboxes : ℕ := 80
def initial_mailbox : ℕ := 37

-- Prove that the algorithm is effective
theorem find_key_effective : 
  ∀ (mailboxes : fin num_mailboxes) (keys : fin num_mailboxes), 
  { permutation : list (fin num_mailboxes) // permutation.nodup ∧ permutation.length = num_mailboxes ∧ 
    ∀ m, m ∈ permutation → (if m = initial_mailbox then m else (keys.filter (λ k, k ∈ permutation))) ≠ [] }
  :=
sorry

-- Prove the expected number of residents who will use the algorithm
theorem expected_residents_to_use_algorithm :
  ∑ i in finset.range num_mailboxes, (1 / (i + 1 : ℝ)) = (real.log 80 + 0.577 + (1 / (2 * 80)) : ℝ)
  :=
sorry

end find_key_effective_expected_residents_to_use_algorithm_l93_93470


namespace area_of_figure_l93_93577

noncomputable def area_enclosed : ℝ :=
  ∫ x in (0 : ℝ)..(2 * Real.pi / 3), 2 * Real.sin x

theorem area_of_figure :
  area_enclosed = 3 := by
  sorry

end area_of_figure_l93_93577


namespace train_passing_time_l93_93464

noncomputable def relative_speed (train_speed man_speed : ℝ) : ℝ :=
  train_speed + man_speed

noncomputable def speed_in_meters_per_second (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000/3600)

noncomputable def passing_time (distance speed_mps : ℝ) : ℝ :=
  distance / speed_mps

theorem train_passing_time (d : ℝ) (V_train V_man : ℝ) (t : ℝ) : 
  d = 330 ∧ V_train = 60 ∧ V_man = 6 ∧ t = 18 → 
  passing_time d (speed_in_meters_per_second (relative_speed V_train V_man)) = t :=
by
  intros h,
  cases h,
  cases h_left,
  unfold passing_time speed_in_meters_per_second relative_speed,
  sorry

end train_passing_time_l93_93464


namespace unique_integer_solution_l93_93594

theorem unique_integer_solution :
  ∀ (x y : ℤ), (∃ (k : ℤ), k = 1998 ∧ (s : ℤ) → (s = √x) ∧
    (∀ n, 1 ≤ n ∧ n ≤ k → (s : ℤ) → (s = √(x + s))) ∧ y = s) → y = 0 ∧ x = 0 :=
by
  sorry

end unique_integer_solution_l93_93594


namespace probability_convex_quadrilateral_l93_93859

theorem probability_convex_quadrilateral (n : ℕ) (h_n : n = 7) :
  let total_chords := (nat.choose n 2),
      total_combinations := (nat.choose total_chords 4),
      favorable_combinations := (nat.choose n 4)
  in (favorable_combinations / total_combinations : ℚ) = (1 / 171 : ℚ) :=
by 
  -- Definitions and calculations are omitted
  sorry

end probability_convex_quadrilateral_l93_93859


namespace part2_inequality_l93_93231

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- The main theorem we want to prove
theorem part2_inequality (a b c : ℝ) (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  |a + 2 * b + 3 * c| ≤ 6 :=
by {
-- Proof goes here
sorry
}

end part2_inequality_l93_93231


namespace algorithm_effective_expected_residents_l93_93492

-- Define the parameters and conditions
def mailboxes : Type := Fin 80
def initial_mailbox : mailboxes := 37

-- Part (a): Prove that the algorithm is effective
theorem algorithm_effective (k : mailboxes -> mailboxes) (h_random : random_permutation k) :
  ∃ n, (k^[n] initial_mailbox) = initial_mailbox :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

-- Part (b): Find the expected number of cycles
noncomputable def harmonic_number (n : ℕ) : ℝ :=
  (finset.range (nat.succ n)).sum (λ i, 1 / (i + 1))

theorem expected_residents (n : ℕ) (h_key_count : n = 80) :
  ∑ k in finset.range n, (1 : ℝ) / (k+1) = harmonic_number 80 :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

end algorithm_effective_expected_residents_l93_93492


namespace smallest_composite_no_prime_factors_less_than_20_l93_93144

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93144


namespace exact_probability_five_shots_l93_93960

theorem exact_probability_five_shots (p : ℝ) (h1 : 0 < p) (h2 : p ≤ 1) :
  (let hit := p
       miss := 1 - p
       comb := 6 in
   comb * hit^3 * miss^2 = 6 * p^3 * (1 - p)^2) :=
by sorry

end exact_probability_five_shots_l93_93960


namespace smallest_composite_no_prime_factors_less_than_20_l93_93031

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93031


namespace find_parabola_eq_find_line_eq_through_chord_midpoint_l93_93626

-- Definition of the focus and parabola
def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Given conditions for M and the distance |MF| = 3
def point_M_on_parabola (p m : ℝ) : Prop := parabola p 2 m
def distance_MF (p m : ℝ) : Prop := (sqrt ((2 - p / 2)^2 + m^2) = 3)

noncomputable def parabola_E_eq (p : ℝ) : ℝ := 4 * p

theorem find_parabola_eq (m : ℝ) 
  (hM : point_M_on_parabola 2 m) 
  (hDist : distance_MF 2 m) : (∃ p > 0, parabola_E_eq p = 4) := 
by {
  sorry
}

-- Line equation through midpoint
def chord_midpoint (N : ℝ × ℝ) := N = (1, 1)
def line_through_midpoint (N : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y = k * x + (1 - k)
def parabola_E_intersection (p k : ℝ) : ℝ → Prop := 
  λ x, 4 * x = (k * x + (1 - k))^2

theorem find_line_eq_through_chord_midpoint (k x y : ℝ)
  (hN : chord_midpoint (1, 1))
  (hLine : line_through_midpoint (1, 1) k x y)
  (hParabolaE : parabola_E_intersection 2 k x) : (x - 2 * y + 1 = 0) := 
by {
  sorry
}

end find_parabola_eq_find_line_eq_through_chord_midpoint_l93_93626


namespace number_of_valid_numbers_l93_93261

def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n < 1000 }
noncomputable def valid_number (n : ℕ) : Prop :=
  let x := n / 100 in
  let y := (n / 10) % 10 in
  let z := n % 10 in
  (x = z) ∧
  ((2 * x + y) < 22) ∧
  (2 * x + y) % 3 ≠ 0

theorem number_of_valid_numbers : 
  { n ∈ three_digit_numbers | valid_number n }.card = 112 := 
sorry

end number_of_valid_numbers_l93_93261


namespace spider_travel_distance_l93_93996

theorem spider_travel_distance (r : ℝ) (journey3 : ℝ) (diameter : ℝ) (leg2 : ℝ) :
    r = 75 → journey3 = 110 → diameter = 2 * r → 
    leg2 = Real.sqrt (diameter^2 - journey3^2) → 
    diameter + leg2 + journey3 = 362 :=
by
  sorry

end spider_travel_distance_l93_93996


namespace union_of_sets_l93_93247

noncomputable def M (a : ℝ) : Set ℝ := {5, Real.logBase 2 a}
def N (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ)
  (hM : M a = {5, Real.logBase 2 a})
  (hN : N a b = {a, b})
  (hIntersection : M a ∩ N a b = {1}) :
  M a ∪ N a b = {1, 2, 5} :=
by
  sorry

end union_of_sets_l93_93247


namespace whole_numbers_between_l93_93731

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93731


namespace smallest_b_for_polynomial_factorization_l93_93161

theorem smallest_b_for_polynomial_factorization :
  ∃ (b : ℕ), (b > 0) ∧ (∀ p q : ℤ, (p + q = b) ∧ (p * q = 2520) → b = 106) :=
begin
  use 106,
  split,
  { exact 106 > 0, },
  { intros p q h,
    cases h with h1 h2,
    sorry,
  }
end

end smallest_b_for_polynomial_factorization_l93_93161


namespace sum_possible_values_of_n_l93_93869

theorem sum_possible_values_of_n :
  let S := {3, 6, 9, 10}
  let T := S ∪ {n} in
  n ≠ 3 ∧ n ≠ 6 ∧ n ≠ 9 ∧ n ≠ 10 ∧
  ((∃ x ∈ T, (T.toFinset.sort (· ≤ ·)).nth 2 = some x ∧ (S.sum + n) / 5 = x) → 
   (∑ n' in {2, 7, 17}, n') = 26) :=
sorry

end sum_possible_values_of_n_l93_93869


namespace systematic_sampling_interval_l93_93898

-- Definitions based on the conditions in part a)
def total_students : ℕ := 1500
def sample_size : ℕ := 30

-- The goal is to prove that the interval k in systematic sampling equals 50
theorem systematic_sampling_interval :
  (total_students / sample_size = 50) :=
by
  sorry

end systematic_sampling_interval_l93_93898


namespace algorithm_effective_expected_residents_l93_93489

-- Define the parameters and conditions
def mailboxes : Type := Fin 80
def initial_mailbox : mailboxes := 37

-- Part (a): Prove that the algorithm is effective
theorem algorithm_effective (k : mailboxes -> mailboxes) (h_random : random_permutation k) :
  ∃ n, (k^[n] initial_mailbox) = initial_mailbox :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

-- Part (b): Find the expected number of cycles
noncomputable def harmonic_number (n : ℕ) : ℝ :=
  (finset.range (nat.succ n)).sum (λ i, 1 / (i + 1))

theorem expected_residents (n : ℕ) (h_key_count : n = 80) :
  ∑ k in finset.range n, (1 : ℝ) / (k+1) = harmonic_number 80 :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

end algorithm_effective_expected_residents_l93_93489


namespace sequence_contains_square_l93_93344

noncomputable def f (n : ℕ) : ℕ := n + Nat.floor (Real.sqrt n)

def seq (m : ℕ) : ℕ → ℕ
| 0       := m
| (n + 1) := f (seq m n)

theorem sequence_contains_square (m : ℕ) : ∃ i, ∃ n, seq m i = n^2 := by
  sorry

end sequence_contains_square_l93_93344


namespace total_volume_of_cubes_l93_93922

theorem total_volume_of_cubes {n : ℕ} (h_n : n = 5) (s : ℕ) (h_s : s = 5) :
  n * (s^3) = 625 :=
by {
  rw [h_n, h_s],
  norm_num,
  sorry
}

end total_volume_of_cubes_l93_93922


namespace smallest_composite_no_prime_lt_20_l93_93061

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93061


namespace train_stops_one_minute_per_hour_l93_93460

theorem train_stops_one_minute_per_hour (D : ℝ) (h1 : D / 400 = T₁) (h2 : D / 360 = T₂) : 
  (T₂ - T₁) * 60 = 1 :=
by
  sorry

end train_stops_one_minute_per_hour_l93_93460


namespace exact_probability_five_shots_l93_93961

theorem exact_probability_five_shots (p : ℝ) (h1 : 0 < p) (h2 : p ≤ 1) :
  (let hit := p
       miss := 1 - p
       comb := 6 in
   comb * hit^3 * miss^2 = 6 * p^3 * (1 - p)^2) :=
by sorry

end exact_probability_five_shots_l93_93961


namespace probability_exactly_5_shots_expected_number_of_shots_l93_93978

-- Define the problem statement and conditions with Lean definitions
variables (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

-- Part (a): Probability of 5 shots needed
theorem probability_exactly_5_shots : 
  (6 * p^3 * (1 - p)^2) = probability_exactly_5_shots (p : ℝ) :=
sorry

-- Part (b): Expected number of shots needed
theorem expected_number_of_shots :
  (3 / p) = expected_number_of_shots (p : ℝ) :=
sorry

end probability_exactly_5_shots_expected_number_of_shots_l93_93978


namespace whole_numbers_between_cubicroots_l93_93704

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93704


namespace part1_part2_l93_93240

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l93_93240


namespace count_not_divisible_by_5_count_not_divisible_by_5_or_3_count_not_divisible_by_5_or_3_or_11_l93_93667

theorem count_not_divisible_by_5 : 
    (Finset.filter (λ n, ¬(5 ∣ n)) (Finset.range (16500 + 1))).card = 13200 :=
by sorry

theorem count_not_divisible_by_5_or_3 : 
    (Finset.filter (λ n, ¬(5 ∣ n) ∧ ¬(3 ∣ n)) (Finset.range (16500 + 1))).card = 8800 :=
by sorry

theorem count_not_divisible_by_5_or_3_or_11 : 
    (Finset.filter (λ n, ¬(5 ∣ n) ∧ ¬(3 ∣ n) ∧ ¬(11 ∣ n)) (Finset.range (16500 + 1))).card = 8000 :=
by sorry

end count_not_divisible_by_5_count_not_divisible_by_5_or_3_count_not_divisible_by_5_or_3_or_11_l93_93667


namespace salad_cucumbers_l93_93541

theorem salad_cucumbers (c t : ℕ) 
  (h1 : c + t = 280)
  (h2 : t = 3 * c) : c = 70 :=
sorry

end salad_cucumbers_l93_93541


namespace smallest_composite_no_prime_factors_below_20_l93_93125

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93125


namespace problem_statement_l93_93770

-- Definitions
def x : ℕ := 3
def y : ℕ := 4

-- Theorem statement
theorem problem_statement : 3 * x - 5 * y = -11 := by
  sorry

end problem_statement_l93_93770


namespace smallest_composite_no_prime_lt_20_l93_93056

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93056


namespace smallest_composite_no_prime_factors_lt_20_l93_93160

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93160


namespace correct_propositions_count_l93_93661

variables (m n l : Line)
variables (α β : Plane)
variables (prop1 prop2 prop3 prop4 : Prop)

-- Defining the conditions
def cond1 : Prop := m ∥ n ∧ n ⊆ α
def cond2 : Prop := l ⊥ α ∧ m ⊥ β ∧ l ∥ m
def cond3 : Prop := m ⊆ α ∧ n ⊆ α ∧ m ∥ β ∧ n ∥ β
def cond4 : Prop := α ⊥ β ∧ (α ∩ β) = m ∧ n ⊆ β ∧ n ⊥ m

-- Propositions based on the conditions
def proposition1 : Prop := cond1 → m ∥ α
def proposition2 : Prop := cond2 → α ∥ β
def proposition3 : Prop := cond3 → α ∥ β
def proposition4 : Prop := cond4 → n ⊥ α

-- Theorem stating the number of correct propositions
theorem correct_propositions_count : (proposition1 ↔ false) ∧ (proposition2 ↔ true) ∧ (proposition3 ↔ false) ∧ (proposition4 ↔ true) → 2 := 
by sorry

end correct_propositions_count_l93_93661


namespace smallest_composite_no_prime_factors_less_20_l93_93044

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93044


namespace john_car_speed_l93_93941

noncomputable def john_travel_speed : ℕ → Prop :=
  λ (D : ℕ), ∃ v : ℕ, 
    (80 / 80 + 0.2 * D / v = D / 50) ∧
    v = 20

theorem john_car_speed : ∀ D : ℕ, john_travel_speed D :=
begin
  intros D,
  have v : ℕ,
    from 20,
  refine ⟨v, _⟩,
  split,
  { sorry },   -- This is where the proof would go, which we are omitting.
  { refl }
end

end john_car_speed_l93_93941


namespace leftover_pizzas_l93_93820

variables (hours : ℕ) (flour_kg : ℕ) (minutes_per_pizza : ℕ) (flour_per_pizza_kg : ℕ)

theorem leftover_pizzas (h : hours = 7) (f : flour_kg = 22) (mpp : minutes_per_pizza = 10)
  (fp : flour_per_pizza_kg = 2/4) :
  let total_flour_pizzas := flour_kg / (flour_per_pizza_kg)
  let total_minutes := hours * 60
  let total_time_pizzas := total_minutes / minutes_per_pizza
  in  total_flour_pizzas - total_time_pizzas = 2 :=
by {
  sorry
}

end leftover_pizzas_l93_93820


namespace present_age_of_R_l93_93979

variables (P_p Q_p R_p : ℝ)

-- Conditions from the problem
axiom h1 : P_p - 8 = 1/2 * (Q_p - 8)
axiom h2 : Q_p - 8 = 2/3 * (R_p - 8)
axiom h3 : Q_p = 2 * Real.sqrt R_p
axiom h4 : P_p = 3/5 * Q_p

theorem present_age_of_R : R_p = 400 :=
by
  sorry

end present_age_of_R_l93_93979


namespace special_divisors_at_most_half_exact_special_divisors_l93_93448

def is_special_divisor (n d : ℕ) : Prop := d ∣ n ∧ (d + 1) ∣ n

theorem special_divisors_at_most_half (n : ℕ) (h_pos : 0 < n) :
  ∃ ns, ns = {d : ℕ | d ∣ n ∧ is_special_divisor n d} ∧ ns.card ≤ n.divisors.card / 2 := sorry

theorem exact_special_divisors (n : ℕ) :
  (∃ (ns : Finset ℕ), ns = {d : ℕ | d ∣ n ∧ is_special_divisor n d} ∧ ns.card = n.divisors.card / 2) ↔ 
  n = 2 ∨ n = 6 ∨ n = 12 := sorry

end special_divisors_at_most_half_exact_special_divisors_l93_93448


namespace find_A_max_height_AD_l93_93300

-- Assuming the necessary imports/constants.
constant triangle (A B C : ℝ) : Prop
constant side_a (A B C a : ℝ) : Prop
constant side_b (A B C b : ℝ) : Prop 
constant side_c (A B C c : ℝ) : Prop 
constant acute (A B C : ℝ) : Prop
constant sinC (A B C sinC : ℝ) : Prop
constant sum_b_c (b c : ℝ) : Prop
constant height_AD (A B C a b c AD : ℝ) : Prop

-- Conditions
axiom triangle_ABC : acute A B C → triangle A B C
axiom sides_abc : side_a A B C a → side_b A B C b → side_c A B C c
axiom sinC_eq : sinC C = 2 * (cos A) * (sin (B + π/3))
axiom bc_sum_eq_6 : sum_b_c b c = 6

-- Proofs
theorem find_A : triangle A B C → side_a A B C a → side_b A B C b → side_c A B C c → 
  sinC C → A = π / 3 :=
by 
  sorry

theorem max_height_AD : triangle A B C → side_a A B C a → side_b A B C b → side_c A B C c →
  sinC C → sum_b_c b c → A = π / 3 → 
  height_AD A B C a b c AD ≤ 3 * sqrt 3 / 2 :=
by
  sorry

end find_A_max_height_AD_l93_93300


namespace james_spent_6_dollars_l93_93315

-- Define the cost of items
def cost_milk : ℕ := 3
def cost_bananas : ℕ := 2

-- Define the sales tax rate as a decimal
def sales_tax_rate : ℚ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℕ := cost_milk + cost_bananas

-- Define the sales tax amount
def sales_tax_amount : ℚ := sales_tax_rate * total_cost_before_tax

-- Define the total amount spent
def total_amount_spent : ℚ := total_cost_before_tax + sales_tax_amount

-- The proof statement
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l93_93315


namespace smallest_composite_no_prime_factors_below_20_l93_93130

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93130


namespace parabola_focus_directrix_distance_l93_93393

theorem parabola_focus_directrix_distance :
  let p : ℝ × ℝ := ⟨0, 1⟩ in   -- Focus of the parabola
  let d := ∀ x : ℝ,  y = -1 in  -- Equation of the directrix
  dist (p.2) (-1) = 2 :=        -- Distance formula between point and line
sorry

end parabola_focus_directrix_distance_l93_93393


namespace algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93482

-- Definition of the mailbox setting and conditions
def mailbox_count : ℕ := 80

def apartment_resident_start : ℕ := 37

noncomputable def randomized_keys_placed_in_mailboxes : Bool :=
  true  -- This is a placeholder; actual randomness is abstracted

-- Statement of the problem
theorem algorithm_will_find_key (mailboxes keys : Fin mailbox_count) (start : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ (sequence : ℕ → Fin mailbox_count), ∀ n : ℕ, sequence n ≠ start → sequence (n+1) ≠ start → 
  (sequence 0 = start ∨ ∃ k : ℕ, sequence k = keys.start → keys = sequence (k+1)).
  sorry

theorem expected_number_of_residents_using_algorithm 
  (mailboxes keys : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ n : ℝ, n ≈ 4.968.
  sorry

end algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93482


namespace max_S_2017_l93_93219

noncomputable def max_S (a b c : ℕ) : ℕ := a + b + c

theorem max_S_2017 :
  ∀ (a b c : ℕ),
  a + b = 1014 →
  c - b = 497 →
  a > b →
  max_S a b c = 2017 :=
by
  intros a b c h1 h2 h3
  sorry

end max_S_2017_l93_93219


namespace smallest_composite_no_prime_factors_less_than_20_l93_93005

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93005


namespace smallest_composite_no_prime_factors_below_20_l93_93132

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93132


namespace smallest_composite_no_prime_factors_less_than_20_l93_93009

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93009


namespace theresa_hours_l93_93435

theorem theresa_hours (h1 h2 h3 h4 h5 h6 : ℕ) (avg : ℕ) (x : ℕ) 
  (H_cond : h1 = 10 ∧ h2 = 8 ∧ h3 = 9 ∧ h4 = 11 ∧ h5 = 6 ∧ h6 = 8)
  (H_avg : avg = 9) : 
  (h1 + h2 + h3 + h4 + h5 + h6 + x) / 7 = avg ↔ x = 11 :=
by
  sorry

end theresa_hours_l93_93435


namespace minimum_value_of_f_l93_93839

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - a - 15|

theorem minimum_value_of_f {a : ℝ} (h0 : 0 < a) (h1 : a < 15) : ∃ Q, (∀ x, a ≤ x ∧ x ≤ 15 → f x a ≥ Q) ∧ Q = 15 := by
  sorry

end minimum_value_of_f_l93_93839


namespace algorithm_effective_expected_residents_l93_93491

-- Define the parameters and conditions
def mailboxes : Type := Fin 80
def initial_mailbox : mailboxes := 37

-- Part (a): Prove that the algorithm is effective
theorem algorithm_effective (k : mailboxes -> mailboxes) (h_random : random_permutation k) :
  ∃ n, (k^[n] initial_mailbox) = initial_mailbox :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

-- Part (b): Find the expected number of cycles
noncomputable def harmonic_number (n : ℕ) : ℝ :=
  (finset.range (nat.succ n)).sum (λ i, 1 / (i + 1))

theorem expected_residents (n : ℕ) (h_key_count : n = 80) :
  ∑ k in finset.range n, (1 : ℝ) / (k+1) = harmonic_number 80 :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

end algorithm_effective_expected_residents_l93_93491


namespace two_points_same_color_distance_one_l93_93409

theorem two_points_same_color_distance_one (colored_plane : ℝ × ℝ → Prop) (h : ∀ x, colored_plane x = C1 ∨ colored_plane x = C2) :
  ∃ x y : ℝ × ℝ, colored_plane x = colored_plane y ∧ dist x y = 1 := 
by
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 0 : ℝ)
  let C := (0.5 : ℝ, real.sqrt 3 / 2)
  have vertices := [A, B, C]
  apply pigeonhole_principle
  exact vertices
  { exact sorry }

end two_points_same_color_distance_one_l93_93409


namespace distinct_rat_nums_l93_93579

theorem distinct_rat_nums (k : ℚ) : 
  (|k| < 100 ∧ ∃ x : ℤ, (7 : ℤ) * (x ^ 2) + x.num * k.den = -20 * k.num ).card = 26 :=
sorry

end distinct_rat_nums_l93_93579


namespace slope_of_DD_l93_93811

-- Reflect a point across the line y = -x
def reflect_across_y_eq_neg_x (p : ℝ×ℝ) : ℝ×ℝ :=
  (-p.2, -p.1)

-- Define points D and D'
def D : ℝ × ℝ := (p, q)
def D' : ℝ × ℝ := reflect_across_y_eq_neg_x D

-- Statement: The slope of line DD' is not -1 but 1.
theorem slope_of_DD'_is_not_neg1_but_pos1 (p q : ℝ) (hp : p < 0) (hq : q > 0) :
  let slope := (D'.2 - D.2) / (D'.1 - D.1) in slope = 1 :=
by
  sorry

end slope_of_DD_l93_93811


namespace solve_for_b_l93_93222

theorem solve_for_b (a b : ℚ) 
  (h1 : 8 * a + 3 * b = -1) 
  (h2 : a = b - 3 ) : 
  5 * b = 115 / 11 := 
by 
  sorry

end solve_for_b_l93_93222


namespace probability_exactly_5_shots_expected_number_of_shots_l93_93974

-- Define the problem statement and conditions with Lean definitions
variables (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

-- Part (a): Probability of 5 shots needed
theorem probability_exactly_5_shots : 
  (6 * p^3 * (1 - p)^2) = probability_exactly_5_shots (p : ℝ) :=
sorry

-- Part (b): Expected number of shots needed
theorem expected_number_of_shots :
  (3 / p) = expected_number_of_shots (p : ℝ) :=
sorry

end probability_exactly_5_shots_expected_number_of_shots_l93_93974


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93087

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93087


namespace probability_arithmetic_progression_l93_93896

def is_arithmetic_progression_with_diff_one (a b c : ℕ) : Prop :=
  (b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ b = c + 1) ∨ (a = c + 1 ∧ c = b + 1) ∨
  (b = c + 1 ∧ c = a + 1) ∨ (a = b + 1 ∧ c = a + 1) ∨ (b = a + 1 ∧ c = b + 1)

def num_sides : ℕ := 4
def total_outcomes : ℕ := num_sides ^ 3

theorem probability_arithmetic_progression :
  (finset.card (finset.filter (λ (x : ℕ × ℕ × ℕ), is_arithmetic_progression_with_diff_one x.1 x.2.1 x.2.2) 
                 (finset.univ : finset (ℕ × ℕ × ℕ))) : ℚ) / total_outcomes.to_rat = 3/16 :=
by
  sorry

end probability_arithmetic_progression_l93_93896


namespace log_sum_equiv_l93_93586

theorem log_sum_equiv :
  (3 / log 3 (1000^4) + 4 / log 7 (1000^4)) = log 10 (3^(1/4) * 7^(1/3)) :=
by
  sorry

end log_sum_equiv_l93_93586


namespace part1_part2_l93_93239

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l93_93239


namespace smallest_composite_no_prime_factors_less_20_l93_93035

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93035


namespace midpoint_product_zero_l93_93341

theorem midpoint_product_zero (x y : ℝ) :
  let A := (2, 6)
  let B := (x, y)
  let C := (4, 3)
  (C = ((2 + x) / 2, (6 + y) / 2)) → (x * y = 0) := by
  intros
  sorry

end midpoint_product_zero_l93_93341


namespace total_volume_of_cubes_l93_93921

theorem total_volume_of_cubes {n : ℕ} (h_n : n = 5) (s : ℕ) (h_s : s = 5) :
  n * (s^3) = 625 :=
by {
  rw [h_n, h_s],
  norm_num,
  sorry
}

end total_volume_of_cubes_l93_93921


namespace whole_numbers_between_cuberoots_l93_93686

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93686


namespace total_study_time_correct_l93_93360

def study_time_wednesday : ℕ := 2
def study_time_thursday : ℕ := 3 * study_time_wednesday
def study_time_friday : ℕ := study_time_thursday / 2
def study_time_weekend : ℕ := study_time_wednesday + study_time_thursday + study_time_friday

def total_study_time : ℕ := study_time_wednesday + study_time_thursday + study_time_friday + study_time_weekend

theorem total_study_time_correct : total_study_time = 22 := by
  have : study_time_wednesday = 2 := by rfl
  have : study_time_thursday = 3 * study_time_wednesday := by rfl
  have : study_time_friday = study_time_thursday / 2 := by rfl
  have : study_time_weekend = study_time_wednesday + study_time_thursday + study_time_friday := by rfl
  have total_study_time_eq : total_study_time = study_time_wednesday + study_time_thursday + study_time_friday + study_time_weekend := by rfl
  calc total_study_time
      = 2 + (3 * 2) + ((3 * 2) / 2) + (2 + (3 * 2) + ((3 * 2) / 2)) : by rw [total_study_time_eq, this, this, this, this]
  ... = 22 : by simp

end total_study_time_correct_l93_93360


namespace second_person_average_pages_per_day_l93_93573

-- Define the given conditions.
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def deshaun_pages_per_book : ℕ := 320
def second_person_percentage : ℝ := 0.75

-- Calculate the total number of pages DeShaun read.
def deshaun_total_pages : ℕ := deshaun_books * deshaun_pages_per_book

-- Calculate the total number of pages the second person read.
def second_person_total_pages : ℕ := (second_person_percentage * deshaun_total_pages).toNat

-- Prove the average number of pages the second person read per day.
def average_pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

theorem second_person_average_pages_per_day :
  average_pages_per_day second_person_total_pages summer_days = 180 :=
by
  sorry

end second_person_average_pages_per_day_l93_93573


namespace probability_queen_heart_jack_l93_93436

theorem probability_queen_heart_jack :
  let cards := {c : ℕ // c < 52} in
  let queens := {q : cards // q.val / 4 = 11} in -- 4 queens
  let hearts := {h : cards // h.val % 13 = 11} in -- 13 hearts
  let jacks := {j : cards // j.val / 4 = 10} in -- 4 jacks
  let is_queen := λ c, c.val / 4 = 11 in
  let is_heart := λ c, c.val % 13 = 11 in
  let is_jack := λ c, c.val / 4 = 10 in
  ( ∑ (h : cards) in hearts, ∑ (j : cards) in jacks,
      if (queens.val - 1).val ∈ {t | t ∈ cards}
      then (∑ (c : cards) in cards, 
        if is_queen c ∧ is_heart h ∧ is_jack j 
        then 1 else 0) else 0) / 
  (52 * 51 * 50) = 1 / 663 := 
sorry

end probability_queen_heart_jack_l93_93436


namespace number_of_whole_numbers_between_cubicroots_l93_93739

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93739


namespace total_cookies_l93_93282

theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l93_93282


namespace smallest_composite_no_prime_factors_lt_20_l93_93159

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93159


namespace smallest_composite_no_prime_factors_less_than_20_l93_93143

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93143


namespace part_a_impossible_part_b_possible_l93_93560

-- Part (a)
theorem part_a_impossible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ¬ ∀ (x : ℝ), (1 < x ∧ x < a) ∧ (a < 2*x ∧ 2*x < a^2) :=
sorry

-- Part (b)
theorem part_b_possible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ∃ (x : ℝ), (a < 2*x ∧ 2*x < a^2) ∧ ¬ (1 < x ∧ x < a) :=
sorry

end part_a_impossible_part_b_possible_l93_93560


namespace inequality_example_l93_93525

theorem inequality_example (a b m : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_m_pos : 0 < m) (h_ba : b = 2) (h_aa : a = 1) :
  (b + m) / (a + m) < b / a :=
sorry

end inequality_example_l93_93525


namespace problem_correct_statement_l93_93624

variables (a b : Line) (α : Plane)

-- Definitions for parallelism and containment
def parallel (x y : Line) : Prop := -- Fill in definition for parallel lines
def contains (x : Line) (α : Plane) : Prop := -- Fill in definition for a line contained in a plane

-- The correct statement according to the given problem and solution
theorem problem_correct_statement (h1 : parallel a b) (h2 : contains b α) :
  -- statement to be proved
  (∀ c : Line, contains c α → parallel a c) := 
  sorry

end problem_correct_statement_l93_93624


namespace smallest_composite_no_prime_factors_less_than_20_l93_93012

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93012


namespace arithmetic_mean_geometric_mean_ratio_l93_93772

theorem arithmetic_mean_geometric_mean_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a + b) / 2 = 3 * Real.sqrt (a * b)) :
  Int.ofNat (Real.floor ((a / b) + 0.5)) = 34 :=
by
  sorry

end arithmetic_mean_geometric_mean_ratio_l93_93772


namespace nonnegative_diff_roots_eq_8sqrt2_l93_93450

noncomputable def roots_diff (a b c : ℝ) : ℝ :=
  if h : b^2 - 4*a*c ≥ 0 then 
    let root1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
    let root2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
    abs (root1 - root2)
  else 
    0

theorem nonnegative_diff_roots_eq_8sqrt2 : 
  roots_diff 1 42 409 = 8 * Real.sqrt 2 :=
sorry

end nonnegative_diff_roots_eq_8sqrt2_l93_93450


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93749

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93749


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93748

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93748


namespace percentage_of_non_honda_red_cars_l93_93944

/-- 
Total car population in Chennai is 9000.
Honda cars in Chennai is 5000.
Out of every 100 Honda cars, 90 are red.
60% of the total car population is red.
Prove that the percentage of non-Honda cars that are red is 22.5%.
--/
theorem percentage_of_non_honda_red_cars 
  (total_cars : ℕ) (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) (total_red_ratio : ℚ) 
  (h : total_cars = 9000) 
  (h1 : honda_cars = 5000) 
  (h2 : red_honda_ratio = 90 / 100) 
  (h3 : total_red_ratio = 60 / 100) : 
  (900 / (9000 - 5000) * 100 = 22.5) := 
sorry

end percentage_of_non_honda_red_cars_l93_93944


namespace algorithm_effective_expected_residents_l93_93490

-- Define the parameters and conditions
def mailboxes : Type := Fin 80
def initial_mailbox : mailboxes := 37

-- Part (a): Prove that the algorithm is effective
theorem algorithm_effective (k : mailboxes -> mailboxes) (h_random : random_permutation k) :
  ∃ n, (k^[n] initial_mailbox) = initial_mailbox :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

-- Part (b): Find the expected number of cycles
noncomputable def harmonic_number (n : ℕ) : ℝ :=
  (finset.range (nat.succ n)).sum (λ i, 1 / (i + 1))

theorem expected_residents (n : ℕ) (h_key_count : n = 80) :
  ∑ k in finset.range n, (1 : ℝ) / (k+1) = harmonic_number 80 :=
begin
  -- This would involve the proof steps, but we skip it here
  sorry
end

end algorithm_effective_expected_residents_l93_93490


namespace boat_upstream_time_l93_93503

-- Define the basic parameters for the problem
variables (B C : ℝ) -- B is the boat's speed in still water, C is the current's speed
variables (D U : ℝ) -- D is the effective downstream speed, U is the effective upstream speed
variables (d : ℝ) -- d is the distance

-- Given conditions
def ratio_condition : Prop := B = 4 * C
def downstream_time_condition : Prop := d = 10 * (B + C)

-- Desired result
def upstream_time_condition : Prop := d / (B - C) = 50 / 3

theorem boat_upstream_time :
  ratio_condition B C ∧ downstream_time_condition B C d →
  upstream_time_condition B C d :=
by
  intros h,
  sorry

end boat_upstream_time_l93_93503


namespace volume_hemisphere_in_cone_l93_93405

noncomputable def volume_of_hemisphere (h : ℝ) (l : ℝ) : ℝ :=
  let R := Math.sqrt (l^2 - h^2)
  let r := Math.sqrt ((R^2 * l^2) / (R^2 + (h - R)^2))
  let V := (2/3) * Real.pi * r^3
  V

theorem volume_hemisphere_in_cone :
  volume_of_hemisphere 4 5 = (1152 / 125) * Real.pi :=
by
  sorry

end volume_hemisphere_in_cone_l93_93405


namespace watermelon_sorting_l93_93892

theorem watermelon_sorting (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 ≠ a2) (h2 : a1 ≠ a3) (h3 : a1 ≠ a4)
  (h4 : a2 ≠ a3) (h5 : a2 ≠ a4) (h6 : a3 ≠ a4) :
  ∃ (order : list ℝ), 
    order = [a_1, a_2, a_3, a_4].sort 
    ∧ order.length = 4
    ∧ (∀ i j : ℕ, i < j → order[i] < order[j])
    ∧ (number_of_weighings : ℕ)
    ∧ number_of_weighings ≤ 5 :=
sorry

end watermelon_sorting_l93_93892


namespace total_hats_purchased_l93_93447

theorem total_hats_purchased (B G : ℕ) (h1 : G = 38) (h2 : 6 * B + 7 * G = 548) : B + G = 85 := 
by 
  sorry

end total_hats_purchased_l93_93447


namespace prob_5_shots_expected_number_shots_l93_93954

variable (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

def prob_exactly_five_shots : ℝ := 6 * p^3 * (1 - p)^2
def expected_shots : ℝ := 3 / p

theorem prob_5_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the probability of exactly 5 shots needed is as calculated
  prob_exactly_five_shots p = 6 * p^3 * (1 - p)^2 :=
by
  sorry

theorem expected_number_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the expected number of shots to hit all targets is as calculated
  expected_shots p = 3 / p :=
by
  sorry

end prob_5_shots_expected_number_shots_l93_93954


namespace whole_numbers_count_between_cubic_roots_l93_93714

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93714


namespace total_observations_after_inclusion_l93_93387

theorem total_observations_after_inclusion (S : ℕ) (N : ℕ) (new_obs : ℕ) (old_avg : ℕ) (new_avg_decrease : ℕ) 
  (h1 : S / 6 = old_avg) (h2 : new_obs = 6) (h3 : old_avg - new_avg_decrease = 12) (h4 : new_avg_decrease = 1) : 
  N = 7 :=
by
  have h5 : old_avg = 13 := by sorry
  have h6 : S = 78 := by sorry
  have h7 : new_obs = 6 := h2
  have new_avg := 12
  have sum_with_new_obs := S + new_obs
  have total_observations := N
  have h8 : sum_with_new_obs / total_observations = new_avg := by sorry
  have h9 : 84 / total_observations = 12 := by sorry
  have h10 : total_observations = 7 := by sorry
  exact h10

end total_observations_after_inclusion_l93_93387


namespace garden_roller_area_l93_93391

theorem garden_roller_area (D : ℝ) (A : ℝ) (π : ℝ) (L_new : ℝ) :
  D = 1.4 → A = 88 → π = 22/7 → L_new = 4 → A = 5 * (2 * π * (D / 2) * L_new) :=
by sorry

end garden_roller_area_l93_93391


namespace gary_chickens_l93_93179

theorem gary_chickens (initial_chickens : ℕ) (multiplication_factor : ℕ) 
  (weekly_eggs : ℕ) (days_in_week : ℕ)
  (h1 : initial_chickens = 4)
  (h2 : multiplication_factor = 8)
  (h3 : weekly_eggs = 1344)
  (h4 : days_in_week = 7) :
  (weekly_eggs / days_in_week) / (initial_chickens * multiplication_factor) = 6 :=
by
  sorry

end gary_chickens_l93_93179


namespace number_drawn_from_first_group_l93_93446

theorem number_drawn_from_first_group : ∃ x : ℕ, x = 6 ∧ (225 + x = 231) :=
by 
  use 6
  split
  rfl
  sorry

end number_drawn_from_first_group_l93_93446


namespace whole_numbers_between_cubicroots_l93_93705

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93705


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93743

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93743


namespace pastries_sold_l93_93168

variable (num_cupcakes num_cookies num_leftover num_sold : ℕ)

axiom cupcakes_baked : num_cupcakes = 7
axiom cookies_baked : num_cookies = 5
axiom pastries_left : num_leftover = 8

theorem pastries_sold : num_sold = num_cupcakes + num_cookies - num_leftover := by
  rw [cupcakes_baked, cookies_baked, pastries_left]
  sorry

end pastries_sold_l93_93168


namespace solve_diophantine_equations_l93_93593

theorem solve_diophantine_equations :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    a * b - 2 * c * d = 3 ∧
    a * c + b * d = 1 } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end solve_diophantine_equations_l93_93593


namespace part_a_part_b_l93_93973

variable (p : ℝ)
variable (h_pos : 0 < p)
variable (h_prob : p ≤ 1)

theorem part_a :
  let q := 1 - p in
  ∃ f : ℕ → ℝ, f 5 = 6 * p^3 * q^2 :=
  by
    sorry

theorem part_b :
  ∃ f : ℕ → ℝ, f 3 = 3 / p :=
  by
    sorry

end part_a_part_b_l93_93973


namespace equilateral_triangle_dot_product_l93_93199

-- Define the vectors and conditions
variables (P B C A : Type)
variable [InnerProductSpace ℝ P]
variable [InnerProductSpace ℝ B]
variable [InnerProductSpace ℝ C]
variable [InnerProductSpace ℝ A]

-- Conditions: equilateral triangle ABC with side length 2
variables 
  (a b : A)
  (t : ℝ)
  (AP AB AC BC : A)
  (side_length : ℝ := 2)
  (is_equilateral : AB = a ∧ AC = b ∧ BC = b - a ∧ a.dot a = side_length^2 ∧ b.dot b = side_length^2 ∧ a.dot b = side_length)

-- Define vector AP in terms of vectors a and b
def vec_AP (a b : A) (t : ℝ) : A :=
  (1 - t) • a + t • b

-- Prove the required dot product
theorem equilateral_triangle_dot_product (a b : A) (t : ℝ) (P B C A : Type)
  [InnerProductSpace ℝ A]
  (side_length : ℝ)
  (is_equilateral : AB = a ∧ AC = b ∧ BC = b - a ∧ a.dot a = side_length^2 ∧ b.dot b = side_length^2 ∧ a.dot b = side_length) :
  (vec_AP a b t) ⋅ (a + b) = 6 :=
by
  sorry

end equilateral_triangle_dot_product_l93_93199


namespace point_of_tangency_range_of_a_l93_93230

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- 1. Prove the x-coordinate of the point of tangency for the tangent line passing through the origin is equal to e
theorem point_of_tangency (a : ℝ) : 
  ∃ x₀, x₀ = Real.exp 1 ∧ f a x₀ - x₀ * (a - (1 / x₀)) + (Real.log x₀) = 0 := 
sorry

-- 2. Prove the range of values for the real number a such that 
-- ∀ x ∈ [1, +∞), f(x) ≥ a(2x - x^2)
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici 1, f a x ≥ a * (2 * x - x^2)) ↔ 1 ≤ a := 
sorry

end point_of_tangency_range_of_a_l93_93230


namespace smallest_composite_no_prime_factors_less_than_20_l93_93113

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93113


namespace correct_conclusions_count_l93_93647

theorem correct_conclusions_count :
  (¬ (¬ p → (q ∨ r)) ↔ (¬ p → ¬ q ∧ ¬ r)) = false ∧
  ((¬ p → q) ↔ (p → ¬ q)) = false ∧
  (¬ ∃ n : ℕ, n > 0 ∧ (n ^ 2 + 3 * n) % 10 = 0 ∧ (∀ n : ℕ, n > 0 → (n ^ 2 + 3 * n) % 10 ≠ 0)) = true ∧
  (¬ ∀ x, x ^ 2 - 2 * x + 3 > 0 ∧ (∃ x, x ^ 2 - 2 * x + 3 < 0)) = false :=
by
  sorry

end correct_conclusions_count_l93_93647


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93080

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93080


namespace max_of_3x_plus_4y_l93_93242

theorem max_of_3x_plus_4y (x y : ℝ) (h : x^2 + y^2 = 2) : 
  ∃ (z : ℝ), z = 3 * x + 4 * y ∧ z ≤ 5 * real.sqrt 2 :=
sorry

end max_of_3x_plus_4y_l93_93242


namespace convex_quadrilateral_possible_sums_l93_93985

noncomputable def sum_of_possible_p : ℕ :=
  let ef : ℕ := 24 in
  let angles : ℕ := 45 in
  let sides := [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 12, 10, 9, 6, 3] in
  sides.sum

theorem convex_quadrilateral_possible_sums :
  let ef : ℕ := 24 in
  let angles : ℕ := 45 in
  let sides := [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 12, 10, 9, 6, 3] in
  sides.sum = 330 :=
by
  sorry

end convex_quadrilateral_possible_sums_l93_93985


namespace intersection_M_N_l93_93658

open Set -- Use the Set module from the library

def M : Set ℝ := { x | log 2 x < 2 }
def N : Set ℤ := { -1, 0, 1, 2 }

theorem intersection_M_N : M ∩ (↑N : Set ℝ) = {1, 2} := 
  sorry

end intersection_M_N_l93_93658


namespace number_of_whole_numbers_between_cubicroots_l93_93742

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93742


namespace positive_sum_l93_93350

noncomputable def a (n : ℕ) : ℝ := (1 / n) * Real.sin (n * Real.pi / 25)
noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))

theorem positive_sum :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 50 → S n > 0) :=
by
  sorry

end positive_sum_l93_93350


namespace polynomial_average_k_l93_93637

theorem polynomial_average_k (h : ∀ x : ℕ, x * (36 / x) = 36 → (x + (36 / x) = 37 ∨ x + (36 / x) = 20 ∨ x + (36 / x) = 15 ∨ x + (36 / x) = 13 ∨ x + (36 / x) = 12)) :
  (37 + 20 + 15 + 13 + 12) / 5 = 19.4 := by
sorry

end polynomial_average_k_l93_93637


namespace whole_numbers_between_cuberoots_l93_93683

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93683


namespace smallest_composite_no_prime_factors_less_than_20_l93_93032

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93032


namespace whole_numbers_between_cuberoots_l93_93691

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93691


namespace MN_passes_through_center_l93_93461

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∃ O : Point, is_circle O A B C ∧ is_circle O A C D

noncomputable def perpendicular (p1 p2 p3 : Point) : Prop :=
  ∃ L : Line, is_on_line p1 L ∧ is_on_line p2 L ∧ is_perpendicular p3 L

theorem MN_passes_through_center 
(A B C D M N : Point) 
(h_cyclic: cyclic_quadrilateral A B C D)
(h_perpendicular_M: perpendicular A B M)
(h_perpendicular_N: perpendicular A D N):
  ∃ O : Point, is_center O A B C D :=
sorry

end MN_passes_through_center_l93_93461


namespace probability_exactly_5_shots_expected_number_of_shots_l93_93977

-- Define the problem statement and conditions with Lean definitions
variables (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

-- Part (a): Probability of 5 shots needed
theorem probability_exactly_5_shots : 
  (6 * p^3 * (1 - p)^2) = probability_exactly_5_shots (p : ℝ) :=
sorry

-- Part (b): Expected number of shots needed
theorem expected_number_of_shots :
  (3 / p) = expected_number_of_shots (p : ℝ) :=
sorry

end probability_exactly_5_shots_expected_number_of_shots_l93_93977


namespace find_a_l93_93633

-- Define the conditions for the lines l1 and l2
def line1 (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def line2 (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - (3/2) = 0

-- Define the condition for parallel lines
def parallel (a : ℝ) : Prop := a^2 - 4 * ((3/4) * a + 1) = 0 ∧ 4 * (-3/2) - 6 * a ≠ 0

-- Define the condition for perpendicular lines
def perpendicular (a : ℝ) : Prop := a * ((3/4) * a + 1) + 4 * a = 0

-- The theorem to prove values of a for which l1 is parallel or perpendicular to l2
theorem find_a (a : ℝ) :
  (parallel a → a = 4) ∧ (perpendicular a → a = 0 ∨ a = -20/3) :=
by
  sorry

end find_a_l93_93633


namespace angle_between_vectors_l93_93250

variables {a b : EuclideanSpace ℝ (Fin 3)}

theorem angle_between_vectors
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 1)
  (hineq : ∀ (x : ℝ), ∥a + x • b∥ ≥ ∥a + b∥) :
  real.angle a b = 2 * real.pi / 3 :=
sorry

end angle_between_vectors_l93_93250


namespace smallest_composite_no_prime_factors_less_than_20_l93_93135

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93135


namespace incorrect_locus_proof_l93_93457

-- Conditions given in the problem
def condition_A (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus ↔ conditions p)

def condition_B (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (conditions p ↔ p ∈ locus)

def condition_C (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus → conditions p) ∧ (∃ q, conditions q ∧ q ∈ locus)

def condition_D (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (p ∈ locus ↔ conditions p)

def condition_E (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (conditions p ↔ p ∈ locus) ∧ (¬ conditions p ↔ p ∉ locus)

-- Statement to be proved
theorem incorrect_locus_proof (locus : Set Point) (conditions : Point → Prop) :
  ¬ condition_C locus conditions :=
sorry

end incorrect_locus_proof_l93_93457


namespace average_age_of_guardians_and_fourth_graders_l93_93386

theorem average_age_of_guardians_and_fourth_graders (num_fourth_graders num_guardians : ℕ)
  (avg_age_fourth_graders avg_age_guardians : ℕ)
  (h1 : num_fourth_graders = 40)
  (h2 : avg_age_fourth_graders = 10)
  (h3 : num_guardians = 60)
  (h4 : avg_age_guardians = 35)
  : (num_fourth_graders * avg_age_fourth_graders + num_guardians * avg_age_guardians) / (num_fourth_graders + num_guardians) = 25 :=
by
  sorry

end average_age_of_guardians_and_fourth_graders_l93_93386


namespace original_number_l93_93771

theorem original_number (x : ℝ) (h : x - x / 3 = 36) : x = 54 :=
by
  sorry

end original_number_l93_93771


namespace carpet_percentage_covered_l93_93983

-- Definitions of the conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def living_room_area : ℝ := 48

-- Area of the carpet
def carpet_area : ℝ := carpet_length * carpet_width

-- Percentage of the living room floor covered by the carpet
def percentage_covered : ℝ := (carpet_area / living_room_area) * 100

-- The theorem we want to prove
theorem carpet_percentage_covered : percentage_covered = 75 := 
by 
  sorry

end carpet_percentage_covered_l93_93983


namespace total_travel_time_l93_93583

theorem total_travel_time (distance1 distance2 speed time1: ℕ) (h1 : distance1 = 100) (h2 : time1 = 1) (h3 : distance2 = 300) (h4 : speed = distance1 / time1) :
  (time1 + distance2 / speed) = 4 :=
by
  sorry

end total_travel_time_l93_93583


namespace curve_is_hyperbola_l93_93875

-- Define the given polar equation as a predicate
def polar_eq (ρ θ : ℝ) : Prop :=
  ρ^2 * (Real.cos (2 * θ)) - 2 * ρ * (Real.cos θ) = 1

-- The main theorem stating that the curve is a hyperbola
theorem curve_is_hyperbola :
  ∃ F : ℝ × ℝ → Prop, (∀ ρ θ, polar_eq ρ θ → F(ρ * Real.cos θ, ρ * Real.sin θ)) 
  ∧ (∀ x y, F (x, y) ↔ (x - 1)^2 - y^2 = 2) : 
  -- Therefore, the curve represented by the given polar equation is a hyperbola.
  (∃ x y, F (x, y)) := 
sorry

end curve_is_hyperbola_l93_93875


namespace minimum_value_fraction_l93_93270

theorem minimum_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 1 ∧ (x = 2/3 ∧ y = 1/3) ∧ 
     ∀ z w, 0 < z ∧ 0 < w ∧ z + w = 1 → (z, w) ≠ (2/3, 1/3) 
           → (4 / (z+2) + 1 / (w+1) > 5/4)) ->
     (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 1 ∧ (x = 2/3 ∧ y = 1/3) 
     ∧ -2 + (4 / (x+2) + 1 / (y+1)) = 1/4) :=
begin
  sorry
end

end minimum_value_fraction_l93_93270


namespace graph_shifted_by_3_is_B_l93_93559

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Assuming f(x) is defined as 0 outside of given intervals for simplicity.

theorem graph_shifted_by_3_is_B : 
  (∀ x, y = f(x) + 3 ↔ graph_B(x, y)) :=
sorry

end graph_shifted_by_3_is_B_l93_93559


namespace perfect_square_trinomial_m_l93_93264

theorem perfect_square_trinomial_m :
  ∀ (m : ℚ), (∃ (a b : ℚ), (4 * (x:ℚ)^2 - (2 * m + 1) * x + 121 = (a * x + b) ^ 2)) ↔ (m = 43 / 2 ∨ m = -45 / 2) := 
begin
  sorry
end

end perfect_square_trinomial_m_l93_93264


namespace total_share_amount_l93_93931

theorem total_share_amount (x y z : ℝ) (hx : y = 0.45 * x) (hz : z = 0.30 * x) (hy_share : y = 63) : x + y + z = 245 := by
  sorry

end total_share_amount_l93_93931


namespace value_of_omega_cannot_be_1_l93_93641

noncomputable def problem_statement : Prop :=
  ∃ (α β : ℝ) (ω : ℝ), π ≤ α ∧ α < β ∧ β ≤ 2 * π ∧ ω > 0 ∧ (sin (5 * π + ω * α) + cos (π / 2 + ω * β) + 2 = 0)

theorem value_of_omega_cannot_be_1 :
  ¬ ∃ (α β : ℝ), π ≤ α ∧ α < β ∧ β ≤ 2 * π ∧ (sin (5 * π + α) + cos (π / 2 + β) + 2 = 0) :=
sorry

end value_of_omega_cannot_be_1_l93_93641


namespace common_tangent_rational_slope_l93_93831

def is_tangent (L : ℝ → ℝ) (P : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y, L y = P x ∧ (𝒟_p (P x) = L y) -- L is tangent to P at x

def parabola1 (x : ℝ) : ℝ := x^2 + 99/100
def parabola2 (y : ℝ) : ℝ := y^2 + 49/4

def gcd_three (a b c : ℕ) : ℕ := nat.gcd a (nat.gcd b c)

theorem common_tangent_rational_slope :
  ∃ a b c : ℕ, (∀ x y : ℝ, is_tangent (λ y, a*x + b*y - c) parabola1 x ∧ is_tangent (λ y, a*x + b*y - c) parabola2 y) ∧
  gcd_three a b c = 1 ∧
  a + b + c = 12 :=
begin
  sorry
end

end common_tangent_rational_slope_l93_93831


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93063

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93063


namespace maximize_profit_l93_93384

def profit_function (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 60 then
    -0.5 * x^2 + 5 * x - 4
  else if x ≥ 60 then
    -x - 81 / x + 55 / 2
  else
    0  -- handle x out of specified domain

theorem maximize_profit :
  (∀ x, profit_function x ≤ 9.5) ∧ (profit_function 9 = 9.5) :=
by
  sorry

end maximize_profit_l93_93384


namespace range_of_a_l93_93183

def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (3 - a) * x - 4 * a
  else Real.log x / Real.log a

-- Proving that f(x) is increasing for 1 < a < 3
theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1 < a ∧ a < 3) := 
by 
  sorry

end range_of_a_l93_93183


namespace inequality_problem_l93_93876

-- Define a and the condition that expresses the given problem as an inequality
variable (a : ℝ)

-- The inequality to prove
theorem inequality_problem : a - 5 > 2 * a := sorry

end inequality_problem_l93_93876


namespace part_a_part_b_l93_93972

variable (p : ℝ)
variable (h_pos : 0 < p)
variable (h_prob : p ≤ 1)

theorem part_a :
  let q := 1 - p in
  ∃ f : ℕ → ℝ, f 5 = 6 * p^3 * q^2 :=
  by
    sorry

theorem part_b :
  ∃ f : ℕ → ℝ, f 3 = 3 / p :=
  by
    sorry

end part_a_part_b_l93_93972


namespace soft_drink_calories_l93_93844

-- Definitions for the conditions
def recommended_intake : ℕ := 150
def exceeded_intake : ℕ := 300
def candy_bars : ℕ := 7
def sugar_per_candy_bar : ℕ := 25
def total_candy_sugar_intake : ℕ := candy_bars * sugar_per_candy_bar
def total_soft_drink_sugar_intake : ℕ := exceeded_intake - total_candy_sugar_intake

-- The percentage of the soft drink's calories that were from added sugar
def sugar_percentage : ℝ := 0.05

-- Prove that the total calories of the soft drink are 2500
theorem soft_drink_calories : ∃ (calories : ℕ), (sugar_percentage * calories.toReal = total_soft_drink_sugar_intake.toReal) ∧ (calories = 2500) :=
by
  sorry

end soft_drink_calories_l93_93844


namespace mat_pow_four_eq_l93_93554

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, -2; 1, 1]

def mat_fourth_power : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-14, -6; 3, -17]

theorem mat_pow_four_eq :
  mat ^ 4 = mat_fourth_power :=
by
  sorry

end mat_pow_four_eq_l93_93554


namespace trapezoid_area_l93_93812

variables (PQ RS PR QS PT : Type)
variables [decidable_eq PQ] [decidable_eq RS] [decidable_eq PR] [decidable_eq QS] [decidable_eq PT]

theorem trapezoid_area
  (PQRS : PQ)
  (PQT_area : ℝ)
  (PST_area : ℝ)
  (PQ_parallel_RS : PQ → RS) 
  (diagonals_intersect_at_T : PR → QS → PT)
  (hPQT : PQT_area = 75)
  (hPST : PST_area = 45) :
  PQRS = 192 :=
sorry

end trapezoid_area_l93_93812


namespace arithmetic_progression_conditions_l93_93380

theorem arithmetic_progression_conditions (n : ℕ) (a d : ℕ) :
  (∀ k, k < n → (a + k * d) % 2 = 0) →  -- progression starts with even number
  (∑ i in range(n) if i % 2 = 0 then a + i * d else 0) = 33 →  -- sum of odd terms
  (∑ i in range(n) if i % 2 = 1 then a + i * d else 0) = 44 →  -- sum of even terms
  ((n = 7 ∧ a = 2 ∧ d = 3) ∨ (n = 7 ∧ a = 8 ∧ d = 1)) := sorry

end arithmetic_progression_conditions_l93_93380


namespace mean_cars_sold_l93_93507

open Rat

theorem mean_cars_sold :
  let monday := 8
  let tuesday := 3
  let wednesday := 10
  let thursday := 4
  let friday := 4
  let saturday := 4
  let total_days := 6
  let total_cars := monday + tuesday + wednesday + thursday + friday + saturday
  let mean := total_cars / total_days
  mean = 33 / 6 :=
by
  let monday := 8
  let tuesday := 3
  let wednesday := 10
  let thursday := 4
  let friday := 4
  let saturday := 4
  let total_days := 6
  let total_cars := monday + tuesday + wednesday + thursday + friday + saturday
  have h1 : total_cars = 8 + 3 + 10 + 4 + 4 + 4 := rfl
  have h2 : total_cars = 33 := by norm_num
  let mean := total_cars / total_days
  have h3 : mean = 33 / 6 := by norm_num
  exact h3

end mean_cars_sold_l93_93507


namespace nonstudent_ticket_price_l93_93439

-- Definitions based on conditions
def student_ticket_price : ℝ := 6
def total_sales : ℝ := 10500
def additional_student_tickets : ℝ := 250
def number_of_each_ticket : ℝ := 850

-- Define the problem in Lean
theorem nonstudent_ticket_price :
  let nonstudent_ticket_price := (total_sales - (number_of_each_ticket + additional_student_tickets) * student_ticket_price) / number_of_each_ticket
  in nonstudent_ticket_price = 4.59 :=
by
  -- We skip the proof
  sorry

end nonstudent_ticket_price_l93_93439


namespace find_key_effective_expected_residents_to_use_algorithm_l93_93468

-- Define mailboxes and keys
def num_mailboxes : ℕ := 80
def initial_mailbox : ℕ := 37

-- Prove that the algorithm is effective
theorem find_key_effective : 
  ∀ (mailboxes : fin num_mailboxes) (keys : fin num_mailboxes), 
  { permutation : list (fin num_mailboxes) // permutation.nodup ∧ permutation.length = num_mailboxes ∧ 
    ∀ m, m ∈ permutation → (if m = initial_mailbox then m else (keys.filter (λ k, k ∈ permutation))) ≠ [] }
  :=
sorry

-- Prove the expected number of residents who will use the algorithm
theorem expected_residents_to_use_algorithm :
  ∑ i in finset.range num_mailboxes, (1 / (i + 1 : ℝ)) = (real.log 80 + 0.577 + (1 / (2 * 80)) : ℝ)
  :=
sorry

end find_key_effective_expected_residents_to_use_algorithm_l93_93468


namespace smallest_composite_no_prime_lt_20_l93_93049

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93049


namespace percentage_mortality_l93_93271

/-- Given that 15% of the members of a population are affected by a disease and 8% of them die,
    prove that the percentage mortality in relation to the entire population is 1.2%. -/
theorem percentage_mortality
  (total_population : ℕ)
  (affected_percentage : ℚ)
  (mortality_percentage_of_affected : ℚ)
  (h1 : affected_percentage = 15 / 100)
  (h2 : mortality_percentage_of_affected = 8 / 100) :
  let mortality_percentage_total := (affected_percentage * mortality_percentage_of_affected) * 100 in
  mortality_percentage_total = 1.2 :=
by
  sorry

end percentage_mortality_l93_93271


namespace count_whole_numbers_between_cubes_l93_93676

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93676


namespace max_traffic_flow_l93_93779

noncomputable def traffic_speed : ℕ → ℚ
| x := if x ≤ 20 then 60
       else if 20 < x ∧ x ≤ 200 then -1/3 * x + 200/3
       else 0

noncomputable def traffic_flow (x : ℕ) : ℚ := x * traffic_speed x

theorem max_traffic_flow :
    (∀ x, 0 ≤ x → x ≤ 200 → traffic_speed x = 
        if x ≤ 20 then 60
        else if 20 < x ∧ x ≤ 200 then -1/3 * x + 200/3 
        else 0) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 200 ∧ ∀ y, 0 ≤ y ∧ y ≤ 200 → traffic_flow y ≤ traffic_flow x) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 200 ∧ traffic_flow x = 3333) := by
    sorry

end max_traffic_flow_l93_93779


namespace algorithm_effective_expected_residents_using_stick_l93_93475

-- Part (a): Prove the algorithm is effective
theorem algorithm_effective :
  ∀ (n : ℕ) (keys : ℕ → ℕ) (start : ℕ),
  (1 ≤ start ∧ start ≤ n) →
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (∃ (sequence : ℕ → ℕ), 
     (sequence 0 = start) ∧ 
     (∀ (i : ℕ), sequence (i + 1) = keys (sequence i)) ∧ 
     (∃ m, sequence m = start)) :=
by
  sorry

-- Part (b): Expected number of residents who need to use the stick is approximately 4.968
-- Here we outline the theorem related to the expected value part.
theorem expected_residents_using_stick (n : ℕ) (H : n = 80) :
  ∀ (keys : ℕ → ℕ),
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (let Hn := ∑ i in range n, 1 / i
   in abs(Hn - 4.968) < ε) :=
by
  sorry

end algorithm_effective_expected_residents_using_stick_l93_93475


namespace sailboat_speed_max_power_l93_93397

-- Define the conditions
def S : ℝ := 7
def v0 : ℝ := 6.3

-- Define the power function N as a function of sailboat speed v
def N (B ρ v : ℝ) : ℝ :=
  (B * S * ρ / 2) * (v0 ^ 2 * v - 2 * v0 * v ^ 2 + v ^ 3)

-- State the theorem we need to prove
theorem sailboat_speed_max_power (B ρ : ℝ) : 
  ∀ (v : ℝ), 
  (∃ v : ℝ, v = 2.1) :=
begin
  -- Proof goes here
  sorry
end

end sailboat_speed_max_power_l93_93397


namespace algorithm_effective_expected_residents_using_stick_l93_93474

-- Part (a): Prove the algorithm is effective
theorem algorithm_effective :
  ∀ (n : ℕ) (keys : ℕ → ℕ) (start : ℕ),
  (1 ≤ start ∧ start ≤ n) →
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (∃ (sequence : ℕ → ℕ), 
     (sequence 0 = start) ∧ 
     (∀ (i : ℕ), sequence (i + 1) = keys (sequence i)) ∧ 
     (∃ m, sequence m = start)) :=
by
  sorry

-- Part (b): Expected number of residents who need to use the stick is approximately 4.968
-- Here we outline the theorem related to the expected value part.
theorem expected_residents_using_stick (n : ℕ) (H : n = 80) :
  ∀ (keys : ℕ → ℕ),
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (let Hn := ∑ i in range n, 1 / i
   in abs(Hn - 4.968) < ε) :=
by
  sorry

end algorithm_effective_expected_residents_using_stick_l93_93474


namespace line_eq_form_l93_93557

def line_equation (x y : ℝ) : Prop :=
  ((3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 3) = 0)

theorem line_eq_form (x y : ℝ) (h : line_equation x y) :
  ∃ (m b : ℝ), y = m * x + b ∧ (m = 3/4 ∧ b = -9/2) :=
by
  sorry

end line_eq_form_l93_93557


namespace arithmetic_mean_of_normal_distribution_l93_93870

theorem arithmetic_mean_of_normal_distribution
  (σ : ℝ) (hσ : σ = 1.5)
  (value : ℝ) (hvalue : value = 11.5)
  (hsd : value = μ - 2 * σ) :
  μ = 14.5 :=
by
  sorry

end arithmetic_mean_of_normal_distribution_l93_93870


namespace smallest_composite_no_prime_factors_lt_20_l93_93156

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93156


namespace find_a_l93_93982

-- Define the price of the small mascot
def price_small := 80

-- Define the price of the large mascot
def price_large := 1.5 * price_small

-- January sales
def january_small_sales := 500
def january_large_sales := 300

-- February sales conditions
def february_small_sales (a : ℕ) := january_small_sales + 10 * a
def february_large_sales := january_large_sales

-- February prices
def february_price_small (a : ℕ) := price_small - a
def february_price_large (a : ℕ) := price_large - a

-- Condition 8: total revenue in February
def total_revenue (a : ℕ) := (february_price_small a) * (february_small_sales a) + (february_price_large a) * february_large_sales

-- Theorem to prove
theorem find_a : ∃ a : ℕ, total_revenue a = 75000 := by
  sorry

end find_a_l93_93982


namespace value_of_stocks_l93_93258

def initial_investment (bonus : ℕ) (stocks : ℕ) : ℕ := bonus / stocks
def final_value_stock_A (initial : ℕ) : ℕ := initial * 2
def final_value_stock_B (initial : ℕ) : ℕ := initial * 2
def final_value_stock_C (initial : ℕ) : ℕ := initial / 2

theorem value_of_stocks 
    (bonus : ℕ) (stocks : ℕ) (h_bonus : bonus = 900) (h_stocks : stocks = 3) : 
    initial_investment bonus stocks * 2 + initial_investment bonus stocks * 2 + initial_investment bonus stocks / 2 = 1350 :=
by
    sorry

end value_of_stocks_l93_93258


namespace tim_kept_amount_l93_93443

-- Definitions as direct conditions
def total_winnings : ℝ := 100
def percentage_given_away : ℝ := 20 / 100

-- The mathematically equivalent proof problem as a theorem statement
theorem tim_kept_amount : total_winnings - (percentage_given_away * total_winnings) = 80 := by
  sorry

end tim_kept_amount_l93_93443


namespace find_RT_length_l93_93857

noncomputable def RT_length (PQ PS : ℝ) (area_triangle area_rectangle : ℝ) : ℝ := 
  (2 * area_rectangle) / PQ

theorem find_RT_length (PQ PS : ℝ) (area_triangle area_rectangle : ℝ)
  (h_dim_PQ : PQ = 4)
  (h_dim_PS : PS = 8)
  (h_area_rectangle : area_rectangle = 32)
  (h_area_eq : area_rectangle = area_triangle)
  (h_share_side : ∃ SR : ℝ, SR = PQ) :
  RT_length PQ PS area_triangle area_rectangle = 16 :=
by
  rw [RT_length, h_dim_PQ, h_area_rectangle]
  unfold RT_length
  rw [mul_comm 2 32]
  rw [div_mul_cancel]
  exact rfl
  exact two_ne_zero
  sorry

end find_RT_length_l93_93857


namespace area_geometric_mean_l93_93381

theorem area_geometric_mean (n : ℕ) (r : ℝ) (h : 0 < r) :
  let t_n := n * r^2 * Real.sin (Real.pi / n) * Real.cos (Real.pi / n)
  let t_2n := n * r^2 * Real.sin (Real.pi / n)
  let T_n := n * r^2 * Real.sin (Real.pi / n) / Real.cos (Real.pi / n)
  in t_2n^2 = t_n * T_n :=
by
  sorry

end area_geometric_mean_l93_93381


namespace complex_quadrant_l93_93801

def z1 := Complex.mk 1 (-2)
def z2 := Complex.mk 2 1
def z := z1 * z2

theorem complex_quadrant : z = Complex.mk 4 (-3) ∧ z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l93_93801


namespace smallest_composite_no_prime_factors_less_than_20_l93_93018

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93018


namespace intersection_point_is_origin_l93_93310

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * (Real.cos θ), ρ * (Real.sin θ))

def line_l (x : ℝ) : ℝ :=
  Real.sqrt 3 * x

def curve_C (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 1 + Real.cos (2 * α))

theorem intersection_point_is_origin :
  ∃ x y : ℝ, (x, y) ∈ (fun α => (2 * Real.cos α, 1 + Real.cos (2 * α))) '' Set.univ ∧ y = Real.sqrt 3 * x ∧ x = 0 ∧ y = 0 :=
sorry

end intersection_point_is_origin_l93_93310


namespace unit_digit_seven_power_500_l93_93451

def unit_digit (x : ℕ) : ℕ := x % 10

theorem unit_digit_seven_power_500 :
  unit_digit (7 ^ 500) = 1 := 
by
  sorry

end unit_digit_seven_power_500_l93_93451


namespace whole_numbers_between_l93_93723

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93723


namespace caramel_price_l93_93410

theorem caramel_price :
  ∃ C : ℝ, 6 * (2 * C) + 3 * C + 4 * C = 57 ∧ C = 3 :=
begin
  use 3,
  split,
  { linarith },
  { refl }
end

end caramel_price_l93_93410


namespace A_for_n_l93_93339

noncomputable def A (n : ℕ) : ℕ :=
  min { k | ∀ k ∈ finset.range n, ∃ S : set (ℝ × ℝ), ∃ l : affine_subspace ℝ (affine_space ℝ), S.card = k ∧ l.inter_points S}

theorem A_for_n (n : ℕ) : 
  A(n) = (n + 1) / 2.floor * (n + 2) / 2.floor :=
sorry

end A_for_n_l93_93339


namespace james_spent_6_dollars_l93_93318

-- Define the constants based on the conditions
def cost_milk : ℝ := 3
def cost_bananas : ℝ := 2
def tax_rate : ℝ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℝ := cost_milk + cost_bananas

-- Define the sales tax
def sales_tax : ℝ := total_cost_before_tax * tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_before_tax + sales_tax

-- The theorem to prove that James spent $6
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l93_93318


namespace mersenne_prime_condition_l93_93948

theorem mersenne_prime_condition (a n : ℕ) (h_a : 1 < a) (h_n : 1 < n) (h_prime : Prime (a ^ n - 1)) : a = 2 ∧ Prime n :=
by
  sorry

end mersenne_prime_condition_l93_93948


namespace finding_a_of_geometric_sequence_l93_93437
noncomputable def geometric_sequence_a : Prop :=
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) ∧ a^2 = 2

theorem finding_a_of_geometric_sequence :
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) → a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
by
  sorry

end finding_a_of_geometric_sequence_l93_93437


namespace smallest_composite_no_prime_lt_20_l93_93053

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93053


namespace smallest_composite_no_prime_factors_less_than_20_l93_93001

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93001


namespace no_solution_l93_93169

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end no_solution_l93_93169


namespace polynomial_div_6_l93_93371

theorem polynomial_div_6 (n : ℕ) : 6 ∣ (2 * n ^ 3 + 9 * n ^ 2 + 13 * n) := 
sorry

end polynomial_div_6_l93_93371


namespace number_of_ways_to_choose_one_book_l93_93363

theorem number_of_ways_to_choose_one_book:
  let chinese_books := 10
  let english_books := 7
  let mathematics_books := 5
  chinese_books + english_books + mathematics_books = 22 := by
    -- The actual proof should go here.
    sorry

end number_of_ways_to_choose_one_book_l93_93363


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93081

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93081


namespace sum_of_squared_residuals_l93_93281

theorem sum_of_squared_residuals (S : ℝ) (r : ℝ) (hS : S = 100) (hr : r = 0.818) : 
    S * (1 - r^2) = 33.0876 :=
by
  rw [hS, hr]
  sorry

end sum_of_squared_residuals_l93_93281


namespace whole_numbers_between_cuberoots_l93_93684

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93684


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93079

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93079


namespace seven_digit_integers_divisible_by_five_l93_93259

theorem seven_digit_integers_divisible_by_five :
  let digits := [2, 2, 2, 5, 5, 9, 9] in
  let is_valid_arrangement (l : List ℕ) := l.length = 7 ∧ l.all (λ d, d ∈ digits) in
  (∃ p : List ℕ, is_valid_arrangement p ∧ p.reverse.head = 5 ∧ p.length = 7) →
  60 = Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 1 * Nat.factorial 2) :=
by
  sorry

end seven_digit_integers_divisible_by_five_l93_93259


namespace whole_numbers_between_cuberoots_l93_93685

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93685


namespace area_sum_eq_l93_93325

variables {α : Type*} [linear_ordered_field α] [ordered_add_comm_group α] [vector_space α α]

-- Define some geometric variables/points and angles
variables (A B C D M N : point α)
variables (S : triangle α α → α)

-- Assume the given angle equality conditions
axiom angle_NAD_eq_MAB (A B C D M N : point α) :
  angle NAD = angle MAB
axiom angle_NBC_eq_MBA (A B C D M N : point α) :
  angle NBC = angle MBA
axiom angle_MCB_eq_NCD (A B C D M N : point α) :
  angle MCB = angle NCD
axiom angle_NDA_eq_MDC (A B C D M N : point α) :
  angle NDA = angle MDC

-- The theorem to be proved
theorem area_sum_eq (A B C D M N : point α) (S : triangle α α → α)
  (h1 : angle NAD = angle MAB)
  (h2 : angle NBC = angle MBA)
  (h3 : angle MCB = angle NCD)
  (h4 : angle NDA = angle MDC) :
  S (triangle.mk A B M) + S (triangle.mk A B N) + S (triangle.mk C D M) + S (triangle.mk C D N) =
  S (triangle.mk B C M) + S (triangle.mk B C N) + S (triangle.mk A D M) + S (triangle.mk A D N) :=
sorry

end area_sum_eq_l93_93325


namespace smallest_composite_no_prime_factors_lt_20_l93_93153

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93153


namespace tournament_min_cost_l93_93828

variables (k : ℕ) (m : ℕ) (S E : ℕ → ℕ)

noncomputable def min_cost (k : ℕ) : ℕ :=
  k * (4 * k^2 + k - 1) / 2

theorem tournament_min_cost (k_pos : 0 < k) (players : m = 2 * k)
  (each_plays_once 
      : ∀ i j, i ≠ j → ∃ d, S d = i ∧ E d = j) -- every two players play once, matches have days
  (one_match_per_day : ∀ d, ∃! i j, i ≠ j ∧ S d = i ∧ E d = j) -- exactly one match per day
  : min_cost k = k * (4 * k^2 + k - 1) / 2 := 
sorry

end tournament_min_cost_l93_93828


namespace find_angle_between_lines_AB_CD_l93_93810

noncomputable def angle_between_lines (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (AB : ℝ) (CD : ℝ) (dist : ℝ) (vol : ℝ) : ℝ := 
  let sinphi := (vol * 3) / (AB * CD * dist / 2)
  in real.arcsin sinphi

theorem find_angle_between_lines_AB_CD :
  let AB := 8
  let CD := 12
  let dist := 6
  let vol := 48
  angle_between_lines ℝ ℝ ℝ ℝ AB CD dist vol = 30 :=
by sorry

end find_angle_between_lines_AB_CD_l93_93810


namespace number_of_elements_in_S_l93_93335

-- Define the function f
def f (x : ℝ) : ℝ := (x + 8) / (2 * x + 1)

-- Define the sequence of functions f_n
noncomputable def f_seq : ℕ → (ℝ → ℝ)
| 0       := f
| (n + 1) := λ x, f (f_seq n x)

-- Define the set S of real numbers x such that f_n(x) = x for some positive integer n
def S : set ℝ := { x | ∃ n : ℕ, n > 0 ∧ (f_seq n) x = x }

-- Statement to prove that the number of elements in S is 2
theorem number_of_elements_in_S : fintype.card S = 2 :=
sorry

end number_of_elements_in_S_l93_93335


namespace num_solutions_l93_93600

theorem num_solutions (k : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + f(y)) = x + y + k) →
  k = 0 →
  ∃! f : ℝ → ℝ, ∀ y : ℝ, f(y) = y :=
by
  sorry

end num_solutions_l93_93600


namespace smallest_composite_no_prime_lt_20_l93_93050

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93050


namespace smallest_composite_no_prime_factors_less_than_20_l93_93015

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93015


namespace container_volume_ratio_l93_93532

theorem container_volume_ratio
  (A B : ℚ)
  (H1 : 3/5 * A + 1/4 * B = 4/5 * B)
  (H2 : 3/5 * A = (4/5 * B - 1/4 * B)) :
  A / B = 11 / 12 :=
by
  sorry

end container_volume_ratio_l93_93532


namespace find_lambda_mu_l93_93180

variable (λ μ : ℝ)

def vector_a (λ : ℝ) : ℝ × ℝ × ℝ := (λ + 1, 0, 2 * λ)
def vector_b (μ : ℝ) : ℝ × ℝ × ℝ := (6, 2 * μ - 1, 2)

def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2, k * b.3)

theorem find_lambda_mu (h : parallel (vector_a λ) (vector_b μ)) : λ = 1 / 5 ∧ μ = 1 / 2 :=
  sorry

end find_lambda_mu_l93_93180


namespace part_I_part_II_l93_93621

-- Definitions based on the problem conditions
variables {a b c k : ℝ}
noncomputable def ellipse_equation (x y : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1
noncomputable def ellipse_vertex := b = 1
noncomputable def ellipse_focal_length := 2 * c = 2 * sqrt 3
noncomputable def point_P := (-2, 1)
noncomputable def line_equation (x y : ℝ) := y - 1 = k * (x + 2)
noncomputable def M (x_B : ℝ) (y_B : ℝ) := (x_B / (1 - y_B), 0)
noncomputable def N (x_C : ℝ) (y_C : ℝ) := (x_C / (1 - y_C), 0)
noncomputable def MN_length (x_B y_B x_C y_C : ℝ) := abs ((x_B / (1 - y_B)) - (x_C / (1 - y_C))) = 2

-- Proof statements
theorem part_I : 
  b = 1 → 
  2 * c = 2 * sqrt 3 → 
  c^2 = a^2 - b^2 → 
  c = sqrt 3 → 
  ellipse_equation 4 1 :=
by sorry

theorem part_II :
  2 * c = 2 * sqrt 3 → 
  b = 1 → 
  c = sqrt 3 →  
  line_equation (-2) 1 = true → 
  (abs ((-(16 * k^2 + 8 * k) / (1 + 4 * k^2)) / (-(1 - k ((-(16 * k^2 + 8 * k) / (1 + 4 * k^2)) + 2))) + (-(16 * k^2 + 8 * k) / (1 + 4 * k^2)) - (abs ((-(16 * k^2 + 8 * k) / (1 + 4 * k^2)) - ((16 * k^2 + 0 * k) / (1 + 4 * k^2))) / (-(c * k ((-(16 * k^2))/ (1 + 4 * k^2)) + 2)) = 2 :=
by sorry

end part_I_part_II_l93_93621


namespace part1_part2_l93_93235

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1 (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, f x a ≥ 4 ↔ x ≤ (3 / 2 : ℝ) ∨ x ≥ (11 / 2 : ℝ) :=
by 
  rw h
  sorry

theorem part2 (h : ∀ x a : ℝ, f x a ≥ 4) :
  ∀ a : ℝ, (a - 1)^2 ≥ 4 ↔ a ≤ -1 ∨ a ≥ 3 :=
by 
  sorry

end part1_part2_l93_93235


namespace am_gm_inequality_l93_93855

theorem am_gm_inequality (n : ℕ) (n_pos : 0 < n) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) : 
  (∏ i, a i) ^ (1 / n : ℝ) ≤ (∑ i, a i) / n :=
begin
  sorry
end

end am_gm_inequality_l93_93855


namespace quadratic_inequality_condition_l93_93617

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) :=
sorry

end quadratic_inequality_condition_l93_93617


namespace stocks_worth_at_year_end_l93_93255

-- Definitions for initial investments
def initial_bonus : ℝ := 900
def investment_A : ℝ := initial_bonus / 3
def investment_B : ℝ := initial_bonus / 3
def investment_C : ℝ := initial_bonus / 3

-- Definitions for the value changes after one year
def value_A_after_one_year : ℝ := 2 * investment_A
def value_B_after_one_year : ℝ := 2 * investment_B
def value_C_after_one_year : ℝ := investment_C / 2

-- Total value after one year
def total_value_after_one_year : ℝ := value_A_after_one_year + value_B_after_one_year + value_C_after_one_year

-- Theorem to prove the total value of stocks at the end of the year
theorem stocks_worth_at_year_end : total_value_after_one_year = 1350 := by
  sorry

end stocks_worth_at_year_end_l93_93255


namespace number_of_solutions_cos2x_plus_3sin2x_eq_1_l93_93262

theorem number_of_solutions_cos2x_plus_3sin2x_eq_1 : 
  ∃ (n : ℕ), n = 48 ∧ ∀ x : ℝ, -30 < x ∧ x < 120 → (cos x)^2 + 3 * (sin x)^2 = 1 → true :=
by
  use 48
  sorry

end number_of_solutions_cos2x_plus_3sin2x_eq_1_l93_93262


namespace part1_part2_part3_l93_93654
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 + real.log x) / (x + 1 - a)

theorem part1 (h : (deriv (λ x, f x a) 1 = 0)) : a = 1 := sorry

theorem part2 : ∃ m, (∀ x ≥ 1, f(x, 1) ≥ m / (x + 1)) ∧ (m ≤ 2) := sorry

theorem part3 :
  real.log 2018 > 2017 - 2 * (∑ k in finset.range 2017, k / (k + 1)) := sorry

end part1_part2_part3_l93_93654


namespace smallest_composite_no_prime_factors_less_than_20_l93_93010

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93010


namespace algorithm_effective_expected_number_of_residents_l93_93486

-- Definitions required from the conditions of the original problem
def num_mailboxes : ℕ := 80

def key_distribution : Equiv.Perm (Fin num_mailboxes) := sorry

def initial_mailbox : Fin num_mailboxes := 37

-- Lean 4 statement for Part (a)
theorem algorithm_effective :
  ∃ m : Fin num_mailboxes, m = initial_mailbox → 
    (fix : ℕ → Fin num_mailboxes)
      (fix 0 = initial_mailbox)
      (∀ n, fix (n+1) = key_distribution (fix n))
      ∃ k, fix k = initial_mailbox := sorry

-- Lean 4 statement for Part (b)
theorem expected_number_of_residents :
  ∀ n, n = num_mailboxes → 
    let Harmonic := λ (n : ℕ), Σ i in Finset.range n, 1 / (i + 1)
    Harmonic n ≈ 4.968 := sorry

end algorithm_effective_expected_number_of_residents_l93_93486


namespace smallest_composite_no_prime_factors_less_than_20_l93_93022

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93022


namespace smallest_composite_no_prime_factors_less_than_20_l93_93101

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93101


namespace negation_equivalent_l93_93456

variable P Q : Prop

theorem negation_equivalent (P Q : Prop) : ¬ (P → ¬ Q) ↔ (P ∧ Q) := by
  sorry

end negation_equivalent_l93_93456


namespace polynomial_solution_exists_l93_93188

open Real

theorem polynomial_solution_exists
    (P : ℝ → ℝ → ℝ)
    (hP : ∃ (f : ℝ → ℝ), ∀ x y : ℝ, P x y = f (x + y) - f x - f y) :
  ∃ (q : ℝ → ℝ), ∀ x y : ℝ, P x y = q (x + y) - q x - q y := sorry

end polynomial_solution_exists_l93_93188


namespace monotonic_intervals_inequality_f_g_l93_93652

def f (x : ℝ) (a : ℝ) := (1/2) * a * x^2 * log x

theorem monotonic_intervals (a : ℝ) (h : 0 < a) :
  ∃ c : ℝ, c = exp (-1/2) ∧ (∀ x ∈ Ioo 0 c, deriv (f x a) < 0) ∧ (∀ x ∈ Ioi c, deriv (f x a) > 0) :=
sorry

theorem inequality_f_g (a : ℝ) (h : 0 < a) (h_min : ∀ x > 0, f x a ≥ -1/(2*real.exp 1)) :
  ∀ x > 0, f x a > x^2 / real.exp x - 3/4 :=
sorry

end monotonic_intervals_inequality_f_g_l93_93652


namespace smallest_composite_no_prime_factors_less_than_20_l93_93140

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93140


namespace rationalize_correct_l93_93375

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, ↑n = x

def rationalize_denominator : ℝ :=
  let expr := 7 / (3 + Real.sqrt 5) * (3 - Real.sqrt 5) / (3 - Real.sqrt 5) in
  expr

theorem rationalize_correct :
  ∃ (A B C D : ℤ), 
    (rationalize_denominator = (A * Real.sqrt B + C) / D) ∧
    D > 0 ∧
    (∀ (p : ℤ), Prime p → ¬ (p ^ 2 ∣ B)) ∧
    Int.gcd (Int.gcd A C) D = 1 ∧
    A + B + C + D = 23 :=
by
  sorry

end rationalize_correct_l93_93375


namespace range_omega_l93_93184

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def f' (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := ω * Real.cos (ω * x + φ)

theorem range_omega (t ω φ : ℝ) (hω_pos : ω > 0) (hf_t_zero : f t ω φ = 0) (hf'_t_pos : f' t ω φ > 0) (no_min_value : ∀ x, t ≤ x ∧ x < t + 1 → ∃ y, y ≠ x ∧ f y ω φ < f x ω φ) : π < ω ∧ ω ≤ (3 * π / 2) :=
sorry

end range_omega_l93_93184


namespace compute_expression_l93_93272

/- Given log definitions -/
def a : ℝ := Real.log 25
def b : ℝ := Real.log 49

/- Main statement -/
theorem compute_expression : 5^(a/b) + 7^(b/a) = 12 := 
by
  sorry

end compute_expression_l93_93272


namespace smallest_composite_no_prime_factors_less_than_20_l93_93107

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93107


namespace smallest_composite_no_prime_factors_less_than_20_l93_93109

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93109


namespace digit_5_equals_digit_2_in_range_l93_93520

def count_digit_occurrences (d : ℕ) (n m : ℕ) : ℕ :=
  (List.range' n (m - n + 1)).map (λ x, x.digits.count d).sum

theorem digit_5_equals_digit_2_in_range :
  count_digit_occurrences 5 101 599 = count_digit_occurrences 2 101 599 :=
by
  sorry

end digit_5_equals_digit_2_in_range_l93_93520


namespace distance_P_focus_l93_93627

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the coordinates of point P
def P (t : ℝ) : (ℝ × ℝ) := (t, 4)

-- Define the focus of the parabola y^2 = 4x
def focus : (ℝ × ℝ) := (1, 0)

-- Define the property that relates point P and focus F
def PF (t : ℝ) : ℝ := abs (t + 1)

-- Prove that the distance |PF| is 5
theorem distance_P_focus : ∀ t : ℝ, parabola t 4 → PF t = 5 :=
by
  intros t ht,
  sorry

end distance_P_focus_l93_93627


namespace initial_bottle_caps_l93_93378

variable (x : Nat)

theorem initial_bottle_caps (h : x + 3 = 29) : x = 26 := by
  sorry

end initial_bottle_caps_l93_93378


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93085

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93085


namespace perp_intersect_iff_eq_zero_l93_93853

/-- Prove that for the perpendiculars dropped from the points \(A_1\), \(B_1\), 
and \(C_1\) to the sides \(BC\), \(CA\), and \(AB\) of the triangle \(ABC\) 
to intersect at one point, it is necessary and sufficient that:
|A_1 B|^2 - |BC_1|^2 + |C_1 A|^2 - |AB_1|^2 + |B_1 C|^2 - |CA_1|^2 = 0.
--/
theorem perp_intersect_iff_eq_zero (ABC A1 B1 C1 : Type) 
  [metric_space ABC] {dist : ABC → ABC → ℝ}
  (bc : ∀ x y : ABC, x ≠ y → dist x y > 0)
  (eq_condition : dist A1 B ^ 2 - dist B C1 ^ 2 + 
                  dist C1 A ^ 2 - dist A B1 ^ 2 + 
                  dist B1 C ^ 2 - dist C A1 ^ 2 = 0) :
  ∃ M : ABC, is_perpendicular_from_to A1 B C M ∧ 
             is_perpendicular_from_to B1 C A M ∧ 
             is_perpendicular_from_to C1 A B M :=
sorry

end perp_intersect_iff_eq_zero_l93_93853


namespace whole_numbers_between_cuberoots_l93_93692

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93692


namespace angles_BF₁E_BF₂E_eq_30_l93_93813

-- Define the setup for the problem conditions
variables (A B C D E F₁ F₂ : Point)
variables (AC BC : ℝ) (h1 : AC = BC) (h2 : AC > BC)
variables (circle_B : Circle B BC) (circle_D : Circle D BC)
variables (h3 : D ∈ circle_B ∩ AC) (h4 : E ∈ circle_B ∩ AB) 
variables (h5 : E ∈ circle_D ∩ AB) (h6 : F₁ ∈ circle_D ∩ AC) 
variables (h7 : F₂ ∈ circle_D ∩ (extension AC))

-- The proof statement
theorem angles_BF₁E_BF₂E_eq_30 : 
  ∠BF₁E = 30 ∧ ∠BF₂E = 30 := 
by 
  sorry

end angles_BF₁E_BF₂E_eq_30_l93_93813


namespace pet_store_cages_l93_93993

theorem pet_store_cages (init_puppies sold_puppies puppies_per_cage : ℕ)
  (h1 : init_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (init_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l93_93993


namespace exists_l_pow_eleven_eq_l93_93840

open Nat

theorem exists_l_pow_eleven_eq (m n : ℕ) (h : ∀ k : ℕ, gcd (11 * k - 1) m = gcd (11 * k - 1) n) :
  ∃ l : ℤ, m = 11^l * n := 
  sorry

end exists_l_pow_eleven_eq_l93_93840


namespace system1_solution_l93_93861

theorem system1_solution (x y : ℝ) (h1 : 2 * x - y = 1) (h2 : 7 * x - 3 * y = 4) : x = 1 ∧ y = 1 :=
by sorry

end system1_solution_l93_93861


namespace intersection_A_B_at_1_range_of_a_l93_93659

-- Problem definitions
def set_A (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def set_B (x a : ℝ) : Prop := x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0

-- Question (I) If a = 1, find A ∩ B
theorem intersection_A_B_at_1 : (∀ x : ℝ, set_A x ∧ set_B x 1 ↔ (1 < x ∧ x ≤ 1 + Real.sqrt 2)) := sorry

-- Question (II) If A ∩ B contains exactly one integer, find the range of a.
theorem range_of_a (h : ∃ x : ℤ, set_A x ∧ set_B x 2) : 3 / 4 ≤ 2 ∧ 2 < 4 / 3 := sorry

end intersection_A_B_at_1_range_of_a_l93_93659


namespace given_polynomial_l93_93246

noncomputable def f (x : ℝ) := x^3 - 2

theorem given_polynomial (x : ℝ) : 
  8 * f (x^3) - x^6 * f (2 * x) - 2 * f (x^2) + 12 = 0 :=
by
  sorry

end given_polynomial_l93_93246


namespace proj_v_on_w_l93_93333

def v : ℝ × ℝ := (12, -8)
def w : ℝ × ℝ := (-9, 6)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let c := (dot_product u v) / (dot_product v v)
  (c * v.1, c * v.2)

theorem proj_v_on_w :
  projection v w = v :=
by 
  sorry

end proj_v_on_w_l93_93333


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93090

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93090


namespace area_triangle_ABC_l93_93784

theorem area_triangle_ABC 
  (a b c d e f : Point)
  (triangle_ABC : Triangle ABC)
  (H1 : midpoint D (B, C))
  (H2 : on_line E (A, C) ∧ ratio AE EC = 2/3)
  (H3 : on_line F (A, D) ∧ ratio AF FD = 2/1)
  (area_DEF : area_triangle DEF = 24) :
  area_triangle ABC = 360 :=
by 
  sorry

end area_triangle_ABC_l93_93784


namespace solid_is_cone_l93_93277

noncomputable def solid (is_front_view_isosceles_triangle : Prop)
  (is_left_view_isosceles_triangle : Prop)
  (is_top_view_circle : Prop) : Prop :=
  ∃ (cone : Type), is_front_view_isosceles_triangle ∧ 
                   is_left_view_isosceles_triangle ∧ 
                   is_top_view_circle ∧ 
                   cone ≠ ∅ 

theorem solid_is_cone (is_front_view_isosceles_triangle : Prop)
  (is_left_view_isosceles_triangle : Prop)
  (is_top_view_circle : Prop)
  (h : solid is_front_view_isosceles_triangle is_left_view_isosceles_triangle is_top_view_circle) :
  ∃ (cone : Type), solid is_front_view_isosceles_triangle is_left_view_isosceles_triangle is_top_view_circle :=
sorry

end solid_is_cone_l93_93277


namespace necessary_and_sufficient_l93_93248

-- Definitions for the geometric objects and the propositions
variable {α β : Type} [Plane α] [Plane β]
variable (l m : Line)
variable (intersect : Intersect l m)
variable (plane_l : InPlane l α)
variable (plane_m : InPlane m α)

-- Neither l nor m lies within β
axiom l_not_in_beta : ¬ InPlane l β
axiom m_not_in_beta : ¬ InPlane m β

-- Proposition p: At least one of l or m intersects with β
def p : Prop := ∃ (L : Line), (L = l ∨ L = m) ∧ IntersectsWithPlane L β

-- Proposition q: Plane α intersects with plane β
def q : Prop := IntersectsWithPlane α β

-- Proof statement: p is both a necessary and sufficient condition for q
theorem necessary_and_sufficient : p ↔ q := by
  sorry

end necessary_and_sufficient_l93_93248


namespace max_sum_of_digits_of_S_l93_93324

def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def distinctDigits (n : ℕ) : Prop :=
  let digits := (n.digits 10).toFinset
  digits.card = (n.digits 10).length

def digitsRange (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 9

theorem max_sum_of_digits_of_S : ∃ a b S, 
  isThreeDigit a ∧ 
  isThreeDigit b ∧ 
  distinctDigits a ∧ 
  distinctDigits b ∧ 
  digitsRange a ∧ 
  digitsRange b ∧ 
  isThreeDigit S ∧ 
  S = a + b ∧ 
  (S.digits 10).sum = 12 :=
sorry

end max_sum_of_digits_of_S_l93_93324


namespace smallest_composite_no_prime_factors_less_20_l93_93037

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93037


namespace orchid_bushes_planted_today_l93_93427

theorem orchid_bushes_planted_today (current_bushes : ℕ) (tomorrow_bushes : ℕ) (final_bushes : ℕ) :
  current_bushes = 47 →
  tomorrow_bushes = 25 →
  final_bushes = 109 →
  ∃ today_bushes : ℕ, current_bushes + today_bushes + tomorrow_bushes = final_bushes ∧ today_bushes = 37 :=
by
  intros h_current h_tomorrow h_final
  use 37
  simp [h_current, h_tomorrow, h_final]
  linarith
  sorry

end orchid_bushes_planted_today_l93_93427


namespace ratio_of_investments_l93_93935

theorem ratio_of_investments {A B C : ℝ} (x y z k : ℝ)
  (h1 : B - A = 100)
  (h2 : A + B + C = 2900)
  (h3 : A = 6 * k)
  (h4 : B = 5 * k)
  (h5 : C = 4 * k) : 
  (x / y = 6 / 5) ∧ (y / z = 5 / 4) ∧ (x / z = 6 / 4) :=
by
  sorry

end ratio_of_investments_l93_93935


namespace cone_lateral_surface_area_l93_93871

noncomputable def lateral_surface_area (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2) in π * r * l

theorem cone_lateral_surface_area : lateral_surface_area 3 4 = 15 * π := by
  sorry

end cone_lateral_surface_area_l93_93871


namespace number_of_whole_numbers_between_cubicroots_l93_93741

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93741


namespace binomial_coefficient_of_x_in_expansion_l93_93800

open Nat

theorem binomial_coefficient_of_x_in_expansion :
  ∃ (coeff : ℤ), coeff = -40 ∧ (coeff = tsum (λ r, (if (10 - 3 * r = 1) then (-1) ^ r * (choose 5 r) * 2^(5-r) else 0))) :=
by
  sorry

end binomial_coefficient_of_x_in_expansion_l93_93800


namespace volume_of_tetrahedron_alpha_leq_2phi_volume_of_tetrahedron_pi_minus_alpha_leq_2phi_l93_93495

variables {R α φ : ℝ}
variables (α_lt_90 : α < 90)
variables (φ_lt_90 : φ < 90)
variables (α_leq_2φ : α ≤ 2 * φ)
variables (phi_lt_pi_minus_alpha : 2 * φ < π - α)

-- Volume of the tetrahedron ABCD when α ≤ 2φ < π - α
theorem volume_of_tetrahedron_alpha_leq_2phi : 
  α ≤ 2 * φ ∧ 2 * φ < (π - α) → 
    (volume AB CD = 2 * R^3 * tan(α / 2) / 3) := 
by
  sorry

-- Volume of the tetrahedron ABCD when π - α ≤ 2φ < π
theorem volume_of_tetrahedron_pi_minus_alpha_leq_2phi : 
  (π - α) ≤ 2 * φ ∧ 2 * φ < π → 
  (volume AB CD = 2 * R^3 * tan(α / 2) / 3 ∨ volume AB CD = 2 * R^3 * cot(α / 2) / 3) := 
by
  sorry

end volume_of_tetrahedron_alpha_leq_2phi_volume_of_tetrahedron_pi_minus_alpha_leq_2phi_l93_93495


namespace geometric_progression_condition_l93_93610

theorem geometric_progression_condition (a b c : ℝ) (h_b_neg : b < 0) : 
  (b^2 = a * c) ↔ (∃ (r : ℝ), a = r * b ∧ b = r * c) :=
sorry

end geometric_progression_condition_l93_93610


namespace angle_equality_l93_93438

variables {A B C O M D P Q S : Type*} 

-- Define all the points and necessary conditions

def scalene_triangle (A B C : Type*) : Prop :=
A ≠ B ∧ B ≠ C ∧ C ≠ A

def circumcenter (A B C O : Type*) [circumcircle_exists : ∃(O : Type*), true]: Prop :=
true

variables (AB_line AC_line : Type*)
variables [perpendicular_to_AB : ∃ (P Q : Type*), true]

def is_midpoint (M B C : Type*) : Prop :=
true

def center_circumcircle_OPQ (S O P Q : Type*) : Prop :=
true

theorem angle_equality
  (h1 : scalene_triangle A B C)
  (h2 : circumcenter A B C O)
  (h3 : ∃ (AB_perp AC_perp : Type*), perpendicular_to_AB AB_perp ∧ perpendicular_to_AB AC_perp)
  (h4 : ∃ (P Q : Type*), intersects_altitude AD P ∧ intersects_altitude AD Q)
  (h5 : is_midpoint M B C)
  (h6 : center_circumcircle_OPQ S O P Q) :
  ∃ (angle_BAS angle_CAM : Type*), angle_equality angle_BAS angle_CAM :=
by sorry

end angle_equality_l93_93438


namespace fraction_distinctly_marked_l93_93991

theorem fraction_distinctly_marked 
  (area_large_rectangle : ℕ)
  (fraction_shaded : ℚ)
  (fraction_further_marked : ℚ)
  (h_area_large_rectangle : area_large_rectangle = 15 * 24)
  (h_fraction_shaded : fraction_shaded = 1/3)
  (h_fraction_further_marked : fraction_further_marked = 1/2) :
  (fraction_further_marked * fraction_shaded = 1/6) :=
by
  sorry

end fraction_distinctly_marked_l93_93991


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93064

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93064


namespace t_of_polynomial_has_factor_l93_93631

theorem t_of_polynomial_has_factor (t : ℤ) :
  (∃ a b : ℤ, x ^ 3 - x ^ 2 - 7 * x + t = (x + 1) * (x ^ 2 + a * x + b)) → t = -5 :=
by
  sorry

end t_of_polynomial_has_factor_l93_93631


namespace smallest_composite_no_prime_factors_below_20_l93_93128

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93128


namespace max_xy_value_l93_93765

theorem max_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 200) : x * y ≤ 28 :=
begin
  sorry
end

example : ∃ x y : ℕ, 7 * x + 4 * y = 200 ∧ x * y = 28 := 
begin
  use [28, 1],
  split,
  { norm_num, },
  { norm_num, },
end

end max_xy_value_l93_93765


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93065

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93065


namespace right_angles_sum_l93_93550

theorem right_angles_sum
  (rect_park : ℕ)
  (flower_beds_in_park : ℕ)
  (square_field : ℕ)
  (goal_areas_in_field : ℕ)
  (rect_right_angles : rect_park = 4)
  (flower_bed_right_angles : ∀ n, flower_beds_in_park = 3 → n * flower_beds_in_park * rect_park = 12)
  (square_right_angles : square_field = 4)
  (goal_area_right_angles: ∀ n, goal_areas_in_field = 4 → n * goal_areas_in_field * square_field = 16)
  :
  rect_park + 12 + square_field + 16 = 36 :=
begin
  sorry
end

end right_angles_sum_l93_93550


namespace pure_imaginary_condition_l93_93773

-- Definition and theorem that needs to be proven.
theorem pure_imaginary_condition (a : ℝ) : 
  ((a + complex.i) * (1 - 2 * complex.i)).re = 0 ↔ a = -2 :=
by
  sorry

end pure_imaginary_condition_l93_93773


namespace clock_angle_11_am_l93_93539

theorem clock_angle_11_am : 
  let minute_hand := 12,
      hour_hand := 11,
      angle_per_hour := 30 in
  (hour_hand - minute_hand).nat_abs * angle_per_hour = 30 := 
by 
  sorry

end clock_angle_11_am_l93_93539


namespace find_angle_A_maximum_height_of_triangle_l93_93297

noncomputable def triangle_side_condition (a b c A B C : ℝ) : Prop :=
  a - b - c - A - B - C = 0 ∧ 
  sin C = 2 * cos A * sin (B + π / 3)

noncomputable def maximum_height_condition (a b c : ℝ) (A : ℝ) : Prop :=
  b + c = 6 ∧ cos A = 1 / 2 ∧ bc ≤ 9

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle_side_condition a b c A B C) : 
  A = π / 3 :=
sorry

theorem maximum_height_of_triangle (a b c : ℝ) (A : ℝ) 
  (h : maximum_height_condition a b c A) : 
  ∃ (D : ℝ), D = 3 * sqrt 3 / 2 :=
sorry

end find_angle_A_maximum_height_of_triangle_l93_93297


namespace smallest_composite_no_prime_factors_less_than_20_l93_93028

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93028


namespace smallest_composite_no_prime_factors_less_than_20_l93_93118

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93118


namespace smallest_composite_no_prime_factors_less_than_20_l93_93016

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93016


namespace total_volume_of_five_cubes_l93_93924

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end total_volume_of_five_cubes_l93_93924


namespace smallest_composite_no_prime_factors_below_20_l93_93120

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93120


namespace whole_numbers_count_between_cubic_roots_l93_93717

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93717


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93752

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93752


namespace complex_modulus_squared_l93_93347

theorem complex_modulus_squared (w : ℂ) (h : |w| = 15) : w * conj w = 225 :=
by
  sorry

end complex_modulus_squared_l93_93347


namespace complex_conjugate_product_l93_93348

variable (w : ℂ)
variable (h : Complex.abs w = 15)

theorem complex_conjugate_product : w * Complex.conj w = 225 := by
  sorry

end complex_conjugate_product_l93_93348


namespace transformed_function_is_g_l93_93225

def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)

def transformation (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x / 2)

theorem transformed_function_is_g :
  transformation f = λ x, Real.sin (1 / 2 * x + Real.pi / 3) :=
by
  -- we would provide a proof here
  sorry

end transformed_function_is_g_l93_93225


namespace algorithm_effective_expected_number_of_residents_l93_93483

-- Definitions required from the conditions of the original problem
def num_mailboxes : ℕ := 80

def key_distribution : Equiv.Perm (Fin num_mailboxes) := sorry

def initial_mailbox : Fin num_mailboxes := 37

-- Lean 4 statement for Part (a)
theorem algorithm_effective :
  ∃ m : Fin num_mailboxes, m = initial_mailbox → 
    (fix : ℕ → Fin num_mailboxes)
      (fix 0 = initial_mailbox)
      (∀ n, fix (n+1) = key_distribution (fix n))
      ∃ k, fix k = initial_mailbox := sorry

-- Lean 4 statement for Part (b)
theorem expected_number_of_residents :
  ∀ n, n = num_mailboxes → 
    let Harmonic := λ (n : ℕ), Σ i in Finset.range n, 1 / (i + 1)
    Harmonic n ≈ 4.968 := sorry

end algorithm_effective_expected_number_of_residents_l93_93483


namespace smallest_composite_no_prime_factors_below_20_l93_93131

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93131


namespace smallest_composite_no_prime_factors_less_20_l93_93036

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93036


namespace smallest_composite_no_prime_factors_less_20_l93_93045

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93045


namespace symmetric_linear_functions_l93_93663

theorem symmetric_linear_functions :
  (∃ (a b : ℝ), ∀ x y : ℝ, (y = a * x + 2 ∧ y = 3 * x - b) → a = 1 / 3 ∧ b = 6) :=
by
  sorry

end symmetric_linear_functions_l93_93663


namespace present_condition_l93_93781

variable {α : Type} [Finite α]

-- We will represent children as members of a type α and assume there are precisely 3n children.
variable (n : ℕ) (h_odd : odd n) [h : Fintype α] (card_3n : Fintype.card α = 3 * n)

noncomputable def makes_present_to (A B : α) : α := sorry -- Create a function that maps pairs of children to exactly one child.

theorem present_condition : ∀ (A B C : α), makes_present_to A B = C → makes_present_to A C = B :=
sorry

end present_condition_l93_93781


namespace smallest_composite_no_prime_factors_less_than_20_l93_93105

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93105


namespace max_distance_PQRS_l93_93211

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_PQRS (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 0 ≤ y) (h3 : y ≤ x) : 
  ∃ (M : ℝ), M = (1 + real.sqrt 5) / 2 ∧ ∀ (o₁ o₂ : ℝ), distance x y 0 x ≤ M :=
sorry

end max_distance_PQRS_l93_93211


namespace regular_pyramid_volume_l93_93826

-- We define the conditions and expected result in Lean

noncomputable def volume_of_regular_pyramid (S : ℝ) (α : ℝ) : ℝ :=
  S * (real.sqrt S) * (real.sqrt (real.cos α)) * (real.sin α) / (2 * (real.sqrt 2) * (real.sqrt (real.sqrt 27)))

theorem regular_pyramid_volume (S α : ℝ) (hS : S > 0) (hα : 0 < α ∧ α < real.pi / 2) :
  -- Given: a regular pyramid with area S of triangle PA_1A_5 and angle α
  -- Prove: Volume of the pyramid equals the specified expression
  volume_of_regular_pyramid S α =
  S * (real.sqrt S) * (real.sqrt (real.cos α)) * (real.sin α) / (2 * (real.sqrt 2) * (real.sqrt (real.sqrt 27))) :=
sorry

end regular_pyramid_volume_l93_93826


namespace fraction_value_l93_93223

variable (u v w x : ℝ)

-- Conditions
def cond1 : Prop := u / v = 5
def cond2 : Prop := w / v = 3
def cond3 : Prop := w / x = 2 / 3

theorem fraction_value (h1 : cond1 u v) (h2 : cond2 w v) (h3 : cond3 w x) : x / u = 9 / 10 := 
by
  sorry

end fraction_value_l93_93223


namespace smallest_composite_no_prime_factors_less_than_20_l93_93093

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93093


namespace find_minimal_degree_polynomial_l93_93601

theorem find_minimal_degree_polynomial :
  ∃ (p : Polynomial ℚ), p.monic ∧
    p.degree = 4 ∧
    p.eval (2 + Real.sqrt 2) = 0 ∧
    p.eval (2 - Real.sqrt 2) = 0 ∧
    p.eval (2 + Real.sqrt 5) = 0 ∧
    p.eval (2 - Real.sqrt 5) = 0 ∧
    p = Polynomial.C (2 : ℚ) +
        Polynomial.C (-20 : ℚ) * Polynomial.X +
        Polynomial.C (15 : ℚ) * (Polynomial.X ^ 2) +
        Polynomial.C (-4 : ℚ) * (Polynomial.X ^ 3) +
        Polynomial.C (1 : ℚ) * (Polynomial.X ^ 4) :=
by
  sorry

end find_minimal_degree_polynomial_l93_93601


namespace smallest_composite_no_prime_factors_less_than_20_l93_93111

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93111


namespace value_of_y_l93_93454

theorem value_of_y (AB AD CD : ℝ) (y : ℝ) (h1 : is_45_45_90_triangle AB AD 10) 
  (h2 : CD = y) (h3 : AD = 10 * Real.sqrt 2) : y = 10 := 
by 
  sorry

end value_of_y_l93_93454


namespace num_whole_numbers_between_l93_93693

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93693


namespace number_of_whole_numbers_between_cubicroots_l93_93733

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93733


namespace find_sinA_and_b_l93_93284

-- Definitions and conditions
def a : ℝ := 3 * Real.sqrt 2
def c : ℝ := Real.sqrt 3
def cos_C : ℝ := (2 * Real.sqrt 2) / 3

-- Proof statement for the problem
theorem find_sinA_and_b (b : ℝ) (sin_A: ℝ) :
  sin_A = Real.sqrt 6 / 3 ∧ (b < a → b = 3) :=
by
  sorry

end find_sinA_and_b_l93_93284


namespace smallest_composite_no_prime_factors_less_20_l93_93046

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93046


namespace largest_angle_of_obtuse_isosceles_triangle_l93_93904

def triangleXYZ : Type := { XYZ : Type // XYZ = Triangle }
def isosceles_triangle (T : triangleXYZ) : Prop := Isosceles T.val
def obtuse_triangle (T : triangleXYZ) : Prop := Obtuse T.val
def angle_X_30_degrees (T : triangleXYZ) : Prop := Angle T.val X = 30

def largest_angle_measure (T : triangleXYZ) : ℕ := 120

theorem largest_angle_of_obtuse_isosceles_triangle (T : triangleXYZ) 
  (h1 : isosceles_triangle T) 
  (h2 : obtuse_triangle T) 
  (h3 : angle_X_30_degrees T) : 
  Angle T.val (largest_interior_angle T.val) = largest_angle_measure T :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l93_93904


namespace abs_diff_of_two_numbers_l93_93778

theorem abs_diff_of_two_numbers (x y : ℝ) (h_sum : x + y = 42) (h_prod : x * y = 437) : |x - y| = 4 :=
sorry

end abs_diff_of_two_numbers_l93_93778


namespace positive_real_solution_unique_l93_93575

noncomputable def polynomial := (x : ℝ) → x^4 + 8 * x^3 + 28 * x^2 + 2023 * x - 1807

theorem positive_real_solution_unique : ∃! x > 0, polynomial x = 0 := sorry

end positive_real_solution_unique_l93_93575


namespace percentage_of_invalid_votes_l93_93793

theorem percentage_of_invalid_votes:
  ∃ (A B V I VV : ℕ), 
    V = 5720 ∧
    B = 1859 ∧
    A = B + 15 / 100 * V ∧
    VV = A + B ∧
    V = VV + I ∧
    (I: ℚ) / V * 100 = 20 :=
by
  sorry

end percentage_of_invalid_votes_l93_93793


namespace circle_locus_l93_93906

theorem circle_locus (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  13 * a^2 + 49 * b^2 - 12 * a - 1 = 0 := 
sorry

end circle_locus_l93_93906


namespace algorithm_effective_expected_residents_using_stick_l93_93473

-- Part (a): Prove the algorithm is effective
theorem algorithm_effective :
  ∀ (n : ℕ) (keys : ℕ → ℕ) (start : ℕ),
  (1 ≤ start ∧ start ≤ n) →
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (∃ (sequence : ℕ → ℕ), 
     (sequence 0 = start) ∧ 
     (∀ (i : ℕ), sequence (i + 1) = keys (sequence i)) ∧ 
     (∃ m, sequence m = start)) :=
by
  sorry

-- Part (b): Expected number of residents who need to use the stick is approximately 4.968
-- Here we outline the theorem related to the expected value part.
theorem expected_residents_using_stick (n : ℕ) (H : n = 80) :
  ∀ (keys : ℕ → ℕ),
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (let Hn := ∑ i in range n, 1 / i
   in abs(Hn - 4.968) < ε) :=
by
  sorry

end algorithm_effective_expected_residents_using_stick_l93_93473


namespace largest_divisible_by_two_power_l93_93342
-- Import the necessary Lean library

open scoped BigOperators

-- Prime and Multiples calculation based conditions
def primes_count : ℕ := 25
def multiples_of_four_count : ℕ := 25

-- Number of subsets of {1, 2, 3, ..., 100} with more primes than multiples of 4
def N : ℕ :=
  let pow := 2^50
  pow * (pow / 2 - (∑ k in Finset.range 26, Nat.choose 25 k ^ 2))

-- Theorem stating that the largest integer k such that 2^k divides N is 52
theorem largest_divisible_by_two_power :
  ∃ (k : ℕ), (2^k ∣ N) ∧ (∀ m : ℕ, 2^m ∣ N → m ≤ 52) :=
sorry

end largest_divisible_by_two_power_l93_93342


namespace smallest_composite_no_prime_factors_less_than_20_l93_93116

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93116


namespace yellow_jelly_bean_probability_l93_93518

theorem yellow_jelly_bean_probability :
  ∀ (p_red p_orange p_green p_total p_yellow : ℝ),
    p_red = 0.15 →
    p_orange = 0.35 →
    p_green = 0.25 →
    p_total = 1 →
    p_red + p_orange + p_green + p_yellow = p_total →
    p_yellow = 0.25 :=
by
  intros p_red p_orange p_green p_total p_yellow h_red h_orange h_green h_total h_sum
  sorry

end yellow_jelly_bean_probability_l93_93518


namespace cube_orthogonal_pairs_l93_93274

-- Define a cube and the concept of orthogonal line-plane pairs within it
def is_orthogonal_line_plane_pair (l : Line) (p : Plane) : Prop :=
  l.perpendicular p

-- Define the problem statement
theorem cube_orthogonal_pairs : ∀ (cube : Cube),
  let edges := cube.edges,
      face_diagonals := cube.face_diagonals,
      orthogonal_pairs := 2 * edges.count + face_diagonals.count
  in orthogonal_pairs = 36 := sorry

end cube_orthogonal_pairs_l93_93274


namespace sequence_term_formula_l93_93640

theorem sequence_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 * a n - 4) →
  (∀ n, a 1 = 4 ∧ (∀ n > 1, a n = 2 * a (n - 1))) →
  (∀ n, a n = 2^(n+1)) :=
by 
  intro hS hgeom,
  sorry

end sequence_term_formula_l93_93640


namespace max_value_of_f_in_interval_l93_93648

noncomputable def f (x m : ℝ) : ℝ := -x^3 + 3 * x^2 + m

theorem max_value_of_f_in_interval (m : ℝ) (h₁ : ∀ x ∈ [-2, 2], - x^3 + 3 * x^2 + m ≥ 1) : 
  ∃ x ∈ [-2, 2], f x m = 21 :=
by
  sorry

end max_value_of_f_in_interval_l93_93648


namespace fraction_ordering_l93_93914

theorem fraction_ordering :
  (4 / 13) < (12 / 37) ∧ (12 / 37) < (15 / 31) ∧ (4 / 13) < (15 / 31) :=
by sorry

end fraction_ordering_l93_93914


namespace exact_probability_five_shots_l93_93963

theorem exact_probability_five_shots (p : ℝ) (h1 : 0 < p) (h2 : p ≤ 1) :
  (let hit := p
       miss := 1 - p
       comb := 6 in
   comb * hit^3 * miss^2 = 6 * p^3 * (1 - p)^2) :=
by sorry

end exact_probability_five_shots_l93_93963


namespace distinct_ratios_min_elements_l93_93343

theorem distinct_ratios_min_elements :
  ∀ (a : Finₓ 2006 → ℕ) (h_pos : ∀ i, 0 < a i),
    (∀ i j, 0 ≤ i → i < 2005 → 0 ≤ j → j < 2005 → (a i / a (i+1)) ≠ (a j / a (j+1))) →
    (∃ (n : ℕ), ∀ i j, 0 ≤ i → i < 2005 → 0 ≤ j → j < 2005 → 
      nat.distinct i j → nat.distinct (a i) (a j) → n = 46) :=
sorry

end distinct_ratios_min_elements_l93_93343


namespace book_arrangement_l93_93672

theorem book_arrangement (math_books : ℕ) (english_books : ℕ) (science_books : ℕ)
  (math_different : math_books = 4) 
  (english_different : english_books = 5) 
  (science_different : science_books = 2) :
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books) = 34560 := 
by
  sorry

end book_arrangement_l93_93672


namespace contradiction_example_l93_93185

theorem contradiction_example (x y : ℝ) (h1 : x + y > 2) (h2 : x ≤ 1) (h3 : y ≤ 1) : False :=
by
  sorry

end contradiction_example_l93_93185


namespace urns_same_color_probability_l93_93496

/-
 Define the given conditions about the balls in urns and the drawing process.
-/

structure Urns :=
  (ball_urn1 : List String) -- List representing colors of balls in first urn
  (ball_urn2 : List String) -- List representing colors of balls in second urn

def urns : Urns :=
  { ball_urn1 := ["blue", "blue", "red", "red", "red", "green", "green", "green", "green", "green"],
    ball_urn2 := ["blue", "blue", "blue", "blue", "red", "red", "green", "green", "green", "green"] }

-- Define probability function (simplified)
noncomputable def probability {A : Type} (s : Finset A) (p : A → Prop) [DecidablePred p] : ℝ :=
  (s.filter p).card / s.card

-- Probability for drawing each color from each urn
def P_color_urn1 (color : String) : ℝ :=
  probability (Finset.ofList urns.ball_urn1) (λ x => x = color)

def P_color_urn2 (color : String) : ℝ :=
  probability (Finset.ofList urns.ball_urn2) (λ x => x = color)

-- Events B1, C1, D1, B2, C2, D2
def P_B1 := P_color_urn1 "blue"
def P_C1 := P_color_urn1 "red"
def P_D1 := P_color_urn1 "green"
def P_B2 := P_color_urn2 "blue"
def P_C2 := P_color_urn2 "red"
def P_D2 := P_color_urn2 "green"

-- Event A: Drawing the same colors from both urns
def P_A := P_B1 * P_B2 + P_C1 * P_C2 + P_D1 * P_D2

-- Theorem statement
theorem urns_same_color_probability :
  P_A = 0.34 :=
by
  /-
    Here, the computation for P_A will be done step by step similar to the solution provided
    in order to verify that it equals 0.34. However, we leave this as sorry for conceptual completion.
  -/
  sorry

end urns_same_color_probability_l93_93496


namespace sailboat_speed_l93_93403

-- Given conditions
def F (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v) ^ 2) / 2

def N (B S ρ v₀ v : ℝ) : ℝ :=
  F B S ρ v₀ v * v

noncomputable def N_derivative (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ / 2) * (v₀^2 - 4 * v₀ * v + 3 * v^2)

axiom derivative_zero_is_max (B S ρ v₀ v : ℝ) (h : N_derivative B S ρ v₀ v = 0) : 
  ∃ v, v = 2.1

theorem sailboat_speed (B S ρ v₀ : ℝ) (hS : S = 7) (hv₀ : v₀ = 6.3) : 
  ∃ v, N_derivative B S ρ v₀ v = 0 ∧ v = 2.1 := 
by
  use 2.1
  split
  -- You would provide the actual proof here connecting N_derivative B S ρ v₀ 2.1 = 0,
  -- but for now, we'll use sorry as a placeholder.
  sorry
  -- Showing that v equals 2.1 is trivial in this context by construction.
  rfl

end sailboat_speed_l93_93403


namespace smallest_composite_no_prime_lt_20_l93_93052

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93052


namespace total_volume_of_cubes_l93_93920

theorem total_volume_of_cubes {n : ℕ} (h_n : n = 5) (s : ℕ) (h_s : s = 5) :
  n * (s^3) = 625 :=
by {
  rw [h_n, h_s],
  norm_num,
  sorry
}

end total_volume_of_cubes_l93_93920


namespace select_4_people_arrangement_3_day_new_year_l93_93379

def select_4_people_arrangement (n k : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.factorial (n - 2) / Nat.factorial 2

theorem select_4_people_arrangement_3_day_new_year :
  select_4_people_arrangement 7 4 = 420 :=
by
  -- proof to be filled in
  sorry

end select_4_people_arrangement_3_day_new_year_l93_93379


namespace range_of_a_l93_93394

noncomputable def domain_of_f : set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

def inequality_solution_set (a : ℝ) : set ℝ := 
  {x : ℝ | (2*a - 1)*x < a}

theorem range_of_a (a : ℝ) : (∀ x, x ∈ domain_of_f → x ∈ inequality_solution_set a) ↔ a < (2 / 3) :=
  sorry

end range_of_a_l93_93394


namespace base_conversion_and_arithmetic_l93_93587

theorem base_conversion_and_arithmetic :
  let num1 := 2 * (8 ^ 2) + 5 * (8 ^ 1) + 4 * (8 ^ 0),
      den1 := 1 * (4 ^ 1) + 2 * (4 ^ 0),
      num2 := 1 * (5 ^ 2) + 3 * (5 ^ 1) + 2 * (5 ^ 0),
      den2 := 2 * (3 ^ 1) + 3 * (3 ^ 0),
      result := (num1 / den1) + (num2 / den2)
  in
  Float.round result = 33 := by
  let num1 := 2 * (8 ^ 2) + 5 * (8 ^ 1) + 4
  let den1 := 1 * 4 + 2
  let num2 := 1 * (5 ^ 2) + 3 * (5 ^ 1) + 2
  let den2 := 2 * 3 + 3
  let result := (num1.toFloat / den1.toFloat) + (num2.toFloat / den2.toFloat)
  show Float.round result = 33 from sorry

end base_conversion_and_arithmetic_l93_93587


namespace smallest_composite_no_prime_factors_less_than_20_l93_93106

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93106


namespace crayons_left_l93_93434

-- Define initial number of crayons and the number taken by Mary
def initial_crayons : ℝ := 7.5
def taken_crayons : ℝ := 2.25

-- Calculate remaining crayons
def remaining_crayons := initial_crayons - taken_crayons

-- Prove that the remaining crayons are 5.25
theorem crayons_left : remaining_crayons = 5.25 := by
  sorry

end crayons_left_l93_93434


namespace dot_product_focus_hyperbola_l93_93330

-- Definitions related to the problem of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

def is_focus (c : ℝ) (x y : ℝ) : Prop := (x = c ∧ y = 0) ∨ (x = -c ∧ y = 0)

-- Problem conditions
def point_on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

def triangle_area (a b c : ℝ × ℝ) (area : ℝ) : Prop :=
  0.5 * (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) = area

def foci_of_hyperbola : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 0), (-2, 0))

-- Main statement to prove
theorem dot_product_focus_hyperbola
  (m n : ℝ)
  (hP : point_on_hyperbola (m, n))
  (hArea : triangle_area (2, 0) (m, n) (-2, 0) 2) :
  ((-2 - m) * (2 - m) + (-n) * (-n)) = 3 :=
sorry

end dot_product_focus_hyperbola_l93_93330


namespace stocks_worth_at_year_end_l93_93256

-- Definitions for initial investments
def initial_bonus : ℝ := 900
def investment_A : ℝ := initial_bonus / 3
def investment_B : ℝ := initial_bonus / 3
def investment_C : ℝ := initial_bonus / 3

-- Definitions for the value changes after one year
def value_A_after_one_year : ℝ := 2 * investment_A
def value_B_after_one_year : ℝ := 2 * investment_B
def value_C_after_one_year : ℝ := investment_C / 2

-- Total value after one year
def total_value_after_one_year : ℝ := value_A_after_one_year + value_B_after_one_year + value_C_after_one_year

-- Theorem to prove the total value of stocks at the end of the year
theorem stocks_worth_at_year_end : total_value_after_one_year = 1350 := by
  sorry

end stocks_worth_at_year_end_l93_93256


namespace distance_to_nearest_integer_arbitrarily_small_l93_93603

theorem distance_to_nearest_integer_arbitrarily_small
  (q : ℝ) (hq : Irrational q) (c : ℝ) :
  ∀ ε > 0, ∃ m : ℕ, ∃ k : ℤ, |c - m * q - k| < ε :=
by
  sorry

end distance_to_nearest_integer_arbitrarily_small_l93_93603


namespace find_alpha_l93_93227

def f (α : ℝ) (x : ℝ) : ℝ := 
  if x < 0 then (x^2 + Real.sin x) / (-x^2 + Real.cos (x + α)) 
  else -x^2 + Real.cos (x + α)

noncomputable def is_odd_function (α : ℝ) : Prop := 
  ∀ x : ℝ, f α (-x) = -f α x

theorem find_alpha (α : ℝ) : α = 3 * Real.pi / 2 :=
  sorry

end find_alpha_l93_93227


namespace hyperbola_properties_l93_93244

theorem hyperbola_properties :
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3 ∧
  (2 * a = 2) ∧
  ¬(∀ fociCoords : List (ℝ × ℝ), fociCoords = [(0, c), (0, -c)]) ∧
  ¬(∀ x : ℝ, ∀ y : ℝ, 
    (y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x)) ∧
  (∀ e : ℝ, e = c / a → e = Real.sqrt 3) :=
by
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  split; sorry

end hyperbola_properties_l93_93244


namespace sum_of_zeros_f_g_eq_2_l93_93651

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^(x - 2) - 1 else x + 2

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem sum_of_zeros_f_g_eq_2 :
  let h (x : ℝ) := f (g x)
  set zeros := {x : ℝ | h x = 0}
  (∑ x in zeros.to_finset, x) = 2 :=
sorry

end sum_of_zeros_f_g_eq_2_l93_93651


namespace smallest_composite_no_prime_factors_less_than_20_l93_93091

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93091


namespace find_key_effective_expected_residents_to_use_algorithm_l93_93472

-- Define mailboxes and keys
def num_mailboxes : ℕ := 80
def initial_mailbox : ℕ := 37

-- Prove that the algorithm is effective
theorem find_key_effective : 
  ∀ (mailboxes : fin num_mailboxes) (keys : fin num_mailboxes), 
  { permutation : list (fin num_mailboxes) // permutation.nodup ∧ permutation.length = num_mailboxes ∧ 
    ∀ m, m ∈ permutation → (if m = initial_mailbox then m else (keys.filter (λ k, k ∈ permutation))) ≠ [] }
  :=
sorry

-- Prove the expected number of residents who will use the algorithm
theorem expected_residents_to_use_algorithm :
  ∑ i in finset.range num_mailboxes, (1 / (i + 1 : ℝ)) = (real.log 80 + 0.577 + (1 / (2 * 80)) : ℝ)
  :=
sorry

end find_key_effective_expected_residents_to_use_algorithm_l93_93472


namespace smallest_multiple_of_90_with_128_divisors_l93_93337

/-- Smallest positive integer that is a multiple of 90 and has exactly 128 divisors. -/
def n : ℕ := 2^6 * 3^5 * 5^2

/-- The number 90. -/
def ninety : ℕ := 90

/-- The total number of divisors function. -/
def number_of_divisors (n : ℕ) : ℕ :=
  (nat.factors n).erase_dup.length

theorem smallest_multiple_of_90_with_128_divisors :
  number_of_divisors n = 128 ∧ n % ninety = 0 → n / ninety = 1728 := by
  intros h
  sorry

end smallest_multiple_of_90_with_128_divisors_l93_93337


namespace smallest_composite_no_prime_lt_20_l93_93062

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93062


namespace drinking_ratio_l93_93942

variable (t_mala t_usha : ℝ) (d_usha : ℝ)

theorem drinking_ratio :
  (t_mala = t_usha) → 
  (d_usha = 2 / 10) →
  (1 - d_usha = 8 / 10) →
  (4 * d_usha = 8) :=
by
  intros h1 h2 h3
  sorry

end drinking_ratio_l93_93942


namespace product_in_fourth_quadrant_l93_93804

def complex_number_1 := (1 : ℂ) - (2 : ℂ) * I
def complex_number_2 := (2 : ℂ) + I

def product := complex_number_1 * complex_number_2

theorem product_in_fourth_quadrant :
  product.re > 0 ∧ product.im < 0 :=
by
  -- the proof is omitted
  sorry

end product_in_fourth_quadrant_l93_93804


namespace combined_tax_rate_l93_93936

theorem combined_tax_rate
  (Mork_income : ℝ)
  (Mindy_income : ℝ)
  (h1 : Mindy_income = 3 * Mork_income)
  (Mork_tax_rate : ℝ := 0.30)
  (Mindy_tax_rate : ℝ := 0.20) :
  (Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income) * 100 = 22.5 :=
by
  sorry

end combined_tax_rate_l93_93936


namespace how_many_more_red_balls_l93_93322

def r_packs : ℕ := 12
def y_packs : ℕ := 9
def r_balls_per_pack : ℕ := 24
def y_balls_per_pack : ℕ := 20

theorem how_many_more_red_balls :
  (r_packs * r_balls_per_pack) - (y_packs * y_balls_per_pack) = 108 :=
by
  sorry

end how_many_more_red_balls_l93_93322


namespace boiling_point_of_water_in_Fahrenheit_is_212_l93_93911

theorem boiling_point_of_water_in_Fahrenheit_is_212 :
  ∀ (boiling_point_C : ℝ), boiling_point_C = 100 → boiling_point_C * (9 / 5) + 32 = 212 :=
by
  intros boiling_point_C h
  rw h
  norm_num
  sorry

end boiling_point_of_water_in_Fahrenheit_is_212_l93_93911


namespace grid_paths_l93_93556

theorem grid_paths (columns rows total_steps right_steps up_steps : ℕ)
  (h_columns : columns = 7)
  (h_rows : rows = 5)
  (h_total_steps : total_steps = 10)
  (h_right_steps : right_steps = 7)
  (h_up_steps : up_steps = 3)
  (h_sum_steps : total_steps = right_steps + up_steps) :
  (Nat.choose total_steps up_steps) = 120 :=
by
  have : total_steps = 10 := h_total_steps
  have : right_steps = 7 := h_right_steps
  have : up_steps = 3 := h_up_steps
  have : total_steps = right_steps + up_steps := by rw [h_right_steps, h_up_steps]
  rw [← this]
  have : Nat.choose 10 3 = Nat.factorial 10 / (Nat.factorial 3 * Nat.factorial (10 - 3)) := Nat.choose_eq_factorial_div_factorial
  rw [Nat.factorial, Nat.factorial, Nat.factorial]
  exact Nat.factorial_rules 10 9 8 7 120
  sorry

end grid_paths_l93_93556


namespace algorithm_effective_expected_residents_using_stick_l93_93476

-- Part (a): Prove the algorithm is effective
theorem algorithm_effective :
  ∀ (n : ℕ) (keys : ℕ → ℕ) (start : ℕ),
  (1 ≤ start ∧ start ≤ n) →
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (∃ (sequence : ℕ → ℕ), 
     (sequence 0 = start) ∧ 
     (∀ (i : ℕ), sequence (i + 1) = keys (sequence i)) ∧ 
     (∃ m, sequence m = start)) :=
by
  sorry

-- Part (b): Expected number of residents who need to use the stick is approximately 4.968
-- Here we outline the theorem related to the expected value part.
theorem expected_residents_using_stick (n : ℕ) (H : n = 80) :
  ∀ (keys : ℕ → ℕ),
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (let Hn := ∑ i in range n, 1 / i
   in abs(Hn - 4.968) < ε) :=
by
  sorry

end algorithm_effective_expected_residents_using_stick_l93_93476


namespace simplify_polynomial_l93_93207

open Polynomial

def arithmetic_sequence (a_0 d : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a_0 + n * d

def p_n (a : ℕ → ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * (nat.choose n k) * (x ^ k) * ((1 - x) ^ (n - k))

theorem simplify_polynomial (a_0 d : ℝ) (a : ℕ → ℝ) (h : arithmetic_sequence a_0 d a)
  (n : ℕ) (x : ℝ) : p_n a x n = a_0 + n * d * x :=
by
  sorry

end simplify_polynomial_l93_93207


namespace hot_water_bottles_sold_l93_93909

theorem hot_water_bottles_sold (T H : ℕ) (h1 : 2 * T + 6 * H = 1200) (h2 : T = 7 * H) : H = 60 := 
by 
  sorry

end hot_water_bottles_sold_l93_93909


namespace calculate_fraction_l93_93549

theorem calculate_fraction : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end calculate_fraction_l93_93549


namespace Sara_house_size_l93_93858

theorem Sara_house_size (nada_size : ℕ) (h1 : nada_size = 450) (h2 : Sara_size = 2 * nada_size + 100) : Sara_size = 1000 :=
by sorry

end Sara_house_size_l93_93858


namespace algorithm_effective_expected_number_of_residents_l93_93485

-- Definitions required from the conditions of the original problem
def num_mailboxes : ℕ := 80

def key_distribution : Equiv.Perm (Fin num_mailboxes) := sorry

def initial_mailbox : Fin num_mailboxes := 37

-- Lean 4 statement for Part (a)
theorem algorithm_effective :
  ∃ m : Fin num_mailboxes, m = initial_mailbox → 
    (fix : ℕ → Fin num_mailboxes)
      (fix 0 = initial_mailbox)
      (∀ n, fix (n+1) = key_distribution (fix n))
      ∃ k, fix k = initial_mailbox := sorry

-- Lean 4 statement for Part (b)
theorem expected_number_of_residents :
  ∀ n, n = num_mailboxes → 
    let Harmonic := λ (n : ℕ), Σ i in Finset.range n, 1 / (i + 1)
    Harmonic n ≈ 4.968 := sorry

end algorithm_effective_expected_number_of_residents_l93_93485


namespace algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93478

-- Definition of the mailbox setting and conditions
def mailbox_count : ℕ := 80

def apartment_resident_start : ℕ := 37

noncomputable def randomized_keys_placed_in_mailboxes : Bool :=
  true  -- This is a placeholder; actual randomness is abstracted

-- Statement of the problem
theorem algorithm_will_find_key (mailboxes keys : Fin mailbox_count) (start : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ (sequence : ℕ → Fin mailbox_count), ∀ n : ℕ, sequence n ≠ start → sequence (n+1) ≠ start → 
  (sequence 0 = start ∨ ∃ k : ℕ, sequence k = keys.start → keys = sequence (k+1)).
  sorry

theorem expected_number_of_residents_using_algorithm 
  (mailboxes keys : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ n : ℝ, n ≈ 4.968.
  sorry

end algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93478


namespace radius_of_incircle_line_EF_tangent_to_circle_l93_93537

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 16 + y^2 = 1

def circle_eq (x y : ℝ) : ℝ → Prop
| r := (x - 2)^2 + y^2 = r^2

def point_A (x y : ℝ) : Prop := ellipse_eq (-4) 0

def point_M (x y : ℝ) : Prop := (x = 0) ∧ (y = 1)

theorem radius_of_incircle (r : ℝ) :
  (∃ (x y : ℝ), point_A x y ∧ circle_eq x y r ∧ 
    ∀ (x1 y1 x2 y2 : ℝ), ellipse_eq x1 y1 ∧ ellipse_eq x2 y2 ∧ 
    (circle_eq x1 y1 r ∧ circle_eq x2 y2 r) ∧ ∃ (k : ℝ), (y1 - y2) = k * (x1 - x2))
    → r = 2/3 :=
sorry

theorem line_EF_tangent_to_circle (r : ℝ) :
  (r = 2/3 ∧ ∃ (x y : ℝ), point_M x y ∧ ∀ (x1 y1 x2 y2 : ℝ), 
    (tangent_to_circle_from_point x y x1 y1 r ∧ tangent_to_circle_from_point x y x2 y2 r) ∧ 
    (∃ (k : ℝ), k * (y1 - y2) = (x1 - x2)))
    → tangent_line_to_circle r :=
sorry

end radius_of_incircle_line_EF_tangent_to_circle_l93_93537


namespace valid_b_values_count_l93_93893

theorem valid_b_values_count : 
  (∃! b : ℤ, ∃ x1 x2 x3 : ℤ, 
    (∀ x : ℤ, x^2 + b * x + 5 ≤ 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧ 
    (20 ≤ b^2 ∧ b^2 < 29)) :=
sorry

end valid_b_values_count_l93_93893


namespace smallest_composite_no_prime_factors_less_than_20_l93_93134

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93134


namespace extreme_value_iff_a_lt_zero_l93_93412

theorem extreme_value_iff_a_lt_zero (a : ℝ) :
  (∃ x : ℝ, deriv (λ x, a * x^3 + x + 3) x = 0) ↔ a < 0 :=
by
  -- Sorry is added to skip the proof
  sorry

end extreme_value_iff_a_lt_zero_l93_93412


namespace second_person_avg_pages_per_day_l93_93567

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end second_person_avg_pages_per_day_l93_93567


namespace number_of_correct_conclusions_l93_93336

noncomputable def f (x : ℝ) (phi : ℝ) := Real.sin (2 * x + phi)

theorem number_of_correct_conclusions (phi : ℝ) 
  (h1 : ∀ x : ℝ, f x phi ≤ |f (Real.pi / 6) phi|) : 
  (1 + 1 + 1 = 3) :=
begin
  -- Placeholder for the proof steps
  sorry
end

end number_of_correct_conclusions_l93_93336


namespace orange_ribbons_count_l93_93291

noncomputable def total_ribbons (x y p o : ℕ) : ℕ :=
  x + y + (24 * 45 / 7 - x - y).to_nat - p - o

theorem orange_ribbons_count (x y p o: ℕ) (hx: x = 6 * 45 / 7) (hy: y = 8 * 45 / 7) (hp: p = 3 * 45 /7 - o) (ho: o = 45): (1 * p / 8 /7 = 19):= 
  sorry

end orange_ribbons_count_l93_93291


namespace line_does_not_pass_second_quadrant_l93_93265

theorem line_does_not_pass_second_quadrant 
  (A B C x y : ℝ) 
  (h1 : A * C < 0) 
  (h2 : B * C > 0) 
  (h3 : A * x + B * y + C = 0) :
  ¬ (x < 0 ∧ y > 0) := 
sorry

end line_does_not_pass_second_quadrant_l93_93265


namespace smallest_composite_no_prime_factors_less_than_20_l93_93103

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93103


namespace smallest_composite_no_prime_factors_less_than_20_l93_93003

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93003


namespace james_spent_6_dollars_l93_93316

-- Define the cost of items
def cost_milk : ℕ := 3
def cost_bananas : ℕ := 2

-- Define the sales tax rate as a decimal
def sales_tax_rate : ℚ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℕ := cost_milk + cost_bananas

-- Define the sales tax amount
def sales_tax_amount : ℚ := sales_tax_rate * total_cost_before_tax

-- Define the total amount spent
def total_amount_spent : ℚ := total_cost_before_tax + sales_tax_amount

-- The proof statement
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l93_93316


namespace prob_5_shots_expected_number_shots_l93_93957

variable (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

def prob_exactly_five_shots : ℝ := 6 * p^3 * (1 - p)^2
def expected_shots : ℝ := 3 / p

theorem prob_5_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the probability of exactly 5 shots needed is as calculated
  prob_exactly_five_shots p = 6 * p^3 * (1 - p)^2 :=
by
  sorry

theorem expected_number_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the expected number of shots to hit all targets is as calculated
  expected_shots p = 3 / p :=
by
  sorry

end prob_5_shots_expected_number_shots_l93_93957


namespace equilateral_triangle_side_length_l93_93390

theorem equilateral_triangle_side_length (r : ℝ) (a : ℝ) (h: r = 1) 
  (t : ∀ (A B C D E F G : Set ℝ), 
  EquilateralTriangle A B C → 
  ∀ (α β γ : ℝ), 
  α = β ∧ β = γ ∧ γ = 60 ∧ A = B ∧ B = C ∧ C = D ∧ E = F ∧ F = G ∧ G = 1):
  a = 2 * Real.sqrt 3 :=
begin
  sorry
end

end equilateral_triangle_side_length_l93_93390


namespace smallest_composite_no_prime_factors_less_20_l93_93042

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93042


namespace quadrilateral_is_square_l93_93513

-- Define necessary geometric entities and properties
variables {α : Type} [euclidean_geometry α]
variables {A B C D O : α}

-- Define the conditions
def is_convex_quadrilateral_inscribed_in_circle (A B C D O : α) : Prop := sorry -- Details to be filled in based on convex and inscribed properties
def angles_at_center_equal_angles_of_quadrilateral (A B C D O : α) : Prop := sorry -- Details to define the specific angle condition

-- The theorem statement: Prove that ABCD is a square
theorem quadrilateral_is_square 
  (h1 : is_convex_quadrilateral_inscribed_in_circle A B C D O) 
  (h2 : angles_at_center_equal_angles_of_quadrilateral A B C D O) : 
  is_square A B C D := 
sorry

end quadrilateral_is_square_l93_93513


namespace triangle_inequality_sum_l93_93638

theorem triangle_inequality_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (c / (a + b)) + (a / (b + c)) + (b / (c + a)) > 1 :=
by
  sorry

end triangle_inequality_sum_l93_93638


namespace pythagorean_tetrahedron_theorem_l93_93305

variable (A B C P : Type)
variables {PA PB PC : ℝ}
variables {S_ABC S_PAB S_PAC S_PBC : ℝ}

-- Given the conditions
axiom PA_perp_PB : PA ⊥ PB
axiom PA_perp_PC : PA ⊥ PC
axiom PB_perp_PC : PB ⊥ PC

-- We need to prove the analogous Pythagorean theorem for the tetrahedron
theorem pythagorean_tetrahedron_theorem
( PA_perp_PB : PA ⊥ PB )
( PA_perp_PC : PA ⊥ PC )
( PB_perp_PC : PB ⊥ PC ) :
  S_ABC ^ 2 = S_PAB ^ 2 + S_PAC ^ 2 + S_PBC ^ 2 :=
sorry

end pythagorean_tetrahedron_theorem_l93_93305


namespace area_of_shaded_region_is_one_third_l93_93998

noncomputable def find_area_of_shaded_region (α : ℝ) : ℝ :=
  if (0 < α ∧ α < π / 2 ∧ (cos α = 3/5)) then 
    1/3 
  else 
    0

theorem area_of_shaded_region_is_one_third (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : cos α = 3 / 5) :
  find_area_of_shaded_region α = 1 / 3 :=
by
  rw [find_area_of_shaded_region, if_pos]
  exacts [h1, h2, h3]
  sorry

end area_of_shaded_region_is_one_third_l93_93998


namespace intersection_points_l93_93407

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the domain of the function
def domain (x : ℝ) : Prop := x ≠ 1

-- The theorem to prove the intersection points
theorem intersection_points {x : ℝ} (hx : domain x): ∃ y : ℝ, y = f x → x = 1 → false :=
begin
  sorry
end

end intersection_points_l93_93407


namespace complex_quadrant_l93_93802

def z1 := Complex.mk 1 (-2)
def z2 := Complex.mk 2 1
def z := z1 * z2

theorem complex_quadrant : z = Complex.mk 4 (-3) ∧ z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l93_93802


namespace possible_even_and_odd_functions_l93_93458

def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem possible_even_and_odd_functions :
  ∃ p q : ℝ → ℝ, is_even_function p ∧ is_odd_function (p ∘ q) ∧ (¬(∀ x, p (q x) = 0)) :=
by
  sorry

end possible_even_and_odd_functions_l93_93458


namespace book_price_l93_93984

theorem book_price (B P : ℝ) 
  (h1 : (1 / 3) * B = 36) 
  (h2 : (2 / 3) * B * P = 252) : 
  P = 3.5 :=
by {
  sorry
}

end book_price_l93_93984


namespace inequality_has_no_solutions_l93_93176

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end inequality_has_no_solutions_l93_93176


namespace no_possible_path_l93_93994

theorem no_possible_path (n : ℕ) (h1 : n > 0) :
  ¬ ∃ (path : ℕ × ℕ → ℕ × ℕ), 
    (∀ (i : ℕ × ℕ), path i = if (i.1 < n - 1 ∧ i.2 < n - 1) then (i.1 + 1, i.2) else if i.2 < n - 1 then (i.1, i.2 + 1) else (i.1 - 1, i.2 - 1)) ∧
    (∀ (i j : ℕ × ℕ), i ≠ j → path i ≠ path j) ∧
    path (0, 0) = (0, 1) ∧
    path (n-1, n-1) = (n-1, 0) :=
sorry

end no_possible_path_l93_93994


namespace smallest_composite_no_prime_factors_less_than_20_l93_93095

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93095


namespace smallest_composite_no_prime_factors_lt_20_l93_93155

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93155


namespace value_of_p_l93_93212

variable (x y : ℝ)

theorem value_of_p (h1 : | x - (1/2) | + sqrt (y^2 - 1) = 0) :
  let p := | x | + | y |
  p = 3/2 :=
by
  sorry

end value_of_p_l93_93212


namespace smallest_composite_no_prime_factors_less_than_20_l93_93008

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93008


namespace minimum_workers_needed_l93_93562

theorem minimum_workers_needed 
  (total_days : ℕ)
  (completed_days : ℕ)
  (initial_workers : ℕ)
  (fraction_completed : ℚ)
  (remaining_fraction : ℚ)
  (remaining_days : ℕ)
  (rate_completed_per_day : ℚ)
  (required_rate_per_day : ℚ)
  (equal_productivity : Prop) 
  : initial_workers = 10 :=
by
  -- Definitions
  let total_days := 40
  let completed_days := 10
  let initial_workers := 10
  let fraction_completed := 1 / 4
  let remaining_fraction := 1 - fraction_completed
  let remaining_days := total_days - completed_days
  let rate_completed_per_day := fraction_completed / completed_days
  let required_rate_per_day := remaining_fraction / remaining_days
  let equal_productivity := true

  -- Sorry is used to skip the proof
  sorry

end minimum_workers_needed_l93_93562


namespace second_person_avg_pages_per_day_l93_93568

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end second_person_avg_pages_per_day_l93_93568


namespace smallest_composite_no_prime_factors_less_than_20_l93_93098

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93098


namespace border_area_correct_l93_93522

-- Definition of the dimensions of the photograph
def photo_height := 8
def photo_width := 10
def frame_border := 3

-- Definition of the areas of the photograph and the framed area
def photo_area := photo_height * photo_width
def frame_height := photo_height + 2 * frame_border
def frame_width := photo_width + 2 * frame_border
def frame_area := frame_height * frame_width

-- Theorem stating that the area of the border is 144 square inches
theorem border_area_correct : (frame_area - photo_area) = 144 := 
by
  sorry

end border_area_correct_l93_93522


namespace perfect_square_position_l93_93416

inductive TermSequence : Nat → Type
| base1 : TermSequence 1
| base2 : TermSequence 2
| recur (n m l : Nat) (an a_m : TermSequence m) (an_1 : TermSequence n) (n_eq : l = m + 1) (m_eq : n = l + 1) : TermSequence l

open TermSequence

noncomputable def a : ℕ → ℕ
| 0 := 1
| 1 := 8
| 2 := 18
| n + 2 := a (n + 1) * a n

theorem perfect_square_position :
  ∀ n, ∃ k, a n = k * k ↔ ∃ m, n = 3 * m := by
  sorry

end perfect_square_position_l93_93416


namespace length_BC_is_expressed_as_fraction_m_n_l93_93780

-- Define the context and conditions
variables (A B C J X Y : Type) 
variables (P: X → Prop) (Q: Y → Prop) (R: BC → Prop) (S: Circumcircle ABC → Prop)

-- Define the specific properties
axiom h1 : ∀ (x y : X), P x ↔ P y
axiom h2 : ∀ (y z : Y), Q y ↔ Q z
axiom h3 : ∀ (x : BC), P x ↔ Q x
axiom h4 : S J

def m := 82
def n := 3

-- Proof of the mathematical statement
theorem length_BC_is_expressed_as_fraction_m_n : m + n = 85 :=
by
  sorry

end length_BC_is_expressed_as_fraction_m_n_l93_93780


namespace diagonal_difference_l93_93980

def initial_matrix : matrix (fin 4) (fin 4) ℕ := 
  ![![1, 2, 3, 4],
    ![8, 9, 10, 11],
    ![15, 16, 17, 18],
    ![22, 23, 24, 25]]

def reversed_matrix : matrix (fin 4) (fin 4) ℕ := 
  ![![4, 3, 2, 1],
    ![8, 9, 10, 11],
    ![18, 17, 16, 15],
    ![22, 23, 24, 25]]

def main_diagonal_sum (m : matrix (fin 4) (fin 4) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2 + m 3 3

def anti_diagonal_sum (m : matrix (fin 4) (fin 4) ℕ) : ℕ :=
  m 0 3 + m 1 2 + m 2 1 + m 3 0

theorem diagonal_difference :
  abs ((main_diagonal_sum reversed_matrix) - (anti_diagonal_sum reversed_matrix)) = 4 := 
by
  sorry

end diagonal_difference_l93_93980


namespace smallest_composite_no_prime_factors_lt_20_l93_93147

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93147


namespace initial_investment_l93_93533

-- Let's define the parameters of the problem.
def annual_rate : ℝ := 0.10
def compounding_times : ℕ := 1
def years : ℕ := 2
def final_amount : ℝ := 8470

-- Let's define the formula for compound interest explicitly.
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- The goal is to prove the initial investment.
theorem initial_investment :
  (∃ P : ℝ, compound_interest P annual_rate compounding_times years = final_amount) ∧
  (compound_interest 7000 annual_rate compounding_times years = final_amount) :=
by
  use 7000
  sorry

end initial_investment_l93_93533


namespace angle_range_of_slope_l93_93775

theorem angle_range_of_slope (k : ℝ) (hk : -1 < k ∧ k < sqrt 3) :
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ ((α < π / 3) ∨ (3 * π / 4 < α)) :=
sorry

end angle_range_of_slope_l93_93775


namespace cucumbers_count_l93_93544

theorem cucumbers_count (C T : ℕ) 
  (h1 : C + T = 280)
  (h2 : T = 3 * C) : C = 70 :=
by sorry

end cucumbers_count_l93_93544


namespace second_person_avg_pages_per_day_l93_93570

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end second_person_avg_pages_per_day_l93_93570


namespace smallest_composite_no_prime_factors_less_20_l93_93048

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93048


namespace smallest_composite_no_prime_factors_less_than_20_l93_93026

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93026


namespace closest_integer_sqrt_35_l93_93364

theorem closest_integer_sqrt_35 : ∃ (n : ℤ), (5 : ℝ) < (sqrt 35) ∧ (sqrt 35) < (6 : ℝ) ∧ (n = 6) :=
by
  sorry

end closest_integer_sqrt_35_l93_93364


namespace smallest_composite_no_prime_factors_less_than_20_l93_93146

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93146


namespace whole_numbers_between_cuberoots_l93_93688

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93688


namespace gcd_factorial_div_l93_93915

theorem gcd_factorial_div (a b g : ℕ) (h₁ : a = 7!) (h₂ : b = 9! / 4!) (h₃ : g = Nat.gcd a b) : g = 5040 := by
  sorry

end gcd_factorial_div_l93_93915


namespace cars_one_person_men_driven_correct_l93_93790

-- Define the conditions
def total_cars : ℕ := 50
def percent_more_than_one_person : ℕ := 20
def percent_one_person_women_driven : ℕ := 60

-- Define the expected outcome
def cars_one_person_men_driven : ℕ := 16

-- Theorem statement to prove the relation
theorem cars_one_person_men_driven_correct :
  let cars_more_than_one_person := (percent_more_than_one_person * total_cars) / 100 in
  let cars_one_person := total_cars - cars_more_than_one_person in
  let cars_one_person_women := (percent_one_person_women_driven * cars_one_person) / 100 in
  let cars_one_person_men := cars_one_person - cars_one_person_women in
  cars_one_person_men = cars_one_person_men_driven :=
by
  sorry

end cars_one_person_men_driven_correct_l93_93790


namespace difference_of_sides_l93_93414

-- Definitions based on conditions
def smaller_square_side (s : ℝ) := s
def larger_square_side (S s : ℝ) (h : (S^2 : ℝ) = 4 * s^2) := S

-- Theorem statement based on the proof problem
theorem difference_of_sides (s S : ℝ) (h : (S^2 : ℝ) = 4 * s^2) : S - s = s := 
by
  sorry

end difference_of_sides_l93_93414


namespace find_a_plus_b_l93_93856

-- Define the conditions and necessary variables
variables (EV EP VF : ℕ) (k : ℝ) (a b : ℕ) 
noncomputable def EG := a * real.sqrt b

-- Define the conditions
axiom EV_val : EV = 42
axiom EP_val : EP = 63
axiom VF_val : VF = 84
axiom area_ratio : k = (2 : ℝ) / 3

-- Prime square non-divisibility condition
axiom b_not_div_prime_square : ∀ (p : ℕ), prime p → p^2 ∣ b → false

theorem find_a_plus_b : EV = 42 ∧ EP = 63 ∧ VF = 84 ∧ 
  k = (2:ℝ) / 3 ∧ (EG = a * real.sqrt b) ∧ (∀ p:ℕ, prime p → p^2 ∣ b → false) → 
  a + b = 27 :=
sorry

end find_a_plus_b_l93_93856


namespace sarah_annual_income_l93_93286

theorem sarah_annual_income (q : ℝ) (I T : ℝ)
    (h1 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) 
    (h2 : T = 0.01 * (q + 0.5) * I) : 
    I = 36000 := by
  sorry

end sarah_annual_income_l93_93286


namespace solution_l93_93210

open Real

variable (x y : ℝ)
def problem_condition : Prop :=
  logBase 2 (4 * cos (x * y) ^ 2 + 1 / (4 * cos (x * y) ^ 2)) = -y^2 + 4 * y - 3

theorem solution : problem_condition x y → y * cos (4 * x) = -1 := by
  intros h
  sorry

end solution_l93_93210


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93073

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93073


namespace num_whole_numbers_between_l93_93701

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93701


namespace whole_numbers_count_between_cubic_roots_l93_93719

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93719


namespace first_knife_cost_l93_93369

theorem first_knife_cost
  (costStructure : ∀ (x : ℕ), x = 0 → ℕ -> ℕ)
  (h1 : ∀ (x : ℕ), x > 0 → x ≤ 3 → costStructure x = 4)
  (h2 : ∀ (x : ℕ), x > 3 → costStructure x = 3)
  (totalKnives : ℕ) (totalCost : ℕ) :
  totalKnives = 9 → totalCost = 32 → ∃ (X : ℕ), costStructure 0 X = 5 := by
  sorry

end first_knife_cost_l93_93369


namespace sequence_formula_l93_93809

open Nat

noncomputable def a_seq : ℕ → ℕ
| 0        := 0  -- Convention to simplify indexing
| 1        := 6
| (n + 2)  := a_seq (n + 1) + a_seq (n + 1) / (n + 2) + (n + 2) + 1

theorem sequence_formula (n : ℕ) (h : n ≠ 0) : a_seq n = (n + 1) * (n + 2) := by
  sorry

end sequence_formula_l93_93809


namespace smallest_composite_no_prime_factors_less_than_20_l93_93025

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93025


namespace ellipse_equation_point_K_on_hyperbola_line_l_equation_l93_93622

-- Problem 1: Proving the equation of the ellipse
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (sqrt 2) / 2 = 1) : 
  a = 2 ∧ b = 1 ∧ ∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1 ↔ x^2 / 2 + y^2 = 1 :=
sorry

-- Problem 2: Proving the hyperbola on which point K lies
theorem point_K_on_hyperbola (x y t y₀ : ℝ) (E D A B K : (ℝ × ℝ)) (h1 : E = (- sqrt 2, 0)) 
  (h2 : D = (sqrt 2, 0)) (h3 : A = (t, y₀)) (h4 : B = (t, -y₀)) (h5 : ∀ (x : ℝ), x^2 / 2 + y₀^2 = 1) :
  ∀ (K : (ℝ × ℝ)), ∃ x y : ℝ, K = (x, y) ∧ x^2 / 2 - y ^ 2 = 1 :=
sorry

-- Problem 3: Proving the equation of the line l
theorem line_l_equation (x P Q : (ℝ × ℝ)) (k : ℝ) (h1 : P = (- 1, 1 / sqrt 2)) 
  (h2 : Q = (- 1, -1 / sqrt 2)) (h3 : ∀ (O : ℝ), (O, 0) → (P • Q)=-1 / 3 ∧ ∀ x y, y = k (x + 1)) :
  k = 1 ∧ (∀ x : ℝ, y = x + 1 ∨ y = -x - 1) :=
sorry

end ellipse_equation_point_K_on_hyperbola_line_l_equation_l93_93622


namespace smallest_composite_no_prime_factors_less_than_20_l93_93014

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93014


namespace plane_equation_l93_93597

noncomputable def plane_eqn (x y z : ℝ) : ℝ :=
2 * x - y + 3 * z + 8

theorem plane_equation
    (x y z : ℝ)
    (hx : x = 1)
    (hy : y = 4)
    (hz : z = -2)
    : plane_eqn x y z = 0 :=
by {
    rw [hx, hy, hz],
    simp [plane_eqn],
    norm_num,
}

end plane_equation_l93_93597


namespace area_of_ABC_l93_93311

noncomputable def area_of_triangle_ABC_given_medians : ℝ :=
  let AF := 10
  let BE := 15
  let θ := real.pi / 3 -- θ in radians for 60 degrees
  let AG := (2/3) * AF
  let BG := (2/3) * BE
  let area_ΔAGB := (1/2) * AG * BG * real.sin θ
  6 * area_ΔAGB

theorem area_of_ABC (AF : ℝ) (BE : ℝ) (θ : ℝ) (hAF : AF = 10) (hBE : BE = 15) (hθ : θ = real.pi / 3) :
  area_of_triangle_ABC_given_medians = 200 * real.sqrt 3 := 
by
  rw [hAF, hBE, hθ]
  simp [area_of_triangle_ABC_given_medians]
  sorry

end area_of_ABC_l93_93311


namespace proof_problem_l93_93200

def period_f (P : ℝ × ℝ) (Q : ℝ → ℝ × ℝ) (f : ℝ → ℝ) := 
  ∀ x, f x = (P.1 * (P.1 - (Q x).1) + P.2 * (P.2 - (Q x).2))

def period_condition (f : ℝ → ℝ) : Prop := 
  ∃ T > 0, ∀ x, f (x + T) = f x

def triangle_ABC (S : ℝ) (BC : ℝ) (A : ℝ) (f : ℝ → ℝ) := 
  ∃ b c : ℝ, BC = b * c ∧ f A = 4 ∧ 
  S = (1/2) * b * c * Real.sin (A + Real.pi / 3)

noncomputable def perimeter_ABC (a b c: ℝ) := a + b + c

theorem proof_problem : 
  (period_f (sqrt 3, 1) (λ x, (Real.cos x, Real.sin x)) 
    (λ x, sqrt 3 * (sqrt 3 - Real.cos x) + 1 - Real.sin x)) → 
  (period_condition (λ x, sqrt 3 * (sqrt 3 - Real.cos x) + 1 - Real.sin x)) ∧ 
  (triangle_ABC ((3 * Real.sqrt 3) / 4) 3 (2 * Real.pi / 3) (λ x, sqrt 3 * (sqrt 3 - Real.cos x) + 1 - Real.sin x)) →
  ∃ a b c, a^2 = b^2 + c^2 - 2 * b * c * Real.cos (2 * Real.pi / 3) ∧ 
    BC * b = 3 ∧ 
    perimeter_ABC a b c = 3 + 2 * Real.sqrt 3
:= by 
  -- Proof goes here 
  sorry

end proof_problem_l93_93200


namespace sailboat_speed_l93_93404

-- Given conditions
def F (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v) ^ 2) / 2

def N (B S ρ v₀ v : ℝ) : ℝ :=
  F B S ρ v₀ v * v

noncomputable def N_derivative (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ / 2) * (v₀^2 - 4 * v₀ * v + 3 * v^2)

axiom derivative_zero_is_max (B S ρ v₀ v : ℝ) (h : N_derivative B S ρ v₀ v = 0) : 
  ∃ v, v = 2.1

theorem sailboat_speed (B S ρ v₀ : ℝ) (hS : S = 7) (hv₀ : v₀ = 6.3) : 
  ∃ v, N_derivative B S ρ v₀ v = 0 ∧ v = 2.1 := 
by
  use 2.1
  split
  -- You would provide the actual proof here connecting N_derivative B S ρ v₀ 2.1 = 0,
  -- but for now, we'll use sorry as a placeholder.
  sorry
  -- Showing that v equals 2.1 is trivial in this context by construction.
  rfl

end sailboat_speed_l93_93404


namespace range_of_f_l93_93241
open Function

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 2) - 1

theorem range_of_f : (image f (Icc 0 3)) = Icc (-5) 31 :=
by
  sorry

end range_of_f_l93_93241


namespace zhang_qiu_jian_problem_l93_93867

-- Define the arithmetic sequence
def arithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

-- Sum of first n terms of an arithmetic sequence
def sumArithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem zhang_qiu_jian_problem :
  sumArithmeticSequence 5 (16 / 29) 30 = 390 := 
by 
  sorry

end zhang_qiu_jian_problem_l93_93867


namespace volume_ratio_l93_93950

-- Assume the necessary geometric entities and volume function are predefined.
-- Define the conditions given in the problem.
variables (S A B C D P Q : Point)
variables (midpoint_P : midpoint S B P)
variables (midpoint_Q : midpoint S D Q)
variables (plane_PQ_divides : divides_plane (Plane A P Q) (Pyramid S A B C D) V1 V2)
variables (V1_lt_V2 : V1 < V2)

-- Define the pyramid being regular.
variables (regular_pyramid : regular_quad_pyramid S A B C D)

-- Define the goal, which is the ratio of the volumes.
noncomputable def ratio_of_volumes := V2 / V1

-- The theorem to prove that the ratio V2 / V1 is 5.
theorem volume_ratio :
  ratio_of_volumes S A B C D P Q midpoint_P midpoint_Q plane_PQ_divides V1_lt_V2 regular_pyramid = 5 := sorry

end volume_ratio_l93_93950


namespace smallest_composite_no_prime_factors_less_than_20_l93_93000

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93000


namespace distance_between_after_3_minutes_l93_93463

def initial_distance : ℝ := 180  -- Initial distance in kilometers
def speed_criminal : ℝ := 8 / 60  -- Speed of the criminal in kilometers per minute (converted from km/h)
def speed_policeman : ℝ := 9 / 60  -- Speed of the policeman in kilometers per minute (converted from km/h)
def time_interval : ℝ := 3  -- Time interval in minutes

theorem distance_between_after_3_minutes :
  (initial_distance - ((speed_policeman * time_interval) - (speed_criminal * time_interval))) = 179.95 :=
sorry

end distance_between_after_3_minutes_l93_93463


namespace smallest_composite_no_prime_factors_below_20_l93_93124

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93124


namespace negation_of_exists_l93_93884

theorem negation_of_exists : (¬ ∃ x : ℝ, x > 0 ∧ x^2 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 ≤ 0 :=
by sorry

end negation_of_exists_l93_93884


namespace exact_probability_five_shots_l93_93959

theorem exact_probability_five_shots (p : ℝ) (h1 : 0 < p) (h2 : p ≤ 1) :
  (let hit := p
       miss := 1 - p
       comb := 6 in
   comb * hit^3 * miss^2 = 6 * p^3 * (1 - p)^2) :=
by sorry

end exact_probability_five_shots_l93_93959


namespace price_of_fudge_delights_l93_93551

theorem price_of_fudge_delights (F : ℝ) : 
  (3 * 4) + (9 * 2) + (2 * 3.5) + F = 42 → F = 5 :=
by
  assume h1 : (3 * 4) + (9 * 2) + (2 * 3.5) + F = 42
  sorry

end price_of_fudge_delights_l93_93551


namespace age_difference_l93_93763

def age1 : ℕ := 10
def age2 : ℕ := age1 - 2
def age3 : ℕ := age2 + 4
def age4 : ℕ := age3 / 2
def age5 : ℕ := age4 + 20
def avg : ℕ := (age1 + age5) / 2

theorem age_difference :
  (age3 - age2) = 4 ∧ avg = 18 := by
  sorry

end age_difference_l93_93763


namespace derivative_depends_only_on_x0_l93_93388

variable {𝕜 : Type*} [nontrivially_normed_field 𝕜]

noncomputable def derivative_depends_on_x0
  (f : 𝕜 → 𝕜) (x0 : 𝕜) : Prop :=
  ∀ (ε : 𝕜), (0 < ε) →
  ∃ (δ : 𝕜), (0 < δ) ∧ 
  ∀ (Δx : 𝕜), (|Δx| < δ) → |(f x0 + Δx - f x0) / Δx - (f x0).lim| < ε

theorem derivative_depends_only_on_x0
  (f : 𝕜 → 𝕜) (x0 : 𝕜) : derivative_depends_on_x0 f x0 :=
begin
  sorry
end

end derivative_depends_only_on_x0_l93_93388


namespace mirasol_initial_amount_l93_93845

/-- 
Mirasol had some money in her account. She spent $10 on coffee beans and $30 on a tumbler. She has $10 left in her account.
Prove that the initial amount of money Mirasol had in her account is $50.
-/
theorem mirasol_initial_amount (spent_coffee : ℕ) (spent_tumbler : ℕ) (left_in_account : ℕ) :
  spent_coffee = 10 → spent_tumbler = 30 → left_in_account = 10 → 
  spent_coffee + spent_tumbler + left_in_account = 50 := 
by
  sorry

end mirasol_initial_amount_l93_93845


namespace quadratic_roots_range_no_real_k_for_reciprocal_l93_93220

theorem quadratic_roots_range (k : ℝ) (h : 12 * k + 4 > 0) : k > -1 / 3 ∧ k ≠ 0 :=
by
  sorry

theorem no_real_k_for_reciprocal (k : ℝ) : ¬∃ (x1 x2 : ℝ), (kx^2 - 2*(k+1)*x + k-1 = 0) ∧ (1/x1 + 1/x2 = 0) :=
by
  sorry

end quadratic_roots_range_no_real_k_for_reciprocal_l93_93220


namespace part_a_part_b_l93_93523

variable {n : ℕ}
variable {m : ℕ}
variable {A : Fin n → Point}
variable {P : Point}
variable {l : Line}
variable {a b : Fin n → ℝ}

-- Conditions from the problem:
axiom regular_polygon : is_regular_n_polygon A
axiom circumscribed : circumscribed_around_circle A P
axiom tangent_line : is_tangent_line l P
axiom distances_a : ∀ i, distance_to_line (A i) l = a i
axiom distances_b : ∀ i, distance_to_line (tangency_point (A i) (A (succ i % n)) P) l = b i

-- Part (a)
theorem part_a : (∏ i, (b i)) / (∏ i, (a i)) = C :=
sorry

-- Part (b)
theorem part_b (hn : n = 2 * m) : (∏ i in (Finset.range m).map (Embedding.finset Fin.succ), (a i)) / (∏ i in (Finset.range m).map (Embedding.finset Fin.ifeven), (a i)) = C :=
sorry

end part_a_part_b_l93_93523


namespace find_key_effective_expected_residents_to_use_algorithm_l93_93469

-- Define mailboxes and keys
def num_mailboxes : ℕ := 80
def initial_mailbox : ℕ := 37

-- Prove that the algorithm is effective
theorem find_key_effective : 
  ∀ (mailboxes : fin num_mailboxes) (keys : fin num_mailboxes), 
  { permutation : list (fin num_mailboxes) // permutation.nodup ∧ permutation.length = num_mailboxes ∧ 
    ∀ m, m ∈ permutation → (if m = initial_mailbox then m else (keys.filter (λ k, k ∈ permutation))) ≠ [] }
  :=
sorry

-- Prove the expected number of residents who will use the algorithm
theorem expected_residents_to_use_algorithm :
  ∑ i in finset.range num_mailboxes, (1 / (i + 1 : ℝ)) = (real.log 80 + 0.577 + (1 / (2 * 80)) : ℝ)
  :=
sorry

end find_key_effective_expected_residents_to_use_algorithm_l93_93469


namespace smallest_composite_no_prime_lt_20_l93_93060

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93060


namespace solution_l93_93785

open Real

variables (a b c A B C : ℝ)

-- Condition: In ΔABC, the sides opposite to angles A, B, and C are a, b, and c respectively
-- Condition: Given equation relating sides and angles in ΔABC
axiom eq1 : a * sin C / (1 - cos A) = sqrt 3 * c
-- Condition: b + c = 10
axiom eq2 : b + c = 10
-- Condition: Area of ΔABC
axiom eq3 : (1 / 2) * b * c * sin A = 4 * sqrt 3

-- The final statement to prove
theorem solution :
    (A = π / 3) ∧ (a = 2 * sqrt 13) :=
by
    sorry

end solution_l93_93785


namespace whole_numbers_between_cubicroots_l93_93703

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93703


namespace whole_numbers_count_between_cubic_roots_l93_93713

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93713


namespace count_whole_numbers_between_roots_l93_93761

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93761


namespace closest_integer_sqrt_35_l93_93365

theorem closest_integer_sqrt_35 : ∃ (n : ℤ), (5 : ℝ) < (sqrt 35) ∧ (sqrt 35) < (6 : ℝ) ∧ (n = 6) :=
by
  sorry

end closest_integer_sqrt_35_l93_93365


namespace part1_part2_l93_93233

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_part2_l93_93233


namespace sin_vertex_angle_of_isosceles_triangle_l93_93602

theorem sin_vertex_angle_of_isosceles_triangle
  (α : ℝ)
  (h : ∀ (K L M N : ℝ), let x₁ := K, y₁ := L, x₂ := M, y₂ := N in x₁ + y₁ = x₂ + y₂) :
  sin α = 4 / 5 := by
  sorry

end sin_vertex_angle_of_isosceles_triangle_l93_93602


namespace cylindrical_coordinates_of_point_l93_93563

noncomputable def cylindrical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x = -r then Real.pi else 0 -- From the step if cos θ = -1
  (r, θ, z)

theorem cylindrical_coordinates_of_point :
  cylindrical_coordinates (-5) 0 (-8) = (5, Real.pi, -8) :=
by
  -- placeholder for the actual proof
  sorry

end cylindrical_coordinates_of_point_l93_93563


namespace num_whole_numbers_between_l93_93700

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93700


namespace count_whole_numbers_between_cubes_l93_93675

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93675


namespace otimes_property_l93_93574

def otimes (a b : ℚ) : ℚ := (a^3) / b

theorem otimes_property : otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = 80 / 27 := by
  sorry

end otimes_property_l93_93574


namespace sum_of_abcd_l93_93836

theorem sum_of_abcd (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : ∀ x, x^2 - 8*a*x - 9*b = 0 → x = c ∨ x = d)
  (h2 : ∀ x, x^2 - 8*c*x - 9*d = 0 → x = a ∨ x = b) :
  a + b + c + d = 648 := sorry

end sum_of_abcd_l93_93836


namespace volume_of_stone_l93_93432

def width := 16
def length := 14
def full_height := 9
def initial_water_height := 4
def final_water_height := 9

def volume_before := length * width * initial_water_height
def volume_after := length * width * final_water_height

def volume_stone := volume_after - volume_before

theorem volume_of_stone : volume_stone = 1120 := by
  unfold volume_stone
  unfold volume_after volume_before
  unfold final_water_height initial_water_height width length
  sorry

end volume_of_stone_l93_93432


namespace find_equations_l93_93623

noncomputable def circleC1 := ∀ x y : ℝ, x^2 + y^2 = 2

noncomputable def tangent_point : (ℝ × ℝ) := (1, 1)

noncomputable def tangent_line := ∀ x y : ℝ, (x + y - 2 = 0)

noncomputable def circleC2_center : ℝ × ℝ := 
  let a := 2 in (a, 2 * a)

noncomputable def chord_length : ℝ := 4 * Real.sqrt 3

noncomputable def circleC2_equation : (ℝ × ℝ) → Prop := 
  λ p, (p.1 - 2)^2 + (p.2 - 4)^2 = 20

theorem find_equations :
  tangent_line tangent_point ∧ 
  circleC2_equation (2, 4) :=
sorry

end find_equations_l93_93623


namespace james_spent_6_dollars_l93_93317

-- Define the constants based on the conditions
def cost_milk : ℝ := 3
def cost_bananas : ℝ := 2
def tax_rate : ℝ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℝ := cost_milk + cost_bananas

-- Define the sales tax
def sales_tax : ℝ := total_cost_before_tax * tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_before_tax + sales_tax

-- The theorem to prove that James spent $6
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l93_93317


namespace sum_num_denom_l93_93885

theorem sum_num_denom (x : ℚ) (h : x = 5.1717171717) : (x.num + x.denom) = 611 := by
  sorry

end sum_num_denom_l93_93885


namespace salary_increase_90_yuan_l93_93383

-- Noncomputable theory, to handle real numbers
noncomputable theory

-- The regression line equation definition
def regression_line (x : ℝ) : ℝ := 60 + 90 * x

-- The statement to be proved
theorem salary_increase_90_yuan (increase_in_productivity : ℝ) (h : increase_in_productivity = 1) :
  regression_line increase_in_productivity - regression_line 0 = 90 :=
by
  rw [h]
  simp [regression_line]
  sorry

end salary_increase_90_yuan_l93_93383


namespace smallest_composite_no_prime_factors_less_than_20_l93_93011

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93011


namespace prob_5_shots_expected_number_shots_l93_93956

variable (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

def prob_exactly_five_shots : ℝ := 6 * p^3 * (1 - p)^2
def expected_shots : ℝ := 3 / p

theorem prob_5_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the probability of exactly 5 shots needed is as calculated
  prob_exactly_five_shots p = 6 * p^3 * (1 - p)^2 :=
by
  sorry

theorem expected_number_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the expected number of shots to hit all targets is as calculated
  expected_shots p = 3 / p :=
by
  sorry

end prob_5_shots_expected_number_shots_l93_93956


namespace sum_max_min_equals_two_l93_93883

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem sum_max_min_equals_two : 
  let y := f x in 
  ∃ (m n : ℝ), (∀ x, y ≤ m) ∧ (∀ x, n ≤ y) ∧ (m = 3) ∧ (n = -1) ∧ (m + n = 2) := 
by 
  sorry

end sum_max_min_equals_two_l93_93883


namespace min_max_f_l93_93353

noncomputable def f (x α β : ℝ) : ℝ := abs (cos x + α * cos (2 * x) + β * cos (3 * x))

theorem min_max_f :
  ∃ M : ℝ, M = (Real.sqrt 3) / 2 ∧
  ∀ α β : ℝ, max (λ x, f x α β) ℝ ≥ M :=
sorry

end min_max_f_l93_93353


namespace smallest_composite_no_prime_factors_less_than_20_l93_93006

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93006


namespace Nell_initial_cards_l93_93846

theorem Nell_initial_cards 
  (cards_given : ℕ)
  (cards_left : ℕ)
  (cards_given_eq : cards_given = 301)
  (cards_left_eq : cards_left = 154) :
  cards_given + cards_left = 455 := by
sorry

end Nell_initial_cards_l93_93846


namespace count_whole_numbers_between_cubes_l93_93681

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93681


namespace not_necessary_nor_sufficient_l93_93213

-- Given conditions: Differentiable function f(x) on (0, 2) with its derivative f'(x)
def differentiable_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x ∈ set.Ioo a b, has_deriv_at f (deriv f x) x

-- Problem statement: Proving the conditions are neither necessary nor sufficient
theorem not_necessary_nor_sufficient
  (f : ℝ → ℝ)
  (h_diff : differentiable_on_interval f 0 2) :
  (∃ x1 x2 ∈ set.Ioo 0 2, deriv f x1 = 0 ∧ deriv f x2 = 0) ↔ 
  (∃ y1 y2 ∈ set.Ioo 0 2, is_extreme f y1 ∧ is_extreme f y2) → false :=
by
  sorry

end not_necessary_nor_sufficient_l93_93213


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93746

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93746


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93075

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93075


namespace sum_first_9_terms_l93_93799

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a (n + 1) = a n + d ∧ a (m + 1) = a m + d

-- Define the sum of the first n terms in an arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n * (a(1) + a(n))) / 2

-- Given condition
variables (a : ℕ → ℝ) [arithmetic_sequence a]

theorem sum_first_9_terms (h : a(4) + a(6) = 16) : sum_of_first_n_terms a 9 = 72 :=
sorry

end sum_first_9_terms_l93_93799


namespace total_volume_of_five_cubes_l93_93925

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end total_volume_of_five_cubes_l93_93925


namespace algorithm_effective_expected_residents_using_stick_l93_93477

-- Part (a): Prove the algorithm is effective
theorem algorithm_effective :
  ∀ (n : ℕ) (keys : ℕ → ℕ) (start : ℕ),
  (1 ≤ start ∧ start ≤ n) →
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (∃ (sequence : ℕ → ℕ), 
     (sequence 0 = start) ∧ 
     (∀ (i : ℕ), sequence (i + 1) = keys (sequence i)) ∧ 
     (∃ m, sequence m = start)) :=
by
  sorry

-- Part (b): Expected number of residents who need to use the stick is approximately 4.968
-- Here we outline the theorem related to the expected value part.
theorem expected_residents_using_stick (n : ℕ) (H : n = 80) :
  ∀ (keys : ℕ → ℕ),
  (∀ k, 1 ≤ keys k ∧ keys k ≤ n) →
  (let Hn := ∑ i in range n, 1 / i
   in abs(Hn - 4.968) < ε) :=
by
  sorry

end algorithm_effective_expected_residents_using_stick_l93_93477


namespace second_person_average_pages_per_day_l93_93572

-- Define the given conditions.
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def deshaun_pages_per_book : ℕ := 320
def second_person_percentage : ℝ := 0.75

-- Calculate the total number of pages DeShaun read.
def deshaun_total_pages : ℕ := deshaun_books * deshaun_pages_per_book

-- Calculate the total number of pages the second person read.
def second_person_total_pages : ℕ := (second_person_percentage * deshaun_total_pages).toNat

-- Prove the average number of pages the second person read per day.
def average_pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

theorem second_person_average_pages_per_day :
  average_pages_per_day second_person_total_pages summer_days = 180 :=
by
  sorry

end second_person_average_pages_per_day_l93_93572


namespace compound_interest_correct_l93_93926

noncomputable def principal : ℝ := 5000
noncomputable def annual_rate : ℝ := 0.04
noncomputable def compounding_per_year : ℕ := 2
noncomputable def years : ℕ := 20

def total_periods : ℕ := compounding_per_year * years
def rate_per_period : ℝ := annual_rate / compounding_per_year

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem compound_interest_correct :
  compound_interest principal rate_per_period total_periods ≈ 11040.20 :=
by
  sorry

end compound_interest_correct_l93_93926


namespace relationship_between_f_neg1_and_f_1_l93_93216

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * f' 2
noncomputable def f' (x : ℝ) : ℝ

-- Define the condition that f' 2 = -4
axiom f'_2_is_neg4 : f' 2 = -4

-- Define the proof to state f(-1) > f(1)
theorem relationship_between_f_neg1_and_f_1 :
  f(-1) > f(1) :=
sorry

end relationship_between_f_neg1_and_f_1_l93_93216


namespace smallest_composite_no_prime_lt_20_l93_93055

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93055


namespace vertex_angle_measure_l93_93792

namespace TriangleProof

variables (a b h : ℝ) (θ φ : ℝ)
variable (isosceles_acute : 0 < θ ∧ θ < π / 2)
variable (congruent_side_condition : a ^ 2 = 3 * b * h)
variable (base_eq : b = 2 * a * cos θ)
variable (height_eq : h = a * sin θ)

theorem vertex_angle_measure : 
  (φ = 180 - 2 * θ) ∧ (sin(2 * θ) = 1 / 3) ∧ φ = 160.52 :=
by
  sorry

end TriangleProof

end vertex_angle_measure_l93_93792


namespace part1_part2_l93_93237

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1 (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, f x a ≥ 4 ↔ x ≤ (3 / 2 : ℝ) ∨ x ≥ (11 / 2 : ℝ) :=
by 
  rw h
  sorry

theorem part2 (h : ∀ x a : ℝ, f x a ≥ 4) :
  ∀ a : ℝ, (a - 1)^2 ≥ 4 ↔ a ≤ -1 ∨ a ≥ 3 :=
by 
  sorry

end part1_part2_l93_93237


namespace find_m_l93_93598

theorem find_m (m : ℤ) (h1 : -180 ≤ m ∧ m ≤ 180) (h2 : Real.sin (m * Real.pi / 180) = Real.cos (810 * Real.pi / 180)) :
  m = 0 ∨ m = 180 :=
sorry

end find_m_l93_93598


namespace smallest_composite_no_prime_factors_less_than_20_l93_93002

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93002


namespace whole_numbers_count_between_cubic_roots_l93_93716

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93716


namespace math_problem_l93_93797

noncomputable def parametric_to_general (t : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + t ∧ y = t + 1 ∧ (x - y - 1 = 0)

noncomputable def polar_to_rect (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), ρ^2 - 4 * ρ * cos θ + 3 = 0 ∧ x = ρ * cos θ ∧ y = ρ * sin θ ∧ (x^2 + y^2 - 4 * x + 3 = 0)

noncomputable def chord_length (A B : ℝ × ℝ) : Prop :=
  let C := (2, 0) in
  let r := 1 in
  let d := 1 / Real.sqrt 2 in
  let AB := (A, B) in
  ∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ abs ((A.1 - B.1)^2 + (A.2 - B.2)^2)^0.5 = Real.sqrt 2

theorem math_problem : ∃ (t : ℝ) (x y : ℝ), parametric_to_general t ∧ polar_to_rect x y ∧ chord_length (2, 1) (1 - Real.sqrt 2 / 2, 2 - Real.sqrt 2 / 2) :=
  sorry

end math_problem_l93_93797


namespace transformations_preserve_pattern_l93_93988

-- Define the properties of the pattern on line $\ell$
def is_repeating_pattern (ℓ : Type) := 
  ∃ (triangles rectangles : list ℓ),
    ∀ (n : ℕ), (triangles.nth n).is_some ∧ (rectangles.nth n).is_some

-- Define the properties of alternating triangles
def alternating_triangles (ℓ : Type) (triangles : list ℓ) :=
  ∀ (n : ℕ), (triangles.nth n).is_some ∧ (
    if even n then 
      (triangles.nth n).get = up_triangle
    else 
      (triangles.nth n).get = down_triangle
  )

-- Define the conditions
variables (ℓ : Type) [inhabited ℓ]
variables triangles rectangles : list ℓ

axiom infinite_line_pattern : is_repeating_pattern ℓ ∧ 
  alternating_triangles ℓ triangles ∧ 
  (∀ n, ∃ rect, rectangles.nth n = some rect ∧ rect ∈ rectangles)

-- The statement to be proved
theorem transformations_preserve_pattern (ℓ : Type) [inhabited ℓ] 
  (triangles rectangles : list ℓ) 
  (h : is_repeating_pattern ℓ ∧ 
       alternating_triangles ℓ triangles ∧ 
       (∀ n, ∃ rect, rectangles.nth n = some rect ∧ rect ∈ rectangles)) :
  {τ : set (ℓ → ℓ) | ∃ t ∈ τ, t ≠ id} = 3 := 
sorry

end transformations_preserve_pattern_l93_93988


namespace local_min_value_at_zero_l93_93226

noncomputable def f (x m : ℝ) : ℝ := (x^2 + x + m) * Real.exp x

def local_max_at_x_neg3 (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, (abs (y + 3) < δ) → (f y ≤ f (-3))

theorem local_min_value_at_zero :
  ∀ m : ℝ, local_max_at_x_neg3 (λ x, f x m) (-3) → f 0 (-1) = -1 :=
begin
  intros m h,
  -- rest of the proof
  sorry
end

end local_min_value_at_zero_l93_93226


namespace hyperbola_parabola_focus_l93_93243

theorem hyperbola_parabola_focus (m n : ℝ) 
  (h1 : ∃ x y, (x^2 / m - y^2 / n = 1)) 
  (h2 : ∀ x y, y = (1 / 12) * x^2 → (0, 3) = (0, y)) 
  (h3 : ∃ e, e = 3) 
  (h4 : ∀ f g, 9 = -n - m → e = 3 / real.sqrt (-n)) 
  : m = -8 ∧ n = -1 := 
sorry

end hyperbola_parabola_focus_l93_93243


namespace total_number_of_triangles_in_figure_l93_93555

theorem total_number_of_triangles_in_figure :
  ∀ (A B C M N : Type) (isosceles : IsIsosceles A B C) (base_length : BaseLength A B 2) 
    (apex : Apex C) (midpoints : Midpoints M N) (horizontal : Horizontal MN) (segments : Segments A B C M N), 
  total_number_of_triangles (figure A B C M N) = 5 :=
by
  sorry

end total_number_of_triangles_in_figure_l93_93555


namespace whole_numbers_between_cubicroots_l93_93707

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93707


namespace factor_roots_l93_93592

theorem factor_roots (t : ℝ) : (x - t) ∣ (8 * x^2 + 18 * x - 5) ↔ (t = 1 / 4 ∨ t = -5) :=
by
  sorry

end factor_roots_l93_93592


namespace prime_divisors_390_num_prime_divisors_390_l93_93670

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

theorem prime_divisors_390 : prime_factors 390 = {2, 3, 5, 13} :=
by {
  sorry
}

theorem num_prime_divisors_390 : (prime_factors 390).to_finset.card = 4 :=
by {
  rw prime_divisors_390,
  simp,
}

end prime_divisors_390_num_prime_divisors_390_l93_93670


namespace solve_for_y_l93_93267

theorem solve_for_y (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
  sorry

end solve_for_y_l93_93267


namespace bicycles_sold_saturday_l93_93362

variable (S : ℕ)

theorem bicycles_sold_saturday :
  let net_increase_friday := 15 - 10
  let net_increase_saturday := 8 - S
  let net_increase_sunday := 11 - 9
  (net_increase_friday + net_increase_saturday + net_increase_sunday = 3) → 
  S = 12 :=
by
  intros h
  sorry

end bicycles_sold_saturday_l93_93362


namespace puppys_speed_l93_93321

theorem puppys_speed (saif_speed kareena_speed puppy_distance puppy_time : ℝ)
    (h1 : saif_speed = 5)
    (h2 : kareena_speed = 6)
    (h3 : puppy_distance = 10)
    (h4 : puppy_time = 1) : 
    puppy_distance / puppy_time = 10 := 
by
  calc
    puppy_distance / puppy_time = 10 / 1 : by rw [h3, h4]
                               ... = 10 : by norm_num

end puppys_speed_l93_93321


namespace additional_workers_needed_l93_93178

theorem additional_workers_needed :
  let initial_workers := 4
  let initial_parts := 108
  let initial_hours := 3
  let target_parts := 504
  let target_hours := 8
  (target_parts / target_hours) / (initial_parts / (initial_hours * initial_workers)) - initial_workers = 3 := by
  sorry

end additional_workers_needed_l93_93178


namespace smallest_composite_no_prime_factors_less_than_20_l93_93023

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93023


namespace sailboat_speed_max_power_l93_93398

-- Define the conditions
def S : ℝ := 7
def v0 : ℝ := 6.3

-- Define the power function N as a function of sailboat speed v
def N (B ρ v : ℝ) : ℝ :=
  (B * S * ρ / 2) * (v0 ^ 2 * v - 2 * v0 * v ^ 2 + v ^ 3)

-- State the theorem we need to prove
theorem sailboat_speed_max_power (B ρ : ℝ) : 
  ∀ (v : ℝ), 
  (∃ v : ℝ, v = 2.1) :=
begin
  -- Proof goes here
  sorry
end

end sailboat_speed_max_power_l93_93398


namespace arrangement_schemes_l93_93512

theorem arrangement_schemes (n m k: ℕ) (h1 : n = 5) (h2 : m = 2) (h3 : k = 4):
  (1 / 2) * (Nat.choose k (k/2)) * (Nat.perm n m) = 60 :=
by {
  sorry
}

end arrangement_schemes_l93_93512


namespace smallest_fourth_number_l93_93989

theorem smallest_fourth_number :
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  let n := 10 * a + b in
  (177 + n = 4 * (33 + a + b)) ∧ (n = 93) := 
sorry

end smallest_fourth_number_l93_93989


namespace lines_pass_through_N_MN_passes_through_fixed_K_l93_93618

-- Segment AB and point M on AB, different from A and B
variables {A B M C D N : Point}
-- Equilateral triangles constructed on the same side of AB
variable (equilateral_1 : EquilateralTriangle A M C)
variable (equilateral_2 : EquilateralTriangle B M D)

-- Circumcircles' intersection points
variable (circumcircle_AMC : Circumcircle (Triangle A M C))
variable (circumcircle_BMD : Circumcircle (Triangle B M D))

-- M and N are the intersection points of the circumcircles
variable (intersection_1 : circumcircle_AMC.contains M)
variable (intersection_2 : circumcircle_AMC.contains N)
variable (intersection_3 : circumcircle_BMD.contains M)
variable (intersection_4 : circumcircle_BMD.contains N)

-- Prove that lines AD and BC pass through point N
theorem lines_pass_through_N :
  Collinear [A, D, N] ∧ Collinear [B, C, N] := by
  sorry

-- Prove that there is a fixed point K such that MN always passes through K
theorem MN_passes_through_fixed_K (fixed_point_K : Point) :
  ∀ (M : Point), M ∈ Segment A B → Collinear [fixed_point_K, M, N] := by
  sorry

end lines_pass_through_N_MN_passes_through_fixed_K_l93_93618


namespace amount_kept_by_Tim_l93_93440

-- Define the conditions
def totalAmount : ℝ := 100
def percentageGivenAway : ℝ := 0.2

-- Prove the question == answer
theorem amount_kept_by_Tim : totalAmount - totalAmount * percentageGivenAway = 80 :=
by
  -- Here the proof would take place
  sorry

end amount_kept_by_Tim_l93_93440


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93071

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93071


namespace value_range_of_quadratic_function_l93_93888

def f (x : ℝ) := x^2 - 4 * x

theorem value_range_of_quadratic_function :
  (∀ x, 0 ≤ x ∧ x ≤ 5 → (f x ≥ -4 ∧ f x ≤ 5)) :=
by
  assume x hx
  sorry

end value_range_of_quadratic_function_l93_93888


namespace probability_exactly_5_shots_expected_number_of_shots_l93_93976

-- Define the problem statement and conditions with Lean definitions
variables (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

-- Part (a): Probability of 5 shots needed
theorem probability_exactly_5_shots : 
  (6 * p^3 * (1 - p)^2) = probability_exactly_5_shots (p : ℝ) :=
sorry

-- Part (b): Expected number of shots needed
theorem expected_number_of_shots :
  (3 / p) = expected_number_of_shots (p : ℝ) :=
sorry

end probability_exactly_5_shots_expected_number_of_shots_l93_93976


namespace smallest_composite_no_prime_factors_lt_20_l93_93151

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93151


namespace tan_alpha_eq_three_sin_cos_l93_93181

theorem tan_alpha_eq_three_sin_cos (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 :=
by 
  sorry

end tan_alpha_eq_three_sin_cos_l93_93181


namespace possible_scenario_exists_l93_93897

-- Define the number of races
def num_races := 3

-- Define properties on how many times each runner beats another
def A_beats_B (race_results : List (Prod (Fin num_races) (Prod (Fin num_races) (Fin num_races)))) : Prop :=
  (race_results.filter (λ result, result.2.1 == 0)).length > num_races / 2

def B_beats_C (race_results : List (Prod (Fin num_races) (Prod (Fin num_races) (Fin num_races)))) : Prop :=
  (race_results.filter (λ result, result.2.1 == 1)).length > num_races / 2

def C_beats_A (race_results : List (Prod (Fin num_races) (Prod (Fin num_races) (Fin num_races)))) : Prop :=
  (race_results.filter (λ result, result.2.1 == 2)).length > num_races / 2

-- The theorem stating the possibility of the described scenario
theorem possible_scenario_exists : 
  ∃ (race_results : List (Prod (Fin num_races) (Prod (Fin num_races) (Fin num_races)))),
    A_beats_B race_results ∧ B_beats_C race_results ∧ C_beats_A race_results :=
begin
  sorry
end

end possible_scenario_exists_l93_93897


namespace rectangle_of_equal_inscribed_radii_l93_93186

variable {Point : Type*} [Geometry Point]

-- Definitions for convex quadrilateral and inscribed circle radii equality
def convex_quadrilateral (A B C D : Point) : Prop := 
  convex_quadrilateral A B C D

def equal_inscribed_radii (A B C D : Point) (r : ℝ) : Prop :=
  inscribed_radius A B C = r ∧
  inscribed_radius B C D = r ∧
  inscribed_radius C D A = r ∧
  inscribed_radius D A B = r

-- Main theorem statement
theorem rectangle_of_equal_inscribed_radii 
  (A B C D : Point) (r : ℝ)
  (h_convex : convex_quadrilateral A B C D)
  (h_equal_radii : equal_inscribed_radii A B C D r) :
  is_rectangle A B C D :=
sorry

end rectangle_of_equal_inscribed_radii_l93_93186


namespace Jeff_probability_is_31_90_l93_93320

noncomputable def Jeff_spins : Prop :=
  let possible_starts := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let is_multiple_of_3 (n : ℕ) := n % 3 = 0
  let starts_at_multiple_of_3 := Finset.filter is_multiple_of_3 (Finset.range 11)
  let starts_at_one_more_than_multiple_of_3 := Finset.filter (λ n, (n % 3 = 1)) (Finset.range 11)
  let starts_at_one_less_than_multiple_of_3 := Finset.filter (λ n, (n % 3 = 2)) (Finset.range 11)
  let probability_of_starting := λ N : Finset ℕ, (N.card : ℚ) / 10
  
  let fair_spinner := [1, -1, -1] -- 1 represents one space right, -1 represents one space left
  let transition_probability (start : ℕ) : ℚ :=
    (fair_spinner.filter (λ x, is_multiple_of_3 (start + x))).card / 3
  
  let probability_of_ending_multiple_of_3 :=
    probability_of_starting starts_at_multiple_of_3 * (transition_probability 1 + transition_probability (-1)) +
    probability_of_starting starts_at_one_more_than_multiple_of_3 * transition_probability 1 * transition_probability 1 +
    probability_of_starting starts_at_one_less_than_multiple_of_3 * transition_probability (-1) * transition_probability (-1)
    
  probability_of_ending_multiple_of_3 = 31 / 90

theorem Jeff_probability_is_31_90 : Jeff_spins :=
by
  sorry

end Jeff_probability_is_31_90_l93_93320


namespace not_necessary_nor_sufficient_l93_93214

-- Given conditions: Differentiable function f(x) on (0, 2) with its derivative f'(x)
def differentiable_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x ∈ set.Ioo a b, has_deriv_at f (deriv f x) x

-- Problem statement: Proving the conditions are neither necessary nor sufficient
theorem not_necessary_nor_sufficient
  (f : ℝ → ℝ)
  (h_diff : differentiable_on_interval f 0 2) :
  (∃ x1 x2 ∈ set.Ioo 0 2, deriv f x1 = 0 ∧ deriv f x2 = 0) ↔ 
  (∃ y1 y2 ∈ set.Ioo 0 2, is_extreme f y1 ∧ is_extreme f y2) → false :=
by
  sorry

end not_necessary_nor_sufficient_l93_93214


namespace hundreds_digit_of_factorial_subtraction_l93_93449

theorem hundreds_digit_of_factorial_subtraction : (30.factorial - 25.factorial) % 1000 / 100 % 10 = 0 :=
by
  sorry

end hundreds_digit_of_factorial_subtraction_l93_93449


namespace whole_numbers_count_between_cubic_roots_l93_93720

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93720


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93070

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93070


namespace gcd_of_16_and_12_l93_93455

theorem gcd_of_16_and_12 : Nat.gcd 16 12 = 4 := by
  sorry

end gcd_of_16_and_12_l93_93455


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93067

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93067


namespace jerry_has_7_figures_l93_93818

noncomputable def action_figures_already_have (total_need: ℕ) (cost_each: ℕ) (total_cost_needed: ℕ) : ℕ :=
  total_need - (total_cost_needed / cost_each)

theorem jerry_has_7_figures (total_need: ℕ) (cost_each: ℕ) (total_cost_needed: ℕ) 
  (h1 : total_need = 16) (h2 : cost_each = 8) (h3 : total_cost_needed = 72) : 
  action_figures_already_have total_need cost_each total_cost_needed = 7 :=
by
  simp [action_figures_already_have, h1, h2, h3]
  sorry

end jerry_has_7_figures_l93_93818


namespace average_age_of_three_l93_93823

theorem average_age_of_three (Tonya_age John_age Mary_age : ℕ)
  (h1 : John_age = 2 * Mary_age)
  (h2 : Tonya_age = 2 * John_age)
  (h3 : Tonya_age = 60) :
  (Tonya_age + John_age + Mary_age) / 3 = 35 := by
  sorry

end average_age_of_three_l93_93823


namespace largest_angle_of_obtuse_isosceles_triangle_l93_93905

def triangleXYZ : Type := { XYZ : Type // XYZ = Triangle }
def isosceles_triangle (T : triangleXYZ) : Prop := Isosceles T.val
def obtuse_triangle (T : triangleXYZ) : Prop := Obtuse T.val
def angle_X_30_degrees (T : triangleXYZ) : Prop := Angle T.val X = 30

def largest_angle_measure (T : triangleXYZ) : ℕ := 120

theorem largest_angle_of_obtuse_isosceles_triangle (T : triangleXYZ) 
  (h1 : isosceles_triangle T) 
  (h2 : obtuse_triangle T) 
  (h3 : angle_X_30_degrees T) : 
  Angle T.val (largest_interior_angle T.val) = largest_angle_measure T :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l93_93905


namespace whole_numbers_between_l93_93729

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93729


namespace count_whole_numbers_between_roots_l93_93755

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93755


namespace smallest_composite_no_prime_factors_lt_20_l93_93158

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93158


namespace unique_photographs_either_fr_J_or_tk_O_l93_93789

variable (P: Type) -- P represents the type of photographs
variable (T J _: P → Prop) -- T p means "p was taken by Octavia"; J p means "p was framed by Jack"

-- Given conditions
variable (h1 : ∃ (S : Finset P), S.filter J.card = 24 ∧ S.filter T.card = 36 ∧ S.filter (λ p, T p ∧ J p).card = 24)
variable (h2 : ∃ (U : Finset P), U.card = 12 ∧ ∀ p ∈ U, J p ∧ ¬ T p)

-- The final theorem
theorem unique_photographs_either_fr_J_or_tk_O : 
  ∃ (V : Finset P), V.card = 48 ∧ ∀ p, (T p ∨ J p) ↔ p ∈ V :=
sorry

end unique_photographs_either_fr_J_or_tk_O_l93_93789


namespace tim_kept_amount_l93_93442

-- Definitions as direct conditions
def total_winnings : ℝ := 100
def percentage_given_away : ℝ := 20 / 100

-- The mathematically equivalent proof problem as a theorem statement
theorem tim_kept_amount : total_winnings - (percentage_given_away * total_winnings) = 80 := by
  sorry

end tim_kept_amount_l93_93442


namespace correct_derivative_option_D_l93_93578

theorem correct_derivative_option_D : 
  (∃ x, (differentiable_at ℝ (λ x : ℝ, real.sqrt x ^ 3) x ∧ deriv (λ x : ℝ, real.sqrt (x ^ 3)) x = (3 / 2) * real.sqrt x)) 
  ∧ 
  ¬ (∃ x, deriv (λ x : ℝ, 1 - x ^ 2) x = 1 - 2 * x) 
  ∧ 
  ¬ (deriv (λ x : ℝ, real.cos (real.pi / 6)) 0 = - real.sin (real.pi / 6))
  ∧ 
  ¬ (∃ x, deriv (λ x : ℝ, real.log (2 * x)) x = 1 / (2 * x)) :=
begin
  sorry
end

end correct_derivative_option_D_l93_93578


namespace points_on_edges_of_cube_l93_93289

noncomputable def number_of_points_on_cube (edge_length : ℝ) : ℕ :=
  if h : edge_length = 1 then
    6
  else
    0

theorem points_on_edges_of_cube :
  ∀ (P A C1 : ℝ × ℝ × ℝ),
    (edge_length = 1) →
    (|PA| + |PC1| = 2) →
    number_of_points_on_cube edge_length = 6 :=
  by
    intros P A C1 edge_length_eq PA_PC1_eq
    sorry

end points_on_edges_of_cube_l93_93289


namespace smallest_positive_value_is_A_l93_93163

noncomputable def expr_A : ℝ := 12 - 4 * Real.sqrt 8
noncomputable def expr_B : ℝ := 4 * Real.sqrt 8 - 12
noncomputable def expr_C : ℝ := 20 - 6 * Real.sqrt 10
noncomputable def expr_D : ℝ := 60 - 15 * Real.sqrt 16
noncomputable def expr_E : ℝ := 15 * Real.sqrt 16 - 60

theorem smallest_positive_value_is_A :
  expr_A = 12 - 4 * Real.sqrt 8 ∧ 
  expr_B = 4 * Real.sqrt 8 - 12 ∧ 
  expr_C = 20 - 6 * Real.sqrt 10 ∧ 
  expr_D = 60 - 15 * Real.sqrt 16 ∧ 
  expr_E = 15 * Real.sqrt 16 - 60 ∧ 
  expr_A > 0 ∧ 
  expr_A < expr_C := 
sorry

end smallest_positive_value_is_A_l93_93163


namespace part1_part2_l93_93630

noncomputable def f (x : ℝ) : ℝ := 3 + 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2

theorem part1 (A : ℝ) (h1 : f A = 5) : A = π / 3 :=
by
  sorry

theorem part2 (a b c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : A = π / 3) : 
  let s := (b * c * sin A) / 2
  in s ≤ sqrt 3 :=
by
  sorry

end part1_part2_l93_93630


namespace number_of_whole_numbers_between_cubicroots_l93_93737

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93737


namespace num_teams_at_least_7_num_teams_at_least_15_l93_93296

-- Part (a)
theorem num_teams_at_least_7 (teams : Type) [fintype teams] 
  (h : ∀ (A B : teams), ∃ (C : teams), C ≠ A ∧ C ≠ B ∧ C defeats A ∧ C defeats B) :
  fintype.card teams ≥ 7 :=
sorry

-- Part (b)
example (T1 T2 T3 T4 T5 T6 T7 : teams)
  (h_def : ∀ (A B : teams), ∃ (C : teams), C defeats A ∧ C defeats B) :
  true :=
by trivial  -- We do not need a proof here; just an example exists.

-- Part (c)
theorem num_teams_at_least_15 (teams : Type) [fintype teams]
  (h : ∀ (A B C : teams), ∃ (D : teams), D ≠ A ∧ D ≠ B ∧ D ≠ C ∧ D defeats A ∧ D defeats B ∧ D defeats C) :
  fintype.card teams ≥ 15 :=
sorry

end num_teams_at_least_7_num_teams_at_least_15_l93_93296


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93069

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93069


namespace inequality_lemma_l93_93372

variable {a b : ℝ} (n : ℕ)

/-- Prove the inequality for given conditions -/
theorem inequality_lemma
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (n_pos : 0 < n) :
  (a + b)^n - a^n - b^n >= (2^n - 2) / (2^(n - 2)) * ab(a + b)^(n - 2) :=
by sorry

end inequality_lemma_l93_93372


namespace fraction_of_field_planted_l93_93590

theorem fraction_of_field_planted (a b : ℕ) (d : ℝ) :
  a = 5 → b = 12 → d = 3 →
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let side_square := (d * hypotenuse - d^2)/(a + b - 2 * d)
  let area_square := side_square^2
  let area_triangle : ℝ := 1/2 * a * b
  let planted_area := area_triangle - area_square
  let fraction_planted := planted_area / area_triangle
  fraction_planted = 9693/10140 := by
  sorry

end fraction_of_field_planted_l93_93590


namespace smallest_expression_at_x_neg_3_l93_93273

theorem smallest_expression_at_x_neg_3 :
  let x := -3 in 
  (x + 3)^2 = 0 ∧ (∀ E, E ∈ {x^2 - 3, (x - 3)^2, x^2, (x + 3)^2, x^2 + 3} → (x + 3)^2 ≤ E) :=
by
  sorry

end smallest_expression_at_x_neg_3_l93_93273


namespace smallest_composite_no_prime_factors_below_20_l93_93122

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93122


namespace num_whole_numbers_between_l93_93695

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93695


namespace discount_problem_l93_93776

-- Let's define the propositions in Lean
def all_books_discounted (discount: ℝ) := ∀ b : ℕ, b ∈ bookstore → discounted_by b discount

def statement_II := ∃ b : ℕ, b ∈ bookstore ∧ ¬ discounted_by b 20
def statement_IV := ¬∀ b : ℕ, b ∈ bookstore → discounted_by b 20

-- Prove that if all_books_discounted 20 is false, then statement_II and statement_IV hold.
theorem discount_problem (h : ¬ all_books_discounted 20) : statement_II ∧ statement_IV :=
by
  sorry

end discount_problem_l93_93776


namespace inequality_c_l93_93351

theorem inequality_c (n : ℕ) (a : ℕ → ℝ) (c : ℝ)
  (h₀ : a n = 0)
  (h₁ : ∀ k < n, a k = c + (∑ i in finset.range (n - k), a (i - k) * (a i + a (i + 1)))) :
  c ≤ 1 / (4 * n) :=
sorry

end inequality_c_l93_93351


namespace f_def_pos_l93_93774

-- Define f to be an odd function
variable (f : ℝ → ℝ)
-- Define f as an odd function
axiom odd_f (x : ℝ) : f (-x) = -f x

-- Define f when x < 0
axiom f_def_neg (x : ℝ) (h : x < 0) : f x = (Real.cos (3 * x)) + (Real.sin (2 * x))

-- State the theorem to be proven:
theorem f_def_pos (x : ℝ) (h : 0 < x) : f x = - (Real.cos (3 * x)) + (Real.sin (2 * x)) :=
sorry

end f_def_pos_l93_93774


namespace a_1012_eq_1_S_2023_eq_2023_l93_93419

noncomputable def S (n : ℕ) : ℕ := sorry -- Need exact definition of S.

variable (a : ℕ → ℕ) -- Definition of the sequence {a_n}.

def sum_to_n (n : ℕ) := S n  -- sum_to_n is the sum of the first n terms of the sequence {a_n}

-- Given conditions of the problem
axiom h1 : S 1023 - S 1000 = 23

-- Proving that a_1012 = 1
theorem a_1012_eq_1 : a 1012 = 1 := 
by sorry

-- Proving that S 2023 = 2023
theorem S_2023_eq_2023 : S 2023 = 2023 :=
by sorry

end a_1012_eq_1_S_2023_eq_2023_l93_93419


namespace smallest_composite_no_prime_factors_less_than_20_l93_93108

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93108


namespace problem_statement_l93_93547

theorem problem_statement : 15 * 30 + 45 * 15 + 15 * 15 = 1350 :=
by
  sorry

end problem_statement_l93_93547


namespace no_solutions_to_inequality_l93_93174

theorem no_solutions_to_inequality (x : ℝ) : ¬(3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  intro h,
  -- Simplify the inequality by dividing each term by 3
  have h_simplified : x^2 + 3 * x + 4 ≤ 0 := by linarith,
  -- Compute the discriminant of the quadratic expression to show it's always positive
  let a := (1 : ℝ),
  let b := (3 : ℝ),
  let c := (4 : ℝ),
  let discriminant := b^2 - 4 * a * c,
  have h_discriminant : discriminant < 0 := by norm_num,
  -- Since discriminant is negative, the quadratic has no real roots, thus x^2 + 3x + 4 > 0
  have h_positive : ∀ x, x^2 + 3 * x + 4 > 0 := 
    by {
      intro x,
      apply (quadratic_not_negative_of_discriminant neg_discriminant).mp,
      exact h_discriminant,
    },
  exact absurd (show x^2 + 3 * x + 4 ≤ 0 from h_simplified) (lt_irrefl 0 (h_positive x)),
}

end no_solutions_to_inequality_l93_93174


namespace perp_iff_x_eq_1_inequality_solution_l93_93249

variables {a x : ℝ}
variables (m n : ℝ × ℝ)

-- Conditions
def vector_m (a x : ℝ) : ℝ × ℝ := (a^x, -a)
def vector_n (a x : ℝ) : ℝ × ℝ := (a^x, a)

-- Theorem for Part 1
theorem perp_iff_x_eq_1 (h₁ : a > 0) (h₂ : a ≠ 1) :
  (vector_m a x).dot (vector_n a x) = 0 ↔ x = 1 :=
by sorry

-- Theorem for Part 2
theorem inequality_solution (h₁ : a > 0) (h₂ : a ≠ 1) :
  (abs (vector_m a x + vector_n a x) < abs (vector_m a x - vector_n a x)) ↔
  (0 < a ∧ a < 1 → 1 < x) ∧ (1 < a → x < 1) :=
by sorry

end perp_iff_x_eq_1_inequality_solution_l93_93249


namespace equation_of_line_through_focus_l93_93656

-- Definition of the parabola
def parabola (x : ℝ) : ℝ := (1 / 4) * x^2

-- Definition of the focus of the parabola
def focus : ℝ × ℝ := (0, 1 / 4)

-- Definition of the equation of the line
def line_through_focus_and_perpendicular_to_axis (y : ℝ) : Prop :=
  y = 1 / 4

-- The theorem we need to prove
theorem equation_of_line_through_focus :
  ∃ y : ℝ, line_through_focus_and_perpendicular_to_axis y :=
sorry

end equation_of_line_through_focus_l93_93656


namespace count_whole_numbers_between_cubes_l93_93680

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93680


namespace whole_numbers_between_l93_93728

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93728


namespace question_l93_93660

def U := {1, 2, 3, 4, 5}
def M := {1, 2, 4}
def N := {2, 4, 5}

def complement (S : Set ℕ) : Set ℕ := U \ S

theorem question :
  (complement M ∩ complement N) = {3} :=
  by 
    -- Proof steps can be inserted here
    sorry

end question_l93_93660


namespace snow_probability_first_week_february_l93_93849

theorem snow_probability_first_week_february :
  let prob_no_snow_first_3_days : ℚ := (3 / 4) ^ 3,
      prob_no_snow_next_4_days : ℚ := (2 / 3) ^ 4,
      prob_no_snow_week : ℚ := prob_no_snow_first_3_days * prob_no_snow_next_4_days,
      prob_snow_at_least_once_week : ℚ := 1 - prob_no_snow_week
  in
    prob_snow_at_least_once_week = 11 / 12 :=
begin
  sorry
end

end snow_probability_first_week_february_l93_93849


namespace no_solution_l93_93170

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end no_solution_l93_93170


namespace count_whole_numbers_between_cubes_l93_93678

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93678


namespace part1_part2_l93_93238

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l93_93238


namespace geometric_seq_a3_equals_3_l93_93187

variable {a : ℕ → ℝ}
variable (h_geometric : ∀ m n p q, m + n = p + q → a m * a n = a p * a q)
variable (h_pos : ∀ n, n > 0 → a n > 0)
variable (h_cond : a 2 * a 4 = 9)

theorem geometric_seq_a3_equals_3 : a 3 = 3 := by
  sorry

end geometric_seq_a3_equals_3_l93_93187


namespace algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93480

-- Definition of the mailbox setting and conditions
def mailbox_count : ℕ := 80

def apartment_resident_start : ℕ := 37

noncomputable def randomized_keys_placed_in_mailboxes : Bool :=
  true  -- This is a placeholder; actual randomness is abstracted

-- Statement of the problem
theorem algorithm_will_find_key (mailboxes keys : Fin mailbox_count) (start : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ (sequence : ℕ → Fin mailbox_count), ∀ n : ℕ, sequence n ≠ start → sequence (n+1) ≠ start → 
  (sequence 0 = start ∨ ∃ k : ℕ, sequence k = keys.start → keys = sequence (k+1)).
  sorry

theorem expected_number_of_residents_using_algorithm 
  (mailboxes keys : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ n : ℝ, n ≈ 4.968.
  sorry

end algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93480


namespace required_range_of_a_l93_93278

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem required_range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (a-1)^x < (a-1)^y) ↔ a ∈ Set.Ioo 2 (⊤) :=
by
  sorry

end required_range_of_a_l93_93278


namespace centroid_triangle_l93_93830

noncomputable def centroid (X Y Z : ℝ^3) : ℝ^3 := (X + Y + Z) / 3

theorem centroid_triangle (X Y Z : ℝ^3) (G : ℝ^3) (hG : G = centroid X Y Z) 
  (h1 : dist G X ^ 2 + dist G Y ^ 2 + dist G Z ^ 2 = 90) :
  dist X Y ^ 2 + dist X Z ^ 2 + dist Y Z ^ 2 = 270 :=
  sorry

end centroid_triangle_l93_93830


namespace whole_numbers_between_cubicroots_l93_93706

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93706


namespace max_value_m_l93_93612

theorem max_value_m (m : ℝ) (h_m1 : m > 1) :
  (∃ x ∈ set.Icc (-2 : ℝ) 0, x^2 + 2 * m * x + (m^2 - m) ≤ 0) → m ≤ 4 :=
by
  intro h
  sorry

example : ∃ m : ℝ, m > 1 ∧ (∀ x ∈ set.Icc (-2 : ℝ) 0, x^2 + 2 * m * x + (m^2 - m) ≤ 0) ∧ m = 4 :=
by
  use 4
  split
  {
    exact lt_add_one 3, -- 4 > 1
  },
  split
  {
    intros x hx,
    sorry,
  },
  refl

end max_value_m_l93_93612


namespace smallest_composite_no_prime_factors_less_than_20_l93_93136

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93136


namespace total_games_won_l93_93429

noncomputable def games_won_by_team (games_played : ℕ) (win_percentage : ℝ) : ℕ :=
  (win_percentage * games_played).round.to_nat

theorem total_games_won :
  let team_a_games := 150
  let team_a_win_percentage := 0.35
  let team_b_games := 110
  let team_b_win_percentage := 0.45
  let team_c_games := 200
  let team_c_win_percentage := 0.30
  let team_a_wins := games_won_by_team team_a_games team_a_win_percentage
  let team_b_wins := games_won_by_team team_b_games team_b_win_percentage
  let team_c_wins := games_won_by_team team_c_games team_c_win_percentage
  in team_a_wins + team_b_wins + team_c_wins = 163 :=
by {
  let team_a_wins : ℕ := games_won_by_team 150 0.35
  let team_b_wins : ℕ := games_won_by_team 110 0.45
  let team_c_wins : ℕ := games_won_by_team 200 0.3
  have h1 : team_a_wins = 53 := by rfl
  have h2 : team_b_wins = 50 := by rfl
  have h3 : team_c_wins = 60 := by rfl
  calc
  team_a_wins + team_b_wins + team_c_wins 
    = 53 + 50 + 60 : by rw [h1, h2, h3]
    ... = 163 : by norm_num
}

end total_games_won_l93_93429


namespace employees_in_january_l93_93937

theorem employees_in_january (E : ℝ) (h : 500 = 1.15 * E) : E = 500 / 1.15 :=
by
  sorry

end employees_in_january_l93_93937


namespace negation_of_universal_statement_l93_93406

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 ≠ x) ↔ ∃ x : ℝ, x^2 = x :=
by sorry

end negation_of_universal_statement_l93_93406


namespace max_missed_problems_l93_93540

/--
To pass a geometry test at Scholars' High, a student must score at least 85%. If the test consists of 50 problems, prove that the greatest number of problems the student can miss and still pass is 7.
-/
theorem max_missed_problems (total_problems : ℕ) (pass_percentage : ℚ) (max_missed : ℕ) :
  total_problems = 50 → pass_percentage = 0.85 → max_missed = 7 → 
  (total_problems - max_missed).to_rat / total_problems.to_rat ≥ pass_percentage :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end max_missed_problems_l93_93540


namespace steak_weight_in_ounces_l93_93356

-- Definitions from conditions
def pounds : ℕ := 15
def ounces_per_pound : ℕ := 16
def steaks : ℕ := 20

-- The theorem to prove
theorem steak_weight_in_ounces : 
  (pounds * ounces_per_pound) / steaks = 12 := by
  sorry

end steak_weight_in_ounces_l93_93356


namespace cannot_determine_red_marbles_l93_93815

variable (Jason_blue : ℕ) (Tom_blue : ℕ) (Total_blue : ℕ)

-- Conditions
axiom Jason_has_44_blue : Jason_blue = 44
axiom Tom_has_24_blue : Tom_blue = 24
axiom Together_have_68_blue : Total_blue = 68

theorem cannot_determine_red_marbles (Jason_blue Tom_blue Total_blue : ℕ) : ¬ ∃ (Jason_red : ℕ), True := by
  sorry

end cannot_determine_red_marbles_l93_93815


namespace least_number_to_add_l93_93917

theorem least_number_to_add (n : ℕ) : 
  ∃ k : ℕ, n = 5432 + k ∧ k % 60 = 28 :=
by {
  use 28,
  split,
  { refl },
  { exact nat.mod_eq_of_lt (by norm_num) },
}

end least_number_to_add_l93_93917


namespace magnitude_of_z_l93_93768

noncomputable def i := Complex.I

theorem magnitude_of_z :
  let z := i + 2 * (i^2) + 3 * (i^3)
  in Complex.abs z = 2 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l93_93768


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93082

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93082


namespace inequality_has_no_solutions_l93_93175

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end inequality_has_no_solutions_l93_93175


namespace find_m_l93_93665

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m
  (m : ℝ)
  (a : ℝ × ℝ := (-2, 1))
  (b : ℝ × ℝ := (m, 2))
  (h : vector_magnitude (a.1 + b.1, a.2 + b.2) = vector_magnitude (a.1 - b.1, a.2 - b.2)) :
  m = 1 :=
sorry

end find_m_l93_93665


namespace number_of_whole_numbers_between_cubicroots_l93_93740

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93740


namespace find_element_in_N2O_mass_percentage_l93_93599

theorem find_element_in_N2O_mass_percentage :
  ∃ (element : String), element = "Oxygen" ∧ 
  ((∑ total_mass, total_mass = 44.02) →
   ((2 * 14.01 / 44.02 * 100) ≠ 36.36 ∧ 
    (16.00 / 44.02 * 100) = 36.36 )) :=
by
  -- Define necessary definitions     
  let mass_N : Float := 2 * 14.01
  let mass_O : Float := 16.00
  let total_mass : Float := mass_N + mass_O

  -- Formulating percentages
  let percentage_N := mass_N / total_mass * 100
  let percentage_O := mass_O / total_mass * 100

  -- Given mass percentage condition
  have given_mass_percentage : Float := 36.36

  -- Example with existential quantifier
  have element_Oxygen : String := "Oxygen"
  use element_Oxygen
  split 
  rfl 
  intro htotal_mass
  split
  show percentage_N ≠ given_mass_percentage, from sorry
  show percentage_O = given_mass_percentage, from sorry


end find_element_in_N2O_mass_percentage_l93_93599


namespace problem_AC_l93_93196

variables {a b c e : ℝ}
variables (F O B : Point)
variables (x y : ℝ)

def is_ellipse := (a > b ∧ b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def semi_focal_distance := c = sqrt (a^2 - b^2)
def eccentricity := e = c / a
def tan_BFO := tan (angle B F O)

theorem problem_AC :
  (is_ellipse ∧ semi_focal_distance ∧ eccentricity) →
  (2 * b > a + c → b^2 > a * c) ∧
  (tan_BFO > 1 → 0 < e ∧ e < sqrt 2 / 2) :=
sorry

end problem_AC_l93_93196


namespace line_through_focus_l93_93422

theorem line_through_focus (b : ℝ) (x y : ℝ) (h : 2 * x + b * y + 3 = 0) : 
  (x^2 + y^2 / 10 = 1) → (x = 0 ∧ (y = 3 ∨ y = -3)) → (b = 1 ∨ b = -1) :=
by
  intros
  cases ‹x = 0 ∧ (y = 3 ∨ y = -3)› with h1 h2
  cases h2 with h2a h2b
  sorry

end line_through_focus_l93_93422


namespace whole_numbers_between_l93_93730

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93730


namespace probability_product_multiple_of_three_l93_93825

theorem probability_product_multiple_of_three :
  (let octahedral_die := [1, 2, 3, 4, 5, 6, 7, 8];
       six_sided_die := [1, 2, 3, 4, 5, 6];
       multiples_of_three := [3, 6] in
     ((∑ i in octahedral_die, ∑ j in six_sided_die, if (i * j) % 3 = 0 then 1 else 0).toFloat) /
     (octahedral_die.length * six_sided_die.length).toFloat = 1 / 2) :=
sorry

end probability_product_multiple_of_three_l93_93825


namespace rides_inconsistency_l93_93553

noncomputable def Stephan_Slides (S : ℤ) : Prop :=
  2.16 * S + 2.16 * 3 = 15.54

noncomputable def Clarence_Rides : Prop :=
  2.16 * 3 + 2.16 * 3 = 17.70

theorem rides_inconsistency : ¬ (∃ S : ℤ, Stephan_Slides S ∧ Clarence_Rides) :=
by
  intro h
  rcases h with ⟨S, hS, hC⟩
  rw [Stephan_Slides, Clarence_Rides] at *
  sorry

end rides_inconsistency_l93_93553


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l93_93965

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l93_93965


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93083

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93083


namespace minimize_distances_l93_93798

structure Point where
  x : ℝ
  y : ℝ

def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def sum_of_distances (P : Point) (points : List Point) : ℝ :=
  points.map (dist P) |>.sum

noncomputable def optimal_point : Point :=
  ⟨2, 4⟩

theorem minimize_distances :
  ∀ (P : Point),
  sum_of_distances optimal_point [⟨1, 2⟩, ⟨1, 5⟩, ⟨3, 6⟩, ⟨7, -1⟩] ≤
  sum_of_distances P [⟨1, 2⟩, ⟨1, 5⟩, ⟨3, 6⟩, ⟨7, -1⟩] :=
sorry

end minimize_distances_l93_93798


namespace smallest_composite_no_prime_factors_less_than_20_l93_93092

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93092


namespace integer_exponentiation_l93_93263

theorem integer_exponentiation
  (a b x y : ℕ)
  (h_gcd : a.gcd b = 1)
  (h_pos_a : 1 < a)
  (h_pos_b : 1 < b)
  (h_pos_x : 1 < x)
  (h_pos_y : 1 < y)
  (h_eq : x^a = y^b) :
  ∃ n : ℕ, 1 < n ∧ x = n^b ∧ y = n^a :=
by sorry

end integer_exponentiation_l93_93263


namespace fresh_grapes_contain_85_percent_water_l93_93606

-- Definitions of the conditions
def percentage_of_water_in_raisins : ℝ := 25
def weight_of_fresh_grapes : ℝ := 100
def weight_of_raisins : ℝ := 20
def weight_of_solid_material_in_raisins : ℝ := 0.75 * weight_of_raisins
def percentage_of_solid_material_in_fresh_grapes (W : ℝ) : ℝ := 100 - W

-- The theorem statement
theorem fresh_grapes_contain_85_percent_water (W : ℝ) :
  weight_of_solid_material_in_raisins = weight_of_solid_material_in_fresh_grapes W / 100 * weight_of_fresh_grapes → 
  W = 85 :=
sorry

end fresh_grapes_contain_85_percent_water_l93_93606


namespace max_power_speed_l93_93400

noncomputable def force (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v)^2) / 2

def power (B S ρ v₀ v : ℝ) : ℝ :=
  force B S ρ v₀ v * v

theorem max_power_speed (B ρ : ℝ) (S : ℝ := 7) (v₀ : ℝ := 6.3) :
  ∃ v, v = 2.1 ∧ (∀ v', power B S ρ v₀ v' ≤ power B S ρ v₀ v) :=
begin
  sorry,
end

end max_power_speed_l93_93400


namespace chandra_pairings_l93_93552

theorem chandra_pairings : 
  let bowls := 5
  let glasses := 6
  (bowls * glasses) = 30 :=
by
  sorry

end chandra_pairings_l93_93552


namespace total_passengers_wearing_hats_l93_93501

/-- Given the total number of passengers, the percentage of women among the passengers, 
the percentage of women wearing hats, and the percentage of men wearing hats, 
prove that the total number of passengers wearing hats is 739. -/
theorem total_passengers_wearing_hats :
  ∃ (women men women_wearing_hats men_wearing_hats total_wearing_hats : ℕ),
  5200 = women + men ∧
  women = 2860 ∧
  men = 2340 ∧
  women_wearing_hats = 458 ∧
  men_wearing_hats = 281 ∧
  total_wearing_hats = women_wearing_hats + men_wearing_hats ∧
  total_wearing_hats = 739 :=
by {
  let women := 2860,
  let men := 2340,
  let women_wearing_hats := 458,
  let men_wearing_hats := 281,
  let total_wearing_hats := women_wearing_hats + men_wearing_hats,
  use [women, men, women_wearing_hats, men_wearing_hats, total_wearing_hats],
  split,
  { simp [women, men] },      -- 5200 = 2860 + 2340
  split,
  { refl },                  -- women = 2860
  split,
  { refl },                  -- men = 2340
  split,
  { refl },                  -- women_wearing_hats = 458
  split,
  { refl },                  -- men_wearing_hats = 281
  split,                     
  { simp [total_wearing_hats] }, -- total_wearing_hats = 458 + 281
  { refl }                   -- total_wearing_hats = 739
}

end total_passengers_wearing_hats_l93_93501


namespace adam_chocolate_boxes_l93_93530

theorem adam_chocolate_boxes 
  (c : ℕ) -- number of chocolate boxes Adam bought
  (h1 : 4 * c + 4 * 5 = 28) : 
  c = 2 := 
by
  sorry

end adam_chocolate_boxes_l93_93530


namespace count_whole_numbers_between_cubes_l93_93679

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93679


namespace smallest_composite_no_prime_factors_less_than_20_l93_93096

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93096


namespace continuous_function_is_constant_l93_93328

open Real

-- Given function and condition
def hypothesis (f : ℝ → ℝ) := ∀ a b : ℝ, f ((a + b) / 2) ∈ {f a, f b}

-- Proof statement: if f is continuous and satisfies the hypothesis, then f is constant
theorem continuous_function_is_constant (f : ℝ → ℝ) (hf : Continuous f) (h : hypothesis f) : ∃ c : ℝ, ∀ x : ℝ, f x = c := 
sorry

end continuous_function_is_constant_l93_93328


namespace total_study_time_correct_l93_93359

def study_time_wednesday : ℕ := 2
def study_time_thursday : ℕ := 3 * study_time_wednesday
def study_time_friday : ℕ := study_time_thursday / 2
def study_time_weekend : ℕ := study_time_wednesday + study_time_thursday + study_time_friday

def total_study_time : ℕ := study_time_wednesday + study_time_thursday + study_time_friday + study_time_weekend

theorem total_study_time_correct : total_study_time = 22 := by
  have : study_time_wednesday = 2 := by rfl
  have : study_time_thursday = 3 * study_time_wednesday := by rfl
  have : study_time_friday = study_time_thursday / 2 := by rfl
  have : study_time_weekend = study_time_wednesday + study_time_thursday + study_time_friday := by rfl
  have total_study_time_eq : total_study_time = study_time_wednesday + study_time_thursday + study_time_friday + study_time_weekend := by rfl
  calc total_study_time
      = 2 + (3 * 2) + ((3 * 2) / 2) + (2 + (3 * 2) + ((3 * 2) / 2)) : by rw [total_study_time_eq, this, this, this, this]
  ... = 22 : by simp

end total_study_time_correct_l93_93359


namespace total_people_correct_current_people_correct_l93_93806

-- Define the conditions as constants
def morning_people : ℕ := 473
def noon_left : ℕ := 179
def afternoon_people : ℕ := 268

-- Define the total number of people
def total_people : ℕ := morning_people + afternoon_people

-- Define the current number of people in the amusement park
def current_people : ℕ := morning_people - noon_left + afternoon_people

-- Theorem proofs
theorem total_people_correct : total_people = 741 := by sorry
theorem current_people_correct : current_people = 562 := by sorry

end total_people_correct_current_people_correct_l93_93806


namespace chord_length_of_ray_in_circle_l93_93807

open Real

def polar_eq_circle (θ : ℝ) : ℝ := 4 * sin θ

theorem chord_length_of_ray_in_circle :
  (polar_eq_circle (π / 4)) = 2 * sqrt 2 := sorry

end chord_length_of_ray_in_circle_l93_93807


namespace whole_numbers_between_cubicroots_l93_93710

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93710


namespace count_3cm_sticks_l93_93433

theorem count_3cm_sticks (length_stick : ℕ) (l1 l2 : ℕ) : 
  (length_stick = 240) → (l1 = 7) → (l2 = 6) →
  (number_of_3cm_sticks : ℕ) := 12
:= sorry

end count_3cm_sticks_l93_93433


namespace smallest_composite_no_prime_factors_less_than_20_l93_93013

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93013


namespace smallest_composite_no_prime_factors_below_20_l93_93129

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93129


namespace whole_numbers_between_l93_93727

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93727


namespace part1_part2_l93_93236

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1 (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, f x a ≥ 4 ↔ x ≤ (3 / 2 : ℝ) ∨ x ≥ (11 / 2 : ℝ) :=
by 
  rw h
  sorry

theorem part2 (h : ∀ x a : ℝ, f x a ≥ 4) :
  ∀ a : ℝ, (a - 1)^2 ≥ 4 ↔ a ≤ -1 ∨ a ≥ 3 :=
by 
  sorry

end part1_part2_l93_93236


namespace cylinder_sphere_surface_area_ratio_l93_93634

def radius (d : ℝ) := d / 2
def surfaceAreaCylinder (r : ℝ) (h : ℝ) := 2 * Real.pi * r * (r + h)
def surfaceAreaSphere (r : ℝ) := 4 * Real.pi * r^2

theorem cylinder_sphere_surface_area_ratio (d : ℝ) (h : ℝ) (r : ℝ = radius d) (h = d) :
  surfaceAreaCylinder r h / surfaceAreaSphere r = 3 / 4 :=
by
  sorry

end cylinder_sphere_surface_area_ratio_l93_93634


namespace geometric_sequence_q_count_l93_93805

theorem geometric_sequence_q_count 
  (a : ℕ → ℝ) 
  (a2a8 : a 2 * a 8 = 36) 
  (a3a7 : a 3 + a 7 = 15) : 
  ∃ q : ℝ, (q = sqrt (13 / 2) ∨ q = -sqrt (13 / 2) ∨ q = sqrt (2 / 13) ∨ q = -sqrt (2 / 13)) := 
by
  sorry

end geometric_sequence_q_count_l93_93805


namespace license_plates_count_l93_93511

theorem license_plates_count : 
  let letters := 26
  let digits := 10
  (letters ^ 3) * (digits ^ 3) = 17576000 :=
by
  let letters := 26
  let digits := 10
  have h1 : letters ^ 3 = 17576 := by
    calc
      26 ^ 3 = 26 * 26 * 26 : by rw pow_succ
            ... = 676 * 26 : by norm_num
            ... = 17576 : by norm_num
  have h2 : digits ^ 3 = 1000 := by
    calc
      10 ^ 3 = 10 * 10 * 10 : by rw pow_succ
            ... = 100 * 10 : by norm_num
            ... = 1000 : by norm_num
  show (letters ^ 3) * (digits ^ 3) = 17576000 from
    calc
      (letters ^ 3) * (digits ^ 3) = 17576 * 1000 : by rw [h1, h2]
                                  ... = 17576000 : by norm_num

end license_plates_count_l93_93511


namespace eval_expression_log_expression_l93_93499

-- Part I: proving the evaluation of an expression
theorem eval_expression :
  3 * (-4)^3 - (1/2)^0 + 0.25^(1/2) * (1 / (sqrt 2))^(-4) = -23 :=
by
  -- Sorry is placed here to denote the absence of proof 
  sorry

-- Part II: proving the logarithmic statement
theorem log_expression (a b : Real) (h1 : 5^a = 3) (h2 : 5^b = 4) :
  Real.logBase 25 12 = (a + b) / 2 :=
by
  -- Sorry is placed here to denote the absence of proof 
  sorry

end eval_expression_log_expression_l93_93499


namespace ab_value_l93_93254

theorem ab_value (a b : ℚ) 
  (h1 : (a + b) ^ 2 + |b + 5| = b + 5) 
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1 / 9 :=
by
  sorry

end ab_value_l93_93254


namespace radius_of_circumscribed_circle_l93_93190

variables {R a b : ℝ}

theorem radius_of_circumscribed_circle (h : ∀ {A B C D : ℝ}, (AC = a) ∧ (BD = b) ∧ (AB ⊥ CD)) :
  R = (a^2 + b^2)^0.5 / 2 :=
sorry

end radius_of_circumscribed_circle_l93_93190


namespace smallest_composite_no_prime_lt_20_l93_93051

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93051


namespace bounded_region_area_correct_l93_93576

noncomputable def bounded_region_area : ℝ := 900

theorem bounded_region_area_correct (x y : ℝ) :
  y^2 + 4 * x * y + 60 * |x| = 600 → 
  ∃ x y : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 
  (y = 15 ∨ y = -15 ∨ y = 30 - 4 * x ∨ y = -15 - 4 * x) ∧ 
  bounded_region_area = 900 :=
begin
  sorry
end

end bounded_region_area_correct_l93_93576


namespace zeta_frac_part_sum_eq_one_fourth_l93_93165

/-- The Riemann zeta function is defined for \(x > 1\) by \(\zeta(x) = \sum_{n = 1}^\infty \frac{1}{n^x}\). -/
def zeta (x : ℝ) : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n ^ x) else 0

/-- The fractional part of a real number \(x\) is defined as \( \{x\} = x - \lfloor x \rfloor \). -/
def frac_part (x : ℝ) : ℝ := x - ⌊x⌋

/-- Prove that the sum of the fractional parts of \(\zeta(2k - 1)\) for \(k \ge 2\) equals \(\frac{1}{4}\). -/
theorem zeta_frac_part_sum_eq_one_fourth : 
  ∑' k in (Set.Ico 2 (⊤ : ℕ)), frac_part (zeta (2 * k - 1)) = 1 / 4 := 
sorry

end zeta_frac_part_sum_eq_one_fourth_l93_93165


namespace construct_point_D_l93_93851

-- Define the circle and points A, B, C, D
variable {k : Type} [metric_space k] [normed_group k]
variables (A B C D : k)
variable (circle : set k) -- Circle k

-- Define that points A, B, C are on the circle
variables (hA : A ∈ circle) (hB : B ∈ circle) (hC : C ∈ circle)

-- Definition of a cyclic quadrilateral
def cyclic_quadrilateral (A B C D : k) (circle : set k) : Prop :=
  A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle

-- Definition of a tangential quadrilateral
def tangential_quadrilateral (A B C D : k) : Prop :=
  dist A B + dist C D = dist A D + dist B C

-- The proof statement
theorem construct_point_D (A B C : k) (circle : set k)
  (hA : A ∈ circle) (hB : B ∈ circle) (hC : C ∈ circle) :
  ∃ D : k, D ∈ circle ∧ cyclic_quadrilateral A B C D circle ∧ tangential_quadrilateral A B C D :=
begin
  -- Proof to be filled in by user
  sorry
end

end construct_point_D_l93_93851


namespace smallest_composite_no_prime_factors_less_than_20_l93_93100

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93100


namespace count_whole_numbers_between_cubes_l93_93677

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93677


namespace parallel_vectors_l93_93642

variables (a : ℝ)

def m : ℝ × ℝ := (a, -2)
def n : ℝ × ℝ := (1, 1 - a)

theorem parallel_vectors : 
  (m a).cross_product (n a) = 0 → a = 2 ∨ a = -1 :=
by sorry

end parallel_vectors_l93_93642


namespace whole_numbers_between_cuberoots_l93_93689

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93689


namespace exact_probability_five_shots_l93_93962

theorem exact_probability_five_shots (p : ℝ) (h1 : 0 < p) (h2 : p ≤ 1) :
  (let hit := p
       miss := 1 - p
       comb := 6 in
   comb * hit^3 * miss^2 = 6 * p^3 * (1 - p)^2) :=
by sorry

end exact_probability_five_shots_l93_93962


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93078

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93078


namespace closest_integer_to_sqrt_35_is_6_l93_93367

theorem closest_integer_to_sqrt_35_is_6 :
  (∀ x : ℝ, 5 < x → x < 6 → (ceil x = 6)) :=
begin
  intro x,
  intros h5 h6,
  suffices : x ≤ 5.5,
  { exact real.ceil_eq_of_le this },
  linarith,
end

end closest_integer_to_sqrt_35_is_6_l93_93367


namespace smallest_composite_no_prime_factors_less_20_l93_93038

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93038


namespace find_f_1_100_l93_93827

def f : ℕ × ℕ → ℝ := sorry

theorem find_f_1_100 (f : ℕ × ℕ → ℝ) (h1 : ∀ i : ℕ, f (i, i+1) = 1 / 3)
  (h2 : ∀ i j k : ℕ, i < k ∧ k < j → f (i, j) = f (i, k) + f (k, j) - 2 * f (i, k) * f (k, j)) :
  f (1, 100) = 1 / 2 * (1 - 1 / 3 ^ 100) :=
begin
  sorry,
end

end find_f_1_100_l93_93827


namespace trig_identity_l93_93585

theorem trig_identity : (Real.sin (15 * Real.pi / 180))^4 - (Real.cos (15 * Real.pi / 180))^4 = -real.sqrt 3 / 2 := by
  sorry

end trig_identity_l93_93585


namespace dormitory_total_expenditure_now_l93_93295

-- Define initial conditions
def initial_students : ℕ := 250
def additional_students : ℕ := 75
def decrease_in_cost : ℕ := 20
def increase_in_expenditure : ℕ := 10000

-- Define variables
noncomputable def original_avg_cost : ℕ := 220
noncomputable def new_students : ℕ := initial_students + additional_students
noncomputable def new_avg_cost : ℕ := original_avg_cost - decrease_in_cost
noncomputable def initial_expenditure : ℕ := initial_students * original_avg_cost
noncomputable def new_expenditure : ℕ := new_students * new_avg_cost
noncomputable def total_expenditure_now : ℕ := 65000

-- State the theorem that needs to be proved
theorem dormitory_total_expenditure_now :
  new_expenditure = total_expenditure_now :=
by {
  -- define the intermediate steps in the proof context
  have h1 : original_avg_cost = (initial_expenditure + increase_in_expenditure) / initial_students,
    sorry,
  have h2 : new_avg_cost = original_avg_cost - decrease_in_cost,
    sorry,
  have h3 : new_expenditure = new_students * new_avg_cost,
    sorry,
  have h4 : new_expenditure = 65000,
    sorry,
  exact h4,
}

end dormitory_total_expenditure_now_l93_93295


namespace find_N_plus_5n_l93_93376

theorem find_N_plus_5n (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  let A := x + y + z,
      B := x^2 + y^2 + z^2,
      C := xy + xz + yz in
  ∃ N n : ℝ, N = 27 ∧ n = 0 ∧ (N + 5 * n = 27) :=
by
  let A := x + y + z 
  let B := x^2 + y^2 + z^2 
  let C := x * y + x * z + y * z 
  use 27, 0
  sorry

end find_N_plus_5n_l93_93376


namespace smallest_composite_no_prime_factors_below_20_l93_93121

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93121


namespace prob_5_shots_expected_number_shots_l93_93955

variable (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

def prob_exactly_five_shots : ℝ := 6 * p^3 * (1 - p)^2
def expected_shots : ℝ := 3 / p

theorem prob_5_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the probability of exactly 5 shots needed is as calculated
  prob_exactly_five_shots p = 6 * p^3 * (1 - p)^2 :=
by
  sorry

theorem expected_number_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the expected number of shots to hit all targets is as calculated
  expected_shots p = 3 / p :=
by
  sorry

end prob_5_shots_expected_number_shots_l93_93955


namespace distance_between_intersections_l93_93794

noncomputable def line_parametric_eqn (t : ℝ) : ℝ × ℝ :=
  (2 + t, t + 1)

noncomputable def curve_polar_eqn (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ + 3 = 0

noncomputable def line_general_eqn (x y : ℝ) : Prop :=
  x - y - 1 = 0

noncomputable def curve_rect_eqn (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 3 = 0

theorem distance_between_intersections : 
  ∃ A B : ℝ × ℝ, 
  line_parametric_eqn (some t) = A ∧ 
  line_parametric_eqn (some s) = B ∧ 
  curve_rect_eqn A.1 A.2 ∧ 
  curve_rect_eqn B.1 B.2 ∧ 
  dist A B = Real.sqrt 2 :=
sorry

end distance_between_intersections_l93_93794


namespace geometric_S_n_over_n_sum_of_S_n_l93_93657

def sequence_a (n : ℕ) : ℕ → ℝ
| 0     := 2
| (n+1) := (n + 2) / n * S n

def S (n : ℕ) : ℝ :=
  ∑ i in range n, sequence_a i

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ (a r : ℝ), (∀ n, s (n + 1) = r * s n) ∧ s 0 = a

theorem geometric_S_n_over_n (n : ℕ) :
  is_geometric_sequence (λ n, S n / n) :=
sorry

def T (n : ℕ) : ℝ :=
  ∑ i in range n, S i

theorem sum_of_S_n (n : ℕ) :
  T n = (n - 1) * 2^(n+1) + 2 :=
sorry

end geometric_S_n_over_n_sum_of_S_n_l93_93657


namespace no_func_exists_l93_93595

noncomputable def f : ℝ → ℝ := sorry

def exists_no_such_function (f : ℝ → ℝ) :=
  ∀ f, (∀ x : ℝ, x ≠ 0 → f(-x) = -f(x)) ∧ 
       (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → 
          f(1 / (x + y)) = f(1 / x) + f(1 / y) + 2 * (x * y - 1000)) →
        false

theorem no_func_exists : ∀ f : ℝ → ℝ, exists_no_such_function f :=
by
  intro f
  unfold exists_no_such_function
  intro h
  sorry

end no_func_exists_l93_93595


namespace length_of_side_c_in_triangle_l93_93782

theorem length_of_side_c_in_triangle :
  ∀ (a b : ℝ) (angleC : ℝ), a = 3 → b = 5 → angleC = 120 → 
  let c := real.sqrt (a^2 + b^2 - 2 * a * b * real.cos (angleC * real.pi / 180)) in
  c = 7 :=
by
  intros a b angleC ha hb hC
  simp [ha, hb, hC, real.cos]

sorry

end length_of_side_c_in_triangle_l93_93782


namespace bear_meat_needs_l93_93981

theorem bear_meat_needs (B_total : ℕ) (cubs : ℕ) (w_cub : ℚ) 
  (h1 : B_total = 210)
  (h2 : cubs = 4)
  (h3 : w_cub = B_total / cubs) : 
  w_cub = 52.5 :=
by 
  sorry

end bear_meat_needs_l93_93981


namespace smallest_composite_no_prime_factors_less_than_20_l93_93024

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93024


namespace two_intersecting_lines_determine_plane_l93_93928

/-- Given two intersecting lines, prove that they determine a plane. -/
theorem two_intersecting_lines_determine_plane (L1 L2 : set ℝ) (P : ℝ)
  (hL1 : ∃ P1 P2 : ℝ, P1 ≠ P2 ∧ P1 ∈ L1 ∧ P2 ∈ L1)
  (hL2 : ∃ P3 P4 : ℝ, P3 ≠ P4 ∧ P3 ∈ L2 ∧ P4 ∈ L2)
  (h_intersect : ∃ P, P ∈ L1 ∧ P ∈ L2) :
  ∃ plane : set ℝ, P ∈ plane ∧ ∀ P' ∈ L1, P' ∈ plane ∧ ∀ P'' ∈ L2, P'' ∈ plane :=
sorry

end two_intersecting_lines_determine_plane_l93_93928


namespace center_of_mass_exists_l93_93591

theorem center_of_mass_exists (A B C O : Point) (p q : ℕ) (hp : 0 < p) (hq : 0 < q) :
  (∀ K L : Point, K ∈ Line_through O A ∧ L ∈ Line_through O C → 
  K ∈ Segment A B ∧ L ∈ Segment B C → 
  p * (K.dist A / K.dist B) + q * (L.dist C / L.dist B) = 1) →
  is_center_of_mass O [⟨A, p⟩, ⟨B, 1⟩, ⟨C, q⟩] := by
  sorry

end center_of_mass_exists_l93_93591


namespace park_area_correct_l93_93413

noncomputable def rect_park_area (speed_km_hr : ℕ) (time_min : ℕ) (ratio_l_b : ℕ) : ℕ := by
  let speed_m_min := speed_km_hr * 1000 / 60
  let perimeter := speed_m_min * time_min
  let B := perimeter * 3 / 8
  let L := B / 3
  let area := L * B
  exact area

theorem park_area_correct : rect_park_area 12 8 3 = 120000 := by
  sorry

end park_area_correct_l93_93413


namespace probability_shorts_not_equal_jersey_l93_93872

-- Definitions based on conditions
def color := {black, gold, white}

def shorts_color : color := _
def jersey_color : color := _

-- Assuming independent and equally likely choices for shorts and jerseys
def total_combinations := 3 * 3
def different_color_combinations := 2 + 2 + 2

-- Lean statement to prove the probability
theorem probability_shorts_not_equal_jersey : 
  (different_color_combinations : ℚ) / total_combinations = 2 / 3 := 
by sorry

end probability_shorts_not_equal_jersey_l93_93872


namespace max_norm_OC_l93_93848

-- Definitions of vectors and their magnitudes
variables {V : Type*} [inner_product_space ℝ V]
variables (OA OB OC : V) (a b : ℝ) (λ μ : ℝ)

-- Given conditions
def condition1 : Prop := ∥OA∥ = a ∧ ∥OB∥ = b
def condition2 : Prop := a^2 + b^2 = 4
def condition3 : Prop := ⟪OA, OB⟫ = 0
def condition4 : OC = λ • OA + μ • OB
def condition5 : (λ - 1/2)^2 * a^2 + (μ - 1/2)^2 * b^2 = 1

-- Proof problem: Prove the maximum value of ∥OC∥ is 2.
theorem max_norm_OC (h1 : condition1 OA OB a b)
                     (h2 : condition2 a b)
                     (h3 : condition3 OA OB)
                     (h4 : condition4 OA OB OC λ μ)
                     (h5 : condition5 a b λ μ) :
  ∥OC∥ ≤ 2 := 
sorry

end max_norm_OC_l93_93848


namespace calculate_expression_l93_93500

theorem calculate_expression :
  107 * 107 + 93 * 93 = 20098 := by
  sorry

end calculate_expression_l93_93500


namespace john_total_spend_l93_93821

variable (tshirt_count : ℕ) (tshirt_cost : ℕ) (pants_cost : ℕ)

def total_cost (tshirt_count tshirt_cost pants_cost : ℕ) : ℕ := 
  (tshirt_count * tshirt_cost) + pants_cost

theorem john_total_spend
  (h1 : tshirt_count = 3)
  (h2 : tshirt_cost = 20)
  (h3 : pants_cost = 50) :
  total_cost tshirt_count tshirt_cost pants_cost = 110 := 
by
  simp [total_cost, h1, h2, h3]
  sorry

end john_total_spend_l93_93821


namespace short_trees_after_planting_l93_93426

-- Defining the conditions as Lean definitions
def current_short_trees : Nat := 3
def newly_planted_short_trees : Nat := 9

-- Defining the question (assertion to prove) with the expected answer
theorem short_trees_after_planting : current_short_trees + newly_planted_short_trees = 12 := by
  sorry

end short_trees_after_planting_l93_93426


namespace magnitude_of_z_l93_93837

theorem magnitude_of_z (i : ℂ) (z : ℂ) (h : z * (i + 1) = i) (h_i : i = complex.I) : |z| = real.sqrt 2 / 2 :=
by
  have h : z * (complex.I + 1) = complex.I := by rw [h_i]; exact h
  have : z = (complex.I * (1 - complex.I)) / ((complex.I + 1) * (1 - complex.I)) :=
    calc
      z = (complex.I / (complex.I + 1)) * ((1 - complex.I) / (1 - complex.I)) : sorry
      ... = ((complex.I * (1 - complex.I)) / ((complex.I + 1) * (1 - complex.I))) : sorry
  have num : (complex.I * (1 - complex.I)) = 1 + complex.I := by
    calc
      complex.I * (1 - complex.I) = complex.I - complex.I * complex.I : by ring
      ... = complex.I - (-1) : by rw [complex.I_mul_I]
      ... = complex.I + 1 : by ring
  have denom : (complex.I + 1) * (1 - complex.I) = 2 := by
    calc
      (complex.I + 1) * (1 - complex.I) = (complex.I + 1) * (1 - complex.I) : by ring
      ... = -complex.I * complex.I + complex.I + 1 - complex.I : by ring
      ... = 1 + 1 : by rw [complex.I_mul_I]; ring
      ... = 2 : by ring
  have mag_z : |z| = real.sqrt (((1 / 2) ^ 2) + ((1 / 2) ^ 2)) :=
    calc
      |z| = complex.abs (((1 + complex.I) / 2)) : by simpa [num, denom] using this
      ... = real.sqrt ((1 / 2) * (1 / 2) + (1 / 2) * (1 / 2)) : by 
        rw [complex.abs_div, complex.abs_of_nonneg (show (2 : ℝ) ≥ 0, by norm_num)]
        rw [complex.abs_of_nonneg (show 1 + complex.I ≥ 0, by norm_num)]
        rw [complex.abs_of_nonneg (show 1 ≥ 0, by norm_num)]
      ... = real.sqrt (2 / 4) : by ring
      ... = real.sqrt 2 / 2 : by rw [sqrt_div, sqrt_2_eq_sqrt_2_I]
  exact mag_z

end magnitude_of_z_l93_93837


namespace interest_rate_D_interest_rate_E_l93_93519

-- Define the principal amounts lent to B, C, D, and E
def principalB : ℝ := 5000
def principalC : ℝ := 3000
def principalD : ℝ := 7000
def principalE : ℝ := 4500

-- Define the times for which the amounts were lent
def timeB : ℝ := 2
def timeC : ℝ := 4
def timeD : ℝ := 3
def timeE : ℝ := 5

-- Define the interests received from B and C
def interestBC : ℝ := 1980

-- Define the interests received from D and E
def interestD : ℝ := 2940
def interestE : ℝ := 3375

-- Calculate the rate of interest for B and C, which is the same
def rateBC : ℝ := (interestBC) / (principalB * timeB + principalC * timeC) * 100

-- Define the rate of interest for D and E
def rateD : ℝ := interestD / (principalD * timeD) * 100
def rateE : ℝ := interestE / (principalE * timeE) * 100

-- The goal is to prove that the rate of interest
-- for the loan to D is 14% per annum, and for E is 15% per annum
theorem interest_rate_D : rateD = 14 := sorry
theorem interest_rate_E : rateE = 15 := sorry

end interest_rate_D_interest_rate_E_l93_93519


namespace intersection_area_l93_93307

-- Define Regions M and N
def regionM (x y : ℝ) : Prop := y ≥ 0 ∧ y ≤ x ∧ y ≤ 2 - x

def regionN (t x y : ℝ) : Prop := t ≤ x ∧ x ≤ t + 1

-- Define the function f(t)
def f (t : ℝ) : ℝ := -t^2 + t + 1/2

-- Main theorem statement
theorem intersection_area (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) :
    ∀ (x y : ℝ), (regionM x y) ∧ (regionN t x y) → y = f t :=
sorry

end intersection_area_l93_93307


namespace min_value_of_a_plus_b_minus_c_l93_93202

open Real

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ) :
  (∀ (x y : ℝ), 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) →
  (∃ c_min, c_min = 2 ∧ ∀ c', c' = a + b - c → c' ≥ c_min) :=
by
  sorry

end min_value_of_a_plus_b_minus_c_l93_93202


namespace number_of_whole_numbers_between_cubicroots_l93_93735

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93735


namespace proof_problem_l93_93182

noncomputable def problem (θ : ℝ) : ℝ := 
  let k := 1 / 2
  let tan_theta := 3
  let cos_theta := real.sqrt (1 / 10)
  let sin_theta := 3 * cos_theta
  (1 - k * cos_theta) / sin_theta - (2 * sin_theta) / (1 + cos_theta)

theorem proof_problem : 
  problem θ = (20 - real.sqrt 10) / (3 * real.sqrt 10) - (6 * real.sqrt 10) / (10 + real.sqrt 10) :=
sorry

end proof_problem_l93_93182


namespace count_whole_numbers_between_cubes_l93_93673

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93673


namespace balls_into_boxes_l93_93370

-- Definitions based on the conditions provided
def num_balls : ℕ := 1996
def num_boxes : ℕ := 10

-- Main theorem statement to prove
theorem balls_into_boxes : 
  (∑ i in range(10), i + 1) = 55 ∧
  (binom (num_balls - 55 + num_boxes - 1) (num_boxes - 1)) = binom 1950 9 := 
by 
  sorry

end balls_into_boxes_l93_93370


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l93_93968

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l93_93968


namespace sum_and_product_of_roots_l93_93580

theorem sum_and_product_of_roots :
  (∀ x, x ∈ set.Icc 0 (2 * Real.pi) → (Real.tan x) ^ 2 - 14 * Real.tan x + 49 = 0) →
  (∃ x1 x2, x1 = Real.arctan 7 ∧ x2 = Real.arctan 7 + Real.pi ∧ 
            x1 + x2 = 2 * Real.arctan 7 + Real.pi ∧ 
            Real.tan (x1 + x2) = -Real.tan (2 * Real.arctan 7) ∧ 
            Real.tan (x1 + x2) = 7 / 24) := sorry

end sum_and_product_of_roots_l93_93580


namespace whole_numbers_between_cubicroots_l93_93709

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93709


namespace simplify_polynomial_l93_93205

open Nat

-- Define arithmetic sequence conditions and the polynomial structure
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def binomial (n k : ℕ) : ℕ := choose n k

noncomputable def p (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in range (n + 1), (a k) * (binomial n k) * (x^k) * ((1 - x)^(n - k))

-- The main theorem
theorem simplify_polynomial
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : arithmetic_seq a d)
  (n : ℕ)
  (x : ℝ)
  : p a n x = a 0 + n * d * x :=
sorry

end simplify_polynomial_l93_93205


namespace Steinbart_concurrency_l93_93791

noncomputable def concurrent_lines {A B C D E F D' E' F' P : Type*}
  [Incircle : ∀ {T : Type*}, tangent_point (T) (A) (B) (C) (D) (E) (F)]
  [Inside : ∀ {T : Type*}, in_incircle (T) (P)]
  [Intersections : ∀ {T : Type*}, secant_lines (T) (P) (D') (E') (F')] :
  Prop := concurrent (AD' : Line) (BE' : Line) (CF' : Line)

-- Definition of the tangent_point
class tangent_point (T : Type*) (A B C D E F : T)

-- Definition of the in_incircle
class in_incircle (T : Type*) (P : T)

-- Definition of the secant_lines
class secant_lines (T : Type*) (P : T) (D' E' F' : T)

-- Definition of concurrent lines
class concurrent (AD' BE' CF' : Type*)

theorem Steinbart_concurrency {T : Type*} 
  (A B C D E F D' E' F' P : T)
  [H1 : tangent_point T A B C D E F]
  [H2 : in_incircle T P]
  [H3 : secant_lines T P D' E' F']
  : concurrent_lines :=
sorry

end Steinbart_concurrency_l93_93791


namespace smallest_composite_no_prime_factors_lt_20_l93_93149

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93149


namespace solution_depends_on_parity_l93_93354

-- Conditions setting
variables {a b n : ℤ}
variables {a_0 a_1 a_2 : ℤ}
variables {Q P : ℕ → ℤ}

-- The nth convergent property
def convergents_property (k : ℕ) : Prop := a * Q k - b * P k = (-1)^k

-- The continued fraction representation
def continued_fraction : Prop :=
  a / b = [a_0; a_1, a_2, ... ∧ convergents_property (n - 1)]

-- Lean statement
theorem solution_depends_on_parity 
  (cf : continued_fraction) : 
  (a * Q (n-1) - b * P (n-1) = 1 ∧ (∃ x y, (x, y) = (Q (n-1), P (n-1)))) ∨ 
  (a * Q (n-1) - b * P (n-1) = -1 ∧ (∃ x y, (x, y) = (-Q (n-1), -P (n-1)))) :=
by sorry

end solution_depends_on_parity_l93_93354


namespace smallest_composite_no_prime_factors_below_20_l93_93127

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93127


namespace area_is_63_l93_93939

structure Point where
  x : ℝ
  y : ℝ

def d1 (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x).abs

def d2 (p1 p2 : Point) : ℝ :=
  (p1.y - p2.y).abs

def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

def rhombus_vertices := [
  Point.mk 0 3.5,
  Point.mk 9 0,
  Point.mk 0 (-3.5),
  Point.mk (-9) 0
]

theorem area_is_63 : area_of_rhombus (d1 (rhombus_vertices[1]) (rhombus_vertices[3])) (d2 (rhombus_vertices[0]) (rhombus_vertices[2])) = 63 := by
  sorry

end area_is_63_l93_93939


namespace find_c_find_angle_l93_93662

open Real

variables (a b c : ℝ × ℝ)
variables (θ : ℝ)

def vec_a := ((2 : ℝ), (1 : ℝ))

-- Part 1: Coordinates of vector c
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def magnitude (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 * u.1 + u.2 * u.2)

def condition1 : Prop :=
  orthogonal c vec_a ∧ magnitude c = 2 * real.sqrt 5

def vec_c1 := ((-2 : ℝ), (4 : ℝ))
def vec_c2 := ((2 : ℝ), (-4 : ℝ))

theorem find_c (h : condition1) : c = vec_c1 ∨ c = vec_c2 :=
sorry

-- Part 2: Angle between vector a and b
def vec_b_magnitude (u : ℝ × ℝ) : Prop :=
  magnitude u = real.sqrt 5 / 2

def vec_perpendicular (u v : ℝ × ℝ) : Prop :=
  orthogonal ((vec_a.1 + 2 * u.1, vec_a.2 + 2 * u.2)) ((2 * vec_a.1 - u.1, 2 * vec_a.2 - u.2))

def condition2 : Prop :=
  vec_b_magnitude b ∧ vec_perpendicular vec_a b

theorem find_angle (h : condition2) : θ = π :=
sorry

end find_c_find_angle_l93_93662


namespace inflection_point_on_line_l93_93198

variable {x : ℝ}

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sin x - Real.cos x

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := -4 * Real.sin x + Real.cos x

-- Define the fourth derivative of f
-- Knowing f'' is a linear combination of sin and cos, f'''' will be a linear combination of sin and cos as well.
def f'''' (x : ℝ) : ℝ := -16 * Real.sin x - Real.cos x

-- Given condition: there exists an x_0 such that f''''(x_0) = 0
variable (x_0 : ℝ) (hx_0 : f'''' x_0 = 0)

-- Define the theorem to prove that (x_0, f(x_0)) lies on the line y = 3x
theorem inflection_point_on_line : f x_0 = 3 * x_0 := by
  sorry

end inflection_point_on_line_l93_93198


namespace delta_fraction_l93_93900

noncomputable def DEF : triangle := ⟨15, 36, 39⟩ -- Define triangle DEF with side lengths

noncomputable def WXYZ_area (ω : ℝ) : ℝ := 
  let s := (15 + 36 + 39) / 2
  let area_DEF := sqrt (s * (s - 15) * (s - 36) * (s - 39))
  γ * ω - δ * ω^2

theorem delta_fraction (p q : ℕ) (h_rel_prime : Nat.coprime p q) : 
  (3 * p / q = 1) → 
  (p = 60) ∧ (q = 169) ∧ ((3 * p / q = 1) → p + q = 229) :=
by
  sorry

end delta_fraction_l93_93900


namespace typing_time_in_hours_l93_93816

def words_per_minute := 32
def word_count := 7125
def break_interval := 25
def break_time := 5
def mistake_interval := 100
def correction_time_per_mistake := 1

theorem typing_time_in_hours :
  let typing_time := (word_count + words_per_minute - 1) / words_per_minute
  let breaks := typing_time / break_interval
  let total_break_time := breaks * break_time
  let mistakes := (word_count + mistake_interval - 1) / mistake_interval
  let total_correction_time := mistakes * correction_time_per_mistake
  let total_time := typing_time + total_break_time + total_correction_time
  let total_hours := (total_time + 60 - 1) / 60
  total_hours = 6 :=
by
  sorry

end typing_time_in_hours_l93_93816


namespace forest_enclosure_l93_93287

theorem forest_enclosure
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_a_lt_100 : ∀ i, a i < 100)
  (d : Fin n → Fin n → ℝ)
  (h_dist : ∀ i j, i < j → d i j ≤ (a i) - (a j)) :
  ∃ f : ℝ, f = 200 :=
by
  -- The proof goes here
  sorry

end forest_enclosure_l93_93287


namespace whole_numbers_count_between_cubic_roots_l93_93718

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93718


namespace algorithm_effective_expected_number_of_residents_l93_93487

-- Definitions required from the conditions of the original problem
def num_mailboxes : ℕ := 80

def key_distribution : Equiv.Perm (Fin num_mailboxes) := sorry

def initial_mailbox : Fin num_mailboxes := 37

-- Lean 4 statement for Part (a)
theorem algorithm_effective :
  ∃ m : Fin num_mailboxes, m = initial_mailbox → 
    (fix : ℕ → Fin num_mailboxes)
      (fix 0 = initial_mailbox)
      (∀ n, fix (n+1) = key_distribution (fix n))
      ∃ k, fix k = initial_mailbox := sorry

-- Lean 4 statement for Part (b)
theorem expected_number_of_residents :
  ∀ n, n = num_mailboxes → 
    let Harmonic := λ (n : ℕ), Σ i in Finset.range n, 1 / (i + 1)
    Harmonic n ≈ 4.968 := sorry

end algorithm_effective_expected_number_of_residents_l93_93487


namespace problem_l93_93224

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then real.sqrt x else (1/2)^x

theorem problem : f (f (-4)) = 4 := by
  sorry

end problem_l93_93224


namespace number_of_whole_numbers_between_cubicroots_l93_93736

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93736


namespace smallest_composite_no_prime_lt_20_l93_93059

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93059


namespace determine_shape_l93_93628

def Point (α : Type) := (x : α, y : α)

variables {α : Type} [NontrivialDivisionRing α]

def R (k : α) (P Q : Point α) : Point α :=
  match P, Q with
  | (x1, y1), (kx1, ky1) => (x1 + kx1, y1 + ky1)

theorem determine_shape (P Q R : Point α) (k : α) (hk : k ≠ 0) :
  (P.Q.is_parallelogram_if (k ≠ -1)) ∧ 
  (P.Q.is_rhombus_if (k = 1)) ∧ 
  (P.Q.is_line_if (k = -1)) :=
  sorry

end determine_shape_l93_93628


namespace angle_B_range_sin_A_plus_sin_C_l93_93329

-- Define the triangle ABC with sides and angles
variables {a b c A B C : ℝ}

-- Assume the triangle is acute-angled and the given cosine condition
variables (h1 : 0 < A ∧ A < π / 2)
variables (h2 : 0 < B ∧ B < π / 2)
variables (h3 : 0 < C ∧ C < π / 2)
variables (h4 : B + C = π - A)
variables (h5 : b * cos C = (2 * a - c) * cos B)

-- Prove the size of angle B
theorem angle_B : B = π / 3 :=
sorry

-- Prove the range of values for sin A + sin C
theorem range_sin_A_plus_sin_C :
  (3 / 2 < real.sin A + real.sin C) ∧ (real.sin A + real.sin C ≤ real.sqrt 3) :=
sorry

end angle_B_range_sin_A_plus_sin_C_l93_93329


namespace count_whole_numbers_between_roots_l93_93754

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93754


namespace math_problem_proof_l93_93229

-- Define the piecewise function f.
def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^3 - x^2 else Real.log x

-- Define the condition that f(x) is monotonically decreasing in a specific interval.
def is_monotonically_decreasing_in_interval : Prop :=
∀ (x : ℝ), x > 0 ∧ x < 2 / 3 → (f x) > (f (x)) -- This checks the decreasing property over the interval (0, 2/3)

-- Define the auxiliary function g.
def g (x : ℝ) : ℝ :=
f x - x

-- Define the condition that g(x) ≤ c for all x.
def g_leq_c (c : ℝ) : Prop :=
∀ x : ℝ, g x ≤ c

-- Define the statement combining all conditions and conclusions.
theorem math_problem_proof : 
  is_monotonically_decreasing_in_interval ∧ (∀ (c : ℝ), (∀ x : ℝ, f x ≤ x + c) → c >= 5 / 27) :=
by sorry

end math_problem_proof_l93_93229


namespace simplify_polynomial_l93_93206

open Nat

-- Define arithmetic sequence conditions and the polynomial structure
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def binomial (n k : ℕ) : ℕ := choose n k

noncomputable def p (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in range (n + 1), (a k) * (binomial n k) * (x^k) * ((1 - x)^(n - k))

-- The main theorem
theorem simplify_polynomial
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : arithmetic_seq a d)
  (n : ℕ)
  (x : ℝ)
  : p a n x = a 0 + n * d * x :=
sorry

end simplify_polynomial_l93_93206


namespace both_statements_correct_l93_93862

-- Define the conditions and claims in Lean

-- Context for Statement A
def statementA_conditions (ξ : ℝ → ℝ) (σ : ℝ) : Prop :=
  ξ ∼ Normal 3 σ^2 ∧ Prob (λ x, ξ x ≤ 2) = 0.3

def statementA_claim (ξ : ℝ → ℝ) (σ : ℝ) : Prop :=
  Prob (λ x, ξ x ≤ 4) = 0.7

-- Context for Statement B
def statementB_conditions (η : ℕ → ℕ) (n : ℕ) (p : ℝ) : Prop :=
  η ∼ Binom n p ∧ ExpectedValue η = 300 ∧ Variance η = 200 ∧ p = 1/3

def statementB_claim (η : ℕ → ℕ) (n : ℕ) (p : ℝ) : Prop :=
  True -- Since we are checking consistency and all given properties hold

-- Lean 4 statement to encapsulate the problem
theorem both_statements_correct (ξ : ℝ → ℝ) (σ : ℝ) (η : ℕ → ℕ) (n : ℕ) (p : ℝ) :
  (statementA_conditions ξ σ → statementA_claim ξ σ) ∧
  (statementB_conditions η n p → statementB_claim η n p) :=
by
  sorry

end both_statements_correct_l93_93862


namespace Emma_lowest_apartment_number_l93_93584

-- Definitions from problem statement
def phone_digits : list ℕ := [5, 4, 8, 1, 9, 8, 3]

def digit_sum (digits : list ℕ) : ℕ := digits.sum

def is_distinct (digits : list ℕ) : Prop := digits.nodup

def EmmaPhoneNumberSumValid : Prop := (digit_sum phone_digits = 38)

def four_digit_apartment_number (digits : list ℕ) : Prop :=
  digit_sum digits = 38 ∧ is_distinct digits ∧ (digits.length = 4)

noncomputable def lowest_apartment_number : ℕ :=
  digits_to_number (digits := list ℕ)

-- THEOREM: Given the conditions(Sum equals phone digits 38, 
-- Contains 4 distinct digits), prove the lowest valid apartment number
theorem Emma_lowest_apartment_number :
  EmmaPhoneNumberSumValid →
  ∃ n, four_digit_apartment_number (digits_of_nat n) :=
by sorry

end Emma_lowest_apartment_number_l93_93584


namespace domain_of_f_l93_93877

noncomputable def f (x : ℝ) : ℝ := sqrt (x + 1) + sqrt (1 - x) + x

theorem domain_of_f :
  {x : ℝ | 0 ≤ x + 1 ∧ 0 ≤ 1 - x} = {x | -1 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_f_l93_93877


namespace whole_numbers_between_cubicroots_l93_93712

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93712


namespace smallest_composite_no_prime_factors_below_20_l93_93126

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93126


namespace michael_final_revenue_l93_93357

noncomputable def total_revenue_before_discount : ℝ :=
  (3 * 45) + (5 * 22) + (7 * 16) + (8 * 10) + (10 * 5)

noncomputable def discount : ℝ := 0.10 * total_revenue_before_discount

noncomputable def discounted_revenue : ℝ := total_revenue_before_discount - discount

noncomputable def sales_tax : ℝ := 0.06 * discounted_revenue

noncomputable def final_revenue : ℝ := discounted_revenue + sales_tax

theorem michael_final_revenue : final_revenue = 464.60 :=
by
  sorry

end michael_final_revenue_l93_93357


namespace distance_from_B_to_orthocenter_l93_93607

theorem distance_from_B_to_orthocenter (a b : ℝ) (B K H : ℝ × ℝ) (BK BH : ℝ)
  (h_k_height : K = (B.1, B.2 + BK)) (h_h_height : H = (B.1 + BH, B.2))
  (h_kh_length : dist K H = a) (h_bd_length : dist B (B.1 + b / 2, B.2 + b / 2) = b)
  : dist B (orthocenter B K H) = real.sqrt (b^2 - a^2) := sorry

end distance_from_B_to_orthocenter_l93_93607


namespace angle_bisector_bisects_median_altitude_angle_iff_right_angle_l93_93814

theorem angle_bisector_bisects_median_altitude_angle_iff_right_angle 
  (A B C : Point) (h_neq : AC ≠ BC) : 
  (angle_bisector (∠ACB) bisects (angle_between (median_from C) (altitude_from C))) ↔ ∠ACB = 90 :=
  by
    sorry -- Proof to be provided

end angle_bisector_bisects_median_altitude_angle_iff_right_angle_l93_93814


namespace proper_fraction_reciprocal_greater_than_one_l93_93361

noncomputable def is_proper_fraction (r : ℚ) : Prop :=
r.num < r.denom

def is_reciprocal_greater_than_one (r : ℚ) : Prop :=
1 < r⁻¹

theorem proper_fraction_reciprocal_greater_than_one (r : ℚ) :
  is_proper_fraction r ↔ is_reciprocal_greater_than_one r :=
sorry

end proper_fraction_reciprocal_greater_than_one_l93_93361


namespace general_term_sum_of_terms_l93_93788

-- Definitions based on the conditions
def a (n : ℕ) : ℕ := n + 1

def S (n : ℕ) : ℕ := n * (2 + (n - 1) * 1) / 2

def b (n : ℕ) : ℕ := 9 / (2 * S (3 * n))

-- Main theorems
theorem general_term :
  ∀ n : ℕ, a n = n + 1 := sorry

theorem sum_of_terms :
  ∀ n : ℕ, (∑ i in Finset.range n, b i) = n / (n + 1) := sorry

end general_term_sum_of_terms_l93_93788


namespace tree_height_at_2_years_l93_93528

def tree_height (year : ℕ) : ℕ :=
  if year = 4 then 256 
  else (tree_height (year + 1)) / 4

theorem tree_height_at_2_years :
  tree_height 2 = 16 :=
sorry

end tree_height_at_2_years_l93_93528


namespace exists_common_point_in_intervals_l93_93326

theorem exists_common_point_in_intervals
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h : ∀ i j : Fin n, ∃ x : ℝ, a i ≤ x ∧ x ≤ b i ∧ a j ≤ x ∧ x ≤ b j) :
  ∃ p : ℝ, ∀ i : Fin n, a i ≤ p ∧ p ≤ b i :=
sorry

end exists_common_point_in_intervals_l93_93326


namespace smallest_composite_no_prime_factors_less_than_20_l93_93142

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93142


namespace sailboat_speed_l93_93402

-- Given conditions
def F (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v) ^ 2) / 2

def N (B S ρ v₀ v : ℝ) : ℝ :=
  F B S ρ v₀ v * v

noncomputable def N_derivative (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ / 2) * (v₀^2 - 4 * v₀ * v + 3 * v^2)

axiom derivative_zero_is_max (B S ρ v₀ v : ℝ) (h : N_derivative B S ρ v₀ v = 0) : 
  ∃ v, v = 2.1

theorem sailboat_speed (B S ρ v₀ : ℝ) (hS : S = 7) (hv₀ : v₀ = 6.3) : 
  ∃ v, N_derivative B S ρ v₀ v = 0 ∧ v = 2.1 := 
by
  use 2.1
  split
  -- You would provide the actual proof here connecting N_derivative B S ρ v₀ 2.1 = 0,
  -- but for now, we'll use sorry as a placeholder.
  sorry
  -- Showing that v equals 2.1 is trivial in this context by construction.
  rfl

end sailboat_speed_l93_93402


namespace TangentLineValue_l93_93217

theorem TangentLineValue (b a k : ℝ) 
    (htangent : ∀ x y, y = x^3 + ax + 1 → y = k * x + b → x = 2 ∧ y = 3) 
    (htangent_curve: 2^3 + 2 * a + 1 = 3) 
    (hslope : k = (derivative (λ x : ℝ, x^3 + a * x + 1)) 2) : 
    b = -15 := 
by 
  sorry

end TangentLineValue_l93_93217


namespace more_efficient_box_holds_more_l93_93504

-- Define the conditions
def initial_volume : ℝ := 24 -- volume in cm^3
def initial_pens : ℝ := 80 -- number of pens
def efficiency_increase : ℝ := 1.2 -- 20% greater efficiency
def new_volume : ℝ := 72 -- new volume in cm^3

-- Calculate the efficiency of the initial box
def initial_efficiency : ℝ := initial_pens / initial_volume

-- Adjust the efficiency for the more efficient material
def new_efficiency : ℝ := initial_efficiency * efficiency_increase

-- Calculate the number of pens the new box can hold
def new_pens : ℝ := new_efficiency * new_volume

-- Proof statement
theorem more_efficient_box_holds_more (h1 : initial_volume = 24)
                                      (h2 : initial_pens = 80)
                                      (h3 : efficiency_increase = 1.2)
                                      (h4 : new_volume = 72) :
  new_pens = 288 := sorry

end more_efficient_box_holds_more_l93_93504


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93744

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93744


namespace pyramid_distance_to_larger_cross_section_l93_93445

theorem pyramid_distance_to_larger_cross_section
  (A1 A2 : ℝ) (d : ℝ)
  (h : ℝ)
  (hA1 : A1 = 256 * Real.sqrt 2)
  (hA2 : A2 = 576 * Real.sqrt 2)
  (hd : d = 12)
  (h_ratio : (Real.sqrt (A1 / A2)) = 2 / 3) :
  h = 36 := 
  sorry

end pyramid_distance_to_larger_cross_section_l93_93445


namespace max_xy_value_l93_93767

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 200) : x * y ≤ 348 :=
begin
  sorry
end

example : ∃ x y : ℕ, 7 * x + 4 * y = 200 ∧ x * y = 348 :=
begin
  use [12, 29], -- These are the integers that satisfy the conditions and provide the maximum value of xy.
  split,
  { exact eq.refl 200, },
  { exact eq.refl 348, },
end

end max_xy_value_l93_93767


namespace whole_numbers_between_l93_93726

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93726


namespace eccentricity_of_curve_l93_93385

noncomputable def arithmetic_mean (x y : ℝ) : ℝ :=
  (x + y) / 2

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

def curve_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b/a)

theorem eccentricity_of_curve :
  let a := arithmetic_mean 1 9
  let b := geometric_mean 1 9
  b > 0 → curve_eccentricity a b = (Real.sqrt 10) / 5 :=
by
  intros
  sorry

end eccentricity_of_curve_l93_93385


namespace complex_number_solution_l93_93894

theorem complex_number_solution :
  ∃ (z : ℂ) (x y : ℕ) (c : ℤ), z = x + y * complex.I ∧ z^3 = -74 + c * complex.I ∧ z = 1 + 5 * complex.I :=
by
  sorry

end complex_number_solution_l93_93894


namespace second_person_avg_pages_per_day_l93_93569

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end second_person_avg_pages_per_day_l93_93569


namespace football_team_progress_l93_93930

constant team_lost : ℤ := -5
constant team_gained : ℤ := 7

theorem football_team_progress : team_lost + team_gained = 2 := sorry

end football_team_progress_l93_93930


namespace smallest_composite_no_prime_factors_lt_20_l93_93148

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93148


namespace coin_type_determination_l93_93424

theorem coin_type_determination (coins : Finset ℕ) (w : ℕ → ℕ)
  (at_least_one_each : ∃ g s c ∈ coins, g ≠ s ∧ s ≠ c ∧ c ≠ g ∧ w g = 3 ∧ w s = 2 ∧ w c = 1)
  (coin_types : coins.card = 100) :
  ∃ weighing_function : Finset ℕ → Finset ℕ → Prop,
    (∀ A B ∈ coins.powerset, weighing_function A B ∨ weighing_function B A) ∧
    (∀ A B, weighing_function A B → ∃ tA tB ∈ Finset.powerset (Finset.univ : Finset ℕ), 
      ∀ x∈ A ∪ B, (x ∈ tA ∨ x ∈ tB) → 
      ((x ∈ tA ∧ w x > 0) ∨ (x ∈ tB ∧ w x < 0) ∨ (x ∈ (A ∪ B) \ (tA ∪ tB) ∧ w x = 0))) ∧
    ∀ k < 100, ∀ coins_subset : Finset ℕ, coins_subset.card = k, 
      (∃ weighings: ℕ, weighings < 101 ∧ 
      (∀ t ∈ coins_subset, ∃ g s c ∈ coins_subset, g ≠ s ∧ s ≠ c ∧ c ≠ g 
      ∧ w g = 3 ∧ w s = 2 ∧ w c = 1)
      ) :=
sorry

end coin_type_determination_l93_93424


namespace rectangle_area_error_percentage_l93_93940

theorem rectangle_area_error_percentage 
  (L W : ℝ)
  (measured_length : ℝ := L * 1.16)
  (measured_width : ℝ := W * 0.95)
  (actual_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width) :
  ((measured_area - actual_area) / actual_area) * 100 = 10.2 := 
by
  sorry

end rectangle_area_error_percentage_l93_93940


namespace intersection_of_complements_l93_93421

open Set

theorem intersection_of_complements (U : Set ℕ) (A B : Set ℕ)
  (hU : U = {1,2,3,4,5,6,7,8})
  (hA : A = {3,4,5})
  (hB : B = {1,3,6}) :
  (U \ A) ∩ (U \ B) = {2,7,8} := by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l93_93421


namespace polygon_inside_rectangle_l93_93934

theorem polygon_inside_rectangle (S : ℝ) (P : Set (ℝ × ℝ)) (hP : ConvexPolygon P) (area_P : area P = S) :
  ∃ R : Set (ℝ × ℝ), (Rectangle R ∧ area R ≤ 2 * S ∧ P ⊆ R) :=
sorry

end polygon_inside_rectangle_l93_93934


namespace smallest_composite_no_prime_factors_lt_20_l93_93152

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93152


namespace whole_numbers_between_l93_93724

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93724


namespace smallest_solution_x_squared_minus_floor_x_squared_eq_23_l93_93164

theorem smallest_solution_x_squared_minus_floor_x_squared_eq_23 :
  ∃ x : ℝ, (int.floor (x^2) - (int.floor x)^2 = 23) ∧ (∀ y : ℝ, int.floor (y^2) - (int.floor y)^2 = 23 → y ≥ x) :=
sorry

end smallest_solution_x_squared_minus_floor_x_squared_eq_23_l93_93164


namespace sequence_b_l93_93834

theorem sequence_b (b : ℕ → ℕ) 
  (h1 : b 1 = 2) 
  (h2 : ∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) : 
  b 10 = 110 :=
sorry

end sequence_b_l93_93834


namespace seq_general_term_l93_93808

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 0 then 1/2
  else if n = 1 then 1/2
  else seq (n - 1) * 3 / (seq (n - 1) + 3)

theorem seq_general_term : ∀ n : ℕ, seq (n + 1) = 3 / (n + 6) :=
by
  intro n
  induction n with
  | zero => sorry
  | succ k ih => sorry

end seq_general_term_l93_93808


namespace parallel_vectors_x_value_l93_93251

theorem parallel_vectors_x_value (x : ℝ) :
  (3, -2) = (λ k, k * (x, 4)) → x = -6 :=
by
  sorry

end parallel_vectors_x_value_l93_93251


namespace train_speed_l93_93527

theorem train_speed (train_length bridge_length total_time : ℕ) (h1 : train_length = 170) (h2 : bridge_length = 205) (h3 : total_time = 30) : 
  (train_length + bridge_length) / total_time * 3.6 = 45 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end train_speed_l93_93527


namespace value_of_smaller_denom_l93_93535

-- We are setting up the conditions given in the problem.
variables (x : ℕ) -- The value of the smaller denomination bill.

-- Condition 1: She has 4 bills of denomination x.
def value_smaller_denomination : ℕ := 4 * x

-- Condition 2: She has 8 bills of $10 denomination.
def value_ten_bills : ℕ := 8 * 10

-- Condition 3: The total value of the bills is $100.
def total_value : ℕ := 100

-- Prove that x = 5 using the given conditions.
theorem value_of_smaller_denom : value_smaller_denomination x + value_ten_bills = total_value → x = 5 :=
by
  intro h
  -- Proof steps would go here
  sorry

end value_of_smaller_denom_l93_93535


namespace sailboat_speed_max_power_l93_93396

-- Define the conditions
def S : ℝ := 7
def v0 : ℝ := 6.3

-- Define the power function N as a function of sailboat speed v
def N (B ρ v : ℝ) : ℝ :=
  (B * S * ρ / 2) * (v0 ^ 2 * v - 2 * v0 * v ^ 2 + v ^ 3)

-- State the theorem we need to prove
theorem sailboat_speed_max_power (B ρ : ℝ) : 
  ∀ (v : ℝ), 
  (∃ v : ℝ, v = 2.1) :=
begin
  -- Proof goes here
  sorry
end

end sailboat_speed_max_power_l93_93396


namespace find_x_when_y_is_20_l93_93639

variable (x y k : ℝ)

axiom constant_ratio : (5 * 4 - 6) / (5 + 20) = k

theorem find_x_when_y_is_20 (h : (5 * x - 6) / (y + 20) = k) (hy : y = 20) : x = 5.68 := by
  sorry

end find_x_when_y_is_20_l93_93639


namespace smallest_composite_no_prime_factors_less_than_20_l93_93099

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93099


namespace amount_of_25_yuan_candy_l93_93990

-- Defining the given conditions
def candy_problem (x y z : ℕ) : Prop :=
  (x + y + z = 100) ∧
  (20 * x + 25 * y + 30 * z = 2570) ∧
  (25 * y + 30 * z = 1970)

-- Proving the main statement
theorem amount_of_25_yuan_candy (x y z : ℕ): candy_problem x y z → y = 26 :=
by
  intro h,
  sorry

end amount_of_25_yuan_candy_l93_93990


namespace variance_scaled_by_2_l93_93889

variable {n : ℕ}
variable {data : Fin n → ℝ}
variable {s : ℝ}
variable (h : var data s)

theorem variance_scaled_by_2 : var (fun i => 2 * data i) (4 * s) :=
sorry

end variance_scaled_by_2_l93_93889


namespace candy_ratio_l93_93864

theorem candy_ratio 
  (tabitha_candy : ℕ)
  (stan_candy : ℕ)
  (julie_candy : ℕ)
  (carlos_candy : ℕ)
  (total_candy : ℕ)
  (h1 : tabitha_candy = 22)
  (h2 : stan_candy = 13)
  (h3 : julie_candy = tabitha_candy / 2)
  (h4 : total_candy = 72)
  (h5 : tabitha_candy + stan_candy + julie_candy + carlos_candy = total_candy) :
  carlos_candy / stan_candy = 2 :=
by
  sorry

end candy_ratio_l93_93864


namespace root_exponent_equiv_l93_93878

theorem root_exponent_equiv :
  (7 ^ (1 / 2)) / (7 ^ (1 / 4)) = 7 ^ (1 / 4) := by
  sorry

end root_exponent_equiv_l93_93878


namespace count_whole_numbers_between_roots_l93_93760

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93760


namespace find_f_find_max_g_value_x_max_g_value_l93_93666

-- Define the given vectors and conditions
variables {x : ℝ}

-- Assume the vectors
def m (x : ℝ) : ℝ × ℝ := (λ x, 2 * cos x)
def n (x : ℝ) : ℝ × ℝ := (sin x + cos x, 1)

-- m is parallel to n condition
axiom parallel_m_n : ∀ x, m(x).1 * n(x).2 = m(x).2 * n(x).1

-- Proof steps
theorem find_f : f(x) = sqrt 2 * sin(2 * x + π / 4) + 1 := 
begin
  sorry
end

theorem find_max_g_value :
  ∀ x ∈ set.Icc(0, π / 8),
  let g := sqrt 2 * sin(4 * x + π / 4)
  in g ≤ sqrt 2
 :=
begin
 sorry
end

theorem x_max_g_value :
  let g := sqrt 2 * sin(4 * (π / 16) + π / 4)
  in g = sqrt 2
 :=
begin
 sorry
end

end find_f_find_max_g_value_x_max_g_value_l93_93666


namespace smallest_composite_no_prime_factors_less_than_20_l93_93110

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93110


namespace light_stripes_total_area_l93_93420

theorem light_stripes_total_area (x : ℝ) (h : 45 * x = 135) :
  2 * x + 4 * x + 6 * x + 8 * x = 60 := 
sorry

end light_stripes_total_area_l93_93420


namespace sum_floors_eq_half_product_l93_93841

theorem sum_floors_eq_half_product {p q : ℕ} (hpq_coprime : Nat.coprime p q) :
  ∑ k in Finset.range(p), if k = 0 then 0 else Nat.floor (k * q / p) = (p - 1) * (q - 1) / 2 := 
sorry

end sum_floors_eq_half_product_l93_93841


namespace saree_original_price_is_approx_550_l93_93415

noncomputable def original_price (sale_price : ℝ) : ℝ :=
  sale_price / (0.82 * 0.88)

theorem saree_original_price_is_approx_550 :
  original_price 396.88 ≈ 550 :=
by
  unfold original_price
  apply (λ x, (396.88 / (0.82 * 0.88)) ≈ x) 
  sorry

end saree_original_price_is_approx_550_l93_93415


namespace part_a_part_b_l93_93971

variable (p : ℝ)
variable (h_pos : 0 < p)
variable (h_prob : p ≤ 1)

theorem part_a :
  let q := 1 - p in
  ∃ f : ℕ → ℝ, f 5 = 6 * p^3 * q^2 :=
  by
    sorry

theorem part_b :
  ∃ f : ℕ → ℝ, f 3 = 3 / p :=
  by
    sorry

end part_a_part_b_l93_93971


namespace remainder_substitution_ways_mod_1000_l93_93995

def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 11 * (12 - (n + 1)) * a n

noncomputable def total_substitution_ways : ℕ :=
a 0 + a 1 + a 2 + a 3 + a 4

theorem remainder_substitution_ways_mod_1000 :
  total_substitution_ways % 1000 = 522 :=
by
  sorry

end remainder_substitution_ways_mod_1000_l93_93995


namespace square_area_is_25_l93_93847

noncomputable def point1 : ℝ × ℝ := (1, 3)
noncomputable def point2 : ℝ × ℝ := (4, 7)

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def side_length : ℝ := distance point1 point2

def square_area (side : ℝ) : ℝ := side ^ 2

theorem square_area_is_25 : square_area side_length = 25 := by
  sorry

end square_area_is_25_l93_93847


namespace length_AK_independent_of_D_l93_93843

variables {A B C D K : Point}
variables (AC AB BC : Length)

-- Define that point D lies on side AC
axiom point_on_triangle_side (A B C D : Point) : lies_on D (segment A C)

-- Construct incircles of respective triangles
axiom incircle_triangles (A B C D : Point) : incircle (triangle A B D) ∧ incircle (triangle A C D)

-- Define the external common tangents (excluding BC) intersect AD at K
axiom external_tangents_intersect (A B C D K : Point) : intersects (external_common_tangents (incircle (triangle A B D)) (incircle (triangle A C D))) (segment A D) K

theorem length_AK_independent_of_D (A B C D K : Point) (AC AB BC : Length) :
  point_on_triangle_side A B C D →
  incircle_triangles A B C D →
  external_tangents_intersect A B C D K →
  length_segment (segment A K) = (AC + AB - BC) / 2 :=
by
  sorry

end length_AK_independent_of_D_l93_93843


namespace cube_factors_count_l93_93669

theorem cube_factors_count :
  let perfect_cube_factors := [8, 27, 64]
  let nums_with_cube_factors := (Finset.range 101).filter (λ n, ∃ k ∈ perfect_cube_factors, k ∣ n)
  nums_with_cube_factors.card = 15 :=
by
  let perfect_cube_factors := [8, 27, 64]
  let nums_with_cube_factors := (Finset.range 101).filter (λ n, ∃ k ∈ perfect_cube_factors, k ∣ n)
  have h : nums_with_cube_factors.card = 15
  { sorry }
  exact h

end cube_factors_count_l93_93669


namespace max_nonempty_subset_T_l93_93332

def max_num_elements_in_T : ℕ :=
  32

theorem max_nonempty_subset_T (S : set (fin 6 → bool)) (T : set (fin 6 → bool))
  (cond1: ∀ u v ∈ T, u ≠ v → dot_product u v ≠ 0):
  ∀ T ⊆ S, (T.nonempty) → T.card ≤ max_num_elements_in_T := 
sorry

noncomputable def dot_product (x y : fin 6 → bool) : ℕ :=
  (finset.range 6).sum (λ i, (if x i = tt then 1 else 0) * (if y i = tt then 1 else 0))

def set_T (S : set (fin 6 → bool)) :=
  { x : fin 6 → bool | ∀ y ∈ S, y ≠ x → dot_product x y ≠ 0 }

end max_nonempty_subset_T_l93_93332


namespace line_passes_through_fixed_point_l93_93605

theorem line_passes_through_fixed_point :
  ∃ (P : ℝ × ℝ), P = (3, -1) ∧ 
  ∀ (k : ℝ), (k + 2) * P.1 + (1 - k) * P.2 - 4 * k - 5 = 0 :=
by
  let P := (3 : ℝ, -1 : ℝ)
  use P
  split
  · refl
  · intro k
    calc
      (k + 2) * P.1 + (1 - k) * P.2 - 4 * k - 5
          = (k + 2) * 3 + (1 - k) * (-1) - 4 * k - 5 : by rfl
      ... = k * 3 + 2 * 3 + 1 * (-1) - k * (-1) - 4 * k - 5 : by ring
      ... = 3 * k + 6 - 1 + k - 4 * k - 5 : by ring
      ... = 0 : by ring
  sorry

end line_passes_through_fixed_point_l93_93605


namespace calculate_f_log2_5_l93_93209

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f(x) = 2^x
axiom cond2 : ∀ x : ℝ, f(x + 1) = (1 - 2 * f(x)) / (2 - f(x))

theorem calculate_f_log2_5 : f (Real.log 5 / Real.log 2) = 5 / 4 := by sorry

end calculate_f_log2_5_l93_93209


namespace all_numbers_same_color_l93_93581

theorem all_numbers_same_color (n : ℕ) (h_n : n ≥ 3) 
(colors : fin (n - 1) → bool)
(hcond1 : ∀ i ∈ fin (n - 1), colors i = colors (n - 1 - i))
(hcond2 : ∃ j ∈ fin (n - 1), j.gcd n = 1 ∧ ∀ i ∈ fin (n - 1), i ≠ j → colors i = colors (|j - i|)) :
  ∀ i j ∈ fin (n - 1), colors i = colors j :=
begin
  sorry
end

end all_numbers_same_color_l93_93581


namespace cucumbers_count_l93_93543

theorem cucumbers_count (C T : ℕ) 
  (h1 : C + T = 280)
  (h2 : T = 3 * C) : C = 70 :=
by sorry

end cucumbers_count_l93_93543


namespace ice_cream_melt_time_l93_93314

theorem ice_cream_melt_time :
  let blocks := 16
  let block_length := 1.0/8.0 -- miles per block
  let distance := blocks * block_length -- in miles
  let speed := 12.0 -- miles per hour
  let time := distance / speed -- in hours
  let time_in_minutes := time * 60 -- converted to minutes
  time_in_minutes = 10 := by sorry

end ice_cream_melt_time_l93_93314


namespace intersection_points_count_l93_93655

def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc (-1) 1 then x^2
  else if x % 2 = 0 then f (x - 2)
  else f (x - 1)

theorem intersection_points_count : 
  (∃ n : ℕ, n = 4 ∧ 
  ∀ y : ℝ, y ∈ set.range f → y = Real.log 5) :=
  sorry

end intersection_points_count_l93_93655


namespace smallest_composite_no_prime_factors_less_than_20_l93_93114

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93114


namespace train_crossing_time_correct_l93_93908

noncomputable def train_crossing_time (speed_kmph : ℕ) (length_m : ℕ) (train_dir_opposite : Bool) : ℕ :=
  if train_dir_opposite then
    let speed_mps := speed_kmph * 1000 / 3600
    let relative_speed := speed_mps + speed_mps
    let total_distance := length_m + length_m
    total_distance / relative_speed
  else 0

theorem train_crossing_time_correct :
  train_crossing_time 54 120 true = 8 :=
by
  sorry

end train_crossing_time_correct_l93_93908


namespace sufficient_not_necessary_condition_l93_93951

variable (φ : ℝ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, ∃ k : ℤ, f (x : ℝ) = cos(x + φ) ∧ is_even (cos (x + φ)) → φ = 0) ∧
  (∃ k : ℤ, φ = k * π → φ ≠ 0) :=
by
  sorry

end sufficient_not_necessary_condition_l93_93951


namespace exists_four_functions_l93_93352

noncomputable def f (x : ℝ) : ℝ := sorry
axiom f_periodic : ∀ x, f(x + 2 * Real.pi) = f(x)

def even_function (g : ℝ → ℝ) : Prop := ∀ x, g(-x) = g(x)
def pi_periodic (g : ℝ → ℝ) : Prop := ∀ x, g(x + Real.pi) = g(x)

theorem exists_four_functions :
  ∃ f_1 f_2 f_3 f_4 : ℝ → ℝ,
    (even_function f_1 ∧ pi_periodic f_1) ∧
    (even_function f_2 ∧ pi_periodic f_2) ∧
    (even_function f_3 ∧ pi_periodic f_3) ∧
    (even_function f_4 ∧ pi_periodic f_4) ∧
    (∀ x, f(x) = f_1(x) + f_2(x) * Real.cos x + f_3(x) * Real.sin x + f_4(x) * Real.sin (2 * x)) :=
sorry

end exists_four_functions_l93_93352


namespace quadratic_two_distinct_real_roots_l93_93280

def quadratic_function_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := -4
  let c := -2
  b * b - 4 * a * c > 0 ∧ a ≠ 0

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  quadratic_function_has_two_distinct_real_roots k ↔ (k > -2 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l93_93280


namespace tan_C_eq_sqrt_5_area_of_triangle_ABC_l93_93285

theorem tan_C_eq_sqrt_5 (cos_A : ℝ) (hcosA: cos_A = 2 / 3) 
    (sin_B : ℝ) (cos_C : ℝ) (hsinB: sin_B = sqrt 5 * cos_C) : 
    tan (cos_C : ℝ) * sqrt (1 - cos_C^2) = sqrt 5 :=
by sorry

theorem area_of_triangle_ABC (a : ℝ) (cos_A : ℝ) (hcosA: cos_A = 2 / 3) 
    (sin_B : ℝ) (cos_C : ℝ) (hsinB: sin_B = sqrt 5 * cos_C)
    (htanC: tan (cos_C : ℝ) * sqrt (1 - cos_C^2) = sqrt 5) (ha: a = sqrt 2) : 
    let c := sqrt 3 in
    let b := sqrt 3 in 
    1 / 2 * a * b * sqrt (5 / 6) = sqrt 5 / 2 :=
by sorry

end tan_C_eq_sqrt_5_area_of_triangle_ABC_l93_93285


namespace problem1_solution_l93_93345

theorem problem1_solution (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ) (ha : 0 < a ∧ a ≤ p) (hb : 0 < b ∧ b ≤ p) (hc : 0 < c ∧ c ≤ p)
  (f : ℕ → ℕ) (hf : ∀ x : ℕ, 0 < x → p ∣ f x) :
  (∀ x, f x = a * x^2 + b * x + c) →
  (p = 2 → a + b + c = 4) ∧ (2 < p → p % 2 = 1 → a + b + c = 3 * p) :=
by
  sorry

end problem1_solution_l93_93345


namespace sum_of_valid_x_values_l93_93524

theorem sum_of_valid_x_values (x y : ℕ) (h1 : x * y = 360) (h2 : x ≥ 18) (h3 : y ≥ 12) :
  (∑ x in {x | ∃ y, x * y = 360 ∧ y ≥ 12 ∧ x ≥ 18}, x) = 92 :=
by 
  sorry

end sum_of_valid_x_values_l93_93524


namespace smallest_composite_no_prime_factors_less_than_20_l93_93138

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93138


namespace count_divisible_by_11_up_to_100_l93_93334

def bn (n : ℕ) : ℕ :=
String.toNat (String.join (List.map repr (List.range' 1 (n + 1))))

def is_divisible_by_11 (n : ℕ) : Prop :=
n % 11 = 0

theorem count_divisible_by_11_up_to_100 :
  (List.countp (λ n, is_divisible_by_11 (bn n)) (List.range' 1 101) = 4) :=
by {
  sorry
}

end count_divisible_by_11_up_to_100_l93_93334


namespace relative_speed_opposite_direction_l93_93999

theorem relative_speed_opposite_direction :
  ∃ (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) (pass_time : ℝ),
    train_length = 110 ∧
    train_speed = 40 * (1000 / 3600) ∧  -- convert km/h to m/s
    man_speed = 4 * (1000 / 3600) ∧  -- convert km/h to m/s
    pass_time = 9 ∧
    train_length / pass_time = (train_speed + man_speed)
:=
by
  -- Definitions of given values
  let train_length : ℝ := 110
  let train_speed : ℝ := 40 * (1000 / 3600)  -- convert km/h to m/s
  let man_speed : ℝ := 4 * (1000 / 3600)  -- convert km/h to m/s
  let pass_time : ℝ := 9

  -- Proof that the given conditions are satisfied
  use [train_length, train_speed, man_speed, pass_time]
  split
  · exact rfl
  split
  · exact rfl
  split
  · exact rfl
  · -- Show the condition train_length / pass_time = (train_speed + man_speed) holds.
    calc
      train_length / pass_time = 110 / 9              : by rfl
                        ...   = 12.22                  : by norm_num
                        ...   = (11.11 + 1.11)        : by norm_num -- the sum is approximately equivalent.

end relative_speed_opposite_direction_l93_93999


namespace find_ellipse_equation_range_area_AF1F2D_l93_93215

noncomputable def ellipse_focal_length := 4
noncomputable def f1 := (2 : Real, 0 : Real)
noncomputable def f2 := (-2 : Real, 0 : Real)
noncomputable def parabola := λ x : Real, x^2

def intersection_point := (2 : Real, Real.sqrt 2)

def ellipse_equation := λ (x y : Real), (x^2 / 8 + y^2 / 4 = 1)

def area_AF1F2D (t : Real) : Real := (8 * Real.sqrt 2 * Real.sqrt (t^2 + 1)) / (t^2 + 2)

theorem find_ellipse_equation :
  ∃ a b : Real, a = 2 * Real.sqrt 2 ∧ b = 2 ∧ ellipse_equation (a, b) := sorry

theorem range_area_AF1F2D :
  ∃ s : Real, 1 ≤ s ∧ s < 3 ∧ 
  (area_AF1F2D s) ∈ Set.Icc (12 * Real.sqrt 2 / 5) (4 * Real.sqrt 2) := sorry

end find_ellipse_equation_range_area_AF1F2D_l93_93215


namespace projection_of_b_in_direction_of_a_l93_93664

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (a b : V) (h₁ : ∥a∥ = 2) (h₂ : ∥b∥ = 2) (angle : real.angle) 
variable (h_angle : angle = real.angle.ofRealDegrees 60)

-- Question: The projection of b in the direction of a.
theorem projection_of_b_in_direction_of_a : 
  inner_product_space.proj a b = (1 : V) :=
by 
  sorry

end projection_of_b_in_direction_of_a_l93_93664


namespace whole_numbers_between_cubicroots_l93_93708

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93708


namespace percent_area_square_in_rectangle_l93_93526

theorem percent_area_square_in_rectangle 
  (s : ℝ) (rect_width : ℝ) (rect_length : ℝ) (h1 : rect_width = 2 * s) (h2 : rect_length = 2 * rect_width) : 
  (s^2 / (rect_length * rect_width)) * 100 = 12.5 :=
by
  sorry

end percent_area_square_in_rectangle_l93_93526


namespace probability_white_ball_l93_93938

/-- 
Given a bag with 3 balls of unknown color (white or non-white),
and 1 white ball is added to the bag, 
prove that the probability of drawing a white ball is 5 / 8 
if all initial configurations are equally likely.
-/
theorem probability_white_ball (bag : list bool) (added_ball : bool) :
  list.length bag = 3 → (∀ ball, ball ∈ bag → ball = tt ∨ ball = ff) →
  (added_ball = tt) →
  (∀ config ∈ [ [ff, ff, ff], [ff, ff, tt], [ff, tt, ff], [tt, ff, ff],
                 [ff, tt, tt], [tt, ff, tt], [tt, tt, ff], [tt, tt, tt] ], config ∈ bag.permutations) →
  (∑ config in [ [ff, ff, ff, tt], [ff, ff, tt, tt], [ff, tt, ff, tt],
                  [tt, ff, ff, tt], [ff, tt, tt, tt], [tt, ff, tt, tt],
                  [tt, tt, ff, tt], [tt, tt, tt, tt] ],
     (list.count tt config / list.length config : ℚ)) / 8 = 5 / 8 :=
by
  intros h1 h2 h3 h4
  sorry

end probability_white_ball_l93_93938


namespace minimize_M_l93_93609

noncomputable def M (x y : ℝ) : ℝ := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9

theorem minimize_M : ∃ x y, M x y = 5 ∧ x = -3 ∧ y = -2 :=
by
  sorry

end minimize_M_l93_93609


namespace triangle_area_0_75_l93_93428

theorem triangle_area_0_75 :
  let A := (1, 0)
  let B := (2, 1.5)
  let C := (2, 0) in
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 0.75 :=
by
  let A := (1, 0)
  let B := (2, 1.5)
  let C := (2, 0)
  show (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 0.75
  sorry

end triangle_area_0_75_l93_93428


namespace angle_in_acute_triangle_leq_60_l93_93194

variable {A B C : ℝ}
variable (h_A m_B : ℝ)
variable [acute_triangle : ∀ A B C, (angle A B C < 90 ∧ angle B A C < 90 ∧ angle C A B < 90)]
variable [longest_altitude : ∀ A B C h_A m_B, (h_A = m_B)]
variable [angle_condition : ∀ A B C, (angle A B C ≤ 60)]

theorem angle_in_acute_triangle_leq_60 (A B C : Point)
    (acute_triangle : angle A B C  < 90 ∧ angle B A C < 90 ∧ angle C A B < 90)
    (longest_altitude : altitude_length A B C = median_length B A C)
    : angle A B C ≤ 60 := 
sorry

end angle_in_acute_triangle_leq_60_l93_93194


namespace sum_of_edges_l93_93890

-- Define the properties of the rectangular solid
variables (a b c : ℝ)
variables (V : ℝ) (S : ℝ)

-- Set the conditions
def geometric_progression := (a * b * c = V) ∧ (2 * (a * b + b * c + c * a) = S) ∧ (∃ k : ℝ, k ≠ 0 ∧ a = b / k ∧ c = b * k)

-- Define the main proof statement
theorem sum_of_edges (hV : V = 1000) (hS : S = 600) (hg : geometric_progression a b c V S) : 
  4 * (a + b + c) = 120 :=
sorry

end sum_of_edges_l93_93890


namespace max_power_speed_l93_93401

noncomputable def force (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v)^2) / 2

def power (B S ρ v₀ v : ℝ) : ℝ :=
  force B S ρ v₀ v * v

theorem max_power_speed (B ρ : ℝ) (S : ℝ := 7) (v₀ : ℝ := 6.3) :
  ∃ v, v = 2.1 ∧ (∀ v', power B S ρ v₀ v' ≤ power B S ρ v₀ v) :=
begin
  sorry,
end

end max_power_speed_l93_93401


namespace no_solutions_to_inequality_l93_93173

theorem no_solutions_to_inequality (x : ℝ) : ¬(3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  intro h,
  -- Simplify the inequality by dividing each term by 3
  have h_simplified : x^2 + 3 * x + 4 ≤ 0 := by linarith,
  -- Compute the discriminant of the quadratic expression to show it's always positive
  let a := (1 : ℝ),
  let b := (3 : ℝ),
  let c := (4 : ℝ),
  let discriminant := b^2 - 4 * a * c,
  have h_discriminant : discriminant < 0 := by norm_num,
  -- Since discriminant is negative, the quadratic has no real roots, thus x^2 + 3x + 4 > 0
  have h_positive : ∀ x, x^2 + 3 * x + 4 > 0 := 
    by {
      intro x,
      apply (quadratic_not_negative_of_discriminant neg_discriminant).mp,
      exact h_discriminant,
    },
  exact absurd (show x^2 + 3 * x + 4 ≤ 0 from h_simplified) (lt_irrefl 0 (h_positive x)),
}

end no_solutions_to_inequality_l93_93173


namespace part1_odd_m_part2_find_pairs_l93_93891

-- Part 1
theorem part1_odd_m 
  (m n : ℕ) (k : ℕ) 
  (h1 : m ≥ n) 
  (h2 : n ≥ 2)
  (h3 : Nat.choose m 2 * Nat.choose (m + n) 2 = k * m * n * Nat.choose (m + n) 2) : 
  Odd m := 
sorry

-- Part 2
theorem part2_find_pairs 
  (m n : ℕ)
  (h1 : m ≥ n) 
  (h2 : n ≥ 2)
  (h3 : m + n ≤ 40)
  (h4 : Nat.choose m 2 + Nat.choose n 2 = m * n) :
  ∃ (pairs : List (ℕ × ℕ)), ∀ p ∈ pairs, let ⟨m, n⟩ := p in m + n ≤ 40 ∧ m ≥ n ∧ n ≥ 2 ∧ Nat.choose m 2 + Nat.choose n 2 = m * n := 
sorry

end part1_odd_m_part2_find_pairs_l93_93891


namespace books_loaned_out_during_the_year_l93_93430

theorem books_loaned_out_during_the_year :
  ∃ (X : ℕ), (0.40 * X).to_nat = 50 ∧ X = 125 :=
by
  have h1 : 0.40 * 125 = 50 := by norm_num
  use 125
  exact ⟨by norm_num, rfl⟩

end books_loaned_out_during_the_year_l93_93430


namespace count_whole_numbers_between_roots_l93_93762

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93762


namespace smallest_composite_no_prime_factors_less_20_l93_93043

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93043


namespace inequality_solution_set_range_of_a_l93_93650

noncomputable def f (x : ℝ) := abs (2 * x - 1) - abs (x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x > 0 } = { x : ℝ | x < -1 / 3 ∨ x > 3 } :=
sorry

theorem range_of_a (x0 : ℝ) (h : f x0 + 2 * a ^ 2 < 4 * a) :
  -1 / 2 < a ∧ a < 5 / 2 :=
sorry

end inequality_solution_set_range_of_a_l93_93650


namespace smallest_composite_no_prime_factors_less_than_20_l93_93133

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93133


namespace value_of_f_neg1_l93_93453

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 2 := by
  sorry

end value_of_f_neg1_l93_93453


namespace jordan_8_miles_time_l93_93824

-- Define the conditions
variable (t_S : ℕ) -- time it took Steve to run 3 miles (in minutes)
variable (t_J : ℕ) -- time it took Jordan to run 2 miles (in minutes)

-- Assigning values based on the problem's conditions
def steve_time : ℕ := 24
def jordan_time : ℕ := steve_time / 2

-- The proof statement we want to check
theorem jordan_8_miles_time : jordan_time * 4 = 48 := by
  -- Utilizing the definitions
  have h : jordan_time = 12 := by
    unfold jordan_time
    unfold steve_time
    norm_num
  rw [h]
  norm_num
  sorry -- Proof details omitted

end jordan_8_miles_time_l93_93824


namespace math_problem_l93_93796

noncomputable def parametric_to_general (t : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + t ∧ y = t + 1 ∧ (x - y - 1 = 0)

noncomputable def polar_to_rect (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), ρ^2 - 4 * ρ * cos θ + 3 = 0 ∧ x = ρ * cos θ ∧ y = ρ * sin θ ∧ (x^2 + y^2 - 4 * x + 3 = 0)

noncomputable def chord_length (A B : ℝ × ℝ) : Prop :=
  let C := (2, 0) in
  let r := 1 in
  let d := 1 / Real.sqrt 2 in
  let AB := (A, B) in
  ∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ abs ((A.1 - B.1)^2 + (A.2 - B.2)^2)^0.5 = Real.sqrt 2

theorem math_problem : ∃ (t : ℝ) (x y : ℝ), parametric_to_general t ∧ polar_to_rect x y ∧ chord_length (2, 1) (1 - Real.sqrt 2 / 2, 2 - Real.sqrt 2 / 2) :=
  sorry

end math_problem_l93_93796


namespace ferry_journey_difference_l93_93465

theorem ferry_journey_difference
  (time_P : ℝ) (speed_P : ℝ) (mult_Q : ℝ) (speed_diff : ℝ)
  (dist_P : ℝ := time_P * speed_P)
  (dist_Q : ℝ := mult_Q * dist_P)
  (speed_Q : ℝ := speed_P + speed_diff)
  (time_Q : ℝ := dist_Q / speed_Q) :
  time_P = 3 ∧ speed_P = 6 ∧ mult_Q = 3 ∧ speed_diff = 3 → time_Q - time_P = 3 := by
  sorry

end ferry_journey_difference_l93_93465


namespace smallest_composite_no_prime_factors_less_than_20_l93_93094

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93094


namespace part_1_solution_set_part_2_range_k_l93_93653

def f (x k : ℝ) : ℝ :=
  if x ≤ k then exp x - x else x^3 - x + 1

def g (x k : ℝ) : ℝ := f x k - 1

theorem part_1_solution_set (k : ℝ) (h : k = 0) :
  {x : ℝ | f x k < 1} = set.Ioo 0 1 :=
sorry

theorem part_2_range_k :
  {k : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 k = 0 ∧ g x2 k = 0} = set.Ico (-1 : ℝ) 1 :=
sorry

end part_1_solution_set_part_2_range_k_l93_93653


namespace num_whole_numbers_between_l93_93698

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93698


namespace bike_tire_fixing_charge_l93_93819

theorem bike_tire_fixing_charge (total_profit rent_profit retail_profit: ℝ) (cost_per_tire_parts charge_per_complex_parts charge_per_complex: ℝ) (complex_repairs tire_repairs: ℕ) (charge_per_tire: ℝ) :
  total_profit  = 3000 → rent_profit = 4000 → retail_profit = 2000 →
  cost_per_tire_parts = 5 → charge_per_complex_parts = 50 → charge_per_complex = 300 →
  complex_repairs = 2 → tire_repairs = 300 →
  total_profit = (tire_repairs * charge_per_tire + complex_repairs * charge_per_complex + retail_profit - tire_repairs * cost_per_tire_parts - complex_repairs * charge_per_complex_parts - rent_profit) →
  charge_per_tire = 20 :=
by 
  sorry

end bike_tire_fixing_charge_l93_93819


namespace whole_numbers_count_between_cubic_roots_l93_93715

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93715


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93076

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93076


namespace smallest_composite_no_prime_factors_below_20_l93_93119

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93119


namespace smallest_composite_no_prime_factors_less_20_l93_93040

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93040


namespace smallest_composite_no_prime_factors_less_than_20_l93_93034

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93034


namespace real_values_of_x_eq_1_l93_93167

theorem real_values_of_x_eq_1 : 
  ∀ x : ℝ, (∀ (z : ℂ), z = 1 - (x / 2) * complex.I → complex.abs z = 1 → x = 0) →
  (∃! (x : ℝ), complex.abs (1 - (x / 2) * complex.I) = 1) :=
by
  sorry

end real_values_of_x_eq_1_l93_93167


namespace percent_increase_is_nine_point_four_l93_93323

def skateboard_cost_last_year := 120
def padding_cost_last_year := 30
def skateboard_increase_percentage := 0.08
def padding_increase_percentage := 0.15

def new_skateboard_cost :=
  skateboard_cost_last_year + skateboard_cost_last_year * skateboard_increase_percentage

def new_padding_cost :=
  padding_cost_last_year + padding_cost_last_year * padding_increase_percentage

def total_cost_last_year :=
  skateboard_cost_last_year + padding_cost_last_year

def total_cost_this_year :=
  new_skateboard_cost + new_padding_cost

def cost_increase :=
  total_cost_this_year - total_cost_last_year

def percent_increase :=
  (cost_increase / total_cost_last_year) * 100

theorem percent_increase_is_nine_point_four :
  percent_increase = 9.4 :=
by
  sorry

end percent_increase_is_nine_point_four_l93_93323


namespace abs_value_identity_l93_93777

theorem abs_value_identity (a : ℝ) (h : a + |a| = 0) : a - |2 * a| = 3 * a :=
by
  sorry

end abs_value_identity_l93_93777


namespace new_concentration_is_37_l93_93529

-- Define the capacity and concentration of two vessels
def vessel1_capacity : ℝ := 2  -- capacity of vessel 1 in liters
def vessel1_concentration : ℝ := 0.35  -- concentration of alcohol in vessel 1

def vessel2_capacity : ℝ := 6  -- capacity of vessel 2 in liters
def vessel2_concentration : ℝ := 0.50  -- concentration of alcohol in vessel 2

-- Total capacity of the 10-liter vessel
def total_capacity : ℝ := 10

-- Total volume of the mixture
def mixture_volume : ℝ := vessel1_capacity + vessel2_capacity -- 8 liters from both vessels

-- Total alcohol content in the mixture
def total_alcohol : ℝ := vessel1_capacity * vessel1_concentration + vessel2_capacity * vessel2_concentration

-- Define the condition
def concentration_of_mixture := (total_alcohol / total_capacity) * 100

-- The theorem to prove
theorem new_concentration_is_37 : concentration_of_mixture = 37 := by
  sorry

end new_concentration_is_37_l93_93529


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93074

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93074


namespace smallest_composite_no_prime_factors_less_than_20_l93_93007

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93007


namespace whole_numbers_between_l93_93732

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93732


namespace largest_angle_of_obtuse_isosceles_triangle_l93_93903

variables (X Y Z : ℝ)

def is_triangle (X Y Z : ℝ) : Prop := X + Y + Z = 180
def is_isosceles_triangle (X Y : ℝ) : Prop := X = Y
def is_obtuse_triangle (X Y Z : ℝ) : Prop := X > 90 ∨ Y > 90 ∨ Z > 90

theorem largest_angle_of_obtuse_isosceles_triangle
  (X Y Z : ℝ)
  (h1 : is_triangle X Y Z)
  (h2 : is_isosceles_triangle X Y)
  (h3 : X = 30)
  (h4 : is_obtuse_triangle X Y Z) :
  Z = 120 :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l93_93903


namespace minimum_balls_same_color_minimum_balls_two_white_l93_93423

-- Define the number of black and white balls.
def num_black_balls : Nat := 100
def num_white_balls : Nat := 100

-- Problem 1: Ensure at least 2 balls of the same color.
theorem minimum_balls_same_color (n_black n_white : Nat) (h_black : n_black = num_black_balls) (h_white : n_white = num_white_balls) : 
  3 ≥ 2 :=
by
  sorry

-- Problem 2: Ensure at least 2 white balls.
theorem minimum_balls_two_white (n_black n_white : Nat) (h_black: n_black = num_black_balls) (h_white: n_white = num_white_balls) :
  102 ≥ 2 :=
by
  sorry

end minimum_balls_same_color_minimum_balls_two_white_l93_93423


namespace complex_modulus_squared_l93_93346

theorem complex_modulus_squared (w : ℂ) (h : |w| = 15) : w * conj w = 225 :=
by
  sorry

end complex_modulus_squared_l93_93346


namespace shaded_area_l93_93389

theorem shaded_area 
  (side_of_square : ℝ)
  (arc_radius : ℝ)
  (side_length_eq_sqrt_two : side_of_square = Real.sqrt 2)
  (radius_eq_one : arc_radius = 1) :
  let square_area := 4
  let sector_area := 3 * Real.pi
  let shaded_area := square_area + sector_area
  shaded_area = 4 + 3 * Real.pi :=
by
  sorry

end shaded_area_l93_93389


namespace shaded_region_area_l93_93787

-- Definitions of geometric shapes involved
def radius_large_circle : ℝ := 6
def radius_small_circle : ℝ := 3
def area_of_shaded_region : ℝ := 18 + 9 * Real.pi

-- Lean statement for the proof problem
theorem shaded_region_area :
  let open_real_triangle := 4 * (1/2 * radius_small_circle * radius_small_circle)
  let open_real_sectors := (4 * 90 / 360) * Real.pi * (radius_small_circle ^ 2)
  let larger_circle_sectors := (1/4) * Real.pi * (radius_large_circle ^ 2)
  open_real_triangle + open_real_sectors - larger_circle_sectors = area_of_shaded_region :=
sorry -- Proof to be provided

end shaded_region_area_l93_93787


namespace count_whole_numbers_between_roots_l93_93753

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93753


namespace smallest_composite_no_prime_lt_20_l93_93057

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l93_93057


namespace cube_side_area_l93_93431

theorem cube_side_area (sum_edges : ℕ) (h1 : sum_edges = 132) : ∃ (area : ℕ), area = 121 :=
by
  -- Consider a cube where each edge has length l.
  let edge_length := 11
  -- Compute the area of one side of the cube.
  let area := edge_length * edge_length
  have h2 : area = 121 :=
    by
      unfold area
      rw edge_length
      calc
        11 * 11 = 121 : by norm_num
  exact ⟨area, h2⟩

end cube_side_area_l93_93431


namespace trigonometric_inequality_l93_93852

open Real Trigonometric

theorem trigonometric_inequality (a b : ℝ) (ha : 0 < a) (ha' : a < π / 2) (hb : 0 < b) (hb' : b < π / 2) :
  (sin a)^3 / (sin b) + (cos a)^3 / (cos b) ≥ 1 / (cos (a - b)) :=
by
  sorry

end trigonometric_inequality_l93_93852


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93747

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93747


namespace sequence_prime_difference_count_l93_93260

/-- 
  The arithmetic sequence in question is {7, 17, 27, 37, ..., 97}.
  Let S be the set of numbers in the sequence that can be expressed as the difference
  of two prime numbers.
  We need to show that the number of elements in S is 5.
--/
theorem sequence_prime_difference_count : 
  let seq := {x : ℕ | ∃ n : ℕ, x = 7 + 10 * n ∧ x ≤ 100} in
  let is_prime (n : ℕ) := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n in
  let is_prime_diff (a : ℕ) := ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ a = p - q in
  {x : ℕ | x ∈ seq ∧ is_prime_diff x}.card = 5 :=
by
  sorry

end sequence_prime_difference_count_l93_93260


namespace half_angle_quadrant_l93_93266

theorem half_angle_quadrant
  (α : ℝ) (k : ℤ)
  (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
by
  sorry

end half_angle_quadrant_l93_93266


namespace students_apply_colleges_l93_93953

    -- Define that there are 5 students
    def students : Nat := 5

    -- Each student has 3 choices of colleges
    def choices_per_student : Nat := 3

    -- The number of different ways the students can apply
    def number_of_ways : Nat := choices_per_student ^ students

    theorem students_apply_colleges : number_of_ways = 3 ^ 5 :=
    by
        -- Proof will be done here
        sorry
    
end students_apply_colleges_l93_93953


namespace geometric_sequence_properties_l93_93615

noncomputable def geometric_sequence_common_ratio_and_formula :
  { q : ℝ // q = 1/2 } × (ℕ → ℝ) :=
sorry

theorem geometric_sequence_properties (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : ∀ n m, n < m → a n > a m)
  (h2 : a 2 + a 3 + a 4 = 28) (h3 : a 3 + 2 = (a 2 + a 4) / 2) :
  (geometric_sequence_common_ratio_and_formula =
  ⟨⟨q, rfl⟩, λ n, (1/2)^(n-6)⟩) :=
by
  sorry

end geometric_sequence_properties_l93_93615


namespace parallel_chords_l93_93444

noncomputable def points_on_circles (A B C D C' D' : Type*) [incidence_geometry A B C D] :
    Prop :=
  let circle1 := circle A C B D in
  let circle2 := circle A C' B D' in
  circle1 ∩ circle2 = {A, B}

theorem parallel_chords 
  (circle1 circle2 : Type*) 
  [incidence_geometry circle1] [incidence_geometry circle2]
  (A B C D C' D' : incidence_geometry.Point circle1 circle2) 
  (h1 : A ∈ circle1) (h2 : B ∈ circle1) 
  (h3 : C ∈ circle1) (h4 : D ∈ circle1) 
  (h5 : A ∈ circle2) (h6 : B ∈ circle2) 
  (h7 : C' ∈ circle2) (h8 : D' ∈ circle2)
  (h9 : distinct A B)
  (h10 : distinct C D)
  (h11 : distinct C' D') :
  parallel (segment C D) (segment C' D') :=
sorry

end parallel_chords_l93_93444


namespace relationship_among_abc_l93_93611

noncomputable def a : ℝ := Real.logBase 0.3 4
noncomputable def b : ℝ := Real.logBase 4 3
noncomputable def c : ℝ := 0.3 ^ (-2)

theorem relationship_among_abc : a < b ∧ b < c :=
by 
  sorry

end relationship_among_abc_l93_93611


namespace radishes_in_first_basket_l93_93546

theorem radishes_in_first_basket :
  ∃ x : ℕ, ∃ y : ℕ, x + y = 88 ∧ y = x + 14 ∧ x = 37 :=
by
  -- Proof goes here
  sorry

end radishes_in_first_basket_l93_93546


namespace smallest_composite_no_prime_factors_lt_20_l93_93150

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93150


namespace Angle_CNB_20_l93_93899

theorem Angle_CNB_20 :
  ∀ (A B C N : Type) 
    (AC BC : Prop) 
    (angle_ACB : ℕ)
    (angle_NAC : ℕ)
    (angle_NCA : ℕ), 
    (AC ↔ BC) →
    angle_ACB = 98 →
    angle_NAC = 15 →
    angle_NCA = 21 →
    ∃ angle_CNB, angle_CNB = 20 :=
by
  sorry

end Angle_CNB_20_l93_93899


namespace count_whole_numbers_between_cubes_l93_93674

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93674


namespace circle_possible_values_l93_93912

theorem circle_possible_values (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 + a - 1 = 0 → -2 < a ∧ a < 2/3) := sorry

end circle_possible_values_l93_93912


namespace permutation_sums_consecutive_l93_93166

theorem permutation_sums_consecutive : 
  ∃ (n : ℕ), ∀ (T : Fin 2000 → Fin 2000), 
    (bijective T) → 
    ∃ (a b : ℕ), (a ≤ b) ∧ (b - a = 299) ∧ ∀ k, a ≤ k ∧ k ≤ b → (∃ i, k = ∑ j in range 2000, (T j + 1) / (j + 1) : ℕ) := sorry

end permutation_sums_consecutive_l93_93166


namespace quadrilateral_area_l93_93616

noncomputable theory

variables {a b : ℝ}

-- Conditions: the side lengths of the parallelogram with area 1 and side lengths a and b, with 0 < a < b < 2a.
-- Question: Find the area of the quadrilateral formed by the internal angle bisectors
theorem quadrilateral_area (h_area : a * b * real.sin (real.pi - 2 * real.arccos (a / b)) = 1) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2 * a) :
  let area := (b - a) ^ 2 / (2 * a * b) in 
  area = (b - a) ^ 2 / (2 * a * b) :=
by sorry

end quadrilateral_area_l93_93616


namespace angle_PRQ_eq_90_l93_93901

/-
Triangle PQR has PQ = 3 * PR. 
Let S and T be on PQ and QR respectively, such that ∠PQS = ∠PRT.
Let U be the intersection of segments QS and RT, 
and suppose that ΔPUT is equilateral. 
We need to prove ∠PRQ = 90°.
-/

variables {P Q R S T U : Type*}
variables [inner_product_space ℝ (euclidean_space 3)]

-- Let P, Q, R be vertices of the triangle.
def PQ : ℝ := 3 * PR
def angle_PQS : real.angle := angle_PRT

-- Given conditions and properties.
def intersection_QS_RT : euclidean_space 3 := U
def equilateral_PUT : bool := true

-- We need to prove ∠ PRQ.
theorem angle_PRQ_eq_90 (h1 : PQ = 3 * PR)
  (h2 : angle_PQS = angle_PRT)
  (h3 : U ∈ line_segment QS ∩ line_segment RT)
  (h4 : equilateral_PUT = true) : 
  angle_PRQ = real.pi / 2 := 
by sorry

end angle_PRQ_eq_90_l93_93901


namespace smallest_composite_no_prime_factors_less_than_20_l93_93019

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93019


namespace clock_hands_right_angle_13_times_l93_93408

noncomputable def hour_hand_speed : ℝ := 1 / 2
noncomputable def minute_hand_speed : ℝ := 6

/--
Prove that the number of times the hour hand and minute hand of a large clock on campus form a right angle between 8:30 AM and 3:30 PM is 13 times.
-/
theorem clock_hands_right_angle_13_times : 
  (∃ (t : set ℝ) (h : ∀ x ∈ t, (x ≥ 8.5 ∧ x ≤ 15.5) ∧ δ x = 90) , ∀ x ∈ t, x = 13) := 
sorry

end clock_hands_right_angle_13_times_l93_93408


namespace smallest_composite_no_prime_factors_less_than_20_l93_93139

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93139


namespace whole_numbers_count_between_cubic_roots_l93_93721

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93721


namespace parallel_lines_area_of_triangle_ACO_l93_93309

theorem parallel_lines 
  (ABCD_square : ∃ A B C D : Point, isSquare A B C D)
  (tangent_circle : ∃ O : Point, isTangentToCircle O D C ∧ isTangentToCircle O D E) :
  let L1 := lineThroughPoints A C,
      L2 := lineThroughPoints D O in
  is_parallel L1 L2 :=
sorry

theorem area_of_triangle_ACO 
  (ABCD_square : ∃ A B C D : Point, isSquare A B C D)
  (square_area : area ABCD = 36) :
  let ACO_triangle := triangle A C O in
  area ACO_triangle = 18 :=
sorry

end parallel_lines_area_of_triangle_ACO_l93_93309


namespace part1_part2_l93_93234

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_part2_l93_93234


namespace correct_operation_l93_93929

def condition_A (x : ℝ) : Prop := (x^2 + x^2 = x^4)
def condition_B (x : ℝ) : Prop := (1 / x^(-2) = x^2)
def condition_C (a b: ℝ) : Prop := (2 * a * b - a * b = 2)
def condition_D (x : ℝ) : Prop := ((x ^ 3) ^ 3 = x ^ 6)

theorem correct_operation (x a b : ℝ) : condition_B x := by
  sorry

end correct_operation_l93_93929


namespace sum_of_missing_endpoint_coords_proof_expected_sum_l93_93368

-- Define the coordinates of the known endpoint and the midpoint
def endpoint1 : ℝ × ℝ := (6, 2)
def midpoint : ℝ × ℝ := (3, 7)

-- Define the missing endpoint
def missing_endpoint : ℝ × ℝ :=
  let (x1, y1) := endpoint1
  let (mx, my) := midpoint
  let x2 := 2 * mx - x1
  let y2 := 2 * my - y1
  (x2, y2)

-- Check if the sum of the coordinates of the missing endpoint is 12
theorem sum_of_missing_endpoint_coords : missing_endpoint.1 + missing_endpoint.2 = 12 :=
by
  let (x1, y1) := endpoint1
  let (mx, my) := midpoint
  let x2 := 2 * mx - x1
  let y2 := 2 * my - y1
  have h1 : x2 = 0 := by
    rw [x2]
    simp
  have h2 : y2 = 12 := by
    rw [y2]
    simp
  rw [missing_endpoint]
  simp [h1, h2]

-- Provide the main theorem 
theorem proof_expected_sum (x1 y1 mx my x2 y2 : ℝ) 
  (h_endpoint : (endpoint1 = (x1, y1))) (h_midpoint : (midpoint = (mx, my))) 
  (h_missing_endpoint : (missing_endpoint = (x2, y2))): missing_endpoint.1 + missing_endpoint.2 = 12 :=
sum_of_missing_endpoint_coords

#reduce proof_expected_sum -- To check the theorem

end sum_of_missing_endpoint_coords_proof_expected_sum_l93_93368


namespace coefficient_x7_in_expansion_l93_93913

theorem coefficient_x7_in_expansion : 
  ∀ (x : ℝ), 
  (coeff_of_x7 (*coeff of \( x^7\) in expansion of*) (expand ((x^2 / 2) - (2 / x)) ^ 8)) = -14 := 
by 
  sorry

end coefficient_x7_in_expansion_l93_93913


namespace num_whole_numbers_between_l93_93697

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93697


namespace sum_of_cubes_correct_l93_93411

noncomputable def sum_of_cubes : ℝ := 
  let a := Real.sqrt 37 in 
  let x := a - 1 in 
  let y := a in 
  let z := a + 1 in 
  x^3 + y^3 + z^3 

theorem sum_of_cubes_correct :
  (∀ (n : ℝ), (n - 1) * n * (n + 1) = 12 * ((n - 1) + n + (n + 1)) → 
  sum_of_cubes = 114 * (Real.sqrt 37)) :=
  by sorry

end sum_of_cubes_correct_l93_93411


namespace smallest_positive_x_satisfying_condition_l93_93162

open Real

noncomputable def smallest_x : ℝ := 369 / 19

theorem smallest_positive_x_satisfying_condition :
  ∃ x ∈ Ioi 0, (∀ y ∈ Ioi 0, (⟦y^3⟧.to_real - y * ⟦y⟧.to_real = 18 → y ≥ x)) ∧ 
                (⟦x^3⟧.to_real - x * ⟦x⟧.to_real = 18) :=
by 
  sorry

end smallest_positive_x_satisfying_condition_l93_93162


namespace find_number_l93_93498

theorem find_number (x : ℝ) : (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 :=
  by 
  sorry

end find_number_l93_93498


namespace NewYearSeasonMarkup_theorem_l93_93992

def NewYearSeasonMarkup (C N : ℝ) : Prop :=
    (0.90 * (1.20 * C * (1 + N)) = 1.35 * C) -> N = 0.25

theorem NewYearSeasonMarkup_theorem (C : ℝ) (h₀ : C > 0) : ∃ (N : ℝ), NewYearSeasonMarkup C N :=
by
  use 0.25
  sorry

end NewYearSeasonMarkup_theorem_l93_93992


namespace hotel_charge_difference_l93_93873

variables (G P R : ℝ)

-- Assumptions based on the problem conditions
variables
  (hR : R = 2 * G) -- Charge for a single room at hotel R is 100% greater than at hotel G
  (hP : P = 0.9 * G) -- Charge for a single room at hotel P is 10% less than at hotel G

theorem hotel_charge_difference :
  ((R - P) / R) * 100 = 55 :=
by
  -- Calculation
  sorry

end hotel_charge_difference_l93_93873


namespace smallest_composite_no_prime_factors_less_than_20_l93_93112

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93112


namespace smallest_composite_no_prime_factors_less_20_l93_93039

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93039


namespace square_segment_midpoint_area_l93_93997

theorem square_segment_midpoint_area :
  let a := 3 / 2
  let quarter_circle_area := (1 / 4) * Real.pi * a^2
  let total_area := 4 * quarter_circle_area
  let square_area := 3^2
  let m := square_area - total_area
  100 * Real.floor (m * 100) = 178 := by
  let a := 3 / 2
  let quarter_circle_area := (1 / 4) * Real.pi * a^2
  let total_area := 4 * quarter_circle_area
  let square_area := 3^2
  let m := square_area - total_area
  have h : (m * 100).floor = 178 by sorry
  exact (100 * (m * 100).floor).symm.trans h

end square_segment_midpoint_area_l93_93997


namespace whole_numbers_between_cuberoots_l93_93687

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93687


namespace median_to_hypotenuse_in_right_triangle_l93_93303

-- Define the given conditions
def AB : ℝ := 10
def AC : ℝ := 6
def BC : ℝ := 8
def midpoint_AB := (10 : ℝ) / 2

-- Define the distance from C to the midpoint of AB
def distance_C_to_midpoint_AB : ℝ := 5

-- State the theorem
theorem median_to_hypotenuse_in_right_triangle 
  (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ c = 10 ∧ a = 6 ∧ b = 8) :
  distance_C_to_midpoint_AB = midpoint_AB :=
by
  sorry

end median_to_hypotenuse_in_right_triangle_l93_93303


namespace correct_calculation_result_l93_93502

theorem correct_calculation_result (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end correct_calculation_result_l93_93502


namespace whole_numbers_between_l93_93725

theorem whole_numbers_between (n : ℕ) : 
    (∑ n in {k | k ∈ Finset.range (8) \ Finset.range (4)}, 1 = 4) :=
by sorry

end whole_numbers_between_l93_93725


namespace house_height_54_l93_93916

noncomputable def height_of_house (shadow_house shadow_tree height_tree : ℕ) : ℕ :=
  ((shadow_house : ℝ) / shadow_tree) * height_tree |> Float.ceil

theorem house_height_54 :
  height_of_house 72 40 30 = 54 := by
  sorry

end house_height_54_l93_93916


namespace fraction_along_diag_AC_l93_93521

-- Define the geometric properties of the kite
structure Kite (A B C D : Type) :=
  (AB BC : ℝ)  -- Sides AB and BC both are 100 meters
  (is_congruent_AB_DC : AC ≃ DC)  -- AB and DC are congruent
  (is_congruent_AD_BC : AD ≃ BC)  -- AD and BC are congruent
  (angle_B_D_120 : ∠B == 120°)
  (angle_A_C_60 : ∠A == 60°)
  (diagonal_AC : ℝ)  -- AC value determined during calculation
  (diagonal_BD : ℝ)  -- BD value determined during calculation
  (area_total : ℝ)  -- Total area of the kite
  (area_along_AC : ℝ)  -- Area serving the longest diagonal AC

-- The theorem statement representing the proof problem
theorem fraction_along_diag_AC {K : Kite A B C D} :
  K.area_along_AC / K.area_total = 7071 / 10000 := by
  sorry

end fraction_along_diag_AC_l93_93521


namespace Misha_can_place_rooks_l93_93358

theorem Misha_can_place_rooks (n : ℕ) (h : n = 100) : 
  ∃ (b : fin n → fin n → bool), 
    (∑ i j, if b i j then 1 else 0) = n ∧ 
    (∀ i j k, b i j = tt → b k j = tt → k = i) ∧ 
    (∀ i j k, b i j = tt → b i k = tt → j = k) := sorry

end Misha_can_place_rooks_l93_93358


namespace expected_difference_is_correct_l93_93531

noncomputable def expected_days_jogging : ℚ :=
  4 / 7 * 365

noncomputable def expected_days_yoga : ℚ :=
  3 / 7 * 365

noncomputable def expected_difference_days : ℚ :=
  expected_days_jogging - expected_days_yoga

theorem expected_difference_is_correct :
  expected_difference_days ≈ 52.71 := 
sorry

end expected_difference_is_correct_l93_93531


namespace area_difference_l93_93288

-- Definitions of the given conditions
structure Triangle :=
(base : ℝ)
(height : ℝ)

def area (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

-- Conditions of the problem
def EFG : Triangle := {base := 8, height := 4}
def EFG' : Triangle := {base := 4, height := 2}

-- Proof statement
theorem area_difference :
  area EFG - area EFG' = 12 :=
by
  sorry

end area_difference_l93_93288


namespace whole_numbers_between_cuberoot50_and_cuberoot500_l93_93751

theorem whole_numbers_between_cuberoot50_and_cuberoot500 :
  ∃ n : ℕ, (∃ n₁ n₂ n₃ n₄ : ℕ, n₁ = 4 ∧ n₂ = 5 ∧ n₃ = 6 ∧ n₄ = 7 ∧ 
    ((n₁ > real.cbrt 50) ∧ (n₁ < real.cbrt 500) ∧
     (n₂ > real.cbrt 50) ∧ (n₂ < real.cbrt 500) ∧
     (n₃ > real.cbrt 50) ∧ (n₃ < real.cbrt 500) ∧
     (n₄ > real.cbrt 50) ∧ (n₄ < real.cbrt 500))) ∧
  (∃ m: ℕ, m = 4) := 
sorry

end whole_numbers_between_cuberoot50_and_cuberoot500_l93_93751


namespace part1_part2_l93_93232

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_part2_l93_93232


namespace remainder_when_3y_divided_by_9_l93_93467

theorem remainder_when_3y_divided_by_9 (y : ℕ) (k : ℕ) (hy : y = 9 * k + 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_when_3y_divided_by_9_l93_93467


namespace whole_numbers_count_between_cubic_roots_l93_93722

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l93_93722


namespace count_whole_numbers_between_roots_l93_93757

theorem count_whole_numbers_between_roots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  (3 < a ∧ a < 4) →
  (7 < b ∧ b < 8) →
  ∃ n : ℕ, n = 4 :=
by
  intros ha hb
  sorry

end count_whole_numbers_between_roots_l93_93757


namespace part_a_part_b_l93_93969

variable (p : ℝ)
variable (h_pos : 0 < p)
variable (h_prob : p ≤ 1)

theorem part_a :
  let q := 1 - p in
  ∃ f : ℕ → ℝ, f 5 = 6 * p^3 * q^2 :=
  by
    sorry

theorem part_b :
  ∃ f : ℕ → ℝ, f 3 = 3 / p :=
  by
    sorry

end part_a_part_b_l93_93969


namespace max_xy_value_l93_93766

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 200) : x * y ≤ 348 :=
begin
  sorry
end

example : ∃ x y : ℕ, 7 * x + 4 * y = 200 ∧ x * y = 348 :=
begin
  use [12, 29], -- These are the integers that satisfy the conditions and provide the maximum value of xy.
  split,
  { exact eq.refl 200, },
  { exact eq.refl 348, },
end

end max_xy_value_l93_93766


namespace plan1_cost_effective_l93_93515

/-- Define the conditions and costs involved in Plan 1 and Plan 2 -/
def cost_plan1 (x a : ℝ) : ℝ := 7 * a * (x / 4 + 36 / x - 1)

def cost_plan2 (x a : ℝ) : ℝ := 2 * a * (x + 126 / x) - 21 * a / 2

/-- Prove that Plan 1 is more cost-effective than Plan 2 for given range of x. -/
theorem plan1_cost_effective (a : ℝ) (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < 14) (h3 : x2 ≥ 14) :
  cost_plan1 x1 a < cost_plan2 x2 a :=
by {
  sorry,
}

end plan1_cost_effective_l93_93515


namespace part_a_part_b_l93_93970

variable (p : ℝ)
variable (h_pos : 0 < p)
variable (h_prob : p ≤ 1)

theorem part_a :
  let q := 1 - p in
  ∃ f : ℕ → ℝ, f 5 = 6 * p^3 * q^2 :=
  by
    sorry

theorem part_b :
  ∃ f : ℕ → ℝ, f 3 = 3 / p :=
  by
    sorry

end part_a_part_b_l93_93970


namespace distance_between_intersections_l93_93795

noncomputable def line_parametric_eqn (t : ℝ) : ℝ × ℝ :=
  (2 + t, t + 1)

noncomputable def curve_polar_eqn (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ + 3 = 0

noncomputable def line_general_eqn (x y : ℝ) : Prop :=
  x - y - 1 = 0

noncomputable def curve_rect_eqn (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 3 = 0

theorem distance_between_intersections : 
  ∃ A B : ℝ × ℝ, 
  line_parametric_eqn (some t) = A ∧ 
  line_parametric_eqn (some s) = B ∧ 
  curve_rect_eqn A.1 A.2 ∧ 
  curve_rect_eqn B.1 B.2 ∧ 
  dist A B = Real.sqrt 2 :=
sorry

end distance_between_intersections_l93_93795


namespace order_of_a_b_c_d_l93_93268

noncomputable def a : ℝ := Real.sin 4
noncomputable def b : ℝ := Real.log 3 / Real.log 5
noncomputable def c : ℝ := Real.log10 6
noncomputable def d : ℝ := 1 / Real.log10 15

theorem order_of_a_b_c_d : a < b ∧ b < c ∧ c < d := by
  sorry

end order_of_a_b_c_d_l93_93268


namespace num_whole_numbers_between_l93_93696

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93696


namespace relationship_of_abc_l93_93632

theorem relationship_of_abc (a b c : ℝ) 
  (h1 : b + c = 6 - 4 * a + 3 * a^2) 
  (h2 : c - b = 4 - 4 * a + a^2) : 
  a < b ∧ b ≤ c := 
sorry

end relationship_of_abc_l93_93632


namespace numberOfSophistsCorrect_l93_93850

-- Define the inhabitants types
inductive Inhabitant : Type
| Knight
| Liar
| Sophist

-- The problem data
def numberOfKnights : ℕ := 40
def numberOfLiars : ℕ := 25
def numberOfSophists : ℕ
variable (numberOfSophists : ℕ)

-- The sofist's statements as conditions
def sophistStatement1 (numberOfLiars : ℕ) : Prop :=
  numberOfLiars = 26

def sophistStatement2 (numberOfSophists numberOfLiars : ℕ) : Prop :=
  numberOfSophists <= numberOfLiars

-- The main theorem to prove
theorem numberOfSophistsCorrect :
  numberOfLiars = 25 →
  sophistStatement1 25 →
  sophistStatement2 numberOfSophists 25 →
  numberOfSophists = 27 :=
by sorry

end numberOfSophistsCorrect_l93_93850


namespace total_volume_of_five_cubes_l93_93923

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end total_volume_of_five_cubes_l93_93923


namespace smallest_composite_no_prime_factors_less_than_20_l93_93029

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93029


namespace car_mileage_before_modification_l93_93506

theorem car_mileage_before_modification (miles_per_gallon_before : ℝ) 
  (fuel_efficiency_modifier : ℝ := 0.75) (tank_capacity : ℝ := 12) 
  (extra_miles_after_modification : ℝ := 96) :
  (1 / fuel_efficiency_modifier) * miles_per_gallon_before * (tank_capacity - 1) = 24 :=
by
  sorry

end car_mileage_before_modification_l93_93506


namespace collinear_points_on_same_circle_l93_93340

-- Given a triangle with vertices A, B, and C
variables {A B C : Point}

-- Definitions of special points with specified conditions
variables {A1 A2 B1 B2 C1 C2 : Point}
variable (hA1A2 : dist A A1 = dist A A2 = dist B C)
variable (hA1_on_AB : lies_on A1 (segment A B))
variable (hA2_on_AC : lies_on A2 (segment A C))
variable (hB1B2 : dist B B1 = dist B B2 = dist A C)
variable (hB1_on_BA : lies_on B1 (segment B A))
variable (hB2_on_BC : lies_on B2 (segment B C))
variable (hC1C2 : dist C C1 = dist C C2 = dist A B)
variable (hC1_on_CA : lies_on C1 (segment C A))
variable (hC2_on_CB : lies_on C2 (segment C B))

-- Statement to show that these points lie on the same circle
theorem collinear_points_on_same_circle :
  cyclic (A1 A2 B1 B2 C1 C2) :=
sorry

end collinear_points_on_same_circle_l93_93340


namespace pigeonhole_principle_example_l93_93417

theorem pigeonhole_principle_example :
  ∀ (S : Finset ℕ), (∀ (x : ℕ), x ∈ S ↔ x ≤100 ∧ 1≤ x) →
  ∃ (A : Finset ℕ), (A.card ≥ 15 ∧ A ⊆ S) → 
  (∃ a b c d ∈ A, a ≠ b ∧ c ≠ d ∧ a + b = c + d) ∨ 
  (∃ e f g ∈ A, e ≠ f ∧ f ≠ g ∧ e + f = 2 * g) :=
by
  -- implementation of the proof goes here
  sorry

end pigeonhole_principle_example_l93_93417


namespace amount_kept_by_Tim_l93_93441

-- Define the conditions
def totalAmount : ℝ := 100
def percentageGivenAway : ℝ := 0.2

-- Prove the question == answer
theorem amount_kept_by_Tim : totalAmount - totalAmount * percentageGivenAway = 80 :=
by
  -- Here the proof would take place
  sorry

end amount_kept_by_Tim_l93_93441


namespace product_first_11_terms_eq_2_pow_11_l93_93614

variable (a : ℕ → ℝ) (r : ℝ)
variable (is_geom_seq : ∀ n, a (n + 1) = a n * r)
variable (a5a6a7_eq : a 5 * a 6 * a 7 = 8)

theorem product_first_11_terms_eq_2_pow_11
  (h_geom : is_geom_seq)
  (h_cond : a5a6a7_eq) : 
  ∏ i in Finset.range 11, a i = 2^11 :=
sorry

end product_first_11_terms_eq_2_pow_11_l93_93614


namespace converse_not_true_without_negatives_l93_93545

theorem converse_not_true_without_negatives (a b c d : ℕ) (h : a + d = b + c) : ¬(a - c = b - d) :=
by
  sorry

end converse_not_true_without_negatives_l93_93545


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93068

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93068


namespace border_material_needed_l93_93874

-- Conditions
def area (r : ℝ) : ℝ := (22 / 7) * r^2
def circumference (r : ℝ) : ℝ := 2 * (22 / 7) * r
def extra_border (C : ℝ) : ℝ := C + 3

-- Proof problem statement
theorem border_material_needed 
  (r : ℝ)
  (h1 : area r = 100)
  (h2 : circumference r + 3 = 38.4) :
  extra_border (circumference r) = 38.4 :=
sorry

end border_material_needed_l93_93874


namespace minimize_circumradius_l93_93192

variables {A B C : Type} [EuclideanGeometry A]
variables {M : A}
variables {AB : Line A}
variables (hM : M ∈ AB)
variables (hABC : Triangle A)
variables (ACM BCM : Triangle A)
variables (circumradius : Triangle A → ℝ)
variables (sum_circumradii : ∀ M, circumradius ACM + circumradius BCM)
variables (min_circumradii : ∀ M, sum_circumradii M)

theorem minimize_circumradius :
  ∃ M : A, M ∈ AB ∧ CM ⟂ AB ∧
  ∀ M' : A, M' ∈ AB → (circumradius (triangle.mk A C M) + circumradius (triangle.mk B C M)) ≥ (circumradius (triangle.mk A C M') + circumradius (triangle.mk B C M')) :=
by {
  sorry
}

end minimize_circumradius_l93_93192


namespace closest_integer_to_sqrt_35_is_6_l93_93366

theorem closest_integer_to_sqrt_35_is_6 :
  (∀ x : ℝ, 5 < x → x < 6 → (ceil x = 6)) :=
begin
  intro x,
  intros h5 h6,
  suffices : x ≤ 5.5,
  { exact real.ceil_eq_of_le this },
  linarith,
end

end closest_integer_to_sqrt_35_is_6_l93_93366


namespace fourth_term_of_geometric_sequence_l93_93879

theorem fourth_term_of_geometric_sequence (x r : ℝ) (h1 : 3 * x + 3 = r * x) (h2 : 6 * x + 6 = r * (3 * x + 3))
  (h3 : r = 2) (h4 : x = -3) : r * (r * (6 * x + 6)) = -24 :=
by
  rw [h4, h3] at *
  simp at *
  exact rfl

end fourth_term_of_geometric_sequence_l93_93879


namespace parabola_focus_directrix_eq_l93_93395

open Real

def distance (p : ℝ × ℝ) (l : ℝ) : ℝ := abs (p.fst - l)

def parabola_eq (focus_x focus_y l : ℝ) : Prop :=
  ∀ x y, (distance (x, y) focus_x = distance (x, y) l) ↔ y^2 = 2 * x - 1

theorem parabola_focus_directrix_eq :
  parabola_eq 1 0 0 :=
by
  sorry

end parabola_focus_directrix_eq_l93_93395


namespace find_incorrect_statement_l93_93927

-- Definitions of conditions
def is_rhombus (p : Type) [geometry p] : Prop :=
  parallelogram p ∧ perpendicular_diagonals p ∧ equal_sides p

def is_rectangle (p : Type) [geometry p] : Prop :=
  parallelogram p ∧ equal_diagonals p ∧ right_angles p

def is_square (q : Type) [geometry q] : Prop :=
  perpendicular_diagonals q ∧ bisecting_diagonals q ∧ right_angles q

def is_parallelogram (q : Type) [geometry q] : Prop :=
  parallel_sides q ∧ equal_sides q

-- The theorem statement based on the given problem
theorem find_incorrect_statement :
  (A : Type) [is_rhombus A] →
  (B : Type) [is_rectangle B] →
  (C : Type) [is_square C] →
  (D : Type) [is_parallelogram D] →
  ¬ is_square C :=
by
  intros _ _
  exact sorry

end find_incorrect_statement_l93_93927


namespace solve_problem_l93_93276

-- Define variables
variables {M X Y : ℕ}

-- Define the condition for distinct digits
def distinct_digits (M X Y : ℕ) : Prop :=
  M ≠ X ∧ X ≠ Y ∧ M ≠ Y

-- Define the given equation
def given_equation (M X Y : ℕ) (n : ℕ) : Prop :=
  let A := 10^(n+1) in
  let B := (10^n + 10^(n-1) + ... + 10) in
  M * A + X * B + Y = X * A + M * B + Y

-- Define the possible answers
def possible_answers (M X Y : ℕ) : Prop :=
  (M = 1 ∧ X = 3 ∧ Y = 5) ∨
  (M = 3 ∧ X = 1 ∧ Y = 5) ∨
  (M = 0 ∧ X = 4 ∧ Y = 5) ∨
  (M = 4 ∧ X = 0 ∧ Y = 5)

-- The theorem statement
theorem solve_problem (h1 : distinct_digits M X Y)
  (h2 : ∀ n ≥ 1, given_equation M X Y n) :
  possible_answers M X Y :=
begin
  sorry
end

end solve_problem_l93_93276


namespace parabola_line_intersection_ratio_l93_93245

theorem parabola_line_intersection_ratio
  (p : ℝ) (hp : 0 < p)
  (A B F : ℝ × ℝ)
  (hA : A.2 ^ 2 = 2 * p * A.1)
  (hB : B.2 ^ 2 = 2 * p * B.1)
  (hF : F = (p / 2, 0))
  (hl : ∃ k : ℝ, k = sqrt 3 ∧ ∀ x y, y = k * (x - p / 2) → (x, y) = A ∨ (x, y) = B)
  (hQuads : ∃ (x1 x2 : ℝ), x1 = 3 / 2 * p ∧ x2 = 1 / 6 * p):
  (dist A F / dist B F) = 3 := sorry

end parabola_line_intersection_ratio_l93_93245


namespace distinct_solutions_subtraction_l93_93338

theorem distinct_solutions_subtraction : 
  ∀ (p q : ℝ), (p ≠ q) ∧ (∀ x : ℝ, (4 * x - 12) / (x ^ 2 + 2 * x - 15) = x + 2 ↔ x = p ∨ x = q) ∧ p > q → p - q = 5 := 
by
  intro p q h
  have h1 : (4 * p - 12) / (p ^ 2 + 2 * p - 15) = p + 2 := (h.2 p).mpr (Or.inl rfl)
  have h2 : (4 * q - 12) / (q ^ 2 + 2 * q - 15) = q + 2 := (h.2 q).mpr (Or.inr rfl)
  sorry

end distinct_solutions_subtraction_l93_93338


namespace smallest_composite_no_prime_factors_less_than_20_l93_93104

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93104


namespace clock_hands_angle_at_720_l93_93548

noncomputable def hour_hand_angle (h m : ℕ) : ℝ :=
  let base_angle := (h % 12) * 30
  in base_angle + (m / 60.0) * 30

noncomputable def minute_hand_angle (m : ℕ) : ℝ :=
  (m % 60) * 6

theorem clock_hands_angle_at_720 : abs (hour_hand_angle 7 20 - minute_hand_angle 20) = 100 :=
by
  sorry

end clock_hands_angle_at_720_l93_93548


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93084

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93084


namespace num_whole_numbers_between_l93_93694

noncomputable def num_whole_numbers_between_cube_roots : ℕ :=
  let lower_bound := real.cbrt 50
  let upper_bound := real.cbrt 500
  set.Ico (floor lower_bound + 1) (ceil upper_bound)

theorem num_whole_numbers_between :
  set.size (num_whole_numbers_between_cube_roots) = 4 :=
sorry

end num_whole_numbers_between_l93_93694


namespace true_propositions_l93_93646

theorem true_propositions :
  (¬ (∀ a b : ℝ, a > b → a^2 > b^2) ∧
   ¬ (∀ a b : ℝ, log a = log b → a = b) ∧
   (∀ x y : ℝ, abs x = abs y ↔ x^2 = y^2) ∧
   (∀ A B C : Type, ∀ (α β : A), (∀ x y : A, sin (x : ℝ) > sin (y : ℝ) ↔ x > y))) :=
by sorry

end true_propositions_l93_93646


namespace incorrect_statement_is_A_l93_93649

noncomputable def f (x : ℝ) : ℝ := x * abs x

theorem incorrect_statement_is_A :
  (∃ (A : Prop), A =
    ¬(∀ x, (f (Real.sin x)) = -f x) ∧
    (Real.sin ∘ f) is_strictly_increasing_on Icc (-1/2 : ℝ) (1/2 : ℝ) ∧
    (f ∘ Real.cos) is_strictly_decreasing_on Ioo 0 1 ∧
    (Real.cos ∘ f) is_strictly_increasing_on Ioo (-1) 0) := sorry

end incorrect_statement_is_A_l93_93649


namespace smallest_composite_no_prime_factors_less_than_20_l93_93030

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l93_93030


namespace number_of_solutions_l93_93668

noncomputable def number_of_nonzero_complex_numbers (z : ℂ) : ℕ :=
  if z ≠ 0 ∧ 0 ∈ set_of_distinct_vertices_of_equilateral_triangle ({0, z, z^4}) then 4 else 0

theorem number_of_solutions : ∃ n : ℕ, n = 4 ∧ ∀ z : ℂ, z ≠ 0 ∧ 0 ∈ set_of_distinct_vertices_of_equilateral_triangle ({0, z, z^4}) → n = 4 :=
begin
  sorry
end

end number_of_solutions_l93_93668


namespace no_three_distinct_integers_P_cycle_l93_93838

-- Define the main statement about the existence of such integers.
theorem no_three_distinct_integers_P_cycle (P : ℤ[X]) :
  ∀ a b c : ℤ, (P.eval a = b ∧ P.eval b = c ∧ P.eval c = a) → 
  ¬ (a ≠ b ∧ b ≠ c ∧ c ≠ a) := 
sorry

end no_three_distinct_integers_P_cycle_l93_93838


namespace second_person_avg_pages_per_day_l93_93566

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end second_person_avg_pages_per_day_l93_93566


namespace smallest_composite_no_prime_factors_less_than_twenty_l93_93088

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l93_93088


namespace path_bound_l93_93558

/-- Definition of P_k: the number of non-intersecting paths of length k starting from point O on a grid 
    where each cell has side length 1. -/
def P_k (k : ℕ) : ℕ := sorry  -- This would normally be defined through some combinatorial method

/-- The main theorem stating the required proof statement. -/
theorem path_bound (k : ℕ) : (P_k k : ℝ) / (3^k : ℝ) < 2 := sorry

end path_bound_l93_93558


namespace find_angle_A_maximum_height_of_triangle_l93_93298

noncomputable def triangle_side_condition (a b c A B C : ℝ) : Prop :=
  a - b - c - A - B - C = 0 ∧ 
  sin C = 2 * cos A * sin (B + π / 3)

noncomputable def maximum_height_condition (a b c : ℝ) (A : ℝ) : Prop :=
  b + c = 6 ∧ cos A = 1 / 2 ∧ bc ≤ 9

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle_side_condition a b c A B C) : 
  A = π / 3 :=
sorry

theorem maximum_height_of_triangle (a b c : ℝ) (A : ℝ) 
  (h : maximum_height_condition a b c A) : 
  ∃ (D : ℝ), D = 3 * sqrt 3 / 2 :=
sorry

end find_angle_A_maximum_height_of_triangle_l93_93298


namespace smallest_composite_no_prime_factors_less_than_20_l93_93137

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93137


namespace simplify_polynomial_l93_93208

open Polynomial

def arithmetic_sequence (a_0 d : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a_0 + n * d

def p_n (a : ℕ → ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * (nat.choose n k) * (x ^ k) * ((1 - x) ^ (n - k))

theorem simplify_polynomial (a_0 d : ℝ) (a : ℕ → ℝ) (h : arithmetic_sequence a_0 d a)
  (n : ℕ) (x : ℝ) : p_n a x n = a_0 + n * d * x :=
by
  sorry

end simplify_polynomial_l93_93208


namespace false_proposition_C_l93_93625

-- Definitions for the given problem context
variables {l m : Line} {α β : Plane}

-- Propositions
def Proposition_A (α β : Plane) (l : Line) : Prop := (α ∥ β ∧ l ⊆ α) → l ∥ β
def Proposition_B (α β : Plane) (l : Line) : Prop := (α ∥ β ∧ l ⟂ α) → l ⟂ β
def Proposition_C (l : Line) (m : Line) (α : Plane) : Prop := (l ∥ α ∧ m ⊆ α) → l ∥ m
def Proposition_D (α β : Plane) (l : Line) (m : Line) : Prop := (α ⟂ β ∧ α ∩ β = l ∧ m ⊆ α ∧ m ⟂ l) → m ⟂ β

-- The objective is to show that Proposition C is false
theorem false_proposition_C (l m : Line) (α β : Plane) : ¬ Proposition_C l m α := sorry

end false_proposition_C_l93_93625


namespace hypotenuse_squared_sum_l93_93304

theorem hypotenuse_squared_sum (A B C : Type) [Add B] [Mul B] [HasPow B ℕ] (h : BC = 2) (AB_squared AC_squared BC_squared : B) :
  AB_squared + AC_squared = BC_squared → AB_squared + AC_squared + BC_squared = 8 :=
by
  sorry

end hypotenuse_squared_sum_l93_93304


namespace total_boys_in_both_rows_equal_l93_93293

/-- In a school assembly, two rows of boys are standing, facing the stage.
  In the first row, Rajan is sixth from the left end,
  and Vinay is tenth from the right end.
  There are 8 boys between Rajan and Vinay within the first row.
  In the second row, Arjun stands directly behind Rajan,
  while Suresh is fifth from the left end.
  The number of boys in both rows are equal.
  Prove that the total number of boys in both rows is 48. -/
theorem total_boys_in_both_rows_equal :
  ∃ (num_boys_first_row : ℕ), 
  let num_boys_second_row := num_boys_first_row in
  let total_boys := num_boys_first_row + num_boys_second_row in
  (num_boys_first_row = 5 + 1 + 8 + 1 + 9) → (total_boys = 48) :=
by
  -- Statement of the problem conditions as math formulas
  let num_boys_first_row := 24
  let num_boys_second_row := num_boys_first_row  -- Since they are equal
  let total_boys := num_boys_first_row + num_boys_second_row
  have h1 : num_boys_first_row = 5 + 1 + 8 + 1 + 9 := rfl
  have h2 : total_boys = 48 := by {
    rw [h1],
    exact add_self 24
  }
  exact ⟨num_boys_first_row, h2⟩

end total_boys_in_both_rows_equal_l93_93293


namespace algorithm_effective_expected_number_of_residents_l93_93484

-- Definitions required from the conditions of the original problem
def num_mailboxes : ℕ := 80

def key_distribution : Equiv.Perm (Fin num_mailboxes) := sorry

def initial_mailbox : Fin num_mailboxes := 37

-- Lean 4 statement for Part (a)
theorem algorithm_effective :
  ∃ m : Fin num_mailboxes, m = initial_mailbox → 
    (fix : ℕ → Fin num_mailboxes)
      (fix 0 = initial_mailbox)
      (∀ n, fix (n+1) = key_distribution (fix n))
      ∃ k, fix k = initial_mailbox := sorry

-- Lean 4 statement for Part (b)
theorem expected_number_of_residents :
  ∀ n, n = num_mailboxes → 
    let Harmonic := λ (n : ℕ), Σ i in Finset.range n, 1 / (i + 1)
    Harmonic n ≈ 4.968 := sorry

end algorithm_effective_expected_number_of_residents_l93_93484


namespace admission_price_for_adults_l93_93895

def total_people := 610
def num_adults := 350
def child_price := 1
def total_receipts := 960

theorem admission_price_for_adults (A : ℝ) (h1 : 350 * A + 260 = 960) : A = 2 :=
by {
  -- proof omitted
  sorry
}

end admission_price_for_adults_l93_93895


namespace second_person_average_pages_per_day_l93_93571

-- Define the given conditions.
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def deshaun_pages_per_book : ℕ := 320
def second_person_percentage : ℝ := 0.75

-- Calculate the total number of pages DeShaun read.
def deshaun_total_pages : ℕ := deshaun_books * deshaun_pages_per_book

-- Calculate the total number of pages the second person read.
def second_person_total_pages : ℕ := (second_person_percentage * deshaun_total_pages).toNat

-- Prove the average number of pages the second person read per day.
def average_pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

theorem second_person_average_pages_per_day :
  average_pages_per_day second_person_total_pages summer_days = 180 :=
by
  sorry

end second_person_average_pages_per_day_l93_93571


namespace range_of_m_l93_93221

theorem range_of_m (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) :
  ( ∀ x : ℝ, a^{2*x} + (1 + 1/m) * a^x + 1 = 0 → a^x > 0 ) ↔
  ( -1/3 ≤ m ∧ m < 0 ) :=
sorry

end range_of_m_l93_93221


namespace trisecting_lines_divide_areas_l93_93933

/-- Given a parallelogram ABCD with area 1, where lines trisect the angles at points 
    A and C, prove the areas into which the parallelogram is divided by these trisecting 
    lines. Specifically, four areas of 1/15 each, two areas of 4/15 each, and one area 
    of 1/5. -/

structure Parallelogram :=
  (A B C D : Point)
  (area : ℝ)
  (angle_A_trisectors : line → line)
  (angle_C_trisectors : line → line)

theorem trisecting_lines_divide_areas
  (abc_parallelogram : Parallelogram)
  (h_area : abc_parallelogram.area = 1)
  (h_trisectors : ∀ trisector in range(abc_parallelogram.angle_A_trisectors ++ abc_parallelogram.angle_C_trisectors), 
    angle_trisector_property abc_parallelogram trisector) :
  (area_of_regions abc_parallelogram angle_trisectors = [1 / 15, 1 / 15, 1 / 15, 1 / 15, 4 / 15, 4 / 15, 1 / 5]) :=
sorry

end trisecting_lines_divide_areas_l93_93933


namespace dasha_rectangle_problem_l93_93947

variables (a b c : ℕ)

theorem dasha_rectangle_problem
  (h1 : a > 0) 
  (h2 : a * (b + c) + a * (b - a) + a^2 + a * (c - a) = 43) 
  : (a = 1 ∧ b + c = 22) ∨ (a = 43 ∧ b + c = 2) :=
by
  sorry

end dasha_rectangle_problem_l93_93947


namespace volume_of_obtuse_isosceles_revolution_l93_93534

noncomputable def cot (x : ℝ) : ℝ := 1 / Mathlib.Tan.tan x

/-- The volume of the solid of revolution formed by rotating an obtuse isosceles triangle around a line through its orthocenter parallel to the base -/
theorem volume_of_obtuse_isosceles_revolution
  (α : ℝ) (a : ℝ)
  (hα1 : 0 < α) (hα2 : α < π) (hα3 : α > π / 2) :
  V = (π * a ^ 3 / 12) * (3 - cot (α / 2) ^ 2) :=
sorry


end volume_of_obtuse_isosceles_revolution_l93_93534


namespace algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93481

-- Definition of the mailbox setting and conditions
def mailbox_count : ℕ := 80

def apartment_resident_start : ℕ := 37

noncomputable def randomized_keys_placed_in_mailboxes : Bool :=
  true  -- This is a placeholder; actual randomness is abstracted

-- Statement of the problem
theorem algorithm_will_find_key (mailboxes keys : Fin mailbox_count) (start : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ (sequence : ℕ → Fin mailbox_count), ∀ n : ℕ, sequence n ≠ start → sequence (n+1) ≠ start → 
  (sequence 0 = start ∨ ∃ k : ℕ, sequence k = keys.start → keys = sequence (k+1)).
  sorry

theorem expected_number_of_residents_using_algorithm 
  (mailboxes keys : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ n : ℝ, n ≈ 4.968.
  sorry

end algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93481


namespace volume_of_prism_is_two_l93_93945

variables {r h : ℝ}

-- Define the conditions
def DK := 2
def DA := Real.sqrt 6

-- Assume the Pythagorean relationships derived in the solution
def equation1 : Prop := (h^2) / 4 + 4 * (r^2) = 4
def equation2 : Prop := h^2 + r^2 = 6

-- Define the function to show the volume of the prism equals 2
def volume_of_prism (S h : ℝ) := S * h

-- Proof statement
theorem volume_of_prism_is_two (h r : ℝ) (h_eq : equation1) (r_eq : equation2) : volume_of_prism ((Real.sqrt 3) / 2) (4 * Real.sqrt 3 / 3) = 2 :=
by sorry

end volume_of_prism_is_two_l93_93945


namespace inequality_abc_l93_93835

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_abc :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) :=
  sorry

end inequality_abc_l93_93835


namespace largest_angle_of_obtuse_isosceles_triangle_l93_93902

variables (X Y Z : ℝ)

def is_triangle (X Y Z : ℝ) : Prop := X + Y + Z = 180
def is_isosceles_triangle (X Y : ℝ) : Prop := X = Y
def is_obtuse_triangle (X Y Z : ℝ) : Prop := X > 90 ∨ Y > 90 ∨ Z > 90

theorem largest_angle_of_obtuse_isosceles_triangle
  (X Y Z : ℝ)
  (h1 : is_triangle X Y Z)
  (h2 : is_isosceles_triangle X Y)
  (h3 : X = 30)
  (h4 : is_obtuse_triangle X Y Z) :
  Z = 120 :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l93_93902


namespace algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93479

-- Definition of the mailbox setting and conditions
def mailbox_count : ℕ := 80

def apartment_resident_start : ℕ := 37

noncomputable def randomized_keys_placed_in_mailboxes : Bool :=
  true  -- This is a placeholder; actual randomness is abstracted

-- Statement of the problem
theorem algorithm_will_find_key (mailboxes keys : Fin mailbox_count) (start : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ (sequence : ℕ → Fin mailbox_count), ∀ n : ℕ, sequence n ≠ start → sequence (n+1) ≠ start → 
  (sequence 0 = start ∨ ∃ k : ℕ, sequence k = keys.start → keys = sequence (k+1)).
  sorry

theorem expected_number_of_residents_using_algorithm 
  (mailboxes keys : Fin mailbox_count) 
  (h_random_keys : randomized_keys_placed_in_mailboxes) :
  ∃ n : ℝ, n ≈ 4.968.
  sorry

end algorithm_will_find_key_expected_number_of_residents_using_algorithm_l93_93479


namespace smallest_composite_no_prime_factors_less_than_20_l93_93020

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93020


namespace product_in_base_10_l93_93589

def binary_to_decimal (b : ℕ) : ℕ :=
  match b with
  | 0 => 0
  | n => 
    if n % 10 = 0 then 2 * binary_to_decimal (n / 10)
    else 2 * binary_to_decimal (n / 10) + 1

def ternary_to_decimal (t : ℕ) : ℕ :=
  match t with
  | 0 => 0
  | n => 
    if n % 10 = 0 then 3 * ternary_to_decimal (n / 10)
    else (n % 10) + 3 * ternary_to_decimal (n / 10)

theorem product_in_base_10 : 
  binary_to_decimal 1011 * ternary_to_decimal 112 = 154 := by
  sorry

end product_in_base_10_l93_93589


namespace eccentricity_of_ellipse_l93_93644

-- Definition of the conditions
variables {a b c : ℝ} (e : ℝ)

-- The ellipse equation and conditions as definitions in Lean
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def a_gt_b_gt_0 : Prop := a > b ∧ b > 0
def line_through_focus_perpendicular (P Q : ℝ × ℝ) : Prop := 
  ∃ F : ℝ × ℝ, F = (c, 0) ∧ (P.1 = F.1 ∧ Q.1 = F.1) ∧ (P.2 ≠ Q.2)
def directrix_right_intersects_Xaxis (M : ℝ × ℝ) : Prop := M = (a^2 / c, 0)
def triangle_PQM_equilateral (P Q M : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, (P = (x, y) ∧ Q = (x, -y) ∧ M = (a^2 / c, 0) ∧ 
              dist P Q = dist Q M ∧ dist Q M = dist M P)

-- The proof statement
theorem eccentricity_of_ellipse :
  ∀ P Q M : ℝ × ℝ,
    a > b ∧ b > 0 →
    ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 →
    line_through_focus_perpendicular P Q →
    directrix_right_intersects_Xaxis M →
    triangle_PQM_equilateral P Q M →
    e = c / a →
    e = (Real.sqrt 3) / 3 :=
by sorry

end eccentricity_of_ellipse_l93_93644


namespace whole_numbers_between_cubicroots_l93_93711

theorem whole_numbers_between_cubicroots :
  ∀ x y : ℝ, (3 < real.cbrt 50 ∧ real.cbrt 50 < 4) ∧ (7 < real.cbrt 500 ∧ real.cbrt 500 < 8) →
  ∃ n : ℕ, n = 4 := 
by
  sorry

end whole_numbers_between_cubicroots_l93_93711


namespace proof_rp_eq_rq_l93_93619

noncomputable def triangle (A B C : Type) [linear_ordered_field A] [euclidean_geometry B] (BC : B) (ABC : BC > 0) : Type :=
{AB BC : B // (AB > BC)}

variables {A B C : Type} [linear_ordered_field A] [euclidean_geometry B] (ABC : triangle A B C)
variables {Ω : circle A B} {M N K P Q R : point B} (H1 : R ∈ Ω.mid_arc A B C) (H2 : angle.eq R "ABC")

-- Let M and N lie on sides AB and BC respectively such that AM = CN
variables (H3 : ∃ M N : point B, M ∈ ABC.AB ∧ N ∈ ABC.BC ∧ segment.length (ABC.A M) = segment.length (ABC.C N))

-- Let K be the intersection of MN and AC
variables (H4 : ∃ K : point B, K ∈ (line MN ∩ line AC))

-- P is the incenter of ΔAMK and Q is the K-excenter of ΔCNK
variables (H5 : ∃ P : point B, is_incenter (triangle AM K) P)
variables (H6 : ∃ Q : point B, is_excenter (triangle CN K) Q)

-- R is the midpoint of the arc ABC of Ω
variables (H7 : ∃ R : point B, is_arc_midpoint (circumcircle ABC) R)

-- Prove RP = RQ
theorem proof_rp_eq_rq (H8 : segment.length (segment RP) = segment.length (segment RQ)) : segment.eq (segment RP) (segment RQ) :=
by { sorry }

end proof_rp_eq_rq_l93_93619


namespace remainder_of_S_mod_500_l93_93331

open BigOperators

-- Definitions based on the conditions in a)
def R : Finset ℕ := (Finset.range 100).image (λ n, (2 ^ n) % 500)
def S : ℕ := ∑ x in R, x

-- Statement of the problem in Lean 4
theorem remainder_of_S_mod_500 : S % 500 = 0 := sorry

end remainder_of_S_mod_500_l93_93331


namespace sum_of_digits_at_least_362_l93_93919

theorem sum_of_digits_at_least_362 (N k : ℕ) (h1 : k = N - 46) (h2 : sum_of_digits k = 352) : sum_of_digits N ≥ 362 :=
sorry

end sum_of_digits_at_least_362_l93_93919


namespace smallest_composite_no_prime_factors_below_20_l93_93123

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l93_93123


namespace whole_numbers_between_cuberoots_l93_93690

theorem whole_numbers_between_cuberoots :
  let a := real.cbrt 50
  let b := real.cbrt 500
  3 < a ∧ a < 4 →
  7 < b ∧ b < 8 →
  {n : ℤ | (a : ℝ) < (n : ℝ) ∧ (n : ℝ) < (b : ℝ)}.card = 4 :=
by
  intros
  sorry

end whole_numbers_between_cuberoots_l93_93690


namespace ellipse_equation_and_m_value_range_l93_93197

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_equation_and_m_value_range (k m : ℝ) (h_ne_zero : k ≠ 0) :
  (∃ a : ℝ, a^2 = 3 ∧ (∀ x y : ℝ, (x^2 / a^2) + y^2 = 1 -> line_intersection (λ x, k*x + m) (λ x y, (x^2 / a^2) + y^2 = 1) = 2 -> |distance (0, -1) M = distance (0, -1) N| -> (∃ m_lower m_upper : ℝ, m_lower < m ∧ m < m_upper))) :=
sorry

end ellipse_equation_and_m_value_range_l93_93197


namespace dist_B1_to_plane_EFG_l93_93538

/-- In a unit cube with points \( E, F, G \) being the midpoints of edges \( AA_1, C_1D_1, \) and \( D_1A_1 \) respectively, prove the distance from point \( B_1 \) to the plane containing \( EFG \) is \( \frac{\sqrt{3}}{2} \). -/
theorem dist_B1_to_plane_EFG : 
  let cube := unit_cube
  let E := midpoint (cube.A) (cube.A1)
  let F := midpoint (cube.C1) (cube.D1)
  let G := midpoint (cube.D1) (cube.A1)
  in distance cube.B1 (plane_through E F G) = sqrt 3 / 2 :=
sorry

end dist_B1_to_plane_EFG_l93_93538


namespace value_of_stocks_l93_93257

def initial_investment (bonus : ℕ) (stocks : ℕ) : ℕ := bonus / stocks
def final_value_stock_A (initial : ℕ) : ℕ := initial * 2
def final_value_stock_B (initial : ℕ) : ℕ := initial * 2
def final_value_stock_C (initial : ℕ) : ℕ := initial / 2

theorem value_of_stocks 
    (bonus : ℕ) (stocks : ℕ) (h_bonus : bonus = 900) (h_stocks : stocks = 3) : 
    initial_investment bonus stocks * 2 + initial_investment bonus stocks * 2 + initial_investment bonus stocks / 2 = 1350 :=
by
    sorry

end value_of_stocks_l93_93257


namespace fourth_animal_in_sequence_is_sheep_l93_93517

def sequence : List String := ["horse", "cow", "pig", "sheep", "rabbit", "squirrel"]

theorem fourth_animal_in_sequence_is_sheep :
  sequence.get? 3 = some "sheep" :=
sorry

end fourth_animal_in_sequence_is_sheep_l93_93517


namespace bisect_angle_ACN_l93_93866

variables {A B C K L M N X : Type} [Triangle A B C] 
  (AE_A : Excircle A B C K L) 
  (BE_B : Excircle B C A M N) 
  (H1 : Line K L) (H2 : Line M N) (HX : Intersection (Line K L) (Line M N) X)

theorem bisect_angle_ACN (H : Line (C X)) : Bisects (Angle A C N) (Line (C X)) :=
by
  sorry

end bisect_angle_ACN_l93_93866


namespace num_roses_given_l93_93425

theorem num_roses_given (n : ℕ) (m : ℕ) (x : ℕ) :
  n = 28 → 
  (∀ (b g : ℕ), b + g = n → b * g = 45 * x) →
  (num_roses : ℕ) = 4 * x →
  (num_tulips : ℕ) = 10 * num_roses →
  (num_daffodils : ℕ) = x →
  num_roses = 16 :=
by
  sorry

end num_roses_given_l93_93425


namespace smallest_composite_no_prime_factors_less_than_20_l93_93141

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93141


namespace binomial_expansion_terms_l93_93308

theorem binomial_expansion_terms (x : ℝ) :
  ∃ k ∈ {0, 1, 2, 3, 4, 5}, x^(10 - 3 * k) = x ∧
  ∃ k ∈ {0, 1, 2, 3, 4, 5}, x^(10 - 3 * k) = 1 / x^2 ∧
  ∃ k ∈ {0, 1, 2, 3, 4, 5}, x^(10 - 3 * k) = x^4 ∧
  ¬∃ k ∈ {0, 1, 2, 3, 4, 5}, x^(10 - 3 * k) = 1 / x^4 :=
begin
  sorry
end

end binomial_expansion_terms_l93_93308


namespace number_of_points_75_ray_not_50_ray_partitional_l93_93832

noncomputable def unitSquare : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

def is_n_ray_partitional (R : set (ℝ × ℝ)) (n : ℕ) (X : ℝ × ℝ) : Prop :=
  ∃ rays : ℕ → ℝ, (∀ i, 0 ≤ rays i ∧ rays i < 2 * π) ∧
  (∀ i, i < n → let θ := i * 2 * π / n in 
    let p1 := (X.1 + cos θ, X.2 + sin θ) in 
    let p2 := (X.1 + cos (θ + 2 * π / n), X.2 + sin (θ + 2 * π / n)) in
    area (X, p1, p2) = (1 / n) * area R)

theorem number_of_points_75_ray_not_50_ray_partitional :
  (∃ (count : ℕ), count = 5625 ∧
    ∀ X, (X ∈ unitSquare) → 
      is_n_ray_partitional unitSquare 75 X ∧ ¬ is_n_ray_partitional unitSquare 50 X) :=
sorry

end number_of_points_75_ray_not_50_ray_partitional_l93_93832


namespace last_a_replacement_l93_93514

-- Function to calculate the sum of first n natural numbers
def sum_of_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Encrypting function for circular shift in an alphabet
def shift_alpha (ch : Char) (n : ℕ) : Char :=
  let base := 'a'.to_nat
  let ch_val := ch.to_nat - base
  let shifted_val := (ch_val + n) % 26
  Char.of_nat (shifted_val + base)

-- Message to analyze and conditions
def message := "Alabama has a panama hat, ha!"
def occurrences_of_char := 10

-- Calculate the shift
def calculated_shift := sum_of_first_n occurrences_of_char % 26

-- Problem statement to prove
theorem last_a_replacement : 
  shift_alpha 'a' calculated_shift = 'd' :=
by {
  -- Proof goes here, omitted per instructions.
  sorry
}

end last_a_replacement_l93_93514


namespace find_A_max_height_AD_l93_93299

-- Assuming the necessary imports/constants.
constant triangle (A B C : ℝ) : Prop
constant side_a (A B C a : ℝ) : Prop
constant side_b (A B C b : ℝ) : Prop 
constant side_c (A B C c : ℝ) : Prop 
constant acute (A B C : ℝ) : Prop
constant sinC (A B C sinC : ℝ) : Prop
constant sum_b_c (b c : ℝ) : Prop
constant height_AD (A B C a b c AD : ℝ) : Prop

-- Conditions
axiom triangle_ABC : acute A B C → triangle A B C
axiom sides_abc : side_a A B C a → side_b A B C b → side_c A B C c
axiom sinC_eq : sinC C = 2 * (cos A) * (sin (B + π/3))
axiom bc_sum_eq_6 : sum_b_c b c = 6

-- Proofs
theorem find_A : triangle A B C → side_a A B C a → side_b A B C b → side_c A B C c → 
  sinC C → A = π / 3 :=
by 
  sorry

theorem max_height_AD : triangle A B C → side_a A B C a → side_b A B C b → side_c A B C c →
  sinC C → sum_b_c b c → A = π / 3 → 
  height_AD A B C a b c AD ≤ 3 * sqrt 3 / 2 :=
by
  sorry

end find_A_max_height_AD_l93_93299


namespace count_whole_numbers_between_cubes_l93_93682

theorem count_whole_numbers_between_cubes :
  (∀ x, 3 < x ∧ x < 4 → real.cbrt 50 = x) →
  (∀ y, 7 < y ∧ y < 8 → real.cbrt 500 = y) →
  ∃ n : ℤ, n = 4 :=
by
  sorry

end count_whole_numbers_between_cubes_l93_93682


namespace find_key_effective_expected_residents_to_use_algorithm_l93_93471

-- Define mailboxes and keys
def num_mailboxes : ℕ := 80
def initial_mailbox : ℕ := 37

-- Prove that the algorithm is effective
theorem find_key_effective : 
  ∀ (mailboxes : fin num_mailboxes) (keys : fin num_mailboxes), 
  { permutation : list (fin num_mailboxes) // permutation.nodup ∧ permutation.length = num_mailboxes ∧ 
    ∀ m, m ∈ permutation → (if m = initial_mailbox then m else (keys.filter (λ k, k ∈ permutation))) ≠ [] }
  :=
sorry

-- Prove the expected number of residents who will use the algorithm
theorem expected_residents_to_use_algorithm :
  ∑ i in finset.range num_mailboxes, (1 / (i + 1 : ℝ)) = (real.log 80 + 0.577 + (1 / (2 * 80)) : ℝ)
  :=
sorry

end find_key_effective_expected_residents_to_use_algorithm_l93_93471


namespace smallest_composite_no_prime_factors_lt_20_l93_93154

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l93_93154


namespace min_correct_answers_for_score_above_60_l93_93292

theorem min_correct_answers_for_score_above_60 :
  ∃ (x : ℕ), 6 * x - 2 * (15 - x) > 60 ∧ x = 12 :=
by
  sorry

end min_correct_answers_for_score_above_60_l93_93292


namespace sum_of_factors_636405_l93_93881

theorem sum_of_factors_636405 :
  ∃ (a b c : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 ∧
    a * b * c = 636405 ∧ a + b + c = 259 :=
sorry

end sum_of_factors_636405_l93_93881


namespace reeyas_first_subject_score_l93_93377

theorem reeyas_first_subject_score
  (second_subject_score third_subject_score fourth_subject_score : ℕ)
  (num_subjects : ℕ)
  (average_score : ℕ)
  (total_subjects_score : ℕ)
  (condition1 : second_subject_score = 76)
  (condition2 : third_subject_score = 82)
  (condition3 : fourth_subject_score = 85)
  (condition4 : num_subjects = 4)
  (condition5 : average_score = 75)
  (condition6 : total_subjects_score = num_subjects * average_score) :
  67 = total_subjects_score - (second_subject_score + third_subject_score + fourth_subject_score) := 
  sorry

end reeyas_first_subject_score_l93_93377


namespace smallest_composite_no_prime_factors_less_than_20_l93_93102

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93102


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l93_93966

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l93_93966


namespace part1_part2_l93_93582

def f (x : ℝ) : ℝ := abs (x - 2) - abs (x - 5)

theorem part1 : ∀ x : ℝ, -3 ≤ f(x) ∧ f(x) ≤ 3 :=
by
  sorry

theorem part2 : {x : ℝ | f x ≥ x^2 - 8*x + 15} = {x : ℝ | 5 - real.sqrt 3 ≤ x ∧ x ≤ 6} :=
by
  sorry

end part1_part2_l93_93582


namespace number_of_whole_numbers_between_cubicroots_l93_93738

theorem number_of_whole_numbers_between_cubicroots :
  3 < Real.cbrt 50 ∧ Real.cbrt 500 < 8 → ∃ n : Nat, n = 4 :=
begin
  sorry
end

end number_of_whole_numbers_between_cubicroots_l93_93738


namespace smallest_composite_no_prime_factors_less_than_20_l93_93117

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93117


namespace total_votes_l93_93536

theorem total_votes (V : ℝ)
  (four_candidates : True)
  (winner_receives_at_least_70_percent : V ≥ 0.70 * V)
  (winner_majority : V * 0.70 - V * 0.30 = 3000) :
  V = 7500 :=
begin
  sorry
end

end total_votes_l93_93536


namespace intersection_points_lie_on_circle_l93_93854

theorem intersection_points_lie_on_circle :
  ∀ x y : ℝ, 
  (x + 2 * y = 19 ∧ y = 1 / x ∨ y + 2 * x = 98 ∧ y = 1 / x) → 
  (x - 34) ^ 2 + (y - 215 / 4) ^ 2 = 49785 / 16 :=
by
  intros x y h
  cases h
  sorry

end intersection_points_lie_on_circle_l93_93854


namespace base_area_of_cuboid_eq_seven_l93_93987

-- Definitions of the conditions
def volume_of_cuboid : ℝ := 28 -- Volume is 28 cm³
def height_of_cuboid : ℝ := 4  -- Height is 4 cm

-- The theorem statement for the problem
theorem base_area_of_cuboid_eq_seven
  (Volume : ℝ)
  (Height : ℝ)
  (h1 : Volume = 28)
  (h2 : Height = 4) :
  Volume / Height = 7 := by
  sorry

end base_area_of_cuboid_eq_seven_l93_93987


namespace g_neither_even_nor_odd_l93_93312

noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem g_neither_even_nor_odd :
  (∀ x, g (-x) = g x → false) ∧ (∀ x, g (-x) = -g x → false) :=
by
  unfold g
  sorry

end g_neither_even_nor_odd_l93_93312


namespace equal_perimeters_of_partition_l93_93620

noncomputable def triangle_partition (a b c : ℝ) (x : ℝ) := a + b + x = b + (c - x) + c - (c - x)

theorem equal_perimeters_of_partition : 
  ∀ (a b c : ℝ), a = 7 → b = 12 → c = 9 → (∃ x : ℝ, triangle_partition a b c x) → (x = 2 ∨ x = 7) :=
begin
  intros,
  sorry,
end

end equal_perimeters_of_partition_l93_93620


namespace smallest_composite_no_prime_factors_less_than_20_l93_93017

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l93_93017


namespace james_money_left_l93_93319

variables (initial_amount : ℕ) (cost_per_ticket : ℕ) (roommate_percentage : ℚ)

def first_three_tickets_cost (n : ℕ) : ℕ := 3 * n

def fourth_ticket_cost (n : ℕ) : ℕ := n / 4

def fifth_ticket_cost (n : ℕ) : ℕ := n / 2

def total_cost (t1 t4 t5 : ℕ) : ℕ := t1 + t4 + t5

def roommate_contrib (total : ℕ) (perc : ℚ) : ℕ := (perc * total).toInt

def james_payment (total roommate_pay : ℕ) : ℕ := total - roommate_pay

def money_left (initial pay : ℕ) : ℕ := initial - pay

theorem james_money_left (H1 : initial_amount = 800) (H2 : cost_per_ticket = 200) (H3 : roommate_percentage = 0.60) :
  money_left
    initial_amount
    (james_payment
      (total_cost
        (first_three_tickets_cost cost_per_ticket)
        (fourth_ticket_cost cost_per_ticket)
        (fifth_ticket_cost cost_per_ticket))
      (roommate_contrib
        (total_cost
          (first_three_tickets_cost cost_per_ticket)
          (fourth_ticket_cost cost_per_ticket)
          (fifth_ticket_cost cost_per_ticket))
        roommate_percentage)) = 500 :=
by {
  sorry
}

end james_money_left_l93_93319


namespace find_m_n_l93_93829

theorem find_m_n (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_sol : (m + Real.sqrt n)^2 - 10 * (m + Real.sqrt n) + 1 = Real.sqrt (m + Real.sqrt n) * (m + Real.sqrt n + 1)) : m + n = 55 :=
sorry

end find_m_n_l93_93829


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l93_93072

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l93_93072


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l93_93967

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l93_93967


namespace units_digit_is_6_l93_93302

noncomputable def units_digit_of_square (A : ℕ) : ℕ :=
  (A * A) % 10

theorem units_digit_is_6 (a b : ℕ) (hb : b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (htens : (10 * a + b)^2 % 100 / 10 = 7) :
  units_digit_of_square (10 * a + b) = 6 :=
sorry

end units_digit_is_6_l93_93302


namespace ratio_of_bases_l93_93301

theorem ratio_of_bases 
(AB CD : ℝ) 
(h_trapezoid : AB < CD) 
(h_AC : ∃ k : ℝ, k = 2 * CD ∧ k = AC) 
(h_altitude : AB = (D - foot)) : 
AB / CD = 3 := 
sorry

end ratio_of_bases_l93_93301


namespace problem_l93_93201

theorem problem {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h : 3 * a * b = a + 3 * b) :
  (3 * a + b >= 16/3) ∧
  (a * b >= 4/3) ∧
  (a^2 + 9 * b^2 >= 8) ∧
  (¬ (b > 1/2)) :=
by
  sorry

end problem_l93_93201


namespace smallest_composite_no_prime_factors_less_20_l93_93047

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l93_93047


namespace differential_savings_l93_93466

def annual_income_before_tax : ℝ := 42400
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.32

theorem differential_savings :
  annual_income_before_tax * initial_tax_rate - annual_income_before_tax * new_tax_rate = 4240 :=
by
  sorry

end differential_savings_l93_93466


namespace distance_focus_to_asymptote_l93_93629

-- Define the hyperbola and the conditions for m
def hyperbola (x y : ℝ) (m : ℝ) := x^2 - m * y^2 = 3 * m
def asymptote (x y : ℝ) (m : ℝ) := x + sqrt m * y = 0
def focus_x (m : ℝ) := sqrt (3 * m + 3)
def condition (m : ℝ) := m > 0

-- Define the distance function from a point to a line
def distance (px py a b c : ℝ) : ℝ := abs (a * px + b * py + c) / sqrt (a^2 + b^2)

-- The main theorem to be proven
theorem distance_focus_to_asymptote (m : ℝ) (h : condition m) :
  let F_x := focus_x m,
      F_y := 0,
      asymptote_a := 1,
      asymptote_b := sqrt m,
      asymptote_c := 0
  in distance F_x F_y asymptote_a asymptote_b asymptote_c = sqrt 3 :=
by {
  sorry
}

end distance_focus_to_asymptote_l93_93629


namespace find_t_over_q_l93_93253

theorem find_t_over_q
  (q r s v t : ℝ)
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := 
sorry

end find_t_over_q_l93_93253


namespace willie_stickers_l93_93459

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 124 → given_stickers = 23 → remaining_stickers = initial_stickers - given_stickers → remaining_stickers = 101 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining.trans rfl

end willie_stickers_l93_93459


namespace totalCups_l93_93508

-- Let's state our definitions based on the conditions:
def servingsPerBox : ℕ := 9
def cupsPerServing : ℕ := 2

-- Our goal is to prove the following statement.
theorem totalCups (hServings: servingsPerBox = 9) (hCups: cupsPerServing = 2) : servingsPerBox * cupsPerServing = 18 := by
  -- The detailed proof will go here.
  sorry

end totalCups_l93_93508


namespace general_formula_a_sum_b_l93_93833

-- Define the sequence {a_n}
def a (n : ℕ) : ℝ := 2 * n + 1

-- Sum of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℝ := ∑ i in finset.range n, a i

-- Conditions given in the problem
axiom a_pos (n : ℕ) : a n > 0
axiom condition_a (n : ℕ) : a n ^ 2 + 2 * a n = 4 * S n + 3

-- Proving the first part: general formula for {a_n}
theorem general_formula_a (n : ℕ) : a n = 2 * n + 1 :=
by
  sorry

-- Define the sequence {b_n}
def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))

-- Sum of the first n terms of {b_n}
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Proving the second part: sum of the first n terms of {b_n}
theorem sum_b (n : ℕ) : T n = n / (3 * (2 * n + 3)) :=
by
  sorry

end general_formula_a_sum_b_l93_93833


namespace students_per_minibus_calculation_l93_93313

-- Define the conditions
variables (vans minibusses total_students students_per_van : ℕ)
variables (students_per_minibus : ℕ)

-- Define the given conditions based on the problem
axiom six_vans : vans = 6
axiom four_minibusses : minibusses = 4
axiom ten_students_per_van : students_per_van = 10
axiom total_students_are_156 : total_students = 156

-- Define the problem statement in Lean
theorem students_per_minibus_calculation
  (h1 : vans = 6)
  (h2 : minibusses = 4)
  (h3 : students_per_van = 10)
  (h4 : total_students = 156) :
  students_per_minibus = 24 :=
sorry

end students_per_minibus_calculation_l93_93313


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l93_93964

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l93_93964


namespace inequality_solution_l93_93418

theorem inequality_solution (x : ℝ) (hx : x > 0) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := 
sorry

end inequality_solution_l93_93418


namespace find_angle_NCB_l93_93783

noncomputable def triangle_ABC (A B C N : Type) := 
  ∃ (α β γ δ : ℝ),
    α = 40 ∧ β = 20 ∧ γ = 30 ∧ δ = 20 ∧ α + β + γ + δ = 180

theorem find_angle_NCB (A B C N : Type) 
  (h₁: ∃ triangle_ABC A B C N): 
  ∃ (θ : ℝ), θ = 10 :=
begin
  sorry
end

end find_angle_NCB_l93_93783


namespace vector_dot_product_orthogonality_l93_93204

variables {a b c : ℝ → ℝ}  -- Assuming vectors are functions from ℝ to ℝ as an example
variable h_non_zero_a : (∃ x, a x ≠ 0)  -- assuming non-zero vector means ∃ an element in the function/domain such that it's non-zero
variable h_non_zero_b_c : (∃ x, (b x - c x) ≠ 0)

theorem vector_dot_product_orthogonality :
  (∀ x, a x * b x = a x * c x) ↔ (∀ x, a x * (b x - c x) = 0) :=
begin
  sorry
end

end vector_dot_product_orthogonality_l93_93204


namespace optimal_lenses_for_green_text_l93_93952

theorem optimal_lenses_for_green_text
  (greenText: Prop)
  (whiteBackground: Prop)
  (redLenses: Prop)
  (greenLenses: Prop)
  (redLensesDarkenGreen: Prop)
  (greenLensesUnchangedGreen: Prop)
  : (redLenses → better_visibility greenText) :=
by
  intros
  sorry

end optimal_lenses_for_green_text_l93_93952


namespace prove_lambda_eq_find_ellipse_equation_l93_93645

variable {a b e : ℝ} (h1 : a > b > 0) 
          (h2 : ∀ (x y : ℝ), \(((x / a)^2 + (y / b)^2 = 1) ↔ (x, y) ∈ C)\))
          (h3 : ∃ (M A B : (ℝ × ℝ)), ∃ λ : ℝ, 
                (line_eq : ∀ (x : ℝ), y = e * x + a) ∧
                (AB_eq : ∃ A B x_A x_B y_A y_B x : ℝ, 
                    ∧ ((a / x = x_B) = (b * x / a = y_B))))
                                             

theorem prove_lambda_eq : 
  ∀ e : ℝ, λ = 1 - e^2 :=
  sorry

theorem find_ellipse_equation :
  ∀ e : ℝ, λ = (3 / 4) →  
       2a + 2c = 6 →
       ∃ (x y : ℝ), 
        ((1 - e) = (3/4)
 
end prove_lambda_eq_find_ellipse_equation_l93_93645


namespace lemonade_production_l93_93822

theorem lemonade_production (lemons lemonades : ℕ) (h : lemonades = 5 * lemons) : ∀ n : ℕ, n = 18 → lemonades = 90 :=
by {
  intros n hn,
  rw hn at h,
  sorry
}

end lemonade_production_l93_93822
