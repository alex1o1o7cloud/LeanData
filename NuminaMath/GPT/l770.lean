import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.Field
import Mathlib.Algebra.ModularArithmetic
import Mathlib.Analysis.Probability.Normal
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Pnat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.EuclideanSpace.Basic
import Mathlib.Topology.Instances.Real
import Probability.ProbabilityMassFunction

namespace f_n_minus_1_divisible_by_n_l770_770186

open Nat

def a_n (n : ℕ) : ℝ :=
  n * (1 + 1 / n) ^ n

noncomputable def f (n : ℕ) : ℝ :=
  ∏ i in range n.succ, a_n i.succ

theorem f_n_minus_1_divisible_by_n (n : ℕ) (h : n > 0) :
  (f n - 1) % n = 0 :=
sorry

end f_n_minus_1_divisible_by_n_l770_770186


namespace range_of_a_l770_770872

def A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a < x ∧ x < a + 1}

theorem range_of_a (a : ℝ)
  (h₀ : a < 1)
  (h₁ : B a ⊆ A) :
  a ∈ {x : ℝ | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
by
  sorry

end range_of_a_l770_770872


namespace height_of_B_after_rotation_l770_770809

theorem height_of_B_after_rotation :
  ∀ (side : ℝ) (θ : ℝ), 
  side = 2 → θ = 30 → 
  height_of_rotated_vertex side θ = 2 :=
by
  intros side θ h_side h_theta
  -- Definitions and assumptions provided in conditions
  let base_line := 0
  let B := point_of_top_vertex_side_θ side θ
  assume height_of_rotated_vertex := function that calculates height of vertex after rotation
  sorry

end height_of_B_after_rotation_l770_770809


namespace find_angle_l770_770953

theorem find_angle (x : ℝ) (h1 : 90 - x = (1/2) * (180 - x)) : x = 90 :=
by
  sorry

end find_angle_l770_770953


namespace line_MN_parallel_angle_bisector_l770_770736

-- Definitions and conditions
variables {A B C E F M N : Type*}

-- Prove the theorem assuming the conditions given
theorem line_MN_parallel_angle_bisector
  (triangle_ABC : Triangle A B C)
  (extension_CA_A : CA extends A)
  (extension_AB_B : AB extends B)
  (AE_eq_BC : AE = BC)
  (BF_eq_AC : BF = AC)
  (circle_tangent_BF_at_N : Tangent (Circle _) BF at N)
  (circle_tangent_BC : Tangent (Circle _) BC)
  (circle_tangent_extension_AC : Tangent (Circle _) (Extension AC) beyond C)
  (M_midpoint_EF : Midpoint M EF) :
  Parallel (Line M N) (AngleBisector A) :=
sorry

end line_MN_parallel_angle_bisector_l770_770736


namespace sequence_formula_l770_770972

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n ≥ 2, a n = 2 * n - 1) := 
by
  sorry

end sequence_formula_l770_770972


namespace value_of_c_distinct_real_roots_l770_770558

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770558


namespace min_expr_value_l770_770497

theorem min_expr_value (a b c : ℝ) (h₀ : b > c) (h₁ : c > a) (h₂ : a > 0) (h₃ : b ≠ 0) :
  (∀ (a b c : ℝ), b > c → c > a → a > 0 → b ≠ 0 → 
   (2 + 6 * a^2 = (a+b)^3 / b^2 + (b-c)^2 / b^2 + (c-a)^3 / b^2) →
   2 <= (a + b)^3 / b^2 + (b - c)^2 / b^2 + (c - a)^3 / b^2) :=
by 
  sorry

end min_expr_value_l770_770497


namespace geometric_sequence_range_of_q_l770_770165

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = a 0 * (1 - q^n) / (1 - q)

theorem geometric_sequence_range_of_q
  (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h1 : geometric_sequence a q)
  (h2 : ∀ n : ℕ, n > 0 → S (2 * n) < 3 * S n)
  (h3 : ∀ n : ℕ, a n > 0) :
  0 < q ∧ q ≤ 1 :=
by
  sorry

end geometric_sequence_range_of_q_l770_770165


namespace max_min_values_in_region_l770_770123

-- Define the function
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D (x y : ℝ) : Prop := (0 ≤ x) ∧ (x - 2 * y ≤ 0) ∧ (x + y - 6 ≤ 0)

-- Define the proof problem
theorem max_min_values_in_region :
  (∀ (x y : ℝ), D x y → z x y ≥ 0) ∧
  (∀ (x y : ℝ), D x y → z x y ≤ 32) :=
by 
  sorry -- Proof omitted

end max_min_values_in_region_l770_770123


namespace find_rate_l770_770431

def simple_interest_rate (P A T : ℕ) : ℕ :=
  ((A - P) * 100) / (P * T)

theorem find_rate :
  simple_interest_rate 750 1200 5 = 12 :=
by
  -- This is the statement of equality we need to prove
  sorry

end find_rate_l770_770431


namespace restaurant_total_earnings_l770_770424

-- Definitions based on conditions
def weekday_earning := 600
def weekend_earning_min := 1000
def weekend_earning_max := 1500
def discount_percentage := 0.10
def special_event_earning := 500
def holiday_season_earning_min := 600
def holiday_season_earning_max := 900
def total_weekdays := 22
def special_occasions := 2
def discount_days := 4
def regular_weekdays := total_weekdays - special_occasions - discount_days
def total_weekends := 8
def promotional_weekends := 3
def regular_weekends := total_weekends - promotional_weekends

-- Calculations based on steps
def regular_weekday_total := regular_weekdays * weekday_earning
def discount_amount := discount_percentage * weekday_earning
def discount_day_earning := weekday_earning - discount_amount
def discount_total := discount_days * discount_day_earning
def special_occasion_earning_avg := (holiday_season_earning_min + holiday_season_earning_max) / 2
def special_occasions_total := special_occasions * special_occasion_earning_avg
def weekend_earning_avg := (weekend_earning_min + weekend_earning_max) / 2
def regular_weekend_total := regular_weekends * weekend_earning_min
def promotional_weekend_total := promotional_weekends * weekend_earning_avg

-- Final total earnings
def total_earnings := regular_weekday_total + discount_total + special_occasions_total + regular_weekend_total + promotional_weekend_total + special_event_earning

-- Theorem statement
theorem restaurant_total_earnings : total_earnings = 22510 := by
  sorry

end restaurant_total_earnings_l770_770424


namespace tony_will_have_4_dollars_in_change_l770_770341

def tony_change : ℕ :=
  let bucket_capacity : ℕ := 2
  let sandbox_depth : ℕ := 2
  let sandbox_width : ℕ := 4
  let sandbox_length : ℕ := 5
  let sand_weight_per_cubic_foot : ℕ := 3
  let water_consumed_per_drink : ℕ := 3
  let trips_per_drink : ℕ := 4
  let bottle_capacity : ℕ := 15
  let bottle_cost : ℕ := 2
  let initial_money : ℕ := 10

  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let number_of_trips := total_sand_weight / bucket_capacity
  let number_of_drinks := number_of_trips / trips_per_drink
  let total_water_consumed := number_of_drinks * water_consumed_per_drink
  let number_of_bottles := total_water_consumed / bottle_capacity
  let total_water_cost := number_of_bottles * bottle_cost
  let change := initial_money - total_water_cost

  change

theorem tony_will_have_4_dollars_in_change : tony_change = 4 := by
  sorry

end tony_will_have_4_dollars_in_change_l770_770341


namespace ana_final_salary_l770_770912

def initial_salary : ℝ := 2500
def june_raise : ℝ := initial_salary * 0.15
def june_bonus : ℝ := 300
def salary_after_june : ℝ := initial_salary + june_raise + june_bonus
def july_pay_cut : ℝ := salary_after_june * 0.25
def final_salary : ℝ := salary_after_june - july_pay_cut

theorem ana_final_salary :
  final_salary = 2381.25 := by
  -- sorry is used here to skip the proof
  sorry

end ana_final_salary_l770_770912


namespace roots_of_polynomial_l770_770126

   -- We need to define the polynomial and then state that the roots are exactly {0, 3, -5}
   def polynomial (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x)

   theorem roots_of_polynomial :
     {x : ℝ | polynomial x = 0} = {0, 3, -5} :=
   by
     sorry
   
end roots_of_polynomial_l770_770126


namespace trapezoid_AD_proof_l770_770671

noncomputable def trapezoid_AD_length (A B C D O : Point) : ℝ :=
if h₁ : is_isosceles_trapezoid A B C D ∧ base AD > base BC ∧ side_length A C = 20 ∧
           angle_A B A C = 45 ∧ is_circumcircle_center O A B C D ∧ orthogonal_line O D A B
then 10 * (sqrt 6 + sqrt 2)
else 0

theorem trapezoid_AD_proof (A B C D O : Point)
  (h₁ : is_isosceles_trapezoid A B C D)
  (h₂ : base AD > base BC)
  (h₃ : side_length A C = 20)
  (h₄ : angle (A B C) = 45)
  (h₅ : is_circumcircle_center O A B C D)
  (h₆ : orthogonal_line O D A B) :
  trapezoid_AD_length A B C D O = 10 * (sqrt 6 + sqrt 2) := 
sorry

end trapezoid_AD_proof_l770_770671


namespace trig_function_value_l770_770196

noncomputable def f : ℝ → ℝ := sorry

theorem trig_function_value:
  (∀ x, f (Real.cos x) = Real.cos (3 * x)) →
  f (Real.sin (Real.pi / 6)) = -1 :=
by
  intro h
  sorry

end trig_function_value_l770_770196


namespace range_of_a_plus_k_l770_770604

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := a^x + k * a^(-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def is_decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem range_of_a_plus_k (a k : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : k = -1)
  (h4 : is_odd_function (f a k)) (h5 : is_decreasing_function (f a k)) :
    a + k ∈ Ioo (-1 : ℝ) (0 : ℝ) :=
sorry

end range_of_a_plus_k_l770_770604


namespace valid_m_values_l770_770676

theorem valid_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) → m < 1 :=
by
  sorry

end valid_m_values_l770_770676


namespace sum_of_cubes_mod_5_l770_770480

theorem sum_of_cubes_mod_5 :
  ( ∑ k in Finset.range 50, (k + 1)^3 ) % 5 = 0 :=
sorry

end sum_of_cubes_mod_5_l770_770480


namespace quadratic_has_distinct_real_roots_l770_770529

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770529


namespace Sam_dimes_remaining_l770_770755

-- Define the initial and borrowed dimes
def initial_dimes_count : Nat := 8
def borrowed_dimes_count : Nat := 4

-- State the theorem
theorem Sam_dimes_remaining : (initial_dimes_count - borrowed_dimes_count) = 4 := by
  sorry

end Sam_dimes_remaining_l770_770755


namespace equal_shaded_unshaded_parts_l770_770347

-- Define the basic context for the problem
variable (ΔA1 ΔA2 H : ℝ)
variable (awhite agray : ℝ)

-- Conditions
axiom congruent_triangles : ΔA1 = ΔA2
axiom area_triangle1 : ΔA1 = H + awhite
axiom area_triangle2 : ΔA2 = H + agray

-- Statement to prove
theorem equal_shaded_unshaded_parts : awhite = agray :=
by 
  -- Using axiom stating the triangles are congruent, hence their areas are same
  have h : ΔA1 = ΔA2 := congruent_triangles
  -- Substituting areas of each triangle
  rw [area_triangle1, area_triangle2] at h
  -- Rearranging to show the areas of white and gray regions are equal
  exact eq_of_add_eq_add_right h

end equal_shaded_unshaded_parts_l770_770347


namespace inverse_proportion_rises_left_to_right_l770_770657

theorem inverse_proportion_rises_left_to_right (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → y = k / x → (x > 0 → y rises as x increases)) → k < 0 := 
begin
  sorry
end

end inverse_proportion_rises_left_to_right_l770_770657


namespace race_distance_l770_770874

theorem race_distance (D : ℕ) (h1 : ∀ A B : ℕ, A = D → B = D - 50) 
                      (h2 : ∀ B C : ℕ, B = D → C = D - 25) 
                      (h3 : ∀ A C : ℕ, A = 400 → C = 342)   
                      : D = 172 := 
by 
  -- Given conditions in the problem
  have h1 : A = D → B = D - 50 := sorry, -- relation between A and B
  have h2 : B = D → C = D - 25 := sorry, -- relation between B and C
  have h3 : A = 400 → C = 342 := sorry, -- relation between A and C
  
  -- Derived ratio and solving for D
  have h_ratio : (D - 25) / D = 342 / 400 := sorry, -- derived from speed consistency
  
  -- Prove final distance D
  have h_equiv : D = 172 := sorry,
  exact h_equiv -- Concluding the proof

end race_distance_l770_770874


namespace tony_will_have_4_dollars_in_change_l770_770340

def tony_change : ℕ :=
  let bucket_capacity : ℕ := 2
  let sandbox_depth : ℕ := 2
  let sandbox_width : ℕ := 4
  let sandbox_length : ℕ := 5
  let sand_weight_per_cubic_foot : ℕ := 3
  let water_consumed_per_drink : ℕ := 3
  let trips_per_drink : ℕ := 4
  let bottle_capacity : ℕ := 15
  let bottle_cost : ℕ := 2
  let initial_money : ℕ := 10

  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let number_of_trips := total_sand_weight / bucket_capacity
  let number_of_drinks := number_of_trips / trips_per_drink
  let total_water_consumed := number_of_drinks * water_consumed_per_drink
  let number_of_bottles := total_water_consumed / bottle_capacity
  let total_water_cost := number_of_bottles * bottle_cost
  let change := initial_money - total_water_cost

  change

theorem tony_will_have_4_dollars_in_change : tony_change = 4 := by
  sorry

end tony_will_have_4_dollars_in_change_l770_770340


namespace inequality_not_necessarily_hold_l770_770645

theorem inequality_not_necessarily_hold (a b c : ℝ) (h : a > b) (h₀ : b > 0) : ∃ c, ac ≤ bc :=
by
  use sorry

end inequality_not_necessarily_hold_l770_770645


namespace quadratic_roots_condition_l770_770513

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770513


namespace point_not_on_graph_l770_770080

theorem point_not_on_graph (x y : ℝ) : 
  (x = 1 ∧ y = 5 → y ≠ 6 / x) :=
by
  intros h
  cases h with hx hy
  rw [hx, hy]
  norm_num
  exact ne_of_gt (by norm_num)
  sorry

end point_not_on_graph_l770_770080


namespace correct_propositions_l770_770181

noncomputable def f : ℝ → ℝ := sorry

def proposition1 : Prop :=
  ∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x) → ∀ x : ℝ, f (2 - x) = f x

def proposition2 : Prop :=
  ∀ x : ℝ, f (x - 2) = f (2 - x)

def proposition3 : Prop :=
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (2 + x) = -f x) → ∀ x : ℝ, f x = f (4 - x)

def proposition4 : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f x = f (-x - 2)) → ∀ x : ℝ, f (2 - x) = f x

theorem correct_propositions : proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l770_770181


namespace sum_sines_l770_770040

theorem sum_sines (α : ℝ) (n : ℕ) (h : n > 0) :
  (Finset.range n).sum (λ k, Real.sin (α + 2 * k * Real.pi / n)) = 0 := sorry

end sum_sines_l770_770040


namespace sum_of_cubes_mod_five_l770_770484

theorem sum_of_cubes_mod_five : 
  (∑ n in Finset.range 51, n^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_five_l770_770484


namespace simpson_line_l770_770045
open_locale euclidean_geometry

theorem simpson_line (A B C P P1 P2 P3 : Point) :
  (∃ ω : Circle, Circle.center ω = circumcenter ⟨(A, B, C), sorry⟩ ∧ P ∈ ω) ↔ is_collinear {P1, P2, P3} :=
begin
  sorry,
end

end simpson_line_l770_770045


namespace base_seven_to_base_ten_l770_770843

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770843


namespace volume_oblique_prism_l770_770129

noncomputable def volume_of_oblique_prism (a : ℝ) :=
  let S_triangle := (a^2 * Real.sqrt 3) / 4 in
  let height := a * (Real.sqrt 3 / 2) in
  S_triangle * height

theorem volume_oblique_prism (a : ℝ) :
  volume_of_oblique_prism a = 3 * a^3 / 8 :=
by
  sorry

end volume_oblique_prism_l770_770129


namespace construct_diameter_segment_l770_770408

-- Define the conditions and the question.
variables {S : Type} -- the type representing the sphere and its surface
variables (A B : S) -- arbitrary points on the surface of the sphere
variables (M L P : S) -- points on the circle on the surface
variables (M' L' P' O1' A' B' : ℝ) -- corresponding points on the plane
variables (radius : ℝ) -- radius of the sphere
variables (diameter : ℝ := 2 * radius) -- diameter of the sphere

-- Assume the essential relationships that translate from the sphere to the plane
axiom transfer_points_congruent : triangle_congruent (M, L, P) (M', L', P')
axiom right_triangle : right_triangle (M', O1', A') -- triangle with hypotenuse equal to radius
axiom perpendicular_distance : perp_distance M (O1) = segment_length (M' O1')

theorem construct_diameter_segment :
  ∃ (A' B' : ℝ), segment_length A' B' = diameter :=
by
  sorry

end construct_diameter_segment_l770_770408


namespace sum_of_distances_l770_770684

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem sum_of_distances :
  let D : ℝ × ℝ := (0, 0)
  let E : ℝ × ℝ := (10, 0)
  let F : ℝ × ℝ := (7, 5)
  let P : ℝ × ℝ := (4, 2)
  distance P D + distance P E + distance P F = 2 * real.sqrt 5 + 2 * real.sqrt 10 + 3 * real.sqrt 2 :=
by
  sorry

end sum_of_distances_l770_770684


namespace solidConstruction_l770_770293

-- Definitions
structure Solid where
  octagonal_faces : Nat
  hexagonal_faces : Nat
  square_faces : Nat

-- Conditions
def solidFromCube (S : Solid) : Prop :=
  S.octagonal_faces = 6 ∧ S.hexagonal_faces = 8 ∧ S.square_faces = 12

def circumscribedSphere (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

def solidFromOctahedron (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

-- Theorem statement
theorem solidConstruction {S : Solid} :
  solidFromCube S ∧ circumscribedSphere S → solidFromOctahedron S :=
by
  sorry

end solidConstruction_l770_770293


namespace Raine_total_steps_l770_770749

-- Define the steps taken to and from school each day
def Monday_steps_to_school := 150
def Monday_steps_back := 170
def Tuesday_steps_to_school := 140
def Tuesday_steps_back := 140 + 30
def Wednesday_steps_to_school := 160
def Wednesday_steps_back := 210
def Thursday_steps_to_school := 150
def Thursday_steps_back := 140 + 30
def Friday_steps_to_school := 180
def Friday_steps_back := 200

-- Define total steps for each day
def Monday_total_steps := Monday_steps_to_school + Monday_steps_back
def Tuesday_total_steps := Tuesday_steps_to_school + Tuesday_steps_back
def Wednesday_total_steps := Wednesday_steps_to_school + Wednesday_steps_back
def Thursday_total_steps := Thursday_steps_to_school + Thursday_steps_back
def Friday_total_steps := Friday_steps_to_school + Friday_steps_back

-- Define the total steps for all five days
def total_steps :=
  Monday_total_steps +
  Tuesday_total_steps +
  Wednesday_total_steps +
  Thursday_total_steps +
  Friday_total_steps

-- Prove that the total steps equals 1700
theorem Raine_total_steps : total_steps = 1700 := 
by 
  unfold total_steps
  unfold Monday_total_steps Tuesday_total_steps Wednesday_total_steps Thursday_total_steps Friday_total_steps
  unfold Monday_steps_to_school Monday_steps_back
  unfold Tuesday_steps_to_school Tuesday_steps_back
  unfold Wednesday_steps_to_school Wednesday_steps_back
  unfold Thursday_steps_to_school Thursday_steps_back
  unfold Friday_steps_to_school Friday_steps_back
  sorry

end Raine_total_steps_l770_770749


namespace angle_S_is_45_l770_770756

-- Defining the setup
variables (P Q R S T : Type)
variables [metric_space P] [metric_space Q] [metric_space R] [metric_space S] [metric_space T]
variables [PT : dist P T = dist T Q]
variables [TR : dist T R = dist T S]
variables [PT_TR : dist P T = dist T R]
variables [angle_equality : ∀ {θP θR : ℝ}, θP = 4 * θR]

-- Theorem to prove
theorem angle_S_is_45 :
  ∃ (θS : ℝ), θS = 45 :=
by
  -- Proof outline will be here
  sorry

end angle_S_is_45_l770_770756


namespace who_broke_the_jar_l770_770743

axiom Squirrel : Type
axiom Gray Blackie Rusty FlameTail : Squirrel

axiom statement_Gray : Prop
axiom statement_Blackie : Prop
axiom statement_Rusty : Prop
axiom statement_FlameTail : Prop

axiom statement_Gray_def : statement_Gray = (∃ (broke : Squirrel), broke = Blackie)
axiom statement_Blackie_def : statement_Blackie = (∃ (broke : Squirrel), broke = FlameTail)
axiom statement_Rusty_def : statement_Rusty = ¬(∃ (broke : Squirrel), broke = Rusty)
axiom statement_FlameTail_def : statement_FlameTail = ¬statement_Blackie

axiom exactly_one_truth : (statement_Gray ∨ statement_Blackie ∨ statement_Rusty ∨ statement_FlameTail) ∧
                          (¬statement_Gray ∨ ¬statement_Blackie ∨ ¬statement_Rusty ∨ ¬statement_FlameTail) ∧
                          (¬statement_Gray ∨ statement_Gray) ∧
                          (¬statement_Blackie ∨ statement_Blackie) ∧
                          (¬statement_Rusty ∨ statement_Rusty) ∧
                          (¬statement_FlameTail ∨ statement_FlameTail) ∧
                          ¬(statement_Gray ∧ statement_Blackie) ∧
                          ¬(statement_Gray ∧ statement_Rusty) ∧
                          ¬(statement_Gray ∧ statement_FlameTail) ∧
                          ¬(statement_Blackie ∧ statement_Rusty) ∧
                          ¬(statement_Blackie ∧ statement_FlameTail) ∧
                          ¬(statement_Rusty ∧ statement_FlameTail)

theorem who_broke_the_jar : statement_FlameTail ∧ (∃ (broke : Squirrel), broke = Rusty) := by
  sorry

end who_broke_the_jar_l770_770743


namespace limit_set_measurability_l770_770720

-- Definitions required for the problem conditions
variable {Ω : Type} {F : Ω → Set (Set Ω)} {measurable_space : Set (Set Ω) → Prop}
noncomputable def is_measurable (A : Set Ω) : Prop := measurable_space A

variables {ξ : ℕ → Ω → ℝ} -- Sequence of random variables

-- The Lean statement to prove the question given the conditions
theorem limit_set_measurability {Ω : Type} {F : Set (Set Ω)} [measurable_space F] (ξ : ℕ → Ω → ℝ) :
  is_measurable {ω : Ω | ¬ ∃ L : ℝ, tendsto (λ n, ξ n ω) at_top (pure L)} ∧
  is_measurable {ω : Ω | ∃ L : ℝ, tendsto (λ n, ξ n ω) at_top (pure L)} :=
sorry

end limit_set_measurability_l770_770720


namespace base_7_to_10_of_23456_l770_770828

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770828


namespace largest_c_for_minus5_in_range_l770_770102

theorem largest_c_for_minus5_in_range :
  ∃ x : ℝ, ∀ c : ℝ, (-5) ∈ (λ x, x^2 + 4 * x + c) → c = -1 :=
by
  sorry

end largest_c_for_minus5_in_range_l770_770102


namespace min_distance_from_curve_to_line_l770_770230

noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt6 : ℝ := Real.sqrt 6

def curve_parametric (α : ℝ) : ℝ × ℝ :=
( sqrt2 * Real.cos α, Real.sin α)

def curve_equation (x y : ℝ) : Prop :=
  (x ^ 2 / 2) + (y ^ 2) = 1

def line_cartesian (x y : ℝ) : Prop :=
  x + y = 8

def distance_to_line (x y : ℝ) : ℝ :=
  (Real.abs (x + y - 8)) / sqrt2

def minimum_distance_condition (α θ : ℝ) : ℝ :=
  (Real.abs (sqrt3 * Real.sin (α + θ) - 8)) / sqrt2

theorem min_distance_from_curve_to_line:
  ∃ α θ, 
  (∀ (x y : ℝ), curve_parametric α = (x, y) → curve_equation x y) ∧
  (∀ (ρ θ' : ℝ), ρ * Real.sin (θ' + (π / 4)) = 4 * sqrt2 → line_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∀ α θ, minimum_distance_condition α θ = (8 * sqrt2 - sqrt6) / 2) := 
begin
  sorry -- Proof omitted
end

end min_distance_from_curve_to_line_l770_770230


namespace quadratic_has_distinct_real_roots_l770_770530

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770530


namespace trapezoid_PQRD_area_l770_770358

structure Point where
  x : ℝ
  y : ℝ

def trapezoid_area (P Q R D : Point) : ℝ :=
  let base1 := abs (Q.y - P.y)
  let base2 := abs (D.y - R.y)
  let height := abs (R.x - P.x)
  (base1 + base2) * height / 2

theorem trapezoid_PQRD_area :
  let P := Point.mk 0 0
  let Q := Point.mk 0 (-3)
  let R := Point.mk 5 0
  let D := Point.mk 5 9
  trapezoid_area P Q R D = 30 := 
by 
  sorry

end trapezoid_PQRD_area_l770_770358


namespace number_of_sequences_l770_770989

-- Define the sequence and its properties
def sequence_condition (a_n : Fin 6 → ℝ) : Prop :=
  (∃ (A B C D E F : Fin 6), a_n A = 2 ∧ a_n B = 2 ∧ a_n C = 2 ∧
                           a_n D = Real.sqrt 3 ∧ a_n E = Real.sqrt 3 ∧ a_n F = 5 ∧
                           List.nodup [A, B, C, D, E, F])

-- Theorem that states the number of different sequences satisfying the conditions
theorem number_of_sequences : ∃! a_n : Fin 6 → ℝ, sequence_condition a_n := by
  -- Placeholder for the proof
  sorry

end number_of_sequences_l770_770989


namespace find_a_10_l770_770154

theorem find_a_10 (a : ℕ → ℚ)
  (h0 : a 1 = 1)
  (h1 : ∀ n : ℕ, a (n + 1) = a n / (a n + 2)) :
  a 10 = 1 / 1023 :=
sorry

end find_a_10_l770_770154


namespace locus_of_T_l770_770982

variables {Point Line Circle : Type} [Geometry Point Line Circle]

-- Define center O, fixed circle C, and line L
variable {O : Point}
variable {C : Circle}
variable {L : Line}

-- Conditions from the problem
def fixed_line_through_O (L : Line) (O : Point) : Prop := L.contains O
def fixed_circle_C (C : Circle) (O : Point) : Prop := C.center = O
def variable_point_P_on_L (P : Point) (L : Line) : Prop := L.contains P
def circle_with_center_P_and_radius_OP (K : Circle) (P O : Point) : Prop :=
  K.center = P ∧ ∃ r, K.radius r ∧ (distance P O = r)
def common_tangent_point_T (T : Point) (C K : Circle) : Prop :=
  ∃ l, l.is_tangent C ∧ l.is_tangent K ∧ l.contains T

-- Theorem to prove
theorem locus_of_T (C : Circle) (L : Line) (O : Point) (hC : fixed_circle_C C O) (hL : fixed_line_through_O L O) :
  ∃ T : Point, 
    (∀ P : Point, variable_point_P_on_L P L → 
      ∃ K : Circle, circle_with_center_P_and_radius_OP K P O ∧ 
      common_tangent_point_T T C K) ∧
  (∃ B1 B2 : Point, L.contains B1 ∧ L.contains B2 ∧ perpendicular L (set.locus_of_T T B1 B2)) := sorry

end locus_of_T_l770_770982


namespace sum_of_reciprocal_s_n_l770_770259

def arithmetic_sequence_sum (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

def sequence_sum_first_n_terms (a₁ d : ℕ) (n : ℕ) : ℕ :=
  List.sum (List.map (λ i, arithmetic_sequence_sum a₁ d i) (List.range n))

theorem sum_of_reciprocal_s_n :
  let S_n := λ n, n * (n + 1)
  let T_10 := List.sum (List.map (λ n, 1 / (S_n (n + 1))) (List.range 10))
  S_1 = 2 → (S_5 / 5 - S_3 / 3 = 2) → T_10 = 10 / 11 :=
by
  sorry

end sum_of_reciprocal_s_n_l770_770259


namespace largest_k_divides_1764_factorial_l770_770445

open Nat

theorem largest_k_divides_1764_factorial :
  let p := 1764
  let k := 293 / 2
  k = 146 :=
by 
  let k := (252 + 36 + 5) / 2;
  have : 293 / 2 = 146,
  sorry

end largest_k_divides_1764_factorial_l770_770445


namespace mice_population_l770_770433

theorem mice_population :
  ∃ (mice_initial : ℕ) (pups_per_mouse : ℕ) (survival_rate_first_gen : ℕ → ℕ) 
    (survival_rate_second_gen : ℕ → ℕ) (num_dead_first_gen : ℕ) (pups_eaten_per_adult : ℕ)
    (total_mice : ℕ),
    mice_initial = 8 ∧ pups_per_mouse = 7 ∧
    (∀ n, survival_rate_first_gen n = (n * 80) / 100) ∧
    (∀ n, survival_rate_second_gen n = (n * 60) / 100) ∧
    num_dead_first_gen = 2 ∧ pups_eaten_per_adult = 3 ∧
    total_mice = mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse)) - num_dead_first_gen + (survival_rate_second_gen ((mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse))) * pups_per_mouse)) - ((mice_initial - num_dead_first_gen) * pups_eaten_per_adult) :=
  sorry

end mice_population_l770_770433


namespace unique_largest_negative_integer_l770_770804

theorem unique_largest_negative_integer :
  ∃! x : ℤ, x = -1 ∧ (∀ y : ℤ, y < 0 → x ≥ y) :=
by
  sorry

end unique_largest_negative_integer_l770_770804


namespace train_stop_time_l770_770112

theorem train_stop_time (speed_excluding_stoppages speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 54)
  (h2 : speed_including_stoppages = 40) :
  ∃ t : ℝ, t ≈ 15.56 ∧ 
  let distance_diff := speed_excluding_stoppages - speed_including_stoppages in
  let speed_per_minute := speed_excluding_stoppages / 60 in
  t = distance_diff / speed_per_minute :=
by
  use 15.56
  sorry

end train_stop_time_l770_770112


namespace solution_set_l770_770163

-- Define that f is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the given function
def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x → f x = x - 1

-- Define the main theorem statement
theorem solution_set (f : ℝ → ℝ) (h_even : even_function f) (h_given : given_function f) :
  {x : ℝ | f x < 0} = set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end solution_set_l770_770163


namespace inequality_property_l770_770079

theorem inequality_property (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : (a / b) > (b / a) := 
sorry

end inequality_property_l770_770079


namespace sin_phi_correct_l770_770704

variables {p q r : EuclideanSpace ℝ (Fin 3)}
variables {φ : ℝ}

-- Conditions from the problem
def norm_p : ‖p‖ = 2 := sorry
def norm_q : ‖q‖ = 4 := sorry
def norm_r : ‖r‖ = 6 := sorry
def cross_product_condition : p × (p × q) = r := sorry

noncomputable def sin_phi : ℝ := sorry

-- Statement to be proved
theorem sin_phi_correct : sin_phi = 3 / 4 :=
begin
  -- Proof would go here
  sorry
end

end sin_phi_correct_l770_770704


namespace illuminate_arena_with_spotlights_l770_770041

theorem illuminate_arena_with_spotlights :
  ∃ f : Fin 100 → Set (Fin 100), 
  (∀ k : Fin 100, 
    is_convex (f k) ∧ 100 - f k ≠ ∅ ∧ (∀ j, j ≠ k → (f k ∪ f j) = (finset.range 100).to_set)) :=
by sorry

end illuminate_arena_with_spotlights_l770_770041


namespace quadratic_two_distinct_real_roots_l770_770516

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770516


namespace problem_l770_770980

-- Define a type for points in 2D space
structure Point2D :=
(x : ℝ)
(y : ℝ)

-- Define a function to calculate the Euclidean distance between two points
def dist (P Q : Point2D) : ℝ :=
(real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2))

-- Define a set of 10 points
variables {P : fin 10 → Point2D}

-- Define the main property function
def property_max_min :=
  ∃ (i j k l : fin 10), (i ≠ j ∧ k ≠ l ∧ 
  dist (P i) (P j) = max (finₙ 10 → finₙ 10 → ℝ) (λ i j, dist (P i) (P j)) ∧
  dist (P k) (P l) = min (finₙ 10 → finₙ 10 → ℝ) (λ i j, dist (P i) (P j)) ∧
  dist (P i) (P j) ≥ 2 * dist (P k) (P l))

-- The theorem to be proven
theorem problem (P : fin 10 → Point2D) : property_max_min :=
sorry

end problem_l770_770980


namespace angle_between_a_plus_b_and_a_minus_b_l770_770625

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

def angle_between_vectors_is_90_degrees : Prop :=
  let a := (sin (real.pi / 12), cos (real.pi / 12))
  let b := (cos (real.pi / 12), sin (real.pi / 12))
  let a_plus_b := (sin (real.pi / 12) + cos (real.pi / 12), sin (real.pi / 12) + cos (real.pi / 12))
  let a_minus_b := (sin (real.pi / 12) - cos (real.pi / 12), cos (real.pi / 12) - sin (real.pi / 12))
  let dot_product := (sin (real.pi / 12) + cos (real.pi / 12)) * (sin (real.pi / 12) - cos (real.pi / 12)) + (sin (real.pi / 12) + cos (real.pi / 12)) * (cos (real.pi / 12) - sin (real.pi / 12))
  dot_product = 0

theorem angle_between_a_plus_b_and_a_minus_b :
  angle_between_vectors_is_90_degrees := 
by {
  sorry
}

end angle_between_a_plus_b_and_a_minus_b_l770_770625


namespace flour_qualification_l770_770903

def acceptable_weight_range := {w : ℝ | 24.75 ≤ w ∧ w ≤ 25.25}

theorem flour_qualification :
  (24.80 ∈ acceptable_weight_range) ∧ 
  (24.70 ∉ acceptable_weight_range) ∧ 
  (25.30 ∉ acceptable_weight_range) ∧ 
  (25.51 ∉ acceptable_weight_range) :=
by 
  -- The proof would go here, but we are adding sorry to skip it.
  sorry

end flour_qualification_l770_770903


namespace extremum_points_range_of_m_l770_770178

theorem extremum_points_range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (λ x, x + (m+1)*Real.exp x) x1 = 0 ∧ (λ x, x + (m+1)*Real.exp x) x2 = 0) →
  -1 - (1/Real.exp 1) < m ∧ m < -1 :=
begin
  sorry
end

end extremum_points_range_of_m_l770_770178


namespace card_area_after_shortening_l770_770732

theorem card_area_after_shortening 
  (original_width original_length : ℕ) 
  (original_area new_area : ℕ) 
  (shorten_by : ℕ)
  (h₀ : original_width = 7)
  (h₁ : original_length = 3)
  (h₂ : original_area = original_width * original_length)
  (h₃ : shorten_by = 2)
  (h₄ : new_area = original_length * (original_width - shorten_by))
  (h₅ : new_area = 15) :
  (let new_side_length := original_length - shorten_by in
   new_side_length * original_width = 7) := 
by 
  sorry

end card_area_after_shortening_l770_770732


namespace largest_number_in_set_l770_770204

theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  let s := {-2 * a, 5 * a, 36 / a, a^3, (-a) ^ 2}
  (-a)^2 = max (max (max (-2 * a) (5 * a)) (max (36 / a) (a^3))) ((-a)^2) := by
sorry

end largest_number_in_set_l770_770204


namespace base_seven_to_base_ten_l770_770847

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770847


namespace area_of_nth_square_l770_770892

theorem area_of_nth_square (n : ℕ) : 
  let S : ℕ → ℕ := λ n, 2^(n-1) in
  ∀ k > 0, S k = 2^(k-1) := sorry

end area_of_nth_square_l770_770892


namespace problem1_problem2_l770_770438

-- Problem 1: Prove that m * m^3 + (-m^2)^3 / m^2 = 0 given m ∈ ℝ and m ≠ 0
theorem problem1 (m : ℝ) (h : m ≠ 0) : m * m^3 + (-m^2)^3 / m^2 = 0 :=
sorry

-- Problem 2: Prove that (x + 2y)(x - 2y) - (x - y)^2 = -5y^2 + 2xy given x, y ∈ ℝ
theorem problem2 (x y : ℝ) : (x + 2y) * (x - 2y) - (x - y)^2 = -5y^2 + 2 * x * y :=
sorry

end problem1_problem2_l770_770438


namespace smallest_natural_number_l770_770411

theorem smallest_natural_number (x : ℕ) : 
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) → x = 122 := 
by
  sorry

end smallest_natural_number_l770_770411


namespace solution_to_absolute_equation_l770_770963

noncomputable def smallest_solution : ℝ :=
  let f (x : ℝ) := x * |x| - 4 * x - 3 in
  if hx₀ : ∃ x : ℝ, 0 ≤ x ∧ f x = 0 then
    let x₀ := classical.some hx₀ in
    if hx₀_pos : 0 ≤ x₀ then min x₀ (-3) else -3
  else
    -3

theorem solution_to_absolute_equation :
  ∃ x : ℝ, x = -3 ∧ (∀ y : ℝ, (y * |y| = 4 * y + 3) → x ≤ y) :=
  sorry

end solution_to_absolute_equation_l770_770963


namespace relation_between_A_and_B_l770_770160

-- Define the sets A and B
def A : Set ℤ := { x | ∃ k : ℕ, x = 7 * k + 3 }
def B : Set ℤ := { x | ∃ k : ℤ, x = 7 * k - 4 }

-- Prove the relationship between A and B
theorem relation_between_A_and_B : A ⊆ B :=
by
  sorry

end relation_between_A_and_B_l770_770160


namespace combined_mpg_l770_770291

theorem combined_mpg :
  let mR := 150 -- miles Ray drives
  let mT := 300 -- miles Tom drives
  let mpgR := 50 -- miles per gallon for Ray's car
  let mpgT := 20 -- miles per gallon for Tom's car
  -- Total gasoline used by Ray and Tom
  let gR := mR / mpgR
  let gT := mT / mpgT
  -- Total distance driven
  let total_distance := mR + mT
  -- Total gasoline used
  let total_gasoline := gR + gT
  -- Combined miles per gallon
  let combined_mpg := total_distance / total_gasoline
  combined_mpg = 25 := by
    sorry

end combined_mpg_l770_770291


namespace visible_factor_numbers_200_to_250_l770_770410

def is_visible_factor_number (n : ℕ) : Prop :=
  let digits := (to_digits 10 n).filter (≠ 0)
  digits.all (λ d, n % d = 0)

def visible_factor_numbers_count (low high : ℕ) : ℕ :=
  (List.range' low (high - low + 1)).count is_visible_factor_number

theorem visible_factor_numbers_200_to_250 : visible_factor_numbers_count 200 250 = 27 :=
by
  sorry

end visible_factor_numbers_200_to_250_l770_770410


namespace circle_center_and_radius_l770_770310

theorem circle_center_and_radius:
  ∀ x y : ℝ, 
  (x + 1) ^ 2 + (y - 3) ^ 2 = 36 
  → ∃ C : (ℝ × ℝ), C = (-1, 3) ∧ ∃ r : ℝ, r = 6 := sorry

end circle_center_and_radius_l770_770310


namespace abc_equal_l770_770587

theorem abc_equal (a b c : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a)
  (h2 : ∀ x : ℝ, b * x^2 + c * x + a ≥ c * x^2 + a * x + b) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l770_770587


namespace number_of_true_propositions_l770_770172

-- Definitions of events A and B and their characteristics
variables (A B : Type) [probability_space A] [probability_space B]

-- Conditions (1), (2), and (3)
def certain_event (A : Type) [probability_space A] : Prop :=
∀ x, P(x ∈ A) = 1

def mutually_exclusive (A B : Type) [probability_space A] [probability_space B] : Prop :=
∀ x, P(x ∈ (A ∩ B)) = 0

def true_propositions (A B : Type) [probability_space A] [probability_space B] : ℕ :=
(if certain_event A then 1 else 0) + 
(if mutually_exclusive A B ∧ P(A ∪ B) = P(A) + P(B) then 1 else 0) + 
(if mutually_exclusive A B ∧ P(A) + P(B) ≠ 1 then 1 else 0)

-- Statement to be proved
theorem number_of_true_propositions (A B : Type) [probability_space A] [probability_space B] :
  true_propositions A B = 2 :=
sorry

end number_of_true_propositions_l770_770172


namespace Cassidy_current_poster_count_l770_770093

theorem Cassidy_current_poster_count :
  ∀ (initial : ℕ) (lost : ℕ) (sold : ℕ) (gain : ℕ) (years : ℕ) (final : ℕ),
  initial = 18 →
  lost = 2 →
  sold = 5 →
  gain = 6 →
  years = 3 →
  final = 2 * initial →
  final - gain = 30 :=
by
  intros initial lost sold gain years final
  assume h_initial h_lost h_sold h_gain h_years h_final
  rw [h_initial, h_lost, h_sold, h_gain, h_final]
  sorry

end Cassidy_current_poster_count_l770_770093


namespace complex_roots_equilateral_l770_770265

noncomputable def omega : ℂ := complex.exp (complex.I * real.pi / 3)

def quadratic_eq_roots (a b : ℂ) (z1 z2 : ℂ) : Prop :=
  (z1 + z2 = -a) ∧ (z1 * z2 = b)

def equilateral_triangle (z1 z2 : ℂ) : Prop :=
  z2 = omega * z1

theorem complex_roots_equilateral (a b z1 z2 : ℂ) : 
  quadratic_eq_roots a b z1 z2 → equilateral_triangle z1 z2 → (a^2 / b = 3) :=
by
  intros h1 h2
  rw [quadratic_eq_roots, equilateral_triangle] at h1 h2
  sorry

end complex_roots_equilateral_l770_770265


namespace pints_in_vat_l770_770906

-- Conditions
def num_glasses : Nat := 5
def pints_per_glass : Nat := 30

-- Problem statement: prove that the total number of pints in the vat is 150
theorem pints_in_vat : num_glasses * pints_per_glass = 150 :=
by
  -- Proof goes here
  sorry

end pints_in_vat_l770_770906


namespace range_of_m_l770_770568

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, P x → Q x m ∧ P x) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end range_of_m_l770_770568


namespace num_batches_needed_l770_770914

def num_students := 150
def attendance_drop := 0.40
def cookies_per_student := 3
def cookies_per_batch := 20

theorem num_batches_needed : 
  let expected_students := num_students * (1 - attendance_drop)
  let total_cookies := expected_students * cookies_per_student
  let batches := total_cookies / cookies_per_batch
  Nat.ceil batches = 14 :=
by
  sorry

end num_batches_needed_l770_770914


namespace expression_evaluation_l770_770090

def evaluate_expression : ℝ := (-1) ^ 51 + 3 ^ (2^3 + 5^2 - 7^2)

theorem expression_evaluation :
  evaluate_expression = -1 + (1 / 43046721) :=
by
  sorry

end expression_evaluation_l770_770090


namespace spherical_distance_epi3_l770_770913

theorem spherical_distance_epi3 (O A B C E F : Point)
  (hO : center_of_sphere O 1)
  (hA : on_sphere A O 1)
  (hB : on_sphere B O 1)
  (hC : on_sphere C O 1)
  (hOA_perp_OB : perp (line_through O A) (line_through O B))
  (hOA_perp_OC : perp (line_through O A) (line_through O C))
  (hOB_perp_OC : perp (line_through O B) (line_through O C))
  (hE : midpoint_arc A B E O)
  (hF : midpoint_arc A C F O) :
  spherical_distance E F = π / 3 :=
sorry 

end spherical_distance_epi3_l770_770913


namespace minimum_framing_needed_l770_770390

-- Definitions given the conditions
def original_width := 5
def original_height := 7
def enlargement_factor := 4
def border_width := 3
def inches_per_foot := 12

-- Conditions translated to definitions
def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor
def bordered_width := enlarged_width + 2 * border_width
def bordered_height := enlarged_height + 2 * border_width
def perimeter := 2 * (bordered_width + bordered_height)
def perimeter_in_feet := perimeter / inches_per_foot

-- Prove that the minimum number of linear feet of framing required is 10 feet
theorem minimum_framing_needed : perimeter_in_feet = 10 := 
by 
  sorry

end minimum_framing_needed_l770_770390


namespace sum_of_cubes_mod_5_l770_770469

theorem sum_of_cubes_mod_5 :
  (∑ i in Finset.range 51, i^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_5_l770_770469


namespace num_three_digit_numbers_l770_770814

def digits : List ℕ := [1, 2, 3, 4, 5]

def three_digit_numbers (d : List ℕ) : List (List ℕ) :=
  (d.permutations.filter (λ l, l.length = 3)).map (λ l, l.take 3)

theorem num_three_digit_numbers : List.length (three_digit_numbers digits) = 60 := by
  sorry

end num_three_digit_numbers_l770_770814


namespace algebra_expr_value_l770_770174

theorem algebra_expr_value (x y : ℝ) (h : x - 2 * y = 3) : 4 * y + 1 - 2 * x = -5 :=
sorry

end algebra_expr_value_l770_770174


namespace locus_of_foci_of_parabolas_l770_770984

theorem locus_of_foci_of_parabolas (P : Point) (e : Line) : 
  ∃ (t : Line), is_parallel t e ∧ is_parabola_with_focus_and_directrix P t 
    ∧ (∀ (F : Point), is_focus F t P → lies_on_parabola F P t) := 
sorry

end locus_of_foci_of_parabolas_l770_770984


namespace probability_two_sums_of_4_is_one_ninth_l770_770036

-- Define the possible outcomes when rolling a 3-sided die
def possible_outcomes : List (ℕ × ℕ) := [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]

-- Define the event of rolling a sum of 4
def event_sum_4 (outcome : ℕ × ℕ) : Bool :=
  match outcome with
  | (x, y) => x + y = 4

-- Calculate the probability of the event
def probability_event (event : ℕ × ℕ -> Bool) : ℚ :=
  let count_event_true := possible_outcomes.countp event
  let count_total := possible_outcomes.length
  count_event_true / count_total

-- Statement of the problem
theorem probability_two_sums_of_4_is_one_ninth :
  probability_event event_sum_4 * probability_event event_sum_4 = 1 / 9 :=
by
  sorry

end probability_two_sums_of_4_is_one_ninth_l770_770036


namespace a1_eq_1_a2_eq_e_minus_2_a3_eq_6_minus_2e_a4_eq_9e_minus_24_e_bound_l770_770132

-- Prove that for n ≥ 0, a_n = ∫_0^1 x^n e^x dx

def a (n : ℕ) := ∫ x in 0..1, x^n * Real.exp x

theorem a1_eq_1 : a 1 = 1 := 
by
  -- proof goes here
  sorry

theorem a2_eq_e_minus_2 : a 2 = Real.exp 1 - 2 := 
by
  -- proof goes here
  sorry

theorem a3_eq_6_minus_2e : a 3 = 6 - 2 * Real.exp 1 := 
by
  -- proof goes here
  sorry

theorem a4_eq_9e_minus_24 : a 4 = 9 * Real.exp 1 - 24 := 
by
  -- proof goes here
  sorry

-- Show that 8/3 < e < 30/11

theorem e_bound : 8 / 3 < Real.exp 1 ∧ Real.exp 1 < 30 / 11 :=
by
  -- proof goes here
  sorry

end a1_eq_1_a2_eq_e_minus_2_a3_eq_6_minus_2e_a4_eq_9e_minus_24_e_bound_l770_770132


namespace recommendation_methods_correct_l770_770891

-- Define the given conditions and the problem statement in Lean
def num_recommendation_methods (n_students : ℕ) (n_universities : ℕ) : ℕ :=
  let group_1_2_2 := (5.choose 1) * (4.choose 2) * (2.choose 2) / (2.factorial)
  let group_1_1_3 := (5.choose 1) * (4.choose 1) * (3.choose 3) / (2.factorial)
  let total_group_methods := group_1_2_2 + group_1_1_3
  let match_universities := 3.factorial
  total_group_methods * match_universities

theorem recommendation_methods_correct :
  num_recommendation_methods 5 3 = 150 :=
by
  -- Proof omitted
  sorry

end recommendation_methods_correct_l770_770891


namespace third_circle_radius_l770_770398

theorem third_circle_radius {A B : Type} [metric_space A] [metric_space B] (r₁ r₂ : ℝ) (r₃ : ℝ) : 
    (distance A B = r₁ + r₂) ∧ (r₁ = 2) ∧ (r₂ = 5) ∧ 
    (∀ C, (distance A C = r₁ + r₃) ∧ (distance B C = r₂ + r₃) ∧ (distance C (A-₃) + distance C (B-₃) = distance A B)) → 
    r₃ = 1 := 
by
  intro conditions
  sorry

end third_circle_radius_l770_770398


namespace dennis_teaching_years_l770_770351

noncomputable def years_taught (V A D E N : ℕ) := V + A + D + E + N
noncomputable def sum_of_ages := 375
noncomputable def teaching_years : Prop :=
  ∃ (A V D E N : ℕ),
    V + A + D + E + N = 225 ∧
    V = A + 9 ∧
    V = D - 15 ∧
    E = A - 3 ∧
    E = 2 * N ∧
    D = 101

theorem dennis_teaching_years : teaching_years :=
by
  sorry

end dennis_teaching_years_l770_770351


namespace no_solution_l770_770712

def is_digit (B : ℕ) : Prop := B < 10

def divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def satisfies_conditions (B : ℕ) : Prop :=
  is_digit B ∧
  divisible_by (12345670 + B) 2 ∧
  divisible_by (12345670 + B) 5 ∧
  divisible_by (12345670 + B) 11

theorem no_solution (B : ℕ) : ¬ satisfies_conditions B :=
sorry

end no_solution_l770_770712


namespace base_7_to_base_10_l770_770837

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770837


namespace inverse_proportion_rises_left_to_right_l770_770656

theorem inverse_proportion_rises_left_to_right (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → y = k / x → (x > 0 → y rises as x increases)) → k < 0 := 
begin
  sorry
end

end inverse_proportion_rises_left_to_right_l770_770656


namespace quadratic_two_distinct_real_roots_l770_770518

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770518


namespace find_constants_monotonicity_range_of_k_l770_770574

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (b - 2 ^ x) / (2 ^ (x + 1) + a)

theorem find_constants (h_odd : ∀ x : ℝ, f x a b = - f (-x) a b) :
  a = 2 ∧ b = 1 :=
sorry

theorem monotonicity (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1) :
  ∀ x y : ℝ, x < y → f y a b ≤ f x a b :=
sorry

theorem range_of_k (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1)
  (h_pos : ∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0) :
  k < 4 / 3 :=
sorry

end find_constants_monotonicity_range_of_k_l770_770574


namespace stratified_sampling_teachers_l770_770890

theorem stratified_sampling_teachers (total_people total_sampled student_in_sample teachers : ℕ)
  (h1 : total_people = 4000)
  (h2 : total_sampled = 200)
  (h3 : student_in_sample = 190)
  (h4 : teachers = 4000 - 4000 * (200 - 190) / 200) :
  teachers = 200 :=
by 
  rw [h1, h2, h3, h4]
  sorry

end stratified_sampling_teachers_l770_770890


namespace quadratic_distinct_roots_l770_770540

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770540


namespace jane_picked_fraction_l770_770898

-- Define the total number of tomatoes initially
def total_tomatoes : ℕ := 100

-- Define the number of tomatoes remaining at the end
def remaining_tomatoes : ℕ := 15

-- Define the number of tomatoes picked in the second week
def second_week_tomatoes : ℕ := 20

-- Define the number of tomatoes picked in the third week
def third_week_tomatoes : ℕ := 2 * second_week_tomatoes

theorem jane_picked_fraction :
  ∃ (f : ℚ), f = 1 / 4 ∧
    (f * total_tomatoes + second_week_tomatoes + third_week_tomatoes + remaining_tomatoes = total_tomatoes) :=
sorry

end jane_picked_fraction_l770_770898


namespace problem_solution_l770_770334

-- Noncomputable if necessary
noncomputable theory

-- Define the conditions
def condition1 (a b c : ℝ) : Prop := a + b + c = 111
def condition2 (a b c : ℝ) : Prop := a + 10 = b - 10 ∧ b - 10 = 3 * c

-- Define the question
def proof_question (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) : Prop :=
  b = 58

-- Statement we need to prove
theorem problem_solution (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) : proof_question a b c h1 h2 :=
by sorry

end problem_solution_l770_770334


namespace probability_interval_l770_770985

noncomputable def normal_distribution : Type :=
{ μ := 2, σ := σ }

axiom prob_X_less_a (a : ℝ) (ha : a < 2) : 𝓝 (X < a) = 0.32

theorem probability_interval (X : normal_distribution) (a : ℝ) (ha : a < 4 - a) :
  P(a < X < 4 - a) = 0.36 :=
by
  sorry.

end probability_interval_l770_770985


namespace sum_of_cubes_mod_five_l770_770486

theorem sum_of_cubes_mod_five : 
  (∑ n in Finset.range 51, n^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_five_l770_770486


namespace sqrt_fourth_root_x_is_point2_l770_770923

def x : ℝ := 0.000001

theorem sqrt_fourth_root_x_is_point2 :
  (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 0.000001)))) ≈ 0.2 := sorry

end sqrt_fourth_root_x_is_point2_l770_770923


namespace berry_saturday_reading_l770_770432

-- Given data
def sunday_pages := 43
def monday_pages := 65
def tuesday_pages := 28
def wednesday_pages := 0
def thursday_pages := 70
def friday_pages := 56
def average_goal := 50
def days_in_week := 7

-- Calculate total pages to meet the weekly goal
def weekly_goal := days_in_week * average_goal

-- Calculate pages read so far from Sunday to Friday
def pages_read := sunday_pages + monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

-- Calculate required pages to read on Saturday
def saturday_pages_required := weekly_goal - pages_read

-- The theorem statement: Berry needs to read 88 pages on Saturday.
theorem berry_saturday_reading : saturday_pages_required = 88 := 
by {
  -- The proof is omitted as per the instructions
  sorry
}

end berry_saturday_reading_l770_770432


namespace sin_double_angle_minus_pi_six_l770_770566

theorem sin_double_angle_minus_pi_six (α : ℝ) (h : sin (α + π / 6) = 1 / 3) : 
  sin (2 * α - π / 6) = -7/9 := 
by 
  sorry

end sin_double_angle_minus_pi_six_l770_770566


namespace distance_traveled_l770_770206

theorem distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 12) : D = 100 := 
sorry

end distance_traveled_l770_770206


namespace relation_A_B_l770_770715

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

def A (b c : ℝ) : set ℝ := {x | f x b c = x}

def B (b c : ℝ) : set ℝ := {x | f (f x b c) b c = x}

theorem relation_A_B (b c : ℝ) (hA : ∃! x, x ∈ A b c) : A b c = B b c :=
sorry

end relation_A_B_l770_770715


namespace sequence_infinite_common_terms_l770_770577

noncomputable def sequence_a (a1 : ℕ) : ℕ → ℕ
| 0     := a1
| (n+1) := sequence_a n + (sequence_a n % 10)

def sequence_2 : ℕ → ℕ
| n := 2^n

theorem sequence_infinite_common_terms (a1 : ℕ) (h : ¬(a1 % 5 = 0)) :
  ∃ infinitely_many n, sequence_a a1 n = sequence_2 n := 
sorry

end sequence_infinite_common_terms_l770_770577


namespace base7_to_base10_l770_770818

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770818


namespace acute_angle_at_3_18_l770_770020

def degrees_of_hand_position (hour marks min marks: ℕ) : ℝ :=
  (hour marks % 12) * 30 + (min marks / 60) * 30

def minute_hand_position (min marks: ℕ) : ℝ :=
  min marks * 6

def diff_positions (pos1 pos2 : ℝ) : ℝ :=
  if pos1 > pos2 then pos1 - pos2 else pos2 - pos1

theorem acute_angle_at_3_18 :
  let hour_pos := degrees_of_hand_position 3 (18 : ℕ),
  let min_pos := minute_hand_position (18 : ℕ),
  diff_positions hour_pos min_pos = 9 :=
by {
  sorry
}

end acute_angle_at_3_18_l770_770020


namespace healthcare_deduction_percentage_is_correct_l770_770765

-- Definitions based on the given problem conditions
def annual_salary : ℝ := 40000
def tax_percentage : ℝ := 0.20
def union_dues : ℝ := 800
def take_home_pay : ℝ := 27200

-- Calculation of amounts lost as per solution steps
def tax_deduction : ℝ := tax_percentage * annual_salary
def remaining_after_taxes_union : ℝ := annual_salary - tax_deduction - union_dues
def healthcare_deduction : ℝ := remaining_after_taxes_union - take_home_pay
def healthcare_percentage (total_salary healthcare_cost : ℝ) : ℝ := (healthcare_cost / total_salary) * 100

-- Statement to prove that the deduction percentage for healthcare is 10%
theorem healthcare_deduction_percentage_is_correct :
  healthcare_percentage annual_salary healthcare_deduction = 10 :=
by sorry

end healthcare_deduction_percentage_is_correct_l770_770765


namespace cyclic_quadrilateral_l770_770703

open EuclideanGeometry

variables {Γ1 Γ2 : Circle} {P Q X : Point}
variables {A B C D : Point}
variables {d1 d2 : Line}

-- Conditions
def condition1 := intersect_at Γ1 Γ2 P Q
def condition2 := on_line_segment P Q X
def condition3 := passes_through d1 X ∧ intersects_at d1 Γ1 A B
def condition4 := passes_through d2 X ∧ intersects_at d2 Γ2 C D

-- Proof Problem Statement
theorem cyclic_quadrilateral (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : cyclic_quadrilateral A B C D :=
sorry

end cyclic_quadrilateral_l770_770703


namespace base_3_is_most_economical_l770_770648

theorem base_3_is_most_economical (m d : ℕ) (h : d ≥ 1) (h_m_div_d : m % d = 0) :
  3^(m / 3) ≥ d^(m / d) :=
sorry

end base_3_is_most_economical_l770_770648


namespace old_pump_fill_time_l770_770427

/-- Representing the condition that the newer pump takes 200 seconds to fill the trough, and it takes 150 seconds to fill the trough using both pumps, 
we aim to prove that the old pump alone takes 600 seconds to fill the trough. -/
theorem old_pump_fill_time :
  let T := 600
  in (1 / T) + (1 / 200) = (1 / 150) :=
by
  sorry

end old_pump_fill_time_l770_770427


namespace sum_scores_may_not_reach_100_l770_770059

-- Defining the teams and the points they can score.
structure Tournament :=
  (teams : Fin 10 → ℕ)
  (points_win : ℕ := 3)
  (points_draw : ℕ := 1)

-- Condition: Each team plays every other team exactly once, and the possible outcomes are win or draw.
def valid_score (score : ℕ) : Prop :=
  ∃ wins draws, score = 3 * wins + draws

-- Condition: The total number of games is C(10, 2) = 45.
def total_games : ℕ := (10 * 9) / 2

-- Defining the sum of all scores
def total_score (t : Tournament) : ℕ :=
  ∑ team in (Finset.fin 10), t.teams team

-- The scoring constraint, derived from the round-robin tournament.
def valid_scores (t : Tournament) : Prop :=
  ∀ team, valid_score (t.teams team)

-- Our theorem: The sum of the scores must be at least 100 - prove that it is not necessarily true.
theorem sum_scores_may_not_reach_100 : 
  ∃ t : Tournament, valid_scores t ∧ total_score t < 100 :=
by
  sorry

end sum_scores_may_not_reach_100_l770_770059


namespace job_time_relation_l770_770871

theorem job_time_relation (a b c m n x : ℝ) 
  (h1 : m / a = 1 / b + 1 / c)
  (h2 : n / b = 1 / a + 1 / c)
  (h3 : x / c = 1 / a + 1 / b) :
  x = (m + n + 2) / (m * n - 1) := 
sorry

end job_time_relation_l770_770871


namespace polynomial_roots_quartic_sum_l770_770267

noncomputable def roots_quartic_sum (a b c : ℂ) : ℂ :=
  a^4 + b^4 + c^4

theorem polynomial_roots_quartic_sum :
  ∀ (a b c : ℂ), (a^3 - 3 * a + 1 = 0) ∧ (b^3 - 3 * b + 1 = 0) ∧ (c^3 - 3 * c + 1 = 0) →
  (a + b + c = 0) ∧ (a * b + b * c + c * a = -3) ∧ (a * b * c = -1) →
  roots_quartic_sum a b c = 18 :=
by
  intros a b c hroot hsum
  sorry

end polynomial_roots_quartic_sum_l770_770267


namespace orange_juice_count_l770_770056

def club_members : ℕ := 30
def lemon_ratio : ℚ := 2/5
def mango_ratio : ℚ := 1/3

theorem orange_juice_count :
  let lemon_count := (lemon_ratio * club_members : ℚ).toNat
  let remaining_after_lemon := club_members - lemon_count
  let mango_count := (mango_ratio * remaining_after_lemon : ℚ).toNat
  let orange_count := club_members - (lemon_count + mango_count)
  orange_count = 12 :=
by
  -- The proof itself is skipped according to the instructions
  sorry

end orange_juice_count_l770_770056


namespace quadratic_has_distinct_real_roots_l770_770534

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770534


namespace jane_vases_per_day_l770_770245

theorem jane_vases_per_day : 
  ∀ (total_vases : ℝ) (days : ℝ), 
  total_vases = 248 → days = 16 → 
  (total_vases / days) = 15.5 :=
by
  intros total_vases days h_total_vases h_days
  rw [h_total_vases, h_days]
  norm_num

end jane_vases_per_day_l770_770245


namespace number_of_non_empty_proper_subsets_of_A_range_of_m_if_A_superset_B_l770_770622

section
variable (A : Set ℤ) (B : Set ℝ) (m : ℝ)

-- Conditions for the sets A and B
def setA : Set ℤ := {x | -2 ≤ x ∧ x ≤ 5}
def setB : Set ℝ := {x | x^2 - 3 * m * x + 2 * m^2 - m - 1 < 0}

-- Proof Statement (1)
theorem number_of_non_empty_proper_subsets_of_A :
  A = setA → 
  (2 ^ (Finset.card (setA ∩ set.univ)) - 2) = 254 := 
by intros hA; sorry

-- Proof Statement (2)
theorem range_of_m_if_A_superset_B :
  (setA : Set ℝ) ⊇ setB → 
  (-1 ≤ m ∧ m ≤ 2) ∨ m = -2 := 
by intros hAB; sorry

end

end number_of_non_empty_proper_subsets_of_A_range_of_m_if_A_superset_B_l770_770622


namespace catherine_bottle_caps_l770_770440

-- Definitions from conditions
def friends : ℕ := 6
def caps_per_friend : ℕ := 3

-- Theorem statement from question and correct answer
theorem catherine_bottle_caps : friends * caps_per_friend = 18 :=
by sorry

end catherine_bottle_caps_l770_770440


namespace total_garbage_collected_l770_770731

def pounds_of_garbage : ℕ := 387
def pounds_less (pounds: ℕ) : ℕ := pounds - 39
def ounces_of_garbage : ℕ := 560
def ounces_to_pounds (ounces: ℕ) : ℕ := ounces / 16

theorem total_garbage_collected:
  let lizzie_group := pounds_of_garbage in
  let second_group := pounds_less lizzie_group in
  let third_group := ounces_to_pounds ounces_of_garbage in
  lizzie_group + second_group + third_group = 770 :=
by
  let lizzie_group := pounds_of_garbage
  let second_group := pounds_less lizzie_group
  let third_group := ounces_to_pounds ounces_of_garbage
  have h1: lizzie_group = 387 := rfl
  have h2: second_group = 348 := rfl
  have h3: third_group = 35 := rfl
  calc
    lizzie_group + second_group + third_group
        = 387 + 348 + 35 : by rw [h1, h2, h3]
    ... = 770 : by norm_num

end total_garbage_collected_l770_770731


namespace quadratic_has_distinct_real_roots_l770_770536

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770536


namespace count_of_quadrilaterals_with_incenter_l770_770969

def is_incenter_of_square (q : Type) : Prop :=
∃ c : Type, true  -- assuming trivially every square has incenter

def is_incenter_of_rectangle (q : Type) (h₁ : ¬ q = square) : Prop :=
∃ c : Type, true  -- assuming trivially every rectangle has incenter

def is_incenter_of_rhombus (q : Type) (h₁ : ¬ q = square) : Prop :=
∃ c : Type, true  -- assuming trivially every rhombus has incenter

def is_incenter_of_kite (q : Type) (h₁ : ¬ q = rhombus) : Prop :=
∃ c : Type, false  -- not all kites have an incenter

def is_incenter_of_general_quadrilateral (q : Type) : Prop :=
∃ c : Type, false  -- generally do not have an incenter

noncomputable def count_incenter_quadrilaterals : ℕ :=
  (if is_incenter_of_square square then 1 else 0) +
  (if is_incenter_of_rectangle rectangle square then 1 else 0) +
  (if is_incenter_of_rhombus rhombus square then 1 else 0) +
  (if is_incenter_of_kite kite rhombus then 1 else 0) +
  (if is_incenter_of_general_quadrilateral general_quadrilateral then 1 else 0)

theorem count_of_quadrilaterals_with_incenter :
  count_incenter_quadrilaterals = 3 :=
by
  -- proof would go here
  sorry

end count_of_quadrilaterals_with_incenter_l770_770969


namespace snail_returns_after_whole_number_of_hours_l770_770066

-- Definitions of conditions
def constant_speed (position : ℕ → ℝ × ℝ) : Prop := ∀ n, dist (position n) (position (n+1)) = dist (position 0) (position 1)
def right_angle_turns_every_15_minutes (position : ℕ → ℝ × ℝ) : Prop := ∀ n, 
  dist (position (4*n)) (position 0) = dist (position (4*n + 1)) (position 1)
  ∧ dist (position (4*n + 1)) (position 1) = dist (position (4*n + 2)) (position 2)
  ∧ dist (position (4*n + 2)) (position 2) = dist (position (4*n + 3)) (position 3)
  ∧ dist (position (4*n + 3)) (position 3) = dist (position (4*n + 4)) (position 4)

-- Statement of the Theorem
theorem snail_returns_after_whole_number_of_hours (position : ℕ → ℝ × ℝ) 
    (h1 : constant_speed position) (h2 : right_angle_turns_every_15_minutes position) : 
    ∃ k : ℕ, k * 60 = 15 * (num_steps position) := 
sorry

-- Additional definitions as necessary
noncomputable def num_steps (position: ℕ → ℝ × ℝ) : ℕ := sorry

end snail_returns_after_whole_number_of_hours_l770_770066


namespace base_conversion_problem_l770_770770

theorem base_conversion_problem :
  ∃ b : ℕ, b > 0 ∧ ((3 * 4 + 2) = (1 * b^2 + 2 * b)) ∧ b = 2 :=
begin
  use 2,
  split,
  { norm_num, },
  split,
  { sorry, },
  { refl, },
end

end base_conversion_problem_l770_770770


namespace sum_cubes_mod_five_l770_770494

theorem sum_cubes_mod_five :
  (∑ n in Finset.range 50, (n + 1)^3) % 5 = 0 :=
by
  sorry

end sum_cubes_mod_five_l770_770494


namespace sum_of_cubes_mod_l770_770476

theorem sum_of_cubes_mod (S : ℕ) (h : S = ∑ n in Finset.range 51, n^3) : S % 5 = 0 :=
sorry

end sum_of_cubes_mod_l770_770476


namespace difference_in_tiles_l770_770416

-- Definition of side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Theorem stating the difference in tiles between the 10th and 9th squares
theorem difference_in_tiles : (side_length 10) ^ 2 - (side_length 9) ^ 2 = 19 := 
by {
  sorry
}

end difference_in_tiles_l770_770416


namespace combined_volume_of_original_and_reflected_cube_is_5_over_4_l770_770751

-- Define the original cube
def cube := {vertices : list (ℝ × ℝ × ℝ) // vertices.length = 8 ∧
                                    ∀ v ∈ vertices, ∃ x y z, v = (x, y, z) ∧ x ∈ {0, 1} ∧ y ∈ {0, 1} ∧ z ∈ {0, 1}}

-- Define the condition that the vertices are reflected along one of its space diagonals.
def reflect_along_diagonal (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := v in (1 - x, 1 - y, 1 - z)

def reflected_cube (c : cube) : cube :=
  { vertices := c.val.map reflect_along_diagonal,
    property := by {
      simp [reflect_along_diagonal],
      sorry -- Proof of the reflection producing 8 vertices still aligned with cube constraints
    }
  }

-- The statement that proves the combined volume of the original and reflected cube.
theorem combined_volume_of_original_and_reflected_cube_is_5_over_4 (c : cube) :
  let original_volume := 1 in
  let reflected_volume := 1 / 4 in -- This needs to be accurately calculated based on reflection
  original_volume + reflected_volume = 5 / 4 :=
sorry

end combined_volume_of_original_and_reflected_cube_is_5_over_4_l770_770751


namespace quadratic_has_distinct_real_roots_l770_770532

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770532


namespace domain_f_over_x_minus_4_l770_770773

variable {f : ℝ → ℝ}

-- condition: domain of f(x) is [0, 8]
axiom dom_f : ∀ x, 0 ≤ x ∧ x ≤ 8 → ∃ y, f x = y

def domain_2x (x : ℝ) : Prop :=
  0 ≤ 2*x ∧ 2*x ≤ 8

def f_over_x_minus_4 (x : ℝ) : ℝ :=
  f (2*x) / (x - 4)

theorem domain_f_over_x_minus_4 : ∀ x, domain_2x x ∧ x ≠ 4 ↔ 0 ≤ x ∧ x < 4 :=
by
  sorry

end domain_f_over_x_minus_4_l770_770773


namespace center_on_y_eq_x_l770_770264

variable (t : ℝ) (h_t : 0 < t)

def R (t : ℝ) : ℝ := t - 2 * (int.floor (t / 2))

def P (R : ℝ) : set (ℝ × ℝ) := { p | (p.1 - R) ^ 2 + (p.2 - R) ^ 2 ≤ R ^ 2 }

theorem center_on_y_eq_x (t : ℝ) (h_t : 0 < t) : ∃ (R : ℝ), (R = t - 2 * (int.floor (t / 2))) ∧ ∀ (R : ℝ), (R = t - 2 * (int.floor (t / 2))) → (R, R) ∈ P R :=
by
  sorry

end center_on_y_eq_x_l770_770264


namespace missed_angle_l770_770338

theorem missed_angle (n : ℕ) (h1 : (n - 2) * 180 ≥ 3239) (h2 : n ≥ 3) : 3240 - 3239 = 1 :=
by
  sorry

end missed_angle_l770_770338


namespace prime_factors_power_l770_770195

-- Given conditions
def a_b_c_factors (a b c : ℕ) : Prop :=
  (∀ x, x = a ∨ x = b ∨ x = c → Prime x) ∧
  a < b ∧ b < c ∧ a * b * c ∣ 1998

-- Proof problem
theorem prime_factors_power (a b c : ℕ) (h : a_b_c_factors a b c) : (b + c) ^ a = 1600 := 
sorry

end prime_factors_power_l770_770195


namespace time_per_lice_check_l770_770794

-- Define the number of students in each grade
def kindergartners := 26
def first_graders := 19
def second_graders := 20
def third_graders := 25

-- Define the total number of students
def total_students := kindergartners + first_graders + second_graders + third_graders

-- Define the total time in minutes
def hours := 3
def minutes_per_hour := 60
def total_minutes := hours * minutes_per_hour

-- Define the correct answer for time per check
def time_per_check := total_minutes / total_students

-- Prove that the time for each check is 2 minutes
theorem time_per_lice_check : time_per_check = 2 := 
by
  sorry

end time_per_lice_check_l770_770794


namespace det_B_pow4_l770_770975

variable (B : Matrix n n ℝ) -- Assuming we are working with real matrices

theorem det_B_pow4 (h : det B = -3) : det (B ^ 4) = 81 :=
by sorry

end det_B_pow4_l770_770975


namespace lines_condition_l770_770430

-- Assume x and y are real numbers representing coordinates on the lines l1 and l2
variables (x y : ℝ)

-- Points on the lines l1 and l2 satisfy the condition |x| - |y| = 0.
theorem lines_condition (x y : ℝ) (h : abs x = abs y) : abs x - abs y = 0 :=
by
  sorry

end lines_condition_l770_770430


namespace smallest_positive_perfect_cube_l770_770262

theorem smallest_positive_perfect_cube (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ m : ℕ, m = (a * b * c^2)^3 ∧ (a^2 * b^3 * c^5 ∣ m)
:=
sorry

end smallest_positive_perfect_cube_l770_770262


namespace instantaneous_velocity_at_3_seconds_l770_770084

def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

def velocity (t : ℝ) : ℝ := (derivative motion_equation) t

theorem instantaneous_velocity_at_3_seconds : velocity 3 = 5 := 
by
  sorry

end instantaneous_velocity_at_3_seconds_l770_770084


namespace monotonically_decreasing_iff_l770_770611

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a^x

theorem monotonically_decreasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a y ≤ f a x) ↔ (3/8 ≤ a ∧ a < 2/3) :=
sorry

end monotonically_decreasing_iff_l770_770611


namespace initial_percentage_of_grape_juice_l770_770647

theorem initial_percentage_of_grape_juice (P : ℝ) 
  (h₀ : 10 + 30 = 40)
  (h₁ : 40 * 0.325 = 13)
  (h₂ : 30 * P + 10 = 13) : 
  P = 0.1 :=
  by 
    sorry

end initial_percentage_of_grape_juice_l770_770647


namespace percent_students_prefer_golf_l770_770326

theorem percent_students_prefer_golf (students_north : ℕ) (students_south : ℕ)
  (percent_golf_north : ℚ) (percent_golf_south : ℚ) :
  students_north = 1800 →
  students_south = 2200 →
  percent_golf_north = 15 →
  percent_golf_south = 25 →
  (820 / 4000 : ℚ) = 20.5 :=
by
  intros h_north h_south h_percent_north h_percent_south
  sorry

end percent_students_prefer_golf_l770_770326


namespace base7_to_base10_l770_770819

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770819


namespace correct_statements_l770_770162

variables (a b : Line) (alpha beta : Plane)

-- Given:
-- 1. a perpendicular to b
-- 2. a perpendicular to alpha
-- 3. b not contained in alpha
def statement_2_correct (a_perp_b : a ⟂ b) (a_perp_alpha : a ⟂ α) (b_not_in_alpha : ¬ b ⊂ α) : Prop :=
  b ∥ α

-- Given:
-- 1. alpha perpendicular to beta
-- 2. a perpendicular to alpha
-- 3. b perpendicular to beta
def statement_3_correct (alpha_perp_beta : α ⟂ β) (a_perp_alpha : a ⟂ α) (b_perp_beta : b ⟂ β) : Prop :=
  a ⟂ b

theorem correct_statements (a_perp_b : a ⟂ b) (a_perp_alpha : a ⟂ α) (b_not_in_alpha : ¬ b ⊂ α)
                          (alpha_perp_beta : α ⟂ β) (b_perp_beta : b ⟂ β) : 
  statement_2_correct a_perp_b a_perp_alpha b_not_in_alpha ∧
  statement_3_correct alpha_perp_beta a_perp_alpha b_perp_beta :=
sorry

end correct_statements_l770_770162


namespace number_of_positive_integer_solutions_l770_770789

theorem number_of_positive_integer_solutions (x : ℕ) (hx : 0 < x) : x < 3 → x ∈ {1, 2} :=
by
  intro h
  fin_cases x <;> try {linarith}

#check number_of_positive_integer_solutions -- (x : ℕ) (hx : x > 0) → x < 3 → x ∈ {1, 2}

end number_of_positive_integer_solutions_l770_770789


namespace exists_line_in_plane_perpendicular_l770_770143

-- Defining the types for plane and line.
variable (α : Type) -- Type for plane
variable (a : Type) -- Type for line

-- Assuming conditions needed
variable [Plane α]
variable [Line a]

-- Statement of the theorem we want to prove
theorem exists_line_in_plane_perpendicular (h_plane : Plane α) (h_line : Line a) : 
  ∃ l : Line α, l ⊥ a :=
sorry

end exists_line_in_plane_perpendicular_l770_770143


namespace third_and_fourth_smallest_primes_greater_than_20_product_l770_770642

theorem third_and_fourth_smallest_primes_greater_than_20_product :
  let primes := [23, 29, 31, 37].nth 2 = some 31 ∧ [23, 29, 31, 37].nth 3 = some 37 in
  31 * 37 = 1147 :=
by
  -- Proof would go here
  sorry

end third_and_fourth_smallest_primes_greater_than_20_product_l770_770642


namespace average_angle_red_blue_vectors_l770_770564

theorem average_angle_red_blue_vectors 
  (N : ℕ) (hN : N > 1) -- N > 1 ensures there are at least two vectors
  (red blue : set ℕ) -- subsets red and blue representing indices of vectors
  (h : ∀ (i : ℕ), i ∈ red ∪ blue) -- all indices are either red or blue
  (disjoint_red_blue : disjoint red blue) -- no vector is both red and blue
  : ∑ (i in red) (j in blue), (360 / N) / (card red * card blue) = 180 :=
sorry

end average_angle_red_blue_vectors_l770_770564


namespace total_percentage_of_pears_thrown_away_l770_770075

/-- 
A vendor sells 20 percent of the pears he had on Monday and throws away 50 percent of 
the remaining pears. On Tuesday, the vendor sells 30 percent of the remaining pears and 
throws away 20 percent of what is left. On Wednesday, the vendor sells 15 percent of the 
remaining pears and throws away 35 percent of what is then left. Finally, on Thursday, 
the vendor sells 10 percent of the remaining pears but has to throw away 40 percent of 
what is left due to spoilage. Prove an equation to find the total percentage of pears 
thrown away by the vendor.
-/
theorem total_percentage_of_pears_thrown_away :
  let P := 1.0 in -- Initial total number of pears normalized to 1.0 for percentage calculation
  let remaining_after_monday := 0.5 * (0.8 * P) in
  let remaining_after_tuesday := 0.8 * (0.7 * remaining_after_monday) in
  let remaining_after_wednesday := 0.65 * (0.85 * remaining_after_tuesday) in
  let remaining_after_thursday := 0.6 * (0.9 * remaining_after_wednesday) in
  let thrown_away_on_monday := P - (0.5 * (0.8 * P)) in
  let thrown_away_on_tuesday := remaining_after_monday - remaining_after_tuesday in
  let thrown_away_on_wednesday := remaining_after_tuesday - (0.65 * (0.85 * remaining_after_tuesday)) in
  let thrown_away_on_thursday := remaining_after_wednesday - (0.6 * (0.9 * remaining_after_wednesday)) in
  let total_percentage_thrown_away := (thrown_away_on_monday + thrown_away_on_tuesday + 
                                       thrown_away_on_wednesday + thrown_away_on_thursday) / P 
  in
  total_percentage_thrown_away = 0.5671936 := 
by
  sorry

end total_percentage_of_pears_thrown_away_l770_770075


namespace quadratic_two_distinct_real_roots_l770_770519

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770519


namespace math_problem_l770_770589

-- Definitions for points and the given conditions
def point := ℝ × ℝ
def A : point := (-1, 2)
def B : point := (0, 1)
def l1 (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0
def dist (P Q : point) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
def on_curve_C (P : point) : Prop := (P.1 - 1)^2 + P.2^2 = 4

-- Condition for the moving point P
def condition_P (P : point) : Prop :=
  dist P A = real.sqrt 2 * dist P B

-- Statement for the equivalent proof problem
theorem math_problem :
  ∀ (P : point), condition_P P → on_curve_C P ∧
  ∀ (Q : point), l1 Q.1 Q.2 → (∃! (M : point), on_curve_C M ∧ Q.1 = M.1 ∧ Q.2 = M.2) →
  dist Q (classical.some (exists_unique.exists (∃! (M : point), on_curve_C M ∧ Q.1 = M.1 ∧ Q.2 = M.2))) = real.sqrt 5 :=
by
  sorry

end math_problem_l770_770589


namespace count_equilateral_triangles_l770_770107

-- Define the structure of a point in a hexagonal lattice
structure HexagonalLatticePoint where
  x : ℝ
  y : ℝ

-- Define the conditions
def is_unit_distance (p1 p2 : HexagonalLatticePoint) : Prop :=
  dist p1 p2 = 1

def is_two_units_distance (p1 p2 : HexagonalLatticePoint) : Prop :=
  dist p1 p2 = 2

-- Define the hexagon properties
def original_hexagon (p : HexagonalLatticePoint) : Prop :=
  -- insert the actual condition to identify original hexagon points

def second_hexagon (p : HexagonalLatticePoint) : Prop :=
  -- insert the actual condition to identify second hexagon points

-- Define what it means to be an equilateral triangle in this context
def is_equilateral_triangle (p1 p2 p3 : HexagonalLatticePoint) : Prop :=
  dist p1 p2 = dist p2 p3 ∧ dist p2 p3 = dist p3 p1

-- The main theorem to prove
theorem count_equilateral_triangles : 
  (∀ p ∈ original_hexagon,
    (∀ q ∈ second_hexagon,
      (∃ r ∈ second_hexagon, is_equilateral_triangle p q r))) →
  (∑ (t ∈ equilateral_triangles), 1 = 18) := sorry

end count_equilateral_triangles_l770_770107


namespace positive_difference_of_complementary_angles_l770_770781

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l770_770781


namespace arithmetic_sequence_value_l770_770668

theorem arithmetic_sequence_value :
  (∀ n : ℕ, 0 < n →
    a (n : ℤ) + 2 * a (n + 1 : ℤ) + 3 * a (n + 2 : ℤ) = 6 * (n : ℤ) + 22) →
  a 2017 = 6058 / 3 :=
sorry

end arithmetic_sequence_value_l770_770668


namespace complementary_angle_difference_l770_770783

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l770_770783


namespace exists_induced_subgraph_l770_770148

theorem exists_induced_subgraph
  (G : Type) [Graph G]
  (e n : ℕ)
  (d : Fin n → ℕ)
  (k : ℕ)
  (h_lt : k < Finset.min' (Finset.image (λ i, d i) (Finset.univ : Finset (Fin n))) (by sorry)) :
  ∃ (H : Subgraph G), 
    (∀ v w, H.adj v w → G.adj v w ∧ v ≠ w) ∧
    ¬ (∃ (f : Fin (k + 1) → H.vertices), ∀ i j, i ≠ j → H.adj (f i) (f j)) ∧
    H.vertices.card ≥ (k * n^2) / (2 * e + n) :=
sorry

end exists_induced_subgraph_l770_770148


namespace sector_area_proof_l770_770597

/--
Given:
- The circumference of a sector is 8 cm.
- The central angle of the sector is 2 radians.

Prove:
- The area of the sector is 4 cm².
-/
theorem sector_area_proof (R : ℝ) (α : ℝ) (l : ℝ) (C : ℝ) (S : ℝ) (hα : α = 2)
  (hl : l = R * α) (hC : C = l + 2 * R) (hC8 : C = 8) : S = 1/2 * l * R :=
begin
  -- The area of the sector is given by the formula:
  have h1 : S = 1/2 * l * R, from sorry,
  -- Given the conditions, prove the area is 4 cm²
  show S = 4, from sorry,
end

end sector_area_proof_l770_770597


namespace quadratic_has_distinct_real_roots_l770_770531

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770531


namespace geometric_inequality_l770_770769

-- Define the geometric setup
variables {A B C H B1 C1 N M Ob Oc : ℝ} -- representing points in ℝ

-- Conditions from the problem statement
def altitude_intersect (ABC : Triangle) (H : Point) :=
  is_altitude ABC B H ∧ is_altitude ABC C H

def circle_passing_points (O : Point) (A C1 N : Point) :=
  is_circle O [A, C1, N]

def midpoint (X Y M : Point) :=
  2 * (line_segment X Y) = line_segment X M + line_segment M Y

-- Main theorem statement
theorem geometric_inequality 
  (ABC : Triangle) 
  (H : Point) 
  (Ob : Point) 
  (Oc : Point)
  (B1 : Point) 
  (C1 : Point)
  (N : Point) 
  (M : Point) 
  (altitudes : altitude_intersect ABC H)
  (circle_ob : circle_passing_points Ob A C1 N)
  (circle_oc : circle_passing_points Oc A B1 M)
  (midpt_BH : midpoint B H N)
  (midpt_CH : midpoint C H M) :
  (distance B1 Ob + distance C1 Oc) > (distance (point_of_line_segment B C) * (4⁻¹)) :=
begin
  sorry,
end

end geometric_inequality_l770_770769


namespace common_ratio_l770_770333

theorem common_ratio (a S r : ℝ) (h1 : S = a / (1 - r))
  (h2 : ar^5 / (1 - r) = S / 81) : r = 1 / 3 :=
sorry

end common_ratio_l770_770333


namespace point_on_curve_iff_l770_770043

def is_on_curve (f : ℝ → ℝ → ℝ) (x₀ y₀ : ℝ) : Prop :=
  f x₀ y₀ = 0

theorem point_on_curve_iff (f : ℝ → ℝ → ℝ) (x₀ y₀ : ℝ) :
  (is_on_curve f x₀ y₀) ↔ (f x₀ y₀ = 0) :=
begin
  split;
  intro h,
  { exact h },
  { exact h },
end

end point_on_curve_iff_l770_770043


namespace maxAdditionalTiles_l770_770312

-- Board definition
structure Board where
  width : Nat
  height : Nat
  cells : List (Nat × Nat) -- List of cells occupied by tiles

def initialBoard : Board := 
  ⟨10, 9, [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2),
            (6,1), (6,2), (7,1), (7,2)]⟩

-- Function to count cells occupied
def occupiedCells (b : Board) : Nat :=
  b.cells.length

-- Function to calculate total cells in a board
def totalCells (b : Board) : Nat :=
  b.width * b.height

-- Function to calculate additional 2x1 tiles that can be placed
def additionalTiles (board : Board) : Nat :=
  (totalCells board - occupiedCells board) / 2

theorem maxAdditionalTiles : additionalTiles initialBoard = 36 := by
  sorry

end maxAdditionalTiles_l770_770312


namespace sabina_loan_l770_770754

-- Define the conditions
def tuition_per_year : ℕ := 30000
def living_expenses_per_year : ℕ := 12000
def duration : ℕ := 4
def sabina_savings : ℕ := 10000
def grant_first_two_years_percent : ℕ := 40
def grant_last_two_years_percent : ℕ := 30
def scholarship_percent : ℕ := 20

-- Calculate total tuition for 4 years
def total_tuition : ℕ := tuition_per_year * duration

-- Calculate total living expenses for 4 years
def total_living_expenses : ℕ := living_expenses_per_year * duration

-- Calculate total cost
def total_cost : ℕ := total_tuition + total_living_expenses

-- Calculate grant coverage
def grant_first_two_years : ℕ := (grant_first_two_years_percent * tuition_per_year / 100) * 2
def grant_last_two_years : ℕ := (grant_last_two_years_percent * tuition_per_year / 100) * 2
def total_grant_coverage : ℕ := grant_first_two_years + grant_last_two_years

-- Calculate scholarship savings
def annual_scholarship_savings : ℕ := living_expenses_per_year * scholarship_percent / 100
def total_scholarship_savings : ℕ := annual_scholarship_savings * (duration - 1)

-- Calculate total reductions
def total_reductions : ℕ := total_grant_coverage + total_scholarship_savings + sabina_savings

-- Calculate the total loan needed
def total_loan_needed : ℕ := total_cost - total_reductions

theorem sabina_loan : total_loan_needed = 108800 := by
  sorry

end sabina_loan_l770_770754


namespace number_of_valid_three_digit_numbers_l770_770277

theorem number_of_valid_three_digit_numbers :
  ∃ count : ℕ, count = 7 ∧ (∀ A B C : ℕ, (A∈ {2, 3, 5, 7}) → (C ∈ {2, 3, 5, 7}) → (B = (A + C) / 2 ∧ B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → (count = 7)) :=
sorry

end number_of_valid_three_digit_numbers_l770_770277


namespace mike_score_l770_770278

-- Definitions based on the conditions
def max_marks : ℕ := 750
def percentage_to_pass : ℝ := 0.3
def short_by : ℕ := 13

-- Calculation of the required marks to pass based on the given conditions
def required_marks_to_pass : ℕ := (percentage_to_pass * max_marks).to_nat

-- Proof statement
theorem mike_score :
  ∃ M, M = required_marks_to_pass - short_by ∧ M = 212 :=
by
  sorry

end mike_score_l770_770278


namespace sum_of_cubes_roots_poly_l770_770263

theorem sum_of_cubes_roots_poly :
  (∀ (a b c : ℂ), (a^3 - 2*a^2 + 2*a - 3 = 0) ∧ (b^3 - 2*b^2 + 2*b - 3 = 0) ∧ (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5) :=
by
  sorry

end sum_of_cubes_roots_poly_l770_770263


namespace number_of_subsets_is_four_l770_770977

noncomputable def M (a : ℝ) : Set ℝ := {x | x^2 - 3*x - a^2 + 2 = 0}

theorem number_of_subsets_is_four (a : ℝ) : ∃ M, (M = {x | x^2 - 3*x - a^2 + 2 = 0}) ∧ (M.card = 2) → 2^M.card = 4 :=
by
  sorry

end number_of_subsets_is_four_l770_770977


namespace mn_vector_l770_770156

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C M N : V)
variables (a b c : V)

-- Given conditions
def OA_eq_a (h1 : O = 0) : A = a := sorry
def OB_eq_b (h1 : O = 0) : B = b := sorry
def OC_eq_c (h1 : O = 0) : C = c := sorry
def M_on_OA (h1 : O = 0) : ∃ k : ℝ, 0 < k ∧ k < 1 ∧ M = k • O + (1 - k) • A := sorry
def M_ratio (h1 : O = 0) (h2 : M_ratio) : ∃ k : ℝ, OM = 2 * MA := sorry
def N_mid_BC (h1 : O = 0) : N = (1/2) • B + (1/2) • C := sorry

-- The theorem we're trying to prove
theorem mn_vector (h1 : O = 0) (h2 : M_on_OA h1) (h3 : N_mid_BC h1) :
  MN = - (2/3) • a + (1/2) • b + (1/2) • c :=
sorry

end mn_vector_l770_770156


namespace sum_cubes_mod_5_l770_770464

-- Define the function sum of cubes up to n
def sum_cubes (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 1) ^ 3)

-- Define the given modulo operation
def modulo (a b : ℕ) : ℕ := a % b

-- The primary theorem to prove
theorem sum_cubes_mod_5 : modulo (sum_cubes 50) 5 = 0 := sorry

end sum_cubes_mod_5_l770_770464


namespace Hilt_payment_l770_770734

def total_cost : ℝ := 2.05
def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10

theorem Hilt_payment (n : ℕ) (h : n_n = n ∧ n_d = n) 
  (h_nickel : ℝ := n * nickel_value)
  (h_dime : ℝ := n * dime_value): 
  (n * nickel_value + n * dime_value = total_cost) 
  →  n = 14 :=
by {
  sorry
}

end Hilt_payment_l770_770734


namespace ginverse_sum_l770_770269

def g (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 3 * x - x ^ 2

theorem ginverse_sum :
  let g_inv := λ y, if y = -4 then 4 else if y = 0 then 3 else if y = 5 then -2 else 0 in
  g_inv (-4) + g_inv (0) + g_inv (5) = 5 := by
    let g_inv := λ y, if y = -4 then 4 else if y = 0 then 3 else if y = 5 then -2 else 0
    sorry

end ginverse_sum_l770_770269


namespace calc_exp_sqrt_pi_l770_770092

theorem calc_exp_sqrt_pi :
  (1 / 2 : ℝ) ^ (-4) - Real.sqrt 8 + (2023 - Real.pi) ^ 0 = 17 - 2 * Real.sqrt 2 := 
by
  sorry

end calc_exp_sqrt_pi_l770_770092


namespace inscribed_circle_radius_l770_770306

noncomputable def radius_of_inscribed_circle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in 24 / s

theorem inscribed_circle_radius :
  ∀ a b : ℝ, ∀ c = 10, (1/2) * a * b = 24 → radius_of_inscribed_circle a b 10 = 2 :=
by
  intros a b c hyp_area hyp_c
  sorry

end inscribed_circle_radius_l770_770306


namespace largest_fraction_l770_770994

theorem largest_fraction (p q r s : ℕ) (hp : 0 < p) (hpq : p < q) (hqr : q < r) (hrs : r < s) : 
  max (max (max (max (↑(p + q) / ↑(r + s)) (↑(p + s) / ↑(q + r))) 
              (↑(q + r) / ↑(p + s))) 
          (↑(q + s) / ↑(p + r))) 
      (↑(r + s) / ↑(p + q)) = (↑(r + s) / ↑(p + q)) :=
sorry

end largest_fraction_l770_770994


namespace weighted_binary_search_minimizes_cost_l770_770098

-- Conditions
def L : List ℕ := List.range' 1 100
def cost (k : ℕ) : ℕ := k + 1
def w_range : Set ℝ := Set.Ico (1 / 2 : ℝ) 1

-- Weighted binary search function
def weighted_binary_search (L : List ℕ) (value : ℕ) (w : ℝ) : ℕ :=
  let rec search hi lo :=
    if hi < lo then -1 else
    let guess := (w * lo + (1 - w) * hi).floor
    let mid := L.get! guess
    if mid > value then search (guess - 1) lo
    else if mid < value then search hi (guess + 1)
    else guess
  search (L.length - 1) 0

-- Set S definition
def S : Set ℝ := { w | w ∈ w_range ∧ minimizes_average_cost w }
-- Defines the condition for minimizing average cost
def minimizes_average_cost (w : ℝ) : Prop := sorry -- This needs a precise mathematical definition

-- Lean 4 Proof Statement
theorem weighted_binary_search_minimizes_cost :
  S = Set.Ioc (1 / 2) (74 / 99) → ∃ x, x = 1 / 2 :=
by
  intro h
  use 1 / 2
  sorry

end weighted_binary_search_minimizes_cost_l770_770098


namespace benny_pays_l770_770917

theorem benny_pays (cost_per_lunch : ℕ) (number_of_people : ℕ) (total_cost : ℕ) :
  cost_per_lunch = 8 → number_of_people = 3 → total_cost = number_of_people * cost_per_lunch → total_cost = 24 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end benny_pays_l770_770917


namespace set_of_intersection_points_medians_l770_770968

noncomputable def circle (center : Point) (radius : ℝ) : Set Point := sorry
noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def homothety (center : Point) (point : Point) (coeff : ℝ) : Point := sorry
noncomputable def centroid (A B C : Point) : Point := sorry

theorem set_of_intersection_points_medians (A B : Point) (C : Point) (r : ℝ) (circ_ABC : C ∈ circle (midpoint A B) r)
  (S : Set Point := circle (midpoint A B) (r / 3))
  (images_A_B : Set Point := {homothety (midpoint A B) A (1/3), homothety (midpoint A B) B (1/3)}) : 
  (∀ C ∈ (circle (midpoint A B) r), centroid A B C ∈ S) ∧ ∀ C ∈ S, C ∉ images_A_B :=
sorry

end set_of_intersection_points_medians_l770_770968


namespace arithmetic_sequence_l770_770276

theorem arithmetic_sequence {
  (a_1 : ℝ) (S_4 : ℝ) (d : ℝ) (S_6 : ℝ),
  a_1 = 1 / 2 →
  S_4 = 20 →
  d = 5 / 2 →
  a_6 = a_1 + 5 * d →
  S_6 = 21 :=
begin
  assume a_1_eq : a_1 = 1 / 2,
  assume S_4_eq : S_4 = 20,
  assume d_eq : d = 5 / 2,
  have a_4_eq : a_4 = a_1 + 3 * d,
  have S_4_formula : S_4 = (4 / 2) * (a_1 + a_4),
  rw [a_1_eq, d_eq, a_4_eq] at S_4_formula,
  sorry,
  sorry
}

end arithmetic_sequence_l770_770276


namespace solution_l770_770729

noncomputable def normal_distribution (μ : ℝ) (σ : ℝ) : ℝ → ℝ := sorry

def problem_statement (ξ : ℝ → ℝ) (σ : ℝ) : Prop :=
  (∀ x, ξ x = normal_distribution 0 σ) ∧
  P(λ x, ξ x < -1) = 0.2

theorem solution (σ : ℝ) (ξ : ℝ → ℝ) (h : problem_statement ξ σ) : 
  P(λ x, -1 < ξ x ∧ ξ x < 1) = 0.6 :=
by
  sorry

end solution_l770_770729


namespace distance_between_points_l770_770954

theorem distance_between_points :
  let p1 := (1:ℝ, 2, -5)
  let p2 := (4:ℝ, 10, -2)
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2) = real.sqrt 82 :=
by
  sorry

end distance_between_points_l770_770954


namespace max_experiments_bisection_l770_770052

-- Define the range and accuracy as constants
def temp_lower_bound : ℝ := 60
def temp_upper_bound : ℝ := 81
def accuracy : ℝ := 1

-- Define the initial range
def initial_range : ℝ := temp_upper_bound - temp_lower_bound

-- Maximum required range after accuracy is met
def max_required_range : ℝ := 2 -- Because ±1°C means a range of 2° is acceptable

-- Function to calculate the number of halvings needed to achieve the required accuracy
def number_of_halvings (initial_range : ℝ) (required_range : ℝ) : ℕ :=
  Nat.ceil (Real.logb (2 / required_range) / Real.logb 2)

-- The proof statement
theorem max_experiments_bisection : number_of_halvings initial_range max_required_range = 4 :=
by
  -- Calculation details to be filled in here
  sorry

end max_experiments_bisection_l770_770052


namespace MaireadRan40Miles_l770_770854

def MaireadRanMiles (R : ℝ) (W : ℝ) (J : ℝ) : Prop :=
  W = (3 / 5) * R ∧ J = 3 * R ∧ R + W + J = 184

theorem MaireadRan40Miles : ∃ R W J, MaireadRanMiles R W J ∧ R = 40 :=
by sorry

end MaireadRan40Miles_l770_770854


namespace at_least_one_zero_l770_770606

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem at_least_one_zero (p q : ℝ) (h_zero : ∃ m : ℝ, f m p q = 0 ∧ f (f (f m p q) p q) p q = 0) :
  f 0 p q = 0 ∨ f 1 p q = 0 :=
sorry

end at_least_one_zero_l770_770606


namespace find_f_3_5_l770_770313

-- Define the set of real numbers
variable {ℝ : Type} [Real ℝ]

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Conditions: f is an odd function
def odd_function : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Conditions: f satisfies f(x + 2) = -f(x) for all x in ℝ
def periodicity : Prop :=
  ∀ x : ℝ, f (x + 2) = -f (x)

-- Conditions: f(x) = x for 0 < x < 1
def initial_condition : Prop :=
  ∀ x : ℝ, (0 < x ∧ x < 1) → f (x) = x

-- Main statement to prove f(3.5) = -0.5 given all conditions
theorem find_f_3_5 (h_odd : odd_function f) (h_periodic : periodicity f) (h_initial : initial_condition f) : 
  f 3.5 = -0.5 :=
by 
  sorry

end find_f_3_5_l770_770313


namespace quadratic_distinct_roots_l770_770553

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770553


namespace total_cost_over_8_weeks_l770_770636

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l770_770636


namespace find_valid_number_l770_770952

def is_perfect_square_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 4 ∨ d = 9

def is_valid_number (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧
  (∀ k, k ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → is_perfect_square_digit k) ∧
  (n % 2 = 0) ∧
  (n % 3 = 0) ∧
  (n % 5 = 0) ∧
  (n % 7 = 0)

theorem find_valid_number : ∃ n : ℕ, is_valid_number n ∧ n = 4410 :=
begin
  use 4410,
  split,
  { unfold is_valid_number,
    split,
    { split; linarith, },
    split,
    { intros k hk,
      fin_cases hk,
      repeat { simp [is_perfect_square_digit] }, },
    repeat { norm_num }, },
  { refl, },
end

end find_valid_number_l770_770952


namespace count_valid_grid_arrangements_l770_770950

open Matrix

noncomputable def valid_grid_arrangements : ℕ :=
  42

theorem count_valid_grid_arrangements :
  ∀ (M : Matrix (Fin 3) (Fin 3) ℕ),
  (∀ i j k, i < j → M i k < M j k) ∧ (∀ i j k, k < j → M i k > M i j) →
  (∀ n ∈ Finset.range 9 + 1, ∃ i j, M i j = n) →
  ∃ M, valid_grid_arrangements = 42 :=
by sorry

end count_valid_grid_arrangements_l770_770950


namespace complex_number_quadrant_l770_770792

theorem complex_number_quadrant (i : ℂ) (hi : i^2 = -1) : 
  let z : ℂ := (3 + 4 * i) * i in
  z.re < 0 ∧ z.im > 0 := 
by 
  sorry

end complex_number_quadrant_l770_770792


namespace not_possible_to_cut_rectangular_paper_l770_770367

theorem not_possible_to_cut_rectangular_paper 
  (a : ℝ) (b : ℝ) (s_area : ℝ) (r_area : ℝ) (ratio : ℝ)
  (h1 : s_area = 100) (h2 : r_area = 90) (h3 : ratio = 5 / 3) :
  ¬ (∃ l w, l / w = ratio ∧ l * w = r_area ∧ l ≤ sqrt s_area ∧ w ≤ sqrt s_area) := by
  sorry

end not_possible_to_cut_rectangular_paper_l770_770367


namespace base_seven_to_base_ten_l770_770844

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770844


namespace rowing_speed_in_still_water_l770_770883

theorem rowing_speed_in_still_water (v : ℝ) :
  (∀ (u : ℝ) (d : ℝ),
    u = 1.2 ∧ 2 * d = 7.82 ∧ d / (v - u) + d / (v + u) = 1 →
    v = 7.82) :=
begin
  sorry
end

end rowing_speed_in_still_water_l770_770883


namespace time_to_run_up_and_down_l770_770920

/-- Problem statement: Prove that the time it takes Vasya to run up and down a moving escalator 
which moves upwards is 468 seconds, given these conditions:
1. Vasya runs down twice as fast as he runs up.
2. When the escalator is not working, it takes Vasya 6 minutes to run up and down.
3. When the escalator is moving down, it takes Vasya 13.5 minutes to run up and down.
--/
theorem time_to_run_up_and_down (up_speed down_speed : ℝ) (escalator_speed : ℝ) 
  (h1 : down_speed = 2 * up_speed) 
  (h2 : (1 / up_speed + 1 / down_speed) = 6) 
  (h3 : (1 / (up_speed + escalator_speed) + 1 / (down_speed - escalator_speed)) = 13.5) : 
  (1 / (up_speed - escalator_speed) + 1 / (down_speed + escalator_speed)) * 60 = 468 := 
sorry

end time_to_run_up_and_down_l770_770920


namespace sum_of_x_coordinates_of_intersections_l770_770309

def g : ℝ → ℝ := sorry  -- Definition of g is unspecified but it consists of five line segments.

theorem sum_of_x_coordinates_of_intersections 
  (h1 : ∃ x1, g x1 = x1 - 2 ∧ (x1 = -2 ∨ x1 = 1 ∨ x1 = 4))
  (h2 : ∃ x2, g x2 = x2 - 2 ∧ (x2 = -2 ∨ x2 = 1 ∨ x2 = 4))
  (h3 : ∃ x3, g x3 = x3 - 2 ∧ (x3 = -2 ∨ x3 = 1 ∨ x3 = 4)) 
  (hx1x2x3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  x1 + x2 + x3 = 3 := by
  -- Proof here
  sorry

end sum_of_x_coordinates_of_intersections_l770_770309


namespace easter_egg_problem_l770_770337

-- Define the conditions as assumptions
def total_eggs : Nat := 63
def helen_eggs (H : Nat) := H
def hannah_eggs (H : Nat) := 2 * H
def harry_eggs (H : Nat) := 2 * H + 3

-- The theorem stating the proof problem
theorem easter_egg_problem (H : Nat) (hh : hannah_eggs H + helen_eggs H + harry_eggs H = total_eggs) : 
    helen_eggs H = 12 ∧ hannah_eggs H = 24 ∧ harry_eggs H = 27 :=
sorry -- Proof is omitted

end easter_egg_problem_l770_770337


namespace find_number_l770_770223

theorem find_number (N : ℝ) 
  (h1 : (5 / 6) * N = (5 / 16) * N + 200) : 
  N = 384 :=
sorry

end find_number_l770_770223


namespace min_phi_for_symmetry_l770_770655

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)

theorem min_phi_for_symmetry (φ : ℝ) (hφ : φ > 0) :
  (∀ x : ℝ, f (x - φ) = f (-(x - φ))) → φ = π / 8 :=
sorry

end min_phi_for_symmetry_l770_770655


namespace eccentricity_of_ellipse_l770_770208

theorem eccentricity_of_ellipse (a b c e : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 - b^2)) 
  (h4 : b = Real.sqrt 3 * c) : e = 1/2 :=
by
  sorry

end eccentricity_of_ellipse_l770_770208


namespace DP_eq_sin_half_angle_A_l770_770252

-- Define the basic geometric setup
variables {A B C D K L P O : Point}
variables (circumcircle : Circle)
variables (angle_A_gt_90 : angle A > 90)
variables (on_circumcircle : OnCircumcircle A B C O)
variables (point_D_on_AB : LiesOn D (Segment A B))
variables (AD_eq_AC : Distance(A, D) = Distance(A, C))
variables (AK_diameter : IsDiameter AK circumcircle)
variables (intersection_L : Intersect AK (LineThrough C D) L)
variables (circle_through_DKL : CircleThrough D K L)
variables (intersection_P : IntersectCirclesDifferByK circle_through_DKL circumcircle P)
variables (AK_eq_2 : Distance(A, K) = 2)
variables (angles_eq_10 : angle BCD = 10 ∧ angle BAP = 10)

-- Define the final important proof statement
theorem DP_eq_sin_half_angle_A : Distance(D, P) = Math.sin (angle A / 2) :=
  by
  sorry

end DP_eq_sin_half_angle_A_l770_770252


namespace sum_of_interior_angles_of_decagon_l770_770801

def sum_of_interior_angles_of_polygon (n : ℕ) : ℕ := (n - 2) * 180

theorem sum_of_interior_angles_of_decagon : sum_of_interior_angles_of_polygon 10 = 1440 :=
by
  -- Proof goes here
  sorry

end sum_of_interior_angles_of_decagon_l770_770801


namespace find_smallest_integer_y_l770_770022

theorem find_smallest_integer_y : ∃ y : ℤ, (8 / 12 : ℚ) < (y / 15) ∧ ∀ z : ℤ, z < y → ¬ ((8 / 12 : ℚ) < (z / 15)) :=
by
  sorry

end find_smallest_integer_y_l770_770022


namespace polynomial_roots_correct_l770_770961

noncomputable def roots : Set ℂ := {x | (4 * x^4 + 8 * x^3 - 53 * x^2 + 8 * x + 4 = 0)}

noncomputable def expected_roots : Set ℂ :=
  { (5 + Complex.sqrt 41) / 4, (5 - Complex.sqrt 41) / 4, 
    (-9 + Complex.sqrt 97) / 4, (-9 - Complex.sqrt 97) / 4 }

theorem polynomial_roots_correct :
  roots = expected_roots := by
  sorry

end polynomial_roots_correct_l770_770961


namespace women_in_first_class_equals_22_l770_770115

def number_of_women (total_passengers : Nat) : Nat :=
  total_passengers * 50 / 100

def number_of_women_in_first_class (number_of_women : Nat) : Nat :=
  number_of_women * 15 / 100

theorem women_in_first_class_equals_22 (total_passengers : Nat) (h1 : total_passengers = 300) : 
  number_of_women_in_first_class (number_of_women total_passengers) = 22 :=
by
  sorry

end women_in_first_class_equals_22_l770_770115


namespace max_value_y_l770_770594

/-- Given x < 0, the maximum value of y = (1 + x^2) / x is -2 -/
theorem max_value_y {x : ℝ} (h : x < 0) : ∃ y, y = 1 + x^2 / x ∧ y ≤ -2 :=
sorry

end max_value_y_l770_770594


namespace least_integer_square_eq_double_plus_64_l770_770849

theorem least_integer_square_eq_double_plus_64 :
  ∃ x : ℤ, x^2 = 2 * x + 64 ∧ ∀ y : ℤ, y^2 = 2 * y + 64 → y ≥ x → x = -8 :=
by
  sorry

end least_integer_square_eq_double_plus_64_l770_770849


namespace base_7_to_base_10_l770_770834

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770834


namespace base7_to_base10_l770_770817

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770817


namespace solve_for_k_l770_770649

theorem solve_for_k (k : ℝ) (h : 2 * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : k = 65 := 
by
  sorry

end solve_for_k_l770_770649


namespace find_angle_a_triangle_is_equilateral_l770_770624

-- Given conditions from the problem
variables {A B C a b c : ℝ}
variables (m : ℝ × ℝ) (n : ℝ × ℝ)
hypothesis angle_range: 0 < A ∧ A < Real.pi
hypothesis m_def: m = (2 * b, 1)
hypothesis n_def: n = (c * Real.cos A + a * Real.cos C, Real.cos A)
hypothesis m_parallel_n: m.1 / m.2 = n.1 / n.2
hypothesis area_def: Real.pi * a ^ 2 = 3 * (1 / 2 * b * c * Real.sin A)

-- Proving the value of angle A
theorem find_angle_a : A = Real.pi / 3 :=
by
  sorry

-- Proving that triangle ABC is equilateral
theorem triangle_is_equilateral (h : A = Real.pi / 3) : a = b ∧ b = c :=
by
  sorry

end find_angle_a_triangle_is_equilateral_l770_770624


namespace three_digit_numbers_count_l770_770639

theorem three_digit_numbers_count :
  let is_valid_number (h t u : ℕ) := 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h > t ∧ t > u
  in (∑ h in finset.range 10, ∑ t in finset.range h, ∑ u in finset.range t, if is_valid_number h t u then 1 else 0) = 84 :=
by
  let is_valid_number (h t u : ℕ) := 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h>t ∧ t>u
  have : ∑ h in finset.range 10, ∑ t in finset.range h, ∑ u in finset.range t, if is_valid_number h t u then 1 else 0 = 84 := by {
    -- skipping the proof
    sorry
  }
  exact this

end three_digit_numbers_count_l770_770639


namespace minimum_boat_speed_required_l770_770087

theorem minimum_boat_speed_required 
  (S v : ℝ) 
  (S_pos : 0 < S)
  (v_pos : 0 < v) :
  ∀ x : ℝ, x ≥ (3 * S + real.sqrt(9 * S ^ 2 + 4 * v ^ 2)) / 2 → 
    (S / (x - v) + S / (x + v) + 1 / 12) ≤ 0.75 :=
                                                
   -- sorry provides the necessary placeholder for the proof.
sorry

end minimum_boat_speed_required_l770_770087


namespace find_g_2021_l770_770457

-- Define the function g and the condition it satisfies
theorem find_g_2021 (g : ℝ → ℝ) : 
  (∀ x y : ℝ, g(x - y) = g(x) + g(y) - 2022 * (x + y)) → g(2021) = 4086462 :=
by
  intro h
  sorry

end find_g_2021_l770_770457


namespace distance_blown_by_storm_l770_770740

-- Definitions based on conditions
def speed : ℤ := 30
def time_travelled : ℤ := 20
def distance_travelled := speed * time_travelled
def total_distance := 2 * distance_travelled
def fractional_distance_left := total_distance / 3

-- Final statement to prove
theorem distance_blown_by_storm : distance_travelled - fractional_distance_left = 200 := by
  sorry

end distance_blown_by_storm_l770_770740


namespace find_z_l770_770979

variable (z_1 z_2 z : ℂ)

-- Given conditions
def condition1 : z_1 = 5 + 10 * Complex.i := sorry
def condition2 : z_2 = 3 - 4 * Complex.i := sorry
def condition3 : 1 / z = 1 / z_1 + 1 / z_2 := sorry

-- Statement to prove
theorem find_z : z = 5 - 2.5 * Complex.i :=
by
  apply condition1
  apply condition2
  apply condition3
  sorry

end find_z_l770_770979


namespace negative_number_unique_l770_770907

-- Define the given numbers for clarity
def numbers := {0, 1 / 2, -Real.sqrt 3, 2}

-- State the theorem
theorem negative_number_unique :
  ∃ x ∈ numbers, x < 0 ∧ ∀ y ∈ numbers, y < 0 → y = -Real.sqrt 3 := by
  sorry

end negative_number_unique_l770_770907


namespace value_of_c_distinct_real_roots_l770_770555

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770555


namespace concert_song_count_l770_770249

theorem concert_song_count (total_concert_time : ℕ)
  (intermission_time : ℕ)
  (song_duration_regular : ℕ)
  (song_duration_special : ℕ)
  (performance_time : ℕ)
  (total_songs : ℕ) :
  total_concert_time = 80 →
  intermission_time = 10 →
  song_duration_regular = 5 →
  song_duration_special = 10 →
  performance_time = total_concert_time - intermission_time →
  performance_time - song_duration_special = (total_songs - 1) * song_duration_regular →
  total_songs = 13 :=
begin
  intros h_total h_intermission h_regular h_special h_performance_time h_remaining_songs,
  sorry
end

end concert_song_count_l770_770249


namespace unique_triplet_nat_gt1_l770_770290

theorem unique_triplet_nat_gt1 :
  ∃! (a b c: ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧
  c ∣ (a * b + 1) ∧ b ∣ (a * c + 1) ∧ a ∣ (b * c + 1) :=
  by
  sorry

end unique_triplet_nat_gt1_l770_770290


namespace remainder_is_64_l770_770447
open Set

noncomputable def minimallyIntersectingSets := 
  { (A, B, C : Set ℕ) // 
    |(A ∩ B).card = 1 ∧ (B ∩ C).card = 1 ∧ (C ∩ A).card = 1 ∧ (A ∩ B ∩ C) = ∅ ∧ 
    A ⊆ {1,2,3,4,5,6,7,8} ∧ B ⊆ {1,2,3,4,5,6,7,8} ∧ C ⊆ {1,2,3,4,5,6,7,8}}

def count_mod_1000 : ℕ := 
  (Finset.card minimallyIntersectingSets.val) % 1000

theorem remainder_is_64 : count_mod_1000 = 64 := 
by 
  sorry

end remainder_is_64_l770_770447


namespace min_area_triangle_ABO_min_sum_distances_OA_OB_l770_770991

-- Given conditions
variables {P : Point} (hx : P.x = 1) (hy : P.y = 4)
variable {O : Point} (ho : O.x = 0) (ho : O.y = 0)
variables {A B : Point}

-- Question 1: Prove the minimum area of triangle ABO is 8
theorem min_area_triangle_ABO (L : Line) (hk : L.contains P) 
(intersect_x : ∃ x : ℝ, (x, 0) ∈ L.points) 
(intersect_y : ∃ y : ℝ, (0, y) ∈ L.points)
(min_S : ∀ k : ℝ, y - 4 = k * (x - 1) ∧ k < 0 → (area_triangle A B O) ≥ 8)
: (area_triangle A B O) = 8 ∧ equation_of_line L = "4x + y - 8 = 0" := sorry

-- Question 2: Prove the equation of line L for minimum sum of distances OA and OB
theorem min_sum_distances_OA_OB (L : Line) (hk : L.contains P)
(intersect_x : ∃ x : ℝ, (x, 0) ∈ L.points) 
(intersect_y : ∃ y : ℝ, (0, y) ∈ L.points)
(min_sum_dist : ∀ k : ℝ, y - 4 = k * (x - 1) ∧ k < 0 → (distance O A + distance O B) ≥ 9)
: equation_of_line L = "2x + y - 6 = 0" := sorry

end min_area_triangle_ABO_min_sum_distances_OA_OB_l770_770991


namespace no_non_intersecting_paths_between_houses_and_wells_l770_770242

theorem no_non_intersecting_paths_between_houses_and_wells :
  ¬ ∃ (houses wells : Fin 3 → Point) (paths : Fin 3 × Fin 3 → Path),
    (∀ i j, paths (i, j) ∩ paths (i, j) = ∅) ∧ 
    (∀ i j, connect (houses i) (wells j) (paths (i, j))) ∧ 
    planar (∪ (λ p, paths p)) := sorry

end no_non_intersecting_paths_between_houses_and_wells_l770_770242


namespace hyperbola_equation_and_angles_equal_l770_770211

open Set

def is_hyperbola (x y : ℝ) (a b : ℝ) : Prop := 
  (x^2 / a^2 - y^2 / b^2 = 1)

theorem hyperbola_equation_and_angles_equal 
  (F1 F2 : ℝ × ℝ) (tangent_slope : ℝ) (Q P : ℝ × ℝ) 
  (hx1 : F1 = (-2,0)) (hx2 : F2 = (2,0)) 
  (hq1 : Q = (1 / 2, 0)) (hq2 : tangent_slope = 2) 
  (hp : ∃ x y : ℝ, is_hyperbola x y 1 (sqrt 3) ∧ P = (x, y)) :
  (is_hyperbola (fst P) (snd P) 1 (sqrt 3)) ∧ 
  (∠ F1 P Q = ∠ F2 P Q) :=
sorry

end hyperbola_equation_and_angles_equal_l770_770211


namespace range_of_y_div_x_l770_770701

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + y^2 + 4*x + 3 = 0) :
  - (Real.sqrt 3) / 3 <= y / x ∧ y / x <= (Real.sqrt 3) / 3 :=
sorry

end range_of_y_div_x_l770_770701


namespace identify_genuine_coin_l770_770808

theorem identify_genuine_coin :
  ∃ (genuine : ℕ → Prop) (fake : ℕ → Prop),
    (∀ n, genuine n → fake n → false) ∧
    -- There exists at least one genuine coin
    (∃ n, genuine n) ∧
    -- There are more than 0 and less than 99 counterfeit coins
    (0 < ∑ n, if fake n then 1 else 0) ∧
    (∑ n, if fake n then 1 else 0 < 99) ∧
    -- All genuine coins weigh the same
    (∀ n m, genuine n → genuine m → weight n = weight m) ∧
    -- All fake coins weigh the same
    (∀ n m, fake n → fake m → weight n = weight m) ∧
    -- Genuine coins are heavier than fake coins
    (∀ n, genuine n → ∀ m, fake m → weight n > weight m)  ∧
    -- Given the setup, it is possible to guarantee discovering a genuine coin
    (∃ n, genuine n ∧ weight n = weight (some n' such that genuine n')) :=
sorry

end identify_genuine_coin_l770_770808


namespace rook_placement_exists_l770_770038

-- Define the chessboard and squares
def Chessboard := ℕ → ℕ → Prop    -- A chessboard function mapping coordinates to white (true) or black (false) squares

-- Define conditions on the chessboard
def has_white_square_in_each_column (b : Chessboard) : Prop :=
  ∀ c:ℕ, ∃ r:ℕ, b r c

def has_entirely_white_column (b : Chessboard) : Prop :=
  ∃ c:ℕ, ∀ r:ℕ, b r c

-- Define the rook placement scenario
structure RookPlacement (b : Chessboard) :=
(rooks : list (ℕ × ℕ))    -- List of coordinates where rooks are placed
(h_rooks_on_white : ∀ rook ∈ rooks, (b rook.fst rook.snd)) -- Rooks are placed on white squares
(h_non_attacking : ∀ rook1 rook2 ∈ rooks, rook1 ≠ rook2 → rook1.fst ≠ rook2.fst ∧ rook1.snd ≠ rook2.snd) -- Rooks do not attack each other
(h_nonempty : rooks ≠ []) -- There is at least one rook
(h_threatens_correctly : ∀ (r c : ℕ), b r c → 
  (∃ rook ∈ rooks, rook.snd = c ∧ rook.fst ≠ r → 
   ∃ rook' ∈ rooks, rook'.fst = r ∧ rook'.snd ≠ c)) -- Threatened squares are correctly covered

-- The final theorem stating the existence of such a placement
theorem rook_placement_exists (b : Chessboard) 
  (hw_col : has_white_square_in_each_column b) 
  (hew_col:  has_entirely_white_column b) : 
  ∃ rp : RookPlacement b, 
    true :=
begin
  -- The proof will be here, but for now:
  sorry
end

end rook_placement_exists_l770_770038


namespace inequality_condition_l770_770617

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1 + a) * x - b

theorem inequality_condition (a b: ℝ) (h : ∀ x : ℝ, f a b x ≥ 0) : (b * (a + 1)) / 2 < 3 / 4 := 
sorry

end inequality_condition_l770_770617


namespace base_seven_to_base_ten_l770_770841

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770841


namespace find_CE_l770_770234

-- Definitions for the conditions
variables {A B E C D : Type} [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry E] [euclidean_geometry C] [euclidean_geometry D]

-- The conditions provided in the problem
def right_angled_triangle_AEB : Prop := is_right_angle (∠ ABE) ∧ is_right_angle (∠ AEB) ∧ ∠ AEB = 60
def right_angled_triangle_BCE : Prop := is_right_angle (∠ BCE) ∧ is_right_angle (∠ BEC) ∧ ∠ BEC = 45
def right_angled_triangle_CDE : Prop := is_right_angle (∠ CDE) ∧ is_right_angle (∠ CED) ∧ ∠ CED = 60
def AE_length : Prop := dist A E = 40

-- The theorem we need to prove
theorem find_CE : 
  right_angled_triangle_AEB ∧ 
  right_angled_triangle_BCE ∧ 
  right_angled_triangle_CDE ∧ 
  AE_length 
  → dist C E = 10 * real.sqrt 6 :=
sorry

end find_CE_l770_770234


namespace ratio_of_sides_product_of_areas_and_segments_l770_770184

variable (S S' S'' : ℝ) (a a' : ℝ)

-- Given condition
axiom proportion_condition : S / S'' = a / a'

-- Proofs that need to be verified
theorem ratio_of_sides (S S' : ℝ) (a a' : ℝ) (h : S / S'' = a / a') :
  S / a = S' / a' :=
sorry

theorem product_of_areas_and_segments (S S' : ℝ) (a a' : ℝ) (h: S / S'' = a / a') :
  S * a' = S' * a :=
sorry

end ratio_of_sides_product_of_areas_and_segments_l770_770184


namespace range_of_a1_l770_770329

theorem range_of_a1 (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 1 / (2 - a n)) (h2 : ∀ n, a (n + 1) > a n) :
  a 1 < 1 :=
sorry

end range_of_a1_l770_770329


namespace complex_number_solution_l770_770210

-- Define the statement of the theorem
theorem complex_number_solution (z : ℂ) (h : (3 - 4 * complex.I + z) * complex.I = 2 + complex.I) : z = -2 + 2 * complex.I :=
sorry

end complex_number_solution_l770_770210


namespace sum_of_cubes_mod_l770_770472

theorem sum_of_cubes_mod (S : ℕ) (h : S = ∑ n in Finset.range 51, n^3) : S % 5 = 0 :=
sorry

end sum_of_cubes_mod_l770_770472


namespace cut_rectangle_from_square_l770_770368

theorem cut_rectangle_from_square
  (side : ℝ) (length : ℝ) (width : ℝ) 
  (square_area_eq : side * side = 100)
  (rectangle_area_eq : length * width = 90)
  (ratio_length_width : 5 * width = 3 * length) : 
  ¬ (length ≤ side ∧ width ≤ side) :=
by 
  sorry

end cut_rectangle_from_square_l770_770368


namespace towel_decrease_percentage_l770_770861

variable (L B : ℝ)
variable (h1 : 0.70 * L = L - (0.30 * L))
variable (h2 : 0.60 * B = B - (0.40 * B))

theorem towel_decrease_percentage (L B : ℝ) 
  (h1 : 0.70 * L = L - (0.30 * L))
  (h2 : 0.60 * B = B - (0.40 * B)) :
  ((L * B - (0.70 * L) * (0.60 * B)) / (L * B)) * 100 = 58 := 
by
  sorry

end towel_decrease_percentage_l770_770861


namespace proof_problem_l770_770612

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log ((1 + x) / (1 + a * x))

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a + 2 / (1 + 2^x)

-- Condition: f(x) is an odd function
axiom f_odd (a : ℝ) (h : a ≠ 1) : ∀ x : ℝ, x ∈ Ioo (-1) 1 → f (-x) a + f x a = 0

-- The theorem combining both parts
theorem proof_problem (a_ne_1 : ∀ a : ℝ, a ≠ 1) :
  ∀ (a : ℝ), a ≠ 1 →
  (∀ x : ℝ, x ∈ Ioo (-1) 1 → f (-x) a + f x a = 0) →
  a = -1 ∧ g (1/2) a + g (-1/2) a = 2 :=
begin
  assume a ha odd_fn,
  have a_eq : a = -1, from sorry, -- Part (1) proof skipped as per instructions
  split,
  { exact a_eq },
  have eval_g : g (1/2) a + g (-1/2) a = 2, from sorry, -- Part (2) proof skipped as per instructions
  { exact eval_g }
end

end proof_problem_l770_770612


namespace wheel_rotation_radians_l770_770077

theorem wheel_rotation_radians (r s : ℝ) (h_r : r = 20) (h_s : s = 40) : s / r = 2 := 
by
  -- introduce the given values
  have h : 40 / 20 = 2 := by norm_num
  rw [h_r, h_s]
  exact h

end wheel_rotation_radians_l770_770077


namespace linear_congruence_solution_l770_770964

theorem linear_congruence_solution : ∃ x : ℤ, (7 * x + 3 ≡ 2 [MOD 17]) ∧ (x ≡ 12 [MOD 17]) :=
by
  sorry

end linear_congruence_solution_l770_770964


namespace negation_of_p_l770_770185

theorem negation_of_p :
  (¬ (∀ x > 0, (x+1)*Real.exp x > 1)) ↔ 
  (∃ x ≤ 0, (x+1)*Real.exp x ≤ 1) :=
sorry

end negation_of_p_l770_770185


namespace sum_of_squares_of_entries_l770_770981

theorem sum_of_squares_of_entries (p q r s t u v w x : ℝ)
  (B : Matrix (Fin 3) (Fin 3) ℝ)
  (hB : B = ![![p, q, r], ![s, t, u], ![v, w, x]])
  (h_transpose_inverse : Bᵀ = B⁻¹) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 + w^2 + x^2 = 3 :=
sorry

end sum_of_squares_of_entries_l770_770981


namespace base7_to_base10_l770_770823

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770823


namespace circular_plot_area_l770_770853

/-- The price per foot of building the fence. -/
def pricePerFoot : ℝ := 58

/-- The total cost of building the fence. -/
def totalCost : ℝ := 3495.28

/-- The circumference of the circular plot. -/
noncomputable def circumference : ℝ := totalCost / pricePerFoot

/-- The radius of the circular plot. -/
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

/-- The area of the circular plot. -/
noncomputable def area : ℝ := Real.pi * radius^2

/-- The problem statement that needs to be proved. -/
theorem circular_plot_area : area ≈ 288.67 := 
sorry

end circular_plot_area_l770_770853


namespace base4_sum_conversion_to_base10_l770_770934

theorem base4_sum_conversion_to_base10 :
  let n1 := 2213
  let n2 := 2703
  let n3 := 1531
  let base := 4
  let sum_base4 := n1 + n2 + n3 
  let sum_base10 :=
    (1 * base^4) + (0 * base^3) + (2 * base^2) + (5 * base^1) + (1 * base^0)
  sum_base10 = 309 :=
by
  sorry

end base4_sum_conversion_to_base10_l770_770934


namespace sum_of_cubes_mod_five_l770_770485

theorem sum_of_cubes_mod_five : 
  (∑ n in Finset.range 51, n^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_five_l770_770485


namespace neg_P_l770_770590

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x ≤ 0

-- State the negation of P
theorem neg_P : ¬P ↔ ∀ x : ℝ, Real.exp x > 0 := 
by 
  sorry

end neg_P_l770_770590


namespace f_5times_8_eq_l770_770197

def f (x : ℚ) : ℚ := 1 / x ^ 2

theorem f_5times_8_eq :
  f (f (f (f (f (8 : ℚ))))) = 1 / 79228162514264337593543950336 := 
  by
    sorry

end f_5times_8_eq_l770_770197


namespace exists_set_E_l770_770941

open Set

def is_closed_under_addition (E : Set (ℤ × ℤ)) : Prop :=
  ∀ ⦃x y : ℤ × ℤ⦄, x ∈ E → y ∈ E → (x + y) ∈ E

def contains_origin (E : Set (ℤ × ℤ)) : Prop :=
  (0, 0) ∈ E

def contains_one_of (E : Set (ℤ × ℤ)) : Prop :=
  ∀ ⦃a b : ℤ⦄, (a, b) ≠ (0, 0) → ((a, b) ∈ E ↔ (-a, -b) ∉ E)

noncomputable def E : Set (ℤ × ℤ) :=
  (Σ (a : {n : ℤ // n > 0}), Set.univ : Set (Σ a : {n : ℤ // n > 0}, ℤ)) ∪
  (Set.univ : Set (Σ a : {0 = 0}, {n : ℤ // n ≥ 0}))

theorem exists_set_E : 
  is_closed_under_addition E ∧ contains_origin E ∧ contains_one_of E := 
sorry

end exists_set_E_l770_770941


namespace probability_longer_piece_eq_x_times_shorter_piece_l770_770069

theorem probability_longer_piece_eq_x_times_shorter_piece (x : ℝ) (h_pos : x > 0) :
  (∃ (C : ℝ) (hC1 : C = 1 / (x + 1)) (hC2 : C ≤ 1 / 2) (hC3 : x ≥ 1) ∨
      (C : ℝ) (hC1 : C = x / (x + 1)) (hC2 : C ≥ 1 / 2) (hC3 : x ≥ 1)) → 
  0 :=
by sorry

end probability_longer_piece_eq_x_times_shorter_piece_l770_770069


namespace value_of_a_l770_770155

theorem value_of_a (a : ℝ) (M : set ℝ) (h : M = {x | a * x^2 + 2 * x + 1 = 0} ∧ ∃! x, x ∈ M) : a = 0 ∨ a = 1 :=
by sorry

end value_of_a_l770_770155


namespace inequality_proof_l770_770294

theorem inequality_proof (n : ℕ) (h : n > 2) : (12 * ∏ k in finset.range n, k.succ ^ 2) > n ^ n :=
sorry

end inequality_proof_l770_770294


namespace interval_and_symmetry_l770_770583

variable {f : ℝ → ℝ}

noncomputable def even (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

theorem interval_and_symmetry (hf_even : even f) (hf_domain : ∀ x : ℝ, true) 
  (hf_increasing : ∀ x y : ℝ, 2 < x ∧ x < 6 ∧ 2 < y ∧ y < 6 → f x < f y) :
  (∀ x y : ℝ, 4 < x ∧ x < 8 ∧ 4 < y ∧ y < 8 → f(2-x) < f(2-y)) ∧ 
  (∀ x: ℝ, x = 2 → f(2-x) = f(2+x)) :=
by
  sorry

end interval_and_symmetry_l770_770583


namespace coefficient_is_integer_not_all_coefficients_need_to_be_integers_l770_770595

variable {α : Type*} [Ring α] [LinearOrderedField α]

/-- Statement 1: Prove that if for any integer x, the value of the trinomial ax² + bx + c is an integer, then at least one of the coefficients a, b, or c is an integer. -/
theorem coefficient_is_integer (a b c : α) (h : ∀ x : ℤ, a * (x : α)^2 + b * (x : α) + c ∈ ℤ) : 
  c ∈ ℤ :=
by {
  specialize h 0,
  norm_cast at h,
  exact h,
}

/-- Statement 2: Prove that it is not necessary for all the coefficients a, b, c to be integers for the trinomial ax² + bx + c to take integer values for any integer x. -/
theorem not_all_coefficients_need_to_be_integers (a b c : α) : 
  (∃ a b : α, ∀ x : ℤ, a * (x : α)^2 + b * (x : α) ∈ ℤ ∧ (a ∉ ℤ ∨ b ∉ ℤ)) :=
by {
  use [1 / 2, 1 / 2, 0],
  intro x,
  split,
  { norm_cast,
    ring,
    use x * (x + 1) / 2,
    ring, },
  { left,
    norm_cast,
    exact half_not_int, },
  }

end coefficient_is_integer_not_all_coefficients_need_to_be_integers_l770_770595


namespace cows_milk_production_l770_770200

theorem cows_milk_production (x : ℕ) (h : x > 0) : 
  let daily_production_per_cow := (x + 2)/(x * (x + 4)) in
  let total_daily_production := (x + 4) * daily_production_per_cow in
  let days_required := (x + 6) / total_daily_production in
  days_required = (x * (x + 4) * (x + 6)) / ((x + 2) * (x + 4)) :=
by
  sorry

end cows_milk_production_l770_770200


namespace find_f_l770_770255

def f (n : ℕ) : ℕ :=
  let a := λ k : ℕ, k^3 - 1 in
  let gcd_list := list.gcd (list.map a (list.filter (λ m, n > m ∧ nat.coprime n m) (list.range n))) in
  gcd_list

theorem find_f (n : ℕ) (h_pos : 0 < n) :
  (6 ∣ n → 32 < n → f n = 2) ∧ (¬ (6 ∣ n) → 7 < n → (odd n ∨ ¬ (3 ∣ n)) → f n = 1) := 
by
  sorry

end find_f_l770_770255


namespace point_not_on_graph_l770_770082

theorem point_not_on_graph :
  ¬ (1 * 5 = 6) :=
by 
  sorry

end point_not_on_graph_l770_770082


namespace slope_of_line_l770_770766

theorem slope_of_line (k : ℝ) (P Q : ℝ × ℝ) (O : ℝ × ℝ := (0, 0)) 
  (h1: k > 0) 
  (h2: P = (-k, k^2)) 
  (h3: Q = (4k, 16k^2))
  (h4: let area := (1/2) * abs (0*(k^2-16k^2) + (-k)*(16k^2-0) + 4k*(0-k^2))
       in area = 80) : 
  let slope := 3*k 
  in slope = 6 := 
sorry

end slope_of_line_l770_770766


namespace condition_for_inequality_l770_770653

variable {ℝ : Type*} [linear_ordered_field ℝ]

def increasing_function (f : ℝ → ℝ) :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

theorem condition_for_inequality (f : ℝ → ℝ) 
  (h_inc : increasing_function f) (a b : ℝ) :
  (a + b > 0) ↔ (f(a) + f(b) > f(-a) + f(-b)) :=
begin
  sorry
end

end condition_for_inequality_l770_770653


namespace solve_for_F_l770_770194

theorem solve_for_F (C F : ℝ) (h1 : C = 5 / 9 * (F - 32)) (h2 : C = 40) : F = 104 :=
by
  sorry

end solve_for_F_l770_770194


namespace best_fit_slope_l770_770370

noncomputable def slope_ls (data : List (ℝ × ℝ)) : ℝ :=
  let n := data.length
  let mean_x := data.map Prod.fst |>.sum / n
  let mean_y := data.map Prod.snd |>.sum / n
  let num := data.foldl (λ acc (x, y) => acc + (x - mean_x) * (y - mean_y)) 0
  let denom := data.foldl (λ acc (x, _) => acc + (x - mean_x) ^ 2) 0
  num / denom

theorem best_fit_slope : slope_ls [(0, 15), (1, 10), (4, 5), (2, 8)] = -2.343 :=
by
  sorry

end best_fit_slope_l770_770370


namespace sequence_elements_are_prime_l770_770039

variable {a : ℕ → ℕ} {p : ℕ → ℕ}

def increasing_seq (f : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → f i < f j

def divisible_by_prime (a p : ℕ → ℕ) : Prop :=
  ∀ n, Prime (p n) ∧ p n ∣ a n

def satisfies_condition (a p : ℕ → ℕ) : Prop :=
  ∀ n k, a n - a k = p n - p k

theorem sequence_elements_are_prime (h1 : increasing_seq a) 
    (h2 : divisible_by_prime a p) 
    (h3 : satisfies_condition a p) :
    ∀ n, Prime (a n) :=
by 
  sorry

end sequence_elements_are_prime_l770_770039


namespace shape_is_plane_l770_770501

noncomputable
def cylindrical_coordinates_shape (r θ z c : ℝ) := θ = 2 * c

theorem shape_is_plane (c : ℝ) : 
  ∀ (r : ℝ) (θ : ℝ) (z : ℝ), cylindrical_coordinates_shape r θ z c → (θ = 2 * c) :=
by
  sorry

end shape_is_plane_l770_770501


namespace base_7_to_10_of_23456_l770_770831

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770831


namespace total_songs_sung_l770_770248

def total_minutes := 80
def intermission_minutes := 10
def long_song_minutes := 10
def short_song_minutes := 5

theorem total_songs_sung : 
  (total_minutes - intermission_minutes - long_song_minutes) / short_song_minutes + 1 = 13 := 
by 
  sorry

end total_songs_sung_l770_770248


namespace longest_route_l770_770878

-- Defining the longest_route problem in Lean
theorem longest_route (A B: Intersection) (paths: Set Path) (n_intersections: ℕ)
  (start: route.A = true) (end: route.B = true) 
  (distinct_intersections: ∀ (p₁ p₂: route), p₁ ≠ p₂ → p₁.intersection ≠ p₂.intersection):
  maximum_route_length_from_A_to_B == 34 :=
by
  sorry

-- Definitions of Intersection, Path, and Route
structure Intersection := (id : ℕ)
structure Path := (start : Intersection) (end : Intersection)
structure Route := (length : ℕ) (paths : List Path) (A B : Bool) (dist_intersects : Bool)

-- Implement the uniqueness condition for intersections
def distinct_intersections := ∀ (route₁ route₂ : Route), route₁ ≠ route₂ → route₁.dist_intersects = true

end longest_route_l770_770878


namespace Jerry_average_speed_l770_770687

variable (J : ℝ) -- Jerry's average speed in miles per hour
variable (C : ℝ) -- Carla's average speed in miles per hour
variable (T_J : ℝ) -- Time Jerry has been driving in hours
variable (T_C : ℝ) -- Time Carla has been driving in hours
variable (D : ℝ) -- Distance covered in miles

-- Given conditions
axiom Carla_speed : C = 35
axiom Carla_time : T_C = 3
axiom Jerry_time : T_J = T_C + 0.5

-- Distance covered by Carla in T_C hours at speed C
axiom Carla_distance : D = C * T_C

-- Distance covered by Jerry in T_J hours at speed J
axiom Jerry_distance : D = J * T_J

-- The goal to prove
theorem Jerry_average_speed : J = 30 :=
by
  sorry

end Jerry_average_speed_l770_770687


namespace soap_bars_problem_l770_770363

theorem soap_bars_problem :
  ∃ (N : ℤ), 200 < N ∧ N < 300 ∧ 2007 % N = 5 :=
sorry

end soap_bars_problem_l770_770363


namespace lines_divide_plane_l770_770236

theorem lines_divide_plane (k : ℝ) :
  (∃ (x y : ℝ), x - 2*y + 1 = 0 ∧ x - 1 = 0) ∧ 
  (∃ (x y : ℝ), x + k*y = 0) →
  (∃ (k_values : set ℝ), k_values = {0, -1, -2} ∧
    (∃ (x y : ℝ), (x - 2*y + 1 = 0) ∧ (x - 1 = 0) ∧ (x + k*y = 0) → k_values.contains k)) :=
by sorry

end lines_divide_plane_l770_770236


namespace value_of_c_distinct_real_roots_l770_770557

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770557


namespace isosceles_triangle_sides_isosceles_triangle_possibilities_l770_770072

-- Part 1
theorem isosceles_triangle_sides (x: ℝ) (h1: 5 * x = 20) :
  (2 * x) = 8 ∧ (2 * x) = 8 ∧ x = 4 :=
by {
  have h2: x = 4, from (h1.symm ▸ rfl),
  rw h2,
  simp,
  sorry
}

-- Part 2
theorem isosceles_triangle_possibilities (a b: ℝ) 
  (h1: a = 6) (h2: b = 6 ∨ b = 7 ∧ a = 8) :
  (b = 7 ∧ a = 6 ∧ 6 = 7 ∧ 7) ∨ (b = 6 ∧ a = 6 ∧ 8 = 8 ∧ 6) :=
by {
  cases h2,
  { rw h1 at h2, exact or.inr ⟨eq.refl 6, eq.refl 6, eq.refl 8⟩, },
  {
    cases h2 with hb ha,
    exact or.inl ⟨hb, ha⟩,
  },
  sorry
}

end isosceles_triangle_sides_isosceles_triangle_possibilities_l770_770072


namespace base7_to_base10_l770_770821

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770821


namespace find_a_l770_770599

noncomputable def smallest_integer_solution : ℤ :=
  Inf {x : ℤ | 1 - (x - 2 : ℚ) / 2 < (1 + x : ℚ) / 3 }

theorem find_a : ∀ a : ℤ, (∃ x : ℤ, 1 - (x - 2 : ℚ) / 2 < (1 + x : ℚ) / 3 ∧ 2 * x - a = 3) → a = 3 :=
by
  intro a h
  let x := smallest_integer_solution
  have h₁ : 1 - (x - 2 : ℚ) / 2 < (1 + x : ℚ) / 3 := sorry
  have h₂ : 2 * x - a = 3 := sorry
  have h₃ : x = 3 := sorry
  have ha : a = 6 - 3 := sorry
  exact ha

end find_a_l770_770599


namespace eval_expression_l770_770136

-- Define the redefined operation
def red_op (a b : ℝ) : ℝ := (a + b)^2

-- Define the target expression to be evaluated
def expr (x y : ℝ) : ℝ := red_op ((x + y)^2) ((x - y)^2)

-- State the theorem
theorem eval_expression (x y : ℝ) : expr x y = 4 * (x^2 + y^2)^2 := by
  sorry

end eval_expression_l770_770136


namespace smallest_n_power_mod_5_l770_770360

theorem smallest_n_power_mod_5 :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (2^N + 1) % 5 = 0 ∧ ∀ M : ℕ, 100 ≤ M ∧ M ≤ 999 ∧ (2^M + 1) % 5 = 0 → N ≤ M := 
sorry

end smallest_n_power_mod_5_l770_770360


namespace calcium_iodide_weight_l770_770850

theorem calcium_iodide_weight
  (atomic_weight_Ca : ℝ)
  (atomic_weight_I : ℝ)
  (moles : ℝ) :
  atomic_weight_Ca = 40.08 →
  atomic_weight_I = 126.90 →
  moles = 5 →
  (atomic_weight_Ca + 2 * atomic_weight_I) * moles = 1469.4 :=
by
  intros
  sorry

end calcium_iodide_weight_l770_770850


namespace original_value_of_tempo_l770_770035

variable (V : ℝ)
variable (insured_fraction : ℝ := 4 / 5)
variable (premium_rate : ℝ := 1.3 / 100)
variable (premium : ℝ := 910)

theorem original_value_of_tempo : V = 87500 :=
by
  have h1 : insured_fraction = 4 / 5 := rfl
  have h2 : premium_rate = 1.3 / 100 := rfl
  have h3 : premium = 910 := rfl

  -- Calculating the insured value
  have insured_value : ℝ := insured_fraction * V

  -- Calculating the premium value
  have premium_calculated : ℝ := premium_rate * insured_value

  -- Given premium is $910, let's solve for V
  have premium_eq : premium_calculated = premium := rfl
  
  -- According to given condition (4/5)V * (1.3/100) = 910
  have equation : premium_rate * insured_fraction * V = premium := by
    rw [insured_value, premium_calculated, premium_eq]
    
  -- Solving for V: V = premium / (premium_rate * insured_fraction)
  have calculate_V : V = premium / (premium_rate * insured_fraction) := by
    exact sorry

  -- Check that the calculated value of V is equal to 87500
  have result : V = 87500 := by
    rw [calculate_V]
    -- Calculation to be filled in
    exact sorry

  exact result

end original_value_of_tempo_l770_770035


namespace fraction_meaningful_l770_770793

theorem fraction_meaningful (x : ℝ) : (1 / (x + 3)).is_defined ↔ x ≠ -3 :=
by sorry

end fraction_meaningful_l770_770793


namespace algebraic_expression_value_l770_770600

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 2) : 2 * x + 4 * y - 1 = 3 :=
sorry

end algebraic_expression_value_l770_770600


namespace farmer_days_to_complete_l770_770402

theorem farmer_days_to_complete
  (initial_productivity : ℕ)
  (daily_increase_percent : ℕ)
  (total_area : ℕ)
  (days_planned : ℕ)
  (days_ahead_of_schedule : ℕ)
  (increased_days : ℕ)
  (initial_days_worked : ℕ)
  (increased_productivity : ℕ)
  (new_productivity : ℕ) :
  initial_days_worked = 2 →
  daily_increase_percent = 25 →
  increased_days = days_planned - 2 →
  increased_productivity = initial_productivity * (1 + daily_increase_percent / 100) →
  initial_productivity * ↑initial_days_worked + increased_productivity * ↑increased_days = total_area →
  total_area = 1440 →
  initial_productivity = 120 →
  increased_productivity = 150 →
  days_planned = 10 →
  days_ahead_of_schedule = 2 →
  new_productivity = 120 + 0.25 * 120 →
  initial_days_worked * initial_productivity + increased_days * new_productivity = 1440 →
  days_planned - days_ahead_of_schedule = 8 :=
by {
  intros,
  rw [initial_days_worked, initial_productivity, daily_increase_percent, increased_days,
      increased_productivity, new_productivity, days_ahead_of_schedule],
  norm_num1,
ready for the proof. sorry
 }

end farmer_days_to_complete_l770_770402


namespace midpoint_locus_tangent_l770_770150

-- Define the given line as a function of t
def line (t : ℝ) (x : ℝ) : ℝ := (2 / t) * (x - t - 1)

-- Define the given parabola
def parabola (y : ℝ) : ℝ := (y^2) / 4

-- Define the locus of the midpoint of the points of intersection
def locus (x y : ℝ) : Prop := (y + 1)^2 = 2 * (x - 1/2)

-- Define the line x + y + 1 = 0
def line_locus (x y : ℝ) : Prop := x + y + 1 = 0

theorem midpoint_locus_tangent (t x y : ℝ) :
  (line t x = y) ∧ (parabola y = x) →
  (∃ x' y', locus x' y' ∧ line_locus x' y') :=
sorry

end midpoint_locus_tangent_l770_770150


namespace positive_difference_median_mode_l770_770359

/-- The list of numbers extracted from the stem-and-leaf plot -/
def stem_leaf_data : List ℕ := 
  [10, 10, 13, 15, 15, 15, 21, 23, 23, 24, 26, 30, 30, 32, 37, 37, 38, 41, 45, 45, 46, 49, 51, 52, 53, 58, 59]

/-- A sealed class to calculate the mode of a list -/
def list_mode (l : List ℕ) : ℕ :=
  l.groupBy id
   .map (λ g => (g.length, g.head!))
   .max .get_orElse (0, 0)
   .snd

/-- A sealed class to calculate the median of a list -/
noncomputable def list_median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· <= ·)
  sorted[(sorted.length / 2)]

/-- The theorem to prove the given statement -/
theorem positive_difference_median_mode :
  (list_median stem_leaf_data - list_mode stem_leaf_data) = 17 := by
  sorry

end positive_difference_median_mode_l770_770359


namespace positive_difference_of_complementary_ratio_5_1_l770_770787

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l770_770787


namespace length_of_BE_l770_770592

-- Definitions and conditions
variables (A B C D: Type)
(noncomputable def h_d : Real := Real.sqrt 2)
(noncomputable def f_h : Real := 5 * Real.sqrt 2)

theorem length_of_BE {E : Real} (extension_point_BE : Real) :
  ∃ BE, BE = 8 :=
by
  -- Given H = sqrt(2) and FH = 5sqrt(2)
  have h_d_def : h_d = Real.sqrt 2 := rfl
  have f_h_def : f_h = 5 * Real.sqrt 2 := rfl
  
  -- Conditions and calculations building up the problem
  have BE_def : BE = 8,
  exact BE_def
  
  existsi BE
  sorry

end length_of_BE_l770_770592


namespace monotonically_increasing_interval_l770_770320

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + 1 / x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x ∈ set.Ioi 1 → ∀ ε > 0, ∃ δ > 0, ∀ y ∈ set.Ioo (x - δ) (x + δ), y > x → f y > f x := 
by
  sorry

end monotonically_increasing_interval_l770_770320


namespace probability_A1_given_B_l770_770226

def num_red_balls_A : ℕ := 5
def num_white_balls_A : ℕ := 2
def num_black_balls_A : ℕ := 3
def total_balls_A : ℕ := num_red_balls_A + num_white_balls_A + num_black_balls_A

def num_red_balls_B : ℕ := 4
def num_white_balls_B : ℕ := 3
def num_black_balls_B : ℕ := 3
def total_balls_B : ℕ := num_red_balls_B + num_white_balls_B + num_black_balls_B + 1

def P_A1 : ℚ := num_red_balls_A / total_balls_A
def P_BA1 : ℚ := (num_red_balls_B + 1) / total_balls_B
def P_A2 : ℚ := num_white_balls_A / total_balls_A
def P_A3 : ℚ := num_black_balls_A / total_balls_A
def P_BA2_A3 : ℚ := num_red_balls_B / total_balls_B

def P_B : ℚ :=
  P_A1 * P_BA1 + P_A2 * P_BA2_A3 + P_A3 * P_BA2_A3

def P_A1_given_B : ℚ :=
  P_A1 * P_BA1 / P_B

theorem probability_A1_given_B :
  P_A1_given_B = 5 / 9 :=
by
  unfold P_A1 P_BA1 P_B P_A2 P_A3 P_BA2_A3 P_A1_given_B
  sorry

end probability_A1_given_B_l770_770226


namespace perpendicular_bisectors_concurrent_of_triangle_l770_770288

noncomputable def perpendicular_bisectors_concurrent (A B C : Point) : Prop :=
∃ O : Point, is_perpendicular_bisector O A B ∧ 
             is_perpendicular_bisector O A C ∧
             is_perpendicular_bisector O B C

theorem perpendicular_bisectors_concurrent_of_triangle (A B C : Point) (hABC : triangle A B C) :
  perpendicular_bisectors_concurrent A B C :=
sorry

end perpendicular_bisectors_concurrent_of_triangle_l770_770288


namespace double_series_sum_l770_770444

theorem double_series_sum :
  (∑' j : ℕ, ∑' k : ℕ, (2 : ℝ) ^ (-(3 * k + 2 * j + (k + j) ^ 2))) = 4 / 3 :=
sorry

end double_series_sum_l770_770444


namespace evaluate_decimal_expressions_l770_770110

theorem evaluate_decimal_expressions :
  let d1 := (2:ℚ) / 3
  let d2 := (2:ℚ) / 9
  let d3 := (4:ℚ) / 9
  let d4 := (1:ℚ) / 3
  d1 + d2 - d3 * d4 = (20:ℚ) / 27 :=
by
  sorry

end evaluate_decimal_expressions_l770_770110


namespace sin_phi_correct_l770_770705

variables {p q r : EuclideanSpace ℝ (Fin 3)}
variables {φ : ℝ}

-- Conditions from the problem
def norm_p : ‖p‖ = 2 := sorry
def norm_q : ‖q‖ = 4 := sorry
def norm_r : ‖r‖ = 6 := sorry
def cross_product_condition : p × (p × q) = r := sorry

noncomputable def sin_phi : ℝ := sorry

-- Statement to be proved
theorem sin_phi_correct : sin_phi = 3 / 4 :=
begin
  -- Proof would go here
  sorry
end

end sin_phi_correct_l770_770705


namespace quadratic_two_distinct_real_roots_l770_770514

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770514


namespace find_train_speed_l770_770421

noncomputable def man_speed := 8 / 3.6 -- Speed of the man in m/s
noncomputable def train_length := 80 -- Length of the train in meters
noncomputable def crossing_time := 9 -- Time to cross the man in seconds

theorem find_train_speed : 
  let train_speed := 24 in -- Speed of the train in km/h
  (train_length / crossing_time) = train_speed * 1000 / 3600 + man_speed :=
sorry

end find_train_speed_l770_770421


namespace sum_cubes_mod_five_l770_770495

theorem sum_cubes_mod_five :
  (∑ n in Finset.range 50, (n + 1)^3) % 5 = 0 :=
by
  sorry

end sum_cubes_mod_five_l770_770495


namespace find_triples_l770_770119

-- Defining the conditions
def divides (x y : ℕ) : Prop := ∃ k, y = k * x

-- The main Lean statement
theorem find_triples (a b c : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  divides a (b * c - 1) → divides b (a * c - 1) → divides c (a * b - 1) →
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 3) ∨
  (a = 3 ∧ b = 2 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 2) ∨
  (a = 5 ∧ b = 2 ∧ c = 3) ∨ (a = 5 ∧ b = 3 ∧ c = 2) :=
sorry

end find_triples_l770_770119


namespace quadratic_has_distinct_real_roots_l770_770537

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770537


namespace number_of_functions_l770_770266

noncomputable def X := {i : ℕ | 1 ≤ i ∧ i ≤ 100}

def satisfies_condition (f : X → X) : Prop :=
  ∀ a b : X, a < b → f b < f a + (b - a)

theorem number_of_functions :
  (∃ (f : X → X), satisfies_condition f) → (∃! (n : ℕ), n = nat.choose 199 100) :=
sorry

end number_of_functions_l770_770266


namespace exponential_distribution_characterization_l770_770272

noncomputable def is_exponential (ξ : ℝ → ℝ) (λ : ℝ) : Prop :=
  ∀ x, ξ x = 1 - exp (-λ * x)

noncomputable def random_variable_meets_condition (ξ η : ℝ → ℝ) : Prop :=
  ∀ x, (min (ξ x) (η x)) = ξ (x / 2)

open Classical

theorem exponential_distribution_characterization {ξ : ℝ → ℝ} {η : ℝ → ℝ} (F : ℝ → ℝ) (λ : ℝ)
  (h1 : ∀ x, 0 ≤ ξ x) 
  (h2 : ∀ x, F x = ξ x) 
  (h3 : F 0 < 1) 
  (h4 : ∃ l : ℝ, has_deriv_at_right F 0 l) 
  (h5 : ∀ x, η x = ξ x)
  (h6 : random_variable_meets_condition ξ η) :
  is_exponential ξ λ ↔ ξ ∧ η d= ξ / 2 :=
sorry

end exponential_distribution_characterization_l770_770272


namespace parallelogram_area_l770_770931

variables (p q a b : ℝ^3)
variables (norm_p : p.norm = 7) (norm_q : q.norm = 2) 
variables (angle_pq : real.angle p q = real.pi / 4)

def a_def : ℝ^3 := 3 * p + q
def b_def : ℝ^3 := p - 3 * q

theorem parallelogram_area : (a_def ∧ b_def → |a_def × b_def| = 70 * real.sqrt 2) :=
by sorry

end parallelogram_area_l770_770931


namespace sum_of_cubes_mod_l770_770474

theorem sum_of_cubes_mod (S : ℕ) (h : S = ∑ n in Finset.range 51, n^3) : S % 5 = 0 :=
sorry

end sum_of_cubes_mod_l770_770474


namespace option_b_correct_l770_770029

theorem option_b_correct : 
  (sqrt 2 * sqrt 3 = sqrt 6) ∧ 
  ¬(sqrt 2 + sqrt 3 = sqrt 5) ∧ 
  ¬(2 + sqrt 2 = 2 * sqrt 2) ∧ 
  ¬(2 / sqrt 2 = sqrt 2 / 2) :=
by
  sorry

end option_b_correct_l770_770029


namespace a_plus_b_eq_three_l770_770998

noncomputable def xi : ℝ → ℝ := sorry  -- Define the discrete random variable ξ
axiom p_a : 2 / 3 = ∑' x, xi x * (if x = a then 1 else 0)
axiom p_b : 1 / 3 = ∑' x, xi x * (if x = b then 1 else 0)
axiom a_lt_b : a < b
axiom expec_xi : ∑' x, xi x * x = 4 / 3
axiom var_xi : ∑' x, xi x * (x - 4 / 3)^2 = 2 / 9

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem a_plus_b_eq_three : a + b = 3 := 
sorry

end a_plus_b_eq_three_l770_770998


namespace parabola_line_intersection_l770_770404

/-- 
Given a parabola \( y^2 = 2x \), a line passing through the focus of 
the parabola intersects the parabola at points \( A \) and \( B \) where 
the sum of the x-coordinates of \( A \) and \( B \) is equal to 2. 
Prove that such a line exists and there are exactly 3 such lines.
--/
theorem parabola_line_intersection :
  ∃ l₁ l₂ l₃ : (ℝ × ℝ) → (ℝ × ℝ), 
    (∀ p, l₁ p = l₂ p ∧ l₁ p = l₃ p → false) ∧
    ∀ (A B : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * A.1) ∧ 
      (B.2 ^ 2 = 2 * B.1) ∧ 
      (A.1 + B.1 = 2) →
      (∃ k : ℝ, 
        ∀ (x : ℝ), 
          ((A.2 = k * (A.1 - 1)) ∧ (B.2 = k * (B.1 - 1))) ∧ 
          (k * (A.1 - 1) = k * (B.1 - 1)) ∧ 
          (k ≠ 0)) :=
sorry

end parabola_line_intersection_l770_770404


namespace sum_nonnegative_l770_770062

def sequence_x : ℕ → ℤ
| 0 := 1
| (2 * k + 1) := (-1)^(k + 1) * sequence_x k
| (2 * k + 2) := -sequence_x (k + 1)

theorem sum_nonnegative : ∀ n : ℕ, 0 < n → (∑ i in finset.range n, sequence_x (i + 1)) ≥ 0 :=
begin
  intro n,
  intro hn,
  -- Proof goes here
  sorry
end

end sum_nonnegative_l770_770062


namespace tangency_points_form_acute_triangle_l770_770439

theorem tangency_points_form_acute_triangle
  (A B C A1 B1 C1 O : Type*)
  (α β γ : ∠ A B C)
  (h1 : O is_incenter_of (triangle A B C))
  (h2 : A1 is_touch_point_of_incircle (side B C))
  (h3 : B1 is_touch_point_of_incircle (side C A))
  (h4 : C1 is_touch_point_of_incircle (side A B))
  : ¬ obtuse ∠(A1 B1 C1) ∧ ¬ obtuse ∠(B1 C1 A1) ∧ ¬ obtuse ∠(C1 A1 B1) := 
sorry

end tangency_points_form_acute_triangle_l770_770439


namespace collinear_vectors_solution_l770_770620

noncomputable def vector_collinear_x : Prop :=
  ∀ (x : ℝ), let a := (2, 1) in
             let b := (2, x) in
             let v1 := (2 * 2 + 3 * 2 * x, 2 * 1 + 3 * x) in
             let v2 := (-2, 1 - 2 * x) in
             -2 * v1.2 = 10 * v2.2 → x = 1

theorem collinear_vectors_solution : vector_collinear_x :=
  sorry

end collinear_vectors_solution_l770_770620


namespace walter_school_expenses_l770_770353

namespace WalterEarnings

def fast_food_weekday_hours := 5 * 4
def fast_food_weekend_hours := 2 * 6
def convenience_store_hours := 5
def fast_food_rate := 5
def convenience_store_rate := 7

def total_fast_food_earnings :=
  (fast_food_weekday_hours + fast_food_weekend_hours) * fast_food_rate

def convenience_store_earnings :=
  convenience_store_hours * convenience_store_rate

def total_earnings :=
  total_fast_food_earnings + convenience_store_earnings

def school_expenses_allocation :=
  (3 / 4) * total_earnings

theorem walter_school_expenses : school_expenses_allocation = 146.25 := by
  sorry

end WalterEarnings

end walter_school_expenses_l770_770353


namespace min_number_of_blocks_is_604_l770_770049

-- Define the dimensions of the wall
def wall_length : ℕ := 150
def wall_height : ℕ := 8
def block_height : ℕ := 1
def block1_length : ℕ := 2
def block2_length : ℕ := (3 / 2).to_int

-- Condition to ensure the wall is finished evenly on both ends
def finished_evenly (length height block1 block2 : ℕ) (block1_count block2_count : ℕ) : Prop :=
  block1_count * block1 + block2_count * block2 = length

-- Minimum number of blocks needed to complete the wall
def min_blocks_needed : ℕ :=
  let odd_row_blocks := wall_length / block1_length
  let even_row_blocks := ((wall_length - 3) / block1_length) + 2
  in (4 * odd_row_blocks) + (4 * even_row_blocks)

-- Main statement to be proven
theorem min_number_of_blocks_is_604 :
  finished_evenly wall_length wall_height block1_length block2_length (4 * (wall_length / block1_length)) (4 * ((wall_length - 3) / block1_length) + 8) ∧
  min_blocks_needed = 604 :=
sorry

end min_number_of_blocks_is_604_l770_770049


namespace remainder_is_correct_l770_770959

theorem remainder_is_correct :
  ∃ q : Polynomial ℤ, 
    let p := (Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 1 * Polynomial.X^5 - Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 1 * Polynomial.X^2)
    let d := (Polynomial.X - Polynomial.C 1) * (Polynomial.X + Polynomial.C 1) * (Polynomial.X - Polynomial.C 2)
    let r := (Polynomial.C (26/3 : ℚ) * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X - Polynomial.C (26/3 : ℚ))
    p = d * q + r := 
begin
  sorry
end

end remainder_is_correct_l770_770959


namespace problem_A_problem_B_problem_C_problem_D_l770_770571

open Complex Real Set

noncomputable def z1 : ℂ := (1 + 3 * I) / (1 - 3 * I)
def conj_z1 : ℂ := (-4 - 3 * I) / 5

theorem problem_A : conj(z1) = conj_z1 := sorry

def z_pure_imaginary (b : ℝ) : ℂ := b * I

theorem problem_B (b : ℝ) (h : b ≠ 0) : (z_pure_imaginary b)^2 < 0 := sorry

def z2 (z : ℂ) : Prop := z - (2 + I) > 0

theorem problem_C (z : ℂ) : ¬ z2 z := sorry

def M : Set ℂ := {z | abs(z + 3 * I) ≤ 3}

def area_M : ℝ := 3 * 3 * π

theorem problem_D : area_M ≠ 6 * π := sorry

end problem_A_problem_B_problem_C_problem_D_l770_770571


namespace problem_l770_770944

-- Definitions based on the given problem
def f (x : ℝ) : ℝ := log x / log 3 + 6

noncomputable def f_inv (x : ℝ) : ℝ := 3 ^ x - 6

-- Given the condition that (f_inv(m) + 6) * (f_inv(n) + 6) = 27 and m + n = 3
def condition (m n : ℝ) : Prop :=
  (f_inv(m) + 6) * (f_inv(n) + 6) = 27

-- Statement of the theorem to prove f(m + n) = 2 under the given conditions
theorem problem (m n : ℝ) (h : condition m n) : f(m + n) = 2 :=
by
  sorry

end problem_l770_770944


namespace collinear_vectors_l770_770177

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (e1 e2 : V) (a : V := 2 • e1 - e2) (λ : ℝ) (b : V := e1 + λ • e2)

theorem collinear_vectors : ¬ collinear ℝ ({e1, e2} : set V) → collinear ℝ ({a, b} : set V) → λ = -1/2 := 
by 
  sorry

end collinear_vectors_l770_770177


namespace polar_eqn_C1_polar_coords_D_ray_intersects_and_area_l770_770324

-- Definitions from the conditions
def curve_C1_parametric (α : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos α, 2 * Real.sin α)

def curve_C2_polar (θ : ℝ) : ℝ :=
  2 * Real.cos θ

def point_O : ℝ × ℝ :=
  (0, 0)

def point_D : ℝ × ℝ :=
  (2, 0)

def ray_l (β : ℝ) (ρ > 0) (0 < β < π) : Prop :=
  ρ = ρ -- Placeholder to define the ray's equation

def area_of_triangle_ABD (A B D : ℝ × ℝ) : ℝ :=
  0.5 * (A.1 * (B.2 - D.2) + B.1 * (D.2 - A.2) + D.1 * (A.2 - B.2))  -- Shoelace formula

-- Proof problem statements
theorem polar_eqn_C1 : ∀ α, (curve_C1_parametric α).fst ^ 2 + (curve_C1_parametric α).snd ^ 2 - 4 * (curve_C1_parametric α).fst = 0 := sorry

theorem polar_coords_D : point_D = (2, 0) := sorry

theorem ray_intersects_and_area (β : ℝ) (Hβ : 0 < β ∧ β < π) :
  area_of_triangle_ABD (curve_C1_parametric β) (curve_C2_polar β, β) point_D = sqrt 3 / 2 → 
  β = π / 6 ∨ β = π / 3 := sorry

end polar_eqn_C1_polar_coords_D_ray_intersects_and_area_l770_770324


namespace lisa_needs_4_weeks_to_eat_all_candies_l770_770730

-- Define the number of candies Lisa has initially.
def candies_initial : ℕ := 72

-- Define the number of candies Lisa eats per week based on the given conditions.
def candies_per_week : ℕ := (3 * 2) + (2 * 2) + (4 * 2) + 1

-- Define the number of weeks it takes for Lisa to eat all the candies.
def weeks_to_eat_all_candies (candies : ℕ) (weekly_candies : ℕ) : ℕ := 
  (candies + weekly_candies - 1) / weekly_candies

-- The theorem statement that proves Lisa needs 4 weeks to eat all 72 candies.
theorem lisa_needs_4_weeks_to_eat_all_candies :
  weeks_to_eat_all_candies candies_initial candies_per_week = 4 :=
by
  sorry

end lisa_needs_4_weeks_to_eat_all_candies_l770_770730


namespace quadratic_distinct_roots_l770_770552

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770552


namespace expected_difference_l770_770089

def roll_probability := {even := 3 / 5, odd := 2 / 5}

def days_in_year := 365

def expected_pancake_days := roll_probability.even * days_in_year

def expected_oatmeal_days := roll_probability.odd * days_in_year

theorem expected_difference : expected_pancake_days - expected_oatmeal_days = 73 :=
by
  -- placeholder for the proof
  sorry

end expected_difference_l770_770089


namespace quadratic_two_distinct_real_roots_l770_770517

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770517


namespace quadratic_roots_condition_l770_770509

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770509


namespace sum_of_digits_of_greatest_prime_divisor_of_2_pow_13_minus_1_l770_770943

-- Define the number of interest
def n : ℕ := 2^13 - 1

-- Define a function to find the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- State the main theorem to be proved
theorem sum_of_digits_of_greatest_prime_divisor_of_2_pow_13_minus_1 :
  sum_of_digits (nat.greatest_prime_divisor n) = 19 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_2_pow_13_minus_1_l770_770943


namespace difference_of_elements_l770_770164

variables (n k m : ℕ) (A : Finset ℕ)

theorem difference_of_elements 
  (k_pos : 2 ≤ k)
  (n_pos : 0 < n)
  (m_pos : 0 < m)
  (n_le_m : n ≤ m)
  (m_lt : m < (2 * k - 1) * n / k)
  (subset_A : A ⊆ Finset.range (m + 1))
  (card_A : A.card = n) :
  ∀ x : ℕ, (0 < x ∧ x < n / (k - 1)) → ∃ a a' ∈ A, x = a - a' :=
by
  sorry

end difference_of_elements_l770_770164


namespace percentage_alcohol_second_vessel_l770_770076

theorem percentage_alcohol_second_vessel :
  (∀ (x : ℝ),
    (0.25 * 3 + (x / 100) * 5 = 0.275 * 10) -> x = 40) :=
by
  intro x h
  sorry

end percentage_alcohol_second_vessel_l770_770076


namespace smallest_int_remainder_two_l770_770023

theorem smallest_int_remainder_two (m : ℕ) (hm : m > 1)
  (h3 : m % 3 = 2)
  (h4 : m % 4 = 2)
  (h5 : m % 5 = 2)
  (h6 : m % 6 = 2)
  (h7 : m % 7 = 2) :
  m = 422 :=
sorry

end smallest_int_remainder_two_l770_770023


namespace angle_F_measure_l770_770240

theorem angle_F_measure (D E F : ℝ) (h₁ : D = 80) (h₂ : E = 2 * F + 24) (h₃ : D + E + F = 180) : F = 76 / 3 :=
by
  sorry

end angle_F_measure_l770_770240


namespace gcd_of_270_and_180_l770_770848

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_of_270_and_180 : gcd 270 180 = 90 := by
  sorry

end gcd_of_270_and_180_l770_770848


namespace hyperbola_eccentricity_directrix_midpoint_l770_770983

theorem hyperbola_eccentricity_directrix_midpoint (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (eccentricity : Real.sqrt 3) 
  (directrix : ℝ := Real.sqrt 3 / 3)
  (intersects_at_two_points : ∀ m : ℝ, ∃ (x1 y1 x2 y2 : ℝ), x1 - y1 + m = 0 ∧ x2 - y2 + m = 0 ∧ x1^2 - y1^2 / 2 = 1 ∧ x2^2 - y2^2 / 2 = 1)
  (midpoint_circle : ∀ (x0 y0 : ℝ), ∃ (x1 y1 x2 y2 : ℝ), x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2 ∧ x0^2 + y0^2 = 5) :
  ∃ m : ℝ, m = 1 ∨ m = -1 := 
sorry

end hyperbola_eccentricity_directrix_midpoint_l770_770983


namespace inequality_range_l770_770385

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem inequality_range (a b x: ℝ) (h : a ≠ 0) :
  (|a + b| + |a - b|) ≥ |a| * f x → 1 ≤ x ∧ x ≤ 2 :=
by
  intro h1
  unfold f at h1
  sorry

end inequality_range_l770_770385


namespace range_sum_l770_770652

noncomputable def f (x : ℝ) : ℝ := 1 + (2^(x+1)) / (2^x + 1) + sin x

theorem range_sum (k : ℝ) (hk : k > 0) (I : set.Icc (-k) k) 
  (m n : ℝ) (h₁ : set.range f ∩ I = set.Icc m n) : m + n = 4 :=
by
  sorry

end range_sum_l770_770652


namespace midpoint_equidistant_l770_770658

theorem midpoint_equidistant (A B : ℝ) (hA : A = -1) (hB : B = 3) : (A + B) / 2 = 1 :=
by
  -- providing the conditions
  rw [hA, hB]
  -- simplifying
  simp
  sorry

end midpoint_equidistant_l770_770658


namespace points_form_small_triangle_l770_770745

open Set

theorem points_form_small_triangle (n : ℕ) (h : 0 < n) (points : Finset (ℝ × ℝ))
  (hc : points.card = 2 * n + 1) (hpoints : ∀ p ∈ points, (0 : ℝ) ≤ p.1 ∧ p.1 ≤ 1 ∧ (0 : ℝ) ≤ p.2 ∧ p.2 ≤ 1) :
  ∃ (A B C : (ℝ × ℝ)), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧
  ∃ area : ℝ, area_triangle A B C area ∧ area ≤ 1 / (2 * n) :=
begin
  sorry
end

-- Definition for deduct area of the triangle given points A, B, C
def area_triangle (A B C : (ℝ × ℝ)) (area : ℝ) : Prop :=
  area = 1 / 2 * (abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)))

end points_form_small_triangle_l770_770745


namespace base_seven_to_base_ten_l770_770840

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770840


namespace quadratic_two_distinct_real_roots_l770_770521

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770521


namespace cone_volume_l770_770401

-- Define cone with given slant height and central angle properties
noncomputable def cone (l : ℝ) (theta : ℝ) (V : ℝ) : Prop :=
  l = 6 ∧ theta = 120 * (π / 180) ∧ V = (1/3) * π * 2^2 * 4 * sqrt(2)

-- Theorem stating the volume of the cone
theorem cone_volume : cone 6 (120 * (π / 180)) (16 * sqrt(2) / 3 * π) :=
by
  sorry

end cone_volume_l770_770401


namespace solve_complex_addition_l770_770698

def A : ℂ := 3 + 2 * complex.I
def O : ℂ := -3 + 1 * complex.I
def P : ℂ := 1 - 2 * complex.I
def S : ℂ := 4 + 5 * complex.I
def T : ℂ := -1 + 0 * complex.I

theorem solve_complex_addition :
  A - O + P + S + T = 10 + 4 * complex.I :=
by
  sorry

end solve_complex_addition_l770_770698


namespace total_daisies_sold_l770_770939

-- Conditions Definitions
def first_day_sales : ℕ := 45
def second_day_sales : ℕ := first_day_sales + 20
def third_day_sales : ℕ := 2 * second_day_sales - 10
def fourth_day_sales : ℕ := 120

-- Question: Prove that the total sales over the four days is 350.
theorem total_daisies_sold :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = 350 := by
  sorry

end total_daisies_sold_l770_770939


namespace shortest_distance_to_line_l770_770382

noncomputable def shortest_distance (x y : ℝ) : ℝ := 
  abs (3 * x + 4 * y - 2) / sqrt (3 ^ 2 + 4 ^ 2) - 3

theorem shortest_distance_to_line (x y : ℝ)
  (h : (x-5)^2 + (y-3)^2 = 9) : shortest_distance x y = 2 := 
sorry

end shortest_distance_to_line_l770_770382


namespace total_cost_over_8_weeks_l770_770635

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l770_770635


namespace searchlight_probability_l770_770415

noncomputable def time_for_full_revolution (revolutions_per_minute : ℝ) : ℝ :=
  (1 / revolutions_per_minute) * 60

def probability_stay_in_dark (time_in_dark time_for_full_revolution : ℝ) : ℝ :=
  time_in_dark / time_for_full_revolution

theorem searchlight_probability :
  probability_stay_in_dark 10 (time_for_full_revolution 3) = 1 / 2 :=
by
  sorry

end searchlight_probability_l770_770415


namespace probability_between_C_and_D_l770_770287

noncomputable def length_AB (x : ℝ) : ℝ := x
noncomputable def length_AD (x : ℝ) : ℝ := x / 4
noncomputable def length_BC (x : ℝ) : ℝ := x / 3 

theorem probability_between_C_and_D (x : ℝ) (h : 0 < x) :
    let AD := length_AD x,
        BC := length_BC x,
        DC := (2 * x / 3) - (x / 4) in
    DC / length_AB x = 5 / 12 := 
by
    -- Definitions to ensure the correctness
    set AB := length_AB x with hAB,
    set AD := length_AD x with hAD,
    set BC := length_BC x with hBC,
    set DC := (2 * x / 3) - (x / 4) with hDC,

    -- Final proof statement
    have h1 : DC = 5 * x / 12 := by
        calc
            DC = (2 * x / 3) - (x / 4) : by 
                exact hDC
            ... = (8 * x - 3 * x) / 12 : by
                ring_nf
            ... = 5 * x / 12 : by
                ring_nf,

    have h2 : AB = x := hAB,

    show DC / AB = 5 / 12,
    calc
        DC / AB = (5 * x / 12) / x : by
            rw [h1, h2]
        ... = 5 / 12 : by
            field_simp [h]

end probability_between_C_and_D_l770_770287


namespace man_l770_770884

theorem man's_salary (S : ℝ) 
  (h_food : S / 5)
  (h_rent : S / 10)
  (h_clothes : 3 * S / 5)
  (h_left : S - (S / 5 + S / 10 + 3 * S / 5) = 16000) : 
  S = 160000 := 
by
  sorry

end man_l770_770884


namespace problem1_problem2_l770_770725

-- Definitions of propositions p and q

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, ax^2 - ax + 1 > 0

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 3^x - 9^x < a - 1

-- Theorem for part (1)
theorem problem1 (a : ℝ) (hp : proposition_p a) : a ∈ Icc 0 4 := 
sorry

-- Theorem for part (2)
theorem problem2 (a : ℝ) (hp : proposition_p a) (hq : proposition_q a) : a ∈ Icc 1 4 := 
sorry

end problem1_problem2_l770_770725


namespace base_seven_to_base_ten_l770_770846

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770846


namespace sqrt_fourth_root_of_000001_is_0_to_nearest_tenth_l770_770926

noncomputable def nested_root_calculation : ℝ :=
  Float.round (Real.sqrt (Real.sqrt (10 ^ (-6 / 4, 80 / 4))))

theorem sqrt_fourth_root_of_000001_is_0_to_nearest_tenth :
  Float.roundWith nested_root_calculation = 0.0 :=
 by sorry

end sqrt_fourth_root_of_000001_is_0_to_nearest_tenth_l770_770926


namespace base_7_to_base_10_l770_770838

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770838


namespace area_difference_l770_770429

-- Given conditions
def AE : ℝ := 15
def DE : ℝ := 20
def AD : ℝ := 25 (by { linarith [(15:ℝ)^2 + (20:ℝ)^2 = 625] })
def area_tri_AED : ℝ := 150

-- Define the problem as a theorem
theorem area_difference (AGH CFH : ℝ) 
  (h_AED : sqrt (AE^2 + DE^2) = AD)
  (h_area_AED : 1/2 * AE * DE = area_tri_AED)
  (h_diff : AGH - CFH = 8.5) :
  (abs (AGH - CFH)) = 8.5 := 
by 
  sorry  -- proof omitted

end area_difference_l770_770429


namespace find_n_l770_770503

def C (k : ℕ) : ℕ :=
  if k = 1 then 0
  else (Nat.factors k).eraseDup.foldr (· + ·) 0

theorem find_n (n : ℕ) : 
  (∀ n, (C (2 ^ n + 1) = C n) ↔ n = 3) := 
by
  sorry

end find_n_l770_770503


namespace StacyBoughtPacks_l770_770298

theorem StacyBoughtPacks (sheets_per_pack days daily_printed_sheets total_packs : ℕ) 
  (h1 : sheets_per_pack = 240)
  (h2 : days = 6)
  (h3 : daily_printed_sheets = 80) 
  (h4 : total_packs = (days * daily_printed_sheets) / sheets_per_pack) : total_packs = 2 :=
by 
  sorry

end StacyBoughtPacks_l770_770298


namespace base7_to_base10_l770_770820

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770820


namespace ratio_of_e_to_l_l770_770395

-- Define the conditions
def e (S : ℕ) : ℕ := 4 * S
def l (S : ℕ) : ℕ := 8 * S

-- Prove the main statement
theorem ratio_of_e_to_l (S : ℕ) (h_e : e S = 4 * S) (h_l : l S = 8 * S) : e S / gcd (e S) (l S) / l S / gcd (e S) (l S) = 1 / 2 := by
  sorry

end ratio_of_e_to_l_l770_770395


namespace sum_of_proper_divisors_600_l770_770024

def sum_of_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ x, x < n) (Finset.divisors n)).sum id

theorem sum_of_proper_divisors_600 : sum_of_proper_divisors 600 = 1260 := 
by
  sorry

end sum_of_proper_divisors_600_l770_770024


namespace patients_per_doctor_l770_770876

theorem patients_per_doctor (patients : ℕ) (doctors : ℕ) (daily_patients : ℕ) (total_doctors : ℕ) :
  (daily_patients = 400) → 
  (total_doctors = 16) → 
  total_doctors * (patients / doctors) = daily_patients →
  patients / doctors = 25 :=
by
  intros hdaily hdoctors heq,
  sorry

end patients_per_doctor_l770_770876


namespace quadratic_distinct_roots_l770_770544

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770544


namespace num_two_digit_numbers_l770_770192

theorem num_two_digit_numbers {a b c : ℕ} (h : {a, b, c} = {1, 3, 4}) :
  (filter (λ x, x.fst ≠ x.snd)
    [ (a, b), (a, c), (b, a), (b, c), (c, a), (c, b) ]).length = 6 := sorry

end num_two_digit_numbers_l770_770192


namespace M_subsetneq_N_l770_770142

-- Definitions based on conditions in step a)
def M : Set ℕ := {x | ∃ a : ℕ+, x = a^2 + 1}
def N : Set ℕ := {x | ∃ b : ℕ+, x = b^2 - 4b + 5}

-- The final statement to prove
theorem M_subsetneq_N : M ⊂ N := by
  sorry

end M_subsetneq_N_l770_770142


namespace find_integer_N_l770_770951

theorem find_integer_N : ∃ N : ℤ, (N ^ 2 ≡ N [ZMOD 10000]) ∧ (N - 2 ≡ 0 [ZMOD 7]) :=
by
  sorry

end find_integer_N_l770_770951


namespace distance_blown_westward_l770_770738

theorem distance_blown_westward
  (time_traveled_east : ℕ)
  (speed : ℕ)
  (travelled_halfway : Prop)
  (new_location_fraction : ℚ) :
  time_traveled_east = 20 →
  speed = 30 →
  travelled_halfway →
  new_location_fraction = 1 / 3 →
  let distance_traveled_east := speed * time_traveled_east,
      total_distance := 2 * distance_traveled_east,
      new_location_distance := new_location_fraction * total_distance in
  distance_traveled_east - new_location_distance = 200 :=
begin
  intros,
  sorry
end

end distance_blown_westward_l770_770738


namespace suitableComprehensiveSurvey_l770_770859

/-- Given a set of survey methods, we want to prove which method is most suitable for a comprehensive survey. -/
theorem suitableComprehensiveSurvey :
  (∀ method ∈ ["A", "B", "C", "D"], method == "A" → "A" satisfies the criteria of a sample survey)
  → (∀ method ∈ ["A", "B", "C", "D"], method == "B" → "B" satisfies the criteria of a comprehensive survey)
  → (∀ method ∈ ["A", "B", "C", "D"], method == "C" → "C" satisfies the criteria of a sample survey)
  → (∀ method ∈ ["A", "B", "C", "D"], method == "D" → "D" satisfies the criteria of a sample survey)
  → "B" is the most suitable method for a comprehensive survey :=
by
  intros hA hB hC hD
  -- Insert appropriate proof steps here
  sorry

end suitableComprehensiveSurvey_l770_770859


namespace total_apples_picked_l770_770918

theorem total_apples_picked (benny_apples : ℕ) (dan_apples : ℕ) (h_benny : benny_apples = 2) (h_dan : dan_apples = 9) :
  benny_apples + dan_apples = 11 :=
by
  sorry

end total_apples_picked_l770_770918


namespace problem_solution_l770_770585

variable (f : ℝ → ℝ)

-- Let f be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f(x) = f(4 - x) for all x in ℝ
def satisfies_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (4 - x)

-- f is increasing on [0, 2]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem problem_solution :
  is_odd_function f →
  satisfies_symmetry f →
  is_increasing_on_interval f 0 2 →
  f 6 < f 4 ∧ f 4 < f 1 :=
by
  intros
  sorry

end problem_solution_l770_770585


namespace equilateral_triangle_side_length_l770_770746

noncomputable def triangle_area {α : Type*} [field α] (a b c : α) : α :=
1/2 * a * b * (real.sqrt 1 - (c/a)^2)

theorem equilateral_triangle_side_length
  (P Q R S : ℝ × ℝ)
  (PQ PR PS s : ℝ)
  (hPQ : PQ = 2)
  (hPR : PR = 3)
  (hPS : PS = 4)
  (insideP : P ∈ (triangle ABC))
  (feetQ : Q = foot P A B)
  (feetR : R = foot P B C)
  (feetS : S = foot P C A)
  :
  s = 12 * real.sqrt 3 :=
sorry

end equilateral_triangle_side_length_l770_770746


namespace soap_ratio_l770_770407

theorem soap_ratio :
  ∀ (total surveyed: Nat) (neither: Nat) (only_e: Nat) (both: Nat),
    surveyed = 200 →
    neither = 80 →
    only_e = 60 →
    both = 40 →
  let only_b := surveyed - (neither + only_e + both) in
    (only_b : both) = 1 : 2 :=
by
    intros;
    rw [←Nat.add_sub_cancel surveyed (neither + only_e + both)];
    have only_b_def: only_b = total surveyed - (neither + only_e + both);
    exact cast_rfl;
    sorry

end soap_ratio_l770_770407


namespace sum_cubes_mod_5_l770_770460

-- Define the function sum of cubes up to n
def sum_cubes (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 1) ^ 3)

-- Define the given modulo operation
def modulo (a b : ℕ) : ℕ := a % b

-- The primary theorem to prove
theorem sum_cubes_mod_5 : modulo (sum_cubes 50) 5 = 0 := sorry

end sum_cubes_mod_5_l770_770460


namespace product_evaluation_l770_770111

theorem product_evaluation (a : ℕ) (h : a = 6) : (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 720 :=
by
  rw [h]
  simp
  sorry

end product_evaluation_l770_770111


namespace surface_area_of_solid_of_revolution_l770_770802

theorem surface_area_of_solid_of_revolution :
  let side_length := 2 in
  let radius := side_length * (Real.sqrt 3 / 2) in
  let slant_height := side_length in
  let surface_area := 2 * π * radius * slant_height in
  surface_area = 4 * Real.sqrt 3 * π :=
by
  let side_length := 2
  let radius := side_length * (Real.sqrt 3 / 2)
  let slant_height := side_length
  let surface_area := 2 * π * radius * slant_height
  sorry

end surface_area_of_solid_of_revolution_l770_770802


namespace sum_of_cubes_mod_5_l770_770467

theorem sum_of_cubes_mod_5 :
  (∑ i in Finset.range 51, i^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_5_l770_770467


namespace sum_cubes_mod_5_l770_770465

-- Define the function sum of cubes up to n
def sum_cubes (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 1) ^ 3)

-- Define the given modulo operation
def modulo (a b : ℕ) : ℕ := a % b

-- The primary theorem to prove
theorem sum_cubes_mod_5 : modulo (sum_cubes 50) 5 = 0 := sorry

end sum_cubes_mod_5_l770_770465


namespace undergraduates_play_sports_l770_770400

theorem undergraduates_play_sports (total_students graduates_pct graduates_no_sport_pct undergraduates_no_sport_pct total_no_sport_pct : ℕ) 
    (total_students_eq : total_students = 800)
    (graduates_pct_eq : graduates_pct = 25)
    (graduates_no_sport_pct_eq : graduates_no_sport_pct = 50)
    (undergraduates_no_sport_pct_eq : undergraduates_no_sport_pct = 20)
    (total_no_sport_pct_eq : total_no_sport_pct = 30) 
    : (let graduates := graduates_pct * total_students / 100 in 
        let undergraduates := total_students - graduates in 
        let undergraduates_no_sport := undergraduates * undergraduates_no_sport_pct / 100 in 
        undergraduates - undergraduates_no_sport = 480) :=
by
  sorry

end undergraduates_play_sports_l770_770400


namespace cuts_x_axis_l770_770099

noncomputable def log_function (x : ℝ) : ℝ := log (-(x : ℝ)) / log (1/2)

theorem cuts_x_axis : ∃ x, x < 0 ∧ log_function x = 0 := by
  sorry

end cuts_x_axis_l770_770099


namespace number_of_ways_to_place_cards_l770_770285

theorem number_of_ways_to_place_cards :
  let cards := {1, 2, 3, 4, 5, 6}
  let envelopes := {1, 2, 3}
  (∃ (f : cards → envelopes), ∃ (e ∈ envelopes, (f 3 = e ∧ f 6 = e)) ∧
    (∀ e' ∈ envelopes, 2 = (f '' {1, 2, 3, 4, 5, 6}).count e')) →
  18 := sorry

end number_of_ways_to_place_cards_l770_770285


namespace max_cars_div_10_quotient_is_400_l770_770280

theorem max_cars_div_10_quotient_is_400 :
  let N := 4000 in
  N / 10 = 400 :=
by
  let N := 4000
  have h : N = 4000 := rfl
  calc
    N / 10 = 4000 / 10 : by rw h
    ... = 400 : by norm_num

end max_cars_div_10_quotient_is_400_l770_770280


namespace units_digit_product_is_2_l770_770025

def units_digit_product : ℕ := 
  (10 * 11 * 12 * 13 * 14 * 15 * 16) / 800 % 10

theorem units_digit_product_is_2 : units_digit_product = 2 := 
by
  sorry

end units_digit_product_is_2_l770_770025


namespace initial_cats_l770_770001

theorem initial_cats (C : ℕ) (h1 : 36 + 12 - 20 + C = 57) : C = 29 :=
by
  sorry

end initial_cats_l770_770001


namespace three_digit_numbers_with_sum_23_and_digits_at_least_5_l770_770970

theorem three_digit_numbers_with_sum_23_and_digits_at_least_5 :
  {n : ℕ | (100 * (n / 100) + 10 * ((n % 100) / 10) + (n % 10)) = n ∧
           5 ≤ (n / 100) ∧ (n / 100) ≤ 9 ∧
           5 ≤ ((n % 100) / 10) ∧ ((n % 100) / 10) ≤ 9 ∧
           5 ≤ (n % 10) ∧ (n % 10) ≤ 9 ∧
           (n / 100) + ((n % 100) / 10) + (n % 10) = 23 }.card = 13 :=
by
  sorry

end three_digit_numbers_with_sum_23_and_digits_at_least_5_l770_770970


namespace abs_eq_neg_of_le_zero_l770_770202

theorem abs_eq_neg_of_le_zero (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_of_le_zero_l770_770202


namespace qualified_flour_l770_770904

def is_qualified_flour (weight : ℝ) : Prop :=
  weight ≥ 24.75 ∧ weight ≤ 25.25

theorem qualified_flour :
  is_qualified_flour 24.80 ∧
  ¬is_qualified_flour 24.70 ∧
  ¬is_qualified_flour 25.30 ∧
  ¬is_qualified_flour 25.51 :=
by
  sorry

end qualified_flour_l770_770904


namespace range_and_monotonic_increase_cos_2alpha_value_l770_770716

-- Define the function f(x)
def f (x : Real) : Real :=
  sqrt 3 * cos (2 * Real.pi - x) - cos (Real.pi / 2 + x) + 1

-- Statement of the proof
theorem range_and_monotonic_increase :
  (∀ x, -1 ≤ f x ∧ f x ≤ 3) ∧
  (∀ k : ℤ, ∀ x, (k : ℤ) * (2 * Real.pi) - 5 * Real.pi / 6 ≤ x ∧ x ≤ (k : ℤ) * (2 * Real.pi) + Real.pi / 6 →
    ∀ y, x ≤ y → f x ≤ f y) :=
sorry

theorem cos_2alpha_value (α : Real) (hα1 : f α = 13 / 5) (hα2 : Real.pi / 6 < α ∧ α < 2 * Real.pi / 3) :
  cos (2 * α) = (7 - 24 * real.sqrt 3) / 50 :=
sorry

end range_and_monotonic_increase_cos_2alpha_value_l770_770716


namespace exclusive_movies_count_l770_770428

-- Define the conditions
def shared_movies : Nat := 15
def andrew_movies : Nat := 25
def john_movies_exclusive : Nat := 8

-- Define the result calculation
def exclusive_movies (andrew_movies shared_movies john_movies_exclusive : Nat) : Nat :=
  (andrew_movies - shared_movies) + john_movies_exclusive

-- Statement to prove
theorem exclusive_movies_count : exclusive_movies andrew_movies shared_movies john_movies_exclusive = 18 := by
  sorry

end exclusive_movies_count_l770_770428


namespace least_positive_integer_solution_l770_770121

theorem least_positive_integer_solution : 
  ∃ x : ℕ, x + 3567 ≡ 1543 [MOD 14] ∧ x = 6 := 
by
  -- proof goes here
  sorry

end least_positive_integer_solution_l770_770121


namespace different_lines_count_l770_770677

theorem different_lines_count : 
  let M := {1, 3}
  let N := {2, 4, 6}
  (M.card * N.card = 6) := by
  sorry

end different_lines_count_l770_770677


namespace quadratic_two_distinct_real_roots_l770_770520

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770520


namespace football_banquet_total_food_l770_770916

-- Definitions representing the conditions
def individual_max_food (n : Nat) := n ≤ 2
def min_guests (g : Nat) := g ≥ 160

-- The proof problem statement
theorem football_banquet_total_food : 
  ∀ (n g : Nat), (∀ i, i ≤ g → individual_max_food n) ∧ min_guests g → g * n = 320 := 
by
  intros n g h
  sorry

end football_banquet_total_food_l770_770916


namespace sum_abs_eq_l770_770203

theorem sum_abs_eq (a b : ℝ) (ha : |a| = 10) (hb : |b| = 7) (hab : a > b) : a + b = 17 ∨ a + b = 3 :=
sorry

end sum_abs_eq_l770_770203


namespace count_special_numbers_l770_770135

def is_term_dec (n : ℕ) : Bool :=
  ∃ a b : ℕ, n = 2^a * 5^b

def nonzero_hundredths_digit (n : ℕ) : Bool :=
  let dec := (1 / n.toReal).digits_base 10 in
  let len := dec.length in
  if len > 2 then dec[len - 3] ≠ 0
  else false

theorem count_special_numbers :
  { n ∈ finset.range 201 | is_term_dec n ∧ nonzero_hundredths_digit n }.card = 14 := 
by
  sorry

end count_special_numbers_l770_770135


namespace altitude_divides_side_in_1_3_ratio_l770_770319

-- Variables representing the side lengths of the triangle
variables (a b c : ℝ)

-- Conditions: side lengths a = sqrt(3), b = 2, c = sqrt(5),
-- and the altitude divides side of length b in the ratio 1:3.
def divides_in_ratio (a b c x : ℝ) : Prop :=
  a = Real.sqrt 3 ∧ b = 2 ∧ c = Real.sqrt 5 ∧
  m^2 + x^2 = a^2 ∧ m^2 + (b - x)^2 = c^2 ∧
  2 / 4 = x

theorem altitude_divides_side_in_1_3_ratio : ∃ x : ℝ, divides_in_ratio (Real.sqrt 3) 2 (Real.sqrt 5) x ∧ (x / (2 - x) = 1 / 3) :=
by
  sorry

end altitude_divides_side_in_1_3_ratio_l770_770319


namespace calculate_expression_l770_770437

theorem calculate_expression :
  (-((1: ℝ)^2022) + |(-real.sqrt 3)| + real.cbrt 8 - (1/2)^(-2)) = real.sqrt 3 - 3 := by
  sorry

end calculate_expression_l770_770437


namespace range_of_a_l770_770182

def f (x : ℝ) : ℝ := Real.cos (Real.pi * x)
def g (x a : ℝ) : ℝ := 2^x * a - (1 / 2)

theorem range_of_a (a : ℝ) : (∃ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ f x1 = g x2 a) ↔ (a ∈ Set.Icc (-1/2 : ℝ) 0 ∪ Set.Icc (0 : ℝ) (3/2 : ℝ)) :=
sorry

end range_of_a_l770_770182


namespace real_solutions_count_l770_770790

theorem real_solutions_count : 
  (x : ℝ) → ((x^2006 + 1) * (finset.sum (finset.range (1003)) (λ i, x^(2*i))) = 2006 * x^2005) → (x = 1) :=
sorry

end real_solutions_count_l770_770790


namespace quadratic_has_distinct_real_roots_l770_770533

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770533


namespace base_area_min_volume_l770_770885

theorem base_area_min_volume (d : ℝ) : 
  ∃ S : ℝ, ∀ V_min, 
    (∃ (PABCD : Type) (a h : ℝ), 
      let M := point (P : PABCD),
      let K := point (P : PABCD),
      let PM := line (M : PABCD),
      let MK := d,
      pyramid PABCD ∧ 
      plane_bisect_dihedral (plane M K) PM) → 
       is_minimal_volume_pyramid V_min PABCD → 
       (S = 8 * d ^ 2) := 
begin
  sorry
end

end base_area_min_volume_l770_770885


namespace table_price_l770_770419

theorem table_price (C T : ℝ) (h1 : 2 * C + T = 0.6 * (C + 2 * T)) (h2 : C + T = 96) : T = 84 := by
  sorry

end table_price_l770_770419


namespace sum_cubes_mod_five_l770_770492

theorem sum_cubes_mod_five :
  (∑ n in Finset.range 50, (n + 1)^3) % 5 = 0 :=
by
  sorry

end sum_cubes_mod_five_l770_770492


namespace range_of_a_l770_770140

theorem range_of_a (a : ℝ) (x : ℝ) (h : cos x = (2 * a - 3) / (4 - a)) (hx : (π / 2 < x ∧ x < π) ∨ (π < x ∧ x < 3 * π / 2)) :
  -1 < a ∧ a < 3 / 2 :=
by
  sorry

end range_of_a_l770_770140


namespace mike_average_rate_l770_770375

def total_distance : ℝ := 640
def half_distance : ℝ := total_distance / 2
def first_half_rate : ℝ := 80

def first_half_time : ℝ := half_distance / first_half_rate
def second_half_time : ℝ := 3 * first_half_time
def total_time : ℝ := first_half_time + second_half_time
def average_rate (d : ℝ) (t : ℝ) : ℝ := d / t

theorem mike_average_rate : average_rate total_distance total_time = 40 := by
  -- We need to verify that the Lean statement builds successfully.
  sorry

end mike_average_rate_l770_770375


namespace tetrahedron_constant_distance_l770_770562

-- Define a regular tetrahedron
structure Tetrahedron where
  vertices : Fin 4 → (ℝ × ℝ × ℝ)
  is_regular : ∀ i j k l, i ≠ j → j ≠ k → k ≠ l → dist (vertices i) (vertices j) = dist (vertices k) (vertices l)

-- Define a point inside a tetrahedron
def point_inside_tetrahedron (T : Tetrahedron) (P : ℝ × ℝ × ℝ) : Prop :=
  ∃ a b c d, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ a + b + c + d = 1 ∧
  P = a • T.vertices ⟨0⟩ + b • T.vertices ⟨1⟩ + c • T.vertices ⟨2⟩ + d • T.vertices ⟨3⟩

-- Define the distance from a point to a face of a tetrahedron
def distance_to_face (P : ℝ × ℝ × ℝ) (face : set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Main theorem
theorem tetrahedron_constant_distance (T : Tetrahedron) :
  (∀ P, point_inside_tetrahedron T P → ∑ face in T.faces, distance_to_face P face) = T.constant :=
sorry

end tetrahedron_constant_distance_l770_770562


namespace trader_profit_per_meter_l770_770899

theorem trader_profit_per_meter :
  ∀ (total_meters : ℕ) (total_profit : ℝ), 
    total_meters = 40 → 
    total_profit = 2200 → 
    (total_profit / total_meters) = 55 := 
by
  intros total_meters total_profit h_meters h_profit
  rw [h_meters, h_profit]
  norm_num
  -- proof skipped
  sorry

end trader_profit_per_meter_l770_770899


namespace dihedral_angle_CFG_E_l770_770718

variables {A B C D E F G : EuclideanGeometry.Point}
variables (tetrahedron : EuclideanGeometry.Tetrahedron A B C D)
variables (midpoint_AB : EuclideanGeometry.Midpoint A B E)
variables (midpoint_BC : EuclideanGeometry.Midpoint B C F)
variables (midpoint_CD : EuclideanGeometry.Midpoint C D G)

theorem dihedral_angle_CFG_E :
  EuclideanGeometry.dihedral_angle C F G E = Real.pi - Real.arccot (Real.sqrt 2 / 2) :=
sorry

end dihedral_angle_CFG_E_l770_770718


namespace clock_angle_at_3_18_l770_770018

theorem clock_angle_at_3_18 :
  let minute_angle := (18 / 60) * 360
      hour_angle := (3 * 30) + (18 / 60) * 30
  in |minute_angle - hour_angle| = 9 := 
by {
  let minute_angle := (18 / 60) * 360
  let hour_angle := (3 * 30) + (18 / 60) * 30
  have h1 : minute_angle = 108 := by norm_num
  have h2 : hour_angle = 99 := by norm_num
  have h3 : |minute_angle - hour_angle| = |108 - 99| := by rw [h1, h2]
  have h4 : |108 - 99| = 9 := by norm_num
  rw [h3, h4]
  norm_num
}

end clock_angle_at_3_18_l770_770018


namespace evaluate_expression_l770_770937

theorem evaluate_expression :
  (|(-1 : ℝ)|^2023 + (Real.sqrt 3)^2 - 2 * Real.sin (Real.pi / 6) + (1 / 2)⁻¹ = 5) :=
by
  sorry

end evaluate_expression_l770_770937


namespace probability_of_selecting_cocaptains_l770_770563

open Nat

-- Define the sizes of the teams
def team_sizes : List ℕ := [6, 8, 9, 10]

-- Define the calculation of the probability of selecting two co-captains for a given team size
def probability_co_captains (n : ℕ) : ℚ :=
  1 / (choose n 2)

-- Define the overall probability
def total_probability : ℚ :=
  (1/4) * ((probability_co_captains 6) + (probability_co_captains 8) +
           (probability_co_captains 9) + (probability_co_captains 10))

theorem probability_of_selecting_cocaptains :
    total_probability = 131 / 5040 :=
  sorry

end probability_of_selecting_cocaptains_l770_770563


namespace sqrt_fourth_root_x_is_point2_l770_770922

def x : ℝ := 0.000001

theorem sqrt_fourth_root_x_is_point2 :
  (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 0.000001)))) ≈ 0.2 := sorry

end sqrt_fourth_root_x_is_point2_l770_770922


namespace cost_per_jar_of_food_l770_770767

theorem cost_per_jar_of_food
  (food_per_half_pound : ℝ)
  (total_turtle_weight : ℝ)
  (ounces_per_jar : ℝ)
  (total_cost : ℝ)
  (jar_cost : ℝ) :
  food_per_half_pound = 1 →
  total_turtle_weight = 30 →
  ounces_per_jar = 15 →
  total_cost = 8 →
  jar_cost = (total_cost / (total_turtle_weight * (2 / food_per_half_pound) / ounces_per_jar)) →
  jar_cost = 2 :=
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end cost_per_jar_of_food_l770_770767


namespace xiao_liang_correct_l770_770221

theorem xiao_liang_correct :
  ∀ (x : ℕ), (0 ≤ x ∧ x ≤ 26 ∧ 30 - x ≤ 24 ∧ 26 - x ≤ 20) →
  let boys_A := x
  let girls_A := 30 - x
  let boys_B := 26 - x
  let girls_B := 24 - girls_A
  ∃ k : ℤ, boys_A - girls_B = 6 := 
by 
  sorry

end xiao_liang_correct_l770_770221


namespace partition_complex_set_l770_770187

noncomputable theory

variables {z : ℕ → ℂ}

def S (n : ℕ) : set ℂ := {z i | i ∈ finset.range n ∧ z i ≠ 0}

theorem partition_complex_set : 
  ∃ (A B C : set ℂ) (S = A ∪ B ∪ C),
    (∀ z ∈ A, ∀ z' ∈ A, angle z z' ≤ π / 2) ∧ 
    (∀ z ∈ B, ∀ z' ∈ B, angle z z' ≤ π / 2) ∧
    (∀ z ∈ C, ∀ z' ∈ C, angle z z' ≤ π / 2) ∧ 
    (∀ z1 ∈ A, ∀ z2 ∈ B, angle z1 z2 > π / 2) ∧
    (∀ z1 ∈ A, ∀ z2 ∈ C, angle z1 z2 > π / 2) ∧
    (∀ z1 ∈ B, ∀ z2 ∈ C, angle z1 z2 > π / 2) := 
sorry

end partition_complex_set_l770_770187


namespace number_of_ferns_is_six_l770_770690

def num_fronds_per_fern : Nat := 7
def num_leaves_per_frond : Nat := 30
def total_leaves : Nat := 1260

theorem number_of_ferns_is_six :
  total_leaves = num_fronds_per_fern * num_leaves_per_frond * 6 :=
by
  sorry

end number_of_ferns_is_six_l770_770690


namespace sum_of_bases_l770_770673

theorem sum_of_bases (R₁ R₂ : ℕ) 
    (h1 : (4 * R₁ + 5) / (R₁^2 - 1) = (3 * R₂ + 4) / (R₂^2 - 1))
    (h2 : (5 * R₁ + 4) / (R₁^2 - 1) = (4 * R₂ + 3) / (R₂^2 - 1)) : 
    R₁ + R₂ = 23 := 
sorry

end sum_of_bases_l770_770673


namespace sqrt_fourth_root_nearest_tenth_l770_770928

theorem sqrt_fourth_root_nearest_tenth :
  (Real.sqrt (Real.root 4 0.000001)) = 0.2 :=
by
  -- proof steps would go here
  sorry

end sqrt_fourth_root_nearest_tenth_l770_770928


namespace prob_equiv_l770_770591

theorem prob_equiv : 
  (∃ x : ℤ, 3^x + 3^x + 3^x + 3^x = 1458) → (∃ x : ℤ, (x + 2) * (x - 2) = 12) :=
by
  sorry

end prob_equiv_l770_770591


namespace find_constants_monotonicity_l770_770180

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_constants (a b c : ℝ) 
  (h1 : f' (-2/3) a b = 0)
  (h2 : f' 1 a b = 0) :
  a = -1/2 ∧ b = -2 :=
by sorry

theorem monotonicity (a b c : ℝ)
  (h1 : a = -1/2) 
  (h2 : b = -2) : 
  (∀ x : ℝ, x < -2/3 → f' x a b > 0) ∧ 
  (∀ x : ℝ, -2/3 < x ∧ x < 1 → f' x a b < 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a b > 0) :=
by sorry

end find_constants_monotonicity_l770_770180


namespace max_dist_to_origin_from_curve_l770_770996

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  let x := 3 + Real.sin θ
  let y := Real.cos θ
  (x, y)

theorem max_dist_to_origin_from_curve :
  ∃ M : ℝ × ℝ, (∃ θ : ℝ, M = curve θ) ∧ Real.sqrt (M.fst^2 + M.snd^2) ≤ 4 :=
by
  sorry

end max_dist_to_origin_from_curve_l770_770996


namespace proposition_implication_l770_770938

variables (p q r : Prop)

theorem proposition_implication:
  (¬p ∧ q ∧ ¬r → (¬p → q) → r = false) ∧
  (p ∧ q ∧ ¬r → (¬p → q) → r = false) ∧
  (p ∧ ¬q ∧ r → (¬p → q) → r = true) ∧
  (¬p ∧ ¬q ∧ r → (¬p → q) → r = true) →
  (finset.filter id
    [ (¬ (¬p ∧ q ∧ ¬r → (¬p → q) → r)),
      (¬ (p ∧ q ∧ ¬r → (¬p → q) → r)),
      ((p ∧ ¬q ∧ r → (¬p → q) → r)),
      ((¬p ∧ ¬q ∧ r → (¬p → q) → r)) 
    ]).card = 2 :=
by 
  sorry

end proposition_implication_l770_770938


namespace arctan_tan_addition_l770_770921

theorem arctan_tan_addition :
  ∃ (θ : ℝ), 0 ≤ θ ∧ θ ≤ 180 ∧ 
  (θ = 75) ∧ 
  Real.arctan (Real.tan 75 + 3 * Real.tan 15) = θ := 
by
  use 75
  split
  exact le_refl 75 -- 0 ≤ 75
  split
  norm_num -- 75 ≤ 180
  split
  norm_num -- θ = 75
  sorry -- The full trigonometric simplification here

end arctan_tan_addition_l770_770921


namespace probability_ratio_l770_770456

noncomputable def p := 10 / (nat.choose 50 5)
noncomputable def q := 2250 / (nat.choose 50 5)

theorem probability_ratio : q / p = 225 := 
by
  sorry

end probability_ratio_l770_770456


namespace largest_number_l770_770865

-- Definitions based on the conditions.
def hcf : ℕ := 62
def factor1 : ℕ := 11
def factor2 : ℕ := 12
def lcm := hcf * factor1 * factor2

-- Theorem: The largest of the two numbers.
theorem largest_number (H : ∀ a b : ℕ, lcm = a * b ∧ Nat.gcd a b = hcf) : 
  ∃ a b, a * b = lcm ∧ Nat.gcd a b = hcf ∧ (a = hcf * factor2 ∨ b = hcf * factor2) ∧ (a = 744 ∨ b = 744) :=
by
  use 62 * 12
  use 11
  simp [hcf, factor2, factor1, lcm, Nat.gcd_comm, Nat.gcd]
  sorry

end largest_number_l770_770865


namespace fraction_of_students_dislike_but_actually_like_l770_770915

theorem fraction_of_students_dislike_but_actually_like :
  ∀ (total_students : ℕ) 
    (likes_dancing : ℕ)
    (dislikes_dancing : ℕ)
    (says_likes_likes : ℕ)
    (says_likes_dislikes : ℕ)
    (says_dislikes_likes : ℕ)
    (says_dislikes_dislikes : ℕ)
    (h1 : likes_dancing = total_students * 60 / 100)
    (h2 : dislikes_dancing = total_students - likes_dancing)
    (h3 : says_likes_likes = likes_dancing * 80 / 100)
    (h4 : says_likes_dislikes = likes_dancing * 20 / 100)
    (h5 : says_dislikes_dislikes = dislikes_dancing * 90 / 100)
    (h6 : says_dislikes_likes = dislikes_dancing * 10 / 100)
    (h7 : says_likes_dislikes + says_dislikes_dislikes = total_students * 50 / 100),
  (says_likes_dislikes : ℕ) / (says_likes_dislikes + says_dislikes_dislikes) = 1 / 4 :=
by {
  -- Assuming a total of 100 students makes the arithmetic cleaner
  let total_students := 100,
  -- Definitions based on conditions
  let likes_dancing := 60,
  let dislikes_dancing := 40,
  let says_likes_likes := 48,
  let says_likes_dislikes := 12,
  let says_dislikes_dislikes := 36,
  let says_dislikes_likes := 4,
  have h1 := rfl,
  have h2 := rfl,
  have h3 := rfl,
  have h4 := rfl,
  have h5 := rfl,
  have h6 := rfl,
  have h7 := rfl,
  sorry
}

end fraction_of_students_dislike_but_actually_like_l770_770915


namespace distinct_paintings_l770_770680

-- Define the conditions of the problem
def num_segments : Nat := 8
def num_blue : Nat := 4
def num_red : Nat := 3
def num_green : Nat := 1

-- Define the symmetries: rotations and reflections
def rotations : List Nat := [0, 45, 90, 135, 180, 225, 270, 315]
def reflections : Nat := 8  -- 4 through vertices and 4 through midpoints

-- The problem statement to prove
theorem distinct_paintings : 
  ∃ n : Nat, 
    (n = 26 ∧
      (num_blue + num_red + num_green = num_segments) ∧
      ∀ seg in 1..num_segments,
        ∃ color : (seg ≤ num_blue + num_red → seg ≤ num_blue → "blue" ∨ "red") → "green") :=
begin
  sorry
end

end distinct_paintings_l770_770680


namespace antonio_correct_answers_l770_770661

theorem antonio_correct_answers :
  ∃ c w : ℕ, c + w = 15 ∧ 6 * c - 3 * w = 36 ∧ c = 9 :=
by
  sorry

end antonio_correct_answers_l770_770661


namespace sum_of_cubes_mod_five_l770_770487

theorem sum_of_cubes_mod_five : 
  (∑ n in Finset.range 51, n^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_five_l770_770487


namespace magic_school_project_l770_770134

noncomputable def expected_days_to_grow_apple_trees : ℚ := 
  -- Harmonic number calculation
  let H_6 := (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/5 + 1/6 
  in 2 * H_6  -- times 2 due to expected value calculation

theorem magic_school_project : 
  let m := 49
  let n := 10
  E(X) = (49 : ℚ) / 10 ∧ 100 * m + n = 4910 :=
by
  sorry

end magic_school_project_l770_770134


namespace vector_dot_product_in_equilateral_triangle_l770_770669

noncomputable def equilateral_triangle_side_length : ℝ := 1

def vector_equality (u v : ℝ) := u = v

variable (A B C D E : Type)
variable [MetricSpace A]
variable (AB AC BC BD BE CA CE AD : ℝ)

axiom equilateral_triangle (A B C : Type) [MetricSpace A] (AB AC BC : ℝ) :
  AB = AC ∧ AC = BC ∧ AB = 1

axiom midpoint_condition (BC BD : ℝ) : 2 * BD = BC
axiom segment_division_condition (CA CE : ℝ) : 3 * CE = CA

axiom vector_conditions (AB AD AC BC CE BE : ℝ) :
  ∃ A B C D E : Type, vector_equality BC (2 * BD) ∧ vector_equality CA (3 * CE) ∧ AD.dot BE = -1 / 4

theorem vector_dot_product_in_equilateral_triangle :
  ∃ A B C D E : Type, equilateral_triangle A B C ∧ vector_conditions AB AD AC BC CE BE := 
begin
  sorry,
end

end vector_dot_product_in_equilateral_triangle_l770_770669


namespace find_AM_length_l770_770153

noncomputable def AM_length (A B C D : Type) [MetricSpace A B C D] 
  (hAB : dist A B = 3) (hBD : dist B D = 3) (hAC : dist A C = 5) (hCD : dist C D = 5)
  (hAD : dist A D = 4) (hBC : dist B C = 4) : ℝ :=
let M := barycenter (insert B (insert C {D})) in
dist A M

theorem find_AM_length 
  (A B C D : Type) [MetricSpace A B C D] 
  (hAB : dist A B = 3) (hBD : dist B D = 3) (hAC : dist A C = 5) (hCD : dist C D = 5)
  (hAD : dist A D = 4) (hBC : dist B C = 4) : 
  AM_length A B C D hAB hBD hAC hCD hAD hBC = 10 / 3 := by 
sorry

end find_AM_length_l770_770153


namespace solve_inequality_l770_770764

theorem solve_inequality (x : ℝ) (h : x ≠ -1) : (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 :=
by
  sorry

end solve_inequality_l770_770764


namespace find_f_2547_l770_770694

theorem find_f_2547 (f : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) :
  f 2547 = 2547 :=
sorry

end find_f_2547_l770_770694


namespace determinant_closed_form_l770_770967

-- Definitions of initial conditions and recurrence relation
def d : ℕ → ℤ
| 1 := 2
| 2 := 3
| n := 2 * d (n - 1) - d (n - 2)

-- Theorem statement
theorem determinant_closed_form (n : ℕ) : n ≥ 1 → d n = n + 1 :=
by
  sorry

end determinant_closed_form_l770_770967


namespace quadratic_distinct_roots_l770_770542

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770542


namespace sequence_formula_l770_770988

theorem sequence_formula (a : ℕ+ → ℕ) (h₁ : a 4 = 20)
  (h₂ : ∀ n : ℕ+, a (n + 1) = 2 * a n - n + 1) :
  ∀ n : ℕ+, a n = 2^n + n :=
begin
  sorry
end

end sequence_formula_l770_770988


namespace newspaper_spending_over_8_weeks_l770_770632

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l770_770632


namespace alice_bracelets_given_away_l770_770423

theorem alice_bracelets_given_away
    (total_bracelets : ℕ)
    (cost_of_materials : ℝ)
    (price_per_bracelet : ℝ)
    (profit : ℝ)
    (bracelets_given_away : ℕ)
    (bracelets_sold : ℕ)
    (total_revenue : ℝ)
    (h1 : total_bracelets = 52)
    (h2 : cost_of_materials = 3)
    (h3 : price_per_bracelet = 0.25)
    (h4 : profit = 8)
    (h5 : total_revenue = profit + cost_of_materials)
    (h6 : total_revenue = price_per_bracelet * bracelets_sold)
    (h7 : total_bracelets = bracelets_sold + bracelets_given_away) :
    bracelets_given_away = 8 :=
by
  sorry

end alice_bracelets_given_away_l770_770423


namespace interior_diagonal_length_l770_770803

variables (a b c : ℝ)

-- Conditions
def surface_area_eq : Prop := 2 * (a * b + b * c + c * a) = 22
def edge_length_eq : Prop := 4 * (a + b + c) = 24

-- Question to be proved
theorem interior_diagonal_length :
  surface_area_eq a b c → edge_length_eq a b c → (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14) :=
by
  intros h1 h2
  sorry

end interior_diagonal_length_l770_770803


namespace cargo_arrival_in_days_l770_770875

-- Definitions for conditions
def days_navigate : ℕ := 21
def days_customs : ℕ := 4
def days_transport : ℕ := 7
def days_departed : ℕ := 30

-- Calculate the days since arrival in Vancouver
def days_arrival_vancouver : ℕ := days_departed - days_navigate

-- Calculate the days since customs processes finished
def days_since_customs_done : ℕ := days_arrival_vancouver - days_customs

-- Calculate the days for cargo to arrive at the warehouse from today
def days_until_arrival : ℕ := days_transport - days_since_customs_done

-- Expected number of days from today for the cargo to arrive at the warehouse
theorem cargo_arrival_in_days : days_until_arrival = 2 := by
  -- Insert the proof steps here
  sorry

end cargo_arrival_in_days_l770_770875


namespace complex_modulus_conjugate_l770_770717

theorem complex_modulus_conjugate (z_1 : ℂ) : complex.norm_sq z_1 = complex.norm_sq (conj z_1) := 
by sorry

end complex_modulus_conjugate_l770_770717


namespace polygon_sides_eq_four_l770_770217

theorem polygon_sides_eq_four (n : ℕ)
  (h_interior : (n - 2) * 180 = 360)
  (h_exterior : ∀ (m : ℕ), m = n -> 360 = 360) :
  n = 4 :=
sorry

end polygon_sides_eq_four_l770_770217


namespace area_of_intersecting_equilateral_triangles_in_square_l770_770346

-- Define the side length of the square
def side_length : ℝ := real.sqrt 7

-- Define the height of the equilateral triangles
def height_of_equilateral_triangle (s : ℝ) : ℝ := (real.sqrt 3) / 2 * s

-- Define the vertical overlap of the equilateral triangles
def vertical_overlap (h : ℝ) (s : ℝ) : ℝ := 2 * h - s

-- Calculate the area of the rhombus formed by the intersection of the two triangles
def area_of_rhombus (d1 d2 : ℝ) : ℝ := 1 / 2 * d1 * d2

theorem area_of_intersecting_equilateral_triangles_in_square :
  area_of_rhombus (vertical_overlap (height_of_equilateral_triangle side_length) side_length) side_length
    = (7 * real.sqrt 3) / 2 - 7 / 2 := by
  sorry

end area_of_intersecting_equilateral_triangles_in_square_l770_770346


namespace quadratic_roots_condition_l770_770511

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770511


namespace domain_of_g_l770_770651

def f : ℝ → ℝ := sorry

theorem domain_of_g 
  (hf_dom : ∀ x, -2 ≤ x ∧ x ≤ 4 → f x = f x) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 ↔ (∃ y, y = f x + f (-x)) := 
by {
  sorry
}

end domain_of_g_l770_770651


namespace smallest_period_pi_l770_770610

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - cos (2 * x)

theorem smallest_period_pi : ∀ x, f (x + π) = f x :=
by
  sorry

end smallest_period_pi_l770_770610


namespace polynomial_in_y_l770_770100

theorem polynomial_in_y {x y : ℝ} (h₁ : x^3 - 6 * x^2 + 11 * x - 6 = 0) (h₂ : y = x + 1/x) :
  x^2 * (y^2 + y - 6) = 0 :=
sorry

end polynomial_in_y_l770_770100


namespace john_school_year_hours_l770_770689

noncomputable def requiredHoursPerWeek (summerHoursPerWeek : ℕ) (summerWeeks : ℕ) 
                                       (summerEarnings : ℕ) (schoolWeeks : ℕ) 
                                       (schoolEarnings : ℕ) : ℕ :=
    schoolEarnings * summerHoursPerWeek * summerWeeks / (summerEarnings * schoolWeeks)

theorem john_school_year_hours :
  ∀ (summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings : ℕ),
    summerHoursPerWeek = 40 →
    summerWeeks = 10 →
    summerEarnings = 4000 →
    schoolWeeks = 50 →
    schoolEarnings = 4000 →
    requiredHoursPerWeek summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings = 8 :=
by
  intros
  sorry

end john_school_year_hours_l770_770689


namespace cot_identity_proof_l770_770371

theorem cot_identity_proof (α β : ℝ) :
  4.51 * (Real.cot α)^2 + (Real.cot β)^2 - 2 * (Real.cos (β - α)) / (Real.sin α * Real.sin β) + 2 = 
  (Real.sin (α - β))^2 / (Real.sin α * Real.sin α * Real.sin β * Real.sin β) :=
by sorry

end cot_identity_proof_l770_770371


namespace base_7_to_base_10_l770_770833

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770833


namespace delta_solution_l770_770643

theorem delta_solution : ∃ Δ : ℤ, 4 * (-3) = Δ - 1 ∧ Δ = -11 :=
by
  -- Using the condition 4(-3) = Δ - 1, 
  -- we need to prove that Δ = -11
  sorry

end delta_solution_l770_770643


namespace correct_options_l770_770031

-- Option A condition
def optionA_condition : Prop :=
  ∀ θ k, θ = (π / 2 + 2 * k * π) ↔ θ = π / 2 + k * π

-- Option B condition
def optionB_condition (α : ℝ) : Prop :=
  (π / 2 < α ∧ α < π) → (π / 4 < α / 2 ∧ α / 2 < π / 2) ∨ (5 * π / 4 < α / 2 ∧ α / 2 < 3 * π / 2)

-- Option C condition
def optionC_condition : Prop :=
  ∀ (angles : list ℝ), (angles.sum = π) → (∀ α ∈ angles, (0 < α ∧ α < π / 2) ∨ (π / 2 < α ∧ α < π))

-- Option D condition
def optionD_condition : Prop :=
  ∀ S α R, (S = 4) ∧ (α = 2) ∧ (S = 1 / 2 * α * R^2) → (R = 2) ∧ (α * R = 4)

-- Theorem to prove Options B and D are correct
theorem correct_options : (optionB_condition ∧ optionD_condition) :=
by
sorry

end correct_options_l770_770031


namespace range_of_x_l770_770104

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x :
  ∀ x : ℝ, (f x > f (2*x - 1)) ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end range_of_x_l770_770104


namespace total_potatoes_l770_770397

theorem total_potatoes (cooked_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) (H1 : cooked_potatoes = 7) (H2 : time_per_potato = 5) (H3 : remaining_time = 45) : (cooked_potatoes + (remaining_time / time_per_potato) = 16) :=
by
  sorry

end total_potatoes_l770_770397


namespace max_rooks_in_one_part_l770_770159

theorem max_rooks_in_one_part (n : ℕ) (hn : 0 < n) :
  ∃ (part₁ part₂ : set (ℕ × ℕ))
    (hpart : part₁ ∪ part₂ = { p | p.1 < 2 * n ∧ p.2 < 2 * n } ∧
           part₁ ∩ part₂ = ∅ ∧
           connected part₁ ∧ connected part₂ ∧
           symmetric part₁ part₂),
    let rooks := { (i, j) | mutually_non_attacking (i, j) (2 * n)},
    ∀ (part : set (ℕ × ℕ)), part ∈ {part₁, part₂} → ∃ (k : ℕ), k = 2 * n - 1 ∧ (set.filter (fun p => rooks p) part).card = k :=
sorry

end max_rooks_in_one_part_l770_770159


namespace mb_less_than_neg_one_l770_770776

theorem mb_less_than_neg_one (m b : ℝ) (h1 : (0 : ℝ), (3 : ℝ)) (h2 : (2 : ℝ), (-1 : ℝ))
    (hf : ∀ x : ℝ, y = m * x + b) : (m * b < -1) :=
begin
  sorry
end

end mb_less_than_neg_one_l770_770776


namespace smallest_n_for_divisibility_property_l770_770361

theorem smallest_n_for_divisibility_property :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n^2 + n % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n^2 + n % k ≠ 0) ∧ 
  ∀ m : ℕ, 0 < m ∧ m < n → ¬ ((∀ k : ℕ, 1 ≤ k ∧ k ≤ m → m^2 + m % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ m ∧ m^2 + m % k ≠ 0)) := sorry

end smallest_n_for_divisibility_property_l770_770361


namespace incorrect_order_l770_770222

open Set

/-
Given cities and roads represented as a graph, with distinct tolls on roads and ensuring connection between any two cities. After removing the most expensive road from any routes that visit each city exactly once, if it becomes impossible to travel between cities A and B, B and C, and C and A, then the king's order was followed incorrectly.
-/

variables (Cities : Type) [Nonempty Cities] [Fintype Cities]
variables (Road : Cities → Cities → ℝ) (distinct_tolls : ∀ a b : Cities, a ≠ b → Road a b ≠ Road b a)
variables (A B C : Cities)
variable [DecidableEq Cities]  -- Ensure decidability of equality for cities

-- Hypothesis that Roads create a connected graph initially
variable (h_connected : ∀ (a b : Cities), a ≠ b → ∃ (path : List Cities), path.head = some a ∧ path.last = some b ∧ ∀ i, Road (path.nth i) (path.nth (i+1)) ↑.val ∈ Road)

-- Hypothesis that closing marked roads makes travel between A, B, and C impossible
variable (h_impossible : ∀ (R: Cities → Cities → ℝ) (annotated: Cities → Cities → Prop),
  annotated(A, B) ∧ annotated(B, C) ∧ annotated(C, A) → ∀ a b c, 
  ¬ (∃ (p: List Cities), p.head = some a ∧ p.last = some b ∧ ∀ i, 
    annotated(p.nth i) (p.nth (i + 1)) → Road (p.nth i) (p.nth (i + 1)) != Road b c))

-- The final goal is to prove the order was followed incorrectly
theorem incorrect_order
  (hp: ∀ a b c, ∃ (p: List Cities),
        p.head = some a ∧ p.last = some b ∧ ∃ edge xy, 
          xy ∈ (p.map Road) ∧
          P.max xy ≠ most expensive edge }) (h_impossible: ¬ (h_connected Cities Road)) : false
  :=
  by 
    -- proof steps would go here
    sorry

end incorrect_order_l770_770222


namespace value_of_c_distinct_real_roots_l770_770554

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770554


namespace factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l770_770114

-- Problem 1: Prove the factorization of x^4 - 3x^2 + 1
theorem factorize_x4_minus_3x2_plus_1 (x : ℝ) : 
  x^4 - 3 * x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := 
by
  sorry

-- Problem 2: Prove the factorization of a^5 + a^4 - 2a + 1
theorem factorize_a5_plus_a4_minus_2a_plus_1 (a : ℝ) : 
  a^5 + a^4 - 2 * a + 1 = (a^2 + a - 1) * (a^3 + a - 1) := 
by
  sorry

-- Problem 3: Prove the factorization of m^5 - 2m^3 - m - 1
theorem factorize_m5_minus_2m3_minus_m_minus_1 (m : ℝ) : 
  m^5 - 2 * m^3 - m - 1 = (m^3 + m^2 + 1) * (m^2 - m - 1) := 
by
  sorry

end factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l770_770114


namespace system_of_equations_solution_l770_770118

theorem system_of_equations_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : x1 + x2 = x3^2) (h2 : x2 + x3 = x4^2)
  (h3 : x3 + x4 = x5^2) (h4 : x4 + x5 = x1^2) (h5 : x5 + x1 = x2^2) :
  x1 = 2 ∧ x2 = 2 ∧ x3 = 2 ∧ x4 = 2 ∧ x5 = 2 := 
sorry

end system_of_equations_solution_l770_770118


namespace sum_of_cubes_mod_5_l770_770471

theorem sum_of_cubes_mod_5 :
  (∑ i in Finset.range 51, i^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_5_l770_770471


namespace locus_of_M_l770_770253

-- Definitions of the centers of the incircle and excircle
variables {A B C M I J I_M J_M : Type}
variable [InCenter A B C I]
variable [ExCenter A B C J]
variable [InCenter B C M I_M]
variable [ExCenter B C M J_M]

-- Definitions of coordinates in the plane
variables {B C : Point}
variable {M : Point} (hM : M ∉ Line B C)

-- Rectangular condition for the quadrilateral formed by these points
def formsRectangle (I I_M J J_M : Point) : Prop :=
  ∠(I, I_M) = 90 ∧ ∠(I, J_M) = 90 ∧ ∠(J, I_M) = 90 ∧ ∠(J, J_M) = 90

-- Final theorem to prove
theorem locus_of_M : formsRectangle I I_M J J_M ↔ M ∈ arc BAC excluding {B, C} :=
sorry

end locus_of_M_l770_770253


namespace least_number_to_subtract_l770_770037

/-- The least number which should be subtracted from 3,381 so that the remainder when divided by 9, 11, and 17 will in each case be 8 is 7. -/
theorem least_number_to_subtract (n : ℕ) (h₁ : n = 3381) (h₂ : ∀ m, ((h₁ - m) % 9 = 8) ∧ ((h₁ - m) % 11 = 8) ∧ ((h₁ - m) % 17 = 8)) : n - 7 = 3374 :=
by
  -- Proof is not required, replacing with sorry
  sorry

end least_number_to_subtract_l770_770037


namespace quadratic_equation_roots_difference_l770_770096

theorem quadratic_equation_roots_difference
  (p q : ℝ)
  (h_pos_p : 0 < p)
  (h_pos_q : 0 < q)
  (h_diff : (real.sqrt (p^2 - 4 * q))^2 = 9) :
  p = real.sqrt (4 * q + 9) := 
sorry

end quadratic_equation_roots_difference_l770_770096


namespace base_7_to_10_of_23456_l770_770827

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770827


namespace sum_abcde_l770_770693

theorem sum_abcde (a b c d e : ℝ) (h₀ : c = 5) (h₁ : a + b = c) (h₂ : a + b + c = d) (h₃ : a + b + c + d = e) : 
  a + b + c + d + e = 40 :=
by
  verify sorry

end sum_abcde_l770_770693


namespace distinct_prime_factors_product_divisors_of_60_l770_770700

theorem distinct_prime_factors_product_divisors_of_60 : ∃ N : ℕ, (N = ∏ d in (finset.filter (∣ 60) (finset.range 61)), d) ∧ nat.factors N = 3 :=
by
  sorry

end distinct_prime_factors_product_divisors_of_60_l770_770700


namespace base7_to_base10_l770_770816

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770816


namespace calc_expression_l770_770936

theorem calc_expression :
  ((
    (
      ((3 + 2)⁻¹ + 2)⁻¹ + 2
    )⁻¹ + 2
  )⁻¹ + 2
  )⁻¹ + 2 = 157 / 65 := 
  sorry

end calc_expression_l770_770936


namespace slope_of_tangent_at_x1_l770_770127

theorem slope_of_tangent_at_x1 : 
  (∃ (y : ℝ → ℝ) (dy_dx : ℝ → ℝ), ∀ x : ℝ, 2 * x^2 + 4 * x + 6 * (y x) = 24 ∧ (dy_dx x = - (2 / 3) * x - 2 / 3)) →
  dy_dx 1 = - 4 / 3 := 
sorry

end slope_of_tangent_at_x1_l770_770127


namespace find_m_l770_770190

def vector_collinear (a b : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, a = (λ * b.1, λ * b.2)

def a (m : ℝ) : ℝ × ℝ := (2 * m + 1, 3)
def b (m : ℝ) : ℝ × ℝ := (2, m)

theorem find_m (m : ℝ) : vector_collinear (a m) (b m) ↔ (m = -3/2 ∨ m = 2) :=
sorry

end find_m_l770_770190


namespace quadratic_two_distinct_real_roots_l770_770515

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l770_770515


namespace problem_1_problem_2_l770_770144

def condition_p (x : ℝ) : Prop := 4 * x ^ 2 + 12 * x - 7 ≤ 0
def condition_q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Problem 1: When a=0, if p is true and q is false, the range of real numbers x
theorem problem_1 (x : ℝ) :
  condition_p x ∧ ¬ condition_q 0 x ↔ -7/2 ≤ x ∧ x < -3 := sorry

-- Problem 2: If p is a sufficient condition for q, the range of real numbers a
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, condition_p x → condition_q a x) ↔ -5/2 ≤ a ∧ a ≤ -1/2 := sorry

end problem_1_problem_2_l770_770144


namespace smallest_k_condition_l770_770151

theorem smallest_k_condition (n k : ℕ) (h_n : n ≥ 2) (h_k : k = 2 * n) :
  ∀ (f : Fin n → Fin n → Fin k), (∀ i j, f i j < k) →
  (∃ a b c d : Fin n, a ≠ c ∧ b ≠ d ∧ f a b ≠ f a d ∧ f a b ≠ f c b ∧ f a b ≠ f c d ∧ f a d ≠ f c b ∧ f a d ≠ f c d ∧ f c b ≠ f c d) :=
sorry

end smallest_k_condition_l770_770151


namespace solution_set_l770_770205

theorem solution_set (x : ℝ) (h : x ∈ {-1, 1, 2, x^2 - 2*x}) : 
  x ∈ {-1, 1, 2, 0, 3} :=
sorry

end solution_set_l770_770205


namespace next_in_seq_1_next_in_seq_2_next_in_seq_3_next_in_seq_4_next_in_seq_5_l770_770873

-- Define the sequences
def seq_1 : List ℕ := [17, 27, 47, 87, 167]
def seq_2 : List ℕ := [1, 2, 3, 5, 8, 13]
def seq_3 : List ℕ := [1, 3, 6, 10, 15, 21]
def seq_4 : List ℕ := [7, 9, 11, 13, 15]
def seq_5 : List ℕ := [2, 3, 5, 7, 11, 13, 17]

-- We will now state that the next elements in the sequences are as found
theorem next_in_seq_1 : List.nth seq_1 5 327 ∧ List.nth seq_1 6 647 := by sorry

theorem next_in_seq_2 : List.nth seq_2 6 21 ∧ List.nth seq_2 7 34 := by sorry

theorem next_in_seq_3 : List.nth seq_3 6 28 ∧ List.nth seq_3 7 36 := by sorry

theorem next_in_seq_4 : List.nth seq_4 5 17 ∧ List.nth seq_4 6 19 := by sorry

theorem next_in_seq_5 : List.nth seq_5 7 19 ∧ List.nth seq_5 8 23 := by sorry

end next_in_seq_1_next_in_seq_2_next_in_seq_3_next_in_seq_4_next_in_seq_5_l770_770873


namespace cotangent_expression_l770_770273

variables (a b c : ℝ) (α β γ : ℝ)

-- Assume a, b, c are the lengths of the sides of a triangle, and α, β, γ are the corresponding angles.
-- Assume a^2 + b^2 = 2023 * c^2.
def is_triangle_lengths (a b c : ℝ) (α β γ : ℝ) : Prop :=
  a^2 + b^2 = 2023 * c^2

-- Assume α, β, and γ are the angles opposite to the sides a, b, and c respectively, and the angles sum up to 180 degrees.
def opposite_angles (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α + β + γ = π

-- The theorem to prove the given math problem
theorem cotangent_expression 
  (a b c α β γ : ℝ)
  (ht : is_triangle_lengths a b c α β γ)
  (ho : opposite_angles a b c α β γ) :
  (Real.cot α) / (Real.cot β + Real.cot γ) = 1011 :=
sorry -- Proof goes here

end cotangent_expression_l770_770273


namespace quadratic_has_distinct_real_roots_l770_770525

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770525


namespace base_7_to_base_10_l770_770835

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770835


namespace eq_line_through_midpoints_area_triangle_ABC_l770_770791

noncomputable def point (x y : ℝ) := (x, y)

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem eq_line_through_midpoints :
  let A := point 5 1
  let B := reflect_x A
  let C := reflect_origin A
  let M_AB := midpoint A B
  let M_BC := midpoint B C
  in 2 * (M_BC.1 - M_AB.1) = - (M_AB.2 - M_BC.2) :=
sorry

theorem area_triangle_ABC :
  let A := point 5 1
  let B := reflect_x A
  let C := reflect_origin A
  in abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 10 :=
sorry

end eq_line_through_midpoints_area_triangle_ABC_l770_770791


namespace midpoint_AB_slope_l_l770_770238

/-- Problem 1: Given the parametric equations of a line l and a curve C, show that 
if α=π/3, the coordinates of the midpoint M of segment AB are (12/13, -√3/13). -/
theorem midpoint_AB (α : ℝ) (M : ℝ × ℝ) : 
  α = π / 3 →
  let l := (λ t : ℝ, (2 + t * cos α, sqrt 3 + t * sin α)) in
  let C := (λ θ : ℝ, (2 * cos θ, sin θ)) in
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
     (∃ tA tB : ℝ, l tA = A ∧ l tB = B) ∧
     (∃ θA θB : ℝ, C θA = A ∧ C θB = B)) →
  M = (12 / 13, - sqrt 3 / 13) → 
  true := 
by 
  sorry

/-- Problem 2: Given the parametric equations of a line l and a curve C, show that 
the slope of the line l is √5/4 given that |PA| * |PB| = |OP|^2 and P = (2, √3). -/
theorem slope_l (α : ℝ) (slope : ℝ) : 
  let P := (2, sqrt 3) in
  let l := (λ t : ℝ, (2 + t * cos α, sqrt 3 + t * sin α)) in
  let C := (λ θ : ℝ, (2 * cos θ, sin θ)) in
  (∀ A B : ℝ × ℝ, 
     (∃ tA tB : ℝ, l tA = A ∧ l tB = B) ∧ 
     (∃ θA θB : ℝ, C θA = A ∧ C θB = B) ∧ 
     A ≠ B ∧ 
     (let PA := (A.1 - P.1)^2 + (A.2 - P.2)^2 in
      let PB := (B.1 - P.1)^2 + (B.2 - P.2)^2 in
      let OP := P.1^2 + P.2^2 in
      PA * PB = OP^2)) →
  slope = sqrt 5 / 4 → 
  true := 
by 
  sorry

end midpoint_AB_slope_l_l770_770238


namespace imaginary_part_of_conjugate_l770_770316

def complex_num : ℂ := (1 / (1 + 2 * complex.I)) + (complex.I / 2)

theorem imaginary_part_of_conjugate : complex.im (complex.conj complex_num) = -1 / 10 :=
  sorry

end imaginary_part_of_conjugate_l770_770316


namespace Tony_change_l770_770343

structure Conditions where
  bucket_capacity : ℕ := 2
  sandbox_depth : ℕ := 2
  sandbox_width : ℕ := 4
  sandbox_length : ℕ := 5
  sand_weight_per_cubic_foot : ℕ := 3
  water_per_4_trips : ℕ := 3
  water_bottle_cost : ℕ := 2
  water_bottle_volume: ℕ := 15
  initial_money : ℕ := 10

theorem Tony_change (conds : Conditions) : 
  let volume_sandbox := conds.sandbox_depth * conds.sandbox_width * conds.sandbox_length
  let total_weight_sand := volume_sandbox * conds.sand_weight_per_cubic_foot
  let trips := total_weight_sand / conds.bucket_capacity
  let drinks := trips / 4
  let total_water := drinks * conds.water_per_4_trips
  let bottles_needed := total_water / conds.water_bottle_volume
  let total_cost := bottles_needed * conds.water_bottle_cost
  let change := conds.initial_money - total_cost
  change = 4 := by
  /- calculations corresponding to steps that are translated from the solution to show that the 
     change indeed is $4 -/
  sorry

end Tony_change_l770_770343


namespace acute_triangle_inequality_l770_770748

-- Variables for sides, medians, and altitudes
variables {a b c : ℝ} (△ABC : triangle)
variables (ma mb mc ha hb hc : ℝ)
variables (R r : ℝ)

-- Definitions of acute triangle, medians, altitudes, circumradius, and inradius
def acute_triangle (△ABC : triangle) : Prop :=
  ∠A < π / 2 ∧ ∠B < π / 2 ∧ ∠C < π / 2

def median (a : ℝ) : Prop :=
  ∃ ma, ma = (1/2) * (sqrt (2 * b^2 + 2 * c^2 - a^2))

def altitude (ha : ℝ) (a : ℝ) := ∃ ha, ha = 2 * (area △ABC) / a

def circumradius (R : ℝ) : Prop := 
  ∃ R, R = (a * b * c)/(4 * area △ABC)

def inradius (r : ℝ) : Prop := 
  ∃ r, r = (area △ABC) / (s)

-- Main theorem statement 
theorem acute_triangle_inequality (△ABC : triangle) 
(acute : acute_triangle △ABC) (med_a : median a ma) (med_b : median b mb) (med_c : median c mc)
(alt_a : altitude ha a) (alt_b : altitude hb b) (alt_c : altitude hc c)
(circumradius_defn : circumradius R) (inradius_defn : inradius r) :
ma/ha + mb/hb + mc/hc ≤ 1 + R / r := 
sorry,

end acute_triangle_inequality_l770_770748


namespace sum_of_cubes_mod_5_l770_770481

theorem sum_of_cubes_mod_5 :
  ( ∑ k in Finset.range 50, (k + 1)^3 ) % 5 = 0 :=
sorry

end sum_of_cubes_mod_5_l770_770481


namespace part_one_value_of_x_part_two_month_of_last_repayment_l770_770321

def total_loan : ℕ := 24000
def first_year_salary_per_month : ℕ := 1500
def base_repayment_per_month : ℕ := 500
def basic_living_expenses : ℕ := 3000
def max_salary : ℕ := 4000
def interest_rate : ℝ := 1.05
def repay_period : ℕ := 36

theorem part_one_value_of_x (x : ℕ) (h_sum : 12 * 500 + (500 + x) + (500 + 2 * x) + ... + (500 + 24 * x) = total_loan) : 
  x = 20 := 
sorry

theorem part_two_month_of_last_repayment (x : ℕ) (h_x : x = 50) (n : ℕ) 
  (h_sum2 : 12 * 500 + (500 + 50) + (500 + 100) + ... + (500 + (n - 12) * 50) >= total_loan) 
  (salary_in_nth_month : ℝ := first_year_salary_per_month * interest_rate^19) :
  n = 31 ∧ salary_in_nth_month - 450 >= basic_living_expenses := 
sorry

end part_one_value_of_x_part_two_month_of_last_repayment_l770_770321


namespace sum_cubes_mod_five_l770_770491

theorem sum_cubes_mod_five :
  (∑ n in Finset.range 50, (n + 1)^3) % 5 = 0 :=
by
  sorry

end sum_cubes_mod_five_l770_770491


namespace at_most_three_prime_divisors_l770_770586

-- Given a and b natural numbers such that a > 1, b is divisible by a^2,
-- and any divisor of b less than sqrt a is also a divisor of a, 
-- we need to prove that a has no more than three distinct prime divisors.

theorem at_most_three_prime_divisors
  (a b : ℕ) 
  (h1 : 1 < a)
  (h2 : a * a ∣ b)
  (h3 : ∀ d : ℕ, d ∣ b → d < Math.sqrt a → d ∣ a) 
  : finset.card (nat.factors a).to_finset ≤ 3 :=
sorry

end at_most_three_prime_divisors_l770_770586


namespace arrangement_32_exists_arrangement_100_exists_l770_770383

theorem arrangement_32_exists :
  ∃ (arrangement : List ℕ), 
    (arrangement.perm (List.range 32).map (+1)) ∧
    ∀ a b, a ≠ b → a ∈ arrangement → b ∈ arrangement → 
    ¬ (∃ k, k ∈ arrangement ∧ k = (a + b) / 2 ∧ a < k ∧ k < b) := by
  sorry

theorem arrangement_100_exists :
  ∃ (arrangement : List ℕ), 
    (arrangement.perm (List.range 100).map (+1)) ∧
    ∀ a b, a ≠ b → a ∈ arrangement → b ∈ arrangement → 
    ¬ (∃ k, k ∈ arrangement ∧ k = (a + b) / 2 ∧ a < k ∧ k < b) := by
  sorry

end arrangement_32_exists_arrangement_100_exists_l770_770383


namespace positive_difference_of_complementary_ratio_5_1_l770_770785

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l770_770785


namespace Q_mod_m_l770_770505

open Nat

def Q (m : ℕ) : ℕ :=
  (List.range m).filter (fun x => gcd x m = 1)
  |> List.prod

theorem Q_mod_m {m : ℕ} (h: m ≥ 3) : Q m ≡ 1 [MOD m] ∨ Q m ≡ -1 [MOD m] :=
by sorry

end Q_mod_m_l770_770505


namespace quadratic_has_distinct_real_roots_l770_770535

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l770_770535


namespace quadratic_has_distinct_real_roots_l770_770522

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770522


namespace distance_between_ports_l770_770063

theorem distance_between_ports (x : ℝ) (speed_ship : ℝ) (speed_water : ℝ) (time_difference : ℝ) 
  (speed_downstream := speed_ship + speed_water) 
  (speed_upstream := speed_ship - speed_water) 
  (time_downstream := x / speed_downstream) 
  (time_upstream := x / speed_upstream) 
  (h : time_downstream + time_difference = time_upstream) 
  (h_ship : speed_ship = 26)
  (h_water : speed_water = 2)
  (h_time : time_difference = 3) : x = 504 :=
by
  -- The proof is omitted 
  sorry

end distance_between_ports_l770_770063


namespace seq_2009_eq_1_l770_770576

def seq (n : ℕ) : ℕ :=
  match n with
  | 1     => 1
  | 2     => 3
  | n + 3 => |seq (n + 1) - seq (n + 2)|

theorem seq_2009_eq_1 : seq 2009 = 1 :=
by
  sorry

end seq_2009_eq_1_l770_770576


namespace sum_of_cubes_mod_5_l770_770470

theorem sum_of_cubes_mod_5 :
  (∑ i in Finset.range 51, i^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_5_l770_770470


namespace solve_cubic_root_equation_l770_770965

theorem solve_cubic_root_equation (x : ℝ) : 
  (∃ x,  x = 2 * Real.sqrt 34 ∨ x = -2 * Real.sqrt 34) ↔ (Real.cbrt (7 - x^2 / 4) = -3) :=
by
  sorry

end solve_cubic_root_equation_l770_770965


namespace integral_solution_l770_770932

noncomputable def integral_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
∫ x in 0..∞, (sin (a * x)) / (x * (x^2 + b^2))

theorem integral_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  integral_problem a b ha hb = (π / (2 * b^2)) * (1 - exp (-a * b)) :=
sorry

end integral_solution_l770_770932


namespace log_base_b_find_l770_770116

theorem log_base_b_find : ∀ (b : ℝ), log b 729 = -2/3 → b = 1 / (3^9) :=
by
  sorry

end log_base_b_find_l770_770116


namespace T_shape_perimeter_l770_770012

def rectangle1_length : ℕ := 3
def rectangle1_width : ℕ := 5
def rectangle2_length : ℕ := 2
def rectangle2_width : ℕ := 6
def overlap : ℕ := 1

theorem T_shape_perimeter :
  2 * (rectangle1_length + rectangle1_width) + 
  2 * (rectangle2_length + rectangle2_width) - 
  2 * overlap = 30 :=
by 
  simp [rectangle1_length, rectangle1_width, rectangle2_length, rectangle2_width, overlap]
  sorry

end T_shape_perimeter_l770_770012


namespace find_y_intercept_l770_770896

def line_y_intercept (m x y : ℝ) (pt : ℝ × ℝ) : ℝ :=
  let y_intercept := pt.snd - m * pt.fst
  y_intercept

theorem find_y_intercept (m x y b : ℝ) (pt : ℝ × ℝ) (h1 : m = 2) (h2 : pt = (498, 998)) :
  line_y_intercept m x y pt = 2 :=
by
  sorry

end find_y_intercept_l770_770896


namespace value_of_c_distinct_real_roots_l770_770556

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770556


namespace john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l770_770251

theorem john_needs_to_sell_1200_pencils_to_make_120_dollars_profit :
  ∀ (buy_rate_pencils : ℕ) (buy_rate_dollars : ℕ) (sell_rate_pencils : ℕ) (sell_rate_dollars : ℕ),
    buy_rate_pencils = 5 →
    buy_rate_dollars = 7 →
    sell_rate_pencils = 4 →
    sell_rate_dollars = 6 →
    ∃ (n_pencils : ℕ), n_pencils = 1200 ∧ 
                        (sell_rate_dollars / sell_rate_pencils - buy_rate_dollars / buy_rate_pencils) * n_pencils = 120 :=
by
  sorry

end john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l770_770251


namespace proportion_correct_l770_770644

theorem proportion_correct (m n : ℤ) (h : 6 * m = 7 * n) (hn : n ≠ 0) : (m : ℚ) / 7 = n / 6 :=
by sorry

end proportion_correct_l770_770644


namespace sum_of_cubes_mod_l770_770475

theorem sum_of_cubes_mod (S : ℕ) (h : S = ∑ n in Finset.range 51, n^3) : S % 5 = 0 :=
sorry

end sum_of_cubes_mod_l770_770475


namespace least_cost_of_grass_seed_l770_770881

-- Definitions of the prices and weights
def price_per_bag (size : Nat) : Float :=
  if size = 5 then 13.85
  else if size = 10 then 20.40
  else if size = 25 then 32.25
  else 0.0

-- The conditions for the weights and costs
def valid_weight_range (total_weight : Nat) : Prop :=
  65 ≤ total_weight ∧ total_weight ≤ 80

-- Calculate the total cost given quantities of each bag size
def total_cost (bag5 : Nat) (bag10 : Nat) (bag25 : Nat) : Float :=
  Float.ofNat bag5 * price_per_bag 5 + Float.ofNat bag10 * price_per_bag 10 + Float.ofNat bag25 * price_per_bag 25

-- Correct cost for the minimum possible cost within the given weight range
def min_possible_cost : Float := 98.75

-- Proof statement to be proven
theorem least_cost_of_grass_seed : ∃ (bag5 bag10 bag25 : Nat), 
  valid_weight_range (bag5 * 5 + bag10 * 10 + bag25 * 25) ∧ total_cost bag5 bag10 bag25 = min_possible_cost :=
sorry

end least_cost_of_grass_seed_l770_770881


namespace necessary_but_not_sufficient_l770_770593

section geometric_progression

variables {a b c : ℝ}

def geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a = b / r ∧ c = b * r

def necessary_condition (a b c : ℝ) : Prop :=
  a * c = b^2

theorem necessary_but_not_sufficient :
  (geometric_progression a b c → necessary_condition a b c) ∧
  (¬ (necessary_condition a b c → geometric_progression a b c)) :=
by sorry

end geometric_progression

end necessary_but_not_sufficient_l770_770593


namespace quadratic_has_distinct_real_roots_l770_770528

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770528


namespace largest_square_perimeter_is_28_l770_770886

-- Definitions and assumptions
def rect_length : ℝ := 10
def rect_width : ℝ := 7

-- Define the largest possible square
def largest_square_side := rect_width

-- Define the perimeter of a square
def perimeter_of_square (side : ℝ) : ℝ := 4 * side

-- Proving statement
theorem largest_square_perimeter_is_28 :
  perimeter_of_square largest_square_side = 28 := 
  by 
    -- sorry is used to skip the proof
    sorry

end largest_square_perimeter_is_28_l770_770886


namespace sum_of_face_products_ne_zero_l770_770674

-- Definition of a cube with vertices and assigned values
structure Cube where
  vertices : Fin 8 → ℤ
  v_values : ∀ i, vertices i = 1 ∨ vertices i = -1

-- Define the product of the numbers at the vertices of a face
def face_product (c : Cube) (f : Fin 6 → Fin 4) : ℤ :=
  ∏ i in finrange 4, c.vertices (f i)

-- Prove that the sum of the defined 14 numbers is never equal to 0
theorem sum_of_face_products_ne_zero (c : Cube) (faces : Fin 6 → Fin 4) (opposite_face_pairs : Fin 6 → (Fin 4 × Fin 4)) :
  ∑ i in finrange 8, face_product c (faces i) + ∑ j in finrange 6, (face_product c (opposite_face_pairs j).1 + face_product c (opposite_face_pairs j).2) ≠ 0 :=
sorry -- Proof to be filled; the statement itself is in line with the problem 

end sum_of_face_products_ne_zero_l770_770674


namespace determinant_of_tangents_l770_770257

axiom A B C : ℝ
axiom is_obtuse_triangle : A + B + C = π ∧ A > π / 2

theorem determinant_of_tangents :
  ∃ (β γ : ℝ), tan (A) = β ∧ tan (B) = γ ∧ 
  ∀ A B C : ℝ, (is_obtuse_triangle) →
  det ![![tan A, 1, 1], ![1, tan B, 1], ![1, 1, tan C]] = 2 :=
begin
  sorry
end

end determinant_of_tangents_l770_770257


namespace valid_cut_off_square_l770_770284

theorem valid_cut_off_square : 
  ∃ config : list (list (option ℕ)), 
    let original_net := (list.map some (fin.mk (2*2 + 1*2) _)) in
    length original_net = 10 ∧
    (list.filter_map id config).length = 9 :=
begin
  sorry
end

end valid_cut_off_square_l770_770284


namespace sum_of_largest_int_series_l770_770702

noncomputable def largest_int_not_exceeding (x : ℝ) : ℤ := int.floor x

theorem sum_of_largest_int_series :
  (∑ k in finset.range 2020, largest_int_not_exceeding (4^k / 5)) = (4^2020 - 1) / 15 - 1010 :=
by
  let ⟨a, r⟩ := (1 / 5 : ℝ, 4 : ℝ)
  sorry

end sum_of_largest_int_series_l770_770702


namespace min_cut_length_l770_770422

-- Definitions of the conditions from part a)
def stick_lengths : List Nat := [9, 18, 21]

-- Main theorem statement
theorem min_cut_length : ∃ x : Nat, (∀ a b c ∈ stick_lengths, a ∈ stick_lengths → b ∈ stick_lengths → c ∈ stick_lengths → a ≠ b → b ≠ c → a ≠ c → ¬ (a - x + b - x > c - x ∨ b - x + c - x > a - x ∨ a - x + c - x > b - x)) ∧ x = 6 :=
begin
  -- This proof is intentionally left incomplete.
  sorry
end

end min_cut_length_l770_770422


namespace trapezoid_area_calc_l770_770672

noncomputable def isoscelesTrapezoidArea : ℝ :=
  let a := 1
  let b := 9
  let h := 2 * Real.sqrt 3
  0.5 * (a + b) * h

theorem trapezoid_area_calc : isoscelesTrapezoidArea = 20 * Real.sqrt 3 := by
  sorry

end trapezoid_area_calc_l770_770672


namespace sqrt_log2_8_log4_8_l770_770454

theorem sqrt_log2_8_log4_8:
  sqrt ((log 2 8) + (log 4 8)) = (3 * sqrt 2) / 2 :=
by
  sorry

end sqrt_log2_8_log4_8_l770_770454


namespace map_line_segments_l770_770947

def point : Type := ℝ × ℝ

def transformation (f : point → point) (p q : point) : Prop := f p = q

def counterclockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

def clockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

theorem map_line_segments :
  (transformation counterclockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation counterclockwise_rotation_180 (2, -5) (-2, 5)) ∨
  (transformation clockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation clockwise_rotation_180 (2, -5) (-2, 5)) :=
by
  sorry

end map_line_segments_l770_770947


namespace count_three_digit_numbers_l770_770641

open Finset

/-- The number of three-digit numbers where the hundreds digit is greater than the tens digit,
and the tens digit is greater than the ones digit. -/
theorem count_three_digit_numbers (U : set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  (U.to_finset.choose 3).card = 84 :=
by sorry

end count_three_digit_numbers_l770_770641


namespace world_grain_supply_is_correct_l770_770139

def world_grain_demand : ℝ := 2400000
def supply_ratio : ℝ := 0.75
def world_grain_supply (demand : ℝ) (ratio : ℝ) : ℝ := ratio * demand

theorem world_grain_supply_is_correct :
  world_grain_supply world_grain_demand supply_ratio = 1800000 := 
by 
  sorry

end world_grain_supply_is_correct_l770_770139


namespace fraction_rented_rooms_l770_770741

variables (R : ℝ) (N : ℝ) (fraction_rented : ℝ)

-- Define the conditions
def total_rooms := R
def ac_rooms := (3 / 5) * R
def rented_ac_rooms := (2 / 3) * ac_rooms
def nonrented_rooms := R - N
def nonrented_ac_rooms := 0.8 * N
def ac_rooms_not_rented := ac_rooms - rented_ac_rooms

-- Prove that the fraction of rented rooms is 3/4 given the conditions
theorem fraction_rented_rooms (h1 : ac_rooms = (3 / 5) * R)
                             (h2 : rented_ac_rooms = (2 / 5) * R)
                             (h3 : ac_rooms_not_rented = (1 / 5) * R) 
                             (h4 : nonrented_ac_rooms = ac_rooms_not_rented)
                             (h5 : N = (1 / 4) * R) :
                             fraction_rented = 3 / 4 :=
by sorry


end fraction_rented_rooms_l770_770741


namespace scientific_notation_of_845_billion_l770_770681

/-- Express 845 billion yuan in scientific notation. -/
theorem scientific_notation_of_845_billion :
  (845 * (10^9 : ℝ)) / (10^9 : ℝ) = 8.45 * 10^3 :=
by
  sorry

end scientific_notation_of_845_billion_l770_770681


namespace acute_angle_at_3_18_l770_770019

def degrees_of_hand_position (hour marks min marks: ℕ) : ℝ :=
  (hour marks % 12) * 30 + (min marks / 60) * 30

def minute_hand_position (min marks: ℕ) : ℝ :=
  min marks * 6

def diff_positions (pos1 pos2 : ℝ) : ℝ :=
  if pos1 > pos2 then pos1 - pos2 else pos2 - pos1

theorem acute_angle_at_3_18 :
  let hour_pos := degrees_of_hand_position 3 (18 : ℕ),
  let min_pos := minute_hand_position (18 : ℕ),
  diff_positions hour_pos min_pos = 9 :=
by {
  sorry
}

end acute_angle_at_3_18_l770_770019


namespace three_digit_numbers_count_l770_770638

theorem three_digit_numbers_count :
  let is_valid_number (h t u : ℕ) := 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h > t ∧ t > u
  in (∑ h in finset.range 10, ∑ t in finset.range h, ∑ u in finset.range t, if is_valid_number h t u then 1 else 0) = 84 :=
by
  let is_valid_number (h t u : ℕ) := 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h>t ∧ t>u
  have : ∑ h in finset.range 10, ∑ t in finset.range h, ∑ u in finset.range t, if is_valid_number h t u then 1 else 0 = 84 := by {
    -- skipping the proof
    sorry
  }
  exact this

end three_digit_numbers_count_l770_770638


namespace complex_modulus_l770_770573

theorem complex_modulus (z : ℂ) (hz : z * (1 + complex.i) = 1 - complex.i) : complex.abs z = 1 :=
by sorry

end complex_modulus_l770_770573


namespace last_letter_of_95th_permutation_is_D_l770_770768

theorem last_letter_of_95th_permutation_is_D :
  let letters := "BDEKO".to_list,
      perm := Nat.factorial 5,
      nth_perm := List.permutations letters,
      sorted_permutations := nth_perm.to_list.quicksort in
  sorted_permutations.nth! 94 = ['K', 'O', 'E', 'B', 'D'] ->
  sorted_permutations.nth! 94.last = 'D' :=
by
  sorry

end last_letter_of_95th_permutation_is_D_l770_770768


namespace quadratic_has_distinct_real_roots_l770_770526

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770526


namespace quadratic_distinct_roots_l770_770539

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770539


namespace positive_difference_of_complementary_angles_l770_770779

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l770_770779


namespace point_not_on_graph_l770_770083

theorem point_not_on_graph :
  ¬ (1 * 5 = 6) :=
by 
  sorry

end point_not_on_graph_l770_770083


namespace infinite_lines_parallel_to_plane_infinite_lines_in_plane_parallel_to_m_l770_770003

-- Define the condition of a point and a plane in the context
variable (Point : Type) (Plane : Type) (Line : Type)

-- Condition: There exists a point outside a plane
variable (P : Point) (α : Plane)
variable (outside_plane : ¬ P ∈ α)

-- Condition: A line m is parallel to a plane α
variable (m : Line)
variable (parallel_m_α : ∀ P1 P2, P1 ∈ m → P2 ∈ m → (P1 ∉ α ∧ P2 ∉ α) ∨ (P1 ∈ α ∧ P2 ∈ α))

-- Theorem 1: Through a point outside a plane, there are infinitely many lines parallel to the plane
theorem infinite_lines_parallel_to_plane (P : Point) (α : Plane) (H : ¬ P ∈ α) :
  ∃ (L : Line), L ∉ α ∧ ∀ (Q : Point), Q ∈ L → ¬ Q ∈ α :=
sorry

-- Theorem 2: If line m is parallel to plane α, then there are infinitely many lines in plane α that are parallel to m
theorem infinite_lines_in_plane_parallel_to_m (m : Line) (α : Plane) (H : ∀ P1 P2, P1 ∈ m → P2 ∈ m → (P1 ∉ α ∧ P2 ∉ α) ∨ (P1 ∈ α ∧ P2 ∈ α)) :
  ∃ (L : Line), ∀ (P : Point), P ∈ L → (∃ (Q : Point), Q ∈ α ∧ ∀ (R : Point), R ∈ L → P ∈ α ∧ Q ∈ m :=
sorry

end infinite_lines_parallel_to_plane_infinite_lines_in_plane_parallel_to_m_l770_770003


namespace cast_cost_l770_770004

theorem cast_cost (C : ℝ) 
  (visit_cost : ℝ := 300)
  (insurance_coverage : ℝ := 0.60)
  (out_of_pocket_cost : ℝ := 200) :
  0.40 * (visit_cost + C) = out_of_pocket_cost → 
  C = 200 := by
  sorry

end cast_cost_l770_770004


namespace ratio_of_radii_l770_770058

-- Define volumes of the spheres
def V1 := 450 * Real.pi
def V2 := 0.08 * V1

-- Define ratio of volumes
def volume_ratio := V2 / V1

-- Given conditions:
axiom volume_V1 : V1 = 450 * Real.pi
axiom volume_V2 : V2 = 0.08 * V1

-- Prove the ratio of radii
theorem ratio_of_radii : ∃ r1 r2 : ℝ, (V2 / V1 = (r2 / r1) ^ 3) ∧ (r2 / r1 = Real.cbrt (2 / 25)) := by
  sorry

end ratio_of_radii_l770_770058


namespace work_done_by_force_l770_770393

def F (x : ℝ) := 4 * x - 1

theorem work_done_by_force :
  let a := 1
  let b := 3
  (∫ x in a..b, F x) = 14 := by
  sorry

end work_done_by_force_l770_770393


namespace no_partition_equal_product_l770_770697

theorem no_partition_equal_product (p : ℕ) (a : ℤ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 3) :
  ¬ ∃ (s t : Finset ℤ), (s ∪ t = Finset.range (p-1) + a) ∧ (s ∩ t = ∅) ∧ (∏ i in s, i = ∏ i in t, i) :=
sorry

end no_partition_equal_product_l770_770697


namespace part1_part2_l770_770260

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
axiom a1 : a 1 = 1
axiom cond : ∀ n : ℕ, 3 * a n - 2 * S n = 2 * n - 1
axiom Sn_sum : ∀ n : ℕ, S n = ∑ i in Finset.range n, a (i + 1)

-- Part (1): Prove the sequence {aₙ + 1} is geometric
theorem part1 (n : ℕ) : ∃ (r : ℝ), n ≥ 2 ∧ (a n + 1) = 3^(n-1) * 2 := 
sorry

-- Part (2): Sum of the first n terms of sequence {bₙ}
def b (n : ℕ) : ℝ := (a n + 1) / (a n * a (n + 1))
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)

theorem part2 (n : ℕ) : T n = (3^n - 1) / (2 * 3^n - 1) := 
sorry

end part1_part2_l770_770260


namespace Tony_temp_above_fever_threshold_l770_770344

def normal_temp : ℕ := 95
def illness_A : ℕ := 10
def illness_B : ℕ := 4
def illness_C : Int := -2
def fever_threshold : ℕ := 100

theorem Tony_temp_above_fever_threshold :
  let T := normal_temp + illness_A + illness_B + illness_C
  T = 107 ∧ (T - fever_threshold) = 7 := by
  -- conditions
  let t_0 := normal_temp
  let T_A := illness_A
  let T_B := illness_B
  let T_C := illness_C
  let F := fever_threshold
  -- calculations
  let T := t_0 + T_A + T_B + T_C
  show T = 107 ∧ (T - F) = 7
  sorry

end Tony_temp_above_fever_threshold_l770_770344


namespace p_p_values_l770_770723

def p (x y : ℤ) : ℤ :=
if 0 ≤ x ∧ 0 ≤ y then x + 2*y
else if x < 0 ∧ y < 0 then x - 3*y
else 4*x + y

theorem p_p_values : p (p 2 (-2)) (p (-3) (-1)) = 6 :=
by
  sorry

end p_p_values_l770_770723


namespace hyperbola_eccentricity_l770_770619

theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0) (H : b ≠ 0) 
  (E : ∀ x y, (x, y) = (1, -1) → (x^2 / a^2 - y^2 / b^2 = 1)) :
  (sqrt (1 + b^2 / a^2) = sqrt 2) :=
by
  let asymptote := ∀ x, (x, -1) = (1, -1) → y = -b / a * x
  have : b / a = 1 := sorry
  show sqrt (1 + b^2 / a^2) = sqrt 2, from sorry
sorrry

end hyperbola_eccentricity_l770_770619


namespace f_2023_l770_770274

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_all : ∀ x : ℕ, f x ≠ 0 → (x ≥ 0)
axiom f_one : f 1 = 1
axiom f_functional_eq : ∀ a b : ℕ, f (a + b) = f a + f b - 3 * f (a * b)

theorem f_2023 : f 2023 = -(2^2022 - 1) := sorry

end f_2023_l770_770274


namespace distance_between_park_and_store_l770_770303

theorem distance_between_park_and_store (d_park_house d_store_house d_total : ℝ) (h1 : d_park_house = 5) (h2 : d_store_house = 8) (h3 : d_total = 16) :
  ∃ (d_park_store : ℝ), d_park_store = 3 ∧ d_park_house + d_park_store + d_store_house = d_total :=
by {
  use 3,
  split,
  { exact rfl },
  { rw [h1, h2, h3],
    norm_num }
}

end distance_between_park_and_store_l770_770303


namespace find_integer_pairs_l770_770117

theorem find_integer_pairs :
  ∃ (n : ℤ) (a : ℤ) (b : ℤ),
    (∀ a b : ℤ, (∃ m : ℤ, a^2 - 4*b = m^2) ∧ (∃ k : ℤ, b^2 - 4*a = k^2) ↔ 
    (a = 0 ∧ ∃ n : ℤ, b = n^2) ∨
    (b = 0 ∧ ∃ n : ℤ, a = n^2) ∨
    (b > 0 ∧ ∃ a : ℤ, a^2 > 0 ∧ b = -1 - a) ∨
    (a > 0 ∧ ∃ b : ℤ, b^2 > 0 ∧ a = -1 - b) ∨
    (a = 4 ∧ b = 4) ∨
    (a = 5 ∧ b = 6) ∨
    (a = 6 ∧ b = 5)) :=
sorry

end find_integer_pairs_l770_770117


namespace newspaper_spending_over_8_weeks_l770_770631

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l770_770631


namespace product_of_distances_l770_770149

noncomputable def parametric_equation_of_line : (ℝ → ℝ × ℝ) :=
  λ t, (-1 - (1 / 2) * t, 2 + (real.sqrt 3 / 2) * t)

theorem product_of_distances (x y t : ℝ) :
  let P := (-1, 2),
      direction_vector := (-1, real.sqrt 3),
      line := parametric_equation_of_line,
      circle := λ (θ : ℝ), 2 * real.cos (θ + (real.pi / 3)),
      M := line t,
      N := line (-t)
  in (  M.1^2 + M.2^2 = circle t ) →
     (  N.1^2 + N.2^2 = circle (-t) ) →
     ((M.1 +1)^2 + (M.2 - 2)^2)^0.5 * ((N.1 +1)^2 + (N.2 - 2)^2)^0.5 = 6 - 2 * real.sqrt 3 :=
by
  sorry

end product_of_distances_l770_770149


namespace one_person_both_days_selection_l770_770131

def volunteers : ℕ := 5

def ways_to_serve_both_days : ℕ :=
  (nat.choose volunteers 1) * ((nat.choose (volunteers - 1) 1) * (nat.choose (volunteers - 2) 1))

theorem one_person_both_days_selection :
  ways_to_serve_both_days = 60 :=
by
  sorry

end one_person_both_days_selection_l770_770131


namespace parabola_equation_hyperbola_equation_l770_770176

-- Definitions based on the conditions
def is_parabola (p : ℝ → ℝ → Prop) := 
  ∃ c : ℝ, ∀ x y : ℝ, p x y ↔ y^2 = 4 * c * x

def is_hyperbola (h : ℝ → ℝ → Prop) := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, h x y ↔ (x^2/a^2 - y^2/b^2 = 1)

-- Problem statements
theorem parabola_equation (p : ℝ → ℝ → Prop) :
  is_parabola p → (p (3/2) (sqrt 6) → p x y → y^2 = 4 * x) :=
sorry

theorem hyperbola_equation (h : ℝ → ℝ → Prop) :
  is_hyperbola h → h (3/2) (sqrt 6) → h x y → 4 * x^2 - (4 * y^2 / 3) = 1 :=
sorry

end parabola_equation_hyperbola_equation_l770_770176


namespace sequence_general_term_sequence_sum_Tn_l770_770987

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = 1 - a n) :
  a n = 1 / 2^n :=
sorry

theorem sequence_sum_Tn (a b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, S n = 1 - a n)
  (h2 : ∀ n, b n = (List.range n).map (λ i, log 4 (a i)).sum) :
  T n = 2^(n+1) + 4 / (n + 1) - 6 :=
sorry

end sequence_general_term_sequence_sum_Tn_l770_770987


namespace total_equivalent_percent_increase_l770_770213

noncomputable def increase_by_percent (base : ℝ) (percent : ℝ) : ℝ :=
  base * (1 + percent / 100)

noncomputable def total_price_after_increases (original_price : ℝ) : ℝ :=
  let price1 := increase_by_percent original_price 37.5
  let price2 := increase_by_percent price1 31
  let price3 := increase_by_percent price2 42.7
  let price4 := increase_by_percent price3 52.3
  let price5 := increase_by_percent price4 27.2
  price5

noncomputable def total_percent_increase (original_price new_price : ℝ) : ℝ :=
  ((new_price - original_price) / original_price) * 100

theorem total_equivalent_percent_increase :
  total_percent_increase 100 (total_price_after_increases 100) ≈ 397.99 :=
by
  sorry

end total_equivalent_percent_increase_l770_770213


namespace asymptotes_of_hyperbola_l770_770774

theorem asymptotes_of_hyperbola :
  (∀ x y : ℝ, (x^2 / 16 - y^2 / 25 = 1) →
    (y = (5 / 4) * x ∨ y = -(5 / 4) * x)) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l770_770774


namespace constant_t_exists_l770_770807

theorem constant_t_exists (c : ℝ) :
  ∃ t : ℝ, (∀ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A.2 = A.1 * c + c) ∧ (B.2 = B.1 * c + c) → (t = -2)) :=
sorry

end constant_t_exists_l770_770807


namespace desired_alcohol_percentage_is_18_l770_770762

noncomputable def final_alcohol_percentage (volume_x volume_y : ℕ) (percentage_x percentage_y : ℚ) : ℚ :=
  let total_volume := (volume_x + volume_y)
  let total_alcohol := (percentage_x * volume_x + percentage_y * volume_y)
  total_alcohol / total_volume * 100

theorem desired_alcohol_percentage_is_18 : 
  final_alcohol_percentage 300 200 0.10 0.30 = 18 := 
  sorry

end desired_alcohol_percentage_is_18_l770_770762


namespace roots_satisfy_conditions_l770_770101

variable (a x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * x2 + x1 + x2 - a = 0
def condition2 : Prop := x1 * x2 - a * (x1 + x2) + 1 = 0

-- Derived quadratic equation
def quadratic_eq : Prop := ∃ x : ℝ, x^2 - x + (a - 1) = 0

theorem roots_satisfy_conditions (h1: condition1 a x1 x2) (h2: condition2 a x1 x2) : quadratic_eq a :=
  sorry

end roots_satisfy_conditions_l770_770101


namespace qualified_flour_l770_770905

def is_qualified_flour (weight : ℝ) : Prop :=
  weight ≥ 24.75 ∧ weight ≤ 25.25

theorem qualified_flour :
  is_qualified_flour 24.80 ∧
  ¬is_qualified_flour 24.70 ∧
  ¬is_qualified_flour 25.30 ∧
  ¬is_qualified_flour 25.51 :=
by
  sorry

end qualified_flour_l770_770905


namespace cut_rectangle_from_square_l770_770369

theorem cut_rectangle_from_square
  (side : ℝ) (length : ℝ) (width : ℝ) 
  (square_area_eq : side * side = 100)
  (rectangle_area_eq : length * width = 90)
  (ratio_length_width : 5 * width = 3 * length) : 
  ¬ (length ≤ side ∧ width ≤ side) :=
by 
  sorry

end cut_rectangle_from_square_l770_770369


namespace sum_of_cubes_mod_5_l770_770466

theorem sum_of_cubes_mod_5 :
  (∑ i in Finset.range 51, i^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_5_l770_770466


namespace arithmetic_sequence_example_l770_770233

theorem arithmetic_sequence_example (a : ℕ → ℝ) (h : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) (h₁ : a 1 + a 19 = 10) : a 10 = 5 :=
by
  sorry

end arithmetic_sequence_example_l770_770233


namespace hyperbola_eccentricity_is_sqrt2_l770_770170

open Real

noncomputable def hyperbola_eccentricity (a b c : ℝ) : ℝ :=
  if (h : a = b) then
    let c := sqrt (a^2 + b^2) in
    c / a
  else 0 -- this should never be used if a = b is a condition

theorem hyperbola_eccentricity_is_sqrt2 (a b : ℝ) (h_eq : (∀ x y : ℝ, (y = x → (y = b/a*x))) ) :
  hyperbola_eccentricity a b (sqrt (a ^ 2 + b ^ 2)) = sqrt 2 :=
by
  sorry -- proof goes here

end hyperbola_eccentricity_is_sqrt2_l770_770170


namespace valentines_initial_l770_770735

theorem valentines_initial (gave_away : ℕ) (left_over : ℕ) (initial : ℕ) : 
  gave_away = 8 → left_over = 22 → initial = gave_away + left_over → initial = 30 :=
by
  intros h1 h2 h3
  sorry

end valentines_initial_l770_770735


namespace two_subsets_union_intersection_l770_770097

def T : Set ℕ := {0, 1, 2, 3, 4, 5} -- Represent the set {a, b, c, d, e, f} by {0, 1, 2, 3, 4, 5}

def choose_subsets (T : Set ℕ) : ℕ :=
  let choices := ((T.card).choose 2 * 2^((T.card) - 2)) / 2
  choices

theorem two_subsets_union_intersection (T : Set ℕ) (hT : T = {0, 1, 2, 3, 4, 5}) :
  choose_subsets T = 120 :=
by
  rw [hT]
  simp [choose_subsets]
  sorry

end two_subsets_union_intersection_l770_770097


namespace tony_needs_19_gallons_l770_770007

-- Define the parameters given in the problem
def num_columns : ℕ := 20
def height : ℝ := 15
def diameter : ℝ := 8
def coverage_per_gallon : ℝ := 400

-- Define the radius since it's needed for the lateral surface area calculation
def radius : ℝ := diameter / 2

-- Define the lateral surface area (LSA) of one column
def lsa_one_column : ℝ := 2 * Real.pi * radius * height

-- Define the total lateral surface area (LSA) of all columns
def total_lsa : ℝ := lsa_one_column * num_columns

-- Define the number of gallons needed, rounding up to the nearest full gallon
def gallons_needed : ℝ := Real.ceil (total_lsa / coverage_per_gallon)

-- The theorem we aim to prove
theorem tony_needs_19_gallons :
  gallons_needed = 19 :=
by
  sorry

end tony_needs_19_gallons_l770_770007


namespace base_7_to_base_10_l770_770832

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770832


namespace range_of_x_l770_770137

variable (x : ℝ)

def min {a b : ℝ} : ℝ :=
if a < b then a else b

theorem range_of_x 
  (h : min ( (3*x - 1) / 2 ) ( (x + 1) / 3 ) = ( (x + 1) / 3 )) : 
  x ≥ 5 / 7 :=
sorry

end range_of_x_l770_770137


namespace ratio_is_one_half_l770_770742

-- Assumptions/Conditions
variable (F : ℕ) (total_on_duty : ℕ)
variable (perc_female_on_duty : ℝ) (total_female_officers : ℕ)
variable (num_female_on_duty : ℕ)

-- Given conditions:
-- 1. 40% of the female officers on a police force were on duty
def forty_percent_female_on_duty : Prop := perc_female_on_duty = 0.40

-- 2. 240 officers were on duty that night
def total_officers_on_duty : Prop := total_on_duty = 240

-- 3. There are 300 female officers on the police force
def total_female_officers_condition : Prop := total_female_officers = 300

-- The number of female officers on duty that night
def num_female_calc : Prop := num_female_on_duty = Nat.floor (perc_female_on_duty * total_female_officers)

-- The ratio of female officers to the total officers on duty that night
def ratio_of_female_to_total : ℝ := num_female_on_duty / total_on_duty

-- The statement to prove:
theorem ratio_is_one_half (h1 : forty_percent_female_on_duty) (h2 : total_officers_on_duty) (h3 : total_female_officers_condition) (h4 : num_female_calc) :
  ratio_of_female_to_total = 1 / 2 := sorry

end ratio_is_one_half_l770_770742


namespace similar_triangle_perimeter_l770_770348

theorem similar_triangle_perimeter 
  (a b c : ℝ) (a_sim : ℝ)
  (h1 : a = b) (h2 : b = c)
  (h3 : a = 15) (h4 : a_sim = 45)
  (h5 : a_sim / a = 3) :
  a_sim + a_sim + a_sim = 135 :=
by
  sorry

end similar_triangle_perimeter_l770_770348


namespace cost_of_seven_CDs_l770_770008

theorem cost_of_seven_CDs (cost_per_two : ℝ) (h1 : cost_per_two = 32) : (7 * (cost_per_two / 2)) = 112 :=
by
  sorry

end cost_of_seven_CDs_l770_770008


namespace binomial_sum_eq_l770_770091

theorem binomial_sum_eq (n : ℕ) (h : 3 ≤ n) :
  (Finset.sum (Finset.range 4) (λ k, Nat.choose n k)) =
  1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6 := 
by
  sorry

end binomial_sum_eq_l770_770091


namespace correct_option_B_l770_770856

-- Define the set where {x | (x-1)(x+2) = 0}
def special_set : Set ℤ := { x | (x-1) * (x+2) = 0 }

-- Prove that 1 is in the special_set
theorem correct_option_B : 1 ∈ special_set :=
by
  -- Note that 1 satisfies the equation (x-1)(x+2)=0
  unfold special_set
  simp
  -- This proof step shows that 1-1 is 0, making the equation product zero
  exact eq.refl 0


end correct_option_B_l770_770856


namespace quadratic_distinct_roots_l770_770545

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770545


namespace product_of_two_numbers_l770_770374

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 400) : x * y = 88 :=
by
  -- Proof goes here
  sorry

end product_of_two_numbers_l770_770374


namespace sum_of_cubes_mod_5_l770_770478

theorem sum_of_cubes_mod_5 :
  ( ∑ k in Finset.range 50, (k + 1)^3 ) % 5 = 0 :=
sorry

end sum_of_cubes_mod_5_l770_770478


namespace compare_abc_l770_770567

variable (a b c : ℝ)

noncomputable def define_a : ℝ := (2/3)^(1/3)
noncomputable def define_b : ℝ := (2/3)^(1/2)
noncomputable def define_c : ℝ := (3/5)^(1/2)

theorem compare_abc (h₁ : a = define_a) (h₂ : b = define_b) (h₃ : c = define_c) :
  a > b ∧ b > c := by
  sorry

end compare_abc_l770_770567


namespace trains_cross_time_l770_770193

-- Define the given lengths and speeds in kilometers per hour (kmph)
def Length_1 : ℕ := 500
def Speed_1 : ℝ := 120
def Length_2 : ℕ := 800
def Speed_2 : ℝ := 80

-- Conversion from kmph to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Calculate the relative speed in m/s when both trains are moving in the same direction
def relative_speed (speed1_kmph speed2_kmph : ℝ) : ℝ := kmph_to_mps speed1_kmph - kmph_to_mps speed2_kmph

-- Calculate the total length to be crossed in meters
def total_length (length1 length2 : ℕ) : ℕ := length1 + length2

-- Calculate the time taken to cross in seconds
def time_to_cross (length1 length2 : ℕ) (speed1_kmph speed2_kmph : ℝ) : ℝ := 
  (↑(total_length length1 length2) : ℝ) / relative_speed speed1_kmph speed2_kmph

-- Lean theorem statement
theorem trains_cross_time : 
  time_to_cross Length_1 Length_2 Speed_1 Speed_2 ≈ 117 :=
  sorry

end trains_cross_time_l770_770193


namespace disproving_statement_exists_l770_770015

theorem disproving_statement_exists (a : ℤ) : ∃ a < 3, a^2 ≥ 9 := by
  use -4
  have h1 : -4 < 3 := by norm_num
  have h2 : (-4)^2 = 16 := by norm_num
  have h3 : 16 ≥ 9 := by norm_num
  exact ⟨h1, by rw [←h2, h3]⟩

end disproving_statement_exists_l770_770015


namespace solve_y_l770_770852

theorem solve_y (x y z : ℝ) (h1 : x = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) :
  y = -1 / 4 :=
by
  subst h1
  subst h2
  rw [pow_two x, pow_two z] at h3
  linarith

end solve_y_l770_770852


namespace quadratic_roots_condition_l770_770506

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770506


namespace distance_blown_westward_l770_770737

theorem distance_blown_westward
  (time_traveled_east : ℕ)
  (speed : ℕ)
  (travelled_halfway : Prop)
  (new_location_fraction : ℚ) :
  time_traveled_east = 20 →
  speed = 30 →
  travelled_halfway →
  new_location_fraction = 1 / 3 →
  let distance_traveled_east := speed * time_traveled_east,
      total_distance := 2 * distance_traveled_east,
      new_location_distance := new_location_fraction * total_distance in
  distance_traveled_east - new_location_distance = 200 :=
begin
  intros,
  sorry
end

end distance_blown_westward_l770_770737


namespace solve_for_y_l770_770678

noncomputable def find_angle_y : Prop :=
  let AB_CD_are_straight_lines : Prop := True
  let angle_AXB : ℕ := 70
  let angle_BXD : ℕ := 40
  let angle_CYX : ℕ := 100
  let angle_YXZ := 180 - angle_AXB - angle_BXD
  let angle_XYZ := 180 - angle_CYX
  let y := 180 - angle_YXZ - angle_XYZ
  y = 30

theorem solve_for_y : find_angle_y :=
by
  trivial

end solve_for_y_l770_770678


namespace trigonometric_inequalities_l770_770721

theorem trigonometric_inequalities :
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  in b < a ∧ a < c :=
by
  sorry

end trigonometric_inequalities_l770_770721


namespace number_of_croutons_l770_770686

def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def crouton_calories : ℕ := 20
def total_salad_calories : ℕ := 350

theorem number_of_croutons : 
  ∃ n : ℕ, n * crouton_calories = total_salad_calories - (lettuce_calories + cucumber_calories) ∧ n = 12 :=
by
  sorry

end number_of_croutons_l770_770686


namespace positive_difference_of_complementary_angles_l770_770780

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l770_770780


namespace calculate_length_of_floor_l770_770318

-- Define the conditions and the objective to prove
variable (breadth length : ℝ)
variable (cost rate : ℝ)
variable (area : ℝ)

-- Given conditions
def length_more_by_percentage : Prop := length = 2 * breadth
def painting_cost : Prop := cost = 529 ∧ rate = 3

-- Objective
def length_of_floor : ℝ := 2 * breadth

theorem calculate_length_of_floor : 
  (length_more_by_percentage breadth length) →
  (painting_cost cost rate) →
  length_of_floor breadth = 18.78 :=
by
  sorry

end calculate_length_of_floor_l770_770318


namespace average_speed_calculation_l770_770879

noncomputable def average_speed (d1 d2 v1 v2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let time1 := d1 / v1
  let time2 := d2 / v2
  let total_time := time1 + time2
  total_distance / total_time

theorem average_speed_calculation :
  average_speed 7 10 7 7 ≈ 7.98 :=
by
  sorry

end average_speed_calculation_l770_770879


namespace find_a_range_l770_770726

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) + f x = x ^ 2) ∧ 
  (∀ x : ℝ, x > 0 → f'' x > x) ∧
  (∀ a : ℝ, f (2 - a) - f a ≥ 2 - 2 * a)

theorem find_a_range (f : ℝ → ℝ) (h : satisfies_conditions f) : ∀ a : ℝ, (f (2 - a) - f a ≥ 2 - 2 * a) → a ≤ 1 :=
by
  intros a ha
  sorry

end find_a_range_l770_770726


namespace train_passes_bus_in_approximately_5point28_seconds_l770_770901

theorem train_passes_bus_in_approximately_5point28_seconds :
  (let length_train := 220 -- in meters
       speed_train_kmh := 90 -- in km/hr
       speed_bus_kmh := 60 -- in km/hr
       speed_train_ms := speed_train_kmh * 1000 / 3600 -- convert to m/s
       speed_bus_ms := speed_bus_kmh * 1000 / 3600 -- convert to m/s
       relative_speed := speed_train_ms + speed_bus_ms -- relative speed in m/s
       time := length_train / relative_speed -- time in seconds
   in abs (time - 5.28) < 0.01) := 
sorry

end train_passes_bus_in_approximately_5point28_seconds_l770_770901


namespace standard_equation_of_ellipse_and_fixed_point_l770_770582

open Real

-- Definitions from the conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def passes_through (x y : ℝ) (px py : ℝ) : Prop := (px = 1) ∧ (py = 3/2)

-- Assuming the following properties about the ellipse and the point it passes through
variables {a b c : ℝ} (x y : ℝ)
axiom ellipse_properties (e: ℝ) : e = 1/2 ∧ ellipse a b x y
axiom passing_point : passes_through x y 1 (3/2)

-- The statement of the mathematic proof problem
theorem standard_equation_of_ellipse_and_fixed_point :
  (∃ (a b : ℝ), a^2 = 4 ∧ b^2 = 3 ∧ ellipse a b 1 (3/2)) ∧
  (∀ (A B : ℝ × ℝ), (let k_PA := (A.2 - sqrt 3) / A.1,
                         k_PB := (B.2 - sqrt 3) / B.1 in
                     k_PA * k_PB = 1/4) →
                     ∃ (m : ℝ), (m = 2 * sqrt 3) → 
                               passes_through A.1 A.2 0 m → 
                               passes_through B.1 B.2 0 m) :=
by
  sorry

end standard_equation_of_ellipse_and_fixed_point_l770_770582


namespace berries_per_bird_per_day_l770_770391

theorem berries_per_bird_per_day (birds : ℕ) (total_berries : ℕ) (days : ℕ) (berries_per_bird_per_day : ℕ) 
  (h_birds : birds = 5)
  (h_total_berries : total_berries = 140)
  (h_days : days = 4) :
  berries_per_bird_per_day = 7 :=
  sorry

end berries_per_bird_per_day_l770_770391


namespace find_e_l770_770268

variable (b e : ℝ)

def f (x : ℝ) : ℝ := 3 * x + b
def g (x : ℝ) : ℝ := b * x + 5

theorem find_e (h : ∀ x, f (g x) = 15 * x + e) : e = 15 :=
sorry

end find_e_l770_770268


namespace number_of_twos_written_l770_770352

-- Define the repeated number and its corresponding cycle properties
def repeated_number := "20192020"
def length_repeated_number := 8
def count_twos (s : String) : Nat := s.toList.count (λ c => c = '2')

-- Conditions
def total_digits_written := 2020
def number_of_twos_in_repeated_number := count_twos repeated_number

-- Prove statement
theorem number_of_twos_written : 
  let cycles := total_digits_written / length_repeated_number
  let remainder := total_digits_written % length_repeated_number
  let total_twos := cycles * number_of_twos_in_repeated_number + count_twos (repeated_number.take remainder)
  total_twos = 757 := 
by
  sorry

end number_of_twos_written_l770_770352


namespace quadratic_has_distinct_real_roots_l770_770523

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770523


namespace quarters_dimes_equivalence_l770_770302

theorem quarters_dimes_equivalence (m : ℕ) (h : 25 * 30 + 10 * 20 = 25 * 15 + 10 * m) : m = 58 :=
by
  sorry

end quarters_dimes_equivalence_l770_770302


namespace unique_infinite_sequence_l770_770584

-- Defining conditions for the infinite sequence of negative integers
variable (a : ℕ → ℤ)
  
-- Condition 1: Elements in sequence are negative integers
def sequence_negative : Prop :=
  ∀ n, a n < 0 

-- Condition 2: For every positive integer n, the first n elements taken modulo n have n distinct remainders
def distinct_mod_remainders (n : ℕ) : Prop :=
  ∀ i j, i < n → j < n → i ≠ j → (a i % n ≠ a j % n) 

-- The main theorem statement
theorem unique_infinite_sequence (a : ℕ → ℤ) 
  (h1 : sequence_negative a) 
  (h2 : ∀ n, distinct_mod_remainders a n) :
  ∀ k : ℤ, ∃! n, a n = k :=
sorry

end unique_infinite_sequence_l770_770584


namespace sum_of_cubes_mod_5_l770_770468

theorem sum_of_cubes_mod_5 :
  (∑ i in Finset.range 51, i^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_5_l770_770468


namespace parabola_properties_l770_770125

-- Definition of the parabola and the tangent line condition
def parabola (p : ℝ) : set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.snd^2 = 2 * p * xy.fst}
def tangent_line : set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.fst / -75 + xy.snd / 15 = 1}

-- Definition of the focus and directrix of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def directrix (p : ℝ) : set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.fst = -p / 2}

-- The main theorem statement as per the problem requirements
theorem parabola_properties:
  ∃ p : ℝ, p = 6 ∧ focus p = (3, 0) ∧ ∀ xy : ℝ × ℝ, xy ∈ directrix p ↔ xy.fst = -3
:=
begin
  -- Proof goes here
  sorry
end

end parabola_properties_l770_770125


namespace roots_inside_unit_circle_l770_770146

theorem roots_inside_unit_circle {n : ℕ} 
  (a : Fin (n + 1) → ℝ) 
  (h : ∀ i j : Fin (n + 1), j < i → a i > a j ∧ a j > 0) : 
  ∀ C : ℂ, (polynomial.sum (Fin n.succ) (λ i, a i • X ^ i)).eval C = 0 → abs C < 1 :=
sorry

end roots_inside_unit_circle_l770_770146


namespace space_taken_by_files_l770_770733

-- Definitions/Conditions
def total_space : ℕ := 28
def space_left : ℕ := 2

-- Statement of the theorem
theorem space_taken_by_files : total_space - space_left = 26 := by sorry

end space_taken_by_files_l770_770733


namespace coin_placement_cases_l770_770379

def possiblePlacement (n : ℕ) : Bool :=
  4 * n == 12

theorem coin_placement_cases :
  possiblePlacement 2 = false ∧
  possiblePlacement 3 = true ∧
  possiblePlacement 4 = false ∧
  possiblePlacement 5 = false ∧
  possiblePlacement 6 = false ∧
  possiblePlacement 7 = false :=
by
  unfold possiblePlacement
  repeat {split}
  all_goals { simp [Nat.mul_eq_zero] }
  sorry

end coin_placement_cases_l770_770379


namespace problem1_problem2_l770_770044

-- Problem 1
theorem problem1 (α : ℝ) (h : Real.tan α = -3/4) :
  (Real.cos ((π / 2) + α) * Real.sin (-π - α)) /
  (Real.cos ((11 * π) / 2 - α) * Real.sin ((9 * π) / 2 + α)) = -3 / 4 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : 0 ≤ α ∧ α ≤ π) (h2 : Real.sin α + Real.cos α = 1 / 5) :
  Real.cos (2 * α - π / 4) = -31 * Real.sqrt 2 / 50 :=
by sorry

end problem1_problem2_l770_770044


namespace range_of_omega_l770_770613

noncomputable theory

def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x) - sqrt 3 * cos (ω * x)

theorem range_of_omega (ω : ℝ) :
  (0 < ω) →
  (∀ x, (0 < x ∧ x < π) → f ω x = -1) → ∃ n ∈ {3}, (n = 3 ↔ (13/6 < ω ∧ ω ≤ 7/2)) :=
begin
  sorry
end

end range_of_omega_l770_770613


namespace survival_probability_l770_770328

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem survival_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) :
  (binomial_coefficient 5 4) * p^4 * (1 - p) = C_{5}^{4} p^4 (1 - p) :=
begin
  sorry
end

end survival_probability_l770_770328


namespace highway_length_l770_770345

-- Define the conditions
def car1_speed := 25 -- Car 1 speed in mph
def car2_speed := 45 -- Car 2 speed in mph
def meeting_time := 2.5 -- meeting time in hours

-- Calculate the combined speed
def combined_speed : ℝ := car1_speed + car2_speed

-- Define the expected highway length
def expected_highway_length := 175 -- in miles

-- Translate the problem to a proof goal in Lean
theorem highway_length : combined_speed * meeting_time = expected_highway_length :=
by
  sorry

end highway_length_l770_770345


namespace find_y_eq_l770_770451

theorem find_y_eq (y : ℝ) : (10 - y)^2 = 4 * y^2 → (y = 10 / 3 ∨ y = -10) :=
by
  intro h
  -- The detailed proof will be provided here
  sorry

end find_y_eq_l770_770451


namespace unique_function_exists_l770_770956

noncomputable def number_of_valid_functions (f : ℝ → ℝ) : ℕ :=
if (∀ x y : ℝ, f(x + f(y) + 1) = x + y + 1) then 1 else 0

theorem unique_function_exists (f : ℝ → ℝ) :
(∀ x y : ℝ, f(x + f(y) + 1) = x + y + 1) → (∀ g : ℝ → ℝ, (∀ x y : ℝ, g(x + g(y) + 1) = x + y + 1) → g = f) → number_of_valid_functions f = 1 :=
begin
  intros h h_unique,
  rw number_of_valid_functions,
  split_ifs,
  { refl, },
  { exfalso,
    apply h h_unique,
    intros g h_g,
    apply h_unique _ h_g, }
end

end unique_function_exists_l770_770956


namespace sqrt_fourth_root_of_000001_is_0_to_nearest_tenth_l770_770927

noncomputable def nested_root_calculation : ℝ :=
  Float.round (Real.sqrt (Real.sqrt (10 ^ (-6 / 4, 80 / 4))))

theorem sqrt_fourth_root_of_000001_is_0_to_nearest_tenth :
  Float.roundWith nested_root_calculation = 0.0 :=
 by sorry

end sqrt_fourth_root_of_000001_is_0_to_nearest_tenth_l770_770927


namespace ninja_star_area_l770_770691

theorem ninja_star_area :
  let base_outer := 2
  let height_outer := 2
  let base_inner := 2
  let height_inner := 1
  let area_outer := 4 * (1/2 * base_outer * height_outer)
  let area_inner := 4 * (1/2 * base_inner * height_inner)
  area_outer + area_inner = 12 :=
by {
  let base_outer := 2
  let height_outer := 2
  let base_inner := 2
  let height_inner := 1
  let area_outer := 4 * (1/2 * base_outer * height_outer)
  let area_inner := 4 * (1/2 * base_inner * height_inner)
  show (area_outer + area_inner = 12), from sorry
}

end ninja_star_area_l770_770691


namespace sum_integers_neg45_to_60_l770_770362

theorem sum_integers_neg45_to_60 : 
  let sum_arithmetic_sequence := λ (a l n : Int), (a + l) * n / 2
  sum_arithmetic_sequence (-45) 60 (60 + 45 + 1) = 795 := 
by 
  have h₁ : (60 + 45 + 1) = 106 := by norm_num
  have h₂ : sum_arithmetic_sequence (-45) 60 106 = 
      ((-45) + 60) * 106 / 2 := rfl
  have h₃ : ((-45) + 60) * 106 / 2 = 795 := by norm_num
  exact h₃

end sum_integers_neg45_to_60_l770_770362


namespace probability_of_twice_l770_770973

def S : set ℕ := {1, 2, 3, 4, 5}

def valid_pairs (a b : ℕ) : Prop := 
  (a ∈ S ∧ b ∈ S ∧ a ≠ b) ∧ (a = 2 * b ∨ b = 2 * a)

def event_count : ℕ := 2
def total_outcomes : ℕ := 10

theorem probability_of_twice :
  (event_count : ℚ) / (total_outcomes : ℚ) = 1 / 5 :=
by
  sorry

end probability_of_twice_l770_770973


namespace mode_of_scores_l770_770795

-- Define the given data.
def scores : List ℕ :=
  [55, 55, 55, 64, 68, 72, 76, 76, 79, 81, 83, 83, 83, 89, 89, 90, 95, 95, 95, 97, 98, 102, 102, 102, 103, 103, 103, 104, 110, 110, 111]

-- Definition of mode
def mode (lst : List ℕ) : List ℕ :=
  let grouped := lst.groupBy id
  let max_freq := grouped.map List.length |>.maximum
  grouped.filter (λ l => List.length l == max_freq).map List.head

theorem mode_of_scores : mode scores = [83, 95, 102, 103] := 
  sorry

end mode_of_scores_l770_770795


namespace original_denominator_is_nine_l770_770420

theorem original_denominator_is_nine (d : ℕ) : 
  (2 + 5) / (d + 5) = 1 / 2 → d = 9 := 
by sorry

end original_denominator_is_nine_l770_770420


namespace percentage_of_candidates_failing_l770_770670

noncomputable def total_candidates := 2500
noncomputable def girls := 1100
noncomputable def boys := total_candidates - girls

noncomputable def pmb := 0.42 
noncomputable def pmg := 0.35 
noncomputable def psb := 0.39 
noncomputable def psg := 0.32 
noncomputable def plb := 0.36 
noncomputable def plg := 0.40 

noncomputable def boys_passing_mathematics := pmb * boys
noncomputable def girls_passing_mathematics := pmg * girls

noncomputable def boys_passing_science := psb * boys
noncomputable def girls_passing_science := psg * girls

noncomputable def boys_passing_language_arts := plb * boys
noncomputable def girls_passing_language_arts := plg * girls

noncomputable def boys_passing_all_subjects : ℝ := min (min boys_passing_mathematics boys_passing_science) boys_passing_language_arts
noncomputable def girls_passing_all_subjects : ℝ := min (min girls_passing_mathematics girls_passing_science) girls_passing_language_arts

noncomputable def total_passing_all_subjects := boys_passing_all_subjects + girls_passing_all_subjects
noncomputable def total_failing := total_candidates - total_passing_all_subjects

noncomputable def percentage_failing := (total_failing / total_candidates) * 100

theorem percentage_of_candidates_failing : percentage_failing = 65.76 := 
by
  sorry

end percentage_of_candidates_failing_l770_770670


namespace part1_sales_volume_and_profit_part2_relationship_part3_selling_price_l770_770051

-- Definitions of conditions
def sales_cost : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 500
def volume_decrease_per_dollar_increase : ℝ := 10
def profit_equation (x : ℝ) : ℝ := (x - sales_cost) * (initial_sales_volume - volume_decrease_per_dollar_increase * (x - initial_selling_price))

-- Part 1: Verifying sales volume and profit at $55 per kilogram
theorem part1_sales_volume_and_profit : 
  let selling_price := 55 in
  let sales_volume := initial_sales_volume - volume_decrease_per_dollar_increase * (selling_price - initial_selling_price) in
  let profit := (selling_price - sales_cost) * sales_volume in
  sales_volume = 450 ∧ profit = 6750 :=
by
  sorry

-- Part 2: Relationship between y (monthly sales profit) and x (selling price)
theorem part2_relationship (x : ℝ) :
  profit_equation x = -10 * x^2 + 1400 * x - 40000 :=
by
  sorry

-- Part 3: Selling price for a monthly sales profit of $8000
theorem part3_selling_price : 
  let target_profit := 8000 in
  let target_sales_cost := 10000 in
  let max_sales_volume := target_sales_cost / sales_cost in
  ∃ x : ℝ, profit_equation x = target_profit ∧ (initial_sales_volume - volume_decrease_per_dollar_increase * (x - initial_selling_price) ≤ max_sales_volume) :=
by
  sorry

end part1_sales_volume_and_profit_part2_relationship_part3_selling_price_l770_770051


namespace cube_vertices_adjacent_to_six_l770_770239

theorem cube_vertices_adjacent_to_six :
  ∃ (adj : Finset ℕ), adj ∈ { {2, 3, 5}, {3, 5, 7}, {2, 3, 7} } ∧ adj.card = 3 :=
by
  sorry

end cube_vertices_adjacent_to_six_l770_770239


namespace congruent_triangles_sides_equal_sixty_deg_angles_equilateral_alternate_interior_angles_angle_bisector_median_isosceles_problem_statement_l770_770857

-- Definitions for the conditions in the problem.
def congruent_triangles (Δ₁ Δ₂ : Type) : Prop :=
  -- Define the properties of congruent triangles here.
  sorry

def equilateral_triangle (Δ : Type) : Prop :=
  -- Define the properties of an equilateral triangle here.
  sorry

def parallel_lines (l1 l2 : Type) : Prop :=
  -- Define the properties of parallel lines here.
  sorry

def isosceles_triangle (Δ : Type) : Prop :=
  -- Define the properties of an isosceles triangle here.
  sorry

-- Propositions to be proved
theorem congruent_triangles_sides_equal (Δ₁ Δ₂ : Type) :
  congruent_triangles Δ₁ Δ₂ → 
  sorry := 
  -- Proof of proposition ①
  sorry

theorem sixty_deg_angles_equilateral (Δ : Type) :
  (∃ A B C : ℝ, A = 60 ∧ B = 60 ∧ A + B + C = 180) → 
  equilateral_triangle Δ :=
  -- Proof of proposition ②
  sorry

theorem alternate_interior_angles (l1 l2 : Type) (t : Type) :
  ¬parallel_lines l1 l2 → 
  sorry := 
  -- Proof of proposition ③
  sorry

theorem angle_bisector_median_isosceles (Δ : Type) :
  isosceles_triangle Δ → 
  ¬(sorry) := 
  -- Proof of proposition ④
  sorry

-- Lean statement for the final combined proof
theorem problem_statement :
  (congruent_triangles_sides_equal Δ₁ Δ₂) ∧
  (sixty_deg_angles_equilateral Δ) ∧
  (alternate_interior_angles l1 l2 t) ∧
  (angle_bisector_median_isosceles Δ) := 
by 
  constructor;
  try {assumption};
  sorry

end congruent_triangles_sides_equal_sixty_deg_angles_equilateral_alternate_interior_angles_angle_bisector_median_isosceles_problem_statement_l770_770857


namespace remaining_part_of_third_candle_l770_770002

theorem remaining_part_of_third_candle
  (t : ℝ)
  (h1 : t > 0)
  (h2 : 0 < (2/5 : ℝ) ∧ (2/5 : ℝ) < 1)
  (h3 : 0 < (3/7 : ℝ) ∧ (3/7 : ℝ) < 1) :
  let second_burned := 1 - (2/5 : ℝ)
  let third_burned := 1 - (3/7 : ℝ)
  let rate_second := second_burned / t
  let rate_third := third_burned / t
  let time_second_full := (2/5 : ℝ) / rate_second
  let third_burned_during_remaining := time_second_full * rate_third in
  (3/7 : ℝ) - third_burned_during_remaining = (1/21 : ℝ) :=
by
  sorry

end remaining_part_of_third_candle_l770_770002


namespace ratio_of_areas_two_adjacent_triangles_to_one_triangle_l770_770060

-- Definition of a regular hexagon divided into six equal triangles
def is_regular_hexagon_divided_into_six_equal_triangles (s : ℝ) : Prop :=
  s > 0 -- s is the area of one of the six triangles and must be positive

-- Definition of the area of a region formed by two adjacent triangles
def area_of_two_adjacent_triangles (s r : ℝ) : Prop :=
  r = 2 * s

-- The proof problem statement
theorem ratio_of_areas_two_adjacent_triangles_to_one_triangle (s r : ℝ)
  (hs : is_regular_hexagon_divided_into_six_equal_triangles s)
  (hr : area_of_two_adjacent_triangles s r) : 
  r / s = 2 :=
by
  sorry

end ratio_of_areas_two_adjacent_triangles_to_one_triangle_l770_770060


namespace quadratic_distinct_roots_l770_770546

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770546


namespace value_of_c_distinct_real_roots_l770_770560

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770560


namespace max_value_of_z_l770_770992

theorem max_value_of_z {x y : ℝ} (h : (x - 3)^2 + (y - 4)^2 = 9) :
  ∃ z_max, ∀ z, z = 3 * x + 4 * y → z ≤ z_max :=
begin
  use 40,
  intro z,
  intros hz,
  sorry -- proof will be provided here.
end

end max_value_of_z_l770_770992


namespace quadratic_roots_condition_l770_770508

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770508


namespace remainder_is_correct_l770_770960

theorem remainder_is_correct :
  ∃ q : Polynomial ℤ, 
    let p := (Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 1 * Polynomial.X^5 - Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 1 * Polynomial.X^2)
    let d := (Polynomial.X - Polynomial.C 1) * (Polynomial.X + Polynomial.C 1) * (Polynomial.X - Polynomial.C 2)
    let r := (Polynomial.C (26/3 : ℚ) * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X - Polynomial.C (26/3 : ℚ))
    p = d * q + r := 
begin
  sorry
end

end remainder_is_correct_l770_770960


namespace area_ratio_of_triangle_AOC_to_ABC_l770_770588

/-- Let A, B, and C be three points in a plane forming triangle ABC, and let O be a point inside it.
    Given that vector OA + vector OC + 2 * vector OB = 0, 
    this lemma states that the ratio of the area of triangle AOC to the area of triangle ABC is 1:2. -/
theorem area_ratio_of_triangle_AOC_to_ABC (A B C O : Point) 
    (h1 : vector OA + vector OC + 2 * vector OB = 0) : 
    area_ratio (triangle A O C) (triangle A B C) = 1 / 2 :=
sorry

end area_ratio_of_triangle_AOC_to_ABC_l770_770588


namespace triangles_congruent_part1_triangles_congruent_part2_l770_770565

-- Definitions of points and lengths
structure Point where
  x : ℝ
  y : ℝ

-- Function to compute length between two points
def length (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2)

-- Structure of Triangle
structure Triangle where
  A B C : Point

-- Hypotheses for the given problem
variables {A B C A1 B1 : Point}
variable (ABC : Triangle)
variable (isosceles : length A C = length B C)
variable (equalSegments : length A C = length B C)
variable (CA1_eq_CB1 : length A A1 = length B B1)

-- Define the goal in terms of triangle congruence
theorem triangles_congruent_part1 :
  length A C = length B C → 
  length A1 C = length B1 C →
  ∠(A C B) = ∠(B C A) →
  Triangle A C B1 ≅ Triangle B C A1 := sorry

theorem triangles_congruent_part2 :
  length A B = length B A → 
  length A1 B = length B1 A →
  ∠(A B C) = ∠(B A C) →
  Triangle A B B1 ≅ Triangle B A A1 := sorry

end triangles_congruent_part1_triangles_congruent_part2_l770_770565


namespace max_cubes_in_box_l770_770377

theorem max_cubes_in_box :
  let L := 8
  let W := 9
  let H := 12
  let V_box := L * W * H
  let V_cube := 27
  t = V_box / V_cube → t = 32 :=
by
  let L := 8
  let W := 9
  let H := 12
  let V_box := L * W * H
  let V_cube := 27
  t = V_box / V_cube
  have : V_box = 864 := sorry
  have : t = 864 / 27 := sorry
  sorry

end max_cubes_in_box_l770_770377


namespace stddev_newData_l770_770175

-- Definitions and conditions
def variance (data : List ℝ) : ℝ := sorry  -- Placeholder for variance definition
def stddev (data : List ℝ) : ℝ := sorry    -- Placeholder for standard deviation definition

-- Given data
def data : List ℝ := sorry                -- Placeholder for the data x_1, x_2, ..., x_8
def newData : List ℝ := data.map (λ x => 2 * x + 1)

-- Given condition
axiom variance_data : variance data = 16

-- Proof of the statement
theorem stddev_newData : stddev newData = 8 :=
by {
  sorry
}

end stddev_newData_l770_770175


namespace num_vals_of_q_l770_770504

-- Given question and condition translated into Lean
theorem num_vals_of_q (x : ℤ) (h : | || q - 5 | - 10 | - x | = 2) : ∃ q1 q2 q3 q4 : ℤ,
  q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ q3 ≠ q4 :=
begin
  sorry
end

end num_vals_of_q_l770_770504


namespace tom_sleep_deficit_l770_770006

-- Define the conditions as given
def weeknights := 5
def weekend_nights := 2
def ideal_sleep_hours := 8
def actual_weeknight_sleep := 5
def actual_weekend_sleep := 6

-- Define the proofs
theorem tom_sleep_deficit : 
  let ideal_week_sleep := (weeknights * ideal_sleep_hours) + (weekend_nights * ideal_sleep_hours) in
  let actual_week_sleep := (weeknights * actual_weeknight_sleep) + (weekend_nights * actual_weekend_sleep) in
  ideal_week_sleep - actual_week_sleep = 19 := 
by
  let ideal_week_sleep := (weeknights * ideal_sleep_hours) + (weekend_nights * ideal_sleep_hours)
  let actual_week_sleep := (weeknights * actual_weeknight_sleep) + (weekend_nights * actual_weekend_sleep)
  have h1 : ideal_week_sleep = 56 := by rfl
  have h2 : actual_week_sleep = 37 := by rfl
  show ideal_week_sleep - actual_week_sleep = 19 from by
    rw [h1, h2]
    apply nat.sub_self_eq_suc_natu

/-- Placeholder for the proof -/
sorry

end tom_sleep_deficit_l770_770006


namespace maximal_value_of_S_l770_770696

-- Defining the structure of the problem
def partition_arithmetic_progressions (n : ℕ) (S : ℕ) : Prop :=
  ∃ (A : list (list ℕ)), 
    (∀ p ∈ A, p.length ≥ 3) ∧
    (∀ p ∈ A, ∃ (d : ℕ), ∀ i j, i < j → j < p.length → p.nth j = (p.nth i) + (j - i) * d) ∧
    (∃ l : list ℕ, l.nodup ∧ l = list.range (3 * n) ∧ (∀ x ∈ l, ∃ p ∈ A, x ∈ p)) ∧
    S = A.map (λ p, (p.nth 1 - p.nth 0)).sum

-- Statement of the problem
theorem maximal_value_of_S (n : ℕ) (hpos : 0 < n) : ∃ S, partition_arithmetic_progressions n S ∧ S = n^2 :=
by
  sorry

end maximal_value_of_S_l770_770696


namespace hyperbola_eccentricity_is_sqrt_two_l770_770166

-- Define the hyperbola and the asymptote condition
def hyperbola_asymptote_condition (a b : ℝ) : Prop :=
  (b / a = 1)

-- Define the hyperbola property
def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

-- The main theorem to prove: the eccentricity is sqrt(2) given the conditions
theorem hyperbola_eccentricity_is_sqrt_two (a b : ℝ)
  (h : hyperbola_asymptote_condition a b) :
  hyperbola_eccentricity a b = real.sqrt 2 :=
by 
  sorry

end hyperbola_eccentricity_is_sqrt_two_l770_770166


namespace proof_longest_vs_shortest_distance_proof_total_distance_traveled_proof_savings_l770_770227
open Real

def daily_distances : list ℝ := [-8, -12, -16, 0, 22, 31, 33]

def gasoline_consumption_rate : ℝ := 5.5 -- liters per 100 km
def gasoline_price_per_liter : ℝ := 8.4 -- yuan per liter

def electric_consumption_rate : ℝ := 15 -- kWh per 100 km
def electricity_price_per_kWh : ℝ := 0.5 -- yuan per kWh

def longest_vs_shortest_distance : ℝ :=
  list.maximum daily_distances - list.minimum daily_distances

def total_distance_traveled : ℝ :=
  list.sum daily_distances

def gasoline_cost (distance : ℝ) : ℝ :=
  (distance / 100) * gasoline_consumption_rate * gasoline_price_per_liter

def electricity_cost (distance : ℝ) : ℝ :=
  (distance / 100) * electric_consumption_rate * electricity_price_per_kWh

def savings (distance : ℝ) : ℝ :=
  gasoline_cost distance - electricity_cost distance

theorem proof_longest_vs_shortest_distance:
  longest_vs_shortest_distance daily_distances = 49 :=
sorry

theorem proof_total_distance_traveled:
  total_distance_traveled daily_distances = 50 :=
sorry

theorem proof_savings:
  savings (total_distance_traveled daily_distances) = 154.8 :=
sorry

end proof_longest_vs_shortest_distance_proof_total_distance_traveled_proof_savings_l770_770227


namespace variance_of_21_points_is_0_20_l770_770650

variable (a : Fin 20 → ℝ)
variable (mean : ℝ)
variable (var : ℝ)

theorem variance_of_21_points_is_0_20 
  (h_mean : (1 / (20 : ℝ)) * ∑ i, (a i) = mean)
  (h_var : (1 / (20 : ℝ)) * ∑ i, (a i - mean) ^ 2 = 0.21) :
  let new_points := λ i : Fin 21, if h : (i : ℕ) < 20 then a ⟨i, h⟩ else mean in
  (1 / (21 : ℝ)) * ∑ i, (new_points i - mean) ^ 2 = 0.20 :=
sorry

end variance_of_21_points_is_0_20_l770_770650


namespace kale_added_per_week_l770_770244

-- Define the daily intake of asparagus and broccoli
def daily_veggie_intake (asparagus broccoli : ℝ) := 
  asparagus + broccoli

-- Define the weekly intake before and after doubling
def weekly_intake_before_doubling (daily : ℝ) := 
  daily * 7

def weekly_intake_after_doubling (weekly : ℝ) := 
  weekly * 2

-- Given conditions
def daily_asparagus := 0.25
def daily_broccoli := 0.25
def total_weekly_intake := 10

-- The final amount of kale added weekly after doubling vegetable intake
theorem kale_added_per_week : 
  let daily_intake := daily_veggie_intake daily_asparagus daily_broccoli
  let weekly_before := weekly_intake_before_doubling daily_intake
  let weekly_after := weekly_intake_after_doubling weekly_before
  total_weekly_intake - weekly_after = 3 :=
by
  sorry

end kale_added_per_week_l770_770244


namespace units_produced_by_line_B_l770_770057

-- State the problem with the given conditions and prove the question equals the answer.
theorem units_produced_by_line_B (total_units : ℕ) (B : ℕ) (A C : ℕ) 
    (h1 : total_units = 13200)
    (h2 : A + B + C = total_units)
    (h3 : ∃ d : ℕ, A = B - d ∧ C = B + d) :
    B = 4400 :=
by
  sorry

end units_produced_by_line_B_l770_770057


namespace exists_ab_l770_770948

-- Define the sets of numbers
def target_set : Set ℝ := {0, 9, 3, 6}

-- Define the numbers a and b
def a : ℝ := 2.4
def b : ℝ := 1.5

-- Define the operations
def sum := a + b
def diff := a - b
def prod := a * b
def quot := a / b

-- Prove the main statement
theorem exists_ab (a b : ℝ) :
  sum ∈ target_set ∧ diff ∈ target_set ∧ prod ∈ target_set ∧ quot ∈ target_set ∧ 
  sum ≠ diff ∧ sum ≠ prod ∧ sum ≠ quot ∧ diff ≠ prod ∧ diff ≠ quot ∧ prod ≠ quot :=
  sorry

end exists_ab_l770_770948


namespace subtraction_of_bases_l770_770113

def base8_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 7^2 + ((n % 100) / 10) * 7^1 + (n % 10) * 7^0

theorem subtraction_of_bases :
  base8_to_base10 343 - base7_to_base10 265 = 82 :=
by
  sorry

end subtraction_of_bases_l770_770113


namespace isosceles_right_triangle_tangent_circle_radius_l770_770399

theorem isosceles_right_triangle_tangent_circle_radius (a : ℝ) :
  ∃ R : ℝ, 
  (∃ (T : Type) [triangle T], is_isosceles_right_triangle T ∧ 
  leg_length T = a ∧ passes_through_vertex_opposite_acute_angle T R ∧ 
  center_on_hypotenuse T R) ∧ R = a * (2 - real.sqrt 2) :=
sorry

end isosceles_right_triangle_tangent_circle_radius_l770_770399


namespace fishbowl_count_l770_770336

def number_of_fishbowls (total_fish : ℕ) (fish_per_bowl : ℕ) : ℕ :=
  total_fish / fish_per_bowl

theorem fishbowl_count (h1 : 23 > 0) (h2 : 6003 % 23 = 0) :
  number_of_fishbowls 6003 23 = 261 :=
by
  -- proof goes here
  sorry

end fishbowl_count_l770_770336


namespace sqrt_fourth_root_x_is_point2_l770_770924

def x : ℝ := 0.000001

theorem sqrt_fourth_root_x_is_point2 :
  (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 0.000001)))) ≈ 0.2 := sorry

end sqrt_fourth_root_x_is_point2_l770_770924


namespace base4_arithmetic_l770_770435

theorem base4_arithmetic :
  let b230 : ℕ := 2 * 4 ^ 2 + 3 * 4 ^ 1 + 0 * 4 ^ 0,
      b21 : ℕ := 2 * 4 ^ 1 + 1 * 4 ^ 0,
      b2 : ℕ := 2 * 4 ^ 0,
      b12 : ℕ := 1 * 4 ^ 1 + 2 * 4 ^ 0,
      b3 : ℕ := 3 * 4 ^ 0 in
  (b230 * b21 / b2 - b12 * b3) = 3 * 4 ^ 3 + 2 * 4 ^ 2 + 2 * 4 ^ 1 + 2 * 4 ^ 0 :=
by
  sorry

end base4_arithmetic_l770_770435


namespace jacket_price_increase_l770_770866

theorem jacket_price_increase :
  ∀ (P : ℝ), 
  let PR1 := P - (0.25 * P),
      PR2 := PR1 - (0.15 * PR1),
      increase_needed := P - PR2 in
  P ≠ 0 → float.toReal ((float.ofReal increase_needed) / (float.ofReal PR2) * 100) ≈ 56.86 :=
by
  sorry

end jacket_price_increase_l770_770866


namespace sum_of_smallest_and_largest_roots_eq_five_l770_770498

theorem sum_of_smallest_and_largest_roots_eq_five :
  ∀ x1 x2 : ℝ, (9^(x1+1) + 2187 = 3^(6*x1 - x1^2)) ∧ (9^(x2+1) + 2187 = 3^(6*x2 - x2^2)) → (x1 ≤ x2) → x1 + x2 = 5 :=
by
  sorry

end sum_of_smallest_and_largest_roots_eq_five_l770_770498


namespace log2_a2015_l770_770237

noncomputable def sequence_properties (a : ℕ → ℝ) :=
  (∀ n : ℕ, a n > 0) ∧
  (∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r)

theorem log2_a2015 (a : ℕ → ℝ)
  (h_seq : sequence_properties a)
  (h_roots : (a 1) * (a 4029) = 16) :
  Real.log 2 (a 2015) = 2 :=
sorry

end log2_a2015_l770_770237


namespace sequences_problem_l770_770231

variable {a : ℕ → ℕ} 

theorem sequences_problem 
  (h1 : a 2 = 4) 
  (h2 : a 4 + a 7 = 15) :
  (∀ n, a n = n + 2) ∧ (let b (n : ℕ) := 2^(a n - 2) in (finset.range 10).sum (λ n, b (n + 1)) = 2046) :=
    by
    sorry

end sequences_problem_l770_770231


namespace maximize_volume_l770_770067

theorem maximize_volume
  (R H A : ℝ) (K : ℝ) (hA : 2 * π * R * H + 2 * π * R * (Real.sqrt (R ^ 2 + H ^ 2)) = A)
  (hK : K = A / (2 * π)) :
  R = (A / (π * Real.sqrt 5)) ^ (1 / 3) :=
sorry

end maximize_volume_l770_770067


namespace base_7_to_10_of_23456_l770_770824

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770824


namespace arithmetic_sequence_general_term_even_sum_b_sequence_odd_sum_b_sequence_l770_770158

section 

-- Given conditions
variable (n : ℕ) (a_2 : ℕ := 17) (S_10 : ℕ := 100)

-- Arithmetic sequence properties
def is_arithmetic_seq (a : ℕ → ℕ) :=
  a 2 = 17 ∧ (∑ i in Finset.range 10, a (i + 1)) = 100

-- General term formula for the sequence {a_n}
def general_term (a : ℕ → ℕ) := ∃ a_1 d, a 1 = a_1 ∧ a 2 = a_1 + d ∧ a n = a_1 + (n - 1) * d

-- Sequence {b_n}
def b_seq (a : ℕ → ℕ) (b : ℕ → ℕ) :=
  ∀ n : ℕ, b n = a n * (if even n then 1 else -1) + 2 ^ n

-- Sum of first n terms when n is even
def sum_even_b (b : ℕ → ℕ) (T : ℕ → ℕ) :=
  ∀ n : ℕ, even n → (∑ i in Finset.range n, b (i + 1)) = 2 ^ (n + 1) - n - 2

-- Sum of first n terms when n is odd
def sum_odd_b (b : ℕ → ℕ) (T : ℕ → ℕ) :=
  ∀ n : ℕ, odd n → (∑ i in Finset.range n, b (i + 1)) = 2 ^ (n + 1) + n - 22

-- Proof statement for the general term formula
theorem arithmetic_sequence_general_term (a : ℕ → ℕ) :
  is_arithmetic_seq a → general_term a :=
sorry

-- Proof statement for the sum of sequence {b_n} when n is even
theorem even_sum_b_sequence (a b : ℕ → ℕ) (h_a : general_term a) :
  b_seq a b → sum_even_b b (λ n, ∑ i in Finset.range n, b (i + 1)) :=
sorry

-- Proof statement for the sum of sequence {b_n} when n is odd
theorem odd_sum_b_sequence (a b : ℕ → ℕ) (h_a : general_term a) :
  b_seq a b → sum_odd_b b (λ n, ∑ i in Finset.range n, b (i + 1)) :=
sorry

end

end arithmetic_sequence_general_term_even_sum_b_sequence_odd_sum_b_sequence_l770_770158


namespace volume_of_body_Omega_l770_770130

noncomputable def volume_of_Omega : ℝ :=
  2 * π

theorem volume_of_body_Omega :
  let Omega := {p : ℝ × ℝ × ℝ | p.2.2 = (9 / 2) * real.sqrt (p.1^2 + p.2.1^2) ∧ 
                                   p.2.2 = (11 / 2) - p.1^2 - p.2.1^2} in
  ∃ V, V = 2 * π ∧ V = volume_of_Omega :=
by
  sorry

end volume_of_body_Omega_l770_770130


namespace percentage_tax_on_first_40000_l770_770218

def tax_on_first_40000 (P : ℝ) : ℝ := (P / 100) * 40000

def tax_on_excess_10000 : ℝ := 0.20 * 10000

def total_tax (P : ℝ) : ℝ := tax_on_first_40000 P + tax_on_excess_10000

def is_income (income : ℝ) : Prop := income = 50000

def tax_paid : ℝ := 8000

theorem percentage_tax_on_first_40000 (P : ℝ)
  (h : total_tax P = tax_paid) :
  P = 15 := by
  sorry

end percentage_tax_on_first_40000_l770_770218


namespace find_loan_amount_l770_770882

-- Definitions as per conditions
def rate_A := 0.1
def rate_C := 0.115
def gain_B := 67.5
def time_years := 3

-- Theorem stating the proof problem
theorem find_loan_amount (P : ℝ) (h1 : rate_A = 0.1) (h2 : rate_C = 0.115) (h3 : gain_B = 67.5) (h4 : time_years = 3) :
  3 * (rate_C - rate_A) * P = gain_B -> P = 1500 :=
by
  sorry

end find_loan_amount_l770_770882


namespace problem_A_problem_B_l770_770570

section complex_problems

open Complex

noncomputable def z1 : ℂ := (1 + 3 * I) / (1 - 3 * I)
noncomputable def conjugate_z1 := conj z1

theorem problem_A : conjugate_z1 = (-4 / 5) - (3 / 5) * I := by
  sorry

theorem problem_B (b : ℝ) (hb : b ≠ 0) : (b * I) ^ 2 < 0 := by
  sorry

end complex_problems

end problem_A_problem_B_l770_770570


namespace average_of_consecutive_integers_l770_770757

theorem average_of_consecutive_integers (c d: ℕ) (h: d = (c+3)) :
  (∑ i in (finset.range 7), ((d: ℕ) + i)) / 7 = c + 6 := by
  sorry

end average_of_consecutive_integers_l770_770757


namespace f_31_value_l770_770171

noncomputable def f : ℝ → ℝ := sorry

theorem f_31_value :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f x = f (2 - x)) →
  (∀ x : Icc 0 1, f x = Real.log2 (x + 1)) →
  f 31 = -1 :=
by
  intros h1 h2 h3
  sorry

end f_31_value_l770_770171


namespace plane_through_A_perpendicular_to_BC_l770_770860

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨-3, 6, 4⟩
def B : Point3D := ⟨8, -3, 5⟩
def C : Point3D := ⟨10, -3, 7⟩

-- Define the vector BC
def vectorBC (B C : Point3D) : Point3D :=
  ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩

-- Equation of the plane
def planeEquation (p : Point3D) (n : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - p.x) + n.y * (y - p.y) + n.z * (z - p.z)

theorem plane_through_A_perpendicular_to_BC : 
  planeEquation A (vectorBC B C) x y z = 0 ↔ x + z - 1 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l770_770860


namespace find_PQR_perimeter_l770_770426

variable (AP BQ CR : ℝ)
variable (a b c : ℝ)
variable (PQ QR RP : ℝ)

noncomputable def equilateral_triangle_side_length := (2 : ℝ)
noncomputable def sides_parallel := (PQ = QR ∧ QR = RP ∧ RP = PQ)
noncomputable def PQR_perimeter := (3 * PQ)
noncomputable def trapezoid_APBR_perimeter := (4 - PQ - CR)
noncomputable def trapezoid_BQCP_perimeter := (4 - PQ - AP)

theorem find_PQR_perimeter (h1 : AP = a) (h2 : BQ = b) (h3 : CR = c)
        (h4 : sides_parallel) (h5 : PQR_perimeter = trapezoid_APBR_perimeter)
        (h6 : PQR_perimeter = trapezoid_BQCP_perimeter) :
    PQ + QR + RP = (12 / 5 : ℝ) :=
  sorry

end find_PQR_perimeter_l770_770426


namespace negation_of_existence_l770_770387

theorem negation_of_existence:
  (¬ ∃ x : ℝ, sin x ≥ 1) = (∀ x : ℝ, sin x ≤ 1) :=
by
  sorry

end negation_of_existence_l770_770387


namespace tyler_final_cd_count_l770_770350

def initial_cds : ℕ := 21
def cds_given_to_sam : ℕ := initial_cds / 3
def cds_after_giving_to_sam : ℕ := initial_cds - cds_given_to_sam

def cds_bought_first : ℕ := 8
def cds_after_first_purchase : ℕ := cds_after_giving_to_sam + cds_bought_first

def cds_given_to_jenny : ℕ := 2
def cds_after_giving_to_jenny : ℕ := cds_after_first_purchase - cds_given_to_jenny

def cds_bought_second : ℕ := 12
def cds_after_second_purchase : ℕ := cds_after_giving_to_jenny + cds_bought_second

theorem tyler_final_cd_count : cds_after_second_purchase = 32 :=
by
  simp [initial_cds, cds_given_to_sam, cds_after_giving_to_sam, cds_bought_first, cds_after_first_purchase, cds_given_to_jenny, cds_after_giving_to_jenny, cds_bought_second, cds_after_second_purchase]
  exact rfl

end tyler_final_cd_count_l770_770350


namespace final_amount_correct_l770_770279

-- Given conditions
def initial_deposit_Lopez : ℝ := 100
def initial_deposit_Johnson : ℝ := 150
def initial_interest_rate : ℝ := 0.20
def new_interest_rate : ℝ := 0.18
def tax_rate : ℝ := 0.05
def first_period : ℝ := 0.5
def second_period : ℝ := 0.5

-- Compound interest calculation for the first 6 months
def semiannual_interest (rate : ℝ) (period : ℝ) := (1 + rate / 2) ^ period

def amount_after_first_period (initial_amount : ℝ) (rate : ℝ) (period : ℝ) : ℝ :=
  initial_amount * semiannual_interest(rate, period)

-- Compound interest calculation for the next 6 months
def amount_after_second_period (amount : ℝ) (rate : ℝ) (period : ℝ) : ℝ :=
  amount * semiannual_interest(rate, period)

-- Total interest earned
def total_interest_earned (final_amount : ℝ) (initial_amount : ℝ) : ℝ :=
  final_amount - initial_amount

-- Tax calculation
def tax_on_interest (interest : ℝ) (tax_rate : ℝ) : ℝ :=
  interest * tax_rate

-- Final amount calculation
def final_amount (initial_deposit : ℝ) (first_rate : ℝ) (second_rate : ℝ)
  (first_period : ℝ) (second_period : ℝ) (tax_rate : ℝ) : ℝ :=
  let A1 := amount_after_first_period initial_deposit first_rate first_period
  let A2 := amount_after_second_period A1 second_rate second_period
  let interest := total_interest_earned A2 initial_deposit
  let tax := tax_on_interest interest tax_rate
  A2 - tax

theorem final_amount_correct :
  final_amount (initial_deposit_Lopez + initial_deposit_Johnson)
               initial_interest_rate new_interest_rate
               first_period second_period tax_rate
  = 272.59 := by
  sorry

end final_amount_correct_l770_770279


namespace encode_last_a_l770_770425

theorem encode_last_a : 
  let text := "Cassandra dares radar advancement"
  let alphabet := ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  let shift (c : Char) (n : Int) : Char :=
    let idx := alphabet.indexOf c
    let new_idx := (idx + n) % 26
    alphabet[new_idx]
  let count_a := text.filter (· == 'a').length
  let total_shift := (List.range count_a).sum
  in shift 'a' total_shift = 'k' := 
by sorry

end encode_last_a_l770_770425


namespace tims_total_earnings_l770_770810

theorem tims_total_earnings (days_of_week : ℕ) (tasks_per_day : ℕ) (tasks_40_rate : ℕ) (tasks_30_rate1 : ℕ) (tasks_30_rate2 : ℕ)
    (rate_40 : ℝ) (rate_30_1 : ℝ) (rate_30_2 : ℝ) (bonus_per_50 : ℝ) (performance_bonus : ℝ)
    (total_earnings : ℝ) :
  days_of_week = 6 →
  tasks_per_day = 100 →
  tasks_40_rate = 40 →
  tasks_30_rate1 = 30 →
  tasks_30_rate2 = 30 →
  rate_40 = 1.2 →
  rate_30_1 = 1.5 →
  rate_30_2 = 2.0 →
  bonus_per_50 = 10 →
  performance_bonus = 20 →
  total_earnings = 1058 :=
by
  intros
  sorry

end tims_total_earnings_l770_770810


namespace sum_of_coefficients_expansion_l770_770455

variable (x : ℝ)

theorem sum_of_coefficients_expansion :
  let expanded_term := -(2 * x - 5) * (3 * x + 4 * (2 * x - 5)) in
  let a := -22 in
  let b := 95 in
  let c := -100 in
  a + b + c = -27 :=
by
  let expanded_term := -(2 * x - 5) * (3 * x + 4 * (2 * x - 5))
  let a := -22
  let b := 95
  let c := -100
  show a + b + c = -27
  sorry

end sum_of_coefficients_expansion_l770_770455


namespace ratio_of_areas_of_squares_l770_770797

open Real

theorem ratio_of_areas_of_squares :
  let side_length_C := 48
  let side_length_D := 60
  let area_C := side_length_C^2
  let area_D := side_length_D^2
  area_C / area_D = (16 : ℝ) / 25 :=
by
  sorry

end ratio_of_areas_of_squares_l770_770797


namespace sqrt_fourth_root_nearest_tenth_l770_770930

theorem sqrt_fourth_root_nearest_tenth :
  (Real.sqrt (Real.root 4 0.000001)) = 0.2 :=
by
  -- proof steps would go here
  sorry

end sqrt_fourth_root_nearest_tenth_l770_770930


namespace trains_meet_in_32_57_seconds_l770_770349

-- Conditions
def length_train_1 : ℝ := 100 -- meters
def length_train_2 : ℝ := 200 -- meters
def distance_between_trains : ℝ := 840 -- meters
def speed_train_1 : ℝ := 54 * 1000 / 3600 -- convert km/h to m/s
def speed_train_2 : ℝ := 72 * 1000 / 3600 -- convert km/h to m/s

-- Prove the time for the trains to meet
theorem trains_meet_in_32_57_seconds :
  (distance_between_trains + length_train_1 + length_train_2) / (speed_train_1 + speed_train_2) = 32.57 := 
by
  sorry

end trains_meet_in_32_57_seconds_l770_770349


namespace black_region_area_l770_770417

theorem black_region_area (a b : ℕ) (h1 : a = 7) (h2 : b = 3) :
  (a * a) - (b * b) = 40 := 
by 
  rw [h1, h2]
  -- Proof steps are omitted.
  sorry

end black_region_area_l770_770417


namespace sin_phi_correct_l770_770706

variables (p q r : EuclideanSpace 3 ℝ)

-- Define the magnitudes as given conditions
axiom norm_p : ∥p∥ = 2
axiom norm_q : ∥q∥ = 4
axiom norm_r : ∥r∥ = 6

-- Define the given vector triple product condition
axiom vector_condition : p × (p × q) = r

noncomputable def sin_phi : ℝ :=
  let φ := real.angle p q in
  real.sin φ

theorem sin_phi_correct : sin_phi p q r = 3 / 4 :=
by
  sorry

end sin_phi_correct_l770_770706


namespace base_7_to_10_of_23456_l770_770830

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770830


namespace incorrect_derivation_l770_770030

-- Definitions of the conditions as Lean predicates
def derivation_A (a b c : ℝ) : Prop := (a > b) → (c - a < c - b)
def derivation_B (a b c : ℝ) : Prop := (c > 0) → ((c / a > c / b) → (a < b))
def derivation_C (a b c d : ℝ) : Prop := (a > b) → (b > 0) → (c > d) → (sqrt (a / d) > sqrt (b / c))
def derivation_D (a b : ℝ) (n : ℕ) : Prop := (n > 0) → (real.pow a (1 / n) < real.pow b (1 / n)) → (a < b)

-- The main statement checking the correctness of the derivations
theorem incorrect_derivation :
  ∃ (a b c : ℝ), ∃ (d n : ℕ), 
    ¬ derivation_B 1 (-1) 1 ∧
    derivation_A a b c ∧
    derivation_C a b c d ∧
    derivation_D a b n :=
begin
  -- restate the main finding in a form that can be proofed:
  use [1, -1, 1, 1, 1],
  -- add sorries to indicate gaps
  sorry
end

end incorrect_derivation_l770_770030


namespace angle_APB_l770_770452

open Real

-- Given conditions
def center_circle : Point := {-1, 5} -- Center of the circle is at (-1, 5)
def radius_circle : ℝ := sqrt 2 -- Radius of the circle is sqrt(2)
def line_of_symmetry : Line := Line.mk (-1, 0, 0) -- Line y=-x
def point_P : Point := {-3, 3} -- Point on the line x+y=0
def line1_tangent, line2_tangent : Line := sorry -- Two tangents from P
def point_A, point_B : Point := sorry -- Points of tangency

-- The theorem to prove
theorem angle_APB : angle (line1_tangent.pt point_A) (line2_tangent.pt point_B) point_P = 60 :=
by
  sorry

end angle_APB_l770_770452


namespace change_after_buying_tickets_l770_770812

def cost_per_ticket := 8
def number_of_tickets := 2
def total_money := 25

theorem change_after_buying_tickets :
  total_money - number_of_tickets * cost_per_ticket = 9 := by
  sorry

end change_after_buying_tickets_l770_770812


namespace mass_percentage_H_in_BaOH₂_l770_770357

def mass_percentage_hydrogen_in_baoh₂
  (molar_mass_Ba : ℝ ≃ 137.33)
  (molar_mass_O : ℝ ≃ 16.00)
  (molar_mass_H : ℝ ≃ 1.01) : ℝ :=
let molar_mass_BaOH₂ := molar_mass_Ba + 2 * molar_mass_O + 2 * molar_mass_H in 
let total_mass_H := 2 * molar_mass_H in 
(total_mass_H / molar_mass_BaOH₂) * 100

theorem mass_percentage_H_in_BaOH₂ : 
  mass_percentage_hydrogen_in_baoh₂ 137.33 16.00 1.01 ≈ 1.179 :=
by {
  sorry
}

end mass_percentage_H_in_BaOH₂_l770_770357


namespace max_value_of_y_l770_770124

open Real

theorem max_value_of_y :
  ∃ x ∈ Icc (-2 * π / 3) (-π / 6), 
    tan (x + 5 * π / 6) - tan (x + π / 4) + cos (x + π / 4) = -3 * sqrt 2 / 2 :=
sorry

end max_value_of_y_l770_770124


namespace find_number_l770_770071

theorem find_number :
  ∃ (N : ℝ), (5 / 4) * N = (4 / 5) * N + 45 ∧ N = 100 :=
by
  sorry

end find_number_l770_770071


namespace max_slope_min_linear_combination_l770_770215

variables {x y : ℝ}

-- Conditions (definitions)
def circle_equation : Prop := (x + 2)^2 + y^2 = 3

-- Statements to prove
theorem max_slope (h : circle_equation) : y / x = sqrt 3 :=
sorry

theorem min_linear_combination (h : circle_equation) : 2 * y - x = 1 - (sqrt 15) / 2 :=
sorry

end max_slope_min_linear_combination_l770_770215


namespace cube_coloring_count_l770_770815

-- Define the condition that the faces of the cube are labeled such that each pair of opposite faces
-- are colored differently from a set of 6 distinct colors.
def is_valid_cube (c : Fin 6 → Fin 6) : Prop :=
  (c 0 ≠ c 5) ∧ (c 1 ≠ c 4) ∧ (c 2 ≠ c 3)

-- Prove that the number of distinct colorings (up to rotations) of the cube meeting the conditions is 360.
theorem cube_coloring_count : (number of distinct cubes formed by painting the faces with 6 different colors, with numbering conditions) = 360 :=
by
  sorry

end cube_coloring_count_l770_770815


namespace problem_mod_l770_770443

theorem problem_mod (a b c d : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) (h4 : d = 2014) :
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end problem_mod_l770_770443


namespace fewer_mpg_in_city_l770_770394

def city_mpg := 14
def city_distance := 336
def highway_distance := 480

def tank_size := city_distance / city_mpg
def highway_mpg := highway_distance / tank_size
def fewer_mpg := highway_mpg - city_mpg

theorem fewer_mpg_in_city : fewer_mpg = 6 := by
  sorry

end fewer_mpg_in_city_l770_770394


namespace part1_part2_part3_l770_770688

-- Defining the probability functions for the respective events
noncomputable def P_A : ℚ := 1 / 4
noncomputable def P_B : ℚ := 3 / 4
noncomputable def S : ℝ := 25 * Real.sqrt 3
noncomputable def S₁ : ℝ := 3 * (Real.pi / 6)
noncomputable def P_C : ℝ := 1 - (Real.sqrt 3 * Real.pi / 150)

-- Stating the theorems to prove
theorem part1 : P_A = 1 / 4 :=
by 
  exact P_A

theorem part2 : P_B = 3 / 4 :=
by 
  exact P_B

theorem part3 : P_C = 1 - (Real.sqrt 3 * Real.pi / 150) :=
by 
  exact P_C

end part1_part2_part3_l770_770688


namespace positive_x_values_l770_770942

noncomputable def logBase (b x : ℝ) : ℝ := log x / log b

theorem positive_x_values (x : ℝ) (h1 : 0 < x) : 
  (logBase 2 x) * (logBase x 9) = logBase 2 9 ↔ x ≠ 1 := by
  sorry

end positive_x_values_l770_770942


namespace total_cost_over_8_weeks_l770_770634

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l770_770634


namespace find_line_eq_l770_770308

noncomputable def circle_eq : Polynomial ℝ := (λ x y: ℝ, x^2 + y^2 + 4*y - 21 = 0)

def point_M := (-3, -3)

theorem find_line_eq :
  (∀ l : ℝ → ℝ → Prop, (l point_M.1 point_M.2) → (l = (λ x y : ℝ, x + 2*y + 9 = 0) ∨ l = (λ x y : ℝ, 2*x - y + 3 = 0))) :=
sorry

end find_line_eq_l770_770308


namespace square_field_area_l770_770034

/--
Given that a man is walking at a speed of 6 km per hour
and crosses a square field diagonally in 9 seconds,
we need to prove that the area of the square field is approximately 112.36 square meters.
-/

variable (walking_speed : ℝ) (cross_time : ℝ) (diagonal_length : ℝ) (side_length : ℝ) (area : ℝ)

-- Defining the conditions
def man_walking_speed_in_km_per_hour : Prop := walking_speed = 6   -- speed of the man in km per hour
def time_to_cross_field_in_seconds : Prop := cross_time = 9      -- time taken to cross the field in seconds

-- Calculation to determine the distance of the diagonal
def diagonal_length_in_meters : Prop :=
  diagonal_length = ((walking_speed / 3600) * cross_time) * 1000

-- Using the Pythagorean theorem to calculate the side length of the square
def calculate_side_length : Prop :=
  side_length = real.sqrt (diagonal_length * diagonal_length / 2)

-- Calculating the area of the square field
def calculate_area : Prop := area = side_length * side_length

-- Assert the area should be approximately 112.36 square meters
def area_of_square_field : Prop := abs (area - 112.36) < 0.01

-- Final statement combining above conditions to prove the desired property
theorem square_field_area 
  (h1 : man_walking_speed_in_km_per_hour walking_speed)
  (h2 : time_to_cross_field_in_seconds cross_time)
  (h3 : diagonal_length_in_meters walking_speed cross_time diagonal_length)
  (h4 : calculate_side_length diagonal_length side_length)
  (h5 : calculate_area side_length area)
  : area_of_square_field area :=
by
  sorry

-- Variables instantiation
example : walking_speed = 6 := rfl
example : cross_time = 9 := rfl
example : diagonal_length = (((6 : ℝ) / 3600) * 9) * 1000 := rfl
example : side_length = real.sqrt (54 * 1000 / 2) := rfl
example : area = (real.sqrt (54 * 1000 / 2)) * (real.sqrt (54 * 1000 / 2)) := rfl

end square_field_area_l770_770034


namespace find_greatest_consecutive_integer_l770_770173

theorem find_greatest_consecutive_integer (n : ℤ) 
  (h : n^2 + (n + 1)^2 = 452) : n + 1 = 15 :=
sorry

end find_greatest_consecutive_integer_l770_770173


namespace sum_possible_x_l770_770855

noncomputable def sum_of_x (x : ℝ) : ℝ :=
  let lst : List ℝ := [1, 2, 5, 2, 3, 2, x]
  let mean := (1 + 2 + 5 + 2 + 3 + 2 + x) / 7
  let median := 2
  let mode := 2
  if lst = List.reverse lst ∧ mean ≠ mode then
    mean
  else 
    0

theorem sum_possible_x : sum_of_x 1 + sum_of_x 5 = 6 :=
by 
  sorry

end sum_possible_x_l770_770855


namespace tangency_bisects_segment_iff_l770_770747

namespace Triangle

variables {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the sides of the triangle
variables (a b c : ℝ)

-- Define points M, H, and T
variables (M H T : A → B → C → Prop)

-- Conditions:
-- M is the midpoint of side AB
-- H is the foot of the altitude from vertex C to side AB
-- T is the point of tangency of the incircle with side AB

-- Prove the necessary and sufficient condition
theorem tangency_bisects_segment_iff (a b c : ℝ) (M H T : Type*)
  (hM : is_midpoint A B M)
  (hH : is_foot_of_altitude C A B H)
  (hT : is_tangency_point A B T) : 
  (c = (a + b) / 2) ↔ (bisects_segment T H M) :=
sorry

end Triangle

end tangency_bisects_segment_iff_l770_770747


namespace base_7_to_10_of_23456_l770_770829

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770829


namespace solve_for_B_l770_770010

theorem solve_for_B (B : ℕ) (h : 3 * B + 2 = 20) : B = 6 :=
by 
  -- This is just a placeholder, the proof will go here
  sorry

end solve_for_B_l770_770010


namespace card_game_maximum_difference_l770_770322

-- Define the card game setup
def card_game := {cards : fin 400 → ℕ}

-- Define the initial card setup (i.e., the numbers on the 400 cards)
def initial_setup (game : card_game) : Prop :=
  ∀ i, game.cards i = i + 1

-- Define the optimal play conditions (no specific strategy because it's abstract)
def optimal_play (game : card_game) : Prop :=
  -- A takes 200 cards, leaving B and A with 200 cards each in the first step
  let A1 := (fin 200 → ℕ) in
  let B1 := (fin 200 → ℕ) in

  -- Continue turns until 200th step
  ∀ n : fin 200, A1 n = sorry ∧ B1 n = sorry -- replace with precise logic as needed

-- Define the final score calculation for players A and B
def final_scores (A_cards B_cards : fin 200 → ℕ) : ℕ × ℕ :=
  (A_cards.sum, B_cards.sum)

-- The theorem to prove the maximum difference
theorem card_game_maximum_difference (game : card_game) (h1 : initial_setup game)
  (h2 : optimal_play game) :
  ∃ C_A C_B : ℕ, let ⟨C_A, C_B⟩ := final_scores (sorry sorry) in C_B - C_A = 20000 :=
by
  sorry

end card_game_maximum_difference_l770_770322


namespace probability_M_greater_than_10_l770_770412

-- Define the function
def f (x a : ℕ) := 700 / (x^2 - 2*x + 2*a)

-- Define the maximum value M
def M (a : ℕ) := 700 / (2*a - 1)

-- Define the probability function
def probability_exceeds_10 : ℕ := (nat.card {a | 1 ≤ a ∧ a ≤ 35}) * 100 / (nat.card {a | 1 ≤ a ∧ a ≤ 100})

-- State the theorem
theorem probability_M_greater_than_10 : probability_exceeds_10 = 35 := by
  sorry

end probability_M_greater_than_10_l770_770412


namespace frequency_of_heads_l770_770225

theorem frequency_of_heads (n h : ℕ) (h_n : n = 100) (h_h : h = 49) : (h : ℚ) / n = 0.49 :=
by
  rw [h_n, h_h]
  norm_num

end frequency_of_heads_l770_770225


namespace betting_possible_l770_770224

theorem betting_possible :
  ∃ (M x1 x2 x3 x4 : ℝ),
    x1 + x2 + x3 + x4 = M ∧
    3 * x1 ≥ M ∧
    4 * x2 ≥ M ∧
    5 * x3 ≥ M ∧
    8 * x4 ≥ M :=
by
  use 1 / (1/3 + 1/4 + 1/5 + 1/8), (1/4 + 1/5 + 1/8) / (1/3 + 1/4 + 1/5 + 1/8), (1/3 + 1/5 + 1/8) / (1/3 + 1/4 + 1/5 + 1/8), (1/3 + 1/4 + 1/8) / (1/3 + 1/4 + 1/5 + 1/8), (1/3 + 1/4 + 1/5) / (1/3 + 1/4 + 1/5 + 1/8)
  split; norm_num
  sorry

end betting_possible_l770_770224


namespace math_problem_l770_770500

theorem math_problem (x : ℝ) :
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ∧ (x^2 - 6 * x + 8 ≥ 0) → (x ∈ set.Icc (-5) 1 ∪ set.Icc 5 11) :=
by 
  sorry

end math_problem_l770_770500


namespace count_three_digit_numbers_l770_770640

open Finset

/-- The number of three-digit numbers where the hundreds digit is greater than the tens digit,
and the tens digit is greater than the ones digit. -/
theorem count_three_digit_numbers (U : set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  (U.to_finset.choose 3).card = 84 :=
by sorry

end count_three_digit_numbers_l770_770640


namespace greatest_distinct_digits_divisible_l770_770955

-- The Lean statement that captures the problem and the answer.
theorem greatest_distinct_digits_divisible :
  ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧
    (∀ d ∈ n.digits 10, d ≠ 0 ∧ n % d = 0) ∧
    (n.digits 10).nodup ∧
    (∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ (∀ d ∈ m.digits 10, d ≠ 0 ∧ m % d = 0) ∧ (m.digits 10).nodup) → m ≤ n) ∧
    n = 9864 :=
by sorry

end greatest_distinct_digits_divisible_l770_770955


namespace unique_determination_of_T_n_l770_770502

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

def sum_of_arithmetic_sums (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (n + 1) * (3 * a + (n - 1) * d)) / 6

theorem unique_determination_of_T_n (a d : ℤ) :
  let S_2023 := 2023 * (a + 1011 * d) in
  ∀ n, (S_2023 = 2023 * (a + 1011 * d)) → 
  (sum_of_arithmetic_sums a d 3034) = (sum_of_arithmetic_sums a d n) → n = 3034 :=
by
  sorry

end unique_determination_of_T_n_l770_770502


namespace quadratic_distinct_roots_l770_770538

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770538


namespace projection_sum_of_squares_l770_770289

theorem projection_sum_of_squares (a : ℝ) (α β γ : ℝ) 
    (h1 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) 
    (h2 : (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = 2) :
    4 * a^2 * ((Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2) = 8 * a^2 := 
by
  sorry

end projection_sum_of_squares_l770_770289


namespace remainder_add_mod_l770_770027

theorem remainder_add_mod (n : ℕ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := 
by sorry

end remainder_add_mod_l770_770027


namespace baez_final_marble_count_l770_770086

def final_marble_count (initial : ℕ) (loss_percent : ℕ) (trade_fraction : ℚ) (gain_multiplier : ℕ) : ℕ :=
  let lost_marbles := initial * loss_percent / 100
  let remaining_after_loss := initial - lost_marbles
  let traded_marbles := (remaining_after_loss * trade_fraction).natFloor
  let remaining_after_trade := remaining_after_loss - traded_marbles
  let received_marbles := remaining_after_trade * gain_multiplier
  remaining_after_trade + received_marbles

theorem baez_final_marble_count :
  final_marble_count 25 20 (1/3 : ℚ) 2 = 42 :=
by
  sorry

end baez_final_marble_count_l770_770086


namespace surface_area_of_tetrahedron_sphere_l770_770578

-- Define the edge length of the regular tetrahedron
def edge_length : ℝ := 2

-- Define the regular tetrahedron with given edge length
structure RegularTetrahedron (a : ℝ) :=
  (edge_length : ℝ := a)
  (is_regular : ∀ {a b c d : ℝ}, a = b ∧ b = c ∧ c = d ∧ d = a)

-- Instance for the edge length and regular property
def tetrahedron := RegularTetrahedron edge_length

-- Define the surface area of the circumscribed sphere
def surface_area_of_circumscribed_sphere (T : RegularTetrahedron edge_length) : ℝ :=
  6 * Real.pi

-- The main theorem statement
theorem surface_area_of_tetrahedron_sphere : surface_area_of_circumscribed_sphere tetrahedron = 6 * Real.pi :=
by sorry

end surface_area_of_tetrahedron_sphere_l770_770578


namespace base_7_to_10_of_23456_l770_770825

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770825


namespace hockey_league_num_games_l770_770867

theorem hockey_league_num_games :
  ∃ (num_teams : ℕ) (num_times : ℕ), 
    num_teams = 16 ∧ num_times = 10 ∧ 
    (num_teams * (num_teams - 1) / 2) * num_times = 2400 := by
  sorry

end hockey_league_num_games_l770_770867


namespace base_7_to_base_10_l770_770836

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770836


namespace number_of_subsets_with_mean_7_l770_770637

theorem number_of_subsets_with_mean_7 : 
  let s := {1, 3, 4, 5, 7, 8, 9, 11, 12, 14} in
  (∃! t ⊆ s, t.card = 2 ∧ (s \ t).sum / (s.card - 2) = 7) :=
sorry

end number_of_subsets_with_mean_7_l770_770637


namespace sum_of_cubes_mod_5_l770_770482

theorem sum_of_cubes_mod_5 :
  ( ∑ k in Finset.range 50, (k + 1)^3 ) % 5 = 0 :=
sorry

end sum_of_cubes_mod_5_l770_770482


namespace clock_angle_at_3_18_l770_770017

theorem clock_angle_at_3_18 :
  let minute_angle := (18 / 60) * 360
      hour_angle := (3 * 30) + (18 / 60) * 30
  in |minute_angle - hour_angle| = 9 := 
by {
  let minute_angle := (18 / 60) * 360
  let hour_angle := (3 * 30) + (18 / 60) * 30
  have h1 : minute_angle = 108 := by norm_num
  have h2 : hour_angle = 99 := by norm_num
  have h3 : |minute_angle - hour_angle| = |108 - 99| := by rw [h1, h2]
  have h4 : |108 - 99| = 9 := by norm_num
  rw [h3, h4]
  norm_num
}

end clock_angle_at_3_18_l770_770017


namespace no_perfect_square_E_l770_770258

noncomputable def E (x : ℝ) : ℤ :=
  round x

theorem no_perfect_square_E (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, E (n + Real.sqrt n) = k * k) :=
  sorry

end no_perfect_square_E_l770_770258


namespace isosceles_right_triangle_area_l770_770911

theorem isosceles_right_triangle_area (a b : ℝ) (h₁ : a = b) (h₂ : a + b = 20) : 
  (1 / 2) * a * b = 50 := 
by 
  sorry

end isosceles_right_triangle_area_l770_770911


namespace range_of_k_has_extreme_values_on_interval_l770_770314

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x - x^2 + 3 * x

theorem range_of_k_has_extreme_values_on_interval (k : ℝ) (h : k ≠ 0) :
  -9/8 < k ∧ k < 0 :=
sorry

end range_of_k_has_extreme_values_on_interval_l770_770314


namespace base_7_to_base_10_l770_770839

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l770_770839


namespace tangents_and_circle_l770_770304

-- Defining the ellipse equation and the given point P
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36
def P : ℝ × ℝ := (3/5, 14/5)
def tangent1 (x y : ℝ) : Prop := -x + 3 * y = 5
def tangent2 (x y : ℝ) : Prop := 8 * x + 9 * y = 30

-- Points of tangency
def P1 : ℝ × ℝ := (-9/5, 12/5)
def P2 : ℝ × ℝ := (12/5, 6/5)

-- Defining the circle equation and its area
def circle (x y : ℝ) : Prop := (x - 11/50)^2 + (y - 14/25)^2 = (Real.sqrt 60905 / 50)^2
noncomputable def circle_area : ℝ := 76.446 * Real.pi

-- The theorem to be proved
theorem tangents_and_circle :
  (tangent1 P.1 P.2 ∧ tangent2 P.1 P.2) ∧ 
  (circle P1.1 P1.2 ∧ circle P2.1 P2.2 ∧ circle P.1 P.2) ∧
  (circle_area = 76.446 * Real.pi) :=
  sorry

end tangents_and_circle_l770_770304


namespace correct_location_l770_770453

variable (A B C D : Prop)

axiom student_A_statement : ¬ A ∧ B
axiom student_B_statement : ¬ B ∧ C
axiom student_C_statement : ¬ B ∧ ¬ D
axiom ms_Hu_response : 
  ( (¬ A ∧ B = true) ∨ (¬ B ∧ C = true) ∨ (¬ B ∧ ¬ D = true) ) ∧ 
  ( (¬ A ∧ B = false) ∨ (¬ B ∧ C = false) ∨ (¬ B ∧ ¬ D = false) = false ) ∧ 
  ( (¬ A ∧ B ∨ ¬ B ∧ C ∨ ¬ B ∧ ¬ D) -> false )

theorem correct_location : B ∨ A := 
sorry

end correct_location_l770_770453


namespace solve_inequality_l770_770296

theorem solve_inequality :
  {x : ℝ | -3 * x^2 + 5 * x + 4 < 0} = {x : ℝ | x < 3 / 4} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solve_inequality_l770_770296


namespace boat_speed_ratio_l770_770392

theorem boat_speed_ratio 
  (speed_still_water : ℝ)
  (speed_current : ℝ) :
  speed_still_water = 20 ∧ 
  speed_current = 6 →
  let downstream_speed := speed_still_water + speed_current in
  let upstream_speed := speed_still_water - speed_current in
  let distance := 1 in
  let time_downstream := distance / downstream_speed in
  let time_upstream := distance / upstream_speed in
  let total_time := time_downstream + time_upstream in
  let total_distance := 2 * distance in
  let average_speed := total_distance / total_time in
  (average_speed / speed_still_water) = 91 / 100 :=
by
  sorry

end boat_speed_ratio_l770_770392


namespace avg_of_remaining_two_l770_770373

-- Definition of the conditions
def avg_of_five_numbers (a b c d e : ℝ) : Prop := (a + b + c + d + e) / 5 = 20
def sum_of_three_is_48 (a b c d e : ℝ) : Prop := a + b + c = 48

-- The proof problem statement
theorem avg_of_remaining_two (a b c d e : ℝ) 
  (h1 : avg_of_five_numbers a b c d e)
  (h2 : sum_of_three_is_48 a b c) : 
  (d + e) / 2 = 26 := 
sorry

end avg_of_remaining_two_l770_770373


namespace sampling_probabilities_equal_l770_770575

noncomputable def populationSize (N : ℕ) := N
noncomputable def sampleSize (n : ℕ) := n

def P1 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P2 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P3 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)

theorem sampling_probabilities_equal (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  P1 N n = P2 N n ∧ P2 N n = P3 N n :=
by
  -- Proof steps will go here
  sorry

end sampling_probabilities_equal_l770_770575


namespace root_exists_in_interval_l770_770317

noncomputable def f (x : ℝ) : ℝ := x^3 - 4

theorem root_exists_in_interval :
  ∃ c ∈ set.Ioo 1 2, f c = 0 :=
by
  have h1 : f 1 < 0 := by norm_num
  have h2 : f 2 > 0 := by norm_num
  sorry

end root_exists_in_interval_l770_770317


namespace washing_machine_capacity_l770_770354

-- Define the conditions:
def shirts : ℕ := 39
def sweaters : ℕ := 33
def loads : ℕ := 9
def total_clothes : ℕ := shirts + sweaters -- which is 72

-- Define the statement to be proved:
theorem washing_machine_capacity : ∃ x : ℕ, loads * x = total_clothes ∧ x = 8 :=
by
  -- proof to be completed
  sorry

end washing_machine_capacity_l770_770354


namespace sum_of_digits_of_x_l770_770389

def is_palindrome (n : ℕ) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

theorem sum_of_digits_of_x 
  (x : ℕ) 
  (hx : x.digits 10 = [a, b, a]) 
  (hx_plus_50 : (x + 50).digits 10 = [c, d, d, c]) :
  x.digits 10.sum = 15 := 
sorry

end sum_of_digits_of_x_l770_770389


namespace domain_of_f_l770_770355

noncomputable def f (x : ℝ) : ℝ := (x^3 + 8) / (x - 8)

theorem domain_of_f : ∀ x : ℝ, x ≠ 8 ↔ ∃ y : ℝ, f x = y :=
  by admit

end domain_of_f_l770_770355


namespace solution_to_trig_equation_l770_770323

theorem solution_to_trig_equation (x : ℝ) (k : ℤ) :
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
  (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) →
  (Real.sin (x / 2) = Real.cos (x / 2)) →
  (∃ k : ℤ, x = (π / 2) + 2 * π * ↑k) :=
by sorry

end solution_to_trig_equation_l770_770323


namespace newspaper_spending_over_8_weeks_l770_770633

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l770_770633


namespace bus_speed_proof_l770_770900
noncomputable def speed_of_bus (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed_mps := train_length / time_to_pass
  let bus_speed_mps := relative_speed_mps - train_speed_mps
  bus_speed_mps * 3.6

theorem bus_speed_proof : 
  speed_of_bus 220 90 5.279577633789296 = 60 :=
by
  sorry

end bus_speed_proof_l770_770900


namespace domain_f_log2_x_to_domain_f_x_l770_770169

variable {f : ℝ → ℝ}

-- Condition: The domain of y = f(log₂ x) is [1/2, 4]
def domain_f_log2_x : Set ℝ := Set.Icc (1 / 2) 4

-- Proof statement
theorem domain_f_log2_x_to_domain_f_x
  (h : ∀ x, x ∈ domain_f_log2_x → f (Real.log x / Real.log 2) = f x) :
  Set.Icc (-1) 2 = {x : ℝ | ∃ y ∈ domain_f_log2_x, Real.log y / Real.log 2 = x} :=
by
  sorry

end domain_f_log2_x_to_domain_f_x_l770_770169


namespace sum_cubes_mod_five_l770_770490

theorem sum_cubes_mod_five :
  (∑ n in Finset.range 50, (n + 1)^3) % 5 = 0 :=
by
  sorry

end sum_cubes_mod_five_l770_770490


namespace AIGolovanov_AYakubov_l770_770157

/-- Given an acute triangle ABC with AB < AC. Let Ω be the circumcircle of ABC and M be the centroid of triangle ABC. AH is the altitude of ABC. MH intersects with Ω at A'. Prove that the circumcircle of triangle A'HB is tangent to AB. -/
theorem AIGolovanov_AYakubov  : 
  ∀ (A B C M H A' : Type) 
    [HCircumcircle : CircumcircleOf ABC]
    [HAcute : AcuteTriangle A B C]
    [HCentre : CentroidOf ABC M]
    [HAltitude : Altitude A H]
    [HIntersect : IntersectionPointOf MH Ω A'],
  IsTangent (Circumcircle (Triangle A' H B)) AB :=
sorry

end AIGolovanov_AYakubov_l770_770157


namespace max_sum_of_factors_l770_770381

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 42) : a + b ≤ 43 :=
by
  -- sorry to skip the proof
  sorry

end max_sum_of_factors_l770_770381


namespace relationship_between_y1_y2_l770_770209

theorem relationship_between_y1_y2 
  (y1 y2 : ℝ) 
  (hA : y1 = 6 / -3) 
  (hB : y2 = 6 / 2) : y1 < y2 :=
by 
  sorry

end relationship_between_y1_y2_l770_770209


namespace range_of_a_l770_770147

noncomputable def f (a x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (f a x) > 0) ↔ a ≤ 2 := 
by
  sorry

end range_of_a_l770_770147


namespace base7_to_base10_l770_770822

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l770_770822


namespace ratio_bananas_dates_l770_770281

theorem ratio_bananas_dates (s c b d a : ℕ)
  (h1 : s = 780)
  (h2 : c = 60)
  (h3 : b = 3 * c)
  (h4 : a = 2 * d)
  (h5 : s = a + b + c + d) :
  b / d = 1 :=
by sorry

end ratio_bananas_dates_l770_770281


namespace quadratic_distinct_roots_l770_770549

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770549


namespace not_possible_to_cut_rectangular_paper_l770_770366

theorem not_possible_to_cut_rectangular_paper 
  (a : ℝ) (b : ℝ) (s_area : ℝ) (r_area : ℝ) (ratio : ℝ)
  (h1 : s_area = 100) (h2 : r_area = 90) (h3 : ratio = 5 / 3) :
  ¬ (∃ l w, l / w = ratio ∧ l * w = r_area ∧ l ≤ sqrt s_area ∧ w ≤ sqrt s_area) := by
  sorry

end not_possible_to_cut_rectangular_paper_l770_770366


namespace emily_took_55_apples_l770_770753

theorem emily_took_55_apples (original_apples : ℕ) (left_apples : ℕ) (h1 : original_apples = 63) (h2 : left_apples = 8) : original_apples - left_apples = 55 :=
by
  rw [h1, h2]
  exact rfl

end emily_took_55_apples_l770_770753


namespace decreasing_interval_a_geq_3_l770_770608

noncomputable def f (a b : ℝ) : ℝ → ℝ :=
  λ x, (1 / 3) * x^3 - (a / 2) * x^2 + (5 - a) * x + b

noncomputable def f' (a : ℝ) : ℝ → ℝ :=
  λ x, x^2 - a * x + (5 - a)

theorem decreasing_interval_a_geq_3
  (a b : ℝ)
  (h_decreasing : ∀ x, 1 < x ∧ x < 2 → f' a x ≤ 0) :
  3 ≤ a := 
sorry

end decreasing_interval_a_geq_3_l770_770608


namespace least_distinct_values_l770_770405

def unique_mode (lst : List ℕ) (m : ℕ) : Prop :=
  ∃! n, n ∈ lst ∧ (∀ k, count k lst = m ↔ k = n)

def num_distinct (lst : List ℕ) : ℕ :=
  lst.eraseDuplicates.length

theorem least_distinct_values (lst : List ℕ) (mode_val : ℕ)
  (h_len : lst.length = 2412)
  (h_mode : unique_mode lst 12) :
  num_distinct lst = 219 :=
sorry

end least_distinct_values_l770_770405


namespace price_difference_proof_l770_770327

theorem price_difference_proof (y : ℝ) (n : ℕ) :
  ∃ n : ℕ, (4.20 + 0.45 * n) = (6.30 + 0.01 * y * n + 0.65) → 
  n = (275 / (45 - y)) :=
by
  sorry

end price_difference_proof_l770_770327


namespace first_fantastic_friday_l770_770409

theorem first_fantastic_friday (january_days : ℕ) (start_day : String) (first_friday : ℕ) :
  january_days = 31 →
  start_day = "Wednesday" →
  first_friday = 5 →
  ∃ fifth_friday : ℕ, fifth_friday = 30 :=
by
  intro january_days_eq
  intro start_day_eq
  intro first_friday_eq
  use 30
  sorry

end first_fantastic_friday_l770_770409


namespace height_difference_l770_770009

theorem height_difference (nA nB : ℕ)
  (d : ℝ)
  (hA : nA = 10)
  (dA : d = 20)
  (heightA : total_heightA = nA * d)
  (hB1 : nB = 9)
  (heightB1 : total_heightB = 180 + 80 * Real.sqrt(3)) : 
  | total_heightA - total_heightB | = | 20 - 80 * Real.sqrt(3) | :=
sorry

end height_difference_l770_770009


namespace percentage_increase_books_l770_770282

theorem percentage_increase_books (
    M : ℕ,                        -- Number of books Matt read last year
    H1 : 2 * M + 4 * M = 300,     -- Total books read by Pete in both years
    H2 : (75 - M) / M * 100 = 50  -- Percentage increase formula application
) : (75 - M) / M * 100 = 50 := 
by
  sorry

end percentage_increase_books_l770_770282


namespace min_uninteresting_vertices_l770_770378

theorem min_uninteresting_vertices (n : ℕ) (h1 : n > 3) 
  (h2 : ∀ (i j : ℕ) (hi : i < n) (hj : j < n) (hij : i ≠ j), 
    dist (vertices i) (vertices j) ≠ dist (vertices (i + 1 % n)) (vertices (j + 1 % n))) 
  : ∃ (m : ℕ), m = 2 := 
sorry

end min_uninteresting_vertices_l770_770378


namespace quadratic_distinct_roots_l770_770548

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770548


namespace max_PQ_square_l770_770271

noncomputable def square_side : ℝ := 1
def point_P : ℝ × ℝ 
def point_Q : ℝ × ℝ 
def point_D : ℝ × ℝ

def is_circumcenter_of_triangle (Q P B C: ℝ × ℝ) : Prop :=
  (dist Q B = dist Q P) ∧ (dist Q B = dist Q C)

def is_circumcenter_of_triangle_PQA (D P Q A: ℝ × ℝ) : Prop :=
  (dist D P = dist D Q) ∧ (dist D P = dist D A)

theorem max_PQ_square 
  (ABCD_square : square_side = 1)
  (circumcenter_Q : is_circumcenter_of_triangle Q P (1, 1) (1, 0))
  (circumcenter_D : is_circumcenter_of_triangle_PQA D P Q (0, 1))
  : ∃ a b: ℚ, PQ^2 = a + sqrt b ∧ PQ^2 = 2 + sqrt 3 :=
sorry

end max_PQ_square_l770_770271


namespace smallest_sum_condition_l770_770775

noncomputable def p : ℕ := 50000
noncomputable def q : ℕ := 60000
noncomputable def r : ℕ := 70000
noncomputable def s : ℕ := 80000

theorem smallest_sum_condition :
  p * q * r * s = 15! ∧ p + q + r + s = 260000 :=
by
  sorry

end smallest_sum_condition_l770_770775


namespace find_original_cost_l770_770750

noncomputable def original_cost (C : ℝ) : Prop :=
  let repairs := 13000
  let selling_price := 64900
  let profit_percent := 0.18
  selling_price = (C + repairs) + (profit_percent * (C + repairs)) ∧
  C = 42000

theorem find_original_cost : ∃ C : ℝ, original_cost C :=
by {
  use 42000,
  unfold original_cost,
  split,
  norm_num,
  norm_num,
  sorry
}

end find_original_cost_l770_770750


namespace s_at_1_l770_770270

def t (x : ℚ) := 5 * x - 12
def s (y : ℚ) := (y + 12) / 5 ^ 2 + 5 * ((y + 12) / 5) - 4

theorem s_at_1 : s 1 = 394 / 25 := by
  sorry

end s_at_1_l770_770270


namespace smallest_n_l770_770695

def is_valid_solution (n m : ℕ) (xs : Fin n → ℝ) : Prop :=
  (∀ i, -1 < xs i ∧ xs i < 1) ∧
  (∑ i, |xs i| = m + |∑ i, xs i|)

theorem smallest_n (m : ℕ) (hm : m > 0) :
  ∃ n : ℕ, n > 0 ∧ (∀ xs : Fin n → ℝ, is_valid_solution n m xs → n = m + 1) :=
by
  sorry

end smallest_n_l770_770695


namespace find_N_aN_bN_cN_dN_eN_l770_770722

theorem find_N_aN_bN_cN_dN_eN:
  ∃ (a b c d e : ℝ) (N : ℝ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧
    (a^2 + b^2 + c^2 + d^2 + e^2 = 1000) ∧
    (N = c * (a + 3 * b + 4 * d + 6 * e)) ∧
    (N + a + b + c + d + e = 150 + 250 * Real.sqrt 62 + 10 * Real.sqrt 50) := by
  sorry

end find_N_aN_bN_cN_dN_eN_l770_770722


namespace find_g_six_l770_770315

axiom g : ℝ → ℝ
axiom functional_property : ∀ x y : ℝ, g(x + y) = g(x) * g(y)
axiom g_two : g(2) = 4

theorem find_g_six : g(6) = 64 :=
by
  sorry

end find_g_six_l770_770315


namespace newspaper_cost_over_8_weeks_l770_770630

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l770_770630


namespace range_of_f_l770_770602

variables (a b c x : ℝ)
hypothesis h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c 

def f (x a b c : ℝ) : ℝ :=
  (x + a)^2 / ((a - b) * (a - c)) +
  (x + b)^2 / ((b - a) * (b - c)) +
  (x + c)^2 / ((c - a) * (c - b))

theorem range_of_f (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∀ x, f x a b c = 1 :=
by
  -- sorry is a placeholder for the proof
  sorry

end range_of_f_l770_770602


namespace max_length_diameter_intersection_l770_770711

theorem max_length_diameter_intersection {P Q X Y Z : ℝ} (diam_PQ : P = 0 ∧ Q = 1)
(mid_X : X = 0.5) (dist_PY : PY = 2/5) :
  ∃ u v w : ℕ, (w ∉ {p * p | p : ℕ}) ∧ 
  (∀ e, (e = |(PQ ∩ XZ) - (PQ ∩ YZ)|) → 
  (e ≤ u - v * real.sqrt w)) :=
sorry

end max_length_diameter_intersection_l770_770711


namespace yield_and_excess_naCl_l770_770021

-- Define the chemical reaction
def reaction1 (NaOH HCl NaCl H2O : ℝ) : Prop :=
  NaOH = 1 ∧ HCl = 1 ∧ NaCl = 1 ∧ H2O = 1

def reaction2 (NaCl AgNO3 AgCl NaNO3 : ℝ) : Prop :=
  NaCl = 0.5 ∧ AgNO3 = 0.5 ∧ AgCl = 0.5 ∧ NaNO3 = 0.5

-- Define the limiting reagent
def limiting_reagent (AgNO3_initial : ℝ) : Prop :=
  AgNO3_initial = 0.5

-- Define the yield calculation
def percent_yield (NaCl_formed NaCl_reacted : ℝ) : ℝ :=
  (NaCl_reacted / NaCl_formed) * 100

-- Define the excess reagent calculation
def excess_reagent (NaCl_initial NaCl_reacted : ℝ) : ℝ :=
  NaCl_initial - NaCl_reacted

-- Define given conditions
def conditions : Prop :=
  reaction1 1 1 1 1 ∧ reaction2 1 0.5 0.5 0.5 ∧ limiting_reagent 0.5

-- Prove the percent yield and the excess reagent remaining
theorem yield_and_excess_naCl :
  ∀ (NaCl_formed NaCl_reacted NaCl_initial : ℝ),
  NaCl_formed = 1 →
  NaCl_reacted = 0.5 →
  NaCl_initial = 1 →
  conditions →
  percent_yield NaCl_formed NaCl_reacted = 50 ∧
  excess_reagent NaCl_initial NaCl_reacted = 0.5 :=
by
  intros
  sorry

end yield_and_excess_naCl_l770_770021


namespace train_pass_man_time_l770_770073

/--
Prove that the train, moving at 120 kmph, passes a man running at 10 kmph in the opposite direction in approximately 13.85 seconds, given the train is 500 meters long.
-/
theorem train_pass_man_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) : 
  length_of_train = 500 →
  speed_of_train = 120 →
  speed_of_man = 10 →
  abs ((500 / ((speed_of_train + speed_of_man) * 1000 / 3600)) - 13.85) < 0.01 :=
by
  intro h1 h2 h3
  -- This is where the proof would go
  sorry

end train_pass_man_time_l770_770073


namespace sum_of_integers_is_106_l770_770788

theorem sum_of_integers_is_106 (n m : ℕ) 
  (h1: n * (n + 1) = 1320) 
  (h2: m * (m + 1) * (m + 2) = 1320) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 106 :=
  sorry

end sum_of_integers_is_106_l770_770788


namespace value_of_c_distinct_real_roots_l770_770559

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770559


namespace intersection_point_of_lines_is_correct_l770_770216

theorem intersection_point_of_lines_is_correct
  (m n : ℝ)
  (h1 : 2 * (-1 : ℝ) + 3 = m)
  (h2 : (-1 : ℝ) - 3 = n)
  (h3 : ∀ x y : ℝ, -2 * x + m = y ↔ x + 4 = y ) :
  ∃ (x y : ℝ), x = -1 ∧ y = 3 ∧ (-2 * x + m = y) ∧ (x + n = y) :=
by
  use [-1, 3]
  split
  · rfl
  split
  · rfl
  · split
    · exact h1
    · exact h2

end intersection_point_of_lines_is_correct_l770_770216


namespace problem_solution_l770_770805

theorem problem_solution :
  (-2: ℤ)^2004 + 3 * (-2: ℤ)^2003 = -2^2003 := 
by
  sorry

end problem_solution_l770_770805


namespace range_of_x_l770_770275

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, f x a ≥ 3 := 3) :
  (∀ x : ℝ, f x a ≤ 5 → 3 ≤ x ∧ x ≤ 8) :=
by 
  sorry

end range_of_x_l770_770275


namespace num_pairs_f_zero_l770_770978

def f (x : ℝ) : ℝ := Real.sin ((Real.pi / 3) * x)
def A : Set ℝ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem num_pairs_f_zero :
  let pairs := {s t | s ∈ A ∧ t ∈ A ∧ s ≠ t ∧ (f s * f t = 0)}
  pairs.to_finset.card = 13 := by
sorry

end num_pairs_f_zero_l770_770978


namespace equilateral_triangle_octagon_area_ratio_l770_770887

theorem equilateral_triangle_octagon_area_ratio
  (s_t s_o : ℝ)
  (h_triangle_area : (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2)) :
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by
  sorry

end equilateral_triangle_octagon_area_ratio_l770_770887


namespace find_w_l770_770300

theorem find_w (p q r u v w : ℝ) 
  (h1 : Polynomial.Coeff 3 = 1)
  (h2 : Polynomial.Coeff 2 = 5)
  (h3 : Polynomial.Coeff 1 = 6)
  (h4 : Polynomial.Coeff 0 = -8)
  (hroots1 : p + q + r = -5)
  (hroots2 : ∀x, Polynomial.eval x (Polynomial.X^3 + u*(Polynomial.X^2) + v*(Polynomial.X) + w) = 0 → x = p + q ∨ x = q + r ∨ x = r + p)
  : w = 8 :=
sorry

end find_w_l770_770300


namespace train_time_from_B_to_C_l770_770305

-- Define the problem parameters and conditions
variables (M : ℝ) (D_AB D_BC D_AC : ℝ)
variable (S : ℝ)

-- Assume the following conditions:
-- 1. The distance between stations A and B is M kilometers longer than the distance between stations B and C.
-- 2. The total distance between stations A and C in terms of M is 6M.
-- 3. The time it took to reach B from A is 7 hours.
-- 4. The speed of the train S is constant.

-- Definitions directly from the conditions
def distance_AB_eq : Prop := D_AB = D_BC + M
def distance_AC_eq : Prop := D_AC = 6 * M
def time_to_B_eq : Prop := D_AB = 7 * S
def speed_def : Prop := S = D_AB / 7

-- Statement of the proof problem
theorem train_time_from_B_to_C (h1 : distance_AB_eq M D_AB D_BC)
                                (h2 : distance_AC_eq M D_AC)
                                (h3 : time_to_B_eq M D_AB S)
                                (h4 : speed_def M D_AB S) :
  D_BC / S = 5 :=
sorry

end train_time_from_B_to_C_l770_770305


namespace benny_final_comic_books_l770_770088

-- Define the initial number of comic books
def initial_comic_books : ℕ := 22

-- Define the comic books sold (half of the initial)
def comic_books_sold : ℕ := initial_comic_books / 2

-- Define the comic books left after selling half
def comic_books_left_after_sale : ℕ := initial_comic_books - comic_books_sold

-- Define the number of comic books bought
def comic_books_bought : ℕ := 6

-- Define the final number of comic books
def final_comic_books : ℕ := comic_books_left_after_sale + comic_books_bought

-- Statement to prove that Benny has 17 comic books at the end
theorem benny_final_comic_books : final_comic_books = 17 := by
  sorry

end benny_final_comic_books_l770_770088


namespace Janelle_has_72_marbles_l770_770246

def JanelleMarbles : Prop :=
  ∀ (initialGreen initialBlue boughtBags marblesPerBag giftGreen giftBlue : ℕ),
    initialGreen = 26 →
    boughtBags = 6 →
    marblesPerBag = 10 →
    giftGreen = 6 →
    giftBlue = 8 →
    let totalMarbles := initialGreen + boughtBags * marblesPerBag in
    let finalTotal := totalMarbles - (giftGreen + giftBlue) in
    finalTotal = 72

theorem Janelle_has_72_marbles : JanelleMarbles :=
by
  intros
  sorry

end Janelle_has_72_marbles_l770_770246


namespace total_people_correct_l770_770744

-- Define the conditions given in the problem
def full_time_teachers := 80
def part_time_teachers := 5
def principal := 1
def vice_principals := 3
def librarians := 2
def guidance_counselors := 6
def other_staff := 25
def total_classes := 40
def avg_students_per_class := 25
def part_time_students := 250

-- Define the FTE conversions
def fte_part_time_teachers := part_time_teachers * 0.5
def total_fte_teachers := full_time_teachers + fte_part_time_teachers
def total_staff := principal + vice_principals + librarians + guidance_counselors + other_staff
def total_full_time_students := total_classes * avg_students_per_class
def fte_part_time_students := part_time_students * 0.5
def total_students := total_full_time_students + fte_part_time_students
def total_fte_people := total_fte_teachers + total_staff + total_students

-- Final adjusted total count accounting for part-time teachers as whole individuals
def total_people_oxford_high : Nat :=
  (total_fte_people + part_time_teachers - fte_part_time_teachers).toNat

-- The proof statement
theorem total_people_correct : total_people_oxford_high = 1247 := 
by
  sorry

end total_people_correct_l770_770744


namespace sum_of_cubes_mod_5_l770_770479

theorem sum_of_cubes_mod_5 :
  ( ∑ k in Finset.range 50, (k + 1)^3 ) % 5 = 0 :=
sorry

end sum_of_cubes_mod_5_l770_770479


namespace avg_speed_4_2_l770_770889

noncomputable def avg_speed_round_trip (D : ℝ) : ℝ :=
  let speed_up := 3
  let speed_down := 7
  let total_distance := 2 * D
  let total_time := D / speed_up + D / speed_down
  total_distance / total_time

theorem avg_speed_4_2 (D : ℝ) (hD : D > 0) : avg_speed_round_trip D = 4.2 := by
  sorry

end avg_speed_4_2_l770_770889


namespace solve_for_x_l770_770945

theorem solve_for_x (x : ℝ) (h : (∛(2 * x * real.sqrt (x^3)) = 6)) : 
  x = real.rpow 108 (2/5) :=
sorry

end solve_for_x_l770_770945


namespace tank_filling_l770_770372

theorem tank_filling (A_rate B_rate : ℚ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) :
  (1 / (A_rate - B_rate)) = 18 :=
by
  sorry

end tank_filling_l770_770372


namespace suff_but_not_nec_l770_770646

theorem suff_but_not_nec (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by {
  sorry
}

end suff_but_not_nec_l770_770646


namespace max_elements_in_A_l770_770152

def prime (p : ℕ) : Prop :=
  1 < p ∧ ∀ n ∣ p, n = 1 ∨ n = p

theorem max_elements_in_A (p n : ℕ) (A : Set (Fin n → Fin p)) (h_prime : prime p)
  (h_bounds : p ≥ n ∧ n ≥ 3)
  (h_condition : ∀ x y ∈ A, x ≠ y → ∃ k l m : Fin n, k ≠ l ∧ l ≠ m ∧ k ≠ m ∧ x k ≠ y k ∧ x l ≠ y l ∧ x m ≠ y m):
  ∃ B ⊆ A, B.card = p^(n-2) := sorry

end max_elements_in_A_l770_770152


namespace sum_of_cubes_mod_five_l770_770488

theorem sum_of_cubes_mod_five : 
  (∑ n in Finset.range 51, n^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_five_l770_770488


namespace total_money_is_twenty_l770_770627

-- Define Henry's initial money
def henry_initial_money : Nat := 5

-- Define the money Henry earned
def henry_earned_money : Nat := 2

-- Define Henry's total money
def henry_total_money : Nat := henry_initial_money + henry_earned_money

-- Define friend's money
def friend_money : Nat := 13

-- Define the total combined money
def total_combined_money : Nat := henry_total_money + friend_money

-- The main statement to prove
theorem total_money_is_twenty : total_combined_money = 20 := sorry

end total_money_is_twenty_l770_770627


namespace quadratic_distinct_roots_l770_770551

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770551


namespace steve_speed_back_l770_770772

theorem steve_speed_back :
  ∀ (d v_total : ℕ), d = 10 → v_total = 6 →
  (2 * (15 / 6)) = 5 :=
by
  intros d v_total d_eq v_total_eq
  sorry

end steve_speed_back_l770_770772


namespace point_not_on_graph_l770_770081

theorem point_not_on_graph (x y : ℝ) : 
  (x = 1 ∧ y = 5 → y ≠ 6 / x) :=
by
  intros h
  cases h with hx hy
  rw [hx, hy]
  norm_num
  exact ne_of_gt (by norm_num)
  sorry

end point_not_on_graph_l770_770081


namespace leg_length_of_isosceles_right_triangle_l770_770778

theorem leg_length_of_isosceles_right_triangle (median_length : ℝ) 
    (h_median : median_length = 8) : 
    ∃ (leg_length : ℝ), leg_length = 8 * Real.sqrt 2 :=
by
  have hypotenuse_length : ℝ := 2 * median_length
  have h_hypotenuse : hypotenuse_length = 16 := by
    rw [←h_median]
    norm_num
  use 8 * Real.sqrt 2
  sorry

end leg_length_of_isosceles_right_triangle_l770_770778


namespace abs_simplification_l770_770198

theorem abs_simplification (x : ℝ) (hx : x < 0) : 
  |x - 3 * real.sqrt ((x - 2) ^ 2)| = 6 - 4 * x :=
by
  sorry

end abs_simplification_l770_770198


namespace base_seven_to_base_ten_l770_770845

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770845


namespace angle_OBC_l770_770679

theorem angle_OBC (OE_parallel_CD : OE ∥ CD)
  (BC_parallel_EA : BC ∥ EA)
  (angle_BCD : ∠BCD = 4 * ∠OBC)
  (angle_AEO : ∠AEO = 4 * ∠OBC)
  (OA_eq_OE : OA = OE)
  (OAE_isosceles : is_isosceles_triangle OA OE)
  (angle_BOA : ∠BOA = 90°):
  ∠OBC = 18° :=
by
  sorry

end angle_OBC_l770_770679


namespace vertex_of_parabola_l770_770946

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 + 10 * y + 2 * x + 9 = 0

-- Define the vertex of the parabola
def is_vertex (x y : ℝ) : Prop := x = 8 ∧ y = -5

-- The theorem we need to prove
theorem vertex_of_parabola :
  ∃ (x y : ℝ), parabola x y ∧ is_vertex x y :=
by
  use [8, -5]
  split
  { unfold parabola
    norm_num
  }
  { unfold is_vertex
    norm_num
  }

end vertex_of_parabola_l770_770946


namespace annual_interest_rate_l770_770910

theorem annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) 
  (hP : P = 5000) 
  (hA : A = 5202) 
  (hn : n = 4) 
  (ht : t = 1 / 2)
  (compound_interest : A = P * (1 + r / n)^ (n * t)) : 
  r = 0.080392 :=
by
  sorry

end annual_interest_rate_l770_770910


namespace newspaper_cost_over_8_weeks_l770_770628

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l770_770628


namespace max_lcm_15_2_3_5_6_9_10_l770_770356

theorem max_lcm_15_2_3_5_6_9_10 : 
  max (max (max (max (max (Nat.lcm 15 2) (Nat.lcm 15 3)) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10) = 45 :=
by
  sorry

end max_lcm_15_2_3_5_6_9_10_l770_770356


namespace tomatoes_picked_second_week_l770_770897

-- Define the constants
def initial_tomatoes : Nat := 100
def fraction_picked_first_week : Nat := 1 / 4
def remaining_tomatoes : Nat := 15

-- Theorem to prove the number of tomatoes Jane picked in the second week
theorem tomatoes_picked_second_week (x : Nat) :
  let T := initial_tomatoes
  let p := fraction_picked_first_week
  let r := remaining_tomatoes
  let first_week_pick := T * p
  let remaining_after_first := T - first_week_pick
  let total_picked := remaining_after_first - r
  let second_week_pick := total_picked / 3
  second_week_pick = 20 := 
sorry

end tomatoes_picked_second_week_l770_770897


namespace sqrt_fourth_root_of_000001_is_0_to_nearest_tenth_l770_770925

noncomputable def nested_root_calculation : ℝ :=
  Float.round (Real.sqrt (Real.sqrt (10 ^ (-6 / 4, 80 / 4))))

theorem sqrt_fourth_root_of_000001_is_0_to_nearest_tenth :
  Float.roundWith nested_root_calculation = 0.0 :=
 by sorry

end sqrt_fourth_root_of_000001_is_0_to_nearest_tenth_l770_770925


namespace correct_statement_is_a_l770_770858

-- Definitions based on given conditions:
def is_regular_pyramid (pyramid : Type) : Prop :=
  ∃ (base : Type) (lateral_face : Type), is_regular_polygon base ∧ is_equilateral_triangle lateral_face ∧ has_base pyramid base ∧ has_lateral_faces pyramid lateral_face

def is_regular_polygon (polygon : Type) : Prop := -- definition placeholder
sorry

def is_equilateral_triangle (triangle : Type) : Prop := -- definition placeholder
sorry

def has_base (structure : Type) (base : Type) : Prop := -- definition placeholder
sorry

def has_lateral_faces (structure : Type) (face : Type) : Prop := -- definition placeholder
sorry

def is_frustum (polyhedron : Type) : Prop := -- definition placeholder
sorry

def are_similar_polygons (poly1 poly2 : Type) : Prop := -- definition placeholder
sorry

def are_trapezoids (faces : Type) : Prop := -- definition placeholder
sorry

def is_right_prism (prism : Type) : Prop := -- definition placeholder
sorry

def are_squares (faces : Type) : Prop := -- definition placeholder
sorry

def is_cuboid (prism : Type) : Prop := -- definition placeholder
sorry

def are_congruent_rectangles (faces1 faces2 : Type) : Prop := -- definition placeholder
sorry

-- The correct statement among A, B, C, and D:
theorem correct_statement_is_a :
  is_regular_pyramid P → ¬is_right_prism R ∧ ¬is_cuboid C ∧ ¬is_frustum F :=
sorry

end correct_statement_is_a_l770_770858


namespace slope_MN_eq_2_l770_770331

-- Define the two points M and N
def M : ℝ × ℝ := (-2, 1)
def N : ℝ × ℝ := (-1, 3)

-- Define the slope formula
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- The theorem statement: The slope of the line passing through points M and N is 2
theorem slope_MN_eq_2 : slope M N = 2 := by
  sorry

end slope_MN_eq_2_l770_770331


namespace max_value_of_expression_l770_770449

theorem max_value_of_expression (x : ℝ) (h : 0 ≤ x ∧ x ≤ 9) :
  (∃ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ ≤ 9 ∧ (sqrt (x₀ + 15) + sqrt (9 - x₀) + sqrt (2 * x₀)) = sqrt 143) :=
by
sorry

end max_value_of_expression_l770_770449


namespace find_principal_amount_l770_770403

variable (P : ℝ)

def interestA_to_B (P : ℝ) : ℝ := P * 0.10 * 3
def interestB_from_C (P : ℝ) : ℝ := P * 0.115 * 3
def gain_B (P : ℝ) : ℝ := interestB_from_C P - interestA_to_B P

theorem find_principal_amount (h : gain_B P = 45) : P = 1000 := by
  sorry

end find_principal_amount_l770_770403


namespace max_omega_l770_770614

theorem max_omega (ω : ℝ) (φ : ℝ) (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = Real.sin (ω * x + φ))
  (h₂ : ω > 0)
  (h₃ : |φ| ≤ Real.pi / 2)
  (h₄ : f (-Real.pi / 4) = 0)
  (h₅ : ∃ a, ∀ x, f (Real.pi / 4 + x) = f (Real.pi / 4 - x))
  (h₆ : ∀ x y, (Real.pi / 18 < x) → (x < y ∧ y < 5 * Real.pi / 36) → f x ≤ f y) :
  ω = 9 := 
sorry

end max_omega_l770_770614


namespace part_a_part_b_l770_770579

variable (A B C G M : Type)

-- Definition of the centroid and its properties
def is_centroid (A B C G : Type) := -- Insert detailed definition of centroid here
sorry

-- Define squared distances
def squared_distance (P Q : Type) : Float := sorry

-- Given
variable (k : Float)
axiom h_centroid : is_centroid A B C G

axiom h_sq_dist_G : squared_distance G A + squared_distance G B + squared_distance G C = g
axiom h_k_gt_g : k > g

theorem part_a (M : Type) (h1 : is_centroid A B C G) :
  squared_distance M A + squared_distance M B + squared_distance M C
  ≥ squared_distance G A + squared_distance G B + squared_distance G C :=
sorry

theorem part_b (M : Type) (h1 : is_centroid A B C G) (h2 : squared_distance G A + squared_distance G B + squared_distance G C = g) (h3 : k > g) :
  squared_distance M A + squared_distance M B + squared_distance M C = k ↔
  ∃ r : Float, r = sqrt ((k - g) / 3) ∧ squared_distance M G = r :=
sorry

end part_a_part_b_l770_770579


namespace second_number_value_l770_770386

def first_number := ℚ
def second_number := ℚ

variables (x y : ℚ)

/-- Given conditions: 
      (1) \( \frac{1}{5}x = \frac{5}{8}y \)
      (2) \( x + 35 = 4y \)
    Prove that \( y = 40 \) 
-/
theorem second_number_value (h1 : (1/5 : ℚ) * x = (5/8 : ℚ) * y) (h2 : x + 35 = 4 * y) : 
  y = 40 :=
sorry

end second_number_value_l770_770386


namespace yz_plane_circle_radius_l770_770894

theorem yz_plane_circle_radius :
  let sphere_center := (3, 5, -8) in
  let plane1_circle_center := (3, 5, 0) in
  let plane1_circle_radius := 2 in
  let plane2_circle_center := (0, 5, -8) in
  ∃ (R : ℝ), (R = Real.sqrt (plane1_circle_radius^2 + (0 - (-8))^2)) ∧
    (Real.sqrt (R^2 - (sphere_center.1 - plane2_circle_center.1)^2) = Real.sqrt 59) :=
begin
  sorry,
end

end yz_plane_circle_radius_l770_770894


namespace sin_angle_CKL_l770_770777

theorem sin_angle_CKL (ABC : Triangle) (r : ℝ) (h1 : r = 2) (D : Point) (h2 : incircle_touches_AC_at D)
    (C : Angle) (h3 : C = arcsin (sqrt 15 / 8)) (K L : Point)
    (h4 : on_extension (AC \ C) K)
    (h5 : on_extension (BC \ C) L)
    (h6 : length_segment AK = semiperimeter ABC)
    (h7 : length_segment BL = semiperimeter ABC)
    (M : Point) (h8 : on_circumcircle ABC M) (h9 : parallel (CM) (KL))
    (h10 : length_segment DM = 4) : sin (angle CKL) = 0.5 := sorry

end sin_angle_CKL_l770_770777


namespace find_a_b_min_l770_770616

def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_a_b_min (a b : ℝ) :
  (∃ a b, f 1 a b = 10 ∧ deriv (f · a b) 1 = 0) →
  a = 4 ∧ b = -11 ∧ ∀ x ∈ Set.Icc (-4:ℝ) 3, f x a b ≥ f 1 4 (-11) := 
by
  -- Skipping the proof
  sorry

end find_a_b_min_l770_770616


namespace roots_of_quartic_eq_l770_770496

theorem roots_of_quartic_eq :
  (∀ x : ℝ, ((x^2 - 5 * x + 6) * (x - 3) * (x + 2)) = 0 →
   x = 2 ∨ x = 3 ∨ x = -2) ∧
  (∀ x : ℝ, x = 2 ∨ x = 3 ∨ x = -2 →
   ((x^2 - 5 * x + 6) * (x - 3) * (x + 2)) = 0) :=
by {
  sorry,
  
  sorry
}

end roots_of_quartic_eq_l770_770496


namespace max_attempts_l770_770806

inductive Key
| useful
| useless

def keys_sequence : List Key :=
  [Key.useless, Key.useless, Key.useless, Key.useless, Key.useful]

def attempts (keys : List Key) : Nat :=
  keys.findIdx (fun k => k = Key.useful) + 1

theorem max_attempts {keys : List Key} (h : keys = keys_sequence) : attempts keys = 5 := by
  rw [h]
  simp [keys_sequence, attempts]
  sorry

end max_attempts_l770_770806


namespace quadratic_distinct_roots_l770_770547

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770547


namespace find_dot_product_2a_minus_b_a_find_norm_a_plus_2b_l770_770167

open Real

variables {α : Type*} [InnerProductSpace ℝ α]

-- Given conditions
def angle_a_b : Real := 120 * π / 180

def a : α
def b : α

axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 1
axiom inner_a_b : ⟪a, b⟫ = 2 * 1 * cos angle_a_b

-- Proof of question 1
theorem find_dot_product_2a_minus_b_a : (2 • a - b) ⬝ a = 9 :=
by
  sorry

-- Proof of question 2
theorem find_norm_a_plus_2b : ∥a + 2 • b∥ = 2 :=
by
  sorry

end find_dot_product_2a_minus_b_a_find_norm_a_plus_2b_l770_770167


namespace smallest_integer_in_set_l770_770665

theorem smallest_integer_in_set (n : ℤ) (h : n + 5 < 2 * (n + 2.5) - 3) : n >= 2 :=
by
  suffices 1.5 < n by
    linarith
  sorry

end smallest_integer_in_set_l770_770665


namespace semicircle_radius_correct_l770_770376

noncomputable def semicircle_radius (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem semicircle_radius_correct (h :127 =113): semicircle_radius 113 = 113 / (Real.pi + 2) :=
by
  sorry

end semicircle_radius_correct_l770_770376


namespace min_nickels_for_sneakers_l770_770094

theorem min_nickels_for_sneakers (n : ℕ) : 
  (let total := 40 + 1.25 + 0.05 * n in total ≥ 45.50) → n ≥ 85 :=
by
  sorry

end min_nickels_for_sneakers_l770_770094


namespace function_property_l770_770598

def f (x : ℝ) : ℝ := sorry

theorem function_property (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy: x ≠ y) :
  f(x - y) + f(y - x) = 0 ∧ 
  (x - y) * (f x - f y) < 0 ∧ 
  f((x + y) / 2) > (f x + f y) / 2 ↔ 
  f = (λ x, 1 / x) :=
sorry

end function_property_l770_770598


namespace base_seven_to_base_ten_l770_770842

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l770_770842


namespace bisector_BE_A_MR_perpendicular_l770_770724

variables {A B C P M R E : Type*} [AffineSpace A] [AffineSpace B] [AffineSpace C]
  [AffineSpace P] [AffineSpace M] [AffineSpace R] [AffineSpace E]

-- Definitions and Conditions
variable (triangle : Prop := ∃ P, P ∈ (interior_triangle A B C))
variable (condition1 : Prop := BP = AC)
variable (midpoint_M : Prop := midpoint (line_segment A P) M)
variable (midpoint_R : Prop := midpoint (line_segment B C) R)
variable (intersection_E : Prop := (line BP) ⊓ (line AC) = E)

-- To Prove
theorem bisector_BE_A_MR_perpendicular :
  triangle →
  condition1 →
  midpoint_M →
  midpoint_R →
  intersection_E →
  (is_angle_bisector B E A → perpendicular_line E (line_segment M R)) :=
by
  sorry

end bisector_BE_A_MR_perpendicular_l770_770724


namespace in_circle_tangent_distance_l770_770877

open Real -- opening the realm of real numbers

def is_inscribed (A B C : Point) (circle : Circle) : Prop :=
  circle.tangent_on A B ∧ circle.tangent_on B C ∧ circle.tangent_on C A

def semiperimeter {α : Type} [MetricSpace α] (A B C : α) : ℝ :=
  (dist A B + dist B C + dist C A) / 2

theorem in_circle_tangent_distance
  (A B C : Point) -- vertices of the triangle
  (circle : Circle) -- the inscribed circle
  (h_inscribed : is_inscribed A B C circle) -- the circle is inscribed
  (x : ℝ) -- the distance from vertex A to the point where the circle touches AB
  (BC a : ℝ) -- side BC is a
  (h_bc : dist B C = a) -- BC is equal to a
  (p : ℝ) -- the semiperimeter of the triangle
  (h_p : semiperimeter A B C = p) -- p is the semiperimeter
  : x = p - a := 
sorry -- proof is to be filled in

end in_circle_tangent_distance_l770_770877


namespace find_a1_l770_770999

variable (a : ℕ → ℚ) (d : ℚ)
variable (S : ℕ → ℚ)
variable (h_seq : ∀ n, a (n + 1) = a n + d)
variable (h_diff : d ≠ 0)
variable (h_prod : (a 2) * (a 3) = (a 4) * (a 5))
variable (h_sum : S 4 = 27)
variable (h_sum_def : ∀ n, S n = n * (a 1 + a n) / 2)

theorem find_a1 : a 1 = 135 / 8 := by
  sorry

end find_a1_l770_770999


namespace K_time_correct_l770_770380

variable (x : ℝ) (K_time M_time : ℝ)

-- Conditions
def K_speed := x
def M_speed := x - 0.5
def K_time := 40 / K_speed
def M_time := 40 / M_speed

-- Time difference condition
def time_difference := M_time - K_time = 0.5

-- Statement to prove
theorem K_time_correct :
  time_difference → K_time = 40 / x :=
sorry

end K_time_correct_l770_770380


namespace quadratic_roots_condition_l770_770512

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770512


namespace exists_constants_for_sum_l770_770105

theorem exists_constants_for_sum :
  ∃ (a b c : ℝ), (a = 3 ∧ b = 11 ∧ c = 10) ∧
  (∀ n : ℕ, 1 * 2^2 + 2 * 3^2 + ∑ i in finset.range n, (i+1) * (i+2)^2 =
    (n * (n + 1) / 12) * (a * n^2 + b * n + c)) :=
by
  use 3, 11, 10
  split
  repeat { norm_num }
  intro n
  sorry

end exists_constants_for_sum_l770_770105


namespace max_blue_points_l770_770229

theorem max_blue_points (n : ℕ) (h_n : n = 2016) :
  ∃ r : ℕ, r * (2016 - r) = 1008 * 1008 :=
by {
  sorry
}

end max_blue_points_l770_770229


namespace probability_of_a_greater_than_b_l770_770893

def problem_statement : Prop :=
  let S1 := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let S2 := {1, 2, 3, 4, 5, 6, 7, 8}
  let choices1 := (set.powersetLen 3 S1).toFinset
  let choices2 := (set.powersetLen 3 S2).toFinset
  let p1 := (∑ s1 in choices1, if ∃ a b c ∈ s1, a > b ∧ b > c then 1 else 0) / (choices1.card : ℝ)
  let p2 := (∑ s2 in choices2, if ∃ d e f ∈ s2, d > e ∧ e > f then 1 else 0) / (choices2.card : ℝ)
  (p1 > p2) = (37 / 56)

theorem probability_of_a_greater_than_b : problem_statement := by sorry

end probability_of_a_greater_than_b_l770_770893


namespace range_of_t_in_triangle_l770_770659

noncomputable def triangle_AB_BC_conditions (A B C : Type) [inner_product_space ℝ (Type)] :=
  ∃ (AB AC BC : ℝ) (AB_vec BC_vec AC_vec : (Type) → ℝ),
    AB = sqrt 3 ∧
    BC = 2 ∧
    AC = 1 ∧
    AB_vec = BA ∧
    BC_vec = BC ∧
    AC_vec = AC ∧
    (norm (AB_vec - (t : ℝ) • BC_vec) ≤ (sqrt 3 * norm AC_vec))

theorem range_of_t_in_triangle (A B C : Type) [inner_product_space ℝ (Type)] :
  triangle_AB_BC_conditions A B C → (0 ≤ t ∧ t ≤ 3/2) := 
sorry

end range_of_t_in_triangle_l770_770659


namespace reasoning_type_l770_770241

-- Definitions
def inductive_reasoning := "reasoning from specific instances to a general conclusion"
def deductive_reasoning := "reasoning from a general premise to a specific conclusion"
def analogical_reasoning := "reasoning from one specific instance to another specific instance"

-- Statement to be proven
theorem reasoning_type :
  (∀ (a b c : ℝ), (a > b) → (a + c > b + c) → (a > b) → (a * c > b * c)) →
  analogical_reasoning :=
sorry

end reasoning_type_l770_770241


namespace period_of_f_is_4ba_l770_770654

-- Assume the conditions as mentioned.
variables {f : ℝ → ℝ}
variables {a b : ℝ}
variables (h1 : function.symmetric (λ x, f (2*x)) (λ x, x - a/2))
variables (h2 : function.symmetric (λ x, f (2*x)) (λ x, x - b/2))
variables (hb_gt_ha : b > a)

-- Lean statement to show the period of f is 4*(b - a).
theorem period_of_f_is_4ba :
  ∃ (P : ℝ), P = 4 * (b - a) ∧ ∀ x : ℝ, f(2 * (x + P))= f(2 * x) :=
sorry

end period_of_f_is_4ba_l770_770654


namespace vector_properties_l770_770714

-- Conditions on vectors a and b
variables (a b : ℝ^3)
variable (t : ℝ)

-- Assume |b| = 1 and a ≠ b
axiom norm_b : ∥b∥ = 1
axiom a_ne_b : a ≠ b

-- Assume for any t in ℝ, |a - t * b| ≥ |a - b|
axiom distance_inequality : ∀ t : ℝ, ∥a - t • b∥ ≥ ∥a - b∥

-- Prove that a ⋅ b = 1 and b is orthogonal to (a - b)
theorem vector_properties : a • b = 1 ∧ b • (a - b) = 0 :=
by sorry

end vector_properties_l770_770714


namespace distribute_books_l770_770335

theorem distribute_books : 
  let B : Finset ℕ := {1, 2, 3, 4, 5} in
  let P : Finset ℕ := {1, 2, 3} in
  (∃ (a b : Finset ℕ) (pa pb pk : P),
    a ≠ b ∧ a.card = 2 ∧ b.card = 2 ∧ (B \ (a ∪ b)).card = 1 ∧ 
    (po_comm : pa ≠ pb ∧ pb ≠ pk ∧ pk ≠ pa) ∧ 
    Finset.card_univ_subtype (Finset.cons pa (Finset.cons pb (Finset.mk [pk] sorry)) sorry) = P.card) →
  (∃ (d : ℕ), d = 90) := 
by
  sorry

end distribute_books_l770_770335


namespace player_A_winning_strategy_l770_770870

theorem player_A_winning_strategy :
  ∃ strategy : -- define type of strategy,
  ∀ (board : -- define type of the board, 
    starting_position : -- define type of starting position iff necessary), 
    player : ℕ,  
    (∀ move_n, move_of_A (board.starting_position)
  sorry

end player_A_winning_strategy_l770_770870


namespace flour_qualification_l770_770902

def acceptable_weight_range := {w : ℝ | 24.75 ≤ w ∧ w ≤ 25.25}

theorem flour_qualification :
  (24.80 ∈ acceptable_weight_range) ∧ 
  (24.70 ∉ acceptable_weight_range) ∧ 
  (25.30 ∉ acceptable_weight_range) ∧ 
  (25.51 ∉ acceptable_weight_range) :=
by 
  -- The proof would go here, but we are adding sorry to skip it.
  sorry

end flour_qualification_l770_770902


namespace range_g_l770_770450

def g (A : ℝ) : ℝ :=
  (Real.cos A * (5 * (Real.sin A)^2 + 2 * (Real.cos A)^4 + 4 * (Real.sin A)^2 * (Real.cos A)^2)) /
  ((Real.cos A / Real.sin A) * (1 / Real.sin A - (Real.cos A)^2 / Real.sin A))

theorem range_g (A : ℝ)
  (h : ∀ k : ℤ, A ≠ k * Real.pi + Real.pi / 3) : 
  ∃ I : Set ℝ, I = Set.Icc 2 5 ∧ (g '' Set.univ) = I :=
by
  sorry

end range_g_l770_770450


namespace square_triangle_DH_l770_770666
open Classical

theorem square_triangle_DH {ABCD : Type*} [square ABCD] (side_length : ℝ) (G : ABCD) (AD GH : ℝ) (H : ABCD) :
  (side_length = 20) ∧ (point_on_side G AD) ∧ (perpendicular DG GH) ∧ (area_triangle DGH = 170) →
  DH = 2 * sqrt 170 :=
by
  sorry

end square_triangle_DH_l770_770666


namespace m_condition_sufficient_not_necessary_l770_770103

-- Define the function f(x) and its properties
def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

-- Define the condition for the function being increasing on (0, ∞)
def is_increasing_on_positives (m : ℝ) :=
  ∀ x y : ℝ, 0 < x → x < y → f m x < f m y

-- Prove that if m > 0, then the function is increasing on (0, ∞)
lemma m_gt_0_sufficient (m : ℝ) (h : 0 < m) : is_increasing_on_positives m :=
sorry

-- Show that the condition is indeed sufficient but not necessary
theorem m_condition_sufficient_not_necessary :
  ∀ m : ℝ, (0 < m → is_increasing_on_positives m) ∧ (is_increasing_on_positives m → 0 < m) :=
sorry

end m_condition_sufficient_not_necessary_l770_770103


namespace initial_kittens_l770_770078

theorem initial_kittens (kittens_given : ℕ) (kittens_left : ℕ) (initial_kittens : ℕ) :
  kittens_given = 4 → kittens_left = 4 → initial_kittens = kittens_given + kittens_left → initial_kittens = 8 :=
by
  intros hg hl hi
  rw [hg, hl] at hi
  -- Skipping proof detail
  sorry

end initial_kittens_l770_770078


namespace inequality_proof_l770_770256

theorem inequality_proof (n : ℕ) (h : n ≥ 3) (x : Fin n → ℝ) (h0 : ∀ i, 0 ≤ x i) (h1 : ∀ i, x i ≤ 1) :
  (∑ i, x i) - ∑ i, x i * x ((i + 1) % n) ≤ ⌊n / 2⌋ :=
by
  apply sorry

end inequality_proof_l770_770256


namespace Kyle_age_l770_770758

-- Let's define the variables for each person's age.
variables (Shelley Kyle Julian Frederick Tyson Casey Sandra David Fiona : ℕ) 

-- Defining conditions based on given problem.
axiom condition1 : Shelley = Kyle - 3
axiom condition2 : Shelley = Julian + 4
axiom condition3 : Julian = Frederick - 20
axiom condition4 : Julian = Fiona + 5
axiom condition5 : Frederick = 2 * Tyson
axiom condition6 : Tyson = 2 * Casey
axiom condition7 : Casey = Fiona - 2
axiom condition8 : Casey = Sandra / 2
axiom condition9 : Sandra = David + 4
axiom condition10 : David = 16

-- The goal is to prove Kyle's age is 23 years old.
theorem Kyle_age : Kyle = 23 :=
by sorry

end Kyle_age_l770_770758


namespace find_integers_l770_770189

theorem find_integers (x y : ℕ) (d : ℕ) (x1 y1 : ℕ) 
  (hx1 : x = d * x1) (hy1 : y = d * y1)
  (hgcd : Nat.gcd x y = d)
  (hcoprime : Nat.gcd x1 y1 = 1)
  (h1 : x1 + y1 = 18)
  (h2 : d * x1 * y1 = 975) : 
  ∃ (x y : ℕ), (Nat.gcd x y > 0) ∧ (x / Nat.gcd x y + y / Nat.gcd x y = 18) ∧ (Nat.lcm x y = 975) :=
sorry

end find_integers_l770_770189


namespace sequence_terms_sequence_sum_l770_770618

def sequence (n : ℕ) : ℚ := 1 / (List.range (n + 1)).sum

theorem sequence_terms :
  sequence 1 = 1 ∧ sequence 2 = 1 / 3 ∧ sequence 3 = 1 / 6 :=
by
  sorry

theorem sequence_sum (n : ℕ) :
  (Finset.range n).sum (λ k, sequence (k+1)) = (2 * n) / (n + 1) :=
by
  sorry

end sequence_terms_sequence_sum_l770_770618


namespace tan_x_value_l770_770997

theorem tan_x_value (x : ℝ) 
  (h1 : cos (2 * x) / (real.sqrt 2 * cos (x + π / 4)) = 1 / 5)
  (h2 : 0 < x) (h3 : x < π) : tan x = -4 / 3 :=
by
  sorry

end tan_x_value_l770_770997


namespace proof_problem_l770_770615

def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ :=
  √3 * sin(ω * x + ϕ) + 2 * sin((ω * x + ϕ) / 2) ^ 2 - 1

theorem proof_problem
  (ω : ℝ)
  (ϕ : ℝ)
  (hω : ω > 0)
  (hϕ1 : 0 < ϕ)
  (hϕ2 : ϕ < π)
  (hx : (x : ℝ) -> f(x, ω, ϕ)) :
  (f(x, ω, ϕ) = 2 * sin(2 * x)) ∧ 
  (∀ k : ℤ, ∃ x : ℝ, (f(x, ω, ϕ) = 2 * sin(2 * x)) ∧ (f x, [π / 4 + k * π, 3π / 4 + k * π])) ∧ 
  (2 * f(x, ω, ϕ) ^ 2 + √3 * f(x, ω, ϕ) - 3 = 0 → x ∈ [-π / 6, 5π / 6] → x.sum = 11π / 6) :=
sorry

end proof_problem_l770_770615


namespace total_songs_sung_l770_770247

def total_minutes := 80
def intermission_minutes := 10
def long_song_minutes := 10
def short_song_minutes := 5

theorem total_songs_sung : 
  (total_minutes - intermission_minutes - long_song_minutes) / short_song_minutes + 1 = 13 := 
by 
  sorry

end total_songs_sung_l770_770247


namespace exists_three_with_gcd_d_l770_770974

theorem exists_three_with_gcd_d (n : ℕ) (nums : Fin n.succ → ℕ) (d : ℕ)
  (h1 : n ≥ 2)  -- because n+1 (number of elements nums : Fin n.succ) ≥ 3 given that n ≥ 2
  (h2 : ∀ i, nums i > 0) 
  (h3 : ∀ i, nums i ≤ 100) 
  (h4 : Nat.gcd (nums 0) (Nat.gcd (nums 1) (nums 2)) = d) : 
  ∃ i j k : Fin n.succ, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ Nat.gcd (nums i) (Nat.gcd (nums j) (nums k)) = d :=
by
  sorry

end exists_three_with_gcd_d_l770_770974


namespace lines_coplanar_final_l770_770254

noncomputable theory

variables (O A B C D E F : Type) [Point O] [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variables (ω : Sphere)
variables (l m n : Line)

-- Definitions of points A1, B1, C1, D1, E1, F1
variables (A1 B1 C1 D1 E1 F1 : Point)
variable (plane_through_points : (Point) -> (Point) -> (Point) -> Plane)

-- Definitions of lines A1D1, B1E1, C1F1
variables (A1D1 : Line) (B1E1 : Line) (C1F1 : Line)
variables (hexagon_pyramid : Pyramid O A B C D E F) (circumscribed_around : Pyramid O A B C D E F -> Sphere -> Prop)

-- Assumptions from problem conditions
axiom a1_def : plane_through_points (touching_point ω (face OFA)) (touching_point ω (face OAB)) (touching_point ω (faceABCDEF)) ∩ OA = A1
-- Similarly define B1, C1, D1, E1, F1...

axiom pyramid_circumscribed: circumscribed_around hexagon_pyramid ω

axiom lines_def : 
  A1D1 = line_through A1 D1 ∧
  B1E1 = line_through B1 E1 ∧
  C1F1 = line_through C1 F1 ∧
  l = A1D1 ∧
  m = B1E1 ∧
  n = C1F1

axiom lines_coplanar : 
  is_coplanar l m ∧
  is_coplanar m n

-- Final result based on conclusions.
theorem lines_coplanar_final : is_coplanar l n :=
sorry

end lines_coplanar_final_l770_770254


namespace unique_function_exists_and_smallest_q_l770_770046

noncomputable def f : ℚ+ → ℚ+ := sorry

theorem unique_function_exists_and_smallest_q 
: (∃! (f : ℚ+ → ℚ+), 
    (∀ q : ℚ+, 0 < q ∧ q < 1/2 → f(q) = 1 + f(q / (1 - 2*q))) ∧
    (∀ q : ℚ+, 1 < q ∧ q ≤ 2 → f(q) = 1 + f(q + 1)) ∧
    (∀ q : ℚ+, f(q) * f(1/q) = 1)) ∧
   (∃ q : ℚ+, f(q) = 19/92 ∧ ∀ r : ℚ+, f(r) = 19/92 → q ≤ r) :=
begin
  sorry
end

end unique_function_exists_and_smallest_q_l770_770046


namespace no_solution_exists_l770_770138

theorem no_solution_exists (x y : ℝ) : ¬ ((2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9)) :=
sorry

end no_solution_exists_l770_770138


namespace log_power_eq_l770_770935

theorem log_power_eq (a b : ℝ) (h : a > 0) (h1 : a ≠ 1) (x : ℝ) : a^(Real.logb a b) = b :=
by
  sorry
  
example : 2^(Real.logb 2 9) = 9 := log_power_eq 2 9 (by norm_num) (by norm_num) (Real.logb 2 9)

end log_power_eq_l770_770935


namespace sqrt_fraction_l770_770108

theorem sqrt_fraction (a : ℝ) (h₁ : a = 7) : 
  (a^(1/4)) / (a^(1/7)) = a^(3 / 28) := 
by
  assume h₁
  have h2 : a = 7 := h₁
  sorry

end sqrt_fraction_l770_770108


namespace find_k_l770_770800

theorem find_k (S : ℕ → ℕ) (a : ℕ → ℕ) (h_eq : ∀ n, 3 * S n = a n * a (n + 1) + 1)
  (h_a1 : a 1 = 1) (h_neq0 : ∀ n, a n ≠ 0) 
  (h_ak : a k = 2018) : k = 1346 :=
begin
  sorry
end

end find_k_l770_770800


namespace largest_possible_n_l770_770307

theorem largest_possible_n :
  ∃ (m n : ℕ), (0 < m) ∧ (0 < n) ∧ (m + n = 10) ∧ (n = 9) :=
by
  sorry

end largest_possible_n_l770_770307


namespace solve_eq_l770_770763

open Real

noncomputable def solution : Set ℝ := { x | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) }

theorem solve_eq : { x : ℝ | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) } = solution := by sorry

end solve_eq_l770_770763


namespace relationship_m_n_k_l_l770_770664

-- Definitions based on the conditions
variables (m n k l : ℕ)

-- Condition: Number of teachers (m), Number of students (n)
-- Each teacher teaches exactly k students
-- Any pair of students has exactly l common teachers

theorem relationship_m_n_k_l (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : 0 < l)
  (hk : k * (k - 1) / 2 = k * (k - 1) / 2) (hl : n * (n - 1) / 2 = n * (n - 1) / 2) 
  (h5 : m * (k * (k - 1)) = (n * (n - 1)) * l) :
  m * k * (k - 1) = n * (n - 1) * l :=
by 
  sorry

end relationship_m_n_k_l_l770_770664


namespace purely_imaginary_complex_number_l770_770168

theorem purely_imaginary_complex_number (a : ℝ) 
  (h1 : (a^2 - 4 * a + 3 = 0))
  (h2 : a ≠ 1) 
  : a = 3 := 
sorry

end purely_imaginary_complex_number_l770_770168


namespace total_tickets_sold_l770_770339

-- Define the conditions
def children_ticket_cost := 6
def adult_ticket_cost := 9
def total_revenue := 1875
def adult_tickets_sold := 175

-- Define the final statement to be proven
theorem total_tickets_sold : 
  ∃ (C : ℕ), C * children_ticket_cost + adult_tickets_sold * adult_ticket_cost = total_revenue ∧
             C + adult_tickets_sold = 225 :=
begin
  sorry
end

end total_tickets_sold_l770_770339


namespace quadrilateral_area_l770_770601

noncomputable def area_of_quadrilateral
  (z1 z2 z3 z4 : ℂ)
  (h1 : z1^2 = 1 + 3 * Real.sqrt 10 * complex.I)
  (h2 : z2^2 = 1 + 3 * Real.sqrt 10 * complex.I)
  (h3 : z3^2 = 2 - 2 * Real.sqrt 2 * complex.I)
  (h4 : z4^2 = 2 - 2 * Real.sqrt 2 * complex.I)
  : ℝ :=
|Complex.imag ((Complex.conj z1) * z2 + (Complex.conj z2) * z3 + (Complex.conj z3) * z4 + (Complex.conj z4) * z1) / 2|

theorem quadrilateral_area :
  ∃ z1 z2 z3 z4 : ℂ,
    z1^2 = 1 + 3 * Real.sqrt 10 * complex.I ∧
    z2^2 = 1 + 3 * Real.sqrt 10 * complex.I ∧
    z3^2 = 2 - 2 * Real.sqrt 2 * complex.I ∧
    z4^2 = 2 - 2 * Real.sqrt 2 * complex.I ∧
    area_of_quadrilateral z1 z2 z3 z4 (by exact h1) (by exact h2) (by exact h3) (by exact h4) = 19 * Real.sqrt 6 - 2 * Real.sqrt 2 :=
begin
  sorry
end

end quadrilateral_area_l770_770601


namespace inequality_solution_l770_770332

def solutionSetInequality (x : ℝ) : Prop :=
  (x > 1 ∨ x < -2)

theorem inequality_solution (x : ℝ) : 
  (x+2)/(x-1) > 0 ↔ solutionSetInequality x := 
  sorry

end inequality_solution_l770_770332


namespace quadratic_has_distinct_real_roots_l770_770524

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770524


namespace tangent_line_at_2_eq_x_minus_4_tangent_lines_through_A_l770_770607

noncomputable def f (x: ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4
noncomputable def f' (x: ℝ) : ℝ := 3 * x^2 - 8 * x + 5

theorem tangent_line_at_2_eq_x_minus_4 :
  ∃ (m b : ℝ), m = f' 2 ∧ f 2 = -2 ∧ (∀ x y : ℝ, y = m * (x - 2) + f 2 → y = x - 4) :=
by
  exists (f' 2), (-4)
  split
  sorry
  split
  sorry
  intros
  sorry

theorem tangent_lines_through_A :
  ∃ (m1 b1 m2 b2 : ℝ), 
  (m1 = 1 ∧ b1 = -4 ∧ (∀ x y : ℝ, y = m1 * (x - 2) + (-2) → y = x - 4)) ∧ 
  (m2 = 0 ∧ b2 = -2 ∧ (∀ x y : ℝ, y = m2 * (x - 1) + (-2) → y = -2)) :=
by
  exists (f' 2), (-4), 0, (-2)
  split
  split
  sorry
  split
  sorry
  intros
  sorry
  split
  sorry
  split
  sorry
  intros
  sorry

end tangent_line_at_2_eq_x_minus_4_tangent_lines_through_A_l770_770607


namespace set_of_points_l770_770962

theorem set_of_points : {p : ℝ × ℝ | (2 * p.1 - p.2 = 1) ∧ (p.1 + 4 * p.2 = 5)} = { (1, 1) } :=
by
  sorry

end set_of_points_l770_770962


namespace race_problem_l770_770243

/-- Jack and Jill run a race covering 12 km in total, starting and finishing at the same point 
     after running 6 km up a hill and returning by the same route. 
     Jack starts 15 minutes before Jill and runs at 12 km/hr uphill and 18 km/hr downhill. 
     Jill runs at 14 km/hr uphill and 20 km/hr downhill. 
     Determine the distance from the top of the hill when they pass each other going in opposite directions (in km). -/
theorem race_problem : ∀ (dist : Real) (time_diff : Real) (jack_up_speed jack_down_speed : Real) 
  (jill_up_speed jill_down_speed : Real) (distance_from_top : Real),
  dist = 12 ∧ time_diff = 1 / 4 ∧ 
  jack_up_speed = 12 ∧ jack_down_speed = 18 ∧ 
  jill_up_speed = 14 ∧ jill_down_speed = 20 ∧ 
  distance_from_top = 6 - ((14 * (37 / 64 - 1 / 4)) / 2) ->
  distance_from_top = 45 / 32 :=
by
  intros dist time_diff jack_up_speed jack_down_speed jill_up_speed jill_down_speed distance_from_top h
  cases h with h_dist h_rest
  cases h_rest with h_time_diff h_other
  cases h_other with h_jack_up_speed h_other
  cases h_other with h_jack_down_speed h_other
  cases h_other with h_jill_up_speed h_other
  cases h_other with h_jill_down_speed h_distance_from_top
  simp only [*, eq_self_iff_true, and_self]
  sorry

end race_problem_l770_770243


namespace min_xy_l770_770199

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : xy = x + 4 * y + 5) : xy ≥ 25 :=
sorry

end min_xy_l770_770199


namespace number_of_solutions_l770_770133

theorem number_of_solutions (k : ℕ) : 
  ∃ (n : ℕ), n = k + 1 ∧ 
  (∃ solutions : fin n → (ℕ × ℕ × ℕ), 
    ∀ i, 
      let (x, y, z) := solutions i in 
        x ≤ y ∧ y ≤ z ∧ 8^k = x^3 + y^3 + z^3 - 3*x*y*z) :=
sorry

end number_of_solutions_l770_770133


namespace selling_price_calculation_l770_770406

-- Given conditions
def cost_price : ℚ := 110
def gain_percent : ℚ := 13.636363636363626

-- Theorem Statement
theorem selling_price_calculation : 
  (cost_price * (1 + gain_percent / 100)) = 125 :=
by
  sorry

end selling_price_calculation_l770_770406


namespace remainder_polynomial_l770_770957

variable (x : ℂ)

theorem remainder_polynomial :
  ∃ q : polynomial ℂ,
    polynomial.eval x x^6 - polynomial.eval x x^5 - polynomial.eval x x^4 +
    polynomial.eval x x^3 + polynomial.eval x x^2 =
    ((polynomial.eval x x^2 - 1) * (polynomial.eval x x - 2)) * q x +
    (9 * polynomial.eval x x^2 - 8) :=
by
  sorry

end remainder_polynomial_l770_770957


namespace divide_sum_eq_100_l770_770032

theorem divide_sum_eq_100 (x : ℕ) (h1 : 100 = 2 * x + (100 - 2 * x)) (h2 : (300 - 6 * x) + x = 100) : x = 40 :=
by
  sorry

end divide_sum_eq_100_l770_770032


namespace focus_is_correct_l770_770384

noncomputable def curve_focus_polar_coordinates : Prop :=
  ∀ (ρ θ : ℝ), (ρ * (cos θ)^2 = 4 * sin θ) → (ρ = 1 ∧ θ = π / 2)

theorem focus_is_correct :
  curve_focus_polar_coordinates :=
by 
  sorry

end focus_is_correct_l770_770384


namespace simplify_expr_correct_l770_770759

-- Define the expression
def simplify_expr (z : ℝ) : ℝ := (3 - 5 * z^2) - (5 + 7 * z^2)

-- Prove the simplified form
theorem simplify_expr_correct (z : ℝ) : simplify_expr z = -2 - 12 * z^2 := by
  sorry

end simplify_expr_correct_l770_770759


namespace length_BP_l770_770709

-- Definitions of the conditions
variables (ω : Circle) (A B C T P : Point)
variables (AB OC : ℝ)
variable (CT : Line)
variables [Circumference ω A] [Circumference ω B] [DiameterCenter A B ω]
variables [TangentLineCT ω C T]
variables [PerpendicularFoot A CT P]

-- Distances given in the conditions
def diameter_len : AB = 24 := sorry
def center_to_c : OC = 36 := sorry

-- The theorem to prove the length of segment BP
theorem length_BP (hx1 : AB = 24) (hx2 : OC = 36) : segment_len B P = 24 * sqrt 3 := sorry

end length_BP_l770_770709


namespace PQ_value_l770_770710

theorem PQ_value (DE DF EF : ℕ) (CF : ℝ) (P Q : ℝ) 
  (h1 : DE = 996)
  (h2 : DF = 995)
  (h3 : EF = 994)
  (hCF :  CF = (995^2 - 4) / 1990)
  (hP : P = (1492.5 - EF))
  (hQ : Q = (s - DF)) :
  PQ = 1 ∧ m + n = 2 :=
by
  sorry

end PQ_value_l770_770710


namespace ab_conditions_l770_770713

theorem ab_conditions (a b : ℝ) : ¬((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by 
  sorry

end ab_conditions_l770_770713


namespace square_area_from_diagonal_l770_770868

theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : (∃ A : ℝ, A = 392) :=
by
  sorry

end square_area_from_diagonal_l770_770868


namespace store_b_cheaper_for_50kg_l770_770396

def price_store_A (x : ℕ) : ℕ :=
  4 * x

def price_store_B (x : ℕ) : ℕ :=
  if x ≤ 6 then 5 * x else 3.5 * x + 9

theorem store_b_cheaper_for_50kg : price_store_B 50 = 184 :=
by trivial -- Lean will know that price_store_B 50 evaluates to 184

end store_b_cheaper_for_50kg_l770_770396


namespace survey_no_preference_students_l770_770048

theorem survey_no_preference_students (total_students pref_mac pref_both pref_windows : ℕ) 
    (h1 : total_students = 210) 
    (h2 : pref_mac = 60) 
    (h3 : pref_both = pref_mac / 3)
    (h4 : pref_windows = 40) : 
    total_students - (pref_mac + pref_both + pref_windows) = 90 :=
by
  sorry

end survey_no_preference_students_l770_770048


namespace quadratic_has_distinct_real_roots_l770_770527

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l770_770527


namespace reciprocal_square_sum_l770_770933

theorem reciprocal_square_sum :
  (let s := (1/4 : ℚ) + (1/6 : ℚ) in 1 / (s * s) = 144 / 25) :=
by
  let s := (1/4 : ℚ) + (1/6 : ℚ)
  show 1 / (s * s) = 144 / 25
  sorry

end reciprocal_square_sum_l770_770933


namespace sum_cubes_mod_5_l770_770462

-- Define the function sum of cubes up to n
def sum_cubes (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 1) ^ 3)

-- Define the given modulo operation
def modulo (a b : ℕ) : ℕ := a % b

-- The primary theorem to prove
theorem sum_cubes_mod_5 : modulo (sum_cubes 50) 5 = 0 := sorry

end sum_cubes_mod_5_l770_770462


namespace problem_A_problem_B_l770_770569

section complex_problems

open Complex

noncomputable def z1 : ℂ := (1 + 3 * I) / (1 - 3 * I)
noncomputable def conjugate_z1 := conj z1

theorem problem_A : conjugate_z1 = (-4 / 5) - (3 / 5) * I := by
  sorry

theorem problem_B (b : ℝ) (hb : b ≠ 0) : (b * I) ^ 2 < 0 := by
  sorry

end complex_problems

end problem_A_problem_B_l770_770569


namespace monotone_function_iteration_l770_770458

theorem monotone_function_iteration (f : ℝ → ℝ) (hf : Monotone f) :
  (∃ n : ℕ, ∀ x : ℝ, (nat.iterate f n x) = -x) → (∀ x : ℝ, f x = -x) :=
by
  sorry

end monotone_function_iteration_l770_770458


namespace concert_song_count_l770_770250

theorem concert_song_count (total_concert_time : ℕ)
  (intermission_time : ℕ)
  (song_duration_regular : ℕ)
  (song_duration_special : ℕ)
  (performance_time : ℕ)
  (total_songs : ℕ) :
  total_concert_time = 80 →
  intermission_time = 10 →
  song_duration_regular = 5 →
  song_duration_special = 10 →
  performance_time = total_concert_time - intermission_time →
  performance_time - song_duration_special = (total_songs - 1) * song_duration_regular →
  total_songs = 13 :=
begin
  intros h_total h_intermission h_regular h_special h_performance_time h_remaining_songs,
  sorry
end

end concert_song_count_l770_770250


namespace sum_cubes_mod_5_l770_770463

-- Define the function sum of cubes up to n
def sum_cubes (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 1) ^ 3)

-- Define the given modulo operation
def modulo (a b : ℕ) : ℕ := a % b

-- The primary theorem to prove
theorem sum_cubes_mod_5 : modulo (sum_cubes 50) 5 = 0 := sorry

end sum_cubes_mod_5_l770_770463


namespace amount_b_l770_770033

variable {a b : ℚ} -- a and b are rational numbers

theorem amount_b (h1 : a + b = 1210) (h2 : (4 / 15) * a = (2 / 5) * b) : b = 484 :=
sorry

end amount_b_l770_770033


namespace value_of_c_distinct_real_roots_l770_770561

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l770_770561


namespace guiding_normal_vector_l770_770120

noncomputable def ellipsoid (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 - 6

def point_M0 : ℝ × ℝ × ℝ := (1, -1, 1)

def normal_vector (x y z : ℝ) : ℝ × ℝ × ℝ := (
  2 * x,
  4 * y,
  6 * z
)

theorem guiding_normal_vector : normal_vector 1 (-1) 1 = (2, -4, 6) :=
by
  sorry

end guiding_normal_vector_l770_770120


namespace find_m_value_l770_770863

theorem find_m_value (m : ℤ) (hm : (-2)^(2 * m) = 2^(12 - m)) : m = 4 := 
by
  sorry

end find_m_value_l770_770863


namespace min_positive_phi_for_symmetry_l770_770212

def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem min_positive_phi_for_symmetry (φ : ℝ) (h : ∀ x, -2 * φ + (π / 4) = n * π ∨ 2 * φ - (π / 4) = -(n * π)) :
  φ = 3 * π / 8 := sorry

end min_positive_phi_for_symmetry_l770_770212


namespace volume_of_hemisphere_volume_of_space_between_cone_and_cylinder_ratio_of_volumes_l770_770026

-- 1. Volume of the Hemisphere
theorem volume_of_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

-- 2. Volume of the space between cone and cylinder
theorem volume_of_space_between_cone_and_cylinder (r h : ℝ) : ℝ :=
  let volume_cylinder := (Real.pi * r^2 * h)
  let volume_cone := (1 / 3) * Real.pi * r^2 * h
  volume_cylinder - volume_cone

-- 3. Ratio of the volume of the cylinder to the volume of the cone
theorem ratio_of_volumes (r h : ℝ) (volume_cylinder : ℝ := Real.pi * r^2 * h) (volume_cone : ℝ := (1 / 3) * Real.pi * r^2 * h) : ℝ :=
  (volume_cylinder / volume_cone)

-- Statements without proof
def problem1_volume_of_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

def problem2_volume_of_space_between (r h : ℝ) : ℝ :=
  let volume_cylinder := (Real.pi * r^2 * h)
  let volume_cone := (1 / 3) * Real.pi * r^2 * h
  volume_cylinder - volume_cone

def problem3_ratio_of_volumes (r h : ℝ) : ℝ :=
  3

#check volume_of_hemisphere
#check volume_of_space_between_cone_and_cylinder
#check ratio_of_volumes
#check problem1_volume_of_hemisphere
#check problem2_volume_of_space_between
#check problem3_ratio_of_volumes

end volume_of_hemisphere_volume_of_space_between_cone_and_cylinder_ratio_of_volumes_l770_770026


namespace simplify_expression_l770_770760

theorem simplify_expression (x : ℝ) (h : x = 1) : (x - 1)^2 + (x + 1) * (x - 1) - 2 * x^2 = -2 :=
by
  sorry

end simplify_expression_l770_770760


namespace sum_y_n_floor_abs_eq_1501_l770_770299

theorem sum_y_n_floor_abs_eq_1501 
  (y : ℕ → ℝ)
  (h : ∀ a : ℕ, 1 ≤ a ∧ a ≤ 1500 → y a + 2 * a = ∑ i in finset.range (1500 + 1), y i + 1501) :
  ∑ i in finset.range (1500 + 1), y i = -1501 → (| ∑ i in finset.range (1500 + 1), y i |).floor = 1501 :=
by
  sorry

end sum_y_n_floor_abs_eq_1501_l770_770299


namespace B_finish_time_l770_770662

-- Definitions based on the conditions of the problem.

def A_distance := 130 -- meters
def A_time := 20 -- seconds
def B_behind := 26 -- meters

-- Define the race problem conditions.
def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

def A_speed := speed A_distance A_time

def B_distance_in_A_time := A_distance - B_behind
def B_speed := speed B_distance_in_A_time A_time

def B_time_to_finish := A_distance / B_speed

-- Theorem stating the correct answer given the conditions.

theorem B_finish_time : B_time_to_finish = 25 := by
  sorry

end B_finish_time_l770_770662


namespace find_a_l770_770596

-- Define point A
def A : ℝ × ℝ := (-2, -1)

-- Define point B with unknown y-coordinate
def B (a : ℝ) : ℝ × ℝ := (3, a)

-- Define the slope formula for two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

-- Conditions
def angle_of_inclination (angle : ℝ) : ℝ := Real.tan angle

theorem find_a :
  let slope_l1 := angle_of_inclination (Real.pi / 4) in
  let slope_l2 := slope A (B a) in
  slope_l1 = slope_l2 → a = 4 :=
by
  sorry

end find_a_l770_770596


namespace sum_of_cubes_mod_5_l770_770483

theorem sum_of_cubes_mod_5 :
  ( ∑ k in Finset.range 50, (k + 1)^3 ) % 5 = 0 :=
sorry

end sum_of_cubes_mod_5_l770_770483


namespace rohan_salary_l770_770864

variable (S : ℝ)

theorem rohan_salary (h₁ : (0.20 * S = 2500)) : S = 12500 :=
by
  sorry

end rohan_salary_l770_770864


namespace game_winner_l770_770011

theorem game_winner (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  (mn % 2 = 1 → first_player_wins) ∧ (mn % 2 = 0 → second_player_wins) :=
sorry

end game_winner_l770_770011


namespace sum_real_roots_x_cubed_eq_64_l770_770201

theorem sum_real_roots_x_cubed_eq_64 : 
  (∃ (x : ℝ), x^3 = 64) → 
  ∑ x in {x : ℝ | x^3 = 64}.to_finset = 4 := 
by
  sorry

end sum_real_roots_x_cubed_eq_64_l770_770201


namespace decrease_in_demand_l770_770621

theorem decrease_in_demand :
  ∀ (p : ℝ) (c : ℝ) (p_new : ℝ) (c_new : ℝ) (q_new : ℝ) (q : ℝ),
    p = c ∧
    p_new = p * 1.20 ∧
    c_new = c * 1.10 ∧
    q = 1 →

    -- Condition for maintaining a 5% increase in total income
    (p_new * q_new - c_new * q_new) = (p - c) * 1.05 →
    
    -- Prove that the demand decrease percentage is 89.5%
    (1 - q_new / q) * 100 = 89.5 := 
begin
  intros,
  sorry
end

end decrease_in_demand_l770_770621


namespace evaluate_expression_l770_770109

theorem evaluate_expression :
  (∑ k in Finset.range 10, Real.logb (4^k) (2^(k^2)) ) *
  (∑ k in Finset.range 50, Real.logb (8^k) (16^k)) = 12833.333333... :=
by 
  sorry

end evaluate_expression_l770_770109


namespace household_savings_prediction_l770_770413

noncomputable def linear_regression_equation (n : ℕ) (x_sum y_sum xy_sum x_squared_sum : ℝ) : Prop :=
  let x_bar := x_sum / n
  let y_bar := y_sum / n
  let b := (xy_sum - n * x_bar * y_bar) / (x_squared_sum - n * x_bar^2)
  let a := y_bar - b * x_bar
  ∀ x : ℝ, y : ℝ, y = b * x + a 

theorem household_savings_prediction
  (n : ℕ)
  (x_sum y_sum xy_sum x_squared_sum : ℝ)
  (hn : n = 10)
  (hx_sum : x_sum = 80)
  (hy_sum : y_sum = 20)
  (hxy_sum : xy_sum = 184)
  (hx_squared_sum : x_squared_sum = 720) : 
  linear_regression_equation n x_sum y_sum xy_sum x_squared_sum ∧
  (let b := (184 - 10 * 8 * 2) / (720 - 10 * 8^2) in b = 0.3) ∧
  (0.3 > 0) ∧
  (let a := 2 - 0.3 * 8 in a = -0.4) ∧
  (let predicted_savings := 0.3 * 12 - 0.4 in predicted_savings = 3.2) :=
by
  sorry

end household_savings_prediction_l770_770413


namespace bisects_angle_ABC_l770_770683

variables {n : ℝ} {A B C D E : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Definitions of the points and the triangle properties
def is_right_isosceles_triangle (A B C : Type) := 
  (AC = BC) ∧ (angle C B A = 90)

def AE_perpendicular_BD (A B E D : Type) := 
  (⊥ (line_through A E) (line_through B D))

-- Given conditions
variables [h_triangle: is_right_isosceles_triangle A B C]
variable D_on_AC : D ∈ line_through A C
variable AE_perp_ext_BD : AE_perpendicular_BD A B E D
variable AE_ratio_BD : distance_between A E = (1 / n) * distance_between B D

-- The statement to be proved
theorem bisects_angle_ABC : bisects (angle A B C) (line_through B D) :=
  sorry

end bisects_angle_ABC_l770_770683


namespace proportion_solutions_l770_770499

variables (x1 x2 x3 x4 : ℤ)

definition satisfies_conditions :=
  x1 = x2 + 6 ∧
  x3 = x4 + 5 ∧
  x1 * x4 = x2 * x3 ∧
  x1^2 + x2^2 + x3^2 + x4^2 = 793

definition correct_answer_1 : Prop :=
  (x1, x2, x3, x4) = (18, 12, 15, 10)

definition correct_answer_2 : Prop :=
  (x1, x2, x3, x4) = (-12, -18, -10, -15)

theorem proportion_solutions :
  satisfies_conditions x1 x2 x3 x4 → correct_answer_1 x1 x2 x3 x4 ∨ correct_answer_2 x1 x2 x3 x4 :=
sorry

end proportion_solutions_l770_770499


namespace acrobats_count_l770_770106

def acrobats_parade (a e : ℕ) : Prop :=
  (2 * a + 4 * e = 60) ∧ (a + e = 20)

theorem acrobats_count : ∃ a e : ℕ, acrobats_parade a e ∧ a = 10 :=
by
  use 10, 10
  split
  · dsimp [acrobats_parade]
    split
    · simp
    · simp
  · rfl

end acrobats_count_l770_770106


namespace smallest_value_AC_l770_770446

theorem smallest_value_AC
  {A B C D : Type} [real_normed_field A]
  (h_isosceles : ∀ {a b : A}, a = b)
  (h_perpendicular : ∀ {bd ac : A}, bd * ac = 0)
  (h_odd_AC_CD : ∀ {ac cd : ℕ}, ac % 2 = 1 ∧ cd % 2 = 1)
  (h_BD_sq : ∀ {bd : A}, bd^2 = 65)
  : ∃ ac : ℕ, ac = 9 :=
by
  -- Setup definitions and proofs here
  sorry

end smallest_value_AC_l770_770446


namespace probability_inside_smaller_spheres_l770_770061

noncomputable def tetrahedron_probability (R : ℝ) (r : ℝ): ℝ := 
  5 / 27

theorem probability_inside_smaller_spheres 
  (R : ℝ) (r : ℝ) (P : EuclideanSpace ℝ (fin 3)) (inside_large_sphere : ∥P∥ ≤ R):
  let prob := tetrahedron_probability R r in
  prob = 5 / 27 :=
by {
  sorry
}

end probability_inside_smaller_spheres_l770_770061


namespace number_of_zeros_of_derivative_l770_770727

theorem number_of_zeros_of_derivative (a : ℝ) :
  let f' := λ x : ℝ, 6 * (x - 1) * (x - a) in
  if a < -1 ∨ a > 3 then
    ∃ x : ℝ, x ∈ Icc (-1 : ℝ) (3 : ℝ) ∧ f' x = 0
  else if a = 1 then
    ∃ x : ℝ, x ∈ Icc (-1 : ℝ) (3 : ℝ) ∧ f' x = 0
  else
    ∃ x₁ x₂ : ℝ, x₁ ∈ Icc (-1 : ℝ) (3 : ℝ) ∧ f' x₁ = 0 ∧ x₂ ∈ Icc (-1 : ℝ) (3 : ℝ) ∧ f' x₂ = 0 ∧ x₁ ≠ x₂ :=
sorry

end number_of_zeros_of_derivative_l770_770727


namespace complementary_angle_difference_l770_770784

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l770_770784


namespace renovation_project_total_l770_770414

def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem renovation_project_total : sand + dirt + cement = 0.67 := 
by
  sorry

end renovation_project_total_l770_770414


namespace total_worth_of_stock_l770_770064

variable (W : ℝ)
variable (overall_profit : ℝ)

def stock_profit_condition1 (W : ℝ) := 0.25 * W * 0.15
def stock_loss_condition2 (W : ℝ) := 0.40 * W * 0.05
def stock_profit_condition3 (W : ℝ) := 0.35 * W * 0.10

def total_profit (W : ℝ) := stock_profit_condition1 W + stock_profit_condition3 W - stock_loss_condition2 W

theorem total_worth_of_stock (h : overall_profit = 750) : W = 750 / 0.0525 := by
  have h1 : total_profit W = 0.0525 * W := by
    sorry
  have h2 : h = 0.0525 * W := by
    sorry
  simp only [h] at h2
  sorry

end total_worth_of_stock_l770_770064


namespace gas_fee_calculation_l770_770220

theorem gas_fee_calculation (x : ℚ) (h_usage : x > 60) :
  60 * 0.8 + (x - 60) * 1.2 = 0.88 * x → x * 0.88 = 66 := by
  sorry

end gas_fee_calculation_l770_770220


namespace prob_at_least_one_head_l770_770028

theorem prob_at_least_one_head (n : ℕ) (hn : n = 3) : 
  1 - (1 / (2^n)) = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_l770_770028


namespace election_winning_votes_l770_770000

noncomputable def total_votes (x y : ℕ) (p : ℚ) : ℚ := 
  (x + y) / (1 - p)

noncomputable def winning_votes (x y : ℕ) (p : ℚ) : ℚ :=
  p * total_votes x y p

theorem election_winning_votes :
  winning_votes 2136 7636 0.54336448598130836 = 11628 := 
by
  sorry

end election_winning_votes_l770_770000


namespace net_percentage_error_l770_770909

noncomputable section
def calculate_percentage_error (true_side excess_error deficit_error : ℝ) : ℝ :=
  let measured_side1 := true_side * (1 + excess_error / 100)
  let measured_side2 := measured_side1 * (1 - deficit_error / 100)
  let true_area := true_side ^ 2
  let calculated_area := measured_side2 * true_side
  let percentage_error := ((true_area - calculated_area) / true_area) * 100
  percentage_error

theorem net_percentage_error 
  (S : ℝ) (h1 : S > 0) : calculate_percentage_error S 3 (-4) = 1.12 := by
  sorry

end net_percentage_error_l770_770909


namespace binom_sum_l770_770095

theorem binom_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := 
by
  sorry

end binom_sum_l770_770095


namespace gloria_initial_pencils_l770_770626

-- Definitions based on conditions
variable (G : ℕ) -- G represents the number of pencils Gloria initially has.
variable (Lisa_pencils : ℕ) := 99
variable (final_pencils : ℕ) := 101

-- The condition of Lisa giving all her pencils to Gloria
axiom pencils_transfer : G + Lisa_pencils = final_pencils

-- Prove that Gloria initially has 2 pencils
theorem gloria_initial_pencils : G = 2 :=
by
  sorry -- Proof omitted

end gloria_initial_pencils_l770_770626


namespace quadratic_distinct_roots_l770_770541

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770541


namespace prob_two_heads_in_three_flips_l770_770880

-- Define the probability of heads for a fair coin
def prob_head : ℚ := 1 / 2

-- Define the probability of tails for a fair coin
def prob_tail : ℚ := 1 / 2

-- Define the events for each coin toss
def A1 : Prop
def A2 : Prop
def A3 : Prop

-- Define the event "exactly two heads appear in three tosses"
def A : Prop := (A1 ∧ A2 ∧ ¬A3) ∨ (A1 ∧ ¬A2 ∧ A3) ∨ (¬A1 ∧ A2 ∧ A3)

-- Define the probabilities for events assuming independence
axiom prob_A1 : ℚ := prob_head
axiom prob_A2 : ℚ := prob_head
axiom prob_A3 : ℚ := prob_head
axiom prob_not_A1 : ℚ := prob_tail
axiom prob_not_A2 : ℚ := prob_tail
axiom prob_not_A3 : ℚ := prob_tail

-- State the theorem about the probability of getting exactly two heads in three flips of a fair coin
theorem prob_two_heads_in_three_flips : 
  ∑ e : (A1 ∧ A2 ∧ ¬A3), ∑ e : (A ∧ ¬A2 ∧ A3), ∑ e : (¬A1 ∧ A2 ∧ A3) = 3 / 8 := 
sorry

end prob_two_heads_in_three_flips_l770_770880


namespace ellipse_probability_l770_770235

open Set Real 

theorem ellipse_probability:
  (∃ a b : ℝ, a ∈ Icc 1 5 ∧ b ∈ Icc 2 4 ∧ 
              a > b ∧ a < 2 * b) → 
  (probability event_of_interest = (15:32)) :=
sorry -- proof to be provided

end ellipse_probability_l770_770235


namespace sin_phi_correct_l770_770707

variables (p q r : EuclideanSpace 3 ℝ)

-- Define the magnitudes as given conditions
axiom norm_p : ∥p∥ = 2
axiom norm_q : ∥q∥ = 4
axiom norm_r : ∥r∥ = 6

-- Define the given vector triple product condition
axiom vector_condition : p × (p × q) = r

noncomputable def sin_phi : ℝ :=
  let φ := real.angle p q in
  real.sin φ

theorem sin_phi_correct : sin_phi p q r = 3 / 4 :=
by
  sorry

end sin_phi_correct_l770_770707


namespace AP_AQ_AR_sum_l770_770986

theorem AP_AQ_AR_sum (ABCDEF : RegularHexagon) (O : Point) (P Q R : Point)
  (hP : Perpendicular A CD P) (hQ : Perpendicular A DE Q) (hR : Perpendicular A EF R)
  (hOP : dist O P = 2) : 
  dist A P + dist A Q + dist A R = 2 * sqrt 3 := 
sorry

end AP_AQ_AR_sum_l770_770986


namespace homothety_of_triangle_ABC_parallel_lines_l770_770580

-- Define points A, B, C and their respective diametrically opposite points A1, B1, C1 on the circumcircle
variables (A B C A1 B1 C1 O H : Type) [circumcircle ABC A1 B1 C1] 

-- Theorem statement: Triangle formed by lines parallel to sides of ABC through A1, B1, C1
-- is homothetic to triangle ABC with scale 2 and centered at the orthocenter H
theorem homothety_of_triangle_ABC_parallel_lines :
  (triangle ((line A1 parallel_to (line B C)) ∧
             (line B1 parallel_to (line C A)) ∧
             (line C1 parallel_to (line A B))) ∧
   homothetic_center H) :=
begin
  sorry
end

end homothety_of_triangle_ABC_parallel_lines_l770_770580


namespace range_of_a_l770_770145

open Real

noncomputable def valid_a (a : ℝ) : Prop := log a (a + 2) > 1 ∧
    log a (a - 1) < 1

theorem range_of_a
  (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∃ x : ℝ, log a (x - 3) - log a (x + 2) - log a (x - 1) = 1) ↔
    a ∈ set.Ioo 0 ((7 : ℝ) - 2 * sqrt 10) / 9 :=
by
  sorry

end range_of_a_l770_770145


namespace newspaper_cost_over_8_weeks_l770_770629

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l770_770629


namespace candy_cost_l770_770055

variables (C : ℝ)

theorem candy_cost {w1 w2 : ℝ} (c1 c2 avg_cost : ℝ) (h1 : w1 = 10) (h2 : w2 = 20) (h3 : c1 = 8) (h4 : avg_cost = 6)
(h_mixture : (w1 + w2) = 30) : (w1 * c1 + w2 * C = (w1 + w2) * avg_cost) → C = 5 :=
by
  intros h_mix
  rw [h1, h2, h3, h4] at h_mix
  simp at h_mix
  linarith

end candy_cost_l770_770055


namespace area_of_shaded_region_zero_l770_770292

theorem area_of_shaded_region_zero
  (J K L M : ℝ × ℝ)
  (C B : ℝ × ℝ)
  (H_J : J = (0, 0))
  (H_K : K = (4, 0))
  (H_L : L = (4, 5))
  (H_M : M = (0, 5))
  (H_C : C = (1.5, 5))
  (H_B : B = (4, 4)) :
  let F := M in
  let E := J in
  let line_CF := {p : ℝ × ℝ | p.snd = 5} in
  let line_BE := {p : ℝ × ℝ | p.snd = p.fst} in
  set.prod {x | 0 ≤ x ∧ x ≤ 4} {y | 0 ≤ y ∧ y ≤ 5} ∩ (line_CF ∩ line_BE) = ∅ →
  ∃ area : ℝ, area = 0 :=
by 
  sorry

end area_of_shaded_region_zero_l770_770292


namespace probability_diff_color_correct_l770_770219

noncomputable def probability_diff_color (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) : ℚ :=
  (red_balls * yellow_balls) / ((total_balls * (total_balls - 1)) / 2)

theorem probability_diff_color_correct :
  probability_diff_color 5 3 2 = 3 / 5 :=
by
  sorry

end probability_diff_color_correct_l770_770219


namespace find_eccentricity_of_ellipse_l770_770995

noncomputable def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def eccentricity (a b : ℝ) := real.sqrt (1 - (b^2 / a^2))

theorem find_eccentricity_of_ellipse
  {a b c t : ℝ}
  (ha : a > 0)
  (hb : b > 0)
  (hc : c = real.sqrt (a^2 - b^2))
  (ht1 : |bf1| = 3 * t)
  (ht2 : |bf2| = 4 * t)
  (h_line : (|bf1| + |bf2|) / t = 2)
  (triangle_abc : is_right_triangle A B F2)
  (ratio : |af2| / |bf2| = 3 / 4) :
  eccentricity a b = 5 / 7 :=
by {
  sorry
}

end find_eccentricity_of_ellipse_l770_770995


namespace incorrect_operation_l770_770365

theorem incorrect_operation (a : ℝ) : ¬ (a^3 + a^3 = 2 * a^6) :=
by
  sorry

end incorrect_operation_l770_770365


namespace simplify_expression_l770_770295

theorem simplify_expression :
  (625: ℝ)^(1/4) * (256: ℝ)^(1/3) = 20 := 
sorry

end simplify_expression_l770_770295


namespace inequality_solution_set_l770_770798

theorem inequality_solution_set :
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} := 
sorry

end inequality_solution_set_l770_770798


namespace no_real_solution_xlog_x_eq_x3_div_1000_l770_770330

theorem no_real_solution_xlog_x_eq_x3_div_1000 :
  ¬ ∃ x : ℝ, x > 0 ∧ x ^ (Real.log10 x) = x ^ 3 / 1000 := 
sorry

end no_real_solution_xlog_x_eq_x3_div_1000_l770_770330


namespace number_of_ordered_pairs_l770_770796

theorem number_of_ordered_pairs (a r : ℕ) (h1 : 0 < a) (h2 : 0 < r) :
  log 8 a + log 8 (a * r) + log 8 (a * r ^ 2) + log 8 (a * r ^ 3) + 
  log 8 (a * r ^ 4) + log 8 (a * r ^ 5) + log 8 (a * r ^ 6) + 
  log 8 (a * r ^ 7) + log 8 (a * r ^ 8) + log 8 (a * r ^ 9) + 
  log 8 (a * r ^ 10) + log 8 (a * r ^ 11) = 2006 :=
begin
  sorry
end

end number_of_ordered_pairs_l770_770796


namespace complementary_angle_difference_l770_770782

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l770_770782


namespace TrianglePGGRComparison_l770_770228

structure RightTriangle (P Q R : Type) :=
  (PQ QR : ℝ)
  (angleQ : ∠Q = 90)
  (midpointM : Point)
  (midpointN : Point)
  (intersectionG : Point)

def isMidpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def distance (A B : Point) : ℝ :=
  sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

theorem TrianglePGGRComparison
  (P Q R : Point)
  (right_triangle : RightTriangle P Q R)
  (PQ_len : distance P Q = 15)
  (QR_len : distance Q R = 20)
  (midpointM_def : isMidpoint P Q M)
  (midpointN_def : isMidpoint P R N)
  (G_def : G = intersection right_triangle.intersectionG)

  : distance P G = 15 ∧ distance G R = 25 ∧ (distance P G / distance G R = 3 / 5) :=
  sorry

end TrianglePGGRComparison_l770_770228


namespace angle_BDE_l770_770301

variables {A B C D E : Type*}
variables [EuclideanGeometry A B C D E]

/-- There exist three congruent triangles (ABC, ACD, ADE) with AB = AC = AD = AE
and angle BAC = 30 degrees. -/
def congruent_triangles (ABC ACD ADE : Triangle) : Prop :=
  congruent ABC ACD ∧ congruent ACD ADE

def equal_sides (AB AC AD AE : Length) : Prop :=
  AB = AC ∧ AC = AD ∧ AD = AE

def angle_BAC (α : Angle) : Prop :=
  α = 30

theorem angle_BDE (ABC ACD ADE : Triangle) (AB AC AD AE : Length) (α : Angle)
  (h1 : congruent_triangles ABC ACD ADE)
  (h2 : equal_sides AB AC AD AE)
  (h3 : angle_BAC α) :
  AngleSum (DE ∪ ED) = 15 :=
by
  sorry

end angle_BDE_l770_770301


namespace simplify_correct_l770_770070

variable (x y : ℝ)

-- Given condition: the original expression to simplify
def original_expr := (1 / 2) * x - 2 * (x - (1 / 3) * y^2) + (-(3 / 2) * x + (1 / 3) * y^2)

-- The goal is to prove that the expression simplifies to -3x + y^2
theorem simplify_correct : original_expr x y = -3 * x + y^2 :=
by
  -- Skipping the actual proof, the goal statement is given
  sorry

end simplify_correct_l770_770070


namespace tangent_line_eq_min_value_condition_min_m_for_product_l770_770605

-- Define the function f(x) = x - 1 - a * ln x
def f (x : ℝ) (a : ℝ) : ℝ := x - 1 - a * Real.log x

-- Problem 1
theorem tangent_line_eq (a : ℝ) (h : a = 2) :
  (∀ x, (x = 4) → 
    let y := f 4 2 in
    let slope := 1 - 2 / 4 in
    y - (3 - 4 * Real.log 2) = slope * (x - 4)) := 
sorry

-- Problem 2
theorem min_value_condition (a : ℝ) :
  (∀ x, f x a ≥ 0) ↔ a = 1 :=
sorry

-- Problem 3
theorem min_m_for_product (m : ℕ) (n : ℕ) :
  ((1 + 1 / 2) * (1 + 1 / 2^2) * ... * (1 + 1 / 2^n) < m) ↔ m = 3 :=
sorry

end tangent_line_eq_min_value_condition_min_m_for_product_l770_770605


namespace sum_of_real_solutions_l770_770128

theorem sum_of_real_solutions : 
  let f := λ x : ℝ, (x - 3) / (x^2 + 3 * x + 2)
  let g := λ x : ℝ, (x - 6) / (x^2 - 12 * x + 32)
  ∃ s : ℝ, (∀ x : ℝ, f x = g x → (x^2 - 7 * x + 7 = 0)) ∧ 
            (∀ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 - 7 * r1 + 7 = 0) ∧ (r2^2 - 7 * r2 + 7 = 0) → 
                          (s = r1 + r2)) ∧ s = 7 :=
by
  sorry

end sum_of_real_solutions_l770_770128


namespace order_of_variables_l770_770161

theorem order_of_variables (a b c : ℝ) (h₁ : a = real.log 2 / real.log 3) 
                           (h₂ : b = real.log 2) (h₃ : c = (0.5 : ℝ)^(-0.5)) : 
  a < b ∧ b < c :=
by
  sorry

end order_of_variables_l770_770161


namespace isosceles_triangle_APQ_l770_770990

noncomputable def acute_triangle (A B C : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] : Prop :=
∃ (D E F : Type) [Altitude A D] [Altitude B E] [Altitude C F], 
  ∃ (K J : Type) [Incenter (B, D, F) K] [Incenter (C, D, E) J],
  ∃ (P Q : Type) [Line J K] [Intersection (Line J K) (Line A B) P] [Intersection (Line J K) (Line A C) Q],
  is_isosceles_triangle A P Q

theorem isosceles_triangle_APQ (A B C D E F K J P Q : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
    [Altitude A D] [Altitude B E] [Altitude C F]
    [Incenter (B, D, F) K] [Incenter (C, D, E) J]
    [Line J K] [Intersection (Line J K) (Line A B) P] [Intersection (Line J K) (Line A C) Q] :
  is_isosceles_triangle A P Q := sorry

end isosceles_triangle_APQ_l770_770990


namespace sum_of_shaded_areas_l770_770047

open Real

-- Definitions and given conditions
def side_length_of_equilateral_triangle : ℝ := 16
def radius_of_central_circle: ℝ := side_length_of_equilateral_triangle / 2
def height_of_equilateral_triangle: ℝ := radius_of_central_circle * √3
def radius_of_inscribed_circle : ℝ := height_of_equilateral_triangle / 3

-- Main theorem
theorem sum_of_shaded_areas : ∃ (a b c : ℝ), 
  aπ - b * √c = (radius_of_central_circle^2 * π * 1/6) * 2 - (radius_of_inscribed_circle^2 *π * 2/6) ∧ 
  a + b + c = 14.22 := 
by {
  use [128/9, 0, 0],
  split,
  { 
    -- prove the first part: aπ - b * √c = (radius_of_central_circle^2 * π * 1/6) * 2 - (radius_of_inscribed_circle^2 *π * 2/6)
    sorry 
  },
  {  
    -- prove the second part: a + b + c = 14.22
    sorry 
  }
}

end sum_of_shaded_areas_l770_770047


namespace number_of_triangles_l770_770949

open Nat

/-- Each side of a square is divided into 8 equal parts, and using the divisions
as vertices (not including the vertices of the square), the number of different 
triangles that can be obtained is 3136. -/
theorem number_of_triangles (n : ℕ := 7) :
  (n * 4).choose 3 - 4 * n.choose 3 = 3136 := 
sorry

end number_of_triangles_l770_770949


namespace count_of_integers_without_specific_digits_l770_770191

theorem count_of_integers_without_specific_digits : 
  let forbidden_digits := [2, 3, 4, 5, 8, 9]
  let filter_digits (n : ℕ) : Bool := n.digits 10.all (λ d, d ∉ forbidden_digits)
  ∀ n ∈ (finset.range 10000).filter filter_digits, 
    (finset.range 10000).filter filter_digits.card = 256 := 
sorry

end count_of_integers_without_specific_digits_l770_770191


namespace probability_b_l770_770068

def P (A B : Prop) : Prop := A ∧ B

theorem probability_b (P_A : Prop) (P_truth_A : P_A = 0.80) (P_A_and_B : P (P_A 0.60) = 0.48) : (P 0.60 0.80) = \boxed{0.60} := sorry

end probability_b_l770_770068


namespace alloy_copper_percentage_l770_770085

theorem alloy_copper_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x)
  (h2 : (30 / 100) * x + (70 / 100) * 27 = 24.9) :
  x = 20 :=
sorry

end alloy_copper_percentage_l770_770085


namespace sum_of_cubes_mod_five_l770_770489

theorem sum_of_cubes_mod_five : 
  (∑ n in Finset.range 51, n^3) % 5 = 0 := by
  sorry

end sum_of_cubes_mod_five_l770_770489


namespace garden_perimeter_l770_770325

-- Definitions for length and breadth
def length := 150
def breadth := 100

-- Theorem that states the perimeter of the rectangular garden
theorem garden_perimeter : (2 * (length + breadth)) = 500 :=
by sorry

end garden_perimeter_l770_770325


namespace sum_coeff_no_y_eq_one_l770_770799

noncomputable def sum_of_coefficients_without_y (n : ℕ) : ℕ :=
if n = 0 then 1 else 1

theorem sum_coeff_no_y_eq_one (n : ℕ) (h : n > 0) : sum_of_coefficients_without_y n = 1 :=
by 
  rw sum_of_coefficients_without_y
  split_ifs
  . contradiction
  . rfl

end sum_coeff_no_y_eq_one_l770_770799


namespace sum_of_cubes_mod_l770_770473

theorem sum_of_cubes_mod (S : ℕ) (h : S = ∑ n in Finset.range 51, n^3) : S % 5 = 0 :=
sorry

end sum_of_cubes_mod_l770_770473


namespace number_of_planes_determined_l770_770895

noncomputable def planes_determined_by_points_and_line 
  (l : Line) (A B C : Point) (outside_l : ∀ (P : Point), P ∈ {A, B, C} → ¬(P ∈ l))
  : Nat :=
if collinear {A, B, C} then 1 else 4

theorem number_of_planes_determined
  (l : Line) (A B C : Point)
  (outside_l : ∀ (P : Point), P ∈ {A, B, C} → ¬(P ∈ l))
  : planes_determined_by_points_and_line l A B C outside_l ∈ {1, 3, 4} :=
sorry

end number_of_planes_determined_l770_770895


namespace even_numbers_in_top_15_rows_of_pascals_triangle_l770_770448

def binomial (n k : ℕ) : ℕ := nat.choose n k

def is_even (n : ℕ) : Prop := n % 2 = 0

def count_even_in_pascals_triangle_up_to_row (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum (λ r, 
    (finset.range (r + 1)).filter (λ k, is_even (binomial r k)).card)

theorem even_numbers_in_top_15_rows_of_pascals_triangle :
  count_even_in_pascals_triangle_up_to_row 15 = 51 :=
sorry

end even_numbers_in_top_15_rows_of_pascals_triangle_l770_770448


namespace positive_difference_of_complementary_ratio_5_1_l770_770786

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l770_770786


namespace remainder_polynomial_l770_770958

variable (x : ℂ)

theorem remainder_polynomial :
  ∃ q : polynomial ℂ,
    polynomial.eval x x^6 - polynomial.eval x x^5 - polynomial.eval x x^4 +
    polynomial.eval x x^3 + polynomial.eval x x^2 =
    ((polynomial.eval x x^2 - 1) * (polynomial.eval x x - 2)) * q x +
    (9 * polynomial.eval x x^2 - 8) :=
by
  sorry

end remainder_polynomial_l770_770958


namespace quadratic_distinct_roots_l770_770543

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l770_770543


namespace quadratic_roots_condition_l770_770510

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770510


namespace base_7_to_10_of_23456_l770_770826

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l770_770826


namespace exponential_simplification_l770_770851

theorem exponential_simplification : 
  (10^0.25) * (10^0.25) * (10^0.5) * (10^0.5) * (10^0.75) * (10^0.75) = 1000 := 
by 
  sorry

end exponential_simplification_l770_770851


namespace investment_by_a_l770_770862

theorem investment_by_a 
  (b_investment : ℕ) (c_investment : ℕ) 
  (b_share : ℕ) (a_share : ℕ) 
  (months : ℕ) 
  (b_investment_eq : b_investment = 21000)
  (c_investment_eq : c_investment = 27000)
  (b_share_eq : b_share = 1540)
  (a_share_eq : a_share = 1100)
  (months_eq : months = 8) : 
  ∃ x : ℕ, x = 15000 :=
by
  let x := 15000
  have h : 1540 * x = 21000 * 1100, by sorry
  have h_x : x = 15000, by sorry
  use x
  exact h_x

end investment_by_a_l770_770862


namespace parallel_lines_m_values_l770_770971

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 → 2 * x + (5 + m) * y = 8) →
  (m = -1 ∨ m = -7) :=
by
  sorry

end parallel_lines_m_values_l770_770971


namespace full_bucket_weight_l770_770050

def weight_full_bucket (c d : ℝ) : ℝ :=
  let x := (3/2 : ℝ) * c - (1/2 : ℝ) * d
  let y := 2 * (d - c)
  x + y

theorem full_bucket_weight (c d : ℝ) :
  (∃ x y : ℝ, x + (1/4 : ℝ) * y = c ∧ x + (3/4 : ℝ) * y = d) → 
  weight_full_bucket c d = (1/2 : ℝ) * d + (1/2 : ℝ) * c :=
by
  intros h
  rcases h with ⟨x, y, h1, h2⟩
  unfold weight_full_bucket
  sorry

end full_bucket_weight_l770_770050


namespace chord_line_l770_770183

-- Definitions or conditions of the problem
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def midpoint (x y : ℝ) : Prop := (x = 1) ∧ (y = 1)

-- The main statement to be proved
theorem chord_line (x y k : ℝ) (h1 : midpoint x y) (h2 : parabola x y) : 
  (2 * x - y - 1 = 0) := 
sorry

end chord_line_l770_770183


namespace sum_first_four_l770_770660

variable {a r : ℝ}

-- Conditions
def sum_first_two : Prop := a * (1 + r) = 7
def sum_first_six : Prop := a * (r^6 - 1) / (r - 1) = 91

-- Goal/Proof
theorem sum_first_four (h1 : sum_first_two) (h2 : sum_first_six) : 
  a + ar + ar^2 + ar^3 = 28 :=
sorry

end sum_first_four_l770_770660


namespace train_average_speed_l770_770074

-- Define the variables used in the conditions
variables (D V : ℝ)
-- Condition: Distance D in 50 minutes at average speed V kmph
-- 50 minutes to hours conversion
def condition1 : D = V * (50 / 60) := sorry
-- Condition: Distance D in 40 minutes at speed 60 kmph
-- 40 minutes to hours conversion
def condition2 : D = 60 * (40 / 60) := sorry

-- Claim: Current average speed V
theorem train_average_speed : V = 48 :=
by
  -- Using the conditions to prove the claim
  sorry

end train_average_speed_l770_770074


namespace problem1_problem2_l770_770623

-- Definitions of the sets A, B, C
def A (a : ℝ) : Set ℝ := { x | x^2 - a*x + a^2 - 12 = 0 }
def B : Set ℝ := { x | x^2 - 2*x - 8 = 0 }
def C (m : ℝ) : Set ℝ := { x | m*x + 1 = 0 }

-- Problem 1: If A = B, then a = 2
theorem problem1 (a : ℝ) (h : A a = B) : a = 2 := sorry

-- Problem 2: If B ∪ C m = B, then m ∈ {-1/4, 0, 1/2}
theorem problem2 (m : ℝ) (h : B ∪ C m = B) : m = -1/4 ∨ m = 0 ∨ m = 1/2 := sorry

end problem1_problem2_l770_770623


namespace find_a5_a6_l770_770232

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

-- Given conditions
axiom h1 : a 1 + a 2 = 5
axiom h2 : a 3 + a 4 = 7

-- Arithmetic sequence property
axiom arith_seq : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)

-- The statement we want to prove
theorem find_a5_a6 : a 5 + a 6 = 9 :=
sorry

end ArithmeticSequence

end find_a5_a6_l770_770232


namespace angle_equivalence_l770_770311

theorem angle_equivalence (angle_deg : Int) (k : Int) :
  angle_deg = -1825 ∧ angle_deg = k * 360 + -25 → 
  angle_deg % 360 = -25 ∧ 
  (angle_deg : ℝ) * (Real.pi / 180) = -(5 / 36) * Real.pi :=
  by {
    intro h,
    cases h with h_deg h_k,
    split,
    { exact Int.mod_eq_of_lt h_deg (by norm_num),
    },
    { norm_num,
      rw [←h_deg],
      norm_num }
  }

end angle_equivalence_l770_770311


namespace sin_identity_l770_770976

theorem sin_identity
  (α : ℝ)
  (h : Real.sin (π / 4 + α) = sqrt 3 / 2) :
  Real.sin (3 * π / 4 - α) = sqrt 3 / 2 := 
sorry

end sin_identity_l770_770976


namespace product_of_digits_l770_770207

-- Define the conditions and state the theorem
theorem product_of_digits (A B : ℕ) (h1 : (10 * A + B) % 12 = 0) (h2 : A + B = 12) : A * B = 32 :=
  sorry

end product_of_digits_l770_770207


namespace quadratic_distinct_roots_l770_770550

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l770_770550


namespace standard_ellipse_equation_slope_sum_constant_l770_770581

-- Assume the necessary definitions and conditions
variables {a b c : ℝ} (a_pos : a > 0) (b_pos : b > 0) (ab_order : a > b) (ecc : a = 2 * c) (ecc_half : c = (1 / 2) * a)
variable [Fact (a^2 = b^2 + c^2)]

-- Given ellipse equation
def ellipse := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

-- Given the conditions on the segment RS
variable (RS_len : 3 = (2 * b^2) / a)

-- Standard equation of the ellipse
theorem standard_ellipse_equation : 
  ellipse 2 sqrt(3) = (x^2 / 4 + y^2 / 3 = 1) := sorry

-- Given point
def T : ℝ × ℝ := (4, 0)

-- Slopes sum theorem
theorem slope_sum_constant (F2 : ℝ × ℝ) (l : ℝ × ℝ → Prop) (R S : ℝ × ℝ) : 
  ∀ l : ℝ × ℝ, passes_through l F2 → intersects l ellipse R S → 
  slope T R + slope T S = 0 := sorry

end standard_ellipse_equation_slope_sum_constant_l770_770581


namespace length_of_EF_l770_770682

-- Define segments and their lengths:
variables (A B C D E F : Type) [linear_ordered_field A]
variables (AB BC CA AD AE AF BD CD : A)

-- Given conditions
axiom h1 : AB = 14
axiom h2 : BC = 6
axiom h3 : CA = 9
axiom h4 : BD = (1 / 10) * BC
axiom h5 : CD = (9 / 10) * BC
axiom h6 : E = Point where inscribed circle of triangle ADC touches AD
axiom h7 : F = Point where inscribed circle of triangle ADB touches AD

-- Proof problem
theorem length_of_EF : EF = 4.9 :=
sorry

end length_of_EF_l770_770682


namespace common_chord_length_l770_770122

noncomputable def length_common_chord (C1 C2 : ℝ → ℝ → Prop) : ℝ :=
  if (C1 0 0 = (0:ℝ)) ∧ (C2 0 0 = (0:ℝ)) then (24 / 5) else 0

axiom circle_C1 : ∀ (x y : ℝ), x^2 + y^2 - 9 = 0
axiom circle_C2 : ∀ (x y : ℝ), x^2 + y^2 - 6*x + 8*y + 9 = 0

theorem common_chord_length :
  length_common_chord (λ x y, circle_C1 x y) (λ x y, circle_C2 x y) = 24 / 5 :=
begin
  sorry
end

end common_chord_length_l770_770122


namespace sum_series_l770_770042

noncomputable def sum_of_first_n (k : ℕ) : ℚ :=
  k * (k + 1) / 2

theorem sum_series (n : ℕ) :
  1 + ∑ k in finset.range (n-1), (1 / sum_of_first_n (k+2)) = 2 * n / (n + 1) :=
by
  sorry

end sum_series_l770_770042


namespace some_knight_without_wine_l770_770388

-- Definitions based on conditions
def Knights : Type := Fin 50

-- Initial state: each knight has a glass of either red or white wine
inductive Wine
| Red
| White

-- We need at least one glass of each type
axiom exists_red_wine : ∃ k : Knights, Wine k = Wine.Red
axiom exists_white_wine : ∃ k : Knights, Wine k = Wine.White

-- Function to represent who has the glass after the first clap
def after_first_clap (w : Knights → Wine) : Knights → Wine
| k => if w k = Wine.Red then w (k - 1 + 50) % 50 else w k  -- circular table

-- Function to represent who has the glass after the second clap
def after_second_clap (w : Knights → Wine) : Knights → Wine
| k => if (after_first_clap w) k = Wine.White 
       then (after_first_clap w) (k - 2 + 50) % 50 
       else (after_first_clap w) k  -- circular table

-- The main statement to prove that some knight is left without wine after the second clap
theorem some_knight_without_wine : 
  ∀ (w : Knights → Wine), 
  (∃ k : Knights, after_second_clap w k ≠ w k) → 
  ∃ k : Knights, after_second_clap w k = none :=
sorry

end some_knight_without_wine_l770_770388


namespace contrapositive_of_quadratic_l770_770771

theorem contrapositive_of_quadratic (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by
  sorry

end contrapositive_of_quadratic_l770_770771


namespace math_problem_l770_770436

def a : ℕ := 202222
def b : ℕ := 202223
def c : ℕ := 202224
def x : ℕ := 404445  -- x = a + b

theorem math_problem :
  \left(\frac{x^{2}}{a * b * c} - \frac{b}{a * c} - \frac{a}{b * c}\right) * 12639 = \frac{1}{8} :=
sorry

end math_problem_l770_770436


namespace thickness_of_stack_l770_770685

theorem thickness_of_stack (books : ℕ) (avg_pages_per_book : ℕ) (pages_per_inch : ℕ) (total_pages : ℕ) (thick_in_inches : ℕ)
    (h1 : books = 6)
    (h2 : avg_pages_per_book = 160)
    (h3 : pages_per_inch = 80)
    (h4 : total_pages = books * avg_pages_per_book)
    (h5 : thick_in_inches = total_pages / pages_per_inch) :
    thick_in_inches = 12 :=
by {
    -- statement without proof
    sorry
}

end thickness_of_stack_l770_770685


namespace polynomial_sum_l770_770993

theorem polynomial_sum (a : Fin 2015 → ℝ) 
  (h₀ : (1 - 2 * x) ^ 2014 = ∑ i in Finset.range 2015, (a i) * x ^ i) 
  (h₁ : a 0 = 1) 
  (h₂ : ∑ i in Finset.range 2015, a i = 1) :
  (∑ k in Finset.range 2015, a 0 + a (k + 1)) = 2014 := 
by
sorried.

end polynomial_sum_l770_770993


namespace two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l770_770719

open Nat

theorem two_pow_m_minus_one_not_divide_three_pow_n_minus_one 
  (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hmo : Odd m) (hno : Odd n) : ¬ (∃ k : ℕ, 2^m - 1 = k * (3^n - 1)) := by
  sorry

end two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l770_770719


namespace value_of_a_if_f_is_odd_f_is_decreasing_range_of_f_on_interval_l770_770603

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 1 / (4^x + 1)

theorem value_of_a_if_f_is_odd :
  (∀ x : ℝ, f x = -f (-x)) → ∃ a : ℝ, a = -1/2 :=
sorry

theorem f_is_decreasing :
  ∃ a : ℝ, a = -1/2 ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 (-1/2) > f x2 (-1/2)) :=
sorry

theorem range_of_f_on_interval :
  (∀ x : ℝ, f x = -f (-x)) →
  ∃ a : ℝ, a = -1/2 ∧ ∀ x : ℝ, x ∈ Ico (-1 : ℝ) 2 → 
  (f x (-1/2) ∈ Ioc (-15/34 : ℝ) (3/10 : ℝ)) :=
sorry

end value_of_a_if_f_is_odd_f_is_decreasing_range_of_f_on_interval_l770_770603


namespace find_a_for_domain_l770_770179

theorem find_a_for_domain 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/3)^(a*x^2 - 4*x + 3))
  (h_domain : ∀ x, x > 0 → x < ∞) :
  a = -1 :=
sorry

end find_a_for_domain_l770_770179


namespace trig_identity_equiv_l770_770141

theorem trig_identity_equiv (θ : ℝ) (h : 3 * sin (-3 * real.pi + θ) + cos (real.pi - θ) = 0) : 
  sin θ * cos θ / cos(2 * θ) = -3 / 8 := 
by
  sorry

end trig_identity_equiv_l770_770141


namespace teal_bakery_pumpkin_pie_l770_770675

theorem teal_bakery_pumpkin_pie (P : ℕ) 
    (pumpkin_price_per_slice : ℕ := 5)
    (custard_price_per_slice : ℕ := 6)
    (pumpkin_pies_sold : ℕ := 4)
    (custard_pies_sold : ℕ := 5)
    (custard_pieces_per_pie : ℕ := 6)
    (total_revenue : ℕ := 340) :
    4 * P * pumpkin_price_per_slice + custard_pies_sold * custard_pieces_per_pie * custard_price_per_slice = total_revenue → P = 8 := 
by
  sorry

end teal_bakery_pumpkin_pie_l770_770675


namespace option_A_results_in_2_l770_770364

theorem option_A_results_in_2 :
  -(-2) = 2 ∧ +(-2) ≠ 2 ∧ -(+2) ≠ 2 ∧ -| -2 | ≠ 2 :=
by
  sorry

end option_A_results_in_2_l770_770364


namespace sum_cubes_mod_five_l770_770493

theorem sum_cubes_mod_five :
  (∑ n in Finset.range 50, (n + 1)^3) % 5 = 0 :=
by
  sorry

end sum_cubes_mod_five_l770_770493


namespace find_y_intersection_of_tangents_l770_770699

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the tangent slope at a point on the parabola
def tangent_slope (x : ℝ) : ℝ := 2 * (x - 1)

-- Define the perpendicular condition for tangents at points A and B
def perpendicular_condition (a b : ℝ) : Prop := (a - 1) * (b - 1) = -1 / 4

-- Define the y-coordinate of the intersection point P of the tangents at A and B
def y_coordinate_of_intersection (a b : ℝ) : ℝ := a * b - a - b + 2

-- Theorem to be proved
theorem find_y_intersection_of_tangents (a b : ℝ) 
  (ha : parabola a = a ^ 2 - 2 * a - 3) 
  (hb : parabola b = b ^ 2 - 2 * b - 3) 
  (hp : perpendicular_condition a b) :
  y_coordinate_of_intersection a b = -1 / 4 :=
sorry

end find_y_intersection_of_tangents_l770_770699


namespace complement_correct_l770_770188

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}

theorem complement_correct : (U \ A) = {2, 4} := by
  sorry

end complement_correct_l770_770188


namespace probability_gold_coin_biased_l770_770919

open Probability

-- Definition of events and probabilities
variables {Ω : Type*} (p : pmf Ω)
variables (A B : set Pred) -- Event A: Gold coin is biased; Event B: Observing the given sequence of coin flips.
variables (P : pred ℝ) -- Probability function

-- Assume mutually exclusive events A and not A.
axiom mutually_exclusive {Ω : Type} (p : pmf Ω) (A : set Ω) :
  P (A ∩ Aᶜ) = 0

-- Assumptions based on conditions from the problem
axiom gold_coin_biased (A : set Ω) : p.heads = 0.6
axiom silver_coin_fair (B : set Ω) : p.heads = 0.5
axiom gold_coin_event (A : set Ω) : P (B | A) = 0.15
axiom gold_coin_biased_prob (A : set Ω) : P (A) = 0.5
axiom silver_coin_event (Aᶜ : set Ω) : P (B | Aᶜ) = 0.12
axiom silver_coin_notbiased_prob (Aᶜ : set Ω) : P (Aᶜ) = 0.5

-- Translate Bayes' theorem to our setup in Lean 
noncomputable def bayes_theorem : ℝ := 
  (P (B | A) * P (A)) / (P (B | A) * P (A) + P (B | Aᶜ) * P (Aᶜ))

-- The final statement asserting the desired probability 
theorem probability_gold_coin_biased : bayes_theorem p A B = 5 / 9 := sorry

end probability_gold_coin_biased_l770_770919


namespace log_sqrt_two_l770_770966

theorem log_sqrt_two : log 2 (2^(1/2)) = 1/2 := by
  sorry

end log_sqrt_two_l770_770966


namespace chimes_2400_occur_on_march_7_l770_770054

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0) ∧ ((y % 100 ≠ 0) ∨ (y % 400 = 0))

def chimes_per_day : ℕ := 258

noncomputable def chimes_in_hour (h : ℕ) : ℕ :=
  match h % 12 with
  | 0     => 36
  | n + 1 => 3 * (n + 1)

def total_chimes (start_hour start_minute : ℕ) (total_chimes_needed : ℕ) : ℕ :=
  let quarters_chimes := ((24 - start_hour - 1) * 4 + (60 - start_minute) / 15) in
  let hour_chimes := ((start_hour + 1)..23).foldl (λ acc h => acc + chimes_in_hour h) 0 in
  quarters_chimes + hour_chimes
  
theorem chimes_2400_occur_on_march_7
  (start_hour start_minute : ℕ)
  (start_month start_day start_year : ℕ)
  (is_leap : is_leap_year start_year)
  (total_chimes_needed : ℕ) :
  start_hour = 17 ∧ start_minute = 45 ∧
  start_month = 2 ∧ start_day = 28 ∧ start_year = 2004 ∧
  total_chimes (start_hour, start_minute) total_chimes_needed = 2400 →
  ∃ (end_month end_day : ℕ),
    end_month = 3 ∧ end_day = 7 :=
by
  sorry

end chimes_2400_occur_on_march_7_l770_770054


namespace range_of_a_l770_770728

open Set

variable (a : ℝ)
def A := {x | -3 ≤ x ∧ x ≤ a}
def B := {y | ∃ x, x ∈ A ∧ y = 3 * x + 10}
def C := {z | ∃ x, x ∈ A ∧ z = 5 - x}

theorem range_of_a (h : B ∩ C = C) : -2 / 3 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l770_770728


namespace inequality_solution_l770_770297

theorem inequality_solution (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x > 1) ∧ 
  (0 < a ∧ a < 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x < 1) :=
by {
  sorry
}

end inequality_solution_l770_770297


namespace variable_cost_per_book_fixed_cost_l770_770065

theorem variable_cost_per_book_fixed_cost (fixed_costs : ℝ) (selling_price_per_book : ℝ) 
(number_of_books : ℝ) (total_costs total_revenue : ℝ) (variable_cost_per_book : ℝ) 
(h1 : fixed_costs = 35630) (h2 : selling_price_per_book = 20.25) (h3 : number_of_books = 4072)
(h4 : total_costs = fixed_costs + variable_cost_per_book * number_of_books)
(h5 : total_revenue = selling_price_per_book * number_of_books)
(h6 : total_costs = total_revenue) : variable_cost_per_book = 11.50 := by
  sorry

end variable_cost_per_book_fixed_cost_l770_770065


namespace problem_A_problem_B_problem_C_problem_D_l770_770572

open Complex Real Set

noncomputable def z1 : ℂ := (1 + 3 * I) / (1 - 3 * I)
def conj_z1 : ℂ := (-4 - 3 * I) / 5

theorem problem_A : conj(z1) = conj_z1 := sorry

def z_pure_imaginary (b : ℝ) : ℂ := b * I

theorem problem_B (b : ℝ) (h : b ≠ 0) : (z_pure_imaginary b)^2 < 0 := sorry

def z2 (z : ℂ) : Prop := z - (2 + I) > 0

theorem problem_C (z : ℂ) : ¬ z2 z := sorry

def M : Set ℂ := {z | abs(z + 3 * I) ≤ 3}

def area_M : ℝ := 3 * 3 * π

theorem problem_D : area_M ≠ 6 * π := sorry

end problem_A_problem_B_problem_C_problem_D_l770_770572


namespace tom_sleep_deficit_l770_770005

-- Define the conditions as given
def weeknights := 5
def weekend_nights := 2
def ideal_sleep_hours := 8
def actual_weeknight_sleep := 5
def actual_weekend_sleep := 6

-- Define the proofs
theorem tom_sleep_deficit : 
  let ideal_week_sleep := (weeknights * ideal_sleep_hours) + (weekend_nights * ideal_sleep_hours) in
  let actual_week_sleep := (weeknights * actual_weeknight_sleep) + (weekend_nights * actual_weekend_sleep) in
  ideal_week_sleep - actual_week_sleep = 19 := 
by
  let ideal_week_sleep := (weeknights * ideal_sleep_hours) + (weekend_nights * ideal_sleep_hours)
  let actual_week_sleep := (weeknights * actual_weeknight_sleep) + (weekend_nights * actual_weekend_sleep)
  have h1 : ideal_week_sleep = 56 := by rfl
  have h2 : actual_week_sleep = 37 := by rfl
  show ideal_week_sleep - actual_week_sleep = 19 from by
    rw [h1, h2]
    apply nat.sub_self_eq_suc_natu

/-- Placeholder for the proof -/
sorry

end tom_sleep_deficit_l770_770005


namespace smallest_sector_angle_l770_770692

theorem smallest_sector_angle (a r : ℕ) (h1 : ∃ a r : ℕ, (a * (r^8 - 1) / (r - 1) = 360) ∧ (∀ i, a * r^i < 360) ∧ (a * r^i).natAbs ∈ finset.range 360) : 
  a = 45 :=
sorry

end smallest_sector_angle_l770_770692


namespace total_canoes_built_l770_770434

-- Definitions of conditions
def initial_canoes : ℕ := 8
def common_ratio : ℕ := 2
def number_of_months : ℕ := 6

-- Sum of a geometric sequence formula
-- Sₙ = a * (r^n - 1) / (r - 1)
def sum_of_geometric_sequence (a r n : ℕ) : ℕ := 
  a * (r^n - 1) / (r - 1)

-- Statement to prove
theorem total_canoes_built : 504 = sum_of_geometric_sequence initial_canoes common_ratio number_of_months := 
  by
  sorry

end total_canoes_built_l770_770434


namespace snooker_vip_seat_price_l770_770418

theorem snooker_vip_seat_price :
  ∃ (V : ℕ), let G := 298 in
  let V_t := G - 276 in
  20 * G + V * V_t = 7500 ∧ G + V_t = 320 ∧ V = 70 :=
begin
  use 70,
  simp,
  split,
  { norm_num },
  split,
  { norm_num },
  { refl }
end

end snooker_vip_seat_price_l770_770418


namespace quadratic_eq_roots_l770_770708

theorem quadratic_eq_roots
  (ω : ℂ) (h₀ : ω ^ 8 = 1) (h₁ : ω ≠ 1) :
  let α := ω + ω^3 + ω^5
  let β := ω^2 + ω^4 + ω^6 + ω^7
  in
  ∃ a b : ℝ, (a = 1 ∧ b = 3) ∧ (α^2 + a * α + b = 0 ∧ β^2 + a * β + b = 0) :=
by
  sorry

end quadratic_eq_roots_l770_770708


namespace largest_x_satisfying_abs_eq_largest_x_is_correct_l770_770459

theorem largest_x_satisfying_abs_eq (x : ℝ) (h : |x - 5| = 12) : x ≤ 17 :=
by
  sorry

noncomputable def largest_x : ℝ := 17

theorem largest_x_is_correct (x : ℝ) (h : |x - 5| = 12) : x ≤ largest_x :=
largest_x_satisfying_abs_eq x h

end largest_x_satisfying_abs_eq_largest_x_is_correct_l770_770459


namespace number_of_rhombuses_l770_770908

theorem number_of_rhombuses {n : ℕ} (T : Type) [DecidableEq T] :
  (n = 10) →
  (∀ (k : ℕ), k ≤ n → ∃! tri : T, (side_length tri = 1) ∧ (tri_in T)) →
  (∀ (rh : ℕ), rh = 8 → ∃! rhombus : T, rhombus_with_8 tris) →
  (total_rhombuses_with_8 tris = 84) := 
by
  intros n_eq ten_partition rhombus_condition
  sorry

end number_of_rhombuses_l770_770908


namespace simplify_expression_l770_770761

theorem simplify_expression (x : ℝ) (h : x = 1) : (x - 1)^2 + (x + 1) * (x - 1) - 2 * x^2 = -2 :=
by
  sorry

end simplify_expression_l770_770761


namespace problem1_problem2_problem3_l770_770609

-- Definition based on the derived function
def f (x : ℝ) : ℝ := (2 * 2^x - 2) / (2^x + 1)

-- Problem 1: Prove that the function derived from the conditions is correct
theorem problem1 (a b : ℝ) (hx1 : a = 2) (hx2 : b = -3) :
  (∀ x, f (2:ℝ) = 6 / 5 ∧ ∀ x : ℝ, f(-x) = -f(x)) :=
  sorry

-- Problem 2: Prove the monotonicity of the function
theorem problem2 : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) :=
  sorry

-- Problem 3: Prove the solution set of the inequality
theorem problem3 (x : ℝ) (hx : 1 < x ∧ x ≤ 6 / 5) :
  f (Real.log (2 * x - 2) / Real.log 2) + f (Real.log (1 - 1/2 * x) / Real.log 2) ≥ 0 :=
  sorry

end problem1_problem2_problem3_l770_770609


namespace find_x0_range_l770_770053

variable {x y x0 : ℝ}

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def angle_condition (x0 : ℝ) : Prop :=
  let OM := Real.sqrt (x0^2 + 3)
  OM ≤ 2

theorem find_x0_range (h1 : circle_eq x y) (h2 : angle_condition x0) :
  -1 ≤ x0 ∧ x0 ≤ 1 := 
sorry

end find_x0_range_l770_770053


namespace christina_options_and_sum_l770_770441

noncomputable def count_divisors_360 : ℕ :=
  ∏ p in (Multiset.toFinset (Nat.factors 360)).filter Nat.prime, (p.count 360.factors + 1)

noncomputable def sum_divisors_360 : ℕ :=
  let factors := (Multiset.toFinset (Nat.factors 360)).filter Nat.prime
  factors.fold (λ p acc => acc * (p^((Nat.factors 360).count p + 1) - 1) / (p - 1)) 1

theorem christina_options_and_sum : 
  count_divisors_360 = 24 ∧ sum_divisors_360 = 1170 :=
by
  sorry

end christina_options_and_sum_l770_770441


namespace hexagon_diagonals_parallel_intersect_single_point_l770_770813

-- Define the vertices and midpoints
variables {A B C A1 B1 C1 O K : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A1] [Inhabited B1] [Inhabited C1] [Inhabited O] [Inhabited K]

-- Define the hexagon property from given conditions
theorem hexagon_diagonals_parallel_intersect_single_point {Circle : Type} 
  (T1 T2 : Circle) [Inscribed T1 Circle] [Inscribed T2 Circle] 
  (vertices_T1 : Set Point) (vertices_T2 : Set Point)
  (T1_is_triangle : is_triangle vertices_T1) 
  (T2_is_triangle : is_triangle vertices_T2)
  (midpoints_condition : ∀ (A B C : Point), midpoint (arc_circle A C) = A1 ∧ midpoint (arc_circle B C) = B1 ∧ midpoint (arc_circle A B) = C1)
  (angle_bisectors_condition : ∀ (A B C : Line), bisector A A1 ∧ bisector B B1 ∧ bisector C C1)
  (incenter : is_incenter O [bisector A A1, bisector B B1, bisector C C1])
  (intersection_points : ∀ (K L M : Point) [Line K [L] ∧ Line K [M]], intersect_single_point K L M)
  (parallel_condition : ∀ (a b : Line), parallel a b → parallel b a)
: ∀ (d1 d2 d3 : Line) [Diagonal d1 [d2] ∧ Diagonal d2 [d3]], parallel d1 T1 ∧ intersect_single_point d1 d2 d3 :=
sorry

end hexagon_diagonals_parallel_intersect_single_point_l770_770813


namespace angle_between_tangents_l770_770869

-- Given side length ratio and definition of the variables
variables {x : ℝ}
def side_of_square := 6 * x
def diameter_of_circle := 10 * x
def radius_of_circle := 5 * x

-- Center of circle and its position relative to the square
variables {O D : Point}
def OP := (side_of_square / 2 : ℝ)
def DO := x * sqrt 109
def sine_angle_KDO := radius_of_circle / DO

-- Lean 4 statement for the given problem
theorem angle_between_tangents (h1 : O.x ≠ D.x) (h2 : O.y ≠ D.y) :
  2 * Real.arcsin (sine_angle_KDO) = 2 * Real.arcsin (5 / sqrt 109) :=
by
  sorry

end angle_between_tangents_l770_770869


namespace Tony_change_l770_770342

structure Conditions where
  bucket_capacity : ℕ := 2
  sandbox_depth : ℕ := 2
  sandbox_width : ℕ := 4
  sandbox_length : ℕ := 5
  sand_weight_per_cubic_foot : ℕ := 3
  water_per_4_trips : ℕ := 3
  water_bottle_cost : ℕ := 2
  water_bottle_volume: ℕ := 15
  initial_money : ℕ := 10

theorem Tony_change (conds : Conditions) : 
  let volume_sandbox := conds.sandbox_depth * conds.sandbox_width * conds.sandbox_length
  let total_weight_sand := volume_sandbox * conds.sand_weight_per_cubic_foot
  let trips := total_weight_sand / conds.bucket_capacity
  let drinks := trips / 4
  let total_water := drinks * conds.water_per_4_trips
  let bottles_needed := total_water / conds.water_bottle_volume
  let total_cost := bottles_needed * conds.water_bottle_cost
  let change := conds.initial_money - total_cost
  change = 4 := by
  /- calculations corresponding to steps that are translated from the solution to show that the 
     change indeed is $4 -/
  sorry

end Tony_change_l770_770342


namespace polynomial_inequality_solution_l770_770940

theorem polynomial_inequality_solution (x : ℝ) :
  (x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29) →
  x^3 - 12 * x^2 + 36 * x + 8 > 0 :=
by
  sorry

end polynomial_inequality_solution_l770_770940


namespace vector_opposite_direction_and_magnitude_l770_770214

theorem vector_opposite_direction_and_magnitude
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (-1, 2)) 
  (h2 : ∃ k : ℝ, k < 0 ∧ b = k • a) 
  (hb : ‖b‖ = Real.sqrt 5) :
  b = (1, -2) :=
sorry

end vector_opposite_direction_and_magnitude_l770_770214


namespace distance_from_altitudes_equal_l770_770667

-- We declare the variables involved in the problem:
variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]
variables (A B C A' B' O : α)

-- We assume all the given conditions:
def is_acute_triangle (A B C : α) : Prop := sorry
def is_circumcenter (O : α) (A B C : α) : Prop := sorry
def is_altitude (A' : α) (A B C : α) : Prop := sorry
def dist_to_line (P L : α) : ℝ := sorry -- placeholder for the distance function

-- The geometry theorem to be proven, translated to Lean:
theorem distance_from_altitudes_equal
  (h_acute : is_acute_triangle A B C)
  (h_circumcenter : is_circumcenter O A B C)
  (h_altitudes_AA' : is_altitude A' A B C)
  (h_altitudes_BB' : is_altitude B' B A C) :
  dist_to_line A' (line_through B O) = dist_to_line B' (line_through A O) :=
sorry

end distance_from_altitudes_equal_l770_770667


namespace hexagon_extension_property_l770_770286

-- Define points on the sides of a regular hexagon
variables {A₁ A₂ A₃ A₄ A₅ A₆ : Point}
-- Define the regular hexagon as well as the long diagonals
variables {Hexagon : RegularHexagon}
variables {long_diagonal : Hexagon.long_diagonals}

-- Define conditions: points A₁ to A₆ being on the sides of Hexagon and extensions of certain segments forming a triangle
def points_on_hexagon_sides (Hexagon : RegularHexagon) : Prop :=
  A₁ ∈ Hexagon.sides ∧ A₂ ∈ Hexagon.sides ∧ A₃ ∈ Hexagon.sides ∧ A₄ ∈ Hexagon.sides ∧
  A₅ ∈ Hexagon.sides ∧ A₆ ∈ Hexagon.sides

def extensions_form_triangle (Hexagon : RegularHexagon) (A₁ A₂ A₃ A₄ A₅ A₆ : Point) : Prop :=
  let S₁ := Hexagon.line_extension_intersection A₅ A₆ A₁ A₂ in
  let S₃ := Hexagon.line_extension_intersection A₁ A₂ A₃ A₄ in
  let S₅ := Hexagon.line_extension_intersection A₃ A₄ A₅ A₆ in
  S₁ ∈ Hexagon.long_diagonals ∧ S₃ ∈ Hexagon.long_diagonals ∧ S₅ ∈ Hexagon.long_diagonals

-- Main proof statement
theorem hexagon_extension_property (Hexagon : RegularHexagon) (A₁ A₂ A₃ A₄ A₅ A₆ : Point)
  (H₁ : points_on_hexagon_sides Hexagon)
  (H₂ : extensions_form_triangle Hexagon A₁ A₂ A₃ A₄ A₅ A₆) :
  extensions_form_triangle Hexagon A₂ A₃ A₄ A₅ A₆ A₁ :=
sorry

end hexagon_extension_property_l770_770286


namespace circle_area_below_line_l770_770016

theorem circle_area_below_line (x y : ℝ)
    (h1 : x^2 - 6*x + y^2 - 14*y + 33 = 0)
    (h2 : ∀ x, y = 7) :
    let r := 5 in
    let area := real.pi * r^2 in
    (area / 2) = (25 * real.pi) / 2 :=
sorry

end circle_area_below_line_l770_770016


namespace sqrt_inequality_l770_770014

theorem sqrt_inequality (a : ℝ) (h : a > 1) : sqrt (a + 1) + sqrt (a - 1) < 2 * sqrt a :=
  sorry

end sqrt_inequality_l770_770014


namespace least_tries_equals_2014_l770_770283

-- Defining that Peter has 2022 magnetic railroad cars of two types
def total_cars : ℕ := 2022

-- Defining the function to calculate the sum of the greatest integer less than
-- or equal to 2022 divided by 2 to the power of n for all n greater than 0
def least_tries_required : ℕ :=
  ∑ n in (Finset.range (Nat.ceil (log 2 2022))), (2022 / 2^n)

-- Stating that the least number of tries required is 2014
theorem least_tries_equals_2014 : least_tries_required = 2014 :=
by
  sorry

end least_tries_equals_2014_l770_770283


namespace a_gt_b_neither_sufficient_nor_necessary_l770_770261

theorem a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) := 
sorry

end a_gt_b_neither_sufficient_nor_necessary_l770_770261


namespace round_repeating_decimal_to_nearest_hundredth_l770_770752

theorem round_repeating_decimal_to_nearest_hundredth :
  ∀ x y : ℝ, (x = 75.3636363636) → y = 75.37 → round (x * 100) = y * 100 :=
by
  intro x y h₁ h₂
  rw h₁
  rw h₂
  sorry


end round_repeating_decimal_to_nearest_hundredth_l770_770752


namespace distance_blown_by_storm_l770_770739

-- Definitions based on conditions
def speed : ℤ := 30
def time_travelled : ℤ := 20
def distance_travelled := speed * time_travelled
def total_distance := 2 * distance_travelled
def fractional_distance_left := total_distance / 3

-- Final statement to prove
theorem distance_blown_by_storm : distance_travelled - fractional_distance_left = 200 := by
  sorry

end distance_blown_by_storm_l770_770739


namespace smaller_tetrahedron_volume_ratio_tetrahedron_volume_ratio_sum_combined_tetrahedron_volume_proof_l770_770663

theorem smaller_tetrahedron_volume_ratio (s₁ s₂ : ℕ) (h_ratio : s₁ / s₂ = 2 / 3) : 
  (s₁ ^ 3 / s₂ ^ 3 = 8 / 27) :=
by
  -- Given h_ratio, proving the volume ratio as follows
  sorry

theorem tetrahedron_volume_ratio_sum (m n : ℕ) (h_m : m = 8) (h_n : n = 27) (h_coprime : Nat.coprime m n) :
  m + n = 35 :=
by
  rw [h_m, h_n]
  norm_num
  sorry

-- Combining the two results to show
theorem combined_tetrahedron_volume_proof (s₁ s₂ : ℕ) (h_ratio : s₁ / s₂ = 2 / 3) : ∃ (m n : ℕ), m = 8 ∧ n = 27 ∧ Nat.coprime m n ∧ m + n = 35 :=
by
  use 8, 27
  split
  exact rfl
  split
  exact rfl
  split
  apply Nat.coprime.symm
  norm_num
  norm_num
  sorry

end smaller_tetrahedron_volume_ratio_tetrahedron_volume_ratio_sum_combined_tetrahedron_volume_proof_l770_770663


namespace sum_cubes_mod_5_l770_770461

-- Define the function sum of cubes up to n
def sum_cubes (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 1) ^ 3)

-- Define the given modulo operation
def modulo (a b : ℕ) : ℕ := a % b

-- The primary theorem to prove
theorem sum_cubes_mod_5 : modulo (sum_cubes 50) 5 = 0 := sorry

end sum_cubes_mod_5_l770_770461


namespace sum_of_cubes_mod_l770_770477

theorem sum_of_cubes_mod (S : ℕ) (h : S = ∑ n in Finset.range 51, n^3) : S % 5 = 0 :=
sorry

end sum_of_cubes_mod_l770_770477


namespace volume_small_pyramid_eq_27_60_l770_770888

noncomputable def volume_of_smaller_pyramid (base_edge : ℝ) (slant_edge : ℝ) (height_above_base : ℝ) : ℝ :=
  let total_height := Real.sqrt ((slant_edge ^ 2) - ((base_edge / (2 * Real.sqrt 2)) ^ 2))
  let smaller_pyramid_height := total_height - height_above_base
  let scale_factor := (smaller_pyramid_height / total_height)
  let new_base_edge := base_edge * scale_factor
  let new_base_area := (new_base_edge ^ 2) * 2
  (1 / 3) * new_base_area * smaller_pyramid_height

theorem volume_small_pyramid_eq_27_60 :
  volume_of_smaller_pyramid (10 * Real.sqrt 2) 12 4 = 27.6 :=
by
  sorry

end volume_small_pyramid_eq_27_60_l770_770888


namespace train_b_speed_l770_770013

/-- Two trains, A and B, start simultaneously from two stations 480 kilometers apart and meet after 2.5 hours. 
Train A travels at a speed of 102 kilometers per hour. What is the speed of train B in kilometers per hour? -/
theorem train_b_speed (d t : ℝ) (speedA speedB : ℝ) (h1 : d = 480) (h2 : t = 2.5) (h3 : speedA = 102)
  (h4 : speedA * t + speedB * t = d) : speedB = 90 := 
by
  sorry

end train_b_speed_l770_770013


namespace two_circles_tangent_tangent_line_AC_length_l770_770811

theorem two_circles_tangent_tangent_line_AC_length :
  ∀ (O1 O2 A B C : Point) (r1 r2 : ℝ) (tangent_line : Line) (tangent_point : Point),
    r1 = 5 ∧ r2 = 4 ∧
    T (Circle O1 r1) (Circle O2 r2) ∧
    T C tangent_point ∧ A B = B C ∧
    tangent_point ∈ (Circle O2 r2) ∧ (intersect_points tangent_line (Circle O2 r2) = [B, C]) →
    dist A C = 12 :=
by
  intros O1 O2 A B C r1 r2 tangent_line tangent_point
    h.1 h.2 h.3 h.4 h.5 h.6 h.7

  -- Proof here
  sorry

end two_circles_tangent_tangent_line_AC_length_l770_770811


namespace find_hypotenuse_square_l770_770442

-- Define the polynomial and its roots
variable {p q r : ℂ}
variable (s t : ℂ)
def Q (z : ℂ) : ℂ := z^3 + s * z + t

-- Conditions: |p|² + |q|² + |r|² = 300, and p, q, r form a right triangle
axiom sum_of_squares : |p|^2 + |q|^2 + |r|^2 = 300
axiom roots_form_right_triangle : true -- replace this with a more rigorous definition if available

-- The goal is to prove k² = 450, where k is the hypotenuse of the right triangle
theorem find_hypotenuse_square : (∃ k : ℝ, k^2 = 450) :=
by
  sorry

end find_hypotenuse_square_l770_770442


namespace quadratic_roots_condition_l770_770507

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l770_770507


namespace sqrt_fourth_root_nearest_tenth_l770_770929

theorem sqrt_fourth_root_nearest_tenth :
  (Real.sqrt (Real.root 4 0.000001)) = 0.2 :=
by
  -- proof steps would go here
  sorry

end sqrt_fourth_root_nearest_tenth_l770_770929
