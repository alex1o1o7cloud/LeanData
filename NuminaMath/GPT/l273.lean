import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Factorization.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSeries
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.TimesContDiff
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Geometry.EuclideanSphere

namespace nicolai_peaches_6_pounds_l273_273568

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end nicolai_peaches_6_pounds_l273_273568


namespace sin_theta_value_l273_273686

theorem sin_theta_value (θ : ℝ) (h1 : 12 * Real.cot θ = 4 * Real.sin θ) (h2 : 0 < θ) (h3 : θ < Real.pi) :
  Real.sin θ = Real.sqrt ((-14 - 6 * Real.sqrt 13) / 4) :=
by
  sorry

end sin_theta_value_l273_273686


namespace floor_multiplication_l273_273198

def Floor (x : ℕ) : ℕ :=
  if h : x > 1 then (2*x) + (2*x - 1) + (2*x - 2) + ... + 2 + 1 else 0

-- We need to define the Floor function by summing over the elements correctly.

def sum_helper (n : ℕ) : ℕ := 
  (n * (n + 1)) / 2

def Floor (x : ℕ) : ℕ :=
  if x > 1 then sum_helper (2*x) else 0

theorem floor_multiplication : (Floor 2) * (Floor 4) = 360 := by
  sorry

end floor_multiplication_l273_273198


namespace units_digit_k_squared_plus_two_to_k_is_7_l273_273860

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_k_squared_plus_two_to_k_is_7 :
  let k := (2008^2 + 2^2008) in
  units_digit (k^2 + 2^k) = 7 :=
by
  let k := (2008^2 + 2^2008)
  sorry

end units_digit_k_squared_plus_two_to_k_is_7_l273_273860


namespace boat_length_is_three_l273_273214

-- Define the given conditions
def breadth_of_boat : ℝ := 2
def sink_height : ℝ := 0.01
def mass_of_man : ℝ := 60
def density_of_water : ℝ := 1000
def gravity : ℝ := 9.81

-- Statement of the theorem
theorem boat_length_is_three :
  let V := mass_of_man * gravity / (density_of_water * gravity) in
  let length := V / (breadth_of_boat * sink_height) in
  length = 3 :=
by
  -- This is where the proof would go
  sorry

end boat_length_is_three_l273_273214


namespace calculate_seasons_l273_273194

theorem calculate_seasons :
  ∀ (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days : ℕ),
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days = 30 →
  (episodes_per_day * days) / episodes_per_season = 3 :=
by
  intros episodes_per_season episodes_per_day days h_eps h_epd h_d
  sorry

end calculate_seasons_l273_273194


namespace median_length_from_A_l273_273871

noncomputable def median_length (a : ℝ) (B C : E) [InnerProductSpace ℝ E] : ℝ :=
  let D := (B + C) / 2 in
  (a * real.sqrt 3) / 2

theorem median_length_from_A {a : ℝ} {A B C : E} [InnerProductSpace ℝ E]
  (hBC : ∥B - C∥ = a)
  (h_angle_BAC : angle B A C = 35)
  (h_angle_BOC : angle B (A + (B + C) / 2) C = 145) :
  median_length a B C = a * real.sqrt 3 / 2 :=
by
  sorry

end median_length_from_A_l273_273871


namespace complex_abs_eq_sqrt_two_div_two_l273_273373

theorem complex_abs_eq_sqrt_two_div_two (a : ℝ) (h : (a + (0:ℂ).im * I) / (2 - I) = 0 + I * _) : 
  ∥(1/2 : ℂ) + (a + I) / (2 - I)∥ = (Real.sqrt 2) / 2 :=
by sorry

end complex_abs_eq_sqrt_two_div_two_l273_273373


namespace triangle_side_length_l273_273471

-- We are given these conditions
variables {A B C P : Type} [InnerProductSpace ℝ P]
variable [FiniteDimensional ℝ P]

-- Define points
variables (A B C : P)
variable (P : P)

-- Given conditions
axiom inscribed_circle (r : ℝ) : r = 1
axiom vector_condition : 3 • (P - A) + 4 • (P - B) + 5 • (P - C) = 0

-- Prove that the length of side AB is sqrt(2)
theorem triangle_side_length :
  dist A B = Real.sqrt 2 :=
sorry

end triangle_side_length_l273_273471


namespace function_cannot_be_decreasing_if_f1_lt_f2_l273_273771

variable (f : ℝ → ℝ)

theorem function_cannot_be_decreasing_if_f1_lt_f2
  (h : f 1 < f 2) : ¬ (∀ x y, x < y → f y < f x) :=
by
  sorry

end function_cannot_be_decreasing_if_f1_lt_f2_l273_273771


namespace cube_faces_edges_vertices_sum_l273_273597

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end cube_faces_edges_vertices_sum_l273_273597


namespace wicket_keeper_age_difference_l273_273926

theorem wicket_keeper_age_difference 
  (team_size : ℕ)
  (average_age_team : ℕ)
  (remaining_players_size : ℕ)
  (average_age_remaining_players : ℕ)
  (captain_age : ℕ)
  (total_age_team : ℕ = team_size * average_age_team)
  (total_age_remaining_players : ℕ = remaining_players_size * average_age_remaining_players)
  (combined_age : ℕ = total_age_team - total_age_remaining_players)
  (avg_wicket_keeper_age: ℕ = combined_age - captain_age) 
  (average_age_team : ℕ = 23)
  (a: team_size = 11) 
  (b: remaining_players_size = 9) 
  (c: average_age_remaining_players = 22)
  (d: captain_age = 23):
  avg_wicket_keeper_age - average_age_team = 9 := sorry

end wicket_keeper_age_difference_l273_273926


namespace f_is_monotonically_increasing_range_of_a_for_local_max_at_1_l273_273770

-- Proof Problem 1: 
-- Prove that when a=1, f(x) = ae^{x-1} + \ln x - (a+1)x is monotonically increasing on (0, +∞)
theorem f_is_monotonically_increasing {x : ℝ} (h : 0 < x) : 
  let a := 1
  in ∀ (x : ℝ), 0 < x -> monotone (λ x, a * exp (x - 1) + log x - (a + 1) * x) :=
sorry

-- Proof Problem 2: 
-- If x=1 is a local maximum point of the function f(x), find the range of real number a.
theorem range_of_a_for_local_max_at_1 (a : ℝ) :
  (∀ x : ℝ, 0 < x -> deriv (λ x, a * exp (x - 1) + log x - (a + 1) * x) 1 = 0) → 
  a ∈ set.Ico (ℝ) (-∞) 1 :=
sorry

end f_is_monotonically_increasing_range_of_a_for_local_max_at_1_l273_273770


namespace third_side_length_l273_273015

/-- Given two sides of a triangle with lengths 4cm and 9cm, prove that the valid length of the third side must be 9cm. -/
theorem third_side_length (a b c : ℝ) (h₀ : a = 4) (h₁ : b = 9) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → (c = 9) :=
by {
  sorry
}

end third_side_length_l273_273015


namespace men_joined_l273_273626

theorem men_joined (M D : ℕ) (D' : ℝ) (M_val : M = 1500) (D_val : D = 17) (D'_val : D' = 14.010989010989011) : 
  ∃ X : ℤ, 1500 * 17 = (1500 + X) * 14.010989010989011 ∧ X = 317 :=
by
  -- definitions as per the conditions given
  let M := 1500
  let D := 17
  let D' := 14.010989010989011
  
  -- sorry since we skip proving for this exercise
  sorry

end men_joined_l273_273626


namespace find_lowest_score_l273_273921

-- Conditions:
def total_average : ℝ := 81.6
def num_students : ℕ := 5
def total_sum : ℝ := num_students * total_average
def other_scores_sum : ℝ := 88 + 84 + 76
def H_plus_L : ℝ := total_sum - other_scores_sum
def avg_without_highest (H : ℝ) : ℝ := (total_sum - H) / (num_students - 1)
def avg_without_lowest (L : ℝ) : ℝ := (total_sum - L) / (num_students - 1)
def average_difference_condition (H L : ℝ) : Prop := avg_without_highest H + 6 = avg_without_lowest L

-- Proof Problem (Question == Answer):
theorem find_lowest_score (H L : ℝ) (h1 : H + L = H_plus_L) (h2 : average_difference_condition H L) : L = 68 :=
sorry

end find_lowest_score_l273_273921


namespace part1_part2_part3_l273_273372

noncomputable def f (x a : ℝ) := x^2 * (x - a)
noncomputable def g (x m a : ℝ) := f x a + m / (x - 1)
noncomputable def f' (x a : ℝ) := (3:ℝ) * x ^ 2 - 2 * a * x
noncomputable def g' (x m a : ℝ) := 3 * x ^ 2 - 6 * x - m / (x - 1)^2

theorem part1 (a : ℝ) (h : 3 * (2:ℝ) ^ 2 - 2 * a * 2 = 0) :
  a = 3 := 
by {
  sorry
}

theorem part2 (a : ℝ) (h₀ : a > 0) :
  let fa := f 2 a in
  (0 < a ∧ a < 3 → ∀x ∈ set.Icc (0:ℝ) 2, f x a ≥ - (4 / (27:ℝ)) * a ^ 3) ∧
  (a ≥ 3 → ∀x ∈ set.Icc (0:ℝ) 2, f x a ≥ 8 - 4 * a) :=
by {
  sorry
}

theorem part3 (m : ℝ) (h₁ : 3 * (3:ℝ) ^ 2 - 6 * 3 - m / (3 - 1)^2 ≥ 0) :
  m ≤ 36 :=
by {
  sorry
}

end part1_part2_part3_l273_273372


namespace sum_of_faces_edges_vertices_l273_273594

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end sum_of_faces_edges_vertices_l273_273594


namespace total_net_loss_l273_273611

theorem total_net_loss 
  (P_x P_y : ℝ)
  (h1 : 1.2 * P_x = 25000)
  (h2 : 0.8 * P_y = 25000) :
  (25000 - P_x) - (P_y - 25000) = -2083.33 :=
by 
  sorry

end total_net_loss_l273_273611


namespace find_ratio_of_square_to_circle_radius_l273_273810

def sector_circle_ratio (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) : Prop :=
  (R = (5 * a * sqrt2) / 2) →
  (r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) →
  (a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2))

theorem find_ratio_of_square_to_circle_radius
  (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) (h1 : R = (5 * a * sqrt2) / 2)
  (h2 : r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) :
  a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2) :=
  sorry

end find_ratio_of_square_to_circle_radius_l273_273810


namespace largest_power_of_2_l273_273316

noncomputable def q : ℝ := ∑ k in finset.range 7, (2 * (k + 1) - 1) * real.log (k + 2)

theorem largest_power_of_2 (q : ℝ) (h : q = ∑ k in finset.range 7, (2 * (k + 1) - 1) * real.log (k + 2)) :
  ∃ n : ℕ, e^q = (2^59) * n ∧ (¬ ∃ m : ℕ, m > 59 ∧ 2^m ∣ e^q) :=
sorry

end largest_power_of_2_l273_273316


namespace population_after_two_years_l273_273129

theorem population_after_two_years :
  let initial_population := 415600 in
  let increased_population := initial_population + (0.25 * initial_population) in
  let final_population := increased_population - (0.30 * increased_population) in
  final_population = 363650 :=
by
  let initial_population := 415600
  let increased_population := initial_population + (0.25 * initial_population)
  let final_population := increased_population - (0.30 * increased_population)
  have h : final_population = 363650
  sorry

end population_after_two_years_l273_273129


namespace farmer_has_42_cows_left_l273_273233

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end farmer_has_42_cows_left_l273_273233


namespace common_difference_arithmetic_l273_273874

noncomputable def common_difference (a b c : ℝ) : ℝ :=
(log c a, log b c, log a b).seq_diff

theorem common_difference_arithmetic (a b c : ℝ) 
  (h₀ : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h₁ : (a^2, b^2, c^2).is_geo_seq) : 
  common_difference a b c = 3 / 2 := 
sorry

end common_difference_arithmetic_l273_273874


namespace parabola_y_intercepts_l273_273785

theorem parabola_y_intercepts :
  let f : ℝ → ℝ := λ y, 3 * y^2 - 5 * y + 2
  ∃ y1 y2 : ℝ, f y1 = 0 ∧ f y2 = 0 ∧ y1 ≠ y2 :=
by
  sorry

end parabola_y_intercepts_l273_273785


namespace intersection_eq_l273_273370

open Set

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 5 }
def B : Set ℝ := { -1, 2, 3, 6 }

-- State the proof problem
theorem intersection_eq : A ∩ B = {2, 3} := 
by 
-- placeholder for the proof steps
sorry

end intersection_eq_l273_273370


namespace min_value_frac_l273_273018

theorem min_value_frac (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ z : ℝ, z = (2 * x * y) / (x + y - 1) ∧ z ≥ 1 - real.sqrt 2 := 
sorry

end min_value_frac_l273_273018


namespace problem1_problem2_l273_273397

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Conditions:
axiom condition1 : sin (α + π / 6) = sqrt 10 / 10
axiom condition2 : cos (α + π / 6) = 3 * sqrt 10 / 10
axiom condition3 : tan (α + β) = 2 / 5

-- Prove:
theorem problem1 : sin (2 * α + π / 6) = (3 * sqrt 3 - 4) / 10 :=
by sorry

theorem problem2 : tan (2 * β - π / 3) = 17 / 144 :=
by sorry

end problem1_problem2_l273_273397


namespace eq1_solution_eq2_solution_l273_273112

theorem eq1_solution (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) :=
sorry

theorem eq2_solution (x : ℝ) : (3 * x^2 - 6 * x + 2 = 0) ↔ (x = 1 + (Real.sqrt 3) / 3 ∨ x = 1 - (Real.sqrt 3) / 3) :=
sorry

end eq1_solution_eq2_solution_l273_273112


namespace nicolai_peaches_pounds_l273_273571

-- Condition definitions
def total_fruit_pounds : ℕ := 8
def mario_oranges_ounces : ℕ := 8
def lydia_apples_ounces : ℕ := 24
def ounces_to_pounds (ounces: ℕ) : ℚ := ounces / 16 

-- Statement we want to prove
theorem nicolai_peaches_pounds :
  let mario_oranges_pounds := ounces_to_pounds mario_oranges_ounces,
      lydia_apples_pounds := ounces_to_pounds lydia_apples_ounces,
      eaten_by_m_and_l := mario_oranges_pounds + lydia_apples_pounds,
      nicolai_peaches_pounds := total_fruit_pounds - eaten_by_m_and_l in
  nicolai_peaches_pounds = 6 :=
by
  sorry

end nicolai_peaches_pounds_l273_273571


namespace prism_volume_l273_273940

theorem prism_volume
  (Q : ℝ)
  (hQ : Q > 0) -- Ensure Q is positive for real context
  (height_eq_lateral_edge : ∃ x : ℝ, x = sqrt Q)
  (cross_section_area : Q = sqrt Q * sqrt Q) :
  V = Q * sqrt (Q / 3) :=
by sorry

end prism_volume_l273_273940


namespace number_of_correct_conclusions_l273_273671

noncomputable def f (x a : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem number_of_correct_conclusions :
  (card (set_of (λ (n : ℕ), 
    (n = 1 → ∃ a > 0, ∀ x > 0, f x a ≥ 0) ∧
    (n = 2 → ∃ a > 0, ∃ x > 0, f x a ≤ 0) ∧
    (n = 3 → ∀ a > 0, ∀ x > 0, f x a ≥ 0) ∧
    (n = 4 → ∀ a > 0, ∃ x > 0, f x a ≤ 0)))) = 3 := 
sorry

end number_of_correct_conclusions_l273_273671


namespace problem1_problem2_l273_273779

-- Definitions based on conditions
def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)
def is_non_collinear (a b : ℝ × ℝ) : Prop := a ≠ b ∧ a ≠ (0,0) ∧ b ≠ (0,0)

-- Problem 1
theorem problem1 (α β : ℝ) (h₁ : is_non_collinear (vector_a α) (vector_b β)) :
  let a := vector_a α
  let b := vector_b β
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0 :=
sorry

-- Problem 2
theorem problem2 (α β : ℝ) (hα : α ∈ Ioo (-π/4) (π/4)) (hβ : β = π/4)
  (h_ab_norm : ∥(Real.cos α + Real.cos β, Real.sin α + Real.sin β)∥ = Real.sqrt (16/5)) :
  Real.sin α = -Real.sqrt 2 / 10 :=
sorry

end problem1_problem2_l273_273779


namespace friendship_configurations_l273_273658

-- Define the individuals in the group
def individuals := ["Alice", "Bob", "Carla", "David", "Evelyn", "Frank", "George"]

-- Define the number of individuals
def n : ℕ := 7

-- Condition: Each individual has the same number of internet friends
def same_friends (f : individual → list individual) : Prop :=
  ∀ i j, length (f i) = length (f j)

-- Proposition
theorem friendship_configurations :
  (∃ f : individual → list individual, same_friends f ∧ nonempty f ∧ f.size = 923) :=
sorry

end friendship_configurations_l273_273658


namespace vec_AB_eq_k_val_eq_l273_273037

-- Define points A and B in the Cartesian coordinate system
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)

-- Define the vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- The desired value of k when vector a is parallel to vector AB
def k (a : ℝ × ℝ) : ℝ := (a.2 * 3) / a.1

theorem vec_AB_eq : AB = (3, 5) := sorry

theorem k_val_eq (a : ℝ × ℝ) (h : a = (1, k a)) (h_parallel : a.2 / a.1 = 5 / 3) : k a = 5 / 3 :=
by {
  have prop : k a = (a.2 * 3) / a.1,
  {
    unfold k,
  },
  rw [←prop, h.2, h_parallel],
  linarith,
}

end vec_AB_eq_k_val_eq_l273_273037


namespace cos_100_eq_neg_sqrt_l273_273750

theorem cos_100_eq_neg_sqrt (a : ℝ) (h : Real.sin (80 * Real.pi / 180) = a) : 
  Real.cos (100 * Real.pi / 180) = -Real.sqrt (1 - a^2) := 
sorry

end cos_100_eq_neg_sqrt_l273_273750


namespace count_pos_int_log_condition_l273_273333

open Real

theorem count_pos_int_log_condition : 
  (∃ (S : Set ℕ), 
    (∀ x ∈ S, 30 < x ∧ x < 70 ∧ log 10 (x - 30 : ℝ) + log 10 (70 - x : ℝ) < 1) ∧ 
    S.toFinset.card = 18) := 
sorry

end count_pos_int_log_condition_l273_273333


namespace neither_sufficient_nor_necessary_l273_273085

variables {α : Type*} [linear_ordered_field α]

def P (a1 b1 c1 a2 b2 c2 : α) : Prop :=
  ∀ x : α, (a1 * x^2 + b1 * x + c1 > 0) ↔ (a2 * x^2 + b2 * x + c2 > 0)

-- The proposition Q is unspecified. Normally, we'd define Q explicitly based on the context of the problem.
-- For example, Q could be about a specific relation between coefficients of the inequalities.
-- Here, we introduce a general placeholder for Q, which should be filled with the specific relation once known.

def Q (a1 b1 c1 a2 b2 c2 : α) : Prop := sorry -- Specify Q based on context.

-- State the theorem based on the correct answer
theorem neither_sufficient_nor_necessary (a1 b1 c1 a2 b2 c2 : α) (hP : P a1 b1 c1 a2 b2 c2) (hQ : Q a1 b1 c1 a2 b2 c2) : 
  ¬ (hQ → hP) ∧ ¬ (hP → hQ) :=
sorry

end neither_sufficient_nor_necessary_l273_273085


namespace sum_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_dividing_factorial_10_l273_273588

theorem sum_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_dividing_factorial_10 :
  let pfs := ([(2, 8), (3, 4), (5, 2)] : List (ℕ × ℕ))  -- prime factors and their exponents of 10!
  let sqrt_pfs := pfs.map (λ ⟨p, e⟩, ⟨p, e / 2⟩)        -- taking the square root by halving exponents
  let sum_exponents := sqrt_pfs.sum (λ ⟨p, e⟩, e)       -- summing the exponents
  sum_exponents = 7 :=
by {
  sorry
}

end sum_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_dividing_factorial_10_l273_273588


namespace range_of_a_for_monotonicity_l273_273756

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f'' (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x - (1/3) * sin (2 * x) + a * sin x
noncomputable def g' (x : ℝ) (a : ℝ) : ℝ := 1 - (2/3) * cos (2 * x) + a * cos x

theorem range_of_a_for_monotonicity :
  (∀ x : ℝ, f'' x ≠ 0) →
  (∀ x : ℝ, f' x > 0) →
  (∀ x : ℝ, f[f(x) - 2017^x] = 2017) →
  (∀ x ∈ set.Icc (π / 3) π, g' x a ≥ 0) →
  -8/3 ≤ a ∧ a ≤ 1/3 :=
sorry

end range_of_a_for_monotonicity_l273_273756


namespace valid_sequences_tied_14_14_l273_273628

-- Define the sequences
def A_wins_sequences_count (n : ℕ) : ℕ :=
  ((nat.choose (2 * n - 2) (n - 1)) / n)

-- The problem statement: Prove the number of valid sequences when the game ends in a tie at 14:14
theorem valid_sequences_tied_14_14 :
  A_wins_sequences_count 14 = (1/14) * (nat.choose 26 13) := 
sorry

end valid_sequences_tied_14_14_l273_273628


namespace part_a_part_b_part_c_l273_273497

-- Part (a)
theorem part_a (X Y : Type) [RV X] [RV Y] [FiniteVar X] [FiniteVar Y] (h_independent : independent X Y) :
  D(X*Y) = D(X) * D(Y) + D(X) * (E(Y))^2 + D(Y) * (E(X))^2 :=
sorry

-- Part (b)
theorem part_b (X Y : Type) [RV X] [RV Y] [FiniteVar X] [FiniteVar Y] :
  cov(X, Y)^2 ≤ D(X) * D(Y) :=
sorry

-- Part (c)
theorem part_c (X Y : Type) (a b c d : ℝ) [RV X] [RV Y] :
  ρ(a*X + b, c*Y + d) = ρ(X, Y) * sign(a*c) :=
sorry

end part_a_part_b_part_c_l273_273497


namespace perpendicular_tangent_line_l273_273124

def f (x : ℝ) : ℝ := x - x^3 - 1

def tangent_slope_at_one : ℝ :=
  by
    let df := deriv f
    let slope_at_1 := df 1
    exact slope_at_1

def is_perpendicular (a : ℝ) : Prop :=
  let tangent_slope := tangent_slope_at_one
  let line_slope := -4 / a
  tangent_slope * line_slope = -1

theorem perpendicular_tangent_line (a : ℝ) : -8 = a ↔ is_perpendicular a := 
  sorry

end perpendicular_tangent_line_l273_273124


namespace green_minus_blue_difference_l273_273583

theorem green_minus_blue_difference :
  ∀ (original_blue_tile_count original_green_tile_count first_layer_border second_layer_border : ℕ),
    original_blue_tile_count = 20 →
    original_green_tile_count = 10 →
    first_layer_border = 6 * 3 →
    second_layer_border = 4 * 6 →
    (original_green_tile_count + (first_layer_border + second_layer_border)) - original_blue_tile_count = 32 :=
begin
  intros original_blue_tile_count original_green_tile_count first_layer_border second_layer_border,
  intros h1 h2 h3 h4,
  sorry
end 

end green_minus_blue_difference_l273_273583


namespace g_seven_l273_273550

-- Definition of the function g satisfying the conditions
variables {ℝ : Type} [linear_ordered_field ℝ]

def g (x : ℝ) : ℝ := sorry

-- Conditions
axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_six : g 6 = 7

-- Theorem to prove
theorem g_seven : g 7 = 49 / 6 :=
sorry

end g_seven_l273_273550


namespace initial_firecrackers_l273_273840

theorem initial_firecrackers (F : ℕ) 
  (h1 : F - 12 > 0)
  (h2 : (F - 12) * 1 / 6 ∈ ℕ)
  (h3 : 15 = (5 / 6) * (F - 12) / 2) : 
  F = 48 := 
sorry

end initial_firecrackers_l273_273840


namespace jebb_total_spent_l273_273055

theorem jebb_total_spent
  (cost_of_food : ℝ) (service_fee_rate : ℝ) (tip : ℝ)
  (h1 : cost_of_food = 50)
  (h2 : service_fee_rate = 0.12)
  (h3 : tip = 5) :
  cost_of_food + (cost_of_food * service_fee_rate) + tip = 61 := 
sorry

end jebb_total_spent_l273_273055


namespace radhika_total_games_l273_273533

-- Define the conditions
def giftsOnChristmas := 12
def giftsOnBirthday := 8
def alreadyOwned := (giftsOnChristmas + giftsOnBirthday) / 2
def totalGifts := giftsOnChristmas + giftsOnBirthday
def expectedTotalGames := totalGifts + alreadyOwned

-- Define the proof statement
theorem radhika_total_games : 
  giftsOnChristmas = 12 ∧ giftsOnBirthday = 8 ∧ alreadyOwned = 10 
  ∧ totalGifts = 20 ∧ expectedTotalGames = 30 :=
by 
  sorry

end radhika_total_games_l273_273533


namespace problem_l273_273729

variable {n : ℕ} (a b : ℕ → ℝ)

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = q * a n

noncomputable def partial_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in Finset.range n, a (i + 1)

noncomputable def arithmetic_sequence (c : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, c (n + 1) - c n = d

theorem problem
  (q : ℝ)
  (hq : q > 0)
  (ha1 : a 1 = 1)
  (hS3 : (a 1 + a 2 + a 3) = 7)
  (hb1 : b 1 = 0)
  (hb3 : b 3 = 1)
  (harith_seq : arithmetic_sequence (λ (n : ℕ), a n + b n) 4) :
  ∑ i in Finset.range n, (b (i + 1)) = n^2 - 2^n + 1 :=
sorry

end problem_l273_273729


namespace rational_approximation_bound_l273_273539

theorem rational_approximation_bound (c : ℝ) (h : c > 0) :
  ∀ (m n : ℤ), (1 ≤ n) → abs (2^(1 / 3 : ℝ) - m / n) > c / (n^3) :=
sorry

end rational_approximation_bound_l273_273539


namespace distance_between_lines_l273_273743

def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y - 1 = 0

theorem distance_between_lines : ∀ (x y : ℝ), 
  ∃ d : ℝ, d = real.abs (-1 - 1) / real.sqrt (1^2 + 1^2) ∧ d = √2 := 
sorry

end distance_between_lines_l273_273743


namespace sum_pairwise_distances_l273_273890

variable {α : Type*} [LinearOrderedField α] 

theorem sum_pairwise_distances (n : ℕ) (blue_points red_points : Fin n → α)
  (h_distinct_blue : ∀ i j, i ≠ j → blue_points i ≠ blue_points j)
  (h_distinct_red : ∀ i j, i ≠ j → red_points i ≠ red_points j) :
  ∑ i j in Finset.univ, if i ≠ j then |blue_points i - blue_points j| else 0 +
  ∑ i j in Finset.univ, if i ≠ j then |red_points i - red_points j| else 0 ≤
  ∑ i in Finset.univ, ∑ j in Finset.univ, |blue_points i - red_points j| :=
sorry

end sum_pairwise_distances_l273_273890


namespace largest_digit_divisible_by_6_l273_273984

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l273_273984


namespace triangle_area_proof_l273_273828

variable {A B C a b c : ℝ}
variable {s : ℝ}

-- Conditions for the problem
def triangle_angle_sum (A B C : ℝ) : Prop := A + B + C = 180
def sides_opposite_angles (a b c A B C : ℝ) : Prop := true
def given_equation (A B C : ℝ) : Prop := 2 * Real.sin A * Real.cos B + Real.sin B = 2 * Real.sin C

-- Derived conditions
def angle_A_sixty (A B C : ℝ) (h : given_equation A B C) : Prop := A = 60

def cosine_rule (a b c A : ℝ) : Prop := a^2 = b^2 + c^2 - 2 * b * c * Real.cos A
def side_a (a : ℝ) : Prop := a = 4 * Real.sqrt 3
def side_sum (b c : ℝ) : Prop := b + c = 8

-- Final condition: area calculation
def area (b c : ℝ) : ℝ := 1/2 * b * c * (Real.sqrt 3 / 2)

-- The theorem to be proved
theorem triangle_area_proof (A B C a b c : ℝ) (h1 : triangle_angle_sum A B C)
  (h2 : sides_opposite_angles a b c A B C)
  (h3 : given_equation A B C)
  (h4 : angle_A_sixty A B C h3)
  (h5 : side_a a)
  (h6 : side_sum b c)
  (h7 : cosine_rule a b c A) :
  area b c = 4 * Real.sqrt 3 / 3 := sorry

end triangle_area_proof_l273_273828


namespace pencils_count_l273_273142

theorem pencils_count (initial_pencils additional_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : additional_pencils = 100) : initial_pencils + additional_pencils = 215 :=
by sorry

end pencils_count_l273_273142


namespace Pria_drove_372_miles_l273_273897

theorem Pria_drove_372_miles (advertisement_mileage : ℕ) (tank_capacity : ℕ) (mileage_difference : ℕ) 
(h1 : advertisement_mileage = 35) 
(h2 : tank_capacity = 12) 
(h3 : mileage_difference = 4) : 
(advertisement_mileage - mileage_difference) * tank_capacity = 372 :=
by sorry

end Pria_drove_372_miles_l273_273897


namespace min_value_fraction_l273_273132

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) = 2) : 
  ∃ x : ℝ, x = (8 * a + b) / (a * b) ∧ x = 9 :=
by
  sorry

end min_value_fraction_l273_273132


namespace area_of_inscribed_triangle_l273_273255

theorem area_of_inscribed_triangle 
  {r : ℝ} (h_radius: r = 5)
  {sides_ratio : ℝ} (h_sides: sides_ratio = 5/12/13) :
  let x := 10 / 13 in 
  let area := 1/2 * (5 * x) * (12 * x) in 
  (area = 3000 / 169) ∧ ((3000 / 169: ℝ).round = 17.75) :=
sorry

end area_of_inscribed_triangle_l273_273255


namespace X_on_AO_l273_273547

open EuclideanGeometry

-- Definitions based on the conditions given
variables {A B C D E F G O K L X : Point}
-- Let Ω and Γ be circles with centers O and A respectively
variable (Ω : Circle (circumradius (triangle A B C)) O)
variable (Γ : Circle (radius (segment A D)) A)

-- Points D and E lie on segment BC such that B, D, E, C are distinct points on the line BC in that order
variable (D : Point)
variable (E : Point)
axiom h1 : Collinear [B, D, E, C]
axiom distinct_points : B ≠ D ∧ D ≠ E ∧ E ≠ C

-- Circle Γ intersects segment BC at points D and E
axiom intersectsBC : Γ ∩ (lineSegment B C) = {D, E}

-- Circles Ω and Γ intersect at points F and G such that A, F, B, C, G appear in that order on Ω
variable (F G : Point)
axiom order : Cyclic [A, F, B, C, G]

-- K is the other intersection of the circumcircle of triangle BDF with segment AB
noncomputable def circumcircle_BDF : Circle (circumradius (triangle B D F)) (circumcenter (triangle B D F))
variable (K : Point)
axiom K_intersection : (circumcircle_BDF ∩ (lineSegment A B)) = {B, K}
axiom K_not_B : K ≠ B

-- L is the other intersection point of the circumcircle of triangle CGE with segment CA
noncomputable def circumcircle_CGE : Circle (circumradius (triangle C G E)) (circumcenter (triangle C G E))
variable (L : Point)
axiom L_intersection : (circumcircle_CGE ∩ (lineSegment C A)) = {C, L}
axiom L_not_C : L ≠ C

-- Assume lines FK and GL intersect at point X
variable (X : Point)
axiom FK_GL_intersection : ∃ X, X ∈ (line F K) ∧ X ∈ (line G L)

-- Prove that point X lies on line AO
theorem X_on_AO : X ∈ (line A O) :=
sorry

end X_on_AO_l273_273547


namespace sin_odd_and_monotonic_l273_273660

-- Define the interval (0, π/2)
def interval (a b : ℝ) := {x : ℝ | a < x ∧ x < b}

-- Define odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define monotonically increasing function on an interval
def is_monotonic_increasing_on (f : ℝ → ℝ) (I : set ℝ) := 
  ∀ x y ∈ I, x < y → f x < f y

-- Define the sine function
def sin (x : ℝ) : ℝ := Real.sin x

-- The theorem to be proven
theorem sin_odd_and_monotonic : 
  is_odd sin ∧ is_monotonic_increasing_on sin (interval 0 (Real.pi / 2)) :=
sorry

end sin_odd_and_monotonic_l273_273660


namespace proof_problem_l273_273854

noncomputable def vectors (a b : ℝ) : Prop :=
∀ (a b : ℝ^3), a ≠ 0 ∧ b ≠ 0 ∧ ∥a + b∥ = ∥a∥ - ∥b∥ → ∃ (λ : ℝ), b = λ • a

theorem proof_problem (a b : ℝ^3) (h : a ≠ 0 ∧ b ≠ 0 ∧ ∥a + b∥ = ∥a∥ - ∥b∥) : 
  ∃ (λ : ℝ), b = λ • a :=
sorry

end proof_problem_l273_273854


namespace enclosed_area_l273_273535

-- Define the problem statement in Lean 4
theorem enclosed_area (AB CD BC DA : ℝ) (T : set (ℝ × ℝ)) (m : ℝ)
  (h1 : AB = 3) (h2 : CD = 3) (h3 : BC = 4) (h4 : DA = 4)
  (h5 : ∀ (P Q : ℝ × ℝ), P ∈ T → Q ∈ T → dist P Q = 3) :
  100 * m = 707 :=
sorry

end enclosed_area_l273_273535


namespace calculate_selling_prices_l273_273238

noncomputable def selling_prices
  (cost1 cost2 cost3 : ℝ) (profit1 profit2 profit3 : ℝ) : ℝ × ℝ × ℝ :=
  let selling_price1 := cost1 + (profit1 / 100) * cost1
  let selling_price2 := cost2 + (profit2 / 100) * cost2
  let selling_price3 := cost3 + (profit3 / 100) * cost3
  (selling_price1, selling_price2, selling_price3)

theorem calculate_selling_prices :
  selling_prices 500 750 1000 20 25 30 = (600, 937.5, 1300) :=
by
  sorry

end calculate_selling_prices_l273_273238


namespace average_of_even_numbers_is_13_l273_273922

theorem average_of_even_numbers_is_13 : 
  ∃ n : ℕ, (∑ k in finset.range n, 2*(k+1)) / n = 13 :=
by
  sorry

end average_of_even_numbers_is_13_l273_273922


namespace number_of_ordered_pairs_l273_273682

theorem number_of_ordered_pairs (h : ∀ (a b : ℕ), (2 ≤ a ∧ a ≤ 2005) ∧ (2 ≤ b ∧ b ≤ 2005) → log a b + 6 * log b a = 5 →  True) : 
  ∃ (n : ℕ), n = 54 := 
sorry

end number_of_ordered_pairs_l273_273682


namespace length_of_AC_l273_273816

-- Define the points and lengths
variables (A B C D : ℝ)
variables (AB DC AD : ℝ)

-- State the conditions
axiom hAB : AB = 15
axiom hDC : DC = 24
axiom hAD : AD = 7

-- Define the proof statement
theorem length_of_AC :
  ∃ (AC : ℝ), AC ≈ 26.8 :=
by
  sorry

end length_of_AC_l273_273816


namespace solve_basketball_points_l273_273805

noncomputable def y_points_other_members (x : ℕ) : ℕ :=
  let d_points := (1 / 3) * x
  let e_points := (3 / 8) * x
  let f_points := 18
  let total := x
  total - d_points - e_points - f_points

theorem solve_basketball_points (x : ℕ) (h1: x > 0) (h2: ∃ y ≤ 24, y = y_points_other_members x) :
  ∃ y, y = 21 :=
by
  sorry

end solve_basketball_points_l273_273805


namespace product_of_invertible_labels_is_3_l273_273939

-- Definitions for the functions' invertibility
def is_invertible (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f a = f b → a = b

-- Definitions of functions based on the conditions provided.
def f1 (x : ℝ) : ℝ :=
  if x = -6 then 3 else
  if x = -5 then 2 else
  if x = -4 then 1 else
  if x = -3 then 0 else
  if x = -2 then 1 else
  if x = -1 then 2 else
  if x = 0  then 3 else
  if x = 1  then 4 else 
  0 -- default value for places out of domain, though not used

def f2 (x : ℝ) : ℝ := -sin x

def f3 (x : ℝ) : ℝ := 2 / x

-- Proof problem statement
theorem product_of_invertible_labels_is_3 :
  let labels := [1, 2, 3] in
  let functions := [(f1, 1), (f2, 2), (f3, 3)] in
  (∏ (label in functions.filter (λ f_label, is_invertible f_label.1)).map (prod.snd)) = 3 :=
by {
  sorry,
}

end product_of_invertible_labels_is_3_l273_273939


namespace minimum_sum_of_distances_squared_l273_273748

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 0 }
def B : Point := { x := 2, y := 0 }

-- Define the moving point P on the circle
def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

-- Distance squared between two points
def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the sum of squared distances from P to points A and B
def sum_distances_squared (P : Point) : ℝ :=
  dist_squared P A + dist_squared P B

-- Statement of the proof problem
theorem minimum_sum_of_distances_squared :
  ∃ P : Point, on_circle P ∧ sum_distances_squared P = 26 :=
sorry

end minimum_sum_of_distances_squared_l273_273748


namespace smallest_sum_proof_l273_273158

theorem smallest_sum_proof (N : ℕ) (p : ℝ) (h1 : 6 * N = 2022) (hp : p > 0) : (N * 1 = 337) :=
by 
  have hN : N = 2022 / 6 := by 
    sorry
  exact hN

end smallest_sum_proof_l273_273158


namespace degrees_to_radians_neg_210_l273_273672

theorem degrees_to_radians_neg_210 :
  -210 * (Real.pi / 180) = - (7 / 6) * Real.pi :=
by
  sorry

end degrees_to_radians_neg_210_l273_273672


namespace units_digit_of_k_squared_plus_2_k_l273_273863

def k := 2008^2 + 2^2008

theorem units_digit_of_k_squared_plus_2_k : 
  (k^2 + 2^k) % 10 = 7 :=
by {
  -- The proof will be inserted here
  sorry
}

end units_digit_of_k_squared_plus_2_k_l273_273863


namespace cone_volume_l273_273957

theorem cone_volume (V_cyl : ℝ) (d : ℝ) (π : ℝ) (V_cyl_eq : V_cyl = 81 * π) (h_eq : 2 * (d / 2) = 2 * d) :
  ∃ (V_cone : ℝ), V_cone = 27 * π * (6 ^ (1/3)) :=
by 
  sorry

end cone_volume_l273_273957


namespace triangle_area_approx_l273_273269

theorem triangle_area_approx (r : ℝ) (a b c : ℝ) (x : ℝ)
  (h1 : r = 5)
  (h2 : a / x = 5)
  (h3 : b / x = 12)
  (h4 : c / x = 13)
  (h5 : a^2 + b^2 = c^2)
  (h6 : c = 2 * r) :
  (1 / 2 * a * b) ≈ 35.50 := by
  -- Proof omitted
  sorry

end triangle_area_approx_l273_273269


namespace dealer_profit_percentage_l273_273224

def rs : Type := Real
def num_articles_purchased := 15
def cost_price_total := 25 : rs
def num_articles_sold := 12
def selling_price_total := 38 : rs
def cost_price_per_article := cost_price_total / num_articles_purchased
def selling_price_per_article := selling_price_total / num_articles_sold
def profit_per_article := selling_price_per_article - cost_price_per_article
def profit_percentage := (profit_per_article / cost_price_per_article) * 100

theorem dealer_profit_percentage : abs (profit_percentage - 89.99) < 0.01 := by
  -- Placeholder for proof
  sorry

end dealer_profit_percentage_l273_273224


namespace maximum_withdraw_l273_273520

theorem maximum_withdraw (initial_amount withdraw deposit : ℕ) (h_initial : initial_amount = 500)
    (h_withdraw : withdraw = 300) (h_deposit : deposit = 198) :
    ∃ x y : ℕ, initial_amount - x * withdraw + y * deposit ≥ 0 ∧ initial_amount - x * withdraw + y * deposit = 194 ∧ initial_amount - x * withdraw = 300 := sorry

end maximum_withdraw_l273_273520


namespace problem1_l273_273622

theorem problem1 (x y : ℤ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 :=
sorry

end problem1_l273_273622


namespace ratio_female_to_total_l273_273512

theorem ratio_female_to_total:
  ∃ (F : ℕ), (6 + 7 * F - 9 = (6 + 7 * F) - 9) ∧ 
             (7 * F - 9 = 67 / 100 * ((6 + 7 * F) - 9)) → 
             F = 3 ∧ 6 = 6 → 
             1 / F = 2 / 6 :=
by sorry

end ratio_female_to_total_l273_273512


namespace find_dot_product_l273_273384

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
axiom cos_theta : real.cos (angle a b) = 1 / 3
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 3

-- Proof Goal
theorem find_dot_product : (2 • a + b) ⬝ b = 11 := sorry

end find_dot_product_l273_273384


namespace sum_b_an_eq_formula_l273_273780

noncomputable def sum_first_10_terms (a b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : b 1 = 1)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = 2) (h4 : ∀ n : ℕ, b (n + 1) / b n = 2) :
  Nat :=
let c : ℕ → ℕ := λ n, b (a n)
in ∑ k in (Finset.range 10), c (k + 1)

theorem sum_b_an_eq_formula
  (a b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : b 1 = 1)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = 2) (h4 : ∀ n : ℕ, b (n + 1) / b n = 2) :
  sum_first_10_terms a b h1 h2 h3 h4 = (4^10 - 1) / 3 :=
begin
  sorry
end

end sum_b_an_eq_formula_l273_273780


namespace joe_spent_255_minutes_l273_273834

-- Define the time taken to cut hair for women, men, and children
def time_per_woman : Nat := 50
def time_per_man : Nat := 15
def time_per_child : Nat := 25

-- Define the number of haircuts for each category
def women_haircuts : Nat := 3
def men_haircuts : Nat := 2
def children_haircuts : Nat := 3

-- Compute the total time spent cutting hair
def total_time_spent : Nat :=
  (women_haircuts * time_per_woman) +
  (men_haircuts * time_per_man) +
  (children_haircuts * time_per_child)

-- The theorem stating the total time spent is equal to 255 minutes
theorem joe_spent_255_minutes : total_time_spent = 255 := by
  sorry

end joe_spent_255_minutes_l273_273834


namespace tom_age_l273_273964

theorem tom_age (S T : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 := by
  sorry

end tom_age_l273_273964


namespace problem_1_problem_2_problem_3_l273_273208

def range_1 : Set ℝ :=
  { y | ∃ x : ℝ, y = 1 / (x - 1) ∧ x ≠ 1 }

def range_2 : Set ℝ :=
  { y | ∃ x : ℝ, y = x^2 + 4 * x - 1 }

def range_3 : Set ℝ :=
  { y | ∃ x : ℝ, y = x + Real.sqrt (x + 1) ∧ x ≥ 0 }

theorem problem_1 : range_1 = {y | y < 0 ∨ y > 0} :=
by 
  sorry

theorem problem_2 : range_2 = {y | y ≥ -5} :=
by 
  sorry

theorem problem_3 : range_3 = {y | y ≥ -1} :=
by 
  sorry

end problem_1_problem_2_problem_3_l273_273208


namespace S13_equals_26_l273_273739

open Nat

variable (a : Nat → ℕ)

-- Define the arithmetic sequence property
def arithmetic_sequence (d a₁ : Nat → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a₁ + n * d

-- Define the summation property
def sum_of_first_n_terms (S : Nat → ℕ) (a₁ : ℕ) (d : ℕ) : Prop :=
   ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2

-- The given condition
def condition (a₁ d : ℕ) : Prop :=
  2 * (a₁ + 4 * d) + 3 * (a₁ + 6 * d) + 2 * (a₁ + 8 * d) = 14

-- The Lean statement for the proof problem
theorem S13_equals_26 (a₁ d : ℕ) (S : Nat → ℕ) 
  (h_seq : arithmetic_sequence a d a₁) 
  (h_sum : sum_of_first_n_terms S a₁ d)
  (h_cond : condition a₁ d) : 
  S 13 = 26 := 
sorry

end S13_equals_26_l273_273739


namespace seq_sum_is_f_l273_273079

def f (x : ℕ) : ℕ := 2^x + x^2

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2^(n-1) + 2 * n - 1

theorem seq_sum_is_f (n : ℕ) : (∑ i in Finset.range n, a_n (i + 1)) = f n :=
by sorry

end seq_sum_is_f_l273_273079


namespace triangle_perimeter_l273_273467

theorem triangle_perimeter (r : ℝ) (A B C P Q R S T : ℝ)
  (triangle_isosceles : A = C)
  (circle_tangent : P = A ∧ Q = B ∧ R = B ∧ S = C ∧ T = C)
  (center_dist : P + Q = 2 ∧ Q + R = 2 ∧ R + S = 2 ∧ S + T = 2) :
  2 * (A + B + C) = 6 := by
  sorry

end triangle_perimeter_l273_273467


namespace smallest_n_for_solutions_l273_273859

-- Define the function f(x) with the fractional part {x}.
def f (x : ℝ) : ℝ := |3 * frac(x) - 1.5|

/-- 
Proof statement: There exists a smallest positive integer n = 250 such that
the equation nf(xf(x)) = 2x has at least 1000 real solutions.
-/
theorem smallest_n_for_solutions : ∃ n : ℕ, n = 250 ∧ (∀ x : ℝ, f(x) = |3 * frac(x) - 1.5|) ∧ 
  (∃ ys : finset ℝ, ys.card ≥ 1000 ∧ ∀ y ∈ ys, n * f(n * f(y)) = 2 * y) := 
by {
  sorry,
}

end smallest_n_for_solutions_l273_273859


namespace calculation_l273_273666

theorem calculation : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 :=
by
  -- We add "sorry" here to indicate where the proof would go.
  sorry

end calculation_l273_273666


namespace nicolai_ate_6_pounds_of_peaches_l273_273576

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end nicolai_ate_6_pounds_of_peaches_l273_273576


namespace product_of_reciprocals_is_9_over_4_l273_273955

noncomputable def product_of_reciprocals (a b : ℝ) : ℝ :=
  (1 / a) * (1 / b)

theorem product_of_reciprocals_is_9_over_4 (a b : ℝ) (h : a + b = 3 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  product_of_reciprocals a b = 9 / 4 :=
sorry

end product_of_reciprocals_is_9_over_4_l273_273955


namespace three_digit_numbers_with_product_24_l273_273130

-- Definitions based on the conditions
def digits (n : ℕ) : Finset ℕ :=
{ k | k < 10 }

-- Condition: Product of digits equals 24
def product_of_digits_eq_24 (n : ℕ) : Prop :=
  (digits n).prod id = 24

-- Statement of the problem
theorem three_digit_numbers_with_product_24 :
  (Finset.card { n : ℕ | n ≥ 100 ∧ n < 1000 ∧ product_of_digits_eq_24 n } = 21) :=
by {
  sorry  -- proof omitted
}

end three_digit_numbers_with_product_24_l273_273130


namespace total_distance_after_four_bounces_l273_273211

-- Define the conditions
def initial_height : ℝ := 20
def bounce_factor : ℝ := 3 / 5

-- Prove the total distance after four bounces
theorem total_distance_after_four_bounces :
  let first_ascent := initial_height,
      first_descent := initial_height,
      second_ascent := first_descent * bounce_factor,
      second_descent := second_ascent,
      third_ascent := second_descent * bounce_factor,
      third_descent := third_ascent,
      fourth_ascent := third_descent * bounce_factor,
      fourth_descent := fourth_ascent in
  (first_ascent + first_descent + second_ascent + second_descent + third_ascent + third_descent + fourth_ascent + fourth_descent + fourth_ascent) = 90.112 := by
  sorry

end total_distance_after_four_bounces_l273_273211


namespace projection_of_perpendicular_vector_l273_273002

theorem projection_of_perpendicular_vector
  (a b : ℝ × ℝ) (n : ℝ)
  (h_a : a = (2, -1))
  (h_b : b = (1, n))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  let c := (a.1 + b.1, a.2 + b.2) in
  let b_norm_sq := b.1 * b.1 + b.2 * b.2 in
  let dot_product := c.1 * b.1 + c.2 * b.2 in
  let projection := (dot_product / b_norm_sq) in
  (projection * b.1, projection * b.2) = (1, 2) := 
by sorry


end projection_of_perpendicular_vector_l273_273002


namespace probability_interval_l273_273806

open MeasureTheory ProbabilityTheory

variables {σ : ℝ}

/-- The random variable ξ follows a normal distribution N(1, σ^2) with σ > 0,
and the probability of ξ taking values in the interval (0, 1) is 0.4, then
the probability of ξ taking values in the interval (2, ∞) is 0.1. -/
theorem probability_interval (hσ : σ > 0)
  (hprob : 𝓝 (1 : ℝ) σ^2 = measure_theory.probability 
    (set.Ioc 0 1) ξ = 0.4) :
  measure_theory.probability 
    (set.Ioc 2 (∞)) ξ = 0.1 := 
sorry

end probability_interval_l273_273806


namespace seafoam_azure_ratio_l273_273903

-- Define the conditions
variables (P S A : ℕ) 

-- Purple Valley has one-quarter as many skirts as Seafoam Valley
axiom h1 : P = S / 4

-- Azure Valley has 60 skirts
axiom h2 : A = 60

-- Purple Valley has 10 skirts
axiom h3 : P = 10

-- The goal is to prove the ratio of Seafoam Valley skirts to Azure Valley skirts is 2 to 3
theorem seafoam_azure_ratio : S / A = 2 / 3 :=
by 
  sorry

end seafoam_azure_ratio_l273_273903


namespace value_range_of_m_for_equation_l273_273956

theorem value_range_of_m_for_equation 
    (x : ℝ) 
    (cos_x : ℝ) 
    (h1: cos_x = Real.cos x) :
    ∃ (m : ℝ), (0 ≤ m ∧ m ≤ 8) ∧ (4 * cos_x + Real.sin x ^ 2 + m - 4 = 0) := sorry

end value_range_of_m_for_equation_l273_273956


namespace max_ratio_a_c_over_b_d_l273_273351

-- Given conditions as Lean definitions
variables {a b c d : ℝ}
variable (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
variable (h2 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3 / 8)

-- The statement to prove the maximum value of the given expression
theorem max_ratio_a_c_over_b_d : ∃ t : ℝ, t = (a + c) / (b + d) ∧ t ≤ 3 :=
by {
  -- The proof of this theorem is omitted.
  sorry
}

end max_ratio_a_c_over_b_d_l273_273351


namespace slices_per_birthday_l273_273909

-- Define the conditions: 
-- k is the age, the number of candles, starting from 3.
variable (k : ℕ) (h : k ≥ 3)

-- Define the function for the number of triangular slices
def number_of_slices (k : ℕ) : ℕ := 2 * k - 5

-- State the theorem to prove that the number of slices is 2k - 5
theorem slices_per_birthday (k : ℕ) (h : k ≥ 3) : 
    number_of_slices k = 2 * k - 5 := 
by
  sorry

end slices_per_birthday_l273_273909


namespace Jerry_weekly_earnings_l273_273485

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end Jerry_weekly_earnings_l273_273485


namespace binary_to_base4_equiv_l273_273673

theorem binary_to_base4_equiv (n : ℕ) (h : n = 11011011) : binary_to_base4 n = 3123 := 
by 
  sorry

end binary_to_base4_equiv_l273_273673


namespace find_k_value_l273_273330

-- Define the function f(x)
def f (x : ℝ) : ℝ := (Real.cot (x / 3)) - (Real.cot (3 * x))

-- Define the expression for f(x) in terms of k
def g (x k : ℝ) : ℝ := (Real.sin (k * x)) / ((Real.sin (x / 3)) * (Real.sin (3 * x)))

theorem find_k_value : ∀ x, f(x) = g(x, 8 / 3) := by
  sorry

end find_k_value_l273_273330


namespace vector_dot_product_l273_273421

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem vector_dot_product :
  let a := (sin_deg 55, sin_deg 35)
  let b := (sin_deg 25, sin_deg 65)
  dot_product a b = (Real.sqrt 3) / 2 :=
by
  sorry

end vector_dot_product_l273_273421


namespace chord_length_HI_l273_273042

theorem chord_length_HI :
  ∀ (A B C D G E F H I O N P: Point)
    (line_AG: Line) (AB_RADIUS BC_RADIUS CD_RADIUS: ℝ)
    (r_O r_N r_P: ℝ),
    -- Conditions
    A ≠ B → B ≠ C → C ≠ D →
    OnSegment A B D → OnSegment A C D →
    Diameter AB_RADIUS O → Diameter BC_RADIUS N → Diameter CD_RADIUS P →
    radius O = 10 → radius N = 20 → radius P = 25 →
    Tangent line_AG P G → Intersects line_AG O E F → 
    Intersects line_AG N H I →
    -- Claim
    length (chord H I N) = 25.56 :=
    sorry

end chord_length_HI_l273_273042


namespace commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l273_273306

def star (x y : ℕ) : ℕ := (x + 2) * (y + 2) - 2

theorem commutative_star : ∀ x y : ℕ, star x y = star y x := by
  sorry

theorem not_distributive_star : ∃ x y z : ℕ, star x (y + z) ≠ star x y + star x z := by
  sorry

theorem special_case_star_false : ∀ x : ℕ, star (x - 2) (x + 2) ≠ star x x - 2 := by
  sorry

theorem no_identity_star : ¬∃ e : ℕ, ∀ x : ℕ, star x e = x ∧ star e x = x := by
  sorry

-- Associativity requires further verification and does not have a definitive statement yet.

end commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l273_273306


namespace max_colorings_for_non_prime_power_l273_273325

-- Define a function to check if a number is a power of a prime
def is_not_prime_power (n : ℕ) : Prop :=
  ∀ p e : ℕ, prime p → e > 0 → n ≠ p ^ e

-- Define the multichromatic condition
def multichromatic_coloring (N : ℕ) (coloring : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, ∀ gcd_ab : ℕ, a ≠ b → a ≠ gcd_ab → b ≠ gcd_ab → 
  (a ∣ N) → (b ∣ N) → (gcd_ab ∣ N) → 
  (coloring a ≠ coloring b ∧ coloring a ≠ coloring gcd_ab ∧ coloring b ≠ coloring gcd_ab)

-- Define the maximum number of multichromatic colorings
def max_multichromatic_colorings (N : ℕ) : ℕ :=
  if is_not_prime_power N then 48 else 0

-- Theorem to be proved: For a positive integer N that is not a power of any prime, the maximum number of multichromatic colorings is 48
theorem max_colorings_for_non_prime_power (N : ℕ) (h1 : 0 < N) (h2 : is_not_prime_power N) : 
  max_multichromatic_colorings N = 48 := 
sorry

end max_colorings_for_non_prime_power_l273_273325


namespace expression_value_l273_273452

theorem expression_value (x y : ℤ) (h1 : x = 7) (h2 : y = -2) : (x - 2 * y)^y = 1 / 121 := 
by 
  sorry

end expression_value_l273_273452


namespace carnival_total_cost_l273_273880

def cost_bumper_car (rides : ℕ) : ℕ := 2 * rides
def cost_space_shuttle (rides : ℕ) : ℕ := 4 * rides
def cost_ferris_wheel (rides : ℕ) : ℕ := 5 * rides

def total_cost (rides_bumper_car rides_space_shuttle rides_ferris_wheel : ℕ × ℕ) : ℕ :=
  let (mara_bc, riley_bc) := (rides_bumper_car, (0 : ℕ))
  let (mara_ss, riley_ss) := ((0 : ℕ), rides_space_shuttle)
  let (mara_fw, riley_fw) := rides_ferris_wheel
  cost_bumper_car mara_bc + cost_space_shuttle riley_ss + cost_ferris_wheel mara_fw + cost_ferris_wheel riley_fw

theorem carnival_total_cost (rides_mara_bc rides_riley_ss rides_mara_fw rides_riley_fw : ℕ)
  (h_mara_bc: rides_mara_bc = 2) (h_riley_ss : rides_riley_ss = 4) 
  (h_mara_fw : rides_mara_fw = 3) (h_riley_fw : rides_riley_fw = 3) :
  total_cost (rides_mara_bc, rides_riley_ss, rides_mara_fw + rides_riley_fw) = 50 := by
  sorry

end carnival_total_cost_l273_273880


namespace triangle_perpendicular_l273_273867

variables {A B C O P Q : Type}

/-- Let ABC be an acute-angled triangle with circumcenter O. Let S be the
circle passing through A, B, and O. The lines AC and BC intersect S at 
the further points P and Q respectively. Show that CO is perpendicular to PQ. -/
theorem triangle_perpendicular {A B C O P Q : Point} 
    (hA : acute A B C) 
    (hO : is_circumcenter O A B C) 
    (hS : circle S A B O) 
    (hP : lie_on P S ∧ on_line A C P) 
    (hQ : lie_on Q S ∧ on_line B C Q) : 
    is_perpendicular (line_through C O) (line_through P Q) :=
sorry

end triangle_perpendicular_l273_273867


namespace length_of_fourth_side_cyclic_quad_l273_273528

theorem length_of_fourth_side_cyclic_quad (AB BC CD : ℝ) (hAB : AB = 4) (hBC : BC = 6) (hCD : CD = 8)
  (area_eq : ∀ A C : ℝ, (triangle_area A B C = triangle_area A C D)) :
  DA = 3 ∨ DA = 16/3 ∨ DA = 12 := sorry

end length_of_fourth_side_cyclic_quad_l273_273528


namespace solve_for_x_l273_273598

theorem solve_for_x : ∀ x : ℝ, (10 - x) ^ 2 = x ^ 2 → x = 5 :=
by
  intros x h
  expand sorry
  solve sorry
  simp
  done

end solve_for_x_l273_273598


namespace binomial_sum_div_p_squared_l273_273060

theorem binomial_sum_div_p_squared (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) (h_k : k = (2 * p) / 3) :
  p^2 ∣ ∑ i in Finset.range (k + 1), Nat.choose p i :=
sorry

end binomial_sum_div_p_squared_l273_273060


namespace find_a8_l273_273465

variable (a_n : ℕ → ℝ)

def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a d : ℝ), ∀ n, a_n n = a + d * n

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a_n 0 + a_n (n - 1))

theorem find_a8 (h_arith : arithmetic_sequence a_n) (h_sum : sum_of_first_n_terms a_n 15 = 90) : a_n 7 = 6 :=
sorry

end find_a8_l273_273465


namespace cone_ration_proof_l273_273246

theorem cone_ration_proof (r h : ℝ) (h_nonzero : r ≠ 0) (h_rolling : ∃ s, s = real.sqrt (r^2 + h^2) ∧ 2 * real.pi * s = 38 * r * real.pi) :
  ∃ p q : ℕ, (p > 0) ∧ (q > 0) ∧ (∀ n : ℕ, prime n → ¬ (n^2 ∣ q)) ∧ (float_of_real (h / r) = p * real.sqrt q) ∧ (p + q = 16) :=
by sorry

end cone_ration_proof_l273_273246


namespace find_dot_product_l273_273387

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
axiom cos_theta : real.cos (angle a b) = 1 / 3
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 3

-- Proof Goal
theorem find_dot_product : (2 • a + b) ⬝ b = 11 := sorry

end find_dot_product_l273_273387


namespace cards_divisible_by_power_of_two_not_eventually_possible_l273_273830

theorem cards_divisible_by_power_of_two_not_eventually_possible :
  (initial_cards : Fin 100 → ℕ) →
  (∀ i < 28, (initial_cards i) % 2 = 1) →
  ∀ d : ℕ, ¬ (∃ n : ℕ, ∃ final_state : Fin n → ℕ,
    (∀ i, initial_cards i ∈ final_state) ∧
    ∃ j < n, (final_state j) % 2^d = 0) := by
  sorry

end cards_divisible_by_power_of_two_not_eventually_possible_l273_273830


namespace digit_assignment_count_l273_273428

noncomputable def digit_assignments : Finset (Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10) :=
  { assignments | let ⟨F, T, C, E, O, K⟩ := assignments in
    F ≠ 0 ∧ E ≠ 0 ∧ K ≠ 0 ∧
    F ≠ E ∧ F ≠ T ∧ F ≠ C ∧ F ≠ O ∧ F ≠ K ∧
    E ≠ T ∧ E ≠ C ∧ E ≠ O ∧ E ≠ K ∧
    T ≠ C ∧ T ≠ O ∧ T ≠ K ∧
    C ≠ O ∧ C ≠ K ∧
    O ≠ K ∧
    (100 * F.val + 10 * T.val + C.val) - (100 * E.val + 10 * T.val + O.val) = 11 * K.val }

theorem digit_assignment_count : digit_assignments.card = 180 :=
  sorry

end digit_assignment_count_l273_273428


namespace minimize_expression_l273_273350

variables {x y : ℝ}

theorem minimize_expression : ∃ (x y : ℝ), 2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 = -2 :=
by sorry

end minimize_expression_l273_273350


namespace sum_of_vectors_l273_273700

def vec1 := (⟨-3, 2, -7⟩ : ℤ × ℤ × ℤ)
def vec2 := (⟨5, -3, 4⟩ : ℤ × ℤ × ℤ)
def vec_sum := (⟨2, -1, -3⟩ : ℤ × ℤ × ℤ)

theorem sum_of_vectors : (vec1.1 + vec2.1, vec1.2 + vec2.2, vec1.3 + vec2.3) = vec_sum :=
by
  unfold vec1 vec2 vec_sum
  simp
  sorry

end sum_of_vectors_l273_273700


namespace part1_part2_l273_273418

variable (a : ℝ)

-- Defining the set A
def setA (a : ℝ) : Set ℝ := { x : ℝ | (x - 2) * (x - 3 * a - 1) < 0 }

-- Part 1: For a = 2, setB should be {x | 2 < x < 7}
theorem part1 : setA 2 = { x : ℝ | 2 < x ∧ x < 7 } :=
by
  sorry

-- Part 2: If setA a = setB, then a = -1
theorem part2 (B : Set ℝ) (h : setA a = B) : a = -1 :=
by
  sorry

end part1_part2_l273_273418


namespace dot_product_eq_eleven_l273_273391

variable {V : Type _} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)
variable (cos_theta : ℝ)
variable (norm_a norm_b : ℝ)

axiom cos_theta_def : cos_theta = 1 / 3
axiom norm_a_def : ‖a‖ = 1
axiom norm_b_def : ‖b‖ = 3

theorem dot_product_eq_eleven
  (cos_theta_def : cos_theta = 1 / 3)
  (norm_a_def : ‖a‖ = 1)
  (norm_b_def : ‖b‖ = 3) :
  (2 • a + b) ⬝ b = 11 := 
sorry

end dot_product_eq_eleven_l273_273391


namespace allan_correct_answers_l273_273892

theorem allan_correct_answers :
  ∀ (x : ℕ), 
    (∃ (correct_pts incorrect_pts : ℕ→ ℤ), (correct_pts x = x) ∧ (incorrect_pts (120 - x) = -0.25 * (120 - x))) → 
    (correct_pts x + incorrect_pts (120 - x) = 100) → 
    x = 104 :=
by sorry

end allan_correct_answers_l273_273892


namespace periodic_odd_function_f_value_of_f_at_8pi_over_3_l273_273936

noncomputable def f : ℝ → ℝ :=
  λ x, if x ∈ set.Icc 0 (π / 2) then -4 * real.sin (2 * x) else 0 -- partially defining it

theorem periodic_odd_function_f (x : ℝ) (h : x ∈ set.Icc 0 (π / 2)) :
  f(x + π) = f(x) ∧ f (-x) = -f(x) :=
  sorry

theorem value_of_f_at_8pi_over_3 :
  f (8 * π / 3) = 2 * real.sqrt 3 :=
  sorry

end periodic_odd_function_f_value_of_f_at_8pi_over_3_l273_273936


namespace problem_travel_time_with_current_l273_273248

theorem problem_travel_time_with_current
  (D r c : ℝ) (t : ℝ)
  (h1 : (r - c) ≠ 0)
  (h2 : D / (r - c) = 60 / 7)
  (h3 : D / r = t - 7)
  (h4 : D / (r + c) = t)
  : t = 3 + 9 / 17 := 
sorry

end problem_travel_time_with_current_l273_273248


namespace mass_of_man_l273_273197

noncomputable def length : ℝ := 3
noncomputable def breadth : ℝ := 2
noncomputable def height : ℝ := 0.02
noncomputable def density : ℝ := 1000

theorem mass_of_man :
  let volume := length * breadth * height in
  let mass := density * volume in
  mass = 120 :=
by
  let volume := length * breadth * height
  let mass := density * volume
  have h_volume : volume = 3 * 2 * 0.02 := by sorry
  have h_mass : mass = 1000 * volume := by sorry
  rw [h_volume] at h_mass
  norm_num at h_mass
  exact h_mass

end mass_of_man_l273_273197


namespace vasya_incorrect_l273_273974

theorem vasya_incorrect 
  {A B C C1 A1 : Type} 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  [MetricSpace C1] [MetricSpace A1]
  (hABC_TRIANGLE : is_triangle A B C)
  (hCC1_MEDIAN : is_median C C1 A B)
  (hAA1_MEDIAN : is_median A A1 B C)
  (hne : distance (C, C1) > distance (A, A1) )
  : ¬ (angle A B C < angle B C A) := 
by
  sorry

end vasya_incorrect_l273_273974


namespace find_side_length_of_cyclic_quadrilateral_l273_273530

theorem find_side_length_of_cyclic_quadrilateral
  (AB BC CD : ℝ) (h1 : AB = 4) (h2 : BC = 6) (h3 : CD = 8)
  (ABC_area_eq_ACD_area : ∀ AC : ℝ, (let s1 := (4 + 6 + AC) / 2 in 
                                     sqrt (s1 * (s1 - 4) * (s1 - 6) * (s1 - AC)))
                                = (let s2 := (8 + x + AC) / 2 in 
                                     sqrt (s2 * (s2 - 8) * (s2 - x) * (s2 - AC)))) :
  x = 3 ∨ x = 16 / 3 ∨ x = 12 :=
by sorry

end find_side_length_of_cyclic_quadrilateral_l273_273530


namespace smallest_nonneg_int_mod_15_l273_273322

theorem smallest_nonneg_int_mod_15 :
  ∃ x : ℕ, x + 7263 ≡ 3507 [MOD 15] ∧ ∀ y : ℕ, y + 7263 ≡ 3507 [MOD 15] → x ≤ y :=
by
  sorry

end smallest_nonneg_int_mod_15_l273_273322


namespace hexagon_diagonal_and_perimeter_l273_273317

theorem hexagon_diagonal_and_perimeter (side : ℝ) (h_side : side = 6) :
  let DA := side * sqrt 3
  let perimeter := 6 * side
  DA = 6 * sqrt 3 ∧ perimeter = 36 := by
  sorry

end hexagon_diagonal_and_perimeter_l273_273317


namespace max_hawthorns_satisfying_conditions_l273_273218

theorem max_hawthorns_satisfying_conditions :
  ∃ x : ℕ, 
    x > 100 ∧ 
    x % 3 = 1 ∧ 
    x % 4 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 6 = 4 ∧ 
    (∀ y : ℕ, 
      y > 100 ∧ 
      y % 3 = 1 ∧ 
      y % 4 = 2 ∧ 
      y % 5 = 3 ∧ 
      y % 6 = 4 → y ≤ 178) :=
sorry

end max_hawthorns_satisfying_conditions_l273_273218


namespace angle_DEF_is_77_5_l273_273022

theorem angle_DEF_is_77_5
  (A B C D E F : Type) 
  (angle_A : ℝ) (angle_C : ℝ) (angle_DEF : ℝ)
  (side_AB : ∀ (D : Type), ∃ (f : A → B), true)
  (side_BC : ∀ (E : Type), ∃ (f : B → C), true)
  (side_AC : ∀ (F : Type), ∃ (f : A → C), true)
  (DE_EQ_EF : ℝ → ℝ → Prop) 
  (DB_EQ_BE : ℝ → ℝ → Prop) :
  angle_A = 45 ∧ angle_C = 85 ∧ DE_EQ_EF D F ∧ DB_EQ_BE D B → angle_DEF = 77.5 :=
by
  sorry

end angle_DEF_is_77_5_l273_273022


namespace dealer_cannot_prevent_goal_l273_273240

theorem dealer_cannot_prevent_goal (m n : ℕ) :
  (m + n) % 4 = 0 :=
sorry

end dealer_cannot_prevent_goal_l273_273240


namespace range_of_a_l273_273802

noncomputable def f (a x : ℝ) : ℝ := exp x - a * x

theorem range_of_a (a : ℝ) : (∀ x > 1, 0 ≤ f' a x) ↔ a ≤ Real.exp 1 :=
by
  rw [f', funext (λ x, (real.exp_neg.mul x).trans (sub_self x).symm)]
  simp only [zero_mul, sub_nonneg]
  exact sorry

end range_of_a_l273_273802


namespace abs_neg_2022_eq_2022_l273_273546

theorem abs_neg_2022_eq_2022 : abs (-2022) = 2022 :=
by
  sorry

end abs_neg_2022_eq_2022_l273_273546


namespace angle_BAC_is_65_degrees_l273_273222

open Real

-- Define the geometrical setting
variables (O A B C D : Point) (circle : Circle) 
  (triangle : Triangle A B C)
  (h1 : O ∈ circle.center) 
  (h2 : triangle ⊂ circle) 
  (h3 : ∠AOB = 90°) 
  (h4 : ∠BOC = 130°) 
  (hD : D ∈ circle)
  (hADperpBC : Line(A, D) ⊥ Line(B, C))

-- Statement to prove
theorem angle_BAC_is_65_degrees : ∠BAC = 65° := by sorry

end angle_BAC_is_65_degrees_l273_273222


namespace parameter_a_for_log_roots_l273_273337

theorem parameter_a_for_log_roots :
  ∀ (a : ℝ), (∃ x1 x2 : ℝ,
    a * (Real.logBase 3 x1)^2 - (2 * a + 3) * (Real.logBase 3 x1) + 6 = 0 ∧
    a * (Real.logBase 3 x2)^2 - (2 * a + 3) * (Real.logBase 3 x2) + 6 = 0 ∧
    x1 = 3 * x2 ∧
    x1 ≠ x2) ↔ (a = 1 ∨ a = 3) := sorry

end parameter_a_for_log_roots_l273_273337


namespace ring_nonzero_nilpotent_iff_l273_273899

variable (R : Type) [Ring R] [Nontrivial R]

-- Specifying the prime characteristic condition
variable (p : ℕ) [Fact (Nat.Prime p)] (hpR : RingChar R = p)

-- Condition on the finite group of units
variable [Fintype Rˣ]

-- Mathematical statement to be proven
theorem ring_nonzero_nilpotent_iff (h1 : p ∣ Fintype.card Rˣ) : 
  ∃ x : R, x ≠ 0 ∧ IsNilpotent x ↔ 
  p ∣ Fintype.card Rˣ := by
s
#output we want to formulate the formal proof problem

end ring_nonzero_nilpotent_iff_l273_273899


namespace Patricia_GPA_Probability_l273_273278

noncomputable def GPA_probability (math grade science grade art grade english_prob_A english_prob_B english_prob_C history_prob_A history_prob_B history_prob_C : ℝ) (required Gpa: ℝ) : ℝ :=
  let points_needed := 5 * required_Gpa - (math grade + science grade + art grade)
  let english_outcomes := [(english_prob_A, 4), (english_prob_B, 3), (english_prob_C, 2)]
  let history_outcomes := [(history_prob_A, 4), (history_prob_B, 3), (history_prob_C, 2)]
  ∑ e in english_outcomes, ∑ h in history_outcomes, if e.2 + h.2 >= points_needed then e.1 * h.1 else 0

theorem Patricia_GPA_Probability
: GPA_probability 4 4 4 0.25 0.50 0.25 0.40 0.50 0.10 3.6 = 0.425 := sorry

end Patricia_GPA_Probability_l273_273278


namespace polynomial_odd_number_of_roots_l273_273714

noncomputable def polynomial_has_odd_number_of_complex_roots (P : Polynomial ℝ) : Prop :=
  (∃ (a2016 a2015 a2014 ... : ℝ),  -- More coefficients follow
    P = a2016 * X^2016 + a2015 * X^2015 + ... + a1 * X + a0) 
  ∧ (a2016 ≠ 0)
  ∧ (|a1 + a3 + ... + a2015| > |a0 + a2 + ... + a2016|)

theorem polynomial_odd_number_of_roots (P : Polynomial ℝ)
  (h: polynomial_has_odd_number_of_complex_roots P) :
  ∃ n, n % 2 = 1 ∧ (∃ z : ℂ, |z| < 1 ∧ P.evalC z = 0) := by
  sorry

end polynomial_odd_number_of_roots_l273_273714


namespace deepak_walking_speed_l273_273297

noncomputable def speed_deepak (circumference: ℕ) (wife_speed_kmph: ℚ) (meet_time_min: ℚ) : ℚ :=
  let meet_time_hr := meet_time_min / 60
  let wife_speed_mpm := wife_speed_kmph * 1000 / 60
  let distance_wife := wife_speed_mpm * meet_time_min
  let distance_deepak := circumference - distance_wife
  let deepak_speed_mpm := distance_deepak / meet_time_min
  deepak_speed_mpm * 60 / 1000

theorem deepak_walking_speed
  (circumference: ℕ) 
  (wife_speed_kmph: ℚ)
  (meet_time_min: ℚ)
  (H1: circumference = 627)
  (H2: wife_speed_kmph = 3.75)
  (H3: meet_time_min = 4.56) :
  speed_deepak circumference wife_speed_kmph meet_time_min = 4.5 :=
by
  sorry

end deepak_walking_speed_l273_273297


namespace jerry_weekly_earnings_l273_273489

theorem jerry_weekly_earnings:
  (tasks_per_day: ℕ) 
  (daily_earnings: ℕ)
  (weekly_earnings: ℕ) :
  (tasks_per_day = 10 / 2) ∧
  (daily_earnings = 40 * tasks_per_day) ∧
  (weekly_earnings = daily_earnings * 7) →
  weekly_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l273_273489


namespace length_JI_is_22_5_l273_273536

variable (A B C D E F J I : Point)
variable (PQ : Line)
variable (AD BE CF y_A y_B y_C x : ℝ)

-- Conditions
axiom AD_perpendicular_to_PQ : perpendicular AD PQ
axiom BE_perpendicular_to_PQ : perpendicular BE PQ
axiom CF_perpendicular_to_PQ : perpendicular CF PQ

axiom AD_length : dist A D = 15
axiom BE_length : dist B E = 9
axiom CF_length : dist C F = 30

axiom PQ_does_not_intersect_ABC : ¬ intersects PQ (triangle A B C) 

axiom I_midpoint_AC : midpoint I A C
axiom JI_perpendicular_to_PQ : perpendicular (Segment.mk J I) PQ

-- Desired result
theorem length_JI_is_22_5 : dist J I = 22.5 := 
begin
  sorry
end

end length_JI_is_22_5_l273_273536


namespace cups_of_vinegar_used_l273_273052

theorem cups_of_vinegar_used :
  ∃ V : ℝ,
    (3 + V + 1 = 5) ∧
    (1/4 * 8 + 1/6 * 18 = 5) ↔ V = 1 :=
by
  -- conditions translated as definitions
  let ketchup := 3
  let honey := 1
  let sauce_per_burger := 1/4
  let sauce_per_pulled_pork := 1/6
  let burgers := 8
  let pulled_porks := 18
  let total_sauce_needed := sauce_per_burger * burgers + sauce_per_pulled_pork * pulled_porks
  -- the equivalence we need to prove
  have sauce_equation := ketchup + honey + V = total_sauce_needed
  have total_sauce := total_sauce_needed = 5
  sorry

end cups_of_vinegar_used_l273_273052


namespace find_point_D_l273_273522

-- Define the points A, B, C, and an unknown point D
variables {A B C D : ℝ × ℝ}

-- Define the midpoints of the segments
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the condition that the midpoints form a rhombus
def is_rhombus (P Q R S : ℝ × ℝ) : Prop :=
(dist P Q = dist Q R) ∧ (dist Q R = dist R S) ∧ (dist R S = dist S P)

-- Calculation of midpoints based on the given points
def midpoint_AB := midpoint A B
def midpoint_BC := midpoint B C
def midpoint_CD := midpoint C D
def midpoint_DA := midpoint D A

-- Points A, B, C as given and D as an unknown
def A : ℝ × ℝ := (2, 8)
def B : ℝ × ℝ := (2, -2)
def C : ℝ × ℝ := (7, 0)

-- Final statement: prove that if the midpoints form a rhombus, D must be (5, 2)
theorem find_point_D (a b : ℝ) (D := (a, b)) :
  is_rhombus midpoint_AB midpoint_BC midpoint_CD midpoint_DA → D = (5, 2) :=
sorry

end find_point_D_l273_273522


namespace grid_point_midpoint_l273_273703

theorem grid_point_midpoint
  (x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 : ℤ) :
  ∃ (a b : ℕ) (i j : fin 5), i ≠ j ∧ 
    ((x1, y1) :: (x2, y2) :: (x3, y3) :: (x4, y4) :: (x5, y5) :: list.nil) !*i
          = (a, b) ∧
    ((x1, y1) :: (x2, y2) :: (x3, y3) :: (x4, y4) :: (x5, y5) :: list.nil) !*j
          = (a, b) :=
by {
  let points := [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)],
  let categories := finset.univ.image (λ p : ℤ × ℤ, (p.1 % 2, p.2 % 2)),
  have pigeonhole: categories.card ≤ 4,
  { dec_trivial },
  obtain ⟨p, hp1, hp2⟩ : ∃ (p ∈ points) (p' ∈ points), p ≠ p' ∧ p.1 % 2 = p'.1 % 2 ∧ p.2 % 2 = p'.2 % 2,
  { have key := finset.card_lt_card_iff.2 _ _ _,
    simp only [finset.image_subset_iff, finset.mem_univ, true_implies_iff] at key,
    obtain ⟨p, hp1, h⟩ := hp1,
    obtain ⟨p', hp2, hp3⟩ := h in ⟨p, hp1, p', hp2, hp3⟩ },
  sorry
}

end grid_point_midpoint_l273_273703


namespace area_of_inscribed_triangle_l273_273257

theorem area_of_inscribed_triangle 
  {r : ℝ} (h_radius: r = 5)
  {sides_ratio : ℝ} (h_sides: sides_ratio = 5/12/13) :
  let x := 10 / 13 in 
  let area := 1/2 * (5 * x) * (12 * x) in 
  (area = 3000 / 169) ∧ ((3000 / 169: ℝ).round = 17.75) :=
sorry

end area_of_inscribed_triangle_l273_273257


namespace last_two_digits_sum_l273_273067

open BigOperators

-- Defining the floor as the greatest integer not exceeding x.
def floor (x : ℝ) : ℤ := Int.floor x

-- Defining the specified sum.
noncomputable def sum_floor_sequence : ℤ :=
  ∑ k in Finset.range 2014, floor (2 ^ (k + 1) / 3 : ℝ)

-- Stating the theorem to prove that the last two digits of the sum are 15.
theorem last_two_digits_sum : (sum_floor_sequence % 100) = 15 := 
sorry

end last_two_digits_sum_l273_273067


namespace find_smallest_prime_minister_number_l273_273243

def isPrime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def distinct_prime_factors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m, m > 1 ∧ m ∣ n ∧ isPrime m)

def count_primes (l : List ℕ) : ℕ :=
  l.length

def is_primer (n : ℕ) : Prop :=
  isPrime (count_primes (distinct_prime_factors n))

def distinct_primer_factors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m, m > 1 ∧ m ∣ n ∧ is_primer m)

def is_primest (n : ℕ) : Prop :=
  is_primer (count_primes (distinct_primer_factors n))

def distinct_primest_factors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m, m > 1 ∧ m ∣ n ∧ is_primest m)

def is_prime_minister (n : ℕ) : Prop :=
  is_primest (count_primes (distinct_primest_factors n))

def smallest_prime_minister_number (N : ℕ) : Prop :=
  N = 378000 ∧ is_prime_minister N ∧ (∀ m: ℕ, is_prime_minister m → m < N → False)

theorem find_smallest_prime_minister_number : ∃ N : ℕ, smallest_prime_minister_number N := 
by 
  -- proof will involve showing 378000 is indeed the smallest such number
  sorry

end find_smallest_prime_minister_number_l273_273243


namespace john_paid_more_than_jane_by_540_l273_273492

noncomputable def original_price : ℝ := 36.000000000000036
noncomputable def discount_percentage : ℝ := 0.10
noncomputable def tip_percentage : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price * (1 - discount_percentage)
noncomputable def john_tip : ℝ := original_price * tip_percentage
noncomputable def jane_tip : ℝ := discounted_price * tip_percentage

noncomputable def john_total_payment : ℝ := discounted_price + john_tip
noncomputable def jane_total_payment : ℝ := discounted_price + jane_tip

noncomputable def difference : ℝ := john_total_payment - jane_total_payment

theorem john_paid_more_than_jane_by_540 :
  difference = 0.5400000000000023 := sorry

end john_paid_more_than_jane_by_540_l273_273492


namespace gen_sequence_term_l273_273735

theorem gen_sequence_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ k, a (k + 1) = 3 * a k + 1) :
  a n = (3^n - 1) / 2 := by
  sorry

end gen_sequence_term_l273_273735


namespace gym_monthly_income_l273_273647

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end gym_monthly_income_l273_273647


namespace value_of_f5_plus_f_minus5_l273_273723

theorem value_of_f5_plus_f_minus5 (a b c m : ℝ) :
  (f : ℝ → ℝ) = (λ x, a * x^7 - b * x^5 + c * x^3 + 2) →
  f (-5) = m →
  f 5 + f (-5) = 4 :=
by
  sorry

end value_of_f5_plus_f_minus5_l273_273723


namespace sample_max_is_86_l273_273712

noncomputable def systematic_sampling_largest (total_products : ℕ) (sample_size : ℕ) (included_product : ℕ) : ℕ :=
let interval := total_products / sample_size;
let first_sample := included_product % interval;
let last_sample_index := sample_size - 1;
first_sample + last_sample_index * interval

theorem sample_max_is_86 
  (total_products = 90)
  (sample_size = 9)
  (included_product = 36) : 
  systematic_sampling_largest total_products sample_size included_product = 86 := 
by sorry

end sample_max_is_86_l273_273712


namespace smallest_sum_with_probability_l273_273163

theorem smallest_sum_with_probability (N : ℕ) (p : ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 6) (h2 : 6 * N = 2022) (h3 : p > 0) :
  ∃ M, M = 337 ∧ (∀ sum, sum = 2022 → P(sum) = p) ∧ (∀ min_sum, min_sum = N → P(min_sum) = p):=
begin
  sorry
end

end smallest_sum_with_probability_l273_273163


namespace box_area_relation_l273_273244

theorem box_area_relation (a b c : ℕ) (h : a = b + c + 10) :
  (a * b) * (b * c) * (c * a) = (2 * (b + c) + 10)^2 := 
sorry

end box_area_relation_l273_273244


namespace mean_of_tim_scores_range_of_tim_scores_l273_273962

noncomputable def mean_score (scores : List ℝ) : ℝ :=
  (scores.sum) / (scores.length)

def range_score (scores : List ℝ) : ℝ :=
  scores.maximum - scores.minimum

def tim_scores : List ℝ := [85, 87, 92, 94, 78, 96]

theorem mean_of_tim_scores :
  mean_score tim_scores = 88.67 :=
by
  -- proof skipped
  sorry

theorem range_of_tim_scores :
  range_score tim_scores = 18 :=
by
  -- proof skipped
  sorry

end mean_of_tim_scores_range_of_tim_scores_l273_273962


namespace find_k_even_function_find_range_of_a_l273_273765

noncomputable def f (x : ℝ) (k : ℝ) := log (4^x + 1) / log 2 + k * x
noncomputable def g (x : ℝ) (a : ℝ) := log (a * 2^x - 4/3 * a) / log 2

theorem find_k_even_function : 
  (∀ x : ℝ, f (-x) k = f x k) → k = -1 :=
sorry

theorem find_range_of_a : 
  (∃ x : ℝ, f x (-1) = g x a) → a = -3 ∨ a > 1 :=
sorry

end find_k_even_function_find_range_of_a_l273_273765


namespace non_intersecting_subset_l273_273517

theorem non_intersecting_subset (n : ℕ) (cells : finset (ℕ × ℕ)) (h : cells.card = n) :
  ∃ (selected_cells : finset (ℕ × ℕ)), selected_cells.card ≥ n / 4 ∧
  (∀ (cell1 cell2 : (ℕ × ℕ)), cell1 ∈ selected_cells → cell2 ∈ selected_cells → 
   cell1 ≠ cell2 → ¬ (adjacent cell1 cell2)) :=
by
  -- Definitions for adjacency and other helper functions would be added here. 
  -- Proof is omitted as per the user's instructions.
  sorry

-- Adjacent cells share a common point: side or corner
def adjacent (c1 c2 : (ℕ × ℕ)) : Prop :=
  (abs (c1.1 - c2.1) = 1 ∧ abs (c1.2 - c2.2) = 0) ∨
  (abs (c1.1 - c2.1) = 0 ∧ abs (c1.2 - c2.2) = 1)

end non_intersecting_subset_l273_273517


namespace donuts_per_student_l273_273885

theorem donuts_per_student 
    (dozens_of_donuts : ℕ)
    (students_in_class : ℕ)
    (percentage_likes_donuts : ℕ)
    (students_who_like_donuts : ℕ)
    (total_donuts : ℕ)
    (donuts_per_student : ℕ) :
    dozens_of_donuts = 4 →
    students_in_class = 30 →
    percentage_likes_donuts = 80 →
    students_who_like_donuts = (percentage_likes_donuts * students_in_class) / 100 →
    total_donuts = dozens_of_donuts * 12 →
    donuts_per_student = total_donuts / students_who_like_donuts →
    donuts_per_student = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end donuts_per_student_l273_273885


namespace quadratic_eq_has_equal_roots_l273_273817

theorem quadratic_eq_has_equal_roots (q : ℚ) :
  (∃ x : ℚ, x^2 - 3 * x + q = 0 ∧ (x^2 - 3 * x + q = 0)) → q = 9 / 4 :=
by
  sorry

end quadratic_eq_has_equal_roots_l273_273817


namespace midpoint_of_AB_l273_273746

def point := ℝ × ℝ

noncomputable def vector_sub (p1 p2 : point) : point :=
  (p1.1 - p2.1, p1.2 - p2.2)

noncomputable def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_of_AB (A B : point) (vAB : point) :
  A = (-3, 2) →
  vector_sub B A = vAB →
  vAB = (6, 0) →
  midpoint A B = (0, 2) :=
by {
  intros,
  sorry
}

end midpoint_of_AB_l273_273746


namespace farmer_has_42_cows_left_l273_273231

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end farmer_has_42_cows_left_l273_273231


namespace Missy_handles_41_claims_l273_273287

def claims_jan := 20
def claims_john := 1.3 * claims_jan
def claims_missy := claims_john + 15

theorem Missy_handles_41_claims : claims_missy = 41 :=
by
  -- proof goes here
  sorry

end Missy_handles_41_claims_l273_273287


namespace mean_transformed_data_l273_273416

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (mean_x : ℝ)
variable (mean_x_condition : mean_x = 5)
noncomputable def mean (xs : Fin n → ℝ) : ℝ :=
  (∑ i, xs i) / n

theorem mean_transformed_data :
  mean x = mean_x →
  mean (λ i, 2 * x i + 1) = 11 :=
by
  intros h₁
  rw [mean, mean_x_condition] at h₁
  sorry

end mean_transformed_data_l273_273416


namespace length_of_boat_is_3_l273_273216

-- Definitions
def breadth : ℝ := 2
def sink_depth : ℝ := 0.01
def man_mass : ℝ := 60
def gravity : ℝ := 9.81
def water_density : ℝ := 1000

-- Volume of water displaced
def volume_displaced (length : ℝ) : ℝ := length * breadth * sink_depth

-- Weight of the man
def weight_man : ℝ := man_mass * gravity

-- Buoyant force is equal to the weight of the man
def buoyant_force_eq_weight (length : ℝ) : Prop :=
  water_density * volume_displaced(length) * gravity = weight_man

-- The main theorem to prove
theorem length_of_boat_is_3 : ∃ length, buoyant_force_eq_weight(length) ∧ length = 3 :=
by
  exists 3
  split
  · unfold buoyant_force_eq_weight volume_displaced weight_man
    norm_num
  sorry

end length_of_boat_is_3_l273_273216


namespace right_triangle_angle_l273_273459

theorem right_triangle_angle (x : ℝ) (h1 : x + 5 * x = 90) : 5 * x = 75 :=
by
  sorry

end right_triangle_angle_l273_273459


namespace prove_positive_a_l273_273732

variable (a b c n : ℤ)
variable (p : ℤ → ℤ)

-- Conditions given in the problem
def quadratic_polynomial (x : ℤ) : ℤ := a*x^2 + b*x + c

def condition_1 : Prop := a ≠ 0
def condition_2 : Prop := n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))

-- Proof goal
theorem prove_positive_a (h1 : a ≠ 0) (h2 : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
by
  sorry

end prove_positive_a_l273_273732


namespace smallest_x_for_div_by9_l273_273016

-- Define the digit sum of the number 761*829 with a placeholder * for x
def digit_sum_with_x (x : Nat) : Nat :=
  7 + 6 + 1 + x + 8 + 2 + 9

-- State the theorem to prove the smallest value of x makes the sum divisible by 9
theorem smallest_x_for_div_by9 : ∃ x : Nat, digit_sum_with_x x % 9 = 0 ∧ (∀ y : Nat, y < x → digit_sum_with_x y % 9 ≠ 0) :=
sorry

end smallest_x_for_div_by9_l273_273016


namespace erased_number_l273_273131

theorem erased_number (n i : ℕ) (h : (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17) : i = 7 :=
sorry

end erased_number_l273_273131


namespace park_probability_l273_273969

noncomputable def n_value (n : ℝ) : Prop :=
  (∃ p q r : ℕ, n = p - q * Real.sqrt r ∧ p = 60 ∧ q = 10 ∧ r = 70 ∧ Nat.gcd r (PrimeSquareFactors r) = 1)

theorem park_probability (n : ℝ) : n_value n → (60 - 10 * Real.sqrt 70 = n) :=
  sorry

end park_probability_l273_273969


namespace time_spent_cutting_hair_l273_273832

theorem time_spent_cutting_hair :
  let women's_time := 50
  let men's_time := 15
  let children's_time := 25
  let women's_haircuts := 3
  let men's_haircuts := 2
  let children's_haircuts := 3
  women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255 :=
by
  -- Definitions
  let women's_time       := 50
  let men's_time         := 15
  let children's_time    := 25
  let women's_haircuts   := 3
  let men's_haircuts     := 2
  let children's_haircuts := 3
  
  show women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255
  sorry

end time_spent_cutting_hair_l273_273832


namespace min_value_of_f_axis_of_symmetry_l273_273558

def f (x : ℝ) : ℝ := 2 ^ (abs x)

theorem min_value_of_f : ∃ x : ℝ, f x = 1 :=
by { use 0, refl }

theorem axis_of_symmetry : ∀ x : ℝ, f x = f (-x) :=
by intro x; rw [abs_neg]

end min_value_of_f_axis_of_symmetry_l273_273558


namespace price_reduction_example_l273_273882

def original_price_per_mango (P : ℝ) : Prop :=
  (115 * P = 383.33)

def number_of_mangoes (P : ℝ) (n : ℝ) : Prop :=
  (n * P = 360)

def new_number_of_mangoes (n : ℝ) (R : ℝ) : Prop :=
  ((n + 12) * R = 360)

def percentage_reduction (P R : ℝ) (reduction : ℝ) : Prop :=
  (reduction = ((P - R) / P) * 100)

theorem price_reduction_example : 
  ∃ P R reduction, original_price_per_mango P ∧
    (∃ n, number_of_mangoes P n ∧ new_number_of_mangoes n R) ∧ 
    percentage_reduction P R reduction ∧ 
    reduction = 9.91 :=
by
  sorry

end price_reduction_example_l273_273882


namespace digits_sum_of_k_l273_273242

theorem digits_sum_of_k (k : ℕ) (h : (k + 2)! + (k + 3)! = k! * 1344) : 
  k = 34 ∧ (3 + 4 = 7) :=
begin
  sorry
end

end digits_sum_of_k_l273_273242


namespace smallest_possible_sum_l273_273155

theorem smallest_possible_sum (N : ℕ) (p : ℚ) (h : p > 0) (hsum : 6 * N = 2022) : 
  ∃ (N : ℕ), N * 1 = 337 :=
by 
  use 337
  sorry

end smallest_possible_sum_l273_273155


namespace max_value_exists_l273_273844

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x * (40 - x)) + real.sqrt (x * (5 - x))

theorem max_value_exists :
  ∃ (x0 : ℝ) (M : ℝ), 0 ≤ x0 ∧ x0 ≤ 5 ∧
  (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ M) ∧
  f x0 = M ∧
  (x0, M) = (40 / 9, 15) :=
begin
  sorry
end

end max_value_exists_l273_273844


namespace correct_option_B_l273_273603

theorem correct_option_B (a : ℤ) : (2 * a) ^ 3 = 8 * a ^ 3 :=
by
  sorry

end correct_option_B_l273_273603


namespace part_a_part_b_l273_273624

noncomputable def seq_x (a b : ℝ) : ℕ → ℝ
| 0       := 1
| (n + 1) := a * seq_x n - b * seq_y n
with seq_y (a b : ℝ) : ℕ → ℝ 
| 0       := 0
| (n + 1) := seq_x n - a * seq_y n

open scoped BigOperators

theorem part_a (a b : ℝ) (k : ℕ) : 
  seq_x a b k = ∑ l in Finset.range ((k + 1) / 2), 
  (-1 : ℝ)^l * a^(k - 2 * l) * (a^2 + b)^l * 
  (∑ m in Finset.range ((k + 1) / 2).finset.filter (λ m, m ≥ l), 
  ((nat.choose k (2 * m)) * (nat.choose m l))) := sorry

noncomputable def u_k (k : ℕ) : ℕ :=
∑ l in Finset.range ((k + 1) / 2), 
  ∑ m in (Finset.range ((k + 1) / 2)).filter (λ m, m ≥ l), 
  ((nat.choose k (2 * m)) * (nat.choose m l))

def z_m_k (m k : ℕ) : ℕ :=
(u_k k) % (2^m)

theorem part_b (m : ℕ) : ∃ T, ∀ k, z_m_k m k = z_m_k m (k + T) ∧ 
  (T = if m ≠ 2 then 2^(m - 1) else 4) := sorry

end part_a_part_b_l273_273624


namespace sum_of_prob_num_den_l273_273710

noncomputable def probability_sum (a b : Finset ℕ) (ha : a.card = 4) (hb : b.card = 4) 
    (hab : a ∩ b = ∅) : ℕ :=
  let total_ways := 10
  let total_arrangements := Nat.choose 8 4
  let p := total_ways / total_arrangements
  let fraction := Rat.mkPnat total_ways (Nat.succ total_ways)
  fraction.num + fraction.denom

theorem sum_of_prob_num_den : 
   ∀ a b : Finset ℕ, 
     a.card = 4 → b.card = 4 → a ∩ b = ∅ → 
     probability_sum a b = 8 :=
by
  intros a b ha hb hab
  unfold probability_sum
  sorry

end sum_of_prob_num_den_l273_273710


namespace sum_ad_eq_two_l273_273199

theorem sum_ad_eq_two (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 2 :=
by
  sorry

end sum_ad_eq_two_l273_273199


namespace profit_distribution_l273_273276

noncomputable def profit_sharing (investment_a investment_d profit: ℝ) : ℝ × ℝ :=
  let total_investment := investment_a + investment_d
  let share_a := investment_a / total_investment
  let share_d := investment_d / total_investment
  (share_a * profit, share_d * profit)

theorem profit_distribution :
  let investment_a := 22500
  let investment_d := 35000
  let first_period_profit := 9600
  let second_period_profit := 12800
  let third_period_profit := 18000
  profit_sharing investment_a investment_d first_period_profit = (3600, 6000) ∧
  profit_sharing investment_a investment_d second_period_profit = (5040, 7760) ∧
  profit_sharing investment_a investment_d third_period_profit = (7040, 10960) :=
sorry

end profit_distribution_l273_273276


namespace determine_plane_by_parallel_lines_l273_273659

/-- Given two parallel lines in space, prove that they uniquely determine a plane. -/
theorem determine_plane_by_parallel_lines
  (L1 L2 : set ℝ³)
  (hL1 : ∀ p1 p2 ∈ L1, ∃ k : ℝ, p2 - p1 = k • (arbitrary p2 - arbitrary p1))  -- L1 is a line
  (hL2 : ∀ p1 p2 ∈ L2, ∃ k : ℝ, p2 - p1 = k • (arbitrary p2 - arbitrary p1))  -- L2 is a line
  (hParallel : ∃ v₁ v₂ : ℝ³, ∀ p1 ∈ L1, ∀ p2 ∈ L2, (p1 - p2) ∈ span ℝ {v₂} ∧ v₁ = v₂) :
  ∃ P : set ℝ³, ∀ p ∈ L1, ∀ q ∈ L2, p ∈ P ∧ q ∈ P :=
sorry

end determine_plane_by_parallel_lines_l273_273659


namespace calc_2a_plus_b_dot_b_l273_273379

open Real

variables (a b : ℝ^3)
variables (cos_angle : ℝ)
variables (norm_a norm_b : ℝ)

-- Conditions from the problem
axiom cos_angle_is_1_3 : cos_angle = 1 / 3
axiom norm_a_is_1 : ∥a∥ = 1
axiom norm_b_is_3 : ∥b∥ = 3

-- Dot product calculation
axiom a_dot_b_is_1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle

-- Prove the target
theorem calc_2a_plus_b_dot_b : 
  (2 * a + b).dot b = 11 :=
by
  -- These assertions setup the problem statement in Lean
  have h1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle, 
  from a_dot_b_is_1,
  have h2 : ∥a∥ = 1, from norm_a_is_1,
  have h3 : ∥b∥ = 3, from norm_b_is_3,
  have h4 : cos_angle = 1 / 3, from cos_angle_is_1_3,
  sorry

end calc_2a_plus_b_dot_b_l273_273379


namespace number_of_true_propositions_l273_273414

-- Define the predicates
def original_proposition (a b : ℝ) := a > b ∧ b > 0 → real.logb (1/2) a < real.logb (1/2) b + 1
def converse_of_proposition (a b : ℝ) := real.logb (1/2) a < real.logb (1/2) b + 1 → a > b ∧ b > 0
def inverse_of_proposition (a b : ℝ) := ¬(a > b ∧ b > 0) → ¬(real.logb (1/2) a < real.logb (1/2) b + 1)
def contrapositive_of_proposition (a b : ℝ) := ¬(real.logb (1/2) a < real.logb (1/2) b + 1) → ¬(a > b ∧ b > 0)

-- The main Lean 4 statement
theorem number_of_true_propositions (a b : ℝ) (h : a > b ∧ b > 0) :
  (original_proposition a b ∧ 
   contrapositive_of_proposition a b ∧ 
   ¬converse_of_proposition a b ∧
   ¬inverse_of_proposition a b) ↔ 2 = 2 := 
by
  sorry

end number_of_true_propositions_l273_273414


namespace average_class_size_difference_l273_273250

theorem average_class_size_difference :
  let enrollments : List ℕ := [60, 30, 20, 5, 3, 2] in
  let total_students := 120 in
  let total_teachers := 6 in
  let t := (enrollments.sum / total_teachers : ℚ) in
  let s := (enrollments.map (λ n, n * n / total_students)).sum in
  (t - s) = -21.151 :=
by
  let enrollments := [60, 30, 20, 5, 3, 2]
  let total_students := 120
  let total_teachers := 6
  let t : ℚ := (enrollments.sum : ℕ) / total_teachers
  let s : ℚ := (enrollments.map (λ n, (n : ℚ) * (n : ℚ) / total_students)).sum
  have h_t := (t = 20)
  have h_s := (s = 41.151)
  have h_diff := (t - s = -21.151)
  exact h_diff

end average_class_size_difference_l273_273250


namespace collinear_projections_circumcircle_l273_273875

open EuclideanGeometry

noncomputable def orthogonal_projection (P A B : Point) : Point := sorry -- definition of the orthogonal projection

theorem collinear_projections_circumcircle 
  (ABC : Triangle) (Γ : Circumcircle ABC) 
  (P : Point) (P_on_Γ : on_circumcircle P Γ):
  let D := orthogonal_projection P (ABC.v1) (ABC.v2) in
  let E := orthogonal_projection P (ABC.v2) (ABC.v3) in
  let F := orthogonal_projection P (ABC.v3) (ABC.v1) in
  collinear D E F :=
by sorry

end collinear_projections_circumcircle_l273_273875


namespace solve_inequality_l273_273323

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 2 * x - 3) * (x ^ 2 - 4 * x + 4) < 0 ↔ (-1 < x ∧ x < 3 ∧ x ≠ 2) := by
  sorry

end solve_inequality_l273_273323


namespace magnitude_sum_of_vectors_l273_273001

variables {α : ℝ}

def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (real.cos α, real.sin α)

theorem magnitude_sum_of_vectors :
  real.sqrt ((vector_a.1 + 2 * vector_b.1) ^ 2 +
             (vector_a.2 + 2 * vector_b.2) ^ 2) = 2 * real.sqrt 3 :=
by
  admit -- This is equivalent to using 'sorry' in Lean 3; used here to skip the proof.

end magnitude_sum_of_vectors_l273_273001


namespace find_phi_l273_273855

noncomputable def product_positive_imaginary_parts (roots : List ℂ) : ℂ :=
  roots.filter (fun z => z.im > 0).prod

theorem find_phi :
  let roots := [cis (360/7) 1, cis (360/7) 2, cis (360/7) 3, cis (360/7) 4, cis (360/7) 5, cis (360/7) 6]
  let Q := product_positive_imaginary_parts roots
  ∃ s > 0 (φ : ℝ), Q = s * (Complex.cos φ + Complex.sin φ * Complex.I) ∧ 0 ≤ φ ∧ φ < 360 ∧ φ ≈ 309 :=
by
  sorry

end find_phi_l273_273855


namespace chemistry_class_students_l273_273251

theorem chemistry_class_students (total_students both_classes biology_class only_chemistry_class : ℕ)
    (h1: total_students = 100)
    (h2 : both_classes = 10)
    (h3 : total_students = biology_class + only_chemistry_class + both_classes)
    (h4 : only_chemistry_class = 4 * (biology_class + both_classes)) : 
    only_chemistry_class = 80 :=
by
  sorry

end chemistry_class_students_l273_273251


namespace trailing_zeros_312_fact_l273_273128

open Nat

theorem trailing_zeros_312_fact : ∑ k in range(4), 312 / 5^k = 76 :=
by
  -- We have:
  -- 312 / 5^1 = 62
  -- 312 / 5^2 = 12
  -- 312 / 5^3 = 2
  -- 312 / 5^4 = 0 (since floor(312 / 625) = 0)
  sorry

end trailing_zeros_312_fact_l273_273128


namespace ellipse_equation_max_area_line_eqn_l273_273740

-- Definitions of ellipse and conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a b c : ℝ) : Prop := c / a = sqrt 3 / 2 ∧ c^2 = a^2 - b^2

-- Given point M
def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Line l intersects ellipse at points P and Q with |AP| = |AQ|
def line_l_intersect (P Q A : ℝ × ℝ) (C : ℝ → ℝ → Prop) : Prop := 
  C P.1 P.2 ∧ C Q.1 Q.2 ∧ A.2 = -1 ∧ (A.1 = 0 → (P.1 = Q.1 ∧ A.1^2 + A.2^2 = (P.1 - A.2)^2))

-- Given the origin O(0, 0)
def origin (O : ℝ × ℝ) : Prop := O.1 = 0 ∧ O.2 = 0

-- Problems to be proven
theorem ellipse_equation : 
  ∀ (a b : ℝ), 0 < b ∧ b < a -> 
    point_M M.1 M.2 ->
    eccentricity a b sqrt(a^2 - b^2) →
    ellipse a b 2 1 →
    ellipse a b 0 (-1) →
    ellipse a b x y ↔ (x^2 / 8) + (y^2 / 2) = 1 :=
by sorry

theorem max_area_line_eqn :
  ∀ (a b : ℝ), 0 < b ∧ b < a →
    point_M M.1 M.2 →
    eccentricity a b sqrt(a^2 - b^2) →
    ellipse a b 2 1 →
    origin O →
  ∃ (k m x y : ℝ), line_l_intersect (x, y) (k*x + m, k*y + m) (0, -1) (ellipse 2 1) ∧ 
      max_area (triangle O (x, y) (k*x + m, k*y + m)) →
        m = 3 ∧ (k = 1 ∨ k = -1 ∨ k = sqrt 2 ∨ k = -sqrt 2) :=
by sorry

end ellipse_equation_max_area_line_eqn_l273_273740


namespace problem_statement_l273_273071

variables {a b c : ℝ}

-- Given conditions
def is_root (a : ℝ) : Prop := (a = a ∧ a ≠ 0)
axiom key_roots (h : a * b * c = 4 ∧ a + b + c = 15 ∧ b * (a + c) + c * a = 20) : true

-- Main statement
noncomputable def s := (Real.sqrt a) + (Real.sqrt b) + (Real.sqrt c)

theorem problem_statement (h : a * b * c = 4 ∧ a + b + c = 15 ∧ b * (a + c) + c * a = 20) : 
  (s^4 - 28 * s^2 - 20 * s) = (305 + 2 * s^2 - 20 * s) :=
by sorry

end problem_statement_l273_273071


namespace largest_digit_divisible_by_6_l273_273989

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273989


namespace B_inter_complement_A_l273_273419

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 2})
variable (hB : B = {2, 3, 4})

theorem B_inter_complement_A : B ∩ (U \ A) = {3, 4} :=
by
  rw [hU, hA, hB]
  rw [compl_set_of, inter_set_of]
  sorry

end B_inter_complement_A_l273_273419


namespace axis_of_symmetry_l273_273432

-- Definitions corresponding to the given problem
variable (f : ℝ → ℝ)

-- Given the condition
def symmetric_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(4 - x)

-- Prove that the line x = 2 is an axis of symmetry of the graph of y = f(x)
theorem axis_of_symmetry (h : symmetric_function f) : ∀ y, f y = f (2 + (2 - y)) :=
by
  sorry

end axis_of_symmetry_l273_273432


namespace point_in_quadrant_III_l273_273413

theorem point_in_quadrant_III (x y : ℤ) (h_x : x = -1) (h_y : y = -2) : (x < 0 ∧ y < 0) → "Quadrant III" :=
by
  intros h_coordinates
  have h1 : x < 0 := by rwa [h_x]
  have h2 : y < 0 := by rwa [h_y]
  exact "Quadrant III"

end point_in_quadrant_III_l273_273413


namespace perpendicular_HB_HC_AO_l273_273868

variable {α : Type*} [EuclideanGeometry α]
variables {A B C O H_B H_C : α}

-- Let ABC be a triangle
noncomputable def triangle_ABC : Prop := triangle A B C

-- Let O be the center of the circumcircle of triangle ABC
noncomputable def center_O : Prop := circumcenter A B C = O

-- Let H_B and H_C be the feet of the altitudes from B and C
noncomputable def feet_altitudes : Prop := altitude_foot B A = H_B ∧ altitude_foot C A = H_C

-- Prove that (H_B H_C) ⊥ (AO)
theorem perpendicular_HB_HC_AO
  (hABC : triangle_ABC)
  (hO : center_O)
  (hFeet : feet_altitudes) :
  is_perpendicular (line_through H_B H_C) (line_through A O) :=
sorry

end perpendicular_HB_HC_AO_l273_273868


namespace algebraic_identity_l273_273623

theorem algebraic_identity (a b : ℝ) : a^2 - 2 * a * b + b^2 = (a - b)^2 :=
by
  sorry

end algebraic_identity_l273_273623


namespace max_customers_in_shop_l273_273138

def max_k_customers (n : ℕ) (exists_k : ℕ → Prop) : ℕ := 
  Nat.find_greatest (λ k, exists_k k) (n + 1)

def customer_property (n k : ℕ) : Prop := 
  ∀ (S : Finset ℕ), S.card = k → 
  (∃ t, ∀ i ∈ S, i = (i % n) + 1) ∨ 
  ∀ i j ∈ S, i ≠ j → ¬ ((C i t) = (C j t))

theorem max_customers_in_shop : 
  ∀ (n : ℕ) (h : n = 2016), 
  ∃ k, customer_property n k ∧ k = 45 :=
by sorry

end max_customers_in_shop_l273_273138


namespace quadrilateral_perimeter_eq_l273_273035

theorem quadrilateral_perimeter_eq (WX WZ YZ : ℝ) (hWX : WX = 24) (hWZ : WZ = 30) (hYZ : YZ = 9) 
  (hW_right : ∃ W : ℝ, W = π/2) (hWY_perp_YZ : ∃ WY : ℝ, WY = sqrt (WX^2 + WZ^2) ∧ ∃ XY : ℝ, XY = sqrt (WY^2 + YZ^2)) :
  WX + WZ + YZ + sqrt (WX^2 + WZ^2 + YZ^2 + 2 * sqrt (WX^2 + WZ^2) * YZ) = 63 + sqrt 1557 :=
by
  sorry

end quadrilateral_perimeter_eq_l273_273035


namespace find_dividend_l273_273202

noncomputable def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend :
  ∀ (divisor quotient remainder : ℕ), 
  divisor = 16 → 
  quotient = 8 → 
  remainder = 4 → 
  dividend divisor quotient remainder = 132 :=
by
  intros divisor quotient remainder hdiv hquo hrem
  sorry

end find_dividend_l273_273202


namespace measure_angle_l273_273340

def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h
def radius_cone := 8
def volume_cone := 256 * Real.pi

noncomputable def slant_height (r h : ℝ) := Real.sqrt (r^2 + h^2)
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def arc_length (r θ : ℝ) := r * θ
noncomputable def degrees (rad : ℝ) := rad * (180 / Real.pi)

theorem measure_angle (R : ℝ) (h : ℝ) (r_cone : ℝ) (vol_cone : ℝ) 
    (h_eq : h = 12) (R_eq : R = 2 * Real.sqrt 13) 
    (arc_eq : arc_length (R) 1 = circumference (r_cone)) : 
    degrees (arc_length (1) 1 - arc_length (R) 1 / circumference (R)) = 72 := 
by
  sorry

end measure_angle_l273_273340


namespace equivalent_problem_l273_273116

variable (x y : ℝ)
variable (hx_ne_zero : x ≠ 0)
variable (hy_ne_zero : y ≠ 0)
variable (h : (3 * x + y) / (x - 3 * y) = -2)

theorem equivalent_problem : (x + 3 * y) / (3 * x - y) = 2 :=
by
  sorry

end equivalent_problem_l273_273116


namespace farmer_has_42_cows_left_l273_273228

def initial_cows : ℕ := 51
def added_cows : ℕ := 5
def fraction_sold : ℚ := 1/4

theorem farmer_has_42_cows_left : 
  let total_cows := initial_cows + added_cows in
  let cows_sold := total_cows * fraction_sold in
  total_cows - cows_sold = 42 := 
by 
  -- The proof would go here, but we are only required to state the theorem.
  sorry

end farmer_has_42_cows_left_l273_273228


namespace Rachel_father_age_when_Rachel_is_25_l273_273905

-- Define the problem conditions:
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Prove the age of Rachel's father when she is 25 years old:
theorem Rachel_father_age_when_Rachel_is_25 : 
  Father_age + (25 - Rachel_age) = 60 := by
    sorry

end Rachel_father_age_when_Rachel_is_25_l273_273905


namespace largest_angle_of_pentagon_l273_273031

-- Define the angles of the pentagon and the conditions on them.
def is_angle_of_pentagon (A B C D E : ℝ) :=
  A = 108 ∧ B = 72 ∧ C = D ∧ E = 3 * C ∧
  A + B + C + D + E = 540

-- Prove the largest angle in the pentagon is 216
theorem largest_angle_of_pentagon (A B C D E : ℝ) (h : is_angle_of_pentagon A B C D E) :
  max (max (max (max A B) C) D) E = 216 :=
by
  sorry

end largest_angle_of_pentagon_l273_273031


namespace set_of_points_plane_z_eq_one_distance_point_P_to_plane_xOy_l273_273045

open EuclideanGeometry

-- Define the plane z = 1
def plane_z_eq_one : Plane ℝ := { normal := ⟨0, 0, 1⟩, point := ⟨0, 0, 1⟩ }

-- Define the point P(2, 3, 5)
def point_P : EuclideanGeometry.Point ℝ 3 := ⟨2, 3, 5⟩

-- Define the plane xOy (z=0)
def plane_xOy : Plane ℝ := { normal := ⟨0, 0, 1⟩, point := ⟨0, 0, 0⟩ }

theorem set_of_points_plane_z_eq_one :
  ∀ (p : EuclideanGeometry.Point ℝ 3), p.z = 1 ↔ p ∈ plane_z_eq_one :=
by
  sorry

theorem distance_point_P_to_plane_xOy : distance point_P plane_xOy = 5 :=
by
  sorry


end set_of_points_plane_z_eq_one_distance_point_P_to_plane_xOy_l273_273045


namespace Jerry_weekly_earnings_l273_273487

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end Jerry_weekly_earnings_l273_273487


namespace problem_solution_l273_273188

def complex_expression : ℕ := 3 * (3 * (4 * (3 * (4 * (2 + 1) + 1) + 2) + 1) + 2) + 1

theorem problem_solution : complex_expression = 1492 := by
  sorry

end problem_solution_l273_273188


namespace largest_whole_number_n_l273_273586

theorem largest_whole_number_n (n : ℕ) : 
  (\frac{1}{4} + \frac{n}{8} < 1.5) → n ≤ 9 :=
begin
  sorry
end

end largest_whole_number_n_l273_273586


namespace pentagon_area_pq_sum_l273_273864

theorem pentagon_area_pq_sum 
  (p q : ℤ) 
  (hp : 0 < q ∧ q < p) 
  (harea : 5 * p * q - q * q = 700) : 
  ∃ sum : ℤ, sum = p + q :=
by
  sorry

end pentagon_area_pq_sum_l273_273864


namespace sine_fraction_identity_l273_273078

theorem sine_fraction_identity (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (sin (4 * c) * sin (8 * c) * sin (12 * c) * sin (16 * c) * sin (20 * c)) / 
  (sin c * sin (3 * c) * sin (5 * c) * sin (7 * c) * sin (9 * c)) = 1 := by
  sorry

end sine_fraction_identity_l273_273078


namespace ce_de_squared_l273_273070

theorem ce_de_squared (O A B C D E : Type) [metric_space O] [has_mem O (circle A B)]
  (r : ℝ) (hR : r = 6) (h_ab : line_segment A B)
  (h_cd : chord C D E) (h_BE : distance B E = 4) (h_angle : angle A E C = 30) 
  : (distance C E)^2 + (distance D E)^2 = 72 := 
sorry

end ce_de_squared_l273_273070


namespace find_n_l273_273315

theorem find_n : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 123456 [MOD 9] ∧ n = 3 := by
  use 3
  split
  · norm_num
  · split
    · norm_num
    · split
      · exact nat.modeq.mod_eq_of_dvd' (by norm_num)
      · rfl

end find_n_l273_273315


namespace max_product_other_sides_l273_273754

theorem max_product_other_sides (a : ℝ) (A B C : ℝ) 
  (h1 : a = 4) 
  (h2 : A = 60) 
  (h3 : A + B + C = 180) 
  (h4 : B = 60)
  (h5 : C = 60) : 
  ∃ b c : ℝ, 
  b * c = 16 :=
by {
  use [4, 4],
  simp,
  norm_num,
}

end max_product_other_sides_l273_273754


namespace standard_eq_of_circle_l273_273353

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

def is_on_line (p : Point) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y + c = 0

def circle_eq (C : Point) (r : ℝ) : (ℝ × ℝ → ℝ) :=
  λ ⟨x, y⟩, (x - C.x) ^ 2 + (y - C.y) ^ 2 - r ^ 2

theorem standard_eq_of_circle :
  ∃ C : Point, 
    is_on_line C 1 (-1) 1 ∧
    distance C ⟨ 0, -6 ⟩ = 5 ∧
    distance C ⟨ 1, -5 ⟩ = 5 ∧
    circle_eq C 5 = λ ⟨x, y⟩, (x + 3) ^ 2 + (y + 2) ^ 2 - 25 :=
by
  sorry

end standard_eq_of_circle_l273_273353


namespace smallest_possible_sum_l273_273154

theorem smallest_possible_sum (N : ℕ) (p : ℚ) (h : p > 0) (hsum : 6 * N = 2022) : 
  ∃ (N : ℕ), N * 1 = 337 :=
by 
  use 337
  sorry

end smallest_possible_sum_l273_273154


namespace triangle_area_approx_l273_273267

theorem triangle_area_approx (r : ℝ) (a b c : ℝ) (x : ℝ)
  (h1 : r = 5)
  (h2 : a / x = 5)
  (h3 : b / x = 12)
  (h4 : c / x = 13)
  (h5 : a^2 + b^2 = c^2)
  (h6 : c = 2 * r) :
  (1 / 2 * a * b) ≈ 35.50 := by
  -- Proof omitted
  sorry

end triangle_area_approx_l273_273267


namespace solve_cos_sin_eq_one_l273_273111

theorem solve_cos_sin_eq_one (n : ℕ) (k : ℤ) :
  ∃ x : ℝ, cos (n * x) - sin (n * x) = 1 ∧ x = 2 * (↑k) * Real.pi / n := sorry

end solve_cos_sin_eq_one_l273_273111


namespace print_shop_color_copies_l273_273332

theorem print_shop_color_copies :
  ∀ (n : ℕ), 
  (2.75 * n = 1.25 * n + 120) → n = 80 :=
by sorry

end print_shop_color_copies_l273_273332


namespace tangent_circle_line_l273_273014

noncomputable theory

theorem tangent_circle_line (m : ℝ) (hm: 0 ≤ m) : 
  ∃ (l : ℝ → ℝ), (∀ x y, x^2 + y^2 = m ↔ l x + y = sqrt (2 * m)) :=
by 
  have l : ℝ → ℝ := λ x, sqrt (2 * m) - x
  use l
  intros x y
  split
  { intro h1,
    have h_dist_eq_radius : sqrt m = sqrt (2 * m)/(sqrt 2) := sorry,
    rw ← h_dist_eq_radius at h1,
    exact sorry },
  { intro h2,
    have h_radius_eq_dist : sqrt (2 * m) / (sqrt 2) = sqrt m := sorry,
    rw ← h_radius_eq_dist at h2,
    exact sorry }

end tangent_circle_line_l273_273014


namespace fractional_expression_simplification_l273_273271

theorem fractional_expression_simplification (x : ℕ) (h : x - 3 < 0) : 
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / 3 :=
by {
  -- Typical proof steps would go here, adhering to the natural conditions.
  sorry
}

end fractional_expression_simplification_l273_273271


namespace solution_set_equality_l273_273435

noncomputable def f (x : ℝ) : ℝ := 
if x ≤ 1 then sin (π / 2 * x) else x^2 + log x

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def solution_set (f : ℝ → ℝ) : set ℝ := {x | f (x - 1) < 1}

theorem solution_set_equality (h_even : is_even f) : 
  solution_set f = {x | 0 < x ∧ x < 2} :=
sorry

end solution_set_equality_l273_273435


namespace slowly_changing_constant_sign_l273_273634

-- Define slowly changing function
def slowly_changing (f : ℝ → ℝ) : Prop :=
  ∀ t > 1, ∃ L : ℝ, (∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f(t * x) / f(x) - L| < ε) ∧ L = 1

-- Define the theorem stating the desired property
theorem slowly_changing_constant_sign (f : ℝ → ℝ) (h : slowly_changing f) :
  ∃ N : ℝ, ∀ x y > N, f(x) * f(y) > 0 :=
sorry

end slowly_changing_constant_sign_l273_273634


namespace largest_digit_divisible_by_6_l273_273992

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l273_273992


namespace angle_BCD_is_30_degrees_l273_273474

variables (A B C I D : Type) [EuclideanGeometry]
variables (AB AC BD BI : ℝ) [IsoscelesTriangle AB AC]
variables (A_angle : ℝ)

/-- Proving that the angle BCD is 30 degrees given the specific conditions in the problem. -/
theorem angle_BCD_is_30_degrees
  (h₁ : AB = AC)
  (h₂ : A_angle = 100)
  (h₃ : is_incenter I A B C)
  (h₄ : BD = BI) :
  angle B C D = 30 := sorry

end angle_BCD_is_30_degrees_l273_273474


namespace tangent_line_equation_range_of_a_l273_273410

-- Given function definition
def f (a x : ℝ) := x^3 + a * x^2 + 3 * x - 9

-- First problem: 
-- If f takes an extreme value at x = -3, prove the equation of the tangent line at (0, f(0)).
theorem tangent_line_equation (a : ℝ) (h : f' a (-3) = 0) : 
  let fa := f 5
  let slope := (deriv (f 5) 0)
  let point := (0, f 5 0)
  slope = 3 → (slope * (x - 0)) - (y + f 5 0) = 0 :=
by sorry

-- Second problem:
-- If f is monotonically decreasing in the interval [1, 2], find the range of values for a.
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → deriv (f a) x ≤ 0) → a ≤ -15 / 4 :=
by sorry

end tangent_line_equation_range_of_a_l273_273410


namespace hyperbola_eccentricity_l273_273086

theorem hyperbola_eccentricity (a b m n e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_mn : m * n = 2 / 9)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) : e = 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_l273_273086


namespace mary_gave_becky_green_crayons_l273_273881

-- Define the initial conditions
def initial_green_crayons : Nat := 5
def initial_blue_crayons : Nat := 8
def given_blue_crayons : Nat := 1
def remaining_crayons : Nat := 9

-- Define the total number of crayons initially
def total_initial_crayons : Nat := initial_green_crayons + initial_blue_crayons

-- Define the number of crayons given away
def given_crayons : Nat := total_initial_crayons - remaining_crayons

-- The crux of the problem
def given_green_crayons : Nat :=
  given_crayons - given_blue_crayons

-- Formal statement of the theorem
theorem mary_gave_becky_green_crayons
  (h_initial_green : initial_green_crayons = 5)
  (h_initial_blue : initial_blue_crayons = 8)
  (h_given_blue : given_blue_crayons = 1)
  (h_remaining : remaining_crayons = 9) :
  given_green_crayons = 3 :=
by {
  -- This should be the body of the proof, but we'll skip it for now
  sorry
}

end mary_gave_becky_green_crayons_l273_273881


namespace area_ratio_ADE_BCED_l273_273046

open EuclideanGeometry

variables {A B C D E F : Point}
variables {AB BC AC AD AE: ℝ}
variables {AB_pos : 0 < AB} {BC_pos : 0 < BC} {AC_pos : 0 < AC}
variables {AD_pos : 0 < AD} {AE_pos : 0 < AE}

-- Conditions specified
def triangle_ABC : Triangle := ⟨A, B, C⟩
def point_D_on_AB : Line := Line.mk A B
def point_E_on_AC : Line := Line.mk A C
def point_F_on_BC : Line := Line.mk B C
def parallel_DF_BE := parallel <| Segment.mk D F <| Segment.mk B E

-- Given numerical lengths
axiom AB_eq_24 : AB = 24
axiom BC_eq_45 : BC = 45
axiom AC_eq_48 : AC = 48
axiom AD_eq_18 : AD = 18
axiom AE_eq_16 : AE = 16

-- The main theorem
theorem area_ratio_ADE_BCED :
  ¬ collinear A B C →
  point_of_line D point_D_on_AB →
  point_of_line E point_E_on_AC →
  point_of_line F point_F_on_BC →
  parallel_DF_BE →
  (area_ratio (triangle a d e) (quadrilateral b c e d) = 1/15) :=
sorry

end area_ratio_ADE_BCED_l273_273046


namespace proof_problem_l273_273000

variables {a b : ℝ × ℝ}
variables (ha : ‖a‖ = 1)
variables (hb : b = (1, real.sqrt 3))
variables (hacb : ‖a + 2 • b‖ = 3)
variables (hadot : (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2)) = 0)

noncomputable def norm_vec (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem proof_problem : 
  norm_vec (2 • a - 3 • b) = 8 ∧ 
  (⊥ • ((dot_product a b) / (norm_vec b) ^ 2)) = (1 / 4, real.sqrt 3 / 4) :=
by 
  sorry

end proof_problem_l273_273000


namespace smallest_possible_sum_l273_273152

theorem smallest_possible_sum (N : ℕ) (p : ℚ) (h : p > 0) (hsum : 6 * N = 2022) : 
  ∃ (N : ℕ), N * 1 = 337 :=
by 
  use 337
  sorry

end smallest_possible_sum_l273_273152


namespace tangent_line_eq_g_monotonicity_k_range_l273_273407

noncomputable def f : ℝ → ℝ → ℝ := λ k x, k * Real.exp x - x^2 / 2

theorem tangent_line_eq (k : ℝ) (h : k = 1) : 
  (e - 1) * 1 - (f 1 1) + 1/2 = 0 := by sorry

noncomputable def g : ℝ → ℝ → ℝ := λ k x, k * Real.exp x - x

theorem g_monotonicity (k : ℝ) :
  (∀ x, k ≤ 0 → deriv (g k) x < 0) ∧
  (∀ x, k > 0 → (x < - real.log k → deriv (g k) x < 0) ∧ (x > -real.log k → deriv (g k) x > 0)) := by sorry

theorem k_range (k : ℝ) :
  (∃ s t : ℝ, 0 < t ∧ t < s ∧ (f s s - f s t) / (s - t) > 1) ↔ 1 ≤ k := by sorry

end tangent_line_eq_g_monotonicity_k_range_l273_273407


namespace tangent_line_at_point_l273_273451

theorem tangent_line_at_point (f : ℝ → ℝ) (h_tangent : ∀ x, (∃ (f'(x) : ℝ), y = f x := x + 2) :
  f'(1) = 1) :
  f(1) + f'(1) = 4 :=
by
  sorry

end tangent_line_at_point_l273_273451


namespace michelle_additional_racks_l273_273191

theorem michelle_additional_racks :
  ∀ (cups_per_pound initial_racks pounds_per_rack: ℕ) (total_flour_cups: ℕ),
  cups_per_pound = 2 →
  initial_racks = 3 →
  pounds_per_rack = 3 →
  total_flour_cups = 3 * 8 →
  (total_flour_cups / cups_per_pound / pounds_per_rack) - initial_racks = 1 :=
by
  intros cups_per_pound initial_racks pounds_per_rack total_flour_cups
  assume h1 h2 h3 h4
  sorry

end michelle_additional_racks_l273_273191


namespace largest_digit_divisible_by_6_l273_273994

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l273_273994


namespace problem_part1_problem_part2_problem_part3_l273_273408

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x^2 - x

theorem problem_part1 :
  (∀ x, 0 < x -> x < 1 / Real.exp 1 -> f (Real.log x + 1) < 0) ∧ 
  (∀ x, x > 1 / Real.exp 1 -> f (Real.log x + 1) > 0) ∧ 
  (f (1 / Real.exp 1) = 1 / Real.exp 1 * Real.log (1 / Real.exp 1)) :=
sorry

theorem problem_part2 (a : ℝ) :
  (∀ x, x > 0 -> f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem problem_part3 (a : ℝ) (m : ℝ) (ha : a = 1/8) :
  (∃ m, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ (3 * f x / (4 * x) + m + g a x = 0))) ↔ 
  (7/8 < m ∧ m < (15/8 - 3/4 * Real.log 3)) :=
sorry

end problem_part1_problem_part2_problem_part3_l273_273408


namespace gunther_typing_l273_273423

theorem gunther_typing : 
  (∀ (number_of_words : ℕ) (minutes_per_set : ℕ) (total_working_minutes : ℕ),
    number_of_words = 160 → minutes_per_set = 3 → total_working_minutes = 480 →
    (total_working_minutes / minutes_per_set * number_of_words) = 25600) :=
begin
  intros number_of_words minutes_per_set total_working_minutes,
  intros h_words h_time h_total,
  rw [h_words, h_time, h_total],
  norm_num,
end

end gunther_typing_l273_273423


namespace range_of_a_l273_273778

open Set

def U := univ : Set ℝ
def A : Set ℝ := {x | x < -1}
def Ac : Set ℝ := {x | x ≥ -1}

def B (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 3}

theorem range_of_a (a : ℝ) : B a ⊆ Ac → a ≥ -1 / 2 := 
by {
  intro h,
  sorry
}

end range_of_a_l273_273778


namespace kolakoski_term_73_74_75_76_l273_273935

def kolakoskiSeq : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 1
| 3     := 1
| 4     := 2
| 5     := 2
| (n+6) := if kolakoskiSeq ((List.foldl (+) 6 (List.take (n + 1) (List.map kolakoskiSeq [0, 1, 2, 3, 4, 5])) / 3) = 1 
            then 1 else 2

theorem kolakoski_term_73_74_75_76 :
  (kolakoskiSeq 72 = 2) ∧ 
  (kolakoskiSeq 73 = 1) ∧ 
  (kolakoskiSeq 74 = 2) ∧ 
  (kolakoskiSeq 75 = 2) :=
  sorry

end kolakoski_term_73_74_75_76_l273_273935


namespace apartments_with_one_resident_l273_273027

theorem apartments_with_one_resident :
  ∀ (total_apartments : ℕ) (at_least_one_resident_percentage at_least_two_residents_percentage: ℝ),
  total_apartments = 500 →
  at_least_one_resident_percentage = 0.90 →
  at_least_two_residents_percentage = 0.72 →
  let at_least_one_resident := (at_least_one_resident_percentage * total_apartments).to_nat,
      at_least_two_residents := (at_least_two_residents_percentage * total_apartments).to_nat,
      only_one_resident := at_least_one_resident - at_least_two_residents
  in only_one_resident = 90 :=
begin
  intros,
  sorry
end

end apartments_with_one_resident_l273_273027


namespace arc_length_of_circle_l273_273557

theorem arc_length_of_circle (r θ : ℝ) (h1 : r = 2) (h2 : θ = 5 * Real.pi / 3) : (θ * r) = 10 * Real.pi / 3 :=
by
  rw [h1, h2]
  -- subsequent steps would go here 
  sorry

end arc_length_of_circle_l273_273557


namespace sum_of_b_p_equals_25905_l273_273705

def b (p : ℕ) : ℕ :=
  let k := Real.floor (Real.sqrt p + 0.5)
  in if 0 < k ∧ |(k : ℝ) - Real.sqrt p| < 0.5 then k else 0

def T : ℕ := ∑ p in Finset.range 3000, b (p + 1)

theorem sum_of_b_p_equals_25905 : T = 25905 :=
by
  sorry

end sum_of_b_p_equals_25905_l273_273705


namespace max_triangle_area_l273_273581

theorem max_triangle_area 
  (A B C : Type) 
  [AffineSpace ℝ (Type*)] 
  (a b c : ℝ) :
  (a = 9) → 
  (b / c = 40 / 41) → 
  (∃ (x : ℝ), b = 40 * x ∧ c = 41 * x ∧ x < 9 ∧ x > 1 / 9) → 
  ∃ (area : ℝ), area ≤ 820 :=
by 
  intros h1 h2 h3
  have := h1
  have := h2
  have := h3
  sorry

end max_triangle_area_l273_273581


namespace inequality_holds_l273_273209

theorem inequality_holds (c : ℝ) (X Y : ℝ) (h1 : X^2 - c * X - c = 0) (h2 : Y^2 - c * Y - c = 0) :
    X^3 + Y^3 + (X * Y)^3 ≥ 0 :=
sorry

end inequality_holds_l273_273209


namespace no_perpendicular_hatching_other_than_cube_l273_273516

def is_convex_polyhedron (P : Polyhedron) : Prop :=
  -- Definition of a convex polyhedron
  sorry

def number_of_faces (P : Polyhedron) : ℕ :=
  -- Function returning the number of faces of polyhedron P
  sorry

def hatching_perpendicular (P : Polyhedron) : Prop :=
  -- Definition that checks if the hatching on adjacent faces of P is perpendicular
  sorry

theorem no_perpendicular_hatching_other_than_cube :
  ∀ (P : Polyhedron), is_convex_polyhedron P ∧ number_of_faces P ≠ 6 → ¬hatching_perpendicular P :=
by
  sorry

end no_perpendicular_hatching_other_than_cube_l273_273516


namespace angle_APB_l273_273818

-- Define the problem conditions
variables (XY : Π X Y : ℝ, XY = X + Y) -- Line XY is a straight line
          (semicircle_XAZ : Π X A Z : ℝ, semicircle_XAZ = X + Z - A) -- Semicircle XAZ
          (semicircle_ZBY : Π Z B Y : ℝ, semicircle_ZBY = Z + Y - B) -- Semicircle ZBY
          (PA_tangent_XAZ_at_A : Π P A X Z : ℝ, PA_tangent_XAZ_at_A = P + A + X - Z) -- PA tangent to XAZ at A
          (PB_tangent_ZBY_at_B : Π P B Z Y : ℝ, PB_tangent_ZBY_at_B = P + B + Z - Y) -- PB tangent to ZBY at B
          (arc_XA : ℝ := 45) -- Arc XA is 45 degrees
          (arc_BY : ℝ := 60) -- Arc BY is 60 degrees

-- Main theorem to prove
theorem angle_APB : ∀ P A B: ℝ, 
  540 - 90 - 135 - 120 - 90 = 105 := by 
  -- Proof goes here
  sorry

end angle_APB_l273_273818


namespace unique_solution_exists_and_X_expression_l273_273059

section problem

variables {n : Type} [fintype n] [decidable_eq n]
variables (A : matrix n n ℝ) (X : matrix n n ℝ)

-- Definitions based on conditions
def is_nilpotent (A : matrix n n ℝ) : Prop := A ^ 3 = 0

-- The main proof problem
theorem unique_solution_exists_and_X_expression (h : is_nilpotent A) :
  ∃! X : matrix n n ℝ, X + A * X + X * A^2 = A ∧ X = A * (1 + A + A^2)^(-1) :=
sorry

end problem

end unique_solution_exists_and_X_expression_l273_273059


namespace exists_odd_digit_div_by_five_power_l273_273206

theorem exists_odd_digit_div_by_five_power (n : ℕ) (h : 0 < n) : ∃ (k : ℕ), 
  (∃ (m : ℕ), k = m * 5^n) ∧ 
  (∀ (d : ℕ), (d = (k / (10^(n-1))) % 10) → d % 2 = 1) :=
sorry

end exists_odd_digit_div_by_five_power_l273_273206


namespace carries_average_speed_is_approx_34_29_l273_273056

noncomputable def CarriesActualAverageSpeed : ℝ :=
  let jerry_speed := 40 -- in mph
  let jerry_time := 1/2 -- in hours, 30 minutes = 0.5 hours
  let jerry_distance := jerry_speed * jerry_time

  let beth_distance := jerry_distance + 5
  let beth_time := jerry_time + (20 / 60) -- converting 20 minutes to hours

  let carrie_distance := 2 * jerry_distance
  let carrie_time := 1 + (10 / 60) -- converting 10 minutes to hours

  carrie_distance / carrie_time

theorem carries_average_speed_is_approx_34_29 : 
  |CarriesActualAverageSpeed - 34.29| < 0.01 :=
sorry

end carries_average_speed_is_approx_34_29_l273_273056


namespace jerry_weekly_earnings_l273_273483

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l273_273483


namespace problem_statement_l273_273044

def C1_parametric (t : ℝ) : ℝ × ℝ :=
  (-4 + cos t, 3 + sin t)

def C1_cartesian_eq (x y : ℝ) : Prop :=
  (x + 4)^2 + (y - 3)^2 = 1

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def C2_polar_eq (θ : ℝ) : ℝ :=
  -6 / sqrt (1 + 8 * (sin θ)^2)

def C2_cartesian_eq (x y : ℝ) : Prop :=
  (x^2) / 36 + (y^2) / 4 = 1

def M_coords (θ : ℝ) : ℝ × ℝ :=
  (-2 + 3 * cos θ, 2 + sin θ)

def C3_line_eq (x y : ℝ) : Prop :=
  x + sqrt(3) * y + 6 * sqrt(3) = 0

def distance_to_line (m : ℝ × ℝ) (a b c : ℝ) : ℝ := 
  let (x, y) := m in
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem problem_statement :
  ∀ x y : ℝ, C1_cartesian_eq x y ↔ ∃ t : ℝ, x = -4 + cos t ∧ y = 3 + sin t ∧
  ∀ θ : ℝ, C2_cartesian_eq (ρ * cos θ) (ρ * sin θ) ↔ ρ = C2_polar_eq θ ∧
  ∀ (θ : ℝ), sin (θ + π / 3) = -1 →
  let M : ℝ × ℝ := M_coords θ in
  distance_to_line M 1 sqrt(3) (6 * sqrt(3)) = 3 * sqrt(3) - 1 :=
by
  sorry

end problem_statement_l273_273044


namespace positional_relationship_l273_273367

variables {Point : Type} [EuclideanSpace Point]

def line (P Q : Point) : Set Point := sorry

def perpendicular (l1 l2 : Set Point) := sorry

def parallel (l : Set Point) (α : Set (Set Point)) := sorry

def lies_in (l : Set Point) (α : Set (Set Point)) := sorry

parameters (m n : Set Point) (α : Set (Set Point))
  (h1 : perpendicular m α)
  (h2 : perpendicular m n)

theorem positional_relationship : parallel n α ∨ lies_in n α :=
sorry

end positional_relationship_l273_273367


namespace total_votes_is_132_l273_273808

theorem total_votes_is_132 (T : ℚ) 
  (h1 : 1 / 4 * T + 1 / 3 * T = 77) : 
  T = 132 := 
  sorry

end total_votes_is_132_l273_273808


namespace faye_country_albums_l273_273192

theorem faye_country_albums (C : ℕ) (h1 : 6 * C + 18 = 30) : C = 2 :=
by
  -- This is the theorem statement with the necessary conditions and question
  sorry

end faye_country_albums_l273_273192


namespace scalene_triangle_hyperbola_concur_l273_273507

noncomputable def scalene_triangle_concurrence_points (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] (h_a h_b h_c : A → Prop) : ℕ :=
  let Ha := λ P, |dist P B - dist P C| = |dist A B - dist A C|,
      Hb := λ P, |dist P C - dist P A| = |dist B C - dist B A|,
      Hc := λ P, |dist P A - dist P B| = |dist C A - dist C B|
  in if (∀ P, Ha P ↔ h_a P) ∧ (∀ P, Hb P ↔ h_b P) ∧ (∀ P, Hc P ↔ h_c P)
     then 2
     else 0

-- Theorem statement
theorem scalene_triangle_hyperbola_concur (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] 
  (h_a h_b h_c : A → Prop) :
  -- Given the conditions described in the problem
  scalene_triangle_concurrence_points A B C h_a h_b h_c = 2 :=
sorry

end scalene_triangle_hyperbola_concur_l273_273507


namespace remainder_division_l273_273996

theorem remainder_division (n : ℕ) :
  n = 2345678901 →
  n % 102 = 65 :=
by sorry

end remainder_division_l273_273996


namespace gym_monthly_revenue_l273_273642

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end gym_monthly_revenue_l273_273642


namespace smallest_abs_value_of_diff_l273_273617

theorem smallest_abs_value_of_diff : ∃ (k l : ℕ), |36^k - 5^l| = 11 :=
sorry

end smallest_abs_value_of_diff_l273_273617


namespace larger_acute_angle_right_triangle_l273_273458

theorem larger_acute_angle_right_triangle (x : ℝ) (h1 : x > 0) (h2 : x + 5 * x = 90) : 5 * x = 75 := by
  sorry

end larger_acute_angle_right_triangle_l273_273458


namespace find_angle_l273_273165

-- Definitions of the conditions
def radius1 := 5
def radius2 := 3
def radius3 := 1

noncomputable def totalArea (r1 r2 r3 : ℝ) : ℝ :=
  ∑ π * r^2 in [r1, r2, r3].map (λ r, r^2)

def shadedArea (u : ℝ) : ℝ := (u * 10) / 17

noncomputable def unshadedArea (total shaded: ℝ) : ℝ := total - shaded

-- The proof statement
theorem find_angle
  (h : totalArea radius1 radius2 radius3 = total)
  (hu : shadedArea u = (totalArea radius1 radius2 radius3 * 10) / 27) :
  ∀ θ, θ = 107 / 459 :=
begin
  sorry
end

end find_angle_l273_273165


namespace pentagon_sums_l273_273178

variable (a b c d e : ℚ)

theorem pentagon_sums (h : (∀ (x y : ℚ), x + y ∈ {a + b, b + c, c + d, d + e, e + a, a + c, b + d, c + e, d + a, e + b} → (x + y).is_integer)) :
  (a + b).is_integer ∧ (b + c).is_integer ∧ (c + d).is_integer ∧ (d + e).is_integer ∧ (e + a).is_integer ∧ 
  (a + c).is_integer ∧ (b + d).is_integer ∧ (c + e).is_integer ∧ (d + a).is_integer ∧ (e + b).is_integer :=
by
  sorry

end pentagon_sums_l273_273178


namespace rahul_share_of_payment_l273_273612

-- Definitions
def rahulWorkDays : ℕ := 3
def rajeshWorkDays : ℕ := 2
def totalPayment : ℤ := 355

-- Theorem statement
theorem rahul_share_of_payment :
  let rahulWorkRate := 1 / (rahulWorkDays : ℝ)
  let rajeshWorkRate := 1 / (rajeshWorkDays : ℝ)
  let combinedWorkRate := rahulWorkRate + rajeshWorkRate
  let rahulShareRatio := rahulWorkRate / combinedWorkRate
  let rahulShare := (totalPayment : ℝ) * rahulShareRatio
  rahulShare = 142 :=
by
  sorry

end rahul_share_of_payment_l273_273612


namespace percent_chemical_a_in_mixture_l273_273542

-- Define the given problem parameters
def percent_chemical_a_in_solution_x : ℝ := 0.30
def percent_chemical_a_in_solution_y : ℝ := 0.40
def proportion_of_solution_x_in_mixture : ℝ := 0.80
def proportion_of_solution_y_in_mixture : ℝ := 1.0 - proportion_of_solution_x_in_mixture

-- Define what we need to prove: the percentage of chemical a in the mixture
theorem percent_chemical_a_in_mixture:
  (percent_chemical_a_in_solution_x * proportion_of_solution_x_in_mixture) + 
  (percent_chemical_a_in_solution_y * proportion_of_solution_y_in_mixture) = 0.32 
:= by sorry

end percent_chemical_a_in_mixture_l273_273542


namespace collinear_A_D_X_l273_273508

-- A, O, M, D, X are points
variables {A O M D X : Type} [inner_product_space ℝ A O M D X]

-- Condition 1: D is the second intersection point of the circumcircle of triangle AOM with ω 
def on_circumcircle (A O M D : Type) : Prop :=
  ∃ ω : circle, ω ∈ circumcircle (triangle A O M) ∧ D ∈ ω

-- Condition 2: Triangle OAM is similar to triangle OXA
def similar_triangles (O A M X : Type) : Prop :=
  ∃ (f : triangle O A M ≃ₜ triangle O X A), f

-- Condition 3: AOMD is a cyclic quadrilateral
def cyclic_quadrilateral (A O M D : Type) : Prop :=
  ∃ ω : circle, A ∈ ω ∧ O ∈ ω ∧ M ∈ ω ∧ D ∈ ω

-- Theorem: Points A, D, and X are collinear
theorem collinear_A_D_X
  (h1 : on_circumcircle A O M D)
  (h2 : similar_triangles O A M X)
  (h3 : cyclic_quadrilateral A O M D) :
  collinear A D X :=
sorry

end collinear_A_D_X_l273_273508


namespace jerry_weekly_earnings_l273_273484

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l273_273484


namespace range_of_a_l273_273774

-- Definitions of propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, ax² - x + (1/16)*a > 0

def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → sqrt(2*x + 1) < 1 + a*x

-- The statement of the theorem
theorem range_of_a (a : ℝ) : ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → a ∈ Icc 1 2 :=
by
  sorry

end range_of_a_l273_273774


namespace exactly_4_situations_with_2_primes_l273_273341

def is_prime (n: ℕ) : Prop := nat.prime n

def primes_up_to_30 : list ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def num_primes (l: list ℕ) : ℕ :=
l.countp is_prime

def consecutive_subseq_with_exactly_2_primes (n: ℕ) : list (list ℕ) :=
(list.range' n 10).filter (λ l, num_primes l = 2)

theorem exactly_4_situations_with_2_primes:
  consecutive_subseq_with_exactly_2_primes 21 = -- we use 21 because 21 + 10 - 1 < 30
    [[18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
     [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]] :=
by sorry

end exactly_4_situations_with_2_primes_l273_273341


namespace units_digit_of_k_squared_plus_2_k_l273_273862

def k := 2008^2 + 2^2008

theorem units_digit_of_k_squared_plus_2_k : 
  (k^2 + 2^k) % 10 = 7 :=
by {
  -- The proof will be inserted here
  sorry
}

end units_digit_of_k_squared_plus_2_k_l273_273862


namespace ellipse_eq_triangle_area_l273_273745

-- Given conditions
variable (M : ℝ × ℝ) (a b c : ℝ) (P : ℝ × ℝ)
variable (hM : M = (Real.sqrt 6, Real.sqrt 2))
variable (hEx : c / a = Real.sqrt 6 / 3)
variable (hIneq : 0 < b ∧ b < a)
variable (hEllipse : (M.1 ^ 2 / a ^ 2) + (M.2 ^ 2 / b ^ 2) = 1)
variable (hP : P = (-3, 2))
variable (hRelation : a ^ 2 = b ^ 2 + c ^ 2)

-- Part 1: Prove the equation of the ellipse
theorem ellipse_eq : (a = Real.sqrt 12) ∧ (b = 2) :=
sorry

-- Part 2: Prove the area of the triangle PAB
theorem triangle_area : 
  (∀ l : ℝ → ℝ → Prop, (λ x y => l x y = x + y + 2) → 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ is_isosceles_triangle P A B ∧
  line_through_points A B l ∧ 
  area_triangle P A B = 9 / 2)) :=
sorry

end ellipse_eq_triangle_area_l273_273745


namespace find_f_f_condition_l273_273720

noncomputable def f (x : ℝ) : ℝ := 
  if x = 1 then 0
  else (2 * x - 1) / (1 - x)

theorem find_f :
  ∀ x : ℝ, x ≠ 1 → f(x) = (2 * x - 1) / (1 - x) :=
by
  intro x hx
  sorry

theorem f_condition :
  ∀ x : ℝ, x ≠ -1 → f (x / (x + 1)) = x - 1 :=
by
  intros x hx
  rw [f, if_neg]
  sorry
  field_simp [hx]
  sorry

end find_f_f_condition_l273_273720


namespace interval_of_monotonic_increase_l273_273441

def f (x : ℝ) : ℝ := (1/2) * Real.sin (2 * x + (Real.pi / 3))

def g (x : ℝ) : ℝ := (1/2) * Real.sin (2 * (x + (Real.pi / 3)) + (Real.pi / 3))

theorem interval_of_monotonic_increase (k : ℤ) :
  ∀ x, (g x = -(1/2) * Real.sin (2 * x))
  → (k * Real.pi + Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 4) :=
sorry

end interval_of_monotonic_increase_l273_273441


namespace hyperbola_standard_equation_l273_273392

theorem hyperbola_standard_equation:
  ∀ (P F₁ F₂ : ℝ × ℝ),
    F₁ = (-5, 0) →
    F₂ = (5, 0) →
    (∃ P, (|dist P F₁ - dist P F₂| = 8)) →
    (∀ x y, (x, y) ∈ hyperbola ↔ x^2 / 16 - y^2 / 9 = 1) :=
by
  sorry

end hyperbola_standard_equation_l273_273392


namespace cecilia_wins_l273_273894

theorem cecilia_wins (a : ℕ) (ha : a > 0) :
  ∃ b : ℕ, b > 0 ∧ gcd a b = 1 ∧ ∃ p1 p2 p3 : ℕ, 
    prime p1 ∧ prime p2 ∧ prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    p1 ∣ (a^3 + b^3) ∧ p2 ∣ (a^3 + b^3) ∧ p3 ∣ (a^3 + b^3) :=
sorry -- proof omitted

end cecilia_wins_l273_273894


namespace min_value_of_squares_l273_273011

theorem min_value_of_squares (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 ≥ 1990 :=
sorry

end min_value_of_squares_l273_273011


namespace triangle_inequality_l273_273475

theorem triangle_inequality 
  (A B C : ℝ) 
  (m n l : ℝ) :
  (√(m^2 + tan (A / 2) * tan (B / 2)) + 
  √(n^2 + tan (B / 2) * tan (C / 2)) + 
  √(l^2 + tan (C / 2) * tan (A / 2))) ≤ 
  √(3 * (m^2 + n^2 + l^2 + 1)) :=
sorry

end triangle_inequality_l273_273475


namespace isosceles_triangle_congruent_side_length_l273_273709

theorem isosceles_triangle_congruent_side_length
  (s : ℝ) (area_sq : Real) (area_tri_sum : Real) (congruence : Real)
  (h₁ : s = 2)
  (h₂ : area_sq = s ^ 2)
  (h₃ : area_tri_sum = area_sq)
  (h₄ : area_tri_sum = 4 * congruence)
  (h₅ : congruence = 1) :
  ∃ l : Real, l = √2 :=
by
  sorry

end isosceles_triangle_congruent_side_length_l273_273709


namespace solve_fractional_equation_l273_273914

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) : 
  (2 * x) / (x - 1) = x / (3 * (x - 1)) + 1 ↔ x = -3 / 2 :=
by sorry

end solve_fractional_equation_l273_273914


namespace closest_whole_number_area_l273_273652

theorem closest_whole_number_area (π : ℝ)
  (hπ : π ≈ Real.pi) -- Assume π is approximately the value of π
  (h_rect_dim : ∃ (l w : ℝ), l = 3 ∧ w = 4)
  (h_circ_diam : ∃ d : ℝ, d = 2)
  (h_non_overlap : true): -- True indicates non-overlapping condition
  ∃ (closest : ℝ), closest = 6 := by
  sorry

end closest_whole_number_area_l273_273652


namespace min_colors_convex_ngon_l273_273183

-- Define the problem conditions and their corresponding constraints

theorem min_colors_convex_ngon (n : ℕ) (h : n ≥ 3) :
  ∃ (coloring_scheme : (fin n) → (fin n) → (fin n)),
    (∀ v : fin n, 
      ∀ e1 e2 : fin n, e1 ≠ e2 → coloring_scheme v e1 ≠ coloring_scheme v e2) ∧ 
    (∀ v : fin n, coloring_scheme v v ≠ v) := 
  sorry

end min_colors_convex_ngon_l273_273183


namespace number_of_x_satisfying_g_of_g_eq_three_l273_273937

def g (x : ℝ) : ℝ :=
if x ≤ 1 then - (1 / 2) * x^2 + x + 3
else (1 / 3) * x^2 - 3 * x + 11

theorem number_of_x_satisfying_g_of_g_eq_three :
  set.count
    { x | -3 ≤ x ∧ x ≤ 5 ∧ g (g x) = 3 } = 1 :=
sorry

end number_of_x_satisfying_g_of_g_eq_three_l273_273937


namespace find_x_l273_273089

-- Let a, b, c, d, e be real numbers
variables (a b c d e : ℝ)

-- Let x be a real number represented as given
noncomputable def solve_for_x : ℝ :=
  ( (c * e / 2 + c * d)^(1 / b) ) - a

-- The theorem to be proven
theorem find_x (h : ((solve_for_x a b c d e + a) ^ b) / c - d = e / 2) : solve_for_x a b c d e = (( (c * e / 2 + c * d)^(1 / b) ) - a) :=
begin
  sorry
end

end find_x_l273_273089


namespace exists_small_factorable_subsequence_l273_273090

def small_factorable (n : ℕ) : Prop := ∃ (a b c d : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ n = a * b * c * d

theorem exists_small_factorable_subsequence :
  ∃ (xs : ℕ), (100 ≤ xs ∧ xs ≤ 999) → 
  ∃ (ys : ℕ), (100 ≤ ys ∧ ys ≤ 999) → 
  ∃ subseq : ℕ, 0 ≤ subseq < 10^6 ∧ small_factorable subseq := 
sorry

end exists_small_factorable_subsequence_l273_273090


namespace nicolai_ate_6_pounds_of_peaches_l273_273575

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end nicolai_ate_6_pounds_of_peaches_l273_273575


namespace fill_sink_time_l273_273713

theorem fill_sink_time {R1 R2 R T: ℝ} (h1: R1 = 1 / 210) (h2: R2 = 1 / 214) (h3: R = R1 + R2) (h4: T = 1 / R):
  T = 105.75 :=
by 
  sorry

end fill_sink_time_l273_273713


namespace finite_cuboid_blocks_l273_273902

/--
Prove that there are only finitely many cuboid blocks with integer dimensions a, b, c
such that abc = 2(a - 2)(b - 2)(c - 2) and c ≤ b ≤ a.
-/
theorem finite_cuboid_blocks :
  ∃ (S : Finset (ℤ × ℤ × ℤ)), ∀ (a b c : ℤ), (abc = 2 * (a - 2) * (b - 2) * (c - 2)) → (c ≤ b) → (b ≤ a) → (a, b, c) ∈ S := 
by
  sorry

end finite_cuboid_blocks_l273_273902


namespace two_layer_coverage_l273_273578

-- Assuming the basic setup
variables (A_212 : ℝ) (cover_area : ℝ) (three_layer_area: ℝ) (two_layer_area: ℝ)

-- Assumptions based on the given conditions
axiom total_runner_area : A_212 = 212
axiom covered_table_area : cover_area = 0.80 * 175
axiom area_three_layers : three_layer_area = 24

-- The proof problem statement
theorem two_layer_coverage : (total_runner_area = 212) → (covered_table_area = 140) → (area_three_layers = 24) → (two_layer_area = 48) :=
by
  intros h1 h2 h3
  sorry

end two_layer_coverage_l273_273578


namespace other_group_less_garbage_l273_273879

theorem other_group_less_garbage :
  387 + (735 - 387) = 735 :=
by
  sorry

end other_group_less_garbage_l273_273879


namespace triangle_area_l273_273262

-- Given conditions
def ratio_5_12_13 (a b c : ℝ) : Prop := 
  (b / a = 12 / 5) ∧ (c / a = 13 / 5)

def right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

def circumscribed_circle (a b c r : ℝ) : Prop :=
  2 * r = c

-- The main theorem we need to prove
theorem triangle_area (a b c r : ℝ) (h_ratio : ratio_5_12_13 a b c) (h_triangle : right_triangle a b c) (h_circle : circumscribed_circle a b c r) (h_r : r = 5) :
  0.5 * a * b ≈ 17.75 :=
by
  sorry

end triangle_area_l273_273262


namespace triangle_area_approx_l273_273266

noncomputable def radius := 5
noncomputable def ratio_side1 := 5
noncomputable def ratio_side2 := 12
noncomputable def ratio_side3 := 13
noncomputable def hypotenuse := 2 * radius

theorem triangle_area_approx : ∃ (area : ℝ), 
  (∀ x : ℝ, 
    13 * x = hypotenuse → 
    area = (1/2) * (5 * x) * (12 * x)
  ) ∧ 
  abs(area - 17.75) < 0.01 :=
by
  sorry

end triangle_area_approx_l273_273266


namespace y_coordinate_range_of_G_l273_273941

noncomputable def ellipse : set (ℝ × ℝ) :=
  { p | let ⟨x, y⟩ := p in x^2 / 25 + y^2 / 9 = 1 }

def left_focus : ℝ × ℝ := (-4, 0)

theorem y_coordinate_range_of_G :
  ∀ (A B G : ℝ × ℝ),
    A ∈ ellipse ∧ B ∈ ellipse ∧
    (∃ k : ℝ, k ≠ 0 ∧ ∃ x₀ y₀ : ℝ, (A.1 + B.1) / 2 = x₀ ∧ (A.2 + B.2) / 2 = y₀ ∧
    line_through_F A B k ∧ perpendicular_bisector A B G) →
    G = (0, y_G) →
  ∃ y : ℝ, y_G = y ∧ y ∈ 
    Icc (-32 / 15) 0 ∪ Icc 0 (32 / 15) :=
begin
  sorry
end

def line_through_F (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ x y : ℝ, y = k (x + 4)

def perpendicular_bisector (A B G : ℝ × ℝ) : Prop :=
  ∃ x₀ y₀ : ℝ, G = (0, y₀) ∧
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  y₀ = (G.2 - M.2) / (M.1 - G.1)

end y_coordinate_range_of_G_l273_273941


namespace sixteen_nails_no_three_collinear_l273_273049

structure Point (n : Nat) where
  x : Fin n
  y : Fin n

def is_collinear {n : Nat} (p1 p2 p3 : Point n) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

def no_three_collinear {n : Nat} (ps : List (Point n)) : Prop :=
  ∀ p1 p2 p3, p1 ∈ ps → p2 ∈ ps → p3 ∈ ps → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ is_collinear p1 p2 p3

theorem sixteen_nails_no_three_collinear : ∃ (ps : List (Point 8)), ps.length = 16 ∧ no_three_collinear ps :=
  let ps := [Point.mk ⟨0, sorry⟩ ⟨0, sorry⟩, Point.mk ⟨0, sorry⟩ ⟨4, sorry⟩,
             Point.mk ⟨1, sorry⟩ ⟨1, sorry⟩, Point.mk ⟨1, sorry⟩ ⟨5, sorry⟩,
             Point.mk ⟨2, sorry⟩ ⟨2, sorry⟩, Point.mk ⟨2, sorry⟩ ⟨6, sorry⟩,
             Point.mk ⟨3, sorry⟩ ⟨3, sorry⟩, Point.mk ⟨3, sorry⟩ ⟨7, sorry⟩,
             Point.mk ⟨4, sorry⟩ ⟨0, sorry⟩, Point.mk ⟨4, sorry⟩ ⟨4, sorry⟩,
             Point.mk ⟨5, sorry⟩ ⟨1, sorry⟩, Point.mk ⟨5, sorry⟩ ⟨5, sorry⟩,
             Point.mk ⟨6, sorry⟩ ⟨2, sorry⟩, Point.mk ⟨6, sorry⟩ ⟨6, sorry⟩,
             Point.mk ⟨7, sorry⟩ ⟨3, sorry⟩, Point.mk ⟨7, sorry⟩ ⟨7, sorry⟩] in
  ⟨ps, by sorry, by sorry⟩

end sixteen_nails_no_three_collinear_l273_273049


namespace true_discount_is_correct_l273_273123

theorem true_discount_is_correct {BD PV : ℝ} (hBD : BD = 78) (hPV : PV = 429) :
  let TD := BD / (1 + BD / PV) in TD = 66.00 :=
by
  sorry

end true_discount_is_correct_l273_273123


namespace data_set_conditions_l273_273252

def data_set : List ℕ := [1, 2, 2, 3]

def average (lst : List ℕ) : ℚ :=
  lst.sum / lst.length

def median (lst : List ℕ) : ℕ :=
  let sorted := lst.sort
  sorted.get! (sorted.length / 2 - 1)

def std_dev (lst : List ℕ) : ℚ :=
  let avg := average lst
  let variance := (lst.map (λ x => (x - avg) ^ 2)).sum / lst.length
  real.sqrt variance

theorem data_set_conditions : 
  average data_set = 2 ∧ median data_set = 2 ∧ std_dev data_set = 1 :=
by
  -- Proof omitted
  sorry

end data_set_conditions_l273_273252


namespace orthocenter_midpoint_l273_273663

theorem orthocenter_midpoint {A B C P D E M : Point} {Ω : Circle} :
  acute_triangle A B C →
  triangle_inscribed A B C Ω →
  tangent_point Ω B P →
  tangent_point Ω C P →
  perpendicular P D (line A B) →
  perpendicular P E (line A C) →
  midpoint_of_segment M B C →
  orthocenter_of_triangle M A D E :=
sorry

end orthocenter_midpoint_l273_273663


namespace perpendicular_line_eqn_l273_273180

noncomputable def slope_of_line := (3:ℝ) / 6

noncomputable def perpendicular_slope (m : ℝ) := -1 / m

noncomputable def line_through_point_slope (x₀ y₀ m : ℝ) : (ℝ → ℝ) :=
  λ x, m * (x - x₀) + y₀

theorem perpendicular_line_eqn :
  ∃ f : ℝ → ℝ,
    (∀ x y, 3 * x - 6 * y = 9 → y = (3 / 6) * x - (9 / 6)) ∧
    (f (-2) = 3) ∧
    (∀ x, f x = -2 * x - 1) :=
by
  let m := slope_of_line
  let mₚ := perpendicular_slope m
  let f := line_through_point_slope (-2) (3:ℝ) mₚ
  use f
  have hₘ : ∀ x y, 3 * x - 6 * y = 9 → y = (3 / 6) * x - (9 / 6),
  { intros x y h,
    rw [eq_sub_iff_add_eq, ← sub_eq_add_neg] at h,
    linarith,
  },
  split; [exact hₘ, split; sorry, intros x, sorry]

end perpendicular_line_eqn_l273_273180


namespace total_amount_spent_l273_273210

theorem total_amount_spent (num_pigs num_hens avg_price_hen avg_price_pig : ℕ)
                          (h_num_pigs : num_pigs = 3)
                          (h_num_hens : num_hens = 10)
                          (h_avg_price_hen : avg_price_hen = 30)
                          (h_avg_price_pig : avg_price_pig = 300) :
                          num_hens * avg_price_hen + num_pigs * avg_price_pig = 1200 :=
by
  sorry

end total_amount_spent_l273_273210


namespace second_solution_sugar_concentration_l273_273200

theorem second_solution_sugar_concentration 
    (W : ℝ) 
    (hW_pos : W > 0)
    (C1 : 0.08 * W - 0.02 * W + (X / 100) * (W / 4) = 0.16 * W) : 
    X = 40 :=
by
  have h_eq : 0.06 * W + X * W / 400 = 0.16 * W := C1
  have h_mul_400 : 400 * (0.06 * W) + X * W = 400 * (0.16 * W) := by ring_nf [h_eq]
  have h_XW : 24 * W + X * W = 64 * W := by ring_nf [h_mul_400]
  have h_final : X * W = 40 * W := by ring_nf
  exact sorry

end second_solution_sugar_concentration_l273_273200


namespace quadratic_function_properties_l273_273393

theorem quadratic_function_properties
  (f : ℝ → ℝ)
  (hf_quad : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c)
  (hf_neg1 : f (-1) = 0)
  (hf_neg3 : f (-3) = 4)
  (hf_pos1 : f (1) = 4)
  : (f = λ x, (x + 1)^2) ∧ (∀ x, f(x - 1) ≥ 4 ↔ x ≤ -2 ∨ x ≥ 2) :=
by
  sorry

end quadratic_function_properties_l273_273393


namespace probability_B_independent_A_D_mutually_exclusive_C_D_l273_273145

namespace BallDraw

def balls := {1, 2, 3, 4, 5, 6 : ℕ}

-- Event definitions
def event_A (b: ℕ) (b ∈ balls) : Prop := b % 2 = 1 -- first ball is odd
def event_B (b: ℕ) (b ∈ balls) : Prop := b % 2 = 0 -- second ball is even
def event_C (b1 b2: ℕ) (b1 ∈ balls) (b2 ∈ balls) : Prop := (b1 + b2) % 2 = 1 -- sum is odd
def event_D (b1 b2: ℕ) (b1 ∈ balls) (b2 ∈ balls) : Prop := (b1 + b2) % 2 = 0 -- sum is even

-- Prove the statements
theorem probability_B : (probability event_B) = 1/2 := sorry

theorem independent_A_D : independent event_A event_D := sorry

theorem mutually_exclusive_C_D : mutually_exclusive event_C event_D := sorry

end BallDraw

end probability_B_independent_A_D_mutually_exclusive_C_D_l273_273145


namespace geom_seq_common_ratio_l273_273470

theorem geom_seq_common_ratio {a : ℕ → ℝ} (q : ℝ) 
  (h1 : a 3 = 6) 
  (h2 : a 1 + a 2 + a 3 = 18) 
  (h_geom_seq : ∀ n, a (n + 1) = a n * q) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end geom_seq_common_ratio_l273_273470


namespace trains_cross_time_l273_273972

theorem trains_cross_time
  (len1 : ℝ) (time1 : ℝ)
  (len2 : ℝ) (time2 : ℝ)
  (speed_reduction_percent : ℝ)
  (uphill_percent : ℝ)
  (H_len1 : len1 = 150)
  (H_time1 : time1 = 12)
  (H_len2 : len2 = 180)
  (H_time2 : time2 = 18)
  (H_speed_reduction : speed_reduction_percent = 10)
  (H_uphill : uphill_percent = 2) :
  let V1 := len1 / time1,
      V2 := len2 / time2,
      V2_reduced := V2 - (V2 * speed_reduction_percent / 100),
      V_relative := V1 + V2_reduced,
      distance_total := len1 + len2,
      cross_time := distance_total / V_relative
  in cross_time ≈ 15.35 :=
by
  sorry

end trains_cross_time_l273_273972


namespace quadratic_real_root_m_l273_273017

theorem quadratic_real_root_m (m : ℝ) (h : 4 - 4 * m ≥ 0) : m = 0 ∨ m = 2 ∨ m = 4 ∨ m = 6 ↔ m = 0 :=
by
  sorry

end quadratic_real_root_m_l273_273017


namespace probability_white_given_popped_l273_273213

-- Define conditional probabilities and the probability calculations
open Rat

theorem probability_white_given_popped
  (P_white : ℚ := 3/4)
  (P_yellow : ℚ := 1/4)
  (P_damaged : ℚ := 1/4)
  (P_ND_white : ℚ := 3/4 * 3/4)
  (P_ND_yellow : ℚ := 1/4 * 3/4)
  (P_popped_given_ND_white : ℚ := 3/5)
  (P_popped_given_ND_yellow : ℚ := 4/5) :
  (P_ND_white * P_popped_given_ND_white) / ((P_ND_white * P_popped_given_ND_white) + (P_ND_yellow * P_popped_given_ND_yellow)) = 9/13 :=
by
  sorry

end probability_white_given_popped_l273_273213


namespace yogurt_count_l273_273537

theorem yogurt_count (Y : ℕ) 
  (ice_cream_cartons : ℕ := 20)
  (cost_ice_cream_per_carton : ℕ := 6)
  (cost_yogurt_per_carton : ℕ := 1)
  (spent_more_on_ice_cream : ℕ := 118)
  (total_cost_ice_cream : ℕ := ice_cream_cartons * cost_ice_cream_per_carton)
  (total_cost_yogurt : ℕ := Y * cost_yogurt_per_carton)
  (expenditure_condition : total_cost_ice_cream = total_cost_yogurt + spent_more_on_ice_cream) :
  Y = 2 :=
by {
  sorry
}

end yogurt_count_l273_273537


namespace greg_needs_more_rotations_l273_273783
-- Import the necessary Mathlib library

/-
Greg is riding his bike around town and notices that on flat ground, each block he rides, his wheels rotate 200 times.
He's now on a trail and wants to make sure he rides at least 8 blocks. His wheels have already rotated 600 times,
and he has ridden 2 blocks on flat ground and 1 block uphill. When Greg rides uphill, his wheels rotate 250 times per block.
For the remaining blocks, he plans to ride 3 more uphill and 2 more on flat ground.
Prove that Greg needs 550 more rotations to reach his goal.
-/
theorem greg_needs_more_rotations :
  let rotations_per_flat_block := 200
  let rotations_per_uphill_block := 250
  let completed_rotations := 600
  let completed_flat_blocks := 2
  let completed_uphill_blocks := 1
  let remaining_flat_blocks := 2
  let remaining_uphill_blocks := 3
  let total_blocks_needed := 8
  let total_rotations_needed := 1150
  let goal := total_blocks_needed
  in 
    remaining_flat_blocks * rotations_per_flat_block + remaining_uphill_blocks * rotations_per_uphill_block - completed_rotations = 550 := by
  { sorry }

end greg_needs_more_rotations_l273_273783


namespace triangle_bisector_ratio_l273_273472

theorem triangle_bisector_ratio
  (A B C D E T F : Type)
  (hD_on_AB : D ∈ line_segment A B)
  (hE_on_AC : E ∈ line_segment A C)
  (hAT_bisects_ANGLE_A : bisects (angle_at A B C) A T)
  (hAT_intersects_DE_at_F : intersects T (line_segment D E) F)
  (hAD : distance A D = 2)
  (hDB : distance D B = 2)
  (hAE : distance A E = 3)
  (hEC : distance E C = 3) :
  AF / AT = 1 / 2 := by
  sorry

end triangle_bisector_ratio_l273_273472


namespace pyramid_volume_l273_273564

theorem pyramid_volume (a : ℝ) : 
  (∃ (h h1 : ℝ), 
    h1 = a * (real.sqrt 2) ∧  -- Distance BF
    h = a * (real.sqrt 6) / 2 ∧ -- Height KD
    true -- Other conditions implicitly holding in context of geometric constraints
  ) → 
  let volume := (1 / 6 : ℝ) * a^3 * (real.sqrt 6) in
  volume = (real.sqrt 6) * a^3 / 18 :=
by {
  sorry
}

end pyramid_volume_l273_273564


namespace find_value_of_x_l273_273007

theorem find_value_of_x
  (x : ℝ)
  (h : log 9 (log 4 (log 3 x)) = 1) :
  x ^ (-2 / 3) = 3 ^ (-174762.6667) :=
by
  sorry

end find_value_of_x_l273_273007


namespace count_distinct_arrangements_l273_273136

-- Definitions based on the problem conditions
def is_vowel (c : Char) : Prop :=
  c = 'e' ∨ c = 'o'

def count_vowels (word : List Char) : Nat :=
  word.countp is_vowel

def count_consonants (word : List Char) : Nat :=
  word.length - count_vowels word

def arrangements_with_non_adjacent_vowels (word : List Char) : Nat :=
  60 * 20 * 3  -- Calculated directly from the solution steps

-- Theorem statement to be proved
theorem count_distinct_arrangements (word : List Char)
  (H : count_vowels word = 3)
  (H1 : count_consonants word = 5)
  (H2 : word.length = 8)
  (H3 : word = ['w', 'e', 'l', 'l', 'c', 'o', 'm', 'e']) :
  arrangements_with_non_adjacent_vowels word = 3600 :=
by
  sorry

end count_distinct_arrangements_l273_273136


namespace kostya_initially_planted_l273_273613

def bulbs_after_planting (n : ℕ) (stages : ℕ) : ℕ :=
  match stages with
  | 0 => n
  | k + 1 => 2 * bulbs_after_planting n k - 1

theorem kostya_initially_planted (n : ℕ) (stages : ℕ) :
  bulbs_after_planting n stages = 113 → n = 15 := 
sorry

end kostya_initially_planted_l273_273613


namespace molecular_weight_of_compound_l273_273587

noncomputable def atomic_weight_carbon : ℝ := 12.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.008
noncomputable def atomic_weight_oxygen : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 1

noncomputable def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weight_carbon) + (num_H * atomic_weight_hydrogen) + (num_O * atomic_weight_oxygen)

theorem molecular_weight_of_compound :
  molecular_weight num_carbon_atoms num_hydrogen_atoms num_oxygen_atoms = 65.048 :=
by
  sorry

end molecular_weight_of_compound_l273_273587


namespace zero_point_interval_l273_273137

noncomputable def f (x : ℝ) : ℝ := (4 / x) - (2^x)

theorem zero_point_interval : ∃ x : ℝ, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end zero_point_interval_l273_273137


namespace solution_in_quadrants_I_and_II_l273_273303

theorem solution_in_quadrants_I_and_II (x y : ℝ) :
  (y > 3 * x) ∧ (y > 6 - 2 * x) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
by
  sorry

end solution_in_quadrants_I_and_II_l273_273303


namespace stephen_pizza_percentage_l273_273916

theorem stephen_pizza_percentage:
  ∃ (S : ℝ), 
  (∃ (total_slices : ℕ := 24) (remaining_slices : ℕ := 9),
     (S >= 0) ∧ (S <= 1) ∧ 
     (0.5 * (1 - S) * total_slices = remaining_slices) 
     ∧ (S = 0.25)) :=
  sorry

end stephen_pizza_percentage_l273_273916


namespace op_assoc_l273_273300

open Real

def op (x y : ℝ) : ℝ := x + y - x * y

theorem op_assoc (x y z : ℝ) : op (op x y) z = op x (op y z) := by
  sorry

end op_assoc_l273_273300


namespace series_converges_to_half_l273_273670

noncomputable def series_value : ℝ :=
  ∑' (n : ℕ), (n^4 + 3*n^3 + 10*n + 10) / (3^n * (n^4 + 4))

theorem series_converges_to_half : series_value = 1 / 2 :=
  sorry

end series_converges_to_half_l273_273670


namespace max_value_of_a2b3c4_l273_273858

open Real

theorem max_value_of_a2b3c4
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 19683 / 472392 :=
sorry

end max_value_of_a2b3c4_l273_273858


namespace movement_west_8m_is_neg_8m_l273_273799

theorem movement_west_8m_is_neg_8m :
  (east_positive : ∀ (d : ℝ), d > 0 → "east" -> notation d = +d) →
  (west_negative : ∀ (d : ℝ), d > 0 → "west" -> notation d = -d) →
  (move_east_10m_is_plus_10 : notation 10 "east" = +10) →
  notation 8 "west" = -8 := by
  intros east_positive west_negative move_east_10m_is_plus_10
  exact sorry

end movement_west_8m_is_neg_8m_l273_273799


namespace five_digit_numbers_no_repeats_five_digit_even_numbers_five_digit_multiple_of_three_numbers_l273_273412

-- Definitions based on conditions
def digits : List ℕ := [1, 2, 3, 4, 7, 9]

-- Proofs without giving the solution steps
theorem five_digit_numbers_no_repeats : 
  ∃ n : ℕ, n = 720 ∧ ∀ (num : List ℕ), num.length = 5 ∧ (num ∈ (digits.permutations.filter (λ l, l.dedup.length = 5))) → n = 720 := 
sorry

theorem five_digit_even_numbers : 
  ∃ n : ℕ, n = 240 ∧ ∀ (num : List ℕ), num.length = 5 ∧ (num ∈ (digits.permutations.filter (λ l, l.dedup.length = 5 ∧ (l.getLast 0 % 2 = 0)))) → n = 240 := 
sorry

theorem five_digit_multiple_of_three_numbers : 
  ∃ n : ℕ, n = 120 ∧ ∀ (num : List ℕ), num.length = 5 ∧ (num ∈ (digits.permutations.filter (λ l, l.dedup.length = 5 ∧ (l.sum % 3 = 0)))) → n = 120 := 
sorry

end five_digit_numbers_no_repeats_five_digit_even_numbers_five_digit_multiple_of_three_numbers_l273_273412


namespace f_neg1_gt_f2_l273_273635

variable (f : ℝ → ℝ)

noncomputable def F (x : ℝ) : ℝ := f (x + 1)

-- Hypotheses
axiom func_defined : ∀ x : ℝ, f x ∈ ℝ
axiom func_increasing : ∀ x y : ℝ, 1 < x → x < y → f x < f y
axiom F_symmetric : ∀ x : ℝ, F x = F (-x)

-- Goal
theorem f_neg1_gt_f2 : f (-1) > f 2 :=
by sorry

end f_neg1_gt_f2_l273_273635


namespace triangle_area_approx_l273_273270

theorem triangle_area_approx (r : ℝ) (a b c : ℝ) (x : ℝ)
  (h1 : r = 5)
  (h2 : a / x = 5)
  (h3 : b / x = 12)
  (h4 : c / x = 13)
  (h5 : a^2 + b^2 = c^2)
  (h6 : c = 2 * r) :
  (1 / 2 * a * b) ≈ 35.50 := by
  -- Proof omitted
  sorry

end triangle_area_approx_l273_273270


namespace next_class_l273_273837

variable (school_start : ℕ := 12)
variable (science_over : ℕ := 16)
variable (schedule : List String := ["Maths", "History", "Geography", "Science", "Music"])

theorem next_class (science_over_time : ℕ) (schedule : List String) (h1 : school_start = 12) (h2 : science_over_time = 16) (h3 : schedule = ["Maths", "History", "Geography", "Science", "Music"]) : 
  List.nth schedule (schedule.indexOf "Science" + 1) = some "Music" :=
begin
  sorry
end

end next_class_l273_273837


namespace apricot_tea_calories_l273_273883

theorem apricot_tea_calories :
  let apricot_juice_weight := 150
  let apricot_juice_calories_per_100g := 30
  let honey_weight := 50
  let honey_calories_per_100g := 304
  let water_weight := 300
  let apricot_tea_weight := apricot_juice_weight + honey_weight + water_weight
  let apricot_juice_calories := apricot_juice_weight * apricot_juice_calories_per_100g / 100
  let honey_calories := honey_weight * honey_calories_per_100g / 100
  let total_calories := apricot_juice_calories + honey_calories
  let caloric_density := total_calories / apricot_tea_weight
  let tea_weight := 250
  let calories_in_250g_tea := tea_weight * caloric_density
  calories_in_250g_tea = 98.5 := by
  sorry

end apricot_tea_calories_l273_273883


namespace gunther_typing_l273_273426

theorem gunther_typing :
  ∀ (wpm : ℚ), (wpm = 160 / 3) → 480 * wpm = 25598 :=
by
  intros wpm h
  sorry

end gunther_typing_l273_273426


namespace nicolai_peaches_pounds_l273_273572

-- Condition definitions
def total_fruit_pounds : ℕ := 8
def mario_oranges_ounces : ℕ := 8
def lydia_apples_ounces : ℕ := 24
def ounces_to_pounds (ounces: ℕ) : ℚ := ounces / 16 

-- Statement we want to prove
theorem nicolai_peaches_pounds :
  let mario_oranges_pounds := ounces_to_pounds mario_oranges_ounces,
      lydia_apples_pounds := ounces_to_pounds lydia_apples_ounces,
      eaten_by_m_and_l := mario_oranges_pounds + lydia_apples_pounds,
      nicolai_peaches_pounds := total_fruit_pounds - eaten_by_m_and_l in
  nicolai_peaches_pounds = 6 :=
by
  sorry

end nicolai_peaches_pounds_l273_273572


namespace lucille_house_difference_l273_273094

def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

def average_height (h1 h2 h3 : ℕ) : ℕ := (h1 + h2 + h3) / 3

def difference (h_average h_actual : ℕ) : ℕ := h_average - h_actual

theorem lucille_house_difference :
  difference (average_height height_lucille height_neighbor1 height_neighbor2) height_lucille = 3 :=
by
  unfold difference
  unfold average_height
  sorry

end lucille_house_difference_l273_273094


namespace time_spent_cutting_hair_l273_273833

theorem time_spent_cutting_hair :
  let women's_time := 50
  let men's_time := 15
  let children's_time := 25
  let women's_haircuts := 3
  let men's_haircuts := 2
  let children's_haircuts := 3
  women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255 :=
by
  -- Definitions
  let women's_time       := 50
  let men's_time         := 15
  let children's_time    := 25
  let women's_haircuts   := 3
  let men's_haircuts     := 2
  let children's_haircuts := 3
  
  show women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255
  sorry

end time_spent_cutting_hair_l273_273833


namespace violet_children_count_l273_273976

theorem violet_children_count 
  (family_pass_cost : ℕ := 120)
  (adult_ticket_cost : ℕ := 35)
  (child_ticket_cost : ℕ := 20)
  (separate_ticket_total_cost : ℕ := 155)
  (adult_count : ℕ := 1) : 
  ∃ c : ℕ, 35 + 20 * c = 155 ∧ c = 6 :=
by
  sorry

end violet_children_count_l273_273976


namespace convert_binary_to_octal_l273_273675

theorem convert_binary_to_octal : binary_to_octal 1011001 = 131 := 
sorry

end convert_binary_to_octal_l273_273675


namespace tangent_parallel_to_line_l273_273968

open EuclideanGeometry

theorem tangent_parallel_to_line {α : Type*} [euclidean_space α]
  (C₁ C₂ : Set (euclidean_point α))
  (M N A B C : euclidean_point α)
  (hC₁ : metric.circle C₁)
  (hC₂ : metric.circle C₂)
  (h_intersection : M ∈ C₁ ∧ M ∈ C₂ ∧ N ∈ C₁ ∧ N ∈ C₂)
  (hA : A ∈ C₁ ∧ A ≠ M ∧ A ≠ N)
  (h_lines : B ∈ line_through A M ∧ B ∈ C₂ ∧ C ∈ line_through A N ∧ C ∈ C₂) :
  tangent_at_point C₁ A ∥ line_through B C := 
sorry

end tangent_parallel_to_line_l273_273968


namespace problem_solution_l273_273406

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log (2 - x) / Real.log 2 else 2 ^ (x - 1)

theorem problem_solution : f (-2) + f (Real.log 12 / Real.log 2) = 9 := by
  sorry

end problem_solution_l273_273406


namespace compare_exponents_l273_273669

theorem compare_exponents:
  let a := 0.7 ^ 0.7
  let b := 0.7 ^ 0.8
  let c := 0.8 ^ 0.7
  in c > a ∧ a > b :=
by
  let a := 0.7 ^ 0.7
  let b := 0.7 ^ 0.8
  let c := 0.8 ^ 0.7
  sorry

end compare_exponents_l273_273669


namespace trapezium_distance_l273_273314

def trapezium_area (a b h : ℝ) : ℝ :=
  1 / 2 * (a + b) * h

theorem trapezium_distance (a b area : ℝ) (h : ℝ): 
  a = 20 → b = 18 → area = 266 → trapezium_area a b h = area → h = 14 :=
by
  intros ha hb harea hdef
  rw [ha, hb, harea] at hdef
  sorry

end trapezium_distance_l273_273314


namespace perimeter_of_polygonal_region_l273_273821

noncomputable def polygonal_perimeter (right_angles : bool) (side_length : ℕ) (area : ℕ) : ℕ :=
if right_angles ∧ side_length = 1 ∧ area = 86 then 44 else 0

-- We are given:
variable (right_angles : bool) (side_length : ℕ) (area : ℕ)
-- Conditions as definitions
axiom h1 : right_angles = true
axiom h2 : side_length = 1
axiom h3 : area = 86

-- The theorem we want to prove:
theorem perimeter_of_polygonal_region : polygonal_perimeter right_angles side_length area = 44 :=
by
  rw [h1, h2, h3]
  -- Provide the proof steps here.
  sorry

end perimeter_of_polygonal_region_l273_273821


namespace length_BM_of_rectangle_l273_273358

theorem length_BM_of_rectangle
  (a b : ℝ) (A B C D K L M : EuclideanGeometry.Point) (AB BC CD DA : EuclideanGeometry.Segment)
  (ha : AB.length = a)
  (hb : BC.length = 1.5 * a)
  (hK : K.midpoint = EuclideanGeometry.midpoint A D)
  (hL : L.on (EuclideanGeometry.lineSegment C D) ∧ EuclideanGeometry.segmentLength C L = EuclideanGeometry.segmentLength A K)
  (hL_KL : EuclideanGeometry.segmentLength K L = 4)
  (hM : M.on ((EuclideanGeometry.line B L) ∩ (EuclideanGeometry.perpendicularBisector D L))) :
  EuclideanGeometry.segmentLength B M = Real.sqrt (a^2 + (1.5 * a)^2) :=
sorry

end length_BM_of_rectangle_l273_273358


namespace value_of_fraction_l273_273119

variable {x y : ℝ}

theorem value_of_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 * x + y) / (x - 3 * y) = -2) :
  (x + 3 * y) / (3 * x - y) = 2 :=
sorry

end value_of_fraction_l273_273119


namespace Missy_handles_41_claims_l273_273288

def claims_jan := 20
def claims_john := 1.3 * claims_jan
def claims_missy := claims_john + 15

theorem Missy_handles_41_claims : claims_missy = 41 :=
by
  -- proof goes here
  sorry

end Missy_handles_41_claims_l273_273288


namespace icosahedron_edge_painting_count_l273_273919

def icosahedron_edge_paintings : ℕ :=
  let edges := 30
  let faces := 20
  let colors := 3
  let choices_per_face := 2 in
  (choices_per_face ^ faces) * (colors ^ (edges - faces))

theorem icosahedron_edge_painting_count :
  icosahedron_edge_paintings = 2^20 * 3^10 :=
by
  -- Proof will go here
  sorry

end icosahedron_edge_painting_count_l273_273919


namespace tangent_angle_k_range_l273_273724

-- Define the circle and line, point P, and the angle condition
noncomputable def circle (x y : ℝ) := x^2 + y^2 = 1
noncomputable def line (x y k : ℝ) := x + y + k = 0

-- Proof problem statement
theorem tangent_angle_k_range (k : ℝ) :
  (∃ x y : ℝ, circle x y ∧ line x y k ∧ ∃ x y, (x^2 + y^2 = 4) ∧ (|k| / real.sqrt 2 ≤ 2)) →
  -2 * real.sqrt 2 ≤ k ∧ k ≤ 2 * real.sqrt 2 :=
by
  sorry

end tangent_angle_k_range_l273_273724


namespace remaining_volume_of_cube_with_hole_l273_273223

variable (side_length cube_volume : ℝ)
variable (radius height cylinder_volume remaining_volume : ℝ)
variable (π : ℝ) [fact (π > 0)]

noncomputable def volume_of_cube (a : ℝ) : ℝ := a^3
noncomputable def volume_of_cylinder (r h : ℝ) (π : ℝ) : ℝ := π * r^2 * h

theorem remaining_volume_of_cube_with_hole :
  let side_length := 6
  let radius := 3
  let height := 4
  let cube_volume := volume_of_cube side_length
  let cylinder_volume := volume_of_cylinder radius height π
  let remaining_volume := cube_volume - cylinder_volume
  remaining_volume = 216 - 36 * π := 
  by sorry

end remaining_volume_of_cube_with_hole_l273_273223


namespace triangle_inequality_inscribed_circle_l273_273847

-- Define the setup conditions for the problem
variables {A B C : ℝ} -- Angles of the triangle
variables {m_a m_b m_c M_a M_b M_c : ℝ} -- Lengths of the angle bisectors and extended bisectors
variables {sin_A sin_B sin_C : ℝ} -- Sine of the angles

-- Define the ratios of bisectors
def l_a : ℝ := m_a / M_a
def l_b : ℝ := m_b / M_b
def l_c : ℝ := m_c / M_c

-- Main theorem statement
theorem triangle_inequality_inscribed_circle
  (h1 : 0 < sin_A) (h2 : 0 < sin_B) (h3 : 0 < sin_C) -- Non-zero sine of angles
  (h_triangle : true) -- Some condition to represent triangle conditions (placeholder)
  : (l_a / sin_A^2) + (l_b / sin_B^2) + (l_c / sin_C^2) ≥ 3 :=
  sorry

end triangle_inequality_inscribed_circle_l273_273847


namespace line_through_point_bisects_chord_l273_273477

theorem line_through_point_bisects_chord 
  (x y : ℝ) 
  (h_parabola : y^2 = 16 * x) 
  (h_point : 8 * 2 - 1 - 15 = 0) :
  8 * x - y - 15 = 0 :=
by
  sorry

end line_through_point_bisects_chord_l273_273477


namespace find_rate_percent_l273_273607

-- Define the parameters given in the problem
def principal (P : ℝ) := P = 1750
def amount (A : ℝ) := A = 2000
def time (T : ℝ) := T = 3
def simple_interest (SI : ℝ) (P A : ℝ) := SI = A - P

-- Define the formula for rate
def rate (R : ℝ) (SI P T : ℝ) := R = (SI * 100) / (P * T)

-- The theorem to prove the rate R based on the given principal, amount, and time
theorem find_rate_percent (P A T SI R : ℝ) 
  (h1 : principal P) (h2 : amount A) (h3 : time T)
  (h4 : simple_interest SI P A) 
  (h5 : rate R SI P T) : R ≈ 4.76 / 100 :=
by 
  sorry

end find_rate_percent_l273_273607


namespace polynomials_same_roots_iff_constant_sign_l273_273103

noncomputable theory

open Complex

def sameRootsSameMultiplicities {P Q : ℂ[X]} : Prop :=
∀ z : ℂ, multiplicity P z = multiplicity Q z

def constantSignFunc (P Q : ℂ[X]) (f : ℂ → ℝ) : Prop :=
∀ z : ℂ, f(z) = abs (P.eval z) - abs (Q.eval z) → f(z) ≥ 0 ∨ f(z) ≤ 0

theorem polynomials_same_roots_iff_constant_sign (P Q : ℂ[X]) :
  (sameRootsSameMultiplicities P Q) ↔ (constantSignFunc P Q (λ z, abs (P.eval z) - abs (Q.eval z))) :=
sorry

end polynomials_same_roots_iff_constant_sign_l273_273103


namespace compute_x_l273_273792

theorem compute_x (x : ℝ) (h : log 2 (x^2) + log 4 x = 6) : x = 2^(12 / 5) :=
by sorry

end compute_x_l273_273792


namespace slower_truck_time_to_pass_driver_l273_273973

-- Definitions of constants and conditions
def length_of_truck : ℝ := 250 -- length of each truck in meters
def speed_fast_kmph : ℝ := 30 -- speed of the faster truck in km/hr
def speed_slow_kmph : ℝ := 20 -- speed of the slower truck in km/hr

-- Conversion factors
def km_to_m : ℝ := 1000 -- 1 km = 1000 meters
def hr_to_s : ℝ := 3600 -- 1 hour = 3600 seconds

-- Converted speeds in m/s
def speed_fast_mps : ℝ := (speed_fast_kmph * km_to_m) / hr_to_s
def speed_slow_mps : ℝ := (speed_slow_kmph * km_to_m) / hr_to_s

-- Relative speed when moving in opposite directions
def relative_speed : ℝ := speed_fast_mps + speed_slow_mps

-- Calculate time to pass
def time_to_pass : ℝ := length_of_truck / relative_speed

-- The theorem we want to prove
theorem slower_truck_time_to_pass_driver : time_to_pass ≈ 18 := 
by sorry

end slower_truck_time_to_pass_driver_l273_273973


namespace intersection_complement_eq_l273_273363

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Define the complement of B in U
def complement_B_in_U : Set ℕ := { x ∈ U | x ∉ B }

-- The main theorem statement stating the required equality
theorem intersection_complement_eq : A ∩ complement_B_in_U = {2, 3} := by
  sorry

end intersection_complement_eq_l273_273363


namespace four_letter_product_l273_273309

-- Define the value of each letter
def letter_value : Char → ℕ
| 'A' := 1
| 'B' := 2
| 'C' := 3
| 'D' := 4
| 'E' := 5
| 'F' := 6
| 'G' := 7
| 'H' := 8
| 'I' := 9
| 'J' := 10
| 'K' := 11
| 'L' := 12
| 'M' := 13
| 'N' := 14
| 'O' := 15
| 'P' := 16
| 'Q' := 17
| 'R' := 18
| 'S' := 19
| 'T' := 20
| 'U' := 21
| 'V' := 22
| 'W' := 23
| 'X' := 24
| 'Y' := 25
| 'Z' := 26
| _ := 0 -- For cases outside A-Z

-- Given condition: The product of values of 'B', 'C', 'D', 'E' is 2520
theorem four_letter_product (x1 x2 x3 x4 : Char) (h1 : x1 = 'E') (h2 : x2 = 'G') (h3 : x3 = 'H') (h4 : x4 = 'I') :
  letter_value 'B' * letter_value 'C' * letter_value 'D' * letter_value 'E' = 2520 →
  letter_value x1 * letter_value x2 * letter_value x3 * letter_value x4 = 2520 :=
by
  sorry -- Proof omitted

end four_letter_product_l273_273309


namespace find_a_value_l273_273207

theorem find_a_value
  (circle_polar : ∀ θ, ∃ ρ, ρ = 4 * cos θ)
  (line_polar : ∀ θ φ a, ∃ ρ, ρ * sin (θ - φ) = a)
  (chord_length : ∀ ρ1 ρ2 θ1 θ2, (ρ1 = 4 * cos θ1) → (ρ2 = 4 * cos θ2) → ρ1 * sin (θ1 - φ) = a → ρ2 * sin (θ2 - φ) = a → dist ((ρ1 * cos θ1, ρ1 * sin θ1), (ρ2 * cos θ2, ρ2 * sin θ2)) = 2) :
  (a = 0 ∨ a = -2) := 
begin
  sorry
end

end find_a_value_l273_273207


namespace sin_inequalities_l273_273601

theorem sin_inequalities (x : ℝ) (hx : 0 < x):
  sin x ≤ x ∧
  sin x ≥ x - (x^3) / 6 ∧
  sin x ≤ x - (x^3) / 6 + (x^5) / 120 ∧
  sin x ≥ x - (x^3) / 6 + (x^5) / 120 - (x^7) / 5040 :=
by
  sorry

end sin_inequalities_l273_273601


namespace prove_min_period_and_max_value_l273_273549

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem prove_min_period_and_max_value :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ y : ℝ, y ≤ f y) :=
by
  -- Proof goes here
  sorry

end prove_min_period_and_max_value_l273_273549


namespace part1_part2_l273_273727

noncomputable section

variable (f : ℝ → ℝ)

-- Conditions for the function f
def is_symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ (-1:ℝ)..1, f(x) = -f(-x)

def is_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ (-1:ℝ)..1, x < y → f(x) > f(y)

-- Domain restrictions
def in_domain (x : ℝ) : Prop :=
  x ∈ (-1:ℝ)..1

-- Part 1: Prove the inequality
theorem part1 (h_symmetric : is_symmetric_about_origin f) (h_decreasing : is_decreasing_on_interval f)
  {x₁ x₂ : ℝ} (hx_in_dom₁ : in_domain x₁) (hx_in_dom₂ : in_domain x₂) 
  (h_sum_not_zero : x₁ + x₂ ≠ 0) : 
  (f(x₁) + f(x₂)) / (x₁ + x₂) < 0 := 
sorry

-- Part 2: Find the range { -2 < m < 1}
theorem part2 (h_symmetric : is_symmetric_about_origin f) (h_decreasing : is_decreasing_on_interval f)
  {m : ℝ} 
  (h_cond : f(m^2 - 1) + f(m - 1) > 0) : 
  (-2 < m ∧ m < 1) := 
sorry

end part1_part2_l273_273727


namespace largest_digit_divisible_by_6_l273_273990

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273990


namespace total_profit_calc_l273_273196

variable (B_investment : ℝ) (B_period : ℝ) (total_profit : ℝ)

-- Given conditions
def A_investment := 3 * B_investment
def A_period := 2 * B_period
def B_profit := 7000
def B_share := B_investment * B_period
def A_share := A_investment * A_period

-- Define total share
def total_share := A_share + B_share

-- Define total profit calculation
theorem total_profit_calc (h : 7000 / total_profit = B_share / total_share) : total_profit = 49000 := by
  sorry

end total_profit_calc_l273_273196


namespace angles_triangle_inequality_l273_273753

theorem angles_triangle_inequality (α β γ : ℝ) (h : α + β + γ = π) : 
  2 * (sin α / α + sin β / β + sin γ / γ) ≤ 
  (1 / β + 1 / γ) * sin α + (1 / γ + 1 / α) * sin β + (1 / α + 1 / β) * sin γ :=
by
  sorry

end angles_triangle_inequality_l273_273753


namespace a_in_M_sufficient_not_necessary_l273_273371

-- Defining the sets M and N
def M := {x : ℝ | x^2 < 3 * x}
def N := {x : ℝ | abs (x - 1) < 2}

-- Stating that a ∈ M is a sufficient but not necessary condition for a ∈ N
theorem a_in_M_sufficient_not_necessary (a : ℝ) (h : a ∈ M) : a ∈ N :=
by sorry

end a_in_M_sufficient_not_necessary_l273_273371


namespace nicolai_peaches_pounds_l273_273573

-- Condition definitions
def total_fruit_pounds : ℕ := 8
def mario_oranges_ounces : ℕ := 8
def lydia_apples_ounces : ℕ := 24
def ounces_to_pounds (ounces: ℕ) : ℚ := ounces / 16 

-- Statement we want to prove
theorem nicolai_peaches_pounds :
  let mario_oranges_pounds := ounces_to_pounds mario_oranges_ounces,
      lydia_apples_pounds := ounces_to_pounds lydia_apples_ounces,
      eaten_by_m_and_l := mario_oranges_pounds + lydia_apples_pounds,
      nicolai_peaches_pounds := total_fruit_pounds - eaten_by_m_and_l in
  nicolai_peaches_pounds = 6 :=
by
  sorry

end nicolai_peaches_pounds_l273_273573


namespace arithmetic_square_root_25_cube_root_neg_27_sqrt_of_sqrt_36_l273_273920

-- Definitions
def sqrt (x : ℝ) : ℝ := Real.sqrt x
def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem arithmetic_square_root_25 : sqrt 25 = 5 :=
by
  sorry

theorem cube_root_neg_27 : cbrt (-27) = -3 :=
by
  sorry

theorem sqrt_of_sqrt_36 : sqrt (sqrt 36) = 6 :=
by
  sorry

end arithmetic_square_root_25_cube_root_neg_27_sqrt_of_sqrt_36_l273_273920


namespace instantaneous_velocity_at_t3_l273_273552

open Real

variable (t : ℝ)

def h : ℝ → ℝ := λ t, 15 * t - t ^ 2

theorem instantaneous_velocity_at_t3 : deriv h 3 = 9 := 
by
  sorry

end instantaneous_velocity_at_t3_l273_273552


namespace triangle_area_l273_273259

-- Given conditions
def ratio_5_12_13 (a b c : ℝ) : Prop := 
  (b / a = 12 / 5) ∧ (c / a = 13 / 5)

def right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

def circumscribed_circle (a b c r : ℝ) : Prop :=
  2 * r = c

-- The main theorem we need to prove
theorem triangle_area (a b c r : ℝ) (h_ratio : ratio_5_12_13 a b c) (h_triangle : right_triangle a b c) (h_circle : circumscribed_circle a b c r) (h_r : r = 5) :
  0.5 * a * b ≈ 17.75 :=
by
  sorry

end triangle_area_l273_273259


namespace expression_value_l273_273189

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
    sorry

end expression_value_l273_273189


namespace negation_of_forall_sin_le_one_l273_273775

theorem negation_of_forall_sin_le_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end negation_of_forall_sin_le_one_l273_273775


namespace donuts_per_student_l273_273887

theorem donuts_per_student (total_donuts : ℕ) (students : ℕ) (percentage_likes_donuts : ℚ) 
    (H1 : total_donuts = 4 * 12) 
    (H2 : students = 30) 
    (H3 : percentage_likes_donuts = 0.8) 
    (H4 : ∃ (likes_donuts : ℕ), likes_donuts = students * percentage_likes_donuts) : 
    (∃ (donuts_per_student : ℚ), donuts_per_student = total_donuts / (students * percentage_likes_donuts)) → donuts_per_student = 2 :=
by
    sorry

end donuts_per_student_l273_273887


namespace cube_faces_edges_vertices_sum_l273_273591

theorem cube_faces_edges_vertices_sum :
  ∀ (F E V : ℕ), F = 6 → E = 12 → V = 8 → F + E + V = 26 :=
by
  intros F E V F_eq E_eq V_eq
  rw [F_eq, E_eq, V_eq]
  rfl

end cube_faces_edges_vertices_sum_l273_273591


namespace relationship_among_abc_l273_273850

theorem relationship_among_abc (x : ℝ) (a b c : ℝ) (h1 : 1 < x) (h2 : x < 10)
  (h_a : a = (Real.log10 x)^2)
  (h_b : b = Real.log10 (x^2))
  (h_c : c = Real.log10 (Real.log10 x)) : c < a ∧ a < b :=
  sorry

end relationship_among_abc_l273_273850


namespace A_will_eat_hat_l273_273961

def Person : Type :=
{
  is_knight : Bool  -- Knight if true, otherwise liar
}

def statement_A : Person → Person → Prop := λ A B,
  (A.is_knight → B.is_knight)

def response_A : Person → Prop := λ A,
  (A.is_knight → True)

theorem A_will_eat_hat (A B : Person) (h1 : statement_A A B) (h2 : response_A A) : 
  (A.is_knight → True) :=
begin
  -- Proof construction goes here
  sorry
end

end A_will_eat_hat_l273_273961


namespace number_of_isosceles_triangles_l273_273827

variables (A B C D E F M : Type) (AB AC BD BC DE EF MF : Set)

def isosceles (x y z : Type) : Prop :=
  ∃ (u v : x ≠ y ∧ x ≠ z ∧ y ≠ z → Bool), u = v

axiom triangle_ABC_isosceles : AB = AC
axiom angle_ABC_80 : ∃ (α β : A ≠ B ∧ A ≠ C ∧ B ≠ C → ℝ), α = 80 ∧ β = 80
axiom BDBisects_ABC : ∃ (P : Type), BD = Set.bisect (α β : angle_ABC_80 α β)
axiom point_D_on_AC : ∃ (Q : Type), Q ∈ AC
axiom point_M_mid_AC : ∃ (R : Type), R = Set.midpoint AC
axiom DE_parallel_AB : ∃ (S : Type), DE ∥ AB
axiom EF_parallel_BD : ∃ (T : Type), EF ∥ BD
axiom MF_parallel_BC : ∃ (U : Type), MF ∥ BC

theorem number_of_isosceles_triangles : ∃ (n : ℕ), n = 7 := 
by 
  sorry

end number_of_isosceles_triangles_l273_273827


namespace minimal_abs_diff_l273_273437

theorem minimal_abs_diff (x y : ℕ) (h : x > 0 ∧ y > 0 ∧ x * y - 5 * x + 6 * y = 216) :
  |x - y| = 36 :=
sorry

end minimal_abs_diff_l273_273437


namespace find_a_values_l273_273440

theorem find_a_values (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 27 * x^3) (h₃ : a - b = 2 * x) :
  a = 3.041 * x ∨ a = -1.041 * x :=
by
  sorry

end find_a_values_l273_273440


namespace circle_equation_line_equations_l273_273725

noncomputable def circle_center (x : ℝ) := x > 0
noncomputable def radius : ℝ := 3
noncomputable def tangent_line (x y : ℝ) := 3 * x - 4 * y + 9 = 0

noncomputable def standard_equation_of_circle (x y : ℝ) := (x - 2)^2 + y^2 = 9

-- First part: Prove the standard equation of circle M
theorem circle_equation (a : ℝ) (h1 : circle_center a) (h2 : tangent_line a 0) : standard_equation_of_circle = (λ x y, (x - 2)^2 + y^2 = 9) :=
sorry

noncomputable def line_passing_through_point (x y k : ℝ) := y = k * x - 3 ∨ x = 0
noncomputable def intersection_condition (x1 x2 : ℝ) := x1^2 + x2^2 = (21 / 2) * x1 * x2
noncomputable def equations_of_line_l (x y : ℝ) := (x - y - 3 = 0) ∨ (17 * x - 7 * y - 21 = 0) ∨ (x = 0)

-- Second part: Prove the equation of line l
theorem line_equations (N : ℝ × ℝ) (h1 : N = (0, -3)) (x1 x2 : ℝ) (h2 : intersection_condition x1 x2) : 
    ∃ (k : ℝ), line_passing_through_point 0 (-3) k ∧ equations_of_line_l 0 (-3) :=
sorry

end circle_equation_line_equations_l273_273725


namespace midpoints_parallelogram_or_line_l273_273236

-- Definitions:
variables (A B C D M N P Q : Type) [AddGroup M] [AddGroup N]
variables [AffineSpace M A] [AffineSpace N B] [AffineSpace N C] [AffineSpace N D]
variables (midpoint : ∀ {X : Type} [AddGroup X] [Module ℝ X], A → A → X)

-- Midpoints of sides and diagonals
def is_midpoint (P : M) (A B : M) : Prop := 2 • P = A + B

-- Theorem stating that midpoints form a parallelogram or lie on a straight line
theorem midpoints_parallelogram_or_line (A B C D : M)
  (M : M) (N : M) (P : M) (Q : M)
  (hM : is_midpoint midpoint A C M)
  (hN : is_midpoint midpoint B D N)
  (hP : is_midpoint midpoint A B P)
  (hQ : is_midpoint midpoint C D Q) :
  ∃ M' N' : M, is_midpoint midpoint M' M N ∧ is_midpoint midpoint N' P Q :=
begin
  sorry -- proof to be written
end

end midpoints_parallelogram_or_line_l273_273236


namespace balls_in_boxes_l273_273895

theorem balls_in_boxes : 
  let boxes := {box1, box2, box3}
  let balls := {1, 2, 3, 4, 5, 6}
  ∃ (arrangement : Set (Set ℕ)), arrangement.card = 3 ∧ 
  (∀ b ∈ arrangement, b.card = 2) ∧ 
  ({1, 2} ∈ arrangement) ∧ 
  (∀ b ∈ arrangement, b ⊆ balls) ∧ 
  ∀ b₁ b₂ ∈ arrangement, b₁ ≠ b₂ → b₁ ∩ b₂ = ∅ 
  → arrangement.count = 18 :=
sorry

end balls_in_boxes_l273_273895


namespace find_circle_equation_l273_273761

open Real

-- Definitions corresponding to the conditions
def circle1 (x y : ℝ) : Prop := (x-2)^2 + (y-1)^2 = r^2

def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 3*x = 0

def passes_through (x y : ℝ) (a b : ℝ) : Prop := x + 2*y - 5 + (circle1 (5) (-2)).radius^2 = 0

-- Formalizing the proof problem
theorem find_circle_equation (r : ℝ) :
  (circle1 (5) (-2)) 
  ∧ (∃ x y, circle1 x y ∧ circle2 x y)
  ∧ passes_through (5) (-2) :=
sorry

end find_circle_equation_l273_273761


namespace total_bill_correct_l273_273521

def scoop_cost : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

def pierre_total : ℕ := pierre_scoops * scoop_cost
def mom_total : ℕ := mom_scoops * scoop_cost
def total_bill : ℕ := pierre_total + mom_total

theorem total_bill_correct : total_bill = 14 :=
by
  sorry

end total_bill_correct_l273_273521


namespace total_number_of_meals_l273_273175

-- Define the conditions: meat choices, vegetable choices, dessert choices, drink choices
def meats := {beef, chicken, pork}
def vegetables := {baked_beans, corn, potatoes, tomatoes, spinach}
def desserts := {brownies, chocolate_cake, chocolate_pudding, ice_cream}
def drinks := {water, soda, juice}

-- Calculate the number of ways to choose meals, given the conditions
theorem total_number_of_meals : 
  (finset.card meats) * (nat.choose (finset.card vegetables) 3) * (finset.card desserts) * (finset.card drinks) = 360 :=
by sorry

end total_number_of_meals_l273_273175


namespace closest_multiple_of_18_l273_273599

def is_multiple_of_2 (n : ℤ) : Prop := n % 2 = 0
def is_multiple_of_9 (n : ℤ) : Prop := n % 9 = 0
def is_multiple_of_18 (n : ℤ) : Prop := is_multiple_of_2 n ∧ is_multiple_of_9 n

theorem closest_multiple_of_18 (n : ℤ) (h : n = 2509) : 
  ∃ k : ℤ, is_multiple_of_18 k ∧ (abs (2509 - k) = 7) :=
sorry

end closest_multiple_of_18_l273_273599


namespace time_to_pass_train_l273_273610

-- Define the conditions
def train_length : ℝ := 120 -- meters
def speed_kmph : ℝ := 60   -- kilometers per hour

-- Convert speed from km/h to m/s
def speed_mps : ℝ := speed_kmph * (1000 / 3600)

-- Calculate time taken for the train to pass the electric pole
def time_to_pass (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

-- Statement to prove that the time taken is approximately 7.2 seconds
theorem time_to_pass_train :
  time_to_pass train_length speed_mps ≈ 7.2 := 
by
  -- proof would go here
  sorry

end time_to_pass_train_l273_273610


namespace find_a_l273_273742

theorem find_a (a : ℝ) (h_a_pos : a > 0) : 
  (let C := ∀ x y : ℝ, (x - a)^2 + (y - 2)^2 = 4 in
   let focus := (-3, 0) in
   let l := ∀ x y : ℝ, x - y + 3 = 0 in
   let chord_length := 2 * sqrt 3 in
   by 
     sorry) →
   a = sqrt 2 - 1 := 
sorry

end find_a_l273_273742


namespace isosceles_trapezoid_inscribed_circle_l273_273741

noncomputable def radius_and_area_of_trapezoid :
  (R : ℝ) × (S : ℝ) :=
  let AD := 40,
      BC := 16 + 9,  -- CD = CE + ED
      CD := 25 := CE + ED,
      x := BC in
  let DW := 15 := real.sqrt (DE * (DE + CE))
  let CW := 20 := real.sqrt (25 * 25 - 15 * 15) in
  let R := 10 := CW / 2 in
  let height := 20 := 2 * R in
  let S := (1 / 2) * (AD + BC) * height in
  (R, S := 400)

theorem isosceles_trapezoid_inscribed_circle (A B C D : ℝ) (R : ℝ) (S: ℝ) 
  (AD : A D) (BC : B C) (CE : C E = 16) (ED : E D = 9):
  is_isosceles_trapezoid A B C D AD BC →
  inscribed_circle Ω B A D →
  touches_segment Ω BC C →
  intersects_line Ω CD E →
  (R = 10) ∧ (S = 400) :=
begin
  sorry
end

end isosceles_trapezoid_inscribed_circle_l273_273741


namespace median_length_from_A_l273_273872

noncomputable def median_length (a : ℝ) (B C : E) [InnerProductSpace ℝ E] : ℝ :=
  let D := (B + C) / 2 in
  (a * real.sqrt 3) / 2

theorem median_length_from_A {a : ℝ} {A B C : E} [InnerProductSpace ℝ E]
  (hBC : ∥B - C∥ = a)
  (h_angle_BAC : angle B A C = 35)
  (h_angle_BOC : angle B (A + (B + C) / 2) C = 145) :
  median_length a B C = a * real.sqrt 3 / 2 :=
by
  sorry

end median_length_from_A_l273_273872


namespace reflection_is_involution_l273_273853

variable (S : Matrix (Fin 2) (Fin 2) ℝ) (u : Matrix (Fin 2) (Fin 1) ℝ)

def reflection_matrix (v : Matrix (Fin 2) (Fin 1) ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  let denom := (vᵀ.mul v) (0, 0)
  2 • (v.mul vᵀ / denom) - 1

theorem reflection_is_involution (v : Matrix (Fin 2) (Fin 1) ℝ) (hv : reflection_matrix v = S) :
  S.mul S = 1 := by
  sorry

-- Example vector and reflection matrix for context
def example_vector : Matrix (Fin 2) (Fin 1) ℝ := ![![4], ![2]]
def example_matrix := reflection_matrix example_vector
#eval example_matrix -- To see the matrix for reflecting over [4, 2]

-- Proving specific case where v = [4, 2]
example : (reflection_matrix example_vector).mul (reflection_matrix example_vector) = 1 := by
  rw [← reflection_is_involution example_vector]
  sorry

end reflection_is_involution_l273_273853


namespace rightmost_nonzero_digit_odd_for_k_l273_273706

def a (n m : ℕ) : ℕ := (nat.factorial (n + m)) / (nat.factorial (n - 1))

def is_rightmost_nonzero_digit_odd (n m : ℕ) : Prop :=
  let an := a n m in
  an % 10 ≠ 0 ∧ (an % 10) % 2 = 1

theorem rightmost_nonzero_digit_odd_for_k :
  ∃ k : ℕ, is_rightmost_nonzero_digit_odd k (k + 10) ∧ ∀ j : ℕ, j < k → ¬ is_rightmost_nonzero_digit_odd j (j + 10) :=
sorry   

end rightmost_nonzero_digit_odd_for_k_l273_273706


namespace find_k_value_l273_273328

theorem find_k_value (x : ℝ) (h : x ≠ 0) : 
  (f : ℝ → ℝ) := λ x, (real.cot (x / 3) - real.cot (3 * x)) →
  f x = (real.sin (8 * x / 3)) / (real.sin (x / 3) * real.sin (3 * x)) :=
begin
  sorry
end

end find_k_value_l273_273328


namespace congruent_quadrilaterals_l273_273548

variable {Point : Type} [EuclideanGeometry Point]

variable (A B C D E F G H I J K L P1 Q1 R1 S1 P2 Q2 R2 S2 : Point)

variables (AC BD : Line)

-- Conditions
axiom AC_perp_BD : AC ⊥ BD
axiom square_ABEF : is_square A B E F
axiom square_BCGH : is_square B C G H
axiom square_CDIJ : is_square C D I J
axiom square_DAKL : is_square D A K L

-- Line Intersections
axiom CL_intersect_DF_AH_BJ : intersection_points CL DF AH BJ P1 Q1 R1 S1
axiom AI_intersect_BK_CE_DG : intersection_points AI BK CE DG P2 Q2 R2 S2

theorem congruent_quadrilaterals :
  quadrilateral_congruent P1 Q1 R1 S1 P2 Q2 R2 S2 :=
sorry

end congruent_quadrilaterals_l273_273548


namespace correct_judgments_l273_273538

noncomputable theory

def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

-- The analytical expression of the function is correct
def condition1 : Prop := ∀ x, f x = 2 * sin (2 * x + π / 6)

-- The graph of the function is symmetric about the point (π / 3, 0)
def condition2 : Prop := f (π / 3 - x) = -f (π / 3 + x)

-- The function is increasing on the interval [0, π / 6]
def condition3 : Prop := ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 6 → f x < f y

-- If the function y = f(x) + a has a minimum value of √3 on [0, π / 2], then a = 2√3
def condition4 (a : ℝ) : Prop := 
  (∀ x ∈ Icc (0 : ℝ) (π / 2), f x + a ≥ sqrt 3) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (π / 2), f x + a = sqrt 3) → a = 2 * sqrt 3

theorem correct_judgments : 
  ¬ condition1 ∧ 
  condition2 ∧ 
  ¬ condition3 ∧ 
  condition4 :=
by sorry

end correct_judgments_l273_273538


namespace angle_PED_eq_angle_PFD_l273_273877

-- Define the triangle and the necessary points and circles
variables {A B C P E F D : Type}

-- Assume the geometry setup conditions
variable (Triangle : A → B → C → Prop)              -- ABC is a triangle
variable (BisectorExternalAngleA : A → B → C → P → Prop) -- P is on the bisector of the external angle at A
variable (CircumcircleABC : A → B → C → P → Prop)    -- P is on the circumcircle of ABC
variable (CirclePassingAP : A → P → (BP: Segment) → (CP : Segment) → E → F → Prop) -- E and F are intersections
variable (AngleBisectorAD : A → B → C → D → Prop)    -- AD is the angle bisector of triangle

-- Prove the target angles are equal
theorem angle_PED_eq_angle_PFD 
  (h1 : Triangle A B C)
  (h2 : BisectorExternalAngleA A B C P)
  (h3 : CircumcircleABC A B C P)
  (h4 : CirclePassingAP A P (λ (BP : Segment)) (λ (CP : Segment)) E F)
  (h5 : AngleBisectorAD A B C D) : 
  ∠ PED = ∠ PFD :=
by 
  sorry

end angle_PED_eq_angle_PFD_l273_273877


namespace no_root_of_equation_l273_273283

theorem no_root_of_equation : ∀ x : ℝ, x - 8 / (x - 4) ≠ 4 - 8 / (x - 4) :=
by
  intro x
  -- Original equation:
  -- x - 8 / (x - 4) = 4 - 8 / (x - 4)
  -- No valid value of x solves the above equation as shown in the given solution
  sorry

end no_root_of_equation_l273_273283


namespace lucille_house_difference_l273_273091

-- Define the heights of the houses as given in the conditions.
def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

-- Define the total height of the houses.
def total_height : ℕ := height_neighbor1 + height_lucille + height_neighbor2

-- Define the average height of the houses.
def average_height : ℕ := total_height / 3

-- Define the height difference between Lucille's house and the average height.
def height_difference : ℕ := average_height - height_lucille

-- The theorem to prove.
theorem lucille_house_difference :
  height_difference = 3 := by
  sorry

end lucille_house_difference_l273_273091


namespace max_value_of_k_l273_273849

theorem max_value_of_k (m : ℝ) (k : ℝ) (h1 : 0 < m) (h2 : m < 1/2) 
  (h3 : ∀ m, 0 < m → m < 1/2 → (1 / m + 2 / (1 - 2 * m) ≥ k)) : k = 8 :=
sorry

end max_value_of_k_l273_273849


namespace parabola_y_intercepts_l273_273784

theorem parabola_y_intercepts :
  let f : ℝ → ℝ := λ y, 3 * y^2 - 5 * y + 2
  ∃ y1 y2 : ℝ, f y1 = 0 ∧ f y2 = 0 ∧ y1 ≠ y2 :=
by
  sorry

end parabola_y_intercepts_l273_273784


namespace problem_solution_l273_273795

theorem problem_solution (a b c d e f g : ℝ) 
  (h1 : a + b + e = 7)
  (h2 : b + c + f = 10)
  (h3 : c + d + g = 6)
  (h4 : e + f + g = 9) : 
  a + d + g = 6 := 
sorry

end problem_solution_l273_273795


namespace math_problem_l273_273402

def f (x : ℝ) : ℝ := Real.log (Real.abs (x - 1))

theorem math_problem {
  domain_not_R : ∀ x : ℝ, x ≠ 1 → x ∈ (set.univ \ {1 : ℝ}),
  monotonicity : ∀ x : ℝ, (x < 1 → f x = Real.log (1 - x)) ∧ (x > 1 → f x = Real.log (x - 1)),
  not_symmetric_about_y : ¬∀ x : ℝ, f x = f (-x),
  even_function : ∀ x : ℝ, f (x + 1) = f (-(x + 1)),
  fa_greater_0_altrelaion : ∀ a : ℝ, f a > 0 → a < 0 ∨ 2 < a
} :
  ((∀ x : ℝ, x ≠ 1 → x ∈ (set.univ \ {1 : ℝ})) ∧
  (∀ x : ℝ, (x < 1 → f x = Real.log (1 - x)) ∧ (x > 1 → f x = Real.log (x - 1))) ∧
  ¬∀ x : ℝ, f x = f (-x) ∧
  (∀ x : ℝ, f (x + 1) = f (-(x + 1))) ∧
  (∀ a : ℝ, f a > 0 → a < 0 ∨ 2 < a)) :=
sorry

end math_problem_l273_273402


namespace find_n_l273_273791

theorem find_n (n : ℚ) (h : 5 ^ (5 * n) = (1 / 5) ^ (2 * n - 15)) : n = 15 / 7 :=
sorry

end find_n_l273_273791


namespace intersecting_lines_exist_l273_273048

noncomputable def exists_intersecting_line (n : ℕ) (segments : Fin 4n → ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∃ l : ℝ × ℝ, ∃ p : ℝ × ℝ, (∃ i j : Fin 4n, i ≠ j ∧ (segments i).1 ≠ (segments j).1 ∧
    (l = (p.1, 0) ∨ l = (0, p.2)) ∧
    ((p ∈ (segments i)) ∧ (p ∈ (segments j)) ∧
    ((p.1 = (segments i).1 ∨ p.1 = (segments j).1) ∨ 
     (p.2 = (segments i).2 ∨ p.2 = (segments j).2))))

theorem intersecting_lines_exist 
  (n : ℕ) 
  (h_radius : ∀ i : Fin 4n, ∥(segments i).2 - (segments i).1∥ = 1) : 
  exists_intersecting_line n segments :=
begin
  sorry
end

end intersecting_lines_exist_l273_273048


namespace parabola_equation_moving_line_fixed_point_l273_273730

variable {A B C P Q : Type}
variables {x y : ℝ}
variables {S : Type} [parabola S]
variables {l : Type} [line l]

-- Conditions for the problem
def is_vertex_origin (S : parabola) : Prop := 
  ∃ vertex, vertex.x = 0 ∧ vertex.y = 0

def is_focus_on_x_axis (S : parabola) : Prop := 
  ∃ focus, focus.y = 0

def vertices_on_parabola (S : parabola) (A B C : Type) : Prop :=
  ∃ (xA yA xB yB xC yC : ℝ), (yA ^ 2 = 16 * xA) ∧ (yB ^ 2 = 16 * xB) ∧ (yC ^ 2 = 16 * xC)

def centroid_at_focus (S : parabola) (A B C : Type) : Prop :=
  let focus := classical.some (is_focus_on_x_axis S) in
  ∃ (cx cy : ℝ), cx = (xA + xB + xC) / 3 ∧ cy = (yA + yB + yC) / 3 ∧ cx = focus.x ∧ cy = focus.y

def line_contains_BC (l : line) : Prop := 
  ∃ l_eq : ℝ, l_eq = 4 * x + y - 20

-- First formalization of the problem to find the equation of the parabola S
theorem parabola_equation (S : parabola) (A B C : Type) (l : line)
  (h1 : is_vertex_origin S) (h2 : is_focus_on_x_axis S)
  (h3 : vertices_on_parabola S A B C) (h4 : centroid_at_focus S A B C)
  (h5 : line_contains_BC l) : 
  ∃ a : ℝ, ∃ b : ℝ, a = 16 ∧ y ^ 2 = a * x ∧ l = 4 * x + y - 20 := 
sorry

-- Second formalization of the problem to prove that moving line passes through a fixed point
theorem moving_line_fixed_point (P Q : Type)
  (S : parabola) (h1 : is_vertex_origin S) (h2 : is_focus_on_x_axis S)
  (h_perp : PO ⟂ OQ)
  (h3: vertices_on_parabola S P Q) : ∃ M : Type, M = (16, 0) ∧ PQ_passes M :=
sorry

end parabola_equation_moving_line_fixed_point_l273_273730


namespace find_integer_pair_l273_273313

theorem find_integer_pair (x y : ℤ) :
  (x + 2)^4 - x^4 = y^3 → (x = -1 ∧ y = 0) :=
by
  intro h
  sorry

end find_integer_pair_l273_273313


namespace angle_bisectors_geq_nine_times_inradius_l273_273947

theorem angle_bisectors_geq_nine_times_inradius 
  (r : ℝ) (f_a f_b f_c : ℝ) 
  (h_triangle : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ r = (1 / 2) * (a + b + c) * r ∧ 
      f_a ≥ (2 * a * b / (a + b) + 2 * a * c / (a + c)) / 2 ∧ 
      f_b ≥ (2 * b * a / (b + a) + 2 * b * c / (b + c)) / 2 ∧ 
      f_c ≥ (2 * c * a / (c + a) + 2 * c * b / (c + b)) / 2)
  : f_a + f_b + f_c ≥ 9 * r :=
sorry

end angle_bisectors_geq_nine_times_inradius_l273_273947


namespace solve_for_x_l273_273543

theorem solve_for_x (x : ℝ) (h : real.cbrt (30*x + real.cbrt (30*x + 27)) = 15) : 
  x = 1674 / 15 :=
  sorry

end solve_for_x_l273_273543


namespace number_of_squares_l273_273205

theorem number_of_squares (n : ℕ) : 
  let grid_points := { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n }
  in let num_squares := ∑ k in finset.range n, k * (n + 1 - k) * (n + 1 - k)
  in num_squares = n * (n + 1) * (n + 1) * (n + 2) / 12 := 
sorry

end number_of_squares_l273_273205


namespace proof_problem_l273_273356

noncomputable def geometric_seq (a_n : ℕ → ℝ) := ∀ n : ℕ+, 6 * (∑ i in finset.range n, a_n i) = 3^(n + 1) - 3

noncomputable def bn_val (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) := ∀ n : ℕ+, b_n n = (1 - 3 * n) * real.logb 3 ((a_n n)^2 * a_n (n + 1))

noncomputable def sum_bn (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) := ∀ n : ℕ, T_n n = ∑ i in finset.range (Nat.succ n), 1 / b_n i

theorem proof_problem :
  ∃ a : ℝ, ∃ (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ),
  (geometric_seq a_n) ∧
  (bn_val a_n b_n) ∧
  (sum_bn b_n T_n) ∧
  a = -3 ∧
  (∀ n : ℕ+, a_n n = 3^(n - 1)) ∧
  (∀ n : ℕ, T_n n = n / (3 * n + 1)) := sorry

end proof_problem_l273_273356


namespace find_m_values_l273_273762

-- Definition of an ellipse equation and its eccentricity condition
def ellipse_condition (m : ℝ) : Prop :=
  let a := if m < 4 then sqrt (1 / m) else 1 / 2 in
  let c := if m < 4 then sqrt (m * (4 - m)) / (2 * m) else sqrt (m * (m - 4)) / (2 * m) in
  let e := if m < 4 then c / (sqrt (1 / m)) else c / (1 / 2) in
  e = sqrt 2 / 2

-- The main theorem we want to prove
theorem find_m_values (m : ℝ) : ellipse_condition m ↔ m = 2 ∨ m = 8 :=
sorry

end find_m_values_l273_273762


namespace count_consecutive_ascending_numbers_l273_273004

open Nat

def is_consecutive_ascending (n : ℕ) : Prop :=
  (n ≥ 10) ∧ (n ≤ 13000) ∧ (let ds := digits 10 n in ∀ i, i < length ds - 1 → ds.get! i + 1 = ds.get! (i + 1))

theorem count_consecutive_ascending_numbers : ∃ (count : ℕ), count = 22 ∧ 
  (∀ n, n ≥ 10 ∧ n ≤ 13000 → is_consecutive_ascending n → n ∈ range 13000) :=
  sorry

end count_consecutive_ascending_numbers_l273_273004


namespace circles_polar_equiv_and_max_dist_product_l273_273815

-- Define parametric equations for circles C1 and C2
def circle_C1_parametric (φ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos φ, 2 * Real.sin φ)

def circle_C2_parametric (β : ℝ) : ℝ × ℝ :=
  (Real.cos β, 1 + Real.sin β)

-- Define polar equations for circles C1 and C2 (to be proved)
def circle_C1_polar (θ : ℝ) : ℝ :=
  4 * Real.cos θ

def circle_C2_polar (θ : ℝ) : ℝ :=
  2 * Real.sin θ

-- Main theorem to prove the polar forms and the maximum product of distances
theorem circles_polar_equiv_and_max_dist_product:
  (∀ (φ θ β : ℝ),
    let C1_point := circle_C1_parametric φ;
    let C2_point := circle_C2_parametric β in
    (C1_point.fst = 2 + 2 * Real.cos φ ∧ C1_point.snd = 2 * Real.sin φ) ∧
    (C2_point.fst = Real.cos β ∧ C2_point.snd = 1 + Real.sin β) ∧
    (circle_C1_polar θ = 4 * Real.cos θ) ∧
    (circle_C2_polar θ = 2 * Real.sin θ)) ∧ 
  (∀ (α : ℝ), 
    let ρ1 := circle_C1_polar α;
    let ρ2 := circle_C2_polar α in
    ρ1 * ρ2 ≤ 4) :=
sorry

end circles_polar_equiv_and_max_dist_product_l273_273815


namespace relationship_among_abc_l273_273501

theorem relationship_among_abc {f : ℝ → ℝ} (hf : ∀ x > 0, x * deriv f x - f x < 0) :
  let a := f ((Real.log 5) / (Real.log 2)) / ((Real.log 5) / (Real.log 2)),
      b := f (2 ^ 0.2) / (2 ^ 0.2),
      c := f (0.2 ^ 2) / (0.2 ^ 2) in a < b ∧ b < c :=
by
  sorry

end relationship_among_abc_l273_273501


namespace farmer_has_42_cows_left_l273_273232

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end farmer_has_42_cows_left_l273_273232


namespace calc_2a_plus_b_dot_b_l273_273377

open Real

variables (a b : ℝ^3)
variables (cos_angle : ℝ)
variables (norm_a norm_b : ℝ)

-- Conditions from the problem
axiom cos_angle_is_1_3 : cos_angle = 1 / 3
axiom norm_a_is_1 : ∥a∥ = 1
axiom norm_b_is_3 : ∥b∥ = 3

-- Dot product calculation
axiom a_dot_b_is_1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle

-- Prove the target
theorem calc_2a_plus_b_dot_b : 
  (2 * a + b).dot b = 11 :=
by
  -- These assertions setup the problem statement in Lean
  have h1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle, 
  from a_dot_b_is_1,
  have h2 : ∥a∥ = 1, from norm_a_is_1,
  have h3 : ∥b∥ = 3, from norm_b_is_3,
  have h4 : cos_angle = 1 / 3, from cos_angle_is_1_3,
  sorry

end calc_2a_plus_b_dot_b_l273_273377


namespace _l273_273525

noncomputable def polynomial_divides (x : ℂ) (n : ℕ) : Prop :=
  (x - 1) ^ 3 ∣ x ^ (2 * n + 1) - (2 * n + 1) * x ^ (n + 1) + (2 * n + 1) * x ^ n - 1

lemma polynomial_division_theorem : ∀ (n : ℕ), n ≥ 1 → ∀ (x : ℂ), polynomial_divides x n :=
by
  intros n hn x
  unfold polynomial_divides
  sorry

end _l273_273525


namespace julie_school_year_work_hours_l273_273308

theorem julie_school_year_work_hours :
  ∀ (hours_per_week_summer weeks_summer earnings_summer weeks_school_year earnings_school_year : ℝ),
  hours_per_week_summer = 48 →
  weeks_summer = 10 →
  earnings_summer = 5000 →
  weeks_school_year = 40 →
  earnings_school_year = 6000 →
  let total_hours_summer := hours_per_week_summer * weeks_summer in
  let hourly_wage := earnings_summer / total_hours_summer in
  let total_hours_school_year := earnings_school_year / hourly_wage in
  let hours_per_week_school_year := total_hours_school_year / weeks_school_year in
  hours_per_week_school_year = 14.4 :=
by
  intros hours_per_week_summer weeks_summer earnings_summer weeks_school_year earnings_school_year
  sorry

end julie_school_year_work_hours_l273_273308


namespace mathematicians_cafeteria_break_l273_273338

theorem mathematicians_cafeteria_break (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_square_free : ∀ p : ℕ, prime p → p^2 ∣ z → false) 
  (hn : n = x - y * real.sqrt z) 
  (h_prob : ∀ n, probability (at_least_one_overlap n) = 0.5) : 
  x = 60 ∧ y = 30 ∧ z = 2 ∧ x + y + z = 92 :=
begin
  sorry
end

end mathematicians_cafeteria_break_l273_273338


namespace horner_polynomial_rewrite_polynomial_value_at_5_l273_273975

def polynomial (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 6 * x^3 - 2 * x^2 - 5 * x - 2

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_polynomial_rewrite :
  polynomial = horner_polynomial := 
sorry

theorem polynomial_value_at_5 :
  polynomial 5 = 7548 := 
sorry

end horner_polynomial_rewrite_polynomial_value_at_5_l273_273975


namespace angle_condition_l273_273851

axiom PointsLine (A B C : Type) : Type

variables {Point : Type} [PointSpace : PointsLine Point]

variables {A B C B1 A1 O : Point}

-- Define midpoints B1 and A1
def is_midpoint (M P Q : Point) : Prop := ∃ P' Q', (P' + Q') / 2 = M ∧ P' = P ∧ Q' = Q

-- Define incenter O
def is_incenter (O : Point) (A B C : Point) : Prop := -- Include properties of the incenter

-- Define the angle measure
def angle (A B C : Point) : Type := -- Suitable angle implementation

-- Given conditions
variables (triangleABC : triangle A B C) (midpoint_B1 : is_midpoint B1 C A) (midpoint_A1 : is_midpoint A1 C B) (incenter_O : is_incenter O A B C)

-- Helper for angle equality given the sides condition.
def sides_condition (a b c : ℝ) : Prop := c / 2 < b ∧ b ≤ c ∧ a = 2 * c - b

-- Main theorem statement
theorem angle_condition (h1 : ∃ a b c : ℝ, sides_condition a b c) :
  angle B1 O A1 = angle B A C + angle A B C :=
sorry

end angle_condition_l273_273851


namespace find_a12_l273_273464

variable {α : Type*} [AddGroup α] [Module ℚ α]

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℚ) (n : ℕ) : ℚ := a + n * d

-- Given conditions
variable (a d : ℚ)
variable (h1 : arithmetic_seq a d 3 = 1)
variable (h2 : arithmetic_seq a d 6 + arithmetic_seq a d 8 = 16)

theorem find_a12 : arithmetic_seq a d 11 = 15 := by
  sorry

end find_a12_l273_273464


namespace C_recurrence_S_recurrence_l273_273614

noncomputable def C (x : ℝ) : ℝ := 2 * Real.cos x
noncomputable def C_n (n : ℕ) (x : ℝ) : ℝ := 2 * Real.cos (n * x)
noncomputable def S_n (n : ℕ) (x : ℝ) : ℝ := Real.sin (n * x) / Real.sin x

theorem C_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  C_n n x = C x * C_n (n - 1) x - C_n (n - 2) x := sorry

theorem S_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  S_n n x = C x * S_n (n - 1) x - S_n (n - 2) x := sorry

end C_recurrence_S_recurrence_l273_273614


namespace calculate_longer_worm_length_l273_273120

variables (shorter_worm_length : ℝ) (length_difference : ℝ)

def longer_worm_length (shorter_worm_length : ℝ) (length_difference : ℝ) : ℝ :=
  shorter_worm_length + length_difference

theorem calculate_longer_worm_length (shorter_worm_length : ℝ) (length_difference : ℝ) :
  shorter_worm_length = 0.1 → length_difference = 0.7 → longer_worm_length shorter_worm_length length_difference = 0.8 :=
by
  intros h_shorter h_difference
  rw [h_shorter, h_difference]
  exact rfl

end calculate_longer_worm_length_l273_273120


namespace dessert_menu_count_l273_273221

def dessert : Type := {dessert // dessert ∈ ["cake", "pie", "ice cream", "pudding", "cookies"]}

def valid_menu (menu : Fin 7 → dessert) : Prop :=
  (∀ i : Fin 6, menu i ≠ menu (i + 1)) ∧ menu 3 = "pie"

theorem dessert_menu_count :
  ∃ (menu : (Fin 7 → dessert)), valid_menu menu ∧ (number_of_valid_menus = 40960) :=
sorry

end dessert_menu_count_l273_273221


namespace trig_identity_l273_273685

theorem trig_identity :
  let sin := Real.sin
  let cos := Real.cos
  (sin 163 * sin 223 + sin 253 * sin 313 = 1 / 2) := 
  by {
    have h1 : sin 253 = cos 163 := sorry,
    have h2 : sin 313 = cos 223 := sorry,
    sorry
  }

end trig_identity_l273_273685


namespace largest_digit_divisible_by_6_l273_273991

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l273_273991


namespace greatest_cds_in_box_l273_273518

theorem greatest_cds_in_box (r c p n : ℕ) (hr : r = 14) (hc : c = 12) (hp : p = 8) (hn : n = 2) :
  n = Nat.gcd r (Nat.gcd c p) :=
by
  rw [hr, hc, hp]
  sorry

end greatest_cds_in_box_l273_273518


namespace solution_set_l273_273355

-- Define f to be a function on ℝ
variable (f : ℝ → ℝ)

-- Condition 1: f is strictly increasing
def strict_increasing (g : ℝ → ℝ) : Prop :=
∀ (x1 x2 : ℝ), x1 < x2 → g(x1) < g(x2)

-- Condition 2: f is symmetric about the point (1, 0)
def symmetric_about_one (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g(x + 1) = -g(1 - x)

-- The proof problem
theorem solution_set (h1 : strict_increasing f) (h2 : symmetric_about_one f) :
  set_of (λ x : ℝ, f(1 - x) > 0) = { x : ℝ | x < 0 } :=
sorry

end solution_set_l273_273355


namespace probability_correct_l273_273946

-- Define the set of segment lengths
def segment_lengths : List ℕ := [1, 3, 5, 7, 9]

-- Define the triangle inequality condition
def forms_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Calculate the number of favorable outcomes, i.e., sets that can form a triangle
def favorable_sets : List (ℕ × ℕ × ℕ) :=
  [(3, 5, 7), (3, 7, 9), (5, 7, 9)]

-- Define the total number of ways to select three segments out of five
def total_combinations : ℕ :=
  10

-- Define the number of favorable sets
def number_of_favorable_sets : ℕ :=
  favorable_sets.length

-- Calculate the probability of selecting three segments that form a triangle
def probability_of_triangle : ℚ :=
  number_of_favorable_sets / total_combinations

-- The theorem to prove
theorem probability_correct : probability_of_triangle = 3 / 10 :=
  by {
    -- Placeholder for the proof
    sorry
  }

end probability_correct_l273_273946


namespace area_of_inscribed_triangle_l273_273256

theorem area_of_inscribed_triangle 
  {r : ℝ} (h_radius: r = 5)
  {sides_ratio : ℝ} (h_sides: sides_ratio = 5/12/13) :
  let x := 10 / 13 in 
  let area := 1/2 * (5 * x) * (12 * x) in 
  (area = 3000 / 169) ∧ ((3000 / 169: ℝ).round = 17.75) :=
sorry

end area_of_inscribed_triangle_l273_273256


namespace largest_digit_divisible_by_6_l273_273980

theorem largest_digit_divisible_by_6 :
  ∃ (N : ℕ), N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0 ∧ (∀ m ∈ {N | N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0}, m ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273980


namespace tangent_line_at_one_inequality_f_l273_273768

noncomputable def f (x : ℝ) : ℝ := x * Real.log (x + 1)

theorem tangent_line_at_one (x : ℝ) : 
  f(1) = Real.log 2 ∧ 
  deriv f 1 = Real.log 2 + 1 / 2 ∧ 
  ∃ (m : ℝ) (b : ℝ), m = Real.log 2 + 1 / 2 ∧ b = - 1 / 2 ∧ (y = (m * x + b)) := 
by 
  sorry

theorem inequality_f (x : ℝ) : 
  x > -1 → f(x) + (1 / 2) * x^3 ≥ x^2 := 
by 
  sorry

end tangent_line_at_one_inequality_f_l273_273768


namespace f_m_plus_1_positive_l273_273776

def f (a x : ℝ) := x^2 + x + a

theorem f_m_plus_1_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := 
  sorry

end f_m_plus_1_positive_l273_273776


namespace min_value_of_quadratic_l273_273082

theorem min_value_of_quadratic (y1 y2 y3 : ℝ) (h1 : 0 < y1) (h2 : 0 < y2) (h3 : 0 < y3) (h_eq : 2 * y1 + 3 * y2 + 4 * y3 = 75) :
  y1^2 + 2 * y2^2 + 3 * y3^2 ≥ 5625 / 29 :=
sorry

end min_value_of_quadratic_l273_273082


namespace length_AF_in_triangle_l273_273453

theorem length_AF_in_triangle :
  ∀ (A B C D E M F : Type)
    [tiA : is_triangle ABC]
    (hAB : length AB = 12)
    (hAngleA : ∠A = 60)
    (hAngleC : ∠C = 60)
    (hD : is_angle_bisector AD ∠BAC)
    (hE_midpoint : is_midpoint E B C)
    (hM_midpoint : is_midpoint M D E)
    (hF_perp : is_perpendicular FM BC),
  length AF = 3 * real.sqrt 3 :=
by
  sorry

end length_AF_in_triangle_l273_273453


namespace largest_proportion_ICF_l273_273515

noncomputable theory

variables {BodyFluid : Type} 

structure TotalBodyFluid :=
  (ICF : BodyFluid)
  (ECF : BodyFluid)

def proportion (x : BodyFluid) : ℚ

-- conditions
axiom fluid_composition : ∀ (T : TotalBodyFluid), proportion T.ICF = 2/3 ∧ proportion T.ECF = 1/3
axiom ecf_components : ∀ (T : TotalBodyFluid), proportion T.ECF = proportion blood_plasma + proportion tissue_fluid + proportion lymph

theorem largest_proportion_ICF : ∀ (T : TotalBodyFluid), proportion T.ICF > proportion T.ECF :=
  by
  intro T
  have h := fluid_composition T
  rw [eq_comm] at h
  sorry

end largest_proportion_ICF_l273_273515


namespace paths_pass_through_F_on_grid_l273_273003

theorem paths_pass_through_F_on_grid :
  ∃ n : ℕ, n = 100 ∧ (∃ E F G : (ℕ × ℕ), 
      E = (0, 5) ∧ F = (3, 3) ∧ 
      G = (5, 0) ∧ 
      E ≠ F ∧ F ≠ G ∧
      (true)) -- Miscellaneous condition to indicate path passing through E, F, and G.
  := 
by
  use 100
  -- Existence of points and their grid location
  use (0, 5)
  use (3, 3)
  use (5, 0)
  repeat { split }
  { exact rfl }
  { exact rfl }
  { exact rfl }
  { sorry }
  { sorry }

end paths_pass_through_F_on_grid_l273_273003


namespace nonnegative_integer_solutions_inequalities_l273_273559

theorem nonnegative_integer_solutions_inequalities :
  ∃ (S : Finset ℤ), S.card = 5 ∧ (∀ x ∈ S, 0 ≤ x ∧ 5 * x + 2 > 3 * (x - 1) ∧ x - 2 ≤ 14 - 3 * x) :=
by
  let S := {0, 1, 2, 3, 4}.to_finset
  use S
  -- Verification of each element is in the set S
  split
  -- Show the cardinality of S is 5
  {
    exact Finset.card_finset_of_five
  }
  -- Show each element in S meets the required conditions
  {
    intros x hx
    simp at hx
    -- Assertion of non-negative condition and inequalities
    have h_pos: 0 ≤ x := by linarith
    have h_ineq1: 5 * x + 2 > 3 * (x - 1) := by linarith
    have h_ineq2: x - 2 ≤ 14 - 3 * x := by linarith
    exact ⟨h_pos, h_ineq1, h_ineq2⟩
  }

end nonnegative_integer_solutions_inequalities_l273_273559


namespace parallel_AK_CM_l273_273526

variables {A B C D M K : Type}
variable [AddGroup A] [Module ℝ A]

-- Given conditions
variables (AB CD AD : A)
variables (h_trapezoid : ∃ k : ℝ, CD = k • AB)
variables (h_AM_MD : AM = 1 / 2 • AD)
variables (h_BK_KC : BK = 1 / 2 • (BC))

-- Prove that lines AK and CM are not parallel
theorem parallel_AK_CM (h_trapezoid : ∃ k : ℝ, CD = k • AB)
  (h_AM_MD : AM = 1 / 2 • AD)
  (h_BK_KC : BK = 1 / 2 • (BC)) :
  ¬ (∃ λ : ℝ, (1 / 2 • (AB + AD + k • AB) = λ • (1 / 2 • AD + k • AB))) :=
sorry

end parallel_AK_CM_l273_273526


namespace star_value_example_l273_273716

def my_star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

theorem star_value_example : my_star 3 5 = 68 := 
by
  sorry

end star_value_example_l273_273716


namespace median_is_106_l273_273820

noncomputable def median_of_list_with_n_n_times_1_to_150 : ℕ :=
  let N := (150 * (150 + 1)) / 2 in
  let median_position := (N + 1) / 2 in
  let n := Nat.floor ((-1 + Real.sqrt (1 + 2 * median_position * 2)) / 2) in
  n

theorem median_is_106 :
  median_of_list_with_n_n_times_1_to_150 = 106 :=
by
  sorry

end median_is_106_l273_273820


namespace smallest_sum_symmetrical_dice_l273_273150

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end smallest_sum_symmetrical_dice_l273_273150


namespace partition_vertices_l273_273039

open Finset

-- Define a tree graph structure
structure Tree (α : Type*) :=
  (vertices : Finset α)
  (edges : Finset (Sym2 α))
  (connected : ∀ v ∈ vertices, ∃ (path : List (Sym2 α)), ∀ u, u ∈ path → (u = ⟦(v, _)⟧ ∨ ⟦(v, u)⟧ ∈ edges) ∧ ∀ u' m, m ∈ path → (u' ∉ m))

-- Define the main theorem to prove the partition
theorem partition_vertices (α : Type*) [DecidableEq α] (T : Tree α) (n d : ℕ) (hn : n ≥ 2) (h_deg : ∀ v ∈ T.vertices, (T.edges.filter (λ e, e.1 = v ∨ e.2 = v)).card ≤ d) :
  ∃ (partitions : Finset (Finset α)), partitions.card ≤ (n / 2 + d) ∧ 
  (∀ e ∈ T.edges, ∀ s t ∈ partitions, e.1 ∈ s → e.2 ∈ t → s ≠ t) ∧
  (∀ s t ∈ partitions, s ≠ t → (T.edges.filter (λ e, (e.1 ∈ s ∧ e.2 ∈ t) ∨ (e.1 ∈ t ∧ e.2 ∈ s))).card ≤ 1) :=
sorry

end partition_vertices_l273_273039


namespace quarters_given_by_mom_l273_273511

theorem quarters_given_by_mom :
  let dimes := 4
  let quarters := 4
  let nickels := 7
  let value_dimes := 0.10 * dimes
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let initial_total := value_dimes + value_quarters + value_nickels
  let final_total := 3.00
  let additional_amount := final_total - initial_total
  additional_amount / 0.25 = 5 :=
by
  sorry

end quarters_given_by_mom_l273_273511


namespace pencil_count_l273_273140

/-- 
If there are initially 115 pencils in the drawer, and Sara adds 100 more pencils, 
then the total number of pencils in the drawer is 215.
-/
theorem pencil_count (initial_pencils added_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : added_pencils = 100) : 
  initial_pencils + added_pencils = 215 := by
  sorry

end pencil_count_l273_273140


namespace area_ratio_of_similar_triangles_l273_273449

theorem area_ratio_of_similar_triangles 
  (ΔABC ΔDEF: Type) 
  [triangle ΔABC] [triangle ΔDEF]
  (h_sim : similar ΔABC ΔDEF)
  (h_perimeter_ratio : perimeter ΔABC / perimeter ΔDEF = 3 / 4) : 
  area ΔABC / area ΔDEF = 9 / 16 := 
sorry

end area_ratio_of_similar_triangles_l273_273449


namespace median_moons_solar_system_l273_273021

theorem median_moons_solar_system :
  let moons := [0, 1, 1, 3, 3, 6, 8, 14, 18, 21] in
  (moons.nth 4).iget + (moons.nth 5).iget / 2 = 4.5 :=
by
  let moons := [0, 1, 1, 3, 3, 6, 8, 14, 18, 21]
  sorry

end median_moons_solar_system_l273_273021


namespace boys_in_class_is_120_l273_273563

-- Definitions from conditions
def num_boys_in_class (number_of_girls number_of_boys : Nat) : Prop :=
  ∃ x : Nat, number_of_girls = 5 * x ∧ number_of_boys = 6 * x ∧
             (5 * x - 20) * 3 = 2 * (6 * x)

-- The theorem proving that given the conditions, the number of boys in the class is 120.
theorem boys_in_class_is_120 (number_of_girls number_of_boys : Nat) (h : num_boys_in_class number_of_girls number_of_boys) :
  number_of_boys = 120 :=
by
  sorry

end boys_in_class_is_120_l273_273563


namespace dot_product_eq_eleven_l273_273389

variable {V : Type _} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)
variable (cos_theta : ℝ)
variable (norm_a norm_b : ℝ)

axiom cos_theta_def : cos_theta = 1 / 3
axiom norm_a_def : ‖a‖ = 1
axiom norm_b_def : ‖b‖ = 3

theorem dot_product_eq_eleven
  (cos_theta_def : cos_theta = 1 / 3)
  (norm_a_def : ‖a‖ = 1)
  (norm_b_def : ‖b‖ = 3) :
  (2 • a + b) ⬝ b = 11 := 
sorry

end dot_product_eq_eleven_l273_273389


namespace z_in_second_quadrant_l273_273925

def i : ℂ := complex.I

def z : ℂ := (2 + i)^2 / (1 - i)

theorem z_in_second_quadrant : z.re < 0 ∧ z.im > 0 := by
  sorry

end z_in_second_quadrant_l273_273925


namespace problem1_problem2_l273_273619

theorem problem1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

theorem problem2 (a b : ℝ) (h : (|a - 2| + |b + 2| = 0) ∨ (|a - 2| * |b + 2| < 0)) : 3a + 2b = 2 := by
  sorry

end problem1_problem2_l273_273619


namespace distance_from_point_to_line_in_polar_coordinates_l273_273814

theorem distance_from_point_to_line_in_polar_coordinates :
  ∀ (ρ θ : ℝ), (ρ = 3) → (θ = π / 6) → ∀ R Θ, (R * sin (Θ + π / 3) = 1) → (let x := ρ * cos θ 
    (let y := ρ * sin θ 
    (let dist := (| √3 / 2 * x + 1 / 2 * y - 1 |) / √((√3 / 2) ^ 2 + (1 / 2) ^ 2))
    dist = 2)) := 
sorry

end distance_from_point_to_line_in_polar_coordinates_l273_273814


namespace rachel_fathers_age_when_rachel_is_25_l273_273907

theorem rachel_fathers_age_when_rachel_is_25 (R G M F Y : ℕ) 
  (h1 : R = 12)
  (h2 : G = 7 * R)
  (h3 : M = G / 2)
  (h4 : F = M + 5)
  (h5 : Y = 25 - R) : 
  F + Y = 60 :=
by sorry

end rachel_fathers_age_when_rachel_is_25_l273_273907


namespace larger_acute_angle_right_triangle_l273_273457

theorem larger_acute_angle_right_triangle (x : ℝ) (h1 : x > 0) (h2 : x + 5 * x = 90) : 5 * x = 75 := by
  sorry

end larger_acute_angle_right_triangle_l273_273457


namespace determine_distance_AB_l273_273177

variable (A B R1 R2 R3 R4 R5 R6 : Type)

-- Assume the points on the lines and intersections
axiom line_AR1 : Prop
axiom line_BR2 : Prop
axiom line_AB : Prop
axiom line_R1R3 : Prop
axiom line_BR1 : Prop
axiom line_R2R4 : Prop
axiom line_R3R5 : Prop

-- Define the intersections based on the given conditions
axiom intersection_R4 : R4 = intersection line_AB line_R1R3
axiom intersection_R5 : R5 = intersection line_BR1 line_R2R4
axiom intersection_R6 : R6 = intersection line_AB line_R3R5

-- Define the distances between the points
def distance (P Q : Type) : ℝ := sorry

-- The problem statement to be proved
theorem determine_distance_AB (h1 : distance A R4 ≠ 0)
    (h2 : distance A R6 ≠ 0)
    (h3 : distance R4 R6 - distance A R4 ≠ 0) :
    distance A B = abs (distance A R4 * distance A R6 / (distance R4 R6 - distance A R4)) :=
sorry

end determine_distance_AB_l273_273177


namespace first_player_wins_l273_273147

-- Define the game state and the rules
structure GameState :=
  (pile1 pile2 : ℕ)

-- A move can either remove stones from pile1 or pile2
inductive Move
| from_pile1 (n : ℕ) : Move
| from_pile2 (n : ℕ) : Move

-- Function to apply a move to a game state
def apply_move : GameState → Move → Option GameState
| ⟨p1, p2⟩, Move.from_pile1 n => if n > 0 ∧ n ≤ p1 then some ⟨p1 - n, p2⟩ else none
| ⟨p1, p2⟩, Move.from_pile2 n => if n > 0 ∧ n ≤ p2 then some ⟨p1, p2 - n⟩ else none

-- Define the initial game state
def initial_state : GameState := ⟨7, 7⟩

-- Define what it means to be a losing state
def losing_state : GameState → Prop
| ⟨0, 0⟩ := true
| _ := false

-- Theorem statement: The first player has a winning strategy from the initial state
theorem first_player_wins : ∃ strategy : (GameState → Option Move), 
  strategy initial_state ≠ none ∧
  ∀ s m, apply_move s m = some s' → strategy s' ≠ none :=
by
  sorry

end first_player_wins_l273_273147


namespace farmer_cows_after_selling_l273_273225

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end farmer_cows_after_selling_l273_273225


namespace parabola_vertex_x_coord_l273_273554

variable {a b c : ℝ}

theorem parabola_vertex_x_coord :
  (∀ (x y : ℝ), (y = a * (-2)^2 + b * (-2) + c) ∧ y = 9 ∨
                 (y = a * (4)^2 + b * (4) + c) ∧ y = 9 ∨
                 (y = a * (5)^2 + b * (5) + c) ∧ y = 13) →
  let vertex_x := 1 in 
  vertex_x = 1 := sorry

end parabola_vertex_x_coord_l273_273554


namespace system_equations_solution_l273_273915

theorem system_equations_solution (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 3) ∧ 
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) ∧ 
  (1 / (x * y * z) = 1) → 
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end system_equations_solution_l273_273915


namespace area_of_triangle_is_75_cm2_l273_273201

-- Define the conditions for the problem
def perimeter (T : Type) [Triangle T] : Real := 60
def inradius (T : Type) [Triangle T] : Real := 2.5
def semiperimeter (T : Type) [Triangle T] : Real := perimeter T / 2

-- Define the area of the triangle based on the inradius and semiperimeter
def area (T : Type) [Triangle T] : Real := inradius T * semiperimeter T

-- The final theorem that states the area of the triangle given the conditions
theorem area_of_triangle_is_75_cm2 (T : Type) [Triangle T] :
  perimeter T = 60 ∧ inradius T = 2.5 → area T = 75 := by
  sorry

end area_of_triangle_is_75_cm2_l273_273201


namespace num_ways_to_choose_cards_equiv_l273_273430

noncomputable def num_ways_to_choose_cards 
  (total_cards : ℕ := 52) 
  (choose_five : ℕ := 5) 
  (cards_in_suit : ℕ := 13) 
  (suits : ℕ := 4) : ℕ := 
  let choose_suits1 := Nat.choose suits 1 in
  let choose_cards_from_suits1 := Nat.choose cards_in_suit 2 in
  let choose_remaining_suits := Nat.choose (suits - 1) 3 in
  let choose_cards_from_remaining_suits := cards_in_suit ^ 3 in
  choose_suits1 * choose_cards_from_suits1 * choose_remaining_suits * choose_cards_from_remaining_suits

theorem num_ways_to_choose_cards_equiv 
  : num_ways_to_choose_cards = 684684 := 
by 
  simp [num_ways_to_choose_cards]; 
  sorry

end num_ways_to_choose_cards_equiv_l273_273430


namespace min_value_n_l273_273343

noncomputable def minN : ℕ :=
  5

theorem min_value_n :
  ∀ (S : Finset ℕ), (∀ n ∈ S, 1 ≤ n ∧ n ≤ 9) ∧ S.card = minN → 
    (∃ T ⊆ S, T ≠ ∅ ∧ 10 ∣ (T.sum id)) :=
by
  sorry

end min_value_n_l273_273343


namespace length_of_boat_is_3_l273_273217

-- Definitions
def breadth : ℝ := 2
def sink_depth : ℝ := 0.01
def man_mass : ℝ := 60
def gravity : ℝ := 9.81
def water_density : ℝ := 1000

-- Volume of water displaced
def volume_displaced (length : ℝ) : ℝ := length * breadth * sink_depth

-- Weight of the man
def weight_man : ℝ := man_mass * gravity

-- Buoyant force is equal to the weight of the man
def buoyant_force_eq_weight (length : ℝ) : Prop :=
  water_density * volume_displaced(length) * gravity = weight_man

-- The main theorem to prove
theorem length_of_boat_is_3 : ∃ length, buoyant_force_eq_weight(length) ∧ length = 3 :=
by
  exists 3
  split
  · unfold buoyant_force_eq_weight volume_displaced weight_man
    norm_num
  sorry

end length_of_boat_is_3_l273_273217


namespace range_of_a_l273_273445

noncomputable def f (a x : ℝ) : ℝ := x^3 + x^2 - a * x - 4
noncomputable def f_derivative (a x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

def has_exactly_one_extremum_in_interval (a : ℝ) : Prop :=
  (f_derivative a (-1)) * (f_derivative a 1) < 0

theorem range_of_a (a : ℝ) :
  has_exactly_one_extremum_in_interval a ↔ (1 < a ∧ a < 5) :=
sorry

end range_of_a_l273_273445


namespace intersection_A_B_complement_l273_273087

def universal_set : Set ℝ := {x : ℝ | True}
def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def B_complement : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_A_B_complement :
  (A ∩ B_complement) = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_complement_l273_273087


namespace math_problem_proof_l273_273365

-- Definitions and statements

open Real

noncomputable def ellipse_equation :=
  ∃ (a b : ℝ), (a > b ∧ b > 0 ∧ (√3 / 2)^2 * a^2 = a^2 - b^2 ∧ (sqrt 3) = a * b) ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → x^2 / (2:ℝ)^2 + y^2 / (1:ℝ)^2 = 1)

noncomputable def circle_equation :=
  ∃ (x y a b : ℝ), (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = (a / b)^2) → ((x-2:ℝ)^2 + (y-1:ℝ)^2 = 4)

noncomputable def line_equation :=
  ∃ (k : ℝ → ℝ), (2:ℝ) - 1 < k ∧ (k * 4 + 2 * k = 0) ∧ (3 * k^2 - 4 * k = 0) →
    ((k = 0) ∨ (k = 4 / 3))

theorem math_problem_proof :
  ellipse_equation ∧ circle_equation ∧ line_equation :=
by {
  split,
  { unfold ellipse_equation, existsi 2, existsi 1,
    split, linarith, split, linarith, split,
    { linarith },
    unfold circle_equation, existsi 2, existsi 1, existsi 2, existsi 1, intros,
    sorry 
  },
  sorry
}

end math_problem_proof_l273_273365


namespace quadratic_identity_l273_273898

theorem quadratic_identity
  (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c) / ((a - b) * (a - c))) +
  (b^2 * (x - a) * (x - c) / ((b - a) * (b - c))) +
  (c^2 * (x - a) * (x - b) / ((c - a) * (c - b))) =
  x^2 :=
sorry

end quadratic_identity_l273_273898


namespace binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l273_273911

-- Definitions and lemma statement
theorem binomial_coefficient_divisible_by_prime
  {p k : ℕ} (hp : Prime p) (hk : 0 < k) (hkp : k < p) :
  p ∣ Nat.choose p k := 
sorry

-- Theorem for k = 0 and k = p cases
theorem binomial_coefficient_extreme_cases {p : ℕ} (hp : Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l273_273911


namespace unique_xy_exists_l273_273912

theorem unique_xy_exists (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y) ^ 2 + 3 * x + y) / 2 := 
sorry

end unique_xy_exists_l273_273912


namespace jerry_weekly_earnings_l273_273490

theorem jerry_weekly_earnings:
  (tasks_per_day: ℕ) 
  (daily_earnings: ℕ)
  (weekly_earnings: ℕ) :
  (tasks_per_day = 10 / 2) ∧
  (daily_earnings = 40 * tasks_per_day) ∧
  (weekly_earnings = daily_earnings * 7) →
  weekly_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l273_273490


namespace gym_monthly_income_l273_273645

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end gym_monthly_income_l273_273645


namespace cube_faces_edges_vertices_sum_l273_273590

theorem cube_faces_edges_vertices_sum :
  ∀ (F E V : ℕ), F = 6 → E = 12 → V = 8 → F + E + V = 26 :=
by
  intros F E V F_eq E_eq V_eq
  rw [F_eq, E_eq, V_eq]
  rfl

end cube_faces_edges_vertices_sum_l273_273590


namespace runner_distance_heard_camera_click_l273_273249

theorem runner_distance_heard_camera_click :
  ∀ (t_click : ℝ) (runner_speed_yd_sec : ℝ) (sound_speed_ft_sec : ℝ) (reduction_percent : ℝ)
  (time_photo : ℝ),
  t_click = 45 →
  runner_speed_yd_sec = 10 →
  sound_speed_ft_sec = 1100 →
  reduction_percent = 0.10 →
  time_photo = 45 →
  let runner_speed_ft_sec := runner_speed_yd_sec * 3 in
  let effective_sound_speed_ft_sec := sound_speed_ft_sec * (1 - reduction_percent) in
  let click_time_sec := t_click + (runner_speed_ft_sec * t_click) / effective_sound_speed_ft_sec in
  runner_speed_ft_sec * click_time_sec / 3 = 464 :=
begin
  intros,
  sorry
end

end runner_distance_heard_camera_click_l273_273249


namespace solve_for_y_l273_273294

theorem solve_for_y : ∃ (y : ℚ), y + 2 - 2 / 3 = 4 * y - (y + 2) ∧ y = 5 / 3 :=
by
  sorry

end solve_for_y_l273_273294


namespace hyperbola_eccentricity_l273_273772

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 6 * x + 5 = 0)
  (h_tangent : ∀ b a : ℝ, 9 * b^2 = 4 * (b^2 + a^2)) :
  let c := (5 * a^2) / 4 in 
  let e := c / a in 
  e = (3 * sqrt 5) / 5 :=
sorry

end hyperbola_eccentricity_l273_273772


namespace value_subtracted_l273_273928

theorem value_subtracted (n v : ℝ) (h1 : 2 * n - v = -12) (h2 : n = -10.0) : v = -8 :=
by
  sorry

end value_subtracted_l273_273928


namespace centroid_distance_squared_l273_273063

noncomputable def trianglePoints (D E F : ℝ × ℝ) : Prop :=
(G = (1 / 3) • (D + E + F))

theorem centroid_distance_squared
  (D E F G : ℝ × ℝ)
  (hG : G = (1 / 3) • (D + E + F))
  (h : (gneptune_distance G D)^2 + (gneptune_distance G E)^2 + (gneptune_distance G F)^2 = 90) :
  (gneptune_distance D E)^2 + (gneptune_distance D F)^2 + (gneptune_distance E F)^2 = 270 := sorry

end centroid_distance_squared_l273_273063


namespace common_ratio_is_two_l273_273728

-- Geometric sequence definition
noncomputable def common_ratio (n : ℕ) (a : ℕ → ℝ) : ℝ :=
a 2 / a 1

-- The sequence has 10 terms
def ten_terms (a : ℕ → ℝ) : Prop :=
∀ n, 1 ≤ n ∧ n ≤ 10

-- The product of the odd terms is 2
def product_of_odd_terms (a : ℕ → ℝ) : Prop :=
(a 1) * (a 3) * (a 5) * (a 7) * (a 9) = 2

-- The product of the even terms is 64
def product_of_even_terms (a : ℕ → ℝ) : Prop :=
(a 2) * (a 4) * (a 6) * (a 8) * (a 10) = 64

-- The problem statement to prove that the common ratio q is 2
theorem common_ratio_is_two (a : ℕ → ℝ) (q : ℝ) (h1 : ten_terms a) 
(h2 : product_of_odd_terms a) (h3 : product_of_even_terms a) : q = 2 :=
by {
  sorry
}

end common_ratio_is_two_l273_273728


namespace tina_sales_ratio_l273_273493

theorem tina_sales_ratio (katya_sales ricky_sales t_sold_more : ℕ) 
  (h_katya : katya_sales = 8) 
  (h_ricky : ricky_sales = 9) 
  (h_tina_sold : t_sold_more = katya_sales + 26) 
  (h_tina_multiple : ∃ m : ℕ, t_sold_more = m * (katya_sales + ricky_sales)) :
  t_sold_more / (katya_sales + ricky_sales) = 2 := 
by 
  sorry

end tina_sales_ratio_l273_273493


namespace triangle_area_l273_273260

-- Given conditions
def ratio_5_12_13 (a b c : ℝ) : Prop := 
  (b / a = 12 / 5) ∧ (c / a = 13 / 5)

def right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

def circumscribed_circle (a b c r : ℝ) : Prop :=
  2 * r = c

-- The main theorem we need to prove
theorem triangle_area (a b c r : ℝ) (h_ratio : ratio_5_12_13 a b c) (h_triangle : right_triangle a b c) (h_circle : circumscribed_circle a b c r) (h_r : r = 5) :
  0.5 * a * b ≈ 17.75 :=
by
  sorry

end triangle_area_l273_273260


namespace least_y_solution_l273_273995

theorem least_y_solution :
  (∃ y : ℝ, 3 * y^2 + 5 * y + 2 = 4 ∧ ∀ z : ℝ, 3 * z^2 + 5 * z + 2 = 4 → y ≤ z) →
  ∃ y : ℝ, y = -2 :=
by
  sorry

end least_y_solution_l273_273995


namespace vertex_with_edges_in_cycles_l273_273061

variables {V : Type*} [DecidableEq V]
variables (G : SimpleGraph V)

theorem vertex_with_edges_in_cycles (h1 : ∀ v : V, G.degree v ≥ 2) :
    ∃ v : V, ∀ e : G.edge_set, e ∈ G.incident_edges v →  G.is_cycle (e :: G.path_to_some_cycle e) :=
sorry

end vertex_with_edges_in_cycles_l273_273061


namespace rectangular_prism_volume_l273_273247

theorem rectangular_prism_volume 
(l w h : ℝ) 
(h1 : l * w = 18) 
(h2 : w * h = 32) 
(h3 : l * h = 48) : 
l * w * h = 288 :=
sorry

end rectangular_prism_volume_l273_273247


namespace distinct_real_roots_of_quadratic_find_k_l273_273422

theorem distinct_real_roots_of_quadratic (k : ℤ) (hk : k ≠ 0) :
  let Δ := (4 * k + 1) ^ 2 - 4 * k * (3 * k + 3) in
  Δ > 0 :=
by
  sorry

theorem find_k (k : ℤ) (hk : k ≠ 0) 
  (h_triang : ∀ x1 x2, x1 + x2 = (4 * k + 1) / k ∧ x1 * x2 = (3 * k + 3) / k → 
    x1^2 + x2^2 = (9 * 5) / 4) :
  k = 2 :=
by
  sorry

end distinct_real_roots_of_quadratic_find_k_l273_273422


namespace smaller_angle_at_3_45_is_157_5_degrees_l273_273182

def degrees_per_minute_minute_hand : ℝ := 360 / 60
def degrees_per_minute_hour_hand : ℝ := 360 / 12 / 60

def minute_hand_position (minutes : ℕ) : ℝ := minutes * degrees_per_minute_minute_hand
def hour_hand_position (hours : ℕ) (minutes : ℕ) : ℝ := (hours * 60 + minutes) * degrees_per_minute_hour_hand

def angle_between_hands (t₁ t₂ : ℝ) : ℝ := abs (abs t₁ - abs t₂)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle <= 180 then angle else 360 - angle

theorem smaller_angle_at_3_45_is_157_5_degrees :
  smaller_angle (angle_between_hands (minute_hand_position 45) (hour_hand_position 3 45)) = 157.5 :=
by
  sorry

end smaller_angle_at_3_45_is_157_5_degrees_l273_273182


namespace clock_hands_inline_l273_273280

open Real

-- Define the conditions that need to be satisfied
def coincide_condition (x : ℝ) : Prop := 6 * x = 120 + (1/2) * x
def opposite_condition (x : ℝ) : Prop := 6 * x = 120 + (1/2) * x + 180

-- Define the valid solutions within the interval between 4 and 5 o'clock
def valid_solution_1 : ℝ := 21 + 9/11
def valid_solution_2 : ℝ := 54 + 6/11

-- The theorem states that valid_solution_1 and valid_solution_2 are the solutions to the problem
theorem clock_hands_inline : 
  coincide_condition valid_solution_1 ∨ opposite_condition valid_solution_1 ∧ 
  (valid_solution_1 > 0) ∧ (valid_solution_1 < 60) ∧ -- Condition for the minute hand within hour
  coincide_condition valid_solution_2 ∨ opposite_condition valid_solution_2 ∧ 
  (valid_solution_2 > 0) ∧ (valid_solution_2 < 60) := sorry

end clock_hands_inline_l273_273280


namespace johns_subtraction_l273_273580

theorem johns_subtraction 
  (a : ℕ) 
  (h₁ : (51 : ℕ)^2 = (50 : ℕ)^2 + 101) 
  (h₂ : (49 : ℕ)^2 = (50 : ℕ)^2 - b) 
  : b = 99 := 
by 
  sorry

end johns_subtraction_l273_273580


namespace largest_digit_divisible_by_6_l273_273985

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l273_273985


namespace f_2_is_6_l273_273072

noncomputable def f : ℤ → ℤ := sorry

axiom f_property (m n : ℤ) : f (m + n) + f (mn - 1) = f m * f n + 5
axiom f_0 : f 0 = 0
axiom f_1 (a : ℤ) : f 1 = a

theorem f_2_is_6 (a : ℤ) (h : a^2 + 5 = 6) : f 2 = 6 :=
by
  sorry

end f_2_is_6_l273_273072


namespace area_eq_25pi_area_approx_eq_78_5_l273_273184

-- Given: d (diameter) = 10 cm
def diameter : ℝ := 10
def radius : ℝ := diameter / 2

-- For π, we use the predefined constant from Mathlib
def pi : ℝ := real.pi

-- Calculate the area in terms of π
def area_in_terms_of_pi : ℝ := pi * radius ^ 2

-- Calculate the approximate area using π ≈ 3.14
def approximate_pi : ℝ := 3.14
def approximate_area : ℝ := approximate_pi * radius ^ 2

-- Theorems stating what needs to be proved
theorem area_eq_25pi : area_in_terms_of_pi = 25 * pi :=
by sorry

theorem area_approx_eq_78_5 : approximate_area = 78.5 :=
by sorry

end area_eq_25pi_area_approx_eq_78_5_l273_273184


namespace find_radius_of_circumcircle_of_ABC_l273_273170

-- Definitions based on given conditions
variable (A B C : Point)
variable (r1 r2 : ℝ)
variable (O1 O2 : Point)
variable (r3 := 8) -- The radius of the third sphere

-- Given Conditions
-- 1. r1 + r2 = 7
def condition1 : r1 + r2 = 7 := sorry

-- 2. The distance between the centers of the two spheres
def condition2 : dist O1 O2 = 17 := sorry

-- 3. The distance from the point A to the centers O1 and O2
def condition3 : dist A O1 = r1 + r3 ∧ dist A O2 = r2 + r3 := sorry

-- Target statement
theorem find_radius_of_circumcircle_of_ABC :
  r1 + r2 = 7 → dist O1 O2 = 17 → (dist A O1 = r1 + 8) ∧ (dist A O2 = r2 + 8) → 
  circumradius A B C = 2 * sqrt 15 :=
sorry

end find_radius_of_circumcircle_of_ABC_l273_273170


namespace term_2008_l273_273504

noncomputable def u_n : ℕ → ℕ 
| 0 => 2
| n => let k := ∑ i in range (n + 1), i + 1
       if k ≤ n then k + 1 else k * k + 1

theorem term_2008 : (u_n 2008 = 4008) :=
sorry

end term_2008_l273_273504


namespace range_of_f_lt_0_l273_273801

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

variable (f : ℝ → ℝ)
variable (h_odd : is_odd f)
variable (h_decreasing : decreasing_on f (Set.Iic 0))
variable (h_at_2 : f 2 = 0)

theorem range_of_f_lt_0 : ∀ x, x ∈ (Set.Ioo (-2) 0 ∪ Set.Ioi 2) → f x < 0 := by
  sorry

end range_of_f_lt_0_l273_273801


namespace sqrt_expr_equality_l273_273285

theorem sqrt_expr_equality : 
  let a := 5
  let b := 2
  sqrt (a^4 + a^4 + a^4 + b^4) = sqrt 1891 :=
by
  let a := 5
  let b := 2
  sorry

end sqrt_expr_equality_l273_273285


namespace collinear_points_l273_273934

-- Definitions of geometric entities and conditions
variables (A B C E F D T : Type) [GeometricEntity A B C E F D T]

-- The excircle of a triangle touches the sides of the triangle at certain points
def excircle_touches_side (triangle_excircle : Type) (side : Type) (touch_point : Type) :=
  True -- Placeholder for the formal definition

-- Extensions of sides touching the excircle
def extensions_touch_excircle (line1 line2 line_excircle_point : Type) :=
  True -- Placeholder for the formal definition

-- Intersection of lines
def intersect_lines (line1 line2 intersection_point : Type) :=
  True -- Placeholder for the formal definition

theorem collinear_points
  (triangle_excircle : Type) 
  (line1 line2 : Type)
  (h1 : excircle_touches_side triangle_excircle B D → excircle_touches_side triangle_excircle C D)
  (h2 : extensions_touch_excircle AB AC E → extensions_touch_excircle AB AC F)
  (h3 : intersect_lines BF CE T) :
  collinear A D T :=
begin
  sorry
end

end collinear_points_l273_273934


namespace area_of_inscribed_triangle_l273_273258

theorem area_of_inscribed_triangle 
  {r : ℝ} (h_radius: r = 5)
  {sides_ratio : ℝ} (h_sides: sides_ratio = 5/12/13) :
  let x := 10 / 13 in 
  let area := 1/2 * (5 * x) * (12 * x) in 
  (area = 3000 / 169) ∧ ((3000 / 169: ℝ).round = 17.75) :=
sorry

end area_of_inscribed_triangle_l273_273258


namespace value_of_expr_l273_273998

theorem value_of_expr (x : ℤ) (h : x = 3) : (2 * x + 6) ^ 2 = 144 := by
  sorry

end value_of_expr_l273_273998


namespace complementary_event_equivalence_l273_273690

-- Define the event E: hitting the target at least once in two shots.
-- Event E complementary: missing the target both times.

def eventE := "hitting the target at least once"
def complementaryEvent := "missing the target both times"

theorem complementary_event_equivalence :
  (complementaryEvent = "missing the target both times") ↔ (eventE = "hitting the target at least once") :=
by
  sorry

end complementary_event_equivalence_l273_273690


namespace min_n_for_positive_product_grid_l273_273176

theorem min_n_for_positive_product_grid : ∀ (n : ℕ), 
  (∀ (A : Matrix (Fin 10) (Fin n) ℤ), 
    (∀ i : Fin 10, ∏ j in Finset.univ, (A i j) > 0) ∧ 
    (∀ j : Fin n, ∏ i in Finset.univ, (A i j) > 0)) → 
  n = 4 :=
by
  sorry

end min_n_for_positive_product_grid_l273_273176


namespace measure_angle_EBC_l273_273726

-- Define the cyclic quadrilateral and conditions
def cyclic_quadrilateral {α : Type} [metric_space α] [normed_group α] [normed_space ℝ α]
  (A B C D E : α) : Prop :=
  ∃ (O : α) (r : ℝ), metric.sphere O r ∧ A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ E ≠ B
  ∧ (∃ p : finset α, p = {A, B, C, D} ∧ p.card = 4 ∧ ∀ X Y : α, (X ∈ p ∧ Y ∈ p ∧ X ≠ Y) → dist O X = r ∧ dist O Y = r)

theorem measure_angle_EBC {α : Type} [metric_space α] [normed_group α] [normed_space ℝ α]
  (A B C D E : α)
  (h_cyclic : cyclic_quadrilateral A B C D E)
  (h_ext : extended A B E)
  (h_AngBAD : ∠BAD = 80)
  (h_AngADC : ∠ADC = 110) :
  ∠EBC = 110 := by
  sorry

end measure_angle_EBC_l273_273726


namespace axis_of_symmetry_is_2_l273_273434

def symmetric_function (f : ℝ → ℝ) := ∀ x, f(x) = f(4 - x)

theorem axis_of_symmetry_is_2 (f : ℝ → ℝ) (h : symmetric_function f) : 
  ∀ x, f(x) = f(2 + (2 - x)) :=
by
  sorry

end axis_of_symmetry_is_2_l273_273434


namespace largest_digit_divisible_by_6_l273_273987

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273987


namespace lucille_house_difference_l273_273092

-- Define the heights of the houses as given in the conditions.
def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

-- Define the total height of the houses.
def total_height : ℕ := height_neighbor1 + height_lucille + height_neighbor2

-- Define the average height of the houses.
def average_height : ℕ := total_height / 3

-- Define the height difference between Lucille's house and the average height.
def height_difference : ℕ := average_height - height_lucille

-- The theorem to prove.
theorem lucille_house_difference :
  height_difference = 3 := by
  sorry

end lucille_house_difference_l273_273092


namespace A_minus_B_l273_273496

theorem A_minus_B : 
  ∀ (n : ℕ), 
    let S := λ k, (1 : ℕ) ^ k + 2 ^ k + 3 ^ k + ... + n ^ k in
    S 1 = (1 / 2) * n^2 + (1 / 2) * n ∧
    S 2 = (1 / 3) * n^3 + (1 / 2) * n^2 + (1 / 6) * n ∧
    S 3 = (1 / 4) * n^4 + (1 / 2) * n^3 + (1 / 4) * n^2 ∧
    S 4 = (1 / 5) * n^5 + (1 / 2) * n^4 + (1 / 3) * n^3 - (1 / 30) * n ∧
    S 5 = (1 / 6) * n^6 + A * n^5 + B * n^4 - (1 / 12) * n^2 → 
    A - B = 1 / 12 := 
by 
  sorry

end A_minus_B_l273_273496


namespace vector_dot_product_property_l273_273381

variables (a b : ℝ^3) (θ : ℝ)

open real_inner_product_space

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

def cos_theta (a b : ℝ^3) : ℝ := (a.dot_product b) / ((magnitude a) * (magnitude b))

theorem vector_dot_product_property
  (h1 : cos_theta a b = 1 / 3)
  (h2 : magnitude a = 1)
  (h3 : magnitude b = 3) :
  (2 • a + b) ∙ b = 11 :=
begin
  sorry
end

end vector_dot_product_property_l273_273381


namespace axis_of_symmetry_l273_273431

-- Definitions corresponding to the given problem
variable (f : ℝ → ℝ)

-- Given the condition
def symmetric_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(4 - x)

-- Prove that the line x = 2 is an axis of symmetry of the graph of y = f(x)
theorem axis_of_symmetry (h : symmetric_function f) : ∀ y, f y = f (2 + (2 - y)) :=
by
  sorry

end axis_of_symmetry_l273_273431


namespace average_improvement_l273_273811

def avg_improvement : ℝ :=
  let pi := Real.pi
  let laps := 11
  let distance := pi * laps
  let this_year_times := [82.5, 84, 86]
  let last_year_times := [106.37, 109.5, 112]
  let avg this_year := (this_year_times.map (λ t, t / distance)).sum / this_year_times.length
  let avg last_year := (last_year_times.map (λ t, t / distance)).sum / last_year_times.length
  avg last_year - avg this_year

theorem average_improvement : avg_improvement = 0.72666 := sorry

end average_improvement_l273_273811


namespace prove_positive_a_l273_273731

variable (a b c n : ℤ)
variable (p : ℤ → ℤ)

-- Conditions given in the problem
def quadratic_polynomial (x : ℤ) : ℤ := a*x^2 + b*x + c

def condition_1 : Prop := a ≠ 0
def condition_2 : Prop := n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))

-- Proof goal
theorem prove_positive_a (h1 : a ≠ 0) (h2 : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
by
  sorry

end prove_positive_a_l273_273731


namespace gym_monthly_income_l273_273646

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end gym_monthly_income_l273_273646


namespace norm_a_angle_ab_norm_a_plus_b_l273_273744

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 2))]
variables (a b : euclidean_space ℝ (fin 2))

-- Conditions
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom dot_product_ab : ⟪a, b⟫ = 3
axiom norm_b : ∥b∥ = 2
axiom orthogonality : ⟪a, a - 3 • b⟫ = 0

-- Proof statements

/-- Prove that the norm of vector a is 3 -/
theorem norm_a : ∥a∥ = 3 :=
sorry

/-- Prove that the angle between vector a and vector b is 60 degrees -/
theorem angle_ab : real.angle_of_real (⟪a, b⟫ / (∥a∥ * ∥b∥)) = real.pi / 3 :=
sorry

/-- Prove that the norm of the sum of vectors a and b is sqrt(19) -/
theorem norm_a_plus_b : ∥a + b∥ = real.sqrt 19 :=
sorry

end norm_a_angle_ab_norm_a_plus_b_l273_273744


namespace Rachel_father_age_when_Rachel_is_25_l273_273904

-- Define the problem conditions:
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Prove the age of Rachel's father when she is 25 years old:
theorem Rachel_father_age_when_Rachel_is_25 : 
  Father_age + (25 - Rachel_age) = 60 := by
    sorry

end Rachel_father_age_when_Rachel_is_25_l273_273904


namespace problem_l273_273718

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem problem (h₁ : f a b c 0 = f a b c 4) (h₂ : f a b c 4 > f a b c 1) : a > 0 ∧ 4 * a + b = 0 :=
by 
  sorry

end problem_l273_273718


namespace sequence_a_n_l273_273450

theorem sequence_a_n {n : ℕ} (S : ℕ → ℚ) (a : ℕ → ℚ)
  (hS : ∀ n, S n = (2/3 : ℚ) * n^2 - (1/3 : ℚ) * n)
  (ha : ∀ n, a n = if n = 1 then S n else S n - S (n - 1)) :
  ∀ n, a n = (4/3 : ℚ) * n - 1 := 
by
  sorry

end sequence_a_n_l273_273450


namespace tangent_line_to_exp_zero_in_interval_range_min_value_of_h_l273_273411

noncomputable def f (x a : ℝ) := x^2 + a * x + 1
noncomputable def g (x : ℝ) := Real.exp x

theorem tangent_line_to_exp (k : ℝ) : 
  (∃ x₀ : ℝ, ∃ y₀ : ℝ, y₀ = Real.exp x₀ ∧ y₀ / x₀ = Real.exp x₀) → k = Real.exp 1 :=
sorry

theorem zero_in_interval_range (a : ℝ) (h : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x a = 0) : 
  -5/2 < a ∧ a < -2 :=
sorry

noncomputable def h (x a : ℝ) := (f x a) * (g x)

theorem min_value_of_h (a : ℝ) : 
  (let H(a) := if a ≤ -2 then (2 + a) * Real.exp 1 
              else if -2 < a ∧ a < 0 then (2 + a) * Real.exp (-a-1) 
              else (2 - a) / Real.exp 1 in
  (∀ x ∈ Icc (-1:ℝ) 1, h x a ≥ H(a))) :=
sorry

end tangent_line_to_exp_zero_in_interval_range_min_value_of_h_l273_273411


namespace problem_statement_l273_273439

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y + z = 6) 
  (h2 : x * y + y * z + z * x = 11) 
  (h3 : x * y * z = 6) : 
  x / (y * z) + y / (z * x) + z / (x * y) = 7 / 3 := 
sorry

end problem_statement_l273_273439


namespace find_f_l273_273752

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : ∀ x : ℝ, f x = x + 1 :=
by
  sorry

end find_f_l273_273752


namespace smallest_sum_symmetrical_dice_l273_273151

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end smallest_sum_symmetrical_dice_l273_273151


namespace cyclic_quadrilateral_iff_angle_eq_circumscribed_l273_273102

section

variables {A B C D : Type*}
variables (quadrilateral : ConvexQuadrilateral A B C D)
variables (angle_CAD_eq_angle_CBD : ∠(A, C, D) = ∠(B, C, D))

theorem cyclic_quadrilateral_iff_angle_eq_circumscribed (quadrilateral : ConvexQuadrilateral A B C D) (angle_CAD_eq_angle_CBD : ∠(A, C, D) = ∠(B, C, D)) : 
  (∃ (circle : Circle), circle.Circumscribes quadrilateral) ↔ angle_CAD_eq_angle_CBD :=
sorry

end

end cyclic_quadrilateral_iff_angle_eq_circumscribed_l273_273102


namespace domain_of_function_l273_273931

def domain_function (x : ℝ) : ℝ := sqrt (3^x - 1)

theorem domain_of_function :
  (∀ y : ℝ, ∃ x : ℝ, y = domain_function x) ↔ (0 ≤ x) :=
by
  sorry

end domain_of_function_l273_273931


namespace highest_average_speed_1_2_hours_l273_273689

def laps_per_hour : ℕ → ℕ
| 0 := 15
| 1 := 22
| 2 := 18
| 3 := 19
| 4 := 20

def lap_length : ℕ := 5  -- miles per lap

noncomputable def distance_covered (hour : ℕ) : ℕ := laps_per_hour hour * lap_length

theorem highest_average_speed_1_2_hours :
  ∀ hour, distance_covered 1 ≥ distance_covered hour := by
  sorry

end highest_average_speed_1_2_hours_l273_273689


namespace a_sequence_formula_c_sequence_sum_l273_273736

-- Define the sequence {a_n}
def a (n : ℕ) : ℚ :=
if n = 0 then 0 else 1 / (2 * n - 1)

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ :=
if n % 2 = 1 then 1 / (19 * a n) else a n * a (n + 2)

-- Prove the sequence properties
theorem a_sequence_formula (n : ℕ) (h : n > 0) :
  a n = 1 / (2 * n - 1) := by
  sorry

theorem c_sequence_sum : 
  ∑ i in Finset.range 20, c (i + 1) = 1300 / 129 := by
  sorry

end a_sequence_formula_c_sequence_sum_l273_273736


namespace gym_monthly_income_l273_273639

theorem gym_monthly_income (bi_monthly_charge : ℕ) (members : ℕ) (monthly_income : ℕ) 
  (h1 : bi_monthly_charge = 18)
  (h2 : members = 300)
  (h3 : monthly_income = 10800) : 
  2 * bi_monthly_charge * members = monthly_income :=
by
  rw [h1, h2, h3]
  norm_num

end gym_monthly_income_l273_273639


namespace probability_points_one_unit_apart_l273_273918

theorem probability_points_one_unit_apart :
  let points := 10
  let rect_length := 3
  let rect_width := 2
  let total_pairs := (points * (points - 1)) / 2
  let favorable_pairs := 10  -- derived from solution steps
  (favorable_pairs / total_pairs : ℚ) = (2 / 9 : ℚ) :=
by
  sorry

end probability_points_one_unit_apart_l273_273918


namespace measure_of_angle_B_l273_273398

theorem measure_of_angle_B (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :
  ∠ B = π / 3 :=
sorry

end measure_of_angle_B_l273_273398


namespace dissection_into_equal_area_triangles_l273_273866

-- Assume necessary definitions of points, lines, and triangles
variables {A B C D E F O : Point}

-- Conditions
variable h1 : is_acute_triangle A B C
variable h2 : is_altitude A D A B C
variable h3 : is_altitude B E A B C
variable h4 : is_altitude C F A B C
variable h5 : is_circumcenter O A B C

-- Theorem statement
theorem dissection_into_equal_area_triangles :
  dissect_equal_area_triangles A B C D E F O :=
sorry

end dissection_into_equal_area_triangles_l273_273866


namespace find_p_l273_273865

noncomputable theory
open_locale classical

theorem find_p
  (p q r : ℂ)
  (h_p_real : p.im = 0)
  (h1 : p + q + r = 4)
  (h2 : p * q + q * r + r * p = 4)
  (h3 : p * q * r = 4) :
  p = 2 + complex.cbrt 11 :=
begin
  sorry
end

end find_p_l273_273865


namespace sum_of_coordinates_eq_16_l273_273896

open Real

def C : ℝ × ℝ := (a, 8)
def D : ℝ × ℝ := (-a, 8)

theorem sum_of_coordinates_eq_16 (a : ℝ) : 
  C.1 + C.2 + D.1 + D.2 = 16 := 
by 
  -- C = (a, 8) and D = (-a, 8)
  -- coordinates sum: a + 8 + (-a) + 8 = 0 + 16 = 16
  sorry

end sum_of_coordinates_eq_16_l273_273896


namespace sum_first_sequence_terms_l273_273950

theorem sum_first_sequence_terms 
  (S : ℕ → ℕ) 
  (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n - S (n - 1) = 2 * n - 1)
  (h2 : S 2 = 3) 
  : a 1 + a 3 = 5 :=
sorry

end sum_first_sequence_terms_l273_273950


namespace seq_a4_eq_2_over_5_l273_273417

def a_seq : ℕ → ℚ
| 0 := 0
| 1 := 1
| (n + 2) := 2 * a_seq (n + 1) / (a_seq (n + 1) + 2)

theorem seq_a4_eq_2_over_5 : a_seq 4 = 2 / 5 :=
by
  sorry

end seq_a4_eq_2_over_5_l273_273417


namespace smallest_r_l273_273305

noncomputable def exists_diff_functions_r (r : ℝ) : Prop :=
∃ (f g : ℝ → ℝ), (differentiable ℝ f) ∧ (differentiable ℝ g) ∧
  (f 0 > 0) ∧ (g 0 = 0) ∧
  (∀ x, abs (deriv f x) ≤ abs (g x)) ∧
  (∀ x, abs (deriv g x) ≤ abs (f x)) ∧
  (f r = 0)

theorem smallest_r : ∃ (r : ℝ), r > 0 ∧ exists_diff_functions_r r ∧ r = (Real.pi / 2) :=
sorry

end smallest_r_l273_273305


namespace nicolai_ate_6_pounds_of_peaches_l273_273574

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end nicolai_ate_6_pounds_of_peaches_l273_273574


namespace selection_schemes_are_54_l273_273711

theorem selection_schemes_are_54 {M F: Type} [Fintype M] [Fintype F] 
  (m: ℕ) (f: ℕ) (hM: Fintype.card M = 3) (hF: Fintype.card F = 3)
  (hc: Choose (3 : ℕ) 1 = 3) (hf: Choose (3 : ℕ) 2 = 3) 
  (ha: A 3 3 = 6) : (3 * 3 * 6 = 54) :=
by 
  sorry

end selection_schemes_are_54_l273_273711


namespace number_of_subsets_of_M_is_eight_l273_273066

def M := {x : ℤ | x^2 + 2 * x - 3 < 0}

theorem number_of_subsets_of_M_is_eight :
  M.toFinset.powerset.card = 8 :=
sorry

end number_of_subsets_of_M_is_eight_l273_273066


namespace general_formula_and_Tn_bounds_l273_273364

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (a b c : ℝ) :=
b * b = a * c

theorem general_formula_and_Tn_bounds (a : ℕ → ℝ) (S_n T_n : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d) (h2 : d ≠ 0) (h3 : a 1 + a 7 = 22)
  (h4 : geometric_sequence (a 3) (a 6) (a 11)) :
  (∀ n, a n = 2 * n + 1) ∧ (∀ n, T_n n = ∑ k in finset.range n, 1 / S_n (k + 1) ∧
                              S_n n = n * (3 + (2 * n + 1)) / 2 ∧
                              ∑ k in finset.range n, 1 / S_n (k + 1) ≥ 1 / 3 ∧ 
                              ∑ k in finset.range n, 1 / S_n (k + 1) < 3 / 4) :=
sorry

end general_formula_and_Tn_bounds_l273_273364


namespace distance_walked_in_9_minutes_l273_273050

theorem distance_walked_in_9_minutes :
  ∀ (rate : ℝ), rate = (2 / 36 : ℝ) → (rate * 9 = 0.5) :=
by 
  intro rate h_rate
  rw [h_rate]
  have : (2 / 36) * 9 = 0.5 := by sorry
  rw [this]

end distance_walked_in_9_minutes_l273_273050


namespace find_circumcircle_radius_l273_273172

-- Conditions definitions
variables (r1 r2 : ℝ)
variables (O1 O2 A B C : Type)
variables (radius_A : ℝ)

-- Hypotheses based on conditions
h1 : r1 + r2 = 7
h2 : (r1 + 8)^2 + (r2 + 8)^2 = 17^2
h3 : O1 ≠ O2
h4 : (radius_A : ℝ) = 8
h5 : (A : ℝ) = 2 * sqrt 15
h6 : A = B
h7 : A = C

-- Expected conclusion
theorem find_circumcircle_radius :
  radius_A = 2 * sqrt 15 := by sorry

end find_circumcircle_radius_l273_273172


namespace gunther_typing_l273_273424

theorem gunther_typing : 
  (∀ (number_of_words : ℕ) (minutes_per_set : ℕ) (total_working_minutes : ℕ),
    number_of_words = 160 → minutes_per_set = 3 → total_working_minutes = 480 →
    (total_working_minutes / minutes_per_set * number_of_words) = 25600) :=
begin
  intros number_of_words minutes_per_set total_working_minutes,
  intros h_words h_time h_total,
  rw [h_words, h_time, h_total],
  norm_num,
end

end gunther_typing_l273_273424


namespace simplify_trig_expr_l273_273913

theorem simplify_trig_expr : 
  real.sqrt (2 - real.sin 1 ^ 2 + real.cos 2) = real.sqrt 3 * real.cos 1 := 
by 
  sorry

end simplify_trig_expr_l273_273913


namespace find_number_of_sides_l273_273953

-- Defining the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Statement of the problem
theorem find_number_of_sides (h : sum_of_interior_angles n = 1260) : n = 9 :=
by
  sorry

end find_number_of_sides_l273_273953


namespace function_increasing_interval_l273_273945

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x ^ 2) / Real.log 2

def domain (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem function_increasing_interval : 
  ∀ x, domain x → 0 < x ∧ x < 1 → ∀ y, domain y → 0 < y ∧ y < 1 → x < y → f x < f y :=
by 
  intros x hx h0 y hy h1 hxy
  sorry

end function_increasing_interval_l273_273945


namespace problem1_problem2a_problem2b_problem3_l273_273719

noncomputable def f (a x : ℝ) := -x^2 + a * x - 2
noncomputable def g (x : ℝ) := x * Real.log x

-- Problem 1
theorem problem1 {a : ℝ} : (∀ x : ℝ, 0 < x → g x ≥ f a x) → a ≤ 3 :=
sorry

-- Problem 2 
theorem problem2a (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1 / Real.exp 1) :
  ∃ xmin : ℝ, g (1 / Real.exp 1) = -1 / Real.exp 1 ∧ 
  ∃ xmax : ℝ, g (m + 1) = (m + 1) * Real.log (m + 1) :=
sorry

theorem problem2b (m : ℝ) (h₀ : 1 / Real.exp 1 ≤ m) :
  ∃ xmin ymax : ℝ, xmin = g m ∧ ymax = g (m + 1) :=
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : 0 < x) : 
  Real.log x + (2 / (Real.exp 1 * x)) ≥ 1 / Real.exp x :=
sorry

end problem1_problem2a_problem2b_problem3_l273_273719


namespace problem1_problem2_l273_273767

noncomputable def f1 (x : ℝ) := Real.log x - (1/3) * x
noncomputable def f2 (x : ℝ) := Real.log x + x - (2 / 2) * x ^ 2

theorem problem1 : ∃ x ∈ Set.Icc Real.exp (Real.exp 2), is_max_on f1 (Set.Icc Real.exp (Real.exp 2)) x ∧ f1 x = Real.log 3 - 1 :=
sorry

theorem problem2 : ∀ t > 0, (∀ y, (Real.log y + y - (2 / t) * y ^ 2 = 0) → y = 1) ↔ t = 2 :=
sorry

end problem1_problem2_l273_273767


namespace find_x_values_l273_273696

theorem find_x_values (x : ℚ) : 
  (floor ((8 * x + 19) / 7) = 16 * (x + 1) / 11) ↔ 
  (x = 17 / 16 ∨ x = 28 / 16 ∨ x = 39 / 16 ∨ x = 50 / 16 ∨ x = 61 / 16) :=
by
  sorry

end find_x_values_l273_273696


namespace second_pipe_fill_time_l273_273970

theorem second_pipe_fill_time :
  ∃ x : ℝ, x ≠ 0 ∧ (1 / 10 + 1 / x - 1 / 20 = 1 / 7.5) ∧ x = 60 :=
by
  sorry

end second_pipe_fill_time_l273_273970


namespace correct_m_l273_273299

noncomputable def seq : ℕ → ℝ
| 0       := 7
| (n + 1) := (seq n)^2 + 6 * (seq n) + 8 / (seq n + 7)

def find_m (limit : ℝ) : ℕ :=
  Nat.find (λ m, seq m ≤ limit)

theorem correct_m : find_m (5 + 1 / 2^15) = 207 :=
by
  sorry

end correct_m_l273_273299


namespace binomial_expansion_b_value_l273_273345

theorem binomial_expansion_b_value (a b x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x ^ 2 + a^5 * x ^ 5) : b = 40 := 
sorry

end binomial_expansion_b_value_l273_273345


namespace width_of_room_l273_273942

theorem width_of_room 
  (length : ℝ) 
  (cost : ℝ) 
  (rate : ℝ) 
  (h_length : length = 6.5) 
  (h_cost : cost = 10725) 
  (h_rate : rate = 600) 
  : (cost / rate) / length = 2.75 :=
by
  rw [h_length, h_cost, h_rate]
  norm_num

end width_of_room_l273_273942


namespace problem_inequality_l273_273352

variable {a b c : ℝ}
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)

noncomputable def sum_sqrt_over_a_minus_sum_one_over_a_ge_sqrt_three : Prop :=
  (Real.sqrt (1 + a^2) / a + Real.sqrt (1 + b^2) / b + Real.sqrt (1 + c^2) / c - 
  (1 / a + 1 / b + 1 / c) ≥ Real.sqrt 3)

theorem problem_inequality (h_eq : a + b + c = a * b * c) : 
  sum_sqrt_over_a_minus_sum_one_over_a_ge_sqrt_three h_pos_a h_pos_b h_pos_c :=
sorry

end problem_inequality_l273_273352


namespace area_of_ABCD_proof_l273_273107

noncomputable def point := ℝ × ℝ

structure Rectangle :=
  (A B C D : point)
  (angle_C_trisected_by_CE_CF : Prop)
  (E_on_AB : Prop)
  (F_on_AD : Prop)
  (AF : ℝ)
  (BE : ℝ)

def area_of_rectangle (rect : Rectangle) : ℝ :=
  let (x1, y1) := rect.A
  let (x2, y2) := rect.C
  (x2 - x1) * (y2 - y1)

theorem area_of_ABCD_proof :
  ∀ (ABCD : Rectangle),
    ABCD.angle_C_trisected_by_CE_CF →
    ABCD.E_on_AB →
    ABCD.F_on_AD →
    ABCD.AF = 2 →
    ABCD.BE = 6 →
    abs (area_of_rectangle ABCD - 150) < 1 :=
by
  sorry

end area_of_ABCD_proof_l273_273107


namespace base_three_to_ten_l273_273584

theorem base_three_to_ten (n : ℕ) (h₀ : n = 20123) : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 3 * 3^0) = 180 :=
by
  rw h₀
  sorry

end base_three_to_ten_l273_273584


namespace bounded_sequence_has_convergent_subsequence_l273_273900

theorem bounded_sequence_has_convergent_subsequence {a : ℕ → ℝ} (M : ℝ)
  (h_bounded : ∀ n : ℕ, |a n| ≤ M) :
  ∃ (n_k : ℕ → ℕ), strict_mono n_k ∧ ∃ L : ℝ, ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K, |a (n_k k) - L| < ε :=
by
  sorry

end bounded_sequence_has_convergent_subsequence_l273_273900


namespace quadratic_prod_sign_l273_273804

-- Definitions of the conditions from (a)
variables {a b c m x1 n x2 p : ℝ}
variable h_a_pos : a > 0
variable h_m_lt_x1 : m < x1
variable h_x1_lt_n : x1 < n
variable h_n_lt_x2 : n < x2
variable h_x2_lt_p : x2 < p

-- Define the quadratic function
def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Theorem statement translating the conditions and the question
theorem quadratic_prod_sign : f m * f n * f p < 0 :=
by
  -- Assuming the specific ordering of roots, we aim to show the product is negative
  sorry

end quadratic_prod_sign_l273_273804


namespace calc_exp_l273_273286

open Real

theorem calc_exp (x y : ℝ) : 
  (-(1/3) * (x^2) * y) ^ 3 = -(x^6 * y^3) / 27 := 
  sorry

end calc_exp_l273_273286


namespace monotonic_intervals_and_non_negative_f_l273_273764

noncomputable def f (m x : ℝ) : ℝ := m / x - m + Real.log x

theorem monotonic_intervals_and_non_negative_f (m : ℝ) : 
  (∀ x > 0, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end monotonic_intervals_and_non_negative_f_l273_273764


namespace jebb_total_spent_l273_273054

theorem jebb_total_spent
  (cost_of_food : ℝ) (service_fee_rate : ℝ) (tip : ℝ)
  (h1 : cost_of_food = 50)
  (h2 : service_fee_rate = 0.12)
  (h3 : tip = 5) :
  cost_of_food + (cost_of_food * service_fee_rate) + tip = 61 := 
sorry

end jebb_total_spent_l273_273054


namespace compound_interest_difference_l273_273179

theorem compound_interest_difference 
  (P : ℝ) (n : ℝ) (r : ℝ) 
  (H_P : P = 8000) 
  (H_n : n = 1.5) 
  (H_diff : 8000 * ((1 + r / 2) ^ 3 - (1 + r) ^ 1.5) = 3.263999999999214) : 
  8000 * ((1 + r / 2) ^ 3 - (1 + r) ^ 1.5) = 3.263999999999214 :=
by 
  sorry

end compound_interest_difference_l273_273179


namespace geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l273_273361

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n ≠ 0 ∧ (a (n + 1) = a n * (a (n + 1) / a n))

theorem geometric_sequence_implies_condition (a : ℕ → ℝ) :
  is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1) := sorry

theorem counterexample_condition_does_not_imply_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a := sorry

theorem geometric_sequence_sufficient_not_necessary (a : ℕ → ℝ) :
  (is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1)) ∧
  ((∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a) := by
  exact ⟨geometric_sequence_implies_condition a, counterexample_condition_does_not_imply_geometric_sequence a⟩

end geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l273_273361


namespace slope_sign_relation_l273_273545

variable (x y : ℝ)
variable (σ_x σ_y : ℝ)
variable (r b : ℝ)
variable (a : ℝ)

-- Assume standard deviations σ_x and σ_y are non-negative
axiom standard_deviations_nonnegative : σ_x ≥ 0 ∧ σ_y ≥ 0

-- Assume there is a linear relationship between x and y
def linear_relation (y : ℝ) (x : ℝ) (b : ℝ) (a : ℝ) : Prop :=
  y = b * x + a

-- Assume the slope of the regression line is defined as r * (σ_y / σ_x)
axiom slope_definition : b = r * (σ_y / σ_x)

-- To prove that sign of b is the same as sign of r
theorem slope_sign_relation :
  (σ_x ≠ 0) → 
  (σ_y ≠ 0) → 
  standard_deviations_nonnegative →
  linear_relation y x b a →
  slope_definition →
  (b > 0 ↔ r > 0) ∧ (b < 0 ↔ r < 0) :=
  by
    intros
    sorry

end slope_sign_relation_l273_273545


namespace geometric_probability_l273_273651

noncomputable def probability_point_within_rectangle (l w : ℝ) (A_rectangle A_circle : ℝ) : ℝ :=
  A_rectangle / A_circle

theorem geometric_probability (l w : ℝ) (r : ℝ) (A_rectangle : ℝ) (h_length : l = 4) 
  (h_width : w = 3) (h_radius : r = 2.5) (h_area_rectangle : A_rectangle = 12) :
  A_rectangle / (Real.pi * r^2) = 48 / (25 * Real.pi) :=
by
  sorry

end geometric_probability_l273_273651


namespace three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l273_273195

theorem three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four :
  (3.242 * 12) / 100 = 0.38904 :=
by 
  sorry

end three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l273_273195


namespace cube_edge_length_QR_isqrt6_l273_273618

noncomputable def cube_diagonal_length (a : ℝ) := real.sqrt (a^2 + a^2)

noncomputable def diagonal_midpoint_length (a : ℝ) := cube_diagonal_length a / 2

noncomputable def qr_length (a : ℝ) := real.sqrt (diagonal_midpoint_length a ^ 2 + a^2)

theorem cube_edge_length_QR_isqrt6 : qr_length 2 = real.sqrt 6 :=
by sorry

end cube_edge_length_QR_isqrt6_l273_273618


namespace largest_digit_divisible_by_6_l273_273993

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l273_273993


namespace cannot_arrange_sequence_l273_273668

theorem cannot_arrange_sequence :
  ¬ (∃ (a b : Fin 4010 → Fin 4011), 
    (∀ i : Fin 2005, a i < b i ∧ b i = a i + i + 1) ∧ 
    2 * (∑ i in Finset.range 2005, a ⟨i, _⟩) + 2005 * 1004 = 2005 * 4011) :=
by
  sorry

end cannot_arrange_sequence_l273_273668


namespace variance_of_X_equals_0_48_l273_273702

-- Definitions from the conditions
def random_variable_values := {0, 1, 2} -- Possible values of X
def event_prob : ℕ → ℝ → ℝ → ℝ
| 0, p, q => q^2
| 1, p, q => 2 * p * q
| 2, p, q => p^2

def expected_value (p q : ℝ) : ℝ := 0 * (q^2) + 1 * (2 * p * q) + 2 * (p^2)

-- Given condition
def given_expected_value : ℝ := 1.2
def p : ℝ := given_expected_value / 2
def q : ℝ := 1 - p

-- Lean proof statement
theorem variance_of_X_equals_0_48 : (2 * p * q) = 0.48 := by
  have p_eq : p = 0.6 := by
    simp only [p, given_expected_value]
    norm_num
  have q_eq : q = 0.4 := by
    simp only [q, p_eq]
    norm_num
  rw [p_eq, q_eq]
  simp only [mul_eq_mul_right_iff, eq_self_iff_true, true_or]
  norm_num
  exact sorry -- replace this with the actual proof if needed

end variance_of_X_equals_0_48_l273_273702


namespace find_a_b_sum_l273_273857

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_b_sum (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 24) : a + b = 6 :=
  sorry

end find_a_b_sum_l273_273857


namespace find_principal_6400_l273_273929

theorem find_principal_6400 (CI SI P : ℝ) (R T : ℝ) 
  (hR : R = 5) (hT : T = 2) 
  (hSI : SI = P * R * T / 100) 
  (hCI : CI = P * (1 + R / 100) ^ T - P) 
  (hDiff : CI - SI = 16) : 
  P = 6400 := 
by 
  sorry

end find_principal_6400_l273_273929


namespace b_n_geometric_sequence_sum_of_c_n_l273_273395

-- Define the sequence a_n
def a_n (n : ℕ) : ℕ := 2 * n

-- Define the sequence b_n
def b_n (n : ℕ) : ℕ := 4 ^ n

-- Prove that b_n is a geometric sequence
theorem b_n_geometric_sequence (n : ℕ) : 
  b_n (n+1) / b_n n = 4 := by
  -- proof
  sorry

-- Define the sequence c_n
def c_n (n : ℕ) : ℕ := a_n n + b_n n

-- Define the sum of the first n terms of a_n
def S_n (n : ℕ) : ℕ := (n * (n + 1))

-- Define the sum of the first n terms of b_n
def T_n (n : ℕ) : ℕ := 4 * (4 ^ n - 1) / 3

-- Define the sum of the first n terms of c_n (A_n)
def A_n (n : ℕ) : ℕ := S_n n + T_n n

-- Prove that the sum of the first n terms of c_n is A_n
theorem sum_of_c_n (n : ℕ) :
  (∑ i in Finset.range n, c_n (i+1)) = A_n n := by
  -- proof
  sorry

end b_n_geometric_sequence_sum_of_c_n_l273_273395


namespace seven_digit_number_count_l273_273005

theorem seven_digit_number_count (h1 : ∀ n, n ∈ (list.range 10)) :
  (∃ l : list ℕ, l.length = 7 ∧ (l.prod = 45^3 ∧ l.all (λ d, d < 10))) →
    (∃ cnt : ℕ, cnt = 350) :=
begin
  sorry
end

end seven_digit_number_count_l273_273005


namespace smallest_sum_with_probability_l273_273162

theorem smallest_sum_with_probability (N : ℕ) (p : ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 6) (h2 : 6 * N = 2022) (h3 : p > 0) :
  ∃ M, M = 337 ∧ (∀ sum, sum = 2022 → P(sum) = p) ∧ (∀ min_sum, min_sum = N → P(min_sum) = p):=
begin
  sorry
end

end smallest_sum_with_probability_l273_273162


namespace pink_highlighters_count_l273_273025

theorem pink_highlighters_count (yellow highlighters_blue total : ℕ) 
(h1 : yellow = 15) (h2 : blue = 8) (h3 : total = 33) : 
total - (yellow + blue) = 10 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end pink_highlighters_count_l273_273025


namespace propA_necessary_not_sufficient_for_propB_l273_273823

variable (A B : Point) (P : Point)
-- Definition of the proposition that the sum of the distances from P to A and B is constant
def PropA := ∃ k : ℝ, |dist P A + dist P B| = k
-- Definition of the proposition that P's trajectory is an ellipse with foci A and B
def PropB := is_ellipse_with_foci A B P

-- Lean theorem statement
theorem propA_necessary_not_sufficient_for_propB (A B P : Point) :
  (PropA A B P → PropB A B P) ∧ (PropB A B P → PropA A B P) :=
by
  sorry

end propA_necessary_not_sufficient_for_propB_l273_273823


namespace find_n_l273_273010

theorem find_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = (Real.log (s / P)) / (Real.log (1 + k + m)) :=
sorry

end find_n_l273_273010


namespace line_m_b_l273_273932

def point (x y : ℝ) := (x, y)

def slope (p1 p2 : (ℝ × ℝ)) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

def intercept (p : (ℝ × ℝ)) (m : ℝ) : ℝ := p.2 - m * p.1

theorem line_m_b (p1 p2 : (ℝ × ℝ))
  (h1 : p1 = point 1 (-2))
  (h2 : p2 = point 4 7) :
  let m := slope p1 p2,
      b := intercept p1 m
  in m + b = -2 :=
by
  sorry

end line_m_b_l273_273932


namespace ratio_of_area_of_inscribed_circle_to_triangle_l273_273127

theorem ratio_of_area_of_inscribed_circle_to_triangle (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  (π * r) / s = (5 * π * r) / (12 * h) :=
by
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  sorry

end ratio_of_area_of_inscribed_circle_to_triangle_l273_273127


namespace prob_at_least_one_multiple_of_4_60_l273_273058

def num_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def total_numbers_in_range (n : ℕ) : ℕ :=
  n

def num_not_multiples_of_4 (n : ℕ) : ℕ :=
  total_numbers_in_range n - num_multiples_of_4 n

def prob_no_multiple_of_4 (n : ℕ) : ℚ :=
  let p := num_not_multiples_of_4 n / total_numbers_in_range n
  p * p

def prob_at_least_one_multiple_of_4 (n : ℕ) : ℚ :=
  1 - prob_no_multiple_of_4 n

theorem prob_at_least_one_multiple_of_4_60 :
  prob_at_least_one_multiple_of_4 60 = 7 / 16 :=
by
  -- Proof is skipped.
  sorry

end prob_at_least_one_multiple_of_4_60_l273_273058


namespace problem1_problem2_l273_273620

theorem problem1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

theorem problem2 (a b : ℝ) (h : (|a - 2| + |b + 2| = 0) ∨ (|a - 2| * |b + 2| < 0)) : 3a + 2b = 2 := by
  sorry

end problem1_problem2_l273_273620


namespace centroid_distance_squared_l273_273062

noncomputable def trianglePoints (D E F : ℝ × ℝ) : Prop :=
(G = (1 / 3) • (D + E + F))

theorem centroid_distance_squared
  (D E F G : ℝ × ℝ)
  (hG : G = (1 / 3) • (D + E + F))
  (h : (gneptune_distance G D)^2 + (gneptune_distance G E)^2 + (gneptune_distance G F)^2 = 90) :
  (gneptune_distance D E)^2 + (gneptune_distance D F)^2 + (gneptune_distance E F)^2 = 270 := sorry

end centroid_distance_squared_l273_273062


namespace median_length_is_correct_l273_273869

noncomputable def median_length (A B C O : Point) (a : ℝ) 
  (h1 : ∠ BAC = 35) 
  (h2 : ∠ BOC = 145) 
  (h3 : dist B C = a)
  (h4 : is_centroid O A B C) : ℝ :=
m

theorem median_length_is_correct
  (A B C O : Point) (a : ℝ) 
  (h1 : ∠ BAC = 35) 
  (h2 : ∠ BOC = 145) 
  (h3 : dist B C = a)
  (h4 : is_centroid O A B C) :
  median_length A B C O a h1 h2 h3 h4 = a * (sqrt 3) / 2 :=
sorry

end median_length_is_correct_l273_273869


namespace no_ten_nearly_convex_polygons_l273_273099

noncomputable def condition_points : set (fin 10 → ℝ × ℝ) :=
  { points | ∀ p1 p2 p3, p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → ¬ collinear ({p1, p2, p3} : set (ℝ × ℝ)) }

theorem no_ten_nearly_convex_polygons (P : condition_points) :
  ¬ ∃ polygons : set (set (ℝ × ℝ)), polygons.card = 10 ∧ 
    ∀ poly ∈ polygons, 
      (is_nearly_convex poly ∧ ∀ p ∈ poly, p ∈ P) := 
sorry

end no_ten_nearly_convex_polygons_l273_273099


namespace sum_of_faces_edges_vertices_l273_273593

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end sum_of_faces_edges_vertices_l273_273593


namespace largest_digit_divisible_by_6_l273_273983

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l273_273983


namespace gym_monthly_income_l273_273640

theorem gym_monthly_income (bi_monthly_charge : ℕ) (members : ℕ) (monthly_income : ℕ) 
  (h1 : bi_monthly_charge = 18)
  (h2 : members = 300)
  (h3 : monthly_income = 10800) : 
  2 * bi_monthly_charge * members = monthly_income :=
by
  rw [h1, h2, h3]
  norm_num

end gym_monthly_income_l273_273640


namespace solve_x_squared_solve_x_cubed_l273_273701

-- Define the first problem with its condition and prove the possible solutions
theorem solve_x_squared {x : ℝ} (h : (x + 1)^2 = 9) : x = 2 ∨ x = -4 :=
sorry

-- Define the second problem with its condition and prove the possible solution
theorem solve_x_cubed {x : ℝ} (h : -2 * (x^3 - 1) = 18) : x = -2 :=
sorry

end solve_x_squared_solve_x_cubed_l273_273701


namespace graph_does_not_pass_through_quadrant_II_l273_273938

noncomputable def linear_function (x : ℝ) : ℝ := 3 * x - 4

def passes_through_quadrant_I (x : ℝ) : Prop := x > 0 ∧ linear_function x > 0
def passes_through_quadrant_II (x : ℝ) : Prop := x < 0 ∧ linear_function x > 0
def passes_through_quadrant_III (x : ℝ) : Prop := x < 0 ∧ linear_function x < 0
def passes_through_quadrant_IV (x : ℝ) : Prop := x > 0 ∧ linear_function x < 0

theorem graph_does_not_pass_through_quadrant_II :
  ¬(∃ x : ℝ, passes_through_quadrant_II x) :=
sorry

end graph_does_not_pass_through_quadrant_II_l273_273938


namespace decimal_to_base7_l273_273296

theorem decimal_to_base7 :
    ∃ k₀ k₁ k₂ k₃ k₄, 1987 = k₀ * 7^4 + k₁ * 7^3 + k₂ * 7^2 + k₃ * 7^1 + k₄ * 7^0 ∧
    k₀ = 0 ∧
    k₁ = 5 ∧
    k₂ = 3 ∧
    k₃ = 5 ∧
    k₄ = 6 :=
by
  sorry

end decimal_to_base7_l273_273296


namespace gallons_in_one_ounce_l273_273057

-- Let's state the conditions based on the problem statement
def drinks_per_time : ℕ := 8
def times_per_day : ℕ := 8
def days : ℕ := 5
def gallons_prepared : ℝ := 2.5
def ounces_per_gallon : ℝ := 128

-- Define water consumption in ounces for 1 day and 5 days
def daily_consumption (drinks_per_time : ℕ) (times_per_day : ℕ) : ℝ := drinks_per_time * times_per_day
def total_consumption (daily : ℝ) (days : ℕ) : ℝ := daily * days

-- Define gallons per ounce calculation
def gallons_per_ounce (gallons : ℝ) (total_ounces : ℝ) : ℝ := gallons / total_ounces

-- The mathematically equivalent proof problem
theorem gallons_in_one_ounce :
  gallons_per_ounce gallons_prepared (total_consumption (daily_consumption drinks_per_time times_per_day) days) = 1 / ounces_per_gallon :=
by
  -- Proof is not required, so we use "sorry" to skip it
  sorry

end gallons_in_one_ounce_l273_273057


namespace problem_solution_l273_273873

noncomputable def a := 504^502 - 504^(-502)
noncomputable def b := 504^502 + 504^(-502)
noncomputable def c := 504^503 - 504^(-503)
noncomputable def d := 504^503 + 504^(-503)

theorem problem_solution : a^2 - b^2 + c^2 - d^2 = -8 :=
by
  sorry

end problem_solution_l273_273873


namespace sum_of_digits_greatest_prime_divisor_16385_l273_273324

-- Definition of the number to be factorized
def num : ℕ := 16385

-- Definition to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ k : ℕ, k > 1 → k < n → ¬(k ∣ n))

-- Function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Definition to find the greatest prime divisor of a number
noncomputable def greatest_prime_divisor (n : ℕ) : ℕ :=
  let divisors := (list.range (n+1)).filter (λ d, d ∣ n ∧ is_prime d)
  divisors.maximum (≤).get_or_else 1

-- Theorem: The sum of the digits of the greatest prime divisor of 16385 is 19.
theorem sum_of_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor 16385) = 19 := by
  sorry

end sum_of_digits_greatest_prime_divisor_16385_l273_273324


namespace triangle_side_lengths_sum_l273_273064

variable (D E F G : ℝ^3)
variable (centroid : G = (D + E + F) / 3)
variable (sum_squares : norm (G - D)^2 + norm (G - E)^2 + norm (G - F)^2 = 90)

theorem triangle_side_lengths_sum : (norm (D - E))^2 + (norm (D - F))^2 + (norm (E - F))^2 = 270 :=
by
  sorry

end triangle_side_lengths_sum_l273_273064


namespace Jerry_weekly_earnings_l273_273486

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end Jerry_weekly_earnings_l273_273486


namespace quadratic_inequality_solution_l273_273020

theorem quadratic_inequality_solution (a : ℝ) (h_a : 0 < a) :
  let b := -2 * a,
      c := -8 * a,
      f := λ x : ℝ, a * x^2 + b * x + c in
  f 2 < f (-1) ∧ f (-1) < f 5 :=
by {
  let b := -2 * a,
  let c := -8 * a,
  let f := λ x : ℝ, a * x^2 + b * x + c,
  have h_b: b = -2 * a := rfl,
  have h_c: c = -8 * a := rfl,
  have h_2_lt_neg1 : a * 4 - 2 * a * 2 - 8 * a < a * 1 - 2 * a * (-1) - 8 * a, {
    calc
      a * 4 - 2 * a * 2 - 8 * a
        = a * 4 - 4 * a - 8 * a : by simp
    ... = -8 * a : by ring
    ... < -6 * a : by { have : 0 < a, from h_a, linarith },
  },
  have h_neg1_lt_5 : a * 1 - 2 * a * (-1) - 8 * a < a * 25 - 2 * a * 5 - 8 * a, {
    calc
      a * 1 - 2 * a * (-1) - 8 * a
        = a * 1 + 2 * a - 8 * a : by ring
    ... = -5 * a : by ring
    ... < 15 * a : by { have : 0 < a, from h_a, linarith },
  },
  exact ⟨h_2_lt_neg1, h_neg1_lt_5⟩,
}

end quadratic_inequality_solution_l273_273020


namespace percentage_of_goods_lost_theorem_l273_273633

def cost_price : ℝ := 100
def profit_percent : ℝ := 10
def loss_percent : ℝ := 12

def selling_price (CP : ℝ) (P : ℝ) : ℝ := CP + (CP * P / 100)
def loss_value (SP : ℝ) (L : ℝ) : ℝ := SP * L / 100
def percentage_goods_lost (loss : ℝ) (CP : ℝ) : ℝ := (loss / CP) * 100

theorem percentage_of_goods_lost_theorem
  (CP : ℝ) (P : ℝ) (L : ℝ) :
  percentage_goods_lost (loss_value (selling_price CP P) L) CP = 13.2 :=
by
  -- Definitions and assumptions here correspond to the given problem conditions.
  let CP := cost_price
  let P := profit_percent
  let L := loss_percent
  let SP := selling_price CP P
  let loss := loss_value SP L
  let percentage := percentage_goods_lost loss CP
  show percentage = 13.2
  sorry

end percentage_of_goods_lost_theorem_l273_273633


namespace false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l273_273604

-- Proposition A
theorem false_proposition_A (a b c : ℝ) (hac : a > b) (hca : b > 0) : ac * c^2 = b * c^2 :=
  sorry

-- Proposition B
theorem false_proposition_B (a b : ℝ) (hab : a < b) : (1/a) < (1/b) :=
  sorry

-- Proposition C
theorem true_proposition_C (a b : ℝ) (hab : a > b) (hba : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
  sorry

-- Proposition D
theorem true_proposition_D (a b : ℝ) (hba : a > |b|) : a^2 > b^2 :=
  sorry

end false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l273_273604


namespace dot_product_eq_eleven_l273_273388

variable {V : Type _} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)
variable (cos_theta : ℝ)
variable (norm_a norm_b : ℝ)

axiom cos_theta_def : cos_theta = 1 / 3
axiom norm_a_def : ‖a‖ = 1
axiom norm_b_def : ‖b‖ = 3

theorem dot_product_eq_eleven
  (cos_theta_def : cos_theta = 1 / 3)
  (norm_a_def : ‖a‖ = 1)
  (norm_b_def : ‖b‖ = 3) :
  (2 • a + b) ⬝ b = 11 := 
sorry

end dot_product_eq_eleven_l273_273388


namespace irrational_count_eq_one_l273_273662

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def count_irrationals (s : set ℝ) : ℕ :=
  (s.filter is_irrational).card

theorem irrational_count_eq_one : 
  count_irrationals { sqrt 3, sqrt 4, 3.1415, 5 / 6 } = 1 :=
by sorry

end irrational_count_eq_one_l273_273662


namespace pencil_count_l273_273139

/-- 
If there are initially 115 pencils in the drawer, and Sara adds 100 more pencils, 
then the total number of pencils in the drawer is 215.
-/
theorem pencil_count (initial_pencils added_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : added_pencils = 100) : 
  initial_pencils + added_pencils = 215 := by
  sorry

end pencil_count_l273_273139


namespace angle_BMA_half_arc_diff_l273_273104

variable {C : Type} [MetricSpace C] [NormedAddTorsor ℝ C] -- Circle context
variables (O A B M C : C) -- Points involved

-- Condition: M is the point of tangency, MB is tangent, MAC is secant
def is_tangent (O M B : C) : Prop := sorry -- Property defining tangency at M
def is_secant (O M A C : C) : Prop := sorry -- Property defining secancy through A and C
def arc_length (O A B : C) : ℝ := sorry -- Length of the arc AB in circle centered at O

theorem angle_BMA_half_arc_diff
  (h_tangent : is_tangent O M B)
  (h_secant : is_secant O M A C) :
  let θ := angle O B A in -- Angle BAC = θ
  let φ := angle O A C in -- Angle BMA = φ
  φ = 1 / 2 * (arc_length O B C - arc_length O A B) := sorry

end angle_BMA_half_arc_diff_l273_273104


namespace quadratic_real_solns_l273_273301

theorem quadratic_real_solns (c : ℝ) (h1 : 0 < c) (h2 : c < 25) :
  ∃ x : ℝ, x^2 - 10 * x + c < 0 :=
begin
  sorry
end

end quadratic_real_solns_l273_273301


namespace Missy_handle_claims_l273_273290

def MissyCapacity (JanCapacity JohnCapacity MissyCapacity : ℕ) :=
  JanCapacity = 20 ∧
  JohnCapacity = JanCapacity + (30 * JanCapacity) / 100 ∧
  MissyCapacity = JohnCapacity + 15

theorem Missy_handle_claims :
  ∃ MissyCapacity, MissyCapacity = 41 :=
by
  use 41
  unfold MissyCapacity
  sorry

end Missy_handle_claims_l273_273290


namespace find_k_l273_273781

section
variables {k : ℝ}
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c : ℝ × ℝ := (k, -2)
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_k (h : dot_product (vector_sub a c) b = 0) : k = 12 :=
sorry
end

end find_k_l273_273781


namespace g_of_3_eq_4_l273_273354

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

def g (x : ℝ) : ℝ := 2^(x - 1)

theorem g_of_3_eq_4 : g 3 = 4 := by
  sorry

end g_of_3_eq_4_l273_273354


namespace simplify_and_evaluate_l273_273109

noncomputable def simplified_expr (x y : ℝ) : ℝ :=
  ((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)

theorem simplify_and_evaluate :
  let x := -1
  let y := 2
  simplified_expr x y = 1 :=
by
  -- Proof will go here
  sorry

end simplify_and_evaluate_l273_273109


namespace sqrt_cube_logarithm_l273_273293

theorem sqrt_cube_logarithm :
  ( ( Real.sqrt 2 * Real.cbrt 3 ) ^ 6 - Real.log2 (Real.log2 16) ) = 70 :=
by
  sorry

end sqrt_cube_logarithm_l273_273293


namespace trajectory_of_X_in_regular_n_gon_l273_273359

theorem trajectory_of_X_in_regular_n_gon (n : ℕ) (hn : n ≥ 5) (R : ℝ) (O A B X Y Z : ℝ×ℝ)
  (congruent_triangles : congruent (triangle O A B) (triangle X Y Z))
  (initial_coincidence : X = O ∧ Y = A ∧ Z = B)
  (Y_Z_trace_boundary : ∀ t, (Y t ∈ boundary_n_gon n R O) ∧ (Z t ∈ boundary_n_gon n R O))
  (X_traces_star : ∀ t, X t = star_shape_path n R O t) :
  ∀ t, distance (X t) O = R * (Real.sec (Real.pi / n) - 1) :=
  sorry

end trajectory_of_X_in_regular_n_gon_l273_273359


namespace hoseok_position_l273_273096

variable (total_people : ℕ) (pos_from_back : ℕ)

theorem hoseok_position (h₁ : total_people = 9) (h₂ : pos_from_back = 5) :
  (total_people - pos_from_back + 1) = 5 :=
by
  sorry

end hoseok_position_l273_273096


namespace expected_white_balls_l273_273114

theorem expected_white_balls (n : ℕ) (h : n > 2) :
  let X := ∑ k in Finset.range n, (n - k - 1) / n in
  X = (n - 1) / 2 := sorry

end expected_white_balls_l273_273114


namespace difference_of_values_l273_273585

theorem difference_of_values :
  let expr1 := (0.85 * 250)^2 / 2.3
  let expr2 := (3/5 * 175) / 2.3
  (expr1 - expr2) = 19587.5 :=
by
  let expr1 := (0.85 * 250)^2 / 2.3
  let expr2 := (3/5 * 175) / 2.3
  have h1 : expr1 = 19633.152173913043
  have h2 : expr2 = 45.65217391304348
  show (expr1 - expr2) = 19587.5
  subst h1
  subst h2
  sorry

end difference_of_values_l273_273585


namespace angle_YTZ_of_circumcenter_l273_273829

theorem angle_YTZ_of_circumcenter (X Y Z T : Type) (T_is_circumcenter : is_circumcenter(X, Y, Z, T))
  (angle_XYZ : angle(X, Y, Z) = 85) (angle_XZY : angle(X, Z, Y) = 49) : 
  angle(Y, T, Z) = 92 := 
by
  sorry

end angle_YTZ_of_circumcenter_l273_273829


namespace true_value_of_product_l273_273220

theorem true_value_of_product (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  let product := (100 * a + 10 * b + c) * (100 * b + 10 * c + a) * (100 * c + 10 * a + b)
  product = 2342355286 → (product % 10 = 6) → product = 328245326 :=
by
  sorry

end true_value_of_product_l273_273220


namespace inscribed_circle_radius_l273_273683

noncomputable def inscribed_radius (XY XZ YZ : ℝ) (s K : ℝ) : ℝ :=
  K / s

noncomputable def semiperimeter (XY XZ YZ : ℝ) : ℝ :=
  (XY + XZ + YZ) / 2

noncomputable def triangle_area (s XY XZ YZ : ℝ) : ℝ :=
  Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))

theorem inscribed_circle_radius (XY XZ YZ : ℝ) (hXY : XY = 8) (hXZ : XZ = 13) (hYZ : YZ = 15) :
  inscribed_radius XY XZ YZ (semiperimeter XY XZ YZ) (triangle_area (semiperimeter XY XZ YZ) XY XZ YZ) = 5 * Real.sqrt 3 / 3 :=
by
  have s : ℝ := semiperimeter XY XZ YZ
  have K : ℝ := triangle_area s XY XZ YZ
  have hs : s = 18 := sorry
  have hK : K = 30 * Real.sqrt 3 := sorry
  show inscribed_radius XY XZ YZ s K = 5 * Real.sqrt 3 / 3
  sorry


end inscribed_circle_radius_l273_273683


namespace valid_functions_count_l273_273401

open Nat

/-- Given correspondences from A to B: 
  - \(A = ℕ\), \(B = {0, 1}\), remainder when elements in \(A\) are divided by 2
  - \(A = {0, 1, 2}\), \(B = {4, 1, 0}\), \(f: x \rightarrow y = x^2\)
  - \(A = {0, 1, 2}\), \(B = {0, 1, \frac{1}{2}}\), \(f: x \rightarrow y = \frac{1}{x}\)
  The number of valid functions from set \(A\) to set \(B\) is 2.
-/

def is_valid_function_1 : ∀ a ∈ ℕ, ∃ b ∈ ({0, 1} : Set ℕ), b = a % 2 := 
sorry

def is_valid_function_2 : ∀ a ∈ ({0, 1, 2} : Set ℕ), ∃ b ∈ ({4, 1, 0} : Set ℕ), b = a^2 :=
sorry

def is_invalid_function_3 : ∀ a ∈ ({0, 1, 2} : Set ℕ), ∃ b ∈ ({0, 1, 1/2} : Set ℝ), true :=
sorry

theorem valid_functions_count : 2 ∈ ({is_valid_function_1, is_valid_function_2} : Set (Set ℕ → Set ℕ → Prop)) ∧
                              ¬ (is_invalid_function_3 : Set (Set ℕ → Set ℝ → Prop)) :=
sorry

end valid_functions_count_l273_273401


namespace triangle_side_lengths_sum_l273_273065

variable (D E F G : ℝ^3)
variable (centroid : G = (D + E + F) / 3)
variable (sum_squares : norm (G - D)^2 + norm (G - E)^2 + norm (G - F)^2 = 90)

theorem triangle_side_lengths_sum : (norm (D - E))^2 + (norm (D - F))^2 + (norm (E - F))^2 = 270 :=
by
  sorry

end triangle_side_lengths_sum_l273_273065


namespace count_linear_equations_is_3_l273_273041

def is_linear_equation (eq : String) : Prop :=
  -- You would define the logic here that matches the equations to check for linearity.
  sorry

def num_linear_equations : List String → Nat :=
  List.countp is_linear_equation

theorem count_linear_equations_is_3 :
  num_linear_equations [
    "3x - y = 2",
    "x + 4/x = 1",
    "x/2 = 1",
    "x = 0",
    "x^2 - 2x - 3 = 0",
    "2x + 1/4 = 1/3"
  ] = 3 :=
  by
    sorry

end count_linear_equations_is_3_l273_273041


namespace find_k_value_l273_273331

-- Define the function f(x)
def f (x : ℝ) : ℝ := (Real.cot (x / 3)) - (Real.cot (3 * x))

-- Define the expression for f(x) in terms of k
def g (x k : ℝ) : ℝ := (Real.sin (k * x)) / ((Real.sin (x / 3)) * (Real.sin (3 * x)))

theorem find_k_value : ∀ x, f(x) = g(x, 8 / 3) := by
  sorry

end find_k_value_l273_273331


namespace extremum_at_one_and_value_at_two_l273_273769

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_one_and_value_at_two (a b : ℝ) (h_deriv : 3 + 2*a + b = 0) (h_value : 1 + a + b + a^2 = 10) : 
  f 2 a b = 18 := 
by 
  sorry

end extremum_at_one_and_value_at_two_l273_273769


namespace potassium_salt_average_molar_mass_l273_273006

noncomputable def average_molar_mass (total_weight : ℕ) (num_moles : ℕ) : ℕ :=
  total_weight / num_moles

theorem potassium_salt_average_molar_mass :
  let total_weight := 672
  let num_moles := 4
  average_molar_mass total_weight num_moles = 168 := by
    sorry

end potassium_salt_average_molar_mass_l273_273006


namespace odd_even_equal_sum_sequence_l273_273427

def equal_sum_sequence (a : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀ n, a(n) + a(n+1) = k

theorem odd_even_equal_sum_sequence (a : ℕ → ℤ) (k : ℤ) :
  equal_sum_sequence a k → (∀ n, a(2*n) = a(0) ∧ a(2*n+1) = a(1)) :=
by
  assume h : equal_sum_sequence a k
  sorry

end odd_even_equal_sum_sequence_l273_273427


namespace modulus_of_z_l273_273012

variable (z : ℂ)
variable (h : z * (1 - complex.i) = 1 + complex.i)

theorem modulus_of_z (h : z * (1 - complex.i) = 1 + complex.i) : |z| = 1 :=
by {
  sorry
}

end modulus_of_z_l273_273012


namespace cumulative_sum_geq_n_l273_273737

theorem cumulative_sum_geq_n
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_cond : ∀ k, 0 < k → a (k + 1) ≥ (k * a k) / (a k ^ 2 + k - 1)) :
  ∀ n, 2 ≤ n → (∑ i in Finset.range n, a (i + 1)) ≥ n :=
by
  intros n hn
  sorry

end cumulative_sum_geq_n_l273_273737


namespace total_games_l273_273567

-- Definitions based on conditions
def num_teams := 20
def num_regular_season_games := num_teams * (num_teams - 1) * 2
def num_playoff_teams := 8

def num_playoff_games : Nat := 
  let quarterfinals := num_playoff_teams / 2
  let semifinals := quarterfinals / 2
  let final := semifinals / 2
  quarterfinals + semifinals + final

-- Main statement to prove
theorem total_games : 
  num_regular_season_games + num_playoff_games = 767 :=
by
  -- Calculate regular season games
  have reg_games : num_regular_season_games = 760 := by
    simp [num_teams, num_regular_season_games]
    norm_num
  -- Calculate playoff games
  have play_games : num_playoff_games = 7 := by
    simp [num_playoff_teams, num_playoff_games]
    norm_num
  -- Combine results
  rw [reg_games, play_games]
  norm_num
  sorry

end total_games_l273_273567


namespace xy_sum_143_l273_273438

theorem xy_sum_143 (x y : ℕ) (h1 : x < 30) (h2 : y < 30) (h3 : x + y + x * y = 143) (h4 : 0 < x) (h5 : 0 < y) :
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 :=
by
  sorry

end xy_sum_143_l273_273438


namespace number_of_cookies_first_friend_took_l273_273494

-- Definitions of given conditions:
def initial_cookies : ℕ := 22
def eaten_by_Kristy : ℕ := 2
def given_to_brother : ℕ := 1
def taken_by_second_friend : ℕ := 5
def taken_by_third_friend : ℕ := 5
def cookies_left : ℕ := 6

noncomputable def cookies_after_Kristy_ate_and_gave_away : ℕ :=
  initial_cookies - eaten_by_Kristy - given_to_brother

noncomputable def cookies_after_second_and_third_friends : ℕ :=
  taken_by_second_friend + taken_by_third_friend

noncomputable def cookies_before_second_and_third_friends_took : ℕ :=
  cookies_left + cookies_after_second_and_third_friends

theorem number_of_cookies_first_friend_took :
  cookies_after_Kristy_ate_and_gave_away - cookies_before_second_and_third_friends_took = 3 := by
  sorry

end number_of_cookies_first_friend_took_l273_273494


namespace no_four_digit_numbers_divisible_by_11_l273_273429

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) :
  (a + b + c + d = 9) ∧ ((a + c) - (b + d)) % 11 = 0 → false :=
by
  sorry

end no_four_digit_numbers_divisible_by_11_l273_273429


namespace isosceles_trapezoid_to_square_l273_273033

theorem isosceles_trapezoid_to_square :
  ∀ (ABCD : Type) (A B C D : ABCD) (BC : ℝ) (AD : ℝ) (angle_ABC : ℝ) (angle_BCD : ℝ),
    is_isosceles_trapezoid ABCD A B C D AD BC ∧
    AD = 3 * BC ∧
    angle_ABC = 45 ∧
    angle_BCD = 45 →
    (can_cut_into_three_parts_form_square ABCD A B C D) :=
by sorry

end isosceles_trapezoid_to_square_l273_273033


namespace find_side_length_of_cyclic_quadrilateral_l273_273529

theorem find_side_length_of_cyclic_quadrilateral
  (AB BC CD : ℝ) (h1 : AB = 4) (h2 : BC = 6) (h3 : CD = 8)
  (ABC_area_eq_ACD_area : ∀ AC : ℝ, (let s1 := (4 + 6 + AC) / 2 in 
                                     sqrt (s1 * (s1 - 4) * (s1 - 6) * (s1 - AC)))
                                = (let s2 := (8 + x + AC) / 2 in 
                                     sqrt (s2 * (s2 - 8) * (s2 - x) * (s2 - AC)))) :
  x = 3 ∨ x = 16 / 3 ∨ x = 12 :=
by sorry

end find_side_length_of_cyclic_quadrilateral_l273_273529


namespace polar_to_rectangular_coordinates_l273_273241

theorem polar_to_rectangular_coordinates
  (x y : ℝ)
  (r θ : ℝ)
  (h1 : x = 7)
  (h2 : y = 6)
  (h3 : r = Real.sqrt (7^2 + 6^2))
  (h4 : Real.cos θ = 7 / r)
  (h5 : Real.sin θ = 6 / r) :
  let r3 := r^3
      θ3 := 3 * θ
      x' := r3 * (4 * (Real.cos θ)^3 - 3 * Real.cos θ)
      y' := r3 * (3 * Real.sin θ - 4 * (Real.sin θ)^3)
  in x' = -413 ∧ y' = 666 :=
by
  sorry

end polar_to_rectangular_coordinates_l273_273241


namespace inverse_function_of_f_l273_273008

noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ 2

noncomputable def f_inv (y : ℝ) : ℝ := 1 - Real.sqrt y

theorem inverse_function_of_f :
  ∀ x, x ≤ 1 → f_inv (f x) = x ∧ ∀ y, 0 ≤ y → f (f_inv y) = y :=
by
  intros
  sorry

end inverse_function_of_f_l273_273008


namespace number_of_divisors_of_2009_squared_l273_273321

theorem number_of_divisors_of_2009_squared (a : ℕ) (h : a = 2009)
  (h2009_factorization : 2009 = 7^2 * 41) :
  num_divisors (2009^6) = 91 :=
sorry

end number_of_divisors_of_2009_squared_l273_273321


namespace selection_options_l273_273963

theorem selection_options (group1 : Fin 5) (group2 : Fin 4) : (group1.1 + group2.1 + 1 = 9) :=
sorry

end selection_options_l273_273963


namespace systematic_sampling_result_l273_273958

-- Define the set of bags numbered from 1 to 30
def bags : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the systematic sampling function
def systematic_sampling (n k interval : ℕ) : List ℕ :=
  List.range k |> List.map (λ i => n + i * interval)

-- Specific parameters for the problem
def number_of_bags := 30
def bags_drawn := 6
def interval := 5
def expected_samples := [2, 7, 12, 17, 22, 27]

-- Statement of the theorem
theorem systematic_sampling_result : 
  systematic_sampling 2 bags_drawn interval = expected_samples :=
by
  sorry

end systematic_sampling_result_l273_273958


namespace minimum_ballots_l273_273691

theorem minimum_ballots (c1 c2 c3 : Nat) (h1 : c1 = 3) (h2 : c2 = 4) (h3 : c3 = 5) : 
  ∃ n, n = 5 ∧ 
       ∀ positions ballots : Nat, 
       positions = 3 ∧
       (ballots = c1 * c2 * c3 / c3 ∧ ballots = c1 * c2 * c3 / c2 ∧ ballots = c1 * c2 * c3 / c1) →
       ballots = n :=
begin
  sorry
end

end minimum_ballots_l273_273691


namespace min_distance_AB_l273_273944

noncomputable def distance_AB (x : ℝ) : ℝ :=
  (1/2) * (x - Real.log x) + 1

theorem min_distance_AB : ∃ x : ℝ, (x > 0) → distance_AB x = 3 / 2 :=
by
  use 1
  intros
  have hx : x = 1 := by linarith
  rw hx
  norm_num
  sorry  -- to be completed with a detailed proof

end min_distance_AB_l273_273944


namespace number_of_friends_l273_273908

-- Define the initial conditions
def rose_apples : Nat := 9
def apples_per_friend : Nat := 3

-- Define the problem statement as a theorem
theorem number_of_friends (h1 : rose_apples = 9) (h2 : apples_per_friend = 3) : (9 / 3 = 3) :=
by
  simp [h1, h2]
  sorry

end number_of_friends_l273_273908


namespace final_price_after_discounts_l273_273630

theorem final_price_after_discounts (m : ℝ) : (0.8 * m - 10) = selling_price :=
by
  sorry

end final_price_after_discounts_l273_273630


namespace triangle_is_isosceles_right_l273_273023

theorem triangle_is_isosceles_right (a b S : ℝ) (h : S = (1/4) * (a^2 + b^2)) :
  ∃ C : ℝ, C = 90 ∧ a = b :=
by
  sorry

end triangle_is_isosceles_right_l273_273023


namespace gym_monthly_revenue_l273_273643

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end gym_monthly_revenue_l273_273643


namespace barker_high_school_team_count_l273_273665

theorem barker_high_school_team_count (students_total : ℕ) (baseball_team : ℕ) (hockey_team : ℕ) 
  (both_sports : ℕ) : 
  students_total = 36 → baseball_team = 25 → hockey_team = 19 → both_sports = (baseball_team + hockey_team - students_total) → both_sports = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end barker_high_school_team_count_l273_273665


namespace max_m_under_conditions_l273_273789

-- Conditions definition
def is_prime (n : ℕ) : Prop :=
  ∀ m ∈ finset.range (n - 2), m + 2 ∣ n → (m + 2 = 1 ∨ m + 2 = n)

-- Specific prime conditions for x, y, z
def distinct_primes (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  is_prime x ∧ is_prime y ∧ is_prime z ∧
  x < 10 ∧ y < 10 ∧ z < 10

-- Condition for additional primes
def valid_combination (x y z : ℕ) : Prop :=
  distinct_primes x y z ∧ 
  is_prime (10 * x + y) ∧ is_prime (10 * z + x)

-- The number m and the proof of maximum
def m (x y z : ℕ) : ℕ := x * y * (10 * x + y)

theorem max_m_under_conditions : ∃ x y z : ℕ, 
  valid_combination x y z ∧ m x y z = 1533 := 
sorry

end max_m_under_conditions_l273_273789


namespace find_b_l273_273763

theorem find_b
  (b : ℝ)
  (f : ℝ → ℝ)
  (h1: ∀ x, (x < 1) → f x = 3 * x - b)
  (h2: ∀ x, (x ≥ 1) → f x = 2 ^ x)
  (h3: f (f (5 / 6)) = 4) :
  b = 11 / 8 :=
begin
  sorry
end

end find_b_l273_273763


namespace single_intersection_point_l273_273336

theorem single_intersection_point (k : ℝ) :
  (∃! x : ℝ, x^2 - 2 * x - k = 0) ↔ k = 0 :=
by
  sorry

end single_intersection_point_l273_273336


namespace exists_prime_for_digit_sum_mod_infinitely_many_n_for_prime_l273_273677

def sum_of_digits_in_base (n r : ℕ) : ℕ :=
  -- This is a placeholder for the actual function that calculates the sum of digits in base r.
  sorry

-- Formalizing the first part of the proof:
theorem exists_prime_for_digit_sum_mod (r : ℕ) (hr2 : r > 2) :
  ∃ p, (nat.prime p) ∧ ∀ n (hn : n > 0), sum_of_digits_in_base n r ≡ n [MOD p] :=
sorry

-- Formalizing the second part of the proof:
theorem infinitely_many_n_for_prime (r : ℕ) (hr1 : r > 1) (p : ℕ) (hp : nat.prime p) :
  ∃ᶠ n in filter.at_top (λ n, n > 0), sum_of_digits_in_base n r ≡ n [MOD p] :=
sorry

end exists_prime_for_digit_sum_mod_infinitely_many_n_for_prime_l273_273677


namespace sum_of_distances_l273_273047

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Math.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem sum_of_distances (A B C P : ℝ × ℝ) (hA : A = (0, 0)) (hB : B = (8, 0))
  (hC : C = (4, 6)) (hP : P = (5, 3)) :
  let AP := distance A P
  let BP := distance B P
  let CP := distance C P
  AP + BP + CP = Real.sqrt 34 + 3 * Real.sqrt 2 + Real.sqrt 10 ∧
  m + n = 4 :=
by
  sorry

end sum_of_distances_l273_273047


namespace range_of_a_l273_273803

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > -1 ∧ x ≤ 1 ∧ 2 ^ x > a) ↔ a ∈ set.Iio 2 :=
sorry

end range_of_a_l273_273803


namespace classroom_trash_l273_273965

variable (t o c : ℕ)

-- Given conditions
def total_trash := t = 1576
def outside_trash := o = 1232

-- Prove the correct answer
theorem classroom_trash (h1 : total_trash t) (h2 : outside_trash o) : c = t - o := by
  have ht : t = 1576 := h1
  have ho : o = 1232 := h2
  rw [ht, ho]
  exact eq.refl 344

end classroom_trash_l273_273965


namespace fx_is_quadratic_pa_is_linear_l273_273999

theorem fx_is_quadratic (a : ℝ) (x : ℝ) (h : a ≠ 0) : ∃ (k : ℝ), (f : ℝ → ℝ) = λ x, k * x^2 :=
by {
    rw [lambda_1.kotlin.print(f // define f as a function f(x) = a x^2)] -- translating the mathematical expression for f(x)
    assume h -- introduce the assumptions
    use a -- instantiate the constant k as a
    exact λ x, a * x ^ 2 -- assert the equality for f(x)
    sorry -- skip the proof steps
}

theorem pa_is_linear (x : ℝ) (a : ℝ) (h : x ≠ 0) : ∃ (k : ℝ), (p : ℝ → ℝ) = λ a, k * a :=
by {
    rw [lambda_2.kotlin.print(p // define p as a function p(a) = a x^2)] -- translating the mathematical expression for p(a)
    assume h -- introduce the assumptions
    use x^2 -- instantiate the constant k as x^2
    exact λ a, x^2 * a -- assert the equality for p(a)
    sorry -- skip the proof steps
}

end fx_is_quadratic_pa_is_linear_l273_273999


namespace BM_plus_CM_greater_than_AY_l273_273971

open_locale euclidean_geometry
open real

noncomputable theory

variables {A B C X Y M : euclidean_geometry.Point} 
          {circ : euclidean_geometry.Circle} 
          {triangle : euclidean_geometry.Triangle}

-- Given Conditions
def given_conditions : Prop :=
  circ.contains A ∧ circ.contains B ∧ circ.contains C ∧ circ.contains X ∧ circ.contains Y ∧
  ¬circ.contains_interior A ∧
  angle A B X = angle A C Y ∧
  M = midpoint A X

-- The theorem statement
theorem BM_plus_CM_greater_than_AY (h : given_conditions) : dist B M + dist C M > dist A Y :=
sorry

end BM_plus_CM_greater_than_AY_l273_273971


namespace percent_decrease_of_square_area_l273_273456

-- Definitions based on conditions provided.
def area_triangle_I : ℝ := 50 * Real.sqrt 3
def area_triangle_III : ℝ := 18 * Real.sqrt 3
def area_square_II : ℝ := 50
def length_AD_decrease : ℝ := 0.15

-- Calculations mentioned in the solution are encapsulated in the proof that follows.
noncomputable def percent_decrease_area_square : ℝ := 
  let original_side := Real.sqrt area_square_II
  let new_side := original_side * (1 - length_AD_decrease)
  let new_area := new_side^2
  let percent_decrease := ((area_square_II - new_area) / area_square_II) * 100
  percent_decrease

theorem percent_decrease_of_square_area :
  percent_decrease_area_square = 27.75 := by
  sorry

end percent_decrease_of_square_area_l273_273456


namespace binomial_expansion_b_value_l273_273346

theorem binomial_expansion_b_value (a b x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x ^ 2 + a^5 * x ^ 5) : b = 40 := 
sorry

end binomial_expansion_b_value_l273_273346


namespace parallelogram_line_intersection_l273_273510

variables (A B C D P Q R : Type*) [add_comm_group Type*]

-- Assume A, B, C, D are vertices of a parallelogram with AC as diagonal
axiom parallelogram (A B C D : Type*) :
  (segment A C) = (segment B D)

-- Assume lines AB, AC, AD are intersected by a line at P, Q, R respectively
axiom intersections (A B C D P Q R : Type*) :
  ∃ l, (P ∈ l ∧ Q ∈ l ∧ R ∈ l) ∧ 
  (∃ l1, l1 = (segment A B) ∧ P ∈ l1) ∧
  (∃ l2, l2 = (segment A C) ∧ Q ∈ l2) ∧
  (∃ l3, l3 = (segment A D) ∧ R ∈ l3)

-- Let AB, AP, AD, AR, AC, AQ be the directed segments
noncomputable def AB := segment A B
noncomputable def AP := segment A P
noncomputable def AD := segment A D
noncomputable def AR := segment A R
noncomputable def AC := segment A C
noncomputable def AQ := segment A Q

theorem parallelogram_line_intersection :
  (\frac{AB}{AP} + \frac{AD}{AR} = \frac{AC}{AQ}) :=
by sorry

end parallelogram_line_intersection_l273_273510


namespace gym_monthly_income_l273_273641

theorem gym_monthly_income (bi_monthly_charge : ℕ) (members : ℕ) (monthly_income : ℕ) 
  (h1 : bi_monthly_charge = 18)
  (h2 : members = 300)
  (h3 : monthly_income = 10800) : 
  2 * bi_monthly_charge * members = monthly_income :=
by
  rw [h1, h2, h3]
  norm_num

end gym_monthly_income_l273_273641


namespace greatest_integer_not_exceeding_l273_273749

variables {x y a : ℝ}

theorem greatest_integer_not_exceeding (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : 
    ⌊x⌋ ≥ ⌊y⌋ := 
begin
  have hx_gt_y : x > y,
  from (real.rpow_lt_rpow_iff (lt_trans h1 h2) (lt_trans h1 zero_lt_one)).mp h3,

  exact int.floor_le_floor hx_gt_y,
end

end greatest_integer_not_exceeding_l273_273749


namespace infinite_n_for_irrational_x_l273_273688

noncomputable def irrational : Set ℝ := { x | ¬(∃ r : ℚ, x = r)}

theorem infinite_n_for_irrational_x (x : ℝ) (h1 : x ∈ irrational) : 
  ∃ infinitely_many n : ℕ, ∀ k : ℕ, (1 ≤ k ∧ k ≤ n) → (1 ≤ fract (↑k * x) ∧ fract (↑k * x) ≥ 1 / (n + 1)) :=
sorry

end infinite_n_for_irrational_x_l273_273688


namespace slope_perpendicular_l273_273186

theorem slope_perpendicular (x1 y1 x2 y2 m : ℚ) 
  (hx1 : x1 = 3) (hy1 : y1 = -4) (hx2 : x2 = -6) (hy2 : y2 = 2) 
  (hm : m = (y2 - y1) / (x2 - x1)) :
  ∀ m_perpendicular: ℚ, m_perpendicular = (-1 / m) → m_perpendicular = 3/2 := 
sorry

end slope_perpendicular_l273_273186


namespace number_of_small_cakes_needed_l273_273277

noncomputable def large_cakes_per_helper_per_hour := 2
noncomputable def small_cakes_per_helper_per_hour := 35
noncomputable def hours_available := 3
noncomputable def large_cakes_needed := 20
noncomputable def total_helpers := 10

theorem number_of_small_cakes_needed : 
  total_helpers * hours_available * small_cakes_per_helper_per_hour - 
  (large_cakes_needed / (large_cakes_per_helper_per_hour * hours_available)).ceil * (hours_available * small_cakes_per_helper_per_hour) = 630 :=
by
  sorry

end number_of_small_cakes_needed_l273_273277


namespace no_divisible_sum_prob_l273_273495

theorem no_divisible_sum_prob (k : ℕ) (hk : k > 0) :
  let total_arrangements := (3 * k + 1)!,
      valid_arrangements := (k + 1)! * k! * k! / (2 * k)! / (3 * k + 1) in
  valid_arrangements / total_arrangements = (k + 1)! * k! / ((2 * k)! * (3 * k + 1)) :=
by
  sorry

end no_divisible_sum_prob_l273_273495


namespace oak_trees_remaining_is_7_l273_273960

-- Define the number of oak trees initially in the park
def initial_oak_trees : ℕ := 9

-- Define the number of oak trees cut down by workers
def oak_trees_cut_down : ℕ := 2

-- Define the remaining oak trees calculation
def remaining_oak_trees : ℕ := initial_oak_trees - oak_trees_cut_down

-- Prove that the remaining oak trees is equal to 7
theorem oak_trees_remaining_is_7 : remaining_oak_trees = 7 := by
  sorry

end oak_trees_remaining_is_7_l273_273960


namespace sum_of_cubics_l273_273077

noncomputable def polynomial_with_roots (p : Polynomial ℝ) (a b c : ℝ) : Prop :=
  p = Polynomial.C 5 * X^3 + Polynomial.C 500 * X + Polynomial.C 3005 ∧
  Polynomial.root p a ∧ Polynomial.root p b ∧ Polynomial.root p c

theorem sum_of_cubics (a b c : ℝ) :
  (polynomial_with_roots (Polynomial.C 5 * X^3 + Polynomial.C 500 * X + Polynomial.C 3005) a b c) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 :=
by
  intro h
  sorry

end sum_of_cubics_l273_273077


namespace complex_trajectory_is_ellipse_l273_273135

open Complex

theorem complex_trajectory_is_ellipse (z : ℂ) (h : abs (z - i) + abs (z + i) = 3) : 
  true := 
sorry

end complex_trajectory_is_ellipse_l273_273135


namespace largest_digit_divisible_by_6_l273_273988

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273988


namespace sum_of_faces_edges_vertices_l273_273592

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end sum_of_faces_edges_vertices_l273_273592


namespace cevian_concurrency_l273_273631

theorem cevian_concurrency 
  {A B C D1 D2 E1 E2 F1 F2 L M N : Point}
  (h_circle : Circle Intersects Sides ABC At D1 D2 E1 E2 F1 F2 In_Order)
  (h_ld : Lines D1E1 And D2F2 Intersect At L)
  (h_mn : Lines E1F1 And E2D2 Intersect At M)
  (h_nn : Lines F1D1 And F2E2 Intersect At N) :
  Concurrent {Line A L, Line B M, Line C N} :=
  sorry

end cevian_concurrency_l273_273631


namespace cube_faces_edges_vertices_sum_l273_273596

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end cube_faces_edges_vertices_sum_l273_273596


namespace distinct_values_in_list_l273_273320

noncomputable def count_distinct_values : ℕ :=
  let list_terms := list.map (λ (n : ℕ), (n^2 / 1000 : ℤ)) (list.range' 1 1000)
  list.nodup list_terms

theorem distinct_values_in_list : count_distinct_values = 751 := by
  sorry

end distinct_values_in_list_l273_273320


namespace f_2019_value_l273_273080

noncomputable def f : ℕ → ℕ := sorry

theorem f_2019_value
  (h : ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) :
  f 2019 = 2019 :=
sorry

end f_2019_value_l273_273080


namespace multiple_of_students_in_restroom_l273_273291

theorem multiple_of_students_in_restroom 
    (num_desks_per_row : ℕ)
    (num_rows : ℕ)
    (desk_fill_fraction : ℚ)
    (total_students : ℕ)
    (students_restroom : ℕ)
    (absent_students : ℕ)
    (m : ℕ) :
    num_desks_per_row = 6 →
    num_rows = 4 →
    desk_fill_fraction = 2 / 3 →
    total_students = 23 →
    students_restroom = 2 →
    (num_rows * num_desks_per_row : ℕ) * desk_fill_fraction = 16 →
    (16 - students_restroom) = 14 →
    total_students - 14 - 2 = absent_students →
    absent_students = 7 →
    2 * m - 1 = 7 →
    m = 4
:= by
    intros;
    sorry

end multiple_of_students_in_restroom_l273_273291


namespace sum_reciprocal_geometric_mean_l273_273009

theorem sum_reciprocal_geometric_mean (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i ∧ a i < 1) :
    ∑ i in Finset.range n, 1 / (1 - a i) ≥ n / (1 - (Finset.prod (Finset.range n) (λ i, a i))^(1 / n : ℝ)) := 
sorry

end sum_reciprocal_geometric_mean_l273_273009


namespace find_second_number_l273_273627

theorem find_second_number (x : ℝ) :
  3 + x + 333 + 3.33 = 369.63 → x = 30.3 :=
by
  intro h
  calc
    x = 369.63 - 3 - 333 - 3.33 : by linarith
      ... = 30.3 : by norm_num

end find_second_number_l273_273627


namespace rice_thickness_estimation_l273_273606
open Real

noncomputable def rice_grains_sum : ℕ := 2^64 - 1
noncomputable def grain_volume : ℝ := 1 / 10^7
noncomputable def arable_land_area : ℝ := 1.5 * 10^13
noncomputable def lg2 : ℝ := 0.30
noncomputable def lg3 : ℝ := 0.48

theorem rice_thickness_estimation :
  let V := (rice_grains_sum * grain_volume : ℝ)
  let h := V / arable_land_area
  64 * lg2 - 7 - lg3 - 13 ≈ -1 →
  h ≈ 0.1 := 
sorry

end rice_thickness_estimation_l273_273606


namespace triangle_inequality_l273_273083

theorem triangle_inequality
  (a b c x y z : ℝ)
  (h_order : a < b ∧ b < c ∧ 0 < x)
  (h_area_eq : c * x = a * y + b * z) :
  x < y + z :=
by
  sorry

end triangle_inequality_l273_273083


namespace polygon_sides_l273_273952

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 1260) : ∃ n : ℕ, (n-2) * 180 = sum_of_interior_angles ∧ n = 9 :=
by {
  sorry
}

end polygon_sides_l273_273952


namespace interval_of_monotonic_increase_l273_273013

open Real

theorem interval_of_monotonic_increase (k : ℤ) :
  let f : ℝ → ℝ := λ x, sin (2 * x + π / 6) + cos (2 * x - π / 3)
  in ∀ x, (k * π - π / 3 < x ∧ x < k * π + π / 6) ↔
           ∀ y, (x < y) → f(x) < f(y) :=
sorry

end interval_of_monotonic_increase_l273_273013


namespace find_dot_product_l273_273386

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
axiom cos_theta : real.cos (angle a b) = 1 / 3
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 3

-- Proof Goal
theorem find_dot_product : (2 • a + b) ⬝ b = 11 := sorry

end find_dot_product_l273_273386


namespace imaginary_part_of_conjugate_l273_273375

theorem imaginary_part_of_conjugate (i : Complex.Im) : 
  Complex.imag (Complex.conj ((2 : ℂ) + i).pow 2) = -4 := by
sorry

end imaginary_part_of_conjugate_l273_273375


namespace complex_fraction_squared_l273_273374

noncomputable def imaginary_unit : ℂ := Complex.i

theorem complex_fraction_squared (m n : ℝ) (h : m * (1 + imaginary_unit) = 1 + n * imaginary_unit) :
  ( (m + n * imaginary_unit) / (m - n * imaginary_unit) )^2 = -1 := 
by 
  sorry

end complex_fraction_squared_l273_273374


namespace unico_fn_l273_273319

noncomputable def find_fn : ℕ → ℕ := sorry

theorem unico_fn (f : ℕ → ℕ) (h1 : f 1 > 0)
  (h2 : ∀ m n : ℕ, f (m^2 + n^2) = f m ^ 2 + f n ^ 2) :
  ∀ n : ℕ, f n = n :=
begin
  -- Proof here
  sorry
end

end unico_fn_l273_273319


namespace andy_cavity_per_candy_cane_l273_273664

theorem andy_cavity_per_candy_cane 
  (cavities_per_candy_cane : ℝ)
  (candy_caned_from_parents : ℝ := 2)
  (candy_caned_each_teacher : ℝ := 3)
  (num_teachers : ℝ := 4)
  (allowance_factor : ℝ := 1/7)
  (total_cavities : ℝ := 16) :
  let total_given_candy : ℝ := candy_caned_from_parents + candy_caned_each_teacher * num_teachers
  let total_bought_candy : ℝ := allowance_factor * total_given_candy
  let total_candy : ℝ := total_given_candy + total_bought_candy
  total_candy / total_cavities = cavities_per_candy_cane :=
by
  sorry

end andy_cavity_per_candy_cane_l273_273664


namespace slope_of_tangent_line_at_1_1_l273_273699

noncomputable def f (x : ℝ) : ℝ := x * exp (x - 1)

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem slope_of_tangent_line_at_1_1 : f' 1 = 2 :=
by
  sorry

end slope_of_tangent_line_at_1_1_l273_273699


namespace triangle_area_approx_l273_273263

noncomputable def radius := 5
noncomputable def ratio_side1 := 5
noncomputable def ratio_side2 := 12
noncomputable def ratio_side3 := 13
noncomputable def hypotenuse := 2 * radius

theorem triangle_area_approx : ∃ (area : ℝ), 
  (∀ x : ℝ, 
    13 * x = hypotenuse → 
    area = (1/2) * (5 * x) * (12 * x)
  ) ∧ 
  abs(area - 17.75) < 0.01 :=
by
  sorry

end triangle_area_approx_l273_273263


namespace angle_calculation_correct_l273_273468

def length_of_side : ℝ := 2
def angle_P_O_Q : ℝ := 26.6
def measure_angle_POQ (l: ℝ) : ℝ := by 
  let OR := 4
  let QS := 2
  let OS := 6 
  let angle_QOS := Real.arctan (1 / 3)
  let angle_P_O_R := 45 
  let angle_POQ := angle_P_O_R - angle_QOS.toDegrees
  exact angle_POQ
  
theorem angle_calculation_correct : measure_angle_POQ length_of_side = angle_P_O_Q := by
  sorry

end angle_calculation_correct_l273_273468


namespace smallest_sum_proof_l273_273157

theorem smallest_sum_proof (N : ℕ) (p : ℝ) (h1 : 6 * N = 2022) (hp : p > 0) : (N * 1 = 337) :=
by 
  have hN : N = 2022 / 6 := by 
    sorry
  exact hN

end smallest_sum_proof_l273_273157


namespace ac_length_l273_273466

def ab := 15
def dc := 26
def ad := 9

def ac (ab dc ad : ℝ) : ℝ :=
  let bd := Math.sqrt (ab^2 - ad^2)
  let bc := Math.sqrt (dc^2 - bd^2)
  let de := bc
  let ce := bd
  let ae := ad + de
  Math.sqrt (ae^2 + ce^2)

theorem ac_length (ab dc ad : ℝ) (h_ab : ab = 15) (h_dc : dc = 26) (h_ad : ad = 9) :
  ac ab dc ad ≈ 34.2 :=
by
  sorry

end ac_length_l273_273466


namespace length_of_MN_l273_273648

theorem length_of_MN (b : ℝ) (h_focus : ∃ b : ℝ, (3/2, b).1 > 0 ∧ (3/2, b).2 * (3/2, b).2 = 6 * (3 / 2)) : 
  |2 * b| = 6 :=
by sorry

end length_of_MN_l273_273648


namespace sin_pi_minus_alpha_l273_273203

theorem sin_pi_minus_alpha (α : ℝ) (h1 : cos α = (sqrt 5) / 3) (h2 : α ∈ Ioo (-π/2) 0) : 
  sin (π - α) = -2 / 3 :=
sorry

end sin_pi_minus_alpha_l273_273203


namespace part_I_part_II_l273_273360

variable {a : ℕ → ℝ} (S : ℕ → ℝ) (λ : ℝ)

-- Conditions as definitions
def a₁ : Prop := a 1 = 1
def a_ne_zero : Prop := ∀ n, a n ≠ 0
def sum_relation : Prop := ∀ n, a n * a (n + 1) = λ * S n - 1

-- Questions in Lean statements
theorem part_I (hₐ₁ : a₁) (hₐ_ne_zero : a_ne_zero) (h_sum_relation : sum_relation) 
  : ∀ n, a (n + 2) - a n = λ := 
sorry

theorem part_II : 
  ∃ λ = (4 : ℝ), ∀ n, a (n + 2) - a n = 2 * (a (n + 1) - a n) := 
sorry

end part_I_part_II_l273_273360


namespace james_ride_time_l273_273838

theorem james_ride_time :
  let distance := 80 
  let speed := 16 
  distance / speed = 5 := 
by
  -- sorry to skip the proof
  sorry

end james_ride_time_l273_273838


namespace drying_time_short_haired_dog_l273_273051

theorem drying_time_short_haired_dog (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : 6 * x + 9 * (2 * x) = 240) : x = 10 :=
by
  sorry

end drying_time_short_haired_dog_l273_273051


namespace find_radius_of_circumcircle_of_ABC_l273_273171

-- Definitions based on given conditions
variable (A B C : Point)
variable (r1 r2 : ℝ)
variable (O1 O2 : Point)
variable (r3 := 8) -- The radius of the third sphere

-- Given Conditions
-- 1. r1 + r2 = 7
def condition1 : r1 + r2 = 7 := sorry

-- 2. The distance between the centers of the two spheres
def condition2 : dist O1 O2 = 17 := sorry

-- 3. The distance from the point A to the centers O1 and O2
def condition3 : dist A O1 = r1 + r3 ∧ dist A O2 = r2 + r3 := sorry

-- Target statement
theorem find_radius_of_circumcircle_of_ABC :
  r1 + r2 = 7 → dist O1 O2 = 17 → (dist A O1 = r1 + 8) ∧ (dist A O2 = r2 + 8) → 
  circumradius A B C = 2 * sqrt 15 :=
sorry

end find_radius_of_circumcircle_of_ABC_l273_273171


namespace dot_product_u_v_eq_neg61_l273_273292

def u : ℝ × ℝ × ℝ := (-5, 2, -3)
def v : ℝ × ℝ × ℝ := (7, -4, 6)

theorem dot_product_u_v_eq_neg61 : (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) = -61 := by
  -- step through the calculation as outlined in the problem
  sorry

end dot_product_u_v_eq_neg61_l273_273292


namespace athlete_weight_l273_273084

theorem athlete_weight (a b c : ℤ) (k₁ k₂ k₃ : ℤ)
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : a = 5 * k₁)
  (h5 : b = 5 * k₂)
  (h6 : c = 5 * k₃) :
  b = 40 :=
by
  sorry

end athlete_weight_l273_273084


namespace min_value_two_inv_expressions_l273_273790

theorem min_value_two_inv_expressions (x y : ℝ) (h : 2^x + 2^y = 5) : 
  2⁻¹^x + 2⁻¹^y ≥ (4 / 5) ∧ (∃ z, z = 2^{-x} + 2^{-y} ∧ z = 4 / 5 ↔ x = y) :=
by sorry

end min_value_two_inv_expressions_l273_273790


namespace range_of_values_includes_one_integer_l273_273715

theorem range_of_values_includes_one_integer (x : ℝ) (h : -1 < 2 * x + 3 ∧ 2 * x + 3 < 1) :
  ∃! n : ℤ, -7 < (2 * x - 3) ∧ (2 * x - 3) < -5 ∧ n = -6 :=
sorry

end range_of_values_includes_one_integer_l273_273715


namespace joe_spent_255_minutes_l273_273835

-- Define the time taken to cut hair for women, men, and children
def time_per_woman : Nat := 50
def time_per_man : Nat := 15
def time_per_child : Nat := 25

-- Define the number of haircuts for each category
def women_haircuts : Nat := 3
def men_haircuts : Nat := 2
def children_haircuts : Nat := 3

-- Compute the total time spent cutting hair
def total_time_spent : Nat :=
  (women_haircuts * time_per_woman) +
  (men_haircuts * time_per_man) +
  (children_haircuts * time_per_child)

-- The theorem stating the total time spent is equal to 255 minutes
theorem joe_spent_255_minutes : total_time_spent = 255 := by
  sorry

end joe_spent_255_minutes_l273_273835


namespace find_S_9_l273_273463

variable (a : ℕ → ℝ)

def arithmetic_sum_9 (S_9 : ℝ) : Prop :=
  (a 1 + a 3 + a 5 = 39) ∧ (a 5 + a 7 + a 9 = 27) ∧ (S_9 = (9 * (a 3 + a 7)) / 2)

theorem find_S_9 
  (h1 : a 1 + a 3 + a 5 = 39)
  (h2 : a 5 + a 7 + a 9 = 27) :
  ∃ S_9, arithmetic_sum_9 a S_9 ∧ S_9 = 99 := 
by
  sorry

end find_S_9_l273_273463


namespace num_tagged_fish_caught_second_time_is_two_l273_273454

theorem num_tagged_fish_caught_second_time_is_two
  (num_first_catch_tagged : ℕ)
  (num_second_catch : ℕ)
  (num_pond : ℕ)
  (approx_equal_percentages : num_first_catch_tagged / num_pond ≈ 50 / num_second_catch ) :
  num_tagged_fish_second_catch = 2 :=
sorry

end num_tagged_fish_caught_second_time_is_two_l273_273454


namespace variance_eta_l273_273357

noncomputable def xi : ℝ := sorry -- Define ξ as a real number (will be specified later)
noncomputable def eta : ℝ := sorry -- Define η as a real number (will be specified later)

-- Conditions
axiom xi_distribution : xi = 3 + 2*Real.sqrt 4 -- ξ follows a normal distribution with mean 3 and variance 4
axiom relationship : xi = 2*eta + 3 -- Given relationship between ξ and η

-- Theorem to prove the question
theorem variance_eta : sorry := sorry

end variance_eta_l273_273357


namespace num_books_second_shop_l273_273105

-- Define the conditions
def num_books_first_shop : ℕ := 32
def cost_first_shop : ℕ := 1500
def cost_second_shop : ℕ := 340
def avg_price_per_book : ℕ := 20

-- Define the proof statement
theorem num_books_second_shop : 
  (num_books_first_shop + (cost_second_shop + cost_first_shop) / avg_price_per_book) - num_books_first_shop = 60 := by
  sorry

end num_books_second_shop_l273_273105


namespace find_c_value_l273_273234

theorem find_c_value (a b c : ℝ) (h_vertex : ∀ x, y = a * x ^ 2 + b * x + c) (h_vertex_point : (3, -5) ∈ set_of_locus (y = a * x ^2 + b * x + c)) (h_point : (5, -3) ∈ set_of_locus (y = a * x ^ 2 + b * x + c)) :
    c = -1 / 2 :=
by 
  sorry

end find_c_value_l273_273234


namespace nicolai_peaches_6_pounds_l273_273570

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end nicolai_peaches_6_pounds_l273_273570


namespace find_k_value_l273_273329

theorem find_k_value (x : ℝ) (h : x ≠ 0) : 
  (f : ℝ → ℝ) := λ x, (real.cot (x / 3) - real.cot (3 * x)) →
  f x = (real.sin (8 * x / 3)) / (real.sin (x / 3) * real.sin (3 * x)) :=
begin
  sorry
end

end find_k_value_l273_273329


namespace find_A_l273_273190

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 10 * A + 3 + 610 + B = 695) : A = 8 :=
by {
  sorry
}

end find_A_l273_273190


namespace eccentricity_of_ellipse_l273_273693

-- Define the conditions of the ellipse and its properties
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the points P and Q
def symmetric_y_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = Q.2

-- Define the slopes of lines AP and AQ
def slope (A P : ℝ × ℝ) : ℝ :=
  (P.2 - A.2) / (P.1 - A.1)

-- Define the product of slopes condition
def product_of_slopes (A P Q : ℝ × ℝ) (k : ℝ) : Prop :=
  slope A P * slope A Q = k

-- Define the eccentricity of the ellipse
def eccentricity (a b : ℝ) : ℝ :=
  Math.sqrt (1 - (b^2 / a^2))

-- The final theorem
theorem eccentricity_of_ellipse (a b : ℝ) (P Q : ℝ × ℝ)
  (h_ellipse_P : ellipse P.1 P.2 a b)
  (h_ellipse_Q : ellipse Q.1 Q.2 a b)
  (h_symmetric : symmetric_y_axis P Q)
  (h_slopes : product_of_slopes (-(a, 0)) P Q (1/4))
  (h_pos : 0 < b ∧ b < a) :
  eccentricity a b = Math.sqrt 3 / 2 :=
begin
  sorry
end

end eccentricity_of_ellipse_l273_273693


namespace quadrilateral_contains_points_l273_273831

-- Definition of a convex pentagon and two points inside it
variables {Point : Type} [AffineSpace Point] (A B C D E P Q : Point)

-- Assume the pentagon is convex and P, Q are inside the pentagon
def convex_pentagon (A B C D E : Point) : Prop := 
  -- detailed definition of convexity of pentagon should be here 
  sorry 

def inside_pentagon (A B C D E P Q : Point) : Prop :=
  -- detailed definition of P, Q being inside the pentagon should be here
  sorry 

-- Lean statement for the proof problem
theorem quadrilateral_contains_points (h_pentagon : convex_pentagon A B C D E)
    (h_inside : inside_pentagon A B C D E P Q) : 
    ∃ (W X Y Z : Point), W ≠ X ∧ X ≠ Y ∧ Y ≠ Z ∧ Z ≠ W ∧
    (W = A ∨ W = B ∨ W = C ∨ W = D ∨ W = E) ∧
    (X = A ∨ X = B ∨ X = C ∨ X = D ∨ X = E) ∧
    (Y = A ∨ Y = B ∨ Y = C ∨ Y = D ∨ Y = E) ∧
    (Z = A ∨ Z = B ∨ Z = C ∨ Z = D ∨ Z = E) ∧
    -- P and Q are inside the quadrilateral formed by W, X, Y, Z
    inside_quadrilateral W X Y Z P Q :=
sorry

end quadrilateral_contains_points_l273_273831


namespace smallest_rectangles_required_l273_273997

theorem smallest_rectangles_required :
  ∀ (r h : ℕ) (area_square length_square : ℕ),
  r = 3 → h = 4 →
  (∀ k, (k: ℕ) ∣ (r * h) → (k: ℕ) = r * h) →
  length_square = 12 →
  area_square = length_square * length_square →
  (area_square / (r * h) = 12) :=
by
  intros
  /- The mathematical proof steps will be filled here -/
  sorry

end smallest_rectangles_required_l273_273997


namespace farmer_cows_after_selling_l273_273226

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end farmer_cows_after_selling_l273_273226


namespace equation_solutions_l273_273679

def greatest_integer_less_or_equal (x : Real) : Int := ⌊x⌋

theorem equation_solutions (x : Real) (hx : x ≠ 0.5) :
  greatest_integer_less_or_equal x - 
  Real.sqrt (greatest_integer_less_or_equal x / (x - 0.5)) - 
  6 / (x - 0.5) = 0 ↔ x = 3.5 ∨ x = -1.5 := 
sorry

end equation_solutions_l273_273679


namespace literacy_rate_of_females_l273_273097

theorem literacy_rate_of_females (total_population adult_males adult_females children : ℕ)
  (pct_males pct_females pct_children : ℚ)
  (literate_adult_males pct_literate_adult_males : ℚ)
  (literate_adult_females pct_literate_adult_females : ℚ)
  (pct_literate_children : ℚ) :
  total_population = 3500 →
  pct_males = 0.60 →
  pct_females = 0.35 →
  pct_children = 0.05 →
  literate_adult_males = 0.55 →
  literate_adult_females = 0.80 →
  pct_literate_children = 0.95 →
  adult_males = (total_population * pct_males).to_nat →
  adult_females = (total_population * pct_females).to_nat →
  children = (total_population * pct_children).to_nat →
  (adult_females + (children / 2)) ≠ 0 →
  (((literate_adult_females * adult_females) + (pct_literate_children * (children / 2))) / 
  (adult_females + (children / 2)) * 100 ≈ 81) :=
by {
  sorry
}

end literacy_rate_of_females_l273_273097


namespace pencils_count_l273_273141

theorem pencils_count (initial_pencils additional_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : additional_pencils = 100) : initial_pencils + additional_pencils = 215 :=
by sorry

end pencils_count_l273_273141


namespace max_expression_value_l273_273068

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a, b, c with their respective norms
variables (a b c : V)
variable (habc : ∥a∥ = 2 ∧ ∥b∥ = 3 ∧ ∥c∥ = 4)

-- Expression to be maximized
def expression (a b c : V) : ℝ :=
  ∥a - 3 • b∥^2 + ∥b - 3 • c∥^2 + ∥c - 3 • a∥^2 + 2 * ∥a + b - c∥^2

-- Statement of the theorem
theorem max_expression_value : expression a b c = 290 :=
by sorry

end max_expression_value_l273_273068


namespace quadrilateral_parallelogram_l273_273088

theorem quadrilateral_parallelogram
  (k : Type*) [metric_space k] [proper_space k] [NormedAddGroup k] [NormedSpace ℝ k]
  (O A B C D K M N : k) 
  (circ_k : circle k O) 
  (chord_AC : chord k A C)
  (chord_BD : chord k B D)
  (intersect_K : K ∈ chord_AC.pts ∧ K ∈ chord_BD.pts) 
  (circumcenter_M : circumcenter k A K B M) 
  (circumcenter_N : circumcenter k C K D N) : 
  parallelogram k O M K N :=
sorry

end quadrilateral_parallelogram_l273_273088


namespace Radhika_total_games_l273_273532

theorem Radhika_total_games :
  let christmas_gifts := 12
      birthday_gifts := 8
      total_gifts := christmas_gifts + birthday_gifts
      previously_owned := (1 / 2) * total_gifts
  in previously_owned + total_gifts = 30 :=
by
  let christmas_gifts := 12
  let birthday_gifts := 8
  let total_gifts := christmas_gifts + birthday_gifts
  let previously_owned := (1 / 2) * total_gifts
  show previously_owned + total_gifts = 30
  sorry

end Radhika_total_games_l273_273532


namespace smallest_sum_symmetrical_dice_l273_273149

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end smallest_sum_symmetrical_dice_l273_273149


namespace integral_f_equals_23_over_4_max_min_f_on_interval_l273_273403

noncomputable def f (x : ℝ) : ℝ := -x^3 + 12 * x

def integral_value : ℝ := ∫ x in 0..1, f x

-- Statement (1) 
theorem integral_f_equals_23_over_4 : integral_value = 23 / 4 :=
sorry

-- Statement (2)
theorem max_min_f_on_interval : 
  (∀ x ∈ set.Icc. -3 1, f x ≤ 11) ∧ 
  (∀ x ∈ set.Icc -3 1, -16 ≤ f x) :=
sorry

end integral_f_equals_23_over_4_max_min_f_on_interval_l273_273403


namespace threeEdgesPerVertexCount_l273_273786

-- Define the regular polyhedra and their properties
inductive RegularPolyhedra
| tetrahedron
| hexahedron (cube)
| dodecahedron
| icosahedron
| octahedron

-- Define a property for having exactly 3 edges per vertex
def hasThreeEdgesPerVertex : RegularPolyhedra → Prop
| RegularPolyhedra.tetrahedron  => true
| RegularPolyhedra.hexahedron   => true
| RegularPolyhedra.dodecahedron => true
| RegularPolyhedra.icosahedron  => false
| RegularPolyhedra.octahedron   => false

-- The proof statement that the number of regular polyhedra with exactly 3 edges at each vertex is 3.
theorem threeEdgesPerVertexCount : (∃ l : List RegularPolyhedra, l.filter hasThreeEdgesPerVertex = [RegularPolyhedra.tetrahedron, RegularPolyhedra.hexahedron, RegularPolyhedra.dodecahedron]) := sorry

end threeEdgesPerVertexCount_l273_273786


namespace age_proof_l273_273788

theorem age_proof (y d : ℕ)
  (h1 : y = 4 * d)
  (h2 : y - 7 = 11 * (d - 7)) :
  y = 48 ∧ d = 12 :=
by
  -- The proof is omitted
  sorry

end age_proof_l273_273788


namespace perpendicular_tangents_l273_273394

-- Define the point P
def P := (1 : ℝ, 1 : ℝ)

-- Define the line equation
def line_eq (a b : ℝ) (p : ℝ × ℝ) : Prop := a * p.1 - b * p.2 - 2 = 0

-- Define the curve equation
def curve_eq (p : ℝ × ℝ) : Prop := p.2 = p.1 ^ 3

-- Define the slope of the tangent to the curve at the point P
def tangent_slope (p : ℝ × ℝ) : ℝ := 3 * p.1 ^ 2

-- Define the condition for perpendicular tangents
def perpendicular_slope (slope1 slope2 : ℝ) : Prop := slope1 * slope2 = -1 

-- Define the problem statement
theorem perpendicular_tangents (a b : ℝ) (h_line : line_eq a b P) (h_curve : curve_eq P)
  (h_perpendicular : perpendicular_slope (a/b) (tangent_slope P)) :
  a / b = -1 / 3 := 
  sorry

end perpendicular_tangents_l273_273394


namespace problem1_l273_273621

theorem problem1 (x y : ℤ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 :=
sorry

end problem1_l273_273621


namespace average_salary_of_all_employees_l273_273462

theorem average_salary_of_all_employees 
    (avg_salary_officers : ℝ)
    (avg_salary_non_officers : ℝ)
    (num_officers : ℕ)
    (num_non_officers : ℕ)
    (h1 : avg_salary_officers = 450)
    (h2 : avg_salary_non_officers = 110)
    (h3 : num_officers = 15)
    (h4 : num_non_officers = 495) :
    (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers)
    / (num_officers + num_non_officers) = 120 := by
  sorry

end average_salary_of_all_employees_l273_273462


namespace simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l273_273782

def A (a b : ℝ) := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) := 4 * a^2 + 6 * a * b + 8 * a

theorem simplify_2A_minus_B {a b : ℝ} :
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a :=
by
  sorry

theorem twoA_minusB_value_when_a_neg2_b_1 :
  2 * A (-2) (1) - B (-2) (1) = 54 :=
by
  sorry

theorem twoA_minusB_independent_of_a {b : ℝ} :
  (∀ a : ℝ, 2 * A a b - B a b = 6 * b - 8 * a) → b = -1 / 2 :=
by
  sorry

end simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l273_273782


namespace base_7_to_base_10_l273_273674

theorem base_7_to_base_10 (d1 d2 d3 d4 : ℕ) (h1 : d1 = 5) (h2 : d2 = 3) (h3 : d3 = 0) (h4 : d4 = 4) :
  7^3 * d1 + 7^2 * d2 + 7^1 * d3 + 7^0 * d4 = 1866 := by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end base_7_to_base_10_l273_273674


namespace minimum_value_l273_273852

noncomputable def point_in_triangle {A B C M : Type} [planar_geometry : add_comm_group A] := sorry

theorem minimum_value 
  {A B C M : Type}
  {m n p x y : ℝ}
  (h1 : point_in_triangle A B C M)
  (h2 : (A - B) • (A - C) = 2 * real.sqrt 3)
  (h3 : angle A B C = real.pi / 6)
  (h4 : f M = (1 / 2, x, y)) :
  ∃ x y, (x + y = 1 / 2) ∧ (1 / x + 4 / y ≥ 18) := 
begin
  sorry,
end

end minimum_value_l273_273852


namespace problem_solution_l273_273825

-- Definition of the parametric equations and the point P
def parametric_equations (r φ : ℝ) : (ℝ × ℝ) := (2 + r * Real.cos φ, r * Real.sin φ)

def passes_through_point (r : ℝ) (φ : ℝ) : Prop :=
  let (x, y) := parametric_equations r φ in
  x = 2 ∧ y = Real.sin (π / 3)

-- Given point P
def point_P : ℝ × ℝ := (2, Real.sin (π / 3))

-- Curve C₂ definition
def curve_C2 (ρ θ : ℝ) : Prop :=
  ρ^2 * (2 + Real.cos (2 * θ)) = 6

-- Main theorem to prove
theorem problem_solution :
  (∀ (r φ : ℝ), r > 0 ∧ passes_through_point r (π / 3) → (∀ ρ θ : ℝ, ρ = 4 * Real.cos θ)) ∧
  (∀ (ρ1 α ρ2 : ℝ), curve_C2 ρ1 α ∧ curve_C2 ρ2 (α + π / 2) → 
  (1 / (ρ1^2) + 1 / (ρ2^2) = 2 / 3)) :=
  sorry

end problem_solution_l273_273825


namespace interval_monotonic_increase_value_a_condition_l273_273405

noncomputable def f (x : ℝ) := 4 * Real.cos x * Real.sin (x + (Real.pi / 6)) - 1

theorem interval_monotonic_increase :
  (∀ k : ℤ, 
  ∀ x : ℝ, 
  - (Real.pi / 3) + k * Real.pi ≤ x ∧ x ≤ (Real.pi / 6) + k * Real.pi → 
  ∃ δ > 0, ∀ y : ℝ, abs (x - y) < δ → f y ≥ f x) :=
by sorry

theorem value_a_condition 
  (a : ℝ) 
  (h : ∀ x : ℝ, x ∈ Ioo (- (Real.pi / 4)) (Real.pi / 4) → 
     Real.sin (x)^2 + a * f (x + (Real.pi / 6)) + 1 > 6 * Real.cos (x)^4) :
  a > 5 / 2 :=
by sorry

end interval_monotonic_increase_value_a_condition_l273_273405


namespace repeating_decimals_product_l273_273311

-- Definitions to represent the conditions
def repeating_decimal_03_as_frac : ℚ := 1 / 33
def repeating_decimal_36_as_frac : ℚ := 4 / 11

-- The statement to be proven
theorem repeating_decimals_product : (repeating_decimal_03_as_frac * repeating_decimal_36_as_frac) = (4 / 363) :=
by {
  sorry
}

end repeating_decimals_product_l273_273311


namespace maximum_dn_l273_273295

-- Definitions of a_n and d_n based on the problem statement
def a (n : ℕ) : ℕ := 150 + (n + 1)^2
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Statement of the theorem
theorem maximum_dn : ∃ M, M = 2 ∧ ∀ n, d n ≤ M :=
by
  -- proof should be written here
  sorry

end maximum_dn_l273_273295


namespace monotonicity_of_f_l273_273681

noncomputable def f (a x : ℝ) : ℝ := (a * x) / (x + 1)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → 0 < a → f a x1 < f a x2) ∧
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → a < 0 → f a x1 > f a x2) :=
by {
  sorry
}

end monotonicity_of_f_l273_273681


namespace value_of_fraction_l273_273118

variable {x y : ℝ}

theorem value_of_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 * x + y) / (x - 3 * y) = -2) :
  (x + 3 * y) / (3 * x - y) = 2 :=
sorry

end value_of_fraction_l273_273118


namespace graph_quadrant_l273_273553

theorem graph_quadrant (x y : ℝ) : 
  y = 3 * x - 4 → ¬ ((x < 0) ∧ (y > 0)) :=
by
  intro h
  sorry

end graph_quadrant_l273_273553


namespace intersection_value_l273_273687

theorem intersection_value (x : ℝ) :
  (∃ y : ℝ, y = 12 / (x^2 + 6) ∧ x^2 + y = 5) ↔ (x = sqrt ((1 + sqrt 73) / 2) ∨ x = -sqrt ((1 + sqrt 73) / 2)) :=
by
  sorry

end intersection_value_l273_273687


namespace dealer_current_profit_l273_273798

/-- Given the conditions of the problem: -/
theorem dealer_current_profit (c s : ℝ) (x : ℝ) 
  (h1 : s = c * (1 + 0.01 * x)) 
  (h2 : s = 0.90 * c * (1 + 0.01 * (x + 15))) : 
  x = 35 :=
by
  -- original premises
  have h3 : c * (1 + 0.01 * x) = 0.90 * c * (1 + 0.01 * x + 0.15), from h2,
  -- Simplify and solve for x
  sorry

end dealer_current_profit_l273_273798


namespace bailey_towel_set_cost_l273_273281

def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def cost_per_guest_set : ℝ := 40.00
def cost_per_master_set : ℝ := 50.00
def discount_rate : ℝ := 0.20

def total_cost_before_discount : ℝ := 
  (guest_bathroom_sets * cost_per_guest_set) + (master_bathroom_sets * cost_per_master_set)

def discount_amount : ℝ := total_cost_before_discount * discount_rate

def final_amount_spent : ℝ := total_cost_before_discount - discount_amount

theorem bailey_towel_set_cost : final_amount_spent = 224.00 := by sorry

end bailey_towel_set_cost_l273_273281


namespace jerry_weekly_earnings_l273_273482

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l273_273482


namespace probability_N16_mod_7_eq_1_l273_273275

def selected_at_random (N: ℤ): Prop :=
  1 ≤ N ∧ N ≤ 2023

theorem probability_N16_mod_7_eq_1 :
  (∑ N in finset.Icc 1 2023, if (N^16 % 7 = 1) then 1 else 0).toReal / 2023 = 1 / 7 := by 
  sorry

end probability_N16_mod_7_eq_1_l273_273275


namespace system1_solution_correct_system2_solution_correct_l273_273113

theorem system1_solution_correct (x y : ℝ) (h1 : x + y = 5) (h2 : 4 * x - 2 * y = 2) :
    x = 2 ∧ y = 3 :=
  sorry

theorem system2_solution_correct (x y : ℝ) (h1 : 3 * x - 2 * y = 13) (h2 : 4 * x + 3 * y = 6) :
    x = 3 ∧ y = -2 :=
  sorry

end system1_solution_correct_system2_solution_correct_l273_273113


namespace length_of_fourth_side_cyclic_quad_l273_273527

theorem length_of_fourth_side_cyclic_quad (AB BC CD : ℝ) (hAB : AB = 4) (hBC : BC = 6) (hCD : CD = 8)
  (area_eq : ∀ A C : ℝ, (triangle_area A B C = triangle_area A C D)) :
  DA = 3 ∨ DA = 16/3 ∨ DA = 12 := sorry

end length_of_fourth_side_cyclic_quad_l273_273527


namespace John_overall_profit_l273_273841

theorem John_overall_profit :
  let CP_grinder := 15000
  let Loss_percentage_grinder := 0.04
  let CP_mobile_phone := 8000
  let Profit_percentage_mobile_phone := 0.10
  let CP_refrigerator := 24000
  let Profit_percentage_refrigerator := 0.08
  let CP_television := 12000
  let Loss_percentage_television := 0.06
  let SP_grinder := CP_grinder * (1 - Loss_percentage_grinder)
  let SP_mobile_phone := CP_mobile_phone * (1 + Profit_percentage_mobile_phone)
  let SP_refrigerator := CP_refrigerator * (1 + Profit_percentage_refrigerator)
  let SP_television := CP_television * (1 - Loss_percentage_television)
  let Total_CP := CP_grinder + CP_mobile_phone + CP_refrigerator + CP_television
  let Total_SP := SP_grinder + SP_mobile_phone + SP_refrigerator + SP_television
  let Overall_profit := Total_SP - Total_CP
  Overall_profit = 1400 := by
  sorry

end John_overall_profit_l273_273841


namespace m_value_in_triangle_l273_273473

theorem m_value_in_triangle (XY YZ XZ YM : ℝ) (hXY : XY = 5) (hYZ : YZ = 12) (hXZ : XZ = 13) (angle_Y : ∠XYZ = 90) (angle_bisector_YM : YM = m * sqrt 2):
  m = 60 / 17 :=
by
  sorry

end m_value_in_triangle_l273_273473


namespace part1_part2_l273_273415

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Part (1): Prove range for m
theorem part1 (m : ℝ) : (∃ x : ℝ, quadratic 1 (-5) m x = 0) ↔ m ≤ 25 / 4 := sorry

-- Part (2): Prove value of m given conditions on roots
theorem part2 (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : 3 * x1 - 2 * x2 = 5) : 
  m = x1 * x2 → m = 6 := sorry

end part1_part2_l273_273415


namespace largest_n_l273_273845

open Real

def is_not_perfect_square (n : ℕ) : Prop :=
  ∀ k : ℕ, k * k ≠ n

def a (n : ℕ) : ℕ :=
  Nat.floor (sqrt n)

def divides (a b : ℕ) : Prop :=
  ∃ k : ℕ, b = a * k

theorem largest_n (n : ℕ) : 
  (is_not_perfect_square n) ∧ (divides ((a n) ^ 3) (n ^ 2)) → n ≤ 24 := 
  sorry

end largest_n_l273_273845


namespace number_of_zeros_f_l273_273298

-- Define the function f(x)
noncomputable def f : ℝ → ℝ := 
  λ x, if x ∈ set.Icc (-1:ℝ) 4 then x^2 - 2^x else sorry

-- The main theorem
theorem number_of_zeros_f : ∀ (f : ℝ → ℝ),
  (∀ x, f(x) = f(x - 5)) →  -- f(x) is periodic with period 5
  (∀ x ∈ set.Icc (-1:ℝ) 4, f(x) = x^2 - 2^x) →  -- f(x) = x^2 - 2^x in (-1, 4]
  ∃ n, n = 1207 ∧  -- The number of zeros of f(x) on [0, 2013] is 1207
  ∀ x ∈ set.Icc (0:ℝ) 2013, f(x) = 0 → true :=
by sorry

end number_of_zeros_f_l273_273298


namespace problem_part_1_problem_part_2_l273_273757

-- We define the function f
def f (x a : ℝ) : ℝ := x - Real.log (x + a)

-- Nonnegative real numbers
def nonneg_real := {x : ℝ // x ≥ 0}

theorem problem_part_1 : ∀ (a > 0), (∃ (x : ℝ), f x a = 0) ↔ a = 1 := 
by
  intros a ha
  exact sorry

theorem problem_part_2 : ∀ (k : ℝ), (∀ (x : nonneg_real), f x.1 (1:ℝ) ≤ k * x.1^2) ↔ k ≥ 1/2 :=
by
  intros k
  exact sorry

end problem_part_1_problem_part_2_l273_273757


namespace incorrect_statement_C_l273_273888

theorem incorrect_statement_C :
  (∀ b h : ℕ, (2 * b) * h = 2 * (b * h)) ∧
  (∀ b h : ℕ, (1 / 2) * b * (2 * h) = 2 * ((1 / 2) * b * h)) ∧
  (∀ r : ℕ, (π * (2 * r) ^ 2 ≠ 2 * (π * r ^ 2))) ∧
  (∀ a b : ℕ, (a / 2) / (2 * b) ≠ a / b) ∧
  (∀ x : ℤ, x < 0 -> 2 * x < x) →
  false :=
by
  intros h
  sorry

end incorrect_statement_C_l273_273888


namespace min_sqrt_mn_l273_273717

theorem min_sqrt_mn (a b m n : ℝ) (h1 : a^2 + b^2 = 3) (h2 : m * a + n * b = 3) :
  sqrt (m^2 + n^2) = sqrt 3 :=
sorry

end min_sqrt_mn_l273_273717


namespace silvia_percentage_shorter_l273_273481

theorem silvia_percentage_shorter :
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  (abs (( (j - s) / j) * 100 - 25) < 1) :=
by
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  show (abs (( (j - s) / j) * 100 - 25) < 1)
  sorry

end silvia_percentage_shorter_l273_273481


namespace smallest_n_for_gn_gt_15_l273_273796

-- Definition of the function that computes the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Definition of the function g based on the sum of digits of (10/3)^n
def g (n : ℕ) : ℕ :=
  sum_of_digits ((10/3)^n).to_nat

-- The main statement we want to prove
theorem smallest_n_for_gn_gt_15 : ∃ n : ℕ, 0 < n ∧ g(n) > 15 ∧ ∀ m < n, g(m) ≤ 15 :=
by
  sorry

end smallest_n_for_gn_gt_15_l273_273796


namespace milly_folding_time_l273_273513

theorem milly_folding_time (
    mix_time : ℕ := 10, 
    bake_time : ℕ := 30, 
    rest_time_each : ℕ := 75, 
    folds : ℕ := 4, 
    total_time_hours : ℕ := 6
  ) : 
  (total_time_hours * 60 - mix_time - bake_time - folds * rest_time_each) / folds = 5 := 
by sorry

end milly_folding_time_l273_273513


namespace maximize_sector_area_l273_273758

noncomputable def sector_radius_angle (r l α : ℝ) : Prop :=
  2 * r + l = 40 ∧ α = l / r

theorem maximize_sector_area :
  ∃ r α : ℝ, sector_radius_angle r 20 α ∧ r = 10 ∧ α = 2 :=
by
  sorry

end maximize_sector_area_l273_273758


namespace angle_BAO_eq_angle_CAH_l273_273362

variable {A B C O H : Type}
variables [triangle : Triangle A B C] [circumcenter : Cirumcenter A B C O] [orthocenter : Orthocenter A B C H]

theorem angle_BAO_eq_angle_CAH : angle BAO = angle CAH := sorry

end angle_BAO_eq_angle_CAH_l273_273362


namespace vector_dot_product_property_l273_273383

variables (a b : ℝ^3) (θ : ℝ)

open real_inner_product_space

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

def cos_theta (a b : ℝ^3) : ℝ := (a.dot_product b) / ((magnitude a) * (magnitude b))

theorem vector_dot_product_property
  (h1 : cos_theta a b = 1 / 3)
  (h2 : magnitude a = 1)
  (h3 : magnitude b = 3) :
  (2 • a + b) ∙ b = 11 :=
begin
  sorry
end

end vector_dot_product_property_l273_273383


namespace max_vertices_Thubkaew_l273_273212

def Chaisri (polygon : Type*) :=
  ∀ (polygon : Type*), ∃ (n : ℕ), n = 2019
  ∧ is_regular_polygon polygon n
  ∧ ∀ (triangle : Type*), is_triangle_in_polygon triangle polygon

def Thubkaew (polygon : Type*) :=
  ∀ (polygon : Type*), is_convex_polygon polygon
  ∧ ∀ (polygon : Type*), 
    ∃ (chaisri_triangulation : Type*),
      (∀ triangle ∈ chaisri_triangulation, Chaisri triangle)
      ∧ (∀ vertex ∈ chaisri_triangulation, vertex ∈ polygon)

theorem max_vertices_Thubkaew : 
  ∀ (thubkaew_polygon : Type*), Thubkaew thubkaew_polygon →
  ∃ (n : ℕ), n ≤ 4038 := 
by {
  sorry
}

end max_vertices_Thubkaew_l273_273212


namespace units_digit_k_squared_plus_two_to_k_is_7_l273_273861

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_k_squared_plus_two_to_k_is_7 :
  let k := (2008^2 + 2^2008) in
  units_digit (k^2 + 2^k) = 7 :=
by
  let k := (2008^2 + 2^2008)
  sorry

end units_digit_k_squared_plus_two_to_k_is_7_l273_273861


namespace Missy_handle_claims_l273_273289

def MissyCapacity (JanCapacity JohnCapacity MissyCapacity : ℕ) :=
  JanCapacity = 20 ∧
  JohnCapacity = JanCapacity + (30 * JanCapacity) / 100 ∧
  MissyCapacity = JohnCapacity + 15

theorem Missy_handle_claims :
  ∃ MissyCapacity, MissyCapacity = 41 :=
by
  use 41
  unfold MissyCapacity
  sorry

end Missy_handle_claims_l273_273289


namespace no_integer_roots_of_quadratic_l273_273436

theorem no_integer_roots_of_quadratic (m n : ℤ) (h₁ : odd m) (h₂ : odd n) (h₃ : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁ * r₂ = n ∧ r₁ + r₂ = -m) :
  ∀ r : ℝ, ¬(r * r + m * r + n = 0 ∧ ∃ k : ℤ, r = k) :=
by
  sorry

end no_integer_roots_of_quadratic_l273_273436


namespace find_a_l273_273019

theorem find_a (x y a : ℝ) 
  (h1 : 2 * x + 7 * y = 11)
  (h2 : 5 * x - 4 * y = 6)
  (h3 : 3 * x - 6 * y + 2 * a = 0) : 
  a = 0 := 
begin
  -- skipping the proof as we are required to provide only the statement
  sorry
end

end find_a_l273_273019


namespace people_from_second_row_joined_l273_273144

theorem people_from_second_row_joined
  (initial_first_row : ℕ) (initial_second_row : ℕ) (initial_third_row : ℕ) (people_waded : ℕ) (remaining_people : ℕ)
  (H1 : initial_first_row = 24)
  (H2 : initial_second_row = 20)
  (H3 : initial_third_row = 18)
  (H4 : people_waded = 3)
  (H5 : remaining_people = 54) :
  initial_second_row - (initial_first_row + initial_second_row + initial_third_row - initial_first_row - people_waded - remaining_people) = 5 :=
by
  sorry

end people_from_second_row_joined_l273_273144


namespace line_passes_through_fixed_point_l273_273366

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ) (m : ℝ), 
    k ≠ 0 → 
    (∃ x y : ℝ, x^2/4 + y^2 = 1 ∧ y = k*x + m) → 
    (∃ A B : ℝ × ℝ, 
      let x1 := A.1, y1 := A.2, x2 := B.1, y2 := B.2 in 
      (1 + 4*k^2) * (x1 * x2) + 8*m*k*x1 + 4*(m^2 - 1) = 0 ∧
      y = k*x + m  → 
      (x1*x2 - 2*(x1 + x2) + 4 + y1*y2 = 0) →
      (5*m^2 + 16*k*m + 12*k^2 = 0) →
    (∃ c : ℝ × ℝ, c = (6/5, 0))) 
:= sorry

end line_passes_through_fixed_point_l273_273366


namespace largest_digit_divisible_by_6_l273_273981

theorem largest_digit_divisible_by_6 :
  ∃ (N : ℕ), N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0 ∧ (∀ m ∈ {N | N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0}, m ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273981


namespace domain_of_function_l273_273930

-- Define the setting and the constants involved
variables {f : ℝ → ℝ}
variable {c : ℝ}

-- The statement about the function's domain
theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (x ≤ 0 ∧ x ≠ -c) :=
sorry

end domain_of_function_l273_273930


namespace intersection_of_A_and_B_l273_273369

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := 
by 
  sorry

end intersection_of_A_and_B_l273_273369


namespace smallest_x_integer_value_l273_273187

theorem smallest_x_integer_value (x : ℤ) (h : (x - 5) ∣ 58) : x = -53 :=
by
  sorry

end smallest_x_integer_value_l273_273187


namespace most_accurate_method_is_independence_test_l273_273602

-- Definitions and assumptions
inductive Methods
| contingency_table
| independence_test
| stacked_bar_chart
| others

def related_or_independent_method : Methods := Methods.independence_test

-- Proof statement
theorem most_accurate_method_is_independence_test :
  related_or_independent_method = Methods.independence_test :=
sorry

end most_accurate_method_is_independence_test_l273_273602


namespace lines_are_parallel_l273_273185

theorem lines_are_parallel : 
  ∀ (x y : ℝ), (2 * x - y = 7) → (2 * x - y - 1 = 0) → False :=
by
  sorry  -- Proof will be filled in later

end lines_are_parallel_l273_273185


namespace min_colors_for_coloring_points_l273_273722

/-- Given 2022 lines in the plane such that no two are parallel and no three are concurrent, 
     the minimum number of colors required to color the set of their intersection points, such that 
     any two points on the same line, whose connecting segment does not contain any other points from the set, 
     have different colors, is 3. -/
theorem min_colors_for_coloring_points (L : finset (set (euclidean_space ℝ (fin 2))))
  (h : L.card = 2022)
  (no_parallel : ∀ l₁ l₂ ∈ L, l₁ ≠ l₂ → ∃! p, p ∈ l₁ ∧ p ∈ l₂)
  (no_concurrent : ∀ l₁ l₂ l₃ ∈ L, l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃ →
    ¬ ∃ p, p ∈ l₁ ∧ p ∈ l₂ ∧ p ∈ l₃) :
  ∃ f : (euclidean_space ℝ (fin 2) → fin 3), 
  (∀ l ∈ L, ∀ p₁ p₂ ∈ l, p₁ ≠ p₂ → 
    ¬ ∃ p, p ∈ segment ℝ p₁ p₂ \ {p₁, p₂} → f p₁ ≠ f p₂) :=
sorry

end min_colors_for_coloring_points_l273_273722


namespace interior_angle_regular_nonagon_l273_273978

theorem interior_angle_regular_nonagon : 
  ∀ (n : ℕ), n = 9 → 
  ∀ regular : Prop, regular → 
  ∑ i in finset.range (n - 2), (λ _, 180 : ℕ → ℝ) i / n = 140 :=
begin
  intros n hn regular hregular,
  rw hn,
  -- the proof is omitted
  sorry,
end

end interior_angle_regular_nonagon_l273_273978


namespace smallest_degree_polynomial_with_roots_l273_273115

theorem smallest_degree_polynomial_with_roots :
  ∃ (p : Polynomial ℚ), p ≠ 0 ∧ 
  p.eval (-1 + Real.sqrt 5) = 0 ∧
  p.eval (-1 - Real.sqrt 5) = 0 ∧
  p.eval (2 - Real.sqrt 3) = 0 ∧
  p.eval (2 + Real.sqrt 3) = 0 ∧
  Polynomial.degree p = 4 :=
begin
  sorry
end

end smallest_degree_polynomial_with_roots_l273_273115


namespace running_speed_l273_273239

theorem running_speed (walk_speed total_distance walk_time total_time run_distance : ℝ) 
  (h_walk_speed : walk_speed = 4)
  (h_total_distance : total_distance = 4)
  (h_walk_time : walk_time = 0.5)
  (h_total_time : total_time = 0.75)
  (h_run_distance : run_distance = total_distance / 2) :
  (2 / ((total_time - walk_time) - 2 / walk_speed)) = 8 := 
by
  -- To be proven
  sorry

end running_speed_l273_273239


namespace find_shortest_side_of_triangle_l273_273122

noncomputable def triangle_shortest_side :=
  let S : ℝ := 6 * real.sqrt 6
  let p : ℝ := 9
  let distance_center_to_vertex := (2 * real.sqrt 42) / 3
  let shortest_side : ℝ := 5
  S = 6 * real.sqrt 6 ∧ 2 * p = 18 ∧ distance_center_to_vertex = (2 * real.sqrt 42) / 3 →
  shortest_side = 5

-- Statement to prove the shortest side
theorem find_shortest_side_of_triangle (S p distance_center_to_vertex : ℝ)
  (hS : S = 6 * real.sqrt 6)
  (hp : 2 * p = 18)
  (hd : distance_center_to_vertex = (2 * real.sqrt 42) / 3) :
  let shortest_side := 5 in
  shortest_side = 5 :=
by
  sorry

end find_shortest_side_of_triangle_l273_273122


namespace area_inner_square_l273_273819

theorem area_inner_square
  (ABCD : Type) [square ABCD] (AB : ℝ) (side_AB : AB = 10)
  (EFGH : Type) [inner_square ABCD EFGH] (BE : ℝ) (BE_perpendicular_EH : BE = 2 ∧ BE ⊥ EH) :
  area EFGH = 100 - 16 * sqrt 6 :=
by
  sorry

end area_inner_square_l273_273819


namespace bowling_tournament_l273_273279

-- Definition of the problem conditions
def playoff (num_bowlers: Nat): Nat := 
  if num_bowlers < 5 then
    0
  else
    2^(num_bowlers - 1)

-- Theorem statement to prove
theorem bowling_tournament (num_bowlers: Nat) (h: num_bowlers = 5): playoff num_bowlers = 16 := by
  sorry

end bowling_tournament_l273_273279


namespace min_value_fraction_l273_273876

theorem min_value_fraction (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + 2 * b + 3 * c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 := 
sorry

end min_value_fraction_l273_273876


namespace jackson_pancakes_l273_273479

theorem jackson_pancakes :
  let P_milk_good : ℝ := 0.8
  let P_egg_good : ℝ := 0.4
  let P_flour_good : ℝ := 0.75
  P_milk_good * P_egg_good * P_flour_good = 0.18 :=
by
  let P_milk_good := 0.8
  let P_egg_good := 0.4
  let P_flour_good := 0.75
  calc
    P_milk_good * P_egg_good * P_flour_good = 0.8 * (0.4 * 0.75) : by rw [mul_assoc]
    ... = 0.8 * 0.3 : by norm_num
    ... = 0.24 : by norm_num
    ... = 0.18 : by norm_num

end jackson_pancakes_l273_273479


namespace smallest_sum_with_probability_l273_273160

theorem smallest_sum_with_probability (N : ℕ) (p : ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 6) (h2 : 6 * N = 2022) (h3 : p > 0) :
  ∃ M, M = 337 ∧ (∀ sum, sum = 2022 → P(sum) = p) ∧ (∀ min_sum, min_sum = N → P(min_sum) = p):=
begin
  sorry
end

end smallest_sum_with_probability_l273_273160


namespace class_average_42_l273_273026

theorem class_average_42 (n : ℕ) (s1 s2 s3 : ℝ) (avg : ℝ) :
  n = 25 →
  s1 = 3 * 95 →
  s2 = 5 * 0 →
  avg = 45 →
  s3 = 17 * avg →
  (s1 + s2 + s3) / n = 42 :=
by
  intros hn hs1 hs2 havg hs3
  rw [hn, hs1, hs2, hs3, havg]
  norm_num
  sorry

end class_average_42_l273_273026


namespace original_cost_of_article_l273_273657

theorem original_cost_of_article (x: ℝ) (h: 0.76 * x = 320) : x = 421.05 :=
sorry

end original_cost_of_article_l273_273657


namespace first_pipe_fill_time_l273_273650

theorem first_pipe_fill_time 
  (T : ℝ)
  (h1 : 48 * (1 / T - 1 / 24) + 18 * (1 / T) = 1) :
  T = 22 :=
by
  sorry

end first_pipe_fill_time_l273_273650


namespace farmer_has_42_cows_left_l273_273229

def initial_cows : ℕ := 51
def added_cows : ℕ := 5
def fraction_sold : ℚ := 1/4

theorem farmer_has_42_cows_left : 
  let total_cows := initial_cows + added_cows in
  let cows_sold := total_cows * fraction_sold in
  total_cows - cows_sold = 42 := 
by 
  -- The proof would go here, but we are only required to state the theorem.
  sorry

end farmer_has_42_cows_left_l273_273229


namespace intersection_sum_l273_273334

-- Define the conditions
def condition_1 (k : ℝ) := k > 0
def line1 (x y k : ℝ) := 50 * x + k * y = 1240
def line2 (x y k : ℝ) := k * y = 8 * x + 544
def right_angles (k : ℝ) := (-50 / k) * (8 / k) = -1

-- Define the point of intersection
def point_of_intersection (m n : ℝ) (k : ℝ) := line1 m n k ∧ line2 m n k

-- Prove that m + n = 44 under the given conditions
theorem intersection_sum (m n k : ℝ) :
  condition_1 k →
  right_angles k →
  point_of_intersection m n k →
  m + n = 44 :=
by
  sorry

end intersection_sum_l273_273334


namespace polynomial_remainder_l273_273304

theorem polynomial_remainder :
  ∀ (x : ℝ),
  let Q := x^{105} / (x^2 - 4*x + 3),
  let R := x^{105} % (x^2 - 4*x + 3) 
  in R = ( (3^105 - 1) / 2) * x + 2 - ( (3^105 - 1) / 2) :=
by
  sorry

end polynomial_remainder_l273_273304


namespace petes_original_number_l273_273519

theorem petes_original_number (x : ℤ) (h : 5 * (3 * x - 6) = 195) : x = 15 :=
sorry

end petes_original_number_l273_273519


namespace simple_proposition_is_4_l273_273101

-- Definitions of the given propositions
def P1 : Prop := ∃ k : ℕ, 12 = 4 * k ∧ 12 = 3 * k
def P2 : Prop := ∀ (ΔABC ΔDEF : Type) (f : ΔABC ≃ ΔDEF), ¬ (∀ x y : ΔABC, f x = f y → x = y)
def P3 : Prop := ∀ (ABC : Type) [is_triangle ABC] (M : Point), (midline_parallel M) ∧ (midline_half_base M)
def P4 : Prop := ∀ (ABC : Triangle), is_isosceles ABC → ∀ (a b : Angle), base_angles_equal ABC a b

-- Prove the correct proposition is the simple one
theorem simple_proposition_is_4 : P4 := sorry

end simple_proposition_is_4_l273_273101


namespace jenny_donuts_combination_l273_273839

theorem jenny_donuts_combination:
  ∃ c : ℕ, c = 110 ∧ 
  (∀ k : ℕ, (2 ≤ k ∧ k ≤ 3) → (∃ x : ℕ, x = (choose 5 k *
  if k = 2 then (choose (4 + k - 1) (k - 1))
  else (choose (2 + k - 1) (k - 1)))) ∧
  (c = ∑ k in {2, 3}, (choose 5 k) * (if k = 2 then (choose (4 + k - 1) (k - 1) ) else (choose (2 + k - 1) (k - 1)))) ) :=
by
  sorry

end jenny_donuts_combination_l273_273839


namespace find_number_of_sides_l273_273954

-- Defining the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Statement of the problem
theorem find_number_of_sides (h : sum_of_interior_angles n = 1260) : n = 9 :=
by
  sorry

end find_number_of_sides_l273_273954


namespace proof_problem_l273_273707

noncomputable def f (x : ℝ) := Real.tan (x + (Real.pi / 4))

theorem proof_problem :
  (- (3 * Real.pi) / 4 < 1 - Real.pi ∧ 1 - Real.pi < -1 ∧ -1 < 0 ∧ 0 < Real.pi / 4) →
  f 0 > f (-1) ∧ f (-1) > f 1 := by
  sorry

end proof_problem_l273_273707


namespace quadratic_polynomial_positive_a_l273_273734

variables {a b c n : ℤ}
def p (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem quadratic_polynomial_positive_a (h : a ≠ 0) 
    (hn : n < p n) (hn_ppn : p n < p (p n)) (hn_pppn : p (p n) < p (p (p n))) :
    a > 0 :=
by
    sorry

end quadratic_polynomial_positive_a_l273_273734


namespace problem_statement_l273_273272

-- Definitions of the functions in the problem
def fA (x : ℝ) : ℝ := -1 / x
def fB (x : ℝ) : ℝ := Real.tan x
def fC (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))
def fD (x : ℝ) : ℝ := x

-- Conditions for oddness and monotonicity
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def monotonic_increasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

-- Problem statement
theorem problem_statement :
  odd_function fC ∧ monotonic_increasing fC (Set.Ioo (-1 : ℝ) 1) ∧
  (¬ (odd_function fA ∧ monotonic_increasing fA {x | x ≠ 0})) ∧
  (¬ (odd_function fB ∧ monotonic_increasing fB Set.univ)) ∧
  (¬ (odd_function fD ∧ monotonic_increasing fD {x | x ≠ 1})) :=
by
  sorry

end problem_statement_l273_273272


namespace problem_statement_l273_273680

noncomputable def b_1 : ℝ := 5 / 8
noncomputable def b_2 : ℝ := 0
noncomputable def b_3 : ℝ := -5 / 16
noncomputable def b_4 : ℝ := 0
noncomputable def b_5 : ℝ := 1 / 16

theorem problem_statement :
  (∀ θ : ℝ, sin(θ) ^ 5 = b_1 * sin(θ) + b_2 * sin(2 * θ) + b_3 * sin(3 * θ) + b_4 * sin(4 * θ) + b_5 * sin(5 * θ)) →
  b_1 ^ 2 + b_2 ^ 2 + b_3 ^ 2 + b_4 ^ 2 + b_5 ^ 2 = 101 / 256 :=
by
  sorry

end problem_statement_l273_273680


namespace quadratic_function_form_exists_l273_273326

theorem quadratic_function_form_exists :
  ∃ (a b c : ℝ), (∀ (x : ℝ), y = a*x^2 + b*x + c) ∧ 
                (a < 0) ∧ 
                (b = 0) ∧ 
                (c > 0) :=
begin
  sorry
end

end quadratic_function_form_exists_l273_273326


namespace distance_traveled_l273_273891

theorem distance_traveled
  (map_scale : ℝ)
  (map_distance_inches : ℝ)
  (mountain_height_feet : ℝ)
  (inclination_angle_degrees : ℝ)
  (inch_to_cm : ℝ)
  (foot_to_m : ℝ)
  (sin_30_deg : ℝ)
  (expected_distance : ℝ) :
  map_scale = 15000 →
  map_distance_inches = 8.5 →
  mountain_height_feet = 3000 →
  inclination_angle_degrees = 30 →
  inch_to_cm = 2.54 →
  foot_to_m = 0.3048 →
  sin_30_deg = 0.5 →
  expected_distance = 5067.3 →
  let map_distance_meters := (map_distance_inches * map_scale * inch_to_cm / 100) in
  let mountain_height_meters := mountain_height_feet * foot_to_m in
  let mountain_distance_meters := mountain_height_meters / sin_30_deg in
  map_distance_meters + mountain_distance_meters = expected_distance := 
begin
  intros,
  sorry
end

end distance_traveled_l273_273891


namespace more_cats_than_dogs_l273_273959

-- Define the number of cats and dogs
def c : ℕ := 23
def d : ℕ := 9

-- The theorem we need to prove
theorem more_cats_than_dogs : c - d = 14 := by
  sorry

end more_cats_than_dogs_l273_273959


namespace distance_from_center_to_plane_optimization_condition_l273_273254

noncomputable def distance_to_plane (O : Point) (r : ℝ) (A B C : Triangle)
  (h_radius : r = 8)
  (h_tangent : ∀ (T : Triangle), T.side_length ∈ {17, 26}) : ℝ :=
  ∃ (d : ℝ), d = √41.4

theorem distance_from_center_to_plane_optimization_condition 
  (O : Point) 
  (r : ℝ) 
  (A B C : Triangle)
  (h_radius : r = 8)
  (h_tangent : ∀ (T : Triangle), T.side_length ∈ {17, 26}) :
  distance_to_plane O r A B C h_radius h_tangent = √41.4 := 
sorry

end distance_from_center_to_plane_optimization_condition_l273_273254


namespace determinant_is_zero_l273_273694

noncomputable def determinant_matrix : ℝ → ℝ → ℝ := λ α β,
  let A := ![
    ![0, real.cos α, -real.sin α],
    ![-real.cos α, 0, real.cos β],
    ![real.sin α, -real.cos β, 0]
  ] in
  matrix.det A

theorem determinant_is_zero (α β : ℝ) : determinant_matrix α β = 0 :=
by sorry

end determinant_is_zero_l273_273694


namespace largest_digit_divisible_by_6_l273_273979

theorem largest_digit_divisible_by_6 :
  ∃ (N : ℕ), N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0 ∧ (∀ m ∈ {N | N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0}, m ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273979


namespace only_integral_solution_l273_273108

def integral_solution (n x y z : ℤ) : Prop :=
  (xy + yz + zx = 3n^2 - 1) ∧
  (x + y + z = 3n) ∧
  (x ≥ y) ∧ (y ≥ z)

theorem only_integral_solution (n x y z : ℤ)
  (h : integral_solution n x y z) :
  x = n + 1 ∧ y = n ∧ z = n - 1 :=
begin
  sorry
end

end only_integral_solution_l273_273108


namespace equivalent_problem_l273_273117

variable (x y : ℝ)
variable (hx_ne_zero : x ≠ 0)
variable (hy_ne_zero : y ≠ 0)
variable (h : (3 * x + y) / (x - 3 * y) = -2)

theorem equivalent_problem : (x + 3 * y) / (3 * x - y) = 2 :=
by
  sorry

end equivalent_problem_l273_273117


namespace simplify_fraction_l273_273541

theorem simplify_fraction (a b c d : ℕ) (h₁ : a = 2) (h₂ : b = 462) (h₃ : c = 29) (h₄ : d = 42) :
  (a : ℚ) / (b : ℚ) + (c : ℚ) / (d : ℚ) = 107 / 154 :=
by {
  sorry
}

end simplify_fraction_l273_273541


namespace calc_2a_plus_b_dot_b_l273_273378

open Real

variables (a b : ℝ^3)
variables (cos_angle : ℝ)
variables (norm_a norm_b : ℝ)

-- Conditions from the problem
axiom cos_angle_is_1_3 : cos_angle = 1 / 3
axiom norm_a_is_1 : ∥a∥ = 1
axiom norm_b_is_3 : ∥b∥ = 3

-- Dot product calculation
axiom a_dot_b_is_1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle

-- Prove the target
theorem calc_2a_plus_b_dot_b : 
  (2 * a + b).dot b = 11 :=
by
  -- These assertions setup the problem statement in Lean
  have h1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle, 
  from a_dot_b_is_1,
  have h2 : ∥a∥ = 1, from norm_a_is_1,
  have h3 : ∥b∥ = 3, from norm_b_is_3,
  have h4 : cos_angle = 1 / 3, from cos_angle_is_1_3,
  sorry

end calc_2a_plus_b_dot_b_l273_273378


namespace range_of_a_for_integer_solutions_l273_273335

theorem range_of_a_for_integer_solutions (a : ℝ) :
  (∃ x : ℤ, (a - 2 < x ∧ x ≤ 3)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_for_integer_solutions_l273_273335


namespace log_point_passes_through_l273_273551

variable {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1)

theorem log_point_passes_through : (3, -1) ∈ { p : ℝ × ℝ | ∃ x, p = (x, log a (x - 2) - 1) } :=
sorry

end log_point_passes_through_l273_273551


namespace area_triangle_ratio_l273_273368

-- Define the basic structure of points and vectors
noncomputable theory
open_locale classical

variables {A B C O : Type} [add_comm_group A] [module ℝ A]

variables (a b c o : A)

-- Defining the condition given in the problem
def is_inside_triangle (o a b c : A) : Prop :=
  o + 2 • b + 3 • c = 0

-- The main theorem statement asserting the ratio of the areas
theorem area_triangle_ratio (h : is_inside_triangle o a b c) : 
  area (triangle a b c) / area (triangle a o c) = 3 :=
sorry

end area_triangle_ratio_l273_273368


namespace largest_digit_divisible_by_6_l273_273982

theorem largest_digit_divisible_by_6 :
  ∃ (N : ℕ), N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0 ∧ (∀ m ∈ {N | N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0}, m ≤ N) :=
sorry

end largest_digit_divisible_by_6_l273_273982


namespace solution_l273_273505

-- Definitions corresponding to the conditions
def R := ℝ
def N := { n : ℕ // 0 < n }
def P := Set N

-- Definition of the function g
def g (q : ℚ) : ℕ := 
  if q = 0 then 1 
  else 
    let m := q.num in
    let n := q.denom in
    if 0 < q
    then 2 ^ m.natAbs * 3 ^ n
    else 2 ^ m.natAbs * 5 ^ n

-- Definition of the function f
def f (x : R) : P := {n : N | ∃ r : ℚ, r ≤ x ∧ g r = n.val}

-- The main theorem statement
theorem solution : ∀ a b : R, a < b → f a ⊂ f b :=
by
  intros a b hab
  sorry

end solution_l273_273505


namespace part1_part2_l273_273420

/-- Given two lines l_1 and l_2 defined as follows: -/
def line_1 (a : ℝ) : (ℝ × ℝ) → Prop := λ (x y) => a * x + 2 * y + 6 = 0
def line_2 (a : ℝ) : (ℝ × ℝ) → Prop := λ (x y) => x + (a - 1) * y + a^2 - 1 = 0

/-- Given two lines l_1 and l_2 are perpendicular -/
def perpendicular (a : ℝ) : Prop := a + 2 * (a - 1) = 0

/-- When l_1 perpendicular l_2, find the value of a. -/
theorem part1 : ∃ a : ℝ, perpendicular a ∧ a = 2 / 3 := 
sorry

/-- Under the condition of part1, if line l_3 is parallel to l_2 and passes through point A (1, -3). 
Find the general equation of l_3. -/
def point_A : ℝ × ℝ := (1, -3)

def line_3 (C : ℝ) : (ℝ × ℝ) → Prop := λ (x y) => x - 1 / 3 * y + C = 0

def line_3_condition (A : ℝ × ℝ) (C : ℝ) : Prop := line_3 C A.fst A.snd

theorem part2 : ∃ C : ℝ, 
  perpendicular (2 / 3) ∧ 
  ∀ A : ℝ × ℝ, A = point_A → (parallel (λ (x y : ℝ) => line_3 C x y) (line_2 (2 / 3)) ∧ 
  line_3 C A.fst A.snd) ∧ C = -2 := 
sorry

end part1_part2_l273_273420


namespace smallest_sum_proof_l273_273156

theorem smallest_sum_proof (N : ℕ) (p : ℝ) (h1 : 6 * N = 2022) (hp : p > 0) : (N * 1 = 337) :=
by 
  have hN : N = 2022 / 6 := by 
    sorry
  exact hN

end smallest_sum_proof_l273_273156


namespace apple_stack_total_l273_273636

theorem apple_stack_total (n m : ℕ) (h1 : n = 4) (h2 : m = 7) (h3 : ∀ k, k ≤ min n m → (k - 1) * (n - k + 1) * (m - k + 1) ≥ 0) : (∑ k in range (min n m), (n - k) * (m - k)) = 60 :=
by {
  have h_nm : min n m = 4,
  { rw [min_eq_left h1, h1, h2], },
  rw [h_nm, sum_range_succ'],
  calc (n * m + (n - 1) * (m - 1) + (n - 2) * (m - 2) + (n - 3) * (m - 3)) 
    = 28 + 18 + 10 + 4 : by sorry
    = 60 : by norm_num
}

end apple_stack_total_l273_273636


namespace trapezoid_perimeter_l273_273656

theorem trapezoid_perimeter : 
  let BC := 60
  let AP := 20
  let DQ := 19
  let h := 30
  let AD := AP + BC + DQ
  let AB := Real.sqrt (AP^2 + h^2)
  let CD := Real.sqrt (DQ^2 + h^2)
  in AD = 20 + 60 + 19 → BC = 60 → AP = 20 → DQ = 19 → h = 30 →
     (AB + BC + CD + AD) = (Real.sqrt 1300 + Real.sqrt 1261 + 159) :=
by
  intros
  sorry

end trapezoid_perimeter_l273_273656


namespace value_of_a7_l273_273760

noncomputable theory

open Classical

variables {a : ℕ → ℝ} (r : ℝ)

-- Conditions
def is_geometric_sequence : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def roots_of_quadratic : Prop :=
  ∃ a₅ a₉ : ℝ, (a 5 = a₅) ∧ (a 9 = a₉) ∧ (a₅ + a₉ = -2016) ∧ (a₅ * a₉ = 9)

-- Question to prove
theorem value_of_a7 (hseq : is_geometric_sequence r) (hroot : roots_of_quadratic) : 
    a 7 = -3 :=
sorry

end value_of_a7_l273_273760


namespace find_t_and_m_l273_273168

theorem find_t_and_m :
  ∀ (x t m : ℝ),
    t = cos (2 * (π / 4) + π / 6) →
    (∀ x, (cos (2 * (π / 4 + m)) = cos 2 x)) →
    m > 0 →
    t = -1 / 2 ∧ m = π / 12 :=
by
  intros x t m ht hP hm
  have ht_value : t = cos (2 * π / 4 + π / 6), by assumption
  have P'_cos2x : cos (2 * (π / 4 + m)) = cos (2 * x), by intro; assumption
  have m_positive : m > 0, by assumption
  sorry

end find_t_and_m_l273_273168


namespace reflection_correct_l273_273653

-- Define vectors and conditions
def vec1 := ⟨3, 1⟩ : ℝ × ℝ
def vec2 := ⟨-1, -3⟩ : ℝ × ℝ
def line := ⟨1, -1⟩ : ℝ × ℝ
def vecToReflect := ⟨0, 2⟩ : ℝ × ℝ

-- Define the reflection result to prove
def reflection_result := ⟨-2, -2⟩ : ℝ × ℝ

-- Theorem to prove
theorem reflection_correct :
  (reflect_over_line vec1 vec2 line vecToReflect) = reflection_result := 
sorry

end reflection_correct_l273_273653


namespace donuts_per_student_l273_273884

theorem donuts_per_student 
    (dozens_of_donuts : ℕ)
    (students_in_class : ℕ)
    (percentage_likes_donuts : ℕ)
    (students_who_like_donuts : ℕ)
    (total_donuts : ℕ)
    (donuts_per_student : ℕ) :
    dozens_of_donuts = 4 →
    students_in_class = 30 →
    percentage_likes_donuts = 80 →
    students_who_like_donuts = (percentage_likes_donuts * students_in_class) / 100 →
    total_donuts = dozens_of_donuts * 12 →
    donuts_per_student = total_donuts / students_who_like_donuts →
    donuts_per_student = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end donuts_per_student_l273_273884


namespace trapezoid_heights_equal_l273_273181

-- Definition of a trapezoid: A quadrilateral with at least one pair of parallel sides.
structure Trapezoid (A B C D : Type) :=
(parallel_sides : A = B ∧ C = D)

-- For our purposes, the height in a trapezoid is defined as the length of the perpendicular segment from a point on one non-parallel side to the opposite parallel side.
def height (A B C D : Type) [hs : Trapezoid A B C D] : Prop :=
  sorry

theorem trapezoid_heights_equal (A B C D : Type) (hs : Trapezoid A B C D) :
  (height A B C D → height A B C D) :=
sorry

end trapezoid_heights_equal_l273_273181


namespace tom_books_l273_273491

-- Definitions based on the conditions
def joan_books : ℕ := 10
def total_books : ℕ := 48

-- The theorem statement: Proving that Tom has 38 books
theorem tom_books : (total_books - joan_books) = 38 := by
  -- Here we would normally provide a proof, but we use sorry to skip this.
  sorry

end tom_books_l273_273491


namespace percent_is_250_l273_273625

def part : ℕ := 150
def whole : ℕ := 60
def percent := (part : ℚ) / (whole : ℚ) * 100

theorem percent_is_250 : percent = 250 := 
by 
  sorry

end percent_is_250_l273_273625


namespace max_pieces_in_8x8_grid_l273_273812

theorem max_pieces_in_8x8_grid : 
  ∃ m n : ℕ, (m = 8) ∧ (n = 9) ∧ 
  (∀ H V : ℕ, (H ≤ n) → (V ≤ n) → 
   (H + V + 1 ≤ 16)) := sorry

end max_pieces_in_8x8_grid_l273_273812


namespace beth_score_l273_273469

-- Conditions
variables (B : ℕ)  -- Beth's points are some number.
def jan_points := 10 -- Jan scored 10 points.
def judy_points := 8 -- Judy scored 8 points.
def angel_points := 11 -- Angel scored 11 points.

-- First team has 3 more points than the second team
def first_team_points := B + jan_points
def second_team_points := judy_points + angel_points
def first_team_more_than_second := first_team_points = second_team_points + 3

-- Statement: Prove that B = 12
theorem beth_score : first_team_more_than_second → B = 12 :=
by
  -- Proof will be provided here
  sorry

end beth_score_l273_273469


namespace dot_product_eq_eleven_l273_273390

variable {V : Type _} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)
variable (cos_theta : ℝ)
variable (norm_a norm_b : ℝ)

axiom cos_theta_def : cos_theta = 1 / 3
axiom norm_a_def : ‖a‖ = 1
axiom norm_b_def : ‖b‖ = 3

theorem dot_product_eq_eleven
  (cos_theta_def : cos_theta = 1 / 3)
  (norm_a_def : ‖a‖ = 1)
  (norm_b_def : ‖b‖ = 3) :
  (2 • a + b) ⬝ b = 11 := 
sorry

end dot_product_eq_eleven_l273_273390


namespace minute_hand_rotation_l273_273448

theorem minute_hand_rotation (minutes : ℕ) (degrees_per_minute : ℝ) (radian_conversion_factor : ℝ) : 
  minutes = 10 → 
  degrees_per_minute = 360 / 60 → 
  radian_conversion_factor = π / 180 → 
  (-(degrees_per_minute * minutes * radian_conversion_factor) = -(π / 3)) := 
by
  intros hminutes hdegrees hfactor
  rw [hminutes, hdegrees, hfactor]
  simp
  sorry

end minute_hand_rotation_l273_273448


namespace arithmetic_geometric_sequences_geometric_sum_l273_273759

theorem arithmetic_geometric_sequences
  (a : ℕ → ℕ)
  (a_arith : ∃ d, ∀ n, a n = a 1 + (n - 1) * d)
  (a1_eq_2 : a 1 = 2)
  (geom_cond : ∃ r, a 4 * a 4 = a 2 * a 8):
  (∀ n, a n = 2) ∨ (∀ n, a n = 2 * n) :=
  sorry

theorem geometric_sum
  (b : ℕ → ℕ)
  (a : ℕ → ℕ)
  (b1_eq_a2 : b 1 = a 2)
  (b2_eq_a4 : b 2 = a 4)
  (a_case1 : ∀ n, a n = 2)
  (Sn_case1 : ∀ n, ∃ Sn, Sn = n * b 1)
  (a_case2 : ∀ n, a n = 2 * n)
  (Sn_case2 : ∀ n, ∃ Sn, Sn = 4 * (2 ^ n - 1)):
  (∀ n, ∃ Sn, Sn = n * b 1) ∨ (∀ n, ∃ Sn, Sn = 4 * (2 ^ n - 1)) :=
  sorry

end arithmetic_geometric_sequences_geometric_sum_l273_273759


namespace enclosed_area_approximation_l273_273074

noncomputable def g (x : ℝ) : ℝ := 1 - sqrt (1 - (x - 0.5)^2)

-- Define the problem and conditions
theorem enclosed_area_approximation : 
  let g := λ x : ℝ, 1 - sqrt (1 - (x - 0.5)^2) in 
  let area := 0.57 in
  ∃ x_lower x_upper : ℝ, x_lower = -1 ∧ x_upper = 2 ∧ (∫ x in x_lower..x_upper, g x) = area :=
sorry

end enclosed_area_approximation_l273_273074


namespace abs_sub_eq_three_l273_273503

theorem abs_sub_eq_three {m n : ℝ} (h1 : m * n = 4) (h2 : m + n = 5) : |m - n| = 3 := 
sorry

end abs_sub_eq_three_l273_273503


namespace truth_tellers_exactly_two_truth_tellers_l273_273134

variable (A B C D : Prop)

def statements (A B C D : Prop) :=
  (A ↔ ¬ (A ∧ B ∧ C ∧ D)) ∧  -- A: None of us did well
  (B ↔ (A ∨ B ∨ C ∨ D)) ∧    -- B: Some of us did well
  (C ↔ ¬ (B ∧ D)) ∧           -- C: At least one of B and D did not do well
  (D ↔ ¬ D)                   -- D: I did not do well

theorem truth_tellers (A B C D : Prop) (h : statements A B C D) : (A ∧ B ∧ C ∧ D) → false :=
  begin
    sorry
  end

theorem exactly_two_truth_tellers (A B C D : Prop) (h : statements A B C D) : (A ↔ false) → (B ↔ true) → (C ↔ true) → (D ↔ false) := 
  by
  intros hA hB hC hD
  sorry

end truth_tellers_exactly_two_truth_tellers_l273_273134


namespace cube_add_divisible_by_8_l273_273901

theorem cube_add_divisible_by_8 (n : ℕ) (h : n > 2) : (⌊(n^(1/3) + (n + 2)^(1/3))^3⌋ + 1) % 8 = 0 := 
by sorry

end cube_add_divisible_by_8_l273_273901


namespace find_value_of_x_squared_and_reciprocal_squared_l273_273444

theorem find_value_of_x_squared_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + (1/x)^2 = 2 := 
sorry

end find_value_of_x_squared_and_reciprocal_squared_l273_273444


namespace smallest_possible_value_of_other_integer_l273_273556

theorem smallest_possible_value_of_other_integer (x b : ℕ) (h_gcd_lcm : ∀ m n : ℕ, m = 36 → gcd m n = x + 5 → lcm m n = x * (x + 5)) : 
  b > 0 → ∃ b, b = 1 ∧ gcd 36 b = x + 5 ∧ lcm 36 b = x * (x + 5) := 
by {
   sorry 
}

end smallest_possible_value_of_other_integer_l273_273556


namespace cost_price_of_article_l273_273274

theorem cost_price_of_article (C MP SP : ℝ) (h1 : MP = 62.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) :
  C = 47.5 :=
sorry

end cost_price_of_article_l273_273274


namespace bolts_in_pack_l273_273312

theorem bolts_in_pack (packs_of_nuts : ℕ) (nuts_per_pack : ℕ) (total_nuts : ℕ) (cond1 : nuts_per_pack = 13) (cond2 : total_nuts = 104) (cond3 : total_nuts % nuts_per_pack = 0) : ∃ (b : ℕ), b ∈ [1, 2, 4, 8, 26, 52, 104] ∧ b ≠ 13 ∧ b = 52 :=
by {
  let factors := [1, 2, 4, 8, 26, 52, 104],
  have h1 : ∃ b ∈ factors, b ≠ 13 ∧ (total_nuts % b = 0) := sorry,
  cases h1 with b hb,
  use b,
  sorry
}

end bolts_in_pack_l273_273312


namespace students_sampled_count_l273_273028

theorem students_sampled_count :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 →
  (∃ k, 1 ≤ k ∧ k ≤ 20 ∧ 
  ∀ i, 601 ≤ 20*i - 5 ∧ 20*i - 5 ≤ 785 ↔ 31 ≤ i ∧ i ≤ 39) →
  (card {(i : ℕ) | 601 ≤ 20*i - 5 ∧ 20*i - 5 ≤ 785}) = 9 :=
by
  intros n hn h
  sorry

end students_sampled_count_l273_273028


namespace probability_correct_l273_273629

structure Bag :=
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

def marbles_drawn_sequence (bag : Bag) : ℚ :=
  let total_marbles := bag.blue + bag.green + bag.yellow
  let prob_blue_first := ↑bag.blue / total_marbles
  let prob_green_second := ↑bag.green / (total_marbles - 1)
  let prob_yellow_third := ↑bag.yellow / (total_marbles - 2)
  prob_blue_first * prob_green_second * prob_yellow_third

theorem probability_correct (bag : Bag) (h : bag = ⟨4, 6, 5⟩) : 
  marbles_drawn_sequence bag = 20 / 455 :=
by
  sorry

end probability_correct_l273_273629


namespace vector_parallel_addition_l273_273773

theorem vector_parallel_addition:
  ∀ (m : ℝ), (∃ m : ℝ, 1 * m + 4 = 0) →
  (3 * (1, -2) + 2 * (2, m) = (7, -14)) :=
by
  intros m h
  cases h with m hm
  have h1: m = -4 := sorry
  rw h1
  calc
    3 * (1, -2) + 2 * (2, -4)
      = 3 * (1, -2) + (4, -8) : by norm_num
  ... = (3 * 1, 3 * -2) + (4, -8) : by norm_num
  ... = (3, -6) + (4, -8) : by norm_num
  ... = (3 + 4, -6 + -8) : by norm_num
  ... = (7, -14) : by norm_num
  ... = (7, -14) : by norm_num
  sorry

end vector_parallel_addition_l273_273773


namespace mildred_oranges_left_l273_273095

def mildred_oranges_initial := 77.5
def father_eats := 2.25
def bella_takes := 3.75

theorem mildred_oranges_left : 
  mildred_oranges_initial - father_eats - bella_takes = 71.5 :=
by
  calc
    mildred_oranges_initial - father_eats - bella_takes = 77.5 - 2.25 - 3.75 := by rfl
    ... = 75.25 - 3.75 := by norm_num
    ... = 71.5 := by norm_num

end mildred_oranges_left_l273_273095


namespace wood_wasted_percentage_is_25_l273_273632

-- Definitions according to the given problem conditions.
def r_cone_base : ℝ := 9  -- radius of the cone base
def h_cone : ℝ := 9      -- height of the cone
def r_sphere : ℝ := 9    -- radius of the wooden sphere

-- volume of the cone
def V_cone : ℝ := (1 / 3) * Real.pi * (r_cone_base ^ 2) * h_cone

-- volume of the sphere
def V_sphere : ℝ := (4 / 3) * Real.pi * (r_sphere ^ 3)

-- percentage of wood wasted when the cone is carved out from the sphere
def percentage_wood_wasted : ℝ :=
  (V_cone / V_sphere) * 100

-- The target theorem to prove
theorem wood_wasted_percentage_is_25 : percentage_wood_wasted = 25 :=
by
  calc
  percentage_wood_wasted = ((1 / 3) * Real.pi * (9 ^ 2) * 9) / ((4 / 3) * Real.pi * (9 ^ 3)) * 100 : by sorry
                       ... = 25 : by sorry

end wood_wasted_percentage_is_25_l273_273632


namespace starting_player_can_ensure_integer_roots_l273_273923

theorem starting_player_can_ensure_integer_roots :
  ∃ (a b c : ℤ), ∀ (x : ℤ), (x^3 + a * x^2 + b * x + c = 0) →
  (∃ r1 r2 r3 : ℤ, x = r1 ∨ x = r2 ∨ x = r3) :=
sorry

end starting_player_can_ensure_integer_roots_l273_273923


namespace radhika_total_games_l273_273534

-- Define the conditions
def giftsOnChristmas := 12
def giftsOnBirthday := 8
def alreadyOwned := (giftsOnChristmas + giftsOnBirthday) / 2
def totalGifts := giftsOnChristmas + giftsOnBirthday
def expectedTotalGames := totalGifts + alreadyOwned

-- Define the proof statement
theorem radhika_total_games : 
  giftsOnChristmas = 12 ∧ giftsOnBirthday = 8 ∧ alreadyOwned = 10 
  ∧ totalGifts = 20 ∧ expectedTotalGames = 30 :=
by 
  sorry

end radhika_total_games_l273_273534


namespace find_constant_c_l273_273794

theorem find_constant_c (c : ℝ) (h : (x + 7) ∣ (c*x^3 + 19*x^2 - 3*c*x + 35)) : c = 3 := by
  sorry

end find_constant_c_l273_273794


namespace flies_on_sixth_lilypad_l273_273461

/-- Define the lily pads in terms of the flies they hold --/
def flies_on_lilypad (n : ℕ) : ℕ := 
match n with
| 0 => 85
| 1 => 40
| 2 => 55
| 3 => 50
| 4 => 65
| 5 => 0
| _ => 0
end

/-- Prove that the number of flies on the sixth lily pad (T_6) is 65
    given the distribution of flies among the lily pads and the conditions. --/
theorem flies_on_sixth_lilypad : flies_on_lilypad 5 = 65 :=
by
  /-- Summing the flies on the known lily pads --/
  let flies_sum := flies_on_lilypad 0 + flies_on_lilypad 1 + flies_on_lilypad 2 + 
                    flies_on_lilypad 3 + flies_on_lilypad 4
  /-- Assume the total frogs in the swamp --/
  let total_frogs := (flies_sum + flies_on_lilypad 5) / 10
  have H : flies_sum = 295 := by
    /-- Summing up directly --/
    exact rfl
  /-- Calculate the expected flies on T_6 --/
  have T_6 : flies_on_lilypad 5 = 10 * total_frogs - flies_sum := by
    sorry
  /-- Conclude T_6 equals 65 --/
  show flies_on_lilypad 5 = 65 from T_6

end flies_on_sixth_lilypad_l273_273461


namespace rearranged_number_divisible_by_27_l273_273029

theorem rearranged_number_divisible_by_27 (n m : ℕ) (hn : m = 3 * n) 
  (hdigits : ∀ a b : ℕ, (a ∈ n.digits 10 ↔ b ∈ m.digits 10)) : 27 ∣ m :=
sorry

end rearranged_number_divisible_by_27_l273_273029


namespace max_toads_l273_273032

def is_frog (coord : ℕ × ℕ) (frogs : set (ℕ × ℕ)) : Prop :=
  coord ∈ frogs

def is_toad (coord : ℕ × ℕ) (frogs : set (ℕ × ℕ)) : Prop :=
  coord ∉ frogs

def is_neighbor (coord1 coord2 : ℕ × ℕ) : Prop :=
  (coord1.1 = coord2.1 ∧ (coord1.2 + 1 = coord2.2 ∨ coord1.2 = coord2.2 + 1)) ∨
  (coord1.2 = coord2.2 ∧ (coord1.1 + 1 = coord2.1 ∨ coord1.1 = coord2.1 + 1))

theorem max_toads (frogs : set (ℕ × ℕ)) :
  (∀ coord ∈ frogs, ∃ neighbor, is_neighbor coord neighbor ∧ is_toad neighbor frogs) ∧
  (∀ coord ∉ frogs, ¬ ∃ neighbor, is_neighbor coord neighbor ∧ is_toad neighbor frogs) ∧
  frogs.card = 32 ∧
  (∀ coord, coord.1 < 8 ∧ coord.2 < 8) →
  64 - frogs.card = 32 :=
by
  sorry

end max_toads_l273_273032


namespace bob_total_spend_in_usd_l273_273282

theorem bob_total_spend_in_usd:
  let coffee_cost_yen := 250
  let sandwich_cost_yen := 150
  let yen_to_usd := 110
  (coffee_cost_yen + sandwich_cost_yen) / yen_to_usd = 3.64 := by
  sorry

end bob_total_spend_in_usd_l273_273282


namespace vector_dot_product_property_l273_273380

variables (a b : ℝ^3) (θ : ℝ)

open real_inner_product_space

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

def cos_theta (a b : ℝ^3) : ℝ := (a.dot_product b) / ((magnitude a) * (magnitude b))

theorem vector_dot_product_property
  (h1 : cos_theta a b = 1 / 3)
  (h2 : magnitude a = 1)
  (h3 : magnitude b = 3) :
  (2 • a + b) ∙ b = 11 :=
begin
  sorry
end

end vector_dot_product_property_l273_273380


namespace circle_problem_tangent_lines_l273_273755

noncomputable theory

def circle_equation (a b : ℝ) : Prop :=
  (a + b + 1 = 0) ∧
  ((-2 - a)^2 + (- b)^2 = 25) ∧
  ((5 - a)^2 + (1 - b)^2 = 25)

def tangent_line_equation (a b : ℝ) : Prop :=
  (x = -3 ∨ y = (8/15) * (x + 3))

theorem circle_problem (a b : ℝ) :
  circle_equation a b →
  ((x - a)^2 + (y - b)^2 = 25) :=
sorry

theorem tangent_lines (a b : ℝ) :
  circle_equation a b →
  tangent_line_equation a b →
  x = -3 ∨ y = (8 / 15) * (x + 3) :=
sorry

end circle_problem_tangent_lines_l273_273755


namespace diving_competition_score_l273_273608

def point_value_of_dive (scores : List ℝ) (degree_of_difficulty : ℝ) : ℝ :=
  let sorted_scores := scores.qsort (≤)
  let effective_scores := sorted_scores.drop 1 |>.dropLast 1
  (effective_scores.sum) * degree_of_difficulty

theorem diving_competition_score :
  point_value_of_dive [7.5, 8.3, 9.0, 6.0, 8.6] 3.2 = 78.08 :=
by {
  sorry
}

end diving_competition_score_l273_273608


namespace fraction_meaningful_l273_273167

theorem fraction_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ (f : ℝ → ℝ), f x = (x + 2) / (x - 1) :=
by
  sorry

end fraction_meaningful_l273_273167


namespace total_jumps_correct_l273_273616

-- Define Ronald's jumps
def Ronald_jumps : ℕ := 157

-- Define the difference in jumps between Rupert and Ronald
def difference : ℕ := 86

-- Define Rupert's jumps
def Rupert_jumps : ℕ := Ronald_jumps + difference

-- Define the total number of jumps
def total_jumps : ℕ := Ronald_jumps + Rupert_jumps

-- State the main theorem we want to prove
theorem total_jumps_correct : total_jumps = 400 := 
by sorry

end total_jumps_correct_l273_273616


namespace max_value_of_sum_on_ellipse_l273_273043

theorem max_value_of_sum_on_ellipse (x y : ℝ) (h : x^2 / 3 + y^2 = 1) : x + y ≤ 2 :=
sorry

end max_value_of_sum_on_ellipse_l273_273043


namespace max_area_triangle_AOB_l273_273824
-- Import the necessary library

-- Define the conditions given in the problem
variables (a b c : ℝ) (h1a : 0 < b) (h2a : b < a) (h3a : a = sqrt (3 + c^2)) (h4a : c = sqrt 2 / 2) (x y : ℝ)
variables (h5 : (2 * x + y^2 = 12) ∧  (x^2 ∈ set.Icc √2 3))
noncomputable def ellipse_standard_eq :=
  (∀ x y : ℝ,  x ^ 2 / 6 + y ^ 2 / 3 = 1)

-- Define the conditions for the points A, B, and M, and the line segment OP
variables (P : ℝ × ℝ) (h_P : P = (2, 1))
variables (A B : ℝ × ℝ) (h_A : x^2 / a^2 + y^2 / b^2 = 1)
variables (h_B : x^2 / a^2 + y^2 / b^2 = 1) (h_mid : x + y = 3)
variables (O M : ℝ × ℝ) (h_OM : M ∈ segment O P)

-- Declare the theorem proving the ellipse standard equation and the maximum area of triangle AOB
theorem max_area_triangle_AOB : 
  ellipse_standard_eq a b M⁻∧ 
  (dEq (area O A B) (3 * sqrt 3 / 2)) :=
sorry

end max_area_triangle_AOB_l273_273824


namespace perpendicular_AJ_BC_l273_273040

variables {A B C D E G H J : Point}
variables {O_A O_B O_C : Circle}

-- Definitions based on conditions
def excircle_tangent_to_sides (t : Triangle) (O_A O_B O_C: Circle) : Prop :=
  O_A.tangent_to t.sideBC ∧
  O_B.tangent_to t.sideAC ∧
  O_C.tangent_to t.sideAB

def excircle_tangent_points (O_A O_B O_C: Circle) 
(D E G H : Point) (t : Triangle) : Prop :=
  tangent_at O_A t.sideBC D ∧
  tangent_at O_B t.sideBC E ∧
  tangent_at O_C t.ext_sideAB G ∧
  tangent_at O_C t.ext_sideAC H

def intersection_point (D G H E J : Point) : Prop :=
  line DG ∩ line HE = J

-- Prove statement
theorem perpendicular_AJ_BC
  (t : Triangle)
  (O_A O_B O_C: Circle)
  (D E G H J : Point) :
  excircle_tangent_to_sides t O_A O_B O_C →
  excircle_tangent_points O_A O_B O_C D E G H t →
  intersection_point D G H E J →
  line AJ ⊥ t.sideBC :=
by sorry

end perpendicular_AJ_BC_l273_273040


namespace paperboy_delivery_ways_l273_273237

def D : ℕ → ℕ
| 0       := 1  -- base case to handle zero houses, though not originally specified
| 1       := 2
| 2       := 4
| 3       := 7
| (n + 1) := D n + D (n - 1) + D (n - 2)

theorem paperboy_delivery_ways : D 10 = 504 :=
by 
  have h₀ : D 0 = 1 := rfl
  have h₁ : D 1 = 2 := rfl
  have h₂ : D 2 = 4 := rfl
  have h₃ : D 3 = 7 := rfl
  have h₄ : D 4 = D 3 + D 2 + D 1 := rfl
  have h₄_value : D 4 = 13 := by 
    rw [h₃, h₂, h₁]
    norm_num
    done
  have h₅ : D 5 = D 4 + D 3 + D 2 := rfl
  have h₅_value : D 5 = 24 := by 
    rw [h₄_value, h₃, h₂]
    norm_num
    done
  have h₆ : D 6 = D 5 + D 4 + D 3 := rfl
  have h₆_value : D 6 = 44 := by 
    rw [h₅_value, h₄_value, h₃]
    norm_num
    done
  have h₇ : D 7 = D 6 + D 5 + D 4 := rfl
  have h₇_value : D 7 = 81 := by 
    rw [h₆_value, h₅_value, h₄_value]
    norm_num
    done
  have h₈ : D 8 = D 7 + D 6 + D 5 := rfl
  have h₈_value : D 8 = 149 := by 
    rw [h₇_value, h₆_value, h₅_value]
    norm_num
    done
  have h₉ : D 9 = D 8 + D 7 + D 6 := rfl
  have h₉_value : D 9 = 274 := by 
    rw [h₈_value, h₇_value, h₆_value]
    norm_num
    done
  have h₁₀ : D 10 = D 9 + D 8 + D 7 := rfl
  rw [h₁₀, h₉_value, h₈_value, h₇_value]
  norm_num
  done


end paperboy_delivery_ways_l273_273237


namespace odd_n_no_div_m_pow_n_minus_1_plus_1_l273_273081

theorem odd_n_no_div_m_pow_n_minus_1_plus_1 {n m : ℕ} (hn : n > 1) (hodd_n : odd n) (hm : m ≥ 1) :
  ¬ n ∣ m^(n - 1) + 1 := 
sorry

end odd_n_no_div_m_pow_n_minus_1_plus_1_l273_273081


namespace smallest_sum_proof_l273_273159

theorem smallest_sum_proof (N : ℕ) (p : ℝ) (h1 : 6 * N = 2022) (hp : p > 0) : (N * 1 = 337) :=
by 
  have hN : N = 2022 / 6 := by 
    sorry
  exact hN

end smallest_sum_proof_l273_273159


namespace diagonal_pairs_forming_60_degrees_l273_273342

theorem diagonal_pairs_forming_60_degrees :
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 :=
by 
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  have calculation : total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 := sorry
  exact calculation

end diagonal_pairs_forming_60_degrees_l273_273342


namespace gina_makes_36_08_per_hour_l273_273344

noncomputable def gina_hourly_wage : ℝ :=
  let gina_rate_rose : ℝ := 6
  let gina_rate_lily : ℝ := 7
  let gina_rate_sunflower : ℝ := 5
  let gina_rate_orchid : ℝ := 8

  let order1_roses : ℝ := 6
  let order1_lilies : ℝ := 14
  let order1_sunflowers : ℝ := 4
  let order1_payment : ℝ := 120

  let order2_orchids : ℝ := 10
  let order2_roses : ℝ := 2
  let order2_payment : ℝ := 80

  let order3_sunflowers : ℝ := 8
  let order3_orchids : ℝ := 4
  let order3_payment : ℝ := 70

  let order1_time := (order1_roses / gina_rate_rose) + (order1_lilies / gina_rate_lily) + (order1_sunflowers / gina_rate_sunflower)
  let order2_time := (order2_orchids / gina_rate_orchid) + (order2_roses / gina_rate_rose)
  let order3_time := (order3_sunflowers / gina_rate_sunflower) + (order3_orchids / gina_rate_orchid)

  let total_time := order1_time + order2_time + order3_time
  let total_payment := order1_payment + order2_payment + order3_payment
  
  total_payment / total_time

theorem gina_makes_36_08_per_hour : gina_hourly_wage ≈ 36.08 := 
by sorry

end gina_makes_36_08_per_hour_l273_273344


namespace triangle_area_l273_273261

-- Given conditions
def ratio_5_12_13 (a b c : ℝ) : Prop := 
  (b / a = 12 / 5) ∧ (c / a = 13 / 5)

def right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

def circumscribed_circle (a b c r : ℝ) : Prop :=
  2 * r = c

-- The main theorem we need to prove
theorem triangle_area (a b c r : ℝ) (h_ratio : ratio_5_12_13 a b c) (h_triangle : right_triangle a b c) (h_circle : circumscribed_circle a b c r) (h_r : r = 5) :
  0.5 * a * b ≈ 17.75 :=
by
  sorry

end triangle_area_l273_273261


namespace boat_length_is_three_l273_273215

-- Define the given conditions
def breadth_of_boat : ℝ := 2
def sink_height : ℝ := 0.01
def mass_of_man : ℝ := 60
def density_of_water : ℝ := 1000
def gravity : ℝ := 9.81

-- Statement of the theorem
theorem boat_length_is_three :
  let V := mass_of_man * gravity / (density_of_water * gravity) in
  let length := V / (breadth_of_boat * sink_height) in
  length = 3 :=
by
  -- This is where the proof would go
  sorry

end boat_length_is_three_l273_273215


namespace find_r_l273_273499

theorem find_r (a b m p r : ℝ) (h1 : a * b = 4)
  (h2 : ∃ (q w : ℝ), (a + 2 / b = q ∧ b + 2 / a = w) ∧ q * w = r) :
  r = 9 :=
sorry

end find_r_l273_273499


namespace find_QNR_l273_273582

-- Definitions for the problem
def isosceles_triangle (P Q R : Type) [geometry P Q R] (PR QR : Real) : Prop := PR = QR
def angle (x : ℝ) : ℝ := x

variables {P Q R N : Type} [geometry P Q R]

-- Conditions of the problem
variables (PR QR : Real) (PRQ PNR PRN : ℝ)
let isosceles : Prop := isosceles_triangle P Q R PR QR
let angle_PRQ := angle 108
let angle_PNR := angle 9
let angle_PRN := angle 27

-- Translate the question into Lean
theorem find_QNR (h1 : isosceles) (h2 : PRQ = 108) (h3 : PNR = 9) (h4 : PRN = 27) : 
  ∃ QNR, QNR = 63 :=
sorry

end find_QNR_l273_273582


namespace remainder_43_pow_97_pow_5_plus_109_mod_163_l273_273600

theorem remainder_43_pow_97_pow_5_plus_109_mod_163 :
    (43 ^ (97 ^ 5) + 109) % 163 = 50 :=
by
  sorry

end remainder_43_pow_97_pow_5_plus_109_mod_163_l273_273600


namespace proof_problem_l273_273498

variables {R : Type*} [inner_product_space ℝ R] {u v1 v2 w : R}

-- Assume u, v1, and v2 are unit vectors and mutually orthogonal
variables (hu : ∥u∥ = 1) (hv1 : ∥v1∥ = 1) (hv2 : ∥v2∥ = 1)
          (horth1 : ⟪u, v1⟫ = 0) (horth2 : ⟪u, v2⟫ = 0) (horth3 : ⟪v1, v2⟫ = 0)

-- Assume w such that u × (v1 + v2) + u = w
variable (hW : u × (v1 + v2) + u = w)

-- Assume w × u = v1 + v2
variable (hWcrossU : w × u = v1 + v2)

-- Prove that u ⋅ (v1 + v2 × w) = 1
theorem proof_problem : ⟪u, v1 + v2 × w⟫ = 1 :=
sorry

end proof_problem_l273_273498


namespace problem1_problem2_l273_273204

-- Problem (1)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 2) : 
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 := 
by 
  sorry

-- Problem (2)
theorem problem2 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = -4 / 3 := 
by
  sorry

end problem1_problem2_l273_273204


namespace find_b_l273_273347

theorem find_b (a b : ℝ) (x : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^5) * x^5):
  b = 40 :=
  sorry

end find_b_l273_273347


namespace polygon_sides_l273_273951

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 1260) : ∃ n : ℕ, (n-2) * 180 = sum_of_interior_angles ∧ n = 9 :=
by {
  sorry
}

end polygon_sides_l273_273951


namespace donuts_per_student_l273_273886

theorem donuts_per_student (total_donuts : ℕ) (students : ℕ) (percentage_likes_donuts : ℚ) 
    (H1 : total_donuts = 4 * 12) 
    (H2 : students = 30) 
    (H3 : percentage_likes_donuts = 0.8) 
    (H4 : ∃ (likes_donuts : ℕ), likes_donuts = students * percentage_likes_donuts) : 
    (∃ (donuts_per_student : ℚ), donuts_per_student = total_donuts / (students * percentage_likes_donuts)) → donuts_per_student = 2 :=
by
    sorry

end donuts_per_student_l273_273886


namespace max_value_of_sum_l273_273500

open Real

theorem max_value_of_sum (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 3) :
  (ab / (a + b) + bc / (b + c) + ca / (c + a)) ≤ 3 / 2 :=
sorry

end max_value_of_sum_l273_273500


namespace ellipse_equation_and_fixed_point_l273_273399

theorem ellipse_equation_and_fixed_point :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a = sqrt 3 ∧ b = 1) ∧ -- Conditions from the problem
  (∀ x y : ℝ, (x = 1 ∧ y = sqrt 6 / 3) → x^2 / a^2 + y^2 / b^2 = 1) ∧ -- Equation of the ellipse passing through a point
  (eccentricity = sqrt 6 / 3 → a^2 = b^2 + (b * sqrt(6) / 3)^2) ∧ -- Condition involving eccentricity 
  (∀ l : ℝ→ ℝ, ∀ P Q : ℝ × ℝ, 
    (\overrightarrow{AP} • \overrightarrow{AQ} = 0) → 
    ∃ c : ℝ, ∀ x : ℝ, (l x = (x, c)) ∧ c = -1/2) := sorry

end ellipse_equation_and_fixed_point_l273_273399


namespace quadratic_polynomial_positive_a_l273_273733

variables {a b c n : ℤ}
def p (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem quadratic_polynomial_positive_a (h : a ≠ 0) 
    (hn : n < p n) (hn_ppn : p n < p (p n)) (hn_pppn : p (p n) < p (p (p n))) :
    a > 0 :=
by
    sorry

end quadratic_polynomial_positive_a_l273_273733


namespace find_omega_and_monotonicity_l273_273404

noncomputable
def f (ω x : ℝ) := 4 * cos (ω * x) * sin (ω * x + (real.pi / 4))

theorem find_omega_and_monotonicity (ω : ℝ) (hω : ω > 0) 
  (hT : ∀ x, f ω (x + real.pi) = f ω x) :
  (ω = 1) ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ (real.pi / 8) → f ω x ≤ f ω (x + (real.pi / 8))) ∧ 
  (∀ x, (real.pi / 8) ≤ x ∧ x ≤ (real.pi / 2) → f ω x ≥ f ω (x + (real.pi / 8))) :=
sorry

end find_omega_and_monotonicity_l273_273404


namespace fraction_meaningful_l273_273166

theorem fraction_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ (f : ℝ → ℝ), f x = (x + 2) / (x - 1) :=
by
  sorry

end fraction_meaningful_l273_273166


namespace hyperbola_conjugate_axis_length_l273_273126

theorem hyperbola_conjugate_axis_length : 
  (2 : ℝ) = 2 :=
by
  let a : ℝ := 2
  let b : ℝ := 1
  have h : 2 * b = 2 := by sorry
  exact h

end hyperbola_conjugate_axis_length_l273_273126


namespace shaded_area_equals_2_l273_273927

open Real

def semicircle_area (r : ℝ) : ℝ := π * r^2 / 2
def quartercircle_area (r : ℝ) : ℝ := π * r^2 / 4
def triangle_area (a b : ℝ) : ℝ := a * b / 2

variables (O P R : ℝ)
variables (h1 : P * P + R * R = 8)
variables (h2 : P = 2)
variables (h3 : R = 2)
variables (h4 : 2 * 2 = 4)
variables (h5 : 2 * sqrt(2) / 2 = sqrt(2))

theorem shaded_area_equals_2 :
  let semicircle_PR := semicircle_area (sqrt 2)
  let quarter_circle := quartercircle_area 2
  let triangle_OPR := triangle_area 2 2
  let segment_area := quarter_circle - triangle_OPR 
  segment_area - semicircle_PR = 2 := 
by {
  let semicircle_PR := semicircle_area (sqrt 2),
  let quarter_circle := quartercircle_area 2,
  let triangle_OPR := triangle_area 2 2,
  let segment_area := quarter_circle - triangle_OPR,
  calc
  segment_area - semicircle_PR
        = (quarter_circle - triangle_OPR - semicircle_PR) : by linarith
    ... = (π * 2^2 / 4 - 2 - π * (sqrt 2)^2 / 2) : by sorry
    ... = (π - 2 - π) : by sorry
    ... = 2 : by sorry,
}

end shaded_area_equals_2_l273_273927


namespace number_can_be_sum_of_distinct_primes_l273_273836

theorem number_can_be_sum_of_distinct_primes (n : ℕ) (h : n ≥ 7 ∧ n ≠ 1 ∧ n ≠ 4 ∧ n ≠ 6) :
  ∃ (p : ℕ → Prop), (∀ n, ∃ p, is_prime p ∧ n ≤ p ∧ p ≤ 2 * n - 7) → 
  ∃ (S : finset ℕ), (∀ x ∈ S, is_prime x ∧ S.sum = n) := 
sorry

end number_can_be_sum_of_distinct_primes_l273_273836


namespace jason_eggs_each_morning_l273_273695

theorem jason_eggs_each_morning (total_eggs : ℕ) (days : ℕ) (eggs_per_day : ℕ) : 
  total_eggs = 42 → days = 14 → eggs_per_day = total_eggs / days → eggs_per_day = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end jason_eggs_each_morning_l273_273695


namespace triangle_area_approx_l273_273264

noncomputable def radius := 5
noncomputable def ratio_side1 := 5
noncomputable def ratio_side2 := 12
noncomputable def ratio_side3 := 13
noncomputable def hypotenuse := 2 * radius

theorem triangle_area_approx : ∃ (area : ℝ), 
  (∀ x : ℝ, 
    13 * x = hypotenuse → 
    area = (1/2) * (5 * x) * (12 * x)
  ) ∧ 
  abs(area - 17.75) < 0.01 :=
by
  sorry

end triangle_area_approx_l273_273264


namespace cost_of_each_book_l273_273605

theorem cost_of_each_book
  (num_whale_books : ℕ := 9)
  (num_fish_books : ℕ := 7)
  (num_magazines : ℕ := 3)
  (magazine_cost : ℕ := 1)
  (total_spent : ℕ := 179) :
  let x := 11 in
  (num_whale_books + num_fish_books) * x + num_magazines * magazine_cost = total_spent :=
by
  -- Assume x = 11 as the cost of each book
  let x : ℕ := 11
  calc
    (num_whale_books + num_fish_books) * x + num_magazines * magazine_cost
    = (9 + 7) * x + 3 * magazine_cost : by sorry
    ... = 16 * x + 3 * magazine_cost : by sorry
    ... = 16 * 11 + 3 * 1 : by sorry
    -- Calculate the left-hand side
    ... = 176 + 3 : by sorry
    -- Calculate the sum
    ... = 179 : by sorry

-- The above code should compile successfully in Lean 4 and states that, given the conditions, the cost of each book is \$11.

end cost_of_each_book_l273_273605


namespace exists_sequence_satisfying_conditions_l273_273524

theorem exists_sequence_satisfying_conditions (n : ℕ) (H : n > 1000) :
  ∃ (a : ℕ → ℤ), -- Sequence exists
  (∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ {-1, 0, 1}) ∧ -- Each number is -1, 0 or 1
  (∑ i in finset.range n, if a (i+1) = 0 then 0 else 1) ≥ (2 * n / 5) ∧ -- At least (2n/5) non-zero
  (∑ i in finset.range n, a (i+1) / (i + 1)) = 0 := -- Sum equal to 0
sorry -- proof to be filled out

end exists_sequence_satisfying_conditions_l273_273524


namespace symmetric_points_x_axis_l273_273800

theorem symmetric_points_x_axis (y x : ℝ) 
  (h1 : (-2, y) = (x, 3) ∨ (-2, y) = (x, -3)) : 
  x + y = -5 :=
  by {
    cases h1;
    {
      simpa using h1,
    },
    {
      simpa using h1,
  },
  sorry -- placeholder for completion of the proof
}

end symmetric_points_x_axis_l273_273800


namespace badminton_costs_l273_273121

variables (x : ℕ) (h : x > 16)

-- Define costs at Store A and Store B
def cost_A : ℕ := 1760 + 40 * x
def cost_B : ℕ := 1920 + 32 * x

-- Lean statement to prove the costs
theorem badminton_costs : 
  cost_A x = 1760 + 40 * x ∧ cost_B x = 1920 + 32 * x :=
by {
  -- This proof is expected but not required for the task
  sorry
}

end badminton_costs_l273_273121


namespace largest_digit_divisible_by_6_l273_273986

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l273_273986


namespace no_closed_non_self_intersecting_polygon_possible_l273_273146

theorem no_closed_non_self_intersecting_polygon_possible 
  (segments : list (fin 2 → ℝ × ℝ)) 
  (h_non_intersecting : ∀ s1 s2 ∈ segments, s1 ≠ s2 → disjoint (set.range s1) (set.range s2)) :
  ¬ ∀ subset : finset (fin 2 → ℝ × ℝ), 
    (subset ⊆ segments) → 
    ∃ additional_segments : list (ℝ × ℝ), 
    closed_non_self_intersecting_polygon (subset.to_list ++ additional_segments) := 
sorry

end no_closed_non_self_intersecting_polygon_possible_l273_273146


namespace select_50_numbers_l273_273721

theorem select_50_numbers
  (x : Fin 100 → ℝ)
  (h_sum : ∑ i, x i = 1)
  (h_diff : ∀ i : Fin 99, |x i - x (i + 1)| < 1 / 50) :
  ∃ (S : Finset (Fin 100)), S.card = 50 ∧ |∑ i in S, x i - 1 / 2| < 1 / 100 :=
sorry

end select_50_numbers_l273_273721


namespace common_tangents_for_circles_l273_273400

theorem common_tangents_for_circles :
  ∀ (λ : ℝ), λ ≠ 0 → 
  ∃ (k1 k2 : ℝ), (k1 = 1 ∧ k2 = 7) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 2 * λ * x - 4 * λ * y + (9 / 2) * λ^2 = 0 → 
  ((y = k1 * x) ∨ (y = k2 * x))) :=
by sorry

end common_tangents_for_circles_l273_273400


namespace parabola_translation_l273_273562

theorem parabola_translation:
  ∀ (x: ℝ), (∃ a b: ℝ, (a = 3 ∧ b = -1) ∧ 
  (3 * (x - 3)^2 + b = 3 * x^2 - 1)) :=
begin
  sorry

end parabola_translation_l273_273562


namespace even_fn_decreasing_on_interval_implies_order_l273_273678

noncomputable def f : ℝ → ℝ := sorry 

theorem even_fn_decreasing_on_interval_implies_order (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_symmetric : ∀ x : ℝ, f (2 - x) = f x)
  (h_decreasing : ∀ x y : ℝ, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f y < f x)
  (α β : ℝ)
  (h_acute : 0 < α ∧ α < real.pi / 2 ∧ 0 < β ∧ β < real.pi / 2)
  (h_obtuse_sum : α + β < real.pi / 2) :
  f (real.sin α) < f (real.cos β) :=
sorry

end even_fn_decreasing_on_interval_implies_order_l273_273678


namespace calc_2a_plus_b_dot_b_l273_273376

open Real

variables (a b : ℝ^3)
variables (cos_angle : ℝ)
variables (norm_a norm_b : ℝ)

-- Conditions from the problem
axiom cos_angle_is_1_3 : cos_angle = 1 / 3
axiom norm_a_is_1 : ∥a∥ = 1
axiom norm_b_is_3 : ∥b∥ = 3

-- Dot product calculation
axiom a_dot_b_is_1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle

-- Prove the target
theorem calc_2a_plus_b_dot_b : 
  (2 * a + b).dot b = 11 :=
by
  -- These assertions setup the problem statement in Lean
  have h1 : a.dot b = ∥a∥ * ∥b∥ * cos_angle, 
  from a_dot_b_is_1,
  have h2 : ∥a∥ = 1, from norm_a_is_1,
  have h3 : ∥b∥ = 3, from norm_b_is_3,
  have h4 : cos_angle = 1 / 3, from cos_angle_is_1_3,
  sorry

end calc_2a_plus_b_dot_b_l273_273376


namespace find_f_expression_l273_273349

variable {F : Type*}

noncomputable def f (x : F) : F := sorry

theorem find_f_expression (x : F) (h : x ≠ 0 ∧ x ≠ -1) :
  f(x) = x / (1 + x) :=
by
  sorry

end find_f_expression_l273_273349


namespace find_d_l273_273327

theorem find_d {x d : ℤ} (h : (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5 = (x + 4) + 6) : d = 37 :=
sorry

end find_d_l273_273327


namespace jerry_weekly_earnings_l273_273488

theorem jerry_weekly_earnings:
  (tasks_per_day: ℕ) 
  (daily_earnings: ℕ)
  (weekly_earnings: ℕ) :
  (tasks_per_day = 10 / 2) ∧
  (daily_earnings = 40 * tasks_per_day) ∧
  (weekly_earnings = daily_earnings * 7) →
  weekly_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l273_273488


namespace number_of_elements_in_A_oplus_B_l273_273676

-- Define the sets A and B
def A : Set ℚ := {1, 2, 4}
def B : Set ℚ := {2, 4, 8}

-- Define the operation A ⊕ B
def A_oplus_B : Set ℚ := {x | ∃ m ∈ A, ∃ n ∈ B, x = m / n}

-- The main theorem to prove
theorem number_of_elements_in_A_oplus_B : (A_oplus_B.toFinset.card = 5) :=
sorry

end number_of_elements_in_A_oplus_B_l273_273676


namespace not_all_numbers_less_than_500_after_operations_l273_273893

noncomputable def initial_sum_of_squares (n : ℕ) : ℝ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem not_all_numbers_less_than_500_after_operations : 
  let initial_sum := initial_sum_of_squares 2000
  initial_sum > 10000 * 500^2 :=
by
  let n := 2000
  let m := 10000
  let b := 500
  have initial_sum_eq : initial_sum_of_squares n = (n * (n + 1) * (2 * n + 1)) / 6 := rfl
  have sq_sum_gt := calc
      (n * (n + 1) * (2 * n + 1)) / 6 > (2000 * 2000 * 4000) / 6 : by sorry
      (2000 * 2000 * 4000) / 6 = 80 * 10^4 * 10^3 / 3 : by sorry
      80 * 10^4 * 10^3 / 3 = 80 * 10^7 / 3 : by sorry
      80 * 10^7 / 3 = 2.667 * 10^8 : by sorry
      2.667 * 10^8 > 2.5 * 10^7 : by sorry
  have max_possible_sum := m * b^2
  show initial_sum > max_possible_sum
  sorry

end not_all_numbers_less_than_500_after_operations_l273_273893


namespace number_of_whole_numbers_in_intervals_l273_273787

theorem number_of_whole_numbers_in_intervals : 
  let interval_start := (5 / 3 : ℝ)
  let interval_end := 2 * Real.pi
  ∃ n : ℕ, interval_start < ↑n ∧ ↑n < interval_end ∧ (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) ∧ 
  (∀ m : ℕ, interval_start < ↑m ∧ ↑m < interval_end → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6)) :=
sorry

end number_of_whole_numbers_in_intervals_l273_273787


namespace distance_5_km_times_l273_273169

noncomputable def time_when_distance_is_5km (t : ℝ) : Prop :=
  let cyclist_distance_from_intersection := 8 - (1/3) * t in
  let motorcyclist_distance_from_intersection := 15 - t in
  let distance_between := real.sqrt ((cyclist_distance_from_intersection)^2 + (motorcyclist_distance_from_intersection)^2) in
  distance_between = 5

theorem distance_5_km_times : ∃ (t : ℝ), t = 12 ∨ t = 19.8 :=
by
  use 12
  left
  sorry
  use 19.8
  right
  sorry

end distance_5_km_times_l273_273169


namespace mean_problem_l273_273447

theorem mean_problem (x y z : ℝ) (h : (28 + x + 42 + y + 78 + 104) / 6 = 62) :
  ((48 + 62 + 98 + z + 124 + (x + y)) / 6 = (452 + z) / 6) ∧
  ((x + y) / 2 = 60) :=
by
  -- Given condition
  have h1 : 28 + x + 42 + y + 78 + 104 = 372,
  { sorry }, -- This step involves algebraic manipulation
  -- Calculate x + y
  have h2 : x + y = 120,
  { sorry }, -- From the given mean equation
  -- Calculate the mean of new set
  have h3 : (48 + 62 + 98 + z + 124 + (x + y)) / 6 = (452 + z) / 6,
  { sorry },
  -- Verify the mean of x and y
  have h4 : (x + y) / 2 = 60,
  { sorry },
  exact ⟨h3, h4⟩

end mean_problem_l273_273447


namespace geometry_problem_l273_273747

/-
Given:
1. Points O and N are in the plane of Δ ABC.
2. |OA| = |OB| = |OC|.
3. NA + NB + NC = 0.
Prove:
- Point O is the circumcenter of Δ ABC.
- Point N is the centroid of Δ ABC.
-/

variables {Point : Type} [inner_product_space ℝ Point]
variables (A B C O N : Point)

def is_circumcenter (O A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

def is_centroid (N A B C : Point) : Prop :=
  (N - A) + (N - B) + (N - C) = 0

theorem geometry_problem
  (hO : is_circumcenter O A B C)
  (hN : is_centroid N A B C) :
  is_circumcenter O A B C ∧ is_centroid N A B C :=
by {
  exact ⟨hO, hN⟩,
}

end geometry_problem_l273_273747


namespace triangle_area_approx_l273_273265

noncomputable def radius := 5
noncomputable def ratio_side1 := 5
noncomputable def ratio_side2 := 12
noncomputable def ratio_side3 := 13
noncomputable def hypotenuse := 2 * radius

theorem triangle_area_approx : ∃ (area : ℝ), 
  (∀ x : ℝ, 
    13 * x = hypotenuse → 
    area = (1/2) * (5 * x) * (12 * x)
  ) ∧ 
  abs(area - 17.75) < 0.01 :=
by
  sorry

end triangle_area_approx_l273_273265


namespace number_of_correct_statements_l273_273561

theorem number_of_correct_statements :
  (irrational_numbers_are_infinite_decimals : ∀ x, irrational x → infinite_decimal x) ∧
  (square_root_of_4_is_pm_2 : sqrt 4 = 2 ∨ sqrt 4 = -2) ∧
  (three_numbers_cube_root_equal_themselves : set.count {x | ∃ y, y^3 = x ∧ y = x} = 3) ∧
  (one_to_one_correspondence_points_and_real_numbers : ∀ x, point_on_number_line x ↔ real x) →
  ∃ n, n = 4 :=
sorry

end number_of_correct_statements_l273_273561


namespace other_solution_zero_l273_273443

theorem other_solution_zero (c : ℝ) (h : 1^2 - 1 + c = 0) : ∃ x₂ : ℝ, x₂ = 0 :=
by
  have h₁ : 1 = 1 := rfl
  have sum_roots : 1 + x₂ = 1 := by
    rw [h₁]
  use 0
  linarith
  sorry

end other_solution_zero_l273_273443


namespace probability_yellow_straight_l273_273455

open ProbabilityTheory

-- Definitions and assumptions
def P_G : ℚ := 2 / 3
def P_S : ℚ := 1 / 2
def P_Y : ℚ := 1 - P_G

lemma prob_Y_and_S (independence : Indep P_Y P_S) : P_Y * P_S = 1 / 6 :=
by
  have P_Y : ℚ := 1 - P_G
  have P_S : ℚ := 1 / 2
  calc P_Y * P_S = (1 - P_G) * P_S : by rfl
             ... = (1 - 2 / 3) * 1 / 2 : by rfl
             ... = 1 / 3 * 1 / 2 : by rfl
             ... = 1 / 6 : by norm_num

-- theorem to prove
theorem probability_yellow_straight (independence : Indep P_Y P_S) : P_Y * P_S = 1 / 6 :=
prob_Y_and_S independence

end probability_yellow_straight_l273_273455


namespace arc_length_of_y_eq_2_plus_cosh_x_is_sinh_1_l273_273667

def hyperbolic_arc_length : ℝ :=
  ∫ x in 0..1, Real.cosh x

theorem arc_length_of_y_eq_2_plus_cosh_x_is_sinh_1 :
  hyperbolic_arc_length = Real.sinh 1 :=
by
  sorry

end arc_length_of_y_eq_2_plus_cosh_x_is_sinh_1_l273_273667


namespace vector_dot_product_property_l273_273382

variables (a b : ℝ^3) (θ : ℝ)

open real_inner_product_space

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

def cos_theta (a b : ℝ^3) : ℝ := (a.dot_product b) / ((magnitude a) * (magnitude b))

theorem vector_dot_product_property
  (h1 : cos_theta a b = 1 / 3)
  (h2 : magnitude a = 1)
  (h3 : magnitude b = 3) :
  (2 • a + b) ∙ b = 11 :=
begin
  sorry
end

end vector_dot_product_property_l273_273382


namespace pascal_sum_difference_l273_273540

open BigOperators

noncomputable def a_i (i : ℕ) := Nat.choose 3005 i
noncomputable def b_i (i : ℕ) := Nat.choose 3006 i
noncomputable def c_i (i : ℕ) := Nat.choose 3007 i

theorem pascal_sum_difference :
  (∑ i in Finset.range 3007, (b_i i) / (c_i i)) - (∑ i in Finset.range 3006, (a_i i) / (b_i i)) = 1 / 2 := by
  sorry

end pascal_sum_difference_l273_273540


namespace tan_alpha_l273_273751

theorem tan_alpha (α : ℝ) (h : sin α + sqrt 2 * cos α = sqrt 3) : tan α = sqrt 2 / 2 := 
by sorry

end tan_alpha_l273_273751


namespace books_about_fish_l273_273193

theorem books_about_fish (F : ℕ) (spent : ℕ) (cost_whale_books : ℕ) (cost_magazines : ℕ) (cost_fish_books_per_unit : ℕ) (whale_books : ℕ) (magazines : ℕ) :
  whale_books = 9 →
  magazines = 3 →
  cost_whale_books = 11 →
  cost_magazines = 1 →
  spent = 179 →
  99 + 11 * F + 3 = spent → F = 7 :=
by
  sorry

end books_about_fish_l273_273193


namespace axis_of_symmetry_is_2_l273_273433

def symmetric_function (f : ℝ → ℝ) := ∀ x, f(x) = f(4 - x)

theorem axis_of_symmetry_is_2 (f : ℝ → ℝ) (h : symmetric_function f) : 
  ∀ x, f(x) = f(2 + (2 - x)) :=
by
  sorry

end axis_of_symmetry_is_2_l273_273433


namespace intersection_points_sum_l273_273555

theorem intersection_points_sum :
  let f := fun x => x^3 - 3*x + 1
  let g := fun x => (-x/3 + 1)
  (∀ (x y : ℝ), y = f x ↔ y = g x) →
  let roots := {x : ℝ | ∃ y : ℝ, (y = f x ∧ y = g x)}
  ∑ x ∈ roots, x = 0 ∧ ∑ y ∈ (roots.map g).to_finset, y = 3 := by
  sorry

end intersection_points_sum_l273_273555


namespace each_friend_gave_6_dollars_l273_273704

theorem each_friend_gave_6_dollars (total_money : ℕ) (num_friends : ℕ)
  (h1 : total_money = 30) (h2 : num_friends = 5) :
  total_money / num_friends = 6 :=
by 
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end each_friend_gave_6_dollars_l273_273704


namespace books_left_over_after_repacking_l273_273807

theorem books_left_over_after_repacking :
  ((1335 * 39) % 40) = 25 :=
sorry

end books_left_over_after_repacking_l273_273807


namespace volume_of_cone_formed_by_half_sector_l273_273235

-- Variables defining the problem
variables (r_half_sector : ℝ) (l : ℝ)

-- The given conditions
def problem_conditions : Prop :=
  r_half_sector = 6 ∧ l = 6 -- The radius of the half-sector and the slant height (same as original circle's radius)

-- The theorem to prove
theorem volume_of_cone_formed_by_half_sector
  (h_r : r_half_sector = 6)
  (h_l : l = 6) : 
  let r := 3 in let h := 3 * Real.sqrt 3 in
  (1 / 3) * Real.pi * r^2 * h = 3 * Real.pi * Real.sqrt 3 :=
sorry

end volume_of_cone_formed_by_half_sector_l273_273235


namespace probability_at_least_three_copresidents_l273_273143

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  if h : k ≤ n then (n.choose k) else 0

/-- The probability of selecting at least three co-presidents when a club is randomly chosen and 
then four members are randomly selected from the club -/
theorem probability_at_least_three_copresidents :
  let p1 := (4 * (10 - 4) + 1) / (binomial_coeff 10 4) in
  let p2 := (4 * (12 - 4) + 1) / (binomial_coeff 12 4) in
  let p3 := (4 * (15 - 4) + 1) / (binomial_coeff 15 4) in
  (1 / 3) * (p1 + p2 + p3) ≈ 0.035 :=
by sorry

end probability_at_least_three_copresidents_l273_273143


namespace find_b_l273_273348

theorem find_b (a b : ℝ) (x : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^5) * x^5):
  b = 40 :=
  sorry

end find_b_l273_273348


namespace monotonic_intervals_range_of_x_range_of_a_l273_273409

section Problem1

def f (x : ℝ) : ℝ := x^3 - 6*x - 1
def f' (x : ℝ) : ℝ := 3*x^2 - 6

theorem monotonic_intervals :
  (∀ x, x < -sqrt 2 ∨ x > sqrt 2 → f'(x) > 0) ∧ 
  (∀ x, -sqrt 2 < x ∧ x < sqrt 2 → f'(x) < 0) :=
sorry

end Problem1

section Problem2

def g (x a : ℝ) : ℝ := 3*x^2 - a*x + 3*a - 3

theorem range_of_x {a : ℝ} (ha : -1 ≤ a ∧ a ≤ 1) :
  ∀ x, g x a < 0 → 0 < x ∧ x < 1/3 :=
sorry

end Problem2

section Problem3

def g' (x : ℝ) : ℝ := 6*x - a
def h (x : ℝ) : ℝ := 6*x + (ln x / x)

theorem range_of_a (x : ℝ) (hx : x ≥ 2) :
  (∀ x, x * g' x + ln x > 0) → a < 12 + ln 2 / 2 :=
sorry

end Problem3

end monotonic_intervals_range_of_x_range_of_a_l273_273409


namespace volume_of_smaller_cube_l273_273253

theorem volume_of_smaller_cube
  (edge_length : ℝ)
  (inscribed_sphere : Prop)
  (inscribed_smaller_cube : Prop)
  (h1 : edge_length = 12)
  (h2 : inscribed_sphere = (sphere.diameter = edge_length))
  (h3 : inscribed_smaller_cube = (smaller_cube.space_diagonal = sphere.diameter)) :
  ∃ V : ℝ, V = 192 * real.sqrt 3 := 
sorry

end volume_of_smaller_cube_l273_273253


namespace no_convex_polyhedron_with_centered_parallelogram_intersection_l273_273697

theorem no_convex_polyhedron_with_centered_parallelogram_intersection (P : set Point) (O : Point) :
  convex P →
  (O ∈ P) →
  (∀ plane : set Point, (O ∈ plane) → (∃ p : parallelogram, (P ∩ plane = p) ∧ (center p = O))) →
  false :=
by
  sorry

end no_convex_polyhedron_with_centered_parallelogram_intersection_l273_273697


namespace smallest_possible_sum_l273_273153

theorem smallest_possible_sum (N : ℕ) (p : ℚ) (h : p > 0) (hsum : 6 * N = 2022) : 
  ∃ (N : ℕ), N * 1 = 337 :=
by 
  use 337
  sorry

end smallest_possible_sum_l273_273153


namespace determinant_zero_solution_l273_273684

noncomputable def sine_cosine_determinant_solutions (k : ℤ) : ℝ :=
  k * Real.pi + Real.pi / 3

theorem determinant_zero_solution : 
  { x : ℝ | ∀ (x : ℝ), ∃ (k : ℤ), 
  ∀ (M: matrix (fin 2) (fin 2) ℝ), 
  (M = ![![Real.sin x, Real.sqrt 3], ![Real.cos x, 1]]) → matrix.det M = 0 → x = sine_cosine_determinant_solutions k } :=
sorry

end determinant_zero_solution_l273_273684


namespace exists_set_S_l273_273075

theorem exists_set_S (m n : ℕ) (C : set (ℝ × ℝ)) (r : ℝ) (R : ℝ) (r_n : ℝ) :
  (R = r / (real.cos (real.pi / m))) ∧
  (r < r_n) ∧
  (r_n < R) ∧
  (r_n = r / (real.cos (real.pi / (2 * m * n)))) ∧
  (C = closed_polygon_region m r) → 
  (∃ S : set (ℝ × ℝ), 
    (∀ (X : fin n), ∃ θ : ℝ, θ ∈ [0, 2 * real.pi] ∧ (∀ x ∈ X, x ∈ C)) ∧ 
    ¬ (∀ x ∈ S, x ∈ C)) :=
begin
  sorry
end

end exists_set_S_l273_273075


namespace math_problem_l273_273977

   theorem math_problem :
     (3^2 * 3^1 + 3^2 * 3^1) / (3^(-2) + 3^(-2)) = 243 := by
   sorry
   
end math_problem_l273_273977


namespace find_dot_product_l273_273385

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
axiom cos_theta : real.cos (angle a b) = 1 / 3
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 3

-- Proof Goal
theorem find_dot_product : (2 • a + b) ⬝ b = 11 := sorry

end find_dot_product_l273_273385


namespace emily_51_49_calculations_l273_273579

theorem emily_51_49_calculations :
  (51^2 = 50^2 + 101) ∧ (49^2 = 50^2 - 99) :=
by
  sorry

end emily_51_49_calculations_l273_273579


namespace geometric_sequence_ratio_l273_273878

variable (a_1 q : ℝ)

def S_3 : ℝ := a_1 * (1 + q + q^2)

def a_3 : ℝ := a_1 * q^2

theorem geometric_sequence_ratio (hq : q > 0) (hS3 : S_3 a_1 q = 7 * a_3 a_1 q) : q = 1 / 2 := by
  sorry

end geometric_sequence_ratio_l273_273878


namespace rachel_fathers_age_when_rachel_is_25_l273_273906

theorem rachel_fathers_age_when_rachel_is_25 (R G M F Y : ℕ) 
  (h1 : R = 12)
  (h2 : G = 7 * R)
  (h3 : M = G / 2)
  (h4 : F = M + 5)
  (h5 : Y = 25 - R) : 
  F + Y = 60 :=
by sorry

end rachel_fathers_age_when_rachel_is_25_l273_273906


namespace omar_total_time_l273_273514

-- Conditions
def lap_distance : ℝ := 400
def first_segment_distance : ℝ := 200
def second_segment_distance : ℝ := 200
def speed_first_segment : ℝ := 6
def speed_second_segment : ℝ := 4
def number_of_laps : ℝ := 7

-- Correct answer we want to prove
def total_time_proven : ℝ := 9 * 60 + 23 -- in seconds

-- Theorem statement claiming total time is 9 minutes and 23 seconds
theorem omar_total_time :
  let time_first_segment := first_segment_distance / speed_first_segment
  let time_second_segment := second_segment_distance / speed_second_segment
  let single_lap_time := time_first_segment + time_second_segment
  let total_time := number_of_laps * single_lap_time
  total_time = total_time_proven := sorry

end omar_total_time_l273_273514


namespace coefficient_x3_expansion_l273_273924

theorem coefficient_x3_expansion :
  @coeff ℕ x^3 (expand (polynomial.C 2 + polynomial.X)^6) = 160 := 
sorry

end coefficient_x3_expansion_l273_273924


namespace cube_division_l273_273846

theorem cube_division (n : ℕ) (hn1 : 6 ≤ n) (hn2 : n % 2 = 0) : 
  ∃ m : ℕ, (n = 2 * m) ∧ (∀ a : ℕ, ∀ b : ℕ, ∀ c: ℕ, a = m^3 - (m - 1)^3 + 1 → b = 3 * m * (m - 1) + 2 → a = b) :=
by
  sorry

end cube_division_l273_273846


namespace gunther_typing_l273_273425

theorem gunther_typing :
  ∀ (wpm : ℚ), (wpm = 160 / 3) → 480 * wpm = 25598 :=
by
  intros wpm h
  sorry

end gunther_typing_l273_273425


namespace sum_sigma_v_l273_273848

noncomputable def S (n : ℕ) := {π // π.perm 1 n}
def σ (π : perm) : ℤ := if π.even then 1 else -1
def v (π : perm) : ℕ := π.fixed_points.card

theorem sum_sigma_v (n : ℕ) :
  ∑ π in S n, σ π / (v π + 1) = (-1)^(n+1) * (n / (n + 1)) :=
by
  -- Proof will be provided later.
  sorry

end sum_sigma_v_l273_273848


namespace max_length_common_chord_l273_273318

theorem max_length_common_chord (a b : ℝ) :
  let line_eqn := (2 * a - 2 * b) * x + (2 * a - 2 * b) * y + 2 * a^2 - 2 * b^2 + 1 = 0
  let center1 := (-a, -a)
  let radius1 := 1
  let d := abs((2 * a - 2 * b) * (-a) + 2 * a^2 - 2 * b^2 + 1) / (2 * sqrt((a - b)^2 + (a - b)^2))
  max_length_common_chord := 2 * sqrt(1 - d^2)
  in max_length_common_chord = 2 :=
by
  sorry

end max_length_common_chord_l273_273318


namespace projection_same_vector_l273_273100

noncomputable def projection_vector (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ :=
let direction := (v₂.1 - v₁.1, v₂.2 - v₁.2) in
let t := -((v₁.1 * direction.1 + v₁.2 * direction.2) / (direction.1 * direction.1 + direction.2 * direction.2)) in
(v₁.1 + t * direction.1, v₁.2 + t * direction.2)

theorem projection_same_vector {v₁ v₂ : ℝ × ℝ} (h : projection_vector v₁ v₂ = projection_vector v₂ v₁) :
  projection_vector v₁ v₂ = (33 / 10, 11 / 10) :=
by
  have dir := (v₂.1 - v₁.1, v₂.2 - v₁.2)
  have t := -((v₁.1 * dir.1 + v₁.2 * dir.2) / (dir.1 * dir.1 + dir.2 * dir.2))
  show projection_vector v₁ v₂ = (33 / 10, 11 / 10)
  sorry

end projection_same_vector_l273_273100


namespace triangle_side_lengths_l273_273793

theorem triangle_side_lengths (a b c : ℝ) 
  (h1 : a + b + c = 18) 
  (h2 : a + b = 2 * c) 
  (h3 : b = 2 * a):
  a = 4 ∧ b = 8 ∧ c = 6 := 
by
  sorry

end triangle_side_lengths_l273_273793


namespace larger_number_is_50_l273_273442

variable (a b : ℕ)
-- Conditions given in the problem
axiom cond1 : 4 * b = 5 * a
axiom cond2 : b - a = 10

-- The proof statement
theorem larger_number_is_50 : b = 50 :=
sorry

end larger_number_is_50_l273_273442


namespace Claire_picked_30_oranges_l273_273509

-- Define the given conditions and values
variables (LiamsOranges : ℕ) (LiamsPricePerOrange ClairesPricePerOrange TotalSavings : ℝ)
variable  ClairesOranges : ℕ

-- Assign values to the conditions
def LiamsOranges := 40
def LiamsPricePerOrange := 2.5 / 2  -- Since Liam sold 2 oranges for 2.50 dollars
def ClairesPricePerOrange := 1.20
def TotalSavings := 86.0

-- Calculate what Liam earned
def LiamsEarnings := LiamsOranges * LiamsPricePerOrange

-- Calculate what Claire must have earned
def ClairesEarnings := TotalSavings - LiamsEarnings

-- Calculate the number of oranges Claire picked
def ClairesOranges := ClairesEarnings / ClairesPricePerOrange

theorem Claire_picked_30_oranges : ClairesOranges = 30 :=
by
  sorry

end Claire_picked_30_oranges_l273_273509


namespace intersection_A_B_l273_273777

def A : set ℝ := {x : ℝ | -4 < x ∧ x < 1}
def B : set ℝ := {x : ℝ | x < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | -4 < x ∧ x < 0} :=
sorry

end intersection_A_B_l273_273777


namespace locus_equation_l273_273708

theorem locus_equation (x y : ℝ) (h1 : y = |y|) (h2 : ∀ M : ℝ × ℝ, M = (x, y) → |M.snd| = sqrt((M.fst)^2 + (M.snd - 2)^2)) :
  y = (x^2) / 4 + 1 := by
sorry

end locus_equation_l273_273708


namespace seating_arrangement_l273_273034

theorem seating_arrangement (boys girls : ℕ) (alternate : boys = 5 ∧ girls = 4 ∧ alternate_seating : boys = 5 ∧ girls = 4): 
  5! * nat.choose 6 4 * 4! = 43200 :=
begin
  sorry
end

end seating_arrangement_l273_273034


namespace right_triangle_angle_l273_273460

theorem right_triangle_angle (x : ℝ) (h1 : x + 5 * x = 90) : 5 * x = 75 :=
by
  sorry

end right_triangle_angle_l273_273460


namespace find_value_of_pqr_l273_273307

noncomputable def polynomial_condition (p q r s : ℝ) : Prop :=
  ∀ x : ℝ, (x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s) = (x^4 + 4*x^3 + 6*x^2 + 4*x + 1) * (x + (real.has_neg.neg 1)) 

theorem find_value_of_pqr 
  (p q r s : ℝ) 
  (h : polynomial_condition p q r s): 
  (p + q) * r = -1.5 := 
sorry

end find_value_of_pqr_l273_273307


namespace area_quadrilateral_CADB_l273_273967

-- Define geometric setup and conditions
def CirclesIntersect (O₁ O₂ : Point) (A B : Point) (r₁ r₂ : ℝ) : Prop := sorry
def Perpendicular (X Y Z : Point) : Prop := sorry
def MinorArc (O : Point) (A B : Point) (angles : ℝ) : Prop := sorry
noncomputable def Radius (O : Point) : ℝ := sorry

-- Define points, radii and angles
axiom O₁ O₂ A B C D : Point
axiom r₂ : ℝ
axiom angle₁ angle₂ : ℝ

-- Conditions of the problem
variable (h₀ : CirclesIntersect O₁ O₂ A B (Radius O₁) r₂)
variable (h₁ : Perpendicular A C A B)
variable (h₂ : Perpendicular B D A B)
variable (h₃ : MinorArc O₁ A B 45)
variable (h₄ : MinorArc O₂ A B 60)
variable (h₅ : Radius O₂ = 10)

-- The main goal
theorem area_quadrilateral_CADB :
  ∃ (a b c k ℓ : ℝ), 
    (a + b + c + k + ℓ = 155) ∧ 
    AreaQuadrilateral C A D B = a + b * sqrt k + c * sqrt ℓ := sorry

end area_quadrilateral_CADB_l273_273967


namespace find_a_l273_273133

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) := x^2 - 3 * a * x + a^2 = 0

-- Define the sum of the squares of the roots
def sum_of_squares_roots (a : ℝ) := 
  let x1 := (3 * a - a * Real.sqrt 5) / 2
  let x2 := (3 * a + a * Real.sqrt 5) / 2
  x1^2 + x2^2

-- The main theorem to prove
theorem find_a (a : ℝ) : 
  sum_of_squares_roots a = 1.75 → a = 0.5 ∨ a = -0.5 :=
by
  sorry

end find_a_l273_273133


namespace smallest_sum_with_probability_l273_273161

theorem smallest_sum_with_probability (N : ℕ) (p : ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 6) (h2 : 6 * N = 2022) (h3 : p > 0) :
  ∃ M, M = 337 ∧ (∀ sum, sum = 2022 → P(sum) = p) ∧ (∀ min_sum, min_sum = N → P(min_sum) = p):=
begin
  sorry
end

end smallest_sum_with_probability_l273_273161


namespace number_of_episodes_last_season_more_than_others_l273_273842

-- Definitions based on conditions
def episodes_per_other_season : ℕ := 22
def initial_seasons : ℕ := 9
def duration_per_episode : ℚ := 0.5
def total_hours_after_last_season : ℚ := 112

-- Derived definitions based on conditions (not solution steps)
def total_hours_first_9_seasons := initial_seasons * episodes_per_other_season * duration_per_episode
def additional_hours_last_season := total_hours_after_last_season - total_hours_first_9_seasons
def episodes_last_season := additional_hours_last_season / duration_per_episode

-- Proof problem statement
theorem number_of_episodes_last_season_more_than_others : 
  episodes_last_season = episodes_per_other_season + 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_episodes_last_season_more_than_others_l273_273842


namespace emily_team_players_l273_273615

theorem emily_team_players (total_points : ℕ) (emily_points : ℕ) (other_players_points : ℕ) (players_count : ℕ) :
    total_points = 39 →
    emily_points = 23 →
    other_players_points = 2 →
    players_count = (39 - 23) / 2 + 1 →
    players_count = 9 :=
by
  intros h_total h_emily h_others h_count
  rw [h_total, h_emily, h_others] at h_count
  exact h_count

end emily_team_players_l273_273615


namespace omelette_combinations_l273_273098

theorem omelette_combinations:
  let fillings := 8 in
  let filling_combinations := 2 ^ fillings in
  let egg_choices := 4 in
  filling_combinations * egg_choices = 1024 :=
by
  sorry

end omelette_combinations_l273_273098


namespace solve_for_t_l273_273110

noncomputable def solveEquation : ℕ :=
  let t := 1 in
  if 4 * 4^t + Math.sqrt (16 * 16^t) + 2^t = 34 then t else 0

theorem solve_for_t : solveEquation = 1 :=
by
  sorry

end solve_for_t_l273_273110


namespace triangle_area_ABC_l273_273024

noncomputable def triangle_area (A B C : ℝ) (angleA : ℝ) : ℝ :=
1/2 * A * B * (Real.sin angleA)

theorem triangle_area_ABC :
  ∀ (A B C : ℝ), 
    angleA = 120 / 180 * Real.pi → 
    A = 5 → 
    C = 7 →
    triangle_area A 3 angleA = 15 * Real.sqrt 3 / 4 :=
by
  intros A B C h1 h2 h3
  rw [triangle_area]
  sorry

end triangle_area_ABC_l273_273024


namespace base4_of_25_and_sum_digits_l273_273030

theorem base4_of_25_and_sum_digits :
  let n := 25
  let b := 4
  let base4_repr := [2* b^2, 2*b + 1]
  list.sum base4_repr = 4 ∧ base4_repr = [1, 2, 1] :=
by
  sorry

end base4_of_25_and_sum_digits_l273_273030


namespace largest_internal_angle_eq_120_l273_273396

-- Define the sequence such that the sum of the first n terms is n^2
def sequence (n : ℕ) : ℕ → ℕ
| 0     := 0
| (n+1) := (n+1)^2 - n^2

-- Define the sides of the triangle using the sequence
def a_2 := sequence 2
def a_3 := sequence 3
def a_4 := sequence 4

-- Define the sides of the triangle ABC
def a := a_2
def b := a_3
def c := a_4

-- Prove that the largest internal angle of the triangle ABC is 120 degrees
theorem largest_internal_angle_eq_120 (a b c : ℕ) (ha : a = 3) (hb : b = 5) (hc : c = 7) : 
  (largest_angle : ℝ) ≠ 120 := sorry

end largest_internal_angle_eq_120_l273_273396


namespace average_speed_first_part_l273_273219

variable (v : ℝ) -- average speed during the first part of the trip

-- Definitions of conditions
def distance_first_part : ℝ := 10
def distance_second_part : ℝ := 12
def speed_second_part : ℝ := 10
def total_distance : ℝ := distance_first_part + distance_second_part
def total_time (v : ℝ) := distance_first_part / v + distance_second_part / speed_second_part
def average_speed_for_trip : ℝ := 10.82

-- The statement to prove
theorem average_speed_first_part (h : v > 0) :
  v = 12 :=
by
  have total_distance_22 : total_distance = 22 := by sorry
  have total_time_eq : total_time v = 22 / average_speed_for_trip := by sorry
  have speed_eq : total_time v = 22 / 10.82 := by sorry
  have calc_v : v = 12 := by sorry
  exact calc_v

end average_speed_first_part_l273_273219


namespace ratio_of_doctors_to_lawyers_l273_273637

theorem ratio_of_doctors_to_lawyers
  (m n : ℕ) -- number of doctors (m) and lawyers (n)
  (h1 : 40 * (m + n) = 35 * m + 50 * n) -- average age condition
  (age_avg_doctors : 35) -- average age of doctors
  (age_avg_lawyers : 50) -- average age of lawyers
  : m / n = 2 :=
by {
  sorry,
}

end ratio_of_doctors_to_lawyers_l273_273637


namespace minimum_is_4_for_f_D_l273_273661

-- Definitions of the functions
def f_A (x : ℝ) : ℝ := x + 4 / x
def f_B (x : ℝ) : ℝ := sin x + 4 / sin x
def f_C (x : ℝ) : ℝ := log x / log 2 + 4 / (log x / log 2)
def f_D (x : ℝ) : ℝ := exp x + 4 / exp x

-- Conditions for the valid range of x in function B (0 < x < π)
def valid_range_B (x : ℝ) : Prop := 0 < x ∧ x < π

-- The theorem stating that among the given functions, only f_D can achieve a minimum value of 4
theorem minimum_is_4_for_f_D : ∀ x, (f_D x = 4 ↔ x = real.log 2) ∧ 
  (∀ x, f_A x ≠ 4) ∧ 
  (∀ x, valid_range_B x → f_B x ≠ 4) ∧ 
  (∀ x, f_C x ≠ 4) :=
by
  sorry

end minimum_is_4_for_f_D_l273_273661


namespace jane_total_worth_l273_273053

open Nat

theorem jane_total_worth (q d : ℕ) (h1 : q + d = 30)
  (h2 : 25 * q + 10 * d + 150 = 10 * q + 25 * d) :
  25 * q + 10 * d = 450 :=
by
  sorry

end jane_total_worth_l273_273053


namespace exists_positive_integer_n_with_N_distinct_prime_factors_l273_273076

open Nat

/-- Let \( N \) be a positive integer. Prove that there exists a positive integer \( n \) such that \( n^{2013} - n^{20} + n^{13} - 2013 \) has at least \( N \) distinct prime factors. -/
theorem exists_positive_integer_n_with_N_distinct_prime_factors (N : ℕ) (h : 0 < N) : 
  ∃ n : ℕ, 0 < n ∧ (n ^ 2013 - n ^ 20 + n ^ 13 - 2013).primeFactors.card ≥ N :=
sorry

end exists_positive_integer_n_with_N_distinct_prime_factors_l273_273076


namespace solve_equation_l273_273565

theorem solve_equation : ∃ x : ℝ, 4^x - 4 * 2^x - 5 = 0 ∧ x = Real.log2 5 := 
by
  sorry

end solve_equation_l273_273565


namespace area_quadrilateral_inscribed_in_circle_l273_273822

open Real

-- Definitions of the sides of the quadrilateral
def AB := 2
def BC := 6
def CD := 4
def DA := 4

-- The area formula for a cyclic quadrilateral using the lengths of the sides
def area_ABCD : Real := 8 * sqrt 3

-- The final theorem stating that the area of quadrilateral ABCD is 8 √3
theorem area_quadrilateral_inscribed_in_circle :
  ∃ A B C D: Real^2, is_cyclic_quadrilateral A B C D ∧ dist A B = AB ∧ dist B C = BC ∧ dist C D = CD ∧ dist D A = DA → 
  polygon_area [A, B, C, D] = area_ABCD :=
by 
  sorry

end area_quadrilateral_inscribed_in_circle_l273_273822


namespace initial_workers_and_hours_l273_273638

variable (x y t : ℝ)
variable (hx1 : 14 * x * y * t = 10 * (x + 4) * (y + 1) * t)
variable (hx2 : 14 * x * y * t = 7 * (x + 10) * (y + 2) * t)

theorem initial_workers_and_hours :
  x = 20 ∧ y = 6 :=
by
  have h1 : 14 * x * y = 10 * (x + 4) * (y + 1),
    from by rwa [mul_eq_mul_right_iff] at hx1,
  have h2 : 14 * x * y = 7 * (x + 10) * (y + 2),
    from by rwa [mul_eq_mul_right_iff] at hx2,
  let eq1 := eq.trans (by symmetry; exact h1) h2,
  have eq2 : 2 * x * y = 5 * y + 20 * x + 20,
    by { linarith },
  have eq3 : x * y = 2 * y + 10 * x + 20,
    by { linarith },
  have y_eq : y = 6,
    from by { linarith },
  have x_eq : x = 20,
    from by { linarith },
  exact ⟨x_eq, y_eq⟩

end initial_workers_and_hours_l273_273638


namespace students_grades_l273_273910

theorem students_grades (S A B C D : ℕ) (hA: S - A = A + 10) (hB: S - B = B + 8) (hC: S - C = C + 6)
  (hA_grade : A >= 3) (hB_grade : B >= 3) (hC_grade : C >= 3) (hD_grade : D >= 3)
  (num_students : 4) :
  ∃ A B C D, A = 3 ∧ B = 4 ∧ C = 5 ∧ D = 4 ∧ S = 16 :=
begin
  sorry
end

end students_grades_l273_273910


namespace lucille_house_difference_l273_273093

def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

def average_height (h1 h2 h3 : ℕ) : ℕ := (h1 + h2 + h3) / 3

def difference (h_average h_actual : ℕ) : ℕ := h_average - h_actual

theorem lucille_house_difference :
  difference (average_height height_lucille height_neighbor1 height_neighbor2) height_lucille = 3 :=
by
  unfold difference
  unfold average_height
  sorry

end lucille_house_difference_l273_273093


namespace elisa_improvement_l273_273692

theorem elisa_improvement (cur_laps cur_minutes prev_laps prev_minutes : ℕ) 
  (h1 : cur_laps = 15) (h2 : cur_minutes = 30) 
  (h3 : prev_laps = 20) (h4 : prev_minutes = 50) : 
  ((prev_minutes / prev_laps : ℚ) - (cur_minutes / cur_laps : ℚ) = 0.5) :=
by
  sorry

end elisa_improvement_l273_273692


namespace number_of_boxes_l273_273164

-- Define the conditions
def apples_per_crate : ℕ := 180
def number_of_crates : ℕ := 12
def rotten_apples : ℕ := 160
def apples_per_box : ℕ := 20

-- Define the statement to prove
theorem number_of_boxes : (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 100 := 
by 
  sorry -- Proof skipped

end number_of_boxes_l273_273164


namespace flower_bed_perimeter_and_area_l273_273245

theorem flower_bed_perimeter_and_area :
  ∀ (length width : ℝ), length = 0.4 → width = 0.3 → 
  (2 * (length + width) = 1.4) ∧ (length * width = 0.12) :=
by
  intros length width h_length h_width
  rw [h_length, h_width]
  split
  { calc
      2 * (0.4 + 0.3) = 2 * 0.7 : by norm_num
      ... = 1.4 : by norm_num
  }
  { calc
      0.4 * 0.3 = 0.12 : by norm_num
  }
  sorry

end flower_bed_perimeter_and_area_l273_273245


namespace find_missing_number_l273_273943

theorem find_missing_number (x : ℕ) (h1 : (1 + 22 + 23 + 24 + x + 26 + 27 + 2) = 8 * 20) : x = 35 :=
  sorry

end find_missing_number_l273_273943


namespace neither_directly_nor_inversely_proportional_A_D_l273_273476

-- Definitions for the equations where y is neither directly nor inversely proportional to x
def equationA (x y : ℝ) : Prop := x^2 + x * y = 0
def equationD (x y : ℝ) : Prop := 4 * x + y^2 = 7

-- Definition for direct or inverse proportionality
def isDirectlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x
def isInverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Proposition that y is neither directly nor inversely proportional to x for equations A and D
theorem neither_directly_nor_inversely_proportional_A_D (x y : ℝ) :
  equationA x y ∧ equationD x y ∧ ¬isDirectlyProportional x y ∧ ¬isInverselyProportional x y :=
by sorry

end neither_directly_nor_inversely_proportional_A_D_l273_273476


namespace median_length_is_correct_l273_273870

noncomputable def median_length (A B C O : Point) (a : ℝ) 
  (h1 : ∠ BAC = 35) 
  (h2 : ∠ BOC = 145) 
  (h3 : dist B C = a)
  (h4 : is_centroid O A B C) : ℝ :=
m

theorem median_length_is_correct
  (A B C O : Point) (a : ℝ) 
  (h1 : ∠ BAC = 35) 
  (h2 : ∠ BOC = 145) 
  (h3 : dist B C = a)
  (h4 : is_centroid O A B C) :
  median_length A B C O a h1 h2 h3 h4 = a * (sqrt 3) / 2 :=
sorry

end median_length_is_correct_l273_273870


namespace calculate_f12_plus_f3_l273_273502

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f(-x) = -f(x) := sorry
lemma f_at_1 : f(1) = 2 := sorry
lemma periodicity (x : ℝ) : f(x + 1) = f(x + 5) := sorry

theorem calculate_f12_plus_f3 : f(12) + f(3) = -2 := 
by {
  sorry
}

end calculate_f12_plus_f3_l273_273502


namespace gym_monthly_revenue_l273_273644

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end gym_monthly_revenue_l273_273644


namespace length_of_segment_is_4_sqrt_10_l273_273038

noncomputable def length_of_segment : ℝ :=
  -- Given parametric equations of the line (l)
  let x (t : ℝ) := (2 * Real.sqrt 5 / 5) * t in
  let y (t : ℝ) := 1 + (Real.sqrt 5 / 5) * t in
  -- Equation of the parabola
  let eqn := ∀ t : ℝ, (y t)^2 = 4 * (x t) in
  -- Solve the quadratic equation resulting from substituting the parametric equations into the parabola
  let discriminant := (6 * Real.sqrt 5)^2 - 4 * 5 in
  let t1_plus_t2 := 6 * Real.sqrt 5 in
  let t1_times_t2 := 5 in
  -- Calculate the distance between the intersections
  Real.sqrt (t1_plus_t2^2 - 4 * t1_times_t2)

theorem length_of_segment_is_4_sqrt_10 :
  length_of_segment = 4 * Real.sqrt 10 := 
  sorry

end length_of_segment_is_4_sqrt_10_l273_273038


namespace operation_to_zero_possible_l273_273813

open Matrix

-- Define the problem setup
noncomputable def m : ℕ := 3 -- Replace with actual m
noncomputable def n : ℕ := 3 -- Replace with actual n

variable (M : Matrix (Fin m) (Fin n) ℤ)

-- Define the condition: Sum of elements in the matrix
def matrix_sum (M : Matrix (Fin m) (Fin n) ℤ) : ℤ :=
  ⊤.sum (λ i j, M i j)

-- Define the theorem statement
theorem operation_to_zero_possible {M : Matrix (Fin m) (Fin n) ℤ} (h_pos : ∀ i j, 0 ≤ M i j) :
  (∃ k : ℤ, ∀ i j, M i j + k = 0) ↔ (matrix_sum M = 0) := by
  sorry

end operation_to_zero_possible_l273_273813


namespace probability_real_compound_l273_273106

noncomputable def set_of_rational_numbers : set ℚ :=
  {q | ∃ n d: ℤ, 1 ≤ d ∧ d ≤ 6 ∧ (0 : ℚ) ≤ (n / d) ∧ (n / d) < 2 ∧ q = n / d}

def cos_pi (a : ℚ) : ℝ := real.cos (a * real.pi)
def sin_pi (b : ℚ) : ℝ := real.sin (b * real.pi)

def is_real (z : ℂ) : Prop := z.im = 0

theorem probability_real_compound :
  let possible_values := (@set_of_rational_numbers).to_finset in
  let pairs := finset.cartesian_product possible_values possible_values in
  let num_real_pairs := pairs.count (λ ab, is_real ((cos_pi ab.1 + complex.I * sin_pi ab.2)^6)) in
  let total_pairs := pairs.card in
  num_real_pairs.to_rat / total_pairs.to_rat = 104 / 729 :=
by
  sorry

end probability_real_compound_l273_273106


namespace tangent_line_equation_l273_273933

theorem tangent_line_equation :
  ∀ (x : ℝ), let y := x^2 + x + 0.5 in
    (∃ (tangent : ℝ → ℝ), tangent x = 1 * (x - 0) + 0.5) →
      (tangent 0 = 0.5 ∧ ∀ x, tangent x = x + 0.5) :=
by
  sorry

end tangent_line_equation_l273_273933


namespace cube_faces_edges_vertices_sum_l273_273595

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end cube_faces_edges_vertices_sum_l273_273595


namespace find_circumcircle_radius_l273_273173

-- Conditions definitions
variables (r1 r2 : ℝ)
variables (O1 O2 A B C : Type)
variables (radius_A : ℝ)

-- Hypotheses based on conditions
h1 : r1 + r2 = 7
h2 : (r1 + 8)^2 + (r2 + 8)^2 = 17^2
h3 : O1 ≠ O2
h4 : (radius_A : ℝ) = 8
h5 : (A : ℝ) = 2 * sqrt 15
h6 : A = B
h7 : A = C

-- Expected conclusion
theorem find_circumcircle_radius :
  radius_A = 2 * sqrt 15 := by sorry

end find_circumcircle_radius_l273_273173


namespace axis_of_symmetry_func_l273_273125

theorem axis_of_symmetry_func :
  ∃ (k : ℤ), ∀ x, 3 * sin (2 * x + (real.pi / 6)) = 3 * sin (2 * (real.pi / 6 - x) + (real.pi / 6)) :=
begin
  use 0,
  sorry
end

end axis_of_symmetry_func_l273_273125


namespace _l273_273302

noncomputable theorem limit_calculation :
  (Real.limit (fun x : ℝ => (sqrt (9 + 2 * x) - 5) / (cbrt x - 2)) 8 = 2.4) :=
by
  sorry

end _l273_273302


namespace circle_area_l273_273036

noncomputable theory

-- Define the equation of the circle.
def circle_equation (x y : ℝ) : Prop :=
    3 * x^2 + 3 * y^2 + 6 * x - 9 * y + 3 = 0

-- Define the radius squared of the circle derived from the given equation.
def radius_squared (r : ℝ) : Prop :=
    r^2 = 2.25

-- Prove that the area of the circle is 2.25π given the circle's equation.
theorem circle_area (x y : ℝ) (h : circle_equation x y) : ∃ (A : ℝ), A = 2.25 * real.pi :=
sorry

end circle_area_l273_273036


namespace KellyGamesLeft_l273_273843

def initialGames : ℕ := 121
def gamesGivenAway : ℕ := 99

theorem KellyGamesLeft : initialGames - gamesGivenAway = 22 := by
  sorry

end KellyGamesLeft_l273_273843


namespace sum_of_elements_in_S_l273_273069

def is_repeating_decimal_form (x : ℝ) : Prop :=
∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ x = (a + b / 10 + c / 100) * (1 / 999)

noncomputable def S : set ℝ := { x | is_repeating_decimal_form x }

theorem sum_of_elements_in_S : ∑ x in S, x = 360 := sorry

end sum_of_elements_in_S_l273_273069


namespace symmetric_graph_l273_273446

def g (x : ℝ) : ℝ := 3^x + 1

theorem symmetric_graph (f : ℝ → ℝ) :
  (∀ x, f(x) = -(g(x))) → (∀ x, f(x) = -3^x - 1) :=
by
  intros h
  funext x
  calc
    f(x) = -(g(x)) : h x
        ... = -(3^x + 1) : by rfl
        ... = -3^x - 1 : by ring
  sorry

end symmetric_graph_l273_273446


namespace dividend_is_86_l273_273609

-- Given conditions and definitions
def R : ℕ := 6
def D (Q : ℕ) : ℕ := 5 * Q
def D_alt : ℕ := 3 * R + 2

-- Dividend to be proven
def V (D : ℕ) (Q : ℕ) (R : ℕ) : ℕ := (D * Q) + R

-- Proof statement
theorem dividend_is_86 (Q : ℕ) (H1 : D(Q) = D_alt) : 
  V (D(Q)) Q R = 86 :=
by
  sorry

end dividend_is_86_l273_273609


namespace correct_propositions_l273_273273

-- Definitions based on the problem conditions
def chi_square_hypothesis : Prop := ∀ (A B : Event), independent A B

def smoking_bronchitis_claim : Prop := 
  chi_square_statistic_value = 7.469 ∧ 
  chi_square_threshold = 6.635 ∧ 
  confidence_level = 0.99 ∧ 
  ¬(99 of 100 over_age_50_with_bronchitis)

def regression_R2_claim : Prop := ∀ (R2 : ℝ), 
  (higher R2) → (smaller_residual_sum_of_squares → better_model_fit)

def normal_distribution_claim : Prop := 
  ∀ (X : ℝ → ℝ), (X ∼ Normal(μ, ρ^2)) → (smaller ρ → higher_probability_X_concentrated_around_μ)

def sin_cos_claim : Prop := 
  ¬∃ x0 : ℝ, (sin x0 + cos x0 > sqrt(2))

-- Statement of the theorem
theorem correct_propositions : 
  (chi_square_hypothesis ∧ regression_R2_claim ∧ normal_distribution_claim) ∧ 
  ¬smoking_bronchitis_claim ∧ ¬sin_cos_claim := by 
  sorry

end correct_propositions_l273_273273


namespace proof_problem_l273_273856

-- Definitions of arithmetic and geometric sequences
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r^n

-- Lean statement of the problem
theorem proof_problem 
  (a b : ℕ → ℝ)
  (h_a_arithmetic : is_arithmetic_sequence a)
  (h_b_geometric : is_geometric_sequence b)
  (h_condition : a 1 - (a 7)^2 + a 13 = 0)
  (h_b7_a7 : b 7 = a 7) :
  b 3 * b 11 = 4 :=
sorry

end proof_problem_l273_273856


namespace slope_of_tangent_l273_273073

-- We define the conditions: f is an even function and the slope of the tangent at (1, f(1)) is 1.
variable {f : ℝ → ℝ}
variable (h_even : ∀ x, f(-x) = f(x))
variable (h1 : deriv f 1 = 1)

-- The statement we need to prove: the slope of the tangent line to the curve at the point (-1, f(-1)) is -1.
theorem slope_of_tangent :
  deriv f (-1) = -1 :=
by
  sorry

end slope_of_tangent_l273_273073


namespace cube_faces_edges_vertices_sum_l273_273589

theorem cube_faces_edges_vertices_sum :
  ∀ (F E V : ℕ), F = 6 → E = 12 → V = 8 → F + E + V = 26 :=
by
  intros F E V F_eq E_eq V_eq
  rw [F_eq, E_eq, V_eq]
  rfl

end cube_faces_edges_vertices_sum_l273_273589


namespace smallest_sum_symmetrical_dice_l273_273148

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end smallest_sum_symmetrical_dice_l273_273148


namespace parabola_tangents_min_area_l273_273174

noncomputable def parabola_tangents (p : ℝ) : Prop :=
  ∃ (y₀ : ℝ), p > 0 ∧ (2 * Real.sqrt (y₀^2 + 2 * p) = 4)

theorem parabola_tangents_min_area (p : ℝ) : parabola_tangents 2 :=
by
  sorry

end parabola_tangents_min_area_l273_273174


namespace sum_of_squares_of_coefficients_l273_273284

theorem sum_of_squares_of_coefficients :
  let p := 3 * (X^5 + 4 * X^3 + 2 * X + 1)
  let coeffs := [3, 12, 6, 3, 0, 0]
  let sum_squares := coeffs.map (λ c => c * c) |>.sum
  sum_squares = 198 := by
  sorry

end sum_of_squares_of_coefficients_l273_273284


namespace markup_percentage_l273_273649

theorem markup_percentage (PP SP SaleP : ℝ) (M : ℝ) (hPP : PP = 60) (h1 : SP = 60 + M * SP)
  (h2 : SaleP = SP * 0.8) (h3 : 4 = SaleP - PP) : M = 0.25 :=
by 
  sorry

end markup_percentage_l273_273649


namespace contrapositive_proposition_l273_273523

-- Define the necessary elements in the context of real numbers
variables {a b c d : ℝ}

-- The statement of the contrapositive
theorem contrapositive_proposition : (a + c ≠ b + d) → (a ≠ b ∨ c ≠ d) :=
sorry

end contrapositive_proposition_l273_273523


namespace ratio_of_areas_l273_273655

-- Definitions and conditions
variables (s r : ℝ)
variables (h1 : 4 * s = 4 * π * r)

-- Statement to prove
theorem ratio_of_areas (h1 : 4 * s = 4 * π * r) : s^2 / (π * r^2) = π := by
  sorry

end ratio_of_areas_l273_273655


namespace geometric_sequence_sum_l273_273809

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_common_ratio : ∀ n, a (n + 1) = 2 * a n)
    (h_sum : a 1 + a 2 + a 3 = 21) : a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l273_273809


namespace noodles_gender_independence_distribution_of_X_expected_variance_Y_l273_273889

/- Part 1: Independence Test -/
theorem noodles_gender_independence
    (a b c d n : ℕ)
    (h_a : a = 30) (h_b : b = 25) (h_c : c = 20) (h_d : d = 25) (h_n : n = 100)
    (α : ℝ) (h_α : α = 0.05)
    (χ2_val : ℝ) (h_χ2_val : χ2_val = χSquared a b c d n) :
  (χ2_val < 3.841) ↔ (noodles is independent of gender) :=
sorry

/- Part 2: Distribution of X -/
theorem distribution_of_X
  (X : ℕ)
  (male_female_distribution : finset ℕ) 
  (h_distribution : male_female_distribution = {0, 1, 2})
  (P : ℕ → ℝ)
  (h_P0 : P 0 = 3 / 10)
  (h_P1 : P 1 = 3 / 5)
  (h_P2 : P 2 = 1 / 10) :
  distribution X male_female_distribution P :=
sorry

/- Part 3: Expected Value and Variance of Y -/
theorem expected_variance_Y
    (Y : ℕ)
    (n : ℕ) (p : ℝ)
    (h_n : n = 3) (h_p : p = 3 / 5)
    (E_Y : ℝ) (h_E_Y : E_Y = 9 / 5)
    (Var_Y : ℝ) (h_Var_Y : Var_Y = 18 / 25) :
  expected_value_and_variance Y n p E_Y Var_Y :=
sorry

end noodles_gender_independence_distribution_of_X_expected_variance_Y_l273_273889


namespace Radhika_total_games_l273_273531

theorem Radhika_total_games :
  let christmas_gifts := 12
      birthday_gifts := 8
      total_gifts := christmas_gifts + birthday_gifts
      previously_owned := (1 / 2) * total_gifts
  in previously_owned + total_gifts = 30 :=
by
  let christmas_gifts := 12
  let birthday_gifts := 8
  let total_gifts := christmas_gifts + birthday_gifts
  let previously_owned := (1 / 2) * total_gifts
  show previously_owned + total_gifts = 30
  sorry

end Radhika_total_games_l273_273531


namespace triangle_area_proof_l273_273310

theorem triangle_area_proof : 
  ∀ (X Y Z P Q R S : Type)
  (circ : ∀ x, (x = X) ∨ (x = Y) ∨ (x = Z))
  (equilateral_XYZ : ∀ a b c, a = X → b = Y → c = Z → (∠ a b c = 60 ∧ ∠ b c a = 60 ∧ ∠ c a b = 60))
  (radius_XYZ : ∀ r, r = 3)
  (XP_len : ∀ x p, x = X → p = P → (distance x p = 10))
  (XQ_len : ∀ x q, x = X → q = Q → (distance x q = 7))
  (parallel_PQ : ∀ p q, p = P → q = Q → (l1 // line x q = l2 // line x p))
  (intersection_R : ∀ l1 l2, intersection l1 l2 = R)
  (collinear_XRS : ∀ r s, r = R → s = S → collinear X r s)
  (distinct_XS : ∀ s, s ≠ X → s = S)
  (area_ZYS : area_ZYS = ∃ a b c, area (triangle Z Y S) = (a * sqrt b) / c ∧ ∀ p: ℕ, a >= 0 ∧ c >= 0 ∧ b >= 0 ∧ b/∃n, p = n^2)
, ∃ a b c, a + b + c = 1552 :=
by
  sorry

end triangle_area_proof_l273_273310


namespace arrangements_count_l273_273654

noncomputable def arrangements_with_elderly_in_middle (volunteers : Nat) (elderly : Nat) : Nat :=
  if elderly = 1 then (Nat.factorial volunteers) else 0

theorem arrangements_count : arrangements_with_elderly_in_middle 4 1 = 24 :=
by
  unfold arrangements_with_elderly_in_middle
  rw Nat.factorial
  sorry

end arrangements_count_l273_273654


namespace triangle_area_approx_l273_273268

theorem triangle_area_approx (r : ℝ) (a b c : ℝ) (x : ℝ)
  (h1 : r = 5)
  (h2 : a / x = 5)
  (h3 : b / x = 12)
  (h4 : c / x = 13)
  (h5 : a^2 + b^2 = c^2)
  (h6 : c = 2 * r) :
  (1 / 2 * a * b) ≈ 35.50 := by
  -- Proof omitted
  sorry

end triangle_area_approx_l273_273268


namespace fractions_proper_or_improper_l273_273339

theorem fractions_proper_or_improper : 
  ∀ (a b : ℚ), (∃ p q : ℚ, a = p / q ∧ p < q) ∨ (∃ r s : ℚ, a = r / s ∧ r ≥ s) :=
by 
  sorry

end fractions_proper_or_improper_l273_273339


namespace min_black_triangles_l273_273506

/-- Let n be an integer with n ≥ 3. Consider all dissections of a convex n-gon 
into triangles by n-3 non-intersecting diagonals, and all colorings of the triangles 
with black and white so that triangles with a common side are always of a different color. 
Prove that the least possible number of black triangles is ⌊(n-1)/3⌋. 
-/
theorem min_black_triangles (n : ℕ) (h : n ≥ 3) : 
  ∃ f : ℕ → ℕ, f n = (n - 1) / 3 :=
begin
  sorry
end

end min_black_triangles_l273_273506


namespace range_of_a_l273_273766

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 2 then 2^x + 1 else x^2 + 2 * a * x

theorem range_of_a (a : ℝ) : -1 < a ∧ a < 3 ↔ f a (f a 1) > 3 * a ^ 2 := by
  sorry

end range_of_a_l273_273766


namespace square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l273_273949

theorem square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive :
  (∀ n : ℝ, 0 ≤ n → n^2 ≤ 0 → False) ↔ (∀ m : ℝ, m < 0 → m^2 ≤ 0) := 
sorry

end square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l273_273949


namespace find_integers_dividing_sum_l273_273698

def is_prime (n : ℕ) : Prop := sorry

def binomial (n k : ℕ) : ℕ := sorry  -- use proper binomial coefficient if needed

noncomputable def sum_S (a p : ℕ) : ℕ :=
  ∑ k in finset.range (p-3 - 2 + 1) + 2, binomial (p-1) k * a^(k-2)

theorem find_integers_dividing_sum (a : ℤ) :
  (∃ p : ℕ, is_prime p ∧ p ≥ 5 ∧ p ∣ sum_S a p) ↔ a ∉ {-2, 1} :=
sorry

end find_integers_dividing_sum_l273_273698


namespace sum_of_squares_219_l273_273560

theorem sum_of_squares_219 :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 219 ∧ a + b + c = 21 := by
  sorry

end sum_of_squares_219_l273_273560


namespace no_valid_partition_natural_numbers_100_l273_273478

theorem no_valid_partition_natural_numbers_100 :
  ¬ ∃ (A : fin 10 → finset ℕ), 
    (∀ i, A i ⊆ finset.range 101) ∧ 
    (∀ i j, i ≠ j → A i ≠ A j) ∧ 
    (∀ i j, i < j → A i.card < A j.card) ∧ 
    (∀ i j, i < j → A i.sum > A j.sum) := 
sorry

end no_valid_partition_natural_numbers_100_l273_273478


namespace a_8_eq_256_l273_273826

noncomputable def a : ℕ → ℕ
| 0 := 1
| (n + 1) := sorry

axiom a_1 : a 1 = 2
axiom a_mul : ∀ p q : ℕ, a (p + q) = a p * a q

theorem a_8_eq_256 : a 8 = 256 :=
sorry

end a_8_eq_256_l273_273826


namespace ratio_debt_manny_to_annika_l273_273966

-- Define the conditions
def money_jericho_has : ℕ := 30
def debt_to_annika : ℕ := 14
def remaining_money_after_debts : ℕ := 9

-- Define the amount Jericho owes Manny
def debt_to_manny : ℕ := money_jericho_has - debt_to_annika - remaining_money_after_debts

-- Prove the ratio of amount Jericho owes Manny to the amount he owes Annika is 1:2
theorem ratio_debt_manny_to_annika :
  debt_to_manny * 2 = debt_to_annika :=
by
  -- Proof goes here
  sorry

end ratio_debt_manny_to_annika_l273_273966


namespace nicolai_peaches_6_pounds_l273_273569

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end nicolai_peaches_6_pounds_l273_273569


namespace rectangle_area_l273_273577

theorem rectangle_area :
  ∃ (a b : ℕ), 
  (∀ (s : ℕ), s = 2 → 
  (b = 3 * s → a = 3 * s + s → a * b = 24)) := 
begin
  -- provided conditions
  use [4, 6],
  intros s hs hbs has,
  rw [hs, hbs, has],
  exact (by norm_num : 4 * 6 = 24),
end

end rectangle_area_l273_273577


namespace product_divisible_by_2006_l273_273917

theorem product_divisible_by_2006 (a : Fin 100 → ℕ) (h_distinct : Function.Injective a) (h_sum : (∑ i, a i) = 9999) : 
  2006 ∣ ∏ i, a i :=
by
  sorry

end product_divisible_by_2006_l273_273917


namespace solve_combinations_l273_273948

-- This function calculates combinations
noncomputable def C (n k : ℕ) : ℕ := if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem solve_combinations (x : ℤ) :
  C 16 (x^2 - x).natAbs = C 16 (5*x - 5).natAbs → x = 1 ∨ x = 3 :=
by
  sorry

end solve_combinations_l273_273948


namespace farmer_has_42_cows_left_l273_273230

def initial_cows : ℕ := 51
def added_cows : ℕ := 5
def fraction_sold : ℚ := 1/4

theorem farmer_has_42_cows_left : 
  let total_cows := initial_cows + added_cows in
  let cows_sold := total_cows * fraction_sold in
  total_cows - cows_sold = 42 := 
by 
  -- The proof would go here, but we are only required to state the theorem.
  sorry

end farmer_has_42_cows_left_l273_273230


namespace farmer_cows_after_selling_l273_273227

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end farmer_cows_after_selling_l273_273227


namespace solve_system_of_equations_l273_273544

theorem solve_system_of_equations :
  ∃ x y : ℚ, (5 * x - 3 * y = -7) ∧ (2 * x + 7 * y = -26) ∧ 
  x = (-127 / 41) ∧ y = (-116 / 41) :=
by
  exists (-127 / 41), (-116 / 41)
  split; sorry

end solve_system_of_equations_l273_273544


namespace line_equation_l273_273738

theorem line_equation (l : ℝ → ℝ → Prop) (a b : ℝ) 
  (h1 : ∀ x y, l x y ↔ y = - (b / a) * x + b) 
  (h2 : l 2 1) 
  (h3 : a + b = 0) : 
  l x y ↔ y = x - 1 ∨ y = x / 2 := 
by
  sorry

end line_equation_l273_273738


namespace remainder_of_polynomials_l273_273797

/-- Definition of the polynomial division with remainder (Remainder Theorem). -/
def remainder (f : ℤ[X]) (a : ℤ) : ℤ :=
  eval a f

/-- Main Theorem: -/
theorem remainder_of_polynomials :
  let p1 (x : ℤ) := x^5 - x^4 + x^3 - x^2 + x - 1
  let s1 := remainder (X^6) 1
  let s2 := remainder p1 (-1)
  s2 = -6 := by {
    let p1 (x : ℤ) := x^5 - x^4 + x^3 - x^2 + x - 1,
    let s1 := remainder (X^6 : ℤ[X]) 1,
    let s2 := remainder p1 (-1),
    have : s1 = 1 := by sorry, -- This requires proof that s1 = 1.
    have : s2 = -6 := by sorry, -- This requires proof that s2 = -6.
    exact this
  }

#check remainder_of_polynomials

end remainder_of_polynomials_l273_273797


namespace zero_point_of_function_l273_273566

theorem zero_point_of_function : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 :=
by
  sorry

end zero_point_of_function_l273_273566


namespace jane_mean_after_extra_credit_l273_273480

-- Define Jane's original scores
def original_scores : List ℤ := [82, 90, 88, 95, 91]

-- Define the extra credit points
def extra_credit : ℤ := 2

-- Define the mean calculation after extra credit
def mean_after_extra_credit (scores : List ℤ) (extra : ℤ) : ℚ :=
  let total_sum := scores.sum + (scores.length * extra)
  total_sum / scores.length

theorem jane_mean_after_extra_credit :
  mean_after_extra_credit original_scores extra_credit = 91.2 := by
  sorry

end jane_mean_after_extra_credit_l273_273480
