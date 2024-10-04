import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.NonPosRoot
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Trigonometry.Basic
import data.finset

namespace arithmetic_sequence_S2016_l73_73272

theorem arithmetic_sequence_S2016 :
  ∀ (S : ℕ → ℤ) (a1 d : ℤ), 
    a1 = -2016 →
    (∀ n : ℕ, S n = n * a1 + n * (n - 1) / 2 * d) →
    (S 2007 / 2007 - S 2005 / 2005 = 2) →
    S 2016 = -2016 :=
begin
  sorry
end

end arithmetic_sequence_S2016_l73_73272


namespace symmetric_circle_equation_l73_73096

theorem symmetric_circle_equation :
  let original_circle := λ x y, (x - 1/2)^2 + (y + 1)^2 = 5/4
  let symmetry_axis := λ x y, x - y + 1 = 0
  (∃ x y : ℝ, symmetry_axis x y) →
  (∃ x y : ℝ, original_circle x y) →
  ∀ x y : ℝ, ((x + 2)^2 + (y - 3/2)^2 = 5/4) :=
by
  sorry

end symmetric_circle_equation_l73_73096


namespace points_symmetric_to_circumcenter_lie_on_altitudes_l73_73676

variables {A B C O D H L : Type}

-- Assume the essential geometric constructs exist
variables [Point A] [Point B] [Point C]
variables [Point O] -- Circumcenter
variables [Point D] -- Midpoint of CB
variables [Point H] -- Orthocenter
variables [Point L] -- Midpoint of AH

-- Definitions we need
def is_circumcenter (O : Point) (A B C : Point) : Prop := sorry
def is_midpoint (M X Y : Point) : Prop := sorry
def is_orthocenter (H : Point) (A B C : Point) : Prop := sorry
def is_altitude (P : Point) (A B C : Point) : Prop := sorry
def is_symmetric (P Q R : Point) : Prop := sorry

-- The theorem statement
theorem points_symmetric_to_circumcenter_lie_on_altitudes
  (h1 : is_circumcenter O A B C)
  (h2 : is_midpoint D C B)
  (h3 : is_orthocenter H A B C)
  (h4 : is_midpoint L A H)
  : ∀ (M : Type) [Point M], is_midpoint M A (median A B C) → (is_symmetric O M P → is_altitude P A B C) :=
sorry

end points_symmetric_to_circumcenter_lie_on_altitudes_l73_73676


namespace no_such_matrix_N_l73_73840

variable (a b c d : ℝ)

def no_matrix_n_exists (M : Matrix (Fin 2) (Fin 2) ℝ) := 
  M ⬝ ![![a, b], ![c, d]] = ![![d, c], ![b, a]] → M = 0

theorem no_such_matrix_N (M : Matrix (Fin 2) (Fin 2) ℝ) : no_matrix_n_exists a b c d M :=
  by
  intro h
  sorry

end no_such_matrix_N_l73_73840


namespace line_ellipse_common_points_l73_73156

theorem line_ellipse_common_points (m : ℝ) : (m ≥ 1 ∧ m ≠ 5) ↔ (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ (x^2 / 5) + (y^2 / m) = 1) :=
by 
  sorry

end line_ellipse_common_points_l73_73156


namespace max_apples_l73_73012

-- Definitions based on the conditions in part (a)
def cost_per_dozen_apples : ℝ := 14
def money_spent_on_kiwis : ℝ := 10
def money_spent_on_bananas : ℝ := money_spent_on_kiwis / 2
def initial_money : ℝ := 50
def subway_fare_each_way : ℝ := 3.5
def total_subway_fare : ℝ := subway_fare_each_way * 2

-- Prove that Brian can buy a maximum of 24 apples
theorem max_apples : 24 = 2 * 12 :=
by
  let total_spent_on_fruits := money_spent_on_kiwis + money_spent_on_bananas
  let total_spent := total_spent_on_fruits + total_subway_fare
  let remaining_money := initial_money - total_spent
  let dozens_of_apples := remaining_money / cost_per_dozen_apples
  have h_dozen_apples : dozens_of_apples = 2,
  { sorry }
  have h_max_apples : 12 * dozens_of_apples = 24,
  { sorry }
  exact h_max_apples

end max_apples_l73_73012


namespace parameterized_line_correct_l73_73703

theorem parameterized_line_correct :
  ∃ s l : ℝ, (∀ (t : ℝ),
    let x := -9 + t * l in
    let y := s + t * (-7) in
    y = (3/4) * x + 3) ∧ s = -15 / 4 ∧ l = -28 / 3 :=
by
  use [-15 / 4, -28 / 3]
  sorry

end parameterized_line_correct_l73_73703


namespace unique_line_through_point_odd_x_prime_y_intercepts_l73_73614

theorem unique_line_through_point_odd_x_prime_y_intercepts :
  ∃! (a b : ℕ), 0 < b ∧ Nat.Prime b ∧ a % 2 = 1 ∧
  (4 * b + 3 * a = a * b) :=
sorry

end unique_line_through_point_odd_x_prime_y_intercepts_l73_73614


namespace equivalent_proof_problem_l73_73138

noncomputable def proof_problem (a b : ℝ) : Prop :=
  arcsin (1 + a^2) - arcsin ((b - 1)^2) ≥ π / 2 → arccos (a^2 - b^2) = π

theorem equivalent_proof_problem (a b : ℝ) : proof_problem a b :=
by
  sorry

end equivalent_proof_problem_l73_73138


namespace inverse_solution_l73_73656

variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

def f (x : ℝ) : ℝ := 1 / (a * x^2 + b * x + c)

theorem inverse_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  f a b c ∘ (λ x, 1 / c) = (λ x, 0) :=
by
  sorry

end inverse_solution_l73_73656


namespace shift_sine_graph_l73_73730

theorem shift_sine_graph (x : ℝ) : 
  shift (λ x, sin (2 * x)) (-π / 6) = (λ x, sin (2 * x + π / 3)) :=
sorry

end shift_sine_graph_l73_73730


namespace frog_escape_probability_l73_73602

def jump_probability (N : ℕ) : ℚ := N / 14

def survival_probability (P : ℕ → ℚ) (N : ℕ) : ℚ :=
  if N = 0 then 0
  else if N = 14 then 1
  else jump_probability N * P (N - 1) + (1 - jump_probability N) * P (N + 1)

theorem frog_escape_probability :
  ∃ (P : ℕ → ℚ), P 0 = 0 ∧ P 14 = 1 ∧ (∀ (N : ℕ), 0 < N ∧ N < 14 → survival_probability P N = P N) ∧ P 3 = 325 / 728 :=
sorry

end frog_escape_probability_l73_73602


namespace angle_between_a_b_cosine_between_sum_diff_l73_73881

open Real

variables (a b : ℝ^3)

-- Normalize the vector norms and dot products
axiom norm_a : ‖a‖ = 1
axiom dot_ab : a ⬝ b = 1 / 2
axiom dot_sum_diff : (a + b) ⬝ (a - b) = 1 / 2

theorem angle_between_a_b (a b : ℝ^3) 
  (norm_a : ‖a‖ = 1) 
  (dot_ab: a ⬝ b = 1 / 2)
  (dot_sum_diff: (a + b) ⬝ (a - b) = 1 / 2) :
  ∃ θ : ℝ, θ = π / 4 ∧ cos θ = (a ⬝ b) / (‖a‖ * ‖b‖) :=
sorry

theorem cosine_between_sum_diff (a b : ℝ^3) 
  (norm_a : ‖a‖ = 1) 
  (dot_ab: a ⬝ b = 1 / 2)
  (dot_sum_diff: (a + b) ⬝ (a - b) = 1 / 2) :
  cos ((a - b) ⬝ (a + b)) / (‖a - b‖ * ‖a + b‖) = sqrt 5 / 5 :=
sorry

end angle_between_a_b_cosine_between_sum_diff_l73_73881


namespace tips_on_Tuesday_l73_73560

-- Define the conditions
def hourlyWage : ℕ := 10
def hoursMonday : ℕ := 7
def tipsMonday : ℕ := 18
def hoursTuesday : ℕ := 5
def hoursWednesday : ℕ := 7
def tipsWednesday : ℕ := 20
def totalEarnings : ℕ := 240

-- Define the proof problem to check if the tips received on Tuesday is $12
theorem tips_on_Tuesday (tipsTuesday : ℕ) : 
  let wageMonday := hourlyWage * hoursMonday,
      wageTuesday := hourlyWage * hoursTuesday,
      wageWednesday := hourlyWage * hoursWednesday,
      totalWage := wageMonday + wageTuesday + wageWednesday,
      totalTips := totalEarnings - totalWage,
      tipsGiven := tipsMonday + tipsWednesday
  in totalTips - tipsGiven = tipsTuesday → tipsTuesday = 12 :=
by
  sorry

end tips_on_Tuesday_l73_73560


namespace portion_length_in_cube_l73_73104

theorem portion_length_in_cube (
  edge_lengths: List ℕ
) ( bottom_face: ℕ -> ℕ x ℕ -> ℕ x ℕ -> ℕ ) ( coordinates: ℕ -> ℕ x ℕ x ℕ ) :
  length_of_XY: ℕ = 6*11 ) :
  portion_length_XY_in_cube_5 = sqrt 75 :=
sorry

end portion_length_in_cube_l73_73104


namespace mark_total_waiting_time_l73_73291

def waiting_days : ℕ := 
  let day_1 := 4
  let day_2 := 20
  let day_3 := 30
  let day_4 := 10
  let day_5 := 14
  let day_6 := 3
  let day_7 := 21
  day_1 + day_2 + day_3 + day_4 + day_5 + day_6 + day_7

def total_minutes (days: ℕ) : ℕ := days * 24 * 60

theorem mark_total_waiting_time : total_minutes waiting_days = 146,880 :=
by
  sorry

end mark_total_waiting_time_l73_73291


namespace tan_alpha_plus_pi_over_4_equals_3_over_22_l73_73510

theorem tan_alpha_plus_pi_over_4_equals_3_over_22
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_equals_3_over_22_l73_73510


namespace tangent_line_is_tangent_l73_73925

noncomputable def func1 (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def func2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem tangent_line_is_tangent
  (a : ℝ) (h_tangent : ∃ x₀ : ℝ, func2 a x₀ = 2 * x₀ ∧ (deriv (func2 a) x₀ = 2))
  (deriv_eq : deriv func1 1 = 2)
  : a = 4 :=
by
  sorry

end tangent_line_is_tangent_l73_73925


namespace probability_odd_primes_l73_73733

noncomputable section
  open Fintype

  def odd_primes : Finset ℕ :=
    {3, 5, 7, 11, 13, 17, 19, 23, 29}
  
  def total_numbers : Finset ℕ := Finset.range 31 \ {0}

  def total_ways := Fintype.card (Finset.powersetLen 2 total_numbers)
  def odd_prime_ways := Fintype.card (Finset.powersetLen 2 odd_primes)

  theorem probability_odd_primes :
    (odd_prime_ways : ℚ) / total_ways = 12 / 145 := {
    sorry
  }

end probability_odd_primes_l73_73733


namespace cabbages_difference_l73_73793

noncomputable def numCabbagesThisYear : ℕ := 4096
noncomputable def numCabbagesLastYear : ℕ := 3969
noncomputable def diffCabbages : ℕ := numCabbagesThisYear - numCabbagesLastYear

theorem cabbages_difference :
  diffCabbages = 127 := by
  sorry

end cabbages_difference_l73_73793


namespace fraction_given_to_sofia_is_correct_l73_73673

-- Pablo, Sofia, Mia, and Ana's initial egg counts
variables {m : ℕ}
def mia_initial (m : ℕ) := m
def sofia_initial (m : ℕ) := 3 * m
def pablo_initial (m : ℕ) := 12 * m
def ana_initial (m : ℕ) := m / 2

-- Total eggs and desired equal distribution
def total_eggs (m : ℕ) := 12 * m + 3 * m + m + m / 2
def equal_distribution (m : ℕ) := 33 * m / 4

-- Eggs each need to be equal
def sofia_needed (m : ℕ) := equal_distribution m - sofia_initial m
def mia_needed (m : ℕ) := equal_distribution m - mia_initial m
def ana_needed (m : ℕ) := equal_distribution m - ana_initial m

-- Fraction of eggs given to Sofia
def pablo_fraction_to_sofia (m : ℕ) := sofia_needed m / pablo_initial m

theorem fraction_given_to_sofia_is_correct (m : ℕ) :
  pablo_fraction_to_sofia m = 7 / 16 :=
sorry

end fraction_given_to_sofia_is_correct_l73_73673


namespace linear_function_points_relation_l73_73204

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ),
    (y1 = 2 * (-3) + 1) →
    (y2 = 2 * 4 + 1) →
    (y1 = -5) ∧ (y2 = 9) :=
by
  intros y1 y2 hy1 hy2
  split
  · exact hy1
  · exact hy2

end linear_function_points_relation_l73_73204


namespace find_blue_balloons_l73_73454

theorem find_blue_balloons (purple_balloons : ℕ) (left_balloons : ℕ) (total_balloons : ℕ) (blue_balloons : ℕ) :
  purple_balloons = 453 →
  left_balloons = 378 →
  total_balloons = left_balloons * 2 →
  total_balloons = purple_balloons + blue_balloons →
  blue_balloons = 303 := by
  intros h1 h2 h3 h4
  sorry

end find_blue_balloons_l73_73454


namespace complement_intersection_l73_73959

def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection : (M ∩ N)ᶜ = { x : ℝ | x < 1 ∨ x > 3 } :=
  sorry

end complement_intersection_l73_73959


namespace intersection_compl_A_compl_B_l73_73165

open Set

variable (x y : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}

theorem intersection_compl_A_compl_B (U A B : Set ℝ) (hU : U = univ) (hA : A = {x | -1 < x ∧ x < 4}) (hB : B = {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}):
  (Aᶜ ∩ Bᶜ) = (Iic (-1) ∪ Ici 5) :=
by
  sorry

end intersection_compl_A_compl_B_l73_73165


namespace cannot_represent_2019_as_sum_of_90_natural_numbers_with_same_digit_sum_l73_73058

theorem cannot_represent_2019_as_sum_of_90_natural_numbers_with_same_digit_sum :
  ¬ ∃ (a : ℕ), ∀ (i : ℕ), i < 90 → (a = (List.replicate 90 a).sum) ∧ (∑ d in a.digits 10, d) = (∑ d in a.digits 10, d) :=
by
  sorry

end cannot_represent_2019_as_sum_of_90_natural_numbers_with_same_digit_sum_l73_73058


namespace three_digit_cubes_divisible_by_8_l73_73566

theorem three_digit_cubes_divisible_by_8 : ∃ (count : ℕ), count = 2 ∧
  ∀ (n : ℤ), (100 ≤ 8 * n^3) ∧ (8 * n^3 ≤ 999) → 
  (8 * n^3 = 216 ∨ 8 * n^3 = 512) := by
  sorry

end three_digit_cubes_divisible_by_8_l73_73566


namespace problem_proof_l73_73623

noncomputable def polar_eq_line (ρ θ : ℝ) : Prop :=
  ρ * cos θ - ρ * sin θ + 4 = 0

noncomputable def rect_eq_curve (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0

def max_min_x_plus_2y_bounds (x y : ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  (x-2)^2 + (y-2)^2 = 2 → 
  (10 - sqrt 6 ≤ x + 2 * y) ∧ (x + 2 * y ≤ 10 + sqrt 6)

theorem problem_proof :
  (∀ ρ θ, polar_eq_line ρ θ) → 
  (∀ x y, rect_eq_curve x y) → 
  (∀ x y (P : (ℝ × ℝ)), (P ∈ C) → max_min_x_plus_2y_bounds x y):
  sorry

end problem_proof_l73_73623


namespace ninth_term_l73_73342

variable (a d : ℤ)
variable (h1 : a + 2 * d = 20)
variable (h2 : a + 5 * d = 26)

theorem ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end ninth_term_l73_73342


namespace friends_Sarah_brought_l73_73390

def total_people_in_house : Nat := 15
def in_bedroom : Nat := 2
def living_room : Nat := 8
def Sarah : Nat := 1

theorem friends_Sarah_brought :
  total_people_in_house - (in_bedroom + Sarah + living_room) = 4 := by
  sorry

end friends_Sarah_brought_l73_73390


namespace omega_value_increasing_intervals_axis_of_symmetry_range_of_function_l73_73547

-- Problem statement and conditions
noncomputable def f (ω x : ℝ) : ℝ := sin (2 * ω * x - π/6) + 1/2
def period (ω : ℝ) : ℝ := π

-- Correct answers as Lean theorems

theorem omega_value : (∀ ω > 0, period ω = π → ω = 1) :=
by
  intros ω hω hperiod
  sorry

theorem increasing_intervals (k : ℤ): (∀ ω > 0, ω = 1 → 
  ∃ a b, ∀ x ∈ set.Icc a b, deriv (λ x, f ω x) x ≥ 0 ↔ (a, b) = (-π/6 + k * π, π/3 + k * π)) :=
by
  intros ω hω hvalue
  sorry

theorem axis_of_symmetry(k : ℤ): (∀ ω > 0, ω = 1 → 
  ∃ x, x = π/3 + k * (π/2)) :=
by
  intros ω hω hvalue
  sorry

theorem range_of_function: (∀ ω x, ω > 0 → ω = 1 → (0 ≤ x ∧ x ≤ 2 * π / 3) → 
  0 ≤ f ω x ∧ f ω x ≤ 3 / 2) :=
by
  intros ω x hω hvalue hx
  sorry

end omega_value_increasing_intervals_axis_of_symmetry_range_of_function_l73_73547


namespace proof_triangle_Angle_B_l73_73257

noncomputable theory

open Real

def triangle_Angle_B (a b : ℝ) (A B : ℝ) :=
  a = 4 ∧ b = 2 * sqrt 2 ∧ A = π / 4 ∧ sin B = 1 / 2 → B = π / 6

theorem proof_triangle_Angle_B : triangle_Angle_B 4 (2 * sqrt 2) (π / 4) (π / 6) :=
by {
  sorry
}

end proof_triangle_Angle_B_l73_73257


namespace hyperbola_focus_sum_eccentricities_l73_73134

noncomputable def ellipse := {x : ℝ × ℝ // (x.1^2 / 9) + (x.2^2 / 25) = 1}

def focal_length_ellipse : ℝ :=
  real.sqrt (25 - 9)

def eccentricity_ellipse : ℝ :=
  focal_length_ellipse / 5

def hyperbola_equation (a b : ℝ) : Prop :=
  ∀ x : ℝ × ℝ, (x.2^2 / a^2) - (x.1^2 / b^2) = 1

theorem hyperbola_focus_sum_eccentricities (a b : ℝ)
  (hf_length : focal_length_ellipse = 4)
  (ecc_sum : (4/a) + (4/5) = 14/5) :
  hyperbola_equation 2 (real.sqrt (4^2 - 2^2)) :=
by
  sorry

end hyperbola_focus_sum_eccentricities_l73_73134


namespace induction_step_expression_l73_73741

theorem induction_step_expression (k : ℕ) : 
  (∑ i in finset.range (2*k+2) \ finset.range (k+1), (2:ℝ) / (↑i+1)) = 
  (∑ i in finset.range (2*k+3) \ finset.range (k+2), (2:ℝ) / (↑i+1)) + (1 / ((2*k+1)*(k+1))) :=
begin
  sorry
end

end induction_step_expression_l73_73741


namespace arithmetic_sequence_ninth_term_l73_73344

theorem arithmetic_sequence_ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end arithmetic_sequence_ninth_term_l73_73344


namespace sum_of_factorials_mod_25_l73_73589

theorem sum_of_factorials_mod_25 :
  (1! + 2! + 3! + 4! + (∑ i in (range 46).map(λ x, x + 5), i!)) % 25 = 8 :=
by sorry

end sum_of_factorials_mod_25_l73_73589


namespace geometric_common_ratio_arithmetic_sequence_l73_73648

theorem geometric_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : S 3 = a 1 * (1 - q^3) / (1 - q)) (h2 : S 3 = 3 * a 1) :
  q = 2 ∨ q^3 = - (1 / 2) := by
  sorry

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h : S 3 = a 1 * (1 - q^3) / (1 - q))
  (h3 : 2 * S 9 = S 3 + S 6) (h4 : q ≠ 1) :
  a 2 + a 5 = 2 * a 8 := by
  sorry

end geometric_common_ratio_arithmetic_sequence_l73_73648


namespace units_digit_42_pow_5_add_27_pow_5_l73_73750

theorem units_digit_42_pow_5_add_27_pow_5 :
  (42 ^ 5 + 27 ^ 5) % 10 = 9 :=
by
  sorry

end units_digit_42_pow_5_add_27_pow_5_l73_73750


namespace continuity_at_3_l73_73660

def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 3 then 3 * x ^ 2 + 1 else b * x - 6

theorem continuity_at_3 (b : ℝ) : ContinuousAt (λ x, f x b) 3 ↔ b = 34 / 3 :=
by
  -- The proof is omitted
  sorry

end continuity_at_3_l73_73660


namespace min_product_ab_l73_73844

theorem min_product_ab (a b : ℝ) (h : 20 * a * b = 13 * a + 14 * b) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a * b = 1.82 :=
sorry

end min_product_ab_l73_73844


namespace grid_sum_at_least_half_m_squared_l73_73609

theorem grid_sum_at_least_half_m_squared (m : ℕ) (grid : Fin m → Fin m → ℕ)
  (h : ∀ i j, grid i j = 0 → (∑ k in Finset.univ, grid i k + ∑ k in Finset.univ, grid k j) ≥ m) :
  (∑ i in Finset.univ, ∑ j in Finset.univ, grid i j) ≥ m * m / 2 := 
sorry

end grid_sum_at_least_half_m_squared_l73_73609


namespace sum_and_difference_repeating_decimals_l73_73468

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_9 : ℚ := 1
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

theorem sum_and_difference_repeating_decimals :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_9 + repeating_decimal_3 = 2 / 9 := 
by 
  sorry

end sum_and_difference_repeating_decimals_l73_73468


namespace math_proof_problem_l73_73909

noncomputable theory

-- Function definitions and conditions
def f (x : ℝ) := sorry  -- Define f as an actual function
def g (x : ℝ) := 2 * f (x + 1) - 2

-- Assumptions
axiom domain_f : ∀ x : ℝ, true  -- f is defined on all real numbers
axiom domain_g : ∀ x : ℝ, true  -- g is defined on all real numbers
axiom eq1 : ∀ x : ℝ, 2 * f x + g (x - 3) = 2
axiom symmetry_f : ∀ x : ℝ, f (2 - x) = f x  -- symmetry about x = 1
axiom value_f_at_1 : f 1 = 3

-- Proof problem statement
theorem math_proof_problem :
  (g 0 = 4) ∧ (∀ x : ℝ, g (x + 4) = g x) ∧ (g 3 = 0) :=
sorry

end math_proof_problem_l73_73909


namespace quadratic_function_solution_exists_l73_73861

theorem quadratic_function_solution_exists :
  ∃ (c d : ℝ), (∀ x : ℝ, (x^2 + 2010 * x + 1776) * (x^2 + c * x + d) = 
             (x^4 + (2*c + 4) * x^3 + (c^2 + 5*c + 4*d + 4) * x^2 + (2*c*d + 3*c + 2*d) * x + (d^2 + c*d + d))) ∧ 
             g x = x^2 + 2006 * x - 117 :=
by
  sorry

end quadratic_function_solution_exists_l73_73861


namespace probability_at_least_three_white_balls_l73_73963

open Nat

theorem probability_at_least_three_white_balls (B W : ℕ) (total_balls_drawn : ℕ) :
  B = 7 → W = 8 → total_balls_drawn = 5 → 
  (at_least_3_white_prob : ℕ → ℚ) :
  at_least_3_white_prob total_balls_drawn = 4/7 := by
  intros B_eq W_eq total_balls_drawn_eq
  sorry

end probability_at_least_three_white_balls_l73_73963


namespace M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l73_73159

-- Definition of the curve using parametric equations
def curve (t : ℝ) : ℝ × ℝ :=
  (3 * t, 2 * t^2 + 1)

-- Questions and proof statements
theorem M1_on_curve_C : ∃ t : ℝ, curve t = (0, 1) :=
by { 
  sorry 
}

theorem M2_not_on_curve_C : ¬ (∃ t : ℝ, curve t = (5, 4)) :=
by { 
  sorry 
}

theorem M3_on_curve_C_a_eq_9 (a : ℝ) : (∃ t : ℝ, curve t = (6, a)) → a = 9 :=
by { 
  sorry 
}

end M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l73_73159


namespace problem1_problem2_l73_73882

-- Definitions for conditions
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Problem 1: For m = 4, p ∧ q implies 4 < x < 5
theorem problem1 (x : ℝ) (h : 4 < x ∧ x < 5) : 
  p x ∧ q x 4 :=
sorry

-- Problem 2: ∃ m, m > 0, m ≤ 2, and 3m ≥ 5 implies (5/3 ≤ m ≤ 2)
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : m ≤ 2) (h3 : 3 * m ≥ 5) : 
  5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end problem1_problem2_l73_73882


namespace expression_value_l73_73465

theorem expression_value :
  (cos 10 * Real.sqrt 3 + sin 10) / Real.sqrt (1 - sin 50^2) = 2 := by
  sorry

end expression_value_l73_73465


namespace correct_solution_to_equation_l73_73383

theorem correct_solution_to_equation :
  ∃ x m : ℚ, (m = 3 ∧ x = 14 / 23 → 7 * (2 - 2 * x) = 3 * (3 * x - m) + 63) ∧ (∃ x : ℚ, (∃ m : ℚ, m = 3) ∧ (7 * (2 - 2 * x) - (3 * (3 * x - 3) + 63) = 0)) →
  x = 2 := 
sorry

end correct_solution_to_equation_l73_73383


namespace largest_number_sigma1_1854_l73_73266

def sigma1 (n : ℕ) : ℕ := (n.divisors).sum

theorem largest_number_sigma1_1854 (n : ℕ) :
  (sigma1 n = 1854) → n = 1234 := 
sorry

end largest_number_sigma1_1854_l73_73266


namespace find_geometric_sequence_element_l73_73989

theorem find_geometric_sequence_element (a b c d e : ℕ) (r : ℚ)
  (h1 : 2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100)
  (h2 : Nat.gcd a e = 1)
  (h3 : r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4)
  : c = 36 :=
  sorry

end find_geometric_sequence_element_l73_73989


namespace anthony_final_pet_count_l73_73814

def initial_pets : ℕ := 120

def percent_loss (n : ℕ) (percent : ℕ) : ℕ :=
  (percent * n + 99) / 100 -- rounding up a percentage loss

def percent_gain (n : ℕ) (percent : ℕ) : ℕ :=
  (percent * n + 99) / 100 -- rounding up a percentage gain

def birth_pets (n : ℕ) (rate : ℚ) (offspring : ℕ) : ℕ :=
  let m := (rate * n).num / (rate * n).denom -- number of pets that gave birth
  m * offspring -- total offspring count

def final_pets : ℕ :=
  let p1 := initial_pets
  let p2 := p1 - percent_loss p1 8
  let p3 := p2 + 15
  let p4 := p3 + birth_pets p3 (3/8) 3
  let p5 := p4 + percent_gain p4 25
  let p6 := p5 - percent_loss p5 9
  let p7 := p6 - percent_loss p6 11
  p7

theorem anthony_final_pet_count : final_pets = 270 :=
  by sorry

end anthony_final_pet_count_l73_73814


namespace three_digit_cubes_divisible_by_eight_l73_73569

theorem three_digit_cubes_divisible_by_eight :
  (∃ n1 n2 : ℕ, 100 ≤ n1 ∧ n1 < 1000 ∧ n2 < n1 ∧ 100 ≤ n2 ∧ n2 < 1000 ∧
  (∃ m1 m2 : ℕ, 2 ≤ m1 ∧ 2 ≤ m2 ∧ n1 = 8 * m1^3  ∧ n2 = 8 * m2^3)) :=
sorry

end three_digit_cubes_divisible_by_eight_l73_73569


namespace jerry_remaining_money_l73_73639

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l73_73639


namespace correct_statement_l73_73812

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def has_axis_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x, f (2 * a - x) = f (x)

def function_range (f : ℝ → ℝ) (I J : Set ℝ) : Prop :=
∀ y, y ∈ set.range f → y ∈ J

def has_min_no_max (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∃ m, (∀ x ∈ I, f x ≥ m) ∧ (∀ M, ∃ x ∈ I, f x > M)

theorem correct_statement :
  ¬is_odd_function (λ x, Real.sin (2 * x - Real.pi / 3)) ∧
  ¬has_axis_of_symmetry (λ x, Real.cos (2 * x - Real.pi / 3)) (Real.pi / 3) ∧
  ¬function_range (λ x, Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)) (Set.Icc 0 (Real.pi / 2)) (Set.Icc 0 (Real.sqrt 2)) ∧
  has_min_no_max (λ x, (Real.cos x + 3) / Real.cos x) (Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :=
by
  sorry

end correct_statement_l73_73812


namespace total_cost_pencils_and_pens_l73_73429

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end total_cost_pencils_and_pens_l73_73429


namespace intersection_range_and_distance_l73_73157

theorem intersection_range_and_distance (b : ℝ) (x1 x2 y1 y2 : ℝ) 
  (h₁ : (3 * x1^2 + 4 * b * x1 + 2 * b^2 - 2 = 0) ∧ 
        (3 * x2^2 + 4 * b * x2 + 2 * b^2 - 2 = 0))
  (h₂ : y1 = x1 + b) (h₃ : y2 = x2 + b) 
  (h₄ : y1^2 = 1 - x1^2/2) (h₅ : y2^2 = 1 - x2^2/2) :
  (-√3 < b ∧ b < √3) ∧
  if (b = 1)
  then (| x1 - x2 | = 4 * √2 / 3) :=
begin
  sorry
end

end intersection_range_and_distance_l73_73157


namespace color_lines_no_finite_region_all_blue_l73_73434

theorem color_lines_no_finite_region_all_blue (n : ℕ) (h_n_large : n > 0) :
  ∃ m : ℕ, m ≥ Nat.sqrt n ∧ ∀ (C : Finset (Fin n → Finset (Fin n))) 
   (h_general_position : ∀ l1 l2 l3 : Fin n, l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → ¬ ∃ p : plane, (l1 ⊆ p) ∧ (l2 ⊆ p) ∧ (l3 ⊆ p)),
    ∀ (L : Fin n → Finset (Fin n)),  ( ∃ B ⊆ C, B.card ≥ Nat.sqrt n ∧ ∀ R : Finset (Fin n), R ⊆ C ∧ R.card <8 → (∀ b ∈ B, b ∈ R → true)) :=
begin
  sorry
end

end color_lines_no_finite_region_all_blue_l73_73434


namespace total_cost_38_pencils_56_pens_l73_73431

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end total_cost_38_pencils_56_pens_l73_73431


namespace find_circle_equation_l73_73908

-- Define the intersection point of the lines x + y + 1 = 0 and x - y - 1 = 0
def center : ℝ × ℝ := (0, -1)

-- Define the chord length AB
def chord_length : ℝ := 6

-- Line equation that intersects the circle
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Circle equation to be proven
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 18

-- Main theorem: Prove that the given circle equation is correct under the conditions
theorem find_circle_equation (x y : ℝ) (hc : x + y + 1 = 0) (hc' : x - y - 1 = 0) 
  (hl : line_eq x y) : circle_eq x y :=
sorry

end find_circle_equation_l73_73908


namespace problem_solution_l73_73974

-- Definitions based on conditions
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C2_x (θ : ℝ) : ℝ := sqrt(2) * cos θ
def C2_y (θ : ℝ) : ℝ := sin θ

-- Definitions based on polar coordinates
def polar_x (ρ θ : ℝ) : ℝ := ρ * cos θ
def polar_y (ρ θ : ℝ) : ℝ := ρ * sin θ

-- Definitions of polar functions
def polar_eq_C1 (ρ θ : ℝ) : Prop := ρ = 2 * cos θ
def polar_eq_C2 (ρ θ : ℝ) : Prop := ρ^2 * (1 + sin θ ^ 2) = 2

-- Ray equation in polar coordinates
def ray (θ : ℝ) : Prop := θ = π / 6

-- Intersection radii
def ρ1 : ℝ := 2 * cos (π / 6) -- for C1
def ρ2 : ℝ := sqrt(2 / (1 + sin (π / 6) ^ 2)) -- for C2

-- Distance between the intersection points
def distance_AB : ℝ := abs (ρ1 - ρ2)

theorem problem_solution :
  (∀ ρ θ, C1 (polar_x ρ θ) (polar_y ρ θ) ↔ polar_eq_C1 ρ θ) ∧
  (∀ ρ θ, (polar_x ρ θ = C2_x θ) ∧ (polar_y ρ θ = C2_y θ) ↔ polar_eq_C2 ρ θ) ∧
  distance_AB = sqrt(3) - (2 * sqrt(10)) / 5 :=
by
  sorry

end problem_solution_l73_73974


namespace side_lengths_le_sqrt3_probability_is_1_over_3_l73_73358

open Real

noncomputable def probability_side_lengths_le_sqrt3 : ℝ :=
  let total_area : ℝ := 2 * π^2
  let satisfactory_area : ℝ := 2 * π^2 / 3
  satisfactory_area / total_area

theorem side_lengths_le_sqrt3_probability_is_1_over_3 :
  probability_side_lengths_le_sqrt3 = 1 / 3 :=
by
  sorry

end side_lengths_le_sqrt3_probability_is_1_over_3_l73_73358


namespace max_sin_sum_of_triangle_l73_73449

theorem max_sin_sum_of_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  sin A + sin B + sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end max_sin_sum_of_triangle_l73_73449


namespace min_value_x_plus_4_div_x_plus_1_l73_73538

theorem min_value_x_plus_4_div_x_plus_1 (x : ℝ) (h : x > -1) : ∃ m, m = 3 ∧ ∀ y, y = x + 4 / (x + 1) → y ≥ m :=
sorry

end min_value_x_plus_4_div_x_plus_1_l73_73538


namespace sine_transformation_correct_intervals_l73_73330

def sine_transform_increasing_intervals (k : ℤ) : Prop :=
  ∀ x, (kπ - π/12 ≤ x ∧ x ≤ kπ + 5π/12) → increasing_on (λ x, Real.sin (2 * x - π/3)) x

theorem sine_transformation_correct_intervals :
  ∀ k : ℤ, sine_transform_increasing_intervals k :=
by
  sorry

end sine_transformation_correct_intervals_l73_73330


namespace jessica_minimal_withdrawal_l73_73751

theorem jessica_minimal_withdrawal 
  (initial_withdrawal : ℝ)
  (initial_fraction : ℝ)
  (minimum_balance : ℝ)
  (deposit_fraction : ℝ)
  (after_withdrawal_balance : ℝ)
  (deposit_amount : ℝ)
  (current_balance : ℝ) :
  initial_withdrawal = 400 →
  initial_fraction = 2/5 →
  minimum_balance = 300 →
  deposit_fraction = 1/4 →
  after_withdrawal_balance = 1000 - initial_withdrawal →
  deposit_amount = deposit_fraction * after_withdrawal_balance →
  current_balance = after_withdrawal_balance + deposit_amount →
  current_balance - minimum_balance ≥ 0 →
  0 = 0 :=
by
  sorry

end jessica_minimal_withdrawal_l73_73751


namespace cost_to_treat_dog_l73_73670

variable (D : ℕ)
variable (cost_cat : ℕ := 40)
variable (num_dogs : ℕ := 20)
variable (num_cats : ℕ := 60)
variable (total_paid : ℕ := 3600)

theorem cost_to_treat_dog : 20 * D + 60 * cost_cat = total_paid → D = 60 := by
  intros h
  -- Proof goes here
  sorry

end cost_to_treat_dog_l73_73670


namespace sum_of_10_consecutive_terms_is_200_l73_73977

theorem sum_of_10_consecutive_terms_is_200 : 
  ∀ (a : ℕ → ℕ), 
    (∀ n, a n = 2 * n + 1) → 
    (let sum_9 := (∑ n in (list.range 10).erase 3, a (n + 4)) in sum_9 = 185) → 
    (let sum_10 := ∑ n in list.range 10, a (n + 4) in sum_10 = 200) :=
begin
  intros a ha sum_9_eq,
  let a_formula := ha,
  have sum_10_eq : (∑ n in list.range 10, a (n + 4)) = 200,
  {
    sorry, -- Proof goes here
  },
  exact sum_10_eq,
end

end sum_of_10_consecutive_terms_is_200_l73_73977


namespace jerry_remaining_money_l73_73640

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l73_73640


namespace scientific_notation_of_1_656_million_l73_73807

theorem scientific_notation_of_1_656_million :
  (1.656 * 10^6 = 1656000) := by
sorry

end scientific_notation_of_1_656_million_l73_73807


namespace train_passes_telegraph_post_in_7_2_seconds_l73_73395

theorem train_passes_telegraph_post_in_7_2_seconds
  (l : ℕ) (v_kmph : ℕ) (conversion_kmph_to_mps : ℕ → ℚ) (time_to_pass : ℚ) :
  l = 80 →
  v_kmph = 40 →
  conversion_kmph_to_mps = λ x, x * 1000 / 3600 →
  time_to_pass = l / (conversion_kmph_to_mps v_kmph) →
  time_to_pass ≈ 7.2 :=
begin
  intros,
  sorry
end

end train_passes_telegraph_post_in_7_2_seconds_l73_73395


namespace smallest_c_exists_l73_73862

theorem smallest_c_exists (x : Fin 51 → ℝ) (M : ℝ) (h_sum : (∑ i, x i) = 0) (h_median : ∃ i : Fin 51, M = x i ∧ (∑ j in (Finset.range 26), x j) < 26 * x i ∧ (∑ j in (Finset.range 25).map (i + 26), x j) > 25 * x i) :
  ∃ c : ℝ, c = 702 / 25 ∧ (∑ i, x i ^ 2) ≥ c * M ^ 2 :=
by
  sorry

end smallest_c_exists_l73_73862


namespace max_stores_visited_l73_73350

theorem max_stores_visited 
  (total_stores : ℕ) (total_visits : ℕ) (people_shopping : ℕ)
  (people_two_stores : ℕ) (remaining_visits : ℕ) 
  (each_visit : people_two_stores * 2) 
  (total_remaining_visits : total_visits - each_visit) 
  (remaining_people : people_shopping - people_two_stores) 
  (remaining_people_visits : remaining_people * 1 ≤ total_remaining_visits) 
  (max_visits_one_person : total_remaining_visits ≤ remaining_people * 1 + 3) 
  (initial_visit : 1) 
  (final_visits_max : 4)  
  :
  final_visits_max = initial_visit + 3 
  :=
by {
  recall final_visits_max : 4,
  sorry -- Proof to show that the maximum number of visits one person could have made is 4
}

end max_stores_visited_l73_73350


namespace count_valid_numbers_l73_73174

theorem count_valid_numbers : 
  let invalid_digits := [0, 1, 8, 9] in
  let range_start := 100 in
  let range_end := 999 in
  let valid_digits := [2, 3, 4, 5, 6, 7] in
  (∃ n : ℕ, range_start ≤ n ∧ n ≤ range_end ∧ 
    (∀ d : ℕ, d ∈ invalid_digits → ¬(d ∈ n.digits 10)) → 
    n.digits 10 ⊆ valid_digits) = 216 :=
by
  sorry

end count_valid_numbers_l73_73174


namespace problem1_proof_problem2_proof_l73_73005

noncomputable
def problem1 : Real := (1 - Real.sqrt 3) ^ 0 + Real.abs (-Real.sqrt 2) - 2 * Real.cos (Float.pi / 4) + (1 / 4) ^ (-1)

theorem problem1_proof : problem1 = 5 := by
  sorry

noncomputable
def problem2_roots := (6 + 4 * Real.sqrt 3) / 6, (6 - 4 * Real.sqrt 3) / 6

theorem problem2_proof (x : Real) :
  (x = (6 + 4 * Real.sqrt 3) / 6 ∨ x = (6 - 4 * Real.sqrt 3) / 6)
  ↔ 3 * x^2 - 6 * x - 1 = 0 := by
  sorry

end problem1_proof_problem2_proof_l73_73005


namespace uncounted_angle_measure_l73_73633

-- Define the given miscalculated sum
def miscalculated_sum : ℝ := 2240

-- Define the correct sum expression for an n-sided convex polygon
def correct_sum (n : ℕ) : ℝ := (n - 2) * 180

-- State the theorem: 
theorem uncounted_angle_measure (n : ℕ) (h1 : correct_sum n = 2340) (h2 : 2240 < correct_sum n) :
  correct_sum n - miscalculated_sum = 100 := 
by sorry

end uncounted_angle_measure_l73_73633


namespace number_of_japanese_selectors_l73_73047

theorem number_of_japanese_selectors (F C J : ℕ) (h1 : J = 3 * C) (h2 : C = F + 15) (h3 : J + C + F = 165) : J = 108 :=
by
sorry

end number_of_japanese_selectors_l73_73047


namespace finite_zero_on_blackboard_l73_73357

theorem finite_zero_on_blackboard
  (r1 r2 r3 : ℝ)
  (h_nonneg : 0 ≤ r1 ∧ 0 ≤ r2 ∧ 0 ≤ r3)
  (a1 a2 a3 : ℤ)
  (h_integer : ¬(a1 = 0 ∧ a2 = 0 ∧ a3 = 0))
  (h_sum_zero : a1 * r1 + a2 * r2 + a3 * r3 = 0) :
  ∃ (n : ℕ), ∃ (f : ℕ → ℝ × ℝ × ℝ),
  (f 0 = (r1, r2, r3)) ∧
  (∀ k < n, ∃ x y, f (k+1) = if x ≤ y then (fst (fst (f k)), y - x) else (x - y, snd (f k))) ∧
  (∃ k ≤ n, let (a, b, c) := f k in a = 0 ∨ b = 0 ∨ c = 0) :=
begin
  sorry
end

end finite_zero_on_blackboard_l73_73357


namespace three_digit_cubes_divisible_by_8_l73_73567

theorem three_digit_cubes_divisible_by_8 : ∃ (count : ℕ), count = 2 ∧
  ∀ (n : ℤ), (100 ≤ 8 * n^3) ∧ (8 * n^3 ≤ 999) → 
  (8 * n^3 = 216 ∨ 8 * n^3 = 512) := by
  sorry

end three_digit_cubes_divisible_by_8_l73_73567


namespace ellipse_equation_const_prod_AN_BM_l73_73530

-- Define the given conditions
variables {a b c x y x0 y0 : ℝ}

-- Define the ellipse and its properties
def ellipse (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def eccentricity (a c : ℝ) := c / a

-- Given conditions
def given_conditions :=
  a > 0 ∧
  b > 0 ∧
  eccentricity a c = sqrt 3 / 2 ∧
  a * b / 2 = 1 ∧
  a^2 - b^2 = c^2

-- The equation of the ellipse
theorem ellipse_equation : given_conditions → ellipse x y 2 1 :=
by sorry

-- The constancy of the product |AN| * |BM|
theorem const_prod_AN_BM (P : ℝ × ℝ) : 
  given_conditions →
  P ∈ { p : ℝ × ℝ | ellipse p.1 p.2 2 1 } →
  let N := -P.1 / (P.2 - 1) in
  let M := -2 * P.2 / (P.1 - 2) in
  abs (2 + N) * abs (1 + M) = 4 :=
by sorry

end ellipse_equation_const_prod_AN_BM_l73_73530


namespace vector_at_t_zero_l73_73419

theorem vector_at_t_zero :
  ∃ a d : ℝ × ℝ, (a + d = (2, 5) ∧ a + 4 * d = (11, -7)) ∧ a = (-1, 9) ∧ a + 0 * d = (-1, 9) :=
by {
  sorry
}

end vector_at_t_zero_l73_73419


namespace unique_three_positive_perfect_square_sums_to_100_l73_73232

theorem unique_three_positive_perfect_square_sums_to_100 :
  ∃! (a b c : ℕ), a^2 + b^2 + c^2 = 100 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end unique_three_positive_perfect_square_sums_to_100_l73_73232


namespace jerry_money_left_after_shopping_l73_73638

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l73_73638


namespace y1_lt_y2_l73_73193

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l73_73193


namespace correct_option_B_l73_73385

theorem correct_option_B (x y a b : ℝ) :
  (3 * x + 2 * x^2 ≠ 5 * x) →
  (-y^2 * x + x * y^2 = 0) →
  (-a * b - a * b ≠ 0) →
  (3 * a^3 * b^2 - 2 * a^3 * b^2 ≠ 1) →
  (-y^2 * x + x * y^2 = 0) :=
by
  intros hA hB hC hD
  exact hB

end correct_option_B_l73_73385


namespace problem_1_problem_2_problem_3_l73_73276

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_def (m n : ℝ) (hm : m > 0) (hn : n > 0) : f(m * n) = f(m) + f(n)
axiom f_neg (x : ℝ) (hx : x > 1) : f(x) < 0

theorem problem_1 : f(1) = 0 :=
sorry

theorem problem_2 : ∀ ⦃x₁ x₂ : ℝ⦄, 0 < x₁ → x₁ < x₂ → f(x₂) < f(x₁) :=
sorry

theorem problem_3 (a x : ℝ) (h_f2 : f(2) = 1/2) : f(a * x + 4) > 1 ↔
       (0 < a → -4 / a < x ∧ x < 0) ∨ 
       (a < 0 → 0 < x ∧ x < -4 / a) ∨
       (a = 0 → False) :=
sorry

end problem_1_problem_2_problem_3_l73_73276


namespace choir_members_count_l73_73965

theorem choir_members_count : ∃ n : ℕ, n = 226 ∧ 
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (200 < n ∧ n < 300) :=
by
  sorry

end choir_members_count_l73_73965


namespace part1_part2_l73_73980

noncomputable theory

-- Definitions
def focus1 := (-√1, 0)
def focus2 := (√1, 0)
def ellipse (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1
def line (m : ℝ) (y : ℝ) : ℝ := m * y + 4
def point_on_line (m : ℝ) (x y : ℝ) : Prop := x = line m y
def point_T := (4, 0)
def slope (p1 p2 : (ℝ × ℝ)) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
def intersection (m : ℝ) (x y : ℝ) : Prop := ellipse x y ∧ point_on_line m x y

-- Statements to Prove
theorem part1 (m : ℝ) (x1 y1 x2 y2 : ℝ) (hA : intersection m x1 y1) (hB : intersection m x2 y2) (hA_between : x1 < 4 ∧ x2 > 4) :
  let k1 := slope (focus2) (x1, y1),
      k2 := slope (focus2) (x2, y2)
  in k1 + k2 = 0 :=
sorry

theorem part2 (m x1 y1 x2 y2 Mx My : ℝ) (hA : intersection m x1 y1) (hB : intersection m x2 y2) (hA_between : x1 < 4 ∧ x2 > 4) 
  (hM : (slope (focus1) (Mx, My)) = (slope (x1, y1) (focus1)) ∧ (slope (focus2) (Mx, My)) = (slope (x2, y2) (focus2))) :
  let MF1 := (Mx - focus1.1)^2 + (My - focus1.2)^2,
      MF2 := (Mx - focus2.1)^2 + (My - focus2.2)^2
  in MF1 - MF2 = 1 :=
sorry

end part1_part2_l73_73980


namespace concyclicity_of_A_K_I_E_l73_73890

noncomputable def are_concyclic {A B C D P E F I J K : Type*}
  [circle A B C D]
  (h1 : intersects AC BD P)
  (h2 : circumcircle_triangle_intersects_again ADP AB E)
  (h3 : circumcircle_triangle_intersects_again BCP AB F)
  (h4 : incenter_triangle I ADE)
  (h5 : incenter_triangle J BCF)
  (h6 : intersects IJ AC K) : Prop :=
concyclic A K I E

theorem concyclicity_of_A_K_I_E 
  {A B C D P E F I J K : Type*}
  [circle A B C D]
  (h1 : intersects AC BD P)
  (h2 : circumcircle_triangle_intersects_again ADP AB E)
  (h3 : circumcircle_triangle_intersects_again BCP AB F)
  (h4 : incenter_triangle I ADE)
  (h5 : incenter_triangle J BCF)
  (h6 : intersects IJ AC K) : 
  are_concyclic h1 h2 h3 h4 h5 h6 :=
sorry

end concyclicity_of_A_K_I_E_l73_73890


namespace least_value_of_a_plus_b_l73_73577

theorem least_value_of_a_plus_b (a b : ℝ) 
  (h1 : Real.log 3 a + Real.log 3 b ≥ 5) 
  (h2 : a - b = 3) : a + b = 33 := 
sorry

end least_value_of_a_plus_b_l73_73577


namespace find_initial_average_price_l73_73264

noncomputable def average_initial_price (P : ℚ) : Prop :=
  let total_cost_of_4_cans := 120
  let total_cost_of_returned_cans := 99
  let total_cost_of_6_cans := 6 * P
  total_cost_of_6_cans - total_cost_of_4_cans = total_cost_of_returned_cans

theorem find_initial_average_price (P : ℚ) :
    average_initial_price P → 
    P = 36.5 := sorry

end find_initial_average_price_l73_73264


namespace even_n_of_even_Omega_P_l73_73501

-- Define the Omega function
def Omega (N : ℕ) : ℕ := 
  N.factors.length

-- Define the polynomial function P
def P (x : ℕ) (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  List.prod (List.map (λ i => x + a i) (List.range n))

theorem even_n_of_even_Omega_P (a : ℕ → ℕ) (n : ℕ)
  (H : ∀ k > 0, Even (Omega (P k a n))) : Even n :=
by
  sorry

end even_n_of_even_Omega_P_l73_73501


namespace brother_paint_time_is_4_l73_73754

noncomputable def brother_paint_time (B : ℝ) : Prop :=
  (1 / 3) + (1 / B) = 1 / 1.714

theorem brother_paint_time_is_4 : ∃ B, brother_paint_time B ∧ abs (B - 4) < 0.001 :=
by {
  sorry -- Proof to be filled in later.
}

end brother_paint_time_is_4_l73_73754


namespace Shekar_marks_l73_73679

theorem Shekar_marks :
  ∃ M : ℕ, 
    M + 279 = 355 ∧
    (65 + 82 + 47 + 85 + M) / 5 = 71 :=
begin
  use 76,
  split,
  { exact rfl },
  { have h: (65 + 82 + 47 + 85 + 76) / 5 = 71, from rfl,
    exact h,
  }
end

end Shekar_marks_l73_73679


namespace find_magnitude_of_b_l73_73136

-- Definitions of the given vectors and conditions
def a : ℝ × ℝ := (sqrt 3, -1)
def θ : ℝ := 2 * Real.pi / 3
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Vector subtraction
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Scaling a vector by a scalar
def scalar_mul (s : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (s * v.1, s * v.2)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Given conditions
def given_condition_1 (b : ℝ × ℝ) : Prop :=
  |vec_sub a (scalar_mul 2 b)| = 2 * sqrt 13

def given_condition_2 (b : ℝ × ℝ) : Prop :=
  dot_product a b = magnitude a * magnitude b * Real.cos θ

-- Theorem to prove
theorem find_magnitude_of_b (b : ℝ × ℝ) (h1 : given_condition_1 b) (h2 : given_condition_2 b) : magnitude b = 3 :=
sorry

end find_magnitude_of_b_l73_73136


namespace smallest_d_value_l73_73031

noncomputable def distance_formula (x y: ℝ) := sqrt (x*x + y*y)

theorem smallest_d_value (d: ℝ) : 
  (distance_formula (5 * sqrt 2) (d + 4) = 4 * d) → d = 2.38 :=
by
  sorry

end smallest_d_value_l73_73031


namespace algebraic_expression_evaluation_final_theorem_l73_73587

variable (a b : ℤ)

-- Conditions
def like_terms (a b : ℤ) : Prop :=
  (2 = 1 - a) ∧ (5 = 3b - 1)

-- Mathematical statement we need to prove
theorem algebraic_expression_evaluation (ha : a = -1) (hb : b = 2) :
  5 * a * b^2 - (6 * a^2 * b - 3 * (a * b^2 + 2 * a^2 * b)) = -32 := by
  sorry

-- Combine conditions with target
theorem final_theorem (h : like_terms a b) : 
  5 * a * b^2 - (6 * a^2 * b - 3 * (a * b^2 + 2 * a^2 * b)) = -32 := by
  obtain ⟨ha, hb⟩ := h
  exact algebraic_expression_evaluation ha hb

end algebraic_expression_evaluation_final_theorem_l73_73587


namespace parabola_point_distance_l73_73584

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem parabola_point_distance (x_M y_M : ℝ) (h₀ : y_M^2 = 4 * x_M)
  (h₁ : distance (x_M, y_M) (1, 0) = 10) : x_M = 9 :=
by
  sorry

end parabola_point_distance_l73_73584


namespace max_segments_diameter_l73_73393

open Nat

theorem max_segments_diameter (n : ℕ) (P : Fin n → Prop) (h : n = 39) (surface_count : ℕ) (h_surface_count : surface_count ≤ Nat.floor (0.72 * n)) :
  ∃ max_segments : ℕ, max_segments = Nat.choose surface_count 2 :=
by
  have n := 39
  have percentage := 0.72
  have max_surface_points := Nat.floor (percentage * n)
  have h_surface_count : surface_count ≤ max_surface_points := sorry
  have max_segments := Nat.choose surface_count 2
  use max_segments
  sorry

end max_segments_diameter_l73_73393


namespace locus_of_points_is_segment_l73_73704

-- Define the conditions in Lean
def is_locus_segment (z : ℂ) : Prop :=
  complex.abs (z - 2 * complex.I) + complex.abs (z + 1) = real.sqrt 5

-- The mathematical equivalent proof problem in Lean
theorem locus_of_points_is_segment : ∃ (z : ℂ), is_locus_segment z ↔ ∃ x y : ℝ, z = x + y * complex.I ∧ dist (x, y) = real.sqrt 5 := 
sorry

end locus_of_points_is_segment_l73_73704


namespace flagstaff_height_is_correct_l73_73020

noncomputable def flagstaff_height : ℝ := 40.25 * 12.5 / 28.75

theorem flagstaff_height_is_correct :
  flagstaff_height = 17.5 :=
by 
  -- These conditions are implicit in the previous definition
  sorry

end flagstaff_height_is_correct_l73_73020


namespace kids_into_movie_total_l73_73852

-- Definitions for the conditions
def total_kids_riverside := 150
def denied_percent_riverside := 0.20
def total_kids_west_side := 100
def denied_percent_west_side := 0.70
def total_kids_mountaintop := 80
def denied_percent_mountaintop := 0.50
def total_kids_oak_grove := 60
def denied_percent_oak_grove := 0.40
def total_kids_lakeview := 110
def denied_percent_lakeview := 0.65

-- Calculate number of kids who got into the movie from each school
def kids_got_in_riverside := total_kids_riverside * (1 - denied_percent_riverside)
def kids_got_in_west_side := total_kids_west_side * (1 - denied_percent_west_side)
def kids_got_in_mountaintop := total_kids_mountaintop * (1 - denied_percent_mountaintop)
def kids_got_in_oak_grove := total_kids_oak_grove * (1 - denied_percent_oak_grove)
def kids_got_in_lakeview := total_kids_lakeview * (1 - denied_percent_lakeview)

-- Summing the number of kids who got into the movie from all schools
def total_kids_got_in := kids_got_in_riverside + kids_got_in_west_side + kids_got_in_mountaintop + kids_got_in_oak_grove + kids_got_in_lakeview

theorem kids_into_movie_total : total_kids_got_in = 264 := by
    -- Placeholder for the proof
    sorry

end kids_into_movie_total_l73_73852


namespace no_valid_selection_l73_73422

-- Define the set of chosen numbers and the conditions
def chosen_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 2022}

def valid_selection (S : finset ℕ) :=
  S.card = 677 ∧ ∀ x y ∈ S, x ≠ y → ¬ (x + y) % 6 = 0

-- State the theorem
theorem no_valid_selection : ¬ ∃ S : finset ℕ, S ⊆ chosen_numbers ∧ valid_selection S := 
sorry

end no_valid_selection_l73_73422


namespace impossible_list_10_numbers_with_given_conditions_l73_73260

theorem impossible_list_10_numbers_with_given_conditions :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 0 ≤ i ∧ i ≤ 7 → (a i * a (i + 1) * a (i + 2)) % 6 = 0) ∧
    (∀ i, 0 ≤ i ∧ i ≤ 8 → (a i * a (i + 1)) % 6 ≠ 0) :=
by
  sorry

end impossible_list_10_numbers_with_given_conditions_l73_73260


namespace range_a_for_decreasing_function_l73_73924

theorem range_a_for_decreasing_function (a : ℝ) :
  (∀ x : ℝ, -2/3 < x ∧ x < -1/3 → deriv (λ x, x^3 + a * x^2 + x + 1) x < 0) →
  a ≥ 7 / 4 :=
by
  sorry

end range_a_for_decreasing_function_l73_73924


namespace perimeter_ABFCEDE_is_correct_l73_73046

-- Define the original problem conditions
variable (ABCD : Type) (BFC : Type) (P : ℝ)
variable (a b c d f e : BFC)
variable (perimeter_ABCD : ℝ := 40)
variable (side_length : ℝ := perimeter_ABCD / 4)
variable (perimeter_new_figure : ℝ)

-- Setting up the lengths based on the properties described
def BF := side_length
def FC := side_length
def BC := side_length * Real.sqrt 2
def DE := side_length * Real.sqrt 2
def EA := side_length * Real.sqrt 2

-- define the perimeter of the new figure ABFCEDE
def new_perimeter : ℝ :=
  BF + FC + side_length + side_length + DE + EA

-- Prove that the perimeter of the new figure is 40 + 20 * √2
theorem perimeter_ABFCEDE_is_correct :
  new_perimeter = 40 + 20 * Real.sqrt 2 := by
  sorry -- Proof omitted

end perimeter_ABFCEDE_is_correct_l73_73046


namespace relationship_y1_y2_l73_73194

theorem relationship_y1_y2 :
  let f : ℝ → ℝ := λ x, 2 * x + 1 in
  let y1 := f (-3) in
  let y2 := f 4 in
  y1 < y2 :=
by {
  -- definitions
  let f := λ x, 2 * x + 1,
  let y1 := f (-3),
  let y2 := f 4,
  -- calculations
  have h1 : y1 = f (-3) := rfl,
  have h2 : y2 = f 4 := rfl,
  -- compare y1 and y2
  rw [h1, h2],
  exact calc
    y1 = f (-3) : rfl
    ... = 2 * (-3) + 1 : rfl
    ... = -5 : by norm_num
    ... < 2 * 4 + 1 : by norm_num
    ... = y2 : rfl
}

end relationship_y1_y2_l73_73194


namespace monotonic_decreasing_interval_f_l73_73706

def f (x : ℝ) : ℝ := log (1/2 : ℝ) (x^2 - 2*x - 3)

theorem monotonic_decreasing_interval_f :
  ∀ x : ℝ, (∀ y : ℝ, (x ∈ set.Ioi 3) → f(x) > f(y) → y ∈ set.Ioi 3) :=
by sorry

end monotonic_decreasing_interval_f_l73_73706


namespace problem_statement_l73_73400

theorem problem_statement :
  let a := (235.47 / 100) * 9876.34,
      b := a / 16.37,
      c := 2836.9 - 1355.8,
      d := (4 / 7) * c
  in b + d = 2266.42 :=
by
  let a := 2.3547 * 9876.34,
      b := a / 16.37,
      c := 2836.9 - 1355.8,
      d := (4 / 7) * c;
  exact eq_of_real_eq true sorry

end problem_statement_l73_73400


namespace plane_angle_proof_l73_73289

noncomputable def angle_between_planes (α β : ℝ) : ℝ :=
  69 + 4 / 60  -- The given angle in degrees and minutes

-- Definitions based on conditions
def half_line_e (O : Point) : HalfLine :=
  -- Assume an appropriate definition for half_line with endpoint O

def plane_S1 : Plane :=
  -- Assume an appropriate definition for plane_S1

def plane_S2 : Plane :=
  -- Assume an appropriate definition for plane_S2, perpendicular to plane_S1

def half_line_OB (O : Point) : HalfLine :=
  -- Assume an appropriate definition for half_line with endpoint O, making 45° with plane_S1

def half_line_OC (O : Point) : HalfLine :=
  -- Assume an appropriate definition for half_line with endpoint O, making 30° with plane_S1

axiom intersection (plane_S1 plane_S2 : Plane) (half_line_e : HalfLine) : Prop :=
  -- Assume an appropriate definition such that half_line_e is the intersection line

axiom angle_with_plane_1 (half_line : HalfLine) (plane : Plane) (angle : ℝ) : Prop :=
  -- Assume an appropriate definition for half_line making angle with a plane

axiom angle_with_plane_2 (half_line : HalfLine) (plane : Plane) (angle : ℝ) : Prop :=
  -- Assume an appropriate definition for half_line making angle with a plane

theorem plane_angle_proof
  (O : Point)
  (e : HalfLine)
  (S1 S2 : Plane)
  (OB OC : HalfLine)
  (H1 : intersection S1 S2 e)
  (H2 : angle_with_plane_1 OB S1 (45))
  (H3 : angle_with_plane_2 OC S1 (30))
  (angle_between_half_lines : ∠(OB, OC) = 60):
  ∠(∠(plane_of_half_lines OB OC, S1)) = 69 + 4 / 60 :=
sorry

end plane_angle_proof_l73_73289


namespace range_of_f1_div_f0_l73_73121

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem range_of_f1_div_f0 (a b c : ℝ) 
  (h_deriv_pos : (deriv (quadratic_function a b c) 0) > 0)
  (h_nonneg : ∀ x : ℝ, quadratic_function a b c x ≥ 0) :
  ∃ r, r = (quadratic_function a b c 1) / (deriv (quadratic_function a b c) 0) ∧ r ≥ 2 :=
sorry

end range_of_f1_div_f0_l73_73121


namespace incircle_tangent_ineq_l73_73911

variables {A B C D1 E1 O O1 : Type}
variables (a b c : ℝ) (h_tangent_circumcircle : true) (h_tangent_ac : true) (h_tangent_ab : true)

def sides (BC CA AB : ℝ) : Prop :=
  BC = a ∧ CA = b ∧ AB = c

theorem incircle_tangent_ineq (BC CA AB AD1 AE1 : ℝ) (h1 : sides BC CA AB)
  (h2 : ∀ {X Y Z D E : ℝ}, h_tangent_circumcircle → h_tangent_ac → h_tangent_ab → sides X Y Z → D = E → true) :
  AD1 = AE1 ∧ AD1 = AE1 ∧ AD1 = (2 * CA * AB) / (BC + CA + AB) :=
sorry

end incircle_tangent_ineq_l73_73911


namespace unique_solution_pair_l73_73856

-- Define the function representing the given equation
def equation (a b x : ℝ) : ℝ := (a * x^2 - 24 * x + b) / (x^2 - 1)

theorem unique_solution_pair : 
  ∃ (a b : ℝ), a = 35 ∧ b = -5819 ∧
  ∀ x : ℝ, (equation a b x = x ↔ x ≠ 1 ∧ x ≠ -1) ∧
  ∃ (x1 x2 : ℝ), x1 ≠ 1 ∧ x1 ≠ -1 ∧ x2 ≠ 1 ∧ x2 ≠ -1 ∧
  equation a b x1 = x1 ∧ equation a b x2 = x2 ∧ x1 + x2 = 12 :=
by
  sorry 

end unique_solution_pair_l73_73856


namespace card_game_impossible_l73_73008

theorem card_game_impossible : 
  ∀ (students : ℕ) (initial_cards : ℕ) (cards_distribution : ℕ → ℕ), 
  students = 2018 → 
  initial_cards = 2018 →
  (∀ n, n < students → (if n = 0 then cards_distribution n = initial_cards else cards_distribution n = 0)) →
  (¬ ∃ final_distribution : ℕ → ℕ, (∀ n, n < students → final_distribution n = 1)) :=
by
  intros students initial_cards cards_distribution stu_eq init_eq init_dist final_dist
  -- Sorry can be used here as the proof is not required
  sorry

end card_game_impossible_l73_73008


namespace bisection_interval_contains_root_l73_73981

theorem bisection_interval_contains_root :
  let f := λ x : ℝ, x^3 - 3 * x + 1 in
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x = 0 → 
   (∀ c1 : ℝ, 0 < c1 ∧ c1 < 1 ∧ f c1 < 0 →
    (∀ c2 : ℝ, 0 < c2 ∧ c2 < c1 ∧ f c2 > 0 →
     (∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c1 ∧ f a > 0 ∧ f b < 0 ∧ (a, b) = (1/4 : ℝ, 1/2 : ℝ))))) :=
by
  sorry

end bisection_interval_contains_root_l73_73981


namespace range_of_a_l73_73549

noncomputable def satisfiesInequality (a : ℝ) (x : ℝ) : Prop :=
  x > 1 → a * Real.log x > 1 - 1/x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 1 → satisfiesInequality a x) ↔ a ∈ Set.Ici 1 := 
sorry

end range_of_a_l73_73549


namespace sum_a_b_l73_73823

theorem sum_a_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 2) (h_bound : a^b < 500)
  (h_max : ∀ a' b', a' > 0 → b' > 2 → a'^b' < 500 → a'^b' ≤ a^b) :
  a + b = 8 :=
by sorry

end sum_a_b_l73_73823


namespace problem_statement_l73_73122

variables {α : Type*} [strict_ordered_field α]

-- Define the given conditions
variables (A B C O P Q : α) -- Points
variables (r : α) (t : α) -- radius and ratio
variables (PA PO PB PC : α) -- distances
variables (angleBAC anglePOQ : α) -- angles

-- Given:
-- 1. Triangle ABC is acute-angled and inscribed in circle O.
-- 2. AB > AC.
-- 3. Point P lies on the extension of BC.
-- 4. Point Q lies on the segment BC.
-- 5. PA is tangent to the circle O.
-- 6. anglePOQ + angleBAC = 90 degrees.
-- 7. PA / PO = t.
-- Prove that PQ / PC = 1 / t^2.

theorem problem_statement (h1 : is_acute_triangle A B C)
  (h2 : AB > AC)
  (h3 : lies_on_extension P BC)
  (h4 : lies_on_segment Q BC)
  (h5 : is_tangent PA O)
  (h6 : anglePOQ + angleBAC = 90)
  (h7 : PA / PO = t) :
  PQ / PC = 1 / t^2 :=
sorry

end problem_statement_l73_73122


namespace quadrilateral_perimeter_area_relation_l73_73486

/-- Given a convex quadrilateral ABCD divided into four triangles by its diagonals, the perimeter P of ABCD, the perimeter Q of the quadrilateral formed by the centers of the inscribed circles of the four triangles, and the area S_ABCD of ABCD, prove that PQ > 4 S_ABCD. -/
theorem quadrilateral_perimeter_area_relation
  (ABCD : ConvexQuadrilateral)
  (P : ℝ) (Pp : P = (ABCD.perimeter))
  (Q : ℝ)
  (Qp : Q = (formed_by_centers_of_inscribed_circles ABCD).perimeter)
  (S_ABCD : ℝ)
  (Sp : S_ABCD = ABCD.area) :
  P * Q > 4 * S_ABCD := by
  sorry

end quadrilateral_perimeter_area_relation_l73_73486


namespace Haley_boxes_needed_l73_73559

theorem Haley_boxes_needed (TotalMagazines : ℕ) (MagazinesPerBox : ℕ) 
  (h1 : TotalMagazines = 63) (h2 : MagazinesPerBox = 9) : 
  TotalMagazines / MagazinesPerBox = 7 := by
sorry

end Haley_boxes_needed_l73_73559


namespace production_value_equation_l73_73599

theorem production_value_equation (x : ℝ) :
  (2000000 * (1 + x)^2) - (2000000 * (1 + x)) = 220000 := 
sorry

end production_value_equation_l73_73599


namespace arithmetic_sequence_a8_l73_73224

def sum_arithmetic_sequence_first_n_terms (a d : ℕ) (n : ℕ): ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_a8 
  (a d : ℕ) 
  (h : sum_arithmetic_sequence_first_n_terms a d 15 = 45) : 
  a + 7 * d = 3 := 
by
  sorry

end arithmetic_sequence_a8_l73_73224


namespace equal_angles_on_diagonals_of_rectangle_l73_73592

theorem equal_angles_on_diagonals_of_rectangle
  (A B C D E F G : Type)
  [rect_ABC: Rectangle ABCD]
  (E_on_AD : on_side E AD)
  (F_on_AB : on_side F AB)
  (BE_eq_DF : dist BE = dist DF)
  (G_intersect : intersect G BE DF) :
  ∠BGC = ∠DGC := 
sorry

end equal_angles_on_diagonals_of_rectangle_l73_73592


namespace ranking_of_scores_l73_73293

variable (Liam Noah Olivia : ℝ)

theorem ranking_of_scores (h₁ : Liam ≠ Noah) (h₂ : Noah ≠ Olivia) (h₃ : Liam ≠ Olivia) 
  (h₄ : Liam > Noah) (h₅ : Noah > Olivia) : 
  Liam > Noah ∧ Noah > Olivia ∧ Liam > Olivia :=
by
  repeat {split}
  {exact h₄}
  {exact h₅}
  {exact lt_trans h₄ h₅}

end ranking_of_scores_l73_73293


namespace sum_b_eq_242520_l73_73867

noncomputable def b (p : ℕ) : ℕ := 
  Nat.find (λ k, abs (k - Real.sqrt p) < 1 / 2)

theorem sum_b_eq_242520 : (∑ p in Finset.range 5000, b (p + 1)) = 242520 :=
by sorry

end sum_b_eq_242520_l73_73867


namespace electric_guitar_count_l73_73961

theorem electric_guitar_count (E A : ℤ) (h1 : E + A = 9) (h2 : 479 * E + 339 * A = 3611) (hE_nonneg : E ≥ 0) (hA_nonneg : A ≥ 0) : E = 4 :=
by
  sorry

end electric_guitar_count_l73_73961


namespace cafeteria_extra_apples_l73_73773

-- Define the conditions from the problem
def red_apples : ℕ := 33
def green_apples : ℕ := 23
def students : ℕ := 21

-- Define the total apples and apples given out based on the conditions
def total_apples : ℕ := red_apples + green_apples
def apples_given : ℕ := students

-- Define the extra apples as the difference between total apples and apples given out
def extra_apples : ℕ := total_apples - apples_given

-- The theorem to prove that the number of extra apples is 35
theorem cafeteria_extra_apples : extra_apples = 35 :=
by
  -- The structure of the proof would go here, but is omitted
  sorry

end cafeteria_extra_apples_l73_73773


namespace MN_equals_R_l73_73540

theorem MN_equals_R
  (O H : Point) (ABC : Triangle) (M N : Point)
  (R : ℝ)
  (circumcenter : IsCircumcenter O ABC)
  (orthocenter : IsOrthocenter H ABC)
  (AM_eq_AO : dist A M = dist A O)
  (AN_eq_AH : dist A N = dist A H)
  (circumradius : Circumradius R ABC) :
  dist M N = R := 
sorry

end MN_equals_R_l73_73540


namespace largest_number_among_options_l73_73386

theorem largest_number_among_options :
  (∃ (A : ℤ) (B : ℤ) (C : ℤ) (D : ℤ) (E : ℤ), 
    A = 30 ^ 20 ∧ 
    B = 10 ^ 30 ∧ 
    C = 30 ^ 10 + 20 ^ 20 ∧
    D = (30 + 10) ^ 20 ∧
    E = (30 * 20) ^ 10 ∧
    D > A ∧ D > B ∧ D > C ∧ D > E) :=
begin
  use 30 ^ 20,
  use 10 ^ 30,
  use 30 ^ 10 + 20 ^ 20,
  use (30 + 10) ^ 20,
  use (30 * 20) ^ 10,
  -- conditions prover
  split,
  ring,

  split,
  ring,

  split,
  ring,

  split,
  ring,

  split,
  ring,

  -- Maximum value proof
  sorry,
end

end largest_number_among_options_l73_73386


namespace hyperbola_eccentricity_range_l73_73905

theorem hyperbola_eccentricity_range
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (F1 F2 P : ℝ × ℝ)
  (h_hyperbola : F1.1 = -a ∧ F2.1 = a ∧ P.2 = sqrt((P.1^2 / a^2) - 1) * b)
  (d : ℝ)
  (h_arithmetic : |dist P F1|, |dist P F2|, and d form an arithmetic sequence) :
  let e := sqrt(1 + (b/a)^2)
  in e ∈ set.Ioo 1 (2 + sqrt 3) :=
sorry

end hyperbola_eccentricity_range_l73_73905


namespace find_solutions_of_equation_l73_73158

theorem find_solutions_of_equation (m n : ℝ) 
  (h1 : ∀ x, (x - m)^2 + n = 0 ↔ (x = -1 ∨ x = 3)) :
  (∀ x, (x - 1)^2 + m^2 = 2 * m * (x - 1) - n ↔ (x = 0 ∨ x = 4)) :=
by
  sorry

end find_solutions_of_equation_l73_73158


namespace num_valid_pairs_l73_73425

theorem num_valid_pairs (a b : ℕ) (h1 : b > a) (h2 : a > 4) (h3 : b > 4)
(h4 : a * b = 3 * (a - 4) * (b - 4)) : 
    (1 + (a - 6) = 1 ∧ 72 = b - 6) ∨
    (2 + (a - 6) = 2 ∧ 36 = b - 6) ∨
    (3 + (a - 6) = 3 ∧ 24 = b - 6) ∨
    (4 + (a - 6) = 4 ∧ 18 = b - 6) :=
sorry

end num_valid_pairs_l73_73425


namespace find_c_plus_d_l73_73944

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
noncomputable def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

theorem find_c_plus_d (c d : ℝ) (h1 : ∀ x : ℝ, g c d (g_inv c d x) = x) : c + d = -2 :=
by {
  have h2: g c d (g_inv c d x) = c * (d * x + c) + d := sorry,
  have h3: c * d * x + c^2 + d = x := sorry,
  have h4: c * d = 1 := sorry,
  have h5: c^2 + d = 0 := sorry,
  have h6: d = -c^2 := sorry,
  have h7: c * (-c^2) = 1 := sorry,
  have h8: -c^3 = 1 := sorry,
  have h9: c = -1 := sorry,
  have h10: d = -1 := sorry,
  rw [h9, h10],
  exact rfl
}

end find_c_plus_d_l73_73944


namespace isosceles_triangle_congruent_side_length_l73_73690

theorem isosceles_triangle_congruent_side_length 
  (D E F G : Point) 
  (h_isosceles : DE = 30) 
  (h_area : area_of_triangle DEF = 90) 
  (h_midpoint : G is_the_midpoint_of DE) 
  (h_altitude : FG is_the_altitude_of DEF) 
  (h_right_triangle : triangle DFG is_a_right_triangle) 
  : DF = sqrt 261 :=
by sorry

end isosceles_triangle_congruent_side_length_l73_73690


namespace radishes_difference_l73_73458

theorem radishes_difference 
    (total_radishes : ℕ)
    (groups : ℕ)
    (first_basket : ℕ)
    (second_basket : ℕ)
    (total_radishes_eq : total_radishes = 88)
    (groups_eq : groups = 4)
    (first_basket_eq : first_basket = 37)
    (second_basket_eq : second_basket = total_radishes - first_basket)
  : second_basket - first_basket = 14 :=
by
  sorry

end radishes_difference_l73_73458


namespace imaginary_part_of_z_l73_73902

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + complex.I) = complex.abs (1 - complex.I) + complex.I) :
  z.im = (1 - real.sqrt 2) / 2 :=
by
  sorry

end imaginary_part_of_z_l73_73902


namespace cupcakes_frosted_in_10_minutes_l73_73460

def frosting_rate (time: ℕ) (cupcakes: ℕ) : ℚ := cupcakes / time

noncomputable def combined_frosting_rate : ℚ :=
  (frosting_rate 25 1) + (frosting_rate 35 1)

def effective_working_time (total_time: ℕ) (work_period: ℕ) (break_time: ℕ) : ℕ :=
  let break_intervals := total_time / work_period
  total_time - break_intervals * break_time

def total_cupcakes (working_time: ℕ) (rate: ℚ) : ℚ :=
  working_time * rate

theorem cupcakes_frosted_in_10_minutes :
  total_cupcakes (effective_working_time 600 240 30) combined_frosting_rate = 36 := by
  sorry

end cupcakes_frosted_in_10_minutes_l73_73460


namespace smallest_k_sum_of_squares_multiple_of_200_l73_73631

-- Define the sum of squares for positive integer k
def sum_of_squares (k : ℕ) : ℕ := (k * (k + 1) * (2 * k + 1)) / 6

-- Prove that the sum of squares for k = 112 is a multiple of 200
theorem smallest_k_sum_of_squares_multiple_of_200 :
  ∃ k : ℕ, sum_of_squares k = sum_of_squares 112 ∧ 200 ∣ sum_of_squares 112 :=
sorry

end smallest_k_sum_of_squares_multiple_of_200_l73_73631


namespace geometric_sequence_and_sum_l73_73525

theorem geometric_sequence_and_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)
  (h : ∀ n, 4 * a n - 3 * S n = 2) (h1 : ∀ {n}, n ≥ 2 → a n = 4 * a (n - 1))
  (h2 : a 1 = 2) (h3 : ∀ n, b n = a n * ((n / 2) + 1)) :
  ∃ r : ℕ, (∀ n, a n = (r : ℕ) ^ n) ∧ 
  ∀ n, T n = ∑ k in finset.range n, b k :=
sorry

end geometric_sequence_and_sum_l73_73525


namespace value_of_b_l73_73001

theorem value_of_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 105 :=
sorry

end value_of_b_l73_73001


namespace emily_spent_12_dollars_l73_73817

variables (cost_per_flower : ℕ)
variables (roses : ℕ)
variables (daisies : ℕ)

def total_flowers : ℕ := roses + daisies

def total_cost : ℕ := total_flowers * cost_per_flower

theorem emily_spent_12_dollars (h1 : cost_per_flower = 3)
                              (h2 : roses = 2)
                              (h3 : daisies = 2) :
  total_cost cost_per_flower roses daisies = 12 :=
by
  simp [total_cost, total_flowers, h1, h2, h3]
  sorry

end emily_spent_12_dollars_l73_73817


namespace evaluate_f_at_7_l73_73940

def f (x : ℝ) : ℝ := (x + 2) / (4 * x - 5)

theorem evaluate_f_at_7 : f 7 = 9 / 23 :=
by
  unfold f
  norm_num
  sorry

end evaluate_f_at_7_l73_73940


namespace cubic_increasing_l73_73360

-- The definition of an increasing function
def increasing_function (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- The function y = x^3
def cubic_function (x : ℝ) : ℝ := x^3

-- The statement we want to prove
theorem cubic_increasing : increasing_function cubic_function :=
sorry

end cubic_increasing_l73_73360


namespace simultaneous_arrival_l73_73371

def walking_speed : ℝ := 1
def distance_MN : ℝ := 15
def alternative_speed (k : ℝ) : ℝ := k
def walking_distance (x : ℝ) : ℝ := x
def travel_distance (x : ℝ) : ℝ := 15 - x
def total_time (x k : ℝ) : ℝ := x + (15 - x) / k

theorem simultaneous_arrival (k : ℝ) (x : ℝ) (hx : x = 60 / 11) : 
  total_time x k = 3 / 11 :=
by
  sorry

end simultaneous_arrival_l73_73371


namespace base_ten_representation_of_22_factorial_l73_73691

theorem base_ten_representation_of_22_factorial (V R C : ℕ) (h1 : C = 0)
  (h2 : (V + R + 46) % 9 = 0) (h3 : (-1 - V + R) % 11 = 0) :
  V + R + C = 8 :=
by
  -- proof steps will ensure conditions hold and the final statement is proven
  sorry

end base_ten_representation_of_22_factorial_l73_73691


namespace angle_bisectors_perpendicular_l73_73522

variables {A B C D K L : Type*} [MetricSpace A] [MetricSpace B] 
[MetricSpace C] [MetricSpace D] [MetricSpace K] [MetricSpace L]

-- Assume the cyclic quadrilateral and the points K and L
axiom cyclic_quadrilateral (A B C D : Type*) : Prop 
axiom intersections (K L : Type*) : Prop

-- Assume the given conditions
axiom cyclic_quadrilateral_conditions : cyclic_quadrilateral A B C D
axiom intersection_conditions : intersections K L

-- The statement that needs proving
theorem angle_bisectors_perpendicular :
  (cyclic_quadrilateral A B C D) → 
  (intersections K L) → 
  ∀ (angle_bisector_BKC angle_bisector_BLA : Type*), 
  Metric.angle angle_bisector_BKC angle_bisector_BLA = π / 2 :=
by {
  assume h₁ h₂,
  -- Proof steps would go here, skipped with 'sorry'
  sorry
}

end angle_bisectors_perpendicular_l73_73522


namespace missy_more_claims_than_john_l73_73827

theorem missy_more_claims_than_john :
  let jan_capacity := 20
  let john_capacity := jan_capacity + 0.3 * jan_capacity
  let missy_capacity := 41
  missy_capacity - john_capacity = 15 := by
  let jan_capacity := 20
  let john_capacity := jan_capacity + 0.3 * jan_capacity
  let missy_capacity := 41
  -- Define Jan's and Missy's claims capacities
  have h1 : john_capacity = 20 + 0.3 * 20 := by sorry
  have h2 : missy_capacity = 41 := by sorry
  -- Proof
  calc
    missy_capacity - john_capacity
      = 41 - john_capacity : by sorry
  ... = 41 - (20 + 0.3 * 20) : by sorry
  ... = 41 - 26 : by sorry
  ... = 15 : by sorry

end missy_more_claims_than_john_l73_73827


namespace cost_of_soft_drink_l73_73822

theorem cost_of_soft_drink 
  (n : ℕ) -- number of candy bars
  (price_per_candy_bar : ℕ) -- price per candy bar
  (total_spent : ℕ) -- total amount spent
  (total_cost_candy_bars : ℕ) -- total cost of candy bars
  (cost_soft_drink : ℕ) : 
  (n = 5) → 
  (price_per_candy_bar = 5) →
  (total_spent = 27) → 
  (total_cost_candy_bars = n * price_per_candy_bar) →
  (cost_soft_drink = total_spent - total_cost_candy_bars) →
  cost_soft_drink = 2 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  exact sorry

end cost_of_soft_drink_l73_73822


namespace part1_part2_l73_73891

noncomputable def f (a : ℝ) (x : ℝ) := Real.logBase a x
noncomputable def g (a : ℝ) (x : ℝ) := a ^ x
noncomputable def h (x : ℝ) := x / Real.exp x

theorem part1
  (a : ℝ) 
  (ha : a = Real.exp) :
  (∀ x : ℝ, x ∈ Set.Ioi 0 → x < 1 → deriv (λ x, h x) x > 0) ∧
  (∀ x : ℝ, x ∈ Set.Ioi 0 → x > 1 → deriv (λ x, h x) x < 0) ∧
  (∀ t : ℝ, t ∈ Set.Icc 0 Real.exp → (∃ x : ℝ, x = 1 → h x = 1 / Real.exp)) := 
by
  sorry

theorem part2
  (a : ℝ)
  (ha : 1 < a) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
  (a^x₁ - (f a x₁) / x₁ = 1) ∧
  (a^x₂ - (f a x₂) / x₂ = 1)) ↔
  (1 < a ∧ a < Real.exp (1 / Real.exp)) :=
by
  sorry

end part1_part2_l73_73891


namespace jeans_speed_correct_l73_73065

/-- Define distances and speeds -/
def total_distance := 2 * d
def chantal_distance1 := (3 / 4) * total_distance
def chantal_distance2 := (1 / 4) * total_distance
def chantal_speed1 := 6
def chantal_speed2 := 2
def chantal_return_speed := 4
def meeting_point := (1 / 4) * d

/-- Define the times taken by Chantal -/
def time1 := chantal_distance1 / chantal_speed1
def time2 := chantal_distance2 / chantal_speed2
def time3 := meeting_point / chantal_return_speed
def total_time := time1 + time2 + time3

/-- Calculate Jean's average speed -/
def jeans_distance := meeting_point
def jeans_avg_speed := jeans_distance / total_time

/-- Statement that we need to prove -/
theorem jeans_speed_correct (d : ℕ) : jeans_avg_speed = (4 / 5) := by
  sorry

end jeans_speed_correct_l73_73065


namespace volumes_not_equal_smallest_k_l73_73521

-- Definitions for the given conditions
noncomputable def cone_volume (R m : ℝ) := (1/3) * π * R^2 * m
noncomputable def cylinder_volume (r : ℝ) := 2 * π * r^3

-- a) Prove that V_1 cannot be equal to V_2.
theorem volumes_not_equal (R m r : ℝ) (h_r : r = R * m / (R + sqrt(R^2 + m^2))) :
  cone_volume R m ≠ cylinder_volume r :=
sorry

-- b) Determine the smallest value of the number k for which the equation V_1 = k V_2 holds
noncomputable def k_min := 4 / 3

theorem smallest_k (R m r : ℝ) (h_r : r = R * m / (R + sqrt(R^2 + m^2))) :
  ∃ k : ℝ, (cone_volume R m = k * cylinder_volume r) ∧ k = k_min :=
sorry

end volumes_not_equal_smallest_k_l73_73521


namespace part1_part2_part3_l73_73163

def U_n (n : ℕ) : finset (fin n → bool) :=
  (finset.pi_finset (finset.range n) (λ _, {0, 1}))

def star {n : ℕ} (α β : fin n → bool) : ℕ :=
  finset.sum (finset.range n) (λ i, max (α i) (β i))

theorem part1 (β : fin 3 → bool) :
  star (λ i, if i = 0 then 0 else if i = 1 then 1 else 0)
       β = 3 ↔ β = (λ i, 1) :=
sorry

theorem part2 (n : ℕ) (α β : fin n → bool)
  (h1 : star α α + star β β = n) :
  max (star α β) = n ∧
  min (star α β) = (if even n then n / 2 else (n + 1) / 2) :=
sorry

theorem part3 (n : ℕ) (S : finset (fin n → bool))
  (h2 : ∀ ⦃α β⦄, α ∈ S → β ∈ S → α ≠ β → star α β ≥ n) :
  S.card ≤ n + 1 :=
sorry

end part1_part2_part3_l73_73163


namespace number_of_solutions_l73_73860

def f (x : ℝ) : ℝ := (1 / (x - 1)) + (2 / (x - 2)) + (3 / (x - 3)) + (4 / (x - 4)) + (5 / (x - 5)) + (6 / (x - 6)) + (7 / (x - 7)) + (8 / (x - 8)) + (9 / (x - 9)) + (10 / (x - 10))

theorem number_of_solutions : ∃ n : ℕ, n = 11 ∧ (∃ s : Fin n.succ → ℝ, ∀ i, f (s i) = 2 * (s i)) :=
by
  sorry

end number_of_solutions_l73_73860


namespace molecular_weight_of_CaH2_l73_73826

/-
  The molecular weight of Calcium hydride (CaH2) taking into account the natural isotopic abundance of Calcium isotopes and the atomic weight of hydrogen.
-/

noncomputable def molecular_weight_CaH2 : ℝ :=
  let avg_weight_Ca :=
    40 * 0.96941 + 44 * 0.02086 + 42 * 0.00647 + 48 * 0.00187 + 43 * 0.00135 + 46 * 0.00004
  in avg_weight_Ca + 2 * 1.008

theorem molecular_weight_of_CaH2 :
  molecular_weight_CaH2 = 42.13243 :=
by
  sorry

end molecular_weight_of_CaH2_l73_73826


namespace residue_intersection_l73_73506

def d (X : set ℕ) : ℝ :=
  Sup {c | c ∈ set.Icc 0 1 ∧ ∀ a < c, ∀ n₀ ∈ ℕ, ∃ m r ∈ ℕ, r ≥ n₀ ∧ (X ∩ set.Icc m (m + r)).card / r ≥ a}

theorem residue_intersection (E F : set ℕ) (dE dF : ℝ) (h1 : d E = dE) (h2 : d F = dF)
  (h3 : dE * dF > 1 / 4) :
  ∀ (p : ℕ) (hp : nat.prime p) (k : ℕ), ∃ (m ∈ E) (n ∈ F), m % p^k = n % p^k :=
begin
  sorry
end

end residue_intersection_l73_73506


namespace range_of_m_l73_73532

theorem range_of_m (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing : ∀ x y ∈ Icc (-2 : ℝ) 0, x < y → f y < f x) 
  (h_condition : ∀ m, f (1 - m) + f (1 - m^2) < 0) : 
  {m : ℝ | -1 ≤ m ∧ m < 1} :=
by {
  -- the proof will be inserted here
  sorry
}

end range_of_m_l73_73532


namespace circle_radius_of_right_triangle_l73_73896

theorem circle_radius_of_right_triangle 
  (A B C : Type)
  (hABC : ∀ {AB BC CA : ℝ}, AB = 4 ∧ BC = 3 ∧ CA = 5) 
  (hOmega : ∀ {ω : Circle ℝ}, ω.passesThrough A ∧ ω.tangentAt C BC):
  ∃ r : ℝ, r = 25 / 8 :=
by
  sorry

end circle_radius_of_right_triangle_l73_73896


namespace price_of_A_is_40_l73_73213

theorem price_of_A_is_40
  (p_a p_b : ℕ)
  (h1 : p_a = 2 * p_b)
  (h2 : 400 / p_a = 400 / p_b - 10) : p_a = 40 := 
by
  sorry

end price_of_A_is_40_l73_73213


namespace range_of_a_range_of_c_l73_73511

variables (a b c : ℝ)
hypothesis (h1 : a > b)
hypothesis (h2 : b > 0)
hypothesis (h3 : a + 2 * b = 3)
definition b_expr : ℝ := (3 - a) / 2
definition c_expr : ℝ := -3 * a + 2 * b_expr

theorem range_of_a : 1 < a ∧ a < 3 :=
by
  have h4 : b = b_expr, from sorry
  sorry

theorem range_of_c : -9 < c_expr ∧ c_expr < -1 :=
by
  have h5 : c = c_expr, from sorry
  have h6 : 1 < a ∧ a < 3, from range_of_a h1 h2 h3
  sorry

end range_of_a_range_of_c_l73_73511


namespace zhang_income_tax_l73_73775

theorem zhang_income_tax :
  ∀ (salary_after_tax : ℝ) (base_income : ℝ) (tax_rate : ℝ) (total_income : ℝ) (tax_paid : ℝ),
  salary_after_tax = 4761 →
  base_income = 3500 →
  tax_rate = 0.03 →
  total_income = base_income + (salary_after_tax - base_income) / (1 - tax_rate) →
  tax_paid = ((salary_after_tax - base_income) / (1 - tax_rate) - base_income) * tax_rate →
  tax_paid = 39 :=
begin
  intros salary_after_tax base_income tax_rate total_income tax_paid
    h_salary_after_tax h_base_income h_tax_rate h_total_income h_tax_paid,
  sorry
end

end zhang_income_tax_l73_73775


namespace suitable_graph_for_milk_components_l73_73669

theorem suitable_graph_for_milk_components
  (water : ℝ)
  (protein : ℝ)
  (fat : ℝ)
  (lactose : ℝ)
  (other_components : ℝ)
  (h_water : water = 0.82)
  (h_protein : protein = 0.043)
  (h_fat : fat = 0.06)
  (h_lactose : lactose = 0.07)
  (h_other_components : other_components = 0.007) :
  ∃ g : String, g = "pie chart" :=
by
  use "pie chart"
  sorry

end suitable_graph_for_milk_components_l73_73669


namespace parallelogram_angle_sum_l73_73612

theorem parallelogram_angle_sum (ABCD : Type) (A B C D : ABCD) 
  (angle : ABCD → ℝ) (h_parallelogram : true) (h_B : angle B = 60) :
  ¬ (angle C + angle A = 180) :=
sorry

end parallelogram_angle_sum_l73_73612


namespace points_on_line_divisibility_l73_73285

theorem points_on_line_divisibility
  (x y x1 y1 x2 y2 x3 y3 : ℤ) 
  (hxy : x * y ≡ 1 [MOD 1979])
  (h1 : x1 * y1 ≡ 1 [MOD 1979])
  (h2 : x2 * y2 ≡ 1 [MOD 1979])
  (h3 : x3 * y3 ≡ 1 [MOD 1979])
  (line_eq : ∃ (a b c : ℤ), gcd a b = 1 ∧ a * x1 + b * y1 = c ∧ a * x2 + b * y2 = c ∧ a * x3 + b * y3 = c) :
  (x1 - x2) % 1979 = 0 ∧ (y1 - y2) % 1979 = 0 ∨
  (x1 - x3) % 1979 = 0 ∧ (y1 - y3) % 1979 = 0 ∨
  (x2 - x3) % 1979 = 0 ∧ (y2 - y3) % 1979 = 0 :=
sorry

end points_on_line_divisibility_l73_73285


namespace tenth_odd_multiple_of_five_divisible_by_seven_is_670_l73_73749

/-- An auxiliary function to help state the predicate for an odd multiple of 5 that is also divisible by 7 --/
def odd_multiple_of_five_divisible_by_seven (x : ℕ) : Prop :=
  x % 2 = 1 ∧ x % 5 = 0 ∧ x % 7 = 0

/-- The nth positive integer that is both odd and a multiple of 5, and also divisible by 7 --/
def nth_odd_multiple_of_five_divisible_by_seven (n : ℕ) : ℕ :=
  nat.find (exists_index_odd_multiple_of_five_divisible_by_seven n 1)

noncomputable def nth_is_odd_multiple_of_five_divisible_by_seven (n : ℕ) : Prop :=
  odd_multiple_of_five_divisible_by_seven (nth_odd_multiple_of_five_divisible_by_seven n)

/-- Lean 4 statement asserting that the tenth positive integer satisfying the conditions is 670 --/
theorem tenth_odd_multiple_of_five_divisible_by_seven_is_670 :
  nth_odd_multiple_of_five_divisible_by_seven 10 = 670 :=
sorry

/-- Auxiliary lemma stating the nth term existence for odd multiples of 5 and divisible by 7 --/
lemma exists_index_odd_multiple_of_five_divisible_by_seven (n m : ℕ) : 
  ∃ k, 
  odd_multiple_of_five_divisible_by_seven k ∧ 
  (list.filter (λ x, odd_multiple_of_five_divisible_by_seven x) (list.range (10 * m + 1))).length = n :=
sorry

end tenth_odd_multiple_of_five_divisible_by_seven_is_670_l73_73749


namespace coefficient_x3y5_l73_73083

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition for the binomial expansion term of (x-y)^7
def expansion_term (r : ℕ) : ℤ := 
  (binom 7 r) * (-1) ^ r

-- The target coefficient for the term x^3 y^5 in (x+y)(x-y)^7
theorem coefficient_x3y5 :
  (expansion_term 5) * 1 + (expansion_term 4) * 1 = 14 :=
by
  -- Proof to be filled in
  sorry

end coefficient_x3y5_l73_73083


namespace solution_l73_73301

noncomputable def problem : Prop :=
  ∃ (α : ℝ) (d e f : ℝ), 
    let DE := 17 
    let EF := 18 
    let FD := 19 
    let α := ∠QDE = ∠QEF = ∠QFD  
    (17 * d + 18 * e + 19 * f) * Math.sin(α) = 216 ∧ 
    (17 * d + 18 * e + 19 * f) * Math.cos(α) = 487 ∧ 
    tan(α) = 216 / 487

theorem solution : problem :=
  sorry

end solution_l73_73301


namespace simplify_fraction_l73_73323

theorem simplify_fraction : 
  ∃ (c d : ℤ), ((∀ m : ℤ, (6 * m + 12) / 3 = c * m + d) ∧ c = 2 ∧ d = 4) → 
  c / d = 1 / 2 :=
by
  sorry

end simplify_fraction_l73_73323


namespace arrangements_15_cents_l73_73087

def numArrangements (n : ℕ) : ℕ :=
  sorry  -- Function definition which outputs the number of arrangements for sum n

theorem arrangements_15_cents : numArrangements 15 = X :=
  sorry  -- Replace X with the correct calculated number

end arrangements_15_cents_l73_73087


namespace sum_of_consecutive_integers_bound_sqrt_40_l73_73939

theorem sum_of_consecutive_integers_bound_sqrt_40 (a b : ℤ) (h₁ : a < Real.sqrt 40) (h₂ : Real.sqrt 40 < b) (h₃ : b = a + 1) : a + b = 13 :=
by
  sorry

end sum_of_consecutive_integers_bound_sqrt_40_l73_73939


namespace symmetric_line_l73_73321

theorem symmetric_line (y : ℝ → ℝ) (h : ∀ x, y x = 2 * x + 1) :
  ∀ x, y (-x) = -2 * x + 1 :=
by
  -- Proof skipped
  sorry

end symmetric_line_l73_73321


namespace count_ordered_pairs_l73_73107

theorem count_ordered_pairs :
  let positive_integers := {n : ℕ // n > 0},
      valid_pairs := {(b, c) ∈ positive_integers × positive_integers | b^2 - 4 * c ≤ 0 ∧ c^2 - 4 * b ≤ 0}
  in (set.finite valid_pairs ∧ set.card valid_pairs = 6) := 
by
  sorry

end count_ordered_pairs_l73_73107


namespace shelves_per_case_l73_73993

noncomputable section

-- Define the total number of ridges
def total_ridges : ℕ := 8640

-- Define the number of ridges per record
def ridges_per_record : ℕ := 60

-- Define the number of records per shelf when the shelf is 60% full
def records_per_shelf : ℕ := (60 * 20) / 100

-- Define the number of ridges per shelf
def ridges_per_shelf : ℕ := records_per_shelf * ridges_per_record

-- Given 4 cases, we need to determine the number of shelves per case
theorem shelves_per_case (cases shelves : ℕ) (h₁ : cases = 4) (h₂ : shelves * ridges_per_shelf = total_ridges) :
  shelves / cases = 3 := by
  sorry

end shelves_per_case_l73_73993


namespace find_annual_interest_rate_l73_73745

noncomputable def annual_interest_rate (P A n t : ℝ) : ℝ :=
  2 * ((A / P)^(1 / (n * t)) - 1)

theorem find_annual_interest_rate :
  Π (P A : ℝ) (n t : ℕ), P = 600 → A = 760 → n = 2 → t = 4 →
  annual_interest_rate P A n t = 0.06020727 :=
by
  intros P A n t hP hA hn ht
  rw [hP, hA, hn, ht]
  unfold annual_interest_rate
  sorry

end find_annual_interest_rate_l73_73745


namespace find_log2_x_l73_73519

theorem find_log2_x (x: ℝ) (h: log (2 * x) / log (5 * x) = log (8 * x) / log (625 * x)) :
  log 2 x = log 5 / (2 * log 2 - 3 * log 5) :=
by
  sorry

end find_log2_x_l73_73519


namespace max_cake_boxes_l73_73051

theorem max_cake_boxes 
  (L_carton W_carton H_carton : ℕ) (L_box W_box H_box : ℕ)
  (h_carton : L_carton = 25 ∧ W_carton = 42 ∧ H_carton = 60)
  (h_box : L_box = 8 ∧ W_box = 7 ∧ H_box = 5) : 
  (L_carton * W_carton * H_carton) / (L_box * W_box * H_box) = 225 := by 
  sorry

end max_cake_boxes_l73_73051


namespace solve_for_x_l73_73864

theorem solve_for_x (x : ℝ) (h₀ : x > 0) (h₁ : 1 / 2 * x * (3 * x) = 96) : x = 8 :=
sorry

end solve_for_x_l73_73864


namespace sum_of_possible_e_l73_73018

theorem sum_of_possible_e (x : ℕ) (h1 : 81 ≤ x) (h2 : x < 729) :
  ∃ e : ℕ, 3^4 ≤ x ∧ x < 3^5 ∨ 3^5 ≤ x ∧ x < 3^6 ∨ 3^6 ≤ x ∧ x < 3^7 ∧ 
  (e = 5 ∨ e = 6 ∨ e = 7) ∧ e ∈ {5, 6, 7} → e ∈ {5, 6, 7} := 
  sorry

end sum_of_possible_e_l73_73018


namespace sum_of_angles_in_polygons_l73_73252

theorem sum_of_angles_in_polygons :
  let angle_pentagon := 108
  let angle_triangle := 60
  angle_pentagon + angle_triangle = 168 := 
by
  let angle_pentagon := 180 * (5 - 2) / 5  -- Calculation for pentagon
  let angle_triangle := 180 * (3 - 2) / 3  -- Calculation for triangle
  have h1 : angle_pentagon = 108 := by sorry -- Verification of pentagon angle
  have h2 : angle_triangle = 60 := by sorry  -- Verification of triangle angle
  rw [h1, h2]
  norm_num

end sum_of_angles_in_polygons_l73_73252


namespace arithmetic_sequences_and_min_m_l73_73123

theorem arithmetic_sequences_and_min_m
    (a : ℕ → ℕ)
    (b : ℕ → ℕ)
    (A1 : a 1 = 1)
    (A2 : ∀ n, a n = 3^(n-1))
    (A3 : ∀ n, b n = 2 * n - 1)
    (S1 : ∀ n, (∑ i in (Finset.range n).map Nat.succ, a i * b i) = (n-1) * 3^n + 1)
    (m : ℝ)
    (H : ∀ n, m * a n ≥ b n - 8)
    : m ≥ 1 / 81 :=
sorry

end arithmetic_sequences_and_min_m_l73_73123


namespace max_y_value_l73_73948

theorem max_y_value (x : ℝ) : ∃ y : ℝ, y = -x^2 + 4 * x + 3 ∧ y ≤ 7 :=
by
  sorry

end max_y_value_l73_73948


namespace total_number_of_triples_equals_338_l73_73863

theorem total_number_of_triples_equals_338 :
  let count_triples := λ n : ℕ, (∑ d in ({1} : Set ℕ), (if d = n * n then 1 else 0)) in
  count_triples 2012 + count_triples 2013 = 338 :=
by
  sorry

end total_number_of_triples_equals_338_l73_73863


namespace investment_plan_optimization_l73_73007

-- Define the given conditions.
def max_investment : ℝ := 100000
def max_loss : ℝ := 18000
def max_profit_A_rate : ℝ := 1.0     -- 100%
def max_profit_B_rate : ℝ := 0.5     -- 50%
def max_loss_A_rate : ℝ := 0.3       -- 30%
def max_loss_B_rate : ℝ := 0.1       -- 10%

-- Define the investment amounts.
def invest_A : ℝ := 40000
def invest_B : ℝ := 60000

-- Calculate profit and loss.
def profit : ℝ := (invest_A * max_profit_A_rate) + (invest_B * max_profit_B_rate)
def loss : ℝ := (invest_A * max_loss_A_rate) + (invest_B * max_loss_B_rate)
def total_investment : ℝ := invest_A + invest_B

-- Prove the required statement.
theorem investment_plan_optimization : 
    total_investment ≤ max_investment ∧ loss ≤ max_loss ∧ profit = 70000 :=
by
  simp [total_investment, profit, loss, invest_A, invest_B, 
    max_investment, max_profit_A_rate, max_profit_B_rate, 
    max_loss_A_rate, max_loss_B_rate, max_loss]
  sorry

end investment_plan_optimization_l73_73007


namespace set_S_infinite_l73_73010

-- Definition of a power
def is_power (n : ℕ) : Prop := 
  ∃ (a k : ℕ), a > 0 ∧ k ≥ 2 ∧ n = a^k

-- Definition of the set S, those integers which cannot be expressed as the sum of two powers
def in_S (n : ℕ) : Prop := 
  ¬ ∃ (a b k m : ℕ), a > 0 ∧ b > 0 ∧ k ≥ 2 ∧ m ≥ 2 ∧ n = a^k + b^m

-- The theorem statement asserting that S is infinite
theorem set_S_infinite : Infinite {n : ℕ | in_S n} :=
sorry

end set_S_infinite_l73_73010


namespace count_valid_numbers_l73_73562

def is_allowed_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 4 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9

def contains_only_allowed_digits (n : ℕ) : Prop :=
  n.digits 10 ∀ digit, is_allowed_digit digit

theorem count_valid_numbers : 
  finset.card (finset.filter contains_only_allowed_digits (finset.range 9000)) = 1079 :=
by
  exact sorry

end count_valid_numbers_l73_73562


namespace triangle_area_l73_73255

theorem triangle_area {A B C M : Type*} [triangle A B C] [segment AB AC AM] 
  (h1 : AB.length = 9) (h2 : AC.length = 17) (h3 : AM.length = 12) :
  area ABC = 40 * real.sqrt 2 :=
by
  sorry

end triangle_area_l73_73255


namespace derivative_of_f_l73_73320

open Real

noncomputable def f (x : ℝ) : ℝ := (2 * π * x) ^ 2

theorem derivative_of_f : deriv f = λ x, 8 * π ^ 2 * x :=
by
  sorry

end derivative_of_f_l73_73320


namespace periodic_function_example_function_correct_l73_73653

theorem periodic_function (a : ℝ) (f : ℝ → ℝ) (h1 : 0 < a) 
    (h2 : ∀ x : ℝ, f(x + a) = 1/2 + real.sqrt(f(x) - f(x)^2)) : ∃ b > 0, ∀ x, f(x + b) = f(x) :=
by
  sorry

noncomputable def example_function : ℝ → ℝ := λ x,
  if ∃ n : ℤ, (2 * n : ℝ) ≤ x ∧ x < 2 * n + 1 then 1 / 2 else 1

theorem example_function_correct (x : ℝ) :
    example_function(x + 1) = 1/2 + real.sqrt(example_function(x) - example_function(x)^2) :=
by
  sorry

end periodic_function_example_function_correct_l73_73653


namespace unique_function_l73_73491

noncomputable def find_function (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 → b > 0 → a + b > 2019 → a + f b ∣ a^2 + b * f a

theorem unique_function (r : ℕ) (f : ℕ → ℕ) :
  find_function f → (∀ x : ℕ, f x = r * x) :=
sorry

end unique_function_l73_73491


namespace range_of_x_l73_73484

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 2^x

theorem range_of_x (x : ℝ) : f(x) + f(x - 1/2) > 1 → x > -1/4 := 
sorry

end range_of_x_l73_73484


namespace saving_rate_l73_73758

theorem saving_rate (initial_you : ℕ) (initial_friend : ℕ) (friend_save : ℕ) (weeks : ℕ) (final_amount : ℕ) :
  initial_you = 160 →
  initial_friend = 210 →
  friend_save = 5 →
  weeks = 25 →
  final_amount = initial_you + weeks * 7 →
  final_amount = initial_friend + weeks * friend_save →
  7 = (final_amount - initial_you) / weeks :=
by
  intros initial_you_val initial_friend_val friend_save_val weeks_val final_amount_equals_you final_amount_equals_friend
  rw [initial_you_val, initial_friend_val, friend_save_val, weeks_val] at *
  have h: 160 + 25 * 7 = 210 + 25 * 5, by sorry
  have final_amount_val: final_amount = 335, by
    rw [final_amount_equals_you]
    exact h
  rw [final_amount_val] at *
  exact sorry

end saving_rate_l73_73758


namespace cos_smallest_angle_correct_l73_73142

noncomputable def cos_smallest_angle (a : ℝ) : ℝ :=
  let A := 2 * Real.arccos ((a + 6) / (2 * a))
  in Real.cos A / 2

theorem cos_smallest_angle_correct (a : ℝ) (h₁ : a > 0) (h₂ : Real.arccos ((a + 6) / (2 * a)) < Real.pi / 2) :
  cos_smallest_angle 12 = 3 / 4 :=
by sorry

end cos_smallest_angle_correct_l73_73142


namespace solution_count_l73_73563

theorem solution_count :
  {x y : Int | abs (x * y) + abs (x - y) = 2 ∧ -10 ≤ x ∧ x ≤ 10 ∧ -10 ≤ y ∧ y ≤ 10}.card = 4 :=
sorry

end solution_count_l73_73563


namespace area_under_f2_l73_73651

noncomputable def f0 (x : ℝ) : ℝ :=
  |x|

noncomputable def f1 (x : ℝ) : ℝ :=
  |f0 x - 1|

noncomputable def f2 (x : ℝ) : ℝ :=
  |f1 x - 2|

theorem area_under_f2 :
  let area := ∫ x in (set.Icc l r), f2 x 
  area = 7 :=
sorry

end area_under_f2_l73_73651


namespace range_of_t_l73_73533

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 := 
by
  sorry

end range_of_t_l73_73533


namespace y1_lt_y2_l73_73191

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l73_73191


namespace problem_statement_l73_73500

-- Define the function f1 as the square of the sum of the digits of k
def f1 (k : Nat) : Nat :=
  let sum_digits := (Nat.digits 10 k).sum
  sum_digits * sum_digits

-- Define the recursive function f_{n+1}(k) = f1(f_n(k))
def fn : Nat → Nat → Nat
| 0, k => k
| n+1, k => f1 (fn n k)

theorem problem_statement : fn 1991 (2^1990) = 256 :=
sorry

end problem_statement_l73_73500


namespace number_of_white_balls_l73_73226

-- Definitions based on the conditions
def total_balls : ℕ := 50
def prob_red : ℝ := 0.15
def prob_black : ℝ := 0.45

-- The Lean statement to prove:
theorem number_of_white_balls :
  let prob_white := 1 - prob_red - prob_black in
  let white_balls := total_balls * prob_white in
  white_balls = 20 := 
by
  sorry

end number_of_white_balls_l73_73226


namespace harbor_distance_l73_73297

-- Definitions from conditions
variable (d : ℝ)

-- Define the assumptions
def condition_dave := d < 10
def condition_elena := d > 9

-- The proof statement that the interval for d is (9, 10)
theorem harbor_distance (hd : condition_dave d) (he : condition_elena d) : d ∈ Set.Ioo 9 10 :=
sorry

end harbor_distance_l73_73297


namespace emily_total_spent_l73_73815

-- Define the given conditions.
def cost_per_flower : ℕ := 3
def num_roses : ℕ := 2
def num_daisies : ℕ := 2

-- Calculate the total number of flowers and the total cost.
def total_flowers : ℕ := num_roses + num_daisies
def total_cost : ℕ := total_flowers * cost_per_flower

-- Statement: Prove that Emily spent 12 dollars.
theorem emily_total_spent : total_cost = 12 := by
  sorry

end emily_total_spent_l73_73815


namespace total_production_first_four_days_max_min_production_difference_total_wage_for_week_l73_73016

open Int

/-- Problem Statement -/
def planned_production : Int := 220

def production_change : List Int :=
  [5, -2, -4, 13, -10, 16, -9]

/-- Proof problem for total production in the first four days -/
theorem total_production_first_four_days :
  let first_four_days := production_change.take 4
  let total_change := first_four_days.sum
  let planned_first_four_days := planned_production * 4
  planned_first_four_days + total_change = 892 := 
by
  sorry

/-- Proof problem for difference in production between highest and lowest days -/
theorem max_min_production_difference :
  let max_change := production_change.maximum.getD 0
  let min_change := production_change.minimum.getD 0
  max_change - min_change = 26 := 
by
  sorry

/-- Proof problem for total wage calculation for the week -/
theorem total_wage_for_week :
  let total_change := production_change.sum
  let planned_week_total := planned_production * 7
  let actual_total := planned_week_total + total_change
  let base_wage := actual_total * 100
  let additional_wage := total_change * 20
  base_wage + additional_wage = 155080 := 
by
  sorry

end total_production_first_four_days_max_min_production_difference_total_wage_for_week_l73_73016


namespace coeff_x2_in_qx3_l73_73161

-- Define the given polynomial q(x)
def q (x : ℝ) : ℝ := x^5 - 4*x^3 + 5*x^2 - 6*x + 3

-- Statement to prove the coefficient of x^2 in (q(x))^3 is 540
theorem coeff_x2_in_qx3 : 
  (coeff (mk (list.range 3) [(0, (q 0)^3), (2, 540)]) : ℝ) = 540 :=
sorry

end coeff_x2_in_qx3_l73_73161


namespace relationship_y1_y2_l73_73198

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l73_73198


namespace smallest_k_sum_sequence_l73_73770

theorem smallest_k_sum_sequence (n : ℕ) (h : 100 = (n + 1) * (2 * 9 + n) / 2) : k = 9 := 
sorry

end smallest_k_sum_sequence_l73_73770


namespace total_cost_pencils_and_pens_l73_73430

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end total_cost_pencils_and_pens_l73_73430


namespace math_problem_statement_l73_73504

def is_composite (k : ℕ) : Prop := ∃ m n : ℕ, 1 < m ∧ 1 < n ∧ k = m * n

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible (a b : ℕ) : Prop := ∃ k, a = b * k

noncomputable def satisfies_conditions (n : ℕ) : Prop :=
0 < n ∧ n ≤ 30 ∧ divisible (factorial n) (sum_first_n n) ∧ is_composite (n + 1)

theorem math_problem_statement : (finset.Icc 1 30).filter satisfies_conditions = finset.range 30.filter satisfies_conditions ∧ finset.card (finset.Icc 1 30).filter satisfies_conditions = 18 := by
  sorry

end math_problem_statement_l73_73504


namespace height_of_table_l73_73225

/-- 
Given:
1. Combined initial measurement (l + h - w + t) = 40
2. Combined changed measurement (w + h - l + t) = 34
3. Width of each wood block (w) = 6 inches
4. Visible edge-on thickness of the table (t) = 4 inches
Prove:
The height of the table (h) is 33 inches.
-/
theorem height_of_table (l h t w : ℕ) (h_combined_initial : l + h - w + t = 40)
    (h_combined_changed : w + h - l + t = 34) (h_width : w = 6) (h_thickness : t = 4) : 
    h = 33 :=
by
  sorry

end height_of_table_l73_73225


namespace possible_to_color_rationals_l73_73839

noncomputable def can_color_rationals : Prop :=
∀ (color : ℚ → bool),
  (∀ (x : ℚ), x ≠ 0 → color x ≠ color (-x)) →
  (∀ (x : ℚ), x ≠ 1/2 → color x ≠ color (1 - x)) →
  (∀ (x : ℚ), x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → color x ≠ color (1 / x)) →
  true

theorem possible_to_color_rationals : can_color_rationals :=
sorry

end possible_to_color_rationals_l73_73839


namespace largest_k_exists_l73_73748

noncomputable def largest_k := 3

theorem largest_k_exists :
  ∃ (k : ℕ), (k = largest_k) ∧ ∀ m : ℕ, 
    (∀ n : ℕ, ∃ a b : ℕ, m + n = a^2 + b^2) ∧ 
    (∀ n : ℕ, ∃ seq : ℕ → ℕ,
      (∀ i : ℕ, seq i = a^2 + b^2) ∧
      (∀ j : ℕ, m ≤ j → a^2 + b^2 ≠ 3 + 4 * j)
    ) := ⟨3, rfl, sorry⟩

end largest_k_exists_l73_73748


namespace find_a_l73_73682

noncomputable def normal_distribution
(xi : ℝ → MeasureTheory.Measure ℝ) (μ : ℝ) (σ : ℝ) :=
  xi = MeasureTheory.Measure.gausian μ σ

theorem find_a (xi : ℝ → MeasureTheory.Measure ℝ) (a : ℝ) :
  (normal_distribution xi 3 2) →
  (ProbabilityTheory.Measure.cdf xi (2 * a - 3) = 0.5) →
  (ProbabilityTheory.Measure.cdf xi (a + 2) = 0.5) →
  a = 7 / 3 :=
sorry

end find_a_l73_73682


namespace value_of_c_plus_d_l73_73941

noncomputable def g (c d : ℝ) : ℝ → ℝ := λ x, c * x + d
noncomputable def g_inv (c d : ℝ) : ℝ → ℝ := λ x, d * x + c

theorem value_of_c_plus_d (c d : ℝ) 
  (hg : ∀ x, g c d (g_inv c d x) = x)
  (hginv : ∀ x, g_inv c d (g c d x) = x) : 
  c + d = -2 :=
sorry

end value_of_c_plus_d_l73_73941


namespace cosine_of_dihedral_angle_l73_73737

theorem cosine_of_dihedral_angle
  (r : ℝ) -- radius of the smaller sphere
  (theta : ℝ) -- measure of the dihedral angle in radians
  (d : ℝ := 5 * r) -- distance between centers of the spheres
  (alpha : ℝ := real.pi / 3) -- angle = 60 degrees in radians
  (hx : d = 5 * r) -- centers of the spheres are at distance 5r.
  (halpha : alpha = real.pi / 3) -- the angle between the line joining the centers and the edge of the dihedral angle is 60 degrees
  (hx_proj : d * real.cos alpha = 2.5 * r) -- projection distance in the plane containing the edge
  : real.cos theta = 0.04 :=
sorry

end cosine_of_dihedral_angle_l73_73737


namespace Herman_bird_food_cups_l73_73173

theorem Herman_bird_food_cups
  (days_december days_january days_february : ℕ)
  (h_december : days_december = 31)
  (h_january : days_january = 31)
  (h_february : days_february = 28) :
  let total_days := days_december + days_january + days_february in
  let cups_per_day := 1 in
  total_days * cups_per_day = 90 :=
by
  sorry

end Herman_bird_food_cups_l73_73173


namespace solve_trig_eq_l73_73498

noncomputable def solution_set : set ℝ :=
{ x | x = (real.pi / 6) ∨ x = (5 * real.pi / 6)}

theorem solve_trig_eq (x : ℝ) (h : 0 < x ∧ x < real.pi) :
  (cos (2 * x) + sin x = 1) ↔ (x = real.pi / 6 ∨ x = 5 * real.pi / 6) :=
by
  sorry

end solve_trig_eq_l73_73498


namespace functional_equation_solution_l73_73492

theorem functional_equation_solution :
  ∀ f : ℕ → ℕ, 
  (∀ x y : ℕ, f(x + y) = f(x) + f(y)) →
  ∃ a : ℕ, ∀ x : ℕ, f(x) = a * x :=
by
  sorry

end functional_equation_solution_l73_73492


namespace find_values_l73_73489

-- Definitions of the given conditions
def circle (radius : ℝ) := {center : ℝ × ℝ // ∀ (P : ℝ × ℝ), dist center P = radius}
def tangent (c1 c2 : circle) := dist c1.center c2.center = c1.radius + c2.radius

noncomputable def problem_statement : Prop :=
  let A := circle 15
  let B := circle 5
  let C := circle 3
  let D := circle 3
  let E := circle (15)
  let F := circle 0
  -- Tangency conditions
  in tangent A B ∧
     tangent A C ∧
     tangent A D ∧
     tangent B E ∧
     tangent C E ∧
     tangent D E ∧ 
     (∃ x : ℝ, tangent B F ∧ tangent E F)

theorem find_values : problem_statement → (15 + 1 + 0) = 16 :=
by sorry

end find_values_l73_73489


namespace average_population_increase_l73_73446

-- Conditions
def population_2000 : ℕ := 450000
def population_2005 : ℕ := 467000
def years : ℕ := 5

-- Theorem statement
theorem average_population_increase :
  (population_2005 - population_2000) / years = 3400 := by
  sorry

end average_population_increase_l73_73446


namespace part_I_part_II_l73_73162

def setA (x : ℝ) : Prop := 0 ≤ x - 1 ∧ x - 1 ≤ 2

def setB (x : ℝ) (a : ℝ) : Prop := 1 < x - a ∧ x - a < 2 * a + 3

def complement_R (x : ℝ) (a : ℝ) : Prop := x ≤ 2 ∨ x ≥ 6

theorem part_I (a : ℝ) (x : ℝ) (ha : a = 1) : 
  setA x ∨ setB x a ↔ (1 ≤ x ∧ x < 6) ∧ 
  (setA x ∧ complement_R x a ↔ 1 ≤ x ∧ x ≤ 2) := 
by
  sorry

theorem part_II (a : ℝ) : 
  (∃ x, setA x ∧ setB x a) ↔ -2/3 < a ∧ a < 2 := 
by
  sorry

end part_I_part_II_l73_73162


namespace linear_function_points_relation_l73_73203

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ),
    (y1 = 2 * (-3) + 1) →
    (y2 = 2 * 4 + 1) →
    (y1 = -5) ∧ (y2 = 9) :=
by
  intros y1 y2 hy1 hy2
  split
  · exact hy1
  · exact hy2

end linear_function_points_relation_l73_73203


namespace problem_statement_l73_73894

variable (p q : Prop)

def p_definition : Prop := ∀ x : ℝ, Real.sin (π - x) = Real.sin x

def q_definition : Prop := ∀ (α β : ℝ), (0 < α ∧ α < π / 2) ∧ (0 < β ∧ β < π / 2) ∧ α > β → Real.sin α > Real.sin β

theorem problem_statement (p_true : p_definition p) (q_false : ¬ q_definition q) : p ∧ ¬ q :=
by
  -- The proof is omitted as requested
  sorry

end problem_statement_l73_73894


namespace max_volume_of_cone_l73_73102

theorem max_volume_of_cone (a : ℝ) (h r : ℝ) (ha : 0 < a)
  (h_eq : r^2 + h^2 = a^2) :
  let V := (1/3) * π * r^2 * h in
  ∃ h_max r_max, h_max = a / real.sqrt 3 ∧ r_max = real.sqrt(a^2 - h_max^2) ∧ 
    V = (2 * π * a^2 * real.sqrt 3) / 27 :=
begin
  -- Proof omitted
  sorry
end

end max_volume_of_cone_l73_73102


namespace arrangement_count_12_people_seating_l73_73402

theorem arrangement_count_12_people_seating :
  ∑ (a b : Fin 12), Multichoose 12 6 ∗ (5!).pow 2 = (12!.mul ((6!).pow 2)).div 6! 6! :=
sorry

end arrangement_count_12_people_seating_l73_73402


namespace intersection_point_l73_73513

/-- 
Prove that for any real numbers \( a \) and \( b \) with \( b > a \), 
the intersection point of the lines \( y = bx + a \) and \( y = ax + b \)
is \((1, a + b)\). 
-/
theorem intersection_point (a b : ℝ) (h : b > a) : ∃ (x y : ℝ), (x = 1) ∧ (y = a + b) ∧ (bx + a = y) ∧ (ax + b = y) :=
begin
  sorry
end

end intersection_point_l73_73513


namespace truncatedConeVolume_l73_73723

theorem truncatedConeVolume 
  (R r h : ℝ) 
  (hR_pos : 0 < R) 
  (hr_pos : 0 < r) 
  (hh_pos : 0 < h) 
  (R_eq : R = 10) 
  (r_eq : r = 5) 
  (h_eq : h = 8) : 
  (volume : ℝ) :=
  volume = (466.67 : ℝ) * Real.pi := 
by
  sorry

end truncatedConeVolume_l73_73723


namespace minimum_value_l73_73895

theorem minimum_value (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + 4 * y^2 + z^2 ≥ 1 / 3 :=
sorry

end minimum_value_l73_73895


namespace fraction_zero_when_x_is_three_l73_73381

theorem fraction_zero_when_x_is_three (x : ℝ) (h : x ≠ -3) : (x^2 - 9) / (x + 3) = 0 ↔ x = 3 :=
by 
  sorry

end fraction_zero_when_x_is_three_l73_73381


namespace sum_double_nested_series_l73_73072

theorem sum_double_nested_series :
  (∑ j : ℕ, ∑ k : ℕ, (2 : ℝ)^(-4 * k - 2 * j - (k + j)^2)) = (4 / 3 : ℝ) :=
by
  sorry

end sum_double_nested_series_l73_73072


namespace projection_of_A_onto_Oxz_is_B_l73_73985

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def projection_onto_Oxz (A : Point3D) : Point3D :=
  { x := A.x, y := 0, z := A.z }

theorem projection_of_A_onto_Oxz_is_B :
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  projection_onto_Oxz A = B :=
by
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  have h : projection_onto_Oxz A = B := rfl
  exact h

end projection_of_A_onto_Oxz_is_B_l73_73985


namespace sum_at_least_half_m_square_l73_73610

theorem sum_at_least_half_m_square (m : ℕ) (a : Fin m → Fin m → ℕ)
  (h : ∀ i j, a i j = 0 → ((∑ k, a i k) + (∑ k, a k j) - a i j) ≥ m) :
  (∑ i j, a i j) ≥ m^2 / 2 := 
sorry

end sum_at_least_half_m_square_l73_73610


namespace problem_am135_l73_73646

theorem problem_am135 :
  let x := 128 + 192 + 256 + 320 + 576 + 704 + 6464
  in (8 ∣ x ∧ 16 ∣ x ∧ 32 ∣ x ∧ 64 ∣ x) :=
by
  let x := 128 + 192 + 256 + 320 + 576 + 704 + 6464
  sorry

end problem_am135_l73_73646


namespace sum_at_least_half_m_square_l73_73611

theorem sum_at_least_half_m_square (m : ℕ) (a : Fin m → Fin m → ℕ)
  (h : ∀ i j, a i j = 0 → ((∑ k, a i k) + (∑ k, a k j) - a i j) ≥ m) :
  (∑ i j, a i j) ≥ m^2 / 2 := 
sorry

end sum_at_least_half_m_square_l73_73611


namespace nuts_division_pattern_l73_73766

noncomputable def smallest_number_of_nuts : ℕ := 15621

theorem nuts_division_pattern :
  ∃ N : ℕ, N = smallest_number_of_nuts ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 
  (∃ M : ℕ, (N - k) % 4 = 0 ∧ (N - k) / 4 * 5 + 1 = N) := sorry

end nuts_division_pattern_l73_73766


namespace seq_a_ratio_sum_seq_b_eq_l73_73886

-- Define the sequences and conditions
axiom seq_a (n : ℕ) : ℝ
axiom h1 : seq_a 1 = 1
axiom h2 : ∀ n, seq_a n * seq_a (n + 1) = 3^n

-- Statement for the first proof
theorem seq_a_ratio (n : ℕ) (hn : n ≥ 2) : seq_a (n + 1) / seq_a (n - 1) = 3 := 
by 
  sorry

-- Define the sequence b_n
def seq_b (n : ℕ) : ℝ :=
if n % 2 = 1 then log 3 (seq_a n) else 2 / seq_a n

-- Define the summation
def sum_seq_b (n : ℕ) : ℝ :=
  (finset.range (2 * n + 1)).sum (λ i, seq_b (i + 1)) 

-- Statement for the second proof
theorem sum_seq_b_eq (n : ℕ) : sum_seq_b n = (n * (n - 1)) / 2 + 1 - 1 / (3^n) := 
by 
  sorry

end seq_a_ratio_sum_seq_b_eq_l73_73886


namespace value_of_expression_l73_73802

theorem value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) : 12 * a - 6 * b + 3 * c - 2 * d = 40 :=
by sorry

end value_of_expression_l73_73802


namespace probability_at_least_one_of_any_two_probability_at_least_one_of_three_l73_73787

open ProbabilityTheory

noncomputable def stock_profitability (pA pB pC : ℝ) : Prop :=
  let A := Event pA
  let B := Event pB
  let C := Event pC
  independent A B ∧ independent A C ∧ independent B C

theorem probability_at_least_one_of_any_two (pA pB pC : ℝ)
  (h_ind: stock_profitability pA pB pC)
  (hA : pA = 0.8)
  (hB : pB = 0.6)
  (hC : pC = 0.5) :
  P(at_least_two_of_three(A, B, C)) = 0.7 := sorry

theorem probability_at_least_one_of_three (pA pB pC : ℝ)
  (h_ind: stock_profitability pA pB pC)
  (hA : pA = 0.8)
  (hB : pB = 0.6)
  (hC : pC = 0.5) :
  P(at_least_one_of_three(A, B, C)) = 0.96 := sorry

def at_least_two_of_three (A B C : Event ℝ) : Event ℝ :=
  A ∧ B ∨ A ∧ C ∨ B ∧ C ∨ A ∧ B ∧ C

def at_least_one_of_three (A B C : Event ℝ) : Event ℝ :=
  A ∨ B ∨ C

end probability_at_least_one_of_any_two_probability_at_least_one_of_three_l73_73787


namespace correct_propositions_l73_73278

-- Definitions for lines and planes
variables (Line Plane : Type)
variables (m n : Line) (α β γ : Plane)

-- Definitions for perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Propositions
def Prop1 (m n : Line) (α : Plane) : Prop :=
  perpendicular m α ∧ perpendicular n α → parallel m n

def Prop2 (m n : Line) (α β γ : Plane) : Prop :=
  intersection α γ = m ∧ intersection β γ = n ∧ parallel m n → parallel_planes α β

def Prop3 (m : Line) (α β γ : Plane) : Prop :=
  parallel_planes α β ∧ parallel_planes β γ ∧ perpendicular m α → perpendicular m γ

def Prop4 (α β γ : Plane) : Prop :=
  perpendicular γ α ∧ perpendicular γ β → parallel_planes α β

-- The main theorem stating Propositions 1 and 3 are correct, 2 and 4 are not
theorem correct_propositions (h1 : Prop1 m n α) (h3 : Prop3 m α β γ) :
  (Prop1 m n α) ∧ ¬(Prop2 m n α β γ) ∧ (Prop3 m α β γ) ∧ ¬(Prop4 α β γ) :=
by
  exact sorry -- Skip the detailed proof

end correct_propositions_l73_73278


namespace right_triangle_area_l73_73374

theorem right_triangle_area (a b : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (right_triangle : ∃ c : ℝ, c^2 = a^2 + b^2) : 
  ∃ A : ℝ, A = 1/2 * a * b ∧ A = 30 := 
by
  sorry

end right_triangle_area_l73_73374


namespace range_of_m_l73_73819

theorem range_of_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) (h4 : 4 / a + 1 / (b - 1) > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l73_73819


namespace max_value_f_on_interval_min_integer_m_inequality_l73_73920

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m / 2) * x^2 + (m - 1) * x - 1

theorem max_value_f_on_interval (m : ℝ) :
  ∃ max_val : ℝ, 
  (m ≥ (2 / 5) → max_val = 4 * m - 3) ∧ 
  (m < (2 / 5) → max_val = (3 / 2) * m - 2) ∧ 
  ∀ x, 1 ≤ x ∧ x ≤ 2 → f m x ≤ max_val := 
sorry

theorem min_integer_m_inequality :
  ∃ m : ℕ, f m x ≥ Real.log x ∀ x : ℝ, x > 0 := 
sorry

end max_value_f_on_interval_min_integer_m_inequality_l73_73920


namespace privateer_overtakes_at_6_08_pm_l73_73032

noncomputable def time_of_overtake : Bool :=
  let initial_distance := 12 -- miles
  let initial_time := 10 -- 10:00 a.m.
  let privateer_speed_initial := 10 -- mph
  let merchantman_speed := 7 -- mph
  let time_to_sail_initial := 3 -- hours
  let distance_covered_privateer := privateer_speed_initial * time_to_sail_initial
  let distance_covered_merchantman := merchantman_speed * time_to_sail_initial
  let relative_distance_after_three_hours := initial_distance + distance_covered_merchantman - distance_covered_privateer
  let privateer_speed_modified := 13 -- new speed
  let merchantman_speed_modified := 12 -- corresponding merchantman speed

  -- Calculating the new relative speed after the privateer's speed is reduced
  let privateer_new_speed := (13 / 12) * merchantman_speed
  let relative_speed_after_damage := privateer_new_speed - merchantman_speed
  let time_to_overtake_remainder := relative_distance_after_three_hours / relative_speed_after_damage
  let total_time := time_to_sail_initial + time_to_overtake_remainder -- in hours

  let final_time := initial_time + total_time -- converting into the final time of the day
  final_time == 18.1333 -- This should convert to 6:08 p.m., approximately 18 hours and 8 minutes in a 24-hour format

theorem privateer_overtakes_at_6_08_pm : time_of_overtake = true :=
  by
    -- Proof will be provided here
    sorry

end privateer_overtakes_at_6_08_pm_l73_73032


namespace bode_law_planet_9_l73_73684

theorem bode_law_planet_9 :
  ∃ (a b : ℝ),
    (a + b = 0.7) ∧ (a + 2 * b = 1) ∧ 
    (70 < a + b * 2^8) ∧ (a + b * 2^8 < 80) :=
by
  -- Define variables and equations based on given conditions
  let a : ℝ := 0.4
  let b : ℝ := 0.3
  
  have h1 : a + b = 0.7 := by 
    sorry  -- Proof that a + b = 0.7
  
  have h2 : a + 2 * b = 1 := by
    sorry  -- Proof that a + 2 * b = 1
  
  have hnine : 70 < a + b * 2^8 ∧ a + b * 2^8 < 80 := by
    -- Calculate a + b * 2^8 and then check the range
    sorry  -- Proof that 70 < a + b * 2^8 < 80

  exact ⟨a, b, h1, h2, hnine⟩

end bode_law_planet_9_l73_73684


namespace determine_missing_digit_l73_73761

variable (n : ℕ) (num1 num2 : ℕ)
variable (multiplier : ℕ) (remaining : ℕ)
variable (digitSum : ℕ) (nearestMultipleOf9 : ℕ)
variable (missingDigit : ℕ)
variable (digitsNum1 digitsNum2 : List ℕ)

-- Assumptions that num1 and num2 contain the same digits in a different order
axiom (same_digits : digitsNum1.val'.toNat ≡ digitsNum2.val'.toNat [MOD 9])
axiom (num1 >= num2)

-- num1 and num2 both have the same digits
-- resulting difference when one number with same digits but in different order is taken and it is multiplied by any number
axiom (difference : ((num1 - num2) * multiplier) = remaining + 9 * n)

-- 'remaining' was the number when 'missingDigit' was removed
axiom (missingDigit ≠ 0)
axiom (remainingDigitSum + missingDigit = nearestMultipleOf9)
axiom (nearestMultipleOf9 % 9 = 0)

-- B can determine the missing digit
theorem determine_missing_digit : missingDigit = nearestMultipleOf9 - digitSum := sorry

end determine_missing_digit_l73_73761


namespace alex_movies_watched_l73_73834

theorem alex_movies_watched (d h a t w : ℕ) (h1 : d = 7) (h2 : h = 12) (a2 : a + 2) (t : 30) (w : t - 2 = d + h + a) : a = 9 :=
by
  sorry

end alex_movies_watched_l73_73834


namespace tor_gamma_proof_l73_73772

def is_square_digit (n : ℕ) : Prop :=
  n ∈ {0, 1, 4, 5, 6, 9}

def valid_in_range (T Γ : ℕ) : Prop :=
  T < Γ ∧ is_square_digit T

theorem tor_gamma_proof (TOR ROT : ℕ) (T Γ : ℕ) (h1 : valid_in_range T Γ) :
  TOR = 1089 ∧ Γ = 9 :=
by
  sorry

end tor_gamma_proof_l73_73772


namespace smallest_positive_integer_l73_73379

theorem smallest_positive_integer (n : ℕ) : 
  (∃ m : ℕ, (4410 * n = m^2)) → n = 10 := 
by
  sorry

end smallest_positive_integer_l73_73379


namespace value_of_x_l73_73210

theorem value_of_x (x : ℝ) (h : x = 88 + 0.25 * 88) : x = 110 :=
sorry

end value_of_x_l73_73210


namespace polynomial_has_zero_l73_73423

noncomputable def possible_zero_of_polynomial (r s : ℤ) (t : ℝ) : Prop :=
  ∃ (α β : ℤ), t = 3 / 2 ∧
  (x : ℂ) → (x - r) * (x - s) * (x - t) * (x^2 + α * x + β) = 0 →
  ∃ (z : ℂ), z = (3 + complex.I * real.sqrt 11) / 2

theorem polynomial_has_zero :
  ∀ r s t : ℤ, ∃ (α β : ℤ), t = 3 / 2 ∧ r ≠ s ∧ t ≠ r ∧ t ≠ s →
  possible_zero_of_polynomial r s t :=
by sorry

end polynomial_has_zero_l73_73423


namespace sum_of_first_n_terms_minimized_l73_73976

open Nat

theorem sum_of_first_n_terms_minimized (a : ℕ → ℤ) (d : ℤ) (h1 : d > 0) 
  (h2 : |a 6| = |a 11|) : 
  ∃ n, n = 8 ∧ (∀ m < 8, ∑ k in range m, a k < 0) ∧ (∀ m ≥ 9, ∑ k in range m, a k > 0) :=
by
  sorry

end sum_of_first_n_terms_minimized_l73_73976


namespace find_divisor_l73_73746

theorem find_divisor (dividend quotient remainder : ℕ) (h₁ : dividend = 176) (h₂ : quotient = 9) (h₃ : remainder = 5) : 
  ∃ divisor, dividend = (divisor * quotient) + remainder ∧ divisor = 19 := by
sorry

end find_divisor_l73_73746


namespace sum_of_basic_terms_div_by_4_l73_73607

theorem sum_of_basic_terms_div_by_4 (n : ℕ) (hn : n ≥ 4) (a : ℕ → ℕ → ℤ)
  (hcell : ∀ i j, a i j = 1 ∨ a i j = -1) :
  (∑ σ : Equiv.Perm (Fin n), ∏ i : Fin n, a i (σ i)) % 4 = 0 :=
  sorry

end sum_of_basic_terms_div_by_4_l73_73607


namespace ratio_of_distances_l73_73080

def distance_to_first_friend : ℝ := 8
def distance_to_second_friend : ℝ := distance_to_first_friend / 2
def distance_to_work : ℝ := 36
def total_distance_to_friends : ℝ := distance_to_first_friend + distance_to_second_friend

theorem ratio_of_distances :
  (distance_to_work / total_distance_to_friends) = 3 := by
  sorry

end ratio_of_distances_l73_73080


namespace find_a_plus_c_l73_73113

theorem find_a_plus_c (a b c d : ℝ) (h1 : ab + bc + cd + da = 40) (h2 : b + d = 8) : a + c = 5 :=
by
  sorry

end find_a_plus_c_l73_73113


namespace find_t_of_ortho_condition_l73_73167

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (c : V) (t : ℝ)

-- Conditions
def is_unit_vector (v : V) := ⟪v, v⟫ = 1

def angle_between_unit_vectors_is_60 (a b : V) : Prop :=
  ⟪a, b⟫ = 1/2

def vector_c (a b : V) (t : ℝ) : V :=
  t • a + (1 - t) • b

-- Problem statement
theorem find_t_of_ortho_condition 
  (huva : is_unit_vector a)
  (huvb : is_unit_vector b)
  (ha60 : angle_between_unit_vectors_is_60 a b)
  (hbc : ⟪b, vector_c a b t⟫ = 0) : t = 2 :=
by sorry

end find_t_of_ortho_condition_l73_73167


namespace amount_left_is_correct_l73_73800

variable (income : ℝ) (childrenPercent : ℝ) (wifePercent : ℝ) (orphanPercent : ℝ)
variable (givenToChildren : ℝ) (givenToWife : ℝ) (remainingAfterFamily : ℝ) (givenToOrphan : ℝ) (amountLeft : ℝ)

def initial_income := income = 1000
def percent_given_to_children_each := childrenPercent = 0.1
def percent_given_to_wife := wifePercent = 0.2
def percent_donated_to_orphan := orphanPercent = 0.1

def total_given_to_children := givenToChildren = 2 * (childrenPercent * income)
def amount_given_to_wife := givenToWife = wifePercent * income
def amount_remaining_after_family := remainingAfterFamily = income - (givenToChildren + givenToWife)
def amount_given_to_orphan := givenToOrphan = orphanPercent * remainingAfterFamily
def final_amount_left := amountLeft = remainingAfterFamily - givenToOrphan

theorem amount_left_is_correct :
  initial_income ∧
  percent_given_to_children_each ∧
  percent_given_to_wife ∧
  percent_donated_to_orphan ∧
  total_given_to_children ∧
  amount_given_to_wife ∧
  amount_remaining_after_family ∧
  amount_given_to_orphan ∧
  final_amount_left →
  amountLeft = 540 :=
by sorry

end amount_left_is_correct_l73_73800


namespace binomial_identity_l73_73866

noncomputable def binomial (a : ℝ) (k : ℕ) : ℝ :=
  (List.prod (List.range k).map (λ n, a - ↑n)) / ↑(Nat.factorial k)

theorem binomial_identity (h1 : binomial (-3/2) 50) (h2 : binomial (1/2) 50) :
  h1 / h2 = -101 := by
  sorry

end binomial_identity_l73_73866


namespace falling_factorial_sum_l73_73551

-- Given conditions
def fallingFactorial (x : ℝ) : ℕ → ℝ
| 0       => 1
| (n + 1) => x * fallingFactorial (x + 1) n

-- Binomial coefficient
def C (n k : ℕ) : ℕ :=
Nat.choose n k

-- Question statement transformed to a Lean theorem
theorem falling_factorial_sum (x y : ℝ) (n : ℕ) :
  fallingFactorial (x + y) n = ∑ k in Finset.range (n + 1), C n k * fallingFactorial x k * fallingFactorial y (n - k) :=
sorry  -- proof to be filled in

end falling_factorial_sum_l73_73551


namespace pure_imaginary_solutions_l73_73843

theorem pure_imaginary_solutions (k : ℝ) (x = k * complex.I) :
  x ^ 4 - 5 * x ^ 3 + 10 * x ^ 2 - 50 * x - 75 = 0 ↔  x = complex.I * real.sqrt 10 ∨ x = -complex.I * real.sqrt 10 :=
by {
  sorry -- The proof would go here.
}

end pure_imaginary_solutions_l73_73843


namespace part_a_part_b_part_c_l73_73214

-- Part (a)
theorem part_a (DE BC EC : ℝ) (hDE : DE = 6) (hBC : BC = 10) (hEC : EC = 3) (AE AC x : ℝ)
  (hSim : DE / BC = AE / AC) (hAC : AC = AE + EC) : AE = 9 / 2 := 
sorry

-- Part (b)
theorem part_b (WX ZY WZ XY MN : ℝ) (hWXZY : WX / ZY = 3 / 4) 
  (WM MZ XN NY : ℝ) (hWM : WM / MZ = 2 / 3) (hXN : XN / NY = 2 / 3) 
  (PW PZ PM : ℝ) (hSim1 : PW / PZ = WX / ZY) (hSim2 : PW / PM = WX / MN) : WX / MN = 15 / 17 := 
sorry

-- Part (c)
theorem part_c (WX ZY MN : ℤ) (hWXZY : WX.toRat / ZY.toRat = 3 / 4) 
  (hSum : WX + MN + ZY = 2541) (t : ℕ) 
  (hRatio : ∃ t : ℕ, MZ.toRat / WM.toRat = (t : ℝ) ∧ NY.toRat / XN.toRat = (t : ℝ)) : 
  MN ∈ [763, 770, 777 & 847] := 
sorry

end part_a_part_b_part_c_l73_73214


namespace number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73230

theorem number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares :
  ∃ (n : ℕ), n = 3 ∧ ∀ a b c : ℕ, a^2 + b^2 + c^2 = 100 → 
    a*b*c ≠ 0 → a^2 ≤ b^2 ≤ c^2 :=
by sorry

end number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73230


namespace least_number_of_students_with_glasses_and_pet_l73_73600

theorem least_number_of_students_with_glasses_and_pet :
  ∀ (total students_with_glasses students_with_pets : ℕ),
  total = 35 →
  students_with_glasses = 18 →
  students_with_pets = 25 →
  ∃ (both : ℕ), both = 8 ∧ students_with_glasses + students_with_pets - total = both :=
by
  intros total students_with_glasses students_with_pets h1 h2 h3
  use (students_with_glasses + students_with_pets - total)
  split
  · exact Eq.refl (students_with_glasses + students_with_pets - total)
  · exact Eq.trans (by rw [h1, h2, h3]; rfl) rfl

end least_number_of_students_with_glasses_and_pet_l73_73600


namespace graph_symmetric_about_x_one_l73_73288

-- Given the assumptions about the function f and the even property of |f(x)|
variables (f : ℝ → ℝ)

-- |f(x)| is an even function
def is_even_function : Prop :=
  ∀ x, |f(x)| = |f(-x)|

-- We need to state that the graph of |f(x-1)| is symmetric about the line x=1
theorem graph_symmetric_about_x_one (h1 : is_even_function f) :
  ∀ x, |f(x - 1)| = |f(2 - x - 1)| :=
by
   sorry

end graph_symmetric_about_x_one_l73_73288


namespace find_r_l73_73470

theorem find_r (b r : ℝ) (h1 : b / (1 - r) = 18) (h2 : b * r^2 / (1 - r^2) = 6) : r = 1/2 :=
by
  sorry

end find_r_l73_73470


namespace sequence_properties_l73_73586

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 3 - 2^n

-- Prove the statements
theorem sequence_properties (n : ℕ) :
  (a (2 * n) = 3 - 4^n) ∧ (a 2 / a 3 = 1 / 5) :=
by
  sorry

end sequence_properties_l73_73586


namespace horse_problem_l73_73249

theorem horse_problem (x : ℕ) :
  150 * (x + 12) = 240 * x :=
sorry

end horse_problem_l73_73249


namespace relationship_AQ_AP_l73_73616

variable (O A B C D P Q : Point)
variable (circle : Circle)
variable (O_center : O = circle.center)
variable (AB_diameter : is_diameter A B circle)
variable (CD_diameter : is_diameter C D circle)
variable (AB_CD_perpendicular : ∠ (A B) (C D) = 90)
variable (AQ_chord : is_chord A Q circle)
variable (P_on_AB : P ∈ segment A B)
variable (Q_on_circumference : Q ∈ circle)

theorem relationship_AQ_AP (A_meets_B_CD: A = line_meet P Q ∧ Q ∈ circle ) :
  (segment_length A Q * segment_length A P) = (segment_length A O) ^ 2 :=
by
  sorry

end relationship_AQ_AP_l73_73616


namespace problem_statement_l73_73140

-- Define the arithmetic sequence conditions
variables (a : ℕ → ℕ) (d : ℕ)
axiom h1 : a 1 = 2
axiom h2 : a 2018 = 2019
axiom arithmetic_seq : ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := (n * a 1) + (n * (n-1) * d / 2)

theorem problem_statement : sum_seq a 5 + a 2014 = 2035 :=
by sorry

end problem_statement_l73_73140


namespace sum_of_money_l73_73061

theorem sum_of_money (J C P : ℕ) 
  (h1 : P = 60)
  (h2 : P = 3 * J)
  (h3 : C + 7 = 2 * J) : 
  J + P + C = 113 := 
by
  sorry

end sum_of_money_l73_73061


namespace triplet_sum_check_l73_73388

noncomputable def sum_triplet (t : ℝ × ℝ × ℝ): ℝ :=
  t.1 + t.2 + t.3

theorem triplet_sum_check
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ)
  (C : ℝ × ℝ × ℝ)
  (D : ℝ × ℝ × ℝ)
  (E : ℝ × ℝ × ℝ) :
  A = (2/5, 2/5, 1/5) → 
  B = (-1, 3, -1) → 
  C = (0.5, 0.2, 0.3) → 
  D = (0.25, -0.45, 0.2) →
  E = (1.2, -0.1, -0.1) →
  sum_triplet A = 1 → 
  sum_triplet B = 1 → 
  sum_triplet C = 1 → 
  sum_triplet E = 1 → 
  sum_triplet D ≠ 1 :=
by
  intro hA hB hC hD hE hA_sum hB_sum hC_sum hE_sum
  rw [hA, hB, hC, hD, hE, sum_triplet] at *
  sorry

end triplet_sum_check_l73_73388


namespace perpendicular_line_through_A_l73_73135

variable (m : ℝ)

-- Conditions
def line1 (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def line2 (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0
def pointA : ℝ × ℝ := (3, 2)

-- Question and proof
theorem perpendicular_line_through_A (h_parallel : ∃ x y, line1 m x y ∧ line2 m x y) :
  ∃ (t : ℝ), ∀ (x y : ℝ), (y = 2 * x + t) ↔ (2 * x - y - 4 = 0) :=
by
  sorry

end perpendicular_line_through_A_l73_73135


namespace train_original_speed_l73_73438

theorem train_original_speed (length_of_train : ℝ) (crossing_time : ℝ) (walking_speed_kmph : ℝ) (incline_percentage : ℝ) : 
  (length_of_train = 1000) ∧ (crossing_time = 15) ∧ (walking_speed_kmph = 10) ∧ (incline_percentage = 0.05) →
  ∃ (original_speed : ℝ), (original_speed = 256) :=
begin
  intros h,
  rcases h with ⟨h_length, h_time, h_speed, h_incline⟩,
  use 256,
  sorry
end

end train_original_speed_l73_73438


namespace remainder_when_m_plus_n_divided_by_2023_is_1_l73_73781

theorem remainder_when_m_plus_n_divided_by_2023_is_1 :
  let p := 2023
  let s := ∑' r : ℕ, ∑' c : ℕ, (1 : ℚ) / (2 * p)^r / p^c
  let frac := s.num / s.denom
  (frac.num + frac.denom) % p = 1 := by
  -- Proof goes here
  sorry

end remainder_when_m_plus_n_divided_by_2023_is_1_l73_73781


namespace complex_translation_l73_73440

theorem complex_translation (z w : ℂ) (h1 : 2 + 3 * complex.i + w = 5 + complex.i) :
  (1 - 2 * complex.i) + w = 4 - 4 * complex.i :=
by 
  sorry

end complex_translation_l73_73440


namespace small_circle_ratio_l73_73250

theorem small_circle_ratio (a b : ℝ) (ha : 0 < a) (hb : a < b) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) :
  a / b = Real.sqrt 6 / 6 :=
by
  sorry

end small_circle_ratio_l73_73250


namespace intersection_A_B_range_of_m_l73_73897

-- Step 1: Define sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

def C (m : ℝ) : Set ℝ := {x | m < x ∧ x < 2 * m - 1}

-- Step 2: Lean statements for the proof

-- (1) Prove A ∩ B = {x | 1 < x < 3}
theorem intersection_A_B : (A ∩ B) = {x | 1 < x ∧ x < 3} :=
by
  sorry

-- (2) Prove the range of m such that C ∪ B = B is (-∞, 2]
theorem range_of_m (m : ℝ) : (C m ∪ B = B) ↔ m ≤ 2 :=
by
  sorry

end intersection_A_B_range_of_m_l73_73897


namespace max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l73_73992

-- Define the traffic flow function
noncomputable def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

-- Condition: v > 0
axiom v_pos (v : ℝ) : v > 0 → traffic_flow v ≥ 0

-- Prove that the average speed v = 40 results in the maximum traffic flow y = 920/83 ≈ 11.08
theorem max_traffic_flow_at_v_40 : traffic_flow 40 = 920 / 83 :=
sorry

-- Prove that to ensure the traffic flow is at least 10 thousand vehicles per hour,
-- the average speed v should be in the range [25, 64]
theorem traffic_flow_at_least_10_thousand (v : ℝ) (h : traffic_flow v ≥ 10) : 25 ≤ v ∧ v ≤ 64 :=
sorry

end max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l73_73992


namespace rationalize_cubic_root_l73_73185

theorem rationalize_cubic_root :
  let u := 1 / (2 - real.cbrt 3)
  in u = (2 + real.cbrt 3) * (4 + real.cbrt 9) / 7 :=
by
  let u := 1 / (2 - real.cbrt 3)
  have h1 : (2 - real.cbrt 3) * (2 + real.cbrt 3) = 4 - real.cbrt 9 := by sorry
  have h2 : (4 - real.cbrt 9) * (4 + real.cbrt 9) = 7 := by sorry
  calc
    u = (2 + real.cbrt 3) / (4 - real.cbrt 9) : by sorry
  ... = (2 + real.cbrt 3) * (4 + real.cbrt 9) / 7 : by sorry

end rationalize_cubic_root_l73_73185


namespace average_speed_for_trip_l73_73000

theorem average_speed_for_trip 
  (Speed1 Speed2 : ℝ) 
  (AverageSpeed : ℝ) 
  (h1 : Speed1 = 110) 
  (h2 : Speed2 = 72) 
  (h3 : AverageSpeed = (2 * Speed1 * Speed2) / (Speed1 + Speed2)) :
  AverageSpeed = 87 := 
by
  -- solution steps would go here
  sorry

end average_speed_for_trip_l73_73000


namespace calculation_proof_l73_73825

theorem calculation_proof
    (a : ℝ) (b : ℝ) (c : ℝ)
    (h1 : a = 3.6)
    (h2 : b = 0.25)
    (h3 : c = 0.5) :
    (a * b) / c = 1.8 := 
by
  sorry

end calculation_proof_l73_73825


namespace exists_chord_with_given_difference_l73_73846

theorem exists_chord_with_given_difference (O P : Point) (r a : ℝ) (h_circle : (dist O P) < r) :
  ∃ (A B : Point), on_circle O r A ∧ on_circle O r B ∧ (|dist A P - dist B P| = a) :=
by
  sorry

end exists_chord_with_given_difference_l73_73846


namespace sum_of_factorials_mod_25_l73_73588

theorem sum_of_factorials_mod_25 :
  (1! + 2! + 3! + 4! + (∑ i in (range 46).map(λ x, x + 5), i!)) % 25 = 8 :=
by sorry

end sum_of_factorials_mod_25_l73_73588


namespace solution_set_of_inequality_l73_73715

theorem solution_set_of_inequality (x : ℝ) (h : 3 * x + 2 > 5) : x > 1 :=
sorry

end solution_set_of_inequality_l73_73715


namespace range_of_a_l73_73952

theorem range_of_a (a : ℝ) :
  (∀ θ : ℝ, complex.abs ((a + real.cos θ) + (2 * a - real.sin θ) * complex.I) ≤ 2) ↔ a ∈ Icc (-1/2 : ℝ) (1/2 : ℝ) :=
sorry

end range_of_a_l73_73952


namespace constant_a_n_general_term_a_n_l73_73002

-- Define the sequence a_n
def a_seq (x y : ℝ) : ℕ → ℝ
| 0     := x
| 1     := y
| (n+2) := (a_seq (n+1) * a_seq n + 1) / (a_seq (n+1) + a_seq n)

-- Define the Fibonacci sequence
def fib : ℕ → ℕ 
| 0     := 1
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- First part: prove existence of n_0 such that a_n is constant for all n ≥ n_0
theorem constant_a_n {x y : ℝ} (h1 : |x| = 1) (h2 : y ≠ -x) : 
  ∃ n_0 : ℕ, ∀ n ≥ n_0, (a_seq x y n) = (a_seq x y n_0) := 
sorry

-- Second part: prove the general formula for a_n
theorem general_term_a_n {x y : ℝ} {n : ℕ} :
  a_seq x y n = 
  (1 + ((y - 1) / (y + 1))^((fib (n - 1))) * ((x - 1) / (x + 1))^((fib (n - 2)))) /  
  (1 - ((y - 1) / (y + 1))^((fib (n - 1))) * ((x - 1) / (x + 1))^((fib (n - 2)))) := 
sorry

end constant_a_n_general_term_a_n_l73_73002


namespace temp_conversion_l73_73742

theorem temp_conversion (T_fahrenheit : ℝ) (T_celsius : ℝ) 
    (h : T_fahrenheit = 113) 
    (conversion_formula : T_celsius = (T_fahrenheit - 32) * 5 / 9) : 
    T_celsius = 45 := 
by
  rw [h, conversion_formula]
  sorry

end temp_conversion_l73_73742


namespace squares_shared_vertex_collinear_median_altitude_l73_73739

theorem squares_shared_vertex_collinear_median_altitude
  (A B C D K M N : Point)
  (h_square1 : square B C D A)
  (h_square2 : square B K M N)
  (h_common_vertex : B = B)
  (h_clockwise1 : vertices_clockwise [B, C, D, A])
  (h_clockwise2 : vertices_clockwise [B, K, M, N]) :
  collinear [B, E, F] :=
sorry

end squares_shared_vertex_collinear_median_altitude_l73_73739


namespace no_real_solution_l73_73650

def f (x : ℝ) : ℝ := 2 * x + 3

-- Defining the inverse of f
def f_inv (y : ℝ) : ℝ := (y - 3) / 2

-- Stating the main theorem
theorem no_real_solution (x : ℝ) : f_inv x ≠ f (x * x) := by sorry

end no_real_solution_l73_73650


namespace problem_statement_l73_73544

noncomputable def binomial_expansion_term (n k : ℕ) : ℝ :=
  (nat.choose n k) * (1 / 2)^(n - k) * (-1)^k * x^(2 * n - 5 * k / 2)

theorem problem_statement :
  (∀ k : ℕ, 2 * 10 - (5 * k / 2) = 0 → k = 8) →
  binomial_expansion_term 10 6 = 105 / 8 →
  (∃ k_values : finset ℕ, ∀ k ∈ k_values, k % 2 = 0 ∧ k < 11 ∧
    (2 * 10 - (5 * k / 2)).denom = 1 ∧ k_values.card = 6) :=
by
  intros h1 h2 h3
  sorry

end problem_statement_l73_73544


namespace probability_at_least_one_six_is_11_div_36_l73_73367

noncomputable def probability_at_least_one_six : ℚ :=
  let total_outcomes := 36
  let no_six_outcomes := 25
  let favorable_outcomes := total_outcomes - no_six_outcomes
  favorable_outcomes / total_outcomes
  
theorem probability_at_least_one_six_is_11_div_36 : 
  probability_at_least_one_six = 11 / 36 :=
by
  sorry

end probability_at_least_one_six_is_11_div_36_l73_73367


namespace salt_added_l73_73036

theorem salt_added (x : ℝ) (hx : x = 104.99999999999997) : 
(let original_salt := 0.20 * 105 in
 let remaining_volume := 105 - (1 / 4) * 105 in
 let new_volume := remaining_volume + 7 in
 let required_salt := (1 / 3) * (new_volume + 11.375) in
 21 + 11.375 = required_salt) :=
by 
  have h1 : x = 104.99999999999997 := hx,
  have original_salt := 0.20 * 105,
  have remaining_volume := 105 - (1 / 4) * 105,
  have new_volume := remaining_volume + 7,
  have required_salt := (1 / 3) * (new_volume + 11.375),
  show 21 + 11.375 = required_salt from sorry

end salt_added_l73_73036


namespace proof_problem_l73_73148

noncomputable def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ) - Real.sqrt 2

def min_distance_condition (ω : ℝ) : Prop :=
  ω > 0 ∧ (π / (2 * ω)) = π / 4

def sum_abscissas_condition (ω φ : ℝ) : Prop :=
  φ > 0 ∧ φ < π ∧ ((π / (4 * ω)) - φ / ω + (3 * π / (4 * ω)) - φ / ω = π / 3)

def option_A (ω φ : ℝ) : Prop :=
  f ω φ (π / 3) = 1 - Real.sqrt 2

def option_B (ω φ : ℝ) : Prop :=
  ∀ x ∈ Icc (-π / 6) (π / 6), 
    -π / 6 < 2 * x + φ < π / 2 → 
      Real.sin.deriv (2 * x + φ) > 0

theorem proof_problem : 
  ∃ ω φ, min_distance_condition ω ∧ sum_abscissas_condition ω φ ∧ option_A ω φ ∧ option_B ω φ :=
sorry

end proof_problem_l73_73148


namespace total_students_l73_73029

variable {α : Type*}
variables (A B C : Set α)

theorem total_students 
  (hA : A.card = 32) 
  (hB : B.card = 33) 
  (hC : C.card = 14) 
  (hAB : (A ∩ B).card = 13) 
  (hBC : (B ∩ C).card = 10) 
  (hCA : (C ∩ A).card = 7) 
  (hABC : (A ∩ B ∩ C).card = 3) : 
  (A ∪ B ∪ C).card = 52 := 
by
  sorry

end total_students_l73_73029


namespace Mr_Kishore_savings_l73_73447

theorem Mr_Kishore_savings :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let miscellaneous := 5200
  let utilities := 3500
  let entertainment := 1800
  let insurance := 4000
  let total_expenses := rent + milk + groceries + education + petrol + miscellaneous + utilities + entertainment + insurance
  let total_expenses_eq : total_expenses = 28000 := by sorry
  let savings_rate := 0.15
  let salary := 28000 / 0.85
  let savings := savings_rate * salary
  savings = 4941.18 :=
begin
  sorry
end

end Mr_Kishore_savings_l73_73447


namespace university_cost_per_box_l73_73382

def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def num_boxes (total_volume box_volume : ℕ) : ℕ :=
  total_volume / box_volume

def cost_per_box (total_cost num_boxes : ℚ) : ℚ :=
  total_cost / num_boxes

theorem university_cost_per_box :
  let length := 20
  let width := 20
  let height := 15
  let total_volume := 3060000
  let total_cost := 459
  let box_vol := box_volume length width height
  let boxes := num_boxes total_volume box_vol
  cost_per_box total_cost boxes = 0.90 :=
by
  sorry

end university_cost_per_box_l73_73382


namespace find_D_double_prime_l73_73300

def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translateUp1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)

def reflectYeqX (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translateDown1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

def D'' (D : ℝ × ℝ) : ℝ × ℝ :=
  translateDown1 (reflectYeqX (translateUp1 (reflectY D)))

theorem find_D_double_prime :
  let D := (5, 0)
  D'' D = (-1, 4) :=
by
  sorry

end find_D_double_prime_l73_73300


namespace angie_carlos_opposite_probability_l73_73813

theorem angie_carlos_opposite_probability :
  let people := Finset.ofList ["Angie", "Bridget", "Carlos", "Diego"] in
  let arrangements := people.toSet.powerset.filter (λ arrangement, arrangement.card = 1) in
  let possible_seatings := arrangements.filter (λ arrangement, 
    arrangement.toList.head = "Angie" ∧
    (arrangement.toList.tail.head = "Bridget" ∨ arrangement.toList.tail.last = "Bridget")) in
  let favorable_outcomes := possible_seatings.filter (λ arrangement, 
    arrangement.toList.head = "Angie" ∧
    arrangement.toList.tail.head = "Bridget" ∧
    arrangement.toList.last = "Carlos") in
  
  (favorable_outcomes.card : ℚ) / (possible_seatings.card : ℚ) = 1/4 :=
by
  sorry

end angie_carlos_opposite_probability_l73_73813


namespace class_books_l73_73299

theorem class_books (init_borrowed: ℕ) (borrow1: ℕ) (returned: ℕ) (borrow2: ℕ) :
  init_borrowed = 54 → borrow1 = 23 → returned = 12 → borrow2 = 15 → 
  (init_borrowed + borrow1 - returned + borrow2 = 80) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact sorry

end class_books_l73_73299


namespace matrix_multiplication_comm_l73_73649

theorem matrix_multiplication_comm {C D : Matrix (Fin 2) (Fin 2) ℝ}
    (h₁ : C + D = C * D)
    (h₂ : C * D = !![5, 1; -2, 4]) :
    (D * C = !![5, 1; -2, 4]) :=
by
  sorry

end matrix_multiplication_comm_l73_73649


namespace shortest_path_distance_l73_73314

noncomputable def shortest_distance : ℝ :=
  let P := (5 : ℝ, 1 : ℝ)
  let L1 := λ x : ℝ, x
  let L2 := λ y : ℝ, 7
  have hdist : ℝ := 4 * real.sqrt 5
  hdist

theorem shortest_path_distance
  (P : ℝ × ℝ)
  (L1 L2 : ℝ → ℝ)
  (hP : P = (5, 1))
  (hL1 : ∀ x, L1 x = x)
  (hL2 : ∀ y, L2 y = 7) :
  ∃ d : ℝ, shortest_distance = 4 * real.sqrt 5 :=
by
  use 4 * real.sqrt 5
  sorry

end shortest_path_distance_l73_73314


namespace complement_of_set_A_is_34_l73_73164

open Set

noncomputable def U : Set ℕ := {n : ℕ | True}

noncomputable def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Complement of A in U
noncomputable def C_U_A : Set ℕ := U \ A

theorem complement_of_set_A_is_34 : C_U_A = {3, 4} :=
by sorry

end complement_of_set_A_is_34_l73_73164


namespace sum_of_squares_of_midpoints_segments_correct_l73_73695

noncomputable def sum_of_squares_of_midpoints_segments 
  (A B C D E F G H : ℝ) 
  (AC BD : ℝ) 
  (angle_AC_BD : ℝ) 
  (h1 : AC = 3) 
  (h2 : BD = 4) 
  (h3 : angle_AC_BD = 75) : ℝ :=
  (AC^2 + BD^2) / 2

theorem sum_of_squares_of_midpoints_segments_correct :
  ∀ (A B C D E F G H : ℝ) (AC BD : ℝ) (angle_AC_BD : ℝ),
  AC = 3 → BD = 4 → angle_AC_BD = 75 → 
  sum_of_squares_of_midpoints_segments A B C D E F G H AC BD angle_AC_BD 3 4 75 = 12.5 :=
by
  intros A B C D E F G H AC BD angle_AC_BD h1 h2 h3
  unfold sum_of_squares_of_midpoints_segments
  rw [h1, h2, h3]
  norm_num
  sorry

end sum_of_squares_of_midpoints_segments_correct_l73_73695


namespace pyramid_ABCD_properties_l73_73982

noncomputable def length_BD : ℚ := 5 / 2
noncomputable def E_midpoint_AB : Prop := ∀ (A B E : Point), midpoint E A B
noncomputable def F_centroid_BCD : Prop := ∀ (B C D F : Point), centroid F B C D
noncomputable def EF_length : ℚ := 8
noncomputable def sphere_touches_planes_ABD_BCD : Prop := ∀ (radius : ℚ) (E F : Point) (ABD BCD : Plane), 
  radius = 5 ∧ 
  touches_sphere E ABD ∧ 
  touches_sphere F BCD 

theorem pyramid_ABCD_properties :
  ∀ (A B C D E F : Point) (ABD BCD : Plane),
    length_BD = 5 / 2 →
    E_midpoint_AB A B E →
    F_centroid_BCD B C D F →
    EF_length = 8 →
    sphere_touches_planes_ABD_BCD 5 E F ABD BCD →
    dihedral_angle ABD BCD = arccos (7 / 25) ∧
    area B C D = 25 ∧
    volume A B C D = 320 / 3 :=
by
  sorry

end pyramid_ABCD_properties_l73_73982


namespace at_least_one_six_in_two_dice_l73_73369

def total_outcomes (dice : ℕ) (sides : ℕ) : ℕ := sides ^ dice
def non_six_outcomes (dice : ℕ) (sides : ℕ) : ℕ := (sides - 1) ^ dice
def at_least_one_six_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  let all := total_outcomes dice sides
  let none := non_six_outcomes dice sides
  (all - none) / all

theorem at_least_one_six_in_two_dice :
  at_least_one_six_probability 2 6 = 11 / 36 :=
by
  sorry

end at_least_one_six_in_two_dice_l73_73369


namespace sum_of_money_l73_73063

noncomputable def Patricia : ℕ := 60
noncomputable def Jethro : ℕ := Patricia / 3
noncomputable def Carmen : ℕ := 2 * Jethro - 7

theorem sum_of_money : Patricia + Jethro + Carmen = 113 := by
  sorry

end sum_of_money_l73_73063


namespace maria_needs_more_cartons_l73_73665

theorem maria_needs_more_cartons
  (total_needed : ℕ)
  (strawberries : ℕ)
  (blueberries : ℕ)
  (already_has : ℕ)
  (more_needed : ℕ)
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : blueberries = 8)
  (h4 : already_has = strawberries + blueberries)
  (h5 : more_needed = total_needed - already_has) :
  more_needed = 9 :=
by sorry

end maria_needs_more_cartons_l73_73665


namespace tangent_eq_tangent_intersect_other_l73_73915

noncomputable def curve (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

/-- Equation of the tangent line to curve C at x = 1 is y = -12x + 8 --/
theorem tangent_eq (tangent_line : ℝ → ℝ) (x : ℝ):
  tangent_line x = -12 * x + 8 :=
by
  sorry

/-- Apart from the tangent point (1, -4), the tangent line intersects the curve C at the points
    (-2, 32) and (2 / 3, 0) --/
theorem tangent_intersect_other (tangent_line : ℝ → ℝ) x:
  curve x = tangent_line x →
  (x = -2 ∧ curve (-2) = 32) ∨ (x = 2 / 3 ∧ curve (2 / 3) = 0) :=
by
  sorry

end tangent_eq_tangent_intersect_other_l73_73915


namespace smallest_value_zero_l73_73279

open Complex

noncomputable def smallest_possible_value (z : ℂ) (h : |z^2 + 1| = |z * (z + I)|) : ℝ :=
|z + 1|

theorem smallest_value_zero (z : ℂ) (h : |z^2 + 1| = |z * (z + I)|) : smallest_possible_value z h = 0 :=
sorry

end smallest_value_zero_l73_73279


namespace find_z_value_l73_73962

variables {BD FC GC FE : Prop}
variables {a b c d e f g z : ℝ}

-- Assume all given conditions
axiom BD_is_straight : BD
axiom FC_is_straight : FC
axiom GC_is_straight : GC
axiom FE_is_straight : FE
axiom sum_is_z : z = a + b + c + d + e + f + g

-- Goal to prove
theorem find_z_value : z = 540 :=
by
  sorry

end find_z_value_l73_73962


namespace possible_value_of_sum_l73_73710

theorem possible_value_of_sum (p q r : ℝ) (h₀ : q = p * (4 - p)) (h₁ : r = q * (4 - q)) (h₂ : p = r * (4 - r)) 
  (h₃ : p ≠ q ∧ p ≠ r ∧ q ≠ r) : p + q + r = 6 :=
sorry

end possible_value_of_sum_l73_73710


namespace option_B_option_C_option_D_l73_73927

variable {R : Type*} [LinearOrderedField R]

structure parabola (p : R) :=
  (focus : R)
  (vertex : R)
  (is_parabola : ∀ {x y}, y^2 = 2*p*x → p > 0)

structure point (R : Type*) :=
  (x : R)
  (y : R)

structure line (R : Type*) :=
  (slope : R)
  (y_intercept : R)

variables {p : R} (C : parabola p)

def A : point R := sorry -- placeholder for coordinates (x1, y1)
def B : point R := sorry -- placeholder for coordinates (x2, y2)
def M : point R := {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}
def AF : point R := {x := p / 2 - A.x, y := -A.y}
def BF : point R := {x := B.x - p / 2, y := B.y}
def AB : R := ((B.x - A.x)^2 + (B.y - A.y)^2)^0.5
def MN : R := (|AF| + |BF|) / 2

-- Prove the following statements:
theorem option_B (F : point R)
  (h1 : A.x * B.x + A.y * B.y = -12)
  (h2 : (A.y)^2 = 2 * p * (A.x))
  (h3 : (B.y)^2 = 2 * p * (B.x)) : p = 4 := 
sorry

theorem option_C (h4 : AF.x = 3 * BF.x)
  (h5 : AF.y = 3 * BF.y)
  (h6 : (A.y)^2 = 2 * p * (A.x))
  (h7 : (B.y)^2 = 2 * p * (B.x)) : 
  (A.y - B.y) / (A.x - B.x) = sqrt 3 := 
sorry

theorem option_D (F : point R)
  (h8 : -- Circle M with AB passes through F 
  ((point.dist F A) * (point.dist F B) = (AF.x)^2 + (AF.y)^2))
  (h9 : (A.y) ^ 2 = 2 * p * (A.x))
  (h10 : (B.y) ^ 2 = 2 * p * (B.x)) : 
  |AB| / |MN| >= sqrt 2 :=
sorry

end option_B_option_C_option_D_l73_73927


namespace passenger_meets_three_freight_trains_l73_73727

-- Definitions of the stations and distances
def num_stations : Nat := 11
def distance_between_stations : Real := 7
def freight_train_speed : Real := 60 -- km/h
def passenger_train_speed : Real := 100 -- km/h
def freight_train_departure_interval : Real := 5 / 60 -- hours

-- Time at which different trains start
def freight_train_start_time : Real := 7 -- AM
def passenger_train_start_time : Real := 8 -- AM

-- Initial distance between the passenger train and the freight train when the passenger train starts
def initial_distance_first_freight_train : Real := 
  (passenger_train_start_time - freight_train_start_time) * freight_train_speed

-- Total distance of the railway line
def total_distance : Real := 10 * distance_between_stations

-- The theorem which states that the passenger train meets 3 freight trains between specified stations
theorem passenger_meets_three_freight_trains :
  ∃ (n : ℕ), 1 ≤ n ∧ n ≤ num_stations - 1 ∧
  (passenger_meets_between_stations n (n+1) 3) :=
sorry

-- Definition of meeting freight trains between two stations
def passenger_meets_between_stations (station1 station2 : ℕ) (num_trains : ℕ) : Prop :=
  ∃ (distance_covered : Real), 
  distance_covered = (passenger_train_speed * (initial_distance_first_freight_train / (passenger_train_speed + freight_train_speed))) ∧ 
  distance_covered >= distance_between_stations * (station1-1) ∧
  distance_covered < distance_between_stations * station2 ∧
  (num_trains = 3)

end passenger_meets_three_freight_trains_l73_73727


namespace num_positive_perfect_square_sets_l73_73238

-- Define what it means for three numbers to form a set that sum to 100 
def is_positive_perfect_square_set (a b c : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ c ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 100

-- Define the main theorem to state there are exactly 4 such sets
theorem num_positive_perfect_square_sets : 
  {s : Finset (ℕ × ℕ × ℕ) // (∃ a b c, (a, b, c) ∈ s ∧ is_positive_perfect_square_set a b c) }.card = 4 :=
sorry

end num_positive_perfect_square_sets_l73_73238


namespace meeting_point_l73_73666

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def North (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
(p.1, p.2 + d)

theorem meeting_point :
  let mark := (3, 10)
  let sandy := (5, 4)
  let midpoint := midpoint mark sandy
  let meeting_point := North midpoint 2
  meeting_point = (4, 9) := by
  sorry

end meeting_point_l73_73666


namespace volume_ratio_sum_is_26_l73_73127

noncomputable def volume_of_dodecahedron (s : ℝ) : ℝ :=
  (15 + 7 * Real.sqrt 5) * s ^ 3 / 4

noncomputable def volume_of_cube (s : ℝ) : ℝ :=
  s ^ 3

noncomputable def volume_ratio_sum (s : ℝ) : ℝ :=
  let ratio := (volume_of_dodecahedron s) / (volume_of_cube s)
  let numerator := 15 + 7 * Real.sqrt 5
  let denominator := 4
  numerator + denominator

theorem volume_ratio_sum_is_26 (s : ℝ) : volume_ratio_sum s = 26 := by
  sorry

end volume_ratio_sum_is_26_l73_73127


namespace regular_rate_is_16_l73_73408

noncomputable def regular_rate : ℝ :=
  let hours_worked := 65
  let total_compensation := 1340
  let regular_hours := 40
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate_multiplier := 1.75
  let total_pay := total_compensation
  let equation := (regular_hours * R) + (overtime_hours * (overtime_rate_multiplier * R)) = total_pay
  R

theorem regular_rate_is_16 : regular_rate = 16 := by
  -- Define the variables
  let hours_worked := 65
  let total_compensation := 1340
  let regular_hours := 40
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate_multiplier := 1.75
  let total_pay := total_compensation

  -- The equation representing the payment structure
  have eq1 : (regular_hours * R) + (overtime_hours * (overtime_rate_multiplier * R)) = total_pay := sorry

  -- Simplify eq1 to find the value of R
  have eq2 : 83.75 * R = 1340 := sorry

  -- Solve for R from eq2
  have R : ℝ := 1340 / 83.75 := sorry

  -- Assert the proof statement
  exact eq.symm (show 16 = R from sorry)

end regular_rate_is_16_l73_73408


namespace unique_real_solution_l73_73095

theorem unique_real_solution (a : ℝ) : 
  (∀ x : ℝ, (x^3 - a * x^2 - (a + 1) * x + (a^2 - 2) = 0)) ↔ (a < 7 / 4) := 
sorry

end unique_real_solution_l73_73095


namespace sum_double_series_eq_four_thirds_l73_73070

theorem sum_double_series_eq_four_thirds :
  (∑' j : ℕ, ∑' k : ℕ, 2^(- (4 * k + 2 * j + (k + j)^2))) = 4 / 3 :=
begin
  sorry
end

end sum_double_series_eq_four_thirds_l73_73070


namespace problem_statement_l73_73645

theorem problem_statement (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : k % 2 = 1) : ¬ n ∣ 2^(n-1) + 1 := by
  sorry

end problem_statement_l73_73645


namespace sin_sum_equals_sin_sum_conditions_l73_73111

theorem sin_sum_equals_sin_sum_conditions (α β : ℝ) :
  (sin α + sin β = sin (α + β)) →
  ((∃ n : ℤ, α + β = 2 * real.pi * n) ∨
   (∃ n : ℤ, α = 2 * real.pi * n ∧ ∃ m : ℝ, β = m) ∨
   (∃ n : ℤ, β = 2 * real.pi * n ∧ ∃ m : ℝ, α = m)) :=
begin
  sorry
end

end sin_sum_equals_sin_sum_conditions_l73_73111


namespace simplify_exponential_expr_l73_73512

theorem simplify_exponential_expr
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 * x ^ (√2) * (2 * x ^ (-√2) * y * z) = 6 * y * z :=
by 
  simp [mul_assoc, mul_comm, pow_add, real.sqrt_two_mul,
        mul_inv_cancel_of_pos hx, one_mul,
        mul_one] -- These simplifications correspond to the steps in the solution

end simplify_exponential_expr_l73_73512


namespace radian_conversion_245_l73_73477

def convert_degrees_to_radians (d : ℝ) : ℝ :=
  d * (Real.pi / 180)

theorem radian_conversion_245 : convert_degrees_to_radians 245 = (49 / 36) * Real.pi :=
by 
  sorry

end radian_conversion_245_l73_73477


namespace comb_identity_rocket_engine_check_l73_73294

-- Define the combinatorial identity in Lean
theorem comb_identity {n k : ℕ} (h₀ : 0 < k) (h₁ : k ≤ n): 
  k * nat.choose n k = n * nat.choose (n - 1) (k - 1) :=
by sorry

-- Define the probability of completing the check by sampling 32 engines
def probability_check_by_32 : ℝ := 25 / 132

-- Expected value of X
def expected_checks_value : ℝ := 272 / 9

-- The main theorem to encompass the problem and its conditions
theorem rocket_engine_check (h₀ : Fintype 8) :
  ∃ (prob_check: ℝ) (expected_checks: ℝ), 
    prob_check = probability_check_by_32 ∧ 
    (∀ n k, 0 < k ∧ k ≤ n → 
      k * nat.choose n k = n * nat.choose (n - 1) (k - 1)) ∧
    expected_checks = expected_checks_value :=
by sorry

end comb_identity_rocket_engine_check_l73_73294


namespace max_difference_l73_73391

theorem max_difference (U V W X Y Z : ℕ) (hUVW : U ≠ V ∧ V ≠ W ∧ U ≠ W)
    (hXYZ : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) (digits_UVW : 1 ≤ U ∧ U ≤ 9 ∧ 1 ≤ V ∧ V ≤ 9 ∧ 1 ≤ W ∧ W ≤ 9)
    (digits_XYZ : 1 ≤ X ∧ X ≤ 9 ∧ 1 ≤ Y ∧ Y ≤ 9 ∧ 1 ≤ Z ∧ Z ≤ 9) :
    U * 100 + V * 10 + W = 987 → X * 100 + Y * 10 + Z = 123 → (U * 100 + V * 10 + W) - (X * 100 + Y * 10 + Z) = 864 :=
by
  sorry

end max_difference_l73_73391


namespace monotonicity_intervals_g_extremum_f_l73_73926

-- Definitions and conditions
def g (x : ℝ) (a : ℝ) : ℝ := (2 / x) - a * log x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 * g x a

-- The Proof Statements

-- (1) Monotonicity intervals of g(x) when a = -2
theorem monotonicity_intervals_g (x : ℝ) : 
  ∀ x > 0, g x (-2) is_decreasing_on (0, 1) ∧ g x (-2) is_increasing_on (1, ∞) := sorry

-- (2) Range of values for a
theorem extremum_f (a : ℝ) : 
  (∀ x ∈ (1/e : ℝ, e : ℝ), (∃! x₀, (x₀ ∈ (1/e : ℝ, e : ℝ)) ∧ (deriv (f x a)) = 0)) → 
  a < -2 * exp 1 ∨ a > 2 / (3 * exp 1) := sorry

end monotonicity_intervals_g_extremum_f_l73_73926


namespace find_f_prime_2_l73_73151

def f (f'_2 : ℝ) (x : ℝ) : ℝ :=
  (1 / 3) * x^3 - f'_2 * x^2 + x - 3

def f_prime (f'_2 : ℝ) (x : ℝ) : ℝ :=
  x^2 - 2 * f'_2 * x + 1

theorem find_f_prime_2 :
  (∃ f'_2 : ℝ, f_prime f'_2 2 = f'_2) → f_prime 1 2 = 1 :=
by 
  -- The proof will be done here
  sorry

end find_f_prime_2_l73_73151


namespace cute_numbers_l73_73094

def is_prime (p : ℕ) : Prop :=
  1 < p ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def cute (n : ℕ) : Prop :=
  ∀ (m : ℕ), 1 < m ∧ m < n → (m.gcd n = 1 → is_prime m)

theorem cute_numbers :
  ∀ n : ℕ, cute n ↔ n ∈ {1, 2, 3, 4, 6, 8, 12, 18, 24, 30} :=
by
  sorry

end cute_numbers_l73_73094


namespace find_b_l73_73499

variable (a b : ℝ^3)
noncomputable def is_parallel (u v : ℝ^3) : Prop :=
  ∃ k : ℝ, u = k • v

noncomputable def is_orthogonal (u v : ℝ^3) : Prop :=
  u ⬝ v = 0

theorem find_b (h1 : a + b = ![8, -4, 2])
    (h2 : is_parallel a ![2, -1, 1])
    (h3 : is_orthogonal b ![2, -1, 1]) :
    b = ![\(2/3), -(1/3), -(5/3)] :=
  sorry

end find_b_l73_73499


namespace construct_right_triangle_l73_73476

def is_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ H M : Type,
    (is_altitude A H) ∧ (is_median A M) ∧ (is_hypotenuse B C)

def is_altitude {A H : Type} [metric_space A] [metric_space H] (a : A) (h : H) :=
  height a = h

def is_median {A M : Type} [metric_space A] [metric_space M] (a : A) (m : M) :=
  median a = m

theorem construct_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (hA : is_right_triangle A B C) (h1 : ∃ AH, is_altitude A AH) (h2 : ∃ AM, is_median A AM) :
  ∃ (right_triangle : Type), 
    right_triangle = triangle.mk A B C ∧ is_right_triangle right_triangle :=
sorry

end construct_right_triangle_l73_73476


namespace number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73231

theorem number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares :
  ∃ (n : ℕ), n = 3 ∧ ∀ a b c : ℕ, a^2 + b^2 + c^2 = 100 → 
    a*b*c ≠ 0 → a^2 ≤ b^2 ≤ c^2 :=
by sorry

end number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73231


namespace power_function_half_l73_73910

theorem power_function_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1/2)) (hx : f 4 = 2) : 
  f (1/2) = (Real.sqrt 2) / 2 :=
by sorry

end power_function_half_l73_73910


namespace non_intersecting_segments_exists_l73_73514

theorem non_intersecting_segments_exists (n : ℕ) (h1: 1 ≤ n) (points : Fin 2n → Point)
  (h_nocollinear : ∀ {p1 p2 p3 : Fin 2n}, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear points p1 p2 p3)
  (coloring : Fin 2n → Color) (h_blue : ∃f : Fin n → Fin 2n, ∀ i j : Fin n, i ≠ j → f i ≠ f j ∧ coloring (f i) = Color.blue)
  (h_red : ∃f : Fin n → Fin 2n, ∀ i j : Fin n, i ≠ j → f i ≠ f j ∧ coloring (f i) = Color.red) :
  ∃ S : Fin n → (Fin 2n × Fin 2n), (∀ i, coloring (S i).1 = Color.blue ∧ coloring (S i).2 = Color.red ∧ 
    (∀ j ≠ i, ¬ intersect (S i) (S j)) ∧ ∀ p ∈ Finset.image (λ x : Fin n, (S x).1) (Finset.univ) ∪ Finset.image (λ x : Fin n, (S x).2) (Finset.univ), 
     ∃ i : Fin n, p = (S i).1 ∨ p = (S i).2) :=
sorry

end non_intersecting_segments_exists_l73_73514


namespace triangle_area_84_li_l73_73705

def calculate_area (a b c : ℝ) : ℝ :=
  sqrt (1 / 4 * (c^2 * a^2 - (c^2 + a^2 - b^2)^2 / 4))

theorem triangle_area_84_li (a b c : ℝ) (h_a : a = 15) (h_b : b = 14) (h_c : c = 13) :
  calculate_area a b c = 84 :=
by {
  rw [h_a, h_b, h_c],
  -- calculation steps go here
  sorry
}

end triangle_area_84_li_l73_73705


namespace sufficient_condition_l73_73184

theorem sufficient_condition (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r :=
by
  intro hp
  apply hqr
  apply hpq
  exact hp

end sufficient_condition_l73_73184


namespace tan_cos_identity_15deg_l73_73485

theorem tan_cos_identity_15deg :
  (1 - (Real.tan (Real.pi / 12))^2) * (Real.cos (Real.pi / 12))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end tan_cos_identity_15deg_l73_73485


namespace common_terms_only_1_and_7_l73_73480

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * sequence_a (n - 1) - sequence_a (n - 2)

def sequence_b (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 7
  else 6 * sequence_b (n - 1) - sequence_b (n - 2)

theorem common_terms_only_1_and_7 :
  ∀ n m : ℕ, (sequence_a n = sequence_b m) → (sequence_a n = 1 ∨ sequence_a n = 7) :=
by {
  sorry
}

end common_terms_only_1_and_7_l73_73480


namespace intersection_points_y_a_x_log_a_x_l73_73628

noncomputable def count_intersection_points (a : ℝ) (hx_pos : 0 < a) (hx_lt1 : a < 1) : ℕ :=
  if a < real.exp (-1:ℝ) then 3 else 1

theorem intersection_points_y_a_x_log_a_x (a : ℝ) (ha : 0 < a) (ha_lt1 : a < 1) :
  (a < real.exp (-1:ℝ) → count_intersection_points a ha ha_lt1 = 3) ∧
  (real.exp (-1:ℝ) ≤ a → count_intersection_points a ha ha_lt1 = 1) :=
by
  sorry

end intersection_points_y_a_x_log_a_x_l73_73628


namespace count_primes_with_digit3_under_200_l73_73179

open Nat

theorem count_primes_with_digit3_under_200 : 
  {n : ℕ | n < 200 ∧ n % 10 = 3 ∧ Prime n}.toFinset.card = 12 :=
by
  sorry

end count_primes_with_digit3_under_200_l73_73179


namespace roll_combinations_l73_73013

-- Definitions for the quantities of each type of roll
def roll_quantities (a b c d : ℕ) : Prop :=
a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 1 ∧ d ≥ 1 ∧ (a + b + c + d = 10)

-- Main theorem statement
theorem roll_combinations : {p : ℕ × ℕ × ℕ × ℕ // roll_quantities p.1 p.2.1 p.2.2.1 p.2.2.2 }.card = 35 := 
sorry

end roll_combinations_l73_73013


namespace equilateral_triangle_sum_l73_73045

noncomputable def side_length : ℝ := 6
noncomputable def area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2
noncomputable def perimeter (s : ℝ) : ℝ := 3 * s
noncomputable def height (s : ℝ) : ℝ := (Real.sqrt 3 / 2) * s
noncomputable def ratio (a p : ℝ) : ℝ := a / p

theorem equilateral_triangle_sum :
  let s := 6 in
  let a := area s in
  let p := perimeter s in
  let r := ratio a p in
  let h := height s in
  let sum := r + h in
  sum = (7 * Real.sqrt 3) / 2 := by
  sorry

end equilateral_triangle_sum_l73_73045


namespace sam_stops_at_quarter_D_l73_73672

theorem sam_stops_at_quarter_D
    (circumference : ℕ)
    (quarters : ℕ)
    (total_distance : ℕ)
    (full_laps : ℕ)
    (remaining_distance : ℕ)
    (start_pos : string) :
    circumference = 100 →
    quarters = 4 →
    total_distance = 10000 →
    total_distance % circumference = 0 →
    total_distance / circumference = full_laps →
    remaining_distance = total_distance % circumference →
    start_pos = "S" →
    full_laps > 0 →
    remaining_distance = 0 →
    full_laps * circumference = total_distance →
    "D" = "D" :=
by sorry

end sam_stops_at_quarter_D_l73_73672


namespace avg_visitors_other_days_l73_73024

theorem avg_visitors_other_days (avg_sunday avg_month : ℕ) (days month_days : ℕ) 
  (starts_with_sunday : days = 30 ∧ month_days = 30):
  avg_sunday = 660 ∧ avg_month = 310 → 
  ∃ (V : ℕ), V = 240 :=
begin
  -- Let avg_sunday = 660
  -- Let avg_month = 310
  -- 5 Sundays in a 30-day month starting with Sunday
  have total_sundays := 5,
  have total_days := 30,
  have total_month := 9300,
  have sun_visitor := 3300,
  
  assume h,
  have avg_sunday_cond := h.1,
  have avg_month_cond := h.2,

  -- V is the average number of visitors on other days
  let V := (total_month - sun_visitor) / 25,
  use V,
  -- Prove V = 240
  have proof_v := (total_month - sun_visitor) = 6000,
  sorry
end

end avg_visitors_other_days_l73_73024


namespace proof_problem_l73_73624

/-- Definition of the problem -/
def problem_statement : Prop :=
  ∃(a b c : ℝ) (A B C : ℝ) (D : ℝ),
    -- Conditions:
    ((b ^ 2 = a * c) ∧
     (2 * Real.cos (A - C) - 2 * Real.cos B = 1) ∧
     (D = 5) ∧
     -- Questions:
     (B = Real.pi / 3) ∧
     (∀ (AC CD : ℝ), (a = b ∧ b = c) → -- Equilateral triangle
       (AC * CD = (1/2) * (5 * AC - AC ^ 2) ∧
       (0 < AC * CD ∧ AC * CD ≤ 25/8))))

-- Lean 4 statement
theorem proof_problem : problem_statement := sorry

end proof_problem_l73_73624


namespace smallest_k_l73_73767

def arith_seq_sum (k n : ℕ) : ℕ :=
  (n + 1) * (2 * k + n) / 2

theorem smallest_k (k n : ℕ) (h_sum : arith_seq_sum k n = 100) :
  k = 9 :=
by
  sorry

end smallest_k_l73_73767


namespace correct_answer_l73_73076

theorem correct_answer (x : ℝ) (h : 3 * x - 10 = 50) : 3 * x + 10 = 70 :=
sorry

end correct_answer_l73_73076


namespace exponentiation_evaluation_l73_73848

theorem exponentiation_evaluation :
  (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end exponentiation_evaluation_l73_73848


namespace number_of_subsets_l73_73335

theorem number_of_subsets (M : Finset ℕ) (h : M.card = 5) : 2 ^ M.card = 32 := by
  sorry

end number_of_subsets_l73_73335


namespace Lisa_days_l73_73678

theorem Lisa_days (L : ℝ) (h : 1/4 + 1/2 + 1/L = 1/1.09090909091) : L = 2.93333333333 :=
by sorry

end Lisa_days_l73_73678


namespace area_of_triangle_AEB_l73_73245

theorem area_of_triangle_AEB :
  ∀ (A B C D F G E : Type)
    [rect : Rectangle ABCD] 
    (hAB : AB = 7) 
    (hBC : BC = 4)
    (hDF : DF = 2)
    (hGC : GC = 3)
    (h_intersection : ∃ E, LinesIntersectAt AF BG E),
  ∃ (area : ℝ), area = (triangle_area A E B) ∧ area = 28 / 5 := 
by
  intros,
  sorry

end area_of_triangle_AEB_l73_73245


namespace marina_cannot_prevent_katya_goal_l73_73351

-- Defining the condition: 8 identical cubes
def num_cubes : ℕ := 8

-- Marina's painting condition: 24 faces blue, 24 faces red
def faces_blue : ℕ := 24
def faces_red : ℕ := 24

-- Katya's goal after assembling into 2x2x2 cube
def surface_total_faces : ℕ := 24
def katya_goal_faces : ℕ := 12

-- Theorem statement
theorem marina_cannot_prevent_katya_goal :
  ∀ (blue_faces red_faces : ℕ), blue_faces + red_faces = surface_total_faces → blue_faces = faces_blue / 2 → red_faces = faces_red / 2 →
  ∃ (rotation_scheme : (ℕ × ℕ) → (ℕ × ℕ)),
    (∀ i, rotation_scheme i = (blue_faces, red_faces))
    → blue_faces = katya_goal_faces ∧ red_faces = katya_goal_faces := 
begin
  sorry
end

end marina_cannot_prevent_katya_goal_l73_73351


namespace ratio_of_areas_l73_73475

theorem ratio_of_areas (s2 : ℝ) : 
  let s1 := s2 * Real.sqrt 2 in
  let s3 := s1 / 2 in
  let area2 := s2^2 in
  let area3 := s3^2 in
  area3 / area2 = 1 / 2 :=
by
  sorry

end ratio_of_areas_l73_73475


namespace expression_evaluation_l73_73462

theorem expression_evaluation :
  5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end expression_evaluation_l73_73462


namespace solve_for_a_l73_73954

theorem solve_for_a (a : ℝ) 
  (h : (2 * a + 16 + (3 * a - 8)) / 2 = 89) : 
  a = 34 := 
sorry

end solve_for_a_l73_73954


namespace quadratic_vertex_l73_73711

noncomputable def quadratic_vertex_max (c d : ℝ) (h : -x^2 + c * x + d ≤ 0) : (ℝ × ℝ) :=
sorry

theorem quadratic_vertex 
  (c d : ℝ)
  (h : -x^2 + c * x + d ≤ 0)
  (root1 root2 : ℝ)
  (h_roots : root1 = -5 ∧ root2 = 3) :
  quadratic_vertex_max c d h = (4, 1) ∧ (∀ x: ℝ, (x - 4)^2 ≤ 1) :=
sorry

end quadratic_vertex_l73_73711


namespace price_decrease_proof_l73_73810

-- Definitions based on the conditions
def original_price (C : ℝ) : ℝ := C
def new_price (C : ℝ) : ℝ := 0.76 * C

theorem price_decrease_proof (C : ℝ) : new_price C = 421.05263157894734 :=
by
  sorry

end price_decrease_proof_l73_73810


namespace count_non_distinct_real_solution_pairs_l73_73108

theorem count_non_distinct_real_solution_pairs :
  { (b, c) : ℕ × ℕ // b > 0 ∧ c > 0 ∧ b^2 ≤ 4 * c ∧ c^2 ≤ 4 * b }.card = 6 :=
by
  -- Proof steps would go here, but are replaced by sorry
  sorry

end count_non_distinct_real_solution_pairs_l73_73108


namespace cone_volume_increase_l73_73957

theorem cone_volume_increase
  (r h : ℝ) :
  let V := (1 / 3) * Real.pi * r^2 * h in
  let V' := (1 / 3) * Real.pi * (2 * r)^2 * (2 * h) in
  V' = 8 * V :=
by
  sorry

end cone_volume_increase_l73_73957


namespace triangle_BED_area_l73_73979

theorem triangle_BED_area :
  ∃ (ABC BED : Triangle) (M D E F : Point),
  obtuse_triangle ∧
  (∠B > 90°, AM = MB, MD ⊥ AC, EC ⊥ AB, EF ⊥ AC) ∧
  area(ABC) = 36 ∧
  (D ∈ AC) ∧ (E ∈ AB) ∧ (M ∈ EB) →
  area(BED) = 18 :=
by sorry

end triangle_BED_area_l73_73979


namespace relationship_y1_y2_l73_73199

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l73_73199


namespace count_primes_with_digit3_under_200_l73_73180

open Nat

theorem count_primes_with_digit3_under_200 : 
  {n : ℕ | n < 200 ∧ n % 10 = 3 ∧ Prime n}.toFinset.card = 12 :=
by
  sorry

end count_primes_with_digit3_under_200_l73_73180


namespace sum_factorials_mod_25_l73_73591

theorem sum_factorials_mod_25 : ((1! + 2! + 3! + 4!) % 25 = 8) := by
  sorry

end sum_factorials_mod_25_l73_73591


namespace train_length_l73_73439

variables 
  (train_speed : ℕ) -- speed of the train in km/h
  (man_speed : ℕ)   -- speed of the man in km/h
  (time_crossing : ℕ) -- time in seconds

def relative_speed_kmh := train_speed + man_speed
def relative_speed_mps := (relative_speed_kmh * 1000 / 3600 : ℕ)
def length_of_train := relative_speed_mps * time_crossing

theorem train_length 
  (h1 : train_speed = 25) 
  (h2 : man_speed = 2)
  (h3 : time_crossing = 44) : 
  length_of_train = 330 :=
by
  rw [length_of_train, relative_speed_mps, h1, h2, h3, ← Nat.cast_add, Nat.cast_mul, Nat.cast_add]
  norm_num
  sorry

end train_length_l73_73439


namespace cost_price_of_one_meter_of_cloth_l73_73760

variable (SellingPrice : ℕ) (MetersSold : ℕ) (ProfitPerMeter : ℕ) (CostPricePerMeter : ℕ)

def calculateTotalProfit (ProfitPerMeter MetersSold : ℕ) : ℕ := ProfitPerMeter * MetersSold
def calculateTotalCostPrice (SellingPrice TotalProfit : ℕ) : ℕ := SellingPrice - TotalProfit
def calculateCostPricePerMeter (TotalCostPrice MetersSold : ℕ) : ℕ := TotalCostPrice / MetersSold

theorem cost_price_of_one_meter_of_cloth (h1 : SellingPrice = 8500) (h2 : MetersSold = 85) (h3 : ProfitPerMeter = 15) :
  CostPricePerMeter = 85 :=
by
  let TotalProfit := calculateTotalProfit ProfitPerMeter MetersSold
  have h4 : TotalProfit = 1275 := by sorry
  let TotalCostPrice := calculateTotalCostPrice SellingPrice TotalProfit
  have h5 : TotalCostPrice = 7225 := by sorry
  have h6 : CostPricePerMeter = calculateCostPricePerMeter TotalCostPrice MetersSold := by sorry
  show CostPricePerMeter = 85, by sorry

end cost_price_of_one_meter_of_cloth_l73_73760


namespace function_properties_l73_73329

theorem function_properties :
  let f : ℝ → ℝ := λ x, 1 - 2 * sin^2 (x - 3 * Real.pi / 4)
  in (∀ x, f (-x) = -f x) ∧ (∃ p > 0, ∀ x, f (x + p) = f x ∧ p = Real.pi) :=
by
  sorry

end function_properties_l73_73329


namespace combined_wattage_l73_73427

theorem combined_wattage (w1 w2 w3 w4 : ℕ) (h1 : w1 = 60) (h2 : w2 = 80) (h3 : w3 = 100) (h4 : w4 = 120) :
  let nw1 := w1 + w1 / 4
  let nw2 := w2 + w2 / 4
  let nw3 := w3 + w3 / 4
  let nw4 := w4 + w4 / 4
  nw1 + nw2 + nw3 + nw4 = 450 :=
by
  sorry

end combined_wattage_l73_73427


namespace smallest_k_l73_73768

def arith_seq_sum (k n : ℕ) : ℕ :=
  (n + 1) * (2 * k + n) / 2

theorem smallest_k (k n : ℕ) (h_sum : arith_seq_sum k n = 100) :
  k = 9 :=
by
  sorry

end smallest_k_l73_73768


namespace sequence_geometric_sum_l73_73139

theorem sequence_geometric_sum
  (a : ℕ → ℝ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 5)
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, (log 2 (a n - 1) - log 2 (a (n-1) - 1)) = d) :
  (∃ r : ℝ, ∀ n : ℕ, a n - 1 = (a 1 - 1) * r ^ n) ∧ 
  (∀ n : ℕ, ∑ i in finset.range n, 1 / (a (i+2) - a (i+1)) = 1 - 1 / 2 ^ n) :=
by {
  sorry
}

end sequence_geometric_sum_l73_73139


namespace primes_with_3_as_ones_digit_below_200_l73_73177

theorem primes_with_3_as_ones_digit_below_200 : 
  (finset.filter (λ x, x % 10 = 3) (finset.filter nat.prime (finset.Ico 1 200))).card = 12 :=
sorry

end primes_with_3_as_ones_digit_below_200_l73_73177


namespace PQ_parallel_AB_l73_73455

theorem PQ_parallel_AB 
  (A B C M K L P Q : Point)
  (hM : M ∈ Line.mk A B)
  (hK : K ∈ Line.mk B C)
  (hL : L ∈ Line.mk C A)
  (hMKAC : Line.mk M K ∥ Line.mk A C)
  (hMLBC : Line.mk M L ∥ Line.mk B C)
  (hP : P = intersection (Line.mk B L) (Line.mk M K))
  (hQ : Q = intersection (Line.mk A K) (Line.mk M L))
  : Line.mk P Q ∥ Line.mk A B :=
sorry

end PQ_parallel_AB_l73_73455


namespace find_a_purely_imaginary_l73_73145

theorem find_a_purely_imaginary (a : ℝ) (i : ℂ) (hi : i^2 = -1) :
  (let z := ((a^2 : ℂ) * i / (2 - i : ℂ)) + ((1 - 2 * a * i : ℂ) / 5) in
   z.im ≠ 0 ∧ z.re = 0) ↔ a = -1 :=
by sorry

end find_a_purely_imaginary_l73_73145


namespace find_constants_l73_73888

-- Definitions based on the given problem
def inequality_in_x (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def roots_eq (a : ℝ) (r1 r2 : ℝ) : Prop :=
  a * r1^2 - 3 * r1 + 2 = 0 ∧ a * r2^2 - 3 * r2 + 2 = 0

def solution_set (a b : ℝ) (x : ℝ) : Prop :=
  x < 1 ∨ x > b

-- Problem statement: given conditions find a and b
theorem find_constants (a b : ℝ) (h1 : 1 < b) (h2 : 0 < a) :
  roots_eq a 1 b ∧ solution_set a b 1 ∧ solution_set a b b :=
sorry

end find_constants_l73_73888


namespace exercise_8_proof_l73_73658

-- Assume n is a natural number and p is a prime number
variables (n : ℕ) (p : ℕ) [hp : nat.prime p]

-- Legendre's formulas as referenced in the problem conditions
def legendre (m p : ℕ) := ∑ k in (finset.range (nat.floor (real.log (m) / real.log (p)) + 1)).filter (λ k, 1 ≤ k), (m / p^k)

-- Define Legendre's formula for binomial coefficient based on exercise 8 
def legendre_binom (n p : ℕ) : ℕ :=
  (∑ k in (finset.range (nat.floor (real.log (2 * n) / real.log (p)) + 1)).filter (λ k, 1 ≤ k),
    (2 * n) / p^k - 2 * (n / p^k))

noncomputable theory
-- The inequality from exercise 5
def exercise_5_inequality (n p k : ℕ) := ((2 * n / p^k) - 2 * (n / p^k)) ≤ 1

-- Main proof statement
theorem exercise_8_proof :
  legendre_binom n p ≤ nat.floor (real.log (2 * n) / real.log p) := by
  sorry -- proof goes here

end exercise_8_proof_l73_73658


namespace triangle_XYZ_right_angle_l73_73987

theorem triangle_XYZ_right_angle (X Y Z XZ YZ : ℝ) (h: ∠X = 90°) 
(hYZ: YZ = 30) (h_tan_sin: tan Z = 3 * sin Z): XY = 20 * sqrt 2 :=
by
  let XY := sqrt (YZ^2 - XZ^2)
  have h_pythagoras : XY^2 + XZ^2 = YZ^2 := sorry
  have h1 : XY / XZ = tan Z := sorry
  have h2 : XY / 30 = sin Z := sorry
  have h3 : tan Z = 3 * sin Z := by exact h_tan_sin
  have h4 : XY / XZ = (XY / 30) * 3 := sorry
  have h5 : 30 / YZ = 1 := by exact (eq.trans (one_div_eq_inv 30).symm (inv_eq_one_div 30))
  have h6 : XY = √(XY^2) := sqrt_sqr (eq_refl XY)
  have h7: XY^2 = 800 := by
    rw [←add_right_eq_self_mpr (eq.symm (eq.trans (add_right_elect 100).symm (eq.trans (pow_two 10: eq.symm (mul_self_pow 2 10) 10))).trans sorry, pow_two 30],
  have XY = sqrt 800 := by rw [add_eq_of_eq_sub of_sorry, ←eq.rfl_mpr (sqrt_eq_iff_sqr_eq.mpr 800_nonneg sorry)]
  have XY = 20 * sqrt 2 := by rw [sqrt 800, eq.symm (mul_comm 20 (sqrt 2)), ←mul_assoc, mul_assoc 20, mul_comm (sqrt 2) 2, sqrt_mul]
  exact ah

end triangle_XYZ_right_angle_l73_73987


namespace diameter_of_circular_field_l73_73495

-- Definition of conditions as parameters
def cost_per_meter : ℝ := 2
def total_cost : ℝ := 213.63

-- Definition of diameter given these conditions
theorem diameter_of_circular_field : (106.815 / Real.pi) ≈ 34 :=
by
  sorry

end diameter_of_circular_field_l73_73495


namespace correct_statements_l73_73353

variable (r : ℝ) (x y : ℝ)

-- Conditions
def cond1 : Prop := r > 0 → (x increases → y increases)
def cond2 : Prop := r < 0 → (x increases → y increases)
def cond3 : Prop := (r = 1 ∨ r = -1) → (perfect correlation (a functional relationship) ∧ all scatter points on a straight line)

-- Statements
def statement1 : Prop := r > 0 → (x increases → y increases)
def statement2 : Prop := r < 0 → (x increases → y increases)
def statement3 : Prop := (r = 1 ∨ r = -1) → (perfect correlation (a functional relationship) ∧ all scatter points on a straight line)

theorem correct_statements :
  statement1 ∧ statement3 :=
by
  sorry

end correct_statements_l73_73353


namespace smallest_M_inequality_l73_73845

theorem smallest_M_inequality :
  ∃ M : ℝ, (∀ a b c : ℝ, abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) ≤ M * (a^2 + b^2 + c^2)^2)
  ∧ M = 9 * Real.sqrt 2 / 32 :=
begin
  use 9 * Real.sqrt 2 / 32,
  split,
  { intros a b c,
    sorry },
  { refl }
end

end smallest_M_inequality_l73_73845


namespace factor_of_land_increase_l73_73052

-- Definitions of the conditions in the problem:
def initial_money_given_by_blake : ℝ := 20000
def money_received_by_blake_after_sale : ℝ := 30000

-- The main theorem to prove
theorem factor_of_land_increase (F : ℝ) : 
  (1/2) * (initial_money_given_by_blake * F) = money_received_by_blake_after_sale → 
  F = 3 :=
by sorry

end factor_of_land_increase_l73_73052


namespace probability_of_winning_pair_l73_73603

-- Define the deck and winning conditions
inductive Color
| Blue
| Yellow

inductive Label
| A
| B
| C

structure Card where
  color : Color
  label : Label

def deck : List Card :=
  [ {color := Color.Blue,  label := Label.A}
  , {color := Color.Blue,  label := Label.B}
  , {color := Color.Blue,  label := Label.C}
  , {color := Color.Yellow, label := Label.A}
  , {color := Color.Yellow, label := Label.B}
  , {color := Color.Yellow, label := Label.C}
  -- Three more cards of each label and color repeated
  -- for a total of six blue and six yellow cards (Example shortened)
  ]

def isWinningPair (card1 card2 : Card) : Bool :=
  (card1.color = card2.color) ∨ ((card1.label = card2.label) ∧ (card1.color ≠ card2.color))

def calculateProbability : ℚ :=
  -- Here we assume the calculation based on the described steps
  3 / 154

-- The proof statement
theorem probability_of_winning_pair :
  calculateProbability = 3 / 154 :=
by
  -- Missing the actual proof
  sorry

end probability_of_winning_pair_l73_73603


namespace set_condition_implies_union_l73_73581

open Set

variable {α : Type*} {M P : Set α}

theorem set_condition_implies_union 
  (h : M ∩ P = P) : M ∪ P = M := 
sorry

end set_condition_implies_union_l73_73581


namespace lines_divide_plane_l73_73675

def numberOfRegions (n : ℕ) (lambda : ℝ → ℝ → ℕ) : ℕ := 1 + n + (λ.sum P, lambda P - 1)

theorem lines_divide_plane (n : ℕ) (lambda : ℝ → ℝ → ℕ) :
  ∃ R U, 
    R = 1 + n + (λ.sum P, lambda P - 1) ∧
    U = 2 * n :=
  sorry

end lines_divide_plane_l73_73675


namespace range_of_m_l73_73401

variables (p q : Prop) (m : ℝ)
def prop_p : Prop := m < 1
def prop_q : Prop := m < 2
def either_p_or_q : Prop := (prop_p p m ∨ prop_q q m)
def both_not_p_and_q : Prop := ¬ (prop_p p m ∧ prop_q q m)

theorem range_of_m : either_p_or_q p q m ∧ both_not_p_and_q p q m → (1 ≤ m ∧ m < 2) :=
sorry

end range_of_m_l73_73401


namespace am_gm_inequality_l73_73166

noncomputable def arithmetic_mean (a c : ℝ) : ℝ := (a + c) / 2

noncomputable def geometric_mean (a c : ℝ) : ℝ := Real.sqrt (a * c)

theorem am_gm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  (arithmetic_mean a c - geometric_mean a c < (c - a)^2 / (8 * a)) :=
sorry

end am_gm_inequality_l73_73166


namespace solve_xyz_l73_73281

theorem solve_xyz (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  (x, y, z) = ( (b + c) / 2, (c + a) / 2, (a + b) / 2 ) :=
sorry

end solve_xyz_l73_73281


namespace select_from_companyA_l73_73829

noncomputable def companyA_representatives : ℕ := 40
noncomputable def companyB_representatives : ℕ := 60
noncomputable def total_representatives : ℕ := companyA_representatives + companyB_representatives
noncomputable def sample_size : ℕ := 10
noncomputable def sampling_ratio : ℚ := sample_size / total_representatives
noncomputable def selected_from_companyA : ℚ := companyA_representatives * sampling_ratio

theorem select_from_companyA : selected_from_companyA = 4 := by
  sorry


end select_from_companyA_l73_73829


namespace dihedral_angle_cosine_l73_73736

noncomputable def dihedral_cosine (r : ℝ) (d : ℝ) (α : ℝ) : Prop :=
  d = 5 * r ∧ α = 60 ∧ cos (α * (Real.pi / 180)) * d / (2 * r) = 1.25

theorem dihedral_angle_cosine (r : ℝ) (θ : ℝ) (h : dihedral_cosine r (5 * r) 60) :
  cos θ = 0.04 :=
sorry

end dihedral_angle_cosine_l73_73736


namespace min_value_of_reciprocal_l73_73183

noncomputable def min_value_condition (m n : ℝ) : Prop := 
  m + n = 1 ∧ m > 0 ∧ n > 0

theorem min_value_of_reciprocal (m n : ℝ) (h : min_value_condition m n) :
  4 ≤ (1 / m) + (1 / n) :=
sorry

end min_value_of_reciprocal_l73_73183


namespace qualified_diameter_given_smoothness_qualified_smoothness_given_diameter_l73_73347

theorem qualified_diameter_given_smoothness :
  let total_parts := 100
  let qualified_diameter := 98
  let qualified_smoothness := 96
  let both_qualified := 94
  (qualified_smoothness ≠ 0) → 
  (both_qualified / qualified_smoothness : ℝ) = 0.979 :=
by simp [total_parts, qualified_diameter, qualified_smoothness, both_qualified]; sorry

theorem qualified_smoothness_given_diameter :
  let total_parts := 100
  let qualified_diameter := 98
  let qualified_smoothness := 96
  let both_qualified := 94
  (qualified_diameter ≠ 0) → 
  (both_qualified / qualified_diameter : ℝ) = 0.959 :=
by simp [total_parts, qualified_diameter, qualified_smoothness, both_qualified]; sorry

end qualified_diameter_given_smoothness_qualified_smoothness_given_diameter_l73_73347


namespace negation_of_proposition_l73_73708

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 1) ↔ ∃ x : ℝ, x^2 + 1 < 1 :=
by sorry

end negation_of_proposition_l73_73708


namespace area_between_chords_l73_73366

/-- 
  Given a circle of radius 10 inches,
  and two equal parallel chords that lie 12 inches apart,
  prove that the area of the circle that lies between the two chords is \(32\sqrt{3} + 21\frac{1}{3}\pi\).
-/
theorem area_between_chords (r : ℝ) (d : ℝ) (h_r : r = 10) (h_d : d = 12) :
  let area := 32 * real.sqrt 3 + (64 / 3) * real.pi in
  area = 32 * real.sqrt 3 + 21 * (1 / 3) * real.pi :=
by
  sorry

end area_between_chords_l73_73366


namespace num_intersecting_points_is_560_l73_73077

noncomputable def num_intersection_points : ℕ :=
let a_values := {-3, -2, -1, 0, 1} in
let b_values := {-2, -1, 0, 1, 2} in
let parabolas := {(a, b) | a ∈ a_values, b ∈ b_values}.to_finset in
let pairs := (parabolas.powerset).filter (λ s, s.card = 2) in
let parallel_pairs := (pairs.filter (λ s, (s.to_finset.to_list.map (λ (p : ℤ × ℤ), p.fst)).nodup)) in
let non_parallel_pairs := pairs.card - parallel_pairs.card in
2 * non_parallel_pairs - 2 * parallel_pairs.card

theorem num_intersecting_points_is_560 : num_intersection_points = 560 :=
sorry

end num_intersecting_points_is_560_l73_73077


namespace perimeter_circular_base_is_24pi_l73_73713

-- Definitions and conditions
def height_of_cylinder : ℝ := 16
def diagonal_of_rectangle : ℝ := 20
def circumference_of_base (h d : ℝ) : ℝ := 
  let l := Math.sqrt (d^2 - h^2) in 2 * Real.pi * (l / (2 * Real.pi))

-- Statement to prove
theorem perimeter_circular_base_is_24pi :
  circumference_of_base height_of_cylinder diagonal_of_rectangle = 24 * Real.pi :=
sorry

end perimeter_circular_base_is_24pi_l73_73713


namespace three_digit_cubes_divisible_by_eight_l73_73568

theorem three_digit_cubes_divisible_by_eight :
  (∃ n1 n2 : ℕ, 100 ≤ n1 ∧ n1 < 1000 ∧ n2 < n1 ∧ 100 ≤ n2 ∧ n2 < 1000 ∧
  (∃ m1 m2 : ℕ, 2 ≤ m1 ∧ 2 ≤ m2 ∧ n1 = 8 * m1^3  ∧ n2 = 8 * m2^3)) :=
sorry

end three_digit_cubes_divisible_by_eight_l73_73568


namespace workman_B_days_l73_73417

theorem workman_B_days (A B : ℝ) (hA : A = (1 / 2) * B) (hTogether : (A + B) * 14 = 1) :
  1 / B = 21 :=
sorry

end workman_B_days_l73_73417


namespace pencils_in_drawer_l73_73349

def initial_pencils : ℕ := 41
def added_pencils : ℕ := 30
def total_pencils : ℕ := initial_pencils + added_pencils

theorem pencils_in_drawer (initial_pencils = 41) (added_pencils = 30) : total_pencils = 71 := by
  unfold total_pencils
  unfold initial_pencils
  unfold added_pencils
  sorry

end pencils_in_drawer_l73_73349


namespace v2004_eq_1_l73_73699

def g (x: ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- assuming default value for undefined cases

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n + 1)

theorem v2004_eq_1 : v 2004 = 1 :=
  sorry

end v2004_eq_1_l73_73699


namespace am_gm_inequality_l73_73283

noncomputable section

variables {n : ℕ} (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) (h_prod : (∏ i, a i) = 1)

theorem am_gm_inequality
    (h_pos : ∀ i, 0 < a i) 
    (h_prod : (∏ i, a i) = 1) :
    (∏ i in range 2, (1 + a i) ^ i) ≥ n ^ n :=
sorry

end am_gm_inequality_l73_73283


namespace find_a_l73_73615

noncomputable def line_parametric (a t : ℝ) : ℝ × ℝ :=
  (a + 3 / 5 * t, 1 + 4 / 5 * t)

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ * θ ^ 2 + 8 * cos θ - ρ = 0

noncomputable def curve_cartesian (x y : ℝ) : Prop :=
  y ^ 2 = 8 * x

theorem find_a (a : ℝ) :
  let PA := (0 - a)^2 + (1 - 1) ^ 2 -- calculating PA in xOy cartesian coordinates
  let PB := (a + 3 / 5 * t - a)^2 + (1 + 4 / 5 * t - 1)^2 -- calculating PB in xOy cartesian coordinates
  PA = 3 * PB -- given condition |PA| = 3|PB|
  →
  (a = 13 / 8 ∨ a = -1 / 4) :=
by sorry

end find_a_l73_73615


namespace sum_100th_row_l73_73471

noncomputable def h : ℕ → ℕ
| 0     := 0  -- Base case for 0th row which realistically doesn't exist but for 1-index.
| 1     := 2  -- Sum of the first row as given
| (n+1) := 2 * h n + 2 * n  -- Recursive definition

theorem sum_100th_row : h 100 = 2^100 + 9900 :=
by
  sorry

end sum_100th_row_l73_73471


namespace sqrt_fraction_value_l73_73132

theorem sqrt_fraction_value (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 9) (h3 : x < y) :
  (sqrt x - sqrt y) / (sqrt x + sqrt y) = -sqrt 3 / 3 :=
by sorry

end sqrt_fraction_value_l73_73132


namespace fraction_of_action_figures_l73_73348

theorem fraction_of_action_figures (T D : ℕ) (hT : T = 24) (hD : D = 18) :
  (T - D) / T = 1 / 4 :=
by
  rw [hT, hD]
  norm_num
  sorry

end fraction_of_action_figures_l73_73348


namespace simplify_fraction_l73_73322

theorem simplify_fraction : 
  ∃ (c d : ℤ), ((∀ m : ℤ, (6 * m + 12) / 3 = c * m + d) ∧ c = 2 ∧ d = 4) → 
  c / d = 1 / 2 :=
by
  sorry

end simplify_fraction_l73_73322


namespace probability_square_root_less_than_eight_eq_3_over_5_l73_73377

open Set

noncomputable def two_digit_numbers : Finset ℕ := (Finset.range 100).filter (λ x, x ≥ 10)
noncomputable def square_root_less_than_eight : Finset ℕ := two_digit_numbers.filter (λ x, Real.sqrt x < 8)

theorem probability_square_root_less_than_eight_eq_3_over_5 :
  (square_root_less_than_eight.card : ℚ) / two_digit_numbers.card = 3 / 5 :=
by
  sorry -- Proof to be completed by the user

end probability_square_root_less_than_eight_eq_3_over_5_l73_73377


namespace apex_dihedral_angle_l73_73054

open_locale real

-- Define the regular square pyramid and the geometry context
variables {O A B S : Point ℝ}

-- Conditions of the problem
def centers_coincide (O : Point ℝ) : Prop :=
  inscribed_sphere_center O ∧ circumscribed_sphere_center O

def regular_pyramid (A B S: Point ℝ) : Prop :=
  is_square_base A B ∧ is_regular_faces A B S

-- Theorem statement
theorem apex_dihedral_angle (O A B S : Point ℝ) 
  (h1 : centers_coincide O)
  (h2 : regular_pyramid A B S) :
  dihedral_angle_at_apex A B S = 45 :=
begin
  sorry
end

end apex_dihedral_angle_l73_73054


namespace shaded_box_is_five_l73_73850

-- Define the grid and conditions
def grid := (ℕ × (ℕ × ℕ × ℕ)) × (ℕ × (ℕ × ℕ × ℕ))

-- Define the function that will take in the grid and prove the assertion
def isValidGrid (g : grid) : Prop :=
  let ((n1, (q1, r1, n2)), (t, (n3, s, p))) := g in
  n1 = 3 ∧ n2 = 8 ∧ n3 = 4 ∧
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36) ∧
  (n1 + q1 + r1 + n2 = 18) ∧ (t + n3 + s + p = 18) ∧
  (n1 + t = 9) ∧ (q1 + n3 = 9) ∧ (r1 + s = 9) ∧ (n2 + p = 9)

-- Define the main statement that encapsulates the problem solution
theorem shaded_box_is_five (g : grid) : isValidGrid g → g = ((3, (4, 3, 8)), (5, (4, 3, 1))) :=
by {
  intros,
  sorry -- Proof goes here.
}

end shaded_box_is_five_l73_73850


namespace hyperbola_standard_equation_l73_73697

theorem hyperbola_standard_equation :
  (∃ c : ℝ, c = Real.sqrt 5) →
  (∃ a b : ℝ, b / a = 2 ∧ a ^ 2 + b ^ 2 = 5) →
  (∃ a b : ℝ, a = 1 ∧ b = 2 ∧ (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)) :=
by
  sorry

end hyperbola_standard_equation_l73_73697


namespace missed_questions_l73_73298

theorem missed_questions (F U : ℕ) (h1 : U = 5 * F) (h2 : F + U = 216) : U = 180 :=
by
  sorry

end missed_questions_l73_73298


namespace expected_value_X_probability_two_white_balls_from_B_after_transfer_l73_73227

-- Definitions of the conditions
def boxA : List Bool := [true, true, true, false, false] -- true for white, false for red
def boxB : List Bool := [true, true, true, true, false] -- true for white, false for red

-- First proof problem: the expected value of X
theorem expected_value_X :
  let X := [0, 1, 2, 3]
  let P_X_0 := (3.choose 2 / 5.choose 2) * (6 / 10)
  let P_X_1 := ((2.choose 1 * 3.choose 1) / 5.choose 2) * (6 / 10) + (3.choose 2 / 5.choose 2) * (4 / 10)
  let P_X_3 := (1 / 10) * (4 / 10)
  let P_X_2 := 1 - P_X_0 - P_X_1 - P_X_3
  let E_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3
  E_X = 6 / 5 :=
by
  sorry

-- Second proof problem: the probability of drawing two white balls from box B after transfer
theorem probability_two_white_balls_from_B_after_transfer :
  let probability_transfer_and_draw :=
    let PA1 := (2.choose 2 / 5.choose 2)
    let PB_given_A1 := (4.choose 2 / 7.choose 2)
    let PA2 := (3.choose 2 / 5.choose 2)
    let PB_given_A2 := (6.choose 2 / 7.choose 2)
    let PA3 := (3.choose 1 * 2.choose 1 / 5.choose 2)
    let PB_given_A3 := (5.choose 2 / 7.choose 2)
    PA1 * PB_given_A1 + PA2 * PB_given_A2 + PA3 * PB_given_A3
  probability_transfer_and_draw = 37 / 70 :=
by
  sorry

end expected_value_X_probability_two_white_balls_from_B_after_transfer_l73_73227


namespace problem1_problem2_l73_73833

-- Problem 1: Prove m = 3 given the conditions
theorem problem1
  (a : ℝ) (h_a : a = 1) (f : ℝ → ℝ) (h_f : ∀ x : ℝ, f x = x^2 - 4 * a * x + 3 * a)
  (h_sol : ∀ x : ℝ, f x < 0 ↔ 1 < x ∧ x < m) :
  m = 3 :=
sorry

-- Problem 2: Prove the range of k is -1 < k < -2 + sqrt(7) given the conditions
theorem problem2
  (a : ℝ) (h_a : 0 < a ∧ a < 3 / 4)
  (h_ineq : ∀ x k : ℝ, x ∈ Icc 0 1 → a ^ (k + 3) < a ^ (x ^ 2 - k * x) ∧ a ^ (x ^ 2 - k * x) < a ^ (k - 3)) :
  ∀ k : ℝ, -1 < k ∧ k < -2 + Real.sqrt 7 :=
sorry

end problem1_problem2_l73_73833


namespace truck_travel_l73_73442

/-- If a truck travels 150 miles using 5 gallons of diesel, then it will travel 210 miles using 7 gallons of diesel. -/
theorem truck_travel (d1 d2 g1 g2 : ℕ) (h1 : d1 = 150) (h2 : g1 = 5) (h3 : g2 = 7) (h4 : d2 = d1 * g2 / g1) : d2 = 210 := by
  sorry

end truck_travel_l73_73442


namespace evaluate_f_g_of_3_l73_73573

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem evaluate_f_g_of_3 : f(g(3)) = 61 :=
by
  -- skipping the proof
  sorry

end evaluate_f_g_of_3_l73_73573


namespace equal_intercepts_doesnt_pass_second_quadrant_l73_73287

-- Definition and conditions for the first proof problem
def line_eq (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

-- First proof statement: Prove the equation of the line with equal intercepts
theorem equal_intercepts (a: ℝ) : 
  (∃ b: ℝ, (a + 1) = -b ∧ -b * b + b + 2 - a = 0) →
  a = -2 → ∀ x y, line_eq -2 x y ↔ x + y + 4 = 0 := by
  sorry

-- Definition and conditions for the second proof problem
-- second quadrant condition means line should not intersect region with x<0 and y>0

-- Second proof statement: Prove that the line does not pass through the second quadrant
theorem doesnt_pass_second_quadrant (a : ℝ) : 
  (∀ x y : ℝ, line_eq a x y → x * y < 0) → (a ≤ -1) := by
  sorry

end equal_intercepts_doesnt_pass_second_quadrant_l73_73287


namespace painting_methods_correct_l73_73021

def num_painting_methods : Nat := 72

theorem painting_methods_correct :
  let vertices : Fin 4 := by sorry -- Ensures there are four vertices
  let edges : Fin 4 := by sorry -- Ensures each edge has different colored endpoints
  let available_colors : Fin 4 := by sorry -- Ensures there are four available colors
  num_painting_methods = 72 :=
sorry

end painting_methods_correct_l73_73021


namespace altara_population_2040_l73_73088

noncomputable def population (start_year : ℕ) (start_pop : ℕ) (years : ℕ) : ℕ :=
  start_pop * 2 ^ (years / 10)

theorem altara_population_2040 :
  population 2020 500 20 = 2000 :=
by
  unfold population
  norm_num
  sorry

end altara_population_2040_l73_73088


namespace expected_max_diff_leq_entropy_l73_73725

-- Definitions and conditions
def n : ℕ := sorry  -- Number of tokens
def X_i (i : ℕ) : ℝ := sorry  -- Ratio of white tokens before ith extraction
def T : ℝ := sorry  -- Maximum difference of X_i
def H (x : ℝ) : ℝ := -x * Real.log x - (1 - x) * Real.log (1 - x)

-- Hypotheses
variable (h1 : 0 < ∃ X_i (1 ≤ i ≤ n))
variable (h2 : ∃ i, X_i i < 1)

-- Goal
theorem expected_max_diff_leq_entropy :
  sorry -- Proof omitted

end expected_max_diff_leq_entropy_l73_73725


namespace min_checkered_rectangles_smallest_number_of_checkered_rectangles_l73_73990

/-- Given a specific grid with cells labeled using different letters (A, B, C, D, E, F),
it is required to determine the smallest number of rectangles that can divide the grid
such that each rectangle consists of one or several cells of the grid, and no rectangle
contains cells with different letters. -/
theorem min_checkered_rectangles (grid : List (Set Char)) (distinct_labels : Set Char) (correct_output : Nat) : Nat :=
  sorry

theorem smallest_number_of_checkered_rectangles : min_checkered_rectangles [["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "E"], ["F", "F"]] {"A", "B", "C", "D", "E", "F"} 7 := 
  sorry

end min_checkered_rectangles_smallest_number_of_checkered_rectangles_l73_73990


namespace linear_function_points_relation_l73_73202

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ),
    (y1 = 2 * (-3) + 1) →
    (y2 = 2 * 4 + 1) →
    (y1 = -5) ∧ (y2 = 9) :=
by
  intros y1 y2 hy1 hy2
  split
  · exact hy1
  · exact hy2

end linear_function_points_relation_l73_73202


namespace initial_music_files_l73_73452

-- Define the conditions
def video_files : ℕ := 21
def deleted_files : ℕ := 23
def remaining_files : ℕ := 2

-- Theorem to prove the initial number of music files
theorem initial_music_files : 
  ∃ (M : ℕ), (M + video_files - deleted_files = remaining_files) → M = 4 := 
sorry

end initial_music_files_l73_73452


namespace true_proposition_l73_73302

noncomputable def prop_p (x : ℝ) : Prop := x > 0 → x^2 - 2*x + 1 > 0

noncomputable def prop_q (x₀ : ℝ) : Prop := x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

theorem true_proposition : ¬ (∀ x > 0, x^2 - 2*x + 1 > 0) ∧ (∃ x₀ > 0, x₀^2 - 2*x₀ + 1 ≤ 0) :=
by
  sorry

end true_proposition_l73_73302


namespace brianna_has_29_more_chocolates_than_alix_l73_73671

noncomputable def chocolates_difference : ℕ :=
  let nick_chocolates := 10 in
  let alix_chocolates := 10 * 3 in
  let alix_after_mom := alix_chocolates - (alix_chocolates * 1 / 4).toNat in
  let total_chocolates_nick_alix := nick_chocolates + alix_after_mom in
  let brianna_chocolates := 2 * total_chocolates_nick_alix in
  let brianna_after_mom := brianna_chocolates - (brianna_chocolates * 20 / 100).toNat in
  brianna_after_mom - alix_after_mom

theorem brianna_has_29_more_chocolates_than_alix :
  chocolates_difference = 29 :=
by
  sorry

end brianna_has_29_more_chocolates_than_alix_l73_73671


namespace find_parking_cost_l73_73694

theorem find_parking_cost :
  ∃ (C : ℝ), (C + 7 * 1.75) / 9 = 2.4722222222222223 ∧ C = 10 :=
sorry

end find_parking_cost_l73_73694


namespace min_1x1_pieces_l73_73726

theorem min_1x1_pieces (board : ℕ × ℕ) (t1_cells t2_cells t3_cells : ℕ)
  (h_board_size : board = (100, 101))
  (h_t1_cells : t1_cells = 5 ∧ t1_cells ≠ 1)
  (h_t2_cells : t2_cells = 5 ∧ t2_cells ≠ 1)
  (h_t3_cells : t3_cells = 5 ∧ t3_cells ≠ 1) :
  ∃ n : ℕ, n = 2 ∧ (∀ pieces_used : ℕ, pieces_used = n → can_cover_board (100, 101) pieces_used) :=
by
  sorry

end min_1x1_pieces_l73_73726


namespace exists_right_triangle_area_twice_hypotenuse_l73_73487

theorem exists_right_triangle_area_twice_hypotenuse : 
  ∃ (a : ℝ), a ≠ 0 ∧ (a^2 / 2 = 2 * a * Real.sqrt 2) ∧ (a = 4 * Real.sqrt 2) :=
by
  sorry

end exists_right_triangle_area_twice_hypotenuse_l73_73487


namespace problem_statement_l73_73836

def op (x y : ℕ) : ℕ := x^2 + 2*y

theorem problem_statement (a : ℕ) : op a (op a a) = 3*a^2 + 4*a := 
by sorry

end problem_statement_l73_73836


namespace Midpoint_correct_l73_73124

-- Definition of a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of the midpoint of two points in 3D space
def Midpoint (A B : Point3D) : Point3D :=
  { x := (A.x + B.x) / 2
  , y := (A.y + B.y) / 2
  , z := (A.z + B.z) / 2 }

-- Given points A and B
def A : Point3D := { x := 3, y := 2, z := 1 }
def B : Point3D := { x := 1, y := 0, z := 5 }

-- Expected midpoint
def M_expected : Point3D := { x := 2, y := 1, z := 3 }

-- The proof statement
theorem Midpoint_correct : Midpoint A B = M_expected := by
  -- Proof is omitted, hence using sorry
  sorry

end Midpoint_correct_l73_73124


namespace part_one_part_two_l73_73116

variables {R : ℝ}
variables {A B C I O E : Type*} [EuclideanGeometry O A B C I E]

def angle_B_is_60_deg := angle B = 60
def angle_A_is_less_than_angle_C := angle A < angle C
def O_is_circumcircle := is_circumcircle O A B C
def I_is_incenter := is_incenter I A B C
def E_is_external_bisector_intersection :=
  exists (extBisector A B C : ℝ) : is_intersection E (extBisector A) (circumcircle O A B C)

theorem part_one (hO : O_is_circumcircle)
  (hI : I_is_incenter)
  (hB : angle_B_is_60_deg)
  (hA_C: angle_A_is_less_than_angle_C)
  (hE : E_is_external_bisector_intersection) :
  dist I O = dist A E := sorry

theorem part_two (hO : O_is_circumcircle)
  (hI : I_is_incenter)
  (hB : angle_B_is_60_deg)
  (hA : angle_A_is_less_than_angle_C)
  (hE : E_is_external_bisector_intersection) :
  2*R < dist O I + dist I A + dist I C ∧ dist O I + dist I A + dist I C < (1 + sqrt 2) * R := sorry

end part_one_part_two_l73_73116


namespace tan_x_plus_2y_eq_pi_div_4_l73_73537

theorem tan_x_plus_2y_eq_pi_div_4
  (x y : ℝ)
  (h1 : Real.Tan x = 1 / 7)
  (h2 : Real.Sin y = Real.sqrt 10 / 10)
  (h3 : 0 < x ∧ x < Real.pi / 2)
  (h4 : 0 < y ∧ y < Real.pi / 2) :
  x + 2 * y = Real.pi / 4 := 
sorry

end tan_x_plus_2y_eq_pi_div_4_l73_73537


namespace xiaoming_total_score_is_six_l73_73698

def is_monomial (n : ℕ) : Prop := ∃ (c : ℕ) (d : ℕ), n = c * d

def non_negative_rational_includes_zero : Prop := ∀ (q : ℚ), q ≥ 0 → q = 0 ∨ q > 0

def unequal_abs_sum_zero (a b : ℤ) : Prop := |a| ≠ |b| → a + b ≠ 0

def monomial_coeff_degree (a : ℤ) : Prop := 
  let coeff := if a < 0 then -1 else 1 in
  let degree := if a = 1 then 0 else 1 in
  coeff = 1 ∧ degree = 1

def rounding_correct (x : ℚ) : Prop := 
  let rounded := (Real.round (10 * x)) / 10 in
  rounded = 34.9

def xiaoming_answer_1 : bool := false
def xiaoming_answer_2 : bool := false
def xiaoming_answer_3 : bool := true
def xiaoming_answer_4 : bool := true
def xiaoming_answer_5 : bool := false

def correct_answer_1 : bool := is_monomial 1
def correct_answer_2 : bool := non_negative_rational_includes_zero
def correct_answer_3 : bool := unequal_abs_sum_zero 3 (-3)
def correct_answer_4 : bool := monomial_coeff_degree (-1)
def correct_answer_5 : bool := rounding_correct 34.945

def xiaoming_score : ℕ := 
  if xiaoming_answer_1 = correct_answer_1 then 2 else 0 +
  if xiaoming_answer_2 = correct_answer_2 then 2 else 0 +
  if xiaoming_answer_3 = correct_answer_3 then 2 else 0 +
  if xiaoming_answer_4 = correct_answer_4 then 2 else 0 +
  if xiaoming_answer_5 = correct_answer_5 then 2 else 0

theorem xiaoming_total_score_is_six : xiaoming_score = 6 := sorry

end xiaoming_total_score_is_six_l73_73698


namespace sum_sequence_l73_73555

noncomputable def S (n : ℕ) : ℕ :=
  2 + ∑ i in finset.range (n + 4), 2^(3 * i + 1)

theorem sum_sequence (n : ℕ) (hn : 0 < n) :
  S n = (2 / 7 * (8^(n + 4) - 1)) :=
by
  sorry

end sum_sequence_l73_73555


namespace hyperbola_focus_coordinates_l73_73693

theorem hyperbola_focus_coordinates (a b : ℝ) (h1 : a = 1) (h2 : b = sqrt 3) : 
  let c := sqrt (a^2 + b^2) in 
  c = 2 :=
by
  rw [h1, h2]
  have h3 : 1^2 + (sqrt 3)^2 = 4 := by norm_num
  rw h3
  exact sqrt_eq_of_sq_eq (by norm_num) (by norm_num)
  sorry

end hyperbola_focus_coordinates_l73_73693


namespace S_is_line_l73_73570

def complex_plane (z : ℂ) : Prop := (2 - 3 * complex.i) * z = complex.conj ((2 - 3 * complex.i) * z)

theorem S_is_line : ∀ z : ℂ, (complex_plane z) → ∃ (m : ℝ), ∀ (a b : ℝ), m = 3/2 ∧ z = a + b * complex.i  :=
by
  sorry

end S_is_line_l73_73570


namespace value_of_expression_l73_73947

theorem value_of_expression (x y : ℝ) (h1 : x = -2) (h2 : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 :=
by
  sorry

end value_of_expression_l73_73947


namespace correct_trigonometric_assertion_l73_73832

theorem correct_trigonometric_assertion :
  ∃ i ∈ {1, 2, 3, 4}, i = 2 ∧ (
    (∀ k : ℤ, ∃ a b : ℝ, (kπ - π / 2 < a ∧ a < kπ + π / 2) ∧ (kπ - π / 2 < b ∧ b < kπ + π / 2) → ¬(∀ x ∈ set.Ioo a b, monotone (@tan x))) ∧
    (∀ α β : ℝ, (0 < α ∧ α < π/2) ∧ (0 < β ∧ β < π/2) → sin α > cos β) ∧
    (∀ k : ℤ, axis_of_symmetry (cos (1/2 * x + 3π/2)) (x = 2kπ)) ∧
    (graph_shift (sin 2x) (-π / 4) = sin (2x + π / 4) → ¬(shift_left (sin 2x) (π / 4) = cos 2x))
  ) :=
begin
  sorry
end

end correct_trigonometric_assertion_l73_73832


namespace unique_three_positive_perfect_square_sums_to_100_l73_73233

theorem unique_three_positive_perfect_square_sums_to_100 :
  ∃! (a b c : ℕ), a^2 + b^2 + c^2 = 100 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end unique_three_positive_perfect_square_sums_to_100_l73_73233


namespace find_integer_root_l73_73338

-- Conditions
variables (p q : ℚ) -- p and q are rational numbers

-- Define that 3 - sqrt(5) is a root of the polynomial
axiom root_3_minus_sqrt_5 : (3 - real.sqrt 5)^3 + p * (3 - real.sqrt 5) + q = 0

-- The polynomial has an integer root r and the sum of the roots is zero
noncomputable def integer_root (p q : ℚ) (r : ℤ) : Prop := 
  (3 - real.sqrt 5) + (3 + real.sqrt 5) + r = 0

-- The proof statement
theorem find_integer_root (p q : ℚ) : ∃ r : ℤ, integer_root p q r :=
begin
  use -6,
  unfold integer_root,
  norm_num,
end

end find_integer_root_l73_73338


namespace solve_for_x_l73_73855

theorem solve_for_x (x : ℝ) :
  5 * (x - 9) = 7 * (3 - 3 * x) + 10 → x = 38 / 13 :=
by
  intro h
  sorry

end solve_for_x_l73_73855


namespace bears_distance_max_l73_73783

theorem bears_distance_max (n m : ℕ) (h1 : ∀ n, (1 + 1/(2 * n * Real.pi) = 1 + 1/(2 * 1 * Real.pi)))
  (h2 : ∀ m, (2 + 1/(m * Real.pi) = 2 + 1/(1 * Real.pi)))
  (h3 : (| (1 + 1/(2 * n * Real.pi)) - (2 + 1/(m * Real.pi)) | ≤ 3.477)) : 
  ∃ d, d = 3.477 :=
by
  sorry

end bears_distance_max_l73_73783


namespace max_a_for_necessary_not_sufficient_condition_l73_73937

theorem max_a_for_necessary_not_sufficient_condition {x a : ℝ} (h : ∀ x, x^2 > 1 → x < a) : a = -1 :=
by sorry

end max_a_for_necessary_not_sufficient_condition_l73_73937


namespace number_of_correct_propositions_is_zero_l73_73327

noncomputable def proposition_1 (cone : Type) (plane : Type) : Prop :=
∀ (c : cone) (p : plane), p cuts c → ¬(part between_base_and_section p c = frustum)

noncomputable def proposition_2 (solid : Type) : Prop :=
∀ (s : solid), (two_parallel_and_similar_faces s) ∧ (rest_are_trapezoids s) → ¬(called_truncated_pyramid s)

noncomputable def proposition_3 (hemisphere : Type) : Prop :=
∀ (h : hemisphere), (rotating_around_diameter_line h full_turn) → ¬(forms_sphere h)

noncomputable def proposition_4 (solid : Type) : Prop :=
∀ (s : solid), (two_parallel_faces s) ∧ (rest_are_parallelograms s) → ¬(called_prism s)

theorem number_of_correct_propositions_is_zero :
  ∀ (cone plane solid hemisphere : Type),
    proposition_1 cone plane ∧
    proposition_2 solid ∧
    proposition_3 hemisphere ∧
    proposition_4 solid → (count_correct 0) := sorry

end number_of_correct_propositions_is_zero_l73_73327


namespace find_m_perpendicular_vectors_l73_73934

theorem find_m_perpendicular_vectors :
  let a := (2 * Real.sqrt 2, 2)
  let b := (0, 2)
  let c m := (m, Real.sqrt 2)
  (let ab := (a.1, a.2 + 2 * b.2)
   in ab.1 * (c m).1 + ab.2 * (c m).2 = 0) -> (m = -3) :=
by
  intros
  let a := (2 * Real.sqrt 2, 2)
  let b := (0, 2)
  let c m := (m, Real.sqrt 2)
  let ab := (a.1, a.2 + 2 * b.2)
  have h : ab.1 * (c m).1 + ab.2 * (c m).2 = 0 -> (m = -3), from sorry
  exact h

end find_m_perpendicular_vectors_l73_73934


namespace cannot_have_2020_l73_73359

theorem cannot_have_2020 (a b c : ℤ) : 
  ∀ (n : ℕ), n ≥ 4 → 
  ∀ (x y z : ℕ → ℤ), 
    (x 0 = a) → (y 0 = b) → (z 0 = c) → 
    (∀ (k : ℕ), x (k + 1) = y k - z k) →
    (∀ (k : ℕ), y (k + 1) = z k - x k) →
    (∀ (k : ℕ), z (k + 1) = x k - y k) → 
    (¬ (∃ k, k > 0 ∧ k ≤ n ∧ (x k = 2020 ∨ y k = 2020 ∨ z k = 2020))) := 
by
  intros
  sorry

end cannot_have_2020_l73_73359


namespace cosine_value_smallest_angle_l73_73144

theorem cosine_value_smallest_angle :
  ∀ (a : ℝ), (a + 3 > 0) → (a + 6 > 0) → 
    (let A := Math.atan2 ((a + 3) * (a + 3) + (a + 6) * (a + 6) - a * a) (2 * (a + 3) * (a + 6)) in
    let C := 2 * A in
    cos A = (a + 6) / (2 * a)) ↔ cos A = 3 / 4 := sorry

end cosine_value_smallest_angle_l73_73144


namespace imaginary_part_complex_num_l73_73497

-- Define the complex number (2i / (1 - i)) + 2
def complex_num : ℂ := (2 * Complex.I) / (1 - Complex.I) + 2

-- Define the imaginary part of the complex number
def imag_part (z : ℂ) : ℝ := z.im

-- State the theorem
theorem imaginary_part_complex_num :
  imag_part complex_num = 1 :=
sorry

end imaginary_part_complex_num_l73_73497


namespace paint_required_for_frame_l73_73037

theorem paint_required_for_frame :
  ∀ (width height thickness : ℕ) 
    (coverage : ℚ),
  width = 6 →
  height = 9 →
  thickness = 1 →
  coverage = 5 →
  (width * height - (width - 2 * thickness) * (height - 2 * thickness) + 2 * width * thickness + 2 * height * thickness) / coverage = 11.2 :=
by
  intros
  sorry

end paint_required_for_frame_l73_73037


namespace problem_statement_l73_73681

noncomputable def g : ℝ → ℝ := sorry -- The function g will be a placeholder for now.
noncomputable def g_inv : ℝ → ℝ := g⁻¹ -- The inverse function of g.

-- Given conditions from the table
axiom g1 : g 1 = 3
axiom g2 : g 2 = 4
axiom g3 : g 3 = 6
axiom g4 : g 4 = 8
axiom g5 : g 5 = 9

-- The main problem statement
theorem problem_statement : g (g 3) + g (g_inv 5) + g_inv (g_inv 6) = "NEI" := by
  sorry

end problem_statement_l73_73681


namespace probability_red_on_other_side_l73_73406

def num_black_black_cards := 4
def num_black_red_cards := 2
def num_red_red_cards := 2

def num_red_sides_total := 
  num_black_black_cards * 0 +
  num_black_red_cards * 1 +
  num_red_red_cards * 2

def num_red_sides_with_red_on_other_side := 
  num_red_red_cards * 2

theorem probability_red_on_other_side :
  (num_red_sides_with_red_on_other_side : ℚ) / num_red_sides_total = 2 / 3 := by
  sorry

end probability_red_on_other_side_l73_73406


namespace dihedral_angle_cosine_l73_73735

noncomputable def dihedral_cosine (r : ℝ) (d : ℝ) (α : ℝ) : Prop :=
  d = 5 * r ∧ α = 60 ∧ cos (α * (Real.pi / 180)) * d / (2 * r) = 1.25

theorem dihedral_angle_cosine (r : ℝ) (θ : ℝ) (h : dihedral_cosine r (5 * r) 60) :
  cos θ = 0.04 :=
sorry

end dihedral_angle_cosine_l73_73735


namespace clubsuit_subtraction_l73_73186

def clubsuit (x y : ℕ) := 4 * x + 6 * y

theorem clubsuit_subtraction :
  (clubsuit 5 3) - (clubsuit 1 4) = 10 :=
by
  sorry

end clubsuit_subtraction_l73_73186


namespace find_a_l73_73130

theorem find_a (a : ℝ) (h : (2 + a * complex.I) / (1 + complex.I) = 3 + complex.I) : a = 4 :=
by
  sorry

end find_a_l73_73130


namespace angle_C_in_triangle_ABC_l73_73593

noncomputable def find_angle_C (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : Prop :=
  C = Real.pi / 6

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : find_angle_C A B C h1 h2 h3 :=
by
  -- proof omitted
  sorry

end angle_C_in_triangle_ABC_l73_73593


namespace remainder_3n_plus_2_l73_73950

-- Define the condition
def n_condition (n : ℤ) : Prop := n % 7 = 5

-- Define the theorem to be proved
theorem remainder_3n_plus_2 (n : ℤ) (h : n_condition n) : (3 * n + 2) % 7 = 3 := 
by sorry

end remainder_3n_plus_2_l73_73950


namespace pablo_distributed_fraction_l73_73220

-- Definitions based on the problem statement
def mia_coins (m : ℕ) := m
def sofia_coins (m : ℕ) := 3 * m
def pablo_coins (m : ℕ) := 12 * m

-- Condition for equal distribution
def target_coins (m : ℕ) := (mia_coins m + sofia_coins m + pablo_coins m) / 3

-- Needs for redistribution
def sofia_needs (m : ℕ) := target_coins m - sofia_coins m
def mia_needs (m : ℕ) := target_coins m - mia_coins m

-- Total distributed coins by Pablo
def total_distributed_by_pablo (m : ℕ) := sofia_needs m + mia_needs m

-- Fraction of coins Pablo distributes
noncomputable def fraction_distributed_by_pablo (m : ℕ) := (total_distributed_by_pablo m) / (pablo_coins m)

-- Theorem to prove
theorem pablo_distributed_fraction (m : ℕ) : fraction_distributed_by_pablo m = 5 / 9 := by
  sorry

end pablo_distributed_fraction_l73_73220


namespace find_m_for_tangent_l73_73079

noncomputable def parabola_hyperbola_tangent_value (m : ℝ) : Prop :=
  let y := λ x : ℝ, x^2 + 2*x + 3 in
  let hyperbola := λ y x : ℝ, y^2 - m*x^2 - 5 in
  ∃ (x : ℝ), hyperbola (y x) x = 0 ∧
             let p := (λ x : ℝ, x^4 + 4*x^3 + (10 - m)*x^2 + 12*x + 4) in
             ∃ (u : ℝ), p u = 0 ∧ p' u = 0

theorem find_m_for_tangent : parabola_hyperbola_tangent_value (-26) :=
  sorry

end find_m_for_tangent_l73_73079


namespace ac_not_9_ac_not_1_l73_73696

-- Variables for the distances
variables (AB BC AC : ℝ)

-- Conditions given in the problem
def conditions : Prop := (AB = 5) ∧ (BC = 3)

-- Prove that AC cannot be 9
theorem ac_not_9 (h : conditions AB BC) : AC ≠ 9 :=
by 
  intro h1
  have h2 : AC <= AB + BC := by sorry
  have h3 : 9 > AB + BC := by sorry
  contradiction

-- Prove that AC cannot be 1
theorem ac_not_1 (h : conditions AB BC) : AC ≠ 1 :=
by 
  intro h1
  have h2 : AB <= AC + BC := by sorry
  have h3 : 5 > AC + BC := by sorry
  contradiction

end ac_not_9_ac_not_1_l73_73696


namespace gdp_scientific_notation_l73_73295

theorem gdp_scientific_notation :
  ∃ (x y : ℝ), 86000 = x * 10 ^ y ∧ 1 ≤ x ∧ x < 10 ∧ x = 8.6 ∧ y = 4 :=
begin
  use [8.6, 4],
  sorry,
end

end gdp_scientific_notation_l73_73295


namespace polynomial_divisibility_property_l73_73505

theorem polynomial_divisibility_property (P : ℕ → ℤ) :
  (∀ n ≥ 1, ∀ (f : ℕ × ℕ × ℕ → ℕ), (∑ x in finset.range(n), ∑ y in finset.range(n), ∑ z in finset.range(n), f (x, y, z)) % P n = 0) →
  ∃ c k : ℤ, ∀ n : ℕ, P n = c * (n : ℤ) ^ k :=
sorry

end polynomial_divisibility_property_l73_73505


namespace quick_calc_formula_l73_73872

variables (a b A B C : ℤ)

theorem quick_calc_formula (h1 : (100 - a) * (100 - b) = (A + B - 100) * 100 + C)
                           (h2 : (100 + a) * (100 + b) = (A + B - 100) * 100 + C) :
  A = 100 ∨ A = 100 ∧ B = 100 ∨ B = 100 ∧ C = a * b :=
sorry

end quick_calc_formula_l73_73872


namespace range_of_a_for_square_root_meaningful_l73_73209

theorem range_of_a_for_square_root_meaningful (a : ℝ) : (sqrt (a - 2)).re ≈ sqrt (a - 2) → a ≥ 2 :=
sorry

end range_of_a_for_square_root_meaningful_l73_73209


namespace factor_expression_l73_73490

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) :=
by sorry

end factor_expression_l73_73490


namespace spherical_to_rectangular_l73_73120

noncomputable def rectangular_from_spherical (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * sin(phi) * cos(theta), rho * sin(phi) * sin(theta), rho * cos(phi))

theorem spherical_to_rectangular : 
  ∀ (rho theta phi : ℝ), 
  (-3, -4, 2) = rectangular_from_spherical rho theta phi → 
  (3, 4, 2) = rectangular_from_spherical rho (theta + Real.pi) phi :=
by
  intros rho theta phi h
  have h1 : -3 = rho * sin phi * cos theta := congr_fun (congr_fun (congr_arg Prod.fst h) rho) phi
  have h2 : -4 = rho * sin phi * sin theta := congr_fun (congr_fun (congr_arg (Prod.fst ∘ Prod.snd) h) rho) phi
  have h3 : 2 = rho * cos phi := congr_fun (congr_fun (congr_arg Prod.snd h) rho) phi
  specialize congr_fun (congr_fun rectangular_from_spherical rho (theta + Real.pi))
  rw [Real.cos_add_pi, Real.sin_add_pi]
  split
  { simp only [h1] }
  { split
    { simp only [h2] }
    { simp only [h3] }
  }
  sorry

end spherical_to_rectangular_l73_73120


namespace sum_of_smallest_and_largest_in_consecutive_integers_l73_73316

theorem sum_of_smallest_and_largest_in_consecutive_integers {m : ℕ} (hm : Even m) (z : ℤ) 
  (hmean : ∃ b : ℤ, (1 : ℤ) + 2 + 3 + ... + (m-1) = (m * z - ∑ k in range m, (b + k)) * m) : 
  (2 * z) = 2 * b + m - 1 :=
by
  sorry

end sum_of_smallest_and_largest_in_consecutive_integers_l73_73316


namespace mean_square_sum_l73_73317

theorem mean_square_sum (x y z : ℝ) 
  (h1 : x + y + z = 27)
  (h2 : x * y * z = 216)
  (h3 : x * y + y * z + z * x = 162) : 
  x^2 + y^2 + z^2 = 405 :=
by
  sorry

end mean_square_sum_l73_73317


namespace percent_of_value_l73_73744

theorem percent_of_value : (33 + 1 / 3) / 100 * 270 = 90 := by
  have h1 : (33 + 1 / 3) / 100 = 1 / 3 := by sorry
  rw [h1]
  exact (one_div 3) * 270 = 90

end percent_of_value_l73_73744


namespace relationship_among_a_b_c_l73_73878

noncomputable def a := 5 ^ Real.log 3.4 / Real.log 2
noncomputable def b := 5 ^ (Real.log 3.6 / 2) / Real.log 2
noncomputable def c := 5 ^ Real.log (10 / 3) / Real.log 2

theorem relationship_among_a_b_c : a > c ∧ c > b := 
by 
  have h1 : a = 5 ^ (Real.log 3.4 / Real.log 2) := rfl
  have h2 : b = 5 ^ ((1/2) * Real.log 3.6 / Real.log 2) := rfl
  have h3 : c = 5 ^ (Real.log (10 / 3) / Real.log 2) := by 
    calc c 
      = 5 ^ (- Real.log 0.3 / Real.log 2) : sorry
      ... = 5 ^ (Real.log (10 / 3) / Real.log 2) : sorry
  have h4 : 3.4 > (10 / 3) ∧ (10 / 3) > Real.sqrt 3.6 := sorry
  have h5 : Real.log 3.4 > Real.log (10 / 3) := sorry
  have h6 : Real.log (10 / 3) > (1 / 2) * Real.log 3.6 := sorry
  have h7 : 5 ^ (Real.log 3.4 / Real.log 2) > 5 ^ (Real.log (10 / 3) / Real.log 2) := by sorry
  have h8 : 5 ^ (Real.log (10 / 3) / Real.log 2) > 5 ^ ((1/2) * Real.log 3.6 / Real.log 2) := sorry
  exact ⟨h7, h8⟩

end relationship_among_a_b_c_l73_73878


namespace ken_house_distance_condition_l73_73265

noncomputable def ken_distance_to_dawn : ℕ := 4 -- This is the correct answer

theorem ken_house_distance_condition (K M : ℕ) (h1 : K = 2 * M) (h2 : K + M + M + K = 12) :
  K = ken_distance_to_dawn :=
  by
  sorry

end ken_house_distance_condition_l73_73265


namespace relationship_y1_y2_l73_73195

theorem relationship_y1_y2 :
  let f : ℝ → ℝ := λ x, 2 * x + 1 in
  let y1 := f (-3) in
  let y2 := f 4 in
  y1 < y2 :=
by {
  -- definitions
  let f := λ x, 2 * x + 1,
  let y1 := f (-3),
  let y2 := f 4,
  -- calculations
  have h1 : y1 = f (-3) := rfl,
  have h2 : y2 = f 4 := rfl,
  -- compare y1 and y2
  rw [h1, h2],
  exact calc
    y1 = f (-3) : rfl
    ... = 2 * (-3) + 1 : rfl
    ... = -5 : by norm_num
    ... < 2 * 4 + 1 : by norm_num
    ... = y2 : rfl
}

end relationship_y1_y2_l73_73195


namespace total_price_of_books_l73_73372

theorem total_price_of_books (total_books : ℕ) (math_books : ℕ) (cost_math_book : ℕ) (cost_history_book : ℕ) (remaining_books := total_books - math_books) (total_math_cost := math_books * cost_math_book) (total_history_cost := remaining_books * cost_history_book ) : total_books = 80 → math_books = 27 → cost_math_book = 4 → cost_history_book = 5 → total_math_cost + total_history_cost = 373 :=
by
  intros
  sorry

end total_price_of_books_l73_73372


namespace measure_of_A_l73_73598

noncomputable def ABC_Condition : Prop := ∀ (A B C a b c : ℝ),
  (a = Real.sqrt 7) ∧
  (b = 3) ∧
  (Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3) ∧
  (a / Real.sin A = b / Real.sin B)

theorem measure_of_A (A B C a b c : ℝ) (h : ABC_Condition A B C a b c) : A = Real.pi / 3 :=
by {
  sorry
}

end measure_of_A_l73_73598


namespace matrix_pow_2023_calc_l73_73466

open matrix

def matrix_pow_2023 := ![![1, 1], ![0, 1]]
def target_matrix := ![![1, 2023], ![0, 1]]

theorem matrix_pow_2023_calc : matrix_pow_2023 ^ 2023 = target_matrix := 
by sorry

end matrix_pow_2023_calc_l73_73466


namespace problem_trig_identity_l73_73548

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + sin x * cos x

theorem problem_trig_identity (θ : ℝ) (k : ℤ) :
  (∃ θ, ∀ x, f θ ≤ f x) →
  ( 2 * θ = 2 * k * real.pi - real.pi / 4) →
  ( (sin (2 * θ) + 2 * cos (2 * θ)) / (sin (2 * θ) - 2 * cos (2 * θ)) = -1 / 3) := by
  sorry

end problem_trig_identity_l73_73548


namespace lines_perpendicular_l73_73933

theorem lines_perpendicular (A1 B1 C1 A2 B2 C2 : ℝ) (h : A1 * A2 + B1 * B2 = 0) :
  ∃(x y : ℝ), A1 * x + B1 * y + C1 = 0 ∧ A2 * x + B2 * y + C2 = 0 → A1 * A2 + B1 * B2 = 0 :=
by
  sorry

end lines_perpendicular_l73_73933


namespace Exam_l73_73292

variables (Lewis : Type) (ReceivesA : Lewis → Prop) (AllQuestionsRight : Lewis → Prop)

theorem Exam (h : ∀ x : Lewis, AllQuestionsRight x → ReceivesA x) :
  (¬ ReceivesA Lewis) → (¬ AllQuestionsRight Lewis) :=
by 
  sorry

end Exam_l73_73292


namespace inequality_proof_l73_73282

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  (2 * a / (a^2 + b * c) + 2 * b / (b^2 + c * a) + 2 * c / (c^2 + a * b)) ≤ (a / (b * c) + b / (c * a) + c / (a * b)) := 
sorry

end inequality_proof_l73_73282


namespace sqrt_range_l73_73956

theorem sqrt_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_range_l73_73956


namespace president_vice_president_ways_l73_73971

theorem president_vice_president_ways : 
  let people := {A, B, C, D, E, F} in
  -- There are 6 people in the group
  let num_ways := (6 * 5) + 4 in
  num_ways = 34 :=
by
  let people := {A, B, C, D, E, F}
  have step1 : 6 * 5 = 30 := by norm_num,
  have step2 : 1 * 4 = 4 := by norm_num,
  have num_ways : (6 * 5) + 4 = 30 + 4 := by rw [step1, step2],
  rw [num_ways]
  exact eq.refl 34

end president_vice_president_ways_l73_73971


namespace largest_of_three_numbers_l73_73728

theorem largest_of_three_numbers (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : x * y + x * z + y * z = -6)
  (h3 : x * y * z = -8) : 
  max x (max y z) = (1 + real.sqrt 17) / 2 :=
by
  sorry

end largest_of_three_numbers_l73_73728


namespace minimum_value_of_expression_l73_73859

noncomputable def f (θ : ℝ) : ℝ := 4 * real.cos θ + 3 / real.sin θ + 2 * real.sqrt 2 * real.tan θ

theorem minimum_value_of_expression : ∃ θ : ℝ, 0 < θ ∧ θ < real.pi / 2 ∧ f θ = 6 * real.sqrt 3 * 2^(1/6) :=
by
  sorry

end minimum_value_of_expression_l73_73859


namespace conclusion_must_be_true_l73_73928

variables {a c x1 y1 x2 y2 : ℝ}

def parabola (x : ℝ) : ℝ := a * (x + 3)^2 + c

theorem conclusion_must_be_true
  (h1 : parabola x1 = y1)
  (h2 : parabola x2 = y2)
  (h3 : |x1 + 3| > |x2 + 3|) : a * (y1 - y2) > 0 :=
sorry

end conclusion_must_be_true_l73_73928


namespace find_k_l73_73652

noncomputable def g (x : ℕ) : ℤ := 2 * x^2 - 8 * x + 8

theorem find_k :
  (g 2 = 0) ∧ 
  (90 < g 9) ∧ (g 9 < 100) ∧
  (120 < g 10) ∧ (g 10 < 130) ∧
  ∃ (k : ℤ), 7000 * k < g 150 ∧ g 150 < 7000 * (k + 1)
  → ∃ (k : ℤ), k = 6 :=
by
  sorry

end find_k_l73_73652


namespace find_common_difference_l73_73128

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a1 a3 a5 a2 a4 a6 : ℤ)

noncomputable def is_arithmetic_sequence : Prop :=
  ∀ n m, (n:ℤ) > 0 → (m:ℤ) > 0 → a (n + m) = 2 * a (n / 2) → a (n + 1) = a n + d

axiom seq_def : ∀ n, a (n + 1) = a n + d

axiom H1 : a 1 + a 3 + a 5 = 105
axiom H2 : a 2 + a 4 + a 6 = 99

theorem find_common_difference : d = -2 :=
by
  sorry

end find_common_difference_l73_73128


namespace number_of_integers_becoming_one_after_10_operations_l73_73411

-- Define the operation on an integer n
def operation (n : ℕ) : ℕ :=
  if even n then n / 2 else n + 1

-- Define the sequence of operations
noncomputable def sequence (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then n else sequence (operation n) (k - 1)

-- Define the predicate that checks if an integer becomes 1 after exactly 10 operations
def becomes_one_after_10_operations (n : ℕ) : Prop :=
  sequence n 10 = 1

-- Define the set of integers that become 1 after exactly 10 operations
noncomputable def integers_becoming_one_after_10_operations : set ℕ :=
  {n | becomes_one_after_10_operations n}

-- The cardinality of this set should be 55
theorem number_of_integers_becoming_one_after_10_operations : 
  card integers_becoming_one_after_10_operations = 55 := 
sorry

end number_of_integers_becoming_one_after_10_operations_l73_73411


namespace arithmetic_sequence_sum_l73_73899

theorem arithmetic_sequence_sum 
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h_arith : ∀ n, a(n + 1) = a(n) + -2)
  (h_a1 : a 1 = 8)
  (h_sum : a 4 + a 6 = 0)
  (h_S : ∀ n, S n = (n * (2 * a 1 + (n - 1) * -2)) / 2) :
  S 8 = 8 :=
by
  sorry

end arithmetic_sequence_sum_l73_73899


namespace formatted_expression_bound_l73_73973

theorem formatted_expression_bound (n : ℕ) (h : n ≥ 3)
  (H : ∀ (s1 s2 : string), s1.length = n → s2.length = n → s1 ≠ s2 → (s1.to_list.filter_not (λ (x : char), s2.to_list.contains x)).length ≥ 3) : 
  ∃ (S' : finset string), S'.card ≤ (2^n / (n + 1)) := 
begin
  sorry
end

end formatted_expression_bound_l73_73973


namespace increasing_sequences_remainder_l73_73463

theorem increasing_sequences_remainder :
  (∃ b : Fin 12 → ℕ, (∀ i : Fin 12, i + 1 ≤ b i) ∧ (∀ i : Fin 12, b i ≤ 2013 ∧ (b i - (i + 1)) % 2 = 1)) ∧ nat.choose 1017 12 % 1000 = 17 :=
by
  sorry

end increasing_sequences_remainder_l73_73463


namespace cost_effective_inequality_l73_73041

def plan_x_cost (m : ℕ) : ℕ := 15 * m
def plan_y_cost (m : ℕ) : ℕ := 3000 + 7 * m

theorem cost_effective_inequality : ∀ (m : ℕ), plan_y_cost m < plan_x_cost m ↔ m ≥ 376 := by
  intro m
  unfold plan_x_cost
  unfold plan_y_cost
  simp
  rw [add_comm, add_lt_add_iff_left, nat.mul_lt_mul_left (by norm_num : 0 < 7)]
  simp
  sorry

end cost_effective_inequality_l73_73041


namespace monotonic_increase_interval_l73_73332

theorem monotonic_increase_interval (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = -x^2) : 
  {x : ℝ | ∃ y : ℝ, x ≤ 0 ∧ f(x) < f(y)} = {x : ℝ | x ∈ set.Iic 0} :=
by
  sorry

end monotonic_increase_interval_l73_73332


namespace find_a_range_l73_73918

theorem find_a_range:
  (∀ (x : ℝ), -2/3 ≤ x ∧ x ≤ 1 → (x + 1) * ln (x + 1) + a / (x + 1) ≥ 1) → a ≥ 1 := sorry

end find_a_range_l73_73918


namespace time_for_8_boys_4_girls_l73_73777

noncomputable def work_done_by_one_boy_in_one_day : ℚ := 1 / (16 * 6)
noncomputable def work_done_by_one_girl_in_one_day : ℚ := 1 / (24 * 6)

noncomputable def total_work_in_one_day (boys girls : ℚ) : ℚ :=
  (8 * work_done_by_one_boy_in_one_day) + (4 * work_done_by_one_girl_in_one_day)

theorem time_for_8_boys_4_girls :=
  have work_done_equivalence : work_done_by_one_boy_in_one_day = work_done_by_one_girl_in_one_day :=
    by sorry
  have total_work := total_work_in_one_day 8 4
  show ℚ, from
  if 16 * 6 = 1 then
    12
  else
    sorry

end time_for_8_boys_4_girls_l73_73777


namespace maximal_segments_plus_cross_points_three_more_than_small_rectangles_l73_73803

variable {Rectangle : Type}
variable (cross_point : Rectangle → Prop)
variable (maximal_segment : Rectangle → Prop)
variable (s n r : ℕ) -- s: number of maximal segments, n: number of cross points, r: number of small rectangles

def is_partitioned_into_small_rectangles (rect : Rectangle) : Prop := sorry

def is_cross_point (pt : Rectangle) : Prop := cross_point pt

def is_maximal_segment (seg : Rectangle) : Prop := maximal_segment seg

theorem maximal_segments_plus_cross_points_three_more_than_small_rectangles 
  (rect : Rectangle)
  (partitioned : is_partitioned_into_small_rectangles rect)
  (cross_points : ∀ pt, is_cross_point pt → pt ∈ rect)
  (max_segments : ∀ seg, is_maximal_segment seg → seg ∈ rect)
  (s_eq : s = ∑ seg in (rect.maximal_segments), 1)
  (n_eq : n = ∑ pt in (rect.cross_points), 1)
  (r_eq : r = ∑ subrect in (rect.small_rectangles), 1)
  :
  s + n = r + 3 := 
sorry

end maximal_segments_plus_cross_points_three_more_than_small_rectangles_l73_73803


namespace silverware_probability_l73_73019

noncomputable def num_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem silverware_probability:
  let forks := 4
  let spoons := 5
  let knives := 7
  let total_pieces := forks + spoons + knives
  let total_ways := num_ways_to_choose total_pieces 3
  let specific_selection_ways := forks * spoons * knives
  (specific_selection_ways : ℚ) / total_ways = 1 / 4 :=
by
  sorry

end silverware_probability_l73_73019


namespace travel_time_eq_l73_73413

theorem travel_time_eq {d v_b t : ℝ} (h1 : d = 50) (h2 : t = 1 / 6) (h3 : ∀ x, v_b = x ∧ v_t = 1.2 * x) :
  ∀ x, (50 / x = 50 / (1.2 * x) + (1 / 6)) :=
by 
  intros x
  have h_vb: v_b = x := sorry
  have h_vt: v_t = 1.2 * x := sorry
  rw [←h_vb, ←h_vt] at h1
  exact h1
  sorry

end travel_time_eq_l73_73413


namespace new_person_weight_l73_73688

theorem new_person_weight
  (W : ℝ) -- total weight of the original 10 persons
  (avg_increase : ∀ avg_weight : ℝ, (W + 40) / 10 = avg_weight + 4)
  (orig_person_weight : ℝ) -- original weight of the person being replaced
  (orig_person_weight_eq : orig_person_weight = 70) :
  ∃ new_weight : ℝ, new_weight = 110 :=
begin
  use 110,
  sorry
end

end new_person_weight_l73_73688


namespace length_of_AB_l73_73244

def quadrilateral_ABCD (A B C D : Type) : Prop :=
  ∠CBA = 90 ∧ ∠BAD = 45 ∧ ∠ADC = 105 ∧ BC = 1 + sqrt 2 ∧ AD = 2 + sqrt 6

theorem length_of_AB {A B C D : Type} (h : quadrilateral_ABCD A B C D) : 
  AB = 2 + sqrt 2 :=
by sorry

end length_of_AB_l73_73244


namespace class_has_24_students_l73_73721

theorem class_has_24_students (n S : ℕ) 
  (h1 : (S - 91 + 19) / n = 87)
  (h2 : S / n = 90) : 
  n = 24 :=
by sorry

end class_has_24_students_l73_73721


namespace find_z_approx_value_l73_73904

theorem find_z_approx_value :
  ∀ (x y z : ℝ), x = 100.48 → y = 100.70 → x * z = y^2 →
  z ≈ 10104.49 :=
by
  intros x y z hx hy h
  sorry

end find_z_approx_value_l73_73904


namespace concave_quadrilateral_area_l73_73617

noncomputable def area_of_concave_quadrilateral (AB BC CD AD : ℝ) (angle_BCD : ℝ) : ℝ :=
  let BD := Real.sqrt (BC * BC + CD * CD)
  let area_ABD := 0.5 * AB * BD
  let area_BCD := 0.5 * BC * CD
  area_ABD - area_BCD

theorem concave_quadrilateral_area :
  ∀ (AB BC CD AD : ℝ) (angle_BCD : ℝ),
    angle_BCD = Real.pi / 2 ∧ AB = 12 ∧ BC = 4 ∧ CD = 3 ∧ AD = 13 → 
    area_of_concave_quadrilateral AB BC CD AD angle_BCD = 24 :=
by
  intros AB BC CD AD angle_BCD h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end concave_quadrilateral_area_l73_73617


namespace unique_sum_of_three_squares_l73_73242

-- Defining perfect squares less than 100.
def perfect_squares : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

-- Predicate that checks if the sum of three perfect squares is equal to 100.
def is_sum_of_three_squares (a b c : ℕ) : Prop :=
  a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ c ∈ perfect_squares ∧ a + b + c = 100

-- The main theorem to be proved.
theorem unique_sum_of_three_squares :
  { (a, b, c) // is_sum_of_three_squares a b c }.to_finset.card = 1 :=
sorry -- Proof would go here.

end unique_sum_of_three_squares_l73_73242


namespace terminating_decimal_count_l73_73868

theorem terminating_decimal_count :
  ∃ n : ℕ, (n = 29) ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 508 ∧ (∃ m : ℕ, k = 17 * m) →
    let d := k / 425 in
    ∃ (a b : ℕ), b ≠ 0 ∧ (b % 2 = 0 ∨ b % 5 = 0) ∧ rat.mk k 425 = (a / b)) :=
sorry

end terminating_decimal_count_l73_73868


namespace magnitude_of_3a_plus_b_l73_73907

open Real -- use the Real namespace for trigonometric and vector operations

noncomputable def solve_magnitude (a b : ℝ) : ℝ :=
  let theta := π / 6
  let norm_a := sqrt(3)
  let norm_b := 1
  let dot_product := norm_a * norm_b * cos(theta)
  sqrt(9 * (norm_a ^ 2) + (norm_b ^ 2) + 6 * dot_product)

theorem magnitude_of_3a_plus_b (a b : ℝ) 
  (ha : |a| = sqrt(3))
  (hb : |b| = 1)
  (θ : Real.pi / 6 = AngleBetween a b) :
  (|3 * a + b| = sqrt(37)) :=
by
  sorry

end magnitude_of_3a_plus_b_l73_73907


namespace triangle_area_l73_73256

theorem triangle_area {A B C M : Type*} [triangle A B C] [segment AB AC AM] 
  (h1 : AB.length = 9) (h2 : AC.length = 17) (h3 : AM.length = 12) :
  area ABC = 40 * real.sqrt 2 :=
by
  sorry

end triangle_area_l73_73256


namespace magnitude_sum_l73_73898

open Real EuclideanGeometry

variables (a b : EuclideanSpace) 
variables (unit_vector_a : ∥a∥ = 1) (unit_vector_b : ∥b∥ = 1) 
variables (angle_60 : angle_between a b = π / 3)

theorem magnitude_sum : ∥ a + 2 • b ∥ = √ 7 := by
  sorry

end magnitude_sum_l73_73898


namespace inscribed_quadrilateral_perpendicular_l73_73621

/-- Given an inscribed quadrilateral ABCD with AB = AC and BC = CD,
    let P be the midpoint of the arc CD that does not contain A,
    and let Q be the intersection point of diagonals AC and BD.
    Prove that PQ is perpendicular to AB. -/
theorem inscribed_quadrilateral_perpendicular
    (A B C D P Q : Point)
    (h_inscribed : InscribedQuadrilateral A B C D)
    (h_eq1 : AB = AC)
    (h_eq2 : BC = CD)
    (h_midpoint_arc_P : IsMidpointOfArc P C D (circumcircle A B C D) A)
    (h_diagonal_intersect_Q : IsIntersectionPoint Q AC BD) :
    IsPerpendicular PQ AB :=
by
  sorry

end inscribed_quadrilateral_perpendicular_l73_73621


namespace compute_relation_l73_73270

-- Define variables and parameters
variables {A B C Q : Point}
variables (a b c R : ℝ)
noncomputable def H : Point := A + B + C
noncomputable def O : Point := circumcenter A B C
noncomputable def QA : ℝ := dist Q A
noncomputable def QB : ℝ := dist Q B
noncomputable def QC : ℝ := dist Q C
noncomputable def QH : ℝ := dist Q H
noncomputable def circumradius := R
noncomputable def side_lengths := (dist A B = a) (dist B C = b) (dist C A = c)

-- Given conditions from the problem
def problem_conditions : Prop :=
  is_orthocenter H A B C ∧
  ∀ A B C : Point, is_circumcenter O A B C ⟹ dist Q O = R ∧
  H = A + B + C ∧
  dist O A = R ∧ dist O B = R ∧ dist O C = R ∧
  dist_sq H = 9 * R^2 - (a^2 + b^2 + c^2)

-- The theorem statement
theorem compute_relation : problem_conditions →
  (QA^2 + QB^2 + QC^2 - QH^2 = a^2 + b^2 + c^2 - 7R^2) :=
sorry

end compute_relation_l73_73270


namespace simplify_expression_l73_73307

theorem simplify_expression (x : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 9 = 45 * x + 27 :=
by
  sorry

end simplify_expression_l73_73307


namespace integralCondition_l73_73284

open MeasureTheory

noncomputable def borelFunction (f : ℝ → ℝ) : Prop :=
  Measurable f

theorem integralCondition
  {f : ℝ → ℝ}
  (h_borel : borelFunction f)
  (h_cond : ∀ λ : ℕ, ∫ x in (Set.Ici 0), Real.exp (-λ * x) * f x ∂(MeasureTheory.measureSpace.volume) = 0) :
  ∀ᵐ x ∂(MeasureTheory.measureSpace.volume), 0 = f x :=
sorry

end integralCondition_l73_73284


namespace solve_for_x_l73_73308

theorem solve_for_x : ∃ (x : ℝ), (x = 160 + 64 * Real.sqrt 6) ∧ (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = Real.root 4 (2 + Real.sqrt x)) :=
by
  sorry

end solve_for_x_l73_73308


namespace smallest_k_sum_sequence_l73_73769

theorem smallest_k_sum_sequence (n : ℕ) (h : 100 = (n + 1) * (2 * 9 + n) / 2) : k = 9 := 
sorry

end smallest_k_sum_sequence_l73_73769


namespace binomial_20_19_l73_73830

theorem binomial_20_19 : nat.choose 20 19 = 20 :=
by {
  -- symmetric property of binomial coefficients and evaluation
  sorry
}

end binomial_20_19_l73_73830


namespace focus_of_shifted_parabola_l73_73841

theorem focus_of_shifted_parabola :
  let a : ℝ := 9
  let h : ℝ := 0
  let k : ℝ := -12
  focus_of_parabola (a, h, k) = (0, -(431 / 36)) := 
by
  sorry

end focus_of_shifted_parabola_l73_73841


namespace desired_percentage_butterfat_l73_73420

theorem desired_percentage_butterfat (cream_volume skim_volume : ℝ)
  (cream_butterfat skim_butterfat : ℝ)
  (h1 : cream_volume = 1)
  (h2 : cream_butterfat = 9.5)
  (h3 : skim_volume = 3)
  (h4 : skim_butterfat = 5.5) :
  let total_butterfat := (cream_volume * cream_butterfat / 100) + (skim_volume * skim_butterfat / 100)
  let total_volume := cream_volume + skim_volume
  let desired_percentage := (total_butterfat / total_volume) * 100
  in desired_percentage = 6.5 := 
by 
  sorry

end desired_percentage_butterfat_l73_73420


namespace frank_maze_time_l73_73871

theorem frank_maze_time :
  ∀ (t_current t_average_other n_other t_max_avg : ℕ),
  t_current = 45 →
  t_average_other = 50 →
  n_other = 4 →
  t_max_avg = 60 →
  t_max_avg * (n_other + 1) - (n_other * t_average_other + t_current) = 55 :=
by
  intros t_current t_average_other n_other t_max_avg h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  rfl

end frank_maze_time_l73_73871


namespace basketball_team_selection_l73_73782

theorem basketball_team_selection (quadruplets : Finset String) (players : Finset String) :
  quadruplets.card = 4 → players.card = 12 →
  (∃ q1 q2 q3 q4 ∈ quadruplets, q1 ≠ q2 ∧ q2 ≠ q3 ∧ q3 ≠ q4 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q4) →
  (∃ p1 p2 p3 p4 ∈ players, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4) →
  combinatorics.choose 4 3 * combinatorics.choose 12 4 = 1980 :=
begin
  assume h1 h2 h3 h4,
  rw [combinatorics.choose_eq, combinatorics.choose_eq],
  sorry
end

end basketball_team_selection_l73_73782


namespace distance_probability_l73_73271

/-- Given a square S with side length 2, the probability that the straight-line distance 
between two randomly chosen points on the sides of S is at least 1 is given by (d - eπ) / f 
with d, e, f being positive integers, where gcd(d, e, f) = 1. We want to prove 
that the solution gives d + e + f = 15. -/
theorem distance_probability (S : set (ℝ × ℝ)) (side_length : ℝ)
  (hS : S = {p | ∃ x y, (p = (x, y) ∧ x ≤ 2 ∧ y ≤ 2)}) :
  (∃ d e f : ℕ, (probability_distance_at_least_1 S = (d - e * Real.pi) / f) ∧
    Int.gcd d (Int.gcd e f) = 1 ∧ (d + e + f = 15)) :=
by {
  -- Definitions and assumptions
  let d := 7,
  let e := 0,
  let f := 8,
  -- Given the lengths and configurations match, this follows
  use [d, e, f],
  -- Provide necessary constraints
  split,
  sorry, -- Skipping proof details as instructed
  split,
  exact Int.gcd_coe_nat d (Int.gcd_coe_nat e f),
  exact eq.refl 15
}

noncomputable def probability_distance_at_least_1 (S : set (ℝ × ℝ)) : ℝ := 7 / 8 -- derived from solution details.

end distance_probability_l73_73271


namespace river_width_l73_73426

theorem river_width
  (depth : ℝ)
  (flow_rate_kmph : ℝ)
  (water_volume_per_minute : ℝ)
  (width : ℝ)
  (h_depth : depth = 4)
  (h_flow_rate_kmph : flow_rate_kmph = 6)
  (h_water_volume_per_minute : water_volume_per_minute = 26000):
  (width = water_volume_per_minute / ((flow_rate_kmph * 1000 / 60) * depth)) → width = 65 :=
by 
  intros h
  rw [h_depth, h_flow_rate_kmph, h_water_volume_per_minute] at h
  have h_flow_rate_m_per_min : (flow_rate_kmph * 1000 / 60) = 100,
    calc
      flow_rate_kmph * 1000 / 60
      = 6 * 1000 / 60 : by rw h_flow_rate_kmph
      = 6000 / 60 : by norm_num
      = 100 : by norm_num,
  rw [h_flow_rate_m_per_min] at h
  have h_width : width = 26000 / (100 * 4),
    calc
      26000 / (100 * 4)
      = 26000 / 400 : by norm_num
      = 65 : by norm_num,
  rwa h_width at h

end river_width_l73_73426


namespace greatest_distance_between_centers_l73_73732

-- Define the conditions
noncomputable def circle_radius : ℝ := 4
noncomputable def rectangle_length : ℝ := 20
noncomputable def rectangle_width : ℝ := 16

-- Define the centers of the circles
noncomputable def circle_center1 : ℝ × ℝ := (4, circle_radius)
noncomputable def circle_center2 : ℝ × ℝ := (rectangle_length - 4, circle_radius)

-- Calculate the greatest possible distance
noncomputable def distance : ℝ := Real.sqrt ((8 ^ 2) + (rectangle_width ^ 2))

-- Statement to prove
theorem greatest_distance_between_centers :
  distance = 8 * Real.sqrt 5 :=
  sorry

end greatest_distance_between_centers_l73_73732


namespace length_segment_aa_l73_73363

def reflection (x : ℝ) : ℝ := -x

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem length_segment_aa' :
  let A := (3 : ℝ, -4 : ℝ)
  let A' := (reflection A.1, A.2)
  distance A.1 A.2 A'.1 A'.2 = 6 :=
by
  simp [reflection, distance, real.sqrt]
  exact sorry

end length_segment_aa_l73_73363


namespace value_of_c_plus_d_l73_73942

noncomputable def g (c d : ℝ) : ℝ → ℝ := λ x, c * x + d
noncomputable def g_inv (c d : ℝ) : ℝ → ℝ := λ x, d * x + c

theorem value_of_c_plus_d (c d : ℝ) 
  (hg : ∀ x, g c d (g_inv c d x) = x)
  (hginv : ∀ x, g_inv c d (g c d x) = x) : 
  c + d = -2 :=
sorry

end value_of_c_plus_d_l73_73942


namespace minimum_area_of_triangle_is_sqrt_58_div_2_l73_73085

noncomputable def smallest_area_of_triangle (t s : ℝ) : ℝ :=
  (1/2) * Real.sqrt (5 * s^2 - 4 * s * t - 4 * s + 2 * t^2 + 10 * t + 13)

theorem minimum_area_of_triangle_is_sqrt_58_div_2 : ∃ t s : ℝ, smallest_area_of_triangle t s = Real.sqrt 58 / 2 := 
  by
  sorry

end minimum_area_of_triangle_is_sqrt_58_div_2_l73_73085


namespace second_number_is_46_l73_73340

theorem second_number_is_46 (sum_is_330 : ∃ (a b c d : ℕ), a + b + c + d = 330)
    (first_is_twice_second : ∀ (b : ℕ), ∃ (a : ℕ), a = 2 * b)
    (third_is_one_third_of_first : ∀ (a : ℕ), ∃ (c : ℕ), c = a / 3)
    (fourth_is_half_difference : ∀ (a b : ℕ), ∃ (d : ℕ), d = (a - b) / 2) :
  ∃ (b : ℕ), b = 46 :=
by
  -- Proof goes here, inserted for illustrating purposes only
  sorry

end second_number_is_46_l73_73340


namespace number_of_b_for_log_b_1024_l73_73176

theorem number_of_b_for_log_b_1024 :
  ∃ n, (n = 4) ∧ ∀ (b : ℕ), (b > 0) → (∃ k : ℕ, (k > 0) ∧ (log b 1024 = k) → (b = 2 ∨ b = 4 ∨ b = 32 ∨ b = 1024)) :=
sorry

end number_of_b_for_log_b_1024_l73_73176


namespace y1_lt_y2_l73_73190

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l73_73190


namespace transformed_sine_eq_neg_cos_l73_73310

theorem transformed_sine_eq_neg_cos (x : ℝ) :
  let f := fun x => sin (x - π / 3)
  let stretched_f := fun x => sin (1 / 2 * x - π / 3)
  let transformed_f := fun x => sin (1 / 2 * (x - π / 3) - π / 3)
  transformed_f x = -cos (x / 2) :=
by
  let f := fun x => sin (x - π / 3)
  let stretched_f := fun x => sin (1 / 2 * x - π / 3)
  let transformed_f := fun x => sin (1 / 2 * (x - π / 3) - π / 3)
  show transformed_f x = -cos (x / 2)
  sorry

end transformed_sine_eq_neg_cos_l73_73310


namespace negation_of_all_ge_0_l73_73707

-- Define the original proposition P
def P : Prop := ∀ x : ℝ, x^2 + x ≥ 0

-- Define the negation of P
def ¬P : Prop := ∃ x : ℝ, x^2 + x < 0

-- State that the negation of P is logically equivalent to ¬P
theorem negation_of_all_ge_0 :
  ¬(∀ x : ℝ, x^2 + x ≥ 0) ↔ ∃ x : ℝ, x^2 + x < 0 :=
sorry

end negation_of_all_ge_0_l73_73707


namespace simplify_fraction_l73_73325

theorem simplify_fraction (m : ℤ) : 
  let c := 2 
  let d := 4 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = (1 / 2 : ℚ) :=
by
  sorry

end simplify_fraction_l73_73325


namespace log_problem_part1_log_problem_part2_l73_73461

theorem log_problem_part1 : 2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) = 2 := 
  sorry

theorem log_problem_part2 : 0.027^(-1 / 3) - (-1 / 7)⁻¹ + (2 + 7 / 9) ^ (1/2) - (Real.sqrt 2 - 1) ^ 0 = 11 := 
  sorry

end log_problem_part1_log_problem_part2_l73_73461


namespace find_percentage_l73_73421

def problem_statement (n P : ℕ) := 
  n = (P / 100) * n + 84

theorem find_percentage : ∃ P, problem_statement 100 P ∧ (P = 16) :=
by
  sorry

end find_percentage_l73_73421


namespace number_of_incorrect_inferences_l73_73483

-- Definitions based on conditions
def prop1 : Prop := ∀ x : ℝ, x^2 - 3 * x + 2 = 0 → x = 1
def converse1 : Prop := ∀ x : ℝ, x ≠ 1 → x^2 - 3 * x + 2 ≠ 0

def prop2 : Prop := ∀ x : ℝ, x^2 = 1 → x = 1
def negation2 : Prop := ∀ x : ℝ, x^2 = 1 → x ≠ 1

def sufficient3 : Prop := ∀ x : ℝ, x < 1 → x^2 - 3 * x + 2 > 0

def prop4 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def negation4 : Prop := ∀ x : ℝ, x^2 + x + 1 < 0

-- Prove the number of incorrect inferences
theorem number_of_incorrect_inferences : nat :=
if h1 : converse1 ↔ true then if h2 : negation2 ↔ false then if h3 : sufficient3 ↔ true then if h4 : negation4 ↔ false then 2 else 3 else 2 else 1 else sorry

end number_of_incorrect_inferences_l73_73483


namespace probability_woman_lawyer_or_man_doctor_l73_73221

theorem probability_woman_lawyer_or_man_doctor :
  let total_members := 1
  let prob_women := 0.65
  let prob_men := 0.35
  let prob_woman_lawyer_given_woman := 0.40
  let prob_woman_doctor_given_woman := 0.30
  let prob_woman_engineer_given_woman := 0.30
  let prob_man_lawyer_given_man := 0.50
  let prob_man_doctor_given_man := 0.20
  let prob_man_engineer_given_man := 0.30
  let prob_woman_lawyer := prob_women * prob_woman_lawyer_given_woman
  let prob_man_doctor := prob_men * prob_man_doctor_given_man
in prob_woman_lawyer + prob_man_doctor = 0.33 :=
by
  sorry

end probability_woman_lawyer_or_man_doctor_l73_73221


namespace even_member_count_can_divide_into_two_neutral_groups_l73_73601

structure Club :=
  (members : Type)
  (friend : members → members)
  (enemy : members → members)
  (each_has_one_friend_and_enemy : ∀ m : members, (friend m ≠ m) ∧ (enemy m ≠ m) ∧ (friend m ≠ enemy m))

-- Part (a): Prove that the number of members is even.
theorem even_member_count (C : Club) : 
  ∃ n : Nat, C.members → List.length ∧ n % 2 = 0 :=
sorry

-- Part (b): Prove that the club can be divided into two neutral groups.
theorem can_divide_into_two_neutral_groups (C : Club) : 
  ∃ (G1 G2 : Set C.members),
  (∀ m : C.members, (m ∈ G1 ∨ m ∈ G2) ∧ ¬(m ∈ G1 ∧ m ∈ G2)) ∧
  (∀ m₁ m₂ : C.members, (m₁ ∈ G1 → m₂ ∈ G1 → ¬(C.friend m₁ = m₂ ∨ C.enemy m₁ = m₂)) ∧
                              (m₁ ∈ G2 → m₂ ∈ G2 → ¬(C.friend m₁ = m₂ ∨ C.enemy m₁ = m₂))) :=
sorry

end even_member_count_can_divide_into_two_neutral_groups_l73_73601


namespace inequality_proof_l73_73502

theorem inequality_proof (α1 α2 α3 : ℝ) (ν : ℝ) 
  (h₀ : 0 < α1) (h₁ : 0 < α2) (h₂ : 0 < α3) (hν : 0 < ν) :
  ∑ i in {1, 2, 3}, 
    (if i = 1 then 1 / (2 * ν * α1 + α2 + α3) 
     else if i = 2 then 1 / (2 * ν * α2 + α1 + α3) 
     else 1 / (2 * ν * α3 + α1 + α2)) > 
  (2 * ν / (2 * ν + 1)) * 
    (∑ i in {1, 2, 3}, 
      (if i = 1 then 1 / (ν * α1 + ν * α2 + α3) 
       else if i = 2 then 1 / (ν * α2 + ν * α3 + α1) 
       else 1 / (ν * α3 + ν * α1 + α2))) := 
by
  sorry

end inequality_proof_l73_73502


namespace opposite_face_of_A_is_F_l73_73033

theorem opposite_face_of_A_is_F : 
  let faces := ["A", "B", "C", "D", "E", "F"],
  let adjacent_to_A := ["B", "C", "D", "E"],
  ∃ opposite_face : String, opposite_face ∉ adjacent_to_A ∧ opposite_face = "F" :=
by {
  let faces := ["A", "B", "C", "D", "E", "F"],
  let adjacent_to_A := ["B", "C", "D", "E"],
  use "F",
  split,
  {
    intro h,
    simp at h,
  },
  refl,
}

end opposite_face_of_A_is_F_l73_73033


namespace lateral_surface_area_of_parallelepiped_l73_73689

-- Define the conditions
variables (a h Q : ℝ)
variable (is_rhombus_base : True)
variable (area_of_cross_section : areas Q)
variable (angle_with_plane : angle 45)

-- Statement of the problem in Lean 4.
theorem lateral_surface_area_of_parallelepiped 
  (h : ℝ) (a : ℝ) (Q : ℝ) (H_base_rhomb : is_rhombus_base) 
  (H_angle : angle_with_plane) 
  (H_area_cross_section : area_of_cross_section) : 
  lateral_surface_area (2 * sqrt 2 * Q) :=
sorry

end lateral_surface_area_of_parallelepiped_l73_73689


namespace cost_of_whitewashing_l73_73396

-- Definitions of the dimensions
def length_room : ℝ := 25.0
def width_room : ℝ := 15.0
def height_room : ℝ := 12.0

def dimensions_door : (ℝ × ℝ) := (6.0, 3.0)
def dimensions_window : (ℝ × ℝ) := (4.0, 3.0)
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 6.0

-- Definition of areas and costs
def area_wall (a b : ℝ) : ℝ := 2 * (a * b)
def area_door : ℝ := (dimensions_door.1 * dimensions_door.2)
def area_window : ℝ := (dimensions_window.1 * dimensions_window.2) * (num_windows)
def total_area_walls : ℝ := (area_wall length_room height_room) + (area_wall width_room height_room)
def area_to_paint : ℝ := total_area_walls - (area_door + area_window)
def total_cost : ℝ := area_to_paint * cost_per_sqft

-- Proof statement
theorem cost_of_whitewashing : total_cost = 5436 := by
  sorry

end cost_of_whitewashing_l73_73396


namespace area_of_pentagon_l73_73457

variables (m : ℕ)

def area_of_quadrilateral (A B C D : ℕ) : ℕ := 23

def distance_adjacent_points_horizontal_vertical (m : ℕ) : Prop := true

theorem area_of_pentagon
  (m : ℕ)
  (h1 : distance_adjacent_points_horizontal_vertical m)
  (h2 : area_of_quadrilateral 1 2 3 4 = 23) :
  ∃ area : ℕ, area = 28 :=
by
  use 28
  sorry

end area_of_pentagon_l73_73457


namespace ratio_two_to_three_nights_ago_l73_73849

def question (x : ℕ) (k : ℕ) : (ℕ × ℕ) := (x, k)

def pages_three_nights_ago := 15
def additional_pages_last_night (x : ℕ) := x + 5
def total_pages := 100
def pages_tonight := 20

theorem ratio_two_to_three_nights_ago :
  ∃ (x : ℕ), 
    (x + additional_pages_last_night x = total_pages - (pages_three_nights_ago + pages_tonight)) 
    ∧ (x / pages_three_nights_ago = 2 / 1) :=
by
  sorry

end ratio_two_to_three_nights_ago_l73_73849


namespace number_of_integers_becoming_one_after_10_operations_l73_73412

-- Define the operation on an integer n
def operation (n : ℕ) : ℕ :=
  if even n then n / 2 else n + 1

-- Define the sequence of operations
noncomputable def sequence (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then n else sequence (operation n) (k - 1)

-- Define the predicate that checks if an integer becomes 1 after exactly 10 operations
def becomes_one_after_10_operations (n : ℕ) : Prop :=
  sequence n 10 = 1

-- Define the set of integers that become 1 after exactly 10 operations
noncomputable def integers_becoming_one_after_10_operations : set ℕ :=
  {n | becomes_one_after_10_operations n}

-- The cardinality of this set should be 55
theorem number_of_integers_becoming_one_after_10_operations : 
  card integers_becoming_one_after_10_operations = 55 := 
sorry

end number_of_integers_becoming_one_after_10_operations_l73_73412


namespace min_balls_in_3x3_grid_l73_73435

/-- Given a 3x3 grid, where the number of balls in each cell is represented by a non-negative integer, 
    the minimum number of balls required to ensure each row and each column contains a different number 
    of balls is 8. -/
theorem min_balls_in_3x3_grid : 
  ∃ (balls : Fin 3 → Fin 3 → ℕ), 
    (∀ i j, i ≠ j → (∑ k, balls i k) ≠ (∑ k, balls j k)) ∧
    (∀ i j, i ≠ j → (∑ k, balls k i) ≠ (∑ k, balls k j)) ∧
    (∑ i j, balls i j >= 8) := sorry

end min_balls_in_3x3_grid_l73_73435


namespace quadrilateral_perimeter_correct_l73_73613

noncomputable def quadrilateral_perimeter : ℝ :=
  let AB := 15
  let BC := 20
  let CD := 9
  let AC := Real.sqrt (AB^2 + BC^2)
  let AD := Real.sqrt (AC^2 + CD^2)
  AB + BC + CD + AD

theorem quadrilateral_perimeter_correct :
  quadrilateral_perimeter = 44 + Real.sqrt 706 := by
  sorry

end quadrilateral_perimeter_correct_l73_73613


namespace profit_per_unit_minimum_units_A_l73_73409

-- Define the conditions
def price_A : ℕ := 120_000
def price_B : ℕ := 150_000
def profit_from_2A_5B : ℕ := 31_000
def profit_from_1A_2B : ℕ := 13_000
def total_units : ℕ := 22
def total_budget : ℕ := 3_000_000

-- Part 1: Define and prove the profit per unit of models A and B
theorem profit_per_unit (x y : ℕ) (h1 : 2 * x + 5 * y = profit_from_2A_5B) (h2 : x + 2 * y = profit_from_1A_2B) :
  x = 3_000 ∧ y = 5_000 := sorry

-- Part 2: Define and prove the minimum units of model A to be purchased
theorem minimum_units_A (m : ℕ) (h : 120_000 * m + 150_000 * (22 - m) ≤ 3_000_000) : m ≥ 10 := sorry

end profit_per_unit_minimum_units_A_l73_73409


namespace intersection_of_curves_l73_73975
noncomputable theory

-- Definitions based on the given conditions
def l1_x (t : ℝ) := 2 - t
def l1_y (k t : ℝ) := k * t

def l2_x (m : ℝ) := -2 + m
def l2_y (k m : ℝ) := m / k

def C1_cartesian_eq (x y : ℝ) := x^2 + y^2 = 4

def C2_polar_eq (θ : ℝ) := ρ = 4 * Real.sin θ

-- Theorem to state and prove the required properties
theorem intersection_of_curves (k : ℝ) (ρ θ x y : ℝ) :
  (C1_cartesian_eq x y → C2_polar_eq θ → ρ = 2 ∧ (θ = π / 6 ∨ θ = 5 * π / 6)) := sorry

end intersection_of_curves_l73_73975


namespace cost_of_all_entries_l73_73806

-- Defining the problem conditions
def num_sequences_A := 8   -- Ways to choose 3 consecutive numbers from 01 to 10
def num_sequences_B := 9   -- Ways to choose 2 consecutive numbers from 11 to 20
def num_sequences_C := 10  -- Ways to choose 1 number from 21 to 30
def num_sequences_D := 6   -- Ways to choose 1 number from 31 to 36
def cost_per_entry := 2    -- Cost of each entry in yuan

-- Define the total number of entries
def total_entries := num_sequences_A * num_sequences_B * num_sequences_C * num_sequences_D

-- Define the total cost to purchase all possible entries
def total_cost := total_entries * cost_per_entry

-- Statement of the problem in Lean
theorem cost_of_all_entries : total_cost = 8640 :=
by
  unfold total_cost total_entries num_sequences_A num_sequences_B num_sequences_C num_sequences_D cost_per_entry
  calc
    8 * 9 * 10 * 6 * 2 = 4320 * 2 : by ring
    ... = 8640 : by norm_num

end cost_of_all_entries_l73_73806


namespace count_ordered_pairs_l73_73106

theorem count_ordered_pairs :
  let positive_integers := {n : ℕ // n > 0},
      valid_pairs := {(b, c) ∈ positive_integers × positive_integers | b^2 - 4 * c ≤ 0 ∧ c^2 - 4 * b ≤ 0}
  in (set.finite valid_pairs ∧ set.card valid_pairs = 6) := 
by
  sorry

end count_ordered_pairs_l73_73106


namespace problem_statement_l73_73115

variable {n : ℕ} {x : Fin n → ℝ}

theorem problem_statement (h1 : n ≥ 2) 
  (h2 : ∑ i, |x i| = 1)
  (h3 : ∑ i, x i = 0) :
  |(∑ i in Finset.range n, x i / (i + 1))| ≤ 1/2 - 1/2^n := by
  sorry

end problem_statement_l73_73115


namespace truck_travel_distance_l73_73445

theorem truck_travel_distance (miles_per_5gallons miles distance gallons rate : ℕ)
  (h1 : miles_per_5gallons = 150) 
  (h2 : gallons = 5) 
  (h3 : rate = miles_per_5gallons / gallons) 
  (h4 : gallons = 7) 
  (h5 : distance = rate * gallons) : 
  distance = 210 := 
by sorry

end truck_travel_distance_l73_73445


namespace scallops_per_pound_l73_73410

theorem scallops_per_pound
  (cost_per_pound : ℝ)
  (scallops_per_person : ℕ)
  (number_of_people : ℕ)
  (total_cost : ℝ)
  (total_scallops : ℕ)
  (total_pounds : ℝ)
  (scallops_per_pound : ℕ)
  (h1 : cost_per_pound = 24)
  (h2 : scallops_per_person = 2)
  (h3 : number_of_people = 8)
  (h4 : total_cost = 48)
  (h5 : total_scallops = scallops_per_person * number_of_people)
  (h6 : total_pounds = total_cost / cost_per_pound)
  (h7 : scallops_per_pound = total_scallops / total_pounds) : 
  scallops_per_pound = 8 :=
sorry

end scallops_per_pound_l73_73410


namespace B_contribution_l73_73404

theorem B_contribution (A_capital : ℝ) (A_time : ℝ) (B_time : ℝ) (total_profit : ℝ) (A_profit_share : ℝ) (B_contributed : ℝ) :
  A_capital * A_time / (A_capital * A_time + B_contributed * B_time) = A_profit_share / total_profit →
  B_contributed = 6000 :=
by
  intro h
  sorry

end B_contribution_l73_73404


namespace non_intersecting_segments_exists_l73_73515

theorem non_intersecting_segments_exists (n : ℕ) (h1: 1 ≤ n) (points : Fin 2n → Point)
  (h_nocollinear : ∀ {p1 p2 p3 : Fin 2n}, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear points p1 p2 p3)
  (coloring : Fin 2n → Color) (h_blue : ∃f : Fin n → Fin 2n, ∀ i j : Fin n, i ≠ j → f i ≠ f j ∧ coloring (f i) = Color.blue)
  (h_red : ∃f : Fin n → Fin 2n, ∀ i j : Fin n, i ≠ j → f i ≠ f j ∧ coloring (f i) = Color.red) :
  ∃ S : Fin n → (Fin 2n × Fin 2n), (∀ i, coloring (S i).1 = Color.blue ∧ coloring (S i).2 = Color.red ∧ 
    (∀ j ≠ i, ¬ intersect (S i) (S j)) ∧ ∀ p ∈ Finset.image (λ x : Fin n, (S x).1) (Finset.univ) ∪ Finset.image (λ x : Fin n, (S x).2) (Finset.univ), 
     ∃ i : Fin n, p = (S i).1 ∨ p = (S i).2) :=
sorry

end non_intersecting_segments_exists_l73_73515


namespace paul_runs_41_miles_l73_73389

-- Conditions as Definitions
def movie1_length : ℕ := (1 * 60) + 36
def movie2_length : ℕ := (2 * 60) + 18
def movie3_length : ℕ := (1 * 60) + 48
def movie4_length : ℕ := (2 * 60) + 30
def total_watch_time : ℕ := movie1_length + movie2_length + movie3_length + movie4_length
def time_per_mile : ℕ := 12

-- Proof Statement
theorem paul_runs_41_miles : total_watch_time / time_per_mile = 41 :=
by
  -- Proof would be provided here
  sorry 

end paul_runs_41_miles_l73_73389


namespace fold_triangle_area_l73_73680

theorem fold_triangle_area (A B C : Point) (h : A ≠ B ∧ B ≠ C ∧ C ≠ A) 
(h_bc_ca_ab : dist B C ≤ dist C A ∧ dist C A ≤ dist A B) : 
  ∃ (P : Point), 
  let folded_area := fold_triangle_along_axis (mk_triangle A B C) P in 
  folded_area ≤ (3 / 5) * (area (mk_triangle A B C)) :=
by sorry

end fold_triangle_area_l73_73680


namespace total_population_l73_73779

theorem total_population (P : ℝ) : 0.96 * P = 23040 → P = 24000 :=
by
  sorry

end total_population_l73_73779


namespace fisherman_can_leave_l73_73399

variables {F : Fin 6 → ℕ}

-- Each fisherman catches a different number of fish
def different_counts (F : Fin 6 → ℕ) := ∀ i j : Fin 6, i ≠ j → F i ≠ F j

-- The total number of fish caught is 100
def total_fish (F : Fin 6 → ℕ) := (Finset.univ : Finset (Fin 6)).sum F = 100

-- Any fisherman can distribute his fish such that the remaining five end up with an equal number:
def distributable (F : Fin 6 → ℕ) :=
  ∀ i : Fin 6, ∃ N : ℕ, (Finset.univ.erase i).sum (λ j, F j) + F i = N * 5

-- The proof goal
theorem fisherman_can_leave (F : Fin 6 → ℕ) 
  (h_diff : different_counts F) 
  (h_total : total_fish F) 
  (h_dist : distributable F) :
  ∃ i : Fin 6, F i = 20 ∧ let G := λ j : Fin 6, if j = i then 0 else F j 
                                                in distributable G := 
sorry

end fisherman_can_leave_l73_73399


namespace truck_travel_distance_l73_73444

theorem truck_travel_distance (miles_per_5gallons miles distance gallons rate : ℕ)
  (h1 : miles_per_5gallons = 150) 
  (h2 : gallons = 5) 
  (h3 : rate = miles_per_5gallons / gallons) 
  (h4 : gallons = 7) 
  (h5 : distance = rate * gallons) : 
  distance = 210 := 
by sorry

end truck_travel_distance_l73_73444


namespace problem_statement_l73_73851

def from_base (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d, acc * b + d) 0

theorem problem_statement : 
  (from_base 10 8 [2, 1, 3] / from_base 10 3 [1, 2] + 
  from_base 10 5 [2, 3, 4] / from_base 10 4 [3, 2]) = 31 + 9 / 14 := 
sorry

end problem_statement_l73_73851


namespace xy_is_perfect_cube_l73_73795

theorem xy_is_perfect_cube (x y : ℕ) (h₁ : x = 5 * 2^4 * 3^3) (h₂ : y = 2^2 * 5^2) : ∃ z : ℕ, (x * y) = z^3 :=
by
  sorry

end xy_is_perfect_cube_l73_73795


namespace expected_value_sum_marbles_l73_73181

theorem expected_value_sum_marbles :
  let S := {1, 2, 3, 4, 5, 6, 7} in
  let valid_pairs := { (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), 
                       (7, 2), (7, 3), (7, 4), (7, 5), (7, 6) } in
  let sums := { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 } in
  (∑ sum in sums, sum) / (size valid_pairs) = 8 :=
sorry

end expected_value_sum_marbles_l73_73181


namespace area_trapezoid_AXYD_l73_73392

theorem area_trapezoid_AXYD (abcd_sqr : square abcd) (side_ab : ab = real.sqrt 2016)
  (x_on_ab : point x ab) (y_on_cd : point y cd) (equal_segments : ax = cy) :
  area_trapezoid_AXYD abcd x y = 2016 :=
sorry

end area_trapezoid_AXYD_l73_73392


namespace sin_cos_identity_l73_73346

-- Definition of the given angles
def angle_20 := 20.0 * real.pi / 180.0
def angle_40 := 40.0 * real.pi / 180.0

-- Definition of pi
def pi := real.pi

-- Using the provided angle addition formula
lemma sin_addition_formula (a b : ℝ) : real.sin (a + b) = real.sin a * real.cos b + real.cos a * real.sin b := 
real.sin_add a b

-- The main statement
theorem sin_cos_identity : real.sin angle_20 * real.cos angle_40 + real.cos angle_20 * real.sin angle_40 = real.sqrt 3 / 2 :=
by 
    rw [←sin_addition_formula angle_20 angle_40]
    -- Simplifying sin (60 degrees)
    have : angle_20 + angle_40 = 60.0 * real.pi / 180.0,
    { simp [angle_20, angle_40, pi],
      norm_num, },
    rw [this],
    norm_num,
    sorry

end sin_cos_identity_l73_73346


namespace Marks_3_pointers_l73_73667

variable (x : ℕ)
hypothesis (h1 : 25 * 2 + 3 * x + 10 + (2 * (25 * 2) + 1.5 * x + 10 / 2) = 201)

theorem Marks_3_pointers : x = 8 :=
by
  sorry

end Marks_3_pointers_l73_73667


namespace part_one_part_two_part_three_l73_73526

-- Define the sequence and the sum of its first n terms
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2 ^ n

-- Prove that a_1 = 2 and a_4 = 40
theorem part_one (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  a 1 = 2 ∧ a 4 = 40 := by
  sorry
  
-- Prove that the sequence {a_{n+1} - 2a_n} is a geometric sequence
theorem part_two (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∃ r : ℕ, (r = 2) ∧ (∀ n, (a (n + 1) - 2 * a n) = r ^ n) := by
  sorry

-- Prove the general term formula for the sequence {a_n}
theorem part_three (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∀ n, a n = 2 ^ (n + 1) - 2 := by
  sorry

end part_one_part_two_part_three_l73_73526


namespace at_least_one_six_in_two_dice_l73_73370

def total_outcomes (dice : ℕ) (sides : ℕ) : ℕ := sides ^ dice
def non_six_outcomes (dice : ℕ) (sides : ℕ) : ℕ := (sides - 1) ^ dice
def at_least_one_six_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  let all := total_outcomes dice sides
  let none := non_six_outcomes dice sides
  (all - none) / all

theorem at_least_one_six_in_two_dice :
  at_least_one_six_probability 2 6 = 11 / 36 :=
by
  sorry

end at_least_one_six_in_two_dice_l73_73370


namespace perpendicular_bisector_eq_line_through_P_eq_l73_73774

-- Definitions for Problem 1
def point (α : Type) := (α × α)
def B : point ℝ := (-4, 0)
def C : point ℝ := (-2, 4)
def midpoint (p1 p2 : point ℝ) : point ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def line (slope intercept : ℝ) (x y : ℝ) := y = slope * x + intercept

-- Conditions for Problem 1
def slope_BC := 2
def midpoint_BC := midpoint B C
def slope_EF := -1 / 2

-- Proof Problem 1 Statement
theorem perpendicular_bisector_eq :
  line slope_EF (midpoint_BC.2 - slope_EF * midpoint_BC.1) = (fun x y => x + 2*y - 1 = 0) := 
sorry

-- Definitions for Problem 2
def P : point ℝ := (-1, -2)
def A (a : ℝ) := (a, 0)
def B' (b : ℝ) := (0, b)

-- Conditions for Problem 2
def midpoint_AB' (a b : ℝ) := midpoint (A a) (B' b)
def midpoint_P := P
def solved_a := -2
def solved_b := -4

-- Proof Problem 2 Statement
theorem line_through_P_eq :
  (fun x y => (x / solved_a) + (y / solved_b) = 1) = (fun x y => 2*x + y + 4 = 0) :=
sorry

end perpendicular_bisector_eq_line_through_P_eq_l73_73774


namespace students_taking_both_chorus_and_band_l73_73966

theorem students_taking_both_chorus_and_band (total_students : ℕ) 
                                             (chorus_students : ℕ)
                                             (band_students : ℕ)
                                             (not_enrolled_students : ℕ) : 
                                             total_students = 50 ∧
                                             chorus_students = 18 ∧
                                             band_students = 26 ∧
                                             not_enrolled_students = 8 →
                                             ∃ (both_chorus_and_band : ℕ), both_chorus_and_band = 2 :=
by
  intros h
  sorry

end students_taking_both_chorus_and_band_l73_73966


namespace trapezoid_area_correct_l73_73206

-- Define necessary constants and assumptions
def base_angle : ℝ := 45
def leg_length : ℝ := 1
def upper_base_length : ℝ := 1

-- Define the function to calculate the area of the trapezoid
def trapezoid_area (b1 b2 h : ℝ) : ℝ := 0.5 * (b1 + b2) * h

-- Define the lower base length using the given conditions
def lower_base_length : ℝ := 1 + Real.sqrt 2

-- Given:
-- 1. base_angle = 45 degrees (which implies the height calculation)
-- 2. leg_length = 1
-- 3. upper_base_length = 1

theorem trapezoid_area_correct :
  trapezoid_area upper_base_length lower_base_length 2 = 2 + Real.sqrt 2 := 
by sorry

end trapezoid_area_correct_l73_73206


namespace total_cost_pencils_and_pens_l73_73428

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end total_cost_pencils_and_pens_l73_73428


namespace integral_equals_result_l73_73765

noncomputable def integral_result : ℝ :=
  ∫ x in -1..1, x^2 * exp (-x / 2)

theorem integral_equals_result : integral_result = 10 * sqrt real.exp 1 - 26 / sqrt real.exp 1 := 
sorry

end integral_equals_result_l73_73765


namespace greatest_prime_factor_of_evens_product_sum_l73_73211

noncomputable def evens_product (x : ℕ) : ℕ :=
  if h : x % 2 = 0 then List.prod (List.filter (λ n, n % 2 = 0) (List.range (x + 1)))
  else 1

theorem greatest_prime_factor_of_evens_product_sum :
  let p := evens_product 12
  let q := evens_product 10
  let sum := p + q
  Nat.greatestPrimeDivisor sum = 7 :=
by
  let p := evens_product 12
  let q := evens_product 10
  let sum := p + q
  have : Nat.greatestPrimeDivisor sum = 7 := sorry
  exact this

end greatest_prime_factor_of_evens_product_sum_l73_73211


namespace value_of_n_l73_73585

noncomputable def f (x : ℝ) (n : ℝ) := x^n + 3^x

theorem value_of_n (n : ℝ) (h1 : f 1 n = 4) (h2 : deriv (λ x, f x n) 1 = 3 + 3 * Real.log 3) : n = 3 :=
sorry

end value_of_n_l73_73585


namespace number_not_always_greater_l73_73797

-- Define the condition
def num_property (n : ℝ) : Prop := 1.25 * n > n

-- Define the theorem with conditions
theorem number_not_always_greater : ∃ (n : ℝ), ¬(num_property n) := by
  use 0
  simp [num_property]
  exact le_refl 0

end number_not_always_greater_l73_73797


namespace elder_son_age_l73_73478

variable (elder_young_diff current_younger_age future_younger_age future_year_difference : ℕ)

-- Conditions
axiom younger_age_future_30_years : future_year_difference = 30
axiom younger_will_be_60 : future_younger_age = 60
axiom younger_age_difference : current_younger_age + future_year_difference = future_younger_age
axiom age_difference : current_younger_age + elder_young_diff = elder_years

-- Proof that the elder son is currently 40 years old
theorem elder_son_age
  (future_year_difference = 30) 
  (future_younger_age = 60) 
  (current_younger_age + future_year_difference = future_younger_age) 
  (current_younger_age + elder_young_diff = elder_years) 
  (elder_young_diff = 10) 
  (current_younger_age = 30) 
  : elder_years = 40 := 
by
  sorry

end elder_son_age_l73_73478


namespace find_expression_value_l73_73536

theorem find_expression_value 
  (m : ℝ) 
  (hroot : m^2 - 3 * m + 1 = 0) : 
  (m - 3)^2 + (m + 2) * (m - 2) = 3 := 
sorry

end find_expression_value_l73_73536


namespace max_right_angle_triangles_in_pyramid_l73_73604

noncomputable def pyramid_max_right_angle_triangles : Nat :=
  let pyramid : Type := { faces : Nat // faces = 4 }
  1

theorem max_right_angle_triangles_in_pyramid (p : pyramid) : pyramid_max_right_angle_triangles = 1 :=
  sorry

end max_right_angle_triangles_in_pyramid_l73_73604


namespace area_of_isosceles_triangle_l73_73364

theorem area_of_isosceles_triangle (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R]
  (PQ PR QR : ℝ) (h₁ : PQ = 26) (h₂ : PR = 26) (h₃ : QR = 30) :
  ∃ A : ℝ, A = 15 * real.sqrt 451 := sorry

end area_of_isosceles_triangle_l73_73364


namespace roots_quadratic_equation_l73_73582

theorem roots_quadratic_equation (a b : ℝ) (h₁ : a + b = 16) (h₂ : a * b = 100) : 
    (Polynomial.X ^ 2 - Polynomial.C 16 * Polynomial.X + Polynomial.C 100 = 0) :=
sorry

end roots_quadratic_equation_l73_73582


namespace track_length_correct_l73_73459

noncomputable def track_length : ℕ :=
  let brenda_first_meet := 120 -- Brenda runs 120 meters
  let sally_additional_run := 160 -- Sally runs an additional 160 meters after the first meeting
  let equation1 := (λ x : ℕ, brenda_first_meet = x / 2 - brenda_first_meet)
  let equation2 := (λ x : ℕ, x / 2 + sally_additional_run = x / 2 + 40)
  let proportion := (λ x : ℕ, 120 * 160 = (x / 2 - 80) * (x / 2 - 120))
  480 -- Given solution

theorem track_length_correct :
  let x := track_length in
  ∃ x, (120 = x / 2 - 120) ∧ (x / 2 + 160 = x / 2 + 40) ∧ (120 * 160 = (x / 2 - 80) * (x / 2 - 120)) ∧ x = 480 :=
by
  sorry

end track_length_correct_l73_73459


namespace find_parameter_a_l73_73574

def takes_extreme_value (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ (f' : ℝ → ℝ), 
  (∀ x, f' x = (deriv f x)) ∧ 
  (f' x = 0)

theorem find_parameter_a :
  ∀ (f : ℝ → ℝ) (a : ℝ),
  (f = λ x, x^3 + a * x^2 + 3 * x - 9) →
  takes_extreme_value f (-3) →
  a = 5 :=
by
  intros f a h₀ h₁
  sorry

end find_parameter_a_l73_73574


namespace min_distance_point_l73_73160

noncomputable def curve (θ : ℝ) : ℝ × ℝ := (-1 + Real.cos θ, Real.sin θ)

def line (P : ℝ × ℝ) : ℝ := P.1 + P.2 - 1

def distance_to_line (P : ℝ × ℝ) : ℝ := abs (line P) / Real.sqrt 2

theorem min_distance_point :
  ∃ θ : ℝ, let P := curve θ in
  distance_to_line P = Real.sqrt 2 - 1 ∧
  P = (-1 + Real.sqrt 2 / 2, Real.sqrt 2 / 2) :=
by
  sorry

end min_distance_point_l73_73160


namespace num_positive_perfect_square_sets_l73_73237

-- Define what it means for three numbers to form a set that sum to 100 
def is_positive_perfect_square_set (a b c : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ c ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 100

-- Define the main theorem to state there are exactly 4 such sets
theorem num_positive_perfect_square_sets : 
  {s : Finset (ℕ × ℕ × ℕ) // (∃ a b c, (a, b, c) ∈ s ∧ is_positive_perfect_square_set a b c) }.card = 4 :=
sorry

end num_positive_perfect_square_sets_l73_73237


namespace g_2010_not_divisible_by_11_g_2011_divisible_by_11_l73_73268

def g : ℕ → ℤ
| 0     := 0
| 1     := 4
| 2     := 10
| 3     := 33
| (n+4) := 9 * g (n+2) - g n -- From the simplified recurrence: g(n+3) ≡ -g(n-1) mod 11

theorem g_2010_not_divisible_by_11 
  (h1 : ∀ n, g (n+6) ≡ g n [MOD 11])
  (h2 : g 2 = 10)
  (h3 : g 3 = 33) :
  g 2010 % 11 ≠ 0 :=
by sorry

theorem g_2011_divisible_by_11 
  (h1 : ∀ n, g (n+6) ≡ g n [MOD 11])
  (h2 : g 2 = 10)
  (h3 : g 3 = 33) :
  g 2011 % 11 = 0 :=
by sorry

end g_2010_not_divisible_by_11_g_2011_divisible_by_11_l73_73268


namespace trapezoidal_garden_solutions_l73_73686

theorem trapezoidal_garden_solutions :
  ∃ (b1 b2 : ℕ), 
    (1800 = (60 * (b1 + b2)) / 2) ∧
    (b1 % 10 = 0) ∧ (b2 % 10 = 0) ∧
    (∃ (n : ℕ), n = 4) := 
sorry

end trapezoidal_garden_solutions_l73_73686


namespace sin_squared_alpha_l73_73913

theorem sin_squared_alpha (α : ℝ) (h₁ : sin α = sin 15) (h₂ : cos α = -cos 15) : 
  sin α ^ 2 = 1/2 + sqrt 3 / 4 := 
sorry

end sin_squared_alpha_l73_73913


namespace connect_points_l73_73516

theorem connect_points (n : ℕ) (h : n ≥ 1) (points : fin (2 * n) → point)
  (no_three_collinear : ∀ {i j k : fin (2 * n)},
    i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k))
  (coloring : fin (2 * n) → color)
  (n_blue : (coloring_points_count coloring color.blue) = n)
  (n_red  : (coloring_points_count coloring color.red) = n) :
  ∃ segments : fin n → (fin (2 * n) × fin (2 * n)),
    (∀ i, coloring (segments i).1 = color.blue ∧ coloring (segments i).2 = color.red) ∧
    non_intersecting_segments segments points :=
sorry

end connect_points_l73_73516


namespace triangle_ABC_area_l73_73253

-- Define the conditions for triangle ABC
variables (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variables (AB AC AM BC : ℝ)

-- Define values for sides and median according to the problem statement
axiom AB_eq_9 : AB = 9
axiom AC_eq_17 : AC = 17
axiom AM_eq_12 : AM = 12

-- Define that M is the midpoint of BC
axiom is_midpoint_MBC : ∀ B C : ℝ, M = (B + C) / 2

-- Function to calculate area of a triangle given three sides using Heron's formula
noncomputable def heron_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem stating the area of the triangle ABC
theorem triangle_ABC_area : heron_area AB AC BC = 110 :=
sorry

end triangle_ABC_area_l73_73253


namespace problem_statement_l73_73267

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_statement (f_bound : ∀ x ∈ Icc 0 1, |f x| ≤ 1)
    (f_diff : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1) :
    ∀ x : ℝ, |f x| ≤ 2 + x^2 :=
by
  sorry

end problem_statement_l73_73267


namespace sum_of_integers_85_to_100_l73_73056

theorem sum_of_integers_85_to_100 : ∑ i in finset.Icc 85 100, i = 1480 :=
by
  sorry

end sum_of_integers_85_to_100_l73_73056


namespace problem_a_problem_b_l73_73724

-- Problems related to handshakes and grouping in a set of n people

theorem problem_a (n : ℕ) (h_n : n ≥ 4) 
  (handshake : ∀ {A B C D : ℕ}, A ≠ B ∧ B ≠ C ∧ C ≠ D → 
    (shook_hands A B ∧ shook_hands B C ∧ shook_hands C D → 
    (shook_hands A C ∨ shook_hands A D ∨ shook_hands B D))) :
  ∃ (X Y : set ℕ), X ≠ ∅ ∧ Y ≠ ∅ ∧ ∀ x ∈ X, ∀ y ∈ Y, (shook_hands x y ∨ ¬ shook_hands x y) :=
  sorry

theorem problem_b (n : ℕ) (h_n : n ≥ 4)
  (handshake : ∀ {A B C D : ℕ}, A ≠ B ∧ B ≠ C ∧ C ≠ D →
    (shook_hands A B ∧ shook_hands B C ∧ shook_hands C D →
    (shook_hands A C ∨ shook_hands A D ∨ shook_hands B D))) :
  ∃ (A B : ℕ),
    { p | p ≠ A ∧ p ≠ B ∧ shook_hands A p } = { p | p ≠ A ∧ p ≠ B ∧ shook_hands B p } :=
  sorry

end problem_a_problem_b_l73_73724


namespace maximize_area_of_triangle_l73_73147

noncomputable
def ellipse : Set (ℝ × ℝ) := { p | let x := p.1 in let y := p.2 in (x^2 / 4) + (y^2 / 3) = 1 }

def line (k : ℝ) : Set (ℝ × ℝ) := { p | let x := p.1 in let y := p.2 in y = k * x - 1 }

def left_focus : ℝ × ℝ := (-2, 0)

def intersects_ellipse (k : ℝ) : Set (ℝ × ℝ) :=
  { p | let x := p.1 in let y := p.2 in
         (x^2 / 4) + (y^2 / 3) = 1 ∧ y = k * x - 1 }

theorem maximize_area_of_triangle (k : ℝ) :
  ∃A B : (ℝ × ℝ), A ≠ B ∧ A ∈ intersects_ellipse k ∧ B ∈ intersects_ellipse k ∧
  let F := left_focus in
  let area := Real.abs ((F.1 * (A.2 - B.2) + A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2)) / 2) in
  (P := (dist F A + dist F B + dist A B)) = 8 → 
  area = 12 * Real.sqrt 2 / 7 := 
sorry

end maximize_area_of_triangle_l73_73147


namespace uncovered_area_l73_73988

-- Definition: Square with side length 4 cm
def side_length_square : ℝ := 4

-- Definition: Circle with diameter 1 cm
def diameter_circle : ℝ := 1

-- Proof statement: Area of part of the square never covered by the circle
theorem uncovered_area (side_length_square = 4) (diameter_circle = 1) : 
  ∃ (area : ℝ), area = 4 + (1 / 4) * Real.pi := 
sorry

end uncovered_area_l73_73988


namespace solve_for_x_l73_73309

theorem solve_for_x (x : ℚ) (h : (x + 2) / (x - 3) = (x - 4) / (x + 5)) : x = 1 / 7 :=
sorry

end solve_for_x_l73_73309


namespace basketball_player_possible_scores_l73_73014

-- Define the conditions
def isValidBasketCount (n : Nat) : Prop := n = 7
def isValidBasketValue (v : Nat) : Prop := v = 1 ∨ v = 2 ∨ v = 3

-- Define the theorem statement
theorem basketball_player_possible_scores :
  ∃ (s : Finset ℕ), s = {n | ∃ n1 n2 n3 : Nat, 
                                n1 + n2 + n3 = 7 ∧ 
                                n = 1 * n1 + 2 * n2 + 3 * n3 ∧ 
                                n1 + n2 + n3 = 7 ∧ 
                                n >= 7 ∧ n <= 21} ∧
                                s.card = 15 :=
by
  sorry

end basketball_player_possible_scores_l73_73014


namespace arithmetic_sequence_product_l73_73274

theorem arithmetic_sequence_product 
  (b : ℕ → ℤ) 
  (h_arith : ∀ n, b n = b 0 + (n : ℤ) * (b 1 - b 0))
  (h_inc : ∀ n, b n ≤ b (n + 1))
  (h4_5 : b 4 * b 5 = 21) : 
  b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := 
by 
  sorry

end arithmetic_sequence_product_l73_73274


namespace sin_pi_minus_alpha_l73_73903

theorem sin_pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = 4 / 5) :
  Real.sin (Real.pi - α) = 3 / 5 := 
sorry

end sin_pi_minus_alpha_l73_73903


namespace max_power_of_two_divides_a_k_l73_73931

def a : ℕ → ℕ
| 1       := 1
| (n + 1) := 5 * a n + 3^n

theorem max_power_of_two_divides_a_k :
  ∀ k, k = 2^2019 → (∃ m, 2^2021 ∣ a k) :=
by
  sorry

end max_power_of_two_divides_a_k_l73_73931


namespace sector_area_l73_73315

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360) * Real.pi * r^2 = (35 * Real.pi) / 3 :=
by
  -- Using the provided conditions to simplify the expression
  rw [h_r, h_θ]
  -- Simplify and solve the expression
  sorry

end sector_area_l73_73315


namespace incenter_coordinates_l73_73986

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (P Q R J : V)
variables (p q r : ℝ)
variables (u v w : ℝ)
variables h1 : p = 8
variables h2 : q = 11
variables h3 : r = 5
variables h4 : J = u • P + v • Q + w • R
variables huvw : u + v + w = 1

theorem incenter_coordinates :
  u = 11/24 ∧ v = 8/24 ∧ w = 5/24 :=
sorry

end incenter_coordinates_l73_73986


namespace average_speed_of_truckX_l73_73365

noncomputable def truckX_speed (Vx Vy : ℝ) (d : ℝ) (t : ℝ) (a : ℝ) : Prop :=
  d + a = Vy * t ∧ Vy = 53 ∧ t = 3 ∧ d = 13 ∧ a = 5

theorem average_speed_of_truckX : ∀ (Vx Vy : ℝ), 
  truckX_speed Vx Vy 13 3 5 → Vx = 47 :=
by
  intros Vx Vy h,
  cases h with eq1 h1,
  cases h1 with hVy h2,
  cases h2 with ht h3,
  cases h3 with hd ha,
  sorry

end average_speed_of_truckX_l73_73365


namespace find_a_l73_73025

noncomputable def calculation (a : ℚ) :=
  let p1 := ⟨-3, 7⟩ : ℚ × ℚ
  let p2 := ⟨2, 1⟩ : ℚ × ℚ
  let direction_vector := ⟨p2.1 - p1.1, p2.2 - p1.2⟩ : ℚ × ℚ
  let k := 1 / direction_vector.2
  let scaled_direction := ⟨k * direction_vector.1, k * direction_vector.2⟩
  (scaled_direction.1 = a ∧ scaled_direction.2 = -1)

theorem find_a (a : ℚ) : 
  (∃ k : ℚ, k * 5 = a ∧ k * (-6) = -1) → a = (5 / 6) :=
by
  intro h
  sorry

end find_a_l73_73025


namespace minimum_value_l73_73273

open Real

-- Statement of the conditions
def conditions (a b c : ℝ) : Prop :=
  -0.5 < a ∧ a < 0.5 ∧ -0.5 < b ∧ b < 0.5 ∧ -0.5 < c ∧ c < 0.5

-- Expression to be minimized
noncomputable def expression (a b c : ℝ) : ℝ :=
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c))

-- Minimum value to prove
theorem minimum_value (a b c : ℝ) (h : conditions a b c) : expression a b c ≥ 4.74 :=
sorry

end minimum_value_l73_73273


namespace equilateral_triangle_of_sequences_l73_73627

-- Given angles A, B, and C
variables (A B C : ℝ)

-- Conditions from the problem
def form_arithmetic_sequence (A B C : ℝ) : Prop := B = A + (C - A) / 2
def form_geometric_sequence (A B C : ℝ) : Prop := sin B ^ 2 = sin A * sin C

-- Prove that the triangle is equilateral
theorem equilateral_triangle_of_sequences (h1 : form_arithmetic_sequence A B C) (h2 : form_geometric_sequence A B C)
  (sum_angles : A + B + C = 180) : A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end equilateral_triangle_of_sequences_l73_73627


namespace eq_solutions_of_equation_l73_73481

open Int

theorem eq_solutions_of_equation (x y : ℤ) :
  ((x, y) = (0, -4) ∨ (x, y) = (0, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-4, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-6, 6) ∨
   (x, y) = (0, 0) ∨ (x, y) = (-10, 4)) ↔
  (x - y) * (x - y) = (x - y + 6) * (x + y) :=
sorry

end eq_solutions_of_equation_l73_73481


namespace sum_of_number_and_conjugate_l73_73075

noncomputable def x : ℝ := 16 - Real.sqrt 2023
noncomputable def y : ℝ := 16 + Real.sqrt 2023

theorem sum_of_number_and_conjugate : x + y = 32 :=
by
  sorry

end sum_of_number_and_conjugate_l73_73075


namespace number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73228

theorem number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares :
  ∃ (n : ℕ), n = 3 ∧ ∀ a b c : ℕ, a^2 + b^2 + c^2 = 100 → 
    a*b*c ≠ 0 → a^2 ≤ b^2 ≤ c^2 :=
by sorry

end number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73228


namespace age_problem_l73_73960

theorem age_problem (M D : ℕ) (h1 : M = 40) (h2 : 2 * D + M = 70) : 2 * M + D = 95 := by
  sorry

end age_problem_l73_73960


namespace correct_option_l73_73752

theorem correct_option (A B C D : Prop) :
  A = (\sqrt[3]{-8} = 2) ∧
  B = (\sqrt{(-3)^2} = -3) ∧
  C = (2\sqrt{5} + 3\sqrt{5} = 5\sqrt{5}) ∧
  D = ((\sqrt{2} + 1)^2 = 3) →
  C :=
by
  sorry

end correct_option_l73_73752


namespace concurrency_of_AD_BE_CF_l73_73280

theorem concurrency_of_AD_BE_CF
  (A B C I D E F : Type)
  [InCenter I A B C]
  [Circle I D E F]
  [Perpendicular I D B C]
  [Perpendicular I E C A]
  [Perpendicular I F A B] :
  Concurrent (Line A D) (Line B E) (Line C F) :=
sorry

end concurrency_of_AD_BE_CF_l73_73280


namespace remainder_when_sum_divided_by_7_l73_73378

-- Define the sequence
def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the arithmetic sequence
def arithmetic_sequence_sum (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
def a : ℕ := 3
def d : ℕ := 7
def last_term : ℕ := 304

-- Calculate the number of terms in the sequence
noncomputable def n : ℕ := (last_term + 4) / 7

-- Calculate the sum
noncomputable def S : ℕ := arithmetic_sequence_sum a d n

-- Lean 4 statement to prove the remainder
theorem remainder_when_sum_divided_by_7 : S % 7 = 3 := by
  -- sorry placeholder for proof
  sorry

end remainder_when_sum_divided_by_7_l73_73378


namespace value_of_a7_minus_a8_l73_73248

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem value_of_a7_minus_a8
  (h_seq: arithmetic_sequence a d)
  (h_sum: a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - a 8 = d :=
sorry

end value_of_a7_minus_a8_l73_73248


namespace intersection_of_A_and_B_l73_73189

-- Definitions of sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Definition of the expected intersection of A and B
def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The main theorem stating the proof problem
theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ (A ∩ B) ↔ x ∈ expected_intersection :=
by
  intro x
  sorry

end intersection_of_A_and_B_l73_73189


namespace truck_travel_l73_73443

/-- If a truck travels 150 miles using 5 gallons of diesel, then it will travel 210 miles using 7 gallons of diesel. -/
theorem truck_travel (d1 d2 g1 g2 : ℕ) (h1 : d1 = 150) (h2 : g1 = 5) (h3 : g2 = 7) (h4 : d2 = d1 * g2 / g1) : d2 = 210 := by
  sorry

end truck_travel_l73_73443


namespace sequence_values_geometric_conditions_l73_73524

noncomputable def sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * S n

def sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
(n+1).sum (λ i, a i)

theorem sequence_values (S : ℕ → ℕ) (a : ℕ → ℕ) :
  a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 18 :=
by {
  have : a 0 = 1 := rfl, -- initial condition
  suffices H : ∀ n, a (n + 1) = 2 * S n, from sorry,
  exact H
}

theorem geometric_conditions (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∃ (p r : ℝ), (∀ n, a (n + 1) = p * S n + r) →
  (∃ q, ∀ n ≥ 2, a (n + 1) = (p+1) * a n) ∧ p ≠ -1 ∧ r = 1) := sorry

end sequence_values_geometric_conditions_l73_73524


namespace angle_phi_max_min_values_g_l73_73546

open Real

noncomputable def f (x φ : ℝ) : ℝ := 
  (1/2) * sin (2 * x) * sin φ + cos x ^ 2 * cos φ - (1/2) * sin (π/2 + φ)

-- Part (I)
theorem angle_phi (h : 0 < φ ∧ φ < π) (h_fx: f (π/6) φ = 1/2) : φ = π / 3 := sorry

-- Part (II)
noncomputable def g (x : ℝ) : ℝ := f (x / 2) (π / 3)

theorem max_min_values_g : (∀ x ∈ Icc 0 (π/4), g x <= 1/2 ∧ g x >= -1/2) ∧ ∃ x ∈ Icc 0 (π/4), g x = 1/2 ∧ ∀ x ∈ Icc 0 (π/4), g x > -1/2 := sorry

end angle_phi_max_min_values_g_l73_73546


namespace number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73229

theorem number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares :
  ∃ (n : ℕ), n = 3 ∧ ∀ a b c : ℕ, a^2 + b^2 + c^2 = 100 → 
    a*b*c ≠ 0 → a^2 ≤ b^2 ≤ c^2 :=
by sorry

end number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l73_73229


namespace find_n_l73_73112

theorem find_n (n : ℕ) (h : Choose n 4 = Factorial 3 * Choose n 3) : n = 27 :=
by
  sorry

end find_n_l73_73112


namespace min_points_to_ensure_distance_lt_half_l73_73175

theorem min_points_to_ensure_distance_lt_half : 
  ∃ (n : ℕ), (∀ pts : fin n → ℝ × ℝ, ∃ i j : fin n, i ≠ j ∧ dist (pts i) (pts j) < 1 / 2) := 
  10 :=
sorry

end min_points_to_ensure_distance_lt_half_l73_73175


namespace max_profit_l73_73039

def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then (1 / 3) * x^2 + x
  else 6 * x + 100 / x - 38

def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then - (1 / 3) * x^2 + 4 * x - 3
  else - x + 100 / x + 35

theorem max_profit :
  ∃ x : ℝ, 0 < x ∧ x = 10 ∧ L x = 15 :=
sorry

end max_profit_l73_73039


namespace find_number_l73_73776

theorem find_number (x : ℝ) (h : 140 = 3.5 * x) : x = 40 :=
by
  sorry

end find_number_l73_73776


namespace rectangle_length_width_difference_l73_73312

noncomputable def difference_between_length_and_width : ℝ :=
  let x := by sorry
  let y := by sorry
  (x - y)

theorem rectangle_length_width_difference {x y : ℝ}
  (h₁ : 2 * (x + y) = 20) (h₂ : x^2 + y^2 = 10^2) :
  difference_between_length_and_width = 10 :=
  by sorry

end rectangle_length_width_difference_l73_73312


namespace find_hcf_l73_73764

-- Given conditions in the problem
variables (a b : ℕ)
variable hcf_lcm_relation : ∀ {a b : ℕ}, lcm a b * gcd a b = a * b
variable lcm_ab : lcm a b = 600
variable prod_ab : a * b = 18000

-- The proof statement
theorem find_hcf (a b : ℕ) (hcf_lcm_relation : ∀ {a b : ℕ}, lcm a b * gcd a b = a * b) (lcm_ab : lcm a b = 600) (prod_ab : a * b = 18000) : gcd a b = 30 :=
by
  sorry

end find_hcf_l73_73764


namespace token_exchange_l73_73040

def booth1 (r : ℕ) (x : ℕ) : ℕ × ℕ × ℕ := (r - 3 * x, 2 * x, x)
def booth2 (b : ℕ) (y : ℕ) : ℕ × ℕ × ℕ := (y, b - 4 * y, y)

theorem token_exchange (x y : ℕ) (h1 : 100 - 3 * x + y = 2) (h2 : 50 + x - 4 * y = 3) :
  x + y = 58 :=
sorry

end token_exchange_l73_73040


namespace remaining_movie_time_l73_73038

def start_time := 200 -- represents 3:20 pm in total minutes from midnight
def end_time := 350 -- represents 5:44 pm in total minutes from midnight
def total_movie_duration := 180 -- 3 hours in minutes

theorem remaining_movie_time : total_movie_duration - (end_time - start_time) = 36 :=
by
  sorry

end remaining_movie_time_l73_73038


namespace apples_more_than_oranges_l73_73717

-- Definitions based on conditions
def total_fruits : ℕ := 301
def apples : ℕ := 164

-- Statement to prove
theorem apples_more_than_oranges : (apples - (total_fruits - apples)) = 27 :=
by
  sorry

end apples_more_than_oranges_l73_73717


namespace find_a_c_m_l73_73922

-- Definitions of the conditions
def f (a c x : ℝ) : ℝ := a * x ^ 2 + 2 * x + c

-- Lean statement for the proof
theorem find_a_c_m (a c m : ℝ) 
  (h_nat_a : ∃ n : ℕ, n ≠ 0 ∧ a = n) 
  (h_nat_c : ∃ n : ℕ, n ≠ 0 ∧ c = n)
  (h_f1 : f a c 1 = 5) 
  (h_f2 : 6 < f a c 2 ∧ f a c 2 < 11)
  (h_ineq : ∀ x ∈ set.Icc (1 / 2 : ℝ) (3 / 2), f a c x - 2 * m * x ≤ 1) :
  a = 1 ∧ c = 2 ∧ m ≥ 9 / 4 := by
  sorry

end find_a_c_m_l73_73922


namespace range_of_special_expression_l73_73114

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - x^2 + a * x + 1

theorem range_of_special_expression (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x1 < x2) 
  (h3 : x1 + x2 = 2) (h4 : x1 * x2 = a) : 
  ∃ y, y = x2 / (1 + x1 / x2) ∧ y ∈ set.Ioo (1/2 : ℝ) (2 : ℝ) :=
sorry

end range_of_special_expression_l73_73114


namespace hens_count_l73_73759

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 := by
  sorry

end hens_count_l73_73759


namespace problem_statement_l73_73328

variables {ℝ : Type} [noncomputable]

-- Define f as an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

-- Define f as decreasing when x ≥ 0
def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : odd_function f) 
  (h2 : decreasing_on_nonneg f) 
  (h3 : a + b > 0) : 
  f(a) + f(b) < 0 := 
sorry

end problem_statement_l73_73328


namespace f_2015_value_l73_73531

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_odd : odd_function f
axiom f_periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom f_definition_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 3^x - 1

theorem f_2015_value : f 2015 = -2 :=
by
  sorry

end f_2015_value_l73_73531


namespace card_distribution_l73_73771

-- Definition of operations and conditions
def operationA (n : ℕ) (cards : Fin n → ℕ) (O : ℕ) (i : Fin n) : (Fin n → ℕ) × ℕ :=
  if cards i >= 3 then
    let cards' := cards i - 3
    let cards_minus := λ j, if j = i then cards' else cards j
    let cards_add := λ j, if j = (i + 1) % n then (cards (i + 1) % n) + 1
                             else if j = (i - 1) % n then (cards (i - 1) % n) + 1
                             else cards_minus j
    (cards_add, O + 1)
  else (cards, O)

def operationB (n : ℕ) (cards : Fin n → ℕ) (O : ℕ) : (Fin n → ℕ) × ℕ :=
  if O >= n then
    (λ i, (cards i) + 1, O - n)
  else (cards, O)

-- Theorem Statement
theorem card_distribution (n : ℕ) (cards : Fin n → ℕ) (O : ℕ) :
  n ≥ 3 →
  (Σ i, cards i) + O ≥ n^2 + 3 * n + 1 →
  ∃ f : ℕ → (Fin n → ℕ) × ℕ,
    (∀ m, f (m + 1) = operationB n (fst (f m)) (snd (f m)) ∨ ∃ i, f (m + 1) = operationA n (fst (f m)) (snd (f m)) i) ∧
    ∀ i, (Σ j, (fst (f i)) j) + (snd (f i)) ≥ n^2 + 3 * n + 1 → 
    ∀ k, (fst (f i)) k ≥ n + 1 :=
begin
  intros h1 h2,
  -- We would need detailed proof steps here
  sorry,  -- Placeholder for the proof
end

end card_distribution_l73_73771


namespace largest_prime_y_in_triangle_l73_73222

-- Define that a number is prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_y_in_triangle : 
  ∃ (x y z : ℕ), is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 90 ∧ y < x ∧ y > z ∧ y = 47 :=
by
  sorry

end largest_prime_y_in_triangle_l73_73222


namespace probability_of_conditions_holds_l73_73801

-- Define the conditions and problem setup
noncomputable def equation_roots_condition (k : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), (x1 + x2 = (9 - 3 * k) / (k^2 - 2 * k - 35)) ∧
                 (x1 * x2 = 2 / (k^2 - 2 * k - 35)) ∧
                 (x1 ≤ 2 * x2)

def k_in_segment (a b : ℝ) (k : ℝ) : Prop := k ∈ set.Icc a b

def roots_condition_probability : ℝ :=
  (λ k, if equation_roots_condition k then 1 else 0).integral (set.Icc 8 13) / 5

-- The theorem to state and prove
theorem probability_of_conditions_holds :
  roots_condition_probability = 0.6 :=
sorry

end probability_of_conditions_holds_l73_73801


namespace sum_of_inradii_l73_73414

theorem sum_of_inradii (ABC : Triangle) (r r_A r_B r_C : ℝ) 
    (h₀ : ABC.isInscribed r) 
    (h₁ : smaller_triangles_with_radii r_A r_B r_C ABC) : 
    r_A + r_B + r_C = r := 
sorry

end sum_of_inradii_l73_73414


namespace find_first_number_l73_73339

theorem find_first_number (x : ℕ) (h : x + 15 = 20) : x = 5 :=
by
  sorry

end find_first_number_l73_73339


namespace min_dist_l73_73552

def parabola : set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

def focus : ℝ × ℝ := (1, 0)

def pointA : ℝ × ℝ := (2, 2)

def on_parabola (P : ℝ × ℝ) : Prop := P ∈ parabola

noncomputable def euclidean_dist (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

theorem min_dist (P : ℝ × ℝ) (hP : on_parabola P) : euclidean_dist P pointA + euclidean_dist P focus = 3 :=
sorry

end min_dist_l73_73552


namespace geometric_sequence_min_value_proof_l73_73967

noncomputable def geometric_sequence_min_value : ℝ :=
  let q : ℝ := sorry -- common ratio of the geometric sequence, positive
  in let a6 := 3 -- given condition
  in (a6 / (q ^ 2)) + (a6 * (q ^ 2))

theorem geometric_sequence_min_value_proof (q : ℝ) (hq : q > 0) :
  geometric_sequence_min_value ≥ 6 :=
by {
  let a6 := 3,
  let a4 := a6 / (q ^ 2),
  let a8 := a6 * (q ^ 2),
  have h : a4 + a8 = (3 / (q ^ 2)) + (3 * (q ^ 2)),
  rw_mod_cast [a6, a4, a8] at h,
  calc 
  (3 / (q ^ 2)) + (3 * (q ^ 2)) ≥ 2 * sqrt ((3 / (q ^ 2)) * (3 * (q ^ 2))) : 
    by { apply add_sqrt_mul, linarith }, -- applying AM-GM inequality
  _ = 6 : by norm_num,
  sorry -- proof verification for equality case
}

end geometric_sequence_min_value_proof_l73_73967


namespace unique_n_value_l73_73003

theorem unique_n_value :
  ∃ (n : ℕ), n > 0 ∧ (∃ k : ℕ, k > 0 ∧ k < 10 ∧ 111 * k = (n * (n + 1) / 2)) ∧ ∀ (m : ℕ), m > 0 → (∃ j : ℕ, j > 0 ∧ j < 10 ∧ 111 * j = (m * (m + 1) / 2)) → m = 36 :=
by
  sorry

end unique_n_value_l73_73003


namespace geometric_sequence_ratio_l73_73541

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_q : q = -1 / 2) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 :=
sorry

end geometric_sequence_ratio_l73_73541


namespace number_of_correct_propositions_l73_73900

-- Definitions of the geometric entities and conditions
variables (a b : Line) (α β : Plane)

-- Proposition 1 condition and statement
def prop1 (h1 : a ∥ b) (h2 : b ∥ α) : a ∥ α := false

-- Proposition 2 condition and statement
def prop2 (h1 : a ⊆ α) (h2 : b ⊆ α) (h3 : a ∥ β) (h4 : b ∥ β) : α ∥ β := false

-- Proposition 3 condition and statement
def prop3 (h1 : angle_between a α = 30) (h2 : a ⥋ b) : angle_between b α = 60 := false

-- Proposition 4 condition and statement
def prop4 (h1 : a ⥋ α) (h2 : b ∥ α) : a ⥋ b := true

-- Main theorem statement: number of correct propositions is 1
theorem number_of_correct_propositions : ∃ n, n = 1 ∧
  ((prop1 a b α = false) ∧ (prop2 a b α β = false) ∧ (prop3 a α b = false) ∧ (prop4 a α b)) := 
sorry

end number_of_correct_propositions_l73_73900


namespace number_of_subsets_S_l73_73565

open Set

def S : Set Int := {-1, 0, 1}

theorem number_of_subsets_S : Fintype.card (set.powerset S) = 8 := 
  sorry

end number_of_subsets_S_l73_73565


namespace sum_of_f_values_l73_73110

def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - 5 / 12

theorem sum_of_f_values : 
  (f 0) + (f (1 / 2017)) + (f (2 / 2017)) + ⋯ + (f (2015 / 2017)) + (f (2016 / 2017)) + (f 1) = 2018 := 
sorry

end sum_of_f_values_l73_73110


namespace sequence_nth_term_sum_tn_reciprocal_l73_73527

theorem sequence_nth_term (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n : ℕ, S (n + 1) - 3 * S n = 1) :
  ∀ n : ℕ, a n = 3^(n - 1) := 
sorry

theorem sum_tn_reciprocal (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (b : ℕ → ℕ) (T : ℕ → ℕ)
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n : ℕ, S (n + 1) - 3 * S n = 1) 
  (h₂ : ∀ n : ℕ, b n = Int.logBase 3 (a (n + 1))) 
  (h₃ : ∀ n : ℕ, T n = (n * (n + 1)) / 2) :
  ∑ i in Finset.range 100, 1 / (T (i + 1)) = 200 / 101 :=
sorry

end sequence_nth_term_sum_tn_reciprocal_l73_73527


namespace junior_average_score_l73_73215

def total_students : ℕ := 20
def proportion_juniors : ℝ := 0.2
def proportion_seniors : ℝ := 0.8
def average_class_score : ℝ := 78
def average_senior_score : ℝ := 75

theorem junior_average_score :
  let num_juniors := total_students * proportion_juniors
  let num_seniors := total_students * proportion_seniors
  let total_score := total_students * average_class_score
  let total_senior_score := num_seniors * average_senior_score
  let total_junior_score := total_score - total_senior_score
  total_junior_score / num_juniors = 90 := 
by
  sorry

end junior_average_score_l73_73215


namespace Leila_donated_2_bags_l73_73644

theorem Leila_donated_2_bags (L : ℕ) (h1 : 25 * L + 7 = 57) : L = 2 :=
by
  sorry

end Leila_donated_2_bags_l73_73644


namespace smaller_acute_angle_in_right_triangle_l73_73223

theorem smaller_acute_angle_in_right_triangle
    (a b : ℝ)
    (h1 : ∠a = 90)
    (ha : ratio_angles a b = 7 / 2)
    (h2 : a + b = 90) : 
    b = 20 :=
by
     sorry

end smaller_acute_angle_in_right_triangle_l73_73223


namespace altitudes_reciprocal_sum_eq_reciprocal_inradius_l73_73277

theorem altitudes_reciprocal_sum_eq_reciprocal_inradius
  (h1 h2 h3 r : ℝ)
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0)
  (h3_pos : h3 > 0)
  (r_pos : r > 0)
  (triangle_area_eq : ∀ (a b c : ℝ),
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧ a + b + c > 0) :
  1 / h1 + 1 / h2 + 1 / h3 = 1 / r := 
by
  sorry

end altitudes_reciprocal_sum_eq_reciprocal_inradius_l73_73277


namespace students_sign_up_ways_l73_73778

theorem students_sign_up_ways :
  let students := 4
  let choices_per_student := 3
  (choices_per_student ^ students) = 3^4 :=
by
  sorry

end students_sign_up_ways_l73_73778


namespace cabbage_production_l73_73792

theorem cabbage_production (x y : ℕ) 
  (h1 : y^2 - x^2 = 127) 
  (h2 : y - x = 1) 
  (h3 : 2 * y = 128) : y^2 = 4096 := by
  sorry

end cabbage_production_l73_73792


namespace x_y_quartic_l73_73575

theorem x_y_quartic (x y : ℝ) (h₁ : x - y = 2) (h₂ : x * y = 48) : x^4 + y^4 = 5392 := by
  sorry

end x_y_quartic_l73_73575


namespace pure_imaginary_b_value_l73_73955

theorem pure_imaginary_b_value (b : ℝ) (i : ℂ) (hi : i = complex.I) :
  (1 + b * i) * (2 + i) = complex.I * (1 + 2 * b) → b = 2 :=
by
  sorry

end pure_imaginary_b_value_l73_73955


namespace boat_speed_against_stream_l73_73972

theorem boat_speed_against_stream
  (dist_along_stream : ℝ)
  (speed_still_water : ℝ)
  (effective_speed_along : ℝ)
  (stream_speed : ℝ)
  (effective_speed_against : ℝ) :
  dist_along_stream = 11 →
  speed_still_water = 9 →
  effective_speed_along = speed_still_water + stream_speed →
  effective_speed_along = 11 →
  stream_speed = 2 →
  effective_speed_against = speed_still_water - stream_speed →
  effective_speed_against = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h4, h5, h6]
  norm_num
  sorry

end boat_speed_against_stream_l73_73972


namespace luke_total_coins_l73_73664

def piles_coins_total (piles_quarters : ℕ) (coins_per_pile_quarters : ℕ) 
                      (piles_dimes : ℕ) (coins_per_pile_dimes : ℕ) 
                      (piles_nickels : ℕ) (coins_per_pile_nickels : ℕ) 
                      (piles_pennies : ℕ) (coins_per_pile_pennies : ℕ) : ℕ :=
  (piles_quarters * coins_per_pile_quarters) +
  (piles_dimes * coins_per_pile_dimes) +
  (piles_nickels * coins_per_pile_nickels) +
  (piles_pennies * coins_per_pile_pennies)

theorem luke_total_coins : 
  piles_coins_total 8 5 6 7 4 4 3 6 = 116 :=
by
  sorry

end luke_total_coins_l73_73664


namespace area_of_PQRS_l73_73674

noncomputable def side_length_square (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def height_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * real.sqrt 3) / 2

-- Conditions
def P_Q_R_S_in_plane_of_WXYZ (P Q R S W X Y Z: ℝ × ℝ) : Prop :=
  ∃ (side_length : ℝ), side_length = side_length_square 25 ∧
  ∀ (WPY XQZ YRW ZSX : Prop),
  WPY ∧ XQZ ∧ YRW ∧ ZSX ∧
  (
    WPY = (P.1 - W.1) ^ 2 + (P.2 - W.2) ^ 2 = side_length ^ 2 ∧
    XQZ = (Q.1 - X.1) ^ 2 + (Q.2 - X.2) ^ 2 = side_length ^ 2 ∧
    YRW = (R.1 - Y.1) ^ 2 + (R.2 - Y.2) ^ 2 = side_length ^ 2 ∧
    ZSX = (S.1 - Z.1) ^ 2 + (S.2 - Z.2) ^ 2 = side_length ^ 2
  )

def diagonal_length (side_length : ℝ): ℝ :=
  side_length * real.sqrt 2

noncomputable def area_PQRS (side_length_square : ℝ): ℝ :=
  let side_length := side_length_square 25
  let height := height_equilateral_triangle side_length
  let diagonal := diagonal_length side_length + 2 * height
  1 / 2 * (diagonal ^ 2)

-- Prove
theorem area_of_PQRS (PQRS : ℝ) (W X Y Z P Q R S : ℝ × ℝ) :
  P_Q_R_S_in_plane_of_WXYZ P Q R S W X Y Z →
  PQRS = 62.5 + 25 * real.sqrt 6 :=
sorry

end area_of_PQRS_l73_73674


namespace maximum_obtuse_rays_l73_73747

theorem maximum_obtuse_rays (v : ℝ^3 → Prop) :
  (∀ u w : ℝ^3, v u → v w → u ≠ w → dot_product u w < 0) → 
  ∃ S : Finset (ℝ^3), S.card = 4 ∧ (∀ u w ∈ S, u ≠ w → dot_product u w < 0) :=
sorry

end maximum_obtuse_rays_l73_73747


namespace probability_A_and_B_at_Position_1_l73_73352

theorem probability_A_and_B_at_Position_1 :
  let volunteers := ['A, 'B, 'C, 'D, 'E],
      positions := [Position1, Position2, Position3, Position4],
      num_positions := 4 in
  (∃ (assign: volunteers → positions), 
      ∃ (distinct_assignment: ∀ x y, x ≠ y → assign x ≠ assign y), 
      (assign 'A = Position1 ∧ assign 'B = Position1)) →
  (1 / 80) := by
sorry

end probability_A_and_B_at_Position_1_l73_73352


namespace mutually_exclusive_not_complementary_l73_73790

-- Definition for boys and girls
def boys : ℕ := 3
def girls : ℕ := 2

-- Definition for the event "exactly 1 boy selected" and "exactly 2 boys selected"
def event_exactly_one_boy_selected (selected : ℕ) : Prop :=
  selected = 1

def event_exactly_two_boys_selected (selected : ℕ) : Prop :=
  selected = 2

-- Main statement to be proved
theorem mutually_exclusive_not_complementary :
  ∀ (selected_boys selected_girls : ℕ),
    selected_boys + selected_girls = 2 →
    (event_exactly_one_boy_selected selected_boys ∧ event_exactly_two_boys_selected selected_boys) → False ∧
    ¬ (event_exactly_one_boy_selected selected_boys ∨ event_exactly_two_boys_selected selected_boys = 2) :=
by
  intros
  sorry

end mutually_exclusive_not_complementary_l73_73790


namespace find_eccentricity_l73_73286

def ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0

def point_on_ellipse (P : ℝ × ℝ) (a b : ℝ) : Prop := 
  ellipse a b ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)

def incenter_area_condition (P F1 F2 I : ℝ × ℝ) : Prop := 
  (S (I, F1, P) + S (I, P, F2) = 2 * S (I, F1, F2))

noncomputable def eccentricity (a b e : ℝ) : Prop := 
  ellipse a b ∧ e = 1 / 2
             
theorem find_eccentricity 
  (a b e : ℝ) (P F1 F2 I : ℝ × ℝ)
  (h_ellipse : ellipse a b)
  (h_point : point_on_ellipse P a b)
  (h_foci : F1 = (-c, 0) ∧ F2 = (c, 0) ∧ c = a * e)
  (h_incenter : incenter_area_condition P F1 F2 I) :
  eccentricity a b e :=
sorry

end find_eccentricity_l73_73286


namespace find_x_l73_73938

-- Definitions according to the conditions in the problem
def ten_pow_log_ten_sixteen : ℝ := 10^(Real.log 16 / Real.log 10)
def equation (x : ℝ) : Prop := ten_pow_log_ten_sixteen = 10 * x + 6

-- Statement to be proved
theorem find_x : ∃ x : ℝ, equation x ∧ x = 1 :=
by
  sorry -- Proof goes here

end find_x_l73_73938


namespace num_positive_perfect_square_sets_l73_73239

-- Define what it means for three numbers to form a set that sum to 100 
def is_positive_perfect_square_set (a b c : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ c ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 100

-- Define the main theorem to state there are exactly 4 such sets
theorem num_positive_perfect_square_sets : 
  {s : Finset (ℕ × ℕ × ℕ) // (∃ a b c, (a, b, c) ∈ s ∧ is_positive_perfect_square_set a b c) }.card = 4 :=
sorry

end num_positive_perfect_square_sets_l73_73239


namespace saving_rate_l73_73757

theorem saving_rate (initial_you : ℕ) (initial_friend : ℕ) (friend_save : ℕ) (weeks : ℕ) (final_amount : ℕ) :
  initial_you = 160 →
  initial_friend = 210 →
  friend_save = 5 →
  weeks = 25 →
  final_amount = initial_you + weeks * 7 →
  final_amount = initial_friend + weeks * friend_save →
  7 = (final_amount - initial_you) / weeks :=
by
  intros initial_you_val initial_friend_val friend_save_val weeks_val final_amount_equals_you final_amount_equals_friend
  rw [initial_you_val, initial_friend_val, friend_save_val, weeks_val] at *
  have h: 160 + 25 * 7 = 210 + 25 * 5, by sorry
  have final_amount_val: final_amount = 335, by
    rw [final_amount_equals_you]
    exact h
  rw [final_amount_val] at *
  exact sorry

end saving_rate_l73_73757


namespace total_sets_of_laces_l73_73049

theorem total_sets_of_laces :
  let members := [8, 12, 10, 15]
  let backups := [1, 2, 3, 4]
  let skates_needed := List.zipWith (λ m b => m * (1 + b)) members backups
  let total_skates := skates_needed.sum
  let sets_per_pair := 6
  total_skates * sets_per_pair = 1002 := 
by
  let members := [8, 12, 10, 15]
  let backups := [1, 2, 3, 4]
  let skates_needed := List.zipWith (λ m b => m * (1 + b)) members backups
  let total_skates := skates_needed.sum
  have H_total_skates : total_skates = 167 :=
    by simp [skates_needed, List.zipWith, List.sum]
  have H_sets_per_pair : sets_per_pair = 6 := by rfl
  calc 
    total_skates * sets_per_pair
        = 167 * 6 : by rw [H_total_skates, H_sets_per_pair]
    ... = 1002 : by norm_num


end total_sets_of_laces_l73_73049


namespace above_limit_l73_73996

/-- John travels 150 miles in 2 hours. -/
def john_travel_distance : ℝ := 150

/-- John travels for 2 hours. -/
def john_travel_time : ℝ := 2

/-- The speed limit is 60 mph. -/
def speed_limit : ℝ := 60

/-- The speed of John during his travel. -/
def john_speed : ℝ := john_travel_distance / john_travel_time

/-- How many mph above the speed limit was John driving? -/
def speed_above_limit : ℝ := john_speed - speed_limit

theorem above_limit : speed_above_limit = 15 := 
by
  unfold speed_above_limit john_speed john_travel_distance john_travel_time speed_limit
  have h1: 150 / 2 = 75 := by norm_num
  have h2: 75 - 60 = 15 := by norm_num
  rw [h1, h2]
  refl

end above_limit_l73_73996


namespace ratio_of_perimeters_l73_73571

def similar_triangles (A B C D E F : Point) := 
  sorry -- Similarity predicate definition

def similarity_ratio (ABC DEF : Triangle) := 1 / 2

theorem ratio_of_perimeters (ABC DEF : Triangle)
  (h1 : similar_triangles ABC DEF)
  (h2 : similarity_ratio ABC DEF = 1 / 2) :
  (perimeter ABC) / (perimeter DEF) = 1 / 2 := 
sorry

end ratio_of_perimeters_l73_73571


namespace polyhedron_converge_to_constant_l73_73469

-- Define the properties of the polyhedron
structure Polyhedron :=
(vertices : Finset ℕ)
(neighbors : ℕ → ℕ → Prop)
(face : ∀ {i j : ℕ}, neighbors i j → (vertices i ∧ vertices j))

-- Define the sequence v_k(n)
def v : ℕ → ℕ → ℤ := sorry

-- Given conditions
axiom P : Polyhedron
axiom initial_values : ∀ k, v k 0 ∈ Int
axiom average_condition : ∀ k n, v k (n + 1) = (∑ j in P.vertices, if P.neighbors k j then v j n else 0) / (P.vertices.card.toNat - 1)

-- Statement to prove
theorem polyhedron_converge_to_constant :
  ∃ N : ℕ, ∀ k l n : ℕ, n ≥ N → v k n = v l n :=
sorry

end polyhedron_converge_to_constant_l73_73469


namespace average_grade_two_years_l73_73436

theorem average_grade_two_years (courses_last_year courses_year_before : ℕ)
  (avg_grade_last_year avg_grade_year_before : ℝ)
  (H1 : courses_last_year = 6)
  (H2 : courses_year_before = 5)
  (H3 : avg_grade_last_year = 100)
  (H4 : avg_grade_year_before = 60) :
  Float.round (1 / (courses_last_year + courses_year_before) * 
              (courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year)) = 81.8 :=
by
  sorry

end average_grade_two_years_l73_73436


namespace original_profit_is_five_percent_l73_73027

def cost_price : ℝ := 600
def new_cost_price : ℝ := cost_price - (0.05 * cost_price)
def new_selling_price : ℝ := new_cost_price + (0.10 * new_cost_price)
def original_selling_price : ℝ := new_selling_price + 3
def original_profit_percentage (CP SP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem original_profit_is_five_percent : original_profit_percentage cost_price original_selling_price = 5 := 
by
  sorry

end original_profit_is_five_percent_l73_73027


namespace acute_angle_locus_obtuse_angle_locus_l73_73101

-- Define the point in a Euclidean space
structure Point (ℝ : Type) :=
(x : ℝ)
(y : ℝ)

-- Define the line segment as two points
structure LineSegment (ℝ : Type) :=
(A : Point ℝ)
(B : Point ℝ)

-- Define the condition that Point lies on the circle with the diameter AB
def is_on_circle_diameter (M : Point ℝ) (AB : LineSegment ℝ) : Prop :=
  let d := (AB.B.x - AB.A.x) * (AB.B.x - AB.A.x) + (AB.B.y - AB.A.y) * (AB.B.y - AB.A.y)
  in (M.x - AB.A.x) * (M.x - AB.B.x) + (M.y - AB.A.y) * (M.y - AB.B.y) = 0

-- Define the locus for acute angle
def locus_acute_angle (M : Point ℝ) (AB : LineSegment ℝ) : Prop :=
  ¬is_on_circle_diameter M AB ∧ ¬(M.y = AB.A.y ∧ M.y = AB.B.y) -- Outside circle with diameter AB excluding line AB

-- Define the locus for obtuse angle
def locus_obtuse_angle (M : Point ℝ) (AB : LineSegment ℝ) : Prop :=
  is_on_circle_diameter M AB ∧ ¬(M.y = AB.A.y ∧ M.y = AB.B.y) -- Inside circle with diameter AB excluding segment AB

-- Theorem stating the locus for acute angle
theorem acute_angle_locus (M : Point ℝ) (AB : LineSegment ℝ) :
  locus_acute_angle M AB := sorry

-- Theorem stating the locus for obtuse angle
theorem obtuse_angle_locus (M : Point ℝ) (AB : LineSegment ℝ) :
  locus_obtuse_angle M AB := sorry

end acute_angle_locus_obtuse_angle_locus_l73_73101


namespace equilateral_triangle_of_complex_numbers_l73_73356

theorem equilateral_triangle_of_complex_numbers
  (z1 z2 z3 : ℂ)
  (h1 : z1 + z2 + z3 = 0)
  (h2 : abs z1 = abs z2 ∧ abs z2 = abs z3) :
  abs (z2 - z1) = abs (z3 - z2) ∧ abs (z3 - z2) = abs (z1 - z3) := by
  sorry

end equilateral_triangle_of_complex_numbers_l73_73356


namespace median_after_insertion_l73_73984

def insert_into_set_and_sort (s : List ℝ) (x : ℝ) : List ℝ :=
  (x :: s).sort

def find_median (s : List ℝ) : ℝ :=
  s.nthLe (s.length / 2) (by apply Nat.div_lt_of_lt_self; linarith)

theorem median_after_insertion :
  find_median (insert_into_set_and_sort [8, 46, 53, 127] 14.11111111111111) = 46 := 
by
  sorry

end median_after_insertion_l73_73984


namespace largest_circle_diameter_l73_73034

theorem largest_circle_diameter
  (A : ℝ) (hA : A = 180)
  (w l : ℝ) (hw : l = 3 * w)
  (hA2 : w * l = A) :
  ∃ d : ℝ, d = 16 * Real.sqrt 15 / Real.pi :=
by
  sorry

end largest_circle_diameter_l73_73034


namespace min_value_sequence_l73_73557

theorem min_value_sequence (a : ℕ → ℕ) (h1 : a 2 = 102) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) - a n = 4 * n) : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (a m) / m ≥ 26) :=
sorry

end min_value_sequence_l73_73557


namespace arithmetic_sequence_product_l73_73275

theorem arithmetic_sequence_product 
  (b : ℕ → ℤ) 
  (h_arith : ∀ n, b n = b 0 + (n : ℤ) * (b 1 - b 0))
  (h_inc : ∀ n, b n ≤ b (n + 1))
  (h4_5 : b 4 * b 5 = 21) : 
  b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := 
by 
  sorry

end arithmetic_sequence_product_l73_73275


namespace total_length_remaining_l73_73808

def initial_figure_height : ℕ := 10
def initial_figure_width : ℕ := 7
def top_right_removed : ℕ := 2
def middle_left_removed : ℕ := 2
def bottom_removed : ℕ := 3
def near_top_left_removed : ℕ := 1

def remaining_top_length : ℕ := initial_figure_width - top_right_removed
def remaining_left_length : ℕ := initial_figure_height - middle_left_removed
def remaining_bottom_length : ℕ := initial_figure_width - bottom_removed
def remaining_right_length : ℕ := initial_figure_height - near_top_left_removed

theorem total_length_remaining :
  remaining_top_length + remaining_left_length + remaining_bottom_length + remaining_right_length = 26 := by
  sorry

end total_length_remaining_l73_73808


namespace courses_selection_l73_73578

-- Definition of the problem
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of ways person A can choose 2 courses from 4
def total_ways : ℕ := C 4 2 * C 4 2

-- Number of ways both choose exactly the same courses
def same_ways : ℕ := C 4 2

-- Prove the number of ways they can choose such that there is at least one course different
theorem courses_selection :
  total_ways - same_ways = 30 := by
  sorry

end courses_selection_l73_73578


namespace union_A_B_complement_A_and_B_A_intersect_C_nonempty_range_l73_73125

def setA : set ℝ := {x | 4 ≤ x ∧ x < 8}
def setB : set ℝ := {x | 5 < x ∧ x < 10}
def setC (a : ℝ) : set ℝ := {x | x > a}

theorem union_A_B : setA ∪ setB = {x | 4 ≤ x ∧ x < 10} :=
by {
  -- proof goes here
  sorry
}

theorem complement_A_and_B : (set.univ \ setA) ∩ setB = {x | 8 ≤ x ∧ x < 10} :=
by {
  -- proof goes here
  sorry
}

theorem A_intersect_C_nonempty_range (a : ℝ) : (setA ∩ setC a) ≠ ∅ → a < 8 :=
by {
  -- proof goes here
  sorry
}

end union_A_B_complement_A_and_B_A_intersect_C_nonempty_range_l73_73125


namespace find_domain_point_l73_73657

noncomputable def g1 (x : ℝ) : ℝ := real.sqrt (2 - x)
noncomputable def gn (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then g1 x else g1 (real.sqrt (nat.succ n ^ 2 + 2 - x))

theorem find_domain_point :
  ∃ M d : ℕ, (∀ x, 2 ≤ x ∧ x ≤ d ↔ gn M x ∈ set.range gn) ∧ d = 34 ∧ (∀ n > M, ¬(∃ y, gn n y ∈ set.range gn)) :=
begin
  sorry
end

end find_domain_point_l73_73657


namespace unit_cube_surface_coverage_l73_73991

theorem unit_cube_surface_coverage :
  ∃ (triangles : ℕ) (area_per_triangle : ℝ),
  let total_area := 6 in
  let area_per_triangle := 1.5 in
  triangles = 4 ∧
  area_per_triangle * triangles = total_area := by
  sorry

end unit_cube_surface_coverage_l73_73991


namespace original_water_bottles_l73_73663

variables (x : ℕ)
constants (p : ℝ) (r : ℝ) (s : ℝ)

-- Given Conditions
axiom original_price : p = 2
axiom reduced_price : r = 1.85
axiom shortfall : s = 9

-- The proof problem
theorem original_water_bottles : 2 * x - 1.85 * x = 9 → x = 60 :=
by
  intro h
  sorry

end original_water_bottles_l73_73663


namespace find_k_values_find_triples_k_3_find_triples_k_1_l73_73763

-- Definition of the problem
def satisfies_equation (k : ℤ) (x y z : ℕ) : Prop :=
  x^2 + y^2 + z^2 = k * x * y * z

-- Part (a) - Prove integer values of k
theorem find_k_values (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) : 
  ∃ k : ℤ, k = 1 ∨ k = 3 :=
by
  sorry

-- Part (b) - Find all possible triples for k = 3
theorem find_triples_k_3 (x y z : ℕ) (h_bound : x < 1000 ∧ y < 1000 ∧ z < 1000) : 
  satisfies_equation 3 x y z ↔ (x, y, z) ∈ {(1, 1, 1), (1, 1, 2), (1, 2, 5), (1, 5, 13), (2, 5, 29), (5, 29, 169)} :=
by
  sorry

-- Find all possible triples for k = 1
theorem find_triples_k_1 (x y z : ℕ) (h_bound : x < 1000 ∧ y < 1000 ∧ z < 1000) : 
  satisfies_equation 1 x y z ↔ (x, y, z) ∈ {(3, 3, 3), (3, 3, 6), (3, 6, 15), (6, 15, 39), (6, 15, 87)} :=
by
  sorry

end find_k_values_find_triples_k_3_find_triples_k_1_l73_73763


namespace sum_double_nested_series_l73_73073

theorem sum_double_nested_series :
  (∑ j : ℕ, ∑ k : ℕ, (2 : ℝ)^(-4 * k - 2 * j - (k + j)^2)) = (4 / 3 : ℝ) :=
by
  sorry

end sum_double_nested_series_l73_73073


namespace expected_diff_students_per_class_l73_73804

theorem expected_diff_students_per_class :
  let num_students := 100
  let num_classes := 5
  let num_teachers := 5
  let student_distrib := [40, 40, 10, 5, 5]
  let t := (student_distrib.map (λ x => x / num_classes)).sum
  let s := (student_distrib.map (λ x => x * (x / num_students))).sum
  t - s = -13.5 :=
by
  let num_students := 100
  let num_classes := 5
  let num_teachers := 5
  let student_distrib := [40, 40, 10, 5, 5]
  let t := (student_distrib.map (λ x => x / num_classes)).sum
  let s := (student_distrib.map (λ x => x * (x / num_students))).sum
  have ht : t = (40 / 5 + 40 / 5 + 10 / 5 + 5 / 5 + 5 / 5) := by sorry
  have hs : s = (40 * (40 / 100) + 40 * (40 / 100) + 10 * (10 / 100) + 5 * (5 / 100) + 5 * (5 / 100)) := by sorry
  have heq : t - s = -13.5 := by sorry
  exact heq

end expected_diff_students_per_class_l73_73804


namespace fraction_inequality_l73_73535

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) : 
  (b / a) > ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l73_73535


namespace jane_reroll_two_dice_l73_73634

noncomputable def probability_reroll_two_dice : ℚ :=
  7 / 18

theorem jane_reroll_two_dice :
  ∀ (d1 d2 d3 : ℕ),
  (d1 ∈ {1, 2, 3, 4, 5, 6}) ∧
  (d2 ∈ {1, 2, 3, 4, 5, 6}) ∧
  (d3 ∈ {1, 2, 3, 4, 5, 6}) →
  -- Jane rerolls to optimize the chances of the sum being 9
  let reroll_strategy := -- define the optimal reroll strategy here
  (probability_reroll_any_two_dice reroll_strategy) = probability_reroll_two_dice
:= sorry

end jane_reroll_two_dice_l73_73634


namespace total_cost_38_pencils_56_pens_l73_73433

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end total_cost_38_pencils_56_pens_l73_73433


namespace determine_m_l73_73917

noncomputable def f (m x : ℝ) := (m^2 - m - 1) * x^(-5 * m - 3)

theorem determine_m : ∃ m : ℝ, (∀ x > 0, f m x = (m^2 - m - 1) * x^(-5 * m - 3)) ∧ (∀ x > 0, (m^2 - m - 1) * x^(-(5 * m + 3)) = (m^2 - m - 1) * x^(-5 * m - 3) → -5 * m - 3 > 0) ∧ m = -1 :=
by
  sorry

end determine_m_l73_73917


namespace dave_unmanaged_items_l73_73081

noncomputable def total_unmanaged_items 
  (shortSleeveShirts : ℕ) 
  (longSleeveShirts : ℕ) 
  (totalSocks : ℕ) 
  (totalHandkerchiefs : ℕ) 
  (managedShirts : ℕ) 
  (managedSocks : ℕ) 
  (managedHandkerchiefs : ℕ) : ℕ :=
  let totalShirts := shortSleeveShirts + longSleeveShirts
  let unmanagedShirts := totalShirts - managedShirts
  let unmanagedSocks := totalSocks - managedSocks
  let unmanagedHandkerchiefs := totalHandkerchiefs - managedHandkerchiefs
  unmanagedShirts + unmanagedSocks + unmanagedHandkerchiefs

theorem dave_unmanaged_items 
  (shortSleeveShirts : 9)
  (longSleeveShirts : 27)
  (totalSocks : 50)
  (totalHandkerchiefs : 34)
  (managedShirts : 20)
  (managedSocks : 30)
  (managedHandkerchiefs : 16) :
  total_unmanaged_items shortSleeveShirts longSleeveShirts totalSocks totalHandkerchiefs managedShirts managedSocks managedHandkerchiefs = 54 :=
by
  -- This is where the proof would go
  sorry

end dave_unmanaged_items_l73_73081


namespace perfect_squares_modulo_nine_l73_73719

-- Definitions
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Problem statement
theorem perfect_squares_modulo_nine (a b c : ℕ) (h1 : is_perfect_square a)
                                    (h2 : is_perfect_square b) (h3 : is_perfect_square c)
                                    (h4 : (a + b + c) % 9 = 0) :
  ∃ i j ∈ ({a, b, c} : set ℕ), a % 9 = b % 9 ∧ i ≠ j :=
begin
  sorry
end

end perfect_squares_modulo_nine_l73_73719


namespace sum_of_integers_101_to_110_l73_73464

theorem sum_of_integers_101_to_110 : ∑ i in (finset.Icc 101 110), i = 1055 := by
  sorry

end sum_of_integers_101_to_110_l73_73464


namespace number_of_solutions_l73_73467

theorem number_of_solutions :
  ∃ (sols : Finset (ℝ × ℝ × ℝ × ℝ)), 
  (∀ (x y z w : ℝ), ((x, y, z, w) ∈ sols) ↔ (x = z + w + z * w * x ∧ y = w + x + w * x * y ∧ z = x + y + x * y * z ∧ w = y + z + y * z * w ∧ x * y + y * z + z * w + w * x = 2)) ∧ 
  sols.card = 5 :=
sorry

end number_of_solutions_l73_73467


namespace area_of_playground_l73_73337

variable (l w : ℝ)

-- Conditions:
def perimeter_eq : Prop := 2 * l + 2 * w = 90
def length_three_times_width : Prop := l = 3 * w

-- Theorem:
theorem area_of_playground (h1 : perimeter_eq l w) (h2 : length_three_times_width l w) : l * w = 379.6875 :=
  sorry

end area_of_playground_l73_73337


namespace total_animals_l73_73169

def pigs : ℕ := 10

def cows : ℕ := 2 * pigs - 3

def goats : ℕ := cows + 6

theorem total_animals : pigs + cows + goats = 50 := by
  sorry

end total_animals_l73_73169


namespace guards_can_protect_point_l73_73629

-- Define the conditions of the problem as Lean definitions
def guardVisionRadius : ℝ := 100

-- Define the proof statement
theorem guards_can_protect_point :
  ∃ (num_guards : ℕ), num_guards * 45 = 360 ∧ guardVisionRadius = 100 :=
by
  sorry

end guards_can_protect_point_l73_73629


namespace infinite_solutions_ari_prog_l73_73305

structure AriProgSol (a b c d x y : ℤ) : Prop :=
  (arith_prog : a = b - 2 * d ∧ c = b + 2 * d)
  (eq1 : a + b + c = x + y)
  (eq2 : a^3 + b^3 + c^3 = x^3 + y^3)

theorem infinite_solutions_ari_prog :
  ∃ (a b c d x y : ℤ), ∃ f : ℕ → (AriProgSol a b c d x y), function.injective f := sorry

end infinite_solutions_ari_prog_l73_73305


namespace number_of_zeros_sequence_inequality_l73_73923

-- Given definitions
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + 1 - a * x

-- Part 1
theorem number_of_zeros (a : ℝ) : 
  (a < 1 → ∀ x > 0, f x a ≠ 0) ∧ 
  (a = 1 → ∃! x, x > 0 ∧ f x a = 0) ∧ 
  (a > 1 → (∃ x1 > 0, f x1 a = 0) ∧ (∃ x2 > 0, f x2 a = 0) ∧ (x1 ≠ x2)) := 
sorry 

-- Sequence definition
def seq (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := 
  if n = 1 then 2 / 3 
  else Real.log ((a (n-1) + 1) / 2) + 1

-- Part 2
theorem sequence_inequality : 
  ∀ n : ℕ+, ∑ i in Finset.range (n + 1), 1 / (seq seq i.succ) < n + 1 := 
sorry

end number_of_zeros_sequence_inequality_l73_73923


namespace jerry_remaining_money_l73_73641

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l73_73641


namespace false_statements_l73_73043

-- Define conditions
def y1 (k : ℝ) (x : ℝ) : ℝ := (Real.cos (k * x))^2 - (Real.sin (k * x))^2
def y2 (a : ℝ) (x : ℝ) (y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0 ∧ 3 * x + (a - 1) * y = a - 7
def y3 (x : ℝ) : ℝ := (x^2 + 4) / Real.sqrt (x^2 + 3)

-- Theorem stating the false nature of the statements
theorem false_statements : 
  (∀ k : ℝ, minimal_period (y1 k) π → k = 1) = false ∧ 
  (∀ a : ℝ, y2 a x y → a = 3) = false ∧ 
  (∀ x : ℝ, y3 x ≥ 2) = false := 
sorry

end false_statements_l73_73043


namespace largest_power_of_2_in_e_p_l73_73380

noncomputable def p : ℝ :=
  (Σ k in Finset.range 7, k * Real.log k)

theorem largest_power_of_2_in_e_p :
  (∃ n : ℕ, e^p = 2^n) ∧ (∀ n : ℕ, (n < 16 ∧ 2^n ∣ e^p) → false) :=
begin
  sorry
end

end largest_power_of_2_in_e_p_l73_73380


namespace largest_positive_x_l73_73100

def largest_positive_solution : ℝ := 1

theorem largest_positive_x 
  (x : ℝ) 
  (h : (2 * x^3 - x^2 - x + 1) ^ (1 + 1 / (2 * x + 1)) = 1) : 
  x ≤ largest_positive_solution := 
sorry

end largest_positive_x_l73_73100


namespace necessary_but_not_sufficient_condition_l73_73893

theorem necessary_but_not_sufficient_condition (x y : ℝ) (hx : x < y) (hq : log x < log y) : 
  (2 ^ x < 2 ^ y) ∧ (log x < log y) ∧ (2 ^ x < 2 ^ y → x < y) := 
  sorry

end necessary_but_not_sufficient_condition_l73_73893


namespace line_equation_of_circle_intersect_l73_73883

-- Define the circle equation
def circle_C (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 5) ^ 2 = 5

-- Conditions: center of the circle, line passing through center intersects the y-axis, midpoint
def line_through_center (l : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, l (x) = k * (x - 3) + 5

def intersects_y_axis (l : ℝ → ℝ) : Prop :=
  ∃ y₀ : ℝ, l 0 = y₀

def midpoint_PB (P A B : ℝ × ℝ) : Prop :=
  (P.1 + B.1) / 2 = A.1 ∧ (P.2 + B.2) / 2 = A.2

-- Main proof problem statement
theorem line_equation_of_circle_intersect :
  ∃ k : ℝ, ∀ x y : ℝ,
    circle_C x y →
    line_through_center (fun x => k * (x - 3) + 5) →
    intersects_y_axis (fun x => k * (x - 3) + 5) →
    (k = 2 ∧ line_eq (2x - y - 1 = 0)) ∨ (k = -2 ∧ line_eq (2x + y - 11 = 0)) :=
sorry

end line_equation_of_circle_intersect_l73_73883


namespace probability_event_A_l73_73820

def unit_square := set.Icc (0 : ℝ) 1 ×ˢ set.Icc (0 : ℝ) 1

def event_A (x y : ℝ) : Prop :=
  1 ≤ x + y ∧ x + y < 1.5

noncomputable def area_event_A : ℝ :=
  3 / 8

theorem probability_event_A :
  ∫ (x : ℝ) in 0..1, ∫ (y : ℝ) in 0..1, if event_A x y then 1 else 0 = 0.375 :=
begin
  sorry
end

end probability_event_A_l73_73820


namespace eisenstein_criterion_l73_73403

theorem eisenstein_criterion (f : Polynomial ℤ) (p : ℕ) [hp : Prime p] :
  (∀ i < f.natDegree, i < f.natDegree → p ∣ f.coeff i) ∧ 
  ¬ p ∣ f.leadingCoeff ∧ 
  ¬ p ^ 2 ∣ f.coeff 0 → 
  ¬ ∃ g h : Polynomial ℚ, g * h = f ∧ g.natDegree ≠ 0 ∧ h.natDegree ≠ 0 := 
by
  sorry

end eisenstein_criterion_l73_73403


namespace find_angle_A_l73_73596

theorem find_angle_A {α β γ : Type} [LinearOrderedField γ] [TrigRing γ] :
  (a b : γ) (angle_B : γ) 
  (h1 : a = Real.sqrt 3) 
  (h2 : b = Real.sqrt 2) 
  (h3 : angle_B = Real.pi / 4) :
  ∃ angle_A : γ, angle_A = Real.pi / 3 ∨ angle_A = 2 * Real.pi / 3 :=
by
  sorry

end find_angle_A_l73_73596


namespace tallest_tree_height_l73_73720

variable (T M S : ℝ)

def middle_tree_height (T : ℝ) : ℝ := (T / 2) - 6
def smallest_tree_height (M : ℝ) : ℝ := M / 4

theorem tallest_tree_height (T : ℝ) (h1 : S = 12) (h2 : smallest_tree_height (middle_tree_height T) = S) : T = 108 :=
by
  sorry

end tallest_tree_height_l73_73720


namespace calculation_result_l73_73057

theorem calculation_result : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end calculation_result_l73_73057


namespace functional_equation_solution_l73_73493

theorem functional_equation_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  f = id ∨ f = λ x, -x :=
by
  sorry

end functional_equation_solution_l73_73493


namespace area_triangle_polar_coordinates_l73_73441

theorem area_triangle_polar_coordinates :
  let P1 := (5 : ℝ, 109 : ℝ)
  let P2 := (4 : ℝ, 49 : ℝ)
  let theta1 := P1.2 * (Real.pi / 180)
  let theta2 := P2.2 * (Real.pi / 180)
  let r1 := P1.1
  let r2 := P2.1
  let delta_theta := theta2 - theta1
  (1 / 2) * r1 * r2 * Real.sin delta_theta = 5 * Real.sqrt 3 := 
by
  sorry

end area_triangle_polar_coordinates_l73_73441


namespace tunnel_length_is_correct_l73_73022

noncomputable def length_of_tunnel {train_length speed_kmh : ℝ} (crossing_time_min : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  let time_sec := crossing_time_min * 60
  let distance_covered := speed_ms * time_sec
  distance_covered - train_length

theorem tunnel_length_is_correct :
  length_of_tunnel 800 78 1 = 500.2 :=
by
  sorry

end tunnel_length_is_correct_l73_73022


namespace find_f_of_sum_m_n_l73_73137

variable (m n : ℝ)
def f (x : ℝ) := Real.log x / Real.log 3 + 6

noncomputable def finverse (y : ℝ) := 3^y - 6

theorem find_f_of_sum_m_n (h : (finverse m + 6) * (finverse n + 6) = 27) : 
  f (m + n) = 2 := by
{
  sorry
}

end find_f_of_sum_m_n_l73_73137


namespace count_non_distinct_real_solution_pairs_l73_73109

theorem count_non_distinct_real_solution_pairs :
  { (b, c) : ℕ × ℕ // b > 0 ∧ c > 0 ∧ b^2 ≤ 4 * c ∧ c^2 ≤ 4 * b }.card = 6 :=
by
  -- Proof steps would go here, but are replaced by sorry
  sorry

end count_non_distinct_real_solution_pairs_l73_73109


namespace inappropriate_survey_method_l73_73451

/-
Parameters:
- A: Using a sampling survey method to understand the water-saving awareness of middle school students in the city (appropriate).
- B: Investigating the capital city to understand the environmental pollution situation of the entire province (inappropriate due to lack of representativeness).
- C: Investigating the audience's evaluation of a movie by surveying those seated in odd-numbered seats (appropriate).
- D: Using a census method to understand the compliance rate of pilots' vision (appropriate).
-/

theorem inappropriate_survey_method (A B C D : Prop) 
  (hA : A = true)
  (hB : B = false)  -- This condition defines B as inappropriate
  (hC : C = true)
  (hD : D = true) : B = false :=
sorry

end inappropriate_survey_method_l73_73451


namespace triangular_section_smaller_than_tetrahedron_face_l73_73303

theorem triangular_section_smaller_than_tetrahedron_face (T : Tetrahedron) (S : Triangle) 
  (hS : S ⊆ T) : ∃ F ∈ faces T, area S < area F :=
sorry

end triangular_section_smaller_than_tetrahedron_face_l73_73303


namespace angle_BAC_measure_l73_73625

theorem angle_BAC_measure
  {A B C X Y : Type}
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space X]
  [metric_space Y]
  (h1 : dist A X = dist X Y)
  (h2 : dist X Y = dist Y B)
  (h3 : ∠ B C = 150)
  (h4 : BC > A dist C)
  (h5 : dist B C = dist X Y + dist Y B)
  : ∠ A B C = 120 := 
sorry

end angle_BAC_measure_l73_73625


namespace symmetry_point_exists_l73_73809

noncomputable def symmetric_planes {α : Type*} [metric_space α] (H₁ : α)
  (g : line α) (system24 system45 : set (plane α)) : Prop :=
let sphere := metric.sphere H₁ (dist H₁ g) in
∀ (P₄ P₅ : plane α), 
  P₄ ∈ system24 ∧ P₅ ∈ system45 → 
  P₄ ⊆ sphere ∧ P₅ ⊆ sphere

theorem symmetry_point_exists {α : Type*} [metric_space α]
  (H₁ : α) (g : line α) (system24 system45 : set (plane α)) :
  symmetric_planes H₁ g system24 system45 →
  ∃ (n : ℕ), n = 4 :=
sorry

end symmetry_point_exists_l73_73809


namespace cosine_value_smallest_angle_l73_73143

theorem cosine_value_smallest_angle :
  ∀ (a : ℝ), (a + 3 > 0) → (a + 6 > 0) → 
    (let A := Math.atan2 ((a + 3) * (a + 3) + (a + 6) * (a + 6) - a * a) (2 * (a + 3) * (a + 6)) in
    let C := 2 * A in
    cos A = (a + 6) / (2 * a)) ↔ cos A = 3 / 4 := sorry

end cosine_value_smallest_angle_l73_73143


namespace emily_spent_12_dollars_l73_73818

variables (cost_per_flower : ℕ)
variables (roses : ℕ)
variables (daisies : ℕ)

def total_flowers : ℕ := roses + daisies

def total_cost : ℕ := total_flowers * cost_per_flower

theorem emily_spent_12_dollars (h1 : cost_per_flower = 3)
                              (h2 : roses = 2)
                              (h3 : daisies = 2) :
  total_cost cost_per_flower roses daisies = 12 :=
by
  simp [total_cost, total_flowers, h1, h2, h3]
  sorry

end emily_spent_12_dollars_l73_73818


namespace hyperbola_standard_equation_l73_73103

theorem hyperbola_standard_equation :
  ∃ (a b : ℝ), (2 * b = 12) ∧ ((10 / 8 = (5 / 4) ∨ (13 / 12 = (5 / 4))) ∧ (2 * c = 26) ∧ 
    (12^2 / b^2 - 12 = 1 ∨ M(0,12) on hyperbola) ∧ 
    ((P(-3,2 * sqrt 7) on hyperbola) ∧ (Q(-6 * sqrt 2,-7) on hyperbola))) ∧ 
  standard_equation = (y^2 / 25 - x^2 / 75 = 1) :=
by
  sorry

end hyperbola_standard_equation_l73_73103


namespace grandfather_older_than_grandmother_l73_73668

noncomputable def Milena_age : ℕ := 7

noncomputable def Grandmother_age : ℕ := Milena_age * 9

noncomputable def Grandfather_age : ℕ := Milena_age + 58

theorem grandfather_older_than_grandmother :
  Grandfather_age - Grandmother_age = 2 := by
  sorry

end grandfather_older_than_grandmother_l73_73668


namespace jeremy_school_distance_l73_73261

def travel_time_rush_hour := 15 / 60 -- hours
def travel_time_clear_day := 10 / 60 -- hours
def speed_increase := 20 -- miles per hour

def distance_to_school (d v : ℝ) : Prop :=
  d = v * travel_time_rush_hour ∧ d = (v + speed_increase) * travel_time_clear_day

theorem jeremy_school_distance (d v : ℝ) (h_speed : v = 40) : d = 10 :=
by
  have travel_time_rush_hour := 1/4
  have travel_time_clear_day := 1/6
  have speed_increase := 20
  
  have h1 : d = v * travel_time_rush_hour := by sorry
  have h2 : d = (v + speed_increase) * travel_time_clear_day := by sorry
  have eqn := distance_to_school d v
  sorry

end jeremy_school_distance_l73_73261


namespace minimum_subsidy_needed_lowest_average_processing_cost_volume_l73_73415

def processing_cost (x : ℝ) : ℝ :=
if x < 120 then 0 else
if x < 144 then (1/3) * x^3 - 80 * x^2 + 5040 * x else
if x < 500 then (1/2) * x^2 - 200 * x + 80000 else 0

def value_generated_per_ton : ℝ := 200

theorem minimum_subsidy_needed :
  ∃ subsidy, ∀ x, 200 ≤ x ∧ x ≤ 300 → (value_generated_per_ton * x - processing_cost x + subsidy ≥ 0)
  ∧ subsidy = 5000 := sorry

theorem lowest_average_processing_cost_volume :
  ∃ x, (120 ≤ x ∧ x < 144 → (processing_cost x / x = (1/3) * (x - 120)^2 + 240) ∧ x = 120)
  ∧ (144 ≤ x ∧ x < 500 → processing_cost x / x ≥ 300) := sorry

end minimum_subsidy_needed_lowest_average_processing_cost_volume_l73_73415


namespace savings_equal_after_25_weeks_l73_73756

theorem savings_equal_after_25_weeks (x : ℝ) :
  (160 + 25 * x = 210 + 125) → x = 7 :=
by 
  apply sorry

end savings_equal_after_25_weeks_l73_73756


namespace count_zhonghuan_numbers_l73_73743

def is_zonghuan (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10 * k + 1 ∧ ¬∃ a b : ℕ, (a > 0 ∧ b > 0 ∧ 10 * a + 1 < n ∧ 10 * b + 1 < n ∧ 10 * a + 1 > 1 ∧ 10 * b + 1 > 1 ∧ 10 * a + 1 * 10 * b + 1 = n)

def count_zonghuan_up_to_991 : ℕ :=
  ∑ i in finset.range 100 | is_zonghuan (10 * (i + 1) + 1), 1

theorem count_zhonghuan_numbers : count_zonghuan_up_to_991 = 87 :=
sorry

end count_zhonghuan_numbers_l73_73743


namespace square_axes_of_symmetry_l73_73035

def is_symmetry_axis (L : ℝ → ℝ → Prop) (S : ℝ → ℝ → Prop) : Prop :=
∀ x₁ y₁ x₂ y₂, S x₁ y₁ ↔ S x₂ y₂ → L x₁ y₁ ↔ L x₂ y₂

def square : Prop :=
exists a : ℝ, ∀ x y, (0 ≤ x ∧ x ≤ a) ∧ (0 ≤ y ∧ y ≤ a)

theorem square_axes_of_symmetry : square → ∃ n : ℕ, n = 4 :=
by
  intro hsquare
  sorry

end square_axes_of_symmetry_l73_73035


namespace complement_intersection_eq_l73_73507

open Set

theorem complement_intersection_eq :
  let U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {5, 6, 7}
  (U \ A ∩ U \ B) = {4, 8} :=
by
  let U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {5, 6, 7}
  sorry

end complement_intersection_eq_l73_73507


namespace find_domain_l73_73153

def a_pos_and_ne_1 (a : ℝ) : Prop := a > 0 ∧ a ≠ 1

def domain_F (x a : ℝ) (h : a_pos_and_ne_1 a) : Prop :=
  -1 / 3 < x ∧ x < 1 / 3

theorem find_domain {a : ℝ} (h : a_pos_and_ne_1 a) :
    ∀ x : ℝ, domain_F x a h :=
begin
  sorry
end

end find_domain_l73_73153


namespace domain_f_max_value_f_inequality_holds_l73_73150

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log a (1 - x) + log a (x + 2)

theorem domain_f (a : ℝ) (h : a > 1) : set.Ioo (-2 : ℝ) 1 = {x : ℝ | 1 - x > 0 ∧ x + 2 > 0 } :=
by sorry

theorem max_value_f (a : ℝ) (h : a > 1) (h_max : (∀ x ∈ set.Ioo (-2 : ℝ) 1, f x a ≤ 2) ∧ ∃ x, f x a = 2) :
  a = 3/2 :=
by sorry

theorem inequality_holds (m : ℝ) (h : ∀ x ∈ set.Ioo (-2 : ℝ) 1, x^2 - m * x + m + 1 > 0) : 
  m > 2 - 2 * real.sqrt 2 :=
by sorry

end domain_f_max_value_f_inequality_holds_l73_73150


namespace distinct_distances_lower_bound_l73_73296

-- Given conditions
variables {n k : ℕ}
variables (dist : ℕ → ℕ → ℝ)
variables (P : Fin n → Point)

-- Definition of distinct distances condition
def distinct_distances (dist : ℕ → ℕ → ℝ) (P : Fin n → Point) : ℕ :=
  -- Placeholder definition to count distinct distances among points P_i P_j
  sorry

-- The theorem to prove
theorem distinct_distances_lower_bound
  (h1 : 2 ≤ n)
  (h2 : k = distinct_distances dist P)
  : k ≥ sqrt (n - 3 / 4) - 1 / 2 := sorry

end distinct_distances_lower_bound_l73_73296


namespace smallest_D_l73_73978

theorem smallest_D {A B C D : ℕ} (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h2 : (A * 100 + B * 10 + C) * B = D * 1000 + C * 100 + B * 10 + D) : 
  D = 1 :=
sorry

end smallest_D_l73_73978


namespace clock_malfunction_fraction_correct_l73_73799

theorem clock_malfunction_fraction_correct : 
  let hours_total := 24
  let hours_incorrect := 6
  let minutes_total := 60
  let minutes_incorrect := 6
  let fraction_correct_hours := (hours_total - hours_incorrect) / hours_total
  let fraction_correct_minutes := (minutes_total - minutes_incorrect) / minutes_total
  (fraction_correct_hours * fraction_correct_minutes) = 27 / 40
:= 
by
  sorry

end clock_malfunction_fraction_correct_l73_73799


namespace jessica_mark_meet_time_jessica_mark_total_distance_l73_73262

noncomputable def jessica_start_time : ℚ := 7.75 -- 7:45 AM
noncomputable def mark_start_time : ℚ := 8.25 -- 8:15 AM
noncomputable def distance_between_towns : ℚ := 72
noncomputable def jessica_speed : ℚ := 14 -- miles per hour
noncomputable def mark_speed : ℚ := 18 -- miles per hour
noncomputable def t : ℚ := 81 / 32 -- time in hours when they meet

theorem jessica_mark_meet_time :
  7.75 + t = 10.28375 -- 10.17 hours in decimal
  :=
by
  -- Proof omitted.
  sorry

theorem jessica_mark_total_distance :
  jessica_speed * t + mark_speed * (t - (mark_start_time - jessica_start_time)) = distance_between_towns
  :=
by
  -- Proof omitted.
  sorry

end jessica_mark_meet_time_jessica_mark_total_distance_l73_73262


namespace protest_days_calculation_l73_73740

-- Given conditions
variables (num_cities : ℕ) (arrests_per_day : ℕ) 
          (days_before_trial : ℕ) (sentence_weeks : ℕ) 
          (total_weeks : ℕ)

-- Definitions based on the given conditions
def total_days := total_weeks * 7
def total_jail_days_per_person := days_before_trial + (sentence_weeks * 7 / 2)
def total_arrests := total_days / total_jail_days_per_person
def total_protest_days := total_arrests / arrests_per_day
def days_of_protest_per_city := total_protest_days / num_cities

-- Theorem statement based on the given conditions resulting in 30 days of protest per city
theorem protest_days_calculation (h1 : num_cities = 21) 
                                 (h2 : arrests_per_day = 10) 
                                 (h3 : days_before_trial = 4) 
                                 (h4 : sentence_weeks = 2) 
                                 (h5 : total_weeks = 9900) :
    days_of_protest_per_city num_cities arrests_per_day days_before_trial sentence_weeks total_weeks = 30 :=
by
  sorry

end protest_days_calculation_l73_73740


namespace correct_statements_l73_73450

-- Define the regression condition
def regression_condition (b : ℝ) : Prop := b < 0

-- Conditon ③: Event A is the complement of event B implies mutual exclusivity
def mutually_exclusive_and_complementary (A B : Prop) : Prop := 
  (A → ¬B) → (¬A ↔ B)

-- Main theorem combining the conditions and questions
theorem correct_statements: 
  (∀ b, regression_condition b ↔ (b < 0)) ∧
  (∀ A B, mutually_exclusive_and_complementary A B → (¬A ≠ B)) :=
by
  sorry

end correct_statements_l73_73450


namespace teresa_speed_l73_73313

def distance : ℝ := 25 -- kilometers
def time : ℝ := 5 -- hours

theorem teresa_speed :
  (distance / time) = 5 := by
  sorry

end teresa_speed_l73_73313


namespace set_union_l73_73661

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_union : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_union_l73_73661


namespace primes_with_3_as_ones_digit_below_200_l73_73178

theorem primes_with_3_as_ones_digit_below_200 : 
  (finset.filter (λ x, x % 10 = 3) (finset.filter nat.prime (finset.Ico 1 200))).card = 12 :=
sorry

end primes_with_3_as_ones_digit_below_200_l73_73178


namespace sequence_inequality_l73_73474

theorem sequence_inequality (x : ℕ → ℝ) (h₀ : x 0 = 1)
  (h_pos : ∀ n, x n > 0) (h_dec : ∀ n, x (n + 1) ≤ x n) :
  ∃ n ≥ 1, (∑ i in finset.range n, (x i)^2 / (x (i + 1))) ≥ 3.999 := 
sorry

end sequence_inequality_l73_73474


namespace cone_volume_l73_73208

theorem cone_volume (r h l : ℝ) (A π V : ℝ)
  (h_l : l = 2) 
  (h_A : A = π)
  (h_r : r = 1)
  (h_h : h = real.sqrt (l^2 - r^2))
  (h_π : π > 0):
  V = (1/3) * π * r^2 * h := 
by
  sorry

end cone_volume_l73_73208


namespace range_of_inclination_angle_l73_73579

theorem range_of_inclination_angle :
  ∀ (x : ℝ), (1 / 2 < (1 + sin (2 * x))) → 
  (1 ≤ (2 / (1 + sin (2 * x)))) → 
  (1 ≤ deriv (λ x => (2 * sin x) / (sin x + cos x)) x) →
  ∀ (P : ℝ), ∃ θ : ℝ, θ ∈ set.Ico (π / 4) (π / 2) :=
sorry

end range_of_inclination_angle_l73_73579


namespace sum_double_nested_series_l73_73074

theorem sum_double_nested_series :
  (∑ j : ℕ, ∑ k : ℕ, (2 : ℝ)^(-4 * k - 2 * j - (k + j)^2)) = (4 / 3 : ℝ) :=
by
  sorry

end sum_double_nested_series_l73_73074


namespace joan_number_of_games_l73_73642

open Nat

theorem joan_number_of_games (a b c d e : ℕ) (h_a : a = 10) (h_b : b = 12) (h_c : c = 6) (h_d : d = 9) (h_e : e = 4) :
  a + b + c + d + e = 41 :=
by
  sorry

end joan_number_of_games_l73_73642


namespace find_ellipse_equation_constant_AN_BM_l73_73529

section ellipse_proof

-- Given conditions
variables (a b : ℝ) (h1 : a > b) (h2 : b > 0)
variables (e : ℝ) (he : e = sqrt 3 / 2)
variables (area_ΔOAB : ℝ) (h3 : area_ΔOAB = 1)

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the points and the area constraints
def point_A := (a, 0)
def point_B := (0, b)
def point_O := (0, 0)
def triangle_area := (1 / 2) * a * b

-- Problem statements
-- 1. Prove the equation of the ellipse
theorem find_ellipse_equation : ellipse_equation a 4 1 :=
sorry

-- 2. Proving |AN| * |BM| is constant
variables {x₀ y₀ : ℝ} (hx₀y₀ : ellipse_equation x₀ y₀)

-- Parametrize PA and PB to find intersection points M and N
def line_PA := λ (x : ℝ), (y₀ / (x₀ - 2)) * (x - 2)
def line_PB := λ (x : ℝ), 1 + ((y₀ - 1) / x₀) * x

def point_My := line_PA 0
def point_Nx := line_PB 0

def AN := abs (2 + (x₀ / (y₀ - 1)))
def BM := abs (1 + (2 * y₀ / (x₀ - 2)))

-- Main theorem
theorem constant_AN_BM : AN * BM = 4 :=
sorry

end ellipse_proof

end find_ellipse_equation_constant_AN_BM_l73_73529


namespace solution_set_xf_x_pos_l73_73539

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_xf_x_pos :
  (∀ x : ℝ, f (-x) = -f x) →  -- odd function
  (∀ x y : ℝ, 0 < x → x < y → y → f x < f y) →  -- monotonically increasing for positive x
  f 3 = 0 →  -- f(log2 8) = f(3) = 0
  {x : ℝ | x * f x > 0} = {x : ℝ | x ∈ (-∞, -3) ∪ (3, +∞)} :=
by
  intros h_odd h_inc h_f3
  sorry

end solution_set_xf_x_pos_l73_73539


namespace inv_38_mod_53_l73_73126

theorem inv_38_mod_53 (h : (16 : ℤ) ⁻¹ ≡ 20 [ZMOD 53]) : (38 : ℤ) ⁻¹ ≡ 25 [ZMOD 53] :=
sorry

end inv_38_mod_53_l73_73126


namespace quadrilateral_area_proof_l73_73424

-- Definitions of points
def A : (ℝ × ℝ) := (1, 3)
def B : (ℝ × ℝ) := (1, 1)
def C : (ℝ × ℝ) := (3, 1)
def D : (ℝ × ℝ) := (2010, 2011)

-- Function to calculate the area of the quadrilateral
def area_of_quadrilateral (A B C D : (ℝ × ℝ)) : ℝ := 
  let area_triangle (P Q R : (ℝ × ℝ)) : ℝ := 
    0.5 * (P.1 * Q.2 + Q.1 * R.2 + R.1 * P.2 - P.2 * Q.1 - Q.2 * R.1 - R.2 * P.1)
  area_triangle A B C + area_triangle A C D

-- Lean statement to prove the desired area
theorem quadrilateral_area_proof : area_of_quadrilateral A B C D = 7 := 
  sorry

end quadrilateral_area_proof_l73_73424


namespace thermostat_range_l73_73788

theorem thermostat_range (T : ℝ) : 
  |T - 22| ≤ 6 ↔ 16 ≤ T ∧ T ≤ 28 := 
by sorry

end thermostat_range_l73_73788


namespace sum_double_series_eq_four_thirds_l73_73069

theorem sum_double_series_eq_four_thirds :
  (∑' j : ℕ, ∑' k : ℕ, 2^(- (4 * k + 2 * j + (k + j)^2))) = 4 / 3 :=
begin
  sorry
end

end sum_double_series_eq_four_thirds_l73_73069


namespace clock_angle_at_3_40_l73_73050

theorem clock_angle_at_3_40 (hour_hand_rate : ℕ → ℤ)
                            (minute_hand_rate : ℕ → ℤ)
                            (abs_diff : ℤ → ℤ → ℤ) :
  (hour_hand_rate 3 = 90) →
  (∀ m, hour_hand_rate m = 90 + (30 * m / 60)) →
  (minute_hand_rate 40 = 40 * 6) →
  (∀ a b, abs_diff a b = |a - b|) →
  abs_diff (minute_hand_rate 40) (hour_hand_rate 40) = 130 :=
by
  intro h₁ h₂ h₃ h₄
  rw [h₁, h₂ 40, h₃]
  norm_num at h₂
  rw h₄
  sorry

end clock_angle_at_3_40_l73_73050


namespace calculate_f_f_5_l73_73880

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^(x - 2) else Real.log (x - 1) / Real.log 2

theorem calculate_f_f_5 : f (f 5) = 1 :=
by 
  sorry

end calculate_f_f_5_l73_73880


namespace students_in_both_competitions_l73_73217

theorem students_in_both_competitions 
  (N A B neither : ℕ) 
  (hN : N = 50) 
  (hA : A = 32) 
  (hB : B = 24) 
  (hneither : neither = 3) 
  (hAtLeastOne : N - neither = 47) :
  A + B - (N - neither) = 9 :=
by
  rw [hN, hA, hB, hneither]
  sorry

end students_in_both_competitions_l73_73217


namespace Jason_attended_36_games_l73_73635

noncomputable def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (percentage_missed : ℕ) : ℕ :=
  let total_planned := planned_this_month + planned_last_month
  let missed_games := (percentage_missed * total_planned) / 100
  total_planned - missed_games

theorem Jason_attended_36_games :
  games_attended 24 36 40 = 36 :=
by
  sorry

end Jason_attended_36_games_l73_73635


namespace midpoint_XY_l73_73659

-- Defining the geometry problem
variable {Point : Type}

-- General definitions and assumptions
variable (A B C D M P X Q Y : Point)
variable (f_AB f_CD : Line Point)
variable (PD AB X M_ PM AC Q AB_ DQ : Line Segment Point)

-- Given conditions
axiom trapezoid (A B C D : Point) : parallel f_AB f_CD
axiom midpoint_AB (A B M : Point) : is_midpoint M A B
axiom point_on_BC (P : Point) : on_segment P B C
axiom intersection_PD_AB (P D X : Point) : intersects (line_through P D) (line_through A B) X
axiom intersection_PM_AC (P M A C Q : Point) : intersects (line_through P M) (line_through A C) Q
axiom intersection_DQ_AB (D Q X A Y : Point) : intersects (line_through D Q) (line_through A B) Y

--Proof that M is the midpoint of XY
theorem midpoint_XY (A B C D M P X Q Y : Point)
  (h_trapezoid : trapezoid A B C D)
  (h_midpoint_AB : midpoint A B M)
  (h_point_on_BC : on_segment P B C)
  (h_intersection_PD_AB : intersects (line_through P D) (line_through A B) X)
  (h_intersection_PM_AC : intersects (line_through P M) (line_through A C) Q)
  (h_intersection_DQ_AB : intersects (line_through D Q) (line_through A B) Y) :
  midpoint X Y M := sorry

end midpoint_XY_l73_73659


namespace eccentricity_of_hyperbola_l73_73534

open Real

def hyperbola_foci (a b : ℝ) : ℝ := sqrt (a^2 + b^2) / a

theorem eccentricity_of_hyperbola (a b : ℝ)
    (h : sin ((b^2 / a) / sqrt (4 * (a^2 + b^2) + (b^4 / a^2))) = 1 / 3) :
    hyperbola_foci a b = sqrt 2 :=
by
  sorry

end eccentricity_of_hyperbola_l73_73534


namespace athletics_meet_medals_l73_73453

namespace Athletics

def medal_distribution_pattern (n m : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 ∧ k ≤ n → let remaining := m - (∑ i in Finset.range (k - 1), i + 1 + 1/7 * (m - i))
                             in let awarded := k + 1/7 * (remaining - k)
                             in remaining = m → ∑ i in Finset.range n, i + 1 + 1/7 * (remaining - i) = m

theorem athletics_meet_medals (n m : ℕ) (h : n = 6 ∧ m = 36) :
  medal_distribution_pattern n m := by
  sorry

end Athletics

end athletics_meet_medals_l73_73453


namespace integer_coordinates_point_exists_l73_73700

theorem integer_coordinates_point_exists (p q : ℤ) (h : p^2 - 4 * q = 0) :
  ∃ a b : ℤ, b = a^2 + p * a + q ∧ (a = -p ∧ b = q) ∧ (a ≠ -p → (a = p ∧ b = q) → (p^2 - 4 * b = 0)) :=
by
  sorry

end integer_coordinates_point_exists_l73_73700


namespace solve_congruences_l73_73932

noncomputable def sequence_a (n : ℕ) : ℕ := 10^n + 10^(2016 - n)

theorem solve_congruences :
  let aₙ := sequence_a in
  (aₙ 336 ≡ 1 [MOD 2017] ∧ aₙ 252^2 ≡ 2 [MOD 2017] ∧ aₙ 756^2 ≡ 2 [MOD 2017] ∧
   aₙ 112^3 - 3 * aₙ 112 ≡ 1 [MOD 2017] ∧ aₙ 560^3 - 3 * aₙ 560 ≡ 1 [MOD 2017] ∧
   aₙ 784^3 - 3 * aₙ 784 ≡ 1 [MOD 2017] ∧
   aₙ 288^3 + aₙ 288^2 - 2 * aₙ 288 ≡ 1 [MOD 2017] ∧
   aₙ 576^3 + aₙ 576^2 - 2 * aₙ 576 ≡ 1 [MOD 2017] ∧
   aₙ 864^3 + aₙ 864^2 - 2 * aₙ 864 ≡ 1 [MOD 2017]) :=
by {
  let aₙ := sequence_a,
  have h1 : aₙ 336 ≡ 1 [MOD 2017] := sorry,
  have h2a : aₙ 252^2 ≡ 2 [MOD 2017] := sorry,
  have h2b : aₙ 756^2 ≡ 2 [MOD 2017] := sorry,
  have h3a : aₙ 112^3 - 3 * aₙ 112 ≡ 1 [MOD 2017] := sorry,
  have h3b : aₙ 560^3 - 3 * aₙ 560 ≡ 1 [MOD 2017] := sorry,
  have h3c : aₙ 784^3 - 3 * aₙ 784 ≡ 1 [MOD 2017] := sorry,
  have h4a : aₙ 288^3 + aₙ 288^2 - 2 * aₙ 288 ≡ 1 [MOD 2017] := sorry,
  have h4b : aₙ 576^3 + aₙ 576^2 - 2 * aₙ 576 ≡ 1 [MOD 2017] := sorry,
  have h4c : aₙ 864^3 + aₙ 864^2 - 2 * aₙ 864 ≡ 1 [MOD 2017] := sorry,
  exact ⟨h1, h2a, h2b, h3a, h3b, h3c, h4a, h4b, h4c⟩,
}

end solve_congruences_l73_73932


namespace valid_permutations_remainder_l73_73647

noncomputable def valid_permutations_count : ℕ := 
  ∑ k in finset.range 5, (nat.choose 5 (k + 1)) * (nat.choose 5 k) * (nat.choose 6 (k + 1))

theorem valid_permutations_remainder :
  valid_permutations_count % 1000 = 520 :=
by sorry

end valid_permutations_remainder_l73_73647


namespace units_digit_7_pow_1995_l73_73951

theorem units_digit_7_pow_1995 : 
  ∃ a : ℕ, a = 3 ∧ ∀ n : ℕ, (7^n % 10 = a) → ((n % 4) + 1 = 3) := 
by
  sorry

end units_digit_7_pow_1995_l73_73951


namespace sum_cot_half_angles_l73_73398

theorem sum_cot_half_angles (α β γ : ℝ) (a b c r : ℝ) (p : ℝ) :
  α + β + γ = π ∧ p = (a + b + c) / 2 ∧ r = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c)).sqrt) →
  Real.cot (α / 2) + Real.cot (β / 2) + Real.cot (γ / 2) = p / r := sorry

end sum_cot_half_angles_l73_73398


namespace double_sum_equals_fraction_l73_73068

noncomputable def double_sum : ℝ :=
  ∑' (j : ℕ), ∑' (k : ℕ), 2 ^ (-(4 * k + 2 * j + (k + j)^2))

theorem double_sum_equals_fraction :
  double_sum = 4 / 3 :=
by sorry

end double_sum_equals_fraction_l73_73068


namespace AX_eq_AY_l73_73258

noncomputable theory
open_locale classical

variables {A B C O M P X Y : Point}
variables {circle_ABC circle_BOC : Circle}

-- Declare basic geometric properties and objects
axiom h1 : Circle.circumcenter A B C O
axiom h2 : is_midpoint M B C
axiom h3 : Line.orthogonal (Line.mk O P) (Line.mk A M)
axiom h4 : Line.meets_circle (Line.mk O P) circle_BOC  P
axiom h5 : Line.intersects (Line.perpendicular_from A (Line.mk O A)) (Line.mk B P) X
axiom h6 : Line.intersects (Line.perpendicular_from A (Line.mk O A)) (Line.mk C P) Y
axiom h7 : P ≠ O -- To ensure P lies on the circumcircle distinct from O

-- The statement to prove:
theorem AX_eq_AY :
  distance A X = distance A Y :=
sorry

end AX_eq_AY_l73_73258


namespace greatest_number_of_elements_l73_73805

/-- A set S of distinct positive integers has the following property: for every integer x in S,
the arithmetic mean of the set of values obtained by deleting x from S is an integer. Given that 
1 belongs to S and that 2002 is the largest element of S, what is the greatest number 
of elements that S can have? -/
theorem greatest_number_of_elements 
  (S : Set ℕ) 
  (h_disjoint : ∀ a b ∈ S, a ≠ b → a ≠ b) 
  (h_mean_is_int : ∀ x ∈ S, (S \ {x}).sum / (S.size - 1) ∈ ℤ) 
  (h_one_in_S : 1 ∈ S) 
  (h_max_2002 : ∀ x ∈ S, x ≤ 2002) : 
  S.size ≤ 30 :=
sorry

end greatest_number_of_elements_l73_73805


namespace right_triangle_area_eq_l73_73341

def area_of_right_triangle (x y h : ℝ) : ℝ := 
  let l := x + y
  if h < 0 then 0 else (1 / 2) * h * (Real.sqrt (l^2 + h^2) - h)

theorem right_triangle_area_eq {x y h : ℝ} (h_sum : x + y = l) (h_nonneg : h ≥ 0) :
  area_of_right_triangle x y h = (1 / 2) * h * (Real.sqrt (l^2 + h^2) - h) :=
sorry

end right_triangle_area_eq_l73_73341


namespace maximal_ratio_of_primes_l73_73269

theorem maximal_ratio_of_primes (p q : ℕ) (hp : p.prime) (hq : q.prime) 
  (h1 : p > q) (h2 : ¬ (240 ∣ p^4 - q^4)) : (maximal_ratio_qp := q / p) = 2 / 3 :=
  sorry

end maximal_ratio_of_primes_l73_73269


namespace angle_C_of_triangle_possible_values_p_l73_73626
  
theorem angle_C_of_triangle (A B C : ℝ) (p : ℝ)
  (h1 : tan A ∈ set_of (λ x, x ∈ real.roots (λ x, x^2 + (x+1)*p + 1)))
  (h2 : tan B ∈ set_of (λ x, x ∈ real.roots (λ x, x^2 + (x+1)*p + 1)))
  (h3 : A + B + C = π) : 
  C = 3 * π / 4 := 
sorry

theorem possible_values_p (p : ℝ)
  (h1 : ∀ (A B : ℝ), tan A ∈ set_of (λ x, x ∈ real.roots (λ x, x^2 + (x+1)*p + 1)) 
    → tan B ∈ set_of (λ x, x ∈ real.roots (λ x, x^2 + (x+1)*p + 1)) 
    → A + B = π / 4 
    → A ∈ Ioo 0 (π / 4) 
    → B ∈ Ioo 0 (π / 4)) : 
  p ∈ Icc (-2 : ℝ) (2 - 2 * real.sqrt 2) :=
sorry

end angle_C_of_triangle_possible_values_p_l73_73626


namespace jerry_money_left_after_shopping_l73_73636

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l73_73636


namespace double_sum_equals_fraction_l73_73067

noncomputable def double_sum : ℝ :=
  ∑' (j : ℕ), ∑' (k : ℕ), 2 ^ (-(4 * k + 2 * j + (k + j)^2))

theorem double_sum_equals_fraction :
  double_sum = 4 / 3 :=
by sorry

end double_sum_equals_fraction_l73_73067


namespace function_passes_through_point_l73_73921

noncomputable def special_function (a : ℝ) (x : ℝ) := a^(x - 1) + 1

theorem function_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  special_function a 1 = 2 :=
by
  -- skip the proof
  sorry

end function_passes_through_point_l73_73921


namespace sum_factorials_mod_25_l73_73590

theorem sum_factorials_mod_25 : ((1! + 2! + 3! + 4!) % 25 = 8) := by
  sorry

end sum_factorials_mod_25_l73_73590


namespace find_multiple_of_sons_age_l73_73709

theorem find_multiple_of_sons_age (F S k : ℕ) 
  (h1 : F = 33)
  (h2 : F = k * S + 3)
  (h3 : F + 3 = 2 * (S + 3) + 10) : 
  k = 3 :=
by
  sorry

end find_multiple_of_sons_age_l73_73709


namespace trapezoid_QR_length_l73_73687

noncomputable def length_QR (PQ RS area altitude : ℕ) : ℝ :=
  24 - Real.sqrt 11 - 2 * Real.sqrt 24

theorem trapezoid_QR_length :
  ∀ (PQ RS area altitude : ℕ), 
  area = 240 → altitude = 10 → PQ = 12 → RS = 22 →
  length_QR PQ RS area altitude = 24 - Real.sqrt 11 - 2 * Real.sqrt 24 :=
by
  intros PQ RS area altitude h_area h_altitude h_PQ h_RS
  unfold length_QR
  sorry

end trapezoid_QR_length_l73_73687


namespace perpendicular_condition_l73_73418

-- Definitions:
variable (α : Type) [Plane α]
variable (l a b : Line)
variable (hlα : Intersects l α ∧ ¬Perpendicular l α)
variable (ha : Projection l α = a)
variable (hb : InPlane b α)

-- Theorem statement: 
theorem perpendicular_condition : Perpendicular a b ↔ Perpendicular b l :=
by
  sorry

end perpendicular_condition_l73_73418


namespace value_of_x_l73_73946

theorem value_of_x (x : ℝ) (h₁ : x > 0) (h₂ : x^3 = 19683) : x = 27 :=
sorry

end value_of_x_l73_73946


namespace unique_three_positive_perfect_square_sums_to_100_l73_73234

theorem unique_three_positive_perfect_square_sums_to_100 :
  ∃! (a b c : ℕ), a^2 + b^2 + c^2 = 100 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end unique_three_positive_perfect_square_sums_to_100_l73_73234


namespace monochromatic_regions_lower_bound_l73_73622

theorem monochromatic_regions_lower_bound (n : ℕ) (h_n_ge_2 : n ≥ 2) :
  ∀ (blue_lines red_lines : ℕ) (conditions :
    blue_lines = 2 * n ∧ red_lines = n ∧ 
    (∀ (i j k l : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      (blue_lines = 2 * n ∧ red_lines = n))) 
  , ∃ (monochromatic_regions : ℕ), 
      monochromatic_regions ≥ (n - 1) * (n - 2) / 2 :=
sorry

end monochromatic_regions_lower_bound_l73_73622


namespace problem1_problem2_problem3_l73_73796

open Real

-- Defining the given conditions
def billion := 10^9
def ten_million := 10^7

-- Given values
def six_billion := 6 * billion
def seven_ten_million := 7 * ten_million

-- The number composed of these values
def number := six_billion + seven_ten_million

-- Function to convert a number to "ten thousand" units
def to_ten_thousand_units (n : ℝ) : ℝ := n / 10^4

-- Function to approximate a number to the nearest billion
def to_nearest_billion (n : ℝ) : ℝ := (n / billion).round * billion

-- Theorem statements
theorem problem1 : number = 60700000000 :=
sorry

theorem problem2 : to_ten_thousand_units number = 6070000 :=
sorry

theorem problem3 : to_nearest_billion number = 61000000000 :=
sorry

end problem1_problem2_problem3_l73_73796


namespace identify_wise_l73_73355

def total_people : ℕ := 30

def is_wise (p : ℕ) : Prop := True   -- This can be further detailed to specify wise characteristics
def is_fool (p : ℕ) : Prop := True    -- This can be further detailed to specify fool characteristics

def wise_count (w : ℕ) : Prop := True -- This indicates the count of wise people
def fool_count (f : ℕ) : Prop := True -- This indicates the count of fool people

def sum_of_groups (wise_groups fool_groups : ℕ) : Prop :=
  wise_groups + fool_groups = total_people

def sum_of_fools (fool_groups : ℕ) (F : ℕ) : Prop :=
  fool_groups = F

theorem identify_wise (F : ℕ) (h1 : F ≤ 8) :
  ∃ (wise_person : ℕ), (wise_person < 30 ∧ is_wise wise_person) :=
by
  sorry

end identify_wise_l73_73355


namespace abs_neg_two_thirds_l73_73685

theorem abs_neg_two_thirds : abs (-2/3 : ℝ) = 2/3 :=
by
  sorry

end abs_neg_two_thirds_l73_73685


namespace unique_sum_of_three_squares_l73_73243

-- Defining perfect squares less than 100.
def perfect_squares : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

-- Predicate that checks if the sum of three perfect squares is equal to 100.
def is_sum_of_three_squares (a b c : ℕ) : Prop :=
  a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ c ∈ perfect_squares ∧ a + b + c = 100

-- The main theorem to be proved.
theorem unique_sum_of_three_squares :
  { (a, b, c) // is_sum_of_three_squares a b c }.to_finset.card = 1 :=
sorry -- Proof would go here.

end unique_sum_of_three_squares_l73_73243


namespace monotonicity_and_extremum_no_zeros_l73_73919

variables (a x : ℝ)
variable (h_a : 0 < a)

def f (x : ℝ) := (a / x) + (x / a) - (a - (1 / a)) * real.log x

theorem monotonicity_and_extremum :
  (∀ x ∈ set.Ioi 0, 0 < x → x < a^2 → deriv (f a) x < 0) ∧
  (∀ x ∈ set.Ioi 0, x > a^2 → deriv (f a) x > 0) ∧
  ∃ x0, f a x0 = infi (f a) :=
  sorry

theorem no_zeros (h_a2 : (1/2 : ℝ) ≤ a ∧ a ≤ 2) : ¬∃ x, f a x = 0 :=
  sorry

end monotonicity_and_extremum_no_zeros_l73_73919


namespace unique_mod_inverse_l73_73654

theorem unique_mod_inverse (a n : ℤ) (coprime : Int.gcd a n = 1) : 
  ∃! b : ℤ, (a * b) % n = 1 % n := 
sorry

end unique_mod_inverse_l73_73654


namespace total_num_animals_l73_73171

-- Given conditions
def num_pigs : ℕ := 10
def num_cows : ℕ := (2 * num_pigs) - 3
def num_goats : ℕ := num_cows + 6

-- Theorem statement
theorem total_num_animals : num_pigs + num_cows + num_goats = 50 := 
by
  sorry

end total_num_animals_l73_73171


namespace grid_sum_at_least_half_m_squared_l73_73608

theorem grid_sum_at_least_half_m_squared (m : ℕ) (grid : Fin m → Fin m → ℕ)
  (h : ∀ i j, grid i j = 0 → (∑ k in Finset.univ, grid i k + ∑ k in Finset.univ, grid k j) ≥ m) :
  (∑ i in Finset.univ, ∑ j in Finset.univ, grid i j) ≥ m * m / 2 := 
sorry

end grid_sum_at_least_half_m_squared_l73_73608


namespace vertical_distance_to_fritz_l73_73219

-- Definitions and coordinates
def diane : ℝ × ℝ := (10, -15)
def edward : ℝ × ℝ := (5, 20)
def fritz : ℝ × ℝ := (7.5, 5)

-- The midpoint calculation
def midpoint (p1 p2: ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- The statement we need to prove
theorem vertical_distance_to_fritz (diane edward fritz : ℝ × ℝ) :
  let m := midpoint diane edward in (fritz.2 - m.2) = 2.5 :=
by
  sorry

end vertical_distance_to_fritz_l73_73219


namespace part_I_part_II_l73_73154

noncomputable def f (x : ℝ) := Real.log (1 + x) - x
noncomputable def g (x : ℝ) := x * Real.log x

theorem part_I : ∃ x : ℝ, x ∈ set.Ioc (-1:ℝ) (⊤ : ℝ) ∧ f(x) = 0 :=
sorry

theorem part_II (a b : ℝ) (h : 0 < a ∧ a < b) : 
0 < g(a) + g(b) - 2*g((a + b)/2) ∧ g(a) + g(b) - 2*g((a + b)/2) < (b - a) * Real.log 2 :=
sorry

end part_I_part_II_l73_73154


namespace y_decreases_as_x_less_than_4_l73_73930

theorem y_decreases_as_x_less_than_4 (x : ℝ) : (x < 4) → ((x - 4)^2 + 3 < (4 - 4)^2 + 3) :=
by
  sorry

end y_decreases_as_x_less_than_4_l73_73930


namespace unique_sum_of_three_squares_l73_73241

-- Defining perfect squares less than 100.
def perfect_squares : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

-- Predicate that checks if the sum of three perfect squares is equal to 100.
def is_sum_of_three_squares (a b c : ℕ) : Prop :=
  a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ c ∈ perfect_squares ∧ a + b + c = 100

-- The main theorem to be proved.
theorem unique_sum_of_three_squares :
  { (a, b, c) // is_sum_of_three_squares a b c }.to_finset.card = 1 :=
sorry -- Proof would go here.

end unique_sum_of_three_squares_l73_73241


namespace connect_points_l73_73517

theorem connect_points (n : ℕ) (h : n ≥ 1) (points : fin (2 * n) → point)
  (no_three_collinear : ∀ {i j k : fin (2 * n)},
    i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k))
  (coloring : fin (2 * n) → color)
  (n_blue : (coloring_points_count coloring color.blue) = n)
  (n_red  : (coloring_points_count coloring color.red) = n) :
  ∃ segments : fin n → (fin (2 * n) × fin (2 * n)),
    (∀ i, coloring (segments i).1 = color.blue ∧ coloring (segments i).2 = color.red) ∧
    non_intersecting_segments segments points :=
sorry

end connect_points_l73_73517


namespace tracy_initial_candies_l73_73362

theorem tracy_initial_candies 
  (x : ℕ)
  (h1 : 4 ∣ x)
  (h2 : 5 ≤ ((x / 2) - 24))
  (h3 : ((x / 2) - 24) ≤ 9) 
  : x = 68 :=
sorry

end tracy_initial_candies_l73_73362


namespace find_second_divisor_l73_73009

theorem find_second_divisor:
  ∃ x: ℝ, (8900 / 6) / x = 370.8333333333333 ∧ x = 4 :=
sorry

end find_second_divisor_l73_73009


namespace hyperbola_iff_m_lt_0_l73_73319

theorem hyperbola_iff_m_lt_0 (m : ℝ) : (m < 0) ↔ (∃ x y : ℝ,  x^2 + m * y^2 = m) :=
by sorry

end hyperbola_iff_m_lt_0_l73_73319


namespace vector_magnitude_problem_l73_73906

theorem vector_magnitude_problem (a b : EuclideanSpace ℝ (Fin 2)) 
                                 (h : ∠ (0 : EuclideanSpace ℝ (Fin 2)) a b = Real.pi / 3) 
                                 (ha : a = ![2, 0]) 
                                 (hb : ∥b∥ = 1) : 
∥a + (2 : ℝ) • b∥ = 2 * Real.sqrt 3 :=
sorry

end vector_magnitude_problem_l73_73906


namespace cos_1030_eq_cos_50_l73_73044

open Real

theorem cos_1030_eq_cos_50 :
  (cos (1030 * π / 180) = cos (50 * π / 180)) :=
by
  sorry

end cos_1030_eq_cos_50_l73_73044


namespace integer_part_of_nested_radical_l73_73099

noncomputable def nested_radical (n : ℕ) : ℝ :=
  Nat.recOn n (1981.0) (λ _ ih, Real.sqrt (1981.0 + ih))

theorem integer_part_of_nested_radical (n : ℕ) (hn : n ≥ 2) :
  ⌊ nested_radical n ⌋ = 45 :=
by
  sorry

end integer_part_of_nested_radical_l73_73099


namespace manicure_cost_before_tip_l73_73729

theorem manicure_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_paid = 39 → tip_percentage = 0.30 → total_paid = cost_before_tip + tip_percentage * cost_before_tip → cost_before_tip = 30 :=
by
  intro h1 h2 h3
  sorry

end manicure_cost_before_tip_l73_73729


namespace determinant_2x2_l73_73133

theorem determinant_2x2 (a b c d : ℝ) 
  (h : Matrix.det (Matrix.of ![![1, a, b], ![2, c, d], ![3, 0, 0]]) = 6) : 
  Matrix.det (Matrix.of ![![a, b], ![c, d]]) = 2 :=
by
  sorry

end determinant_2x2_l73_73133


namespace odd_function_implication_increasing_function_solve_inequality_l73_73901

noncomputable def f (x a b : ℝ) := (x + a) / (x^2 + b * x + 1)

variables {a b : ℝ} {x : ℝ}

-- f(x) is odd on [-1, 1] implies a = 0 and b = 0
theorem odd_function_implication (h_odd : ∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) : 
  a = 0 ∧ b = 0 :=
begin
  sorry
end

noncomputable def g (x : ℝ) := x / (x^2 + 1)

-- g(x) (rewritten f(x) with a=0, b=0) is increasing on [-1, 1]
theorem increasing_function : ∀ x1 x2 : ℝ, -1 ≤ x1 → x1 < x2 → x2 ≤ 1 → g x1 < g x2 :=
begin
  sorry
end

-- Solve inequality g(x) - g(1 - x) < 0
theorem solve_inequality : ∀ x: ℝ, 0 ≤ x → x < 1 → g x < g (1 - x) ↔ 0 ≤ x ∧ x < 1/2 :=
begin
  sorry
end

end odd_function_implication_increasing_function_solve_inequality_l73_73901


namespace combined_list_correct_l73_73632

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25

def combined_list : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_correct : combined_list = 275 := by
  unfold combined_list
  unfold james_friends
  unfold john_friends
  unfold shared_friends
  sorry

end combined_list_correct_l73_73632


namespace solution_set_of_inequality_l73_73542

variables {f : ℝ → ℝ}

-- Conditions
axiom odd_function : ∀ x : ℝ, f(x + 1) = -f(-x - 1)
axiom decreasing_function : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0

-- Goal
theorem solution_set_of_inequality : { x : ℝ | f(2x - 3) > 0 } = set.Iio 2 :=
sorry

end solution_set_of_inequality_l73_73542


namespace math_problem_l73_73251

open_locale big_operators

noncomputable theory

-- Definitions based on the conditions provided
variables {O A B C D E P : Type}
-- r is the radius of the circle centered at O
variable (r : ℝ)
-- Definitions based on the given conditions
def is_center (O: Type) : Prop := true -- Definition for center O

def perpendicular (AB BC : Type) : Prop := true -- AB ⊥ BC 

def on_straight_line (A O E : Type) : Prop := true -- A, O, E are collinear 

def length_three_times_radius (AB : Type) (r : ℝ) : Prop := true -- AB = 3r

def ap_twice_ad (AP AD : Type) : Prop := true -- AP = 2AD

-- Define the problem statement
theorem math_problem (O A B C D E P : Type) 
  (r : ℝ)
  (h1 : is_center O)
  (h2 : perpendicular AB BC)
  (h3 : on_straight_line A O E)
  (h4 : length_three_times_radius AB r)
  (h5 : ap_twice_ad AP AD):
  (∀ (AP PB AB DO AD DE OB AO : ℝ),
     ¬ (AP^2 = PB * AB) ∧
     ¬ (AP * DO = PB * AD) ∧
     ¬ (AB^2 = AD * DE) ∧
     ¬ (AB * AD = OB * AO)) :=
sorry

end math_problem_l73_73251


namespace function_extreme_values_l73_73152

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 - 4 * x + 4

theorem function_extreme_values :
  ∃ (a : ℝ),
    (f a 3 = 1) ∧
    (f a = λ x, (1 / 27) * x^3 - 4 * x + 4) ∧
    (∀ x ∈ (Icc (-1 : ℝ) 3), f (1 / 27) x ≤ f (1 / 27) 2) ∧
    (∀ x ∈ (Icc (-1 : ℝ) 3), f (1 / 27) x ≤ f (1 / 27) (-1)) :=
by {
  use 1 / 27,
  split,
  { norm_num,
    ring },
  { split,
    { ext,
      simp [f, ring] },
    { split,
      { intros x hx,
        exact le_trans (sub_le_self _ zero_le_four) (sub_le_self _ zero_le_four) },
      { intros x hx,
        exact le_trans (sub_le_self _ zero_le_four) (sub_le_self _ zero_le_four) },
      sorry } }

end function_extreme_values_l73_73152


namespace exist_three_points_triangle_area_leq_l73_73472

/-- Given five points in the plane where no three are collinear, and the area of the convex hull
of these points is S, there exist three points which form a triangle of area at most (5 - sqrt 5) / 10 * S. -/
theorem exist_three_points_triangle_area_leq {α : Type*} [linear_ordered_field α] 
  (points : fin 5 → affine_plane α) (h : ∀ (i j k : fin 5), (points i).1 ≠ (points j).1) : 
  ∃ (i j k : fin 5), triangle_area (points i) (points j) (points k) ≤ (5 - real.sqrt 5) / 10 * convex_hull_area points :=
by sorry

end exist_three_points_triangle_area_leq_l73_73472


namespace triangle_area_l73_73618

-- Define the conditions and problem
def BC : ℝ := 10
def height_from_A : ℝ := 12
def AC : ℝ := 13

-- State the main theorem
theorem triangle_area (BC height_from_A AC : ℝ) (hBC : BC = 10) (hheight : height_from_A = 12) (hAC : AC = 13) : 
  (1/2 * BC * height_from_A) = 60 :=
by 
  -- Insert the proof
  sorry

end triangle_area_l73_73618


namespace cosine_of_dihedral_angle_l73_73738

theorem cosine_of_dihedral_angle
  (r : ℝ) -- radius of the smaller sphere
  (theta : ℝ) -- measure of the dihedral angle in radians
  (d : ℝ := 5 * r) -- distance between centers of the spheres
  (alpha : ℝ := real.pi / 3) -- angle = 60 degrees in radians
  (hx : d = 5 * r) -- centers of the spheres are at distance 5r.
  (halpha : alpha = real.pi / 3) -- the angle between the line joining the centers and the edge of the dihedral angle is 60 degrees
  (hx_proj : d * real.cos alpha = 2.5 * r) -- projection distance in the plane containing the edge
  : real.cos theta = 0.04 :=
sorry

end cosine_of_dihedral_angle_l73_73738


namespace great_circle_through_two_points_l73_73789

-- Definitions for the conditions of the problem.
def is_great_circle (sphere_center : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → ℝ) (sphere_radius : ℝ) : Prop :=
  ∀ x : ℝ × ℝ × ℝ, ‖x - sphere_center‖ = sphere_radius → plane x = 0

def is_collinear (p1 p2 c : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, p1 = c + k • (p2 - c)

-- The problem statement to be proved.
theorem great_circle_through_two_points (sphere_center : ℝ × ℝ × ℝ) (sphere_radius : ℝ) (p1 p2 : ℝ × ℝ × ℝ)
  (h1 : ∃ plane : ℝ × ℝ × ℝ → ℝ, is_great_circle sphere_center plane sphere_radius) :
  (is_collinear p1 p2 sphere_center → ∃ṅ : ℕ∞, ∃! (plane : ℝ × ℝ × ℝ → ℝ) (n > 1), is_great_circle sphere_center plane sphere_radius) ∧
  (¬is_collinear p1 p2 sphere_center → ∃! (plane : ℝ × ℝ × ℝ → ℝ), is_great_circle sphere_center plane sphere_radius) := 
sorry

end great_circle_through_two_points_l73_73789


namespace sum_of_inner_segments_l73_73456

/-- Given the following conditions:
  1. The sum of the perimeters of the three quadrilaterals is 25 centimeters.
  2. The sum of the perimeters of the four triangles is 20 centimeters.
  3. The perimeter of triangle ABC is 19 centimeters.
Prove that AD + BE + CF = 13 centimeters. -/
theorem sum_of_inner_segments 
  (perimeter_quads : ℝ)
  (perimeter_tris : ℝ)
  (perimeter_ABC : ℝ)
  (hq : perimeter_quads = 25)
  (ht : perimeter_tris = 20)
  (hABC : perimeter_ABC = 19) 
  : AD + BE + CF = 13 :=
by
  sorry

end sum_of_inner_segments_l73_73456


namespace height_of_wall_l73_73784

theorem height_of_wall (L_brick W_brick H_brick : ℝ) 
                       (L_wall W_wall : ℝ) 
                       (num_bricks : ℕ)
                       (H_wall : ℝ) :
  L_brick = 25 → 
  W_brick = 11.25 → 
  H_brick = 6 → 
  L_wall = 800 →  -- 8 m = 800 cm
  W_wall = 22.5 → 
  num_bricks = 6400 → 
  let V_brick := L_brick * W_brick * H_brick in
  let V_total_bricks := V_brick * num_bricks in
  L_wall * W_wall * H_wall = V_total_bricks →
  H_wall = 599.444 := 
by
  intros h1 h2 h3 h4 h5 h6 V_brick V_total_bricks h7
  sorry

end height_of_wall_l73_73784


namespace jerry_money_left_after_shopping_l73_73637

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l73_73637


namespace painter_remaining_hours_l73_73798

theorem painter_remaining_hours (total_rooms : ℕ) (hours_per_room : ℕ) (painted_rooms : ℕ) (hours_remaining : ℕ) 
    (total_eq : total_rooms = 10) 
    (hours_per_room_eq : hours_per_room = 8) 
    (painted_rooms_eq : painted_rooms = 8) 
    (hours_remaining_eq : hours_remaining = 16) : 
    (total_rooms - painted_rooms) * hours_per_room = hours_remaining := 
by
  rw [total_eq, hours_per_room_eq, painted_rooms_eq, hours_remaining_eq]
  norm_num
  sorry

end painter_remaining_hours_l73_73798


namespace find_d_eq_minus_43_over_5_l73_73092

noncomputable def d : ℚ :=
  -43 / 5

theorem find_d_eq_minus_43_over_5 :
  ∃ (d : ℚ), (floor d)^2 * 3 + floor d * 14 - 45 = 0 ∧ -- Condition for the integer part of d
  frac d * 5 * frac d - frac d * 18 + 8 = 0 ∧ -- Condition for the fractional part of d
  0 ≤ frac d ∧ frac d < 1 ∧ -- Fractional part constraints
  d = -43 / 5 := -- Result to be proven
by {
  use -43 / 5,
  split,
  {
    exact (by calc
    (⌊-43 / 5⌋ : ℤ) = -9 : by norm_cast; exact int.floor_div (-43) 5
              ... : true.intro),
  },
  split,
  {
    sorry,
  },
  split,
  {
    split; linarith,
  },
  {
    refl,
  }
}

end find_d_eq_minus_43_over_5_l73_73092


namespace pentagon_reconstruction_l73_73523

-- Define the points and vectors
variable (A A' B B' C C' D D' E E' : Type)
variables [AddGroup A] [Module ℝ A]
variables [AddGroup A'] [Module ℝ A']
variables [AddGroup B] [Module ℝ B]
variables [AddGroup B'] [Module ℝ B']
variables [AddGroup C] [Module ℝ C]
variables [AddGroup C'] [Module ℝ C']
variables [AddGroup D] [Module ℝ D]
variables [AddGroup D'] [Module ℝ D']
variables [AddGroup E] [Module ℝ E]
variables [AddGroup E'] [Module ℝ E']

open_locale big_operators

def p := (1 : ℝ) / 31
def q := (5 : ℝ) / 31
def r := (10 : ℝ) / 31
def s := (15 : ℝ) / 31
def t := (1 : ℝ) / 31

theorem pentagon_reconstruction 
  (H1 : B = (1 / 2) • A + (1 / 2) • A')
  (H2 : C = (1 / 2) • B + (1 / 2) • B')
  (H3 : D = (1 / 2) • C + (1 / 2) • C')
  (H4 : E = (1 / 2) • D + (1 / 2) • D')
  (H5 : A = (1 / 2) • E + (1 / 2) • E') :
  A = p • A' + q • B' + r • C' + s • D' + t • E' :=
sorry

end pentagon_reconstruction_l73_73523


namespace salmon_trip_l73_73488

theorem salmon_trip (male_salmons : ℕ) (female_salmons : ℕ) : male_salmons = 712261 → female_salmons = 259378 → male_salmons + female_salmons = 971639 :=
  sorry

end salmon_trip_l73_73488


namespace correct_propositions_l73_73916

theorem correct_propositions (h1 : ∃ α : ℝ, sin α * cos α = 1) 
                            (h2 : ∀ x : ℝ, sin (3 * π / 2 + x) = sin (3 * π / 2 + -x))
                            (h3 : ∃ k : ℤ, ∀ x : ℝ, x = π / 8 → sin (2 * x + 5 * π / 4) = sin (2 * (-x) + 5 * π / 4))
                            (h4 : ∀ (α β : ℝ), 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ α > β → sin α > sin β) :
                            (h2 ∧ h3) ∧ ¬h1 ∧ ¬h4 :=
by
  sorry

end correct_propositions_l73_73916


namespace max_odd_integers_l73_73448

theorem max_odd_integers (a1 a2 a3 a4 a5 a6 a7 : ℕ) (hpos : ∀ i, i ∈ [a1, a2, a3, a4, a5, a6, a7] → i > 0) 
  (hprod : a1 * a2 * a3 * a4 * a5 * a6 * a7 % 2 = 0) : 
  ∃ l : List ℕ, l.length = 6 ∧ (∀ i, i ∈ l → i % 2 = 1) ∧ ∃ e : ℕ, e % 2 = 0 ∧ e ∈ [a1, a2, a3, a4, a5, a6, a7] :=
by
  sorry

end max_odd_integers_l73_73448


namespace log_a_8_eq_3_l73_73929

def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem log_a_8_eq_3 (a : ℝ) (h : power_function a (1 / 2) = 1 / 4) : log a 8 = 3 :=
by sorry

end log_a_8_eq_3_l73_73929


namespace probability_of_exacts_one_hit_l73_73089

/-- 
  Represents a numeric value for hit as true and miss as false
-/
def is_hit (n : ℕ) : Bool := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

/-- 
  Check if in a group of three shots, exactly one of them is a hit
-/
def hits_exactly_once (group : List ℕ) : Bool :=
  if group.length = 3 then group.filter is_hit |>.length = 1 else false

/-- 
  Given 20 sets of three-shot results, calculate the probability 
  of hitting exactly once in them
-/
def calculate_probability (shots: List (List ℕ)) : ℚ :=
  let exactly_once := shots.filter hits_exactly_once
  (exactly_once.length : ℚ) / (shots.length : ℚ)

/-- 
  The 20 sets of results provided for analysis
-/
def shots : List (List ℕ) :=
  [[1, 0, 7], [9, 5, 6], [1, 8, 1], [9, 3, 5], [2, 7, 1],
   [8, 3, 2], [6, 1, 2], [4, 5, 8], [3, 2, 9], [6, 8, 3],
   [3, 3, 1], [2, 5, 7], [3, 9, 3], [0, 2, 7], [5, 5, 6],
   [4, 9, 8], [7, 3, 0], [1, 1, 3], [5, 3, 7], [9, 8, 9]]

/-- 
  Statement that calculates and proves the probability of hitting 
  exactly once in 3 shots from the given data
-/
theorem probability_of_exacts_one_hit : 
  calculate_probability shots = 9 / 20 :=
by sorry

end probability_of_exacts_one_hit_l73_73089


namespace negation_of_implication_l73_73877

variable (a b c : ℝ)

theorem negation_of_implication :
  (¬(a + b + c = 3) → a^2 + b^2 + c^2 < 3) ↔
  ¬((a + b + c = 3) → a^2 + b^2 + c^2 ≥ 3) := by
sorry

end negation_of_implication_l73_73877


namespace diagonal_sums_difference_l73_73780

def matrix : Matrix (Fin 5) (Fin 5) ℕ :=
  ![
    ![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![15, 16, 17, 18, 19],
    ![21, 22, 23, 24, 25],
    ![26, 27, 28, 29, 30]
  ]

def reverse_rows (m : Matrix (Fin 5) (Fin 5) ℕ) : Matrix (Fin 5) (Fin 5) ℕ :=
  ![
    m 0, 
    m 1, 
    ![m 2 4, m 2 3, m 2 2, m 2 1, m 2 0],
    m 3, 
    ![m 4 4, m 4 3, m 4 2, m 4 1, m 4 0]
  ]

def main_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  finset.univ.sum (fun i => m i i)

def secondary_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  finset.univ.sum (fun i => m i (4 - i : Fin 5))

def positive_difference_between_diagonal_sums : ℕ :=
  let m' := reverse_rows matrix
  let main_sum := main_diagonal_sum m'
  let secondary_sum := secondary_diagonal_sum m'
  (abs (Int.ofNat secondary_sum - Int.ofNat main_sum)).natAbs

theorem diagonal_sums_difference :
  positive_difference_between_diagonal_sums = 8 :=
sorry

end diagonal_sums_difference_l73_73780


namespace right_triangle_perpendicular_ratio_l73_73714

theorem right_triangle_perpendicular_ratio {a b c r s : ℝ}
 (h : a^2 + b^2 = c^2)
 (perpendicular : r + s = c)
 (ratio_ab : a / b = 2 / 3) :
 r / s = 4 / 9 :=
sorry

end right_triangle_perpendicular_ratio_l73_73714


namespace greta_brother_letters_l73_73168

theorem greta_brother_letters
  (B : ℕ)
  (h1 : Greta_received B + 10)
  (h2 : Mother_received 4 * B + 20)
  (h3 : B + (B + 10) + (4 * B + 20) = 270) :
  B = 40 :=
sorry

end greta_brother_letters_l73_73168


namespace inequality_three_integer_solutions_l73_73838

theorem inequality_three_integer_solutions (c : ℤ) :
  (∃ s1 s2 s3 : ℤ, s1 < s2 ∧ s2 < s3 ∧ 
    (∀ x : ℤ, x^2 + c * x + 1 ≤ 0 ↔ x = s1 ∨ x = s2 ∨ x = s3)) ↔ (c = -4 ∨ c = 4) := 
by 
  sorry

end inequality_three_integer_solutions_l73_73838


namespace flash_time_fraction_of_hour_l73_73794

theorem flash_time_fraction_of_hour (flash_interval : ℕ) (flashes : ℕ) (seconds_per_hour : ℕ) 
  (time_flashes : ℕ) : (flash_interval = 20) ∧ (flashes = 180) ∧ (seconds_per_hour = 3600) 
  ∧ (time_flashes = flash_interval * flashes) → (time_flashes / seconds_per_hour = 1) :=
by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := h
  have h5 : time_flashes = 3600 := by rw [h4, h1, h2]; refl
  have h6 : 3600 / 3600 = 1 := nat.div_self (by norm_num)
  rw [h5, h3]; exact h6

end flash_time_fraction_of_hour_l73_73794


namespace exists_infinitely_many_triples_l73_73306

theorem exists_infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b c : ℕ), a^2 + b^2 + c^2 + 2016 = a * b * c :=
sorry

end exists_infinitely_many_triples_l73_73306


namespace impossibility_of_divisible_sums_l73_73630

theorem impossibility_of_divisible_sums :
  ¬ (∃ f : ℕ × ℕ → ℕ,
    ∀ m n : ℕ, m > 100 → n > 100 → (∑ i in finset.range m, ∑ j in finset.range n, f (i, j)) % (m + n) = 0) :=
by
  sorry

end impossibility_of_divisible_sums_l73_73630


namespace linear_function_points_relation_l73_73205

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ),
    (y1 = 2 * (-3) + 1) →
    (y2 = 2 * 4 + 1) →
    (y1 = -5) ∧ (y2 = 9) :=
by
  intros y1 y2 hy1 hy2
  split
  · exact hy1
  · exact hy2

end linear_function_points_relation_l73_73205


namespace y_coordinate_equidistant_l73_73373

theorem y_coordinate_equidistant :
  ∀ y : ℝ, 
    (∀ A B C : ℝ × ℝ, A = (-3, 0) ∧ B = (-2, 5) ∧ C = (1, 3) →
       ((0, y).dist A = (0, y).dist B ∧ (0, y).dist B = (0, y).dist C) → 
       y = 19 / 4) := 
by
  sorry

end y_coordinate_equidistant_l73_73373


namespace closer_to_A_or_B_l73_73885

variables {Point : Type} [MetricSpace Point]
variables (A B M : Point) (l : Set Point)
-- Definition of a point being on one side of the line
def on_one_side_of_line (X : Point) : Prop := - sorry -- Define the specific geometric concept

-- Definition of a point being on the other side of the line
def on_other_side_of_line (X : Point) : Prop := sorry -- Define the specific geometric concept

-- The line l is the perpendicular bisector of segment AB
variable (is_perpendicular_bisector : l = {X | dist A X = dist B X})

-- The main theorem proving both required properties
theorem closer_to_A_or_B (X : Point) (hX1 : X ∉ l) :
  (on_one_side_of_line X → dist X A < dist X B) ∧
  (on_other_side_of_line X → dist X B < dist X A) := 
sorry

end closer_to_A_or_B_l73_73885


namespace distance_from_origin_point_16_neg9_point_16_neg9_is_in_fourth_quadrant_l73_73605

-- Given conditions
def point : ℝ × ℝ := (16, -9)
def distance_from_origin (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1 ^ 2 + p.2 ^ 2)
def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statements
theorem distance_from_origin_point_16_neg9 :
  distance_from_origin point = Real.sqrt 337 :=
by sorry

theorem point_16_neg9_is_in_fourth_quadrant :
  is_in_fourth_quadrant point :=
by sorry

end distance_from_origin_point_16_neg9_point_16_neg9_is_in_fourth_quadrant_l73_73605


namespace base5_first_digit_of_1024_l73_73375

theorem base5_first_digit_of_1024: 
  ∀ (d : ℕ), (d * 5^4 ≤ 1024) ∧ (1024 < (d+1) * 5^4) → d = 1 :=
by
  sorry

end base5_first_digit_of_1024_l73_73375


namespace quarters_to_add_l73_73023

variable (initial_quarters : ℕ) (value_of_one_quarter : ℝ) (target_value : ℝ)

def quarters_needed_to_reach_target (initial_quarters : ℕ) (value_of_one_quarter : ℝ) (target_value : ℝ) :=
  let current_value := initial_quarters * value_of_one_quarter
  let remaining_value := target_value - current_value
  remaining_value / value_of_one_quarter

theorem quarters_to_add : 
  (quarters_needed_to_reach_target 267 0.25 100.00) = 133 := 
by
  -- Declare initial variables
  let initial_quarters := 267
  let value_of_one_quarter := 0.25
  let target_value := 100.00

  -- Calculate current value
  let current_value := initial_quarters * value_of_one_quarter
  -- Calculate remaining value needed
  let remaining_value := target_value - current_value
  -- Determine quarters needed to reach the target
  let quarters_needed := remaining_value / value_of_one_quarter

  -- Assert the result
  show quarters_needed = 133 from sorry

end quarters_to_add_l73_73023


namespace relationship_y1_y2_l73_73196

theorem relationship_y1_y2 :
  let f : ℝ → ℝ := λ x, 2 * x + 1 in
  let y1 := f (-3) in
  let y2 := f 4 in
  y1 < y2 :=
by {
  -- definitions
  let f := λ x, 2 * x + 1,
  let y1 := f (-3),
  let y2 := f 4,
  -- calculations
  have h1 : y1 = f (-3) := rfl,
  have h2 : y2 = f 4 := rfl,
  -- compare y1 and y2
  rw [h1, h2],
  exact calc
    y1 = f (-3) : rfl
    ... = 2 * (-3) + 1 : rfl
    ... = -5 : by norm_num
    ... < 2 * 4 + 1 : by norm_num
    ... = y2 : rfl
}

end relationship_y1_y2_l73_73196


namespace A_cannot_win_for_n_minus_1_values_l73_73865

-- Definitions based on conditions
variable (n : Nat) (s : Nat)

-- Conditions
axiom positive_n : n > 0
axiom pile_of_stones : s > 0
axiom players_take_turns : True
axiom player_A_goes_first : True
axiom valid_move : (m : Nat) → m = 1 ∨ Prime m ∨ (∃ k, m = k * n)

-- Formulating the problem:
-- Prove that the number of values of \( s \) for which the player \( A \) cannot win is \( n-1 \)
theorem A_cannot_win_for_n_minus_1_values :
  ∃ (S : Finset Nat), S.card = n - 1 ∧ ∀ s ∈ S, ¬ (A_wins s)
  sorry

end A_cannot_win_for_n_minus_1_values_l73_73865


namespace find_k_l73_73473

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^3 - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → k = - 485 / 3 :=
by
  sorry

end find_k_l73_73473


namespace ellipse_focal_length_l73_73326

theorem ellipse_focal_length :
  let a := real.sqrt 5
  let b := real.sqrt 4
  let c := real.sqrt (a^2 - b^2)
  in 2 * c = 2 :=
by
  let a := real.sqrt 5
  let b := real.sqrt 4
  let c := real.sqrt (a^2 - b^2)
  show 2 * c = 2
  sorry

end ellipse_focal_length_l73_73326


namespace cube_difference_positive_l73_73572

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l73_73572


namespace length_CD_l73_73595

-- Given data
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (AB BC : ℝ)

noncomputable def triangle_ABC : Prop :=
  AB = 5 ∧ BC = 7 ∧ ∃ (angle_ABC : ℝ), angle_ABC = 90

-- The target condition to prove
theorem length_CD {CD : ℝ} (h : triangle_ABC AB BC) : CD = 7 :=
by {
  -- proof would be here
  sorry
}

end length_CD_l73_73595


namespace maximum_k_is_165_l73_73576

open Nat

-- Define the product of the integers from 20 to 2020
noncomputable def product_20_to_2020 : ℕ := ∏ i in (finset.range 2001).filter (λ n, n ≥ 20), i

-- Define lemma to count occurrences of 13 in the range 20 to 2020
lemma count_prime_factors_13_in_product_20_to_2020 : ∀ k : ℕ, (product_20_to_2020 % (13^k) = 0 → k ≤ 165) :=
by
-- Proof will be provided in the lemma
sorry

-- Define the theorem statement
theorem maximum_k_is_165 : ∃ m : ℕ, product_20_to_2020 = 26^165 * m :=
by
  use product_20_to_2020 / 26^165
  apply count_prime_factors_13_in_product_20_to_2020
  sorry

end maximum_k_is_165_l73_73576


namespace sn_bound_l73_73554

open Real

def a (n : ℕ) : ℝ := 
  if n % 2 = 0 then (n / 2 + 1)^2 
  else (n / 2) * (n / 2 + 1)

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, 1 / a (i + 1)

theorem sn_bound (n : ℕ) (hn : n > 0) : 
  S n > (4 * n) / (3 * (n + 3)) := 
sorry

end sn_bound_l73_73554


namespace two_digit_solutions_l73_73716

theorem two_digit_solutions (n : ℕ) (a b : ℕ) :
  (10 * a + b = n) ∧ (9 ≤ n) ∧ (n < 100) ∧ (real.sqrt n = a + real.sqrt b) →
  n = 64 ∨ n = 81 :=
by
  intros h
  sorry

end two_digit_solutions_l73_73716


namespace relationship_y1_y2_l73_73200

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l73_73200


namespace difference_not_always_less_l73_73259

open Real
open Classical

variable (A B C A1 : Point)

theorem difference_not_always_less :
  ∃ (A B C : Point), 
    let A1 := foot_of_altitude A B C in
    |distance A B - distance A C| ≥ |distance A1 B - distance A1 C| := 
by
  sorry

end difference_not_always_less_l73_73259


namespace sum_double_series_eq_four_thirds_l73_73071

theorem sum_double_series_eq_four_thirds :
  (∑' j : ℕ, ∑' k : ℕ, 2^(- (4 * k + 2 * j + (k + j)^2))) = 4 / 3 :=
begin
  sorry
end

end sum_double_series_eq_four_thirds_l73_73071


namespace total_games_in_season_l73_73216

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem total_games_in_season
  (teams : ℕ)
  (games_per_pair : ℕ)
  (h_teams : teams = 30)
  (h_games_per_pair : games_per_pair = 6) :
  (choose 30 2 * games_per_pair) = 2610 :=
  by
    sorry

end total_games_in_season_l73_73216


namespace reciprocals_of_logs_form_GP_l73_73182

variables {a b c r n : ℝ}
variables (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : b = a * r^3) (h5 : c = a * r^5)
variable (n_pos : n > 1)

theorem reciprocals_of_logs_form_GP :
  let log_a_n := (log n) / (log a),
      log_b_n := (log n) / (log (a * r^3)),
      log_c_n := (log n) / (log (a * r^5)) in
  let recip_log_a_n := 1 / log_a_n,
      recip_log_b_n := 1 / log_b_n,
      recip_log_c_n := 1 / log_c_n in
  (recip_log_b_n / recip_log_a_n) = (log a + 3 * log r) / (log a) ∧
  (recip_log_c_n / recip_log_b_n) = (log a + 5 * log r) / (log a + 3 * log r) :=
by sorry

end reciprocals_of_logs_form_GP_l73_73182


namespace log_a1_plus_log_a9_l73_73620

variable {a : ℕ → ℝ}
variable {log : ℝ → ℝ}

-- Assume the provided conditions
axiom is_geometric_sequence : ∀ n, a (n + 1) / a n = a 1 / a 0
axiom a3a5a7_eq_one : a 3 * a 5 * a 7 = 1
axiom log_mul : ∀ x y, log (x * y) = log x + log y
axiom log_one_eq_zero : log 1 = 0

theorem log_a1_plus_log_a9 : log (a 1) + log (a 9) = 0 := 
by {
    sorry
}

end log_a1_plus_log_a9_l73_73620


namespace triangle_bc_over_a_range_l73_73887

variables {a b c : ℝ}

theorem triangle_bc_over_a_range
  (h1 : b + c ≤ 2 * a)
  (h2 : c + a ≤ 2 * b)
  (h3 : a + b > c)
  (h4 : b + c > a)
  (h5 : c + a > b) :
  (2 / 3 : ℝ) < b / a ∧ b / a < (3 / 2 : ℝ) :=
begin
  sorry,
end

end triangle_bc_over_a_range_l73_73887


namespace cara_marbles_l73_73059

theorem cara_marbles :
  ∃ (T : ℕ), 
  let Y := 20 in
  let G := Y / 2 in
  let B := T / 4 in
  let R := B in
  T = Y + G + R + B ∧ B = T / 4 ∧ T = 60 :=
begin
  sorry
end

end cara_marbles_l73_73059


namespace emily_total_spent_l73_73816

-- Define the given conditions.
def cost_per_flower : ℕ := 3
def num_roses : ℕ := 2
def num_daisies : ℕ := 2

-- Calculate the total number of flowers and the total cost.
def total_flowers : ℕ := num_roses + num_daisies
def total_cost : ℕ := total_flowers * cost_per_flower

-- Statement: Prove that Emily spent 12 dollars.
theorem emily_total_spent : total_cost = 12 := by
  sorry

end emily_total_spent_l73_73816


namespace area_change_l73_73543

-- Define the height of the triangle as 8 cm
def height : ℝ := 8

-- Define the initial and final base lengths
def initial_base : ℝ := 16
def final_base : ℝ := 5

-- Define the areas corresponding to initial and final base lengths
def area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height
def initial_area := area initial_base height  -- should be 64 cm^2
def final_area := area final_base height      -- should be 20 cm^2

-- Prove that the areas change as specified in the problem
theorem area_change : initial_area = 64 ∧ final_area = 20 := by
  sorry

end area_change_l73_73543


namespace problem_l73_73889

theorem problem (x y : ℝ)
  (z1 z2 z3 : ℂ)
  (h1 : z1 = -1 + complex.i)
  (h2 : z2 = 1 + complex.i)
  (h3 : z3 = 1 + 4 * complex.i)
  (OA OB OC : ℝ × ℝ)
  (hOA : OA = (-1, 1))
  (hOB : OB = (1, 1))
  (hOC : OC = (1, 4))
  (h : OC = (x * (-1) + y * 1, x * 1 + y * 1)) :
  x + y = 4 :=
sorry

end problem_l73_73889


namespace handshakes_count_l73_73683

-- Define the number of people
def num_people : ℕ := 10

-- Define a function to calculate the number of handshakes
noncomputable def num_handshakes (n : ℕ) : ℕ :=
  (n - 1) * n / 2

-- The main statement to be proved
theorem handshakes_count : num_handshakes num_people = 45 := by
  -- Proof will be filled in here
  sorry

end handshakes_count_l73_73683


namespace student_arrangement_l73_73786

theorem student_arrangement
  (n : ℕ)
  (h1 : (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ d = 4 ∨ d = n / 4 ∨ d ∈ {other divisor pairs}))
  (h2 : n % 4 = 0)
  (cond_smallest : ∃ m, 4 ∣ m ∧ (∀ d : ℕ, d ∣ m → d = 1 ∨ d = m ∨ d = 4 ∨ d = m / 4 ∨ d ∈ {other divisor pairs})) :
  n = 60 :=
sorry

end student_arrangement_l73_73786


namespace range_of_a_l73_73580

theorem range_of_a (a : ℝ) (p : ∀ x ∈ set.Icc 1 2, x^2 ≥ a) (q : ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) :
  (a ≤ 1 ∧ (a ≤ -2 ∨ a ≥ 1)) → a ∈ set.Icc (-∞) (-2) ∪ {1} :=
by sorry

end range_of_a_l73_73580


namespace three_presses_exceed_1000_l73_73437

theorem three_presses_exceed_1000 (x : ℕ) (h : x = 5) : nat.iterate (λ n, n * n) 3 x > 1000 :=
by {
  simp [nat.iterate],
  rw h,
  norm_num,
  sorry
}

end three_presses_exceed_1000_l73_73437


namespace soccer_tournament_solution_l73_73028

-- Define the statement of the problem
theorem soccer_tournament_solution (k : ℕ) (n m : ℕ) (h1 : k ≥ 1) (h2 : n = (k+1)^2) (h3 : m = k*(k+1) / 2)
  (h4 : n > m) : 
  ∃ k : ℕ, n = (k + 1) ^ 2 ∧ m = k * (k + 1) / 2 ∧ k ≥ 1 := 
sorry

end soccer_tournament_solution_l73_73028


namespace distance_A_runs_l73_73407

variable (D : ℕ)
variable (A_speed B_speed : ℕ → ℕ)

axiom A_speed_def : A_speed D = D / 28
axiom B_speed_def : B_speed D = D / 32
axiom A_beats_B : D + 32 = ((A_speed D) * 32 / A_speed_def 1)

theorem distance_A_runs : D = 224 :=
by
  sorry

end distance_A_runs_l73_73407


namespace sum_of_money_l73_73062

noncomputable def Patricia : ℕ := 60
noncomputable def Jethro : ℕ := Patricia / 3
noncomputable def Carmen : ℕ := 2 * Jethro - 7

theorem sum_of_money : Patricia + Jethro + Carmen = 113 := by
  sorry

end sum_of_money_l73_73062


namespace binomial_coefficient_term_containing_x_squared_l73_73619

theorem binomial_coefficient_term_containing_x_squared :
  let x := Real; -- Assuming x is a real number
  ∃ r : ℕ, (x^2 - (2/x))^7 = ∑ k in Finset.range 8, (binomial 7 k) * (-2)^k * x^(14 - 3*k)
  ∧ 14 - 3*r = 2
  ∧ (binomial 7 r) = 560 := 
by {
  sorry
}

end binomial_coefficient_term_containing_x_squared_l73_73619


namespace inequality_proof_l73_73869

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + x + 2 * x^2) * (2 + 3 * y + y^2) * (4 + z + z^2) ≥ 60 * x * y * z :=
by
  sorry

end inequality_proof_l73_73869


namespace ratio_equivalence_l73_73207

theorem ratio_equivalence (x : ℝ) (h : 3 / x = 3 / 16) : x = 16 := 
by
  sorry

end ratio_equivalence_l73_73207


namespace arithmetic_sum_10_terms_l73_73718

def arithmetic_sequence (a_n : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a_n (n+1) = a_n n + d

def sum_first_n_terms (s : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, s i

theorem arithmetic_sum_10_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h_arith : arithmetic_sequence a)
  (h1 : a 0 + a 1 = 5)
  (h2 : a 2 + a 3 = 9)
  (ha : ∀ n, S n = sum_first_n_terms a n) :
  S 10 = 65 :=
sorry

end arithmetic_sum_10_terms_l73_73718


namespace sequence_count_l73_73564

theorem sequence_count :
  ∃ f : ℕ → ℕ,
    (f 3 = 1) ∧ (f 4 = 1) ∧ (f 5 = 1) ∧ (f 6 = 2) ∧ (f 7 = 2) ∧
    (∀ n, n ≥ 8 → f n = f (n-4) + 2 * f (n-5) + f (n-6)) ∧
    f 15 = 21 :=
by {
  sorry
}

end sequence_count_l73_73564


namespace john_speed_above_limit_l73_73997

theorem john_speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) 
  (h1 : distance = 150) (h2 : time = 2) (h3 : speed_limit = 60) : 
  (distance / time) - speed_limit = 15 :=
by
  -- steps to show the proof
  sorry

end john_speed_above_limit_l73_73997


namespace conic_section_eccentricity_l73_73945

noncomputable def eccentricity_of_conic_section (m : ℝ) : ℝ :=
  if m = 2 then (Real.sqrt 2) / 2 else Real.sqrt 2

theorem conic_section_eccentricity (m x : ℝ) :
  (4^(x + 1/2) - 9 * 2^x + 4 = 0) ∧ (2^x = 4 ∨ 2^x = 1/4) → 
  (eccentricity_of_conic_section m = (Real.sqrt 2) / 2 ∨ eccentricity_of_conic_section m = Real.sqrt 2) := 
by
  sorry

end conic_section_eccentricity_l73_73945


namespace cos_smallest_angle_correct_l73_73141

noncomputable def cos_smallest_angle (a : ℝ) : ℝ :=
  let A := 2 * Real.arccos ((a + 6) / (2 * a))
  in Real.cos A / 2

theorem cos_smallest_angle_correct (a : ℝ) (h₁ : a > 0) (h₂ : Real.arccos ((a + 6) / (2 * a)) < Real.pi / 2) :
  cos_smallest_angle 12 = 3 / 4 :=
by sorry

end cos_smallest_angle_correct_l73_73141


namespace find_triangle_side_l73_73594

noncomputable def compute_triangle_side (AB AC : ℕ) (y : ℕ) (cos_θ : ℚ) : ℕ :=
  let left := (21 - y) / (2 * y)
  let right := (1530 - x^2) / 1386
  if (left == right) then sqrt ((4446 * y - 29106) / (2 * y)) else 0

theorem find_triangle_side (AB AC BC x y : ℕ) (h1 : AB = 33) (h2 : AC = 21) 
(h3 : BC = x) (h4 : 8 ≤ y ∧ y ≤ 20) (h5 : y = 11) : x = 30 :=
by {
  -- Let cos A from triangle ADE be equal to cos A from triangle ABC
  let cos_θ := ( 21 - y) / (2 * y ),
  let computed_side := compute_triangle_side AB AC y cos_θ,
  have h6 : computed_side = 30 := by sorry,
  exact h6,
start_cancel : computated_side := 0
}

end find_triangle_side_l73_73594


namespace valid_parameterizations_l73_73333

noncomputable def pointOnLine (x y : ℝ) : Prop :=
  y = 2 * x + 7

noncomputable def directionVector (vx vy : ℝ) : Prop :=
  ∃ k : ℝ, vx = k * 1 ∧ vy = k * 2

noncomputable def paramOptionA (t : ℝ) :=
  ∃ x y, x = 0 + t * 2 ∧ y = 7 + t * 1

noncomputable def paramOptionB (t : ℝ) :=
  ∃ x y, x = -7/2 + t * -1 ∧ y = 0 + t * -2

noncomputable def paramOptionC (t : ℝ) :=
  ∃ x y, x = 1 + t * 6 ∧ y = 9 + t * 3

noncomputable def paramOptionD (t : ℝ) :=
  ∃ x y, x = 2 + t * 1/2 ∧ y = -1 + t * 1

noncomputable def paramOptionE (t : ℝ) :=
  ∃ x y, x = -7 + t * 1/10 ∧ y = -7 + t * 1/5

theorem valid_parameterizations :
  ∀ t : ℝ, 
  (paramOptionA t → (pointOnLine (0 + t * 2) (7 + t * 1) ∧ directionVector 2 1) ∧
   paramOptionB t → (pointOnLine (-7/2 + t * -1) (0 + t * -2) ∧ directionVector -1 -2) ∧
   paramOptionC t → (pointOnLine (1 + t * 6) (9 + t * 3) ∧ directionVector 6 3) ∧
   paramOptionD t → (pointOnLine (2 + t * 1/2) (-1 + t * 1) ∧ directionVector 1/2 1) ∧
   paramOptionE t → (pointOnLine (-7 + t * 1/10) (-7 + t * 1/5) ∧ directionVector 1/10 1/5)) :=
by
  sorry

end valid_parameterizations_l73_73333


namespace directrices_distance_correct_l73_73119

noncomputable def directrices_distance 
  (foci_distance : ℝ) 
  (asymptote_slope : ℝ) : ℝ :=
  let a := sqrt (foci_distance * 2 / 13)
  let directrix_x := (a * a) / sqrt foci_distance
  2 * directrix_x

theorem directrices_distance_correct : 
  directrices_distance 26 (3 / 2) = 8 * sqrt 26 / 13 :=
by
  sorry

end directrices_distance_correct_l73_73119


namespace sum_of_squares_of_roots_l73_73553

theorem sum_of_squares_of_roots (x1 x2 : ℝ) 
    (h1 : 2 * x1^2 + 3 * x1 - 5 = 0) 
    (h2 : 2 * x2^2 + 3 * x2 - 5 = 0)
    (h3 : x1 + x2 = -3 / 2)
    (h4 : x1 * x2 = -5 / 2) : 
    x1^2 + x2^2 = 29 / 4 :=
by
  sorry

end sum_of_squares_of_roots_l73_73553


namespace sum_first_20_terms_l73_73247

variable {α : Type*} [Add α] [Mul α] [One α] [Div α] [Sub α] [HasSmul ℕ α] [Zero α]

-- Assuming a random arithmetic sequence as a list of real numbers
variable (a : ℕ → α) (a₆ a₉ a₁₂ a₁₅ : α)

-- Define the specific condition for the sequence
def condition1 (a : ℕ → α) : Prop := a₆ + a₉ + a₁₂ + a₁₅ = (34 : α)

-- Define the formula for the sum of the first 20 terms of an arithmetic sequence
def S20 (a : ℕ → α) : α := (20 : α) * ((a 1) + (a 20)) / (2 : α)

-- Define the main theorem that proves the sum of the first 20 terms equals 170
theorem sum_first_20_terms (a : ℕ → α) [Noncomputable α] (h : condition1 a) : S20 a = (170 : α) :=
by
  sorry

end sum_first_20_terms_l73_73247


namespace more_sightings_than_triple_cape_may_l73_73082

def daytona_shark_sightings := 26
def cape_may_shark_sightings := 7

theorem more_sightings_than_triple_cape_may :
  daytona_shark_sightings - 3 * cape_may_shark_sightings = 5 :=
by
  sorry

end more_sightings_than_triple_cape_may_l73_73082


namespace average_range_of_subset_l73_73583

theorem average_range_of_subset
  (x : ℕ → ℝ)
  (h_avg : (∑ i in finset.range 10, x i) / 10 = 6)
  (h_stddev : real.sqrt (1 / 10 * (∑ i in finset.range 10, (x i - 6) ^ 2)) = real.sqrt 2) :
  6 - real.sqrt 2 ≤ (∑ i in finset.range 5, x i) / 5 ∧ (∑ i in finset.range 5, x i) / 5 ≤ 6 + real.sqrt 2 := 
by
  sorry

end average_range_of_subset_l73_73583


namespace correlated_pair_l73_73387

def is_correlated (X Y : Type) : Prop := sorry

def pair_A := ("A person's weight", "education level")
def pair_B := ("radius of a circle", "circumference")
def pair_C := ("standard of living", "purchasing power")
def pair_D := ("an adult's wealth", "weight")

theorem correlated_pair :
  is_correlated ("standard of living", "purchasing power") ∧
  ¬is_correlated ("A person's weight", "education level") ∧
  ¬is_correlated ("radius of a circle", "circumference") ∧
  ¬is_correlated ("an adult's wealth", "weight") :=
by
  sorry

end correlated_pair_l73_73387


namespace tetrahedron_circumradius_l73_73246

theorem tetrahedron_circumradius 
  (A B C D : Point ℝ 3)
  (hAB : dist A B = 3)
  (hAC : dist A C = 3)
  (hBD : dist B D = 4)
  (hBC : dist B C = 4)
  (hBD_perp_ABC : ∃ (n : Vector ℝ 3), n ≠ 0 ∧ n ⟂ (B-C) ∧ n ⟂ (B-A) ∧ n ⟂ (D-B)) :
  ∃ (R : ℝ), R = 2 :=
by sorry

end tetrahedron_circumradius_l73_73246


namespace savings_equal_after_25_weeks_l73_73755

theorem savings_equal_after_25_weeks (x : ℝ) :
  (160 + 25 * x = 210 + 125) → x = 7 :=
by 
  apply sorry

end savings_equal_after_25_weeks_l73_73755


namespace value_of_f_2017_l73_73879

noncomputable def f (x : ℝ) : ℝ := x^2 - x * (f' 0) - 1

theorem value_of_f_2017 : f 2017 = 2016 * 2018 :=
by
  -- Proof omitted
  sorry

end value_of_f_2017_l73_73879


namespace identify_propositions_l73_73936

-- Definitions based on the conditions
def P1 : Prop := ∀ (X : Set), ∅ ⊆ X
def P2 (x : ℝ) : Prop := x > 1 → x > 2
def P3 : Prop := false  -- A question cannot be a proposition
def P4 : Prop := ∀ (l1 l2 : Line), (¬∃ p : Point, p ∈ l1 ∧ p ∈ l2) → (l1.parallel l2)

-- The main theorem to prove which statements are propositions
theorem identify_propositions : (P1 ∧ P4) ∧ (¬P2 ∧ ¬P3) :=
by
  -- Skipping the full proof details
  sorry

end identify_propositions_l73_73936


namespace calc_tan_product_l73_73336

noncomputable def orthocenter_divides_altitude (HK HY YK : ℝ) (tanX tanZ : ℝ) : Prop :=
  HK + HY = YK ∧ tanX * tanZ = 3.5

theorem calc_tan_product (HK HY : ℝ) (X Y Z K : Type*)
  [Triangle X Y Z] [Orthocenter H of X Y Z divisible by YK] :
  orthocenter_divides_altitude 8 20 (20 + 8) _ _ :=
by
  sorry

end calc_tan_product_l73_73336


namespace geometric_sequence_properties_sum_of_squares_formula_l73_73118

theorem geometric_sequence_properties (q : ℝ) (a : ℕ → ℝ) :
 (q > 1) → 
 (a 3 * a 5 * a 7 = 512) → 
 (a 3 - 1 + a 7 - 9 = 2 * (a 5 - 3)) → 
 (a 3 = 8 / q^2) → 
 (a 7 = 8 * q^2) → 
 (a 1 = 2) ∧ (q = Real.sqrt 2) :=
 by
  sorry

theorem sum_of_squares_formula (n : ℕ) (a : ℕ → ℝ) :
 (∀ n, a n = Real.pow (Real.sqrt 2) (n + 1)) →
 (S n = ∑ i in finset.range n, a i^2) →
 (S n = 2^(n + 2) - 4) :=
 by
  sorry

end geometric_sequence_properties_sum_of_squares_formula_l73_73118


namespace rotated_log_eq_10_neg_x_l73_73701

theorem rotated_log_eq_10_neg_x : 
  ∀ x : ℝ, (∃ y : ℝ, y = log 10 x) → (∃ y' : ℝ, y' = 10^(-x)) :=
by 
  sorry

end rotated_log_eq_10_neg_x_l73_73701


namespace num_positive_perfect_square_sets_l73_73236

-- Define what it means for three numbers to form a set that sum to 100 
def is_positive_perfect_square_set (a b c : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ c ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 100

-- Define the main theorem to state there are exactly 4 such sets
theorem num_positive_perfect_square_sets : 
  {s : Finset (ℕ × ℕ × ℕ) // (∃ a b c, (a, b, c) ∈ s ∧ is_positive_perfect_square_set a b c) }.card = 4 :=
sorry

end num_positive_perfect_square_sets_l73_73236


namespace find_t_l73_73334

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 3 * t ^ 2 - 2

-- Define the derivative of the displacement function (velocity)
def velocity (t : ℝ) : ℝ := deriv displacement t

-- The theorem statement: for what value of t is the velocity equal to 1?
theorem find_t : ∃ t : ℝ, velocity t = 1 → t = 1 / 6 :=
by
  sorry

end find_t_l73_73334


namespace display_glasses_count_l73_73835

noncomputable def tall_cupboards := 2
noncomputable def wide_cupboards := 2
noncomputable def narrow_cupboards := 2
noncomputable def shelves_per_narrow_cupboard := 3
noncomputable def glasses_tall_cupboard := 30
noncomputable def glasses_wide_cupboard := 2 * glasses_tall_cupboard
noncomputable def glasses_narrow_cupboard := 45
noncomputable def broken_shelf_glasses := glasses_narrow_cupboard / shelves_per_narrow_cupboard

theorem display_glasses_count :
  (tall_cupboards * glasses_tall_cupboard) +
  (wide_cupboards * glasses_wide_cupboard) +
  (1 * (broken_shelf_glasses * (shelves_per_narrow_cupboard - 1)) + glasses_narrow_cupboard) =
  255 :=
by sorry

end display_glasses_count_l73_73835


namespace domain_of_f_l73_73149

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  Real.log ((m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1)

theorem domain_of_f (m : ℝ) :
  (∀ x : ℝ, 0 < (m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1) ↔ (m > 7/3 ∨ m ≤ 1) :=
by { sorry }

end domain_of_f_l73_73149


namespace five_digit_number_is_71925_l73_73854

-- Statement definition
theorem five_digit_number_is_71925 
  (A B C D E F G H I : ℕ) 
  (h1 : {A, B, C, D, E, F, G, H, I} = {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h2 : A + B + D + E = 28) 
  (h3 : C + F + G + H + I = 45 - 28) :
  10000 * A + 1000 * C + 100 * E + 10 * G + I = 71925 := 
sorry

end five_digit_number_is_71925_l73_73854


namespace total_weight_of_10_cans_is_4550_l73_73015

/-- A proof that the total weight of 10 sampled cans is 4550g, given their weights and the standard weight. -/
theorem total_weight_of_10_cans_is_4550 :
  let s := 454
  let w := [444, 459, 454, 459, 454, 454, 449, 454, 459, 464]
  ∑ i in w, i = 4550 := 
by
  sorry

end total_weight_of_10_cans_is_4550_l73_73015


namespace least_possible_BC_l73_73006

-- Define given lengths
def AB := 7 -- cm
def AC := 18 -- cm
def DC := 10 -- cm
def BD := 25 -- cm

-- Define the proof statement
theorem least_possible_BC : 
  ∃ (BC : ℕ), (BC > AC - AB) ∧ (BC > BD - DC) ∧ BC = 16 := by
  sorry

end least_possible_BC_l73_73006


namespace ellipse_semi_major_axis_l73_73712

theorem ellipse_semi_major_axis (m : ℝ) (h : ∀ a b c : ℝ, a^2 = m → b^2 = 4 → a^2 = b^2 + c^2 → 2 * c = 2 → m = 5) :
  ∃ (m : ℝ), m = 5 :=
begin
  use 5,
  sorry
end

end ellipse_semi_major_axis_l73_73712


namespace ninth_term_l73_73343

variable (a d : ℤ)
variable (h1 : a + 2 * d = 20)
variable (h2 : a + 5 * d = 26)

theorem ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end ninth_term_l73_73343


namespace find_fg_l73_73131

noncomputable theory

-- Define f and g as functions
def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

-- Condition: f(x) is odd
def odd_f (x : ℝ) : Prop := f(-x) = -f(x)

-- Condition: g(x) is even
def even_g (x : ℝ) : Prop := g(-x) = g(x)

-- Condition: f(x) + g(x) = 2^x + 2x
def func_eq (x : ℝ) : Prop := f(x) + g(x) = 2^x + 2*x

-- Theorem to be proven
theorem find_fg (x : ℝ) (hf : odd_f x) (hg : even_g x) (heq : func_eq x) :
  g(x) = (2^x + 2^(-x)) / 2 ∧ f(x) = 2^(x - 1) + 2 * x - 2^(-x - 1) :=
begin
  sorry
end

end find_fg_l73_73131


namespace intersection_point_on_circle_l73_73892

-- Define the lines
def l1 (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0

-- Define the circle
def circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 5

-- The theorem to prove is that the intersection point of l1 and l2 is on the circle
theorem intersection_point_on_circle :
  ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ circle x y :=
sorry

end intersection_point_on_circle_l73_73892


namespace reserved_land_for_future_expansion_l73_73643

def TotalLand : ℕ := 150
def HouseAndMachineryLand : ℕ := 25
def CattleRearingLand : ℕ := 40
def CropProductionLand : ℕ := 70

theorem reserved_land_for_future_expansion :
  let ReservedLand := TotalLand - (HouseAndMachineryLand + CattleRearingLand + CropProductionLand)
  in ReservedLand = 15 :=
by
  sorry

end reserved_land_for_future_expansion_l73_73643


namespace probability_xi_greater_than_2_l73_73875

noncomputable def normal_distribution (σ: ℝ) := sorry -- Define the normal distribution, e.g., pdf or cdf.

variables (σ : ℝ) (xi : ℝ)

axiom xi_normal : xi ~ normal_distribution(σ)
axiom condition : ∫(x in -2..0), normal_distribution(sigma).pdf(x) = 0.4

theorem probability_xi_greater_than_2 : ∫(x in 2..∞), normal_distribution(sigma).pdf(xi) = 0.1 :=
by sorry

end probability_xi_greater_than_2_l73_73875


namespace area_per_broccoli_is_one_l73_73791

   -- Define the conditions as constants in the Lean environment
   constant x y : ℕ
   constant increased_by : ℕ := 79
   constant broccolis_this_year : ℕ := 1600

   -- Define the assumptions
   axioms 
     (h1 : y * y = broccolis_this_year)
     (h2 : y * y - x * x = increased_by)
   
   -- State the theorem to prove equivalent to the given problem
   theorem area_per_broccoli_is_one : (y * y) / broccolis_this_year = 1 :=
   by
     -- The actual proof goes here
     sorry
   
end area_per_broccoli_is_one_l73_73791


namespace find_n_cosine_l73_73098

theorem find_n_cosine (n : ℕ) (h₀ : 0 ≤ n ∧ n ≤ 180) (h₁ : real.cos (n * real.pi / 180) = real.cos (942 * real.pi / 180)) :
    n = 138 :=
sorry

end find_n_cosine_l73_73098


namespace roots_of_polynomial_l73_73494

theorem roots_of_polynomial :
  Polynomial.roots (Polynomial.C 1 * Polynomial.X ^ 3 - Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 4 * Polynomial.X + Polynomial.C 4) = { -2, 1, 2 } :=
sorry

end roots_of_polynomial_l73_73494


namespace possible_values_of_m_l73_73078

open Complex

noncomputable def poly1 (p q r s : ℂ) (m : ℂ) : Prop :=
  p * m^3 + q * m^2 + r * m + s = 0

noncomputable def poly2 (p q r s : ℂ) (m : ℂ) : Prop :=
  q * m^3 + r * m^2 + s * m + p = 0

theorem possible_values_of_m (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  poly1 p q r s m → poly2 p q r s m → (m = 1 ∨ m = -1 ∨ m = Complex.i ∨ m = -Complex.i) :=
by
  sorry

end possible_values_of_m_l73_73078


namespace ln_f_greater_than_one_max_a_l73_73311

def f (x a : ℝ) : ℝ := abs (x - 5 / 2) + abs (x - a)

theorem ln_f_greater_than_one (a : ℝ) (h : a = -1 / 2) : ∀ x : ℝ, log (f x a) > 1 :=
by
  sorry

theorem max_a : ∃ (a : ℝ), (∀ x : ℝ, f x a ≥ a) ∧ (∀ b : ℝ, (∀ x : ℝ, f x b ≥ b) → b ≤ 5 / 4) :=
by
  sorry

end ln_f_greater_than_one_max_a_l73_73311


namespace arithmetic_sequence_ninth_term_l73_73345

theorem arithmetic_sequence_ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end arithmetic_sequence_ninth_term_l73_73345


namespace trig_identity_l73_73086

theorem trig_identity :
  sin (-1071 * degree) * sin (99 * degree) + sin (-171 * degree) * sin (-261 * degree) = 0 :=
by
  sorry

end trig_identity_l73_73086


namespace coprime_sum_product_maximized_l73_73503

theorem coprime_sum_product_maximized (k : ℕ) : 
  ∃ (x y : ℕ), (2 * k = x + y) ∧ (Nat.gcd x y = 1) ∧ 
    (xy_maximized : ∀ (x' y' : ℕ), (2 * k = x' + y') → (Nat.gcd x' y' = 1) → x * y ≥ x' * y') :=
by
  sorry

end coprime_sum_product_maximized_l73_73503


namespace collinear_vectors_sum_l73_73873

theorem collinear_vectors_sum (x y : ℝ) 
  (h1 : ∃ λ : ℝ, (-1, y, 2) = (λ * x, λ * (3 / 2), λ * 3)) : 
  x + y = -1 / 2 :=
sorry

end collinear_vectors_sum_l73_73873


namespace room_length_proof_l73_73702

noncomputable def length_of_room (width cost_per_sq_meter total_cost : ℝ) : ℝ :=
  let area := total_cost / cost_per_sq_meter in
  area / width

theorem room_length_proof : length_of_room 3.75 1200 24750 = 5.5 := 
  by
    sorry

end room_length_proof_l73_73702


namespace repeating_decimal_cycle_length_l73_73949

-- Definitions and assumptions based directly from the conditions in a)
variable (p q : ℕ)
variable (h_p_prime : Nat.Prime p)
variable (h_q_prime : Nat.Prime q)
variable (h_q_def : q = 2 * p + 4)
variable (h_decimal_expansion : ∀ n, 1 / q = 0.[166 repeating])

-- The statement to prove the relationship between q and the repeating decimal cycle.
theorem repeating_decimal_cycle_length (p q : ℕ) (h_p_prime : Nat.Prime p) (h_q_prime : Nat.Prime q) (h_q_def : q = 2 * p + 4) (h_decimal_expansion : ∀ n, 1 / q = 0.[166 repeating]) :
  ∃ n, 10 ^ n ≡ 1 [MOD q] ∧ n = 166 :=
by
  sorry

end repeating_decimal_cycle_length_l73_73949


namespace molecular_weight_4_moles_BaBr2_l73_73376

noncomputable def atomic_weight_Ba : ℝ := 137.33
noncomputable def atomic_weight_Br : ℝ := 79.90
noncomputable def moles : ℝ := 4

theorem molecular_weight_4_moles_BaBr2 :
  let molecular_weight_BaBr2 := atomic_weight_Ba + 2 * atomic_weight_Br in
  let molecular_weight_4_moles := moles * molecular_weight_BaBr2 in
  molecular_weight_4_moles = 1188.52 :=
by
  sorry

end molecular_weight_4_moles_BaBr2_l73_73376


namespace plane_equation_l73_73496

variable (x y z : ℝ)

/-- Equation of the plane passing through points (0, 2, 3) and (2, 0, 3) and perpendicular to the plane 3x - y + 2z = 7 is 2x - 2y + z - 1 = 0. -/
theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (∀ (x y z : ℝ), (A * x + B * y + C * z + D = 0 ↔ 
  ((0, 2, 3) = (0, 2, 3) ∨ (2, 0, 3) = (2, 0, 3)) ∧ (3 * x - y + 2 * z = 7))) ∧
  A = 2 ∧ B = -2 ∧ C = 1 ∧ D = -1 :=
by
  sorry

end plane_equation_l73_73496


namespace num_solutions_x2_plus_y2_eq_x3_plus_y_l73_73842

theorem num_solutions_x2_plus_y2_eq_x3_plus_y :
  (∃ (a : ℕ) (ha : 0 < a), ∃ (b : ℕ) (hb : 0 < b), a^2 + b^2 = a^3 + b) = 2 :=
begin
  sorry
end

end num_solutions_x2_plus_y2_eq_x3_plus_y_l73_73842


namespace smallest_triangle_perimeter_l73_73662

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem smallest_triangle_perimeter :
  ∃ (l m n : ℤ), l > m ∧ m > n ∧
  fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧
  fractional_part (3^l / 10^4) = fractional_part (3^n / 10^4) ∧
  l + m + n = 3003 := sorry

end smallest_triangle_perimeter_l73_73662


namespace locus_of_centers_of_circles_l73_73858

structure Point (α : Type _) :=
(x : α)
(y : α)

noncomputable def perpendicular_bisector {α : Type _} [LinearOrderedField α] (A B : Point α) : Set (Point α) :=
  {C | ∃ m b : α, C.y = m * C.x + b ∧ A.y = m * A.x + b ∧ B.y = m * B.x + b ∧
                 (A.x - B.x) * C.x + (A.y - B.y) * C.y = (A.x^2 + A.y^2 - B.x^2 - B.y^2) / 2}

theorem locus_of_centers_of_circles {α : Type _} [LinearOrderedField α] (A B : Point α) :
  (∀ (C : Point α), (∃ r : α, r > 0 ∧ ∃ k: α, (C.x - A.x)^2 + (C.y - A.y)^2 = r^2 ∧ (C.x - B.x)^2 + (C.y - B.y)^2 = r^2) 
  → C ∈ perpendicular_bisector A B) :=
by
  sorry

end locus_of_centers_of_circles_l73_73858


namespace find_c_plus_d_l73_73943

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
noncomputable def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

theorem find_c_plus_d (c d : ℝ) (h1 : ∀ x : ℝ, g c d (g_inv c d x) = x) : c + d = -2 :=
by {
  have h2: g c d (g_inv c d x) = c * (d * x + c) + d := sorry,
  have h3: c * d * x + c^2 + d = x := sorry,
  have h4: c * d = 1 := sorry,
  have h5: c^2 + d = 0 := sorry,
  have h6: d = -c^2 := sorry,
  have h7: c * (-c^2) = 1 := sorry,
  have h8: -c^3 = 1 := sorry,
  have h9: c = -1 := sorry,
  have h10: d = -1 := sorry,
  rw [h9, h10],
  exact rfl
}

end find_c_plus_d_l73_73943


namespace parallelogram_area_l73_73053

noncomputable theory

variables (p q : ℝ^3)
variables (a b : ℝ^3)
variables (θ : ℝ)

-- Conditions
def vector_a : ℝ^3 := p + 3 • q
def vector_b : ℝ^3 := p - 2 • q
def norm_p : ℝ := ∥p∥
def norm_q : ℝ := ∥q∥
def angle_pq : ℝ := θ

-- Specific values
axiom norm_p_eq : norm_p p = 2
axiom norm_q_eq : norm_q q = 3
axiom angle_pq_eq : angle_pq p q = π / 3

-- Theorem: Area of parallelogram
theorem parallelogram_area : ∥vector_a p q × vector_b p q∥ = 15 * real.sqrt 3 :=
sorry

end parallelogram_area_l73_73053


namespace simplify_fraction_l73_73324

theorem simplify_fraction (m : ℤ) : 
  let c := 2 
  let d := 4 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = (1 / 2 : ℚ) :=
by
  sorry

end simplify_fraction_l73_73324


namespace cos_sin_necessary_not_sufficient_condition_l73_73212

noncomputable def triangle_condition_necessary_but_not_sufficient (A B C : ℝ) (h : A + B + C = 180) : Prop :=
  (C = 90) → (cos A + sin A = cos B + sin B) ∧ 
  ¬((cos A + sin A = cos B + sin B) → (C = 90))

/-- Proof that "cosA + sinA = cosB + sinB" is a necessary but not sufficient condition for "C=90°" in ΔABC -/
theorem cos_sin_necessary_not_sufficient_condition (A B C : ℝ) (h_triangle : A + B + C = 180) : 
  triangle_condition_necessary_but_not_sufficient A B C h_triangle :=
sorry

end cos_sin_necessary_not_sufficient_condition_l73_73212


namespace sum_of_zeros_after_transform_l73_73331

theorem sum_of_zeros_after_transform :
  ∃ (f : ℝ → ℝ), 
    (f = λ x, 2 * (x - 3)^2 + 4) ∧
    (∃ (g : ℝ → ℝ), g = λ y x, -2 * (x - 4)^2 + 3 ∧
       (∃ (h : ℝ → ℝ), h = λ x, -2 * (x - 8)^2 ∧
          (∃ z : ℝ, (z = 8) ∧ 
              h 8 = 0 ∧ ∀ x, h x = 0 → x = 8
          )
       )
    )
:= sorry

end sum_of_zeros_after_transform_l73_73331


namespace item_list_price_l73_73042

theorem item_list_price :
  ∃ x : ℝ, (0.15 * (x - 15) = 0.25 * (x - 25)) ∧ x = 40 :=
by
  exists 40
  split
  · linarith
  · rfl

end item_list_price_l73_73042


namespace solve_ratio_BD_DC_l73_73597

noncomputable def ΔABC (A B C D : Point) : Prop := 
  distance A B = 3 ∧ 
  distance B C = 6 ∧ 
  distance A C = 4 ∧ 
  distance A D = 3 ∧ 
  D ∈ line_segment B C

theorem solve_ratio_BD_DC {A B C D : Point} (h : ΔABC A B C D) : 
  (distance B D) / (distance D C) = 29 / 7 :=
sorry

end solve_ratio_BD_DC_l73_73597


namespace monochromatic_K4_exists_l73_73847

section

-- Definitions
variables (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop)
variable [complete_graph : ∀ v1 v2 : V, E v1 v2]

-- Given conditions
variable (color : E → bool)
variable (h_complete : @IsComplete V E)
variable (vertices : Fintype.card V = 18)

theorem monochromatic_K4_exists : ∃ (subset : Finset V), subset.card = 4 ∧
  ∃ (color_choice : bool), ∀ (u v ∈ subset), (E u v) → color (E u v) = color_choice := sorry

end

end monochromatic_K4_exists_l73_73847


namespace circumscribed_sphere_surface_area_l73_73983

noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

def body_diagonal_length (edge_length : ℝ) : ℝ :=
  real.sqrt 3 * edge_length

axiom tetrahedron_coordinates :
  ∃ (A B C D : ℝ × ℝ × ℝ), 
  A = (2, 2, 1) ∧ B = (2, 2, -1) ∧ C = (0, 2, 1) ∧ D = (0, 0, 1)

theorem circumscribed_sphere_surface_area : ∃ (S : ℝ), 
  (let edge_length := 2;
       radius := body_diagonal_length edge_length / 2;
       surface_area := 4 * real.pi * radius^2
       in surface_area = 12 * real.pi) :=
by
  obtain ⟨A, B, C, D, hA, hB, hC, hD⟩ := tetrahedron_coordinates
  use 12 * real.pi
  sorry

end circumscribed_sphere_surface_area_l73_73983


namespace circum_sphere_tetrahedron_l73_73970

theorem circum_sphere_tetrahedron (A B C D A₁ B₁ C₁ : Point) (circumsphere : Sphere)
  (h_tetra : Tetrahedron A B C D)
  (h_plane : Plane (Radius circumsphere D) ⊥ Plane (A₁ B₁ C₁))
  (h_A₁ : Intersect (Line D A) Plane (A₁ B₁ C₁) A₁)
  (h_B₁ : Intersect (Line D B) Plane (A₁ B₁ C₁) B₁)
  (h_C₁ : Intersect (Line D C) Plane (A₁ B₁ C₁) C₁) :
  Cocyclic [A, B, C, A₁, B₁, C₁] :=
sorry

end circum_sphere_tetrahedron_l73_73970


namespace first_arrangement_columns_l73_73968

variable (total_people columns1 columns2 : ℕ)
variable (people_per_column1 : ℕ := 30)
variable (people_per_column2 : ℕ := 32)
variable (num_columns2 : ℕ := 15)

-- Conditions
axiom total_people_calc : total_people = people_per_column2 * num_columns2
axiom columns1_calc : columns1 = total_people / people_per_column1

-- Question: number of columns in the first arrangement
theorem first_arrangement_columns : columns1 = 16 :=
by
  have h_total : total_people = 480 := by rw [total_people_calc]; norm_num
  have h_columns : columns1 = 480 / 30 := by rw [columns1_calc, h_total]
  norm_num at h_columns
  exact h_columns

end first_arrangement_columns_l73_73968


namespace correct_function_l73_73811

noncomputable def candidate1 (x : ℝ) : ℝ := Real.log2 x
noncomputable def candidate2 (x : ℝ) : ℝ := x * abs x
noncomputable def candidate3 (x : ℝ) : ℝ := x^2 + 1
noncomputable def candidate4 (x : ℝ) : ℝ := 2^x

theorem correct_function :
  (∀ x : ℝ, candidate2 (Real.sqrt 2 * x) = 2 * candidate2 x) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f (Real.sqrt 2 * x) = 2 * f x) →
    (f = candidate2)) :=
by
  sorry

end correct_function_l73_73811


namespace number_of_girls_at_camp_l73_73354

theorem number_of_girls_at_camp (total_people : ℕ) (difference_boys_girls : ℕ) (nb_girls : ℕ) :
  total_people = 133 ∧ difference_boys_girls = 33 ∧ 2 * nb_girls + 33 = total_people → nb_girls = 50 := 
by
  intros
  sorry

end number_of_girls_at_camp_l73_73354


namespace unique_sum_of_three_squares_l73_73240

-- Defining perfect squares less than 100.
def perfect_squares : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

-- Predicate that checks if the sum of three perfect squares is equal to 100.
def is_sum_of_three_squares (a b c : ℕ) : Prop :=
  a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ c ∈ perfect_squares ∧ a + b + c = 100

-- The main theorem to be proved.
theorem unique_sum_of_three_squares :
  { (a, b, c) // is_sum_of_three_squares a b c }.to_finset.card = 1 :=
sorry -- Proof would go here.

end unique_sum_of_three_squares_l73_73240


namespace function_eq_l73_73188

noncomputable def f (x : ℝ) : ℝ := x^4 - 2

theorem function_eq (f : ℝ → ℝ) (h1 : ∀ x : ℝ, deriv f x = 4 * x^3) (h2 : f 1 = -1) : 
  ∀ x : ℝ, f x = x^4 - 2 :=
by
  intro x
  -- Proof omitted
  sorry

end function_eq_l73_73188


namespace no_such_a_b_exists_l73_73520

open Set

def A (a b : ℝ) : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, a * x + b) }

def B : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, 3 * (x : ℝ) ^ 2 + 15) }

def C : Set (ℝ × ℝ) :=
  { p | p.1 ^ 2 + p.2 ^ 2 ≤ 144 }

theorem no_such_a_b_exists :
  ¬ ∃ (a b : ℝ), 
    ((A a b ∩ B).Nonempty) ∧ ((a, b) ∈ C) :=
sorry

end no_such_a_b_exists_l73_73520


namespace ellipse_k_range_l73_73146

theorem ellipse_k_range (k : ℝ) 
  (h1 : 4 + k > 0) 
  (h2 : 1 - k > 0) 
  (h3 : 4 + k ≠ 1 - k) :
  k ∈ set.Ioo (-4 : ℝ) (-3 / 2) ∪ set.Ioo (-3 / 2) (1 : ℝ) :=
sorry

end ellipse_k_range_l73_73146


namespace smallest_class_size_l73_73969

theorem smallest_class_size (n : ℕ) (h : 5 * n + 1 > 40) : ∃ k : ℕ, k >= 41 :=
by sorry

end smallest_class_size_l73_73969


namespace collinear_vectors_sum_l73_73874

theorem collinear_vectors_sum (x y : ℝ) 
  (h1 : ∃ λ : ℝ, (-1, y, 2) = (λ * x, λ * (3 / 2), λ * 3)) : 
  x + y = -1 / 2 :=
sorry

end collinear_vectors_sum_l73_73874


namespace fraction_equality_l73_73753

theorem fraction_equality {x y : ℝ} (h : x + y ≠ 0) (h1 : x - y ≠ 0) : 
  (-x + y) / (-x - y) = (x - y) / (x + y) := 
sorry

end fraction_equality_l73_73753


namespace count_valid_permutations_l73_73561

def is_multiple_of_seven (n : Nat) : Prop := n % 7 = 0

def is_valid_permutation (original : Nat) (perm : Nat) : Prop :=
  List.permutations (original.digits) = (perm.digits) -- assuming a digits function and permutations

/--
How many integers between 200 and 800, inclusive, have the property that some permutation 
of its digits is a multiple of 7 between 200 and 800?
-/
theorem count_valid_permutations : 
  {i : Nat | 200 ≤ i ∧ i ≤ 800 ∧ ∃ p, p ≠ i ∧ is_valid_permutation i p ∧ is_multiple_of_seven p ∧ 200 ≤ p ∧ p ≤ 800 }.card = 251 :=
sorry

end count_valid_permutations_l73_73561


namespace honey_water_percentage_l73_73384

-- Definitions of conditions
def nectar_mass : ℝ := 1.3
def honey_mass : ℝ := 1
def water_percentage_nectar : ℝ := 0.5
def solids_mass_nectar : ℝ := (1 - water_percentage_nectar) * nectar_mass

-- Statement to prove
theorem honey_water_percentage : 
  let solids_mass_honey := honey_mass * (1 - x / 100) in
  solids_mass_nectar = solids_mass_honey →
  x = 35 :=
by
  intros solids_mass_honey h
  sorry

end honey_water_percentage_l73_73384


namespace y1_lt_y2_l73_73192

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l73_73192


namespace chord_length_perpendicular_bisector_l73_73785

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) :
  ∃ (CD : ℝ), CD = 10 * Real.sqrt 3 :=
by
  -- The proof is omitted.
  sorry

end chord_length_perpendicular_bisector_l73_73785


namespace remainder_when_sum_divided_by_30_l73_73999

theorem remainder_when_sum_divided_by_30 {c d : ℕ} (p q : ℕ)
  (hc : c = 60 * p + 58)
  (hd : d = 90 * q + 85) :
  (c + d) % 30 = 23 :=
by
  sorry

end remainder_when_sum_divided_by_30_l73_73999


namespace hyperbola_eqn_and_lambda_range_l73_73884

-- Define the given conditions and statements
theorem hyperbola_eqn_and_lambda_range
  (a b : ℝ) (W : Type) [hW : has_lt.W]
  (F_1 F_2 N M Q H A B: W) 
  (h_hyperbola : W = {p : ℝ*ℝ | (p.1^2) / (a^2) - (p.2^2) / (b^2) = 1} )
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (hF_1 : F_1 ∈ W) 
  (hF_2 : F_2 ∈ W) 
  (hN : N = (0, b)) 
  (hM : M = (a, 0)) 
  (hdot : vector.dot_product ((-a, b) : ℝ*ℝ) ((c-a, 0) : ℝ*ℝ) = -1)
  (hangle : angle ((0, b), (a, 0), (c, 0)) = 120) 
  (hlineQ : Q = (0, -2)) 
  (hintersect : exists (k : ℝ), k ≠ 0 ∧ 
                               ∃ A B : ℝ*ℝ, 
                               (A ∈ W ∧ B ∈ W) ∧
                               (0 < x_coordinate(A) ∧ x_coordinate(A) < x_coordinate(B)) ∧
                               2 < k ∧ k < sqrt(7)) 
  (hH : H = (7, 0)) 
  (houtside : distance H (center_of_circle A B) > radius_of_circle A B) : 
  ((a, b) = (1, sqrt 3) ∧ 
   hyperbola_eqn = (λ p : ℝ*ℝ, x^2 - (y^2) / 3 = 1) ∧ 
   (1 < λ < 7)) := 
sorry

end hyperbola_eqn_and_lambda_range_l73_73884


namespace shaded_area_l_shaped_region_l73_73853

theorem shaded_area_l_shaped_region (s₁ s₂ s₃ s₄ : ℕ) (h₁ : s₁ = 6) (h₂ : s₂ = 2) (h₃ : s₃ = 4) (h₄ : s₄ = 2) :
  let area_ABCD := s₁ * s₁
  let area_small_squares := s₂ * s₂ + s₃ * s₃ + s₄ * s₄
  area_ABCD - area_small_squares = 12 :=
by
  rw [h₁, h₂, h₃, h₄]
  sorry

end shaded_area_l_shaped_region_l73_73853


namespace division_problem_l73_73394

theorem division_problem
  (R : ℕ) (D : ℕ) (Q : ℕ) (Div : ℕ)
  (hR : R = 5)
  (hD1 : D = 3 * Q)
  (hD2 : D = 3 * R + 3) :
  Div = D * Q + R :=
by
  have hR : R = 5 := hR
  have hD2 := hD2
  have hDQ := hD1
  -- Proof continues with steps leading to the final desired conclusion
  sorry

end division_problem_l73_73394


namespace lisa_max_non_a_quizzes_l73_73821

def lisa_goal : ℕ := 34
def quizzes_total : ℕ := 40
def quizzes_taken_first : ℕ := 25
def quizzes_with_a_first : ℕ := 20
def remaining_quizzes : ℕ := quizzes_total - quizzes_taken_first
def additional_a_needed : ℕ := lisa_goal - quizzes_with_a_first

theorem lisa_max_non_a_quizzes : 
  additional_a_needed ≤ remaining_quizzes → 
  remaining_quizzes - additional_a_needed ≤ 1 :=
by
  sorry

end lisa_max_non_a_quizzes_l73_73821


namespace seating_arrangements_correct_l73_73731

-- Definitions based on the conditions
def chairs : Type := Fin 12  -- 12 chairs in a round table, numbered 1-12

-- Define what it means for a seating arrangement to be valid
def valid_arrangement (arrangement : chairs → (bool × bool)) : Prop :=
  -- Alternating men and women
  (∀ i : chairs, (arrangement i).fst != (arrangement ((i + 1) % 12)).fst) 
  ∧
  -- No one sits next to, two seats away from, or across from their spouse
  (∀ (i j : chairs), 
    arrangement i = arrangement j →
    arrangement i ≠ arrangement ((j + 1) % 12) ∧
    arrangement i ≠ arrangement ((j + 2) % 12) ∧
    arrangement i ≠ arrangement ((j + 6) % 12))

-- The number of seating arrangements
def num_arrangements : Nat :=
  -- Directly using exact answer calculation
  17280

open chairs

-- The theorem to prove that given the conditions, the number of seating valid seating arrangements is 17280
theorem seating_arrangements_correct : 
  (∑ p in (Finset.univ : Finset (chairs → (bool × bool))), valid_arrangement p) = num_arrangements :=
sorry

end seating_arrangements_correct_l73_73731


namespace group_collected_amount_l73_73416

theorem group_collected_amount (n : ℕ) (hn : n = 54) : (n * n : ℤ) / 100 = 29.16 :=
by
  sorry

end group_collected_amount_l73_73416


namespace periodic_b_l73_73105

def periodic_sequence (b : ℕ → ℝ) (T : ℕ) : Prop :=
  ∀ n : ℕ, b (n + T) = b n

def b (m : ℝ) : ℕ → ℝ
| 0       := m
| (n + 1) := if 0 < b n ∧ b n ≤ 1 then 1 / (b n) else b n - 1

theorem periodic_b (m : ℝ) (h : m = (sqrt 2 - 1)) (hm : 0 < m ∧ m < 1):
  periodic_sequence (b m) 5 :=
sorry

end periodic_b_l73_73105


namespace octahedron_path_count_l73_73831

theorem octahedron_path_count:
  let faces := {1, 2, 3, 4, 5, 6, 7, 8} -- representing the 8 faces of the octahedron
  let top := 1 -- representing the top face
  let bottom := 8 -- representing the bottom face
  let top_ring := {2, 3, 4} -- representing the top ring of faces
  let bottom_ring := {5, 6, 7} -- representing the bottom ring of faces
  (∃ sequences, ∀ seq ∈ sequences, 
    (seq.head = top ∧
    seq.last = bottom ∧
    (∀ (i : ℕ), i < length(seq) - 1 → adjacent (seq[i]) (seq[i+1])) ∧
    (∀ (f : ℕ), f ∈ seq → 
      f = top → (∃ j ∈ top_ring, seq[1] = j) ∧
      f ∈ top_ring → (∃ k ∈ bottom_ring, adjacent f k) ∧
      f ∈ bottom_ring → (seq[length(seq) - 1 = bottom])) ∧
    (∀ p q ∈ seq, p ≠ q))) → 
  count(sequences) = 3 :=
sorry

end octahedron_path_count_l73_73831


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l73_73004

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : a > b ∧ b > 0) : e^(a - b) > 1 :=
by
  sorry

theorem not_necessary_condition (a b : ℝ) (h : e^(a - b) > 1) : ¬(b > 0) ∨ a > b :=
by
  sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l73_73004


namespace carol_initial_peanuts_l73_73064

theorem carol_initial_peanuts (p_initial p_additional p_total : Nat) (h1 : p_additional = 5) (h2 : p_total = 7) (h3 : p_initial + p_additional = p_total) : p_initial = 2 :=
by
  sorry

end carol_initial_peanuts_l73_73064


namespace total_cost_38_pencils_56_pens_l73_73432

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end total_cost_38_pencils_56_pens_l73_73432


namespace probability_at_least_one_six_is_11_div_36_l73_73368

noncomputable def probability_at_least_one_six : ℚ :=
  let total_outcomes := 36
  let no_six_outcomes := 25
  let favorable_outcomes := total_outcomes - no_six_outcomes
  favorable_outcomes / total_outcomes
  
theorem probability_at_least_one_six_is_11_div_36 : 
  probability_at_least_one_six = 11 / 36 :=
by
  sorry

end probability_at_least_one_six_is_11_div_36_l73_73368


namespace cost_of_building_fence_l73_73397

-- Define the conditions
def area_of_circle := 289 -- Area in square feet
def price_per_foot := 58  -- Price in rupees per foot

-- Define the equations used in the problem
noncomputable def radius := Real.sqrt (area_of_circle / Real.pi)
noncomputable def circumference := 2 * Real.pi * radius
noncomputable def cost := circumference * price_per_foot

-- The statement to prove
theorem cost_of_building_fence : cost = 1972 :=
  sorry

end cost_of_building_fence_l73_73397


namespace sum_of_money_l73_73060

theorem sum_of_money (J C P : ℕ) 
  (h1 : P = 60)
  (h2 : P = 3 * J)
  (h3 : C + 7 = 2 * J) : 
  J + P + C = 113 := 
by
  sorry

end sum_of_money_l73_73060


namespace total_animals_l73_73170

def pigs : ℕ := 10

def cows : ℕ := 2 * pigs - 3

def goats : ℕ := cows + 6

theorem total_animals : pigs + cows + goats = 50 := by
  sorry

end total_animals_l73_73170


namespace nesbitt_inequality_l73_73655

theorem nesbitt_inequality (a b c : ℝ) (h_pos1 : 0 < a) (h_pos2 : 0 < b) (h_pos3 : 0 < c) (h_abc: a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
sorry

end nesbitt_inequality_l73_73655


namespace quadratic_one_root_iff_discriminant_zero_l73_73912

theorem quadratic_one_root_iff_discriminant_zero (m : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y^2 - m*y + 1 ≤ 0 ↔ y = x) ↔ (m = 2 ∨ m = -2) :=
by 
  -- We assume the discriminant condition which implies the result
  sorry

end quadratic_one_root_iff_discriminant_zero_l73_73912


namespace unique_three_positive_perfect_square_sums_to_100_l73_73235

theorem unique_three_positive_perfect_square_sums_to_100 :
  ∃! (a b c : ℕ), a^2 + b^2 + c^2 = 100 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end unique_three_positive_perfect_square_sums_to_100_l73_73235


namespace I1I2_parallel_to_EF_l73_73048

variables {A B C D E F P Q R I I1 I2 : Point} 

-- Given conditions
axiom triangle_ABC : Triangle A B C
axiom incircle_touch_BC_at_D : InCircle I A B C ∧ TouchesAt I D B C
axiom incircle_touch_CA_at_E : InCircle I A B C ∧ TouchesAt I E C A
axiom incircle_touch_AB_at_F : InCircle I A B C ∧ TouchesAt I F A B
axiom segment_AD_intersects_incircle_at_P : LineSegment A D ∩ InCircle I A B C = {P}
axiom line_P_parallel_to_AC_meets_incircle_at_R : Parallel (LineThrough P) (LineThrough A C) ∧ Line P meets InCircle I A B C at R
axiom line_P_parallel_to_AB_meets_incircle_at_Q : Parallel (LineThrough P) (LineThrough A B) ∧ Line P meets InCircle I A B C at Q
axiom I1_is_incenter_ΔDPQ : InCenter I1 ΔDPQ
axiom I2_is_incenter_ΔDPR : InCenter I2 ΔDPR

-- Theorem to prove
theorem I1I2_parallel_to_EF : Parallel (LineSegment I1 I2) (LineSegment E F) :=
sorry

end I1I2_parallel_to_EF_l73_73048


namespace find_least_q_l73_73722

theorem find_least_q : 
  ∃ q : ℕ, 
    (q ≡ 0 [MOD 7]) ∧ 
    (q ≥ 1000) ∧ 
    (q ≡ 1 [MOD 3]) ∧ 
    (q ≡ 1 [MOD 4]) ∧ 
    (q ≡ 1 [MOD 5]) ∧ 
    (q = 1141) :=
by
  sorry

end find_least_q_l73_73722


namespace distance_between_points_l73_73055

-- Define the coordinates of the two points
def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (3, -2)

-- Define the distance function using the Euclidean distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- The theorem to prove
theorem distance_between_points :
  distance point1 point2 = 9 :=
by 
-- proof here 
sorry

end distance_between_points_l73_73055


namespace double_sum_equals_fraction_l73_73066

noncomputable def double_sum : ℝ :=
  ∑' (j : ℕ), ∑' (k : ℕ), 2 ^ (-(4 * k + 2 * j + (k + j)^2))

theorem double_sum_equals_fraction :
  double_sum = 4 / 3 :=
by sorry

end double_sum_equals_fraction_l73_73066


namespace correct_choice_l73_73218

def person := Type
variable (A B C D : person) 

-- Statements made by each person
def statement_A : Prop := ¬(A = true)
def statement_B : Prop := C = true
def statement_C : Prop := D = true
def statement_D : Prop := ¬(D = true)

-- Only one person is telling the truth
def one_truth_teller (a b c d : Prop) : Prop :=
(a ∧ ¬b ∧ ¬c ∧ ¬d) ∨ (¬a ∧ b ∧ ¬c ∧ ¬d) ∨ (¬a ∧ ¬b ∧ c ∧ ¬d) ∨ (¬a ∧ ¬b ∧ ¬c ∧ d)

-- Only one person stole the jewelry
def one_thief (a b c d : person) : Prop :=
(a ∧ ¬b ∧ ¬c ∧ ¬d) ∨ (¬a ∧ b ∧ ¬c ∧ ¬d) ∨ (¬a ∧ ¬b ∧ c ∧ ¬d) ∨ (¬a ∧ ¬b ∧ ¬c ∧ d)

-- Main theorem statement to prove
theorem correct_choice :
  one_truth_teller statement_A statement_B statement_C statement_D ∧ 
  one_thief A B C D →
  (A = true) :=
begin
  sorry
end

end correct_choice_l73_73218


namespace john_speed_above_limit_l73_73998

theorem john_speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) 
  (h1 : distance = 150) (h2 : time = 2) (h3 : speed_limit = 60) : 
  (distance / time) - speed_limit = 15 :=
by
  -- steps to show the proof
  sorry

end john_speed_above_limit_l73_73998


namespace polynomial_roots_l73_73914

noncomputable def cubic_eq (α β γ : ℝ) := 
  (1 + 2 * α) * (β - γ) * x^3 + (2 + α) * (γ - 1) * x^2 + (3 - α) * (1 - β) * x + (4 + α) * (β - γ)

theorem polynomial_roots (α β γ : ℝ) (h: cubic_eq α β γ 1 = 0) : 
  ∃ Q : Polynomial ℝ, Q = (2 + α) * x^2 + (3 - α) * x + (4 + α) ∧ ∀ x, cubic_eq α β γ x = (x - 1) * Q :=
sorry

end polynomial_roots_l73_73914


namespace amount_after_interest_l73_73030

def initial_amount : ℝ := 440.55
def cost_of_goods : ℝ := 122.25
def interest_rate : ℝ := 0.03
def time_period : ℝ := 1

theorem amount_after_interest : 
  let remaining_amount := initial_amount - cost_of_goods in
  let interest := remaining_amount * interest_rate * time_period in
  let final_amount := remaining_amount + interest in
  final_amount = 327.85 :=
by
  -- proof steps
  sorry

end amount_after_interest_l73_73030


namespace relation_between_a_b_l73_73117

variable (f : ℝ → ℝ)
variables (a b : ℝ)

-- assuming a decreasing function on the interval (-5, +∞)
axiom h_decreasing : ∀ x y : ℝ, -5 < x → x < y → y ∈ set.Ici (-5) → f y < f x

-- defining evenness of f(x-5)
axiom h_even : ∀ x : ℝ, f (x - 5) = f (-(x - 5))

-- definitions of a and b
def a : ℝ := f (-6)
def b : ℝ := f (-3)

theorem relation_between_a_b : a > b := by
  sorry

end relation_between_a_b_l73_73117


namespace greatest_power_of_3_in_factorial_l73_73953

-- Definitions and conditions
def r : ℕ := 30.factorial

-- Main statement of the theorem
theorem greatest_power_of_3_in_factorial (k : ℕ) : 
  (3^k ∣ r) ∧ ∀ l, (3^l ∣ r) → l ≤ k := 
  k = 14 :=
begin
  sorry
end

end greatest_power_of_3_in_factorial_l73_73953


namespace knight_tour_n_eq_4_only_l73_73093

def is_knight_move (n : ℕ) (p1 p2 : Fin n × Fin n) : Prop :=
  let dx := (p1.1.val - p2.1.val).natAbs
  let dy := (p1.2.val - p2.2.val).natAbs
  (dx = 2 ∧ dy = 1) ∨ (dx = 1 ∧ dy = 2)

noncomputable def knight_tour_possible (n : ℕ) (cells : Finset (Fin n × Fin n)) : Prop :=
  ∃ (cycle : List (Fin n × Fin n)),
  (cycle.length = n + 1)
    ∧ (∀ (i : Fin n.succ), cycle.nth i ≠ none)
    ∧ (∀ (i : Fin n), is_knight_move n (cycle.nthLe i sorry) (cycle.nthLe ((i + 1) % (n + 1)) sorry))
    ∧ (cycle.nodup)
    ∧ (cycle.head? ≠ none ∧ cycle.head? = cycle.getLast sorry)

theorem knight_tour_n_eq_4_only :
  ∀ n : ℕ, (∃ cells : Finset (Fin n × Fin n),
    cells.card = n
    ∧ (∀ (p q : Fin n × Fin n), p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2)
    ∧ knight_tour_possible n cells) ↔ n = 4 :=
by sorry

end knight_tour_n_eq_4_only_l73_73093


namespace team_B_avg_speed_l73_73734

variables (v : ℝ) -- Team B's average speed
variables (tB tA : ℝ)   -- Team B's time, Team A's time

def course_length : ℝ := 300

-- Time taken by Team B to complete the course
def team_B_time (v : ℝ) : ℝ := course_length / v

-- Team A's average speed
def team_A_avg_speed (v : ℝ) : ℝ := v + 5

-- Time taken by Team A to complete the course
def team_A_time (v : ℝ) : ℝ := course_length / (team_A_avg_speed v)

-- The time difference between Team B and Team A as given in the problem
def time_difference (tB tA : ℝ) := tB - tA = 3

theorem team_B_avg_speed :
  ∃ v, team_B_time v = tB ∧
       team_A_time v = tA ∧
       time_difference tB tA ∧
       v = 20 :=
begin
  sorry
end

end team_B_avg_speed_l73_73734


namespace ratio_problem_l73_73508

theorem ratio_problem (m n p q : ℚ) 
  (h1 : m / n = 12) 
  (h2 : p / n = 4) 
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 :=
by
  sorry

end ratio_problem_l73_73508


namespace find_unique_number_l73_73935

theorem find_unique_number : 
  ∃ X : ℕ, 
    (X % 1000 = 376 ∨ X % 1000 = 625) ∧ 
    (X * (X - 1) % 10000 = 0) ∧ 
    (Nat.gcd X (X - 1) = 1) ∧ 
    ((X % 625 = 0) ∨ ((X - 1) % 625 = 0)) ∧ 
    ((X % 16 = 0) ∨ ((X - 1) % 16 = 0)) ∧ 
    X = 9376 :=
by sorry

end find_unique_number_l73_73935


namespace boat_distance_ratio_l73_73017

theorem boat_distance_ratio :
  ∀ (D_u D_d : ℝ),
  (3.6 = (D_u + D_d) / ((D_u / 4) + (D_d / 6))) →
  D_u / D_d = 4 :=
by
  intros D_u D_d h
  sorry

end boat_distance_ratio_l73_73017


namespace bestCompletion_is_advantage_l73_73824

-- Defining the phrase and the list of options
def phrase : String := "British students have a language ____ for jobs in the USA and Australia"

def options : List (String × String) := 
  [("A", "chance"), ("B", "ability"), ("C", "possibility"), ("D", "advantage")]

-- Defining the best completion function (using a placeholder 'sorry' for the logic which is not the focus here)
noncomputable def bestCompletion (phrase : String) (options : List (String × String)) : String :=
  "advantage"  -- We assume given the problem that this function correctly identifies 'advantage'

-- Lean theorem stating the desired property
theorem bestCompletion_is_advantage : bestCompletion phrase options = "advantage" :=
by sorry

end bestCompletion_is_advantage_l73_73824


namespace find_x_l73_73026

def line_segment_start := (3 : ℝ, 7 : ℝ)
def line_segment_length := 15
def line_segment_end (x : ℝ) := (x, -4)
def condition_x (x : ℝ) := x < 0
def distance (p q : ℝ × ℝ) := Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

theorem find_x :
  ∃ (x : ℝ),
  distance line_segment_start (line_segment_end x) = line_segment_length ∧
  condition_x x ∧
  x = 3 - Real.sqrt 104 :=
sorry

end find_x_l73_73026


namespace distance_AB_l73_73606

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def dist (A B : Point3D) : ℝ :=
  Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2 + (B.z - A.z) ^ 2)

def A : Point3D := ⟨2, 3, 1⟩
def B : Point3D := ⟨-2, 3, 1⟩

theorem distance_AB : dist A B = 4 :=
sorry

end distance_AB_l73_73606


namespace correct_conclusions_l73_73545

-- Assuming inclination angle and perpendicular relationship property
def inclination_perpendicular (α1 α2 : ℝ) (h : α1 ≠ α2) : Prop := abs (α1 - α2) = 90

-- Given the inclination angle of the line is obtuse
def obtuse_condition (a : ℝ) (h : -2 < a ∧ a < 0) : Prop := 
a^2 + 2 * a < 0

-- Given the inclination angle of the line x * tan(π / 7) + y = 0
def inclination_angle : Prop := tan (6 * real.pi / 7) = -tan (real.pi / 7)

-- Folding paper condition
def folding_paper (m n : ℝ) (h_folding_paper : ¬(2 * m - n + 5 = 0)) : Prop := 
m + n ≠ 36/5

-- Proof problem statement
theorem correct_conclusions (α1 α2 a m n : ℝ) (h1 : inclination_perpendicular α1 α2) 
    (h2 : obtuse_condition a) (h3 : inclination_angle) (h4 : folding_paper m n) :
  ((h1 ∧ h2 ∧ h3) ∧ ¬h4) → ((1) ∧ (2) ∧ (3)) :=
by sorry

end correct_conclusions_l73_73545


namespace mixed_oil_rate_l73_73187

-- Definitions for the given conditions
def oil1_volume := 10
def oil1_price_per_litre := 50
def oil2_volume := 5
def oil2_price_per_litre := 68

-- Let us now state the theorem we want to prove
theorem mixed_oil_rate :
  let total_volume := oil1_volume + oil2_volume in
  let total_cost := (oil1_volume * oil1_price_per_litre) + (oil2_volume * oil2_price_per_litre) in
    total_cost / total_volume = 56 :=
by
  sorry

end mixed_oil_rate_l73_73187


namespace cylinder_volume_difference_correct_l73_73677

noncomputable def Sam_radius : ℝ := 7 / (2 * Real.pi)
noncomputable def Sam_height : ℝ := 10
noncomputable def Sam_volume : ℝ := Real.pi * (Sam_radius)^2 * Sam_height

noncomputable def Chris_radius : ℝ := 12 / (2 * Real.pi)
noncomputable def Chris_height : ℝ := 9
noncomputable def Chris_volume : ℝ := Real.pi * (Chris_radius)^2 * Chris_height

noncomputable def volume_difference_multiplied_by_pi : ℝ := Real.pi * Real.abs (Chris_volume - Sam_volume)

theorem cylinder_volume_difference_correct :
  volume_difference_multiplied_by_pi = 1051 / 4 := by
    sorry

end cylinder_volume_difference_correct_l73_73677


namespace rectangle_through_four_points_exists_l73_73762

variables (A B C D : Point) (L : ℝ)

theorem rectangle_through_four_points_exists (A B C D : Point) (L : ℝ) :
  ∃ (X X' Y Y' : Point), 
  X ≠ X' ∧ 
  Y ≠ Y' ∧ 
  ∃ (rect : Rectangle), 
  ((rect.side1.pass_through_point A) ∧ 
   (rect.side2.pass_through_point B) ∧ 
   (rect.side3.pass_through_point C) ∧ 
   (rect.side4.pass_through_point D)) ∧ 
  (dist X X' = L) := sorry

end rectangle_through_four_points_exists_l73_73762


namespace compute_f_10_l73_73837

def f : ℕ → ℤ 
| 1 := 3
| 2 := 5
| (n+3) := f (n + 2) - 2 * f (n + 1) + 2 * (n + 3)

theorem compute_f_10 : f 10 = 21 := by
  sorry

end compute_f_10_l73_73837


namespace find_k_l73_73558

-- Definitions of conditions
variables (x y k : ℤ)

-- System of equations as given in the problem
def system_eq1 := x + 2 * y = 7 + k
def system_eq2 := 5 * x - y = k

-- Condition that solutions x and y are additive inverses
def y_is_add_inv := y = -x

-- The statement we need to prove
theorem find_k (hx : system_eq1 x y k) (hy : system_eq2 x y k) (hz : y_is_add_inv x y) : k = -6 :=
by
  sorry -- proof will go here

end find_k_l73_73558


namespace tan_double_angle_l73_73509

theorem tan_double_angle (α : ℝ) (h : sin α = -4 * cos α) : tan (2 * α) = 8 / 15 :=
sorry

end tan_double_angle_l73_73509


namespace max_ab_condition_max_ab_value_l73_73958

theorem max_ab_condition (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a + b = 1) (h2 : a = b) : ab = 1 / 4 :=
sorry

end max_ab_condition_max_ab_value_l73_73958


namespace slips_distribution_l73_73361

theorem slips_distribution (numbers : List ℝ) (slips : List (Set ℝ)) :
  let A := slips.nth 0
  let B := slips.nth 1
  let C := slips.nth 2
  let D := slips.nth 3
  let E := slips.nth 4
  numbers = [1.5, 1.5, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5] →
  (∃ sums : List ℝ, sums = [5, 7, 8, 9, 7]) →
  (∀ cup ∈ slips, cup.sum ∈ sums ∧ sums.nodup) →
  (∀ i j, i < j → sums.nth i < sums.nth j) →
  (2.5 ∈ C) →
  (3 ∈ B) →
  (4.5 ∈ D) :=
begin
  assume (numbers : List ℝ) (slips : List (Set ℝ)),
  let A := slips.nth 0,
  let B := slips.nth 1,
  let C := slips.nth 2,
  let D := slips.nth 3,
  let E := slips.nth 4,
  assume h_numbers : numbers = [1.5, 1.5, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5],
  assume h_sums : ∃ sums : List ℝ, sums = [5, 7, 8, 9, 7],
  assume h_sums_nodup : ∀ cup ∈ slips, cup.sum ∈ sums ∧ sums.nodup,
  assume h_increasing : ∀ i j, i < j → sums.nth i < sums.nth j,
  assume h_c : 2.5 ∈ C,
  assume h_b : 3 ∈ B,
  show 4.5 ∈ D, from sorry
end

end slips_distribution_l73_73361


namespace plane_equation_exists_l73_73097

def normal_vector (n : ℝ × ℝ × ℝ) (plane : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z, plane x y z = n.1 * x + n.2 * y + n.3 * z

def plane_contains_point (plane : ℝ → ℝ → ℝ → ℝ) (pt : ℝ × ℝ × ℝ) : Prop :=
  plane pt.1 pt.2 pt.3 = 0

def gcd_condition (a b c d : ℤ) : Prop :=
  Int.gcd (Int.gcd (Int.gcd (Int.abs a) (Int.abs b)) (Int.abs c)) (Int.abs d) = 1

theorem plane_equation_exists :
  ∃ (A B C D : ℤ),
    A > 0 ∧
    normal_vector (3, -2, 4) (λ x y z, A * x + B * y + C * z + D) ∧
    plane_contains_point (λ x y z, 3 * x - 2 * y + 4 * z + D) (2, -3, 5) ∧
    gcd_condition A B C D :=
begin
  sorry
end

end plane_equation_exists_l73_73097


namespace p_n_div_5_iff_not_mod_4_zero_l73_73518

theorem p_n_div_5_iff_not_mod_4_zero (n : ℕ) (h : 0 < n) : 
  (1 + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
by {
  sorry
}

end p_n_div_5_iff_not_mod_4_zero_l73_73518


namespace sum_of_exponents_and_coefficients_of_3144_l73_73091

theorem sum_of_exponents_and_coefficients_of_3144 :
  let p2 : ℕ → ℕ := λ n, 2^n in
  let exps := [11, 10, 6, 3] in
  let coeffs := [1, 1, 1, 1] in
  (sum exps = 30) ∧ (sum coeffs = 4) := by
  let p2 : ℕ → ℕ := λ n, 2^n
  let exps := [11, 10, 6, 3]
  let coeffs := [1, 1, 1, 1]
  have h_exp : sum exps = 30 := by {
    -- Proof-calculation would go here.
    sorry
  }
  have h_coeff : sum coeffs = 4 := by {
    -- Proof-calculation would go here.
    sorry
  }
  exact ⟨h_exp, h_coeff⟩

end sum_of_exponents_and_coefficients_of_3144_l73_73091


namespace largest_derivative_at_1_l73_73318

noncomputable def f1 := λ x : ℝ, -x^2
noncomputable def f2 := λ x : ℝ, 1 / x
noncomputable def f3 := λ x : ℝ, 2 * x + 1
noncomputable def f4 := λ x : ℝ, sqrt x

lemma f1_derivative_at_1 : (deriv f1 1) = -2 := sorry
lemma f2_derivative_at_1 : (deriv f2 1) = -1 := sorry
lemma f3_derivative_at_1 : (deriv f3 1) = 2 := sorry
lemma f4_derivative_at_1 : (deriv f4 1) = 1 / 2 := sorry

theorem largest_derivative_at_1 : 
  (deriv f3 1) > (deriv f1 1) ∧
  (deriv f3 1) > (deriv f2 1) ∧
  (deriv f3 1) > (deriv f4 1) :=
by {
  sorry
}

end largest_derivative_at_1_l73_73318


namespace probability_white_then_black_l73_73011

-- Definition of conditions
def total_balls := 5
def white_balls := 3
def black_balls := 2

def first_draw_white_probability (total white : ℕ) : ℚ :=
  white / total

def second_draw_black_probability (remaining_white remaining_black : ℕ) : ℚ :=
  remaining_black / (remaining_white + remaining_black)

-- The theorem statement
theorem probability_white_then_black :
  first_draw_white_probability total_balls white_balls *
  second_draw_black_probability (total_balls - 1) black_balls
  = 3 / 10 :=
by
  sorry

end probability_white_then_black_l73_73011


namespace T_n_formula_l73_73528

noncomputable def a_n : ℕ → ℤ :=
  λ n, 14 - 2 * n

lemma a1_eq_12 : a_n 1 = 12 := by
  simp [a_n]

lemma S10_eq_30 : (10 * 12 + (10 * (9 / 2)) * (-2) = 30) := by
  norm_num

def b_n : ℕ → ℤ :=
  λ n, abs (12 - 2 * n)

noncomputable def T_n : ℕ → ℤ
  | n, hn := if hn : n ≤ 6 then
    -n^2 + 11*n
  else
    n^2 - 11*n + 60

theorem T_n_formula (n : ℕ) : T_n n =
  if n ≤ 6 then
    -n^2 + 11*n
  else
    n^2 - 11*n + 60 := by
  simp [T_n]; split_ifs; try {norm_num}; sorry

end T_n_formula_l73_73528


namespace triangle_ABC_area_l73_73254

-- Define the conditions for triangle ABC
variables (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variables (AB AC AM BC : ℝ)

-- Define values for sides and median according to the problem statement
axiom AB_eq_9 : AB = 9
axiom AC_eq_17 : AC = 17
axiom AM_eq_12 : AM = 12

-- Define that M is the midpoint of BC
axiom is_midpoint_MBC : ∀ B C : ℝ, M = (B + C) / 2

-- Function to calculate area of a triangle given three sides using Heron's formula
noncomputable def heron_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem stating the area of the triangle ABC
theorem triangle_ABC_area : heron_area AB AC BC = 110 :=
sorry

end triangle_ABC_area_l73_73254


namespace total_cost_of_toys_l73_73994

def cost_of_toy_cars : ℝ := 14.88
def cost_of_toy_trucks : ℝ := 5.86

theorem total_cost_of_toys :
  cost_of_toy_cars + cost_of_toy_trucks = 20.74 :=
by
  sorry

end total_cost_of_toys_l73_73994


namespace actual_time_when_watch_shows_8_PM_l73_73828

-- Definitions based on the problem's conditions
def initial_time := 8  -- 8:00 AM
def incorrect_watch_time := 14 * 60 + 42  -- 2:42 PM converted to minutes
def actual_time := 15 * 60  -- 3:00 PM converted to minutes
def target_watch_time := 20 * 60  -- 8:00 PM converted to minutes

-- Define to calculate the rate of time loss
def time_loss_rate := (actual_time - incorrect_watch_time) / (actual_time - initial_time * 60)

-- Hypothesis that the watch loses time at a constant rate
axiom constant_rate : ∀ t, t >= initial_time * 60 ∧ t <= actual_time → (t * time_loss_rate) = (actual_time - incorrect_watch_time)

-- Define the target time based on watch reading 8:00 PM
noncomputable def target_actual_time := target_watch_time / time_loss_rate

-- Main theorem: Prove that given the conditions, the target actual time is 8:32 PM
theorem actual_time_when_watch_shows_8_PM : target_actual_time = (20 * 60 + 32) :=
sorry

end actual_time_when_watch_shows_8_PM_l73_73828


namespace eight_letter_good_words_count_l73_73479

-- Define a function to count good words of length 8
def count_good_words : ℕ :=
  let letters := ['A', 'B', 'C'] in
  let valid_next := λ c, match c with
    | 'A' => ['A', 'C']
    | 'B' => ['A', 'B']
    | 'C' => ['B', 'C']
    | _ => []  -- This case should never be reached
  in
  (letters.length) * (valid_next 'A').length ^ 6

theorem eight_letter_good_words_count : count_good_words = 192 :=
  by {
    -- Here we define the count_good_words to be 192 based on the conditions
    -- but we won't provide the proof here
    sorry
  }

end eight_letter_good_words_count_l73_73479


namespace above_limit_l73_73995

/-- John travels 150 miles in 2 hours. -/
def john_travel_distance : ℝ := 150

/-- John travels for 2 hours. -/
def john_travel_time : ℝ := 2

/-- The speed limit is 60 mph. -/
def speed_limit : ℝ := 60

/-- The speed of John during his travel. -/
def john_speed : ℝ := john_travel_distance / john_travel_time

/-- How many mph above the speed limit was John driving? -/
def speed_above_limit : ℝ := john_speed - speed_limit

theorem above_limit : speed_above_limit = 15 := 
by
  unfold speed_above_limit john_speed john_travel_distance john_travel_time speed_limit
  have h1: 150 / 2 = 75 := by norm_num
  have h2: 75 - 60 = 15 := by norm_num
  rw [h1, h2]
  refl

end above_limit_l73_73995


namespace interval_contains_solution_l73_73084

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 2

theorem interval_contains_solution :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

end interval_contains_solution_l73_73084


namespace rectangle_circle_dimensions_l73_73692

theorem rectangle_circle_dimensions :
  ∃ (a b : ℝ), (2 * (a + b) = 6 * Real.pi) ∧ (a = 2 * b) ∧ (a = 2 * Real.pi) ∧ (b = Real.pi) :=
by
  use (2 * Real.pi)
  use Real.pi
  constructor
  { -- Show 2 * (a + b) = 6 * π
    rw [add_mul, mul_assoc],
    norm_num,
    ring,
    simp },
  constructor
  { -- Show a = 2 * b
    norm_num,
    ring },
  constructor
  { -- Show a = 2 * π
    ring },
  { -- Show b = π
    ring }

end rectangle_circle_dimensions_l73_73692


namespace tan_sum_alpha_beta_l73_73550

theorem tan_sum_alpha_beta
    (α β : ℝ)
    (h1 : ∀ x y : ℝ, x * (tan α) - y - 3 * (tan β) = 0)
    (h2 : tan α = 2)
    (h3 : tan β = -1/3):
    tan (α + β) = 1 := 
by
  sorry

end tan_sum_alpha_beta_l73_73550


namespace coeff_x2_term_l73_73482

-- Define the polynomial expression to be expanded
def polynomial_expr : polynomial ℚ := (1 + 2 * polynomial.X)^3 * (1 - polynomial.X)^4

-- Function to extract the coefficient of a given term
def coeff_of_term (p : polynomial ℚ) (n : ℕ) : ℚ := p.coeff n

-- Proof statement to verify the coefficient of x² term is -6
theorem coeff_x2_term : coeff_of_term polynomial_expr 2 = -6 :=
by
  -- Provide a sorry since the actual proof is not required
  sorry

end coeff_x2_term_l73_73482


namespace solve_star_eq_five_l73_73870

def star (a b : ℝ) : ℝ := a + b^2

theorem solve_star_eq_five :
  ∃ x₁ x₂ : ℝ, star x₁ (x₁ + 1) = 5 ∧ star x₂ (x₂ + 1) = 5 ∧ x₁ = 1 ∧ x₂ = -4 :=
by
  sorry

end solve_star_eq_five_l73_73870


namespace min_value_m_range_of_x_l73_73129

open Real

theorem min_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + 3 * b^2 = 3) : sqrt 5 * a + b ≤ 4 :=
begin
  sorry
end

theorem range_of_x (x : ℝ) : (2 * |x - 1| + |x| ≥ 4) ↔ (x ≤ -2/3 ∨ x ≥ 2) :=
begin
  sorry
end

end min_value_m_range_of_x_l73_73129


namespace score_analysis_l73_73964

theorem score_analysis :
  let scores := [92, 89, 95, 91, 93]
  let remaining_scores := (scores.erase 95).erase 89 in
  (remaining_scores.mean = 92) ∧ (remaining_scores.variance = 2 / 3) :=
by
  sorry

end score_analysis_l73_73964


namespace eva_fruit_diet_l73_73090

noncomputable def dietary_requirements : Prop :=
  ∃ (days_in_week : ℕ) (days_in_month : ℕ) (apples : ℕ) (bananas : ℕ) (pears : ℕ) (oranges : ℕ),
    days_in_week = 7 ∧
    days_in_month = 30 ∧
    apples = 2 * days_in_week ∧
    bananas = days_in_week / 2 ∧
    pears = 4 ∧
    oranges = days_in_month / 3 ∧
    apples = 14 ∧
    bananas = 4 ∧
    pears = 4 ∧
    oranges = 10

theorem eva_fruit_diet : dietary_requirements :=
sorry

end eva_fruit_diet_l73_73090


namespace expression_non_negative_l73_73876

theorem expression_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 :=
by
  sorry

end expression_non_negative_l73_73876


namespace hyperbola_equation_l73_73155

theorem hyperbola_equation 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h_asymptote : ∀ x y, y = 4/3 * x ∨ y = -4/3 * x ↔ y^2 / (4/3 * x)^2 = 1)
  (h_focus : ∀ x y, x = 5 ∧ y = 0 ↔ x^2 + y^2 = 25) :
  C = (∃ a b : ℝ, a^2 = 9 ∧ b^2 = 16 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) :=
begin
  sorry
end

end hyperbola_equation_l73_73155


namespace words_per_page_l73_73405

theorem words_per_page (p : ℕ) 
  (h1 : 180 > 0) (h2 : ∀ i, 1 ≤ p ∧ p ≤ 150)
  (h3 : 180 * p ≡ 203 [MOD 229]) : p = 94 :=
by sorry

end words_per_page_l73_73405


namespace sum_irrational_is_irrational_l73_73304

theorem sum_irrational_is_irrational (a b : ℚ) (ha : irrational (real.sqrt a)) (hb : irrational (real.sqrt b)) :
  irrational (real.sqrt a + real.sqrt b) :=
sorry

end sum_irrational_is_irrational_l73_73304


namespace john_children_probability_l73_73263

open Nat

def probability_at_least_half_girls : Prop :=
  let total_children := 7
  let total_outcomes := 2 ^ total_children
  let favorable_outcomes := (binomial total_children 4) + (binomial total_children 5) + (binomial total_children 6) + (binomial total_children 7)
  favorable_outcomes / total_outcomes = 1 / 2

theorem john_children_probability : probability_at_least_half_girls :=
by
  sorry

end john_children_probability_l73_73263


namespace find_perpendicular_line_through_point_l73_73857

noncomputable def perpendicular_line_equation (x0 y0 : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) := x0 + f' 1 * y0 - (f x0) * f' 1 - 1 = 0

theorem find_perpendicular_line_through_point {x y : ℝ} (h1 : y = Real.exp x) (h2 : (1, Real.exp 1) = (1, e)) :
  perpendicular_line_equation 1 e Real.exp (λ x, Real.exp x) := 
by {
  -- Condition: P(1, e) is on the curve y = e^x
  have h : e = Real.exp 1, sorry,
  exact sorry
}

end find_perpendicular_line_through_point_l73_73857


namespace total_num_animals_l73_73172

-- Given conditions
def num_pigs : ℕ := 10
def num_cows : ℕ := (2 * num_pigs) - 3
def num_goats : ℕ := num_cows + 6

-- Theorem statement
theorem total_num_animals : num_pigs + num_cows + num_goats = 50 := 
by
  sorry

end total_num_animals_l73_73172


namespace find_S20_l73_73556

def sequence :=
  ∀ n : ℕ, { a_n : ℕ // 2 * (finset.range (n+1)).sum a_n = a_n * a_(n+1) }

def a1 := 1

theorem find_S20 : 
  (∀ a : sequence, a_1 = 1 ∧ (∀ n, 2 * (∑ i in finset.range (n+1), a.n i) = a.n (n+1) * a.n n) → 
  (∑ i in finset.range 21, a_20 i) = 210) := 
sorry

end find_S20_l73_73556


namespace Marcella_initial_pairs_l73_73290

theorem Marcella_initial_pairs (shoes_lost shoes_left : ℕ) (pairs_left : ℕ) (h_lost : shoes_lost = 9) (h_left : pairs_left = 21) :
  let initial_individual_shoes := (pairs_left * 2) + shoes_lost in
  let initial_pairs := initial_individual_shoes / 2 in
  initial_pairs = 25 :=
by 
  intros
  sorry

end Marcella_initial_pairs_l73_73290


namespace relationship_y1_y2_l73_73201

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l73_73201


namespace relationship_y1_y2_l73_73197

theorem relationship_y1_y2 :
  let f : ℝ → ℝ := λ x, 2 * x + 1 in
  let y1 := f (-3) in
  let y2 := f 4 in
  y1 < y2 :=
by {
  -- definitions
  let f := λ x, 2 * x + 1,
  let y1 := f (-3),
  let y2 := f 4,
  -- calculations
  have h1 : y1 = f (-3) := rfl,
  have h2 : y2 = f 4 := rfl,
  -- compare y1 and y2
  rw [h1, h2],
  exact calc
    y1 = f (-3) : rfl
    ... = 2 * (-3) + 1 : rfl
    ... = -5 : by norm_num
    ... < 2 * 4 + 1 : by norm_num
    ... = y2 : rfl
}

end relationship_y1_y2_l73_73197
