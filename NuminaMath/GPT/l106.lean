import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Functional
import Mathlib.Algebra.Quadratics
import Mathlib.Algebra.Ring.InjSurj
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.Geometry.Rectangle
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Binomial
import Mathlib.Combinatorics.Graph.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.Constructions

namespace circle_radii_l106_106745

noncomputable def smaller_circle_radius (r : ℝ) :=
  r = 4

noncomputable def larger_circle_radius (r : ℝ) :=
  r = 9

theorem circle_radii (r : ℝ) (h1 : ∀ (r: ℝ), (r + 5) - r = 5) (h2 : ∀ (r: ℝ), 2.4 * r = 2.4 * r):
  smaller_circle_radius r → larger_circle_radius (r + 5) :=
by
  sorry

end circle_radii_l106_106745


namespace number_composite_l106_106171

theorem number_composite : ¬ prime (10^1962 + 1) :=
sorry

end number_composite_l106_106171


namespace arithmetic_sequence_proof_l106_106590

noncomputable def arithmetic_sequence_solution : ℕ → ℝ := λ n, 2 * n - 1

noncomputable def Sn (n : ℕ) : ℝ := n^2

theorem arithmetic_sequence_proof (n : ℕ) : 
  (∀ n, 4 * Sn n = (arithmetic_sequence_solution n + 1)^2) → 
  arithmetic_sequence_solution 1 = 1 ∧ 
  arithmetic_sequence_solution 2 = 3 ∧ 
  (∀ n, arithmetic_sequence_solution n = 2 * n - 1) ∧ 
  (∃ n, (Sn n - 7 / 2 * arithmetic_sequence_solution n) = -17/2) :=
by 
  intro h,
  sorry

end arithmetic_sequence_proof_l106_106590


namespace train_length_approx_200_l106_106048

noncomputable def length_of_train (time_crossing_pole : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_crossing_pole

theorem train_length_approx_200 :
  length_of_train 5.80598713393251 124 ≈ 200 :=
sorry

end train_length_approx_200_l106_106048


namespace train_length_from_speed_l106_106400

-- Definitions based on conditions
def seconds_to_cross_post : ℕ := 40
def seconds_to_cross_bridge : ℕ := 480
def bridge_length_meters : ℕ := 7200

-- Theorem statement to be proven
theorem train_length_from_speed :
  (bridge_length_meters / seconds_to_cross_bridge) * seconds_to_cross_post = 600 :=
sorry -- Proof is not provided

end train_length_from_speed_l106_106400


namespace cubic_two_common_points_x_axis_l106_106166

theorem cubic_two_common_points_x_axis (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ^ 3 - 3 * x1 + c = 0 ∧ x2 ^ 3 - 3 * x2 + c = 0 ∧
    (∀ x ∈ Ioo (-1 : ℝ) 1, x^3 - 3 * x + c > 0) ∧ 
    ((∀ x ≤ -1, x^3 - 3*x + c ≠ 0) ∨ (∀ x ≥ 1, x^3 - 3*x + c ≠ 0)))
  ↔ c = -2 ∨ c = 2 :=
by
  sorry

end cubic_two_common_points_x_axis_l106_106166


namespace geo_seq_ratio_l106_106143

variable {a : ℕ → ℝ}

-- Conditions
axiom pos_terms : ∀ n, 0 < a n
axiom arith_seq : 3 * a 1, 1 / 2 * a 3, 2 * a 2 form an arithmetic sequence

-- Question
theorem geo_seq_ratio : ∀ {a : ℕ → ℝ} (pos_terms : ∀ n, 0 < a n) (arith_seq : 3 * a 1, 1 / 2 * a 3, 2 * a 2 form an arithmetic sequence), 
  \frac{a 8 + a 9}{a 6 + a 7} = 9 :=
sorry

end geo_seq_ratio_l106_106143


namespace find_value_of_a_l106_106587

noncomputable def value_of_a : ℝ := 
  let A := (-1 : ℝ, a) in
  let B := (a, 8 : ℝ) in
  let line_through_A_B := B.2 - A.2 = k * (B.1 - A.1) in
  let given_line := 2 = k in
  a

theorem find_value_of_a (a : ℝ) : 
  ∃ a : ℝ, ∀ k : ℝ, 
  (B.2 - A.2 = k * (B.1 - A.1) → 2 = k) → 
  a :=
begin
  sorry
end

end find_value_of_a_l106_106587


namespace initial_number_is_correct_l106_106370

theorem initial_number_is_correct (x : ℝ) (h : 8 * x - 4 = 2.625) : x = 0.828125 :=
by
  sorry

end initial_number_is_correct_l106_106370


namespace wrestler_teams_possible_l106_106315

theorem wrestler_teams_possible :
  ∃ (team1 team2 team3 : Finset ℕ),
  (team1 ∪ team2 ∪ team3 = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (team1 ∩ team2 = ∅) ∧ (team1 ∩ team3 = ∅) ∧ (team2 ∩ team3 = ∅) ∧
  (team1.card = 3) ∧ (team2.card = 3) ∧ (team3.card = 3) ∧
  (team1.sum id = 15) ∧ (team2.sum id = 15) ∧ (team3.sum id = 15) ∧
  (∀ x ∈ team1, ∀ y ∈ team2, x > y) ∧
  (∀ x ∈ team2, ∀ y ∈ team3, x > y) ∧
  (∀ x ∈ team3, ∀ y ∈ team1, x > y) := sorry

end wrestler_teams_possible_l106_106315


namespace problem_evaluation_l106_106091

theorem problem_evaluation : (726 * 726) - (725 * 727) = 1 := 
by 
  sorry

end problem_evaluation_l106_106091


namespace variance_of_data_set_l106_106203

noncomputable def variance (X : List ℝ) : ℝ :=
let n := X.length
let mean := (X.sum / n)
let sum_sq_diff := Finset.univ.sum (λ i, (X.nthLe i (by linarith) - mean)^2)
in sum_sq_diff / (n - 1)

theorem variance_of_data_set :
  variance [4, 2, 1, 0, 0, 0, 0] = 7 / 3 :=
by
  sorry

end variance_of_data_set_l106_106203


namespace book_cost_l106_106726

theorem book_cost 
  (initial_money : ℕ) 
  (num_books : ℕ) 
  (money_left : ℕ) 
  (h_init : initial_money = 79) 
  (h_books : num_books = 9) 
  (h_left : money_left = 16) : 
  (initial_money - money_left) / num_books = 7 :=
by
  rw [h_init, h_books, h_left] 
  norm_num
  sorry

end book_cost_l106_106726


namespace incorrect_variance_min_standard_deviation_l106_106044

theorem incorrect_variance (a b : ℝ) (h1 : a^2 + b^2 = 9) (h2 : (3 + a + b) / 3 > 1) : 
  (6 - (a + b + 3) / 3)^2 ≠ 2 :=
by
  sorry

theorem min_standard_deviation (a b : ℝ) (h1 : a^2 + b^2 = 9) :
  (minSd : ℝ) × (legs : ℝ × ℝ) :=
by
  let s := sqrt 2 - 1
  let l := 3 * sqrt 2 / 2
  exist (s, (l, l))
  sorry

end incorrect_variance_min_standard_deviation_l106_106044


namespace bird_population_decline_l106_106514

noncomputable theory

open Real

theorem bird_population_decline :
  ∃ n : ℕ, (P_0 : ℝ) = 100 ∧ (P : ℝ → ℝ) = λ n, 100 * (0.65 ^ n) ∧ P n < 15 → n = 5 :=
by {
  let P_0 := 100,
  let P := λ n, 100 * (0.65 ^ n),
  have h_inequality : ∀ n, P n < 15 ↔ 0.65 ^ n < 0.15, {
    intro n,
    simp [P],
    split,
    { intros h, 
      -- Proof omitted, just need assumptions in Lean
      sorry
    },
    { intros h, 
      -- Proof omitted, just need assumptions in Lean
      sorry
    },
  },
  use 5,
  split,
  { exact rfl, },
  split,
  { exact rfl, },
  {
    -- Verify by computing the logarithmic check
    let n_bound := log 0.15 / log 0.65,
    have h_n_bound : 4.404 < 5 := by norm_num,
    sorry,
  }
}

end bird_population_decline_l106_106514


namespace parabola_focus_l106_106634

theorem parabola_focus (a : ℝ) (h1 : ∀ x y, x^2 = a * y ↔ y = x^2 / a)
(h2 : focus_coordinates = (0, 5)) : a = 20 := 
sorry

end parabola_focus_l106_106634


namespace point_distance_to_focus_of_parabola_with_focus_distance_l106_106585

def parabola_with_focus_distance (focus_distance : ℝ) (p : ℝ × ℝ) : Prop :=
  let f := (0, focus_distance)
  let directrix := -focus_distance
  let (x, y) := p
  let distance_to_focus := Real.sqrt ((x - 0)^2 + (y - focus_distance)^2)
  let distance_to_directrix := abs (y - directrix)
  distance_to_focus = distance_to_directrix

theorem point_distance_to_focus_of_parabola_with_focus_distance 
  (focus_distance : ℝ) (y_axis_distance : ℝ) (p : ℝ × ℝ)
  (h_focus_distance : focus_distance = 4)
  (h_y_axis_distance : abs (p.1) = 1) :
  parabola_with_focus_distance focus_distance p →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - focus_distance)^2) = 5 :=
by
  sorry

end point_distance_to_focus_of_parabola_with_focus_distance_l106_106585


namespace complex_number_solution_l106_106146

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i^2 = -1) (hz : i * (z - 1) = 1 - i) : z = -i :=
by sorry

end complex_number_solution_l106_106146


namespace tetrahedron_vertex_angle_sum_l106_106716

theorem tetrahedron_vertex_angle_sum (A B C D : Type) (angles_at : Type → Type → Type → ℝ) :
  (∃ A, (∀ X Y Z W, X = A ∨ Y = A ∨ Z = A ∨ W = A → angles_at X Y A + angles_at Z W A > 180)) →
  ¬ (∃ A B, A ≠ B ∧ 
    (∀ X Y, X = A ∨ Y = A → angles_at X Y A + angles_at Y X A > 180) ∧ 
    (∀ X Y, X = B ∨ Y = B → angles_at X Y B + angles_at Y X B > 180)) := 
sorry

end tetrahedron_vertex_angle_sum_l106_106716


namespace area_relation_square_area_relation_rectangle_l106_106132

section ProofProblem

variables {A B C D E F : Type} [linear_ordered_field F] 
variables (A B C D E F : F × F)
variable (equilateral : ∀ {A B E: F × F}, equilateral_triangle A B E)
variables (convex : ∀ {A C D F: F × F}, convex_quad A C D F)
variables (common_vertex : (A.1 = 0 ∧ A.2 = 0))
variables (B_on_CF : B.1 > C.1 ∧ B.1 < F.1 ∧ B.2 > 0 ∧ B.2 < F.2)
variables (E_on_FD : E.1 > F.1 ∧ E.1 < D.1 ∧ E.2 > 0 ∧ E.2 < D.2)

theorem area_relation_square 
  (square : ∀ {A C D F: F × F}, is_square A C D F) :
  area ADE + area ABC = area BEF :=
sorry

theorem area_relation_rectangle 
  (rectangle : ∀ {A C D F: F × F}, is_rectangle A C D F) :
  area ADE + area ABC = area BEF :=
sorry

end ProofProblem

end area_relation_square_area_relation_rectangle_l106_106132


namespace find_missing_score_l106_106111

theorem find_missing_score
  (scores : List ℕ)
  (h_scores : scores = [73, 83, 86, 73, x])
  (mean : ℚ)
  (h_mean : mean = 79.2)
  (h_length : scores.length = 5)
  : x = 81 := by
  sorry

end find_missing_score_l106_106111


namespace average_age_of_boys_l106_106208

theorem average_age_of_boys
  (N : ℕ) (G : ℕ) (A_G : ℕ) (A_S : ℚ) (B : ℕ)
  (hN : N = 652)
  (hG : G = 163)
  (hA_G : A_G = 11)
  (hA_S : A_S = 11.75)
  (hB : B = N - G) :
  (163 * 11 + 489 * x = 11.75 * 652) → x = 12 := by
  sorry

end average_age_of_boys_l106_106208


namespace intersection_A_B_l106_106913

def A : Set ℝ := { x | log 2 (2 - x) ≤ 1 }
def B : Set ℝ := { -1, 0, 1, 2 }

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l106_106913


namespace truncated_cone_volume_l106_106824

theorem truncated_cone_volume 
  (V_initial : ℝ)
  (r_ratio : ℝ)
  (V_final : ℝ)
  (r_ratio_eq : r_ratio = 1 / 2)
  (V_initial_eq : V_initial = 1) :
  V_final = 7 / 8 :=
  sorry

end truncated_cone_volume_l106_106824


namespace contact_prob_correct_l106_106532

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end contact_prob_correct_l106_106532


namespace smallest_positive_period_monotonically_decreasing_interval_length_of_side_a_l106_106154

noncomputable def f (x : ℝ) := sin (2 * x + π / 6) + 2 * (sin x)^2

theorem smallest_positive_period : 
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
by sorry

theorem monotonically_decreasing_interval (k : ℤ) : 
  ∀ x, (x ∈ [π / 3 + k * π, 5 * π / 6 + k * π]) → (∀ y, y ∈ [x, x + π] → f y ≤ f x) :=
by sorry

variables (A b c : ℝ)
variables (A_pos : 0 < A) (A_le_pi : A ≤ π)
variables (b_pos : 0 < b) (c_pos : 0 < c)
variables (A_cond : f (A / 2) = 3 / 2) (bc_sum : b + c = 7)
variables (area_cond : 1 / 2 * b * c * sin A = 2 * sqrt 3)

theorem length_of_side_a 
  (a : ℝ) : 
  a = 5 :=
by sorry

end smallest_positive_period_monotonically_decreasing_interval_length_of_side_a_l106_106154


namespace inequlity_for_k_one_smallest_k_l106_106693

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

theorem inequlity_for_k_one (a b c : ℝ) (h : triangle_sides a b c) :
  a^3 + b^3 + c^3 < (a + b + c) * (a * b + b * c + c * a) :=
sorry

theorem smallest_k (a b c k : ℝ) (h : triangle_sides a b c) (hk : k = 1) :
  a^3 + b^3 + c^3 < k * (a + b + c) * (a * b + b * c + c * a) :=
sorry

end inequlity_for_k_one_smallest_k_l106_106693


namespace variance_incorrect_min_std_deviation_l106_106038

-- Definitions for the given conditions.
variable (a b : ℝ)

-- The right triangle condition given by Pythagorean theorem.
def right_triangle (a b : ℝ) : Prop :=
  a^2 + b^2 = 9

-- Problems to verify
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b) : 
  ¬(variance {a, b, 3} = 2) := sorry

theorem min_std_deviation {a b : ℝ} (h : right_triangle a b) :
  let s := sqrt(2) - 1,
  (a = b) → (a = 3 * sqrt(2) / 2) → (std_deviation {a, b, 3} = s) := sorry

end variance_incorrect_min_std_deviation_l106_106038


namespace total_cost_of_cow_and_calf_l106_106383

-- Define variables for the cost of the cow and the cost of the calf.
def cost_of_cow : ℝ := 880
def cost_of_calf : ℝ := 110

-- Define the condition that "the cost of the cow is 8 times the cost of the calf."
def cow_is_8_times_calf (c : ℝ) (calf : ℝ) : Prop := c = 8 * calf

-- Define the main theorem stating the total cost of the cow and the calf.
theorem total_cost_of_cow_and_calf : 
  cost_of_cow = 880 ∧ cost_of_calf = 110 → cost_of_cow + cost_of_calf = 990 :=
by
  intros h,
  cases h with hcow hcalf,
  rw [hcow, hcalf],
  norm_num,
  sorry

end total_cost_of_cow_and_calf_l106_106383


namespace watermelon_volume_correct_l106_106841

noncomputable def volume_of_watermelon (r_cross_section : ℝ) (distance_center : ℝ) : ℝ :=
  let r_sphere := Real.sqrt (r_cross_section^2 + distance_center^2)
  in (4 / 3) * Real.pi * r_sphere^3

theorem watermelon_volume_correct :
  volume_of_watermelon 4 3 = (500 / 3) * Real.pi :=
by
  sorry

end watermelon_volume_correct_l106_106841


namespace pascal_triangle_ratio_l106_106639

theorem pascal_triangle_ratio (n r : ℕ) 
  (h1 : (3 * r + 3 = 2 * n - 2 * r))
  (h2 : (4 * r + 8 = 3 * n - 3 * r - 3)) : 
  n = 34 :=
sorry

end pascal_triangle_ratio_l106_106639


namespace count_augmented_visible_factor_numbers_200_250_l106_106834

def is_augmented_visible_factor_number (n : ℕ) : Prop :=
  let digits := [2, (n / 10) % 10, n % 10] in
  let nonzero_digits := digits.filter (λ d => d ≠ 0) in
  let digit_sum := digits.sum in
  nonzero_digits.all (λ d => n % d = 0) ∧ n % digit_sum = 0 ∧ n % ((n / 10) % 10) = 0

def count_augmented_visible_factor_numbers (a b : ℕ) : ℕ :=
  ((a ≤ b) → List.range (b - a + 1)).filter (λ x => is_augmented_visible_factor_number (a + x)).length

theorem count_augmented_visible_factor_numbers_200_250 :
  count_augmented_visible_factor_numbers 200 250 = 10 :=
sorry

end count_augmented_visible_factor_numbers_200_250_l106_106834


namespace sum_first_2500_terms_l106_106839

def sequence_sum (b : ℕ → ℤ) (n : ℕ) : ℤ :=
  (list.range n).map b |>.sum

noncomputable def sequence (b : ℕ → ℤ) : Prop :=
  ∀ n, n ≥ 3 → b n = b (n - 1) - b (n - 2)

theorem sum_first_2500_terms
  (b : ℕ → ℤ)
  (h0 : sequence b)
  (h1 : sequence_sum b 2000 = 2501)
  (h2 : sequence_sum b 2501 = 2000) :
  sequence_sum b 2500 = 2501 :=
begin
  sorry,
end

end sum_first_2500_terms_l106_106839


namespace series_converges_to_l106_106073

noncomputable def series_sum := ∑' n : Nat, (4 * n + 3) / ((4 * n + 1) ^ 2 * (4 * n + 5) ^ 2)

theorem series_converges_to : series_sum = 1 / 200 := 
by 
  sorry

end series_converges_to_l106_106073


namespace line_through_point_with_equal_intercepts_l106_106900

theorem line_through_point_with_equal_intercepts (a b : ℝ) :
  (∀ x y : ℝ, (x = 1 ∧ y = 2) → (↥(x + y = a ∨ y = b * x))) → (a = 3 ∧ b = 2) :=
by
  sorry

end line_through_point_with_equal_intercepts_l106_106900


namespace binom_18_10_l106_106437

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106437


namespace count_multiples_less_than_300_l106_106979

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end count_multiples_less_than_300_l106_106979


namespace number_of_integers_satisfying_inequality_l106_106180

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 / |(n:ℤ)| : ℚ) ≥ 1 / 5}.to_finset.card = 10 :=
by
  sorry

end number_of_integers_satisfying_inequality_l106_106180


namespace rohan_conveyance_expense_l106_106270

theorem rohan_conveyance_expense :
  ∀ (salary savings food_percent house_rent_percent entertainment_percent conveyance_percent : ℝ),
    salary = 10000 →
    savings = 2000 →
    food_percent = 0.40 →
    house_rent_percent = 0.20 →
    entertainment_percent = 0.10 →
    conveyance_percent = 0.10 →
    (conveyance_percent = (10 / 100) : ℝ) :=
  sorry

end rohan_conveyance_expense_l106_106270


namespace average_weight_of_removed_carrots_l106_106361

def total_weight_30_carrots_kg : ℝ := 5.94
def total_weight_30_carrots_g : ℕ := (total_weight_30_carrots_kg * 1000).toNat
def num_carrots_removed : ℕ := 3
def num_carrots_remaining : ℕ := 27
def avg_weight_remaining_carrots_g : ℕ := 200
def total_weight_remaining_carrots_g : ℕ := num_carrots_remaining * avg_weight_remaining_carrots_g
def total_weight_removed_carrots_g : ℕ := total_weight_30_carrots_g - total_weight_remaining_carrots_g
def avg_weight_removed_carrots_g : ℕ := total_weight_removed_carrots_g / num_carrots_removed

theorem average_weight_of_removed_carrots :
  avg_weight_removed_carrots_g = 180 :=
by
  -- The proof steps will go here, but currently omitted as per instruction
  sorry

end average_weight_of_removed_carrots_l106_106361


namespace distance_from_library_to_post_office_l106_106232

theorem distance_from_library_to_post_office :
  ∀ (d1 d2 d3 d_total : ℝ),
    d1 = 0.3 ∧ d3 = 0.4 ∧ d_total = 0.8 →
    d2 = d_total - d1 - d3 →
    d2 = 0.1 :=
by
  intros d1 d2 d3 d_total h1 h2
  cases h1 with h1_1 h1_rest
  cases h1_rest with h1_3 h1_total
  rw [h1_1, h1_3, h1_total] at h2
  exact h2

end distance_from_library_to_post_office_l106_106232


namespace problem1_problem2_problem3_l106_106228

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (T : ℝ) (h : ∀ x, f(x + T) = T * f(x)) : 
  ¬ (f = λ x, x) :=
by sorry

-- Problem 2
theorem problem2 (f : ℝ → ℝ) (T : ℝ) (hT : T ≠ 0) (h : ∀ x, f(x + T) = T * f(x)) : 
  (f = λ x, x + ln x → ∀ x, -3 < x → x < -2 → f x = 1/4 * (x + 4 + ln (x + 4))) :=
by sorry

-- Problem 3
theorem problem3 (f : ℝ → ℝ) (T : ℝ) (h : ∀ x, f(x + T) = T * f(x)) : 
  (f = λ x, sin (k * x) → ∃ m : ℤ, k = m * real.pi) :=
by sorry

end problem1_problem2_problem3_l106_106228


namespace support_strip_width_l106_106805

variables (A B : set ℝ) 
variables [convex ℝ A] [convex ℝ B]
variables (a b c : ℝ)

-- Assumes A and B are bounded convex sets
noncomputable def bounded_convex_sets (A B : set ℝ) [convex ℝ A] [convex ℝ B] :=
  ∃ K1 K2 : ℝ, ∀ x ∈ A, ∀ y ∈ B, x ≤ K1 ∧ y ≤ K2 

-- Support strip width condition
variables (support_width_A : ℝ) [support_width_A = a]
variables (support_width_B : ℝ) [support_width_B = b]
variables (support_width_C : ℝ) [support_width_C = c]

-- Minkowski Sum
def minkowski_sum (A B : set ℝ) : set ℝ := {z | ∃ x ∈ A, ∃ y ∈ B, z = x + y}

-- Main theorem for problem a
theorem support_strip_width (hA: convex_set A) (hB: convex_set B) (hC: convex_set (A + B)) 
  (h_width_A: support_width_A = a) (h_width_B: support_width_B = b) : support_width_C = a + b := 
sorry

end support_strip_width_l106_106805


namespace division_value_l106_106835

theorem division_value (x : ℝ) (h : 1376 / x - 160 = 12) : x = 8 := 
by sorry

end division_value_l106_106835


namespace incorrect_variance_min_standard_deviation_l106_106041

theorem incorrect_variance (a b : ℝ) (h1 : a^2 + b^2 = 9) (h2 : (3 + a + b) / 3 > 1) : 
  (6 - (a + b + 3) / 3)^2 ≠ 2 :=
by
  sorry

theorem min_standard_deviation (a b : ℝ) (h1 : a^2 + b^2 = 9) :
  (minSd : ℝ) × (legs : ℝ × ℝ) :=
by
  let s := sqrt 2 - 1
  let l := 3 * sqrt 2 / 2
  exist (s, (l, l))
  sorry

end incorrect_variance_min_standard_deviation_l106_106041


namespace area_of_triangle_PQR_l106_106657

open Real

-- Definition of the problem
structure Triangle (α : Type) :=
(P Q R S : α)
(PS SR : ℝ)
(ps_pos : 0 < PS)
(sr_pos : 0 < SR)
(ps_value : PS = 4)
(sr_value : SR = 9)
(Q_foot_altitude : S = Q_foot PS SR)
(PQ_perpendicular_PR : perpendicular PQ PR)

-- The theorem statement
theorem area_of_triangle_PQR (α : Type) [linear_ordered_field α] [archimedean α] [has_sqrt α]:
  ∀ P Q R S : α,
  ∀ PS SR : ℝ, 
  ∀ ps_pos : 0 < PS, 
  ∀ sr_pos : 0 < SR,
  ∀ ps_value : PS = 4,
  ∀ sr_value : SR = 9,
  ∀ Q_foot_altitude : S = Q_foot PS SR,
  ∀ PQ_perpendicular_PR : perpendicular PQ PR,
  let PR := PS + SR,
  let QS := sqrt (PS * SR),
  ∃ area : ℝ, 
  area = (1/2) * PS * QS := 
sorry

end area_of_triangle_PQR_l106_106657


namespace variance_incorrect_min_standard_deviation_l106_106031

open Real

-- Define a right triangle with hypotenuse of length 3
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ c = 3

-- Prove that the variance of the side lengths cannot be 2
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b 3) : 
  ¬(let x := [a, b, 3] in
    let mean_square := (x.map (λ x, x^2)).sum / 3 in
    let mean := x.sum / 3 in
    mean_square - mean^2 = 2) :=
sorry

-- Prove the minimum standard deviation and corresponding lengths
theorem min_standard_deviation {a b : ℝ} (h : right_triangle a b 3) :
  a = b → a = b → a = 3 * real.sqrt(2) / 2 → b = 3 * real.sqrt(2) / 2 →
  (let variance := (h.1.map (λ x, x^2)).sum / 2 - ((h.1.sum / 2)^2) in
  let std_dev_min := real.sqrt(variance) in
  std_dev_min = real.sqrt(2) - 1) :=
sorry

end variance_incorrect_min_standard_deviation_l106_106031


namespace binom_18_10_l106_106448

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106448


namespace max_point_ratio_polar_to_cartesian_equations_l106_106213

theorem max_point_ratio (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  let x := 2 + t
      y := 2 - t
      ρ := 2 * Real.sin θ
      |ON| := 2 * Real.sin α
      |OM| := 4 / (Real.sin α + Real.cos α)
  in Real.max ((|ON| / |OM|) : ℝ) = (Real.sqrt 2 + 1) / 4 :=
begin
  sorry
end

theorem polar_to_cartesian_equations
  (t θ : ℝ)
  (h_t : 0 < θ ∧ θ < π/2)
  (curve_C : Π(ρ : ℝ), ρ = 2 * Real.sin θ)
  (line_l : (x = 2 + t) ∧ (y = 2 - t)) :
  (∃ θ, curve_C θ ∧ ((x ^ 2 + (y - 1) ^ 2 = 1) ∧
                      (ρ = 4 / (Real.sin θ + Real.cos θ)))) :=
begin
  sorry
end

end max_point_ratio_polar_to_cartesian_equations_l106_106213


namespace binom_18_10_l106_106489

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106489


namespace range_of_a_l106_106298

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici 3) := 
sorry

end range_of_a_l106_106298


namespace intersection_with_x_axis_intersection_with_y_axis_l106_106218

theorem intersection_with_x_axis (x y : ℝ) : y = -2 * x + 4 ∧ y = 0 ↔ x = 2 ∧ y = 0 := by
  sorry

theorem intersection_with_y_axis (x y : ℝ) : y = -2 * x + 4 ∧ x = 0 ↔ x = 0 ∧ y = 4 := by
  sorry

end intersection_with_x_axis_intersection_with_y_axis_l106_106218


namespace equation_of_line_AD_equation_of_line_through_D_parallel_to_AC_l106_106619

variables (A B C D : ℝ × ℝ)
variables (AD_line BC_line AC_line D_parallel : ℝ → ℝ)

noncomputable def A := (2, 2: ℝ)
noncomputable def B := (-4, 0: ℝ)
noncomputable def C := (3, -1: ℝ)

-- Definition of perpendicular line
def perpendicular_through (P Q R : ℝ × ℝ) : Prop :=
  let k_QR := (R.2 - Q.2) / (R.1 - Q.1) in
  let k_PQ := (Q.2 - P.2) / (Q.1 - P.1) in
  k_PQ * k_QR = -1

-- Definition of slope form line equation y = mx + b
def line_eq (m b: ℝ) (x y: ℝ) : Prop := y = m * x + b

-- Finding D
def point_D (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let m1 := 7 in -- slope of AD
  let m2 := -1 / 7 in -- slope of BC
  let b1 := A.2 - m1 * A.1 in
  let b2 := B.2 - m2 * B.1 in
  let x := (b2 - b1) / (m1 - m2) in
  let y := m1 * x + b1 in
  ⟨x, y⟩

-- Line through D parallel to AC
def parallel_to_AC_through_D := -3

-- Theorem statement for AD equation
theorem equation_of_line_AD : ∃ (m b : ℝ), (line_eq m b (2:ℝ) (2:ℝ)) ∧ m = 7 ∧ b = -12 :=
by 
  use [7, -12]
  split; try { simp [line_eq] } 
  sorry

-- Theorem statement for line through D parallel to AC
theorem equation_of_line_through_D_parallel_to_AC : ∃ (m b : ℝ), 
  let D := point_D A B C in (line_eq m b D.1 D.2) ∧ m = -3 ∧ b = 4 :=
by 
  use [-3, 4]
  have D := point_D A B C
  split; try { simp [line_eq] } 
  sorry

end equation_of_line_AD_equation_of_line_through_D_parallel_to_AC_l106_106619


namespace T_perimeter_visible_is_22_l106_106758

theorem T_perimeter_visible_is_22 :
  let V := (2, 6 : ℕ)
  let H := (3, 2 : ℕ)
  let perimeter (w h : ℕ) := 2 * w + 2 * h
  let visible_vertical_perimeter := perimeter (V.1) (V.2) - 4
  let visible_horizontal_perimeter := perimeter (H.1) (H.2)
  visible_vertical_perimeter + visible_horizontal_perimeter = 22 := by
  sorry

end T_perimeter_visible_is_22_l106_106758


namespace contact_probability_l106_106541

variable (m : ℕ := 6) (n : ℕ := 7) (p : ℝ)

theorem contact_probability :
  let total_pairs := m * n in
  let probability_no_contact := (1 - p) ^ total_pairs in
  let probability_contact := 1 - probability_no_contact in
  probability_contact = 1 - (1 - p) ^ 42 :=
by
  -- This is where the proof would go
  sorry

end contact_probability_l106_106541


namespace arithmetic_series_sum_l106_106550

theorem arithmetic_series_sum : 
  let a := -48
  let d := 2
  let l := 0
  let n := 25
  S = (n * (a + l)) / 2 
  (S = -600) :=
sorry

end arithmetic_series_sum_l106_106550


namespace T_n_bounds_l106_106389

noncomputable def a_n (n : ℕ) : ℕ :=
  2^(n - 1)

noncomputable def b_n (n : ℕ) : ℕ :=
  2 * n - 1

noncomputable def c_n (n : ℕ) : ℚ :=
  1 / (b_n n * b_n (n + 1))

noncomputable def T_n (n : ℕ) : ℚ :=
  (List.range n).sum (λ i, c_n i)

theorem T_n_bounds (n : ℕ) : 
  1 / 3 ≤ T_n n ∧ T_n n < 1 / 2 :=
by
  sorry

end T_n_bounds_l106_106389


namespace binom_18_10_l106_106438

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106438


namespace binomial_coeff_18_10_l106_106480

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106480


namespace rectangle_length_l106_106810

theorem rectangle_length (w l : ℝ) (hP : (2 * l + 2 * w) / w = 5) (hA : l * w = 150) : l = 15 :=
by
  sorry

end rectangle_length_l106_106810


namespace km_per_gallon_proof_l106_106063

-- Define the given conditions
def distance := 100
def gallons := 10

-- Define what we need to prove the correct answer
def kilometers_per_gallon := distance / gallons

-- Prove that the calculated kilometers per gallon is equal to 10
theorem km_per_gallon_proof : kilometers_per_gallon = 10 := by
  sorry

end km_per_gallon_proof_l106_106063


namespace num_solutions_cot_tan_l106_106903

noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x

theorem num_solutions_cot_tan (θ : ℝ) (h_θ : 0 < θ ∧ θ < 2 * Real.pi) :
  ∃ N : ℕ, N = 22 ∧ 
    ∀ θ : ℝ, (cot (4 * Real.pi * Real.cos θ) = Real.tan (4 * Real.pi * Real.sin θ)) 
      ↔ 0 < θ ∧ θ < 2 * Real.pi :=
sorry

end num_solutions_cot_tan_l106_106903


namespace car_speed_is_correct_l106_106414

theorem car_speed_is_correct :
  (∀ (distance_per_gallon total_tank tank_fraction hours : ℝ),
    distance_per_gallon = 30 ∧
    total_tank = 15 ∧
    tank_fraction = 0.5555555555555556 ∧
    hours = 5 →
    let gallons_used := tank_fraction * total_tank in
    let distance_traveled := gallons_used * distance_per_gallon in
    let speed := distance_traveled / hours in
    speed = 50) :=
sorry

end car_speed_is_correct_l106_106414


namespace log_term_evaluation_l106_106869

theorem log_term_evaluation : (Real.log 2)^2 + (Real.log 5)^2 + 2 * (Real.log 2) * (Real.log 5) = 1 := by
  sorry

end log_term_evaluation_l106_106869


namespace barley_percentage_is_80_l106_106708

variables (T C : ℝ) -- Total land and cleared land
variables (B : ℝ) -- Percentage of cleared land planted with barley

-- Given conditions
def cleared_land (T : ℝ) : ℝ := 0.9 * T
def total_land_approx : ℝ := 1000
def potato_land (C : ℝ) : ℝ := 0.1 * C
def tomato_land : ℝ := 90
def barley_percentage (C : ℝ) (B : ℝ) : Prop := C - (potato_land C) - tomato_land = (B / 100) * C

-- Theorem statement to prove
theorem barley_percentage_is_80 :
  cleared_land total_land_approx = 900 → barley_percentage 900 80 :=
by
  intros hC
  rw [cleared_land, total_land_approx] at hC
  simp [barley_percentage, potato_land, tomato_land]
  sorry

end barley_percentage_is_80_l106_106708


namespace cryptarithm_solution_l106_106277

-- Definitions for the digits
def K : ℕ := 5
def O : ℕ := 0
def Ш : ℕ := 3
def A : ℕ := 0
def С : ℕ := 1
def Б : ℕ := 0

-- Representing the words as numbers
def KOSHKA := K * 10000 + O * 1000 + Ш * 100 + K * 10 + A
def СОБАКА := С * 100000 + O * 10000 + Б * 1000 + A * 100 + K * 10 + A

theorem cryptarithm_solution : 3 * KOSHKA = СОБАКА :=
by
  -- We put 'sorry' here to skip the proof
  sorry

end cryptarithm_solution_l106_106277


namespace convert_spherical_to_rectangular_l106_106883

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular (
  (ρ θ φ : ℝ) (hρ : ρ = 3) (hθ : θ = 3*Real.pi/2) (hφ : φ = Real.pi/3) :
  spherical_to_rectangular ρ θ φ = (0, -3*Real.sqrt 3 / 2, 3 / 2) :=
by
  rw [hρ, hθ, hφ]
  simp [spherical_to_rectangular]
  sorry

end convert_spherical_to_rectangular_l106_106883


namespace binomial_coeff_18_10_l106_106478

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106478


namespace chocolate_bars_shared_equally_l106_106367

theorem chocolate_bars_shared_equally (total_bars : ℕ) (persons : ℕ) (mike_rita_anita : persons = 3) (bars : total_bars = 12) : 
  (total_bars / persons) * 2 = 8 := 
by 
  have h1 : total_bars = 12 := bars
  have h2 : persons = 3 := mike_rita_anita
  rw [h1, h2] -- substitute values of total_bars and persons
  norm_num -- simplify the arithmetic expression
  sorry

end chocolate_bars_shared_equally_l106_106367


namespace latest_time_to_reach_80_degrees_l106_106638

theorem latest_time_to_reach_80_degrees :
  ∀ (t : ℝ), (-t^2 + 14 * t + 40 = 80) → t ≤ 10 :=
by
  sorry

end latest_time_to_reach_80_degrees_l106_106638


namespace f_identity_l106_106187

noncomputable def f (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

theorem f_identity (x : ℝ) (h : -1 < x ∧ x < 1) :
  f ( (3 * x + x^3) / (1 + 3 * x^2) ) = 3 * f x :=
sorry

end f_identity_l106_106187


namespace binom_18_10_l106_106459

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106459


namespace fraction_equals_repeating_decimal_l106_106527

noncomputable def repeating_decimal_fraction : ℚ :=
  let a : ℚ := 46 / 100
  let r : ℚ := 1 / 100
  (a / (1 - r))

theorem fraction_equals_repeating_decimal :
  repeating_decimal_fraction = 46 / 99 :=
by
  sorry

end fraction_equals_repeating_decimal_l106_106527


namespace number_of_strikers_l106_106390

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end number_of_strikers_l106_106390


namespace proof_problem_l106_106593

noncomputable def P1 : Prop :=
∀ (a : ℝ), ∀ (x : ℝ), (x = 4) → (2 = log a (x - 3) + 2)

def P4 : Prop :=
∀ (x : ℝ), ln x = has_inv.inv (exp x)

theorem proof_problem : P1 ∧ P4 := by
  exact sorry

end proof_problem_l106_106593


namespace find_a3_l106_106648

noncomputable theory
open_locale classical

structure ArithSeq (α : Type*) :=
(first_term : α)
(common_diff : α)

def nth_term {α : Type*} [add_group α] (seq : ArithSeq α) (n : ℕ) : α :=
seq.first_term + (n - 1) • seq.common_diff

theorem find_a3 (a_5p6 a_8 : ℝ) (h1 : a_5p6 = 16) (h2 : a_8 = 12) :
  ∃ a1 d, a1 + 2 * d = 4 :=
begin
  let a1 : ℝ := 4 / 5,
  let d : ℝ := 8 / 5,
  use [a1, d],
  rw h1,
  rw h2,
  sorry
end

end find_a3_l106_106648


namespace field_trip_bread_pieces_l106_106846

theorem field_trip_bread_pieces :
  (students_per_group : ℕ) (num_groups : ℕ) (sandwiches_per_student : ℕ) (pieces_per_sandwich : ℕ)
  (H1 : students_per_group = 6)
  (H2 : num_groups = 5)
  (H3 : sandwiches_per_student = 2)
  (H4 : pieces_per_sandwich = 2)
  : 
  let total_students := num_groups * students_per_group in
  let total_sandwiches := total_students * sandwiches_per_student in
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich in
  total_pieces_bread = 120 :=
by
  let total_students := num_groups * students_per_group
  let total_sandwiches := total_students * sandwiches_per_student
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich
  sorry

end field_trip_bread_pieces_l106_106846


namespace sum_vectors_spheres_cubes_l106_106135

-- Lean definition for the problem setup

noncomputable def problem_given_spheres_cubes (G1 G2 : Type) (K1 K2 : Type) 
  (O1 O2 : G1) (A : Finₓ 8 → K1) (B : Finₓ 8 → K2) 
  (vector : G1 → G2 → G2) : Prop :=
(∑ i:Finₓ 8, ∑ j:Finₓ 8, vector (A i) (B j)) = (64 : ℕ) • vector O1 O2

-- Here, G1 and G2 represent the types of the spheres' points.
-- K1 and K2 represent the types of the vertices of the inscribed cubes.
-- O1 and O2 are the centers of the first and second sphere, respectively.
-- A and B are functions mapping an index to a vertex of K1 and K2.
-- vector is a function that takes two points and gives a vector between them.
-- ∑ is the summation over the vertices.

theorem sum_vectors_spheres_cubes {G1 G2 K1 K2 : Type} (O1 O2 : G1) (A : Finₓ 8 → K1) (B : Finₓ 8 → K2)
  (vector : G1 → G2 → G2) :
  problem_given_spheres_cubes G1 G2 K1 K2 O1 O2 A B vector :=
sorry

end sum_vectors_spheres_cubes_l106_106135


namespace claire_sleep_hours_l106_106874

def hours_in_day := 24
def cleaning_hours := 4
def cooking_hours := 2
def crafting_hours := 5
def tailoring_hours := crafting_hours

theorem claire_sleep_hours :
  hours_in_day - (cleaning_hours + cooking_hours + crafting_hours + tailoring_hours) = 8 := by
  sorry

end claire_sleep_hours_l106_106874


namespace train_pass_bridge_time_l106_106851

theorem train_pass_bridge_time
  (train_length : ℕ)
  (train_speed_kmph : ℕ)
  (time_seconds : ℕ)
  (bridge_length : ℕ) :
  train_length = 360 →
  train_speed_kmph = 45 →
  time_seconds = 40 →
  bridge_length = 140 →
  (train_speed_kmph * 1000 / 3600) * time_seconds = train_length + bridge_length :=
by
  { intros train_length_is train_speed_is time_is bridge_length_is,
    sorry }

end train_pass_bridge_time_l106_106851


namespace percentage_problem_l106_106332

variable (x : ℝ)
variable (y : ℝ)

theorem percentage_problem : 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 → x = 33.52 := by
  sorry

end percentage_problem_l106_106332


namespace repaint_to_single_color_l106_106310

def num_glass_pieces : ℕ := 1987

def color := ℕ → ℕ

def red (c : ℕ) := c % 3 = 0
def yellow (c : ℕ) := c % 3 = 1
def blue (c : ℕ) := c % 3 = 2

noncomputable def operation (c1 c2 : color) : Prop :=
  (c1 ≠ c2) → (∃ c3 : color, c3 ≠ c1 ∧ c3 ≠ c2)

theorem repaint_to_single_color (initial_colors : fin num_glass_pieces → ℕ) :
  ∃ final_color : color, (∀ i, final_color i = final_color 0) ∧ yellow final_color 0 :=
sorry

end repaint_to_single_color_l106_106310


namespace triangle_HD_ratio_HA_l106_106646

-- Definitions of the sides of the triangle
def side_a := 11
def side_b := 13
def side_c := 16

-- Definition that the point H is the orthocenter of the triangle
-- A: the vertex from which the altitude AD is drawn
-- D: foot of the altitude from A to the side BC of length 13
-- H: orthocenter, the intersection of the altitudes
def HD_ratio_HA (H A D : Type) : Prop := 
  ∀ (triangle : Triangle side_a side_b side_c) (H : Point) (A : Point) (D : Point),
    is_orthocenter H triangle →
    is_foot H A D triangle →
    ratio (distance H D) (distance H A) = 0

theorem triangle_HD_ratio_HA : ∃ (H A D : Type), HD_ratio_HA H A D := by
  sorry

end triangle_HD_ratio_HA_l106_106646


namespace cheryl_probability_correct_l106_106821

noncomputable def probability_cheryl_same_color : ℝ :=
  let total_ways := (Finset.card (Finset.powersetLen 2 (Finset.range 6))) *
                    (Finset.card (Finset.powersetLen 2 (Finset.range 4))) *
                    (Finset.card (Finset.powersetLen 2 (Finset.range 2)))
  let favorable_ways := 3 * (Finset.card (Finset.powersetLen 2 (Finset.range 4))) * (Finset.card (Finset.powersetLen 2 (Finset.range 2)))
  in favorable_ways / total_ways

theorem cheryl_probability_correct :
  probability_cheryl_same_color = 1/5 :=
by
  sorry

end cheryl_probability_correct_l106_106821


namespace DC_correct_l106_106215

noncomputable def solve_for_DC 
  (AB : ℝ) (ADB_right_angle : ∀ α β : ℝ, α = 0 → β = π / 2) 
  (cos_A : ℝ) (sin_C : ℝ) 
  (h_AB : AB = 30) 
  (h_cos_A : cos_A = 4 / 5) 
  (h_sin_C : sin_C = 1 / 5) 
  : ℝ := 
  let AD := (cos_A * AB) in
  let BD := real.sqrt (AB^2 - AD^2) in
  let BC := BD / sin_C in
  real.sqrt (BC^2 - BD^2)

theorem DC_correct 
  (AB : ℝ) (ADB_right_angle : ∀ α β : ℝ, α = 0 → β = π / 2) 
  (cos_A : ℝ) (sin_C : ℝ) 
  (h_AB : AB = 30) 
  (h_cos_A : cos_A = 4 / 5) 
  (h_sin_C : sin_C = 1 / 5) 
  : solve_for_DC AB ADB_right_angle cos_A sin_C h_AB h_cos_A h_sin_C = 88 * real.sqrt 2 := 
by 
  unfold solve_for_DC
  sorry

end DC_correct_l106_106215


namespace num_ways_distribute_balls_l106_106986

-- Definition of the combinatorial problem
def indistinguishableBallsIntoBoxes : ℕ := 11

-- Main theorem statement
theorem num_ways_distribute_balls : indistinguishableBallsIntoBoxes = 11 := by
  sorry

end num_ways_distribute_balls_l106_106986


namespace bread_needed_for_sandwiches_l106_106850

def students_per_group := 5
def groups := 5
def sandwiches_per_student := 2
def pieces_of_bread_per_sandwich := 2

theorem bread_needed_for_sandwiches : 
  students_per_group * groups * sandwiches_per_student * pieces_of_bread_per_sandwich = 100 := 
by
  sorry

end bread_needed_for_sandwiches_l106_106850


namespace minimum_even_integers_proof_l106_106327

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def minimum_even_integers (x y a b m n : ℤ) : ℕ :=
  (if is_even x then 1 else 0)
  + (if is_even y then 1 else 0)
  + (if is_even a then 1 else 0)
  + (if is_even b then 1 else 0)
  + (if is_even m then 1 else 0)
  + (if is_even n then 1 else 0)

theorem minimum_even_integers_proof :
  ∀ (x y a b m n : ℤ), 
    x + y = 26 → 
    x + y + a + b = 41 →
    x + y + a + b + m + n = 57 →
    minimum_even_integers x y a b m n = 1 :=
by
  intros x y a b m n h1 h2 h3
  sorry

end minimum_even_integers_proof_l106_106327


namespace remaining_average_l106_106282

-- Definitions
def original_average (n : ℕ) (avg : ℝ) := n = 50 ∧ avg = 38
def discarded_numbers (a b : ℝ) := a = 45 ∧ b = 55

-- Proof Statement
theorem remaining_average (n : ℕ) (avg : ℝ) (a b : ℝ) (s : ℝ) :
  original_average n avg →
  discarded_numbers a b →
  s = (n * avg - (a + b)) / (n - 2) →
  s = 37.5 :=
by
  intros h_avg h_discard h_s
  sorry

end remaining_average_l106_106282


namespace sum_of_integer_solutions_l106_106772

theorem sum_of_integer_solutions : 
  let S := {x : ℤ | 5 * x + 2 > 3 * (x - 1) ∧ 1/2 * x - 1 ≤ 7 - 3/2 * x} in
  ∑ x in S, x = 7 :=
by {
  sorry
}

end sum_of_integer_solutions_l106_106772


namespace problem_inequality_l106_106918

theorem problem_inequality (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ ab + 3*b + 2*c := 
by 
  sorry

end problem_inequality_l106_106918


namespace trigonometric_identity_75_l106_106907

noncomputable def cos_sq (x : ℝ) : ℝ := real.cos x ^ 2
noncomputable def sin_sq (x : ℝ) : ℝ := real.sin x ^ 2

theorem trigonometric_identity_75 :
  (cos_sq (75 * real.pi / 180) ^ 2 - sin_sq (75 * real.pi / 180) ^ 2) = -real.sqrt 3 / 2 :=
by 
  -- Proof is omitted.
  sorry

end trigonometric_identity_75_l106_106907


namespace pipes_fill_tank_in_7_minutes_l106_106354

theorem pipes_fill_tank_in_7_minutes (T : ℕ) (R_A R_B R_combined : ℚ) 
  (h1 : R_A = 1 / 56) 
  (h2 : R_B = 7 * R_A)
  (h3 : R_combined = R_A + R_B)
  (h4 : T = 1 / R_combined) : 
  T = 7 := by 
  sorry

end pipes_fill_tank_in_7_minutes_l106_106354


namespace tan_3825_l106_106876

noncomputable def tan_3825_eq_1 : Prop :=
  let P := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  ∈ 225.
  1. tan(3825 * 360) => P

@[simp]
theorem tan_3825 : Real.tan 3825 = 1 := 
let P := (-Real.sqrt 2 / 2, -Real.sqrt 2 /2) in
 calc
   Real.tan 3825 = Real.tan (3825 - 10 * 360: = _ := Result
<Real.root 2/ 2, - Real.root 2/ 2)

end tan_3825_l106_106876


namespace train_crossing_bridge_time_l106_106628

/-- A train 100 meters long takes to cross a bridge 150 meters long if the speed of the train is reduced
by 3% due to wind resistance. The original speed of the train is 53.7 km/h. Prove that the time taken
to cross the bridge is approximately 17.28 seconds. -/
theorem train_crossing_bridge_time:
  let L_train := 100 -- Length of the train in meters
  let L_bridge := 150 -- Length of the bridge in meters
  let V_train_kmph := 53.7 -- Speed of the train in km/h
  let decrease_factor := 0.03 -- Speed decrease factor due to wind resistance
  let V_train_decreased_kmph := V_train_kmph * (1 - decrease_factor) -- Decreased speed in km/h
  let V_train_mps := V_train_decreased_kmph * 1000 / 3600 -- Decreased speed in m/s
  let total_distance := L_train + L_bridge -- Total distance to be traveled in meters
  let time := total_distance / V_train_mps -- Time in seconds
  abs (time - 17.28) < 0.1 := 
by
  unfold V_train_kmph decrease_factor V_train_decreased_kmph V_train_mps total_distance time
  norm_num
  sorry

end train_crossing_bridge_time_l106_106628


namespace mirror_breaks_into_60_pieces_l106_106412

theorem mirror_breaks_into_60_pieces
  (P : ℕ)
  (h1 : P > 0)
  (h2 : ∀ n, n = P - (P / 2))
  (h3 : ∀ k, k = (P / 2) - 3)
  (h4 : ∀ m, m = (2 * ((P / 2) - 3)) / 3)
  (h5 : h4 m = 9) :
  P = 60 :=
sorry

end mirror_breaks_into_60_pieces_l106_106412


namespace percentage_increase_in_expenses_l106_106384

-- Define the variables and conditions
def monthly_salary : ℝ := 7272.727272727273
def original_savings_percentage : ℝ := 0.10
def new_savings : ℝ := 400
def original_savings : ℝ := original_savings_percentage * monthly_salary
def savings_difference : ℝ := original_savings - new_savings
def original_expenses : ℝ := (1 - original_savings_percentage) * monthly_salary

-- Formalize the question as a theorem
theorem percentage_increase_in_expenses (P : ℝ) :
  P = (savings_difference / original_expenses) * 100 ↔ P = 5 := 
sorry

end percentage_increase_in_expenses_l106_106384


namespace single_elimination_games_l106_106209

theorem single_elimination_games (n : Nat) (h : n = 256) : n - 1 = 255 := by
  sorry

end single_elimination_games_l106_106209


namespace min_value_of_function_l106_106902

noncomputable def min_function_value : ℝ :=
  infi (λ x : ℝ, infi (λ y : ℝ, if (0.4 ≤ x ∧ x ≤ 0.6) ∧ (0.3 ≤ y ∧ y ≤ 0.4) then (x * y) / (x ^ 2 + y ^ 2) else 1))

theorem min_value_of_function : min_function_value = 0.5 := sorry

end min_value_of_function_l106_106902


namespace obtuse_angle_probability_l106_106267

noncomputable def probability_obtuse_angle : ℝ :=
  let F : ℝ × ℝ := (0, 3)
  let G : ℝ × ℝ := (5, 0)
  let H : ℝ × ℝ := (2 * Real.pi + 2, 0)
  let I : ℝ × ℝ := (2 * Real.pi + 2, 3)
  let rectangle_area : ℝ := (2 * Real.pi + 2) * 3
  let semicircle_radius : ℝ := Real.sqrt (2.5^2 + 1.5^2)
  let semicircle_area : ℝ := (1 / 2) * Real.pi * semicircle_radius^2
  semicircle_area / rectangle_area

theorem obtuse_angle_probability :
  probability_obtuse_angle = 17 / (24 + 4 * Real.pi) :=
by
  sorry

end obtuse_angle_probability_l106_106267


namespace line_through_point_intersecting_circle_with_chord_length_l106_106290

theorem line_through_point_intersecting_circle_with_chord_length (P : ℝ × ℝ) (r : ℝ) (d : ℝ) :
  P = (4, 8) →
  r = 5 →
  d = 6 →
  (∃ m b, m * 4 + b = 8 ∧ m^2 + b^2 = 25 ∧ (abs (-4 * m + 8) / sqrt (m^2 + 1)) = 4 ∧
  ( ∃ x y, (3 * x - 4 * y + 20 = 0) ∧ (x^2 + y^2 = 25) )
  ∨ (x = 4 ∧ y ≠ 8 ∧ x / sqrt(1) = 4)) :=
by
  sorry

end line_through_point_intersecting_circle_with_chord_length_l106_106290


namespace simplify_and_rationalize_l106_106730

-- Conditions
def sqrt6 := Real.sqrt (6)
def sqrt5 := Real.sqrt (5)
def sqrt8 := Real.sqrt (8)
def sqrt9 := Real.sqrt (9)
def sqrt10 := Real.sqrt (10)
def sqrt11 := Real.sqrt (11)
def expr := (sqrt6 / sqrt5) * (sqrt8 / sqrt9) * (sqrt10 / sqrt11)
def result := 4 * Real.sqrt 66 / 33

-- Theorem
theorem simplify_and_rationalize :
  expr = result :=
by
  sorry

end simplify_and_rationalize_l106_106730


namespace product_of_z_and_conj_l106_106580

def z : ℂ := complex.I * (1 - complex.I)
def z_conj : ℂ := conj z

theorem product_of_z_and_conj :
  z * z_conj = 2 :=
sorry

end product_of_z_and_conj_l106_106580


namespace min_value_exists_l106_106584

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9 ∧ y ≥ 2

theorem min_value_exists : ∃ x y : ℝ, point_on_circle x y ∧ x + Real.sqrt 3 * y = 2 * Real.sqrt 3 - 2 := 
sorry

end min_value_exists_l106_106584


namespace smallest_positive_period_maximum_value_triangle_area_l106_106157

def f (x : ℝ) := cos (2 * x) + 2 * sin x * sin x

theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem maximum_value : 
  ∃ M, ∀ x, f x ≤ M ∧ (∃ k : ℤ, f (k * π) = M ∧ M = 2) := sorry

theorem triangle_area (A : ℝ) (b a : ℝ) (ha : 0 < A ∧ A < π/2) (hb : b = 5) (ha' : a = 7)
  (hf : f A = 0) : ∃ c : ℝ, ∃ S : ℝ, c = 8 ∧ S = 1/2 * b * c * sin A ∧ S = 10 := sorry

end smallest_positive_period_maximum_value_triangle_area_l106_106157


namespace triangle_area_rational_l106_106320

theorem triangle_area_rational
  (x1 y1 x2 y2 x3 y3 : ℤ)
  (h : y1 = y2) :
  ∃ (k : ℚ), 
    k = abs ((x2 - x1) * y3) / 2 := sorry

end triangle_area_rational_l106_106320


namespace number_of_mappings_l106_106114

def P : set ℕ := {0, 1}
def Q : set ℤ := {-1, 0, 1}
def f : (P → Q) := sorry

theorem number_of_mappings : 
  ∃ (f : P → Q), (∃ g h, g ∈ P ∧ h ∈ P ∧ g < h ∧ f g > f h) ∧ (finset.card {f | ∃ g h, g ∈ P ∧ h ∈ P ∧ g < h ∧ f g > f h }) = 3 := sorry

end number_of_mappings_l106_106114


namespace sum_q_t_at_12_l106_106676

open Finset Polynomial

noncomputable def T : Finset (Fin 12 → ℝ) := univ.image (λ (b : Fin 12 → Bool), λ i, if b i then 1 else 0)

noncomputable def q_t (t : Fin 12 → ℝ) : Polynomial ℝ :=
  interpolate (Finset.univ.image (λ i : Fin 12, (i : ℝ))).toFinset (λ i, t i)

noncomputable def q : Polynomial ℝ :=
  ∑ t in T, q_t t

theorem sum_q_t_at_12 : q.eval 12 = 2048 := by
  sorry

end sum_q_t_at_12_l106_106676


namespace campers_difference_l106_106814

theorem campers_difference 
  (morning : ℕ)
  (afternoon : ℕ)
  (h_morning : morning = 44) 
  (h_afternoon : afternoon = 39) : 
  morning - afternoon = 5 := 
by 
  -- Given definitions
  rw [h_morning, h_afternoon]
  -- Perform subtraction
  simp
  done

end campers_difference_l106_106814


namespace chocolate_bar_cost_l106_106088

def total_bars := 11
def bars_left := 7
def bars_sold := total_bars - bars_left
def total_money := 16
def cost := total_money / bars_sold

theorem chocolate_bar_cost : cost = 4 :=
by
  sorry

end chocolate_bar_cost_l106_106088


namespace calculate_expression_l106_106255

theorem calculate_expression
  (x y : ℚ)
  (D E : ℚ × ℚ)
  (hx : x = (D.1 + E.1) / 2)
  (hy : y = (D.2 + E.2) / 2)
  (hD : D = (15, -3))
  (hE : E = (-4, 12)) :
  3 * x - 5 * y = -6 :=
by
  subst hD
  subst hE
  subst hx
  subst hy
  sorry

end calculate_expression_l106_106255


namespace variance_not_2_minimum_standard_deviation_l106_106029

-- Definition for a right triangle with hypotenuse 3
structure RightTriangle where
  (a : ℝ) (b : ℝ)
  hypotenuse : ℝ := 3
  pythagorean_property : a^2 + b^2 = 9

-- Part (a) - Prove that the variance cannot be 2
theorem variance_not_2 (triangle : RightTriangle) : 
  (6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) ≠ 2 := sorry

-- Part (b) - Prove the minimum standard deviation and corresponding leg lengths
theorem minimum_standard_deviation (triangle : RightTriangle) : 
  (exists (a b : ℝ), a > 0 ∧ b > 0 ∧ a = b ∧ a = b = 3 * (real.sqrt 2) / 2 ∧ 
  real.sqrt ((6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) = real.sqrt (2) - 1)) := sorry

end variance_not_2_minimum_standard_deviation_l106_106029


namespace discriminant_negative_of_positive_parabola_l106_106636

variable (a b c : ℝ)

theorem discriminant_negative_of_positive_parabola (h1 : ∀ x : ℝ, a * x^2 + b * x + c > 0) (h2 : a > 0) : b^2 - 4*a*c < 0 := 
sorry

end discriminant_negative_of_positive_parabola_l106_106636


namespace projection_of_e1_onto_e2_is_neg_one_l106_106949

noncomputable def e1 : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def e2 : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the conditions
axiom e1_unit : ∥e1∥ = 1
axiom e2_unit : ∥e2∥ = 1
axiom condition : ∥2 • e1 + e2∥ = ∥e1∥

-- Define the theorem to prove
theorem projection_of_e1_onto_e2_is_neg_one :
  (e1 ⬝ e2) = -1 := 
sorry

end projection_of_e1_onto_e2_is_neg_one_l106_106949


namespace sum_of_first_8_terms_log_geom_seq_l106_106651

theorem sum_of_first_8_terms_log_geom_seq (a : ℕ → ℝ) 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h4 : a 4 = 2)
  (h5 : a 5 = 5) :
  (log 10 (a 1) + log 10 (a 2) + log 10 (a 3) + log 10 (a 4) + 
   log 10 (a 5) + log 10 (a 6) + log 10 (a 7) + log 10 (a 8) = 4) :=
sorry

end sum_of_first_8_terms_log_geom_seq_l106_106651


namespace energy_moved_charge_to_center_l106_106108

open Real

-- Definitions for the initial setup
def initial_potential_energy (d : ℝ) (Q : ℝ) : ℝ :=
  let x := 40 / (4 + Real.sqrt 2)
  in 4 * x + 2 * x * (1 / Real.sqrt 2)

-- Definition for the new potential energy when one charge is moved to center
def new_potential_energy (d : ℝ) (Q : ℝ) : ℝ :=
  let x := 40 / (4 + Real.sqrt 2)
  let x_center := Real.sqrt 2 * x
  in 4 * x_center + 3 * x

-- Proof theorem statement
theorem energy_moved_charge_to_center (d Q : ℝ) :
  initial_potential_energy d Q = 40 →
  new_potential_energy d Q = (320 + 200 * Real.sqrt 2) / 7 :=
by
  sorry

end energy_moved_charge_to_center_l106_106108


namespace variance_incorrect_min_standard_deviation_l106_106032

open Real

-- Define a right triangle with hypotenuse of length 3
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ c = 3

-- Prove that the variance of the side lengths cannot be 2
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b 3) : 
  ¬(let x := [a, b, 3] in
    let mean_square := (x.map (λ x, x^2)).sum / 3 in
    let mean := x.sum / 3 in
    mean_square - mean^2 = 2) :=
sorry

-- Prove the minimum standard deviation and corresponding lengths
theorem min_standard_deviation {a b : ℝ} (h : right_triangle a b 3) :
  a = b → a = b → a = 3 * real.sqrt(2) / 2 → b = 3 * real.sqrt(2) / 2 →
  (let variance := (h.1.map (λ x, x^2)).sum / 2 - ((h.1.sum / 2)^2) in
  let std_dev_min := real.sqrt(variance) in
  std_dev_min = real.sqrt(2) - 1) :=
sorry

end variance_incorrect_min_standard_deviation_l106_106032


namespace exists_perfect_square_in_sequence_of_f_l106_106250

noncomputable def f (n : ℕ) : ℕ :=
  ⌊(n : ℝ) + Real.sqrt n⌋₊

theorem exists_perfect_square_in_sequence_of_f (m : ℕ) (h : m = 1111) :
  ∃ k, ∃ n, f^[n] m = k * k := 
sorry

end exists_perfect_square_in_sequence_of_f_l106_106250


namespace factorize_expression_l106_106092

variable {R : Type} [CommRing R]

theorem factorize_expression (x y : R) : 
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := 
by 
  sorry

end factorize_expression_l106_106092


namespace number_of_strikers_l106_106391

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end number_of_strikers_l106_106391


namespace circle_center_radius_sum_l106_106239

theorem circle_center_radius_sum :
  ∃ (c d r : ℝ), 
    (∀ (x y : ℝ), (x^2 - 4 * y - 34 = -y^2 + 12 * x + 74) ↔ ((x - c)^2 + (y - d)^2 = r^2)) ∧
    (c = 6) ∧ 
    (d = -2) ∧ 
    (r = real.sqrt 68) ∧ 
    (c + d + r = 4 + 2 * real.sqrt 17) :=
by
  sorry

end circle_center_radius_sum_l106_106239


namespace middle_digit_base5_l106_106832

theorem middle_digit_base5 {M : ℕ} (x y z : ℕ) (hx : 0 ≤ x ∧ x < 5) (hy : 0 ≤ y ∧ y < 5) (hz : 0 ≤ z ∧ z < 5)
    (h_base5 : M = 25 * x + 5 * y + z) (h_base8 : M = 64 * z + 8 * y + x) : y = 0 :=
sorry

end middle_digit_base5_l106_106832


namespace probability_of_one_white_ball_of_two_drawn_l106_106818

theorem probability_of_one_white_ball_of_two_drawn :
  let total_balls := 5
  let white_balls := 3
  let black_balls := 2
  let drawn_balls := 2
  let total_combinations := Nat.choose total_balls drawn_balls
  let favorable_combinations := Nat.choose white_balls 1 * Nat.choose black_balls 1
  (favorable_combinations : ℚ) / total_combinations = 3 / 5 :=
begin
  -- Definitions based on conditions
  let total_balls := 5,
  let white_balls := 3,
  let black_balls := 2,
  let drawn_balls := 2,
  let total_combinations := Nat.choose total_balls drawn_balls,
  let favorable_combinations := Nat.choose white_balls 1 * Nat.choose black_balls 1,
  -- Show the calculation
  have h_total_combinations : total_combinations = 10,
  { sorry },
  have h_favorable_combinations : favorable_combinations = 6,
  { sorry },
  -- Show the probability calculation
  show (favorable_combinations : ℚ) / total_combinations = 3 / 5,
  { rw [h_total_combinations, h_favorable_combinations],
    norm_num },
end

end probability_of_one_white_ball_of_two_drawn_l106_106818


namespace fabric_area_l106_106093

theorem fabric_area (length width : ℝ) (h_length : length = 8) (h_width : width = 3) : 
  length * width = 24 := 
by
  rw [h_length, h_width]
  norm_num

end fabric_area_l106_106093


namespace max_volume_pyramid_l106_106793

theorem max_volume_pyramid 
  (AB AC : ℝ)
  (sin_BAC : ℝ)
  (angle_cond : ∀ (SA SB SC : ℝ), SA = SB ∧ SB = SC ∧ SC = SA → ∀ θ, θ ≤ 60 → true)
  (h : ℝ)
  (V : ℝ)
  (AB_eq : AB = 3)
  (AC_eq : AC = 5)
  (sin_BAC_eq : sin_BAC = 3/5)
  (height_cond : h = (5 * Real.sqrt 3) / 2)
  (volume_cond : V = (1/3) * (1/2 * 3 * 5 * (3/5)) * h) :
  V = (5 * Real.sqrt 174) / 4 := sorry

end max_volume_pyramid_l106_106793


namespace max_number_with_divisors_l106_106529

def number_of_divisors (n : ℕ) : ℕ :=
  (n+1) * (n+1) * (n+1)

theorem max_number_with_divisors (a : ℕ) (n_2 n_3 n_5 : ℕ) (r : ℕ) (k : ℕ) (p : ℕ) : 
    (a % 30 = 0) ∧ 
    ((number_of_divisors (n_2 + n_3 + n_5) * ∏ i in finset.range r, k) = 30) → 
    a = 11250 := 
by 
  sorry

end max_number_with_divisors_l106_106529


namespace locus_of_point_P_l106_106970

/-- Given three points in the coordinate plane A(0,3), B(-√3, 0), and C(√3, 0), 
    and a point P on the coordinate plane such that PA = PB + PC, 
    determine the equation of the locus of point P. -/
noncomputable def locus_equation : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | (P.1^2 + (P.2 - 1)^2 = 4) ∧ (P.2 ≤ 0)}

theorem locus_of_point_P :
  ∀ (P : ℝ × ℝ),
  (∃ A B C : ℝ × ℝ, A = (0, 3) ∧ B = (-Real.sqrt 3, 0) ∧ C = (Real.sqrt 3, 0) ∧ 
     dist P A = dist P B + dist P C) →
  P ∈ locus_equation :=
by
  intros P hp
  sorry

end locus_of_point_P_l106_106970


namespace value_of_expression_l106_106188

theorem value_of_expression (m : ℝ) (h : 2 * m ^ 2 - 3 * m - 1 = 0) : 4 * m ^ 2 - 6 * m = 2 :=
sorry

end value_of_expression_l106_106188


namespace find_c_l106_106165

noncomputable def y (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c (c : ℝ) (h : ∃ a b : ℝ, a ≠ b ∧ y a c = 0 ∧ y b c = 0) :
  c = -2 ∨ c = 2 :=
by sorry

end find_c_l106_106165


namespace orange_juice_serving_size_l106_106049

theorem orange_juice_serving_size:
  (concentrate_to_water_ratio : ℕ)
  (cans_of_concentrate : ℕ)
  (volume_per_can : ℕ)
  (servings : ℕ)
  (total_volume : ℕ)
  (serving_size : ℕ)
  : concentrate_to_water_ratio = 3 
    → cans_of_concentrate = 35
    → volume_per_can = 12
    → servings = 280
    → total_volume = (cans_of_concentrate * (concentrate_to_water_ratio + 1) * volume_per_can)
    → serving_size = total_volume / servings
    → serving_size = 6 := by
  intros h_ratio h_cans h_volume h_servings h_total_volume h_serving_size
  sorry

end orange_juice_serving_size_l106_106049


namespace solve_inequality_l106_106891

theorem solve_inequality (x : ℝ) : (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ (-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_l106_106891


namespace cycle_price_reduction_l106_106767

theorem cycle_price_reduction (original_price : ℝ) :
  let price_after_first_reduction := original_price * 0.75
  let price_after_second_reduction := price_after_first_reduction * 0.60
  (original_price - price_after_second_reduction) / original_price = 0.55 :=
by
  sorry

end cycle_price_reduction_l106_106767


namespace solution_set_of_equation_l106_106579

def int_part (x : ℝ) : ℤ := ⌊x⌋  -- integer part of x
def frac_part (x : ℝ) : ℝ := x - int_part x  -- fractional part of x

theorem solution_set_of_equation (x : ℝ) (H : int_part x * frac_part x + x = 2 * frac_part x + 6) :
  x = 14 / 3 ∨ x = 21 / 4 ∨ x = 6 :=
sorry

end solution_set_of_equation_l106_106579


namespace min_k_and_unique_weights_l106_106569

theorem min_k_and_unique_weights (n : ℕ) (h_pos : 0 < n) :
  ∃ k : ℕ, (k = Nat.find (λ j, n ≤ (3^j - 1) / 2))
  ∧ (∀ m : ℕ, (n = (3^m - 1) / 2) ↔ (m = Nat.find (λ j, n ≤ (3^j - 1) / 2))) :=
by
  sorry

end min_k_and_unique_weights_l106_106569


namespace binom_18_10_l106_106458

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106458


namespace mike_and_rita_chocolates_l106_106365

theorem mike_and_rita_chocolates :
  ( ∃ (chocolates : ℕ), chocolates = 12 )
  ∧ ( ∃ (persons : ℕ), persons = 3 )
  → ( ∃ (shared : ℕ), shared = (12 / 3) * 2 )
  → shared = 8 :=
by
  intros h1 h2
  have h3 : 12 / 3 = 4 := by sorry
  have h4 : shared = (4 * 2) := by sorry
  rw [h3] at h4
  exact h4

end mike_and_rita_chocolates_l106_106365


namespace cardinality_of_D_l106_106572

def γ (α β : List ℕ) : List ℕ := List.zipWith (fun a b => abs (a - b)) α β

def D (A : Set (List ℕ)) : Set (List ℕ) := { γ α β | α ∈ A, β ∈ A }

theorem cardinality_of_D (A : Set (List ℕ)) [Fintype A] :
  Fintype.card (D A) ≥ Fintype.card A :=
sorry

end cardinality_of_D_l106_106572


namespace f_closed_form_l106_106939

noncomputable def f : ℕ+ → ℝ
| ⟨1, _⟩   := 1
| ⟨n+1, h⟩ := 2 * f ⟨n, Nat.succ_pos n⟩ / (f ⟨n, Nat.succ_pos n⟩ + 2)

lemma f_succ (n : ℕ+) : f ⟨n.val + 1, Nat.succ_pos n.val⟩ = 2 * f n / (f n + 2) :=
by cases n; simp [f]; sorry

theorem f_closed_form (n : ℕ+) : f n = 2 / (n + 1 : ℕ) :=
by
  induction n using Nat.strong_induction_on with n ih,
  cases n,
  { simp [f], },
  { rw [f_succ ⟨n.succ, sorry⟩, ih n.succ (Nat.succ_pos n.succ), ih n (Nat.succ_pos n)],
    sorry }

end f_closed_form_l106_106939


namespace binom_18_10_l106_106454

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106454


namespace simplify_expr_l106_106632

theorem simplify_expr (a b x : ℝ) (h₁ : x = a^3 / b^3) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) := 
by 
  sorry

end simplify_expr_l106_106632


namespace contact_prob_correct_l106_106533

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end contact_prob_correct_l106_106533


namespace probability_difference_l106_106363

theorem probability_difference (red_marbles black_marbles : ℤ) (h_red : red_marbles = 1500) (h_black : black_marbles = 1500) :
  |(22485 / 44985 : ℚ) - (22500 / 44985 : ℚ)| = 15 / 44985 := 
by {
  sorry
}

end probability_difference_l106_106363


namespace frequency_is_0_4_l106_106912

def weights : List ℕ := [125, 120, 122, 105, 130, 114, 116, 95, 120, 134]

def in_range (x : ℕ) : Prop := 114.5 ≤ x ∧ x < 124.5

def count_in_range : ℕ := weights.countp in_range

def frequency_in_range : ℝ := (count_in_range : ℝ) / (weights.length : ℝ)

theorem frequency_is_0_4 : frequency_in_range = 0.4 :=
by
  -- This is where we would provide the actual proof
  sorry

end frequency_is_0_4_l106_106912


namespace equation_solution_l106_106303

theorem equation_solution (x : ℝ) : (3 : ℝ)^(x-1) = 1/9 ↔ x = -1 :=
by sorry

end equation_solution_l106_106303


namespace real_roots_of_fraction_sum_l106_106997

theorem real_roots_of_fraction_sum (n : ℕ) (a : Fin n → ℝ) (h_distinct : Function.Injective a) (h_nonzero : ∀ i, a i ≠ 0) :
  ∃ S : Fin n → ℝ, |{ x | (∃ i, x ∈ Set.Ioo (a i) (a (Fin.succ i))) ∧ (Finset.sum (Finset.univ) (λ i, a i / (a i - x)) = n)}| ≥ n - 1 :=
by
  sorry

end real_roots_of_fraction_sum_l106_106997


namespace shaded_area_l106_106060

-- Definitions coming from the conditions.
def rectangle_area (ABCD : ℝ) : Prop :=
  ABCD = 1

def midpoint (AM AD : ℝ) : Prop :=
  AM = (1 / 2) * AD

def point_N (AN BN AB : ℝ) : Prop :=
  BN = AB - AN ∧ AN = (1 / 2) * BN

-- Statement of the problem in Lean 4.
theorem shaded_area (ABCD AD AB AN BN AM : ℝ)
  (h1 : rectangle_area ABCD)
  (h2 : midpoint AM AD)
  (h3 : point_N AN BN AB) :
  let triangle_area := (1 / 2) * (1 / 3) * (1 / 2),
      triangle_ABD := (1 / 2) * ABCD in
  triangle_ABD - triangle_area = 5 / 12 :=
by sorry

end shaded_area_l106_106060


namespace perfect_square_count_between_20_and_150_l106_106626

theorem perfect_square_count_between_20_and_150 :
  let lower_bound := 20
  let upper_bound := 150
  let smallest_ps := 25
  let largest_ps := 144
  let count_squares (a b : Nat) := b - a
  count_squares 4 12 = 8 := sorry

end perfect_square_count_between_20_and_150_l106_106626


namespace rational_numbers_integer_sum_l106_106723

theorem rational_numbers_integer_sum
  (x y z : ℚ)
  (h1 : (x + y^2 + z^2) ∈ ℤ)
  (h2 : (x^2 + y + z^2) ∈ ℤ)
  (h3 : (x^2 + y^2 + z) ∈ ℤ) :
  2 * x ∈ ℤ := 
sorry

end rational_numbers_integer_sum_l106_106723


namespace min_of_f_l106_106104

-- Define the max function as per the condition
def my_max (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Define the function f(x)
def f (x : ℝ) : ℝ := my_max (|x + 1|) (|x - 2|)

-- The main theorem to prove
theorem min_of_f : ∃ x : ℝ, f x = 3 / 2 :=
by
  -- Proof to be filled in later
  sorry

end min_of_f_l106_106104


namespace parabola_vertex_coordinates_l106_106743

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), (y = -3 * (x - 1)^2 - 2) → (x, y) = (1, -2) := 
begin
  intros x y h,
  sorry
end

end parabola_vertex_coordinates_l106_106743


namespace intersection_M_N_l106_106691

def M : Set ℕ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0 }

theorem intersection_M_N :
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_l106_106691


namespace binom_18_10_l106_106468

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106468


namespace smallest_positive_period_of_f_range_of_a_l106_106100

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π) :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, f x ≤ a) → a ≥ Real.sqrt 2 :=
by
  sorry

end smallest_positive_period_of_f_range_of_a_l106_106100


namespace general_term_formula_sum_first_n_terms_l106_106603

section SequenceProblem

variable {n : ℕ} (hn : n > 0)

-- Definitions
def f (x : ℝ) : ℝ := x^2 + 2 * x

def S (n : ℕ) : ℝ := f n

def a (n : ℕ) : ℝ :=
  if n = 1 then S 1
  else S n - S (n - 1)

-- Statements to prove
theorem general_term_formula (n : ℕ) (hn : n > 0) : a n = 2 * n + 1 :=
  sorry

theorem sum_first_n_terms (n : ℕ) (hn : n > 0) : 
  (∑ k in FinRange (n + 1), (1 / (a k * (a k - 2)))) = (n / (2 * n + 1)) :=
  sorry

end SequenceProblem

end general_term_formula_sum_first_n_terms_l106_106603


namespace profit_percentage_correct_l106_106045

-- Definitions of given conditions
def selling_price : Real := 100
def cost_price : Real := 76.92

-- Statement to prove
theorem profit_percentage_correct : 
  let profit := selling_price - cost_price in
  let profit_percentage := (profit / cost_price) * 100 in
  abs (profit_percentage - 29.99) < 0.01 :=
by
  sorry

end profit_percentage_correct_l106_106045


namespace num_of_positive_divisors_l106_106522

-- Given conditions
variables {x y z : ℕ}
variables (p1 p2 p3 : ℕ) -- primes
variables (h1 : x = p1 ^ 3) (h2 : y = p2 ^ 3) (h3 : z = p3 ^ 3)
variables (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)

-- Lean statement to prove
theorem num_of_positive_divisors (hx3 : x = p1 ^ 3) (hy3 : y = p2 ^ 3) (hz3 : z = p3 ^ 3) 
    (Hdist : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
    ∃ n : ℕ, n = 10 * 13 * 7 ∧ n = (x^3 * y^4 * z^2).factors.length :=
sorry

end num_of_positive_divisors_l106_106522


namespace cylinder_height_l106_106409

noncomputable def cone_base_radius := 15  -- cm
noncomputable def cone_height := 20  -- cm
noncomputable def cylinder_base_radius := 30  -- cm
noncomputable def expected_cylinder_height := 1.6666666  -- cm

theorem cylinder_height (V_cone : ℝ) (V_cylinder : ℝ) :
  V_cone = (1 / 3) * real.pi * (cone_base_radius ^ 2) * cone_height →
  V_cylinder = V_cone →
  V_cylinder = real.pi * (cylinder_base_radius ^ 2) * expected_cylinder_height :=
by
    sorry

end cylinder_height_l106_106409


namespace ratio_copper_zinc_l106_106297

theorem ratio_copper_zinc (total_mass zinc_mass : ℕ) (h1 : total_mass = 100) (h2 : zinc_mass = 35) : 
  ∃ (copper_mass : ℕ), 
    copper_mass = total_mass - zinc_mass ∧ (copper_mass / 5, zinc_mass / 5) = (13, 7) :=
by {
  sorry
}

end ratio_copper_zinc_l106_106297


namespace amount_subtracted_l106_106368

theorem amount_subtracted (N A : ℝ) (h1 : N = 100) (h2 : 0.80 * N - A = 60) : A = 20 :=
by 
  sorry

end amount_subtracted_l106_106368


namespace sphere_contains_one_rational_point_l106_106500

def is_rational (x : ℚ) : Prop := true  -- Checks if x is a rational number

def rational_point (p : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (x y z : ℚ),
  p = (x : ℝ) • ![1, 0, 0] + (y : ℝ) • ![0, 1, 0] + (z : ℝ) • ![0, 0, 1]

def sphere_eq (x y z : ℝ) : ℝ := (x - real.sqrt 2)^2 + y^2 + z^2 - 2

theorem sphere_contains_one_rational_point :
  (∃ p : EuclideanSpace ℝ (Fin 3), rational_point p ∧ sphere_eq p 0) ∧ 
  (∀ p₁ p₂ : EuclideanSpace ℝ (Fin 3), rational_point p₁ → rational_point p₂ → sphere_eq p₁ 0 → sphere_eq p₂ 0 → p₁ = p₂) :=
begin
  sorry
end

end sphere_contains_one_rational_point_l106_106500


namespace variance_incorrect_min_standard_deviation_l106_106033

open Real

-- Define a right triangle with hypotenuse of length 3
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ c = 3

-- Prove that the variance of the side lengths cannot be 2
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b 3) : 
  ¬(let x := [a, b, 3] in
    let mean_square := (x.map (λ x, x^2)).sum / 3 in
    let mean := x.sum / 3 in
    mean_square - mean^2 = 2) :=
sorry

-- Prove the minimum standard deviation and corresponding lengths
theorem min_standard_deviation {a b : ℝ} (h : right_triangle a b 3) :
  a = b → a = b → a = 3 * real.sqrt(2) / 2 → b = 3 * real.sqrt(2) / 2 →
  (let variance := (h.1.map (λ x, x^2)).sum / 2 - ((h.1.sum / 2)^2) in
  let std_dev_min := real.sqrt(variance) in
  std_dev_min = real.sqrt(2) - 1) :=
sorry

end variance_incorrect_min_standard_deviation_l106_106033


namespace problem_gx_three_real_roots_l106_106597

theorem problem_gx_three_real_roots (ω : ℝ) (hω : ω > 0) :
  (∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.pi * 7 / 12 ∧ 2 * sin (2 * ω * x + Real.pi / 6) = sqrt 3) ↔
  (13 / 7 ≤ ω ∧ ω < 15 / 7) :=
by
  sorry

end problem_gx_three_real_roots_l106_106597


namespace train_crossing_time_l106_106808

noncomputable def time_to_cross_bridge (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_crossing_time :
  time_to_cross_bridge 100 145 65 = 13.57 :=
by
  sorry

end train_crossing_time_l106_106808


namespace problem_triangle_ABC_l106_106971

noncomputable def ellipse_locus : Prop :=
  ∃ (a b : ℝ), a = 2 ∧ b = √3 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) → y ≠ 0

noncomputable def min_distance : Prop :=
  ∃ y₀ : ℝ, 0 < y₀ ∧ y₀ ≤ √3 ∧ (abs ((2 : ℝ) * 3 / (2 * y₀) - y₀ / 6) = √3 / 3)

theorem problem_triangle_ABC :
  (∃ (B C : ℝ×ℝ), B = (-1, 0) ∧ C = (1, 0)) ∧ (forall A : ℝ × ℝ, dist B A + dist C A = 4) →
  ellipse_locus ∧ min_distance := by
  -- proof steps will be inserted here 
  sorry

end problem_triangle_ABC_l106_106971


namespace polynomials_no_common_points_l106_106515

theorem polynomials_no_common_points
  (n : ℕ)
  (P Q : Polynomial ℝ) (hP : P.degree = n) (hQ : Q.degree = n) :
  (∃ a b : ℝ, ∃ k l : ℕ, k ≤ n ∧ l ≤ n ∧ (P + Polynomial.C a * Polynomial.X ^ k) ≠ (Q + Polynomial.C b * Polynomial.X ^ l)) ↔ (n = 1 ∨ (even n ∧ 0 < n)) :=
sorry

end polynomials_no_common_points_l106_106515


namespace sphere_with_one_rational_point_exists_l106_106505

theorem sphere_with_one_rational_point_exists :
  ∃ (x y z : ℚ), (x - real.sqrt 2)^2 + y^2 + z^2 = 2 ∧ (∀ (a b c : ℚ), (a - real.sqrt 2)^2 + b^2 + c^2 = 2 → (a = 0 ∧ b = 0 ∧ c = 0)) :=
begin
  sorry,
end

end sphere_with_one_rational_point_exists_l106_106505


namespace arithmetic_sequence_nth_term_l106_106308

theorem arithmetic_sequence_nth_term (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  (a₁ = 11) →
  (d = -3) →
  (-49 = a₁ + (n - 1) * d) →
  (n = 21) :=
by 
  intros h₁ h₂ h₃
  sorry

end arithmetic_sequence_nth_term_l106_106308


namespace max_value_of_expression_l106_106243

theorem max_value_of_expression (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ 7 * Real.sqrt 2 :=
sorry

end max_value_of_expression_l106_106243


namespace point_on_line_l106_106589

theorem point_on_line (x : ℝ) : 
    (∃ k : ℝ, (-4) = k * (-4) + 8) → 
    (-4 = 2 * x + 8) → 
    x = -6 := 
sorry

end point_on_line_l106_106589


namespace bisection_method_min_calculations_l106_106780

theorem bisection_method_min_calculations :
  (∃ (n : ℕ), (1.4 < x ∧ x < 1.5) ∧ x^2 = 2 ∧ (0.1 / 2^n < 0.001)) → n ≥ 7 :=
by sorry

end bisection_method_min_calculations_l106_106780


namespace animals_total_l106_106864

-- Given definitions and conditions
def ducks : ℕ := 25
def rabbits : ℕ := 8
def chickens := 4 * ducks

-- Proof statement
theorem animals_total (h1 : chickens = 4 * ducks)
                     (h2 : ducks - 17 = rabbits)
                     (h3 : rabbits = 8) :
  chickens + ducks + rabbits = 133 := by
  sorry

end animals_total_l106_106864


namespace segment_equal_perpendicular_l106_106283

-- Given rectangle ABCD and points M, K as described.
variables (A B C D M K : Point)

-- Assume ABCD is a rectangle
axiom rectangle_ABCD : Rectangle A B C D

-- Define points M and K as the intersection points of the described angle bisectors
axiom bisector_B_M : ∃ P, P ∈ line(AD) ∧ is_bisector_angle B P M
axiom bisector_D_K : ∃ Q, Q ∈ line(AB) ∧ is_bisector_exterior_angle D Q K

-- Define properties of intersection points respecting their geometric constraints
axiom M_on_AD : M ∈ line(AD)
axiom K_on_AB : K ∈ line(AB)

-- Statement to prove: MK is equal and perpendicular to diagonal BD
theorem segment_equal_perpendicular : MK = BD ∧ is_perpendicular MK BD :=
  by sorry

end segment_equal_perpendicular_l106_106283


namespace hexagon_perimeter_l106_106756

def side_length : ℝ := 4
def number_of_sides : ℕ := 6

theorem hexagon_perimeter :
  6 * side_length = 24 := by
    sorry

end hexagon_perimeter_l106_106756


namespace problem_g_eq_l106_106749

noncomputable def g : ℝ → ℝ := sorry

theorem problem_g_eq :
  (∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x + x) →
  g 3 = ( -31 - 3 * 3^(1/3)) / 8 :=
by
  intro h
  -- proof goes here
  sorry

end problem_g_eq_l106_106749


namespace incorrect_variance_min_standard_deviation_l106_106043

theorem incorrect_variance (a b : ℝ) (h1 : a^2 + b^2 = 9) (h2 : (3 + a + b) / 3 > 1) : 
  (6 - (a + b + 3) / 3)^2 ≠ 2 :=
by
  sorry

theorem min_standard_deviation (a b : ℝ) (h1 : a^2 + b^2 = 9) :
  (minSd : ℝ) × (legs : ℝ × ℝ) :=
by
  let s := sqrt 2 - 1
  let l := 3 * sqrt 2 / 2
  exist (s, (l, l))
  sorry

end incorrect_variance_min_standard_deviation_l106_106043


namespace binom_18_10_eq_43758_l106_106467

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106467


namespace cricketer_running_percentage_l106_106376

def runs (total: ℕ) (boundaries: ℕ) (sixes: ℕ) (runs_per_boundary: ℕ) (runs_per_six: ℕ): ℕ :=
  total - (boundaries * runs_per_boundary + sixes * runs_per_six)

def percentage (value: ℕ) (total: ℕ): Float :=
  (value.toFloat / total.toFloat) * 100

theorem cricketer_running_percentage :
  let total_runs := 152
  let boundaries := 12
  let sixes := 2
  let runs_per_boundary := 4
  let runs_per_six := 6
  let runs_by_running := runs total_runs boundaries sixes runs_per_boundary runs_per_six
  let percentage_by_running := percentage runs_by_running total_runs
  percentage_by_running ≈ 60.53 :=
by
  sorry

end cricketer_running_percentage_l106_106376


namespace min_abs_sum_l106_106248

theorem min_abs_sum (p q r s : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_matrix : (⟨⟨p, q⟩, ⟨r, s⟩⟩ : Matrix (Fin 2) (Fin 2) ℤ) ^ 2 = (⟨⟨16, 0⟩, ⟨0, 16⟩⟩) :
    Matrix (Fin 2) (Fin 2) ℤ) : |p| + |q| + |r| + |s| = 10 :=
sorry

end min_abs_sum_l106_106248


namespace max_value_8a_3b_5c_l106_106245

theorem max_value_8a_3b_5c (a b c : ℝ) (h_condition : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ (Real.sqrt 373) / 6 :=
by
  sorry

end max_value_8a_3b_5c_l106_106245


namespace parabola_directrix_focus_chord_fixed_point_l106_106212

-- Parabola definition and properties
def parabola_eq (x y : ℝ) : Prop := x * x = 4 * y

-- Conditions and goals
theorem parabola_directrix_focus :
  (∀ x y : ℝ, parabola_eq x y → y = -1 ∧ (0, 1) = (0, 1)) := 
  sorry

theorem chord_fixed_point (O A B : ℝ × ℝ) (hO : O = (0, 0))
  (chord_AB : parabola_eq (O.1) (O.2) ∧ parabola_eq (A.1) (A.2) ∧ parabola_eq (B.1) (B.2))
  (dot_product : (O.1 * A.1) + (O.2 * A.2) + (O.1 * B.1) + (O.2 * B.2) = -4) :
  (∀ line_eq : ℝ → ℝ, (line_eq O.1 = 2) ∧ ∃ k b : ℝ, line_eq = k * O.1 + b) := 
  sorry

end parabola_directrix_focus_chord_fixed_point_l106_106212


namespace sum_of_zeros_of_even_function_is_zero_l106_106586

open Function

theorem sum_of_zeros_of_even_function_is_zero (f : ℝ → ℝ) (hf: Even f) (hx: ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) :
  x1 + x2 + x3 + x4 = 0 := by
  sorry

end sum_of_zeros_of_even_function_is_zero_l106_106586


namespace problem_part1_problem_part2_problem_part3_l106_106176

open Real

def vector_a (x : ℝ) : ℝ × ℝ := (cos x, sin x)
def vector_b (x : ℝ) : ℝ × ℝ := (sin x, cos x)
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2
def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
def f (x : ℝ) : ℝ := dot_product (vector_a x) (vector_b x) - sqrt 2 * vector_norm ((vector_a x).1 + (vector_b x).1, (vector_a x).2 + (vector_b x).2)

theorem problem_part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) :
  0 ≤ dot_product (vector_a x) (vector_b x) ∧ dot_product (vector_a x) (vector_b x) ≤ 1 := 
sorry

theorem problem_part2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2):
  vector_norm ((vector_a x).1 + (vector_b x).1, (vector_a x).2 + (vector_b x).2) = 2 * sin (x + π / 4) :=
sorry

theorem problem_part3 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2):
  -2 ≤ f x ∧ f x ≤ 1 - 2 * sqrt 2 :=
sorry

end problem_part1_problem_part2_problem_part3_l106_106176


namespace rectangle_division_segments_l106_106127

-- Definitions of the concepts involved
def is_parallel_to_R (rect: ℝ × ℝ) (R: ℝ × ℝ) : Prop := true

def is_node (vertex: ℝ × ℝ) (rect: ℝ × ℝ) : Prop := true

def is_basic_segment (segment: (ℝ × ℝ) × (ℝ × ℝ)) (rect: ℝ × ℝ) : Prop := 
segment.1 ≠ segment.2 

-- Main problem statement in Lean 4
theorem rectangle_division_segments (R: ℝ × ℝ) (rects: List (ℝ × ℝ))  
(h: ∀ r ∈ rects, is_parallel_to_R r R) (h_disjoint: pairwise (disjoint_on ℝ rects))
(h_cover: ∀ (p ∈ R), ∃ r ∈ rects, p ∈ r) :
  4122 ≤ (∑ r in rects, count_basic_segments r) ∧ (∑ r in rects, count_basic_segments r) ≤ 6049 := 
sorry

end rectangle_division_segments_l106_106127


namespace reflections_of_orthocenter_lie_on_circumcircle_l106_106672

theorem reflections_of_orthocenter_lie_on_circumcircle
  (ABC : Type) [triangle ABC]
  (acute_angled_triangle : is_acute_angled_triangle ABC)
  (H : point) (orthocenter : is_orthocenter H ABC) :
  ∀ side ∈ sides_of_triangle ABC, 
    let H' := reflection_of_point H side in
    lies_on_circumcircle H' ABC :=
by
  sorry

end reflections_of_orthocenter_lie_on_circumcircle_l106_106672


namespace zero_condition_l106_106962

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 + 2 * x ^ 2 - 1

theorem zero_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x : ℝ, f a x = 0 → (x = x₁ ∨ x = x₂)) ↔ 
  (a = 0 ∨ a = ± (4 * real.sqrt 6 / 9)) :=
by
  sorry

end zero_condition_l106_106962


namespace percentage_of_hexagon_area_is_equilateral_triangle_l106_106077

theorem percentage_of_hexagon_area_is_equilateral_triangle
  (s : ℝ) :
  let triangle_area := (sqrt 3 / 4) * s^2
      rectangle_area := (s * (s / 2))
      hexagon_area := triangle_area + rectangle_area
      percentage := (triangle_area / hexagon_area) * 100 in
      percentage = (sqrt 3 / (sqrt 3 + 2)) * 100 :=
begin
  sorry
end

end percentage_of_hexagon_area_is_equilateral_triangle_l106_106077


namespace angle_of_inclination_l106_106082

theorem angle_of_inclination (α : ℝ) (h: 0 ≤ α ∧ α < 180) (slope_eq : Real.tan (Real.pi * α / 180) = Real.sqrt 3) :
  α = 60 :=
sorry

end angle_of_inclination_l106_106082


namespace intersection_complement_l106_106612

open Set

variable (U A B : Set ℕ)

-- Definitions based on conditions given in the problem
def universal_set : Set ℕ := {1, 2, 3, 4, 5}
def set_A : Set ℕ := {2, 4}
def set_B : Set ℕ := {4, 5}

-- Proof statement
theorem intersection_complement :
  A = set_A → 
  B = set_B → 
  U = universal_set → 
  (A ∩ (U \ B)) = {2} := 
by
  intros hA hB hU
  sorry

end intersection_complement_l106_106612


namespace fewer_green_pens_than_pink_l106_106321

-- Define the variables
variables (G B : ℕ)

-- State the conditions
axiom condition1 : G < 12
axiom condition2 : B = G + 3
axiom condition3 : 12 + G + B = 21

-- Define the problem statement
theorem fewer_green_pens_than_pink : 12 - G = 9 :=
by
  -- Insert the proof steps here
  sorry

end fewer_green_pens_than_pink_l106_106321


namespace convex_polygon_of_100_points_l106_106717

theorem convex_polygon_of_100_points (N : ℕ) :
  (∀ (points : finset (euclidean_space ℝ (fin 2))), points.card = N ∧ (∀ (p1 p2 p3 : euclidean_space ℝ (fin 2)) (h1 : p1 ∈ points) (h2 : p2 ∈ points) (h3 : p3 ∈ points), ¬ collinear ℝ ({p1, p2, p3} : set (euclidean_space ℝ (fin 2))))) →
  (∃ (subset_points : finset (euclidean_space ℝ (fin 2))), subset_points.card = 100 ∧ convex_hull ℝ (↑subset_points : set (euclidean_space ℝ (fin 2))).is_convex) :=
sorry

end convex_polygon_of_100_points_l106_106717


namespace intersection_A_B_l106_106238

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l106_106238


namespace number_of_solutions_l106_106106

theorem number_of_solutions (f : ℕ → ℕ) (n : ℕ) : 
  (∀ n, f n = n^4 + 2 * n^3 - 20 * n^2 + 2 * n - 21) →
  (∀ n, 0 ≤ n ∧ n < 2013 → 2013 ∣ f n) → 
  ∃ k, k = 6 :=
by
  sorry

end number_of_solutions_l106_106106


namespace parallelogram_area_l106_106575

open Matrix -- Opening the Matrix namespace.

-- Defining the points A, B, C, and D as 3D vectors.
def A : ℝ × ℝ × ℝ := (4, -3, 2)
def B : ℝ × ℝ × ℝ := (6, -7, 5)
def C : ℝ × ℝ × ℝ := (5, -2, 0)
def D : ℝ × ℝ × ℝ := (7, -6, 3)

-- Function to compute the vector difference.
def vector_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

-- Function to compute the cross product of two 3D vectors.
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- Function to compute the norm (magnitude) of a 3D vector.
def vector_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Main theorem to prove that ABCD is a parallelogram and computing the area.
theorem parallelogram_area :
  let ab := vector_sub B A,
      cd := vector_sub D C,
      ca := vector_sub C A,
      cross := cross_product ab ca in
  ab = cd ∧ vector_norm cross = Real.sqrt 110 :=
by
  sorry

end parallelogram_area_l106_106575


namespace minimum_price_to_cover_costs_l106_106374

variable (P : ℝ)

-- Conditions
def prod_cost_A := 80
def ship_cost_A := 2
def prod_cost_B := 60
def ship_cost_B := 3
def fixed_costs := 16200
def units_A := 200
def units_B := 300

-- Cost calculations
def total_cost_A := units_A * prod_cost_A + units_A * ship_cost_A
def total_cost_B := units_B * prod_cost_B + units_B * ship_cost_B
def total_costs := total_cost_A + total_cost_B + fixed_costs

-- Revenue requirement
def revenue (P_A P_B : ℝ) := units_A * P_A + units_B * P_B

theorem minimum_price_to_cover_costs :
  (units_A + units_B) * P ≥ total_costs ↔ P ≥ 103 :=
sorry

end minimum_price_to_cover_costs_l106_106374


namespace sphere_with_one_rational_point_exists_l106_106507

theorem sphere_with_one_rational_point_exists :
  ∃ (x y z : ℚ), (x - real.sqrt 2)^2 + y^2 + z^2 = 2 ∧ (∀ (a b c : ℚ), (a - real.sqrt 2)^2 + b^2 + c^2 = 2 → (a = 0 ∧ b = 0 ∧ c = 0)) :=
begin
  sorry,
end

end sphere_with_one_rational_point_exists_l106_106507


namespace tan_of_alpha_l106_106938

theorem tan_of_alpha (α : ℝ) (h1 : Real.sin α = 1 / 4) (h2 : α ∈ (π / 2, π)) : 
  Real.tan α = -Real.sqrt 15 / 15 := by
  sorry

end tan_of_alpha_l106_106938


namespace solve_for_y_l106_106118


theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
    Matrix.det ![
        ![y + b, y, y],
        ![y, y + b, y],
        ![y, y, y + b]] = 0 → y = -b := by
  sorry

end solve_for_y_l106_106118


namespace red_or_blue_not_planar_l106_106512

theorem red_or_blue_not_planar (K : Type) [fintype K] [decidable_eq K] (hK : fintype.card K = 11)
  (color : {x y : K // x ≠ y} → bool) :
  ¬ is_planar {x y : K // x ≠ y | color ⟨x, y⟩ = tt} ∨ ¬ is_planar {x y : K // x ≠ y | color ⟨x, y⟩ = ff} :=
sorry

end red_or_blue_not_planar_l106_106512


namespace probability_A_not_first_B_not_last_l106_106425

noncomputable def probability_not_in_positions (total_people : ℕ) (chosen_people : ℕ) (valid_arrangements : ℕ) (total_arrangements : ℕ) : ℝ :=
(valid_arrangements : ℝ) / total_arrangements

theorem probability_A_not_first_B_not_last :
  let total_people := 5
  let chosen_people := 3
  let total_arrangements := 5 * 4 * 3
  let valid_arrangements := 39 in
  probability_not_in_positions total_people chosen_people valid_arrangements total_arrangements = 13 / 20 :=
by
  sorry

end probability_A_not_first_B_not_last_l106_106425


namespace parabola_vertex_coordinates_l106_106742

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), (y = -3 * (x - 1)^2 - 2) → (x, y) = (1, -2) := 
begin
  intros x y h,
  sorry
end

end parabola_vertex_coordinates_l106_106742


namespace speed_of_remaining_trip_l106_106827

theorem speed_of_remaining_trip (total_distance first_part_distance: ℝ) (first_part_speed average_speed: ℝ)
    (total_distance_eq : total_distance = 60)
    (first_part_distance_eq : first_part_distance = 30)
    (first_part_speed_eq : first_part_speed = 60)
    (average_speed_eq : average_speed = 40) : 
    let remaining_part_speed := (total_distance - first_part_distance) / 
    ((total_distance / average_speed) - (first_part_distance / first_part_speed)) in
    remaining_part_speed = 30 :=
by
  sorry

end speed_of_remaining_trip_l106_106827


namespace no_adjacent_standing_prob_l106_106734

def coin_flip_probability : ℚ :=
  let a2 := 3
  let a3 := 4
  let a4 := a3 + a2
  let a5 := a4 + a3
  let a6 := a5 + a4
  let a7 := a6 + a5
  let a8 := a7 + a6
  let a9 := a8 + a7
  let a10 := a9 + a8
  let favorable_outcomes := a10
  favorable_outcomes / (2 ^ 10)

theorem no_adjacent_standing_prob :
  coin_flip_probability = (123 / 1024 : ℚ) :=
by sorry

end no_adjacent_standing_prob_l106_106734


namespace find_a_l106_106927

open Real

noncomputable def line_parametric_eqs (t : ℝ) : ℝ × ℝ :=
  (1 / 2 * t, -1 + sqrt 3 / 2 * t)

def polar_curve_eq (a : ℝ) (theta : ℝ) : ℝ :=
  if 2 * a * sin theta - cos theta^2 = 0 then sqrt (2 * a * sin theta) else 0

theorem find_a (a : ℝ) (t1 t2 : ℝ) (h1 : a > 0)
  (h2 : 2 * t1 * t2 - (4 * sqrt 3 * a) * (t1 + t2) + 8 * a = 0)
  (h3 : (t1 + t2)^2 - 4 * t1 * t2 = t1 * t2) :
  a = 5 / 6 :=
sorry

end find_a_l106_106927


namespace binom_18_10_l106_106444

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106444


namespace strictly_increasing_interval_l106_106085

def f (x : ℝ) : ℝ := Real.log x - x

theorem strictly_increasing_interval :
  ∀ x, 0 < x ∧ x < 1 → (f x > f (0)) :=
begin
  sorry
end

end strictly_increasing_interval_l106_106085


namespace smallest_integer_problem_l106_106342

theorem smallest_integer_problem (m : ℕ) (h1 : Nat.lcm 60 m / Nat.gcd 60 m = 28) : m = 105 := sorry

end smallest_integer_problem_l106_106342


namespace percent_absent_math_dept_l106_106061

theorem percent_absent_math_dept (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_absent_fraction : ℚ) (female_absent_fraction : ℚ)
  (h1 : total_students = 160) 
  (h2 : male_students = 90) 
  (h3 : female_students = 70) 
  (h4 : male_absent_fraction = 1 / 5) 
  (h5 : female_absent_fraction = 2 / 7) :
  ((male_absent_fraction * male_students + female_absent_fraction * female_students) / total_students) * 100 = 23.75 :=
by
  sorry

end percent_absent_math_dept_l106_106061


namespace union_A_B_l106_106577

def setA : Set ℝ := { x | Real.log x / Real.log (1/2) > -1 }
def setB : Set ℝ := { x | 2^x > Real.sqrt 2 }

theorem union_A_B : setA ∪ setB = { x | 0 < x } := by
  sorry

end union_A_B_l106_106577


namespace infinite_triangles_same_color_same_area_l106_106766

theorem infinite_triangles_same_color_same_area 
  (colors : Fin 2013 → Point) 
  (color_of : Point → (Fin 2013 → Prop)) 
  (triangle_color : Triangle → (Fin 2013 → Prop)) :
  ∃ (X : Fin 2013), ∃ (T : Triangle), triangle_color T X ∧ ∃ (area : ℝ), ∀ (T' : Triangle), triangle_color T' X → area T = area T' :=
by sorry

end infinite_triangles_same_color_same_area_l106_106766


namespace discount_price_l106_106508

theorem discount_price (original_price : ℝ) (discount_percent : ℝ) (final_price : ℝ) :
  original_price = 800 ∧ discount_percent = 15 → final_price = 680 :=
by
  intros h
  cases' h with hp hd
  sorry

end discount_price_l106_106508


namespace contact_probability_l106_106536

theorem contact_probability (p : ℝ) :
  let m := 6 in
  let n := 7 in
  let number_of_pairs := m * n in
  1 - (1 - p) ^ number_of_pairs = 1 - (1 - p) ^ 42 :=
by
  let m := 6
  let n := 7
  let number_of_pairs := m * n
  have h1 : number_of_pairs = 42 := by norm_num
  rw h1
  sorry

end contact_probability_l106_106536


namespace symmetric_point_construction_l106_106124

theorem symmetric_point_construction (A O : Point) : ∃ A' : Point, midpoint O A A' :=
sorry

end symmetric_point_construction_l106_106124


namespace vertex_of_parabola_l106_106738

/-- The given parabola y = -3(x-1)^2 - 2 has its vertex at (1, -2). -/
theorem vertex_of_parabola : ∃ h k : ℝ, (h = 1 ∧ k = -2) ∧ ∀ x : ℝ, y = -3 * (x - h) ^ 2 + k :=
begin
  use [1, -2],
  split,
  { split; refl },
  { intro x,
    refl }
end

end vertex_of_parabola_l106_106738


namespace range_of_m_l106_106159

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (m x : ℝ) : ℝ := m * x + 1
noncomputable def h (x : ℝ) : ℝ := (1 / x) - (2 * Real.log x / x)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2)) ∧ (g m x = 2 - 2 * f x)) ↔
  (-2 * Real.exp (-3/2) ≤ m ∧ m ≤ 3 * Real.exp 1) :=
sorry

end range_of_m_l106_106159


namespace true_props_proved_l106_106940

-- Definitions of lines and planes
variables {Line : Type} {Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Assumptions given the conditions
axiom non_coincident_lines : m ≠ n
axiom non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Definitions of perpendicular and parallel relations
axiom perp : Line → Plane → Prop
axiom parallel : Plane → Plane → Prop
axiom line_parallel : Line → Plane → Prop
axiom line_skew : Line → Line → Prop

-- Given propositions
axiom prop1 : (∀ m α β : Line, perp m α → perp m β → parallel α β)
axiom prop2 : (∀ α β γ : Plane, perp α γ → perp β α → parallel α β)
axiom prop3 : (∀ m n : Line, ∀ α β : Plane, line_parallel m α → line_parallel n β → line_parallel m n → parallel α β)
axiom prop4 : (∀ m n : Line, ∀ α β : Plane, line_skew m n → perp m α → line_parallel m β → perp n β → line_parallel n α → perp α β)

noncomputable def true_propositions : Prop :=
  prop1 ∧ prop4

-- The theorem asserting the true propositions
theorem true_props_proved : true_propositions := sorry

end true_props_proved_l106_106940


namespace yura_has_winning_strategy_a_yura_has_winning_strategy_b_l106_106801

-- Definition of the game condition for part (a)
def yura_wins_strategy_a (n : ℕ) : Prop :=
  ∀ m ≥ n, (m < 60 → ∃ d | d ∣ m ∧ (m + d) < 60)

-- Proving the number of values for which Yura has a winning strategy for part (a)
theorem yura_has_winning_strategy_a : ∃ count, count = 29 ∧ 
  (count = (Finset.filter (λ n : ℕ, yura_wins_strategy_a n ∨ (n % 2 = 0)) (Finset.range (60 - 2))).card) := 
sorry

-- Definition of the game condition for part (b)
def yura_wins_strategy_b (n : ℕ) : Prop :=
  ∀ m ≥ n, (m < 60 → ∃ d | d ∣ m ∧ (m + d) ≥ 60)

-- Proving the number of values for which Yura has a winning strategy for part (b)
theorem yura_has_winning_strategy_b : ∃ count, count = 44 ∧ 
  (count = (Finset.filter (λ n : ℕ, yura_wins_strategy_b n ∨ (n ≥ 30 ∨ (n % 2 = 0 && (2 ≤ n) && (n < 30)))) (Finset.range (60 - 2))).card) := 
sorry

end yura_has_winning_strategy_a_yura_has_winning_strategy_b_l106_106801


namespace sequence_properties_l106_106571

noncomputable def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

theorem sequence_properties (a : ℕ → ℕ) (h_pos : ∀ n, 0 < a n)
    (h_seq : ∀ n, (a n) ^ 2 = 4 * S a n - 2 * a n - 1) :
    a 0 = 1 ∧ a 1 = 3 ∧ ∀ n, a n = 2 * n + 1 := 
by {
    sorry
}

end sequence_properties_l106_106571


namespace parallel_planes_k_equiv_l106_106947

-- Conditions
def n1 : ℝ × ℝ × ℝ := (1, 2, -2)
def n2 (k : ℝ) : ℝ × ℝ × ℝ := (-2, -4, k)
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ c : ℝ, b = (c * a.1, c * a.2, c * a.3)

-- Proof goal
theorem parallel_planes_k_equiv (k : ℝ) : parallel n1 (n2 k) ↔ k = 4 := by
  sorry

end parallel_planes_k_equiv_l106_106947


namespace domain_of_f_l106_106784

def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ∈ set.Ioo (-∞) 6 ∪ set.Ioo 6 ∞) :=
by
  sorry

end domain_of_f_l106_106784


namespace no_such_sequence_exists_l106_106893

theorem no_such_sequence_exists : ¬ ∃ (a : ℕ → ℕ), (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, a (n + 2) = a (n + 1) + nat.sqrt (a n + a (n + 1))) :=
begin
  sorry
end

end no_such_sequence_exists_l106_106893


namespace reflections_of_orthocenter_on_circumcircle_l106_106671

variable (A B C H: Type)
variable [AffineGeometry A B C H]

def is_orthocenter (ABC: Triangle) (H: Point) : Prop := 
  -- Definition of orthocenter would go here

def is_reflection_over_side (H: Point) (side: Line) : Point := 
  -- Reflection function definition would go here

theorem reflections_of_orthocenter_on_circumcircle 
  (ABC: Triangle)
  (H: Point)
  (is_acute: acute_triangle ABC)
  (H_is_orthocenter: is_orthocenter ABC H) :
  ∀ side: Line, (reflect H side ∈ circumcircle ABC.side) :=
by 
  sorry

end reflections_of_orthocenter_on_circumcircle_l106_106671


namespace range_of_a_l106_106136

theorem range_of_a (a : ℝ) :
  (a + 1 > 0 ∧ 3 - 2 * a > 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a < 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a > 0)
  → (2 / 3 < a ∧ a < 3 / 2) ∨ (a < -1) :=
by
  sorry

end range_of_a_l106_106136


namespace find_t_l106_106099

variables (a b : ℂ) (t : ℂ)

-- Conditions
def condition1 : |a| = 2 := sorry
def condition2 : |b| = 5 := sorry
def condition3 : a * b = t - 3 * complex.I := sorry

-- Theorem statement
theorem find_t (h1 : |a| = 2) (h2 : |b| = 5) (h3 : a * b = t - 3 * complex.I) : t = complex.sqrt 91 := 
sorry

end find_t_l106_106099


namespace binom_18_10_l106_106469

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106469


namespace addition_example_l106_106300

theorem addition_example : 300 + 2020 + 10001 = 12321 := 
by 
  sorry

end addition_example_l106_106300


namespace number_of_strikers_correct_l106_106393

-- Defining the initial conditions
def number_of_goalies := 3
def number_of_defenders := 10
def number_of_players := 40
def number_of_midfielders := 2 * number_of_defenders

-- Lean statement to prove
theorem number_of_strikers_correct : 
  let total_non_strikers := number_of_goalies + number_of_defenders + number_of_midfielders,
      number_of_strikers := number_of_players - total_non_strikers 
  in number_of_strikers = 7 :=
by
  sorry

end number_of_strikers_correct_l106_106393


namespace solution_l106_106566

def f : ℕ → ℕ := sorry
def p : ℕ → ℕ := sorry

axiom f_initial : f 1 = 1
axiom f_inequality (n : ℕ) (hn : 0 < n) : f (n + 1) ≥ f n + 2^n
axiom p_nonnegative (n : ℕ) : p n ≥ 0

theorem solution (n : ℕ) (hn : 0 < n) :
  f n = 2^n + ∑ i in Finset.range (n+1), p i :=
sorry

end solution_l106_106566


namespace mouse_jump_frog_jump_diff_l106_106753

open Nat

theorem mouse_jump_frog_jump_diff :
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  mouse_jump - frog_jump = 20 :=
by
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  have h1 : frog_jump = 29 := by decide
  have h2 : mouse_jump = 49 := by decide
  have h3 : mouse_jump - frog_jump = 20 := by decide
  exact h3

end mouse_jump_frog_jump_diff_l106_106753


namespace magnitude_of_a_plus_2b_l106_106174

variables (x : ℝ)

def a : ℝ × ℝ := (2, -1)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem magnitude_of_a_plus_2b (h : dot_product a (b x) = 0) : ‖(2 + 2 * x, -1 + 2)‖ = real.sqrt 10 := by 
  sorry

end magnitude_of_a_plus_2b_l106_106174


namespace eight_boys_travel_distance_l106_106894

theorem eight_boys_travel_distance (r : ℝ) (n : ℕ) (h_r : r = 50) (h_n : n = 8) :
  let diameter := 2 * r,
      pair_distance := 2 * diameter,
      total_distance := (n / 2) * pair_distance
  in total_distance = 800 :=
by 
  sorry -- Complete this proof by filling in the omitted steps.

end eight_boys_travel_distance_l106_106894


namespace convex_quad_no_parallel_sides_l106_106934

/--
If four identical triangles form a convex quadrilateral,
then it must have parallel sides.
-/
theorem convex_quad_no_parallel_sides (T : Type) [is_triangle T] :
  ¬ (∀ (quad : quadrilateral T), quad.is_convex → quad.has_parallel_sides) :=
sorry

end convex_quad_no_parallel_sides_l106_106934


namespace probability_xy_odd_l106_106199

theorem probability_xy_odd :
  let xs := {1, 2, 3, 4}
  let ys := {5, 6, 7}
  let odd_xs := xs.filter (λ x, x % 2 = 1)
  let odd_ys := ys.filter (λ y, y % 2 = 1)
  let total_outcomes := xs.card * ys.card
  let favorable_outcomes := odd_xs.card * odd_ys.card
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 3 :=
by
  -- Definitions for the purpose of stating the theorem
  let xs := {1, 2, 3, 4}
  let ys := {5, 6, 7}
  let odd_xs := xs.filter (λ x, x % 2 = 1)
  let odd_ys := ys.filter (λ y, y % 2 = 1)
  let total_outcomes := xs.card * ys.card
  let favorable_outcomes := odd_xs.card * odd_ys.card
  have h1 : xs = {1, 2, 3, 4} := rfl
  have h2 : ys = {5, 6, 7} := rfl
  have hx : odd_xs = {1, 3} := rfl
  have hy : odd_ys = {5, 7} := rfl
  have ht : total_outcomes = 12 := by simp [total_outcomes, xs, ys]
  have hf : favorable_outcomes = 4 := by simp [favorable_outcomes, odd_xs, odd_ys]
  have hp : (4 / 12 : ℚ) = 1 / 3 := by norm_num
  show (favorable_outcomes / total_outcomes : ℚ) = 1 / 3, from sorry

end probability_xy_odd_l106_106199


namespace giraffes_count_l106_106318

/-- Number of animals that are not giraffes -/
variable (x : ℕ)

/-- The number of giraffes is 3 times the number of non-giraffe animals -/
def giraffes (x : ℕ) : ℕ := 3 * x

/-- There are 290 more giraffes than non-giraffe animals -/
def giraffes_gt_others (x : ℕ) : Prop := giraffes x = x + 290

/-- Prove that the number of giraffes is 435 -/
theorem giraffes_count (x : ℕ) (h : giraffes_gt_others x) : giraffes x = 435 :=
by
  sorry

end giraffes_count_l106_106318


namespace projection_matrix_inverse_P_l106_106242

-- Given conditions
def vector := ℝ × ℝ
def projection_matrix (v : vector) : Matrix (Fin 2) (Fin 2) ℝ :=
  let (x, y) := v
  let norm_sq := x^2 + y^2
  (1 / norm_sq) • !![(x^2, x*y), (x*y, y^2)]

-- The problem's matrix for projection onto the vector (-3, -2)
def P : Matrix (Fin 2) (Fin 2) ℝ := projection_matrix (-3, -2)

-- The proof problem statement
theorem projection_matrix_inverse_P : ¬ (invertible P) ∧ (Matrix.mulVec P P = 0) := by
  sorry

end projection_matrix_inverse_P_l106_106242


namespace contact_probability_l106_106540

variable (m : ℕ := 6) (n : ℕ := 7) (p : ℝ)

theorem contact_probability :
  let total_pairs := m * n in
  let probability_no_contact := (1 - p) ^ total_pairs in
  let probability_contact := 1 - probability_no_contact in
  probability_contact = 1 - (1 - p) ^ 42 :=
by
  -- This is where the proof would go
  sorry

end contact_probability_l106_106540


namespace domain_of_f_intervals_of_monotonicity_extremal_values_l106_106959

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x 

theorem domain_of_f : ∀ x, 0 < x → f x = (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x :=
by
  intro x hx
  exact rfl

theorem intervals_of_monotonicity :
  (∀ x, 0 < x ∧ x < 1 → f x < f 1) ∧
  (∀ x, 1 < x ∧ x < 4 → f x > f 1 ∧ f x < f 4) ∧
  (∀ x, 4 < x → f x > f 4) :=
sorry

theorem extremal_values :
  (f 1 = - (9 / 2)) ∧ 
  (f 4 = -12 + 4 * Real.log 4) :=
sorry

end domain_of_f_intervals_of_monotonicity_extremal_values_l106_106959


namespace allison_wins_l106_106050

/-- Conditions describing the die faces /--
def allison_faces := {3, 3, 3, 4, 4, 4}
def brian_faces := {0, 1, 2, 3, 4, 5}
def noah_faces := {2, 2, 2, 6, 6, 6}

/-- Function to calculate the winning probability for Allison /--
noncomputable def winning_probability : ℚ :=
  have p_brian_less_3 := 3 / 6
  have p_brian_less_4 := 4 / 6
  have p_noah_less_3 := 3 / 6
  have p_noah_less_4 := 3 / 6

  have p_allison_3 := p_brian_less_3 * p_noah_less_3
  have p_allison_4 := p_brian_less_4 * p_noah_less_4

  (1 / 2) * (p_allison_3 + p_allison_4)

theorem allison_wins :
  winning_probability = 7 / 24 :=
by
  sorry

end allison_wins_l106_106050


namespace prob_B_second_shot_prob_A_i_th_shot_expected_number_shots_A_l106_106713

-- Define the shooting accuracies and initial probabilities
def shooting_accuracy_A : ℝ := 0.6
def shooting_accuracy_B : ℝ := 0.8
def initial_prob_A : ℝ := 0.5
def initial_prob_B : ℝ := 0.5

-- Probability that player B takes the second shot
theorem prob_B_second_shot : (initial_prob_A * (1 - shooting_accuracy_A) + initial_prob_B * shooting_accuracy_B) = 0.6 :=
by sorry

-- Probability that player A takes the i-th shot
def prob_A_i (i : ℕ) : ℝ :=
1/3 + (1/6) * (2/5)^(i-1)

-- Expected number of times player A shoots in the first n shots
def expected_A_shots (n : ℕ) : ℝ :=
(5/18) * (1 - (2/5)^n) + n/3

-- The probability of player A taking the i-th shot
theorem prob_A_i_th_shot (i : ℕ) : prob_A_i i = (1/3 + (1/6) * (2/5)^(i-1)) :=
by sorry

-- The expected number of times player A shoots in the first n shots
theorem expected_number_shots_A (n : ℕ) : expected_A_shots n = ((5/18) * (1 - (2/5)^n) + n/3) :=
by sorry

end prob_B_second_shot_prob_A_i_th_shot_expected_number_shots_A_l106_106713


namespace oil_cost_calculation_l106_106698

def cost_of_beef (pounds_of_beef : ℕ) (cost_per_pound : ℕ) : ℕ :=
  pounds_of_beef * cost_per_pound

def cost_of_chicken (people : ℕ) (cost_per_person : ℕ) : ℕ :=
  people * cost_per_person

def total_paid (cost_of_beef : ℕ) (cost_of_chicken : ℕ) : ℕ :=
  cost_of_beef + cost_of_chicken

def cost_of_oil (total_cost : ℕ) (total_paid : ℕ) : ℕ :=
  total_cost - total_paid

theorem oil_cost_calculation :
  let pounds_of_beef := 3 in
  let cost_per_pound := 4 in
  let people := 3 in
  let cost_per_person := 1 in
  let total_grocery_cost := 16 in
  cost_of_oil total_grocery_cost (total_paid (cost_of_beef pounds_of_beef cost_per_pound) (cost_of_chicken people cost_per_person)) = 1 :=
by
  sorry

end oil_cost_calculation_l106_106698


namespace painted_cube_eq_unpainted_l106_106853

theorem painted_cube_eq_unpainted {n : ℕ} (h : n > 2) 
  (h_condition : ∀ unit_cubes, painted_faces_two = unpainted)
  : n = 2 * Real.sqrt 3 + 2 := 
by
  sorry

end painted_cube_eq_unpainted_l106_106853


namespace sum_of_faces_l106_106311

theorem sum_of_faces (n_side_faces_per_prism : ℕ) (n_non_side_faces_per_prism : ℕ)
  (num_prisms : ℕ) (h1 : n_side_faces_per_prism = 3) (h2 : n_non_side_faces_per_prism = 2) 
  (h3 : num_prisms = 3) : 
  n_side_faces_per_prism * num_prisms + n_non_side_faces_per_prism * num_prisms = 15 :=
by
  sorry

end sum_of_faces_l106_106311


namespace square_area_l106_106946

-- Definitions for conditions
def fixed_point : ℝ × ℝ := (1/4, 0)
def tangent_line_x_eq := -1/4

-- Definitions for the parabola trajectory
def trajectory (x y : ℝ) : Prop := y^2 = x

-- Definitions for the square ABCD conditions
def ab_line (x y : ℝ) : Prop := y = x + 4
def points_on_trajectory (x y : ℝ) : Prop := trajectory x y

-- Theorem Statement
theorem square_area (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : ab_line x₁ y₁)
  (h2 : points_on_trajectory x₂ y₂)
  (h3 : points_on_trajectory x₂ y₂) :
  (∃ side_length: ℝ, side_length = 3 * Real.sqrt 2 ∨ side_length = 5 * Real.sqrt 2) → 
  ((3 * Real.sqrt 2)^2 = 18 ∧ (5 * Real.sqrt 2)^2 = 50) :=
by
  sorry

end square_area_l106_106946


namespace contact_probability_l106_106546

-- Definition of the number of tourists in each group
def num_tourists_group1 : ℕ := 6
def num_tourists_group2 : ℕ := 7
def total_pairs : ℕ := num_tourists_group1 * num_tourists_group2

-- Definition of probability for no contact
def p : ℝ -- probability of contact
def prob_no_contact := (1 - p) ^ total_pairs

-- The theorem to be proven
theorem contact_probability : 1 - prob_no_contact = 1 - (1 - p) ^ total_pairs :=
by
  sorry

end contact_probability_l106_106546


namespace min_value_l106_106249

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) : 
  ∃ c : ℝ, (c = 3 / 4) ∧ (∀ (a b c : ℝ), a = x ∧ b = y ∧ c = z → 
    (1/(a + 3*b) + 1/(b + 3*c) + 1/(c + 3*a)) ≥ c) :=
sorry

end min_value_l106_106249


namespace ratio_is_7_to_10_l106_106419

-- Given conditions in the problem translated to Lean definitions
def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 10 * leopards
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := 670
def other_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + alligators
def cheetahs : ℕ := total_animals - other_animals

-- The ratio of cheetahs to snakes to be proven
def ratio_cheetahs_to_snakes (cheetahs snakes : ℕ) : ℚ := cheetahs / snakes

theorem ratio_is_7_to_10 : ratio_cheetahs_to_snakes cheetahs snakes = 7 / 10 :=
by
  sorry

end ratio_is_7_to_10_l106_106419


namespace part1_part2_l106_106254

noncomputable def f (x : ℝ) : ℝ :=
  abs (2 * x - 3) + abs (x - 5)

theorem part1 : { x : ℝ | f x ≥ 4 } = { x : ℝ | x ≥ 2 ∨ x ≤ 4 / 3 } :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x < a) ↔ a > 7 / 2 :=
by
  sorry

end part1_part2_l106_106254


namespace find_genuine_coin_with_two_weighings_l106_106051

theorem find_genuine_coin_with_two_weighings (coins : Fin 100 → Prop) (genuine counterfeit : ℝ)
  (h_genuine_not_counterfeit : genuine > counterfeit) (h_counterfeits : ∃ S : Finset (Fin 100), S.card = 4 ∧ ∀ x ∈ S, coins x)
  (h_genuine : ∀ x ∉ { y | coins y }, weight x = genuine) (h_counterfeit : ∀ x ∈ { y | coins y }, weight x = counterfeit) :
  ∃ x : Fin 100, (weight x = genuine) :=
by
  -- Proof goes here
  sorry

end find_genuine_coin_with_two_weighings_l106_106051


namespace sum_of_digits_least_N_l106_106692

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2 * N : ℚ) / 5⌉) / (N + 1)

theorem sum_of_digits_least_N (k : ℕ) (h_k : k = 2) (h1 : ∀ N, P N k < 8 / 10 ) :
  ∃ N : ℕ, (N % 10) + (N / 10) = 1 ∧ (P N k < 8 / 10) ∧ (∀ M : ℕ, M < N → P M k ≥ 8 / 10) := by
  sorry

end sum_of_digits_least_N_l106_106692


namespace exists_number_added_to_sum_of_digits_gives_2014_l106_106371

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem exists_number_added_to_sum_of_digits_gives_2014 : 
  ∃ (n : ℕ), n + sum_of_digits n = 2014 :=
sorry

end exists_number_added_to_sum_of_digits_gives_2014_l106_106371


namespace calories_burned_in_40_minutes_l106_106012

-- Defining the given conditions
variables (caloriesPerMinute : ℝ) (caloriesIn25Minutes : ℝ) (minutesIn25Minutes : ℝ) (minutesIn40Minutes : ℝ)

-- Given values
def caloriesPerMinute := 300 / 25
def caloriesIn25Minutes := 300
def minutesIn25Minutes := 25
def minutesIn40Minutes := 40

-- Statement to prove
theorem calories_burned_in_40_minutes :
  (caloriesIn40Minutes : ℝ) =
    (caloriesIn25Minutes * minutesIn40Minutes / minutesIn25Minutes) :=
by
  sorry

end calories_burned_in_40_minutes_l106_106012


namespace sufficient_but_not_necessary_l106_106941

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x + y = 1 → xy ≤ 1 / 4) ∧ (∃ x y : ℝ, xy ≤ 1 / 4 ∧ x + y ≠ 1) := by
  sorry

end sufficient_but_not_necessary_l106_106941


namespace hannah_highest_score_l106_106621

theorem hannah_highest_score :
  ∀ (total_questions : ℕ) (wrong_answers_student1 : ℕ) (percentage_student2 : ℚ),
  total_questions = 40 →
  wrong_answers_student1 = 3 →
  percentage_student2 = 0.95 →
  ∃ (correct_answers_hannah : ℕ), correct_answers_hannah > 38 :=
by {
  intros,
  sorry,
}

end hannah_highest_score_l106_106621


namespace binom_18_10_l106_106452

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106452


namespace find_p_q_of_odd_function_and_value_f_monotone_decreasing_on_interval_l106_106560

-- Define the function f and the conditions given in the problem
def f (x : ℝ) (p q : ℝ) : ℝ := (x^2 + p) / (x + q)

-- Question 1: Prove that if f is an odd function and f(2) = 4, then p = 4 and q = 0
theorem find_p_q_of_odd_function_and_value (p q : ℝ) (h1 : ∀ x : ℝ, f x p q = -f (-x) p q) (h2 : f 2 p q = 4) : p = 4 ∧ q = 0 :=
by sorry

-- Define the function given p = 4 and q = 0
def f_fixed (x : ℝ) : ℝ := x + 4 / x

-- Question 2: Prove that f is monotonically decreasing on (0, 2)
theorem f_monotone_decreasing_on_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → f_fixed x < f_fixed (x + 1) :=
by sorry

end find_p_q_of_odd_function_and_value_f_monotone_decreasing_on_interval_l106_106560


namespace initial_girls_count_l106_106006

theorem initial_girls_count (p : ℕ) (h₁ : 0.6 * p = 0.6 * (p : ℝ))
  (h₂ : (0.6 * p - 3) / p = 0.5) : 
  0.6 * p = 18 :=
by
  sorry

end initial_girls_count_l106_106006


namespace bread_needed_for_sandwiches_l106_106849

def students_per_group := 5
def groups := 5
def sandwiches_per_student := 2
def pieces_of_bread_per_sandwich := 2

theorem bread_needed_for_sandwiches : 
  students_per_group * groups * sandwiches_per_student * pieces_of_bread_per_sandwich = 100 := 
by
  sorry

end bread_needed_for_sandwiches_l106_106849


namespace intersection_P_Q_l106_106915

open Set

noncomputable def P : Set ℝ := {-1, 0, Real.sqrt 2}

def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

theorem intersection_P_Q : P ∩ Q = {-1, 0} :=
by
  sorry

end intersection_P_Q_l106_106915


namespace alex_integer_list_count_l106_106406

theorem alex_integer_list_count : 
  let n := 12 
  let least_multiple := 2^6 * 3^3
  let count := least_multiple / n
  count = 144 :=
by
  sorry

end alex_integer_list_count_l106_106406


namespace packs_split_l106_106661

/-- James and his friend split 4 packs of stickers given specific conditions. --/
theorem packs_split (packs_stickers : ℕ) (sticker_cost : ℕ) (half_payment : ℕ) (james_payment : ℕ) 
  (packs_stickers_def : packs_stickers = 30)
  (sticker_cost_def : sticker_cost = 10) -- using cents to avoid decimals
  (half_payment_def : half_payment = 2 * james_payment) 
  (james_payment_def : james_payment = 600) : 
  (james_payment * 2) / sticker_cost / packs_stickers = 4 :=
by
  rw [packs_stickers_def, sticker_cost_def, half_payment_def, james_payment_def]
  norm_num
  sorry

end packs_split_l106_106661


namespace price_of_each_tomato_l106_106781

theorem price_of_each_tomato
  (customers_per_month : ℕ)
  (lettuce_per_customer : ℕ)
  (lettuce_price : ℕ)
  (tomatoes_per_customer : ℕ)
  (total_monthly_sales : ℕ)
  (total_lettuce_sales : ℕ)
  (total_tomato_sales : ℕ)
  (price_per_tomato : ℝ)
  (h1 : customers_per_month = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomatoes_per_customer = 4)
  (h5 : total_monthly_sales = 2000)
  (h6 : total_lettuce_sales = customers_per_month * lettuce_per_customer * lettuce_price)
  (h7 : total_tomato_sales = total_monthly_sales - total_lettuce_sales)
  (h8 : total_lettuce_sales = 1000)
  (h9 : total_tomato_sales = 1000)
  (total_tomatoes_sold : ℕ := customers_per_month * tomatoes_per_customer)
  (h10 : total_tomatoes_sold = 2000) :
  price_per_tomato = 0.50 :=
by
  sorry

end price_of_each_tomato_l106_106781


namespace balls_in_boxes_l106_106182

open Nat

theorem balls_in_boxes : 
  let balls := 7
  let boxes := 4
  (∑ (x : Fin 11), match x.1 with
  | 0 => 1 
  | 1 => choose 7 6
  | 2 => choose 7 5 * choose 2 2
  | 3 => choose 7 5
  | 4 => choose 7 4 * choose 3 3
  | 5 => choose 7 4 * choose 3 2
  | 6 => choose 7 4
  | 7 => choose 7 3 * choose 4 3 / 2
  | 8 => choose 7 3 * choose 4 2 / 2
  | 9 => choose 7 3 * choose 4 2
  | 10 => choose 7 2 * choose 5 2 * choose 3 2 / 2
  end) = 890 :=
by 
  sorry

end balls_in_boxes_l106_106182


namespace place_mat_length_correct_l106_106001

noncomputable def place_mat_length (r w : ℝ) : ℝ := 10 * Real.sin (Real.pi / 8)

theorem place_mat_length_correct (r : ℝ) (w : ℝ) (x : ℝ) (h_r : r = 5) (h_w : w = 1) 
  (h_mats : ∀ (i : Fin 8), 
    let side := place_mat_length r w in
    side^2 = (w / 2)^2 + (r - x)^2) : 
  x = 3.82 :=
by sorry

end place_mat_length_correct_l106_106001


namespace first_digit_base8_853_l106_106789

theorem first_digit_base8_853 (n : ℕ) (h : n = 853) : (n / (8 ^ 3)) = 1 :=
by
  have h₁ : (8 ^ 3) = 512 := by norm_num
  rw [h, h₁]
  norm_num

end first_digit_base8_853_l106_106789


namespace quadrilateral_is_kite_l106_106234

variable (A B C D : Type) [EuclideanGeometry A B C D]

variable (α λ : ℝ)
variable (b k : ℝ)
variable (AB CB AD DC : ℝ)

def is_kite (AB AD BC CD : ℝ) : Prop :=
  AB = BC ∧ AD = CD

theorem quadrilateral_is_kite 
  (h1 : ∃ AB CB AD DC : ℝ, is_kite AB AD CB DC) 
  (h2 : ∠ CBD = 2 * ∠ ADB) 
  (h3 : ∠ ABD = 2 * ∠ CDB) 
  (h4 : AB = CB)
  : is_kite AB AD CB DC := 
sorry

end quadrilateral_is_kite_l106_106234


namespace contact_probability_l106_106538

theorem contact_probability (p : ℝ) :
  let m := 6 in
  let n := 7 in
  let number_of_pairs := m * n in
  1 - (1 - p) ^ number_of_pairs = 1 - (1 - p) ^ 42 :=
by
  let m := 6
  let n := 7
  let number_of_pairs := m * n
  have h1 : number_of_pairs = 42 := by norm_num
  rw h1
  sorry

end contact_probability_l106_106538


namespace maurice_rides_l106_106703

theorem maurice_rides (M : ℕ) 
    (h1 : ∀ m_attended : ℕ, m_attended = 8)
    (h2 : ∀ matt_other : ℕ, matt_other = 16)
    (h3 : ∀ total_matt : ℕ, total_matt = matt_other + m_attended)
    (h4 : total_matt = 3 * M) : M = 8 :=
by 
  sorry

end maurice_rides_l106_106703


namespace complex_number_quadrant_l106_106680

open Complex

def z : ℂ := (2 - I) / (1 + I)

theorem complex_number_quadrant : (Re z > 0) ∧ (Im z < 0) := by
  sorry

end complex_number_quadrant_l106_106680


namespace arithmetic_sequence_increasing_not_necessarily_positive_l106_106241

theorem arithmetic_sequence_increasing_not_necessarily_positive
  (d : ℝ) (a₁ : ℝ) (h : d ≠ 0) :
  (∀ n : ℕ, n > 0 → (S_n d a₁ n < S_{n+1} d a₁ n)) → ∃ n : ℕ, n > 0 ∧ S_n d a₁ n ≤ 0 :=
by
  sorry

def S_n (d : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0  
  else d / 2 * n ^ 2 + (a₁ - d / 2) * n

noncomputable def S_{n+1} (d : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ :=
  S_n d a₁ (n + 1)

end arithmetic_sequence_increasing_not_necessarily_positive_l106_106241


namespace prove_seq_formula_l106_106130

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 1
| 1     => 5
| n + 2 => (2 * (seq a (n + 1))^2 - 3 * (seq a (n + 1)) - 9) / (2 * (seq a n))

theorem prove_seq_formula : ∀ (n : ℕ), seq a n = 2^(n + 2) - 3 :=
by
  sorry  -- Proof not needed for the mathematical translation

end prove_seq_formula_l106_106130


namespace integral_of_sqrt_1_minus_x_sq_l106_106090

theorem integral_of_sqrt_1_minus_x_sq :
  ∫ x in -1..0, sqrt (1 - x^2) = Real.pi / 4 :=
by
  sorry

end integral_of_sqrt_1_minus_x_sq_l106_106090


namespace cleaning_time_is_100_l106_106705

def time_hosing : ℕ := 10
def time_shampoo_per : ℕ := 15
def num_shampoos : ℕ := 3
def time_drying : ℕ := 20
def time_brushing : ℕ := 25

def total_time : ℕ :=
  time_hosing + (num_shampoos * time_shampoo_per) + time_drying + time_brushing

theorem cleaning_time_is_100 :
  total_time = 100 :=
by
  sorry

end cleaning_time_is_100_l106_106705


namespace unit_of_measurements_l106_106095

axiom bottle_cough_syrup_capacity : ℕ → Prop 
axiom warehouse_capacity : ℕ → Prop 
axiom barrel_gasoline_capacity : ℕ → Prop 

theorem unit_of_measurements :
  (∀ n, bottle_cough_syrup_capacity n → n = 150 ∧ "milliliters") ∧
  (∀ n, warehouse_capacity n → n = 400 ∧ "tons") ∧
  (∀ n, barrel_gasoline_capacity n → n = 150 ∧ "liters") :=
by
  sorry

end unit_of_measurements_l106_106095


namespace graph_of_g_plus_g_neg_x_l106_106078

noncomputable def g (x : ℝ) : ℝ :=
if -3 ≤ x ∧ x ≤ 0 then 3 - x
else if 0 ≤ x ∧ x ≤ 2 then (sqrt (9 - (x - 1.5)^2)) - 3
else if 2 ≤ x ∧ x ≤ 4 then 3 * (x - 2)
else 0

theorem graph_of_g_plus_g_neg_x : 
  ∀ x : ℝ, 
  (-4 ≤ x ∧ x ≤ -3 ∧ g(x) + g(-x) = -3 * (x + 2)) ∨
  (-3 ≤ x ∧ x ≤ 0 ∧ g(x) + g(-x) = 6) ∨
  (0 ≤ x ∧ x ≤ 2 ∧ g(x) + g(-x) = 2 * sqrt(9 - (x - 1.5)^2) - 6) ∨
  (2 ≤ x ∧ x ≤ 4 ∧ g(x) + g(-x) = 3 * (x - 2)) :=
by
  intros x
  sorry

end graph_of_g_plus_g_neg_x_l106_106078


namespace who_drank_most_juice_l106_106662

theorem who_drank_most_juice : 
  ∃ person : String, 
  (person = "Yuna") ∧ 
  let Jimin := 0.7 in
  let Eunji := Jimin - 0.1 in
  let Yoongi := 4 / 5 in
  let Yuna := Jimin + 0.2 in
  (Yuna > Jimin) ∧ (Yuna > Eunji) ∧ (Yuna > Yoongi) := by
  sorry

end who_drank_most_juice_l106_106662


namespace vertex_of_parabola_l106_106740

/-- The given parabola y = -3(x-1)^2 - 2 has its vertex at (1, -2). -/
theorem vertex_of_parabola : ∃ h k : ℝ, (h = 1 ∧ k = -2) ∧ ∀ x : ℝ, y = -3 * (x - h) ^ 2 + k :=
begin
  use [1, -2],
  split,
  { split; refl },
  { intro x,
    refl }
end

end vertex_of_parabola_l106_106740


namespace domain_of_sqrt_function_l106_106747

theorem domain_of_sqrt_function :
  { x : ℝ | 2 * x - 1 ≥ 0 } = set.Ici (1 / 2) :=
by
  sorry

end domain_of_sqrt_function_l106_106747


namespace determine_polynomial_after_queries_l106_106732

theorem determine_polynomial_after_queries (k : ℕ) (h : k > 0) :
  ∃ (f : ℤ → ℤ) (p q : ℤ), 
  (∀ x : ℤ, f(x) = x^3 + p * x + q) ∧
  (∃ r s t : ℤ, abs r < 3 * 2^k ∧ abs s < 3 * 2^k ∧ abs t < 3 * 2^k ∧
   f(r) = 0 ∧ f(s) = 0 ∧ f(t) = 0) ∧
  ∃ strategy : (ℤ → Prop),
  (∀ x : ℤ, strategy x = (f(x) = 0 ∨ f(x) > 0 ∨ f(x) < 0)) ∧ 
  (∀ p q : ℤ, 
    let f := (λ x, x^3 + p * x + q) in
    ∃ r s t : ℤ,
    abs r < 3 * 2^k ∧ abs s < 3 * 2^k ∧ abs t < 3 * 2^k ∧
    f(r) = 0 ∧ f(s) = 0 ∧ f(t) = 0 ∧
    (λ (affirmations made : ℤ), made ≤ 2 * k + 1)) :=
sorry

end determine_polynomial_after_queries_l106_106732


namespace binom_18_10_l106_106436

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106436


namespace limit_f_at_1_l106_106598

def f (x : ℝ) : ℝ := (5 / 3) * x - Real.log (2 * x + 1)

theorem limit_f_at_1 :
  filter.tendsto (λ x, (f (1 + x) - f 1) / x) (nhds 0) (nhds 1) :=
sorry

end limit_f_at_1_l106_106598


namespace field_trip_bread_l106_106843

theorem field_trip_bread (group_size : ℕ) (groups : ℕ) 
    (students_per_group : group_size = 5 + 1)
    (total_groups : groups = 5)
    (sandwiches_per_student : ℕ := 2) 
    (bread_per_sandwich : ℕ := 2) : 
    (groups * group_size * sandwiches_per_student * bread_per_sandwich) = 120 := 
by 
    have students_per_group_lemma : group_size = 6 := by sorry
    have total_students := groups * group_size
    have _ : total_students = 30 := by sorry
    have total_sandwiches := total_students * sandwiches_per_student
    have _ : total_sandwiches = 60 := by sorry
    have total_bread := total_sandwiches * bread_per_sandwich
    have _ : total_bread = 120 := by sorry
    exact id 120

end field_trip_bread_l106_106843


namespace conjugate_in_fourth_quadrant_l106_106737

noncomputable def z : ℂ := (3 + 2 * Complex.i) / (1 - Complex.i)

def conjugate_z : ℂ := Complex.conj z

def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem conjugate_in_fourth_quadrant : in_fourth_quadrant conjugate_z :=
by
  sorry

end conjugate_in_fourth_quadrant_l106_106737


namespace binom_18_10_l106_106472

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106472


namespace count_digit_7_l106_106815

theorem count_digit_7 :
  let count_7 := (λ (n : ℕ) => n.digits.filter (λ digit => digit = 7)).length in
  (list.range' 1 999).sum (λ n => count_7 n) = 268 :=
by {
  sorry
}

end count_digit_7_l106_106815


namespace max_value_sum_l106_106141

theorem max_value_sum {n : ℕ} (n_pos : 0 < n) :
  ∃ (x : fin (2 * n) → ℤ), 
    (∀ i, -1 ≤ x i ∧ x i ≤ 1) ∧ 
    (∑ 1 ≤ r < s ≤ 2 * n, (s - r - n) * (x r) * (x s) = n * (n - 1)) :=
sorry

end max_value_sum_l106_106141


namespace projection_of_a_onto_b_l106_106121

open Real

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (h : a ⋅ (a + b) = 0)

theorem projection_of_a_onto_b :
  (a ⋅ b) / ‖b‖ = -1 / 2 :=
sorry

end projection_of_a_onto_b_l106_106121


namespace nancy_indian_food_freq_l106_106260

-- Definitions based on the problem
def antacids_per_indian_day := 3
def antacids_per_mexican_day := 2
def antacids_per_other_day := 1
def mexican_per_week := 2
def total_antacids_per_month := 60
def weeks_per_month := 4
def days_per_week := 7

-- The proof statement
theorem nancy_indian_food_freq :
  ∃ (I : ℕ), (total_antacids_per_month = 
    weeks_per_month * (antacids_per_indian_day * I + 
    antacids_per_mexican_day * mexican_per_week + 
    antacids_per_other_day * (days_per_week - I - mexican_per_week))) ∧ I = 3 :=
by
  sorry

end nancy_indian_food_freq_l106_106260


namespace log_sum_geometric_sequence_l106_106170

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a n > 0

noncomputable def given_condition (a : ℕ → ℝ) : Prop :=
∀ n ≥ 3, a 5 * a (2 * n - 5) = 2 ^ (2 * n)

theorem log_sum_geometric_sequence (a : ℕ → ℝ) :
  (geometric_sequence a) ∧ (given_condition a) → 
  ∀ n ≥ 1, (list.sum (list.map (λ k, real.logb 2 (a (2 * k + 1))) (list.range n))) = n^2 :=
begin
  -- Proof will be provided here
  sorry
end

end log_sum_geometric_sequence_l106_106170


namespace range_of_n_l106_106954

theorem range_of_n (m n : ℝ) (h1 : m^2 + n ≠ 0) (h2 : 3 * m^2 - n ≠ 0)
  (h3 : (m^2 + n) * (3 * m^2 - n) > 0)
  (h4 : 2 * real.sqrt ((m^2 + n) * (3 * m^2 - n)) = 4) : -1 < n ∧ n < 3 :=
by
  sorry

end range_of_n_l106_106954


namespace find_a_range_l106_106149

open Set

def solveSetInequality (a : ℝ) : Set ℝ :=
  { x | x^2 - 2*a*x - 3*a^2 ≤ 0 }

def solveFractionInequality : Set ℝ :=
  { x | x + 1 ≤ 0 ∨ x - 3 = 0 }

def rangeA : Set ℝ := Icc (-1) 3

def rangeB (a : ℝ) : Set ℝ :=
  if a = 0 then {0}
  else if a > 0 then Icc (-a) (3*a)
  else Icc (3*a) (-a)

theorem find_a_range (a : ℝ) : (rangeB(a) ⊆ rangeA) ↔ (-1/3 ≤ a ∧ a < 1) :=
  sorry

end find_a_range_l106_106149


namespace more_visitors_that_day_l106_106405

def number_of_visitors_previous_day : ℕ := 100
def number_of_visitors_that_day : ℕ := 666

theorem more_visitors_that_day :
  number_of_visitors_that_day - number_of_visitors_previous_day = 566 :=
by
  sorry

end more_visitors_that_day_l106_106405


namespace paint_cost_is_624_rs_l106_106292

-- Given conditions:
-- Length of floor is 21.633307652783934 meters.
-- Length is 200% more than the breadth (i.e., length = 3 * breadth).
-- Cost to paint the floor is Rs. 4 per square meter.

noncomputable def length : ℝ := 21.633307652783934
noncomputable def cost_per_sq_meter : ℝ := 4
noncomputable def breadth : ℝ := length / 3
noncomputable def area : ℝ := length * breadth
noncomputable def total_cost : ℝ := area * cost_per_sq_meter

theorem paint_cost_is_624_rs : total_cost = 624 := by
  sorry

end paint_cost_is_624_rs_l106_106292


namespace game_is_fair_l106_106344

-- Define the conditions of the game and the probabilities
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the probabilities
def probability_xiao_ming_wins : ℚ := 1 / 2
def probability_xiao_liang_wins : ℚ := 1 / 2

-- Prove that the game is fair
theorem game_is_fair : probability_xiao_ming_wins = probability_xiao_liang_wins :=
by
  rw [probability_xiao_ming_wins, probability_xiao_liang_wins]
  sorry

end game_is_fair_l106_106344


namespace x_minus_y_options_l106_106316

theorem x_minus_y_options (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) :
  (x - y ≠ 2013) ∧ (x - y ≠ 2014) ∧ (x - y ≠ 2015) ∧ (x - y ≠ 2016) := 
sorry

end x_minus_y_options_l106_106316


namespace Mary_put_crayons_l106_106313

def initial_crayons : ℕ := 7
def final_crayons : ℕ := 10
def added_crayons (i f : ℕ) : ℕ := f - i

theorem Mary_put_crayons :
  added_crayons initial_crayons final_crayons = 3 := 
by
  sorry

end Mary_put_crayons_l106_106313


namespace area_AEF_is_20_l106_106219

structure Rectangle :=
(AB CD : ℝ)

structure LineSegment :=
(length : ℝ)

structure Triangle :=
(area : ℝ)

def BE : LineSegment := { length := 5 }
def EC : LineSegment := { length := 4 }
def CF : LineSegment := { length := 4 }
def FD : LineSegment := { length := 1 }

def AD : ℝ := BE.length + EC.length
def AB : ℝ := CF.length + FD.length

def area_Rectangle (rect: Rectangle) : ℝ :=
  rect.AB * rect.CD

def area_triangle (base height : ℝ) : ℝ :=
  0.5 * base * height

axiom find_area_AEF :
  ∀ (rect: Rectangle) (BE EC CF FD : LineSegment),
  rect.CD = AD ∧ rect.AB = AB ∧
  area_Rectangle rect = 45 ∧
  area_triangle EC.length CF.length = 8 ∧
  area_triangle AB BE.length = 12.5 ∧
  area_triangle AD FD.length = 4.5 →
  Triangle.area {area := 45 - 8 - 12.5 - 4.5}

theorem area_AEF_is_20 (rect: Rectangle) :
  ∀ (BE EC CF FD : LineSegment),
  AD = 9 → AB = 5 →
  Triangle.area {area := 20} :=
by intros; sorry

end area_AEF_is_20_l106_106219


namespace quadratic_two_distinct_roots_l106_106302

theorem quadratic_two_distinct_roots (a : ℝ) : 
  let b := -2 * a,
      c := a^2 - 4,
      Δ := b^2 - 4 * 1 * c in
  Δ > 0 :=
by
  let b := -2 * a
  let c := a^2 - 4
  let Δ := b^2 - 4 * 1 * c
  have h1 : Δ = 4 * a^2 - 4 * a^2 + 16 := by sorry
  have h2 : Δ = 16 := by sorry
  have h3 : 16 > 0 := by sorry
  exact h3

end quadratic_two_distinct_roots_l106_106302


namespace magnitude_of_angle_C_l106_106222

theorem magnitude_of_angle_C (A B C a b c : ℝ)
  (h₁ : sin A = 2 * sin B)
  (h₂ : a + b = Real.sqrt 3 * c)
  (h₃ : 0 < A ∧ A < π)
  (h₄ : 0 < B ∧ B < π)
  (h₅ : 0 < C ∧ C < π)
  (h₆ : a > 0)
  (h₇ : b > 0)
  (h₈ : c > 0)
  (h₉ : a * sin B = b * sin A)
  (h₁₀ : a^2 = b^2 + c^2 - 2 * b * c * cos A) :
  C = π / 3 :=
by
  sorry

end magnitude_of_angle_C_l106_106222


namespace eight_sided_die_expected_value_l106_106231

theorem eight_sided_die_expected_value :
  let p := λ i : ℕ, if i = 8 then 1 / 3 else 2 / 15 in
  (∑ i in (finset.range 8).map (finset.natCastEmbℚ), (p i) * i) = 6.4 :=
by
  let p := λ i : ℕ, if i = 8 then 1 / 3 else 2 / 15
  have finset_sum : (∑ i in (finset.range 8).map (finset.natCastEmbℚ), p i * i) = (2/15 * (1+2+3+4+5+6+7)) + (1/3 * 8)
  sorry

end eight_sided_die_expected_value_l106_106231


namespace fruit_total_l106_106774

theorem fruit_total (C O B N : ℕ) (h1 : C = 12) (h2 : O = 150) (h3 : B = 16) (h4 : N = 30) :
  C * O + B * N = 2280 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end fruit_total_l106_106774


namespace Rebecca_groups_of_eggs_l106_106724

theorem Rebecca_groups_of_eggs (eggs: ℕ) (group_size: ℕ) (h: eggs = 9) (h_group: group_size = 3) : 
  eggs / group_size = 3 :=
by
  rw [h, h_group]
  norm_num

end Rebecca_groups_of_eggs_l106_106724


namespace maurice_riding_times_l106_106700

variable (M : ℕ) -- The number of times Maurice had been horseback riding before visiting Matt
variable (h1 : 8) -- The times Maurice rode during his visit
variable (h2 : 8) -- The times Matt rode with Maurice
variable (h3 : 16) -- The additional times Matt rode
variable (h4 : 24 = 3 * M) -- The total number of times Matt rode during the two weeks is three times the number of times Maurice had ridden before his visit

theorem maurice_riding_times : M = 8 :=
by
  sorry

end maurice_riding_times_l106_106700


namespace contact_probability_l106_106545

-- Definition of the number of tourists in each group
def num_tourists_group1 : ℕ := 6
def num_tourists_group2 : ℕ := 7
def total_pairs : ℕ := num_tourists_group1 * num_tourists_group2

-- Definition of probability for no contact
def p : ℝ -- probability of contact
def prob_no_contact := (1 - p) ^ total_pairs

-- The theorem to be proven
theorem contact_probability : 1 - prob_no_contact = 1 - (1 - p) ^ total_pairs :=
by
  sorry

end contact_probability_l106_106545


namespace Cherry_weekly_earnings_l106_106872

theorem Cherry_weekly_earnings :
  let cost_3_5 := 2.50
  let cost_6_8 := 4.00
  let cost_9_12 := 6.00
  let cost_13_15 := 8.00
  let num_5kg := 4
  let num_8kg := 2
  let num_10kg := 3
  let num_14kg := 1
  let daily_earnings :=
    (num_5kg * cost_3_5) + (num_8kg * cost_6_8) + (num_10kg * cost_9_12) + (num_14kg * cost_13_15)
  let weekly_earnings := daily_earnings * 7
  weekly_earnings = 308 := by
  sorry

end Cherry_weekly_earnings_l106_106872


namespace smallest_perimeter_scalene_triangle_with_odd_primes_l106_106388

theorem smallest_perimeter_scalene_triangle_with_odd_primes :
  ∃ a b c : ℕ, prime a ∧ prime b ∧ prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ odd a ∧ odd b ∧ odd c ∧ 
  (a + 2 = b ∨ b + 2 = c ∨ a + 2 = c ∨ b + 4 = c ∨ a + 4 = c) ∧ 
  a + b + c = 23 ∧ prime (a + b + c) :=
begin
  sorry,
end

end smallest_perimeter_scalene_triangle_with_odd_primes_l106_106388


namespace a_is_4_l106_106382

noncomputable def find_a (a : ℝ) : Prop :=
  let slope := (7 - 5) / (a - 3) in
  slope = 2

theorem a_is_4 : ∃ a : ℝ, find_a a ∧ a = 4 :=
by
  sorry

end a_is_4_l106_106382


namespace problem_statement_l106_106631

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 5) : 
  -a - m * c * d - b = -5 ∨ -a - m * c * d - b = 5 := 
  sorry

end problem_statement_l106_106631


namespace solve_equation_l106_106802

noncomputable def log_sqrt_3 (x: ℝ) : ℝ := real.log x / real.log (real.sqrt 3)

theorem solve_equation (x: ℝ) 
    (hx_pos : x > 0)
    (hcond : log_sqrt_3 3 - real.log 9 / real.log x >= 0) : 
    log_sqrt_3 x * real.sqrt (log_sqrt_3 3 - real.log 9 / real.log x) + 4 = 0 ↔ x = 1 / 3 := by
  sorry

end solve_equation_l106_106802


namespace ratio_in_range_l106_106613

theorem ratio_in_range {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end ratio_in_range_l106_106613


namespace fraction_of_area_outside_l106_106058

-- The conditions as definitions in Lean 4
def is_isosceles_triangle (A B C : Type) [triangle A B C] : Prop := 
  ∠BAC = 80 ∧ ∠ABC = 50 ∧ ∠BCA = 50

def circle_inscribed (A B C O : Type) [triangle A B C] [circle O (radius : ℝ)] : Prop := 
  tangent_at_sides {D : Type | tangent_point O A B D} ∧
  tangent_at_sides {E : Type | tangent_point O B C E} ∧
  tangent_at_sides {F : Type | tangent_point O C A F}

noncomputable def area_outside_fraction (A B C O : Type) [triangle A B C] [circle O (radius : ℝ)] : ℝ := 
  let s := (2 * side_length A B + side_length A C) / 2 in
  let Aₜ := 1/2 * (side_length A B) * (side_length A C) * (angle_sin (∠ BAC)) in
  let Aᵢ := π * (radius * radius) in
  (Aₜ - Aᵢ) / Aₜ

theorem fraction_of_area_outside (A B C O : Type) [triangle A B C] [circle O (radius : ℝ)]
    (h_iso : is_isosceles_triangle A B C)
    (h_circ : circle_inscribed A B C O) :
  area_outside_fraction A B C O = 3 / 4 := 
sorry

end fraction_of_area_outside_l106_106058


namespace prove_solution_l106_106733

noncomputable def problem_statement : Prop := ∀ x : ℝ, (16 : ℝ)^(2 * x - 3) = (4 : ℝ)^(3 - x) → x = 9 / 5

theorem prove_solution : problem_statement :=
by
  intro x h
  -- The proof would go here
  sorry

end prove_solution_l106_106733


namespace negation_of_exists_cube_pos_l106_106761

theorem negation_of_exists_cube_pos :
  (¬ (∃ x : ℝ, x^3 > 0)) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by
  sorry

end negation_of_exists_cube_pos_l106_106761


namespace marked_price_l106_106744

theorem marked_price (cost_price selling_discount profit_per_item : ℝ) 
  (h1 : cost_price = 10) 
  (h2 : selling_discount = 0.20) 
  (h3 : profit_per_item = 2) : 
  let marked_price := (12 : ℝ) / (1 - selling_discount) in 
  marked_price = 15 := by 
  sorry

end marked_price_l106_106744


namespace Kristy_baked_cookies_l106_106668

theorem Kristy_baked_cookies 
  (ate_by_Kristy : ℕ) (given_to_brother : ℕ) 
  (taken_by_first_friend : ℕ) (taken_by_second_friend : ℕ)
  (taken_by_third_friend : ℕ) (cookies_left : ℕ) 
  (h_K : ate_by_Kristy = 2) (h_B : given_to_brother = 1) 
  (h_F1 : taken_by_first_friend = 3) (h_F2 : taken_by_second_friend = 5)
  (h_F3 : taken_by_third_friend = 5) (h_L : cookies_left = 6) :
  ate_by_Kristy + given_to_brother 
  + taken_by_first_friend + taken_by_second_friend 
  + taken_by_third_friend + cookies_left = 22 := 
by
  sorry

end Kristy_baked_cookies_l106_106668


namespace true_statement_exists_l106_106617

noncomputable def vector_a : ℝ × ℝ := (1, real.sqrt 3)
noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := (real.cos θ, real.sin θ)

theorem true_statement_exists (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ real.pi) :
  (|vector_a + vector_b θ| = |vector_a - vector_b θ| → θ = 5 * real.pi / 6) :=
sorry

end true_statement_exists_l106_106617


namespace largest_integer_same_cost_l106_106323

def cost_base_10 (n : ℕ) : ℕ :=
  (n.digits 10).sum

def cost_base_2 (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem largest_integer_same_cost : ∃ n < 1000, 
  cost_base_10 n = cost_base_2 n ∧
  ∀ m < 1000, cost_base_10 m = cost_base_2 m → n ≥ m :=
sorry

end largest_integer_same_cost_l106_106323


namespace club_truncator_season_with_more_wins_than_losses_l106_106427

theorem club_truncator_season_with_more_wins_than_losses :
  let p := 31
  let q := 81
  let prob_win = (1 / 3 : ℚ)
  let prob_lose = (1 / 3 : ℚ)
  let prob_tie = (1 / 3 : ℚ)
  (∃ p q : ℕ, p = 31 ∧ q = 81 ∧ nat.coprime p q ∧ (1 - (19 / 81)) / 2 = (p / q)) ∧ p + q = 112 :=
by
  sorry

end club_truncator_season_with_more_wins_than_losses_l106_106427


namespace anna_has_9_cupcakes_left_l106_106858

def cupcakes_left (initial : ℕ) (given_away_fraction : ℚ) (eaten : ℕ) : ℕ :=
  let remaining = initial * (1 - given_away_fraction)
  remaining - eaten

theorem anna_has_9_cupcakes_left :
  cupcakes_left 60 (4/5 : ℚ) 3 = 9 := by
  sorry

end anna_has_9_cupcakes_left_l106_106858


namespace variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106023

-- Definitions and conditions for Problem a) and b)
noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def variance (a b c : ℝ) : ℝ :=
  let mean_squares := (a^2 + b^2 + c^2) / 3
  let mean := (a + b + c) / 3
  mean_squares - mean^2

-- Prove that the variance is less than 2 given the conditions for Problem a)
theorem variance_triangle_less_than_2 {a b : ℝ} (h : is_right_triangle a b 3) : 
  variance a b 3 < 2 := sorry

-- Define the minimum standard deviation and legs for Problem b)
noncomputable def std_dev_of_legs (a b : ℝ) : ℝ :=
  let sides_squared_mean := (a^2 + b^2) / 2
  let mean_legs := (a + b) / 2
  real.sqrt (sides_squared_mean - mean_legs^2)

-- Prove that the minimum standard deviation of the legs of a right triangle with hypotenuse 3 is sqrt(2) - 1
theorem min_std_dev_of_triangle_legs {a b : ℝ} (h : is_right_triangle a b 3) :
  a = b → std_dev_of_legs a b = real.sqrt(2) - 1 :=
begin
  sorry
end

-- Prove the lengths of the legs for minimum standard deviation
theorem lengths_of_min_std_dev (a b : ℝ) (h : is_right_triangle a b 3) :
  a = b → a = 3 * real.sqrt(2) / 2 ∧ b = 3 * real.sqrt(2) / 2 :=
begin
  sorry
end

end variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106023


namespace quadratic_real_roots_iff_l106_106153

-- Define the statement of the problem in Lean
theorem quadratic_real_roots_iff (m : ℝ) :
  (∃ x : ℂ, m * x^2 + 2 * x - 1 = 0) ↔ (m ≥ -1 ∧ m ≠ 0) := 
by
  sorry

end quadratic_real_roots_iff_l106_106153


namespace polynomial_roots_relationship_l106_106897

-- Defining the polynomials and their roots
variables (a b c : ℝ) (α β γ : ℝ) (h k : ℝ)

-- Conditions on the roots and the coefficients
def roots_condition : Prop :=
  -a = α + β + γ ∧ b = α * β + β * γ + γ * α ∧ -c = α * β * γ

def transformed_polynomial_roots_condition : Prop :=
  -a^3 = α^3 + β^3 + γ^3

-- Conclusion -- find the relationship between a, b, c such that c = ab and b <= 0
theorem polynomial_roots_relationship (h_eq : h = a) (k_eq : k = b) (h_ineq : k ≤ 0) :
  roots_condition a b c α β γ → 
  transformed_polynomial_roots_condition a b c α β γ → 
  c = a * b ∧ b ≤ 0 := 
by 
  sorry -- proof goes here

end polynomial_roots_relationship_l106_106897


namespace cost_per_book_l106_106728

theorem cost_per_book (initial_amount : ℤ) (remaining_amount : ℤ) (num_books : ℤ) (cost_per_book : ℤ) :
  initial_amount = 79 →
  remaining_amount = 16 →
  num_books = 9 →
  cost_per_book = (initial_amount - remaining_amount) / num_books →
  cost_per_book = 7 := 
by
  sorry

end cost_per_book_l106_106728


namespace crumble_topping_correct_amount_l106_106259

noncomputable def crumble_topping_total_mass (flour butter sugar : ℕ) (factor : ℚ) : ℚ :=
  factor * (flour + butter + sugar) / 1000  -- convert grams to kilograms

theorem crumble_topping_correct_amount {flour butter sugar : ℕ} (factor : ℚ) (h_flour : flour = 100) (h_butter : butter = 50) (h_sugar : sugar = 50) (h_factor : factor = 2.5) :
  crumble_topping_total_mass flour butter sugar factor = 0.5 :=
by
  sorry

end crumble_topping_correct_amount_l106_106259


namespace sin_double_angle_l106_106145

theorem sin_double_angle (α : ℝ) (h : sin α - 3 * cos α = 0) : sin (2 * α) = 3 / 5 :=
sorry

end sin_double_angle_l106_106145


namespace binomial_coeff_18_10_l106_106482

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106482


namespace primes_divide_binom_l106_106081

theorem primes_divide_binom (m : ℕ) (n : ℕ) :
  m ≥ 2 ∧ (m / 3 ≤ n ∧ n ≤ m / 2) →
  (∀ k : ℕ, n = k * (m-2*n) → k = 1) ↔ prime m :=
sorry

end primes_divide_binom_l106_106081


namespace problem_statement_l106_106496

-- Define operations "※" and "#"
def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

-- Define the proof statement
theorem problem_statement : hash 4 (star (star 6 8) (hash 3 5)) = 103 := by
  sorry

end problem_statement_l106_106496


namespace sequence_general_formula_l106_106930

-- Definitions according to conditions in a)
def seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n + 1

def S (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  n * seq (n + 1) - 3 * n^2 - 4 * n

-- The proof goal
theorem sequence_general_formula (n : ℕ) (h : 0 < n) :
  seq n = 2 * n + 1 :=
by
  sorry

end sequence_general_formula_l106_106930


namespace kim_initial_classes_l106_106664

-- Necessary definitions for the problem
def hours_per_class := 2
def total_hours_after_dropping := 6
def classes_after_dropping := total_hours_after_dropping / hours_per_class
def initial_classes := classes_after_dropping + 1

theorem kim_initial_classes : initial_classes = 4 :=
by
  -- Proof will be derived here
  sorry

end kim_initial_classes_l106_106664


namespace find_fraction_spent_on_domestic_needs_l106_106013

-- Definitions based on problem conditions
def income : ℝ := 200
def provident_fund (I : ℝ) : ℝ := I * (1 / 16)
def remaining_after_pf (I : ℝ) : ℝ := I - provident_fund I
def insurance_premium (remaining : ℝ) : ℝ := remaining * (1 / 15)
def remaining_after_insurance (remaining : ℝ) : ℝ := remaining - insurance_premium remaining
def bank_deposit : ℝ := 50

-- The fraction F to be found
def fraction_spent_on_domestic_needs (remaining : ℝ) (F : ℝ) : Prop :=
  remaining - remaining * F = bank_deposit

-- The theorem to prove
theorem find_fraction_spent_on_domestic_needs :
  ∃ F : ℝ, fraction_spent_on_domestic_needs (remaining_after_insurance (remaining_after_pf income)) F ∧ F = 5 / 7 :=
by
  sorry

end find_fraction_spent_on_domestic_needs_l106_106013


namespace unique_rational_point_on_sphere_l106_106504
open Real

-- Define a predicate for a rational point in 3D space
def is_rational_point (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y z : ℚ), (p = (x, y, z))

-- Define the specific sphere equation
def on_sphere (p : ℝ × ℝ × ℝ) : Prop :=
  (p.1 - Real.sqrt 2) ^ 2 + p.2 ^ 2 + p.3 ^ 2 = 2

theorem unique_rational_point_on_sphere :
  ∃! (p : ℝ × ℝ × ℝ), is_rational_point p ∧ on_sphere p :=
begin
  sorry
end

end unique_rational_point_on_sphere_l106_106504


namespace initial_distance_l106_106229

/-- Suppose Jack walks at a speed of 3 feet per second toward Christina,
    Christina walks at a speed of 3 feet per second toward Jack, and their dog Lindy
    runs at a speed of 10 feet per second back and forth between Jack and Christina.
    Given that Lindy travels a total of 400 feet when they meet, prove that the initial
    distance between Jack and Christina is 240 feet. -/
theorem initial_distance (initial_distance_jack_christina : ℝ)
  (jack_speed : ℝ := 3)
  (christina_speed : ℝ := 3)
  (lindy_speed : ℝ := 10)
  (lindy_total_distance : ℝ := 400):
  initial_distance_jack_christina = 240 :=
sorry

end initial_distance_l106_106229


namespace find_pairs_satisfying_system_l106_106516

theorem find_pairs_satisfying_system (x y : ℝ) :
  x * real.sqrt (1 - y^2) = 1 / 4 * (real.sqrt 3 + 1) ∧
  y * real.sqrt (1 - x^2) = 1 / 4 * (real.sqrt 3 - 1) →
  (x = (real.sqrt 6 + real.sqrt 2) / 4 ∧ y = real.sqrt 2 / 2) ∨
  (x = real.sqrt 2 / 2 ∧ y = (real.sqrt 6 - real.sqrt 2) / 4) :=
sorry

end find_pairs_satisfying_system_l106_106516


namespace sector_area_approx_l106_106735

noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * π * r^2

theorem sector_area_approx :
  sector_area 12 42 ≈ 52.8 :=
by
  have h : sector_area 12 42 = (42 / 360) * π * 144 :=
    by simp [sector_area, Real.pi]
  rw [h] 
  have h2 : (42 / 360) * 144 = 16.8 :=
    by norm_num
  rw [←mul_assoc, h2, mul_comm (16.8 : ℝ) π]
  norm_num
  sorry

end sector_area_approx_l106_106735


namespace binom_18_10_eq_43758_l106_106463

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106463


namespace intersection_A_B_l106_106237

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l106_106237


namespace no_solution_l106_106087

-- Define the problem conditions: two numbers with the same digits rearranged
def has_same_digits (n1 n2 : ℕ) : Prop :=
  ∀ d : ℕ, nat.count_digit d n1 = nat.count_digit d n2

-- Define the puzzle's equation
def puzzle_equation (APPLE SPANIEL : ℕ) : Prop :=
  APPLE - SPANIEL = 2018 * 2019

-- Main theorem: the puzzle has no solution
theorem no_solution (APPLE SPANIEL : ℕ) (h : has_same_digits APPLE SPANIEL) : ¬ puzzle_equation APPLE SPANIEL :=
sorry

end no_solution_l106_106087


namespace find_angle_and_perimeter_l106_106932

theorem find_angle_and_perimeter (a b c A B C : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
(hc_eq : 2 * cos C * (a * cos B + b * cos A) = c)
(hC_range : 0 < C ∧ C < π)
(hc_val : c = sqrt 7) (hab : a * b = 6)
: (C = π / 3) ∧ (a + b + c = 5 + sqrt 7) :=
by
  sorry

end find_angle_and_perimeter_l106_106932


namespace functional_expression_selling_price_for_profit_l106_106696

-- Define the initial conditions
def cost_price : ℚ := 8
def initial_selling_price : ℚ := 10
def initial_sales_volume : ℚ := 200
def sales_decrement_per_yuan_increase : ℚ := 20

-- Functional expression between y (items) and x (yuan)
theorem functional_expression (x : ℚ) : 
  (200 - 20 * (x - 10) = -20 * x + 400) :=
sorry

-- Determine the selling price to achieve a daily profit of 640 yuan
theorem selling_price_for_profit (x : ℚ) (h1 : 8 ≤ x) (h2 : x ≤ 15) : 
  ((x - 8) * (400 - 20 * x) = 640) → (x = 12) :=
sorry

end functional_expression_selling_price_for_profit_l106_106696


namespace length_of_AB_l106_106998

-- Definitions of the circles and the necessary geometric properties and conditions
def circle1 (x y: ℝ) : Prop := x^2 + y^2 = 5
def circle2 (x y: ℝ) (m: ℝ) : Prop := (x - m)^2 + y^2 = 20
noncomputable def AB_length (m: ℝ) (x y: ℝ) : ℝ := 4

-- Theorem to be proven
theorem length_of_AB (m x y: ℝ) (h1: (circle1 x y)) (h2: (circle2 x y m)) (tangents_perpendicular: ∃A B, A ≠ B 
  ∧ (circle1 A.1 A.2) ∧ (circle2 B.1 B.2 m) ∧ (tangents_perpendicular_at_point A B)): 
  AB_length m x y = 4 := 
sorry

end length_of_AB_l106_106998


namespace pos_integers_divisible_by_2_3_5_7_less_than_300_l106_106981

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end pos_integers_divisible_by_2_3_5_7_less_than_300_l106_106981


namespace sum_fractions_l106_106922

def a : ℝ := sorry -- a is a positive constant
def f (x : ℝ) : ℝ := (a^x) / (a^x + real.sqrt a)

theorem sum_fractions :
  ∑ k in finset.range 1000, f ((k + 1) / 1001) = 500 :=
sorry

end sum_fractions_l106_106922


namespace monthly_installment_amount_l106_106284

variable (cashPrice : ℕ) (deposit : ℕ) (monthlyInstallments : ℕ) (savingsIfCash : ℕ)

-- Defining the conditions
def conditions := 
  cashPrice = 8000 ∧ 
  deposit = 3000 ∧ 
  monthlyInstallments = 30 ∧ 
  savingsIfCash = 4000

-- Proving the amount of each monthly installment
theorem monthly_installment_amount (h : conditions cashPrice deposit monthlyInstallments savingsIfCash) : 
  (12000 - deposit) / monthlyInstallments = 300 :=
sorry

end monthly_installment_amount_l106_106284


namespace binomial_coeff_18_10_l106_106481

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106481


namespace cubic_root_conditions_l106_106878

-- Define the cubic polynomial
def cubic (a b : ℝ) (x : ℝ) : ℝ := x^3 + a * x + b

-- Define a predicate for the cubic equation having exactly one real root
def has_one_real_root (a b : ℝ) : Prop :=
  ∀ y : ℝ, cubic a b y = 0 → ∃! x : ℝ, cubic a b x = 0

-- Theorem statement
theorem cubic_root_conditions (a b : ℝ) :
  (a = -3 ∧ b = -3) ∨ (a = -3 ∧ b > 2) ∨ (a = 0 ∧ b = 2) → has_one_real_root a b :=
sorry

end cubic_root_conditions_l106_106878


namespace amy_problems_per_hour_l106_106553

theorem amy_problems_per_hour (math_problems : ℕ) (spelling_problems : ℕ) (total_hours : ℕ) :
  math_problems = 18 ∧ spelling_problems = 6 ∧ total_hours = 6 →
  (math_problems + spelling_problems) / total_hours = 4 :=
by 
  intro h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  calc
    (18 + 6) / 6 = 24 / 6 : by simp [h1, h3]
    ... = 4            : by norm_num

end amy_problems_per_hour_l106_106553


namespace root_in_interval_l106_106755

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log x

theorem root_in_interval : ∃ x ∈ set.Icc 2 3, f x = 0 :=
sorry

end root_in_interval_l106_106755


namespace max_abs_value_l106_106909

theorem max_abs_value (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : |x - 2 * y + 1| ≤ 5 :=
by
  sorry

end max_abs_value_l106_106909


namespace right_rect_prism_x_val_l106_106019

theorem right_rect_prism_x_val 
  (x : ℝ) 
  (a : ℝ := log 5 x)
  (b : ℝ := log 8 x)
  (c : ℝ := log 10 x) :
  2 * (a * b + b * c + c * a) = a * b * c → 
  x = 100000000 := 
by 
  -- sorry is used as placeholder for the proof
  sorry

end right_rect_prism_x_val_l106_106019


namespace find_p_l106_106296

theorem find_p 
  (p q : ℝ)
  (h3 : p + q = 2)
  (h_gt : p > 0 ∧ q > 0)
  (hq_eq : (11.choose 2) * p^9 * q^2 = (11.choose 3) * p^8 * q^3) :
  p = 3 / 2 :=
by
  sorry

end find_p_l106_106296


namespace problem_solution_l106_106294

variables {x y a : ℝ}
def H : Prop := ∃ x y, x^2 / 9 - y^2 / 16 = 1
def E : Prop := ∃ x y, x^2 / 35 + y^2 / 10 = 1
def condition2 (a : ℝ) (P F1 F2 : ℝ × ℝ) : Prop :=
  (a > 0) ∧ (∥P - F1∥ + ∥P - F2∥ = a + 9/a)
def condition3 (P A B : ℝ × ℝ) : Prop :=
  ∥P - A∥ - ∥P - B∥ = 6
def condition4 (P A B : ℝ × ℝ) : Prop :=
  (∥P - A∥ = 10 - ∥P - B∥) ∧ (∥A - B∥ = 8)
def condition5 (P F : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ∥P - F∥ = ∥P - λ x, l x∥

theorem problem_solution :
  (∃ (P F1 F2 : ℝ × ℝ), condition2 a P F1 F2) ∧
  (∃ (P A B : ℝ × ℝ), condition3 P A B) ∧
  (∃ (P A B : ℝ × ℝ), condition4 P A B) ∧
  (∃ (P F : ℝ × ℝ) (l : ℝ → ℝ), condition5 P F l) →
  count_correct_statements [H, E, condition2 a _ _, condition3 _ _ _, condition4 _ _ _, condition5 _ _ _] = 2 :=
begin
  sorry
end

end problem_solution_l106_106294


namespace identify_n_within_bound_l106_106583

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem identify_n_within_bound (n : ℕ) (h₁ : n > 0) (h₂ : n ≤ fibonacci 10)
  (strategy : ∀ (i : ℕ), i < 10 → Prop := λ i, n ≤ fibonacci (10 - i)) :
  ∃ m, n = m ∧ m ≤ 144 :=
by
  sorry

end identify_n_within_bound_l106_106583


namespace distance_between_towns_l106_106263

theorem distance_between_towns : ∀ (map_distance : ℝ) (scale_factor : ℝ), 
  map_distance = 45 → scale_factor = 10 → (map_distance * scale_factor = 450) := 
by
  intros map_distance scale_factor
  intro h_map_distance
  intro h_scale_factor
  rw [h_map_distance, h_scale_factor]
  norm_num
  sorry

end distance_between_towns_l106_106263


namespace decreasing_condition_l106_106595

noncomputable def f (a x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem decreasing_condition (a : ℝ) :
  (∀ x > 1, (Real.log x - 1) / (Real.log x)^2 + a ≤ 0) → a ≤ -1/4 := by
  sorry

end decreasing_condition_l106_106595


namespace sphere_contains_one_rational_point_l106_106501

def is_rational (x : ℚ) : Prop := true  -- Checks if x is a rational number

def rational_point (p : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (x y z : ℚ),
  p = (x : ℝ) • ![1, 0, 0] + (y : ℝ) • ![0, 1, 0] + (z : ℝ) • ![0, 0, 1]

def sphere_eq (x y z : ℝ) : ℝ := (x - real.sqrt 2)^2 + y^2 + z^2 - 2

theorem sphere_contains_one_rational_point :
  (∃ p : EuclideanSpace ℝ (Fin 3), rational_point p ∧ sphere_eq p 0) ∧ 
  (∀ p₁ p₂ : EuclideanSpace ℝ (Fin 3), rational_point p₁ → rational_point p₂ → sphere_eq p₁ 0 → sphere_eq p₂ 0 → p₁ = p₂) :=
begin
  sorry
end

end sphere_contains_one_rational_point_l106_106501


namespace final_price_for_tiffany_l106_106777

noncomputable def calculate_final_price (n : ℕ) (c : ℝ) (d : ℝ) (s : ℝ) : ℝ :=
  let total_cost := n * c
  let discount := d * total_cost
  let discounted_price := total_cost - discount
  let sales_tax := s * discounted_price
  let final_price := discounted_price + sales_tax
  final_price

theorem final_price_for_tiffany :
  calculate_final_price 9 4.50 0.20 0.07 = 34.67 :=
by
  sorry

end final_price_for_tiffany_l106_106777


namespace cube_root_value_sum_l106_106498

theorem cube_root_value_sum :
  ∃ (a b c : ℕ), (c ≠ 0) ∧ (a > 0) ∧ (b > 0) ∧ root_of_polynomial = (c ≠ 0) → 
  ((a = 81) ∧ (b = 9) ∧ (c = 27) →
  a + b + c = 117) :
  sorry

end cube_root_value_sum_l106_106498


namespace largest_angle_in_triangle_l106_106223

theorem largest_angle_in_triangle (a b c : ℝ) (h1 : a + 3 * b + 3 * c = a ^ 2) (h2 : a + 3 * b - 3 * c = -4) 
  (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : a + b > c) (h7 : a + c > b) (h8 : b + c > a) : 
  ∃ C : ℝ, C = 120 ∧ (by exact sorry) := sorry

end largest_angle_in_triangle_l106_106223


namespace find_analytical_expression_l106_106155

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_analytical_expression : 
  (∀ x : ℝ, f(2 * x) = 4 * x^2 + 3 * x) → (f(x) = x^2 + (3 / 2) * x) :=
by
  intros h
  have : ∀ t : ℝ, f t = (t^2 + (3 / 2) * t) := sorry
  exact this x

end find_analytical_expression_l106_106155


namespace constant_term_expansion_is_70_l106_106890

theorem constant_term_expansion_is_70 :
  let expr := (x - 2 + 1/x)^4 in
  ∃ c : ℝ, expand_binomial expr = c ∧ c = 70 := by
  sorry

end constant_term_expansion_is_70_l106_106890


namespace lowest_possible_number_of_students_l106_106369

theorem lowest_possible_number_of_students :
  Nat.lcm 18 24 = 72 :=
by
  sorry

end lowest_possible_number_of_students_l106_106369


namespace cos_transform_l106_106116

theorem cos_transform : 
  ∀ α : ℝ, sin (π + α) = 1 / 3 → cos ((3 * π / 2) - α) = -1 / 3 :=
by
  intro α h
  sorry

end cos_transform_l106_106116


namespace binom_18_10_l106_106488

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106488


namespace solve_quadratic_eqn_l106_106276

theorem solve_quadratic_eqn : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_quadratic_eqn_l106_106276


namespace fraction_equals_repeating_decimal_l106_106526

noncomputable def repeating_decimal_fraction : ℚ :=
  let a : ℚ := 46 / 100
  let r : ℚ := 1 / 100
  (a / (1 - r))

theorem fraction_equals_repeating_decimal :
  repeating_decimal_fraction = 46 / 99 :=
by
  sorry

end fraction_equals_repeating_decimal_l106_106526


namespace lana_average_speed_l106_106669

theorem lana_average_speed (initial_reading : ℕ) (final_reading : ℕ) (time_first_day : ℕ) (time_second_day : ℕ) :
  initial_reading = 1991 → 
  final_reading = 2332 → 
  time_first_day = 5 → 
  time_second_day = 7 → 
  (final_reading - initial_reading) / (time_first_day + time_second_day : ℝ) = 28.4 :=
by
  intros h_init h_final h_first h_second
  rw [h_init, h_final, h_first, h_second]
  norm_num
  sorry

end lana_average_speed_l106_106669


namespace area_of_quadrilateral_ABCD_l106_106210

-- Define the quadrilateral ABCD and its properties
structure Quadrilateral (A B C D : Type) :=
(side_a : ℝ)
(side_b : ℝ)
(side_c : ℝ)
(angle_B : ℝ)
(angle_C : ℝ)
(area : ℝ)

-- Given conditions of quadrilateral ABCD
def quadABCD : Quadrilateral Point := {
  side_a := 4,
  side_b := 6,
  side_c := 7,
  angle_B := 135,
  angle_C := 135,
  area := 16.5 * Real.sqrt 2
}

-- Theorem to prove that the area of quadrilateral ABCD is 16.5*sqrt(2)
theorem area_of_quadrilateral_ABCD : quadABCD.area = 16.5 * Real.sqrt 2 :=
by sorry

end area_of_quadrilateral_ABCD_l106_106210


namespace variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106024

-- Definitions and conditions for Problem a) and b)
noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def variance (a b c : ℝ) : ℝ :=
  let mean_squares := (a^2 + b^2 + c^2) / 3
  let mean := (a + b + c) / 3
  mean_squares - mean^2

-- Prove that the variance is less than 2 given the conditions for Problem a)
theorem variance_triangle_less_than_2 {a b : ℝ} (h : is_right_triangle a b 3) : 
  variance a b 3 < 2 := sorry

-- Define the minimum standard deviation and legs for Problem b)
noncomputable def std_dev_of_legs (a b : ℝ) : ℝ :=
  let sides_squared_mean := (a^2 + b^2) / 2
  let mean_legs := (a + b) / 2
  real.sqrt (sides_squared_mean - mean_legs^2)

-- Prove that the minimum standard deviation of the legs of a right triangle with hypotenuse 3 is sqrt(2) - 1
theorem min_std_dev_of_triangle_legs {a b : ℝ} (h : is_right_triangle a b 3) :
  a = b → std_dev_of_legs a b = real.sqrt(2) - 1 :=
begin
  sorry
end

-- Prove the lengths of the legs for minimum standard deviation
theorem lengths_of_min_std_dev (a b : ℝ) (h : is_right_triangle a b 3) :
  a = b → a = 3 * real.sqrt(2) / 2 ∧ b = 3 * real.sqrt(2) / 2 :=
begin
  sorry
end

end variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106024


namespace find_valid_permutations_l106_106517

def no_adjacent_sum_div_by (p : List Nat) (d : Nat) : Prop :=
  ∀ i, i < p.length - 1 → (p[i] + p[i+1]) % d ≠ 0

def valid_permutation (p : List Nat) : Prop := 
  list.perm p [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ 
  no_adjacent_sum_div_by p 7 ∧ 
  no_adjacent_sum_div_by p 13

theorem find_valid_permutations :
  (Finset.filter valid_permutation (Finset.univ : Finset (List Nat))).card = 74880 := 
sorry

end find_valid_permutations_l106_106517


namespace domain_of_f_l106_106787

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = x} = {x : ℝ | x ≠ 6} := by
  sorry

end domain_of_f_l106_106787


namespace total_cost_4kg_mangos_3kg_rice_5kg_flour_l106_106355

def cost_per_kg_mangos (M : ℝ) (R : ℝ) := (10 * M = 24 * R)
def cost_per_kg_flour_equals_rice (F : ℝ) (R : ℝ) := (6 * F = 2 * R)
def cost_of_flour (F : ℝ) := (F = 24)

theorem total_cost_4kg_mangos_3kg_rice_5kg_flour 
  (M R F : ℝ) 
  (h1 : cost_per_kg_mangos M R) 
  (h2 : cost_per_kg_flour_equals_rice F R) 
  (h3 : cost_of_flour F) : 
  4 * M + 3 * R + 5 * F = 1027.2 :=
by {
  sorry
}

end total_cost_4kg_mangos_3kg_rice_5kg_flour_l106_106355


namespace max_value_f_max_value_at_maximum_value_of_f_l106_106945

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2) / (x^2)

theorem max_value_f : ∀ x > 0, ∃ c : ℝ, f x = (Real.log x + c) / x^2 ∧ f 1 = 2 :=
by {
  sorry
}

theorem max_value_at : f (Real.exp (-3/2)) = (Real.exp 3) / 2 :=
by {
  exact rfl
}

theorem maximum_value_of_f {x : ℝ} (hx : 0 < x) :
  ∃ y, y = (Real.exp 3) / 2 ∧ ∀ z, z > 0 → f z ≤ y :=
by {
  use f (Real.exp (-3 / 2)),
  split,
  { rw max_value_at },
  { intro z,
    rw ←max_value_at,
    sorry }
}

end max_value_f_max_value_at_maximum_value_of_f_l106_106945


namespace attendance_mean_and_median_corrected_l106_106759

-- Define the initial list of attendances
def initial_attendance := [20, 35, 24, 30, 25]

-- Define the corrected list of attendances
def corrected_attendance := [25, 35, 30, 30, 25]

-- Define a function to calculate the mean of a list of integers
def mean (lst : List ℕ) : ℕ :=
  lst.sum / lst.length

-- Define a function to calculate the median of a list of integers
def median (lst : List ℕ) : ℕ :=
  let sorted := lst.qsort (· < ·)
  sorted.get (sorted.length / 2) sorry

-- Define the statement to prove the changes in mean and median
theorem attendance_mean_and_median_corrected :
  mean corrected_attendance = mean initial_attendance + 2.2 ∧
  median corrected_attendance = median initial_attendance + 5 := 
by {
  sorry
}

end attendance_mean_and_median_corrected_l106_106759


namespace variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106020

-- Definitions and conditions for Problem a) and b)
noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def variance (a b c : ℝ) : ℝ :=
  let mean_squares := (a^2 + b^2 + c^2) / 3
  let mean := (a + b + c) / 3
  mean_squares - mean^2

-- Prove that the variance is less than 2 given the conditions for Problem a)
theorem variance_triangle_less_than_2 {a b : ℝ} (h : is_right_triangle a b 3) : 
  variance a b 3 < 2 := sorry

-- Define the minimum standard deviation and legs for Problem b)
noncomputable def std_dev_of_legs (a b : ℝ) : ℝ :=
  let sides_squared_mean := (a^2 + b^2) / 2
  let mean_legs := (a + b) / 2
  real.sqrt (sides_squared_mean - mean_legs^2)

-- Prove that the minimum standard deviation of the legs of a right triangle with hypotenuse 3 is sqrt(2) - 1
theorem min_std_dev_of_triangle_legs {a b : ℝ} (h : is_right_triangle a b 3) :
  a = b → std_dev_of_legs a b = real.sqrt(2) - 1 :=
begin
  sorry
end

-- Prove the lengths of the legs for minimum standard deviation
theorem lengths_of_min_std_dev (a b : ℝ) (h : is_right_triangle a b 3) :
  a = b → a = 3 * real.sqrt(2) / 2 ∧ b = 3 * real.sqrt(2) / 2 :=
begin
  sorry
end

end variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106020


namespace sum_of_sides_l106_106198

-- Definitions: Given conditions
def ratio (a b c : ℕ) : Prop := 
a * 5 = b * 3 ∧ b * 7 = c * 5

-- Given that the longest side is 21 cm and the ratio of the sides is 3:5:7
def similar_triangle (x y : ℕ) : Prop :=
ratio x y 21

-- Proof statement: The sum of the lengths of the other two sides is 24 cm
theorem sum_of_sides (x y : ℕ) (h : similar_triangle x y) : x + y = 24 :=
sorry

end sum_of_sides_l106_106198


namespace cubic_two_common_points_x_axis_l106_106167

theorem cubic_two_common_points_x_axis (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ^ 3 - 3 * x1 + c = 0 ∧ x2 ^ 3 - 3 * x2 + c = 0 ∧
    (∀ x ∈ Ioo (-1 : ℝ) 1, x^3 - 3 * x + c > 0) ∧ 
    ((∀ x ≤ -1, x^3 - 3*x + c ≠ 0) ∨ (∀ x ≥ 1, x^3 - 3*x + c ≠ 0)))
  ↔ c = -2 ∨ c = 2 :=
by
  sorry

end cubic_two_common_points_x_axis_l106_106167


namespace intersection_points_l106_106125

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := -f x
noncomputable def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  let a := 2
  let b := 1
  10 * a + b = 21 :=
by
  sorry

end intersection_points_l106_106125


namespace probability_heads_greater_than_tails_in_4_tosses_l106_106637

-- Define the fair coin toss and the event to count the number of heads
def fairCoinToss : Type := { p : ℝ // p = 0.5 }

-- Define the binomial distribution
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the probability of getting more heads than tails in 4 tosses
def probabilityMoreHeadsThanTails : ℝ :=
  binomialProbability 4 3 0.5 + binomialProbability 4 4 0.5

-- The main theorem
theorem probability_heads_greater_than_tails_in_4_tosses :
  probabilityMoreHeadsThanTails = 5 / 16 :=
sorry

end probability_heads_greater_than_tails_in_4_tosses_l106_106637


namespace propositions_correct_l106_106928

variables {α β : Type} [Plane α] [Plane β]
variables (l m : Line) {p : Point}

def perpendicular (x y : Plane) := ∀ (p1 p2 : p), x ≠ y ∧ LineThrough x p2 ∋ p1
def contains (y : Plane) (x : Line) := ∀ p : Point, p ∈ x → p ∈ y
def parallel (x y : Plane) := ∀ p1 p2 : p, x ≠ y ∧ ∃ ! p₃ : p, x ∋ p₂

theorem propositions_correct
  (h1 : l ⊥ α)
  (h2 : m ∈ β) :
  (α ∥ β → l ⊥ m) ∧ (l ∥ m → α ⊥ β) :=
  by sorry

end propositions_correct_l106_106928


namespace point_p_inside_circle_l106_106948

-- Defining the conditions
def radius : ℝ := 4
def OP : ℝ := 3

-- Defining point P's position relative to the circle
theorem point_p_inside_circle (r : ℝ) (d : ℝ) (h₁ : r = 4) (h₂ : d = 3) : d < r :=
by {
  rw [h₁, h₂],
  show 3 < 4,
  norm_num,
}

end point_p_inside_circle_l106_106948


namespace matrix_inverse_exists_or_zero_l106_106528

theorem matrix_inverse_exists_or_zero {R : Type*} [CommRing R] :
  let A : Matrix (Fin 2) (Fin 2) R := ![![4, 8], ![-4, -8]]
  (det A = 0) → (∀ (A_inv : Matrix (Fin 2) (Fin 2) R), A_inv = ![![0, 0], ![0, 0]]) :=
by
  intro A h
  sorry

end matrix_inverse_exists_or_zero_l106_106528


namespace parabola_b_value_l106_106295

theorem parabola_b_value (a b c p : ℝ) (h1 : ∀ x, y = a * x^2 + b * x + c) (h2 : ∀ x, x = (p,p)) (h3 : ∀ x, (0, -p)) (h4 : p ≠ 0) :
    b = 4 :=
by
  sorry

end parabola_b_value_l106_106295


namespace binom_18_10_l106_106447

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106447


namespace binom_18_10_l106_106491

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106491


namespace pos_integers_divisible_by_2_3_5_7_less_than_300_l106_106983

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end pos_integers_divisible_by_2_3_5_7_less_than_300_l106_106983


namespace intersection_dot_product_zero_l106_106967

theorem intersection_dot_product_zero :
  ∀ (A B : ℝ × ℝ), 
  (∃ (x y : ℝ), (x - y + 2 = 0) ∧ ((x - 3)^2 + (y - 3)^2 = 4) ∧
    ∃ (C : ℝ × ℝ), C = (3, 3) ∧ C = (3, 3)) →
  (∃ (A B : ℝ × ℝ), ∃ (x1 y1 x2 y2 : ℝ), A = (x1, y1) ∧ B = (x2, y2) ∧
    x1 - y1 + 2 = 0 ∧ (x1 - 3)^2 + (y1 - 3)^2 = 4 ∧
    x2 - y2 + 2 = 0 ∧ (x2 - 3)^2 + (y2 - 3)^2 = 4 ∧
    let C := (3, 3) in ∃ (u v : ℝ), A = (u, v) ∧ B = (u, v) ∧
    ((C.1 - A.1)*(C.1 - B.1) + (C.2 - A.2)*(C.2 - B.2) = 0)) :=
by
  sorry

end intersection_dot_product_zero_l106_106967


namespace area_of_original_figure_l106_106014

theorem area_of_original_figure (a : ℝ) (A : ℝ)
  (H1 : A = (1/2) * a^2)
  (H2 : A = (sqrt 2 / 4) * (area_of_figure)) :
  area_of_figure = sqrt 2 * a^2 :=
by
    -- Proof omitted
    sorry

end area_of_original_figure_l106_106014


namespace hannah_highest_score_l106_106624

-- Definitions based on conditions
def total_questions : ℕ := 40
def wrong_questions : ℕ := 3
def correct_percent_student_1 : ℝ := 0.95

-- The Lean statement representing the proof problem
theorem hannah_highest_score :
  ∃ q : ℕ, (q > (total_questions - wrong_questions) ∧ q > (total_questions * correct_percent_student_1)) ∧ q = 39 :=
by
  sorry

end hannah_highest_score_l106_106624


namespace mod_remainder_l106_106336

theorem mod_remainder (a b c d : ℕ) (h1 : a = 11) (h2 : b = 9) (h3 : c = 7) (h4 : d = 7) :
  (a^d + b^(d + 1) + c^(d + 2)) % d = 1 := 
by 
  sorry

end mod_remainder_l106_106336


namespace binom_18_10_l106_106430

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106430


namespace total_miles_flown_final_arrival_time_l106_106640

structure DayInfo where
  miles : ℕ
  stopover_hours : ℕ

structure WeekInfo where
  days : List DayInfo

def week1 : WeekInfo := ⟨[
  ⟨1134, 3⟩, -- Monday
  ⟨1475, 2.5⟩, -- Wednesday
  ⟨1290, 1⟩ -- Friday
]⟩

def week2 : WeekInfo := ⟨[
  ⟨1550, 1.5⟩, -- Tuesday
  ⟨1340, 2⟩, -- Thursday
  ⟨1444, 3⟩ -- Saturday
]⟩

def week3 : WeekInfo := ⟨[
  ⟨1200, 4⟩, -- Sunday
  ⟨1000, 1.5⟩, -- Tuesday
  ⟨1360, 2.5⟩ -- Thursday
]⟩

def week4 : WeekInfo := ⟨[
  ⟨1250, 2⟩, -- Monday
  ⟨1300, 4⟩, -- Wednesday
  ⟨1400, 2⟩ -- Friday
]⟩

def week5 : WeekInfo := ⟨[
  ⟨1180, 3⟩, -- Monday
  ⟨1320, 1.5⟩, -- Thursday
  ⟨1460, 3.5⟩ -- Saturday
]⟩

def all_weeks : List WeekInfo := [week1, week2, week3, week4, week5]

def cruising_speed := 500
def start_time_hour := 8

theorem total_miles_flown :
  (all_weeks.map (λ week => week.days.map (λ day => day.miles).sum).sum) = 19703 := sorry

theorem final_arrival_time :
  let final_day := week5.days.getLast? none;
  final_day.map (λ day => 
    let flight_time := day.miles / cruising_speed;
    let total_time := start_time_hour + flight_time + day.stopover_hours;
    total_time = 14 + (25 / 60)) = some true := sorry

end total_miles_flown_final_arrival_time_l106_106640


namespace P_lt_Q_l106_106914

theorem P_lt_Q (a : ℝ) (ha : a ≥ 0) :
  let P := sqrt (a + 2) + sqrt (a + 5)
  let Q := sqrt (a + 3) + sqrt (a + 4)
  P < Q := by
sorry

end P_lt_Q_l106_106914


namespace minimum_SMS_messages_to_win_l106_106712

theorem minimum_SMS_messages_to_win :
  ∀ (host_num : ℕ), 100 ≤ host_num ∧ host_num < 1000 ∧ 
  ∀ d1 d2 d3, 
    (d1 = host_num / 100 ∨ d1 = (host_num / 100 + 1) % 4 + 1 ∨ d1 = (host_num / 100 + 3) % 4 + 1) ∧
    (d2 = host_num / 10 % 10 ∨ d2 = (host_num / 10 % 10 + 1) % 4 + 1 ∨ d2 = (host_num / 10 % 10 + 3) % 4 + 1) ∧
    (d3 = host_num % 10 ∨ d3 = (host_num % 10 + 1) % 4 + 1 ∨ d3 = (host_num % 10 + 3) % 4 + 1) →
    100*d1 + 10*d2 + d3 is_winning :=
begin
  sorry
end

end minimum_SMS_messages_to_win_l106_106712


namespace goods_train_length_l106_106804

noncomputable def train_length (speed_kph length_platform_m crossing_time_s : ℕ) : ℕ :=
  let speed_mps := speed_kph * 1000 / 3600
  let distance_covered := speed_mps * crossing_time_s
  distance_covered - length_platform_m

theorem goods_train_length
  (speed_kph : ℕ) (length_platform_m : ℕ) (crossing_time_s : ℕ)
  (h_speed : speed_kph = 72) (h_length : length_platform_m = 220) (h_time : crossing_time_s = 26) :
  train_length speed_kph length_platform_m crossing_time_s = 300 :=
by
  rw [h_speed, h_length, h_time]
  unfold train_length
  norm_num
  sorry

end goods_train_length_l106_106804


namespace binom_18_10_l106_106474

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106474


namespace fraction_power_multiplication_l106_106492

theorem fraction_power_multiplication :
  ( (1 / 3) ^ 4 * (1 / 5) = 1 / 405 ) :=
by
  sorry

end fraction_power_multiplication_l106_106492


namespace horse_rider_spent_less_l106_106379

noncomputable def ratio_b5_a5 (a1 b1 : ℝ) (d : ℝ) : Prop :=
  b1 + 4 * d = a1 * (1.1^4) * 0.985

theorem horse_rider_spent_less 
  (a1 b1 : ℝ)
  (h_eq : a1 = b1)
  (h_geom : ∀ i, a1 * (1.1^(i-1)))
  (d : ℝ)
  (h_arith : ∀ i, b1 + (i-1) * d)
  (h_total_time : ∑ i in finset.range 5, a1 * 1.1^(i-1) = ∑ i in finset.range 5, b1 + (i-1) * d)
  : b1 + 4 * d = a1 * (1.1^4) * 0.985 := 
sorry

end horse_rider_spent_less_l106_106379


namespace urn_problem_l106_106411

variable (B W : ℕ)

theorem urn_problem (initial_B initial_W : ℕ)
    (h_initial : initial_B = 150 ∧ initial_W = 150)
    (h_op1 : ∀ B W, B ≥ 3 → ∃ B' W', B' = B - 1 ∧ W' = W)
    (h_op2 : ∀ B W, B ≥ 2 ∧ W ≥ 1 → ∃ B' W', B' = B ∧ W' = W - 1)
    (h_op3 : ∀ B W, B ≥ 1 ∧ W ≥ 2 → ∃ B' W', B' = B - 1 ∧ W' = W)
    (h_op4 : ∀ B W, W ≥ 3 → ∃ B' W', B' = B + 1 ∧ W' = W - 1) :
  ∃ (final_B final_W : ℕ), final_B = 50 ∧ final_W = 50 := sorry

end urn_problem_l106_106411


namespace number_of_strikers_l106_106397

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end number_of_strikers_l106_106397


namespace four_digit_numbers_with_distinct_digits_and_average_property_l106_106179

theorem four_digit_numbers_with_distinct_digits_and_average_property :
  let digits := {d | d ∈ Finset.range 10} in
  let valid_sequences := {s : Finset ℕ | ∃ d₁ d₂ d₃ d₄ ∈ digits, s = {d₁, d₂, d₃, d₄} ∧
                                    d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₃ ≠ d₄ ∧ d₄ ≠ d₁ ∧
                                    d₁ ∉ s ∨ d₂ ∉ s ∨ d₃ ∉ s ∨ d₄ ∉ s = false ∧
                                    (∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ 2 * (a + b + c - max (max a b) c) = a + b + c)} in
  valid_sequences.card * 24 = 216 :=
begin
  let digits := {d | d < 10},
  let valid_sequences := {s : Finset ℕ | ∃ d₁ d₂ d₃ d₄, d₁ ∈ digits ∧ d₂ ∈ digits ∧ d₃ ∈ digits ∧ d₄ ∈ digits ∧
                                    s = {d₁, d₂, d₃, d₄} ∧
                                    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄ ∧
                                    (∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ 2 * (a + b + c - max (max a b) c) = a + b + c)},
  have h : valid_sequences.card * 24 = 216,
  { sorry },
  exact h,
end

end four_digit_numbers_with_distinct_digits_and_average_property_l106_106179


namespace field_trip_bread_pieces_l106_106847

theorem field_trip_bread_pieces :
  (students_per_group : ℕ) (num_groups : ℕ) (sandwiches_per_student : ℕ) (pieces_per_sandwich : ℕ)
  (H1 : students_per_group = 6)
  (H2 : num_groups = 5)
  (H3 : sandwiches_per_student = 2)
  (H4 : pieces_per_sandwich = 2)
  : 
  let total_students := num_groups * students_per_group in
  let total_sandwiches := total_students * sandwiches_per_student in
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich in
  total_pieces_bread = 120 :=
by
  let total_students := num_groups * students_per_group
  let total_sandwiches := total_students * sandwiches_per_student
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich
  sorry

end field_trip_bread_pieces_l106_106847


namespace quadratic_eq_has_double_root_l106_106910

theorem quadratic_eq_has_double_root (m : ℝ) :
  (m - 2) ^ 2 - 4 * (m + 1) = 0 ↔ m = 0 ∨ m = 8 := 
by
  sorry

end quadratic_eq_has_double_root_l106_106910


namespace determine_periods_l106_106054

noncomputable def period_pi (f : ℝ → ℝ) : Prop :=
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = π

def y1 : ℝ → ℝ := λ x, cos (abs (2 * x))
def y2 : ℝ → ℝ := λ x, abs (cos x)
def y3 : ℝ → ℝ := λ x, cos (2 * x + π / 6)
def y4 : ℝ → ℝ := λ x, tan (2 * x - π / 4)

theorem determine_periods :
    period_pi y1 ∧ period_pi y2 ∧ period_pi y3 ∧ ¬ period_pi y4 := 
sorry

end determine_periods_l106_106054


namespace sum_fractions_l106_106921

def a : ℝ := sorry -- a is a positive constant
def f (x : ℝ) : ℝ := (a^x) / (a^x + real.sqrt a)

theorem sum_fractions :
  ∑ k in finset.range 1000, f ((k + 1) / 1001) = 500 :=
sorry

end sum_fractions_l106_106921


namespace value_of_a_minus_b_l106_106892

theorem value_of_a_minus_b (a b : ℚ) :
  (∀ (x : ℚ), 0 < x →
   (a / (2^x - 3) + b / (2^x + 1) = (3 * 2^x + 1) / ((2^x - 3) * (2^x + 1)))) →
  (a - b = 2) :=
by
  sorry

end value_of_a_minus_b_l106_106892


namespace number_of_strikers_l106_106398

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end number_of_strikers_l106_106398


namespace correct_statement_l106_106630

-- Define the necessary variables
variables {a b c : ℝ}

-- State the theorem including the condition and the conclusion
theorem correct_statement (h : a > b) : b - c < a - c :=
by linarith


end correct_statement_l106_106630


namespace variance_incorrect_min_standard_deviation_l106_106034

open Real

-- Define a right triangle with hypotenuse of length 3
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ c = 3

-- Prove that the variance of the side lengths cannot be 2
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b 3) : 
  ¬(let x := [a, b, 3] in
    let mean_square := (x.map (λ x, x^2)).sum / 3 in
    let mean := x.sum / 3 in
    mean_square - mean^2 = 2) :=
sorry

-- Prove the minimum standard deviation and corresponding lengths
theorem min_standard_deviation {a b : ℝ} (h : right_triangle a b 3) :
  a = b → a = b → a = 3 * real.sqrt(2) / 2 → b = 3 * real.sqrt(2) / 2 →
  (let variance := (h.1.map (λ x, x^2)).sum / 2 - ((h.1.sum / 2)^2) in
  let std_dev_min := real.sqrt(variance) in
  std_dev_min = real.sqrt(2) - 1) :=
sorry

end variance_incorrect_min_standard_deviation_l106_106034


namespace convex_polygon_diagonals_equal_length_at_most_5_sides_l106_106984

theorem convex_polygon_diagonals_equal_length_at_most_5_sides
  (n : ℕ)
  (h1 : convex (polygon n))
  (h2 : diagonals_equal_length (polygon n))
  : n ≤ 5 :=
sorry

end convex_polygon_diagonals_equal_length_at_most_5_sides_l106_106984


namespace amy_pencils_count_l106_106055

variable (initial_pencils : ℕ) (monday_pencils : ℕ) (tuesday_pencils : ℕ) (gifted_pencils : ℕ)

def total_pencils_after_monday := initial_pencils + monday_pencils
def total_pencils_after_tuesday := total_pencils_after_monday + tuesday_pencils
def final_pencils := total_pencils_after_tuesday - gifted_pencils

theorem amy_pencils_count :
  initial_pencils = 3 → 
  monday_pencils = 7 → 
  tuesday_pencils = 4 → 
  gifted_pencils = 2 → 
  final_pencils = 12 :=
by
  intros h1 h2 h3 h4
  unfold final_pencils total_pencils_after_tuesday total_pencils_after_monday
  rw [h1, h2, h3, h4]
  simp
  sorry

end amy_pencils_count_l106_106055


namespace hexagon_diagonals_l106_106800

theorem hexagon_diagonals : (6 * (6 - 3)) / 2 = 9 := 
by 
  sorry

end hexagon_diagonals_l106_106800


namespace binom_18_10_l106_106431

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106431


namespace final_bug_population_is_zero_l106_106417

def initial_population := 400
def spiders := 12
def spider_consumption := 7
def ladybugs := 5
def ladybug_consumption := 6
def mantises := 8
def mantis_consumption := 4

def day1_population := initial_population * 80 / 100

def predators_consumption_day := (spiders * spider_consumption) +
                                 (ladybugs * ladybug_consumption) +
                                 (mantises * mantis_consumption)

def day2_population := day1_population - predators_consumption_day
def day3_population := day2_population - predators_consumption_day
def day4_population := max 0 (day3_population - predators_consumption_day)
def day5_population := max 0 (day4_population - predators_consumption_day)
def day6_population := max 0 (day5_population - predators_consumption_day)

def day7_population := day6_population * 70 / 100

theorem final_bug_population_is_zero: 
  day7_population = 0 :=
  by
  sorry

end final_bug_population_is_zero_l106_106417


namespace rectangle_area_approx_l106_106711

theorem rectangle_area_approx
    (w : ℝ) (d : ℝ) (area : ℝ)
    (h_w : w = 13)
    (h_d : d = 17)
    (h_area : area = 2 * Real.sqrt 30 * w) :
    area ≈ 142.35 := by
    sorry

end rectangle_area_approx_l106_106711


namespace difference_qr_l106_106807

-- Definitions of p, q, r in terms of the common multiplier x
def p (x : ℕ) := 3 * x
def q (x : ℕ) := 7 * x
def r (x : ℕ) := 12 * x

-- Given condition that the difference between p and q's share is 4000
def condition1 (x : ℕ) := q x - p x = 4000

-- Theorem stating that the difference between q and r's share is 5000
theorem difference_qr (x : ℕ) (h : condition1 x) : r x - q x = 5000 :=
by
  -- Proof placeholder
  sorry

end difference_qr_l106_106807


namespace binomial_coeff_18_10_l106_106476

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106476


namespace binomial_coeff_18_10_l106_106477

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106477


namespace parallel_lines_m_eq_minus_seven_l106_106972

theorem parallel_lines_m_eq_minus_seven
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m)
  (l₂ : ∀ x y : ℝ, 2 * x + (5 + m) * y = 8)
  (parallel : ∀ x y : ℝ, (3 + m) * 4 = 2 * (5 + m)) :
  m = -7 :=
sorry

end parallel_lines_m_eq_minus_seven_l106_106972


namespace find_some_number_l106_106955

theorem find_some_number :
  ∃ some_number : ℝ, (3.242 * some_number) / 100 = 0.032420000000000004 ∧ some_number = 1 :=
by
  have h : (3.242 * 1) / 100 = 0.03242 := by norm_num
  exact ⟨1, h, rfl⟩

end find_some_number_l106_106955


namespace distinct_pairs_count_l106_106098

theorem distinct_pairs_count : 
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ x = x^2 + y^2 ∧ y = 3 * x * y) ∧ 
    S.card = 4 :=
by
  sorry

end distinct_pairs_count_l106_106098


namespace fraction_equivalent_to_0_46_periodic_l106_106524

theorem fraction_equivalent_to_0_46_periodic :
  let a := (46 : ℚ) / 100
  let r := (1 : ℚ) / 100
  let geometric_series_sum (a r : ℚ) :=
    if r.abs < 1 then a / (1 - r) else 0
  geometric_series_sum a r = 46 / 99 := by
    sorry

end fraction_equivalent_to_0_46_periodic_l106_106524


namespace CentralBankInterest_BankBenefits_RegistrationNecessity_l106_106720

-- Define the conditions
variable (CentralBankUsesLoyaltyProgram : Prop)
variable (BanksHaveClientsParticipating : Prop)
variable (LoyaltyProgramOffersBonusesRequiresRegistration : Prop)

-- Define the goals to prove
theorem CentralBankInterest : 
  CentralBankUsesLoyaltyProgram → 
  BanksHaveClientsParticipating → 
  LoyaltyProgramOffersBonusesRequiresRegistration → 
  (PromotionOfNationalPaymentSystem ∧ EconomicStimulus) := 
  by 
  intros h1 h2 h3 
  split 
  sorry

theorem BankBenefits : 
  CentralBankUsesLoyaltyProgram → 
  BanksHaveClientsParticipating → 
  LoyaltyProgramOffersBonusesRequiresRegistration → 
  (IncreasedCustomerLoyalty ∧ HigherTransactionVolumes) := 
  by 
  intros h1 h2 h3 
  split 
  sorry

theorem RegistrationNecessity : 
  CentralBankUsesLoyaltyProgram → 
  BanksHaveClientsParticipating → 
  LoyaltyProgramOffersBonusesRequiresRegistration → 
  (DataCollectionAndMarketing ∧ FraudPreventionAndSecurity) := 
  by 
  intros h1 h2 h3 
  split 
  sorry

end CentralBankInterest_BankBenefits_RegistrationNecessity_l106_106720


namespace inheritance_split_l106_106272

theorem inheritance_split (total_money : ℝ) (num_people : ℕ) (amount_per_person : ℝ) 
  (h1 : total_money = 874532.13) (h2 : num_people = 7) 
  (h3 : amount_per_person = total_money / num_people) : 
  amount_per_person = 124933.16 := by 
  sorry

end inheritance_split_l106_106272


namespace cost_price_of_book_l106_106064

theorem cost_price_of_book (selling_price profit_percent : ℝ) (h1 : selling_price = 300) (h2 : profit_percent = 0.20) : 
  let cost_price := selling_price / (1 + profit_percent) in
  cost_price = 250 :=
by
  have h3 : cost_price = 300 / 1.20, by
    unfold cost_price
    rw [h1, h2]
    norm_num
  rw [h3]
  norm_num
  sorry

end cost_price_of_book_l106_106064


namespace mike_and_rita_chocolates_l106_106364

theorem mike_and_rita_chocolates :
  ( ∃ (chocolates : ℕ), chocolates = 12 )
  ∧ ( ∃ (persons : ℕ), persons = 3 )
  → ( ∃ (shared : ℕ), shared = (12 / 3) * 2 )
  → shared = 8 :=
by
  intros h1 h2
  have h3 : 12 / 3 = 4 := by sorry
  have h4 : shared = (4 * 2) := by sorry
  rw [h3] at h4
  exact h4

end mike_and_rita_chocolates_l106_106364


namespace circle_radius_l106_106809

theorem circle_radius {r : ℤ} (center: ℝ × ℝ) (inside_pt: ℝ × ℝ) (outside_pt: ℝ × ℝ)
  (h_center: center = (2, 1))
  (h_inside: dist center inside_pt < r)
  (h_outside: dist center outside_pt > r)
  (h_inside_pt: inside_pt = (-2, 1))
  (h_outside_pt: outside_pt = (2, -5))
  (h_integer: r > 0) :
  r = 5 :=
by
  sorry

end circle_radius_l106_106809


namespace equal_diagonals_imply_quad_or_pent_l106_106193

theorem equal_diagonals_imply_quad_or_pent (n : ℕ) (F : convex_polygon) 
  (h_n : n ≥ 4) (h_diagonals : ∀ i j, 1 ≤ i < j ≤ n → 
    length (diagonal F i j) = length (diagonal F 1 3)) :
  F ∈ {polygon | polygon_sides polygon = 4} ∪ {polygon | polygon_sides polygon = 5} := 
sorry

end equal_diagonals_imply_quad_or_pent_l106_106193


namespace angle_skew_lines_range_l106_106281

theorem angle_skew_lines_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ ≤ 90) : 0 < θ ∧ θ ≤ 90 :=
by sorry

end angle_skew_lines_range_l106_106281


namespace ratio_both_to_onlyB_is_2_l106_106762

variables (num_A num_B both: ℕ)

-- Given conditions
axiom A_eq_2B : num_A = 2 * num_B
axiom both_eq_500 : both = 500
axiom both_multiple_of_only_B : ∃ k : ℕ, both = k * (num_B - both)
axiom only_A_eq_1000 : (num_A - both) = 1000

-- Define the Lean theorem statement
theorem ratio_both_to_onlyB_is_2 : (both : ℝ) / (num_B - both : ℝ) = 2 := 
sorry

end ratio_both_to_onlyB_is_2_l106_106762


namespace area_triangle_MDA_l106_106643

noncomputable def area_of_triangle_MDA (r : ℝ) : ℝ := 
  let AM := r / 3
  let OM := (r ^ 2 - (AM ^ 2)).sqrt
  let AD := AM / 2
  let DM := AD / (1 / 2)
  1 / 2 * AD * DM

theorem area_triangle_MDA (r : ℝ) : area_of_triangle_MDA r = r ^ 2 / 36 := by
  sorry

end area_triangle_MDA_l106_106643


namespace number_of_sad_children_l106_106707

-- Definitions of the given conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20

-- The main statement to be proved
theorem number_of_sad_children : 
  total_children - happy_children - neither_happy_nor_sad_children = 10 := 
by 
  sorry

end number_of_sad_children_l106_106707


namespace container_capacity_l106_106348

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 18 = 0.75 * C) : 
  C = 40 :=
by
  -- proof steps would go here
  sorry

end container_capacity_l106_106348


namespace number_of_sequences_with_three_consecutive_heads_l106_106826

theorem number_of_sequences_with_three_consecutive_heads
  (a_n : ℕ → ℕ)
  (h0 : a_n 0 = 1)
  (h1 : a_n 1 = 2)
  (h2 : a_n 2 = 4)
  (h_recur : ∀ n > 2, a_n n = a_n (n - 1) + a_n (n - 2) + a_n (n - 3)) :
  let total_sequences := 2^10 in
  let a_10 := a_n 10 in
  total_sequences - a_10 = 520 :=
by 
  sorry

end number_of_sequences_with_three_consecutive_heads_l106_106826


namespace geometric_series_sum_l106_106493

theorem geometric_series_sum :
  let a := 2
  let r := 5
  let n := 5
  (\sum i in Finset.range n, a * r ^ i) = 1562 := 
by
  sorry

end geometric_series_sum_l106_106493


namespace quadratic_has_real_roots_b_3_c_1_l106_106570

theorem quadratic_has_real_roots_b_3_c_1 :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x * x + 3 * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  x₁ = (-3 + Real.sqrt 5) / 2 ∧
  x₂ = (-3 - Real.sqrt 5) / 2 :=
by
  sorry

end quadratic_has_real_roots_b_3_c_1_l106_106570


namespace shift_sine_graph_left_by_pi_over_8_l106_106322

theorem shift_sine_graph_left_by_pi_over_8 :
  ∀ x : ℝ, 2 * sin (2 * x) = 2 * sin (2 * (x + (Real.pi / 8)) - (Real.pi / 4)) :=
by
  assume x
  sorry

end shift_sine_graph_left_by_pi_over_8_l106_106322


namespace quilt_shaded_fraction_l106_106494

theorem quilt_shaded_fraction :
  let quilt_area := 4 * 4
  let shaded_diagonal_squares := 4 * (1 / 2)
  let shaded_full_square := 1
  let shaded_area := shaded_diagonal_squares + shaded_full_square
  let fraction_shaded := shaded_area / quilt_area
  in fraction_shaded = 3 / 16 :=
by
  let quilt_area : ℕ := 4 * 4
  let shaded_diagonal_squares : ℕ := 4 * 1 / 2
  let shaded_full_square : ℕ := 1
  let shaded_area : ℕ := shaded_diagonal_squares + shaded_full_square
  let fraction_shaded : ℚ := shaded_area / quilt_area
  show fraction_shaded = 3 / 16
  sorry

end quilt_shaded_fraction_l106_106494


namespace variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106021

-- Definitions and conditions for Problem a) and b)
noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def variance (a b c : ℝ) : ℝ :=
  let mean_squares := (a^2 + b^2 + c^2) / 3
  let mean := (a + b + c) / 3
  mean_squares - mean^2

-- Prove that the variance is less than 2 given the conditions for Problem a)
theorem variance_triangle_less_than_2 {a b : ℝ} (h : is_right_triangle a b 3) : 
  variance a b 3 < 2 := sorry

-- Define the minimum standard deviation and legs for Problem b)
noncomputable def std_dev_of_legs (a b : ℝ) : ℝ :=
  let sides_squared_mean := (a^2 + b^2) / 2
  let mean_legs := (a + b) / 2
  real.sqrt (sides_squared_mean - mean_legs^2)

-- Prove that the minimum standard deviation of the legs of a right triangle with hypotenuse 3 is sqrt(2) - 1
theorem min_std_dev_of_triangle_legs {a b : ℝ} (h : is_right_triangle a b 3) :
  a = b → std_dev_of_legs a b = real.sqrt(2) - 1 :=
begin
  sorry
end

-- Prove the lengths of the legs for minimum standard deviation
theorem lengths_of_min_std_dev (a b : ℝ) (h : is_right_triangle a b 3) :
  a = b → a = 3 * real.sqrt(2) / 2 ∧ b = 3 * real.sqrt(2) / 2 :=
begin
  sorry
end

end variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106021


namespace monotonic_intervals_f_max_value_g_l106_106169

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - (a + 1) * x + a * Real.log x
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := x^3 + b * x^2 - (2 * b + 4) * x + Real.log x

theorem monotonic_intervals_f (h₁ : a = -4) :
  (∀ x, 1 < x → 0 < (f x a)' ∧ ∀ x, 0 < x ∧ x < 1 → (f x a)' < 0) :=
sorry

theorem max_value_g (a : ℝ) (b : ℝ) 
  (h₂ : 1 < a ∧ a ≤ 2)
  (h₃ : f' a = g' a)
  (h₄ : ¬ 3 * a^2 + (2 * b + 3) * a - 1 = 0) :
  g 1 b ≤ 5 / 4 :=
sorry

end monotonic_intervals_f_max_value_g_l106_106169


namespace max_f_value_l106_106334

def f (y : ℝ) : ℝ := -3 * y^2 + 18 * y - 7

theorem max_f_value : ∃ y : ℝ, (∀ z : ℝ, f(z) ≤ f(y)) ∧ f(y) = 20 :=
by
  sorry

end max_f_value_l106_106334


namespace smallest_number_satisfying_conditions_l106_106010

theorem smallest_number_satisfying_conditions :
  ∃ x : ℕ, (x ≡ 2 [MOD 5]) ∧ (x ≡ 3 [MOD 7]) ∧ (x ≡ 7 [MOD 11]) ∧ (∀ y : ℕ,
    (y ≡ 2 [MOD 5]) ∧ (y ≡ 3 [MOD 7]) ∧ (y ≡ 7 [MOD 11]) → x ≤ y) := 
by 
  use 227
  split
  repeat { split; norm_num }
  intros y hy
  cases' hy with h1 hy
  cases' hy with h2 h3
  sorry

end smallest_number_satisfying_conditions_l106_106010


namespace max_volume_pyramid_l106_106791

/-- Given:
  AB = 3,
  AC = 5,
  sin ∠BAC = 3/5,
  All lateral edges SA, SB, SC form the same angle with the base plane, not exceeding 60°.
  Prove: the maximum volume of pyramid SABC is 5sqrt(174)/4. -/
theorem max_volume_pyramid 
    (A B C S : Type) 
    (AB : ℝ) 
    (AC : ℝ) 
    (alpha : ℝ) 
    (h : ℝ)
    (V : ℝ)
    (sin_BAC : ℝ) :
    AB = 3 →
    AC = 5 →
    sin_BAC = 3 / 5 →
    alpha ≤ 60 →
    V = (1 / 3) * (1 / 2 * AB * AC * sin_BAC) * h →
    V = 5 * sqrt 174 / 4 :=
by
  intros
  sorry

end max_volume_pyramid_l106_106791


namespace jackson_souvenirs_total_l106_106230

def jacksons_collections := 
  let hermit_crabs := 120
  let spiral_shells_per_hermit_crab := 8
  let starfish_per_spiral_shell := 5
  let sand_dollars_per_starfish := 3
  let coral_structures_per_sand_dollars := 4
  let spiral_shells := hermit_crabs * spiral_shells_per_hermit_crab
  let starfish := spiral_shells * starfish_per_spiral_shell
  let sand_dollars := starfish * sand_dollars_per_starfish
  let coral_structures := sand_dollars / coral_structures_per_sand_dollars
  hermit_crabs + spiral_shells + starfish + sand_dollars + coral_structures

theorem jackson_souvenirs_total : jacksons_collections = 22880 := by sorry

end jackson_souvenirs_total_l106_106230


namespace remainder_233_div_d_is_53_l106_106226

theorem remainder_233_div_d_is_53 :
  ∀ (a b c d : ℕ), 1 < a → a < b → b < c →
  a + c = 13 → d = a * b * c →
  (233 % d = 53) :=
begin
  intros a b c d h1 h2 h3 h4 h5,
  sorry
end

end remainder_233_div_d_is_53_l106_106226


namespace velocity_at_t1_l106_106856

theorem velocity_at_t1 : 
  (∀ t : ℝ, s t = -t^2 + 2 * t) →
  deriv s 1 = 0 :=
begin
  intro h,
  calc deriv s 1
  = deriv (λ t, -t^2 + 2 * t) 1 : by rw h
  ... = -2 * 1 + 2 : 
    begin 
      norm_num, 
      simp [deriv],
      norm_num,
    end
  ... = 0 : by norm_num,
end

end velocity_at_t1_l106_106856


namespace ratio_initial_doubled_ratio_1_2_l106_106009

def initial_number (x : ℕ) : Prop :=
  3 * (2 * x + 9) = 81

theorem ratio_initial_doubled (x : ℕ) (h : initial_number x) :
  x = 9 :=
    sorry

theorem ratio_1_2 : ∀ x, initial_number x → ratio x (2*x) = (1 : ℚ) / 2 :=
by sorry

end ratio_initial_doubled_ratio_1_2_l106_106009


namespace find_constants_l106_106659

open BigOperators

theorem find_constants (a b c : ℕ) :
  (∀ n : ℕ, n > 0 → (∑ k in Finset.range n, k.succ * (k.succ + 1) ^ 2) = (n * (n + 1) * (a * n^2 + b * n + c)) / 12) →
  (a = 3 ∧ b = 11 ∧ c = 10) :=
by
  sorry

end find_constants_l106_106659


namespace circle_radius_prime_l106_106655

theorem circle_radius_prime
  (r : ℕ)
  (is_prime : Nat.Prime r)
  (center : (ℤ × ℤ))
  (inside1 inside2 outside1 outside2 : (ℤ × ℤ))
  (h_center : center = (-2, -3))
  (h_inside1 : inside1 = (-2, 2))
  (h_inside2 : inside2 = (1, 0))
  (h_outside1 : outside1 = (5, -3))
  (h_outside2 : outside2 = (-7, 4))
  (h_dist_inside1 : Real.sqrt((inside1.1 - center.1)^2 + (inside1.2 - center.2)^2) < r)
  (h_dist_inside2 : Real.sqrt((inside2.1 - center.1)^2 + (inside2.2 - center.2)^2) < r)
  (h_dist_outside1 : Real.sqrt((outside1.1 - center.1)^2 + (outside1.2 - center.2)^2) > r)
  (h_dist_outside2 : Real.sqrt((outside2.1 - center.1)^2 + (outside2.2 - center.2)^2) > r) :
  r = 5 :=
by sorry

end circle_radius_prime_l106_106655


namespace count_multiples_less_than_300_l106_106978

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end count_multiples_less_than_300_l106_106978


namespace quadrilateral_inscribed_area_l106_106015

def quadrilateral_area (R d1 d2 : ℝ) (O_to_P : ℝ) : ℝ :=
  1/2 * d1 * d2

theorem quadrilateral_inscribed_area :
  ∀ (R d1 d2 : ℝ), (R = 13) → (d1 = 18) → (O_to_P = 4 * Real.sqrt 6) →
  (R^2 = (d1/2)^2 + (d2/2)^2) →
  quadrilateral_area R d1 d2 O_to_P = 72 * Real.sqrt 33 :=
by
  intros R d1 d2 hR hd1 hO_to_P hPythagorean
  sorry

end quadrilateral_inscribed_area_l106_106015


namespace product_divisible_by_four_l106_106329

noncomputable def probability_divisible_by_four : ℚ :=
  let p_odd := 1 / 2 in
  let p_two := 1 / 6 in
  let pr_not_div2 := ((p_odd) ^ 8) in
  let pr_div2_not_div4 := 8 * p_two * (p_odd ^ 7) in
  let pr_not_div4 := pr_not_div2 + pr_div2_not_div4 in
  1 - pr_not_div4

theorem product_divisible_by_four (ans : ℚ) : 
  ans = (757/768) ↔ probability_divisible_by_four = ans :=
by
  sorry

end product_divisible_by_four_l106_106329


namespace Hallie_hours_worked_on_Monday_l106_106620

theorem Hallie_hours_worked_on_Monday :
  ∃ H : ℕ, (10 * H + 18) + (10 * 5 + 12) + (10 * 7 + 20) = 240 ∧ H = 7 := 
by
  exists 7
  -- we only need to assert the existance and the correct rewrite step:
  sorry

end Hallie_hours_worked_on_Monday_l106_106620


namespace function_sqrt_plus_one_l106_106992

variable (f : ℝ → ℝ)
variable (x : ℝ)

theorem function_sqrt_plus_one (h1 : ∀ x : ℝ, f x = 3) (h2 : x ≥ 0) : f (Real.sqrt x) + 1 = 4 :=
by
  sorry

end function_sqrt_plus_one_l106_106992


namespace div_poly_iff_coprime_l106_106888

theorem div_poly_iff_coprime (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∃ k : ℕ, (1 + ↑x ^ n + ↑x ^ (2 * n) + ... + ↑x ^ (m * n)) = k * (1 + ↑x + ↑x ^ 2 + ... + ↑x ^ m)) ↔ (m + 1).gcd n = 1 := 
sorry

end div_poly_iff_coprime_l106_106888


namespace simplify_expression_l106_106731

variable {a b : ℝ}

theorem simplify_expression (h1 : a > 0) (h2 : b > 0) :
  (∛(a * b^2 * sqrt (a * b⁻¹)) * sqrt (a^3 * b)) = a^2 * b :=
sorry

end simplify_expression_l106_106731


namespace proof_problem_l106_106110

theorem proof_problem {A B C P X Y Z : Point} (h1 : Circle A B C) (h2 : P ∈ arc B C) 
  (h3 : ⟂ (line P X) (line B C)) (h4 : ⟂ (line P Y) (line C A)) (h5 : ⟂ (line P Z) (line A B)) :
  (distance B C / distance P X) = (distance C A / distance P Y + distance A B / distance P Z) := 
  sorry

end proof_problem_l106_106110


namespace binom_18_10_eq_43758_l106_106465

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106465


namespace number_of_strikers_correct_l106_106394

-- Defining the initial conditions
def number_of_goalies := 3
def number_of_defenders := 10
def number_of_players := 40
def number_of_midfielders := 2 * number_of_defenders

-- Lean statement to prove
theorem number_of_strikers_correct : 
  let total_non_strikers := number_of_goalies + number_of_defenders + number_of_midfielders,
      number_of_strikers := number_of_players - total_non_strikers 
  in number_of_strikers = 7 :=
by
  sorry

end number_of_strikers_correct_l106_106394


namespace range_of_b_minus_a_l106_106596

theorem range_of_b_minus_a
  (a b : ℝ)
  (h1 : ∀ x ∈ set.Icc a b, x^2 - 2 * x ∈ set.Icc (-1 : ℝ) 3) :
  2 ≤ b - a ∧ b - a ≤ 4 :=
by 
  sorry

end range_of_b_minus_a_l106_106596


namespace compute_abs_w_l106_106233

def z := ((-7 + 15 * complex.I)^2 * (18 - 9 * complex.I)^3) / (5 + 12 * complex.I)
def w := complex.conj z / z

theorem compute_abs_w : complex.abs w = 1 := by
  sorry

end compute_abs_w_l106_106233


namespace smallest_k_l106_106101

theorem smallest_k :
  ∃ k : ℝ, (0 < k) ∧ (∀ x y : ℝ, (0 ≤ x) → (0 ≤ y) → (sqrt (x * y) + k * (abs (x - y))^2 ≥ sqrt (x^2 + y^2))) ∧ (∀ l : ℝ, (0 < l) → (∀ x y : ℝ, (0 ≤ x) → (0 ≤ y) → (sqrt (x * y) + l * (abs (x - y))^2 ≥ sqrt (x^2 + y^2))) → l ≥ k) :=
begin
  use 1,
  split,
  { exact zero_lt_one, },
  { split,
    { intros x y hx hy,
      -- inequality proof placeholder
      sorry, },
    { intros l hl H,
      -- smallest k value proof placeholder
      sorry, }
  }
end

end smallest_k_l106_106101


namespace intersection_of_P_and_M_l106_106990

def P (x : ℝ) : ℝ := x^2 - 3*x + 1
def M (x : ℝ) : ℝ := real.sqrt (x + 2) * real.sqrt (5 - x)

theorem intersection_of_P_and_M :
  (set.Icc (-5/4) 5) = { y : ℝ | ∃ x : ℝ, y = P x } ∩ { y : ℝ | ∃ x : ℝ, y = M x } :=
by sorry

end intersection_of_P_and_M_l106_106990


namespace Gilda_marbles_left_l106_106557

theorem Gilda_marbles_left (M : ℝ) (h1 : M > 0) :
  let remaining_after_pedro := M - 0.30 * M
  let remaining_after_ebony := remaining_after_pedro - 0.40 * remaining_after_pedro
  remaining_after_ebony / M * 100 = 42 :=
by
  sorry

end Gilda_marbles_left_l106_106557


namespace polynomial_coefficients_sum_coeffs_l106_106112

open BigOperators

noncomputable def poly : ℕ → ℕ → ℤ
| n 0 := 1
| n (k+1) := (poly n k) * (n - k) / (k + 1)

theorem polynomial_coefficients (a : ℕ → ℤ) :
  ∀ x : ℤ, (1 - 2 * x)^7 = ∑ k in Finset.range 8, a k * x^k :=
begin
  sorry
end

theorem sum_coeffs (a : ℕ → ℤ) (h : (1 - 2 * x)^7 = ∑ k in Finset.range 8, a k * x^k) :
  (∑ k in Finset.range 7, a (k + 1)) = -2 ∧
  (∑ k in Finset.range 4, a (2 * k + 1)) = -1094 ∧
  (∑ k in Finset.range 4, a (2 * k)) = 1093 :=
begin
  assume x,
  sorry
end

end polynomial_coefficients_sum_coeffs_l106_106112


namespace chocolate_bars_shared_equally_l106_106366

theorem chocolate_bars_shared_equally (total_bars : ℕ) (persons : ℕ) (mike_rita_anita : persons = 3) (bars : total_bars = 12) : 
  (total_bars / persons) * 2 = 8 := 
by 
  have h1 : total_bars = 12 := bars
  have h2 : persons = 3 := mike_rita_anita
  rw [h1, h2] -- substitute values of total_bars and persons
  norm_num -- simplify the arithmetic expression
  sorry

end chocolate_bars_shared_equally_l106_106366


namespace eval_expression_l106_106513

theorem eval_expression : 
    (Int.floor (-7 - 0.5) * Int.ceil (7 + 0.5) * 
    Int.floor (-6 - 0.5) * Int.ceil (6 + 0.5) * 
    Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5) * 
    Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5) * 
    Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5) * 
    Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5) * 
    Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5) * 
    Int.floor (-0.5) * Int.ceil (0.5)) = -1625702400 := 
by
  sorry

end eval_expression_l106_106513


namespace sequence_properties_l106_106173

theorem sequence_properties {a : ℕ → ℝ} (h : ∀ n, a (n + 1) = a n ^ 2 - n * a n + 1) :
  (a 1 = 2 → a 2 = 3 ∧ a 3 = 4 ∧ a 4 = 5 ∧ (∀ n, a n = n + 1)) ∧ 
  (∀ a 1 ≥ 3, (∀ n, a n ≥ n + 2) ∧ (∀ n, (∑ k in finset.range n, 1 / (1 + a (k + 1))) ≤ 1 / 2))
:= by {
  sorry
}

end sequence_properties_l106_106173


namespace problem_correct_l106_106862

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def is_nat_lt_10 (n : ℕ) : Prop := n < 10
def not_zero (n : ℕ) : Prop := n ≠ 0

structure Matrix4x4 :=
  (a₀₀ a₀₁ a₀₂ a₀₃ : ℕ)
  (a₁₀ a₁₁ a₁₂ a₁₃ : ℕ)
  (a₂₀ a₂₁ a₂₂ a₂₃ : ℕ)
  (a₃₀ a₃₁ a₃₂ a₃₃ : ℕ)

def valid_matrix (M : Matrix4x4) : Prop :=
  -- Each cell must be a natural number less than 10
  is_nat_lt_10 M.a₀₀ ∧ is_nat_lt_10 M.a₀₁ ∧ is_nat_lt_10 M.a₀₂ ∧ is_nat_lt_10 M.a₀₃ ∧
  is_nat_lt_10 M.a₁₀ ∧ is_nat_lt_10 M.a₁₁ ∧ is_nat_lt_10 M.a₁₂ ∧ is_nat_lt_10 M.a₁₃ ∧
  is_nat_lt_10 M.a₂₀ ∧ is_nat_lt_10 M.a₂₁ ∧ is_nat_lt_10 M.a₂₂ ∧ is_nat_lt_10 M.a₂₃ ∧
  is_nat_lt_10 M.a₃₀ ∧ is_nat_lt_10 M.a₃₁ ∧ is_nat_lt_10 M.a₃₂ ∧ is_nat_lt_10 M.a₃₃ ∧

  -- Cells in the same region must contain the same number
  M.a₀₀ = M.a₁₀ ∧ M.a₀₀ = M.a₂₀ ∧ M.a₀₀ = M.a₃₀ ∧
  M.a₂₀ = M.a₂₁ ∧
  M.a₂₂ = M.a₂₃ ∧ M.a₂₂ = M.a₃₂ ∧ M.a₂₂ = M.a₃₃ ∧
  M.a₀₃ = M.a₁₃ ∧
  
  -- Cells in the leftmost column cannot contain the number 0
  not_zero M.a₀₀ ∧ not_zero M.a₁₀ ∧ not_zero M.a₂₀ ∧ not_zero M.a₃₀ ∧

  -- The four-digit number formed by the first row is 2187
  is_four_digit (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃) ∧ 
  (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃ = 2187) ∧
  
  -- The four-digit number formed by the second row is 7387
  is_four_digit (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃) ∧ 
  (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃ = 7387) ∧
  
  -- The four-digit number formed by the third row is 7744
  is_four_digit (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃) ∧ 
  (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃ = 7744) ∧
  
  -- The four-digit number formed by the fourth row is 7844
  is_four_digit (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃) ∧ 
  (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃ = 7844)

noncomputable def problem_solution : Matrix4x4 :=
{ a₀₀ := 2, a₀₁ := 1, a₀₂ := 8, a₀₃ := 7,
  a₁₀ := 7, a₁₁ := 3, a₁₂ := 8, a₁₃ := 7,
  a₂₀ := 7, a₂₁ := 7, a₂₂ := 4, a₂₃ := 4,
  a₃₀ := 7, a₃₁ := 8, a₃₂ := 4, a₃₃ := 4 }

theorem problem_correct : valid_matrix problem_solution :=
by
  -- The proof would go here to show that problem_solution meets valid_matrix
  sorry

end problem_correct_l106_106862


namespace gnome_voting_l106_106750

theorem gnome_voting (n : ℕ) :
  (∀ g : ℕ, g < n →  
   (g % 3 = 0 → (∃ k : ℕ, k * 4 = n))
   ∧ (n ≠ 0 ∧ (∀ i : ℕ, i < n → (i + 1) % n ≠ (i + 2) % n) → (∃ k : ℕ, k * 4 = n))) := 
sorry

end gnome_voting_l106_106750


namespace ratio_of_people_on_buses_l106_106301

theorem ratio_of_people_on_buses (P_2 P_3 P_4 : ℕ) 
  (h1 : P_1 = 12) 
  (h2 : P_3 = P_2 - 6) 
  (h3 : P_4 = P_1 + 9) 
  (h4 : P_1 + P_2 + P_3 + P_4 = 75) : 
  P_2 / P_1 = 2 := 
by
  sorry

end ratio_of_people_on_buses_l106_106301


namespace match_functions_l106_106122

def F₁ (x : ℝ) : ℝ := 0.005 + 1/2 * sin (2 * x) + 1/4 * sin (4 * x) + 
                        1/8 * sin (8 * x) + 1/16 * sin (16 * x) + 
                        1/32 * sin (32 * x)

def F₂ (x : ℝ) : ℝ := F₁ (F₁ (x + 0.25))

def F₃ (x : ℝ) : ℝ := F₁ ((1 - x) * F₁ ((1 - x)^2))

def F₄ (x : ℝ) : ℝ := F₁ x + 0.05 * sin (2 * π * x)

def F₅ (x : ℝ) : ℝ := F₁ (x + 1.45) + 0.65

def π (f : ℝ → ℝ) : ℝ → ℝ := f ∘ f ∘ f ∘ f ∘ f

theorem match_functions :
  π F₁ = D ∧ π F₂ = A ∧ π F₃ = C ∧ π F₄ = B ∧ π F₅ = E :=
sorry

end match_functions_l106_106122


namespace tom_takes_2_pills_daily_l106_106778

-- Definitions representing the conditions
def frequency_doctor_visits := 2 -- doctor visits per year since every 6 months
def cost_per_doctor_visit := 400 -- cost per visit in $
def cost_per_pill := 5 -- initial cost per pill in $
def insurance_coverage := 0.80 -- insurance covers 80% of the cost
def total_annual_expense := 1530 -- total annual expense in $

-- Derived values
def annual_doctor_visits_cost : ℕ := frequency_doctor_visits * cost_per_doctor_visit
def cost_per_pill_after_insurance : ℕ := cost_per_pill * (1 - insurance_coverage)
def annual_medication_expense : ℕ := total_annual_expense - annual_doctor_visits_cost

-- Proof problem statement: prove that Tom takes 2 pills per day
theorem tom_takes_2_pills_daily :
  (annual_medication_expense / cost_per_pill_after_insurance) / 365 = 2 := by
sorry

end tom_takes_2_pills_daily_l106_106778


namespace location_in_quadrant_l106_106682

noncomputable def z : ℂ := (2 - complex.i) / (1 + complex.i)

theorem location_in_quadrant :
  0 < z.re ∧ z.im < 0 :=
by
  sorry

end location_in_quadrant_l106_106682


namespace container_volume_ratio_l106_106867

theorem container_volume_ratio (V1 V2 : ℚ)
  (h1 : (3 / 5) * V1 = (2 / 3) * V2) :
  V1 / V2 = 10 / 9 :=
by sorry

end container_volume_ratio_l106_106867


namespace contact_probability_l106_106537

theorem contact_probability (p : ℝ) :
  let m := 6 in
  let n := 7 in
  let number_of_pairs := m * n in
  1 - (1 - p) ^ number_of_pairs = 1 - (1 - p) ^ 42 :=
by
  let m := 6
  let n := 7
  let number_of_pairs := m * n
  have h1 : number_of_pairs = 42 := by norm_num
  rw h1
  sorry

end contact_probability_l106_106537


namespace necessary_and_sufficient_condition_l106_106966

-- Definitions based on conditions
def line_eq (m : ℝ) : ℝ → ℝ → Prop := λ x y, m * x + y + 1 - 2 * m = 0
def circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2 * x + 4 * y + 1 = 0

-- Main theorem statement
theorem necessary_and_sufficient_condition :
  ∀ m ∈ ℝ,
  ( (∀ x y, circle_eq x y → (abs (distance (x, y) (m, 1 - 2 * m)) = 1 → m = 0)) ∧
    ( (m = 0) → (∀ x y, circle_eq x y → abs (distance (x, y) (m, 1 - 2 * m)) = 1)) ) :=
sorry

end necessary_and_sufficient_condition_l106_106966


namespace range_of_m_max_area_OAP_l106_106951

noncomputable def curve1 (a x y : ℝ) := (x^2) / (a^2) + y^2 = 1
noncomputable def curve2 (m x y : ℝ) := y^2 = 2 * (x + m)
def valid_a (a : ℝ) := 0 < a ∧ a < 1/2

-- Part 1: Prove the range of m
theorem range_of_m (a m : ℝ) (h_a_pos : 0 < a)
                  (h_curve1 : ∀ x y, curve1 a x y)
                  (h_curve2 : ∀ x y, curve2 m x y) :
                  ((-a < m ∧ m < a) ∨ m = (a^2 + 1) / 2) := sorry

-- Part 2: Maximum area of triangle OAP
theorem max_area_OAP (a : ℝ) (h_a_pos : valid_a a) :
  ∃ S : ℝ, S = (1/2) * a * (real.sqrt(1 - a^4)) := sorry

end range_of_m_max_area_OAP_l106_106951


namespace composition_of_linear_maps_l106_106675

-- Define Q as the set of rational numbers
def Q := ℚ

noncomputable def f (a b: Q) : Q → Q := λ x, a * x + b

theorem composition_of_linear_maps
  (a b c d : Q) :
  (f a b) ∘ (f c d) = f (a * c) (a * d + b) :=
by 
  sorry

end composition_of_linear_maps_l106_106675


namespace compare_magnitudes_l106_106071

theorem compare_magnitudes : -0.5 > -0.75 :=
by
  have h1 : |(-0.5: ℝ)| = 0.5 := by norm_num
  have h2 : |(-0.75: ℝ)| = 0.75 := by norm_num
  have h3 : (0.5: ℝ) < 0.75 := by norm_num
  sorry

end compare_magnitudes_l106_106071


namespace grocer_initial_stock_l106_106005

noncomputable def initial_coffee_stock (x : ℝ) : Prop :=
  let initial_decaf := 0.20 * x
  let additional_coffee := 100
  let additional_decaf := 0.50 * additional_coffee
  let total_coffee := x + additional_coffee
  let total_decaf := initial_decaf + additional_decaf
  0.26 * total_coffee = total_decaf

theorem grocer_initial_stock :
  ∃ x : ℝ, initial_coffee_stock x ∧ x = 400 :=
by
  sorry

end grocer_initial_stock_l106_106005


namespace sum_of_real_roots_l106_106102

theorem sum_of_real_roots : 
  let P := ∏ k in (finset.range 100).map (λ x, x + 1), (λ x, x^2 - 11 * x + k)
  (∑ k in finset.range 30, 11) = 330 := 
by
  sorry

end sum_of_real_roots_l106_106102


namespace unique_rational_point_on_sphere_l106_106503
open Real

-- Define a predicate for a rational point in 3D space
def is_rational_point (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y z : ℚ), (p = (x, y, z))

-- Define the specific sphere equation
def on_sphere (p : ℝ × ℝ × ℝ) : Prop :=
  (p.1 - Real.sqrt 2) ^ 2 + p.2 ^ 2 + p.3 ^ 2 = 2

theorem unique_rational_point_on_sphere :
  ∃! (p : ℝ × ℝ × ℝ), is_rational_point p ∧ on_sphere p :=
begin
  sorry
end

end unique_rational_point_on_sphere_l106_106503


namespace cake_exactly_two_iced_sides_l106_106000

/-- A $5 \times 5 \times 5$ cube with icing on the top, front, and right sides only. -/
structure Cake :=
  (dimensions : ℕ × ℕ × ℕ)
  (icing : set (ℕ × ℕ × ℕ))

/-- The cake described in the problem where dimensions are (5, 5, 5) and icing is on the top, front, and right sides. -/
def cake : Cake :=
{ dimensions := (5, 5, 5),
  icing := { xyz | (xyz.3 = 5) ∨ (xyz.1 = 5) ∨ (xyz.2 = 5) } }

/-- Function to count the number of 1x1x1 cubes with icing on exactly two sides in a given Cake. -/
def count_exactly_two_iced_sides (c : Cake) : ℕ :=
  (finset.univ.product finset.univ).count (λ xyz,
    let ⟨x, y, z⟩ := xyz in
    ((x = 5) ∨ (y = 5) ∨ (z = 5)) ∧ -- At least one side has icing
    [x = 5, y = 5, z = 5].count id = 2) - 1

/-- Prove that the number of 1x1x1 cubes with icing on exactly two sides for the specified cake is 5. -/
theorem cake_exactly_two_iced_sides : count_exactly_two_iced_sides cake = 5 :=
sorry

end cake_exactly_two_iced_sides_l106_106000


namespace isosceles_right_triangle_l106_106715

theorem isosceles_right_triangle 
  (a b c m_a m_b m_c : ℝ) (h : a ≤ b) 
  (ha : a ≤ m_a) (hb : b ≤ m_b) :
  (∃ (T : Type) [inhabited T], ∀ (t : T), a^2 + b^2 = c^2) → 
  (m_a = b) ∧ (m_b = a) ∧ (c = a * b * real.sqrt 2) ∧ (b = a) := 
sorry

end isosceles_right_triangle_l106_106715


namespace science_students_sampled_l106_106660

theorem science_students_sampled
  (total_students : ℕ)
  (liberal_arts_students : ℕ)
  (science_students : ℕ)
  (sample_size : ℕ)
  (sample_proportion : ℚ)
  (sampled_science_students : ℕ)
  (h1 : total_students = 140)
  (h2 : liberal_arts_students = 40)
  (h3 : science_students = total_students - liberal_arts_students)
  (h4 : sample_size = 14)
  (h5 : sample_proportion = (sample_size : ℚ) / total_students)
  (h6 : sampled_science_students = int(sample_proportion * science_students)) :
  sampled_science_students = 10 :=
by sorry

end science_students_sampled_l106_106660


namespace length_of_train_l106_106350

theorem length_of_train (speed_kmh : ℕ) (time_seconds : ℕ) (h_speed : speed_kmh = 60) (h_time : time_seconds = 36) :
  let time_hours := (time_seconds : ℚ) / 3600
  let distance_km := (speed_kmh : ℚ) * time_hours
  let distance_m := distance_km * 1000
  distance_m = 600 :=
by
  sorry

end length_of_train_l106_106350


namespace horse_saddle_ratio_l106_106831

theorem horse_saddle_ratio (total_cost : ℕ) (saddle_cost : ℕ) (horse_cost : ℕ) 
  (h_total : total_cost = 5000)
  (h_saddle : saddle_cost = 1000)
  (h_sum : horse_cost + saddle_cost = total_cost) : 
  horse_cost / saddle_cost = 4 :=
by sorry

end horse_saddle_ratio_l106_106831


namespace find_k_l106_106221

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 3 = 2 * (n + k) + 5) : k = 3 / 2 := 
by 
  sorry

end find_k_l106_106221


namespace domain_of_function_l106_106083

theorem domain_of_function : 
  {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x} :=
by 
  sorry

end domain_of_function_l106_106083


namespace no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l106_106806

-- Part (a): Prove that it is impossible to arrange five distinct-sized squares to form a rectangle.
theorem no_rectangle_with_five_distinct_squares (s1 s2 s3 s4 s5 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s4 ≠ s5) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5)) :=
by
  -- Proof placeholder
  sorry

-- Part (b): Prove that it is impossible to arrange six distinct-sized squares to form a rectangle.
theorem no_rectangle_with_six_distinct_squares (s1 s2 s3 s4 s5 s6 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s1 ≠ s6 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s2 ≠ s6 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s3 ≠ s6 ∧ s4 ≠ s5 ∧ s4 ≠ s6 ∧ s5 ≠ s6) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧ (s6 ≤ l ∧ s6 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5 + s6)) :=
by
  -- Proof placeholder
  sorry

end no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l106_106806


namespace triangle_angles_ineq_l106_106341

theorem triangle_angles_ineq {A B C : Type} [EuclideanGeometry A B C] (h : ¬(AB = AC)) : ¬(∠B = ∠C) :=
sorry

end triangle_angles_ineq_l106_106341


namespace binom_18_10_eq_43758_l106_106464

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106464


namespace solution_set_ineq_l106_106565

variable (f : ℝ → ℝ)

axiom h1 : f 2 = 7
axiom h2 : ∀ x : ℝ, (deriv f x) < 3

theorem solution_set_ineq : {x : ℝ | f x < 3 * x + 1} = set.Ioi 2 := by
  sorry

end solution_set_ineq_l106_106565


namespace equation_nth_position_l106_106261

theorem equation_nth_position (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 :=
by
  sorry

end equation_nth_position_l106_106261


namespace sufficient_but_not_necessary_l106_106358

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

def z (a : ℝ) : ℂ := ⟨a^2 - 4, a + 1⟩

theorem sufficient_but_not_necessary (a : ℝ) (h : a = -2) : 
  is_purely_imaginary (z a) ∧ ¬(∀ a, is_purely_imaginary (z a) → a = -2) :=
by
  sorry

end sufficient_but_not_necessary_l106_106358


namespace binom_18_10_l106_106490

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106490


namespace find_A_l106_106748

-- Define the four-digit number being a multiple of 9 and the sum of its digits condition
def digit_sum_multiple_of_9 (A : ℤ) : Prop :=
  (3 + A + A + 1) % 9 = 0

-- The Lean statement for the proof problem
theorem find_A (A : ℤ) (h : digit_sum_multiple_of_9 A) : A = 7 :=
sorry

end find_A_l106_106748


namespace variance_incorrect_min_std_deviation_l106_106039

-- Definitions for the given conditions.
variable (a b : ℝ)

-- The right triangle condition given by Pythagorean theorem.
def right_triangle (a b : ℝ) : Prop :=
  a^2 + b^2 = 9

-- Problems to verify
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b) : 
  ¬(variance {a, b, 3} = 2) := sorry

theorem min_std_deviation {a b : ℝ} (h : right_triangle a b) :
  let s := sqrt(2) - 1,
  (a = b) → (a = 3 * sqrt(2) / 2) → (std_deviation {a, b, 3} = s) := sorry

end variance_incorrect_min_std_deviation_l106_106039


namespace min_value_f_ln_x_inequality_l106_106605

-- Problem 1: Prove the minimum value of f(x) is -1 / e
theorem min_value_f (x : ℝ) (h_pos : 0 < x) :
  let f := λ x : ℝ, x * Real.log x in
  ∃ c : ℝ, c > 0 ∧ f(c) = -1 / Real.exp 1 := sorry

-- Problem 2: Prove ∀ x ∈ (0, +∞), ln x > 1 / e^x - 2 / (ex)
theorem ln_x_inequality (x : ℝ) (h_pos : 0 < x) :
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) := sorry

end min_value_f_ln_x_inequality_l106_106605


namespace number_of_ways_to_divide_friends_l106_106178

theorem number_of_ways_to_divide_friends : 
  ∀ (friends : Fin 8) (teams : Fin 4),
  (∑ i in Finset.univ, (4 ^ 8)) = 65536 :=
by 
  sorry

end number_of_ways_to_divide_friends_l106_106178


namespace solve_system_l106_106278

-- Define the system of equations
def eq1 (x y : ℝ) : Prop := 2 * x - y = 8
def eq2 (x y : ℝ) : Prop := 3 * x + 2 * y = 5

-- State the theorem to be proved
theorem solve_system : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 3 ∧ y = -2 := 
by 
  exists 3
  exists -2
  -- Proof steps would go here, but we're using sorry to indicate it's incomplete
  sorry

end solve_system_l106_106278


namespace question1_question2_l106_106920

variables (a b : ℝ × ℝ)

-- Conditions
def condition1 := (∥a∥ = 1)
def condition2 := (∥b∥ = sqrt 3)
def condition3 := (a + b = (sqrt 3, 1))

-- Question 1: Prove |a - b| = 2
theorem question1 (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) :
  ∥a - b∥ = 2 :=
sorry

-- Question 2: Prove the angle between a + b and a - b is 2π / 3
theorem question2 (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) :
  let θ := real.arccos ((/ ((a + b) • (a - b)) (2 * 2))) in θ = (2 * real.pi) / 3 :=
sorry

end question1_question2_l106_106920


namespace all_points_collinear_l106_106684

-- Definitions based on the conditions
variables {Point : Type} [Finite Point] [Nonempty Point]

-- Condition on the set of points
def collinear (A B C : Point) : Prop :=
  ∃ (l : Line), A ∈ l ∧ B ∈ l ∧ C ∈ l

variable (E : Finset Point)

-- Hypothesis based on the condition that for any two points
-- there is a third point that makes them collinear
def all_pairs_have_collinear_third (E : Finset Point) : Prop :=
  ∀ (A B : Point), A ∈ E → B ∈ E → A ≠ B → ∃ (C : Point), C ∈ E ∧ collinear A B C

-- Question to prove:
theorem all_points_collinear (h : all_pairs_have_collinear_third E) : 
  ∃ (l : Line), ∀ (A : Point), A ∈ E → A ∈ l :=
sorry

end all_points_collinear_l106_106684


namespace range_of_m_l106_106126

theorem range_of_m (m : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x - 1) * (x - (m - 1)) > 0) → m > 1 :=
by
  intro h
  sorry

end range_of_m_l106_106126


namespace second_shirt_price_l106_106018

-- Define the conditions
def price_first_shirt := 82
def price_third_shirt := 90
def min_avg_price_remaining_shirts := 104
def total_shirts := 10
def desired_avg_price := 100

-- Prove the price of the second shirt
theorem second_shirt_price : 
  ∀ (P : ℝ), 
  (price_first_shirt + P + price_third_shirt + 7 * min_avg_price_remaining_shirts = total_shirts * desired_avg_price) → 
  P = 100 :=
by
  sorry

end second_shirt_price_l106_106018


namespace total_tires_correct_l106_106403

-- Define the number of vehicles
def total_vehicles : ℕ := 24

-- Define the fraction of vehicles that are motorcycles
def motorcycles_fraction : ℚ := 1 / 3

-- Define the fraction of cars that have a spare tire
def cars_with_spare_tire_fraction : ℚ := 1 / 4

-- Define the total number of motorcycles
def number_of_motorcycles : ℕ := (motorcycles_fraction * total_vehicles).to_nat

-- Define the total number of cars
def number_of_cars : ℕ := total_vehicles - number_of_motorcycles

-- Define the number of cars with a spare tire
def cars_with_spare_tire : ℕ := (cars_with_spare_tire_fraction * number_of_cars).to_nat

-- Define the number of cars without a spare tire
def cars_without_spare_tire : ℕ := number_of_cars - cars_with_spare_tire

-- Calculate the total number of tires
def total_tires : ℕ := 
  (number_of_motorcycles * 2) + 
  (cars_with_spare_tire * 5) + 
  (cars_without_spare_tire * 4)

-- Theorem statement to prove
theorem total_tires_correct : total_tires = 84 := by
  sorry

end total_tires_correct_l106_106403


namespace a_seq_formula_b_seq_formula_T_n_formula_l106_106770

noncomputable def a_seq : ℕ → ℝ
| 0     := (1 / 2)
| (n + 1) := (1 / 2) * a_seq n

def b_seq : ℕ → ℝ
| 0     := 4
| (n + 1) := 3 * b_seq n - 2

noncomputable def c_seq (a_seq : ℕ → ℝ) (b_seq : ℕ → ℝ) (n : ℕ) : ℝ :=
a_seq n * Real.log (b_seq (2 * n -1) - 1) / Real.log 3

noncomputable def T_n (a_seq : ℕ → ℝ) (b_seq : ℕ → ℝ) (n : ℕ) : ℝ :=
(Sum (Finset.range (n + 1)).map (λ k, c_seq a_seq b_seq k))

theorem a_seq_formula : ∀ n : ℕ, a_seq n = (1 / 2) ^ n :=
sorry

theorem b_seq_formula : ∀ n : ℕ, b_seq n = 3^n + 1 :=
sorry

theorem T_n_formula : ∀ n : ℕ, T_n a_seq b_seq n = 3 - (2 * n + 3) / 2^n :=
sorry

end a_seq_formula_b_seq_formula_T_n_formula_l106_106770


namespace root_of_quadratic_gives_value_l106_106581

theorem root_of_quadratic_gives_value (a : ℝ) (h : a^2 + 3 * a - 5 = 0) : a^2 + 3 * a + 2021 = 2026 :=
by {
  -- We will skip the proof here.
  sorry
}

end root_of_quadratic_gives_value_l106_106581


namespace sin_squared_sum_lt_one_l106_106195

theorem sin_squared_sum_lt_one 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < π / 2) 
  (hy : 0 < y ∧ y < π / 2) 
  (hz : 0 < z ∧ z < π / 2) 
  (h : tan x + tan y + tan z = 2) :
  sin x ^ 2 + sin y ^ 2 + sin z ^ 2 < 1 := 
sorry

end sin_squared_sum_lt_one_l106_106195


namespace problem_statement_l106_106357

variable {n : ℕ}
variable {a : Fin n → ℝ}

def is_non_increasing (a : Fin n → ℝ) : Prop :=
  ∀ (i j : Fin n), i < j → a i ≥ a j

def sum_powers_nonnegative (a : Fin n → ℝ) : Prop :=
  ∀ k : ℕ, 0 < k → 0 ≤ ∑ i, (a i) ^ k

noncomputable def p (a : Fin n → ℝ) : ℝ :=
  Finset.max' (Finset.image (λ i, |a i|) Finset.univ) sorry

theorem problem_statement (h1 : is_non_increasing a) (h2 : sum_powers_nonnegative a) :
  p a = a ⟨0, sorry⟩ ∧ ∀ (x : ℝ), x > a ⟨0, sorry⟩ → ∏ i, (x - a i) ≤ x ^ n - (a ⟨0, sorry⟩) ^ n := 
sorry

end problem_statement_l106_106357


namespace find_angle_B_find_area_of_ABC_l106_106225

noncomputable def angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) : ℝ := 
  if b * Real.cos C = -a then Real.pi - 2 * Real.arctan (a / c)
  else 2 * Real.pi / 3

theorem find_angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) :
  angle_B a b c C h1 = 2 * Real.pi / 3 := 
sorry

noncomputable def area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) : ℝ :=
  if position = 1 then /- calculation for BD bisector case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)
  else /- calculation for midpoint case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)

theorem find_area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) (hB : angle_B a b c C h1 = 2 * Real.pi / 3) :
  area_of_ABC a b c C (2 * Real.pi / 3) d position h1 h2 h3 = Real.sqrt 3 := 
sorry

end find_angle_B_find_area_of_ABC_l106_106225


namespace find_value_l106_106163

-- Defining the conditions and the function
variable {a b : ℝ}

def f (x : ℝ) : ℝ := a * x^3 + b * (Real.sin x) + 4

-- Given condition
axiom h : f (Real.log10 (Real.log 10 / Real.log 2)) = 5

-- Target statement to prove
theorem find_value : f (Real.log10 (Real.log 2)) = 3 :=
sorry

end find_value_l106_106163


namespace math_problem_l106_106653

-- Definition of the parametric equations of circle C
def parametric_circle (α : ℝ) := (x : ℝ, y : ℝ) :=
  (3 + 2 * Real.cos α, 2 * Real.sin α)

-- Polar equation of line l
def polar_line (ρ θ : ℝ) := ρ * Real.cos θ + ρ * Real.sin θ + 1 = 0

-- Standard form equation for circle
def standard_form_circle_eq : Prop :=
  ∀ x y α, (parametric_circle α).1 = x → (parametric_circle α).2 = y →
  (x - 3) ^ 2 + y ^ 2 = 4

-- Conversion polar equation of line l to rectangular coordinates
def rectangular_line_eq : Prop :=
  ∀ ρ θ, polar_line ρ θ → 
  (∃ x y, ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan2 y x ∧ x + y + 1 = 0)

-- Minimum value of |PA| * |PB| through any point on line l
def minimum_value_of_PA_PB : ℝ :=
  12

-- Complete proof problem
theorem math_problem :
  standard_form_circle_eq ∧ rectangular_line_eq ∧
  (∀ P : ℝ × ℝ, (P.1 + P.2 + 1) = 0 → (∃ A B, (A.1 = P.1) ∧ (B.1 = P.2) ∧ |P.1 - A.1| * |P.2 - B.2| = minimum_value_of_PA_PB)) :=
  by sorry

end math_problem_l106_106653


namespace prob1_derivative_prob2_derivative_prob3_derivative_at_neg1_prob4_derivative_at_pi8_l106_106813

noncomputable def prob1 (a x k : ℝ) : ℝ :=
(3 * a * x - x ^ 2) ^ k

theorem prob1_derivative (a x k : ℝ) : 
  deriv (λ x, prob1 a x k) x = k * (3 * a - 2 * x) * (3 * a * x - x ^ 2) ^ (k - 1) :=
sorry

noncomputable def prob2 (α : ℝ) : ℝ :=
2 * sqrt (sin (α / 3))

theorem prob2_derivative (α : ℝ) : 
  deriv (λ α, prob2 α) α = (cos (α / 3)) / (3 * sqrt (sin (α / 3))) :=
sorry

noncomputable def prob3 (t : ℝ) : ℝ :=
(t / (2 * t + 1)) ^ 10

theorem prob3_derivative_at_neg1 : 
  deriv (λ t, prob3 t) (-1) = 10 :=
sorry

noncomputable def prob4 (ϕ : ℝ) : ℝ :=
(sin (2 * ϕ)) ^ 3 - (cos (2 * ϕ)) ^ 3

theorem prob4_derivative_at_pi8 : 
  deriv (λ ϕ, prob4 ϕ) (π / 8) = 3 * sqrt 2 :=
sorry

end prob1_derivative_prob2_derivative_prob3_derivative_at_neg1_prob4_derivative_at_pi8_l106_106813


namespace find_a_perpendicular_lines_l106_106614

variable (a : ℝ)

theorem find_a_perpendicular_lines :
  (∃ a : ℝ, ∀ x y : ℝ, (a * x - y + 2 * a = 0) ∧ ((2 * a - 1) * x + a * y + a = 0) → a = 0 ∨ a = 1) := 
sorry

end find_a_perpendicular_lines_l106_106614


namespace binomial_coeff_18_10_l106_106483

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106483


namespace pos_integers_divisible_by_2_3_5_7_less_than_300_l106_106982

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end pos_integers_divisible_by_2_3_5_7_less_than_300_l106_106982


namespace f_expression_F_parity_odd_log_inequality_solution_l106_106958

noncomputable def a := 2

def f (x : ℝ) : ℝ := a^x
def F (x : ℝ) : ℝ := f(x) - f(-x)

theorem f_expression : ∀ x : ℝ, f(x) = 2^x := 
by sorry

theorem F_parity_odd : ∀ x : ℝ, F(-x) = -F(x) := 
by sorry

theorem log_inequality_solution (x : ℝ) : 
  -2 < x ∧ x < -1/2 ↔ log a (1 - x) > log a (x + 2) :=
by sorry

end f_expression_F_parity_odd_log_inequality_solution_l106_106958


namespace number_of_whole_numbers_between_l106_106987

noncomputable def fourth_root_17 := real.rpow 17 (1/4)
noncomputable def fourth_root_340 := real.rpow 340 (1/4)

theorem number_of_whole_numbers_between :
  ∃ n : ℕ, n = 2 ∧ ∃ a b : ℝ, a = fourth_root_17 ∧ b = fourth_root_340 ∧
    ∀ (x : ℕ), (a < x ∧ x < b) → (x = 3 ∨ x = 4) :=
by {
  sorry
}

end number_of_whole_numbers_between_l106_106987


namespace john_overall_percentage_correct_l106_106201

-- Definitions for the given conditions
variable (t : ℝ) -- Total number of problems
variable (e_solved_alone : ℝ) := 0.75 * t -- Problems Emily solved alone
variable (j_solved_alone : ℝ) := 0.75 * t -- Problems John solved alone
variable (e_accuracy_alone : ℝ) := 0.70 -- Emily's accuracy for problems she solved alone
variable (j_accuracy_alone : ℝ) := 0.85 -- John's accuracy for problems he solved alone
variable (e_overall_accuracy : ℝ) := 0.76 -- Emily's overall accuracy

-- Define the number of problems correctly solved together by Emily and John
variable (y : ℝ) := e_overall_accuracy * t - e_accuracy_alone * e_solved_alone

-- Define John's overall number of correct answers
def john_overall_correct : ℝ := j_accuracy_alone * j_solved_alone + y

-- Define John's overall percentage of correct answers
def john_overall_percentage : ℝ := (john_overall_correct / t) * 100

-- The theorem states that John's overall percentage of correct answers is 87.25%
theorem john_overall_percentage_correct : john_overall_percentage t e_accuracy_alone j_accuracy_alone e_overall_accuracy = 87.25 := by
  -- Proof is not required, so we use sorry
  sorry

end john_overall_percentage_correct_l106_106201


namespace area_S_l106_106688

open Real

variables {P : Type*} [Real : metric_space P]

def Point := P

def Set (p : P) : Prop :=
  -- Define the circle Omega with radius 8 centered at point O
  let Omega := {p : P | dist p (0 : P) ≤ 8} in
  -- Define point M on the circle Omega
  let M : P := (8, 0) in
  -- Define the set S containing points P
  let S := {P : P |
    ((8 : Point) ∈ Omega ∨
    (∃ (A B C D : Point), A.1 = 4 ∧ B.2 = 5 ∧ B.3 = 5 ∧
    dist E (0 : Point) ≤ 8 
    ∧
    (dist A O ≤ AB ∨ dist P B) 
    ∧ ((B) (X.O)))⟩ 

def area (S : Set) : ℝ :=
  164 + 64 * π 

-- The theorem which states that the area of the set S is 164 + 64π
theorem area_S : 
  let Omega := {P : Point | dist p (0 : Point) ≤ 8}
  ∃ S : Set
  (∀ p : P, Set contains p ) 
  show (area Set. p) 
  (area S =164 + 64 * π :=
sorry

end area_S_l106_106688


namespace triangle_area_l106_106401

theorem triangle_area (x : ℝ) (h1 : 6 * x = 6) (h2 : 8 * x = 8) (h3 : 10 * x = 2 * 5) : 
  1 / 2 * 6 * 8 = 24 := 
sorry

end triangle_area_l106_106401


namespace book_cost_l106_106725

theorem book_cost 
  (initial_money : ℕ) 
  (num_books : ℕ) 
  (money_left : ℕ) 
  (h_init : initial_money = 79) 
  (h_books : num_books = 9) 
  (h_left : money_left = 16) : 
  (initial_money - money_left) / num_books = 7 :=
by
  rw [h_init, h_books, h_left] 
  norm_num
  sorry

end book_cost_l106_106725


namespace shaded_square_ratio_l106_106123

theorem shaded_square_ratio (side_length : ℝ) (H : side_length = 5) :
  let large_square_area := side_length ^ 2
  let shaded_square_area := (side_length / 2) ^ 2
  shaded_square_area / large_square_area = 1 / 4 :=
by
  sorry

end shaded_square_ratio_l106_106123


namespace solve_quadratic_l106_106304

theorem solve_quadratic (x : ℝ) : x^2 = x ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solve_quadratic_l106_106304


namespace trapezoid_area_l106_106372

theorem trapezoid_area
  (A B C D E : Type)
  (CD AE : ℝ)
  (CD_eq : CD = 6 * Real.sqrt 13)
  (AE_eq : AE = 8)
  (BC_parallel_AD : ∀ (BC AD : Type), is_parallel BC AD)
  (circle_center_diag_AC : ∀ (AC : Type), lies_on_circle {A, B} AC)
  (circle_touches_CD_at_C : ∀ (CD : Type), touches_circle CD C)
  (circle_intersects_AD_at_E : ∀ (AD : Type), intersects_circle AD E) :
  area_trapezoid ABCD = 204 :=
by
  sorry

end trapezoid_area_l106_106372


namespace determine_z_l106_106607

-- Definition of the imaginary unit i
def i : ℂ := complex.I

-- Given conditions
def given_equation (z : ℂ) : Prop := (2 / (1 + i) = complex.conj z + i)

-- Lean statement of the proof problem
theorem determine_z (z : ℂ) (h : given_equation z) : z = (1 : ℂ) + 2 * i :=
sorry

end determine_z_l106_106607


namespace perimeter_of_square_fence_l106_106324

theorem perimeter_of_square_fence :
  ∀ (n : ℕ) (post_gap post_width : ℝ), 
  4 * n - 4 = 24 →
  post_gap = 6 →
  post_width = 5 / 12 →
  4 * ((n - 1) * post_gap + n * post_width) = 156 :=
by
  intros n post_gap post_width h1 h2 h3
  sorry

end perimeter_of_square_fence_l106_106324


namespace three_vertices_visible_from_outside_l106_106925

-- Definitions based on problem conditions
variable (M : Polyhedron)

def is_convex (M : Polyhedron) : Prop := sorry  -- Definition of convex polyhedron
def vertex (M : Polyhedron) : Type := sorry  -- Type of vertices of the polyhedron
def is_vertex (A : vertex M) : Prop := sorry  -- Condition that A is a vertex of polyhedron M
def is_outside (P : Point) (M : Polyhedron) : Prop := sorry  -- Condition that P is outside of polyhedron M
def visible (P : Point) (A : vertex M) : Prop := sorry  -- Define visibility of a vertex from a point

-- The proof problem
theorem three_vertices_visible_from_outside (hM : is_convex M) (A B C : vertex M)
  (hvA : is_vertex A) (hvB : is_vertex B) (hvC : is_vertex C) :
  ∃ (P : Point), is_outside P M ∧ visible P A ∧ visible P B ∧ visible P C :=
sorry

end three_vertices_visible_from_outside_l106_106925


namespace math_proof_example_l106_106317

open Nat

noncomputable def a_gcd_b : ℕ := (10 : ℕ) ^ (80 / 3 / 3).to_nat
noncomputable def b_lcm_a : ℕ := 10 ^ (760 / 3 / 3).to_nat

def p := (10 ^ (280 / 2 / 1)).to_nat
def q := (10 ^ (280 / 2 / 1)).to_nat

theorem math_proof_example : a_gcd_b + b_lcm_a = 1172 := by sorry

end math_proof_example_l106_106317


namespace correct_propositions_l106_106956

-- Define the propositions
def prop1 : Prop :=
  ∀ x : ℝ, 2*Real.cos(1/3*x + π/4)^2 - 1 = -Real.sin(2/3*x)

def prop2 : Prop :=
  ∃ α : ℝ, Real.sin α + Real.cos α = 3/2

def prop3 : Prop :=
  ∀ α β : ℝ, (0 < α ∧ α < β ∧ β < π/2) → Real.tan α < Real.tan β

def prop4 : Prop :=
  ∀ x : ℝ, Real.sin (2*x + 5*π/4) = Real.sin (2*(π/8) + 5*π/4 - 2*x)

def prop5 : Prop :=
  ∀ x y : ℝ, (x = π/12 ∧ y = 0) → Real.sin (2*x + π/3) = -Real.sin (2*(-x - π/12) + π/3)

-- The Lean statement asserting which propositions are correct
theorem correct_propositions : prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 ∧ ¬prop5 :=
by
  sorry

end correct_propositions_l106_106956


namespace sec_tan_equation_l106_106991

variable (x : ℝ)
noncomputable def sec (x : ℝ) : ℝ := 1 / (Real.cos x)
noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem sec_tan_equation : sec x + tan x = 3 → sec x - tan x = 1 / 3 :=
by
  assume h1 : sec x + tan x = 3
  have h2 : (sec x + tan x) * (sec x - tan x) = sec x * sec x - tan x * tan x := sorry
  have h3 : sec x * sec x - tan x * tan x = 1 := sorry
  sorry

end sec_tan_equation_l106_106991


namespace midpoint_coordinates_B_l106_106765

noncomputable def B : (ℕ × ℕ) := (1, 1)
noncomputable def I : (ℕ × ℕ) := (2, 4)
noncomputable def G : (ℕ × ℕ) := (5, 1)

def translate_point (p : ℕ × ℕ) (dx dy : ℤ) : (ℤ × ℤ) :=
  (p.1 + dx, p.2 + dy)

def rotate_point_90_clockwise_about (center p : (ℤ × ℤ)) : (ℤ × ℤ) :=
  let (cx, cy) := center
  let (px, py) := p
  (cx + (py - cy), cy - (px - cx))

noncomputable def B_translated := translate_point B 4 (-3)
noncomputable def G_translated := translate_point G 4 (-3)

def midpoint (p1 p2 : ℤ × ℤ) : (ℤ × ℤ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def M_translated := midpoint B_translated G_translated

noncomputable def M_rotated := rotate_point_90_clockwise_about B_translated M_translated

theorem midpoint_coordinates_B'G' :
  M_rotated = (5, 0) :=
by
  sorry

end midpoint_coordinates_B_l106_106765


namespace problem_sum_f_l106_106924

noncomputable def f (a : ℝ) [a_pos : 0 < a] (x : ℝ) : ℝ := a^x / (a^x + real.sqrt a)

theorem problem_sum_f (a : ℝ) [a_pos : 0 < a] : 
  ∑ k in finset.range 1000, f a ((k + 1 : ℝ) / 1001) = 500 :=
by
  sorry

end problem_sum_f_l106_106924


namespace employed_females_percentage_l106_106352

theorem employed_females_percentage (total_population : ℝ) (total_employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  total_employed_percentage = 0.7 →
  employed_males_percentage = 0.21 →
  total_population > 0 →
  (total_employed_percentage - employed_males_percentage) / total_employed_percentage * 100 = 70 :=
by
  intros h1 h2 h3
  -- Proof is omitted.
  sorry

end employed_females_percentage_l106_106352


namespace evaluate_expression_l106_106089

theorem evaluate_expression : ((3^4)^3 + 5) - ((4^3)^4 + 5) = -16245775 := by
  sorry

end evaluate_expression_l106_106089


namespace donald_laptop_cost_l106_106511

theorem donald_laptop_cost (original_price : ℕ) (reduction_percent : ℕ) (reduced_price : ℕ) (h1 : original_price = 800) (h2 : reduction_percent = 15) : reduced_price = 680 :=
by
  -- Definitions of the conditions
  have h3 : reduction_percent / 100 * original_price = 120 := sorry  -- Calculation of the discount (15/100)*800
  have h4 : original_price - 120 = 680 := sorry  -- Subtracting discount from original price
  -- Conclusion
  exact h4

end donald_laptop_cost_l106_106511


namespace hyperbola_eccentricity_range_l106_106008

-- Define the eccentricity of the hyperbola
def eccentricity (a b : ℝ) : ℝ := (Real.sqrt (a^2 + b^2)) / a

-- Define the conditions for the hyperbola
variables (a b e : ℝ) (h1 : a > 0) (h2 : b > 0) 
          (h3 : ∀ a b, eccentetric (a b) > sqrt  2) ) 
          (h4: ∀ a b, eccentricity (a b) < sqrt 10)

-- State the goal
theorem hyperbola_eccentricity_range : 
  (1, sqrt 2) ∧ (eccentricity h2 h2) < (sqrt 10) := sorry

end hyperbola_eccentricity_range_l106_106008


namespace trig_identity_proof_l106_106420

theorem trig_identity_proof :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 3 :=
by sorry

end trig_identity_proof_l106_106420


namespace loyalty_program_theorem_l106_106721

-- Definitions representing the conditions
def CentralBankInterestPromotion (cb: Type) : Prop := 
  ∀ (loyaltyProgram: Type), encouragesNationalAdoption(loyaltyProgram)

def CentralBankInterestStimulus (cb: Type) : Prop := 
  ∀ (loyaltyProgram: Type), stimulatesConsumerSpending(loyaltyProgram)

def BankBenefitLoyalty (bank: Type) : Prop := 
  ∀ (loyaltyProgram: Type), attractsAndRetainsCustomers(bank, loyaltyProgram)

def BankBenefitTransactionVolume (bank: Type) : Prop := 
  ∀ (loyaltyProgram: Type), increasesCardUsage(bank, loyaltyProgram)

def RegistrationRequirement (system: Type) : Prop :=
  ∀ (loyaltyProgram: Type), (collectsUserData(system, loyaltyProgram) ∧ preventsFraud(system, loyaltyProgram))

-- Theorem to prove the hypothesis
theorem loyalty_program_theorem
  (cb: Type) (bank: Type) (system: Type)
  (h1: CentralBankInterestPromotion cb)
  (h2: CentralBankInterestStimulus cb)
  (h3: BankBenefitLoyalty bank)
  (h4: BankBenefitTransactionVolume bank)
  (h5: RegistrationRequirement system):
  (∀ (loyaltyProgram: Type), 
    encouragesNationalAdoption(loyaltyProgram) →
    stimulatesConsumerSpending(loyaltyProgram) →
    attractsAndRetainsCustomers(bank, loyaltyProgram) →
    increasesCardUsage(bank, loyaltyProgram) →
    (collectsUserData(system, loyaltyProgram) ∧ preventsFraud(system, loyaltyProgram))) :=
by
  intros loyaltyProgram h_ena h_scs h_aar h_ic فرضية
  sorry

end loyalty_program_theorem_l106_106721


namespace cubic_identity_l106_106996

theorem cubic_identity (x : ℝ) (h : x + (1/x) = -3) : x^3 + (1/x^3) = -18 :=
by
  sorry

end cubic_identity_l106_106996


namespace expected_value_dodecahedral_die_l106_106788

theorem expected_value_dodecahedral_die : 
  let outcomes := (finset.range 12).map (λ n, n + 1)
  in (1 / 12: ℝ) * (outcomes.sum id) = 6.5 := by
  sorry

end expected_value_dodecahedral_die_l106_106788


namespace twentieth_term_in_sequence_l106_106898

theorem twentieth_term_in_sequence : 
  (let sequence_term (n : ℕ) : ℚ := 
     let denominator := Nat.find (λ m, m * (m + 1) / 2 ≥ n)
     let k := n - (denominator * (denominator - 1) / 2)
     (k : ℚ) / (denominator + 1 : ℚ)
   in sequence_term 20 = 6 / 7) :=
sorry

end twentieth_term_in_sequence_l106_106898


namespace f1_nth_iter_f2_nth_iter_f3_nth_iter_f4_nth_iter_l106_106926

-- (1) f(x) = x + c
def iterate_f1 (f : ℝ → ℝ) (c : ℝ) (n : ℕ) (x : ℝ) :=
  match n with
  | 0     => x
  | (n+1) => f (iterate_f1 f c n x)

theorem f1_nth_iter (c : ℝ) (n : ℕ) (x : ℝ) : iterate_f1 (λ x, x + c) c n x = x + n * c := 
  sorry

-- (2) f(x) = ax + b where a ≠ 1
def iterate_f2 (f : ℝ → ℝ) (a b : ℝ) (h : a ≠ 1) (n : ℕ) (x : ℝ) :=
  match n with
  | 0     => x
  | (n+1) => f (iterate_f2 f a b h n x)

theorem f2_nth_iter (a b : ℝ) (h : a ≠ 1) (n : ℕ) (x : ℝ) : 
  iterate_f2 (λ x, a * x + b) a b h n x = a ^ n * x + (1 - a ^ n) / (1 - a) * b := 
  sorry

-- (3) f(x) = x / (1 + ax)
def iterate_f3 (f : ℝ → ℝ) (a : ℝ) (n : ℕ) (x : ℝ) :=
  match n with
  | 0     => x
  | (n+1) => f (iterate_f3 f a n x)

theorem f3_nth_iter (a : ℝ) (n : ℕ) (x : ℝ) : iterate_f3 (λ x, x / (1 + a * x)) a n x = x / (1 + n * a * x) := 
  sorry

-- (4) f(x) = x^m
def iterate_f4 (f : ℝ → ℝ) (m : ℕ) (n : ℕ) (x : ℝ) :=
  match n with
  | 0     => x
  | (n+1) => f (iterate_f4 f m n x)

theorem f4_nth_iter (m : ℕ) (n : ℕ) (x : ℝ) : iterate_f4 (λ x, x ^ m) m n x = x ^ m ^ n := 
  sorry

end f1_nth_iter_f2_nth_iter_f3_nth_iter_f4_nth_iter_l106_106926


namespace find_quotient_l106_106288

theorem find_quotient
  (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 131) (h2 : divisor = 14) (h3 : remainder = 5)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 :=
by
  sorry

end find_quotient_l106_106288


namespace correct_subtraction_l106_106988

/-- Given a number n where subtracting 63 results in 8,
we aim to find the result of subtracting 36 from n
and proving that the result is 35. -/
theorem correct_subtraction (n : ℕ) (h : n - 63 = 8) : n - 36 = 35 :=
by
  sorry

end correct_subtraction_l106_106988


namespace calculate_difference_l106_106679

def f (x : ℝ) : ℝ := x + 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem calculate_difference :
  f (g 5) - g (f 5) = -2 := by
  sorry

end calculate_difference_l106_106679


namespace binom_18_10_l106_106441

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106441


namespace area_under_f_prime_area_value_l106_106599

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x
def f' (x : ℝ) : ℝ := (1 / x) + 2 * x - 3

theorem area_under_f_prime : 
  ∫ x in (1/2)..1, f' x = -(f 1 - f (1 / 2)) :=
by {
  sorry
}

theorem area_value :
  ∫ x in (1/2)..1, f' x = (3 / 4) - Real.log 2 :=
by {
  sorry
}

end area_under_f_prime_area_value_l106_106599


namespace dice_game_expected_winnings_zero_l106_106852

variable (die_roll : ℕ → ℕ)

def winnings : ℕ → ℤ
| 1 := 2
| 2 := 2
| 3 := 4
| 4 := 4
| 5 := -6
| 6 := -6
| _ := 0

noncomputable def expected_winnings : ℕ → ℚ :=
  (∑ i in Finset.range 6, (1 / 6 : ℚ) * (winnings (i + 1))) / 6

theorem dice_game_expected_winnings_zero : expected_winnings die_roll = 0 :=
by sorry

end dice_game_expected_winnings_zero_l106_106852


namespace scarves_per_box_l106_106345

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ := 8) 
  (mittens_per_box : ℕ := 6) 
  (total_clothing : ℕ := 80) 
  (total_mittens : ℕ := boxes * mittens_per_box) 
  (total_scarves : ℕ := total_clothing - total_mittens) 
  (scarves_per_box : ℕ := total_scarves / boxes) 
  : scarves_per_box = 4 := 
by 
  sorry

end scarves_per_box_l106_106345


namespace binom_18_10_l106_106433

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106433


namespace intersection_point_of_lines_is_correct_l106_106144

theorem intersection_point_of_lines_is_correct:
  ∃ (x y : ℝ), 
    let l1 := line.through_angle (-2, 0) (real.pi / 6) in
    let l2 := line.perpendicular_through (2, 0) l1 in
    (x, y) = (1, real.sqrt 3) ∧ l1.contains (x, y) ∧ l2.contains (x, y) :=
sorry

end intersection_point_of_lines_is_correct_l106_106144


namespace no_real_solution_l106_106518

-- Define the hypothesis: the sum of partial fractions
theorem no_real_solution : 
  ¬ ∃ x : ℝ, 
    (1 / ((x - 1) * (x - 3)) + 
     1 / ((x - 3) * (x - 5)) + 
     1 / ((x - 5) * (x - 7))) = 1 / 8 := 
by
  sorry

end no_real_solution_l106_106518


namespace parking_space_area_l106_106016

theorem parking_space_area
  (L : ℕ) (W : ℕ)
  (hL : L = 9)
  (hSum : 2 * W + L = 37) : L * W = 126 := 
by
  sorry

end parking_space_area_l106_106016


namespace length_MN_l106_106269

-- Definitions of the given conditions in the problem:
def rect_ABCD (AB BC : ℝ) := AB = 6 ∧ BC = 2
def perpendicular_to_DB (MN DB : ℝ) := MN ⊥ DB
def A_on_DM (A D M : point) := lies_on A (line_through D M)
def C_on_DN (C D N : point) := lies_on C (line_through D N)

-- Theorem statement proving the length of MN
theorem length_MN
  (AB BC : ℝ)
  (MN BD : ℝ)
  (A D M C N : point)
  (h_rect : rect_ABCD AB BC)
  (h_perp : perpendicular_to_DB MN BD)
  (h_A_on_DM : A_on_DM A D M)
  (h_C_on_DN : C_on_DN C D N)
  (h_BD : BD = sqrt (AB^2 + BC^2))
  : MN = 4 * sqrt 10 :=
sorry

end length_MN_l106_106269


namespace binom_18_10_l106_106473

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106473


namespace parallel_line_equation_perpendicular_line_equation_l106_106944

theorem parallel_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, 4 * x - y - 7 = 0) :=
sorry

theorem perpendicular_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, x + 4 * y - 6 = 0) :=
sorry

end parallel_line_equation_perpendicular_line_equation_l106_106944


namespace quadratic_inequality_real_solution_l106_106172

theorem quadratic_inequality_real_solution (a : ℝ) :
  (∃ x : ℝ, 2*x^2 + (a-1)*x + 1/2 ≤ 0) ↔ (a ≤ -1 ∨ 3 ≤ a) := 
sorry

end quadratic_inequality_real_solution_l106_106172


namespace ufo_convention_attendees_l106_106865

theorem ufo_convention_attendees 
  (F M : ℕ) 
  (h1 : F + M = 450) 
  (h2 : M = F + 26) : 
  M = 238 := 
sorry

end ufo_convention_attendees_l106_106865


namespace worker_C_work_rate_worker_C_days_l106_106694

theorem worker_C_work_rate (A B C: ℚ) (hA: A = 1/10) (hB: B = 1/15) (hABC: A + B + C = 1/4) : C = 1/12 := 
by
  sorry

theorem worker_C_days (C: ℚ) (hC: C = 1/12) : 1 / C = 12 :=
by
  sorry

end worker_C_work_rate_worker_C_days_l106_106694


namespace vertex_of_parabola_l106_106739

/-- The given parabola y = -3(x-1)^2 - 2 has its vertex at (1, -2). -/
theorem vertex_of_parabola : ∃ h k : ℝ, (h = 1 ∧ k = -2) ∧ ∀ x : ℝ, y = -3 * (x - h) ^ 2 + k :=
begin
  use [1, -2],
  split,
  { split; refl },
  { intro x,
    refl }
end

end vertex_of_parabola_l106_106739


namespace simplify_expression_l106_106274

theorem simplify_expression (w x : ℝ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w - 2 * x - 4 * x - 6 * x - 8 * x - 10 * x + 24 = 
  45 * w - 30 * x + 24 :=
by sorry

end simplify_expression_l106_106274


namespace find_a_l106_106608

theorem find_a (a : ℝ) (t : ℝ) :
  (4 = 1 + 3 * t) ∧ (3 = a * t^2 + 2) → a = 1 :=
by
  sorry

end find_a_l106_106608


namespace volume_of_solid_area_l106_106870

def volume_of_revolution_hyperbola (y1 y2 : ℝ) : ℝ :=
  have h : y1 < y2 := by linarith
  π * ∫ y in y1..y2, (4 : ℝ) / y^2

theorem volume_of_solid_area :
  volume_of_revolution_hyperbola 1 4 = 3 * Real.pi := 
by
  sorry

end volume_of_solid_area_l106_106870


namespace exists_smaller_similar_triangles_of_different_sizes_l106_106495

open Classical

noncomputable def isosceles_right_triangle : Type := Σ (A B C : ℝ × ℝ), 
  isosceles_right_triangle_property A B C

def isosceles_right_triangle_property (A B C : ℝ × ℝ) : Prop :=
  (A = (0, 0)) ∧ (B = (7, 0)) ∧ (C = (0, 7)) ∧
  (dist A B = dist A C)

theorem exists_smaller_similar_triangles_of_different_sizes : 
    ∀ (T : isosceles_right_triangle), ∃ (S : set (isosceles_right_triangle)), 
    ∀ t1 t2 ∈ S, t1 ≠ t2 → similar t1 t2 ∧ size t1 ≠ size t2 :=
by
  sorry

end exists_smaller_similar_triangles_of_different_sizes_l106_106495


namespace payment_first_trip_payment_second_trip_l106_106840

-- Define conditions and questions
variables {x y : ℝ}

-- Conditions: discounts and expenditure
def discount_1st_trip (x : ℝ) := 0.9 * x
def discount_2nd_trip (y : ℝ) := 300 * 0.9 + (y - 300) * 0.8

def combined_discount (x y : ℝ) := 300 * 0.9 + (x + y - 300) * 0.8

-- Given conditions as equations
axiom eq1 : discount_1st_trip x + discount_2nd_trip y - combined_discount x y = 19
axiom eq2 : x + y - (discount_1st_trip x + discount_2nd_trip y) = 67

-- The proof statements
theorem payment_first_trip : discount_1st_trip 190 = 171 := by sorry

theorem payment_second_trip : discount_2nd_trip 390 = 342 := by sorry

end payment_first_trip_payment_second_trip_l106_106840


namespace gambler_target_win_percentage_l106_106828

-- Define the initial conditions
def initial_games_played : ℕ := 20
def initial_win_rate : ℚ := 0.40

def additional_games_played : ℕ := 20
def additional_win_rate : ℚ := 0.80

-- Define the proof problem statement
theorem gambler_target_win_percentage 
  (initial_wins : ℚ := initial_win_rate * initial_games_played)
  (additional_wins : ℚ := additional_win_rate * additional_games_played)
  (total_games_played : ℕ := initial_games_played + additional_games_played)
  (total_wins : ℚ := initial_wins + additional_wins) :
  ((total_wins / total_games_played) * 100 : ℚ) = 60 := 
by
  -- Skipping the proof steps
  sorry

end gambler_target_win_percentage_l106_106828


namespace probability_at_least_one_white_ball_l106_106641

def balls : Finset ℕ := {0, 1, 2, 3, 4}  -- represent the 5 balls
def white_balls : Finset ℕ := {1, 2}  -- represent the white balls

def all_pairs : Finset (ℕ × ℕ) := balls.product balls
def valid_pairs : Finset (ℕ × ℕ) := all_pairs.filter (λ p, p.1 ≠ p.2 ∧ (p.1 ∈ white_balls ∨ p.2 ∈ white_balls))

theorem probability_at_least_one_white_ball :
  ((valid_pairs.card : ℚ) / (all_pairs.card : ℚ)) = 7 / 10 :=
by
  sorry

end probability_at_least_one_white_ball_l106_106641


namespace find_c_l106_106164

noncomputable def y (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c (c : ℝ) (h : ∃ a b : ℝ, a ≠ b ∧ y a c = 0 ∧ y b c = 0) :
  c = -2 ∨ c = 2 :=
by sorry

end find_c_l106_106164


namespace two_digits_satisfy_property_l106_106985

-- Define the concept of a two-digit number with its properties
def is_two_digit (N : ℕ) : Prop := N ≥ 10 ∧ N < 100

-- Define the property where the sum of N and the cube of the reversed digits' number is a perfect square
def satisfies_property (N : ℕ) : Prop :=
  ∃ t u : ℕ, N = 10 * t + u ∧ 
  ((10 * u + t)^3 + N) = n^2 ∧ ℕ.is_square ((10 * u + t)^3 + N)

-- Main theorem statement: the number of two-digit integers satisfying the property is 2
theorem two_digits_satisfy_property : 
  {N : ℕ | is_two_digit N ∧ satisfies_property N}.card = 2 := 
sorry

end two_digits_satisfy_property_l106_106985


namespace exists_rational_linear_function_l106_106969

theorem exists_rational_linear_function (y1 y2 y3 : ℝ) (h_distinct : y1 ≠ y2 ∧ y2 ≠ y3 ∧ y1 ≠ y3) :
  ∃ f : ℝ → ℝ, (f 0 = y1) ∧ (f 1 = y2) ∧ (f ⊤ = y3) :=
sorry

end exists_rational_linear_function_l106_106969


namespace contact_probability_l106_106543

variable (m : ℕ := 6) (n : ℕ := 7) (p : ℝ)

theorem contact_probability :
  let total_pairs := m * n in
  let probability_no_contact := (1 - p) ^ total_pairs in
  let probability_contact := 1 - probability_no_contact in
  probability_contact = 1 - (1 - p) ^ 42 :=
by
  -- This is where the proof would go
  sorry

end contact_probability_l106_106543


namespace inequality_a_inequality_b_inequality_c_inequality_d_l106_106718

variables {n : ℕ} {x : ℕ → ℝ}

noncomputable theory

-- Define the condition that all \(x_i > 0\)
def all_positive (x : ℕ → ℝ) := ∀ i, x i > 0

-- Define the condition that \(n > 0\)
def n_positive (n : ℕ) := n > 0

-- Prove: \( n(x_{1} + \ldots + x_{n}) \geq (\sqrt{x_{1}} + \ldots + \sqrt{x_{n}})^{2} \)
theorem inequality_a (h1 : all_positive x) (h2 : n_positive n) :
  n * (∑ i in finset.range n, x i) ≥ (∑ i in finset.range n, real.sqrt (x i)) ^ 2 :=
sorry

-- Prove: \( \frac{n^{3}}{(x_{1} + \ldots + x_{n})^{2}} \leq \frac{1}{x_{1}^{2}} + \ldots + \frac{1}{x_{n}^{2}} \)
theorem inequality_b (h1 : all_positive x) (h2 : n_positive n) :
  (n^3) / (∑ i in finset.range n, x i)^2 ≤ ∑ i in finset.range n, 1 / (x i)^2 :=
sorry

-- Prove: \( n x_{1} \ldots x_{n} \leq x_{1}^{n} + \ldots + x_{n}^{n} \)
theorem inequality_c (h1 : all_positive x) (h2 : n_positive n) :
  n * (∏ i in finset.range n, x i) ≤ ∑ i in finset.range n, (x i) ^ n :=
sorry

-- Prove: \( (x_{1} + \ldots + x_{n})\left(\frac{1}{x_{1}} + \ldots + \frac{1}{x_{n}}\right) \geq n^{2} \)
theorem inequality_d (h1 : all_positive x) (h2 : n_positive n) :
  (∑ i in finset.range n, x i) * (∑ i in finset.range n, 1 / (x i)) ≥ n^2 :=
sorry

end inequality_a_inequality_b_inequality_c_inequality_d_l106_106718


namespace geometric_transformed_sequence_l106_106677

noncomputable def is_geometric_sequence (T : ℕ → ℝ) : Prop :=
  ∃ r, r ≠ 1 ∧ ∀ n, T (n + 1) = r * T n

variable 
  (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (k : ℕ)
  (h1 : ∃ r, r ≠ 1 ∧ ∀ n, b (n + 1) = r * b n)
  (h2 : T 0 = 1)
  (h3 : ∀ n, T (n + 1) = T n * b (n + 1))
  (h4 : k > 0)

theorem geometric_transformed_sequence : 
  is_geometric_sequence (λ n, (T ((n + 1) * k)) / (T (n * k))) :=
  sorry

end geometric_transformed_sequence_l106_106677


namespace question1_question2_l106_106150

variable {n : ℕ}

-- Conditions
def Sn (n : ℕ) : ℕ := 2 * a n - n
def a (n : ℕ) : ℕ := 2^(n-1) - 1 -- Derived from solution but needed to solve the condition
def b (n : ℕ) : ℕ := n -- Derived from the arithmetic definition in the problem

-- Sequences
def T (n : ℕ) : ℕ := 1 - 1 / (n + 1)

theorem question1 : 
  ∀n:ℕ, n > 0 → Sn n = 2 * a n - n ∧ Sn (n-1) = 2 * a (n-1) - (n - 1) →
  (a (n + 1)) = 2 * (a n + 1) :=
by sorry

theorem question2 : 
  ∀n:ℕ, n > 0 → b 3 = a 2 ∧ b 7 = a 3 →
  (∑ i in range n, (1 / ((b i) * (b (i + 1))))) = T n :=
by sorry

end question1_question2_l106_106150


namespace minimum_games_played_l106_106644

theorem minimum_games_played (n : ℕ) (h1 : n = 16) :
  ∃ (games : ℕ), games = 56 ∧
  ∀ (A B C : fin n), 
    A ≠ B → B ≠ C → C ≠ A → 
    (played A B ∨ played B C ∨ played C A) :=
begin
  sorry
end

end minimum_games_played_l106_106644


namespace real_number_a_pure_imaginary_l106_106592

-- Definition of an imaginary number
def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- Given conditions and the proof problem statement
theorem real_number_a_pure_imaginary (a : ℝ) :
  pure_imaginary (⟨(a + 1) / 2, (1 - a) / 2⟩) → a = -1 :=
by
  sorry

end real_number_a_pure_imaginary_l106_106592


namespace binom_18_10_l106_106486

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106486


namespace count_multiples_of_5_or_7_not_35_l106_106181

theorem count_multiples_of_5_or_7_not_35 (n : ℕ) (h : n = 2333) :
  let multiples_of_5 := λ x, ∃ k, 1 ≤ k ∧ k ≤ x ∧ k % 5 = 0,
      multiples_of_7 := λ x, ∃ k, 1 ≤ k ∧ k ≤ x ∧ k % 7 = 0,
      multiples_of_35 := λ x, ∃ k, 1 ≤ k ∧ k ≤ x ∧ k % 35 = 0 in
  (nat.card { k | multiples_of_5 k ∨ multiples_of_7 k } -
   nat.card { k | multiples_of_35 k } = 733) := sorry

end count_multiples_of_5_or_7_not_35_l106_106181


namespace parallel_vectors_x_value_l106_106618

theorem parallel_vectors_x_value : 
  ∀ x : ℝ,
    let a := (2 : ℝ, 1 : ℝ),
        b := (x, -2 : ℝ) 
    in (a.2 / a.1 = b.2 / b.1) → x = -4 :=
by
  intros x a b h
  sorry

end parallel_vectors_x_value_l106_106618


namespace not_in_range_l106_106080

-- Definitions for g and conditions
variable {p q r s : ℝ} (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)

def g (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

-- Conditions
variable (h11 : g 11 = 11)
variable (h35 : g 35 = 35)
variable (h75 : g 75 = 75)
variable (h_g_g : ∀ x, x ≠ -s / r → g (g x) = x)

-- The main statement to be proved
theorem not_in_range : ∀ y, ∃ x, g x = y ∨ y = p / r :=
sorry

end not_in_range_l106_106080


namespace final_jacket_price_l106_106046

def original_price : ℝ := 25
def first_discount : ℝ := 0.40
def second_discount : ℝ := 0.25
def final_discount : ℝ := 0.10

theorem final_jacket_price : 
  let first_disc_price := original_price * (1 - first_discount) in
  let second_disc_price := first_disc_price * (1 - second_discount) in
  let final_price := second_disc_price * (1 - final_discount) in
  Float.round (final_price * 100) / 100 = 10.13 :=
by
  sorry

end final_jacket_price_l106_106046


namespace wall_length_is_260_l106_106822

-- Define the dimensions of a brick in cm.
def brick_length_cm : ℝ := 20
def brick_width_cm : ℝ := 10
def brick_height_cm : ℝ := 7.5

-- Convert the volume of a brick to m³.
def brick_volume_m3 : ℝ := (brick_length_cm / 100) * (brick_width_cm / 100) * (brick_height_cm / 100)

-- Define the dimensions of the wall in meters.
def wall_height_m : ℝ := 2
def wall_width_m : ℝ := 0.75
def number_of_bricks : ℕ := 26000

-- Total volume of bricks required.
def total_brick_volume_m3 : ℝ := brick_volume_m3 * number_of_bricks

-- Define the length of the wall.
noncomputable def wall_length_m : ℝ := total_brick_volume_m3 / (wall_height_m * wall_width_m)

-- Theorem stating the length of the wall.
theorem wall_length_is_260 : wall_length_m = 260 :=
by
  sorry

end wall_length_is_260_l106_106822


namespace max_value_of_expression_l106_106244

theorem max_value_of_expression (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ 7 * Real.sqrt 2 :=
sorry

end max_value_of_expression_l106_106244


namespace relationship_among_abc_l106_106919

noncomputable def a : ℝ := 0.7 ^ 0.4
noncomputable def b : ℝ := 0.4 ^ 0.7
noncomputable def c : ℝ := 0.4 ^ 0.4

theorem relationship_among_abc : a > c ∧ c > b := by
  sorry

end relationship_among_abc_l106_106919


namespace lemon_juice_volume_is_7_07_l106_106380

noncomputable def volume_of_lemon_juice (height : ℝ) (diameter : ℝ) (ratio_lj_oj : ℝ) : ℝ :=
  let full_volume := Real.pi * (diameter / 2) ^ 2 * (height / 3)
  in (1 / ratio_lj_oj) * full_volume

theorem lemon_juice_volume_is_7_07 :
  volume_of_lemon_juice 9 3 2 = 7.07 :=
by
  have h_full_volume : Real.pi * (3 / 2) ^ 2 * (9 / 3) = 6.75 * Real.pi := by sorry
  have h_lemon_ratio : (1 / 3) = 1 / (1 + 2) := by sorry
  rw [volume_of_lemon_juice, h_full_volume, h_lemon_ratio]
  have h_volume : 6.75 * Real.pi / 3 = 2.25 * Real.pi := by sorry
  have h_result : 2.25 * Real.pi = 7.07 := by sorry
  exact h_result

end lemon_juice_volume_is_7_07_l106_106380


namespace total_earnings_l106_106086

-- Define the earnings in each month
def earnings_july : ℝ := 150
def earnings_august (july_earnings : ℝ) : ℝ := (3 * july_earnings) / 0.8
def earnings_september (august_earnings : ℝ) : ℝ := (2 * august_earnings) / 1.2
def earnings_october (september_earnings : ℝ) : ℝ := september_earnings + 0.1 * september_earnings
def earnings_november (october_earnings : ℝ) : ℝ := 0.95 * october_earnings

-- Prove the total earnings over five months
theorem total_earnings :
  let july := earnings_july
  let august := earnings_august july
  let september := earnings_september august
  let october := earnings_october september
  let november := earnings_november october
  july + august + september + october + november = 3661.44 :=
by
  let july := earnings_july
  let august := earnings_august july
  let september := earnings_september august
  let october := earnings_october september
  let november := earnings_november october
  have h_july : july = 150 := rfl
  have h_august : august = (3 * 150) / 0.8 := rfl
  have h_september : september = (2 * ((3 * 150) / 0.8)) / 1.2 := rfl
  have h_october : october = ((2 * ((3 * 150) / 0.8)) / 1.2) + 0.1 * ((2 * ((3 * 150) / 0.8)) / 1.2) := rfl
  have h_november : november = 0.95 * (((2 * ((3 * 150) / 0.8)) / 1.2) + 0.1 * (((2 * ((3 * 150) / 0.8)) / 1.2))) := rfl
  calc
    july + august + september + october + november
    = 150 + (3 * 150 / 0.8) + (2 * (3 * 150 / 0.8) / 1.2) + ((2 * (3 * 150 / 0.8) / 1.2) + 0.1 * (2 * (3 * 150 / 0.8) / 1.2)) + 0.95 * (((2 * (3 * 150 / 0.8)) / 1.2) + 0.1 * (((2 * (3 * 150 / 0.8)) / 1.2)))
    = 150 + 562.50 + 937.50 + 1031.25 + 979.69 := sorry

end total_earnings_l106_106086


namespace relationship_f_2011_2014_l106_106148

noncomputable def quadratic_func : Type := ℝ → ℝ

variable (f : quadratic_func)

-- The function is symmetric about x = 2013
axiom symmetry (x : ℝ) : f (2013 + x) = f (2013 - x)

-- The function opens upward (convexity)
axiom opens_upward (a b : ℝ) : f ((a + b) / 2) ≤ (f a + f b) / 2

theorem relationship_f_2011_2014 :
  f 2011 > f 2014 := 
sorry

end relationship_f_2011_2014_l106_106148


namespace range_of_a_if_union_conditions_l106_106113

variable (a : ℝ)
def A := {x : ℝ | 2 ≤ x ∧ x ≤ 6}
def B := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}

theorem range_of_a_if_union_conditions : A ∪ B = A → a ≥ 1 :=
by
  sorry

end range_of_a_if_union_conditions_l106_106113


namespace max_value_pq_qr_rs_sp_l106_106764

def max_pq_qr_rs_sp (p q r s : ℕ) : ℕ :=
  p * q + q * r + r * s + s * p

theorem max_value_pq_qr_rs_sp :
  ∀ (p q r s : ℕ), (p = 1 ∨ p = 5 ∨ p = 3 ∨ p = 6) → 
                    (q = 1 ∨ q = 5 ∨ q = 3 ∨ q = 6) →
                    (r = 1 ∨ r = 5 ∨ r = 3 ∨ r = 6) → 
                    (s = 1 ∨ s = 5 ∨ s = 3 ∨ s = 6) →
                    p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
                    max_pq_qr_rs_sp p q r s ≤ 56 := by
  sorry

end max_value_pq_qr_rs_sp_l106_106764


namespace contact_prob_correct_l106_106534

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end contact_prob_correct_l106_106534


namespace gallery_total_photos_l106_106695

noncomputable def original_photos : ℕ := 1200
noncomputable def first_day_multiplier : ℚ := 3 / 5
noncomputable def additional_photos_second_day : ℕ := 230

noncomputable def first_day_photos : ℕ := (first_day_multiplier * original_photos).toNat
noncomputable def second_day_photos : ℕ := first_day_photos + additional_photos_second_day
noncomputable def total_trip_photos : ℕ := first_day_photos + second_day_photos
noncomputable def final_total_photos : ℕ := original_photos + total_trip_photos

theorem gallery_total_photos : final_total_photos = 2870 := by
  sorry

end gallery_total_photos_l106_106695


namespace books_returned_thursday_is_correct_l106_106340

-- Define the conditions
def books_on_wednesday_morning : ℕ := 98
def books_checked_out_wednesday : ℕ := 43
def books_checked_out_thursday : ℕ := 5
def books_returned_friday : ℕ := 7
def books_on_friday : ℕ := 80

-- Calculate number of books Suzy had after Wednesday
def books_after_wednesday := books_on_wednesday_morning - books_checked_out_wednesday

-- Define the number of books returned on Thursday
def books_returned_thursday : ℕ := sorry 

-- Calculate number of books Suzy had after Thursday
def books_after_thursday := books_after_wednesday + books_returned_thursday - books_checked_out_thursday

-- Calculate number of books Suzy had after Friday
def books_after_friday := books_after_thursday + books_returned_friday

-- Prove that the number of books returned on Thursday is 23
theorem books_returned_thursday_is_correct : books_returned_thursday = 23 :=
by
  have h1 : books_after_wednesday = 55 := by
    calc 
      books_after_wednesday = 98 - 43 : rfl
      ... = 55 : by norm_num
  have h2 : books_after_thursday = books_after_wednesday + books_returned_thursday - 5 := rfl
  have h3 : books_after_friday = books_after_thursday + books_returned_friday := rfl
  have h4 : books_after_friday = 80 := by
    calc
      books_after_friday = books_after_thursday + 7 : rfl
      ... = (books_after_wednesday + books_returned_thursday - 5) + 7 : by rw [h2]
      ... = 55 + books_returned_thursday + 2 : by rw [h1]; ring
      ... = 57 + books_returned_thursday : by ring
  show books_returned_thursday = 23, from
    calc 
      books_returned_thursday = 80 - 57 : by rw [←h4, add_comm, add_sub_cancel']
      ... = 23 : by norm_num 

end books_returned_thursday_is_correct_l106_106340


namespace binom_18_10_l106_106428

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106428


namespace feet_of_altitudes_of_excircle_centers_l106_106687

variables {A B C O1 O2 O3 : Type*}

-- Define the centers of the excircles
def is_excircumcenter_of (O : Type*) (Δ : Type*) (a b c : Type*) : Prop := sorry

-- Define feet of the altitudes
def is_feet_of_altitudes (A B C O1 O2 O3 : Type*) : Prop := sorry

-- State the theorem
theorem feet_of_altitudes_of_excircle_centers 
  (hO1 : is_excircumcenter_of O1 A B C)
  (hO2 : is_excircumcenter_of O2 A B C)
  (hO3 : is_excircumcenter_of O3 A B C)
  : is_feet_of_altitudes A B C O1 O2 O3 := sorry

end feet_of_altitudes_of_excircle_centers_l106_106687


namespace comics_problem_l106_106863

theorem comics_problem (total_pages : ℕ) (pages_per_comic : ℕ) (total_comics_in_box : ℕ) (repaired_pages : ℕ) (comics_after_repair : ℕ) (h1 : total_pages = 150) (h2 : pages_per_comic = 25) (h3 : comics_after_repair = 11) :
  let repaired_comics := repaired_pages / pages_per_comic,
  repaired_pages = total_pages → total_pages / pages_per_comic = 6 → comics_after_repair - 6 = 5 :=
by {
  sorry
}

end comics_problem_l106_106863


namespace trains_initial_positions_after_target_time_l106_106293

-- Define constants for the traversal times
def red_traversal := 7
def blue_traversal := 8
def green_traversal := 9

-- Define the round trip times
def red_round_trip := red_traversal * 2
def blue_round_trip := blue_traversal * 2
def green_round_trip := green_traversal * 2

-- Define the target time in minutes
def target_time := 2016

-- Define a function to compute the least common multiple (LCM)
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Compute the LCM of the round trip times
def all_trains_lcm := lcm (lcm red_round_trip blue_round_trip) green_round_trip

-- Prove that after 2016 minutes, trains are at their initial stations
theorem trains_initial_positions_after_target_time :
  (target_time % all_trains_lcm) = 0 :=
by 
  have h1 : red_round_trip = 14 := rfl
  have h2 : blue_round_trip = 16 := rfl
  have h3 : green_round_trip = 18 := rfl
  have h4 : all_trains_lcm = Nat.lcm (Nat.lcm 14 16) 18 := rfl
  
  -- Calculation of LCM
  have h5 : Nat.lcm 14 16 = 112 := by sorry
  have h6 : all_trains_lcm = Nat.lcm 112 18 := by sorry
  have h7 : all_trains_lcm = 1008 := by sorry
  
  -- Conclude the proof
  calc
  target_time % all_trains_lcm 
      = 2016 % 1008 : by rw h7
  ... = 0 : by sorry

end trains_initial_positions_after_target_time_l106_106293


namespace triangle_inequality_l106_106931

variables (a b c : ℝ)

theorem triangle_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b > c) (h₄ : b + c > a) (h₅ : c + a > b) :
  (|a^2 - b^2| / c) + (|b^2 - c^2| / a) ≥ (|c^2 - a^2| / b) :=
by
  sorry

end triangle_inequality_l106_106931


namespace tank_depth_unique_l106_106782

theorem tank_depth_unique
  (rate : ℝ)
  (length width depth : ℝ)
  (time : ℝ)
  (h_rate : rate = 4)
  (h_length : length = 6)
  (h_width : width = 4)
  (h_time : time = 18)
  (h_volume : rate * time = length * width * depth) :
  depth = 3 :=
by {
  rw [h_rate, h_length, h_width, h_time] at h_volume,
  simp at h_volume,
  exact sorry
}

end tank_depth_unique_l106_106782


namespace set_intersection_l106_106236

noncomputable def A : Set ℤ := {-1, 0, 1, 2}

noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log 2 (4 - x^2) ∧ -2 < x ∧ x < 2}

theorem set_intersection : A ∩ B = {-1, 0, 1} :=
  sorry

end set_intersection_l106_106236


namespace equal_labels_l106_106816

theorem equal_labels (a : Fin 99 → ℕ) (h₁ : ∀ i, 1 ≤ a i ∧ a i ≤ 99)
  (h₂ : ∀ (s : Finset (Fin 99)), s.Nonempty → (∑ i in s, a i) % 100 ≠ 0) :
  ∃ b, ∀ i, a i = b := by
sorry

end equal_labels_l106_106816


namespace cost_effective_for_3000_cost_equal_at_2500_l106_106656

def cost_company_A (x : Nat) : Nat :=
  2 * x / 10 + 500

def cost_company_B (x : Nat) : Nat :=
  4 * x / 10

theorem cost_effective_for_3000 : cost_company_A 3000 < cost_company_B 3000 := 
by {
  sorry
}

theorem cost_equal_at_2500 : cost_company_A 2500 = cost_company_B 2500 := 
by {
  sorry
}

end cost_effective_for_3000_cost_equal_at_2500_l106_106656


namespace find_Q_over_P_l106_106754

theorem find_Q_over_P (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -7 → x ≠ 0 → x ≠ 5 →
    (P / (x + 7 : ℝ) + Q / (x^2 - 6 * x) = (x^2 - 6 * x + 14) / (x^3 + x^2 - 30 * x))) :
  Q / P = 12 :=
  sorry

end find_Q_over_P_l106_106754


namespace find_y_l106_106325

-- Given conditions
variable (CE CD AC BC : ℝ)
variable (h1 h2 : CE / CD = 3 / 4)
variable (h3 : CE = 3)
variable (h4 : CD = 4)
variable (h5 : AC = 10)
variable (y : ℝ)

-- Triangle similarity
theorem find_y : y = 31 / 3 :=
by
  have proportion := (h1) ▸ (h2)  -- Establish proportion CE/CD = AC/BC
  calc
    3/4 = 10/(y + 3) : by sorry -- Given proportion 3/4 = 10/(y + 3)
    3 * (y + 3) = 40 : by sorry -- Cross-multiplying
    3y + 9 = 40 : by sorry -- Simplifying
    3y = 31 : by sorry -- Subtracting 9 from both sides
    y = 31/3 : by sorry -- Dividing both sides by 3

end find_y_l106_106325


namespace c1_minus_c4_eq_9_l106_106594

def f (x : ℝ) (c₁ c₂ c₃ c₄ : ℝ) : ℝ :=
  (x^2 - 8 * x + c₁) * (x^2 - 8 * x + c₂) * (x^2 - 8 * x + c₃) * (x^2 - 8 * x + c₄)

def is_natural_star (x : ℝ) : Prop := ∃ n : ℕ, ↑n = x ∧ n > 0

noncomputable def set_M : set ℝ :=
  {x | f x c₁ c₂ c₃ c₄ = 0 ∧ is_natural_star x}

variables {c₁ c₂ c₃ c₄ : ℝ}
  (h1 : c₁ ≥ c₂)
  (h2 : c₂ ≥ c₃)
  (h3 : c₃ ≥ c₄)
  (h4 : c₁ = 16)
  (h5 : c₄ = 7)

theorem c1_minus_c4_eq_9 : c₁ - c₄ = 9 :=
sorry

end c1_minus_c4_eq_9_l106_106594


namespace flat_odot_length_correct_l106_106075

noncomputable def sides : ℤ × ℤ × ℤ := (4, 5, 6)

noncomputable def semiperimeter (a b c : ℤ) : ℚ :=
  (a + b + c) / 2

noncomputable def length_flat_odot (a b c : ℤ) : ℚ :=
  (semiperimeter a b c) - b

theorem flat_odot_length_correct : length_flat_odot 4 5 6 = 2.5 := by
  sorry

end flat_odot_length_correct_l106_106075


namespace field_trip_bread_pieces_l106_106845

theorem field_trip_bread_pieces :
  (students_per_group : ℕ) (num_groups : ℕ) (sandwiches_per_student : ℕ) (pieces_per_sandwich : ℕ)
  (H1 : students_per_group = 6)
  (H2 : num_groups = 5)
  (H3 : sandwiches_per_student = 2)
  (H4 : pieces_per_sandwich = 2)
  : 
  let total_students := num_groups * students_per_group in
  let total_sandwiches := total_students * sandwiches_per_student in
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich in
  total_pieces_bread = 120 :=
by
  let total_students := num_groups * students_per_group
  let total_sandwiches := total_students * sandwiches_per_student
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich
  sorry

end field_trip_bread_pieces_l106_106845


namespace bus_speed_in_kmph_l106_106823

-- Definitions based on the conditions
def distance : ℝ := 900.072
def time : ℝ := 30
def speed_in_m_per_s : ℝ := distance / time
def conversion_factor : ℝ := 3.6

-- The statement to be proven
theorem bus_speed_in_kmph : (speed_in_m_per_s * conversion_factor) = 108.00864 := by
  sorry

end bus_speed_in_kmph_l106_106823


namespace binom_18_10_l106_106435

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106435


namespace sqrt_expression_l106_106107

noncomputable def problem_sequence (x₁ d : ℝ) : ℕ → ℝ
| 0       => x₁
| (n+1)   => x₁ + n * d

theorem sqrt_expression (x₁ d : ℝ) (n : ℕ) (h_dis : ∀ m k, m ≠ k → problem_sequence x₁ d m ≠ problem_sequence x₁ d k)
  (h_rel : ∀ n, let x := problem_sequence x₁ d n; x = (problem_sequence x₁ d (n-1) + 298 * x + problem_sequence x₁ d (n+1)) / 300)
  : sqrt((problem_sequence x₁ d 2023 - problem_sequence x₁ d 1) / 2021 * 2022 / (problem_sequence x₁ d 2023 - x₁)) - 2023 = -2022 := 
sorry

end sqrt_expression_l106_106107


namespace expression_range_l106_106935

open Real -- Open the real number namespace

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) : 
  0 ≤ (x * y - x) / (x^2 + (y - 1)^2) ∧ (x * y - x) / (x^2 + (y - 1)^2) ≤ 12 / 25 :=
sorry -- Proof to be filled in.

end expression_range_l106_106935


namespace total_tires_correct_l106_106404

-- Define the number of vehicles
def total_vehicles : ℕ := 24

-- Define the fraction of vehicles that are motorcycles
def motorcycles_fraction : ℚ := 1 / 3

-- Define the fraction of cars that have a spare tire
def cars_with_spare_tire_fraction : ℚ := 1 / 4

-- Define the total number of motorcycles
def number_of_motorcycles : ℕ := (motorcycles_fraction * total_vehicles).to_nat

-- Define the total number of cars
def number_of_cars : ℕ := total_vehicles - number_of_motorcycles

-- Define the number of cars with a spare tire
def cars_with_spare_tire : ℕ := (cars_with_spare_tire_fraction * number_of_cars).to_nat

-- Define the number of cars without a spare tire
def cars_without_spare_tire : ℕ := number_of_cars - cars_with_spare_tire

-- Calculate the total number of tires
def total_tires : ℕ := 
  (number_of_motorcycles * 2) + 
  (cars_with_spare_tire * 5) + 
  (cars_without_spare_tire * 4)

-- Theorem statement to prove
theorem total_tires_correct : total_tires = 84 := by
  sorry

end total_tires_correct_l106_106404


namespace sin_arcsin_add_arctan_add_arccos_eq_l106_106065

theorem sin_arcsin_add_arctan_add_arccos_eq :
  sin (Real.arcsin (4/5) + Real.arctan (3/2) + Real.arccos (1/3)) = (17 - 12 * Real.sqrt 2) / (15 * Real.sqrt 13) :=
by
  sorry

end sin_arcsin_add_arctan_add_arccos_eq_l106_106065


namespace problem_sign_of_trig_product_l106_106186

open Real

theorem problem_sign_of_trig_product (θ : ℝ) (hθ : π / 2 < θ ∧ θ < π) :
  sin (cos θ) * cos (sin (2 * θ)) < 0 :=
sorry

end problem_sign_of_trig_product_l106_106186


namespace solve_quadratic_l106_106305

theorem solve_quadratic (x : ℝ) : x^2 = x ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solve_quadratic_l106_106305


namespace B_subscribed_fraction_correct_l106_106854

-- Define the total capital and the shares of A, C
variables (X : ℝ) (profit : ℝ) (A_share : ℝ) (C_share : ℝ)

-- Define the conditions as given in the problem
def A_capital_share := 1 / 3
def C_capital_share := 1 / 5
def total_profit := 2430
def A_profit_share := 810

-- Define the calculation of B's share
def B_capital_share := 1 - (A_capital_share + C_capital_share)

-- Define the expected correct answer for B's share
def expected_B_share := 7 / 15

-- Theorem statement
theorem B_subscribed_fraction_correct :
  B_capital_share = expected_B_share :=
by
  sorry

end B_subscribed_fraction_correct_l106_106854


namespace binom_18_10_l106_106484

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106484


namespace Victor_lives_now_l106_106798

theorem Victor_lives_now : ∃ L : ℤ, 14 - L = 12 ∧ L = 2 := by
  existsi 2
  split
  . exact rfl
  . exact rfl

end Victor_lives_now_l106_106798


namespace binom_18_10_l106_106451

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106451


namespace orthocenter_symmetric_to_A4_l106_106373

theorem orthocenter_symmetric_to_A4
  (C : ℝ)
  (A1 A2 A3 A4 : ℝ × ℝ)
  (t1 t2 t3 t4 : ℝ)
  (hA1 : A1 = (t1, C / t1))
  (hA2 : A2 = (t2, C / t2))
  (hA3 : A3 = (t3, C / t3))
  (hA4 : A4 = (t4, C / t4))
  (hCircle : ∀ (t : ℝ), (∃ a b r : ℝ, (t - a)^2 + (C / t - b)^2 = r^2)) :
  let O := (0, 0) in
  let orthocenterA123 := 
    let Hx := (A1.1 + A2.1 + A3.1) / 3 in
    let Hy := (A1.2 + A2.2 + A3.2) / 3 in
    (Hx, Hy)
  in
  orthocenterA123 = (-A4.1, -A4.2) := sorry

end orthocenter_symmetric_to_A4_l106_106373


namespace find_a_in_geometric_sequence_l106_106567

theorem find_a_in_geometric_sequence (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = 3^(n+1) + a) →
  (∃ a, ∀ n, S n = 3^(n+1) + a ∧ (18 : ℝ) ^ 2 = (S 1 - (S 1 - S 2)) * (S 2 - S 3) → a = -3) := 
by
  sorry

end find_a_in_geometric_sequence_l106_106567


namespace service_cost_is_correct_l106_106205

def service_cost_per_vehicle(cost_per_liter: ℝ)
                            (num_minivans: ℕ) 
                            (num_trucks: ℕ)
                            (total_cost: ℝ) 
                            (minivan_tank_liters: ℝ)
                            (truck_size_increase_pct: ℝ) 
                            (total_fuel: ℝ) 
                            (total_fuel_cost: ℝ) 
                            (total_service_cost: ℝ)
                            (num_vehicles: ℕ) 
                            (service_cost_per_vehicle: ℝ) : Prop :=
  cost_per_liter = 0.70 ∧
  num_minivans = 4 ∧
  num_trucks = 2 ∧
  total_cost = 395.4 ∧
  minivan_tank_liters = 65 ∧
  truck_size_increase_pct = 1.2 ∧
  total_fuel = (4 * minivan_tank_liters) + (2 * (minivan_tank_liters * (1 + truck_size_increase_pct))) ∧
  total_fuel_cost = total_fuel * cost_per_liter ∧
  total_service_cost = total_cost - total_fuel_cost ∧
  num_vehicles = num_minivans + num_trucks ∧
  service_cost_per_vehicle = total_service_cost / num_vehicles

-- Now, we state the theorem we want to prove.
theorem service_cost_is_correct :
  service_cost_per_vehicle 0.70 4 2 395.4 65 1.2 546 382.2 13.2 6 2.2 :=
by {
    sorry
}

end service_cost_is_correct_l106_106205


namespace angle_KCL_45_degrees_l106_106264

-- Define constants and conditions
variables {A B C K L : Type}
variables [IsoscelesRightTriangle A B C]
variables [PointsOnHypotenuse K L]
variables [Ratios AK KL LB : ℝ]

-- Define the ratios between the segments on the hypotenuse
axiom ak_kl_lb_ratio : AK / KL = 1 / 2
axiom kl_lb_ratio : KL / LB = 2 / sqrt 3

-- Goal: ∠KCL = 45 degrees
theorem angle_KCL_45_degrees : ∠KCL = 45 :=
  sorry

end angle_KCL_45_degrees_l106_106264


namespace nth_equation_sum_series_ab_sum_series_l106_106262

-- (1)
theorem nth_equation (n : ℕ) (hn : n ≠ 0): (1 : ℝ) / (n * (n + 1)) = (1 : ℝ) / n - (1 : ℝ) / (n + 1) := sorry

-- (2)
theorem sum_series_ab (a b : ℝ) (h1 : |a * b - 2| + (b - 1)^2 = 0) : 
  ∑ n in finset.range 2024, 1 / ((a + n) * (b + n)) = 2024 / 2025 := sorry

-- (3)
theorem sum_series : 
  ∑ n in finset.range 1011, 1 / (2 * (2 * n + 4)) = 1011 / 4048 := sorry

end nth_equation_sum_series_ab_sum_series_l106_106262


namespace geometric_b_sequence_general_formula_for_a_l106_106256

noncomputable def sequence_a : ℕ → ℝ
| 0     := 1
| 1     := 5 / 3
| (n+2) := (5 * sequence_a (n+1) - 2 * sequence_a n) / 3

def sequence_b (n : ℕ) : ℝ := sequence_a (n + 1) - sequence_a n

theorem geometric_b_sequence (n : ℕ) : sequence_b (n + 1) = (2 / 3) * sequence_b n :=
by sorry

theorem general_formula_for_a (n : ℕ) : sequence_a n = 3 - 3 * (2 / 3) ^ n :=
by sorry

end geometric_b_sequence_general_formula_for_a_l106_106256


namespace binom_18_10_eq_43758_l106_106466

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106466


namespace average_speed_l106_106353

/--
On the first day of her vacation, Louisa traveled 100 miles.
On the second day, traveling at the same average speed, she traveled 175 miles.
If the 100-mile trip took 3 hours less than the 175-mile trip,
prove that her average speed (in miles per hour) was 25.
-/
theorem average_speed (v : ℝ) (h1 : 100 / v + 3 = 175 / v) : v = 25 :=
by 
  sorry

end average_speed_l106_106353


namespace limit_of_S_over_r_squared_l106_106649

noncomputable def circle_limit (O P A B : ℝ → ℝ × ℝ) (r : ℝ) : Prop :=
 ∀ ε > 0, ∃ δ > 0, ∀ t > δ, 
  let S := (O t).fst ^ 2 - r ^ 2 in
  (S / r^2) > ε

theorem limit_of_S_over_r_squared 
  (O P A B : ℝ → ℝ × ℝ) (r : ℝ) (h_circle: ∀ t, dist (O t) (A t) = r ∧ dist (O t) (B t) = r)
  (h_tangent: ∀ t, dist (P t) (A t) = dist (P t) (B t))
  (h_parallel: ∀ t, (A t).snd = (B t).snd)
  (h_geometric: ∀ t, ∃ S, S = (P t).fst^2 - r^2) :
  circle_limit O P A B r :=
by
  sorry

end limit_of_S_over_r_squared_l106_106649


namespace same_terminal_side_l106_106052

theorem same_terminal_side
  (k : ℤ)
  (angle1 := (π / 5))
  (angle2 := (21 * π / 5)) :
  ∃ k : ℤ, angle2 = 2 * k * π + angle1 := by
  sorry

end same_terminal_side_l106_106052


namespace maurice_rides_l106_106702

theorem maurice_rides (M : ℕ) 
    (h1 : ∀ m_attended : ℕ, m_attended = 8)
    (h2 : ∀ matt_other : ℕ, matt_other = 16)
    (h3 : ∀ total_matt : ℕ, total_matt = matt_other + m_attended)
    (h4 : total_matt = 3 * M) : M = 8 :=
by 
  sorry

end maurice_rides_l106_106702


namespace colton_greatest_groups_l106_106070

/-- Colton has 24 blue marbles, 17 white marbles, 13 red marbles, 7 green marbles, and 5 yellow marbles. 
    The greatest number of identical groups Colton can make without any marbles left over is 1. -/
theorem colton_greatest_groups :
  ∀ (blue white red green yellow : ℕ),
  blue = 24 → white = 17 → red = 13 → green = 7 → yellow = 5 →
  nat.gcd (nat.gcd (nat.gcd (nat.gcd blue white) red) green) yellow = 1 :=
by intros blue white red green yellow h1 h2 h3 h4 h5
   rw [h1, h2, h3, h4, h5]
   simp [nat.gcd]
   sorry

end colton_greatest_groups_l106_106070


namespace tan_cot_product_l106_106142

theorem tan_cot_product (α : ℝ) (h : sin (2 * α) = 2 * sin 4) : 
  tan (α + 2 * real.pi / 90) * cot (α - 2 * real.pi / 90) = 3 :=
sorry

end tan_cot_product_l106_106142


namespace kristy_initial_cookies_l106_106665

-- Define the initial conditions
def initial_cookies (total_cookies_left : Nat) (c1 c2 c3 : Nat) (c4 c5 c6 : Nat) : Nat :=
  total_cookies_left + c1 + c2 + c3 + c4 + c5 + c6

-- Now we can state the theorem
theorem kristy_initial_cookies :
  initial_cookies 6 5 5 3 1 2 = 22 :=
by
  -- Proof is omitted
  sorry

end kristy_initial_cookies_l106_106665


namespace find_p_plus_q_l106_106224

-- Definitions of the conditions from problem
variables {A B C D : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variables (triangle_ABC : Triangle A B C)
variables (right_angle_C : IsRightAngle (Angle A C B))
variables (altitude_from_C_meets_AB_at_D : AltitudeFromPoint (Angle A C B) D)
variables (integers_sides : (∃ a b c : ℕ, triangle_ABC.has_sides a b c))
variables (BD_length : side_length D B = 29^4)

-- Statement to prove p + q = 459
theorem find_p_plus_q : ∃ p q : ℕ, (RelativelyPrime p q) ∧ (p + q = 459) :=
by sorry

end find_p_plus_q_l106_106224


namespace binom_18_10_l106_106455

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106455


namespace problem_one_problem_two_l106_106963

noncomputable def f (x m : ℝ) : ℝ := x^2 - (m-1) * x + 2 * m

theorem problem_one (m : ℝ) : (∀ x : ℝ, 0 < x → f x m > 0) ↔ (-2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5) :=
by
  sorry

theorem problem_two (m : ℝ) : (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x m = 0) ↔ (m ∈ Set.Ioo (-2 : ℝ) 0) :=
by
  sorry

end problem_one_problem_two_l106_106963


namespace vector_parallel_l106_106615

variables {t : ℝ}

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, t)

theorem vector_parallel (h : (1 : ℝ) / (3 : ℝ) = (3 : ℝ) / t) : t = 9 :=
by 
  sorry

end vector_parallel_l106_106615


namespace log_inequality_solution_l106_106117

theorem log_inequality_solution (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, is_max (ax^2 + x + 1)) :
  {x : ℝ | log a (x - 1) > 0} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end log_inequality_solution_l106_106117


namespace count_odd_divisors_lt_100_l106_106627

  theorem count_odd_divisors_lt_100 : 
    (∃ n, n = ∑ k in Finset.range 100, if Int.factors (k+1).card % 2 = 1 then 1 else 0) ∧ n = 9 := sorry
  
end count_odd_divisors_lt_100_l106_106627


namespace geometry_problem_statement_l106_106650

-- Noncomputable definition only if required in certain geometric constructions.
noncomputable def geometry_problem : Prop :=
  ∀ (A B C D O P : Type)
    [IsIntersection O A C B D]  -- O is the intersection of diagonals AC and BD
    [IsTrapezoid A B C D]       -- ABCD is a trapezoid with AB || CD
    [OnLineSegment P A D]       -- P is on the leg AD
    [∠APB = ∠CPD],              -- given ∠APB = ∠CPD
  1/(AB) + 1/(CD) = 1/(OP)      -- Prove that 1/AB + 1/CD = 1/OP

-- Main theorem to be stated in Lean
theorem geometry_problem_statement : geometry_problem := 
  by 
  sorry

end geometry_problem_statement_l106_106650


namespace max_triangles_l106_106561

def choose (n k : ℕ) : ℕ := nat.choose n k

theorem max_triangles (points_on_a : ℕ) (points_on_b : ℕ) (h1 : points_on_a = 5) (h2 : points_on_b = 8) :
  choose 5 2 * choose 8 1 + choose 5 1 * choose 8 2 = 220 :=
by sorry

end max_triangles_l106_106561


namespace range_of_t_for_obtuse_angle_l106_106937

def obtuse_angle_range (e1 e2 : ℝ) (t : ℝ) : Prop :=
  let a := t * e1 + e2 in
  let b := e1 + t * e2 in
  e1 ≠ 0 ∧ e2 ≠ 0 ∧ e1 * e2 = 0 ∧ 1 * 1 = 1 ∧ ((a * b) < 0 ↔ t ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioo (-(∞ : ℝ)) (-1))

theorem range_of_t_for_obtuse_angle (e1 e2 : ℝ) :
  obtuse_angle_range e1 e2 t :=
begin
  sorry
end

end range_of_t_for_obtuse_angle_l106_106937


namespace binom_18_10_l106_106449

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106449


namespace max_value_8a_3b_5c_l106_106246

theorem max_value_8a_3b_5c (a b c : ℝ) (h_condition : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ (Real.sqrt 373) / 6 :=
by
  sorry

end max_value_8a_3b_5c_l106_106246


namespace additional_cost_per_person_l106_106829

-- Define the initial conditions and variables used in the problem
def base_cost := 1700
def discount_per_person := 50
def car_wash_earnings := 500
def initial_friends := 6
def final_friends := initial_friends - 1

-- Calculate initial cost per person with all friends
def discounted_base_cost_initial := base_cost - (initial_friends * discount_per_person)
def total_cost_after_car_wash_initial := discounted_base_cost_initial - car_wash_earnings
def cost_per_person_initial := total_cost_after_car_wash_initial / initial_friends

-- Calculate final cost per person after Brad leaves
def discounted_base_cost_final := base_cost - (final_friends * discount_per_person)
def total_cost_after_car_wash_final := discounted_base_cost_final - car_wash_earnings
def cost_per_person_final := total_cost_after_car_wash_final / final_friends

-- Proving the amount each friend has to pay more after Brad leaves
theorem additional_cost_per_person : cost_per_person_final - cost_per_person_initial = 40 := 
by
  sorry

end additional_cost_per_person_l106_106829


namespace range_a_l106_106289

def A (x : ℝ) : Prop := x < -1 ∨ x > 3

def B (y : ℝ) (a : ℝ) : Prop := -a < y ∧ y ≤ 4 - a

theorem range_a (a : ℝ) : 
  (∀ y, (∃ x, x ≤ 2 ∧ y = 2^x - a) → ((A (2^x - a) ∧ B (2^x - a) a) = B (2^x - a) a)) →
  a ∈ (-∞, -3] ∪ (5, ∞) :=
by
  intro h
  sorry

end range_a_l106_106289


namespace binom_18_10_l106_106487

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106487


namespace find_y_l106_106833

theorem find_y (y : ℝ) (h_cond : y = (1 / y) * (-y) - 3) : y = -4 := 
sorry

end find_y_l106_106833


namespace max_volume_pyramid_l106_106790

/-- Given:
  AB = 3,
  AC = 5,
  sin ∠BAC = 3/5,
  All lateral edges SA, SB, SC form the same angle with the base plane, not exceeding 60°.
  Prove: the maximum volume of pyramid SABC is 5sqrt(174)/4. -/
theorem max_volume_pyramid 
    (A B C S : Type) 
    (AB : ℝ) 
    (AC : ℝ) 
    (alpha : ℝ) 
    (h : ℝ)
    (V : ℝ)
    (sin_BAC : ℝ) :
    AB = 3 →
    AC = 5 →
    sin_BAC = 3 / 5 →
    alpha ≤ 60 →
    V = (1 / 3) * (1 / 2 * AB * AC * sin_BAC) * h →
    V = 5 * sqrt 174 / 4 :=
by
  intros
  sorry

end max_volume_pyramid_l106_106790


namespace binom_18_10_l106_106456

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106456


namespace projection_matrix_l106_106901

open Matrix

noncomputable def P : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![4/9, -4/9, -2/9],
    ![-4/9, 4/9, 2/9],
    ![-2/9, 2/9, 1/9]]

def vector_v (x y z : ℝ) : Fin 3 → ℝ
| 0 => x
| 1 => y
| 2 => z

theorem projection_matrix (x y z : ℝ) :
  let v := vector_v x y z
  let u := vector_v 2 (-2) (-1)
in  (P.mulVec v) = (u.mul (dot_product v u / dot_product u u)) :=
by
  sorry

end projection_matrix_l106_106901


namespace root_interval_l106_106197

theorem root_interval (f : ℝ → ℝ) (m n : ℤ) 
  (h1 : f = λ x, 2 * x^2 + x - 4)
  (h2 : ∃! x, m < x ∧ x < n ∧ f x = 0)
  (h3 : n = m + 1) : m = 1 :=
by sorry

end root_interval_l106_106197


namespace find_c_for_minimum_of_cos_graph_l106_106879

theorem find_c_for_minimum_of_cos_graph
  (c : ℝ)
  (h1 : ∀ x : ℝ, 3 * Real.cos (5 * x + c) ∈ {y : ℝ | y = 3 * Real.cos(5 * x + c)})
  (h2 : 0 < c)
  (h3 : 3 * Real.cos c = -3) :
  c = Real.pi :=
sorry

end find_c_for_minimum_of_cos_graph_l106_106879


namespace contact_probability_l106_106544

-- Definition of the number of tourists in each group
def num_tourists_group1 : ℕ := 6
def num_tourists_group2 : ℕ := 7
def total_pairs : ℕ := num_tourists_group1 * num_tourists_group2

-- Definition of probability for no contact
def p : ℝ -- probability of contact
def prob_no_contact := (1 - p) ^ total_pairs

-- The theorem to be proven
theorem contact_probability : 1 - prob_no_contact = 1 - (1 - p) ^ total_pairs :=
by
  sorry

end contact_probability_l106_106544


namespace inequality_holds_for_all_real_l106_106268

theorem inequality_holds_for_all_real (a : ℝ) : a + a^3 - a^4 - a^6 < 1 :=
by
  sorry

end inequality_holds_for_all_real_l106_106268


namespace tan_of_α_trigonometric_expression_l106_106151

open Real

noncomputable def α := α

-- Condition: The terminal side of angle α passes through point P(1, -2)
axiom terminal_side_through_point : ∃ α, (cos α, sin α) = (1 / sqrt(5), -2 / sqrt(5))

-- Problem 1: tan α = -2
theorem tan_of_α : tan α = -2 := sorry

-- Problem 2: Given tan α = -2, prove: 
-- (sin(π-α) + cos(-α)) / (2cos(π/2 - α) - sin(π/2 + α)) = 1 / 5
theorem trigonometric_expression : 
  (sin (π - α) + cos (- α)) / (2 * cos (π / 2 - α) - sin (π / 2 + α)) = 1 / 5 := sorry

end tan_of_α_trigonometric_expression_l106_106151


namespace seq_50_value_l106_106128

-- Definitions based on the problem conditions.
def seq (n : ℕ) : ℝ := if n = 1 then 3 else 3 * (sum (λ i, seq i) n)^2 / (3 * (sum (λ i, seq i) n) - 2)

def sum_seq (n : ℕ) : ℝ :=
  if n = 0 then 0 else seq n + sum_seq (n-1)

-- Statement of the problem in Lean.
theorem seq_50_value : seq 50 = -9 / 63788 := sorry

end seq_50_value_l106_106128


namespace petya_cards_sum_gt_0_75_l106_106265

theorem petya_cards_sum_gt_0_75
  (x : Fin 10 → ℕ)
  (h : ∀ i, 1 ≤ x i ∧ x i ≤ 10)
  (distinct : Function.Injective x) :
  ∑ i in Finset.range 9, 1 / (x i + x (i + 1)) > 0.75 := 
sorry

end petya_cards_sum_gt_0_75_l106_106265


namespace bus_number_of_seats_l106_106642

theorem bus_number_of_seats (S : ℕ) 
  (morning_people : ℕ) (morning_free_seats : ℕ)
  (evening_people : ℕ) (evening_free_seats : ℕ)
  (morning_condition : morning_people = 13) 
  (morning_free_condition : morning_free_seats = 9)
  (evening_condition : evening_people = 10) 
  (evening_free_condition : evening_free_seats = 6)
  (morning_computation : morning_people + morning_free_seats = S)
  (evening_computation : evening_people + evening_free_seats = S) :
  S = 16 :=
begin
  sorry
end

end bus_number_of_seats_l106_106642


namespace perimeter_shaded_region_l106_106652

noncomputable def circumference_perimeter_shaded_region
  (radius : ℝ)
  (c : ℝ)
  (arc_angle : ℝ)
  (n_circles : ℕ)
  (total : ℝ) : Prop :=
  c = 48 ∧ arc_angle = 120 ∧ n_circles = 3 → total = 48

-- The statement of the theorem
theorem perimeter_shaded_region :
  ∃ radius, ∃ c, ∃ arc_angle, ∃ n_circles, ∃ total,
    circumference_perimeter_shaded_region radius c arc_angle n_circles total :=
begin
  use [24 / π, 48, 120, 3, 48],
  unfold circumference_perimeter_shaded_region,
  intros,
  exact ⟨rfl, ⟨rfl, rfl⟩⟩,
  sorry
end

end perimeter_shaded_region_l106_106652


namespace calendars_ordered_l106_106837

theorem calendars_ordered 
  (C D : ℝ) 
  (h1 : C + D = 500) 
  (h2 : 0.75 * C + 0.50 * D = 300) 
  : C = 200 :=
by
  sorry

end calendars_ordered_l106_106837


namespace binom_18_10_l106_106475

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106475


namespace alfreds_scooter_cost_l106_106407

theorem alfreds_scooter_cost :
  ∃ P : ℝ, (P + 800) * 1.1154 = 5800 ∧ P = 4400 :=
begin
  use 4400,
  split,
  { norm_num, },
  { norm_num, }
end

end alfreds_scooter_cost_l106_106407


namespace proper_subsets_of_M_inter_N_l106_106610

def M := {x : ℤ | ∃ y : ℝ, y = real.sqrt (4 - x^2)}
def N := {y : ℝ | ∃ x : ℝ, y = 3^(x + 1)}

theorem proper_subsets_of_M_inter_N :
  let inter_set := {1, 2}
  ∃ (P : finset (finset ℕ)), P.card = 2^inter_set.card - 1 ∧ ∀ A ∈ P, A ⊆ inter_set ∧ A ≠ inter_set :=
by
  sorry

end proper_subsets_of_M_inter_N_l106_106610


namespace binom_18_10_l106_106470

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106470


namespace triangle_groups_count_l106_106812

theorem triangle_groups_count (total_points collinear_groups groups_of_three total_combinations : ℕ)
    (h1 : total_points = 12)
    (h2 : collinear_groups = 16)
    (h3 : groups_of_three = (total_points.choose 3))
    (h4 : total_combinations = groups_of_three - collinear_groups) :
    total_combinations = 204 :=
by
  -- This is where the proof would go
  sorry

end triangle_groups_count_l106_106812


namespace cubic_identity_l106_106995

theorem cubic_identity (x : ℝ) (h : x + (1/x) = -3) : x^3 + (1/x^3) = -18 :=
by
  sorry

end cubic_identity_l106_106995


namespace reflections_of_orthocenter_lie_on_circumcircle_l106_106673

theorem reflections_of_orthocenter_lie_on_circumcircle
  (ABC : Type) [triangle ABC]
  (acute_angled_triangle : is_acute_angled_triangle ABC)
  (H : point) (orthocenter : is_orthocenter H ABC) :
  ∀ side ∈ sides_of_triangle ABC, 
    let H' := reflection_of_point H side in
    lies_on_circumcircle H' ABC :=
by
  sorry

end reflections_of_orthocenter_lie_on_circumcircle_l106_106673


namespace cos_75_deg_identity_l106_106875

theorem cos_75_deg_identity :
  real.cos (75 * real.pi / 180) = (real.sqrt 6 - real.sqrt 2) / 4 :=
by 
  -- We skip the actual proof.
  sorry

end cos_75_deg_identity_l106_106875


namespace non_real_roots_interval_l106_106184

theorem non_real_roots_interval (b : ℝ) : (2 * x^2 + b * x + 16).has_nonreal_roots ↔ b ∈ Ioo (-8 * real.sqrt 2) (8 * real.sqrt 2) :=
sorry

end non_real_roots_interval_l106_106184


namespace intersect_at_pi_over_3_l106_106604

theorem intersect_at_pi_over_3 :
  ∃ φ : ℝ, 
    | φ | < (real.pi / 2) ∧ 
    (cos (real.pi / 3 - real.pi / 6) = sin (2 * real.pi / 3 + φ)) → 
    φ = - real.pi / 3 := 
begin
  sorry,
end

end intersect_at_pi_over_3_l106_106604


namespace solve_expression_l106_106797

theorem solve_expression : 3 ^ (1 ^ (0 ^ 2)) - ((3 ^ 1) ^ 0) ^ 2 = 2 := by
  sorry

end solve_expression_l106_106797


namespace complex_number_solution_l106_106152

theorem complex_number_solution (z : ℂ) (h : (1 + 2 * complex.I) * z = 4 + 3 * complex.I) :
  z = 2 - complex.I :=
sorry

end complex_number_solution_l106_106152


namespace additional_plates_correct_l106_106279

-- Define the conditions
def original_set_1 : Finset Char := {'B', 'F', 'J', 'N', 'T'}
def original_set_2 : Finset Char := {'E', 'U'}
def original_set_3 : Finset Char := {'G', 'K', 'R', 'Z'}

-- Define the sizes of the original sets
def size_set_1 := (original_set_1.card : Nat) -- 5
def size_set_2 := (original_set_2.card : Nat) -- 2
def size_set_3 := (original_set_3.card : Nat) -- 4

-- Sizes after adding new letters
def new_size_set_1 := size_set_1 + 1 -- 6
def new_size_set_2 := size_set_2 + 1 -- 3
def new_size_set_3 := size_set_3 + 1 -- 5

-- Calculate the original and new number of plates
def original_plates : Nat := size_set_1 * size_set_2 * size_set_3 -- 5 * 2 * 4 = 40
def new_plates : Nat := new_size_set_1 * new_size_set_2 * new_size_set_3 -- 6 * 3 * 5 = 90

-- Calculate the additional plates
def additional_plates : Nat := new_plates - original_plates -- 90 - 40 = 50

-- The proof statement
theorem additional_plates_correct : additional_plates = 50 :=
by
  -- Proof can be filled in here
  sorry

end additional_plates_correct_l106_106279


namespace equal_area_set_l106_106133

open Classical
noncomputable theory

-- Point and triangle definitions
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B P : Point)

-- Function to calculate the area of a triangle using determinant
def triangle_area (t : Triangle) : ℝ :=
  (1 / 2) * abs (
    t.A.x * (t.B.y - t.P.y) +
    t.B.x * (t.P.y - t.A.y) +
    t.P.x * (t.A.y - t.B.y)
  )

-- Given conditions
variables (A B C D P : Point)
variables (h1 : (D.x - C.x) ^ 2 + (D.y - C.y) ^ 2 = 4 * ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2))
variables (h2 : ¬((B.x - A.x) / (B.y - A.y) = (D.x - C.x) / (D.y - C.y)))

-- Proof statement
theorem equal_area_set (A B C D : Point) (h1 : (D.x - C.x) ^ 2 + (D.y - C.y) = 4 * ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2))
  (h2 : ¬((B.x - A.x) / (B.y - A.y) = (D.x - C.x) / (D.y - C.y)))
  : 
  {P : Point | triangle_area ⟨A, B, P⟩ = triangle_area ⟨C, D, P⟩} = 
  ⋃ (I : Point) (hI : ∃ t u v w : ℝ, t * tan(u) = 2 * tan(v) ∧ I = point_inter (line_through A B) (line_through C D)),
  {p : Point | is_on_line p I}
    :=
 sorry

end equal_area_set_l106_106133


namespace satisfiable_edges_l106_106331

open Classical
open Finset

variable (V : Type*) [Fintype V] (E : Finset (Finset V)) (k : ℕ)

def satisfied (coloring : V → ℕ) (edge : Finset V) : Prop :=
  ∃ (v₁ v₂ : V) (hv₁ : v₁ ∈ edge) (hv₂ : v₂ ∈ edge), v₁ ≠ v₂ ∧ coloring v₁ ≠ coloring v₂

theorem satisfiable_edges (hE : ∀ e ∈ E, ∃ x y, x ≠ y ∧ (x ∈ e) ∧ (y ∈ e)) :
  ∃ (coloring : V → ℕ), ∑ e in E.filter (λ e, satisfied coloring e), 1 ≥ (k - 1) * E.card / k := 
begin
  sorry
end

end satisfiable_edges_l106_106331


namespace anna_cupcakes_remaining_l106_106861

theorem anna_cupcakes_remaining :
  let total_cupcakes := 60
  let cupcakes_given_away := (4 / 5 : ℝ) * total_cupcakes
  let cupcakes_after_giving := total_cupcakes - cupcakes_given_away
  let cupcakes_eaten := 3
  let cupcakes_left := cupcakes_after_giving - cupcakes_eaten
  cupcakes_left = 9 :=
by
  sorry

end anna_cupcakes_remaining_l106_106861


namespace binom_18_10_eq_43758_l106_106460

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106460


namespace unknown_sum_of_digits_l106_106216

theorem unknown_sum_of_digits 
  (A B C D : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h2 : D = 1)
  (h3 : (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D) : 
  A + B = 0 := 
sorry

end unknown_sum_of_digits_l106_106216


namespace hannah_highest_score_l106_106622

theorem hannah_highest_score :
  ∀ (total_questions : ℕ) (wrong_answers_student1 : ℕ) (percentage_student2 : ℚ),
  total_questions = 40 →
  wrong_answers_student1 = 3 →
  percentage_student2 = 0.95 →
  ∃ (correct_answers_hannah : ℕ), correct_answers_hannah > 38 :=
by {
  intros,
  sorry,
}

end hannah_highest_score_l106_106622


namespace benny_seashells_l106_106416

def seashells_benny_had : ℕ := 66
def seashells_given_to_jason : ℕ := 52
def seashells_benny_now_has : ℕ := 14

theorem benny_seashells :
  seashells_benny_had - seashells_given_to_jason = seashells_benny_now_has :=
by 
suffices seashells_benny_had - seashells_given_to_jason = 14 by simp only [seashells_benny_now_has, eq_self_iff_true, forall_true_left],
sorry

end benny_seashells_l106_106416


namespace sum_of_possible_a_l106_106158

def f (x : ℝ) : ℝ :=
if x > 0 then log 2 x else x^2 + 4 * x + 1

theorem sum_of_possible_a :
  (∃ a : ℝ, f (f a) = 1) →
  ∑ a in {4, 1, 1/16, -2 - real.sqrt 5, -2 - real.sqrt 3, -2 + real.sqrt 3}.to_finset, a = -15/16 - real.sqrt 5 :=
by
  sorry

end sum_of_possible_a_l106_106158


namespace surface_area_parallelepiped_l106_106836

theorem surface_area_parallelepiped
  (P : Type)
  [parallelepiped P]
  (W B : ℕ)
  (h1 : B = W * 53 / 52)
  (h2 : 1 < side1 P)
  (h3 : 1 < side2 P)
  (h4 : 1 < side3 P)
  : surface_area P = 142 := 
sorry 

end surface_area_parallelepiped_l106_106836


namespace hank_route_distance_l106_106709

theorem hank_route_distance 
  (d : ℝ) 
  (h1 : ∃ t1 : ℝ, t1 = d / 70 ∧ t1 = d / 70 + 1 / 60) 
  (h2 : ∃ t2 : ℝ, t2 = d / 75 ∧ t2 = d / 75 - 1 / 60) 
  (time_diff : (d / 70 - d / 75) = 1 / 30) : 
  d = 35 :=
sorry

end hank_route_distance_l106_106709


namespace standard_ellipse_eq_points_N_F_Q_collinear_l106_106933

-- Definition for the equation of the ellipse
def ellipse_eq (x y : ℝ) (a : ℝ) : Prop :=
  (a > 0) ∧ (x^2 / a^2 + y^2 / (7 - a^2) = 1)

-- Proof for the standard equation of ellipse
theorem standard_ellipse_eq (a : ℝ) (ha : a > 0) (focal_dist : ℝ) 
  (h_focal_dist : focal_dist = 2) (h_foci_on_x : a^2 > 7 - a^2) : 
  a^2 = 4 → ellipse_eq 1 0 a := 
begin
  intro h_a2,
  sorry,  -- Proof details
end

-- Definitions for part (Ⅱ)
def line_eq (x y k : ℝ) : Prop := y = k * (x - 4)

def focus {a : ℝ} (ha : 0 < a) : ℝ := 1  -- Right focus of the ellipse

-- Proof that points N, F, and Q are collinear
theorem points_N_F_Q_collinear (a k : ℝ) (ha : a > 0) (h_focal_dist : focal_dist = 2) 
  (h_foci_on_x : a^2 > 7 - a^2) (h_P : line_eq P_x1 P_y1 k) (h_Q : line_eq Q_x2 Q_y2 k) 
  (h_N : line_eq P_x1 (-P_y1) k) (h_focus : focus ha = (1, 0)) :
  collinear N_x N_y F_x F_y Q_x Q_y :=
begin
  sorry, -- Proof details
end

end standard_ellipse_eq_points_N_F_Q_collinear_l106_106933


namespace square_area_problem_l106_106328

theorem square_area_problem
    (x1 y1 x2 y2 : ℝ)
    (h1 : y1 = x1^2)
    (h2 : y2 = x2^2)
    (line_eq : ∃ a : ℝ, a = 2 ∧ ∃ b : ℝ, b = -22 ∧ ∀ x y : ℝ, y = 2 * x - 22 → (y = y1 ∨ y = y2)) :
    ∃ area : ℝ, area = 180 ∨ area = 980 :=
sorry

end square_area_problem_l106_106328


namespace probability_ab_minus_a_minus_b_is_odd_l106_106426

theorem probability_ab_minus_a_minus_b_is_odd : 
  ∃ p : ℚ, p = 17 / 22 ∧ 
    (∀ a b : ℕ, a ≠ b ∧ 1 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 12 → 
      (let prod_expr := a * b - a - b 
      in prod_expr % 2 = 1 → false)) :=
by
  sorry

end probability_ab_minus_a_minus_b_is_odd_l106_106426


namespace parabola_coefficients_l106_106280

-- Define the parabolic conditions
def is_parabola (a b c : ℝ) := ∀ x, y = ax^2 + bx + c

def vertex_at (a : ℝ) (b : ℝ) (c : ℝ) (vx : ℝ) (vy : ℝ) : Prop :=
  ∀ x, (a*(vx - 2)^2 + vy = a*x^2 + b*x + c)

def vertical_symmetry (a : ℝ) := a = a

def contains_point (a : ℝ) (b : ℝ) (c : ℝ) (px : ℝ) (py : ℝ) : Prop :=
  py = a*px^2 + b*px + c

-- The proof statement
theorem parabola_coefficients {a b c : ℝ} :
  (vertex_at a b c 2 4) ∧ (vertical_symmetry a) ∧ (contains_point a b c 0 5) →
  (a = 1/4 ∧ b = -1 ∧ c = 5) :=
by
  next 1 sorry

end parabola_coefficients_l106_106280


namespace solve_for_z_l106_106520

theorem solve_for_z (z : ℂ) : z^4 - 6 * z^2 + 9 = 0 ↔ z = complex.sqrt 3 ∨ z = -complex.sqrt 3 :=
by sorry

end solve_for_z_l106_106520


namespace field_trip_bread_l106_106842

theorem field_trip_bread (group_size : ℕ) (groups : ℕ) 
    (students_per_group : group_size = 5 + 1)
    (total_groups : groups = 5)
    (sandwiches_per_student : ℕ := 2) 
    (bread_per_sandwich : ℕ := 2) : 
    (groups * group_size * sandwiches_per_student * bread_per_sandwich) = 120 := 
by 
    have students_per_group_lemma : group_size = 6 := by sorry
    have total_students := groups * group_size
    have _ : total_students = 30 := by sorry
    have total_sandwiches := total_students * sandwiches_per_student
    have _ : total_sandwiches = 60 := by sorry
    have total_bread := total_sandwiches * bread_per_sandwich
    have _ : total_bread = 120 := by sorry
    exact id 120

end field_trip_bread_l106_106842


namespace solution_set_inequality_l106_106161

noncomputable def f (x : ℝ) : ℝ :=
  log ((1 + x) / (1 - x)) + sin x

theorem solution_set_inequality (a : ℝ) :
  sqrt 3 < a ∧ a < 2 → f(a-2) + f(a^2-4) < 0 := sorry

end solution_set_inequality_l106_106161


namespace round_nearest_thousandth_l106_106271

noncomputable def recurringDecimal : Real := 36 + 36/99

theorem round_nearest_thousandth : Real.floor (recurringDecimal * 1000) / 1000 = 36.363 :=
by
  sorry

end round_nearest_thousandth_l106_106271


namespace average_speed_is_6_point_5_l106_106871

-- Define the given values
def total_distance : ℝ := 42
def riding_time : ℝ := 6
def break_time : ℝ := 0.5

-- Prove the average speed given the conditions
theorem average_speed_is_6_point_5 :
  (total_distance / (riding_time + break_time)) = 6.5 :=
by
  sorry

end average_speed_is_6_point_5_l106_106871


namespace circle_nonzero_numbers_l106_106415

theorem circle_nonzero_numbers (n : ℕ) (nonzero : n = 2022) (numbers : Fin n → ℤ)
  (h_abs_eq_sum : ∀ i : Fin n, |numbers i| = |numbers (Fin.of_nat ((i + 1) % n)) + numbers (Fin.of_nat ((i - 1) % n))|) :
  ∃ seq : Fin n → ℤ, ∀ i : Fin n, |seq i| = |seq (Fin.of_nat ((i + 1) % n)) + seq (Fin.of_nat ((i - 1) % n))| :=
sorry

end circle_nonzero_numbers_l106_106415


namespace variance_incorrect_min_std_deviation_l106_106037

-- Definitions for the given conditions.
variable (a b : ℝ)

-- The right triangle condition given by Pythagorean theorem.
def right_triangle (a b : ℝ) : Prop :=
  a^2 + b^2 = 9

-- Problems to verify
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b) : 
  ¬(variance {a, b, 3} = 2) := sorry

theorem min_std_deviation {a b : ℝ} (h : right_triangle a b) :
  let s := sqrt(2) - 1,
  (a = b) → (a = 3 * sqrt(2) / 2) → (std_deviation {a, b, 3} = s) := sorry

end variance_incorrect_min_std_deviation_l106_106037


namespace minimum_decimal_digits_l106_106794

-- Definition of the fraction
def fraction : ℚ := 987654321 / (2^30 * 5^6)

-- Theorem statement: minimum number of digits to the right of the decimal point
theorem minimum_decimal_digits : ∃ n : ℕ, n = 30 ∧ (fraction.approximations (n + 1) - fraction.approximations n < 10 ^ (-(n + 1)).toRational) :=
by
  sorry

end minimum_decimal_digits_l106_106794


namespace percentage_gold_coins_l106_106059

theorem percentage_gold_coins (total_objects beads coins: ℕ) (H1: 0.3 * total_objects = beads) 
  (H2: coins = total_objects - beads)
  (H3: 0.3 * coins = (coins - (0.3 * coins))) :
  (0.7 * coins) / total_objects = 0.49 :=
by
  sorry

end percentage_gold_coins_l106_106059


namespace locus_equal_angle_with_plane_l106_106568

variable {Π : Type} [plane Π] (A B : Π) (outside : A ∉ Π ∧ B ∉ Π)

theorem locus_equal_angle_with_plane :
  ∃ (X : Π), (on_plane Π X) ∧ ((angle_with_plane Π A X) = (angle_with_plane Π B X)) →
  (is_apollonius_circle Π A B X) ∨ (is_perpendicular_bisector Π A B X) :=
by
  sorry

end locus_equal_angle_with_plane_l106_106568


namespace initial_investment_was_317_84_l106_106109

theorem initial_investment_was_317_84 :
  ∃ x : ℝ, x * (1 + 0.12)^4 ≈ 500 ∧ x ≈ 317.84 := by
  sorry

end initial_investment_was_317_84_l106_106109


namespace trapezoid_CD_length_l106_106094

-- Define necessary constants and variables
variables (A B C D F E : Type)
variables (AB CD BC : ℝ)

-- Assume conditions about the trapezoid and angles
variables (h1 : 6 = AB) (h2 : 4 * real.sqrt 3 = BC)
variables (h3 : ∠ B C D = 60) (h4 : ∠ C D A = 45)
variables (h5 : AB ∥ CD)

-- Define the lengths CE, EF, and DF that are used to calculate CD
noncomputable def CE := 2 * real.sqrt 3
noncomputable def EF := 6
noncomputable def DF := 4

-- Prove that CD is equal to 10 + 2 * real.sqrt 3
theorem trapezoid_CD_length : 
  CE + EF + DF = 10 + 2 * real.sqrt 3 :=
by 
  have hCE : CE = 2 * real.sqrt 3 := rfl
  have hEF : EF = 6 := rfl
  have hDF : DF = 4 := rfl
  sorry

end trapezoid_CD_length_l106_106094


namespace variance_not_2_minimum_standard_deviation_l106_106028

-- Definition for a right triangle with hypotenuse 3
structure RightTriangle where
  (a : ℝ) (b : ℝ)
  hypotenuse : ℝ := 3
  pythagorean_property : a^2 + b^2 = 9

-- Part (a) - Prove that the variance cannot be 2
theorem variance_not_2 (triangle : RightTriangle) : 
  (6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) ≠ 2 := sorry

-- Part (b) - Prove the minimum standard deviation and corresponding leg lengths
theorem minimum_standard_deviation (triangle : RightTriangle) : 
  (exists (a b : ℝ), a > 0 ∧ b > 0 ∧ a = b ∧ a = b = 3 * (real.sqrt 2) / 2 ∧ 
  real.sqrt ((6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) = real.sqrt (2) - 1)) := sorry

end variance_not_2_minimum_standard_deviation_l106_106028


namespace contact_probability_l106_106542

variable (m : ℕ := 6) (n : ℕ := 7) (p : ℝ)

theorem contact_probability :
  let total_pairs := m * n in
  let probability_no_contact := (1 - p) ^ total_pairs in
  let probability_contact := 1 - probability_no_contact in
  probability_contact = 1 - (1 - p) ^ 42 :=
by
  -- This is where the proof would go
  sorry

end contact_probability_l106_106542


namespace arithmetic_progression_rth_term_l106_106105

open Nat

theorem arithmetic_progression_rth_term (n r : ℕ) (Sn : ℕ → ℕ) 
  (h : ∀ n, Sn n = 5 * n + 4 * n^2) : Sn r - Sn (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l106_106105


namespace problem_solution_l106_106003

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_solution :
  (∀ x : ℝ, f x > (derivative f) x) ∧
  (∀ x : ℝ, f x + 2018 + f (-x) + 2018 = 0) →
  {x : ℝ | f x + 2018 * exp x < 0} = {x : ℝ | 0 < x} :=
by
  intro h
  sorry

end problem_solution_l106_106003


namespace smallest_divisor_of_visible_product_l106_106057

theorem smallest_divisor_of_visible_product 
  (die : Fin 8 → ℕ)
  (h_die : ∀ i, die i = i + 1) :
  ∃ (d : ℕ), 
    (∀ i : Fin 8, 
      let Q := ∏ j in Finset.univ.filter (λ j, j ≠ i), die j 
      in d ∣ Q) 
    ∧ d = 192 := 
by
  sorry

end smallest_divisor_of_visible_product_l106_106057


namespace part_one_part_two_l106_106175

open Real

def vector := ℝ × ℝ

noncomputable def vec_a : vector := (1, 2)
noncomputable def vec_b : vector := (1, -1)
noncomputable def v1 : vector := (2 * 1 + 1, 2 * 2 - 1)
noncomputable def v2 : vector := (1 - 1, 2 - (-1))

noncomputable def dot_product (u v : vector) : ℝ :=
  (u.1 * v.1) + (u.2 * v.2)

noncomputable def magnitude (u : vector) : ℝ :=
  sqrt (u.1 * u.1 + u.2 * u.2)

noncomputable def θ (u v : vector) : ℝ :=
  acos ((dot_product u v) / (magnitude u * magnitude v))

theorem part_one :
  θ ((2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2)) (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) = π / 4 :=
sorry

noncomputable def k := 0

theorem part_two :
  dot_product (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2) (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2) = 0 → k = 0 :=
sorry

end part_one_part_two_l106_106175


namespace average_distance_is_six_l106_106007

def diagonal_length (a : ℝ) : ℝ := real.sqrt (a * a + a * a)

def final_position (a dist_diag dist_90 : ℝ) : (ℝ × ℝ) :=
  let d := diagonal_length a
  let x := (dist_diag * a) / d
  let y := (dist_diag * a) / d
  (x + dist_90, y)

def distances_to_sides (a : ℝ) (p : ℝ × ℝ) : (ℝ × ℝ × ℝ × ℝ) :=
  let (x, y) := p
  (x, y, a - x, a - y)

def average_distance (dists : (ℝ × ℝ × ℝ × ℝ)) : ℝ :=
  let (d1, d2, d3, d4) := dists
  (d1 + d2 + d3 + d4) / 4

theorem average_distance_is_six :
  average_distance (distances_to_sides 12 (final_position 12 7.2 3)) = 6 :=
by sorry

end average_distance_is_six_l106_106007


namespace field_trip_bread_l106_106844

theorem field_trip_bread (group_size : ℕ) (groups : ℕ) 
    (students_per_group : group_size = 5 + 1)
    (total_groups : groups = 5)
    (sandwiches_per_student : ℕ := 2) 
    (bread_per_sandwich : ℕ := 2) : 
    (groups * group_size * sandwiches_per_student * bread_per_sandwich) = 120 := 
by 
    have students_per_group_lemma : group_size = 6 := by sorry
    have total_students := groups * group_size
    have _ : total_students = 30 := by sorry
    have total_sandwiches := total_students * sandwiches_per_student
    have _ : total_sandwiches = 60 := by sorry
    have total_bread := total_sandwiches * bread_per_sandwich
    have _ : total_bread = 120 := by sorry
    exact id 120

end field_trip_bread_l106_106844


namespace sqrt_expression_simplify_l106_106877

theorem sqrt_expression_simplify : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 :=
by 
  sorry

end sqrt_expression_simplify_l106_106877


namespace cost_per_spool_l106_106257

theorem cost_per_spool
  (p : ℕ) (f : ℕ) (y : ℕ) (t : ℕ) (n : ℕ)
  (hp : p = 15) (hf : f = 24) (hy : y = 5) (ht : t = 141) (hn : n = 2) :
  (t - (p + y * f)) / n = 3 :=
by sorry

end cost_per_spool_l106_106257


namespace ellipse_constant_property_l106_106573

noncomputable def ellipse (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_constant_property :
  ∀ (a b : ℝ) (h₁ : a > b) (h₂ : b > 0)
    (P Q : ℝ × ℝ) (line : ℝ → ℝ)
    (hp : P = (-2, 0)) (hq : Q = (-2, -1))
    (hf : ∀ t, t ∈ C ↔ (x => (line x) ∈ lower_half_ellipse)
    (M : ∀ x y : ℝ, ellipse a b x y) 
    (ha : a = 2)
    (hb : b = 1), 
  (focal_length := 2 * sqrt 3) 
  (focal_property : (sqrt (a^2 - b^2) = sqrt 3)) :
  (∀ A B : ℝ×ℝ, 
    (A ∈ ellipse a b) ∧ (B ∈ ellipse a b) → 
    (|Q.1 - A.1|) + |Q.1 - B.1| - |Q.1 - A.1| * |Q.1 - B.1| = 0) :=
sorry

end ellipse_constant_property_l106_106573


namespace problem_sum_f_l106_106923

noncomputable def f (a : ℝ) [a_pos : 0 < a] (x : ℝ) : ℝ := a^x / (a^x + real.sqrt a)

theorem problem_sum_f (a : ℝ) [a_pos : 0 < a] : 
  ∑ k in finset.range 1000, f a ((k + 1 : ℝ) / 1001) = 500 :=
by
  sorry

end problem_sum_f_l106_106923


namespace minimum_value_l106_106760

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.cos x)^2 - 2 * (Real.sin x) + 9 / 2

theorem minimum_value :
  ∃ (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3)), f x = 2 :=
by
  use Real.pi / 6
  sorry

end minimum_value_l106_106760


namespace inequality_sum_sq_A_le_4_sum_sq_a_l106_106563

open BigOperators

variables {α : Type*} [LinearOrderedField α]
variables (a : ℕ → α) (n : ℕ)

def A (i : ℕ) : α :=
  if i = 0 then 0 else (i : α) / (i * i + i - 1) * ∑ k in finset.range i, a (k + 1)

theorem inequality_sum_sq_A_le_4_sum_sq_a
  (ha : ∀ i, i ≠ 0 → a i > 0) :
  (∑ k in finset.range n, (A a k) ^ 2) ≤ 4 * (∑ k in finset.range n, (a (k + 1)) ^ 2) :=
sorry

end inequality_sum_sq_A_le_4_sum_sq_a_l106_106563


namespace contact_probability_l106_106547

-- Definition of the number of tourists in each group
def num_tourists_group1 : ℕ := 6
def num_tourists_group2 : ℕ := 7
def total_pairs : ℕ := num_tourists_group1 * num_tourists_group2

-- Definition of probability for no contact
def p : ℝ -- probability of contact
def prob_no_contact := (1 - p) ^ total_pairs

-- The theorem to be proven
theorem contact_probability : 1 - prob_no_contact = 1 - (1 - p) ^ total_pairs :=
by
  sorry

end contact_probability_l106_106547


namespace total_books_l106_106273

theorem total_books (Sam_books : ℕ) (Joan_books : ℕ) (h₁ : Sam_books = 110) (h₂ : Joan_books = 102) :
  Sam_books + Joan_books = 212 :=
by
  rw [h₁, h₂]
  norm_num

end total_books_l106_106273


namespace sum_of_even_numbers_from_1_to_200_l106_106905

open BigOperators

theorem sum_of_even_numbers_from_1_to_200 :
  ∑ k in (finset.filter (λ x, x % 2 = 0) (finset.range 201)), k = 10100 :=
by
  sorry

end sum_of_even_numbers_from_1_to_200_l106_106905


namespace complex_multiplication_l106_106285

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2 * i) = 2 + i :=
by
  sorry

end complex_multiplication_l106_106285


namespace kristy_initial_cookies_l106_106666

-- Define the initial conditions
def initial_cookies (total_cookies_left : Nat) (c1 c2 c3 : Nat) (c4 c5 c6 : Nat) : Nat :=
  total_cookies_left + c1 + c2 + c3 + c4 + c5 + c6

-- Now we can state the theorem
theorem kristy_initial_cookies :
  initial_cookies 6 5 5 3 1 2 = 22 :=
by
  -- Proof is omitted
  sorry

end kristy_initial_cookies_l106_106666


namespace chenny_friends_l106_106424

theorem chenny_friends (initial_candies : ℕ) (needed_candies : ℕ) (candies_per_friend : ℕ) (h1 : initial_candies = 10) (h2 : needed_candies = 4) (h3 : candies_per_friend = 2) :
  (initial_candies + needed_candies) / candies_per_friend = 7 :=
by
  sorry

end chenny_friends_l106_106424


namespace smallest_largest_multiples_693_correct_l106_106497

noncomputable def smallest_and_largest_multiples_of_693 : ℕ × ℕ :=
  (1024375968, 9876523041)

theorem smallest_largest_multiples_693_correct :
  ∃ (smallest largest : ℕ), 
    smallest_and_largest_multiples_of_693 = (smallest, largest) ∧ 
    (∀ x, (x % 693 = 0 → 
           (∀ d, d ∈ (x.digits 10) ↔ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ 
           ¬(0 = x.digits 10).head)) ∧
    (smallest = 1024375968) ∧ 
    (largest = 9876523041) :=
by
  use 1024375968, 9876523041
  split
  case h1 =>
    exact ⟨rfl⟩
  case h2 =>
    split
    sorry

end smallest_largest_multiples_693_correct_l106_106497


namespace remainder_of_3_pow_2023_mod_7_l106_106796

theorem remainder_of_3_pow_2023_mod_7 :
  3 ^ 2023 % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l106_106796


namespace pentagon_perimeter_abc_l106_106011

open Real

def distance (p q : ℝ × ℝ) : ℝ :=
  (sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2))

def perimeter (points : List (ℝ × ℝ)) : ℝ :=
  points.zip points.tail |>.map (λ ⟨p, q⟩ => distance p q) |>.sum

def a (points : List (ℝ × ℝ)) : ℝ :=
  (perimeter points) - ((b points) * sqrt 2 + (c points) * sqrt 3)

def b (points : List (ℝ × ℝ)) : ℝ :=
  ((perimeter points) - (a points)) / sqrt 2 - (c points) * sqrt (3 / 2)

def c (points : List (ℝ × ℝ)) : ℝ :=
  ((perimeter points) - (a points)) / sqrt 3 - (b points) * sqrt (2 / 3)

theorem pentagon_perimeter_abc (points : List (ℝ × ℝ)) (h : points = [(0,0), (2,1), (5,1), (3,0), (1,-1), (0,0)]) :
  a points + b points + c points = 7 :=
by
  sorry

end pentagon_perimeter_abc_l106_106011


namespace expression_value_l106_106339

/--
Prove that for a = 51 and b = 15, the expression (a + b)^2 - (a^2 + b^2) equals 1530.
-/
theorem expression_value (a b : ℕ) (h1 : a = 51) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 1530 := by
  rw [h1, h2]
  sorry

end expression_value_l106_106339


namespace projection_of_vector_l106_106616

-- Define the necessary vectors as variables
variables (a b : ℝ^3)

-- Given conditions
def norm_a : Prop := ∥a∥ = 1
def perp_a_b : Prop := a ⬝ b = 0

-- Defining the projection
def projection (u v : ℝ^3) : ℝ := (u ⬝ v) / ∥v∥

-- Statement of the problem
theorem projection_of_vector :
  norm_a a → perp_a_b a b → projection (2 • b - a) a = -1 :=
by {
  intros,
  -- actual proof will be skipped
  sorry
}

end projection_of_vector_l106_106616


namespace sum_of_integer_solutions_eq_zero_l106_106904

/--
Let \( f(x) = x^4 - 13x^2 + 36 \). Prove that the sum of all integer solutions of the equation \( f(x) = 0 \) is \( 0 \).
-/
theorem sum_of_integer_solutions_eq_zero : 
  let solutions := {x : ℤ | x^4 - 13 * x^2 + 36 = 0}
  ∑ x in solutions, x = 0 :=
by
  sorry

end sum_of_integer_solutions_eq_zero_l106_106904


namespace twelve_people_circle_number_picked_l106_106779

theorem twelve_people_circle_number_picked (a : ℕ → ℤ) :
  (∑ i in (Finset.range 12), a i) = 78 ∧
  (∀ i, (a ((i - 1) % 12) + a ((i + 1) % 12)) / 2 = (i + 1) % 12 + 1) →
  a 6 = 9 :=
by
  sorry

end twelve_people_circle_number_picked_l106_106779


namespace solve_equation_l106_106275

theorem solve_equation : ∃ x : ℝ, 81 = 3 * 27^(x - 1) ∧ x = 2 :=
by
  use 2
  split
  · sorry
  · sorry

end solve_equation_l106_106275


namespace maurice_riding_times_l106_106701

variable (M : ℕ) -- The number of times Maurice had been horseback riding before visiting Matt
variable (h1 : 8) -- The times Maurice rode during his visit
variable (h2 : 8) -- The times Matt rode with Maurice
variable (h3 : 16) -- The additional times Matt rode
variable (h4 : 24 = 3 * M) -- The total number of times Matt rode during the two weeks is three times the number of times Maurice had ridden before his visit

theorem maurice_riding_times : M = 8 :=
by
  sorry

end maurice_riding_times_l106_106701


namespace quadratic_inequality_solution_empty_set_l106_106053

open Real

theorem quadratic_inequality_solution_empty_set :
  ∀ x : ℝ, (¬ ∃ x : ℝ, x^2 - 2 * x + 3 < 0) ∧ (∃ x : ℝ, (x + 4) * (x - 1) < 0) ∧
    (∃ x : ℝ, (x + 3) * (x - 1) > 0) ∧ (∃ x : ℝ, 2 * x^2 - 3 * x - 2 > 0) :=
by {
  split,
  {
    intro x,
    intro h,
    have h_discriminant : (-2 : ℝ)^2 - 4 * 1 * 3 = -8 := by norm_num,
    rw lt_irrefl at h, contradiction,
  },
  split,
  {
    use 0,
    norm_num,
  },
  split,
  {
    use (-4 : ℝ),
    by norm_num,
  },
  {
    use 0,
    norm_num,
  },
  sorry
}

end quadratic_inequality_solution_empty_set_l106_106053


namespace range_of_m_l106_106558

noncomputable def A := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
noncomputable def B (m : ℝ) := {x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1}

theorem range_of_m (m : ℝ) (h : B m ⊆ A) : -2 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l106_106558


namespace binom_18_10_l106_106457

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106457


namespace count_ways_to_select_team_l106_106704

theorem count_ways_to_select_team : 
  (nat.choose 7 3) * (nat.choose 9 3) = 2940 := 
by
  sorry

end count_ways_to_select_team_l106_106704


namespace sara_schavenger_hunt_l106_106413

theorem sara_schavenger_hunt :
  let monday := 1 -- Sara rearranges the books herself
  let tuesday := 2 -- Sara can choose from Liam or Mia
  let wednesday := 4 -- There are 4 classmates
  let thursday := 3 -- There are 3 new volunteers
  let friday := 1 -- Sara and Zoe do it together
  monday * tuesday * wednesday * thursday * friday = 24 :=
by
  sorry

end sara_schavenger_hunt_l106_106413


namespace find_coordinates_of_M_l106_106654

noncomputable def distance (p₁ p₂ : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p₁.1 - p₂.1) ^ 2 + (p₁.2 - p₂.2) ^ 2 + (p₁.3 - p₂.3) ^ 2)

theorem find_coordinates_of_M:
  let A := (1 : ℝ, 0, 2)
  let B := (1 : ℝ, -3, 1)
  let M := (0 : ℝ, -1, 0)
  (∃ (y : ℝ), M = (0, y, 0) ∧ distance M A = distance M B) →
  M = (0, -1, 0) :=
by
  intros h
  cases h with y hy
  rcases hy with ⟨hy₁, hy₂⟩
  dsimp [distance] at hy₂
  sorry

end find_coordinates_of_M_l106_106654


namespace binom_18_10_l106_106453

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l106_106453


namespace number_of_polynomials_in_G_l106_106674

def polynomial_of_form (Q : polynomial ℤ) (n : ℕ) : Prop :=
  ∃ (c : list ℤ), c.length = n - 1 ∧ Q = polynomial.monomial n 1 + 
  (finset.range (n - 1)).sum (λ k, polynomial.monomial k (c.nth_le k sorry)) + polynomial.C 36

def has_distinct_integer_roots_of_form (Q : polynomial ℤ) : Prop :=
  ∀ (a b : ℤ), Q.has_root (a + (b : ℤ) * I) → 
               ∃ (p : polynomial ℤ), Q = p * (X ^ 2 - 2 * (polynomial.C a) * X + polynomial.C (a^2 + b^2))

def G' : set (polynomial ℤ) :=
  { Q | ∃ n, polynomial_of_form Q n ∧ has_distinct_integer_roots_of_form Q }

theorem number_of_polynomials_in_G' : fintype.card G' = 528 := 
sorry

end number_of_polynomials_in_G_l106_106674


namespace drain_time_l106_106819

noncomputable def fill_rate := 1 / 10 -- Filling rate with drain closed (tubs per minute)
noncomputable def net_fill_rate := 1 / 60 -- Net filling rate with drain open (tubs per minute)
noncomputable def drain_rate := fill_rate - net_fill_rate -- Drain rate (tubs per minute)

theorem drain_time (F : ℝ) (D : ℝ) 
  (h1 : F = 1 / 10) 
  (h2 : F - D = 1 / 60) : 
  1 / D = 12 :=
by
  rw [h1, h2]
  have hD: D = 1 / 12 := by
    calc
      D = F - 1 / 60 : by rw h2.symm
    ... = 1 / 10 - 1 / 60 : by rw h1
    ... = 1 / 12 :
      calc
        1 / 10 - 1 / 60 = 6 / 60 - 1 / 60 : by norm_num
        ... = 5 / 60 : by norm_num
        ... = 1 / 12 : by norm_num
  rw hD
  norm_num
  sorry

end drain_time_l106_106819


namespace range_of_a_l106_106936

theorem range_of_a (a : ℝ) : (a + 1) ^ (-1 / 2) < (3 - 2 * a) ^ (-1 / 2) →
  a > 2 / 3 ∧ a < 3 / 2 := by
sorry

end range_of_a_l106_106936


namespace solve_for_n_l106_106377

theorem solve_for_n (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2) ^ 2 = 12 * 12 * (n - 2)) :
  n = 26 :=
by {
  sorry
}

end solve_for_n_l106_106377


namespace find_b_for_quadratic_inequality_l106_106338

theorem find_b_for_quadratic_inequality:
  (∀ x : ℝ, ¬ (x < 1 ∨ x > 5) → -x^2 + b * x - 5 ≥ 0) →
  (∃ b : ℝ, b = 6) :=
by
  -- Here we take the assumptions and translate them into our problem context
  intro h,
  use 6,
  sorry

end find_b_for_quadratic_inequality_l106_106338


namespace find_M_l106_106552

theorem find_M : ∃ M : ℕ, 6! * 10! = 15 * M! ∧ M = 11 := 
by
  sorry

end find_M_l106_106552


namespace variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106022

-- Definitions and conditions for Problem a) and b)
noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def variance (a b c : ℝ) : ℝ :=
  let mean_squares := (a^2 + b^2 + c^2) / 3
  let mean := (a + b + c) / 3
  mean_squares - mean^2

-- Prove that the variance is less than 2 given the conditions for Problem a)
theorem variance_triangle_less_than_2 {a b : ℝ} (h : is_right_triangle a b 3) : 
  variance a b 3 < 2 := sorry

-- Define the minimum standard deviation and legs for Problem b)
noncomputable def std_dev_of_legs (a b : ℝ) : ℝ :=
  let sides_squared_mean := (a^2 + b^2) / 2
  let mean_legs := (a + b) / 2
  real.sqrt (sides_squared_mean - mean_legs^2)

-- Prove that the minimum standard deviation of the legs of a right triangle with hypotenuse 3 is sqrt(2) - 1
theorem min_std_dev_of_triangle_legs {a b : ℝ} (h : is_right_triangle a b 3) :
  a = b → std_dev_of_legs a b = real.sqrt(2) - 1 :=
begin
  sorry
end

-- Prove the lengths of the legs for minimum standard deviation
theorem lengths_of_min_std_dev (a b : ℝ) (h : is_right_triangle a b 3) :
  a = b → a = 3 * real.sqrt(2) / 2 ∧ b = 3 * real.sqrt(2) / 2 :=
begin
  sorry
end

end variance_triangle_less_than_2_min_std_dev_of_triangle_legs_lengths_of_min_std_dev_l106_106022


namespace monotonic_intervals_of_f_at_a_4_min_value_of_f_x1_sub_f_x2_l106_106602

noncomputable theory

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) := (1/2) * x^2 - a * x + real.log x

-- Conditions for Problem I
def f_x_a4 (x : ℝ) := f x 4

-- Statement I
theorem monotonic_intervals_of_f_at_a_4 :
  (∀ x, 0 < x → x < 2 - real.sqrt 3 → f_x_a4 x > f_x_a4 0) ∧
  (∀ x, 2 - real.sqrt 3 < x → x < 2 + real.sqrt 3 → f_x_a4 x < f_x_a4 (2 - real.sqrt 3)) ∧
  (∀ x, 2 + real.sqrt 3 < x → f_x_a4 x > f_x_a4 (2 + real.sqrt 3)) :=
sorry

-- Conditions for Problem II
axiom a_ge_5_over_2 : (a : ℝ) (a >= 5/2)
axiom x1_lt_x2 : (x1 x2 : ℝ) (x1 < x2) (x1*x1 - a*x1 + 1 = 0) (x2*x2 - a*x2 + 1 = 0)

-- Statement II
theorem min_value_of_f_x1_sub_f_x2 : 
  f x1 a - f x2 a = (15/8) - 2 * real.log 2 :=
sorry

end monotonic_intervals_of_f_at_a_4_min_value_of_f_x1_sub_f_x2_l106_106602


namespace find_p_q_l106_106134

def set_A : Set ℝ := {x | abs (x - 1) > 2}
def set_B (p q : ℝ) : Set ℝ := {x | x^2 + p * x + q ≤ 0}

theorem find_p_q :
  (set_A ∪ set_B p q = Set.univ) ∧ (set_A ∩ set_B p q = Ico (-2) (-1)) →
  p = -1 ∧ q = -6 :=
by sorry

end find_p_q_l106_106134


namespace incorrect_variance_min_standard_deviation_l106_106040

theorem incorrect_variance (a b : ℝ) (h1 : a^2 + b^2 = 9) (h2 : (3 + a + b) / 3 > 1) : 
  (6 - (a + b + 3) / 3)^2 ≠ 2 :=
by
  sorry

theorem min_standard_deviation (a b : ℝ) (h1 : a^2 + b^2 = 9) :
  (minSd : ℝ) × (legs : ℝ × ℝ) :=
by
  let s := sqrt 2 - 1
  let l := 3 * sqrt 2 / 2
  exist (s, (l, l))
  sorry

end incorrect_variance_min_standard_deviation_l106_106040


namespace vacation_cost_l106_106773

theorem vacation_cost (C : ℝ)
  (h1 : C / 5 - C / 8 = 60) :
  C = 800 :=
sorry

end vacation_cost_l106_106773


namespace count_integers_between_sqrt_neg3_and_sqrt_5_l106_106976

theorem count_integers_between_sqrt_neg3_and_sqrt_5 : 
  (finset.filter (λ x, -Real.sqrt 3 < x ∧ x < Real.sqrt 5) (finset.range 5)).card = 4 :=
by
  sorry

end count_integers_between_sqrt_neg3_and_sqrt_5_l106_106976


namespace num_possible_y_l106_106757

theorem num_possible_y : 
  (∃ (count : ℕ), count = (54 - 26 + 1) ∧ 
  ∀ (y : ℤ), 25 < y ∧ y < 55 ↔ (26 ≤ y ∧ y ≤ 54)) :=
by {
  sorry 
}

end num_possible_y_l106_106757


namespace range_of_beta_l106_106942

-- Conditions as hypotheses
variables {α β : ℝ}
variable hα : α ∈ set.Ioo 0 (π / 2)
variable hβ : β ∈ set.Ioo 0 (π / 2)
variable h : (α / (2 * (1 + cos (α / 2)))) < tan β ∧ tan β < ((1 - cos α) / α)

-- New equivalence proof problem
theorem range_of_beta 
  (hα : α ∈ set.Ioo 0 (π / 2)) 
  (hβ : β ∈ set.Ioo 0 (π / 2)) 
  (h : (α / (2 * (1 + cos (α / 2)))) < tan β ∧ tan β < ((1 - cos α) / α)) : 
  (α / 4) < β ∧ β < (α / 2) :=
sorry

end range_of_beta_l106_106942


namespace probability_of_meeting_at_cafe_l106_106067

open Set

/-- Define the unit square where each side represents 1 hour (from 2:00 to 3:00 PM). -/
def unit_square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

/-- Define the overlap condition for Cara and David meeting at the café. -/
def overlap_region : Set (ℝ × ℝ) :=
  { p | max (p.1 - 0.5) 0 ≤ p.2 ∧ p.2 ≤ min (p.1 + 0.5) 1 }

/-- The area of the overlap region within the unit square. -/
noncomputable def overlap_area : ℝ :=
  ∫ x in Icc 0 1, (min (x + 0.5) 1 - max (x - 0.5) 0)

theorem probability_of_meeting_at_cafe : overlap_area / 1 = 1 / 2 :=
by
  sorry

end probability_of_meeting_at_cafe_l106_106067


namespace sum_of_edges_geometric_progression_l106_106017

noncomputable def sum_of_edges (a : ℝ) (r : ℝ) : ℝ :=
  4 * (a / r + a + a * r)

theorem sum_of_edges_geometric_progression (a r : ℝ) :
  (a / r) * a * (a * r) = 432 ∧
  2 * ((a^2) / r + a^2 * r + a^2) = 360 →
  r = 1 →
  sum_of_edges a r = 72 * real.cbrt 2 :=
by
  sorry

end sum_of_edges_geometric_progression_l106_106017


namespace quadratic_three_distinct_solutions_l106_106729

open Classical

variable (a b c : ℝ) (x1 x2 x3 : ℝ)

-- Conditions:
variables (hx1 : a * x1^2 + b * x1 + c = 0)
          (hx2 : a * x2^2 + b * x2 + c = 0)
          (hx3 : a * x3^2 + b * x3 + c = 0)
          (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

-- Proof problem
theorem quadratic_three_distinct_solutions : a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end quadratic_three_distinct_solutions_l106_106729


namespace volume_of_earth_dug_out_l106_106349

theorem volume_of_earth_dug_out (diameter depth : ℝ) (h_diameter : diameter = 2) (h_depth : depth = 8) :
  let r := diameter / 2
  let volume := Real.pi * r^2 * depth
  volume ≈ 25.13272 :=
by
  have r_def : r = 1 := by rw [h_diameter]; norm_num
  have volume_def : volume = Real.pi * 1^2 * 8 := by rw [r_def, h_depth]; norm_num
  suffices volume_approx : Real.pi * 8 ≈ 25.13272 by { rwa volume_def }
  sorry

end volume_of_earth_dug_out_l106_106349


namespace sphere_contains_one_rational_point_l106_106499

def is_rational (x : ℚ) : Prop := true  -- Checks if x is a rational number

def rational_point (p : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (x y z : ℚ),
  p = (x : ℝ) • ![1, 0, 0] + (y : ℝ) • ![0, 1, 0] + (z : ℝ) • ![0, 0, 1]

def sphere_eq (x y z : ℝ) : ℝ := (x - real.sqrt 2)^2 + y^2 + z^2 - 2

theorem sphere_contains_one_rational_point :
  (∃ p : EuclideanSpace ℝ (Fin 3), rational_point p ∧ sphere_eq p 0) ∧ 
  (∀ p₁ p₂ : EuclideanSpace ℝ (Fin 3), rational_point p₁ → rational_point p₂ → sphere_eq p₁ 0 → sphere_eq p₂ 0 → p₁ = p₂) :=
begin
  sorry
end

end sphere_contains_one_rational_point_l106_106499


namespace Guangdong_college_entrance_exam2013_geometric_sequence_l106_106346

theorem Guangdong_college_entrance_exam2013_geometric_sequence :
  let a : ℕ → ℤ := λ n, (-2)^(n - 1)
  in a 1 + |a 2| + a 3 + |a 4| = 15 :=
by
  let a : ℕ → ℤ := λ n, (-2)^(n - 1)
  sorry

end Guangdong_college_entrance_exam2013_geometric_sequence_l106_106346


namespace cost_per_book_l106_106727

theorem cost_per_book (initial_amount : ℤ) (remaining_amount : ℤ) (num_books : ℤ) (cost_per_book : ℤ) :
  initial_amount = 79 →
  remaining_amount = 16 →
  num_books = 9 →
  cost_per_book = (initial_amount - remaining_amount) / num_books →
  cost_per_book = 7 := 
by
  sorry

end cost_per_book_l106_106727


namespace loyalty_program_theorem_l106_106722

-- Definitions representing the conditions
def CentralBankInterestPromotion (cb: Type) : Prop := 
  ∀ (loyaltyProgram: Type), encouragesNationalAdoption(loyaltyProgram)

def CentralBankInterestStimulus (cb: Type) : Prop := 
  ∀ (loyaltyProgram: Type), stimulatesConsumerSpending(loyaltyProgram)

def BankBenefitLoyalty (bank: Type) : Prop := 
  ∀ (loyaltyProgram: Type), attractsAndRetainsCustomers(bank, loyaltyProgram)

def BankBenefitTransactionVolume (bank: Type) : Prop := 
  ∀ (loyaltyProgram: Type), increasesCardUsage(bank, loyaltyProgram)

def RegistrationRequirement (system: Type) : Prop :=
  ∀ (loyaltyProgram: Type), (collectsUserData(system, loyaltyProgram) ∧ preventsFraud(system, loyaltyProgram))

-- Theorem to prove the hypothesis
theorem loyalty_program_theorem
  (cb: Type) (bank: Type) (system: Type)
  (h1: CentralBankInterestPromotion cb)
  (h2: CentralBankInterestStimulus cb)
  (h3: BankBenefitLoyalty bank)
  (h4: BankBenefitTransactionVolume bank)
  (h5: RegistrationRequirement system):
  (∀ (loyaltyProgram: Type), 
    encouragesNationalAdoption(loyaltyProgram) →
    stimulatesConsumerSpending(loyaltyProgram) →
    attractsAndRetainsCustomers(bank, loyaltyProgram) →
    increasesCardUsage(bank, loyaltyProgram) →
    (collectsUserData(system, loyaltyProgram) ∧ preventsFraud(system, loyaltyProgram))) :=
by
  intros loyaltyProgram h_ena h_scs h_aar h_ic فرضية
  sorry

end loyalty_program_theorem_l106_106722


namespace find_x_plus_2y_l106_106591

open Complex

theorem find_x_plus_2y (a x y : ℝ) (h : (a + Complex.i) / (2 + Complex.i) = x + y * Complex.i) : x + 2 * y = 1 :=
sorry

end find_x_plus_2y_l106_106591


namespace increasing_function_range_l106_106140

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a ^ x

theorem increasing_function_range (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
  (h₂ : ∀ x y : ℝ, x < y → f a x ≤ f a y) :
  a ∈ set.Ico (3 / 2) 2 := 
begin
  sorry
end

end increasing_function_range_l106_106140


namespace interval_of_monotonic_increase_l106_106752

theorem interval_of_monotonic_increase (α : ℝ) (f : ℝ → ℝ) (h : f = (λ x, x^α)) (h2 : f 2 = 4) : 
  (0 : ℝ) ≤ x → x < ∞ → deriv f x ≥ 0 :=
sorry

end interval_of_monotonic_increase_l106_106752


namespace variance_not_2_minimum_standard_deviation_l106_106027

-- Definition for a right triangle with hypotenuse 3
structure RightTriangle where
  (a : ℝ) (b : ℝ)
  hypotenuse : ℝ := 3
  pythagorean_property : a^2 + b^2 = 9

-- Part (a) - Prove that the variance cannot be 2
theorem variance_not_2 (triangle : RightTriangle) : 
  (6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) ≠ 2 := sorry

-- Part (b) - Prove the minimum standard deviation and corresponding leg lengths
theorem minimum_standard_deviation (triangle : RightTriangle) : 
  (exists (a b : ℝ), a > 0 ∧ b > 0 ∧ a = b ∧ a = b = 3 * (real.sqrt 2) / 2 ∧ 
  real.sqrt ((6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) = real.sqrt (2) - 1)) := sorry

end variance_not_2_minimum_standard_deviation_l106_106027


namespace binom_18_10_l106_106485

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l106_106485


namespace number_of_subsets_l106_106989

-- Define the set S as {1, 2, 3}
def S : Set ℕ := {1, 2, 3}

-- State the theorem
theorem number_of_subsets (A : Set ℕ) (h : A ⊆ S) : 
  (finset.powerset (finset.of_set S)).card = 8 :=
  sorry

end number_of_subsets_l106_106989


namespace company_ordered_weight_of_stone_l106_106002

theorem company_ordered_weight_of_stone :
  let weight_concrete := 0.16666666666666666
  let weight_bricks := 0.16666666666666666
  let total_material := 0.8333333333333334
  let weight_stone := total_material - (weight_concrete + weight_bricks)
  weight_stone = 0.5 :=
by
  sorry

end company_ordered_weight_of_stone_l106_106002


namespace incorrect_variance_min_standard_deviation_l106_106042

theorem incorrect_variance (a b : ℝ) (h1 : a^2 + b^2 = 9) (h2 : (3 + a + b) / 3 > 1) : 
  (6 - (a + b + 3) / 3)^2 ≠ 2 :=
by
  sorry

theorem min_standard_deviation (a b : ℝ) (h1 : a^2 + b^2 = 9) :
  (minSd : ℝ) × (legs : ℝ × ℝ) :=
by
  let s := sqrt 2 - 1
  let l := 3 * sqrt 2 / 2
  exist (s, (l, l))
  sorry

end incorrect_variance_min_standard_deviation_l106_106042


namespace balanced_polygons_characterization_l106_106076

def convex_polygon (n : ℕ) (vertices : Fin n → Point) : Prop := 
  -- Definition of convex_polygon should go here
  sorry

def is_balanced (n : ℕ) (vertices : Fin n → Point) (M : Point) : Prop := 
  -- Definition of is_balanced should go here
  sorry

theorem balanced_polygons_characterization :
  ∀ (n : ℕ) (vertices : Fin n → Point) (M : Point),
  convex_polygon n vertices →
  is_balanced n vertices M →
  n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end balanced_polygons_characterization_l106_106076


namespace binom_18_10_l106_106445

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106445


namespace vector_magnitude_l106_106359

theorem vector_magnitude (e₁ e₂ : ℝ) (h₁ : ‖e₁‖ = 1) (h₂ : ‖e₂‖ = 1) 
  (h_dot : inner e₁ e₂ = real.sqrt 3 / 2) : 
  ‖e₁ - real.sqrt 3 • e₂‖ = 1 := sorry

end vector_magnitude_l106_106359


namespace limit_of_circumcenter_l106_106710

theorem limit_of_circumcenter {A_n B_n M : ℕ → ℝ × ℝ}
  (hypA: ∀ n : ℕ, A_n n = (n / (n + 1), (n + 1) / n))
  (hypB: ∀ n : ℕ, B_n n = ((n + 1) / n, n / (n + 1)))
  (hypM: M = (λ n, (1, 1)))
  : 
  (∀ P_n : ℕ → ℝ × ℝ, is_circumcenter (A_n n) (B_n n) (M n) (P_n n) 
          → (tendsto (λ n, (P_n n).1) at_top (𝓝 2)) 
          ∧ (tendsto (λ n, (P_n n).2) at_top (𝓝 2))) 
:= sorry

open_locale big_operators

end limit_of_circumcenter_l106_106710


namespace cube_inverse_sum_l106_106993

theorem cube_inverse_sum (x : ℂ) (h : x + 1/x = -3) : x^3 + (1/x)^3 = -18 :=
by
  sorry

end cube_inverse_sum_l106_106993


namespace men_who_like_both_math_and_lit_women_who_like_only_lit_l106_106204

variable (students : Finset Student)
variables (men women : Finset Student)
variables (likesMath likesLit : Finset Student)
variables (both : Finset Student)
variables (neitherMen neitherWomen : Finset Student)
variables (onlyMath : Finset Student)

-- We assume the following:
-- (1) There are 35 students in the class.
axiom H_count : students.card = 35
-- (2) 7 men like mathematics.
axiom men_likes_math : (men ∩ likesMath).card = 7
-- (3) 6 men like literature.
axiom men_likes_lit : (men ∩ likesLit).card = 6
-- (4) 5 men and 8 women said they do not like either.
axiom men_neither : neitherMen.card = 5
axiom women_neither : neitherWomen.card = 8
-- (5) There are 16 men in the class.
axiom men_count : men.card = 16
-- (6) 5 students like both mathematics and literature.
axiom likes_both : both.card = 5
-- (7) 11 students like only mathematics.
axiom only_math : onlyMath.card = 11

-- Define the required values for the proof
def men_both : Finset Student := men ∩ both
def women_only_lit : Finset Student := women ∩ (likesLit \ likesMath)

-- Proof for part (a): number of men who like both mathematics and literature
theorem men_who_like_both_math_and_lit : men_both.card = 2 := by
  sorry

-- Proof for part (b): number of women who like only literature
theorem women_who_like_only_lit : women_only_lit.card = 6 := by
  sorry

end men_who_like_both_math_and_lit_women_who_like_only_lit_l106_106204


namespace exists_1000_consecutive_numbers_with_5_primes_l106_106319

theorem exists_1000_consecutive_numbers_with_5_primes :
  ∃ n : ℕ, ∃ k : ℕ, (k = 1000) ∧ (∀ m : ℕ, (1001! + 2 ≤ m ∧ m ≤ 1001! + 1001 → ¬ Prime m)) ∧
  (let P_n := λ n, (Finset.filter Prime (Finset.range (n + k)) \ Finset.range n).card in
    /* There exists an n such that the number of primes in the range [n, n+999] is exactly 5 */
    ∃ n : ℕ, P_n n = 5) :=
  sorry

end exists_1000_consecutive_numbers_with_5_primes_l106_106319


namespace distinct_patterns_4x4_two_shaded_l106_106975

theorem distinct_patterns_4x4_two_shaded :
  let grid_size := 4
  let total_squares := grid_size * grid_size
  let shaded_squares := 2
  ∃ (patterns : Set (Fin (total_squares) → Bool)), 
    (patterns.count = 20) ∧
    (∀ p1 p2 ∈ patterns, p1 ≠ p2 → ¬(p1.flip = p2) ∧ ¬(p1.rotate = p2) ∧ ¬(p1.mirror = p2)) :=
sorry

end distinct_patterns_4x4_two_shaded_l106_106975


namespace interval_of_increase_l106_106084

theorem interval_of_increase : 
    ∀ (x : ℝ), (∃ f : ℝ → ℝ, f = λ x, 3 * x - x^3 ∧ (differentiable ℝ f) ∧ 
                (∀ t ∈ Icc (-1:ℝ) 1, f' t ≥ 0)) → Icc (-1:ℝ) 1 :=
by
    sorry

end interval_of_increase_l106_106084


namespace tangent_parallel_to_line_l106_106307

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_to_line :
  ∃ a b : ℝ, (f a = b) ∧ (3 * a^2 + 1 = 4) ∧ (P = (1, 0) ∨ P = (-1, -4)) :=
by
  sorry

end tangent_parallel_to_line_l106_106307


namespace domain_of_f_l106_106785

def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ∈ set.Ioo (-∞) 6 ∪ set.Ioo 6 ∞) :=
by
  sorry

end domain_of_f_l106_106785


namespace rectangular_C₁_general_C₂_intersection_and_sum_l106_106220

-- Definition of curve C₁ in polar coordinates
def C₁_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ ^ 2 = Real.sin θ

-- Definition of curve C₂ in parametric form
def C₂_param (k x y : ℝ) : Prop := 
  x = 8 * k / (1 + k^2) ∧ y = 2 * (1 - k^2) / (1 + k^2)

-- Rectangular coordinate equation of curve C₁ is x² = y
theorem rectangular_C₁ (ρ θ : ℝ) (x y : ℝ) (h₁ : ρ * Real.cos θ ^ 2 = Real.sin θ)
  (h₂ : x = ρ * Real.cos θ) (h₃ : y = ρ * Real.sin θ) : x^2 = y :=
sorry

-- General equation of curve C₂ is x² / 16 + y² / 4 = 1 with y ≠ -2
theorem general_C₂ (k x y : ℝ) (h₁ : x = 8 * k / (1 + k^2))
  (h₂ : y = 2 * (1 - k^2) / (1 + k^2)) : x^2 / 16 + y^2 / 4 = 1 ∧ y ≠ -2 :=
sorry

-- Given point M and parametric line l, prove the value of sum reciprocals of distances to points of intersection with curve C₁ is √7
theorem intersection_and_sum (t m₁ m₂ x y : ℝ) 
  (M : ℝ × ℝ) (hM : M = (0, 1/2))
  (hline : x = Real.sqrt 3 * t ∧ y = 1/2 + t)
  (hintersect1 : 3 * m₁^2 - 2 * m₁ - 2 = 0)
  (hintersect2 : 3 * m₂^2 - 2 * m₂ - 2 = 0)
  (hroot1_2 : m₁ + m₂ = 2/3 ∧ m₁ * m₂ = -2/3) : 
  1 / abs (M.fst - x) + 1 / abs (M.snd - y) = Real.sqrt 7 :=
sorry

end rectangular_C₁_general_C₂_intersection_and_sum_l106_106220


namespace segment_division_constants_l106_106240

noncomputable def point_on_segment (P A B : ℝ^3) :=
  ∃ (t u : ℝ), P = t • A + u • B ∧ t + u = 1

theorem segment_division_constants (P A B : ℝ^3)
  (hP : point_on_segment P A B)
  (hRatio : ∀ (AP PB : ℝ), AP / PB = 5 / 3) :
  ∃ (t u : ℝ), P = t • A + u • B ∧ t = (3 / 8) ∧ u = (5 / 8) :=
by
  sorry

end segment_division_constants_l106_106240


namespace problem_solution_l106_106115

-- Lean 4 statement of the proof problem
theorem problem_solution (m : ℝ) (U : Set ℝ := Univ) (A : Set ℝ := {x | x^2 + 3*x + 2 = 0}) 
  (B : Set ℝ := {x | x^2 + (m + 1)*x + m = 0}) (h : ∀ x, x ∈ (U \ A) → x ∉ B) : 
  m = 1 ∨ m = 2 :=
by 
  -- This is where the proof would normally go
  sorry

end problem_solution_l106_106115


namespace number_of_strikers_l106_106392

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end number_of_strikers_l106_106392


namespace rational_sum_implies_rational_term_find_valid_n_l106_106096

-- Defining the conditions: arithmetic sequence and rational sum:
def is_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_weighted_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (k + 1) * a k

-- Main statement to prove:
theorem rational_sum_implies_rational_term (n : ℕ) (a : ℕ → ℚ) (d : ℚ) 
  (h1 : n ≥ 3) 
  (h2 : is_arithmetic_sequence a d)
  (h3 : (sum_weighted_sequence a n).isRational) : 
  ∃ k : ℕ, k < n ∧ (a k).isRational := sorry

-- Conclusion based on the theorem proving:
theorem find_valid_n (n : ℕ) (h : n ≥ 3) : 
  (∀ a : ℕ → ℚ, ∀ d : ℚ, is_arithmetic_sequence a d → (sum_weighted_sequence a n).isRational → 
  ∃ k : ℕ, k < n ∧ (a k).isRational) ↔ (n % 3 = 1) := sorry

end rational_sum_implies_rational_term_find_valid_n_l106_106096


namespace complex_number_quadrant_l106_106681

open Complex

def z : ℂ := (2 - I) / (1 + I)

theorem complex_number_quadrant : (Re z > 0) ∧ (Im z < 0) := by
  sorry

end complex_number_quadrant_l106_106681


namespace count_repeating_decimals_l106_106551

theorem count_repeating_decimals
  (h1 : ∀ n : ℕ, 1 ≤ n → n ≤ 150)
  (h2 : ∀ n : ℕ, ¬ (n + 1) % 3 = 0) :
  ∃ k, k = 84 ∧ ∀ n : ℕ, 1 ≤ n → n ≤ 150 → ¬ (n + 1) % 3 = 0 → 
    (∀ p : ℕ, prime p → p ∣ (n + 1) → p ≠ 2 ∧ p ≠ 5) →
    k = n :=
sorry

end count_repeating_decimals_l106_106551


namespace binom_18_10_l106_106471

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l106_106471


namespace find_f_neg3_l106_106886

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x^2 - 2 * x else -(x^2 - 2 * -x)

theorem find_f_neg3 (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, 0 < x → f x = x^2 - 2 * x) : f (-3) = -3 :=
by
  sorry

end find_f_neg3_l106_106886


namespace log_sqrt_30_l106_106139

variable (a b : ℝ)

theorem log_sqrt_30 (h1 : log 3 2 = a) (h2 : 3 ^ b = 5) : log 3 (sqrt 30) = (1 / 2) * (a + b + 1) :=
by
  sorry

end log_sqrt_30_l106_106139


namespace solution_l106_106803

-- Definitions for equations and ODZ (Domain of Validity)
def eq1 (x y : ℝ) := 3^(1 + 2 * real.logb 3 (y - x)) = 48
def eq2 (x y : ℝ) := 2 * real.logb 5 (2 * y - x - 12) - real.logb 5 (y - x) = real.logb 5 (y + x)
def odz (x y : ℝ) := (y - x > 0) ∧ (y + x > 0) ∧ (2 * y - x - 12 > 0)

-- The final lean statement
theorem solution (x y : ℝ) (h: x = 16 ∧ y = 20) : eq1 x y ∧ eq2 x y ∧ odz x y :=
by {
  split,
  { sorry }, -- proof that 3^(1 + 2 * real.logb 3 (20 - 16)) = 48
  split,
  { sorry }, -- proof that 2 * real.logb 5 (2 * 20 - 16 - 12) - real.logb 5 (20 - 16) = real.logb 5 (20 + 16)
  { sorry }  -- proof that (20 - 16 > 0) ∧ (20 + 16 > 0) ∧ (2 * 20 - 16 - 12 > 0)
}

end solution_l106_106803


namespace contact_prob_correct_l106_106535

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end contact_prob_correct_l106_106535


namespace exists_ngon_with_given_angles_l106_106763

theorem exists_ngon_with_given_angles (n : ℕ) (α : Fin n → Real) 
  (h_sum: (∑ i, α i) = (n - 2) * Real.pi)
  (h_range: ∀ i, 0 < α i ∧ α i < 2 * Real.pi)
  : ∃ (A : Fin n → (EuclideanSpace ℝ (Fin 2))), 
    ∃ (angles : Fin n → Real), 
      (angles = α) ∧ 
      (∃ (T : Fin n → (EuclideanSpace ℝ (Fin 2))), 
         (∀ i, Dist (T i) (T (Fin.rotate 1 i)) = 
           dist (A i) (A (Fin.rotate 1 i)))) :=
sorry

end exists_ngon_with_given_angles_l106_106763


namespace inequality_abc_l106_106576

theorem inequality_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) :=
sorry

end inequality_abc_l106_106576


namespace problem1_problem2_l106_106119

theorem problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x + y ≥ 6 :=
sorry

theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x * y ≥ 9 :=
sorry

end problem1_problem2_l106_106119


namespace binomial_expansion_props_l106_106950

open Nat

theorem binomial_expansion_props (x : ℝ) :
  -- Given: the binomial expansion condition
  let n := 8 in
  let expansion := (3 * x - 1 / (2 * 3 * x)) ^ n in
  -- Prove:
  -- (I) The fourth term of the expansion is -7 * x ^ (2 / 3)
  (binomial_coefficient n 3 * (3 * x) ^ 5 * (- 1 / (2 * 3 * x)) ^ 3 = -7 * x ^ (2 / 3)) ∧
  -- (II) The constant term of the expansion is 35 / 8
  (binomial_coefficient n 4 * (- 1 / 2) ^ 4 = 35 / 8) :=
  sorry

end binomial_expansion_props_l106_106950


namespace bird_families_to_Asia_l106_106314

theorem bird_families_to_Asia (total_families initial_families left_families went_to_Africa went_to_Asia: ℕ) 
  (h1 : total_families = 85) 
  (h2 : went_to_Africa = 23) 
  (h3 : left_families = 25) 
  (h4 : went_to_Asia = total_families - left_families - went_to_Africa) 
  : went_to_Asia = 37 := 
by 
  rw [h1, h2, h3] at h4 
  simp at h4 
  exact h4

end bird_families_to_Asia_l106_106314


namespace min_distinct_roots_l106_106817

def min_distinct_complex_roots (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) : ℕ :=
  n * (m - 1) + 1

theorem min_distinct_roots (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0)
  (f : ℂ[X]) (hf : polynomial.degree f = n) :
  ∃ d : ℕ, d = min_distinct_complex_roots m n hm hn :=
  sorry

end min_distinct_roots_l106_106817


namespace reflections_of_orthocenter_on_circumcircle_l106_106670

variable (A B C H: Type)
variable [AffineGeometry A B C H]

def is_orthocenter (ABC: Triangle) (H: Point) : Prop := 
  -- Definition of orthocenter would go here

def is_reflection_over_side (H: Point) (side: Line) : Point := 
  -- Reflection function definition would go here

theorem reflections_of_orthocenter_on_circumcircle 
  (ABC: Triangle)
  (H: Point)
  (is_acute: acute_triangle ABC)
  (H_is_orthocenter: is_orthocenter ABC H) :
  ∀ side: Line, (reflect H side ∈ circumcircle ABC.side) :=
by 
  sorry

end reflections_of_orthocenter_on_circumcircle_l106_106670


namespace binom_18_10_l106_106439

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106439


namespace part1_part2_1_part2_2_part2_3_l106_106964

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 - a) * x - 2 * a

-- Conditions
variables (a : ℝ) (u v w : ℝ)
def symmetric_about_one : Prop := ∀ x : ℝ, f (2 - x) a = f x a

-- Problem 1: Range of f on [0, 4] if symmetric about x = 1
theorem part1 (H : symmetric_about_one a): set.range (λ x : ℝ, f x 4) ∩ set.Icc 0 4 = set.Icc (-9 : ℝ) (-5 : ℝ) :=
sorry

-- Problem 2: Range of f(x) > 0
theorem part2_1 (Ha : a = -2) : ∀ x : ℝ, x ≠ -2 → f x a > 0 :=
sorry

theorem part2_2 (Ha : a > -2) : ∀ x : ℝ, (x < -2 ∨ x > a) → f x a > 0 :=
sorry

theorem part2_3 (Ha : a < -2) : ∀ x : ℝ, -2 < x ∧ x < a → f x a > 0 :=
sorry

end part1_part2_1_part2_2_part2_3_l106_106964


namespace arc_length_of_curve_l106_106422

def f (x : ℝ) : ℝ := 1 - Real.log (x^2 - 1)

def arcLength (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (Real.deriv f x) ^ 2)

theorem arc_length_of_curve : arcLength f 3 4 = 1 + 2 * Real.log (6 / 5) :=
by
  sorry

end arc_length_of_curve_l106_106422


namespace distance_proof_l106_106287

-- Define the given problem and conditions
variable (R d : Float)

def distanceFromChord (R d : Float) : Float :=
  Real.sqrt ((1/2) * R * (R + d))

-- State the theorem to be proven
theorem distance_proof :
  distanceFromChord R d = Real.sqrt ((1/2) * R * (R + d)) :=
by
  sorry

end distance_proof_l106_106287


namespace problem1_problem2_problem3_problem4_l106_106066

-- Question 1
theorem problem1 : (-0.5) - (-3.2) + (2.8) - (6.5) = -1 := 
by 
  sorry

-- Question 2
theorem problem2 : - (155 + 15/38) ÷ 5 = - (31 + 3/38) := 
by 
  sorry

-- Question 3
theorem problem3 : 17 - 8 ÷ (-2) + 4 * (-5) - 1 ÷ 2 * (1/2) = 3/4 := 
by 
  sorry

-- Question 4
theorem problem4 : [-1 ^ 2023 + (-3) ^ 2 * (1/3 - 1/2)] * 3/10 ÷ (-0.1 ^ 2) = 75 := 
by 
  sorry

end problem1_problem2_problem3_problem4_l106_106066


namespace max_volume_pyramid_l106_106792

theorem max_volume_pyramid 
  (AB AC : ℝ)
  (sin_BAC : ℝ)
  (angle_cond : ∀ (SA SB SC : ℝ), SA = SB ∧ SB = SC ∧ SC = SA → ∀ θ, θ ≤ 60 → true)
  (h : ℝ)
  (V : ℝ)
  (AB_eq : AB = 3)
  (AC_eq : AC = 5)
  (sin_BAC_eq : sin_BAC = 3/5)
  (height_cond : h = (5 * Real.sqrt 3) / 2)
  (volume_cond : V = (1/3) * (1/2 * 3 * 5 * (3/5)) * h) :
  V = (5 * Real.sqrt 174) / 4 := sorry

end max_volume_pyramid_l106_106792


namespace polynomial_root_condition_l106_106521

theorem polynomial_root_condition (a : ℝ) :
  (∃ x1 x2 x3 : ℝ,
    (x1^3 - 6 * x1^2 + a * x1 + a = 0) ∧
    (x2^3 - 6 * x2^2 + a * x2 + a = 0) ∧
    (x3^3 - 6 * x3^2 + a * x3 + a = 0) ∧
    ((x1 - 3)^3 + (x2 - 3)^3 + (x3 - 3)^3 = 0)) →
  a = -9 :=
by
  sorry

end polynomial_root_condition_l106_106521


namespace heating_rate_l106_106857

/-- 
 Andy is making fudge. He needs to raise the temperature of the candy mixture from 60 degrees to 240 degrees. 
 Then, he needs to cool it down to 170 degrees. The candy heats at a certain rate and cools at a rate of 7 degrees/minute.
 It takes 46 minutes for the candy to be done. Prove that the heating rate is 5 degrees per minute.
-/
theorem heating_rate (initial_temp heating_temp cooling_temp : ℝ) (cooling_rate total_time : ℝ) 
  (h1 : initial_temp = 60) (h2 : heating_temp = 240) (h3 : cooling_temp = 170) 
  (h4 : cooling_rate = 7) (h5 : total_time = 46) : 
  ∃ (H : ℝ), H = 5 :=
by 
  -- We declare here that the rate H exists and is 5 degrees per minute.
  let H : ℝ := 5
  existsi H
  sorry

end heating_rate_l106_106857


namespace remaining_pentagon_perimeter_l106_106402

theorem remaining_pentagon_perimeter : 
  ∀ (P Q R S T U : Type) 
  (side_length_PQR side_length_STU : ℕ) 
  (h1 : side_length_PQR = 5) 
  (h2 : side_length_STU = 2) 
  (h3 : P ≠ Q ∧ Q ≠ R ∧ R ≠ P)
  (h4 : S = Q) 
  (is_equilateral_triangle_PQR : triangle P Q R) 
  (is_equilateral_triangle_STU : triangle S T U)
  (cut_triangle_from_PQR : ∃ (P' : Type) (Q' : Type) (R' : Type), P' = P ∧ Q' = Q),
  ∃ (pentagon : ℕ), pentagon = 14 :=
by 
  sorry

end remaining_pentagon_perimeter_l106_106402


namespace quadratic_decrease_y_lt_0_l106_106768

/--
Let $y = -x^2 + 6x - 5$.
Prove that when $x > 5$, $y < 0$ and $y$ decreases as $x$ increases.
-/
theorem quadratic_decrease_y_lt_0 (x : ℝ) :
  (x > 5) → (y = -x^2 + 6x - 5) → (y < 0) ∧ (∀ x₁ x₂, x₁ > x₂ → y₁ < y₂) :=
sorry

end quadratic_decrease_y_lt_0_l106_106768


namespace area_of_triangular_cross_section_l106_106351

theorem area_of_triangular_cross_section {A B C D M : Type*} 
  (hTetra : tetrahedron A B C D) 
  (triCrossSec : triangle A B M) 
  : 
  (area triCrossSec) ≤ max (area (triangle A B C)) (area (triangle A B D)) := 
by
  sorry

end area_of_triangular_cross_section_l106_106351


namespace closest_integer_ratio_l106_106999

theorem closest_integer_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
    (h4 : ((a + b) / 2) = 2 * Real.sqrt (a * b)) : 
    ∃ (n : ℤ), n = 14 ∧ closest_integer (a / b) n :=
by
  sorry

end closest_integer_ratio_l106_106999


namespace periodic_function_l106_106686

noncomputable theory
open Function

theorem periodic_function {f : ℝ → ℝ} (h : ∀ x : ℝ, f(x + 1) + f(x - 1) = Real.sqrt 2 * f x) : ∃ T : ℝ, T ≠ 0 ∧ (∀ x : ℝ, f(x + T) = f x) := 
begin
  use 8,
  split,
  { norm_num,
  },
  { intro x,
    sorry,
  },
end

end periodic_function_l106_106686


namespace binomial_coeff_18_10_l106_106479

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l106_106479


namespace correct_propositions_count_l106_106957

def f (x : ℝ) : ℝ := 4 * Real.cos (2 * x + Real.pi / 3) + 1

def symmetry_center (p : ℝ × ℝ) : Prop := p = (-5 * Real.pi / 12, 0)
def graph_symmetry : Prop := ∀ x : ℝ, f (1 - x) = f (x - 1)
def negation_correct : Prop := (¬ ∀ x > 0, x^2 + 2 * x - 3 > 0) = (∃ x ≤ 0, x^2 + 2 * x - 3 ≤ 0)
def sine_ordering (α β : ℝ) (hα : α ∈ Icc 0 Real.pi) (hβ : β ∈ Icc 0 Real.pi) : Prop := α > β → ¬ (Real.sin α > Real.sin β)

theorem correct_propositions_count :
  (symmetry_center (-5 * Real.pi / 12, 0)) ∧ 
  graph_symmetry ∧ 
  negation_correct ∧ 
  ¬ (∀ α β : ℝ, ∀ hα : α ∈ Icc 0 Real.pi, ∀ hβ : β ∈ Icc 0 Real.pi, sine_ordering α β hα hβ) 
  → 3 = 3 := by
  intros
  sorry

end correct_propositions_count_l106_106957


namespace variance_not_2_minimum_standard_deviation_l106_106025

-- Definition for a right triangle with hypotenuse 3
structure RightTriangle where
  (a : ℝ) (b : ℝ)
  hypotenuse : ℝ := 3
  pythagorean_property : a^2 + b^2 = 9

-- Part (a) - Prove that the variance cannot be 2
theorem variance_not_2 (triangle : RightTriangle) : 
  (6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) ≠ 2 := sorry

-- Part (b) - Prove the minimum standard deviation and corresponding leg lengths
theorem minimum_standard_deviation (triangle : RightTriangle) : 
  (exists (a b : ℝ), a > 0 ∧ b > 0 ∧ a = b ∧ a = b = 3 * (real.sqrt 2) / 2 ∧ 
  real.sqrt ((6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) = real.sqrt (2) - 1)) := sorry

end variance_not_2_minimum_standard_deviation_l106_106025


namespace cos_7theta_l106_106185

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -45682/8192 :=
by
  sorry

end cos_7theta_l106_106185


namespace probability_exactly_one_instrument_l106_106206

-- Definitions of the conditions
def total_people : ℕ := 800
def frac_one_instrument : ℚ := 1 / 5
def people_two_or_more_instruments : ℕ := 64

-- Statement of the problem
theorem probability_exactly_one_instrument :
  let people_at_least_one_instrument := frac_one_instrument * total_people
  let people_exactly_one_instrument := people_at_least_one_instrument - people_two_or_more_instruments
  let probability := people_exactly_one_instrument / total_people
  probability = 3 / 25 :=
by
  -- Definitions
  let people_at_least_one_instrument : ℚ := frac_one_instrument * total_people
  let people_exactly_one_instrument : ℚ := people_at_least_one_instrument - people_two_or_more_instruments
  let probability : ℚ := people_exactly_one_instrument / total_people
  
  -- Sorry statement to skip the proof
  exact sorry

end probability_exactly_one_instrument_l106_106206


namespace find_b_l106_106196

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

def derivative_at_one (a : ℝ) : ℝ := a + 1

def tangent_line (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem find_b (a b : ℝ) (h_deriv : derivative_at_one a = 2) (h_tangent : tangent_line b 1 = curve a 1) :
  b = -1 :=
by
  sorry

end find_b_l106_106196


namespace binom_18_10_l106_106446

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106446


namespace cards_sum_divisible_by_100_l106_106562

theorem cards_sum_divisible_by_100 :
  let cards := {1, 2, …, 6000} in
  ∃ (pairs : finset (ℕ × ℕ)), pairs.card = 179940 ∧
  (∀ (x y : ℕ), (x, y) ∈ pairs → x ∈ cards ∧ y ∈ cards ∧ (x ≠ y) ∧ (x + y) % 100 = 0) :=
by
  sorry

end cards_sum_divisible_by_100_l106_106562


namespace math_problem_l106_106952

variable (X : ℕ → ℚ)
variable (P : ℚ → ℚ)
variable (a : ℚ)

noncomputable def condition1 : Prop := 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → P (X k) = a * k

noncomputable def condition2 : Prop :=
  ∑ k in finset.range 5, P (X (k + 1)) = 1

noncomputable def question : Prop :=
  P (1/5) + P (2/5) = 1/5

theorem math_problem : condition1 X P a ∧ condition2 X P →
  question X P :=
by
  sorry

end math_problem_l106_106952


namespace solution_set_l106_106192

-- Define the differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable [Differentiable ℝ f]

-- Define the conditions
def condition1 : Prop := ∀ x : ℝ, f x + deriv (deriv f) x > 2
def condition2 : Prop := f 0 = 4

-- Define the inequality to solve
def inequality (x : ℝ) : Prop := exp x * f x > 2 * exp x + 2

-- The proof statement
theorem solution_set (hf1 : condition1 f) (hf2 : condition2 f) :
  {x : ℝ | inequality f x} = {x : ℝ | x > 0} := by
  sorry

end solution_set_l106_106192


namespace intersection_A_B_l106_106578

-- Define set A
def A : Set ℕ := {0, 1, 2, 3}

-- Define set B based on the given condition
def B : Set ℝ := {x | 2 * x^2 - 9 * x + 9 ≤ 0}

-- Define the target intersection set AB
def AB : Set ℝ := {2, 3}

-- Prove the intersection of A and B is {2, 3}
theorem intersection_A_B : (A : Set ℝ) ∩ B = AB :=
by sorry

end intersection_A_B_l106_106578


namespace circle_radius_in_right_triangle_l106_106291

theorem circle_radius_in_right_triangle (c α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) :
    ∃ r : ℝ, r = c / (2 + real.cot (α / 2) + real.cot (π / 4 - α / 2)) :=
begin
  existsi (c / (2 + real.cot (α / 2) + real.cot (π / 4 - α / 2))),
  exact eq.refl _,
end

end circle_radius_in_right_triangle_l106_106291


namespace binom_18_10_eq_43758_l106_106461

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106461


namespace parabola_vertex_coordinates_l106_106741

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), (y = -3 * (x - 1)^2 - 2) → (x, y) = (1, -2) := 
begin
  intros x y h,
  sorry
end

end parabola_vertex_coordinates_l106_106741


namespace johns_time_learning_basics_l106_106663

theorem johns_time_learning_basics :
  ∀ (B : ℝ), 
  (let acclimation := 1 in
   let research := 1.75 * B in
   let dissertation := 0.5 in
   acclimation + B + research + dissertation = 7) →
  B = 2 := 
by
  intro B h
  sorry

end johns_time_learning_basics_l106_106663


namespace no_real_solution_l106_106519

-- Define the hypothesis: the sum of partial fractions
theorem no_real_solution : 
  ¬ ∃ x : ℝ, 
    (1 / ((x - 1) * (x - 3)) + 
     1 / ((x - 3) * (x - 5)) + 
     1 / ((x - 5) * (x - 7))) = 1 / 8 := 
by
  sorry

end no_real_solution_l106_106519


namespace binom_18_10_l106_106450

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l106_106450


namespace apples_harvested_from_garden_l106_106699

def number_of_pies : ℕ := 10
def apples_per_pie : ℕ := 8
def apples_to_buy : ℕ := 30

def total_apples_needed : ℕ := number_of_pies * apples_per_pie

theorem apples_harvested_from_garden : total_apples_needed - apples_to_buy = 50 :=
by
  sorry

end apples_harvested_from_garden_l106_106699


namespace problem_l106_106217

theorem problem : 
  let N := 63745.2981
  let place_value_7 := 1000 -- The place value of the digit 7 (thousands place)
  let place_value_2 := 0.1 -- The place value of the digit 2 (tenths place)
  place_value_7 / place_value_2 = 10000 :=
by
  sorry

end problem_l106_106217


namespace geometric_series_sum_l106_106906

theorem geometric_series_sum : 
  ∀ (a r : ℚ) (n : ℕ), 
  a = 1 / 4 ∧ r = 1 / 2 ∧ n = 7 → 
  a * (1 - r^n) / (1 - r) = 127 / 256 :=
by
  intros a r n h,
  cases h with h1 hrn,
  cases hrn with hr hn,
  rw [h1, hr, hn],
  sorry

end geometric_series_sum_l106_106906


namespace fixed_points_l106_106678

def f (n : ℕ) : ℕ :=
  if n > 2000 then n - 12 else f (f (n + 14))

theorem fixed_points : {n : ℕ | f n = n} = {1989, 1990} := 
sorry

end fixed_points_l106_106678


namespace find_k_value_l106_106574

noncomputable def ellipse_eccentricity (e : ℚ) := 
  e = (Real.sqrt 6) / 3

noncomputable def ab_length (len : ℚ) :=
  len = 2 * (Real.sqrt 3) / 3

noncomputable def ellipse_equation (a b : ℚ) :=
  Ellipse.equation (a^2) (b^2) = (x^2 / 3) + y^2 = 1

noncomputable def intersect_line_and_ellipse (k x1 x2 y1 y2 : ℚ) :=
  line y = k * x + 2 intersects ellipse ∧ 
  circumscribed_circle (-1, 0) through (x1, y1) and (x2, y2) ∧
  x1 + x2 = - (12 * k) / (1 + 3 * k^2) ∧ 
  x1 * x2 = 9 / (1 + 3 * k^2) ∧
  ((x1 + 1)*(x2 + 1) + y1*y2 = 0) → 
  k == (7 / 6)

theorem find_k_value : 
  ∃ (k : ℚ), intersect_line_and_ellipse k -1 -1 -1 -1 := 
sorry

end find_k_value_l106_106574


namespace pure_imaginary_a_value_l106_106138

theorem pure_imaginary_a_value (a : ℝ) : (∃ b : ℝ, (1 + 2 * complex.I) / (a + complex.I) = b * complex.I) → a = -2 :=
by
  sorry

end pure_imaginary_a_value_l106_106138


namespace integral_x_squared_plus_sin_x_l106_106072

theorem integral_x_squared_plus_sin_x :
  ∫ x in -1..1, x^2 + sin x = 2/3 :=
by
  sorry

end integral_x_squared_plus_sin_x_l106_106072


namespace cafeteria_extra_fruits_l106_106769

def extra_fruits (ordered wanted : Nat) : Nat :=
  ordered - wanted

theorem cafeteria_extra_fruits :
  let red_apples_ordered := 6
  let red_apples_wanted := 5
  let green_apples_ordered := 15
  let green_apples_wanted := 8
  let oranges_ordered := 10
  let oranges_wanted := 6
  let bananas_ordered := 8
  let bananas_wanted := 7
  extra_fruits red_apples_ordered red_apples_wanted = 1 ∧
  extra_fruits green_apples_ordered green_apples_wanted = 7 ∧
  extra_fruits oranges_ordered oranges_wanted = 4 ∧
  extra_fruits bananas_ordered bananas_wanted = 1 := 
by
  sorry

end cafeteria_extra_fruits_l106_106769


namespace sphere_ratios_l106_106968

theorem sphere_ratios (R r : ℝ) (h : R / r = 2 / 3) :
  (S1 S2 V1 V2 : ℝ)
  (S1 = 4 * Real.pi * R^2) 
  (S2 = 4 * Real.pi * r^2) 
  (V1 = (4 / 3) * Real.pi * R^3) 
  (V2 = (4 / 3) * Real.pi * r^3) :
  (S1 / S2 = 4 / 9) ∧ (V1 / V2 = 8 / 27) := by
  sorry

end sphere_ratios_l106_106968


namespace ball_returns_to_Bella_after_14_throws_l106_106911

def num_girls := 14

def next_girl (current_girl : Nat) : Nat :=
  (current_girl + 3) % num_girls

theorem ball_returns_to_Bella_after_14_throws :
  ∃ n : Nat, n = 14 ∧ (Array.iterate (fun (current : Nat) => next_girl current) n 1) = 1 := 
  by
  sorry

end ball_returns_to_Bella_after_14_throws_l106_106911


namespace div_ad_bc_by_k_l106_106251

theorem div_ad_bc_by_k 
  (a b c d l k m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n) : 
  k ∣ (a * d - b * c) :=
sorry

end div_ad_bc_by_k_l106_106251


namespace greatest_x_l106_106333

theorem greatest_x (x : ℕ) (h : x^2 < 32) : x ≤ 5 := 
sorry

end greatest_x_l106_106333


namespace cube_inverse_sum_l106_106994

theorem cube_inverse_sum (x : ℂ) (h : x + 1/x = -3) : x^3 + (1/x)^3 = -18 :=
by
  sorry

end cube_inverse_sum_l106_106994


namespace remainder_when_dividing_polynomial_l106_106549

noncomputable def P(x : ℝ) := x^5 + 3
noncomputable def Q(x : ℝ) := (x - 3)^2

theorem remainder_when_dividing_polynomial :
  ∃ (R : ℝ → ℝ), (λ x, R x) = (λ x, 405 * x - 969) ∧ (∃ (S : ℝ → ℝ), P = λ x, Q(x) * S(x) + R(x)) :=
sorry

end remainder_when_dividing_polynomial_l106_106549


namespace pencil_count_l106_106286

theorem pencil_count (P N X : ℝ) 
  (h1 : 96 * P + 24 * N = 520) 
  (h2 : X * P + 4 * N = 60) 
  (h3 : P + N = 15.512820512820513) :
  X = 3 :=
by
  sorry

end pencil_count_l106_106286


namespace cut_rectangle_into_square_l106_106885

theorem cut_rectangle_into_square :
  ∃ (parts : list (finset (fin 16 × fin 16))),
    (∀ p ∈ parts, ∃ n m, ¬(n = 1 ∧ m = 1) ∧ (finset.card p = n * m)) ∧
    (finset.univ = parts.sum) :=
begin
  -- Proof goes here
  sorry
end

end cut_rectangle_into_square_l106_106885


namespace find_k_l106_106633

theorem find_k (k : ℝ) :
  ∃ f g : ℝ → ℝ × ℝ → ℝ, (∀ x y, (2 * x ^ 2 - 6 * y ^ 2 + x * y + k * x + 6 = f x y * g x y)) →
  (k = 7 ∨ k = -7) :=
by
  sorry

end find_k_l106_106633


namespace convert_spherical_to_rectangular_l106_106882

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular (
  (ρ θ φ : ℝ) (hρ : ρ = 3) (hθ : θ = 3*Real.pi/2) (hφ : φ = Real.pi/3) :
  spherical_to_rectangular ρ θ φ = (0, -3*Real.sqrt 3 / 2, 3 / 2) :=
by
  rw [hρ, hθ, hφ]
  simp [spherical_to_rectangular]
  sorry

end convert_spherical_to_rectangular_l106_106882


namespace exists_lucky_n_tuple_l106_106227

theorem exists_lucky_n_tuple (n : ℕ) : 
  (∃ (a : ℕ → ℕ), 
    (∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → i ≠ j → a i % n ≠ a j % n) ∧
    (∀ k, 1 ≤ k ∧ k < n →
       let diffs := a (k+1) - a k in
       diffs % n ≠ 0 ∧
       ∀ l m, 1 ≤ l ∧ l < n → 1 ≤ m ∧ m < n → l ≠ m → diffs % n ≠ (a (l+1) - a l) % n)) ↔
  (n = 1 ∨ n % 2 = 0) := 
by
  sorry

end exists_lucky_n_tuple_l106_106227


namespace distinct_equilateral_triangles_count_l106_106611

-- Define the regular decagon and the concept of equilateral triangle
def regular_decagon_vertices (A : ℕ → (ℝ × ℝ)) :=
  ∀ i, A i = (cos (2 * i * π / 10), sin (2 * i * π / 10))

-- Definition of an equilateral triangle condition
def is_equilateral_triangle (T : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ⟨(x1, y1), (x2, y2), (x3, y3)⟩ := T in
  (x2 - x1)^2 + (y2 - y1)^2 = (x3 - x2)^2 + (y3 - y2)^2 ∧
  (x3 - x2)^2 + (y3 - y2)^2 = (x1 - x3)^2 + (y1 - y3)^2

-- Now state the problem in Lean 4
theorem distinct_equilateral_triangles_count :
  let A := λ i : ℕ, (cos (2 * i * π / 10), sin (2 * i * π / 10)) in
  regular_decagon_vertices A →
  (∃ S : finset ((ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)),
    S.card = 84 ∧
    (∀ T ∈ S, ∃ i j : ℕ,
      0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 10 ∧
      i ≠ j ∧ is_equilateral_triangle (T) ∧
      T.1 = A i ∨ T.2 = A j)) :=
sorry

end distinct_equilateral_triangles_count_l106_106611


namespace donald_laptop_cost_l106_106510

theorem donald_laptop_cost (original_price : ℕ) (reduction_percent : ℕ) (reduced_price : ℕ) (h1 : original_price = 800) (h2 : reduction_percent = 15) : reduced_price = 680 :=
by
  -- Definitions of the conditions
  have h3 : reduction_percent / 100 * original_price = 120 := sorry  -- Calculation of the discount (15/100)*800
  have h4 : original_price - 120 = 680 := sorry  -- Subtracting discount from original price
  -- Conclusion
  exact h4

end donald_laptop_cost_l106_106510


namespace smaller_square_area_l106_106399

theorem smaller_square_area (ABCD_radius : ℝ) (EFGH_side : ℝ) 
(h1 : ABCD_radius = real.sqrt 2 / 2)
(h2 : EFGH_side^2 = (1:ℝ/25)) :
    EFGH_side^2 = (1/25)
:= by
    sorry

end smaller_square_area_l106_106399


namespace minimum_value_f_a_is_3_over_8_has_exactly_one_zero_two_zeros_implies_a_in_0_1_l106_106162

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - x - real.log x

/-- Question 1: Prove that when a = 3/8, the minimum value of f(x) is -1/2 - ln(2) -/
theorem minimum_value_f_a_is_3_over_8 : 
  ∃ x, f (3 / 8) x = -(1 / 2) - real.log 2 := sorry

/-- Question 2: Prove that f(x) has exactly one zero when -1 <= a <= 0 -/
theorem has_exactly_one_zero (a : ℝ) (h : -1 ≤ a ∧ a ≤ 0) : 
  ∃! x, f a x = 0 := sorry

/-- Question 3: If f(x) has two zeros, then a is in the interval (0, 1) -/
theorem two_zeros_implies_a_in_0_1 (a : ℝ) : 
  (∃ x1 x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → (0 < a ∧ a < 1) := sorry

end minimum_value_f_a_is_3_over_8_has_exactly_one_zero_two_zeros_implies_a_in_0_1_l106_106162


namespace unique_rational_point_on_sphere_l106_106502
open Real

-- Define a predicate for a rational point in 3D space
def is_rational_point (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y z : ℚ), (p = (x, y, z))

-- Define the specific sphere equation
def on_sphere (p : ℝ × ℝ × ℝ) : Prop :=
  (p.1 - Real.sqrt 2) ^ 2 + p.2 ^ 2 + p.3 ^ 2 = 2

theorem unique_rational_point_on_sphere :
  ∃! (p : ℝ × ℝ × ℝ), is_rational_point p ∧ on_sphere p :=
begin
  sorry
end

end unique_rational_point_on_sphere_l106_106502


namespace locus_of_point_C_l106_106103

theorem locus_of_point_C (A B C : Type*) [MetricSpace C] (a b d c : ℝ) (M : C) 
  (distA : C → C) (distB : C → C)
  (hAB : distA A = distA B)
  (hM : distA M = distB M)
  (h1 : a^2 + b^2 = d^2)
  (h2 : distA C = a)
  (h3 : distB C = b)
  (midpoint_eq : distA M = distB M)
  (h4 : (A, B).dist = c)
  (h5 : c = 2 * distA M)
  (valid_radius : d^2 > c^2 / 2) :
∃ r : ℝ, r = sqrt(d^2 / 2 - (c / 2)^2) ∧ (distB C = distA M + r) := sorry

end locus_of_point_C_l106_106103


namespace lean_problem_statement_l106_106953

noncomputable def standard_equation_of_ellipse (a b : ℝ) (h_ellipse : (a > b) ∧ (b > 0)) (point_on_ellipse : Real.sqrt 3 / 2) (focal_length : ℝ) : ℝ := 
  let c := Real.sqrt 3
  let eq1 := 1 / (a ^ 2) + 3 / (4 * b ^ 2) = 1
  let eq2 := b ^ 2 = a ^ 2 - 3
  let a2 := 4
  let b2 := 1
  eq a2 * x^2 + b2 * y^2 = 1

noncomputable def intersect_line_ellipse (a b : ℝ) (m : ℝ) (intersect_ellipse : Real.sqrt 2 * x + m) (tan_AMB_eq : ℝ) : ℝ:= 
  let tanAMC := Real.sqrt 2
  let eq := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ∧ (m = 1 ∨ m = -1)
  eq 
  
-- Proof is omitted.
theorem lean_problem_statement (a b : ℝ) (m : ℝ)
  (h_ellipse : (a > b) ∧ (b > 0)) 
  (point_on_ellipse : (1, -Real.sqrt(3) / 2))
  (focal_length : 2 * Real.sqrt 3 = 3) 
  (line_l : y = Real.sqrt(2) * x + m)
  (tan_AMB_eq : tan(∠ AMB) = -2 * Real.sqrt(2)) : 
  ∃ a b m, (4 * x^2 + y^2 = 1) ∧ (m = 1 ∨ m = -1) := 
  sorry

end lean_problem_statement_l106_106953


namespace fraction_equivalent_to_0_46_periodic_l106_106525

theorem fraction_equivalent_to_0_46_periodic :
  let a := (46 : ℚ) / 100
  let r := (1 : ℚ) / 100
  let geometric_series_sum (a r : ℚ) :=
    if r.abs < 1 then a / (1 - r) else 0
  geometric_series_sum a r = 46 / 99 := by
    sorry

end fraction_equivalent_to_0_46_periodic_l106_106525


namespace binom_18_10_l106_106440

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106440


namespace binom_18_10_l106_106443

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106443


namespace binom_18_10_l106_106434

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106434


namespace hannah_highest_score_l106_106623

-- Definitions based on conditions
def total_questions : ℕ := 40
def wrong_questions : ℕ := 3
def correct_percent_student_1 : ℝ := 0.95

-- The Lean statement representing the proof problem
theorem hannah_highest_score :
  ∃ q : ℕ, (q > (total_questions - wrong_questions) ∧ q > (total_questions * correct_percent_student_1)) ∧ q = 39 :=
by
  sorry

end hannah_highest_score_l106_106623


namespace sum_series_eq_final_sum_l106_106074

-- Given N is greater than or equal to 2
variable (N : ℕ) (h : 2 ≤ N)

-- Define the series term
def seriesTerm (n : ℕ) : ℝ :=
  (6 * n^3 - 2 * n^2 - 2 * n + 2) / (n^6 - n^5 + n^4 - n^3 + n^2 - n)

-- Statement of the theorem
theorem sum_series_eq_final_sum :
  ∑ n in Finset.range (N - 1) + 1, seriesTerm n = FinalSum := 
begin
  -- Placeholder for the actual proof
  sorry
end

end sum_series_eq_final_sum_l106_106074


namespace hyperbola_equation_l106_106606

variable (a b c : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (asymptote_cond : -b / a = -1 / 2)
variable (foci_cond : c = 5)
variable (hyperbola_rel : a^2 + b^2 = c^2)

theorem hyperbola_equation : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ -b / a = -1 / 2 ∧ c = 5 ∧ a^2 + b^2 = c^2 
  ∧ ∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1)) := 
sorry

end hyperbola_equation_l106_106606


namespace sin_prob_gt_half_l106_106190

noncomputable theory

open Real

theorem sin_prob_gt_half (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) : 
  let prob := (pi / 2) / pi in
  prob = 1 / 2 := 
by
  sorry

end sin_prob_gt_half_l106_106190


namespace g_primitivable_l106_106252

variable (f : ℝ → ℝ)
variable (hf1 : f 0 = 0)
variable (hf2 : ∀ x, f' x ≠ 0)
variable (hf3 : ∀ x, ∃ df₂, DifferentiableAt ℝ (fun y => deriv (fun x => deriv (f:ℝ → ℝ) x) y) x)

def g (x : ℝ) : ℝ :=
if h : x = 0 then 0 else cos (1 / f x)

theorem g_primitivable : ∃ (F : ℝ → ℝ), deriv F = g :=
sorry

end g_primitivable_l106_106252


namespace consecutive_integers_divisible_by_12_l106_106191

theorem consecutive_integers_divisible_by_12 (a b c d : ℤ) 
  (h1 : b = a + 1) (h2 : c = b + 1) (h3 : d = c + 1) : 
  12 ∣ (a * b + a * c + a * d + b * c + b * d + c * d + 1) := 
sorry

end consecutive_integers_divisible_by_12_l106_106191


namespace problem_omega_value_and_range_l106_106961

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (ω * x) + cos (2 * ω * x)

theorem problem_omega_value_and_range :
  ∀ (ω : ℝ), ω > 0 →
  (∀ x, f ω x = 2 * sin (ω * x) + cos (2 * ω * x)) →
  ∃ t > 0, (∀ x, f ω (x + t) = f ω x) →
  (t = π ↔ ω = 1) →
  (∀ x, x ∈ set.Ioo 0 (π / 2) → f 1 x ∈ set.Ioo (-1) (sqrt 2)) :=
by
  sorry

end problem_omega_value_and_range_l106_106961


namespace problem1_problem2_problem3_problem4_l106_106423

-- Problem 1: Prove (1 * -6) + -13 = -19
theorem problem1 : (1 * -6) + -13 = -19 := by 
  sorry

-- Problem 2: Prove (3/5) + (-3/4) = -3/20
theorem problem2 : (3/5 : ℚ) + (-3/4) = -3/20 := by 
  sorry

-- Problem 3: Prove 4.7 + (-0.8) + 5.3 + (-8.2) = 1
theorem problem3 : (4.7 + (-0.8) + 5.3 + (-8.2) : ℝ) = 1 := by 
  sorry

-- Problem 4: Prove (-1/6) + (1/3) + (-1/12) = 1/12
theorem problem4 : (-1/6 : ℚ) + (1/3) + (-1/12) = 1/12 := by 
  sorry

end problem1_problem2_problem3_problem4_l106_106423


namespace Kristy_baked_cookies_l106_106667

theorem Kristy_baked_cookies 
  (ate_by_Kristy : ℕ) (given_to_brother : ℕ) 
  (taken_by_first_friend : ℕ) (taken_by_second_friend : ℕ)
  (taken_by_third_friend : ℕ) (cookies_left : ℕ) 
  (h_K : ate_by_Kristy = 2) (h_B : given_to_brother = 1) 
  (h_F1 : taken_by_first_friend = 3) (h_F2 : taken_by_second_friend = 5)
  (h_F3 : taken_by_third_friend = 5) (h_L : cookies_left = 6) :
  ate_by_Kristy + given_to_brother 
  + taken_by_first_friend + taken_by_second_friend 
  + taken_by_third_friend + cookies_left = 22 := 
by
  sorry

end Kristy_baked_cookies_l106_106667


namespace functional_solution_l106_106896

theorem functional_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(f(x)^2 + f(y)) = x * f(x) + y) →
  (∀ x : ℝ, f(x) = x ∨ f(x) = -x) :=
by
  intros h x
  sorry

end functional_solution_l106_106896


namespace pool_width_l106_106556

variable (length : ℝ) (depth : ℝ) (chlorine_per_cubic_foot : ℝ) (chlorine_cost_per_quart : ℝ) (total_spent : ℝ)
variable (w : ℝ)

-- defining the conditions
def pool_conditions := length = 10 ∧ depth = 6 ∧ chlorine_per_cubic_foot = 120 ∧ chlorine_cost_per_quart = 3 ∧ total_spent = 12

-- goal statement
theorem pool_width : pool_conditions length depth chlorine_per_cubic_foot chlorine_cost_per_quart total_spent →
  w = 8 :=
by
  sorry

end pool_width_l106_106556


namespace palindrome_prime_diff_l106_106387

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def prime_differences (l : List ℕ) : List ℕ :=
  (l.zip l.tail).map (λ p => (p.2 - p.1).natAbs)

theorem palindrome_prime_diff (l : List ℕ) : 
  (∀ x ∈ l, is_palindrome x ∧ x > 0) →
  List.sorted (.≤) l →
  ∀ d ∈ (prime_differences l), d = 2 ∨ d = 11 := by
  sorry

end palindrome_prime_diff_l106_106387


namespace parking_arrangements_l106_106207

theorem parking_arrangements : 
  let spaces := {1, 2, 3, 4, 5, 6, 7, 8},
      vehicles : Type := @Fin 4,
      vehicle_type : vehicles → Prop := λ x, x < 2 ∨ x >= 2,
      adjacent (p q : ℕ) : Prop := abs (p - q) = 1
  in
  ∑ (t₁ t₂ c₁ c₂ : ℕ) in spaces,
    vehicle_type t₁ = tt ∧ vehicle_type t₂ = tt ∧ vehicle_type c₁ = ff ∧ vehicle_type c₂ = ff ∧
    adjacent t₁ t₂ ∧ adjacent c₁ c₂ ∧ 
    t₁ ≠ t₂ ∧ t₁ ≠ c₁ ∧ t₁ ≠ c₂ ∧ t₂ ≠ c₁ ∧ t₂ ≠ c₂ ∧ c₁ ≠ c₂
  = 120 :=
begin
  sorry
end

end parking_arrangements_l106_106207


namespace problem_solution_l106_106189

theorem problem_solution
  (x : ℝ) (a b : ℕ) (hx_pos : 0 < x) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_eq : x ^ 2 + 5 * x + 5 / x + 1 / x ^ 2 = 40)
  (h_form : x = a + Real.sqrt b) :
  a + b = 11 :=
sorry

end problem_solution_l106_106189


namespace cube_coloring_methods_l106_106069

theorem cube_coloring_methods (colors : Fin 4) : ∃ N : ℕ, N = 96 ∧ 
  ∀ (faces : Fin 6 → colors) (adj : (Fin 6 → Fin 6) → Prop), 
  (∀ i j, adj i j → faces i ≠ faces j) :=
by
  sorry

end cube_coloring_methods_l106_106069


namespace additional_oxygen_time_l106_106418

-- Definition of the problem conditions
def S0 : ℝ := 0.6
def S1 : ℝ := 0.8
def S_target : ℝ := 0.9
def t1 : ℝ := 1

-- Exponential model for blood oxygen saturation
def S (t : ℝ) (K : ℝ) : ℝ := S0 * Real.exp(K * t)

-- Given information: S(1) = 0.8, hence we derive K
noncomputable def K : ℝ := Real.log(S1 / S0)

-- Goal: Find the additional time required to reach 0.9 saturation
noncomputable def t_total : ℝ := (Real.log(S_target / S0)) / K

-- Additional time is the total time minus the initial 1 hour.
noncomputable def t_additional : ℝ := t_total - t1

-- Theorem stating that the calculated additional time equals 0.5 hours
theorem additional_oxygen_time :
  t_additional = 0.5 := by sorry

end additional_oxygen_time_l106_106418


namespace S_k_S_n_minus_k_geq_binom_squared_prod_l106_106685

noncomputable def S_k (a : ℕ → ℝ) (n k : ℕ) : ℝ :=
  ∑ b in Finset.powersetLen k (Finset.range n), (b.val.map a).prod

theorem S_k_S_n_minus_k_geq_binom_squared_prod 
  (a : ℕ → ℝ) (n k : ℕ) (hk : 1 ≤ k) (hnk : k ≤ n) : 
  S_k a n k * S_k a n (n - k) ≥ (nat.choose n k)^2 * (Finset.range n).prod a :=
sorry

end S_k_S_n_minus_k_geq_binom_squared_prod_l106_106685


namespace shelter_cats_l106_106202

theorem shelter_cats (initial_dogs initial_cats additional_cats : ℕ) 
  (h1 : initial_dogs = 75)
  (h2 : initial_dogs * 7 = initial_cats * 15)
  (h3 : initial_dogs * 11 = 15 * (initial_cats + additional_cats)) : 
  additional_cats = 20 :=
by
  sorry

end shelter_cats_l106_106202


namespace final_amount_to_pay_l106_106974

noncomputable def ring_cost : Float := 12.0
noncomputable def num_rings : Float := 2.0
noncomputable def sales_tax_rate : Float := 0.05
noncomputable def discount_rate : Float := 0.10

theorem final_amount_to_pay :
  let total_cost := ring_cost * num_rings in
  let sales_tax := total_cost * sales_tax_rate in
  let before_discount := total_cost + sales_tax in
  let discount := before_discount * discount_rate in
  let final_cost := before_discount - discount in
  final_cost = 22.68 :=
by
  sorry

end final_amount_to_pay_l106_106974


namespace magdalena_fraction_picked_l106_106697

noncomputable def fraction_picked_first_day
  (produced_apples: ℕ)
  (remaining_apples: ℕ)
  (fraction_picked: ℚ) : Prop :=
  ∃ (f : ℚ),
  produced_apples = 200 ∧
  remaining_apples = 20 ∧
  (f = fraction_picked) ∧
  (200 * f + 2 * 200 * f + (200 * f + 20)) = 200 - remaining_apples ∧
  fraction_picked = 1 / 5

theorem magdalena_fraction_picked :
  fraction_picked_first_day 200 20 (1 / 5) :=
sorry

end magdalena_fraction_picked_l106_106697


namespace inequality_proof_l106_106582

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b+c-a)^2 / ((b+c)^2+a^2) + (c+a-b)^2 / ((c+a)^2+b^2) + (a+b-c)^2 / ((a+b)^2+c^2) ≥ 3 / 5 :=
by sorry

end inequality_proof_l106_106582


namespace number_of_strikers_l106_106396

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end number_of_strikers_l106_106396


namespace rhino_reaches_target_l106_106811

-- Define the state of a rhino as a tuple of four integers (a, b, c, d)
structure RhinoState where
  a : ℕ  -- Vertical folds on the left side
  b : ℕ  -- Horizontal folds on the left side
  c : ℕ  -- Vertical folds on the right side
  d : ℕ  -- Horizontal folds on the right side
  deriving DecidableEq

-- Initial state (0,2,2,1)
def initialState : RhinoState := ⟨0, 2, 2, 1⟩

-- Target state (2,0,2,1)
def targetState : RhinoState := ⟨2, 0, 2, 1⟩

-- Transition relation based on the problem's conditions
-- upDownTransition: (a b c d) -> (a b' c' d') when 2 horizontal folds are smoothed out
-- frontBackTransition: (a b c d) -> (a' b' c d') when 2 vertical folds are smoothed out

def upDownTransition (s : RhinoState) : Option RhinoState :=
  if s.b ≥ 2 then some ⟨s.a, s.b - 2, s.c + 1, s.d + 1⟩ else none

def frontBackTransition (s : RhinoState) : Option RhinoState :=
  if s.a ≥ 2 then some ⟨s.a - 2, s.b + 1, s.c + 1, s.d⟩ else none

-- Reachability definition: given an initial state, can we eventually reach a target state
def reachable : RhinoState → RhinoState → Prop 
| s t := (∀ n, (upDownTransition s = some t ∨ frontBackTransition s = some t ∨ sorry))

-- Formal statement to be proven
theorem rhino_reaches_target :
  reachable initialState targetState :=
sorry

end rhino_reaches_target_l106_106811


namespace range_of_m_for_g_decreasing_range_of_m_for_h_two_zeros_l106_106160

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

noncomputable def g (m x : ℝ) : ℝ := m * x / (1 + x)

def is_decreasing_on_negative_one_infinity (m : ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), -1 < x1 ∧ x1 < x2 → g m x1 > g m x2

def h (m x : ℝ) : ℝ := f x + g m x

def has_two_distinct_zeros_in_interval (m : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), -1 < x1 ∧ x1 < 1 ∧ -1 < x2 ∧ x2 < 1 ∧ x1 ≠ x2 ∧ h m x1 = 0 ∧ h m x2 = 0

theorem range_of_m_for_g_decreasing (m : ℝ) :
  is_decreasing_on_negative_one_infinity m → m < 0 :=
sorry

theorem range_of_m_for_h_two_zeros (m : ℝ) :
  (m < 0) ∧ (m ∈ (-1, 0) ∨ m = (-(1:ℝ) - Real.sqrt 2) / 2) →
  (has_two_distinct_zeros_in_interval m →
  (m ∈ (-1, 0) ∨ m = (-(1:ℝ) - Real.sqrt 2) / 2)) :=
sorry

end range_of_m_for_g_decreasing_range_of_m_for_h_two_zeros_l106_106160


namespace cos_beta_value_l106_106137

theorem cos_beta_value (α β : ℝ) 
  (h1 : cos α = 1 / 7) 
  (h2 : cos (α + β) = -11 / 14) 
  (h3 : 0 < α) 
  (h4 : α < π / 2) 
  (h5 : 0 < β) 
  (h6 : β < π / 2) : 
  cos β = 1 / 2 :=
sorry

end cos_beta_value_l106_106137


namespace incenter_exists_l106_106658

variables {X Y Z : Type} [AddCommGroup X] [Module ℝ X]
variables (x y z : ℝ) (vX vY vZ : X)
variable (J : X)

-- Conditions
def sides_are_given : Prop :=
  x = 8 ∧ y = 11 ∧ z = 5

def is_incenter (p q r : ℝ) : Prop :=
  J = p • vX + q • vY + r • vZ ∧ p + q + r = 1

-- Claim
theorem incenter_exists
  (p q r : ℝ) (hp : p = 1 / 3) (hq : q = 11 / 24) (hr : r = 5 / 24) :
  sides_are_given x y z →
  is_incenter x y z vX vY vZ J p q r := by
  sorry

end incenter_exists_l106_106658


namespace train_speed_l106_106347

-- Define the conditions in terms of distance and time
def train_length : ℕ := 160
def crossing_time : ℕ := 8

-- Define the expected speed
def expected_speed : ℕ := 20

-- The theorem stating the speed of the train given the conditions
theorem train_speed : (train_length / crossing_time) = expected_speed :=
by
  -- Note: The proof is omitted
  sorry

end train_speed_l106_106347


namespace train_passes_platform_in_43_2_seconds_l106_106047

open Real

noncomputable def length_of_train : ℝ := 360
noncomputable def length_of_platform : ℝ := 180
noncomputable def speed_of_train_kmph : ℝ := 45
noncomputable def speed_of_train_mps : ℝ := (45 * 1000) / 3600  -- Converting km/hr to m/s

noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_of_train_mps

theorem train_passes_platform_in_43_2_seconds :
  time_to_pass_platform = 43.2 := by
  sorry

end train_passes_platform_in_43_2_seconds_l106_106047


namespace sphere_with_one_rational_point_exists_l106_106506

theorem sphere_with_one_rational_point_exists :
  ∃ (x y z : ℚ), (x - real.sqrt 2)^2 + y^2 + z^2 = 2 ∧ (∀ (a b c : ℚ), (a - real.sqrt 2)^2 + b^2 + c^2 = 2 → (a = 0 ∧ b = 0 ∧ c = 0)) :=
begin
  sorry,
end

end sphere_with_one_rational_point_exists_l106_106506


namespace period_of_f_max_min_value_of_f_l106_106600

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3) + sin (2 * x - π / 3) + 2 * cos x ^ 2 - 1

theorem period_of_f : is_periodic f π :=
sorry

theorem max_min_value_of_f : 
  ∃ min max, min = -1 ∧ max = sqrt 2 ∧ ∀ x ∈ (-π / 4 .. π / 4), min ≤ f x ∧ f x ≤ max :=
sorry

end period_of_f_max_min_value_of_f_l106_106600


namespace range_of_a_l106_106601

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 - x ^ 2 + x - 5

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ x_max x_min : ℝ, x_max ≠ x_min ∧
  f a x_max = max (f a x_max) (f a x_min) ∧ f a x_min = min (f a x_max) (f a x_min)) → 
  a < 1 / 3 ∧ a ≠ 0 := sorry

end range_of_a_l106_106601


namespace regular_pay_limit_l106_106386

theorem regular_pay_limit (x : ℝ) : 3 * x + 6 * 13 = 198 → x = 40 :=
by
  intro h
  -- proof skipped
  sorry

end regular_pay_limit_l106_106386


namespace find_m_value_l106_106588

theorem find_m_value (m : ℝ) (h : ∃ x : ℝ, (sin (x + π/2) + cos (x - π/2) + m = 2 * √2)) : 
  m = √2 :=
sorry

end find_m_value_l106_106588


namespace this_week_usage_less_next_week_usage_less_l106_106799

def last_week_usage : ℕ := 91

def usage_this_week : ℕ := (4 * 8) + (3 * 10)

def usage_next_week : ℕ := (5 * 5) + (2 * 12)

theorem this_week_usage_less : last_week_usage - usage_this_week = 29 := by
  -- proof goes here
  sorry

theorem next_week_usage_less : last_week_usage - usage_next_week = 42 := by
  -- proof goes here
  sorry

end this_week_usage_less_next_week_usage_less_l106_106799


namespace packaging_combinations_l106_106820

theorem packaging_combinations (wrapping_papers ribbons gift_cards decorative_tags : ℕ) (h_wp : wrapping_papers = 10) (h_ribbons : ribbons = 5) (h_gc : gift_cards = 6) (h_dt : decorative_tags = 2) : wrapping_papers * ribbons * gift_cards * decorative_tags = 600 := by
  rw [h_wp, h_ribbons, h_gc, h_dt]
  norm_num
  sorry

end packaging_combinations_l106_106820


namespace probability_of_blue_candy_l106_106312

theorem probability_of_blue_candy (green blue red : ℕ) (h1 : green = 5) (h2 : blue = 3) (h3 : red = 4) :
  (blue : ℚ) / (green + blue + red : ℚ) = 1 / 4 :=
by
  rw [h1, h2, h3]
  norm_num


end probability_of_blue_candy_l106_106312


namespace set_intersection_l106_106235

noncomputable def A : Set ℤ := {-1, 0, 1, 2}

noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log 2 (4 - x^2) ∧ -2 < x ∧ x < 2}

theorem set_intersection : A ∩ B = {-1, 0, 1} :=
  sorry

end set_intersection_l106_106235


namespace part_1_part_2_l106_106129

noncomputable def a_seq : ℕ → ℚ
| 1 := 1
| n := (2 * (∑ i in finset.range (n+1), a_seq i)^2) / (2 * (∑ i in finset.range (n+1), a_seq i) - 1)

noncomputable def S_seq (n : ℕ) : ℚ := ∑ i in finset.range (n+1), a_seq i

theorem part_1 (n : ℕ) (hn : 2 ≤ n) :
  (1 / S_seq n) - (1 / S_seq (n - 1)) = 2 := sorry

theorem part_2 (n : ℕ) (hn : 2 ≤ n) :
  ∑ i in finset.range (n+1), (1 / (i + 1)) * S_seq (i + 1) < 3 / 2 := sorry

end part_1_part_2_l106_106129


namespace find_min_value_of_quadratic_l106_106965

def minimum_value (f : ℝ → ℝ) : ℝ := Inf (set.range f)

theorem find_min_value_of_quadratic :
  minimum_value (λ x : ℝ, x^2 + 6 * x + 13) = 4 :=
sorry

end find_min_value_of_quadratic_l106_106965


namespace star_two_three_star_two_neg_six_neg_two_thirds_l106_106887

def star (a b : ℝ) : ℝ := (a + b) / 3

theorem star_two_three : star 2 3 = 5 / 3 := by
  sorry

theorem star_two_neg_six_neg_two_thirds : star (star 2 (-6)) (-2 / 3) = -2 / 3 := by
  sorry

end star_two_three_star_two_neg_six_neg_two_thirds_l106_106887


namespace count_multiples_less_than_300_l106_106980

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end count_multiples_less_than_300_l106_106980


namespace range_of_b_l106_106559

variable (a b c : ℝ)

theorem range_of_b (h1 : a * c = b^2) (h2 : a + b + c = 3) : -3 ≤ b ∧ b ≤ 1 :=
sorry

end range_of_b_l106_106559


namespace a_minus_b_is_minus_seven_l106_106247

theorem a_minus_b_is_minus_seven (a b : ℝ) (i : ℂ) (h : i = complex.I) 
  (h_conj : complex.conj (a + b * i / i) = (2 - i) ^ 2) : 
  a - b = -7 :=
by
  sorry

end a_minus_b_is_minus_seven_l106_106247


namespace triangle_side_inequality_l106_106689

theorem triangle_side_inequality (a b c : ℝ) (γ : ℝ) (h1 : 0 ≤ γ ∧ γ ≤ π)
  (h2 : a > 0 ∧ b > 0 ∧ c > 0)
  (h3 : c = (a^2 + b^2 - 2*a*b*cos γ)^(1/2)) :
  c ≥ (a + b) * sin γ / 2 :=
by sorry

end triangle_side_inequality_l106_106689


namespace angleQDP_ninety_degrees_l106_106200

variables {A B P Q C D M l : Prop}
variables (PA PC PM QB QD QM : ℝ)

-- Given conditions
def midpoint (M : Prop) (A B : Prop) : Prop := 
  M = (A + B) / 2

def on_opposite_sides (P Q : Prop) (l : Prop) : Prop :=
  P ≠ Q ∧ P ∈ l ∧ Q ∈ l

def on_ray (C : Prop) (P A : Prop) : Prop :=
  C ∈ line_through P A

def perp (P Q : Prop) : Prop :=
  ∠P Q = 90

-- Main statement
theorem angleQDP_ninety_degrees :
  midpoint M A B →
  on_opposite_sides P Q l →
  on_ray C P A →
  on_ray D Q B →
  PA * PC = PM ^ 2 →
  QB * QD = QM ^ 2 →
  perp P C Q →
  perp Q D P := 
sorry

end angleQDP_ninety_degrees_l106_106200


namespace missing_number_is_twelve_l106_106895

theorem missing_number_is_twelve (x : ℝ) :
  abs (9 - 8 * (3 - x)) - abs (5 - 11) = 75 → x = 12 :=
by
  intro h
  rw [abs_sub (5 : ℝ) 11] at h
  norm_num at h
  sorry

end missing_number_is_twelve_l106_106895


namespace isosceles_trapezoid_area_l106_106899

theorem isosceles_trapezoid_area (h : ℝ) 
  (angle_circumscribed : ∀ {A B C D : Type}, is_circumscribed A B C D → central_angle A B C D = 60) : 
  ∃ S : ℝ, S = h^2 * sqrt 3 :=
by
  sorry

end isosceles_trapezoid_area_l106_106899


namespace domain_of_f_l106_106786

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = x} = {x : ℝ | x ≠ 6} := by
  sorry

end domain_of_f_l106_106786


namespace number_of_strikers_correct_l106_106395

-- Defining the initial conditions
def number_of_goalies := 3
def number_of_defenders := 10
def number_of_players := 40
def number_of_midfielders := 2 * number_of_defenders

-- Lean statement to prove
theorem number_of_strikers_correct : 
  let total_non_strikers := number_of_goalies + number_of_defenders + number_of_midfielders,
      number_of_strikers := number_of_players - total_non_strikers 
  in number_of_strikers = 7 :=
by
  sorry

end number_of_strikers_correct_l106_106395


namespace trajectory_of_moving_circle_l106_106929

noncomputable def ellipse_trajectory_eq (x y : ℝ) : Prop :=
  (x^2)/25 + (y^2)/9 = 1

theorem trajectory_of_moving_circle
  (x y : ℝ)
  (A : ℝ × ℝ)
  (C : ℝ × ℝ)
  (radius_C : ℝ)
  (hC : (x + 4)^2 + y^2 = 100)
  (hA : A = (4, 0))
  (radius_C_eq : radius_C = 10) :
  ellipse_trajectory_eq x y :=
sorry

end trajectory_of_moving_circle_l106_106929


namespace distance_between_CK_and_A1D_is_one_third_l106_106564

/-
Given a cube ABCD A1B1C1D1 with edge length 1 and K being the midpoint of 
edge DD1, prove that the distance between the lines CK and A1D is 1/3.
-/

open EuclideanGeometry    -- Assume we have a module that handles Euclidean geometry

structure Cube :=
  (A B C D A1 B1 C1 D1 : Point)
  (edge_length : ℝ)
  (cube_conditions : 
    A.dist B = edge_length ∧ 
    B.dist C = edge_length ∧ 
    C.dist D = edge_length ∧ 
    D.dist A = edge_length ∧ 
    A1.dist B1 = edge_length ∧ 
    B1.dist C1 = edge_length ∧ 
    C1.dist D1 = edge_length ∧ 
    D1.dist A1 = edge_length ∧ 
    A.dist A1 = edge_length ∧ 
    B.dist B1 = edge_length ∧ 
    C.dist C1 = edge_length ∧ 
    D.dist D1 = edge_length)

def midpoint (P Q : Point) : Point := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩

theorem distance_between_CK_and_A1D_is_one_third (cube : Cube) (K : Point)
  (hK : K = midpoint cube.D cube.D1) : 
  distance_between_lines (line_through cube.C K) (line_through cube.A1 cube.D) = 1 / 3 :=
by
  sorry

end distance_between_CK_and_A1D_is_one_third_l106_106564


namespace percentage_of_500_l106_106360

theorem percentage_of_500 (P : ℝ) : 0.1 * (500 * P / 100) = 25 → P = 50 :=
by
  sorry

end percentage_of_500_l106_106360


namespace rational_solutions_of_quadratic_l106_106554

theorem rational_solutions_of_quadratic (k : ℕ) (hk : 0 < k ∧ k ≤ 10) :
  ∃ (x : ℚ), k * x^2 + 20 * x + k = 0 ↔ (k = 6 ∨ k = 8 ∨ k = 10) :=
by sorry

end rational_solutions_of_quadratic_l106_106554


namespace tan_alpha_beta_eq_two_range_norm_b_add_c_l106_106973

variable (α β : ℝ)
def a : ℝ × ℝ := (4 * Real.cos α, Real.sin α)
def b : ℝ × ℝ := (Real.sin β, 4 * Real.cos β)
def c : ℝ × ℝ := (Real.cos β, -4 * Real.sin β)

-- Prove tan(α + β) = 2 given that a is perpendicular to (b - 2c)
theorem tan_alpha_beta_eq_two
  (h : ∀ (a b c : ℝ × ℝ), a.1 * (b.1 - 2 * c.1) + a.2 * (b.2 - 2 * c.2) = 0) :
  Real.tan (α + β) = 2 := sorry

-- Prove the range of |b + c| is [0, 4 * sqrt 2]
theorem range_norm_b_add_c :
  ∃ y, y ∈ Set.Icc 0 (4 * Real.sqrt 2) ∧
  ∀ (b c : ℝ × ℝ), (y^2 = (b.1 + c.1)^2 + (b.2 + c.2)^2) := sorry

end tan_alpha_beta_eq_two_range_norm_b_add_c_l106_106973


namespace variance_not_2_minimum_standard_deviation_l106_106026

-- Definition for a right triangle with hypotenuse 3
structure RightTriangle where
  (a : ℝ) (b : ℝ)
  hypotenuse : ℝ := 3
  pythagorean_property : a^2 + b^2 = 9

-- Part (a) - Prove that the variance cannot be 2
theorem variance_not_2 (triangle : RightTriangle) : 
  (6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) ≠ 2 := sorry

-- Part (b) - Prove the minimum standard deviation and corresponding leg lengths
theorem minimum_standard_deviation (triangle : RightTriangle) : 
  (exists (a b : ℝ), a > 0 ∧ b > 0 ∧ a = b ∧ a = b = 3 * (real.sqrt 2) / 2 ∧ 
  real.sqrt ((6 - ( (triangle.a + triangle.b + 3) / 3 ) ^ 2) = real.sqrt (2) - 1)) := sorry

end variance_not_2_minimum_standard_deviation_l106_106026


namespace frequency_group_16_5_to_18_5_l106_106609

def sample_set : Set ℝ := {15, 11, 13, 15, 17, 19, 15, 18, 20, 19, 16, 14, 15, 17, 16, 12, 14, 15, 16, 18}
def group_range := (16.5, 18.5)
def interval := 2

theorem frequency_group_16_5_to_18_5 :
  let frequency := {x ∈ sample_set | group_range.1 < x ∧ x <= group_range.2}.to_finset.card
  let proportion := (frequency : ℝ) / sample_set.to_finset.card
  proportion = 0.2 :=
by {
  sorry
}

end frequency_group_16_5_to_18_5_l106_106609


namespace Todd_ate_5_cupcakes_l106_106855

theorem Todd_ate_5_cupcakes (original_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) (remaining_cupcakes : ℕ) :
  original_cupcakes = 50 ∧ packages = 9 ∧ cupcakes_per_package = 5 ∧ remaining_cupcakes = packages * cupcakes_per_package →
  original_cupcakes - remaining_cupcakes = 5 :=
by
  sorry

end Todd_ate_5_cupcakes_l106_106855


namespace equal_water_levels_after_one_hour_l106_106868

theorem equal_water_levels_after_one_hour
  (H_X H_Y : ℝ)
  (V_X V_Y R_X R_Y : ℝ)
  (h1 : H_Y = 1.5 * H_X)
  (h2 : R_X * 2 = V_X)
  (h3 : R_Y * 1.5 = V_Y)
  (h4 : V_Y = 1.5 * V_X) :
  ∃ t : ℝ, t = 1 ∧ (H_X - t * (H_X / 2) = 1.5 * H_X - t * H_X) := by
  use 1
  split
  . refl
  . sorry

end equal_water_levels_after_one_hour_l106_106868


namespace greatest_possible_price_per_notebook_l106_106079

theorem greatest_possible_price_per_notebook (budget entrance_fee : ℝ) (notebooks : ℕ) (tax_rate : ℝ) (price_per_notebook : ℝ) :
  budget = 160 ∧ entrance_fee = 5 ∧ notebooks = 18 ∧ tax_rate = 0.05 ∧ price_per_notebook * notebooks * (1 + tax_rate) ≤ (budget - entrance_fee) →
  price_per_notebook = 8 :=
by
  sorry

end greatest_possible_price_per_notebook_l106_106079


namespace difference_of_squares_consecutive_l106_106266

theorem difference_of_squares_consecutive (n : ℕ) (h_pos : 0 < n) (h_sum : 2 * n + 1 < 100) : 
  ∃ k ∈ {2, 64, 79, 96, 131}, (n + 1)^2 - n^2 = k := by
  use 79
  sorry

end difference_of_squares_consecutive_l106_106266


namespace product_of_midpoint_coordinates_l106_106335

-- Define the endpoints of the line segment
def point_A : ℝ × ℝ := (4, -1)
def point_B : ℝ × ℝ := (-2, 7)

-- Define the function to calculate the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Calculate the midpoint of the given endpoints
def midpoint_AB : ℝ × ℝ := midpoint point_A point_B

-- Prove that the product of the coordinates of the midpoint is 3
theorem product_of_midpoint_coordinates :
  (midpoint_AB.1 * midpoint_AB.2) = 3 :=
by
  sorry

end product_of_midpoint_coordinates_l106_106335


namespace parabola_equation_l106_106309

noncomputable def vertex : (ℝ × ℝ) := (0, 0)

noncomputable def directrix_eq (x : ℝ) : Prop := x = 3

theorem parabola_equation : ∀ (p : ℝ), p = 6 → vertex = (0, 0) → directrix_eq 3 → (y : ℝ) * y = -12 * (x : ℝ) := by
  intro p hp hv hd
  rw [hp]; 
  sorry

end parabola_equation_l106_106309


namespace vector_calculation_l106_106916

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (1, -1)
def vec_result : ℝ × ℝ := (3 * vec_a.fst - 2 * vec_b.fst, 3 * vec_a.snd - 2 * vec_b.snd)
def target_vec : ℝ × ℝ := (1, 5)

theorem vector_calculation :
  vec_result = target_vec :=
sorry

end vector_calculation_l106_106916


namespace largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l106_106838

-- Define the sequence and its cyclic property
def cyclicSequence (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq (n + 4) = 1000 * (seq n % 10) + 100 * (seq (n + 1) % 10) + 10 * (seq (n + 2) % 10) + (seq (n + 3) % 10)

-- Define the property of T being the sum of the sequence
def sumOfSequence (seq : ℕ → ℕ) (T : ℕ) : Prop :=
  T = seq 0 + seq 1 + seq 2 + seq 3

-- Define the statement that T is always divisible by 101
theorem largest_prime_divisor_of_sum_of_cyclic_sequence_is_101
  (seq : ℕ → ℕ) (T : ℕ)
  (h1 : cyclicSequence seq)
  (h2 : sumOfSequence seq T) :
  (101 ∣ T) := 
sorry

end largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l106_106838


namespace lcm_of_fractions_l106_106421

-- Definitions based on the problem's conditions
def numerators : List ℕ := [7, 8, 3, 5, 13, 15, 22, 27]
def denominators : List ℕ := [10, 9, 8, 12, 14, 100, 45, 35]

-- LCM and GCD functions for lists of natural numbers
def list_lcm (l : List ℕ) : ℕ := l.foldr lcm 1
def list_gcd (l : List ℕ) : ℕ := l.foldr gcd 0

-- Main proposition
theorem lcm_of_fractions : list_lcm numerators / list_gcd denominators = 13860 :=
by {
  -- to be proven
  sorry
}

end lcm_of_fractions_l106_106421


namespace min_men_in_group_l106_106776

variable (M T R A MTR ∩ A : ℕ)

theorem min_men_in_group : 
  (M = 81) → (T = 75) → (R = 85) → (A = 70) → (M ∩ T ∩ R ∩ A = 11) → 85 ≤ M ∪ T ∪ R ∪ A := 
by
  intros hM hT hR hA hMTRA
  sorry

end min_men_in_group_l106_106776


namespace domain_of_composed_function_l106_106635

theorem domain_of_composed_function
  (f : ℝ → ℝ)
  (dom_f : ∀ x, 0 ≤ x ∧ x ≤ 4 → f x ≠ 0) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → f (x^2) ≠ 0 :=
by
  sorry

end domain_of_composed_function_l106_106635


namespace garden_division_proof_l106_106004

structure Garden :=
  (area : ℝ)
  (trees : ℕ)

def equalParts (garden : Garden) : Prop :=
  ∃ parts : Fin 4 → (ℝ × ℕ), 
    (∀ i, parts i.1 = garden.area / 4) ∧ 
    (∀ i, parts i.2 = garden.trees / 4) ∧ 
    (∀ i j, parts i = parts j)

def distinctWays (garden : Garden) : ℕ := 
  24

theorem garden_division_proof (G : Garden) (h : equalParts G) :
  distinctWays G = 24 :=
sorry

end garden_division_proof_l106_106004


namespace projection_of_a_onto_b_l106_106120

open Real

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (h : a ⋅ (a + b) = 0)

theorem projection_of_a_onto_b :
  (a ⋅ b) / ‖b‖ = -1 / 2 :=
sorry

end projection_of_a_onto_b_l106_106120


namespace limit_of_f_at_1_l106_106156

def f (x : ℝ) : ℝ := 3 * x + 1

theorem limit_of_f_at_1 : (limit (λ Δx, ((f (1 - Δx) - f 1) / Δx)) 0 = -3) :=
by
  sorry

end limit_of_f_at_1_l106_106156


namespace instantaneous_velocity_at_3_l106_106299

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- State the main theorem that we need to prove
theorem instantaneous_velocity_at_3 : (deriv displacement 3 = 5) := by
  sorry

end instantaneous_velocity_at_3_l106_106299


namespace anna_has_9_cupcakes_left_l106_106859

def cupcakes_left (initial : ℕ) (given_away_fraction : ℚ) (eaten : ℕ) : ℕ :=
  let remaining = initial * (1 - given_away_fraction)
  remaining - eaten

theorem anna_has_9_cupcakes_left :
  cupcakes_left 60 (4/5 : ℚ) 3 = 9 := by
  sorry

end anna_has_9_cupcakes_left_l106_106859


namespace area_of_smaller_circle_l106_106326

noncomputable def radius_smaller := sqrt (12 / 5)
noncomputable def area_smaller := π * (radius_smaller ^ 2)

theorem area_of_smaller_circle 
  (tangent_condition : PA = AB = 6) 
  (radius_relation : rad_larger = 3 * rad_smaller):
  area_smaller = (12 * π) / 5 := 
sorry

end area_of_smaller_circle_l106_106326


namespace craig_apples_total_l106_106884

-- Conditions
def initial_apples := 20.0
def additional_apples := 7.0

-- Question turned into a proof problem
theorem craig_apples_total : initial_apples + additional_apples = 27.0 :=
by
  sorry

end craig_apples_total_l106_106884


namespace smallest_circle_radius_l106_106795

noncomputable def smallest_radius (n : ℕ) : ℝ :=
  if n = 2 then 1 else
  if n = 3 then 1 else
  if n = 4 then 1 else
  if n = 5 then 1 else
  if n = 6 then 1 else
  if n = 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  if n = 11 then 1.61 else
  0  -- default value for other n, not part of the problem range

theorem smallest_circle_radius :
  ∀ n, n ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10, 11} : Set ℕ) →
  (∀ (i j : ℕ) (h_i : i < n) (h_j : j < n) (h_ne : i ≠ j),
    dist (points n i) (points n j) ≥ 1) →
  (∃ R, R = smallest_radius n) :=
by
  intro n h_n h_d
  exists smallest_radius n
  sorry

end smallest_circle_radius_l106_106795


namespace cole_drive_time_l106_106068

noncomputable def T_work (D : ℝ) : ℝ := D / 75
noncomputable def T_home (D : ℝ) : ℝ := D / 105

theorem cole_drive_time (v1 v2 T : ℝ) (D : ℝ) 
  (h_v1 : v1 = 75) (h_v2 : v2 = 105) (h_T : T = 4)
  (h_round_trip : T_work D + T_home D = T) : 
  T_work D = 140 / 60 :=
sorry

end cole_drive_time_l106_106068


namespace p_value_at_5_l106_106378

noncomputable def p (x : ℕ) : ℚ := sorry -- the definition of p will be skipped

theorem p_value_at_5 (h : ∀ n ∈ {1, 2, 3, 4}, p n = 2 / (n^3)) : 
  p 5 = 139 / 1500 := 
sorry

end p_value_at_5_l106_106378


namespace solve_inequality_for_a_l106_106889

theorem solve_inequality_for_a (a : ℝ) :
  (∀ x : ℝ, abs (x^2 + 3 * a * x + 4 * a) ≤ 3 → x = -3 * a / 2)
  ↔ (a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13) :=
by 
  sorry

end solve_inequality_for_a_l106_106889


namespace sum_of_roots_quadratic_eq_l106_106746

theorem sum_of_roots_quadratic_eq : ∀ P Q : ℝ, (3 * P^2 - 9 * P + 6 = 0) ∧ (3 * Q^2 - 9 * Q + 6 = 0) → P + Q = 3 :=
by
  sorry

end sum_of_roots_quadratic_eq_l106_106746


namespace initial_soccer_balls_l106_106706

theorem initial_soccer_balls (x : ℝ) (h1 : 0.40 * x = y) (h2 : 0.20 * (0.60 * x) = z) (h3 : 0.80 * (0.60 * x) = 48) : x = 100 := by
  sorry

end initial_soccer_balls_l106_106706


namespace cone_height_is_48_l106_106375

noncomputable def cone_height (V : ℝ) (vertex_angle : ℝ) : ℝ :=
  let r := ((3 * V) / (vertex_angle * Real.sqrt 3))^(1/3)
  in r * Real.sqrt 3

theorem cone_height_is_48 :
  cone_height (12288 * Real.pi) (Real.pi / 3) = 48.0 :=
sorry

end cone_height_is_48_l106_106375


namespace tangents_when_k_zero_exists_point_P_on_y_axis_l106_106647

-- Definition for the curve
def curve (x y : ℝ) : Prop := x^2 = 4 * y

-- Definition for the line with k=0
def line (y a : ℝ) : Prop := y = a

-- Conditions: a > 0
variable (a : ℝ) (ha : a > 0)

-- Problem (1)
theorem tangents_when_k_zero :
  let M := (2 * real.sqrt a, a)
  let N := (-2 * real.sqrt a, a)
  (curve M.1 M.2) ∧ (curve N.1 N.2) ∧ (line M.2 a) ∧ (line N.2 a) →
  ( ∃ (tangentM : ∀ x y : ℝ, ℝ), tangentM (2 * real.sqrt a) a = real.sqrt a * x - y - a ) ∧
  ( ∃ (tangentN : ∀ x y : ℝ, ℝ), tangentN (-2 * real.sqrt a) a = real.sqrt a * x + y + a ) :=
by sorry

-- Problem (2)
theorem exists_point_P_on_y_axis :
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = -a ∧
  ( ∀ k : ℝ,
    let M := (2 * real.sqrt a, a)
    let N := (-2 * real.sqrt a, a)
    let angles_are_equal : Prop := ∠(P.front, M) = ∠(P.front, N)
    angles_are_equal ) :=
by sorry

end tangents_when_k_zero_exists_point_P_on_y_axis_l106_106647


namespace integer_solutions_of_inequality_count_l106_106625

theorem integer_solutions_of_inequality_count :
  let a := -2 - Real.sqrt 6
  let b := -2 + Real.sqrt 6
  ∃ n, n = 5 ∧ ∀ x : ℤ, x < a ∨ b < x ↔ (4 * x^2 + 16 * x + 15 ≤ 23) → n = 5 :=
by sorry

end integer_solutions_of_inequality_count_l106_106625


namespace sin_values_l106_106629

theorem sin_values (B : ℝ) (h : 3 * (Real.tan B) - (Real.sec B) = 1) : 
  ∃ x ∈ {0, 3 / 5}, Real.sin B = x :=
by {
  sorry
}

end sin_values_l106_106629


namespace cindy_correct_answer_l106_106873

-- Define the conditions given in the problem
def x : ℤ := 272 -- Cindy's miscalculated number

-- The outcome of Cindy's incorrect operation
def cindy_incorrect (x : ℤ) : Prop := (x - 7) = 53 * 5

-- The outcome of Cindy's correct operation
def cindy_correct (x : ℤ) : ℤ := (x - 5) / 7

-- The main theorem to prove
theorem cindy_correct_answer : cindy_incorrect x → cindy_correct x = 38 :=
by
  sorry

end cindy_correct_answer_l106_106873


namespace range_of_magnitude_l106_106177

open Real

def a (θ : ℝ) : ℝ × ℝ := (1, sin θ)
def b (θ : ℝ) : ℝ × ℝ := (cos θ, sqrt 3)

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

def vec_diff (θ : ℝ) : ℝ × ℝ := (a θ).1 - (b θ).1, (a θ).2 - (b θ).2

def mag_diff (θ : ℝ) : ℝ := magnitude (vec_diff θ)

theorem range_of_magnitude (θ : ℝ) : 1 ≤ mag_diff θ ∧ mag_diff θ ≤ 3 :=
by
  sorry

end range_of_magnitude_l106_106177


namespace relationship_among_a_b_c_l106_106917
noncomputable theory

def a := Real.sqrt 0.5
def b := Real.sqrt 0.3
def c := Real.logb 0.3 0.2

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l106_106917


namespace location_in_quadrant_l106_106683

noncomputable def z : ℂ := (2 - complex.i) / (1 + complex.i)

theorem location_in_quadrant :
  0 < z.re ∧ z.im < 0 :=
by
  sorry

end location_in_quadrant_l106_106683


namespace overweight_condition_equiv_determine_coefficients_l106_106830

noncomputable def ideal_weight (h : ℝ) : ℝ := 22 * h^2
noncomputable def overweight_condition (W h : ℝ) : Prop := W > 24.2 * h^2

theorem overweight_condition_equiv (W h c d e : ℝ) :
  overweight_condition W h ↔ W > c * h^2 + d * h + e :=
begin
  sorry
end

theorem determine_coefficients (W h : ℝ) :
  (∃ c d e, overweight_condition W h ↔ W > c * h^2 + d * h + e) → (c, d, e) = (24.2, 0, 0) :=
begin
  intro hW,
  cases hW with c heq,
  cases heq with d heq_de,
  cases heq_de with e heq_condition,
  have : c = 24.2,
  have : d = 0,
  have : e = 0,
  simp [this],
  sorry
end

end overweight_condition_equiv_determine_coefficients_l106_106830


namespace bread_needed_for_sandwiches_l106_106848

def students_per_group := 5
def groups := 5
def sandwiches_per_student := 2
def pieces_of_bread_per_sandwich := 2

theorem bread_needed_for_sandwiches : 
  students_per_group * groups * sandwiches_per_student * pieces_of_bread_per_sandwich = 100 := 
by
  sorry

end bread_needed_for_sandwiches_l106_106848


namespace binom_18_10_l106_106442

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l106_106442


namespace num_values_of_k_l106_106194

theorem num_values_of_k (k : ℤ) (h1 : 121 < k^2) (h2 : k^2 < 225) : set.card {k : ℤ | 121 < k^2 ∧ k^2 < 225} = 3 :=
sorry

end num_values_of_k_l106_106194


namespace binom_18_10_eq_43758_l106_106462

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l106_106462


namespace ellipse_equation_range_of_sums_l106_106131

-- Define the ellipse parameters a, b, c, and e
variables {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hac : a > b > 0) (he : e = 1 / 2)

-- The given conditions
def ellipse (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def eccentricity := c / a = e
def max_area := (√(a^2 - c^2)) * c = 4 * √3

-- Part 1: Prove the equation of the ellipse
theorem ellipse_equation : ellipse (16:ℝ) (12:ℝ) :=
sorry

-- Part 2: Range of sum of magnitudes of vectors AC and BD
variables (A B C D : ℝ × ℝ)
def perpendicular_vectors := (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0

theorem range_of_sums (h_perp : perpendicular_vectors A B C D) (h_intersect : A = (-2, 0)) :
(|A - C| + |B - D|) ∈ set.Ico (96 / 7) 14 :=
sorry

end ellipse_equation_range_of_sums_l106_106131


namespace remainder_when_dividing_polynomial_l106_106548

noncomputable def P(x : ℝ) := x^5 + 3
noncomputable def Q(x : ℝ) := (x - 3)^2

theorem remainder_when_dividing_polynomial :
  ∃ (R : ℝ → ℝ), (λ x, R x) = (λ x, 405 * x - 969) ∧ (∃ (S : ℝ → ℝ), P = λ x, Q(x) * S(x) + R(x)) :=
sorry

end remainder_when_dividing_polynomial_l106_106548


namespace min_phi_for_symmetry_l106_106751

theorem min_phi_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧ 
  (∀ (x : ℝ), 
    let g := λ x, sin (5 * pi / 12 - (x + φ))^2 - sin (pi / 12 + (x + φ))^2 in
    g (pi / 6 - x) = g (pi / 6 + x) ∧ φ = pi / 4) :=
begin
  sorry

end min_phi_for_symmetry_l106_106751


namespace domain_g_eq_l106_106147

noncomputable def domain_f : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

noncomputable def g_domain := {x | f x ∈ domain_f ∧ f (-x) ∈ domain_f}

theorem domain_g_eq : g_domain = {x | -2 ≤ x ∧ x ≤ 2} := 
by
  sorry

end domain_g_eq_l106_106147


namespace statement_c_incorrect_l106_106825

def heights : List ℝ := [10, 20, 30, 40, 50, 60, 70]
def times : List ℝ := [4.23, 3.00, 2.45, 2.13, 1.89, 1.71, 1.59]

theorem statement_c_incorrect : ¬(∀ (i : Fin 6), (times.get i - times.get (i + 1)) = 1.23) :=
by
  sorry

end statement_c_incorrect_l106_106825


namespace find_thirteenth_result_l106_106645

theorem find_thirteenth_result
  (seq : Fin 25 → ℤ)
  (h_avg_25 : (∑ i, seq i) / 25 = 18)
  (h_avg_12_first : (∑ i in Finset.range 12, seq i) / 12 = 10)
  (h_avg_12_last : (∑ i in Finset.range 12, seq ⟨i + 13, by linarith⟩) / 12 = 20) :
  seq 12 = 90 := 
by 
  -- Proof omitted
  sorry

end find_thirteenth_result_l106_106645


namespace max_attempts_to_open_doors_l106_106062

theorem max_attempts_to_open_doors
  (n : Nat) (h_n : n = 17) :
  ∑ i in Finset.range n, i = 136 := by
  sorry

end max_attempts_to_open_doors_l106_106062


namespace binom_18_10_l106_106429

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106429


namespace boat_upstream_time_l106_106362

theorem boat_upstream_time (stream_speed boat_speed downstream_time distance upstream_time : ℝ) 
  (h1 : stream_speed = 3) 
  (h2 : boat_speed = 15) 
  (h3 : downstream_time = 1) 
  (h4 : distance = (boat_speed + stream_speed) * downstream_time) 
  (h5 : upstream_time = distance / (boat_speed - stream_speed)) : 
  upstream_time = 1.5 :=
by
  rw [h1, h2, h3] at h4
  rw [h1, h2, h4] at h5
  simp at h5
  exact h5

#check boat_upstream_time

end boat_upstream_time_l106_106362


namespace smallest_positive_m_integral_solutions_l106_106337

theorem smallest_positive_m_integral_solutions (m : ℕ) :
  (∃ (x y : ℤ), 10 * x * x - m * x + 660 = 0 ∧ 10 * y * y - m * y + 660 = 0 ∧ x ≠ y)
  → m = 170 := sorry

end smallest_positive_m_integral_solutions_l106_106337


namespace retailer_initial_profit_thought_l106_106410

-- Define cost price and marked price
def cost_price : ℝ := 100
def marked_price : ℝ := cost_price + 0.4 * cost_price
-- Define selling price after discount
def selling_price : ℝ := marked_price - 0.25 * marked_price
-- Define actual profit percentage after discount
def actual_profit_percentage : ℝ := (selling_price - cost_price) / cost_price * 100

-- Define initial expected profit percentage
def expected_selling_price : ℝ := marked_price
def initial_expected_profit : ℝ := expected_selling_price - cost_price
def initial_expected_profit_percentage : ℝ := initial_expected_profit / cost_price * 100

-- Theorem to prove the retailer initially thought he would make a 40% profit
theorem retailer_initial_profit_thought :
  initial_expected_profit_percentage = 40 := by
  sorry

end retailer_initial_profit_thought_l106_106410


namespace variance_incorrect_min_standard_deviation_l106_106030

open Real

-- Define a right triangle with hypotenuse of length 3
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ c = 3

-- Prove that the variance of the side lengths cannot be 2
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b 3) : 
  ¬(let x := [a, b, 3] in
    let mean_square := (x.map (λ x, x^2)).sum / 3 in
    let mean := x.sum / 3 in
    mean_square - mean^2 = 2) :=
sorry

-- Prove the minimum standard deviation and corresponding lengths
theorem min_standard_deviation {a b : ℝ} (h : right_triangle a b 3) :
  a = b → a = b → a = 3 * real.sqrt(2) / 2 → b = 3 * real.sqrt(2) / 2 →
  (let variance := (h.1.map (λ x, x^2)).sum / 2 - ((h.1.sum / 2)^2) in
  let std_dev_min := real.sqrt(variance) in
  std_dev_min = real.sqrt(2) - 1) :=
sorry

end variance_incorrect_min_standard_deviation_l106_106030


namespace minimum_value_of_f_on_interval_l106_106530

noncomputable def f (x : ℝ) : ℝ := (Real.tan x)^2 + 2 * Real.tan x + 6 / (Real.tan x) + 9 / (Real.tan x)^2 + 4

theorem minimum_value_of_f_on_interval :
  ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x = 10 + 4 * Real.sqrt 3 :=
begin
  sorry,
end

end minimum_value_of_f_on_interval_l106_106530


namespace plane_eq_through_point_line_l106_106097

-- Definitions of the plane problem
def point : ℝ × ℝ × ℝ := (1, -3, 6)

def line (x y z : ℝ) : Prop := (x - 2) / 4 = (y + 1) / -1 ∧ (y + 1) / -1 = (z - 3) / 2

-- Problem: Prove the plane equation given the conditions
theorem plane_eq_through_point_line (A B C D : ℤ) (hA : A = 1) (hB : B = -18) (hC : C = -7) (hD : D = -13) :
  (gcd A B C D = 1) → 
  (∀ (x y z : ℝ), line x y z → A * x + B * y + C * z + D = 0) ∧ -- The plane contains the line
  (A * point.1 + B * point.2 + C * point.3 + D = 0) →          -- The plane passes through the point
  A * point.1 + B * point.2 + C * point.3 + D = 0 := 
sorry

end plane_eq_through_point_line_l106_106097


namespace binom_18_10_l106_106432

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l106_106432


namespace coeff_m5n3_in_expansion_of_m_plus_n_pow8_l106_106523

theorem coeff_m5n3_in_expansion_of_m_plus_n_pow8 :
  let m := (m : ℕ)
  let n := (n : ℕ)
  ∀ (c : ℕ), c = Nat.choose 8 5 → c = 56 :=
by
  intros
  unfold Nat.choose
  sorry

end coeff_m5n3_in_expansion_of_m_plus_n_pow8_l106_106523


namespace vertical_angles_eq_l106_106343

theorem vertical_angles_eq (A B : Type) (are_vertical : A = B) :
  A = B := 
by
  exact are_vertical

end vertical_angles_eq_l106_106343


namespace radius_of_circle_from_spherical_l106_106771

noncomputable def spherical_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem radius_of_circle_from_spherical :
  ∀ φ : ℝ, let (x, y, _) := spherical_to_cartesian 2 0 φ in Real.sqrt (x^2 + y^2) = 2 := by
sorry

end radius_of_circle_from_spherical_l106_106771


namespace correct_remainder_l106_106056

-- Define the problem
def count_valid_tilings (n k : Nat) : Nat :=
  Nat.factorial (n + k) / (Nat.factorial n * Nat.factorial k) * (3 ^ (n + k) - 3 * 2 ^ (n + k) + 3)

noncomputable def tiles_mod_1000 : Nat :=
  let pairs := [(8, 0), (6, 1), (4, 2), (2, 3), (0, 4)]
  let M := pairs.foldl (λ acc (nk : Nat × Nat) => acc + count_valid_tilings nk.1 nk.2) 0
  M % 1000

theorem correct_remainder : tiles_mod_1000 = 328 :=
  by sorry

end correct_remainder_l106_106056


namespace variance_incorrect_min_std_deviation_l106_106036

-- Definitions for the given conditions.
variable (a b : ℝ)

-- The right triangle condition given by Pythagorean theorem.
def right_triangle (a b : ℝ) : Prop :=
  a^2 + b^2 = 9

-- Problems to verify
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b) : 
  ¬(variance {a, b, 3} = 2) := sorry

theorem min_std_deviation {a b : ℝ} (h : right_triangle a b) :
  let s := sqrt(2) - 1,
  (a = b) → (a = 3 * sqrt(2) / 2) → (std_deviation {a, b, 3} = s) := sorry

end variance_incorrect_min_std_deviation_l106_106036


namespace moles_of_nanao_l106_106531

def balanced_equation (hno3 nahco3 nanao co2 h2o : ℕ) : Prop :=
  hno3 = nahco3 ∧ nanao = nahco3 ∧ co2 = nahco3 ∧ h2o = nahco3

theorem moles_of_nanao (HNO3 NaHCO3 : ℕ)
  (reaction : balanced_equation HNO3 NaHCO3 1 1 1) :
  HNO3 = 1 ∧ NaHCO3 = 1 → nanao = 1 := 
begin
  intros h,
  cases h with h₁ h₂,
  rw [h₁, h₂],
  exact reaction.2.1,
end

end moles_of_nanao_l106_106531


namespace g_at_neg1_l106_106690

def g (x : ℝ) : ℝ :=
if x < 2 then 4 * x - 6 else 10 - 3 * x

theorem g_at_neg1 : g (-1) = -10 := by
  sorry

end g_at_neg1_l106_106690


namespace jogger_ahead_of_train_l106_106381

noncomputable def distance_ahead_of_train (v_j v_t : ℕ) (L_t t : ℕ) : ℕ :=
  let relative_speed_kmh := v_t - v_j
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  let total_distance := relative_speed_ms * t
  total_distance - L_t

theorem jogger_ahead_of_train :
  distance_ahead_of_train 10 46 120 46 = 340 :=
by
  sorry

end jogger_ahead_of_train_l106_106381


namespace sum_of_smallest_8_values_l106_106908

-- Define the function T_n
def T (n : ℕ) : ℕ := (n - 1) * n * (n + 1) * (3 * n + 2) / 24

-- Define a predicate for T_n being divisible by 5
def divisible_by_5 (n : ℕ) : Prop := T n % 5 = 0

-- Find the 8 smallest values of n such that T_n is divisible by 5
def smallest_8_values (n : ℕ) : List ℕ :=
  list.take 8 (list.filter (λ x, x ≥ 2 ∧ divisible_by_5 x) (list.range (n + 50)))

-- Define the summation of those values
def sum_smallest_8_values : ℕ := (smallest_8_values 50).sum

-- The main proof statement
theorem sum_of_smallest_8_values : sum_smallest_8_values = 148 :=
by
  sorry

end sum_of_smallest_8_values_l106_106908


namespace pages_in_book_l106_106183

theorem pages_in_book
  (x : ℝ)
  (h1 : x - (x / 6 + 10) = (5 * x) / 6 - 10)
  (h2 : (5 * x) / 6 - 10 - ((1 / 5) * ((5 * x) / 6 - 10) + 20) = (2 * x) / 3 - 28)
  (h3 : (2 * x) / 3 - 28 - ((1 / 4) * ((2 * x) / 3 - 28) + 25) = x / 2 - 46)
  (h4 : x / 2 - 46 = 72) :
  x = 236 := 
sorry

end pages_in_book_l106_106183


namespace num_integers_between_200_and_300_with_conditions_l106_106977

-- Define conditions for the problem
def is_three_digit_integer (n : ℕ) : Prop :=
  200 ≤ n ∧ n < 300

def has_three_different_digits_in_increasing_order (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 3 ∧
  digits.nth 0 < digits.nth 1 ∧
  digits.nth 1 < digits.nth 2 ∧
  digits.nth 0 ≠ digits.nth 1 ∧
  digits.nth 1 ≠ digits.nth 2 ∧
  digits.nth 0 ≠ digits.nth 2

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem num_integers_between_200_and_300_with_conditions :
  { n : ℕ // is_three_digit_integer n ∧ has_three_different_digits_in_increasing_order n ∧ is_divisible_by_5 n }.card = 2 :=
sorry

end num_integers_between_200_and_300_with_conditions_l106_106977


namespace range_of_a_inequality_mn_l106_106960

noncomputable def f (x a : ℝ) : ℝ := Real.sqrt (abs (2 * x - 1) + abs (x + 1) - a)

theorem range_of_a :
  (∀ x, f x a ∈ ℝ) →
  a ≤ (3 / 2) := sorry

theorem inequality_mn (m n k : ℝ) (hm : m > 0) (hn : n > 0) :
  k = 3 / 2 →
  m + n = 2 * k →
  (1 / m) + (4 / n) ≥ 3 := sorry

end range_of_a_inequality_mn_l106_106960


namespace variance_incorrect_min_std_deviation_l106_106035

-- Definitions for the given conditions.
variable (a b : ℝ)

-- The right triangle condition given by Pythagorean theorem.
def right_triangle (a b : ℝ) : Prop :=
  a^2 + b^2 = 9

-- Problems to verify
theorem variance_incorrect {a b : ℝ} (h : right_triangle a b) : 
  ¬(variance {a, b, 3} = 2) := sorry

theorem min_std_deviation {a b : ℝ} (h : right_triangle a b) :
  let s := sqrt(2) - 1,
  (a = b) → (a = 3 * sqrt(2) / 2) → (std_deviation {a, b, 3} = s) := sorry

end variance_incorrect_min_std_deviation_l106_106035


namespace domain_of_f_l106_106168

noncomputable def f (x : ℝ) : ℝ := log (4^x - 2^(x+1) + 1) / log (1 / 2)

theorem domain_of_f : ∀ x : ℝ, f x ∈ [0, +∞) → (0 < x ∧ x ≤ 1) :=
by
  intro x hx
  sorry

end domain_of_f_l106_106168


namespace CentralBankInterest_BankBenefits_RegistrationNecessity_l106_106719

-- Define the conditions
variable (CentralBankUsesLoyaltyProgram : Prop)
variable (BanksHaveClientsParticipating : Prop)
variable (LoyaltyProgramOffersBonusesRequiresRegistration : Prop)

-- Define the goals to prove
theorem CentralBankInterest : 
  CentralBankUsesLoyaltyProgram → 
  BanksHaveClientsParticipating → 
  LoyaltyProgramOffersBonusesRequiresRegistration → 
  (PromotionOfNationalPaymentSystem ∧ EconomicStimulus) := 
  by 
  intros h1 h2 h3 
  split 
  sorry

theorem BankBenefits : 
  CentralBankUsesLoyaltyProgram → 
  BanksHaveClientsParticipating → 
  LoyaltyProgramOffersBonusesRequiresRegistration → 
  (IncreasedCustomerLoyalty ∧ HigherTransactionVolumes) := 
  by 
  intros h1 h2 h3 
  split 
  sorry

theorem RegistrationNecessity : 
  CentralBankUsesLoyaltyProgram → 
  BanksHaveClientsParticipating → 
  LoyaltyProgramOffersBonusesRequiresRegistration → 
  (DataCollectionAndMarketing ∧ FraudPreventionAndSecurity) := 
  by 
  intros h1 h2 h3 
  split 
  sorry

end CentralBankInterest_BankBenefits_RegistrationNecessity_l106_106719


namespace series_plus_fraction_equals_one_l106_106783

-- Define the series sum function
def seriesSum : ℝ :=
  ∑ n in Finset.range (24).filter (λ x, x ≥ 2), 1 / (n^(3/2) - (n+1)^(3/2))

-- Define the fraction to be added 
def fractionToAdd : ℝ :=
  1 - (1 / 2^(3/2) - 1 / 24^(3/2))

-- State the theorem
theorem series_plus_fraction_equals_one : seriesSum + fractionToAdd = 1 :=
by 
  sorry

end series_plus_fraction_equals_one_l106_106783


namespace general_term_formula_sum_b_formula_l106_106880

-- Define the sequence {a_n} based on the given condition
def a (n : ℕ) : ℚ :=
  match n with
  | 1     => 2
  | (n+1) => 2 / (2 * (n+1) - 1)

-- Define the partial sum of the given sequence a_n over (2n-1)a_n for the given condition
def partial_sum (n : ℕ) : ℚ :=
  fin.sum (fin n) (λ i, (2 * (i.1 + 1) - 1) * a (i.1 + 1))

-- Prove that for all n, a_n = 2/((2n-1))
theorem general_term_formula (n : ℕ) : a n = 2 / (2 * n - 1) :=
by sorry

-- Define the sequence {b_n} as a_n / (2n + 1)
def b (n : ℕ) : ℚ :=
  a n / (2 * n + 1)

-- Define the sum of the first n terms of the sequence {b_n}
def sum_b (n : ℕ) : ℚ :=
  fin.sum (fin n) (λ i, b (i.1 + 1))

-- Prove the sum of the first n terms of the sequence {b_n}
theorem sum_b_formula (n : ℕ) : sum_b n = 2 * n / (2 * n + 1) :=
by sorry

end general_term_formula_sum_b_formula_l106_106880


namespace contact_probability_l106_106539

theorem contact_probability (p : ℝ) :
  let m := 6 in
  let n := 7 in
  let number_of_pairs := m * n in
  1 - (1 - p) ^ number_of_pairs = 1 - (1 - p) ^ 42 :=
by
  let m := 6
  let n := 7
  let number_of_pairs := m * n
  have h1 : number_of_pairs = 42 := by norm_num
  rw h1
  sorry

end contact_probability_l106_106539


namespace projections_lie_in_same_plane_l106_106736

-- Definitions translating the conditions
variable {α : Type*} [plane α]
variable {A B C D E M : α}

@[simp] def quadrilateral {α : Type*} {plane α} (A B C D : α) := ∀ (X Y : α), X ∈ {A, B, C, D} ∧ Y ∈ {A, B, C, D} ∧ X ≠ Y ↔ (X = A ∨ X = B ∨ X = C ∨ X = D) ∧ (Y = A ∨ Y = B ∨ Y = C ∨ Y = D)

@[simp] def perpendicular_diagonals {α : Type*} [plane α] (A B C D M : α) := 
  ((line A C) ∩ (line B D) = {M}) ∧
  ∠ A M C = 90 ∧
  ∠ B M D = 90

@[simp] def height {α : Type*} [plane α] (A B C D E M : α) := 
  ∀ (X : α), X ∈ {A, B, C, D} ∧ 
  (segment E M) ⟂ (segment X M)

-- Proof problem in Lean 4 statement

theorem projections_lie_in_same_plane
  {A B C D E M : α}
  (h1 : quadrilateral A B C D)
  (h2 : perpendicular_diagonals A B C D M)
  (h3 : height A B C D E M) :
  ∃ (P : α), ∀ (F : α), F ∈ {A, B, C, D} ∧ (proj M (plane E F)) ⟂ (plane A B C D E) :=
sorry

end projections_lie_in_same_plane_l106_106736


namespace plot_length_l106_106356

theorem plot_length (b l : ℕ) (h1 : l = b + 20) (h2 : 5300 = 2 * ( l + b ) * 26.50) : l = 60 := 
by
  sorry

end plot_length_l106_106356


namespace min_toothpicks_to_remove_l106_106555

-- Let T be the type representing the triangular lattice figure in terms of rows of toothpicks
structure Lattice (rows : Nat) : Type where
  bottom : Nat -- number of toothpicks in the bottom row
  next : Nat -- number of toothpicks in the next row
  aboveNext : Nat -- number of toothpicks in the above next row
  topmost : Nat -- number of toothpicks in the topmost row

-- Define our specific instance of the lattice
def triangularLattice : Lattice 4 := {
  bottom := 6,
  next := 5,
  aboveNext := 4,
  topmost := 3
}

-- Define a proof problem: The minimum number of toothpicks to remove so that no triangles remain in the lattice
theorem min_toothpicks_to_remove (L : Lattice 4) : Nat :=
  18

-- The statement above says that to remove all triangles in the given lattice,
-- the minimum number of toothpicks to remove is 18.

end min_toothpicks_to_remove_l106_106555


namespace solve_system_l106_106330

noncomputable def x_solution (C₁ C₂ : ℝ) (t : ℝ) : ℝ :=
  -C₁ * exp(2 * t) + 4 * C₂ * exp(-3 * t) + t + t^2

noncomputable def y_solution (C₁ C₂ : ℝ) (t : ℝ) : ℝ :=
  C₁ * exp(2 * t) + C₂ * exp(-3 * t) - 1/2 + t^2

theorem solve_system (C₁ C₂ : ℝ) :
  (∀ t, (differentiable_at ℝ (λ t, x_solution C₁ C₂ t) t) ∧ 
            (differentiable_at ℝ (λ t, y_solution C₁ C₂ t) t)) ∧
  (∀ t, deriv (λ t, x_solution C₁ C₂ t) t = -2 * (x_solution C₁ C₂ t) - 4 * (y_solution C₁ C₂ t) + 1 + 4 * t) ∧
  (∀ t, deriv (λ t, y_solution C₁ C₂ t) t = -(x_solution C₁ C₂ t) + (y_solution C₁ C₂ t) + 3/2 * t^2) :=
begin
  sorry
end

end solve_system_l106_106330


namespace find_a_m_18_l106_106214

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (a1 : ℝ)
variable (m : ℕ)

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) :=
  ∀ n : ℕ, a n = a1 * r^n

def problem_conditions (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :=
  (geometric_sequence a a1 r) ∧
  a m = 3 ∧
  a (m + 6) = 24

theorem find_a_m_18 (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :
  problem_conditions a r a1 m → a (m + 18) = 1536 :=
by
  sorry

end find_a_m_18_l106_106214


namespace no_solution_for_n_eq_neg1_l106_106881

theorem no_solution_for_n_eq_neg1 (x y z : ℝ) : ¬ (∃ x y z, (-1) * x^2 + y = 2 ∧ (-1) * y^2 + z = 2 ∧ (-1) * z^2 + x = 2) :=
by
  sorry

end no_solution_for_n_eq_neg1_l106_106881


namespace pratyya_payel_min_difference_l106_106714

theorem pratyya_payel_min_difference (n m : ℕ) (h : n > m ∧ n - m ≥ 4) :
  ∀ t : ℕ, (2^(t+1) * n - 2^(t+1)) > 2^(t+1) * m + 2^(t+1) :=
by
  sorry

end pratyya_payel_min_difference_l106_106714


namespace sum_of_reciprocals_l106_106253

def a_n (n : ℕ) : ℚ := ( ↑n * (↑n - 1)) / 2

theorem sum_of_reciprocals :
  (∑ n in Finset.range (2022 + 1).filter (fun n => 2 ≤ n), (1 / a_n (n + 2))) = 4044 / 2023 := 
sorry

end sum_of_reciprocals_l106_106253


namespace size_of_each_group_l106_106775

theorem size_of_each_group 
  (skittles : ℕ) (erasers : ℕ) (groups : ℕ)
  (h_skittles : skittles = 4502) (h_erasers : erasers = 4276) (h_groups : groups = 154) :
  (skittles + erasers) / groups = 57 :=
by
  sorry

end size_of_each_group_l106_106775


namespace anna_cupcakes_remaining_l106_106860

theorem anna_cupcakes_remaining :
  let total_cupcakes := 60
  let cupcakes_given_away := (4 / 5 : ℝ) * total_cupcakes
  let cupcakes_after_giving := total_cupcakes - cupcakes_given_away
  let cupcakes_eaten := 3
  let cupcakes_left := cupcakes_after_giving - cupcakes_eaten
  cupcakes_left = 9 :=
by
  sorry

end anna_cupcakes_remaining_l106_106860


namespace discount_price_l106_106509

theorem discount_price (original_price : ℝ) (discount_percent : ℝ) (final_price : ℝ) :
  original_price = 800 ∧ discount_percent = 15 → final_price = 680 :=
by
  intros h
  cases' h with hp hd
  sorry

end discount_price_l106_106509


namespace angle_A_is_140_l106_106211

namespace Proof

variables {A B C D : Type} [parallelogram ABCD]
variable (angleD : ℝ)
variable (angleA : ℝ)
hypothesis angleD_value : angleD = 40
hypothesis property_of_parallelogram : angleA + angleD = 180

theorem angle_A_is_140 : angleA = 140 :=
by {
  rw angleD_value at property_of_parallelogram,
  linarith,
}

end Proof

end angle_A_is_140_l106_106211


namespace average_speed_l106_106306

theorem average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 90) (h2 : d2 = 75) (ht1 : t1 = 1) (ht2 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 82.5 :=
by
  sorry

end average_speed_l106_106306


namespace number_of_ways_to_elect_officers_l106_106866

theorem number_of_ways_to_elect_officers (total_candidates past_officers positions : ℕ)
  (total_candidates_eq : total_candidates = 18)
  (past_officers_eq : past_officers = 8)
  (positions_eq : positions = 6) :
  ∃ k : ℕ, k = 16338 := 
by
  have h1 : nat.choose 18 6 = 18564 := by sorry
  have h2 : nat.choose 10 6 = 210 := by sorry
  have h3 : 8 * (nat.choose 10 5) = 2016 := by sorry
  have h4 : nat.choose 18 6 - nat.choose 10 6 - 8 * (nat.choose 10 5) = 16338 := by sorry
  use 16338
  exact h4

end number_of_ways_to_elect_officers_l106_106866


namespace man_speed_in_still_water_l106_106385

def swimming_speed (d_down : ℕ) (t_down : ℕ) (d_up : ℕ) (t_up : ℕ) 
  (v_m : ℕ) (v_s : ℕ) : Prop :=
  (d_down / t_down = v_m + v_s) ∧ (d_up / t_up = v_m - v_s)

theorem man_speed_in_still_water (d_down : ℕ) (t_down : ℕ) 
  (d_up : ℕ) (t_up : ℕ) (v_s : ℕ) (h₁ : d_down / t_down = 16) 
  (h₂ : d_up / t_up = 6) : ∃ v_m : ℕ, v_m = 11 := 
by
  let v_m := 11
  use v_m
  rw [← h₁, ← h₂]
  sorry -- proof skipped

end man_speed_in_still_water_l106_106385


namespace adult_ticket_cost_l106_106258

theorem adult_ticket_cost (C : ℝ) (h1 : ∀ (a : ℝ), a = C + 8)
  (h2 : ∀ (s : ℝ), s = C + 4)
  (h3 : 5 * C + 2 * (C + 8) + 2 * (C + 4) = 150) :
  ∃ (a : ℝ), a = 22 :=
by {
  sorry
}

end adult_ticket_cost_l106_106258


namespace alice_bob_total_dollars_l106_106408

-- Define Alice's amount in dollars
def alice_amount : ℚ := 5 / 8

-- Define Bob's amount in dollars
def bob_amount : ℚ := 3 / 5

-- Define the total amount in dollars
def total_amount : ℚ := alice_amount + bob_amount

theorem alice_bob_total_dollars : (alice_amount + bob_amount : ℚ) = 1.225 := by
    sorry

end alice_bob_total_dollars_l106_106408


namespace tan_double_angle_third_quadrant_l106_106943

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : sin (π - α) = - (3 / 5)) : 
  tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_third_quadrant_l106_106943
