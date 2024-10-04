import Mathlib
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.EuclideanGeometry
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.GroupPower.Order
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Manifold
import Mathlib.Init.Data.Int.Basic
import Mathlib.Logic.Basic
import Mathlib.Probability.Distribution
import Mathlib.Probability.Theory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace speed_of_second_train_l137_137443

theorem speed_of_second_train
  (t₁ : ℕ := 2)  -- Time the first train sets off (2:00 pm in hours)
  (s₁ : ℝ := 70) -- Speed of the first train in km/h
  (t₂ : ℕ := 3)  -- Time the second train sets off (3:00 pm in hours)
  (t₃ : ℕ := 10) -- Time when the second train catches the first train (10:00 pm in hours)
  : ∃ S : ℝ, S = 80 := sorry

end speed_of_second_train_l137_137443


namespace tan_add_pi_four_cos_beta_l137_137852

theorem tan_add_pi_four (α : ℝ) (h1 : sin α = 3/5) (h2 : α ∈ (π/2, π)) : 
  tan (α + π / 4) = 1/7 :=
sorry

theorem cos_beta (α β : ℝ) (h1 : α ∈ (π/2, π)) (h2 : β ∈ (0, π/2)) 
  (h3 : cos (α - β) = 1/3) : cos β = (6 * real.sqrt 2 - 4) / 15 := 
sorry

end tan_add_pi_four_cos_beta_l137_137852


namespace sally_total_peaches_l137_137684

-- Assumptions from the conditions
def initial_peaches : ℕ := 13
def peaches_given_away : ℕ := 6
def peaches_first_orchard : ℕ := 55
def peaches_second_orchard : ℕ := 110

-- Proof statement: Sally's total number of peaches now
theorem sally_total_peaches : 
  let remaining_after_giveaway := initial_peaches - peaches_given_away in
  let total_after_first_orchard := remaining_after_giveaway + peaches_first_orchard in
  let total_peaches := total_after_first_orchard + peaches_second_orchard in
  total_peaches = 172 :=
  by 
  -- The specifics of the proof would go here
  sorry

end sally_total_peaches_l137_137684


namespace exist_two_pies_differing_in_both_l137_137159

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l137_137159


namespace adding_subtracting_method_is_valid_method_l137_137705

theorem adding_subtracting_method_is_valid_method : 
  ∃ method, method = "addition-subtraction" ∧ 
  (∃ solver, solver method = "reduces the system to a single variable") := 
sorry

end adding_subtracting_method_is_valid_method_l137_137705


namespace nth_term_of_sequence_99_l137_137692

def sequence_rule (n : ℕ) : ℕ :=
  if n < 20 then n * 9
  else if n % 2 = 0 then n / 2
  else if n > 19 ∧ n % 7 ≠ 0 then n - 5
  else n + 7

noncomputable def sequence_nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.repeat sequence_rule n start

theorem nth_term_of_sequence_99 :
  sequence_nth_term 65 98 = 30 :=
sorry

end nth_term_of_sequence_99_l137_137692


namespace non_congruent_triangles_count_l137_137545

noncomputable def points := [
  (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), -- Row 1
  (0.5, 1), (1.5, 1), (2.5, 1), (3.5, 1), (4.5, 1) -- Row 2
]

/--
Given a set of 10 points arranged on a grid with coordinates:
- Row 1 (y = 0): (0,0), (1,0), (2,0), (3,0), (4,0)
- Row 2 (y = 1): (0.5,1), (1.5,1), (2.5,1), (3.5,1), (4.5,1)

Prove that the number of non-congruent triangles that can be formed by choosing vertices from these points is 4.
-/
theorem non_congruent_triangles_count : 
  ∃ (triangles : finset (finset (ℝ × ℝ))), 
    (∀ t ∈ triangles, t.card = 3) ∧ 
    (points : finset (ℝ × ℝ)) = points.to_finset ∧ 
    triangles.card = 4 :=
sorry

end non_congruent_triangles_count_l137_137545


namespace magnitude_of_vector_sum_l137_137580

open Real

variable {α : Type*} [InnerProductSpace ℝ α] (a b : α)

theorem magnitude_of_vector_sum (ha : ‖a‖ = 3) (hb : ‖b‖ = 4) (angle_ab : real_angle a b = 120) :
  ‖a + 2 • b‖ = sqrt 61 :=
by
  sorry

end magnitude_of_vector_sum_l137_137580


namespace target_winning_percentage_l137_137427

theorem target_winning_percentage 
  (initial_games : ℕ) (initial_win_pct : ℚ) 
  (additional_games : ℚ) (additional_win_pct : ℚ) : 
  (initial_games = 40) →
  (initial_win_pct = 0.40) →
  (additional_games = 39.999999999999986) →
  (additional_win_pct = 0.80) →
  let total_games := initial_games + additional_games.to_nat,
      initial_wins := initial_win_pct * initial_games,
      additional_wins := additional_win_pct * additional_games,
      total_wins := initial_wins + additional_wins in
  (total_wins / total_games * 100 = 60) :=
by 
  intro h1 h2 h3 h4
  have eq1 : initial_games + additional_games.to_nat = 80 := sorry
  have eq2 : initial_win_pct * 40 + additional_win_pct * 40 = 48 := sorry
  have eq3 : (48 / 80) * 100 = 60 := sorry
  exact eq3

end target_winning_percentage_l137_137427


namespace question_1_question_2_l137_137625

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B : ℝ × ℝ := (0, 2)
noncomputable def C (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C x).1 - A.1, (C x).2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2

theorem question_1 (x : ℝ) :
  f x = 2 * Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4) + 4 ∧ 
  ∀ x, f (x + Real.pi) = f x := 
sorry

theorem question_2 (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  2 < f x ∧ f x ≤ 4 + 2 * Real.sqrt 2 :=
sorry

end question_1_question_2_l137_137625


namespace total_number_of_trees_is_10_l137_137683

-- Define the known quantities
def percentage_tree_A : ℝ := 0.5
def percentage_tree_B : ℝ := 0.5

def oranges_per_tree_A : ℤ := 10
def good_percentage_A : ℝ := 0.6

def oranges_per_tree_B : ℤ := 15
def good_percentage_B : ℝ := 1 / 3

def total_good_oranges : ℤ := 55

-- Number of trees
theorem total_number_of_trees_is_10 : 
  ∀ (n : ℤ),
  (percentage_tree_A * (oranges_per_tree_A * good_percentage_A) + 
   percentage_tree_B * (oranges_per_tree_B * good_percentage_B)) * n = total_good_oranges →
  n = 10 :=
by
  intros n h
  sorry

end total_number_of_trees_is_10_l137_137683


namespace probability_even_and_div_by_three_l137_137274

-- Definitions of the conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_div_by_three (n : ℕ) : Prop := n % 3 = 0

-- Definition of die sides
def sides := {1, 2, 3, 4, 5, 6}

-- Probability computation
def probability_even_first_die : ℚ :=
  ↑(sides.filter is_even).card / ↑sides.card

def probability_div_by_three_second_die : ℚ :=
  ↑(sides.filter is_div_by_three).card / ↑sides.card

def combined_probability : ℚ :=
  probability_even_first_die * probability_div_by_three_second_die

-- The main hypothesis to prove
theorem probability_even_and_div_by_three :
  combined_probability = 1/6 :=
by
  sorry

end probability_even_and_div_by_three_l137_137274


namespace value_of_a_l137_137536

theorem value_of_a (a : ℝ) (h : 0 < a)
  (length_condition : (λ s, ∃ x_1 x_2, x_1 ≤ x_2 ∧ s = {x | x_1 ≤ x ∧ x ≤ x_2} ∧ length s = 1/2) 
    ({x : ℝ | sqrt (x + a) + sqrt (x - a) ≤ sqrt (2 * (x + 1))})) :
  a = 3/4 := by
  sorry

end value_of_a_l137_137536


namespace pies_differ_in_both_l137_137169

-- Defining types of pies
inductive Filling where
  | apple : Filling
  | cherry : Filling

inductive Preparation where
  | fried : Preparation
  | baked : Preparation

structure Pie where
  filling : Filling
  preparation : Preparation

-- The set of all possible pies
def allPies : Set Pie :=
  { ⟨Filling.apple, Preparation.fried⟩,
    ⟨Filling.apple, Preparation.baked⟩,
    ⟨Filling.cherry, Preparation.fried⟩,
    ⟨Filling.cherry, Preparation.baked⟩ }

-- Theorem stating that we can buy two pies that differ in both filling and preparation
theorem pies_differ_in_both (pies : Set Pie) (h : 3 ≤ pies.card) :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
  sorry

end pies_differ_in_both_l137_137169


namespace range_of_a_l137_137997

open Set

theorem range_of_a (a x : ℝ) (p : ℝ → Prop) (q : ℝ → ℝ → Prop)
    (hp : p x → |x - a| > 3)
    (hq : q x a → (x + 1) * (2 * x - 1) ≥ 0)
    (hsuff : ∀ x, ¬p x → q x a) :
    {a | ∀ x, (¬ (|x - a| > 3) → (x + 1) * (2 * x - 1) ≥ 0) → (( a ≤ -4) ∨ (a ≥ 7 / 2))} :=
by
  sorry

end range_of_a_l137_137997


namespace athlete_difference_is_30_l137_137774

def initial_athletes : ℕ := 600
def leaving_rate : ℕ := 35
def leaving_duration : ℕ := 6
def arrival_rate : ℕ := 20
def arrival_duration : ℕ := 9

def athletes_left : ℕ := leaving_rate * leaving_duration
def new_athletes : ℕ := arrival_rate * arrival_duration
def remaining_athletes : ℕ := initial_athletes - athletes_left
def final_athletes : ℕ := remaining_athletes + new_athletes
def athlete_difference : ℕ := initial_athletes - final_athletes

theorem athlete_difference_is_30 : athlete_difference = 30 :=
by
  show athlete_difference = 30
  -- Proof goes here
  sorry

end athlete_difference_is_30_l137_137774


namespace proof_problem_l137_137100

def func_property1 (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, f x1 + f x2 = 2 * f ((x1 + x2) / 2) * f ((x1 - x2) / 2)

def func_property2 (f : ℝ → ℝ) : Prop :=
f (π / 2) = 0

noncomputable def is_periodic_with_2pi (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (x + 2 * π) = f x

def is_function_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

def recursive_relation (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ) (n : ℕ), f (2^n * x) = (2 * (f (2^(n-1) * x))^2) - 1

theorem proof_problem (f : ℝ → ℝ)
  (h1 : func_property1 f)
  (h2 : func_property2 f):
  is_periodic_with_2pi f ∧ is_function_even f ∧ recursive_relation f :=
by
  sorry

end proof_problem_l137_137100


namespace smallest_multiple_of_112_has_4_digits_l137_137520

theorem smallest_multiple_of_112_has_4_digits : 
  (∀ n : ℕ, n < 1008 → n % 112 = 0 → false) ∧ (1008 % 112 = 0) ∧ (1008.digits 10).length = 4 := 
by
  sorry

end smallest_multiple_of_112_has_4_digits_l137_137520


namespace f_at_4_l137_137930

def f_inverse (x : ℝ) (hx : x > 0) : ℝ :=
  x^2

theorem f_at_4 (h : ∃ f : ℝ → ℝ, ∀ x, x > 0 → f_inverse (f x) (f x > 0) = x) : 
  f (4 : ℝ) = 2 := 
sorry

end f_at_4_l137_137930


namespace average_salary_all_workers_l137_137324

theorem average_salary_all_workers 
  (n : ℕ) (avg_salary_technicians avg_salary_rest total_avg_salary : ℝ)
  (h1 : n = 7) 
  (h2 : avg_salary_technicians = 8000) 
  (h3 : avg_salary_rest = 6000)
  (h4 : total_avg_salary = avg_salary_technicians) : 
  total_avg_salary = 8000 :=
by sorry

end average_salary_all_workers_l137_137324


namespace total_books_l137_137815

theorem total_books (d k g : ℕ) 
  (h1 : d = 6) 
  (h2 : k = d / 2) 
  (h3 : g = 5 * (d + k)) : 
  d + k + g = 54 :=
by
  sorry

end total_books_l137_137815


namespace max_elements_no_sum_divisible_by_5_l137_137641

open Finset Function

def no_sum_divisible_by {T : Finset ℕ} (s : ℕ) : Prop :=
∀ {a b : ℕ}, a ∈ T → b ∈ T → a ≠ b → (a + b) % s ≠ 0

theorem max_elements_no_sum_divisible_by_5 :
  ∀ T : Finset ℕ, T ⊆ range 1 101 → no_sum_divisible_by 5 T → T.card ≤ 60 :=
by
  intros
  sorry

end max_elements_no_sum_divisible_by_5_l137_137641


namespace general_term_l137_137117

open Nat

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

theorem general_term (n : ℕ) (hn : n > 0) : (S n - S (n - 1)) = 4 * n - 5 := by
  sorry

end general_term_l137_137117


namespace abs_quotient_eq_sqrt_7_div_2_l137_137923

theorem abs_quotient_eq_sqrt_7_div_2 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 2) :=
by
  sorry

end abs_quotient_eq_sqrt_7_div_2_l137_137923


namespace find_ABC_l137_137703

def f (x : ℝ) (A B C : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem find_ABC (A B C : ℤ) (h : ∀ x : ℝ, x > 5 → f x A B C > 0.5) :
  A + B + C = -18 := 
sorry

end find_ABC_l137_137703


namespace increasing_interval_of_f_l137_137350

noncomputable def f (x : ℝ) : ℝ := x⁻²

theorem increasing_interval_of_f :
  (∀ x y : ℝ, x < y → f x < f y → x ∈ set.Ioo (-(∞ : ℝ)) 0 ∧ y ∈ set.Ioo (-(∞ : ℝ)) 0) :=
sorry

end increasing_interval_of_f_l137_137350


namespace max_trig_expression_l137_137502

theorem max_trig_expression (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_trig_expression_l137_137502


namespace number_of_perfect_square_ratios_l137_137070

theorem number_of_perfect_square_ratios :
  (∃ n : ℕ, ∀ n < 30 ∧ ∃ k : ℕ, n / (30 - n) = k^2) = 4 := 
sorry

end number_of_perfect_square_ratios_l137_137070


namespace tetrahedron_plane_division_l137_137407

structure Tetrahedron (A B C D : Point) : Prop :=
  (midpoint_AD : Point)
  (on_extension_AB : Point)
  (on_extension_AC : Point)

variables {A B C D M N K : Point}

noncomputable def plane_division_ratios
  (M : Point) (N : Point) (K : Point)
  [Tetrahedron A B C D] 
  (hM : M = midpoint (A, D))
  (hBN : BN = AB)
  (hCK : CK = 2 * AC)
  : (ratios : Ratios) :=
  { ratio_DC := (3, 2),
    ratio_DB := (2, 1) }

theorem tetrahedron_plane_division
  (hT : Tetrahedron A B C D)
  (hM_mid : M = midpoint (A, D))
  (hN_ext : ∃ N, N ∈ ext (A, B) ∧ BN = AB)
  (hK_ext : ∃ K, K ∈ ext (A, C) ∧ CK = 2 * AC) :
  plane_division_ratios M N K :=
by
  sorry

end tetrahedron_plane_division_l137_137407


namespace volume_square_pyramid_l137_137905

/-- A square pyramid has a base side length of 20 cm and a height of 20 cm. Prove that the volume of the pyramid is 2666 2/3 cubic centimeters. -/
theorem volume_square_pyramid : 
  let base_side_length := 20
  let height := 20
  volume (pyramid square_pyramid.base base_side_length height) = 2666 + 2/3 := 
sorry

end volume_square_pyramid_l137_137905


namespace distance_from_O_to_plane_ABC_is_sqrt3_by_3_l137_137864

-- Defining the given conditions
variables {A B C S O : Point}
variables [EuclideanSpace ℝ V] 
variables (plane_abc : Plane ℝ)
variables (sphere : Sphere ℝ)

-- Hypotheses based on given conditions
axiom iso_triangle_abc : isosceles_right_triangle A B C AB 
axiom base_hypotenuse : distance A B = 2
axiom sa_equal_sb : distance S A = distance S B
axiom sb_equal_sc : distance S B = distance S C
axiom sc_equal_sa : distance S C = distance S A
axiom sa_value : distance S A = 2
axiom ab_value : distance A B = 2
axiom points_on_sphere : is_on_sphere A O sphere ∧ is_on_sphere B O sphere ∧ 
                         is_on_sphere C O sphere ∧ is_on_sphere S O sphere

-- Definition of the problem to be proven
theorem distance_from_O_to_plane_ABC_is_sqrt3_by_3 : 
  distance_from_point_to_plane O plane_abc = real.sqrt 3 / 3 :=
by 
  sorry

end distance_from_O_to_plane_ABC_is_sqrt3_by_3_l137_137864


namespace x_plus_y_equals_8_5_l137_137033

-- Definitions
def segment_CD := 5
def segment_C'D' := 7
def midpoint_C := segment_CD / 2
def midpoint_D := segment_CD / 2
def midpoint_C' := segment_C'D' / 2
def midpoint_D' := segment_C'D' / 2
def point_P (a : ℝ) := 2 * a
def k (x : ℝ) := 2 - (3.5 / x)

-- Total distance function calculation
def calculate_total_distance (a : ℝ) : ℝ :=
  let x := point_P a
  let y := k x * x
  x + y

-- Main statement: We prove that x + y = 8.5 under the condition given
theorem x_plus_y_equals_8_5 (a : ℝ) : (calculate_total_distance a) = 8.5 :=
by sorry

end x_plus_y_equals_8_5_l137_137033


namespace three_kids_savings_l137_137321

noncomputable def total_savings (teagan_pennies teagan_one_dollar rex_nickels rex_quarters rex_pounds rex_conv_rate toni_dimes toni_five_dollars : ℕ) (conv_rate : ℝ) : ℝ :=
  let teagan_total := teagan_pennies * 0.01 + teagan_one_dollar in
  let rex_total := rex_nickels * 0.05 + rex_quarters * 0.25 + rex_pounds * conv_rate in
  let toni_total := toni_dimes * 0.10 + toni_five_dollars * 5.00 in
  teagan_total + rex_total + toni_total

theorem three_kids_savings :
  total_savings 200 15 100 45 8 1.38 330 12 = 137.29 :=
by
  -- This is where the proof would go, but it is omitted as per the instructions
  sorry

end three_kids_savings_l137_137321


namespace decreasing_on_0_4_iff_l137_137112

noncomputable def f (k x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

theorem decreasing_on_0_4_iff (k : ℝ) : 
  (∀ x ∈ Ioo 0 4, deriv (f k) x ≤ 0) ↔ (k ≤ 1 / 3) :=
sorry

end decreasing_on_0_4_iff_l137_137112


namespace triangle_area_expression_l137_137304
open Real

variables {A B C : Type}
variables (α β γ : ℝ)
variables (b c : ℝ)
variables {area : Type}
variables [has_area : has_area (A B C)]
variables [has_angles : has_angles α β γ]
variables [has_lengths : has_lengths b c]

theorem triangle_area_expression
  (h₁ : has_area.angle_bisectors_equal α) :
  area = b * c * sin (α / 2) ^ 2 * tan ((β + γ) / 2) * tan ((β - γ) / 2) :=
sorry

end triangle_area_expression_l137_137304


namespace phil_books_remaining_pages_l137_137279

/-- We define the initial number of books and the number of pages per book. -/
def initial_books : Nat := 10
def pages_per_book : Nat := 100
def lost_books : Nat := 2

/-- The goal is to find the total number of pages Phil has left after losing 2 books. -/
theorem phil_books_remaining_pages : (initial_books - lost_books) * pages_per_book = 800 := by 
  -- The proof will go here
  sorry

end phil_books_remaining_pages_l137_137279


namespace correct_relationship_l137_137553

noncomputable def a := 0.6^(1 / 2)
noncomputable def b := 0.6^(1 / 3)
noncomputable def c := Real.log 3 / Real.log 0.6

theorem correct_relationship : c < a ∧ a < b :=
by
  -- The proof goes here
  sorry

end correct_relationship_l137_137553


namespace exists_eight_consecutive_with_two_lucky_exists_twelve_consecutive_without_lucky_thirteen_consecutive_contains_lucky_l137_137261

-- Define a number as lucky if the sum of its digits is divisible by 7
def is_lucky_number (n : ℕ) : Prop :=
  (sum_digits n) % 7 = 0

-- Step 1: There exist eight consecutive numbers where exactly two are lucky
theorem exists_eight_consecutive_with_two_lucky :
  ∃ (a : ℕ), (∃ b, b = a + 1) ∧ (∀ i, i ≥ a ∧ i < a + 8 → i + 1 = b + 1 → (is_lucky_number a ∨ is_lucky_number (a + 7))) ∧
  (∃ c, c = a + 7) ∧ (∀ d, d = a + 6 → ¬is_lucky_number d) :=
sorry

-- Step 2: There exist twelve consecutive numbers where none are lucky
theorem exists_twelve_consecutive_without_lucky :
  ∃ (a : ℕ), ∀ i, i ≥ a ∧ i < a + 12 → ¬is_lucky_number i :=
sorry

-- Step 3: Any sequence of thirteen consecutive numbers contains at least one lucky number
theorem thirteen_consecutive_contains_lucky :
  ∀ (a : ℕ), ∃ i, i ≥ a ∧ i < a + 13 ∧ is_lucky_number i :=
sorry

-- Supporting definition to sum the digits of a number
noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum


end exists_eight_consecutive_with_two_lucky_exists_twelve_consecutive_without_lucky_thirteen_consecutive_contains_lucky_l137_137261


namespace prob_4_vertices_same_plane_of_cube_l137_137080

/-- A cube contains 8 vertices. We need to calculate the probability that 4 chosen vertices lie on the same plane. -/
theorem prob_4_vertices_same_plane_of_cube : 
  let total_ways := (Nat.choose 8 4)
  let favorable_ways := 12
  in (favorable_ways / total_ways : ℚ) = 6 / 35 :=
by
  sorry

end prob_4_vertices_same_plane_of_cube_l137_137080


namespace converse_false_l137_137331

variable {a b : ℝ}

theorem converse_false : (¬ (∀ a b : ℝ, (ab = 0 → a = 0))) :=
by
  sorry

end converse_false_l137_137331


namespace pies_differ_in_both_l137_137170

-- Defining types of pies
inductive Filling where
  | apple : Filling
  | cherry : Filling

inductive Preparation where
  | fried : Preparation
  | baked : Preparation

structure Pie where
  filling : Filling
  preparation : Preparation

-- The set of all possible pies
def allPies : Set Pie :=
  { ⟨Filling.apple, Preparation.fried⟩,
    ⟨Filling.apple, Preparation.baked⟩,
    ⟨Filling.cherry, Preparation.fried⟩,
    ⟨Filling.cherry, Preparation.baked⟩ }

-- Theorem stating that we can buy two pies that differ in both filling and preparation
theorem pies_differ_in_both (pies : Set Pie) (h : 3 ≤ pies.card) :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
  sorry

end pies_differ_in_both_l137_137170


namespace part1_part2_part3_l137_137478

-- Definition of R_p sequence and its conditions
def Rp_seq (p : ℝ) (a : ℕ → ℝ) : Prop :=
  (a 1 + p ≥ 0) ∧ (a 2 + p = 0) ∧
  (∀ n : ℕ, 0 < n → a (4 * n - 1) < a (4 * n)) ∧
  (∀ m n : ℕ, 0 < m → 0 < n → a (m + n) ∈ {a m + a n + p, a m + a n + p + 1})

-- Part 1: The sequence cannot be an R_2 sequence.
theorem part1 : ¬(Rp_seq 2 (λ n, [2, -2, 0, 1].nth (n - 1).getOrElse 0)) :=
sorry

-- Part 2: For an R_0 sequence, find the value of a_5.
theorem part2 (a : ℕ → ℝ) (hR0 : Rp_seq 0 a) : a 5 = 1 :=
sorry

-- Part 3: Sum of the first n terms of the sequence and the condition on S_n.
theorem part3 (p : ℝ) (a : ℕ → ℝ) (hRp : Rp_seq p a) (S : ℕ → ℝ) 
  (hS : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) :
  (∀ n : ℕ, 0 < n → S n ≥ S 10) ↔ p = 2 :=
sorry

end part1_part2_part3_l137_137478


namespace probability_both_visible_l137_137465

noncomputable def emma_lap_time : ℕ := 100
noncomputable def ethan_lap_time : ℕ := 75
noncomputable def start_time : ℕ := 0
noncomputable def photo_start_minute : ℕ := 12 * 60 -- converted to seconds
noncomputable def photo_end_minute : ℕ := 13 * 60 -- converted to seconds
noncomputable def photo_visible_angle : ℚ := 1 / 3

theorem probability_both_visible :
  ∀ start_time photo_start_minute photo_end_minute emma_lap_time ethan_lap_time photo_visible_angle,
  start_time = 0 →
  photo_start_minute = 12 * 60 →
  photo_end_minute = 13 * 60 →
  emma_lap_time = 100 →
  ethan_lap_time = 75 →
  photo_visible_angle = 1 / 3 →
  (∃ t, photo_start_minute ≤ t ∧ t < photo_end_minute ∧
        (t % emma_lap_time ≤ (photo_visible_angle * emma_lap_time) / 2 ∨
         t % emma_lap_time ≥ emma_lap_time - (photo_visible_angle * emma_lap_time) / 2) ∧
        (t % ethan_lap_time ≤ (photo_visible_angle * ethan_lap_time) / 2 ∨
         t % ethan_lap_time ≥ ethan_lap_time - (photo_visible_angle * ethan_lap_time) / 2)) ↔
  true :=
sorry

end probability_both_visible_l137_137465


namespace difference_white_black_l137_137371

def total_stones : ℕ := 928
def white_stones : ℕ := 713
def black_stones : ℕ := total_stones - white_stones

theorem difference_white_black :
  (white_stones - black_stones = 498) :=
by
  -- Leaving the proof for later
  sorry

end difference_white_black_l137_137371


namespace min_triangle_cuts_l137_137043

theorem min_triangle_cuts (N : ℕ) (hT : ∀ (T : Type) (s : ℕ), (s = 12) → (N = 16)) :
  ∃ (N : ℕ), N = 16 :=
by { use 16, sorry }

end min_triangle_cuts_l137_137043


namespace distinct_persons_serving_on_boards_l137_137479

theorem distinct_persons_serving_on_boards 
  (A B C : set ℕ)
  (hA : A.card = 8) 
  (hB : B.card = 8) 
  (hC : C.card = 8)
  (hABC : (A ∩ B ∩ C).card = 4) 
  (hAB : (A ∩ B).card = 5) 
  (hBC : (B ∩ C).card = 5) 
  (hAC : (A ∩ C).card = 5) :
  (A ∪ B ∪ C).card = 16 :=
sorry

end distinct_persons_serving_on_boards_l137_137479


namespace calculate_length_l137_137349

-- Given definitions
def width : ℝ := 25
def height_decrease : ℝ := 0.5
def volume_to_remove_gallons : ℝ := 5250
def conversion_factor : ℝ := 7.5
def volume_to_remove_cubic_feet : ℝ := volume_to_remove_gallons / conversion_factor -- which equals 700 cubic feet

theorem calculate_length : 
  ∃ (length : ℝ), 
    (volume_to_remove_cubic_feet = length * width * height_decrease) ∧ 
    length = 56 := 
by 
  sorry

end calculate_length_l137_137349


namespace angle_RPS_is_1_degree_l137_137952

-- Definitions of the given angles
def angle_QRS : ℝ := 150
def angle_PQS : ℝ := 60
def angle_PSQ : ℝ := 49
def angle_QPR : ℝ := 70

-- Definition for the calculated angle QPS
def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Definition for the target angle RPS
def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The theorem we aim to prove
theorem angle_RPS_is_1_degree : angle_RPS = 1 := by
  sorry

end angle_RPS_is_1_degree_l137_137952


namespace largest_side_of_rectangle_l137_137805

theorem largest_side_of_rectangle (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 1920) : 
    max l w = 101 := 
sorry

end largest_side_of_rectangle_l137_137805


namespace part1_part2_l137_137669

-- Part 1
theorem part1 : sqrt (1 / 4 * (1 / 5 - 1 / 6)) = 1 / 5 * sqrt (5 / 24) :=
by
  sorry

-- Part 2
theorem part2 (n : ℕ) (h : 1 ≤ n) :
  sqrt (1 / n * (1 / (n + 1) - 1 / (n + 2))) = 1 / (n + 1) * sqrt ((n + 1) / (n * (n + 2))) :=
by
  sorry

end part1_part2_l137_137669


namespace remaining_lemons_proof_l137_137299

-- Definitions for initial conditions
def initial_lemons_first_tree   := 15
def initial_lemons_second_tree  := 20
def initial_lemons_third_tree   := 25

def sally_picked_first_tree     := 7
def mary_picked_second_tree     := 9
def tom_picked_first_tree       := 12

def lemons_fell_each_tree       := 4
def animals_eaten_per_tree      := lemons_fell_each_tree / 2

-- Definitions for intermediate calculations
def remaining_lemons_first_tree_full := initial_lemons_first_tree - sally_picked_first_tree - tom_picked_first_tree
def remaining_lemons_first_tree      := if remaining_lemons_first_tree_full < 0 then 0 else remaining_lemons_first_tree_full

def remaining_lemons_second_tree := initial_lemons_second_tree - mary_picked_second_tree

def mary_picked_third_tree := (remaining_lemons_second_tree : ℚ) / 2
def remaining_lemons_third_tree_full := (initial_lemons_third_tree : ℚ) - mary_picked_third_tree
def remaining_lemons_third_tree      := Nat.floor remaining_lemons_third_tree_full

-- Adjusting for fallen and eaten lemons
def final_remaining_lemons_first_tree_full := remaining_lemons_first_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_first_tree      := if final_remaining_lemons_first_tree_full < 0 then 0 else final_remaining_lemons_first_tree_full

def final_remaining_lemons_second_tree     := remaining_lemons_second_tree - lemons_fell_each_tree + animals_eaten_per_tree

def final_remaining_lemons_third_tree_full := remaining_lemons_third_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_third_tree      := if final_remaining_lemons_third_tree_full < 0 then 0 else final_remaining_lemons_third_tree_full

-- Lean 4 statement to prove the equivalence
theorem remaining_lemons_proof :
  final_remaining_lemons_first_tree = 0 ∧
  final_remaining_lemons_second_tree = 9 ∧
  final_remaining_lemons_third_tree = 18 :=
by
  -- The proof is omitted as per the requirement
  sorry

end remaining_lemons_proof_l137_137299


namespace total_hotdogs_brought_l137_137718

-- Define the number of hotdogs brought by the first and second neighbors based on given conditions.

def first_neighbor_hotdogs : Nat := 75
def second_neighbor_hotdogs : Nat := first_neighbor_hotdogs - 25

-- Prove that the total hotdogs brought by the neighbors equals 125.
theorem total_hotdogs_brought :
  first_neighbor_hotdogs + second_neighbor_hotdogs = 125 :=
by
  -- statement only, proof not required
  sorry

end total_hotdogs_brought_l137_137718


namespace vector_projection_l137_137507

noncomputable def projection_vector := λ (v w : ℝ × ℝ × ℝ), 
  let dot_vw := v.1 * w.1 + v.2 * w.2 + v.3 * w.3 in
  let dot_ww := w.1 * w.1 + w.2 * w.2 + w.3 * w.3 in
  (dot_vw / dot_ww) • w

theorem vector_projection : 
  projection_vector (4, -1, 3) (3, 1, 2) = (51/14, 17/14, 17/7) := 
  sorry

end vector_projection_l137_137507


namespace problem_sequence_a100_l137_137933

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2 else nat.rec_on n (λ k ih, ih + 2 * (k-1)) 2

theorem problem_sequence_a100 :
  sequence 100 = 9902 :=
by sorry

end problem_sequence_a100_l137_137933


namespace det_A_pow_l137_137139

variable {A : Matrix ℝ ℝ}

theorem det_A_pow (h : det A = -7) : det (A^5) = -16807 := 
by sorry

end det_A_pow_l137_137139


namespace can_buy_two_pies_different_in_both_l137_137177

structure Pie :=
  (filling : Type)
  (preparation : Type)

def apple : Type := unit
def cherry : Type := unit
def fried : Type := unit
def baked : Type := unit

def apple_fried : Pie := { filling := apple, preparation := fried }
def apple_baked : Pie := { filling := apple, preparation := baked }
def cherry_fried : Pie := { filling := cherry, preparation := fried }
def cherry_baked : Pie := { filling := cherry, preparation := baked }

def possible_pies : List Pie := [apple_fried, apple_baked, cherry_fried, cherry_baked]

theorem can_buy_two_pies_different_in_both 
  (available_pies : List Pie) 
  (h : available_pies.length ≥ 3) : 
  ∃ (p1 p2 : Pie), p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation := 
  sorry

end can_buy_two_pies_different_in_both_l137_137177


namespace range_of_m_l137_137892

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → (m ≤ 2 ∧ m ≠ -2) :=
begin
  sorry
end

end range_of_m_l137_137892


namespace cost_of_45_lilies_l137_137035

-- Definitions of the given conditions
def cost_per_lily := 30 / 18
def lilies_18_bouquet_cost := 30
def number_of_lilies_in_bouquet := 45

-- Theorem stating the mathematical proof problem
theorem cost_of_45_lilies : cost_per_lily * number_of_lilies_in_bouquet = 75 := by
  -- The proof is omitted
  sorry

end cost_of_45_lilies_l137_137035


namespace lana_pages_after_adding_duane_l137_137972

theorem lana_pages_after_adding_duane :
  ∀ (lana_initial_pages duane_total_pages : ℕ), 
  lana_initial_pages = 8 → 
  duane_total_pages = 42 → 
  lana_initial_pages + (duane_total_pages / 2) = 29 :=
by
  intros lana_initial_pages duane_total_pages h_lana h_duane
  rw [h_lana, h_duane]
  norm_num

end lana_pages_after_adding_duane_l137_137972


namespace equilateral_triangle_of_arithmetic_progression_l137_137348

theorem equilateral_triangle_of_arithmetic_progression 
  (ABC : Triangle)
  (angles_AP : arithmeticProgression ABC.angles)
  (altitudes_AP : arithmeticProgression ABC.altitudes) :
  ABC.isEquilateral :=
sorry

end equilateral_triangle_of_arithmetic_progression_l137_137348


namespace sqrt_expression_eq_seven_div_two_l137_137039

theorem sqrt_expression_eq_seven_div_two :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 / Real.sqrt 24) = 7 / 2 :=
by
  sorry

end sqrt_expression_eq_seven_div_two_l137_137039


namespace tangency_point_on_line_of_centers_l137_137676

noncomputable def line_of_centers {P : Type} [MetricSpace P] (C1 C2 : Set P) 
  (O₁ O₂ T : P) (h₁ : C1 = metric.sphere O₁ r₁) (h₂ : C2 = metric.sphere O₂ r₂) 
  (hT : T ∈ C1 ∧ T ∈ C2) : Set P :=
{p | ∃ (k : ℝ), p = O₁ + k • (O₂ - O₁)} -- Definition of line through O1 and O2

theorem tangency_point_on_line_of_centers {P : Type} [MetricSpace P] 
  (C1 C2 : Set P) (O₁ O₂ T : P)
  (h₁ : C1 = metric.sphere O₁ r₁) (h₂ : C2 = metric.sphere O₂ r₂) 
  (hT : T ∈ C1 ∧ T ∈ C2) :
  T ∈ line_of_centers C1 C2 O₁ O₂ T h₁ h₂ hT :=
sorry

end tangency_point_on_line_of_centers_l137_137676


namespace gift_combinations_l137_137773

theorem gift_combinations :
  let wrapping_paper_varieties := 8
  let ribbon_colors := 5
  let gift_card_types := 4
  let gift_sticker_types := 6
in wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_sticker_types = 960 := by
  sorry

end gift_combinations_l137_137773


namespace sum_of_integers_l137_137715

theorem sum_of_integers (x y : ℤ) (h1 : x ^ 2 + y ^ 2 = 130) (h2 : x * y = 36) (h3 : x - y = 4) : x + y = 4 := 
by sorry

end sum_of_integers_l137_137715


namespace max_point_of_f_xe_ge_f_l137_137476

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  real.log x + 0.5 * a * x^2 + x + 1

-- Condition for the first problem
def a_neg_two : ℝ := -2

-- Prove that f(1/2) is the maximum point for a = -2
theorem max_point_of_f {x : ℝ} (h : 0 < x) : 
  ∀ y : ℝ, 0 < y → y ≠ 1/2 → f a_neg_two x ≤ f a_neg_two y := 
sorry

-- Condition for the second problem
def a_pos_two : ℝ := 2

-- Prove that xe^x ≥ f(x) for all x > 0 when a = 2
theorem xe_ge_f {x : ℝ} (h : 0 < x) : 
  x * real.exp x ≥ f a_pos_two x := 
sorry

end max_point_of_f_xe_ge_f_l137_137476


namespace find_k_l137_137650

variables (e1 e2 e3 : ℝ × ℝ) (k : ℝ)

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = 1

def orthogonal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def area_of_triangle (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.2 - v1.2 * v2.1) / 2

theorem find_k
  (h1: is_unit_vector e1)
  (h2: is_unit_vector e2)
  (h3: is_unit_vector e3)
  (h4: e3 = (1/2) • e1 + k • e2)
  (h5: area_of_triangle e1 e2 = 1/2) :
  k = (real.sqrt 3) / 2 :=
sorry

end find_k_l137_137650


namespace area_of_transformed_region_l137_137637

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![5, 3]]
def area_T : ℝ := 9

-- Theorem statement
theorem area_of_transformed_region : 
  let det_matrix := matrix.det
  (det_matrix = 9) → (area_T = 9) → (area_T * det_matrix = 81) :=
by
  intros h₁ h₂
  sorry

end area_of_transformed_region_l137_137637


namespace hockey_games_per_month_calculation_l137_137725

-- Define the given conditions
def months_in_season : Nat := 14
def total_hockey_games : Nat := 182

-- Prove the number of hockey games played each month
theorem hockey_games_per_month_calculation :
  total_hockey_games / months_in_season = 13 := by
  sorry

end hockey_games_per_month_calculation_l137_137725


namespace inlet_rate_480_l137_137029

theorem inlet_rate_480 (capacity : ℕ) (T_outlet : ℕ) (T_outlet_inlet : ℕ) (R_i : ℕ) :
  capacity = 11520 →
  T_outlet = 8 →
  T_outlet_inlet = 12 →
  R_i = 480 :=
by
  intros
  sorry

end inlet_rate_480_l137_137029


namespace total_female_officers_on_force_l137_137671

variable (P : ℕ)

def percent_60 := (3 : ℝ) / 5
def total_officers_on_duty := 360
def female_officers_on_duty := total_officers_on_duty / 2

theorem total_female_officers_on_force :
  percent_60 * (P : ℝ) = (female_officers_on_duty : ℝ) → P = 300 :=
by
  sorry

end total_female_officers_on_force_l137_137671


namespace vector_difference_magnitude_l137_137579

open Real

variables (a b : ℝ^3)
variables (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 3) (angle_ab : real_angle a b = π / 3)

theorem vector_difference_magnitude :
  ∥a - b∥ = sqrt 7 := by
  sorry

end vector_difference_magnitude_l137_137579


namespace triangle_side_ratios_l137_137958

theorem triangle_side_ratios (a b c h_a h_b h_c : ℕ) 
  (h₁ : h_a = 6) 
  (h₂ : h_b = 4) 
  (h₃ : h_c = 3) 
  (h₄ : ∀ S, 2 * S = a * h_a) 
  (h₅ : ∀ S, 2 * S = b * h_b) 
  (h₆ : ∀ S, 2 * S = c * h_c) 
  : a : b : c = 2 : 3 : 4 :=
sorry

end triangle_side_ratios_l137_137958


namespace prism_edge_coloring_l137_137848

theorem prism_edge_coloring (n : ℕ) :
  (∀ (V E : Type) (prism : Π (V : Type), Prisms V E) 
       (coloring : Π {E : Type}, E → Fin 3), 
    (∀ v : V, ∃ e1 e2 e3 : E, 
      e1 ≠ e2 ∧ e2 ≠ e3 ∧ e3 ≠ e1 ∧ 
      coloring e1 ≠ coloring e2 ∧ coloring e2 ≠ coloring e3 ∧ coloring e3 ≠ coloring e1)) 
  → (∀ f : Fin n,
      ∃ edges : Finset (Fin 3), 
      edges.card = 3 ∧ 
      ∀ e1 e2 e3 ∈ edges, 
      e1 ≠ e2 ∧ e2 ≠ e3) 
  → (∃ k : ℕ, n = 3 * k) :=
sorry

end prism_edge_coloring_l137_137848


namespace intervals_of_monotonic_increase_area_triangle_ABC_l137_137108

-- Define function f(x) as given
def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sqrt 3 * sin x * cos x

-- Question I: Prove the intervals of monotonic increase for the function f(x)
theorem intervals_of_monotonic_increase (k : ℤ) : 
  ∀ x : ℝ, f(x) is increasing on [2 * k * π - (2 * π) / 3, 2 * k * π + π / 3] :=
sorry

-- Question II: Prove the area of triangle ABC
theorem area_triangle_ABC (A : ℝ) (a b c : ℝ) 
  (h1 : f(A) = 1) 
  (h2 : a = sqrt 3)
  (h3 : b + c = 3) : 
  ∃ area : ℝ, area = (sqrt 3) / 2 :=
sorry


end intervals_of_monotonic_increase_area_triangle_ABC_l137_137108


namespace coefficient_of_term_containing_inverse_x_in_expansion_l137_137045

theorem coefficient_of_term_containing_inverse_x_in_expansion : 
  let term := (2*x^2 - (1/(3*x^3)))^7 in
  ∃ (c : ℚ), (term = c * (1/x)) → c = - (560 / 27) :=
by
  sorry

end coefficient_of_term_containing_inverse_x_in_expansion_l137_137045


namespace pies_differ_in_both_l137_137165

-- Definitions of pie types
inductive Filling
| apple
| cherry

inductive PreparationMethod
| fried
| baked

structure Pie where
  filling : Filling
  method : PreparationMethod

-- The set of all possible pie types
def pies : Set Pie := {
  {filling := Filling.apple, method := PreparationMethod.fried},
  {filling := Filling.cherry, method := PreparationMethod.fried},
  {filling := Filling.apple, method := PreparationMethod.baked},
  {filling := Filling.cherry, method := PreparationMethod.baked}
}

-- The statement to prove: If there are at least three types of pies available, then there exist two pies that differ in both filling and preparation method.
theorem pies_differ_in_both :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.method ≠ p2.method :=
begin
  sorry
end

end pies_differ_in_both_l137_137165


namespace increase_mean_maybe_median_increase_variance_l137_137561

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable {x_nplus1 : ℝ}
variable (median : ℝ)
variable (mean : ℝ)
variable (variance : ℝ)

noncomputable def newMean (x : Fin n → ℝ) (x_nplus1 : ℝ) : ℝ :=
  (∑ i, x i + x_nplus1) / (n + 1)

noncomputable def newMedian (x : Fin n → ℝ) (x_nplus1 : ℝ) : ℝ :=
  if even ((n + 1) / 2) then 
    (x ((n + 1) / 2) + x_nplus1) / 2
  else 
    x ((n + 1) / 2)

noncomputable def newVariance (x : Fin n → ℝ) (x_nplus1 : ℝ) (new_mean : ℝ) : ℝ :=
  (∑ i, (x i - new_mean) ^ 2 + (x_nplus1 - new_mean) ^ 2) / ((n:ℝ) + 1)

theorem increase_mean_maybe_median_increase_variance 
  (n_pos : n ≥ 3) (x_annual_incomes : ∀ i : Fin n, i.val < n ∧ x i > 0)
  (x_nplus1_max : ∀ i : Fin n, x_nplus1 > x i) :
  newMean x x_nplus1 > mean ∧ 
  (newMedian x x_nplus1 = median ∨ newMedian x x_nplus1 > median) ∧ 
  newVariance x x_nplus1 (newMean x x_nplus1) > variance := 
by 
  sorry

end increase_mean_maybe_median_increase_variance_l137_137561


namespace rounding_eighty_two_and_repeating_367_l137_137298

def round_to_hundredth (x : ℚ) : ℚ :=
  let hundredths := (10^2 : ℚ)
  (rat.truncate (x * hundredths) + (if x * hundredths - rat.truncate (x * hundredths) ≥ 0.5 then 1 else 0)) / hundredths

noncomputable def eighty_two_and_repeating_367 : ℚ := rat.mk 82 1 + rat.mk 367 999

theorem rounding_eighty_two_and_repeating_367 :
  round_to_hundredth eighty_two_and_repeating_367 = 82.37 :=
by
  sorry

end rounding_eighty_two_and_repeating_367_l137_137298


namespace number_of_proper_subsets_l137_137574

theorem number_of_proper_subsets {α : Type*} {P : Finset α} (hP : P = {1, 2, 3}) : 
  (P.powerset.filter (λ S, S ≠ P)).card = 7 :=
by
  sorry

end number_of_proper_subsets_l137_137574


namespace distance_M0_to_plane_l137_137757

-- Define points M0, M1, M2, M3
def M0 : ℝ × ℝ × ℝ := (-6, 7, -10)
def M1 : ℝ × ℝ × ℝ := (3, 10, -1)
def M2 : ℝ × ℝ × ℝ := (-2, 3, -5)
def M3 : ℝ × ℝ × ℝ := (-6, 0, -3)

-- Plane equation passing through M1, M2, M3
def plane_eq (x y z : ℝ) : Prop := -2 * x + 2 * y - z - 15 = 0

-- Distance formula from point to plane
def distance_to_plane (A B C D x₀ y₀ z₀ : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C * z₀ + D) / (sqrt (A^2 + B^2 + C^2))

-- Prove that the distance is 7
theorem distance_M0_to_plane : distance_to_plane (-2) 2 (-1) (-15) (-6) 7 (-10) = 7 :=
by
  sorry

end distance_M0_to_plane_l137_137757


namespace shape_of_triangle_ABC_value_of_k_l137_137851

-- Define the vectors
def BA := (-1, -2, 2 : ℝ × ℝ × ℝ)
def BC := (-2, 2, 1 : ℝ × ℝ × ℝ)
def AC := (-1, 4, -1 : ℝ × ℝ × ℝ)

-- Aliases for the conditions
def a := BA
def b := BC
def c := AC

-- Proof statement for the shape of triangle ABC
theorem shape_of_triangle_ABC (BA BC AC : ℝ × ℝ × ℝ) (hBA : BA = (-1, -2, 2))
  (hBC : BC = (-2, 2, 1)) (hAC : AC = (-1, 4, -1)) :
  ∥BA∥ = ∥BC∥ ∧ ∥AC∥ = ∥BA∥ * (Real.sqrt 2) ∧ ∥AC∥^2 = ∥BA∥^2 + ∥BC∥^2 :=
by
  sorry

-- Proof statement for the value of k
theorem value_of_k (a b c : ℝ × ℝ × ℝ) (ha : a = (-1, -2, 2)) (hb : b = (-2, 2, 1))
  (hc : c = (-1, 4, -1)) (k : ℝ) :
  (-2 * a.1 + k * b.1, -2 * a.2 + k * b.2, -2 * a.3 + k * b.3) =
  (k * c.1, k * c.2, k * c.3) → k = 2 :=
by
  sorry

end shape_of_triangle_ABC_value_of_k_l137_137851


namespace probability_same_color_l137_137317

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def green_plates : ℕ := 3
def total_plates : ℕ := 14

noncomputable def comb (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_same_color :
  let total_ways := comb total_plates 2 in
  let red_ways := comb red_plates 2 in
  let blue_ways := comb blue_plates 2 in
  let green_ways := comb green_plates 2 in
  let same_color_ways := red_ways + blue_ways + green_ways in
  (same_color_ways / total_ways) = (28 / 91) :=
by
  let total_ways := comb total_plates 2
  let red_ways := comb red_plates 2
  let blue_ways := comb blue_plates 2
  let green_ways := comb green_plates 2
  let same_color_ways := red_ways + blue_ways + green_ways
  sorry

end probability_same_color_l137_137317


namespace terminating_decimal_count_l137_137067

theorem terminating_decimal_count : 
  ∃ (count : ℕ), count = 1 ∧ 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → 
  (let d := (n * (3^2) * 7) / 2520 in
  (d.denom = 1 → n = 504)) := 
begin
  sorry
end

end terminating_decimal_count_l137_137067


namespace general_formula_a_min_value_b_l137_137861

-- Definition of sum of first n terms
def S (n : ℕ) : ℕ := n^2 + 2 * n

-- Definition of a_n in terms of the sum of the first n terms
def a : ℕ → ℕ 
| 1 => 3
| n + 1 => S (n + 1) - S n

-- General formula for the sequence a_n
theorem general_formula_a (n : ℕ) : a n = 2 * n + 1 := 
by 
  sorry

-- Definition of b_n
def b (n : ℕ) : ℕ := 3 * n^2 - 4 * a (2 * n - 1)

-- Minimum value of the sequence b_n
theorem min_value_b : ∃ n : ℕ, b n = -17 :=
by 
  sorry

end general_formula_a_min_value_b_l137_137861


namespace max_k_l137_137249

/-- Define the set S and conditions on subsets A_i -/
def S := {i | 1 ≤ i ∧ i ≤ 10}

-- Define the property of subsets
def isValidSubset (A : set ℕ) : Prop := 
  A ⊆ S ∧ (A.card = 5)

-- Define the condition of pairwise intersections 
def validIntersections (A : set (set ℕ)) : Prop := 
  ∀ (A_i A_j : set ℕ), A_i ∈ A ∧ A_j ∈ A ∧ A_i ≠ A_j → (A_i ∩ A_j).card ≤ 2

-- Prove that the maximum value of k is 6
theorem max_k (k : ℕ) (A : set (set ℕ) ) : 
  (∀ A_i ∈ A, isValidSubset A_i) ∧
  validIntersections A →
  k = A.card →
  k ≤ 6 := 
sorry

end max_k_l137_137249


namespace residue_of_T_mod_2012_l137_137250

def T : ℤ := ∑ i in Finset.range 2012, if i % 2 = 0 then -↑i else ↑i

theorem residue_of_T_mod_2012 : T % 2012 = 1006 := by
  sorry

end residue_of_T_mod_2012_l137_137250


namespace construct_vertex_A_l137_137811

-- Definitions of given conditions
variables {B C O : Point} {MC : ℝ}

-- Definition of the orthocenter
def is_orthocenter (M : Point) (B C : Point) : Prop :=
  ∃ (A : Point), is_triangle A B C ∧ M = orthocenter A B C

-- The main theorem statement
theorem construct_vertex_A (B C O : Point) (MC : ℝ) :
  ∃ A : Point, is_triangle A B C ∧ dist (orthocenter A B C) C = MC ∧
  is_on_circumcircle A B C ∧ circumcenter A B C = O :=
sorry

end construct_vertex_A_l137_137811


namespace simplify_polynomial_l137_137310

theorem simplify_polynomial :
  (3 * x ^ 5 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 6) + (7 * x ^ 4 + x ^ 3 - 3 * x ^ 2 + x - 9) =
  3 * x ^ 5 + 7 * x ^ 4 - x ^ 3 + 2 * x ^ 2 - 7 * x - 3 :=
by
  sorry

end simplify_polynomial_l137_137310


namespace isodynamic_centers_inversion_l137_137677

noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def center (S : Circle) : Point := sorry
noncomputable def isodynamic_circle (A B C : Point) : Circle := sorry
noncomputable def inversion (P : Point) (S : Circle) : Point := sorry
noncomputable def is_isodynamic_center (P : Point) (A B C : Point) : Prop := sorry

theorem isodynamic_centers_inversion
  (A B C : Point)
  (S : Circle := circumcircle A B C)
  (O : Point := center S)
  (Sₐ : Circle := isodynamic_circle A B C)
  (Iₛ Iₛ' : Point)
  (h₁ : is_isodynamic_center Iₛ A B C)
  (h₂ : is_isodynamic_center Iₛ' A B C)
  (h₃ : ∀ (P : Point), P ∈ Sₐ → P ∈ S → inversion P S = P)
  (h₄ : ∀ (P : Point), P ∈ Sₐ → (P = inversion P S))
  : inversion Iₛ S = Iₛ' ∧ inversion Iₛ' S = Iₛ :=
sorry

end isodynamic_centers_inversion_l137_137677


namespace eval_power_81_11_over_4_l137_137054

theorem eval_power_81_11_over_4 : 81^(11/4) = 177147 := by
  sorry

end eval_power_81_11_over_4_l137_137054


namespace math_problem_l137_137488

theorem math_problem : 1537 + (180 / 60) * 15 - 237 = 1345 := by
  -- calculation steps
  have h1: 180 / 60 = 3 := by norm_num
  have h2: 3 * 15 = 45 := by norm_num
  have h3: 1537 + 45 = 1582 := by norm_num
  have h4: 1582 - 237 = 1345 := by norm_num
  rw [h1, h2, h3, h4]
  norm_num

end math_problem_l137_137488


namespace average_time_to_find_Waldo_l137_137724

theorem average_time_to_find_Waldo :
  let num_books := 15 in
  let puzzles_per_book := 30 in
  let total_time := 1350 in
  (total_time / (num_books * puzzles_per_book) = 3) :=
by
  let num_books := 15
  let puzzles_per_book := 30
  let total_time := 1350
  let average_time := total_time / (num_books * puzzles_per_book)
  show average_time = 3
  sorry

end average_time_to_find_Waldo_l137_137724


namespace two_pies_differ_l137_137178

theorem two_pies_differ (F_A F_C B_A B_C : Bool) :
  (F_A ∨ F_C ∨ B_A ∨ B_C) →
  (F_A ∧ F_C ∧ B_A ∧ B_C) ∧ 
  (∀ a b, (a ≠ b) → (a.filling ≠ b.filling ∧ a.preparation ≠ b.preparation)) :=
by
  intros H1 H2
  sorry

end two_pies_differ_l137_137178


namespace area_perimeter_quadratic_l137_137338

theorem area_perimeter_quadratic (a x y : ℝ) (h1 : x = 4 * a) (h2 : y = a^2) : y = (x / 4)^2 :=
by sorry

end area_perimeter_quadratic_l137_137338


namespace find_angle_4_l137_137084

/-- Given angle conditions, prove that angle 4 is 22.5 degrees. -/
theorem find_angle_4 (angle : ℕ → ℝ) 
  (h1 : angle 1 + angle 2 = 180) 
  (h2 : angle 3 = angle 4) 
  (h3 : angle 1 = 85) 
  (h4 : angle 5 = 45) 
  (h5 : angle 1 + angle 5 + angle 6 = 180) : 
  angle 4 = 22.5 :=
sorry

end find_angle_4_l137_137084


namespace inverse_proportion_function_has_m_value_l137_137341

theorem inverse_proportion_function_has_m_value
  (k : ℝ)
  (h1 : 2 * -3 = k)
  {m : ℝ}
  (h2 : 6 = k / m) :
  m = -1 :=
by
  sorry

end inverse_proportion_function_has_m_value_l137_137341


namespace sqrt_multiplication_simplified_l137_137467

noncomputable def problem_statement (x : ℝ) : ℝ :=
  real.sqrt (48 * x) * real.sqrt (3 * x) * real.sqrt (50 * x)

theorem sqrt_multiplication_simplified (x : ℝ) (h : 0 ≤ x) :
  problem_statement x = 60 * x * real.sqrt (2 * x) :=
by
  sorry

end sqrt_multiplication_simplified_l137_137467


namespace fill_time_with_leak_is_correct_l137_137421

-- Define the conditions
def time_to_fill_without_leak := 8
def time_to_empty_with_leak := 24

-- Define the rates
def fill_rate := 1 / time_to_fill_without_leak
def leak_rate := 1 / time_to_empty_with_leak
def effective_fill_rate := fill_rate - leak_rate

-- Prove the time to fill with leak
def time_to_fill_with_leak := 1 / effective_fill_rate

-- The theorem to prove that the time is 12 hours
theorem fill_time_with_leak_is_correct :
  time_to_fill_with_leak = 12 := by
  simp [time_to_fill_without_leak, time_to_empty_with_leak, fill_rate, leak_rate, effective_fill_rate, time_to_fill_with_leak]
  sorry

end fill_time_with_leak_is_correct_l137_137421


namespace sum_of_fractions_l137_137687

theorem sum_of_fractions (p q : ℤ) (hq : q ≠ 0) (gcd_pq : Int.gcd p q = 1) :
  ∃ (n : ℕ) (m : ℕ) (ns : Fin m → ℕ), p * (n + 2) = q * (∑ i, ((ns i) - 1)) := 
sorry

end sum_of_fractions_l137_137687


namespace radical_product_l137_137733

theorem radical_product :
  (64 ^ (1 / 3) * 16 ^ (1 / 4) * 64 ^ (1 / 6) = 16) :=
by
  sorry

end radical_product_l137_137733


namespace domain_of_composite_function_l137_137598

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → -1 ≤ x + 1) →
  (∀ x, -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3 → -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3 → 0 ≤ x ∧ x ≤ 1) :=
by
  sorry

end domain_of_composite_function_l137_137598


namespace segment_AV_length_l137_137000

theorem segment_AV_length
  (U : Type)
  (C : ℝ)
  (r : ℝ)
  (UA UV : ℝ)
  (UAV_angle : ℝ)
  (hC : C = 18 * π)
  (hr : 2 * π * r = C)
  (hUA : UA = r)
  (hUV : UV = r)
  (hUAV_angle : UAV_angle = 45 * π / 180) :
  ∃ AV : ℝ, AV = 9 * sqrt (2 - sqrt 2) :=
by
  have r_eq : r = 9 := by linarith, -- Using the circumference to find radius
  use 9 * sqrt (2 - sqrt 2), sorry

end segment_AV_length_l137_137000


namespace three_digit_numbers_with_789_l137_137585

theorem three_digit_numbers_with_789 : 
  ∃ n : ℕ, n = 606 ∧ ∀ x (h : 100 ≤ x ∧ x ≤ 999), 
  (¬ (∃ (a b c : ℕ), x = 100*a + 10*b + c ∧ 1 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 6 ∧ 0 ≤ c ∧ c ≤ 6) → x) -> x :=
proof
  sorry

end three_digit_numbers_with_789_l137_137585


namespace original_price_lamp_l137_137007

theorem original_price_lamp
  (P : ℝ)
  (discount_rate : ℝ)
  (discounted_price : ℝ)
  (discount_is_20_perc : discount_rate = 0.20)
  (new_price_is_96 : discounted_price = 96)
  (price_after_discount : discounted_price = P * (1 - discount_rate)) :
  P = 120 :=
by
  sorry

end original_price_lamp_l137_137007


namespace distance_between_lines_AE_BF_l137_137623

noncomputable def distance_between_lines (AB AD AA1 : ℝ) (E F : ℝ × ℝ × ℝ) : ℝ :=
  12

/-
Given:
- AB = 18
- AD = 36
- AA1 = 9
- E is the midpoint of A1B1
- F is the midpoint of B1C1

Prove:
The distance between lines AE and BF is 12.
-/
theorem distance_between_lines_AE_BF : 
  ∀ (A B D A1 B1 C C1 E F : ℝ × ℝ × ℝ),
    A = (0, 0, 0) ∧ 
    B = (18, 0, 0) ∧ 
    D = (0, 36, 0) ∧ 
    A1 = (0, 0, 9) ∧ 
    B1 = (18, 0, 9) ∧ 
    C = (18, 36, 0) ∧ 
    C1 = (18, 36, 9) ∧ 
    E = (9, 0, 9) ∧ 
    F = (18, 18, 9) →
    distance_between_lines 18 36 9 (9, 0, 9) (18, 18, 9) = 12 :=
by
  intros A B D A1 B1 C C1 E F h
  apply distance_between_lines
  sorry -- To be proved

end distance_between_lines_AE_BF_l137_137623


namespace radius_of_sphere_l137_137434

theorem radius_of_sphere (PA PB PC : ℝ) (h1 : PA = 2 * real.sqrt 2) (h2 : PB = 2 * real.sqrt 2) (h3 : PC = 3) : 
    let space_diagonal := real.sqrt (PA^2 + PB^2 + PC^2)
    let radius := space_diagonal / 2
    radius = 5 / 2 := 
by {
    sorry
}

end radius_of_sphere_l137_137434


namespace weibull_distribution_double_exponential_distribution_integer_fractional_parts_independent_integer_part_distribution_fractional_part_distribution_l137_137756

open MeasureTheory ProbabilityTheory

variables {λ : ℝ} (hλ : λ > 0) {α : ℝ} (hα : α > 0)

noncomputable def exp_pdf (λ : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else λ * exp (-λ * x)

noncomputable def weibull_density (λ : ℝ) (α : ℝ) (y : ℝ) : ℝ :=
  λ * α * y^(α - 1) * exp (-(λ * y^α))

noncomputable def double_exp_density (λ : ℝ) (y : ℝ) : ℝ :=
  λ * exp (y - λ * exp y)

theorem weibull_distribution :
  ∀ y > 0, PDF (λ X, X ^ (1 / α)) (weibull_density λ α) :=
sorry

theorem double_exponential_distribution :
  ∀ y : ℝ, PDF (λ X, log X) (double_exp_density λ) :=
sorry

theorem integer_fractional_parts_independent :
  ∀ (n : ℕ) α ∈ [0, 1), independent (λ X, ⌊X⌋) (λ X, X - ⌊X⌋) :=
sorry

theorem integer_part_distribution :
  ∀ n : ℕ, P (λ X, ⌊X⌋ = n) = exp (-λ * n) * (1 - exp (-λ)) :=
sorry

theorem fractional_part_distribution :
  ∀ α ∈ [0, 1), P (λ X, fract X ≤ α) = (1 - exp (-λ * α)) / (1 - exp (-λ)) :=
sorry

end weibull_distribution_double_exponential_distribution_integer_fractional_parts_independent_integer_part_distribution_fractional_part_distribution_l137_137756


namespace sum_of_exterior_angles_of_octagon_l137_137714

theorem sum_of_exterior_angles_of_octagon : 
  ∀ (n : ℕ), n = 8 → sum_exterior_angles n = 360 :=
by
  intro n
  intro h_n
  rw [h_n]
  -- The proof steps would go here
  sorry

end sum_of_exterior_angles_of_octagon_l137_137714


namespace parabola_properties_l137_137524

theorem parabola_properties :
  ∀ (x y : ℝ), (y = 3 * x^2 + 6 * x + 5) →
    ((-1, 2) = (-((6:ℝ) / (2 * 3)), 3 * (-1:ℝ)^2 + 6 * (-1) + 5) ∧
     (x = -1) ∧
     (y = 5)) :=
by
  intros x y h
  split
  sorry
  sorry
  sorry

end parabola_properties_l137_137524


namespace find_width_of_room_l137_137835

theorem find_width_of_room (length room_cost cost_per_sqm total_cost width W : ℕ) 
  (h1 : length = 13)
  (h2 : cost_per_sqm = 12)
  (h3 : total_cost = 1872)
  (h4 : room_cost = length * W * cost_per_sqm)
  (h5 : total_cost = room_cost) : 
  W = 12 := 
by sorry

end find_width_of_room_l137_137835


namespace can_buy_two_pies_different_in_both_l137_137175

structure Pie :=
  (filling : Type)
  (preparation : Type)

def apple : Type := unit
def cherry : Type := unit
def fried : Type := unit
def baked : Type := unit

def apple_fried : Pie := { filling := apple, preparation := fried }
def apple_baked : Pie := { filling := apple, preparation := baked }
def cherry_fried : Pie := { filling := cherry, preparation := fried }
def cherry_baked : Pie := { filling := cherry, preparation := baked }

def possible_pies : List Pie := [apple_fried, apple_baked, cherry_fried, cherry_baked]

theorem can_buy_two_pies_different_in_both 
  (available_pies : List Pie) 
  (h : available_pies.length ≥ 3) : 
  ∃ (p1 p2 : Pie), p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation := 
  sorry

end can_buy_two_pies_different_in_both_l137_137175


namespace number_of_books_in_library_l137_137368

def number_of_bookcases : ℕ := 28
def shelves_per_bookcase : ℕ := 6
def books_per_shelf : ℕ := 19

theorem number_of_books_in_library : number_of_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end number_of_books_in_library_l137_137368


namespace fix_multiplication_l137_137914

theorem fix_multiplication mistake (x : ℝ) (hx : x * 8 = 56) :
  (x / 8).round = 0.88 :=
by
  sorry

end fix_multiplication_l137_137914


namespace tangent_line_at_1_l137_137339

-- Define the function f
def f (x : ℝ) := x - 2 * Real.log x

-- Define the tangent line equation
def tangent_line_eq (x y : ℝ) := x + y - 2 = 0

-- The main statement: prove that the tangent line to f at (1, f(1)) is given by x + y - 2 = 0
theorem tangent_line_at_1 : tangent_line_eq 1 (f 1) :=
by
  sorry

end tangent_line_at_1_l137_137339


namespace petya_friends_count_l137_137672

-- Define the number of classmates
def total_classmates : ℕ := 28

-- Each classmate has a unique number of friends from 0 to 27
def unique_friends (n : ℕ) : Prop :=
  n ≥ 0 ∧ n < total_classmates

-- We state the problem where Petya's number of friends is to be proven as 14
theorem petya_friends_count (friends : ℕ) (h : unique_friends friends) : friends = 14 :=
sorry

end petya_friends_count_l137_137672


namespace compare_neg_two_and_neg_one_l137_137806

theorem compare_neg_two_and_neg_one : -2 < -1 :=
by {
  -- Proof is omitted
  sorry
}

end compare_neg_two_and_neg_one_l137_137806


namespace triangle_distance_BD_l137_137215

theorem triangle_distance_BD
  (A B C D E : Type)
  (h_triangle_ABC : ∠C = 90 ∨ AC = 9 ∨ BC = 12)
  (h_points_on_lines : (D ∈ line(A, B) ∧ E ∈ line(B, C)))
  (h_angle_BED_angle : ∠BED = 90)
  (h_DE : DE = 6) :
  BD = 10 := 
sorry

end triangle_distance_BD_l137_137215


namespace max_degree_vertex_triangle_l137_137731

theorem max_degree_vertex_triangle 
  (n : ℕ)
  (points : finset (fin n))
  (segments : finset (fin n × fin n))
  (h1 : segments.card = nat.floor ((n^2 : ℚ) / 4))
  (h2 : ∃ (a b c: fin n), (a, b) ∈ segments ∧ (b, c) ∈ segments ∧ (c, a) ∈ segments) :
  ∃ v : fin n, 
    (∀ u : fin n, ((u, v) ∈ segments ∨ (v, u) ∈ segments) → sorry) →
    ∀ a b : fin n, (a, b) ∈ segments → (b, v) ∈ segments →
    (∃ c : fin n, (v, a) ∈ segments ∧ (a, c) ∈ segments ∧ (c, v) ∈ segments) :=
begin
  sorry
end

end max_degree_vertex_triangle_l137_137731


namespace appointment_duration_l137_137456

-- Define the given conditions
def total_workday_hours : ℕ := 8
def permits_per_hour : ℕ := 50
def total_permits : ℕ := 100
def stamping_time : ℕ := total_permits / permits_per_hour
def appointment_time : ℕ := (total_workday_hours - stamping_time) / 2

-- State the theorem and ignore the proof part by adding sorry
theorem appointment_duration : appointment_time = 3 := by
  -- skipping the proof steps
  sorry

end appointment_duration_l137_137456


namespace projection_correct_l137_137511

noncomputable def projection (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_vw := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let dot_ww := w.1 * w.1 + w.2 * w.2 + w.3 * w.3
  let scalar := dot_vw / dot_ww
  (scalar * w.1, scalar * w.2, scalar * w.3)

theorem projection_correct :
  projection (4, -1, 3) (3, 1, 2) = (51 / 14, 17 / 14, 17 / 7) := 
by
  -- The proof would go here.
  sorry

end projection_correct_l137_137511


namespace regression_analysis_correct_l137_137200

-- Definition of the regression analysis context
def regression_analysis_variation (forecast_var : Type) (explanatory_var residual_var : Type) : Prop :=
  forecast_var = explanatory_var ∧ forecast_var = residual_var

-- The theorem to prove
theorem regression_analysis_correct :
  ∀ (forecast_var explanatory_var residual_var : Type),
  regression_analysis_variation forecast_var explanatory_var residual_var →
  (forecast_var = explanatory_var ∧ forecast_var = residual_var) :=
by
  intro forecast_var explanatory_var residual_var h
  exact h

end regression_analysis_correct_l137_137200


namespace chords_of_tangents_equal_l137_137532

-- Define the centers and radii of two circles
variables (O_1 O_2 : Type) 
variables (r R : ℝ)

-- Define the property of chords connecting the tangents to be equal
theorem chords_of_tangents_equal 
  (c1 c2 : Type) -- circles
  (O_1_center : O_1) 
  (O_2_center : O_2) 
  (r_circle1 : r) 
  (R_circle2 : R) 
  (AB CD : ℝ) -- chords 
  (tangent_to_c2 : ∀ P : O_1, P ∉ c2) -- tangent from O_1 to c2
  (tangent_to_c1 : ∀ Q : O_2, Q ∉ c1) -- tangent from O_2 to c1
  (AB_eq : AB = 2 * (r * R / dist O_1_center O_2_center)) 
  (CD_eq : CD = 2 * (r * R / dist O_1_center O_2_center))
  : AB = CD := 
by 
  have eq : 2 * (r * R / dist O_1_center O_2_center) = 2 * (r * R / dist O_1_center O_2_center),
  from rfl,
  sorry

end chords_of_tangents_equal_l137_137532


namespace cos_C_area_triangle_l137_137156

noncomputable def triangle_properties (A B C a b c : ℝ) : Prop :=
  B = real.pi / 3 ∧
  (a - b + c) * (a + b - c) = (3 / 7) * b * c

theorem cos_C {A B C a b c : ℝ} (h : triangle_properties A B C a b c) : 
  ∃ C, cos C = 1 / 7 := 
by
  sorry

theorem area_triangle {A B C a b c : ℝ} (h : triangle_properties A B C a b c) (ha : a = 5) : 
  ∃ area, area = 10 * real.sqrt 3 := 
by
  sorry

end cos_C_area_triangle_l137_137156


namespace maximum_number_of_rectangles_is_12_l137_137373

-- Define the side length of the square
def side_length : ℝ := 14

-- Define the dimensions of the rectangle
def rectangle_width : ℝ := 2
def rectangle_length : ℝ := 8

-- Define the areas of the square and the rectangle
def area_square : ℝ := side_length * side_length
def area_rectangle : ℝ := rectangle_width * rectangle_length

-- Define the maximum number of rectangles that can fit in the square
def max_rectangles : ℝ := area_square / area_rectangle

-- Prove that the maximum number of rectangles is 12 (since we can't have a fraction of a rectangle)
theorem maximum_number_of_rectangles_is_12 : max_rectangles.floor = 12 := by
  calc
    max_rectangles = 196 / 16 := by sorry
    _ = 12.25 := by sorry
    _ = 12 := by sorry

end maximum_number_of_rectangles_is_12_l137_137373


namespace present_value_l137_137586

/-
Given:
- Annual interest rate (r) = 0.07
- Future Value (FV) = 600,000 (in dollars)
- Time period (n) = 15 (years)

Prove that the Present Value (PV) needed is approximately $217,474.41.
-/
noncomputable def future_value : ℝ := 600000
noncomputable def rate : ℝ := 0.07
noncomputable def years : ℕ := 15

/--
  Calculate PV using the formula:
  PV = FV / (1 + r)^n
-/
theorem present_value :
  let PV := future_value / (1 + rate)^years 
  in abs (PV - 217474.41) < 0.005 := 
by
  sorry

end present_value_l137_137586


namespace correct_statements_in_triangle_l137_137157

noncomputable def triangle_conditions_1 (A B C a b c : ℝ) : Prop :=
  a * cosine B = b * sine A → B = π / 4

noncomputable def triangle_conditions_2 (B : ℝ) (b a : ℝ) : Prop :=
  B = π / 4 → b = 2 → a = sqrt 3 → ∃ (num_triangle : ℕ), num_triangle = 2

noncomputable def triangle_conditions_3 (A B C a b c : ℝ) : Prop :=
  (2 * b = a + c) → (sine B ^ 2 = sine A * sine C) → (b ^ 2 = a * c) → a = b ∧ b = c

noncomputable def triangle_conditions_4 (B : ℝ) : Prop :=
  (a = 5) → (c = 2) → (4 * sine B = 4) → cos B = 3 / 5

theorem correct_statements_in_triangle (A B C a b c : ℝ) :
  triangle_conditions_1 A B C a b c ∧
  ¬ triangle_conditions_2 B b a ∧
  triangle_conditions_3 A B C a b c ∧
  ¬ triangle_conditions_4 B :=
sorry

end correct_statements_in_triangle_l137_137157


namespace apples_given_to_Larry_l137_137968

-- Define the initial conditions
def initial_apples : ℕ := 75
def remaining_apples : ℕ := 23

-- The statement that we need to prove
theorem apples_given_to_Larry : initial_apples - remaining_apples = 52 :=
by
  -- skip the proof
  sorry

end apples_given_to_Larry_l137_137968


namespace Janet_earnings_l137_137226

/--
Janet works as an exterminator and also sells molten metal casts of fire ant nests on the Internet.
She gets paid $70 an hour for exterminator work and makes $20/pound on her ant nest sculptures.
Given that she does 20 hours of exterminator work and sells a 5-pound sculpture and a 7-pound sculpture,
prove that Janet's total earnings are $1640.
-/
theorem Janet_earnings :
  let hourly_rate_exterminator := 70
  let hours_worked := 20
  let rate_per_pound := 20
  let sculpture_one_weight := 5
  let sculpture_two_weight := 7

  let exterminator_earnings := hourly_rate_exterminator * hours_worked
  let total_sculpture_weight := sculpture_one_weight + sculpture_two_weight
  let sculpture_earnings := rate_per_pound * total_sculpture_weight

  let total_earnings := exterminator_earnings + sculpture_earnings
  total_earnings = 1640 := 
by
  sorry

end Janet_earnings_l137_137226


namespace asymptoticSolution_l137_137960

noncomputable def solutionAsymptoticBehavior : Prop :=
  ∀ (x : ℝ), x ≠ 0 → 
    (∃ (y : ℝ), x^5 + x^2 * y^2 = y^6 ∧ 
      tendsto (fun x => y) (nhds 0) (nhds 0) ∧ 
      tendsto (fun x => y / x^(1/2)) (nhds 0) (nhds (0)))

theorem asymptoticSolution :
  ∀ (x : ℝ), solutionAsymptoticBehavior :=
by
  intro x
  sorry

end asymptoticSolution_l137_137960


namespace mm_perpendicular_am_l137_137446

variables {α : Type*} [EuclideanGeometry α]

noncomputable def is_midpoint (M X Y : α) :=
  2 • M = X + Y

-- Define the problem/conditions as hypotheses in Lean
variables (A B C D X Y M M' : α)
variable [euclidean_geometry α]

-- Hypotheses/conditions based on the problem statements
hypothesis h1 : triangle A B C
hypothesis h2 : altitude D A B C
hypothesis h3 : on_circle X A B D
hypothesis h4 : on_circle Y A C D
hypothesis h5 : collinear X D Y
hypothesis h6 : is_midpoint M X Y
hypothesis h7 : is_midpoint M' B C

-- The goal/theorem to prove
theorem mm_perpendicular_am : ∀ {A B C D X Y M M' : α}, 
  triangle A B C →
  altitude D A B C →
  on_circle X A B D →
  on_circle Y A C D →
  collinear X D Y →
  is_midpoint M X Y →
  is_midpoint M' B C →
  perpendicular (line_through M M') (line_through A M) :=
sorry

end mm_perpendicular_am_l137_137446


namespace collinear_G_E_N_l137_137461

-- Define the mathematical structures and conditions
structure Circle (α : Type*) := 
(center : α) 
(radius : ℝ)

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

structure Point (α : Type*) := 
(x y : α)

structure Line (α : Type*) :=
(point1 point2 : Point α)

noncomputable def tangent (c : Circle α) (p : Point α) : Prop :=
dist c.center p.x = c.radius

variables (O1 O2 : Circle α) (A B C D F E G M N : Point α) (l : Line α) (AF : Line α)

-- Conditions
axiom cond1 : tangent O2 A
axiom cond2 : tangent O1 A
axiom cond3 : l.point1 = G ∧ ∃ p : Point α, p ≠ G ∧ l.point2 = p ∧ dist p G = dist B G
axiom cond4 : F ∈ O1 ∧ E ∈ O2
axiom cond5 : M ∈ O1 ∧ dist F G = dist O1.center M
axiom cond6 : N ∈ O2 ∧ dist M N = dist O2.center A
axiom parallel_AF_l : dist AF.point1 AF.point2 = dist l.point1 l.point2

-- Theorem: Prove collinearity of points G, E, and N
theorem collinear_G_E_N : collinear α {G, E, N} :=
sorry

end collinear_G_E_N_l137_137461


namespace max_product_abc_l137_137556

noncomputable def maximum_product_abc (a b c : ℕ) : ℕ :=
  if h : a > 0 ∧ b > 0 ∧ c > 0 ∧
         (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ r1 + r2 = 2 * a ∧ r1 * r2 = b) ∧
         (∃ s1 s2 : ℕ, s1 > 0 ∧ s2 > 0 ∧ s1 + s2 = 2 * b ∧ s1 * s2 = c) ∧
         (∃ t1 t2 : ℕ, t1 > 0 ∧ t2 > 0 ∧ t1 + t2 = 2 * c ∧ t1 * t2 = a) 
  then a * b * c 
  else 0

theorem max_product_abc : maximum_product_abc a b c = 1 :=
by
  sorry

end max_product_abc_l137_137556


namespace magnitude_z_minus_2_l137_137871

theorem magnitude_z_minus_2 (i : ℂ) (hi : i * i = -1) (z : ℂ) (hz : z = 2 * i / (1 + i)) : 
  |z - 2| = Real.sqrt 2 :=
sorry

end magnitude_z_minus_2_l137_137871


namespace num_factors_gt_3_of_2550_l137_137584

noncomputable def num_divisors (n : ℕ) : ℕ :=
(n.factors.group_by id).vals.map (fun x => x.length + 1).prod

def num_factors_gt_3 (n : ℕ) : ℕ :=
(n.divisors.filter (λ d, num_divisors d > 3)).length

theorem num_factors_gt_3_of_2550 : num_factors_gt_3 2550 = 9 := by
  sorry

end num_factors_gt_3_of_2550_l137_137584


namespace add_base8_l137_137787

-- Define the base 8 numbers 5_8 and 16_8
def five_base8 : ℕ := 5
def sixteen_base8 : ℕ := 1 * 8 + 6

-- Convert the result to base 8 from the sum in base 10
def sum_base8 (a b : ℕ) : ℕ :=
  let sum_base10 := a + b
  let d1 := sum_base10 / 8
  let d0 := sum_base10 % 8
  d1 * 10 + d0 

theorem add_base8 (x y : ℕ) (hx : x = five_base8) (hy : y = sixteen_base8) :
  sum_base8 x y = 23 :=
by
  sorry

end add_base8_l137_137787


namespace general_term_a_n_sum_of_b_n_l137_137116

def sequence_a (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ 
  (∀ n, a (n + 1) = 2 * S n + 2) ∧ 
  (∀ n, S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1))

def geometric_mean (b : ℕ → ℝ) (a : ℕ → ℕ) : Prop :=
  ∀ n, b n = real.sqrt ((n / (a n).to_real) * (n / (a (n + 2)).to_real))

def sum_T (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, T n = ∑ i in finset.range (n + 1), b i

theorem general_term_a_n (a S : ℕ → ℕ) (n : ℕ) :
  sequence_a a S → a n = 2 * 3^(n - 1) :=
begin
  intro h,
  sorry
end

theorem sum_of_b_n (b : ℕ → ℝ) (a : ℕ → ℕ) (T : ℕ → ℝ) (n : ℕ) :
  (∀ n, a n = 2 * 3^(n - 1)) →
  geometric_mean b a → sum_T T b →
  T n = (3 / 8) * (1 - 1 / 3^n) - (n / (4 * 3^n)) :=
begin
  intros h1 h2 h3,
  sorry
end

end general_term_a_n_sum_of_b_n_l137_137116


namespace year3023_is_gui_wei_l137_137360

-- Define heavenly stems and earthly branches
def heavenly_stems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthly_branches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Define the problem data
def year2023_heavenly_stem := 10 -- Gui
def year2023_earthly_branch := 4 -- Mao

-- Function to find the pair given a start year configuration, difference and cycles
def sexagenary_cycle_year (diff : Nat) : (String × String) :=
  let hs_index := (year2023_heavenly_stem - 1 + diff) % heavenly_stems.length
  let eb_index := (year2023_earthly_branch - 1 + diff) % earthly_branches.length
  (heavenly_stems[hs_index], earthly_branches[eb_index])

-- Statement to prove that year 3023 is "Gui Wei"
theorem year3023_is_gui_wei : sexagenary_cycle_year 1000 = ("Gui", "Wei") :=
  by sorry

end year3023_is_gui_wei_l137_137360


namespace similarity_coefficient_l137_137018

theorem similarity_coefficient (α : ℝ) :
  (2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)))
  = 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end similarity_coefficient_l137_137018


namespace min_pool_cost_is_5400_l137_137809

noncomputable def minimum_pool_cost : ℝ :=
  let V := 18
  let d := 2
  let bottom_cost := 200
  let wall_cost := 150
  let xy := V / d in
  let cost (x y : ℝ) := bottom_cost * xy + wall_cost * 4 * (x + y) in
  let am_gm_constraint := real.sqrt xy in
  if xy > 0 then
    cost (real.sqrt xy) (real.sqrt xy)
  else 0

theorem min_pool_cost_is_5400 : minimum_pool_cost = 5400 := by
  sorry

end min_pool_cost_is_5400_l137_137809


namespace sequence_prime_divisor_l137_137231

noncomputable def sequence (n : ℕ) : ℕ → ℕ
| 1       => n - 1
| (k + 1) => n^(n^k) - 1

theorem sequence_prime_divisor (n : ℕ) (h : n > 1) (k : ℕ) :
  ∃ p : ℕ, p.prime ∧ (p ∣ sequence n (k + 1)) ∧ (∀ j < k + 1, ¬ (p ∣ sequence n j)) := 
sorry

end sequence_prime_divisor_l137_137231


namespace sin_double_angle_l137_137917

variable (θ : Real) 

-- Specify the given condition
def tan_condition : Prop := tan θ + 1 / tan θ = 4

-- State the theorem to be proven
theorem sin_double_angle (h : tan_condition θ) : sin (2 * θ) = 1 / 2 :=
sorry

end sin_double_angle_l137_137917


namespace problem_1_problem_2_l137_137881

noncomputable def f (a x : ℝ) : ℝ := a * real.log x + 2 * x

theorem problem_1 (h : -4 ≠ 0) : has_extremum (f (-4)) (local_min := 4 - 4 * real.log 2) :=
by sorry

theorem problem_2 (h : ∀(x : ℝ), f a x + a ≥ 0) : -2 ≤ a ∧ a < 0 :=
by sorry

end problem_1_problem_2_l137_137881


namespace curveC1_general_equation_curveC2_cartesian_equation_minimum_distance_l137_137949

noncomputable def curveC1GeneralEquation (x y : ℝ) (α : ℝ) : Prop :=
  (x = sqrt 3 * Real.cos α) ∧ (y = Real.sin α)

noncomputable def curveC2PolarEquation (ρ θ : ℝ) : Prop :=
  (ρ * Real.sin (θ - Real.pi / 4) = 2 * sqrt 2)

theorem curveC1_general_equation (x y : ℝ) (h : ∃ α, curveC1GeneralEquation x y α) :
  x^2 / 3 + y^2 = 1 :=
sorry

theorem curveC2_cartesian_equation (x y : ℝ) (h : ∃ ρ θ, ρ = sqrt (x^2 + y^2) ∧ 
  θ = Real.arctan2 y x ∧ curveC2PolarEquation ρ θ) :
  x - y + 4 = 0 :=
sorry

theorem minimum_distance (x y : ℝ) (h1 : ∃ α, curveC1GeneralEquation x y α) (a b : ℝ)
  (h2 : ∃ ρ θ, ρ = sqrt (a^2 + b^2) ∧ θ = Real.arctan2 b a ∧ curveC2PolarEquation ρ θ) :
  ∃ d, d = sqrt 2 ∧ ∀ p : ℝ × ℝ, p = (x, y) → ∀ q : ℝ × ℝ, q = (a, b) → 
  dist p q = d :=
sorry

end curveC1_general_equation_curveC2_cartesian_equation_minimum_distance_l137_137949


namespace prob_not_equal_genders_l137_137273

noncomputable def probability_more_grandsons_or_granddaughters : ℚ :=
1 - ((Nat.choose 12 6 : ℚ) / (2^12))

theorem prob_not_equal_genders {n : ℕ} (hn : n = 12)
  (equal_prob : ∀ i, i < n → i ≥ 0 → (Nat.of_num 1 / Nat.of_num 2 : ℚ) = 1 / 2) :
  probability_more_grandsons_or_granddaughters = 793 / 1024 := by
  have hn_pos : 0 < n := by linarith
  have hij : _hid := sorry
  sorry

end prob_not_equal_genders_l137_137273


namespace number_of_integers_for_perfect_square_ratio_l137_137068

theorem number_of_integers_for_perfect_square_ratio :
  {n : ℤ | ∃ k : ℤ, n / (30 - n) = k ^ 2}.finite.to_finset.card = 3 := 
sorry

end number_of_integers_for_perfect_square_ratio_l137_137068


namespace max_value_of_trig_expression_l137_137501

theorem max_value_of_trig_expression (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_value_of_trig_expression_l137_137501


namespace f_at_4_l137_137931

def f_inverse (x : ℝ) (hx : x > 0) : ℝ :=
  x^2

theorem f_at_4 (h : ∃ f : ℝ → ℝ, ∀ x, x > 0 → f_inverse (f x) (f x > 0) = x) : 
  f (4 : ℝ) = 2 := 
sorry

end f_at_4_l137_137931


namespace express_114_as_ones_and_threes_with_min_ten_ones_l137_137135

theorem express_114_as_ones_and_threes_with_min_ten_ones :
  ∃n: ℕ, n = 35 ∧ ∃ x y : ℕ, x + 3 * y = 114 ∧ x ≥ 10 := sorry

end express_114_as_ones_and_threes_with_min_ten_ones_l137_137135


namespace num_boys_in_class_l137_137790

-- Definitions based on conditions
def num_positions (p1 p2 : Nat) (total : Nat) : Nat :=
  if h : p1 < p2 then p2 - p1
  else total - (p1 - p2)

theorem num_boys_in_class (p1 p2 : Nat) (total : Nat) :
  p1 = 6 ∧ p2 = 16 ∧ num_positions p1 p2 total = 10 → total = 22 :=
by
  intros h
  sorry

end num_boys_in_class_l137_137790


namespace length_DF_l137_137314

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 0⟩
def C : Point := ⟨2, 2⟩
def D : Point := ⟨0, 2⟩
def E : Point := ⟨2, 1⟩

def on_AE (F : Point) : Prop :=
  ∃ x : ℝ, F = ⟨x, x / 2⟩

def perpendicular (F : Point) : Prop :=
  let slope_AE := (1 - 0) / (2 - 0)
  let slope_DF := (F.y - D.y) / (F.x - D.x)
  slope_AE * slope_DF = -1

def length (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

theorem length_DF (F : Point) (hF_on_AE : on_AE F) (h_perpendicular : perpendicular F) :
  length D F = 4 / Real.sqrt 5 :=
sorry

end length_DF_l137_137314


namespace log_sufficient_for_product_product_not_necessary_for_log_l137_137853

theorem log_sufficient_for_product (a b : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  (log a b > 0) → ((a - 1) * (b - 1) > 0) := by
  sorry

theorem product_not_necessary_for_log (a b : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  ((a - 1) * (b - 1) > 0) → ¬(log a b > 0) := by
  sorry

end log_sufficient_for_product_product_not_necessary_for_log_l137_137853


namespace funfair_tickets_l137_137531

theorem funfair_tickets 
  (T R : ℕ)
  (h1 : T = 100 * R)
  (h2 : 0.35 * T - 100 = 950) :
  R = 30 := by
  -- We assume the proof steps follow here.
  sorry

end funfair_tickets_l137_137531


namespace distinct_natural_numbers_inequality_l137_137090

theorem distinct_natural_numbers_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ)
  (h_distinct : list.nodup [a₁, a₂, a₃, a₄, a₅, a₆, a₇]) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 :=
by
  sorry

end distinct_natural_numbers_inequality_l137_137090


namespace cryptarithm_main_l137_137490

noncomputable def cryptarithm_solution1 (Φ E BP A J : ℕ) : Prop :=
  Φ = 2 ∧ E = 4 ∧ BP = 79 ∧ A = 1 ∧ J = 158 ∧ Φ / E + BP / A / J = 1

noncomputable def cryptarithm_solution2 (Φ E BP A J : ℕ) : Prop :=
  Φ = 6 ∧ E = 8 ∧ BP = 35 ∧ A = 1 ∧ J = 140 ∧ Φ / E + BP / A / J = 1

noncomputable def cryptarithm_solution3 (Φ E BP A J : ℕ) : Prop :=
  Φ = 4 ∧ E = 5 ∧ BP = 72 ∧ A = 1 ∧ J = 360 ∧ Φ / E + BP / A / J = 1

-- Main statement asserting that these solutions indeed sum to 1
theorem cryptarithm_main :
  ∃ (Φ E BP A J : ℕ),
    (cryptarithm_solution1 Φ E BP A J ∨ cryptarithm_solution2 Φ E BP A J ∨ cryptarithm_solution3 Φ E BP A J) :=
by {
  use [2, 4, 79, 1, 158],
  left,
  repeat {split; try {refl}; try {norm_num}}
} ∨
by {
  use [6, 8, 35, 1, 140],
  right, left,
  repeat {split; try {refl}; try {norm_num}}
} ∨
by {
  use [4, 5, 72, 1, 360],
  right, right,
  repeat {split; try {refl}; try {norm_num}}
}

end cryptarithm_main_l137_137490


namespace vector_projection_l137_137509

noncomputable def projection_vector := λ (v w : ℝ × ℝ × ℝ), 
  let dot_vw := v.1 * w.1 + v.2 * w.2 + v.3 * w.3 in
  let dot_ww := w.1 * w.1 + w.2 * w.2 + w.3 * w.3 in
  (dot_vw / dot_ww) • w

theorem vector_projection : 
  projection_vector (4, -1, 3) (3, 1, 2) = (51/14, 17/14, 17/7) := 
  sorry

end vector_projection_l137_137509


namespace system_of_equations_xy_l137_137576

theorem system_of_equations_xy (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = 5) :
  x - y = 2 := sorry

end system_of_equations_xy_l137_137576


namespace Janet_earnings_l137_137225

/--
Janet works as an exterminator and also sells molten metal casts of fire ant nests on the Internet.
She gets paid $70 an hour for exterminator work and makes $20/pound on her ant nest sculptures.
Given that she does 20 hours of exterminator work and sells a 5-pound sculpture and a 7-pound sculpture,
prove that Janet's total earnings are $1640.
-/
theorem Janet_earnings :
  let hourly_rate_exterminator := 70
  let hours_worked := 20
  let rate_per_pound := 20
  let sculpture_one_weight := 5
  let sculpture_two_weight := 7

  let exterminator_earnings := hourly_rate_exterminator * hours_worked
  let total_sculpture_weight := sculpture_one_weight + sculpture_two_weight
  let sculpture_earnings := rate_per_pound * total_sculpture_weight

  let total_earnings := exterminator_earnings + sculpture_earnings
  total_earnings = 1640 := 
by
  sorry

end Janet_earnings_l137_137225


namespace parallelogram_area_l137_137993

open Real

variables (r s : ℝ^3)
-- Condition 1: r and s are unit vectors
hypothesis (norm_r : ‖r‖ = 1)
hypothesis (norm_s : ‖s‖ = 1)
-- Condition 2: The angle between r and s is 45 degrees
hypothesis (angle_45 : real.inner_product_space.angle r s = π / 4)

-- Define the vectors u and v given their relationships
def u : ℝ^3 := (3 * s - r) / 2
def v : ℝ^3 := (3 * r + 3 * s) / 2

-- Prove that the area of the parallelogram with given diagonals is 9√2 / 4
theorem parallelogram_area : 
  let diag1 := r + 3 * s,
      diag2 := 3 * r + s in
  by sorry
/equivalent to/ ‖u × v‖ = 9 * sqrt 2 / 4 := sorry

end parallelogram_area_l137_137993


namespace cos_symmetry_center_l137_137340

noncomputable def symmetryCenter (k : ℤ) : ℝ × ℝ :=
  (2 * k * Real.pi + Real.pi / 3, 0)

theorem cos_symmetry_center :
  ∃ k : ℤ, symmetryCenter k = (Real.pi / 3, 0) :=
  by
    use 0
    simp [symmetryCenter, Real.pi]
    sorry

end cos_symmetry_center_l137_137340


namespace distinct_nat_numbers_inequality_l137_137091

theorem distinct_nat_numbers_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ)
  (h_distinct: ∀ i j : ℕ, i ≠ j → a[i] ≠ a[j]) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 :=
sorry

end distinct_nat_numbers_inequality_l137_137091


namespace average_glasses_is_15_l137_137799

variable (S L : ℕ)

-- Conditions:
def box1 := 12 -- One box contains 12 glasses
def box2 := 16 -- Another box contains 16 glasses
def total_glasses := 480 -- Total number of glasses
def diff_L_S := 16 -- There are 16 more larger boxes

-- Equations derived from conditions:
def eq1 : Prop := (12 * S + 16 * L = total_glasses)
def eq2 : Prop := (L = S + diff_L_S)

-- We need to prove that the average number of glasses per box is 15:
def avg_glasses_per_box := total_glasses / (S + L)

-- The statement we need to prove:
theorem average_glasses_is_15 :
  (12 * S + 16 * L = total_glasses) ∧ (L = S + diff_L_S) → avg_glasses_per_box = 15 :=
by
  sorry

end average_glasses_is_15_l137_137799


namespace avg_age_of_coaches_l137_137694

theorem avg_age_of_coaches (n_girls n_boys n_coaches : ℕ)
  (avg_age_girls avg_age_boys avg_age_members : ℕ)
  (h_girls : n_girls = 30)
  (h_boys : n_boys = 15)
  (h_coaches : n_coaches = 5)
  (h_avg_age_girls : avg_age_girls = 18)
  (h_avg_age_boys : avg_age_boys = 19)
  (h_avg_age_members : avg_age_members = 20) :
  (n_girls * avg_age_girls + n_boys * avg_age_boys + n_coaches * 35) / (n_girls + n_boys + n_coaches) = avg_age_members :=
by sorry

end avg_age_of_coaches_l137_137694


namespace converse_false_inverse_false_main_proof_l137_137118

def p := ∀ (Q : Quadrilateral), Q.isSquare
def q := ∀ (Q : Quadrilateral), Q.isRectangle

theorem converse_false (p q : Prop) (hpq : p → q) : ¬(q → p) :=
sorry

theorem inverse_false (p q : Prop) (hpq : p → q) : ¬(¬p → ¬q) :=
sorry

theorem main_proof : (¬(∀ (Q : Quadrilateral), Q.isRectangle → Q.isSquare)) ∧ (¬(¬(∀ (Q : Quadrilateral), Q.isSquare) → ¬(∀ (Q : Quadrilateral), Q.isRectangle))) :=
by
  apply and.intro
  {
    apply converse_false
    sorry
  }
  {
    apply inverse_false
    sorry
  }

end converse_false_inverse_false_main_proof_l137_137118


namespace white_on_saturday_l137_137006

noncomputable def dandelion_bloom := 
  { t : ℕ // t >= 0 }

def blooming_stages (d : dandelion_bloom) : ℕ :=
if d.val < 3 then 0 -- yellow
else if d.val < 4 then 1 -- white
else 2 -- seeds dispersed

-- Conditions:
-- Monday's counts
def monday_yellow : ℕ := 20
def monday_white : ℕ := 14

-- Wednesday's counts
def wednesday_yellow : ℕ := 15
def wednesday_white : ℕ := 11

-- Additional Definitions arising from conditions:
def monday_total : ℕ := monday_yellow + monday_white
def wednesday_total : ℕ := wednesday_yellow + wednesday_white
def new_bloomed : ℕ := wednesday_total - monday_yellow

-- Theorem to prove the correct answer.
theorem white_on_saturday : ∀ (initial_monday_total initial_wednesday_total monday_yellows : ℕ),
  initial_monday_total = 34 → -- Monday's 20 yellow + 14 white
  initial_wednesday_total = 26 → -- Wednesday's 15 yellow + 11 white
  monday_yellows = 20 → -- Monday's yellow dandelions
  (initial_wednesday_total - monday_yellows) = 6 :=  -- Newly bloomed turning white on Saturday
by
  assume initial_monday_total initial_wednesday_total monday_yellows,
  assume h₁ : initial_monday_total = 34,
  assume h₂ : initial_wednesday_total = 26,
  assume h₃ : monday_yellows = 20,
  rw [h₂, h₃],
  simp,
  exact 6

end white_on_saturday_l137_137006


namespace min_ratio_CD_AD_l137_137543

-- Define the geometric configuration and constraints 
structure Trapezoid (A B C D : Type) [MetricSpace A] :=
    (is_right_angle_A : RightAngle A)
    (is_right_angle_D : RightAngle D)
    (is_perpendicular_BD_BC : Perpendicular BD BC)

-- Define a proof problem for the ratio of sides in the trapezoid
theorem min_ratio_CD_AD 
    {A B C D : Type} [MetricSpace A]
    (trapezoid: Trapezoid A B C D) : 
    ∃ (ratio : ℝ), ratio = 2 := 
by
  sorry

end min_ratio_CD_AD_l137_137543


namespace finite_solutions_to_equation_l137_137305

theorem finite_solutions_to_equation :
  ∃ (n : ℕ), ∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧ (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) = 1 / 1983) → 
  (a ≤ n ∧ b ≤ n ∧ c ≤ n) :=
sorry

end finite_solutions_to_equation_l137_137305


namespace number_of_tangents_l137_137103

-- Define the line equation kx - y - k + 1 = 0
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - k + 1 = 0

-- Define the curve equation E: y = ax^3 + bx^2 + 5/3
def curve (a b x y : ℝ) : Prop := y = a * x^3 + b * x^2 + 5 / 3

-- Define the tangent line condition that tangents at points A and C are parallel
def tangent_parallel (a b : ℝ) : Prop := 
  ∀ x₁ x₂, x₁ ≠ x₂ → 3 * a * (x₁^2) + 2 * b * x₁ = 3 * a * (x₂^2) + 2 * b * x₂

-- Define the symmetric property about the center of symmetry
def symmetry_center (a b : ℝ) : Prop := a + b + 5 / 3 = 1 ∧ b = -1

-- Define the condition where the tangent passes through the point (b, a)
def point_tangent_condition (x₀ y₀ a b : ℝ) : Prop :=
  y₀ = a * x₀ * x₀ - 2 * x₀ + (1/3) * x₀ * x₀ * x₀ - x₀ + 5 / 3

-- Main statement: the number of tangents to the curve E passing through point (b, a)
theorem number_of_tangents (a b : ℝ) (hk : k ∈ ℝ) (hab_nonzero : a ≠ 0 ∧ b ≠ 0) 
                          (tangent_cond : tangent_parallel a b) (sym_center : symmetry_center a b) : 
  ∃ n : ℕ, n = 1 := sorry

end number_of_tangents_l137_137103


namespace find_k_value_l137_137652

noncomputable def arithmetic_seq (a d : ℤ) : ℕ → ℤ
| n => a + (n - 1) * d

theorem find_k_value (a d : ℤ) (k : ℕ) 
  (h1 : arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 24)
  (h2 : (Finset.range 11).sum (λ i => arithmetic_seq a d (5 + i)) = 110)
  (h3 : arithmetic_seq a d k = 16) : 
  k = 16 :=
sorry

end find_k_value_l137_137652


namespace widget_cost_reduction_l137_137668

theorem widget_cost_reduction (W R : ℝ) (h1 : 6 * W = 36) (h2 : 8 * (W - R) = 36) : R = 1.5 :=
by
  sorry

end widget_cost_reduction_l137_137668


namespace simplify_and_evaluate_expression_l137_137309

theorem simplify_and_evaluate_expression :
  let x := -1
  let y := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 :=
by
  let x := -1
  let y := Real.sqrt 2
  sorry

end simplify_and_evaluate_expression_l137_137309


namespace can_buy_two_pies_different_in_both_l137_137173

structure Pie :=
  (filling : Type)
  (preparation : Type)

def apple : Type := unit
def cherry : Type := unit
def fried : Type := unit
def baked : Type := unit

def apple_fried : Pie := { filling := apple, preparation := fried }
def apple_baked : Pie := { filling := apple, preparation := baked }
def cherry_fried : Pie := { filling := cherry, preparation := fried }
def cherry_baked : Pie := { filling := cherry, preparation := baked }

def possible_pies : List Pie := [apple_fried, apple_baked, cherry_fried, cherry_baked]

theorem can_buy_two_pies_different_in_both 
  (available_pies : List Pie) 
  (h : available_pies.length ≥ 3) : 
  ∃ (p1 p2 : Pie), p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation := 
  sorry

end can_buy_two_pies_different_in_both_l137_137173


namespace least_number_to_subtract_l137_137742

-- Define the problem and prove that this number, when subtracted, makes the original number divisible by 127.
theorem least_number_to_subtract (n : ℕ) (h₁ : n = 100203) (h₂ : 127 > 0) : 
  ∃ k : ℕ, (100203 - 72) = 127 * k :=
by
  sorry

end least_number_to_subtract_l137_137742


namespace proof_problem_l137_137894

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l137_137894


namespace hyperbola_proof_chord_line_proof_l137_137114

noncomputable def hyperbola_equation : Prop := 
  ∃ (a b : ℝ) (ha : a > 0) (hb : b > 0), 
    (∀ (x y : ℝ), ((x = 3) ∧ (y = sqrt 7) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) ∧ 
    (∃ (f1 f2 : ℝ × ℝ), f1 = (-2, 0) ∧ f2 = (2, 0) ∧ (x^2 / 18 + y^2 / 14 = 1) → 
    (f1 = (-2,0) ∧ f2 = (2,0))) ∧ 
    (x^2 / 2 - y^2 / 2 = 1))

noncomputable def chord_line_equation : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - y1^2 = 2) (h2 : x2^2 - y2^2 = 2) (p : ℝ × ℝ), 
    p = (1, 2) → 
    ((y1 - y2) / (x1 - x2) = 1 / 2) ∧ 
    (∃ (k : ℝ), y - 2 = k * (x - 1) → k = 1 / 2 ∧ x - 2y + 3 = 0)

theorem hyperbola_proof : hyperbola_equation := sorry

theorem chord_line_proof : chord_line_equation := sorry

end hyperbola_proof_chord_line_proof_l137_137114


namespace quadratic_has_one_solution_l137_137049

theorem quadratic_has_one_solution (m : ℝ) : 3 * (49 / 12) - 7 * (49 / 12) + m = 0 → m = 49 / 12 :=
by
  sorry

end quadratic_has_one_solution_l137_137049


namespace attendance_decrease_is_20_percent_l137_137354

variable (attendance_last_year : ℝ) (projected_increase : ℝ) (actual_attendance_percentage : ℝ)

def projected_attendance := attendance_last_year * (1 + projected_increase)
def actual_attendance := projected_attendance * actual_attendance_percentage
def percent_decrease := (attendance_last_year - actual_attendance) / attendance_last_year * 100

theorem attendance_decrease_is_20_percent
    (h1 : projected_increase = 0.25)
    (h2 : actual_attendance_percentage = 0.64) :
    percent_decrease attendance_last_year projected_increase actual_attendance_percentage = 20 :=
sorry

end attendance_decrease_is_20_percent_l137_137354


namespace area_of_ABCD_l137_137953

noncomputable def area_quadrilateral_ABCD : ℝ :=
  let AB := 15
  let BC := 5
  let CD := 12
  let AD := 13
  let BD := real.sqrt (BC^2 + CD^2)
  let sBDA := (AB + BD + AD) / 2
  let area_BCD := (BC * CD) / 2
  let area_BDA := 
    real.sqrt (
      sBDA * (sBDA - AB) * (sBDA - BD) * (sBDA - AD)
    )
  area_BCD + area_BDA

theorem area_of_ABCD :
  let AB := 15
  let BC := 5
  let CD := 12
  let AD := 13
  let BD := real.sqrt (BC^2 + CD^2)
  let sBDA := (AB + BD + AD) / 2
  let area_BCD := (BC * CD) / 2
  let area_BDA := 
    real.sqrt (
      sBDA * (sBDA - AB) * (sBDA - BD) * (sBDA - AD)
    )
  area_BCD + area_BDA = 106 := by
  sorry

end area_of_ABCD_l137_137953


namespace triangle_side_relation_l137_137196

theorem triangle_side_relation (A B C D E : Point) (h1 : IsTriangle A B C)
  (h2 : IsMedian CE A B C) (h3 : Perpendicular AD CE) : 
  side A B = 2 * side A C :=
by
  sorry

end triangle_side_relation_l137_137196


namespace exists_int_between_sqrt3_sqrt13_l137_137400

theorem exists_int_between_sqrt3_sqrt13 : ∃ (n : ℤ), real.sqrt 3 < n ∧ n < real.sqrt 13 := 
sorry

end exists_int_between_sqrt3_sqrt13_l137_137400


namespace divisors_of_2009_l137_137607

theorem divisors_of_2009 (dividend remainder : ℕ) (h_dividend : dividend = 2016) (h_remainder : remainder = 7):
  ∃ div_count : ℕ, div_count = 4 :=
by
  have h1 : 2009 = dividend - remainder := by
    rw [h_dividend, h_remainder]
    exact Nat.sub_eq_of_eq_add' (Eq.symm (Nat.add_comm _ _))
  sorry

end divisors_of_2009_l137_137607


namespace length_of_AB_l137_137606

def point := ℝ × ℝ

noncomputable def length (A B : point) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem length_of_AB
  (A B C D : point)
  (h1 : (C.2 = B.2 + A.2) ∧ (A.1 = C.1))
  (h2 : angle A B C = 45)
  (h3 : (D.2 = B.2 + D.2) ∧ (A.1 = D.1))
  (h4 : angle A B D = 30)
  (h5 : length C D = 6) :
  length A B = 12 - 6 * real.sqrt 2 :=
sorry

end length_of_AB_l137_137606


namespace third_selected_individual_is_16_l137_137257

-- Define the population
def population : List Nat := List.range' 1 21  -- population range from 1 to 20

-- Define the random number table as a list of lists of Nat (2x7 table)
def randomNumberTable : List (List Nat) := [
  [1818, 792, 4544, 1716, 5809,  7983, 8619],
  [6206, 7650,  310, 5523, 6405,  526, 6238]
]

-- Extract the number sequences from the table
def sequenceFromRandomTable : List Nat := [
  58, 09, 79, 83, 86, 19
]

-- Select valid individuals (i.e., numbers within the population range)
def validSelections := sequenceFromRandomTable.filter (λ x => x ≤ 20)

-- Theorem statement
theorem third_selected_individual_is_16 : validSelections.nth 2 = some 16 :=
by
  sorry

end third_selected_individual_is_16_l137_137257


namespace trapezoid_exists_in_marked_vertices_l137_137191

-- Definitions
noncomputable def regular_n_gon (n : ℕ) := sorry -- Definition of a regular n-gon
noncomputable def marked_vertices (polygon : Type) (k : ℕ) := sorry -- Definition of k marked vertices in a polygon

-- Given parameters
def n := 1981 -- Number of sides of the polygon
def k := 64 -- Number of marked vertices

-- The problem statement
theorem trapezoid_exists_in_marked_vertices 
    (polygon : regular_n_gon n) 
    (marked : marked_vertices polygon k) :
    ∃ (a b c d : marked), 
      (∃ (segments : (a, b), (c, d) ∈ marked), 
        -- defines the existence of parallel segments forming a trapezoid
        ((a ≠ b) ∧ (c ≠ d)) ∧ (is_parallel ⟨a, b⟩ ⟨c, d⟩)) 
 := sorry

end trapezoid_exists_in_marked_vertices_l137_137191


namespace find_coordinates_of_vector_CB_l137_137093

def point := ℝ × ℝ × ℝ

def midpoint (A B : point) : point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def vector (A B : point) : point :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

theorem find_coordinates_of_vector_CB
  (A B : point)
  (hA : A = (1, -2, -7))
  (hB : B = (3, 10, 9)) :
  let C := midpoint A B in
  vector C B = (1, 6, 8) :=
by
  sorry

end find_coordinates_of_vector_CB_l137_137093


namespace angle_bisector_l137_137260

-- Define the semicircle, points, and intersections
variable {A B C D S T : Point}

-- Assume the necessary geometric relationships
axiom semicircle (A B : Point) (P : Point) : ∃ O : Point, dist A O = dist B O ∧ dist P O = dist A O
axiom points_on_semicircle (A B C D : Point) : 
  ∃ O : Point, (C ≠ D) ∧ dist A O = dist B O ∧ dist C O = dist A O ∧ dist D O = dist A O
axiom intersection (A B C D S : Point) : line A C ∩ line B D = {S}
axiom foot_of_perpendicular (S A B T : Point) : 
  orthogonal_projection (line A B) S = T

-- State the goal to prove
theorem angle_bisector (A B C D S T : Point) 
  (h1 : semicircle A B C) 
  (h2 : points_on_semicircle A B C D) 
  (h3 : intersection A B C D S) 
  (h4 : foot_of_perpendicular S A B T) : 
  angle_bisector (line S T) (angle C T D) :=
sorry

end angle_bisector_l137_137260


namespace geese_initial_formation_l137_137449

theorem geese_initial_formation (G : ℕ) 
  (h1 : G / 2 + 4 = 12) : G = 16 := 
sorry

end geese_initial_formation_l137_137449


namespace proof_question_l137_137878

open Set Real

def U := univ
def A := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B := {x : ℝ | x^2 - 2 * x < 0}

theorem proof_question : A ∪ (U \ B) = {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 2} := by
  sorry

end proof_question_l137_137878


namespace factorial_equivalence_l137_137801

-- Define the constants involved
def a := 3^2
def b := 4
def c := 6^2
def d := 5

-- Calculate the intermediate result
def intermediate_result := a * b * c * d

-- Define the final result
def final_result := 6480

-- Formulate the proof problem
theorem factorial_equivalence : factorial intermediate_result = factorial final_result := 
by
  -- Provide the placeholders, actual proof omitted
  sorry

end factorial_equivalence_l137_137801


namespace area_of_enclosed_region_l137_137834

theorem area_of_enclosed_region :
  let graph_eq : ℝ → ℝ → Prop := λ x y, (|x - 80| + |y| = |x / 3|)
  ∃ pts : set (ℝ × ℝ), (∀ x y, (x, y) ∈ pts ↔ graph_eq x y) ∧
  let enclosing_pts := {p : ℝ × ℝ | ∃ x y, (x, y) ∈ pts},
  let region_area : ℝ := 1 / 2 * (120 - 60) * (20 - (-20)),
  region_area = 1200 :=
by
  let graph_eq : ℝ → ℝ → Prop := λ x y, (|x - 80| + |y| = |x / 3|)
  let enclosing_pts := {p : ℝ × ℝ | ∃ x y, (x, y) ∈ {pts : set (ℝ × ℝ) | ∀ x y, (x, y) ∈ pts ↔ graph_eq x y}}
  let region_area : ℝ := 1 / 2 * (120 - 60) * (20 - (-20))
  have h_area_eq : region_area = 1200
  sorry
  exact ⟨enclosing_pts, ⟨λ x y, ⟨⟨x, y⟩, rfl⟩, h_area_eq⟩⟩

end area_of_enclosed_region_l137_137834


namespace RY_eq_RZ_l137_137023

variable (A B C P Q R Y Z : Type)
variable [IsTriangle A B C]
variable [Incidence (XA BC P)]
variable [Incidence (XB CA Q)]
variable [Incidence (XC AB R)]
variable [ParallelThrough R PQ Y]
variable [Incidence Y BC]
variable [Incidence Z AP]

theorem RY_eq_RZ : RY = RZ := by
  sorry

end RY_eq_RZ_l137_137023


namespace find_a_for_eq_three_solutions_l137_137074

theorem find_a_for_eq_three_solutions (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, 
    8^(abs (x1 - a)) * log 1/5 (x1^2 + 2 * x1 + 5) + 2^(x1^2 + 2 * x1) * log (sqrt 5) (3 * abs (x1 - a) + 4) = 0 ∧
    8^(abs (x2 - a)) * log 1/5 (x2^2 + 2 * x2 + 5) + 2^(x2^2 + 2 * x2) * log (sqrt 5) (3 * abs (x2 - a) + 4) = 0 ∧
    8^(abs (x3 - a)) * log 1/5 (x3^2 + 2 * x3 + 5) + 2^(x3^2 + 2 * x3) * log (sqrt 5) (3 * abs (x3 - a) + 4) = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ↔ 
  (a = -7 / 4 ∨ a = -1 ∨ a = -1 / 4) := 
sorry

end find_a_for_eq_three_solutions_l137_137074


namespace maximal_area_for_OMAN_l137_137863

-- Define the basic geometrical entities and conditions
variables (O B C A M N : Type)
variables (triangle_OBC : Triangle O B C)
variable (alpha : Angle)
variable (beta : Angle)
variable (A_on_BC : A ∈ Line B C)
variable (angle_BOC : Angle.triangle O B C = alpha)
variable (angle_MAN : Angle.triangle M A N = beta)
variable (alpha_beta_cond : alpha + beta < π)
variables (angle_B_cond : ∀ {A B C : Point}, ∠ABC ≤ π/2 + β/2)
variables (angle_C_cond : ∀ {A B C : Point}, ∠ACB ≤ π/2 + β/2)

-- The proof statement
theorem maximal_area_for_OMAN 
  (h : is_parallel_to (Line M N) (Line B C)) 
  (h2 : |M - A| = |A - N|) : 
  maximal_area (Quadrilateral O M A N) :=
sorry

end maximal_area_for_OMAN_l137_137863


namespace fill_tank_time_l137_137426

theorem fill_tank_time (R L E : ℝ) (fill_time : ℝ) (leak_time : ℝ) (effective_rate : ℝ) : 
  (R = 1 / fill_time) → 
  (L = 1 / leak_time) →
  (E = R - L) →
  (fill_time = 10) →
  (leak_time = 110) →
  (E = 1 / effective_rate) →
  effective_rate = 11 :=
by
  sorry

end fill_tank_time_l137_137426


namespace triangle_area_EFC_l137_137729

-- Define the problem conditions and the proof statement

variables (A B C E F : Type) [EuclideanGeometry] (area : Triangle → ℝ)
variables (ABC AEF EBC EFC : Triangle)
variables (EF_BC_parallel : Parallel EF BC)

-- Given conditions
variables (h1 : area ABC = 1)
variables (h2 : EF.parallel BC)
variables (h3 : area AEF = area EBC)

-- Proof statement
theorem triangle_area_EFC :
  area EFC = sqrt(5) - 2 :=
by
  sorry

end triangle_area_EFC_l137_137729


namespace smallest_integer_value_of_x_satisfying_eq_l137_137048

theorem smallest_integer_value_of_x_satisfying_eq (x : ℤ) (h : |x^2 - 5*x + 6| = 14) : 
  ∃ y : ℤ, (y = -1) ∧ ∀ z : ℤ, (|z^2 - 5*z + 6| = 14) → (y ≤ z) :=
sorry

end smallest_integer_value_of_x_satisfying_eq_l137_137048


namespace cn_geometric_seq_l137_137723

-- Given conditions
def Sn (n : ℕ) : ℚ := (3 * n^2 + 5 * n) / 2
def an (n : ℕ) : ℕ := 3 * n + 1
def bn (n : ℕ) : ℕ := 2^n

theorem cn_geometric_seq : 
  ∃ q : ℕ, ∃ (c : ℕ → ℕ), (∀ n : ℕ, c n = q^n) ∧ (∀ n : ℕ, ∃ m : ℕ, c n = an m ∧ c n = bn m) :=
sorry

end cn_geometric_seq_l137_137723


namespace value_of_a8_l137_137876

noncomputable def f (x : ℝ) : ℝ :=
2 * f (2 - x) - x^2 + 8 * x - 8

def tangent_line_at_one (x : ℝ) : ℝ :=
2 * x - 1

def a (n : ℕ) : ℝ := - (n : ℝ) / 2 + 3 / 2

theorem value_of_a8 (a₁_ne_1 : a 1 ≠ 1) : a 8 = -5 / 2 :=
sorry

end value_of_a8_l137_137876


namespace count_pairs_l137_137636

theorem count_pairs (p : ℕ) (hp : p.prime) (h2 : p % 3 = 2) :
  ∃ (s : Finset (ℕ × ℕ)), s.card ≤ p-1 ∧
    ∀ x ∈ s, (x.1 > 0 ∧ x.1 < p) ∧ (x.2 > 0 ∧ x.2 < p) ∧ (x.1^2 - x.2^3 - 1) % p = 0 :=
begin
  sorry
end

end count_pairs_l137_137636


namespace hyperbola_solution_l137_137334

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 = 1

theorem hyperbola_solution : 
  (∀ x y, ellipse_eq x y → (x = 2 ∧ y = 1) → hyperbola_eq x y) :=
by
  intros x y h_ellipse h_point
  sorry

end hyperbola_solution_l137_137334


namespace parabola_vertex_sum_eq_l137_137489

theorem parabola_vertex_sum_eq : 
  ∃ (a b c : ℚ), 
    (∀ (x : ℚ), (∀ (y : ℚ), y = a * x ^ 2 + b * x + c ↔ y = a * (x - 3) ^ 2 - 2)) ∧ 
    (a * 0 ^ 2 + b * 0 + c = 5) ∧ 
    a + b + c = 10 / 9 :=
begin
  sorry
end

end parabola_vertex_sum_eq_l137_137489


namespace fill_in_the_blank_l137_137059

theorem fill_in_the_blank (x : ℕ) (h : (x - x) + x * x + x / x = 50) : x = 7 :=
sorry

end fill_in_the_blank_l137_137059


namespace final_pile_count_l137_137526

-- Definitions related to the problem
def deck_of_cards (n : ℕ) := { x : ℕ | 1 ≤ x ∧ x ≤ n }
def card_piles (n : ℕ) := list (deck_of_cards n)
def valid_k (m : ℕ) := { k : ℕ | 1 ≤ k ∧ k < m }
def rearrange_piles : ∀ {n m : ℕ} (piles : card_piles n) (k : valid_k m), card_piles n
  | _, _, [], _ := []
  | _, _, (p :: ps), ⟨k, h1, h2⟩ := sorry

-- Final theorem
theorem final_pile_count (n : ℕ) (piles : card_piles n)
  (h : ∀ pile, pile ∈ piles → pile = {1, 2, ..., n})
  : (piles.length = 1) → (∃ c : ℕ, c = 2^(n-2)) :=
sorry

end final_pile_count_l137_137526


namespace sample_correlation_coefficient_is_one_l137_137610

theorem sample_correlation_coefficient_is_one 
  (n : ℕ) (x y : Finₓ n → ℝ)
  (h₀ : 2 ≤ n)
  (h₁ : ¬ ∀ i j, x i = x j)
  (h₂ : ∀ i, y i = (1 / 2 : ℝ) * (x i) + 1) : 
  sampleCorrelationCoefficient x y = 1 := 
sorry

end sample_correlation_coefficient_is_one_l137_137610


namespace train_crosses_pole_in_8_seconds_l137_137751

-- Definitions based on conditions
def length_of_train : ℝ := 320  -- in meters
def speed_of_train_kmph : ℝ := 144  -- in km/hr

-- Conversion of speed from km/hr to m/s
def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

-- Theorem statement
theorem train_crosses_pole_in_8_seconds :
  (length_of_train / speed_of_train_mps) = 8 :=
by
  sorry

end train_crosses_pole_in_8_seconds_l137_137751


namespace fractional_equation_solution_l137_137888

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l137_137888


namespace unique_square_side_length_l137_137904

def side_length_of_square (a : ℝ) : ℝ :=
  let side_length := Real.sqrt (Real.sqrt 72) in
  side_length

theorem unique_square_side_length (a : ℝ) (h : ∃! quad : (ℝ × ℝ) → Prop, 
  quad = (λ v, v.snd = v.fst^3 + a * v.fst))
  : side_length_of_square a = Real.sqrt (Real.sqrt 72) :=
by
  sorry

end unique_square_side_length_l137_137904


namespace intersecting_midpoint_d_value_l137_137342

theorem intersecting_midpoint_d_value 
    (d : ℝ)
    (line_eq : ∀ x y : ℝ, 2 * x + 3 * y = d)
    (segment_midpoint : (5, 7)) : d = 31 := 
by 
  have midpoint_x : 5 = (3 + 7) / 2 := by norm_num
  have midpoint_y : 7 = (4 + 10) / 2 := by norm_num
  have midpoint_on_line : 2 * 5 + 3 * 7 = d := line_eq 5 7
  rw [midpoint_x, midpoint_y] at midpoint_on_line
  norm_num at midpoint_on_line
  exact midpoint_on_line

end intersecting_midpoint_d_value_l137_137342


namespace loan_amount_calculation_l137_137661

noncomputable def original_loan_amount (M : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  M * ((1 - (1 + r)^(-n)) / r)

theorem loan_amount_calculation :
  original_loan_amount 807 0.10 3 ≈ 2009 := by
  sorry

end loan_amount_calculation_l137_137661


namespace probability_at_least_one_inferior_l137_137452

-- Definitions based on the given conditions
def total_pencils : ℕ := 10
def good_pencils : ℕ := 8
def inferior_pencils : ℕ := 2

-- Function to calculate combinations
noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

-- The probability of drawing at least one inferior pencil, given the conditions.
theorem probability_at_least_one_inferior :
  let total_ways := comb total_pencils 2 in
  let ways_no_inferior := comb good_pencils 2 in
  total_ways = 45 →
  ways_no_inferior = 28 →
  (1 - (ways_no_inferior : ℚ) / total_ways) = 17 / 45 :=
by
  sorry

end probability_at_least_one_inferior_l137_137452


namespace parallelogram_area_given_diagonals_l137_137990

open Real

variables (r s : ℝ^3)

def is_unit_vector (v : ℝ^3) : Prop :=
  ‖v‖ = 1

def angle_between_vectors (v w : ℝ^3) : ℝ :=
  acos ((dot_product v w) / (‖v‖ * ‖w‖))

noncomputable def parallelogram_area (d1 d2 : ℝ^3) : ℝ :=
  ‖d1 × d2‖ / 2

theorem parallelogram_area_given_diagonals
  (hr : is_unit_vector r)
  (hs : is_unit_vector s)
  (h_angle : angle_between_vectors r s = π / 4) :
  parallelogram_area (r + 3 • s) (3 • r + s) = 9 * sqrt 2 / 4 :=
sorry

end parallelogram_area_given_diagonals_l137_137990


namespace number_of_blocks_l137_137459

theorem number_of_blocks (children_per_block : ℕ) (total_children : ℕ) (h1: children_per_block = 6) (h2: total_children = 54) : (total_children / children_per_block) = 9 :=
by {
  sorry
}

end number_of_blocks_l137_137459


namespace ellipse_with_foci_on_y_axis_l137_137926

theorem ellipse_with_foci_on_y_axis (m : ℝ) : 
  (∃ (f : ℝ → ℝ → Prop), (∀ x y, f x y ↔ (x ^ 2 / (25 - m) + y ^ 2 / (16 + m) = 1) ∧ (m > 9 / 2) ∧ (m < 25)) :=
sorry

end ellipse_with_foci_on_y_axis_l137_137926


namespace product_of_factors_l137_137789

theorem product_of_factors (x : ℕ) (hx1 : 1 < x) (hx2 : nat.num_divisors x = 6) :
  ∏ d in nat.divisors x, d = x^3 := sorry

end product_of_factors_l137_137789


namespace minimize_area_quadrilateral_l137_137959

theorem minimize_area_quadrilateral 
  (O A : Point)
  (phi psi beta : ℝ)
  (h_cond : phi + psi + beta > 180) :
  (phi > 90 - beta / 2 → psi > 90 - beta / 2 →
   ∃ M N : Point, (|MA| = |AN|) ∧ ∠(MN)A = beta ∧ minimize_area_quadrilateral O M A N) 
  ∨ 
  (phi ≤ 90 - beta / 2 ∨ psi ≤ 90 - beta / 2 →
   ∃ M N : Point, O = M ∨ O = N ∧ degenerate_to_triangle O M A N) :=
sorry

end minimize_area_quadrilateral_l137_137959


namespace swap_rows_matrix_l137_137837

def swapMatrix (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.vecCons (Matrix.vecCons c d) (Matrix.vecCons a b)

theorem swap_rows_matrix : (let N : Matrix (Fin 2) (Fin 2) ℝ := ![![0, 1], ![1, 0]],
                                M : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]])
                                N.mul M = swapMatrix a b c d :=
by
  sorry

end swap_rows_matrix_l137_137837


namespace find_a_l137_137119

variables (U : Set ℕ) (a : ℕ)
noncomputable def A := {1, |a - 5|, 9}
def complement_A := {5, 7}

theorem find_a (hU : U = {1, 3, 5, 7, 9}) (h_complement : complement_A = {5, 7}) : 
  a = 2 ∨ a = 8 :=
sorry

end find_a_l137_137119


namespace dot_product_focus_line_pass_through_fixed_point_l137_137203

-- Define the parabola and the conditions
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

def line (l : ℝ → ℝ → Prop) (t : ℝ) : Prop :=
  ∃ y : ℝ, l (t * y + 1) y

def focus : ℝ × ℝ := (1, 0)

-- Define the points A and B
variables (A B : ℝ × ℝ)

-- Assert they lie on the parabola and the line
def lie_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

def intersect_line (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, l P.1 P.2

-- Vector dot product
def dot_product (O A B : ℝ × ℝ) : ℝ :=
  (A.1 * B.1) + (A.2 * B.2)

-- Problem (Ⅰ) rewritten in Lean 4
theorem dot_product_focus_line:
  (∀l : ℝ → ℝ → Prop, (line l) ∧ (∀P, intersect_line l P → lie_on_parabola P)) →
  (intersect_line line_and_focus A) →
  (intersect_line line_and_focus B) →
  (dot_product (0, 0) A B = -3) :=
sorry

-- Problem (Ⅱ) rewritten in Lean 4
theorem pass_through_fixed_point:
  (∀l : ℝ → ℝ → Prop, (line l) ∧ (∀P, intersect_line l P → lie_on_parabola P)) →
  (dot_product (0, 0) A B = -4) →
  (∃ (P : ℝ × ℝ), intersect_line line_and_focus P ∧ P = (2, 0)) :=
sorry

end dot_product_focus_line_pass_through_fixed_point_l137_137203


namespace sum_of_elements_in_A_l137_137573

def A : set ℕ := {x | ∃ (a0 a1 a2 a3 : ℕ), a0 ∈ {0, 1, 2} ∧ a1 ∈ {0, 1, 2} ∧ a2 ∈ {0, 1, 2} ∧ a3 ∈ {1, 2} ∧ x = a0 + a1 * 3 + a2 * 3^2 + a3 * 3^3}

theorem sum_of_elements_in_A : ∑ x in A, x = 2889 := 
sorry

end sum_of_elements_in_A_l137_137573


namespace ladder_distance_from_wall_l137_137322

noncomputable def distance_from_wall (angle : ℝ) (hypotenuse : ℝ) : ℝ :=
  hypothenuse * Real.cos angle

theorem ladder_distance_from_wall :
  distance_from_wall (Real.pi / 3) 9.2 = 4.6 :=
by
  sorry

end ladder_distance_from_wall_l137_137322


namespace cross_section_existence_l137_137285

variables {Point : Type} [LinearOrder Point]

def parallelepiped (A B C D A1 B1 C1 D1 M N K P E Q F : Point) : Prop :=
  M ∈ segment BC ∧ N ∈ segment AA1 ∧ K ∈ segment C1D1 ∧ 
  (plane A1B1C1D1 ∩ plane through [M, N, K]) = intersection points forming the polygonal cross-section [P, E, Q, F]

-- Lean 4 statement for the proof problem
theorem cross_section_existence 
  (A B C D A1 B1 C1 D1 M N K P E Q F : Point)
  (parallelepiped (A B C D A1 B1 C1 D1 M N K P E Q F)) : 
  (∃ P E Q F : Point,
    (M ∈ segment BC) ∧ 
    (N ∈ segment AA1) ∧ 
    (K ∈ segment C1D1) ∧ 
    (plane A1B1C1D1 ∩ plane_through M N K = points [P, E, Q, F])) :=
sorry

end cross_section_existence_l137_137285


namespace min_period_sin_cos_eq_2pi_l137_137706

noncomputable def minimum_positive_period : ℝ := 2 * Real.pi

def function_sin_cos (x : ℝ) : ℝ := Real.sin x * |Real.cos x|

theorem min_period_sin_cos_eq_2pi : (∃ T > 0, ∀ x, function_sin_cos (x + T) = function_sin_cos x) ∧ 
  ∀ T > 0, (∃ x, function_sin_cos (x + T) ≠ function_sin_cos x) → minimum_positive_period = 2 * Real.pi :=
sorry

end min_period_sin_cos_eq_2pi_l137_137706


namespace inequality_proof_l137_137094

variable (x y : ℝ)

theorem inequality_proof (h : x > y) : 2 * x + 1 / (x^2 - 2 * x * y + y^2) ≥ 2 * y + 3 :=
begin
  sorry
end

end inequality_proof_l137_137094


namespace geom_sequence_general_formula_maximum_n_maximum_n_other_l137_137208

-- Definitions for the conditions
def a (n : ℕ) : ℕ := 2^n

def b : ℕ → ℕ
| 1       := 2
| (n + 1) := (n + 1) * 2^n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ i, b (i + 1))

-- Proof statements
theorem geom_sequence_general_formula :
  ∀ n : ℕ, 2^n = a n :=
by sorry

theorem maximum_n :
  S 3 - 3 * (a 3) + 6 ≥ 0 :=
by sorry

theorem maximum_n_other :
  ∀ n > 3, S n - n * (a n) + 6 < 0 :=
by sorry

end geom_sequence_general_formula_maximum_n_maximum_n_other_l137_137208


namespace distinct_three_digit_numbers_count_l137_137128

theorem distinct_three_digit_numbers_count : 
  (∃ digits : Finset ℕ, 
    digits = {1, 2, 3, 4, 5} ∧ 
    ∀ n ∈ digits, n ∈ {1, 2, 3, 4, 5} ∧ 
    (digits.card = 5)) → 
  card (finset.univ.image (λ p : Finset (ℕ × ℕ × ℕ), 
      {x | x.fst ∈ digits ∧ x.snd ∈ digits ∧ x.snd.snd ∈ digits ∧ x.fst ≠ x.snd ∧ x.snd ≠ x.snd.snd ∧ x.fst ≠ x.snd.snd})) = 60 := by
  sorry

end distinct_three_digit_numbers_count_l137_137128


namespace sin_alpha_minus_pi_div_4_l137_137097

variables (α : ℝ)
hypothesis (h1 : real.cos α = -4/5)
hypothesis (h2 : α = real.pi / 4 + 2 * real.pi * n - a)

theorem sin_alpha_minus_pi_div_4 :
  real.sin (α - real.pi / 4) = sqrt(2) / 10 :=
sorry

end sin_alpha_minus_pi_div_4_l137_137097


namespace flour_adjustment_l137_137010

-- Define the conditions
def initial_cookies : ℕ := 30
def initial_flour : ℕ := 2
def desired_cookies : ℕ := 5 * 12

-- Define the amount of flour needed for the adjusted recipe
def flour_needed : ℕ :=
  if desired_cookies = 2 * initial_cookies then
    2 * initial_flour
  else
    0  -- Placeholder for cases that do not match the specific condition

-- The theorem to prove
theorem flour_adjustment : flour_needed = 4 := by
  derive sorry

end flour_adjustment_l137_137010


namespace travel_cost_is_correct_l137_137437

-- Definitions of the conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 60
def road_width : ℝ := 15
def cost_per_sq_m : ℝ := 3

-- Areas of individual roads
def area_road_length := road_width * lawn_breadth
def area_road_breadth := road_width * lawn_length
def intersection_area := road_width * road_width

-- Adjusted area for roads discounting intersection area
def total_area_roads := area_road_length + area_road_breadth - intersection_area

-- Total cost of traveling the roads
def total_cost := total_area_roads * cost_per_sq_m

theorem travel_cost_is_correct : total_cost = 5625 := by
  sorry

end travel_cost_is_correct_l137_137437


namespace number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l137_137076

-- Part (a)
theorem number_of_ways_to_choose_4_from_28 :
  (Nat.choose 28 4) = 20475 :=
sorry

-- Part (b)
theorem number_of_ways_to_choose_3_from_27_with_kolya_included :
  (Nat.choose 27 3) = 2925 :=
sorry

end number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l137_137076


namespace limit_of_f_l137_137495

noncomputable def f (x : ℝ) : ℝ :=
  ( (x + 1) / (2 * x) ) ^ ( log (x + 2) / log (2 - x) )

theorem limit_of_f :
  filter.tendsto f (nhds 1) (nhds (sqrt 3)) :=
by
  sorry

end limit_of_f_l137_137495


namespace value_of_n_if_combinations_equal_l137_137915

theorem value_of_n_if_combinations_equal (n : ℕ) (h : Combination n 3 = Combination n 5) : n = 8 :=
by
  sorry

end value_of_n_if_combinations_equal_l137_137915


namespace balls_in_boxes_distinguished_boxes_l137_137137

theorem balls_in_boxes_distinguished_boxes :
  ∃ (n : ℕ), n = nat.choose 9 2 ∧ n = 36 :=
by {
  -- just write the statement and use 'sorry' to skip proof
  use 36,
  split,
  { unfold nat.choose,
    -- calculation details omitted
    sorry, },
  { refl, },
}

end balls_in_boxes_distinguished_boxes_l137_137137


namespace tetrahedron_volume_proof_l137_137056

noncomputable def tetrahedron_volume (ABC_area : ℝ) (BCD_area : ℝ) (BC_length : ℝ) (angle_ABC_BCD : ℝ) : ℝ :=
  1/3 * ABC_area * (2 * BCD_area / BC_length) * sin angle_ABC_BCD

theorem tetrahedron_volume_proof :
  tetrahedron_volume 150 50 8 (π / 4) = 312.5 * real.sqrt 2 :=
by sorry

end tetrahedron_volume_proof_l137_137056


namespace sequence_properties_l137_137206

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else n + 1

def b (n : ℕ) : ℝ :=
  (a n : ℝ) + (1 / 2^(a n))

noncomputable def S (n : ℕ) : ℝ :=
  (n^2 + 3*n + 1) / 2 - 1 / 2^(n+1)

theorem sequence_properties :
  (a 2 = 3) ∧ (a 3 + a 6 = 11) ∧
  (∀ n, b n = a n + 1 / 2^(a n)) ∧
  (∀ n, S n = ∑ i in finset.range n, b (i + 1)) :=
by
  sorry

end sequence_properties_l137_137206


namespace janet_total_earnings_l137_137223

-- Definitions based on conditions from step a)
def hourly_wage := 70
def hours_worked := 20
def rate_per_pound := 20
def weight_sculpture1 := 5
def weight_sculpture2 := 7

-- Statement for the proof problem
theorem janet_total_earnings : 
  let earnings_from_extermination := hourly_wage * hours_worked
  let earnings_from_sculpture1 := rate_per_pound * weight_sculpture1
  let earnings_from_sculpture2 := rate_per_pound * weight_sculpture2
  earnings_from_extermination + earnings_from_sculpture1 + earnings_from_sculpture2 = 1640 := 
by
  sorry

end janet_total_earnings_l137_137223


namespace speed_of_faster_train_l137_137383

-- Define constants based on the conditions
def L_s : ℝ := 200  -- length of slower train in meters
def L_f : ℝ := 800  -- length of faster train in meters
def T : ℝ := 35.997120230381576  -- time to cross each other in seconds
def V_s : ℝ := 40  -- speed of slower train in km/hr

-- Convert distances and durations to the required units
def D : ℝ := (L_s + L_f) / 1000  -- distance in kilometers
def T_hours : ℝ := T / 3600  -- time in hours

-- Calculate the relative speed
def V_r : ℝ := D / T_hours  -- V_r is relative speed in km/hr

-- Define the speed of the faster train we need to prove
def V_f : ℝ := V_r - V_s

theorem speed_of_faster_train :
  V_f = 60.008 := by
  sorry

end speed_of_faster_train_l137_137383


namespace six_digit_number_divisibility_l137_137460

theorem six_digit_number_divisibility : 
  (∃ X : ℕ, 100 ≤ X ∧ X < 1000 ∧ (523000 + X) % 504 = 0) :=
by
  use 152
  have : 523000 + 152 = 523152 := by norm_num
  have : 523152 % 504 = 0 := by norm_num
  exact ⟨152, by norm_num, by norm_num, by norm_num⟩,

  use 656
  have : 523000 + 656 = 523656 := by norm_num
  have : 523656 % 504 = 0 := by norm_num
  exact ⟨656, by norm_num, by norm_num, by norm_num⟩,

end six_digit_number_divisibility_l137_137460


namespace power_function_decreasing_at_point_l137_137558

noncomputable def is_power_function_decreasing (f : ℝ → ℝ) : Prop :=
∀ x y, (0 < x ∧ 0 < y ∧ x < y) → (f x > f y)

theorem power_function_decreasing_at_point (α : ℝ) :
  (∀ x, f x = x^α) → f 2 = (2:ℝ)^α →
  (f 2 = (sqrt 2 / 2)) →
  is_power_function_decreasing f :=
by
  sorry

end power_function_decreasing_at_point_l137_137558


namespace negation_of_proposition_l137_137115

def equilateral (T : Type) [IsTriangle T] : Prop := sorry

theorem negation_of_proposition :
  (∃ T : Type, IsTriangle T ∧ equilateral T) →
  (∀ T : Type, IsTriangle T → ¬ equilateral T) :=
sorry

end negation_of_proposition_l137_137115


namespace probability_complement_l137_137709

theorem probability_complement : 
  (P_having_quiz : ℚ) (P_not_having_quiz : ℚ) 
  (h : P_having_quiz = 5 / 8) :
  P_not_having_quiz = 1 - P_having_quiz → P_not_having_quiz = 3 / 8 :=
by
  intros
  rw h
  simp
  sorry 

end probability_complement_l137_137709


namespace tayzia_tip_percentage_l137_137320

theorem tayzia_tip_percentage :
  let women's_haircut_cost : ℕ := 48
  let children's_haircut_cost : ℕ := 36
  let number_of_daughters : ℕ := 2
  let tip_amount : ℕ := 24
  let total_haircut_cost := women's_haircut_cost + number_of_daughters * children's_haircut_cost
  let tip_percentage := (tip_amount.to_rat / total_haircut_cost.to_rat) * 100
  tip_percentage = 20 := 
by
  let women's_haircut_cost : ℕ := 48
  let children's_haircut_cost : ℕ := 36
  let number_of_daughters : ℕ := 2
  let tip_amount : ℕ := 24
  let total_haircut_cost := women's_haircut_cost + number_of_daughters * children's_haircut_cost
  let tip_percentage := (tip_amount.to_rat / total_haircut_cost.to_rat) * 100
  sorry

end tayzia_tip_percentage_l137_137320


namespace find_equation_with_new_roots_l137_137562

variable {p q r s : ℝ}

theorem find_equation_with_new_roots 
  (h_eq : ∀ x, x^2 - p * x + q = 0 ↔ (x = r ∧ x = s))
  (h_r_nonzero : r ≠ 0)
  (h_s_nonzero : s ≠ 0)
  : 
  ∀ x, (x^2 - ((q^2 + 1) * (p^2 - 2 * q) / q^2) * x + (q + 1/q)^2) = 0 ↔ 
       (x = r^2 + 1/(s^2) ∧ x = s^2 + 1/(r^2)) := 
sorry

end find_equation_with_new_roots_l137_137562


namespace polynomial_remainder_l137_137838

theorem polynomial_remainder :
  let f := (λ x : ℝ, x^5 - 2 * x^4 + 3 * x^3 + 4)
  let g := (λ x : ℝ, x^2 - 4 * x + 6)
  let remainder := (λ x : ℝ, 2 * x - 44)
  ∀ x : ℝ, ∃ q : ℝ → ℝ, f x = q x * g x + remainder x :=
begin
  sorry
end

end polynomial_remainder_l137_137838


namespace at_least_four_identical_differences_l137_137855

theorem at_least_four_identical_differences (a : Fin 20 → ℕ) (h1 : ∀ i j, i ≠ j → a i ≠ a j) (h2 : ∀ i, a i < 70) :
  ∃ d, ∃ l, l ≥ 4 ∧ (∃ i j, i ≠ j ∧ d = abs (a i - a j) ∧ (∃ S : Finset (Fin 20), S.card = l ∧ ∀ i j ∈ S, i ≠ j → abs (a i - a j) = d)) :=
by sorry

end at_least_four_identical_differences_l137_137855


namespace car_travel_speed_l137_137416

noncomputable def car_speed (v : ℝ) : Prop :=
let t120 := 1 / 120 in  -- Hours to travel 1 km at 120 km/h
let t120_sec := t120 * 3600 in  -- Convert time from hours to seconds
let t_v := t120_sec + 5 in  -- Time taken by car at speed v in seconds
let t_v_hours := t_v / 3600 in  -- Convert time back to hours
let v_computed := 1 / t_v_hours in  -- Speed in km/h
v_computed = 102.857

theorem car_travel_speed : ∃ v : ℝ, car_speed v :=
begin
  use 102.857,
  unfold car_speed,
  -- Further steps to prove would go here
  sorry
end

end car_travel_speed_l137_137416


namespace tasty_triples_sum_l137_137019

noncomputable def is_tasty_triple (a b c : ℕ) : Prop :=
a < b ∧ b < c ∧ Nat.lcm a b c ∣ (a + b + c - 1)

theorem tasty_triples_sum :
  (∑ (a b c : ℕ) in finset.filter (λ (t : ℕ × ℕ × ℕ), is_tasty_triple t.1 t.2.1 t.2.2) 
   (finset.range 100).product (finset.range 100).product (finset.range 100), (a + b + c)) = 44 :=
sorry

end tasty_triples_sum_l137_137019


namespace vertex_of_parabola_l137_137698

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, (x - 2)^2 + 1 = (x - h)^2 + k) ∧ (h = 2 ∧ k = 1) :=
by
  use 2, 1
  split
  · intro x
    ring
  · exact ⟨rfl, rfl⟩

end vertex_of_parabola_l137_137698


namespace unique_line_through_point_5_4_with_conditions_l137_137201

-- Define the problem in Lean 4
theorem unique_line_through_point_5_4_with_conditions :
  ∃! (a b : ℕ), prime a ∧ composite b ∧
                (∃ k : Rational, (k * a = 5) ∧ (k * b = 4)) := 
sorry

end unique_line_through_point_5_4_with_conditions_l137_137201


namespace leap_years_count_l137_137430

theorem leap_years_count :
  let years := (2000:List ℕ).upto 5000 100
  let is_leap_year (y : ℕ) := y % 1000 = 300 ∨ y % 1000 = 700
  (years.filter is_leap_year).length = 6 :=
by
  sorry

end leap_years_count_l137_137430


namespace sin_tan_Q_right_triangle_l137_137608

theorem sin_tan_Q_right_triangle (P Q R : Type) [triangle : Triangle P Q R]
  (h_angle_R : angle Q R P = 90)
  (h_eq : 3 * real.sin (angle P Q R) = 4 * real.cos (angle P Q R)) :
  real.sin (angle P Q R) = 4 / 5 ∧ real.tan (angle P Q R) = 4 / 3 :=
by
  sorry

end sin_tan_Q_right_triangle_l137_137608


namespace prob_not_equal_genders_l137_137272

noncomputable def probability_more_grandsons_or_granddaughters : ℚ :=
1 - ((Nat.choose 12 6 : ℚ) / (2^12))

theorem prob_not_equal_genders {n : ℕ} (hn : n = 12)
  (equal_prob : ∀ i, i < n → i ≥ 0 → (Nat.of_num 1 / Nat.of_num 2 : ℚ) = 1 / 2) :
  probability_more_grandsons_or_granddaughters = 793 / 1024 := by
  have hn_pos : 0 < n := by linarith
  have hij : _hid := sorry
  sorry

end prob_not_equal_genders_l137_137272


namespace solve_for_z_l137_137595

open Complex

theorem solve_for_z (z : ℂ) (h : 2 * z * I = 1 + 3 * I) : 
  z = (3 / 2) - (1 / 2) * I :=
by
  sorry

end solve_for_z_l137_137595


namespace find_position_vector_l137_137233

variable (C D Q : Type)
variable [AddCommGroup Q] [VectorSpace ℚ Q]
variables (CQ : Q) (QD : Q)
variables (CVec DVec : Q) (QC QD CD : ℚ)

def is_point_on_segment (CQ QD CD : ℚ) : Prop :=
  CQ + QD = CD

def position_vector (t u : ℚ) (CVec DVec: Q) : Q :=
  t • CVec + u • DVec

theorem find_position_vector
  (CVec DVec : Q) (CQ QD : ℚ)
  (h_ratio : CQ / QD = 3 / 5) :
  ∃ t u, position_vector t u CVec DVec = (5 / 8) • CVec + (3 / 8) • DVec ∧ 
    t = 5 / 8 ∧ u = 3 / 8 :=
  sorry

end find_position_vector_l137_137233


namespace gcd_228_1995_l137_137386

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := 
by
  sorry

end gcd_228_1995_l137_137386


namespace linear_function_passing_origin_l137_137600

theorem linear_function_passing_origin (m : ℝ) :
  (∃ (y x : ℝ), y = -2 * x + (m - 5) ∧ y = 0 ∧ x = 0) → m = 5 :=
by
  sorry

end linear_function_passing_origin_l137_137600


namespace arithmetic_sequence_solution_l137_137880

theorem arithmetic_sequence_solution
  (a d : ℤ)
  (h_sum : a + (a + d) + (a + 2d) = 9)
  (h_product : a * (a + d) = 6 * (a + 2d))
  : a = 4 ∧ a + d = 3 ∧ a + 2d = 2 :=
by
  -- Proof here
  sorry

end arithmetic_sequence_solution_l137_137880


namespace sqrt_xyz_sum_l137_137999

theorem sqrt_xyz_sum {x y z : ℝ} (h₁ : y + z = 24) (h₂ : z + x = 26) (h₃ : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end sqrt_xyz_sum_l137_137999


namespace syllogism_correct_l137_137486

-- Define that natural numbers are integers
axiom nat_is_int : ∀ (n : ℕ), ∃ (m : ℤ), m = n

-- Define that 4 is a natural number
axiom four_is_nat : ∃ (n : ℕ), n = 4

-- The syllogism's conclusion: 4 is an integer
theorem syllogism_correct : ∃ (m : ℤ), m = 4 :=
by
  have h1 := nat_is_int 4
  have h2 := four_is_nat
  exact h1

end syllogism_correct_l137_137486


namespace total_books_count_l137_137814

def Darla_books := 6
def Katie_books := Darla_books / 2
def Darla_Katie_combined_books := Darla_books + Katie_books
def Gary_books := 5 * Darla_Katie_combined_books
def total_books := Darla_books + Katie_books + Gary_books

theorem total_books_count :
  total_books = 54 :=
by
  simp [Darla_books, Katie_books, Darla_Katie_combined_books, Gary_books, total_books]
  sorry

end total_books_count_l137_137814


namespace lisa_photos_last_weekend_l137_137264

def photos_of_animals : ℕ := 10
def photos_of_flowers : ℕ := 3 * photos_of_animals
def photos_of_scenery : ℕ := photos_of_flowers - 10
def total_photos_this_week : ℕ := photos_of_animals + photos_of_flowers + photos_of_scenery
def photos_last_weekend : ℕ := total_photos_this_week - 15

theorem lisa_photos_last_weekend : photos_last_weekend = 45 :=
by
  sorry

end lisa_photos_last_weekend_l137_137264


namespace linear_eq_must_be_one_l137_137146

theorem linear_eq_must_be_one (m : ℝ) : (∀ x y : ℝ, (m + 1) * x + 3 * y ^ m = 5 → (m = 1)) :=
by
  intros x y h
  sorry

end linear_eq_must_be_one_l137_137146


namespace min_sum_distinct_positive_integers_l137_137525

theorem min_sum_distinct_positive_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (1 / a + 1 / b = k1 * (1 / c)) ∧ (1 / a + 1 / c = k2 * (1 / b)) ∧ (1 / b + 1 / c = k3 * (1 / a))) :
  a + b + c ≥ 11 :=
sorry

end min_sum_distinct_positive_integers_l137_137525


namespace intersect_length_inequality_l137_137235

-- Define the structure of an equilateral triangle.
structure EquilateralTriangle (V : Type) [InnerProductSpace ℝ V] :=
(A B C : V)
(is_equilateral : dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B)

-- Define a point P inside the triangle ABC.
structure PointInsideTriangle (V : Type) [InnerProductSpace ℝ V] (T : EquilateralTriangle V) :=
(P : V)
(is_inside : sorry) -- Placeholder for the condition that P is inside the triangle

-- Define intersection points A1, B1, C1 on the sides of the triangle.
structure IntersectPoints (V : Type) [InnerProductSpace ℝ V] (T : EquilateralTriangle V) (P : PointInsideTriangle V T) :=
(A1 B1 C1 : V)
(A1_on_BC : sorry) -- Placeholder for condition that A1 lies on segment BC
(B1_on_CA : sorry) -- Placeholder for condition that B1 lies on segment CA
(C1_on_AB : sorry) -- Placeholder for condition that C1 lies on segment AB

-- The main theorem to be proven.
theorem intersect_length_inequality (V : Type) [InnerProductSpace ℝ V]
  (T : EquilateralTriangle V) (P : PointInsideTriangle V T) (I : IntersectPoints V T P) :
  (dist I.A1 I.B1) * (dist I.B1 I.C1) * (dist I.C1 I.A1) ≥ (dist I.A1 T.B) * (dist I.B1 T.C) * (dist I.C1 T.A) :=
sorry

end intersect_length_inequality_l137_137235


namespace ellipse_focus_distance_l137_137865

theorem ellipse_focus_distance (m : ℝ) (a b c : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m + y^2 / 16 = 1)
  (focus_distance : ∀ P : ℝ × ℝ, ∃ F1 F2 : ℝ × ℝ, dist P F1 = 3 ∧ dist P F2 = 7) :
  m = 25 := 
  sorry

end ellipse_focus_distance_l137_137865


namespace hyperbola_eccentricity_l137_137857

theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : x₀^2 / a^2 - y₀^2 / b^2 = 1)
  (h₄ : a ≤ x₀ ∧ x₀ ≤ 2 * a)
  (h₅ : x₀ / a^2 * 0 - y₀ / b^2 * b = 1)
  (h₆ : - (a * a / (2 * b)) = 2) :
  (1 + b^2 / a^2 = 3) :=
sorry

end hyperbola_eccentricity_l137_137857


namespace asymptotes_hyperbola_l137_137906

noncomputable def hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) := 
  (x y : ℝ) -> x ^ 2 / (a ^ 2) - y ^ 2 / (b ^ 2) = 1

noncomputable def point_on_hyperbola (C : (x y : ℝ) → Prop) (x y : ℝ) : Prop := C x y

noncomputable def distance (P1 P2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

theorem asymptotes_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) 
  (C : (x y : ℝ) → Prop := hyperbola a b a_pos b_pos)
  (P : ℝ × ℝ) (on_hyperbola : point_on_hyperbola C P.1 P.2) 
  (F1 F2 : ℝ × ℝ) 
  (dist_foci : distance P F1 + distance P F2 = 6 * a)
  (angle_triangle : ∃ θ : ℝ, θ = real.pi / 6 ∧ 
    ∃ P1 P2 P3 : ℝ × ℝ, P1 = F1 ∧ P2 = F2 ∧ P3 = P →
    (∃ θ : ℝ, θ = real.pi / 6)) : 
  (∃ k : ℝ, k = real.sqrt 2 ∧ 
  (∀ x y : ℝ, y = k * x ∨ y = -k * x)) :=
sorry

end asymptotes_hyperbola_l137_137906


namespace equal_segments_medians_not_implies_regular_l137_137732

-- Definitions of regular tetrahedron and inscribed sphere along with properties
structure Tetrahedron :=
(vertices : fin 4 → ℝ × ℝ × ℝ)

def is_regular_tetrahedron (t : Tetrahedron) : Prop :=
sorry -- A detailed definition of regular tetrahedron goes here

def median_segments_within_inscribed_sphere_equal (t : Tetrahedron) : Prop :=
sorry -- A property that the medians segments within the inscribed sphere are equal

-- Main theorem statement
theorem equal_segments_medians_not_implies_regular (t : Tetrahedron) :
  median_segments_within_inscribed_sphere_equal(t) → ¬ is_regular_tetrahedron(t) :=
sorry

end equal_segments_medians_not_implies_regular_l137_137732


namespace can_buy_two_pies_different_in_both_l137_137174

structure Pie :=
  (filling : Type)
  (preparation : Type)

def apple : Type := unit
def cherry : Type := unit
def fried : Type := unit
def baked : Type := unit

def apple_fried : Pie := { filling := apple, preparation := fried }
def apple_baked : Pie := { filling := apple, preparation := baked }
def cherry_fried : Pie := { filling := cherry, preparation := fried }
def cherry_baked : Pie := { filling := cherry, preparation := baked }

def possible_pies : List Pie := [apple_fried, apple_baked, cherry_fried, cherry_baked]

theorem can_buy_two_pies_different_in_both 
  (available_pies : List Pie) 
  (h : available_pies.length ≥ 3) : 
  ∃ (p1 p2 : Pie), p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation := 
  sorry

end can_buy_two_pies_different_in_both_l137_137174


namespace girls_more_than_boys_l137_137711

/-- 
In a class with 42 students, where the ratio of boys to girls is 3:4, 
prove that there are 6 more girls than boys.
-/
theorem girls_more_than_boys (students total_students : ℕ) (boys girls : ℕ) (ratio_boys_girls : 3 * girls = 4 * boys)
  (total_students_count : boys + girls = total_students)
  (total_students_value : total_students = 42) : girls - boys = 6 :=
by
  sorry

end girls_more_than_boys_l137_137711


namespace number_of_valid_permutations_l137_137247

-- Define the problem in Lean
def num_valid_F_matrices (M : Matrix (Fin 2014) (Fin 2014) ℝ) : ℝ :=
  if M.det ≠ 0 then (nat.double_factorial 2013) ^ 2 else 0

-- Main theorem statement
theorem number_of_valid_permutations (M : Matrix (Fin 2014) (Fin 2014) ℝ) :
  M.det ≠ 0 → num_valid_F_matrices M = (nat.double_factorial 2013) ^ 2 :=
by {
  sorry,
}

end number_of_valid_permutations_l137_137247


namespace white_area_is_110_l137_137428

-- Conditions
def signHeight : ℝ := 8
def signWidth : ℝ := 20
def signArea : ℝ := signHeight * signWidth

def F_area : ℝ := (6 * 1) + 2 * (1 * 4) -- area of letter F
def O_area : ℝ := 4 * 4 - 4 -- area of one letter O boundary
def total_O_area : ℝ := 2 * O_area -- total area for two O's
def D_area : ℝ := 6 + (2 * Math.pi * (2 * 2) / 2) -- area of letter D

def total_black_area : ℝ := F_area + total_O_area + D_area

-- Prove the white area is approximately 110 square units
theorem white_area_is_110 : 
  abs (signArea - total_black_area - 110) < 1 := by
  sorry

end white_area_is_110_l137_137428


namespace work_problem_l137_137752

theorem work_problem (P Q R W t_q : ℝ) (h1 : P = Q + R) 
    (h2 : (P + Q) * 10 = W) 
    (h3 : R * 35 = W) 
    (h4 : Q * t_q = W) : 
    t_q = 28 := 
by
    sorry

end work_problem_l137_137752


namespace linear_function_through_point_l137_137833

def linear_function (a b : ℝ) (x : ℝ) : ℝ := a * x + b

theorem linear_function_through_point : 
    ∃ a : ℝ, (linear_function a 1 (-2) = 0) ∧ (linear_function a 1 = λ x, (1 / 2) * x + 1) :=
begin
  sorry
end

end linear_function_through_point_l137_137833


namespace possible_perimeter_sums_l137_137014

theorem possible_perimeter_sums (side_length : ℝ) (h : side_length = 1) : 
  ∃ S, (S = 8 ∨ S = 10) ∧ (∃ r₁ r₂ r₃ r₄ : ℝ × ℝ, 
  r₁.1 * r₁.2 + r₂.1 * r₂.2 + r₃.1 * r₃.2 + r₄.1 * r₄.2 = side_length ^ 2 ∧
  (2 * (r₁.1 + r₁.2) + 2 * (r₂.1 + r₂.2) + 2 * (r₃.1 + r₃.2) + 2 * (r₄.1 + r₄.2)) = S) :=
begin
  sorry
end

end possible_perimeter_sums_l137_137014


namespace time_addition_correct_l137_137222

theorem time_addition_correct :
    let A := (190 + 3) % 12, 
        B := (45: ℕ),
        C := (30: ℕ) in
    A + B + C = 76 :=
by
    sorry

end time_addition_correct_l137_137222


namespace average_of_two_numbers_l137_137849

theorem average_of_two_numbers {a b c d e f : ℕ}
  (h_set : {a, b, c, d, e, f} = {1871, 1998, 2023, 2030, 2114, 2128})
  (h_mean_four : (a + b + c + d) / 4 = 2015) :
  ((e + f) / 2) = 2052 := 
sorry

end average_of_two_numbers_l137_137849


namespace sum_of_min_max_values_of_f_l137_137363

noncomputable def f (x : Real) : Real :=
  Real.cos (2 * x) + 2 * Real.sin x

theorem sum_of_min_max_values_of_f :
  (Real.min (f (- π/2)) (f (π/3)) + Real.max (f (- π/2)) (f (π/3))) = -3 / 2 :=
by
  sorry

end sum_of_min_max_values_of_f_l137_137363


namespace greendale_points_l137_137295

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l137_137295


namespace lambda_range_l137_137570

theorem lambda_range (λ : ℝ) (h : ∀ n : ℤ, 1 ≤ n → (n + 1) ^ 2 + λ * (n + 1) - (n ^ 2 + λ * n) > 0) :
  λ > -3 :=
by
  sorry

end lambda_range_l137_137570


namespace percentage_bob_correct_l137_137804

def bob_water (corn_acres : ℕ) (cotton_acres : ℕ) (beans_acres : ℕ) : ℕ :=
  (corn_acres * 20) + (cotton_acres * 80) + (beans_acres * 40)

def bob := bob_water 3 9 12
def brenda := (6 * 20) + (7 * 80) + (14 * 40)
def bernie := (2 * 20) + (12 * 80)

def total_water_usage := bob + brenda + bernie

def percentage_bob := (bob.to_rat / total_water_usage.to_rat) * 100

theorem percentage_bob_correct : percentage_bob ≈ 36 := sorry

end percentage_bob_correct_l137_137804


namespace range_of_m_l137_137890

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → (m ≤ 2 ∧ m ≠ -2) :=
begin
  sorry
end

end range_of_m_l137_137890


namespace problem_1_problem_2_l137_137198

open nat

-- Defining the total sum of an arithmetic sequence
def sum_arith_seq (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Problem 1: Given the first box contains 11 grains of rice and the total sum is 351.
theorem problem_1 (a1 d : ℕ) (n : ℕ) (h1 : a1 = 11) (h2 : n = 9) (h3 : sum_arith_seq a1 d n = 351) : d = 7 :=
by
  sorry

-- Problem 2: Given the third box contains 23 grains of rice and the total sum is 351.
theorem problem_2 (a1 d : ℕ) (n : ℕ) (h1 : a1 + 2 * d = 23) (h2 : n = 9) (h3 : sum_arith_seq a1 d n = 351) : d = 8 :=
by
  sorry

end problem_1_problem_2_l137_137198


namespace total_earnings_proof_l137_137433

-- Definitions of the given conditions
def monthly_earning : ℕ := 4000
def monthly_saving : ℕ := 500
def total_savings_needed : ℕ := 45000

-- Lean statement for the proof problem
theorem total_earnings_proof : 
  (total_savings_needed / monthly_saving) * monthly_earning = 360000 :=
by
  sorry

end total_earnings_proof_l137_137433


namespace find_H_coordinate_l137_137377

-- Defining the vertices
def E : (ℝ × ℝ × ℝ) := (2, -3, 4)
def F : (ℝ × ℝ × ℝ) := (0, 5, -6)
def G : (ℝ × ℝ × ℝ) := (-2, 3, 4)

-- The definition of point H
def H : (ℝ × ℝ × ℝ) := (0, -5, 14)

-- Midpoint of a line segment
def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ( (A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2 )

-- The proof statement
theorem find_H_coordinate :
  (EFGH_is_a_parallelogram : E F G H) →
  midpoint E G = midpoint F H →
  H = (0, -5, 14) :=
by
  sorry

end find_H_coordinate_l137_137377


namespace tan_x_value_l137_137142

theorem tan_x_value (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2)
  (h₂ : (sin x) ^ 4 / 9 + (cos x) ^ 4 / 4 = 1 / 13) :
  tan x = 3 / 2 :=
sorry

end tan_x_value_l137_137142


namespace tree_planting_l137_137210

/-- The city plans to plant 500 thousand trees. The original plan 
was to plant x thousand trees per day. Due to volunteers, the actual number 
of trees planted per day exceeds the original plan by 30%. As a result, 
the task is completed 2 days ahead of schedule. Prove the equation. -/
theorem tree_planting
    (x : ℝ) 
    (hx : x > 0) : 
    (500 / x) - (500 / ((1 + 0.3) * x)) = 2 :=
sorry

end tree_planting_l137_137210


namespace distance_A_B1_l137_137213

/-- In a rectangular prism ABCD-A1B1C1D1 with AB = 5, BC = 4, and BB1 = 3, 
the distance between the points A and B1 is 12/5. -/
theorem distance_A_B1 (AB BC BB1 : ℝ) (hAB : AB = 5) (hBC : BC = 4) (hBB1 : BB1 = 3) : 
  dist A B1 = 12 / 5 :=
by
  sorry

end distance_A_B1_l137_137213


namespace graph_of_equation_is_hyperbola_l137_137808

theorem graph_of_equation_is_hyperbola (x y : ℝ):
  (x^2 - 9 * y^2 + 6 * x = 0) → ∃ a b h k : ℝ, ∀ x y : ℝ, (x + 3)^2 / 9 - y^2 = 1 :=
by
  sorry

end graph_of_equation_is_hyperbola_l137_137808


namespace largest_angle_of_scalene_triangle_l137_137195

-- Define the problem statement in Lean
theorem largest_angle_of_scalene_triangle (x : ℝ) (hx : x = 30) : 3 * x = 90 :=
by {
  -- Given that the smallest angle is x and x = 30 degrees
  sorry
}

end largest_angle_of_scalene_triangle_l137_137195


namespace projection_correct_l137_137515

noncomputable def projection_of_vector_on_line
  (v : ℝ × ℝ × ℝ)
  (d : ℝ × ℝ × ℝ)
: ℝ × ℝ × ℝ :=
  let dot_prod := v.1 * d.1 + v.2 * d.2 + v.3 * d.3,
      norm_sq := d.1 ^ 2 + d.2 ^ 2 + d.3 ^ 2 in
  (dot_prod / norm_sq) • d

theorem projection_correct :
  projection_of_vector_on_line (4, -1, 3) (3, 1, 2) = (51/14, 17/14, 17/7) :=
by
  sorry

end projection_correct_l137_137515


namespace power_of_two_l137_137541

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def factorial (n : ℕ) : ℕ :=
  if h: n = 0 then 1 else n * factorial (n - 1)

def prod_k (n : ℕ) : ℚ := (List.range n).map (λ k, factorial k / factorial (2 * k)).prod

theorem power_of_two (n : ℕ) (A B : ℕ)
  (hA_pos : 0 < A) (hB_pos : 0 < B) (h_coprime : is_coprime A B)
  (h_eq : (B: ℚ) / A = factorial (n * (n + 1) / 2) * prod_k n) :
  ∃ k : ℕ, A = 2^k :=
sorry

end power_of_two_l137_137541


namespace area_of_rectangle_inscribed_in_semicircle_l137_137678

-- Definitions
def semicircle_diameter := 34
def FD : ℝ := 9
def DA : ℝ := 16
def CD := sqrt (FD * (semicircle_diameter - FD))

-- Statement to prove
theorem area_of_rectangle_inscribed_in_semicircle (h1 : DA = 16) (h2 : FD = 9) (h3 : semicircle_diameter = 34) :
  DA * CD = 240 :=
by
  sorry

end area_of_rectangle_inscribed_in_semicircle_l137_137678


namespace total_weekly_cost_l137_137633

-- Define conditions
def rabbit_weeks : ℕ := 5
def parrot_weeks : ℕ := 3
def total_spent : ℕ := 114
def rabbit_weekly_cost : ℕ := 12

-- Define variables for solution
def T : ℕ
def P : ℕ

-- Lean theorem to prove the total weekly cost 
theorem total_weekly_cost :
  (rabbit_weekly_cost * rabbit_weeks + P * parrot_weeks = total_spent) →
  T = rabbit_weekly_cost + P →
  T = 30 :=
by
  sorry

end total_weekly_cost_l137_137633


namespace smallest_n_exceed_sum_l137_137519

theorem smallest_n_exceed_sum :
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  ∃ (n : ℕ), (1 / 2) * (1 - 1 / 3 ^ n) > 40 / 81 ∧
  ∀ (m : ℕ), m < n → (1 / 2) * (1 - 1 / 3 ^ m) ≤ 40 / 81 :=
by
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  have h1 : ∃ n, (1 / 2) * (1 - 1 / 3 ^ n) > 40 / 81 :=
    sorry -- prove that such an n exists
  have h2 : ∀ m, (1 / 2) * (1 - 1 / 3 ^ m) ≤ 40 / 81 → m ≥ 5 :=
    sorry -- prove that for all m < 5, the inequality does not hold
  exact ⟨5, by simp [h1], by simp [h2]⟩ 

end smallest_n_exceed_sum_l137_137519


namespace log_increasing_interval_l137_137149

noncomputable def is_increasing_on_interval (a : ℝ) (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

theorem log_increasing_interval (a : ℝ) :
  1 < a ∧ a ≤ 4 ↔ is_increasing_on_interval a (λ x : ℝ, log a (4 * x + a / x)) (set.Icc 1 2) :=
sorry

end log_increasing_interval_l137_137149


namespace sameFunctionOptionD_l137_137793

section FunctionsEquality

variable (x : ℝ)

def funcA_f := fun (x : ℝ) => if x ≠ 0 then x^2 else 0
def funcA_g := fun (x : ℝ) => x^2

def funcB_f := fun (x : ℝ) => x - 1
def funcB_g := fun (x : ℝ) => if x ≠ -1 then (x^2 - 1) / (x + 1) else 0

def funcC_f := fun (x : ℝ) => abs x
def funcC_g := fun (x : ℝ) => x

def funcD_f := fun (x : ℝ) => x + (1 / x)
def funcD_g := fun (x : ℝ) => if x ≠ 0 then (x^2 + 1) / x else 0

theorem sameFunctionOptionD :
  (∀ x : ℝ, funcD_f x = funcD_g x) ∧
  (¬ ∀ x : ℝ, funcA_f x = funcA_g x) ∧
  (¬ ∀ x : ℝ, funcB_f x = funcB_g x) ∧
  (¬ ∀ x : ℝ, funcC_f x = funcC_g x) :=
  by
  sorry

end FunctionsEquality

end sameFunctionOptionD_l137_137793


namespace rod_mass_is_rho_l_rod_moment_of_inertia_is_rho_l_cube_l137_137038

noncomputable def rod_mass (ρ : ℝ) (l : ℝ) : ℝ := ∫ x in 0..l, ρ
noncomputable def rod_moment_of_inertia (ρ : ℝ) (l : ℝ) : ℝ := ∫ x in 0..l, x^2 * ρ

theorem rod_mass_is_rho_l (ρ l : ℝ) : rod_mass ρ l = ρ * l :=
sorry

theorem rod_moment_of_inertia_is_rho_l_cube (ρ l : ℝ) : rod_moment_of_inertia ρ l = (ρ * l^3) / 3 :=
sorry

end rod_mass_is_rho_l_rod_moment_of_inertia_is_rho_l_cube_l137_137038


namespace side_length_of_square_equals_circumference_of_circle_l137_137355

theorem side_length_of_square_equals_circumference_of_circle :
  let radius := 3 in
  let circumference := 2 * Real.pi * radius in
  let perimeter_of_square x := 4 * x in
  ∃ x : ℝ, perimeter_of_square x = circumference ∧ Real.Approx x 4.71 0.01 :=
by
  let radius := 3
  let circumference := 2 * Real.pi * radius
  let perimeter_of_square := λ x : ℝ, 4 * x
  use (3 * Real.pi / 2)
  have h : 4 * (3 * Real.pi / 2) = circumference, from sorry,
  have approx : Real.Approx (3 * Real.pi / 2) 4.71 0.01, from sorry,
  exact ⟨h, approx⟩

end side_length_of_square_equals_circumference_of_circle_l137_137355


namespace range_of_t_l137_137847

theorem range_of_t (a b t : ℝ) (h1 : a * (-1)^2 + b * (-1) + 1 / 2 = 0)
    (h2 : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x^2 + b * x + 1 / 2))
    (h3 : t = 2 * a + b) : 
    -1 < t ∧ t < 1 / 2 :=
  sorry

end range_of_t_l137_137847


namespace limit_of_f_l137_137496

noncomputable def f (x : ℝ) : ℝ :=
  ( (x + 1) / (2 * x) ) ^ ( log (x + 2) / log (2 - x) )

theorem limit_of_f :
  filter.tendsto f (nhds 1) (nhds (sqrt 3)) :=
by
  sorry

end limit_of_f_l137_137496


namespace car_speed_difference_l137_137469

theorem car_speed_difference :
  ∀ (d : ℝ) (v_R : ℝ) (Δt : ℝ) (v_P : ℝ),
  d = 800 ∧
  v_R = 58.4428877022476 ∧
  Δt = 2 ∧
  t_R = d / v_R ∧
  t_P = t_R - Δt ∧
  t_P = d / v_P →
  v_P - v_R ≈ 9.983 := 
by {
  intros,
  let t_R := d / v_R,
  let t_P := t_R - Δt,
  have h1 : t_P = d / v_P := sorry,
  have h2 : v_P = d / t_P := sorry,
  have h3 : v_P - v_R ≈ 9.983 := sorry,
  exact h3
}

end car_speed_difference_l137_137469


namespace snow_leopards_arrangement_l137_137663

theorem snow_leopards_arrangement :
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  end_positions * factorial_six = 1440 :=
by
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  show end_positions * factorial_six = 1440
  sorry

end snow_leopards_arrangement_l137_137663


namespace isosceles_triangle_possible_x_l137_137361

theorem isosceles_triangle_possible_x (x : ℝ) (h_pos : 0 < x) (h_lt : x < 90)
  (h_sides : ∃ b : ℝ, ∃ h_dep : b = sin x,
                       (sin x = sin x) ∧ (sin x = sin x) ∧ (b = sin 7 * x)
                       ∧ (2 * x < 180) ∧ (b + b = 180 \cos (2 * x))) :
  x = 36 ∨ x = 40 := by
  sorry

end isosceles_triangle_possible_x_l137_137361


namespace gnomes_and_ponies_l137_137303

theorem gnomes_and_ponies (g p : ℕ) (h1 : g + p = 15) (h2 : 2 * g + 4 * p = 36) : g = 12 ∧ p = 3 :=
by
  sorry

end gnomes_and_ponies_l137_137303


namespace length_of_AB_l137_137624

open Classical

variable (A B C D E F : Type)
variable [OrderedRing A]
variable [AddGroup B]
variable [AddGroup C]
variable [AddGroup D]
variable [AddGroup E]
variable [AddGroup F]
variables (AD BD EC BC AFC DBEF : A)
variable (AD_eq_2BD : AD = 2 * BD)
variable (AD_eq_EC : AD = EC)
variable (BC_val : BC = 18)
variable (area_eq : AFC = DBEF)
variable (area_ratio : AFC / DBEF = 2)

theorem length_of_AB (AB : A) :
  AB = 9 :=
  sorry

end length_of_AB_l137_137624


namespace sequence_fibonacci_ineq_l137_137255

noncomputable def floor (x : ℝ) : ℤ := int.floor x

def fibonacci (n : ℕ) : ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

/-- Define the sequence according to the problem -/
def sequence (x : ℝ) : ℕ → ℝ
| 0       := x
| (n + 1) :=
  if h : sequence n = 0 then 0
  else (1 / sequence n) - (floor (1 / sequence n) : ℝ)

/-- Prove the inequality involving the sum of the sequence terms and the fractions of the Fibonacci sequence -/
theorem sequence_fibonacci_ineq {x : ℝ} (h : 0 ≤ x ∧ x < 1) (n : ℕ) :
  (finset.range n).sum (λ k, sequence x k) < (finset.range n).sum (λ k, (fibonacci k : ℝ) / fibonacci (k + 1)) :=
sorry

end sequence_fibonacci_ineq_l137_137255


namespace maximal_size_l137_137542

-- Define the set of divisors D_n.
def D_n (n : ℕ) : set ℕ := {d | ∃ α β γ : ℕ, α ≤ n ∧ β ≤ n ∧ γ ≤ n ∧ d = 2^α * 3^β * 5^γ}

-- Define what it means for a subset S of D_n to be maximal and no element divides another element.
def valid_subset (S : set ℕ) (n : ℕ) : Prop :=
  S ⊆ D_n n ∧ ∀ a b ∈ S, a ≠ b → ¬(a ∣ b ∨ b ∣ a)

noncomputable def max_size_subset (n : ℕ) : ℕ :=
  ⌊ (3 * (n + 1)^2 + 1) / 4 ⌋

theorem maximal_size (n : ℕ) (h : 0 < n) :
  ∃ S : set ℕ, valid_subset S n ∧ S.size = max_size_subset n := 
sorry

end maximal_size_l137_137542


namespace conjugate_of_z_l137_137875

noncomputable def z : ℂ := (3 + 1 * complex.I) / (1 - 2 * complex.I)

theorem conjugate_of_z :
  complex.conj z = (1 / 5) - (7 / 5) * complex.I :=
by
  -- This is where the proof goes, but as required, we use sorry to skip the proof
  sorry

end conjugate_of_z_l137_137875


namespace problem_statement_l137_137587

theorem problem_statement (x y : ℝ) (h₁ : 2.5 * x = 0.75 * y) (h₂ : x = 20) : y = 200 / 3 := by
  sorry

end problem_statement_l137_137587


namespace hockey_games_per_month_l137_137727

theorem hockey_games_per_month {
  total_games : ℕ,
  months_in_season : ℕ
} (h1 : total_games = 182) (h2 : months_in_season = 14) :
  total_games / months_in_season = 13 := by
  sorry

end hockey_games_per_month_l137_137727


namespace length_segment_AB_l137_137429

noncomputable def line_param_x (t : ℝ) : ℝ := 1 + t
noncomputable def line_param_y (t : ℝ) : ℝ := -2 + t
def ellipse_eq (x y : ℝ) : Prop := x^2 + 2 * y^2 = 8

theorem length_segment_AB :
  (∀ t: ℝ, let x := line_param_x t in let y := line_param_y t in ellipse_eq x y) →
  (x1 x2 y1 y2 : ℝ)
  (H1 : x1 = 1 + u)
  (H2 : y1 = -2 + u)
  (H3 : x2 = 1 + v)
  (H4 : y2 = -2 + v)
  (H5 : ellipse_eq x1 y1)
  (H6 : ellipse_eq x2 y2)
  (H7 : x1 ≠ x2)
  → (x0 y0 : ℝ)
  (H8 : y0 = -2 + x0)
  (H9 : y0 = -2 + x1)
  (H10 : y0 = -2 + x2)
  (H11 : ellipse_eq x0 x0)
  (H12 : ellipse_eq x1 y1)
  (H13 : ellipse_eq x2 y2)
  (H14 : x1 ≠ x2)
  → (|AB| = (sqrt(1 + 1^2) * (x1 - x2))) = ∀ x y : ℝ, ellipse_eq x y → ∃ t :  ℝ, y = -2 + x :=
sorry

end length_segment_AB_l137_137429


namespace length_of_second_train_l137_137414

theorem length_of_second_train (l1 : ℕ) (s1_kmph : ℕ) (s2_kmph : ℕ) (t : ℕ) (l2 : ℕ) : 
  l1 = 250 →
  s1_kmph = 120 →
  s2_kmph = 80 →
  t = 9 →
  l2 = 249.95 :=
begin
  sorry
end

end length_of_second_train_l137_137414


namespace value_of_f10_l137_137901

def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ (x y : ℝ), f(x) + f(2 * x + y) + 5 * x * y = f(3 * x - y) + 2 * x ^ 2 + 1

theorem value_of_f10 : f 10 = -49 :=
by
  sorry

end value_of_f10_l137_137901


namespace function_properties_l137_137258

-- Definitions based on conditions
def f (x : ℝ) : ℝ := log (1 + x) - log (1 - x)

lemma domain_of_f (x : ℝ) : -1 < x ∧ x < 1 → true := sorry

-- Statement of the problem to be proved
theorem function_properties :
  (∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x) ∧ (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f x < f y) :=
sorry

end function_properties_l137_137258


namespace circle_equation_l137_137204

theorem circle_equation
  (C : ℝ × ℝ)
  (center_in_first_quadrant : C.1 > 0 ∧ C.2 > 0)
  (intersects_x_axis_at_A_B : ∀ x, (x = 1 ∨ x = 3) → ∃ y, (y = 0 ∧ (C.1 - x)^2 + C.2^2 = (C.1 - 2)^2 + C.2^2))
  (tangent_line : ∀ x y, x - y + 1 = 0 → let d := abs ((C.1 - y + 1) / (real.sqrt 2)) in d = real.sqrt ((2-1)^2 + 1^2)) :
    ∃ h k r, (∃ center_r : (h = 2 ∧ k = 1) ∧ r = real.sqrt 2) ∧  ((h - 2)^2 + (k - 1)^2 = r^2 ∧ (C = ⟨h, k⟩) ∧ (r = real.sqrt 2)) :=
begin
  sorry
end

end circle_equation_l137_137204


namespace eleanor_distance_between_meetings_l137_137053

-- Conditions given in the problem
def track_length : ℕ := 720
def eric_time : ℕ := 4
def eleanor_time : ℕ := 5
def eric_speed : ℕ := track_length / eric_time
def eleanor_speed : ℕ := track_length / eleanor_time
def relative_speed : ℕ := eric_speed + eleanor_speed
def time_to_meet : ℚ := track_length / relative_speed

-- Proof task: prove that the distance Eleanor runs between consective meetings is 320 meters.
theorem eleanor_distance_between_meetings : eleanor_speed * time_to_meet = 320 := by
  sorry

end eleanor_distance_between_meetings_l137_137053


namespace value_of_x_if_additive_inverses_l137_137144

theorem value_of_x_if_additive_inverses (x : ℝ) 
  (h : 4 * x - 1 + (3 * x - 6) = 0) : x = 1 := by
sorry

end value_of_x_if_additive_inverses_l137_137144


namespace find_eccentricity_l137_137454

noncomputable def ellipse := {a b x y : ℝ // a > 0 ∧ b > 0 ∧ a > b ∧ (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1)}
noncomputable def intersect_line := {a b x1 x2 y : ℝ // ellipse a b x1 y ∧ ellipse a b x2 y}
noncomputable def isosceles_right := {a b f x1 x2 y : ℝ // intersect_line a b x1 x2 y ∧ ∃A B F, is_isosceles_right_triangle A B F}

theorem find_eccentricity (a b : ℝ) (e : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (c : ℝ) (h₄ : a^2 = b^2 + c^2) (y : ℝ) (h₅ : y = 2 * c)
  (h₆ : e^2 = c^2 / a^2) (h₇ : ∀ (x₁ x₂ : ℝ), intersect_line a b x₁ x2 y → isosceles_right a b x₁ x₂ f y) :
  e = sqrt(2) - 1 := sorry

end find_eccentricity_l137_137454


namespace domain_shift_l137_137597

theorem domain_shift (f : ℝ → ℝ) (h : set.Icc 0 1 ⊆ {x | f (x + 2) ∈ set.univ}) : 
  set.Icc (-2 : ℝ) (-1) ⊆ {x | f x ∈ set.univ} :=
sorry

end domain_shift_l137_137597


namespace lambda_equals_neg3_l137_137122

def m (λ : ℝ) : ℝ × ℝ := (λ + 1, 1)

def n (λ : ℝ) : ℝ × ℝ := (λ + 2, 2)

theorem lambda_equals_neg3 (λ : ℝ) (h : ((m λ).fst + (n λ).fst, (m λ).snd + (n λ).snd) = (2 * λ + 3, 3)
  ∧ ((m λ).fst - (n λ).fst, (m λ).snd - (n λ).snd) = (-1, -1)
  ∧ ((2 * λ + 3, 3) . fst * (-1, -1) . fst + ( (2 * λ + 3, 3) . snd * (-1, -1) . snd) = 0) ):
  λ = -3 := by
  sorry

end lambda_equals_neg3_l137_137122


namespace solve_quadratic_inequality_l137_137153

theorem solve_quadratic_inequality:
(forall x : ℝ,  (x - 3) * (x - 4) < 0 → (3 < x ∧ x < 4)) →
(forall x : ℝ,  (x - 3) * (x - 4) < 0 ↔ x^2 - 7 * x + 12 < 0) →
set_of (λ x : ℝ, 12*x^2 - 7*x + 1 > 0) = (set_of (λ x, x < 1/4) ∪ set_of (λ x, x > 1/3)) :=
by {
    intros H1 H2,
    sorry
}

end solve_quadratic_inequality_l137_137153


namespace anti_Pascal_impossible_l137_137221

theorem anti_Pascal_impossible (n : ℕ) (h1 : n = 2018) : 
  ¬ ∃ (f : ℕ → ℕ), (∀ k, k ∈ finset.range (∑ k in finset.range n.succ, k.succ) ↔ (f k ∈ finset.range (∑ k in finset.range n, k.succ)) 
  ∧ ∀ i h2 j h3, f i = |f (i + 1) - f (i + 2)) :
  sorry

end anti_Pascal_impossible_l137_137221


namespace transformation_result_l137_137356

noncomputable def rotate_y (θ : ℝ) : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ :=
λ p, let (x, y, z) := p in 
  (x * real.cos θ + z * real.sin θ, y, -x * real.sin θ + z * real.cos θ)

noncomputable def rotate_z (θ : ℝ) : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ :=
λ p, let (x, y, z) := p in 
  (x * real.cos θ - y * real.sin θ, x * real.sin θ + y * real.cos θ, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let (x, y, z) := p in (x, -y, z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let (x, y, z) := p in (-x, y, z)

theorem transformation_result :
  let p := (2, 3, 2)
  let p1 := rotate_y (real.pi / 2) p
  let p2 := reflect_xz p1
  let p3 := rotate_z (real.pi / 2) p2
  let p4 := reflect_yz p3
  p4 = (-3, 2, -2) :=
by simp [rotate_y, rotate_z, reflect_xz, reflect_yz]; sorry

end transformation_result_l137_137356


namespace perp_necessary_and_sufficient_condition_l137_137648

variables {α β : Type} [plane α] [plane β] {m n : Type} [line m] [line n]

-- Assume the conditions given in the problem
variables (h_diff_planes : α ≠ β)
variables (h_diff_lines : m ≠ n)
variables (h_n_perp_alpha : n ⟂ α)
variables (h_n_perp_beta : n ⟂ β)

-- Define the goal as a theorem
theorem perp_necessary_and_sufficient_condition :
  (m ⟂ α ↔ m ⟂ β) :=
sorry

end perp_necessary_and_sufficient_condition_l137_137648


namespace cottonCandyToPopcornRatio_l137_137417

variable (popcornEarningsPerDay : ℕ) (netEarnings : ℕ) (rentCost : ℕ) (ingredientCost : ℕ)

theorem cottonCandyToPopcornRatio
  (h_popcorn : popcornEarningsPerDay = 50)
  (h_net : netEarnings = 895)
  (h_rent : rentCost = 30)
  (h_ingredient : ingredientCost = 75)
  (h : ∃ C : ℕ, 5 * C + 5 * popcornEarningsPerDay - rentCost - ingredientCost = netEarnings) :
  ∃ r : ℕ, r = 3 :=
by
  sorry

end cottonCandyToPopcornRatio_l137_137417


namespace simple_interest_is_50_l137_137601
-- Define the principal amount P
variable (P : ℝ)

-- Define the simple interest calculation
def simple_interest (P r t : ℝ) : ℝ := P * r * t

-- Define the compound interest calculation
def compound_interest (P r t : ℝ) : ℝ := P * (1 + r)^t - P

-- Define the given conditions
def rate (r : ℝ) : Prop := r = 0.05
def time (t : ℝ) : Prop := t = 2
def compound_interest_value (CI : ℝ) : Prop := CI = 51.25

-- The theorem to be proved
theorem simple_interest_is_50 (P : ℝ) (r : ℝ) (t : ℝ) (CI : ℝ)
  (h1 : rate r) (h2 : time t) (h3 : compound_interest_value CI) :
  simple_interest P 0.05 2 = 50 := sorry

end simple_interest_is_50_l137_137601


namespace count_zero_sequences_l137_137647

theorem count_zero_sequences : 
    ∃! (T : Set (ℤ × ℤ × ℤ)), (∀ (b1 b2 b3 : ℤ), (b1, b2, b3) ∈ T → 1 ≤ b1 ∧ b1 ≤ 15 ∧ 1 ≤ b2 ∧ b2 ≤ 15 ∧ 1 ≤ b3 ∧ b3 ≤ 15) ∧ ∑ s in T, ∃ n ≥ 4, 
    let b : List ℤ := ([b1, b2, b3] ++ List.range (n - 3)).zipWith (λ b_i b_j b_k => b_j * |b_i - b_k|) in b.nth (n - 1) = 0 := 
    15 :=
sorry

end count_zero_sequences_l137_137647


namespace symmetrical_point_l137_137618

-- Definition of symmetry with respect to the x-axis
def symmetrical (x y: ℝ) : ℝ × ℝ := (x, -y)

-- Coordinates of the original point A
def A : ℝ × ℝ := (-2, 3)

-- Coordinates of the symmetrical point
def symmetrical_A : ℝ × ℝ := symmetrical (-2) 3

-- The theorem we want to prove
theorem symmetrical_point :
  symmetrical_A = (-2, -3) :=
by
  -- Provide the proof here
  sorry

end symmetrical_point_l137_137618


namespace unique_prime_pair_l137_137072

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem unique_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ 
  p^2 + 2 * p * q^2 + 1 = Prime 41 ∧ ∀ (p' q' : ℕ), is_prime p' ∧ is_prime q' ∧ 
  p'^2 + 2 * p' * q'^2 + 1 = Prime 41 → (p', q') = (2, 3) :=
begin
  existsi 2,
  existsi 3,
  split,
  { exact Nat.prime_two },
  { split,
    { exact Nat.prime_three },
    { split,
      { exact Prime 41 },
      { intros p' q' h_prime_p' h_prime_q' h_eq_41,
        sorry }  -- Proof omitted
    }
  }
end

end unique_prime_pair_l137_137072


namespace find_a_in_expansion_geometric_mean_l137_137842

theorem find_a_in_expansion_geometric_mean :
  let a := (25:ℚ) / (9:ℚ) in
  let binom := Nat.choose in
  ∀ ax : ℚ, 
    let coeff_x2 := binom 7 5 * ax^2 in
    let coeff_x3 := binom 7 4 * ax^3 in
    let coeff_x5 := binom 7 2 * ax^5 in
    (coeff_x3^2 = coeff_x2 * coeff_x5) → 
    ax = a :=
begin
  intros,
  sorry
end

end find_a_in_expansion_geometric_mean_l137_137842


namespace gcd_228_1995_l137_137387

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := 
by
  sorry

end gcd_228_1995_l137_137387


namespace fractional_equation_solution_l137_137886

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l137_137886


namespace exist_two_pies_differing_in_both_l137_137161

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l137_137161


namespace common_root_quadratics_l137_137843

theorem common_root_quadratics (r : ℝ)
  (h : ∃ s : ℝ, s^2 + (r - 1) * s + 6 = 0 ∧ s^2 + (2 * r + 1) * s + 22 = 0) :
  (∃ (a b : ℕ), (a = 12) ∧ (b = 5) ∧ (a.gcd b = 1) ∧ (100 * a + b = 1205)) :=
by {
  use [12, 5],
  split,
  { refl },
  split,
  { refl },
  split,
  { norm_num },
  { norm_num }
}

end common_root_quadratics_l137_137843


namespace distinct_three_digit_numbers_l137_137132

theorem distinct_three_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in
  ∃ (count : ℕ), count = 60 := by
  sorry

end distinct_three_digit_numbers_l137_137132


namespace frac_sum_eq_one_l137_137528

variable {x y : ℝ}

theorem frac_sum_eq_one (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x) + (1 / y) = 1 :=
by sorry

end frac_sum_eq_one_l137_137528


namespace ellipse_eccentricity_and_projection_l137_137546

-- Definition of the ellipse and given condition
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def a : ℝ := 3
def foci_distance (d : ℝ) : Prop := d = 6
def foci_coordinates (c : ℝ) : Prop := c = sqrt 5
def eccentricity (e : ℝ) : Prop := e = sqrt 5 / 3

-- Statement of the problem in Lean
theorem ellipse_eccentricity_and_projection :
  (∀ x y : ℝ, ellipse x y → 
  (∃ e : ℝ, eccentricity e) ∧ 
  (∃ Q : ℝ × ℝ, 
    let P := (√5, Q.2) in
    P.1 = √5 ∧
    (P.fst - foci_coordinates) ^ 2 + P.snd ^ 2 = 1)) :=
sorry

end ellipse_eccentricity_and_projection_l137_137546


namespace hockey_games_per_month_calculation_l137_137726

-- Define the given conditions
def months_in_season : Nat := 14
def total_hockey_games : Nat := 182

-- Prove the number of hockey games played each month
theorem hockey_games_per_month_calculation :
  total_hockey_games / months_in_season = 13 := by
  sorry

end hockey_games_per_month_calculation_l137_137726


namespace arccos_neg_half_l137_137474

theorem arccos_neg_half : ∃ θ ∈ Icc (0 : ℝ) π, cos θ = - (1 / 2) ∧ θ = 2 * π / 3 := 
  sorry

end arccos_neg_half_l137_137474


namespace flat_fee_l137_137780

theorem flat_fee (f n : ℝ) (h1 : f + 4 * n = 320) (h2 : f + 7 * n = 530) : f = 40 := by
  -- Proof goes here
  sorry

end flat_fee_l137_137780


namespace maria_baggies_l137_137759

-- Definitions of the conditions
def total_cookies (chocolate_chip : Nat) (oatmeal : Nat) : Nat :=
  chocolate_chip + oatmeal

def cookies_per_baggie : Nat :=
  3

def number_of_baggies (total_cookies : Nat) (cookies_per_baggie : Nat) : Nat :=
  total_cookies / cookies_per_baggie

-- Proof statement
theorem maria_baggies :
  number_of_baggies (total_cookies 2 16) cookies_per_baggie = 6 := 
sorry

end maria_baggies_l137_137759


namespace factorize_expr1_factorize_expr2_l137_137487

theorem factorize_expr1 (x y : ℝ) : 
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2 * y) :=
by
  sorry

theorem factorize_expr2 (x y : ℝ) : 
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) :=
by
  sorry

end factorize_expr1_factorize_expr2_l137_137487


namespace problem_statement_l137_137812

noncomputable def possible_values_sum (a b c d : ℤ) : ℤ :=
if a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 56 ∧ 
  (a - d = 10) ∧
  ({a - b, b - d} = {4, 6} ∨ {a - b, b - d} = {5, 5}) ∧
  ({a - c, c - d} = {4, 6} ∨ {a - c, c - d} = {5, 5}) ∧
  (b - c = 2 ∨ c - b = 2) then 19 else 0

theorem problem_statement : possible_values_sum a b c d = 19 :=
sorry

end problem_statement_l137_137812


namespace determine_parallelogram_l137_137199

-- Define the conditions
variables {A B C D O : Type} 

-- Condition 1: AB parallel to CD and AD parallel to BC
def condition1 (AB CD AD BC : Type) : Prop :=
  parallel AB CD ∧ parallel AD BC

-- Condition 2: AB equals CD and AD equals BC
def condition2 (AB CD AD BC : Type) : Prop :=
  AB = CD ∧ AD = BC

-- Condition 3: AO equals CO and BO equals DO
def condition3 (AO CO BO DO : Type) : Prop :=
  AO = CO ∧ BO = DO

-- Condition 4: AB parallel to CD, AD equals BC
def condition4 (AB CD AD BC : Type) : Prop :=
  parallel AB CD ∧ AD = BC

-- Define the main theorem to prove: Only condition1, condition2, and condition3 prove that ABCD is a parallelogram
theorem determine_parallelogram (AB CD AD BC AO CO BO DO : Type)
  (h1 : condition1 AB CD AD BC)
  (h2 : condition2 AB CD AD BC)
  (h3 : condition3 AO CO BO DO)
  (h4 : condition4 AB CD AD BC)
  (is_parallelogram : ∀ {P Q R S : Type}, parallel P R ∧ parallel Q S ∨ P = R ∧ Q = S ∨ P = S ∧ Q = R → True) : 
  (is_parallelogram h1) ∧ (is_parallelogram h2) ∧ (is_parallelogram h3) ∧ ¬(is_parallelogram h4) :=
sorry

end determine_parallelogram_l137_137199


namespace no_square_contains_exactly_7_lattice_points_l137_137194

def lattice_point (x y : ℤ) : Prop := true

noncomputable def rotated_square_contains_exactly_7_lattice_points : Prop :=
  ∃ (S : set (ℤ × ℤ)),
  (S ⊆ {p : ℤ × ℤ | ∃ a b, p = (a + b, b - a)}) ∧
  (∃ (square_center : ℤ × ℤ) (side_length : ℤ), side_length > 0 ∧
    ∀ (p : ℤ × ℤ), (p ∈ S ↔ (p.1 ∈ set.Icc (square_center.1 - side_length) (square_center.1 + side_length) ∧
    p.2 ∈ set.Icc (square_center.2 - side_length) (square_center.2 + side_length)))) ∧
  (∃ exact_lattice_points : list (ℤ × ℤ), list.length exact_lattice_points = 7 ∧ 
    ∀ p ∈ exact_lattice_points, lattice_point p.1 p.2 ∧ p ∈ S)

theorem no_square_contains_exactly_7_lattice_points :
  ¬ rotated_square_contains_exactly_7_lattice_points := sorry

end no_square_contains_exactly_7_lattice_points_l137_137194


namespace train_return_time_l137_137031

open Real

theorem train_return_time
  (C_small : Real := 1.5)
  (C_large : Real := 3)
  (speed : Real := 10)
  (initial_connection : String := "A to C")
  (switch_interval : Real := 1) :
  (126 = 2.1 * 60) :=
sorry

end train_return_time_l137_137031


namespace vector_projection_l137_137508

noncomputable def projection_vector := λ (v w : ℝ × ℝ × ℝ), 
  let dot_vw := v.1 * w.1 + v.2 * w.2 + v.3 * w.3 in
  let dot_ww := w.1 * w.1 + w.2 * w.2 + w.3 * w.3 in
  (dot_vw / dot_ww) • w

theorem vector_projection : 
  projection_vector (4, -1, 3) (3, 1, 2) = (51/14, 17/14, 17/7) := 
  sorry

end vector_projection_l137_137508


namespace function_increasing_l137_137148

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) :=
  ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2

theorem function_increasing {f : ℝ → ℝ}
  (H : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2) :
  is_monotonically_increasing f :=
by
  sorry

end function_increasing_l137_137148


namespace smallest_number_of_students_l137_137464

theorem smallest_number_of_students
  (tenth_graders eighth_graders ninth_graders : ℕ)
  (ratio1 : 7 * eighth_graders = 4 * tenth_graders)
  (ratio2 : 9 * ninth_graders = 5 * tenth_graders) :
  (∀ n, (∃ a b c, a = 7 * b ∧ b = 4 * n ∧ a = 9 * c ∧ c = 5 * n) → n = 134) :=
by {
  -- We currently just assume the result for Lean to be syntactically correct
  sorry
}

end smallest_number_of_students_l137_137464


namespace no_perfect_square_less_than_20000_multiple_of_4_as_consecutive_square_difference_l137_137134

theorem no_perfect_square_less_than_20000_multiple_of_4_as_consecutive_square_difference :
  ∀ a ∈ { n | n < 20000 ∧ ∃ k, n = k^2 ∧ n % 4 = 0 }, ¬∃ b, a = (b + 1)^2 - b^2 :=
by
  intros a ha
  obtain ⟨h_lt, ⟨k, hk⟩, h_mod⟩ := ha
  rw [hk, nat.sq_sub_sq] at h_mod
  have : 2 * b + 1 % 4 ≠ 0, sorry
  exact this h_mod

end no_perfect_square_less_than_20000_multiple_of_4_as_consecutive_square_difference_l137_137134


namespace symmetric_conic_transform_l137_137064

open Real

theorem symmetric_conic_transform (x y : ℝ) 
  (h1 : 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0)
  (h2 : x - y + 1 = 0) : 
  5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0 := 
sorry

end symmetric_conic_transform_l137_137064


namespace rectangle_length_reduction_l137_137345

theorem rectangle_length_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_length := L * (1 - 10 / 100)
  let new_width := W * (10 / 9)
  (new_length * new_width = L * W) → 
  x = 10 := by sorry

end rectangle_length_reduction_l137_137345


namespace matrix_linear_combination_l137_137989

-- Definitions of the matrix M and vectors u, v, w and their properties.
variables (M : Matrix (Fin 2) (Fin 1) ℝ)
variables (u v w : Fin 1 → ℝ)

-- Statements expressing the given conditions.
variables (hu : M.mul_vec u = ![3, -1])
variables (hv : M.mul_vec v = ![-2, 4])
variables (hw : M.mul_vec w = ![5, -3])

-- The proof statement
theorem matrix_linear_combination :
  M.mul_vec ((3 : ℝ) • u - v + (2 : ℝ) • w) = ![21, -13] :=
  sorry

end matrix_linear_combination_l137_137989


namespace classrooms_student_hamster_difference_l137_137051

-- Define the problem conditions
def students_per_classroom := 22
def hamsters_per_classroom := 3
def number_of_classrooms := 5

-- Define the problem statement
theorem classrooms_student_hamster_difference :
  (students_per_classroom * number_of_classrooms) - 
  (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end classrooms_student_hamster_difference_l137_137051


namespace tom_annual_car_leasing_cost_l137_137381

theorem tom_annual_car_leasing_cost :
  let miles_mwf := 50 * 3  -- Miles driven on Monday, Wednesday, and Friday
  let miles_other_days := 100 * 4 -- Miles driven on the other days (Sunday, Tuesday, Thursday, Saturday)
  let weekly_miles := miles_mwf + miles_other_days -- Total miles driven per week

  let cost_per_mile := 0.1 -- Cost per mile
  let weekly_fee := 100 -- Weekly fee

  let weekly_cost := weekly_miles * cost_per_mile + weekly_fee -- Total weekly cost

  let weeks_per_year := 52
  let annual_cost := weekly_cost * weeks_per_year -- Annual cost

  annual_cost = 8060 :=
by
  sorry

end tom_annual_car_leasing_cost_l137_137381


namespace simplify_and_evaluate_l137_137308

variable (x y : ℝ)
variable (condition_x : x = 1/3)
variable (condition_y : y = -6)

theorem simplify_and_evaluate :
  3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + (3/2) * x^2 * y)) + 2 * (3 * x * y^2 - x * y) = -4 :=
by
  rw [condition_x, condition_y]
  sorry

end simplify_and_evaluate_l137_137308


namespace find_c_l137_137337

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2
noncomputable def f' (x c : ℝ) := 3 * x ^ 2 - 4 * c * x + c ^ 2
noncomputable def f'' (x c : ℝ) := 6 * x - 4 * c

theorem find_c (c : ℝ) : f' 2 c = 0 ∧ f'' 2 c < 0 → c = 6 :=
by {
  sorry
}

end find_c_l137_137337


namespace tan_theta_l137_137916

theorem tan_theta (θ : ℝ) (h : Real.sin (θ / 2) - 2 * Real.cos (θ / 2) = 0) : Real.tan θ = -4 / 3 :=
sorry

end tan_theta_l137_137916


namespace trapezoid_perimeter_is_correct_l137_137956

def trapezoid_perimeter 
  (PQ RS : Real) (height : Real) (angleQ : Real)
  (PQ_parallel_RS : PQ = 10 ∧ RS = 20 ∧ height = 5 ∧ angleQ = π/6)
  : Real :=
PQ + RS + 2 * (RS - PQ) / 2 * 1 / cos(angleQ)

theorem trapezoid_perimeter_is_correct 
  (PQ RS : Real) (height : Real) (angleQ : Real)
  (h : PQ_parallel_RS : PQ = 10 ∧ RS = 20 ∧ height = 5 ∧ angleQ = π/6) 
  : trapezoid_perimeter PQ RS height angleQ PQ_parallel_RS = 60 := 
by
  sorry

end trapezoid_perimeter_is_correct_l137_137956


namespace percentage_of_games_not_won_is_40_l137_137712

def ratio_games_won_to_lost (games_won games_lost : ℕ) : Prop := 
  games_won / gcd games_won games_lost = 3 ∧ games_lost / gcd games_won games_lost = 2

def total_games (games_won games_lost ties : ℕ) : ℕ :=
  games_won + games_lost + ties

def percentage_games_not_won (games_won games_lost ties : ℕ) : ℕ :=
  ((games_lost + ties) * 100) / (games_won + games_lost + ties)

theorem percentage_of_games_not_won_is_40
  (games_won games_lost ties : ℕ)
  (h_ratio : ratio_games_won_to_lost games_won games_lost)
  (h_ties : ties = 5)
  (h_no_other_games : games_won + games_lost + ties = total_games games_won games_lost ties) :
  percentage_games_not_won games_won games_lost ties = 40 := 
sorry

end percentage_of_games_not_won_is_40_l137_137712


namespace distinct_three_digit_numbers_l137_137131

theorem distinct_three_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in
  ∃ (count : ℕ), count = 60 := by
  sorry

end distinct_three_digit_numbers_l137_137131


namespace students_prefer_mac_l137_137765

-- Define number of students in survey, and let M be the number who prefer Mac to Windows
variables (M E no_pref windows_pref : ℕ)
-- Total number of students surveyed
variable (total_students : ℕ)
-- Define that the total number of students is 210
axiom H_total : total_students = 210
-- Define that one third as many of the students who prefer Mac equally prefer both brands
axiom H_equal_preference : E = M / 3
-- Define that 90 students had no preference
axiom H_no_pref : no_pref = 90
-- Define that 40 students preferred Windows to Mac
axiom H_windows_pref : windows_pref = 40
-- Define that the total number of students is the sum of all groups
axiom H_students_sum : M + E + no_pref + windows_pref = total_students

-- The statement we need to prove
theorem students_prefer_mac :
  M = 60 :=
by sorry

end students_prefer_mac_l137_137765


namespace sequence_sums_l137_137651

noncomputable section

def a_n (n : ℕ) : ℕ := 3 * n
def b_n (n : ℕ) : ℕ := 3 ^ n
def c_n (n : ℕ) : ℕ := if n % 2 = 1 then 1 else b_n (n / 2)

theorem sequence_sums (n : ℕ) : 
  a_1 = 3 ∧ 
  b_1 = 3 ∧ 
  b_n 2 = a_n 3 ∧ 
  b_n 3 = 4 * a_n 2 + 3 →
  ary_sum (a_i, c_i) (2 * n) = (2 * n - 1) * 3 ^ (n + 2) + 6 * n ^ 2 + 9 :=
sorry

end sequence_sums_l137_137651


namespace fifth_term_geometric_sequence_l137_137336

theorem fifth_term_geometric_sequence (x y : ℚ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x + y
    let a2 := x - y
    let a3 := x / y
    let a4 := x * y
    let r := (x - y)/(x + y)
    (a4 * r = (2 / 3)) :=
by
  -- Proof omitted
  sorry

end fifth_term_geometric_sequence_l137_137336


namespace trajectory_of_center_line_passing_through_fixed_point_l137_137538

open Real

theorem trajectory_of_center 
  (F : ℝ × ℝ) (hF : F = (1, 0))
  (tangent_line : ℝ → ℝ → Prop) (htangent : ∀ x y, tangent_line x y ↔ x = -1) :
  ∃ C M, (λ x y, y^2 = 4 * x ∧ (M = C → (F = C ∨ tangent_line C.1 C.2))) := 
sorry

theorem line_passing_through_fixed_point 
  (y1 y2 x1 x2 : ℝ) (L : ℝ × ℝ → ℝ × ℝ → Prop) 
  (hL : L (x1, y1) (x2, y2)) 
  (h_curve : (y1^2 = 4 * x1) ∧ (y2^2 = 4 * x2)) 
  (hy1y2 : y1 * y2 ≠ -16) :
  ∃ (fixed_point : ℝ × ℝ), fixed_point = (4, 0) ∧ 
  ∀ A B, L A B → (A = (x1, y1) ∧ B = (x2, y2) → 
     (∃ k b, ∀ x y, y = k * x + b ∧ (fixed_point.1, fixed_point.2) = (4, 0))) :=
sorry

end trajectory_of_center_line_passing_through_fixed_point_l137_137538


namespace area_of_the_stripe_l137_137021

/-- A white cylindrical silo has a diameter of 30 feet and a height of 80 feet.
    A red stripe with a horizontal width of 4 feet is painted on the silo,
    making three complete revolutions around it. Prove that the
    area of the stripe is 360π square feet. -/
theorem area_of_the_stripe (d h w : ℝ) (revolutions : ℕ)
  (hd : d = 30) (hh : h = 80) (hw : w = 4) (hr : revolutions = 3) :
  let C := π * d in let total_length := (revolutions : ℝ) * C in 
  w * total_length = 360 * π :=
by
  simp [*]
  sorry

end area_of_the_stripe_l137_137021


namespace lambda_three_sufficient_but_not_necessary_l137_137535

noncomputable def vec (a b : ℝ) := (a, b)

def parallel_vectors (a1 a2 b1 b2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1

theorem lambda_three_sufficient_but_not_necessary (λ : ℝ) (h : vector.parallel (vec 3 λ) (vec (λ - 1) 2)) :
  (λ = 3 → vector.parallel (vec 3 λ) (vec (λ - 1) 2)) ∧ (¬ ∀ λ, (vector.parallel (vec 3 λ) (vec (λ - 1) 2)) → (λ = 3)) :=
by sorry

end lambda_three_sufficient_but_not_necessary_l137_137535


namespace find_a_b_minimum_distance_l137_137899

-- Definition for the first part of the problem
def conditions_for_a_b (a b : ℝ) : Prop :=
  4 * a > b ∧ b > 0 ∧ 2 * a + b / 2 = 5 / 2 ∧ a * b = 3 / 2

theorem find_a_b :
  ∃ (a b : ℝ), conditions_for_a_b a b ∧ a = 3 / 4 ∧ b = 2 :=
by
  sorry

-- Definition for the second part of the problem
def distance_sq (x₀ : ℝ) : ℝ :=
  (x₀ - 2)^2 + ((3 / 4 * x₀) + (2 / x₀) - 4)^2

theorem minimum_distance (x₀ : ℝ) :
  x₀ > 0 → ∃ (d : ℝ), d = sqrt (distance_sq x₀) ∧ d = sqrt 2 :=
by
  sorry

end find_a_b_minimum_distance_l137_137899


namespace reflection_point_lies_on_bc_l137_137978

variables (A B C E F M : Type) [Field K] [Geometry K]

-- Definitions according to the conditions
def is_triangle (ABC : Triangle K) : Prop :=
  ∃ (A B C : Point K), ∠ A = 60 ∧ ∠ B + ∠ C = 120

def bisector (ABC : Triangle K) (P : Point K) (a : Angle) (B C E : Point K) : Prop :=
  ∠ BAP = ∠ CAP ∧ E ∈ Line AC

def reflection_point (A M E F : Point K) : Prop :=
  reflection A (Line EF) = M

-- Target theorem to prove
theorem reflection_point_lies_on_bc 
  (ABC : Triangle K) 
  (hABC : is_triangle ABC) 
  (hBE : bisector ABC B ϕ B C E) 
  (hCF : bisector ABC C ψ C B F) 
  (h_reflection : reflection_point A M E F) : 
  M ∈ Line BC :=
by
  -- Proof omitted.
  sorry

end reflection_point_lies_on_bc_l137_137978


namespace ellipse_equation_midpoint_trajectory_max_triangle_area_l137_137205

-- Define conditions
def f1 : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
def f2 : ℝ × ℝ := (2 * Real.sqrt 2, 0)
def min_dist : ℝ := 3 - 2 * Real.sqrt 2

-- Statements to prove
theorem ellipse_equation :
  ∃ a b : ℝ, a = 3 ∧ b^2 = a^2 - (2 * Real.sqrt 2)^2 ∧
  (∀ x y : ℝ, (x, y) ∈ set.of_rel (λ p : ℝ × ℝ, (p.1^2 / a^2) + p.2^2 = 1)) :=
by
  sorry

theorem midpoint_trajectory (x y : ℝ) :
  (∃ t : ℝ, y = -2 * x + t) →
  (∀ N : ℝ × ℝ, (N.1 - 18 * N.2 = 0) ∧ (-18 * Real.sqrt 37 / 37 < N.1) ∧ (N.1 < 18 * Real.sqrt 37 / 37)) :=
by
  sorry

theorem max_triangle_area :
  (∀ m : ℝ, ∃ y1 y2 : ℝ, y1 + y2 = (4 * Real.sqrt 2 * m) / (m^2 + 9) ∧
  y1 * y2 = (-1) / (m^2 + 9)) →
  ∃ t : ℝ, t = 16 ∧ m = Real.sqrt 7 ∧
  (let area := 1.5 in area = 3 / 2) :=
by
  sorry

end ellipse_equation_midpoint_trajectory_max_triangle_area_l137_137205


namespace part1_converse_part1_negation_part1_contrapositive_part2_solution_l137_137760

-- Part (1)
theorem part1_converse (x : ℝ) :
  (x = 1 ∨ x = 2) → (x^2 - 3 * x + 2 = 0) :=
sorry

theorem part1_negation (x : ℝ) :
  (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 1 ∧ x ≠ 2) :=
sorry

theorem part1_contrapositive (x : ℝ) :
  (x ≠ 1 ∧ x ≠ 2) → (x^2 - 3 * x + 2 ≠ 0) :=
sorry

-- Part (2)
theorem part2_solution :
  (∀ x : ℝ, (-1 < x ∧ x < 3) ↔ (x^2 + ((-3)+1)*x + (-3) < 0)) →
   (-3 = -3) :=
by
  intros h1
  refl

end part1_converse_part1_negation_part1_contrapositive_part2_solution_l137_137760


namespace red_triangles_intersect_yellow_cannot_be_2023_l137_137616

theorem red_triangles_intersect_yellow_cannot_be_2023 :
  ∀ (points : Finset (ℝ × ℝ × ℝ)),
    (yellow_points red_points : Finset (ℝ × ℝ × ℝ)),
    yellow_points.card = 3 →
    red_points.card = 40 →
    points = yellow_points ∪ red_points →
    ∀ (plane : set (Finset (ℝ × ℝ × ℝ))),
    (∀ (p1 p2 p3 p4 : (ℝ × ℝ × ℝ)), 
      (p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points) → 
      ¬ collinear {p1, p2, p3, p4}) →
    (∀ (yellow_triangle red_triangle : Finset (ℝ × ℝ × ℝ)),
      yellow_triangle ⊆ yellow_points →
      red_triangle ⊆ red_points →
      yellow_triangle.card = 3 →
      red_triangle.card = 3 →
      (∃! (intersection_point : (ℝ × ℝ × ℝ)),
               intersection_point ∈ yellow_triangle ∩ red_triangle)) →
  false :=
begin
  intros points yellow_points red_points h_yellow h_red h_points plane h_collinear h_intersect,
  sorry -- skip the proof
end

end red_triangles_intersect_yellow_cannot_be_2023_l137_137616


namespace lisa_photos_last_weekend_l137_137263

def photos_of_animals : ℕ := 10
def photos_of_flowers : ℕ := 3 * photos_of_animals
def photos_of_scenery : ℕ := photos_of_flowers - 10
def total_photos_this_week : ℕ := photos_of_animals + photos_of_flowers + photos_of_scenery
def photos_last_weekend : ℕ := total_photos_this_week - 15

theorem lisa_photos_last_weekend : photos_last_weekend = 45 :=
by
  sorry

end lisa_photos_last_weekend_l137_137263


namespace lattice_points_count_l137_137088

-- Definitions of the ellipse and hyperbola
def ellipse (x y m : ℝ) : Prop := 
  (x^2 / (m + 9)) + (y^2 / m) = 1

def hyperbola (x y m : ℝ) : Prop := 
  (x^2 / (9 - m)) - (y^2 / m) = 1

-- Defining the condition that m is between 0 and 9
def valid_m (m : ℝ) : Prop :=
  0 < m ∧ m < 9

-- The number of lattice points inside the region bounded by the ellipse and hyperbola
def numLatticePoints (m : ℝ) : ℕ :=
  25

-- The theorem statement
theorem lattice_points_count (m : ℝ) (hm : valid_m m) :
  ∃ n : ℕ, n = numLatticePoints m ∧ n = 25 := 
by
  use numLatticePoints m
  split
  · rfl
  · sorry

end lattice_points_count_l137_137088


namespace x_squared_plus_y_squared_l137_137591

theorem x_squared_plus_y_squared (x y : ℝ) (h₀ : x + y = 10) (h₁ : x * y = 15) : x^2 + y^2 = 70 :=
by
  sorry

end x_squared_plus_y_squared_l137_137591


namespace required_run_rate_is_25_l137_137750

-- Define initial conditions
def target_runs : ℕ := 282
def initial_run_rate : ℝ := 3.2
def initial_overs : ℕ := 10
def remaining_overs : ℕ := 10

-- Define required run rate
def required_run_rate (scored_runs total_runs : ℕ) (overs : ℕ) : ℝ :=
  (↑total_runs - ↑scored_runs) / overs

-- Initial runs scored in the first 10 overs
def initial_runs_scored : ℝ := initial_run_rate * initial_overs

-- Proof statement
theorem required_run_rate_is_25 :
  required_run_rate initial_runs_scored target_runs remaining_overs = 25 :=
by
  -- Solution steps to be added here later
  sorry

end required_run_rate_is_25_l137_137750


namespace inequality_solution_l137_137312

theorem inequality_solution (x : ℝ) : 
  (0 < (x + 2) / ((x - 3)^3)) ↔ (x < -2 ∨ x > 3)  :=
by
  sorry

end inequality_solution_l137_137312


namespace inscribed_circle_radius_right_triangle_l137_137290

theorem inscribed_circle_radius_right_triangle : 
  ∀ (DE EF DF : ℝ), 
    DE = 6 →
    EF = 8 →
    DF = 10 →
    ∃ (r : ℝ), r = 2 :=
by
  intros DE EF DF hDE hEF hDF
  sorry

end inscribed_circle_radius_right_triangle_l137_137290


namespace selections_with_at_least_one_paperback_l137_137369

-- Define variables for the numbers of paperbacks and hardbacks
def num_paperbacks : ℕ := 2
def num_hardbacks : ℕ := 5

-- Define the total number of books
def total_books : ℕ := num_paperbacks + num_hardbacks

-- Define the total number of possible selections (subsets of the set of books)
def total_selections : ℕ := 2^total_books

-- Define the number of selections that include no paperbacks (only hardbacks)
def selections_without_paperbacks : ℕ := 2^num_hardbacks

-- Prove the number of selections with at least one paperback
theorem selections_with_at_least_one_paperback : total_selections - selections_without_paperbacks = 96 :=
by
  have h1 : total_books = 7 := rfl
  have h2 : total_selections = 128 := by norm_num
  have h3 : selections_without_paperbacks = 32 := by norm_num
  rw [h2, h3]
  exact rfl

end selections_with_at_least_one_paperback_l137_137369


namespace greendale_points_l137_137294

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l137_137294


namespace white_dandelions_on_saturday_l137_137004

-- Define the conditions.
def dandelion_blooming_stage : Nat → String
| 0 => "yellow"
| 1 => "yellow"
| 2 => "yellow"
| 3 => "white"
| 4 => "shed"

variable (monday yellow monday white : Nat) (wednesday yellow wednesday white : Nat)
variable (total_wednesday : Nat := wednesday yellow + wednesday white)

-- Define the initial conditions.
def initial_conditions := monday yellow = 20 ∧ monday white = 14 ∧ 
                          wednesday yellow = 15 ∧ wednesday white = 11

-- Definition of the transition from Monday to Wednesday.
def transition_conditions := total_wednesday = 26 ∧
                             total wednesday - monday yellow = 6

-- Goal: The number of white dandelions on Saturday.
theorem white_dandelions_on_saturday 
  (h1 : initial_conditions)
  (h2 : transition_conditions) :
  ∃ n : Nat, n = 6 :=
by
  sorry

end white_dandelions_on_saturday_l137_137004


namespace photos_last_weekend_45_l137_137265

theorem photos_last_weekend_45 (photos_animals photos_flowers photos_scenery total_photos_this_weekend photos_last_weekend : ℕ)
  (h1 : photos_animals = 10)
  (h2 : photos_flowers = 3 * photos_animals)
  (h3 : photos_scenery = photos_flowers - 10)
  (h4 : total_photos_this_weekend = photos_animals + photos_flowers + photos_scenery)
  (h5 : photos_last_weekend = total_photos_this_weekend - 15) :
  photos_last_weekend = 45 :=
sorry

end photos_last_weekend_45_l137_137265


namespace total_books_count_l137_137813

def Darla_books := 6
def Katie_books := Darla_books / 2
def Darla_Katie_combined_books := Darla_books + Katie_books
def Gary_books := 5 * Darla_Katie_combined_books
def total_books := Darla_books + Katie_books + Gary_books

theorem total_books_count :
  total_books = 54 :=
by
  simp [Darla_books, Katie_books, Darla_Katie_combined_books, Gary_books, total_books]
  sorry

end total_books_count_l137_137813


namespace max_elements_in_T_l137_137645

def T := { x : ℕ | 1 ≤ x ∧ x ≤ 100 }

theorem max_elements_in_T : 
  ∃ (T : set ℕ), T ⊆ {1..100} ∧ 
  (∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a + b) % 5 ≠ 0) ∧ 
  (∀ U, U ⊆ {1..100} ∧ (∀ (a b : ℕ), a ∈ U → b ∈ U → a ≠ b → (a + b) % 5 ≠ 0) → U.card ≤ 41) :=
begin
  sorry
end

end max_elements_in_T_l137_137645


namespace john_total_distance_traveled_l137_137966

theorem john_total_distance_traveled :
  let d1 := 45 * 2.5
  let d2 := 60 * 3.5
  let d3 := 40 * 2
  let d4 := 55 * 3
  d1 + d2 + d3 + d4 = 567.5 := by
  sorry

end john_total_distance_traveled_l137_137966


namespace g_six_composed_l137_137996

def g (x : ℤ) : ℤ := x^2 - 3 * x

theorem g_six_composed (x : ℤ) : g (g(g(g(g(g(x)))))) = 483722099743810 := by
  have h1 : g 2 = -2 := by rw [g, pow_two, mul_comm]; norm_num
  have h2 : g (-2) = 10 := by rw [g, pow_two, mul_comm]; norm_num
  have h3 : g 10 = 70 := by rw [g, pow_two, mul_comm]; norm_num
  have h4 : g 70 = 4690 := by rw [g, pow_two, mul_comm]; norm_num
  have h5 : g 4690 = 21994030 := by rw [g, pow_two, mul_comm]; norm_num
  have h6 : g 21994030 = 483722099743810 := by rw [g, pow_two, mul_comm]; norm_num
  sorry

end g_six_composed_l137_137996


namespace cannot_construct_infinite_brick_wall_with_wires_l137_137040

theorem cannot_construct_infinite_brick_wall_with_wires :
  ¬ (∃ (wire_sequence : ℕ → ℕ), (∀ n, wire_sequence n = n) ∧ 
    (∃ (brick_wall : ℕ → ℕ × ℕ → bool), (∀ ⟨x, y⟩, brick_wall x y = true ∨ brick_wall x y = false) ∧ 
    (infinite_brick_wall brick_wall) ∧ 
    (no_overlap wire_sequence brick_wall))) :=
sorry

def infinite_brick_wall (brick_wall : ℕ → ℕ × ℕ → bool) : Prop :=
  ∀ x y, ∃ z w, brick_wall z w = true

def no_overlap (wire_sequence : ℕ → ℕ) (brick_wall : ℕ → ℕ × ℕ → bool) : Prop :=
  ∀ n, ∀ x y, ∀ m, (m ≠ n → 
  (brick_wall (x + wire_sequence n) y = false ∨ brick_wall (x + wire_sequence m) y = false ∨ 
   brick_wall x (y + wire_sequence n) = false ∨ brick_wall x (y + wire_sequence m) = false))

end cannot_construct_infinite_brick_wall_with_wires_l137_137040


namespace no_three_distinct_positive_perfect_squares_sum_to_100_l137_137947

theorem no_three_distinct_positive_perfect_squares_sum_to_100 :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ (m n p : ℕ), a = m^2 ∧ b = n^2 ∧ c = p^2) ∧ a + b + c = 100 :=
by
  sorry

end no_three_distinct_positive_perfect_squares_sum_to_100_l137_137947


namespace average_age_of_women_l137_137753

variables (A : ℝ) (W : ℝ)
variables (H1 : ∀ (A : ℝ), 8 * A - 48 + W = 8 * (A + 2))

theorem average_age_of_women : (W / 2) = 32 :=
by
  have h1 : 8 * A - 48 + W = 8 * A + 16 := H1 A
  have h2 : W = 64 := by linarith [h1]
  show (W / 2) = 32 from by rw [h2]; norm_num

end average_age_of_women_l137_137753


namespace sequence_equation_l137_137860

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n:ℕ, n > 0 → a (n + 1) = a n + 2 * n)

theorem sequence_equation :
  ∀ (a : ℕ → ℕ), sequence a → ∀ n: ℕ, n > 0 → a n = n^2 - n + 1 :=
  by 
  intros a h n hn
  sorry

end sequence_equation_l137_137860


namespace distinct_four_digit_integers_with_one_repeating_digit_l137_137123

-- Define the set of odd digits
def odd_digits : Finset ℕ := {1, 3, 5, 7, 9}

-- Define the problem statement
theorem distinct_four_digit_integers_with_one_repeating_digit : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ (∀ d ∈ n.digits 10, d ∈ odd_digits) ∧ (∃ d ∈ odd_digits, n.digits 10.count d = 2)}.card = 360 := 
by
  sorry

end distinct_four_digit_integers_with_one_repeating_digit_l137_137123


namespace frustum_surface_areas_l137_137807

variable (r1 r2 h s lateral_area total_area : ℝ)

-- Conditions
def frustum_conditions :=
  r1 = 10 ∧ r2 = 4 ∧ h = 9 ∧ s = Real.sqrt (h ^ 2 + (r1 - r2) ^ 2)

-- Proof of lateral surface area
def lateral_surface_area_proof :=
  frustum_conditions r1 r2 h s lateral_area total_area →
  lateral_area = π * (r1 + r2) * s

-- Proof of total surface area including bases
def total_surface_area_proof :=
  frustum_conditions r1 r2 h s lateral_area total_area →
  total_area = lateral_area + π * r1 ^ 2 + π * r2 ^ 2

-- Theorem stating the final results
theorem frustum_surface_areas :
  frustum_conditions r1 r2 h s lateral_area total_area →
    lateral_area = 42 * π * Real.sqrt 13 ∧
    total_area = 116 * π + 42 * π * Real.sqrt 13 :=
by 
  intros, 
  sorry

end frustum_surface_areas_l137_137807


namespace reflection_proof_l137_137326

def original_center : (ℝ × ℝ) := (8, -3)
def reflection_line (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)
def reflected_center : (ℝ × ℝ) := reflection_line original_center

theorem reflection_proof : reflected_center = (-3, -8) := by
  sorry

end reflection_proof_l137_137326


namespace lillian_total_mushrooms_l137_137262

theorem lillian_total_mushrooms :
  let safe := 9
  let poisonous := 2 * safe
  let uncertain := 5
  safe + poisonous + uncertain = 32 :=
by
  let safe := 9
  let poisonous := 2 * safe
  let uncertain := 5
  show safe + poisonous + uncertain = 32 from sorry

end lillian_total_mushrooms_l137_137262


namespace fractional_eq_range_m_l137_137884

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l137_137884


namespace distance_traveled_in_20_seconds_l137_137013

-- Define the initial distance, common difference, and total time
def initial_distance : ℕ := 8
def common_difference : ℕ := 9
def total_time : ℕ := 20

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := initial_distance + (n - 1) * common_difference

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_terms (n : ℕ) : ℕ := n * (initial_distance + nth_term n) / 2

-- The main theorem to be proven
theorem distance_traveled_in_20_seconds : sum_of_terms 20 = 1870 := 
by sorry

end distance_traveled_in_20_seconds_l137_137013


namespace smallest_solution_x4_minus_50x2_plus_576_eq_0_l137_137821

theorem smallest_solution_x4_minus_50x2_plus_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 576 = 0) ∧ ∀ y : ℝ, (y^4 - 50 * y^2 + 576 = 0) → x ≤ y :=
begin
  use - real.sqrt 26,
  split,
  { -- Prove that -sqrt(26) is indeed a solution
    sorry
  },
  { -- Prove that -sqrt(26) is the smallest solution
    sorry
  }
end

end smallest_solution_x4_minus_50x2_plus_576_eq_0_l137_137821


namespace prob_I_prob_II_prob_III_l137_137565

noncomputable theory
open Classical

-- Problem (I)
def f (x a : ℝ) : ℝ := sin x - a * x
theorem prob_I (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → 0 ≤ a) → (∀ x : ℝ, 0 < x ∧ x < 1 → f x a > 0) :=
sorry

-- Problem (II)
def g (x : ℝ) : ℝ := ln x - x + 1
theorem prob_II : (∃ x : ℝ, x > 0 ∧ g x ≤ g 1) :=
sorry

-- Problem (III)
theorem prob_III (n : ℕ) (hn : 0 < n) : log (n + 1) < (finset.range n).sum (λ k, (1 / (k + 1 : ℝ))) :=
sorry

end prob_I_prob_II_prob_III_l137_137565


namespace factorial_quotient_l137_137044

-- Define the new factorial-like function
def nf (n a : ℕ) : ℕ :=
  (finset.range (n / (2 * a))).prod (λ k, n - 2 * k * a)

/-- Calculate the quotient of the new factorial-like functions 40_4! divided by 10_2! equates to 9072. -/
theorem factorial_quotient : (nf 40 4) / (nf 10 2) = 9072 :=
begin
  sorry -- Proof is omitted as per instructions
end

end factorial_quotient_l137_137044


namespace sin_alpha_value_trigonometric_expression_value_l137_137096

namespace TrigonometryProof

-- Given conditions
variables {α : ℝ}
axiom cos_alpha : cos α = -sqrt 5 / 5
axiom alpha_range : π < α ∧ α < 3 * π / 2

-- (I) Prove the value of sin α
theorem sin_alpha_value : sin α = -2 * sqrt 5 / 5 := sorry

-- (II) Prove the value of the given trigonometric expression
theorem trigonometric_expression_value :
  (sin (π + α) + 2 * sin (3 * π / 2 + α)) / (cos (3 * π - α) + 1) = sqrt 5 - 1 := sorry

end TrigonometryProof

end sin_alpha_value_trigonometric_expression_value_l137_137096


namespace set_of_points_forms_line_l137_137207

theorem set_of_points_forms_line (z : ℂ) (h : |z + 1| = |z - complex.I|) :
    ∃ a b c : ℝ, a * z.re + b * z.im + c = 0 :=
sorry

end set_of_points_forms_line_l137_137207


namespace largest_integer_product_l137_137720

theorem largest_integer_product :
  ∃ (digits : List ℕ), 
      (∀ (d : ℕ), d ∈ digits → d > 0) ∧            -- each digit is positive
      List.chain (<) 0 digits ∧                   -- digits are strictly increasing
      (List.sum (List.map (λ d, d * d) digits) = 62) ∧  -- sum of squares of digits is 62
      (digits = [2, 3, 4, 7] ∧                    -- the largest integer with these properties
      digits.prod = 168) :=                       -- the product of its digits is 168
by
  sorry

end largest_integer_product_l137_137720


namespace daily_increase_cans_l137_137399

theorem daily_increase_cans (x : ℕ) : 
  let first_day_cans := 20 in
  let days := 5 in
  let total_goal := 150 in
  (first_day_cans + (first_day_cans + x) + (first_day_cans + 2 * x) + (first_day_cans + 3 * x) + (first_day_cans + 4 * x) = total_goal) → x = 5 :=
by
  intros
  sorry

end daily_increase_cans_l137_137399


namespace grading_ways_l137_137776

theorem grading_ways (students : ℕ) (grades : ℕ) (h_students : students = 12) (h_grades : grades = 4) :
  (grades ^ students) = 16777216 :=
by
  -- Using the given conditions, we need to show that 4^12 = 16777216
  rw [h_students, h_grades]
  -- Then, compute 4^12
  norm_num
  -- This confirms that 4^12 = 16777216
  sorry

end grading_ways_l137_137776


namespace commute_distance_l137_137445

theorem commute_distance (D : ℝ)
  (h1 : ∀ t : ℝ, t > 0 → t = D / 45)
  (h2 : ∀ t : ℝ, t > 0 → t = D / 30)
  (h3 : D / 45 + D / 30 = 1) :
  D = 18 :=
by
  sorry

end commute_distance_l137_137445


namespace cos_theta_unit_circle_l137_137106

theorem cos_theta_unit_circle :
  ∀ (θ : ℝ) (x y : ℝ),
  (x = -3/5) →
  (y = 4/5) →
  (x^2 + y^2 = 1) →
  (x = -3/5) →
  (cos θ = x) →
  cos θ = -3/5 :=
by
  intros θ x y hx hy hunit hcos
  rw [hx, hy] at *
  exact hcos

end cos_theta_unit_circle_l137_137106


namespace range_of_x_squared_f_x_lt_x_squared_minus_f_1_l137_137818

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def satisfies_inequality (f f' : ℝ → ℝ) : Prop :=
∀ x : ℝ, 2 * f x + x * f' x < 2

theorem range_of_x_squared_f_x_lt_x_squared_minus_f_1 (f f' : ℝ → ℝ)
  (h_even : even_function f)
  (h_ineq : satisfies_inequality f f')
  : {x : ℝ | x^2 * f x - f 1 < x^2 - 1} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
sorry

end range_of_x_squared_f_x_lt_x_squared_minus_f_1_l137_137818


namespace simon_age_is_10_l137_137451

-- Define the conditions
def alvin_age := 30
def half_alvin_age := alvin_age / 2
def simon_age := half_alvin_age - 5

-- State the theorem
theorem simon_age_is_10 : simon_age = 10 :=
by
  sorry

end simon_age_is_10_l137_137451


namespace binom_bound_l137_137982

def binomial_mod (n k p : ℕ) := nat.choose n k % p

theorem binom_bound (k x p : ℕ) (h1 : p = 4 * k + 1)
  (h2 : |x| ≤ (p - 1) / 2) (h3 : binomial_mod (2 * k) k p = x % p) :
  |x| ≤ 2 * real.sqrt p :=
sorry

end binom_bound_l137_137982


namespace midpoint_trajectory_equation_l137_137985

noncomputable def ellipse_equation_and_foci
    (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b)
    (A : ℝ × ℝ) (hA : A = (1, 3/2))
    (sum_dist_foci : ℝ) (h_sum_dist : sum_dist_foci = 4) :
    Prop :=
    let C := ellipse_eq a b in
    let F1 := (-sqrt (a^2 - b^2), 0) in
    let F2 := (sqrt (a^2 - b^2), 0) in
    (C = (fun x y => x^2 / a^2 + y^2 / b^2 = 1)) ∧ (F1 = (-1, 0)) ∧ (F2 = (1, 0))

theorem midpoint_trajectory_equation
    (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b)
    (K : ℝ × ℝ) (Q : ℝ × ℝ) (hK_ellipse : ∀ (x y : ℝ), (K = (x, y)) → (x^2 / a^2 + y^2 / b^2 = 1))
    (hQ_midpoint : Q = ((-1 + (K.1)) / 2, (K.2) / 2)) :
    (Q.1 + 1/2)^2 + 4 * (Q.2)^2 / 3 = 1 :=
sorry

end midpoint_trajectory_equation_l137_137985


namespace remaining_seat_number_l137_137604

/-
In a class of 60 students, a systematic sampling method is used to draw a sample of size 5.
It is known that the students with seat numbers 3, 15, 39, and 51 are all in the sample.
Prove the seat number of the remaining student in the sample is 27.
-/

theorem remaining_seat_number (n : ℕ) : 
  n = 60 ∧ 
  ∃ (s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 t: ℕ → ℕ) (t(0) = 3 ∧ t(1) = 15 ∧ t(2) = 27 ∧ t(3) = 39 ∧ t(4) = 51) ∧ t(5) ∈ s: 
  t : ℕ → ℕ → ℕ :=
by
  sorry

end remaining_seat_number_l137_137604


namespace projection_correct_l137_137510

noncomputable def projection (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_vw := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let dot_ww := w.1 * w.1 + w.2 * w.2 + w.3 * w.3
  let scalar := dot_vw / dot_ww
  (scalar * w.1, scalar * w.2, scalar * w.3)

theorem projection_correct :
  projection (4, -1, 3) (3, 1, 2) = (51 / 14, 17 / 14, 17 / 7) := 
by
  -- The proof would go here.
  sorry

end projection_correct_l137_137510


namespace pages_left_proof_l137_137280

-- Definitions based on the given conditions
def initial_books : ℕ := 10
def pages_per_book : ℕ := 100
def lost_books : ℕ := 2

-- Calculate the number of books left
def books_left : ℕ := initial_books - lost_books

-- Calculate the total number of pages left
def total_pages_left : ℕ := books_left * pages_per_book

-- Theorem statement representing the proof problem
theorem pages_left_proof : total_pages_left = 800 := by
  -- Call the necessary built-ins and perform the calculation steps within the proof.
  simp [total_pages_left, books_left, initial_books, lost_books, pages_per_book]
  sorry -- Proof steps will go here, or use a by computation or decidability tactics


end pages_left_proof_l137_137280


namespace pies_differ_in_both_l137_137172

-- Defining types of pies
inductive Filling where
  | apple : Filling
  | cherry : Filling

inductive Preparation where
  | fried : Preparation
  | baked : Preparation

structure Pie where
  filling : Filling
  preparation : Preparation

-- The set of all possible pies
def allPies : Set Pie :=
  { ⟨Filling.apple, Preparation.fried⟩,
    ⟨Filling.apple, Preparation.baked⟩,
    ⟨Filling.cherry, Preparation.fried⟩,
    ⟨Filling.cherry, Preparation.baked⟩ }

-- Theorem stating that we can buy two pies that differ in both filling and preparation
theorem pies_differ_in_both (pies : Set Pie) (h : 3 ≤ pies.card) :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
  sorry

end pies_differ_in_both_l137_137172


namespace probability_in_dark_l137_137781

theorem probability_in_dark (rev_per_min : ℕ) (given_prob : ℝ) (h1 : rev_per_min = 3) (h2 : given_prob = 0.25) :
  given_prob = 0.25 :=
by
  sorry

end probability_in_dark_l137_137781


namespace find_N_numbers_l137_137370

theorem find_N_numbers (N : ℕ) (hN : N ≠ 20) 
  (S : finset ℕ)
  (h : ∀ S₁ S₂ : finset ℕ, S₁.card = 10 → S₁ ∪ S₂ = S → 
           S₁.sum id > S.sum id - S₁.sum id) : 
  N = 19 := 
sorry

end find_N_numbers_l137_137370


namespace num_squares_8x8_num_squares_nxn_l137_137403

-- Part (a) for an 8x8 chessboard
theorem num_squares_8x8 : 
  let board_size := 8 in
  (∑ k in finset.range (board_size + 1), (board_size + 1 - k) * (board_size + 1 - k)) = 204 :=
by
  sorry

-- Part (b) for a general n x n chessboard
theorem num_squares_nxn (n : ℕ) : 
  (∑ k in finset.range (n + 1), (n + 1 - k) * (n + 1 - k)) = n * (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

end num_squares_8x8_num_squares_nxn_l137_137403


namespace four_points_concyclic_l137_137795

variable {α : Type*}
variables (A B C E F H M N P Q : α)
variables [metric_space α] [normed_group α] [normed_space ℝ α] [inner_product_space ℝ α]

-- Definitions for the problem
def is_altitude (p q r t : α) : Prop :=
  orthogonal (p -ᵥ q) (r -ᵥ q) ∧ collinear {q, r, t}

def circle_with_diameter (p q : α) : set α :=
  { x | ∥x -ᵥ midpoint ℝ p q∥ = ∥p -ᵥ midpoint ℝ p q∥ }

def is_on_line (x l1 l2 : α) : Prop :=
  x ∈ line_through ℝ l1 l2

-- Problem statement
theorem four_points_concyclic
  (h_be : is_altitude B E A C)
  (h_cf : is_altitude C F A B)
  (h_circle_ab : M ∈ circle_with_diameter A B ∧ N ∈ circle_with_diameter A B)
  (h_on_cf : is_on_line M C F ∧ is_on_line N C F)
  (h_circle_ac : P ∈ circle_with_diameter A C ∧ Q ∈ circle_with_diameter A C)
  (h_on_be : is_on_line P B E ∧ is_on_line Q B E) :
  cyclic [M, P, N, Q] :=
sorry

end four_points_concyclic_l137_137795


namespace log_mul_eq_log_add_l137_137287

variable {a x1 x2 : ℝ}

theorem log_mul_eq_log_add (y1 y2 : ℝ) (h1 : y1 = Real.log a x1) (h2 : y2 = Real.log a x2) :
  Real.log a (x1 * x2) = Real.log a x1 + Real.log a x2 := sorry

end log_mul_eq_log_add_l137_137287


namespace problem_statement_l137_137998

variable {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0)
variable {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)

theorem problem_statement (m M : ℝ) 
  (hm : m = Real.Inf {z | ∃ x y, x ≠ 0 ∧ y ≠ 0 ∧ z = |a*x + b*y| / (|x| + |y|)}) 
  (hM : M = Real.Sup {z | ∃ x y, x ≠ 0 ∧ y ≠ 0 ∧ z = |a*x + b*y| / (|x| + |y|)}) : 
  M - m = (|a + b|) / 2 := 
sorry

end problem_statement_l137_137998


namespace translation_of_function_l137_137382

theorem translation_of_function (x : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = -3*x) ∧ (∃ (g : ℝ → ℝ), ∀ x, g x = f x + 2)) →
  (∀ x, (λ x, -3*x + 2) x = ((λ x, (λ x, -3*x) x + 2)) x) :=
by
  sorry

end translation_of_function_l137_137382


namespace prove_x_range_l137_137286

theorem prove_x_range {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (h : 9 / a + 1 / b = 2) : 
  (∀ a b : ℝ, 0 < a → 0 < b → 9 / a + 1 / b = 2 → a + b ≥ x^2 + 2 * x) ↔ (-4 ≤ x ∧ x ≤ 2) :=
begin
  sorry,
end

end prove_x_range_l137_137286


namespace range_of_m_l137_137147

open Real

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (h : ∃ x, f x = 0) :
  (∀ x, f x = 2^(-(abs (x - 1))) - m) → m ∈ Ioc 0 1 := sorry

end range_of_m_l137_137147


namespace exist_two_pies_differing_in_both_l137_137158

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l137_137158


namespace inequality_solution_set_min_value_of_x_plus_y_l137_137110

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem inequality_solution_set (a : ℝ) :
  (if a < 0 then (∀ x : ℝ, f a x > 0 ↔ (1/a < x ∧ x < 2))
   else if a = 0 then (∀ x : ℝ, f a x > 0 ↔ x < 2)
   else if 0 < a ∧ a < 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 2 ∨ 1/a < x))
   else if a = 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x ≠ 2))
   else if a > 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 1/a ∨ x > 2))
   else false) := 
sorry

theorem min_value_of_x_plus_y (a : ℝ) (h : 0 < a) (x y : ℝ) (hx : y ≥ f a (|x|)) :
  x + y ≥ -a - (1/a) := 
sorry

end inequality_solution_set_min_value_of_x_plus_y_l137_137110


namespace unique_zero_point_in_interval_l137_137566

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 - 2 * x + b

theorem unique_zero_point_in_interval (b : ℝ) :
  (∃! x ∈ Ioo 2 4, f x b = 0) → (-8 < b ∧ b < 0) := by
    sorry

end unique_zero_point_in_interval_l137_137566


namespace sum_of_permutations_l137_137874

theorem sum_of_permutations :
  ∀ (a : Fin 5 → Fin 5),
    (∃ j : Fin 5, (a j) = j) ∧
    (∀ i : Fin 5, ∃ p : Fin 5 → Fin 5, Perm p ∧ p i = a i) →
    (∑ i, a i) = 675 :=
by sorry

end sum_of_permutations_l137_137874


namespace cost_calculation_l137_137458

-- Define the cost of two dozen apples
def cost_two_dozen : ℝ := 15.60

-- Define the factor to calculate the cost of three dozen apples
def factor : ℝ := 3 / 2

-- Define the expected cost of three dozen apples
def cost_three_dozen : ℝ := 23.40

-- Proof that the cost of three dozen apples is $23.40, given that the cost of two dozen apples is $15.60 and the factor is 1.5
theorem cost_calculation : cost_two_dozen * factor = cost_three_dozen :=
by
  sorry

end cost_calculation_l137_137458


namespace problem_AC_length_l137_137620

noncomputable def length_AC_approx : ℝ := ∑ \sqrt873

theorem problem_AC_length :
  ∀ (AB DC AD AC : ℝ),
    AB = 15 → DC = 26 → AD = 9 →
    AC = length_AC_approx 873 ≈ 29.5 :=
by
  intros AB DC AD AC hAB hDC hAD
  sorry

end problem_AC_length_l137_137620


namespace solve_equation_l137_137689

theorem solve_equation (x : ℝ) (h : (4 * x ^ 2 + 6 * x + 2) / (x + 2) = 4 * x + 7) : x = -4 / 3 :=
by
  sorry

end solve_equation_l137_137689


namespace sequence_form_l137_137492

-- Defining the sequence a_n as a function f
def seq (f : ℕ → ℕ) : Prop :=
  ∃ c : ℝ, (0 < c) ∧ ∀ m n : ℕ, Nat.gcd (f m + n) (f n + m) > (c * (m + n))

-- Proving that if there exists such a sequence, then it is of the form n + c
theorem sequence_form (f : ℕ → ℕ) (h : seq f) :
  ∃ c : ℤ, ∀ n : ℕ, f n = n + c :=
sorry

end sequence_form_l137_137492


namespace minimum_value_of_f_l137_137099

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
def ab_ne_zero : Prop := a * b ≠ 0

def f (x : ℝ) : ℝ := a * x^3 + b * Real.arcsin x + 3

def f_has_max_value_10 : Prop := ∃ x : ℝ, f x = 10

theorem minimum_value_of_f :
  ab_ne_zero → f_has_max_value_10 → ∃ x : ℝ, f x = -4 := 
by
  intro h1 h2
  sorry

end minimum_value_of_f_l137_137099


namespace a0_eq_zero_l137_137656

open Matrix

variables {m n k : ℕ}
variables {A : Matrix (Fin m) (Fin n) ℂ}
variables {B : Matrix (Fin n) (Fin m) ℂ}
variables {a : ℕ → ℂ}

theorem a0_eq_zero (h1 : m ≥ n) (h2 : n ≥ 2)
  (h3 : a 0 • (1 : Matrix (Fin m) (Fin m) ℂ) + 
      a 1 • (A ⬝ B) + 
      a 2 • ((A ⬝ B) ⬝ (A ⬝ B)) + 
      ∑ i in range k, a (i + 3) • (A ⬝ B) ^ (i + 3) = 0) 
  (h4 : a 0 • (1 : Matrix (Fin n) (Fin n) ℂ) + 
      a 1 • (B ⬝ A) + 
      a 2 • ((B ⬝ A) ⬝ (B ⬝ A)) + 
      ∑ i in range k, a (i + 3) • (B ⬝ A) ^ (i + 3) ≠ 0) : 
  a 0 = 0 :=
sorry

end a0_eq_zero_l137_137656


namespace sum_of_sequence_l137_137113

theorem sum_of_sequence (a : ℕ → ℚ) (h: ∀ n : ℕ, a n = (2 / (n^2 + n))) :
    (∑ i in Finset.range 99, a (i + 1)) = 99 / 50 :=
by
  sorry

end sum_of_sequence_l137_137113


namespace probability_five_cards_one_from_each_suit_and_extra_l137_137921

/--
Given five cards chosen with replacement from a standard 52-card deck, 
the probability of having exactly one card from each suit, plus one 
additional card from any suit, is 3/32.
-/
theorem probability_five_cards_one_from_each_suit_and_extra 
  (cards : ℕ) (total_suits : ℕ)
  (prob_first_diff_suit : ℚ) 
  (prob_second_diff_suit : ℚ) 
  (prob_third_diff_suit : ℚ) 
  (prob_fourth_diff_suit : ℚ) 
  (prob_any_suit : ℚ) 
  (total_prob : ℚ) :
  cards = 5 ∧ total_suits = 4 ∧ 
  prob_first_diff_suit = 3 / 4 ∧ 
  prob_second_diff_suit = 1 / 2 ∧ 
  prob_third_diff_suit = 1 / 4 ∧ 
  prob_fourth_diff_suit = 1 ∧ 
  prob_any_suit = 1 →
  total_prob = 3 / 32 :=
by {
  sorry
}

end probability_five_cards_one_from_each_suit_and_extra_l137_137921


namespace max_independent_set_of_30_l137_137938

variables (S : Finset ℕ) (f : ℕ → Finset ℕ)

-- Each participant knows at most 5 others.
def knows_at_most_5 (x : ℕ) : Prop := (f x).card ≤ 5

-- In any group of 5 people, there are always two who do not know each other.
def group5_no_mutual_knowledge (A : Finset ℕ) (hA : A.card = 5) : Prop :=
∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ ¬(f a).contains b

-- There exists a subset of size k such that no two people in this subset know each other.
noncomputable def max_independent_subset (S : Finset ℕ) (f : ℕ → Finset ℕ) : ℕ :=
  if h : ∃ A : Finset ℕ, A ⊆ S ∧ A.card = 6 ∧ ∀ a b ∈ A, a ≠ b → ¬(f a).contains b then 6 else 0

-- Main proof statement
theorem max_independent_set_of_30 (S : Finset ℕ) (hS : S.card = 30)
  (f : ℕ → Finset ℕ) (h_knows : ∀ x ∈ S, knows_at_most_5 S f x)
  (h_group5 : ∀ A : Finset ℕ, A ⊆ S → A.card = 5 → group5_no_mutual_knowledge S f A (by simp)) :
  max_independent_subset S f = 6 :=
sorry

end max_independent_set_of_30_l137_137938


namespace print_height_is_25_l137_137431

-- Conditions as definitions
def painting_height : ℝ := 10
def painting_width : ℝ := 15
def print_width : ℝ := 37.5

-- Aspect ratio as a definition
def aspect_ratio := painting_width / painting_height

-- Conclusion we want to prove
theorem print_height_is_25 : ∃ (print_height : ℝ), (print_width / print_height) = aspect_ratio ∧ print_height = 25 :=
by
  sorry

end print_height_is_25_l137_137431


namespace trig_identity_l137_137307

theorem trig_identity :
  (sin (15 * Real.pi / 180) + sin (45 * Real.pi / 180)) / 
  (cos (15 * Real.pi / 180) + cos (45 * Real.pi / 180)) = 
  tan (30 * Real.pi / 180) :=
by
  sorry

end trig_identity_l137_137307


namespace seedlings_to_achieve_profit_l137_137425

def profit_per_pot (s : ℕ) (seedlings : ℕ) (profit_per_seedling : ℕ) : ℕ := seedlings * profit_per_seedling

theorem seedlings_to_achieve_profit : 
  ∀ (x : ℕ), (x + 3) * (3 - 0.5 * x) = 10 → (x + 3 = 4 ∨ x + 3 = 5) :=
begin
  sorry
end

end seedlings_to_achieve_profit_l137_137425


namespace product_prices_correct_l137_137770

noncomputable def product_A_cost : ℝ := 17
noncomputable def product_B_cost : ℝ := 28
noncomputable def product_C_cost : ℝ := 38

noncomputable def product_A_profit_margin : ℝ := 0.20
noncomputable def product_B_profit_margin : ℝ := 0.25
noncomputable def product_C_profit_margin : ℝ := 0.15

noncomputable def commission_rate : ℝ := 0.20

def final_price (cost : ℝ) (profit_margin : ℝ) (commission_rate : ℝ) : ℝ :=
  let selling_price_before_commission := cost * (1 + profit_margin)
  in selling_price_before_commission / (1 - commission_rate)

theorem product_prices_correct :
  final_price product_A_cost product_A_profit_margin commission_rate = 25.50 ∧
  final_price product_B_cost product_B_profit_margin commission_rate = 43.75 ∧
  final_price product_C_cost product_C_profit_margin commission_rate = 54.63 :=
by sorry

end product_prices_correct_l137_137770


namespace find_theta_l137_137548

open Real

theorem find_theta (x y : ℝ) (hx : x = (sqrt 3) / 2) (hy : y = -1 / 2) (hxy : y < 0 ∧ x > 0) (hθ : θ = Real.arctan (y / x) ∨ θ = Real.arctan (y / x) + π ∨ θ = Real.arctan (y / x) + 2*π) : θ = 11*π / 6 :=
by
  sorry

end find_theta_l137_137548


namespace min_value_of_expr_l137_137594

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  (b / (c + d)) + (c / (a + b))

theorem min_value_of_expr
  (a d : ℝ) (b c : ℝ)
  (h1: a ≥ 0) (h2: d ≥ 0) (h3: b > 0) (h4: c > 0) (h5: b + c ≥ a + d) :
  ∃ min_val : ℝ, min_val = sqrt 2 - 1 / 2 ∧ 
    ∀ x y z w : ℝ, h1 → h2 → h3 → h4 → h5 → min_value_expr x y z w ≥ min_val :=
sorry

end min_value_of_expr_l137_137594


namespace can_buy_two_pies_different_in_both_l137_137176

structure Pie :=
  (filling : Type)
  (preparation : Type)

def apple : Type := unit
def cherry : Type := unit
def fried : Type := unit
def baked : Type := unit

def apple_fried : Pie := { filling := apple, preparation := fried }
def apple_baked : Pie := { filling := apple, preparation := baked }
def cherry_fried : Pie := { filling := cherry, preparation := fried }
def cherry_baked : Pie := { filling := cherry, preparation := baked }

def possible_pies : List Pie := [apple_fried, apple_baked, cherry_fried, cherry_baked]

theorem can_buy_two_pies_different_in_both 
  (available_pies : List Pie) 
  (h : available_pies.length ≥ 3) : 
  ∃ (p1 p2 : Pie), p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation := 
  sorry

end can_buy_two_pies_different_in_both_l137_137176


namespace arithmetic_sequence_15th_term_l137_137741

theorem arithmetic_sequence_15th_term : 
  let a₁ := 3
  let d := 4
  let n := 15
  a₁ + (n - 1) * d = 59 :=
by
  let a₁ := 3
  let d := 4
  let n := 15
  sorry

end arithmetic_sequence_15th_term_l137_137741


namespace minimum_turns_for_closed_route_l137_137184

theorem minimum_turns_for_closed_route 
  (m n : ℕ) 
  (h_m : m = 10) 
  (h_n : n = 10) : 
  let intersections := m * n in
  let minimum_turns := 20 in
  True := sorry

end minimum_turns_for_closed_route_l137_137184


namespace centroid_distance_sum_l137_137288

variables {α : Type*} [LinearOrder α] [AddGroup α] [Module α ℝ]

def is_centroid (G : α) (A B C : α) : Prop := 
  ∃ (λ₁ λ₂ λ₃ : ℝ), λ₁ + λ₂ + λ₃ = 1 ∧ λ₁ > 0 ∧ λ₂ > 0 ∧ λ₃ > 0 ∧ G = λ₁ • A + λ₂ • B + λ₃ • C

def is_projection (P X : α) (r : α → ℝ) : Prop :=
  ∀ (y : α), r P = r X + (P - X) • y
  
theorem centroid_distance_sum {A B C G : α} {r : α → ℝ}
  (hG : is_centroid G A B C)
  (hr : ∀ X ∈ {A, B, C}, r X ≠ r G) :
  r A = r B + r C := 
sorry

end centroid_distance_sum_l137_137288


namespace distance_Albany_Syracuse_l137_137786

-- Conditions
variable (D : ℝ) -- Distance from Albany to Syracuse
variable (speed_Al_Sy : ℝ := 50) -- Speed from Albany to Syracuse (miles/hour)
variable (speed_Sy_Al : ℝ := 38.71) -- Speed from Syracuse back to Albany (miles/hour)
variable (total_time : ℝ := 5.5) -- Total travel time (hours)

-- Theorem to be proved
theorem distance_Albany_Syracuse :
  D / speed_Al_Sy + D / speed_Sy_Al = total_time → D ≈ 121.1 :=
by
  sorry

end distance_Albany_Syracuse_l137_137786


namespace hyperbola_eccentricity_l137_137104

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h : a > 0 ∧ b > 0) (h_perpendicular: (b / a) * (-b / a) = -1) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  in c / a

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) :
  eccentricity_of_hyperbola a a ⟨h, h⟩ (by simp [mul_neg_eq_neg_mul_symm, Real.div_mul_div, mul_inv_cancel, ne_of_gt h]) = Real.sqrt 2 :=
  sorry

end hyperbola_eccentricity_l137_137104


namespace proof_problem_l137_137896

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l137_137896


namespace janet_total_earnings_l137_137224

-- Definitions based on conditions from step a)
def hourly_wage := 70
def hours_worked := 20
def rate_per_pound := 20
def weight_sculpture1 := 5
def weight_sculpture2 := 7

-- Statement for the proof problem
theorem janet_total_earnings : 
  let earnings_from_extermination := hourly_wage * hours_worked
  let earnings_from_sculpture1 := rate_per_pound * weight_sculpture1
  let earnings_from_sculpture2 := rate_per_pound * weight_sculpture2
  earnings_from_extermination + earnings_from_sculpture1 + earnings_from_sculpture2 = 1640 := 
by
  sorry

end janet_total_earnings_l137_137224


namespace remainder_of_x_l137_137691

theorem remainder_of_x (x : ℕ) 
(H1 : 4 + x ≡ 81 [MOD 16])
(H2 : 6 + x ≡ 16 [MOD 36])
(H3 : 8 + x ≡ 36 [MOD 64]) :
  x ≡ 37 [MOD 48] :=
sorry

end remainder_of_x_l137_137691


namespace valid_n_l137_137831

theorem valid_n (n : ℕ) (x : Fin n → ℤ) (y : ℤ)
  (h_nonzero : (∀ i, x i ≠ 0) ∧ y ≠ 0)
  (h_sum_zero : ∑ i, x i = 0)
  (h_sum_squares : ∑ i, x i ^ 2 = n * y ^ 2) :
  n > 0 → ¬ (n = 1 ∨ n = 3) → 
  {k : ℕ // n = 2 * k ∨ n = 3 + 2 * k} :=
by
  sorry

end valid_n_l137_137831


namespace pd_squared_expression_l137_137283

theorem pd_squared_expression
  (a b c : ℝ)
  (P_interior_of_ABC : ∃ (P : ℝ × ℝ × ℝ), P ∈ interior (triangle (A B C (regular_tetrahedron ABCD)))
  (PA PB PC : ℝ) 
  (h_PA : PA = a) 
  (h_PB : PB = b) 
  (h_PC : PC = c) 
  : 
  PD^2 = 1/2 * (a^2 + b^2 + c^2) + (1/(2 * sqrt 3)) * sqrt (2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) - (a^4 + b^4 + c^4)))
  :=
begin
  sorry
end

end pd_squared_expression_l137_137283


namespace positive_difference_between_two_numbers_l137_137716

theorem positive_difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : y^2 - 4 * x^2 = 80) : 
  |y - x| = 179.33 := 
by sorry

end positive_difference_between_two_numbers_l137_137716


namespace find_portion_equally_divided_profit_l137_137268

noncomputable def portion_equally_divided_profit (Mary_invest : ℕ) (Mike_invest : ℕ) 
(Total_profit : ℝ) (extra_Mary_received : ℝ) : ℝ :=
(E : ℝ) :=
  (E / 2 + 11 / 20 * (Total_profit - E)) = (E / 2 + 9 / 20 * (Total_profit - E) + extra_Mary_received)

theorem find_portion_equally_divided_profit :
  portion_equally_divided_profit 550 450 14999.999999999995 1000 = 5000 := by
  sorry

end find_portion_equally_divided_profit_l137_137268


namespace max_playground_area_l137_137662

/-- Mara is setting up a fence around a rectangular playground with given constraints.
    We aim to prove that the maximum area the fence can enclose is 10000 square feet. --/
theorem max_playground_area (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 400) 
  (h2 : l ≥ 100) 
  (h3 : w ≥ 50) : 
  l * w ≤ 10000 :=
sorry

end max_playground_area_l137_137662


namespace problem_statement_l137_137219

noncomputable def angle_B (b c : ℝ) (A : ℝ) (h : b * tan A + b * tan (π / 3) = (sqrt 3 * c) / cos A) : ℝ :=
  π / 3

noncomputable def side_c (a b : ℝ) (h1 : a = 8) (h2 : b = 4 * sqrt 7) : ℝ :=
  12

noncomputable def sin_2A_plus_pi_over_4 (a b c A : ℝ) (h1 : a = 8) (h2 : b = 4 * sqrt 7) (h3 : c = 12) : ℝ :=
  (sqrt 2 + 4 * sqrt 6) / 14

theorem problem_statement (a b c A : ℝ) :
  b * tan A + b * tan (π / 3) = (sqrt 3 * c) / cos A →
  a = 8 →
  b = 4 * sqrt 7 →
  c = 12 →
  angle_B b c A (by sorry) = π / 3 ∧
  side_c a b (by rfl) (by rfl) = 12 ∧
  sin_2A_plus_pi_over_4 a b c A (by rfl) (by rfl) (by rfl) = (sqrt 2 + 4 * sqrt 6) / 14 :=
by sorry

end problem_statement_l137_137219


namespace smallest_total_books_l137_137719

-- Definitions based on conditions
def physics_books (x : ℕ) := 3 * x
def chemistry_books (x : ℕ) := 2 * x
def biology_books (x : ℕ) := (3 / 2 : ℚ) * x

-- Total number of books
def total_books (x : ℕ) := physics_books x + chemistry_books x + biology_books x

-- Statement of the theorem
theorem smallest_total_books :
  ∃ x : ℕ, total_books x = 15 ∧ 
           (∀ y : ℕ, y < x → total_books y % 1 ≠ 0) :=
sorry

end smallest_total_books_l137_137719


namespace pies_differ_in_both_l137_137167

-- Definitions of pie types
inductive Filling
| apple
| cherry

inductive PreparationMethod
| fried
| baked

structure Pie where
  filling : Filling
  method : PreparationMethod

-- The set of all possible pie types
def pies : Set Pie := {
  {filling := Filling.apple, method := PreparationMethod.fried},
  {filling := Filling.cherry, method := PreparationMethod.fried},
  {filling := Filling.apple, method := PreparationMethod.baked},
  {filling := Filling.cherry, method := PreparationMethod.baked}
}

-- The statement to prove: If there are at least three types of pies available, then there exist two pies that differ in both filling and preparation method.
theorem pies_differ_in_both :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.method ≠ p2.method :=
begin
  sorry
end

end pies_differ_in_both_l137_137167


namespace area_trapezoid_120_angle_BQP_45_AP_x_minus_7_PQ_13_when_areas_equal_l137_137951

-- Definitions of the given conditions.
variables (AB DC AD PQ x : ℝ)
variables (angle_DAB angle_ADC angle_BQP : ℝ)
variables (P Q T : Type*)
variables [metric_space P] [metric_space Q] [metric_space T]

-- Given conditions
def is_trapezoid (ABCD : Type*) [metric_space ABCD] (AB DC AD PQ angle_DAB angle_ADC : ℝ) : Prop :=
  angle_DAB = 90 ∧ angle_ADC = 90 ∧ AB = 7 ∧ DC = 17 ∧ AD = 10 ∧ PQ = x

-- Question (a): Prove the area of trapezoid ABCD is 120.
def area_trapezoid (AB DC AD angle_DAB angle_ADC : ℝ) (area : ℝ) : Prop :=
  ∃ (h : angle_DAB = 90 ∧ angle_ADC = 90), 
    area = 0.5 * AD * (AB + DC) ∧ area = 120

-- Question (b): Prove the measure of ∠ BQP is 45°.
def angle_BQP_proof (angle_BQP : ℝ) : Prop :=
  angle_BQP = 45

-- Question (c): Given PQ = x, prove that AP is x - 7.
def length_AP (PQ x AP : ℝ) : Prop :=
  PQ = x → AP = x - 7

-- Question (d): Prove that PQ = 13 for the areas of ABQP and PQCD to be equal.
def equal_areas (PQ x : ℝ) (area_ABCD area_ABQP area_PQCD : ℝ) : Prop :=
  area_ABCD = 0.5 * 10 * (7 + 17) ∧
  area_ABCD = 120 ∧
  area_ABQP = 60 ∧
  area_PQCD = 60 ∧
  area_ABQP = (PQ + 7) * (PQ - 7) / 2 ∧
  area_PQCD = (PQ + 17) * (17 - PQ) / 2 ∧
  PQ = x ∧
  x = 13

-- Lean theorems to be proven according to the problem
theorem area_trapezoid_120 (h : is_trapezoid ABCD AB DC AD PQ angle_DAB angle_ADC) : 
  area_trapezoid AB DC AD angle_DAB angle_ADC 120 :=
sorry

theorem angle_BQP_45 (h : is_trapezoid ABCD AB DC AD PQ angle_DAB angle_ADC) : 
  angle_BQP_proof angle_BQP :=
sorry

theorem AP_x_minus_7 (h : is_trapezoid ABCD AB DC AD PQ angle_DAB angle_ADC) : 
  length_AP PQ x (x - 7) :=
sorry

theorem PQ_13_when_areas_equal (h : is_trapezoid ABCD AB DC AD PQ angle_DAB angle_ADC) : 
  equal_areas PQ x 120 60 60 :=
sorry

end area_trapezoid_120_angle_BQP_45_AP_x_minus_7_PQ_13_when_areas_equal_l137_137951


namespace number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l137_137797

theorem number_of_sixth_graders_who_bought_more_pens_than_seventh_graders 
  (p : ℕ) (h1 : 178 % p = 0) (h2 : 252 % p = 0) :
  (252 / p) - (178 / p) = 5 :=
sorry

end number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l137_137797


namespace derek_age_l137_137471

theorem derek_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end derek_age_l137_137471


namespace max_value_PA_PB_l137_137854

noncomputable def m : ℝ := sorry
def A : (ℝ × ℝ) := (1, 0)
def B : (ℝ × ℝ) := (2, 3)

def l1 (m : ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, x + m * y - 1 = 0

def l2 (m : ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, m * x - y - 2 * m + 3 = 0

def perpendicular (m : ℝ) : Prop :=
  m * (-1/m) = -1

def P_intersects (P : ℝ × ℝ) (l1 l2 : ℝ × ℝ → Prop) : Prop :=
  l1 P ∧ l2 P ∧ P ≠ A ∧ P ≠ B

def distance (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_value_PA_PB 
  (h_m_real : m ∈ ℝ)
  (h_A_on_l1 : l1 m A)
  (h_B_on_l2 : l2 m B)
  (h_perpendicular : perpendicular m)
  (h_P_intersects : ∃ P, P_intersects P (l1 m) (l2 m)) : 
  (∀ P, l1 m P ∧ l2 m P ∧ P ≠ A ∧ P ≠ B → (distance P A + distance P B) ≤ 2 * real.sqrt 5) :=
by
  intros P h
  sorry

end max_value_PA_PB_l137_137854


namespace time_to_sweep_one_room_l137_137470

theorem time_to_sweep_one_room (x : ℕ) :
  (10 * x) = (2 * 9 + 6 * 2) → x = 3 := by
  sorry

end time_to_sweep_one_room_l137_137470


namespace angle_complement_supplement_l137_137453

theorem angle_complement_supplement :
  ∃ A : ℝ, (90 - A) = (1 / 3) * (180 - A) - 10 ∧ A = 60 :=
by {
  -- We state the existence of an angle A satisfying the given condition and the expected result.
  use 60, -- Correct angle measure according to the final solution.
  split,
  {
    -- Proving the condition given in the problem for A = 60.
    -- Here we directly check the derived condition.
    calc
    90 - 60 = 30  : by norm_num
    ...         = (1 / 3) * (180 - 60) - 10 : by norm_num
  },
  {
    -- A = 60 is trivially true in this construction, hence the second part of split is satisfied.
    exact rfl,
  }
}

end angle_complement_supplement_l137_137453


namespace value_of_b_l137_137364

-- Define the variables and conditions
variables (a b c : ℚ)
axiom h1 : a + b + c = 150
axiom h2 : a + 10 = b - 3
axiom h3 : b - 3 = 4 * c 

-- The statement we want to prove
theorem value_of_b : b = 655 / 9 := 
by 
  -- We start with assumptions h1, h2, and h3
  sorry

end value_of_b_l137_137364


namespace coefficient_of_x6_in_expansion_proof_l137_137734

noncomputable def coefficient_of_x6_in_expansion : Nat :=
  90720

theorem coefficient_of_x6_in_expansion_proof :
  let p := 3
  let q := 2
  let n := 8
  (∑ k in Finset.range (n+1), Nat.choose n k * (p*x)^(n-k) * q^k) =
  coefficient_of_x6_in_expansion :=
sorry

end coefficient_of_x6_in_expansion_proof_l137_137734


namespace jason_cutting_hours_l137_137629

-- Definitions derived from conditions
def time_to_cut_one_lawn : ℕ := 30  -- minutes
def lawns_per_day := 8 -- number of lawns Jason cuts each day
def days := 2 -- number of days (Saturday and Sunday)
def minutes_in_an_hour := 60 -- conversion factor from minutes to hours

-- The proof problem
theorem jason_cutting_hours : 
  (time_to_cut_one_lawn * lawns_per_day * days) / minutes_in_an_hour = 8 := sorry

end jason_cutting_hours_l137_137629


namespace more_pens_than_pencils_l137_137015

-- Define the number of pencils (P) and pens (Pe)
def num_pencils : ℕ := 15 * 80

-- Define the number of pens (Pe) is more than twice the number of pencils (P)
def num_pens (Pe : ℕ) : Prop := Pe > 2 * num_pencils

-- State the total cost equation in terms of pens and pencils
def total_cost_eq (Pe : ℕ) : Prop := (5 * Pe + 4 * num_pencils = 18300)

-- Prove that the number of more pens than pencils is 1500
theorem more_pens_than_pencils (Pe : ℕ) (h1 : num_pens Pe) (h2 : total_cost_eq Pe) : (Pe - num_pencils = 1500) :=
by
  sorry

end more_pens_than_pencils_l137_137015


namespace outcome_transactions_l137_137269

-- Definition of initial property value and profit/loss percentages.
def property_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

-- Calculate selling price after 15% profit.
def selling_price : ℝ := property_value * (1 + profit_percentage)

-- Calculate buying price after 5% loss based on the above selling price.
def buying_price : ℝ := selling_price * (1 - loss_percentage)

-- Calculate the net gain/loss.
def net_gain_or_loss : ℝ := selling_price - buying_price

-- Statement to be proved.
theorem outcome_transactions : net_gain_or_loss = 862.5 := by
  sorry

end outcome_transactions_l137_137269


namespace minimum_languages_required_l137_137028

theorem minimum_languages_required (n : ℕ) : (∃ n, 10 ≤ n ∧ binomial n (nat.floor (n / 2)) ≥ 250) :=
by {
  use 10,
  split,
  exact nat.le_refl 10,
  simp,
  have h : binomial 10 5 = 252 := by norm_num,
  rw h,
  norm_num,
}

end minimum_languages_required_l137_137028


namespace triangle_BXN_equilateral_l137_137983

theorem triangle_BXN_equilateral (A B C M N X : Type) [IsTriangle A B C] 
  (hM : IsMidpoint M A C) 
  (hCN : IsAngleBisector C N A B) 
  (hX : IsIntersection X (Median B M) (Bisector C N))
  (hEquilateral : IsEquilateralTriangle B X N) 
  (hAC : distance A C = 2) :
  distance_squared B X = (10 - 6 * Real.sqrt 2) / 7 := 
sorry

end triangle_BXN_equilateral_l137_137983


namespace color_dot_figure_l137_137480

-- Definitions reflecting the problem conditions
def num_colors : ℕ := 3
def first_triangle_coloring_ways : ℕ := 6
def subsequent_triangle_coloring_ways : ℕ := 3
def additional_dot_coloring_ways : ℕ := 2

-- The theorem stating the required proof
theorem color_dot_figure : first_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           additional_dot_coloring_ways = 108 := by
sorry

end color_dot_figure_l137_137480


namespace positive_difference_16_l137_137696

def avg_is_37 (y : ℤ) : Prop := (45 + y) / 2 = 37

def positive_difference (a b : ℤ) : ℤ := if a > b then a - b else b - a

theorem positive_difference_16 (y : ℤ) (h : avg_is_37 y) : positive_difference 45 y = 16 :=
by
  sorry

end positive_difference_16_l137_137696


namespace greendale_high_school_points_l137_137292

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l137_137292


namespace find_evening_tickets_l137_137351

noncomputable def matinee_price : ℕ := 5
noncomputable def evening_price : ℕ := 12
noncomputable def threeD_price : ℕ := 20
noncomputable def matinee_tickets : ℕ := 200
noncomputable def threeD_tickets : ℕ := 100
noncomputable def total_revenue : ℕ := 6600

theorem find_evening_tickets (E : ℕ) (hE : total_revenue = matinee_tickets * matinee_price + E * evening_price + threeD_tickets * threeD_price) :
  E = 300 :=
by
  sorry

end find_evening_tickets_l137_137351


namespace area_of_30_60_90_triangle_l137_137704

theorem area_of_30_60_90_triangle (h : ℝ) (theta : ℝ) (hypotenuse : h = 12) (angle : theta = 30) :
  let a := 6 in
  let b := 6 * (Real.sqrt 3) in
  (1 / 2) * a * b = 18 * Real.sqrt 3 :=
by
  sorry

end area_of_30_60_90_triangle_l137_137704


namespace age_sum_l137_137634

def leonard_age : ℕ := 6
def nina_age (leonard_age : ℕ) : ℕ := leonard_age + 4
def jerome_age (nina_age : ℕ) : ℕ := 2 * nina_age

theorem age_sum :
  let leonard := leonard_age;
      nina := nina_age leonard;
      jerome := jerome_age nina
  in leonard + nina + jerome = 36 :=
by
  sorry

end age_sum_l137_137634


namespace projection_correct_l137_137514

noncomputable def projection_of_vector_on_line
  (v : ℝ × ℝ × ℝ)
  (d : ℝ × ℝ × ℝ)
: ℝ × ℝ × ℝ :=
  let dot_prod := v.1 * d.1 + v.2 * d.2 + v.3 * d.3,
      norm_sq := d.1 ^ 2 + d.2 ^ 2 + d.3 ^ 2 in
  (dot_prod / norm_sq) • d

theorem projection_correct :
  projection_of_vector_on_line (4, -1, 3) (3, 1, 2) = (51/14, 17/14, 17/7) :=
by
  sorry

end projection_correct_l137_137514


namespace jenny_used_two_twenty_dollar_bills_l137_137961

-- Definitions based on conditions
def cost_per_page := 0.10
def num_copies := 7
def pages_per_essay := 25
def cost_per_pen := 1.50
def num_pens := 7
def change_received := 12.0
def bill_value := 20.0

-- The total cost calculation
def total_printing_cost := num_copies * pages_per_essay * cost_per_page
def total_pen_cost := num_pens * cost_per_pen
def total_cost := total_printing_cost + total_pen_cost

-- The given money calculation
def money_given := total_cost + change_received

-- The required number of bills calculation
def num_twenty_dollar_bills := money_given / bill_value

theorem jenny_used_two_twenty_dollar_bills : num_twenty_dollar_bills = 2 := by
  sorry

end jenny_used_two_twenty_dollar_bills_l137_137961


namespace find_k_l137_137082

-- Define the arithmetic sequence and sum of first n terms
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

theorem find_k 
  (a d : ℤ) 
  (S_n : ℕ → ℤ := sum_first_n_terms a d) 
  (h1 : S_n 10 > 0) 
  (h2 : S_n 11 = 0) 
  (h3 : ∀ n ∈ (set.univ : set ℕ), S_n n ≤ S_n (k : ℕ)) :
  k ∈ {5, 6} :=
sorry

end find_k_l137_137082


namespace distance_from_center_to_triangle_incenter_l137_137063

-- Define the triangle and its side lengths
def triangle := { a : ℝ, b : ℝ, c : ℝ }

-- Define the conditions
def a : ℝ := 13
def b : ℝ := 14
def c : ℝ := 15
def R : ℝ := 10

-- Define function to calculate semiperimeter
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define function to calculate the area using Heron's formula
def area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c in
  (1 / 4) * real.sqrt ((s * (s - a) * (s - b) * (s - c)))

-- Define function to calculate the inradius
def inradius (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c in
  (2 * area a b c) / (a + b + c)

-- Define the Lean statement to prove
theorem distance_from_center_to_triangle_incenter :
  let r := inradius a b c
  OI = real.sqrt (R * R - r * r) = 2 * real.sqrt 21 :=
sorry

end distance_from_center_to_triangle_incenter_l137_137063


namespace jill_hourly_wage_l137_137963

theorem jill_hourly_wage {total_earnings tip_rate avg_orders_per_hour hours_per_shift shifts_worked hourly_wage : ℝ}
    (h1 : tip_rate = 0.15)
    (h2 : avg_orders_per_hour = 40)
    (h3 : shifts_worked = 3)
    (h4 : hours_per_shift = 8)
    (h5 : total_earnings = 240) :
    let total_orders := avg_orders_per_hour * shifts_worked * hours_per_shift in
    let total_tips := tip_rate * total_orders in
    let earnings_from_hourly_wage := total_earnings - total_tips in
    let total_hours_worked := shifts_worked * hours_per_shift in
    hourly_wage = earnings_from_hourly_wage / total_hours_worked :=
by {
    -- Proof would go here
    sorry
}

end jill_hourly_wage_l137_137963


namespace exist_two_pies_differing_in_both_l137_137160

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l137_137160


namespace dot_product_a_b_l137_137578

noncomputable def vector_dot_product : ℝ :=
  sorry

variables (a b : E) [inner_product_space ℝ E]

axiom norm_a : ‖a‖ = 1
axiom norm_b : ‖b‖ = 2
axiom norm_a_minus_2b : ‖a - 2 • b‖ = real.sqrt 10

theorem dot_product_a_b : inner a b = 7 / 4 :=
by
  sorry

end dot_product_a_b_l137_137578


namespace exists_subseq_sum_div_by_100_l137_137675

theorem exists_subseq_sum_div_by_100 (a : Fin 100 → ℤ) :
  ∃ (s : Finset (Fin 100)), (s.sum (λ i, a i)) % 100 = 0 := 
by 
  sorry

end exists_subseq_sum_div_by_100_l137_137675


namespace central_angle_of_regular_hexagon_l137_137328

theorem central_angle_of_regular_hexagon :
  ∀ (total_angle : ℝ) (sides : ℝ), total_angle = 360 → sides = 6 → total_angle / sides = 60 :=
by
  intros total_angle sides h_total_angle h_sides
  rw [h_total_angle, h_sides]
  norm_num

end central_angle_of_regular_hexagon_l137_137328


namespace segment_PM_length_l137_137032

theorem segment_PM_length (O P M A B : Point) (AB_len : Real) :
  AO = (4 / 5) * AB_len ∧
  BP = (2 / 3) * AB_len ∧
  M = (A + B) / 2 ∧
  OM = 2 →
  PM = (10 / 9) :=
by
  sorry

end segment_PM_length_l137_137032


namespace max_value_abs_expr_l137_137922

theorem max_value_abs_expr (x y : ℝ) (h : x^2 + y^2 ≤ 1) : 
  ∃ (M : ℝ), M = sqrt 2 ∧ ∀ (x y : ℝ), x^2 + y^2 ≤ 1 → abs (x^2 + 2 * x * y - y^2) ≤ M :=
sorry

end max_value_abs_expr_l137_137922


namespace tenured_professors_percentage_l137_137798

noncomputable def percentage_tenured (W M T TM : ℝ) := W = 0.69 ∧ (1 - W) = M ∧ (M * 0.52) = TM ∧ (W + T - TM) = 0.90 → T = 0.7512

-- Define the mathematical entities
variables (W M T TM : ℝ)

-- The main statement
theorem tenured_professors_percentage : percentage_tenured W M T TM := by
  sorry

end tenured_professors_percentage_l137_137798


namespace num_sol_x_squared_eq_x_divides_num_invertible_elems_l137_137974

variable (A : Type) [CommRing A] [Fintype A] [DecidableEq A]

theorem num_sol_x_squared_eq_x_divides_num_invertible_elems 
  [OddCard : Fact ((Fintype.card A) % 2 = 1)] :
  ∃ k : ℕ, (Fintype.card {x : A // x * x = x}) * k = Fintype.card {x : A // isUnit x} :=
sorry

end num_sol_x_squared_eq_x_divides_num_invertible_elems_l137_137974


namespace repeating_decimal_sum_l137_137398

theorem repeating_decimal_sum : 
  let frac := (47 : ℚ) / 99 
  in frac.num + frac.den = 146 :=
by 
  let frac := (47 : ℚ) / 99 
  have h_frac : frac = 47 / 99 := rfl
  sorry

end repeating_decimal_sum_l137_137398


namespace composite_19_8n_plus_17_l137_137673

theorem composite_19_8n_plus_17 (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
by 
  sorry

end composite_19_8n_plus_17_l137_137673


namespace max_perimeter_of_triangle_l137_137937

theorem max_perimeter_of_triangle (A B C a b c p : ℝ) 
  (h_angle_A : A = 2 * Real.pi / 3)
  (h_a : a = 3)
  (h_perimeter : p = a + b + c) 
  (h_sine_law : b = 2 * Real.sqrt 3 * Real.sin B ∧ c = 2 * Real.sqrt 3 * Real.sin C) :
  p ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end max_perimeter_of_triangle_l137_137937


namespace exist_two_pies_differing_in_both_l137_137162

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l137_137162


namespace projection_correct_l137_137513

noncomputable def projection_of_vector_on_line
  (v : ℝ × ℝ × ℝ)
  (d : ℝ × ℝ × ℝ)
: ℝ × ℝ × ℝ :=
  let dot_prod := v.1 * d.1 + v.2 * d.2 + v.3 * d.3,
      norm_sq := d.1 ^ 2 + d.2 ^ 2 + d.3 ^ 2 in
  (dot_prod / norm_sq) • d

theorem projection_correct :
  projection_of_vector_on_line (4, -1, 3) (3, 1, 2) = (51/14, 17/14, 17/7) :=
by
  sorry

end projection_correct_l137_137513


namespace triangle_ABC_proof_l137_137602

theorem triangle_ABC_proof (A B C : ℝ) (a b c : ℝ) (CD AD : ℝ):
  a < b -> b < c ->
  b * sin C = sqrt 3 * a ->
  sin B * sin C = cos (A - C) + cos B ->
  A + B + C = real.pi ->
  CD = 3 ->
  AD = sqrt 7 ->
  cos C = -1 / 2 ∧ (1 / 2 * a * b * sin C = sqrt 3 / 2 ∨ 1 / 2 * a * b * sin C = sqrt 3 / 8) :=
begin
  sorry
end

end triangle_ABC_proof_l137_137602


namespace ptolemys_theorem_radius_proof_l137_137041

noncomputable def cyclic_quadrilateral_radius (AB CD BC AD BD : ℝ)
    (sqrt3 : ℝ)
    (ptolemy_theorem : AB * CD + BC * AD = 4 * sqrt3)
    (equal_relation : AC = sqrt3 * BD)
    (angle_relation : ADC = 2 * BAD) : ℝ :=
  ∃ (R : ℝ), R = 2  -- Considering the correct answer provided

theorem ptolemys_theorem_radius_proof
    {AB CD BC AD AC BD: ℝ}
    (h_equal_rel: AC = Real.sqrt 3 * BD)
    (h_angle_rel: ∠ADC = 2 * ∠BAD)
    (h_ptolemy: AB * CD + BC * AD = 4 * Real.sqrt 3) :
    (cyclic_quadrilateral_radius AB CD BC AD BD (Real.sqrt 3) h_ptolemy h_equal_rel h_angle_rel) = 2 := 
  sorry 

end ptolemys_theorem_radius_proof_l137_137041


namespace Evelyn_bottle_caps_l137_137485

theorem Evelyn_bottle_caps (initial_caps found_caps total_caps : ℕ)
  (h1 : initial_caps = 18)
  (h2 : found_caps = 63) :
  total_caps = 81 :=
by
  sorry

end Evelyn_bottle_caps_l137_137485


namespace ellipse_focus_sum_constant_l137_137248

theorem ellipse_focus_sum_constant (a b : ℝ) (h : a > b > 0) 
  (P F1 F2 M N : EuclideanSpace ℝ 2) 
  (h_ellipse : P ∈ { p : EuclideanSpace ℝ 2 | (p.1)^2/a^2 + (p.2)^2/b^2 = 1 })
  (h_foci : (F1,F2) = ((-sqrt (a^2 - b^2), 0), (sqrt (a^2 - b^2), 0)))
  (h_intersection1 : ∃ M ∈ { p : EuclideanSpace ℝ 2 | (p.1)^2/a^2 + (p.2)^2/b^2 = 1 }, is_point_on_line P F1 M)
  (h_intersection2 : ∃ N ∈ { p : EuclideanSpace ℝ 2 | (p.1)^2/a^2 + (p.2)^2/b^2 = 1 }, is_point_on_line P F2 N) :
  |P.dist F1| / |F1.dist M| + |P.dist F2| / |F2.dist N| = 2 * (a^2 + (a^2 - b^2)) / b^2 :=
sorry

/-- Helper definition to define a point is on a line --/
def is_point_on_line {P1 P2 P : EuclideanSpace ℝ 2} : Prop := 
  ∃ k : ℝ, P = (P1 + k • (P2 - P1))

/-- Distance between two points in the Euclidean space of 2 dimensions --/
def dist (P₁ P₂ : EuclideanSpace ℝ 2) : ℝ :=
  (P₂ - P₁).norm

end ellipse_focus_sum_constant_l137_137248


namespace sequence_50th_term_l137_137954

def term_sequence (n m : ℕ) : ℚ :=
  if n + m - 1 ≤ 0 then 0 else (n : ℚ) / m

def find_group (k : ℕ) : ℕ :=
  Nat.find (λ n, (nat_mul_sum (n)) ≥ k)

def term_in_group (k group : ℕ) : ℕ :=
  k - nat_mul_sum (group - 1)

theorem sequence_50th_term : 
  term_sequence (term_in_group 50 10) (11 - term_in_group 50 10) = 5/6 :=
  sorry

end sequence_50th_term_l137_137954


namespace train_speed_l137_137016

theorem train_speed (distance time : ℕ) (h1 : distance = 500) (h2 : time = 20) : distance / time = 25 := 
by
  rw [h1, h2]
  exact (500 : ℕ) / 20 = 25

end train_speed_l137_137016


namespace solution_set_inequality_l137_137559

theorem solution_set_inequality
  (a b : ℝ)
  (h_sol : ∀ x : ℝ, 2 < x ∧ x < 3 → ax^2 + 5x + b > 0) :
  ∀ x : ℝ, (x < -1/2 ∨ x > -1/3) → bx^2 - 5x + a < 0 :=
by sorry

end solution_set_inequality_l137_137559


namespace sqrt_infinite_nest_eq_two_l137_137140

theorem sqrt_infinite_nest_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := 
sorry

end sqrt_infinite_nest_eq_two_l137_137140


namespace expression_divisible_512_l137_137686

theorem expression_divisible_512 (n : ℤ) (h : n % 2 ≠ 0) : (n^12 - n^8 - n^4 + 1) % 512 = 0 := 
by 
  sorry

end expression_divisible_512_l137_137686


namespace solution_to_quadratic_l137_137777

theorem solution_to_quadratic :
  ∀ (p q : ℝ), (p * p - 6 * p - 36 = 0) ∧ (q * q - 6 * q - 36 = 0) ∧ (p ≥ q) →
  3 * p + 2 * q = 15 + 3 * Real.sqrt 5 :=
by
  intros p q hpq conditions
  sorry

end solution_to_quadratic_l137_137777


namespace total_money_l137_137022

theorem total_money (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 330) (h3 : C = 30) : 
  A + B + C = 500 :=
by
  sorry

end total_money_l137_137022


namespace find_sequence_l137_137571

noncomputable def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ∈ ℕ+, a (n + 2) = 3 * a (n + 1) + 18 * a n + 2^n

noncomputable def explicit_formula (n : ℕ) : ℚ :=
  (6^n) / 32 + (7 * (-3)^n) / 16 - (2^n) / 16

theorem find_sequence (a : ℕ → ℚ) (n : ℕ) (h : sequence a) : a n = explicit_formula n :=
sorry

end find_sequence_l137_137571


namespace inv_undefined_l137_137367

noncomputable def g : ℕ → ℕ
| 2 := 4
| 4 := 16
| 6 := 11
| 9 := 2
| 11 := 6
| 14 := 9
| 18 := 14
| _ := 0  -- g is 0 for inputs not in the table

theorem inv_undefined :
  ¬ ∃ x : ℕ, g x = (10 / 9 : ℚ) :=
begin
  intro h,
  cases h with x hx,
  have hx_nat : (10 / 9 : ℚ) ∉ g '' {2, 4, 6, 9, 11, 14, 18},
  { simp [g],
    dec_trivial },
  exact hx_nat hx,
end

end inv_undefined_l137_137367


namespace octagon_area_coprime_l137_137621

-- Define basic geometric constructs and given conditions for the problem
variables (O : Point) (s1 s2 : Square) (oct : Octagon)
assume (concentric : concentric_squares s1 s2 O) (side_length : s1.side = 1)
assume (AB_length : length (s1.side_segment AB) = 37/81)

-- The corresponding proof statement
theorem octagon_area_coprime (M N : ℕ) (H : is_coprime M N) : 
  M + N = 155 :=
by sorry

end octagon_area_coprime_l137_137621


namespace max_elements_subset_no_sum_mod_5_divisible_l137_137640

theorem max_elements_subset_no_sum_mod_5_divisible (T : Set ℕ)
  (hT₁ : ∀ x ∈ T, x ∈ Finset.range 101)
  (hT₂ : ∀ x y ∈ T, x ≠ y → ¬ (x + y) % 5 = 0) : T.card ≤ 40 := sorry

end max_elements_subset_no_sum_mod_5_divisible_l137_137640


namespace increasing_f_on_interval_sequence_strictly_increasing_l137_137658

-- 1. Prove that f(x) is increasing on (0, 1)
theorem increasing_f_on_interval : ∀ x y : ℝ, 0 < x → x < 1 → 0 < y → y < 1 → x < y → (x - x * real.log x) < (y - y * real.log y) :=
  sorry

-- 2. Prove that a_n < a_{n+1} < 1 for the sequence {a_n}
theorem sequence_strictly_increasing (a : ℕ → ℝ) (h0 : ∀ n, 0 < a n) (h1 : ∀ n, a n < 1) (h_rec : ∀ n, a (n + 1) = a n - a n * real.log (a n)) :
  ∀ n, a n < a (n + 1) ∧ a (n + 1) < 1 :=
  sorry

end increasing_f_on_interval_sequence_strictly_increasing_l137_137658


namespace total_arrangement_methods_l137_137822

variable (M F: Type) [Fintype M] [Fintype F]

def num_doctors := 4
def num_male := 2
def num_female := 2
def num_hospitals := 3
def at_least_one_doctor (h : Finset (M ⊕ F)) : Prop := 
  0 < h.card

theorem total_arrangement_methods :
  let male_doctors : Finset M := Finset.univ.filter (λ (m : M), m ∈ univ)
  let female_doctors : Finset F := Finset.univ.filter (λ (f : F), f ∈ univ)
  let hospitals : Finset (M ⊕ F) := univ

  (∀ (h : Finset (M ⊕ F)), at_least_one_doctor h) →
  (∀ (m1 m2 : M), m1 ≠ m2 → none ∈ hospitals \ (finset.filter m1 hospitals) ∩ finset.filter m2 hospitals) →
  Fintype.card {arrangement // ∃ (A : Finset M) (B : Finset F)
      (C : Finset (M ⊕ F)), 
      A ∪ (C \ (C.filter A)) ∪ (C \ (C.union (A.filter B))) ∈ univ 
      ∧ (∀ i ∈ A, i ∈ hospitals \ B) 
      ∧ ∀ (j : F), j ∈ B} = 18 := 
sorry

end total_arrangement_methods_l137_137822


namespace central_angle_regular_hexagon_l137_137329

theorem central_angle_regular_hexagon :
  ∀ (circle_degrees hexagon_sides : ℕ), circle_degrees = 360 ∧ hexagon_sides = 6 → circle_degrees / hexagon_sides = 60 :=
by {
  intros circle_degrees hexagon_sides h,
  cases h with h_circle h_hexagon,
  rw [h_circle, h_hexagon],
  norm_num,
  done
}

end central_angle_regular_hexagon_l137_137329


namespace medians_concurrent_l137_137037

variables {T : Type} [linear_ordered_field T]
variables {A B C A' B' C' P : T × T}
variables {side1 side2 side3 : T}

-- Conditions
def midpoint (X Y M : T × T) : Prop := 2 * (fst M) = (fst X + fst Y) ∧ 2 * (snd M) = (snd X + snd Y)
def median_intersection (G X Y X' : T × T) : Prop := ∃ K : T, 2 * (K * (fst G - fst X)) = fst G - fst X' ∧ 2 * (K * (snd G - snd X)) = snd G - snd X'

-- Definitions of areas of small triangles formed by medians
def area_eq (a1 a2 a3 : T) : Prop := a1 = a2 ∧ a2 = a3

theorem medians_concurrent
  (h1 : midpoint B C A')
  (h2 : midpoint A C B')
  (h3 : midpoint A B C')
  (h_inter : median_intersection P A A' P) 
  (h_inter2 : median_intersection P B B' P)
  (h_area : area_eq side1 side2 ∧ area_eq side2 side3) :
  ∃ G : T × T, median_intersection G A A' G ∧ median_intersection G B B' G ∧ median_intersection G C C' G := sorry

end medians_concurrent_l137_137037


namespace steps_to_zero_iff_multiple_of_31_l137_137008

theorem steps_to_zero_iff_multiple_of_31 (N : ℕ) (hN_positive : 0 < N) :
  (∃ a : ℕ, (iter_step N a = 0) ∧ (a < 10)) ↔ 31 ∣ N :=
sorry -- Proof is omitted

end steps_to_zero_iff_multiple_of_31_l137_137008


namespace necessary_but_not_sufficient_l137_137083

variables {α β a b : Type}

-- Definitions given in the problem conditions
variables (plane α plane β : Type) (line a : α) (line b : β)
def prop_p : Prop := ∀ common_point, common_point ∈ a → common_point ∈ b → false
def prop_q : Prop := parallel α β

-- The statement to be proven
theorem necessary_but_not_sufficient :
  (prop_q → prop_p) ∧ ¬(prop_p → prop_q) := by
    sorry

end necessary_but_not_sufficient_l137_137083


namespace remainder_of_T_mod_1000_l137_137987

noncomputable def binomial_coefficient := Nat.choose

def T : ℕ :=
  ∑ n in Finset.range 501, (-1)^n * binomial_coefficient 1000 (4 * n)

theorem remainder_of_T_mod_1000 (r : ℕ) : 
  T % 1000 = r := 
sorry

end remainder_of_T_mod_1000_l137_137987


namespace problem_892_10000_digits_and_place_values_l137_137767

-- Definitions for number of digits and highest place value
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

def highest_place_value (n : ℕ) : ℕ :=
  if n = 0 then 1 else 10 ^ nat.log10 n

-- Main theorem statements for the problem
theorem problem_892_10000_digits_and_place_values :
  num_digits 892 = 3 ∧ highest_place_value 892 = 100 ∧
  num_digits 10000 = 5 ∧ highest_place_value 10000 = 10000 :=
by
  sorry

end problem_892_10000_digits_and_place_values_l137_137767


namespace lana_total_pages_l137_137970

theorem lana_total_pages (lana_initial_pages : ℕ) (duane_total_pages : ℕ) :
  lana_initial_pages = 8 ∧ duane_total_pages = 42 →
  (lana_initial_pages + duane_total_pages / 2) = 29 :=
by
  sorry

end lana_total_pages_l137_137970


namespace non_congruent_squares_on_grid_l137_137912

theorem non_congruent_squares_on_grid :
  let n := 7 in
  let count_hv_squares := (n-1)^2 + (n-2)^2 + (n-3)^2 + (n-4)^2 + (n-5)^2 + (n-6)^2 in
  let count_diag_squares := (n-1)^2 + (n-2)^2 in
  let count_rect_diag_squares := (n-1)*(n-2) + (n-2)*(n-1) + (n-1)*(n-3) + (n-3)*(n-1) in
  count_hv_squares + count_diag_squares + count_rect_diag_squares = 284 :=
by
  let n := 7
  let count_hv_squares := (n - 1)^2 + (n - 2)^2 + (n - 3)^2 + (n - 4)^2 + (n - 5)^2 + (n - 6)^2
  let count_diag_squares := (n - 1)^2 + (n - 2)^2
  let count_rect_diag_squares := (n - 1) * (n - 2) + (n - 2) * (n - 1) + (n - 1) * (n - 3) + (n - 3) * (n - 1)
  have count_sum := count_hv_squares + count_diag_squares + count_rect_diag_squares
  show count_sum = 284 from sorry

end non_congruent_squares_on_grid_l137_137912


namespace find_f_of_4_l137_137928

-- Define the inverse function and the condition x > 0
def inv_function (x : ℝ) : ℝ := x^2
axiom positive_domain (x : ℝ) : x > 0

-- Problem statement
theorem find_f_of_4 : ∀ (x : ℝ), positive_domain x → inv_function x = 16 → f 4 = 2 :=
by
  sorry

end find_f_of_4_l137_137928


namespace series_sum_l137_137521

theorem series_sum (n : ℕ) : 
  (∑ i in Finset.range n, (1:ℝ) / ((i + 1) * (i + 2))) = n / (n + 1) := 
sorry

end series_sum_l137_137521


namespace second_divisor_203_l137_137839

theorem second_divisor_203 (x : ℕ) (h1 : 210 % 13 = 3) (h2 : 210 % x = 7) : x = 203 :=
by sorry

end second_divisor_203_l137_137839


namespace volume_of_box_l137_137392

theorem volume_of_box (l w h : ℝ) (h1 : l * w = 24) (h2 : w * h = 16) (h3 : l * h = 6) :
  l * w * h = 48 :=
by
  sorry

end volume_of_box_l137_137392


namespace stacy_days_to_finish_l137_137690

-- Definitions based on the conditions
def total_pages : ℕ := 81
def pages_per_day : ℕ := 27

-- The theorem statement
theorem stacy_days_to_finish : total_pages / pages_per_day = 3 := by
  -- the proof is omitted
  sorry

end stacy_days_to_finish_l137_137690


namespace lemon_juice_needed_for_one_dozen_cupcakes_l137_137967

theorem lemon_juice_needed_for_one_dozen_cupcakes 
  (lemon_juice_per_lemon : ℕ)
  (lemons_for_three_dozen : ℕ)
  (juice_for_three_dozen : ℕ)
  (H1 : lemon_juice_per_lemon = 4)
  (H2 : lemons_for_three_dozen = 9)
  (H3 : juice_for_three_dozen = lemons_for_three_dozen * lemon_juice_per_lemon) :
  juice_for_three_dozen / 3 = 12 := 
by
  have H4 : juice_for_three_dozen = 9 * 4, from eq.trans H3 (by rw [H2, H1]),
  rw H4,
  norm_num

end lemon_juice_needed_for_one_dozen_cupcakes_l137_137967


namespace max_value_of_f_l137_137347

noncomputable def f (x : ℝ) : ℝ := sin x - (real.sqrt 3 * cos x)

theorem max_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f(x) ≥ f(y) ∧ f(x) = 2 :=
sorry

end max_value_of_f_l137_137347


namespace sum_in_base3_l137_137391

def base3_representation (n : ℕ) : ℕ ∧ ℕ := 
  let digit0 := n % 3
  let digit1 := (n / 3) % 3
  let digit2 := (n / (3^2)) % 3
  let digit3 := (n / (3^3)) % 3
  let digit4 := (n / (3^4)) % 3
  (digit0 + 10 * digit1 + 100 * digit2 + 1000 * digit3 + 10000 * digit4, sorry)

theorem sum_in_base3 : base3_representation (30 + 45) = (22010, sorry) :=
  sorry

end sum_in_base3_l137_137391


namespace max_elements_in_T_l137_137644

def T := { x : ℕ | 1 ≤ x ∧ x ≤ 100 }

theorem max_elements_in_T : 
  ∃ (T : set ℕ), T ⊆ {1..100} ∧ 
  (∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a + b) % 5 ≠ 0) ∧ 
  (∀ U, U ⊆ {1..100} ∧ (∀ (a b : ℕ), a ∈ U → b ∈ U → a ≠ b → (a + b) % 5 ≠ 0) → U.card ≤ 41) :=
begin
  sorry
end

end max_elements_in_T_l137_137644


namespace limit_evaluation_l137_137599

open Classical

variables {a b x0 : ℝ}
variables {f : ℝ → ℝ}

-- Definitions from the conditions
def is_differentiable (f : ℝ → ℝ) (a b : ℝ) := ∀ x ∈ set.Ioo a b, ∃ f' : ℝ → ℝ, differentiable_at ℝ f x
def derivative_at (f : ℝ → ℝ) (x0 : ℝ) := ∀ h, tendsto (λ h, (f (x0) - f (x0 - 2 * h)) / h) (𝓝 0) (𝓝 (4 : ℝ))

-- The proof statement itself
theorem limit_evaluation 
  (h1 : x0 ∈ set.Ioo a b)
  (h2 : is_differentiable f a b)
  (h3 : (derivative_at f x0)) : 
  tendsto (λ h, (f (x0) - f (x0 - 2 * h)) / h) (𝓝 0) (𝓝 8) :=
sorry

end limit_evaluation_l137_137599


namespace image_of_center_l137_137473

def original_center : ℤ × ℤ := (3, -4)

def reflect_x (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)
def reflect_y (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, p.2)
def translate_down (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1, p.2 - d)

theorem image_of_center :
  (translate_down (reflect_y (reflect_x original_center)) 10) = (-3, -6) :=
by
  sorry

end image_of_center_l137_137473


namespace max_trig_expression_l137_137504

theorem max_trig_expression (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_trig_expression_l137_137504


namespace car_lease_annual_cost_l137_137378

/-- Tom decides to lease a car. He drives 50 miles on four specific days a week,
and 100 miles on the other three days. He pays $0.1 per mile and a weekly fee of $100.
Prove that the total annual cost he has to pay is $7800.
--/
theorem car_lease_annual_cost :
  let weekly_miles := 4 * 50 + 3 * 100
      weekly_mileage_cost := weekly_miles * 0.1
      weekly_total_cost := weekly_mileage_cost + 100
      annual_cost := weekly_total_cost * 52
  in annual_cost = 7800 :=
by
  let weekly_miles := 4 * 50 + 3 * 100
  let weekly_mileage_cost := weekly_miles * 0.1
  let weekly_total_cost := weekly_mileage_cost + 100
  let annual_cost := weekly_total_cost * 52
  sorry

end car_lease_annual_cost_l137_137378


namespace card_numbers_l137_137372

def numbers_on_cards : List ℕ := [3, 5, 9]

theorem card_numbers (x y z : ℕ) (h1 : x + y + z = 17) 
  (h2 : 13 + 15 + 23 = 3 * (x + y + z))
  (h3 : x ≤ y) (h4 : y ≤ z) (h5 : x ≥ 1) (h6 : z ≤ 10) :
  List ℕ := numbers_on_cards := 
begin
  sorry
end

end card_numbers_l137_137372


namespace prove_difference_l137_137365

theorem prove_difference (x y : ℝ) (h1 : x + y = 500) (h2 : x * y = 22000) : y - x = -402.5 :=
sorry

end prove_difference_l137_137365


namespace math_problem_l137_137976

variable (n : ℕ)
variable (a b : Fin n → ℝ)

theorem math_problem
    (h1 : n ≥ 3)
    (h2 : ∀ i, 0 < a i)
    (h3 : (Finset.univ : Finset (Fin n)).sum a = 1)
    (h4 : (Finset.univ : Finset (Fin n)).sum (λ i => (b i)^2) = 1)
    : (Finset.univ : Finset (Fin n)).sum (λ i => a i * (b i + a ((i + 1) % n))) < 1 := sorry

end math_problem_l137_137976


namespace sqrt_cos_poly_value_correct_l137_137046

noncomputable def sqrt_cos_poly_value (θ1 θ2 θ3 : ℝ) : ℝ :=
  real.sqrt ((3 - cos θ1 ^ 2) * (3 - cos θ2 ^ 2) * (3 - cos θ3 ^ 2))

theorem sqrt_cos_poly_value_correct :
  (∀ k : ℕ, k = 1 ∨ k = 2 ∨ k = 4 → cos (k * real.pi / 9 * 9) = 0) ∧
  (∀ x : ℝ, x = cos (real.pi / 9) ^ 2 ∨ x = cos (2 * real.pi / 9) ^ 2 ∨ x = cos (4 * real.pi / 9) ^ 2 ↔ x^4 - 126 * x^3 + 441 * x^2 - 490 * x + 121 = 0) →
  sqrt_cos_poly_value (real.pi / 9) (2 * real.pi / 9) (4 * real.pi / 9) = 11 / 9 :=
by
  sorry

end sqrt_cos_poly_value_correct_l137_137046


namespace find_line_equation_l137_137568

open Real

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the line passing through (0,2)
def LineThruPoint (x y k : ℝ) : Prop := y = k * x + 2

-- Define when line intersects parabola
def LineIntersectsParabola (x1 y1 x2 y2 k : ℝ) : Prop :=
  LineThruPoint x1 y1 k ∧ LineThruPoint x2 y2 k ∧ Parabola x1 y1 ∧ Parabola x2 y2

-- Define when circle with diameter MN passes through origin O
def CircleThroughOrigin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem find_line_equation (k : ℝ) 
    (h₀ : k ≠ 0)
    (h₁ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k)
    (h₂ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k ∧ CircleThroughOrigin x1 y1 x2 y2) :
  (∃ x y, LineThruPoint x y k ∧ y = -x + 2) :=
sorry

end find_line_equation_l137_137568


namespace avg_and_variance_decrease_l137_137343

noncomputable def original_heights : List ℝ := [180, 184, 188, 190, 192, 194]
noncomputable def new_heights : List ℝ := [180, 184, 188, 190, 192, 188]

noncomputable def avg (heights : List ℝ) : ℝ :=
  heights.sum / heights.length

noncomputable def variance (heights : List ℝ) (mean : ℝ) : ℝ :=
  (heights.map (λ h => (h - mean) ^ 2)).sum / heights.length

theorem avg_and_variance_decrease :
  let original_mean := avg original_heights
  let new_mean := avg new_heights
  let original_variance := variance original_heights original_mean
  let new_variance := variance new_heights new_mean
  new_mean < original_mean ∧ new_variance < original_variance :=
by
  sorry

end avg_and_variance_decrease_l137_137343


namespace length_sawed_off_l137_137418

-- Define the lengths as constants
def original_length : ℝ := 8.9
def final_length : ℝ := 6.6

-- State the property to be proven
theorem length_sawed_off : original_length - final_length = 2.3 := by
  sorry

end length_sawed_off_l137_137418


namespace parallelogram_area_l137_137992

open Real

variables (r s : ℝ^3)
-- Condition 1: r and s are unit vectors
hypothesis (norm_r : ‖r‖ = 1)
hypothesis (norm_s : ‖s‖ = 1)
-- Condition 2: The angle between r and s is 45 degrees
hypothesis (angle_45 : real.inner_product_space.angle r s = π / 4)

-- Define the vectors u and v given their relationships
def u : ℝ^3 := (3 * s - r) / 2
def v : ℝ^3 := (3 * r + 3 * s) / 2

-- Prove that the area of the parallelogram with given diagonals is 9√2 / 4
theorem parallelogram_area : 
  let diag1 := r + 3 * s,
      diag2 := 3 * r + s in
  by sorry
/equivalent to/ ‖u × v‖ = 9 * sqrt 2 / 4 := sorry

end parallelogram_area_l137_137992


namespace chords_squared_product_eq_five_l137_137284

theorem chords_squared_product_eq_five (A1 A2 A3 A4 A5 : Point)
(circle : Circle)
(h_division : div_perfectly_circle five_eq_parts circle (points := [A1, A2, A3, A4, A5]))
(radius_unit : circle.radius = 1) :
  ((chord_length circle A1 A2) * (chord_length circle A1 A3))^2 = 5 := 
  sorry

end chords_squared_product_eq_five_l137_137284


namespace triangle_construction_l137_137810

theorem triangle_construction (BC : Real) (alpha : Real) (D : Real) :
  ∃ (A B C : Point), -- Points representing vertices of the triangle
  BC = dist B C ∧   -- Condition for BC side length
  angle A B C = alpha ∧ -- Condition for alpha angle
  incenter BC alpha D = -- Function defining the incenter based on given conditions
  sorry

end triangle_construction_l137_137810


namespace polynomial_integral_equation_l137_137975

noncomputable def f (x : ℝ) : ℝ := (3 * x - 1) / 2
def C : ℝ := 5 / 24

theorem polynomial_integral_equation :
  (∫ y in (0 : ℝ)..x, f y) + (∫ y in (0 : ℝ)..(1 : ℝ), (x + y)^2 * f y) = x^2 + C :=
by
  sorry

end polynomial_integral_equation_l137_137975


namespace number_of_oarsmen_l137_137721

theorem number_of_oarsmen (n : ℕ) (h1 : 1.8 * n = 18) : n = 10 :=
by
  sorry

end number_of_oarsmen_l137_137721


namespace online_textbooks_cost_l137_137664

theorem online_textbooks_cost (x : ℕ) :
  (5 * 10) + x + 3 * x = 210 → x = 40 :=
by
  sorry

end online_textbooks_cost_l137_137664


namespace high_sulfur_oil_samples_count_l137_137779

/--
Given:
- The total number of samples in the container is 296.
- The probability of a sample being a heavy oil sample is 1/8.
- The probability of a sample being a light low-sulfur oil sample among all samples is 22/37.
- There are no low-sulfur samples among the heavy oil samples.
Prove:
- The number of high-sulfur oil samples in the container is 142.
-/
theorem high_sulfur_oil_samples_count :
  let total_samples := 296
  let heavy_oil_probability := 1 / 8
  let light_low_sulfur_oil_probability := 22 / 37
  let heavy_oil_samples := Int.ofNat(total_samples * heavy_oil_probability)
  let light_oil_samples := Int.ofNat(total_samples - heavy_oil_samples)
  let light_low_sulfur_oil_samples := Int.ofNat(light_oil_samples * light_low_sulfur_oil_probability)
  let light_high_sulfur_oil_samples := light_oil_samples - light_low_sulfur_oil_samples
  let total_high_sulfur_oil_samples := heavy_oil_samples + light_high_sulfur_oil_samples
  total_high_sulfur_oil_samples = 142 :=
by sorry

end high_sulfur_oil_samples_count_l137_137779


namespace digits_in_Q_l137_137234

def a : ℕ := 4859832567883512999
def b : ℕ := a + 1
def c : ℕ := 5234987654321678
def Q : ℕ := b * c

theorem digits_in_Q : (Q.log10.to_nat + 1) = 41 := 
by
  -- proof can be filled here
  sorry

end digits_in_Q_l137_137234


namespace carmen_more_miles_l137_137333

-- Definitions for the conditions
def carmen_distance : ℕ := 90
def daniel_distance : ℕ := 75

-- The theorem statement
theorem carmen_more_miles : carmen_distance - daniel_distance = 15 :=
by
  sorry

end carmen_more_miles_l137_137333


namespace sin_theta_is_sqrt_15_over_4_l137_137649

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (θ : ℝ)

-- Given Conditions
def non_zero (x : V) : Prop := x ≠ 0
def not_parallel (x y : V) : Prop := ∀ k: ℝ, x ≠ k • y
def condition (a b c : V) : Prop := (a × b) × c = (1/4 : ℝ) * ∥b∥ * ∥c∥ • a
def angle_between (b c : V) (θ : ℝ) : Prop := real_inner b c = ∥b∥ * ∥c∥ * real.cos θ

-- Prove sin θ of angle between b and c is √15/4
theorem sin_theta_is_sqrt_15_over_4
  (ha : non_zero a) (hb : non_zero b) (hc : non_zero c)
  (ha_b : not_parallel a b) (ha_c : not_parallel a c) (hb_c : not_parallel b c)
  (hcond : condition a b c)
  (θ : ℝ) (hθ : angle_between b c θ) :
  real.sin θ = (real.sqrt 15) / 4 :=
sorry

end sin_theta_is_sqrt_15_over_4_l137_137649


namespace factor_polynomial_l137_137057

theorem factor_polynomial :
  (x : ℝ) → (x^2 + 4x + 4 - 81x^4 = (-9x^2 + x + 2) * (9x^2 + x + 2)) :=
by
  intro x
  sorry

end factor_polynomial_l137_137057


namespace six_inch_cube_value_is_2700_l137_137772

noncomputable def value_of_six_inch_cube (value_four_inch_cube : ℕ) : ℕ :=
  let volume_four_inch_cube := 4^3
  let volume_six_inch_cube := 6^3
  let scaling_factor := volume_six_inch_cube / volume_four_inch_cube
  value_four_inch_cube * scaling_factor

theorem six_inch_cube_value_is_2700 : value_of_six_inch_cube 800 = 2700 := by
  sorry

end six_inch_cube_value_is_2700_l137_137772


namespace polynomial_solution_l137_137829
-- Import necessary library

-- Define the property to be checked
def polynomial_property (P : Real → Real) : Prop :=
  ∀ a b c : Real, 
    P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

-- The statement that needs to be proven
theorem polynomial_solution (a b : Real) : polynomial_property (λ x => a * x^2 + b * x) := 
by
  sorry

end polynomial_solution_l137_137829


namespace vote_ratio_l137_137412

-- Given the side lengths AB, AC, and BC in a specific ratio
variables {k : ℝ} (AB AC BC : ℝ)
variable (votes : ℝ)

-- The sides in the ratio of 5:12:13
def sides_ratio : Prop := 
  AB = 5 * k ∧ AC = 12 * k ∧ BC = 13 * k

-- To prove the voting ratio 2:3:3
theorem vote_ratio (AB AC BC : ℝ) (h : sides_ratio AB AC BC) : 
  ∃ (ratio : ℚ), ratio = 2/5 := 
sorry

end vote_ratio_l137_137412


namespace central_angle_regular_hexagon_l137_137330

theorem central_angle_regular_hexagon :
  ∀ (circle_degrees hexagon_sides : ℕ), circle_degrees = 360 ∧ hexagon_sides = 6 → circle_degrees / hexagon_sides = 60 :=
by {
  intros circle_degrees hexagon_sides h,
  cases h with h_circle h_hexagon,
  rw [h_circle, h_hexagon],
  norm_num,
  done
}

end central_angle_regular_hexagon_l137_137330


namespace range_of_a_l137_137344

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 2 ^ (x^2 - 4 * x) > 2 ^ (2 * a * x + a)) :
  -4 < a ∧ a < -1 :=
by
  /-
  Here's a brief explanation of the proof that goes here:
  Given the exponential inequality, converting it to a quadratic form in terms of \( x \),
  and ensuring the quadratic equation has no real roots, i.e., its discriminant is negative.
  Then solving for \( a \).
  -/
  sorry

end range_of_a_l137_137344


namespace complex_conjugate_problem_l137_137244

open Complex

theorem complex_conjugate_problem :
  let z := 2 * (3 + 5 * Complex.i) / (1 - Complex.i)^2 in
  Complex.conj z = -5 - 3 * Complex.i :=
by
  -- Definitions
  let z := 2 * (3 + 5 * Complex.i) / (1 - Complex.i)^2
  -- The theorem to be proven
  show Complex.conj z = -5 - 3 * Complex.i
  sorry

end complex_conjugate_problem_l137_137244


namespace max_profit_achieved_at_m_eq_3_l137_137002

variable (m : ℝ) (x y : ℝ)

-- Defining the conditions
def technical_reform (m : ℝ) : Prop := (m ≥ 0)

def annual_production (m : ℝ) : ℝ := 3 - 2/(m + 1)

def profit (m : ℝ) : ℝ := 28 - m - 16/(m + 1)

theorem max_profit_achieved_at_m_eq_3 (h : technical_reform m) : 
  (m = 3) ∧ (profit 3 = 21) :=
sorry

end max_profit_achieved_at_m_eq_3_l137_137002


namespace range_of_m_l137_137927

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_incr : ∀ x y, x < y → f x < f y) : 
  f (2 * m - 1) + f (3 - m) > 0 ↔ m > -2 := 
by 
  sorry

end range_of_m_l137_137927


namespace most_accurate_value_announcement_l137_137945

theorem most_accurate_value_announcement :
  ∀ (D : ℝ), D = 3.58247 → ∀ (error : ℝ), error = ±0.00418 →
  (∀ D_upper D_lower, D_upper = D + error → D_lower = D - error → 
  (D_upper_round_1 = 3.6 ∧ D_lower_round_1 = 3.6 →
  D_announce = 3.6)) :=
begin
  intros D HD error herr D_upper_lower D_lower,
  sorry
end

end most_accurate_value_announcement_l137_137945


namespace fractional_equation_solution_l137_137889

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l137_137889


namespace grocery_store_distance_l137_137319

theorem grocery_store_distance 
    (park_house : ℕ) (park_store : ℕ) (total_distance : ℕ) (grocery_store_house: ℕ) :
    park_house = 5 ∧ park_store = 3 ∧ total_distance = 16 → grocery_store_house = 8 :=
by 
    sorry

end grocery_store_distance_l137_137319


namespace pages_left_proof_l137_137281

-- Definitions based on the given conditions
def initial_books : ℕ := 10
def pages_per_book : ℕ := 100
def lost_books : ℕ := 2

-- Calculate the number of books left
def books_left : ℕ := initial_books - lost_books

-- Calculate the total number of pages left
def total_pages_left : ℕ := books_left * pages_per_book

-- Theorem statement representing the proof problem
theorem pages_left_proof : total_pages_left = 800 := by
  -- Call the necessary built-ins and perform the calculation steps within the proof.
  simp [total_pages_left, books_left, initial_books, lost_books, pages_per_book]
  sorry -- Proof steps will go here, or use a by computation or decidability tactics


end pages_left_proof_l137_137281


namespace euler_criterion_no_primitive_root_l137_137316

open Nat

/-- Given m does not have a primitive root and a is relatively prime to m,
prove that a^(φ(m)/2) ≡ 1 [MOD m]. -/
theorem euler_criterion_no_primitive_root
  (m : ℕ) (a : ℕ)
  (h1 : ¬ ∃ g, is_primitive_root g m)
  (h2 : gcd a m = 1)
  : a ^ (φ m / 2) ≡ 1 [MOD m] :=
sorry

end euler_criterion_no_primitive_root_l137_137316


namespace number_of_roses_cut_l137_137374

-- Let's define the initial and final conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses Mary cut from her garden
def roses_cut := final_roses - initial_roses

-- Now, we state the theorem we aim to prove
theorem number_of_roses_cut : roses_cut = 10 :=
by
  -- Proof goes here
  sorry

end number_of_roses_cut_l137_137374


namespace circle_diameter_MN_l137_137907

-- Defining the points M and N
def M : ℝ × ℝ := (0, 2)
def N : ℝ × ℝ := (2, -2)

-- Function to compute the midpoint
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Function to compute the squared Euclidean distance
def distance_squared (A B : ℝ × ℝ) : ℝ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

-- Equation of the circle
def circle_equation (O : ℝ × ℝ) (r_squared : ℝ) : ℝ × ℝ → Prop :=
  λ P, (P.1 - O.1) ^ 2 + (P.2 - O.2) ^ 2 = r_squared

-- Prove the circle's equation given M and N
theorem circle_diameter_MN :
  let O := midpoint M N
  let r_squared := distance_squared M N / 4
  O = (1, 0) ∧ r_squared = 5 →
  ∀ P : ℝ × ℝ, circle_equation O 5 P ↔ (P.1 - 1) ^ 2 + P.2 ^ 2 = 5
:=
by
  intro O r_squared h
  rcases h with ⟨hO, hr⟩
  simp [midpoint, distance_squared, circle_equation] at *
  rw [hO, hr]
  intro P
  split <;> intro hs <;> linarith

end circle_diameter_MN_l137_137907


namespace original_bill_amount_l137_137766

/-- 
If 8 people decided to split the restaurant bill evenly and each paid $314.15 after rounding
up to the nearest cent, then the original bill amount was $2513.20.
-/
theorem original_bill_amount (n : ℕ) (individual_share : ℝ) (total_amount : ℝ) 
  (h1 : n = 8) (h2 : individual_share = 314.15) 
  (h3 : total_amount = n * individual_share) : 
  total_amount = 2513.20 :=
by
  sorry

end original_bill_amount_l137_137766


namespace angle_CAB_is_36_l137_137615

-- Define a regular pentagon has the interior angle of 108 degrees
def regular_pentagon_interior_angle (P : Type) [polygon P] : Prop :=
  ∀ (A B C D E : P), P.regular → P.interior_angle = 108

namespace pentagon_problem

variables {P : Type} [polygon P] (A B C D E : P) (hP : P.regular)
variables (h_interior_angle : ∀ (X Y Z : P), P.interior_angle = 108)

-- Prove that the angle CAB is 36 degrees
theorem angle_CAB_is_36 (hABC : triangle A B C) : angle A B C = 36 := by
  -- Define and use the interior angle condition
  have hABC_interior : angle B A C + angle A B C + angle B C A = 180 := by sorry
  -- Solve the equation to get the correct answer
  sorry

end pentagon_problem

end angle_CAB_is_36_l137_137615


namespace abs_m_minus_one_plus_m_eq_one_iff_l137_137588

theorem abs_m_minus_one_plus_m_eq_one_iff (m : ℝ) : |m - 1| + m = 1 → m ≤ 1 :=
begin
  sorry
end

end abs_m_minus_one_plus_m_eq_one_iff_l137_137588


namespace row_in_pascal_triangle_l137_137603

open Nat

noncomputable def binom (n k : ℕ) : ℚ := nat.binom n k

theorem row_in_pascal_triangle {n r : ℕ} (h1 : (binom n r) / (binom n (r + 1)) = 2 / 3)
                              (h2 : (binom n (r + 1)) / (binom n (r + 2)) = 3 / 4) :
                              n = 50 :=
by
  sorry

end row_in_pascal_triangle_l137_137603


namespace remainder_of_large_number_l137_137390

theorem remainder_of_large_number :
  (1235678901 % 101) = 1 :=
by
  have h1: (10^8 % 101) = 1 := sorry
  have h2: (10^6 % 101) = 1 := sorry
  have h3: (10^4 % 101) = 1 := sorry
  have h4: (10^2 % 101) = 1 := sorry
  have large_number_decomposition: 1235678901 = 12 * 10^8 + 35 * 10^6 + 67 * 10^4 + 89 * 10^2 + 1 := sorry
  -- Proof using the decomposition and modulo properties
  sorry

end remainder_of_large_number_l137_137390


namespace tom_annual_car_leasing_cost_l137_137380

theorem tom_annual_car_leasing_cost :
  let miles_mwf := 50 * 3  -- Miles driven on Monday, Wednesday, and Friday
  let miles_other_days := 100 * 4 -- Miles driven on the other days (Sunday, Tuesday, Thursday, Saturday)
  let weekly_miles := miles_mwf + miles_other_days -- Total miles driven per week

  let cost_per_mile := 0.1 -- Cost per mile
  let weekly_fee := 100 -- Weekly fee

  let weekly_cost := weekly_miles * cost_per_mile + weekly_fee -- Total weekly cost

  let weeks_per_year := 52
  let annual_cost := weekly_cost * weeks_per_year -- Annual cost

  annual_cost = 8060 :=
by
  sorry

end tom_annual_car_leasing_cost_l137_137380


namespace value_of_x_if_additive_inverses_l137_137143

theorem value_of_x_if_additive_inverses (x : ℝ) 
  (h : 4 * x - 1 + (3 * x - 6) = 0) : x = 1 := by
sorry

end value_of_x_if_additive_inverses_l137_137143


namespace find_a_5_in_arithmetic_sequence_l137_137950

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

theorem find_a_5_in_arithmetic_sequence (h : arithmetic_sequence a 1 2) : a 5 = 9 :=
sorry

end find_a_5_in_arithmetic_sequence_l137_137950


namespace pies_differ_in_both_l137_137166

-- Definitions of pie types
inductive Filling
| apple
| cherry

inductive PreparationMethod
| fried
| baked

structure Pie where
  filling : Filling
  method : PreparationMethod

-- The set of all possible pie types
def pies : Set Pie := {
  {filling := Filling.apple, method := PreparationMethod.fried},
  {filling := Filling.cherry, method := PreparationMethod.fried},
  {filling := Filling.apple, method := PreparationMethod.baked},
  {filling := Filling.cherry, method := PreparationMethod.baked}
}

-- The statement to prove: If there are at least three types of pies available, then there exist two pies that differ in both filling and preparation method.
theorem pies_differ_in_both :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.method ≠ p2.method :=
begin
  sorry
end

end pies_differ_in_both_l137_137166


namespace limit_of_function_l137_137497

theorem limit_of_function :
  tendsto (λ x : ℝ, ( (x + 1) / (2 * x) ) ^ ( real.log (x + 2) / real.log (2 - x) )) (nhds 1) (nhds (real.sqrt 3)) :=
sorry

end limit_of_function_l137_137497


namespace john_pin_discount_l137_137964

theorem john_pin_discount :
  ∀ (n_pins price_per_pin amount_spent discount_rate : ℝ),
    n_pins = 10 →
    price_per_pin = 20 →
    amount_spent = 170 →
    discount_rate = ((n_pins * price_per_pin - amount_spent) / (n_pins * price_per_pin)) * 100 →
    discount_rate = 15 :=
by
  intros n_pins price_per_pin amount_spent discount_rate h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end john_pin_discount_l137_137964


namespace compare_area_l137_137693

noncomputable def S₃ (P : ℝ) : ℝ := 
  let a := P / 3
  in (real.sqrt 3 / 4) * a^2

noncomputable def S₄ (P : ℝ) : ℝ := 
  let b := P / 4
  in b^2

noncomputable def S₆ (P : ℝ) : ℝ := 
  let c := P / 6
  in (3 * real.sqrt 3 / 2) * c^2

theorem compare_area (P : ℝ) (hP : P > 0) : S₆ P > S₄ P ∧ S₄ P > S₃ P := 
by
  sorry

end compare_area_l137_137693


namespace part_a_part_b_l137_137710

/-- Given a quadrilateral ABCD inscribed in a circle of radius R,
    where S_a, S_b, S_c, and S_d are circles of radius R centered at the orthocenters
    of triangles BCD, CAD, DAB, and ABC respectively.
    Prove that these four circles intersect at one point. --/
theorem part_a 
  (R : ℝ)
  (A B C D : point)
  (H : is_inscribed_quadrilateral ABCD)
  (orthocenter_BCD orthocenter_CAD orthocenter_DAB orthocenter_ABC : point)
  (circumradius_BCD : distance orthocenter_BCD = R)
  (circumradius_CAD : distance orthocenter_CAD = R)
  (circumradius_DAB : distance orthocenter_DAB = R)
  (circumradius_ABC : distance orthocenter_ABC = R) : 
  ∃ K : point, is_on_circle K R orthocenter_BCD ∧ 
               is_on_circle K R orthocenter_CAD ∧ 
               is_on_circle K R orthocenter_DAB ∧ 
               is_on_circle K R orthocenter_ABC := sorry

/-- Given a quadrilateral ABCD inscribed in a circle of radius R,
    where the nine-point circles of triangles ABC, BCD, CDA, and DAB intersect at one point.
    Prove that these nine-point circles intersect at one point. --/
theorem part_b 
  (R : ℝ)
  (A B C D : point)
  (H : is_inscribed_quadrilateral ABCD)
  (nine_point_center_ABC nine_point_center_BCD 
   nine_point_center_CDA nine_point_center_DAB : point)
  (nine_point_radius_ABC : distance nine_point_center_ABC = R / 2)
  (nine_point_radius_BCD : distance nine_point_center_BCD = R / 2)
  (nine_point_radius_CDA : distance nine_point_center_CDA = R / 2)
  (nine_point_radius_DAB : distance nine_point_center_DAB = R / 2) : 
  ∃ X : point, is_on_circle X (R / 2) nine_point_center_ABC ∧ 
               is_on_circle X (R / 2) nine_point_center_BCD ∧ 
               is_on_circle X (R / 2) nine_point_center_CDA ∧ 
               is_on_circle X (R / 2) nine_point_center_DAB := sorry

end part_a_part_b_l137_137710


namespace christen_potatoes_l137_137911

theorem christen_potatoes :
  let total_potatoes := 60
  let homer_rate := 4
  let christen_rate := 6
  let alex_potatoes := 2
  let homer_minutes := 6
  homer_minutes * homer_rate + christen_rate * ((total_potatoes + alex_potatoes - homer_minutes * homer_rate) / (homer_rate + christen_rate)) = 24 := 
sorry

end christen_potatoes_l137_137911


namespace probability_not_equal_genders_l137_137271

noncomputable def probability_more_grandsons_or_more_granddaughters : ℚ :=
  let total_ways := 2 ^ 12
  let equal_distribution_ways := (Nat.choose 12 6)
  let probability_equal := (equal_distribution_ways : ℚ) / (total_ways : ℚ)
  1 - probability_equal

theorem probability_not_equal_genders (n : ℕ) (p : ℚ) (hp : p = 1 / 2) (hn : n = 12) :
  probability_more_grandsons_or_more_granddaughters = 793 / 1024 :=
by
  sorry

end probability_not_equal_genders_l137_137271


namespace find_P_l137_137276

def is_red_point (T : ℤ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ T = 81 * ↑x + 100 * ↑y

def is_blue_point (T : ℤ) : Prop :=
  ¬ is_red_point T

def is_reflection_red (P T : ℤ) : Prop :=
  is_red_point (2 * P - T)

def is_reflection_blue (P T : ℤ) : Prop :=
  is_blue_point (2 * P - T)

theorem find_P :
  ∃ P : ℤ, ∀ T : ℤ,
    (is_red_point T → is_blue_point (2 * P - T)) ∧ 
    (is_blue_point T → is_red_point (2 * P - T)) :=
begin
  use 3960,
  sorry
end

end find_P_l137_137276


namespace probability_of_at_least_one_correct_pairing_l137_137749

theorem probability_of_at_least_one_correct_pairing (n : ℕ) :
  (1 - ∑ k in finset.range (n + 1), (-1 : ℝ) ^ k / nat.factorial k) =
  ∑ k in finset.range (n + 1), (-1 : ℝ) ^ (k + 1) / nat.factorial k :=
by
  sorry

end probability_of_at_least_one_correct_pairing_l137_137749


namespace pipe_segment_probability_l137_137444

-- Define the problem domain and relevant parameters
def gas_pipe_probability : Prop :=
  let L := 200 in  -- Total length of the gas pipe in cm
  let min_segment := 50 in  -- Minimum segment length in cm
  let total_area := (L * L) / 2 in  -- Area of the right triangle 200 x 200
  let valid_area := ((L - min_segment) * (L - min_segment)) / 2 in  -- Area of the valid right triangle 150 x 150
  let probability := valid_area / total_area in
  probability = 1 / 16

-- Statement to be proven
theorem pipe_segment_probability : gas_pipe_probability :=
by
  sorry

end pipe_segment_probability_l137_137444


namespace recursive_formula_sequence_l137_137357

noncomputable def a : ℕ → ℕ
| 0       := 0          -- We define a(0) = 0 for convenience
| 1       := 1
| (n + 1) := a n + (n + 1)

theorem recursive_formula_sequence :
  ∀ n >= 1, a (n + 1) = a n + (n + 1) :=
by
  intro n hn
  induction n with
  | zero => simp
  | succ n ih =>
    cases n
    simp
    unfold a
    sorry

end recursive_formula_sequence_l137_137357


namespace two_pies_differ_l137_137181

theorem two_pies_differ (F_A F_C B_A B_C : Bool) :
  (F_A ∨ F_C ∨ B_A ∨ B_C) →
  (F_A ∧ F_C ∧ B_A ∧ B_C) ∧ 
  (∀ a b, (a ≠ b) → (a.filling ≠ b.filling ∧ a.preparation ≠ b.preparation)) :=
by
  intros H1 H2
  sorry

end two_pies_differ_l137_137181


namespace taverns_reduction_l137_137024

variable a b c : Nat
variable days : Nat
variable more_than_half : Nat → Nat → Prop
variable divisible_by_5 : Nat → Prop
axiom decree (n : Nat) : (more_than_half a n → divisible_by_5 a → a = a * 1 / 5) ∨
                         (more_than_half b n → divisible_by_5 b → b = b * 1 / 5) ∨
                         (more_than_half c n → divisible_by_5 c → c = c * 1 / 5)

def init_taverns_1 : Nat := 60
def init_taverns_2 : Nat := 35
def init_taverns_3 : Nat := 20

def after_three_days (t1 t2 t3 : Nat) : Prop :=
  t1 < 60 ∧ t2 < 35 ∧ t3 < 20

theorem taverns_reduction :
  ∃ t1 t2 t3 : Nat, init_taverns_1 = t1 ∧ init_taverns_2 = t2 ∧ init_taverns_3 = t3 ∧
                 (after_three_days ((t1 / 5) + (t2 / 5) + (t3 / 5)) 
                   ((t1 / 5) + (t2 / 5) + (t3 / 5)) 
                   ((t1 / 5) + (t2 / 5) + (t3 / 5))) :=
by {
  use [12, 7, 4],
  sorry
}

end taverns_reduction_l137_137024


namespace find_angle_MLC_l137_137979

-- Definitions from the conditions
variables {A B C K L M : Type} [triangle : Triangle A B C]
variables (h1 : Between K A B) (h2 : Between L A B)
variables (h3 : ∠ A C K = ∠ K C L) (h4 : ∠ K C L = ∠ L C B)
variables (h5 : Between M B C) (h6 : ∠ M K C = ∠ B K M)
variables (h7 : isAngleBisector (Line.mk M L) (∠ K M B))

-- The goal is to prove that ∠ M L C = 30°
theorem find_angle_MLC : ∠ M L C = 30° :=
sorry

end find_angle_MLC_l137_137979


namespace part1_min_value_part1_min_value_achievable_part2_range_of_a_l137_137659

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1_min_value (x : ℝ) : f x (-3) ≥ 4 :=
sorry

theorem part1_min_value_achievable (x : ℝ) : -3 ≤ x ∧ x ≤ 1 → f x (-3) = 4 :=
sorry

theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≤ 2 * a + 2 * |x - 1|) → a ≥ 1 / 3 :=
sorry

#eval part1_min_value 1 -- This should print "True" since 4 ≥ 4 is true (proof omitted)
#eval part2_range_of_a 1 -- This should print "False" since 1 < 1/3 (proof omitted)

end part1_min_value_part1_min_value_achievable_part2_range_of_a_l137_137659


namespace limit_of_function_l137_137498

theorem limit_of_function :
  tendsto (λ x : ℝ, ( (x + 1) / (2 * x) ) ^ ( real.log (x + 2) / real.log (2 - x) )) (nhds 1) (nhds (real.sqrt 3)) :=
sorry

end limit_of_function_l137_137498


namespace prob_even_first_odd_second_dice_roll_l137_137665

def six_sided_die := {1, 2, 3, 4, 5, 6}

def probability (event : Set ℕ) : ℚ :=
  event.card.toRat / six_sided_die.card.toRat

def even_numbers := {2, 4, 6}
def odd_numbers := {1, 3, 5}

theorem prob_even_first_odd_second_dice_roll :
  probability even_numbers * probability odd_numbers = 1 / 4 :=
by
  -- Skipping the proof step
  sorry

end prob_even_first_odd_second_dice_roll_l137_137665


namespace fractional_equation_solution_l137_137887

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l137_137887


namespace triangle_median_problem_l137_137217

theorem triangle_median_problem 
  (YZ : ℝ) (hYZ : YZ = 10) 
  (XM : ℝ) (hXM : XM = 7) 
  : let N := max_XZ_XY_square_sum YZ XM
    let n := min_XZ_XY_square_sum YZ XM
    N - n = 0 := by 
sorry

noncomputable def max_XZ_XY_square_sum (YZ XM : ℝ) : ℝ := sorry
noncomputable def min_XZ_XY_square_sum (YZ XM : ℝ) : ℝ := sorry

end triangle_median_problem_l137_137217


namespace matrix_problem_l137_137150

variable {α : Type*} [Field α] [DecidableEq α]

theorem matrix_problem (A : Matrix (Fin 2) (Fin 2) α) (I : Matrix (Fin 2) (Fin 2) α) (hAinv : A.det ≠ 0)
  (hcond : (A - 3 • I) ⬝ (A - 5 • I) = 0) :
  A + 10 • A⁻¹ = 8 • I :=
sorry

end matrix_problem_l137_137150


namespace coefficient_of_x6_in_expansion_proof_l137_137735

noncomputable def coefficient_of_x6_in_expansion : Nat :=
  90720

theorem coefficient_of_x6_in_expansion_proof :
  let p := 3
  let q := 2
  let n := 8
  (∑ k in Finset.range (n+1), Nat.choose n k * (p*x)^(n-k) * q^k) =
  coefficient_of_x6_in_expansion :=
sorry

end coefficient_of_x6_in_expansion_proof_l137_137735


namespace ratio_OC_OA_half_l137_137617

variables (s : ℝ) (A B C D N O : point)

def Square (A B C D : point) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s ∧
  dist A C = s * real.sqrt 2 ∧ dist B D = s * real.sqrt 2

def point_on_line_segment (x y P : point) (r : ℝ) : Prop :=
  dist P x = r * dist x y ∧ dist P y = (1 - r) * dist x y

def intersection_point (l1 l2 : line) (P : point) : Prop :=
  P ∈ l1 ∧ P ∈ l2

open_locale classical

theorem ratio_OC_OA_half :
  Square A B C D s →
  dist N C = (1/3 : ℝ) * s →
  dist N D = (2/3 : ℝ) * s →
  intersection_point (line_through A C) (line_through B N) O →
  dist O C / dist O A = 1 / 2 :=
by {
  intros hSquare hNC hND hIntersect,
  sorry
}

end ratio_OC_OA_half_l137_137617


namespace skw_Sn_kur_Sn_l137_137256

open ProbabilityTheory

variables {n : ℕ}
variables {ξ : ℕ → ℝ}

def Ex (X : ℝ) : ℝ := sorry  -- Expectation of X
def Var (X : ℝ) : ℝ := sorry -- Variance of X

-- Skewness parameter
def skw (ξ : ℝ) : ℝ := Ex ((ξ - Ex ξ) ^ 3) / (Var ξ) ^ (3 / 2)

-- Kurtosis parameter
def kur (ξ : ℝ) : ℝ := Ex ((ξ - Ex ξ) ^ 4) / (Var ξ) ^ 2

-- Sum of i.i.d. random variables
def Sn (n : ℕ) (ξ : ℕ → ℝ) : ℝ := (Finset.range n).sum ξ

theorem skw_Sn (h_iid : ∀ i j, i ≠ j → ξ i ∼ ξ j [Distribution.identically_independently_distributed]) :
  skw (Sn n ξ) = skw (ξ 0) / sqrt n :=
sorry

theorem kur_Sn (h_iid : ∀ i j, i ≠ j → ξ i ∼ ξ j [Distribution.identically_independently_distributed]) :
  kur (Sn n ξ) = 3 + (kur (ξ 0) - 3) / n :=
sorry

end skw_Sn_kur_Sn_l137_137256


namespace f_eq_2x_pow_5_l137_137653

def f (x : ℝ) : ℝ := (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1

theorem f_eq_2x_pow_5 (x : ℝ) : f x = (2*x)^5 :=
by
  sorry

end f_eq_2x_pow_5_l137_137653


namespace less_than_edges_bound_l137_137939

theorem less_than_edges_bound 
  (n : ℕ) (h_n : n ≥ 5) 
  (G : Type) [simple_graph G] 
  (V : G → Prop) 
  (E : G → G → Prop) 
  (edge_color : E -> Prop) -- (true for red, false for blue)
  (no_monochrome_cycle : ∀ (c : Prop), ∀ (h : G), (G.cycle_with_color h c) → ¬ c) -- 
  : G.number_of_edges < ⌊n^2/3⌋ :=
sorry

end less_than_edges_bound_l137_137939


namespace tree_count_l137_137681

theorem tree_count (A_tree_good : ℕ) (B_tree_good : ℕ) (total_good : ℕ) (percent_A_B : ℕ → Prop) (good_oranges_A : ℕ → Prop) (good_oranges_B : ℕ → Prop) : 
  (percent_A_B(50) ∧ percent_A_B(50)) ∧ (good_oranges_A(6)) ∧ (good_oranges_B(5)) ∧ (total_good(55)) 
  → ∃ T, T = 10 :=
by {
  sorry
}

-- Definitions
def percent_A_B (P : ℕ) : ℕ → Prop 
| 50 := true
| _  := false
    
def good_oranges_A : ℕ := 
  6
     
def good_oranges_B : ℕ := 
  5

def total_good : ℕ := 
  55

end tree_count_l137_137681


namespace math_proof_l137_137626

variable {a b c A B C : ℝ}
variable {S : ℝ}

noncomputable def problem_statement (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) : Prop :=
    (∃ A B : ℝ, (A = 2 * B) ∧ (A = 90)) 

theorem math_proof (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) :
    problem_statement h1 h2 :=
    sorry

end math_proof_l137_137626


namespace x_intercept_of_perpendicular_line_l137_137065

/-- The intercept on the x-axis of the line that is perpendicular to the line
    3x - 4y - 7 = 0 and forms a triangle with both coordinate axes having an area of 6
    is 3 or -3. -/
theorem x_intercept_of_perpendicular_line (m : ℝ) :
  let line := (4 * @Eq._match_1 ℝ 0 m 0) + (3 * @Eq._match_1 ℝ 0 0 ⟨m⟩) + m = 0 in
  (3 * (-4) * 3 = 0) → (∃ (m : ℝ), 2 / 3 * m ^ 2 = 144) →
  (@mul_assoc ℝ 12 (1 / 2) (1 / 12)) →
  x_intercept_of_line 4 3 m = 3 ∨ x_intercept_of_line 4 3 m = -3 :=
sorry

end x_intercept_of_perpendicular_line_l137_137065


namespace product_of_primes_l137_137468

theorem product_of_primes :
  let p1 := 11
  let p2 := 13
  let p3 := 997
  p1 * p2 * p3 = 142571 :=
by
  sorry

end product_of_primes_l137_137468


namespace photos_last_weekend_45_l137_137266

theorem photos_last_weekend_45 (photos_animals photos_flowers photos_scenery total_photos_this_weekend photos_last_weekend : ℕ)
  (h1 : photos_animals = 10)
  (h2 : photos_flowers = 3 * photos_animals)
  (h3 : photos_scenery = photos_flowers - 10)
  (h4 : total_photos_this_weekend = photos_animals + photos_flowers + photos_scenery)
  (h5 : photos_last_weekend = total_photos_this_weekend - 15) :
  photos_last_weekend = 45 :=
sorry

end photos_last_weekend_45_l137_137266


namespace odd_periodic_pi_l137_137791

def fA (x : ℝ) : ℝ := Real.sin x ^ 2
def fB (x : ℝ) : ℝ := Real.tan (2 * x)
def fC (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)
def fD (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem odd_periodic_pi : (∃ f : ℝ → ℝ, f = fD ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x))
  ∧ ¬ (∃ f : ℝ → ℝ, f = fA ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x))
  ∧ ¬ (∃ f : ℝ → ℝ, f = fB ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x))
  ∧ ¬ (∃ f : ℝ → ℝ, f = fC ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x)) :=
by
  -- We will skip the proof steps, only state the theorem
  sorry

end odd_periodic_pi_l137_137791


namespace balls_in_consecutive_bins_l137_137376

theorem balls_in_consecutive_bins (p q : ℕ) (h : p + q = 703) :
  ∃ p q : ℕ, p + q = 703 ∧ (p.gcd q = 1 ∧ (∀ (a b c : ℕ), 
  probability_of_bins_is_consecutive a b c = 3^(-3*a-3+b+c-a)) ∧
  (probability_all_consecutive = ∑' n, 3^(-3*n - 3) = 1 / 702) :=
begin
  sorry
end

end balls_in_consecutive_bins_l137_137376


namespace worker_weekly_pay_l137_137748

variable (regular_rate : ℕ) -- Regular rate of Rs. 10 per survey
variable (total_surveys : ℕ) -- Worker completes 100 surveys per week
variable (cellphone_surveys : ℕ) -- 60 surveys involve the use of cellphone
variable (increased_rate : ℕ) -- Increased rate 30% higher than regular rate

-- Defining given values
def reg_rate : ℕ := 10
def total_survey_count : ℕ := 100
def cellphone_survey_count : ℕ := 60
def inc_rate : ℕ := reg_rate + 3

-- Calculating payments
def regular_survey_count : ℕ := total_survey_count - cellphone_survey_count
def regular_pay : ℕ := regular_survey_count * reg_rate
def cellphone_pay : ℕ := cellphone_survey_count * inc_rate

-- Total pay calculation
def total_pay : ℕ := regular_pay + cellphone_pay

-- Theorem to be proved
theorem worker_weekly_pay : total_pay = 1180 := 
by
  -- instantiate variables
  let regular_rate := reg_rate
  let total_surveys := total_survey_count
  let cellphone_surveys := cellphone_survey_count
  let increased_rate := inc_rate
  
  -- skip proof
  sorry

end worker_weekly_pay_l137_137748


namespace complement_of_A_in_U_l137_137577

namespace SetTheory

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by
  sorry

end SetTheory

end complement_of_A_in_U_l137_137577


namespace cubic_root_identity_l137_137674

theorem cubic_root_identity (x1 x2 x3 : ℝ) (h1 : x1^3 - 3*x1 - 1 = 0) (h2 : x2^3 - 3*x2 - 1 = 0) (h3 : x3^3 - 3*x3 - 1 = 0) (h4 : x1 < x2) (h5 : x2 < x3) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end cubic_root_identity_l137_137674


namespace sum_of_areas_of_rectangles_l137_137052

theorem sum_of_areas_of_rectangles :
  let widths := List.repeat 3 8
  let lengths := List.map (fun n => n^2) [1, 2, 3, 4, 5, 6, 7, 8]
  let areas := List.zipWith (*) widths lengths
  List.sum areas = 612 := by
  sorry

end sum_of_areas_of_rectangles_l137_137052


namespace negation_of_statement_l137_137352

theorem negation_of_statement :
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n > x^2) ↔ (∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2) := by
sorry

end negation_of_statement_l137_137352


namespace min_value_of_n_for_even_function_l137_137819

noncomputable def determinant (a₁ a₂ a₃ a₄ : ℝ) : ℝ :=
  a₁ * a₄ - a₂ * a₃

def f (x : ℝ) : ℝ :=
  determinant (sqrt 3) (sin x) 1 (cos x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem min_value_of_n_for_even_function :
  ∃ n : ℝ, n = 5 * π / 6 ∧ is_even_function (λ x => f (x - n)) :=
sorry

end min_value_of_n_for_even_function_l137_137819


namespace find_coords_c_cosine_theta_l137_137098

-- Define the three vectors in the plane
variables {a b c : ℝ × ℝ}
-- Given that vector a is (1, -2)
def vec_a : ℝ × ℝ := (1, -2)
-- Given that |c| = 2 * sqrt 5 and c is parallel to a
axiom magnitude_c : real.sqrt (c.1 ^ 2 + c.2 ^ 2) = 2 * real.sqrt 5
axiom c_parallel_a : ∃ k : ℝ, c = (k * vec_a.1, k * vec_a.2)

-- Given that |b| = 1
axiom magnitude_b : real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 1
-- Given that (a + b) is perpendicular to (a - 2b)
axiom perp_condition : (vec_a.1 + b.1) * (vec_a.1 - 2 * b.1) + (vec_a.2 + b.2) * (vec_a.2 - 2 * b.2) = 0

-- Prove the coordinates of vector c
theorem find_coords_c : (c = (-2, 4)) ∨ (c = (2, -4)) :=
sorry

-- Prove the cosine of the angle between a and b is (3 * sqrt 5) / 5
theorem cosine_theta : real.cos (real.angle vec_a b) = (3 * real.sqrt 5) / 5 :=
sorry

end find_coords_c_cosine_theta_l137_137098


namespace intersection_with_y_axis_l137_137332

theorem intersection_with_y_axis :
  let parabola := λ x : ℝ, x^2 + 2 * x - 3 in
  parabola 0 = -3 :=
by
  let parabola := λ x : ℝ, x^2 + 2 * x - 3
  show parabola 0 = -3
  sorry

end intersection_with_y_axis_l137_137332


namespace angle_between_vectors_l137_137141

variable (a b : ℝ^3)
variable (θ : ℝ)

def magnitude (v : ℝ^3) : ℝ := (v.1^2 + v.2^2 + v.3^2).sqrt
def dot_product (u v : ℝ^3) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem angle_between_vectors (h1 : magnitude a = 1) (h2 : magnitude b = 6)
(h3 : dot_product a b = 3) : θ = Real.pi / 3 := 
sorry

end angle_between_vectors_l137_137141


namespace f_equals_n_l137_137655

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined (n : ℕ) (h₁ : n > 0) : ∃ z : ℤ, f n = z
axiom f_integer (n : ℕ) (h₁ : n > 0) : f n ∈ ℤ
axiom f_2 : f 2 = 2
axiom f_mn (m n : ℕ) (h₁ : m > 0) (h₂ : n > 0) : f (m * n) ≤ f m * f n
axiom f_m_gt_f_n (m n : ℕ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m > n) : f m > f n

theorem f_equals_n (n : ℕ) (h₁ : n > 0) : f n = n := sorry

end f_equals_n_l137_137655


namespace probability_of_X_le_1_l137_137075

noncomputable def C (n k : ℕ) : ℚ := Nat.choose n k

noncomputable def P_X_le_1 := 
  (C 4 3 / C 6 3) + (C 4 2 * C 2 1 / C 6 3)

theorem probability_of_X_le_1 : P_X_le_1 = 4 / 5 := by
  sorry

end probability_of_X_le_1_l137_137075


namespace verify_propositions_l137_137554

-- Definitions for distinct lines and planes that are parallel or perpendicular
variables {Line Plane : Type} 
variables {l m n : Line} {α β γ : Plane}

-- Distinctness of lines and planes
axiom distinct_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n
axiom distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositional predicates for parallel and perpendicular relationships
class Parallel (a b : Type) := (parallel : a → b → Prop)
class Perpendicular (a b : Type) := (perpendicular : a → b → Prop)
class Subset (a b : Type) := (subset : a → b → Prop)

-- Aliases to use the defined predicates for Lines and Planes
variables [Parallel Line Plane] [Perpendicular Plane Plane] [Subset Line Plane]
variables (P : Parallel Line Plane) (Q : Perpendicular Plane Plane) (S : Subset Line Plane)

-- Given conditions for propositions 1 and 3
axiom prop1_cond1 : P.parallel α β
axiom prop1_cond2 : S.subset l α

axiom prop3_cond1 : ¬ S.subset m α
axiom prop3_cond2 : S.subset n α
axiom prop3_cond3 : P.parallel m n

-- Lean statement to verify the propositions 1 and 3
theorem verify_propositions : 
  P.parallel l β ∧ P.parallel m α :=
sorry  -- Proof to be added

end verify_propositions_l137_137554


namespace quadratic_function_three_intersections_l137_137569

theorem quadratic_function_three_intersections (k : ℝ) :
  (∃ x y : ℝ, y = k * x^2 - 4 * x - 3 ∧ (y = 0 ∨ x = 0)) ∧
  ∃ a b : ℝ, (a * k + b ≠ 0 ∧ a ≠ 0) ∧ (b * k - a ≠ 0) ∧
  (let Δ := (-4)^2 - 4 * k * (-3) in Δ > 0) ↔
  k > -4/3 ∧ k ≠ 0 :=
by {
  sorry
}

end quadratic_function_three_intersections_l137_137569


namespace probability_one_white_one_black_l137_137415

def white_ball_count : ℕ := 8
def black_ball_count : ℕ := 7
def total_ball_count : ℕ := white_ball_count + black_ball_count
def total_ways_to_choose_2_balls : ℕ := total_ball_count.choose 2
def favorable_ways : ℕ := white_ball_count * black_ball_count

theorem probability_one_white_one_black : 
  (favorable_ways : ℚ) / (total_ways_to_choose_2_balls : ℚ) = 8 / 15 :=
by
  sorry

end probability_one_white_one_black_l137_137415


namespace proof_lambda_mu_ratio_l137_137246

variables {A B C : ℝ × ℝ}
variables {λ μ ω : ℝ}

-- Defining the circle O (the origin of coordinates) and points
def is_on_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1

-- The angle condition
def angle_AOB (A B : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), θ = 2 * π / 3 ∧ dist (0, 0) A = 1 ∧ dist (0, 0) B = 1 ∧ 
  ∃ (O : ℝ × ℝ), O = (0,0) ∧ is_on_circle A ∧ is_on_circle B ∧ 
  ∃ (angle : ℝ), angle = real.arccos ((A.1 * B.1 + A.2 * B.2) / (real.sqrt (A.1^2 + A.2^2) * real.sqrt (B.1^2 + B.2^2))) 

-- The linear combination condition
def linear_combination (C A B : ℝ × ℝ) (λ μ : ℝ) : Prop :=
  C = (λ * A.1 + μ * B.1, λ * A.2 + μ * B.2)

-- The ω condition
def omega_condition (λ μ : ℝ) (ω : ℝ) : Prop :=
  ω = sqrt 3 * λ + λ + μ

-- Main statement to prove
theorem proof_lambda_mu_ratio 
  (hA : is_on_circle A) (hB : is_on_circle B) 
  (h_angle : angle_AOB A B)
  (h_linear_comb : linear_combination C A B λ μ)
  (h_omega_max : ∀ x y : ℝ, omega_condition x y (sqrt 3 * x + x + y) ≤ ω) :
  λ / μ = (sqrt 3 + 1) / 2 :=
sorry

end proof_lambda_mu_ratio_l137_137246


namespace equal_sets_l137_137409

theorem equal_sets : ∀ x t : ℝ, (x ≤ 1 ↔ t ≤ 1) → ({x : ℝ | x ≤ 1} = {t : ℝ | t ≤ 1}) := 
by 
  intros x t h
  simp [h]
  sorry

end equal_sets_l137_137409


namespace parallelogram_perimeter_area_sum_l137_137481

theorem parallelogram_perimeter_area_sum 
  (A B C D : Point)
  (hA : A = (1, 3))
  (hB : B = (6, 3))
  (hC : C = (4, 0))
  (hD : D = (-1, 0)) :
  let sideLength (p1 p2 : Point) : ℝ :=
    EuclideanGeometry.dist p1 p2 in
  let p := 2 * (sideLength A B + sideLength A D) in
  let base := sideLength A D in
  let height := 3 in
  let a := base * height in
  p + a = 25 + 2 * Real.sqrt 13 :=
by
  sorry

end parallelogram_perimeter_area_sum_l137_137481


namespace greendale_high_school_points_l137_137293

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l137_137293


namespace problem_statement_l137_137850

theorem problem_statement (x y : ℝ) 
  (hA : A = (x + y) * (y - 3 * x))
  (hB : B = (x - y)^4 / (x - y)^2)
  (hCond : 2 * y + A = B - 6) :
  y = 2 * x^2 - 3 ∧ (y + 3)^2 - 2 * x * (x * y - 3) - 6 * x * (x + 1) = 0 :=
by
  sorry

end problem_statement_l137_137850


namespace balls_in_boxes_distinguished_boxes_l137_137136

theorem balls_in_boxes_distinguished_boxes :
  ∃ (n : ℕ), n = nat.choose 9 2 ∧ n = 36 :=
by {
  -- just write the statement and use 'sorry' to skip proof
  use 36,
  split,
  { unfold nat.choose,
    -- calculation details omitted
    sorry, },
  { refl, },
}

end balls_in_boxes_distinguished_boxes_l137_137136


namespace john_total_time_l137_137965

noncomputable def total_time_spent : ℝ :=
  let landscape_pictures := 10
  let landscape_drawing_time := 2
  let landscape_coloring_time := landscape_drawing_time * 0.7
  let landscape_enhancing_time := 0.75
  let total_landscape_time := (landscape_drawing_time + landscape_coloring_time + landscape_enhancing_time) * landscape_pictures
  
  let portrait_pictures := 15
  let portrait_drawing_time := 3
  let portrait_coloring_time := portrait_drawing_time * 0.75
  let portrait_enhancing_time := 1.0
  let total_portrait_time := (portrait_drawing_time + portrait_coloring_time + portrait_enhancing_time) * portrait_pictures
  
  let abstract_pictures := 20
  let abstract_drawing_time := 1.5
  let abstract_coloring_time := abstract_drawing_time * 0.6
  let abstract_enhancing_time := 0.5
  let total_abstract_time := (abstract_drawing_time + abstract_coloring_time + abstract_enhancing_time) * abstract_pictures
  
  total_landscape_time + total_portrait_time + total_abstract_time

theorem john_total_time : total_time_spent = 193.25 :=
by sorry

end john_total_time_l137_137965


namespace series_sum_result_final_m_plus_n_l137_137362

-- Define the series summand
def series_summand (n : ℕ) : ℚ :=
  1 / (n * (n + 1) * (n + 2))

-- Define the series sum
def series_sum : ℚ :=
  (Finset.range 13).sum (λ k, series_summand (k + 2))

-- Define the final result
def final_result : ℚ :=
  13 / 160

-- The proof problem statement
theorem series_sum_result : series_sum = final_result :=
sorry

-- The statement we're proving
theorem final_m_plus_n : 13 + 160 = 173 :=
by norm_num

end series_sum_result_final_m_plus_n_l137_137362


namespace area_of_parallelogram_l137_137323

variable (b : ℕ)
variable (h : ℕ)
variable (A : ℕ)

-- Condition: The height is twice the base.
def height_twice_base := h = 2 * b

-- Condition: The base is 9.
def base_is_9 := b = 9

-- Condition: The area of the parallelogram is base times height.
def area_formula := A = b * h

-- Question: Prove that the area of the parallelogram is 162.
theorem area_of_parallelogram 
  (h_twice : height_twice_base h b) 
  (b_val : base_is_9 b) 
  (area_form : area_formula A b h): A = 162 := 
sorry

end area_of_parallelogram_l137_137323


namespace comparison_of_a_b_c_l137_137550

variable (x : ℝ)
variable (h₀ : 0 < x)
variable (h₁ : x < π / 2)

def a := Real.sin x
def b := Real.exp (Real.sin x)
def c := Real.log (Real.sin x)

theorem comparison_of_a_b_c : c < a ∧ a < b :=
by
  have : 0 < Real.sin x := Real.sin_pos_of_pos_of_lt_pi h₀ h₁
  have : Real.sin x < 1 := Real.sin_lt_one_of_pos_of_lt_pi h₀ h₁
  sorry

end comparison_of_a_b_c_l137_137550


namespace num_ways_to_designated_face_l137_137011

-- Define the structure of the dodecahedron
inductive Face
| Top
| Bottom
| TopRing (n : ℕ)   -- n ranges from 1 to 5
| BottomRing (n : ℕ)  -- n ranges from 1 to 5
deriving Repr, DecidableEq

-- Define adjacency relations on Faces (simplified)
def adjacent : Face → Face → Prop
| Face.Top, Face.TopRing n          => true
| Face.TopRing n, Face.TopRing m    => (m = (n % 5) + 1) ∨ (m = ((n + 3) % 5) + 1)
| Face.TopRing n, Face.BottomRing m => true
| Face.BottomRing n, Face.BottomRing m => true
| _, _ => false

-- Predicate for specific face on the bottom ring
def designated_bottom_face (f : Face) : Prop :=
  match f with
  | Face.BottomRing 1 => true
  | _ => false

-- Define the number of ways to move from top to the designated bottom face
noncomputable def num_ways : ℕ :=
  5 + 10

-- Lean statement that represents our equivalent proof problem
theorem num_ways_to_designated_face :
  num_ways = 15 := by
  sorry

end num_ways_to_designated_face_l137_137011


namespace two_pies_differ_l137_137182

theorem two_pies_differ (F_A F_C B_A B_C : Bool) :
  (F_A ∨ F_C ∨ B_A ∨ B_C) →
  (F_A ∧ F_C ∧ B_A ∧ B_C) ∧ 
  (∀ a b, (a ≠ b) → (a.filling ≠ b.filling ∧ a.preparation ≠ b.preparation)) :=
by
  intros H1 H2
  sorry

end two_pies_differ_l137_137182


namespace max_value_of_trig_expression_l137_137500

theorem max_value_of_trig_expression (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_value_of_trig_expression_l137_137500


namespace compound_interest_rate_l137_137062

theorem compound_interest_rate : 
  let P := 14800
  let interest := 4265.73
  let A := 19065.73
  let t := 2
  let n := 1
  let r := 0.13514
  (P : ℝ) * (1 + r)^t = A :=
by
-- Here we will provide the steps of the proof
sorry

end compound_interest_rate_l137_137062


namespace trapezoid_perimeter_l137_137438

noncomputable def semiCircularTrapezoidPerimeter (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2) : ℝ :=
-((x^2) / 8) + 2 * x + 32

theorem trapezoid_perimeter 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2)
  (r : ℝ) 
  (h_r : r = 8) 
  (AB : ℝ) 
  (h_AB : AB = 2 * r)
  (CD_on_circumference : true) :
  semiCircularTrapezoidPerimeter x hx = -((x^2) / 8) + 2 * x + 32 :=   
sorry

end trapezoid_perimeter_l137_137438


namespace intersection_points_eq_2_l137_137820

def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
def eq2 (x y : ℝ) : Prop := (x + 2 * y - 3) * (3 * x - 4 * y + 6) = 0

theorem intersection_points_eq_2 : ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 2 := 
sorry

end intersection_points_eq_2_l137_137820


namespace parabola_triangle_areas_l137_137560

-- Define necessary points and expressions
variables (x1 y1 x2 y2 x3 y3 : ℝ)
variables (m n : ℝ)
def parabola_eq (x y : ℝ) := y ^ 2 = 4 * x
def median_line (m n x y : ℝ) := m * x + n * y - m = 0
def areas_sum_sq (S1 S2 S3 : ℝ) := S1 ^ 2 + S2 ^ 2 + S3 ^ 2 = 3

-- Main statement
theorem parabola_triangle_areas :
  (parabola_eq x1 y1 ∧ parabola_eq x2 y2 ∧ parabola_eq x3 y3) →
  (m ≠ 0) →
  (median_line m n 1 0) →
  (x1 + x2 + x3 = 3) →
  ∃ S1 S2 S3 : ℝ, areas_sum_sq S1 S2 S3 :=
by sorry

end parabola_triangle_areas_l137_137560


namespace intersection_of_A_and_B_l137_137575

-- Given sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Prove the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := 
by
  sorry

end intersection_of_A_and_B_l137_137575


namespace incident_ray_tangential_to_circle_l137_137722

/-
  Problem statement:
  Given two semi-infinite plane mirrors inclined at a non-zero angle with inner surfaces reflective,
  prove that for any given incident ray (lying on a normal plane to the mirrors), all lines of incident/reflected rays are tangential to a circle centered at the intersection point of the mirrors, 
  with radius determined by the distance and angle of incidence.
-/

def intersection_of_mirrors : Type := sorry   -- Point B, intersection of mirrors

def is_incident_ray (ray : intersection_of_mirrors → Prop) : Prop := sorry
def angle_of_incidence (ray : intersection_of_mirrors → ℝ) : ℝ := sorry
def distance_to_intersection (point : intersection_of_mirrors) : ℝ := sorry
def tangential_to_circle (ray : intersection_of_mirrors → Prop) (center : intersection_of_mirrors) (radius : ℝ) : Prop := sorry

theorem incident_ray_tangential_to_circle
  (ray : intersection_of_mirrors → Prop)
  (B : intersection_of_mirrors)
  (θ : ℝ)
  (l : ℝ)
  (h1 : is_incident_ray ray)
  (h2 : θ = angle_of_incidence ray B)
  (h3 : l = distance_to_intersection B):
  tangential_to_circle ray B (l * Real.cos θ) :=
sorry

end incident_ray_tangential_to_circle_l137_137722


namespace nth_equation_l137_137275

theorem nth_equation (n : ℕ) (hn : n > 0) : 9 * n + (n - 1) = 10 * n - 1 :=
sorry

end nth_equation_l137_137275


namespace find_x_l137_137060

theorem find_x (x : ℝ) (h : 4 * log 3 x = log 3 (5 * x)) : x = real.cbrt 5 :=
by
  sorry

end find_x_l137_137060


namespace pies_differ_in_both_l137_137164

-- Definitions of pie types
inductive Filling
| apple
| cherry

inductive PreparationMethod
| fried
| baked

structure Pie where
  filling : Filling
  method : PreparationMethod

-- The set of all possible pie types
def pies : Set Pie := {
  {filling := Filling.apple, method := PreparationMethod.fried},
  {filling := Filling.cherry, method := PreparationMethod.fried},
  {filling := Filling.apple, method := PreparationMethod.baked},
  {filling := Filling.cherry, method := PreparationMethod.baked}
}

-- The statement to prove: If there are at least three types of pies available, then there exist two pies that differ in both filling and preparation method.
theorem pies_differ_in_both :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.method ≠ p2.method :=
begin
  sorry
end

end pies_differ_in_both_l137_137164


namespace obtuse_triangle_covering_l137_137866

noncomputable def exists_isosceles_right_triangle : Prop :=
  ∀ (ABC : Triangle), ABC.is_obtuse ∧ ABC.circumradius = 1 →
  ∃ (T : Triangle), T.is_isosceles_right ∧ T.hypotenuse_length = sqrt 2 + 1 ∧ T.covers ABC

theorem obtuse_triangle_covering (ABC : Triangle) (h1 : ABC.is_obtuse) (h2 : ABC.circumradius = 1) :
  ∃ (T : Triangle), T.is_isosceles_right ∧ T.hypotenuse_length = sqrt 2 + 1 ∧ T.covers ABC :=
sorry

end obtuse_triangle_covering_l137_137866


namespace line_distance_condition_l137_137335

theorem line_distance_condition (c : ℝ) :
  (∃ c1 c2 : ℝ, (c1 = -11 ∧ c2 = 9) ∧
   (3 * x - 4 * y + c1 = 0 ∨ 3 * x - 4 * y + c2 = 0) ∧
   (abs (c + 1) / sqrt (3 ^ 2 + (-4) ^ 2) = 2)) :=
begin
  sorry
end

end line_distance_condition_l137_137335


namespace find_a3_plus_a5_l137_137540

-- Define an arithmetic-geometric sequence
def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, 0 < r ∧ ∃ b : ℝ, a n = b * r ^ n

-- Define the given condition
def given_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

-- Define the target theorem statement
theorem find_a3_plus_a5 (a : ℕ → ℝ) 
  (pos_sequence : is_arithmetic_geometric a) 
  (cond : given_condition a) : 
  a 3 + a 5 = 5 :=
sorry

end find_a3_plus_a5_l137_137540


namespace sqrt_4_plus_2_sqrt_3_sqrt_5_minus_2_sqrt_6_sqrt_m_plus_minus_2_sqrt_n_sqrt_4_minus_sqrt_15_l137_137447

-- Part 1
theorem sqrt_4_plus_2_sqrt_3 : sqrt (4 + 2 * sqrt 3) = sqrt 3 + 1 :=
by sorry

theorem sqrt_5_minus_2_sqrt_6 : sqrt (5 - 2 * sqrt 6) = sqrt 3 - sqrt 2 :=
by sorry

-- Part 2
theorem sqrt_m_plus_minus_2_sqrt_n {m n a b : ℝ} (h_pos_m : 0 < m) (h_pos_n : 0 < n)
    (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : a > b) (h_add : a + b = m) (h_mul : a * b = n) : 
    sqrt (m + 2 * sqrt n) = sqrt a + sqrt b ∧ sqrt (m - 2 * sqrt n) = sqrt a - sqrt b :=
by sorry

-- Part 3
theorem sqrt_4_minus_sqrt_15 : sqrt (4 - sqrt 15) = (sqrt 10 / 2) - (sqrt 6 / 2) :=
by sorry

end sqrt_4_plus_2_sqrt_3_sqrt_5_minus_2_sqrt_6_sqrt_m_plus_minus_2_sqrt_n_sqrt_4_minus_sqrt_15_l137_137447


namespace find_b_l137_137879

variables (U : Set ℝ) (A : Set ℝ) (b : ℝ)

theorem find_b (hU : U = Set.univ)
               (hA : A = {x | 1 ≤ x ∧ x < b})
               (hComplA : U \ A = {x | x < 1 ∨ x ≥ 2}) :
  b = 2 :=
sorry

end find_b_l137_137879


namespace equidistant_points_l137_137420

theorem equidistant_points (O : Point) (r d : ℝ) (h : r > 0):
  let T₁ := {p : Point | dist(O, p) = d}
  let T₂ := {p : Point | dist(O, p) = d + r}
  let T₃ := {p : Point | dist(O, p) = d + 2r}
  let C := {p : Point | dist(O, p) = r}
  T₁ ∩ T₂ ∩ T₃ ∩ C = {p : Point | dist(O, p) = r} ∧ points_in_intersection = 2 :=
sorry

end equidistant_points_l137_137420


namespace difference_of_triangular_23_and_21_l137_137794

def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem difference_of_triangular_23_and_21 : triangular 23 - triangular 21 = 45 :=
sorry

end difference_of_triangular_23_and_21_l137_137794


namespace problem_1_problem_2a_problem_2b_l137_137581

open Real
open ComplexConjugate

variables (A B C : ℝ)
variables (m n : ℝ × ℝ)
variables (AB BC : ℝ)

noncomputable def vec_m : ℝ × ℝ := (2 * sqrt 3, 1)
noncomputable def vec_n (A : ℝ) : ℝ × ℝ := (cos (A / 2) ^ 2, sin A)

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

theorem problem_1 : magnitude (vec_n (π / 2)) = sqrt 5 / 2 := sorry

theorem problem_2a (m n : ℝ × ℝ) (A B C : ℝ) (hC : C = 2 * π / 3) (hAB : AB = 3) :
  vec_m ∙ vec_n (π / 6) = max (vec_m ∙ vec_n A) := sorry

theorem problem_2b (A : ℝ) (hA : A = π / 6) (hC : C = 2 * π / 3) (hAB : AB = 3) :
  BC = sqrt 3 :=
  by sorry

end problem_1_problem_2a_problem_2b_l137_137581


namespace magnitude_of_w_l137_137635

noncomputable def z : ℂ := ((5 - 3 * complex.I) ^ 2 * (7 + 11 * complex.I) ^ 3) / (2 - 5 * complex.I)
noncomputable def w : ℂ := conj z / z

theorem magnitude_of_w : complex.abs w = 1 := by
  sorry

end magnitude_of_w_l137_137635


namespace least_number_subtracted_l137_137755

theorem least_number_subtracted (n m k : ℕ) (h1 : n = 3830) (h2 : k = 15) (h3 : n % k = m) (h4 : m = 5) : 
  (n - m) % k = 0 :=
by
  sorry

end least_number_subtracted_l137_137755


namespace demand_decrease_is_correct_l137_137001

variable (P Q : ℝ) -- original price and demand
variable (demand_decrease_proportion : ℝ) -- the decrease in demand we want to prove

def new_price := 1.2 * P
def new_income := 1.1 * (P * Q)
def new_demand := (1.1 * (P * Q)) / (1.2 * P)
def demand_decrease := Q - new_demand

theorem demand_decrease_is_correct :
  demand_decrease_proportion = 1 / 12 →
  demand_decrease_proportion = demand_decrease / Q :=
by
  intros h1
  rw [h1, demand_decrease, new_demand]
  field_simp [new_price, new_income]
  sorry

end demand_decrease_is_correct_l137_137001


namespace polynomial_equation_solution_l137_137828

open Polynomial

theorem polynomial_equation_solution (P : ℝ[X])
(h : ∀ (a b c : ℝ), P.eval (a + b - 2 * c) + P.eval (b + c - 2 * a) + P.eval (c + a - 2 * b) = 
      3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)) : 
∃ (a b : ℝ), P = Polynomial.C a * X^2 + Polynomial.C b * X := 
sorry

end polynomial_equation_solution_l137_137828


namespace find_x_l137_137609

-- Definitions based on problem conditions
def PQ := 6 -- PQ = 6 cm
def PR := 8 -- PR = 8 cm
def QR := Real.sqrt (PQ^2 + PR^2) -- QR is the hypotenuse of the right triangle PQR

-- Definition of the semi-perimeter
def s := (PQ + PR + QR) / 2

-- Definition of the area of the triangle
def area := (PQ * PR) / 2

-- Definition of the radius of the inscribed circle
def r := area / s

-- The length x (what we're asked to find and prove it's equal to 6 cm)
def x := QR - 2 * r

-- The theorem statement we're asked to prove
theorem find_x : x = 6 := by
  sorry

end find_x_l137_137609


namespace count_valid_n_l137_137583

-- Definition of equally spaced on the unit circle
def equally_spaced_on_unit_circle (n : ℕ) (z : ℕ → ℂ) : Prop :=
  ∀ i j : ℕ, i < n ∧ j < n → z i = exp (2 * π * I * i / n)

-- Proving the number of valid n's
theorem count_valid_n :
  let valid_n (n : ℕ) :=
    (n ≥ 1) ∧
    ∀ (z : ℕ → ℂ),
      (∀ i, i < n → complex.abs (z i) = 1) ∧
      (complex.sum (fin n) z = 0) ∧
      (∀ i, ∃ k : ℤ, z i = complex.exp (k * (π / 4) * complex.I)) →
      equally_spaced_on_unit_circle n z
  in
  (finset.card (finset.filter valid_n (finset.range 9)) = 3) :=
sorry

end count_valid_n_l137_137583


namespace part_1_part_2_part_3_l137_137544

noncomputable def a (n : ℕ) : ℕ → ℕ 
| 1 := 5
| 2 := 13
| 3 := 33
| k := if k ≥ 4 then 2 * a (k-1) + 2 ^ k - 1 else 5

def b (n : ℕ) (λ : Int) := (a n + λ) / 2 ^ n

def c (n : ℕ) (λ : Int) := (1 / b n λ) ^ 2

def T (n : ℕ) (λ : Int) := ∑ i in (range n), c i λ

theorem part_1 : a 1 = 5 ∧ a 2 = 13 ∧ a 3 = 33 := by
  sorry

theorem part_2 : ∃ λ : Int, ∀ n ≥ 2, b n λ - b (n - 1) λ = 1 := by
  use -1
  sorry

theorem part_3 : ∀ n : ℕ, T n (-1) < 3 / 4 := by
  sorry

end part_1_part_2_part_3_l137_137544


namespace cartesian_eq_C2_polar_eq_C1_min_distance_P_to_C1_l137_137708

noncomputable def parametric_C1 (t : ℝ) : ℝ × ℝ := 
  (1 + t * Real.cos (π / 4), 5 + t * Real.sin (π / 4))

noncomputable def parametric_C2 (φ : ℝ) : ℝ × ℝ := 
  (Real.cos φ, Real.sqrt 3 * Real.sin φ)

theorem cartesian_eq_C2 :
  ∀ (x y : ℝ), (∃ φ : ℝ, x = Real.cos φ ∧ y = Real.sqrt 3 * Real.sin φ) ↔ x^2 + y^2 / 3 = 1 :=
sorry

theorem polar_eq_C1 : 
  ∀ (θ ρ : ℝ), (∃ t : ℝ, (1 + t * Real.cos (π / 4) = ρ * Real.cos θ) ∧ (5 + t * Real.sin (π / 4) = ρ * Real.sin θ)) 
                ↔ ρ * (Real.cos θ - Real.sin θ) + 4 = 0 :=
sorry

theorem min_distance_P_to_C1 :
  ∀ (φ : ℝ), 
  let P := parametric_C2 φ in 
  let dist := | - P.1 + P.2 - 4 | / Real.sqrt 2 in
  (∃ φ : ℝ, φ = 2 * π / 3) → dist = sqrt 2 :=
sorry

end cartesian_eq_C2_polar_eq_C1_min_distance_P_to_C1_l137_137708


namespace find_initial_speeds_l137_137462
noncomputable theory

-- Definitions of the problem
def side_length : ℝ := 108 -- cm
def speed_ratio : ℝ := 4 / 5

-- Initial time calculation
def first_meeting_time : ℝ :=
  let total_distance := 3 * side_length
  in total_distance / (4 + 5 * (4 / 5))

-- Speed definitions
def initial_speed_Alpha : ℝ := (4 / 5) * initial_speed_Beta
def initial_speed_Beta : ℝ := initial_speed_Alpha * (5 / 4)

-- Distances covered
def distance_covered_Alpha : ℝ := initial_speed_Alpha * first_meeting_time
def distance_covered_Beta : ℝ := initial_speed_Beta * first_meeting_time

-- The target proof statement: finding the initial speeds
theorem find_initial_speeds : 
  initial_speed_Alpha = 12.5 ∧ 
  initial_speed_Beta = 15.625 :=
by
  -- We need to prove the calculations here.
  sorry

end find_initial_speeds_l137_137462


namespace find_f_of_4_l137_137929

-- Define the inverse function and the condition x > 0
def inv_function (x : ℝ) : ℝ := x^2
axiom positive_domain (x : ℝ) : x > 0

-- Problem statement
theorem find_f_of_4 : ∀ (x : ℝ), positive_domain x → inv_function x = 16 → f 4 = 2 :=
by
  sorry

end find_f_of_4_l137_137929


namespace probability_not_equal_genders_l137_137270

noncomputable def probability_more_grandsons_or_more_granddaughters : ℚ :=
  let total_ways := 2 ^ 12
  let equal_distribution_ways := (Nat.choose 12 6)
  let probability_equal := (equal_distribution_ways : ℚ) / (total_ways : ℚ)
  1 - probability_equal

theorem probability_not_equal_genders (n : ℕ) (p : ℚ) (hp : p = 1 / 2) (hn : n = 12) :
  probability_more_grandsons_or_more_granddaughters = 793 / 1024 :=
by
  sorry

end probability_not_equal_genders_l137_137270


namespace tangent_line_eq_g_monotonic_interval_range_a_l137_137238

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := Real.log x + (-1 / x^2)

theorem tangent_line_eq (x : ℝ) (h : x = 1) : 
  (f x).diff x = x - 1 :=
by
  sorry

theorem g_monotonic_interval (x : ℝ) : 
  (diff (g x) > 0 ∧ x > 1) ∨ (diff (g x) < 0 ∧ 0 < x < 1) :=
by
  sorry

theorem range_a (a : ℝ) : 
  (0 < a < Real.exp 1) ∧ (∀ x > 0, g a - g x < 1/a) :=
by
  sorry

end tangent_line_eq_g_monotonic_interval_range_a_l137_137238


namespace concave_numbers_count_l137_137935

/--
A three-digit number is concave if its tens digit is less than both its hundreds digit and its units digit.
-/
def is_concave_number (a b c : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8} ∧
  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  b < a ∧
  b < c

/-- The total number of concave numbers is 285. -/
theorem concave_numbers_count : 
  (Finset.card { (a, b, c) ∈ Finset.univ : is_concave_number a b c }) = 285 :=
by {
  sorry
}

end concave_numbers_count_l137_137935


namespace option_D_is_base6_number_l137_137744

def contains_only_base6_digits (n : Nat) : Prop :=
  ∀ (d ∈ Nat.digits 10 n), d ∈ {0, 1, 2, 3, 4, 5}

theorem option_D_is_base6_number :
  contains_only_base6_digits 3142 :=
sorry

end option_D_is_base6_number_l137_137744


namespace height_of_tank_l137_137366

variable (length width height : ℝ)
variable (volume_added rise_in_level : ℝ)
variable (fraction_initial fraction_remaining : ℝ)

-- Conditions outlined from step a
def conditions :=
  (width = length / 2) ∧
  (volume_added = 76 * 0.1) ∧
  (rise_in_level = 0.38) ∧
  (fraction_initial = 3 / 8) ∧
  (fraction_remaining = 2 / 7)

-- The proof goal: height of the tank
def tank_height : Prop :=
  height = 1.12

theorem height_of_tank (h : conditions) : tank_height :=
begin
  sorry,
end

end height_of_tank_l137_137366


namespace log_base_8_32_l137_137910

theorem log_base_8_32 : ∀ a b N M n : ℝ, 
  (0 < a) → (a ≠ 1) → (0 < b) → (0 < N) → (N ≠ 1) → (0 < M) → 
  (∀ a n : ℝ, 0 < a → log a (a ^ n) = n) → 
  (∀ N M n : ℝ, 0 < N → N ≠ 1 → 0 < M → log N M = log n M / log n N) → 
  log 8 32 = 5 / 3 :=
by 
  sorry

end log_base_8_32_l137_137910


namespace number_of_integers_for_perfect_square_ratio_l137_137069

theorem number_of_integers_for_perfect_square_ratio :
  {n : ℤ | ∃ k : ℤ, n / (30 - n) = k ^ 2}.finite.to_finset.card = 3 := 
sorry

end number_of_integers_for_perfect_square_ratio_l137_137069


namespace tennis_balls_ordered_l137_137747

variables (W Y : ℕ)
def original_eq (W Y : ℕ) := W = Y
def ratio_condition (W Y : ℕ) := W / (Y + 90) = 8 / 13
def total_tennis_balls (W Y : ℕ) := W + Y = 288

theorem tennis_balls_ordered (W Y : ℕ) (h1 : original_eq W Y) (h2 : ratio_condition W Y) : total_tennis_balls W Y :=
sorry

end tennis_balls_ordered_l137_137747


namespace exponents_problem_l137_137408

theorem exponents_problem :
  5000 * (5000^9) * 2^(1000) = 5000^(10) * 2^(1000) := by sorry

end exponents_problem_l137_137408


namespace fractional_eq_range_m_l137_137882

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l137_137882


namespace sheets_prepared_l137_137401

variable (sheets_used sheets_left sheets_total : ℕ)

-- Define the conditions
def condition_1 : sheets_used = 12 := by rfl
def condition_2 : sheets_left = 9 := by rfl

-- Prove the total number of sheets prepared at the beginning
theorem sheets_prepared (h1 : sheets_used = 12) (h2 : sheets_left = 9) : sheets_total = 21 :=
  by
  -- (simplifying the statement)
  have h : sheets_used + sheets_left = 21 := by sorry
  exact h

end sheets_prepared_l137_137401


namespace probability_four_vertices_same_plane_proof_l137_137077

noncomputable def probability_four_vertices_same_plane : ℚ := 
  let total_ways := Nat.choose 8 4
  let favorable_ways := 12
  favorable_ways / total_ways

theorem probability_four_vertices_same_plane_proof : 
  probability_four_vertices_same_plane = 6 / 35 :=
by
  -- include necessary definitions and calculations for the actual proof
  sorry

end probability_four_vertices_same_plane_proof_l137_137077


namespace central_angle_of_regular_hexagon_l137_137327

theorem central_angle_of_regular_hexagon :
  ∀ (total_angle : ℝ) (sides : ℝ), total_angle = 360 → sides = 6 → total_angle / sides = 60 :=
by
  intros total_angle sides h_total_angle h_sides
  rw [h_total_angle, h_sides]
  norm_num

end central_angle_of_regular_hexagon_l137_137327


namespace total_surface_area_of_cut_cube_l137_137020

-- Define the side length of the cube given its volume of 2 cubic feet
def cube_side_length (volume : ℝ) : ℝ := Real.cbrt volume

-- Define the heights of the pieces resulting from the cuts
def height_E : ℝ := 2 / 3
def height_F : ℝ := 5 / 14
def height_G (side_length : ℝ) : ℝ := side_length - height_E - height_F

-- Define the surface areas for the top and bottom of all pieces (6 identical faces)
def top_bottom_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

-- Define the additional surface areas resulting from cuts
def surface_area_after_cuts (side_length : ℝ) (height_G : ℝ) : ℝ :=
  let common_surface_area := 2 * side_length^2
  in common_surface_area * 3

-- Define the total surface area of the solid after cuts
def total_surface_area (volume : ℝ) : ℝ :=
  let side_length := cube_side_length volume
  let height_g := height_G side_length
  top_bottom_surface_area(side_length) + surface_area_after_cuts(side_length, height_g)

-- State the theorem
theorem total_surface_area_of_cut_cube :
  total_surface_area 2 = 15.87 := by
  sorry

end total_surface_area_of_cut_cube_l137_137020


namespace which_could_be_true_l137_137121

-- Define the planes and lines
variables (α β : Plane) (l m : Line)

-- Define the conditions
axiom skew_lines_perpendicular : skew l m ∧ l ⟂ m
axiom plane_alpha_contains : contains α l ∧ contains α m
axiom plane_beta_contains : contains β l ∧ contains β m

theorem which_could_be_true :
  (α ∥ β) ∨ (α ⟂ β) ∨ (l ∥ β) ∨ (m ⟂ α) :=
by
  sorry

end which_could_be_true_l137_137121


namespace carp_probability_l137_137940

noncomputable def total_fish (grass_carp: ℕ) : ℕ := 1600 + grass_carp + 800

noncomputable def grass_carp_freq (grass_carp: ℕ) : Prop := (grass_carp : ℚ) / (total_fish grass_carp) = 0.5

theorem carp_probability :
  (∃ (grass_carp: ℕ), grass_carp_freq grass_carp) →
  (1600 : ℚ) / (total_fish 2400) = 1 / 3 := 
by
  intro h,
  let grass_carp := 2400,
  have h1 : total_fish grass_carp = 4800, from rfl,
  rw [show (total_fish grass_carp) = 4800 at h1, from h1],
  have h2 : grass_carp_freq 2400, from (by assumption : grass_carp_freq 2400),
  rw show (1600 : ℚ) / 4800 = 1 / 3, by norm_num,
  sorry

end carp_probability_l137_137940


namespace part1_part2_part3_l137_137202

theorem part1 {k b : ℝ} (h₀ : k ≠ 0) (h₁ : b = 1) (h₂ : k + b = 2) : 
  ∀ x : ℝ, y = k * x + b → y = x + 1 :=
by sorry

theorem part2 : ∃ (C : ℝ × ℝ), 
  C.1 = 3 ∧ C.2 = 4 ∧ y = x + 1 :=
by sorry

theorem part3 {n k : ℝ} (h₀ : k ≠ 0) 
  (h₁ : ∀ x : ℝ, x < 3 → (2 / 3) * x + n > x + 1 ∧ (2 / 3) * x + n < 4) 
  (h₂ : ∀ x : ℝ, y = (2 / 3) * x + n → y = 4 ∧ x = 3) :
  n = 2 :=
by sorry

end part1_part2_part3_l137_137202


namespace number_of_perfect_square_ratios_l137_137071

theorem number_of_perfect_square_ratios :
  (∃ n : ℕ, ∀ n < 30 ∧ ∃ k : ℕ, n / (30 - n) = k^2) = 4 := 
sorry

end number_of_perfect_square_ratios_l137_137071


namespace sum_of_complex_exponentials_l137_137475

theorem sum_of_complex_exponentials :
  15 * complex.exp (complex.I * π / 6) + 15 * complex.exp (complex.I * 5 * π / 6) = 
  15 * complex.exp (complex.I * π / 2) :=
by
  -- proof goes here
  sorry

end sum_of_complex_exponentials_l137_137475


namespace probability_of_sum_in_20_l137_137784

open_locale classical
noncomputable theory

def balls : finset ℕ := {5, 5, 5, 10, 10}

def total_ways := (balls.card.choose 3)

def favorable_ways := (({5}.card.choose 2) * ({10}.card.choose 1))

theorem probability_of_sum_in_20 : 
  favorable_ways / total_ways = 3 / 5 :=
sorry

end probability_of_sum_in_20_l137_137784


namespace total_number_of_trees_is_10_l137_137682

-- Define the known quantities
def percentage_tree_A : ℝ := 0.5
def percentage_tree_B : ℝ := 0.5

def oranges_per_tree_A : ℤ := 10
def good_percentage_A : ℝ := 0.6

def oranges_per_tree_B : ℤ := 15
def good_percentage_B : ℝ := 1 / 3

def total_good_oranges : ℤ := 55

-- Number of trees
theorem total_number_of_trees_is_10 : 
  ∀ (n : ℤ),
  (percentage_tree_A * (oranges_per_tree_A * good_percentage_A) + 
   percentage_tree_B * (oranges_per_tree_B * good_percentage_B)) * n = total_good_oranges →
  n = 10 :=
by
  intros n h
  sorry

end total_number_of_trees_is_10_l137_137682


namespace exists_trapezoid_in_marked_vertices_l137_137193

def is_regular_polygon (n : ℕ) (polygon : Type) [Fintype polygon] : Prop :=
  ∀ (u v w : polygon), u ≠ v → (∃ (p q : polygon), p ≠ q ∧ (p, q) ≠ (u, v) ∧ (p, q) = (v, w))

def is_trapezoid {α : Type*} [AffineSpace α] {polygon : Finset α} (v₁ v₂ v₃ v₄ : α) : Prop :=
  ∃ (m : AffineSubspace ℝ α),
  v₁ ∈ m ∧ v₂ ∈ m ∧ v₃ ∉ m ∧ v₄ ∉ m

theorem exists_trapezoid_in_marked_vertices :
  ∀ (polygon : Type) [Fintype polygon] (n : ℕ) (m : ℕ),
  is_regular_polygon n polygon →
  m = 64 → n = 1981 → ∃ (v₁ v₂ v₃ v₄ : polygon), is_trapezoid v₁ v₂ v₃ v₄ :=
begin
  intros,
  sorry
end

end exists_trapezoid_in_marked_vertices_l137_137193


namespace cos_omega_area_of_triangle_ABC_l137_137216

-- Definitions typically put here

-- Condition definitions only from the original problem
def vectors_parallel (a b : ℝ × ℝ) : Prop := a.2 * b.1 = a.1 * b.2

def angle_in_first_quadrant (A : ℝ) : Prop := 0 < A ∧ A < π / 2

def sin_omega_minus_A (ω A : ℝ) : Prop := sin(ω - A) = 3 / 5 ∧ 0 < ω ∧ ω < π / 2

def triangle_side_lengths (BC : ℝ) (AC : ℝ) (AB : ℝ) : Prop := 
  BC = 2 * sqrt 3 ∧ AC + AB = 4

-- Lean Statement to prove cos omega
theorem cos_omega (A ω : ℝ) (h1 : vectors_parallel (sin A, 1) (cos A, sqrt 3))
    (h2 : angle_in_first_quadrant A) (h3 : sin_omega_minus_A ω A) :
    cos ω = (4 * sqrt 3 - 3) / 10 :=
  sorry

-- Lean Statement to prove the area of the triangle ABC
theorem area_of_triangle_ABC (A BC AC AB : ℝ)
    (h1 : angle_in_first_quadrant A) (h2 : BC = 2 * sqrt 3) (h3 : AC + AB = 4) :
    nat.sin A = 1/2 →  nat.angle_in_first_quadrant (A)
    (area : cos A) = 2 - sqrt 3 :=
    sorry

end cos_omega_area_of_triangle_ABC_l137_137216


namespace round_nearest_hundredth_l137_137297

theorem round_nearest_hundredth : 
  let x := 37.837837837 in
  (Real.round (x * 100) / 100 = 37.84) :=
  sorry

end round_nearest_hundredth_l137_137297


namespace trapezoid_AD_BC_ratio_l137_137862

variables {A B C D M N K : Type} {AD BC CM MD NA CN : ℝ}

-- Definition of the trapezoid and the ratio conditions
def is_trapezoid (A B C D : Type) : Prop := sorry -- Assume existence of a trapezoid for lean to accept the statement
def ratio_CM_MD (CM MD : ℝ) : Prop := CM / MD = 4 / 3
def ratio_NA_CN (NA CN : ℝ) : Prop := NA / CN = 4 / 3

-- Proof statement for the given problem
theorem trapezoid_AD_BC_ratio 
  (h_trapezoid: is_trapezoid A B C D)
  (h_CM_MD: ratio_CM_MD CM MD)
  (h_NA_CN: ratio_NA_CN NA CN) :
  AD / BC = 7 / 12 :=
sorry

end trapezoid_AD_BC_ratio_l137_137862


namespace irrational_number_between_one_and_three_l137_137282

theorem irrational_number_between_one_and_three : ∃ (x : ℝ), irrational x ∧ 1 < x ∧ x < 3 :=
begin
  use sqrt 2,
  split,
  { exact real.sqrt_two_not_rational, },
  split,
  { linarith [real.sqrt_pos.mpr (by norm_num : 0 < 2)], },
  { linarith [real.sqrt_lt_sqrt_iff (lt_trans zero_lt_one (by norm_num : 1 < 9)), (by norm_num : 2 < 9)], },
end

end irrational_number_between_one_and_three_l137_137282


namespace christian_age_in_eight_years_l137_137472

-- Definitions from the conditions
def christian_current_age : ℕ := 72
def brian_age_in_eight_years : ℕ := 40

-- Theorem to prove
theorem christian_age_in_eight_years : ∃ (age : ℕ), age = christian_current_age + 8 ∧ age = 80 := by
  sorry

end christian_age_in_eight_years_l137_137472


namespace dan_gave_marbles_l137_137477

-- Conditions as definitions in Lean 4
def original_marbles : ℕ := 64
def marbles_left : ℕ := 50
def marbles_given : ℕ := original_marbles - marbles_left

-- Theorem statement proving the question == answer given the conditions.
theorem dan_gave_marbles : marbles_given = 14 := by
  sorry

end dan_gave_marbles_l137_137477


namespace net_effect_on_sale_value_l137_137932

theorem net_effect_on_sale_value 
  (P Original_Sales_Volume : ℝ) 
  (reduced_by : ℝ := 0.18) 
  (sales_increase : ℝ := 0.88) 
  (additional_tax : ℝ := 0.12) :
  P * Original_Sales_Volume * ((1 - reduced_by) * (1 + additional_tax) * (1 + sales_increase) - 1) = P * Original_Sales_Volume * 0.7184 :=
  by
  sorry

end net_effect_on_sale_value_l137_137932


namespace vector_addition_l137_137994

-- Step 1: Define the vectors a and b
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 4)

-- Step 2: State the theorem
theorem vector_addition : 2 • a + b = (1 : ℝ, 2 : ℝ) :=
by
  sorry

end vector_addition_l137_137994


namespace motorcycle_distance_travelled_l137_137396

theorem motorcycle_distance_travelled:
  let a := 40 in
  let d := -10 in
  let n := 5 in
  let s_n := (n * (2 * a + (n - 1) * d)) / 2 in
  s_n = 100 :=
by
  sorry

end motorcycle_distance_travelled_l137_137396


namespace tangent_sum_l137_137466

theorem tangent_sum :
  (Finset.sum (Finset.range 2019) (λ k => Real.tan ((k + 1) * Real.pi / 47) * Real.tan ((k + 2) * Real.pi / 47))) = -2021 :=
by
  -- proof will be completed here
  sorry

end tangent_sum_l137_137466


namespace polar_eq_proof_l137_137948

-- Define the parametric equations for curve C
def param_eq (α : ℝ) : ℝ × ℝ :=
  (1 + sqrt 2 * cos α, 1 + sqrt 2 * sin α)

-- Define the polar equation for curve C
def polar_eq (θ : ℝ) : ℝ :=
  2 * (cos θ + sin θ)

-- Given conditions
def A_polar : ℝ × ℝ :=
  (2, (5 / 6) * Real.pi)

def OB_polar : ℝ :=
  sqrt 6

def OA : ℝ := 2

theorem polar_eq_proof :
  (∃ ρ θ, (param_eq α = (1 + sqrt 2 * cos α, 1 + sqrt 2 * sin α) →
   polar_eq θ = ρ) ∧ 
  (A_polar = (2, (5 / 6) * Real.pi)) ∧
  (|OA * OB_polar| = 2 * sqrt 6) →
  (angle_OAB = Real.pi * 3 / 4 ∨ angle_OAB = Real.pi * 5 / 12)) := sorry

end polar_eq_proof_l137_137948


namespace avg_pages_hr_difference_l137_137973

noncomputable def avg_pages_hr_diff (total_pages_ryan : ℕ) (hours_ryan : ℕ) (books_brother : ℕ) (pages_per_book : ℕ) (hours_brother : ℕ) : ℚ :=
  (total_pages_ryan / hours_ryan : ℚ) - (books_brother * pages_per_book / hours_brother : ℚ)

theorem avg_pages_hr_difference :
  avg_pages_hr_diff 4200 78 15 250 90 = 12.18 :=
by
  sorry

end avg_pages_hr_difference_l137_137973


namespace arithmetic_sequence_product_l137_137359

theorem arithmetic_sequence_product (a : ℕ → ℚ) (d : ℚ) (a7 : a 7 = 20) (diff : d = 2) : 
  let a1 := a 1,
      a2 := a 2
  in a1 * a2 = 80 :=
by
  sorry

end arithmetic_sequence_product_l137_137359


namespace handshakes_proof_l137_137036

def total_handshakes (team_size : ℕ) (referee_count : ℕ) : ℕ :=
  let inter_team_handshakes := team_size * team_size
  let intra_team_handshakes := team_size * (team_size - 1) / 2
  let total_intra_team_handshakes := 2 * intra_team_handshakes
  let referee_handshakes := 2 * team_size * referee_count
  inter_team_handshakes + total_intra_team_handshakes + referee_handshakes

theorem handshakes_proof : total_handshakes 6 3 = 102 :=
by
  -- Definitions of handshakes
  let inter_team_handshakes := 6 * 6
  let intra_team_handshakes := (6 * (6 - 1)) / 2
  let total_intra_team_handshakes := 2 * intra_team_handshakes
  let referee_handshakes := 2 * 6 * 3
  -- Calculate the total handshakes
  have total_handshakes := inter_team_handshakes + total_intra_team_handshakes + referee_handshakes
  -- Assert the total is 102
  show total_handshakes = 102
  sorry

end handshakes_proof_l137_137036


namespace prob_4_vertices_same_plane_of_cube_l137_137079

/-- A cube contains 8 vertices. We need to calculate the probability that 4 chosen vertices lie on the same plane. -/
theorem prob_4_vertices_same_plane_of_cube : 
  let total_ways := (Nat.choose 8 4)
  let favorable_ways := 12
  in (favorable_ways / total_ways : ℚ) = 6 / 35 :=
by
  sorry

end prob_4_vertices_same_plane_of_cube_l137_137079


namespace sum_abs_slopes_AB_l137_137701

-- Defining the geometrical setup
def point := (ℤ × ℤ)
def trapezoid (A B C D : point) : Prop :=
    ∃ M N : ℤ,  A = (20, 100) ∧ D = (21, 107) ∧
    (B : point) ∧ (C : point) ∧ (B.2 - A.2) * (D.1 - C.1) = (B.1 - A.1) * (D.2 - C.2) ∧
    ¬((B.1 = A.1 ∨ C.1 = D.1) ∨ (B.2 = A.2 ∨ C.2 = D.2))

-- Prove that this holds for the sum of the absolute values of all possible slopes for AB
theorem sum_abs_slopes_AB : 
    ∀ A B C D : point,
    trapezoid A B C D →
    ∃ (m n : ℕ), Nat.coprime m n ∧ (∑ slope in (all_possible_slopes A B C D), abs slope) = (119/12) ∧ (m+n = 131) :=
by
  intros A B C D h
  sorry

-- Helper function listing all possible slopes
noncomputable def all_possible_slopes (A B C D : point) : list ℚ := 
    sorry

end sum_abs_slopes_AB_l137_137701


namespace proof_problem_l137_137902

def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)

-- Prove that the smallest positive period of f(x) is 2π
def is_period_2pi : Prop :=
  ∀ x, f (x + 2 * Real.pi) = f x

-- Prove that f(π/2) is not the maximum value of f(x)
def is_not_maximum_value : Prop :=
  f (Real.pi / 2) < 1

-- Prove that shifting the graph of y = sin(x) to the left by π/3 results in y = f(x)
def shifting_result : Prop :=
  ∀ x, Real.sin (x + Real.pi / 3) = f x

theorem proof_problem :
  is_period_2pi ∧ is_not_maximum_value ∧ shifting_result :=
by {
  -- period proof
  sorry,
  -- not maximum value proof
  sorry,
  -- shifting result proof
  sorry
}

end proof_problem_l137_137902


namespace Alex_final_silver_tokens_l137_137788

variable (x y : ℕ)

def final_red_tokens (x y : ℕ) : ℕ := 90 - 3 * x + 2 * y
def final_blue_tokens (x y : ℕ) : ℕ := 65 + 2 * x - 4 * y
def silver_tokens (x y : ℕ) : ℕ := x + y

theorem Alex_final_silver_tokens (h1 : final_red_tokens x y < 3)
                                 (h2 : final_blue_tokens x y < 4) :
  silver_tokens x y = 67 := 
sorry

end Alex_final_silver_tokens_l137_137788


namespace one_third_1206_percent_400_l137_137277

theorem one_third_1206_percent_400 : (402 = (1.005 * 400)) :=
by
  have h1 : 1206 / 3 = 402 := by norm_num
  have h2 : 1.005 * 400 = 402 := by norm_num
  exact h2

end one_third_1206_percent_400_l137_137277


namespace probability_four_vertices_same_plane_proof_l137_137078

noncomputable def probability_four_vertices_same_plane : ℚ := 
  let total_ways := Nat.choose 8 4
  let favorable_ways := 12
  favorable_ways / total_ways

theorem probability_four_vertices_same_plane_proof : 
  probability_four_vertices_same_plane = 6 / 35 :=
by
  -- include necessary definitions and calculations for the actual proof
  sorry

end probability_four_vertices_same_plane_proof_l137_137078


namespace proof_problem_l137_137897

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l137_137897


namespace soccer_club_girls_count_l137_137782

theorem soccer_club_girls_count
  (total_members : ℕ)
  (attended : ℕ)
  (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : (1/3 : ℚ) * G + B = 18) : G = 18 := by
  sorry

end soccer_club_girls_count_l137_137782


namespace negation_of_all_students_punctual_l137_137707

open_locale classical

-- Definitions
def student (x : Type) : Prop := sorry
def punctual (x : Type) : Prop := sorry

-- Theorem
theorem negation_of_all_students_punctual {α : Type} :
  (¬ ∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) :=
by
  sorry

end negation_of_all_students_punctual_l137_137707


namespace orthocenters_cyclic_l137_137657

noncomputable def cyclic_quad (A1 A2 A3 A4 : ℂ) : Prop :=
  ∃ (O : ℂ) (R : ℝ), ∀ i ∈ {A1, A2, A3, A4}, complex.abs (i - O) = R

def orthocenter (A B C : ℂ) : ℂ :=
  let Δ := (C - A) * (B - A).conj;
  let H := A + ((B - A) * (C - A).conj + (C - A) * (B - A).conj) / (Δ + Δ.conj);
  H

theorem orthocenters_cyclic (A1 A2 A3 A4 : ℂ) (h_cyclic : cyclic_quad A1 A2 A3 A4) :
  ∃ (O' : ℂ) (R' : ℝ),
    ∀ (H : ℂ),
      H ∈ {orthocenter A2 A3 A4, orthocenter A3 A4 A1, orthocenter A4 A1 A2, orthocenter A1 A2 A3} →
      complex.abs (H - O') = R' := 
sorry

end orthocenters_cyclic_l137_137657


namespace greendale_high_school_points_l137_137291

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l137_137291


namespace value_of_2xy_l137_137085

noncomputable def y (x : ℝ) : ℝ := sqrt (2 * x - 5) + sqrt (5 - 2 * x) - 3

theorem value_of_2xy :
  let x := (5 : ℝ) / 2
  let y := y x
  2 * x * y = -15 :=
by
  sorry

end value_of_2xy_l137_137085


namespace volume_of_box_l137_137393

theorem volume_of_box (l w h : ℝ) (h1 : l * w = 24) (h2 : w * h = 16) (h3 : l * h = 6) :
  l * w * h = 48 :=
by
  sorry

end volume_of_box_l137_137393


namespace knights_threatening_each_other_l137_137614

-- Define the size of the chessboard
def n : ℕ := 16

-- Conditions under which knights threaten each other
def threatened (x1 y1 x2 y2 : ℕ) : Prop :=
  (|x1 - x2| = 1 ∧ |y1 - y2| = 2) ∨ (|x1 - x2| = 2 ∧ |y1 - y2| = 1)

-- The main theorem we want to prove
theorem knights_threatening_each_other : 
  ∑ x1 in (finset.range n), ∑ y1 in (finset.range n), ∑ x2 in (finset.range n), ∑ y2 in (finset.range n), 
  if threatened x1 y1 x2 y2 then 1 else 0 = 1680 :=
sorry

end knights_threatening_each_other_l137_137614


namespace medal_winners_combinations_l137_137185

theorem medal_winners_combinations:
  ∀ n k : ℕ, (n = 6) → (k = 3) → (n.choose k = 20) :=
by
  intros n k hn hk
  simp [hn, hk]
  -- We can continue the proof using additional math concepts if necessary.
  sorry

end medal_winners_combinations_l137_137185


namespace cost_of_45_roses_l137_137034

theorem cost_of_45_roses
  (initial_roses : ℕ) (initial_cost : ℝ)
  (total_roses : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ)
  (proportional_cost : ℝ) :
  initial_roses = 15 →
  initial_cost = 30 →
  total_roses = 45 →
  discount_threshold = 30 →
  discount_rate = 0.10 →
  proportional_cost = (initial_cost / initial_roses) * total_roses →
  total_roses > discount_threshold →
  proportional_cost * (1 - discount_rate) = 81 := by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  sorry

end cost_of_45_roses_l137_137034


namespace find_remainder_l137_137977

def is_valid_word (word : List ℕ) : Prop :=
  word.length = 12 ∧ (∀ i, i < 11 → abs (word.nth_le i sorry  - word.nth_le (i + 1) sorry) = 1)

def count_valid_words (alphabet : List ℕ) (len : ℕ) :=
  (List.list_all_pairs len alphabet).countp is_valid_word

theorem find_remainder : 
  let A := count_valid_words [0, 1, 2, 3, 4, 5, 6] 12 in
  A % 2008 = 1200 :=
by 
  sorry

end find_remainder_l137_137977


namespace min_chord_circle_l137_137858

variables {A B O C : Point} {l : Line} 

/-- Assume A and B are points on opposite sides of line l,
    and O is the intersection of line l with segment AB.
    Prove that the center C of a circle passing through A and B,
    with the minimum length of the chord intercepted by l,
    is the intersection of the perpendicular bisector of segment AB 
    and the perpendicular line to l passing through O. -/
theorem min_chord_circle
  (A B O C : Point) 
  (l : Line)
  (A_opposite_B : A ≠ B ∧ ¬ (same_side l A B))
  (O_intersection : O ∈ l ∧ O ∈ segment A B) 
  (C_on_G : C ∈ perpendicular_bisector A B ∧ C ∈ perpendicular_to l O) :
  ∀ S : Circle, 
    (A ∈ S) ∧ (B ∈ S) ∧ (∀ M N : Point, (M ∈ S) ∧ (N ∈ S) ∧ (M ≠ N) ∧ (M, N ∈ l) → (chord_length M N = minimum_chord_length) ↔ C ∈ center S) := 
sorry

end min_chord_circle_l137_137858


namespace combination_identity_l137_137802

-- Lean statement defining the proof problem
theorem combination_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 :=
  sorry

end combination_identity_l137_137802


namespace sum_of_divisors_of_n_squared_plus_n_plus_two_l137_137840

theorem sum_of_divisors_of_n_squared_plus_n_plus_two :
  (∑ n in Finset.filter (λ n, n ∣ (n^2 + n + 2)) (Finset.range 100), n) = 3 :=
by
  sorry

end sum_of_divisors_of_n_squared_plus_n_plus_two_l137_137840


namespace no_real_roots_of_x_squared_plus_5_l137_137047

theorem no_real_roots_of_x_squared_plus_5 : ¬ ∃ (x : ℝ), x^2 + 5 = 0 :=
by
  sorry

end no_real_roots_of_x_squared_plus_5_l137_137047


namespace inverse_of_matrixA_l137_137836
-- Import the necessary library

-- Define the matrix
def matrixA : Matrix (Fin 2) (Fin 2) ℚ := 
  ![[5, -3], [-2, 1]]

-- Define the expected inverse matrix
def inverseMatrixA : Matrix (Fin 2) (Fin 2) ℚ := 
  ![[-1, -3], [-2, -5]]

-- State the theorem
theorem inverse_of_matrixA : matrixA⁻¹ = inverseMatrixA :=
by
  sorry

end inverse_of_matrixA_l137_137836


namespace pies_differ_in_both_l137_137168

-- Defining types of pies
inductive Filling where
  | apple : Filling
  | cherry : Filling

inductive Preparation where
  | fried : Preparation
  | baked : Preparation

structure Pie where
  filling : Filling
  preparation : Preparation

-- The set of all possible pies
def allPies : Set Pie :=
  { ⟨Filling.apple, Preparation.fried⟩,
    ⟨Filling.apple, Preparation.baked⟩,
    ⟨Filling.cherry, Preparation.fried⟩,
    ⟨Filling.cherry, Preparation.baked⟩ }

-- Theorem stating that we can buy two pies that differ in both filling and preparation
theorem pies_differ_in_both (pies : Set Pie) (h : 3 ≤ pies.card) :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
  sorry

end pies_differ_in_both_l137_137168


namespace range_of_m_l137_137891

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → (m ≤ 2 ∧ m ≠ -2) :=
begin
  sorry
end

end range_of_m_l137_137891


namespace log_sum_zero_l137_137482

theorem log_sum_zero : 
  (∑ i in Finset.range 44, Real.log 2 (Real.tan (1 + i) * π / 180)) = 0 :=
by
  sorry

end log_sum_zero_l137_137482


namespace max_elements_no_sum_divisible_by_5_l137_137643

open Finset Function

def no_sum_divisible_by {T : Finset ℕ} (s : ℕ) : Prop :=
∀ {a b : ℕ}, a ∈ T → b ∈ T → a ≠ b → (a + b) % s ≠ 0

theorem max_elements_no_sum_divisible_by_5 :
  ∀ T : Finset ℕ, T ⊆ range 1 101 → no_sum_divisible_by 5 T → T.card ≤ 60 :=
by
  intros
  sorry

end max_elements_no_sum_divisible_by_5_l137_137643


namespace minimum_value_sum_of_squares_l137_137242

theorem minimum_value_sum_of_squares :
  ∃ (p q r s t u v w : ℤ), 
  {p, q, r, s, t, u, v, w} = {-6, -4, -1, 0, 3, 5, 7, 10} → 
  ∀ {x : ℤ}, x = p + q + r + s → 
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 98 := 
by
  sorry

end minimum_value_sum_of_squares_l137_137242


namespace triangle_area_percentage_l137_137432

noncomputable def percent_triangle_area (x : ℝ) : ℝ :=
  let area_triangle := 0.5 * 2 * x * (x * Real.sqrt 3 / 2) in
  let area_rectangle := 2 * x^2 in
  let area_pentagon := area_triangle + area_rectangle in
  (area_triangle / area_pentagon) * 100

theorem triangle_area_percentage (x : ℝ) (hx : x > 0) :
  percent_triangle_area x = (Real.sqrt 3 / (Real.sqrt 3 + 4)) * 100 :=
sorry

end triangle_area_percentage_l137_137432


namespace coefficient_x6_in_expansion_l137_137736

theorem coefficient_x6_in_expansion :
  (∃ c : ℕ, c = 81648 ∧ (3 : ℝ) ^ 6 * c * 2 ^ 2  = c * (3 : ℝ) ^ 6 * 4) :=
sorry

end coefficient_x6_in_expansion_l137_137736


namespace liza_phone_bill_eq_70_l137_137670

theorem liza_phone_bill_eq_70 (initial_balance rent payment paycheck electricity internet final_balance phone_bill : ℝ)
  (h1 : initial_balance = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity = 117)
  (h5 : internet = 100)
  (h6 : final_balance = 1563)
  (h_balance_before_phone_bill : initial_balance - rent + paycheck - (electricity + internet) = 1633)
  (h_final_balance_def : 1633 - phone_bill = final_balance) :
  phone_bill = 70 := sorry

end liza_phone_bill_eq_70_l137_137670


namespace triangle_distance_BD_l137_137214

theorem triangle_distance_BD
  (A B C D E : Type)
  (h_triangle_ABC : ∠C = 90 ∨ AC = 9 ∨ BC = 12)
  (h_points_on_lines : (D ∈ line(A, B) ∧ E ∈ line(B, C)))
  (h_angle_BED_angle : ∠BED = 90)
  (h_DE : DE = 6) :
  BD = 10 := 
sorry

end triangle_distance_BD_l137_137214


namespace triangularizable_iff_invariant_subspaces_l137_137289

variables (V : Type*) [AddCommGroup V] [Module ℝ V] 
variables (n : ℕ) (T : V →ₗ[ℝ] V)

theorem triangularizable_iff_invariant_subspaces :
  (∃ (basis : Fin n → V), let M := LinearMap.to_matrix basis basis T
                         in is_upper_triangular M) ↔
  (∃ (V_i : Fin n → Submodule ℝ V), 
    (∀ i, T '' V_i i ≤ V_i i) ∧ strict_mono ((↑) ∘ V_i)) :=
sorry

end triangularizable_iff_invariant_subspaces_l137_137289


namespace find_smallest_k_l137_137739

theorem find_smallest_k : ∃ (k : ℕ), 64^k > 4^20 ∧ ∀ (m : ℕ), (64^m > 4^20) → m ≥ k := sorry

end find_smallest_k_l137_137739


namespace tower_count_l137_137768

theorem tower_count (red_cubes blue_cubes green_cubes yellow_cubes total_cubes tower_height : ℕ)
  (H1 : red_cubes = 3)
  (H2 : blue_cubes = 4)
  (H3 : green_cubes = 2)
  (H4 : yellow_cubes = 2)
  (H5 : total_cubes = 11)
  (H6 : tower_height = 10) 
  (Hfixed_yellow : 1 ≤ yellow_cubes) :
  (∑ n in {red_cubes, blue_cubes, green_cubes, yellow_cubes}, n) = total_cubes →
  (∃ (valid_arrangements : ℕ), valid_arrangements = 1260)
:= 
by 
  -- Variables for the remaining cubes after fixing a yellow cube at the top
  let remaining_red_cubes := red_cubes
  let remaining_blue_cubes := blue_cubes
  let remaining_green_cubes := green_cubes
  let remaining_yellow_cubes := yellow_cubes - 1
  
  -- Total remaining cubes should be 9
  have Hremaining_cubes : remaining_red_cubes + remaining_blue_cubes + remaining_green_cubes + remaining_yellow_cubes = 9 := sorry,
  
  -- Calculating the number of valid arrangements
  let valid_arrangements := (Nat.factorial 9) / (Nat.factorial remaining_red_cubes * Nat.factorial remaining_blue_cubes * Nat.factorial remaining_green_cubes * Nat.factorial remaining_yellow_cubes),
  
  -- Proving the final result
  use valid_arrangements,
  have Heq : valid_arrangements = 1260 := sorry,
  exact Heq

end tower_count_l137_137768


namespace log_five_one_over_sqrt_five_l137_137824

theorem log_five_one_over_sqrt_five : log 5 (1 / real.sqrt 5) = -1 / 2 := sorry

end log_five_one_over_sqrt_five_l137_137824


namespace angle_AKB_150_deg_l137_137628

-- Definitions for the geometric setup.
variable (α β : Type*) [InnerProductSpace ℝ α]
variables {A B C K : α}
variables {angleKAC : ℝ}

def is_isosceles (A B C : α) : Prop :=
  dist A B = dist B C

def angle_eq (A B C : α) (d : ℝ) : Prop :=
  inner (B - A) (C - A) = d * (norm (B - A) * norm (C - A))

-- Main theorem statement.
theorem angle_AKB_150_deg (h_iso : is_isosceles A B C)
  (h_CK_AB : dist C K = dist A B)
  (h_CK_CB : dist C K = dist B C)
  (h_angle_30 : angleKAC = 30)
  : angle_eq A K B (cos (150 * π / 180)) :=
sorry

end angle_AKB_150_deg_l137_137628


namespace sum_first_1985_bad_numbers_eq_l137_137436

def is_bad_number (n : ℕ) : Prop :=
  nat.bits n |> list.count 1 |> (λ x, x % 2 = 0)

theorem sum_first_1985_bad_numbers_eq :
  (finset.filter is_bad_number (finset.range 4000)).sum $ λ x, x = 
  2^21 + 2^20 + 2^19 + 2^18 + 2^13 + 2^11 + 2^9 + 2^8 + 2^5 + 2 + 1 :=
sorry

end sum_first_1985_bad_numbers_eq_l137_137436


namespace Beth_finishes_first_l137_137457

variable (A R: ℝ)

def mowing_time (area speed: ℝ) : ℝ :=
  area / speed

theorem Beth_finishes_first 
  (Beth_lawn : ℝ := A / 3)
  (Carlos_lawn : ℝ := A / 4)
  (Andy_speed : ℝ := R)
  (Beth_speed : ℝ := R / 2)
  (Carlos_speed : ℝ := R / 4)
  (Andy_time := mowing_time A Andy_speed)
  (Beth_time := mowing_time Beth_lawn Beth_speed)
  (Carlos_time := mowing_time Carlos_lawn Carlos_speed) :
  min Andy_time (min Beth_time Carlos_time) = Beth_time := 
by
  have Andy_time_def : Andy_time = A / R := 
    rfl
  have Carlos_time_def : Carlos_time = A / R :=
    by rw [mowing_time, Carlos_lawn, Carlos_speed, div_div]
  have Beth_time_def : Beth_time = 2 * A / (3 * R) :=
    by rw [mowing_time, Beth_lawn, Beth_speed, div_div, mul_comm (3:ℝ), div_eq_mul_one_div]

  rw [min_comm Andy_time _, min_assoc, min_comm Andy_time _]
  rw [Andy_time_def, Carlos_time_def, Beth_time_def]
  linarith

end Beth_finishes_first_l137_137457


namespace smallest_x_for_factorial_divisibility_l137_137936

theorem smallest_x_for_factorial_divisibility :
  ∃ x : ℕ, 0 < x ∧ 100000 ∣ factorial x ∧ ∀ y : ℕ, 0 < y ∧ 100000 ∣ factorial y → x ≤ y :=
begin
  sorry
end

end smallest_x_for_factorial_divisibility_l137_137936


namespace parabola_vertex_is_two_one_l137_137699

theorem parabola_vertex_is_two_one : 
  ∀ x y : ℝ, (y = (x - 2)^2 + 1) → (2, 1) = (2, 1) :=
by
  intros x y hyp
  sorry

end parabola_vertex_is_two_one_l137_137699


namespace minimum_harmonic_sum_l137_137237

theorem minimum_harmonic_sum
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
by
  sorry

end minimum_harmonic_sum_l137_137237


namespace total_elements_in_set_X_l137_137302

-- Define the sets X and Y and the necessary conditions
variable (X Y : Set ℕ)
variable (x y : ℕ)
variable (X_union_Y : ℕ)
variable (X_inter_Y : ℕ)

-- Assign the given conditions
axiom h1 : X_union_Y = 5000
axiom h2 : x = 3 * y
axiom h3 : X_inter_Y = 1500

-- The union equation derived from the conditions
noncomputable def total_elements_X : ℕ :=
  x

-- Statement of the theorem with the target proof
theorem total_elements_in_set_X :
  (X_union_Y - X_inter_Y) + 1500 = 5000  ∧ x = 3 * y ∧ X_inter_Y = 1500 → total_elements_X = 4875 :=
begin
  sorry
end

end total_elements_in_set_X_l137_137302


namespace parallelogram_diagonal_length_l137_137984

theorem parallelogram_diagonal_length (ABCD : Parallelogram ℝ) (Area_24 : area ABCD = 24)
  (P Q : Point ℝ) (hp : IsProjection P A BD) (hq : IsProjection Q C BD)
  (R S : Point ℝ) (hr : IsProjection R B AC) (hs : IsProjection S D AC)
  (PQ_8 : distance P Q = 8) (RS_10 : distance R S = 10) :
  ∃ (m n p : ℝ), (m > 0 ∧ n > 0 ∧ p > 0 ∧ RationalSquareRootFree p) ∧ (distance BD ^ 2 = m + n * real.sqrt p) ∧ (m + n + p = 50) :=
sorry

end parallelogram_diagonal_length_l137_137984


namespace equation_of_line_l137_137869

/-- Given M(4, 2) as the midpoint of the chord AB intercepted by the ellipse 
  x^2 + 4y^2 = 36 on the line l, the equation of the line l is x + 2y - 8 = 0. -/
theorem equation_of_line
  (M : ℝ × ℝ)
  (hM : M = (4, 2))
  (A B : ℝ × ℝ)
  (hA : A.1^2 + 4 * A.2^2 = 36)
  (hB : B.1^2 + 4 * B.2^2 = 36)
  (h_mid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ (l : ℝ → ℝ → Prop), (∀ x y, l x y ↔ x + 2 * y - 8 = 0) :=
by
  use λ x y, x + 2 * y - 8 = 0
  sorry

end equation_of_line_l137_137869


namespace vendor_has_maaza_l137_137424

theorem vendor_has_maaza (liters_pepsi : ℕ) (liters_sprite : ℕ) (total_cans : ℕ) (gcd_pepsi_sprite : ℕ) (cans_pepsi : ℕ) (cans_sprite : ℕ) (cans_maaza : ℕ) (liters_per_can : ℕ) (total_liters_maaza : ℕ) :
  liters_pepsi = 144 →
  liters_sprite = 368 →
  total_cans = 133 →
  gcd_pepsi_sprite = Nat.gcd liters_pepsi liters_sprite →
  gcd_pepsi_sprite = 16 →
  cans_pepsi = liters_pepsi / gcd_pepsi_sprite →
  cans_sprite = liters_sprite / gcd_pepsi_sprite →
  cans_maaza = total_cans - (cans_pepsi + cans_sprite) →
  liters_per_can = gcd_pepsi_sprite →
  total_liters_maaza = cans_maaza * liters_per_can →
  total_liters_maaza = 1616 :=
by
  sorry

end vendor_has_maaza_l137_137424


namespace probability_of_shaded_triangle_l137_137209

def triangle (A B C : Type) : Prop := true

-- Define the four triangles present
def is_triangle_ABC (A B C : Type) : Prop := triangle A B C
def is_triangle_ABD (A B D : Type) : Prop := triangle A B D
def is_triangle_ADC (A D C : Type) : Prop := triangle A D C
def is_triangle_BDC (B D C : Type) : Prop := triangle B D C

-- Define the shading condition
def is_shaded (T : Prop) : Prop :=
  T = is_triangle_ADC _ _ _
  ∨ T = is_triangle_BDC _ _ _

-- Total number of triangles is 4
def total_triangles := 4

-- Number of shaded triangles is 2
def shaded_triangles := 2

-- Probability calculation
def shaded_probability : ℚ :=
  shaded_triangles / total_triangles

theorem probability_of_shaded_triangle :
  shaded_probability = 1 / 2 := by
  -- proof would go here
  sorry

end probability_of_shaded_triangle_l137_137209


namespace smallest_part_of_division_l137_137138

theorem smallest_part_of_division (x : ℝ) (h : 2 * x + (1/2) * x + (1/4) * x = 105) : 
  (1/4) * x = 10.5 :=
sorry

end smallest_part_of_division_l137_137138


namespace sum_of_extremes_of_g_l137_137240

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - abs (3 * x - 9)

theorem sum_of_extremes_of_g :
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 10) → let largest := max (max (g 1) (g 3)) (g 5) in
                               let smallest := min (min (g 1) (g 3)) (g 5) in
                               largest + smallest = 1 :=
begin
  sorry
end

end sum_of_extremes_of_g_l137_137240


namespace water_left_after_usage_l137_137632

theorem water_left_after_usage : 
  ∀ (initial_gallons used_gallons : ℚ), initial_gallons = 3 ∧ used_gallons = 5/4 → initial_gallons - used_gallons = 7/4 :=
by
  intros 
  rintro ⟨h1, h2⟩
  rw [h1, h2]
  exact sub_eq_of_eq_add (show (7 / 4 + 5 / 4) = 3 by norm_num)
  sorry

end water_left_after_usage_l137_137632


namespace length_of_MN_l137_137435

theorem length_of_MN (a b : ℝ) (h_ab : ∃ A B C E M N : Type, E ∈ line_segment A B ∧ isosceles_triangle A B C ∧ 
                                        are_incircletouched_points ACE EMI ∧ are_incircletouched_points ECB CNI):
  let MN := |a - b| / 2
  in MN = |a - b| / 2 :=
sorry

end length_of_MN_l137_137435


namespace find_n_l137_137713

theorem find_n (n : ℕ) (h1 : (∃ k : ℕ, n = 2 * k + 1) ∧ (∑ i in (finset.filter (λ x, even x) (finset.range n)), i) = 85 * 86) : n = 171 :=
sorry

end find_n_l137_137713


namespace trajectory_of_moving_circle_center_l137_137547

theorem trajectory_of_moving_circle_center :
  ∀ (x y : ℝ),
    ((x + 1)^2 + y^2 = 1) ∧ ((x - 1)^2 + y^2 = 25) →
    (∃ c : ℝ, (c + 1)^2 + y^2 ≤ 1 ∧ (c - 1)^2 + y^2 ≤ 25) →
    (∃ r : ℝ, (r + c + 1) = (r + 5 - r) = 6) →
    (x^2 / 9 + y^2 / 8 = 1) :=
by
  intro x y h h1 h2
  sorry

end trajectory_of_moving_circle_center_l137_137547


namespace max_value_of_m_l137_137856

theorem max_value_of_m
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (2 / a) + (1 / b) = 1 / 4)
  (h4 : ∀ a b, 2 * a + b ≥ 9 * m) :
  m = 4 := 
sorry

end max_value_of_m_l137_137856


namespace determine_trajectory_of_center_l137_137539

noncomputable def trajectory_of_center (C : Point) : Prop :=
  ∃ (F : Point) (p : ℝ), F = (2, 0) ∧ (C ∈ parabola_focus_directrix p F)

theorem determine_trajectory_of_center :
  ∀ (C : Point), trajectory_of_center C → C.y^2 = 8 * C.x :=
by
  -- We are given the conditions:
  -- 1. The moving circle passes through a fixed point F(2, 0).
  -- 2. The moving circle is tangent to the line x = -2.
  -- We now need to prove that the trajectory of the center C satisfies the equation:
  -- y^2 = 8x
  sorry

end determine_trajectory_of_center_l137_137539


namespace leak_empties_cistern_in_24_hours_l137_137769

theorem leak_empties_cistern_in_24_hours :
  (fill_without_leak_time: ℝ) → (fill_with_leak_time: ℝ) →
  fill_without_leak_time = 6 →
  fill_with_leak_time = 8 →
  invert (1 / fill_without_leak_time - 1 / fill_with_leak_time) = 24 :=
by
  intros fill_without_leak_time fill_with_leak_time h_fill_without_leak h_fill_with_leak
  rw [h_fill_without_leak, h_fill_with_leak]
  sorry

end leak_empties_cistern_in_24_hours_l137_137769


namespace find_number_of_rational_roots_l137_137775

noncomputable def number_of_rational_roots (p : Polynomial ℤ) : ℕ := sorry

theorem find_number_of_rational_roots :
  ∀ (b4 b3 b2 b1 : ℤ), (number_of_rational_roots (8 * Polynomial.X ^ 5 
      + b4 * Polynomial.X ^ 4 
      + b3 * Polynomial.X ^ 3 
      + b2 * Polynomial.X ^ 2 
      + b1 * Polynomial.X 
      + 24) = 28) := 
by
  intro b4 b3 b2 b1
  sorry

end find_number_of_rational_roots_l137_137775


namespace number_of_divisors_8_factorial_l137_137913

open Nat

theorem number_of_divisors_8_factorial :
  let n := 8!
  let factorization := [(2, 7), (3, 2), (5, 1), (7, 1)]
  let numberOfDivisors := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  n = 2^7 * 3^2 * 5^1 * 7^1 ->
  n.factors.count = 4 ->
  numberOfDivisors = 96 :=
by
  sorry

end number_of_divisors_8_factorial_l137_137913


namespace arithmetic_sequence_problem_l137_137619

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) -- condition for arithmetic sequence
  (h_condition : a 3 + a 5 + a 7 + a 9 + a 11 = 100) : 
  3 * a 9 - a 13 = 40 :=
sorry

end arithmetic_sequence_problem_l137_137619


namespace calculate_fraction_l137_137817

def at_op (a b : ℤ) := a * b - b^2
def hash_op (a b : ℤ) := a + b - 2 * a * b^2

theorem calculate_fraction : (8 @ 3) / (8 # 3) = -15 / 133 := by
  sorry

end calculate_fraction_l137_137817


namespace axis_of_symmetry_l137_137549

-- Define points and the parabola equation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 2 5
def B := Point.mk 4 5

def parabola (b c : ℝ) (p : Point) : Prop :=
  p.y = 2 * p.x^2 + b * p.x + c

theorem axis_of_symmetry (b c : ℝ) (hA : parabola b c A) (hB : parabola b c B) : ∃ x_axis : ℝ, x_axis = 3 :=
by
  -- Proof to be provided
  sorry

end axis_of_symmetry_l137_137549


namespace task2_probability_l137_137745

variable (P_Task1 : ℝ) (P_Task1_and_not_Task2 : ℝ) (P_Task2 : ℝ)

-- Define the conditions
def conditions := (P_Task1 = 3 / 8) ∧ (P_Task1_and_not_Task2 = 0.15)

-- Define the independence condition
def independent (P_Task1 P_Task2 : ℝ) : Prop :=
  ∀ (P_not_Task2 : ℝ), P_Task1_and_not_Task2 = P_Task1 * (1 - P_Task2)

-- The theorem statement
theorem task2_probability :
  conditions P_Task1 P_Task1_and_not_Task2 →
  independent P_Task1 P_Task2 →
  P_Task2 = 0.6 :=
by
  sorry

end task2_probability_l137_137745


namespace solve_exact_integer_solutions_l137_137846

theorem solve_exact_integer_solutions (a : ℝ) :
  (∃ x1 x2 x3 : ℤ, ∀ x : ℤ, ||x - 2| - 1| = a ↔ x = x1 ∨ x = x2 ∨ x = x3) → a = 1 :=
by
  sorry

end solve_exact_integer_solutions_l137_137846


namespace triangle_area_ratio_l137_137218

noncomputable def area_ratio (p q r : ℝ) : ℝ :=
  1 - r / (r * p + r + 1) - p / (p * q + p + 1) - q / (q * r + q + 1)

theorem triangle_area_ratio (p q r : ℝ) (D E F P Q R : Type)
  (BD_DC : Type := p)
  (CE_EA : Type := q)
  (AF_FB : Type := r)
  (CF_BE_PQ : Prop := true)
  (BE_AD_R : Prop := true) :
  area_ratio p q r = 1 - r / (r * p + r + 1) - p / (p * q + p + 1) - q / (q * r + q + 1) :=
by
  sorry

end triangle_area_ratio_l137_137218


namespace distinct_three_digit_numbers_count_l137_137124

theorem distinct_three_digit_numbers_count :
  ∃ (numbers : Finset (Fin 1000)), (∀ n ∈ numbers, (n / 100) < 5 ∧ (n / 10 % 10) < 5 ∧ (n % 10) < 5 ∧ 
  (n / 100) ≠ (n / 10 % 10) ∧ (n / 100) ≠ (n % 10) ∧ (n / 10 % 10) ≠ (n % 10)) ∧ numbers.card = 60 := 
sorry

end distinct_three_digit_numbers_count_l137_137124


namespace log_five_one_over_sqrt_five_l137_137823

theorem log_five_one_over_sqrt_five : log 5 (1 / real.sqrt 5) = -1 / 2 := sorry

end log_five_one_over_sqrt_five_l137_137823


namespace sin_cos_15_deg_l137_137803

theorem sin_cos_15_deg : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 :=
by
  -- known values and formulas
  have double_angle : ∀ θ, Real.sin (2 * θ) = 2 * Real.sin θ * Real.cos θ := Real.sin_double
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by norm_num

  -- substitution and calculation
  rw [← Real.sin_double] at *,
  norm_num,
  sorry  -- Proof steps are omitted, as instructed.

end sin_cos_15_deg_l137_137803


namespace race_winner_l137_137943

-- Definitions and conditions based on the problem statement
def tortoise_speed : ℕ := 5  -- Tortoise speed in meters per minute
def hare_speed_1 : ℕ := 20  -- Hare initial speed in meters per minute
def hare_time_1 : ℕ := 3  -- Hare initial running time in minutes
def hare_speed_2 : ℕ := 10  -- Hare speed when going back in meters per minute
def hare_time_2 : ℕ := 2  -- Hare back running time in minutes
def hare_sleep_time : ℕ := 5  -- Hare sleeping time in minutes
def hare_speed_3 : ℕ := 25  -- Hare final speed in meters per minute
def track_length : ℕ := 130  -- Total length of the race track in meters

-- The problem statement
theorem race_winner :
  track_length / tortoise_speed > hare_time_1 + hare_time_2 + hare_sleep_time + (track_length - (hare_speed_1 * hare_time_1 - hare_speed_2 * hare_time_2)) / hare_speed_3 :=
sorry

end race_winner_l137_137943


namespace simplify_expression_l137_137311

theorem simplify_expression :
  (∃ (a b c d : ℝ), 
   a = 14 * Real.sqrt 2 ∧ 
   b = 12 * Real.sqrt 2 ∧ 
   c = 8 * Real.sqrt 2 ∧ 
   d = 12 * Real.sqrt 2 ∧ 
   ((a / b) + (c / d) = 11 / 6)) :=
by 
  use 14 * Real.sqrt 2, 12 * Real.sqrt 2, 8 * Real.sqrt 2, 12 * Real.sqrt 2
  simp
  sorry

end simplify_expression_l137_137311


namespace geometric_sequence_ratio_l137_137941

variable {a : ℕ → ℝ} -- Define the geometric sequence {a_n}

-- Conditions: The sequence is geometric with positive terms
variable (q : ℝ) (hq : q > 0) (hgeo : ∀ n, a (n + 1) = q * a n)

-- Additional condition: a2, 1/2 a3, and a1 form an arithmetic sequence
variable (hseq : a 1 - (1 / 2) * a 2 = (1 / 2) * a 2 - a 0)

theorem geometric_sequence_ratio :
  (a 3 + a 4) / (a 2 + a 3) = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_sequence_ratio_l137_137941


namespace evaluate_y_l137_137995

def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt b)

theorem evaluate_y 
  (h1: ∀ a b, bowtie a b = a + Real.sqrt (b + Real.sqrt b))
  (h2: bowtie 5 y = 15) :
  y = 90 :=
sorry

end evaluate_y_l137_137995


namespace roshini_spent_on_sweets_l137_137267

variable (initial_amount friends_amount total_friends_amount sweets_amount : ℝ)

noncomputable def Roshini_conditions (initial_amount friends_amount total_friends_amount sweets_amount : ℝ) :=
  initial_amount = 10.50 ∧ friends_amount = 6.80 ∧ sweets_amount = 3.70 ∧ 2 * 3.40 = 6.80

theorem roshini_spent_on_sweets :
  ∀ (initial_amount friends_amount total_friends_amount sweets_amount : ℝ),
    Roshini_conditions initial_amount friends_amount total_friends_amount sweets_amount →
    initial_amount - friends_amount = sweets_amount :=
by
  intros initial_amount friends_amount total_friends_amount sweets_amount h
  cases h
  sorry

end roshini_spent_on_sweets_l137_137267


namespace tan_alpha_value_l137_137567

-- Definition: α is an obtuse angle
def is_obtuse (α : ℝ) : Prop := α > π / 2 ∧ α < π

-- Definition: α satisfies the given equation
def satisfies_equation (α : ℝ) : Prop := 
  (sin α - 3 * cos α) / (cos α - sin α) = tan (2 * α)

-- Theorem: If α is an obtuse angle and satisfies the given equation, then tan α = 2 - √7
theorem tan_alpha_value (α : ℝ) (h1 : is_obtuse α) (h2 : satisfies_equation α) : 
  tan α = 2 - Real.sqrt 7 := 
by
  sorry

end tan_alpha_value_l137_137567


namespace problem1_problem2_l137_137120

-- Define the universal set and sets A and B
def U := ℝ
def A : set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Problem 1: When m = 1
def B1 : set ℝ := {x | 1 < x ∧ x < 4}

-- Proving the statements in Problem 1
theorem problem1 : (A ∪ B1 = {x | 0 < x ∧ x < 4}) ∧ (A ∩ (U \ B1) = {x | 0 < x ∧ x ≤ 1}) :=
by sorry

-- Problem 2: General m
def B (m : ℝ) : set ℝ := {x | m < x ∧ x < m + 3}

-- Proving the implication in Problem 2
theorem problem2 (m : ℝ) : (A ⊆ B m) → (m ∈ set.Ioo (-1 : ℝ) 0 ∨ m = 0) :=
by sorry

end problem1_problem2_l137_137120


namespace concert_singers_l137_137605

def Singer := ℕ
def Order := List Singer

variable (wishes : Singer → Set Singer)
variable (good_orders : Finset (List Singer))

def is_good_order (ord : Order) : Prop :=
  ∀ s ∈ List.toFinset ord,
    ∀ t ∈ wishes s, List.indexOf t ord < List.indexOf s ord

def number_of_good_orders : ℕ :=
  (good_orders.filter is_good_order).card

theorem concert_singers (wishes : Singer → Set Singer) :
  (number_of_good_orders wishes) = 2010 := by
  sorry

end concert_singers_l137_137605


namespace pyramid_volume_max_l137_137325

theorem pyramid_volume_max (
    (MN NK SM SN : ℝ) (h1 : MN = 5) (h2 : NK = 2) (h3 : SM = 3) (h4 : SN = 4) :
    ∃ (SK SL V : ℝ), SK = 2 * Real.sqrt 5 ∧ SL = Real.sqrt 13 ∧ V = 8 :=
begin
  sorry
end

end pyramid_volume_max_l137_137325


namespace max_lateral_surface_area_l137_137441

theorem max_lateral_surface_area : ∀ (x y : ℝ), 6 * x + 3 * y = 12 → (3 * x * y) ≤ 6 :=
by
  intros x y h
  have xy_le_2 : x * y ≤ 2 :=
    by
      sorry
  have max_area_6 : 3 * x * y ≤ 6 :=
    by
      sorry
  exact max_area_6

end max_lateral_surface_area_l137_137441


namespace log_geometric_sequence_sum_l137_137873

-- Definitions based on the given conditions.
variable {a : ℕ → ℝ}
variable {n : ℕ}
variable {r : ℝ}

-- Geometric sequence condition: a_n = a_1 * r^(n - 1)
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n, a n = a 1 * r ^ (n - 1)

-- Positive terms condition
axiom pos_terms (a : ℕ → ℝ) : ∀ n, a n > 0

-- Given condition a_4 * a_5 * a_6 = 8
axiom product_cond (a : ℕ → ℝ) : a 4 * a 5 * a 6 = 8

-- The proof problem statement
theorem log_geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h_geom : is_geometric_sequence a r)
  (h_pos : pos_terms a) (h_prod : product_cond a) : 
  (Finset.range 9).sum (λ i, Real.log (a (i + 1))) / Real.log 2 = 9 :=
by
  sorry

end log_geometric_sequence_sum_l137_137873


namespace train_crosses_man_in_time_l137_137017

noncomputable def time_to_cross (L : ℝ) (v_train : ℝ) (v_man : ℝ) : ℝ :=
  let relative_speed_kmph := v_train + v_man
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  L / relative_speed_mps

theorem train_crosses_man_in_time :
  time_to_cross 500 174.98560115190784 5 ≈ 10.0008 := 
sorry

end train_crosses_man_in_time_l137_137017


namespace original_employees_l137_137439

theorem original_employees (x : ℝ) (h₁ : x * 0.85 = 195) : x ≈ 229 :=
by
  sorry

end original_employees_l137_137439


namespace det_B_eq_neg13_l137_137988

variables (x y : ℝ)

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := ![![x, 2], ![-3, y]]
noncomputable def B_inv : Matrix (Fin 2) (Fin 2) ℝ := (1 / (x * y + 6)) • ![![y, -2], ![3, x]]

theorem det_B_eq_neg13 
  (hB_inv : B x y - B_inv x y = ![![1, 0], ![0, 1]]) :
  det (B x y) = -13 := 
sorry

end det_B_eq_neg13_l137_137988


namespace obtuse_triangle_of_sinA_cosB_tanC_lt_zero_l137_137095

theorem obtuse_triangle_of_sinA_cosB_tanC_lt_zero
  (A B C : ℝ) (hA : A > 0) (h: sin A * cos B * tan C < 0) :
  ∃ (angle : ℝ), angle > π / 2 ∧ (angle = B ∨ angle = C) :=
sorry

end obtuse_triangle_of_sinA_cosB_tanC_lt_zero_l137_137095


namespace total_rainfall_November_l137_137187

def rain_first_15_days : ℕ := 4

def days_first_15 : ℕ := 15

def rain_last_15_days : ℕ := 2 * rain_first_15_days

def days_last_15 : ℕ := 15

def total_rainfall : ℕ := 
  (rain_first_15_days * days_first_15) + (rain_last_15_days * days_last_15)

theorem total_rainfall_November : total_rainfall = 180 := by
  sorry

end total_rainfall_November_l137_137187


namespace product_divisible_by_C1_to_Cn_l137_137555

noncomputable def C (s : ℕ) := s * (s + 1)

theorem product_divisible_by_C1_to_Cn 
  (k m n : ℕ) 
  (h1 : 0 < k) 
  (h2 : 0 < m) 
  (h3 : 0 < n) 
  (h4 : Nat.Prime (m+k+1))
  (h5 : m+k+1 > n+1): 
  ∏ i in Finset.range n, (C (m + i + 1) - C k) % ∏ i in Finset.range n, C (i + 1) = 0 := 
sorry

end product_divisible_by_C1_to_Cn_l137_137555


namespace max_rectangle_area_l137_137903

noncomputable def curve_parametric_equation (θ : ℝ) :
    ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem max_rectangle_area :
  ∃ (θ : ℝ), (θ ∈ Set.Icc 0 (2 * Real.pi)) ∧
  ∀ (x y : ℝ), (x, y) = curve_parametric_equation θ →
  |(1 + 2 * Real.cos θ) * (1 + 2 * Real.sin θ)| = 3 + 2 * Real.sqrt 2 :=
sorry

end max_rectangle_area_l137_137903


namespace smallest_a_l137_137251

theorem smallest_a (a b : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) :
  (∀ (x : ℤ), cos (a * x + b) = sin (36 * x)) → a = 36 :=
by
  intro h
  sorry

end smallest_a_l137_137251


namespace at_least_one_half_l137_137353

theorem at_least_one_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
by
  sorry

end at_least_one_half_l137_137353


namespace total_students_in_school_l137_137944

variable (students : ℕ)
variable (h1 : 0.45 * students = blue_shirt_students)
variable (h2 : 0.23 * students = red_shirt_students)
variable (h3 : 0.15 * students = green_shirt_students)
variable (h4 : 0.17 * students = 136)

theorem total_students_in_school (students : ℕ) 
  (h1 : 0.45 * students + 0.23 * students + 0.15 * students + 0.17 * students = students)
  (h2 : 0.17 * students = 136) : students = 800 :=
by sorry

end total_students_in_school_l137_137944


namespace range_of_g_eq_l137_137259

noncomputable def f : ℝ → ℝ := sorry
def a := 2
def A : set ℝ := { x | x ≠ 0 ∧ f x ≤ 1 }
def g (x : ℝ) : ℝ := f (2 * x) * f (1 / x)

theorem range_of_g_eq : (∀ x ∈ A, f (1 / x) = a * f x - x) → f 1 = 1 → 
  set.range g = set.Icc (25 / 18) (3 / 2) :=
by
  intros h1 h2
  sorry

end range_of_g_eq_l137_137259


namespace regression_line_equation_l137_137101

theorem regression_line_equation (slope : ℝ) (point : ℝ × ℝ) :
  slope = 1.23 ∧ point = (4, 5) → ∃ b : ℝ, ∀ x : ℝ, (point.snd = slope * point.fst + b) := 
by 
  intro h
  cases h with h_slope h_point
  existsi 0.08
  sorry

end regression_line_equation_l137_137101


namespace number_of_valid_4x4_tables_l137_137758

-- Define what it means for a 4x4 table to be valid
def is_valid_table (table : Fin 4 × Fin 4 → Fin 2) : Prop :=
  (∀ i : Fin 4, Finset.card {j | table (i, j) = 1} = 2) ∧
  (∀ j : Fin 4, Finset.card {i | table (i, j) = 1} = 2)

-- Define the set of all valid 4x4 tables
def valid_tables := {table : Fin 4 × Fin 4 → Fin 2 | is_valid_table table}

-- State the main theorem
theorem number_of_valid_4x4_tables : Finset.card valid_tables = 90 :=
sorry

end number_of_valid_4x4_tables_l137_137758


namespace smallest_abundant_number_not_multiple_of_4_is_18_l137_137518

def is_abundant (n : ℕ) : Prop :=
  ∑ (d : ℕ) in (Finset.filter (λ d, d ∣ n ∧ d < n) (Finset.range n)), d > n

def not_multiple_of_4 (n : ℕ) : Prop :=
  ¬ (4 ∣ n)

def smallest_abundant_not_multiple_of_4 : ℕ :=
  Nat.find (exists.intro 18 (and.intro (is_abundant 18) (not_multiple_of_4 18)))

theorem smallest_abundant_number_not_multiple_of_4_is_18 :
  smallest_abundant_not_multiple_of_4 = 18 :=
sorry

end smallest_abundant_number_not_multiple_of_4_is_18_l137_137518


namespace area_of_circle_is_correct_l137_137738

-- Define the diameter of the circle
def diameter : ℝ := 10

-- Define the radius as half of the diameter
def radius : ℝ := diameter / 2

-- Define the expected area of the circle
def expected_area : ℝ := 25 * Real.pi

-- Prove that the area of a circle with given diameter equals the expected area
theorem area_of_circle_is_correct : Real.pi * radius ^ 2 = expected_area := 
sorry

end area_of_circle_is_correct_l137_137738


namespace probability_A_not_selected_B_selected_probability_at_least_one_not_selected_l137_137422

def math_group : Type := {S1, S2, S3}
def science_group : Type := {Z1, Z2, Z3}
def humanities_group : Type := {R1, R2, R3}

def total_events : set (math_group × science_group × humanities_group) := 
  set.univ

def events_A_not_selected_B_selected : set (math_group × science_group × humanities_group) := 
  { e | e.1 ≠ S1 ∧ e.2 = Z2 }

def events_both_A_B_selected : set (math_group × science_group × humanities_group) := 
  { e | e.1 = S1 ∧ e.2 = Z2 }

theorem probability_A_not_selected_B_selected : 
  (card events_A_not_selected_B_selected / card total_events = 2 / 9) := 
sorry

theorem probability_at_least_one_not_selected : 
  (1 - card events_both_A_B_selected / card total_events = 8 / 9) := 
sorry

end probability_A_not_selected_B_selected_probability_at_least_one_not_selected_l137_137422


namespace binary_to_decimal_correct_l137_137702

open Nat

theorem binary_to_decimal_correct : 
  binary_to_nat 111011001001 = 3785 := 
by sorry

end binary_to_decimal_correct_l137_137702


namespace largest_element_lg11_l137_137593

variable (x y : ℝ)
variable (A : Set ℝ)  (B : Set ℝ)

-- Conditions
def condition1 : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)) := sorry
def condition2 : B = Set.insert 0 (Set.insert 1 ∅) := sorry
def condition3 : B ⊆ A := sorry

-- Statement
theorem largest_element_lg11 (x y : ℝ)

  (Aeq : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)))
  (Beq : B = Set.insert 0 (Set.insert 1 ∅))
  (subset : B ⊆ A) :
  ∃ M ∈ A, ∀ a ∈ A, a ≤ M ∧ M = Real.log 11 :=
sorry

end largest_element_lg11_l137_137593


namespace natasha_average_speed_l137_137667

theorem natasha_average_speed :
  ∀ (time_up time_down : ℝ) (speed_up : ℝ),
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (2 * (time_up * speed_up) / (time_up + time_down) = 3) :=
by
  intros time_up time_down speed_up h_time_up h_time_down h_speed_up
  rw [h_time_up, h_time_down, h_speed_up]
  sorry

end natasha_average_speed_l137_137667


namespace probability_of_condition_l137_137243

noncomputable def roots_of_unity (n : ℕ) : set ℂ :=
  {z : ℂ | z ^ n = 1}

theorem probability_of_condition :
  let n := 2021
      roots := roots_of_unity n
      sqrt_condition := sqrt(3 + sqrt 5) in
  v ∈ roots ∧ w ∈ roots ∧ v ≠ w →
  v = 1 →
  (∃ k : ℕ, k < n ∧ sqrt_condition ≤ abs (1 + exp (2 * real.pi * complex.I * k / n))) →
  (308 / 2020 = 0.15247524752475247) :=
by
  sorry

end probability_of_condition_l137_137243


namespace car_lease_annual_cost_l137_137379

/-- Tom decides to lease a car. He drives 50 miles on four specific days a week,
and 100 miles on the other three days. He pays $0.1 per mile and a weekly fee of $100.
Prove that the total annual cost he has to pay is $7800.
--/
theorem car_lease_annual_cost :
  let weekly_miles := 4 * 50 + 3 * 100
      weekly_mileage_cost := weekly_miles * 0.1
      weekly_total_cost := weekly_mileage_cost + 100
      annual_cost := weekly_total_cost * 52
  in annual_cost = 7800 :=
by
  let weekly_miles := 4 * 50 + 3 * 100
  let weekly_mileage_cost := weekly_miles * 0.1
  let weekly_total_cost := weekly_mileage_cost + 100
  let annual_cost := weekly_total_cost * 52
  sorry

end car_lease_annual_cost_l137_137379


namespace coeff_x2003_of_expanded_expr_l137_137397

theorem coeff_x2003_of_expanded_expr :
  let expr := (1 + x) * (1 + 2 * x^3) * (1 + 4 * x^9) * (1 + 8 * x^27) * (1 + 16 * x^81) * (1 + 32 * x^243) * (1 + 64 * x^729)
  is the coefficient of x^2003 in expr^2 equal to 2^30 :=
sorry

end coeff_x2003_of_expanded_expr_l137_137397


namespace fractional_eq_range_m_l137_137885

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l137_137885


namespace distinct_nat_numbers_inequality_l137_137092

theorem distinct_nat_numbers_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ)
  (h_distinct: ∀ i j : ℕ, i ≠ j → a[i] ≠ a[j]) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 :=
sorry

end distinct_nat_numbers_inequality_l137_137092


namespace sin_expression_simplification_cosine_sum_simplification_l137_137688

-- Part 1
theorem sin_expression_simplification (A : ℝ) :
  (sin (5 * A)) - 5 * (sin (3 * A)) + 10 * (sin A) = 16 * (sin A)^5 :=
sorry

-- Part 2
theorem cosine_sum_simplification (n : ℕ) (x : ℝ) (hn : n > 0):
  (∑ k in finset.range (n + 1), (nat.choose n k) * (cos (k * x))) = 
  2^n * (cos (x / 2))^n * (cos (n * x / 2)) :=
sorry

end sin_expression_simplification_cosine_sum_simplification_l137_137688


namespace parabola_tangents_intersection_y_coord_l137_137245

theorem parabola_tangents_intersection_y_coord
  (a b : ℝ)
  (ha : A = (a, a^2 + 1))
  (hb : B = (b, b^2 + 1))
  (tangent_perpendicular : ∀ t1 t2 : ℝ, t1 * t2 = -1):
  ∃ y : ℝ, y = 3 / 4 :=
by
  sorry

end parabola_tangents_intersection_y_coord_l137_137245


namespace seq_an_general_formula_sum_bn_formula_l137_137955

-- a_n sequence general formula problem
theorem seq_an_general_formula (a_n : ℕ → ℝ) (p : ℝ) (h1 : ∀ n, a_n > 0) (h2 : p > 0) (h3 : p ≠ 1)
  (h4 : ∀ n, (p - 1) * (∑ i in Finset.range (n + 1), a_n i) = p^2 - a_n n) :
  ∀ n, a_n n = (1 / p)^(n - 1) * p :=
sorry

-- sum of sequence b_n problem
theorem sum_bn_formula (a_n b_n : ℕ → ℝ) (p : ℝ) (h1 : p = 1/2) (h2 : ∀ n, b_1 * a_n + (∑ i in Finset.range (n + 1), b_n i * a_n (n - i)) = 2^n - (1 / 2) * n - 1)
  (h3 : ∀ n, a_n n = 2^(n - 1)) :
  ∀ n, (∑ i in Finset.range (n + 1), b_n i) = n * (n + 1) / 2 :=
sorry

end seq_an_general_formula_sum_bn_formula_l137_137955


namespace r_daily_earnings_l137_137404

-- Given conditions as definitions
def daily_earnings (P Q R : ℕ) : Prop :=
(P + Q + R) * 9 = 1800 ∧ (P + R) * 5 = 600 ∧ (Q + R) * 7 = 910

-- Theorem statement corresponding to the problem
theorem r_daily_earnings : ∃ R : ℕ, ∀ P Q : ℕ, daily_earnings P Q R → R = 50 :=
by sorry

end r_daily_earnings_l137_137404


namespace range_of_m_l137_137893

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → (m ≤ 2 ∧ m ≠ -2) :=
begin
  sorry
end

end range_of_m_l137_137893


namespace find_ak_bk_l137_137844

theorem find_ak_bk (n : ℕ) (hn : n ≥ 2) 
  (a b : ℕ → ℝ)
  (h_a : ∀ k, 1 ≤ k ∧ k ≤ n → 0 ≤ a k ∧ a k ≤ 1)
  (h_b : ∀ k, 1 ≤ k ∧ k ≤ n → 1 ≤ b k)
  (h_sum_ak_bk : ∑ k in finset.range (n + 1), a k + b k = 2 * n)
  (h_sum_ak2_bk2 : ∑ k in finset.range (n + 1), a k ^ 2 + b k ^ 2 = n ^ 2 + 3 * n) :
  (∀ k, 1 ≤ k ∧ k ≤ n → a k = 0) ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ n - 1 → b k = 1) ∧ 
  b n = n + 1 := by
  sorry

end find_ak_bk_l137_137844


namespace seq_growth_exists_q_l137_137527

theorem seq_growth_exists_q (f : ℕ → ℝ) (n : ℕ → ℝ) (C : ℝ) (h1 : ∃ q > 1, ∀ i ≥ 1, n i ≥ q^(i - 1)) :
  (∀ i : ℕ, 0 < n i) ∧ (∀ i, f (n i) < C / (n i)^2) ∧ 
  (∀ n ∈ ℕ, ∀ m ∈ ℤ, |sqrt 2 * n - m| > 1 / (4 * n)) 
  → ∃ q > 1, ∀ i ≥ 1, n i ≥ q^(i - 1) := by
  sorry

end seq_growth_exists_q_l137_137527


namespace storage_unit_capacity_l137_137783

theorem storage_unit_capacity :
    let shelf1 := 5 * 8,
        shelf2 := 4 * 10,
        shelf3 := 3 * 12 in
    shelf1 + shelf2 + shelf3 = 116 :=
by
  sorry

end storage_unit_capacity_l137_137783


namespace percentage_of_adults_is_40_l137_137419

variables (A C : ℕ)

-- Given conditions as definitions
def total_members := 120
def more_children_than_adults := 24
def percentage_of_adults (A : ℕ) := (A.toFloat / total_members.toFloat) * 100

-- Lean 4 statement to prove the percentage of adults
theorem percentage_of_adults_is_40 (h1 : A + C = 120)
                                   (h2 : C = A + 24) :
  percentage_of_adults A = 40 :=
by
  sorry

end percentage_of_adults_is_40_l137_137419


namespace inequality1_inequality2_inequality3_l137_137762

-- Part 1
theorem inequality1 (x y : ℝ) (p q : ℝ) 
    (hx : 0 < x) (hy : 0 < y) 
    (hpq_range : 1 ≤ p) (hpq_inv_sum : 1 / p + 1 / q = 1) : 
    x * y ≤ x^p / p + y^q / q :=
sorry

-- Part 2
theorem inequality2 (n : ℕ) (x y : Fin n → ℝ) (p q : ℝ) 
    (hx : ∀ i, 0 < x i) (hy : ∀ i, 0 < y i)
    (hpq_range : 1 ≤ p) (hpq_inv_sum : 1 / p + 1 / q = 1) : 
    (∑ i, x i * y i) ≤ (∑ i, (x i)^p)^(1 / p) * (∑ i, (y i)^q)^(1 / q) :=
sorry

-- Part 3
theorem inequality3 (x y : ℝ) : 
    (1 + x^2 * y + x^4 * y^2)^3 ≤ (1 + x^3 + x^6)^2 * (1 + y^3 + y^6) :=
sorry

end inequality1_inequality2_inequality3_l137_137762


namespace disproving_statement_l137_137763

theorem disproving_statement (a b c : ℝ) (h₁ : a = -1) (h₂ : b = -2) (h₃ : c = -3) : ¬ (a > b > c → a + b > c) := sorry

end disproving_statement_l137_137763


namespace find_k_if_parallel_l137_137908

-- Define the vectors a and b
variables {k : ℝ}

def a : ℝ × ℝ := (1, k)
def b : ℝ × ℝ := (9, k - 6)

-- Define what it means for vectors to be parallel
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- The theorem statement
theorem find_k_if_parallel (h_parallel : are_parallel a b) : k = - (3 / 4) :=
  sorry

end find_k_if_parallel_l137_137908


namespace tangent_line_at_1_neg3_l137_137872

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then log (-x) + 3 * x else log x - 3 * x

theorem tangent_line_at_1_neg3 :
  (∀ x : ℝ, f x = f (-x)) →
  f 1 = -3 →
  ∃ m b : ℝ, (∀ y x : ℝ, y = f x → y + 3 = -2 * (x - 1)) ∧ (∀ y x : ℝ, y = m * x + b → y = -2 * x - 1) :=
by
  intros even_func_ext fx_eq_neg3
  existsi -2
  existsi -1
  split
  intros y x hx
  sorry
  intro y
  intros
  sorry

end tangent_line_at_1_neg3_l137_137872


namespace max_elements_in_T_l137_137646

def T := { x : ℕ | 1 ≤ x ∧ x ≤ 100 }

theorem max_elements_in_T : 
  ∃ (T : set ℕ), T ⊆ {1..100} ∧ 
  (∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a + b) % 5 ≠ 0) ∧ 
  (∀ U, U ⊆ {1..100} ∧ (∀ (a b : ℕ), a ∈ U → b ∈ U → a ≠ b → (a + b) % 5 ≠ 0) → U.card ≤ 41) :=
begin
  sorry
end

end max_elements_in_T_l137_137646


namespace fencing_required_l137_137402

def L : ℝ := 20
def A : ℝ := 650
def W : ℝ := A / L
def F : ℝ := 2 * W + L

theorem fencing_required : F = 85 :=
by
  simp [L, A, W, F]
  sorry

end fencing_required_l137_137402


namespace complex_real_im_sum_zero_l137_137385

open Complex

theorem complex_real_im_sum_zero (z : ℂ) 
  (h1 : conj z * (1 + I) = 2 * I) :
  z.re + z.im = 0 :=
sorry

end complex_real_im_sum_zero_l137_137385


namespace proof_of_multiplication_and_subtraction_l137_137395

theorem proof_of_multiplication_and_subtraction :
  let product := 0.000218 * 5432000 in
  let result := product - 500 in
  abs (result - 580) < abs (result - 520)
  ∧ abs (result - 580) < abs (result - 600)
  ∧ abs (result - 580) < abs (result - 650) := 
by
  sorry

end proof_of_multiplication_and_subtraction_l137_137395


namespace p_subset_q_range_of_a_l137_137081

variable (a x : ℝ)

def A := { x | |3 * x - 4| > 2 }
def B := { x | 1 / (x^2 - x - 2) > 0 }
def C := { x | (x - a) * (x - (a + 1)) ≥ 0 }
def complement_R (X: set ℝ) := { x | x ∉ X }

def p := complement_R A
def q := complement_R B
def r := C

theorem p_subset_q : p ⊆ q := sorry

theorem range_of_a : (r ⊆ p) → (r ≠ p) → (a ≥ 2 ∨ a ≤ -1 / 3) := sorry

end p_subset_q_range_of_a_l137_137081


namespace trapezoid_exists_in_marked_vertices_l137_137190

-- Definitions
noncomputable def regular_n_gon (n : ℕ) := sorry -- Definition of a regular n-gon
noncomputable def marked_vertices (polygon : Type) (k : ℕ) := sorry -- Definition of k marked vertices in a polygon

-- Given parameters
def n := 1981 -- Number of sides of the polygon
def k := 64 -- Number of marked vertices

-- The problem statement
theorem trapezoid_exists_in_marked_vertices 
    (polygon : regular_n_gon n) 
    (marked : marked_vertices polygon k) :
    ∃ (a b c d : marked), 
      (∃ (segments : (a, b), (c, d) ∈ marked), 
        -- defines the existence of parallel segments forming a trapezoid
        ((a ≠ b) ∧ (c ≠ d)) ∧ (is_parallel ⟨a, b⟩ ⟨c, d⟩)) 
 := sorry

end trapezoid_exists_in_marked_vertices_l137_137190


namespace ferns_have_1260_leaves_l137_137228

def num_ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def leaves_per_frond : ℕ := 30
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem ferns_have_1260_leaves : total_leaves = 1260 :=
by 
  -- proof goes here
  sorry

end ferns_have_1260_leaves_l137_137228


namespace rainfall_november_is_180_l137_137188

-- Defining the conditions
def daily_rainfall_first_15_days := 4 -- inches per day
def days_in_first_period := 15
def total_days_in_november := 30
def multiplier_for_second_period := 2

-- Calculation based on the problem's conditions
def total_rainfall_november := 
  (daily_rainfall_first_15_days * days_in_first_period) + 
  (multiplier_for_second_period * daily_rainfall_first_15_days * (total_days_in_november - days_in_first_period))

-- Prove that the total rainfall in November is 180 inches
theorem rainfall_november_is_180 : total_rainfall_november = 180 :=
by
  -- Proof steps (to be filled in)
  sorry

end rainfall_november_is_180_l137_137188


namespace distinct_three_digit_numbers_count_l137_137126

theorem distinct_three_digit_numbers_count :
  ∃ (numbers : Finset (Fin 1000)), (∀ n ∈ numbers, (n / 100) < 5 ∧ (n / 10 % 10) < 5 ∧ (n % 10) < 5 ∧ 
  (n / 100) ≠ (n / 10 % 10) ∧ (n / 100) ≠ (n % 10) ∧ (n / 10 % 10) ≠ (n % 10)) ∧ numbers.card = 60 := 
sorry

end distinct_three_digit_numbers_count_l137_137126


namespace projection_magnitude_correct_l137_137909

-- Define vectors a and b
def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

-- Function to compute dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Function to compute magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Function to compute the magnitude of the projection of vector a on vector b
def projection_magnitude (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

-- Statement of the theorem to be proved
theorem projection_magnitude_correct :
  projection_magnitude a b = Real.sqrt 5 :=
by
  sorry

end projection_magnitude_correct_l137_137909


namespace min_entries_to_unique_sums_l137_137042

-- Define a structure for a 4x4 matrix
structure Matrix4x4 where
  a b c d e f g h i j k l m n o p : ℝ

-- Define the conditions of the problem
def rowSum (M : Matrix4x4) : Fin 4 → ℝ
  | 0 => M.a + M.b + M.c + M.d
  | 1 => M.e + M.f + M.g + M.h
  | 2 => M.i + M.j + M.k + M.l
  | 3 => M.m + M.n + M.o + M.p

def colSum (M : Matrix4x4) : Fin 4 → ℝ
  | 0 => M.a + M.e + M.i + M.m
  | 1 => M.b + M.f + M.j + M.n
  | 2 => M.c + M.g + M.k + M.o
  | 3 => M.d + M.h + M.l + M.p

def rowCondition (M : Matrix4x4) := rowSum M 0 = rowSum M 1 ∧ rowSum M 1 = rowSum M 2
def colCondition (M : Matrix4x4) := colSum M 0 = colSum M 1 ∧ colSum M 1 = colSum M 2 ∧ colSum M 2 = colSum M 3

-- The statement that we need to prove
theorem min_entries_to_unique_sums (M : Matrix4x4) (h₁ : rowCondition M) (h₂ : colCondition M) : ∃ (N : Matrix4x4), (∃ k, rowSum N ≠ rowSum N ∨ colSum N ≠ colSum N) ∧ k = 4 := 
  sorry

end min_entries_to_unique_sums_l137_137042


namespace sum_bn_converges_l137_137030

noncomputable def v2 (n : ℕ) : ℕ :=
if h : n = 0 then 0 else
  let v := λ m, if even m then Nat.get (Nat.mkEven m) + 1 else 0 in
  Nat.strongRecOn n (λ n ih, if n = 1 then 0 else v n + ih (n / 2) (Nat.div_lt_self (Ne.zero_lt h) (by norm_num)))

def a (n : ℕ) : ℝ := real.exp (-v2 n)
def b (n : ℕ) : ℝ := ∏ i in Finset.range n, a (i + 1)

theorem sum_bn_converges : Summable (λ n, b n) := sorry

end sum_bn_converges_l137_137030


namespace day_of_week_proof_l137_137924

def day_of_week_17th_2003 := "Wednesday"
def day_of_week_305th_2003 := "Thursday"

theorem day_of_week_proof (d17 : day_of_week_17th_2003 = "Wednesday") : day_of_week_305th_2003 = "Thursday" := 
sorry

end day_of_week_proof_l137_137924


namespace parallelogram_area_given_diagonals_l137_137991

open Real

variables (r s : ℝ^3)

def is_unit_vector (v : ℝ^3) : Prop :=
  ‖v‖ = 1

def angle_between_vectors (v w : ℝ^3) : ℝ :=
  acos ((dot_product v w) / (‖v‖ * ‖w‖))

noncomputable def parallelogram_area (d1 d2 : ℝ^3) : ℝ :=
  ‖d1 × d2‖ / 2

theorem parallelogram_area_given_diagonals
  (hr : is_unit_vector r)
  (hs : is_unit_vector s)
  (h_angle : angle_between_vectors r s = π / 4) :
  parallelogram_area (r + 3 • s) (3 • r + s) = 9 * sqrt 2 / 4 :=
sorry

end parallelogram_area_given_diagonals_l137_137991


namespace find_d_l137_137590

theorem find_d (d : ℚ) : (x + 5) | (d * x^3 + 17 * x^2 - 2 * d * x + 50) → d = 95 / 23 :=
by
  sorry

end find_d_l137_137590


namespace largest_base_eight_l137_137388

-- Definition of the main problem's parameters and the result of the calculations
def eleven_pow_five := 161051
def two_pow_five := 32

-- Conditions need to calculate the sum of the digits of eleven_pow_five in base b
def sum_base_digits (n : Nat) (b : Nat) : Nat :=
  n.toDigits b |>.foldr (· + ·) 0

-- Problem statement
theorem largest_base_eight :
  ∀ (b : Nat), b ≤ 8 → (sum_base_digits eleven_pow_five b ≠ two_pow_five) :=
by
  intro b hb
  sorry

end largest_base_eight_l137_137388


namespace distinct_three_digit_numbers_count_l137_137129

theorem distinct_three_digit_numbers_count : 
  (∃ digits : Finset ℕ, 
    digits = {1, 2, 3, 4, 5} ∧ 
    ∀ n ∈ digits, n ∈ {1, 2, 3, 4, 5} ∧ 
    (digits.card = 5)) → 
  card (finset.univ.image (λ p : Finset (ℕ × ℕ × ℕ), 
      {x | x.fst ∈ digits ∧ x.snd ∈ digits ∧ x.snd.snd ∈ digits ∧ x.fst ≠ x.snd ∧ x.snd ≠ x.snd.snd ∧ x.fst ≠ x.snd.snd})) = 60 := by
  sorry

end distinct_three_digit_numbers_count_l137_137129


namespace continuous_stick_shortening_l137_137058

noncomputable def polynomial (x : ℝ) : ℝ := x^3 - x^2 - x - 1

theorem continuous_stick_shortening :
  ∃ t > 1, t < 2 ∧ polynomial t = 0 → ∀ (a b c : ℝ), (a = t^3 ∧ b = t^2 ∧ c = t) → 
  (a = t^3 ∧ b = t^2 ∧ c = t) → a + b > c → ∃! (l : List ℝ), ∀ e ∈ l, e = t^3 ∨ e = t^2 ∨ e = t := 
begin
  sorry
end

end continuous_stick_shortening_l137_137058


namespace number_of_valid_sets_l137_137506

theorem number_of_valid_sets :
  ∃ A : set (set ℕ),
    (∀ a ∈ A, a.card = 2 ∧
              (∀ x ∈ a, ∀ y ∈ a, x ≠ y ∧ x ≥ 7 ∧ x ≤ 136 ∧ y ≥ 7 ∧ y ≤ 136 ∧ x ≠ y → 
              (x = 30 ∨ y = 30) ∧ 22 * y ≤ 100 * x ∧ 22 * x ≤ 100 * y)) ∧
    A.card = 129 :=
by sorry

end number_of_valid_sets_l137_137506


namespace dot_product_of_vectors_l137_137563

theorem dot_product_of_vectors
  (x y : ℝ)
  (h_circle : (x - 2)^2 + y^2 = 1)
  (P : ℝ × ℝ)
  (h_P : P = (3, 4))
  (A B : ℝ × ℝ)
  (h_intersects : line_through P (A) ∩ { p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 1 } = {A, B}) :
  let PA := ((A.1 - P.1), (A.2 - P.2))
  let PB := ((B.1 - P.1), (B.2 - P.2))
  in PA.1 * PB.1 + PA.2 * PB.2 = 16 :=
sorry

end dot_product_of_vectors_l137_137563


namespace shark_teeth_multiple_l137_137785

/-
Given:
1. A tiger shark has 180 teeth.
2. A hammerhead shark has 1/6 the number of teeth that a tiger shark has.
3. The great white shark has 420 teeth.

Prove:
The multiple of the sum of the teeth of a tiger shark and a hammerhead shark that a great white shark has is 2.
-/

theorem shark_teeth_multiple (tiger_shark_teeth : ℕ) (hammerhead_shark_ratio : ℚ) (great_white_teeth : ℕ) :
  tiger_shark_teeth = 180 →
  hammerhead_shark_ratio = 1/6 →
  great_white_teeth = 420 →
  great_white_teeth = 2 * (tiger_shark_teeth + (hammerhead_shark_ratio * tiger_shark_teeth).to_nat) :=
by
  intros h1 h2 h3
  sorry

end shark_teeth_multiple_l137_137785


namespace max_elements_no_sum_divisible_by_5_l137_137642

open Finset Function

def no_sum_divisible_by {T : Finset ℕ} (s : ℕ) : Prop :=
∀ {a b : ℕ}, a ∈ T → b ∈ T → a ≠ b → (a + b) % s ≠ 0

theorem max_elements_no_sum_divisible_by_5 :
  ∀ T : Finset ℕ, T ⊆ range 1 101 → no_sum_divisible_by 5 T → T.card ≤ 60 :=
by
  intros
  sorry

end max_elements_no_sum_divisible_by_5_l137_137642


namespace find_p_plus_q_l137_137761

noncomputable def smallest_nonzero_x (p q : ℕ) : ℚ :=
  -Real.sqrt p / q

theorem find_p_plus_q :
  ∃ p q : ℕ, 
    smallest_nonzero_x p q = -Real.sqrt 2 / 2 ∧
    p ≠ 0 ∧
    q ≠ 0 ∧
    ¬ (∃ k : ℕ, k > 1 ∧ k ^ 2 ∣ p) ∧
    p + q = 4 := 
by
  sorry

end find_p_plus_q_l137_137761


namespace cos_theta_on_line_y_eq_2x_l137_137105

theorem cos_theta_on_line_y_eq_2x :
  ∃ θ : ℝ, cos θ = (↑(5).sqrt) / 5 ∨ cos θ = - (↑(5).sqrt) / 5 ∧
  (¬ (θ = 0) ∧ ∃ p : ℝ × ℝ, (p = (1, 2) ∨ p = (-1, -2)) ∧ p.2 = 2 * p.1) := 
sorry

end cos_theta_on_line_y_eq_2x_l137_137105


namespace unique_function_l137_137061

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (-f x - f y) = 1 - x - y

theorem unique_function :
  ∀ f : ℤ → ℤ, (functional_equation f) → (∀ x : ℤ, f x = x - 1) :=
by
  intros f h
  sorry

end unique_function_l137_137061


namespace problem_1_problem_2_l137_137086

def function_properties (f : ℝ → ℝ) (f' : ℝ → ℝ) (I : set ℝ) (c1 c2 : ℝ) : Prop :=
  (∀ x ∈ I, 0 < f'(x) ∧ f'(x) < 2 ∧ f'(x) ≠ 1) ∧
  f c1 = c1 ∧
  f c2 = 2 * c2 ∧
  (∀ (a b : ℝ), a ∈ I → b ∈ I → a < b → ∃ x ∈ set.Ioo a b, f b - f a = (b - a) * f' x)

variable {f : ℝ → ℝ} {f' : ℝ → ℝ} {I : set ℝ} {c1 c2 : ℝ}

theorem problem_1 (h : function_properties f f' I c1 c2) :
  ∀ x : ℝ, f x = x → x = c1 :=
sorry

theorem problem_2 (h : function_properties f f' I c1 c2) :
  ∀ x : ℝ, x > c2 → f x < 2 * x :=
sorry

end problem_1_problem_2_l137_137086


namespace combine_sqrt_27_with_sqrt_12_l137_137025

theorem combine_sqrt_27_with_sqrt_12 : 
  let sqrt_12 := sqrt 12 
  let sqrt_27 := sqrt 27 
  (sqrt_27 = sqrt (9 * 3)) ∧ 
  (sqrt_12 = sqrt (4 * 3)) → 
  sqrt_27 = 3 * sqrt (3) ∧ sqrt_12 = 2 * sqrt (3) → 
  ∃ x : ℝ, sqrt_12 + sqrt_27 = x * sqrt (3) := 
by 
  sorry

end combine_sqrt_27_with_sqrt_12_l137_137025


namespace Jo_and_Kate_sum_difference_l137_137630

theorem Jo_and_Kate_sum_difference :
  let S_Jo := (150 * 151) / 2
  let S_Kate := 15 * 600
  | S_Kate - S_Jo | = 2325 := by
  sorry

end Jo_and_Kate_sum_difference_l137_137630


namespace solution_l137_137589

noncomputable def proof_problem (x y : ℤ) : Prop :=
  (3 ^ x * 4 ^ y = 19683) ∧ (x - y = 9) → (x = 9)

theorem solution (x y : ℤ) : proof_problem x y := sorry

end solution_l137_137589


namespace rectangular_prism_parallel_edges_l137_137133

theorem rectangular_prism_parallel_edges (length width height : ℝ) (h1 : length ≠ width) (h2 : width ≠ height) (h3 : length ≠ height) : 
    number_of_parallel_edge_pairs length width height = 12 :=
by
  sorry

def number_of_parallel_edge_pairs (length width height : ℝ) : ℕ :=
  -- definition based on the math logic being applied in the problem
  4 + 4 + 4  -- 4 pairs from each dimension

end rectangular_prism_parallel_edges_l137_137133


namespace find_b_in_triangle_l137_137154

open Real

theorem find_b_in_triangle (a A B : ℝ) (ha : a = 3) (hA : A = π / 3) (hB : B = π / 4) : 
  let b := a * (sin B) / (sin A) in b = sqrt 6 :=
by
  let b := a * (sin B) / (sin A)
  sorry

end find_b_in_triangle_l137_137154


namespace inequality_holds_equality_condition_l137_137232

variables {x y z : ℝ}
-- Assuming positive real numbers and the given condition
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom h : x * y + y * z + z * x = x + y + z

theorem inequality_holds : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) ≤ 1 :=
by
  sorry

theorem equality_condition : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end inequality_holds_equality_condition_l137_137232


namespace sun_city_population_correct_l137_137315

noncomputable def willowdale_population : Nat := 2000
noncomputable def roseville_population : Nat := 3 * willowdale_population - 500
noncomputable def sun_city_population : Nat := 2 * roseville_population + 1000

theorem sun_city_population_correct : sun_city_population = 12000 := by
  sorry

end sun_city_population_correct_l137_137315


namespace sum_repeating_decimals_l137_137066

theorem sum_repeating_decimals (h₁ : ∀ n, (1 ≤ n ∧ n ≤ 200)) :
  (∀ n, (n+1).prime_factors ⊈ {2, 5} → repeating_decimal (n / (n + 1))) →
  sum_of_repeating_decimals (1, 200) = 19127 :=
by
  -- definitions to start working on the problem
  sorry

end sum_repeating_decimals_l137_137066


namespace fractional_eq_range_m_l137_137883

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l137_137883


namespace distinct_three_digit_numbers_count_l137_137125

theorem distinct_three_digit_numbers_count :
  ∃ (numbers : Finset (Fin 1000)), (∀ n ∈ numbers, (n / 100) < 5 ∧ (n / 10 % 10) < 5 ∧ (n % 10) < 5 ∧ 
  (n / 100) ≠ (n / 10 % 10) ∧ (n / 100) ≠ (n % 10) ∧ (n / 10 % 10) ≠ (n % 10)) ∧ numbers.card = 60 := 
sorry

end distinct_three_digit_numbers_count_l137_137125


namespace find_area_of_triangle_l137_137867
  
def point (α : Type) := α × α

variables (A B P₁ P₂ P₃ : point ℝ)
variables (l₁ l₂ l₃ : point ℝ → Prop)

def minimize_sum_of_squared_distances (p: point ℝ) :=
  let x1 := p.1, y1 := p.2,
      a1 := A.1, a2 := A.2, 
      b1 := B.1, b2 := B.2 in
  (a1 - x1)^2 + (a2 - y1)^2 + (b1 - x1)^2 + (b2 - y1)^2

def line_l1 (p : point ℝ) : Prop := p.1 = 0
def line_l2 (p : point ℝ) : Prop := p.2 = 0
def line_l3 (p : point ℝ) : Prop := p.1 + 3*p.2 - 1 = 0

theorem find_area_of_triangle : 
  ∃ (P₁ P₂ P₃ : point ℝ), (
  line_l1 P₁ ∧
  line_l2 P₂ ∧
  line_l3 P₃ ∧
  (minimize_sum_of_squared_distances A P₁ + minimize_sum_of_squared_distances B P₁) ≤
  min (minimize_sum_of_squared_distances A P₂ + minimize_sum_of_squared_distances B P₂)
  (minimize_sum_of_squared_distances A P₃ + minimize_sum_of_squared_distances B P₃) ∧
  let
  P₁ := (0, 3),
  P₂ := (2, 0),
  P₃ := (1, 0) in 
  1 / 2 * abs ((1 - 0) * (3 - 0)) = 3 / 2
  ) := sorry

end find_area_of_triangle_l137_137867


namespace two_pies_differ_l137_137179

theorem two_pies_differ (F_A F_C B_A B_C : Bool) :
  (F_A ∨ F_C ∨ B_A ∨ B_C) →
  (F_A ∧ F_C ∧ B_A ∧ B_C) ∧ 
  (∀ a b, (a ≠ b) → (a.filling ≠ b.filling ∧ a.preparation ≠ b.preparation)) :=
by
  intros H1 H2
  sorry

end two_pies_differ_l137_137179


namespace soccer_games_total_l137_137754

variable (wins losses ties total_games : ℕ)

theorem soccer_games_total
    (h1 : losses = 9)
    (h2 : 4 * wins + 3 * losses + ties = 8 * total_games) :
    total_games = 24 :=
by
  sorry

end soccer_games_total_l137_137754


namespace equation_of_line_line_passes_through_fixed_point_perpendicular_line_through_fixed_point_l137_137410

-- Problem (1)
theorem equation_of_line (x y : ℝ) (h1 : (x, y) = (-(1/2:ℝ), -(3/2:ℝ))) (h2 : ∃ (a b : ℝ), a = 3 * b ∧ (x * b + y * a = a * b)) :
  (3 * x - y = 0) ∨ (3 * x + y + 3 = 0) :=
sorry

-- Problem (2)(i)
theorem line_passes_through_fixed_point (x y λ : ℝ) (h1 : 3 * x + λ * y - 2 + 2 * λ * x + 4 * y + 2 * λ = 0) :
  ∃ (a b : ℝ), (a, b) = (-2, 2) :=
sorry

-- Problem (2)(ii)
theorem perpendicular_line_through_fixed_point (x y : ℝ) (h1: (x, y) = (-2, 2)) (h2 : ∃ k : ℝ, k = -(2/3) ∧ (3 * x - 2 * y + 4 = 0)) :
  (2 * x + 3 * y - 2 = 0) :=
sorry

end equation_of_line_line_passes_through_fixed_point_perpendicular_line_through_fixed_point_l137_137410


namespace main_theorem_l137_137611

noncomputable theory

-- Define the basic geometric entities and structures
variables {A B C : Type} [plane_geometry : geometric_plane]

-- Define various circles and their tangency properties
variables (ω Ω : circle ABC) (R : ℝ)
variables (ω_A Ω_A : circle ABC)
variables (P_A Q_A P_B Q_B P_C Q_C : point ABC)

-- Assuming required tangency properties
axiom ωA_tangent_to_Ω_at_A : tangent (ω_A) (Ω) A
axiom ωA_tangent_to_ω : tangent (ω_A) (ω)
axiom ΩA_tangent_to_Ω_at_A : tangent (Ω_A) (Ω) A
axiom ΩA_tangent_to_ω : tangent (Ω_A) (ω)

-- Equality conditions for an equilateral triangle
axiom triangle_equilateral : equilateral_triangle ABC

-- Defining the distances between centers
def PAQA_dist (A B : point ABC) := distance P_A Q_A
def PBQB_dist (A B : point ABC) := distance P_B Q_B
def PCQC_dist (A B : point ABC) := distance P_C Q_C

-- The final statement to be proven
theorem main_theorem (h : acute_angled_triangle ABC) :
  (8 * PAQA_dist P_A Q_A * PBQB_dist P_B Q_B * PCQC_dist P_C Q_C ≤ R^3) :=
sorry

end main_theorem_l137_137611


namespace other_root_l137_137151

theorem other_root (k : ℝ) : 
  5 * (2:ℝ)^2 + k * (2:ℝ) - 8 = 0 → 
  ∃ q : ℝ, 5 * q^2 + k * q - 8 = 0 ∧ q ≠ 2 ∧ q = -4/5 :=
by {
  sorry
}

end other_root_l137_137151


namespace projection_correct_l137_137512

noncomputable def projection (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_vw := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let dot_ww := w.1 * w.1 + w.2 * w.2 + w.3 * w.3
  let scalar := dot_vw / dot_ww
  (scalar * w.1, scalar * w.2, scalar * w.3)

theorem projection_correct :
  projection (4, -1, 3) (3, 1, 2) = (51 / 14, 17 / 14, 17 / 7) := 
by
  -- The proof would go here.
  sorry

end projection_correct_l137_137512


namespace ellipse_properties_l137_137087

noncomputable theory

open Real

theorem ellipse_properties
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
  (P F1 F2 : ℝ × ℝ)
  (on_ellipse_P : P ∈ {p : ℝ × ℝ | (p.1 / a) ^ 2 + (p.2 / b) ^ 2 = 1})
  (equilateral_triangle : (dist P F1 + dist F1 F2 + dist F2 P) = 6)
  (triangle_equilateral : dist P F1 = dist F1 F2 ∧ dist F1 F2 = dist F2 P ∧ dist F2 P = dist P F1)
  (A B C D : ℝ × ℝ)
  (AC_BD_intersect_F1 : ∃ F1, F1 = (A + C) / 2 ∧ F1 = (B + D) / 2)
  (dot_product_eq_zero : (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0)
  (slope_AC : C.2 - A.2 = √3 * (C.1 - A.1)) :
  (a = 2) ∧ (b = √3) ∧
  (AC_length : ℝ) (BD_length : ℝ) (area_ABCD : ℝ) :
  AC_length = ∥ A - C ∥ ∧ BD_length = ∥ B - D ∥ ∧ area_ABCD = (1/2 * AC_length * BD_length) :=
by
  sorry

end ellipse_properties_l137_137087


namespace repeating_decimal_representation_l137_137406

noncomputable def ABAC := 4649

theorem repeating_decimal_representation :
  ∃ d : ℕ, (d = 0.0002151) ∧ (1 / (ABAC: ℚ)) = (0.0002151 : ℚ) :=
by
  -- The proof is omitted here, insert the appropriate proof steps.
  sorry

end repeating_decimal_representation_l137_137406


namespace water_formed_l137_137832

theorem water_formed (CaOH2 CO2 CaCO3 H2O : Nat) 
  (h_balanced : ∀ n, n * CaOH2 + n * CO2 = n * CaCO3 + n * H2O)
  (h_initial : CaOH2 = 2 ∧ CO2 = 2) : 
  H2O = 2 :=
by
  sorry

end water_formed_l137_137832


namespace sin_cos_alpha_l137_137552

theorem sin_cos_alpha (α : ℝ) (h0 : sin α * cos α = -12 / 25) (h1 : 0 < α) (h2 : α < π) : 
  sin α - cos α = 7 / 5 :=
sorry

end sin_cos_alpha_l137_137552


namespace fractions_not_equivalent_for_all_x_l137_137660

-- Define the fractions
def frac1 (x : ℝ) := (x + 3) / (x - 5)
def frac2 : ℝ := 3 / (-5)

-- State the proof problem
theorem fractions_not_equivalent_for_all_x (x : ℝ) (h : x ≠ 5) : 
  (frac1 x = frac2 ↔ x = 0) :=
sorry

end fractions_not_equivalent_for_all_x_l137_137660


namespace cyclic_iff_eq_dist_l137_137946

noncomputable def isCyclicQuadrilateral (A B C D P : Point) : Prop :=
  -- Definition of cyclic quadrilateral
  cyclic A B C D

theorem cyclic_iff_eq_dist
  (A B C D P : Point)
  (h1 : convex_quadrilateral A B C D)
  (h2 : ¬is_angle_bisector BD (angle ABC))
  (h3 : ¬is_angle_bisector BD (angle CDA))
  (h4 : interior_point A B C D P)
  (h5 : angle PBC = angle ABD)
  (h6 : angle PDC = angle BDA) :
  isCyclicQuadrilateral A B C D ↔ distance A P = distance C P :=
sorry

end cyclic_iff_eq_dist_l137_137946


namespace opposite_of_three_l137_137792

theorem opposite_of_three (A B C D : ℤ) 
  (hA : A = -(-3)) 
  (hB : B = |3|) 
  (hC : C = |-3|) 
  (hD : D = +(-3)) :
  -3 = D :=
by {
  sorry
}

end opposite_of_three_l137_137792


namespace rainfall_november_is_180_l137_137189

-- Defining the conditions
def daily_rainfall_first_15_days := 4 -- inches per day
def days_in_first_period := 15
def total_days_in_november := 30
def multiplier_for_second_period := 2

-- Calculation based on the problem's conditions
def total_rainfall_november := 
  (daily_rainfall_first_15_days * days_in_first_period) + 
  (multiplier_for_second_period * daily_rainfall_first_15_days * (total_days_in_november - days_in_first_period))

-- Prove that the total rainfall in November is 180 inches
theorem rainfall_november_is_180 : total_rainfall_november = 180 :=
by
  -- Proof steps (to be filled in)
  sorry

end rainfall_november_is_180_l137_137189


namespace phil_books_remaining_pages_l137_137278

/-- We define the initial number of books and the number of pages per book. -/
def initial_books : Nat := 10
def pages_per_book : Nat := 100
def lost_books : Nat := 2

/-- The goal is to find the total number of pages Phil has left after losing 2 books. -/
theorem phil_books_remaining_pages : (initial_books - lost_books) * pages_per_book = 800 := by 
  -- The proof will go here
  sorry

end phil_books_remaining_pages_l137_137278


namespace ellipse_proof_l137_137868

-- Defining the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Focus of the ellipse
def focus_R (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Distance ratio condition
def distance_ratio (P : ℝ × ℝ) (F : ℝ × ℝ) (m : ℝ) : Prop :=
  let d_PF := real.sqrt ((P.1 - F.1)^2 + P.2^2)
  let d_Pl := abs (P.1 - m)
  (d_PF / d_Pl) = 1/2

-- Proving the equation of the line l and the fixed point
theorem ellipse_proof :
  (∀ F, focus_R F → ∃ m, ∀ P, ellipse P.1 P.2 → distance_ratio P F m → m = 4) ∧
  (∃ (fixed_point : ℝ × ℝ), fixed_point = (1, 0)) :=
by
  -- First part: proving the equation of line l
  use (λ F hF => 4), 
  intro F hF,
  rw [focus_R] at hF,
  use (λ P hP => 4),
  intro P hP,
  rw [distance_ratio],
  sorry,
  -- Second part: proving the fixed point
  exists (1, 0),
  trivial
  

end ellipse_proof_l137_137868


namespace find_m_real_find_m_imaginary_l137_137523

-- Define the real part condition
def real_part_condition (m : ℝ) : Prop :=
  m^2 - 3 * m - 4 = 0

-- Define the imaginary part condition
def imaginary_part_condition (m : ℝ) : Prop :=
  m^2 - 2 * m - 3 = 0 ∧ m^2 - 3 * m - 4 ≠ 0

-- Theorem for the first part
theorem find_m_real : ∀ (m : ℝ), (real_part_condition m) → (m = 4 ∨ m = -1) :=
by sorry

-- Theorem for the second part
theorem find_m_imaginary : ∀ (m : ℝ), (imaginary_part_condition m) → (m = 3) :=
by sorry

end find_m_real_find_m_imaginary_l137_137523


namespace monotonic_intervals_range_of_a_min_value_of_c_l137_137898

noncomputable def f (a c x : ℝ) : ℝ :=
  a * Real.log x + (x - c) * abs (x - c)

-- 1. Monotonic intervals
theorem monotonic_intervals (a c : ℝ) (ha : a = -3 / 4) (hc : c = 1 / 4) :
  ((∀ x, 0 < x ∧ x < 3 / 4 → f a c x > f a c (x - 1)) ∧ (∀ x, 3 / 4 < x → f a c x > f a c (x - 1))) :=
sorry

-- 2. Range of values for a
theorem range_of_a (a c : ℝ) (hc : c = a / 2 + 1) (h : ∀ x > c, f a c x ≥ 1 / 4) :
  -2 < a ∧ a ≤ -1 :=
sorry

-- 3. Minimum value of c
theorem min_value_of_c (a c x1 x2 : ℝ) (hx1 : x1 = Real.sqrt (-a / 2)) (hx2 : x2 = c)
  (h_tangents_perpendicular : f a c x1 * f a c x2 = -1) :
  c = 3 * Real.sqrt 3 / 2 :=
sorry

end monotonic_intervals_range_of_a_min_value_of_c_l137_137898


namespace inequality_solution_nonempty_l137_137934

theorem inequality_solution_nonempty (a : ℝ) :
  (∃ x : ℝ, x ^ 2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end inequality_solution_nonempty_l137_137934


namespace arithmetic_sequence_product_l137_137358

theorem arithmetic_sequence_product (a : ℕ → ℚ) (d : ℚ) (a7 : a 7 = 20) (diff : d = 2) : 
  let a1 := a 1,
      a2 := a 2
  in a1 * a2 = 80 :=
by
  sorry

end arithmetic_sequence_product_l137_137358


namespace max_elements_subset_no_sum_mod_5_divisible_l137_137639

theorem max_elements_subset_no_sum_mod_5_divisible (T : Set ℕ)
  (hT₁ : ∀ x ∈ T, x ∈ Finset.range 101)
  (hT₂ : ∀ x y ∈ T, x ≠ y → ¬ (x + y) % 5 = 0) : T.card ≤ 40 := sorry

end max_elements_subset_no_sum_mod_5_divisible_l137_137639


namespace perimeter_less_than_twice_area_l137_137440

noncomputable def probability_perimeter_less_than_twice_area : ℚ :=
  let favorable_outcomes := {s | s > 2 ∧ s ∈ ({1, 2, 3, 4, 5, 6} : set ℕ)} in
  (favorable_outcomes.to_finset.card : ℚ) / 6

theorem perimeter_less_than_twice_area : 
  probability_perimeter_less_than_twice_area = 2 / 3 :=
by
  sorry

end perimeter_less_than_twice_area_l137_137440


namespace password_factoring_theorem_l137_137622

-- For part (1)
def factor_x3_minus_xy2_passwords (x y : ℕ) : Prop :=
  (x = 21) ∧ (y = 7) →
  (factor x^3 - x * y^2 = (x - y) * x * (x + y)) →
  (213871 = 21 * 10^5 + 14 * 10^2 + 28 ∨
   212814 = 21 * 10^5 + 28 * 10^2 + 14 ∨
   142128 = 14 * 10^5 + 21 * 10^2 + 28)

-- For part (2)
def right_triangle_password (x y : ℕ) : Prop :=
  (x + y + 13 = 30) ∧ (x^2 + y^2 = 169) →
  (factor x^3 * y + x * y^3 = 60 * 169) →
  (601690 = 6 * 10^5 + 0 * 10^4 + 1 * 10^3 + 6 * 10^2 + 9)

-- For part (3)
def m_n_values_from_password (m n : ℕ) : Prop :=
  (factor x^3 + (m - 3 * n) * x^2 - n * x - 21 = (x - 3) * (x + 1) * (x + 7)) ∧
  (242834 = 24 * 10^5 + 28 * 10^2 + 34) →
  (m = 56) ∧ (n = 17)

-- The main theorem
theorem password_factoring_theorem (x y m n : ℕ) : 
    factor_x3_minus_xy2_passwords x y ∧
    right_triangle_password x y ∧
    m_n_values_from_password m n :=
by
  sorry

end password_factoring_theorem_l137_137622


namespace evaluate_expression_l137_137055

theorem evaluate_expression (x : ℝ) (h : x < -1) : 
  sqrt ((x + 2) / (1 - (x - 2) / (x + 1))) = sqrt ((x^2 + 3 * x + 2) / 3) :=
by
  sorry

end evaluate_expression_l137_137055


namespace population_increase_20th_century_l137_137405

theorem population_increase_20th_century (P : ℕ) :
  let population_mid_century := 3 * P
  let population_end_century := 12 * P
  (population_end_century - P) / P * 100 = 1100 :=
by
  sorry

end population_increase_20th_century_l137_137405


namespace triangle_equilateral_side_length_l137_137375

theorem triangle_equilateral_side_length (α β : ℝ) (a b : ℝ) (hα : α = 60) (hβ : β = 60) (ha : a = 1) (hb : b = 1) : ∃ c : ℝ, c = 1 :=
by
  use 1
  sorry

end triangle_equilateral_side_length_l137_137375


namespace fly_speed_calc_l137_137730

def speed_of_fly (relative_speed time_hours total_distance : ℝ) : ℝ :=
  total_distance / time_hours

theorem fly_speed_calc : 
  ∀ (v₁ v₂ : ℝ) (d : ℝ) (total_dist : ℝ),
  v₁ = 10 → v₂ = 10 → d = 50 → total_dist = 37.5 →
  speed_of_fly (v₁ + v₂) (d / (v₁ + v₂)) total_dist = 15 :=
by
  intros
  -- With the conditions provided, we get:
  -- v₁ = 10, v₂ = 10, d = 50, total_dist = 37.5
  -- Need to show that: speed_of_fly (10 + 10) (50 / (10 + 10)) 37.5 = 15
  sorry

end fly_speed_calc_l137_137730


namespace exists_m_for_division_l137_137845

theorem exists_m_for_division (n : ℕ) (h : 0 < n) : ∃ m : ℕ, n ∣ (2016 ^ m + m) := by
  sorry

end exists_m_for_division_l137_137845


namespace central_symmetry_of_triangles_l137_137551

variable (A B C P M_A P_A Q_A R_A R_B R_C H_A H_B H_C : Type)
  [InnerProductSpace ℝ (A × B × C)]
  [Midpoints : ∀ (X Y Z : Type), Prop]
  [AcuteTriangle : ∀ (Δ : Type), Prop]
  [Reflection : ∀ (X M Y : Type), Prop]
  [Projection : ∀ (X Y K : Type), Prop]

noncomputable theory

def isReflection (X M Y : Type) := sorry
def isMidpoint (X Y Z : Type) := sorry
def isProjection (X Y K : Type) := sorry
def centrallySymmetric (Δ1 Δ2 : Type) := sorry

theorem central_symmetry_of_triangles :
  ∀ {A B C P : Type} [AcuteTriangle (A × B × C)]
  (Mid_A : isMidpoint (M_A B C))
  (Reflect_PA : isReflection (P M_A P_A))
  (Reflect_QA : isReflection (P_A BC Q_A))
  (Mid_RA : isMidpoint (A Q_A R_A))
  (Mid_RB : isMidpoint (B Q_B R_B))
  (Mid_RC : isMidpoint (C Q_C R_C))
  (Proj_HA : isProjection (P BC H_A))
  (Proj_HB : isProjection (P AC H_B))
  (Proj_HC : isProjection (P AB H_C)),
  centrallySymmetric (R_A R_B R_C) (H_A H_B H_C) := 
  sorry

end central_symmetry_of_triangles_l137_137551


namespace max_min_values_existence_of_k_l137_137533

variable (θ : ℝ) (k : ℝ)

def a_vector : ℝ × ℝ := (Real.cos (3 * θ / 2), Real.sin (3 * θ / 2))

def b_vector : ℝ × ℝ := (Real.cos (θ / 2), -Real.sin (θ / 2))

def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

def vector_norm (u : ℝ × ℝ) := Real.sqrt (u.1^2 + u.2^2)

def value_expr := dot_product (a_vector θ) (b_vector θ) / vector_norm ( (λ(x y : ℝ), (x + y)) (a_vector θ) (b_vector θ) )

def max_value := 1/2
def min_value := -1/2

theorem max_min_values :
  θ ∈ Set.Icc 0 (Real.pi/3) → 
  (value_expr θ = Real.cos θ - 1/(2 * Real.cos θ)) →
  value_expr θ ∈ Set.Icc min_value max_value :=
sorry

theorem existence_of_k :
  θ ∈ Set.Icc 0 (Real.pi/3) →
  ∃ k, 0 < k ∧ k ≤ 2 + Real.sqrt 3 ∧
  vector_norm (λ(x y : ℝ), (x*k + y)) (a_vector θ) (b_vector θ) = Real.sqrt 3 * vector_norm (λ(x y : ℝ), (x - y*k)) (a_vector θ) (b_vector θ) :=
sorry

end max_min_values_existence_of_k_l137_137533


namespace find_a_and_monotonicity_l137_137717

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x

theorem find_a_and_monotonicity :
  (∃ a : ℝ, (∃ (f' : ℝ → ℝ), f' = λ x, (1 / x) - a ∧ f' 1 = 0) ∧ a = 1) ∧
  (∀ a : ℝ, a = 1 →
    (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (f x 1) x > 0) ∧ 
    (∀ x : ℝ, 1 < x → deriv (f x 1) x < 0)) :=
by {
  sorry
}

end find_a_and_monotonicity_l137_137717


namespace sum_of_incircle_radii_of_cyclic_quadrilateral_l137_137778

open Real

theorem sum_of_incircle_radii_of_cyclic_quadrilateral (r1 r2 r3 r4 R : ℝ)
  (h1 : O1.dist O = (R ^ 2 - 2 * R * r1) ^ (1 / 2))
  (h2 : O2.dist O = (R ^ 2 - 2 * R * r2) ^ (1 / 2))
  (h3 : O3.dist O = (R ^ 2 - 2 * R * r3) ^ (1 / 2))
  (h4 : O4.dist O = (R ^ 2 - 2 * R * r4) ^ (1 / 2))
  (rectangle_property : O1.dist O + O2.dist O = O3.dist O + O4.dist O) :
  r1 + r2 = r3 + r4 :=
by
  sorry

end sum_of_incircle_radii_of_cyclic_quadrilateral_l137_137778


namespace monkey_climbs_60m_in_39_minutes_l137_137592

noncomputable def monkey_climb_time : Nat := 39

theorem monkey_climbs_60m_in_39_minutes :
  (∀ (m t : Nat), (m = 6 * (t / 2 + 1) - 3 * (t / 2) ∧ t % 2 = 0) →
                   (t < 2 * 60/3) → (m < 60))
  → (∀ (m t : Nat), (m = 6 * ((t - 1) / 2 + 1) - 3 * ((t - 1) / 2)) ∧ t % 2 = 1 →
                   (t = 39) ∧ (m ≥ 60)) :=
begin
  sorry
end

end monkey_climbs_60m_in_39_minutes_l137_137592


namespace tree_count_l137_137680

theorem tree_count (A_tree_good : ℕ) (B_tree_good : ℕ) (total_good : ℕ) (percent_A_B : ℕ → Prop) (good_oranges_A : ℕ → Prop) (good_oranges_B : ℕ → Prop) : 
  (percent_A_B(50) ∧ percent_A_B(50)) ∧ (good_oranges_A(6)) ∧ (good_oranges_B(5)) ∧ (total_good(55)) 
  → ∃ T, T = 10 :=
by {
  sorry
}

-- Definitions
def percent_A_B (P : ℕ) : ℕ → Prop 
| 50 := true
| _  := false
    
def good_oranges_A : ℕ := 
  6
     
def good_oranges_B : ℕ := 
  5

def total_good : ℕ := 
  55

end tree_count_l137_137680


namespace aunt_may_milk_leftover_l137_137800

noncomputable def milk_leftover : Real :=
let morning_milk := 5 * 13 + 4 * 0.5 + 10 * 0.25
let evening_milk := 5 * 14 + 4 * 0.6 + 10 * 0.2

let morning_spoiled := morning_milk * 0.1
let cheese_produced := morning_milk * 0.15
let remaining_morning_milk := morning_milk - morning_spoiled - cheese_produced
let ice_cream_sale := remaining_morning_milk * 0.7

let evening_spoiled := evening_milk * 0.05
let remaining_evening_milk := evening_milk - evening_spoiled
let cheese_shop_sale := remaining_evening_milk * 0.8

let leftover_previous_day := 15
let remaining_morning_after_sale := remaining_morning_milk - ice_cream_sale
let remaining_evening_after_sale := remaining_evening_milk - cheese_shop_sale

leftover_previous_day + remaining_morning_after_sale + remaining_evening_after_sale

theorem aunt_may_milk_leftover : 
  milk_leftover = 44.7735 := 
sorry

end aunt_may_milk_leftover_l137_137800


namespace total_books_l137_137816

theorem total_books (d k g : ℕ) 
  (h1 : d = 6) 
  (h2 : k = d / 2) 
  (h3 : g = 5 * (d + k)) : 
  d + k + g = 54 :=
by
  sorry

end total_books_l137_137816


namespace polynomial_solution_l137_137830
-- Import necessary library

-- Define the property to be checked
def polynomial_property (P : Real → Real) : Prop :=
  ∀ a b c : Real, 
    P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

-- The statement that needs to be proven
theorem polynomial_solution (a b : Real) : polynomial_property (λ x => a * x^2 + b * x) := 
by
  sorry

end polynomial_solution_l137_137830


namespace ferns_have_1260_leaves_l137_137227

def num_ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def leaves_per_frond : ℕ := 30
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem ferns_have_1260_leaves : total_leaves = 1260 :=
by 
  -- proof goes here
  sorry

end ferns_have_1260_leaves_l137_137227


namespace distinct_natural_numbers_inequality_l137_137089

theorem distinct_natural_numbers_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ)
  (h_distinct : list.nodup [a₁, a₂, a₃, a₄, a₅, a₆, a₇]) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 :=
by
  sorry

end distinct_natural_numbers_inequality_l137_137089


namespace coefficient_x6_in_expansion_l137_137737

theorem coefficient_x6_in_expansion :
  (∃ c : ℕ, c = 81648 ∧ (3 : ℝ) ^ 6 * c * 2 ^ 2  = c * (3 : ℝ) ^ 6 * 4) :=
sorry

end coefficient_x6_in_expansion_l137_137737


namespace prime_in_A_l137_137229

open Nat

def is_in_A (x : ℕ) : Prop :=
  ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a * b ≠ 0

theorem prime_in_A (p : ℕ) [Fact (Nat.Prime p)] (h : is_in_A (p^2)) : is_in_A p :=
  sorry

end prime_in_A_l137_137229


namespace complex_modulus_addition_l137_137484

open Complex -- Use the Complex namespace, considering we are dealing with complex numbers

theorem complex_modulus_addition : |(3 - 5 * Complex.I)| + |(3 + 5 * Complex.I)| = 2 * Real.sqrt 34 := by
  sorry

end complex_modulus_addition_l137_137484


namespace greendale_points_l137_137296

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l137_137296


namespace west_60plus_population_percentage_l137_137463

def population_data := 
  { under20_NE := 25, under20_MW := 30, under20_S := 40, under20_W := 20,
    age20_59_NE := 45, age20_59_MW := 50, age20_59_S := 55, age20_59_W := 30,
    age60_NE := 10, age60_MW := 15, age60_S := 20, age60_W := 25 }

noncomputable def percentage60plusWest : ℕ :=
  let total60 := population_data.age60_NE + population_data.age60_MW + population_data.age60_S + population_data.age60_W
  let west60 := population_data.age60_W
  (west60 * 100 / total60).round

theorem west_60plus_population_percentage : percentage60plusWest = 36 :=
by
  sorry

end west_60plus_population_percentage_l137_137463


namespace remainder_sum_div_8_l137_137919

theorem remainder_sum_div_8 (n : ℤ) : (((8 - n) + (n + 5)) % 8) = 5 := 
by {
  sorry
}

end remainder_sum_div_8_l137_137919


namespace find_x_l137_137572

variable {x : ℝ}

def A : Set ℝ := {-1, 0}
def B : Set ℝ := {0, 1, x + 2}

theorem find_x (h : A ⊆ B) : x = -3 :=
sorry

end find_x_l137_137572


namespace maximize_product_minimize_product_l137_137826

-- Define the numbers that need to be arranged
def numbers : List ℕ := [2, 4, 6, 8]

-- Prove that 82 * 64 is the maximum product arrangement
theorem maximize_product : ∃ a b c d : ℕ, (a = 8) ∧ (b = 2) ∧ (c = 6) ∧ (d = 4) ∧ 
  (a * 10 + b) * (c * 10 + d) = 5248 :=
by
  existsi 8, 2, 6, 4
  constructor; constructor
  repeat {assumption}
  sorry

-- Prove that 28 * 46 is the minimum product arrangement
theorem minimize_product : ∃ a b c d : ℕ, (a = 2) ∧ (b = 8) ∧ (c = 4) ∧ (d = 6) ∧ 
  (a * 10 + b) * (c * 10 + d) = 1288 :=
by
  existsi 2, 8, 4, 6
  constructor; constructor
  repeat {assumption}
  sorry

end maximize_product_minimize_product_l137_137826


namespace only_nice_number_is_three_l137_137230

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def nice (n : ℕ) : Prop :=
  ∃ (xs ys : ℕ → ℕ), 
    xs 1 = 1 ∧ ys 1 = 3 ∧
    (∀ k, xs (k+1) = P (xs k) ∧ ys (k+1) = Q (ys k) ∨ xs (k+1) = Q (xs k) ∧ ys (k+1) = P (ys k)) ∧
    xs n = ys n

theorem only_nice_number_is_three (n : ℕ) : nice n ↔ n = 3 :=
by
  sorry

end only_nice_number_is_three_l137_137230


namespace T_odd_divisible_by_n_T_even_divisible_by_half_n_l137_137423

def roads (n : ℕ) : Type :=
  { r : set (ℕ × ℕ) // 
    (∀ (a b : ℕ), a ≠ b → (a, b) ∈ r → (b, a) ∉ r)
    ∧ (∀ (a : ℕ), (a, (a + 1) % n) ∉ r)
    ∧ (1, n) ∉ r
    ∧ (∀ (a b: ℕ), r.connected a b)
    ∧ r.card = n - 1 }

noncomputable def T (n: ℕ) : ℕ :=
  { r : roads n // true }.card

theorem T_odd_divisible_by_n {n : ℕ} (h_odd: n % 2 = 1) : T n % n = 0 :=
  sorry

theorem T_even_divisible_by_half_n {n : ℕ} (h_even: n % 2 = 0) : T n % (n / 2) = 0 :=
  sorry

end T_odd_divisible_by_n_T_even_divisible_by_half_n_l137_137423


namespace cube_divided_into_5_tetrahedra_l137_137220

theorem cube_divided_into_5_tetrahedra : ∃ (T : set (set ℝ^3)), (∀ t ∈ T, t ≠ ∅ ∧ convex_hull t = t ∧ nonempty t ∧ (∃ A B C D : ℝ^3, t = { x | ∃ α β γ δ, x = α * A + β * B + γ * C + δ * D ∧ α + β + γ + δ = 1 ∧ ∀ i, α i ≥ 0 })) ∧ (⋃₀ T = cube) ∧ (¬ (∃ (U V ∈ T), U ≠ V ∧ U ∩ V ≠ ∅)) ∧ (|T| = 5) :=
sorry

end cube_divided_into_5_tetrahedra_l137_137220


namespace find_the_added_number_l137_137743

theorem find_the_added_number (n : ℤ) : (1 + n) / (3 + n) = 3 / 4 → n = 5 :=
  sorry

end find_the_added_number_l137_137743


namespace lana_total_pages_l137_137969

theorem lana_total_pages (lana_initial_pages : ℕ) (duane_total_pages : ℕ) :
  lana_initial_pages = 8 ∧ duane_total_pages = 42 →
  (lana_initial_pages + duane_total_pages / 2) = 29 :=
by
  sorry

end lana_total_pages_l137_137969


namespace find_f_2010_l137_137981

open Nat

variable (f : ℕ → ℕ)

axiom strictly_increasing : ∀ m n : ℕ, m < n → f m < f n

axiom function_condition : ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_2010 : f 2010 = 3015 := sorry

end find_f_2010_l137_137981


namespace modulus_of_complex_z_l137_137107

open Complex

theorem modulus_of_complex_z (z : ℂ) (h : z * (2 - 3 * I) = 6 + 4 * I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 :=
by
  sorry

end modulus_of_complex_z_l137_137107


namespace one_greater_one_smaller_l137_137962

theorem one_greater_one_smaller (a b : ℝ) (h : ( (1 + a * b) / (a + b) )^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (b > 1 ∧ -1 < a ∧ a < 1) ∨ (a < -1 ∧ -1 < b ∧ b < 1) ∨ (b < -1 ∧ -1 < a ∧ a < 1) :=
by
  sorry

end one_greater_one_smaller_l137_137962


namespace min_value_expression_l137_137654

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + z = 3) (h2 : z = (x + y) / 2) : 
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) = 3 / 2 :=
by sorry

end min_value_expression_l137_137654


namespace conj_z_in_fourth_quadrant_l137_137516

-- Define the complex number and its conjugate
noncomputable def z : ℂ := (5 + 2*complex.I)^2
noncomputable def conj_z : ℂ := complex.conj z

-- Define a predicate to check if a complex number is in the fourth quadrant
def in_fourth_quadrant (c : ℂ) : Prop :=
  c.re > 0 ∧ c.im < 0

-- Statement of the theorem
theorem conj_z_in_fourth_quadrant : in_fourth_quadrant conj_z :=
  sorry

end conj_z_in_fourth_quadrant_l137_137516


namespace blackboard_problem_l137_137253

theorem blackboard_problem (n : ℕ) (h_pos : 0 < n) :
  ∃ x, (∀ (t : ℕ), t < n - 1 → ∃ a b : ℕ, a + b + 2 * (t + 1) = n + 1 ∧ a > 0 ∧ b > 0) → 
  x ≥ 2 ^ ((4 * n ^ 2 - 4) / 3) :=
by
  sorry

end blackboard_problem_l137_137253


namespace running_speed_proof_l137_137666

structure RunWalk :=
  (run_time : ℕ) -- in minutes
  (run_speed : ℕ) -- in mph
  (walk_speed : ℕ) -- in mph
  (walk_time : ℕ) -- in minutes)
  (total_distance : ℕ) -- in miles 

noncomputable def problem_conditions (rw : RunWalk) : Prop :=
  let run_distance := rw.run_speed * (rw.run_time / 60 : ℕ) in
  let walk_distance := rw.walk_speed * (rw.walk_time / 60 : ℕ) in
  run_distance + walk_distance = rw.total_distance

theorem running_speed_proof (rw : RunWalk) (h : problem_conditions rw) : rw.run_speed = 6 :=
  sorry

def example : RunWalk :=
  { run_time := 20,
    run_speed := 6,
    walk_speed := 2,
    walk_time := 30,
    total_distance := 3 }

lemma example_running_speed : problem_conditions example :=
  by
    simp [problem_conditions, example]
    rfl

#eval running_speed_proof example example_running_speed -- expected to be true if theorem is proven

end running_speed_proof_l137_137666


namespace find_parabola_eq_l137_137877

noncomputable def parabola_eq (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  x^2 = 2 * p * y

theorem find_parabola_eq (p b x_A x_B : ℝ)
  (h_angle : Math.atan (√3) = π / 3)
  (h_line_eq : ∀ x y, y = √3 * x + b)
  (h_intersect : parabola_eq p x_A (√3 * x_A + b) ∧ parabola_eq p x_B (√3 * x_B + b))
  (h_sum_roots : x_A + x_B = 3)
  (h_p_pos : 0 < p) :
  ∀ y, parabola_eq (√3 / 2) y -> parabola_eq p y :=
by
  sorry

end find_parabola_eq_l137_137877


namespace find_B_l137_137394

-- Definitions of conditions
def A : ℕ := 5 -- Based on the given solution
def B : ℕ := 2 -- Based on the given solution

-- Single digit constraint
def single_digit (x : ℕ) : Prop := x < 10

-- Encoder function for the original problem condition
def original_problem := (380 + A - 10 * B - 1 = 364)

-- Main theorem statement
theorem find_B : B = 2 ∧ original_problem :=
by {
  have hA : single_digit A, { exact dec_trivial }, sorry,
  have hB : single_digit B, { exact dec_trivial }, sorry
}

end find_B_l137_137394


namespace circles_intersection_l137_137183

theorem circles_intersection (r : ℝ) (Rs : List ℝ) (hR : r = 3) (hRs : Rs.sum = 25) :
  ∃ l : Line, ∃ C ⊆ Rs, C.length ≥ 9 ∧ l.intersects_all C.sublists := 
sorry

end circles_intersection_l137_137183


namespace circle_tangent_to_ellipse_l137_137027

theorem circle_tangent_to_ellipse :
  let a := 7
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  r = 2 → 
  let ellipse_eq := ∀ x y, (x^2 / (a^2)) + (y^2 / (b^2)) = 1
  let circle_eq := ∀ x y, ((x - c)^2 + y^2 = r^2)
  ∃ r, r = 2 ∧ ∀ x y, ellipse_eq x y → circle_eq x y → 
  ∀ x y, x = 2 * Real.sqrt 6 + r → true := sorry

end circle_tangent_to_ellipse_l137_137027


namespace sum_of_roots_of_quadratic_l137_137522

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, x^2 + 2000*x - 2000 = 0 ->
  (∃ x1 x2 : ℝ, (x1 ≠ x2 ∧ x1^2 + 2000*x1 - 2000 = 0 ∧ x2^2 + 2000*x2 - 2000 = 0 ∧ x1 + x2 = -2000)) :=
sorry

end sum_of_roots_of_quadratic_l137_137522


namespace schools_participation_l137_137825

-- Definition of the problem conditions
def school_teams : ℕ := 3

-- Paula's rank p must satisfy this
def total_participants (p : ℕ) : ℕ := 2 * p - 1

-- Predicate indicating the number of participants condition:
def participants_condition (p : ℕ) : Prop := total_participants p ≥ 75

-- Translation of number of participants to number of schools
def number_of_schools (n : ℕ) : ℕ := 3 * n

-- The statement to prove:
theorem schools_participation : ∃ (n p : ℕ), participants_condition p ∧ p = 38 ∧ number_of_schools n = total_participants p ∧ n = 25 := 
by 
  sorry

end schools_participation_l137_137825


namespace find_angles_and_base_of_triangle_l137_137612

def isosceles_triangle (A B C : Point) (h_iso : dist A B = dist B C) : Prop :=
  triangle A B C

def medians_right_angle 
  (A B C M N D : Point)
  (h_median_am : is_median A M)
  (h_median_cn : is_median C N)
  (h_centroid : centroid D)
  (h_perpendicular : orthogonal AM CN) : Prop :=
  true

def quadrilateral_area 
  (N B M D : Point)
  (h_area : area (quadrilateral N B M D) = 4) : Prop :=
  true

theorem find_angles_and_base_of_triangle 
  (A B C M N D : Point)
  (h1 : isosceles_triangle A B C (dist_eq A B B C))
  (h2 : medians_right_angle A B C M N D 
         (is_median A M) (is_median C N) (centroid D) 
         (orthogonal (med AM) (med CN)))
  (h3 : quadrilateral_area N B M D (area_eq 4)) :
  ∃ α β γ : ℝ, 
  α = arctan 3 ∧ γ = arctan 3 ∧ β = π - 2 * arctan 3 :=
sorry

end find_angles_and_base_of_triangle_l137_137612


namespace total_rainfall_November_l137_137186

def rain_first_15_days : ℕ := 4

def days_first_15 : ℕ := 15

def rain_last_15_days : ℕ := 2 * rain_first_15_days

def days_last_15 : ℕ := 15

def total_rainfall : ℕ := 
  (rain_first_15_days * days_first_15) + (rain_last_15_days * days_last_15)

theorem total_rainfall_November : total_rainfall = 180 := by
  sorry

end total_rainfall_November_l137_137186


namespace max_binomial_coeff_l137_137557

theorem max_binomial_coeff (x : ℝ) : 
  ∃ (r n : ℕ) (c : ℝ), 
    (1 + 2 * real.sqrt x) ^ n = Σ i in range (n+1), (real.to_nnreal (n.choose i * 2^i * x^(i / 2))) ∧ 
    r = 5 ∧ 
    n = 7 ∧ 
    c = 672 ∧ 
    T_6 = c * x^(5 / 2) :=
by 
  sorry

end max_binomial_coeff_l137_137557


namespace polynomial_equation_solution_l137_137827

open Polynomial

theorem polynomial_equation_solution (P : ℝ[X])
(h : ∀ (a b c : ℝ), P.eval (a + b - 2 * c) + P.eval (b + c - 2 * a) + P.eval (c + a - 2 * b) = 
      3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)) : 
∃ (a b : ℝ), P = Polynomial.C a * X^2 + Polynomial.C b * X := 
sorry

end polynomial_equation_solution_l137_137827


namespace range_of_m_l137_137534

theorem range_of_m (x m : ℝ) (h₀ : -2 ≤ x ∧ x ≤ 11)
  (h₁ : 1 - 3 * m ≤ x ∧ x ≤ 3 + m)
  (h₂ : ¬ (-2 ≤ x ∧ x ≤ 11) → ¬ (1 - 3 * m ≤ x ∧ x ≤ 3 + m)) :
  m ≥ 8 :=
by
  sorry

end range_of_m_l137_137534


namespace pies_differ_in_both_l137_137163

-- Definitions of pie types
inductive Filling
| apple
| cherry

inductive PreparationMethod
| fried
| baked

structure Pie where
  filling : Filling
  method : PreparationMethod

-- The set of all possible pie types
def pies : Set Pie := {
  {filling := Filling.apple, method := PreparationMethod.fried},
  {filling := Filling.cherry, method := PreparationMethod.fried},
  {filling := Filling.apple, method := PreparationMethod.baked},
  {filling := Filling.cherry, method := PreparationMethod.baked}
}

-- The statement to prove: If there are at least three types of pies available, then there exist two pies that differ in both filling and preparation method.
theorem pies_differ_in_both :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.method ≠ p2.method :=
begin
  sorry
end

end pies_differ_in_both_l137_137163


namespace binomial_coeff_ratio_l137_137102

theorem binomial_coeff_ratio {x : ℂ} :
  let a := (finset.range 7).choose 3,
      b := ((finset.range 7).choose 4) * (2 ^ 4) in
  (b / a) = 12 :=
by
  -- We would provide the proof here, but it's omitted as per the instruction.
  sorry

end binomial_coeff_ratio_l137_137102


namespace probability_non_red_face_l137_137318

theorem probability_non_red_face :
  let red_faces := 5
  let yellow_faces := 3
  let blue_faces := 1
  let green_faces := 1
  let total_faces := 10
  let non_red_faces := yellow_faces + blue_faces + green_faces
  (non_red_faces / total_faces : ℚ) = 1 / 2 :=
by
  -- Using Lean's Rat library for rational numbers.
  let red_faces := 5
  let yellow_faces := 3
  let blue_faces := 1
  let green_faces := 1
  let total_faces := 10
  let non_red_faces := yellow_faces + blue_faces + green_faces
  show (non_red_faces / total_faces : ℚ) = 1 / 2
  sorry

end probability_non_red_face_l137_137318


namespace range_of_m_for_complex_quadrants_l137_137596

theorem range_of_m_for_complex_quadrants (m : ℝ) : 
  let z := (m + 1) - (m - 3) * complex.I in 
  (∃ θ : ℝ, z = complex.ofReal (cos θ) + complex.I * sin θ ∧ 
          ((π / 2 < θ ∧ θ < π) ∨ (3 * π / 2 < θ ∧ θ < 2 * π))) → 
  m ∈ Set.Iio (-1) ∪ Set.Ioi 3 :=
by
  sorry

end range_of_m_for_complex_quadrants_l137_137596


namespace average_first_12_correct_l137_137695

noncomputable def average_of_first_12 (S : ℕ → ℕ) (sum25 : ℕ) : ℕ :=
  let sum1_to_12 := (∑ i in range 12, S i)
  let sum13 := S 12
  let sum14_to_25 := (∑ i in range 13 25, S i)
  (sum1_to_12 + sum13 + sum14_to_25) / 25 = 20 ->
  sum14_to_25 / 12 = 17 ->
  sum13 = 128 ->
  sum1_to_12 / 12

theorem average_first_12_correct (S : ℕ → ℕ) :
  let A := average_of_first_12 S (25 * 20) in
  A = 14 :=
by sorry

end average_first_12_correct_l137_137695


namespace simplify_trigonometric_expression_l137_137306

theorem simplify_trigonometric_expression (x : ℝ) :
  (\cot x - 2 * \cot (2 * x) = \tan x) →
  (\tan x + 4 * \tan (2 * x) + 16 * \tan (4 * x) + 64 * \cot (16 * x) = \tan x) :=
by
  sorry

end simplify_trigonometric_expression_l137_137306


namespace sandy_carrots_l137_137685

theorem sandy_carrots (n m : ℕ) (h1 : n = 6) (h2 : m = 3) : n - m = 3 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}


end sandy_carrots_l137_137685


namespace total_juice_drunk_l137_137300

noncomputable def juiceConsumption (samDrink benDrink : ℕ) (samConsRatio benConsRatio : ℚ) : ℚ :=
  let samConsumed := samConsRatio * samDrink
  let samRemaining := samDrink - samConsumed
  let benConsumed := benConsRatio * benDrink
  let benRemaining := benDrink - benConsumed
  let benToSam := (1 / 2) * benRemaining + 1
  let samTotal := samConsumed + benToSam
  let benTotal := benConsumed - benToSam
  samTotal + benTotal

theorem total_juice_drunk : juiceConsumption 12 20 (2 / 3 : ℚ) (2 / 3 : ℚ) = 32 :=
sorry

end total_juice_drunk_l137_137300


namespace triangle_sides_condition_l137_137613

-- In an oblique triangle ABC, the sides opposite to angles A, B, and C are denoted as a, b, and c respectively.
-- The given condition is: (tan C / tan A) + (tan C / tan B) = 1.

variable (A B C : ℝ) (a b c : ℝ)

-- Defining the condition given in the problem
def condition1 := (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 1

-- The theorem we want to prove
theorem triangle_sides_condition (h : condition1 A B C a b c) : (a^2 + b^2) / c^2 = 3 := by
  sorry

end triangle_sides_condition_l137_137613


namespace k_mod_3_not_1_l137_137859

noncomputable def partitionable (M : Set ℕ) : Prop :=
  ∃ A B C : Set ℕ, A ∪ B ∪ C = M ∧ A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
  (∑ x in A, x) = (∑ x in B, x) ∧ (∑ x in B, x) = (∑ x in C, x)

theorem k_mod_3_not_1 (k : ℕ) (hk : 0 < k)
  (hM : let M := { x | ∃ n, n = 3^31 + x ∧ x ≤ k } in partitionable M) :
  k % 3 ≠ 1 := 
sorry

end k_mod_3_not_1_l137_137859


namespace scale_length_discrepancy_l137_137012

theorem scale_length_discrepancy
  (scale_length_feet : ℝ)
  (parts : ℕ)
  (part_length_inches : ℝ)
  (ft_to_inch : ℝ := 12)
  (total_length_inches : ℝ := parts * part_length_inches)
  (scale_length_inches : ℝ := scale_length_feet * ft_to_inch) :
  scale_length_feet = 7 → 
  parts = 4 → 
  part_length_inches = 24 →
  total_length_inches - scale_length_inches = 12 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end scale_length_discrepancy_l137_137012


namespace johnny_marble_choices_l137_137631

theorem johnny_marble_choices : ∃ n k : ℕ, n = 9 ∧ k = 4 ∧ nat.choose n k = 126 := 
by
  use 9
  use 4
  simp
  sorry

end johnny_marble_choices_l137_137631


namespace steps_back_to_start_l137_137942

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def steps_for_move (n : ℕ) : ℤ :=
  if is_prime n then 2 else -3

def total_steps : ℤ :=
  (Finset.range (30 - 1)).sum (λ i, steps_for_move (i + 2))

theorem steps_back_to_start :
  (total_steps = -37) → abs(total_steps) = 37 := by
  sorry

end steps_back_to_start_l137_137942


namespace area_hexagon_l137_137679

theorem area_hexagon (AB AC BC ABDE CAHI BCFG : ℝ)
  (h1 : AB = 3)
  (h2 : AC = 4)
  (h3 : BC = 5)
  (h4 : ABDE = 9)
  (h5 : CAHI = 9)
  (h6 : BCFG = 30) :
  ABDE + CAHI + BCFG = 52.5 :=
by
  have h7 : ABDE + CAHI = 18,
    from Eq.symm (by ring [h4, h5]),
  have h8 : ABDE + CAHI + BCFG = 18 + 30,
    from Eq.symm (by ring [h7, h6]),
  linarith

end area_hexagon_l137_137679


namespace range_of_a_l137_137900

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x < 0) → (∃ x : ℝ, f a x = 0) → a < -Real.sqrt 2 := by
  sorry

end range_of_a_l137_137900


namespace range_of_quadratic_function_l137_137918

noncomputable def quadratic_range 
  (a b c : ℝ) (h_a : a < 0) : set ℝ :=
  let f := λ x : ℝ, a * x^2 + b * x + c in
  let vertex_x := -b / (2 * a) in
  let f_vertex := f vertex_x in
  let f_0 := f 0 in
  let f_2 := f 2 in
  let lower_bound := 4 * a + 2 * b + c in
  if 0 ≤ vertex_x ∧ vertex_x ≤ 2 then
    set.Icc lower_bound (f_vertex + c)
  else
    set.Icc lower_bound c

-- Proof for the calculated range, omitted here
theorem range_of_quadratic_function (a b c : ℝ) (h_a : a < 0) :
  quadratic_range a b c h_a = 
    if 0 ≤ -b / (2 * a) ∧ -b / (2 * a) ≤ 2 then
      set.Icc (4 * a + 2 * b + c) (-b^2 / (4 * a) + c)
    else
      set.Icc (4 * a + 2 * b + c) c :=
sorry

end range_of_quadratic_function_l137_137918


namespace white_dandelions_on_saturday_l137_137003

-- Define the conditions.
def dandelion_blooming_stage : Nat → String
| 0 => "yellow"
| 1 => "yellow"
| 2 => "yellow"
| 3 => "white"
| 4 => "shed"

variable (monday yellow monday white : Nat) (wednesday yellow wednesday white : Nat)
variable (total_wednesday : Nat := wednesday yellow + wednesday white)

-- Define the initial conditions.
def initial_conditions := monday yellow = 20 ∧ monday white = 14 ∧ 
                          wednesday yellow = 15 ∧ wednesday white = 11

-- Definition of the transition from Monday to Wednesday.
def transition_conditions := total_wednesday = 26 ∧
                             total wednesday - monday yellow = 6

-- Goal: The number of white dandelions on Saturday.
theorem white_dandelions_on_saturday 
  (h1 : initial_conditions)
  (h2 : transition_conditions) :
  ∃ n : Nat, n = 6 :=
by
  sorry

end white_dandelions_on_saturday_l137_137003


namespace distinct_three_digit_numbers_count_l137_137127

theorem distinct_three_digit_numbers_count : 
  (∃ digits : Finset ℕ, 
    digits = {1, 2, 3, 4, 5} ∧ 
    ∀ n ∈ digits, n ∈ {1, 2, 3, 4, 5} ∧ 
    (digits.card = 5)) → 
  card (finset.univ.image (λ p : Finset (ℕ × ℕ × ℕ), 
      {x | x.fst ∈ digits ∧ x.snd ∈ digits ∧ x.snd.snd ∈ digits ∧ x.fst ≠ x.snd ∧ x.snd ≠ x.snd.snd ∧ x.fst ≠ x.snd.snd})) = 60 := by
  sorry

end distinct_three_digit_numbers_count_l137_137127


namespace worker_allocation_correct_l137_137455

variable (x y : ℕ)
variable (H1 : x + y = 50)
variable (H2 : x = 30)
variable (H3 : y = 20)
variable (H4 : 120 * (50 - x) = 2 * 40 * x)

theorem worker_allocation_correct 
  (h₁ : x = 30) 
  (h₂ : y = 20) 
  (h₃ : x + y = 50) 
  (h₄ : 120 * (50 - x) = 2 * 40 * x) 
  : true := 
by
  sorry

end worker_allocation_correct_l137_137455


namespace parabola_vertex_is_two_one_l137_137700

theorem parabola_vertex_is_two_one : 
  ∀ x y : ℝ, (y = (x - 2)^2 + 1) → (2, 1) = (2, 1) :=
by
  intros x y hyp
  sorry

end parabola_vertex_is_two_one_l137_137700


namespace max_value_of_trig_expression_l137_137499

theorem max_value_of_trig_expression (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_value_of_trig_expression_l137_137499


namespace BQ_not_perpendicular_PD_l137_137796

-- Define the vertices of the tetrahedron
variables {A B C D S : ℝ³}

-- Define the points P and Q as described in the conditions
def P : ℝ³ := (1/2 * (S + A))
def Q (m : ℝ) : ℝ³ := (1 - m) * S + m * C

-- Define vectors DP and BQ
def DP : ℝ³ := P - D
def BQ (m : ℝ) : ℝ³ := Q m - B

-- Constraint: 0 ≤ m ≤ 2
variables (m : ℝ) (h : 0 ≤ m ∧ m ≤ 2)

-- Define potential values for the vertices
variables [fact (A = (1:ℝ, 0, 0))]
variables [fact (B = (0, Math.sqrt 2, 0))]
variables [fact (C = (-1, 0, 0))]
variables [fact (D = (0, -Math.sqrt 2, 0))]
variables [fact (S = (0, 0, 2))]

-- Prove that BQ cannot be perpendicular to PD
theorem BQ_not_perpendicular_PD : 
  ¬ (BQ m) ⬝ DP = 0 := sorry

end BQ_not_perpendicular_PD_l137_137796


namespace find_BE_l137_137211

-- Define the conditions
variables {V : Type*} [AddCommGroup V] [VectorSpace ℚ V] -- Vector space over rationals
variables (P A B C D E : V)
variables (a b c : V)

-- Define the given vectors
variables (hPA : P = a) (hPB : P = b) (hPC : P = c)

-- Define the midpoint condition
def midpoint (E P D : V) : Prop := 2 • E = P + D

-- Conclusion
theorem find_BE (h1 : midpoint E P D) (h2 : a = P - A) (h3 : b = P - B) (h4 : c = P - C):
  E - B = (1/2 : ℚ) • a - (3/2 : ℚ) • b + (1/2 : ℚ) • c :=
sorry

end find_BE_l137_137211


namespace length_of_tank_l137_137771

namespace TankProblem

def field_length : ℝ := 90
def field_breadth : ℝ := 50
def field_area : ℝ := field_length * field_breadth

def tank_breadth : ℝ := 20
def tank_depth : ℝ := 4

def earth_volume (L : ℝ) : ℝ := L * tank_breadth * tank_depth

def remaining_field_area (L : ℝ) : ℝ := field_area - L * tank_breadth

def height_increase : ℝ := 0.5

theorem length_of_tank (L : ℝ) :
  earth_volume L = remaining_field_area L * height_increase →
  L = 25 :=
by
  sorry

end TankProblem

end length_of_tank_l137_137771


namespace candies_bought_friday_l137_137050

-- Definitions based on the given conditions
def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_left (c : ℕ) : Prop := c = 4
def candies_eaten (c : ℕ) : Prop := c = 6

-- Theorem to prove the number of candies bought on Friday
theorem candies_bought_friday (c_left c_eaten : ℕ) (h_left : candies_left c_left) (h_eaten : candies_eaten c_eaten) : 
  (10 - (candies_bought_tuesday + candies_bought_thursday) = 2) :=
  by
    sorry

end candies_bought_friday_l137_137050


namespace three_rays_with_common_point_l137_137236

-- Define the problem
def S_a (a : ℕ) : set (ℝ × ℝ) :=
  {p | let (x, y) := p in 
       (a = x + 2 ∧ y - 4 ≤ a) ∨ 
       (a = y - 4 ∧ x + 2 ≤ a) ∨ 
       (x + 2 = y - 4 ∧ a ≤ x + 2)}

theorem three_rays_with_common_point (a : ℕ) (h : 0 < a) : 
  ∃ p, p = (a-2 : ℝ, a+4 : ℝ) ∧
  (∃ S1 S2 S3, (Ray S1 ∧ ((a - 2, a + 4) ∈ S1) ∧ (a ≠ x + 2)) ∧
  (Ray S2 ∧ ((a - 2, a + 4) ∈ S2) ∧ (a ≠ y - 4)) ∧
  (Ray S3 ∧ ((a - 2, a + 4) ∈ S3) ∧ (x + 2 ≠ y - 4)) ∧
  (S_a a = S1 ∪ S2 ∪ S3)) :=
begin
  sorry
end

end three_rays_with_common_point_l137_137236


namespace base_subtraction_correct_l137_137483

theorem base_subtraction_correct :
  let nine_digit := 3 * 9^2 + 2 * 9^1 + 4 * 9^0,
      six_digit := 2 * 6^2 + 1 * 6^1 + 5 * 6^0
  in nine_digit - six_digit = 182 :=
by
  let nine_digit := 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  let six_digit := 2 * 6^2 + 1 * 6^1 + 5 * 6^0
  exact sorry

end base_subtraction_correct_l137_137483


namespace find_maximum_value_l137_137537

theorem find_maximum_value {n : ℕ} (a : Fin (2 * n + 1) → ℝ)
  (h : ∑ i in Finset.range (2 * n - 1), (a (i+1) - a i) ^ 2 = 1) :
  let lhs := ∑ i in Finset.range n, a (n + 1 + i) 
  let rhs := ∑ i in Finset.range n, a i
  in lhs - rhs ≤ Real.sqrt (n * (2 * n^2 + 1) / 3) :=
sorry

end find_maximum_value_l137_137537


namespace max_daily_net_income_l137_137442

def f (x : ℕ) : ℤ :=
  if 3 ≤ x ∧ x ≤ 6 then 50 * x - 115
  else if 6 < x ∧ x ≤ 20 then -3 * x ^ 2 + 68 * x - 115
  else 0

theorem max_daily_net_income :
  ∃ (x : ℕ), 3 ≤ x ∧ x ≤ 20 ∧ f x = 270 :=
begin
  sorry
end

end max_daily_net_income_l137_137442


namespace solve_inequality_l137_137313

open Set

theorem solve_inequality (x : ℝ) :
  { x | (x^2 - 9) / (x^2 - 16) > 0 } = (Iio (-4)) ∪ (Ioi 4) :=
by
  sorry

end solve_inequality_l137_137313


namespace max_trig_expression_l137_137503

theorem max_trig_expression (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_trig_expression_l137_137503


namespace white_on_saturday_l137_137005

noncomputable def dandelion_bloom := 
  { t : ℕ // t >= 0 }

def blooming_stages (d : dandelion_bloom) : ℕ :=
if d.val < 3 then 0 -- yellow
else if d.val < 4 then 1 -- white
else 2 -- seeds dispersed

-- Conditions:
-- Monday's counts
def monday_yellow : ℕ := 20
def monday_white : ℕ := 14

-- Wednesday's counts
def wednesday_yellow : ℕ := 15
def wednesday_white : ℕ := 11

-- Additional Definitions arising from conditions:
def monday_total : ℕ := monday_yellow + monday_white
def wednesday_total : ℕ := wednesday_yellow + wednesday_white
def new_bloomed : ℕ := wednesday_total - monday_yellow

-- Theorem to prove the correct answer.
theorem white_on_saturday : ∀ (initial_monday_total initial_wednesday_total monday_yellows : ℕ),
  initial_monday_total = 34 → -- Monday's 20 yellow + 14 white
  initial_wednesday_total = 26 → -- Wednesday's 15 yellow + 11 white
  monday_yellows = 20 → -- Monday's yellow dandelions
  (initial_wednesday_total - monday_yellows) = 6 :=  -- Newly bloomed turning white on Saturday
by
  assume initial_monday_total initial_wednesday_total monday_yellows,
  assume h₁ : initial_monday_total = 34,
  assume h₂ : initial_wednesday_total = 26,
  assume h₃ : monday_yellows = 20,
  rw [h₂, h₃],
  simp,
  exact 6

end white_on_saturday_l137_137005


namespace quadratic_roots_transform_l137_137241

theorem quadratic_roots_transform {p q : ℝ} (h1 : 3 * p^2 + 5 * p - 7 = 0) (h2 : 3 * q^2 + 5 * q - 7 = 0) : (p - 2) * (q - 2) = 5 := 
by 
  sorry

end quadratic_roots_transform_l137_137241


namespace period_of_f_monotonically_increasing_intervals_l137_137109

def f (x : ℝ) : ℝ := cos (π / 3 * x + π / 3) - 2 * (cos (π / 6 * x))^2

theorem period_of_f : ∀ x, f (x + 6) = f x :=
by sorry

theorem monotonically_increasing_intervals : ∀ x k : ℤ, 
  6 * k + 1 ≤ x ∧ x ≤ 6 * k + 4 → ∀ x1 x2, x1 ≤ x2 → x1 ≤ x2 → f x1 ≤ f x2 :=
by sorry

end period_of_f_monotonically_increasing_intervals_l137_137109


namespace smallest_pos_int_roots_unity_l137_137740

theorem smallest_pos_int_roots_unity :
  let f : ℂ → ℂ := λ z, z^4 - z^3 + 1
  let is_root_14th_unity (z : ℂ) : Prop := z^14 = 1
  ∀ z : ℂ, f z = 0 → ∃ n : ℕ, n > 0 ∧ (∀ z' : ℂ, f z' = 0 → is_root_14th_unity z') ∧ n = 14  := 
  sorry

end smallest_pos_int_roots_unity_l137_137740


namespace abc_geometric_progression_l137_137145

theorem abc_geometric_progression
    (a b c : ℝ)
    (h : 9 * b^2 - 4 * a * c = 0) : 
    let r := (3*b)/(2*a) in 
    let s := (3*b)/(2*c) in 
    r = s :=
by sorry

end abc_geometric_progression_l137_137145


namespace condition1_correct_condition2_correct_l137_137301

-- Given definitions and conditions
def boys : Finset String := {"B1", "B2", "B3", "B4"}
def girls : Finset String := {"G1", "G2", "G3", "G4", "G5"}
def girl_A : String := "G1"
def boy_A : String := "B1"
def girl_B : String := "G2"

-- Condition 1: Select 2 boys and 3 girls, and girl A must be selected
def ways_condition1 : ℕ :=
  (Finset.choose 2 boys).card * (Finset.choose 2 (girls \ {girl_A})).card

-- Condition 2: Select at most 4 girls, and boy A and girl B cannot be selected at the same time
def ways_condition2_case1 : ℕ :=
  (Finset.choose 4 (boys \ {boy_A} ∪ (girls \ {girl_B}))).card

def ways_condition2_case2 : ℕ :=
  (Finset.choose 5 ((boys \ {boy_A}) ∪ girls)).card

def total_ways_condition2 : ℕ :=
  ways_condition2_case1 + ways_condition2_case2

-- Proof Statements
theorem condition1_correct : ways_condition1 = 36 :=
  by sorry

theorem condition2_correct : total_ways_condition2 = 91 :=
  by sorry

end condition1_correct_condition2_correct_l137_137301


namespace bristol_to_carlisle_routes_l137_137746

-- Given conditions
def r_bb := 6
def r_bs := 3
def r_sc := 2

-- The theorem we want to prove
theorem bristol_to_carlisle_routes :
  (r_bb * r_bs * r_sc) = 36 :=
by
  sorry

end bristol_to_carlisle_routes_l137_137746


namespace f_geq_g_l137_137529

def f (a : ℕ) : ℕ :=
  (Nat.divisors a).count (λ b => b % 10 = 1 ∨ b % 10 = 9)

def g (a : ℕ) : ℕ :=
  (Nat.divisors a).count (λ b => b % 10 = 3 ∨ b % 10 = 7)

theorem f_geq_g (a : ℕ) : f a ≥ g a :=
  sorry

end f_geq_g_l137_137529


namespace no_three_digit_number_divisible_by_15_l137_137582

theorem no_three_digit_number_divisible_by_15 (digits : Finset ℕ) (h_digits : digits = {2, 3, 5, 6, 9}) :
  (∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (∀ d ∈ digits, d ∈ (digit_list n)) ∧ n % 15 = 0 ∧ ¬(∃ d1 d2 d3, d1 ≠ d2 ∧ d2 ≠ d3 ∧ n = 100 * d1 + 10 * d2 + d3)) → False :=
by
  sorry

end no_three_digit_number_divisible_by_15_l137_137582


namespace sufficient_condition_parallel_planes_l137_137073

-- Definitions for lines and planes
variable {Line Plane : Type}
variable {m n : Line}
variable {α β : Plane}

-- Relations between lines and planes
variable (parallel_line : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Condition for sufficient condition for α parallel β
theorem sufficient_condition_parallel_planes
  (h1 : parallel_line m n)
  (h2 : perpendicular_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  parallel_plane α β :=
sorry

end sufficient_condition_parallel_planes_l137_137073


namespace exists_trapezoid_in_marked_vertices_l137_137192

def is_regular_polygon (n : ℕ) (polygon : Type) [Fintype polygon] : Prop :=
  ∀ (u v w : polygon), u ≠ v → (∃ (p q : polygon), p ≠ q ∧ (p, q) ≠ (u, v) ∧ (p, q) = (v, w))

def is_trapezoid {α : Type*} [AffineSpace α] {polygon : Finset α} (v₁ v₂ v₃ v₄ : α) : Prop :=
  ∃ (m : AffineSubspace ℝ α),
  v₁ ∈ m ∧ v₂ ∈ m ∧ v₃ ∉ m ∧ v₄ ∉ m

theorem exists_trapezoid_in_marked_vertices :
  ∀ (polygon : Type) [Fintype polygon] (n : ℕ) (m : ℕ),
  is_regular_polygon n polygon →
  m = 64 → n = 1981 → ∃ (v₁ v₂ v₃ v₄ : polygon), is_trapezoid v₁ v₂ v₃ v₄ :=
begin
  intros,
  sorry
end

end exists_trapezoid_in_marked_vertices_l137_137192


namespace sum_even_numbers_l137_137841

def is_even (n : ℕ) : Prop := n % 2 = 0

def largest_even_less_than_or_equal (n m : ℕ) : ℕ :=
if h : m % 2 = 0 ∧ m ≤ n then m else
if h : m % 2 = 1 ∧ (m - 1) ≤ n then m - 1 else 0

def smallest_even_less_than_or_equal (n : ℕ) : ℕ :=
if h : 2 ≤ n then 2 else 0

theorem sum_even_numbers (n : ℕ) (h : n = 49) :
  largest_even_less_than_or_equal n 48 + smallest_even_less_than_or_equal n = 50 :=
by sorry

end sum_even_numbers_l137_137841


namespace interest_rate_half_yearly_l137_137493

variable (P : ℝ) (A : ℝ) (t : ℝ) (n : ℝ) (r : ℝ)

theorem interest_rate_half_yearly :
  P = 4000 →
  A = 4242.38423530919772 →
  t = 1.5 →
  n = 2 →
  (A = P * (1 + r/n)^(n*t)) →
  r ≈ 0.0396 :=
by
  intros hP hA ht hn hFormula
  sorry

end interest_rate_half_yearly_l137_137493


namespace max_elements_subset_no_sum_mod_5_divisible_l137_137638

theorem max_elements_subset_no_sum_mod_5_divisible (T : Set ℕ)
  (hT₁ : ∀ x ∈ T, x ∈ Finset.range 101)
  (hT₂ : ∀ x y ∈ T, x ≠ y → ¬ (x + y) % 5 = 0) : T.card ≤ 40 := sorry

end max_elements_subset_no_sum_mod_5_divisible_l137_137638


namespace two_pies_differ_l137_137180

theorem two_pies_differ (F_A F_C B_A B_C : Bool) :
  (F_A ∨ F_C ∨ B_A ∨ B_C) →
  (F_A ∧ F_C ∧ B_A ∧ B_C) ∧ 
  (∀ a b, (a ≠ b) → (a.filling ≠ b.filling ∧ a.preparation ≠ b.preparation)) :=
by
  intros H1 H2
  sorry

end two_pies_differ_l137_137180


namespace sin_A_in_right_triangle_l137_137957

theorem sin_A_in_right_triangle (A B C : Type) [MetricSpace A] [Triangle ABC] :
  angle B = π / 2 ∧ AB = 13 ∧ BC = 5 → sin A = 5 / 12 :=
by
  sorry

end sin_A_in_right_triangle_l137_137957


namespace arrange_in_ascending_order_l137_137870

noncomputable def a : ℝ := Real.log (1 / 2)
noncomputable def b : ℝ := (1 / 3) ^ 0.8
noncomputable def c : ℝ := 2 ^ (1 / 3)

theorem arrange_in_ascending_order : a < b ∧ b < c :=
by sorry

end arrange_in_ascending_order_l137_137870


namespace least_integer_value_l137_137411

theorem least_integer_value (x : ℝ) (h : |3 * x - 4| ≤ 25) : x = -7 :=
sorry

end least_integer_value_l137_137411


namespace minimum_value_on_interval_l137_137564

noncomputable def f (x : ℝ) : ℝ := 3*x^4 - 8*x^3 - 18*x^2 + 6

theorem minimum_value_on_interval :
  (∀ x ∈ (set.Icc (-1 : ℝ) 1), f x ≤ 6) ∧ (∃ x ∈ (set.Icc (-1 : ℝ) 1), f x = 6) →
  (∃ y ∈ (set.Icc (-1 : ℝ) 1), f y = -17) :=
sorry

end minimum_value_on_interval_l137_137564


namespace least_x_for_inequality_l137_137494

theorem least_x_for_inequality : 
  ∃ (x : ℝ), (-x^2 + 9 * x - 20 ≤ 0) ∧ ∀ y, (-y^2 + 9 * y - 20 ≤ 0) → x ≤ y ∧ x = 4 := 
by
  sorry

end least_x_for_inequality_l137_137494


namespace distinct_three_digit_numbers_l137_137130

theorem distinct_three_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in
  ∃ (count : ℕ), count = 60 := by
  sorry

end distinct_three_digit_numbers_l137_137130


namespace lana_pages_after_adding_duane_l137_137971

theorem lana_pages_after_adding_duane :
  ∀ (lana_initial_pages duane_total_pages : ℕ), 
  lana_initial_pages = 8 → 
  duane_total_pages = 42 → 
  lana_initial_pages + (duane_total_pages / 2) = 29 :=
by
  intros lana_initial_pages duane_total_pages h_lana h_duane
  rw [h_lana, h_duane]
  norm_num

end lana_pages_after_adding_duane_l137_137971


namespace find_p8_l137_137254

noncomputable def p (x : ℝ) : ℝ := if x = 1 then 0 
                               else if x = 2 then 1
                               else if x = 3 then 2 
                               else if x = 4 then 3 
                               else if x = 5 then 4 
                               else if x = 6 then 5 
                               else if x = 7 then 6 
                               else (x - 2)*(x - 3)*(x - 4)*(x - 5)*(x - 6)*(x - 7) + (x - 1)

theorem find_p8 : p 8 = 727 :=
by {
   simp [p], 
   norm_num,
   sorry
}

end find_p8_l137_137254


namespace vertex_of_parabola_l137_137697

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, (x - 2)^2 + 1 = (x - h)^2 + k) ∧ (h = 2 ∧ k = 1) :=
by
  use 2, 1
  split
  · intro x
    ring
  · exact ⟨rfl, rfl⟩

end vertex_of_parabola_l137_137697


namespace proof_problem_l137_137895

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l137_137895


namespace trees_determination_impossible_l137_137764

theorem trees_determination_impossible :
  ∀ (n : ℕ) (trees : Fin n → ℕ) (tags : Fin n → ℕ),
    n = 2000 →
    (∀ i : Fin n, tags i = if i = 0 then count_oaks trees 0 1
                          else if i = n - 1 then count_oaks trees (n - 2) (n - 1)
                          else count_oaks trees (i - 1) (i + 1)) →
    ¬ ∃ (is_oak : Fin n → Prop),
      (∀ i : Fin n, is_oak i ↔ trees i = 1) :=
begin
  intros n trees tags h₁ h₂,
  sorry
end

noncomputable def count_oaks (trees : Fin 2000 → ℕ) (a b : Fin 2000) : ℕ :=
  if a < b then trees a + trees b + trees (b - 1)
  else trees a + trees b + trees (a - 1)

end trees_determination_impossible_l137_137764


namespace value_of_A_l137_137346

theorem value_of_A 
  (H M A T E: ℤ)
  (H_value: H = 10)
  (MATH_value: M + A + T + H = 35)
  (TEAM_value: T + E + A + M = 42)
  (MEET_value: M + 2*E + T = 38) : 
  A = 21 := 
by 
  sorry

end value_of_A_l137_137346


namespace pies_differ_in_both_l137_137171

-- Defining types of pies
inductive Filling where
  | apple : Filling
  | cherry : Filling

inductive Preparation where
  | fried : Preparation
  | baked : Preparation

structure Pie where
  filling : Filling
  preparation : Preparation

-- The set of all possible pies
def allPies : Set Pie :=
  { ⟨Filling.apple, Preparation.fried⟩,
    ⟨Filling.apple, Preparation.baked⟩,
    ⟨Filling.cherry, Preparation.fried⟩,
    ⟨Filling.cherry, Preparation.baked⟩ }

-- Theorem stating that we can buy two pies that differ in both filling and preparation
theorem pies_differ_in_both (pies : Set Pie) (h : 3 ≤ pies.card) :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
  sorry

end pies_differ_in_both_l137_137171


namespace smallest_odd_f_l137_137252

def f (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (∑ k in finset.range (n+1), if (n - 2^k) < 0 then 0 else f (n - 2^k))

theorem smallest_odd_f (n : ℕ) (h : n > 2013) : f n % 2 = 1 ↔ n = 2047 := by
  sorry

end smallest_odd_f_l137_137252


namespace count_elements_starting_with_1_l137_137986

theorem count_elements_starting_with_1 :
  let S := { k | 0 ≤ k ∧ k ≤ 1500 }
  ∃! n ∈ S, ∃ (a : ℕ), leading_digit (3^k) = 1 ∧ (784 = S.card) := sorry

end count_elements_starting_with_1_l137_137986


namespace hockey_games_per_month_l137_137728

theorem hockey_games_per_month {
  total_games : ℕ,
  months_in_season : ℕ
} (h1 : total_games = 182) (h2 : months_in_season = 14) :
  total_games / months_in_season = 13 := by
  sorry

end hockey_games_per_month_l137_137728


namespace cryptarithm_main_l137_137491

noncomputable def cryptarithm_solution1 (Φ E BP A J : ℕ) : Prop :=
  Φ = 2 ∧ E = 4 ∧ BP = 79 ∧ A = 1 ∧ J = 158 ∧ Φ / E + BP / A / J = 1

noncomputable def cryptarithm_solution2 (Φ E BP A J : ℕ) : Prop :=
  Φ = 6 ∧ E = 8 ∧ BP = 35 ∧ A = 1 ∧ J = 140 ∧ Φ / E + BP / A / J = 1

noncomputable def cryptarithm_solution3 (Φ E BP A J : ℕ) : Prop :=
  Φ = 4 ∧ E = 5 ∧ BP = 72 ∧ A = 1 ∧ J = 360 ∧ Φ / E + BP / A / J = 1

-- Main statement asserting that these solutions indeed sum to 1
theorem cryptarithm_main :
  ∃ (Φ E BP A J : ℕ),
    (cryptarithm_solution1 Φ E BP A J ∨ cryptarithm_solution2 Φ E BP A J ∨ cryptarithm_solution3 Φ E BP A J) :=
by {
  use [2, 4, 79, 1, 158],
  left,
  repeat {split; try {refl}; try {norm_num}}
} ∨
by {
  use [6, 8, 35, 1, 140],
  right, left,
  repeat {split; try {refl}; try {norm_num}}
} ∨
by {
  use [4, 5, 72, 1, 360],
  right, right,
  repeat {split; try {refl}; try {norm_num}}
}

end cryptarithm_main_l137_137491


namespace num_ways_to_sum_420_as_consecutive_integers_l137_137197

def consecutive_sum (k n : ℕ) : ℕ := k * n + (k * (k - 1)) / 2

def valid_k (k : ℕ) : Prop := (k = 3 ∨ k % 2 = 0)

theorem num_ways_to_sum_420_as_consecutive_integers :
  (finset.filter (λ k, ∃ n : ℕ, consecutive_sum k n = 420 ∧ valid_k k) (finset.range 421)).card = 1 :=
sorry

end num_ways_to_sum_420_as_consecutive_integers_l137_137197


namespace quadratic_real_roots_range_l137_137152

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 3 * x - 9 / 4 = 0) →
  (k >= -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l137_137152


namespace curve_C_rectangular_eqn_point_D_coordinates_l137_137212
noncomputable def curve_equation (θ : ℝ) : ℝ × ℝ :=
let ρ := 2 * Real.cos θ in (ρ * Real.cos θ, ρ * Real.sin θ)

theorem curve_C_rectangular_eqn : 
∀ (x y : ℝ), (∃ θ ∈ Set.Icc 0 (Real.pi / 2), (x, y) = curve_equation θ) ↔ ((x - 1)^2 + y^2 = 1 ∧ 0 ≤ y ∧ y ≤ 1) :=
sorry

theorem point_D_coordinates (x y : ℝ) :
(∃ θ ∈ Set.Icc 0 (Real.pi / 2), (x, y) = curve_equation θ ∧ 
  let tan_slope := Real.tan θ in
  (tan_slope = /- slope of the perpendicular line -/ (-1 / sqrt 3))) →
(x, y) = (3/2, sqrt 3/2) :=
sorry

end curve_C_rectangular_eqn_point_D_coordinates_l137_137212


namespace trapezoid_third_largest_angle_l137_137009

theorem trapezoid_third_largest_angle (a d : ℝ)
  (h1 : 2 * a + 3 * d = 200)      -- Condition: 2a + 3d = 200°
  (h2 : a + d = 70) :             -- Condition: a + d = 70°
  a + 2 * d = 130 :=              -- Question: Prove a + 2d = 130°
by
  sorry

end trapezoid_third_largest_angle_l137_137009


namespace min_value_n_constant_term_l137_137389

-- Define the problem statement
theorem min_value_n_constant_term (n r : ℕ) (h : 2 * n = 5 * r) : n = 5 :=
by sorry

end min_value_n_constant_term_l137_137389


namespace _l137_137627

noncomputable def triangle_XYZ_conditions : Prop :=
∃ (X Y Z : Type) [Inhabited X] [Inhabited Y] [Inhabited Z]
(α : X → Y → Z → Prop)
(angle_Y : α Y Y Y)
(yz : YZ = 4)
(xz : XZ = 5)
(angle_Y_def : ∀ x y z, α x y z → angle_Y = 90) 
(triangle_def : ∀ x y z, α x y z → X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z)
(Pythagorean_theorem : ∀ xy xz yz, xy = sqrt(xz^2 - yz^2))
(tan_X : ∀ yz xy, tan X = yz / xy)
(sin_Z : ∀ yz xz, sin Z = yz / xz), 
tan X = 4/3 ∧ sin Z = 4/5

#eval triangle_XYZ_conditions

end _l137_137627


namespace tangent_line_at_1_l137_137111

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - f 1 * x^2 + 2 * f' 0 * x - Real.exp 1)
noncomputable def g (x : ℝ) : ℝ := (3 / 2 * x^2 - x + 1)

theorem tangent_line_at_1 :
  let y := f 1, t := (fun x => Real.exp x - Real.exp 1 - 1)
  f' 1 = Real.exp 1 ∧ (∀ x, f x + Real.exp 1 ≥ g x) := sorry

end tangent_line_at_1_l137_137111


namespace solve_for_x_l137_137920

theorem solve_for_x (x : ℝ) (hx_pos : 0 < x) (h : (sqrt (8 * x) * sqrt (10 * x) * sqrt (3 * x) * sqrt (15 * x) = 15)) : x = 1 / 2 :=
sorry

end solve_for_x_l137_137920


namespace reasonable_statements_l137_137026

def statement_A : Prop :=
  "Both the mark-recapture method and the quadrat method are sampling survey methods."

def statement_B : Prop :=
  "The key to sampling is to achieve random sampling."

def statement_C : Prop :=
  "The mark-recapture method is used to investigate the population density of animals with strong mobility."

def statement_D : Prop :=
  "Population density can accurately reflect the trend of population quantity changes."

theorem reasonable_statements :
  (statement_A ∧ statement_B ∧ statement_C) ∧ ¬statement_D :=
by
  sorry

end reasonable_statements_l137_137026


namespace triangle_problem_part1_triangle_problem_part2_l137_137155

variables (A B C : ℝ) (a b c : ℝ)

/-- Proof problem statements for triangle ABC. -/
theorem triangle_problem_part1 
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : (1/2) * a * b * sin C = sqrt 3) :
  a = 2 ∧ b = 2 := 
sorry

theorem triangle_problem_part2 
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : sin C + sin (B - A) = 2 * sin (2 * A)) :
  (1/2) * a * b * sin C = (2 * sqrt 3) / 3 :=
sorry

end triangle_problem_part1_triangle_problem_part2_l137_137155


namespace sum_of_k_for_g_2k_eq_6_l137_137239

-- Define the condition of the function g
def g (x : ℝ) : ℝ :=
  x^2 - 4 * x + 3

-- Define the main statement to be proven
theorem sum_of_k_for_g_2k_eq_6 : 
  ∑ k in {k | g (2 * k) = 6}, k = 1 :=
by
  sorry

end sum_of_k_for_g_2k_eq_6_l137_137239


namespace sum_of_integers_is_zero_l137_137384

theorem sum_of_integers_is_zero {n : ℕ} (h : n > 2) (a : ℕ → ℤ) 
  (h_nonzero : ∀ i, a i ≠ 0)
  (h_divisible : ∀ i, ∑ j in finset.range n \ {i}, a j ∣ a i) :
  ∑ i in finset.range n, a i = 0 :=
sorry

end sum_of_integers_is_zero_l137_137384


namespace sum_of_prime_values_of_f_is_zero_l137_137530

noncomputable def f (n : ℤ) : ℤ :=
  n^4 - 400 * n^2 + 10000

theorem sum_of_prime_values_of_f_is_zero :
  ∑ n in (Finset.filter (λ n => Nat.Prime (f n)) (Finset.range 1)), f n = 0 :=
by
  sorry

end sum_of_prime_values_of_f_is_zero_l137_137530


namespace main_theorem_l137_137980

open Set 

variables (M : Set ℝ)

/-- Given conditions on M --/

-- Condition (a)
axiom cond_a : ∀ (x : ℝ) (n : ℤ), x ∈ M → x + (n : ℝ) ∈ M

-- Condition (b)
axiom cond_b : ∀ (x : ℝ), x ∈ M → -x ∈ M

-- Condition (c)
axiom cond_c : ∃ (I₁ I₂ : Set ℝ), is_interval I₁ ∧ is_interval I₂ ∧ nonempty I₁ ∧ nonempty I₂ ∧
                  (∀ x ∈ I₁, x ∈ M) ∧ (∀ x ∈ I₂, x ∈ (univ \ M))

/-- Definition of M(x) --/
def M_of (M : Set ℝ) (x : ℝ) : Set ℤ := {n : ℤ | (n : ℝ) * x ∈ M}

theorem main_theorem (α β : ℝ) (h : M_of M α = M_of M β) :
  (∃ r : ℚ, α + β = r) ∨ (∃ r : ℚ, α - β = r) :=
sorty

end main_theorem_l137_137980


namespace sector_radius_l137_137925

theorem sector_radius (α S r : ℝ) (h1 : α = 3/4 * Real.pi) (h2 : S = 3/2 * Real.pi) :
  S = 1/2 * r^2 * α → r = 2 :=
by
  sorry

end sector_radius_l137_137925


namespace remainder_division_l137_137517

theorem remainder_division (p: ℤ → ℤ) (a: ℤ) (r: ℤ) (h : p = λ x, x^5 + 2 * x^3 + x + 3) (ha: a = 2) (hr: r = 53) : 
  p(a) = r := by
  sorry

end remainder_division_l137_137517


namespace train_cross_time_l137_137413

theorem train_cross_time (length_of_train : ℕ) (speed_in_kmh : ℕ) (conversion_factor : ℕ) (speed_in_mps : ℕ) (time : ℕ) :
  length_of_train = 120 →
  speed_in_kmh = 72 →
  conversion_factor = 1000 / 3600 →
  speed_in_mps = speed_in_kmh * conversion_factor →
  time = length_of_train / speed_in_mps →
  time = 6 :=
by
  intros hlength hspeed hconversion hspeed_mps htime
  have : conversion_factor = 5 / 18 := sorry
  have : speed_in_mps = 20 := sorry
  exact sorry

end train_cross_time_l137_137413


namespace alice_bob_meet_after_turns_l137_137450

def circular_track_meeting (total_points : ℕ) (start_point : ℕ) (alice_moves : ℕ) (bob_moves : ℕ) : ℕ :=
  total_points / (alice_moves)   -- Corrected to match the correct number of turns.

theorem alice_bob_meet_after_turns :
  circular_track_meeting 15 15 4 4 = 4 :=
by
  unfold circular_track_meeting
  simp
  exact 4      -- As corrected, the first meeting points align at the 4th turn.

end alice_bob_meet_after_turns_l137_137450


namespace total_steps_traveled_l137_137448

def steps_per_mile : ℕ := 2000
def walk_to_subway : ℕ := 2000
def subway_ride_miles : ℕ := 7
def walk_to_rockefeller : ℕ := 3000
def cab_ride_miles : ℕ := 3

theorem total_steps_traveled :
  walk_to_subway +
  (subway_ride_miles * steps_per_mile) +
  walk_to_rockefeller +
  (cab_ride_miles * steps_per_mile)
  = 24000 := 
by 
  sorry

end total_steps_traveled_l137_137448


namespace minimum_value_fraction_l137_137505

theorem minimum_value_fraction (x : ℝ) (h : x > 6) : (∃ c : ℝ, c = 12 ∧ ((x = c) → (x^2 / (x - 6) = 18)))
  ∧ (∀ y : ℝ, y > 6 → y^2 / (y - 6) ≥ 18) :=
by {
  sorry
}

end minimum_value_fraction_l137_137505
