import Mathlib
import Mathlib.
import Mathlib.Algebra
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Floor
import Mathlib.Algebra.GCD.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupBasic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics
import Mathlib.Combinatorics.CombinatorialLemmas.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sum
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Vector.Basic
import Mathlib.Field.Basic
import Mathlib.Geometry.Euclid_Relations
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.Angle
import Mathlib.LinearAlgebra.Matrix
import Mathlib.MeasureTheory
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic

namespace four_possible_x_values_l754_754292

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l754_754292


namespace probability_non_adjacent_sum_l754_754231

-- Definitions and conditions from the problem
def total_trees := 13
def maple_trees := 4
def oak_trees := 3
def birch_trees := 6

-- Total possible arrangements of 13 trees
def total_arrangements := Nat.choose total_trees birch_trees

-- Number of ways to arrange birch trees with no two adjacent
def favorable_arrangements := Nat.choose (maple_trees + oak_trees + 1) birch_trees

-- Probability calculation
def probability_non_adjacent := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

-- This value should be simplified to form m/n in lowest terms
def fraction_part_m := 7
def fraction_part_n := 429

-- Verify m + n
def sum_m_n := fraction_part_m + fraction_part_n

-- Check that sum_m_n is equal to 436
theorem probability_non_adjacent_sum :
  sum_m_n = 436 := by {
    -- Placeholder proof
    sorry
}

end probability_non_adjacent_sum_l754_754231


namespace sphere_surface_area_of_circumscribing_cuboid_l754_754228

theorem sphere_surface_area_of_circumscribing_cuboid :
  ∀ (a b c : ℝ), a = 5 ∧ b = 4 ∧ c = 3 → 4 * Real.pi * ((Real.sqrt ((a^2 + b^2 + c^2)) / 2) ^ 2) = 50 * Real.pi :=
by
  -- introduction of variables and conditions
  intros a b c h
  obtain ⟨_, _, _⟩ := h -- decomposing the conditions
  -- the proof is skipped
  sorry

end sphere_surface_area_of_circumscribing_cuboid_l754_754228


namespace smallest_value_of_x_l754_754202

theorem smallest_value_of_x (x : ℝ) (h : 4 * x^2 - 20 * x + 24 = 0) : x = 2 :=
    sorry

end smallest_value_of_x_l754_754202


namespace train_speed_correct_l754_754248

def length_train : ℝ := 100
def length_bridge : ℝ := 250
def time_to_cross : ℝ := 34.997200223982084

def total_distance : ℝ := length_train + length_bridge
def speed_m_per_s : ℝ := total_distance / time_to_cross
def speed_kmph : ℝ := speed_m_per_s * 3.6

theorem train_speed_correct :
  speed_kmph = 36.0008228577942852 :=
by
  -- Definitions and calculations can be used to show the expected result.
  sorry

end train_speed_correct_l754_754248


namespace possible_number_of_circles_l754_754184

/--
In a plane, given three distinct lines, the possible number of circles tangent to all three lines
can only be one of the following: 0, 2, or 4.
-/
theorem possible_number_of_circles (L1 L2 L3 : set (ℝ × ℝ)) (h1 : L1 ≠ L2) (h2 : L2 ≠ L3) (h3 : L1 ≠ L3) :
  ∃ n : ℕ, (n = 0 ∨ n = 2 ∨ n = 4) := 
sorry

end possible_number_of_circles_l754_754184


namespace negative_half_less_than_negative_third_l754_754275

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l754_754275


namespace limit_sequence_eq_five_l754_754216

noncomputable def limit_sequence : ℝ :=
  real.limit (λ n : ℕ, (n * real.sqrt (real.nat_abs n) + real.sqrt (25 * real.nat_abs (n ^ 4) - 81)) / 
                         ((n - 7 * real.sqrt n) * real.sqrt (n ^ 2 - n + 1)))

theorem limit_sequence_eq_five : limit_sequence = 5 := 
begin
  sorry
end

end limit_sequence_eq_five_l754_754216


namespace clerical_ratio_l754_754226

theorem clerical_ratio (total_employees clerical_employees : ℕ)
  (total_employees = 3600)
  (h : clerical_employees / 2 = 0.2 * (total_employees - clerical_employees / 2)) :
  clerical_employees / total_employees = 1 / 3 :=
by
   sorry

end clerical_ratio_l754_754226


namespace equal_segments_l754_754956

variables (A B1 B2 C1 C2 I1 I2 D E F : Type)
variables [Incenter A B1 C1 I1] [Incenter A B2 C2 I2]
variables [SimilarTriangles A B1 C1 A B2 C2]
variables [Intersect B1 B2 C1 C2 D]
variables [Intersect I1 I2 B1 B2 E]
variables [Intersect I1 I2 C1 C2 F]

theorem equal_segments : E = F → (D.E = D.F) :=
begin
  sorry
end

end equal_segments_l754_754956


namespace union_cardinality_inequality_l754_754440

open Set

/-- Given three finite sets A, B, and C such that A ∩ B ∩ C = ∅,
prove that |A ∪ B ∪ C| ≥ 1/2 (|A| + |B| + |C|) -/
theorem union_cardinality_inequality (A B C : Finset ℕ)
  (h : (A ∩ B ∩ C) = ∅) : (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := sorry

end union_cardinality_inequality_l754_754440


namespace hyperbola_eccentricity_range_l754_754436

theorem hyperbola_eccentricity_range
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (x y : ℝ) (P_on_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (e : ℝ) (eccentricity_def : e = (sin (angle P F2 F1)) / (sin (angle P F1 F2))) :
  1 < e ∧ e ≤ sqrt 2 + 1 := by
  sorry

end hyperbola_eccentricity_range_l754_754436


namespace distinct_solutions_difference_l754_754999

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : r > s)
  (h_eq : ∀ x : ℝ, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
begin
  sorry
end

end distinct_solutions_difference_l754_754999


namespace overall_percentage_of_games_won_l754_754247

theorem overall_percentage_of_games_won :
  ∀ (g1 g2 : ℕ), 
      let total_games := 119.9999999999999 in
      let first_games := 30 in 
      let won_first_games := 0.40 * first_games in 
      let remaining_games := total_games - first_games in
      let won_remaining_games := 0.80 * remaining_games in
      g1 = nat.ceil(won_first_games) ->
      g2 = nat.ceil(won_remaining_games) ->
      ((g1 + g2) / total_games) * 100 = 70 :=
by
  sorry

end overall_percentage_of_games_won_l754_754247


namespace volume_of_circumscribed_sphere_l754_754243

-- Definitions based on the conditions
def length : ℝ := 1
def width : ℝ := 2
def height : ℝ := 3

-- Lean statement to verify the volume of the circumscribing sphere
theorem volume_of_circumscribed_sphere : 
  let space_diagonal := Real.sqrt (length^2 + width^2 + height^2),
      radius := space_diagonal / 2,
      volume := (4 / 3) * Real.pi * radius^3 in
  volume = (7 * Real.sqrt 14) / 3 * Real.pi :=
by sorry

end volume_of_circumscribed_sphere_l754_754243


namespace polar_explorer_distance_l754_754724

noncomputable def total_distance_travelled (v d : ℝ) : ℝ :=
  d = 320

theorem polar_explorer_distance :
  ∀ (v : ℝ), (0 < v) →
  let new_speed := (3 / 5) * v in
  let d1 := 24 * v in
  let delay_dogs := (d1 + 120) / v + (320 - (d1 + 120)) / new_speed - (d1 / v + (320 - d1) / new_speed)  = 24 in
  let total_delay := (d1 / v + (320 - d1) / new_speed) = (d1 / v + (320 - d1) / new_speed) + 48 in
  total_distance_travelled v 320 :=
begin
  sorry
end

end polar_explorer_distance_l754_754724


namespace complex_z_sub_conjugate_eq_neg_i_l754_754844

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754844


namespace george_more_apples_than_amelia_l754_754808

theorem george_more_apples_than_amelia :
  ∀ (A_G A_A G_O A_O : ℕ),
    A_G + A_O + A_A + A_O = 107 →
    G_O = 45 →
    A_O = 45 - 18 →
    A_A = 15 →
    A_G = 20 →
    (A_G - A_A) = 5 :=
by
  intros A_G A_A G_O A_O h1 h2 h3 h4 h5
  simp [h4, h5]
  sorry

end george_more_apples_than_amelia_l754_754808


namespace trisha_take_home_pay_l754_754656

theorem trisha_take_home_pay
  (hourly_pay : ℝ := 15)
  (hours_per_week : ℝ := 40)
  (weeks_per_year : ℝ := 52)
  (withholding_percentage : ℝ := 0.20) :
  let annual_gross_pay := hourly_pay * hours_per_week * weeks_per_year,
      amount_withheld := annual_gross_pay * withholding_percentage,
      annual_take_home_pay := annual_gross_pay - amount_withheld
  in annual_take_home_pay = 24960 := by
    sorry

end trisha_take_home_pay_l754_754656


namespace find_angle_A_l754_754961

-- Variables representing angles A and B
variables (A B : ℝ)

-- The conditions of the problem translated into Lean
def angle_relationship := A = 2 * B - 15
def angle_supplementary := A + B = 180

-- The theorem statement we need to prove
theorem find_angle_A (h1 : angle_relationship A B) (h2 : angle_supplementary A B) : A = 115 :=
by { sorry }

end find_angle_A_l754_754961


namespace round_to_thousandth_l754_754606

-- Define a theorem that states rounding 2.00956 to the nearest thousandth results in 2.010.
theorem round_to_thousandth (x : ℝ) (h : x = 2.00956) : Real.round_to 0.001 x = 2.010 := 
by sorry

end round_to_thousandth_l754_754606


namespace z_conjugate_difference_l754_754855

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754855


namespace expected_non_allergic_l754_754580

theorem expected_non_allergic (p : ℝ) (n : ℕ) (h : p = 1 / 4) (hn : n = 300) : n * p = 75 :=
by sorry

end expected_non_allergic_l754_754580


namespace AEFD_cyclic_l754_754559

-- Define the existence of an isosceles trapezoid ABCD
variable (A B C D X E F: Type)
variable [IsoscelesTrapezoid A B C D] -- Define the isosceles trapezoid with AD > BC
variable (X_def : Intersection (AngleBisector A B C) B C = X)
variable (E_def : Intersection (LineThroughParallel D B (AngleBisector D B C)) X = E)
variable (F_def : Intersection (LineThroughParallel D C (AngleBisector D C B)) X = F)

-- The theorem statement that AEFD is a cyclic quadrilateral
theorem AEFD_cyclic {A B C D X E F : Type}
  [IsoscelesTrapezoid A B C D] -- ABCD is an isosceles trapezoid
  (h1 : Intersection (AngleBisector A B C) B C = X) -- X is defined
  (h2 : Intersection (LineThroughParallel D B (AngleBisector D B C)) X = E) -- E is defined
  (h3 : Intersection (LineThroughParallel D C (AngleBisector D C B)) X = F) -- F is defined
  : CyclicQuadrilateral A E F D :=
begin
  sorry
end

end AEFD_cyclic_l754_754559


namespace max_integer_sums_l754_754785

-- Define the 10 cards with real numbers
variables {cards : Finset ℝ} (h_card_count : cards.card = 10)

-- We're given the condition that not all subset sums are integers
def not_all_sums_integers : Prop := ∃ C : Finset ℝ, C ⊆ cards ∧ ¬ ∃ (C' ⊆ cards), (C.sum id).den = 1

-- Theorem statement
theorem max_integer_sums (h_not_all_sums_integers : not_all_sums_integers cards) : 
  ∃ n : ℕ, n = 511 := 
sorry

end max_integer_sums_l754_754785


namespace range_of_a_l754_754872

variable (a : ℝ)

-- Definitions of propositions p and q
def p := ∀ x : ℝ, x^2 - 2*x - a ≥ 0
def q := ∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0

-- Lean 4 statement of the proof problem
theorem range_of_a : ¬ p a ∧ q a → -1 < a ∧ a ≤ 5/8 := by
  sorry

end range_of_a_l754_754872


namespace find_amounts_l754_754244

noncomputable def amounts_of_solution (x y : ℝ) : Prop :=
  (x + y = 140) ∧ (0.40 * x + 0.90 * y = 112)

theorem find_amounts :
  ∃ x y : ℝ, amounts_of_solution x y ∧ x = 28 ∧ y = 112 :=
by {
  use [28, 112],
  split,
  { unfold amounts_of_solution, 
    split;
    norm_num, },
  split; norm_num,
}

end find_amounts_l754_754244


namespace minimum_elements_l754_754903

open Finset

-- Definitions corresponding to conditions
def odd_functions (A : Finset ℝ) : Finset ℝ := sorry
def increasing_functions (A : Finset ℝ) : Finset ℝ := sorry
def passing_through_origin (A : Finset ℝ) : Finset ℝ := sorry

-- Conditions
variable {A : Finset ℝ}
variable h_odd : odd_functions A = range 10
variable h_increasing : increasing_functions A = range 8
variable h_origin : passing_through_origin A = range 12

-- Equivalent proof problem statement
theorem minimum_elements (A : Finset ℝ) :
  (odd_functions A).card = 10 →
  (increasing_functions A).card = 8 →
  (passing_through_origin A).card = 12 →
  (A.card ≥ 14) :=
by
  intros ho hi hp
  sorry

end minimum_elements_l754_754903


namespace sequence_term_3001_exists_exactly_4_values_l754_754322

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l754_754322


namespace compound_propositions_truth_l754_754900

variable (x a b : ℝ)
def p : Prop := (x^2 - 3 * x + 2 = 0) → (x = 1)
def q : Prop := (a ^ (1/2) > b ^ (1/2)) ↔ (Real.log a > Real.log b)

theorem compound_propositions_truth :
  (p ∨ q) ∧ (¬p ∨ ¬q) ∧ (p ∧ ¬q) :=
by
  have hp : p := sorry -- Proof of p is true
  have hq : ¬q := sorry -- Proof that q is false
  exact ⟨or.inl hp, or.inr hq, and.intro hp hq⟩

end compound_propositions_truth_l754_754900


namespace two_colorable_G2000_l754_754938

-- Define a graph with vertices and edges
structure Graph (V : Type) :=
  (E : set (V × V))

def is_2_colorable {V : Type} (G : Graph V) : Prop :=
  ∃ (coloring : V → Prop), 
    ∀ (v u : V), (v, u) ∈ G.E → coloring v ≠ coloring u

-- Define our specific instance with 2000 vertices
constant V2000 : Type
constant vertices_2000 : fintype V2000
constant edges_2000 : set (V2000 × V2000)

def G2000 : Graph V2000 := { E := edges_2000 }

theorem two_colorable_G2000 : is_2_colorable G2000 :=
sorry

end two_colorable_G2000_l754_754938


namespace sum_of_natural_numbers_l754_754623

noncomputable def number_of_ways (n : ℕ) : ℕ :=
  2^(n-1)

theorem sum_of_natural_numbers (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, k = number_of_ways n :=
by
  use 2^(n-1)
  sorry

end sum_of_natural_numbers_l754_754623


namespace triangle_BA_plus_BD_eq_AC_l754_754410

theorem triangle_BA_plus_BD_eq_AC (A B C D : Point) (hABC : Triangle A B C)
  (hAngleB : ∠B = 3 * ∠C) (hAngleBDC : ∠BDC = 2 * ∠C) :
  distance B D + distance B A = distance A C := 
sorry

end triangle_BA_plus_BD_eq_AC_l754_754410


namespace sample_variance_is_two_l754_754638

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : (1 / 5) * ((-1 - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end sample_variance_is_two_l754_754638


namespace bettys_parents_gift_l754_754758

def parents_gift := 15

theorem bettys_parents_gift :
  ∀ (wallet_cost initial_money additional_money parents_gift : ℕ),
    wallet_cost = 100 ∧
    initial_money = wallet_cost / 2 ∧
    additional_money = 5 ∧
    (initial_money + parents_gift + 2 * parents_gift = wallet_cost - additional_money) →
    parents_gift = 15 :=
by
  intros wallet_cost initial_money additional_money P
  intro h
  simp at h
  cases h
  sorry

end bettys_parents_gift_l754_754758


namespace find_a_l754_754463

open Complex

theorem find_a (a : ℝ) (h : (2 + Complex.I * a) / (1 + Complex.I * Real.sqrt 2) = -Complex.I * Real.sqrt 2) :
  a = Real.sqrt 2 := by
  sorry

end find_a_l754_754463


namespace range_of_a_l754_754041

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + a ≤ 1) → a ∈ ℝ := 
begin
  -- proof is not required
  sorry
end

end range_of_a_l754_754041


namespace find_s_l754_754548

theorem find_s (s : ℝ) :
  let P := (s - 3, 2)
  let Q := (1, s + 2)
  let M := ((s - 2) / 2, (s + 4) / 2)
  let dist_sq := (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2
  dist_sq = 3 * s^2 / 4 →
  s = -5 + 5 * Real.sqrt 2 ∨ s = -5 - 5 * Real.sqrt 2 :=
by
  intros P Q M dist_sq h
  sorry

end find_s_l754_754548


namespace range_frac_sum_l754_754630

variable {a b : ℝ}

theorem range_frac_sum (h : a + b = 2) : 
  set.range (λ x : {a : ℝ // a ≠ 0 ∧ a ≠ 2}, (1 / x.1) + (4 / (2 - x.1))) = 
  set.Iic (1 / 2) ∪ set.Ici (9 / 2) :=
sorry

end range_frac_sum_l754_754630


namespace transform_sinusoidal_l754_754599

noncomputable def f (x : ℝ) : ℝ := sin (4 * x + π / 3)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x)

theorem transform_sinusoidal :
  (∀ x, g (x + π / 12) = sin (2 * (x + π / 12))) ∧
  (∀ x, sin (2 * (x + π / 12)) = sin (2 * x + π / 6)) ∧
  (∀ x, f (x / 2 - π / 12) = g (x / 2 - π / 12))
  → ∃ f, f x = sin (4 * x + π / 3) :=
by
  sorry

end transform_sinusoidal_l754_754599


namespace trigonometric_identity_l754_754761

theorem trigonometric_identity :
  Real.sin (17 * Real.pi / 180) * Real.sin (223 * Real.pi / 180) + 
  Real.sin (253 * Real.pi / 180) * Real.sin (313 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l754_754761


namespace parry_position_probability_l754_754048

theorem parry_position_probability :
    let total_members := 20
    let positions := ["President", "Vice President", "Secretary", "Treasurer"]
    let remaining_for_secretary := 18
    let remaining_for_treasurer := 17
    let prob_parry_secretary := (1 : ℚ) / remaining_for_secretary
    let prob_parry_treasurer_given_not_secretary := (1 : ℚ) / remaining_for_treasurer
    let overall_probability := prob_parry_secretary + prob_parry_treasurer_given_not_secretary * (remaining_for_treasurer / remaining_for_secretary)
    overall_probability = (1 : ℚ) / 9 := 
by
  sorry

end parry_position_probability_l754_754048


namespace num_factors_of_1806_with_exactly_4_factors_l754_754916

theorem num_factors_of_1806_with_exactly_4_factors : 
  (∀ n : ℕ, n > 0 ∧ n ∣ 1806 ∧ (nat.factors_count n = 4) → ∃! n_list : list ℕ, n_list.length = 3) :=
sorry

end num_factors_of_1806_with_exactly_4_factors_l754_754916


namespace pascal_even_rows_count_l754_754027

open Nat

/-- The number of rows in the first 30 rows of Pascal's triangle, 
excluding row 0 and row 1, that have exclusively even numbers 
(excluding the 1 at each end) is 4. -/
theorem pascal_even_rows_count : 
  let rows := [2, 4, 8, 16]
  rows.count (λ n, n < 30) = 4 := 
  by
    have h1: rows = [2, 4, 8, 16] := rfl
    have h2 : ∀ r, r ∈ rows → r < 30 := by
      intro r hr
      cases hr
      · exact by decide
      cases hr
      · exact by decide
      cases hr
      · exact by decide
      cases hr
      · exact by decide
      contradiction
    simp only [List.count, List.mem_cons, List.mem_nil, if_true] at *
    exact congr_arg length (List.map_ext _ (λ x hx, by simp [hx]) _)
  sorry

end pascal_even_rows_count_l754_754027


namespace problem1_problem2_problem3_l754_754000

-- Define the function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b - 2^x) / (2^(x + 1) + a)

-- Problem 1
theorem problem1 (h_odd : ∀ x, f x a b = -f (-x) a b) : a = 2 ∧ b = 1 :=
sorry

-- Problem 2
theorem problem2 : (∀ x, f x 2 1 = -f (-x) 2 1) → ∀ x y, x < y → f x 2 1 > f y 2 1 :=
sorry

-- Problem 3
theorem problem3 (h_pos : ∀ x ≥ 1, f (k * 3^x) 2 1 + f (3^x - 9^x + 2) 2 1 > 0) : k < 4 / 3 :=
sorry

end problem1_problem2_problem3_l754_754000


namespace polynomial_perfect_square_l754_754168

theorem polynomial_perfect_square (m : ℤ) : (∃ a : ℤ, a^2 = 25 ∧ x^2 + m*x + 25 = (x + a)^2) ↔ (m = 10 ∨ m = -10) :=
by sorry

end polynomial_perfect_square_l754_754168


namespace percentage_increase_in_radius_l754_754716

theorem percentage_increase_in_radius (r R : ℝ) (h : π * R^2 = π * r^2 + 1.25 * (π * r^2)) :
  R = 1.5 * r :=
by
  -- Proof goes here
  sorry

end percentage_increase_in_radius_l754_754716


namespace relationship_between_M_and_N_l754_754017

-- Definitions
def M : set ℤ := {-1, 0, 1}

def N : set ℤ := {x | ∃ a b : ℤ, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

-- The statement to be proved
theorem relationship_between_M_and_N : M ∩ N = N := 
sorry

end relationship_between_M_and_N_l754_754017


namespace smallest_n_for_divisibility_l754_754201

theorem smallest_n_for_divisibility (n : ℕ) (h1 : 24 ∣ n^2) (h2 : 1080 ∣ n^3) : n = 120 :=
sorry

end smallest_n_for_divisibility_l754_754201


namespace ellipse_area_irrational_l754_754899

noncomputable def ellipse_area (a b : ℚ) : ℝ :=
  Real.pi * a * b

theorem ellipse_area_irrational 
  (a b : ℚ) : irrational (ellipse_area a b) :=
by {
  let A := ellipse_area a b,
  have h_rat : a * b ∈ ℚ := mul_rat (a) (b),
  have h_pi_irr : irrational Real.pi := Real.irrational_pi,
  exact irrational.mul_rat h_pi_irr h_rat,
  sorry
}

end ellipse_area_irrational_l754_754899


namespace complex_conjugate_difference_l754_754821

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754821


namespace sum_x_y_is_neg2_l754_754471

def equation1 (x : ℝ) : Prop := 2^x + 4 * x + 12 = 0
def equation2 (y : ℝ) : Prop := Real.log2 ((y - 1)^3) + 3 * y + 12 = 0

theorem sum_x_y_is_neg2 (x y : ℝ) (h1 : equation1 x) (h2 : equation2 y) : x + y = -2 :=
by
  sorry

end sum_x_y_is_neg2_l754_754471


namespace digit_B_identification_l754_754071

theorem digit_B_identification (B : ℕ) 
  (hB_range : 0 ≤ B ∧ B < 10) 
  (h_units_digit : (5 * B % 10) = 5) 
  (h_product : (10 * B + 5) * (90 + B) = 9045) : 
  B = 9 :=
sorry

end digit_B_identification_l754_754071


namespace num_valid_pairs_l754_754234

def valid_pairs (m n : ℤ) : Prop :=
  let set_mn := {-3, -2, -1, 0, 1, 2, 3}
  m ∈ set_mn ∧ n ∈ set_mn ∧ m ≠ n ∧ (n^2 < (5/4)m^2)

theorem num_valid_pairs : 
  (finset.univ.filter (λ mn : ℤ × ℤ,
    valid_pairs mn.1 mn.2)).card = 18 := 
sorry

end num_valid_pairs_l754_754234


namespace net_gain_mr_A_l754_754572

def home_worth : ℝ := 12000
def sale1 : ℝ := home_worth * 1.2
def sale2 : ℝ := sale1 * 0.85
def sale3 : ℝ := sale2 * 1.1

theorem net_gain_mr_A : sale1 - sale2 + sale3 = 3384 := by
  sorry -- Proof will be provided here

end net_gain_mr_A_l754_754572


namespace marble_probability_is_correct_l754_754806

def marbles_probability
  (total_marbles: ℕ) 
  (red_marbles: ℕ) 
  (blue_marbles: ℕ) 
  (green_marbles: ℕ)
  (choose_marbles: ℕ) 
  (required_red: ℕ) 
  (required_blue: ℕ) 
  (required_green: ℕ): ℚ := sorry

-- Define conditions
def total_marbles := 7
def red_marbles := 3
def blue_marbles := 2
def green_marbles := 2
def choose_marbles := 4
def required_red := 2
def required_blue := 1
def required_green := 1

-- Proof statement
theorem marble_probability_is_correct : 
  marbles_probability total_marbles red_marbles blue_marbles green_marbles choose_marbles required_red required_blue required_green = (12 / 35 : ℚ) :=
sorry

end marble_probability_is_correct_l754_754806


namespace no_solution_to_a_l754_754597

theorem no_solution_to_a (x : ℝ) :
  (4 * x - 1) / 6 - (5 * x - 2 / 3) / 10 + (9 - x / 2) / 3 ≠ 101 / 20 := 
sorry

end no_solution_to_a_l754_754597


namespace percentile_75th_is_39_l754_754485

-- Given the conditions
def scores : List ℕ := [29, 30, 38, 25, 37, 40, 42, 32]
def n : ℕ := scores.length

-- Definition of the 75th percentile function
def percentile (p : ℚ) (xs : List ℕ) : ℚ :=
  let xs_sorted := xs.qsort (≤)
  let pos := p * xs_sorted.length
  if pos.floor = pos then
    (Rational.of_nat (xs_sorted.nth_le (pos.to_nat - 1) sorry))
  else
    let lower := xs_sorted.nth_le (pos.floor.to_nat - 1) sorry
    let upper := xs_sorted.nth_le (pos.ceil.to_nat - 1) sorry
    (lower + upper) / 2

-- Statement to prove
theorem percentile_75th_is_39 : percentile (3 / 4) scores = (39 : ℚ) :=
by
  sorry

end percentile_75th_is_39_l754_754485


namespace negation_universal_proposition_l754_754622

theorem negation_universal_proposition :
  ¬(∀ x : ℝ, x > 1 → (1/2)^x < 1/2) ↔ ∃ x_0 : ℝ, x_0 > 1 ∧ (1/2)^x_0 ≥ 1/2 :=
by
  sorry

end negation_universal_proposition_l754_754622


namespace prod_is_96_l754_754372

noncomputable def prod_of_nums (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : ℝ := x * y

theorem prod_is_96 (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : prod_of_nums x y h1 h2 = 96 :=
by
  sorry

end prod_is_96_l754_754372


namespace plane_does_not_intersect_interior_of_tetrahedron_l754_754185

theorem plane_does_not_intersect_interior_of_tetrahedron :
  let V := 4 -- number of vertices in a tetrahedron
  let C := (finset.powerset_len 3 (finset.range V)).card -- combination of choosing 3 out of 4 vertices
  let favorable :=
    (finset.filter (λ s, ∃ vs : finset ℝ, vs.card = 3 ∧ s = vs) (finset.powerset_len 3 (finset.range V))).card -- sets of vertices which do not intersect the interior (which is all combinations)
  C = favorable :=
  by
  have hV : V = 4 := rfl -- four vertices in a tetrahedron
  have hC : C = 4 := by
    rw finset.powerset_len,
    rw finset.card,
    sorry -- combinatorial calculation here
  have hf : favorable = 4 := by
    sorry -- all combinations are favorable
  rw [hC, hf],
  exact rfl

end plane_does_not_intersect_interior_of_tetrahedron_l754_754185


namespace polar_to_line_distance_l754_754070

theorem polar_to_line_distance : 
  let point_polar := (2, Real.pi / 3)
  let line_polar := (2, 0)  -- Corresponding (rho, theta) for the given line
  let point_rect := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
  let line_rect := 2  -- x = 2 in rectangular coordinates
  let distance := abs (line_rect - point_rect.1)
  distance = 1 := by
{
  sorry
}

end polar_to_line_distance_l754_754070


namespace equal_side_length_is_4_or_10_l754_754151

-- Define the conditions
def isosceles_triangle (base_length equal_side_length : ℝ) :=
  base_length = 7 ∧
  (equal_side_length > base_length ∧ equal_side_length - base_length = 3) ∨
  (equal_side_length < base_length ∧ base_length - equal_side_length = 3)

-- Lean 4 statement to prove
theorem equal_side_length_is_4_or_10 (base_length equal_side_length : ℝ) 
  (h : isosceles_triangle base_length equal_side_length) : 
  equal_side_length = 4 ∨ equal_side_length = 10 :=
by 
  sorry

end equal_side_length_is_4_or_10_l754_754151


namespace cos_C_of_triangle_l754_754513

theorem cos_C_of_triangle (A B C : ℝ) (hA : sin A = 4 / 5) (hB : cos B = 12 / 13) :
  cos C = -16 / 65 :=
sorry

end cos_C_of_triangle_l754_754513


namespace problem1_problem2_l754_754334

def f (x : ℝ) : ℝ := abs (x + 4)

theorem problem1 (a : ℝ) :
  (∀ x : ℝ, f (2*x + a) + f (2*x - a) ≥ 4) → (a = 2 ∨ a = -2) :=
by
  intro h
  sorry

theorem problem2 (x : ℝ) :
  f x > 1 - (1/2) * x ↔ (x > -2 ∨ x < -10) :=
by
  intro h
  sorry

end problem1_problem2_l754_754334


namespace percentage_female_officers_on_duty_l754_754123

/-- There were 300 officers on duty one night,
and half of these 300 officers were female.
The total number of female officers on the police force is 1000.
Prove that the percentage of female officers on duty that night was 15%. --/
theorem percentage_female_officers_on_duty
  (total_on_duty : ℕ)
  (half_female_on_duty : total_on_duty % 2 = 0 ∧ ((total_on_duty / 2) : ℕ))
  (total_female_officers : ℕ) :
  total_on_duty = 300 →
  half_female_on_duty = (total_on_duty / 2, sorry) →
  total_female_officers = 1000 →
  (total_on_duty / 2 : ℕ) * 100 / total_female_officers = 15 := 
begin
  sorry
end

end percentage_female_officers_on_duty_l754_754123


namespace hexagonal_coloring_l754_754771

def color_count : ℕ :=
  let hexagons := 7
  let colors := { blue, orange, red, green }
  let center_color := blue
  4

theorem hexagonal_coloring :
  ∀ (θ : color_count = 4),
  color_count = 4 :=
by sorry

end hexagonal_coloring_l754_754771


namespace sum_a_1_to_100_l754_754400

noncomputable def f (n : ℕ) : ℝ :=
  n^2 * Real.cos (n * Real.pi)

noncomputable def a (n : ℕ) : ℝ :=
  f n + f (n + 1)

theorem sum_a_1_to_100 :
  (∑ n in Finset.range 100, a (n + 1)) = -100 :=
by
  sorry

end sum_a_1_to_100_l754_754400


namespace z_conjugate_difference_l754_754856

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754856


namespace ellipse_equation_from_hyperbola_l754_754412

theorem ellipse_equation_from_hyperbola {
  (focus_condition : ∀ (a b : ℝ), 2 * a^2 - 2 * b^2 = 1 → a = sqrt (1 / 2) → b = sqrt (1 / 2) → 
  ∃ c : ℝ, c = sqrt 2 ∧ (a^2 + b^2 = c^2))
  (eccentricity_condition : ∀ (e1 e2 : ℝ), e1 = sqrt 2 → e2 = 1 / (sqrt 2) → true)
} :
  ∃ (x y : ℝ → ℝ), (∀ (a b : ℝ), a = sqrt 2 → b = 1 → 1 = x^2 / a + y^2 / b) → 
  ∃ (a b : ℝ), a = sqrt 2 ∧ b = 1 ∧ (x^2 / a + y^2 / b = 1) :=
by
  sorry

end ellipse_equation_from_hyperbola_l754_754412


namespace find_point_A_l754_754491

noncomputable def area_of_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

noncomputable def exists_point_A (K O Z : ℝ × ℝ) (area_Q : ℝ) : Prop :=
  ∃ (A : ℝ × ℝ), quadrilateral_area K O Z A = area_Q

theorem find_point_A : 
  let K := (0, 1)
  let O := (3, 0)
  let Z := (4, 3)
  exists_point_A K O Z 4 :=
sorry

end find_point_A_l754_754491


namespace range_of_x0_l754_754404

variable (n : ℕ) (a b p : ℝ)
variable (x : Fin (n+1) → ℝ)

-- Define the conditions: the sums involved.
def sum_eq_a := ∑ i in Finset.range (n+1), x i = a
def sum_squares_eq_b := ∑ i in Finset.range (n+1), (x i)^2 = b

-- Proving the possible range of x₀ given those conditions.
theorem range_of_x0 (h_sum : sum_eq_a n a x) (h_sum_squares : sum_squares_eq_b n b x) : 
0 ≤ x 0 ∧ x 0 ≤ 2 * p / n := 
by 
  sorry

end range_of_x0_l754_754404


namespace find_minimum_value_of_f_l754_754098

def f (x : ℝ) : ℝ := (x ^ 2 + 4 * x + 5) * (x ^ 2 + 4 * x + 2) + 2 * x ^ 2 + 8 * x + 1

theorem find_minimum_value_of_f : ∃ x : ℝ, f x = -9 :=
by
  sorry

end find_minimum_value_of_f_l754_754098


namespace locus_of_points_l754_754868

theorem locus_of_points (ABC : Triangle) (equilateral : is_equilateral ABC) :
  let P := Point in
  (distance_from_point_to_side P AB = geometric_mean (distance_from_point_to_side P BC) (distance_from_point_to_side P CA)) ->
  exists arc, (arc_subtended_by_angle P AB = 120) :=
sorry

end locus_of_points_l754_754868


namespace sugar_flour_difference_l754_754568

theorem sugar_flour_difference :
  ∀ (flour_required_kg sugar_required_lb flour_added_kg kg_to_lb),
    flour_required_kg = 2.25 →
    sugar_required_lb = 5.5 →
    flour_added_kg = 1 →
    kg_to_lb = 2.205 →
    (sugar_required_lb / kg_to_lb * 1000) - ((flour_required_kg - flour_added_kg) * 1000) = 1244.8 :=
by
  intros flour_required_kg sugar_required_lb flour_added_kg kg_to_lb
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- sorry is used to skip the actual proof
  sorry

end sugar_flour_difference_l754_754568


namespace integer_k_values_l754_754004

theorem integer_k_values (a b k : ℝ) (m : ℝ) (ha : a > 0) (hb : b > 0) (hba_int : ∃ n : ℤ, n ≠ 0 ∧ b = (n : ℝ) * a) 
  (hA : a = a * k + m) (hB : 8 * b = b * k + m) : k = 9 ∨ k = 15 := 
by
  sorry

end integer_k_values_l754_754004


namespace average_headcount_l754_754664

theorem average_headcount 
  (h1 : ℕ := 11500) 
  (h2 : ℕ := 11600) 
  (h3 : ℕ := 11300) : 
  (Float.round ((h1 + h2 + h3 : ℕ : Float) / 3) = 11467) :=
sorry

end average_headcount_l754_754664


namespace four_possible_x_values_l754_754294

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l754_754294


namespace shorter_diagonal_length_l754_754187

theorem shorter_diagonal_length (EF GH EG FH : ℝ) (acute_E : Prop) (acute_F : Prop)
  (h1 : EF = 40) (h2 : GH = 28) (h3 : EG = 13) (h4 : FH = 15) : 
  ∃ (d : ℝ), d = 27 ∧ 
    (∃ a b c d1 d2 : ℝ, 
       EG = a ∧ 
       FH = b ∧ 
       (a^2 + d1^2 = EG^2) ∧ 
       ((12 - a)^2 + d1^2 = FH^2) ∧ 
       (c^2 + d2^2 = d) ∧ 
       d = min (sqrt ((a + 28)^2 + d1^2)) (sqrt ((12 - a)^2 + d1^2))) :=
sorry

end shorter_diagonal_length_l754_754187


namespace triangle_side_relation_l754_754812

variable {ABC A'B'C' : Type} -- representing the triangles
variables {a b c a' b' c' : ℝ} -- side lengths
variable {B B' A A' : ℝ} -- angles

-- conditions
axiom angle_B_eq_angle_B' : B = B'
axiom angle_A_plus_angle_A'_eq_180 : A + A' = 180

theorem triangle_side_relation
  (triangle_ABC : Triangle ABC a b c)
  (triangle_A'B'C' : Triangle A'B'C' a' b' c')
  (h1 : B = B')
  (h2 : A + A' = 180) :
  a * a' = b * b' + c * c' :=
sorry

end triangle_side_relation_l754_754812


namespace am_gm_product_ineq_l754_754561

open Real

theorem am_gm_product_ineq (n : ℕ) (a : Fin n → ℝ) (h₁ : ∀ i, 0 < a i) (h₂ : (∏ i in Finset.univ, a i) = 1) :
  (∏ i in Finset.univ, (1 + a i)) ≥ 2^n := by
  sorry

end am_gm_product_ineq_l754_754561


namespace volume_ratio_l754_754132

theorem volume_ratio : 
  let side_cube := 1 * 100 in
  let width_cuboid := 50 in
  let depth_cuboid := 50 in
  let height_cuboid := 20 in
  let volume_cube := side_cube ^ 3 in
  let volume_cuboid := width_cuboid * depth_cuboid * height_cuboid in
  volume_cube / volume_cuboid = 20 :=
by
  sorry

end volume_ratio_l754_754132


namespace cos_of_complementary_angle_l754_754964

theorem cos_of_complementary_angle (X Y Z : ℝ) (hX : X = 90) (hSinY : sin Y = 3 / 5) : cos Z = 3 / 5 :=
by
  -- conditions: X = 90°, sin Y = 3/5
  -- goal: cos Z = 3/5
  sorry

end cos_of_complementary_angle_l754_754964


namespace largest_perimeter_l754_754912

-- Definitions of the given conditions and the proof problem.
noncomputable def base : ℝ := 10
noncomputable def height_0 : ℝ := 12
noncomputable def height_5 : ℝ := 15
noncomputable def height_10 : ℝ := 17

def h (k : ℕ) : ℝ :=
  if k ≤ 5 then height_0 + (3 * k / 5)
  else height_5 + (2 * (k - 5) / 5)

def perimeter (k : ℕ) : ℝ :=
  1 + real.sqrt (h k ^ 2 + k ^ 2) + real.sqrt (h (k + 1) ^ 2 + (k + 1) ^ 2)

theorem largest_perimeter : ∃ k, 0 ≤ k ∧ k ≤ 9 ∧ perimeter k = 39.78 := sorry

end largest_perimeter_l754_754912


namespace amount_each_parent_has_to_pay_l754_754715

-- Definitions using the conditions from the problem
def old_salary : ℝ := 60000
def raise_percentage : ℝ := 0.25
def tax_rate : ℝ := 0.10
def number_of_kids : ℝ := 15
def exchange_rate_usd_to_eur : ℝ := 0.85
def exchange_rate_usd_to_gbp : ℝ := 0.75
def exchange_rate_usd_to_jpy : ℝ := 110

-- Question we want to prove with the answer
theorem amount_each_parent_has_to_pay:
  let new_salary := old_salary + (old_salary * raise_percentage) in
  let salary_after_tax := new_salary - (new_salary * tax_rate) in
  let amount_per_parent := salary_after_tax / number_of_kids in
  let amount_in_eur := amount_per_parent / exchange_rate_usd_to_eur in
  let amount_in_gbp := amount_per_parent / exchange_rate_usd_to_gbp in
  let amount_in_jpy := amount_per_parent * exchange_rate_usd_to_jpy in
  amount_in_eur = 5294.12 ∧
  amount_in_gbp = 6000 ∧
  amount_in_jpy = 495000 :=
by
  sorry

end amount_each_parent_has_to_pay_l754_754715


namespace find_values_l754_754553

noncomputable def interior_angle (n : ℕ) : ℚ :=
  180 * (n - 2) / n

def is_non_integer_non_repeating_decimal (q : ℚ) : Prop :=
  let q_d := q.num / q.denom in q_d ≠ ⌊q_d⌋

theorem find_values (n : ℕ) : (3 ≤ n ∧ n ≤ 15) → 
    ({n : ℕ | 3 ≤ n ∧ n ≤ 15 ∧ is_non_integer_non_repeating_decimal (interior_angle n)}.card = 2) :=
sorry

end find_values_l754_754553


namespace tan_subtraction_l754_754926

theorem tan_subtraction (α β : ℝ) (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_subtraction_l754_754926


namespace sequence_term_3001_exists_exactly_4_values_l754_754324

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l754_754324


namespace acute_angle_sine_l754_754696

theorem acute_angle_sine (A B C D M N : ℝ × ℝ) (θ : ℝ) :
  (A = (0, 0)) ∧ (B = (4, 0)) ∧ (C = (4, 6)) ∧ (D = (0, 6)) ∧ 
  (M = (4, 3)) ∧ (N = (2, 6)) ∧
  (M = (B + C) / 2) ∧ (N = (C + D) / 2) →
  sin θ = (Real.sqrt 95) / 12 :=
by
  sorry

end acute_angle_sine_l754_754696


namespace distinct_solutions_diff_l754_754989

theorem distinct_solutions_diff {r s : ℝ} 
  (h1 : ∀ a b : ℝ, (a ≠ b → ( ∃ x, x ≠ a ∧ x ≠ b ∧ (x = r ∨ x = s) ∧ (x-5)(x+5) = 25*x - 125) )) 
  (h2 : r > s) : r - s = 15 :=
by
  sorry

end distinct_solutions_diff_l754_754989


namespace lens_price_l754_754566

variable (P : ℝ)
variable (discounted_price : ℝ := 240)

theorem lens_price (h1 : discounted_price = 240)
                   (h2 : 0.8 * P = discounted_price) : 
                   P = 300 := 
by 
  rw [h1, h2]
  sorry

end lens_price_l754_754566


namespace find_m_trajectory_of_Q_l754_754960

noncomputable def condition_point (A : Point := Point.mk (Real.sqrt 2) 0) := true

noncomputable def condition_line (l : Real → Real → Prop := λ rho theta, rho * Real.sin (theta - Real.pi / 4) = _) := true

def distance_from_point_to_line (d : Real := 3) := true

theorem find_m (A : Point) (l : Real → Real → Prop) (d : Real) (m : Real) (hA : condition_point A) (hl : condition_line l) (hd : distance_from_point_to_line d) :
  m = 2 :=
sorry

theorem trajectory_of_Q (m : Real := 2) (P : Real → Real → Prop := λ ρ₀ θ₀, ρ₀ * Real.sin (θ₀ - Real.pi / 4) = 2) (Q : Real → Real → Prop := λ ρ θ, ρ = (1 / 2) * Real.sin (θ - Real.pi / 4))
  (hP : condition_line P) :
  ∀ ρ θ, Q ρ θ :=
sorry

end find_m_trajectory_of_Q_l754_754960


namespace least_pos_int_with_divisors_l754_754620

theorem least_pos_int_with_divisors (m k : ℕ) (h1 : nat.dvd_not_unit 10 m = false)
  (h2 : ∃ n : ℕ, (count_divisors n = 2023) ∧ (n = m * 10^k)) : 
  m + k = 999846 :=
sorry

end least_pos_int_with_divisors_l754_754620


namespace smallest_positive_period_max_value_in_interval_min_value_in_interval_l754_754890

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem smallest_positive_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = 2 := by
  sorry

theorem min_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = -1 := by
  sorry

end smallest_positive_period_max_value_in_interval_min_value_in_interval_l754_754890


namespace correct_ordering_of_f_values_l754_754713

variable {f : ℝ → ℝ}

theorem correct_ordering_of_f_values
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_decreasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end correct_ordering_of_f_values_l754_754713


namespace radius_large_circle_l754_754643

/-- Definitions for the problem context -/
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_circles (c1 c2 : Circle) : Prop :=
dist c1.center c2.center = c1.radius + c2.radius

/-- Theorem to prove the radius of the large circle -/
theorem radius_large_circle 
  (small_circle : Circle)
  (h_radius : small_circle.radius = 2)
  (large_circle : Circle)
  (h_tangency1 : tangent_circles small_circle large_circle)
  (small_circle2 : Circle)
  (small_circle3 : Circle)
  (h_tangency2 : tangent_circles small_circle small_circle2)
  (h_tangency3 : tangent_circles small_circle small_circle3)
  (h_tangency4 : tangent_circles small_circle2 large_circle)
  (h_tangency5 : tangent_circles small_circle3 large_circle)
  (h_tangency6 : tangent_circles small_circle2 small_circle3)
  : large_circle.radius = 2 * (Real.sqrt 3 + 1) :=
sorry

end radius_large_circle_l754_754643


namespace number_of_divisors_l754_754386

theorem number_of_divisors (n : ℕ) :
  n > 0 → ∃! n : ℕ, (∀ m : ℕ, (∑ i in Finset.range (n + 1), i + 1) ∣ 8 * n) ∧ n ∈ {1, 3, 7, 15} :=
by
  sorry

end number_of_divisors_l754_754386


namespace find_b_when_a_is_1600_l754_754127

variable (a b : ℝ)

def inversely_vary (a b : ℝ) : Prop := a * b = 400

theorem find_b_when_a_is_1600 
  (h1 : inversely_vary 800 0.5)
  (h2 : inversely_vary a b)
  (h3 : a = 1600) :
  b = 0.25 := by
  sorry

end find_b_when_a_is_1600_l754_754127


namespace find_c_l754_754631

-- Define the circumradius
def R : ℝ := 5 / 6

-- Define the cosines of angles A and B
def cosA : ℝ := 12 / 13
def cosB : ℝ := 3 / 5

-- Goal: Prove that c, the side opposite to angle C, equals 21/13
theorem find_c (R : ℝ) (cosA : ℝ) (cosB : ℝ) (hR : R = 5 / 6) (hcosA : cosA = 12 / 13) (hcosB : cosB = 3 / 5) : 
  let sinA := sqrt (1 - cosA^2)
  let sinB := sqrt (1 - cosB^2)
  let sinC := sinA * cosB + cosA * sinB
  let c := 2 * R * sinC
  c = 21 / 13 := by
    sorry

end find_c_l754_754631


namespace exists_cyclic_quadrilateral_of_same_color_l754_754862

-- Structure defining a regular pentagon with vertices A, B, C, D, E
structure Pentagon (α : Type) :=
(A B C D E : α)

-- Define the property of the vertices that they are the midpoints of the previous pentagon
def is_midpoint_pentagon_sequence (P : Fin 11 → Pentagon ℝ) : Prop :=
  ∀ n, 1 ≤ n < 11 → (
    P n).A = midpoint (P (n-1)).A (P (n-1)).B ∧
    (P n).B = midpoint (P (n-1)).B (P (n-1)).C ∧
    (P n).C = midpoint (P (n-1)).C (P (n-1)).D ∧
    (P n).D = midpoint (P (n-1)).D (P (n-1)).E ∧
    (P n).E = midpoint (P (n-1)).E (P (n-1)).A

-- Each vertex is colored either red or blue
def vertex_coloring (P : Fin 11 → Pentagon ℝ) (coloring : ℝ → Prop) : Prop :=
  ∀ i, coloring (P i).A ∨ ¬ coloring (P i).A ∧
       coloring (P i).B ∨ ¬ coloring (P i).B ∧
       coloring (P i).C ∨ ¬ coloring (P i).C ∧
       coloring (P i).D ∨ ¬ coloring (P i).D ∧
       coloring (P i).E ∨ ¬ coloring (P i).E

-- The theorem stating there exist four vertices of the same color forming a cyclic quadrilateral
theorem exists_cyclic_quadrilateral_of_same_color :
  ∀ (P : Fin 11 → Pentagon ℝ) (coloring : ℝ → Prop),
    is_midpoint_pentagon_sequence P →
    vertex_coloring P coloring →
    ∃ (a b c d : ℝ), 
    (coloring a ∧ coloring b ∧ coloring c ∧ coloring d) ∧ 
    are_cyclic_quadrilateral a b c d :=
sorry

end exists_cyclic_quadrilateral_of_same_color_l754_754862


namespace area_of_triangle_AOB_l754_754069

theorem area_of_triangle_AOB:
  let A := (2, 2 * Real.pi / 3) in
  let B := (3, Real.pi / 6) in
  let O := (0, 0) in
  let angle_AOB := 2 * Real.pi / 3 - Real.pi / 6 in
  angle_AOB = Real.pi / 2 →
  (1 / 2 * (2:ℝ) * (3:ℝ) = 3) :=
by
  intros,
  sorry

end area_of_triangle_AOB_l754_754069


namespace ice_cream_flavors_l754_754453

theorem ice_cream_flavors : ∑ [, ∏ options := 4, scoops = 5, ∑ combinations: ℕ \satisfying\]
(pcond) (total_ways =  : combn): possible flavor == 56.

end ice_cream_flavors_l754_754453


namespace room_length_calculation_l754_754946

-- Definitions of the problem conditions
def room_volume : ℝ := 10000
def room_width : ℝ := 10
def room_height : ℝ := 10

-- Statement to prove
theorem room_length_calculation : ∃ L : ℝ, L = room_volume / (room_width * room_height) ∧ L = 100 :=
by
  sorry

end room_length_calculation_l754_754946


namespace triple_composition_f_3_l754_754467

def f (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_f_3 : f (f (f 3)) = 107 :=
by
  sorry

end triple_composition_f_3_l754_754467


namespace find_n_l754_754172

-- Define the sequence using the provided recurrence relation
noncomputable def a : ℕ → ℝ
| 0     := 134
| 1     := 150
| (k+2) := a (k) - (k + 1) / a (k+1)

-- Prove the main statement
theorem find_n (n : ℕ) : (a n = 0) → n = 201 :=
by
  -- Proof will be provided here
  sorry

end find_n_l754_754172


namespace fractional_part_sum_le_l754_754107

open Real Int
noncomputable theory

theorem fractional_part_sum_le (n : ℕ) :
  (∑ i in Finset.range (n^2 + 1) \ {0}, (sqrt i - ⌊sqrt i⌋)) ≤ (n^2 - 1) / 2 :=
begin
  sorry -- proof is left as an exercise
end

end fractional_part_sum_le_l754_754107


namespace subset_no_subset_divides_l754_754587

theorem subset_no_subset_divides (n : ℕ) :
    ∃ A ⊆ {1..2^n}, A.card = n ∧
    ∀ B C ⊆ A, B ≠ C ∧ B ≠ ∅ ∧ C ≠ ∅ →
    (∑ b in B, b) ∣ (∑ c in C, c) → False :=
by
  sorry

end subset_no_subset_divides_l754_754587


namespace intersection_points_l754_754590

-- Define parameters: number of sides for each polygon
def n₆ := 6
def n₇ := 7
def n₈ := 8
def n₉ := 9

-- Condition: polygons are inscribed in the same circle, no shared vertices, no three sides intersect at a common point
def polygons_are_disjoint (n₁ n₂ : ℕ) (n₃ n₄ : ℕ) (n₅ : ℕ) : Prop :=
  true -- Assume this is a primitive condition encapsulating given constraints

-- Prove the number of intersection points is 80
theorem intersection_points : polygons_are_disjoint n₆ n₇ n₈ n₉ n₅ → 
  2 * (n₆ + n₇ + n₇ + n₈) + 2 * (n₇ + n₈) + 2 * n₉ = 80 :=
by  
  sorry

end intersection_points_l754_754590


namespace z_conjugate_difference_l754_754854

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754854


namespace projection_and_difference_magnitudes_l754_754443

noncomputable def vector_a : ℝ × ℝ := (- real.sqrt 3, 3)
noncomputable def magnitude_b : ℝ := 1
noncomputable def angle_ab : ℝ := real.pi / 6

theorem projection_and_difference_magnitudes :
  let proj_b_onto_a := ((abs (vector_a.1)) + (abs (vector_a.2))) 
  ((1 * (real.cos angle_ab) * (vector_a.1/2* (abs (vector_a.1)) + (abs (vector_a.2))),
  (1 * (real.cos angle_ab) * (vector_a.2/2* (abs (vector_a.1)) + (abs (vector_a.2))))),
  diff_magnitude := real.sqrt ((abs (vector_a.1)^2 + abs (vector_a.2)^2) + magnitude_b^2 
                          - 2 * ((abs (vector_a.1)) * magnitude_b * (real.cos angle_ab)))
(
  proj_b_onto_a = (- real.sqrt 3 / 4, 3 / 4) 
  ∧ 
  diff_magnitude = real.sqrt 7
)
    :=
begin
  sorry
end

end projection_and_difference_magnitudes_l754_754443


namespace z_conjugate_difference_l754_754853

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754853


namespace find_a_and_b_l754_754104

-- Definitions of the sets A and B
def A : Set ℝ := { x : ℝ | 2 * x^2 + 7 * x - 15 < 0 }
def B (a b : ℝ) : Set ℝ := { x : ℝ | x^2 + a * x + b ≤ 0 }

-- The conditions provided in the problem
def condition1 : Prop := A ∩ B a b = ∅
def condition2 : Prop := A ∪ B a b = { x : ℝ | -5 < x ∧ x ≤ 2 }

-- The final proof problem statement
theorem find_a_and_b (a b : ℝ) : condition1 → condition2 → a = -7 / 2 ∧ b = 3 :=
by
    intro h1 h2
    sorry

end find_a_and_b_l754_754104


namespace domain_function_l754_754765

def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def domain_of_f (f : ℝ → ℝ) : Set ℝ := 
  {x : ℝ | f x ≠ 0}

theorem domain_function :
  domain_of_f (λ x => 1 / (quadratic 1 (-8) 18 x).floor) = 
  set_of (λ x => x ≤ 1 ∨ 17 ≤ x) :=
by
  sorry

end domain_function_l754_754765


namespace volume_of_hemisphere_l754_754229

theorem volume_of_hemisphere (d : ℝ) (hd : d = 10) :
  ∃ V, V = (2 / 3) * real.pi * (5)^3 ∧ V = 250 * real.pi / 3 :=
by
  have r : ℝ := d / 2,
  rw hd at r,
  use (2 / 3) * real.pi * r^3,
  rw ←hd, 
  simp [r],
  simp [pow_three, mul_assoc],
  sorry

end volume_of_hemisphere_l754_754229


namespace passing_time_correct_l754_754692

noncomputable def speed_conversion (kmph: ℝ) : ℝ :=
  kmph * 1000 / 3600

noncomputable def relative_speed (speed1: ℝ) (speed2: ℝ) : ℝ :=
  speed_conversion(speed1) + speed_conversion(speed2)

noncomputable def passing_time (distance: ℝ) (speed1: ℝ) (speed2: ℝ) : ℝ :=
  distance / relative_speed(speed1, speed2)

theorem passing_time_correct :
  passing_time 100 45 30 ≈ 4.8 := 
by 
  sorry

end passing_time_correct_l754_754692


namespace percentile_75th_is_39_l754_754486

-- Given the conditions
def scores : List ℕ := [29, 30, 38, 25, 37, 40, 42, 32]
def n : ℕ := scores.length

-- Definition of the 75th percentile function
def percentile (p : ℚ) (xs : List ℕ) : ℚ :=
  let xs_sorted := xs.qsort (≤)
  let pos := p * xs_sorted.length
  if pos.floor = pos then
    (Rational.of_nat (xs_sorted.nth_le (pos.to_nat - 1) sorry))
  else
    let lower := xs_sorted.nth_le (pos.floor.to_nat - 1) sorry
    let upper := xs_sorted.nth_le (pos.ceil.to_nat - 1) sorry
    (lower + upper) / 2

-- Statement to prove
theorem percentile_75th_is_39 : percentile (3 / 4) scores = (39 : ℚ) :=
by
  sorry

end percentile_75th_is_39_l754_754486


namespace num_shortest_paths_A_to_B_l754_754164

theorem num_shortest_paths_A_to_B :
  let n := 5 in
  let total_steps := 2 * n in
  let north_steps := n in
  let east_steps := n in
  (total_steps.choose north_steps) = 252 :=
by {
  sorry
}

end num_shortest_paths_A_to_B_l754_754164


namespace probability_composite_probability_prime_l754_754345

def numbers := {1, 2, 3, 4, 5, 6, 7, 8}
def composite_numbers := {4, 6, 8}
def prime_numbers := {2, 3, 5, 7}

theorem probability_composite : (composite_numbers.to_finset.card : ℚ) / (numbers.to_finset.card : ℚ) = 3 / 8 := by
  sorry

theorem probability_prime : (prime_numbers.to_finset.card : ℚ) / (numbers.to_finset.card : ℚ) = 1 / 2 := by
  sorry

end probability_composite_probability_prime_l754_754345


namespace max_elements_l754_754980

/-- Define the properties and main theorem statement -/
def has_100_digits (n : ℕ) : Prop := 
  10^99 ≤ n ∧ n < 10^100

def atom (s : ℕ) (S : set ℕ) : Prop :=
  ∀ x y ∈ S, s % (x + y) ≠ 0

theorem max_elements (S : set ℕ) (h100digits : ∀ s ∈ S, has_100_digits s)
  (h_atoms : ∀ s ∈ S, (¬ atom s S → (∃ a ∈ S, ∃ b ∈ S, s = a + b)))
  (h_cardinal : {s ∈ S | atom s S}.to_finset.card ≤ 10) :
  S.to_finset.card ≤ Nat.choose 19 10 - 1 :=
sorry

end max_elements_l754_754980


namespace sum_of_x_and_y_l754_754461

theorem sum_of_x_and_y (x y : ℚ) (h1 : 1/x + 1/y = 3) (h2 : 1/x - 1/y = -7) : x + y = -3/10 :=
by
  sorry

end sum_of_x_and_y_l754_754461


namespace domain_of_f_l754_754764

def f (x : ℝ) : ℝ := 1 / ↑(Int.floor (x ^ 2 - 8 * x + 18))

theorem domain_of_f : ∀ x : ℝ, ∃ y : ℝ, f x = y := by
  sorry  -- The proof, showing that f(x) is always defined, is omitted.

end domain_of_f_l754_754764


namespace domain_of_k_l754_754341

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 9) + 1 / (x ^ 2 - 9) + 1 / (x ^ 3 - 9 * x)

theorem domain_of_k :
  {x : ℝ | (k x).is_defined_at x} = ℝ \ {-9, -3, 0, 3} :=
by
  sorry

end domain_of_k_l754_754341


namespace distribution_ways_l754_754194

theorem distribution_ways :
  ∃ (n : ℕ) (erasers pencils notebooks pens : ℕ),
  pencils = 4 ∧ notebooks = 2 ∧ pens = 3 ∧ 
  n = 6 := sorry

end distribution_ways_l754_754194


namespace Hugo_win_probability_l754_754050

noncomputable def probability_Hugo_wins_with_6 :
  ℕ × ℕ × ℕ × ℕ × ℕ → ℝ := sorry

theorem Hugo_win_probability (players_rolls : Fin 5 → Fin 6) (Hugo_win : Bool) :
  Hugo_win = true → probability_Hugo_wins_with_6 (players_rolls 0, players_rolls 1, players_rolls 2, players_rolls 3, players_rolls 4) = 4375 / 7776 :=
by
  sorry

end Hugo_win_probability_l754_754050


namespace perfect_score_prob_equal_prob_scores_l754_754238

def prob_correct_second : ℚ := 2 / 3
def prob_correct_third : ℚ := 1 / 2
def prob_correct_fourth : ℚ := 1 / 4

theorem perfect_score_prob (P : ℚ) (h : P = 2 / 3) : 
  (P * prob_correct_third * prob_correct_fourth = 1 / 12) :=
by {
  rw h,
  exact dec_trivial
}

theorem equal_prob_scores (P : ℚ) (h1 : (P * prob_correct_third * prob_correct_fourth) = (3*P*prob_correct_fourth + 3*(1-P)*prob_correct_fourth + (1-P)*prob_correct_fourth)) :
  P = 2 / 3 :=
sorry

end perfect_score_prob_equal_prob_scores_l754_754238


namespace find_disjoint_paths_l754_754232

variables {V : Type} [fintype V]

structure Graph (V : Type) [fintype V] :=
(edges : V → V → Prop)
(symm : ∀ {u v : V}, edges u v → edges v u)
(no_loops : ∀ {u v : V}, edges u v → u ≠ v)

variable (G : Graph V)

def path (G : Graph V) (u v : V) : Prop :=
∃ (P : list V), P.head = some u ∧ P.reverse.head = some v ∧
               ∀ (i : ℕ), i < P.length - 1 → G.edges (P.nth_le i _) (P.nth_le (i + 1) _)

def disjoint_paths (G : Graph V) (u v : V) : Prop :=
∃ (P1 P2 : list V), 
  P1 ≠ P2 ∧ P1.head = some u ∧ P1.reverse.head = some v ∧ 
  P2.head = some u ∧ P2.reverse.head = some v ∧
  (∀ (x : V), x ∈ P1 → x ∈ P2 → x = u ∨ x = v) ∧
  (∀ (i : ℕ), i < P1.length - 1 → G.edges (P1.nth_le i _) (P1.nth_le (i + 1) _)) ∧
  (∀ (i : ℕ), i < P2.length - 1 → G.edges (P2.nth_le i _) (P2.nth_le (i + 1) _))

theorem find_disjoint_paths 
  (h1 : ∀ (u v w : V), path G u v → ∃ P, path G v w ∧ ∃ P', path G u w ∧ disjoint_paths G u v) 
  (h2 : 3 ≤ fintype.card V) 
  (A B : V) (G : Graph V) : 
  disjoint_paths G A B := 
sorry

end find_disjoint_paths_l754_754232


namespace find_digit_B_l754_754174

def six_digit_number (B : ℕ) : ℕ := 303200 + B

def is_prime_six_digit (B : ℕ) : Prop := Prime (six_digit_number B)

theorem find_digit_B :
  ∃ B : ℕ, (B ≤ 9) ∧ (is_prime_six_digit B) ∧ (B = 9) :=
sorry

end find_digit_B_l754_754174


namespace tips_fraction_of_income_l754_754208

theorem tips_fraction_of_income
  (S T : ℝ)
  (h1 : T = (2 / 4) * S) :
  T / (S + T) = 1 / 3 :=
by
  -- Proof goes here
  sorry

end tips_fraction_of_income_l754_754208


namespace number_of_girls_l754_754056

-- Definitions of the conditions
variables (G B : ℕ)
def total_candidates := 2000
def pass_percentage_boys := 0.30
def pass_percentage_girls := 0.32

def total_percentage_failed := 0.691
def total_passed := (1.0 - total_percentage_failed) * total_candidates

-- Given conditions:
axiom candidates_total : G + B = total_candidates
axiom boys_passed : ∀ B, P_B = pass_percentage_boys * B
axiom girls_passed : ∀ G, P_G = pass_percentage_girls * G
axiom passed_total : P = total_passed

-- The statement to prove
theorem number_of_girls : G = 900 := sorry

end number_of_girls_l754_754056


namespace Q_solution_l754_754984

def Q (x : ℝ) : ℝ := Q(0) + Q(1) * x + Q(2) * x^2

theorem Q_solution :
  (Q(-1) = 2) →
  (∃ c0 c1 c2 : ℝ, Q(x) = c0 + c1 * x + c2 * x^2 ∧ 
                    c0 = -4/5 ∧ c1 = -2 ∧ c2 = 4/5) :=
  sorry

end Q_solution_l754_754984


namespace repeatingDecimalSum_is_fraction_l754_754359

noncomputable def repeatingDecimalSum : ℚ :=
  (0.3333...).val + (0.040404...).val + (0.005005...).val

theorem repeatingDecimalSum_is_fraction : repeatingDecimalSum = 1134 / 2997 := by
  sorry

end repeatingDecimalSum_is_fraction_l754_754359


namespace radii_of_circumcircles_are_equal_l754_754558

theorem radii_of_circumcircles_are_equal
  (ABCD : Type*) [parallelogram ABCD] 
  (not_rectangle : ¬is_rectangle ABCD)
  (P : Type*) (inside : is_inside P ABCD)
  (radii_perpendicular : ∃ O1 O2 EF, circumcircle O1 PAB ∧ circumcircle O2 PCD ∧ common_chord EF ∧ perpendicular AD EF) :
  radius (circumcircle O1 PAB) = radius (circumcircle O2 PCD) := 
sorry

end radii_of_circumcircles_are_equal_l754_754558


namespace number_of_girls_attending_winter_festival_l754_754108

variables (g b : ℝ)
variables (totalStudents attendFestival: ℝ)

theorem number_of_girls_attending_winter_festival
  (H1 : g + b = 1500)
  (H2 : (3/5) * g + (2/5) * b = 800) :
  (3/5 * g) = 600 :=
sorry

end number_of_girls_attending_winter_festival_l754_754108


namespace janet_percentage_l754_754971

-- Define the number of snowballs made by Janet and her brother
def janet_snowballs : ℕ := 50
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage function
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- State the theorem to be proved
theorem janet_percentage : percentage janet_snowballs total_snowballs = 25 := by
  sorry

end janet_percentage_l754_754971


namespace equation_of_line_l_equations_of_line_m_l754_754421

-- Define the point P and condition for line l
def P := (2, (7 : ℚ)/4)
def l_slope : ℚ := 3 / 4

-- Define the given equation form and conditions for line l
def condition_l (x y : ℚ) : Prop := y - (7 / 4) = (3 / 4) * (x - 2)
def equation_l (x y : ℚ) : Prop := 3 * x - 4 * y = 5

theorem equation_of_line_l :
  ∀ x y : ℚ, condition_l x y → equation_l x y :=
sorry

-- Define the distance condition for line m
def equation_m (x y n : ℚ) : Prop := 3 * x - 4 * y + n = 0
def distance_condition_m (n : ℚ) : Prop := 
  |(-1 + n : ℚ)| / 5 = 3

theorem equations_of_line_m :
  ∃ n : ℚ, distance_condition_m n ∧ (equation_m 2 (7/4) n) ∨ 
            equation_m 2 (7/4) (-14) :=
sorry

end equation_of_line_l_equations_of_line_m_l754_754421


namespace ratio_friday_to_monday_l754_754082

-- Definitions from conditions
def rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def total_toys : ℕ := 3 * rabbits

-- Definition to represent the number of toys bought on Friday
def friday_toys : ℕ := total_toys - (monday_toys + wednesday_toys + saturday_toys)

-- Theorem to prove the ratio is 4:1
theorem ratio_friday_to_monday : friday_toys / monday_toys = 4 := by
  -- Placeholder for the proof
  sorry

end ratio_friday_to_monday_l754_754082


namespace equidistant_point_is_circumcenter_l754_754626

theorem equidistant_point_is_circumcenter {A B C : Type} [EuclideanGeometry A B C]
  (P : A) (T : Triangle A B C) :
  (∀ (X : Vertex T), dist P X = dist P T.v1) ↔ 
  (P = circumcenter T) :=
sorry

end equidistant_point_is_circumcenter_l754_754626


namespace MNPQ_is_rectangle_l754_754057

variable {Point : Type}
variable {A B C D M N P Q : Point}

def is_parallelogram (A B C D : Point) : Prop := sorry
def altitude (X Y : Point) : Prop := sorry
def rectangle (M N P Q : Point) : Prop := sorry

theorem MNPQ_is_rectangle 
  (h_parallelogram : is_parallelogram A B C D)
  (h_alt1 : altitude B M)
  (h_alt2 : altitude B N)
  (h_alt3 : altitude D P)
  (h_alt4 : altitude D Q) :
  rectangle M N P Q :=
sorry

end MNPQ_is_rectangle_l754_754057


namespace complex_conj_difference_l754_754838

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754838


namespace two_lines_in_3d_space_l754_754777

theorem two_lines_in_3d_space : 
  ∀ x y z : ℝ, x^2 + 2 * x * (y + z) + y^2 = z^2 + 2 * z * (y + x) + x^2 → 
  (∃ a : ℝ, y = -z ∧ x = 0) ∨ (∃ b : ℝ, z = - (2 / 3) * x) :=
  sorry

end two_lines_in_3d_space_l754_754777


namespace rectangle_length_l754_754246

theorem rectangle_length (side_length_m : ℝ) (small_square_side_mm : ℝ) (width_mm : ℝ) :
  side_length_m = 1 → small_square_side_mm = 1 → width_mm = 1 →
  (let length_mm := (side_length_m * 1000) ^ 2 / width_mm in length_mm = 10^6) := 
by
  intros h1 h2 h3
  have : side_length_m * 1000 = 1000, from by { rw h1, norm_num, }
  have : (side_length_m * 1000) ^ 2 = 10^6, from by { rw this, norm_num, }
  have : width_mm = 1, from h3
  have : (side_length_m * 1000) ^ 2 / width_mm = 10^6, from by { rw [this, this_1], norm_num, }
  exact this

end rectangle_length_l754_754246


namespace four_possible_x_values_l754_754293

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l754_754293


namespace num_x_for_3001_in_sequence_l754_754299

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l754_754299


namespace angle_coincides_with_graph_y_eq_neg_abs_x_l754_754479

noncomputable def angle_set (α : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + 225 ∨ α = k * 360 + 315}

theorem angle_coincides_with_graph_y_eq_neg_abs_x (α : ℝ) :
  α ∈ angle_set α ↔ 
  ∃ k : ℤ, (α = k * 360 + 225 ∨ α = k * 360 + 315) :=
by
  sorry

end angle_coincides_with_graph_y_eq_neg_abs_x_l754_754479


namespace number_of_whole_numbers_without_1_or_2_l754_754032

/-- There are 439 whole numbers between 1 and 500 that do not contain the digit 1 or 2. -/
theorem number_of_whole_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 439 ∧ ∀ m, 1 ≤ m ∧ m ≤ 500 → ∀ d ∈ (m.digits 10), d ≠ 1 ∧ d ≠ 2 :=
sorry

end number_of_whole_numbers_without_1_or_2_l754_754032


namespace limit_sequence_l754_754213

theorem limit_sequence :
  (∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(n * (11 * n)^(1/4) + sqrt (25 * n^4 - 81)) / ((n - 7 * sqrt n) * sqrt (n^2 - n + 1)) - L| < ε)) ↔ L = 5 := 
begin
  sorry
end

end limit_sequence_l754_754213


namespace smallest_altitude_leq_three_l754_754629

theorem smallest_altitude_leq_three (a b c : ℝ) (r : ℝ) 
  (ha : a = max a (max b c)) 
  (r_eq : r = 1) 
  (area_eq : ∀ (S : ℝ), S = (a + b + c) / 2 ∧ S = a * h / 2) :
  ∃ h : ℝ, h ≤ 3 :=
by
  sorry

end smallest_altitude_leq_three_l754_754629


namespace walking_west_negation_l754_754480

theorem walking_west_negation (distance_east distance_west : Int) (h_east : distance_east = 6) (h_west : distance_west = -10) : 
    (10 : Int) = - distance_west := by
  sorry

end walking_west_negation_l754_754480


namespace tan_double_angle_l754_754816

theorem tan_double_angle (θ : ℝ) (h1 : cos θ = -3 / 5) (h2 : 0 < θ ∧ θ < π) : tan (2 * θ) = 24 / 7 :=
by
  sorry

end tan_double_angle_l754_754816


namespace sum_of_fractions_le_half_n_l754_754562

theorem sum_of_fractions_le_half_n (n : ℕ) (a : Fin n → ℝ) (hpos : ∀ i, a i > 0) (hsum : (Finset.univ.sum (λ i, a i)) = 1) :
  let S := ∑ i j, (a i * a j) / (a i + a j) + ∑ k, (a k * a k) / (2 * a k)
  in S ≤ n / 2 := 
sorry

end sum_of_fractions_le_half_n_l754_754562


namespace total_toys_l754_754539

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end total_toys_l754_754539


namespace simplify_expression_l754_754594

variables (a b : ℝ)

theorem simplify_expression : 
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := by
  -- proof here
  sorry

end simplify_expression_l754_754594


namespace complex_z_sub_conjugate_eq_neg_i_l754_754848

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754848


namespace fractional_expression_evaluation_l754_754462

theorem fractional_expression_evaluation
  (m n r t : ℚ)
  (h1 : m / n = 4 / 3)
  (h2 : r / t = 9 / 14) :
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end fractional_expression_evaluation_l754_754462


namespace recurring_decimals_sum_l754_754365

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end recurring_decimals_sum_l754_754365


namespace constant_term_sqrtx_l754_754670

theorem constant_term_sqrtx (x : ℝ) : 
  -- Given conditions:
  let a := (11: ℕ) * 2 / 3, 
  let b := 11 - a,
  -- a and b must be integers closest to their calculated values such that a + b = 11 and \(2b = a\)
  b = 4 ∧ a = 7 →  
  -- Prove:
  (∑ k in (0:finset.range 12), combinatorics.nat.choose 11 k * (sqrt x ^ (11 - k)) * ((7 / x) ^ k)) =
  792330 := 
by 
  sorry

end constant_term_sqrtx_l754_754670


namespace area_of_quadrilateral_APQC_l754_754647

-- Define the geometric entities and conditions
structure RightTriangle (a b c : ℝ) :=
  (hypotenuse_eq: c = Real.sqrt (a ^ 2 + b ^ 2))

-- Triangles PAQ and PQC are right triangles with given sides
def PAQ := RightTriangle 9 12 (Real.sqrt (9^2 + 12^2))
def PQC := RightTriangle 12 9 (Real.sqrt (15^2 - 12^2))

-- Prove that the area of quadrilateral APQC is 108 square units
theorem area_of_quadrilateral_APQC :
  let area_PAQ := 1/2 * 9 * 12
  let area_PQC := 1/2 * 12 * 9
  area_PAQ + area_PQC = 108 :=
by
  sorry

end area_of_quadrilateral_APQC_l754_754647


namespace complex_conjugate_difference_l754_754819

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754819


namespace limit_sequence_l754_754214

theorem limit_sequence :
  (∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(n * (11 * n)^(1/4) + sqrt (25 * n^4 - 81)) / ((n - 7 * sqrt n) * sqrt (n^2 - n + 1)) - L| < ε)) ↔ L = 5 := 
begin
  sorry
end

end limit_sequence_l754_754214


namespace magnitude_of_2a_minus_b_l754_754907

/-- Definition of the vectors a and b --/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

/-- Proposition stating the magnitude of 2a - b --/
theorem magnitude_of_2a_minus_b : 
  (Real.sqrt ((2 * a.1 - b.1) ^ 2 + (2 * a.2 - b.2) ^ 2)) = Real.sqrt 10 :=
by
  sorry

end magnitude_of_2a_minus_b_l754_754907


namespace speed_of_first_half_of_journey_l754_754743

theorem speed_of_first_half_of_journey
  (total_time : ℝ)
  (speed_second_half : ℝ)
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (second_half_distance : ℝ)
  (time_second_half : ℝ)
  (time_first_half : ℝ)
  (speed_first_half : ℝ) :
  total_time = 15 →
  speed_second_half = 24 →
  total_distance = 336 →
  first_half_distance = total_distance / 2 →
  second_half_distance = total_distance / 2 →
  time_second_half = second_half_distance / speed_second_half →
  time_first_half = total_time - time_second_half →
  speed_first_half = first_half_distance / time_first_half →
  speed_first_half = 21 :=
by intros; sorry

end speed_of_first_half_of_journey_l754_754743


namespace distinct_solutions_difference_l754_754997

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : r > s)
  (h_eq : ∀ x : ℝ, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
begin
  sorry
end

end distinct_solutions_difference_l754_754997


namespace cube_surface_area_l754_754227

noncomputable def volume := 512 
noncomputable def side_len := real.cbrt(volume)
noncomputable def surface_area := 6 * side_len^2

theorem cube_surface_area : surface_area = 384 := by
  have h1 : side_len = 8 := by
    -- side_len = real.cbrt(512) = 8
    sorry
  have h2 : surface_area = 6 * 8^2 := by
    -- SA = 6 * side_len^2
    sorry
  show surface_area = 384 from by
    calc
      surface_area = 6 * 64 := by rwa h2
      ... = 384 := by norm_num

end cube_surface_area_l754_754227


namespace sphere_surface_area_l754_754880

theorem sphere_surface_area (l w h : ℝ) (π : ℝ) (d : ℝ) (r : ℝ) (S : ℝ) :
  l = 3 →
  w = 4 →
  h = 5 →
  d = Real.sqrt (l^2 + w^2 + h^2) →
  r = d / 2 →
  S = 4 * π * r^2 →
  S = 50 * π :=
by {
  intros hl hw hh hd hr hs,
  rw [hl, hw, hh] at hd,
  simp at hd,
  rw [hd] at hr,
  simp [Real.sqrt] at hr,
  rw [hr] at hs,
  simp at hs,
  exact hs,
  sorry
}

end sphere_surface_area_l754_754880


namespace cost_of_each_apple_l754_754757

-- Definitions: Conditions in the problem
def total_amount : ℕ := 360
def total_apples : ℕ := 18 * 5

-- Theorem: Cost of each apple equals 4 dollars
theorem cost_of_each_apple (total_amount total_apples : ℕ) : (total_amount / total_apples) = 4 := by
  have h1 : total_apples = 90 := by rw [total_apples, show 18 * 5 = 90 from rfl]
  have h2 : total_amount = 360 := rfl
  have h3 : 360 / 90 = 4 := rfl
  sorry

end cost_of_each_apple_l754_754757


namespace problem_l754_754884

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then
    if x < 2 then (Real.log2 (x + 1))
    else f (x - 2)
  else f (-x)

theorem problem (f_even : ∀ x : ℝ, f (-x) = f x)
                (f_periodic : ∀ x : ℝ, f (x + 2) = f x)
                (f_defn : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log2 (x + 1)) :
                f (-2015) + f (2016) = 1 :=
by {
  sorry
}

end problem_l754_754884


namespace tan_cos_expression_l754_754377

theorem tan_cos_expression : 
  tan 70 * cos 10 * (sqrt 3 * tan 20 - 1) = -1 :=
by
  sorry

end tan_cos_expression_l754_754377


namespace sequence_term_3001_exists_exactly_4_values_l754_754321

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l754_754321


namespace trisha_take_home_pay_l754_754648

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end trisha_take_home_pay_l754_754648


namespace distance_between_points_l754_754369

theorem distance_between_points :
  ∀ (A B : ℝ × ℝ), A = (3, 5) → B = (-4, 1) → 
  (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = real.sqrt 65) :=
by
  intros A B hA hB
  rw [hA, hB]
  -- rest of the proof would go here
  sorry

end distance_between_points_l754_754369


namespace phase_shift_correct_l754_754646

noncomputable def graph_shift : Prop :=
  ∀ (x : ℝ), 
    let f := cos (2 * x + π / 3),
        g := sin (2 * x) in
    f = g (x + 5 * π / 12)

-- Define the phase shift problem
theorem phase_shift_correct :
  to_shift (y = cos (2 * x + π / 3)) (y = sin 2 * x) (5 * π / 12) :=
sorry

end phase_shift_correct_l754_754646


namespace find_a_l754_754105

theorem find_a (f : ℝ → ℝ) (a : ℝ) (h1 : ∀ x, f(x) = -Real.logb 2 (-x) + a) (h2 : f(-2) + f(-4) = 1) : a = 2 :=
by
  sorry

end find_a_l754_754105


namespace decimal_222nd_digit_13_over_481_l754_754196

theorem decimal_222nd_digit_13_over_481 :
  let dec_rep : List ℕ := [0, 2, 7] -- the repeating decimal sequence
  (dec_rep.length = 3) →
  let n : ℕ := 222
  let remainder : ℕ := n % dec_rep.length
  (remainder == 0) →
  dec_rep.getLast 7 = 7 :=
by
  intros
  sorry

end decimal_222nd_digit_13_over_481_l754_754196


namespace solve_trig_equation_l754_754596

theorem solve_trig_equation {
  x : ℝ,
  k : ℤ
} : 
  5 * (1 - Real.cos x) = 4 * Real.sin x ↔ (x = 0 ∨ x = 2 * Real.arcsin (4 / 5) + 4 * k * Real.pi ∨ x = 2 * (Real.pi - Real.arcsin (4 / 5)) + 4 * k * Real.pi) :=
by
  sorry

end solve_trig_equation_l754_754596


namespace radius_of_large_circle_l754_754645

noncomputable def small_circle_radius : ℝ := 2
noncomputable def larger_circle_radius : ℝ :=
  let side_length := 2 * small_circle_radius in
  let altitude := (side_length * Real.sqrt 3) / 2 in
  side_length + altitude

theorem radius_of_large_circle :
  let r := (2 + 4 * Real.sqrt 3) / 2 in
  larger_circle_radius = r := by
  sorry

end radius_of_large_circle_l754_754645


namespace olly_needs_24_shoes_l754_754573

-- Define the number of paws for different types of pets
def dogs : ℕ := 3
def cats : ℕ := 2
def ferret : ℕ := 1

def paws_per_dog : ℕ := 4
def paws_per_cat : ℕ := 4
def paws_per_ferret : ℕ := 4

-- The theorem we want to prove
theorem olly_needs_24_shoes : 
  dogs * paws_per_dog + cats * paws_per_cat + ferret * paws_per_ferret = 24 :=
by
  sorry

end olly_needs_24_shoes_l754_754573


namespace total_number_of_toys_l754_754537

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l754_754537


namespace binom_19_10_l754_754288

theorem binom_19_10 : 
  (nat.choose 19 10) = 92378 :=
by
  have h1 : (nat.choose 17 7) = 19448 := by sorry
  have h2 : (nat.choose 17 9) = 24310 := by sorry
  sorry

end binom_19_10_l754_754288


namespace area_of_rotated_hexagon_l754_754746

theorem area_of_rotated_hexagon (R : ℝ) :
  let O := (0, 0 : ℝ × ℝ) in
  let A := (R, 0) in
  let B := (R * (Real.cos (2 / 3 * Real.pi)), R * (Real.sin (2 / 3 * Real.pi))) in
  let C := (R * (Real.cos (4 / 3 * Real.pi)), R * (Real.sin (4 / 3 * Real.pi))) in
  let A1 := (R * (Real.cos (Real.pi / 2)), R * (Real.sin (Real.pi / 2))) in
  let B1 := (R * (Real.cos (7 / 6 * Real.pi)), R * (Real.sin (7 / 6 * Real.pi))) in
  let C1 := (R * (Real.cos (11 / 6 * Real.pi)), R * (Real.sin (11 / 6 * Real.pi))) in
  ∀ (O A B C A1 B1 C1 : ℝ × ℝ),
  ∃ (hex_area : ℝ), hex_area = 9 * R^2 / 4 :=
sorry

end area_of_rotated_hexagon_l754_754746


namespace error_percentage_in_area_l754_754947

theorem error_percentage_in_area (L W : ℝ) :
  let L' := 1.16 * L,
      W' := 0.95 * W,
      actual_area := L * W,
      calculated_area := L' * W'
  in ((calculated_area - actual_area) / actual_area) * 100 = 10.2 := by
  sorry

end error_percentage_in_area_l754_754947


namespace trisha_take_home_pay_l754_754652

def hourly_wage : ℝ := 15
def hours_per_week : ℝ := 40
def weeks_per_year : ℝ := 52
def tax_rate : ℝ := 0.2

def annual_gross_pay : ℝ := hourly_wage * hours_per_week * weeks_per_year
def amount_withheld : ℝ := tax_rate * annual_gross_pay
def annual_take_home_pay : ℝ := annual_gross_pay - amount_withheld

theorem trisha_take_home_pay :
  annual_take_home_pay = 24960 := 
by
  sorry

end trisha_take_home_pay_l754_754652


namespace gcd_of_expressions_l754_754673

theorem gcd_of_expressions :
  Nat.gcd (121^2 + 233^2 + 345^2) (120^2 + 232^2 + 346^2) = 5 :=
sorry

end gcd_of_expressions_l754_754673


namespace hexadecagon_area_perimeter_l754_754731

noncomputable def central_angle_of_regular_polygon (n : ℕ) : ℝ :=
  360 / n

noncomputable def area_of_triangle (r : ℝ) (θ : ℝ) : ℝ :=
  (1 / 2) * r^2 * Real.sin θ

noncomputable def total_area_of_regular_polygon (n : ℕ) (r : ℝ) : ℝ :=
  n * area_of_triangle r (Real.toRadians $ central_angle_of_regular_polygon n)

noncomputable def side_length_of_regular_polygon (r : ℝ) (n : ℕ) : ℝ :=
  2 * r * Real.sin (Real.toRadians $ central_angle_of_regular_polygon n / 2)

noncomputable def perimeter_of_regular_polygon (n : ℕ) (r : ℝ) : ℝ :=
  n * side_length_of_regular_polygon r n

theorem hexadecagon_area_perimeter (r : ℝ) :
  (total_area_of_regular_polygon 16 r = 3.0616 * r^2) ∧ (perimeter_of_regular_polygon 16 r = 6.2432 * r) := by
  sorry

end hexadecagon_area_perimeter_l754_754731


namespace sequence_x_values_3001_l754_754307

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l754_754307


namespace find_n_l754_754982

noncomputable def Q : ℕ → ℚ
| 0     := 1
| (n+1) := (1/3) * (1 - Q n)

theorem find_n : Q 8 = 547 / 2187 := sorry

end find_n_l754_754982


namespace find_last_week_rope_l754_754977

/-- 
Description: Mr. Sanchez bought 4 feet of rope less than he did the previous week. 
Given that he bought 96 inches in total, find how many feet he bought last week.
--/
theorem find_last_week_rope (F : ℕ) :
  12 * (F - 4) = 96 → F = 12 := by
  sorry

end find_last_week_rope_l754_754977


namespace intersection_of_sets_l754_754018

theorem intersection_of_sets {A B : Set Nat} (hA : A = {1, 3, 9}) (hB : B = {1, 5, 9}) :
  A ∩ B = {1, 9} :=
sorry

end intersection_of_sets_l754_754018


namespace num_boys_l754_754736

variable (B G : ℕ)

def ratio_boys_girls (B G : ℕ) : Prop := B = 7 * G
def total_students (B G : ℕ) : Prop := B + G = 48

theorem num_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : 
  B = 42 :=
by
  sorry

end num_boys_l754_754736


namespace spiral_strip_length_l754_754704

-- Define the conditions
def circumference := 18 -- base circumference in inches
def height := 8 -- height in inches
def num_revolutions := 2 -- number of revolutions of the spiral strip

-- Given condition that describes the rectangle formed by unwrapping the lateral surface
def rectangle_length := num_revolutions * circumference
def rectangle_width := height

-- The proof statement
theorem spiral_strip_length :
  (rectangle_length^2 + rectangle_width^2 = 1360) → 
  (real.sqrt (rectangle_length^2 + rectangle_width^2) = real.sqrt 1360) :=
by 
  intro h
  exact eq.symm (real.sqrt_inj (le_of_lt (by norm_num [rectangle_length, rectangle_width])) h)

-- Variables declaration
#eval circumference
#eval height
#eval num_revolutions
#eval rectangle_length
#eval rectangle_width

end spiral_strip_length_l754_754704


namespace total_cans_given_away_l754_754272

-- Define constants
def initial_stock : ℕ := 2000

-- Define conditions day 1
def people_day1 : ℕ := 500
def cans_per_person_day1 : ℕ := 1
def restock_day1 : ℕ := 1500

-- Define conditions day 2
def people_day2 : ℕ := 1000
def cans_per_person_day2 : ℕ := 2
def restock_day2 : ℕ := 3000

-- Define the question as a theorem
theorem total_cans_given_away : (people_day1 * cans_per_person_day1 + people_day2 * cans_per_person_day2) = 2500 := by
  sorry

end total_cans_given_away_l754_754272


namespace wheel_rpm_is_approximately_5000_23_l754_754691

noncomputable def bus_wheel_rpm (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speed_cm_per_min := (speed * 1000 * 100) / 60
  speed_cm_per_min / circumference

-- Conditions
def radius := 35
def speed := 66

-- Question (to be proved)
theorem wheel_rpm_is_approximately_5000_23 : 
  abs (bus_wheel_rpm radius speed - 5000.23) < 0.01 :=
by
  sorry

end wheel_rpm_is_approximately_5000_23_l754_754691


namespace four_digit_positive_integers_count_l754_754022

def first_two_digit_choices : Finset ℕ := {2, 3, 6}
def last_two_digit_choices : Finset ℕ := {3, 7, 9}

theorem four_digit_positive_integers_count :
  (first_two_digit_choices.card * first_two_digit_choices.card) *
  (last_two_digit_choices.card * (last_two_digit_choices.card - 1)) = 54 := by
sorry

end four_digit_positive_integers_count_l754_754022


namespace radius_large_circle_l754_754642

/-- Definitions for the problem context -/
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_circles (c1 c2 : Circle) : Prop :=
dist c1.center c2.center = c1.radius + c2.radius

/-- Theorem to prove the radius of the large circle -/
theorem radius_large_circle 
  (small_circle : Circle)
  (h_radius : small_circle.radius = 2)
  (large_circle : Circle)
  (h_tangency1 : tangent_circles small_circle large_circle)
  (small_circle2 : Circle)
  (small_circle3 : Circle)
  (h_tangency2 : tangent_circles small_circle small_circle2)
  (h_tangency3 : tangent_circles small_circle small_circle3)
  (h_tangency4 : tangent_circles small_circle2 large_circle)
  (h_tangency5 : tangent_circles small_circle3 large_circle)
  (h_tangency6 : tangent_circles small_circle2 small_circle3)
  : large_circle.radius = 2 * (Real.sqrt 3 + 1) :=
sorry

end radius_large_circle_l754_754642


namespace points_concyclic_l754_754420

noncomputable def proof_concyclic_points (A B C D E F I O O1 O2 : Type*) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] [Point I] [Circle O] [Circle O1] [Circle O2] : Prop :=
  let circumcenter (O : Circle) (triangle : Type*) := true -- Placeholder for the circumcenter predicate
  let tangent (C1 C2 : Circle) (P : Point) := true -- Placeholder for the tangent predicate
  let incenter (I : Point) (triangle : Type*) := true -- Placeholder for the incenter predicate
  let intersects (C1 C2 : Circle) (P Q : Point) := true -- Placeholder for the intersection predicate

  (circumcenter O (Triangle ABC)) ∧
  (tangent O O1 A) ∧
  (tangent O1 (Line BC) D) ∧
  (incenter I (Triangle ABC)) ∧
  (circumcenter O2 (Triangle IBC)) ∧
  (intersects O1 O2 E F) ∧
  (concyclic_points O1 E O2 F)
  
axiom concyclic_points : ∀ {P Q R S : Point}, (circumcircle P Q R S) → concyclic_points P Q R S

theorem points_concyclic (A B C D E F I O O1 O2 : Type*) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] [Point I] [Circle O] [Circle O1] [Circle O2] :
  proof_concyclic_points A B C D E F I O O1 O2 :=
by
  unfold proof_concyclic_points
  sorry

end points_concyclic_l754_754420


namespace thirtieth_digit_sum_fractions_l754_754197

theorem thirtieth_digit_sum_fractions :
  (fractional_digit_sum (1/13 + 1/11) 30) = 9 :=
sorry

/-- Function to obtain the nth digit after the decimal point of a fraction's sum in decimal -/
noncomputable def fractional_digit_sum (r : ℚ) (n : ℕ) : ℕ :=
sorry

end thirtieth_digit_sum_fractions_l754_754197


namespace cupcakes_left_l754_754110

theorem cupcakes_left (packages : ℕ) (cupcakes_per_package : ℕ) (eaten_cupcakes : ℕ) 
  (h_packages : packages = 3) (h_cupcakes_per_package : cupcakes_per_package = 4) 
  (h_eaten_cupcakes : eaten_cupcakes = 5) :
  packages * cupcakes_per_package - eaten_cupcakes = 7 :=
by 
  rw [h_packages, h_cupcakes_per_package, h_eaten_cupcakes]
  simp
  sorry

end cupcakes_left_l754_754110


namespace correct_proposition_D_l754_754552

variables {Point : Type*} [inner_product_space ℝ Point]
noncomputable def line := set Point
noncomputable def plane := set Point

variables {m n : line} {α β : plane} 

-- Conditions
axiom lines_distinct : m ≠ n
axiom planes_distinct : α ≠ β 

-- Definitions
def parallel (l1 l2 : line) : Prop := ∃ v, ∀ x ∈ l1, ∃ c : ℝ, (x : Point) + c • v ∈ l2
def perpendicular (l : line) (p : plane) : Prop := ∃ n ∈ p, ∀ x ∈ l, ∀ y ∈ p, inner_product x y = 0

-- Theorem
theorem correct_proposition_D : (parallel m n) ∧ (perpendicular n β) → (perpendicular m β) :=
by sorry

end correct_proposition_D_l754_754552


namespace angle_C_in_parallelogram_l754_754058

theorem angle_C_in_parallelogram (ABCD : Type)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A = angle_C)
  (h2 : angle_B = angle_D)
  (h3 : angle_A + angle_B = 180)
  (h4 : angle_A / angle_B = 3) :
  angle_C = 135 :=
  sorry

end angle_C_in_parallelogram_l754_754058


namespace is_isosceles_triangle_l754_754075

theorem is_isosceles_triangle (A B C : ℝ) 
  (h1 : A + B + C = π) 
  (h2 : (sin A + sin B) * (cos A + cos B) = 2 * sin C) :
  A = B :=
sorry

end is_isosceles_triangle_l754_754075


namespace log_sum_cube_eq_three_halves_l754_754876

theorem log_sum_cube_eq_three_halves :
  ∀ (a b : ℝ), a^3 + b^3 = (a + b) * (a^2 - a * b + b^2) →
  (log 2)^3 + 3 * (log 2) * (log 5) + (log 5)^3 + 1 / 2 = 3 / 2 := by
  intros a b h
  sorry

end log_sum_cube_eq_three_halves_l754_754876


namespace binom_19_10_l754_754289

theorem binom_19_10 : 
  (nat.choose 19 10) = 92378 :=
by
  have h1 : (nat.choose 17 7) = 19448 := by sorry
  have h2 : (nat.choose 17 9) = 24310 := by sorry
  sorry

end binom_19_10_l754_754289


namespace collinear_MNK_l754_754020

open EuclideanGeometry

theorem collinear_MNK 
  (ABC : Triangle)
  (K : Point)
  (M : Point)
  (N : Point)
  (hK : is_angle_bisector_extern K A B C)
  (hM : is_midpoint_of_arc M A C (circumcircle ABC))
  (hN : is_on_bisector N C ∧ is_parallel AN BM) :
  collinear {K, M, N} :=
begin
  sorry
end

end collinear_MNK_l754_754020


namespace distinct_solutions_difference_l754_754998

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : r > s)
  (h_eq : ∀ x : ℝ, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
begin
  sorry
end

end distinct_solutions_difference_l754_754998


namespace find_x_l754_754043

theorem find_x (x : ℝ) :
  let P1 := (2, 10)
  let P2 := (6, 2)
  
  -- Slope of the line joining (2, 10) and (6, 2)
  let slope12 := (P2.2 - P1.2) / (P2.1 - P1.1)
  
  -- Slope of the line joining (2, 10) and (x, -3)
  let P3 := (x, -3)
  let slope13 := (P3.2 - P1.2) / (P3.1 - P1.1)
  
  -- Condition that both slopes are equal
  slope12 = slope13
  
  -- To Prove: x must be 8.5
  → x = 8.5 :=
sorry

end find_x_l754_754043


namespace find_x_values_for_3001_l754_754328

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l754_754328


namespace exists_three_digit_number_cube_ends_in_777_l754_754344

theorem exists_three_digit_number_cube_ends_in_777 :
  ∃ x : ℤ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 := 
sorry

end exists_three_digit_number_cube_ends_in_777_l754_754344


namespace can_combine_fig1_can_combine_fig2_l754_754700

-- Given areas for rectangle partitions
variables (S1 S2 S3 S4 : ℝ)
-- Condition: total area of black rectangles equals total area of white rectangles
variable (h1 : S1 + S2 = S3 + S4)

-- Proof problem for Figure 1
theorem can_combine_fig1 : ∃ A : ℝ, S1 + S2 = A ∧ S3 + S4 = A := by
  sorry

-- Proof problem for Figure 2
theorem can_combine_fig2 : ∃ B : ℝ, S1 + S2 = B ∧ S3 + S4 = B := by
  sorry

end can_combine_fig1_can_combine_fig2_l754_754700


namespace sum_even_less_100_correct_l754_754760

-- Define the sequence of even, positive integers less than 100
def even_seq (n : ℕ) : Prop := n % 2 = 0 ∧ 0 < n ∧ n < 100

-- Sum of the first n positive integers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Sum of the even, positive integers less than 100
def sum_even_less_100 : ℕ := 2 * sum_n 49

theorem sum_even_less_100_correct : sum_even_less_100 = 2450 := by
  sorry

end sum_even_less_100_correct_l754_754760


namespace smallest_perfect_square_gt_50_with_odd_factors_l754_754373

theorem smallest_perfect_square_gt_50_with_odd_factors :
  ∃ (n : ℕ), n > 50 ∧ ∃ (m : ℕ), n = m^2 ∧
             (∀ (k : ℕ), k > 50 ∧ ∃ (l : ℕ), k = l^2 → n ≤ k) ∧
             odd (finset.card (finset.filter (has_dvd.dvd n) (finset.range (n + 1))))) :=
by
  use 64
  split
  . show 64 > 50, by norm_num
  split
  . use 8
    show 64 = 8^2, by norm_num
  split
  . intro k
    intro hk
    obtain ⟨l, hl⟩ := hk.2
    rw hl at hk
    rw hl
    exact nat.pow_le_pow_of_le_right (nat.succ_pos 1) 
         (lt_of_le_of_lt hk.1 (by norm_num)).le
  show odd (finset.card (finset.filter (has_dvd.dvd 64) (finset.range 65))), by norm_num
  sorry

end smallest_perfect_square_gt_50_with_odd_factors_l754_754373


namespace tims_drive_distance_l754_754784

theorem tims_drive_distance :
  let t1 := 120
  let t2 := 165
  ∃ y : ℕ, 
    (let speed_usual := y / t1
    let speed_reduced := speed_usual - 1 / 2
    let time_half_usual := (y / 2) / speed_usual
    let time_half_reduced := (y / 2) / speed_reduced
    in time_half_usual + time_half_reduced = t2) ∧ y = 140 := by
  sorry

end tims_drive_distance_l754_754784


namespace solve_inequalities_solve_linear_system_l754_754221

-- System of Inequalities
theorem solve_inequalities (x : ℝ) (h1 : x + 2 > 1) (h2 : 2 * x < x + 3) : -1 < x ∧ x < 3 :=
by
  sorry

-- System of Linear Equations
theorem solve_linear_system (x y : ℝ) (h1 : 3 * x + 2 * y = 12) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_inequalities_solve_linear_system_l754_754221


namespace find_theta_l754_754882

-- Define the conditions
def P : ℝ × ℝ := (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4))

-- Define the Lean theorem statement
theorem find_theta (θ : ℝ) (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) (h₂ : P = (Real.sin θ, Real.cos θ)) :
  θ = 7 * Real.pi / 4 :=
sorry

end find_theta_l754_754882


namespace trisha_take_home_pay_l754_754651

def hourly_wage : ℝ := 15
def hours_per_week : ℝ := 40
def weeks_per_year : ℝ := 52
def tax_rate : ℝ := 0.2

def annual_gross_pay : ℝ := hourly_wage * hours_per_week * weeks_per_year
def amount_withheld : ℝ := tax_rate * annual_gross_pay
def annual_take_home_pay : ℝ := annual_gross_pay - amount_withheld

theorem trisha_take_home_pay :
  annual_take_home_pay = 24960 := 
by
  sorry

end trisha_take_home_pay_l754_754651


namespace ratio_length_to_perimeter_ratio_width_to_perimeter_l754_754732

theorem ratio_length_to_perimeter (L W : ℕ) (P : ℕ) (hL : L = 24) (hW : W = 14) (hP : P = 2 * (L + W)) : 
  (L : ℚ) / P = 6 / 19 := 
  by
    -- We assert the given conditions
    rw [hL, hW, hP]
    -- Calculate perimeter
    norm_num
    -- The proof is omitted
    sorry

theorem ratio_width_to_perimeter (L W : ℕ) (P : ℕ) (hL : L = 24) (hW : W = 14) (hP : P = 2 * (L + W)) : 
  (W : ℚ) / P = 7 / 38 := 
  by
    -- We assert the given conditions
    rw [hL, hW, hP]
    -- Calculate perimeter
    norm_num
    -- The proof is omitted
    sorry

end ratio_length_to_perimeter_ratio_width_to_perimeter_l754_754732


namespace quadratic_eq_has_two_positive_roots_quadratic_eq_no_real_roots_prob_l754_754861

noncomputable def probability_two_positive_roots : ℚ := 
  (1 : ℚ) / 9

noncomputable def probability_no_real_roots : ℝ := 
  Real.pi / 4

theorem quadratic_eq_has_two_positive_roots {a b : ℕ} : 
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧
  2 * (a - 2) > 0 ∧ - b ^ 2 + 16 > 0 ∧
  (a - 2) ^ 2 + b ^ 2 ≥ 16) → 
  probability_two_positive_roots = 1 / 9 :=
sorry

theorem quadratic_eq_no_real_roots_prob {a b : ℝ} :
  (2 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 4 ∧
  (a - 2) ^ 2 + b ^ 2 < 16) → 
  probability_no_real_roots = Real.pi / 4 :=
sorry

end quadratic_eq_has_two_positive_roots_quadratic_eq_no_real_roots_prob_l754_754861


namespace even_function_f_f_at_0_f_at_neg_four_analytic_expression_f_l754_754158

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log (1 / 2)
  else if x = 0 then 0
  else log (-x) / log (1 / 2)

theorem even_function_f (x : ℝ) : f(-x) = f(x) :=
by
  sorry

theorem f_at_0 : f(0) = 0 :=
by
  sorry

theorem f_at_neg_four : f(-4) = -2 :=
by
  sorry

theorem analytic_expression_f (x : ℝ) : 
  f(x) = 
    if x > 0 then log x / log (1 / 2)
    else if x = 0 then 0
    else log (-x) / log (1 / 2) :=
by
  sorry

end even_function_f_f_at_0_f_at_neg_four_analytic_expression_f_l754_754158


namespace num_pos_integers_n_divide_8n_l754_754388

theorem num_pos_integers_n_divide_8n (n : ℕ) : 
  (∃ N : ℕ, N = 4 ∧ 
  { x : ℕ | 1 ≤ x ∧ 
    divides (8 * x) ((x * (x + 1)) / 2) 
  }.card = N) := sorry

end num_pos_integers_n_divide_8n_l754_754388


namespace equilateral_triangles_l754_754088

-- Define that ABC is a triangle with circumcircle Omega
variables {A B C : Point}
variables {Omega : Circle}

-- Let G be the centroid of the triangle ABC
def centroid (A B C G : Point) : Prop := 
  is_centroid A B C G

-- Let AG, BG, and CG extend to meet Omega again at A1, B1, and C1
variables {A_1 B_1 C_1 : Point}
variables {AG BG CG : Line}
variables (extends_AG : extend_line_AG_to_A1 A G A_1 Omega)
variables (extends_BG : extend_line_BG_to_B1 B G B_1 Omega)
variables (extends_CG : extend_line_CG_to_C1 C G C_1 Omega)

-- Given angle conditions
variables 
(angle_A : ∠ BAC = ∠ A_1 B_1 C_1) 
(angle_B : ∠ ABC = ∠ A_1 C_1 B_1) 
(angle_C : ∠ ACB = ∠ B_1 A_1 C_1)

-- Prove that ABC and A_1 B_1 C_1 are equilateral triangles
theorem equilateral_triangles 
  (h_centroid : centroid A B C G)
  (h_extend_AG : extends_AG)
  (h_extend_BG : extends_BG)
  (h_extend_CG : extends_CG)
  (h_angle_A : angle_A)
  (h_angle_B : angle_B)
  (h_angle_C : angle_C) : 
  is_equilateral_triangle A B C ∧ is_equilateral_triangle A_1 B_1 C_1 :=
sorry

end equilateral_triangles_l754_754088


namespace intersection_A_B_l754_754902

noncomputable def A : set ℝ := {x | (1 - x) * (1 + x) ≥ 0}
noncomputable def B : set ℝ := {y | ∃ x < 0, y = 2^x}
noncomputable def result : set ℝ := (0, 1)

theorem intersection_A_B : (A ∩ B) = result :=
by sorry

end intersection_A_B_l754_754902


namespace common_root_values_max_n_and_a_range_l754_754012

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+1) * x - 4 * (a+5)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x + 5

-- Part 1
theorem common_root_values (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ g a x = 0) → a = -9/16 ∨ a = -6 ∨ a = -4 ∨ a = 0 :=
sorry

-- Part 2
theorem max_n_and_a_range (a : ℝ) (m n : ℕ) (x0 : ℝ) :
  (m < n ∧ (m : ℝ) < x0 ∧ x0 < (n : ℝ) ∧ f a x0 < 0 ∧ g a x0 < 0) →
  n = 4 ∧ -1 ≤ a ∧ a ≤ -2/9 :=
sorry

end common_root_values_max_n_and_a_range_l754_754012


namespace sequence_x_values_3001_l754_754310

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l754_754310


namespace average_speed_is_correct_l754_754146

-- Defining the speeds of the activities
def swim_speed : ℝ := 1  -- miles per hour
def run_speed : ℝ := 7  -- miles per hour
def bike_speed : ℝ := 15  -- miles per hour

-- Defining the equal time spent on each activity
def time_spent : ℝ := 1  -- hour for simplicity

-- Defining the total distances covered in each activity
def swim_distance := swim_speed * time_spent  -- miles
def run_distance := run_speed * time_spent  -- miles
def bike_distance := bike_speed * time_spent  -- miles

-- Defining the total distance and time
def total_distance := swim_distance + run_distance + bike_distance  -- miles
def total_time := 3 * time_spent  -- hours (since 3 activities)

-- Calculating the average speed
def average_speed := total_distance / total_time  -- miles per hour

-- Proving that the average speed is approximately 7.67 miles per hour
theorem average_speed_is_correct : average_speed = 23 / 3 :=
by {
  unfold average_speed total_distance total_time,
  unfold swim_distance run_distance bike_distance,
  unfold swim_speed run_speed bike_speed,
  unfold time_spent,
  norm_num,
  sorry
}

end average_speed_is_correct_l754_754146


namespace minimum_f_zero_inequality_n_r_S_floor_eq_211_l754_754987

noncomputable def f (x : ℝ) (r : ℚ) : ℝ := (1 + x)^(r+1) - (r+1) * x - 1

theorem minimum_f_zero (r : ℚ) (hr : 0 < r) : ∀ x : ℝ, -1 < x → (x = 0 → f x r = 0) ∧ (x ≠ 0 → f x r > 0) := sorry

theorem inequality_n_r 
  (n : ℕ) (r : ℚ) (hn : 0 < n) (hr : 0 < r) : 
  (n : ℝ) ^ r > (1 / (r + 1 : ℝ)) * ((n : ℝ) ^ (r+1) - (n -1 : ℝ) ^ (r+1)) ∧ 
  (n : ℝ) ^ r < (1 / (r+1 : ℝ)) * (((n + 1 : ℝ) ^ (r+1)) - (n : ℝ) ^ (r+1)) := sorry

noncomputable def S : ℝ := Real.sum (List.range' 381 (3125-381+1))

theorem S_floor_eq_211 
  (S : ℝ) : ⌈S⌉ = 211 := sorry

end minimum_f_zero_inequality_n_r_S_floor_eq_211_l754_754987


namespace probability_of_rolling_green_face_l754_754601

theorem probability_of_rolling_green_face (total_faces green_faces : ℕ) (h_total : total_faces = 12)
  (h_red : 5) (h_blue : 4) (h_yellow : 2) (h_green : green_faces = 1) :
  (green_faces : ℚ) / total_faces = 1 / 12 :=
by
  simp [h_total, h_green]
  sorry

end probability_of_rolling_green_face_l754_754601


namespace recurring_sum_fractions_l754_754350

theorem recurring_sum_fractions :
  let x := (1 / 3) in
  let y := (4 / 99) in
  let z := (5 / 999) in
  x + y + z = (742 / 999) :=
by 
  sorry

end recurring_sum_fractions_l754_754350


namespace arc_length_of_intersection_l754_754413

noncomputable def minor_arc_length (O O1 : ℝ → ℝ → Prop) (MN : ℝ) : Prop :=
  O = (λ x y, x^2 + y^2 = 9) ∧
  O1 = (λ x y, (x-3)^2 + y^2 = 27) ∧
  MN = sqrt 3 * Real.pi

theorem arc_length_of_intersection :
  ∃ (MN : ℝ), minor_arc_length (λ x y, x^2 + y^2 = 9) (λ x y, (x-3)^2 + y^2 = 27) MN := 
begin
  use sqrt 3 * Real.pi,
  unfold minor_arc_length,
  refine ⟨rfl, rfl, rfl⟩,
end

end arc_length_of_intersection_l754_754413


namespace range_of_a_l754_754617

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 - 3 * a * x^2 + (2 * a + 1) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 - 6 * a * x + (2 * a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f' a x = 0 ∧ ∀ y : ℝ, f' a y ≠ 0) →
  (a > 1 ∨ a < -1 / 3) :=
sorry

end range_of_a_l754_754617


namespace brad_money_l754_754038

noncomputable def money_problem : Prop :=
  ∃ (B J D : ℝ), 
    J = 2 * B ∧
    J = (3/4) * D ∧
    B + J + D = 68 ∧
    B = 12

theorem brad_money : money_problem :=
by {
  -- Insert proof steps here if necessary
  sorry
}

end brad_money_l754_754038


namespace last_three_digits_2021st_term_l754_754543

noncomputable def s (n : ℕ) : ℕ := n^3
noncomputable def x (n : ℕ) : ℕ := s n - s (n - 1)

theorem last_three_digits_2021st_term : (x 2021) % 1000 = 261 :=
by 
  -- Add the proof steps here.
  sorry

end last_three_digits_2021st_term_l754_754543


namespace yellow_peaches_cannot_be_determined_l754_754639

noncomputable def yellow_peaches_indeterminable : Prop :=
  let r := 7
  let g := 8
  let condition := g = r + 1
  ∃ (y : ℕ), true → false

theorem yellow_peaches_cannot_be_determined (r g : ℕ) (h1 : r = 7) (h2 : g = 8) (h3 : g = r + 1) : yellow_peaches_indeterminable :=
by
  unfold yellow_peaches_indeterminable
  use 0  -- or any ℕ value as ∃ (y : ℕ), true → false is not satisfiable
  intro h
  contradiction

end yellow_peaches_cannot_be_determined_l754_754639


namespace cos_C_of_triangle_l754_754514

theorem cos_C_of_triangle (A B C : ℝ) (hA : sin A = 4 / 5) (hB : cos B = 12 / 13) :
  cos C = -16 / 65 :=
sorry

end cos_C_of_triangle_l754_754514


namespace distance_from_point_to_line_polar_l754_754959

theorem distance_from_point_to_line_polar :
  let P : ℝ × ℝ := (2, π / 6)
  let L : ℝ × ℝ → Prop := λ (ρ θ : ℝ), ρ * sin (θ - π / 6) = 1
  ∀ x y : ℝ, (x, y) = (sqrt 3, 1) →
              (∃ (A B C : ℝ), A * x + B * y + C = 0 ∧ 
               (A, B, C) = (1, -sqrt 3, 2)) →
              abs (1 * sqrt 3 + (-sqrt 3) * 1 + 2) / sqrt (1^2 + (sqrt 3)^2) = 1 :=
by
  intros
  sorry

end distance_from_point_to_line_polar_l754_754959


namespace cos_A_eq_one_third_area_of_triangle_ABC_l754_754483

-- Variable declarations for triangle ABC with given conditions
variables {A B C : Type} [Real.vector_space A] [Real.vector_space B] [Real.vector_space C]

-- Given conditions
variable (a b c : ℝ)
variable (cosA : ℝ)
variable (cosB : ℝ)

-- Question 1: Prove cos A = 1/3 given the particular condition
theorem cos_A_eq_one_third (h1 : a * (Real.cos B) = (3 * c - b) * (Real.cos A)) : Real.cos A = 1 / 3 :=
sorry

-- Additional variables for question 2
variable (M : Type) [Real.vector_space M]
variable (AM : Real.vector_space ℝ)
variable (area : ℝ)

-- Vector conditions
variable (AB AC AM : ℝ)

-- Question 2: Prove the area of triangle ABC is 7√2 given additional conditions
theorem area_of_triangle_ABC (h2 : b = 3) 
    (h3 : (vector_AM : vector_space ℝ AM -> ℝ) = 3 * Real.sqrt 2) 
    (h4 : cosA = 1 / 3) :
  area = 7 * Real.sqrt 2 :=
sorry

end cos_A_eq_one_third_area_of_triangle_ABC_l754_754483


namespace find_prime_p_l754_754367

theorem find_prime_p (p : ℕ) (h₀ : p.prime) (h₁ : p ≤ 1000) :
  (∃ m n : ℕ, n ≥ 2 ∧ 2 * p + 1 = m^n) ↔ p = 13 := by
sorry

end find_prime_p_l754_754367


namespace positive_integers_dividing_8n_sum_l754_754382

theorem positive_integers_dividing_8n_sum :
  {n : ℕ // n > 0 ∧ (∃ k : ℕ, 8 * n = k * (n * (n + 1) / 2))}.card = 4 := sorry

end positive_integers_dividing_8n_sum_l754_754382


namespace total_revenue_correct_l754_754682

noncomputable def total_revenue : ℚ := 
  let revenue_v1 := 23 * 5 * 0.50
  let revenue_v2 := 28 * 6 * 0.60
  let revenue_v3 := 35 * 7 * 0.50
  let revenue_v4 := 43 * 8 * 0.60
  let revenue_v5 := 50 * 9 * 0.50
  let revenue_v6 := 64 * 10 * 0.60
  revenue_v1 + revenue_v2 + revenue_v3 + revenue_v4 + revenue_v5 + revenue_v6

theorem total_revenue_correct : total_revenue = 1096.20 := 
by
  sorry

end total_revenue_correct_l754_754682


namespace cos_300_eq_half_l754_754343

theorem cos_300_eq_half : cos (300 * real.pi / 180) = 1 / 2 := by
  sorry

end cos_300_eq_half_l754_754343


namespace translate_point_left_l754_754064

theorem translate_point_left (A : ℝ × ℝ) (A_coords : A = (1, 2)) :
  let A1 := (A.1 - 1, A.2) in A1 = (0, 2) :=
by
  sorry

end translate_point_left_l754_754064


namespace koschei_never_equal_l754_754714

-- Define the problem setup 
def coins_at_vertices (n1 n2 n3 n4 n5 n6 : ℕ) : Prop := 
  ∃ k : ℕ, n1 = k ∧ n2 = k ∧ n3 = k ∧ n4 = k ∧ n5 = k ∧ n6 = k

-- Define the operation condition
def operation_condition (n1 n2 n3 n4 n5 n6 : ℕ) : Prop :=
  ∃ x : ℕ, (n1 - x = x ∧ n2 + 6 * x = x) ∨ (n2 - x = x ∧ n3 + 6 * x = x) ∨ 
  (n3 - x = x ∧ n4 + 6 * x = x) ∨ (n4 - x = x ∧ n5 + 6 * x = x) ∨ 
  (n5 - x = x ∧ n6 + 6 * x = x) ∨ (n6 - x = x ∧ n1 + 6 * x = x)

-- The main theorem 
theorem koschei_never_equal (n1 n2 n3 n4 n5 n6 : ℕ) : 
  (∃ x : ℕ, coins_at_vertices n1 n2 n3 n4 n5 n6) → False :=
by
  sorry

end koschei_never_equal_l754_754714


namespace num_x_for_3001_in_sequence_l754_754300

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l754_754300


namespace min_value_of_f_l754_754097

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 4*x + 5) * (x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = -9 :=
  sorry

end min_value_of_f_l754_754097


namespace max_at_x4_is_neg_9_over_2_l754_754090

noncomputable def find_a (a : ℝ) : Prop :=
  let f := λ x : ℝ, a / (x - 1) + 1 / (x - 2) + 1 / (x - 6)
  ∃ x : ℝ, (3 < x ∧ x < 5) ∧ (∀ y : ℝ, (3 < y ∧ y < 5) → f y ≤ f 4) ∧ x = 4

theorem max_at_x4_is_neg_9_over_2 (a : ℝ) (h : find_a a) : a = -9 / 2 :=
sorry

end max_at_x4_is_neg_9_over_2_l754_754090


namespace solution_l754_754408

noncomputable def is_monotonically_decreasing (a : ℝ) (x : ℝ) : Prop :=
  ∀ y z : ℝ, 0 < y → y < z → x = log a ((y + 1)) → log a ((z + 1)) < log a ((y + 1))

noncomputable def has_two_distinct_roots (a : ℝ) : Prop :=
  (2*a - 3)^2 - 4 > 0

noncomputable def condition (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ (is_monotonically_decreasing a 1 ∨ has_two_distinct_roots a)

noncomputable def either_but_not_both (a : ℝ) : Prop :=
  (is_monotonically_decreasing a 1 ∨ has_two_distinct_roots a) ∧ ¬(is_monotonically_decreasing a 1 ∧ has_two_distinct_roots a)

theorem solution (a : ℝ) (h : condition a) : either_but_not_both a → (1/2 ≤ a ∧ a < 1) ∨ (a > 5/2) :=
by sorry

end solution_l754_754408


namespace math_problem_l754_754065

noncomputable def parametric_eq_line := 
  λ t : ℝ, (6 + (real.sqrt 2)/2 * t, (real.sqrt 2)/2 * t)

noncomputable def polar_eq_curve := 
  λ θ : ℝ, 6 * real.cos θ

theorem math_problem (M : ℝ × ℝ) (A B : ℝ × ℝ) :
  (∀ t : ℝ, parametric_eq_line t = (6 + (real.sqrt 2)/2 * t, (real.sqrt 2)/2 * t)) →
  (∀ θ : ℝ, polar_eq_curve θ = 6 * real.cos θ) →
  (∃ l : ℝ → ℝ × ℝ,
    (∀ t : ℝ, l t = (-1 + (real.sqrt 2)/2 * t, (real.sqrt 2)/2 * t)) ∧
    ∃ t1 t2 : ℝ,
    (t1 + t2 = 4 * real.sqrt 2) ∧
    (t1 * t2 = 7) ∧
    (t1 > 0) ∧
    (t2 > 0) ∧
    (abs (t1 - t2) = 2)) :=
by
  sorry

end math_problem_l754_754065


namespace even_function_condition_range_of_g_l754_754002

-- Part (1): Given conditions and then prove m and f(x)
theorem even_function_condition (m : ℤ) (h_even : ∀ x : ℝ, x ^ (-2 * m ^ 2 + m + 3) = (-x) ^ (-2 * m ^ 2 + m + 3))
(h_ineq : 3 ^ (-2 * m ^ 2 + m + 3) < 5 ^ (-2 * m ^ 2 + m + 3)) :
  m = 1 ∧ ∀ x : ℝ, x ^ (-2 * 1 ^ 2 + 1 + 3) = x ^ 2 := 
sorry

-- Part (2): Given conditions for g(x) and proving the range
theorem range_of_g (a : ℝ) (h_a : a > 0) (h_a1 : a ≠ 1) :
  let g (x : ℝ) := log a (x ^ 2 - 2 * x) in
  (∀ x : ℝ, 2 < x ∧ x ≤ 3 → 0 < x ^ 2 - 2 * x ∧ x ^ 2 - 2 * x ≤ 3)
  → 
  ((a > 1 → ∀ x : ℝ, 2 < x ∧ x ≤ 3 → g(x) ∈ set.Iic (real.log a 3)) ∧
   (a < 1 → ∀ x : ℝ, 2 < x ∧ x ≤ 3 → g(x) ∈ set.Ici (real.log a 3)) ) := 
sorry

end even_function_condition_range_of_g_l754_754002


namespace num_five_digit_integers_with_product_900_l754_754029

theorem num_five_digit_integers_with_product_900 :
  { n : ℕ | n >= 10000 ∧ n <= 99999 ∧ (∏ c in n.digits 10, c) = 900 }.card = 210 :=
sorry

end num_five_digit_integers_with_product_900_l754_754029


namespace complex_z_sub_conjugate_eq_neg_i_l754_754843

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754843


namespace sqrt_inequality_pos_l754_754725

variable (y : ℝ)

theorem sqrt_inequality_pos (hy : y > 0) : sqrt (2 * y) < 3 * y ↔ y > 2 / 9 := by
  sorry

end sqrt_inequality_pos_l754_754725


namespace find_m_of_decreasing_power_function_l754_754159

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 - 2m - 3)

-- State the main theorem
theorem find_m_of_decreasing_power_function :
  (∀ x : ℝ, x > 0 → deriv (f m) x < 0) → (m = 2) :=
by
  -- Introduce assumptions
  intros h_decreasing

  -- Start by noting the form of f(x), and that f(x) is a decreasing power function
  sorry

end find_m_of_decreasing_power_function_l754_754159


namespace min_value_of_f_l754_754096

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 4*x + 5) * (x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = -9 :=
  sorry

end min_value_of_f_l754_754096


namespace mod_pow_eq_l754_754141

theorem mod_pow_eq (m : ℕ) (h1 : 13^4 % 11 = m) (h2 : 0 ≤ m ∧ m < 11) : m = 5 := by
  sorry

end mod_pow_eq_l754_754141


namespace trisha_take_home_pay_l754_754650

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end trisha_take_home_pay_l754_754650


namespace arithmetic_sequence_general_term_l754_754504

theorem arithmetic_sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = 5 * n^2 + 3 * n)
  (hS₁ : a 1 = S 1)
  (hS₂ : ∀ n, a (n + 1) = S (n + 1) - S n) :
  ∀ n, a n = 10 * n - 2 :=
by
  sorry

end arithmetic_sequence_general_term_l754_754504


namespace pugsley_spiders_l754_754119

variable (P W : ℕ)

theorem pugsley_spiders:
  (W + 2 = 9 * (P - 2)) →
  (P + 6 = W - 6) →
  P = 4 :=
by
  intro h1 h2
  have hW : W = P + 12 := by
    linarith
  rw [hW] at h1
  linarith

end pugsley_spiders_l754_754119


namespace parabola_focus_and_directrix_l754_754611

-- Declare the equation of the parabola as a condition
def parabola_equation (x y : ℝ) : Prop := y^2 = -8 * x

-- Define a theorem that states the coordinates of the focus and the equation of the directrix
theorem parabola_focus_and_directrix :
  (∀ x y : ℝ, parabola_equation x y → (x = -2 ∧ y = 0)) ∧ (∀ p : ℝ, parabola_equation p (sqrt (-8 * p)) → p / 2 = 2) :=
by
  sorry

end parabola_focus_and_directrix_l754_754611


namespace find_angle_B_l754_754933

theorem find_angle_B 
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 :=
sorry

end find_angle_B_l754_754933


namespace joan_initial_balloons_l754_754079

variable y : ℕ -- y represents the number of balloons Joan initially had.
variable h : y + 2 = 10 -- this is the condition given in the problem.

theorem joan_initial_balloons : y = 8 := 
by
  sorry

end joan_initial_balloons_l754_754079


namespace final_remaining_is_correct_total_sum_is_correct_l754_754122

open Nat

-- Definition of the initial sequence of numbers
def initial_sequence (n : ℕ) : List ℕ := List.range' 1 n

-- Definition of the sum of the first n natural numbers
def sum_of_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition of final remaining number
def final_remaining_number (n : ℕ) : ℕ :=
  sum_of_natural_numbers n

-- Definition of the number of operations
def n_operations (n : ℕ) : ℕ := (n + 15) / 16

-- Definition of the total sum of all numbers
noncomputable def total_sum_of_all_numbers (n : ℕ) : ℕ :=
  let initial_sum := sum_of_natural_numbers n
  in initial_sum + 100k {op_sum | op_sum = initial_sum * (\all 16th operation)} 

-- Theorem for the final remaining number
theorem final_remaining_is_correct : final_remaining_number 2011 = 2023066 :=
by
  exact (by norm_num : 2011 * 1007 = 2023066)

-- Theorem for the total sum of all numbers
theorem total_sum_is_correct : total_sum_of_all_numbers 2011 = 7822326 :=
sorry

end final_remaining_is_correct_total_sum_is_correct_l754_754122


namespace f_neg_a_l754_754430

noncomputable def f (x : ℝ) : ℝ := x^3 - sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 3) : f (-a) = -1 :=
by {
  let ha := (h : a^3 - sin a + 1 = 3),
  let key := (ha.symm ▸ 2 : a^3 - sin a = 2),
  have hf_neg := (-a)^3 - sin (-a) + 1,
  rw [←hf_neg, key],
  have sin_neg := sin (-a) = -sin a,
  rw [sin_neg],
  sorry
}

end f_neg_a_l754_754430


namespace distinct_solutions_diff_l754_754993

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l754_754993


namespace votes_for_candidate_A_l754_754942

noncomputable def totalVotes : ℕ := 560000
def invalidVotesPercentage : ℕ := 15
def validVotesPercentage := 100 - invalidVotesPercentage
def candidateAPercentage : ℂ := 75

def validVotes := (validVotesPercentage : ℂ) / 100 * (totalVotes : ℂ)
def votesForCandidateA := (candidateAPercentage / 100) * validVotes

theorem votes_for_candidate_A :
  votesForCandidateA = 357000 := by
  sorry

end votes_for_candidate_A_l754_754942


namespace sphere_radius_in_tetrahedron_l754_754965

variable (a r : ℝ)

theorem sphere_radius_in_tetrahedron (h1 : ∀ s₁ s₂ s₃ s₄ : Sphere, s₁.radius = r ∧ s₁.touches s₂ ∧ s₁.touches s₃ ∧ s₁.touches s₄)
    (h2 : ∀ f : Face, ∃ s : Sphere, s.touches f)
    (h3 : ∀ t : Tetrahedron, t.edge_length = a ∧ t.has_four_equal_spheres_inside)
    : r = (a * (Real.sqrt 6 - 1)) / 10 := sorry

end sphere_radius_in_tetrahedron_l754_754965


namespace sequence_a100_is_1175_l754_754939

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1010 ∧
  a 2 = 1012 ∧
  ∀ n : ℕ, 1 ≤ n → a n + 2 * a (n + 1) + a (n + 2) = 5 * n

theorem sequence_a100_is_1175
  (a : ℕ → ℤ)
  (h : sequence a) :
  a 100 = 1175 := 
sorry

end sequence_a100_is_1175_l754_754939


namespace remainder_when_sum_divided_by_15_l754_754205

theorem remainder_when_sum_divided_by_15 (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 12) 
  (h3 : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
  sorry

end remainder_when_sum_divided_by_15_l754_754205


namespace part1_A_intersect_B_part1_complement_union_A_B_part2_range_of_m_l754_754873

noncomputable def set_A : set ℝ := {x | 2 < x ∧ x ≤ 6}
noncomputable def set_B : set ℝ := {x | 0 < x ∧ x < 4}
noncomputable def set_C (m : ℝ) : set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}

theorem part1_A_intersect_B : set_A ∩ set_B = {x | 2 < x ∧ x < 4} :=
by {
  sorry
}

theorem part1_complement_union_A_B : compl (set_A ∪ set_B) = {x | x ≤ 0 ∨ x > 6} :=
by {
  sorry
}

theorem part2_range_of_m (m : ℝ) (h : set_C m ⊆ set_C m ∩ set_B) : m ≤ 5 / 2 :=
by {
  sorry
}

end part1_A_intersect_B_part1_complement_union_A_B_part2_range_of_m_l754_754873


namespace domain_of_f_x_squared_l754_754895

theorem domain_of_f_x_squared {f : ℝ → ℝ} (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ y, f (x ^ 2) = y := 
by 
  sorry

end domain_of_f_x_squared_l754_754895


namespace correct_answer_l754_754206

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f a < f b

def option_A := λ x : ℝ, x⁻²
def option_B := λ x : ℝ, x^2 + 3 * x + 2
def option_C := λ x : ℝ, Real.log x
def option_D := λ x : ℝ, 3^(abs x)

theorem correct_answer :
  (is_even option_D ∧ is_increasing_on option_D {x | 0 < x}) ∧
  ¬ (is_even option_A ∧ is_increasing_on option_A {x | 0 < x}) ∧
  ¬ (is_even option_B ∧ is_increasing_on option_B {x | 0 < x}) ∧
  ¬ (is_even option_C ∧ is_increasing_on option_C {x | 0 < x}) :=
by
  sorry

end correct_answer_l754_754206


namespace cosine_of_angle_between_a_and_b_k_values_for_perpendicular_vectors_l754_754909

open Real EuclideanGeometry

def point_A : ℝ × ℝ × ℝ := (-2, 0, 2)
def point_B : ℝ × ℝ × ℝ := (-1, 1, 2)
def point_C : ℝ × ℝ × ℝ := (-3, 0, 4)

def vector_a : ℝ × ℝ × ℝ := (1, 1, 0)
def vector_b : ℝ × ℝ × ℝ := (-1, 0, 2)

-- Assuming out usual Euclidean dot product and magnitude functions

noncomputable def cosine_between_vectors :=
  let dot_product := λ (u v : ℝ × ℝ × ℝ), u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let magnitude := λ (u : ℝ × ℝ × ℝ), Real.sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)
  (dot_product vector_a vector_b) / (magnitude vector_a * magnitude vector_b)

theorem cosine_of_angle_between_a_and_b : cosine_between_vectors = - Real.sqrt 10 / 10 :=
by
  -- Proof goes here
  sorry

theorem k_values_for_perpendicular_vectors (k : ℝ) :
  (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2, k * vector_a.3 + vector_b.3)
  • (k * vector_a.1 - 2 * vector_b.1, k * vector_a.2 - 2 * vector_b.2, k * vector_a.3 - 2 * vector_b.3) = 0 → 
  k = -5 / 2 ∨ k = 2 :=
by
  -- Proof goes here
  sorry

end cosine_of_angle_between_a_and_b_k_values_for_perpendicular_vectors_l754_754909


namespace find_a_l754_754379

theorem find_a (a : ℝ) : (∃ (a : ℝ), (∀ (w : ℂ), w = (1 + 2 * complex.i) / (a + complex.i) → w.im = 0) → a = 1 / 2) :=
begin
  sorry
end

end find_a_l754_754379


namespace incorrect_statements_l754_754186

-- Defining the sample points
variables (x y : ℕ → ℝ)
variable  (n : ℕ)
variable  (r : ℝ)

-- The correlation coefficient conditions for sample points
noncomputable def correlation_coefficient : ℝ := sorry

-- Statement definitions
def statement_A : Prop := (∀ i, y i = -2 * (x i) + 1) → correlation_coefficient = 1
def statement_B : Prop := (∀ i, y i = -2 * (x i) + 1) → correlation_coefficient = -2
def statement_C : Prop := (|correlation_coefficient| ≥ |r|) → stronger_linear_correlation
def statement_D : Prop := (|correlation_coefficient| ≤ |r|) → stronger_linear_correlation

-- Theorem stating that statements A, B, and D are incorrect
theorem incorrect_statements : ¬ (statement_A ∧ statement_B ∧ statement_C ∧ ¬statement_D) := 
by 
  sorry

end incorrect_statements_l754_754186


namespace boxes_in_attic_l754_754131

theorem boxes_in_attic (B : ℕ)
  (h1 : 6 ≤ B)
  (h2 : ∀ T : ℕ, T = (B - 6) / 2 ∧ T = 10)
  (h3 : ∀ O : ℕ, O = 180 + 2 * T ∧ O = 20 * T) :
  B = 26 :=
by
  sorry

end boxes_in_attic_l754_754131


namespace intersection_is_singleton_l754_754905

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_is_singleton : M ∩ N = {1} :=
by sorry

end intersection_is_singleton_l754_754905


namespace problem_statement_l754_754155

noncomputable def f : ℝ → ℝ := sorry

noncomputable def a : ℝ := Real.log 3 / Real.log 2

theorem problem_statement :
  (∀ x : ℝ, f(x) + f(x + 1) = 1) →
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) → f(x) = 2^x - x) →
  f(a) + f(2 * a) + f(3 * a) = 17 / 16 :=
sorry

end problem_statement_l754_754155


namespace proportion_red_MMs_l754_754258

theorem proportion_red_MMs (R B : ℝ) (h1 : R + B = 1) 
  (h2 : R * (4 / 5) = B * (1 / 6)) :
  R = 5 / 29 :=
by
  sorry

end proportion_red_MMs_l754_754258


namespace circle_representation_l754_754039

theorem circle_representation (a : ℝ): 
  (∃ (x y : ℝ), (x^2 + y^2 + 2*x + a = 0) ∧ (∃ D E F, D = 2 ∧ E = 0 ∧ F = -a ∧ (D^2 + E^2 - 4*F > 0))) ↔ (a > -1) :=
by 
  sorry

end circle_representation_l754_754039


namespace marie_erasers_l754_754112

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) 
  (h1 : initial_erasers = 95) (h2 : lost_erasers = 42) : final_erasers = 53 :=
by
  sorry

end marie_erasers_l754_754112


namespace ratio_of_hypothetical_to_actual_children_l754_754394

theorem ratio_of_hypothetical_to_actual_children (C H : ℕ) 
  (h1 : H = 16 * 8)
  (h2 : ∀ N : ℕ, N = C / 8 → C * N = 512) 
  (h3 : C^2 = 512 * 8) : H / C = 2 := 
by 
  sorry

end ratio_of_hypothetical_to_actual_children_l754_754394


namespace repeatingDecimalSum_is_fraction_l754_754361

noncomputable def repeatingDecimalSum : ℚ :=
  (0.3333...).val + (0.040404...).val + (0.005005...).val

theorem repeatingDecimalSum_is_fraction : repeatingDecimalSum = 1134 / 2997 := by
  sorry

end repeatingDecimalSum_is_fraction_l754_754361


namespace total_toys_correct_l754_754535

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l754_754535


namespace PK_eq_ML_l754_754577

-- Define the quadrilateral with perpendicular diagonals
-- Define the similarity of the triangles
def quadrilateral (A B C D M P L K : Point) : Prop :=
  Perpendicular (⟦A, C⟧) (⟦B, D⟧) ∧
  SimilarTriangle (Triangle.mk A B M) (Triangle.mk B C P) ∧ 
  SimilarTriangle (Triangle.mk C D L) (Triangle.mk D A K)

-- Assume quadrilateral ABCD with mentioned properties and prove PK = ML
theorem PK_eq_ML {A B C D M P L K : Point} :
  quadrilateral A B C D M P L K → dist P K = dist M L :=
by
  sorry

end PK_eq_ML_l754_754577


namespace max_value_lemma_l754_754549

noncomputable def max_value_abc (a b c : ℝ) : ℝ := 
  sqrt (2 * (a^2 + b^2 + c^2))

theorem max_value_lemma (a b c : ℝ) : 
  ∃ θ : ℝ, a * real.cos θ + b * real.sin θ + c * real.sin (2 * θ) = sqrt (2 * (a^2 + b^2 + c^2)) := 
sorry

end max_value_lemma_l754_754549


namespace cricketer_average_after_22nd_inning_l754_754710

theorem cricketer_average_after_22nd_inning (A : ℚ) 
  (h1 : 21 * A + 134 = (A + 3.5) * 22)
  (h2 : 57 = A) :
  A + 3.5 = 60.5 :=
by
  exact sorry

end cricketer_average_after_22nd_inning_l754_754710


namespace cos_C_in_triangle_l754_754523

theorem cos_C_in_triangle
  (A B C : ℝ) (sin_A : ℝ) (cos_B : ℝ)
  (h1 : sin_A = 4 / 5)
  (h2 : cos_B = 12 / 13) :
  cos (π - A - B) = -16 / 65 :=
by
  -- Proof steps would be included here
  sorry

end cos_C_in_triangle_l754_754523


namespace max_temp_range_l754_754610

theorem max_temp_range (temps : Fin 5 → ℝ)
  (h_avg : (∑ i in Finset.univ, temps i) / 5 = 45)
  (h_min : ∃ i, temps i = 42) : 
  ∃ r, r = 15 ∧ r = (Finset.max' Finset.univ (by apply Finset.nonempty_univ) - Finset.min' Finset.univ (by apply Finset.nonempty_univ)) :=
sorry

end max_temp_range_l754_754610


namespace find_minimum_value_l754_754371

theorem find_minimum_value (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : x > 0): 
  (∃ m : ℝ, ∀ x > 0, (a^2 + x^2) / x ≥ m ∧ ∃ x₀ > 0, (a^2 + x₀^2) / x₀ = m) :=
sorry

end find_minimum_value_l754_754371


namespace angle_URV_52_l754_754506

theorem angle_URV_52 
(PW QX SR TR : ℝ → ℝ) -- Presuming lines are defined by real functions
(S T U V : ℝ) -- Assuming points on real line
(angle_SUV angle_VTX angle_URV : ℝ)
(h1 : parallel PW QX)
(h2 : S ∈ QX ∧ T ∈ QX)
(h3 : intersects PW SR U)
(h4 : intersects PW TR V)
(h5 : angle_SUV = 120)
(h6 : angle_VTX = 112) : 
angle_URV = 52 := 
sorry

end angle_URV_52_l754_754506


namespace abs_difference_of_numbers_l754_754177

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) : |x - y| = 4 * Real.sqrt 10 :=
by
  sorry

end abs_difference_of_numbers_l754_754177


namespace complex_conj_difference_l754_754835

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754835


namespace number_of_divisors_l754_754384

theorem number_of_divisors (n : ℕ) :
  n > 0 → ∃! n : ℕ, (∀ m : ℕ, (∑ i in Finset.range (n + 1), i + 1) ∣ 8 * n) ∧ n ∈ {1, 3, 7, 15} :=
by
  sorry

end number_of_divisors_l754_754384


namespace complex_conj_difference_l754_754840

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754840


namespace decreasing_function_range_l754_754889

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else -a * x

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (1 / 8 ≤ a ∧ a < 1 / 3) :=
by
  sorry

end decreasing_function_range_l754_754889


namespace solve_system_l754_754033

theorem solve_system (s t : ℚ) (h1 : 7 * s + 6 * t = 156) (h2 : s = t / 2 + 3) : s = 192 / 19 :=
sorry

end solve_system_l754_754033


namespace tangent_lines_value_count_l754_754192

theorem tangent_lines_value_count (r s d : ℝ) (hr : r > s) : 
  ∃ k_values : Finset ℕ, (k_values = {0, 1, 2, 3, 4}) ∧ 
  ∀ (k ∈ k_values), k = if d = 0 then 0 
    else if d = r - s then 1 
    else if d = r + s then 3 
    else if d > r + s then 4 
    else 2 :=
by
  sorry

end tangent_lines_value_count_l754_754192


namespace adam_spent_on_ferris_wheel_l754_754220

-- Define the conditions
def ticketsBought : Nat := 13
def ticketsLeft : Nat := 4
def costPerTicket : Nat := 9

-- Define the question and correct answer as a proof goal
theorem adam_spent_on_ferris_wheel : (ticketsBought - ticketsLeft) * costPerTicket = 81 := by
  sorry

end adam_spent_on_ferris_wheel_l754_754220


namespace distinct_solutions_diff_l754_754991

theorem distinct_solutions_diff {r s : ℝ} 
  (h1 : ∀ a b : ℝ, (a ≠ b → ( ∃ x, x ≠ a ∧ x ≠ b ∧ (x = r ∨ x = s) ∧ (x-5)(x+5) = 25*x - 125) )) 
  (h2 : r > s) : r - s = 15 :=
by
  sorry

end distinct_solutions_diff_l754_754991


namespace z_conjugate_difference_l754_754851

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754851


namespace cos_C_given_sin_A_and_cos_B_l754_754517

theorem cos_C_given_sin_A_and_cos_B (A B C : ℝ) (h_triangle : A + B + C = real.pi)
  (h_sinA : real.sin A = 4 / 5) (h_cosB : real.cos B = 12 / 13) :
  real.cos C = -16 / 65 :=
sorry

end cos_C_given_sin_A_and_cos_B_l754_754517


namespace equilateral_triangles_centroid_area_diff_equivalence_l754_754867

variables {Δ δ : Type*}

theorem equilateral_triangles_centroid_area_diff_equivalence
  (ABC : Triangle)
  (A1 B1 C1 A2 B2 C2 : Point)
  (h_outward_eq : are_equilateral_triangles A1 B1 C1 ABC)
  (h_inward_eq : are_equilateral_triangles A2 B2 C2 ABC)
  (centroid_ABC : Point)
  (H1 : is_centroid centroid_ABC ABC)
  (H2 : centroid_ABC = centroid_of_triangle Δ)
  (H3 : centroid_ABC = centroid_of_triangle δ) :
  (is_equilateral_triangle Δ) ∧ 
  (is_equilateral_triangle δ) ∧ 
  (area_difference Δ δ = area ABC) :=
begin
  sorry
end

end equilateral_triangles_centroid_area_diff_equivalence_l754_754867


namespace average_student_headcount_l754_754668

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end average_student_headcount_l754_754668


namespace find_larger_number_l754_754929

theorem find_larger_number 
  (x y : ℚ) 
  (h1 : 4 * y = 9 * x) 
  (h2 : y - x = 12) : 
  y = 108 / 5 := 
sorry

end find_larger_number_l754_754929


namespace minimum_value_x2_minus_x1_range_of_a_l754_754892

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) := a * x
noncomputable def F (x : ℝ) (a : ℝ) := f x - g x a

-- Question (I)
theorem minimum_value_x2_minus_x1 : ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ a = 1 / 3 ∧ f x₁ = g x₂ a → x₂ - x₁ = 3 := 
sorry

-- Question (II)
theorem range_of_a (a : ℝ) : (∀ x ≥ 0, F x a ≥ F (-x) a) ↔ a ≤ 2 :=
sorry

end minimum_value_x2_minus_x1_range_of_a_l754_754892


namespace closet_area_l754_754211

noncomputable def diagonal_closet_area : ℝ :=
  let d := 7 in
  let w := 4 in
  let l := Real.sqrt (d^2 - w^2) in
  w * l

theorem closet_area : abs (diagonal_closet_area - 22.96) < 0.01 :=
by
  let d := 7
  let w := 4
  have hd : d^2 = 49 := by norm_num
  have hw : w^2 = 16 := by norm_num
  have hl := Real.sqrt_eq (hd.symm ▸ sub_self_sub hw)
  let l := Real.sqrt (d^2 - w^2)
  let area := w * l
  have harea : area = 4 * ‖Real.sqrt 33‖ := sorry
  show abs (area - 22.96) < 0.01
  sorry

end closet_area_l754_754211


namespace sin_cos_combination_l754_754268

theorem sin_cos_combination :
  sin (-120 * (π / 180)) * cos (1290 * (π / 180)) + cos (-1020 * (π / 180)) * sin (-1050 * (π / 180)) = 1 :=
by
  sorry

end sin_cos_combination_l754_754268


namespace max_non_intersecting_segments_l754_754403

theorem max_non_intersecting_segments (n m : ℕ) (hn: 1 < n) (hm: m ≥ 3): 
  ∃ L, L = 3 * n - m - 3 :=
by
  sorry

end max_non_intersecting_segments_l754_754403


namespace min_s_value_l754_754632

theorem min_s_value (n : ℕ) (x : fin n → ℝ)
  (h1 : ∀ i, x i > 0)
  (h2 : ∑ i, x i = 1) :
  ∃ s, s = 1 - (2 : ℝ) ^ (-1 / n) ∧
  s = max ((λ i, x i / (1 + (∑ j in finset.range i, x ⟨j, fin.is_lt j⟩))) (fin n)) :=
sorry

end min_s_value_l754_754632


namespace percentage_of_girls_passed_l754_754496

/-- In an examination with 2000 candidates, out of which 900 are girls and the remaining are boys, 
  if 38% of the boys passed and the total percentage of failed candidates is 64.7%, 
  then 32% of the girls passed the examination. -/
theorem percentage_of_girls_passed (total_candidates girls boys : ℕ) (boys_passed_percentage : ℝ)
  (total_failed_percentage : ℝ) (girls_passed_percentage : ℝ) :
  total_candidates = 2000 ∧
  girls = 900 ∧
  boys = total_candidates - girls ∧
  boys_passed_percentage = 0.38 ∧
  total_failed_percentage = 0.647 ∧
  (girls_passed_percentage = 
    ((total_candidates - (total_failed_percentage * total_candidates).to_nat - 
     (boys_passed_percentage * boys).to_nat) / girls.to_nat) * 100) →
  girls_passed_percentage = 32 := 
by
  intros h,
  obtain ⟨hc1, hc2, hc3, hc4, hc5, hc6⟩ := h,
  sorry

end percentage_of_girls_passed_l754_754496


namespace largest_possible_n_l754_754641
open scoped Classical

def board_range := (-10 : ℤ) .. 10

def is_red : ℤ → Prop := sorry -- the coloring function should be defined.
def sum_red_squares (is_red: ℤ → Prop) : ℤ := 
  (Finset.Ico board_range.lower board_range.upper).filter is_red 
  |>.sum id

def fair_coin_flip_steps : List (ℕ → ℤ) := 
  List.replicate 10 (fun k => if k % 2 = 0 then 1 else -1)

def ending_square (steps: List (ℕ → ℤ)) (initial: ℤ) : ℤ :=
  steps.foldl (fun acc step_fun => acc + step_fun acc) initial

def prob_finishes_red_square (is_red: ℤ → Prop) (steps: List (ℕ → ℤ)) : ℚ :=
  let outcomes := steps.permus.map (ending_square 0)
  let favorable := outcomes.filter is_red
  favorable.length / outcomes.length

theorem largest_possible_n (is_red: ℤ → Prop) (n: ℤ) (a: ℕ) (b: ℕ) :
  n = sum_red_squares is_red ∧ prob_finishes_red_square is_red fair_coin_flip_steps = (a : ℚ) / b ∧ a + b = 2001
  → n = 3 := 
sorry

end largest_possible_n_l754_754641


namespace complex_z_sub_conjugate_eq_neg_i_l754_754846

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754846


namespace card_trick_sum_l754_754706

theorem card_trick_sum (n k S : ℕ)
  (cards : Finset ℕ)
  (h_total_cards : cards.card = 52)
  (face_value : ∀ {card : ℕ}, card ∈ cards → card = 11 ∨ card = 12 ∨ card = 13 → card := 10 )
  (h_stack_condition : ∀ {v : ℕ} (hv : v ∈ cards), v + Finset.card (Finset.Ico v 12) = 13)
  (h_pile_left : Finset.card (Finset.Ico (12 - v) 13) = k)
  : S = 13 * (n - 4) + k := by
  sorry

end card_trick_sum_l754_754706


namespace apple_juice_amount_correct_l754_754603

noncomputable def apple_juice (T : ℝ) (pc pf pj : ℝ) : ℝ :=
  let apples_for_cider := pc * T
  let remaining_apples := T - apples_for_cider
  let fresh_apples := pf * remaining_apples
  let exported_apples := remaining_apples - fresh_apples
  let juice_apples := pj * exported_apples
  juice_apples

theorem apple_juice_amount_correct :
  apple_juice 6 0.30 0.40 0.60 ≈ 1.5 :=
by
  sorry

end apple_juice_amount_correct_l754_754603


namespace sequence_a_sum_T_l754_754864

noncomputable def S (a : ℕ → ℝ) (n : ℕ) (h : n > 0) : ℝ := (3 / 2) * a n - 1

def a_n (n : ℕ) : ℝ :=
  if h : n > 0 then nat.rec_on n 2 (λ n' ih, 3 * ih) else 0

def b_n (n : ℕ) : ℤ :=
  match n with
  | 0     => 5
  | n + 1 => b_n n + a_n n

def T (n : ℕ) : ℝ := ∑ i in range n, (log 9 (b_n i - 4))

theorem sequence_a (n : ℕ) (h : n > 0) : a_n n = 2 * 3^(n-1) :=
  sorry

theorem sum_T (n : ℕ) : T n = (n^2 - n) / 4 :=
  sorry

end sequence_a_sum_T_l754_754864


namespace skirt_price_is_13_l754_754531

-- Definitions based on conditions
def skirts_cost (S : ℝ) : ℝ := 2 * S
def blouses_cost : ℝ := 3 * 6
def total_cost (S : ℝ) : ℝ := skirts_cost S + blouses_cost
def amount_spent : ℝ := 100 - 56

-- The statement we want to prove
theorem skirt_price_is_13 (S : ℝ) (h : total_cost S = amount_spent) : S = 13 :=
by sorry

end skirt_price_is_13_l754_754531


namespace binomial_19_10_l754_754286

theorem binomial_19_10 :
  ∀ (binom : ℕ → ℕ → ℕ),
  binom 17 7 = 19448 → binom 17 9 = 24310 →
  binom 19 10 = 92378 :=
by
  intros
  sorry

end binomial_19_10_l754_754286


namespace maxwell_distance_20_l754_754689

theorem maxwell_distance_20 :
  ∀ (distance_homes : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ),
  distance_homes = 50 ∧ maxwell_speed = 4 ∧ brad_speed = 6 →
  ∃ d : ℝ, d = 20 ∧ (d / maxwell_speed) = ((distance_homes - d) / brad_speed) :=
by
  intros distance_homes maxwell_speed brad_speed h,
  rcases h with ⟨h_distance_homes, h_maxwell_speed, h_brad_speed⟩,
  use 20,
  split,
  { refl },
  { rw [h_distance_homes, h_maxwell_speed, h_brad_speed],
    field_simp,
    norm_num }

end maxwell_distance_20_l754_754689


namespace projection_onto_3_neg4_l754_754798

noncomputable def projection_matrix (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 in
  let dot_vv := v.1 * v.1 + v.2 * v.2 in
  ((dot_uv / dot_vv) * v.1, (dot_uv / dot_vv) * v.2)

theorem projection_onto_3_neg4 :
  projection_matrix (3, -4) = λ u, ((9 / 25) * u.1 + (-12 / 25) * u.2, (-12 / 25) * u.1 + (16 / 25) * u.2) :=
by
  sorry

end projection_onto_3_neg4_l754_754798


namespace valid_votes_for_A_l754_754945

open Nat

noncomputable def valid_votes (total_votes : ℕ) (invalid_percentage : ℤ) : ℕ :=
  total_votes * (100 - invalid_percentage.toNat) / 100

noncomputable def votes_for_candidate (valid_votes : ℕ) (candidate_percentage : ℤ) : ℕ :=
  valid_votes * candidate_percentage.toNat / 100

theorem valid_votes_for_A (total_votes : ℕ)
  (invalid_percentage : ℤ) (candidate_percentage : ℤ) (valid_votes_for_A : ℕ) :
  total_votes = 560000 → invalid_percentage = 15 →
  candidate_percentage = 75 → valid_votes_for_A = 357000 :=
by
  intros h_total_votes h_invalid_percentage h_candidate_percentage
  have h_valid_votes := valid_votes total_votes invalid_percentage
  have h_votes_for_A := votes_for_candidate h_valid_votes candidate_percentage
  rw [h_total_votes, h_invalid_percentage, h_candidate_percentage] at h_valid_votes h_votes_for_A
  simp only [valid_votes] at h_valid_votes
  simp only [votes_for_candidate] at h_votes_for_A
  rw [show (560000 * 85 / 100) = 476000, from sorry] at h_valid_votes
  rw [show (476000 * 75 / 100) = 357000, from sorry] at h_votes_for_A
  exact h_votes_for_A

end valid_votes_for_A_l754_754945


namespace average_of_multiplied_numbers_l754_754609

theorem average_of_multiplied_numbers {α : Type} [LinearOrderedField α] (a : Fin 7 → α) 
  (h_avg : (∑ i, a i) / 7 = 26) : ((∑ i, 5 * a i) / 7 = 130) :=
sorry

end average_of_multiplied_numbers_l754_754609


namespace exists_filling_table_with_sums_if_and_only_if_even_l754_754391

theorem exists_filling_table_with_sums_if_and_only_if_even (n : ℕ) :
  (∃ T : matrix (fin n) (fin n) ℕ,
     (∀ i, T i = 0 ∨ T i = 1 ∨ T i = 2) ∧
     (∀ v, ∃ i, (∑ x, T i x = v) ∧ (∑ y, T y i = v)) ∧
     (∀ v w, v ≠ w → (∃ i, ∑ x, T i x = v) → (∃ j, ∑ y, T y j = v) →
            (∃ i', ∑ x, T i' x = w) → (∃ j', ∑ y, T j' j = w))) ↔ even n :=
sorry

end exists_filling_table_with_sums_if_and_only_if_even_l754_754391


namespace total_bananas_in_collection_l754_754150

-- Definitions based on the conditions
def group_size : ℕ := 18
def number_of_groups : ℕ := 10

-- The proof problem statement
theorem total_bananas_in_collection : group_size * number_of_groups = 180 := by
  sorry

end total_bananas_in_collection_l754_754150


namespace arithmetic_sqrt_of_sqrt_16_is_2_l754_754148

-- Define the conditions
def sqrt_of_16 : ℝ := real.sqrt 16
def sqrt_of_4 : ℝ := real.sqrt 4

-- State the theorem
theorem arithmetic_sqrt_of_sqrt_16_is_2 :
  sqrt_of_4 = 2 :=
by
  -- Skip the proof
  sorry

end arithmetic_sqrt_of_sqrt_16_is_2_l754_754148


namespace shortest_distance_to_line_l754_754881

theorem shortest_distance_to_line :
  ∀ (P : ℝ × ℝ), (P.1 - 2)^2 + P.2^2 = 1 → 
  let l := λ (P : ℝ × ℝ), P.1 - P.2 + 2 = 0 in
  abs ((P.1 - P.2 + 2) / Real.sqrt (1^2 + (-1)^2)) - 1 = 2 * Real.sqrt 2 - 1 :=
by
  sorry

end shortest_distance_to_line_l754_754881


namespace remarkablePolygonExistsWith4Cells_remarkablePolygonExistsWithNCells_l754_754660

-- Definitions based on conditions identified
def isRectangle (polygon : Set (ℕ × ℕ)) : Prop :=
  ∃ a b : ℕ, polygon = { (i, j) | i < a ∧ j < b }

def isRemarkable (polygon : Set (ℕ × ℕ)) : Prop :=
  ¬ isRectangle polygon ∧ ∃ k : ℕ, ∃ subPolygons : Finset (Set (ℕ × ℕ)),
    subPolygons.card = k ∧ 
    (∀ p ∈ subPolygons, isSimular polygon p) ∧ 
    polygon = subPolygons.bUnion id

-- Theorem (a): existence of a remarkable polygon with 4 cells
theorem remarkablePolygonExistsWith4Cells : ∃ polygon : Set (ℕ × ℕ), 
  polygon.card = 4 ∧ isRemarkable polygon :=
sorry

-- Theorem (b): existence of remarkable polygon for any n > 4
theorem remarkablePolygonExistsWithNCells (n : ℕ) : n > 4 → ∃ polygon : Set (ℕ × ℕ), 
  polygon.card = n ∧ isRemarkable polygon :=
sorry

end remarkablePolygonExistsWith4Cells_remarkablePolygonExistsWithNCells_l754_754660


namespace votes_for_candidate_A_l754_754943

noncomputable def totalVotes : ℕ := 560000
def invalidVotesPercentage : ℕ := 15
def validVotesPercentage := 100 - invalidVotesPercentage
def candidateAPercentage : ℂ := 75

def validVotes := (validVotesPercentage : ℂ) / 100 * (totalVotes : ℂ)
def votesForCandidateA := (candidateAPercentage / 100) * validVotes

theorem votes_for_candidate_A :
  votesForCandidateA = 357000 := by
  sorry

end votes_for_candidate_A_l754_754943


namespace june_time_to_bernard_l754_754976

theorem june_time_to_bernard (distance_Julia : ℝ) (time_Julia : ℝ) (distance_Bernard_June : ℝ) (time_Bernard : ℝ) (distance_June_Bernard : ℝ)
  (h1 : distance_Julia = 2) (h2 : time_Julia = 6) (h3 : distance_Bernard_June = 5) (h4 : time_Bernard = 15) (h5 : distance_June_Bernard = 7) :
  distance_June_Bernard / (distance_Julia / time_Julia) = 21 := by
    sorry

end june_time_to_bernard_l754_754976


namespace multiplication_in_A_l754_754402

def A : Set ℤ :=
  {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem multiplication_in_A (x1 x2 : ℤ) (h1 : x1 ∈ A) (h2 : x2 ∈ A) :
  x1 * x2 ∈ A :=
sorry

end multiplication_in_A_l754_754402


namespace num_regions_l754_754068

-- Define conditions
def lines_not_parallel (n : ℕ) (lines : List (List ℝ)) : Prop :=
  ∀ (i j : ℕ), i < j → j < n → lines i ≠ lines j

def no_three_lines_same_point (n : ℕ) (lines : List (List ℝ)) : Prop :=
  ∀ (i j k : ℕ), i < j → j < k → k < n → 
  ∀ (p : ℝ × ℝ), (p ∈ lines i) → (p ∈ lines j) → (p ∈ lines k) → false

-- Define number of regions function
def f (n : ℕ) : ℕ := 
  if h : n > 0 then 
    (n^2 + n + 2) / 2 
  else 
    0

-- Lean statement to prove
theorem num_regions (n : ℕ) (h₁ : n > 0) (lines : List (List ℝ)) 
  (h₂ : lines_not_parallel n lines) 
  (h₃ : no_three_lines_same_point n lines) : 
  (f 3 = 7) ∧ (f n = (n^2 + n + 2) / 2) :=
by sorry

end num_regions_l754_754068


namespace repeating_decimal_sum_l754_754356

theorem repeating_decimal_sum : (0.\overline{3} : ℚ) + (0.\overline{04} : ℚ) + (0.\overline{005} : ℚ) = 1135 / 2997 := 
sorry

end repeating_decimal_sum_l754_754356


namespace non_zero_digits_fraction_l754_754265

def count_non_zero_digits (n : ℚ) : ℕ :=
  -- A placeholder for the actual implementation.
  sorry

theorem non_zero_digits_fraction : count_non_zero_digits (120 / (2^4 * 5^9 : ℚ)) = 3 :=
  sorry

end non_zero_digits_fraction_l754_754265


namespace bill_difference_proof_l754_754253

variable (a b c : ℝ)

def alice_condition := (25/100) * a = 5
def bob_condition := (20/100) * b = 6
def carol_condition := (10/100) * c = 7

theorem bill_difference_proof (ha : alice_condition a) (hb : bob_condition b) (hc : carol_condition c) :
  max a (max b c) - min a (min b c) = 50 :=
by sorry

end bill_difference_proof_l754_754253


namespace find_k_if_f_even_l754_754040

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := log 9 (9^x + 1) + k * x

theorem find_k_if_f_even (k : ℝ) : 
  (∀ x : ℝ, f x k = f (-x) k) → k = -1 := 
by
  sorry

end find_k_if_f_even_l754_754040


namespace function_inequality_l754_754802

theorem function_inequality 
  (f : ℝ → ℝ) 
  (h_diff : ∀ x ∈ Ioo 0 (π / 2), differentiable_at ℝ f x)
  (h_ineq : ∀ x ∈ Ioo 0 (π / 2), tan x * (deriv f x) < f x) :
  f (π / 6) * sin 1 > (1 / 2) * f 1 :=
sorry

end function_inequality_l754_754802


namespace lcm_pairs_count_2000_l754_754028

-- Definitions based on conditions
def is_lcm_2000 (a b : ℕ) : Prop :=
  Nat.lcm a b = 2000

noncomputable def num_pairs_with_lcm_2000 : ℕ :=
  32

-- Statement to prove
theorem lcm_pairs_count_2000 :
  {p : ℕ × ℕ | is_lcm_2000 p.1 p.2}.card = num_pairs_with_lcm_2000 := 
sorry

end lcm_pairs_count_2000_l754_754028


namespace square_triangle_equal_area_l754_754507

theorem square_triangle_equal_area (x : ℝ) :
  let side := 16 in
  let square_area := 16 * 16 in
  let triangle_area := (1 / 2) * 64 * x in
  (32 * x = square_area) → x = 8 :=
by
  intros _ _
  sorry

end square_triangle_equal_area_l754_754507


namespace find_d_squared_l754_754230

noncomputable def g (c d : ℝ) (z : ℂ) : ℂ := (c + d*complex.I) * z

theorem find_d_squared (c d : ℝ) (h1 : ∀ z : ℂ, complex.abs (g c d z - z) = complex.abs (g c d z))
(h2 : complex.abs (c + d*complex.I) = 7) : d^2 = 195 / 4 := by
  sorry

end find_d_squared_l754_754230


namespace seashells_at_end_of_month_l754_754788

-- Given conditions as definitions
def initial_seashells : ℕ := 50
def increase_per_week : ℕ := 20

-- Define function to calculate seashells in the nth week
def seashells_in_week (n : ℕ) : ℕ :=
  initial_seashells + n * increase_per_week

-- Lean statement to prove the number of seashells in the jar at the end of four weeks is 130
theorem seashells_at_end_of_month : seashells_in_week 4 = 130 :=
by
  sorry

end seashells_at_end_of_month_l754_754788


namespace ava_first_coupon_day_l754_754754

theorem ava_first_coupon_day (first_coupon_day : ℕ) (coupon_interval : ℕ) 
    (closed_day : ℕ) (days_in_week : ℕ):
  first_coupon_day = 2 →  -- starting on Tuesday (considering Monday as 1)
  coupon_interval = 13 →
  closed_day = 7 →        -- Saturday is represented by 7
  days_in_week = 7 →
  ∀ n : ℕ, ((first_coupon_day + n * coupon_interval) % days_in_week) ≠ closed_day :=
by 
  -- Proof can be filled here.
  sorry

end ava_first_coupon_day_l754_754754


namespace average_headcount_correct_l754_754663

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end average_headcount_correct_l754_754663


namespace probability_of_2_gold_no_danger_l754_754742

variable (caves : Finset Nat) (n : Nat)

-- Probability definitions
def P_gold_no_danger : ℚ := 1 / 5
def P_danger_no_gold : ℚ := 1 / 10
def P_neither : ℚ := 4 / 5

-- Probability calculation
def P_exactly_2_gold_none_danger : ℚ :=
  10 * (P_gold_no_danger) ^ 2 * (P_neither) ^ 3

theorem probability_of_2_gold_no_danger :
  (P_exactly_2_gold_none_danger) = 128 / 625 :=
sorry

end probability_of_2_gold_no_danger_l754_754742


namespace projection_onto_3_neg4_l754_754797

noncomputable def projection_matrix (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 in
  let dot_vv := v.1 * v.1 + v.2 * v.2 in
  ((dot_uv / dot_vv) * v.1, (dot_uv / dot_vv) * v.2)

theorem projection_onto_3_neg4 :
  projection_matrix (3, -4) = λ u, ((9 / 25) * u.1 + (-12 / 25) * u.2, (-12 / 25) * u.1 + (16 / 25) * u.2) :=
by
  sorry

end projection_onto_3_neg4_l754_754797


namespace janet_percentage_l754_754970

-- Define the number of snowballs made by Janet and her brother
def janet_snowballs : ℕ := 50
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage function
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- State the theorem to be proved
theorem janet_percentage : percentage janet_snowballs total_snowballs = 25 := by
  sorry

end janet_percentage_l754_754970


namespace compare_negatives_l754_754283

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l754_754283


namespace find_minimum_value_of_f_l754_754099

def f (x : ℝ) : ℝ := (x ^ 2 + 4 * x + 5) * (x ^ 2 + 4 * x + 2) + 2 * x ^ 2 + 8 * x + 1

theorem find_minimum_value_of_f : ∃ x : ℝ, f x = -9 :=
by
  sorry

end find_minimum_value_of_f_l754_754099


namespace part1_part2_part3_l754_754340

-- Part (1)
theorem part1 (A B C : ℤ × ℤ) (P : ℤ × ℤ → Prop)
  (hP_A : A = (-1, 2)) (hP_B : B = (4, -3)) (hP_C : C = (-3, 4)) :
  ¬ P A ∧ ¬ P B ∧ P C :=
by {
  let P := fun (x y: ℤ) => 2 * x + 3 * y = 6,
  sorry
}

-- Part (2)
theorem part2 (m n : ℤ) (P : ℝ × ℤ → Prop) :
  (0 ≤ m) ∧ (0 ≤ n) ∧ 2 * n - real.sqrt m = 1 ∧ P (real.sqrt m, n) → real.sqrt (2 * m - n) = 4 :=
by {
  let P := fun (x : ℝ) (y : ℤ) => 2 * x + y = 8,
  sorry
}

-- Part (3)
theorem part3 (k : ℕ) (P : ℕ × ℤ → Prop) :
  (1 ≤ k) ∧ P (k) :=
by {
  let P := fun (x y: ℤ) => 2 * x + y = 1 ∧ k * x + 2 * y = 5,
  sorry
}

end part1_part2_part3_l754_754340


namespace max_volume_tetrahedron_l754_754510

theorem max_volume_tetrahedron (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ P : ℝ, (P ∈ Icc 0 (√(a*a + b*b)) ∧
  volume_tetrahedron a b ha hb P = (1 / 6) * (a^2 * b^2) / (a^(2/3) + b^(2/3))^(3/2)) := 
sorry

end max_volume_tetrahedron_l754_754510


namespace hands_straight_line_time_l754_754759

noncomputable def time_when_hands_straight_line : List (ℕ × ℚ) :=
  let x₁ := 21 + 9 / 11
  let x₂ := 54 + 6 / 11
  [(4, x₁), (4, x₂)]

theorem hands_straight_line_time :
  time_when_hands_straight_line = [(4, 21 + 9 / 11), (4, 54 + 6 / 11)] :=
by
  sorry

end hands_straight_line_time_l754_754759


namespace product_of_real_roots_leq_half_power_l754_754901

theorem product_of_real_roots_leq_half_power {n : ℕ} (a : Fin n → ℝ) 
  (h₀ : n ≥ 1) (h₁ : (∀ i : Fin n, 0 < (a i) ∧ (a i) < 1))
  (h₂ : |(∏ i : Fin n, a i)| = (∏ i : Fin n, 1 - (a i))) :
  (∏ i : Fin n, a i) ≤ (1 / (2 : ℝ)) ^ n :=
by
  sorry

end product_of_real_roots_leq_half_power_l754_754901


namespace central_angle_is_correct_area_of_sector_is_correct_l754_754493

-- Define the conditions
def r : ℝ := 8
def l : ℝ := 12

-- Define the central angle calculation
def α : ℝ := l / r

-- Define the area of the sector calculation
def S : ℝ := (1 / 2) * l * r

-- State the proof problems
theorem central_angle_is_correct : α = 3 / 2 :=
by
  sorry

theorem area_of_sector_is_correct : S = 48 :=
by
  sorry

end central_angle_is_correct_area_of_sector_is_correct_l754_754493


namespace sequence_x_values_3001_l754_754309

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l754_754309


namespace f_at_pi_over_4_l754_754428

noncomputable def f (x : ℝ) := let c := f' (π / 4) in c * Real.cos x + Real.sin x

theorem f_at_pi_over_4 (c : ℝ) (h : ∀ x, f x = c * Real.cos x + Real.sin x) :
  f (π / 4) = 1 :=
by
  sorry

end f_at_pi_over_4_l754_754428


namespace dihedral_angle_D1_AE_C_BF_parallel_to_AD1E_l754_754509

structure RectangularPrism (V : Type) [inner_product_space ℝ V] :=
(A : V) (B : V) (C : V) (D : V) (A1 : V) (B1 : V) (C1 : V) (D1 : V)
(A_A1 : ∥A1 - A∥ = 2)
(square_base : ∥B - A∥ = 1 ∧ ∥D - A∥ = 1 ∧ orthonormal ℝ ![B - A, D - A])

def midpoint {V : Type} [inner_product_space ℝ V] (P Q : V) : V := (P + Q) / 2

variables {V : Type} [inner_product_space ℝ V]
variables (P : RectangularPrism V)

def E := midpoint P.B1 P.B
def F := midpoint P.D P.A

theorem dihedral_angle_D1_AE_C : 
  let AE := submodule.span ℝ ![P.A, E] in
  let planes := submodule.span ℝ ![P.D1, E] in
  ∠((P.D1 - P.A), AE : submodule ℝ V) = 90 :=
sorry

theorem BF_parallel_to_AD1E :
  let AD1E := submodule.span ℝ ![P.A, P.D1, E] in
  (P.B - F) ∈ module.span ℝ ![AD1E] :=
sorry

end dihedral_angle_D1_AE_C_BF_parallel_to_AD1E_l754_754509


namespace distinct_solutions_diff_l754_754988

theorem distinct_solutions_diff {r s : ℝ} 
  (h1 : ∀ a b : ℝ, (a ≠ b → ( ∃ x, x ≠ a ∧ x ≠ b ∧ (x = r ∨ x = s) ∧ (x-5)(x+5) = 25*x - 125) )) 
  (h2 : r > s) : r - s = 15 :=
by
  sorry

end distinct_solutions_diff_l754_754988


namespace shape_is_line_l754_754380

noncomputable def is_line_in_spherical_coordinates (c d : ℝ) : Prop :=
  ∀ (ρ : ℝ), (ρ > 0) → ∃ (x y z : ℝ), 
    x = ρ * sin d * cos c ∧
    y = ρ * sin d * sin c ∧
    z = ρ * cos d

theorem shape_is_line (c d : ℝ) : is_line_in_spherical_coordinates c d :=
begin
  sorry,
end

end shape_is_line_l754_754380


namespace no_self_intersection_probability_l754_754579

noncomputable def probability_no_self_intersection: ℝ :=
  let α_range := (0 : ℝ, π)
  let β_range := (0 : ℝ, 2 * π)
  let total_area := (π - 0) * (2 * π - 0)
  let self_intersection_area := (π * π) / 6
  let non_self_intersection_area := total_area - self_intersection_area in
  non_self_intersection_area / total_area

theorem no_self_intersection_probability : 
  probability_no_self_intersection = 11 / 12 := 
sorry

end no_self_intersection_probability_l754_754579


namespace sum_reciprocals_l754_754863

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom h₁ : a 1 = 2
axiom h₂ : ∀ n : ℕ, 0 < n → a (n + 1) - a n = 2
axiom h₃ : ∀ n : ℕ, S n = n * (n + 1)

theorem sum_reciprocals (n : ℕ) : (finset.sum (finset.range n) (λ k, 1 / S (k + 1))) = n / (n + 1) :=
by
  sorry

end sum_reciprocals_l754_754863


namespace polynomial_irreducible_l754_754135

-- Define the polynomial A
def A (X Y Z : ℝ) : ℝ := 2 * X^2 * Y + 2 * X * Y * Z^3 - X * Z + Y^3 * Z

-- State the irreducibility of the polynomial in Lean
theorem polynomial_irreducible : irreducible (A (X Y Z)) := 
sorry

end polynomial_irreducible_l754_754135


namespace arithmetic_series_sum_l754_754267

theorem arithmetic_series_sum :
  ∑ k in Finset.range 11, (2 * k + 1) = 121 :=
by
  sorry

end arithmetic_series_sum_l754_754267


namespace cos_C_given_sin_A_and_cos_B_l754_754516

theorem cos_C_given_sin_A_and_cos_B (A B C : ℝ) (h_triangle : A + B + C = real.pi)
  (h_sinA : real.sin A = 4 / 5) (h_cosB : real.cos B = 12 / 13) :
  real.cos C = -16 / 65 :=
sorry

end cos_C_given_sin_A_and_cos_B_l754_754516


namespace positive_divisors_of_x_l754_754747

theorem positive_divisors_of_x (x : ℕ) (h : ∀ d : ℕ, d ∣ x^3 → d = 1 ∨ d = x^3 ∨ d ∣ x^2) : (∀ d : ℕ, d ∣ x → d = 1 ∨ d = x ∨ d ∣ p) :=
by
  sorry

end positive_divisors_of_x_l754_754747


namespace min_value_of_f_l754_754166

def f (x : Real) : Real :=
  Real.cos x ^ 2 - 2 * Real.cos (x / 2) ^ 2

theorem min_value_of_f :
  ∃ x : Real, -1 ≤ Real.cos x ∧ Real.cos x ≤ 1 ∧ f x = -5 / 4 :=
by
  sorry

end min_value_of_f_l754_754166


namespace number_of_solutions_cos_eq_l754_754958

theorem number_of_solutions_cos_eq (p : ℝ) (h : 0 ≤ p) :
  {x : ℝ | 0 ≤ x ∧ x ≤ p ∧ cos (7 * x) = cos (5 * x)}.finite.toFinset.card = 7 := 
sorry

end number_of_solutions_cos_eq_l754_754958


namespace select_sets_l754_754545

open Finset

variable {X : Type} [DecidableEq X] [Fintype X]

theorem select_sets (k : ℕ) (A : Fin k → Finset X)
  (hA1 : ∀ i, A i.card ≤ 3)
  (hA2 : ∀ x ∈ (univ : Finset X), (filter ( λ i, x ∈ A i) univ).card ≥ 4) :
  ∃ S : Finset (Fin k), S.card = ⌊3 * k / 7⌋ ∧ (⋃ i ∈ S, A i) = univ :=
sorry

end select_sets_l754_754545


namespace length_of_DN_l754_754482

theorem length_of_DN (A B C D N : Point)
  (hD_midpoint : midpoint D B C)
  (hAN_bisector : angle_bisector AN BAC)
  (hBN_perp_AN : perp BN AN)
  (hAB_length : length A B = 15)
  (hAC_length : length A C = 17) : length D N = 1 := 
sorry

end length_of_DN_l754_754482


namespace f_eq_f_and_neq_implies_ab_eq_one_l754_754888

variable {a b : ℝ}

def f (x : ℝ) := abs (Real.log x / Real.log 3)

theorem f_eq_f_and_neq_implies_ab_eq_one 
  (h1 : a ≠ b) 
  (h2 : f a = f b) 
  : a * b = 1 :=
sorry

end f_eq_f_and_neq_implies_ab_eq_one_l754_754888


namespace intersection_of_M_and_N_l754_754439

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0} := 
by sorry

end intersection_of_M_and_N_l754_754439


namespace sum_B_1_to_255_l754_754703

def binary_representation (n : Nat) : String :=
  natToBinaryString n

def count_1_runs (s : String) : Nat :=
  -- Function to count 1-runs in a given binary string
  sorry

def B (n : Nat) : Nat :=
  count_1_runs (binary_representation n)

theorem sum_B_1_to_255 : (List.range 255).sum (λ n, B (n + 1)) = 255 :=
  sorry

end sum_B_1_to_255_l754_754703


namespace ratio_of_volumes_l754_754114

variables (A B : ℚ)

theorem ratio_of_volumes 
  (h1 : (3/8) * A = (5/8) * B) :
  A / B = 5 / 3 :=
sorry

end ratio_of_volumes_l754_754114


namespace total_cans_given_away_l754_754270

noncomputable def total_cans_initial : ℕ := 2000
noncomputable def cans_taken_first_day : ℕ := 500
noncomputable def restocked_first_day : ℕ := 1500
noncomputable def people_second_day : ℕ := 1000
noncomputable def cans_per_person_second_day : ℕ := 2
noncomputable def restocked_second_day : ℕ := 3000

theorem total_cans_given_away :
  (cans_taken_first_day + (people_second_day * cans_per_person_second_day) = 2500) :=
by
  sorry

end total_cans_given_away_l754_754270


namespace part1_min_value_of_f_part2_extreme_points_of_g_l754_754429

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + (a - 1) * (x - 1)

theorem part1_min_value_of_f (a : ℝ) (h : a = -1) : f (Real.exp 1) a = 2 - Real.exp 1 := by
  have h_perpendicular : a - 1 = -2 := by simp [h]
  rw [h] at h_perpendicular
  sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (f x a + a^2 - 2 * a)

noncomputable def h (x : ℝ) (a : ℝ) : ℝ :=
  x * Real.log x + Real.log x + (a - 1) * x + a^2 - 2 * a + 1

theorem part2_extreme_points_of_g (a : ℝ) (h₁ : -1 ≤ a) : 
  (a ≥ 2 ∨ (-1 ≤ a ∧ a ≤ -1/Real.exp 1) → ∀ x, x > 1/Real.exp 1 → h x a > 0) ∧
  (-1/Real.exp 1 < a ∧ a < 2 → ∃ x, x > 1/Real.exp 1 ∧ h x a = 0) := by
  sorry

end part1_min_value_of_f_part2_extreme_points_of_g_l754_754429


namespace range_of_m_l754_754613

def f (x : ℝ) : ℝ := x^2 - 4 * x - 6

theorem range_of_m (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ m → -10 ≤ f x ∧ f x ≤ -6) →
  2 ≤ m ∧ m ≤ 4 := 
sorry

end range_of_m_l754_754613


namespace find_lambda_l754_754444

-- Define the vectors and the perpendicular condition
def vector_a (λ : ℝ) : ℝ × ℝ := (λ + 1, 2)
def vector_b : ℝ × ℝ := (-1, 1)

-- Define the dot product function for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the main theorem
theorem find_lambda (λ : ℝ) (h : dot_product (vector_a λ) vector_b = 0) : λ = 1 :=
by sorry

end find_lambda_l754_754444


namespace number_of_divisors_l754_754385

theorem number_of_divisors (n : ℕ) :
  n > 0 → ∃! n : ℕ, (∀ m : ℕ, (∑ i in Finset.range (n + 1), i + 1) ∣ 8 * n) ∧ n ∈ {1, 3, 7, 15} :=
by
  sorry

end number_of_divisors_l754_754385


namespace positive_integers_dividing_8n_sum_l754_754381

theorem positive_integers_dividing_8n_sum :
  {n : ℕ // n > 0 ∧ (∃ k : ℕ, 8 * n = k * (n * (n + 1) / 2))}.card = 4 := sorry

end positive_integers_dividing_8n_sum_l754_754381


namespace max_min_integer_solutions_l754_754871

theorem max_min_integer_solutions (k a b : ℕ) (hpos : k > 0 ∧ a > 0 ∧ b > 0) (hgcd : Nat.gcd a b = 1) :
  exists n_max n_min : ℕ,
    (ax + by = n_max ∧ (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (ax + by = n_max)) ∧ ∀ (x y : ℕ), x > 0 ∧ y > 0 → ¬(ax + by = n_max + 1)) ∧
    (ax + by = n_min ∧ (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (ax + by = n_min)) ∧ ∀ (x y : ℕ), x > 0 ∧ y > 0 → ¬(ax + by = n_min - 1)) ∧
    n_max = (k + 1) * a * b ∧
    n_min = (k - 1) * a * b + a + b := sorry

end max_min_integer_solutions_l754_754871


namespace ratio_is_1_to_2_l754_754173

def Ps : ℚ := 603.75
def Rs : ℚ := 14
def Ts : ℚ := 6

def Pc : ℚ := 7000
def Rc : ℚ := 7
def Tc : ℚ := 2

def calculate_si (P : ℚ) (R : ℚ) (T : ℚ) : ℚ :=
  (P * R * T) / 100

def calculate_ci (P : ℚ) (R : ℚ) (T : ℚ) : ℚ :=
  P * ((1 + R / 100) ^ T - 1)

def ratio_si_ci (si ci : ℚ) : ℚ :=
  si / ci

theorem ratio_is_1_to_2 (Ps Rs Ts Pc Rc Tc : ℚ) :
  ratio_si_ci (calculate_si Ps Rs Ts) (calculate_ci Pc Rc Tc) = 1 / 2 :=
by {
  sorry
}

end ratio_is_1_to_2_l754_754173


namespace count_three_digit_numbers_no_5_no_9_l754_754451

/-- 
The total number of three-digit whole numbers where no digit is 5 or 9 equals 448.
-/
theorem count_three_digit_numbers_no_5_no_9 : 
  ∃ n : ℕ, n = 448 ∧ 
  (∃ f : ℕ → ℕ, (∀ x ∈ {1, 2, 3, 4, 6, 7, 8}, f 1 = x) ∧ 
    (∀ k ∈ {0, 1, 2, 3, 4, 6, 7, 8}, f 2 = k) ∧ 
    (∀ l ∈ {0, 1, 2, 3, 4, 6, 7, 8}, f 3 = l)) := 
begin
  sorry
end

end count_three_digit_numbers_no_5_no_9_l754_754451


namespace grape_juice_percentage_l754_754115

theorem grape_juice_percentage :
  ∀ (total_apples total_grapes : ℕ) (juice_from_apples juice_from_grapes : ℕ) (apples_used grapes_used: ℕ),
    total_apples = 12 →
    total_grapes = 12 →
    juice_from_apples = 10 →
    juice_from_grapes = 9 →
    apples_used = 4 →
    grapes_used = 3 →
    (let apple_juice_per_apple := juice_from_apples.to_rat / apples_used.to_rat in
     let grape_juice_per_grape := juice_from_grapes.to_rat / grapes_used.to_rat in
     let total_blended_apples := 6 in
     let total_blended_grapes := 6 in
     let total_apple_juice := apple_juice_per_apple * total_blended_apples in
     let total_grape_juice := grape_juice_per_grape * total_blended_grapes in
     (total_grape_juice / (total_apple_juice + total_grape_juice)) = (54.55 : ℚ) / 100) :=
begin
  intros total_apples total_grapes juice_from_apples juice_from_grapes apples_used grapes_used,
  intros h1 h2 h3 h4 h5 h6,
  let apple_juice_per_apple := juice_from_apples.to_rat / apples_used.to_rat,
  let grape_juice_per_grape := juice_from_grapes.to_rat / grapes_used.to_rat,
  let total_blended_apples := 6,
  let total_blended_grapes := 6,
  let total_apple_juice := apple_juice_per_apple * total_blended_apples,
  let total_grape_juice := grape_juice_per_grape * total_blended_grapes,
  have h : (total_grape_juice / (total_apple_juice + total_grape_juice)) = (18 / 33 : ℚ),
  { norm_num [apple_juice_per_apple, grape_juice_per_grape, total_blended_apples, total_blended_grapes, total_apple_juice, total_grape_juice], },
  rw [h],
  norm_num,
end

end grape_juice_percentage_l754_754115


namespace systematic_sampling_fifth_group_l754_754659

theorem systematic_sampling_fifth_group :
  ∀ (N : ℕ) (n : ℕ) (k : ℕ) (first_draw : ℕ) (interval : ℕ)
  (groups : ℕ) (group_index : ℕ),
  N = 160 →
  n = 20 →
  k = 5 →
  first_draw = 3 →
  interval = 8 →
  groups = 20 →
  group_index = 5 →
  first_draw + (group_index - 1) * interval = 35 :=
by
  intros N n k first_draw interval groups group_index
  intro hN
  intro hn
  intro hk
  intro hfirst_draw
  intro hinterval
  intro hgroups
  intro hgroup_index
  rw hfirst_draw
  rw hgroup_index
  rw hinterval
  rw hk
  norm_num
  sorry

end systematic_sampling_fifth_group_l754_754659


namespace arithmetic_sequence_problem_l754_754505

variable (a : ℕ → ℝ) (d : ℝ) (m : ℕ)

noncomputable def a_seq := ∀ n, a n = a 1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : a 1 = 0)
  (h2 : d ≠ 0)
  (h3 : a m = a 1 + a 2 + a 3 + a 4 + a 5) :
  m = 11 :=
sorry

end arithmetic_sequence_problem_l754_754505


namespace problem1_problem2_l754_754896

-- Definition of f(x)
def f (x : ℝ) : ℝ := abs (x - 1)

-- Definition of g(x)
def g (x t : ℝ) : ℝ := t * abs x - 2

-- Problem 1: Proof that f(x) > 2x + 1 implies x < 0
theorem problem1 (x : ℝ) : f x > 2 * x + 1 → x < 0 := by
  sorry

-- Problem 2: Proof that if f(x) ≥ g(x) for all x, then t ≤ 1
theorem problem2 (t : ℝ) : (∀ x : ℝ, f x ≥ g x t) → t ≤ 1 := by
  sorry

end problem1_problem2_l754_754896


namespace cos_C_in_triangle_l754_754525

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end cos_C_in_triangle_l754_754525


namespace shortest_distance_to_circle_l754_754675

def center : ℝ × ℝ := (8, 7)
def radius : ℝ := 5
def point : ℝ × ℝ := (1, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

theorem shortest_distance_to_circle :
  distance point center - radius = Real.sqrt 130 - 5 :=
by
  sorry

end shortest_distance_to_circle_l754_754675


namespace count_x_values_l754_754316

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l754_754316


namespace blue_lipstick_students_l754_754575

def total_students : ℕ := 200
def students_with_lipstick : ℕ := total_students / 2
def students_with_red_lipstick : ℕ := students_with_lipstick / 4
def students_with_blue_lipstick : ℕ := students_with_red_lipstick / 5

theorem blue_lipstick_students : students_with_blue_lipstick = 5 :=
by
  sorry

end blue_lipstick_students_l754_754575


namespace geometric_inequality_l754_754582

theorem geometric_inequality (A B C O : Point) 
  (hAOB : angle A O B = 120) 
  (hBOC : angle B O C = 120) 
  (hCOA : angle C O A = 120) 
  (hO_in_triangle : inside_triangle O A B C) :
  (AO^2 / BC + BO^2 / CA + CO^2 / AB) ≥ (AO + BO + CO) / sqrt 3 := 
begin
  sorry
end

end geometric_inequality_l754_754582


namespace part_one_part_two_l754_754811

-- Given that tan α = 2, prove that the following expressions are correct:

theorem part_one (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (Real.pi - α) + Real.cos (α - Real.pi / 2) - Real.cos (3 * Real.pi + α)) / 
  (Real.cos (Real.pi / 2 + α) - Real.sin (2 * Real.pi + α) + 2 * Real.sin (α - Real.pi / 2)) = 
  -5 / 6 := 
by
  -- Proof skipped
  sorry

theorem part_two (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) + Real.sin α * Real.cos α = -1 / 5 := 
by
  -- Proof skipped
  sorry

end part_one_part_two_l754_754811


namespace cos_C_given_sin_A_and_cos_B_l754_754519

theorem cos_C_given_sin_A_and_cos_B (A B C : ℝ) (h_triangle : A + B + C = real.pi)
  (h_sinA : real.sin A = 4 / 5) (h_cosB : real.cos B = 12 / 13) :
  real.cos C = -16 / 65 :=
sorry

end cos_C_given_sin_A_and_cos_B_l754_754519


namespace intersection_A_B_l754_754906

variable (x : ℝ)

noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def B := {x : ℝ | 2^(x + 1) > 1}

theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end intersection_A_B_l754_754906


namespace fixed_point_exists_l754_754676

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y : ℝ, (x = 2 ∧ y = -2 ∧ (ax - 5 = y)) :=
by
  sorry

end fixed_point_exists_l754_754676


namespace hyperbola_eccentricity_l754_754897

theorem hyperbola_eccentricity {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ (y = 3 * x)) → (sqrt (1 + (b / a)^2) > sqrt 10) :=
by
  sorry

end hyperbola_eccentricity_l754_754897


namespace recurring_decimals_sum_l754_754363

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end recurring_decimals_sum_l754_754363


namespace dilution_problem_l754_754130

/-- Samantha needs to add 7.2 ounces of water to achieve a 25% alcohol concentration
given that she starts with 12 ounces of solution containing 40% alcohol. -/
theorem dilution_problem (x : ℝ) : (12 + x) * 0.25 = 4.8 ↔ x = 7.2 :=
by sorry

end dilution_problem_l754_754130


namespace period_of_y_l754_754266

-- Define the function y
noncomputable def y (x : ℝ) : ℝ :=
  Real.sin (10 * x) + Real.cos (5 * x)

-- Define the periods of the component functions
def T1 : ℝ := π / 5
def T2 : ℝ := 2 * π / 5

-- Assertion about the function y
theorem period_of_y : ∃ T : ℝ, (∀ x : ℝ, y (x + T) = y x) ∧ T = 2 * π / 5 := 
by
  sorry

end period_of_y_l754_754266


namespace range_of_x_geq_f_geq_g_minimum_value_of_y_l754_754011

noncomputable def f (x : ℝ) : ℝ := log (x + 1) / log 2

noncomputable def g (x : ℝ) : ℝ := log (3 * x + 1) / log 2

theorem range_of_x_geq_f_geq_g :
  ∀ x : ℝ, (g x ≥ f x) ↔ (x ≥ 0) :=
by
  sorry

theorem minimum_value_of_y :
  ∀ x : ℝ, (x ≥ 0) → (g x - f x) = 0 :=
by
  sorry

end range_of_x_geq_f_geq_g_minimum_value_of_y_l754_754011


namespace age_sum_of_scrolls_l754_754233

-- Define the age of the first scroll
def S1 : ℕ := 4080

-- Define the age of the subsequent scrolls
def S2 : ℕ := S1 + (1 / 2:ℝ) * S1
def S3 : ℕ := S2 + (1 / 2:ℝ) * S2
def S4 : ℕ := S3 + (1 / 2:ℝ) * S3
def S5 : ℕ := S4 + (1 / 2:ℝ) * S4

-- The sum of the ages of all five scrolls
def Sum : ℕ := S1 + S2 + S3 + S4 + S5

-- The theorem to prove
theorem age_sum_of_scrolls : Sum = 48805 := 
by
  sorry

end age_sum_of_scrolls_l754_754233


namespace count_x_values_l754_754314

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l754_754314


namespace string_cut_probability_l754_754240

theorem string_cut_probability :
  let C := λ (x : ℝ), x in
  let total_range := 2 - 0.2 in
  let valid_range := 0.5 - 0.2 in
  let probability := valid_range / total_range in
  (∀ x : ℝ, 0.2 ≤ x ∧ x ≤ 0.5 → 2 - x ≥ 3 * x) →
  probability = 3 / 8 :=
by
  intro C total_range valid_range probability h
  sorry

end string_cut_probability_l754_754240


namespace inequality_proof_l754_754398

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  b + (1 / a) > a + (1 / b) := 
by sorry

end inequality_proof_l754_754398


namespace anna_candies_l754_754256

theorem anna_candies : ∃ x : ℕ, x + 86 = 91 := 
by
  use 5
  calc
    5 + 86 = 91 : by norm_num
-- sorry

end anna_candies_l754_754256


namespace constant_term_of_binomial_expansion_l754_754477

theorem constant_term_of_binomial_expansion : 
  (∀ (n : ℕ), (2 * 1 + 1 / 1)^n = 256 → n = 8) → 
  let n := 8 in
  ∃ (r : ℕ), 8 - 2 * r = 0 ∧ 
  (2:ℚ)^(8-r) * (Nat.choose 8 r) = 1120 :=
by
  intro h
  have hn := h 8
  rw [pow_succ, pow_succ, pow_succ, pow_succ, pow_succ, pow_succ, pow_succ] at hn
  use 4
  split
  rfl
  norm_num
  sorry

end constant_term_of_binomial_expansion_l754_754477


namespace sum_T_1_to_10_eq_130750_l754_754801

def arithmetic_sum (p : ℕ) : ℕ :=
  2500 * p - 1225

def T (p : ℕ) : ℕ :=
  arithmetic_sum p + 100 * p

theorem sum_T_1_to_10_eq_130750 : ∑ p in finset.range 10, T (p + 1) = 130750 := 
by
  sorry

end sum_T_1_to_10_eq_130750_l754_754801


namespace verify_greening_equation_l754_754225

-- Definitions based on conditions
def task_area := 800000  -- 800,000 square meters
def efficiency_increase := 0.35  -- 35%
def time_saved := 40  -- 40 days
def actual_daily_greened_area (x : ℝ) := x  -- (x thousand square meters)

-- Original planned daily greened area in thousand square meters, using actual efficiency
def planned_daily_greened_area (x : ℝ) := x / (1 + efficiency_increase)

-- Proof problem: Verify the equation represents the given scenario
theorem verify_greening_equation (x : ℝ) (Hx : x > 0) :
  (80 * (1 + efficiency_increase)) / x - 80 / x = time_saved :=
by
  sorry

end verify_greening_equation_l754_754225


namespace triangle_inequality_proof_l754_754418

variable (a b c : ℝ)

-- Condition that a, b, c are side lengths of a triangle
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- Theorem stating the required inequality and the condition for equality
theorem triangle_inequality_proof :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c ∧ c = a) :=
sorry

end triangle_inequality_proof_l754_754418


namespace cos_C_of_triangle_l754_754515

theorem cos_C_of_triangle (A B C : ℝ) (hA : sin A = 4 / 5) (hB : cos B = 12 / 13) :
  cos C = -16 / 65 :=
sorry

end cos_C_of_triangle_l754_754515


namespace integral_evaluation_l754_754435

def f (x : ℝ) (f''_1 : ℝ) : ℝ := f''_1 * x^2 + x + 2

theorem integral_evaluation (f''_1 : ℝ) (h : f''_1 = -1) :
    ∫ x in 0..1, f x f''_1 = 13/6 :=
by
  have : f''_1 * 1^2 + 1 + 2 = -1 * 1^2 + 1 + 2, from by rw [h]
  have : f 1 f''_1 = -1^2 + 1 + 2, from (by rw [f, h]; exact rfl)
  have : ∫ x in 0..1, f x f''_1 = ∫ x in 0..1, -x^2 + x + 2, from by rw [f, h]
  calc
    ∫ x in 0..1, -x^2 + x + 2 = (-(1/3)*1^3 + (1/2)*1^2 + 2*1) - (-(1/3)*0^3 + (1/2)*0^2 + 2*0) : by 
      { apply integral_monoid, sorry }
    ... = 13/6 : by norm_num

end integral_evaluation_l754_754435


namespace find_fx_sum_of_solutions_l754_754339

noncomputable def f : ℝ → ℝ 
| x => if x = 2 then 1 else Real.log (abs (x - 2))

theorem find_fx_sum_of_solutions
  (b c : ℝ)
  (h : ∃ x1 x2 x3 x4 x5 : ℝ, 
        f(x1) * f(x1) + b * f(x1) + c = 0 ∧ 
        f(x2) * f(x2) + b * f(x2) + c = 0 ∧ 
        f(x3) * f(x3) + b * f(x3) + c = 0 ∧ 
        f(x4) * f(x4) + b * f(x4) + c = 0 ∧
        f(x5) * f(x5) + b * f(x5) + c = 0 ∧
        x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ 
        x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ 
        x3 ≠ x4 ∧ x3 ≠ x5 ∧ 
        x4 ≠ x5) :
  f (x1 + x2 + x3 + x4 + x5) = Real.log 8 := 
sorry

end find_fx_sum_of_solutions_l754_754339


namespace sum_of_operations_l754_754624

def operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem sum_of_operations : operation 12 5 + operation 8 3 = 174 := by
  sorry

end sum_of_operations_l754_754624


namespace sum_squares_pascals_triangle_l754_754791

theorem sum_squares_pascals_triangle (n : ℕ) :
  (∑ k in Finset.range (n + 1), (Nat.choose n k)^2) = Nat.choose (2 * n) n := by
  sorry

end sum_squares_pascals_triangle_l754_754791


namespace average_first_five_results_l754_754608

theorem average_first_five_results (results : Fin 11 → ℝ)
  (h1 : (∑ i, results i) / 11 = 42)
  (h2 : (∑ i in Finset.range 7, results (i + 4)) / 7 = 52)
  (h3 : results 4 = 147) :
  (∑ i in Finset.range 5, results i) / 5 = 49 := 
sorry

end average_first_five_results_l754_754608


namespace compare_negatives_l754_754284

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l754_754284


namespace smallest_odd_n_such_that_series_is_integer_l754_754985

noncomputable def a : ℝ := Real.pi / 4032

theorem smallest_odd_n_such_that_series_is_integer :
  ∃ (n : ℕ), n % 2 = 1 ∧ ∀ k ≤ n, 4032 ∣ (k * (k + 1)) ∧ 2 * (Finset.sum (Finset.range (n + 1)) (λ k, Real.cos (k^2 * a) * Real.sin (k * a))) ∈ ℤ :=
sorry

end smallest_odd_n_such_that_series_is_integer_l754_754985


namespace side_length_ratio_sum_correct_l754_754170

-- Define the given condition of the problem.
def area_ratio : ℚ := 50 / 98

-- Define the statement that proves the correct sum of a, b, and c.
theorem side_length_ratio_sum_correct :
  ∃ (a b c : ℕ), (a * c^-1 = 5 * 7^-1) ∧ b = 0 ∧ (a + b + c = 12) :=
by
  use 5, 0, 7
  simp
  sorry

end side_length_ratio_sum_correct_l754_754170


namespace final_result_always_4_l754_754183

-- The function that performs the operations described in the problem
def transform (x : Nat) : Nat :=
  let step1 := 2 * x
  let step2 := step1 + 3
  let step3 := step2 * 5
  let step4 := step3 + 7
  let last_digit := step4 % 10
  let step6 := last_digit + 18
  step6 / 5

-- The theorem statement claiming that for any single-digit number x, the result of transform x is always 4
theorem final_result_always_4 (x : Nat) (h : x < 10) : transform x = 4 := by
  sorry

end final_result_always_4_l754_754183


namespace negative_half_less_than_negative_third_l754_754277

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l754_754277


namespace no_points_greater_than_one_over_sqrt_six_l754_754409

-- Variables for the tetrahedron
variables {A B C D : Point} -- Assume Point is a defined type for points in space
variable (edge_length : ℝ) -- Assume ℝ as the type for real numbers

-- Condition that A, B, C, D form a regular tetrahedron with edge length 1
constants (is_regular_tetrahedron : RegularTetrahedron A B C D edge_length) 

-- Diameter spheres with tetrahedron edges as diameters
-- Let S be the intersection of these six spheres
noncomputable def S : Set Point :=
  ⋂ (e in {$ (A, B), (A, C), (A, D), (B, C), (B, D), (C, D)}), 
    Sphere (midpoint e) (edge_length / 2)

-- The actual mathematical statement we wish to prove
theorem no_points_greater_than_one_over_sqrt_six : 
  ∀ (P Q : Point), P ∈ S → Q ∈ S → dist P Q ≤ 1 / Real.sqrt 6 := 
  sorry

end no_points_greater_than_one_over_sqrt_six_l754_754409


namespace triangle_ratio_l754_754188

theorem triangle_ratio (AC BC AD : ℕ) (hAC : AC = 3) (hBC : BC = 4) (hAD : AD = 12)
  (h_right_ABC : AC^2 + BC^2 = (Nat.sqrt (AC^2 + BC^2))^2)
  (h_right_ABD : AC^2 + AD^2 = (Nat.sqrt (AC^2 + AD^2))^2)
  (rel_prime : Nat.rel_prime 63 65) : 
  63 + 65 = 128 :=
by
  sorry

end triangle_ratio_l754_754188


namespace union_A_B_interval_l754_754874

def setA (x : ℝ) : Prop := x ≥ -1
def setB (y : ℝ) : Prop := y ≥ 1

theorem union_A_B_interval :
  {x | setA x} ∪ {y | setB y} = {z : ℝ | z ≥ -1} :=
by
  sorry

end union_A_B_interval_l754_754874


namespace range_z_correct_l754_754883

-- Assuming inequalities define a region R for points (x, y)
def region (x y : ℝ) : Prop := sorry -- Specify the actual inequalities here

def range_z := {z : ℝ | ∃ x y, region x y ∧ z = x - y}

theorem range_z_correct : range_z = set.Icc (-1 : ℝ) 2 :=
by
  sorry

end range_z_correct_l754_754883


namespace calc_z_conj_diff_l754_754828

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754828


namespace budget_ratio_l754_754634

theorem budget_ratio : 
  ∀ (total_budget education_budget public_spaces_budget policing_budget : ℕ), 
  total_budget = 32 ∧ 
  education_budget = 12 ∧ 
  public_spaces_budget = 4 ∧ 
  policing_budget = (total_budget - (education_budget + public_spaces_budget)) →
  policing_budget : total_budget = 16 : 32 → 
  policing_budget : total_budget = 1 : 2 := 
by
  sorry

end budget_ratio_l754_754634


namespace find_radius_l754_754937

open Real

variables (O : Point)
variables (AB : LineSegment)
variables (OC : LineSegment)
variables (OD : LineSegment)
variables (D : Point)

def OC_eq_2 : Real := 2
def OD_eq_sqrt3 : Real := sqrt 3
def angle_CDA_eq_120 : Real := 120

theorem find_radius 
  (TangentCircle : Circle) 
  (tangent_to_AD : isTangent TangentCircle AD)
  (tangent_to_DC : isTangent TangentCircle DC)
  (tangent_to_arc_AC : isTangent TangentCircle arcAC)
  (OC_eq_2 : length OC = 2)
  (OD_eq_sqrt3 : length OD = sqrt 3)
  (angle_CDA_eq_120 : angle D A C = 120) :
  TangentCircle.radius = 2 * sqrt 21 - 9 :=
sorry

end find_radius_l754_754937


namespace sequence_sum_l754_754091

theorem sequence_sum (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) :
  a_3 = 7 → (∀ n, n ≥ 2 → a_n = 2 * a_(n-1) + a_2 - 2) →
  S_n = (λ n, ∑ i in finset.range (n+1), a_i) →
  ∀ n, S_n n = 2^(n+1) - 2 - n :=
by
  sorry

end sequence_sum_l754_754091


namespace certain_number_exists_l754_754419

theorem certain_number_exists (a b : ℝ) (C : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) (h3 : a * (a - 4) = C) (h4 : b * (b - 4) = C) : 
  C = -3 := 
sorry

end certain_number_exists_l754_754419


namespace sequence_not_square_of_rational_l754_754542

def seq (a0 : ℚ) (n : ℕ) : ℚ :=
nat.rec_on n a0 (λ n a_n, a_n + 2 / a_n)

theorem sequence_not_square_of_rational :
  ∀ n, ∀ x : ℚ, seq 2016 n ≠ x^2 :=
by
  sorry

end sequence_not_square_of_rational_l754_754542


namespace recurring_decimals_sum_l754_754362

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end recurring_decimals_sum_l754_754362


namespace julian_frames_l754_754975

theorem julian_frames (frames_per_page pages : ℕ) (h1 : frames_per_page = 11) (h2 : pages = 13) :
  frames_per_page * pages = 143 :=
by 
  rw [h1, h2]
  norm_num

end julian_frames_l754_754975


namespace trisha_take_home_pay_l754_754649

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end trisha_take_home_pay_l754_754649


namespace range_of_a_if_f_decreasing_l754_754036

open Real

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem range_of_a_if_f_decreasing :
  ∀ {a : ℝ}, (∀ x y ∈ Icc (1:ℝ) 2, x < y → f a y ≤ f a x) → 2 ≤ a :=
by
  intros
  sorry

end range_of_a_if_f_decreasing_l754_754036


namespace winning_candidate_percentage_l754_754495

def initial_votes : List Nat :=
  [4571, 9892, 17653, 3217, 15135, 11338, 8629]

def redistribution_round_1 : List Nat :=
  [1571, 802, 251, 1072, 305, 216]  -- Votes redistributed from Candidate 4

def redistribution_round_2 : List Nat :=
  [3162, 1029, 1378, 412, 161]  -- Votes redistributed from Candidate 1

def redistribution_round_3 : List Nat :=
  [2001, 2696, 4372, 6048]  -- Votes redistributed from Candidate 7

def redistribution_round_4 : List Nat :=
  [7500, 6248, 1107]  -- Votes redistributed from Candidate 2

-- Function to compute the new vote counts after redistribution
def redistribute_votes (votes : Vector Nat n) (redistributed : List Nat) (eliminated_idx : Nat) : List Nat :=
  let reduced_votes := votes.toList.take eliminated_idx ++ votes.toList.drop (eliminated_idx + 1)
  List.zipWith (· + ·) reduced_votes redistributed

-- Initial state of votes (before any elimination)
def initial_state : Vector Nat 7 := ⟨initial_votes, by simp [initial_votes.length]⟩

-- State after each round of elimination
def state_round_1 : List Nat := redistribute_votes initial_state redistribution_round_1 3
def state_round_2 : List Nat := redistribute_votes ⟨state_round_1, by simp [state_round_1.length]⟩ redistribution_round_2 0
def state_round_3 : List Nat := redistribute_votes ⟨state_round_2, by simp [state_round_2.length]⟩ redistribution_round_3 4
def state_round_4 : List Nat := redistribute_votes ⟨state_round_3, by simp [state_round_3.length]⟩ redistribution_round_4 1

-- Winning candidate and their vote count
def winning_votes : Nat := state_round_4.head ∘ List.maximum
def total_votes : Nat := state_round_4.foldr (· + ·) 0

-- Percentage calculation
def winning_percentage (winning_votes total_votes : Nat) : Float :=
  (winning_votes.toFloat / total_votes.toFloat) * 100

-- The main theorem stating the winning candidate's percentage of the total votes
theorem winning_candidate_percentage :
  winning_percentage winning_votes total_votes ≈ 38.06 :=
sorry

end winning_candidate_percentage_l754_754495


namespace equation_of_line_l754_754422

theorem equation_of_line (l : ℝ → ℝ) :
  (∀ (P : ℝ × ℝ), P = (4, 2) → 
    ∃ (a b : ℝ), ((P = ( (4 - a), (2 - b)) ∨ P = ( (4 + a), (2 + b))) ∧ 
    ((4 - a)^2 / 36 + (2 - b)^2 / 9 = 1) ∧ ((4 + a)^2 / 36 + (2 + b)^2 / 9 = 1)) ∧
    (P.2 = l P.1)) →
  (∀ (x y : ℝ), y = l x ↔ 2 * x + 3 * y - 16 = 0) :=
by
  intros h P hp
  sorry -- Placeholder for the proof

end equation_of_line_l754_754422


namespace complex_conjugate_difference_l754_754825

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754825


namespace total_toys_correct_l754_754534

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l754_754534


namespace Q_polynomial_l754_754686

def cos3x_using_cos2x (cos_α : ℝ) := (2 * cos_α^2 - 1) * cos_α - 2 * (1 - cos_α^2) * cos_α

def Q (x : ℝ) := 4 * x^3 - 3 * x

theorem Q_polynomial (α : ℝ) : Q (Real.cos α) = Real.cos (3 * α) := by
  rw [Real.cos_three_mul]
  sorry

end Q_polynomial_l754_754686


namespace maximum_angle_C_l754_754074

theorem maximum_angle_C (A B C : Point) (AB AC BA BC CA CB : Vector)
(h_condition : (AB • AC) + (BA • BC) = 2 * (CA • CB)) :
  ∃ (C_max : ℝ), C_max = π / 3 :=
begin
  -- Proof goes here
  sorry
end

end maximum_angle_C_l754_754074


namespace loss_percentage_l754_754733

-- Definitions related to the problem
def CPA : Type := ℝ
def SPAB (CPA: ℝ) : ℝ := 1.30 * CPA
def SPBC (CPA: ℝ) : ℝ := 1.040000000000000036 * CPA

-- Theorem to prove the loss percentage when B sold the bicycle to C 
theorem loss_percentage (CPA : ℝ) (L : ℝ) (h1 : SPAB CPA * (1 - L) = SPBC CPA) : 
  L = 0.20 :=
by
  sorry

end loss_percentage_l754_754733


namespace circle_properties_l754_754983

theorem circle_properties :
  ∃ p q s : ℝ, 
  (∀ x y : ℝ, x^2 + 16 * y + 89 = -y^2 - 12 * x ↔ (x + p)^2 + (y + q)^2 = s^2) ∧ 
  p + q + s = -14 + Real.sqrt 11 :=
by
  use -6, -8, Real.sqrt 11
  sorry

end circle_properties_l754_754983


namespace find_a_l754_754001

def f (x : ℝ) (a : ℝ) : ℝ := sin x ^ 2 + cos x + (5 / 8) * a - 3 / 2

theorem find_a (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ (real.pi / 2) → f x a ≥ 2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ (real.pi / 2) ∧ f x a = 2) ↔ a = 3.6 := 
sorry

end find_a_l754_754001


namespace product_of_B_l754_754563

def A : set ℝ := {2, 0, 1, 6}
def B : set ℝ := {k | k^2 - 2 ∈ A ∧ k - 2 ∉ A}

theorem product_of_B : ∏ (x : ℝ) in B, x = 96 :=  sorry

end product_of_B_l754_754563


namespace evaporation_fraction_l754_754707

theorem evaporation_fraction (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1)
  (h : (1 - x) * (3 / 4) = 1 / 6) : x = 7 / 9 :=
by
  sorry

end evaporation_fraction_l754_754707


namespace total_toys_l754_754540

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end total_toys_l754_754540


namespace problem_statement_l754_754560

variables {A B C D E F I J N H : Type*}
-- Define triangle ABC and the points D, E, F according to the given conditions
variable [triangle ABC]
variable [is_angle_bisector A B C E Einternal] -- internal bisector of ∠CAB
variable [is_angle_bisector A B C D Eexternal] -- external bisector of ∠CAB
variable [point_on_segment B C F]

-- Circumcircle of triangle ADF and intersections with AB and AC
variable [circumcircle A D F ω]
variable [intersection ω AB I]
variable [intersection ω AC J]

variable [midpoint N I J]
variable [foot E D N H]

noncomputable def E_is_incenter_AHF : Prop :=
is_incenter_of_triangle E A H F

-- Main theorem statement
theorem problem_statement : E_is_incenter_AHF :=
sorry

end problem_statement_l754_754560


namespace sequences_and_infinite_terms_l754_754425

theorem sequences_and_infinite_terms :
    (∀ n : ℕ, a_n = 3n - 1) ∧ 
    (∀ n : ℕ, b_n = 2^n) ∧
    (∃∞ k, ∃ n, b_k = a_n) :=
by
  -- Definitions
  let a_2 := 5
  let a_8 := 23
  let b_1 := 2
  let a_n (n : ℕ) := 3 * n - 1
  let b_n (n : ℕ) := 2 ^ n

  -- Conditions
  have a_2_eq: a_2 = 5 := rfl
  have a_8_eq: a_8 = 23 := rfl
  have b_1_eq: b_1 = 2 := rfl
  have geo_prop (s t : ℕ): b_n (s + t) = b_n s * b_n t := sorry -- Provided by problem

  -- Proving general terms
  have a_gen: ∀ n : ℕ, a_n n = 3 * n - 1 := 
    by sorry -- Already derived in problem solution

  have b_gen: ∀ n : ℕ, b_n n = 2 ^ n := 
    by sorry -- Already derived in problem solution

  -- Proving infinite terms
  have inf_terms: ∃∞ k : ℕ, ∃ n : ℕ, b_n k = a_n n :=
    by sorry -- Derived in problem solution

  exact ⟨a_gen, b_gen, inf_terms⟩

end sequences_and_infinite_terms_l754_754425


namespace sum_two_and_four_l754_754203

theorem sum_two_and_four : 2 + 4 = 6 := by
  sorry

end sum_two_and_four_l754_754203


namespace projectile_explosion_height_l754_754726

noncomputable def explosion_height (c : ℝ) (t : ℝ) (v_sound : ℝ) (g : ℝ) : ℝ :=
  333 * (5 - t)

theorem projectile_explosion_height :
  ∃ t : ℝ, 
    t > 0 ∧ 
    (99 * t - (1/2) * 9.806 * t^2 = 333 * (5 - t)) ∧ 
    explosion_height 99 t 333 9.806 ≈ 431.51 :=
sorry

end projectile_explosion_height_l754_754726


namespace common_chord_length_l754_754191

-- Definition: radii and distance
def r1 : ℝ := 12
def r2 : ℝ := 15
def d : ℝ := 20

-- Theorem: length of the common chord
theorem common_chord_length : ∃ c : ℝ, c = 2 * real.sqrt (r1^2 - ((d^2 - r2^2 + r1^2)/(2*d))^2) ∧ c ≈ 25.38 :=
by
sorry

end common_chord_length_l754_754191


namespace friends_same_group_probability_l754_754147

open ProbabilityTheory

variable {Ω : Type*} [Fintype Ω]

/-- 600 students at King Middle School are divided into three groups of equal size for lunch.
  Each group has lunch at a different time. A computer randomly assigns each student to one
  of three lunch groups. -/
theorem friends_same_group_probability :
  let students := 600 
  let groups := 3
  let group_size := students / groups
  let Al Bob Carol : Ω
  let choice : Ω → Fin groups := λ s, (random_choice s) in
  P (event (choice Al = choice Bob ∧ choice Bob = choice Carol)) = 1 / 9 :=
by
  sorry

end friends_same_group_probability_l754_754147


namespace tan_double_angle_l754_754813

theorem tan_double_angle (θ : ℝ) (h1 : cos θ = -3/5) (h2 : 0 < θ ∧ θ < π) : tan (2 * θ) = 24/7 := by
  sorry

end tan_double_angle_l754_754813


namespace average_headcount_l754_754666

theorem average_headcount 
  (h1 : ℕ := 11500) 
  (h2 : ℕ := 11600) 
  (h3 : ℕ := 11300) : 
  (Float.round ((h1 + h2 + h3 : ℕ : Float) / 3) = 11467) :=
sorry

end average_headcount_l754_754666


namespace binomial_coeff_arith_seq_expansion_l754_754397

open BigOperators

-- Given the binomial expansion of (sqrt(x) + 2/sqrt(x))^n
-- we need to prove that the condition on binomial coefficients
-- implies that n = 7, and the expansion contains no constant term.
theorem binomial_coeff_arith_seq_expansion (x : ℝ) (n : ℕ) :
  (2 * Nat.choose n 2 = Nat.choose n 1 + Nat.choose n 3) ↔ n = 7 ∧ ∀ r : ℕ, x ^ (7 - 2 * r) / 2 ≠ x ^ 0 := by
  sorry

end binomial_coeff_arith_seq_expansion_l754_754397


namespace number_of_twos_eq_one_l754_754739

variable (x₁ x₂ x₃ x₄ x₅ : ℕ)

-- Conditions
def avg_condition : Prop := (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 3
def variance_condition : Prop := ((x₁ - 3)^2 + (x₂ - 3)^2 + (x₃ - 3)^2 + (x₄ - 3)^2 + (x₅ - 3)^2) / 5 = 0.4

-- The sets of possible values for each throw
axiom x₁_range : x₁ ∈ {1, 2, 3, 4, 5, 6}
axiom x₂_range : x₂ ∈ {1, 2, 3, 4, 5, 6}
axiom x₃_range : x₃ ∈ {1, 2, 3, 4, 5, 6}
axiom x₄_range : x₄ ∈ {1, 2, 3, 4, 5, 6}
axiom x₅_range : x₅ ∈ {1, 2, 3, 4, 5, 6}

theorem number_of_twos_eq_one (h_avg : avg_condition x₁ x₂ x₃ x₄ x₅) (h_var : variance_condition x₁ x₂ x₃ x₄ x₅) : 
  (multiset.count 2 {x₁, x₂, x₃, x₄, x₅}.multiset.to_set) = 1 := sorry

end number_of_twos_eq_one_l754_754739


namespace find_x_values_for_3001_l754_754327

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l754_754327


namespace num_valid_schedules_l754_754719

def num_ways_to_schedule : Nat :=
  8!

theorem num_valid_schedules (h1 : ∀ (Brown Green Clark May : Nat) (Brown < Green)
                             (Clark < May), num_ways_to_schedule / 4 = 10080) : True :=
begin
  sorry
end

end num_valid_schedules_l754_754719


namespace minimum_positive_temperature_announcement_l754_754753

-- Problem conditions translated into Lean
def num_interactions (x : ℕ) : ℕ := x * (x - 1)
def total_interactions := 132
def total_positive := 78
def total_negative := 54
def positive_temperature_count (x y : ℕ) : ℕ := y * (y - 1)
def negative_temperature_count (x y : ℕ) : ℕ := (x - y) * (x - 1 - y)
def minimum_positive_temperature (x y : ℕ) := 
  x = 12 → 
  total_interactions = total_positive + total_negative →
  total_positive + total_negative = num_interactions x →
  total_positive = positive_temperature_count x y →
  sorry -- proof goes here

theorem minimum_positive_temperature_announcement : ∃ y, 
  minimum_positive_temperature 12 y ∧ y = 3 :=
by {
  sorry -- proof goes here
}

end minimum_positive_temperature_announcement_l754_754753


namespace acute_triangle_inequality_l754_754586

variables {A B C : Type} [triangle ABC] 
noncomputable def R : ℝ := circumradius ABC
noncomputable def r : ℝ := inradius ABC
noncomputable def a : ℝ := side ABC A B
noncomputable def b : ℝ := side ABC B C
noncomputable def c : ℝ := side ABC C A

theorem acute_triangle_inequality :
  is_acute_triangle ABC -> a^2 + b^2 + c^2 ≥ 4 * (R + r)^2 := 
by
  sorry

end acute_triangle_inequality_l754_754586


namespace dot_product_sum_eq_neg277_l754_754093

variable {V : Type*} [inner_product_space ℝ V]
variables (u v w : V)

-- Given conditions
def u_norm : ∥u∥ = 4 := sorry
def v_norm : ∥v∥ = 3 := sorry
def w_norm : ∥w∥ = 5 := sorry
def sum_zero : u + 2 • v + 3 • w = 0 := sorry

-- Proof statement
theorem dot_product_sum_eq_neg277 : 
  2 * (inner_product u v + inner_product v w + inner_product w u) = -277 := 
  sorry

end dot_product_sum_eq_neg277_l754_754093


namespace pure_imaginary_satisfies_a_l754_754474

theorem pure_imaginary_satisfies_a (a : ℝ) (h : (a^2 - 3 * a + 2) + (a - 2) * complex.I.im = 0) : a = 1 :=
sorry

end pure_imaginary_satisfies_a_l754_754474


namespace glove_pairs_l754_754940

theorem glove_pairs (patterns : ℕ) (pairs_per_pattern : ℕ) (total_gloves : ℕ) 
    (h₁ : patterns = 4)
    (h₂ : pairs_per_pattern = 3)
    (h₃ : total_gloves = patterns * pairs_per_pattern * 2)
    : total_gloves - pairs_per_pattern * patterns + 1 = 13 := 
by {
    rw [h₁, h₂],
    sorry
}

end glove_pairs_l754_754940


namespace tan_difference_l754_754927

theorem tan_difference (α β : ℝ) (hα : Real.tan α = 5) (hβ : Real.tan β = 3) : 
    Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_difference_l754_754927


namespace sequence_term_3001_exists_exactly_4_values_l754_754320

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l754_754320


namespace proof_length_EF_l754_754128

-- Define the conditions as given
structure Rectangle (A B C D E F : Type) :=
(ab : ℝ)
(bc : ℝ)
(perp_bd_ef : E.Adj B F)
(a_on_de : E.Containing A D E)
(c_on_df : F.Containing C D F)

-- Define the problem statement
noncomputable def length_EF (A B C D E F : Type) [Rectangle A B C D E F]: ℝ := 
sorry -- To be defined based on the geometric configuration

-- Prove the length of EF is 70/3
theorem proof_length_EF {A B C D E F : Type} [rect : Rectangle A B C D E F]: 
  length_EF A B C D E F = 70 / 3 := 
sorry

end proof_length_EF_l754_754128


namespace intersection_points_on_line_l754_754557

noncomputable def truncated_tetrahedron (A B C A1 B1 C1 : Point) : Prop :=
  is_truncated_tetrahedron A B C A1 B1 C1

noncomputable def point_S (A B C A1 B1 C1 : Point) : Point :=
  intersection_of_extended_lateral_edges A A1 B B1 C C1

noncomputable def point_O (A B C : Point) : Point :=
  intersection_of_medians_of_base A B C

theorem intersection_points_on_line (A B C A1 B1 C1 : Point) (h : truncated_tetrahedron A B C A1 B1 C1) :
  let S := point_S A B C A1 B1 C1,
      O := point_O A B C
  in collinear_points S O (intersection_of_planes ABC A1 B1 C1 BCA A1 C1 CAB A1 B1) :=
sorry

end intersection_points_on_line_l754_754557


namespace number_of_point_configurations_l754_754030

theorem number_of_point_configurations : 
  ∃ (configurations : Finset (Finset (ℝ × ℝ))), 
    (∀ f ∈ configurations, 
      (∃ d1 d2 : ℝ, 
        (∃ A B C D : ℝ × ℝ, 
          {dist | ∃ (pair : (ℝ × ℝ) × (ℝ × ℝ)), mem_pair equiv pair ∧ dist = (euclidean_distance pair.fst pair.snd)}.card = 6 ∧
          dist = d1 ∨ dist = d2)) ∧
    configurations.card = 6)
    :=
begin
  sorry
end

end number_of_point_configurations_l754_754030


namespace equalize_balls_l754_754181

theorem equalize_balls (m n : ℕ) (h1 : n < m) (h2 : Nat.gcd m n = 1) : 
  ∀ (initial_distribution : Fin m → ℕ), ∃ k : ℕ, 
    ∀ i j : Fin m, (initial_distribution i + k * (nat_choose i n)) = (initial_distribution j + k * (nat_choose j n)) := 
sorry

end equalize_balls_l754_754181


namespace solution_exists_l754_754605

theorem solution_exists {n : ℕ} (x : fin n → fin n → ℝ) 
  (h : ∀ i j k : fin n, x i j + x j k + x k i = 0) :
  ∃ a : fin n → ℝ, ∀ i j : fin n, x i j = a i - a j :=
sorry

end solution_exists_l754_754605


namespace star_associativity_example_l754_754803

-- Define the operation star as described in the problem
def star (a b : ℝ) : ℝ := (a + b) / (a - b)

-- State the theorem to be proved
theorem star_associativity_example : star (star 1 2) 3 = 0 := by
  sorry

end star_associativity_example_l754_754803


namespace badge_ratio_l754_754752

theorem badge_ratio (total_delegates with_preprinted_bages no_badges : ℕ) 
  (h_sum : total_delegates = 36) 
  (h_preprinted : with_preprinted_bages = 16) 
  (h_no_badges : no_badges = 10) : 
  let without_preprinted := total_delegates - with_preprinted_bages,
      made_own_badges := without_preprinted - no_badges in
  made_own_badges / without_preprinted = 1 / 2 :=
by
  sorry

end badge_ratio_l754_754752


namespace Vasya_not_11_more_than_Kolya_l754_754083

def is_L_shaped (n : ℕ) : Prop :=
  n % 2 = 1

def total_cells : ℕ :=
  14400

theorem Vasya_not_11_more_than_Kolya (k v : ℕ) :
  (is_L_shaped k) → (is_L_shaped v) → (k + v = total_cells) → (k % 2 = 0) → (v % 2 = 0) → (v - k ≠ 11) := 
by
  sorry

end Vasya_not_11_more_than_Kolya_l754_754083


namespace find_f_of_neg_half_l754_754817

noncomputable def f : ℝ → ℝ :=
λ x, if -1 < x ∧ x < 0 then 1 / (f (x + 1)) else if 0 ≤ x ∧ x < 1 then x else 0

theorem find_f_of_neg_half :
  f (-1 / 2) = 2 :=
sorry

end find_f_of_neg_half_l754_754817


namespace goldbach_conjecture_refuted_l754_754021

-- Definition of Goldbach's conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → even n → ∃ p q : ℕ, prime p ∧ prime q ∧ n = p + q

-- Definition of a counterexample to Goldbach's conjecture
def counterexample_to_goldbach (n : ℕ) : Prop :=
  n > 2 ∧ even n ∧ ∀ p q : ℕ, ¬ (prime p ∧ prime q ∧ n = p + q)

-- The proof problem
theorem goldbach_conjecture_refuted : ∃ n : ℕ, counterexample_to_goldbach n → ¬ goldbach_conjecture :=
sorry

end goldbach_conjecture_refuted_l754_754021


namespace calc_z_conj_diff_l754_754827

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754827


namespace prove_AB_and_circle_symmetry_l754_754949

-- Definition of point A
def pointA : ℝ × ℝ := (4, -3)

-- Lengths relation |AB| = 2|OA|
def lengths_relation(u v : ℝ) : Prop :=
  u^2 + v^2 = 100

-- Orthogonality condition for AB and OA
def orthogonality_condition(u v : ℝ) : Prop :=
  4 * u - 3 * v = 0

-- Condition that ordinate of B is greater than 0
def ordinate_condition(v : ℝ) : Prop :=
  v - 3 > 0

-- Equation of the circle given in the problem
def given_circle_eqn(x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Symmetric circle equation to be proved
def symmetric_circle_eqn(x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

theorem prove_AB_and_circle_symmetry :
  (∃ u v : ℝ, lengths_relation u v ∧ orthogonality_condition u v ∧ ordinate_condition v ∧ u = 6 ∧ v = 8) ∧
  (∃ x y : ℝ, given_circle_eqn x y → symmetric_circle_eqn x y) :=
by
  sorry

end prove_AB_and_circle_symmetry_l754_754949


namespace population_growth_nearest_hundred_l754_754952

theorem population_growth_nearest_hundred  :
  (let births_per_year := 365 * (24 / 6),
       deaths_per_year := 365 * (24 / 36),
       net_increase := births_per_year - deaths_per_year
   in (Float.round ((10 / 3) * 365)).toNat) = 1200 :=
by
  sorry

end population_growth_nearest_hundred_l754_754952


namespace regular_octahedron_has_4_pairs_l754_754054

noncomputable def regular_octahedron_parallel_edges : ℕ :=
  4

theorem regular_octahedron_has_4_pairs
  (h : true) : regular_octahedron_parallel_edges = 4 :=
by
  sorry

end regular_octahedron_has_4_pairs_l754_754054


namespace temperature_at_500_meters_l754_754574

theorem temperature_at_500_meters
  (init_temp : ℝ)
  (decrease_per_100m : ℝ)
  (elevation_gain : ℝ) :
  init_temp = 28 →
  decrease_per_100m = 0.7 →
  elevation_gain = 500 →
  let temp_decrease := (elevation_gain / 100) * decrease_per_100m
  in init_temp - temp_decrease = 24.5 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end temperature_at_500_meters_l754_754574


namespace salary_increase_after_five_years_l754_754116

theorem salary_increase_after_five_years :
  ∀ (S : ℝ), (S * (1.15)^5 - S) / S * 100 = 101.14 := by
sorry

end salary_increase_after_five_years_l754_754116


namespace repeating_decimal_sum_l754_754355

theorem repeating_decimal_sum : (0.\overline{3} : ℚ) + (0.\overline{04} : ℚ) + (0.\overline{005} : ℚ) = 1135 / 2997 := 
sorry

end repeating_decimal_sum_l754_754355


namespace value_of_x_l754_754044

theorem value_of_x (x : ℝ) (h1 : |x| - 1 = 0) (h2 : x - 1 ≠ 0) : x = -1 := 
sorry

end value_of_x_l754_754044


namespace number_of_stamps_l754_754049

theorem number_of_stamps 
    (foreign : ℕ)
    (old : ℕ)
    (both : ℕ)
    (neither : ℕ) 
    (h_foreign : foreign = 90)
    (h_old : old = 80)
    (h_both : both = 20)
    (h_neither : neither = 50) : 
    ∃ S : ℕ, S = 200 :=
by
  -- Defining the number of stamps that are either foreign or more than 10 years old
  let foreign_or_old := foreign + old - both

  -- Stating the total number of stamps in the collection
  let S := foreign_or_old + neither

  -- Providing the necessary conditions
  have h_S : S = 200, from by
    rw [foreign_or_old, h_foreign, h_old, h_both, h_neither]
    rw [show 90 + 80 - 20 = 150 by norm_num]
    rw [show 150 + 50 = 200 by norm_num]

  -- Stating the existence of the total number of stamps
  existsi S
  exact h_S

end number_of_stamps_l754_754049


namespace negative_half_less_than_negative_third_l754_754276

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l754_754276


namespace sqrt_of_non_square_is_irrational_l754_754136

theorem sqrt_of_non_square_is_irrational (a : ℤ) (h : ¬ ∃ k : ℤ, a = k * k) : irrational (Real.sqrt a) := sorry

end sqrt_of_non_square_is_irrational_l754_754136


namespace simplify_and_evaluate_l754_754138

theorem simplify_and_evaluate (x : ℝ) (h : x = 2 + Real.sqrt 2) :
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l754_754138


namespace find_S11_l754_754951

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function

-- Define conditions
def arithmetic_sequence (a : ℕ → ℚ) :=
∀ n m, a (n + m) = a n + a m

def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n / 2 : ℚ) * (a 1 + a n)

-- Define the problem statement to be proved
theorem find_S11 (h_arith : arithmetic_sequence a) (h_eq : a 3 + a 6 + a 9 = 54) : 
  S 11 a = 198 :=
sorry

end find_S11_l754_754951


namespace cos_C_given_sin_A_and_cos_B_l754_754518

theorem cos_C_given_sin_A_and_cos_B (A B C : ℝ) (h_triangle : A + B + C = real.pi)
  (h_sinA : real.sin A = 4 / 5) (h_cosB : real.cos B = 12 / 13) :
  real.cos C = -16 / 65 :=
sorry

end cos_C_given_sin_A_and_cos_B_l754_754518


namespace marko_formed_number_l754_754567

theorem marko_formed_number (a b : ℕ) (h_prime_a : Nat.Prime a) (h_prime_b : Nat.Prime b) (h_digits : a / 100 = 0 ∧ a ≥ 10 ∧ b / 100 = 0 ∧ b ≥ 10) 
(h_combination : let c := 100 * a + b in c - a * b = 154) : (100 * a + b) = 1997 := by
  sorry

end marko_formed_number_l754_754567


namespace calc_z_conj_diff_l754_754832

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754832


namespace no_statements_false_l754_754772

variables {p q : ℝ} {P Q A B : ℝ}

-- Conditions
def is_on_circle (r : ℝ) (center : ℝ) (point : ℝ) : Prop :=
  abs (center - point) = r

def AB_on_PQ := A = B ∨ (A < B ∧ B < P ∨ P < B ∧ B < A)

def AB_PQ_distance_eq (p q a b pq ab : ℝ) : Prop :=
  p - q = pq + ab

def AB_PQ_distance_eq_2 (p q a b pq ab : ℝ) : Prop :=
  p + q = pq - ab

def AB_PQ_distance_less (p q a b pq ab : ℝ) : Prop :=
  p + q < pq + ab

def AB_PQ_distance_less_2 (p q a b pq ab : ℝ) : Prop :=
  p - q < pq - ab

-- Lean theorem statement
theorem no_statements_false : ¬ (AB_on_PQ ∧ 
    (AB_PQ_distance_eq p q A B PQ ab ∨ 
    AB_PQ_distance_eq_2 p q A B PQ ab ∨ 
    AB_PQ_distance_less p q A B PQ ab ∨ 
    AB_PQ_distance_less_2 p q A B PQ ab)) :=
sorry

end no_statements_false_l754_754772


namespace trader_sold_95_pens_l754_754740

theorem trader_sold_95_pens
  (C : ℝ)   -- cost price of one pen
  (N : ℝ)   -- number of pens sold
  (h1 : 19 * C = 0.20 * N * C):  -- condition: profit from selling N pens is equal to the cost of 19 pens, with 20% gain percentage
  N = 95 := by
-- You would place the proof here.
  sorry

end trader_sold_95_pens_l754_754740


namespace rectangular_box_dimensions_l754_754727

theorem rectangular_box_dimensions (m n : ℕ) (h_relatively_prime : Nat.coprime m n) 
(width length : ℕ) (height : ℕ) (triangle_area : ℝ) 
(h_width : width = 10) 
(h_length : length = 20)
(h_height : height = m / n)
(h_triangle_area: triangle_area = 40) : m + n = 17 :=
sorry

end rectangular_box_dimensions_l754_754727


namespace ratio_initial_doubled_l754_754720

theorem ratio_initial_doubled (x : ℤ) (h : 3 * (2 * x + 9) = 75) : x:2*x = 1:2 :=
by
  sorry

end ratio_initial_doubled_l754_754720


namespace Frank_days_to_finish_book_l754_754807

theorem Frank_days_to_finish_book (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 22) (h2 : total_pages = 12518) : total_pages / pages_per_day = 569 := by
  sorry

end Frank_days_to_finish_book_l754_754807


namespace angle_AEB_is_90_l754_754076

-- The given conditions and definitions encapsulated in Lean
variables {A B C D M H E : Type*}
variables [Triangle ABC] [Triangle D M H] [Circumcircle D M H]
variables (bisector_AD : AngleBisector A B C D)
variables (altitude_AH : Altitude A H)
variables (midpoint_M : Midpoint A D M)
variables (circum_intersects : ∃ E, E ≠ M ∧ CircumcircleMDH.contains E ∧ LiesOn C M E)

-- The main theorem statement
theorem angle_AEB_is_90 (hABC1 : AB > AC) :
  ∠ (A,E,B) = 90 :=
by 
  sorry

end angle_AEB_is_90_l754_754076


namespace wicket_count_l754_754683

theorem wicket_count (initial_avg new_avg : ℚ) (runs_last_match wickets_last_match : ℕ) (delta_avg : ℚ) (W : ℕ) :
  initial_avg = 12.4 →
  new_avg = 12.0 →
  delta_avg = 0.4 →
  runs_last_match = 26 →
  wickets_last_match = 8 →
  initial_avg * W + runs_last_match = new_avg * (W + wickets_last_match) →
  W = 175 := by
  sorry

end wicket_count_l754_754683


namespace student_age_is_24_l754_754235

/-- A man is 26 years older than his student. In two years, his age will be twice the age of his student.
    Prove that the present age of the student is 24 years old. -/
theorem student_age_is_24 (S M : ℕ) (h1 : M = S + 26) (h2 : M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end student_age_is_24_l754_754235


namespace zero_nim_sum_move_nonzero_nim_sum_move_winning_strategy_move_for_345_l754_754161

variable (n : ℕ)
variable {m : ℕ}
variable {piles : List ℕ}

-- Given the condition for nim-sum
def nim_sum (piles : List ℕ) : ℕ :=
  List.foldr xor 0 piles

-- Part (a)
theorem zero_nim_sum_move (piles : List ℕ) (h : nim_sum piles = 0)
: ∃ piles', nim_sum piles' ≠ 0 :=
sorry

-- Part (b)
theorem nonzero_nim_sum_move (piles : List ℕ) (h : nim_sum piles ≠ 0)
: ∃ piles', nim_sum piles' = 0 :=
sorry

-- Part (c)
theorem winning_strategy (piles : List ℕ)
: (nim_sum piles = 0 ∧ ∀ piles', nim_sum piles' ≠ 0) ∨ (nim_sum piles ≠ 0 ∧ ∃ piles', nim_sum piles' = 0) :=
sorry

-- Part (d)
def move_for_piles (piles : List ℕ) : List ℕ :=
if piles = [3, 4, 5] then [1, 4, 5] else piles

theorem move_for_345 (piles : List ℕ) (h : piles = [3, 4, 5])
: move_for_piles piles = [1, 4, 5] :=
by {simp [move_for_piles, h]}

end zero_nim_sum_move_nonzero_nim_sum_move_winning_strategy_move_for_345_l754_754161


namespace distinct_solutions_diff_l754_754992

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l754_754992


namespace point_in_second_quadrant_l754_754950

theorem point_in_second_quadrant (x : ℚ) : 
  let P : ℚ × ℚ := (-1, x^2 + 1) in
  P.1 < 0 ∧ P.2 > 0 :=
by
  sorry

end point_in_second_quadrant_l754_754950


namespace range_of_m_l754_754564

noncomputable def f (a x: ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem range_of_m (a m x₁ x₂: ℝ) (h₁: a ∈ Set.Icc (-3) (0)) (h₂: x₁ ∈ Set.Icc (0) (2)) (h₃: x₂ ∈ Set.Icc (0) (2)) : m ∈ Set.Ici (5) → m - a * m^2 ≥ |f a x₁ - f a x₂| :=
sorry

end range_of_m_l754_754564


namespace inscribed_triangle_l754_754697

theorem inscribed_triangle (r : ℝ) (A B C : ℝ → ℝ → ℝ) 
(h1 : ∃ p q : ℝ, A = (p, 0) ∧ B = (q, 0) ∧ (q - p = 2 * r) )
(h2 : C ≠ A ∧ C ≠ B)
(h3 : ∃ x y : ℝ, C = (x, y) ∧  x^2 + y^2 = r^2)
: ∀ s : ℝ, 
  s = (AC(C,A) + BC(C,B)) → 
  s^2 ≤ 8 * r^2 :=
sorry

end inscribed_triangle_l754_754697


namespace students_answered_both_correctly_l754_754117

theorem students_answered_both_correctly (x y z w total : ℕ) (h1 : x = 22) (h2 : y = 20) 
  (h3 : z = 3) (h4 : total = 25) (h5 : x + y - w - z = total) : w = 17 :=
by
  sorry

end students_answered_both_correctly_l754_754117


namespace complex_conjugate_difference_l754_754818

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754818


namespace units_digit_of_result_l754_754618

theorem units_digit_of_result (
  a b c : ℕ
  (h1 : a = c - 3) 
  (h2 : 100 * a + 10 * b + c = 101 * c + 10 * b - 300) 
  (h3 : 100 * c + 10 * b + a = 101 * c + 10 * b - 3)
) : (2 * (101 * c + 10 * b - 300) - (101 * c + 10 * b - 3)) % 10 = 1 := 
by
  sorry

end units_digit_of_result_l754_754618


namespace basketball_75th_percentile_is_39_l754_754488

def basketScores : List ℕ := [29, 30, 38, 25, 37, 40, 42, 32]

def percentile (p : ℕ) (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (· < ·)
  let n := lst.length
  if n = 0 then 
    0
  else
    let pos := ((p * n + 99) / 100)
    if pos < 1 then 
      sorted_lst.get! 0
    else if pos ≥ n then
      sorted_lst.get! (n - 1)
    else
      (sorted_lst.get! (pos - 1) + sorted_lst.get! pos) / 2

theorem basketball_75th_percentile_is_39 :
  percentile 75 basketScores = 39 :=
by
  sorry

end basketball_75th_percentile_is_39_l754_754488


namespace four_letter_arrangements_count_l754_754913

-- Definitions based on the conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def first_letter (l : List Char) : Prop := l.head = 'D'
def contains_A (l : List Char) : Prop := 'A' ∈ l.tail
def no_repeats (l : List Char) : Prop := l.nodup

-- Problem statement
theorem four_letter_arrangements_count : 
  ∃ l : List Char, 
    l.length = 4 ∧ 
    first_letter l ∧ 
    contains_A l ∧ 
    no_repeats l ∧ 
    (l.countp (∈ letters) = 4) := sorry

end four_letter_arrangements_count_l754_754913


namespace find_x_values_for_3001_l754_754333

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l754_754333


namespace interval_decrease_min_value_range_of_k_l754_754007

noncomputable def f (k : ℝ) (x : ℝ) := log x + k / x

-- Problem (I)
theorem interval_decrease_min_value (x : ℝ) (h : 0 < x) (h_tangent : f e > 0) :
  (f e) = 2 ∧ (∀ z, 0 < z ∧ z < e → f z > 2) :=
sorry

-- Problem (II)
theorem range_of_k (k x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) :
  f x1 - x1 < f x2 - x2 → k ≥ 1/4 :=
sorry

end interval_decrease_min_value_range_of_k_l754_754007


namespace car_speed_first_hour_l754_754176

theorem car_speed_first_hour
  (x : ℕ)
  (speed_second_hour : ℕ := 80)
  (average_speed : ℕ := 90)
  (total_time : ℕ := 2)
  (h : average_speed = (x + speed_second_hour) / total_time) :
  x = 100 :=
by
  sorry

end car_speed_first_hour_l754_754176


namespace necklace_cost_l754_754529

def bead_necklaces := 3
def gemstone_necklaces := 3
def total_necklaces := bead_necklaces + gemstone_necklaces
def total_earnings := 36

theorem necklace_cost :
  (total_earnings / total_necklaces) = 6 :=
by
  -- Proof goes here
  sorry

end necklace_cost_l754_754529


namespace min_value_l754_754931

theorem min_value (a b : ℝ) (h : a * b > 0) : (∃ x, x = a^2 + 4 * b^2 + 1 / (a * b) ∧ ∀ y, y = a^2 + 4 * b^2 + 1 / (a * b) → y ≥ 4) :=
sorry

end min_value_l754_754931


namespace cos_C_in_triangle_l754_754521

theorem cos_C_in_triangle
  (A B C : ℝ) (sin_A : ℝ) (cos_B : ℝ)
  (h1 : sin_A = 4 / 5)
  (h2 : cos_B = 12 / 13) :
  cos (π - A - B) = -16 / 65 :=
by
  -- Proof steps would be included here
  sorry

end cos_C_in_triangle_l754_754521


namespace complex_z_sub_conjugate_eq_neg_i_l754_754849

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754849


namespace board_meeting_distinct_ways_l754_754223

def numDistinctArrangements : ℕ := 56

theorem board_meeting_distinct_ways :
  let total_seats := 10
  let stools_needed := 5
  let rocking_chairs_needed := 5
  let first_and_last_stools := 2
  combinatorics.binomial (total_seats - first_and_last_stools) (stools_needed - first_and_last_stools) = numDistinctArrangements :=
by
  sorry

end board_meeting_distinct_ways_l754_754223


namespace sum_of_distances_equilateral_triangle_l754_754589

-- defining the problem as a Lean statement
theorem sum_of_distances_equilateral_triangle (A B C P : Point) (h_eq_triangle : equilateral_triangle A B C) :
  ∃ K, ∀ D1 D2 D3, perpendicular_from D1 P A B ∧ perpendicular_from D2 P B C ∧ perpendicular_from D3 P C A → 
  PD_dist_sum = K :=
by
  sorry

end sum_of_distances_equilateral_triangle_l754_754589


namespace total_boxes_count_l754_754712

theorem total_boxes_count 
    (apples_per_crate : ℕ) (apples_crates : ℕ) 
    (oranges_per_crate : ℕ) (oranges_crates : ℕ) 
    (bananas_per_crate : ℕ) (bananas_crates : ℕ) 
    (rotten_apples_percentage : ℝ) (rotten_oranges_percentage : ℝ) (rotten_bananas_percentage : ℝ)
    (apples_per_box : ℕ) (oranges_per_box : ℕ) (bananas_per_box : ℕ) :
    apples_per_crate = 42 → apples_crates = 12 → 
    oranges_per_crate = 36 → oranges_crates = 15 → 
    bananas_per_crate = 30 → bananas_crates = 18 → 
    rotten_apples_percentage = 0.08 → rotten_oranges_percentage = 0.05 → rotten_bananas_percentage = 0.02 →
    apples_per_box = 10 → oranges_per_box = 12 → bananas_per_box = 15 →
    ∃ total_boxes : ℕ, total_boxes = 126 :=
by sorry

end total_boxes_count_l754_754712


namespace tree_height_relationship_l754_754250

theorem tree_height_relationship (x : ℕ) : 
  let y := 2.5 + 0.22 * x in
  y = 2.5 + 0.22 * x :=
by
  let y := 2.5 + 0.22 * x
  exact rfl

end tree_height_relationship_l754_754250


namespace find_m_l754_754042

theorem find_m (m x : ℝ) (h : √(2 * x + m) = x) (hx : x = 1) : m = -1 :=
sorry

end find_m_l754_754042


namespace no_convex_polygon_with_equal_sides_and_obtuse_triangles_l754_754783

open Classical

noncomputable theory

def is_convex (P : Polygon) : Prop := sorry
def all_side_lengths_equal (P : Polygon) : Prop := sorry
def any_three_vertices_form_obtuse_triangle (P : Polygon) : Prop := sorry

theorem no_convex_polygon_with_equal_sides_and_obtuse_triangles :
  ¬ ∃ P : Polygon, is_convex P ∧ all_side_lengths_equal P ∧ any_three_vertices_form_obtuse_triangle P :=
sorry

end no_convex_polygon_with_equal_sides_and_obtuse_triangles_l754_754783


namespace axis_of_symmetry_of_inverse_proportional_function_l754_754162

variable (k : ℝ)

-- Define the inverse proportional function
def inverseProportional (x : ℝ) : ℝ := k / x

-- Define the candidate axes of symmetry
def candidateAxis (x : ℝ) : ℝ :=
  if k > 0 then -x else x

-- Theorem statement
theorem axis_of_symmetry_of_inverse_proportional_function (k ≠ 0) (k ≠ 1) :
    ∀ (x : ℝ), candidateAxis k x = (if k > 0 then y = -x else y = x) :=
sorry

end axis_of_symmetry_of_inverse_proportional_function_l754_754162


namespace certain_number_z_l754_754037

theorem certain_number_z (x y z : ℝ) (h1 : 0.5 * x = y + z) (h2 : x - 2 * y = 40) : z = 20 :=
by 
  sorry

end certain_number_z_l754_754037


namespace f_f_f_3_l754_754465

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_f_f_3 : f(f(f(3))) = 107 := 
by
  sorry

end f_f_f_3_l754_754465


namespace phase_and_vertical_shift_of_sin_l754_754779

def phase_shift_vertical_shift (y : ℝ → ℝ) (h : y = λ x, sin (4 * x - π / 2) + 2) : Prop :=
  (∀ x, y x = sin (4 * x - π / 2) + 2) → 
  (phase_shift y = π / 8 ∧ vertical_shift y = 2)

theorem phase_and_vertical_shift_of_sin:
  phase_shift_vertical_shift (λ x, sin (4 * x - π / 2) + 2) (by funext; simp [sin, *]) :=
sorry

end phase_and_vertical_shift_of_sin_l754_754779


namespace ratio_RN_NQ_l754_754936

/-- 
  In triangle PQR, points M and N lie on segments PQ and PR respectively. 
  Segments PM and QN intersect at point S.
  Given the ratios PS/SM = 4 and QS/SN = 3, 
  we want to prove that RN/NQ = 4/3.
-/
theorem ratio_RN_NQ (P Q R M N S : Type)
  [IsPoint P] [IsPoint Q] [IsPoint R] [IsPoint M] [IsPoint N] [IsPoint S]
  (PQ : Line P Q) (PR : Line P R) (PM : Line P M) (QN : Line Q N)
  (hM : M ∈ PQ) (hN : N ∈ PR) (hS : S ∈ PM ∧ S ∈ QN)
  (hPS_SM : ratio (length (S, P)) (length (S, M)) = 4)
  (hQS_SN : ratio (length (S, Q)) (length (S, N)) = 3) :
  ratio (length (R, N)) (length (N, Q)) = 4 / 3 :=
sorry

end ratio_RN_NQ_l754_754936


namespace complex_conj_difference_l754_754841

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754841


namespace sin_squared_alpha_beta_range_l754_754810

theorem sin_squared_alpha_beta_range (α β : ℝ) 
  (h : 3 * (sin α)^2 - 2 * sin α + 2 * (sin β)^2 = 0) : 
  0 ≤ (sin α)^2 + (sin β)^2 ∧ (sin α)^2 + (sin β)^2 ≤ 4 / 9 :=
sorry

end sin_squared_alpha_beta_range_l754_754810


namespace race_distance_100_l754_754059

noncomputable def race_distance (a b c d : ℝ) :=
  (d / a = (d - 20) / b) ∧
  (d / b = (d - 10) / c) ∧
  (d / a = (d - 28) / c) 

theorem race_distance_100 (a b c d : ℝ) (h1 : d / a = (d - 20) / b) (h2 : d / b = (d - 10) / c) (h3 : d / a = (d - 28) / c) : 
  d = 100 :=
  sorry

end race_distance_100_l754_754059


namespace length_of_BD_l754_754963

open Real

/-- In triangle ABC, with ∠B = 90°, AB = 3, and BC = 4,
let the bisector of ∠BAC meet BC at D. Then the length of BD is 12/7. -/
theorem length_of_BD
  (A B C D : Point)
  (hB : angle A B C = π/2)
  (hAB : dist A B = 3)
  (hBC : dist B C = 4)
  (h_segment : is_angle_bisector A B C D) :
  dist B D = 12/7 := by
  sorry

end length_of_BD_l754_754963


namespace pipe_A_rate_l754_754124

theorem pipe_A_rate :
  ∃ (A : ℝ), (let tank_capacity : ℝ := 750
                    pipe_B_rate : ℝ := 30
                    pipe_C_rate : ℝ := 20
                    cycle_minutes : ℝ := 3
                    total_minutes : ℝ := 45
                    net_rate_per_cycle := A + pipe_B_rate - pipe_C_rate
                    number_of_cycles := total_minutes / cycle_minutes
                    total_water_added := number_of_cycles * net_rate_per_cycle
  in total_water_added = tank_capacity) ∧ A = 40 :=
by
  have A : ℝ := 40
  let tank_capacity : ℝ := 750
  let pipe_B_rate : ℝ := 30
  let pipe_C_rate : ℝ := 20
  let cycle_minutes : ℝ := 3
  let total_minutes : ℝ := 45
  let net_rate_per_cycle := A + pipe_B_rate - pipe_C_rate
  let number_of_cycles := total_minutes / cycle_minutes
  let total_water_added := number_of_cycles * net_rate_per_cycle
  have h1 : total_water_added = tank_capacity := by
      simp only [tank_capacity, pipe_B_rate, pipe_C_rate, cycle_minutes, total_minutes, 
        net_rate_per_cycle, number_of_cycles, total_water_added, A]
      norm_num
  exact ⟨A, h1, rfl⟩

end pipe_A_rate_l754_754124


namespace maximum_value_of_S_l754_754437

-- Define the sequence {a_n} where a_1 is 2
def a : ℕ → ℝ
| 0     := 0 -- Not defined, use 1-based indexing
| 1     := 2
| (n+2) := 3 * a (n + 1) - 4^n

-- Define the partial sum sequence {S_n}
def S (n : ℕ) : ℝ :=
  if n = 0 then 0
  else 1 / 6 * (3 * a (n + 1) + 4^n - 1)

-- Statement to prove: the maximum value of S_n is 35
theorem maximum_value_of_S : ∃ n : ℕ, S n = 35 := 
sorry

end maximum_value_of_S_l754_754437


namespace rubber_ball_radius_l754_754717

theorem rubber_ball_radius (r : ℝ) (radius_exposed_section : ℝ) (depth : ℝ) 
  (h1 : radius_exposed_section = 20) 
  (h2 : depth = 12) 
  (h3 : (r - depth)^2 + radius_exposed_section^2 = r^2) : 
  r = 22.67 :=
by
  sorry

end rubber_ball_radius_l754_754717


namespace parallelogram_count_l754_754782

theorem parallelogram_count (n : ℕ) : 
  let S := 3 * Nat.choose (n+2) 4
  in True := 
begin
  sorry
end

end parallelogram_count_l754_754782


namespace area_of_quadrilateral_XCYZ_l754_754598

theorem area_of_quadrilateral_XCYZ (A B C D W X Y Z : ℝ) :
  A = 0 ∧ B = 0 ∧ C = 10 ∧ D = 10 ∧ W = 5 ∧ X = 5 ∧ Y = 5 ∧ Z = 5 → 
  XCYZ.area = 25 := sorry

end area_of_quadrilateral_XCYZ_l754_754598


namespace geometric_sequence_first_term_l754_754885

noncomputable def sum_of_geometric_sequence (n : ℕ) (t : ℝ) : ℝ := 2010^n + t
noncomputable def a1 (t : ℝ) : ℝ := sum_of_geometric_sequence 1 t
noncomputable def a2 (t : ℝ) : ℝ := sum_of_geometric_sequence 2 t - sum_of_geometric_sequence 1 t
noncomputable def a3 (t : ℝ) : ℝ := sum_of_geometric_sequence 3 t - sum_of_geometric_sequence 2 t

theorem geometric_sequence_first_term (t : ℝ) (h : a1 t * a3 t = (a2 t)^2) : a1 t = 2009 :=
by
  -- Definition of a1, a2, and a3 based on given conditions
  let a1_val := 2010 + t
  let a2_val := 2009 * 2010
  let a3_val := 2009 * 2010^2
  have h1 : a1 t = a1_val := by sorry
  have h2 : a2 t = a2_val := by sorry
  have h3 : a3 t = a3_val := by sorry
  -- Using given property to solve for t and subsequently a1
  have key_eq : a1_val * a3_val = a2_val^2 := by sorry
  have t_val := -1 := by sorry
  show a1_val = 2009 := by sorry

-- Proof skipped (using sorry)

end geometric_sequence_first_term_l754_754885


namespace area_triangle_ABC_l754_754953

noncomputable def point := ℝ × ℝ

structure Parallelogram (A B C D : point) : Prop :=
(parallel_AB_CD : ∃ m1 m2, m1 ≠ m2 ∧ (A.2 - B.2) / (A.1 - B.1) = m1 ∧ (C.2 - D.2) / (C.1 - D.1) = m2)
(equal_heights : ∃ h, (B.2 - A.2 = h) ∧ (C.2 - D.2 = h))
(area_parallelogram : (B.1 - A.1) * (B.2 - A.2) + (C.1 - D.1) * (C.2 - D.2) = 27)
(thrice_length : (C.1 - D.1) = 3 * (B.1 - A.1))

theorem area_triangle_ABC (A B C D : point) (h : Parallelogram A B C D) : 
  ∃ triangle_area : ℝ, triangle_area = 13.5 :=
by
  sorry

end area_triangle_ABC_l754_754953


namespace sum_g_eq_1000_5_l754_754555

def g (x: ℝ) : ℝ := 4 / (16^x + 4)

theorem sum_g_eq_1000_5 : 
  (∑ k in finset.range 2001, g ((k + 1) / 2002)) = 1000.5 :=
sorry

end sum_g_eq_1000_5_l754_754555


namespace log_of_fractional_root_l754_754348

theorem log_of_fractional_root : log 4 (1 / (4 ^ (1/3))) = -1 / 3 :=
by
  sorry

end log_of_fractional_root_l754_754348


namespace unique_positive_integer_b_quadratic_solution_l754_754778

theorem unique_positive_integer_b_quadratic_solution (c : ℝ) :
  (∃! (b : ℕ), ∀ (x : ℝ), x^2 + (b^2 + (1 / b^2)) * x + c = 3) ↔ c = 5 :=
sorry

end unique_positive_integer_b_quadratic_solution_l754_754778


namespace greatest_possible_value_l754_754459

theorem greatest_possible_value (x : ℝ) (h : 13 = x^2 + 1 / x^2) : x + 1 / x ≤ Real.sqrt 15 :=
begin
  sorry
end

end greatest_possible_value_l754_754459


namespace a_2022_l754_754886

universe u

noncomputable def sequence (n : ℕ) : ℕ
| 0     := 0
| (n+1) := 2 * (∑ i in Finset.range (n + 1), sequence i)

def a (n : ℕ) : ℕ :=
match n with
| 0     := 0
| 1     := 1
| (n+1) := 2 * (Finset.sum (Finset.range n) (λ i, a (i + 1)))

theorem a_2022 : a 2022 = 2 * 3^2020 := by
  sorry

end a_2022_l754_754886


namespace max_intersection_points_four_circles_l754_754393

theorem max_intersection_points_four_circles
  (C1 C2 C3 C4 : Set Point)
  (h1 : ∀ (L : Line), ∀ (C : Set Point), (C1 = C ∨ C2 = C ∨ C3 = C ∨ C4 = C) → ∃ (A B : Point), A ∈ L ∧ A ∈ C ∧ B ∈ L ∧ B ∈ C ∧ A ≠ B)
  (coplanar : ∃ P : Plane, ∀ C ∈ {C1, C2, C3, C4}, C ⊆ P) :
  ∃ (L : Line), ∀ (C : Set Point), (C ∈ {C1, C2, C3, C4}) → (∃ (A B : Point), A ∈ L ∧ A ∈ C ∧ B ∈ L ∧ B ∈ C ∧ A ≠ B) :=
begin
  sorry
end

end max_intersection_points_four_circles_l754_754393


namespace A_subscribed_fraction_l754_754252

theorem A_subscribed_fraction 
  (total_profit : ℝ) (A_share : ℝ) 
  (B_fraction : ℝ) (C_fraction : ℝ) 
  (A_fraction : ℝ) :
  total_profit = 2430 →
  A_share = 810 →
  B_fraction = 1/4 →
  C_fraction = 1/5 →
  A_fraction = A_share / total_profit →
  A_fraction = 1/3 :=
by
  intros h_total_profit h_A_share h_B_fraction h_C_fraction h_A_fraction
  sorry

end A_subscribed_fraction_l754_754252


namespace real_value_of_x_l754_754454

variable {x : ℝ}

def is_real (z : ℂ) : Prop := z.im = 0

theorem real_value_of_x (h : is_real ((x : ℂ) + Complex.I) ^ 2) : x = 0 :=
by
  sorry

end real_value_of_x_l754_754454


namespace complex_conj_difference_l754_754836

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754836


namespace tax_amount_l754_754978

theorem tax_amount
  (lee_money : ℕ)
  (friend_money : ℕ)
  (cost_chicken_wings : ℕ)
  (cost_chicken_salad : ℕ)
  (cost_soda : ℕ)
  (num_sodas : ℕ)
  (total_change : ℕ)
  (total_spent : ℕ)
  (before_tax_total : ℕ):
  lee_money + friend_money = 18 →
  cost_chicken_wings = 6 →
  cost_chicken_salad = 4 →
  cost_soda = 1 →
  num_sodas = 2 →
  total_change = 3 →
  total_spent = lee_money + friend_money - total_change →
  before_tax_total = cost_chicken_wings + cost_chicken_salad + (cost_soda * num_sodas) →
  total_spent - before_tax_total = 3 :=
begin
  intros,
  sorry
end

end tax_amount_l754_754978


namespace max_operations_proof_l754_754616

-- Define the range of numbers on the blackboard
def numbers := {1, 2, 3, ..., 510}

-- Define the condition for the operation: sum of two numbers is prime
def is_prime_sum (x y : ℕ) : Prop := Nat.Prime (x + y)

-- Define the function to count the maximum number of operations 
noncomputable def max_operations : ℕ := 255

-- The main theorem stating the maximum number of operations possible on the blackboard is 255
theorem max_operations_proof : 
    ∃ (operations : list (ℕ × ℕ)), 
    (∀ (op ∈ operations), is_prime_sum (op.1) (op.2) ∧ (op.1 ∈ numbers ∧ op.2 ∈ numbers)) ∧
    (operations.length = max_operations) := sorry

end max_operations_proof_l754_754616


namespace algebraic_expression_value_l754_754006

theorem algebraic_expression_value (a b : ℝ) (h : ∃ x : ℝ, x = 2 ∧ 3 * (a - x) = 2 * (b * x - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 :=
by sorry

end algebraic_expression_value_l754_754006


namespace zero_points_C_exist_l754_754490

theorem zero_points_C_exist (A B C : ℝ × ℝ) (hAB_dist : dist A B = 12) (h_perimeter : dist A B + dist A C + dist B C = 52)
    (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 100) : 
    false :=
by
  sorry

end zero_points_C_exist_l754_754490


namespace radius_inscribed_sphere_is_1_by_4_l754_754865

-- Conditions
structure Tetrahedron :=
(P A B C : ℝ → ℝ → ℝ)

def is_perpendicular (v1 v2 : ℝ → ℝ → ℝ) : Prop :=
v1.dot v2 = 0

noncomputable def radius_of_inscribed_sphere (tetra : Tetrahedron) : ℝ :=
sorry -- The actual calculation would be here.

-- Given conditions
variables (tetra : Tetrahedron)
-- PA = 2, PB = 1, PC = 1, and they are mutually perpendicular
axiom PA_length : (tetra.P - tetra.A).norm = 2
axiom PB_length : (tetra.P - tetra.B).norm = 1
axiom PC_length : (tetra.P - tetra.C).norm = 1
axiom perp_PA_PB : is_perpendicular (tetra.P - tetra.A) (tetra.P - tetra.B)
axiom perp_PB_PC : is_perpendicular (tetra.P - tetra.B) (tetra.P - tetra.C)
axiom perp_PC_PA : is_perpendicular (tetra.P - tetra.C) (tetra.P - tetra.A)

-- Proof problem statement
theorem radius_inscribed_sphere_is_1_by_4 :
radius_of_inscribed_sphere tetra = 1/4 :=
sorry

end radius_inscribed_sphere_is_1_by_4_l754_754865


namespace non_obtuse_triangle_inequality_l754_754052

-- Define non-obtuse angles
variables {A B C : ℝ}
-- Conditions for a non-obtuse triangle
hypothesis hA : A ≤ Real.pi / 2
hypothesis hB : B ≤ Real.pi / 2
hypothesis hC : C ≤ Real.pi / 2

-- The main theorem to be proved
theorem non_obtuse_triangle_inequality (hA : A ≤ Real.pi / 2) (hB : B ≤ Real.pi / 2) (hC : C ≤ Real.pi / 2) :
  (1 - Real.cos (2 * A)) * (1 - Real.cos (2 * B)) / (1 - Real.cos (2 * C)) +
  (1 - Real.cos (2 * C)) * (1 - Real.cos (2 * A)) / (1 - Real.cos (2 * B)) +
  (1 - Real.cos (2 * B)) * (1 - Real.cos (2 * C)) / (1 - Real.cos (2 * A)) ≥ 9 / 2 :=
sorry

end non_obtuse_triangle_inequality_l754_754052


namespace arrange_classes_l754_754749

def arrangements (classes: List String) (constraint: (String → String → Prop)): ℕ :=
  (Finset.univ : Finset (Listperm classes)).filter (fun l => constraint "Math" "History").card

def constraint (m h : String) := m = "Math" → h = "History"

def five_classes : List String := ["Chinese", "Math", "Physics", "History", "Foreign Language"]

theorem arrange_classes :
  arrangements five_classes constraint = 60 :=
by
  sorry

end arrange_classes_l754_754749


namespace geom_seq_m_equals_11_l754_754955

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ (n : ℕ), a n = a1 * q ^ n

theorem geom_seq_m_equals_11 {a : ℕ → ℝ} {q : ℝ} (hq : q ≠ 1) 
  (h : geometric_sequence a 1 q) : 
  a 11 = a 1 * a 2 * a 3 * a 4 * a 5 := 
by sorry

end geom_seq_m_equals_11_l754_754955


namespace find_x_values_for_3001_l754_754329

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l754_754329


namespace sum_of_squares_of_real_solutions_l754_754375

theorem sum_of_squares_of_real_solutions :
  (∑ x in ({x : ℝ | x ^ 256 = 256 ^ 32}), x^2) = 8 :=
by
  sorry

end sum_of_squares_of_real_solutions_l754_754375


namespace trisha_take_home_pay_l754_754653

def hourly_wage : ℝ := 15
def hours_per_week : ℝ := 40
def weeks_per_year : ℝ := 52
def tax_rate : ℝ := 0.2

def annual_gross_pay : ℝ := hourly_wage * hours_per_week * weeks_per_year
def amount_withheld : ℝ := tax_rate * annual_gross_pay
def annual_take_home_pay : ℝ := annual_gross_pay - amount_withheld

theorem trisha_take_home_pay :
  annual_take_home_pay = 24960 := 
by
  sorry

end trisha_take_home_pay_l754_754653


namespace average_length_remaining_strings_l754_754210

theorem average_length_remaining_strings 
  (T1 : ℕ := 6) (avg_length1 : ℕ := 80) 
  (T2 : ℕ := 2) (avg_length2 : ℕ := 70) :
  (6 * avg_length1 - 2 * avg_length2) / 4 = 85 := 
by
  sorry

end average_length_remaining_strings_l754_754210


namespace circle_diameter_diameter_of_circle_with_given_area_l754_754672

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) (pi : ℝ) (h1 : A = 225 * pi) (h2 : A = pi * r^2) : d = 2 * r := by
  sorry

theorem diameter_of_circle_with_given_area (h : A = 225 * pi) : d = 30 := by
  have r : ℝ := 15
  have d : ℝ := 2 * r
  exact calc
    d = 2 * 15 := by sorry

end circle_diameter_diameter_of_circle_with_given_area_l754_754672


namespace calculate_expression_l754_754263

theorem calculate_expression : (3.65 - 1.25) * 2 = 4.80 := 
by 
  sorry

end calculate_expression_l754_754263


namespace number_ordering_l754_754671

theorem number_ordering : (10^5 < 2^20) ∧ (2^20 < 5^10) :=
by {
  -- We place the proof steps here
  sorry
}

end number_ordering_l754_754671


namespace p_12_eq_neg3_l754_754260

noncomputable def p (x : ℝ) : ℝ := ax^2 + bx + c

theorem p_12_eq_neg3 (a b c : ℝ) (h_sym : ∀ k : ℝ, p(a, b, c) (6 + k) = p(a, b, c) (6 - k))
(h_pass : p(a, b, c) 0 = -3) : p(a, b, c) 12 = -3 :=
by
  sorry

end p_12_eq_neg3_l754_754260


namespace bf_bisects_area_l754_754417

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions for points and their vectors in space
variables (A B C D M E F : V)

-- Given conditions as per the problem
def midpoint (A C M : V) : Prop := (M = (A + C) / 2)
def parallel (u v : V) : Prop := ∃ t : ℝ, u = t • v

-- Formal statement of the theorem
theorem bf_bisects_area (h1 : midpoint A C M) 
                        (h2 : parallel (E - F) (B - D)) 
                        (h3 : ∃ E F, line_through M E ∧ line_through M F ∧ (line_through E C) ∧ (line_through F D)) :
  2 * (triangle_area B C F) = quadrilateral_area A B C D := 
sorry

end bf_bisects_area_l754_754417


namespace sum_of_multiples_of_4_between_34_and_135_l754_754693

theorem sum_of_multiples_of_4_between_34_and_135 :
  let first := 36
  let last := 132
  let n := (last - first) / 4 + 1
  let sum := n * (first + last) / 2
  sum = 2100 := 
by
  sorry

end sum_of_multiples_of_4_between_34_and_135_l754_754693


namespace smallest_value_y_l754_754781

theorem smallest_value_y (y : ℝ) : (|y - 8| = 15) → y = -7 :=
by
  sorry

end smallest_value_y_l754_754781


namespace complex_conjugate_difference_l754_754824

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754824


namespace find_x_l754_754106

-- Define the vectors and collinearity condition
def vector_a : ℝ × ℝ := (3, 6)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 8)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (b.1 = k * a.1) ∧ (b.2 = k * a.2)

-- Define the proof problem
theorem find_x (x : ℝ) (h : collinear vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l754_754106


namespace solve_ab_cd_l754_754035

theorem solve_ab_cd (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -2) 
  (h3 : a + c + d = 5) 
  (h4 : b + c + d = 4) 
  : a * b + c * d = 26 / 9 := 
by {
  sorry
}

end solve_ab_cd_l754_754035


namespace solve_equation_l754_754792

theorem solve_equation (x : ℝ) :
  (10 / (sqrt (x - 10) - 10) + 
   2 / (sqrt (x - 10) - 5) + 
   14 / (sqrt (x - 10) + 5) + 
   20 / (sqrt (x - 10) + 10) = 0) →
  (x = 190 / 9 ∨ x = 5060 / 256) :=
sorry

end solve_equation_l754_754792


namespace scientific_notation_40_9_billion_l754_754499

theorem scientific_notation_40_9_billion :
  (40.9 * 10^9) = 4.09 * 10^9 :=
by
  sorry

end scientific_notation_40_9_billion_l754_754499


namespace seashells_in_jar_at_end_of_month_l754_754786

noncomputable def seashells_in_week (initial: ℕ) (increment: ℕ) (week: ℕ) : ℕ :=
  initial + increment * week

theorem seashells_in_jar_at_end_of_month :
  seashells_in_week 50 20 0 +
  seashells_in_week 50 20 1 +
  seashells_in_week 50 20 2 +
  seashells_in_week 50 20 3 = 320 :=
sorry

end seashells_in_jar_at_end_of_month_l754_754786


namespace four_possible_x_values_l754_754298

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l754_754298


namespace cos_C_in_triangle_l754_754524

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end cos_C_in_triangle_l754_754524


namespace angle_between_a_b_is_45_degrees_find_vector_c_l754_754809

-- Definitions for vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 6)

-- Proof statement for calculating and verifying the angle θ
theorem angle_between_a_b_is_45_degrees : 
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  let magnitude_a := real.sqrt (a.1^2 + a.2^2) in
  let magnitude_b := real.sqrt (b.1^2 + b.2^2) in
  let cos_theta := dot_product / (magnitude_a * magnitude_b) in
  real.arccos cos_theta * 180 / real.pi = 45 :=
by sorry

-- Definition and proof statement for vector c being collinear with b,
-- and perpendicularity condition.
def collinear_with_b (c : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, c = (λ * b.1, λ * b.2)

def perpendicular_diff_with_a (c : ℝ × ℝ) : Prop :=
  let diff := (a.1 - c.1, a.2 - c.2) in
  diff.1 * a.1 + diff.2 * a.2 = 0

theorem find_vector_c :
  ∃ c : ℝ × ℝ, collinear_with_b c ∧ perpendicular_diff_with_a c ∧ c = (-1, 3) :=
by sorry

end angle_between_a_b_is_45_degrees_find_vector_c_l754_754809


namespace seeds_in_buckets_l754_754182

-- Definition of seeds in each bucket
variables (A B C D : ℕ)

-- Conditions
def condition1 : Prop := A = B + 10
def condition2 : Prop := B = 30
def condition3 : Prop := A + B + C + D = 250
def condition4 : Prop := D = 2 * C

-- Conclusion
def conclusion : Prop := A = 40 ∧ B = 30 ∧ C = 60 ∧ D = 120

-- The final theorem statement
theorem seeds_in_buckets :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 → conclusion :=
by
  sorry

end seeds_in_buckets_l754_754182


namespace students_with_same_number_of_friends_l754_754593

theorem students_with_same_number_of_friends (n : ℕ) (knows : ℕ → ℕ → Prop)
  (reciprocal : ∀ (a b : ℕ), knows a b ↔ knows b a) :
  ∃ (a b : ℕ), a ≠ b ∧ (∑ i in (finset.range n), ite (knows a i) 1 0 = ∑ i in (finset.range n), ite (knows b i) 1 0) :=
by
  sorry

end students_with_same_number_of_friends_l754_754593


namespace max_f_value_l754_754405

noncomputable def f (x y z : ℝ) : ℝ := 
  let term := λ (x y z : ℝ), x * (2 * y - z) / (1 + x + 3 * y)
  term x y z + term y z x + term z x y

theorem max_f_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  f x y z ≤ 1 / 7 :=
begin
  sorry
end

end max_f_value_l754_754405


namespace cost_of_article_l754_754687

variable (C : ℝ) 
variable (G : ℝ)
variable (H1 : G = 380 - C)
variable (H2 : 1.05 * G = 420 - C)

theorem cost_of_article : C = 420 :=
by
  sorry

end cost_of_article_l754_754687


namespace plaster_cost_correct_l754_754685

def length : ℝ := 25
def width : ℝ := 12
def depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.30

def area_longer_walls : ℝ := 2 * (length * depth)
def area_shorter_walls : ℝ := 2 * (width * depth)
def area_bottom : ℝ := length * width
def total_area : ℝ := area_longer_walls + area_shorter_walls + area_bottom

def calculated_cost : ℝ := total_area * cost_per_sq_meter
def correct_cost : ℝ := 223.2

theorem plaster_cost_correct : calculated_cost = correct_cost := by
  sorry

end plaster_cost_correct_l754_754685


namespace main_theorem_l754_754735

-- Definitions based on conditions
variables (A P H M E C : ℕ) 
-- Thickness of an algebra book
def x := 1
-- Thickness of a history book (twice that of algebra)
def history_thickness := 2 * x
-- Length of shelf filled by books
def z := A * x

-- Condition equations based on shelf length equivalences
def equation1 := A = P
def equation2 := 2 * H * x = M * x
def equation3 := E * x + C * history_thickness = z

-- Prove the relationship
theorem main_theorem : C = (M * (A - E)) / (2 * A * H) :=
by
  sorry

end main_theorem_l754_754735


namespace num_x_for_3001_in_sequence_l754_754305

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l754_754305


namespace volume_tetrahedron_l754_754055

open Real

variable (R : ℝ) (α : ℝ) (φ: ℝ)

theorem volume_tetrahedron (hR : R > 0) (hα : 0 < α ∧ α < π / 2) (hφ : 0 < φ ∧ φ < π / 2) :
  if (α / 2 ≤ φ ∧ φ < π / 2 - α / 2) then
    let V := (2 / 3) * R^3 * tan (α / 2) in
    V = (2 / 3) * R^3 * tan (α / 2)
  else if (π / 2 - α / 2 ≤ φ ∧ φ < π / 2) then
    let V1 := (2 / 3) * R^3 * tan (α / 2) in 
    let V2 := (2 / 3) * R^3 * cot (α / 2) in
    (V1 = (2 / 3) * R^3 * tan (α / 2) ∨ V2 = (2 / 3) * R^3 * cot (α / 2))
  else false :=
by
  sorry

end volume_tetrahedron_l754_754055


namespace unique_peg_placement_l754_754770

theorem unique_peg_placement :
  ∃! arrangement : ℕ × ℕ × ℕ × ℕ × ℕ,
    (∃ (yellow_positions : fin 7 → fin 7) 
       (red_positions : fin 5 → fin 5)
       (green_positions : fin 4 → fin 4)
       (blue_positions : fin 3 → fin 3)
       (orange_positions : fin 2 → fin 2),
      (∀ i j, i ≠ j → yellow_positions i ≠ yellow_positions j ∧
                         red_positions i ≠ red_positions j ∧
                         green_positions i ≠ green_positions j ∧
                         blue_positions i ≠ blue_positions j ∧
                         orange_positions i ≠ orange_positions j ∧
                         distinct_rows_columns_diagonals yellow_positions red_positions green_positions blue_positions orange_positions)) :=
begin
  sorry
end

def distinct_rows_columns_diagonals
  (yellow_positions : fin 7 → fin 7)
  (red_positions : fin 5 → fin 5)
  (green_positions : fin 4 → fin 4)
  (blue_positions : fin 3 → fin 3)
  (orange_positions : fin 2 → fin 2) : Prop :=
∀ (i j : ℕ) (i_colored j_colored : ℕ),
  if i_colored = j_colored 
  then ∀ (color_positions : fin i_colored → fin i_colored), color_positions i ≠ color_positions j ∨ color_positions i ≠ (i + j)%nat ∨ color_positions i ≠ (i - j)%nat
  else true


end unique_peg_placement_l754_754770


namespace simplify_fraction_l754_754157

theorem simplify_fraction (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a ^ (-6) - b ^ (-6)) / (a ^ (-3) - b ^ (-3)) = a ^ (-1) + b ^ (-1) :=
by sorry

end simplify_fraction_l754_754157


namespace no_common_points_l754_754776

theorem no_common_points : 
  ∀ (x y : ℝ), ¬(x^2 + y^2 = 9 ∧ x^2 + y^2 = 4) := 
by
  sorry

end no_common_points_l754_754776


namespace matrix_result_l754_754092

open Matrix

noncomputable def M : Matrix ℝ 2 2 := sorry

def v1 : Vector ℝ 2 := ⟨[1, -2], by simp⟩
def v2 : Vector ℝ 2 := ⟨[-4, 6], by simp⟩
def v3 : Vector ℝ 2 := ⟨[7, -1], by simp⟩
def w1 : Vector ℝ 2 := ⟨[2, 1], by simp⟩
def w2 : Vector ℝ 2 := ⟨[0, -2], by simp⟩
def w3 : Vector ℝ 2 := ⟨[-38, -6], by simp⟩

theorem matrix_result :
  (M.mulVec v1 = w1) ∧ (M.mulVec v2 = w2) → (M.mulVec v3 = w3) :=
by
  sorry

end matrix_result_l754_754092


namespace average_headcount_correct_l754_754662

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end average_headcount_correct_l754_754662


namespace polynomial_n_values_possible_num_values_of_n_l754_754627

theorem polynomial_n_values_possible :
  ∃ (n : ℤ), 
    (∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → x > 0) ∧
    (∃ a : ℤ, a > 0 ∧ ∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → 
      x = a ∨ x = a / 4 + r ∨ x = a / 4 - r) ∧
    1 ≤ r^2 ∧ r^2 ≤ 4090499 :=
sorry

theorem num_values_of_n : 
  ∃ (n_values : ℤ), n_values = 4088474 :=
sorry

end polynomial_n_values_possible_num_values_of_n_l754_754627


namespace necessary_but_not_sufficient_condition_geometric_sequence_l754_754152

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * a (n - 1) / a n 

theorem necessary_but_not_sufficient_condition_geometric_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  (is_geometric_sequence a → (∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2)) ∧ (∃ b : ℕ → ℝ, (b n = 0 ∨ b n = b (n - 1) ∨ b n = b (n + 1)) ∧ ¬ is_geometric_sequence b) := 
sorry

end necessary_but_not_sufficient_condition_geometric_sequence_l754_754152


namespace sum_cubed_identity_l754_754142

theorem sum_cubed_identity
  (p q r : ℝ)
  (h1 : p + q + r = 5)
  (h2 : pq + pr + qr = 7)
  (h3 : pqr = -10) :
  p^3 + q^3 + r^3 = -10 := 
by
  sorry

end sum_cubed_identity_l754_754142


namespace problem_solution_l754_754908

/-- Definition of the points A, B, and C as elements of ℝ^2 (plane) -/
variables (A B C : ℝ × ℝ)

/-- Given conditions: distances AB = 3, BC = 4, CA = 5 -/
def dist_Ab : ℝ := 3
def dist_Bc : ℝ := 4
def dist_Ca : ℝ := 5

/-- The main theorem to prove. -/
theorem problem_solution :
  let AB := dist_Ab in
  let BC := dist_Bc in
  let CA := dist_Ca in
  AB * (BC + (BC * (CA + (CA * AB)))) = -25 :=
by
  sorry

end problem_solution_l754_754908


namespace cos_C_in_triangle_l754_754522

theorem cos_C_in_triangle
  (A B C : ℝ) (sin_A : ℝ) (cos_B : ℝ)
  (h1 : sin_A = 4 / 5)
  (h2 : cos_B = 12 / 13) :
  cos (π - A - B) = -16 / 65 :=
by
  -- Proof steps would be included here
  sorry

end cos_C_in_triangle_l754_754522


namespace player_b_wins_l754_754193

theorem player_b_wins : 
  ∃ B_strategy : (ℕ → ℕ → Prop), (∀ A_turn : ℕ → Prop, 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (A_turn i ↔ ¬ A_turn (i + 1))) → 
  ((B_strategy 1 2019) ∨ ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2019 ∧ B_strategy k (k + 1) ∧ ¬ A_turn k)) :=
sorry

end player_b_wins_l754_754193


namespace ages_proof_l754_754498

def hans_now : ℕ := 8

def sum_ages (annika_now emil_now frida_now : ℕ) :=
  hans_now + annika_now + emil_now + frida_now = 58

def annika_age_in_4_years (annika_now : ℕ) : ℕ :=
  3 * (hans_now + 4)

def emil_age_in_4_years (emil_now : ℕ) : ℕ :=
  2 * (hans_now + 4)

def frida_age_in_4_years (frida_now : ℕ) :=
  2 * 12

def annika_frida_age_difference (annika_now frida_now : ℕ) : Prop :=
  annika_now = frida_now + 5

theorem ages_proof :
  ∃ (annika_now emil_now frida_now : ℕ),
    sum_ages annika_now emil_now frida_now ∧
    annika_age_in_4_years annika_now = 36 ∧
    emil_age_in_4_years emil_now = 24 ∧
    frida_age_in_4_years frida_now = 24 ∧
    annika_frida_age_difference annika_now frida_now :=
by
  sorry

end ages_proof_l754_754498


namespace cost_of_soap_for_year_l754_754111

theorem cost_of_soap_for_year
  (months_per_bar cost_per_bar : ℕ)
  (months_in_year : ℕ)
  (h1 : months_per_bar = 2)
  (h2 : cost_per_bar = 8)
  (h3 : months_in_year = 12) :
  (months_in_year / months_per_bar) * cost_per_bar = 48 := by
  sorry

end cost_of_soap_for_year_l754_754111


namespace count_x_values_l754_754315

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l754_754315


namespace cracker_distribution_l754_754113

theorem cracker_distribution (total_crackers : ℕ) (friends : ℕ) (h1 : total_crackers = 22) (h2 : friends = 11) : (total_crackers / friends = 2) :=
by
  rw [h1, h2]
  norm_num
  sorry

end cracker_distribution_l754_754113


namespace triangles_from_nonagon_l754_754291

-- Definitions based on the problem conditions
def nonagon_vertices : ℕ := 9

-- Main theorem statement translating the problem
theorem triangles_from_nonagon :
  nat.choose nonagon_vertices 3 = 84 :=
by sorry

end triangles_from_nonagon_l754_754291


namespace z_conjugate_difference_l754_754857

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754857


namespace max_value_vector_sum_proof_l754_754005

noncomputable def max_value_vector_sum (A B C : ℝ × ℝ) (O : ℝ × ℝ) (M : ℝ × ℝ) : ℝ :=
  let distance_CM := real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2)
  let sum_vectors := real.sqrt (A.1^2 + A.2^2) + real.sqrt (B.1^2 + B.2^2) + real.sqrt ((M.1 + 1)^2 + (M.2 + 1)^2)
  if (distance_CM = 1) then sum_vectors else 0

theorem max_value_vector_sum_proof (A B C : ℝ × ℝ) (O : ℝ × ℝ) (M : ℝ × ℝ)
  (hA : A = (0, 1)) (hB : B = (1, 0)) (hC : C = (0, -2)) (hO : O = (0, 0))
  (hM : real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) = 1) :
  max_value_vector_sum A B C O M = real.sqrt 2 + 1 :=
by {
  sorry
}

end max_value_vector_sum_proof_l754_754005


namespace find_a_and_max_value_l754_754427

noncomputable def f (x a : ℝ) := 2 * x^3 - 6 * x^2 + a

theorem find_a_and_max_value :
  (∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≥ 0) ∧ (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≤ 3)) :=
by
  sorry

end find_a_and_max_value_l754_754427


namespace intersection_A_B_l754_754472

-- Definitions for sets A and B
def A : set ℝ := { x | x - 2 < 0 }
def B : set ℝ := { x | 2^x > 1 }

-- Statement of the proof problem
theorem intersection_A_B : (A ∩ B) = { x | 0 < x ∧ x < 2 } :=
by {
  sorry
}

end intersection_A_B_l754_754472


namespace distinct_solutions_diff_l754_754994

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l754_754994


namespace cos_alpha_relation_complementary_angles_l754_754615

theorem cos_alpha_relation (α x : ℝ) (h_edge_angle_eq : ∀ β γ : ℝ, β = γ = α) (h_face_angle : ∀ θ : ℝ, θ = x) :
  cos x = cos α / (1 + cos α) :=
sorry

theorem complementary_angles (α x: ℝ) (h : α + x = π / 2) :
  α = 27.96875 ∧ x = 62.03125 :=
sorry

end cos_alpha_relation_complementary_angles_l754_754615


namespace S_n_formula_max_S_n_at_6_or_7_largest_n_S_n_positive_l754_754003

section arithmetic_sequence

variable (n : ℕ)
variable (a1 : ℕ := 12)
variable (d : ℤ := -2)

def S_n : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem S_n_formula : S_n n = -↑n * n + 13 * n := by
  -- S_n = 12n + (n * (n - 1) / 2) * -2 = -n^2 + 13n
  sorry

theorem max_S_n_at_6_or_7 : (S_n 6 = 42) ∧ (S_n 7 = 42) := by
  -- From S_n = -n^2 + 13n, maximum value occurs when n = 6 or n = 7
  sorry

theorem largest_n_S_n_positive : 12 = max { n : ℕ | S_n n > 0 } := by
  -- -n^2 + 13n > 0 ⇔ 0 < n < 13, maximum n = 12
  sorry

end arithmetic_sequence

end S_n_formula_max_S_n_at_6_or_7_largest_n_S_n_positive_l754_754003


namespace count_x_values_l754_754317

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l754_754317


namespace fourth_number_in_10th_row_l754_754619

theorem fourth_number_in_10th_row : 
  (n : ℕ) (h₁ : n = 10)
  (h₂ : ∀ k, (1 ≤ k ≤ 7) → k + 7 * (n - 1) = ((n - 1) * 7 + k)) :
  (show (n : ℕ) (3 + (7 * (n - 2)) = 67) by sorry) :=
begin
  sorry,
end

end fourth_number_in_10th_row_l754_754619


namespace repeating_decimal_sum_l754_754357

theorem repeating_decimal_sum : (0.\overline{3} : ℚ) + (0.\overline{04} : ℚ) + (0.\overline{005} : ℚ) = 1135 / 2997 := 
sorry

end repeating_decimal_sum_l754_754357


namespace common_area_of_overlapping_45_45_90_triangles_l754_754657

-- Define the hypotenuse length of the 45-45-90 triangles
def hypotenuse_length : ℝ := 10

-- Define the length of the legs based on the properties of the 45-45-90 triangle
def leg_length : ℝ := hypotenuse_length / Real.sqrt 2

-- Define the area of a 45-45-90 triangle
def triangle_area : ℝ := 0.5 * leg_length * leg_length

-- Hypothesis: The triangles are congruent, hypotenuses coincide, and we need to find the common area
theorem common_area_of_overlapping_45_45_90_triangles 
  (hypotenuse_eq : hypotenuse_length = 10)
  (leg_eq : leg_length = 10 / Real.sqrt 2) : 
  triangle_area = 25 :=
by
  rw [hypotenuse_eq, leg_eq]
  sorry

end common_area_of_overlapping_45_45_90_triangles_l754_754657


namespace dogs_who_eat_none_l754_754053

def total_dogs : ℕ := 150
def dogs_like_watermelon : ℕ := 30
def dogs_like_salmon : ℕ := 70
def dogs_like_chicken : ℕ := 15
def dogs_like_watermelon_and_salmon : ℕ := 10
def dogs_like_salmon_and_chicken : ℕ := 7
def dogs_like_watermelon_and_chicken : ℕ := 5
def dogs_like_all_three : ℕ := 3

theorem dogs_who_eat_none :
  let dogs_only_watermelon := dogs_like_watermelon - dogs_like_watermelon_and_salmon - dogs_like_watermelon_and_chicken + dogs_like_all_three in
  let dogs_only_salmon := dogs_like_salmon - dogs_like_watermelon_and_salmon - dogs_like_salmon_and_chicken + dogs_like_all_three in
  let dogs_only_chicken := dogs_like_chicken - dogs_like_watermelon_and_chicken - dogs_like_salmon_and_chicken + dogs_like_all_three in
  let dogs_w_and_s := dogs_like_watermelon_and_salmon - dogs_like_all_three in
  let dogs_s_and_c := dogs_like_salmon_and_chicken - dogs_like_all_three in
  let dogs_w_and_c := dogs_like_watermelon_and_chicken - dogs_like_all_three in
  let dogs_at_least_one := dogs_only_watermelon + dogs_only_salmon + dogs_only_chicken + dogs_w_and_s + dogs_s_and_c + dogs_w_and_c + dogs_like_all_three in
  (total_dogs - dogs_at_least_one) = 54 :=
by
  sorry

end dogs_who_eat_none_l754_754053


namespace exp_conjugate_sum_l754_754415

variable (α β : ℝ)

def exp_sum_condition : Prop :=
  exp (Complex.i * α) + exp (Complex.i * β) = Complex.ofReal (2 / 3) + Complex.mk 0 (5 / 8)

theorem exp_conjugate_sum (h : exp_sum_condition α β) :
  exp (-Complex.i * α) + exp (-Complex.i * β) = Complex.ofReal (2 / 3) - Complex.mk 0 (5 / 8) :=
sorry

end exp_conjugate_sum_l754_754415


namespace find_BD_l754_754511

theorem find_BD 
  (A B C D : Type)
  (AC BC : ℝ)
  (AD CD : ℝ)
  (AC_eq : AC = 10)
  (BC_eq : BC = 10)
  (AD_eq : AD = 12)
  (CD_eq : CD = 5) :
  ∃ (BD : ℝ), BD ≈ 6.79 :=
by {
  sorry
}

end find_BD_l754_754511


namespace zhou_catches_shuttle_probability_l754_754207

-- Condition 1: Shuttle arrival time and duration
def shuttle_arrival_start : ℕ := 420 -- 7:00 AM in minutes since midnight
def shuttle_duration : ℕ := 15

-- Condition 2: Zhou's random arrival time window
def zhou_arrival_start : ℕ := 410 -- 6:50 AM in minutes since midnight
def zhou_arrival_end : ℕ := 465 -- 7:45 AM in minutes since midnight

-- Total time available for Zhou to arrive (55 minutes) 
def total_time : ℕ := zhou_arrival_end - zhou_arrival_start

-- Time window when Zhou needs to arrive to catch the shuttle (15 minutes)
def successful_time : ℕ := shuttle_arrival_start + shuttle_duration - shuttle_arrival_start

-- Calculate the probability that Zhou catches the shuttle
theorem zhou_catches_shuttle_probability : 
  (successful_time : ℚ) / total_time = 3 / 11 := 
by 
  -- We don't need the actual proof steps, just the statement
  sorry

end zhou_catches_shuttle_probability_l754_754207


namespace parallelogram_area_is_correct_l754_754441

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 2, 3⟩
def B : Point3D := ⟨2, 5, 2⟩
def C : Point3D := ⟨-2, 3, 6⟩

noncomputable def vectorAB (A B : Point3D) : Point3D :=
  { x := B.x - A.x
  , y := B.y - A.y
  , z := B.z - A.z 
  }

noncomputable def vectorAC (A C : Point3D) : Point3D :=
  { x := C.x - A.x
  , y := C.y - A.y
  , z := C.z - A.z 
  }

noncomputable def dotProduct (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def sinAngle (u v : Point3D) : ℝ :=
  Real.sqrt (1 - (dotProduct u v / (magnitude u * magnitude v)) ^ 2)

noncomputable def parallelogramArea (u v : Point3D) : ℝ :=
  magnitude u * magnitude v * sinAngle u v

theorem parallelogram_area_is_correct :
  parallelogramArea (vectorAB A B) (vectorAC A C) = 6 * Real.sqrt 5 := by
  sorry

end parallelogram_area_is_correct_l754_754441


namespace numbers_divisible_by_4_or_7_l754_754452

theorem numbers_divisible_by_4_or_7 (n : ℕ) (h : n = 60) :
  let count_4 := 15 in
  let count_7 := 8 in
  let count_4_and_7 := 2 in
  count_4 + count_7 - count_4_and_7 = 21 := by
  -- Given conditions and calculations from the problem
  have h₁ : count_4 = 60 / 4 := by sorry
  have h₂ : count_7 = 60 / 7 := by sorry
  have h₃ : count_4_and_7 = 60 / 28 := by sorry
  -- Start the main proof
  sorry

end numbers_divisible_by_4_or_7_l754_754452


namespace radius_of_circle_l754_754473

theorem radius_of_circle (s θ : ℝ) (h1 : s = 4) (h2 : θ = 2) : ∃ r : ℝ, s = r * θ ∧ r = 2 :=
by
  -- We will define r and show that it satisfies the given conditions
  let r := s / θ
  use r
  split
  · rw [h1, h2]
    exact (by norm_num : 4 = r * θ)
  · unfold r
    rw [h1, h2]
    exact (by norm_num : r = 2)

end radius_of_circle_l754_754473


namespace cos_C_in_triangle_l754_754526

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end cos_C_in_triangle_l754_754526


namespace books_in_series_l754_754637

-- Define the number of movies
def M := 14

-- Define that the number of books is one more than the number of movies
def B := M + 1

-- Theorem statement to prove that the number of books is 15
theorem books_in_series : B = 15 :=
by
  sorry

end books_in_series_l754_754637


namespace correct_choice_l754_754745

variable (a b : ℝ) (p q : Prop) (x : ℝ)

-- Proposition A: Incorrect because x > 3 is a sufficient condition for x > 2.
def propositionA : Prop := (∀ x : ℝ, x > 3 → x > 2) ∧ ¬ (∀ x : ℝ, x > 2 → x > 3)

-- Proposition B: Incorrect negation form.
def propositionB : Prop := ¬ (¬p → ¬q) ∧ (q → p)

-- Proposition C: Incorrect because it should be 1/a > 1/b given 0 < a < b.
def propositionC : Prop := (a > 0 ∧ b < 0) ∧ ¬ (1/a < 1/b)

-- Proposition D: Correct negation form.
def propositionD_negation_correct : Prop := 
  (¬ ∃ x : ℝ, x^2 = 1) = ( ∀ x : ℝ, x^2 ≠ 1)

theorem correct_choice : propositionD_negation_correct := by
  sorry

end correct_choice_l754_754745


namespace order_of_points_on_inv_prop_function_l754_754470

def inv_proportional_function (x : ℝ) : ℝ := -2 / x

theorem order_of_points_on_inv_prop_function :
  let y1 := inv_proportional_function (-1)
  let y2 := inv_proportional_function 2
  let y3 := inv_proportional_function 3
  y1 > y3 ∧ y3 > y2 :=
by
  let y1 := inv_proportional_function (-1)
  let y2 := inv_proportional_function 2
  let y3 := inv_proportional_function 3
  sorry

end order_of_points_on_inv_prop_function_l754_754470


namespace complex_conj_difference_l754_754837

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754837


namespace arithmetic_identity_l754_754285

theorem arithmetic_identity : 45 * 27 + 73 * 45 = 4500 := by sorry

end arithmetic_identity_l754_754285


namespace find_a_l754_754986

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2 * x - 1 else 1 / x

theorem find_a (a : ℝ) (h : f a = -1 / 4) : a = -4 ∨ a = 3 / 8 :=
sorry

end find_a_l754_754986


namespace complex_conjugate_difference_l754_754822

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754822


namespace rick_friends_count_l754_754129

-- Definitions
def total_cards : ℕ := 130
def kept_cards : ℕ := 15
def miguel_cards : ℕ := 13
def cards_per_friend : ℕ := 12
def sisters : ℕ := 2
def cards_per_sister : ℕ := 3

-- Main theorem statement
theorem rick_friends_count :
  let remaining_after_keeping := total_cards - kept_cards,
      remaining_after_miguel := remaining_after_keeping - miguel_cards,
      total_sister_cards := sisters * cards_per_sister,
      remaining_for_friends := remaining_after_miguel - total_sister_cards,
      friends_count := remaining_for_friends / cards_per_friend
  in friends_count = 8 := sorry

end rick_friends_count_l754_754129


namespace triangle_median_inequality_l754_754102

variable {a b c m_a m_b m_c : ℝ}

-- Given conditions
axiom side_lengths_of_triangle (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
axiom medians_corresponding_to_sides (ma_def : m_a > 0) (mb_def : m_b > 0) (mc_def : m_c > 0)

-- The proof problem
theorem triangle_median_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (ma_def : m_a > 0) (mb_def : m_b > 0) (mc_def : m_c > 0) :
  m_a * ((b / a - 1) * (c / a - 1)) + 
  m_b * ((a / b - 1) * (c / b - 1)) + 
  m_c * ((a / c - 1) * (b / c - 1)) ≥ 0 := 
by 
  sorry

end triangle_median_inequality_l754_754102


namespace sum_of_elements_in_T_l754_754095

noncomputable def digit_sum : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 504
noncomputable def repeating_sum : ℕ := digit_sum * 1111
noncomputable def sum_T : ℚ := repeating_sum / 9999

theorem sum_of_elements_in_T : sum_T = 2523 := by
  sorry

end sum_of_elements_in_T_l754_754095


namespace walkway_stopped_time_l754_754722

theorem walkway_stopped_time
  (v w : ℝ)
  (h1 : v + w = 2)
  (h2 : v - w = 2 / 3) :
  (100 / v) = 75 :=
by
  -- extracted from the given conditions
  have h3 : 2 * v = 8 / 3,
  { calc
    (v + w) + (v - w) = 2 + 2 / 3       : by rw [h1, h2]
                   ... = 6 / 3 + 2 / 3  : by norm_num
                   ... = 8 / 3          : by norm_num },

  have h4 : v = 4 / 3,
  { linarith },

  -- substituting the derived value of v into the final expression
  calc
  (100 / v) = 100 / (4 / 3) : by rw h4
         ... = 100 * (3 / 4) : by norm_num
         ... = 75           : by norm_num

end walkway_stopped_time_l754_754722


namespace binomial_19_10_l754_754287

theorem binomial_19_10 :
  ∀ (binom : ℕ → ℕ → ℕ),
  binom 17 7 = 19448 → binom 17 9 = 24310 →
  binom 19 10 = 92378 :=
by
  intros
  sorry

end binomial_19_10_l754_754287


namespace vasya_kolya_difference_impossible_l754_754086

theorem vasya_kolya_difference_impossible : 
  ∀ k v : ℕ, (∃ q₁ q₂ : ℕ, 14400 = q₁ * 2 + q₂ * 2 + 1 + 1) → ¬ ∃ k, ∃ v, (v - k = 11 ∧ 14400 = k * q₁ + v * q₂) :=
by sorry

end vasya_kolya_difference_impossible_l754_754086


namespace smaller_rectangle_perimeter_l754_754241

theorem smaller_rectangle_perimeter (P : ℝ) (n m : ℕ) (A : ℕ) (L : ℝ) 
    (total_rect_perimeter : P = 100) 
    (cuts_count : n = 6 ∧ m = 9)
    (total_cuts_length : L = 405)
    (total_smaller_rects : A = 70) :
    (∀ height width, P = 2 * (height + width) → ∃ height width, 2 * (5 + 1.5) = 13) :=
by 
  sorry

end smaller_rectangle_perimeter_l754_754241


namespace greatest_value_of_sum_l754_754457

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + 1 / x^2) : 
  ∃ y, y = x + 1/x ∧ y ≤ sqrt 15 :=
begin
  -- The proof will be here.
  sorry
end

end greatest_value_of_sum_l754_754457


namespace infinite_real_numbers_for_real_sqrt_l754_754390

theorem infinite_real_numbers_for_real_sqrt :
  set.infinite {x : ℝ | 2 - (x - 3)^2 ≥ 0} :=
sorry

end infinite_real_numbers_for_real_sqrt_l754_754390


namespace total_number_of_toys_l754_754536

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l754_754536


namespace tan_double_angle_l754_754814

theorem tan_double_angle (θ : ℝ) (h1 : cos θ = -3/5) (h2 : 0 < θ ∧ θ < π) : tan (2 * θ) = 24/7 := by
  sorry

end tan_double_angle_l754_754814


namespace distinct_solutions_diff_l754_754995

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l754_754995


namespace calculate_amount_after_two_years_l754_754790

noncomputable def amount_after_years (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + rate) ^ years

theorem calculate_amount_after_two_years :
  amount_after_years 51200 0.125 2 = 64800 :=
by
  sorry

end calculate_amount_after_two_years_l754_754790


namespace construct_n_gon_l754_754217

variables {M : Type} [metric_space M] (n : ℕ) {l : fin (n-1) → set M}

-- Definition of midpoint coinciding with point M
def midpoint_condition (A1 A2 : M) (midpoint : M) : Prop :=
  midpoint = M

-- Definition of perpendicular bisector condition
def perp_bisector_condition (A : fin n → M) (l : fin (n-1) → set M): Prop :=
  ∀ i : fin (n-1), exists P Q : M, P ≠ Q ∧ (orthogonal_projection ↑l i) (segment P Q) ⊆ l i

-- Definition of angle and bisector condition at A1
def angle_bisector_at_A1 (A1 A2 : M) (alpha : Real) (M : M) : Prop :=
  ∠A1 M A2 = alpha ∧ midpoint_condition A1 A2 M

-- Definition of angle bisectors coinciding with lines
def angle_bisectors_condition (A : fin n → M) (l : fin (n-1) → set M) : Prop :=
  ∀ i : fin (n-1), angle_bisector (A i) (A i.succ) (A i.succ.succ) M ⊆ l i 

theorem construct_n_gon (n : ℕ) (M : M) (l : fin (n-1) → set M) (alpha : Real) :
  (∃ A : fin n → M,
    midpoint_condition (A 0) (A 1) M ∧
    perp_bisector_condition A l ∧
    angle_bisector_at_A1 (A 0) (A 1) alpha M ∧
    angle_bisectors_condition A l) ↔ 
  (if n % 2 = 1 
    then True -- odd: construction valid and unique
    else False) := -- even: no consistent solution except special cases
sorry

end construct_n_gon_l754_754217


namespace find_n_l754_754769

theorem find_n (a : ℕ → ℝ) (d : ℝ) (n : ℕ) (S_odd : ℝ) (S_even : ℝ)
  (h1 : ∀ k, a (2 * k - 1) = a 0 + (2 * k - 2) * d)
  (h2 : ∀ k, a (2 * k) = a 1 + (2 * k - 1) * d)
  (h3 : 2 * n + 1 = n + (n + 1))
  (h4 : S_odd = (n + 1) * (a 0 + n * d))
  (h5 : S_even = n * (a 1 + (n - 1) * d))
  (h6 : S_odd = 4)
  (h7 : S_even = 3) : n = 3 :=
by
  sorry

end find_n_l754_754769


namespace selection_ways_with_boys_and_girls_l754_754133

-- Define the problem conditions
def boys : ℕ := 5
def girls : ℕ := 3
def total_students : ℕ := boys + girls
def group_size : ℕ := 3

-- The theorem stating the problem
theorem selection_ways_with_boys_and_girls :
  ∃ n, n = 45 ∧ ∀ (chosen_boys : ℕ) (chosen_girls : ℕ), 
  chosen_boys + chosen_girls = group_size → 
  0 < chosen_boys → 0 < chosen_girls →
  ∃ (ways : ℕ), ways = (nat.choose boys chosen_boys) * (nat.choose girls chosen_girls) := 
begin
  sorry
end

end selection_ways_with_boys_and_girls_l754_754133


namespace no_valid_circle_permutation_l754_754969

theorem no_valid_circle_permutation :
  ∀ (a : Fin 2024 → ℕ), (∃ (b : Fin 2024 → ℕ), (∀ i : Fin 2024, (∃ j : Fin 2024, a i * a ((i + 1) % 2024) = b j!)) ∧ (Multiset.of_fn b = {1!, 2!, ..., 2024!})) → False :=
by
  intros a h
  sorry

end no_valid_circle_permutation_l754_754969


namespace projection_matrix_correct_l754_754796

def projection_matrix (u : ℝ × ℝ) : vector (vector ℝ 2) :=
  let denom := (u.1^2 + u.2^2 : ℝ) in
  let scalar := λ x y : ℝ, (x * y) / denom in
  (2 : ℕ)![ (2 : ℕ)![ scalar u.1 u.1, scalar u.1 u.2], 
            (2 : ℕ)![ scalar u.2 u.1, scalar u.2 u.2 ] ]

def expected_matrix : vector (vector ℝ 2) :=
  (2 : ℕ)![ (2 : ℕ)![9 / 25, -12 / 25], 
            (2 : ℕ)![ -12 / 25, 16 / 25] ]
            
theorem projection_matrix_correct : projection_matrix (3, -4) = expected_matrix := 
  sorry

end projection_matrix_correct_l754_754796


namespace range_of_k_l754_754396

theorem range_of_k {k : ℝ} (h1 : 2 ≤ ∫ x in 1..2, k * x + 1) (h2 : ∫ x in 1..2, k * x + 1 ≤ 4) :
  k ∈ set.Icc (2/3 : ℝ) 2 :=
by
  sorry

end range_of_k_l754_754396


namespace Q_divisible_by_Q₁_l754_754126

-- Definitions of the polynomials and natural numbers involved
def poly (R : Type) [CommRing R] := Polynomial R

-- Let P(x) be a polynomial
variable {R : Type} [CommRing R]
variable (P : poly R)
variable (x : R)

-- Condition: P(x) ≠ x
axiom P_ne_x : P.eval x ≠ x

-- Define Q₁(x) = P(x) - x
def Q₁ : poly R := P - Polynomial.C x

-- Recursive definition of Qₙ(x)
noncomputable def Q_n (n : ℕ) : poly R :=
  if h : n = 0 then 0
  else nat.cases_on n 0 (λ m, (Qₙ P m).comp P - Polynomial.C x)

-- Main theorem statement
theorem Q_divisible_by_Q₁ (n : ℕ) : Q_n P n ∣ Q₁ P :=
sorry

end Q_divisible_by_Q₁_l754_754126


namespace recurring_sum_fractions_l754_754353

theorem recurring_sum_fractions :
  let x := (1 / 3) in
  let y := (4 / 99) in
  let z := (5 / 999) in
  x + y + z = (742 / 999) :=
by 
  sorry

end recurring_sum_fractions_l754_754353


namespace average_headcount_l754_754665

theorem average_headcount 
  (h1 : ℕ := 11500) 
  (h2 : ℕ := 11600) 
  (h3 : ℕ := 11300) : 
  (Float.round ((h1 + h2 + h3 : ℕ : Float) / 3) = 11467) :=
sorry

end average_headcount_l754_754665


namespace sequence_x_values_3001_l754_754311

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l754_754311


namespace shaded_area_closest_to_l754_754066

-- Definitions based on the given conditions
variables {AB BC : ℝ} (hAB : AB = 12) (hBC : BC = 18)

-- Main theorem to prove the total shaded area assuming the given conditions.
theorem shaded_area_closest_to :
  let
    AE := AB,
    EX := (1/2) * AE,
    AX := math.sqrt(3) * EX,
    triangle_EXA_area := (1/2) * EX * AX,
    XB := AB - AX,
    rectangle_BXYC_area := XB * BC,
    EY := AE,
    HY := EY / math.sqrt(3),
    triangle_EYH_area := (1/2) * EY * HY,
    area_AEHCB := triangle_EXA_area + rectangle_BXYC_area + triangle_EYH_area,
    total_shaded_area := 2 * area_AEHCB
  in 
  |total_shaded_area - 203.4| < 0.1 :=
by
  sorry

end shaded_area_closest_to_l754_754066


namespace area_sum_triangle_l754_754045

theorem area_sum_triangle (ABC D E : Type) [Triangle ABC] [Midpoint D AC] [Midpoint E BC]
  (AC_is_2 : length AC = 2)
  (angle_BAC : angle BAC = 50) (angle_ABC : angle ABC = 90) (angle_ACB : angle ACB = 40)
  (angle_DEC : angle DEC = 70) :
  area ABC + 2 * area CDE = 3 * sin 50 :=
by
  sorry

end area_sum_triangle_l754_754045


namespace find_angle_l754_754178

noncomputable def angle_phi (a b d : ℝ^3) : set ℝ :=
  {φ | cos φ = (a • d) / ((‖a‖ * ‖d‖)) ∨ cos φ = - (a • d) / ((‖a‖ * ‖d‖))}

theorem find_angle
  (a b d : ℝ^3)
  (ha : ‖a‖ = real.sqrt 2)
  (hb : ‖b‖ = real.sqrt 2)
  (hd : ‖d‖ = 3)
  (h : a.cross (a.cross d) + b = 0) :
  angle_phi a b d = { real.arccos (real.sqrt 34 / 6), 180 - real.arccos (real.sqrt 34 / 6) } :=
sorry

end find_angle_l754_754178


namespace acute_triangle_cosine_identity_l754_754957

variable {A B C O H B1 C1 : ℝ}

theorem acute_triangle_cosine_identity 
  (h1 : ∠BAC < π / 2)
  (h2 : ∠ABC < π / 2)
  (h3 : ∠ACB < π / 2)
  (hAB_AC : AB > AC)
  (circumcenter_O : circumcenter ∆ABC = O)
  (orthocenter_H : orthocenter ∆ABC = H)
  (BH_int_AC : line BH ∩ line AC = {B1})
  (CH_int_AB : line CH ∩ line AB = {C1})
  (OH_parallel_B1C1 : parallel OH B1C1) :
  cos(2 * ∠ABC) + cos(2 * ∠ACB) + 1 = 0 := 
sorry

end acute_triangle_cosine_identity_l754_754957


namespace janet_percentage_of_snowballs_l754_754973

-- Define the number of snowballs made by Janet
def janet_snowballs : ℕ := 50

-- Define the number of snowballs made by Janet's brother
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage calculation function
def calculate_percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Proof statement
theorem janet_percentage_of_snowballs : calculate_percentage janet_snowballs total_snowballs = 25 := 
by
  sorry

end janet_percentage_of_snowballs_l754_754973


namespace selling_price_correct_l754_754718

noncomputable def cost_price : ℝ := 100
noncomputable def gain_percent : ℝ := 0.15
noncomputable def profit : ℝ := gain_percent * cost_price
noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 115 := by
  sorry

end selling_price_correct_l754_754718


namespace baker_made_cakes_l754_754755

theorem baker_made_cakes (sold_cakes left_cakes total_cakes : ℕ) (h1 : sold_cakes = 108) (h2 : left_cakes = 59) :
  total_cakes = sold_cakes + left_cakes → total_cakes = 167 := by
  intro h
  rw [h1, h2] at h
  exact h

-- The proof part is omitted since only the statement is required

end baker_made_cakes_l754_754755


namespace greatest_possible_value_l754_754458

theorem greatest_possible_value (x : ℝ) (h : 13 = x^2 + 1 / x^2) : x + 1 / x ≤ Real.sqrt 15 :=
begin
  sorry
end

end greatest_possible_value_l754_754458


namespace quadratic_root_conjugate_l754_754858

variable (a b c : ℝ) (x : ℂ)

theorem quadratic_root_conjugate
  (h1 : a ≠ 0)
  (h2 : a * (1 + complex.I)^2 + b * (1 + complex.I) + c = 0) :
  a * (1 - complex.I)^2 + b * (1 - complex.I) + c = 0 :=
sorry

end quadratic_root_conjugate_l754_754858


namespace problem_statement_l754_754930

/-- Define the properties of even and odd functions --/
def even (f : ℝ → ℝ) := ∀ x, f(x) = f(-x)
def odd (g : ℝ → ℝ) := ∀ x, g(x) = -g(-x)

/-- Given functions f and g with the specified properties --/
variables (f g : ℝ → ℝ)
axiom h1 : even f
axiom h2 : odd g
axiom h3 : ∀ x, f(x) + 2 * g(x) = Real.exp x

/-- The proof problem: Show that g(-1) < f(-2) < f(-3) --/
theorem problem_statement : g(-1) < f(-2) ∧ f(-2) < f(-3) :=
by
  sorry

end problem_statement_l754_754930


namespace difference_in_number_of_girls_and_boys_l754_754604

def ratio_boys_girls (b g : ℕ) : Prop := b * 3 = g * 2

def total_students (b g : ℕ) : Prop := b + g = 30

theorem difference_in_number_of_girls_and_boys
  (b g : ℕ)
  (h1 : ratio_boys_girls b g)
  (h2 : total_students b g) :
  g - b = 6 :=
sorry

end difference_in_number_of_girls_and_boys_l754_754604


namespace equal_distances_plane_count_l754_754584

noncomputable def number_of_planes_equalizing_distances (A B C P : Point) (h: P ∉ Plane(A, B, C)) : ℕ :=
  4  -- There are exactly 4 such planes as per the problem statement

theorem equal_distances_plane_count (A B C P : Point) (h: P ∉ Plane(A, B, C)) :
  number_of_planes_equalizing_distances A B C P h = 4 :=
sorry

end equal_distances_plane_count_l754_754584


namespace seashells_in_jar_at_end_of_month_l754_754787

noncomputable def seashells_in_week (initial: ℕ) (increment: ℕ) (week: ℕ) : ℕ :=
  initial + increment * week

theorem seashells_in_jar_at_end_of_month :
  seashells_in_week 50 20 0 +
  seashells_in_week 50 20 1 +
  seashells_in_week 50 20 2 +
  seashells_in_week 50 20 3 = 320 :=
sorry

end seashells_in_jar_at_end_of_month_l754_754787


namespace sequence_a4_eq_neg3_l754_754016

theorem sequence_a4_eq_neg3 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 4 = -3 :=
by
  sorry

end sequence_a4_eq_neg3_l754_754016


namespace complex_conjugate_difference_l754_754823

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754823


namespace g_inv_undefined_at_one_l754_754924

def g (x : ℝ) : ℝ := (x - 2) / (x - 5)

noncomputable def g_inv (x : ℝ) : ℝ := (-5 * x + 2) / (1 - x)

theorem g_inv_undefined_at_one : ∀ x : ℝ, x = 1 → ¬(∃ y : ℝ, g_inv y = x) :=
by
  intros
  rw [g_inv]
  intro h
  simp at h
  sorry

end g_inv_undefined_at_one_l754_754924


namespace find_a_b_l754_754734

variable {a b : ℝ}

theorem find_a_b (h1 : {a, b / a, 1} = {a^2, a + b, 0}) :
  a^2019 + b^2018 = -1 :=
sorry

end find_a_b_l754_754734


namespace lean_proof_problem_l754_754768

open EuclideanGeometry

variables {A B C M O O_B O_C K L : Point} (ω : Circle)

theorem lean_proof_problem 
  (hM : PointOnSegment M B C)
  (hO_B : CenterOfCircle O_B (Triangle A B M))
  (hO_C : CenterOfCircle O_C (Triangle A C M))
  (hCircle : CircleThrough ω A M)
  (hCenterBC : CenterOnLine ω B C)
  (hIntersectK : MO_B = CirclesIntersect ω K (LinePassingThrough M O_B))
  (hIntersectL : MO_C = CirclesIntersect ω L (LinePassingThrough M O_C))
  (hReflection : Reflection A B C) :
  LineIntersection (Line B K) (Line C L) ∈ ω := sorry

end lean_proof_problem_l754_754768


namespace extended_triangle_area_seven_times_l754_754476

noncomputable theory
open Set

-- Define extension points for the triangle
variables {A B C : ℝ} -- assume simple real number coordinates for simplicity

-- Triangle area function (dummy placeholder, needs proper geometric implementation)
def triangle_area (A B C : ℝ) : ℝ := sorry

-- Define the extended points
def A1 := A + (B - A)
def B1 := B + (C - B)
def C1 := C + (A - C)

-- Lean theorem statement for the problem
theorem extended_triangle_area_seven_times (ABC_area : ℝ) :
  triangle_area A B C = ABC_area →
  triangle_area A1 B1 C1 = 7 * ABC_area :=
by
  intro h
  sorry

end extended_triangle_area_seven_times_l754_754476


namespace decreasing_interval_ln_sin_neg_2x_plus_pi_div_3_l754_754799

theorem decreasing_interval_ln_sin_neg_2x_plus_pi_div_3 (k : ℤ) :
    ∀ x, k * π - π / 12 ≤ x ∧ x < k * π + π / 6 ↔ 
    ∀ x, ∃ t, t = sin (2 * x - π / 3) ∧ t < 0 :=
by
  sorry

end decreasing_interval_ln_sin_neg_2x_plus_pi_div_3_l754_754799


namespace round_trip_ticket_percentage_l754_754121

theorem round_trip_ticket_percentage (p : ℕ → Prop) : 
  (∀ n, p n → n = 375) → (∀ n, p n → n = 375) :=
by
  sorry

end round_trip_ticket_percentage_l754_754121


namespace bc_money_l754_754744

variables (A B C : ℕ)

theorem bc_money (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : C = 50) : B + C = 150 :=
sorry

end bc_money_l754_754744


namespace minimal_abs_diff_l754_754034

theorem minimal_abs_diff (a b : ℤ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a * b - 3 * a + 7 * b = 222) : |a - b| = 54 :=
by
  sorry

end minimal_abs_diff_l754_754034


namespace recurring_sum_fractions_l754_754351

theorem recurring_sum_fractions :
  let x := (1 / 3) in
  let y := (4 / 99) in
  let z := (5 / 999) in
  x + y + z = (742 / 999) :=
by 
  sorry

end recurring_sum_fractions_l754_754351


namespace centroid_of_tetrahedron_l754_754077

theorem centroid_of_tetrahedron (A B C D G: Point) (h1: G is the centroid of triangle BCD):
  vector A G = 1 / 3 * (vector A B + vector A C + vector A D) :=
sorry

end centroid_of_tetrahedron_l754_754077


namespace parabola_equation_and_slope_sum_l754_754014

noncomputable theory

-- Given conditions
def parabola_passes_through (p m : ℝ) : Prop :=
  m ^ 2 = 2 * p

def directrix_intersects (p : ℝ) : Prop :=
  ∃ B : ℝ × ℝ, B = (-p/2, 0)

def distance_AB (p m : ℝ) : Prop :=
  real.sqrt ((1 + p/2)^2 + m^2) = 2 * real.sqrt 2

-- Proving the solution
theorem parabola_equation_and_slope_sum (p m : ℝ) :
  parabola_passes_through p m →
  directrix_intersects p →
  distance_AB p m →
  (y : ℝ) (H1 : y ^ 2 = 4) (x0 : ℝ) (k1 k2 k3 : ℝ) :
  x0 = 1 →
  ∀ P : ℝ × ℝ, P = (x0, 2) →
  (P.2 = 2 * real.sqrt x0) →
  (real.sqrt (x0) = P.1) →
  ∃ k1 k2 k3 : ℝ, -- assuming slopes exist and are non-zero
  k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 →
  1 / k1 + 1 / k2 - 1 / k3 = 1 :=
by sorry

end parabola_equation_and_slope_sum_l754_754014


namespace solve_quadratic_eq_l754_754175

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2024 * x ↔ x = 0 ∨ x = 2024 :=
by sorry

end solve_quadratic_eq_l754_754175


namespace domain_of_g_l754_754894

noncomputable def f : ℝ → ℝ := sorry

def dom_f : set ℝ := {x | -8 ≤ x ∧ x ≤ 1}

def g (x : ℝ) := f (2 * x + 1) / (x + 2)

theorem domain_of_g : ∀ x : ℝ, x ∈ (set.Icc (-9 / 2) 0 \ {-2}) ↔ x ∈ dom_f ∧ x + 2 ≠ 0 :=
by
  sorry

end domain_of_g_l754_754894


namespace X_is_N_l754_754544

theorem X_is_N (X : Set ℕ) (h_nonempty : ∃ x, x ∈ X)
  (h_condition1 : ∀ x ∈ X, 4 * x ∈ X)
  (h_condition2 : ∀ x ∈ X, Nat.floor (Real.sqrt x) ∈ X) : 
  X = Set.univ := 
sorry

end X_is_N_l754_754544


namespace find_alpha_l754_754163

def point (α : ℝ) : Prop := 3^α = Real.sqrt 3

theorem find_alpha (α : ℝ) (h : point α) : α = 1/2 := 
by 
  sorry

end find_alpha_l754_754163


namespace walnuts_count_l754_754051

def nuts_problem (p a c w : ℕ) : Prop :=
  p + a + c + w = 150 ∧
  a = p / 2 ∧
  c = 4 * a ∧
  w = 3 * c

theorem walnuts_count (p a c w : ℕ) (h : nuts_problem p a c w) : w = 96 :=
by sorry

end walnuts_count_l754_754051


namespace count_x_values_l754_754318

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l754_754318


namespace circumcircle_tangent_incircle_l754_754411

theorem circumcircle_tangent_incircle
  (A B C D E F K L : Type)
  [triangle A B C]
  (incircle_tangent_BC : tangent incircle ABC BC D)
  (incircle_tangent_CA : tangent incircle ABC CA E)
  (incircle_tangent_AB : tangent incircle ABC AB F)
  (K_on_CA : on K CA)
  (L_on_AB : on L AB)
  (not_eq_KA : K ≠ A)
  (not_eq_LA : L ≠ A)
  (angle_EDK_EQ_ADK : angle E D K = angle A D E)
  (angle_FDL_EQ_ADF : angle F D L = angle A D F) :
  tangent (circumcircle AKL) (incircle ABC) D := 
sorry

end circumcircle_tangent_incircle_l754_754411


namespace curve_line_intersection_l754_754407

variable {t a : ℝ}

/-- Condition 1: The curve C has polar coordinate equation ρ^2 cos 2θ = a^2 -/
def polar_eqn (ρ θ : ℝ) : Prop := ρ^2 * Real.cos(2 * θ) = a^2

/-- Condition 2: The line l passes through point P(2,1) with parameter equation -/
def line_eqn (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) : Prop :=
  P = (2, 1) ∧ ∀ t, l t = (2 + (Real.sqrt 3) / 2 * t, 1 + t / 2)

/-- Condition 3: Line l intersects curve C at points A and B with |PA||PB| = 2 -/
def intersection_and_product (A B P : ℝ × ℝ) : Prop :=
  A ≠ B ∧ P ≠ A ∧ P ≠ B ∧
  (A, B, P).2 = A ∧ (A, B, P).2 = B ∧ 
  (Real.dist P A) * (Real.dist P B) = 2

/-- Main theorem statement -/
theorem curve_line_intersection 
  (P : ℝ × ℝ) (ρ θ : ℝ) (l : ℝ → ℝ × ℝ)
  (A B : ℝ × ℝ) :
  polar_eqn ρ θ ∧
  line_eqn P l ∧
  intersection_and_product A B P →
  ((∃ a, a^2 = 4) ∧ (∃ a, ∥pA a - ∥ = 4 * Real.sqrt 3 - 2)) :=
sorry

end curve_line_intersection_l754_754407


namespace reflected_rectangle_has_no_point_neg_3_4_l754_754636

structure Point where
  x : ℤ
  y : ℤ
  deriving DecidableEq, Repr

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def is_not_vertex (pts: List Point) (p: Point) : Prop :=
  ¬ (p ∈ pts)

theorem reflected_rectangle_has_no_point_neg_3_4 :
  let initial_pts := [ Point.mk 1 3, Point.mk 1 1, Point.mk 4 1, Point.mk 4 3 ]
  let reflected_pts := initial_pts.map reflect_y
  is_not_vertex reflected_pts (Point.mk (-3) 4) :=
by
  sorry

end reflected_rectangle_has_no_point_neg_3_4_l754_754636


namespace recurring_decimals_sum_l754_754364

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end recurring_decimals_sum_l754_754364


namespace total_cans_given_away_l754_754273

-- Define constants
def initial_stock : ℕ := 2000

-- Define conditions day 1
def people_day1 : ℕ := 500
def cans_per_person_day1 : ℕ := 1
def restock_day1 : ℕ := 1500

-- Define conditions day 2
def people_day2 : ℕ := 1000
def cans_per_person_day2 : ℕ := 2
def restock_day2 : ℕ := 3000

-- Define the question as a theorem
theorem total_cans_given_away : (people_day1 * cans_per_person_day1 + people_day2 * cans_per_person_day2) = 2500 := by
  sorry

end total_cans_given_away_l754_754273


namespace find_M_l754_754569

theorem find_M (side_length : ℝ) (surface_area_cube : ℝ)
  (surface_area_sphere : ℝ) (sphere_volume : ℝ)
  (h1 : side_length = 3) 
  (h2 : surface_area_cube = 6 * side_length^2) 
  (h3 : surface_area_cube = surface_area_sphere)
  (h4 : sphere_volume = (4/3) * π * (√(surface_area_sphere / (4 * π)))^3) :
  ∃ (M : ℝ), M = 54 ∧ sphere_volume = (M * √27) / √π :=
by
  sorry

end find_M_l754_754569


namespace time_for_second_train_l754_754189

section BulletTrains

-- Define the given conditions
def length_of_train : ℝ := 120  -- meters
def time_first_train : ℝ := 10  -- seconds
def time_to_cross : ℝ := 16.666666666666668  -- seconds
def speed_first_train : ℝ := length_of_train / time_first_train  -- meters/second
def distance_to_cross : ℝ := 2 * length_of_train  -- meters

-- Define the unknown time for the second bullet train
variable (T2 : ℝ)

-- Prove that T2 is 50 seconds
theorem time_for_second_train : 
  (speed_first_train + (length_of_train / T2)) * time_to_cross = distance_to_cross → 
  T2 = 50 :=
by
  -- We skip the proof itself
  sorry

end BulletTrains

end time_for_second_train_l754_754189


namespace probability_correct_l754_754179

noncomputable def probability_of_two_or_more_co_presidents : ℚ :=
  let p_club (n : ℕ) := 
    (Nat.choose 3 2 * Nat.choose (n - 3) 2 + Nat.choose 3 3 * Nat.choose (n - 3) 1) /
    Nat.choose n 4
  let p_10 := p_club 10
  let p_12 := p_club 12
  let p_15 := p_club 15
  (1 / 3) * (p_10 + p_12 + p_15)

theorem probability_correct : probability_of_two_or_more_co_presidents = 2 / 9 := by
  sorry

end probability_correct_l754_754179


namespace sufficient_but_not_necessary_condition_for_monotonicity_l754_754698

theorem sufficient_but_not_necessary_condition_for_monotonicity (a : ℝ) :
  (∀ x ∈ Icc (1:ℝ) 2, monotone (f a)) ↔ (a ≥ 2) ∨ (a ≤ 1) :=
sorry

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x + 3

-- Definitions for monotone and other relevant constructs should be marked imported. 
-- For example:
-- open_locale classical
-- open set topological_space 

end sufficient_but_not_necessary_condition_for_monotonicity_l754_754698


namespace determine_a_l754_754921

theorem determine_a (a x y : ℝ) (h : (a + 1) * x^(|a|) + y = -8) (h_linear : ∀ x y, (a + 1) * x^(|a|) + y = -8 → x ^ 1 = x): a = 1 :=
by 
  sorry

end determine_a_l754_754921


namespace find_y_l754_754870

theorem find_y (y : ℝ) (h : (y - 8) / (5 - (-3)) = -5 / 4) : y = -2 :=
by sorry

end find_y_l754_754870


namespace perimeter_of_remaining_quadrilateral_is_12_l754_754251

-- Definitions based on conditions
def is_equilateral_triangle (T : Triangle) : Prop :=
  T.a = T.b ∧ T.b = T.c

def perimeter_quadrilateral (T : Triangle) (D E : Point) (remaining_Q : Quadrilateral) : ℝ :=
  let a_b := dist T.A T.B
  let b_c := dist T.B T.C
  let c_a := dist T.C T.A
  -- corners cut from equilateral triangle T to form quadrilateral=ACE
  let a_d := 4 - 0.5
  let c_e := 4 - 0.5
  let d_e := 1
  a_d + c_e + (dist remaining_Q.A remaining_Q.C) + d_e

-- Main statement to be proved
theorem perimeter_of_remaining_quadrilateral_is_12 (T : Triangle) (D E : Point) (remaining_Q : Quadrilateral) :
  is_equilateral_triangle T ∧ dist D E = 1 ∧ dist D B = 0.5 ∧ dist E B = 0.5 → 
  perimeter_quadrilateral T D E remaining_Q = 12 :=
by
  sorry

end perimeter_of_remaining_quadrilateral_is_12_l754_754251


namespace find_x_values_for_3001_l754_754332

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l754_754332


namespace complex_z_sub_conjugate_eq_neg_i_l754_754845

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754845


namespace additional_grazing_area_l754_754684

theorem additional_grazing_area (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 18) : 
  let A1 := π * r1^2 in
  let A2 := π * r2^2 in
  A2 - A1 = 180 * π :=
by
  sorry

end additional_grazing_area_l754_754684


namespace fixed_point_of_line_l754_754156

theorem fixed_point_of_line (k : ℝ) : ∃ (p : ℝ × ℝ), p = (-3, 4) ∧ ∀ (x y : ℝ), (y - 4 = -k * (x + 3)) → (-3, 4) = (x, y) :=
by
  sorry

end fixed_point_of_line_l754_754156


namespace part1_part2_l754_754904

section
  variable {x m : ℝ}

  def setA : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
  def setB : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}
  def complementB := {x | x > m + 2 ∨ x < m - 2}

  -- Part 1
  theorem part1 (h : setA ∪ setB = setA) : m = 1 := 
  sorry

  -- Part 2
  theorem part2 (h : setA ⊆ complementB) : m > 5 ∨ m < -3 :=
  sorry
end

end part1_part2_l754_754904


namespace greatest_value_of_sum_l754_754456

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + 1 / x^2) : 
  ∃ y, y = x + 1/x ∧ y ≤ sqrt 15 :=
begin
  -- The proof will be here.
  sorry
end

end greatest_value_of_sum_l754_754456


namespace four_possible_x_values_l754_754297

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l754_754297


namespace calc_z_conj_diff_l754_754826

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754826


namespace term_sequence_10th_l754_754448

theorem term_sequence_10th :
  let a (n : ℕ) := (-1:ℚ)^(n+1) * (2*n)/(2*n + 1)
  a 10 = -20/21 := 
by
  sorry

end term_sequence_10th_l754_754448


namespace paper_area_l754_754730

variable (L W : ℕ)

theorem paper_area (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : L * W = 140 := by
  sorry

end paper_area_l754_754730


namespace smallest_positive_period_of_sin_squared_l754_754010

-- Define the function y = sin^2 x
noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2

-- State the theorem about the smallest positive period of this function
theorem smallest_positive_period_of_sin_squared : ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', 0 < T' < T → (∀ x, f (x + T') ≠ f x)) := 
sorry

end smallest_positive_period_of_sin_squared_l754_754010


namespace parabola_focus_coordinates_midpoint_of_segment_l754_754015

noncomputable theory

theorem parabola_focus_coordinates :
    ∃ (p : ℝ) (x_focus y_focus : ℝ), (p > 0) ∧ (y_focus^2 = 2 * p * x_focus) ∧ (x_focus + y_focus - 1 = 0) ∧ (p = 2) ∧ (y_focus = 0) :=
sorry

theorem midpoint_of_segment :
    ∀ (x1 x2 x_focus y_focus : ℝ),
    (x_focus + y_focus - 1 = 0) ∧ 
    (y_focus^2 = 4 * x_focus) ∧ 
    (y_focus = x_focus - 1) ∧ 
    (x1 + x2 = 6) ∧ 
    (x1 * x2 = 1) → 
    (x1 + x2) / 2 = 3 :=
sorry

end parabola_focus_coordinates_midpoint_of_segment_l754_754015


namespace count_integers_with_zero_digit_l754_754026

def contains_zero (n : Nat) : Prop :=
  ∃ d, d ∈ (Nat.digits 10 n) ∧ d = 0

theorem count_integers_with_zero_digit :
  (Finset.filter contains_zero (Finset.range (3500 + 1))).card = 773 := 
sorry

end count_integers_with_zero_digit_l754_754026


namespace exists_region_for_tangents_l754_754793

noncomputable def tangentLineParabola1 (b : ℝ) : (ℝ → ℝ) := (λ x, (1 - 2 * b) * x + b ^ 2)
noncomputable def tangentLineParabola2 (a b : ℝ) : (ℝ → ℝ) := (λ x, a * (1 - 2 * b) * x + a * b ^ 2)

noncomputable def existsThreeTangentsRegion (a : ℝ) (b c : ℝ) : Prop :=
  a ≥ 2 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1

theorem exists_region_for_tangents (a : ℝ) :
  (∃ (b c : ℝ), existsThreeTangentsRegion a b c) :=
sorry

end exists_region_for_tangents_l754_754793


namespace evening_temperature_l754_754478

-- Definitions based on conditions
def noon_temperature : ℤ := 2
def temperature_drop : ℤ := 3

-- The theorem statement
theorem evening_temperature : noon_temperature - temperature_drop = -1 := 
by
  -- The proof is omitted
  sorry

end evening_temperature_l754_754478


namespace path_area_correct_l754_754728

def length_field : Nat := 85
def width_field : Nat := 55
def width_path : Nat := 2.5

def area_of_path :=
  let length_including_path := length_field + 2 * width_path
  let width_including_path := width_field + 2 * width_path
  let area_including_path := length_including_path * width_including_path
  let area_field := length_field * width_field
  area_including_path - area_field

theorem path_area_correct : area_of_path = 725 := by
  sorry

end path_area_correct_l754_754728


namespace find_x_values_for_3001_l754_754331

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l754_754331


namespace range_of_r_l754_754780

noncomputable def r (x : ℝ) : ℝ := x^6 + 6 * x^3 + 9

theorem range_of_r : set.range (r ∘ (λ x, x)) = set.Ici 9 :=
by sorry

end range_of_r_l754_754780


namespace postage_78_5_l754_754677

/-- The postage of a letter is computed based on its weight as follows:
  - 0.8 yuan for a letter weighing no more than 20g,
  - 1.6 yuan for a letter weighing more than 20g but not exceeding 40g,
  - an additional 0.8 yuan is added for every additional 20g (for letters weighing within 100g).

  If someone sends a letter weighing 78.5g, then the postage should be 3.2 yuan.
-/
def postage (w : ℝ) : ℝ :=
  if w ≤ 20 then 0.8
  else if w ≤ 40 then 1.6
  else if w ≤ 60 then 2.4
  else if w ≤ 80 then 3.2
  else 0 -- Ignore the cases for weight > 100g in this context

theorem postage_78_5 : postage 78.5 = 3.2 :=
by
  -- The assumption provides that we directly calculate the postage.
  sorry

end postage_78_5_l754_754677


namespace problem_correct_l754_754072

noncomputable def polar_equation_C1 (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, 1 / (ρ^2) = (Real.cos θ)^2 + (Real.sin θ)^2 / 4

noncomputable def length_AB : ℝ :=
  4 / Real.sqrt 7 + 4

theorem problem_correct :
  (∃ (θ : ℝ), polar_equation_C1 θ) ∧
  (ρ_A, ρ_B : ℝ)
  (ρ_A = -4) ∧
  (ρ_B = 4 / Real.sqrt 7) →
  | (ρ_A + ρ_B) | = length_AB :=
sorry

end problem_correct_l754_754072


namespace tan_difference_l754_754928

theorem tan_difference (α β : ℝ) (hα : Real.tan α = 5) (hβ : Real.tan β = 3) : 
    Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_difference_l754_754928


namespace line_l2_passes_through_point_perpendicular_lines_parallel_lines_maximum_distance_not_2_l754_754898

section

variables {a x y : ℝ}

def l1 (a : ℝ) : set (ℝ × ℝ) := {p | a * p.1 + 2 * p.2 + 3 * a = 0}
def l2 (a : ℝ) : set (ℝ × ℝ) := {p | 3 * p.1 + (a - 1) * p.2 + 3 - a = 0}

theorem line_l2_passes_through_point : 
  (-2 / 3, 1) ∈ l2 a := 
sorry

theorem perpendicular_lines (ha :  l1 a ⊥ l2 a) :
  a = 2 / 5 := 
sorry

theorem parallel_lines (hb : l1 a ∥ l2 a) : 
  a = 3 := 
sorry

theorem maximum_distance_not_2 (hc : O (0, 0)) :
  (∀ d, ¬(d = 2)) := 
sorry

end

end line_l2_passes_through_point_perpendicular_lines_parallel_lines_maximum_distance_not_2_l754_754898


namespace probability_event_A_l754_754869

theorem probability_event_A (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
    {p : ℝ × ℝ | let (x, y) := p; x^2 + y^2 < 1}.measure / {p : ℝ × ℝ | let (x, y) := p; -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1}.measure = π / 4 :=
sorry

end probability_event_A_l754_754869


namespace symmetry_center_l754_754140

noncomputable def transformed_function (x : ℝ) : ℝ :=
  2 * Real.sin (x - π / 3)

theorem symmetry_center :
  ∃ k : ℤ, (∃ x : ℝ, x = π / 3 + k * π) ∧ transformed_function (π / 3) = 0 :=
by
  -- sorry as proof part placeholder
  sorry

end symmetry_center_l754_754140


namespace probability_passing_C_not_D_l754_754492

-- Defining coordinates and the concept of a path
structure Point :=
  (east : ℕ)
  (south : ℕ)

def move_probability (start end_ : Point) (moves : ℕ × ℕ) : ℚ :=
  let total_moves := moves.1 + moves.2 in
  let num_paths := Nat.choose total_moves moves.1 in
  (num_paths : ℚ)

def paths_via (start intermediate end_ : Point) : ℚ :=
  (move_probability start intermediate (intermediate.east - start.east, intermediate.south - start.south)) *
  (move_probability intermediate end_ (end_.east - intermediate.east, end_.south - intermediate.south))

-- Define points A, B, C, and D
def A := Point.mk 0 0
def C := Point.mk 3 2
def B := Point.mk 5 5
def D := Point.mk 1 3

-- Calculate probability
theorem probability_passing_C_not_D :
  (paths_via A C B - paths_via A D B) / move_probability A B (5, 5) = 22 / 63 := 
  sorry

end probability_passing_C_not_D_l754_754492


namespace pencils_and_pens_cost_l754_754153

theorem pencils_and_pens_cost (p q : ℝ)
  (h1 : 8 * p + 3 * q = 5.60)
  (h2 : 2 * p + 5 * q = 4.25) :
  3 * p + 4 * q = 9.68 :=
sorry

end pencils_and_pens_cost_l754_754153


namespace repeating_decimal_sum_l754_754354

theorem repeating_decimal_sum : (0.\overline{3} : ℚ) + (0.\overline{04} : ℚ) + (0.\overline{005} : ℚ) = 1135 / 2997 := 
sorry

end repeating_decimal_sum_l754_754354


namespace rabbit_initial_carrots_l754_754595

theorem rabbit_initial_carrots :
  ∃ c : ℕ, (3 * c - 30) * 3 - 30) * 3 - 30) * 3 - 30 = 0 ∧ c = 15 :=
sorry

end rabbit_initial_carrots_l754_754595


namespace arithmetic_and_harmonic_mean_of_reciprocals_of_first_four_primes_l754_754264

noncomputable def arithmetic_mean (a b c d : ℚ) := (a + b + c + d) / 4

noncomputable def harmonic_mean (a b c d : ℚ) :=
  4 / (1 / a + 1 / b + 1 / c + 1 / d)

theorem arithmetic_and_harmonic_mean_of_reciprocals_of_first_four_primes :
  ∀ (p1 p2 p3 p4 : ℕ), 
  p1 = 2 → p2 = 3 → p3 = 5 → p4 = 7 →
  let a := 1 / (p1 : ℚ) in
  let b := 1 / (p2 : ℚ) in
  let c := 1 / (p3 : ℚ) in
  let d := 1 / (p4 : ℚ) in
    arithmetic_mean a b c d = 247 / 840 ∧
    harmonic_mean a b c d = 4 / 17 :=
begin
  intros p1 p2 p3 p4 hp1 hp2 hp3 hp4,
  rw [hp1, hp2, hp3, hp4],
  let a := 1 / (2 : ℚ),
  let b := 1 / (3 : ℚ),
  let c := 1 / (5 : ℚ),
  let d := 1 / (7 : ℚ),
  split,
  { -- proof for arithmetic_mean
    sorry },
  { -- proof for harmonic_mean
    sorry }
end

end arithmetic_and_harmonic_mean_of_reciprocals_of_first_four_primes_l754_754264


namespace least_pebbles_2021_l754_754581

noncomputable def least_pebbles (n : ℕ) : ℕ :=
  n + n / 2

theorem least_pebbles_2021 :
  least_pebbles 2021 = 3031 :=
by
  sorry

end least_pebbles_2021_l754_754581


namespace distinct_solutions_difference_l754_754996

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : r > s)
  (h_eq : ∀ x : ℝ, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
begin
  sorry
end

end distinct_solutions_difference_l754_754996


namespace distance_P_to_plane_OAB_eq_sqrt17_l754_754962

open Real

-- Define the point P
def P : ℝ × ℝ × ℝ := (3, -4, 1)

-- Define the normal vector of the plane OAB
def n : ℝ × ℝ × ℝ := (2, -2, 3)

-- Distance from point to plane function
def distance_from_point_to_plane (P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  abs ((P.1 * n.1) + (P.2 * n.2) + (P.3 * n.3)) / sqrt (n.1^2 + n.2^2 + n.3^2)

-- Statement of the theorem we need to prove
theorem distance_P_to_plane_OAB_eq_sqrt17 : distance_from_point_to_plane P n = sqrt 17 := by
  sorry

end distance_P_to_plane_OAB_eq_sqrt17_l754_754962


namespace sin_cos_identity_l754_754447

theorem sin_cos_identity (α β : ℝ) (h1 : sin α = cos β) (h2 : cos α = sin (2 * β)) :
  sin β ^ 2 + cos α ^ 2 = 1 :=
sorry

end sin_cos_identity_l754_754447


namespace sum_of_acutes_tan_eq_pi_over_4_l754_754103

theorem sum_of_acutes_tan_eq_pi_over_4 {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h : (1 + Real.tan α) * (1 + Real.tan β) = 2) : α + β = π / 4 :=
sorry

end sum_of_acutes_tan_eq_pi_over_4_l754_754103


namespace greatest_root_of_polynomial_l754_754794

theorem greatest_root_of_polynomial :
  let f : ℝ → ℝ := λ x, 12*x^4 - 8*x^2 + 1
  in ∀ x, (x^2 = 1/2 ∨ x^2 = 1/6) → x ≤ sqrt 2 / 2 :=
by
  intros f x hx
  sorry

end greatest_root_of_polynomial_l754_754794


namespace circle_area_double_l754_754678

noncomputable def increased_radius (r n : ℝ) : Prop :=
  r = n * (Real.sqrt 2 + 1)

theorem circle_area_double (r n : ℝ) (h: 0 < r ∧ 0 ≤ n):
  (π * (r + n) ^ 2 = 2 * π * r ^ 2) → increased_radius r n :=
by
  intro h_area
  have h_eq : (r + n) ^ 2 = 2 * r ^ 2 := by
    calc
      (r + n) ^ 2 = 2 * r ^ 2 : by simpa [pi, mul_assoc, mul_comm] using h_area
  sorry

end circle_area_double_l754_754678


namespace exist_prime_not_dividing_l754_754878

theorem exist_prime_not_dividing (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, 0 < n → ¬ (q ∣ n^p - p) := 
sorry

end exist_prime_not_dividing_l754_754878


namespace average_student_headcount_l754_754669

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end average_student_headcount_l754_754669


namespace cos_C_in_triangle_l754_754527

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end cos_C_in_triangle_l754_754527


namespace z_conjugate_difference_l754_754850

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754850


namespace total_cans_given_away_l754_754271

noncomputable def total_cans_initial : ℕ := 2000
noncomputable def cans_taken_first_day : ℕ := 500
noncomputable def restocked_first_day : ℕ := 1500
noncomputable def people_second_day : ℕ := 1000
noncomputable def cans_per_person_second_day : ℕ := 2
noncomputable def restocked_second_day : ℕ := 3000

theorem total_cans_given_away :
  (cans_taken_first_day + (people_second_day * cans_per_person_second_day) = 2500) :=
by
  sorry

end total_cans_given_away_l754_754271


namespace only_CondD_is_true_l754_754680
-- Import the necessary math library

-- Define the variables
variables {a b : ℝ}

-- Define the conditions
def CondA := - (a - b) = a - b
def CondB := - (-a - b) = a - b
def CondC := a^2 + 2 * (a - 2 * b) = a^2 + 2 * a - 2 * b
def CondD := (a - 2) * (a - 2 * b) = a^2 - 2 * a + 4 * b

-- Define the theorem to prove that only CondD holds
theorem only_CondD_is_true : ¬CondA ∧ ¬CondB ∧ ¬CondC ∧ CondD :=
by
  sorry

end only_CondD_is_true_l754_754680


namespace find_n_vals_l754_754571

def C (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 20 then 15 * n
  else if 21 ≤ n ∧ n ≤ 40 then 13 * n - 10
  else if n > 40 then 12 * n - 30
  else 0 -- default case for n < 1, assume cost is 0

theorem find_n_vals : 
  (finset.filter (λ n, C (n + 1) < C n) (finset.range 100)).card = 4 :=
sorry

end find_n_vals_l754_754571


namespace count_of_valid_numbers_l754_754450

noncomputable def count_valid_numbers : ℕ :=
  let valid_numbers := {n : ℕ | ∃ x y, n = 101 * x + 10 * y + x ∧
                                x ≠ y ∧
                                0 ≤ x ∧ x < 10 ∧
                                0 ≤ y ∧ y < 10 ∧
                                (2 * x + y) < 18 ∧
                                ¬((2 * x + y) % 3 = 0)} in
  valid_numbers.to_finset.card

theorem count_of_valid_numbers : count_valid_numbers = 42 := -- Adjust the number accordingly
sorry

end count_of_valid_numbers_l754_754450


namespace price_of_sugar_and_salt_l754_754169

theorem price_of_sugar_and_salt:
  (∀ (sugar_price salt_price : ℝ), 2 * sugar_price + 5 * salt_price = 5.50 ∧ sugar_price = 1.50 →
  3 * sugar_price + salt_price = 5) := 
by 
  sorry

end price_of_sugar_and_salt_l754_754169


namespace distribute_4_balls_into_4_boxes_l754_754031

noncomputable def distribute_indistinguishable_balls (n k : ℕ) : ℕ :=
  binomial (n + k - 1) (k - 1)

theorem distribute_4_balls_into_4_boxes :
  distribute_indistinguishable_balls 4 4 = 35 :=
by
  -- We use a binomial coefficient to calculate the number of ways
  -- to distribute 4 indistinguishable balls into 4 distinguishable boxes
  sorry

end distribute_4_balls_into_4_boxes_l754_754031


namespace inverse_condition_l754_754167

def f (a x : ℝ) : ℝ := x^2 - 2 * a * x - 3

theorem inverse_condition (a : ℝ) :
  (∃ (g : ℝ → ℝ), ∀ x ∈ set.Icc 1 2, g (f a x) = x) ↔ (a ≤ 1 ∨ a ≥ 2) :=
sorry

end inverse_condition_l754_754167


namespace calc_z_conj_diff_l754_754831

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754831


namespace find_F_internal_tangent_l754_754877

theorem find_F_internal_tangent
  (F : ℝ)
  (h₁ : ∀ x y : ℝ, x^2 + y^2 = 1 →  x^2 + y^2 - 6*x - 8*y + F = 0 → (x, y) = (3, 4) ∨ ∃ r1 r2 : ℝ, r1 = 1 ∧ r2 = (real.sqrt (25 - F)) ∧ real.dist (0, 0) (3, 4) = abs(r1 - r2)) :
  F = -11 :=
  sorry


end find_F_internal_tangent_l754_754877


namespace rhino_folds_could_not_happen_l754_754212

-- Definition of the problem conditions
variable (V H : ℕ)
variable (total_folds : V + H = 17)
variable (scratches : ℕ)

-- Statement of the mathematical problem
theorem rhino_folds_could_not_happen :
  V + H = 17 → 
  (∀ scratches, (∃ V' H',
    V' = H ∧ H' = V ∧
    V' + H' = 17)) → False :=
by
  intros
  sorry

end rhino_folds_could_not_happen_l754_754212


namespace distinct_solutions_diff_l754_754990

theorem distinct_solutions_diff {r s : ℝ} 
  (h1 : ∀ a b : ℝ, (a ≠ b → ( ∃ x, x ≠ a ∧ x ≠ b ∧ (x = r ∨ x = s) ∧ (x-5)(x+5) = 25*x - 125) )) 
  (h2 : r > s) : r - s = 15 :=
by
  sorry

end distinct_solutions_diff_l754_754990


namespace limit_sequence_eq_five_l754_754215

noncomputable def limit_sequence : ℝ :=
  real.limit (λ n : ℕ, (n * real.sqrt (real.nat_abs n) + real.sqrt (25 * real.nat_abs (n ^ 4) - 81)) / 
                         ((n - 7 * real.sqrt n) * real.sqrt (n ^ 2 - n + 1)))

theorem limit_sequence_eq_five : limit_sequence = 5 := 
begin
  sorry
end

end limit_sequence_eq_five_l754_754215


namespace age_difference_l754_754236

-- Defining the current age of the son
def S : ℕ := 26

-- Defining the current age of the man
def M : ℕ := 54

-- Defining the condition that in two years, the man's age is twice the son's age
def condition : Prop := (M + 2) = 2 * (S + 2)

-- The theorem that states how much older the man is than the son
theorem age_difference : condition → M - S = 28 := by
  sorry

end age_difference_l754_754236


namespace four_possible_x_values_l754_754296

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l754_754296


namespace maximum_cookies_andy_could_have_eaten_l754_754255

theorem maximum_cookies_andy_could_have_eaten (x : ℕ) (hx1 : ∃ y : ℕ, y = 2 * x ∧ x + y = 30) : x = 10 :=
by
  obtain ⟨y, hy1, hy2⟩ := hx1
  rw hy1 at hy2
  linarith

end maximum_cookies_andy_could_have_eaten_l754_754255


namespace rational_solution_for_k_is_6_l754_754392

theorem rational_solution_for_k_is_6 (k : ℕ) (h : 0 < k) :
  (∃ x : ℚ, k * x ^ 2 + 12 * x + k = 0) ↔ k = 6 :=
by { sorry }

end rational_solution_for_k_is_6_l754_754392


namespace triple_composition_f_3_l754_754466

def f (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_f_3 : f (f (f 3)) = 107 :=
by
  sorry

end triple_composition_f_3_l754_754466


namespace tangent_circle_line_l754_754922

theorem tangent_circle_line (r : ℝ) (h_pos : 0 < r) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 = r^2) 
  (h_line : ∀ x y : ℝ, x + y = r + 1) : 
  r = 1 + Real.sqrt 2 := 
by 
  sorry

end tangent_circle_line_l754_754922


namespace rational_roots_count_l754_754800

theorem rational_roots_count (b₃ b₂ b₁ : ℚ) :
    ∃ (s : finset ℚ), s.card = 22 ∧ ∀ x ∈ s, (6 * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 10 = 0) :=
sorry

end rational_roots_count_l754_754800


namespace exists_point_M_on_EC_l754_754875

-- Given two non-congruent isosceles right triangles ABC and ADE
variables {A B C D E : Type*}
variable [incidence_geometry : MetricSpace A]

-- Conditions
-- Triangle ABC is an isosceles right triangle
def triangle_ABC_isosceles_right : Prop :=
  isosceles_right (triangle A B C)

-- Triangle ADE is an isosceles right triangle
def triangle_ADE_isosceles_right : Prop :=
  isosceles_right (triangle A D E)

-- Triangle ABC and ADE are non-congruent
def non_congruent_triangles : Prop :=
  ¬congruent (triangle A B C) (triangle A D E)

-- Define point M on the line segment EC
def point_on_segment (E C : incidence_geometry) : Type := {M : incidence_geometry // M ∈ line_segment E C}

-- The objective theorem
theorem exists_point_M_on_EC (h1 : triangle_ABC_isosceles_right) 
                              (h2 : triangle_ADE_isosceles_right) 
                              (h3 : non_congruent_triangles) :
  ∃ M : incidence_geometry, M ∈ line_segment E C ∧ isosceles_right (triangle B M D) :=
by
  sorry

end exists_point_M_on_EC_l754_754875


namespace find_a_l754_754431

noncomputable def f (a : ℝ) (g : ℝ → ℝ) : ℝ → ℝ :=
λ x, if x > 0 then a + real.log x else g x - x

theorem find_a (g : ℝ → ℝ) (h_odd_f : ∀ x, f a g (-x) = - f a g x) (h_g_neg_e : g (-real.exp 1) = 0) :
  a = -1 - real.exp 1 := 
sorry

end find_a_l754_754431


namespace prove_concurrency_l754_754089

noncomputable def concurrent_lines (Γ : Type) (A B C D : Γ) (A' B' C' D' : Γ)
  (inscribed_square : is_square A B C D)
  (tangency : ∀ (P : Γ), P ∈ {A', B', C', D'} → P ∈ Γ)
  (circle_tangent_a : is_tangent_interior Γ A B A')
  (circle_tangent_b : is_tangent_interior Γ B C B')
  (circle_tangent_c : is_tangent_interior Γ C D C')
  (circle_tangent_d : is_tangent_interior Γ D A D')
  : Prop :=
are_concurrent [A, A'], [B, B'], [C, C'], [D, D']

theorem prove_concurrency (Γ : Type) [metric_space Γ] [inner_product_space ℝ Γ]
  (A B C D : Γ) (A' B' C' D' : Γ)
  (inscribed_square : is_square A B C D)
  (tangency : ∀ (P : Γ), P ∈ {A', B', C', D'} → P ∈ Γ)
  (circle_tangent_a : is_tangent_interior Γ A B A')
  (circle_tangent_b : is_tangent_interior Γ B C B')
  (circle_tangent_c : is_tangent_interior Γ C D C')
  (circle_tangent_d : is_tangent_interior Γ D A D') : 
  concurrent_lines Γ A B C D A' B' C' D' :=
sorry

end prove_concurrency_l754_754089


namespace symmetric_pairs_9_3_symmetric_pairs_3_y_symmetric_pairs_x_2_symmetric_pairs_a_b_symmetric_pairs_a_b_alt_l754_754773

-- Part 1: Symmetric pairs of (9, 3)
theorem symmetric_pairs_9_3 : 
  (∀ (a b : ℝ), a = 9 ∧ b = 3 →
    (1 / (Real.sqrt a), Real.sqrt b) = (1 / 3, Real.sqrt 3) ∧ 
    (Real.sqrt b, 1 / (Real.sqrt a)) = (Real.sqrt 3, 1 / 3)) := by
  intro a b h
  sorry

-- Part 2: Finding y when symmetric pairs of (3, y) are the same
theorem symmetric_pairs_3_y (y : ℝ) (hy : y > 0) : 
  (1 / (Real.sqrt 3) = Real.sqrt y) ↔ y = 1 / 3 := by
  intro hy
  sorry

-- Part 3: Finding x if one symmetric pair of (x, 2) is (sqrt(2), 1)
theorem symmetric_pairs_x_2 (x : ℝ) (hx : x > 0) : 
  (1 / (Real.sqrt x) = Real.sqrt 2) ↔ x = 1 := by
  intro hx
  sorry

-- Part 3: Finding a, b if one symmetric pair of (a, b) is (sqrt(3), 3 * sqrt(2))
theorem symmetric_pairs_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / (Real.sqrt a) = Real.sqrt 3 ∧ Real.sqrt b = 3 * Real.sqrt 2) ↔ (a = 1 / 3 ∧ b = 18) :=
   sorry

theorem symmetric_pairs_a_b_alt (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / (Real.sqrt a) = 3 * Real.sqrt 2 ∧ Real.sqrt b = Real.sqrt 3) ↔ (a = 1 / 18 ∧ b = 3) := 
  sorry

end symmetric_pairs_9_3_symmetric_pairs_3_y_symmetric_pairs_x_2_symmetric_pairs_a_b_symmetric_pairs_a_b_alt_l754_754773


namespace prove_PoincareObservation_l754_754219

noncomputable def PoincareObservation 
  (X : ℕ → ℕ → ℝ)
  (ξ : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n m : ℕ)
  (hn : n ≥ 1)
  (hm : m ≥ 1) :
  Prop :=
  tendsto (λ (n : ℕ), (λ i, X n i) '' {1..m}) 
          (𝓝 (λ i, ξ i '' {1..m}))

theorem prove_PoincareObservation 
  (X : ℕ → ℕ → ℝ)
  (ξ : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n m : ℕ)
  (hn : n ≥ 1)
  (hm : m ≥ 1)
  (h_uniform : ∀ n, ∀ i ∈ finset.range n, MeasureTheory.measure_probability_measure.map (MeasureTheory.probability_measure_of_measurable_set (λ x, X n x) finset.range n) = spherical_measure (S n))
  (h_gaussian : ∀ i, MeasureTheory.probability_measure (λ x, ξ i x)) :
  PoincareObservation X ξ S n m hn hm :=
begin
  sorry
end

end prove_PoincareObservation_l754_754219


namespace cos_value_of_expression_l754_754920

theorem cos_value_of_expression (α : ℝ) (h : cos (π / 8 - α) = 1 / 6) :
  cos (3 * π / 4 + 2 * α) = 17 / 18 :=
sorry

end cos_value_of_expression_l754_754920


namespace log_equality_l754_754349

theorem log_equality (x : ℝ) : (8 : ℝ)^x = 16 ↔ x = 4 / 3 :=
by
  sorry

end log_equality_l754_754349


namespace wire_around_field_l754_754370

theorem wire_around_field 
  (area_square : ℕ)
  (total_length_wire : ℕ)
  (h_area : area_square = 69696)
  (h_total_length : total_length_wire = 15840) :
  (total_length_wire / (4 * Int.natAbs (Int.sqrt area_square))) = 15 :=
  sorry

end wire_around_field_l754_754370


namespace remainder_of_x_divided_by_30_l754_754600

theorem remainder_of_x_divided_by_30:
  ∀ x : ℤ,
    (4 + x ≡ 9 [ZMOD 8]) ∧ 
    (6 + x ≡ 8 [ZMOD 27]) ∧ 
    (8 + x ≡ 49 [ZMOD 125]) ->
    (x ≡ 17 [ZMOD 30]) :=
by
  intros x h
  sorry

end remainder_of_x_divided_by_30_l754_754600


namespace radius_of_large_circle_l754_754644

noncomputable def small_circle_radius : ℝ := 2
noncomputable def larger_circle_radius : ℝ :=
  let side_length := 2 * small_circle_radius in
  let altitude := (side_length * Real.sqrt 3) / 2 in
  side_length + altitude

theorem radius_of_large_circle :
  let r := (2 + 4 * Real.sqrt 3) / 2 in
  larger_circle_radius = r := by
  sorry

end radius_of_large_circle_l754_754644


namespace arrangement_count_l754_754695

theorem arrangement_count : 
  let entities := ["仁", "礼", "义", "智", "信"]
  let adjacent_pairs := [("礼", "义"), ("智", "信")]
  ∃ (n : ℕ), (n = 24) ∧
    ∀ (arrangement : List String),
      arrangement ∈ (entities.permutations.filter (λ l, 
        (l.index_of adjacent_pairs[0].1) + 1 = (l.index_of adjacent_pairs[0].2) 
        ∨ (l.index_of adjacent_pairs[0].2) + 1 = (l.index_of adjacent_pairs[0].1)) ∧
        ((l.index_of adjacent_pairs[1].1) + 1 = (l.index_of adjacent_pairs[1].2) 
        ∨ (l.index_of adjacent_pairs[1].2) + 1 = (l.index_of adjacent_pairs[1].1)))
    sorry

end arrangement_count_l754_754695


namespace angle_between_a_and_c_is_zero_l754_754101

variable {V : Type*} [InnerProductSpace ℝ V]

variables (a b c : V)

-- Condition: a, b, and c are unit vectors
axiom h1 : ‖a‖ = 1
axiom h2 : ‖b‖ = 1
axiom h3 : ‖c‖ = 1

-- Condition: a + 2b + c = 0
axiom h4 : a + (2 : ℝ) • b + c = 0

-- Condition: b ⋅ c = 0
axiom h5 : ⟪b, c⟫ = 0

-- Theorem: The angle between a and c is 0 degrees
theorem angle_between_a_and_c_is_zero : real.angle a c = 0 := by
  sorry

end angle_between_a_and_c_is_zero_l754_754101


namespace magnitude_of_a_plus_b_in_range_l754_754911

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def theta_domain : Set ℝ := {θ : ℝ | -Real.pi / 2 < θ ∧ θ < Real.pi / 2}

open Real

theorem magnitude_of_a_plus_b_in_range (θ : ℝ) (hθ : θ ∈ theta_domain) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (cos θ, sin θ)
  1 < sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) ∧ sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) < (3 + 2 * sqrt 2) :=
sorry

end magnitude_of_a_plus_b_in_range_l754_754911


namespace count_congruent_to_5_mod_13_lt_500_l754_754449

/-- The statement that there are 39 positive integers less than 500 that are congruent to 5 modulo 13. -/
theorem count_congruent_to_5_mod_13_lt_500 : 
  finset.card (finset.filter (λ n : ℕ, n % 13 = 5) (finset.Ico 1 500)) = 39 := 
sorry

end count_congruent_to_5_mod_13_lt_500_l754_754449


namespace compare_negatives_l754_754282

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l754_754282


namespace add_base6_numbers_eq_l754_754262

theorem add_base6_numbers_eq : 
    let n1 := 1 * 6^3 + 4 * 6^2 + 5 * 6^1 + 2 * 6^0
    let n2 := 2 * 6^3 + 3 * 6^2 + 5 * 6^1 + 4 * 6^0
    let sum_base10 := n1 + n2
    966 = sum_base10 → 966 = 4 * 6^3 + 2 * 6^2 + 5 * 6^1 + 0 * 6^0

end add_base6_numbers_eq_l754_754262


namespace circle_intersects_x_axis_at_fixed_points_l754_754750

-- Define the conditions from the problem
variable (t p : ℝ) (t_pos : t > 0) (p_pos : p > 0)

-- Define the circle with diameter MN in Lean and its intersection with the x-axis
theorem circle_intersects_x_axis_at_fixed_points :
  ∃ x1 x2 : ℝ, 
    (x1 = -t + Real.sqrt(2*p*t) ∧ x2 = -t - Real.sqrt(2*p*t)) ∧
    (∀ y : ℝ, (y^2 = 2*p*x1) ∧ (y^2 = 2*p*x2)) :=
by
  sorry

end circle_intersects_x_axis_at_fixed_points_l754_754750


namespace normal_distribution_properties_l754_754708

noncomputable def normal_distribution := sorry -- Define the normal distribution if not predefined
variables (X : ℝ) (μ σ : ℝ)

-- Given that X is normally distributed with mean 2 and variance 4
axiom X_normal : normal_distribution X 2 4

-- Define mean and standard deviation
def mean (X : ℝ) := 2
def variance (X : ℝ) := 4
def stddev (X : ℝ) := real.sqrt (variance X)

-- Define the probabilities
def prob_greater_than_mean : Prop := 
  @probability_theorems.normal_distribution.prob_greater_than X 2 2 = 1 / 2 -- Placeholder for the actual probability theorem

def prob_symmetric_around_mean (a b : ℝ): Prop := 
  @probability_theorems.normal_distribution.prob_symmetric X 2 a b -- Placeholder for actual symmetric probability theorem

theorem normal_distribution_properties:
  (mean X = 2) ∧ 
  (stddev X = 2) ∧ 
  (prob_greater_than_mean) ∧ 
  (prob_symmetric_around_mean 3 1) := 
sorry

end normal_distribution_properties_l754_754708


namespace ratio_of_angles_l754_754046

-- Define the angles and sides in the given triangle
variables {A B C : ℝ} (a b c : ℝ)

-- Given conditions
def condition1 : Prop := a^2 = b * (b + c)

-- The proof statement
theorem ratio_of_angles (h : condition1) : B / A = 1 / 2 :=
sorry -- proof is omitted

end ratio_of_angles_l754_754046


namespace find_k_l754_754446

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (k : ℝ) : ℝ × ℝ := (2 * k, 3)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

theorem find_k : ∃ k : ℝ, dot_product a (vector_add (scalar_mult 2 a) (b k)) = 0 ∧ k = -8 :=
by
  sorry

end find_k_l754_754446


namespace car_speed_l754_754705

def distance : ℝ := 375
def time : ℝ := 3
def speed : ℝ := distance / time

theorem car_speed :
  speed = 125 := 
by
  unfold distance time speed
  sorry

end car_speed_l754_754705


namespace sin_cos_bound_sin_cos_least_upper_bound_l754_754406

theorem sin_cos_bound (x y z : ℝ) (m n : ℕ) (hm : 2 ≤ m) (hn : 2 ≤ n) :
  sin x ^ m * cos y ^ n + sin y ^ m * cos z ^ n + sin z ^ m * cos x ^ n ≤ 1 :=
sorry

theorem sin_cos_least_upper_bound (x y z : ℝ) (m n : ℕ) (hm : 2 ≤ m) (hn : 2 ≤ n) :
  ∃ a, (∀ q r s : ℝ, sin q ^ m * cos r ^ n + sin r ^ m * cos s ^ n + sin s ^ m * cos q ^ n ≤ a) ∧
    (∀ b, (∀ q r s : ℝ, sin q ^ m * cos r ^ n + sin r ^ m * cos s ^ n + sin s ^ m * cos q ^ n ≤ b) → a ≤ b) :=
begin
  existsi (1 : ℝ),
  split,
  { intros q r s, 
    exact sin_cos_bound q r s m n hm hn },
  { intros b hb,
    specialize hb (π / 2) 0 0,
    rw [sin_pi_div_two, cos_zero, sin_zero, cos_zero, sin_zero, cos_pi_div_two] at hb,
    linarith },
end

end sin_cos_bound_sin_cos_least_upper_bound_l754_754406


namespace average_growth_rate_le_half_sum_l754_754709

variable (p q x : ℝ)

theorem average_growth_rate_le_half_sum : 
  (1 + p) * (1 + q) = (1 + x) ^ 2 → x ≤ (p + q) / 2 :=
by
  intro h
  sorry

end average_growth_rate_le_half_sum_l754_754709


namespace complex_conj_difference_l754_754834

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754834


namespace compare_negatives_l754_754280

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l754_754280


namespace minimal_transpositions_l754_754546

def transposition {α : Type} (A : list (list α)) (i j k l : ℕ) : list (list α) :=
  let Ai := A[i] in
  let Bi := list.update A i (list.update Ai j (Ai[k] : ℕ)) in
  list.update Bi i (list.update (Bi[i]) k (Ai[j] : ℕ))

theorem minimal_transpositions (n : ℕ) (A B : matrix (fin n) (fin n) ℕ) (hA : ∀ i j, A i j ∈ finset.range (n^2 + 1)) (hB : ∀ i j, B i j ∈ finset.range (n^2 + 1)) :
  ∃ m, m = 2 * n * (n - 1) ∧
  ∃ f : fin m → fin m → fin n × fin n × fin n × fin n, ∀ k, transposition A (f k).1.1 (f k).1.2 (f k).2.1 (f k).2.2 = B :=
sorry

end minimal_transpositions_l754_754546


namespace walking_time_l754_754237

theorem walking_time (v : ℕ) (d : ℕ) (h1 : v = 10) (h2 : d = 4) : 
    ∃ (T : ℕ), T = 24 := 
by
  sorry

end walking_time_l754_754237


namespace compare_triangle_quadrilateral_l754_754694

open Set Finite

-- Define the unit disc.
def unit_disc : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

-- Probability that a set of points forms a convex hull which is a triangle or quadrilateral.
noncomputable def probability_convex_hull (n : ℕ) (f : Fin n → (ℝ × ℝ)) : Prop :=
  let points := (Set.range f)
  -- Convex hull definition and its vertex condition need to be formalized
  false -- Placeholder for actual conditions and probability computations

-- Problem statement: Compare the probabilities for n=2019.
theorem compare_triangle_quadrilateral (n : ℕ) (h : n = 2019)
  (f : Fin n → (ℝ × ℝ))
  (hf : ∀ i, f i ∈ unit_disc)
  (random_indep : ∀ i j, i ≠ j → (f i, f j) ∈ (λ p : (ℝ × ℝ) × (ℝ × ℝ), p.1 ≠ p.2)) :
  probability_convex_hull n f → 
  ∃ p_triangle p_quadrilateral : ℝ,
    p_quadrilateral > p_triangle :=
sorry

end compare_triangle_quadrilateral_l754_754694


namespace valid_set_example_l754_754679

def is_valid_set (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, x ≠ y

theorem valid_set_example : is_valid_set { x : ℝ | x > Real.sqrt 2 } :=
sorry

end valid_set_example_l754_754679


namespace find_angle_ABC_l754_754979

variables (AB BC CD DA AB' BC' CD' DA' : ℝ)
variables (DAB ABC : ℝ)
variables (MCD : ℝ := 15)
variables (M : Point)  -- M is a point but we'll need to consider the geometric environment

-- Assume the conditions in Lean statement
def parallelogram (ABBCDCDA : Prop) : Prop := 
  (AB = CD) ∧ (BC = DA) ∧ (AB > BC) ∧ (DAB < ABC)

def perp_bis_intersect (M : Point) (perpendicular_bisector_AB perpendicular_bisector_BC : Line) (AD_extension : Line) : Prop :=
  (M ∈ perpendicular_bisector_AB) ∧  (M ∈ perpendicular_bisector_BC) ∧ (M ∈ AD_extension)

-- Equivalent mathematical proof problem in Lean 4
theorem find_angle_ABC :
  parallelogram ABBCDCDA → perp_bis_intersect M perpendicular_bisector_AB perpendicular_bisector_BC AD_extension → 
  MCD = 15 →
  ABC = 7.5 :=
begin
  sorry,
end

end find_angle_ABC_l754_754979


namespace randy_quiz_goal_l754_754588

def randy_scores : List ℕ := [90, 98, 92, 94]
def randy_next_score : ℕ := 96
def randy_goal_average : ℕ := 94

theorem randy_quiz_goal :
  let total_score := randy_scores.sum
  let required_total_score := 470
  total_score + randy_next_score = required_total_score →
  required_total_score / randy_goal_average = 5 :=
by
  intro h
  sorry

end randy_quiz_goal_l754_754588


namespace field_area_l754_754242

theorem field_area
  (L : ℕ) (W : ℕ) (A : ℕ)
  (h₁ : L = 20)
  (h₂ : 2 * W + L = 100)
  (h₃ : A = L * W) :
  A = 800 := by
  sorry

end field_area_l754_754242


namespace compare_abc_l754_754399

theorem compare_abc (a b c : ℝ) (h1 : a = Real.logb 3 (3 / 2)) (h2 : b = Real.logb (2 / 3) (1 / 2)) (h3 : c = 2^(-1 / 2)) : a < c ∧ c < b :=
  by
  sorry

end compare_abc_l754_754399


namespace counting_five_digit_numbers_l754_754914

theorem counting_five_digit_numbers :
  ∃ (M : ℕ), 
    (∃ (b : ℕ), (∃ (y : ℕ), 10000 * b + y = 8 * y ∧ 10000 * b = 7 * y ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1429 ≤ y ∧ y ≤ 9996)) ∧ 
    (M = 1224) := 
by
  sorry

end counting_five_digit_numbers_l754_754914


namespace find_price_of_pants_l754_754966

theorem find_price_of_pants
  (price_jacket : ℕ)
  (num_jackets : ℕ)
  (price_shorts : ℕ)
  (num_shorts : ℕ)
  (num_pants : ℕ)
  (total_cost : ℕ)
  (h1 : price_jacket = 10)
  (h2 : num_jackets = 3)
  (h3 : price_shorts = 6)
  (h4 : num_shorts = 2)
  (h5 : num_pants = 4)
  (h6 : total_cost = 90)
  : (total_cost - (num_jackets * price_jacket + num_shorts * price_shorts)) / num_pants = 12 :=
by sorry

end find_price_of_pants_l754_754966


namespace recurring_sum_fractions_l754_754352

theorem recurring_sum_fractions :
  let x := (1 / 3) in
  let y := (4 / 99) in
  let z := (5 / 999) in
  x + y + z = (742 / 999) :=
by 
  sorry

end recurring_sum_fractions_l754_754352


namespace first_player_can_win_l754_754120

-- Definition of the grid and moves
variable (k : ℕ) (initial_column : ℕ) (row : ℕ)
def valid_game_conditions : Prop :=
  (k > 2) ∧ (initial_column = 1) ∧ (0 < row ∧ row <= 2012)

theorem first_player_can_win (k : ℕ) (initial_column : ℕ) (row : ℕ) 
  (h: valid_game_conditions k initial_column row) : 
  ∃ strategy, strategy_guarantees_win strategy := 
sorry

end first_player_can_win_l754_754120


namespace julia_savings_l754_754081

theorem julia_savings :
  let original_price_per_notebook := 3
  let number_of_notebooks := 10
  let discount_rate_primary := 0.30
  let discount_rate_bulk := 0.10
  let total_discount_rate := discount_rate_primary + discount_rate_bulk
  let discounted_price_per_notebook := original_price_per_notebook * (1 - total_discount_rate)
  let total_cost_without_discount := number_of_notebooks * original_price_per_notebook
  let total_cost_with_discount := number_of_notebooks * discounted_price_per_notebook
  let total_savings := total_cost_without_discount - total_cost_with_discount
  in total_savings = 12 := 
by {
  sorry
}

end julia_savings_l754_754081


namespace f1_is_closed_f2_k_range_l754_754859

def is_closed_function {α : Type*} (D : set α) (f : α → α) : Prop :=
  (monotone f ∨ antitone f) ∧ 
  ∃ a b : α, (a < b) ∧ (∀ x ∈ (set.Icc a b), f x ∈ set.Icc a b)

def f1 (x : ℝ) : ℝ := -x^3
def f1_closed := is_closed_function set.univ f1

def f2 (k : ℝ) (x : ℝ) : ℝ := k + sqrt (x + 2)
def f2_closed (k : ℝ) := is_closed_function (set.Ici (-2)) (f2 k)

theorem f1_is_closed : ∃ (a b : ℝ), (a < b) ∧ (set.Icc a b ⊆ set.univ) ∧ (∀ x ∈ set.Icc a b, f1 x ∈ set.Icc a b) :=
by { use [-1, 1], split, norm_num, split, simp, sorry }

theorem f2_k_range : ∀ k : ℝ, f2_closed k → (-9/4 : ℝ) < k ∧ k ≤ -2 :=
by { sorry }

end f1_is_closed_f2_k_range_l754_754859


namespace num_x_for_3001_in_sequence_l754_754303

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l754_754303


namespace num_x_for_3001_in_sequence_l754_754302

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l754_754302


namespace negative_half_less_than_negative_third_l754_754279

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l754_754279


namespace average_student_headcount_l754_754667

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end average_student_headcount_l754_754667


namespace solve_quadratic_inequality_l754_754139

theorem solve_quadratic_inequality (a : ℝ) (x : ℝ) :
  (x^2 - a * x + a - 1 ≤ 0) ↔
  (a < 2 ∧ a - 1 ≤ x ∧ x ≤ 1) ∨
  (a = 2 ∧ x = 1) ∨
  (a > 2 ∧ 1 ≤ x ∧ x ≤ a - 1) := 
by
  sorry

end solve_quadratic_inequality_l754_754139


namespace shaded_area_l754_754954

theorem shaded_area 
    (rect1_base rect1_height : ℕ) 
    (rect2_base rect2_height : ℕ) 
    (rect3_base rect3_height : ℕ) 
    (tri1_base tri1_height : ℕ) 
    (tri2_base tri2_height : ℕ) 
    (h1 : rect1_base = 2) (h2 : rect1_height = 3) 
    (h3 : rect2_base = 3) (h4 : rect2_height = 4) 
    (h5 : rect3_base = 4) (h6 : rect3_height = 5) 
    (h7 : tri1_base = 12) (h8 : tri1_height = 4) 
    (h9 : tri2_base = 3) (h10 : tri2_height = 2) :
    let rect_area1 := rect1_base * rect1_height,
        rect_area2 := rect2_base * rect2_height,
        rect_area3 := rect3_base * rect3_height,
        triangle_area1 := (tri1_base * tri1_height) / 2,
        triangle_area2 := (tri2_base * tri2_height) / 2,
        total_area := rect_area1 + rect_area2 + rect_area3,
        shaded_area := total_area - triangle_area1 - triangle_area2 in
    shaded_area = 11 :=
by
  sorry

end shaded_area_l754_754954


namespace expected_heads_l754_754109

theorem expected_heads (n : ℕ := 100) (p : ℚ := 1/2) :
  let flip_probability := [p, p/2, p/4, p/8, p/16]
  let expected_heads := n * (flip_probability.sum)
in expected_heads = 93.75 := by
  sorry

end expected_heads_l754_754109


namespace action_figure_prices_l754_754087

noncomputable def prices (x y z w : ℝ) : Prop :=
  12 * x + 8 * y + 5 * z + 10 * w = 220 ∧
  x / 4 = y / 3 ∧
  x / 4 = z / 2 ∧
  x / 4 = w / 1

theorem action_figure_prices :
  ∃ x y z w : ℝ, prices x y z w ∧
    x = 220 / 23 ∧
    y = (3 / 4) * (220 / 23) ∧
    z = (1 / 2) * (220 / 23) ∧
    w = (1 / 4) * (220 / 23) :=
  sorry

end action_figure_prices_l754_754087


namespace correct_inequalities_l754_754338

variables {f : ℝ → ℝ} (h1 : ∀ x, f(-x) = -f(x)) (h2 : ∀ x y, x < y → f(x) > f(y))  {a b : ℝ} (h3 : a + b ≤ 0)

theorem correct_inequalities: 
  (f(a) * f(-a) ≤ 0) ∧ (f(a) + f(b) ≥ f(-a) + f(-b)) :=
by 
  sorry

end correct_inequalities_l754_754338


namespace urn_marbles_100_white_l754_754528

theorem urn_marbles_100_white 
(initial_white initial_black final_white final_black : ℕ) 
(h_initial : initial_white = 150 ∧ initial_black = 50)
(h_operations : 
  (∀ n, (initial_white - 3 * n + 2 * n = final_white ∧ initial_black + n = final_black) ∨
  (initial_white - 2 * n - 1 = initial_white ∧ initial_black = final_black) ∨
  (initial_white - 1 * n - 2 = final_white ∧ initial_black - 1 * n = final_black) ∨
  (initial_white - 3 * n + 2 = final_white ∧ initial_black + 1 * n = final_black)) →
  ((initial_white = 150 ∧ initial_black = 50) →
   ∃ m: ℕ, final_white = 100)) :
∃ n: ℕ, initial_white - 3 * n + 2 * n = 100 ∧ initial_black + n = final_black :=
sorry

end urn_marbles_100_white_l754_754528


namespace KH_parallel_IO_l754_754941

variables (a b c : ℝ) (A B C D E K I O H : Type)

-- Given Conditions
variables [triangle_ABC_acute : acute_triangle A B C]
variables [sides_lengths : sides_lengths A B C a b c]
variables (a_gt_b : a > b) (b_gt_c : b > c)
variables (incenter_I : incenter I A B C)
variables (circumcenter_O : circumcenter O A B C)
variables (orthocenter_H : orthocenter H A B C)
variables (point_D_on_BC : lies_on D (line B C)) (point_E_on_CA : lies_on E (line C A))
variables (AE_eq_BD : AE = BD) (CD_plus_CE_eq_AB : CD + CE = AB)
variables (intersection_K : intersection K (line B E) (line A D))

-- To prove
theorem KH_parallel_IO (a b c : ℝ) (A B C D E K I O H : Type) 
  [acute_triangle A B C] 
  [sides_lengths A B C a b c] 
  (a_gt_b : a > b) (b_gt_c : b > c) 
  (incenter_I : incenter I A B C) 
  (circumcenter_O : circumcenter O A B C) 
  (orthocenter_H : orthocenter H A B C)
  (point_D_on_BC : lies_on D (line B C)) 
  (point_E_on_CA : lies_on E (line C A)) 
  (AE_eq_BD : AE = BD) 
  (CD_plus_CE_eq_AB : CD + CE = AB) 
  (intersection_K : intersection K (line B E) (line A D)):
  parallel (segment K H) (segment I O) ∧ length (segment K H) = 2 * length (segment I O) :=
sorry

end KH_parallel_IO_l754_754941


namespace baker_initial_cakes_cannot_be_determined_l754_754756

theorem baker_initial_cakes_cannot_be_determined (initial_pastries sold_cakes sold_pastries remaining_pastries : ℕ)
  (h1 : initial_pastries = 148)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : sold_pastries + remaining_pastries = initial_pastries) :
  True :=
by
  sorry

end baker_initial_cakes_cannot_be_determined_l754_754756


namespace impact_of_egg_price_decrease_l754_754346

theorem impact_of_egg_price_decrease (
    average_price_yuan_per_kg : ℝ,
    decrease_rate_weekly : ℝ,
    decrease_rate_yearly : ℝ,
    price_constant : ℝ,
    option1 : Prop,
    option2 : Prop,
    option3 : Prop,
    option4 : Prop)
    (h1 : average_price_yuan_per_kg = 8.03)
    (h2 : decrease_rate_weekly = 0.05)
    (h3 : decrease_rate_yearly = 0.093)
    (h4 : ∀ x, price_constant = x)
    (h5 : option1 ↔ (option1 = "Demand for eggs increases, production scale expands"))
    (h6 : option2 ↔ (option2 = "Output of cakes increases, market supply expands"))
    (h7 : option3 ↔ (option3 = "Production efficiency improves, economic benefits increase"))
    (h8 : option4 ↔ (option4 = "Product quality improves, enterprise competitiveness strengthens")) :
    (option1 ∧ option2) ∧ ¬ (option3 ∧ option4) :=
begin
  sorry
end

end impact_of_egg_price_decrease_l754_754346


namespace area_of_trapezoid_EFGH_l754_754199

/-- Define the vertices of the trapezoid -/
structure Point :=
(x : ℕ)
(y : ℕ)

def E : Point := { x := 0, y := 0 }
def F : Point := { x := 0, y := 3 }
def G : Point := { x := 5, y := 0 }
def H : Point := { x := 5, y := 7 }

def length (P Q : Point) : ℕ :=
if P.x = Q.x then abs (Q.y - P.y)
else if P.y = Q.y then abs (Q.x - P.x)
else 0 -- Since we only consider vertical or horizontal lengths

def EF_length : ℕ := length E F
def GH_length : ℕ := length G H
def height : ℕ := length E G

def area_trapezoid (a b h : ℕ) : ℕ :=
(a + b) * h / 2

/-- The area of the trapezoid EFGH --/
theorem area_of_trapezoid_EFGH : area_trapezoid EF_length GH_length height = 25 :=
by
  -- EF = 3, GH = 7, height = 5
  change area_trapezoid 3 7 5 = 25
  sorry

end area_of_trapezoid_EFGH_l754_754199


namespace exists_subset_with_conditions_l754_754100

theorem exists_subset_with_conditions 
  (n : ℕ) 
  (r : Fin n → ℝ) :
  ∃ S ⊆ (Fin n), 
  (∀ i : Fin (n-2), 1 ≤ ((fun {x | x ∈ S ∧ (x = i ∨ x = i+1 ∨ x = i+2)}).to_finset.card) ∧ ((fun {x | x ∈ S ∧ (x = i ∨ x = i+1 ∨ x = i+2)}).to_finset.card) ≤ 2) ∧
  abs ((∑ i in S.to_finset, r i)) ≥ (1/6) * (∑ i in Finset.univ, abs (r i)) :=
sorry

end exists_subset_with_conditions_l754_754100


namespace compare_negatives_l754_754274

theorem compare_negatives : -4 < -2.1 := 
sorry

end compare_negatives_l754_754274


namespace trigonometry_expression_zero_l754_754934

variable {r : ℝ} {A B C : ℝ}
variable (a b c : ℝ) (sinA sinB sinC : ℝ)

-- The conditions from the problem
axiom Law_of_Sines_a : a = 2 * r * sinA
axiom Law_of_Sines_b : b = 2 * r * sinB
axiom Law_of_Sines_c : c = 2 * r * sinC

-- The theorem statement
theorem trigonometry_expression_zero :
  a * (sinC - sinB) + b * (sinA - sinC) + c * (sinB - sinA) = 0 :=
by
  -- Skipping the proof
  sorry

end trigonometry_expression_zero_l754_754934


namespace shaded_area_concentric_circles_l754_754067

theorem shaded_area_concentric_circles (R : ℝ) (r : ℝ) (hR : π * R^2 = 100 * π) (hr : r = R / 2) :
  (1 / 2) * π * R^2 + (1 / 2) * π * r^2 = 62.5 * π :=
by
  -- Given conditions
  have R10 : R = 10 := sorry  -- Derived from hR
  have r5 : r = 5 := sorry    -- Derived from hr and R10
  -- Proof steps likely skipped
  sorry

end shaded_area_concentric_circles_l754_754067


namespace evaluate_expression_l754_754923

theorem evaluate_expression (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  rw [hx, hy]
  sorry

end evaluate_expression_l754_754923


namespace sine_cosine_ratio_l754_754424

open Real

theorem sine_cosine_ratio {m : ℝ} (h1 : m < 0) :
  let r := Real.sqrt ((4 * m)^2 + (-3 * m)^2),
  sin_alpha := -3 * m / -5 * m,
  cos_alpha := 4 * m / -5 * m in 
  (2 * sin_alpha + cos_alpha) / (sin_alpha - cos_alpha) = (2 / 7) :=
by
  let r := Real.sqrt ((4 * m)^2 + (-3 * m)^2)
  have h2 : r = -5 * m := sorry
  let sin_alpha := -3 * m / r
  let cos_alpha := 4 * m / r
  have h3 : sin_alpha = 3 / 5 := sorry
  have h4 : cos_alpha = -4 / 5 := sorry
  calc
    (2 * sin_alpha + cos_alpha) / (sin_alpha - cos_alpha)
        = (2 * (3 / 5) + (-4 / 5)) / ((3 / 5) - (-4 / 5)) : by rw [h3, h4]
    ... = ((6 / 5) - (4 / 5)) / ((3 / 5) + (4 / 5)) : by ring
    ... = (2 / 5) / (7 / 5) : by ring
    ... = 2 / 7 : by ring

end sine_cosine_ratio_l754_754424


namespace expression_evaluation_l754_754200

theorem expression_evaluation :
  (0.15)^3 - (0.06)^3 / (0.15)^2 + 0.009 + (0.06)^2 = 0.006375 :=
by
  sorry

end expression_evaluation_l754_754200


namespace average_headcount_correct_l754_754661

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end average_headcount_correct_l754_754661


namespace sequence_term_3001_exists_exactly_4_values_l754_754325

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l754_754325


namespace greatest_value_of_sum_l754_754455

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + 1 / x^2) : 
  ∃ y, y = x + 1/x ∧ y ≤ sqrt 15 :=
begin
  -- The proof will be here.
  sorry
end

end greatest_value_of_sum_l754_754455


namespace length_of_lawn_l754_754729

-- Given conditions
constant width_of_lawn : ℝ := 40
constant road_width : ℝ := 10
constant travelling_cost : ℝ := 3300
constant cost_per_sq_m : ℝ := 3

-- The length of the lawn (L) needs to be found such that:
theorem length_of_lawn (L : ℝ) : 
  10 * L + 10 * (L - 2 * road_width) = travelling_cost / cost_per_sq_m → 
  L = 65 :=
begin
  sorry
end

end length_of_lawn_l754_754729


namespace angle_BAD_is_60_l754_754585

variables {A B C D E F : Type} [plane_geometry A B C D]

-- Given conditions
-- E lies on segment AB such that AE = 5 * BE
-- F lies on segment BC such that BF = 5 * CF
-- DEF is an equilateral triangle

-- Define the condition for E and F
def is_on_segment (E : P) (A B : P) : Prop :=
∃ (x : ℝ), 0 < x ∧ y < 1 ∧ E = (x • A + (1 - x) • B)

def ratio_condition (AE BE : ℝ) : Prop := AE = 5 * BE ∧ E = x * A + (1 - x) * B

def ratio_condition (BF CF : ℝ) : Prop := BF = 5 * CF ∧ F = y * B + (1 - y) * C

-- Define def equilateral triangle DEF
def equilateral_triangle (D E F : P) : Prop :=
dist D E = dist E F ∧ dist E F = dist F D ∧ dist F D = dist D E

variables (hE : is_on_segment E A B) 
          (hF : is_on_segment F B C) 
          (h_eq_triangle : equilateral_triangle D E F)

-- Prove the angle BAD is 60 degrees
theorem angle_BAD_is_60 : ∃ (θ : ℝ), θ = 60 ∧ ∠BAD = θ := sorry

end angle_BAD_is_60_l754_754585


namespace time_to_pass_pole_l754_754741

def train_length : ℕ := 250
def platform_length : ℕ := 1250
def time_to_pass_platform : ℕ := 60

theorem time_to_pass_pole : 
  (train_length + platform_length) / time_to_pass_platform * train_length = 10 :=
by
  sorry

end time_to_pass_pole_l754_754741


namespace radius_of_inscribed_circle_in_quarter_circle_l754_754591

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ :=
  R * (Real.sqrt 2 - 1)

theorem radius_of_inscribed_circle_in_quarter_circle 
  (R : ℝ) (hR : R = 6) : inscribed_circle_radius R = 6 * Real.sqrt 2 - 6 :=
by
  rw [inscribed_circle_radius, hR]
  -- Apply the necessary simplifications and manipulations from the given solution steps here
  sorry

end radius_of_inscribed_circle_in_quarter_circle_l754_754591


namespace hex_B1C_base10_l754_754336

theorem hex_B1C_base10 : (11 * 16^2 + 1 * 16^1 + 12 * 16^0) = 2844 :=
by
  sorry

end hex_B1C_base10_l754_754336


namespace a_meets_b_a_meets_b_second_time_l754_754583

noncomputable def angular_velocity := ℤ

def initial_position (P : ℚ × ℚ) := P

def point_a := initial_position (1, 0)
def point_b := initial_position (1/2, real.sqrt 3 / 2)

def coordinate_intersection_a (t : ℝ) := 
  (cos (3 * t), sin (3 * t))

def coordinate_intersection_b (t : ℝ) := 
  (cos (t + real.pi / 3), sin (t + real.pi / 3))

theorem a_meets_b :
  ∃ t : ℝ, coordinate_intersection_a t = (0, 1) :=
sorry

theorem a_meets_b_second_time :
  ∃ t : ℝ, coordinate_intersection_a (t + real.pi) = (0, -1) :=
sorry

end a_meets_b_a_meets_b_second_time_l754_754583


namespace projection_matrix_correct_l754_754795

def projection_matrix (u : ℝ × ℝ) : vector (vector ℝ 2) :=
  let denom := (u.1^2 + u.2^2 : ℝ) in
  let scalar := λ x y : ℝ, (x * y) / denom in
  (2 : ℕ)![ (2 : ℕ)![ scalar u.1 u.1, scalar u.1 u.2], 
            (2 : ℕ)![ scalar u.2 u.1, scalar u.2 u.2 ] ]

def expected_matrix : vector (vector ℝ 2) :=
  (2 : ℕ)![ (2 : ℕ)![9 / 25, -12 / 25], 
            (2 : ℕ)![ -12 / 25, 16 / 25] ]
            
theorem projection_matrix_correct : projection_matrix (3, -4) = expected_matrix := 
  sorry

end projection_matrix_correct_l754_754795


namespace number_of_valid_codes_is_zero_l754_754347

theorem number_of_valid_codes_is_zero :
  ∀ (A B C D E : Fin 5), 
    B = 2 * A ∧ D = C / 2 →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E →
  0 = 0
:=
begin
  sorry,
end

end number_of_valid_codes_is_zero_l754_754347


namespace expression_evaluation_l754_754269

noncomputable def negative_number_power_zero : ℝ := -2023 ^ 0
noncomputable def sine_of_45_degrees : ℝ := 4 * Real.sin (Real.pi / 4)
noncomputable def absolute_value_sqrt_eight : ℝ := |Real.sqrt 8|

theorem expression_evaluation : negative_number_power_zero - sine_of_45_degrees + absolute_value_sqrt_eight = 1 := by
  dsimp [negative_number_power_zero, sine_of_45_degrees, absolute_value_sqrt_eight]
  rw [Real.pow_zero]
  norm_num
  sorry -- This is where the remaining steps would go to complete the proof

end expression_evaluation_l754_754269


namespace am_gm_inequality_l754_754556

theorem am_gm_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧ (x + y) / 2 = Real.sqrt (x * y) ↔ x = y := by
  sorry

end am_gm_inequality_l754_754556


namespace domain_function_l754_754766

def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def domain_of_f (f : ℝ → ℝ) : Set ℝ := 
  {x : ℝ | f x ≠ 0}

theorem domain_function :
  domain_of_f (λ x => 1 / (quadratic 1 (-8) 18 x).floor) = 
  set_of (λ x => x ≤ 1 ∨ 17 ≤ x) :=
by
  sorry

end domain_function_l754_754766


namespace cost_per_person_is_1400_l754_754337

-- Define the necessary conditions
def number_of_people : ℕ := 15
def airfare_and_hotel_cost : ℝ := 13500
def food_expenses : ℝ := 4500
def transportation_expenses : ℝ := 3000

-- Define the total cost per person
def total_cost_per_person := (airfare_and_hotel_cost + food_expenses + transportation_expenses) / number_of_people

-- Theorem to prove the total cost per person is $1400.00
theorem cost_per_person_is_1400 :
  total_cost_per_person = 1400 := by
  sorry

end cost_per_person_is_1400_l754_754337


namespace count_x_values_l754_754319

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l754_754319


namespace Micheal_Adam_work_together_l754_754570

theorem Micheal_Adam_work_together {W : ℝ} (h1 : W > 0) :
  let M := W / 199.99999999999983 in
  let A := (W - 11 * (M + ((W - 11 * (W / 199.99999999999983 + A)) / 10))) / 10 in
  let combined_rate := M + A in
  let days := W / combined_rate in
  days = 20 :=
by
  let M := W / 199.99999999999983
  let A := (W - 11 * (W / 199.99999999999983 + (W - 11 * (W / 199.99999999999983 + W * (9/200) / 21)) / 10)) / 10
  let combined_rate := M + A 
  let days := W / combined_rate
  sorry

end Micheal_Adam_work_together_l754_754570


namespace length_of_CD_l754_754948

theorem length_of_CD
  (A B C D : Type)
  (AB : ℝ) -- length of AB
  (A_midpoint : (A + B) / 2 = D)
  (right_triangle : ∠ACB = 90) :
  AB = 10 →  
  CD = 5 :=
sorry

end length_of_CD_l754_754948


namespace beach_problem_l754_754180

theorem beach_problem 
  (x : ℕ) 
  (h1 : ∑ n in {x, 20, 18} = 62 )
  (h2 : 62 - 5 - x = 54) :
  x = 3 :=
by
  sorry

end beach_problem_l754_754180


namespace integral_triangles_with_eq_area_perimeter_l754_754607

def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

def integer_sides_eq_area_perimeter (a b c : ℕ) : Bool :=
  let p := a + b + c
  let s := (p : ℝ) / 2
  let A := heron_area a b c
  A == p

theorem integral_triangles_with_eq_area_perimeter :
  ∀ (a b c : ℕ),
  integer_sides_eq_area_perimeter a b c = true →
  (a, b, c) = (6, 25, 29) ∨ (a, b, c) = (7, 15, 20) ∨ (a, b, c) = (9, 10, 17) ∨ (a, b, c) = (5, 12, 13) ∨ (a, b, c) = (6, 8, 10) :=
by
  sorry

end integral_triangles_with_eq_area_perimeter_l754_754607


namespace symmetry_implies_phi_value_l754_754009

theorem symmetry_implies_phi_value {φ : ℝ} (h0 : 0 < φ) (h1 : φ < π) 
  (h_sym : ∀ x: ℝ, sin(2 * x + φ) = sin(2 * (2 * π / 12 - x) + φ)) : φ = π / 6 :=
sorry

end symmetry_implies_phi_value_l754_754009


namespace triangle_proof_l754_754481

-- Given definitions and conditions
variables {AC BC DF EF AB hDE : ℝ} -- side lengths and height

-- Given conditions
def lengths_equal : AC = DF ∧ BC = EF := sorry

def ab_twice_height : AB = 2 * hDE := sorry

-- Theorem statement for angles and area
theorem triangle_proof :
  ∀ (AC = DF) (BC = EF) (AB = 2 * hDE),
  (let ∠ACB = ∡ACB in
   let ∠DFE = ∡DFE in
   ∠ACB + ∠DFE = π ∧ 2 * (Area DEF) = Area ABC) := sorry

end triangle_proof_l754_754481


namespace monotonic_intervals_and_value_of_a_l754_754008

def f (x a : ℝ) : ℝ := -(1/3) * x^3 + x^2 + 3 * x + a

theorem monotonic_intervals_and_value_of_a :
  (∀ x y a : ℝ, x ∈ Ioo (-1) 3 → y ∈ Ioo (-1) 3 → f x a < f y a → x < y)
  ∧ (∀ x y a : ℝ, x ∈ Iic (-1) → y ∈ Iic (-1) → f x a < f y a → x > y)
  ∧ (∀ x y a : ℝ, x ∈ Ici 3 → y ∈ Ici 3 → f x a < f y a → x > y)
  ∧ (∀ a : ℝ, (∃ x : ℝ, x ∈ Icc (-3) 3 ∧ f x a = 7/3) → a = 13/3) :=
sorry

end monotonic_intervals_and_value_of_a_l754_754008


namespace quantities_change_as_P_moves_perpendicularly_l754_754125

-- Define the points, triangle, midpoints, and conditions.
variables (A B P : Type) [point P] [on_line P A B] (M N : midpoint P A) (M N : midpoint P B) 

-- Define the properties.
variables (A B P : EuclideanGeometry.Point)
variables [EuclideanGeometry.is_midpoint M P A] 
variables [EuclideanGeometry.is_midpoint N P B]
variables { hP : EuclideanGeometry.perpendicular P A B}

-- Theorem
theorem quantities_change_as_P_moves_perpendicularly :
  let mn_length_unchanged := EuclideanGeometry.M.midpoint_length M N,
      perimeter_changes := EuclideanGeometry.perimeter_change P A B,
      area_triangle_changes := EuclideanGeometry.area_triangle_change P A B,
      area_trapezoid_changes := EuclideanGeometry.area_trapezoid_change A B N M P,
      total_changes := if mn_length_unchanged then 0 + (if perimeter_changes then 1 else 0) + (if area_triangle_changes then 1 else 0) + (if area_trapezoid_changes then 1 else 0) else 1 in
  total_changes = 3 :=
sorry

end quantities_change_as_P_moves_perpendicularly_l754_754125


namespace Vasya_not_11_more_than_Kolya_l754_754084

def is_L_shaped (n : ℕ) : Prop :=
  n % 2 = 1

def total_cells : ℕ :=
  14400

theorem Vasya_not_11_more_than_Kolya (k v : ℕ) :
  (is_L_shaped k) → (is_L_shaped v) → (k + v = total_cells) → (k % 2 = 0) → (v % 2 = 0) → (v - k ≠ 11) := 
by
  sorry

end Vasya_not_11_more_than_Kolya_l754_754084


namespace parallel_vectors_l754_754445

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (P : a = (1, m) ∧ b = (m, 2) ∧ (a.1 / m = b.1 / 2)) :
  m = -Real.sqrt 2 ∨ m = Real.sqrt 2 :=
by
  sorry

end parallel_vectors_l754_754445


namespace curve_polar_eq_l754_754501

theorem curve_polar_eq (α θ ρ: ℝ) : 
  (x = 1 + 2 * cos α ∧ y = 2 * sin α ∧ (θ = π / 4 ∧ ρ ∈ ℝ)) →
   (ρ^2 - 2*ρ*cos θ - 3 = 0 ∧ y = x ∧ (|AB| = sqrt 14)) :=
by sorry

end curve_polar_eq_l754_754501


namespace total_toys_correct_l754_754533

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l754_754533


namespace trisha_take_home_pay_l754_754654

theorem trisha_take_home_pay
  (hourly_pay : ℝ := 15)
  (hours_per_week : ℝ := 40)
  (weeks_per_year : ℝ := 52)
  (withholding_percentage : ℝ := 0.20) :
  let annual_gross_pay := hourly_pay * hours_per_week * weeks_per_year,
      amount_withheld := annual_gross_pay * withholding_percentage,
      annual_take_home_pay := annual_gross_pay - amount_withheld
  in annual_take_home_pay = 24960 := by
    sorry

end trisha_take_home_pay_l754_754654


namespace total_toys_l754_754541

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end total_toys_l754_754541


namespace comb_identity_l754_754621

theorem comb_identity (n : ℕ) : 
    ∑ k in Finset.range (n / 2 + 1), (Nat.choose n (2 * k)) * 2 ^ (n - 2 * k) * (Nat.choose (2 * k) k) = 1 := 
by
    sorry

end comb_identity_l754_754621


namespace transformations_map_figure_l754_754290

def transformation_maps_figure_onto_itself (figure : Type) (l : line) (rotate : ℝ) (translate : ℝ) (reflect_l : ℝ) (reflect_perpendicular : ℝ) : Prop :=
  ((∃ point : ℝ, rotation_around_point figure l rotate) ∧
  (translation_parallel_to_line figure l translate) ∧
  (reflection_across_line figure l reflect_l) ∧
  (reflection_across_perpendicular_line figure l reflect_perpendicular))

theorem transformations_map_figure (figure : Type) (l : line) (rotate : ℝ) (translate : ℝ) (reflect_l : ℝ) (reflect_perpendicular : ℝ) : Prop :=
  transformation_maps_figure_onto_itself figure l rotate translate reflect_l reflect_perpendicular := by
  sorry

end transformations_map_figure_l754_754290


namespace shaded_figure_area_l754_754368

noncomputable def shaded_area_of_rotating_semicircle (R : ℝ) : ℝ :=
  2 * π * R^2 / 3

theorem shaded_figure_area 
  (R : ℝ) 
  (α : ℝ) 
  (hα : α = π / 3) : 
  shaded_area_of_rotating_semicircle R = 2 * π * R^2 / 3 := 
by 
  sorry

end shaded_figure_area_l754_754368


namespace tileable_if_and_only_if_l754_754775

def is_tileable (n : ℕ) : Prop :=
  ∃ k : ℕ, 15 * n = 4 * k

theorem tileable_if_and_only_if (n : ℕ) :
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) ↔ is_tileable n :=
sorry

end tileable_if_and_only_if_l754_754775


namespace find_angle_l754_754061

noncomputable def α := Real.arccos (-(1/3))

variables (OA OB OC OD : ℝ^3) (h_non_collinear : ¬ (collinear OA OB OC OD))
variables (h_angle : ∀ (v1 v2 : ℝ^3), (v1 ≠ v2) → (v1 = OA ∨ v1 = OB ∨ v1 = OC ∨ v1 = OD) → (v2 = OA ∨ v2 = OB ∨ v2 = OC ∨ v2 = OD) → (Real.angle v1 v2 = α))

theorem find_angle : α = Real.arccos (-(1/3)) :=
sorry

end find_angle_l754_754061


namespace extreme_value_derivative_zero_l754_754699

theorem extreme_value_derivative_zero (f : ℝ → ℝ) (x₀ : ℝ) (h_extreme : ∃ x₀, IsLocalMin f x₀ ∨ IsLocalMax f x₀) :
  f' x₀ = 0 :=
by
  sorry

end extreme_value_derivative_zero_l754_754699


namespace simplify_trig_l754_754137

theorem simplify_trig (h : (Real.pi / 2) < 2 ∧ 2 < (3 * Real.pi / 4)) :
  sqrt (1 + Real.sin 4) + sqrt (1 - Real.sin 4) = 2 * Real.sin 2 :=
sorry

end simplify_trig_l754_754137


namespace f_prime_at_2_l754_754401

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x * f' 2

-- State the problem as a theorem to be proved
theorem f_prime_at_2 (f : ℝ → ℝ) (f' : ℝ → ℝ) (f_def : ∀ x, f x = 3 * x^2 - 2 * x * f' 2) :
  f' 2 = 4 :=
sorry

end f_prime_at_2_l754_754401


namespace num_x_for_3001_in_sequence_l754_754304

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l754_754304


namespace john_total_spent_l754_754974

theorem john_total_spent (vacuum_cost dishwasher_cost coupon : ℕ) :
  vacuum_cost = 250 →
  dishwasher_cost = 450 →
  coupon = 75 →
  (vacuum_cost + dishwasher_cost - coupon) = 625 :=
by
  intros h_vacuum h_dishwasher h_coupon
  rw [h_vacuum, h_dishwasher, h_coupon]
  norm_num
  sorry

end john_total_spent_l754_754974


namespace security_of_oil_platform_l754_754804

-- Define the given constants.
def num_radars : ℕ := 7
def radar_radius : ℝ := 26
def ring_width : ℝ := 20

-- Distance from the center of the platform to the position of the radars.
def max_distance_to_radars := 24 / Real.sin (Real.pi / 7)

-- Area of the coverage ring around the platform.
def coverage_ring_area := (960 * Real.pi) / Real.tan (Real.pi / 7)

theorem security_of_oil_platform :
  True := sorry

end security_of_oil_platform_l754_754804


namespace cos_difference_l754_754866

theorem cos_difference (α β : ℝ) (h_α_acute : 0 < α ∧ α < π / 2)
                      (h_β_acute : 0 < β ∧ β < π / 2)
                      (h_cos_α : Real.cos α = 1 / 3)
                      (h_cos_sum : Real.cos (α + β) = -1 / 3) :
  Real.cos (α - β) = 23 / 27 := 
sorry

end cos_difference_l754_754866


namespace find_x_l754_754469

theorem find_x (x : ℕ) : (x % 7 = 0) ∧ (x^2 > 200) ∧ (x < 30) ↔ (x = 21 ∨ x = 28) :=
by
  sorry

end find_x_l754_754469


namespace angle_ABC_is_60_l754_754118

noncomputable def triangle_abc (A B C D E P : Type*) := 
  ∃ (angle_ABC : ℝ),
    angle_ABC = 60 ∧
    ∃ (AD DE EC : ℝ),
      AD = DE ∧ DE = EC ∧ AE ≠ DC ∧ 
      P ∈ line.ofPoints A E ∧
      P ∈ line.ofPoints D C ∧
      dist A P = dist C P

theorem angle_ABC_is_60 (A B C D E P : Type*) (h : triangle_abc A B C D E P) :
  ∃ (angle_ABC : ℝ), angle_ABC = 60 :=
by sorry

end angle_ABC_is_60_l754_754118


namespace chocolate_discount_l754_754612

theorem chocolate_discount :
    let original_cost : ℝ := 2
    let final_price : ℝ := 1.43
    let discount := original_cost - final_price
    discount = 0.57 := by
  sorry

end chocolate_discount_l754_754612


namespace find_T9_l754_754628

-- Define the conditions and the given geometric sequence
variable {a : ℕ → ℝ}
variable {T_n : ℕ → ℝ}
variable {q : ℝ}

-- Condition: The product of the first n terms of a geometric sequence {a_n} is T_n 
def geometric_sequence_product (n : ℕ) : Prop :=
  T_n n = ∏ i in finset.range n, a i

-- Condition: 2a_3 = a_4^2
def condition (a : ℕ → ℝ) : Prop :=
  2 * a 3 = (a 4)^2

-- Given the conditions, prove that T_9 = 512
theorem find_T9 (h1 : geometric_sequence_product 9) (h2 : condition a) (h3 : a 5 = 2) :
  T_n 9 = 512 :=
sorry

end find_T9_l754_754628


namespace f_f_f_3_l754_754464

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_f_f_3 : f(f(f(3))) = 107 := 
by
  sorry

end f_f_f_3_l754_754464


namespace find_common_ratio_l754_754879

variable {α : Type*} [LinearOrderedField α] [NormedLinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop := ∀ n, a (n+1) = q * a n

def sum_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop := ∀ n, S n = (Finset.range n).sum a

theorem find_common_ratio
  (a : ℕ → α)
  (S : ℕ → α)
  (q : α)
  (pos_terms : ∀ n, 0 < a n)
  (geometric_seq : geometric_sequence a q)
  (sum_eq : sum_first_n_terms a S)
  (eqn : S 1 + 2 * S 5 = 3 * S 3) :
  q = (2:α)^(3 / 2) / 2^(3 / 2) :=
by
  sorry

end find_common_ratio_l754_754879


namespace find_number_l754_754919

theorem find_number :
  ∃ x : ℚ, x * (-1/2) = 1 ↔ x = -2 := 
sorry

end find_number_l754_754919


namespace hourly_wage_difference_l754_754690

theorem hourly_wage_difference (P Q: ℝ) (H_p: ℝ) (H_q: ℝ) (h1: P = 1.5 * Q) (h2: H_q = H_p + 10) (h3: P * H_p = 420) (h4: Q * H_q = 420) : P - Q = 7 := by
  sorry

end hourly_wage_difference_l754_754690


namespace num_digits_1024_base_8_l754_754023

-- Define a function that finds the number of digits in the base-8 representation of a given number.
def num_digits_base_8 (n : ℕ) : ℕ :=
  Nat.ceil (Real.log n / Real.log 8)

-- The main theorem which states that the number of digits in the base-8 representation of 1024 is 4.
theorem num_digits_1024_base_8 : num_digits_base_8 1024 = 4 := 
by 
  sorry

end num_digits_1024_base_8_l754_754023


namespace compare_negatives_l754_754281

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l754_754281


namespace fixed_point_l754_754160

-- Conditions
variables (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)

-- Function definition
def f (x : ℝ) : ℝ := 1 + 2 * a ^ (x - 1)

-- Lean 4 statement for the mathematically equivalent proof problem
theorem fixed_point : (f a 1 h1 h2) = (1, 3) :=
sorry

end fixed_point_l754_754160


namespace second_car_catch_up_l754_754190

/--
Two cars start from the same point and travel along a straight road.
1. The first car travels at a constant speed of 60 km/h.
2. The second car travels at a constant speed of 80 km/h.
3. The second car starts 30 minutes after the first car.
Prove that the second car will catch up to the first car in 1.5 hours.
-/
theorem second_car_catch_up 
    (start_same_point : ∀ x : ℝ, x ∈ ℝ)
    (speed_first_car : ℝ := 60)
    (speed_second_car : ℝ := 80)
    (delay_second_car : ℝ := 0.5) :
    ∃ t : ℝ , t = 1.5 := 
sorry

end second_car_catch_up_l754_754190


namespace sum_of_first_2n_terms_l754_754633

-- Definitions based on conditions
variable (n : ℕ) (S : ℕ → ℝ)

-- Conditions
def condition1 : Prop := S n = 24
def condition2 : Prop := S (3 * n) = 42

-- Statement to be proved
theorem sum_of_first_2n_terms {n : ℕ} (S : ℕ → ℝ) 
    (h1 : S n = 24) (h2 : S (3 * n) = 42) : S (2 * n) = 36 := by
  sorry

end sum_of_first_2n_terms_l754_754633


namespace calc_z_conj_diff_l754_754833

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754833


namespace musical_chairs_l754_754195

def is_prime_power (m : ℕ) : Prop :=
  ∃ (p k : ℕ), Nat.Prime p ∧ k > 0 ∧ m = p ^ k

theorem musical_chairs (n m : ℕ) (h1 : 1 < m) (h2 : m ≤ n) (h3 : ¬ is_prime_power m) :
  ∃ f : Fin n → Fin n, (∀ x, f x ≠ x) ∧ (∀ x, (f^[m]) x = x) :=
sorry

end musical_chairs_l754_754195


namespace trisha_take_home_pay_l754_754655

theorem trisha_take_home_pay
  (hourly_pay : ℝ := 15)
  (hours_per_week : ℝ := 40)
  (weeks_per_year : ℝ := 52)
  (withholding_percentage : ℝ := 0.20) :
  let annual_gross_pay := hourly_pay * hours_per_week * weeks_per_year,
      amount_withheld := annual_gross_pay * withholding_percentage,
      annual_take_home_pay := annual_gross_pay - amount_withheld
  in annual_take_home_pay = 24960 := by
    sorry

end trisha_take_home_pay_l754_754655


namespace relationship_nearsighted_electronics_prob_nearsighted_less_device_use_prob_two_out_of_five_nearsighted_l754_754060

noncomputable def K_squared (a b c d : ℕ) : ℝ :=
  let n := (a + b + c + d : ℕ) in
  n * ((a * d - b * c) ^ 2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem relationship_nearsighted_electronics : 
  let a := 35
  let b := 5
  let c := 5
  let d := 5
  let critical_value := 6.635
  K_squared a b c d > critical_value :=
by
  let a := 35
  let b := 5
  let c := 5
  let d := 5
  let critical_value := 6.635
  calc 
    K_squared a b c d = 7.031 : sorry
    7.031 > 6.635 : by norm_num

theorem prob_nearsighted_less_device_use :
  let total_nearsighted := 40
  let not_more_1hr := 5
  (not_more_1hr : ℚ) / total_nearsighted = 1 / 8 :=
by norm_num

theorem prob_two_out_of_five_nearsighted :
  let p_nearsighted := 4/5
  let p_not_nearsighted := 1/5
  let n := 5
  let k := 2
  (nat.choose n k * p_nearsighted ^ k * p_not_nearsighted ^ (n - k) : ℚ) = 32 / 625 :=
by norm_num

end relationship_nearsighted_electronics_prob_nearsighted_less_device_use_prob_two_out_of_five_nearsighted_l754_754060


namespace valid_votes_for_A_l754_754944

open Nat

noncomputable def valid_votes (total_votes : ℕ) (invalid_percentage : ℤ) : ℕ :=
  total_votes * (100 - invalid_percentage.toNat) / 100

noncomputable def votes_for_candidate (valid_votes : ℕ) (candidate_percentage : ℤ) : ℕ :=
  valid_votes * candidate_percentage.toNat / 100

theorem valid_votes_for_A (total_votes : ℕ)
  (invalid_percentage : ℤ) (candidate_percentage : ℤ) (valid_votes_for_A : ℕ) :
  total_votes = 560000 → invalid_percentage = 15 →
  candidate_percentage = 75 → valid_votes_for_A = 357000 :=
by
  intros h_total_votes h_invalid_percentage h_candidate_percentage
  have h_valid_votes := valid_votes total_votes invalid_percentage
  have h_votes_for_A := votes_for_candidate h_valid_votes candidate_percentage
  rw [h_total_votes, h_invalid_percentage, h_candidate_percentage] at h_valid_votes h_votes_for_A
  simp only [valid_votes] at h_valid_votes
  simp only [votes_for_candidate] at h_votes_for_A
  rw [show (560000 * 85 / 100) = 476000, from sorry] at h_valid_votes
  rw [show (476000 * 75 / 100) = 357000, from sorry] at h_votes_for_A
  exact h_votes_for_A

end valid_votes_for_A_l754_754944


namespace reservoir_water_percentage_l754_754259

variables (C : ℝ) (normal_level : ℝ) (end_of_month_water : ℝ) (percentage : ℝ)

-- Definitions derived from conditions in part a)
def total_capacity := C
def normal_level_water := total_capacity - 10
def end_of_month_water_amount := 14

-- Condition 2: This amount is twice the normal level
def condition_2 := end_of_month_water_amount = 2 * normal_level_water

-- Condition 3: The normal level is 10 million gallons short of total capacity
def condition_3 := normal_level_water = total_capacity - 10

-- Define the percentage calculation
def calculate_percentage (amount capacity : ℝ) : ℝ := (amount / capacity) * 100

-- The goal is to prove the question given the conditions:
theorem reservoir_water_percentage
    (C : ℝ) (hC : 2 * (C - 10) = 14) :
    calculate_percentage 14 C = 82.35 :=
by sorry

end reservoir_water_percentage_l754_754259


namespace greatest_difference_in_set_B_l754_754640

theorem greatest_difference_in_set_B :
  ∃ (A B : Finset ℕ) (M N : ℕ),
    (A.card = 8) ∧ (B.card = 8) ∧
    (A.sum = 39) ∧ (B.sum = 39) ∧
    (∀ (a : ℕ), a ∈ A → a > 0) ∧
    (∀ (b : ℕ), b ∈ B → b > 0) ∧
    (Set.Pairwise B (≠)) ∧
    (M = 32) ∧ (N = 11) ∧
    (M - N = 21) := sorry

end greatest_difference_in_set_B_l754_754640


namespace function_intersects_x_axis_exactly_two_points_l754_754891

def piecewise_func (x : ℝ) (m : ℝ) :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * x ^ 3 + 3 * x ^ 2 + m else m * x + 5

theorem function_intersects_x_axis_exactly_two_points (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ piecewise_func x₁ m = 0 ∧ piecewise_func x₂ m = 0) ↔ m ∈ Set.Ioo (-5) 0 :=
sorry

end function_intersects_x_axis_exactly_two_points_l754_754891


namespace mirror_side_length_l754_754245

theorem mirror_side_length
  (width_wall : ℝ)
  (length_wall : ℝ)
  (area_wall : ℝ)
  (area_mirror : ℝ)
  (side_length_mirror : ℝ)
  (h1 : width_wall = 32)
  (h2 : length_wall = 20.25)
  (h3 : area_wall = width_wall * length_wall)
  (h4 : area_mirror = area_wall / 2)
  (h5 : side_length_mirror * side_length_mirror = area_mirror)
  : side_length_mirror = 18 := by
  sorry

end mirror_side_length_l754_754245


namespace train_speed_l754_754249

noncomputable def length_meters := 140
noncomputable def time_seconds := 4.666293363197611

noncomputable def length_kilometers := length_meters / 1000
noncomputable def time_hours := time_seconds / 3600

theorem train_speed :
  let speed_km_per_hr := length_kilometers / time_hours in
  speed_km_per_hr ≈ 108 :=
by 
  sorry

end train_speed_l754_754249


namespace point_in_quadrant_2_l754_754502

def point : ℝ × ℝ := (-1, 2)

def is_positive (a : ℝ) : Prop := a > 0
def is_negative (a : ℝ) : Prop := a < 0

def quadrant (p : ℝ × ℝ) : String :=
  if is_positive p.1 ∧ is_positive p.2 then "First"
  else if is_negative p.1 ∧ is_positive p.2 then "Second"
  else if is_negative p.1 ∧ is_negative p.2 then "Third"
  else if is_positive p.1 ∧ is_negative p.2 then "Fourth"
  else "Origin"

theorem point_in_quadrant_2 : quadrant point = "Second" := by
  sorry

end point_in_quadrant_2_l754_754502


namespace james_earnings_l754_754530

-- Define the conditions
def rain_gallons_per_inch : ℕ := 15
def rain_monday : ℕ := 4
def rain_tuesday : ℕ := 3
def price_per_gallon : ℝ := 1.2

-- State the theorem to be proved
theorem james_earnings : (rain_monday * rain_gallons_per_inch + rain_tuesday * rain_gallons_per_inch) * price_per_gallon = 126 :=
by
  sorry

end james_earnings_l754_754530


namespace tan_subtraction_l754_754925

theorem tan_subtraction (α β : ℝ) (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_subtraction_l754_754925


namespace complex_z_sub_conjugate_eq_neg_i_l754_754842

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754842


namespace arithmetic_mean_solution_l754_754762

-- Define the Arithmetic Mean statement
theorem arithmetic_mean_solution (x : ℝ) (h : (x + 5 + 17 + 3 * x + 11 + 3 * x + 6) / 5 = 19) : 
  x = 8 :=
by
  sorry -- Proof is not required as per the instructions

end arithmetic_mean_solution_l754_754762


namespace coloring_of_triangle_l754_754625

theorem coloring_of_triangle
  (K P S A B C D E F : Type)
  (color : Type)
  (blue red yellow : color)
  (c : K → color)
  (pK : c K = blue)
  (pP : c P = red)
  (pS : c S = yellow)
  (colors_midpoint_match : ∀ (X Y : Type), (c (midpoint X Y) = c X ∨ c (midpoint X Y) = c Y)) :
  ∃ (t : triangle), vertices_different_colors t :=
sorry

end coloring_of_triangle_l754_754625


namespace four_possible_x_values_l754_754295

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l754_754295


namespace variance_equal_x_l754_754932

theorem variance_equal_x (x : ℝ) : 
  let s1 := [2, 3, 4, 5, x] in
  let s2 := [101, 102, 103, 104, 105] in
  ∀ (var : List ℝ → ℝ), var s1 = var s2 → (x = 6 ∨ x = 1) :=
by
  sorry

end variance_equal_x_l754_754932


namespace digit_D_value_l754_754165

/- The main conditions are:
1. A, B, C, D are digits (0 through 9)
2. Addition equation: AB + CA = D0
3. Subtraction equation: AB - CA = 00
-/

theorem digit_D_value (A B C D : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (hD : D < 10)
  (add_eq : 10 * A + B + 10 * C + A = 10 * D + 0)
  (sub_eq : 10 * A + B - (10 * C + A) = 0) :
  D = 1 :=
sorry

end digit_D_value_l754_754165


namespace num_x_for_3001_in_sequence_l754_754301

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l754_754301


namespace density_of_seq_l754_754702

noncomputable def is_dense {α : Type*} [topological_space α] (s : set α) : Prop :=
∀ x : α, ∀ U ∈ nhds x, s ∩ U ≠ ∅

noncomputable def seq_not_all_diff (f : ℝ → ℝ) : set ℝ :=
{x | ∃ (n : ℕ), ∀ (m : ℕ), x ≠ (f^[n]) (f^[m] x) }

variables {f : ℝ → ℝ}

noncomputable def is_convex (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ (x y : ℝ) (t : ℝ), x ∈ I → y ∈ I → t ∈ I → f (t * x + (1-t) * y) ≤ t * f x + (1-t) * f y

theorem density_of_seq 
  (h_convex : is_convex f (set.Icc 0 (1/2))) 
  (h_f_0 : f 0 = 0)
  (h_f_half : f (1/2) = 1)
  (h_differentiable_middle : ∀ x ∈ set.Ioo 0 (1/2), differentiable_at ℝ f x)
  (h_differentiable_right : differentiable_within_at ℝ f set.Ici 0 0)
  (h_differentiable_left : differentiable_within_at ℝ f set.Iic (1/2) (1/2))
  (h_f_prime_gt_1 : 1 < deriv f 0)
  (h_extension : ∀ x ∈ set.Ioo (1/2) 1, f x = f (1-x)) :
  is_dense (seq_not_all_diff f) (set.Icc 0 1) :=
sorry

end density_of_seq_l754_754702


namespace system_solution_unique_l754_754366

theorem system_solution_unique (w x y z : ℝ) (h1 : w + x + y + z = 12)
  (h2 : w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27) :
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := 
sorry

end system_solution_unique_l754_754366


namespace limit_log3_div_tan_pi_eq_l754_754767

noncomputable def log_base_3 (x : ℝ) : ℝ := log x / log 3

noncomputable def limit_expr (x : ℝ) : ℝ := (log_base_3 x - 1) / tan (π * x)

theorem limit_log3_div_tan_pi_eq :
  tendsto (λ x, limit_expr x) (nhds 3) (nhds (1 / (3 * π * log 3))) :=
sorry

end limit_log3_div_tan_pi_eq_l754_754767


namespace proof_of_problem_statement_l754_754860

noncomputable def problem_statement : Prop :=
  ∀ (k : ℝ) (m : ℝ),
    (0 < m ∧ m < 3/2) → 
    (-3/(4 * m) = k) → 
    (k < -1/2)

theorem proof_of_problem_statement : problem_statement :=
  sorry

end proof_of_problem_statement_l754_754860


namespace complex_z_sub_conjugate_eq_neg_i_l754_754847

def complex_z : ℂ := (1 - complex.i) / (2 + 2 * complex.i)

theorem complex_z_sub_conjugate_eq_neg_i : complex_z - conj complex_z = -complex.i := by
sorry

end complex_z_sub_conjugate_eq_neg_i_l754_754847


namespace cardinality_A_is_3_l754_754438

open Real

-- Define the set A 
def A : Set ℝ := {x | ∃ k : ℕ, k ≤ 4 ∧ x = sin (k * π / 4)}

-- Statement to prove the cardinality of the set A
theorem cardinality_A_is_3 : (A.to_finset.card = 3) :=
sorry

end cardinality_A_is_3_l754_754438


namespace total_volume_snowballs_l754_754565

-- Define the radii of the snowballs
def r1 := 4
def r2 := 6
def r3 := 3
def r4 := 7

-- Define the formula for the volume of a sphere
def volume (r : ℕ) := (4/3:ℚ) * Real.pi * r^3

-- Prove the total volume of four snowballs
theorem total_volume_snowballs :
  volume r1 + volume r2 + volume r3 + volume r4 = (2600/3:ℚ) * Real.pi :=
by
  sorry

end total_volume_snowballs_l754_754565


namespace students_with_both_l754_754602

-- Definitions and Conditions
def total_students : Finset ℕ := {n | n ∈ Finset.range 30}
def students_with_glasses : Finset ℕ := {1, 3, 7, 10, 23, 27}
def students_with_hair_tied : Finset ℕ := {1, 9, 11, 20, 23}

-- Theorem statement
theorem students_with_both : students_with_glasses ∩ students_with_hair_tied = {1, 23} :=
sorry

end students_with_both_l754_754602


namespace count_x_values_l754_754313

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l754_754313


namespace find_angle_A_find_area_S_l754_754935

-- (1) Prove that angle A is pi/4 given the condition in problem
theorem find_angle_A
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (h₄ : A > 0 ∧ A < π)
  (h₅ : B > 0 ∧ B < π)
  (h₆ : C > 0 ∧ C < π)
  (h₇ : A + B + C = π)
  (h_condition : (sqrt 2 * b - c) / a = cos C / cos A) :
  A = π / 4 :=
by
  sorry

-- (2) Prove the area S given specific sides and angles
theorem find_area_S
  (a b c A B C : ℝ)
  (h₁ : a = 10)
  (h₂ : b = 8 * sqrt 2)
  (h₃ : c ∈ {14, 2}) -- c will be determined in the context
  (h₄ : A = π / 4)
  (h₅ : sin C > 0) -- since C is the smallest angle
  (h₆ : A + B + C = π) :
  (1 / 2) * b * c * sin A = 8 :=
by
  sorry

end find_angle_A_find_area_S_l754_754935


namespace problem_C_l754_754550

def g (a b : ℝ) := a * Real.sqrt b - (1 / 4) * b

theorem problem_C :
  ∃ a, a ≥ 1 ∧ ∀ b, b > 0 → g a 4 ≥ g a b :=
sorry

end problem_C_l754_754550


namespace count_valid_subsets_l754_754025

-- Definitions for the problem conditions
def subset_constraints (T : Finset ℕ) : Prop := 
  ∃ (k : ℕ), k = T.card ∧ 
  (∀ (x : ℕ), x ∈ T → k ≤ x) ∧ 
  (∀ (x y : ℕ), x ∈ T → y ∈ T → x ≠ y + 1 ∧ x ≠ y - 1)

-- Main proof statement
theorem count_valid_subsets : 
  (Finset.filter subset_constraints (Finset.powerset (Finset.range 21))).card = 2765 :=
  sorry

end count_valid_subsets_l754_754025


namespace distance_from_origin_l754_754723

noncomputable def m : ℝ :=
  let x := 1 + 4 * Real.sqrt 7 in
  let y := 9 in
  Real.sqrt ((x ^ 2) + (y ^ 2))

theorem distance_from_origin :
  ∀ (x y : ℝ), y = 9 ∧ (Real.sqrt ((x - 1) ^ 2 + (9 - 6) ^ 2) = 11) ∧ (x > 1) →
  m = Real.sqrt (194 + 32 * Real.sqrt 7) :=
by
  intros x y h
  exact sorry

end distance_from_origin_l754_754723


namespace angle_BDC_10_l754_754144

-- Definitions taken directly from conditions
variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable (triangle_ABC : Triangle A B C)
variable (triangle_ACD : Triangle A C D)

-- Congruence of triangles
variable (congruent_ABC_ACD : Congruent triangle_ABC triangle_ACD)

-- Equal sides and angle condition
axiom AB_eq_AC : distance (A, B) = distance (A, C)
axiom AC_eq_AD : distance (A, C) = distance (A, D)
axiom angle_BAC_20 : angle A B C = 20

-- Final proof statement
theorem angle_BDC_10 : angle B D C = 10 := by
  sorry

end angle_BDC_10_l754_754144


namespace probability_of_yellow_face_l754_754145

def total_faces : ℕ := 12
def red_faces : ℕ := 5
def yellow_faces : ℕ := 4
def blue_faces : ℕ := 2
def green_faces : ℕ := 1

theorem probability_of_yellow_face : (yellow_faces : ℚ) / (total_faces : ℚ) = 1 / 3 := by
  sorry

end probability_of_yellow_face_l754_754145


namespace initial_fraction_brown_hats_l754_754239

variable (H : ℕ) -- Number of hats
variable (B : ℝ) -- Fraction of brown hats
variable (frac_sold : ℝ) := 2 / 3 -- Fraction of hats sold
variable (brown_sold : ℝ) := 4 / 5 -- Fraction of brown hats sold
variable (brown_unsold_frac : ℝ) := 0.15 -- Fraction of unsold hats that are brown

theorem initial_fraction_brown_hats :
  (H ≠ 0) →
  (1 / 5 * B * H) / (1 / 3 * H) = brown_unsold_frac → 
  B = 1 / 4 :=
by
  intros hH h
  sorry

end initial_fraction_brown_hats_l754_754239


namespace vasya_kolya_difference_impossible_l754_754085

theorem vasya_kolya_difference_impossible : 
  ∀ k v : ℕ, (∃ q₁ q₂ : ℕ, 14400 = q₁ * 2 + q₂ * 2 + 1 + 1) → ¬ ∃ k, ∃ v, (v - k = 11 ∧ 14400 = k * q₁ + v * q₂) :=
by sorry

end vasya_kolya_difference_impossible_l754_754085


namespace find_x_given_scores_l754_754416

theorem find_x_given_scores : 
  ∃ x : ℝ, (9.1 + 9.3 + x + 9.2 + 9.4) / 5 = 9.3 ∧ x = 9.5 :=
by {
  sorry
}

end find_x_given_scores_l754_754416


namespace promotional_ratio_l754_754257

def total_emails : ℕ := 400
def spam_fraction : ℝ := 1 / 4
def important_emails : ℕ := 180

theorem promotional_ratio (total_emails: ℕ) (spam_fraction: ℝ) (important_emails: ℕ) :
  let spam_emails := (spam_fraction * total_emails)
  let remaining_emails := total_emails - spam_emails
  let promotional_emails := remaining_emails - important_emails
  (promotional_emails / remaining_emails) = (1 / 2.5) :=
by
  sorry

end promotional_ratio_l754_754257


namespace calculate_constants_l754_754475

noncomputable def parabola_tangent_to_line (a b : ℝ) : Prop :=
  let discriminant := (b - 2) ^ 2 + 28 * a
  discriminant = 0

theorem calculate_constants
  (a b : ℝ)
  (h_tangent : parabola_tangent_to_line a b) :
  a = -((b - 2) ^ 2) / 28 ∧ b ≠ 2 :=
by
  sorry

end calculate_constants_l754_754475


namespace partition_solution_equivalence_l754_754967

-- Definition for the partition of the set {0, 1, 2, 3, ...}
def is_odd_num_binary_digits (n : ℕ) : Prop := 
  (n.to_digits 2).length % 2 = 1

def is_even_num_binary_digits (n : ℕ) : Prop := 
  (n.to_digits 2).length % 2 = 0

-- Sets A and B defined by the number of binary digits being odd or even
def A : set ℕ := {n : ℕ | is_odd_num_binary_digits n}
def B : set ℕ := {n : ℕ | is_even_num_binary_digits n}

theorem partition_solution_equivalence : 
  ∃ (A B : set ℕ), (A ∪ B = set.univ) ∧ (∀ n, ∃! (x y : ℕ), n = x + y ∧ x ≠ y ∧ (x ∈ A ↔ y ∈ A) ∧ (x ∈ B ↔ y ∈ B)) :=
sorry

end partition_solution_equivalence_l754_754967


namespace complex_conjugate_difference_l754_754820

noncomputable def z : ℂ := (1 - I) / (2 + 2 * I) -- Define z as stated in the problem

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l754_754820


namespace problem_statement_l754_754171

noncomputable def given_condition (x : ℝ) : Prop := 
  (x + 1/x = real.sqrt 3)

theorem problem_statement (x : ℝ) (h : given_condition x) : 
  x^7 - 5 * x^5 + x^2 = -1 := 
begin 
  sorry 
end

end problem_statement_l754_754171


namespace find_x_values_for_3001_l754_754330

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l754_754330


namespace severe_flood_probability_next_10_years_l754_754751

variable (A B C : Prop)
variable (P : Prop → ℝ)
variable (P_A : P A = 0.8)
variable (P_B : P B = 0.85)
variable (thirty_years_no_flood : ¬A)

theorem severe_flood_probability_next_10_years :
  P C = (P B - P A) / (1 - P A) := by
  sorry

end severe_flood_probability_next_10_years_l754_754751


namespace total_votes_400_l754_754688

theorem total_votes_400 
    (V : ℝ)
    (h1 : ∃ (c1_votes c2_votes : ℝ), c1_votes = 0.70 * V ∧ c2_votes = 0.30 * V)
    (h2 : ∃ (majority : ℝ), majority = 160)
    (h3 : ∀ (c1_votes c2_votes majority : ℝ), c1_votes - c2_votes = majority) : V = 400 :=
by 
  sorry

end total_votes_400_l754_754688


namespace probability_of_symmetry_line_in_7x7_grid_l754_754494

def is_symmetry_line (n : ℕ) (grid : (ℕ × ℕ) → Bool) (R S : ℕ × ℕ) : Prop :=
  let (rx, ry) := R in
  let (sx, sy) := S in
  (sx = rx ∨ sy = ry ∨ sx - rx = sy - ry ∨ sx - rx = ry - sy)

theorem probability_of_symmetry_line_in_7x7_grid :
  ∀ (grid : (ℕ × ℕ) → Bool) (R : ℕ × ℕ),
  (∃ S : ℕ × ℕ, grid S ∧ S ≠ R ∧ is_symmetry_line 7 grid R S) →
  (1 / 2 : ℚ) :=
by
  let points := {p : ℕ × ℕ | p.1 < 7 ∧ p.2 < 7}
  have h_total : points.to_finset.card = 49 := by sorry
  let R := (3, 3) -- Center of a 7x7 grid
  have h_center : points.to_finset 3 3 := by sorry
  let other_points := points.to_finset.erase R
  have h_others : other_points.card = 48 := by sorry
  let symmetric_points := { S | S ∈ points.to_finset ∧ S ≠ R ∧ is_symmetry_line 7 (λ _ _, true) R S }
  have h_symmetric : symmetric_points.card = 24 := by sorry
  let probability := symmetric_points.card / other_points.card
  have h_prob : probability = 1 / 2 := by sorry
  exact h_prob

end probability_of_symmetry_line_in_7x7_grid_l754_754494


namespace cube_sphere_volume_ratio_l754_754711

theorem cube_sphere_volume_ratio (s : ℝ) (r : ℝ) (h : r = (Real.sqrt 3 * s) / 2):
  (s^3) / ((4 / 3) * Real.pi * r^3) = (2 * Real.sqrt 3) / Real.pi :=
by
  sorry

end cube_sphere_volume_ratio_l754_754711


namespace volume_s_l754_754094

def condition1 (x y : ℝ) : Prop := |9 - x| + y ≤ 12
def condition2 (x y : ℝ) : Prop := 3 * y - x ≥ 18
def S (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y

def is_volume_correct (m n : ℕ) (p : ℕ) :=
  (m + n + p = 153) ∧ (m = 135) ∧ (n = 8) ∧ (p = 10)

theorem volume_s (m n p : ℕ) :
  (∀ x y : ℝ, S x y) → is_volume_correct m n p :=
by 
  sorry

end volume_s_l754_754094


namespace seashells_at_end_of_month_l754_754789

-- Given conditions as definitions
def initial_seashells : ℕ := 50
def increase_per_week : ℕ := 20

-- Define function to calculate seashells in the nth week
def seashells_in_week (n : ℕ) : ℕ :=
  initial_seashells + n * increase_per_week

-- Lean statement to prove the number of seashells in the jar at the end of four weeks is 130
theorem seashells_at_end_of_month : seashells_in_week 4 = 130 :=
by
  sorry

end seashells_at_end_of_month_l754_754789


namespace a_gt_b_l754_754378

variable (n : ℕ) (a b : ℝ)
variable (n_pos : n > 1) (a_pos : 0 < a) (b_pos : 0 < b)
variable (a_eqn : a^n = a + 1)
variable (b_eqn : b^{2 * n} = b + 3 * a)

theorem a_gt_b : a > b :=
by {
  -- Proof is needed here
  sorry
}

end a_gt_b_l754_754378


namespace room_analysis_l754_754578

-- First person's statements
def statement₁ (n: ℕ) (liars: ℕ) :=
  n ≤ 3 ∧ liars = n

-- Second person's statements
def statement₂ (n: ℕ) (liars: ℕ) :=
  n ≤ 4 ∧ liars < n

-- Third person's statements
def statement₃ (n: ℕ) (liars: ℕ) :=
  n = 5 ∧ liars = 3

theorem room_analysis (n liars : ℕ) :
  (¬ statement₁ n liars) ∧ statement₂ n liars ∧ ¬ statement₃ n liars → (n = 4 ∧ liars = 2) :=
by
  sorry

end room_analysis_l754_754578


namespace units_digit_of_17_mul_24_l754_754376

def last_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_mul_24 :
  last_digit (17 * 24) = 8 :=
by 
  have h1 : last_digit 17 = 7 := rfl
  have h2 : last_digit 24 = 4 := rfl
  have h3 : 7 * 4 = 28 := rfl
  have h4 : last_digit 28 = 8 := rfl
  simp [last_digit, h3, h4]
  sorry

end units_digit_of_17_mul_24_l754_754376


namespace tile_chessboard_2n_l754_754395

theorem tile_chessboard_2n (n : ℕ) (board : Fin (2^n) → Fin (2^n) → Prop) (i j : Fin (2^n)) 
  (h : board i j = false) : ∃ tile : Fin (2^n) → Fin (2^n) → Bool, 
  (∀ i j, board i j = true ↔ tile i j = true) :=
sorry

end tile_chessboard_2n_l754_754395


namespace ratio_third_to_first_second_l754_754748

-- Define the times spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_total : ℕ := 90
def time_third_step : ℕ := time_total - (time_first_step + time_second_step)

-- Define the combined time for the first two steps
def time_combined_first_second : ℕ := time_first_step + time_second_step

-- The goal is to prove that the ratio of the time spent on the third step to the combined time spent on the first and second steps is 1:1
theorem ratio_third_to_first_second : time_third_step = time_combined_first_second :=
by
  -- Proof goes here
  sorry

end ratio_third_to_first_second_l754_754748


namespace tan_double_angle_l754_754815

theorem tan_double_angle (θ : ℝ) (h1 : cos θ = -3 / 5) (h2 : 0 < θ ∧ θ < π) : tan (2 * θ) = 24 / 7 :=
by
  sorry

end tan_double_angle_l754_754815


namespace projection_of_a_on_sum_ab_l754_754910

variables (a b : ℝ^3)

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

noncomputable def angle_between (v w : ℝ^3) : ℝ := real.acos (v.dot_product w / (magnitude v * magnitude w))

theorem projection_of_a_on_sum_ab :
  magnitude a = 2 →
  magnitude b = 2 →
  angle_between a b = real.pi / 3 →
  ((a.dot_product (a + b)) / magnitude (a + b)) = real.sqrt 3 :=
by
  intros ha hb hab
  sorry

end projection_of_a_on_sum_ab_l754_754910


namespace cricket_players_count_l754_754489

theorem cricket_players_count (hockey football softball total : ℕ) 
  (hockey_eq : hockey = 12)
  (football_eq : football = 16)
  (softball_eq : softball = 13)
  (total_eq : total = 51) :
  total - (hockey + football + softball) = 10 :=
by
  rw [hockey_eq, football_eq, softball_eq, total_eq]
  sorry

end cricket_players_count_l754_754489


namespace initial_coloring_books_count_l754_754737

theorem initial_coloring_books_count 
  (sold_books : ℕ) 
  (shelves : ℕ)
  (books_per_shelf : ℕ)
  (initial_books : ℕ) 
  (h1 : sold_books = 37)
  (h2 : shelves = 7)
  (h3 : books_per_shelf = 7)
  (h4 : initial_books = sold_books + (shelves * books_per_shelf)) :
  initial_books = 86 :=
by
  rw [h1, h2, h3]
  exact h4.symm

end initial_coloring_books_count_l754_754737


namespace trigonometric_expression_l754_754434

theorem trigonometric_expression (a : ℝ) (α : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
  (h₂ : ∀ (x : ℝ), f x = log a (x - 2) + 4)
  (h₃ : f 3 = 4)
  (h₄ : (sin α / cos α) = (4 / 3)) : 
  (sin α + 2 * cos α) / (sin α - cos α) = 10 :=
sorry

end trigonometric_expression_l754_754434


namespace range_of_a_l754_754013

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 0 → a * 4^x - 2^x + 2 > 0) → a > -1 :=
by sorry

end range_of_a_l754_754013


namespace magnitude_OB_l754_754073

open EuclideanGeometry

-- Define the points in the 3D Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the origin point
def O : Point3D := { x := 0, y := 0, z := 0 }

-- Define point A with given coordinates (1, 2, 3)
def A : Point3D := { x := 1, y := 2, z := 3 }

-- Define the orthogonal projection of A onto the yOz plane, which makes the x-coordinate 0
def B : Point3D := { x := 0, y := A.y, z := A.z }

-- Calculate the magnitude of OB using the distance formula
def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2)

-- Now we define the theorem stating that |OB| is sqrt(13)
theorem magnitude_OB : distance O B = Real.sqrt 13 := by
  -- The proof steps are omitted and marked as sorry
  sorry

end magnitude_OB_l754_754073


namespace fourth_triangle_exists_l754_754968

theorem fourth_triangle_exists (a b c d : ℝ)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (h4 : a + b > d) (h5 : a + d > b) (h6 : b + d > a)
  (h7 : a + c > d) (h8 : a + d > c) (h9 : c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b :=
by
  -- I skip the proof with "sorry"
  sorry

end fourth_triangle_exists_l754_754968


namespace polynomial_at_most_one_integer_root_l754_754134

theorem polynomial_at_most_one_integer_root (n : ℤ) :
  ∀ x1 x2 : ℤ, (x1 ≠ x2) → 
  (x1 ^ 4 - 1993 * x1 ^ 3 + (1993 + n) * x1 ^ 2 - 11 * x1 + n = 0) → 
  (x2 ^ 4 - 1993 * x2 ^ 3 + (1993 + n) * x2 ^ 2 - 11 * x2 + n = 0) → 
  false :=
by
  sorry

end polynomial_at_most_one_integer_root_l754_754134


namespace find_CQ_and_k_l754_754658

-- Definitions from conditions
variables {A A1 C B B1 Q P1 : Type*}
variables [fintype A] [fintype A1] [fintype C] [fintype B] [fintype B1] [fintype Q] [fintype P1]
variable (congruent_right_triangles : triangle A1 B1 C ≅ triangle A B C)
variable (angle_A1CB1_eq_angle_ACB_90 : ∠ A1 C B1 = 90 ∧ ∠ A C B = 90)
variable (angle_A1_eq_angle_A_30 : ∠ A1 = 30 ∧ ∠ A = 30)
variable (rotated_triangle : rotate (triangle A1 B1 C) C 45 = triangle A1' B1' C')
variable (intersection_P1_A1C_AB : intersection (line A1 C) (line A B) = P1)
variable (intersection_Q_A1B1_BC : intersection (line A1 B1) (line B C) = Q)
variable (A_P1_eq_2 : distance A P1 = 2)

-- Prove that CQ = sqrt(6) and the area ratio yields k = sqrt(3)
theorem find_CQ_and_k :
  (distance C Q = sqrt 6) ∧
  (let k := sqrt 3 in (area (triangle C B1 Q)) / (area (triangle C P1 A)) = 1 / k) :=
sorry

end find_CQ_and_k_l754_754658


namespace total_tiles_needed_l754_754078

def kitchen_width : ℕ := 15
def kitchen_length : ℕ := 20
def border_width : ℕ := 2
def tile1_length : ℕ := 1
def tile1_width : ℕ := 2
def tile2_side : ℕ := 3

theorem total_tiles_needed : 
  let total_tiles :=
    let adjusted_length := kitchen_length - 2 * border_width in
    let adjusted_width := kitchen_width - 2 * border_width in
    let border_tiles_length := 2 * (adjusted_length / tile1_width + adjusted_width / tile1_width) in
    let border_tiles_width := 2 * (adjusted_length / tile1_length + adjusted_width / tile1_length) in
    let border_tiles := border_tiles_length + border_tiles_width in
    let inner_area := adjusted_length * adjusted_width in
    let inner_tiles := (inner_area + tile2_side * tile2_side - 1) / (tile2_side * tile2_side) in -- Uses ceiling function for integer division
    border_tiles + inner_tiles
  in total_tiles = 48 :=
by
  sorry

end total_tiles_needed_l754_754078


namespace z_conjugate_difference_l754_754852

theorem z_conjugate_difference :
  let z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)
  in z - z.conj = -complex.I := 
  by
    let z := (1 - complex.I) / (2 + 2 * complex.I)
    sorry

end z_conjugate_difference_l754_754852


namespace probability_sum_one_m_value_correct_expected_value_correct_variance_correct_stddev_correct_l754_754738

noncomputable def X : Type := ℕ
noncomputable def P : X → ℝ
| 1 := 0.2
| 2 := 0.3
| 4 := 0.4
| 6 := 0.1
| _ := 0

theorem probability_sum_one : P 1 + P 2 + P 4 + P 6 = 1 := by
  sorry

theorem m_value_correct : P 4 = 0.4 := by
  sorry

theorem expected_value_correct : (1 * P 1) + (2 * P 2) + (4 * P 4) + (6 * P 6) = 3 := by
  sorry

theorem variance_correct : 
  let E := (1 * P 1) + (2 * P 2) + (4 * P 4) + (6 * P 6) in
  ((1 - E)^2 * P 1) + ((2 - E)^2 * P 2) + ((4 - E)^2 * P 4) + ((6 - E)^2 * P 6) = 2.4 := by
  sorry

theorem stddev_correct : real.sqrt 2.4 = 2 * real.sqrt 15 / 5 := by
  sorry

end probability_sum_one_m_value_correct_expected_value_correct_variance_correct_stddev_correct_l754_754738


namespace find_B_find_y_range_l754_754484

noncomputable def triangle_solutions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (2 * a - c) * Real.cos B = b * Real.cos C ∧
  0 < A ∧ A < Real.pi ∧
  0 < B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem find_B {a b c A B C : ℝ} (h : triangle_solutions a b c A B C) :
  B = Real.pi / 3 := sorry

theorem find_y_range {A C : ℝ} (hA : 0 < A ∧ A < 2 * Real.pi / 3) (hC : 0 < C ∧ C < 2 * Real.pi / 3) :
  B = Real.pi / 3 → 
  let y := Real.cos (A / 2) ^ 2 * Real.sin (C / 2) ^ 2 - 1 in
  -3/4 < y ∧ y < 3/4 := sorry

end find_B_find_y_range_l754_754484


namespace select_4_with_both_sexes_from_4_boys_3_girls_l754_754592

theorem select_4_with_both_sexes_from_4_boys_3_girls :
  (nat.choose 4 3 * nat.choose 3 1) + (nat.choose 4 2 * nat.choose 3 2) + (nat.choose 4 1 * nat.choose 3 3) = 34 :=
by
  sorry

end select_4_with_both_sexes_from_4_boys_3_girls_l754_754592


namespace probability_of_scoring_above_80_and_passing_exam_l754_754701

theorem probability_of_scoring_above_80_and_passing_exam :
  (let P_gt_90 := 0.18,
       P_80_to_89 := 0.51,
       P_70_to_79 := 0.15,
       P_60_to_69 := 0.09 in
   P_gt_90 + P_80_to_89 = 0.69 ∧
   P_gt_90 + P_80_to_89 + P_70_to_79 + P_60_to_69 = 0.93) :=
by
  intros
  sorry

end probability_of_scoring_above_80_and_passing_exam_l754_754701


namespace base_8_to_base_4_l754_754335

theorem base_8_to_base_4 (n : ℕ) (h : n = 6 * 8^2 + 5 * 8^1 + 3 * 8^0) : 
  (n : ℕ) = 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 2 * 4^1 + 3 * 4^0 :=
by
  -- Conversion proof goes here
  sorry

end base_8_to_base_4_l754_754335


namespace ratio_of_distances_l754_754547

-- Define the conditions
def is_regular_tetrahedron (V : Type*) (V1 V2 V3 V4 : V) (d : ℝ) : Prop :=
  dist V1 V2 = d ∧ dist V1 V3 = d ∧ dist V1 V4 = d ∧ dist V2 V3 = d ∧ dist V2 V4 = d ∧ dist V3 V4 = d

-- Define point P equidistant from all vertices
def equidistant_from_all_vertices (V : Type*) [MetricSpace V] (V1 V2 V3 V4 P : V) (d : ℝ) : Prop :=
  dist P V1 = dist P V2 ∧ dist P V2 = dist P V3 ∧ dist P V3 = dist P V4 ∧ dist P V4 = d

-- Define the distance measure for faces and edges from point P
noncomputable def sum_distances_from_point_to_faces (V : Type*) [MetricSpace V] (V1 V2 V3 V4 P : V) : ℝ := sorry

noncomputable def sum_distances_from_point_to_edges (V : Type*) [MetricSpace V] (V1 V2 V3 V4 P : V) : ℝ := sorry

-- Prove the ratio of distances
theorem ratio_of_distances (V : Type*) [MetricSpace V] (V1 V2 V3 V4 P : V) (d : ℝ) (hp : is_regular_tetrahedron V V1 V2 V3 V4 2)
  (heq : equidistant_from_all_vertices V V1 V2 V3 V4 P d) : 
  (sum_distances_from_point_to_faces V V1 V2 V3 V4 P) / (sum_distances_from_point_to_edges V V1 V2 V3 V4 P) = (Real.sqrt 6) / 3 := sorry

end ratio_of_distances_l754_754547


namespace vector_n_solution_l754_754019

open Real

-- Define vectors and their properties
def vector (R : Type) := (R × R)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2

def norm (v : vector ℝ) : ℝ :=
sqrt (v.1^2 + v.2^2)

-- Define the given problem conditions
variables (v_m : vector ℝ) (v_n : vector ℝ) (v_q : vector ℝ)
           (A B C : ℝ)

axiom h1 : v_m = (1, 1)
axiom h2 : ∃ (θ : ℝ), θ = 3/4 * π ∧ θ = real.arccos ((dot_product v_m v_n) / (norm v_m * norm v_n))
axiom h3 : dot_product v_m v_n = -1
axiom h4 : v_q = (1, 0)
axiom h5 : norm (v_q + v_n) = norm (v_q - v_n)
axiom h6 : A + C = 2 * B

-- Prove the conclusions: 
theorem vector_n_solution : v_n = (-1, 0) ∨ v_n = (0, -1) := sorry

def cos_squared_half_angle (x : ℝ) : ℝ := (1 + real.cos x) / 2

noncomputable def v_p (A B C : ℝ) : vector ℝ :=
( real.cos A, 2 * cos_squared_half_angle C )

noncomputable theorem range_of_norm_v_n_plus_v_p 
  (n : vector ℝ) (p : vector ℝ) (h_n : n = (-1, 0) ∨ n = (0, -1)) 
  (h_cos_squared_half_angle : ∀ θ, cos_squared_half_angle (θ/2) = (1 + real.cos θ) / 2) :
  let q := v_q in norm (v_n + v_p A B C) ∈ set.Ico (sqrt 2 / 2) (sqrt 5 / 2) := sorry

end vector_n_solution_l754_754019


namespace domain_of_f_l754_754763

def f (x : ℝ) : ℝ := 1 / ↑(Int.floor (x ^ 2 - 8 * x + 18))

theorem domain_of_f : ∀ x : ℝ, ∃ y : ℝ, f x = y := by
  sorry  -- The proof, showing that f(x) is always defined, is omitted.

end domain_of_f_l754_754763


namespace complement_intersection_l754_754222

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def A := {1, 4, 8}
def B := {3, 4, 7}
def complement (S T : Set ℕ) := {x | x ∈ S ∧ x ∉ T}

theorem complement_intersection :
  (complement U A ∩ B) = {3, 7} := by
  sorry

end complement_intersection_l754_754222


namespace positive_integers_dividing_8n_sum_l754_754383

theorem positive_integers_dividing_8n_sum :
  {n : ℕ // n > 0 ∧ (∃ k : ℕ, 8 * n = k * (n * (n + 1) / 2))}.card = 4 := sorry

end positive_integers_dividing_8n_sum_l754_754383


namespace find_radius_l754_754774

theorem find_radius (QP QO r : ℝ) (hQP : QP = 420) (hQO : QO = 427) : r = 77 :=
by
  -- Given QP^2 + r^2 = QO^2
  have h : (QP ^ 2) + (r ^ 2) = (QO ^ 2) := sorry
  -- Calculate the squares
  have h1 : (420 ^ 2) = 176400 := sorry
  have h2 : (427 ^ 2) = 182329 := sorry
  -- r^2 = 182329 - 176400
  have h3 : r ^ 2 = 5929 := sorry
  -- Therefore, r = 77
  exact sorry

end find_radius_l754_754774


namespace negative_half_less_than_negative_third_l754_754278

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l754_754278


namespace total_number_of_toys_l754_754538

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l754_754538


namespace encryption_query_determines_digits_l754_754154

theorem encryption_query_determines_digits :
  ∃ f : Char → ℕ, 
  (bijective f) ∧ 
  (∀ (A B : Char), 
    digit_sum_query f A B ≠ 0 → (∃ n, is_two_digit (digit_sum_query f A B) n)) → 
  ∃ Q : ℕ, Q ≤ 5 :=
begin
  -- placeholder for the proof
  sorry
end

end encryption_query_determines_digits_l754_754154


namespace proposition_D_true_l754_754442

variables (α β : set ℝ) (a b : set ℝ)

axiom planes_diff : α ≠ β
axiom lines_non_coincident : a ≠ b
axiom line_parallel_plane (a : set ℝ) (α : set ℝ) : Prop := ∀ x y ∈ a, x ≠ y → (∃z ∈ α, x + y = z)
axiom line_not_in_plane (a : set ℝ) (α : set ℝ) : ¬ (∀ x ∈ a, x ∈ α)
axiom parallel_planes (α β : set ℝ) : Prop := ∀ x ∈ α, ∃ y ∈ β, x = y
  
theorem proposition_D_true :
  (parallel_planes α β) ∧ (line_not_in_plane a α) ∧ (line_not_in_plane a β) ∧ (line_parallel_plane a α) →
  (line_parallel_plane a β) :=
sorry

end proposition_D_true_l754_754442


namespace trapezoid_area_is_180_l754_754497

-- Definition of an isosceles trapezoid.
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD AC : ℝ)
  (AB_eq_CD : AB = 10)
  (CD_eq_AB : CD = 10)

-- Definitions for the perpendiculars BH and DK
structure PerpendicularsToDiagonal (B D H K : Type) :=
  (BH DK : ℝ)
  (BH_to_DIAG : BH)
  (DK_to_DIAG : DK)
  (BH_DK_on_AC : ℝ)
  (ratio_AH_AK_AC : ∃ (k : ℝ), AH = 5 * k ∧ AK = 14 * k ∧ AC = 15 * k)

noncomputable def area_of_trapezoid {A B C D : Type} [IsoscelesTrapezoid A B C D] [PerpendicularsToDiagonal B D H K] : ℝ :=
  (1 / 2) * AC * (BH + DK)

theorem trapezoid_area_is_180 (A B C D : Type) [IsoscelesTrapezoid A B C D] [PerpendicularsToDiagonal B D H K] :
  area_of_trapezoid = 180 :=
sorry

end trapezoid_area_is_180_l754_754497


namespace num_pos_integers_n_divide_8n_l754_754387

theorem num_pos_integers_n_divide_8n (n : ℕ) : 
  (∃ N : ℕ, N = 4 ∧ 
  { x : ℕ | 1 ≤ x ∧ 
    divides (8 * x) ((x * (x + 1)) / 2) 
  }.card = N) := sorry

end num_pos_integers_n_divide_8n_l754_754387


namespace repeating_decimal_to_fraction_l754_754204

theorem repeating_decimal_to_fraction :
  let x := 0.3746 + (46 : ℕ) / 9990 in x = 3709 / 9900 :=
by
  sorry

end repeating_decimal_to_fraction_l754_754204


namespace number_of_elements_in_T_l754_754551

/-- Define the function g -/
def g (x : ℝ) : ℝ := (x + 8) / x

/-- Define the sequence of functions g_n -/
def g_seq : ℕ → (ℝ → ℝ)
| 0       := g
| (n + 1) := g ∘ g_seq n

/-- Define the set T -/
def T : Set ℝ := {x | ∃ n : ℕ, g_seq n x = x }

theorem number_of_elements_in_T : fintype.card T = 2 :=
sorry

end number_of_elements_in_T_l754_754551


namespace find_ordered_pair_l754_754143

variable {R : Type} [linear_ordered_field R]

theorem find_ordered_pair (p q : R) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : ∀ x, x^2 + 2*p*x + q = 0 → x = p ∨ x = q) :
  (p, q) = (1 : R, -3 : R) := 
by {
  sorry
}

end find_ordered_pair_l754_754143


namespace cos_C_in_triangle_l754_754520

theorem cos_C_in_triangle
  (A B C : ℝ) (sin_A : ℝ) (cos_B : ℝ)
  (h1 : sin_A = 4 / 5)
  (h2 : cos_B = 12 / 13) :
  cos (π - A - B) = -16 / 65 :=
by
  -- Proof steps would be included here
  sorry

end cos_C_in_triangle_l754_754520


namespace repeatingDecimalSum_is_fraction_l754_754360

noncomputable def repeatingDecimalSum : ℚ :=
  (0.3333...).val + (0.040404...).val + (0.005005...).val

theorem repeatingDecimalSum_is_fraction : repeatingDecimalSum = 1134 / 2997 := by
  sorry

end repeatingDecimalSum_is_fraction_l754_754360


namespace compound_props_l754_754414

variables (x y : ℝ)

def P : Prop := ∀ x y : ℝ, x > y → -x > -y
def Q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

theorem compound_props:
  (¬ P ∨ ¬ Q) ∧ (¬ P ∨ Q) :=
begin
  sorry
end

end compound_props_l754_754414


namespace number_of_routes_from_1_to_7_l754_754918

-- Definition of the problem
def routes (n : Nat) : Nat
| 1 := 1
| 2 := routes 1
| 3 := routes 2 + routes 1
| 4 := routes 3 + routes 2
| 5 := routes 4 + routes 3
| 6 := routes 5 + routes 4
| 7 := routes 6 + routes 5
| _ := 0

theorem number_of_routes_from_1_to_7 : routes 7 = 13 :=
by
  -- Proof goes here
  sorry

end number_of_routes_from_1_to_7_l754_754918


namespace initial_tanks_hold_fifteen_fish_l754_754532

theorem initial_tanks_hold_fifteen_fish (t : Nat) (additional_tanks : Nat) (fish_per_additional_tank : Nat) (total_fish : Nat) :
  t = 3 ∧ additional_tanks = 3 ∧ fish_per_additional_tank = 10 ∧ total_fish = 75 → 
  ∀ (F : Nat), (F * t) = 45 → F = 15 :=
by
  sorry

end initial_tanks_hold_fifteen_fish_l754_754532


namespace graph_symmetric_monotonicity_interval_false_max_value_one_min_value_a_l754_754433

noncomputable def f (a x : ℝ) : ℝ :=
  a * (sqrt (-((x + 2) * (x - 6))) / (sqrt (x + 2) + sqrt (6 - x)))

theorem graph_symmetric {a : ℝ} : (∀ x ∈ set.Icc (-2 : ℝ) 6, f a x = f a (4 - x)) :=
by sorry

theorem monotonicity_interval_false {a : ℝ} (h : ∀ x ∈ set.Icc (-2 : ℝ) 2, monotone_on (f a) (set.Icc (-2 : ℝ) 2)) : a < 0 = false :=
by sorry

theorem max_value_one {a : ℝ} (ha : a = 1) : (∀ x ∈ set.Icc (-2 : ℝ) 6, f a x ≤ 1) ∧ (∃ x ∈ set.Icc (-2 : ℝ) 6, f a x = 1) :=
by sorry

theorem min_value_a {a : ℝ} (ha : a < 0) : (∀ x ∈ set.Icc (-2 : ℝ) 6, f a x ≥ a) ∧ (∃ x ∈ set.Icc (-2 : ℝ) 6, f a x = a) :=
by sorry

end graph_symmetric_monotonicity_interval_false_max_value_one_min_value_a_l754_754433


namespace street_length_approx_l754_754721

-- Given conditions
def speed_kmh : ℝ := 5.95   -- Speed in km/h
def time_min : ℝ := 6       -- Time in minutes

-- Conversion factors
def km_to_m : ℝ := 1000     -- 1 km = 1000 meters
def hr_to_min : ℝ := 60     -- 1 hour = 60 minutes

-- Derived conversion
def speed_m_per_min : ℝ := speed_kmh * km_to_m / hr_to_min

-- Desired length of the street in meters
def length_of_street : ℝ := speed_m_per_min * time_min

-- Theorem stating the length of the street is approximately 595 meters
theorem street_length_approx :
  abs (length_of_street - 595) < 1 :=
by
  -- This is the statement only, actual proof is omitted.
  sorry

end street_length_approx_l754_754721


namespace weekly_allowance_is_4_50_l754_754209

variables (A : ℝ)
constant h1 : A * (3/5) + (A * (2/5) * (1/3) + 1.20) = A
constant h2 : A * (2/5) * (2/3) = 4/15 * A
constant h3 : (4/15) * A = 1.20

-- Prove the correct answer
theorem weekly_allowance_is_4_50 : A = 4.50 :=
by
  sorry

end weekly_allowance_is_4_50_l754_754209


namespace solution_set_of_inequality_l754_754374

theorem solution_set_of_inequality (x : ℝ) : (1 / x ≤ x) ↔ (-1 ≤ x ∧ x < 0) ∨ (x ≥ 1) := sorry

end solution_set_of_inequality_l754_754374


namespace number_of_possible_sets_l754_754423

theorem number_of_possible_sets (M : Set ℕ) (h : M ∪ {1} = {1, 2, 3}) :
  ({ N | N ∪ {1} = {1, 2, 3} }.toFinset.card = 2) :=
sorry

end number_of_possible_sets_l754_754423


namespace calc_z_conj_diff_l754_754829

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754829


namespace lambda_value_dot_product_value_l754_754063

def vector := ℝ × ℝ

def a : vector := (3, -4)
def b : vector := (2, -3)
def c : vector := (0, 4 + Real.sqrt 3)
def d : vector := (-1, 3 + Real.sqrt 3)

def parallel (v w : vector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem lambda_value : ∃ λ : ℝ, (λ = -1/3) ∧
  parallel (a.1 - λ * b.1, a.2 - λ * b.2) (3 * a.1 + b.1, 3 * a.2 + b.2) := by
  sorry

theorem dot_product_value : dot_product (a.1 + c.1, a.2 + c.2) (b.1 + d.1, b.2 + d.2) = 6 := by
  sorry

end lambda_value_dot_product_value_l754_754063


namespace midlines_tangent_fixed_circle_l754_754218

-- Definitions of geometric objects and properties
structure Point :=
(x : ℝ) (y : ℝ)

structure Circle :=
(center : Point) (radius : ℝ)

-- Assumptions (conditions)
variable (ω1 ω2 : Circle)
variable (l1 l2 : Point → Prop) -- Representing line equations in terms of points
variable (angle : Point → Prop) -- Representing the given angle sides

-- Tangency conditions
axiom tangency1 : ∀ p : Point, l1 p → p ≠ ω1.center ∧ (ω1.center.x - p.x) ^ 2 + (ω1.center.y - p.y) ^ 2 = ω1.radius ^ 2
axiom tangency2 : ∀ p : Point, l2 p → p ≠ ω2.center ∧ (ω2.center.x - p.x) ^ 2 + (ω2.center.y - p.y) ^ 2 = ω2.radius ^ 2

-- Non-intersecting condition for circles
axiom nonintersecting : (ω1.center.x - ω2.center.x) ^ 2 + (ω1.center.y - ω2.center.y) ^ 2 > (ω1.radius + ω2.radius) ^ 2

-- Conditions for tangent circles and middle line being between them
axiom betweenness : ∀ p, angle p → (ω1.center.y < p.y ∧ p.y < ω2.center.y)

-- Midline definition and fixed circle condition
theorem midlines_tangent_fixed_circle :
  ∃ (O : Point) (d : ℝ), ∀ (T : Point → Prop), 
  (∃ (p1 p2 : Point), l1 p1 ∧ l2 p2 ∧ T p1 ∧ T p2) →
  (∀ (m : Point), T m ↔ ∃ (p1 p2 p3 p4 : Point), T p1 ∧ T p2 ∧ angle p3 ∧ angle p4 ∧ 
  m.x = (p1.x + p2.x + p3.x + p4.x) / 4 ∧ m.y = (p1.y + p2.y + p3.y + p4.y) / 4) → 
  (∀ (m : Point), (m.x - O.x) ^ 2 + (m.y - O.y) ^ 2 = d^2)
:= 
sorry

end midlines_tangent_fixed_circle_l754_754218


namespace sequence_x_values_3001_l754_754312

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l754_754312


namespace sequence_x_values_3001_l754_754306

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l754_754306


namespace sequence_term_3001_exists_exactly_4_values_l754_754326

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l754_754326


namespace find_t_l754_754500

variable (t : ℝ)
variable (θ : ℝ)
variable (sinθ : ℝ) := t / Real.sqrt (4 + t^2)
variable (cosθ : ℝ) := -2 / Real.sqrt (4 + t^2)

theorem find_t (h : sinθ + cosθ = Real.sqrt 5 / 5) : t = 4 :=
sorry

end find_t_l754_754500


namespace eagles_score_l754_754047

variables (F E : ℕ)

theorem eagles_score (h1 : F + E = 56) (h2 : F = E + 8) : E = 24 := 
sorry

end eagles_score_l754_754047


namespace sequence_x_values_3001_l754_754308

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l754_754308


namespace value_of_a_plus_c_l754_754468

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def f_inv (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem value_of_a_plus_c : a + c = -1 :=
sorry

end value_of_a_plus_c_l754_754468


namespace solve_for_a_l754_754508

theorem solve_for_a (a : ℝ) (h_a : a > 5 / 6) :
  (∃ t : ℝ, (2 + t, -1 + t) ∈ { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 2 * a * p.1 - 2 * a * p.2 }) ∧
  (∃ t1 t2 : ℝ, t1 + t2 = -1 ∧ t1 * t2 = (5 - 6 * a) / 2 ∧
  (t1 - t2) ^ 2 = 6 * (abs (t1 * t2))) :=
begin
  sorry
end

end solve_for_a_l754_754508


namespace strictly_increasing_interval_l754_754342

noncomputable def f (x : ℝ) : ℝ := (1/2)^(-x^2 + 6*x - 2)

theorem strictly_increasing_interval :
  ∃ (a : ℝ), (∀ x y : ℝ, x ∈ (a, +∞) ∧ y ∈ (a, +∞) ∧ x < y → f x < f y) :=
begin
  use 3,
  sorry
end

end strictly_increasing_interval_l754_754342


namespace find_k_at_4_l754_754981

-- Definitions based on conditions
def h (x : ℝ) : ℝ := x^3 - 2*x + 1

def k (roots : ℝ × ℝ × ℝ) (x : ℝ) : ℝ := 
  let ⟨a, b, c⟩ := roots
  (x - a^2) * (x - b^2) * (x - c^2)

theorem find_k_at_4 :
  let roots := (a, b, c)
  (h(a) = 0) (h(b) = 0) (h(c) = 0) (k roots (0) = 1) →
  k roots (4) = 15 :=
by
  intros roots h_conds k_conds
  sorry

end find_k_at_4_l754_754981


namespace solve_for_x_l754_754805

theorem solve_for_x (x : ℝ) : 4 * x - 8 + 3 * x = 12 + 5 * x → x = 10 :=
by
  intro h
  sorry

end solve_for_x_l754_754805


namespace sequence_term_3001_exists_exactly_4_values_l754_754323

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l754_754323


namespace repeatingDecimalSum_is_fraction_l754_754358

noncomputable def repeatingDecimalSum : ℚ :=
  (0.3333...).val + (0.040404...).val + (0.005005...).val

theorem repeatingDecimalSum_is_fraction : repeatingDecimalSum = 1134 / 2997 := by
  sorry

end repeatingDecimalSum_is_fraction_l754_754358


namespace max_f_sin_find_m_l754_754426

-- Definitions from conditions
def f (m : ℝ) (x : ℝ) := -2 * x ^ 2 + 4 * m * x - 1

-- Problem 1: Prove the maximum value of f (sin θ) / (sin θ) given f(x) = -2x^2 + 8x - 1
theorem max_f_sin (m := 2) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  let s := Real.sin θ in
  f 2 s / s ≤ -2 * Real.sqrt 2 + 8 :=
sorry

-- Problem 2: Prove that the values of m ensuring the maximum value of y = f(x) for x ∈ [-1, 1] is 7 are -2.5 or 2.5.
theorem find_m (m : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f m x ≤ 7) →
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ f m x = 7) →
  m = -2.5 ∨ m = 2.5 :=
sorry

end max_f_sin_find_m_l754_754426


namespace probability_max_3_l754_754224

-- Define the set of cards and the drawn cards configuration
def cards := ({1, 2, 3} : Finset ℕ).product ({1, 2} : Finset ℕ)

-- Define a draw to be a subset of two cards
def draw (c : Finset (ℕ × ℕ)) : Finset (Finset (ℕ × ℕ)) :=
  c.powerset.filter (λ s, s.card = 2)

-- Event "the maximum number on the two drawn cards is 3"
def event_max_is_3 (s : Finset (ℕ × ℕ)) : Prop :=
  ∃ a b, (a, b) ∈ s ∧ ((a = 3) ∨ (b = 3))

-- Calculate the number of favorable outcomes
def favorable_outcomes : Finset (Finset (ℕ × ℕ)) :=
  (draw cards).filter (λ s, event_max_is_3 s)

-- Calculate the total number of outcomes
def total_outcomes : Finset (Finset (ℕ × ℕ)) :=
  draw cards

-- Calculate the probability
noncomputable def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

-- Main theorem to prove
theorem probability_max_3 : probability = 3 / 5 :=
  sorry

end probability_max_3_l754_754224


namespace varphi_value_l754_754893

noncomputable def f (x ϕ : ℝ) : ℝ := √2 * Real.sin (x + π / 4 + ϕ)

theorem varphi_value
  (ϕ : ℝ)
  (h1 : ∀ x : ℝ, f x ϕ = - f (-x) ϕ)
  (h2 : -π / 2 ≤ ϕ ∧ ϕ ≤ π / 2) :
  ϕ = -π / 4 :=
by
  sorry

end varphi_value_l754_754893


namespace parabola_equation_l754_754635

/-- The vertex of a parabola is at the origin and its focus is on the x-axis. The line 2x - y = 0 intersects 
the parabola at points A and B. If P(1, 2) is the midpoint of segment AB, then the equation of the parabola 
is y^2 = 8x. -/
theorem parabola_equation (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  let parabola := λ (x y : ℝ), y^2 = 8 * x in
  let line := λ x y : ℝ, 2 * x - y = 0 in
  P = (1,2) ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2 →
  ∀ x y, parabola x y :=
by sorry

end parabola_equation_l754_754635


namespace cm_bc_ratio_l754_754576

noncomputable def ratio_cm_bc (A B C K L M : Type) [Triangle ABC K L M] :=
∀ (A B C K L M) (AK KB AL LC : ℝ) (H1 : AK / KB = 4 / 7) (H2 : AL / LC = 3 / 2),
  ∃ (CM BC : ℝ), CM / BC = 8 / 13

theorem cm_bc_ratio :
  (∀ (A B C K L M : Point) (AK KB AL LC : ℝ), 
    AK / KB = 4 / 7 ∧ AL / LC = 3 / 2 →
    ∃ (CM BC : ℝ), CM / BC = 8 / 13) := by
  sorry

end cm_bc_ratio_l754_754576


namespace distinct_increasing_digits_l754_754024

theorem distinct_increasing_digits (c : ℕ) : 
  (∃ (n : ℕ), 2050 ≤ n ∧ n < 2300 ∧ (∀ (i j : ℕ), i < j → to_string n.nth i < to_string n.nth j) ∧ (to_string n).to_list.nodup) → c = 20 := 
sorry

end distinct_increasing_digits_l754_754024


namespace janet_percentage_of_snowballs_l754_754972

-- Define the number of snowballs made by Janet
def janet_snowballs : ℕ := 50

-- Define the number of snowballs made by Janet's brother
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage calculation function
def calculate_percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Proof statement
theorem janet_percentage_of_snowballs : calculate_percentage janet_snowballs total_snowballs = 25 := 
by
  sorry

end janet_percentage_of_snowballs_l754_754972


namespace five_digit_numbers_count_l754_754915

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_five_digit_numbers (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10 in
  let d2 := n / 1000 % 10 in
  let d3 := n / 100 % 10 in
  let d4 := n / 10 % 10 in
  let d5 := n % 10 in
  is_even d1 ∧ is_odd d2 ∧ 
  (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
   d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧
   d3 ≠ d4 ∧ d3 ≠ d5 ∧
   d4 ≠ d5) ∧
  is_even (d1 + d2 + d3 + d4 + d5)

theorem five_digit_numbers_count : 
  ∃ n, n = 2160 ∧ (∃ m, valid_five_digit_numbers m) :=
by
  sorry

end five_digit_numbers_count_l754_754915


namespace equal_chord_lengths_l754_754503

open EuclideanGeometry

noncomputable def acute_triangle (A B C : Point) : Prop :=
  ∃ (A₁ B₁ C₁ : Point), isAltitude A B C A₁ ∧ isAltitude B A C B₁ ∧ isAltitude C A B C₁

noncomputable def isOrthocenter (M A B C A₁ B₁ C₁ : Point) : Prop :=
  intersectionOfAltitudes A B C A₁ B₁ C₁ = M

noncomputable def isDiameter (A₁ : Point) (circle : Circle) : Prop :=
  ∃ A, circle.center = midpoint (A, A₁) ∧ distance (circle.center, A) = distance (circle.center, A₁)

noncomputable def perpendicularChordThruPoint
  (M : Point) (altitude : Line) (circle : Circle) : Length :=
  let q := diameterPerpChord circle altitude in
  if passesThrough q M then lengthOfChord q else 0

theorem equal_chord_lengths (A B C A₁ B₁ C₁ M : Point)
  (cirA cirB cirC : Circle)
  (hAcute : acute_triangle A B C)
  (hAltitudes : isAltitude A B C A₁ ∧ isAltitude B A C B₁ ∧ isAltitude C A B C₁)
  (hOrthocenter : isOrthocenter M A B C A₁ B₁ C₁)
  (hDiameters : isDiameter A₁ cirA ∧ isDiameter B₁ cirB ∧ isDiameter C₁ cirC)
  : perpendicularChordThruPoint M (altitude A A₁) cirA =
    perpendicularChordThruPoint M (altitude B B₁) cirB ∧
    perpendicularChordThruPoint M (altitude C C₁) cirC := by
  sorry

end equal_chord_lengths_l754_754503


namespace greatest_possible_value_l754_754460

theorem greatest_possible_value (x : ℝ) (h : 13 = x^2 + 1 / x^2) : x + 1 / x ≤ Real.sqrt 15 :=
begin
  sorry
end

end greatest_possible_value_l754_754460


namespace arithmetic_sequence_75th_term_diff_l754_754254

noncomputable def sum_arith_sequence (n : ℕ) (a d : ℚ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_75th_term_diff {n : ℕ} {a d : ℚ}
  (hn : n = 150)
  (sum_seq : sum_arith_sequence n a d = 15000)
  (term_range : ∀ k, 0 ≤ k ∧ k < n → 20 ≤ a + k * d ∧ a + k * d ≤ 150)
  (t75th : ∃ L G, L = a + 74 * d ∧ G = a + 74 * d) :
  G - L = (7500 / 149) :=
sorry

end arithmetic_sequence_75th_term_diff_l754_754254


namespace constant_term_of_expansion_l754_754887

-- Condition and definitions
variables {a x : ℝ}
def expansion (x : ℝ) := (a * x + 1 / x) * (2 * x - 1 / x)^5
def sum_of_coefficients (x : ℝ) := (a + 1) * (2 - 1)^5

-- Main theorem
theorem constant_term_of_expansion : 
  (sum_of_coefficients 1 = 2) → a = 1 → 
  constant_term (expansion x) = 40 :=
sorry

end constant_term_of_expansion_l754_754887


namespace johns_least_payback_days_l754_754080

theorem johns_least_payback_days (P r : ℝ) (hP : P = 20) (hr : r = 0.10) :
  ∃ n : ℕ, P * (1 + r)^n ≥ 2 * P ∧ (∀ m : ℕ, m < n → P * (1 + r)^m < 2 * P) :=
begin
  use 8,
  -- We can fill in the proof here later
  sorry,
end

end johns_least_payback_days_l754_754080


namespace subsets_containing_6_count_l754_754917

-- Set definition
def my_set : set ℕ := {1, 2, 3, 4, 6}

-- Condition: A subset must contain the number 6
def contains_6 (s : set ℕ) : Prop := 6 ∈ s

-- The number of subsets of {1, 2, 3, 4, 6} that contain the number 6
theorem subsets_containing_6_count : 
  (finset.univ.powerset.filter (λ s, contains_6 s)).card = 16 := by sorry

end subsets_containing_6_count_l754_754917


namespace cos_C_of_triangle_l754_754512

theorem cos_C_of_triangle (A B C : ℝ) (hA : sin A = 4 / 5) (hB : cos B = 12 / 13) :
  cos C = -16 / 65 :=
sorry

end cos_C_of_triangle_l754_754512


namespace domain_of_f_l754_754614

def domain_f : Set ℝ := {x | x - 4 ≥ 0} ∩ {x | x ≠ 5}

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, y = sqrt (x - 4) + 1 / (x - 5) } = {x | 4 ≤ x ∧ x < 5} ∪ {x | 5 < x} :=
by
  sorry

end domain_of_f_l754_754614


namespace output_is_three_l754_754674

-- Define the initial values
def initial_a : ℕ := 1
def initial_b : ℕ := 2

-- Define the final value of a after the computation
def final_a : ℕ := initial_a + initial_b

-- The theorem stating that the final value of a is 3
theorem output_is_three : final_a = 3 := by
  sorry

end output_is_three_l754_754674


namespace find_minutes_per_mile_l754_754681

-- Conditions
def num_of_movies : ℕ := 2
def avg_length_of_movie_hours : ℝ := 1.5
def total_distance_miles : ℝ := 15

-- Question and proof target
theorem find_minutes_per_mile :
  (num_of_movies * avg_length_of_movie_hours * 60) / total_distance_miles = 12 :=
by
  -- Insert the proof here (not required as per the task instructions)
  sorry

end find_minutes_per_mile_l754_754681


namespace num_pos_integers_n_divide_8n_l754_754389

theorem num_pos_integers_n_divide_8n (n : ℕ) : 
  (∃ N : ℕ, N = 4 ∧ 
  { x : ℕ | 1 ≤ x ∧ 
    divides (8 * x) ((x * (x + 1)) / 2) 
  }.card = N) := sorry

end num_pos_integers_n_divide_8n_l754_754389


namespace divisor_of_p_l754_754554

theorem divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40)
  (hqr : Nat.gcd q r = 45) (hrs : Nat.gcd r s = 60)
  (hspr : 100 < Nat.gcd s p ∧ Nat.gcd s p < 150)
  : 7 ∣ p :=
sorry

end divisor_of_p_l754_754554


namespace midpoint_of_intersects_slope_of_line_l754_754062

-- Definitions based on the problem
def parametric_line (t : ℝ) (α : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)
def parametric_curve (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- First problem
theorem midpoint_of_intersects
  (α : ℝ) (t1 t2 : ℝ)
  (hα : α = Real.pi / 3)
  (intersect1 : parametric_line t1 α = parametric_curve some_θ1)
  (intersect2 : parametric_line t2 α = parametric_curve some_θ2)
  (h_distinct : t1 ≠ t2) :
  let A := parametric_line t1 α,
      B := parametric_line t2 α,
      M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  M = (12 / 13, -Real.sqrt 3 / 13) :=
by sorry

-- Second problem
theorem slope_of_line
  (α : ℝ) (t1 t2 : ℝ)
  (hPA_PB_OP_squared : (2 + t1 * Real.cos α - 2) * (2 + t2 * Real.cos α - 2)
                        + (Real.sqrt 3 + t1 * Real.sin α - Real.sqrt 3)
                        * (Real.sqrt 3 + t2 * Real.sin α - Real.sqrt 3)
                        = 7) :
  let slope := Real.sin α / Real.cos α in
  slope = Real.sqrt 5 / 4 ∨ slope = -Real.sqrt 5 / 4 :=
by sorry

end midpoint_of_intersects_slope_of_line_l754_754062


namespace problem_B_problem_D_l754_754432

def f (x a : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - x

-- Problem B: Prove that if the center of symmetry of f(x) is at (1, f(1)), then a = -1.
theorem problem_B (a : ℝ) (h_symm : ∀ x, f (1 - x) a = f (1 + x) a) : a = -1 :=
  sorry

-- Problem D: Prove that f(x) must have three zeros.
theorem problem_D (a : ℝ) (h_disc : (2 * a / 3)^2 + 4 / 3 > 0) : ∀ x, ∃ x1 x2 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 ≠ x2 :=
  sorry

end problem_B_problem_D_l754_754432


namespace basketball_75th_percentile_is_39_l754_754487

def basketScores : List ℕ := [29, 30, 38, 25, 37, 40, 42, 32]

def percentile (p : ℕ) (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (· < ·)
  let n := lst.length
  if n = 0 then 
    0
  else
    let pos := ((p * n + 99) / 100)
    if pos < 1 then 
      sorted_lst.get! 0
    else if pos ≥ n then
      sorted_lst.get! (n - 1)
    else
      (sorted_lst.get! (pos - 1) + sorted_lst.get! pos) / 2

theorem basketball_75th_percentile_is_39 :
  percentile 75 basketScores = 39 :=
by
  sorry

end basketball_75th_percentile_is_39_l754_754487


namespace prize_draw_total_50_probability_l754_754261

theorem prize_draw_total_50_probability :
  let ben_prizes := {5, 10, 20}
  let jamie_prizes := {30, 40}
  let total := { p | p ∈ ben_prizes ∧ p + ∈ jamie_prizes ∧ p + 50 } in
  let prob :=
    ditr_choice_total / (ben_choice_total * jamie_choice_total)
  have h1 : finite ben_prizes := sorry
  have h2 : finite jamie_prizes := sorry
  have h3 : continuous measure_of_total := sorry  
  (sum total = 50) = (1 / ben_choice_total) * (1 / jamie_choice_total) := by sorry

end prize_draw_total_50_probability_l754_754261


namespace calc_z_conj_diff_l754_754830

-- condition
def z : ℂ := (1 - complex.I) / (2 + 2 * complex.I)

-- statement to prove
theorem calc_z_conj_diff : z - conj z = -complex.I :=
by
  -- Proof details would go here
  sorry

end calc_z_conj_diff_l754_754830


namespace average_salary_of_all_workers_is_correct_l754_754149

noncomputable def average_salary_all_workers (n_total n_tech : ℕ) (avg_salary_tech avg_salary_others : ℝ) : ℝ :=
  let n_others := n_total - n_tech
  let total_salary_tech := n_tech * avg_salary_tech
  let total_salary_others := n_others * avg_salary_others
  let total_salary := total_salary_tech + total_salary_others
  total_salary / n_total

theorem average_salary_of_all_workers_is_correct :
  average_salary_all_workers 21 7 12000 6000 = 8000 :=
by
  unfold average_salary_all_workers
  sorry

end average_salary_of_all_workers_is_correct_l754_754149


namespace area_above_the_line_l754_754198

-- Definitions of the circle and the line equations
def circle_eqn (x y : ℝ) := (x - 5)^2 + (y - 3)^2 = 1
def line_eqn (x y : ℝ) := y = x - 5

-- The main statement to prove
theorem area_above_the_line : 
  ∃ (A : ℝ), A = (3 / 4) * Real.pi ∧ 
  ∀ (x y : ℝ), 
    circle_eqn x y ∧ y > x - 5 → 
    A > 0 := 
sorry

end area_above_the_line_l754_754198


namespace complex_conj_difference_l754_754839

theorem complex_conj_difference (z : ℂ) (hz : z = (1 - I) / (2 + 2 * I)) : z - conj z = -I :=
sorry

end complex_conj_difference_l754_754839
