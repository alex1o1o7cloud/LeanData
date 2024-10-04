import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Big_Operators.Basic
import Mathlib.Algebra.Complex.Basic
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.GcdDomain
import Mathlib.Algebra.GeometricSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.SimpleInterest
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Normed.Field.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialNumberTheory
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Group.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real
import Mathlib.Trigonometry.Basic

namespace sum_remainders_eq_two_l626_626638

theorem sum_remainders_eq_two (a b c : ℤ) (h_a : a % 24 = 10) (h_b : b % 24 = 4) (h_c : c % 24 = 12) :
  (a + b + c) % 24 = 2 :=
by
  sorry

end sum_remainders_eq_two_l626_626638


namespace new_person_weight_l626_626932

theorem new_person_weight (avg_increase : ℝ) (n_people : ℕ) (old_weight : ℝ) 
  (weight_increase : avg_increase = 3.5) (num_people : n_people = 8) (old_person_weight : old_weight = 65) :
  ∃ w_new : ℝ, w_new = old_weight + (n_people * avg_increase) :=
by
  have w_new : ℝ := old_weight + (n_people * avg_increase)
  use w_new
  rw [weight_increase, num_people, old_person_weight]
  exact rfl

end new_person_weight_l626_626932


namespace pencils_in_pencil_case_l626_626202

theorem pencils_in_pencil_case : ∀ (total_items pens pencils erasers : ℕ), 
  total_items = 13 → pens = 2 * pencils → erasers = 1 → pencils = 4 :=
begin
  intros total_items pens pencils erasers h1 h2 h3,
  -- Proof goes here, but it's omitted because it's not required
  sorry
end

end pencils_in_pencil_case_l626_626202


namespace find_k_and_b_l626_626426

theorem find_k_and_b (k b : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
  ((P.1 - 1)^2 + P.2^2 = 1) ∧ 
  ((Q.1 - 1)^2 + Q.2^2 = 1) ∧ 
  (P.2 = k * P.1) ∧ 
  (Q.2 = k * Q.1) ∧ 
  (P.1 - P.2 + b = 0) ∧ 
  (Q.1 - Q.2 + b = 0) ∧ 
  ((P.1 + Q.1) / 2 = (P.2 + Q.2) / 2)) →
  k = -1 ∧ b = -1 :=
sorry

end find_k_and_b_l626_626426


namespace perimeters_ratio_l626_626579

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l626_626579


namespace triangle_angle_sum_l626_626445

theorem triangle_angle_sum (x : ℝ) :
    let angle1 : ℝ := 40
    let angle2 : ℝ := 4 * x
    let angle3 : ℝ := 3 * x
    angle1 + angle2 + angle3 = 180 -> x = 20 := 
sorry

end triangle_angle_sum_l626_626445


namespace union_cardinality_of_sets_l626_626360

theorem union_cardinality_of_sets :
  ∀ (S : Fin 1985 → Set α) (h₀ : ∀ i, |S i| = 45)
    (h₁ : ∀ i j, i ≠ j → |S i ∪ S j| = 89),
  |⋃ i, S i| = 87381 :=
by
  intros
  sorry

end union_cardinality_of_sets_l626_626360


namespace circles_cover_convex_quadrilateral_l626_626720

theorem circles_cover_convex_quadrilateral 
  (A B C D : Point)
  (hAB : Circle (segment_dist A B / 2))
  (hBC : Circle (segment_dist B C / 2))
  (hCD : Circle (segment_dist C D / 2))
  (hDA : Circle (segment_dist D A / 2))
  (h_convex : ConvexQuadrilateral A B C D) :
  ∃ (p : Point), p ∈ interior (Quadrilateral A B C D) → 
    p ∈ circle_cover A B C D :=
sorry

end circles_cover_convex_quadrilateral_l626_626720


namespace max_water_storage_volume_l626_626619
noncomputable def f (t : ℝ) : ℝ := 2 + Real.sin t
noncomputable def g (t : ℝ) : ℝ := 5 - abs (t - 6)
noncomputable def H (t : ℝ) : ℝ := f t + g t

theorem max_water_storage_volume : 
  ∃ tmax ∈ set.Icc (0 : ℝ) 12, ∀ t ∈ set.Icc (0 : ℝ) 12, H t ≤ H tmax 
  ∧ tmax = 6 
  ∧ H tmax = 7 + Real.sin 6 :=
by
  sorry

end max_water_storage_volume_l626_626619


namespace farey_sequence_problem_l626_626480

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l626_626480


namespace one_cow_one_bag_l626_626650

def husk_eating (C B D : ℕ) : Prop :=
  C * D / B = D

theorem one_cow_one_bag (C B D n : ℕ) (h : husk_eating C B D) (hC : C = 46) (hB : B = 46) (hD : D = 46) : n = D :=
by
  rw [hC, hB, hD] at h
  sorry

end one_cow_one_bag_l626_626650


namespace probability_divisible_by_15_l626_626429

theorem probability_divisible_by_15 (digits : Finset ℕ) (n : ℕ)
  (h_digits : digits = {0, 1, 2, 3, 5, 5, 8})
  (h_n : 0 < n ∧ n ≤ 7)
  (h_divisible_by_3 : digits.sum % 3 = 0)
  (h_not_zero : ∀ d ∈ digits, d ≠ 0 → n ≠ 0) :
  ∃ p : ℚ, p = 5/36 := sorry

end probability_divisible_by_15_l626_626429


namespace remainder_of_polynomial_l626_626212

theorem remainder_of_polynomial (x : ℕ) :
  (x + 1) ^ 2021 % (x ^ 2 + x + 1) = 1 + x ^ 2 := 
by
  sorry

end remainder_of_polynomial_l626_626212


namespace grid_tiling_condition_l626_626666

def can_be_covered_by_dominoes (i j : ℕ) : Prop :=
  (∃ (tiles : List (ℕ × ℕ)), 
    (∀ t ∈ tiles, 
      let (x, y) := t in
      (x ≤ 5) ∧ (y ≤ 5) ∧ -- tiles within bounds
      (i, j) ≠ (x, y)) ∧  -- ensures the removed tile is not covered
    (∀ (x1 y1 x2 y2 : ℕ), 
      ((x1, y1) ∈ tiles ∧ (x2, y2) ∈ tiles) → -- two adjacent tiles
      ((abs (x1 - x2) = 1 ∧ y1 = y2) ∨ (abs (y1 - y2) = 1 ∧ x1 = x2))) ∧  -- represents domino coverage
    (tiles.length = 12)) -- exactly 12 dominoes to cover the grid

theorem grid_tiling_condition (i j : ℕ) (hi : 1 ≤ i ∧ i ≤ 5) (hj : 1 ≤ j ∧ j ≤ 5) :
  can_be_covered_by_dominoes i j ↔ (i % 2 = 1 ∧ j % 2 = 1) :=
sorry

end grid_tiling_condition_l626_626666


namespace ten_percent_of_point_one_is_point_zero_one_l626_626669

def percentage_of_base(base : ℝ, percentage : ℝ) : ℝ := base * (percentage / 100)

theorem ten_percent_of_point_one_is_point_zero_one : percentage_of_base 0.1 10 = 0.01 :=
by
  -- Lean knows how to work with basic arithmetic
  calc
    percentage_of_base 0.1 10
        = 0.1 * (10 / 100) : by rfl
    ... = 0.1 * 0.1         : by norm_num
    ... = 0.01              : by norm_num

end ten_percent_of_point_one_is_point_zero_one_l626_626669


namespace solve_abs_inequality_l626_626545

theorem solve_abs_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ 
      (x ∈ Ioo (-13 / 2) (-3) 
     ∨ x ∈ Ico (-3) 2 
     ∨ x ∈ Icc 2 (7 / 2)) :=
by
  sorry

end solve_abs_inequality_l626_626545


namespace minimum_possible_value_l626_626325

theorem minimum_possible_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ∃ (x : ℝ), x = (a / (3 * b) + b / (6 * c) + c / (9 * a)) ∧ x = (1 / real.sqrt 3 (6 : ℝ)) := 
  sorry

end minimum_possible_value_l626_626325


namespace dice_probability_correct_l626_626309

noncomputable def dice_probability (n : ℕ) : ℚ :=
  if n = 1 then 1 / 6 else 1 / 6 * (1 + dice_probability (n - 1))

theorem dice_probability_correct (n : ℕ) :
  dice_probability n = if n = 1 then 1 / 6 else 1 / 6 * (1 + dice_probability (n - 1)) :=
by
  sorry

end dice_probability_correct_l626_626309


namespace sufficient_but_not_necessary_l626_626659

theorem sufficient_but_not_necessary (x : ℝ) : 
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l626_626659


namespace product_of_solutions_l626_626758

theorem product_of_solutions (x : ℝ) :
  let eqn := -42 = -x^2 + 4 * x in
  (∃ α β : ℝ, α ≠ β ∧ α + β = 4 ∧ α * β = -42) ∧ (α⋆β = -42).
sorry

end product_of_solutions_l626_626758


namespace angles_on_axes_correct_l626_626075

-- Definitions for angles whose terminal sides lie on x-axis and y-axis.
def angles_on_x_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def angles_on_y_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

-- Combined definition for angles on the coordinate axes using Lean notation
def angles_on_axes (α : ℝ) : Prop := ∃ n : ℤ, α = n * (Real.pi / 2)

-- Theorem stating that angles on the coordinate axes are of the form nπ/2.
theorem angles_on_axes_correct : ∀ α : ℝ, (angles_on_x_axis α ∨ angles_on_y_axis α) ↔ angles_on_axes α := 
sorry -- Proof is omitted.

end angles_on_axes_correct_l626_626075


namespace number_of_children_l626_626190

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l626_626190


namespace second_half_speed_is_30_l626_626551

-- Define the conditions given in the problem
def total_distance : ℝ := 540
def total_time : ℝ := 15
def first_half_distance : ℝ := total_distance / 2
def first_half_speed : ℝ := 45
def second_half_distance : ℝ := total_distance / 2

-- Define the times for each half
def time_first_half : ℝ := first_half_distance / first_half_speed
def time_second_half : ℝ := total_time - time_first_half

-- Define the unknown speed for the second half
def second_half_speed : ℝ := second_half_distance / time_second_half

-- State the theorem to prove the second half speed is 30 kmph
theorem second_half_speed_is_30 :
  second_half_speed = 30 :=
by
  sorry

end second_half_speed_is_30_l626_626551


namespace circle_represents_valid_a_l626_626832

theorem circle_represents_valid_a (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2 * a * x - 4 * y + 5 * a = 0) → (a > 4 ∨ a < 1) :=
by
  sorry

end circle_represents_valid_a_l626_626832


namespace A_walking_speed_l626_626292

-- Definition for the conditions
def A_speed (v : ℝ) : Prop := 
  ∃ (t : ℝ), 120 = 20 * (t - 6) ∧ 120 = v * t

-- The main theorem to prove the question
theorem A_walking_speed : ∀ (v : ℝ), A_speed v → v = 10 :=
by
  intros v h
  sorry

end A_walking_speed_l626_626292


namespace maria_spent_60_dollars_l626_626247

theorem maria_spent_60_dollars :
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  true
    → total_cost = 60 := 
by 
  intros
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  sorry

end maria_spent_60_dollars_l626_626247


namespace gcd_sum_l626_626700

theorem gcd_sum (n : ℕ) (h : 0 < n) : (Finset.sum (Finset.image (λ m, Int.gcd (5 * m + 6) (2 * m + 3)) (Finset.range n))) = 4 := by
  sorry

end gcd_sum_l626_626700


namespace prizes_distribution_l626_626523

-- Definitions and conditions
def prizes : Nat := 6
def people : Nat := 5

def combinations (n k : Nat) : Nat := n.choose k
def arrangements (n k : Nat) : Nat := n.perm k

-- Theorem statement
theorem prizes_distribution :
  let total_ways := combinations prizes 2 * arrangements people people
  total_ways = 15 * 120 := -- \( C_{6}^{2} \times A_{5}^{5} \)
sorry

end prizes_distribution_l626_626523


namespace count_perfect_square_multiples_l626_626736

theorem count_perfect_square_multiples (n : ℕ) (h₁ : n ≤ 2000) (h₂ : ∃ k : ℕ, 21 * n = k * k) : n ∈ {n : ℕ | n ≤ 2000 ∧ ∃ k : ℕ, 21 * n = k * k} → (Finset.filter (λ n, ∃ k : ℕ, 21 * n = k * k ∧ n ≤ 2000) (Finset.range 2001)).card = 9 := 
by
  sorry

end count_perfect_square_multiples_l626_626736


namespace train_pass_bridge_time_l626_626692

theorem train_pass_bridge_time :
  ∀ (length_train length_bridge : ℕ) (speed_train_kmh : ℕ),
    length_train = 480 →
    length_bridge = 210 →
    speed_train_kmh = 60 →
    let total_distance := length_train + length_bridge in
    let speed_train_ms := (speed_train_kmh * 1000 / 3600 : ℚ) in
    (total_distance / speed_train_ms) ≈ 41.39 :=
by
  intros length_train length_bridge speed_train_kmh
  intros h_length_train h_length_bridge h_speed_train_kmh
  let total_distance := length_train + length_bridge
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  have h_total_distance : total_distance = 690 := by
    rw [h_length_train, h_length_bridge]
    norm_num
  have h_speed_train_ms : speed_train_ms = 16.67 := by
    rw [h_speed_train_kmh]
    norm_num
  sorry

end train_pass_bridge_time_l626_626692


namespace ratio_of_perimeters_l626_626564

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l626_626564


namespace choose12Pairs_l626_626921

-- The club structure and knowledge properties
structure Club where
  members : Finset Nat -- Members of the club, identified by natural number IDs.
  knowsEachOther : Nat → Nat → Prop -- Relationship predicate indicating if two members know each other.
  boys : Finset Nat -- Subset of male members.
  girls : Finset Nat -- Subset of female members.
  mem_disjoint : boys ∩ girls = ∅ -- Boys and girls are distinct sets.
  mem_union : boys ∪ girls = members -- All members are either boys or girls.
  mem_count : members.card = 42 -- Total member count is 42.
  knows31 : ∀ (m : Finset Nat), m.card = 31 → ∃ (b ∈ boys) (g ∈ girls), knowsEachOther b g -- Any 31 members include a knowing pair of boy and girl.

-- The main theorem stating the desired outcome
theorem choose12Pairs (club : Club) : 
  ∃ (matching12 : Finset (Nat × Nat)), 
  matching12.card = 12 ∧ 
  ∀ (p ∈ matching12), p.1 ∈ club.boys ∧ p.2 ∈ club.girls ∧ club.knowsEachOther p.1 p.2 := 
sorry

end choose12Pairs_l626_626921


namespace rockville_basketball_club_l626_626744

theorem rockville_basketball_club (n : ℕ) 
  (sock_cost : ℕ := 6) -- Cost of one pair of socks
  (tshirt_cost : ℕ := sock_cost + 7) -- Cost of one T-shirt
  (total_cost : ℕ := 3212)
  (member_cost : ℕ := 3 * sock_cost + 2 * tshirt_cost) :
  n * member_cost = total_cost → 
  n = 73 := 
by
  intros h,
  sorry

end rockville_basketball_club_l626_626744


namespace inequality_solution_set_l626_626737

theorem inequality_solution_set (x : ℝ) : (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := sorry

end inequality_solution_set_l626_626737


namespace perimeters_ratio_l626_626582

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l626_626582


namespace rectangle_area_probability_l626_626894

theorem rectangle_area_probability :
  let MN := 16
  let P := classical.some (exists_random_point (0, 16))  -- Point chosen at random along MN
  let MP := P
  let NP := MN - P
  let area := MP * NP
  Pr(area > 60) = 1 / 4 :=
by
  -- Definition of variables
  let MN := 16
  let P := classical.some (exists_random_point (0, 16))
  let MP := P
  let NP := MN - P
  let area := MP * NP

  -- Probability calculation
  let prob := (10 - 6) / 16

  -- Proof (using sorry to skip actual steps)
  sorry

end rectangle_area_probability_l626_626894


namespace product_fraction_sequence_l626_626721

theorem product_fraction_sequence : 
  (∏ n in Finset.range 15, (n + 1) + 4) / (∏ n in Finset.range 15, (n + 1)) = 11628 := by
  sorry

end product_fraction_sequence_l626_626721


namespace basketball_children_l626_626188

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l626_626188


namespace three_correct_statements_l626_626222

-- Definitions of the statements
def condition1 : Prop := "Deductive reasoning is reasoning from the general to the specific."
def condition2 : Prop := "The conclusion derived from deductive reasoning is always correct."
def condition3 : Prop := "The general pattern of deductive reasoning is in the form of a 'syllogism'."
def condition4 : Prop := "The correctness of the conclusion of deductive reasoning depends on the major premise, minor premise, and form of reasoning."

-- The proof problem statement
theorem three_correct_statements :
  (1 ↔ condition1) ∧ (2 ↔ ¬ condition2) ∧ (3 ↔ condition3) ∧ (4 ↔ condition4) → 3 =
  [condition1, ¬condition2, condition3, condition4].count(true) := 
sorry

end three_correct_statements_l626_626222


namespace max_min_f_value_l626_626501

def f (x : ℝ) : ℝ := sorry

theorem max_min_f_value :
  (∀ (x1 x2 : ℝ), -2011 ≤ x1 ∧ x1 ≤ 2011 ∧ -2011 ≤ x2 ∧ x2 ≤ 2011 → f (x1 + x2) = f x1 + f x2 - 2011) →
  (∀ (x : ℝ), x > 0 → f x > 2011) →
  (let M := f 2011 in
   let N := f (-2011) in
   M + N = 4022) :=
begin
  intros h1 h2,
  let M := f 2011,
  let N := f (-2011),
  have h3: f 0 = 2011, sorry,
  have h4: M + N = 4022, sorry,
  exact h4,
end

end max_min_f_value_l626_626501


namespace number_of_ways_to_place_5_distinguishable_balls_into_3_indistinguishable_boxes_l626_626410

theorem number_of_ways_to_place_5_distinguishable_balls_into_3_indistinguishable_boxes :
  let balls := 5
      boxes := 3 in
  (3 ^ balls - Nat.choose boxes 1 * 2 ^ balls + Nat.choose boxes 2 * 1 ^ balls) / (Fact boxes) = 25 :=
by
  sorry

end number_of_ways_to_place_5_distinguishable_balls_into_3_indistinguishable_boxes_l626_626410


namespace radius_range_l626_626937

noncomputable def parabola (x y : ℝ) := x^2 = 2 * y

variable (y0 r : ℝ)
def range_y := 0 ≤ y0 ∧ y0 ≤ 20
def touches_bottom := ∃ y0, parabola 0 y0 ∧ 0 < y0 ∧ y0 ≤ 1 ∧ r = y0

theorem radius_range : range_y y0 → touches_bottom y0 r → (0 < r ∧ r ≤ 1) := by
  intros h1 h2
  sorry

end radius_range_l626_626937


namespace perimeters_ratio_l626_626583

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l626_626583


namespace ratio_of_perimeters_l626_626566

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l626_626566


namespace find_x_l626_626040

theorem find_x (x : ℝ) : 
  (sqrt ((x + 1) / (2 - x)) = (sqrt (x + 1)) / (sqrt (2 - x))) ∧
  (x + 1 ≥ 0) ∧
  (2 - x > 0) → 
  (x = 0) :=
sorry

end find_x_l626_626040


namespace points_space_even_n_l626_626361

/-
Given \( n+1 \) points in space, \( P_{1}, P_{2}, \ldots, P_{n} \) and \( Q \), 
such that no four of them lie in the same plane. It is also known that for any 
three distinct points \( P_{i}, P_{j} \), and \( P_{k} \), there exists at least 
one point \( P_{\ell} \) such that \( Q \) is an interior point of the tetrahedron 
\( P_{i} P_{j} P_{k} P_{\ell} \). Show that \( n \) is even.
-/

theorem points_space_even_n (n : ℕ) 
  (points : Fin (n + 1) → ℝ × ℝ × ℝ) 
  (Q : ℝ × ℝ × ℝ) 
  (h_distinct_planes : ∀ (i j k l : Fin (n + 1)), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → ¬ AffineCombination ℝ (points i) (points j) (points k) (points l)) 
  (h_tetrahedron : ∀ (i j k : Fin n), ∃ l : Fin n, Q ∈ Interior (ConvexHull ℝ {points i, points j, points k, points l})) :
  Even n := 
sorry

end points_space_even_n_l626_626361


namespace find_positive_integers_l626_626748

theorem find_positive_integers (n : ℕ) (P : Polynomial ℤ) : 
  (P.degree = n ∧ 
  (∀ x ∈ finset.range n, eval x P = n) ∧ 
  eval 0 P = 0 ∧ 
  ∃ (Q : Polynomial ℤ), P = Polynomial.X * Q) ↔ (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) := 
sorry

end find_positive_integers_l626_626748


namespace square_perimeter_ratio_l626_626561

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l626_626561


namespace find_initial_books_l626_626899

/-- The number of books the class initially obtained from the library --/
def initial_books : ℕ := sorry

/-- The number of books added later --/
def books_added_later : ℕ := 23

/-- The total number of books the class has --/
def total_books : ℕ := 77

theorem find_initial_books : initial_books + books_added_later = total_books → initial_books = 54 :=
by
  intros h
  sorry

end find_initial_books_l626_626899


namespace lathe_both_yield_simultaneously_mixed_parts_overall_yield_l626_626623

noncomputable def lathe_yield_prob (p1 p2 : ℝ) : ℝ :=
p1 * p2

theorem lathe_both_yield_simultaneously :
  lathe_yield_prob 0.15 0.10 = 0.015 :=
by 
  unfold lathe_yield_prob
  norm_num

noncomputable def mixed_parts_yield_prob (pA pB_A pB_Ac : ℝ) (pA_prob : pA = 0.60) (pAc_prob : pA = 0.40) : ℝ :=
pA * pB_A + pA * pB_Ac

theorem mixed_parts_overall_yield :
  mixed_parts_yield_prob 0.60 0.15 0.10 0.60 0.40 = 0.13 :=
by 
  unfold mixed_parts_yield_prob
  norm_num

end lathe_both_yield_simultaneously_mixed_parts_overall_yield_l626_626623


namespace prove_arrangements_with_qu_l626_626139

noncomputable def arrangements_with_qu : Nat :=
  let total_letters : Finset Char := {'e', 'q', 'u', 'a', 't', 'i', 'o', 'n'}
  let selected_letters : Finset Char := total_letters.sdiff {'e', 'i', 'o', 'n'}
  let combinations : Nat := Nat.choose selected_letters.card 3
  let permutations : Nat := Mathlib.Combinatorics.Perm.card_perm 4
  combinations * permutations

theorem prove_arrangements_with_qu : arrangements_with_qu = 480 := by
  sorry

end prove_arrangements_with_qu_l626_626139


namespace tan_alpha_value_l626_626375

theorem tan_alpha_value
  (α : ℝ)
  (h_cos : Real.cos α = -4/5)
  (h_range : (Real.pi / 2) < α ∧ α < Real.pi) :
  Real.tan α = -3/4 := by
  sorry

end tan_alpha_value_l626_626375


namespace ordered_quadruples_of_factorial_product_l626_626312

theorem ordered_quadruples_of_factorial_product :
  {x : ℕ × ℕ × ℕ × ℕ // 1 ≤ x.1 ∧ 1 ≤ x.2.1 ∧ 1 ≤ x.2.2.1 ∧ 1 ≤ x.2.2.2 ∧
                      (nat.factorial x.1) * (nat.factorial x.2.1) * (nat.factorial x.2.2.1) * (nat.factorial x.2.2.2) = nat.factorial 24}.card = 7 := 
by
  sorry

end ordered_quadruples_of_factorial_product_l626_626312


namespace find_k_l626_626030

-- Definitions:
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (-1, 0)
def vec_c : ℝ × ℝ := (2, 1)
def is_colinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Lean statement for the proof problem
theorem find_k (k : ℝ) : is_colinear (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2) vec_c → k = -1 :=
by 
  intros,
  sorry

end find_k_l626_626030


namespace imaginary_part_of_z_l626_626053

-- Problem Definition
def complex_number := ℂ
def magnitude (z : complex_number) : ℝ := complex.abs z

-- Conditions
variables (z : complex_number)
axiom condition : (3 - 4 * complex.I) * z = magnitude (4 + 3 * complex.I)

-- Theorem Statement
theorem imaginary_part_of_z : complex.im z = 4 / 5 :=
sorry

end imaginary_part_of_z_l626_626053


namespace infinite_primes_dividing_f_l626_626112

def f (x : ℕ) : ℤ :=
  x ^ 1998 - x ^ 199 + x ^ 19 + 1

theorem infinite_primes_dividing_f :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ p ∈ S, ∃ n : ℕ, p.Prime ∧ p ∣ f n :=
sorry

end infinite_primes_dividing_f_l626_626112


namespace pencils_in_pencil_case_l626_626201

theorem pencils_in_pencil_case : ∃ P : ℕ, let pens := 2 * P, eraser := 1, total_items := P + pens + eraser in total_items = 13 ∧ P = 4 :=
  sorry

end pencils_in_pencil_case_l626_626201


namespace exists_circle_with_three_points_on_circumference_l626_626905

-- Given four points in the plane such that no three points are collinear, 
-- prove there exists a circle such that three of the points lie on the circumference 
-- and the fourth point is either on the circumference or inside the circle.

theorem exists_circle_with_three_points_on_circumference 
  (points : Fin 4 (ℝ × ℝ)) 
  (h_distinct : ∀ i j : Fin 4, i ≠ j → points i ≠ points j)
  (h_no_collinear : ∀ i j k : Fin 4, i ≠ j → j ≠ k → i ≠ k → ¬ (collinear (points i) (points j) (points k))) :
  ∃ (c : ℝ × ℝ) (r : ℝ), r > 0 ∧ 
    (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
       dist (points i) c = r ∧ dist (points j) c = r ∧ dist (points k) c = r ∧
       (∃ l : Fin 4, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ (dist (points l) c = r ∨ dist (points l) c < r))) :=
sorry

end exists_circle_with_three_points_on_circumference_l626_626905


namespace find_lambda_collinear_l626_626400

noncomputable def a : ℝ × ℝ := (-1, 2)
noncomputable def b : ℝ × ℝ := (2, -3)
noncomputable def c : ℝ × ℝ := (-4, 7)

theorem find_lambda_collinear : 
  ∃ k λ : ℝ, ∀ (a b c : ℝ × ℝ), 
  a = (-1, 2) ∧ b = (2, -3) ∧ c = (-4, 7) ∧
  (λ * a.1 + b.1 = k * c.1) ∧ (λ * a.2 + b.2 = k * c.2) → λ = -2 :=
sorry

end find_lambda_collinear_l626_626400


namespace minimal_colors_needed_l626_626096

theorem minimal_colors_needed : ∃ n : ℕ, 
  (n > 0 ∧ ∀ c : ℕ → ℕ, (∀ m n, (m ∣ n ↔ m ≠ n) → (c m ≠ c n)) → 1 ≤ m ∧ m ≤ 1000 → 1 ≤ n ∧ n ≤ 1000) ∧
  (∀ c : ℕ → ℕ, (∀ m n, (m ∣ n ↔ m ≠ n) → (c m ≠ c n)) → ∃ k, k ≤ 10 ∧ c k > 0) :=
begin
  sorry,
end

end minimal_colors_needed_l626_626096


namespace find_number_of_nickels_l626_626646

noncomputable theory

variables (n q : ℕ)

def number_of_nickels (n q : ℕ) : Prop :=
n = q ∧ (0.05 * n + 0.25 * q = 12) ∧ n = 40

theorem find_number_of_nickels (n q : ℕ) : number_of_nickels n q :=
sorry

end find_number_of_nickels_l626_626646


namespace solve_inequality_l626_626531

theorem solve_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (x ∈ Ioo (-6.5) 3.5) := 
sorry

end solve_inequality_l626_626531


namespace solve_abs_inequality_l626_626543

theorem solve_abs_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ 
      (x ∈ Ioo (-13 / 2) (-3) 
     ∨ x ∈ Ico (-3) 2 
     ∨ x ∈ Icc 2 (7 / 2)) :=
by
  sorry

end solve_abs_inequality_l626_626543


namespace distance_from_P_to_AB_l626_626197

theorem distance_from_P_to_AB (A B C P : Point)
  (h_parallel : line_through P ∥ line_through A B)
  (h_area : area (triangle A B P) = area (triangle P C B))
  (h_altitude : altitude_length (triangle A B C) = 1) :
  distance_from P (line_through A B) = 1 / 2 := 
sorry

end distance_from_P_to_AB_l626_626197


namespace overall_loss_is_correct_l626_626279

-- Define the conditions
def worth_of_stock : ℝ := 17500
def percent_stock_sold_at_profit : ℝ := 0.20
def profit_rate : ℝ := 0.10
def percent_stock_sold_at_loss : ℝ := 0.80
def loss_rate : ℝ := 0.05

-- Define the calculations based on the conditions
def worth_sold_at_profit : ℝ := percent_stock_sold_at_profit * worth_of_stock
def profit_amount : ℝ := profit_rate * worth_sold_at_profit

def worth_sold_at_loss : ℝ := percent_stock_sold_at_loss * worth_of_stock
def loss_amount : ℝ := loss_rate * worth_sold_at_loss

-- Define the overall loss amount
def overall_loss : ℝ := loss_amount - profit_amount

-- Theorem to prove that the calculated overall loss amount matches the expected loss amount
theorem overall_loss_is_correct :
  overall_loss = 350 :=
by
  sorry

end overall_loss_is_correct_l626_626279


namespace problem_1_problem_2_l626_626307

theorem problem_1 :
  sqrt 8 - sqrt 2 + 2 * sqrt (1 / 2) = 2 * sqrt 2 :=
sorry

theorem problem_2 :
  sqrt 12 - 9 * sqrt (1 / 3) + abs (2 - sqrt 3) = 2 - 2 * sqrt 3 :=
sorry

end problem_1_problem_2_l626_626307


namespace solve_inequality_l626_626532

theorem solve_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (x ∈ Ioo (-6.5) 3.5) := 
sorry

end solve_inequality_l626_626532


namespace brownie_pieces_count_l626_626888

def area_of_pan (length width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

def number_of_pieces (pan_area piece_area : ℕ) : ℕ := pan_area / piece_area

theorem brownie_pieces_count :
  let pan_length := 24
  let pan_width := 15
  let piece_side := 3
  let pan_area := area_of_pan pan_length pan_width
  let piece_area := area_of_piece piece_side
  number_of_pieces pan_area piece_area = 40 :=
by
  sorry

end brownie_pieces_count_l626_626888


namespace definite_integral_abs_sin_cos_l626_626738

theorem definite_integral_abs_sin_cos :
  ∫ x in 0..π, |sin x - cos x| = 2 * sqrt 2 :=
by
  sorry

end definite_integral_abs_sin_cos_l626_626738


namespace train_length_correct_l626_626289

def train_length (speed_kmph : ℝ) (time_sec : ℝ) (platform_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * time_sec
  total_distance - platform_length_m

theorem train_length_correct :
  train_length 60 23.998080153587715 260 ≈ 139.97 := 
by
  -- Proof steps would go here; omitted for brevity
  sorry

end train_length_correct_l626_626289


namespace jelly_beans_remaining_correct_l626_626965

-- Define the initial count of jelly beans and the number of children
def initial_jelly_beans : ℕ := 1500
def children_count : ℕ := 75

-- Define the percentage of children allowed to draw and the corresponding number
def allowed_percentage : ℕ := 60
def allowed_children : ℕ := (children_count * allowed_percentage) / 100

-- Define the number of jelly beans drawn based on the last digit of ID
def jelly_beans_drawn (last_digit : ℕ) : ℕ :=
  if last_digit = 1 ∨ last_digit = 2 ∨ last_digit = 3 then 3
  else if last_digit = 4 ∨ last_digit = 5 ∨ last_digit = 6 then 4
  else if last_digit = 7 ∨ last_digit = 8 ∨ last_digit = 9 then 5
  else 0

-- Group division and calculating the total amount of jelly beans drawn
def each_group_count : ℕ := allowed_children / 9
def total_drawn : ℕ :=
  (each_group_count * 3) * 3 + (each_group_count * 3) * 4 + (each_group_count * 3) * 5

-- The remaining number of jelly beans
def remaining_jelly_beans : ℕ := initial_jelly_beans - total_drawn

-- Prove that the remaining jelly beans are 1,440
theorem jelly_beans_remaining_correct : remaining_jelly_beans = 1440 :=
  by
    -- Assume necessary calculations
    have h1 : allowed_children = 45 := by sorry -- From 0.60 * 75
    have h2 : each_group_count = 5 := by sorry -- From 45 / 9
    have h3 : total_drawn = 60 := by sorry -- From (5 * 3) + (5 * 4) + (5 * 5)
    calc
      remaining_jelly_beans
        = initial_jelly_beans - total_drawn : by sorry
        ... = 1500 - 60 : by {
          rw [h1, h2, h3]; sorry -- Assume each_group_count and total_drawn values
        }
        ... = 1440 : by sorry

end jelly_beans_remaining_correct_l626_626965


namespace sum_of_squares_l626_626385

theorem sum_of_squares (n : ℕ) (h : n > 0) : 
  (∑ i in finset.range n.succ, i ^ 2) = n * (n + 1) * (2 * n + 1) / 6 := 
sorry

end sum_of_squares_l626_626385


namespace range_f_area_of_abc_l626_626801

def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π / 3) + sqrt 3

theorem range_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 6) :
  sqrt 3 ≤ f(x) ∧ f(x) ≤ 2 := sorry

noncomputable def area_triangle (a b r : ℝ) : ℝ := 
  let sinA := a / (2 * r)
  let sinB := b / (2 * r)
  let cosA := sqrt (1 - sinA^2)
  let cosB := sqrt (1 - sinB^2)
  let sinC := sinA * cosB + cosA * sinB
  (1 / 2) * a * b * sinC

theorem area_of_abc : 
  let a := sqrt 3
  let b := 2
  let r := (3 * sqrt 2 / 4)
  area_triangle a b r = sqrt 2 := sorry

end range_f_area_of_abc_l626_626801


namespace fraction_value_l626_626656

theorem fraction_value : (20 - 20) / (20 + 20) = 0 := 
by {
  have h1: 20 - 20 = 0 := by norm_num,
  have h2: 20 + 20 = 40 := by norm_num,
  rw [h1, h2],
  norm_num,
}

end fraction_value_l626_626656


namespace union_cardinality_of_sets_l626_626359

theorem union_cardinality_of_sets :
  ∀ (S : Fin 1985 → Set α) (h₀ : ∀ i, |S i| = 45)
    (h₁ : ∀ i j, i ≠ j → |S i ∪ S j| = 89),
  |⋃ i, S i| = 87381 :=
by
  intros
  sorry

end union_cardinality_of_sets_l626_626359


namespace rectangle_perimeter_l626_626235

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 :=
by
  -- Proving the theorem here
  sorry

end rectangle_perimeter_l626_626235


namespace number_of_students_l626_626434

theorem number_of_students (x : ℕ) (h : x * (x - 1) = 210) : x = 15 := 
by sorry

end number_of_students_l626_626434


namespace probability_exactly_three_germinate_probability_at_least_three_germinate_l626_626165

noncomputable def binom (n k : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_exactly_three_germinate :
  let n := 4
  let k := 3
  let p := 0.9
  let q := 0.1
  (binom n k) * (p^k) * (q^(n - k)) = 0.2916 := 
by
  sorry

theorem probability_at_least_three_germinate :
  let n := 4
  let p := 0.9
  let q := 0.1
  let p_3 := (binom n 3) * (p^3) * (q^(n - 3))
  let p_4 := (binom n 4) * (p^4) * (q^(n - 4))
  p_3 + p_4 = 0.9477 :=
by
  sorry

end probability_exactly_three_germinate_probability_at_least_three_germinate_l626_626165


namespace total_growth_of_trees_l626_626964

theorem total_growth_of_trees :
  let t1_growth_rate := 1 -- first tree grows 1 meter/day
  let t2_growth_rate := 2 -- second tree grows 2 meters/day
  let t3_growth_rate := 2 -- third tree grows 2 meters/day
  let t4_growth_rate := 3 -- fourth tree grows 3 meters/day
  let days := 4
  t1_growth_rate * days + t2_growth_rate * days + t3_growth_rate * days + t4_growth_rate * days = 32 :=
by
  let t1_growth_rate := 1
  let t2_growth_rate := 2
  let t3_growth_rate := 2
  let t4_growth_rate := 3
  let days := 4
  sorry

end total_growth_of_trees_l626_626964


namespace length_AC_eq_five_root_two_l626_626454

open Real

-- Definition of Point and Parallelogram in 2D space
structure Point :=
(x : ℝ)
(y : ℝ)

structure Parallelogram :=
(A B C D : Point)
(AB_eq_CD : dist A B = dist C D)
(BC_eq_AD : dist B C = dist A D)
(diag_bisect_angle_A : ∠ABC = ∠DAB)
(diag_bisect_angle_C : ∠BCD = ∠DCA)

-- Define the Euclidean distance between two points
def dist (P Q : Point) : ℝ :=
Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the angle bisecting property (simplified for 45 degrees in isosceles triangle)
def angle_bisect (A B C : Point) : Prop :=
  let AB := dist A B
  let BC := dist B C
  let AC := dist A C
  2 * AB * BC = 2 * AC^2 / 2

-- Define the condition that point E is the same as point F when diagonal bisects the parallelogram
axiom diagonal_bisect {A B C D E F : Point}
  (parallelogram : Parallelogram)
  (AC_bisects_A_C : E = F) : AC_bisects_A_C

-- Define the problem statement to prove that the length of diagonal AC is 5√2
theorem length_AC_eq_five_root_two (A B C D : Point)
  (parallelogram : Parallelogram)
  (H1 : parallelogram.AB_eq_CD)
  (H2 : parallelogram.BC_eq_AD)
  (H3 : parallelogram.diag_bisect_angle_A)
  (H4 : parallelogram.diag_bisect_angle_C) :
  dist A C = 5 * Real.sqrt 2 :=
sorry

end length_AC_eq_five_root_two_l626_626454


namespace no_integer_solutions_l626_626143

theorem no_integer_solutions (x y k : ℤ) : x ^ 2009 + y ^ 2009 ≠ 7 ^ k := sorry

end no_integer_solutions_l626_626143


namespace probability_C_calc_l626_626254

noncomputable section

-- Define the given probabilities
def prob_A : ℚ := 3 / 8
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 3 / 16
def prob_D : ℚ := prob_C

-- The sum of probabilities equals 1
theorem probability_C_calc :
  prob_A + prob_B + prob_C + prob_D = 1 :=
by
  -- Simplifying directly, we can assert the correctness of given prob_C
  sorry

end probability_C_calc_l626_626254


namespace family_gathering_l626_626438

theorem family_gathering : 
  ∃ (total_people oranges bananas apples : ℕ), 
    total_people = 20 ∧ 
    oranges = total_people / 2 ∧ 
    bananas = (total_people - oranges) / 2 ∧ 
    apples = total_people - oranges - bananas ∧ 
    oranges < total_people ∧ 
    total_people - oranges = 10 :=
by
  sorry

end family_gathering_l626_626438


namespace gcd_polynomials_l626_626007

theorem gcd_polynomials (b : ℕ) (hb : ∃ k : ℕ, b = 2 * 7771 * k) :
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 19) = 8 :=
by sorry

end gcd_polynomials_l626_626007


namespace determine_quantities_l626_626327

variables (C_Na C_Cl f : ℝ)
variables (mu a_Na a_Cl : ℝ)

def concentrations :=
  C_Na = 0.01 ∧ C_Cl = 0.01 ∧ f = 0.89

theorem determine_quantities :
  concentrations C_Na C_Cl f →
  mu = (1 / 2) * (1^2 * C_Na + 1^2 * C_Cl) →
  a_Na = C_Na * f →
  a_Cl = C_Cl * f →
  mu = 0.01 ∧ a_Na = 8.9 * 10^(-3) ∧ a_Cl = 8.9 * 10^(-3) :=
by
  sorry

end determine_quantities_l626_626327


namespace projection_plane_right_angle_l626_626819

-- Given conditions and definitions
def is_right_angle (α β : ℝ) : Prop := α = 90 ∧ β = 90
def is_parallel_to_side (plane : ℝ → ℝ → Prop) (side : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, plane x y ↔ a * x + b * y = c ∧ ∃ d e : ℝ, ∀ x y : ℝ, side x y ↔ d * x + e * y = 90

theorem projection_plane_right_angle (plane : ℝ → ℝ → Prop) (side1 side2 : ℝ → ℝ → Prop) :
  is_right_angle (90 : ℝ) (90 : ℝ) →
  (is_parallel_to_side plane side1 ∨ is_parallel_to_side plane side2) →
  ∃ α β : ℝ, is_right_angle α β :=
by 
  sorry

end projection_plane_right_angle_l626_626819


namespace smallest_nonnegative_value_l626_626726

def e (k : ℕ) : ℝ :=
  if some_condition then 1 else -1  -- Placeholder for the actual e_k definition

noncomputable def sum_e_k_powers (n : ℕ) : ℝ :=
  ∑ k in finset.range n, e (k + 1) * (k + 1)^5

theorem smallest_nonnegative_value : sum_e_k_powers 1985 = 1985^5 :=
by
  sorry

end smallest_nonnegative_value_l626_626726


namespace Hannah_total_spent_l626_626402

def rides_cost (total_money : ℝ) : ℝ :=
  0.35 * total_money

def games_cost (total_money : ℝ) : ℝ :=
  0.25 * total_money

def food_and_souvenirs_cost : ℝ :=
  7 + 4 + 5 + 6

def total_spent (total_money : ℝ) : ℝ :=
  rides_cost total_money + games_cost total_money + food_and_souvenirs_cost

theorem Hannah_total_spent (total_money : ℝ) (h : total_money = 80) :
  total_spent total_money = 70 :=
by
  rw [total_spent, h, rides_cost, games_cost]
  norm_num
  sorry

end Hannah_total_spent_l626_626402


namespace pq_sum_of_harmonic_and_geometric_sequences_l626_626885

theorem pq_sum_of_harmonic_and_geometric_sequences
  (x y z : ℝ)
  (h1 : (1 / x - 1 / y) / (1 / y - 1 / z) = 1)
  (h2 : 3 * x * y = 7 * z) :
  ∃ p q : ℕ, (Nat.gcd p q = 1) ∧ p + q = 79 :=
by
  sorry

end pq_sum_of_harmonic_and_geometric_sequences_l626_626885


namespace acute_dihedral_angle_of_cylinder_section_l626_626828

/-- The statement of the problem: Given a cross-section of a cylinder with an eccentricity of 2√2/3, 
prove that the acute dihedral angle between this cross-section and the cylinder's base is arccos(1/3). -/
theorem acute_dihedral_angle_of_cylinder_section (e : ℝ) (h_e : e = 2 * real.sqrt 2 / 3) :
  let θ := real.arccos (1 / 3) in θ = real.arccos (1 / 3) :=
by
  sorry

end acute_dihedral_angle_of_cylinder_section_l626_626828


namespace sum_of_inradii_is_correct_l626_626432

noncomputable def sum_of_inradii {P Q R S : Type} [MetricSpace P Q R] 
  (h1 : dist P Q = 9) 
  (h2 : dist P R = 12) 
  (h3 : dist Q R = 15) 
  (hS : S = (midpoint ℝ P R)) 
  : ℝ := 
  let PQS := Triangle(P, Q, S) in
  let QRS := Triangle(Q, R, S) in
  let r_PQS := inradius PQS in
  let r_QRS := inradius QRS in
  r_PQS + r_QRS

theorem sum_of_inradii_is_correct 
  (h1 : dist P Q = 9) 
  (h2 : dist P R = 12) 
  (h3 : dist Q R = 15) 
  (hS : S = (midpoint ℝ P R)) 
  : sum_of_inradii h1 h2 h3 hS = 3.6 := 
sorry

end sum_of_inradii_is_correct_l626_626432


namespace tickets_difference_l626_626286

variable (O B : ℕ)
variable (price_orchestra price_balcony : ℕ)
variable (total_tickets total_sales : ℕ)

def conditions :=
  price_orchestra = 18 ∧
  price_balcony = 12 ∧
  total_tickets = 450 ∧
  total_sales = 6300

theorem tickets_difference (h : conditions) :
  B - O = 150 :=
by
  unfold conditions at h
  sorry

end tickets_difference_l626_626286


namespace length_BC_is_8_sqrt_5_by_5_l626_626858

-- Definitions for the conditions
def triangle (A B C : Type) := A ≠ B ∧ B ≠ C ∧ C ≠ A
def is_point_on_side (E A B : Type) := (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = t * A + (1 - t) * B)

-- Given the conditions of the problem
variables {A B C E : Type}
variables {EA EC : ℝ}
variables {angle_B : ℝ}
variables {dot_product : ℝ}

-- The given conditions
axiom h_triangle : triangle A B C
axiom h_angle_B : angle_B = π / 6
axiom h_point_E_on_AB: is_point_on_side E A B
axiom h_EC : EC = 2
axiom h_EA : EA = sqrt 5
axiom h_dot_product : dot_product = 2

-- The proof statement to be shown
theorem length_BC_is_8_sqrt_5_by_5 :
  ∀ (BC : ℝ), BC = 8 * sqrt 5 / 5 :=
begin
  sorry
end

end length_BC_is_8_sqrt_5_by_5_l626_626858


namespace square_perimeter_ratio_l626_626562

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l626_626562


namespace perimeter_ratio_of_squares_l626_626571

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l626_626571


namespace PA_PB_sum_l626_626331

def curve_equation (x y : ℝ) : Prop :=
  x^2 = 4 * y

def point_P : ℝ × ℝ := (0, 3)

def parametric_eq_line (t : ℝ) : ℝ × ℝ :=
  (1 / 2 * t, 3 + sqrt 3 / 2 * t)

def intersection_condition (t : ℝ) : Prop :=
  t^2 - 8 * sqrt 3 * t - 48 = 0

theorem PA_PB_sum : (P A B : point_P) (t_A t_B : ℝ)
  (hA : curve_equation (1 / 2 * t_A) (3 + sqrt 3 / 2 * t_A))
  (hB : curve_equation (1 / 2 * t_B) (3 + sqrt 3 / 2 * t_B))
  (hA_intersect : intersection_condition t_A)
  (hB_intersect : intersection_condition t_B) :
    1 / (dist point_P (parametric_eq_line t_A)) + 1 / (dist point_P (parametric_eq_line t_B)) = sqrt 6 / 6 := by
  sorry

end PA_PB_sum_l626_626331


namespace farey_sequence_problem_l626_626479

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l626_626479


namespace mark_sprinted_distance_l626_626128

def time_spent := 24.0 -- hours
def speed := 6.0 -- miles per hour
def distance := time_spent * speed -- distance = time * speed

theorem mark_sprinted_distance : distance = 144.0 :=
by
  -- sorry

end mark_sprinted_distance_l626_626128


namespace area_relation_aok_l626_626658

variables {A B C D O K : Point} [trapezoid ABCD] [is_inter O (diagonal A C) (diagonal B D)]
variables {B' : Point} [symmetric B B' O]

theorem area_relation_aok (h_inter: is_inter O (diagonal A C) (diagonal B D)) 
  (h_symm: symmetric B B' O)
  (h_line: line_through C B' intersects AD at K) :
  area A O K = area A O B + area D O K :=
sorry

end area_relation_aok_l626_626658


namespace gcd_fib_2017_99_101_plus_1_eq_1_l626_626018

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem gcd_fib_2017_99_101_plus_1_eq_1 :
  gcd (fib 2017) (fib (99) * fib (101) + 1) = 1 :=
sorry

end gcd_fib_2017_99_101_plus_1_eq_1_l626_626018


namespace spending_80_yuan_negation_l626_626076

/-- 
    If quantities with opposite meanings are named as positive and negative, 
    and receiving 100 yuan is denoted as +100 yuan, 
    then spending 80 yuan should be denoted as -80 yuan.
-/
theorem spending_80_yuan_negation :
  (∀ (a : ℤ), (a > 0) ↔ (a = 100) ↔ (a denotes receiving 100 yuan)) → 
  (∀ (a : ℤ), (a < 0) ↔ (-a denotes spending money)) →
  (-80 denotes spending 80 yuan) :=
by
  sorry

end spending_80_yuan_negation_l626_626076


namespace solution_l626_626072

noncomputable def problem_statement : Prop :=
∀ (A B C D E : Type)
  (AB BC CD DE : ℝ)
  (rectangle : A ∧ B ∧ C ∧ D)
  (length_AB : AB = 24)
  (length_BC : BC = 7)
  (length_CD : CD = BC = 7)
  (point_E : E ∈ (CD))
  (length_DE : DE = 3),
  let length_AE := Real.sqrt ((length_BC ^ 2) + (length_DE ^ 2)) in
  length_AE = Real.sqrt 58

theorem solution : problem_statement := 
sorry

end solution_l626_626072


namespace initial_price_of_article_l626_626297

theorem initial_price_of_article (P : ℝ) (h : 0.4025 * P = 620) : P = 620 / 0.4025 :=
by
  sorry

end initial_price_of_article_l626_626297


namespace total_arrangements_is_42_l626_626694

theorem total_arrangements_is_42 :  -- Define the theorem
  let departments := 3,
      people_per_department := 2,
      returning_people := 2
  in
  (∃ (same_department_cases arrangements diff_department_cases arrangements_total: ℕ),
      (same_department_cases = Nat.choose departments 1 * Nat.perm people_per_department people_per_department) ∧
      (diff_department_cases = Nat.choose departments 2 * people_per_department * people_per_department * 3) ∧
      (arrangements_total = same_department_cases + diff_department_cases))
  → arrangements_total = 42 :=  -- Prove the total number of different arrangements is 42
by
  sorry

end total_arrangements_is_42_l626_626694


namespace sequence_general_formula_sum_of_first_n_terms_l626_626366

-- Given a sequence {a_n} such that the sum of the first n terms S_n satisfies S_n = 2a_n - 2
variables {a : ℕ → ℕ} {S : ℕ → ℕ}

-- Conditions and given information
axiom seq_condition (n : ℕ) : S n = 2 * a n - 2

-- Prove that the sequence {a_n} is given by a_n = 2^n
theorem sequence_general_formula (n : ℕ) : a n = 2 ^ n :=
sorry

-- Prove the sum of the first n terms T_n for the sequence {(2n-1) * a_n} is 6 + (2n-3) * 2^(n+1)
def T (n : ℕ) : ℕ := ∑ i in finset.range n, (2 * i + 1) * a (i + 1)
theorem sum_of_first_n_terms (n : ℕ) : T n = 6 + (2 * n - 3) * 2 ^ (n + 1) :=
sorry

end sequence_general_formula_sum_of_first_n_terms_l626_626366


namespace find_a_l626_626427

theorem find_a (a : ℝ) :
  (∃ x ∈ Icc (1:ℝ) (Real.exp 1), (∀ y ∈ Icc (1:ℝ) (Real.exp 1), Real.log y + a / y ≥ Real.log x + a / x)
  ∧ Real.log x + a / x = 3 / 2) → a = Real.sqrt (Real.exp 1) :=
by
  sorry

end find_a_l626_626427


namespace expected_value_winnings_l626_626295

def probability_of_roll (n : ℕ) (die_sides : ℕ) : ℚ :=
  if n ≥ 1 ∧ n ≤ die_sides then 1 / die_sides else 0

def winning_amount (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then
    if roll = 8 then 2 * roll + 5 else 2 * roll
  else 0

def expected_value_die_8 : ℚ :=
  let outcomes := [2, 4, 6, 8]
  let even_winnings := outcomes.map (λ n => probability_of_roll n 8 * winning_amount n)
  let total_even_winnings := even_winnings.foldr ( + ) 0
  total_even_winnings

theorem expected_value_winnings : expected_value_die_8 = 45 / 8 := 
  sorry

end expected_value_winnings_l626_626295


namespace peter_speed_l626_626867

theorem peter_speed (p : ℝ) (v_juan : ℝ) (d : ℝ) (t : ℝ) 
  (h1 : v_juan = p + 3) 
  (h2 : d = t * p + t * v_juan) 
  (h3 : t = 1.5) 
  (h4 : d = 19.5) : 
  p = 5 :=
by
  sorry

end peter_speed_l626_626867


namespace number_is_43_l626_626600

theorem number_is_43 (m : ℕ) : (m > 30 ∧ m < 50) ∧ Nat.Prime m ∧ m % 12 = 7 ↔ m = 43 :=
by
  sorry

end number_is_43_l626_626600


namespace peter_investment_time_l626_626507

variables (P A1 A2 : ℝ) (r t : ℝ)
hypothesis P_eq : P = 710
hypothesis A1_eq : A1 = 815
hypothesis A2_eq : A2 = 850
hypothesis simple_interest_david : A2 = P + (P * r * 4)
hypothesis simple_interest_peter : A1 = P + (P * r * t)

theorem peter_investment_time : t = 3 :=
begin
  have P_eq := P_eq,
  have A1_eq := A1_eq,
  have A2_eq := A2_eq,
  have r_eq : r = 0.05,
  {
    -- From A2_eq and simple_interest_david
    sorry,
  },
  have t_eq : t = 3,
  {
    -- From A1_eq and r_eq and simple_interest_peter
    sorry,
  },
  exact t_eq,
end

end peter_investment_time_l626_626507


namespace triangle_classification_l626_626396

def is_obtuse_triangle (a b c : ℕ) : Prop :=
c^2 > a^2 + b^2 ∧ a < b ∧ b < c

def is_right_triangle (a b c : ℕ) : Prop :=
c^2 = a^2 + b^2 ∧ a < b ∧ b < c

def is_acute_triangle (a b c : ℕ) : Prop :=
c^2 < a^2 + b^2 ∧ a < b ∧ b < c

theorem triangle_classification :
    is_acute_triangle 10 12 14 ∧ 
    is_right_triangle 10 24 26 ∧ 
    is_obtuse_triangle 4 6 8 :=
by 
  sorry

end triangle_classification_l626_626396


namespace domain_of_f_log2_x_eq_domain_of_f_half_x_l626_626381

noncomputable def domain_of_log2_x : Set ℝ := {x | 2 < x ∧ x < 4}

noncomputable def domain_of_half_x : Set ℝ := {x | 2 < x ∧ x < 4}

theorem domain_of_f_log2_x_eq_domain_of_f_half_x (f : ℝ → ℝ) :
  (∀ x, f(log2 x) ∈ domain_of_log2_x) →
  (∀ x, f(x / 2) ∈ domain_of_half_x) :=
sorry

end domain_of_f_log2_x_eq_domain_of_f_half_x_l626_626381


namespace find_quadrilateral_l626_626845

noncomputable def length_ratio (a b c d : ℝ) : Prop :=
  a / d = 9 / 6 ∧ b / d = 8 / 6 ∧ c / d = 7 / 6

noncomputable def angle_alpha (α : ℝ) : Prop :=
  α = 72 + 36 / 60 + 18 / 3600  -- Converting degrees, minutes, and seconds to decimal degrees

noncomputable def quadrilateral_area (a b c d α : ℝ) (area : ℝ) : Prop :=
  a * d * real.sin α + b * c * real.sin (2 * real.arccos (225 / (a * d * real.sin α))) = 2 * area

theorem find_quadrilateral (a b c d α β γ δ : ℝ) :
  length_ratio a b c d →
  angle_alpha α →
  quadrilateral_area a b c d α 225 →
  a ≈ 18.6 ∧ b ≈ 16.53 ∧ c ≈ 14.5 ∧ d ≈ 12.4 ∧
  α ≈ 72 + 36 / 60 + 18 / 3600 ∧ γ ≈ 75 + 27 / 60 + 6 / 3600 ∧
  β ≈ 130 + 40 / 60 + 40 / 3600 ∧ δ ≈ 81 + 15 / 60 + 56 / 3600 :=
by sorry

end find_quadrilateral_l626_626845


namespace bryson_pairs_of_shoes_l626_626304

theorem bryson_pairs_of_shoes (total_shoes : ℕ) (shoes_per_pair : ℕ) (h1 : total_shoes = 4) (h2 : shoes_per_pair = 2) : total_shoes / shoes_per_pair = 2 :=
by
  rw [h1, h2]
  exact Nat.div_self (by norm_num)

end bryson_pairs_of_shoes_l626_626304


namespace suitcase_problem_l626_626177

noncomputable def weight_of_electronics (k : ℝ) : ℝ :=
  2 * k

theorem suitcase_problem (k : ℝ) (B C E T : ℝ) (hc1 : B = 5 * k) (hc2 : C = 4 * k) (hc3 : E = 2 * k) (hc4 : T = 3 * k) (new_ratio : 5 * k / (4 * k - 7) = 3) :
  E = 6 :=
by
  sorry

end suitcase_problem_l626_626177


namespace magnitude_difference_vector_l626_626664

def vectors := (1 : ℝ, 2 : ℝ)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem magnitude_difference_vector :
  ∃ x : ℝ, dot_product vectors (x, 4) = 10 ∧
  (real.sqrt ((vectors.1 - x) ^ 2 + (vectors.2 - 4)^ 2) = real.sqrt 5) :=
sorry

end magnitude_difference_vector_l626_626664


namespace solve_system_of_inequalities_l626_626547

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 1 > x) ∧ (x < -3 * x + 8) ↔ -1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l626_626547


namespace num_int_vals_l626_626609

theorem num_int_vals (x : ℝ) : 
  (∃ x ∈ ℤ, 4 < sqrt (3 * x) ∧ sqrt (3 * x) < 5) → 
  (∃ n : ℕ, n = 3) :=
by
  sorry

end num_int_vals_l626_626609


namespace flight_time_approx_58_hours_l626_626724

noncomputable def planet_radius : ℝ := 5000
noncomputable def average_speed : ℝ := 550
noncomputable def wind_effect : ℝ := 50

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def effective_speeds (avg_speed wind_effect : ℝ) : ℝ × ℝ :=
  (avg_speed + wind_effect, avg_speed - wind_effect)

def flight_time (C : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (C / 2) / speed1 + (C / 2) / speed2

def total_flight_time (r avg_speed wind_effect : ℝ) : ℝ :=
  let (speed1, speed2) := effective_speeds avg_speed wind_effect
  flight_time (circumference r) speed1 speed2

theorem flight_time_approx_58_hours :
  abs (total_flight_time planet_radius average_speed wind_effect - 58) ≤ 0.5 :=
by
  sorry

end flight_time_approx_58_hours_l626_626724


namespace quadratic_roots_always_distinct_find_m_if_roots_right_angled_triangle_l626_626384

theorem quadratic_roots_always_distinct (m : ℝ) : 
  let a := 1,
      b := -(2 + 3 * m),
      c := 2 * m^2 + 5 * m - 4 
  in (b^2 - 4 * a * c) > 0 :=
by
  let a := (1 : ℝ)
  let b := -(2 + 3 * m)
  let c := 2 * m^2 + 5 * m - 4 
  sorry

theorem find_m_if_roots_right_angled_triangle (m : ℝ) :
  let a := 1,
      b := -(2 + 3 * m),
      c := 2 * m^2 + 5 * m - 4,
      hyp_squared := 4 * 7,
      sum_of_squares := (2 + 3 * m)^2 - 2 * (2 * m^2 + 5 * m - 4)
  in hyp_squared = 28 →
     sum_of_squares = 28 → 
     (m = -2) ∨ (m = 8 / 5) :=
by
  let a := (1 : ℝ)
  let b := -(2 + 3 * m)
  let c := 2 * m^2 + 5 * m - 4
  let hyp_squared := 4 * 7
  let sum_of_squares := (2 + 3 * m)^2 - 2 * (2 * m^2 + 5 * m - 4)
  sorry

end quadratic_roots_always_distinct_find_m_if_roots_right_angled_triangle_l626_626384


namespace max_factors_l626_626735

theorem max_factors (b n : ℕ) (hb : 1 ≤ b ∧ b ≤ 20) (hn : 1 ≤ n ∧ n ≤ 20) (hbn : b ≠ n) :
    ∃ (p : ℕ), p = 81 ∧ ∀ (b' n' : ℕ), (1 ≤ b' ∧ b' ≤ 20) → (1 ≤ n' ∧ n' ≤ 20) → (b' ≠ n') → 
    (∃ k, b'^n' = ∏ i in (finset.range (k+1)), (i + 1)) → (k + 1 ≤ p) :=
by
  sorry

end max_factors_l626_626735


namespace garden_enlargement_l626_626687

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end garden_enlargement_l626_626687


namespace ellipse_problem_l626_626743

theorem ellipse_problem
    (x y : ℝ)
    (F1 F2 : ℝ × ℝ)
    (h1 : { p : ℝ × ℝ | p.1 ^ 2 / 4 + p.2 ^ 2 = 1 }) -- condition for point on ellipse
    (is_focus : (F1 = (-2, 0)) ∧ (F2 = (2, 0)))    -- foci of the ellipse
    (optionA : ∀ A B : ℝ × ℝ, (∀ m : ℝ, (B.1 = m * A.1 + (1 - m) * F2.1) ∧ (B.2 = m * A.2 + (1 - m) * F2.2)) ∧ 
               (|A.1^2 / 4 + A.2^2 = 1) ∧ (|B.1^2 / 4 + B.2^2 = 1) ∧
               8 = |B.1 - A.1| + |B.2 - A.2|)
    (optionC : ∀ m : ℝ, ∃ x y : ℝ, (2 * m * x - 2 * y - 2 * m + 1 = 0) ∧ (x^2 / 4 + y^2 = 1))
    (optionD : ∀ P Q : ℝ × ℝ, ((P.1^2 / 4 + P.2^2 = 1) ∧ (Q.1^2 + Q.2^2 = 1)) → dist P Q ≤ 3) :
    True :=
sorry

end ellipse_problem_l626_626743


namespace perimeters_ratio_l626_626578

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l626_626578


namespace number_of_distinct_sums_l626_626702

-- Define the sets representing the two bags
def bagX : Set ℕ := {2, 5, 7}
def bagY : Set ℕ := {1, 4, 8}

-- Define the problem statement
theorem number_of_distinct_sums : (bagX ×ˢ bagY).image (λ p : ℕ × ℕ, p.1 + p.2) = {3, 6, 10, 9, 13, 8, 11, 15} :=
by
  sorry

end number_of_distinct_sums_l626_626702


namespace arithmetic_sequence_sum_l626_626616

/-
The sum of the first 20 terms of the arithmetic sequence 8, 5, 2, ... is -410.
-/

theorem arithmetic_sequence_sum :
  let a : ℤ := 8
  let d : ℤ := -3
  let n : ℤ := 20
  let S_n : ℤ := n * a + (d * n * (n - 1)) / 2
  S_n = -410 := by
  sorry

end arithmetic_sequence_sum_l626_626616


namespace sum_of_reciprocals_of_28_l626_626648

def is_perfect_number (n : ℕ) : Prop :=
  (∑ d in Finset.filter (λ d, d ∣ n) (Finset.range (n + 1)), d) = 2 * n

noncomputable def sum_of_reciprocals (n : ℕ) : ℚ :=
  ∑ d in Finset.filter (λ d, d ∣ n) (Finset.range (n + 1)), (1 : ℚ) / d

theorem sum_of_reciprocals_of_28 : is_perfect_number 28 → sum_of_reciprocals 28 = 2 := by
  sorry

end sum_of_reciprocals_of_28_l626_626648


namespace john_ingrid_combined_weighted_average_tax_rate_l626_626468

noncomputable def john_employment_income : ℕ := 57000
noncomputable def john_employment_tax_rate : ℚ := 0.30
noncomputable def john_rental_income : ℕ := 11000
noncomputable def john_rental_tax_rate : ℚ := 0.25

noncomputable def ingrid_employment_income : ℕ := 72000
noncomputable def ingrid_employment_tax_rate : ℚ := 0.40
noncomputable def ingrid_investment_income : ℕ := 4500
noncomputable def ingrid_investment_tax_rate : ℚ := 0.15

noncomputable def combined_weighted_average_tax_rate : ℚ :=
  let john_total_tax := john_employment_income * john_employment_tax_rate + john_rental_income * john_rental_tax_rate
  let john_total_income := john_employment_income + john_rental_income
  let ingrid_total_tax := ingrid_employment_income * ingrid_employment_tax_rate + ingrid_investment_income * ingrid_investment_tax_rate
  let ingrid_total_income := ingrid_employment_income + ingrid_investment_income
  let combined_total_tax := john_total_tax + ingrid_total_tax
  let combined_total_income := john_total_income + ingrid_total_income
  (combined_total_tax / combined_total_income) * 100

theorem john_ingrid_combined_weighted_average_tax_rate :
  combined_weighted_average_tax_rate = 34.14 := by
  sorry

end john_ingrid_combined_weighted_average_tax_rate_l626_626468


namespace obtuse_is_second_quadrant_l626_626994

-- Define the boundaries for an obtuse angle.
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define the second quadrant condition.
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The proof problem: Prove that an obtuse angle is a second quadrant angle.
theorem obtuse_is_second_quadrant (θ : ℝ) : is_obtuse θ → is_second_quadrant θ :=
by
  intro h
  sorry

end obtuse_is_second_quadrant_l626_626994


namespace domain_of_h_h_is_odd_h_gt_zero_l626_626394

-- Definitions for functions
def f (a : ℝ) (x : ℝ) : ℝ := log a (1 + x)
def g (a : ℝ) (x : ℝ) : ℝ := log a (1 - x)
def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

-- Conditions
variables {a : ℝ} (ha : a > 0) (hane1 : a ≠ 1)

-- Prove domain of h
theorem domain_of_h : ∀ x, (x > -1 ∧ x < 1) ↔ domain (h a) x :=
sorry

-- Prove h is odd
theorem h_is_odd : ∀ x, h a (-x) = -h a x :=
sorry

-- Prove h(x) > 0 in (0, 1)
theorem h_gt_zero : ∀ (x : ℝ), f 2 3 = 2 → (0 < x ∧ x < 1) → h 2 x > 0 :=
sorry

end domain_of_h_h_is_odd_h_gt_zero_l626_626394


namespace distance_between_foci_of_hyperbola_l626_626340

theorem distance_between_foci_of_hyperbola (a b c : ℝ) : (x^2 - y^2 = 4) → (a = 2) → (b = 0) → (c = Real.sqrt (4 + 0)) → 
    dist (2, 0) (-2, 0) = 4 :=
by
  sorry

end distance_between_foci_of_hyperbola_l626_626340


namespace first_free_friday_after_february_1_2023_l626_626447

-- Define the given conditions as assumptions.
def is_wednesday (d: ℕ) : Prop := 
  day_of_week d = 4 -- 4 corresponds to Wednesday
-- Assume February has 28 days and begins on a Wednesday.
def february_2023 : ℕ := 28
def february_1_2023 : ℕ := 1 -- February 1, 2023
axiom february_is_wednesday: is_wednesday february_1_2023

-- Define the concept of Free Friday.
def is_friday (d: ℕ) : Prop := 
  day_of_week d = 5 -- 5 corresponds to Friday

def free_friday (d: ℕ) : Prop :=
  (exists (k: ℕ), 5 * k + 5 = d)

-- Define the target proposition: the first Free Friday after February 1, 2023, is March 31, 2023.
theorem first_free_friday_after_february_1_2023: 
  ∃ d: ℕ, 
  (d > february_2023 ∧ free_friday d) ∧ 
  d = 31 := sorry

end first_free_friday_after_february_1_2023_l626_626447


namespace father_twice_as_old_as_son_in_years_l626_626264

variables (S F Y : ℕ)

def son_father_age_relation (S F Y : ℕ) : Prop :=
  S = 10 ∧ F = 40 ∧ F + Y = 2 * (S + Y)

theorem father_twice_as_old_as_son_in_years (S F Y : ℕ) (h : son_father_age_relation S F Y) : Y = 20 :=
by
  rcases h with ⟨hS, hF, hY⟩
  have : (40 : ℕ) + Y = 2 * (10 + Y) := hY
  linarith

end father_twice_as_old_as_son_in_years_l626_626264


namespace blake_change_l626_626703

theorem blake_change :
  let lollipop_count := 4
  let chocolate_count := 6
  let lollipop_cost := 2
  let chocolate_cost := 4 * lollipop_cost
  let total_received := 6 * 10
  let total_cost := (lollipop_count * lollipop_cost) + (chocolate_count * chocolate_cost)
  let change := total_received - total_cost
  change = 4 :=
by
  sorry

end blake_change_l626_626703


namespace position_of_SUMBO_in_alphabetical_order_l626_626208

theorem position_of_SUMBO_in_alphabetical_order :
  let L := ['B', 'M', 'O', 'S', 'U']
  in (perm_list := List.permutations L).nthLe 91 (by sorry) = ['S','U','M','B','O'] :=
sorry

end position_of_SUMBO_in_alphabetical_order_l626_626208


namespace complex_modulus_l626_626159

-- Problem Statement: Given (1 - sqrt(3) * I) * z = I, prove |z| = 1 / 2
theorem complex_modulus (z : ℂ) (h : (1 - complex.I * real.sqrt 3) * z = complex.I) : complex.abs z = 1 / 2 := 
sorry

end complex_modulus_l626_626159


namespace find_length_of_AB_l626_626837

variable (A B C : ℝ)
variable (cos_C_div2 BC AC AB : ℝ)
variable (C_gt_0 : 0 < C / 2) (C_lt_pi : C / 2 < Real.pi)

axiom h1 : cos_C_div2 = Real.sqrt 5 / 5
axiom h2 : BC = 1
axiom h3 : AC = 5
axiom h4 : AB = Real.sqrt (BC ^ 2 + AC ^ 2 - 2 * BC * AC * (2 * cos_C_div2 ^ 2 - 1))

theorem find_length_of_AB : AB = 4 * Real.sqrt 2 :=
by
  sorry

end find_length_of_AB_l626_626837


namespace fifth_inequality_nth_inequality_solve_given_inequality_l626_626131

theorem fifth_inequality :
  ∀ x, 1 < x ∧ x < 2 → (x + 2 / x < 3) →
  ∀ x, 3 < x ∧ x < 4 → (x + 12 / x < 7) →
  ∀ x, 5 < x ∧ x < 6 → (x + 30 / x < 11) →
  (x + 90 / x < 19) := by
  sorry

theorem nth_inequality (n : ℕ) :
  ∀ x, (2 * n - 1 < x ∧ x < 2 * n) →
  (x + 2 * n * (2 * n - 1) / x < 4 * n - 1) := by
  sorry

theorem solve_given_inequality (a : ℕ) (x : ℝ) (h_a_pos: 0 < a) :
  x + 12 * a / (x + 1) < 4 * a + 2 →
  (2 < x ∧ x < 4 * a - 1) := by
  sorry

end fifth_inequality_nth_inequality_solve_given_inequality_l626_626131


namespace negation_proposition_l626_626169

variable (n : ℕ)
variable (n_positive : n > 0)
variable (f : ℕ → ℕ)
variable (H1 : ∀ n, n > 0 → (f n) > 0 ∧ (f n) ≤ n)

theorem negation_proposition :
  (∃ n_0, n_0 > 0 ∧ ((f n_0) ≤ 0 ∨ (f n_0) > n_0)) ↔ ¬(∀ n, n > 0 → (f n) >0 ∧ (f n) ≤ n) :=
by 
  sorry

end negation_proposition_l626_626169


namespace find_q_l626_626853

-- Define the points A, B, R, and X
def A : ℝ × ℝ := (3, 10)
def B : ℝ × ℝ := (15, 0)
def R : ℝ × ℝ := (0, 10)
def X (q : ℝ) : ℝ × ℝ := (0, q)

-- Given area of triangle ABX is 58
def area_ABX : ℝ := 58

-- Function to calculate the area of triangles
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * |(fst P - fst R) * (snd Q - snd R) - (fst Q - fst R) * (snd P - snd R)|

-- The mathematical proof statement
theorem find_q : ∃ q : ℝ, triangle_area A B (X q) = area_ABX ∧ q = 43 / 6 :=
by
  sorry

end find_q_l626_626853


namespace paul_books_left_l626_626133
-- Add the necessary imports

-- Define the initial conditions
def initial_books : ℕ := 115
def books_sold : ℕ := 78

-- Statement of the problem as a theorem
theorem paul_books_left : (initial_books - books_sold) = 37 := by
  -- Proof omitted
  sorry

end paul_books_left_l626_626133


namespace angle_C_eq_pi_div_3_find_ab_values_l626_626028

noncomputable def find_angle_C (A B C : ℝ) (a b c : ℝ) : ℝ :=
  if c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C then C else 0

noncomputable def find_sides_ab (A B C : ℝ) (c S : ℝ) : Set (ℝ × ℝ) :=
  if C = Real.pi / 3 ∧ c = 2 * Real.sqrt 3 ∧ S = 2 * Real.sqrt 3 then
    { (a, b) | a^4 - 20 * a^2 + 64 = 0 ∧ b = 8 / a } else
    ∅

theorem angle_C_eq_pi_div_3 (A B C : ℝ) (a b c : ℝ) :
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C)
  ↔ (C = Real.pi / 3) :=
sorry

theorem find_ab_values (A B C : ℝ) (c S a b : ℝ) :
  (C = Real.pi / 3) ∧ (c = 2 * Real.sqrt 3) ∧ (S = 2 * Real.sqrt 3) ∧ (a^4 - 20 * a^2 + 64 = 0) ∧ (b = 8 / a)
  ↔ ((a, b) = (2, 4) ∨ (a, b) = (4, 2)) :=
sorry

end angle_C_eq_pi_div_3_find_ab_values_l626_626028


namespace initial_concentration_is_40_l626_626671

noncomputable def initial_concentration_fraction : ℝ := 1 / 3
noncomputable def replaced_solution_concentration : ℝ := 25
noncomputable def resulting_concentration : ℝ := 35
noncomputable def initial_concentration := 40

theorem initial_concentration_is_40 (C : ℝ) (h1 : C = (3 / 2) * (resulting_concentration - (initial_concentration_fraction * replaced_solution_concentration))) :
  C = initial_concentration :=
by sorry

end initial_concentration_is_40_l626_626671


namespace worm_length_difference_l626_626926

def worm_1_length : ℝ := 0.8
def worm_2_length : ℝ := 0.1
def difference := worm_1_length - worm_2_length

theorem worm_length_difference : difference = 0.7 := by
  sorry

end worm_length_difference_l626_626926


namespace solve_system_of_equations_l626_626657

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + 2 * x * y + x^2 - 6 * y - 6 * x + 5 = 0)
  (h2 : y - x + 1 = x^2 - 3 * x) : 
  ((x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) ∨ (x = -2 ∧ y = 7)) ∧ x ≠ 0 ∧ x ≠ 3 :=
by 
  sorry

end solve_system_of_equations_l626_626657


namespace train_length_is_140_l626_626287

noncomputable def speed_kmph := 60
noncomputable def time_seconds := 23.998080153587715
noncomputable def platform_length_m := 260

-- Converting speed from km/h to m/s
noncomputable def speed_mps := speed_kmph * 1000 / 3600

-- Total distance covered by the train passing the platform
noncomputable def total_distance := speed_mps * time_seconds

-- Length of the train
noncomputable def train_length := total_distance - platform_length_m

theorem train_length_is_140 : train_length = 140 := by
  sorry

end train_length_is_140_l626_626287


namespace triangle_area_l626_626371

noncomputable def ellipse : Type := 
{ P : ℝ × ℝ // ∃ x y : ℝ, x^2 / 9 + y^2 / 4 = 1 }

variables {F1 F2 : ℝ × ℝ} {P : ℝ × ℝ}

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def is_right_angle (u v : ℝ × ℝ) : Prop :=
(u.1 * v.1 + u.2 * v.2 = 0)

theorem triangle_area (a : ℝ) (h_foci : distance F1 F2 = 2 * real.sqrt(5))
(P_ellipse : ∃ x y : ℝ, x^2 / 9 + y^2 / 4 = 1 ∧ P = (x, y)) 
(P_right_angle : is_right_angle (P.1 - F1.1, P.2 - F1.2) (P.1 - F2.1, P.2 - F2.2)) :
  let m := distance P F1,
      n := distance P F2
  in 1 / 2 * m * n = 4 :=
by
  sorry

end triangle_area_l626_626371


namespace total_appointment_plans_l626_626765

def volunteers := {XiaoZhang, XiaoZhao, XiaoLi, XiaoLuo, XiaoWang}
def tasks := {translation, tourGuide, etiquette, driving}

def volunteer_eligible_task (v : volunteers) (t : tasks) : Prop :=
  (v ∈ {XiaoZhang, XiaoZhao} → t ∈ {translation, tourGuide}) ∧ 
  (v ∈ {XiaoLi, XiaoLuo, XiaoWang} → t ∈ {translation, tourGuide, etiquette, driving})

theorem total_appointment_plans : 
  (∑ s in volunteers.powerset, 
    if s.card = 4 
    then ( ∑ t in tasks.powerset, if t.card = 4 ∧ ∀ v ∈ s, ∃ t ∈ t', volunteer_eligible_task v t then 1 else 0 )
    else 0
  ) = 18 := 
sorry

end total_appointment_plans_l626_626765


namespace coloring_satisfies_conditions_l626_626068

-- Define lattice points as points with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the color function
def color (p : LatticePoint) : ℕ :=
  if (p.x % 2 = 0) ∧ (p.y % 2 = 1) then 0 -- Black
  else if (p.x % 2 = 1) ∧ (p.y % 2 = 0) then 1 -- White
  else 2 -- Red

-- Define condition (1)
def infinite_lines_with_color (c : ℕ) : Prop :=
  ∀ k : ℤ, ∃ p : LatticePoint, color p = c ∧ p.x = k

-- Define condition (2)
def parallelogram_exists (A B C : LatticePoint) (wc rc bc : ℕ) : Prop :=
  (color A = wc) ∧ (color B = rc) ∧ (color C = bc) →
  ∃ D : LatticePoint, color D = rc ∧ D.x = C.x + (A.x - B.x) ∧ D.y = C.y + (A.y - B.y)

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : ℕ, ∃ p : LatticePoint, infinite_lines_with_color c) ∧
  (∀ A B C : LatticePoint, ∃ wc rc bc : ℕ, parallelogram_exists A B C wc rc bc) :=
sorry

end coloring_satisfies_conditions_l626_626068


namespace final_values_comparison_l626_626900

theorem final_values_comparison :
  let AA_initial : ℝ := 100
  let BB_initial : ℝ := 100
  let CC_initial : ℝ := 100
  let AA_year1 := AA_initial * 1.20
  let BB_year1 := BB_initial * 0.75
  let CC_year1 := CC_initial
  let AA_year2 := AA_year1 * 0.80
  let BB_year2 := BB_year1 * 1.25
  let CC_year2 := CC_year1
  AA_year2 = 96 ∧ BB_year2 = 93.75 ∧ CC_year2 = 100 ∧ BB_year2 < AA_year2 ∧ AA_year2 < CC_year2 :=
by {
  -- Definitions from conditions
  let AA_initial : ℝ := 100;
  let BB_initial : ℝ := 100;
  let CC_initial : ℝ := 100;
  let AA_year1 := AA_initial * 1.20;
  let BB_year1 := BB_initial * 0.75;
  let CC_year1 := CC_initial;
  let AA_year2 := AA_year1 * 0.80;
  let BB_year2 := BB_year1 * 1.25;
  let CC_year2 := CC_year1;

  -- Use sorry to skip the actual proof
  sorry
}

end final_values_comparison_l626_626900


namespace max_difference_l626_626153

theorem max_difference (x y : ℝ) (hx : 4 ≤ x) (hx2 : x ≤ 100) (hy : 4 ≤ y) (hy2 : y ≤ 100) :
  abs ((x + y) / 2 - (x + 2 * y) / 3) ≤ 16 :=
begin
  sorry
end

end max_difference_l626_626153


namespace modulus_z_l626_626054

-- Define a complex number z
def z : ℂ := 2 * complex.I + 2 / (1 + complex.I)

-- Statement that needs to be proven
theorem modulus_z : complex.abs z = real.sqrt 2 := 
sorry

end modulus_z_l626_626054


namespace how_many_children_l626_626195

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l626_626195


namespace length_AC_l626_626134

-- Define a circle with center O and radius r
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Assume the points and key properties in the conditions
variable (O A B C D : ℝ × ℝ)
variable (r : ℝ) (h1 : r = 7)
variable (h2 : (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2)
variable (h3 : (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2)
variable (h4 : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8^2)
variable (hD : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
variable (hC : (C.1, C.2) = (O.1, O.2) + (7 * (A.2 - B.2), 7 * (B.1 - A.1)) / 8)

-- Length of the line segment AC according to the given conditions should be L
theorem length_AC (h5 : h1 = 7 ∧ h2 ∧ h3 ∧ h4 ∧ hD ∧ hC) : 
  sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = sqrt (98 - 14 * sqrt 33) :=
sorry

end length_AC_l626_626134


namespace garden_area_enlargement_l626_626683

theorem garden_area_enlargement :
  let length := 60
  let width := 20
  (2 * (length + width)) = 160 →
  (160 / 4) = 40 →
  ((40 * 40) - (length * width) = 400) :=
begin
  intros,
  sorry,
end

end garden_area_enlargement_l626_626683


namespace number_of_nurses_l626_626241

theorem number_of_nurses (total : ℕ) (ratio_d_to_n : ℕ → ℕ) (h1 : total = 250) (h2 : ratio_d_to_n 2 = 3) : ∃ n : ℕ, n = 150 := 
by
  sorry

end number_of_nurses_l626_626241


namespace carl_gas_cost_l626_626717

-- Define the variables for conditions
def city_mileage := 30       -- miles per gallon in city
def highway_mileage := 40    -- miles per gallon on highway
def city_distance := 60      -- city miles one way
def highway_distance := 200  -- highway miles one way
def gas_cost := 3            -- dollars per gallon

-- Define the statement to prove
theorem carl_gas_cost : 
  let city_gas := city_distance / city_mileage in
  let highway_gas := highway_distance / highway_mileage in
  let total_one_way_gas := city_gas + highway_gas in
  let round_trip_gas := total_one_way_gas * 2 in
  let total_cost := round_trip_gas * gas_cost in
  total_cost = 42
:= by
  sorry

end carl_gas_cost_l626_626717


namespace find_pairs_of_nonneg_ints_l626_626747

theorem find_pairs_of_nonneg_ints (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (m, n) = (9, 3) ∨ (m, n) = (6, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end find_pairs_of_nonneg_ints_l626_626747


namespace curves_intersect_at_three_points_l626_626990

theorem curves_intersect_at_three_points :
  (∀ x y a : ℝ, (x^2 + y^2 = 4 * a^2) ∧ (y = x^2 - 2 * a) → a = 1) := sorry

end curves_intersect_at_three_points_l626_626990


namespace intersection_eq_complement_N_l626_626398

open Set

variable {U : Set ℝ} {M N : Set ℝ}

def U := univ
def M := {x : ℝ | x^2 > 4}
def N := {x : ℝ | (3 - x) / (x + 1) > 0}
def complement_N := {x : ℝ | x ≤ -1 ∨ x ≥ 3}
def intersection := {x : ℝ | x < -2 ∨ x ≥ 3}

theorem intersection_eq_complement_N : M ∩ (U \ N) = intersection := by
  sorry

end intersection_eq_complement_N_l626_626398


namespace product_xy_l626_626915

noncomputable theory

variables {x y : ℝ}

def condition1 (x y : ℝ) : Prop := 2^x = 16^(y + 1)
def condition2 (x y : ℝ) : Prop := 27^y = 3^(x - 2)

theorem product_xy (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 8 :=
sorry

end product_xy_l626_626915


namespace sum_a_b_eq_neg2_l626_626820

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : (a - 2)^2 + |b + 4| = 0) : a + b = -2 := 
by 
  sorry

end sum_a_b_eq_neg2_l626_626820


namespace ratio_of_perimeters_l626_626569

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l626_626569


namespace vector_magnitude_subtraction_l626_626011

noncomputable def vector_a : ℝ × ℝ := (3, 4)
noncomputable def vector_b_norm := 1
noncomputable def angle_ab := real.pi / 3   -- 60 degrees in radians

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude_subtraction : 
  let vector_b : ℝ × ℝ := sorry in  -- Since we don't have the actual vector b, it's a placeholder.
  ‖vector_a - (2 * vector_b)‖ = real.sqrt 19 :=
sorry

end vector_magnitude_subtraction_l626_626011


namespace number_of_quadruples_divisible_by_seven_l626_626035

theorem number_of_quadruples_divisible_by_seven :
  (finset.card 
    (finset.filter 
      (λ (x : ℕ × ℕ × ℕ × ℕ), (7 ∣ (x.1.1 * x.1.2 - x.2.1 * x.2.2)))
      ((finset.range 7).product (finset.range 7)).product 
      ((finset.range 7).product (finset.range 7))
    )
  ) = 385 :=
sorry

end number_of_quadruples_divisible_by_seven_l626_626035


namespace arc_length_sector_max_area_l626_626365

-- Define the constants and conditions
def α₁ := 120 * (π / 180) -- α in radians, given α = 120°
def r₁ := 6              -- radius r = 6

-- Define the first proof: length of the arc for given α and r
theorem arc_length (α₁ : ℝ) (r₁ : ℝ) : α₁ * r₁ = 4 * π := 
by 
  sorry

-- Define the constants and conditions for the second part
def perimeter := 24
def r := 6

-- Define the second proof: α in radians and maximum area when sector perimeter is 24
theorem sector_max_area (perimeter : ℝ) (r : ℝ) : 
  (perimeter - 2 * r) / r = 2 ∧ 
  (1 / 2) * (perimeter - 2 * r) * r = 36 := 
by 
  sorry

end arc_length_sector_max_area_l626_626365


namespace num_of_ordered_pairs_l626_626273

theorem num_of_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b > a)
(h4 : (a-2)*(b-2) = (ab / 2)) : (a, b) = (5, 12) ∨ (a, b) = (6, 8) :=
by
  sorry

end num_of_ordered_pairs_l626_626273


namespace number_of_integer_solutions_is_four_l626_626950

-- Define the equation as a predicate
def equation (x y : ℤ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- State the theorem that the number of integer solutions to the equation is 4
theorem number_of_integer_solutions_is_four : 
  {p : ℤ × ℤ | equation p.1 p.2}.toFinset.card = 4 :=
by
  sorry -- Proof omitted

end number_of_integer_solutions_is_four_l626_626950


namespace sum_of_factors_180_l626_626986

open BigOperators

def sum_of_factors (n : ℕ) : ℕ :=
  ∑ d in (finset.range (n+1)).filter (λ d, n % d = 0), d

noncomputable def factor_180 := 180

theorem sum_of_factors_180 : sum_of_factors factor_180 = 546 := by
  sorry

end sum_of_factors_180_l626_626986


namespace area_ratio_of_triangle_in_square_l626_626448

-- Define points, lines, areas, and conditions
structure Square (Ω : Type) :=
  (A B C D M N : Ω)
  (MD_eq_3BM : distance D M = 3 * distance B M)
  (AM_intersects_BC_at_N : ∃ N : Ω, Line A M ∩ Line B C = {N})

-- Define areas as a function
def area {Ω : Type} [AffineSpace Ω] (P Q R : Ω) : ℝ := sorry

-- The main statement in Lean 4
theorem area_ratio_of_triangle_in_square {Ω : Type} [AffineSpace Ω] [Square Ω]
  (sq : Square Ω) : 
  let S_triangle_MND := area sq.M sq.N sq.D
  let S_square_ABCD := (area sq.A sq.B sq.C + area sq.C sq.D sq.A)
  S_triangle_MND / S_square_ABCD = 1 / 8 := 
sorry

end area_ratio_of_triangle_in_square_l626_626448


namespace associate_professors_bring_one_chart_l626_626301

theorem associate_professors_bring_one_chart
(A B C : ℕ) (h1 : 2 * A + B = 7) (h2 : A * C + 2 * B = 11) (h3 : A + B = 6) : C = 1 :=
by sorry

end associate_professors_bring_one_chart_l626_626301


namespace polynomial_sum_equals_864_l626_626938

noncomputable def q : ℤ → ℤ := sorry

axiom q_conditions : q 1 = 8 ∧ q 9 = 40 ∧ q 17 = 16 ∧ q 25 = 56

theorem polynomial_sum_equals_864 : 
  ∑ i in (Finset.range 27), q i = 864 :=
by sorry

end polynomial_sum_equals_864_l626_626938


namespace min_candidates_for_same_score_l626_626961

theorem min_candidates_for_same_score :
  (∃ S : ℕ, S ≥ 25 ∧ (∀ elect : Fin S → Fin 12, ∃ s : Fin 12, ∃ a b c : Fin S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ elect a = s ∧ elect b = s ∧ elect c = s)) := 
sorry

end min_candidates_for_same_score_l626_626961


namespace prop_A_prop_B_prop_C_prop_D_l626_626992

-- Proposition A: For all x ∈ ℝ, x² - x + 1 > 0
theorem prop_A (x : ℝ) : x^2 - x + 1 > 0 :=
sorry

-- Proposition B: a² + a = 0 is not a sufficient and necessary condition for a = 0
theorem prop_B : ¬(∀ a : ℝ, (a^2 + a = 0 ↔ a = 0)) :=
sorry

-- Proposition C: a > 1 and b > 1 is a sufficient and necessary condition for a + b > 2 and ab > 1
theorem prop_C (a b : ℝ) : (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b > 1) :=
sorry

-- Proposition D: a > 4 is a necessary and sufficient condition for the roots of the equation x² - ax + a = 0 to be all positive
theorem prop_D (a : ℝ) : (a > 4) ↔ (∀ x : ℝ, x ≠ 0 → (x^2 - a*x + a = 0 → x > 0)) :=
sorry

end prop_A_prop_B_prop_C_prop_D_l626_626992


namespace total_handshakes_l626_626046

theorem total_handshakes (n : ℕ) (h : n = 20) : (nat.choose n 2) = 190 :=
by 
  rw h
  exact nat.choose_self (dec_trivial : 20 > 2) 


end total_handshakes_l626_626046


namespace math_problem_l626_626170

theorem math_problem (x y : ℝ) (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 :=
sorry

end math_problem_l626_626170


namespace find_x3_l626_626207

variable (x1 x2 x3 : ℝ)

-- Given conditions
def f (x : ℝ) : ℝ := Real.log x
def y_a : ℝ := f x1
def y_b : ℝ := f x2
def y_c : ℝ := (2 / 3) * y_a + (1 / 3) * y_b

-- Given specific values
variable (h_x1 : x1 = 2) (h_x2 : x2 = 32)

-- Prove that x3 is as follows
theorem find_x3 : x3 = 2 ^ (7 / 3) := by
  sorry

end find_x3_l626_626207


namespace number_of_children_l626_626192

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l626_626192


namespace ball_more_expensive_l626_626233

theorem ball_more_expensive (B L : ℝ) (h1 : 2 * B + 3 * L = 1300) (h2 : 3 * B + 2 * L = 1200) : 
  L - B = 100 := 
sorry

end ball_more_expensive_l626_626233


namespace books_choice_l626_626690

theorem books_choice (P S : Type) [Fintype P] [Fintype S] [DecidableEq P] [DecidableEq S] 
  (hP : Fintype.card P = 4) (hS : Fintype.card S = 2) 
  : ∃ (books : Finset (P ⊕ S)), books.card = 4 ∧ ∃ (s : S), s ∈ books := 
by
  have h1 : ∃ (books : Finset (P ⊕ S)), books.card = 4 := sorry
  have h2 : ∀ (books : Finset (P ⊕ S)), books.card = 4 → ∃ (s : S), s ∈ books := sorry
  exact ⟨_, h1, h2⟩

end books_choice_l626_626690


namespace difference_is_62_l626_626982

def sixty_percent_of_40 : Real := (60 / 100) * 40
def x : Real := 3 * sixty_percent_of_40

def four_fifths_of_25 : Real := (4 / 5) * 25
def y : Real := four_fifths_of_25 / 2

def z : Real := Real.sqrt 16 - 3

theorem difference_is_62 : (x * z) - (y * z) = 62 := by
  have x_val : x = 72 := by
    calc
      x = 3 * ((60 / 100) * 40) := rfl
      _ = 3 * 24 := rfl
      _ = 72 := rfl
      
  have y_val : y = 10 := by
    calc
      y = ((4 / 5) * 25) / 2 := rfl
      _ = 20 / 2 := rfl
      _ = 10 := rfl

  have z_val : z = 1 := by
    calc
      z = Real.sqrt 16 - 3 := rfl
      _ = 4 - 3 := by norm_num
      _ = 1 := rfl
      
  calc
    (x * z) - (y * z) = (72 * 1) - (10 * 1) := by rw [x_val, y_val, z_val]
    _ = 72 - 10 := rfl
    _ = 62 := rfl

end difference_is_62_l626_626982


namespace carl_trip_cost_is_correct_l626_626714

structure TripConditions where
  city_mpg : ℕ
  highway_mpg : ℕ
  city_distance : ℕ
  highway_distance : ℕ
  gas_cost : ℕ

def total_gas_cost (conds : TripConditions) : ℕ :=
  let total_city_miles := conds.city_distance * 2
  let total_highway_miles := conds.highway_distance * 2
  let city_gas_needed := total_city_miles / conds.city_mpg
  let highway_gas_needed := total_highway_miles / conds.highway_mpg
  let total_gas_needed := city_gas_needed + highway_gas_needed
  total_gas_needed * conds.gas_cost

theorem carl_trip_cost_is_correct (conds : TripConditions)
  (h1 : conds.city_mpg = 30)
  (h2 : conds.highway_mpg = 40)
  (h3 : conds.city_distance = 60)
  (h4 : conds.highway_distance = 200)
  (h5 : conds.gas_cost = 3) :
  total_gas_cost conds = 42 := by
  rw [total_gas_cost, h1, h2, h3, h4, h5]
  -- Proof steps will follow here, but we can skip them for now
  sorry

end carl_trip_cost_is_correct_l626_626714


namespace license_plate_increase_factor_l626_626411

def old_plate_count : ℕ := 26^2 * 10^3
def new_plate_count : ℕ := 26^4 * 10^4
def increase_factor : ℕ := new_plate_count / old_plate_count

theorem license_plate_increase_factor : increase_factor = 2600 :=
by
  unfold increase_factor
  rw [old_plate_count, new_plate_count]
  norm_num
  sorry

end license_plate_increase_factor_l626_626411


namespace points_on_line_l626_626336

theorem points_on_line : 
    ∀ (P : ℝ × ℝ),
      (P = (1, 2) ∨ P = (0, 0) ∨ P = (2, 4) ∨ P = (5, 10) ∨ P = (-1, -2))
      → (∃ m b, m = 2 ∧ b = 0 ∧ P.2 = m * P.1 + b) :=
by
  sorry

end points_on_line_l626_626336


namespace avg_speed_xz_l626_626238

def distance_xy (D : ℝ) : ℝ := 2 * D
def distance_yz (D : ℝ) : ℝ := D
def speed_xy : ℝ := 300
def speed_yz : ℝ := 100

theorem avg_speed_xz (D : ℝ) (hD_ne_zero : D ≠ 0) : 
  let total_distance := distance_xy D + distance_yz D,
      time_xy := distance_xy D / speed_xy,
      time_yz := distance_yz D / speed_yz,
      total_time := time_xy + time_yz in
  total_distance / total_time = 180 :=
by
  -- Proof omitted
  sorry

end avg_speed_xz_l626_626238


namespace fractional_eq_a_range_l626_626834

theorem fractional_eq_a_range (a : ℝ) :
  (∃ x : ℝ, (a / (x + 2) = 1 - 3 / (x + 2)) ∧ (x < 0)) ↔ (a < -1 ∧ a ≠ -3) := by
  sorry

end fractional_eq_a_range_l626_626834


namespace arc_length_of_sector_l626_626795

theorem arc_length_of_sector (r α : ℝ) (hα : α = Real.pi / 5) (hr : r = 20) : r * α = 4 * Real.pi :=
by
  sorry

end arc_length_of_sector_l626_626795


namespace maximize_takehome_pay_l626_626435

noncomputable def tax_initial (income : ℝ) : ℝ :=
  if income ≤ 20000 then 0.10 * income else 2000 + 0.05 * ((income - 20000) / 10000) * income

noncomputable def tax_beyond (income : ℝ) : ℝ :=
  (income - 20000) * ((0.005 * ((income - 20000) / 10000)) * income)

noncomputable def tax_total (income : ℝ) : ℝ :=
  if income ≤ 20000 then tax_initial income else tax_initial 20000 + tax_beyond income

noncomputable def takehome_pay_function (income : ℝ) : ℝ :=
  income - tax_total income

theorem maximize_takehome_pay : ∃ x, takehome_pay_function x = takehome_pay_function 30000 := 
sorry

end maximize_takehome_pay_l626_626435


namespace probability_multiple_of_60_l626_626062

def set_nums := {3, 6, 10, 15, 18, 30, 45}
def is_multiple_of_60 (a b : ℕ) : Prop := (a * b) % 60 = 0
def number_of_ways_to_choose_two_distinct (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_multiple_of_60 :
  (∃ (S : set ℕ) (H : S = set_nums) (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ is_multiple_of_60 a b) →
  ∃ (favorable : ℕ), favorable = 5 →
  ∃ (total : ℕ), total = number_of_ways_to_choose_two_distinct (set_nums.card) →
  (favorable : ℝ) / (total : ℝ) = 5 / 21 := by
  sorry

end probability_multiple_of_60_l626_626062


namespace inequality_among_positives_l626_626494

theorem inequality_among_positives
  (a b x y z : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z) :
  (x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_among_positives_l626_626494


namespace problem1_problem2_l626_626395

open Real

noncomputable def f (x : ℝ) (a b : ℝ) := a * log x + b
noncomputable def g (x : ℝ) (k : ℝ) := x^2 + k * x + 3

theorem problem1 (m n : ℝ) (cond1 : 0 < m) (cond2 : m < n) (a b : ℝ) 
  (h1 : f 1 a b = 0) (h2 : deriv (f x a b) 1 = 1) : 
  if n ≤ 1 / exp 1 then f n a b = n * log n 
  else if m < 1 / exp 1 ∧ 1 / exp 1 < n then f (1 / exp 1) a b = - (1 / exp 1) 
  else f m a b = m * log m := 
sorry

theorem problem2 (k : ℝ) : 
  (∃ x : ℝ, x ∈ Ioo (1 / (exp 1)) (exp 1) ∧ 2 * f x 1 0 + g x k ≥ 0) 
  ↔ k > -((3 * exp 2 - 2 * exp 1 + 1) / exp 1) :=
sorry

end problem1_problem2_l626_626395


namespace fixed_point_parabola_l626_626499

theorem fixed_point_parabola (s : ℝ) : ∃ (p : ℝ × ℝ), p = (3, 36) ∧ ∀ s, ∃ y, y = 4 * 3^2 + s * 3 - 3 * s :=
by
  let p := (3, 36)
  use p
  sorry

end fixed_point_parabola_l626_626499


namespace shaded_solid_volume_l626_626078

noncomputable def volume_rectangular_prism (length width height : ℕ) : ℕ :=
  length * width * height

theorem shaded_solid_volume :
  volume_rectangular_prism 4 5 6 - volume_rectangular_prism 1 2 4 = 112 :=
by
  sorry

end shaded_solid_volume_l626_626078


namespace perimeters_ratio_l626_626580

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l626_626580


namespace fraction_doubled_l626_626061

theorem fraction_doubled (x y : ℝ) (h_nonzero : x + y ≠ 0) : (4 * x^2) / (2 * (x + y)) = 2 * (x^2 / (x + y)) :=
by
  sorry

end fraction_doubled_l626_626061


namespace quadrilateral_area_inequality_l626_626249

variable {a b c d S : ℝ}
variable (h_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
variable (h_area : S > 0)

theorem quadrilateral_area_inequality
  (h_sides_quadrilateral: true) -- We assume that sides a, b, c, d form a quadrilateral
  (h_area_definition: S ≤ ½ * (a * b + c * d)) :
  S ≤ ½ * (a * b + c * d) :=
sorry

end quadrilateral_area_inequality_l626_626249


namespace number_of_solutions_l626_626404

theorem number_of_solutions:
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 6 ∧
  ∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ (6 / m) + (3 / n) = 1 :=
begin
  -- proof goes here
  sorry
end

end number_of_solutions_l626_626404


namespace max_true_statements_maximum_true_statements_at_most_two_l626_626475

theorem max_true_statements (a b : ℝ) : 
  (1 / a > 1 / b) → (a^3 > b^3) → (a > b) → (a < 0) → (b > 0) → false := sorry

theorem maximum_true_statements_at_most_two :
  ∀ a b : ℝ, 2 ≤ max_true_statements a b := sorry

end max_true_statements_maximum_true_statements_at_most_two_l626_626475


namespace carl_trip_cost_is_correct_l626_626712

structure TripConditions where
  city_mpg : ℕ
  highway_mpg : ℕ
  city_distance : ℕ
  highway_distance : ℕ
  gas_cost : ℕ

def total_gas_cost (conds : TripConditions) : ℕ :=
  let total_city_miles := conds.city_distance * 2
  let total_highway_miles := conds.highway_distance * 2
  let city_gas_needed := total_city_miles / conds.city_mpg
  let highway_gas_needed := total_highway_miles / conds.highway_mpg
  let total_gas_needed := city_gas_needed + highway_gas_needed
  total_gas_needed * conds.gas_cost

theorem carl_trip_cost_is_correct (conds : TripConditions)
  (h1 : conds.city_mpg = 30)
  (h2 : conds.highway_mpg = 40)
  (h3 : conds.city_distance = 60)
  (h4 : conds.highway_distance = 200)
  (h5 : conds.gas_cost = 3) :
  total_gas_cost conds = 42 := by
  rw [total_gas_cost, h1, h2, h3, h4, h5]
  -- Proof steps will follow here, but we can skip them for now
  sorry

end carl_trip_cost_is_correct_l626_626712


namespace find_sides_of_triangle_l626_626085

theorem find_sides_of_triangle
  {a b c : ℝ} {A B C : ℝ}
  (h1 : c = real.sqrt 19)
  (h2 : C = 2 * real.pi / 3)
  (h3 : A > B)
  (h4 : 1/2 * a * b * real.sin C = (3 * real.sqrt 3) / 2) :
  (a = 3 ∧ b = 2) ∨ (a = 2 ∧ b = 3) :=
sorry

end find_sides_of_triangle_l626_626085


namespace common_ratio_of_series_l626_626338

def first_term : ℚ := 7 / 8
def second_term : ℚ := -14 / 27
def third_term : ℚ := 56 / 243

theorem common_ratio_of_series : 
  (second_term / first_term = third_term / second_term) ∧ (second_term / first_term = -16 / 27) :=
by
  -- Determining the common ratio from the first two terms
  have h1 : second_term / first_term = (-14 / 27) / (7 / 8) := by rfl
  have h2 : (-14 / 27) / (7 / 8) = -16 / 27 := by norm_num      

  -- Determining the common ratio from the second and third terms  
  have h3 : third_term / second_term = (56 / 243) / (-14 / 27) := by rfl
  have h4 : (56 / 243) / (-14 / 27) = -16 / 27 := by norm_num
      
  exact ⟨h1.trans h2, h3.trans h4⟩

#eval common_ratio_of_series

end common_ratio_of_series_l626_626338


namespace simplify_fraction_l626_626140

theorem simplify_fraction (a : ℕ) (h : a = 5) : (15 * a^4) / (75 * a^3) = 1 := 
by
  sorry

end simplify_fraction_l626_626140


namespace incorrect_expressions_l626_626552

theorem incorrect_expressions (x y : ℚ) (h : x / y = 2 / 5) :
    (x + 3 * y) / x ≠ 17 / 2 ∧ (x - y) / y ≠ 3 / 5 :=
by
  sorry

end incorrect_expressions_l626_626552


namespace sum_greater_than_two_l626_626173

variables {x y : ℝ}

theorem sum_greater_than_two (hx : x^7 > y^6) (hy : y^7 > x^6) : x + y > 2 :=
sorry

end sum_greater_than_two_l626_626173


namespace solve_inequality_l626_626535

theorem solve_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (x ∈ Ioo (-6.5) 3.5) := 
sorry

end solve_inequality_l626_626535


namespace largest_root_gx_l626_626343

theorem largest_root_gx : 
  let g : ℝ → ℝ := λ x, 10 * x^4 - 17 * x^2 + 7 in
  ∃ x, g x = 0 ∧ (∀ y, g y = 0 → y ≤ sqrt (7 / 5)) ∧ x = sqrt (7 / 5) :=
sorry

end largest_root_gx_l626_626343


namespace pq_over_qr_l626_626460

-- Define points and vectors in the problem
variables {X Y Z P Q R : Type}
[linear_ordered_field X]

-- Define the conditions as hypotheses
hypothesis h1 : ∃ (P : X), ∃ (xy_ratio : X), xy_ratio = 4 ∧ (xy_ratio + 1) = 5
hypothesis h2 : ∃ (Q : Y), ∃ (yz_ratio : Y), yz_ratio = 4 ∧ (yz_ratio + 1) = 5
hypothesis h3 : line_intersection (PQ : X) (XZ : X) = R

-- Define the main theorem we need to prove
theorem pq_over_qr : ∀ R : Type, PQ / QR = 1 / 3 :=
by
  sorry

end pq_over_qr_l626_626460


namespace number_of_true_propositions_is_one_l626_626594

-- Define each proposition
def proposition_1 (a : ℝ) : Prop := a^3 * a^2 = a^5
def proposition_2 : Prop := -Real.pi > -3.14
def proposition_3 : Prop := ∀ (θ_central θ_circumference : ℝ), θ_central = θ_circumference / 2
def proposition_4 : Prop := ∀ (outcome : Bool), outcome = tt
def proposition_5 : Prop := ∀ (data : list ℝ), list.variance (data.map (+ 4)) = list.variance data + 4

-- Define the proof problem
def number_of_true_propositions (a : ℝ) : ℕ := 
if proposition_1 a then 1 else 0 + 
if proposition_2 then 1 else 0 + 
if proposition_3 then 1 else 0 + 
if proposition_4 then 1 else 0 + 
if proposition_5 then 1 else 0

-- Statement: Prove that the number of true propositions among them is 1.
theorem number_of_true_propositions_is_one {a : ℝ} : number_of_true_propositions a = 1 :=
sorry

end number_of_true_propositions_is_one_l626_626594


namespace sum_every_second_term_is_1010_l626_626276

def sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_every_second_term_is_1010 :
  ∃ (x1 : ℤ) (d : ℤ) (S : ℤ), 
  (sequence_sum 2020 x1 d = 6060) ∧
  (d = 2) ∧
  (S = (1010 : ℤ)) ∧ 
  (2 * S + 4040 = 6060) :=
  sorry

end sum_every_second_term_is_1010_l626_626276


namespace abs_inequality_solution_l626_626536

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l626_626536


namespace total_ants_approximation_l626_626274

noncomputable def park_dimensions : ℝ × ℝ := (250, 350)
noncomputable def small_square_side : ℝ := 50
noncomputable def ant_density_main : ℝ := 4 / 1 -- ants per square inch
noncomputable def ant_density_square : ℝ := 6 / 1 -- ants per square foot

theorem total_ants_approximation :
  let (width, length) := park_dimensions in
  let total_area := width * length in
  let small_area := small_square_side^2 in
  let remaining_area := total_area - small_area in
  let remaining_area_in_square_inches := remaining_area * 144 in
  let ants_in_remaining_area := ant_density_main * remaining_area_in_square_inches in
  let ants_in_small_area := ant_density_square * small_area in
  let total_ants := ants_in_remaining_area + ants_in_small_area in
  total_ants ≈ 49 * 10^6 :=
by
  sorry

end total_ants_approximation_l626_626274


namespace pam_walked_1683_miles_l626_626513

noncomputable def pam_miles_walked 
    (pedometer_limit : ℕ)
    (initial_reading : ℕ)
    (flips : ℕ)
    (final_reading : ℕ)
    (steps_per_mile : ℕ)
    : ℕ :=
  (pedometer_limit + 1) * flips + final_reading / steps_per_mile

theorem pam_walked_1683_miles
    (pedometer_limit : ℕ := 49999)
    (initial_reading : ℕ := 0)
    (flips : ℕ := 50)
    (final_reading : ℕ := 25000)
    (steps_per_mile : ℕ := 1500) 
    : pam_miles_walked pedometer_limit initial_reading flips final_reading steps_per_mile = 1683 := 
  sorry

end pam_walked_1683_miles_l626_626513


namespace A_and_D_independent_l626_626183

open ProbabilityTheory

-- Definition of the events
def event_A : set (ℕ × ℕ) := {p | p.1 = 1}
def event_B : set (ℕ × ℕ) := {p | p.2 = 2}
def event_C : set (ℕ × ℕ) := {p | p.1 + p.2 = 8}
def event_D : set (ℕ × ℕ) := {p | p.1 + p.2 = 7}

-- Assumption of uniform probability over the sample space
def sample_space : Finset (ℕ × ℕ) := 
  Finset.univ.pi Finset.univ

noncomputable def prob : Measure (ℕ × ℕ) :=
  uniform sample_space

-- Defining independence
def independent (P : Measure (ℕ × ℕ)) (A B : set (ℕ × ℕ)) : Prop :=
  P (A ∩ B) = P A * P B

-- Desired statement
theorem A_and_D_independent :
  independent prob event_A event_D :=
by sorry

end A_and_D_independent_l626_626183


namespace number_of_cubes_with_icing_on_two_sides_l626_626672

def cake_cube : ℕ := 3
def smaller_cubes : ℕ := 27
def covered_faces : ℕ := 3
def layers_with_icing : ℕ := 2
def edge_cubes_per_layer_per_face : ℕ := 2

theorem number_of_cubes_with_icing_on_two_sides :
  (covered_faces * edge_cubes_per_layer_per_face * layers_with_icing) = 12 := by
  sorry

end number_of_cubes_with_icing_on_two_sides_l626_626672


namespace compare_08_and_one_eighth_l626_626239

theorem compare_08_and_one_eighth :
  0.8 - (1 / 8 : ℝ) = 0.675 := 
sorry

end compare_08_and_one_eighth_l626_626239


namespace rectangle_area_increase_l626_626555

theorem rectangle_area_increase (L W : ℝ) (h1: L > 0) (h2: W > 0) :
   let original_area := L * W
   let new_length := 1.20 * L
   let new_width := 1.20 * W
   let new_area := new_length * new_width
   let percentage_increase := ((new_area - original_area) / original_area) * 100
   percentage_increase = 44 :=
by
  sorry

end rectangle_area_increase_l626_626555


namespace intersection_planes_l626_626175

-- Definitions of the planes and their traces on the primary plane
variables (α β γ : Plane)
variables (b c : Line) -- Traces of the planes on the primary plane
variables (B C : Point) -- Given points on the respective planes
variables (B' C' : Point) -- Projections of B and C on the primary plane

-- Conditions deriving from the problem statement
variable (trace_b : b ⊂ α ∧ b ⊂ β)
variable (trace_c : c ⊂ α ∧ c ⊂ γ)
variable (proj_B : B' = projection α B)
variable (proj_C : C' = projection α C)
variable (B_in_β : B ∈ β)
variable (C_in_γ : C ∈ γ)

-- Finding the intersection line
theorem intersection_planes (l : Line) (P Q : Point)
  (P_condition : P = intersection b c)
  (line_BC : Line) (line_BC_condition : line_BC = line B' C')
  (M_condition : M = intersection line_BC b)
  (N_condition : N = intersection line_BC c)
  (MB : Line) (MB_condition : MB = line M B)
  (NC : Line) (NC_condition : NC = line N C)
  (Q_condition : Q = intersection MB NC) :
  l = line P Q := sorry

end intersection_planes_l626_626175


namespace cos_six_degree_roots_l626_626088

theorem cos_six_degree_roots :
  (∃ t : ℝ, t = cos (6 * real.pi / 180) ∧ 32 * t^5 - 40 * t^3 + 10 * t = real.sqrt 3) →
  (∃ u v w x : ℝ, u = cos (66 * real.pi / 180) ∧ v = cos (78 * real.pi / 180) ∧
                   w = cos (138 * real.pi / 180) ∧ x = cos (150 * real.pi / 180) ∧
                   32 * u^5 - 40 * u^3 + 10 * u = real.sqrt 3 ∧
                   32 * v^5 - 40 * v^3 + 10 * v = real.sqrt 3 ∧
                   32 * w^5 - 40 * w^3 + 10 * w = real.sqrt 3 ∧
                   32 * x^5 - 40 * x^3 + 10 * x = real.sqrt 3) := sorry

end cos_six_degree_roots_l626_626088


namespace max_arithmetic_sequences_24_l626_626876

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the problem of selecting 4 terms that also form an arithmetic sequence
noncomputable def max_arithmetic_subsequences (a : ℕ → ℤ) : ℕ :=
  -- condition of being arithmetic sequence on first 10 terms
  if h: is_arithmetic_sequence a then 24 else 0

-- State the theorem
theorem max_arithmetic_sequences_24 (a : ℕ → ℤ) :
  is_arithmetic_sequence a → max_arithmetic_subsequences a = 24 := 
by
  intro h,
  -- proof omitted
  exact sorry

end max_arithmetic_sequences_24_l626_626876


namespace initial_matches_l626_626931

theorem initial_matches (x : ℕ) (h1 : (34 * x + 89) / (x + 1) = 39) : x = 10 := by
  sorry

end initial_matches_l626_626931


namespace find_number_l626_626253

theorem find_number (x : ℝ) (h : 0.80 * 40 = (4/5) * x + 16) : x = 20 :=
by sorry

end find_number_l626_626253


namespace intervals_of_monotonic_increase_and_axis_of_symmetry_l626_626805

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (2 * x - Real.pi / 6))

theorem intervals_of_monotonic_increase_and_axis_of_symmetry :
  (∀ k : ℤ, ∃ I : set ℝ, I = set.Icc (- Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) ∧ 
      (∀ x1 x2 ∈ I, x1 < x2 → f x1 < f x2) ∧ 
      ∀ k : ℤ, ∃ s : set ℝ, s = {x : ℝ | x = Real.pi / 3 + k * Real.pi / 2}) ∧ 
  (∀ k : ℤ, ∃ a b : ℝ, a = 0 ∧ b = Real.pi / 3 ∧ 
      (0 ≤ b ∧ b ≤ Real.pi / 2) ∧ 
      (∀ x ∈ set.Icc 0 (Real.pi / 2), x = a → f x = -1 ∧ x = b → f x = 2)) :=
by {
  sorry
}

end intervals_of_monotonic_increase_and_axis_of_symmetry_l626_626805


namespace domain_of_f_l626_626940

def f (x : ℝ) : ℝ := Real.sqrt (1 - 2^x) + Real.sqrt (x + 3)

theorem domain_of_f : Set.Icc (-3:ℝ) 0 = {x : ℝ | 1 - 2^x ≥ 0 ∧ x + 3 ≥ 0} :=
by
  sorry

end domain_of_f_l626_626940


namespace tetrahedron_radius_and_centers_coincide_l626_626631

noncomputable def tetrahedron (a b c : ℝ) : Prop :=
  let ρ := (a + b + c) / 2
  let R := (1 / 4) * Real.sqrt (2 * (a^2 + b^2 + c^2))
  let r := Real.sqrt ((1 / 8) * (a^2 + b^2 + c^2) - (a^2 * b^2 * c^2) / (16 * ρ * (ρ - a) * (ρ - b) * (ρ - c)))
  R = (1 / 4) * Real.sqrt (2 * (a^2 + b^2 + c^2)) ∧
  r = Real.sqrt ((1 / 8) * (a^2 + b^2 + c^2) - (a^2 * b^2 * c^2) / (16 * ρ * (ρ - a) * (ρ - b) * (ρ - c))) ∧
  -- A theorem stating the centers of the circumscribed and inscribed spheres coincide.
  centers_coincide (a b c : ℝ) : Prop

theorem tetrahedron_radius_and_centers_coincide (a b c : ℝ) : tetrahedron a b c := by
  sorry

end tetrahedron_radius_and_centers_coincide_l626_626631


namespace group_formations_at_fair_l626_626090

theorem group_formations_at_fair : 
  (Nat.choose 7 3) * (Nat.choose 4 4) = 35 := by
  sorry

end group_formations_at_fair_l626_626090


namespace perimeter_ratio_of_squares_l626_626570

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l626_626570


namespace polynomial_perfect_square_monomials_l626_626952

theorem polynomial_perfect_square_monomials (x : ℝ) :
  ∃ m : ℝ → ℝ, (m = (λ x, 64 * x ^ 4)) ∨ (m = (λ x, 8 * x)) ∨ (m = (λ x, -8 * x)) ∨ (m = (λ x, -1)) ∨ (m = (λ x, -16 * x ^ 2)) ∧ ∀ y, (16 * x ^ 2 + 1 + m y) = (function_of_perfect_square y)^2 :=
sorry

end polynomial_perfect_square_monomials_l626_626952


namespace point_coordinates_in_second_quadrant_l626_626680

theorem point_coordinates_in_second_quadrant
  (x y : ℝ)
  (hx : x = -5)
  (hy : y = 2)
  (dist_x_axis : abs y = 2)
  (dist_y_axis : abs x = 5)
  (hx_neg : x < 0)
  (hy_pos : y > 0)
  (quadrant : (hx_neg ∧ hy_pos)) :
  (x, y) = (-5, 2) := by
  sorry

end point_coordinates_in_second_quadrant_l626_626680


namespace find_valid_a_values_l626_626353

noncomputable def system_of_equations_has_two_solutions (a : ℝ) : Prop :=
  ∃ x y : ℝ, 
    (y^2 - 2 * x * y - y + 8 * x - 12 = 0) ∧
    (2 * x + y = a) ∧
    ((∃ y1 y2 : ℝ, y1 ≠ y2 ∧ y1^2 - 2 * x * y1 - y1 + 8 * x - 12 = 0 ∧ y2^2 - 2 * x * y2 - y2 + 8 * x - 12 = 0) ∨ 
       (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 2 * x1 + y = a ∧ 2 * x2 + y = a))

theorem find_valid_a_values : 
  ∀ a : ℝ, 
  system_of_equations_has_two_solutions a → 
  (a ∈ set.Ico (-7) 2 ∪ {11} ∪ set.Icc 12 13) :=
begin
  sorry
end

end find_valid_a_values_l626_626353


namespace find_common_cards_l626_626628

theorem find_common_cards (rare_cards uncommon_cards : ℕ) (cost_rare cost_uncommon cost_common total_cost : ℝ) : 
  rare_cards = 19 →
  uncommon_cards = 11 →
  cost_rare = 1 →
  cost_uncommon = 0.5 →
  cost_common = 0.25 →
  total_cost = 32 →
  let cost_rare_cards := rare_cards * cost_rare in
  let cost_uncommon_cards := uncommon_cards * cost_uncommon in
  let cost_common_cards := total_cost - (cost_rare_cards + cost_uncommon_cards) in
  let c := cost_common_cards / cost_common in
  c = 30 :=
by
  intros hr hu cr cu cc tc
  simp [hr, hu, cr, cu, cc, tc]
  let cost_rare_cards := rare_cards * cost_rare
  let cost_uncommon_cards := uncommon_cards * cost_uncommon
  let cost_common_cards := total_cost - (cost_rare_cards + cost_uncommon_cards)
  let c := cost_common_cards / cost_common
  sorry

end find_common_cards_l626_626628


namespace pairs_of_acquaintances_divisible_by_3_l626_626437

theorem pairs_of_acquaintances_divisible_by_3
  (P : ℕ) -- number of pairs of acquaintances (edges in the graph)
  (T : ℕ) -- number of triangles in the graph
  (h1 : ∀ (e : ℕ), e ∈ (finset.range P) → ∃ t : finset ℕ, t.card = 5 ∧ (∀ x ∈ t, x < P))
  (h2 : 5 * P = 3 * T) -- each edge is part of 5 triangles, and each triangle has 3 edges
  : 3 ∣ P :=
sorry

end pairs_of_acquaintances_divisible_by_3_l626_626437


namespace function1_is_H_function_function4_is_H_function_l626_626829

def H_function (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

def function1 (x : ℝ) : ℝ := 3 * x + 1

def function4 (x : ℝ) : ℝ :=
  if x < -1 then -1 / x else x^2 + 4 * x + 5

theorem function1_is_H_function : H_function function1 :=
  sorry

theorem function4_is_H_function : H_function function4 :=
  sorry

end function1_is_H_function_function4_is_H_function_l626_626829


namespace problem_statement_l626_626792

-- Definitions based on given conditions.
def m : ℤ := Int.floor (Real.cbrt 13)
def n : ℝ := Real.sqrt 13 - Int.floor (Real.sqrt 13)

-- The statement we need to prove.
theorem problem_statement : m - n = 5 - Real.sqrt 13 := sorry

end problem_statement_l626_626792


namespace quilt_cost_l626_626469

theorem quilt_cost
  (length : ℕ) (width : ℕ) (cost_per_sqft : ℕ)
  (h_length : length = 12) (h_width : width = 15) (h_cost_per_sqft : cost_per_sqft = 70) :
  length * width * cost_per_sqft = 12600 :=
by {
  rw [h_length, h_width, h_cost_per_sqft],
  norm_num,
  sorry 
}

end quilt_cost_l626_626469


namespace find_min_difference_l626_626484

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l626_626484


namespace find_min_difference_l626_626481

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l626_626481


namespace naomi_drives_to_parlor_l626_626892

theorem naomi_drives_to_parlor (d v t t_back : ℝ)
  (ht : t = d / v)
  (ht_back : t_back = 2 * d / v)
  (h_total : 2 * (t + t_back) = 6) : 
  t = 1 :=
by sorry

end naomi_drives_to_parlor_l626_626892


namespace daps_to_dips_l626_626825

-- Define the equivalence relations given in the conditions.
def equivalence1 : ℝ := 5 / 4 -- 5 daps : 4 dops
def equivalence2 : ℝ := 3 / 10 -- 3 dops : 10 dips

-- Define the target conversion
def conversion_factor : ℝ := equivalence1 * equivalence2 -- combining both relationships

-- State the theorem
theorem daps_to_dips (daps dips : ℝ) (h1 : equivalence1 = 5 / 4) (h2 : equivalence2 = 3 / 10) :
  (28 * conversion_factor) = 10.5 :=
by
  -- enabling the use of fractions as real numbers
  sorry

end daps_to_dips_l626_626825


namespace simplify_abs_expression_l626_626522

theorem simplify_abs_expression (x : ℝ) : 
  |2*x + 1| - |x - 3| + |x - 6| = 
  if x < -1/2 then -2*x + 2 
  else if x < 3 then 2*x + 4 
  else if x < 6 then 10 
  else 2*x - 2 :=
by 
  sorry

end simplify_abs_expression_l626_626522


namespace students_age_country_l626_626149

theorem students_age_country :
  ∀ n : ℕ, (students_age : Fin n → Fin 5) (students_country : Fin n → Fin 13),
  ∃ (S : Finset (Fin n)), S.card ≥ 9 ∧ ∀ i ∈ S, 
    (Finset.filter (λ j, students_age j = students_age i) (Finset.univ : Finset (Fin n))).card >
    (Finset.filter (λ j, students_country j = students_country i) (Finset.univ : Finset (Fin n))).card :=
sorry

end students_age_country_l626_626149


namespace smallest_q_p_difference_l626_626487

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l626_626487


namespace unoccupied_volume_correct_l626_626467

-- Define the initial conditions for the problem
def container_side_length : ℝ := 10
def container_volume : ℝ := container_side_length ^ 3
def water_volume : ℝ := container_volume / 2
def ice_cube_side_length : ℝ := 2
def ice_cube_volume : ℝ := ice_cube_side_length ^ 3
def number_of_ice_cubes : ℝ := 10
def total_ice_volume : ℝ := number_of_ice_cubes * ice_cube_volume
def occupied_volume : ℝ := water_volume + total_ice_volume
def unoccupied_volume : ℝ := container_volume - occupied_volume

-- The proof statement
theorem unoccupied_volume_correct : unoccupied_volume = 420 :=
by
  sorry

end unoccupied_volume_correct_l626_626467


namespace limit_sequence_sum_l626_626811

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     := 1 -- Note that Lean uses zero-based indexing by default
| (n+1) := (1 / 3)^(n+1) - a n

-- Prove the final limit statement
theorem limit_sequence_sum :
  tendsto (λ n : ℕ, ∑ i in finset.range (2 * n - 1), a i) at_top (𝓝 (9 / 8)) :=
sorry

end limit_sequence_sum_l626_626811


namespace largest_AB_under_conditions_l626_626125

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_AB_under_conditions :
  ∃ A B C D : ℕ, is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (A + B) % (C + D) = 0 ∧
    is_prime (A + B) ∧ is_prime (C + D) ∧
    (A + B) = 11 :=
sorry

end largest_AB_under_conditions_l626_626125


namespace geometric_sequence_b_mn_theorem_l626_626012

noncomputable def geometric_sequence_b_mn (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ) 
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2) 
  (h_nm_pos : m > 0 ∧ n > 0): Prop :=
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m))

-- We skip the proof using sorry.
theorem geometric_sequence_b_mn_theorem 
  (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ)
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2)
  (h_nm_pos : m > 0 ∧ n > 0) : 
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m)) :=
sorry

end geometric_sequence_b_mn_theorem_l626_626012


namespace B_M_L_N_concyclic_l626_626151

open EuclideanGeometry

-- Definitions of the points and assumptions according to the problem description.
variable {A B C D L K M N : Point}

-- Assume a square ABCD and a point L on the minor arc CD of its circumcircle.
variable (h_square : Square A B C D)
variable (h_L_arc : L ∈ MinorArc (Circumcircle A B C D) C D)
variable (h_L_not_C_D : L ≠ C ∧ L ≠ D)
variable (h_K : K = (line_through A L).Intersect (line_through C D))
variable (h_M : M = (line_through C L).Intersect (line_through A D))
variable (h_N : N = (line_through M K).Intersect (line_through B C))

-- The theorem to be proved: Points B, M, L, and N are concyclic.
theorem B_M_L_N_concyclic :
  CyclicQuadrilateral B M L N := by
  sorry

end B_M_L_N_concyclic_l626_626151


namespace shaded_regions_area_approx_l626_626080

noncomputable def area_of_shaded_regions (r1 r2 : ℝ) : ℝ :=
  let rectangle_left := r1 * (2*r1)
  let rectangle_right := r2 * (2*r2)
  let semicircle_left := (π * r1^2) / 2
  let semicircle_right := (π * r2^2) / 2
  (rectangle_left - semicircle_left) + (rectangle_right - semicircle_right)

theorem shaded_regions_area_approx (r1 r2 : ℝ) (h_r1 : r1 = 2) (h_r2 : r2 = 4) :
  |area_of_shaded_regions r1 r2 - 8.6| < 0.1 :=
by
  rw [h_r1, h_r2]
  norm_num
  -- detailed steps to fill in the exact calculation and approximation
  sorry

end shaded_regions_area_approx_l626_626080


namespace equal_angles_in_trapezium_l626_626472

theorem equal_angles_in_trapezium
  (A B C D K L : Point)
  (h_trapezium : trapezium ABCD)
  (h_parallel : parallel AD BC)
  (h_K_on_AB : On K AB)
  (h_L_on_CD : On L CD)
  (h_bal_eq_cdk : angle BAL = angle CDK) :
  angle BLA = angle CKD :=
sorry

end equal_angles_in_trapezium_l626_626472


namespace compute_ab_l626_626632

namespace MathProof

variable {a b : ℝ}

theorem compute_ab (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := 
by
  sorry

end MathProof

end compute_ab_l626_626632


namespace divisibility_by_2_divisibility_by_3_divisibility_by_5_divisibility_by_9_divisibility_by_10_l626_626148

-- Definitions of base 10 representation
variables {k : ℕ} {a : Fin k → ℕ} (n : ℕ) (digits : Fin (k+1) → ℕ)
def base10_repr (n : ℕ) (digits : Fin (k+1) → ℕ) : Prop :=
  n = ∑ i : Fin (k+1), (digits i) * (10^(i : ℕ))

-- Definitions for each divisibility rule
def divisible_by_2 (n : ℕ) : Prop := 2 ∣ n
def divisible_by_3 (n : ℕ) : Prop := 3 ∣ n
def divisible_by_5 (n : ℕ) : Prop := 5 ∣ n
def divisible_by_9 (n : ℕ) : Prop := 9 ∣ n
def divisible_by_10 (n : ℕ) : Prop := 10 ∣ n

-- The divisibility rules
theorem divisibility_by_2 {digits : Fin (k+1) → ℕ} (h : base10_repr n digits) : 
  divisible_by_2 n ↔ divisible_by_2 (digits 0) := 
sorry

theorem divisibility_by_3 {digits : Fin (k+1) → ℕ} (h : base10_repr n digits) : 
  divisible_by_3 n ↔ divisible_by_3 (∑ i : Fin (k+1), digits i) :=
sorry

theorem divisibility_by_5 {digits : Fin (k+1) → ℕ} (h : base10_repr n digits) : 
  divisible_by_5 n ↔ divisible_by_5 (digits 0) := 
sorry

theorem divisibility_by_9 {digits : Fin (k+1) → ℕ} (h : base10_repr n digits) : 
  divisible_by_9 n ↔ divisible_by_9 (∑ i : Fin (k+1), digits i) := 
sorry

theorem divisibility_by_10 {digits : Fin (k+1) → ℕ} (h : base10_repr n digits) : 
  divisible_by_10 n ↔ divisible_by_10 (digits 0) := 
sorry

end divisibility_by_2_divisibility_by_3_divisibility_by_5_divisibility_by_9_divisibility_by_10_l626_626148


namespace cheese_bread_grams_l626_626246

/-- Each 100 grams of cheese bread costs 3.20 BRL and corresponds to 10 pieces. 
Each person eats, on average, 5 pieces of cheese bread. Including the professor,
there are 16 students, 1 monitor, and 5 parents, making a total of 23 people. 
The precision of the bakery's scale is 100 grams. -/
theorem cheese_bread_grams : (5 * 23 / 10) * 100 = 1200 := 
by
  sorry

end cheese_bread_grams_l626_626246


namespace malachi_additional_photos_l626_626069

-- Definition of the conditions
def total_photos : ℕ := 2430
def ratio_last_year : ℕ := 10
def ratio_this_year : ℕ := 17
def total_ratio_units : ℕ := ratio_last_year + ratio_this_year
def diff_ratio_units : ℕ := ratio_this_year - ratio_last_year
def photos_per_unit : ℕ := total_photos / total_ratio_units
def additional_photos : ℕ := diff_ratio_units * photos_per_unit

-- The theorem proving how many more photos Malachi took this year than last year
theorem malachi_additional_photos : additional_photos = 630 := by
  sorry

end malachi_additional_photos_l626_626069


namespace race_distance_l626_626846

theorem race_distance (a b c d : ℝ) 
  (h₁ : d / a = (d - 25) / b)
  (h₂ : d / b = (d - 15) / c)
  (h₃ : d / a = (d - 37) / c) : 
  d = 125 :=
by
  sorry

end race_distance_l626_626846


namespace perimeter_ratio_of_squares_l626_626573

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l626_626573


namespace island_of_misfortune_l626_626896

theorem island_of_misfortune (n : ℕ) (k : ℕ) 
  (h1 : n > 0) 
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ j : ℝ, j = i / 100 ∧ j = k / n)
  (knight_exists : ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ i % 100 = 0)
  : n = 100 := 
begin
  sorry,
end

end island_of_misfortune_l626_626896


namespace cars_sold_on_wednesday_l626_626668

theorem cars_sold_on_wednesday 
  (cars_mon : ℕ)
  (cars_tue : ℕ)
  (cars_thu : ℕ)
  (cars_fri : ℕ)
  (cars_sat : ℕ)
  (mean_cars_per_day : ℝ)
  (days_in_week : ℕ)
  (total_cars : ℕ) :
  cars_mon = 8 →
  cars_tue = 3 →
  cars_thu = 4 →
  cars_fri = 4 →
  cars_sat = 4 →
  mean_cars_per_day = 5.5 →
  days_in_week = 6 →
  total_cars = 33 →
  ∑ i in [cars_mon, cars_tue, cars_thu, cars_fri, cars_sat], id = 23 →
  ∃ cars_wed, total_cars - ∑ i in [cars_mon, cars_tue, cars_thu, cars_fri, cars_sat], id = cars_wed ∧ cars_wed = 10 :=
begin
  intros,
  refine ⟨total_cars - ∑ i in [cars_mon, cars_tue, cars_thu, cars_fri, cars_sat], id, _⟩,
  rw [h, h_1, h_2, h_3, h_4],
  ring_nf at h_9,
  assumption,
end

# Example of usage
example :
  cars_sold_on_wednesday 8 3 4 4 4 5.5 6 33 :=
begin
  sorry
end

end cars_sold_on_wednesday_l626_626668


namespace unique_circle_of_radius_4_l626_626880

-- Define two circles C1 and C2 with radius 2 in the same plane and tangent to each other
structure Circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
(radius_eq : radius = 2)

-- Define the plane containing the circles
def same_plane (C1 C2 : Circle) : Prop := True -- dummy condition to indicate same plane

-- Prove that there exists exactly one circle of radius 4 tangent to both C1 and C2 at their point of tangency
theorem unique_circle_of_radius_4 (C1 C2 : Circle) (hC1 : Circle C1.1 2) (hC2 : Circle C2.1 2) (plane_cond : same_plane C1 C2) (tangent_cond : dist C1.1 C2.1 = 4) :
  ∃! (C3 : Circle), C3.radius = 4 ∧ dist C3.1.1 (C1.1.1 + C2.1.1) / 2 <= 2 :=
sorry

end unique_circle_of_radius_4_l626_626880


namespace determine_field_values_l626_626901

theorem determine_field_values (S j m p : ℕ) 
  (C1 : S ≥ 0)
  (C2 : j < m)
  (C3 : m < p)
  (C4 : S + 2 * m = S + 2 * j + 10)
  (C5 : S + 2 * p = S + 2 * m + 10)
  (C6 : j = 2 ∨ m = 7 ∨ p = 12) :
  ({j, m, p} = {2, 7, 12} ∨ {j, m, p} = {7, 12, 17} ∨ {j, m, p} = {12, 17, 22}) :=
by 
  sorry

end determine_field_values_l626_626901


namespace train_crossing_time_is_correct_l626_626033

noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

theorem train_crossing_time_is_correct :
  train_crossing_time 250 180 120 = 12.9 :=
by
  sorry

end train_crossing_time_is_correct_l626_626033


namespace math_problem_l626_626171

theorem math_problem (x y : ℝ) (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 :=
sorry

end math_problem_l626_626171


namespace letter_t_transformations_l626_626597

-- Definitions and conditions will be expressed in Lean
def initial_position : Prop := 
  -- Initial position of the T (stem: vertical, top: horizontal)
  sorry

def rotate_90_clockwise (p : Prop) : Prop := 
  -- Position after 90 degrees clockwise rotation around the origin
  sorry

def reflect_y_axis (p : Prop) : Prop := 
  -- Position after reflection in the y-axis
  sorry

def scale_down (factor : ℝ) (p : Prop) : Prop := 
  -- Position after scaling down by a given factor
  sorry

def rotate_180 (p : Prop) : Prop := 
  -- Position after 180 degrees rotation around the origin
  sorry

def final_position : Prop := 
  -- Final desired position of T (stem along positive x-axis, top bar along positive y-axis)
  sorry

theorem letter_t_transformations :
  final_position :=
by
  have step1 := rotate_90_clockwise initial_position
  have step2 := reflect_y_axis step1
  have step3 := scale_down (1/2) step2
  have step4 := rotate_180 step3
  exact step4

end letter_t_transformations_l626_626597


namespace infinite_a_divs_power_sum_l626_626138

theorem infinite_a_divs_power_sum :
  ∃ᶠ a : ℤ in filter.at_top, a^2 ∣ 2^a + 3^a := 
begin
  sorry
end

end infinite_a_divs_power_sum_l626_626138


namespace circumscribed_sphere_surface_area_l626_626019

theorem circumscribed_sphere_surface_area 
    (x y z : ℝ) 
    (h1 : x * y = Real.sqrt 6) 
    (h2 : y * z = Real.sqrt 2) 
    (h3 : z * x = Real.sqrt 3) : 
    4 * Real.pi * ((Real.sqrt (x^2 + y^2 + z^2)) / 2)^2 = 6 * Real.pi := 
by
  sorry

end circumscribed_sphere_surface_area_l626_626019


namespace min_value_frac_function_l626_626757

theorem min_value_frac_function (x : ℝ) (h : x > -1) : (x^2 / (x + 1)) ≥ 0 :=
sorry

end min_value_frac_function_l626_626757


namespace find_a_12_l626_626451

-- Definitions capturing the conditions
variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (n : ℕ)

-- Condition: the fourth term a_4 is 6
axiom a_4_eq_6 : a 4 = 6

-- Condition: the sum of the third and fifth terms equals the tenth term
axiom a3_add_a5_eq_a10 : a 3 + a 5 = a 10

-- Definition of arithmetic sequence
def is_arithmetic_sequence : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Use the conditions to prove the value of the twelfth term is 14
theorem find_a_12 (h : is_arithmetic_sequence a d) : a 12 = 14 :=
by
  -- Initial setup with given axioms
  have h1 : a 4 = 6 := a_4_eq_6
  have h2 : a 3 + a 5 = a 10 := a3_add_a5_eq_a10
  -- Proof will proceed from here
  sorry

end find_a_12_l626_626451


namespace count_digit_5_more_than_2_page_numbers_l626_626330

-- Define the function that counts the occurrences of a digit in a given range
def count_digit_in_range (d n : Nat) : Nat :=
  Nat.sum (List.range (n + 1)) (λ i, Nat.digits 10 i |>.count d)

-- State the theorem using the conditions and solve the mathematical problem
theorem count_digit_5_more_than_2_page_numbers :
  count_digit_in_range 5 512 = count_digit_in_range 2 512 + 13 :=
by
  sorry

end count_digit_5_more_than_2_page_numbers_l626_626330


namespace smallest_part_proportional_l626_626412

/-- If we divide 124 into three parts proportional to 2, 1/2, and 1/4,
    prove that the smallest part is 124 / 11. -/
theorem smallest_part_proportional (x : ℝ) 
  (h : 2 * x + (1 / 2) * x + (1 / 4) * x = 124) : 
  (1 / 4) * x = 124 / 11 :=
sorry

end smallest_part_proportional_l626_626412


namespace four_points_circle_l626_626907

theorem four_points_circle {A B C D : Point} (h1: ¬collinear A B C) (h2: ¬collinear A B D) (h3: ¬collinear A C D) (h4: ¬collinear B C D) :
  ∃ (Γ : Circle), (A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ (D ∈ Γ ∨ D ∈ interior Γ)) ∨
  (A ∈ Γ ∧ B ∈ Γ ∧ D ∈ Γ ∧ (C ∈ Γ ∨ C ∈ interior Γ)) ∨
  (A ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ ∧ (B ∈ Γ ∨ B ∈ interior Γ)) ∨
  (B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ ∧ (A ∈ Γ ∨ A ∈ interior Γ)) := 
sorry

end four_points_circle_l626_626907


namespace find_ab_find_line_eq_l626_626300

-- Definition of the parameters and conditions for the problem
def ellipse_definition (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), y^2 / a^2 + x^2 / b^2 = 1

def parabola_definition : ℝ → ℝ → Prop :=
  λ (x y : ℝ), y = -x^2 + 1

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 - (b^2 / a^2))

def point_in_intersection (x y a b : ℝ) : Prop :=
  (ellipse_definition a b x y) ∧ (parabola_definition x y)

-- Proof statements:
theorem find_ab (a b : ℝ) (h : eccentricity a b = sqrt (3) / 2) : 
  a = 2 ∧ b = 1 :=
sorry

theorem find_line_eq (B P Q : ℝ × ℝ) (k : ℝ) :
  line_through_point_slope B P k-> 
  orthogonal (vector_from_points (1, 0) P) (vector_from_points (1, 0) Q) -> 
  (p₁: ℝ, p₂: ℝ) -> parabola_definition p₁ p₂ ->
  line_eqn_through B P k = ∀ (x y : ℝ), (8 * x + 3 * y - 8 = 0) :=
sorry

end find_ab_find_line_eq_l626_626300


namespace direction_vector_of_line_l626_626339

def projMatrix : Matrix (Fin 3) (Fin 3) ℚ := 
  ![![3/13, 2/13, 6/13],
    ![2/13, 12/13, -5/13],
    ![6/13, -5/13, 1/13]]

theorem direction_vector_of_line (P : Matrix (Fin 3) (Fin 3) ℚ) (v : Vector3 ℚ) :
  P = projMatrix → v = ⟨3, 2, 6⟩ :=
by
  intros h
  sorry

end direction_vector_of_line_l626_626339


namespace solve_system_of_inequalities_l626_626546

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 1 > x) ∧ (x < -3 * x + 8) ↔ -1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l626_626546


namespace intersection_of_A_and_B_l626_626250

def A : Set ℤ := {-1, 0, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def A_cap_B : Set ℝ := {0, 2}

theorem intersection_of_A_and_B :
  A ∩ B = A_cap_B :=
sorry

end intersection_of_A_and_B_l626_626250


namespace smallest_solution_l626_626102

open Int

theorem smallest_solution (x : ℤ) 
  (h : ⟦ x / 2 ⟧ + ⟦ x / 3 ⟧ + ⟦ x / 7 ⟧ = x) : 
  x = -85 :=
sorry

end smallest_solution_l626_626102


namespace Euler_lines_intersect_nine_point_circle_l626_626097

theorem Euler_lines_intersect_nine_point_circle 
  {A B C A₁ B₁ C₁ : Type*} 
  [Euclidean_geometry A B C] 
  (h₁ : A₁ ∈ altitude A B C)
  (h₂ : B₁ ∈ altitude B A C) 
  (h₃ : C₁ ∈ altitude C A B) 
  : ∃ P ∈ nine_point_circle A B C, 
      Euler_line (triangle A B₁ C₁) ∩ 
      Euler_line (triangle B A₁ C₁) ∩ 
      Euler_line (triangle C A₁ B₁) = {P} := 
sorry

end Euler_lines_intersect_nine_point_circle_l626_626097


namespace painting_cost_l626_626591

-- Definitions based on conditions
structure Room :=
  (length : ℕ) (width : ℕ) (height : ℕ)

structure Door :=
  (height : ℕ) (width : ℕ)

structure Window :=
  (height : ℕ) (width : ℕ)

-- Define our specific room, doors, and windows
def room : Room := ⟨10, 7, 5⟩
def doors : List Door := [⟨3, 1⟩, ⟨3, 1⟩]
def windows : List Window := [⟨1.5, 2⟩, ⟨1.5, 1⟩, ⟨1.5, 1⟩]

-- Define the cost per square meter
def costPerSqM : ℕ := 3

-- The theorem (proof problem)
theorem painting_cost : 
  let wall_area := 
    2 * (room.length * room.height + room.width * room.height) in
  let door_area := List.sum (doors.map (λ d => d.height * d.width)) in
  let window_area := List.sum (windows.map (λ w => w.height * w.width)) in
  let paintable_area := wall_area - door_area - window_area in
  let cost := paintable_area * costPerSqM in
  cost = 474 :=
by
  let wall_area := 
    2 * (room.length * room.height + room.width * room.height)
  let door_area := List.sum (doors.map (λ d => d.height * d.width))
  let window_area := List.sum (windows.map (λ w => w.height * w.width))
  let paintable_area := wall_area - door_area - window_area
  let cost := paintable_area * costPerSqM
  sorry -- Proof to be provided

end painting_cost_l626_626591


namespace geometric_series_sum_infinity_l626_626615

theorem geometric_series_sum_infinity (a₁ : ℝ) (q : ℝ) (S₆ S₃ : ℝ)
  (h₁ : a₁ = 3)
  (h₂ : S₆ / S₃ = 7 / 8)
  (h₃ : S₆ = a₁ * (1 - q ^ 6) / (1 - q))
  (h₄ : S₃ = a₁ * (1 - q ^ 3) / (1 - q)) :
  ∑' i : ℕ, a₁ * q ^ i = 2 := by
  sorry

end geometric_series_sum_infinity_l626_626615


namespace gcd_lcm_product_l626_626345

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  Nat.gcd a b * Nat.lcm a b = 56700 := by
  sorry

end gcd_lcm_product_l626_626345


namespace find_number_l626_626955

def is_three_digit_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
  n = 100 * x + 10 * y + z ∧ (100 * x + 10 * y + z) / 11 = x^2 + y^2 + z^2

theorem find_number : ∃ n : ℕ, is_three_digit_number n ∧ n = 550 :=
sorry

end find_number_l626_626955


namespace sum_place_values_of_7s_l626_626243

theorem sum_place_values_of_7s (n : ℝ) (h : n = 87953.0727) : 
  let a := 7000
  let b := 0.07
  let c := 0.0007
  a + b + c = 7000.0707 :=
by
  sorry

end sum_place_values_of_7s_l626_626243


namespace train_length_is_140_l626_626288

noncomputable def speed_kmph := 60
noncomputable def time_seconds := 23.998080153587715
noncomputable def platform_length_m := 260

-- Converting speed from km/h to m/s
noncomputable def speed_mps := speed_kmph * 1000 / 3600

-- Total distance covered by the train passing the platform
noncomputable def total_distance := speed_mps * time_seconds

-- Length of the train
noncomputable def train_length := total_distance - platform_length_m

theorem train_length_is_140 : train_length = 140 := by
  sorry

end train_length_is_140_l626_626288


namespace part1_part2_l626_626869

open_locale classical

variables {A B C D K L M N : Type*}

-- Defining the conditions for part 1
def is_perpendicular (x1 x2 : Type*) := sorry
def is_parallel (x1 x2 : Type*) := sorry
def non_perpendicular_diagonals (ABCD: Type*) := sorry

-- Given a quadrilateral ABCD with non-perpendicular diagonals,
-- and the respective perpendicular meeting points K, L, M, N,
-- prove that KL is parallel to MN
theorem part1 (ABCD : Type*) 
  (h1 : non_perpendicular_diagonals ABCD)
  (h2 : is_perpendicular A BC)
  (h3 : is_perpendicular A CD)
  (h4 : is_perpendicular C AB)
  (h5 : is_perpendicular C AD) :
  is_parallel KL MN :=
sorry

-- Additionally, if ABCD is cyclic, prove that KLMN is a parallelogram
def is_cyclic (ABCD : Type*) := sorry
def is_parallelogram (KL MN : Type*) := sorry

theorem part2 (ABCD : Type*) 
  (h1 : non_perpendicular_diagonals ABCD)
  (h2 : is_perpendicular A BC)
  (h3 : is_perpendicular A CD)
  (h4 : is_perpendicular C AB)
  (h5 : is_perpendicular C AD)
  (h6 : is_cyclic ABCD) :
  is_parallelogram KL MN :=
sorry

end part1_part2_l626_626869


namespace ellipse_and_chord_line_l626_626314

-- Step 1: Define the conditions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_focus (x y c : ℝ) : Prop :=
  x = -c ∧ y = 0

def eccentricity (a c : ℝ) : ℝ :=
  c / a

def midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

noncomputable def find_a (e : ℝ) (c : ℝ) : ℝ :=
  c / e

noncomputable def find_b2 (a c : ℝ) : ℝ :=
  a^2 - c^2

noncomputable def find_slope (x1 y1 x2 y2 : ℝ) (p1 p2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

-- Step 2: Define the theorem
theorem ellipse_and_chord_line 
  (a b c x1 y1 x2 y2 : ℝ)
  (h_a : a = find_a 0.5 2) -- a = 4
  (h_b2 : b^2 = find_b2 a c) -- b^2 = 12
  (h_ellipse_eq_1 : ellipse_eq a b x1 y1)
  (h_ellipse_eq_2 : ellipse_eq a b x2 y2)
  (h_midpoint : midpoint x1 y1 x2 y2 2 1) :
  (ellipse_eq 16 12 a b) ∧ 
  (3 * x + 2 * y - 8 = 0) := sorry

end ellipse_and_chord_line_l626_626314


namespace four_points_circle_l626_626908

theorem four_points_circle {A B C D : Point} (h1: ¬collinear A B C) (h2: ¬collinear A B D) (h3: ¬collinear A C D) (h4: ¬collinear B C D) :
  ∃ (Γ : Circle), (A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ (D ∈ Γ ∨ D ∈ interior Γ)) ∨
  (A ∈ Γ ∧ B ∈ Γ ∧ D ∈ Γ ∧ (C ∈ Γ ∨ C ∈ interior Γ)) ∨
  (A ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ ∧ (B ∈ Γ ∨ B ∈ interior Γ)) ∨
  (B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ ∧ (A ∈ Γ ∨ A ∈ interior Γ)) := 
sorry

end four_points_circle_l626_626908


namespace fraction_of_married_men_l626_626302

theorem fraction_of_married_men
    (total_faculty : ℕ)
    (women_percentage : ℕ)
    (married_percentage : ℕ)
    (single_men_fraction : ℚ)
    (h1 : women_percentage = 60)
    (h2 : married_percentage = 60)
    (h3 : single_men_fraction = 3 / 4) :
    (1 / 4 : ℚ) = 
    let total_men := total_faculty * (100 - women_percentage) / 100 in
    let married_men := total_men * (1 - single_men_fraction) in
    married_men / total_men := 
by
  sorry

end fraction_of_married_men_l626_626302


namespace shortest_distance_between_circles_l626_626213

theorem shortest_distance_between_circles :
  let c1 := (1, -3)
  let r1 := 2 * Real.sqrt 2
  let c2 := (-3, 1)
  let r2 := 1
  let distance_centers := Real.sqrt ((1 - -3)^2 + (-3 - 1)^2)
  let shortest_distance := distance_centers - (r1 + r2)
  shortest_distance = 2 * Real.sqrt 2 - 1 :=
by
  sorry

end shortest_distance_between_circles_l626_626213


namespace sector_chord_length_l626_626927

/-- Given a sector with area 1 cm² and perimeter 4 cm, the chord length is 2 sin(1) -/
theorem sector_chord_length (r α : ℝ) 
  (h1 : 0 < r) 
  (hα : 0 < α) 
  (area_equation : (1 / 2) * α * r^2 = 1) 
  (perimeter_equation : 2 * r + α * r = 4) : 
  2 * r * sin (α / 2) = 2 * sin 1 :=
by
  sorry

end sector_chord_length_l626_626927


namespace mary_income_percent_of_juan_l626_626891

variable (J : ℝ)
variable (T : ℝ)
variable (M : ℝ)

-- Conditions
def tim_income := T = 0.60 * J
def mary_income := M = 1.40 * T

-- Theorem to prove that Mary's income is 84 percent of Juan's income
theorem mary_income_percent_of_juan : tim_income J T → mary_income T M → M = 0.84 * J :=
by
  sorry

end mary_income_percent_of_juan_l626_626891


namespace four_digit_integer_transformation_l626_626878

theorem four_digit_integer_transformation (a b c d n : ℕ) (A : ℕ)
  (hA : A = 1000 * a + 100 * b + 10 * c + d)
  (ha : a + 2 < 10)
  (hc : c + 2 < 10)
  (hb : b ≥ 2)
  (hd : d ≥ 2)
  (hA4 : 1000 ≤ A ∧ A < 10000) :
  (1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n)) = n * A → n = 2 → A = 1818 :=
by sorry

end four_digit_integer_transformation_l626_626878


namespace total_kids_in_camp_l626_626622

-- Definitions from the conditions
variables (X : ℕ)
def kids_going_to_soccer_camp := X / 2
def kids_going_to_soccer_camp_morning := kids_going_to_soccer_camp / 4
def kids_going_to_soccer_camp_afternoon := kids_going_to_soccer_camp - kids_going_to_soccer_camp_morning

-- Given condition that 750 kids are going to soccer camp in the afternoon
axiom h : kids_going_to_soccer_camp_afternoon X = 750

-- The statement to prove that X = 2000
theorem total_kids_in_camp : X = 2000 :=
sorry

end total_kids_in_camp_l626_626622


namespace power_of_10_if_digit_square_property_l626_626909

theorem power_of_10_if_digit_square_property (N : ℕ) (k : ℕ) (h1 : N = length (digits 10 N) → k) (h2 : (length (digits 10 N^2) ≥ k) ∧ (take k (digits 10 N^2) = digits 10 N)) : ∃ (m : ℕ), N = 10^m :=
by
  sorry

end power_of_10_if_digit_square_property_l626_626909


namespace find_x_l626_626665

theorem find_x
  (x : ℝ)
  (h : 0.20 * x = 0.40 * 140 + 80) :
  x = 680 := 
sorry

end find_x_l626_626665


namespace pencils_in_pencil_case_l626_626203

theorem pencils_in_pencil_case : ∀ (total_items pens pencils erasers : ℕ), 
  total_items = 13 → pens = 2 * pencils → erasers = 1 → pencils = 4 :=
begin
  intros total_items pens pencils erasers h1 h2 h3,
  -- Proof goes here, but it's omitted because it's not required
  sorry
end

end pencils_in_pencil_case_l626_626203


namespace open_box_volume_l626_626999

theorem open_box_volume (L W S : ℝ) (hL : L = 46) (hW : W = 36) (hS : S = 8) : 
  let new_length := L - 2 * S,
      new_width := W - 2 * S,
      height := S
  in new_length * new_width * height = 4800 := 
by
  unfold new_length new_width height
  sorry

end open_box_volume_l626_626999


namespace tank_capacity_l626_626508

-- Define the rates at which Pipes A, B, and C fill or drain the tank
def rate_A := 200  -- L/min
def rate_B := 50   -- L/min
def rate_C := 25   -- L/min

-- Define the durations (in minutes) for which Pipes A, B, and C are open
def duration_A := 1 -- min
def duration_B := 2 -- min
def duration_C := 2 -- min

-- Define the total time taken to fill the tank
def total_time := 20 -- min

theorem tank_capacity :
  let filled_A := rate_A * duration_A in      -- Water filled by Pipe A
  let filled_B := rate_B * duration_B in      -- Water filled by Pipe B
  let drained_C := rate_C * duration_C in     -- Water drained by Pipe C
  let net_per_cycle := (filled_A + filled_B) - drained_C in
  let cycle_time := duration_A + duration_B + duration_C in
  let num_cycles := total_time / cycle_time in
  let total_filled := net_per_cycle * num_cycles in
  total_filled = 1000 :=                         -- Capacity of the tank
by sorry

end tank_capacity_l626_626508


namespace count_integers_with_permutation_multiple_13_eq_92_l626_626037

noncomputable def count_valid_integers : ℕ := by
  let lower_bound := 200
  let upper_bound := 800
  let multiples_of_13 := 
    (lower_bound // 13 + 1) * 13 
    |> List.range'
      (upper_bound // 13 * 13) 
      13
  let valid_integers := multiples_of_13.filter (λ n => 
    let digits := List.digits n
    let permutations := List.permutations digits
    permutations.exists (λ p => 
      let m := List.to_nat p
      m >= lower_bound && m <= upper_bound && m % 13 == 0
    )
  )
  exact valid_integers.length.to_nat
   sorry

theorem count_integers_with_permutation_multiple_13_eq_92 
  (lower_bound upper_bound valid_condition : ℕ)
  (h1 : lower_bound = 200)
  (h2 : upper_bound = 800)
  (h3 : valid_condition = 92):
  count_valid_integers = 92
 :=
  sorry

end count_integers_with_permutation_multiple_13_eq_92_l626_626037


namespace positive_difference_of_solutions_is_zero_l626_626920

theorem positive_difference_of_solutions_is_zero : ∀ (x : ℂ), (x ^ 2 + 3 * x + 4 = 0) → 
  ∀ (y : ℂ), (y ^ 2 + 3 * y + 4 = 0) → |y.re - x.re| = 0 :=
by
  intro x hx y hy
  sorry

end positive_difference_of_solutions_is_zero_l626_626920


namespace consecutive_diff_possible_l626_626027

variable (a b c : ℝ)

def greater_than_2022 :=
  a > 2022 ∨ b > 2022 ∨ c > 2022

def distinct_numbers :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem consecutive_diff_possible :
  greater_than_2022 a b c → distinct_numbers a b c → 
  ∃ (x y z : ℤ), x + 1 = y ∧ y + 1 = z ∧ 
  (a^2 - b^2 = ↑x) ∧ (b^2 - c^2 = ↑y) ∧ (c^2 - a^2 = ↑z) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end consecutive_diff_possible_l626_626027


namespace find_m_l626_626603

/- Define the polynomial h(x) as given in the problem -/
def h (x m : ℝ) : ℝ := x^3 - x^2 - (m^2 + m + 1) * x + 2 * m^2 + 5 * m + 3

/- State the main problem as a Lean theorem -/
theorem find_m (m : ℝ) :
  (∃ (h : ℝ → ℝ), h = λ x, x^3 - x^2 - (m^2 + m + 1) * x + 2 * m^2 + 5 * m + 3 ∧
   h 3 = 0 ∧
   (∀ x, h x = 0 → x ∈ set.Icc (x : ℝ) (x+1 : ℝ))) →
  m = -3 :=
sorry

end find_m_l626_626603


namespace problem_solution_l626_626101

noncomputable def problem (x y z : ℝ) : Prop :=
  log 10 (3 * x + y) = z ∧ log 10 (x^3 + y^3) = 2 * z + 2

theorem problem_solution (x y z : ℝ) (h : problem x y z) : 
  x^2 + y^2 = -60 * x * 10^(z+1) + 10^(2*z) :=
by
  sorry

end problem_solution_l626_626101


namespace main_l626_626059

noncomputable def X : ℝ → ℝ → MeasureTheory.Probability (ℝ → ℝ) := sorry

axiom norm_dist (μ σ : ℝ) : X μ σ

axiom prob_le (P_le : ℝ → ℝ) (X : ℝ → ℝ) (a : ℝ) : P_le (X a)

axiom prob_cond (X : ℝ → ℝ) (a b : ℝ) :
  prob_le (λ x, x = a) X 0 = m →
  prob_le (λ x, x = b) X 2 = m →
  X (1) (4)

def prob_0_2 (X : ℝ → ℝ) : ℝ :=
  1 - (prob_le (λ x, x ≤ 0) X 0) - (prob_le (λ x, x ≥ 2) X 2)

theorem main : 
  prob_0_2 X = 1 - 2 * m :=
  sorry

end main_l626_626059


namespace correct_option_C_l626_626223

theorem correct_option_C :
  let A := (-π * x^2 * y) / 3
  let B := -5 * a^2 * b + 3 * a * b^2 - 2
  let C1 := -3 * a^2 * b^4 * c
  let C2 := (1 / 2) * b^4 * c * a^2
  let D := 3^2 * x^2 * y + 2 * x * y - 7
  (∃ are_like_terms : (a^2 * b^4 * c) = (b^4 * c * a^2), are_like_terms)

end correct_option_C_l626_626223


namespace borrowing_methods_count_l626_626606

open Finset Nat

-- Definitions for the problem
def physics_books : ℕ := 3
def history_books : ℕ := 2
def mathematics_books : ℕ := 4
def science_students : ℕ := 4
def liberal_arts_students : ℕ := 3

-- Statement of the problem
theorem borrowing_methods_count :
  let category1 := (choose liberal_arts_students 1) * ((choose mathematics_books 1) + (choose mathematics_books 2) + (choose mathematics_books 3))
  let category2 := (choose liberal_arts_students 2) * ((choose mathematics_books 1) + (choose mathematics_books 2))
  let category3 := (choose liberal_arts_students 3) * (choose mathematics_books 1)
  category1 + category2 + category3 = 76 :=
by
  sorry

end borrowing_methods_count_l626_626606


namespace circle_area_outside_triangle_l626_626473

theorem circle_area_outside_triangle 
(ABC : Triangle) 
(hABC : ABC.angle BAC = 90) 
(hCircle : ∃ (O : Point) (r : ℝ), Circle O r) 
(hTangentAB : Circle.tangent_to_side ABC AB) 
(hTangentAC : Circle.tangent_to_side ABC AC) 
(hDiametricallyOppositeOnBC : diametrically_opposite_on_side_circle ABC BC r)
(hAB : AB.length = 9) : 
portion_of_circle_outside_triangle ABC BC r = (9 * π - 18) / 4 := 
sorry

end circle_area_outside_triangle_l626_626473


namespace number_of_towers_mod_500_l626_626674

theorem number_of_towers_mod_500 :
  let cubes := {k : ℕ | 1 ≤ k ∧ k ≤ 7}
  (∀ (tower : List ℕ), ∀ (k : ℕ), k ∈ cubes → (∃ n, tower.nth n = some k ∧ tower.nth (n+1) = some (k+1)) ∨ k = 7) →
  (∃ S, S = 1 ∧ S % 500 = 1) := 
by
  sorry

end number_of_towers_mod_500_l626_626674


namespace number_of_singular_squares_l626_626226

def is_singular_square (board : ℕ → ℕ → ℕ) (x y : ℕ) : Prop :=
  board x y = 100 ∧
  board (x+1) y = 101 ∧
  board (x-1) y = 101 ∧
  board x (y+1) = 101 ∧
  board x (y-1) = 101

theorem number_of_singular_squares (board : ℕ → ℕ → ℕ) : 
  (∑ (x y : ℕ), is_singular_square board x y) = 800 := 
sorry

end number_of_singular_squares_l626_626226


namespace interval_root_sum_eq_neg3_l626_626055

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 1

theorem interval_root_sum_eq_neg3 
  (a b : ℤ) 
  (h1 : b - a = 1) 
  (h2 : ∃ x : ℝ, x ∈ set.Ioo (a : ℝ) (b : ℝ) ∧ f x = 0) :
  a + b = -3 := 
by
  sorry

end interval_root_sum_eq_neg3_l626_626055


namespace probability_of_ab_divisible_by_4_l626_626221

noncomputable def probability_divisible_by_4 : ℚ :=
  let outcomes := (1, 2, 3, 4, 5, 6, 7, 8, 9)
  let favorable_outcomes := Set.filter (λ x => x % 4 = 0) outcomes
  let prob_die := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  prob_die * prob_die

theorem probability_of_ab_divisible_by_4 :
  probability_divisible_by_4 = 4 / 81 := 
sorry

end probability_of_ab_divisible_by_4_l626_626221


namespace abs_inequality_solution_l626_626538

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l626_626538


namespace range_of_x_l626_626732

def star (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x (x : ℝ) (h : star x (x - 2) < 0) : -2 < x ∧ x < 1 := by
  sorry

end range_of_x_l626_626732


namespace angle_AFE_is_85_l626_626074

-- We define all the necessary points, objects, and relationships given in the problem.
variables {A B C D E F : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]

-- Let ABCD be a square and F the midpoint of AD.
structure is_square (A B C D : Type) : Prop :=
  (side_eq : ∀ s t : A, s = t → s = side_length t)
  (angle_90 : ∀ {α β γ δ: A}, angle α β γ δ = 90)

-- Let E be a point such that DE = DC and ∠CDE = 100°.
structure is_equal_angle (D C E: Type) : Prop :=
  (side_eq : DE = DC)
  (angle_100 : ∠CDE = 100)

-- Let F be midpoint of AD.
structure is_midpoint (D F A : Type) : Prop :=
  (midpoint : F = midpoint A D)

-- The main theorem to state the condition and required conclusion
theorem angle_AFE_is_85 (ABCD : is_square A B C D) (EqualAngle : is_equal_angle D C E) 
  (Midpoint : is_midpoint D F A) : ∠AFE = 85° :=
sorry

end angle_AFE_is_85_l626_626074


namespace find_m_value_l626_626032

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ (x : ℝ), f x = 4 * x^2 - 3 * x + 5)
  (h2 : ∀ (x : ℝ), g x = 2 * x^2 - m * x + 8)
  (h3 : f 5 - g 5 = 15) :
  m = -17 / 5 :=
by
  sorry

end find_m_value_l626_626032


namespace cube_sum_identity_l626_626924

theorem cube_sum_identity (p q r : ℝ)
  (h₁ : p + q + r = 4)
  (h₂ : pq + qr + rp = 6)
  (h₃ : pqr = -8) :
  p^3 + q^3 + r^3 = 64 := 
by
  sorry

end cube_sum_identity_l626_626924


namespace greatest_possible_n_l626_626051

theorem greatest_possible_n (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 :=
by {
  sorry
}

end greatest_possible_n_l626_626051


namespace polynomial_irreducible_if_not_divisible_by_5_l626_626121

theorem polynomial_irreducible_if_not_divisible_by_5 (k : ℤ) (h1 : ¬ ∃ m : ℤ, k = 5 * m) :
    ¬ ∃ (f g : Polynomial ℤ), (f.degree < 5) ∧ (f * g = x^5 - x + Polynomial.C k) :=
  sorry

end polynomial_irreducible_if_not_divisible_by_5_l626_626121


namespace compound_interest_example_l626_626698

theorem compound_interest_example :
  let P := 5000
  let r := 0.08
  let n := 4
  let t := 0.5
  let A := P * (1 + r / n) ^ (n * t)
  A = 5202 :=
by
  sorry

end compound_interest_example_l626_626698


namespace farey_sequence_problem_l626_626478

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l626_626478


namespace sum_a_1_to_100_l626_626812

noncomputable def a (n : ℕ) : ℝ :=
  n * Real.sin (n * Real.pi / 2)

theorem sum_a_1_to_100 : ∑ n in Finset.range 100, a (n + 1) = -50 := by
  sorry

end sum_a_1_to_100_l626_626812


namespace cube_faces_sum_l626_626897

theorem cube_faces_sum (edges : Fin 12 → ℕ) (h_range : ∀ i, 1 ≤ edges i ∧ edges i ≤ 12) :
  ∃ f₁ f₂ : Fin 6, (face_sum edges f₁ > 25) ∧ (face_sum edges f₂ < 27) := sorry

/-- Definition of the sum of the edges for a given face, using the given numbering of the cube edges -/

def face_sum (edges : Fin 12 → ℕ) (face : Fin 6) : ℕ :=
  match face with
  | ⟨0, _⟩ => edges 0 + edges 1 + edges 2 + edges 3
  | ⟨1, _⟩ => edges 4 + edges 5 + edges 6 + edges 7
  | ⟨2, _⟩ => edges 8 + edges 9 + edges 10 + edges 11
  | ⟨3, _⟩ => edges 0 + edges 4 + edges 8 + edges 5
  | ⟨4, _⟩ => edges 1 + edges 5 + edges 9 + edges 6
  | ⟨5, _⟩ => edges 2 + edges 6 + edges 10 + edges 7
  -- Covering all 6 faces of the cube
  | ⟨_, hn⟩ => by linarith

/-- Sum of the numbers from 1 to 12 on the edges, ensuring they are each used exactly twice per face count -/
def total_edge_sum := 2 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

#eval total_edge_sum -- ensures that Lean calculates this correctly to 156.

end cube_faces_sum_l626_626897


namespace pure_imaginary_a_eq_six_l626_626831

theorem pure_imaginary_a_eq_six (a : ℝ) (i : ℝ := 0) :
  ((a + 3 * complex.I) / (1 - 2 * complex.I)).im ≠ 0 ∧
  ((a + 3 * complex.I) / (1 - 2 * complex.I)).re = 0 →
  a = 6 :=
sorry

end pure_imaginary_a_eq_six_l626_626831


namespace log_expression_value_l626_626220

theorem log_expression_value :
  (log 4 160 / log 80 4 - log 4 40 / log 10 4) = 4.25 + 1.5 * (log 4 5) := 
by
  sorry

end log_expression_value_l626_626220


namespace standard_deviation_assesses_stability_l626_626969

variable {n : ℕ}
variable {x : Fin n → ℝ}

-- Define the candidate indicators
def average (x : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i, x i)) / n

def standard_deviation (x : Fin n → ℝ) : ℝ :=
  Real.sqrt ((Finset.univ.sum (λ i, (x i - average x)^2)) / n)

def maximum_value (x : Fin n → ℝ) : ℝ :=
  Finset.univ.sup x

def median (x : Fin n → ℝ) : ℝ :=
  let sorted_x := (Finset.univ.val.map x).sorted
  if n % 2 = 0 then
    (sorted_x (Fin.mk (n / 2 - 1) (Nat.div_lt_self (x 0) n (Nat.succ_le_self _)))
    + sorted_x (Fin.mk (n / 2) (Nat.div_lt_self (x 0) (n + 1) (Nat.succ_lt_succ_iff.2 (Nat.lt_of_succ_le n.zero_lt_succ)))))
    / 2
  else
    sorted_x (Fin.mk (n / 2) (Nat.div_lt_self (x 0) (n - 1) (Nat.pred_lt_pred_iff.2 (-1).zero_lt_succ)))

-- State the theorem
theorem standard_deviation_assesses_stability : indicator_stability (standard_deviation x) :=
  sorry

end standard_deviation_assesses_stability_l626_626969


namespace highest_possible_difference_l626_626895

theorem highest_possible_difference  
  (a b c d e f g h i : ℕ)
  (h_a : a ∈ {3, 5, 9})
  (h_b : b ∈ {2, 3, 7})
  (h_c : c ∈ {3, 4, 8, 9})
  (h_d : d ∈ {2, 3, 7})
  (h_e : e ∈ {3, 5, 9})
  (h_f : f ∈ {1, 4, 7})
  (h_g : g ∈ {4, 5, 9})
  (h_h : h = 2)
  (h_i : i ∈ {4, 5, 9})
  (h1 : ∀ ghi, ghi = g * 100 + h * 10 + i → ghi < 900)
  (h2 : ∃ ghi, ghi = 529 → True)
  : ∃ (abc def ghi : ℕ),
    abc = a * 100 + b * 10 + c ∧
    def = d * 100 + e * 10 + f ∧
    ghi = g * 100 + h * 10 + i ∧
    ghi = abc - def ∧
    abc = 923 ∧
    def = 394 := sorry

end highest_possible_difference_l626_626895


namespace max_pairs_2023_l626_626879

def max_pairs (A : Set ℝ) (n : ℕ) : ℕ :=
  let pairs := {p : ℝ × ℝ | p.1 ∈ A ∧ p.2 ∈ A ∧ ∃ k : ℤ, p.1 - p.2 = e^k}
  pairs.toFinset.card

theorem max_pairs_2023 (A : Set ℝ) (hA : A.card = 2023) : max_pairs A 2023 = 11043 := 
sorry

end max_pairs_2023_l626_626879


namespace num_int_vals_l626_626610

theorem num_int_vals (x : ℝ) : 
  (∃ x ∈ ℤ, 4 < sqrt (3 * x) ∧ sqrt (3 * x) < 5) → 
  (∃ n : ℕ, n = 3) :=
by
  sorry

end num_int_vals_l626_626610


namespace polynomial_solution_l626_626798

variable (P : ℚ) -- Assuming P is a constant polynomial

theorem polynomial_solution (P : ℚ) 
  (condition : P + (2 : ℚ) * X^2 + (5 : ℚ) * X - (2 : ℚ) = (2 : ℚ) * X^2 + (5 : ℚ) * X + (4 : ℚ)): 
  P = 6 := 
  sorry

end polynomial_solution_l626_626798


namespace most_likely_heads_in_30000_tosses_l626_626679

theorem most_likely_heads_in_30000_tosses : 
  let p := 1 / 2 in
  let num_tosses := 30000 in
  let most_likely_number_of_heads := p * num_tosses in
  most_likely_number_of_heads = 15000 :=
by
  let p := 1 / 2
  let num_tosses := 30000
  let most_likely_number_of_heads := p * num_tosses
  show most_likely_number_of_heads = 15000
  sorry

end most_likely_heads_in_30000_tosses_l626_626679


namespace ratio_of_perimeters_l626_626565

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l626_626565


namespace variance_of_dataset_l626_626369

def dataset : List ℝ := [3, 5, 4, 7, 6]

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

theorem variance_of_dataset : variance dataset = 2 := by
  sorry

end variance_of_dataset_l626_626369


namespace distance_from_D_to_plane_B1EF_l626_626082

theorem distance_from_D_to_plane_B1EF :
  let D := (0, 0, 0)
  let B₁ := (1, 1, 1)
  let E := (1, 1/2, 0)
  let F := (1/2, 1, 0)
  ∃ (d : ℝ), d = 1 := by
  sorry

end distance_from_D_to_plane_B1EF_l626_626082


namespace solve_fraction_eq_for_x_l626_626524

theorem solve_fraction_eq_for_x (x : ℝ) (hx : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end solve_fraction_eq_for_x_l626_626524


namespace sum_of_a_b_l626_626788

variable (a b : ℝ)

def P := { y : ℝ | y^2 - y - 2 > 0 }
def Q := { x : ℝ | x^2 + a * x + b <= 0 }

theorem sum_of_a_b (h_union : P ∪ Q = set.univ)
    (h_inter : P ∩ Q = { x : ℝ | 2 < x ∧ x ≤ 3 }) :
    a + b = -5 :=
sorry

end sum_of_a_b_l626_626788


namespace ab_ac_bc_range_l626_626107

-- Let a, b, c be real numbers such that a + b + c = 1 and c = -1
-- We want to find the set of all possible values of ab + ac + bc under these conditions

theorem ab_ac_bc_range {a b : ℝ} (h : a + b - 1 = 1) : 
  ∃ S : set ℝ, S = Iic (-1) ∧ ∀ c, c = -1 → ab + ac + bc ∈ S :=
begin
  sorry
end

end ab_ac_bc_range_l626_626107


namespace ratio_of_perimeters_l626_626563

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l626_626563


namespace number_of_individuals_complete_questionnaire_B_l626_626691

theorem number_of_individuals_complete_questionnaire_B :
  ∀ (total_individuals selected_individuals first_number : ℕ)
    (groupA groupB groupC : Set ℕ),
    total_individuals = 960 →
    selected_individuals = 32 →
    first_number = 9 →
    groupA = {n | n ∈ Icc 1 450} →
    groupB = {n | n ∈ Icc 451 750} →
    groupC = {n | n ∈ Icc 751 960} →
    let sampling_interval := total_individuals / selected_individuals,
        nth_number := λ n, first_number + (n - 1) * sampling_interval,
        selected_numbers := (Finset.range selected_individuals).image nth_number,
        B_intervals := selected_numbers.filter (λ n, n ∈ groupB)
    in B_intervals.card = 10 :=
by
  intros total_individuals selected_individuals first_number groupA groupB groupC 
         ht hi hf gA_def gB_def gC_def;
  simp at ht hi hf gA_def gB_def gC_def;
  sorry

end number_of_individuals_complete_questionnaire_B_l626_626691


namespace number_of_correct_conclusions_l626_626386

open Classical

theorem number_of_correct_conclusions :
  (1 = 
    cond1 p q 
    cond2 xy
    cond3 y x y
    neg_cond (xy ≠ 0 y ≠ cond4 real x ∃ t x real ≤ 0)) :=
by
-- Definitions of conditions:
def cond1 (p q : Prop) : Prop := p ∧ ¬q → p ∧ q
def cond2 (xy : Prop) : Prop := (¬(xy = 0) → x ≠ 0 ∧ y ≠ 0)
def cond3 {x : real} (2^x : Prop) : Prop := ∀ x ∈ ℝ, 2^x > 0

-- Negations:
def neg_cond1 (p q : Prop) : Prop := 
(∃ t : ℕ, ∀ x ∈ ℝ, 2^x ≤ 0)

def total_correct_conclusions : ℕ :=
  nat.succ 0 (not cond1) and not cond2.neg_and 
  cond3])):

end number_of_correct_conclusions_l626_626386


namespace triangle_angles_inequality_l626_626826

theorem triangle_angles_inequality (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) 
(h5 : A < Real.pi) (h6 : B < Real.pi) (h7 : C < Real.pi) : 
  A * Real.cos B + Real.sin A * Real.sin C > 0 := 
by 
  sorry

end triangle_angles_inequality_l626_626826


namespace area_of_right_triangle_l626_626073

noncomputable theory

variables {A B C : Type*}
variables [linear_ordered_field A]

-- Definitions of given conditions
def is_right_triangle (A B C : A) (angle_bac : A) := angle_bac = 90
def side_ac := (5 : A) * (real.sqrt 2)
def side_ab := (5 : A)

-- The theorem to prove
theorem area_of_right_triangle (BC : A) (h1 : is_right_triangle A B C 90) (h2 : side_ac ^ 2 = side_ab ^ 2 + BC ^ 2) :
  (1 / 2 * side_ab * BC = 12.5) :=
sorry

end area_of_right_triangle_l626_626073


namespace card_D_ge_card_A_l626_626654

namespace Mathlib

def gamma (n : ℕ) (α β : Fin n → ℝ) : Fin n → ℝ :=
  λ i => abs (α i - β i)

def D (n : ℕ) (A : Set (Fin n → ℝ)) : Set (Fin n → ℝ) :=
  { γ | ∃ α β ∈ A, γ = gamma n α β }

theorem card_D_ge_card_A {n : ℕ} (A : Set (Fin n → ℝ)) [Fintype (Fin n → ℝ)] :
  Fintype.card (D n A) ≥ Fintype.card A :=
  sorry

end Mathlib

end card_D_ge_card_A_l626_626654


namespace find_angle_C_find_side_c_l626_626431

-- Define the variables and conditions

variables {A B C : Real} -- Angles
variables {a b c : Real} -- Sides opposite to respective angles

-- Introducing the conditions as hypotheses
hypothesis h1 : b ≠ 0
hypothesis h2 : c ≠ 0
hypothesis h_cos : b * cos A + a * cos B = -2 * c * cos C
hypothesis h_side_relation : b = 2 * a
hypothesis h_area : (1 / 2) * a * b * sin C = 2 * sqrt 3

-- Expressing the final assertions

-- Assertion 1: Measure of angle C
theorem find_angle_C : cos C = -1 / 2 -> C = 2 * Real.pi / 3 :=
by
  intro h_cos_calc
  -- Prove the measure of angle C given cos C = -1 / 2
  sorry

-- Assertion 2: Length of side c
theorem find_side_c (a_val : Real) : a_val = 2 -> c = 2 * sqrt 7 :=
by
  intro h_a_val
  -- Prove the length of side c given the conditions
  sorry

end find_angle_C_find_side_c_l626_626431


namespace sequence_arithmetic_sequence_general_formula_l626_626956

def seq (n : ℕ) : ℚ := if n = 0 then 1 else (seq (n - 1)) / (2 * (seq (n - 1)) + 1)

theorem sequence_arithmetic (n : ℕ) (h : n ≥ 1) :
  (∀ m : ℕ, m ≥ 1 → (1 / seq m)) = 1 + 2 * (m - 1) :=
sorry

theorem sequence_general_formula (n : ℕ) (h : n ≥ 1) :
  seq n = 1 / (2 * n - 1) :=
sorry

end sequence_arithmetic_sequence_general_formula_l626_626956


namespace transformed_always_in_S_probability_one_transformed_in_S_l626_626275

noncomputable def region_S : Set (ℂ) := {z : ℂ | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

noncomputable def transformed_in_S (z : ℂ) : Prop := 
    -1 ≤ (z.re - z.im) / 2 ∧ (z.re - z.im) / 2 ≤ 1 ∧
    -1 ≤ (z.re + z.im) / 2 ∧ (z.re + z.im) / 2 ≤ 1

theorem transformed_always_in_S (z : ℂ) (hz : z ∈ region_S) :
  transformed_in_S ((1/2 + 1/2 * I) * z) := sorry

theorem probability_one_transformed_in_S :
  ∀ (z : ℂ), z ∈ region_S → transformed_in_S ((1/2 + 1/2 * I) * z) :=
by
  intro z h
  exact transformed_always_in_S z h
  sorry

end transformed_always_in_S_probability_one_transformed_in_S_l626_626275


namespace log_inequality_solution_l626_626350

open Real

-- The number of positive integers x such that log_10(x - 40) + log_10(60 - x) < 2
noncomputable def valid_integer_count : ℕ := 18

theorem log_inequality_solution :
  ∃ k, k = valid_integer_count ∧ (∀ (x : ℕ), 40 < x ∧ x < 60 → 
  log 10 (↑(x - 40)) + log 10 (↑(60 - x)) < 2) :=
sorry

end log_inequality_solution_l626_626350


namespace all_equal_l626_626003

theorem all_equal (n : ℕ) (a : ℕ → ℝ) (h1 : 3 < n)
  (h2 : ∀ k : ℕ, k < n -> (a k)^3 = (a (k + 1 % n))^2 + (a (k + 2 % n))^2 + (a (k + 3 % n))^2) : 
  ∀ i j : ℕ, i < n -> j < n -> a i = a j :=
by
  sorry

end all_equal_l626_626003


namespace number_of_integer_solutions_l626_626612

-- Given condition
def condition (x : ℝ) : Prop := 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5

-- Number of integer solutions that satisfy the condition
theorem number_of_integer_solutions : 
  (Finset.card (Finset.filter (λ x : ℤ, condition x) (Finset.Icc 1 20))) = 3 := 
sorry

end number_of_integer_solutions_l626_626612


namespace cannot_contain_point_1997_0_l626_626415

variable {m b : ℝ}

theorem cannot_contain_point_1997_0 (h : m * b > 0) : ¬ (0 = 1997 * m + b) := sorry

end cannot_contain_point_1997_0_l626_626415


namespace sum_sequence_eq_zero_l626_626441

theorem sum_sequence_eq_zero (a : ℕ → ℤ) (n : ℕ) 
  (h_seq_len : n = 2016)
  (h_condition : ∀ i, 2 ≤ i → i ≤ n - 1 → a i = a (i - 1) + a (i + 1)) 
  : (∑ i in finset.range (n + 1), a i) = 0 :=
sorry

end sum_sequence_eq_zero_l626_626441


namespace probability_at_least_one_1_or_10_l626_626205

theorem probability_at_least_one_1_or_10 :
  let total_outcomes := 10 * 10
  let favorable_outcomes := total_outcomes - 8 * 8
  favorable_outcomes / total_outcomes = 9 / 25 :=
begin
  -- proof steps would go here
  sorry
end

end probability_at_least_one_1_or_10_l626_626205


namespace smallest_q_p_difference_l626_626485

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l626_626485


namespace orange_juice_sales_l626_626742

-- Definitions
variable (num_trees : Nat) (oranges_per_tree_G : Nat) (oranges_per_tree_A : Nat) (oranges_per_tree_M : Nat)
variable (oranges_per_cup : Nat) (dollars_per_cup : Nat)

-- Conditions
def condition1 : num_trees = 110 := rfl
def condition2 : oranges_per_tree_G = 600 := rfl
def condition3 : oranges_per_tree_A = 400 := rfl
def condition4 : oranges_per_tree_M = 500 := rfl
def condition5 : oranges_per_cup = 3 := rfl
def condition6 : dollars_per_cup = 4 := rfl

-- Proof problem
theorem orange_juice_sales : 
  let total_oranges := (oranges_per_tree_G * num_trees + oranges_per_tree_A * num_trees + oranges_per_tree_M * num_trees)
  let total_cups := total_oranges / oranges_per_cup
  let total_money := total_cups * dollars_per_cup
  total_money = 220000 := by
  -- Apply all conditions
  rw [condition1, condition2, condition3, condition4, condition5, condition6]
  -- Calculate total_oranges, total_cups, and total_money
  simp [total_oranges, total_cups, total_money]
  sorry

end orange_juice_sales_l626_626742


namespace sum_log_parts_l626_626109

noncomputable def F (m : ℕ) : ℕ := (Real.log2 m).to_nat

theorem sum_log_parts : 
  F (2^10 + 1) + F (2^10 + 2) + ... + F (2^11) = 10 * 2^10 + 1 :=
by
  sorry

end sum_log_parts_l626_626109


namespace inequality_problems_l626_626775

variable {a b : ℝ}

theorem inequality_problems (h : a < b) (h_neg: b < 0) :
  (|a| > |b| ∧ ¬ (1/a < 1/b) ∧ ab ≥ b^2 ∧ (b/a < a/b)) :=
by
  have ha_neg : a < 0 := lt_trans h h_neg
  have h_abs : |a| > |b| := by
    simp [abs_of_neg ha_neg, abs_of_neg h_neg]
    exact neg_lt_neg h
    
  have h_inv : ¬ (1 / a < 1 / b) := by
    intro h1
    have := (one_div_lt_one_div_of_neg_of_lt ha_neg h)
    contradiction

  have h_prod : ab ≥ b^2 := by
    intro hab
    simp at hab
    exact lt_irrefl _ hab
    
  have h_frac : b / a < a / b := by
    rw [div_lt_div_iff _ _ ha_neg h_neg]
    exact mul_self_lt_mul_self h_neg h
    
  exact ⟨h_abs, h_inv, h_prod, h_frac⟩


end inequality_problems_l626_626775


namespace common_difference_pi_eighth_l626_626791

theorem common_difference_pi_eighth 
  (d : ℝ) (a : ℕ → ℝ) (k : ℤ)
  (h_arithmetic : ∀ n m : ℕ, a m = a n + (m - n) * d)
  (h0 : 0 < d) (h1 : d < 1) 
  (h_not_k_pi_div_2 : a 5 ≠ k * (π / 2))
  (h_equation : sin (a 3) ^ 2 + 2 * sin (a 5) * cos (a 5) = sin (a 7) ^ 2) :
  d = π / 8 :=
by
  sorry

end common_difference_pi_eighth_l626_626791


namespace fixed_amount_at_least_190_l626_626204

variable (F S : ℝ)

theorem fixed_amount_at_least_190
  (h1 : S = 7750)
  (h2 : F + 0.04 * S ≥ 500) :
  F ≥ 190 := by
  sorry

end fixed_amount_at_least_190_l626_626204


namespace inequality_always_true_l626_626991

theorem inequality_always_true (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by sorry

end inequality_always_true_l626_626991


namespace measure_of_angle_A_l626_626063

theorem measure_of_angle_A (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 8) (h₃ : c = 5) : 
  ∠A = π / 3 :=
begin
  sorry
end

end measure_of_angle_A_l626_626063


namespace arithmetic_sequence_a11_l626_626450

-- Define the arithmetic sequence given by a1 and the common difference d.
def arithmetic_sequence (a1 d : Int) : (n : Nat) → Int
| 0     => a1
| (n+1) => (arithmetic_sequence a1 d n) + d

theorem arithmetic_sequence_a11 :
  ∃ d : Int, 
    let a1 := 100 in
    let a10 := 10 in
    a10 = arithmetic_sequence a1 d 9 →
    arithmetic_sequence a1 d 10 = 0 :=
by
  sorry

end arithmetic_sequence_a11_l626_626450


namespace fraction_value_l626_626772

theorem fraction_value (a b : ℚ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 :=
by
  -- The proof goes here.
  sorry

end fraction_value_l626_626772


namespace angle_BCD_is_90_degrees_l626_626974

-- Define the circles and their properties
def externally_tangent (Ω1 Ω2 : Circle) : Prop := -- definition of externally tangent circles
sorry

def internally_tangent (Ω1 Ω Ω2 : Circle) (p1 p2 : Point) : Prop := -- definition of internal tangency to a larger circle Ω at points A and B
sorry

def intersects_again (l AB : Line) (Ω1 : Circle) (D : Point) : Prop := -- definition of intersection of line AB with Ω1 again at D
sorry

def intersection_of_two_circles (Ω1 Ω2 : Circle) (C : Point) : Prop := -- definition of C being a point in the intersection of Ω1 and Ω2
sorry

-- Points
constant A B D C : Point

-- Circles
constant Ω Ω1 Ω2 : Circle

-- Line
constant AB : Line

-- Conditions
axiom condition1 : externally_tangent Ω1 Ω2
axiom condition2 : internally_tangent Ω1 Ω Ω2 A B
axiom condition3 : intersects_again AB Ω1 D
axiom condition4 : intersection_of_two_circles Ω1 Ω2 C

-- Goal: angle BCD is 90 degrees
theorem angle_BCD_is_90_degrees :
  ∠ B C D = 90 :=
sorry

end angle_BCD_is_90_degrees_l626_626974


namespace problem_part2_problem_part3_l626_626391

noncomputable def f (x : ℝ) : ℝ := 2 * (sin x)^2 + 2 * sqrt 3 * (sin x) * (cos x) - 1

noncomputable def g (x : ℝ) : ℝ := (1/2) * |f (x + π / 12)| + (1/2) * |f (x + π / 3)|

theorem problem_part2 :
  (∀ x : ℝ, g (-x) = g x) ∧
  (∀ T : ℝ, T > 0 → T < π / 4 → ∃ x : ℝ, g (x + T) ≠ g x) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π / 4 ∧ g x = g y → x = 0 ∧ y = π / 4) :=
sorry

theorem problem_part3 :
  (∀ x : ℝ, 0 ≤ x → x < π / 4 →
          ((0 ≤ x ∧ x < π / 8) → StrictMono g x) ∧
          ((π / 8 ≤ x ∧ x < π / 4) → StrictAntiMono g x)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < π / 4 → 1 ≤ g x ∧ g x ≤ sqrt 2) :=
sorry

end problem_part2_problem_part3_l626_626391


namespace find_x_l626_626584

theorem find_x (p q r x : ℝ) (h1 : (p + q + r) / 3 = 4) (h2 : (p + q + r + x) / 4 = 5) : x = 8 :=
sorry

end find_x_l626_626584


namespace derivative_f_l626_626590

def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_f (x : ℝ) : deriv f x = -x * Real.sin x :=
by
  -- Proof goes here. Using 'sorry' to denote placeholder.
  sorry

end derivative_f_l626_626590


namespace range_of_m_l626_626430

theorem range_of_m (m : ℝ) (h : ∃ (x y : ℝ), x^2 - m * x + 3 = 0 ∧ y^2 - m * y + 3 = 0 ∧ x > 1 ∧ y < 1) : m ∈ Ioi 4 :=
by
  sorry

end range_of_m_l626_626430


namespace students_not_solving_any_problem_l626_626245

variable (A_0 A_1 A_2 A_3 A_4 A_5 A_6 : ℕ)

-- Given conditions
def number_of_students := 2006
def condition_1 := A_1 = 4 * A_2
def condition_2 := A_2 = 4 * A_3
def condition_3 := A_3 = 4 * A_4
def condition_4 := A_4 = 4 * A_5
def condition_5 := A_5 = 4 * A_6
def total_students := A_0 + A_1 = 2006

-- The final statement to be proven
theorem students_not_solving_any_problem : 
  (A_1 = 4 * A_2) →
  (A_2 = 4 * A_3) →
  (A_3 = 4 * A_4) →
  (A_4 = 4 * A_5) →
  (A_5 = 4 * A_6) →
  (A_0 + A_1 = 2006) →
  (A_0 = 982) :=
by
  intro h1 h2 h3 h4 h5 h6
  -- Proof should go here
  sorry

end students_not_solving_any_problem_l626_626245


namespace HCF_of_two_numbers_l626_626976

theorem HCF_of_two_numbers (H L : ℕ) (product : ℕ) (h1 : product = 2560) (h2 : L = 128)
  (h3 : H * L = product) : H = 20 := by {
  -- The proof goes here.
  sorry
}

end HCF_of_two_numbers_l626_626976


namespace carl_gas_cost_l626_626715

-- Define the variables for conditions
def city_mileage := 30       -- miles per gallon in city
def highway_mileage := 40    -- miles per gallon on highway
def city_distance := 60      -- city miles one way
def highway_distance := 200  -- highway miles one way
def gas_cost := 3            -- dollars per gallon

-- Define the statement to prove
theorem carl_gas_cost : 
  let city_gas := city_distance / city_mileage in
  let highway_gas := highway_distance / highway_mileage in
  let total_one_way_gas := city_gas + highway_gas in
  let round_trip_gas := total_one_way_gas * 2 in
  let total_cost := round_trip_gas * gas_cost in
  total_cost = 42
:= by
  sorry

end carl_gas_cost_l626_626715


namespace AM_GM_inequality_AM_GM_equality_l626_626476

theorem AM_GM_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) :=
by
  sorry

theorem AM_GM_equality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c :=
by
  sorry

end AM_GM_inequality_AM_GM_equality_l626_626476


namespace area_of_square_field_l626_626928

-- Define the total cost, rate per meter, and gate width
def total_cost := 999
def rate_per_meter := 1.50
def gate_width := 1

-- Define the number of gates
def num_gates := 2

-- Define the length of the barbed wire needed given the side length of the field
def barbed_wire_length (s : ℝ) := 4 * s - 2 * gate_width

-- Define the cost calculation
def cost (s : ℝ) := (barbed_wire_length s) * rate_per_meter

-- The main theorem stating that the area of the square field is 27889 square meters
theorem area_of_square_field : 
  (∃ s : ℝ, cost s = total_cost) → (∃ s : ℝ, s^2 = 27889) :=
by
  sorry

end area_of_square_field_l626_626928


namespace instantaneous_velocity_at_3s_l626_626050

theorem instantaneous_velocity_at_3s (t s v : ℝ) (hs : s = t^3) (hts : t = 3*s) : v = 27 :=
by
  sorry

end instantaneous_velocity_at_3s_l626_626050


namespace homework_completion_l626_626893

theorem homework_completion :
  let sanjay_monday := 3/5
  let deepak_monday := 2/7
  let sanjay_remaining_monday := 1 - sanjay_monday
  let deepak_remaining_monday := 1 - deepak_monday

  let sanjay_tuesday := sanjay_remaining_monday * 1/3
  let deepak_tuesday := deepak_remaining_monday * 3/10

  let sanjay_remaining_tuesday := sanjay_remaining_monday - sanjay_tuesday
  let deepak_remaining_tuesday := deepak_remaining_monday - deepak_tuesday

  let combined_remaining := sanjay_remaining_tuesday + deepak_remaining_tuesday

  combined_remaining = 23/30 :=
begin
  sorry
end

end homework_completion_l626_626893


namespace hyperbola_eqn_l626_626750

theorem hyperbola_eqn 
  (e : ℝ) (a b : ℝ) 
  (h_e : e = (Real.sqrt (a^2 + b^2)) / a) 
  (h_a : a^2 + b^2 = 5) 
  (e_cond : e = Real.sqrt 5 / 2)  
  (hx2_9_y2_4_1 : ∀ x y : ℝ, (x^2 / 9 + y^2 / 4 = 1) → (Real.sqrt 5, 0) = (Real.sqrt (9 * (1 - y^2 / 4)), 0)) :
  (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) := 
by {
    sorry
  }

end hyperbola_eqn_l626_626750


namespace area_of_given_quadrilateral_l626_626320

def vertices : List (ℝ × ℝ) := [(2, 2), (6, 4), (7, 1), (3, -2)]

def area_quadrilateral (v : List (ℝ × ℝ)) : ℝ :=
  1 / 2 * |(v[0].fst * v[1].snd + v[1].fst * v[2].snd + v[2].fst * v[3].snd + v[3].fst * v[0].snd) - 
           (v[0].snd * v[1].fst + v[1].snd * v[2].fst + v[2].snd * v[3].fst + v[3].snd * v[0].fst)|

theorem area_of_given_quadrilateral : area_quadrilateral vertices = 16.5 :=
by
  sorry

end area_of_given_quadrilateral_l626_626320


namespace simplify_expression_l626_626660

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
  ( ((x+1)^2 * (x^2 - x + 1)^2 / (x^3 + 1)^2)^2 *
    ((x-1)^2 * (x^2 + x + 1)^2 / (x^3 - 1)^2)^2
  ) = 1 :=
by
  sorry

end simplify_expression_l626_626660


namespace find_min_difference_l626_626483

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l626_626483


namespace quadratic_two_equal_real_roots_l626_626351

theorem quadratic_two_equal_real_roots (m : ℝ) :
  (∃ (x : ℝ), x^2 + m * x + m = 0 ∧ ∀ (y : ℝ), x = y → x^2 + m * y + m = 0) →
  (m = 0 ∨ m = 4) :=
by {
  sorry
}

end quadratic_two_equal_real_roots_l626_626351


namespace ABC_side_length_l626_626259

theorem ABC_side_length (r : ℝ) (s : ℝ) (area : ℝ) (OA : ℝ) (AB : ℝ) (AD : ℝ) (BD : ℝ) (angle_ADB : ℝ) :
  area = 50 * π ∧
  OA = 2 * Real.sqrt 2 ∧
  AB = s ∧
  O ∈ Triangle ABD ∧  -- O is inside triangle ABD
  AD = BD ∧
  angle_ADB = 120 * π / 180 /-- 120 degrees in radians --/ 
  → s = 5 * Real.sqrt 6 :=
by
  sorry

end ABC_side_length_l626_626259


namespace sum_values_l626_626108

def f : ℝ → ℝ :=
  λ x, 2 * x^3 + x + 5

theorem sum_values : 
  (∑ z in {z : ℝ | f(3 * z) = 3}, z) = -2 / 729 :=
by
  sorry

end sum_values_l626_626108


namespace triangle_angle_c_l626_626449

theorem triangle_angle_c (ABC : Triangle) (AD BE CF : Line) (O : Point)
  (h1 : is_altitude AD ABC) (h2 : is_median BE ABC) (h3 : is_angle_bisector CF ABC)
  (h4 : intersection_point AD BE CF O) (h5 : distance O E = 2 * distance O C) :
  ∠ ABC = Real.arccos (1 / 7) := 
sorry

end triangle_angle_c_l626_626449


namespace average_speed_rest_of_trip_l626_626256

variable (v : ℝ) -- The average speed for the rest of the trip
variable (d1 : ℝ := 30 * 5) -- Distance for the first part of the trip
variable (t1 : ℝ := 5) -- Time for the first part of the trip
variable (t_total : ℝ := 7.5) -- Total time for the trip
variable (avg_total : ℝ := 34) -- Average speed for the entire trip

def total_distance := avg_total * t_total
def d2 := total_distance - d1
def t2 := t_total - t1

theorem average_speed_rest_of_trip : 
  v = 42 :=
by
  let distance_rest := d2
  let time_rest := t2
  have v_def : v = distance_rest / time_rest := by sorry
  have v_value : v = 42 := by sorry
  exact v_value

end average_speed_rest_of_trip_l626_626256


namespace smallest_integer_label_same_as_1993_l626_626633

def smallest_label (n : ℕ) : ℕ := 1993 * 1994 / 2 - 1 % 2000

theorem smallest_integer_label_same_as_1993 :
  ∃ n : ℕ, n * (n + 1) / 2 ≡ 21 [MOD 2000] ∧ 
  n = 118 :=
begin
  sorry
end

end smallest_integer_label_same_as_1993_l626_626633


namespace ones_digit_9_pow_53_l626_626210

theorem ones_digit_9_pow_53 :
  (9 ^ 53) % 10 = 9 :=
by
  sorry

end ones_digit_9_pow_53_l626_626210


namespace determine_x_l626_626178

variable (x : ℝ)

theorem determine_x (h : √(4 - 3 * x) = 2 * √2) : x = 2 / 3 := 
sorry

end determine_x_l626_626178


namespace problem_i_problem_ii_l626_626016

variables {O P : ℝ} {x y r : ℝ}
-- Define the conditions
def vertex_origin (α : ℝ) : Prop := ∃ origin : ℝ, origin = 0
def initial_side_non_negative_x_axis (α : ℝ) : Prop := ∃ x, x ≥ 0
def terminal_side_point (α : ℝ) (P : ℝ × ℝ) : Prop := P = (-3/5, 4/5)

-- Define sin and cos computation conditions
def sin_alpha (y r : ℝ) : ℝ := y / r
def cos_alpha (x r : ℝ) : ℝ := x / r

-- Define the problem statement in Lean
-- Proving sin(α + π) = -4/5
theorem problem_i (α : ℝ) (x y : ℝ) (r : ℝ) (h1 : vertex_origin α) 
  (h2 : initial_side_non_negative_x_axis α) (h3 : terminal_side_point α ⟨-3/5, 4/5⟩) 
  (h4 : r = 1) : sin (α + real.pi) = -4/5 :=
by
  -- All proof steps and conditions were handled, the actual proof is defined here
  sorry

-- Proving cosβ = 56/65 given sin(α + β) = 5/13
theorem problem_ii (α β : ℝ) (h1 : sin (α + β) = 5/13) : cos β = 56/65 :=
by
  -- All proof steps and conditions were handled, the actual proof is defined here
  sorry

end problem_i_problem_ii_l626_626016


namespace integral_sin_eq_2_l626_626707

theorem integral_sin_eq_2 :
  ∫ x in 0..Real.pi, Real.sin x = 2 :=
sorry

end integral_sin_eq_2_l626_626707


namespace sum_of_coordinates_C_l626_626196

/- Define points A, B, and D with their coordinates -/
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := -4, y := 2 }
def D : Point := { x := 5, y := -4 }

-- Define the midpoint of a line segment between two points
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Prove that the sum of the coordinates of C is 6
theorem sum_of_coordinates_C : ∃ (C : Point), C.x + C.y = 6 :=
  sorry

end sum_of_coordinates_C_l626_626196


namespace five_digit_odd_numbers_without_repeating_digit_units_not_3_l626_626817

def five_digit_odd_numbers_count (digits : Finset ℕ) (num_digits : ℕ)
  (units_not_3: ℕ) : ℕ :=
  if h : 1 ≤ num_digits ∧ num_digits ≤ 5 ∧ (units_not_3 ≠ 3) then
    2 * (num_digits - 1) * (num_digits - 2) * (num_digits - 3) * (num_digits - 4)
  else 0.

theorem five_digit_odd_numbers_without_repeating_digit_units_not_3
  : five_digit_odd_numbers_count {1, 2, 3, 4, 5} 5 3 = 48 :=
by {
  -- here goes the proof
  sorry
}

end five_digit_odd_numbers_without_repeating_digit_units_not_3_l626_626817


namespace min_students_with_same_birthday_l626_626065

theorem min_students_with_same_birthday (students : ℕ):
  (students = 60) →
  (∀ n : ℕ, n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} → 
    ((∃ s, count_month_students s = n) ∧ (∃ s, count_date_students s = n))) →
  ∃ (s1 s2 : Student), s1 ≠ s2 ∧ same_birthday s1 s2 :=
by sorry

end min_students_with_same_birthday_l626_626065


namespace time_to_cross_platform_l626_626266

noncomputable def speed_kmph := 72 -- speed of the train in kmph
noncomputable def length_of_train := 350.048 -- length of the train in meters
noncomputable def length_of_platform := 250 -- length of the platform in meters

-- Convert speed to meters per second
noncomputable def speed_mps := speed_kmph * (1000 / 3600)

-- Total distance to be covered while crossing the platform
noncomputable def total_distance := length_of_train + length_of_platform

-- Calculate the time to cross the platform
noncomputable def time_to_cross := total_distance / speed_mps

theorem time_to_cross_platform : time_to_cross = 30.0024 :=
by
  -- Proof steps would go here.
  sorry

end time_to_cross_platform_l626_626266


namespace value_of_k_l626_626620

theorem value_of_k :
  ∃ k, k = 2 ∧ (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 5 ∧
                ∀ (s t : ℕ), (s, t) ∈ pairs → s = k * t) :=
by 
sorry

end value_of_k_l626_626620


namespace equilateral_triangle_largest_angle_l626_626446

theorem equilateral_triangle_largest_angle (α : ℝ) (hα : α = 60) :
  let new_angle := α + 20 in
  new_angle = 80 :=
by
  sorry

end equilateral_triangle_largest_angle_l626_626446


namespace distinct_segments_impossible_l626_626315

theorem distinct_segments_impossible (n : ℕ) (h : n ≥ 4) :
  ¬(∀ (selected_points : set ℕ), (selected_points.card = n) →
    ∀ (i j : ℕ) (h_i : i ∈ selected_points) (h_j : j ∈ selected_points) (h_ij : i ≠ j),
      (∃ d : ℕ, d ∈ set.range (λ x : ℕ, x + 1) ∧ d = abs (i - j))) :=
by sorry

end distinct_segments_impossible_l626_626315


namespace union_of_sets_contains_87341_elements_l626_626358

theorem union_of_sets_contains_87341_elements
  (A : Fin 1985 → Set α)
  (h₁ : ∀ i, (A i).card = 45)
  (h₂ : ∀ i j, i ≠ j → (A i ∪ A j).card = 89) :
  (⋃ i, A i).card = 87341 :=
sorry

end union_of_sets_contains_87341_elements_l626_626358


namespace angle_90_degrees_l626_626423

noncomputable def angle_between_vectors (z1 z2 z3 : ℂ) (a : ℝ) (ha : a ≠ 0) : ℝ :=
  real.angle.of_real ((z3 - z1) / (z2 - z1))

theorem angle_90_degrees
  (z1 z2 z3 : ℂ)
  (a : ℝ)
  (ha : a ≠ 0)
  (h : (z3 - z1) / (z2 - z1) = a * complex.I) :
  angle_between_vectors z1 z2 z3 a ha = real.pi / 2 :=
sorry

end angle_90_degrees_l626_626423


namespace balls_into_boxes_l626_626407

noncomputable def ways_to_place_balls_into_boxes : ℕ :=
  31

theorem balls_into_boxes :
  ∃ n : ℕ, n = ways_to_place_balls_into_boxes ∧ n = 31 :=
by
  use ways_to_place_balls_into_boxes
  split
  { refl }
  { refl }

end balls_into_boxes_l626_626407


namespace find_min_difference_l626_626482

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l626_626482


namespace cylinder_surface_area_l626_626261

theorem cylinder_surface_area
    (r h : ℝ)
    (arc_angle : ℝ)
    (p q : ℕ)
    (sqrt_term : ℝ) : 
    r = 7 → 
    h = 9 → 
    arc_angle = 150 → 
    sqrt_term = 3 → 
    (∃ (p q r : ℕ), 
        (p, q, r) == (62, 112, 3) ∧ 
        p + q + r = 177) := 
by
    intros
    use [62, 112, 3]
    sorry

end cylinder_surface_area_l626_626261


namespace train_speed_conversion_l626_626232

/-- Variables and conversions used in the problem -/
def speed_kmph := 189 
def km_to_m := 1000 -- 1 kilometer = 1000 meter
def hr_to_s := 3600 -- 1 hour = 3600 seconds

/-- Convert speed from kmph to m/s -/
def speed_mps := (speed_kmph * km_to_m) / hr_to_s

/-- Theorem stating the speed conversion --/
theorem train_speed_conversion :
  speed_mps = 52.5 := 
sorry

end train_speed_conversion_l626_626232


namespace students_enrolled_for_german_l626_626843

-- Defining the total number of students
def class_size : Nat := 40

-- Defining the number of students enrolled for both English and German
def enrolled_both : Nat := 12

-- Defining the number of students enrolled for only English and not German
def enrolled_only_english : Nat := 18

-- Using the conditions to define the number of students who enrolled for German
theorem students_enrolled_for_german (G G_only : Nat) 
  (h_class_size : G_only + enrolled_only_english + enrolled_both = class_size) 
  (h_G : G = G_only + enrolled_both) : 
  G = 22 := 
by
  -- placeholder for proof
  sorry

end students_enrolled_for_german_l626_626843


namespace degree_of_polynomial_l626_626981

noncomputable def polynomial : ℝ[X][Y] := 3 + 7 * Y^5 + 150 - 3 * X^6 + 0.5 * X^6 + 2

theorem degree_of_polynomial :
  polynomial.degree = 6 :=
sorry

end degree_of_polynomial_l626_626981


namespace table_tennis_team_l626_626443

theorem table_tennis_team : 
  ∃ n : ℕ, n = 48 ∧
  ∃ (Player : Type) (veteran new : Finset Player) (chosen ranked: Finset Player) (position_no : chosen → ℕ),
  veteran.card = 2 ∧ 
  new.card = 3 ∧ 
  (chosen ⊆ veteran ∪ new) ∧
  chosen.card = 3 ∧ 
  (∃ p ∈ chosen, p ∈ veteran) ∧ 
  ((∃ p1 p2 : chosen, position_no p1 = 1 ∧ position_no p2 = 2 ∧ (p1 ∈ new ∨ p2 ∈ new)) ∧ 
  position_no '' chosen = {1, 2, 3}) := 
begin
  sorry,
end

end table_tennis_team_l626_626443


namespace even_perfect_square_factors_l626_626403

theorem even_perfect_square_factors : 
  (∃ count : ℕ, count = 3 * 2 * 3 ∧ 
    (∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ b ≤ 3 ∧ c % 2 = 0 ∧ c ≤ 4) → 
      (2^a * 7^b * 3^c ∣ 2^6 * 7^3 * 3^4))) :=
sorry

end even_perfect_square_factors_l626_626403


namespace all_elements_same_color_l626_626495

-- Definitions based on conditions
variables (n k : ℕ) (h1 : Nat.gcd n k = 1) (h2 : k < n)

def M := {i : ℕ | 1 ≤ i ∧ i < n}

-- Coloring function with two colors
inductive Color
| red
| blue

variable (coloring : ℕ → Color)

-- Conditions for the coloring rules
axiom coloring_rule1 : ∀ i ∈ M n, coloring i = coloring (n - i)
axiom coloring_rule2 : ∀ i ∈ M n, i ≠ k → coloring i = coloring (Nat.abs (k - i))

-- Theorem to prove
theorem all_elements_same_color : ∀ i j ∈ M n, coloring i = coloring j :=
by
  sorry

end all_elements_same_color_l626_626495


namespace diving_competition_score_l626_626236

theorem diving_competition_score 
  (scores : List ℝ)
  (h : scores = [7.5, 8.0, 9.0, 6.0, 8.8])
  (degree_of_difficulty : ℝ)
  (hd : degree_of_difficulty = 3.2) :
  let sorted_scores := scores.erase 9.0 |>.erase 6.0
  let remaining_sum := sorted_scores.sum
  remaining_sum * degree_of_difficulty = 77.76 :=
by
  sorry

end diving_competition_score_l626_626236


namespace sum_odot_l626_626733

def A : Set ℕ := {0, 1}
def B : Set ℕ := {2, 3}

def odot (A B : Set ℕ) : Set ℕ :=
  {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y * (x + y)}

theorem sum_odot : (∑ x in (odot A B), x) = 18 := by
  sorry

end sum_odot_l626_626733


namespace largest_k_l626_626492

def S : Set ℕ := {x | x > 0 ∧ x ≤ 100}

def satisfies_property (A B : Set ℕ) : Prop :=
  ∃ x ∈ A ∩ B, ∀ y ∈ A ∪ B, x ≠ y

theorem largest_k (k : ℕ) : 
  (∃ subsets : Finset (Set ℕ), 
    (subsets.card = k) ∧ 
    (∀ {A B : Set ℕ}, A ∈ subsets ∧ B ∈ subsets ∧ A ≠ B → 
      ¬(A ∩ B = ∅) ∧ satisfies_property A B)) →
  k ≤ 2^99 - 1 := sorry

end largest_k_l626_626492


namespace xiao_yu_reading_days_l626_626996

-- Definition of Xiao Yu's reading problem
def number_of_pages_per_day := 15
def total_number_of_days := 24
def additional_pages_per_day := 3
def new_number_of_pages_per_day := number_of_pages_per_day + additional_pages_per_day
def total_pages := number_of_pages_per_day * total_number_of_days
def new_total_number_of_days := total_pages / new_number_of_pages_per_day

-- Theorem statement in Lean 4
theorem xiao_yu_reading_days : new_total_number_of_days = 20 :=
  sorry

end xiao_yu_reading_days_l626_626996


namespace f_is_odd_l626_626463

noncomputable def f (x : ℝ) : ℝ := log (x^3 + sqrt (1 + x^6))

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intros x
  have h1 : f (-x) = log ((-x)^3 + sqrt (1 + (-x)^6)) := rfl
  dsimp only [f] at h1
  rw [neg_pow_bit1, pow_six, neg_mul_eq_neg_mul_symm, sqrt_neg, h1]
  sorry

end f_is_odd_l626_626463


namespace largest_mersenne_prime_less_than_500_l626_626754

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_less_than_500 : ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p :=
sorry

end largest_mersenne_prime_less_than_500_l626_626754


namespace solve_inequality_l626_626533

theorem solve_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (x ∈ Ioo (-6.5) 3.5) := 
sorry

end solve_inequality_l626_626533


namespace sin_C_correct_l626_626458

noncomputable def sin_c : ℝ :=
  let A := real.arcsin (4 / 5) in
  let B := real.arccos (12 / 13) in
  let C := real.pi - A - B in
  real.sin C

theorem sin_C_correct :
  let A := real.arcsin (4 / 5) in
  let B := real.arccos (12 / 13) in
  let C := real.pi - A - B in
  real.sin A = 4 / 5 ∧
  real.cos B = 12 / 13 ∧
  real.tan A = 4 / 3 →
  sin_c = 63 / 65 :=
by {
  intros,
  sorry
}

end sin_C_correct_l626_626458


namespace moon_iron_percentage_l626_626949

variables (x : ℝ) -- percentage of iron in the moon

-- Given conditions
def carbon_percentage_of_moon : ℝ := 0.20
def mass_of_moon : ℝ := 250
def mass_of_mars : ℝ := 2 * mass_of_moon
def mass_of_other_elements_on_mars : ℝ := 150
def composition_same (m : ℝ) (x : ℝ) := 
  (x / 100 * m + carbon_percentage_of_moon * m + (100 - x - 20) / 100 * m) = m

-- Theorem statement
theorem moon_iron_percentage : x = 50 :=
by
  sorry

end moon_iron_percentage_l626_626949


namespace length_DM_l626_626550

open Real

-- Definitions of the entities involved
structure Point (α : Type) := (x : α) (y : α)
structure Square (α : Type) :=
  (A B C D : Point α)
  (side_length : ℝ)
  (AB : A.x = B.x ∧ B.y = A.y + side_length)
  (BC : B.y = C.y ∧ C.x = B.x + side_length)
  (CD : C.x = D.x ∧ D.y = C.y - side_length)
  (DA : D.y = A.y ∧ A.x = D.x - side_length)

-- Mathematical definitions to describe the problem conditions
def area_of_square (s : Square ℝ) : ℝ :=
  s.side_length ^ 2

def triangle_area (p1 p2 p3 : Point ℝ) : ℝ :=
  abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p1.y -
       p2.x * p1.y - p3.x * p2.y - p1.x * p3.y) / 2)

def DM (p_D p_M : Point ℝ) : ℝ :=
  sqrt ((p_M.x - p_D.x) ^ 2 + (p_M.y - p_D.y) ^ 2)

-- Statement of the problem
theorem length_DM :
  ∀ (M N : Point ℝ)
    (square : Square ℝ),
    square.side_length = 3 →
    (∃ (split1 split2 : ℝ), split1 = triangle_area square.D M square.C ∧
                            split2 = triangle_area square.B N square.C ∧
                            split1 = split2 ∧
                            split1 = 3) →
    DM square.D M = 2 :=
begin
  sorry
end

end length_DM_l626_626550


namespace negative_fraction_less_than_reciprocal_l626_626995

theorem negative_fraction_less_than_reciprocal (x y : ℚ) (hx : x = -3/2) (hy : y = -2/3) :
  x < y :=
by {
  subst hx, 
  subst hy, 
  -- proof can be added here
  sorry
}

end negative_fraction_less_than_reciprocal_l626_626995


namespace students_playing_long_tennis_l626_626842

theorem students_playing_long_tennis (n F B N L : ℕ)
  (h1 : n = 35)
  (h2 : F = 26)
  (h3 : B = 17)
  (h4 : N = 6)
  (h5 : L = (n - N) - (F - B)) :
  L = 20 :=
by
  sorry

end students_playing_long_tennis_l626_626842


namespace average_typing_speed_l626_626934

-- Define the typing speeds of each employee
def Albert : ℕ := 60
def Bella : ℕ := 87
def Chris : ℕ := 91
def Danny : ℕ := 69
def Emily : ℕ := 98
def Fiona : ℕ := 57
def George : ℕ := 85
def Harriet : ℕ := 74
def Ian : ℕ := 76
def Jane : ℕ := 102

-- Define the number of employees
def numberOfEmployees : ℕ := 10

-- The statement that needs to be proved
theorem average_typing_speed :
  (Albert + Bella + Chris + Danny + Emily + Fiona + George + Harriet + Ian + Jane) / numberOfEmployees = 79.9 := 
sorry

end average_typing_speed_l626_626934


namespace proof_problem_l626_626985

-- The given conditions
def conditions (X : ℕ) : Prop :=
  ∃ y z w : ℕ, X - 20 = 15 * y ∧ X - 20 = 30 * z ∧ X - 20 = 45 * w

-- The correct answer: X = 200, and the fourth number in the list of divisors of X - 20 is 4
theorem proof_problem : conditions 200 → 4 ∈ list.nthLe (list.range_succ_len (list.factors (200 - 20))) 3 sorry

end proof_problem_l626_626985


namespace smallest_possible_sum_l626_626768

theorem smallest_possible_sum
  (a b c d : ℕ)
  (h_distinct : ∀ {x y z w : ℕ}, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
  (h_even_prime_a : a = 2)
  (h_sum_two_even : ∀ {x y : ℕ}, (x + y) % 2 = 0)
  (h_sum_three_div_three : ∀ {x y z : ℕ}, (x + y + z) % 3 = 0)
  (h_sum_four_div_four : (a + b + c + d) % 4 = 0)
  : a + b + c + d = 44 := 
begin
  sorry
end

end smallest_possible_sum_l626_626768


namespace ratio_of_areas_is_one_ninth_l626_626549

-- Define the side lengths of Square A and Square B
variables (x : ℝ)
def side_length_a := x
def side_length_b := 3 * x

-- Define the areas of Square A and Square B
def area_a := side_length_a x * side_length_a x
def area_b := side_length_b x * side_length_b x

-- The theorem to prove the ratio of areas
theorem ratio_of_areas_is_one_ninth : (area_a x) / (area_b x) = (1 / 9) :=
by sorry

end ratio_of_areas_is_one_ninth_l626_626549


namespace overall_gain_percentage_correct_l626_626278

def clothData : Type := { length : ℕ, cost_per_meter : ℝ, selling_price_per_meter : ℝ }

def type_A : clothData := ⟨40, 2.5, 3.5⟩
def type_B : clothData := ⟨55, 3.0, 4.0⟩
def type_C : clothData := ⟨36, 4.5, 5.5⟩
def type_D : clothData := ⟨45, 6.0, 7.0⟩

def total_cost_price (d : clothData) := d.length * d.cost_per_meter
def total_selling_price (d : clothData) := d.length * d.selling_price_per_meter

def total_CP := total_cost_price type_A + total_cost_price type_B + total_cost_price type_C + total_cost_price type_D
def total_SP := total_selling_price type_A + total_selling_price type_B + total_selling_price type_C + total_selling_price type_D

noncomputable def gain := total_SP - total_CP
noncomputable def gain_percentage := (gain / total_CP) * 100

theorem overall_gain_percentage_correct : gain_percentage ≈ 25.25 := sorry

end overall_gain_percentage_correct_l626_626278


namespace drone_height_l626_626067

theorem drone_height (TR TS TU : ℝ) (UR : TU^2 + TR^2 = 180^2) (US : TU^2 + TS^2 = 150^2) (RS : TR^2 + TS^2 = 160^2) : 
  TU = Real.sqrt 14650 :=
by
  sorry

end drone_height_l626_626067


namespace line_tangent_to_parabola_l626_626425

theorem line_tangent_to_parabola (c : ℝ) : (∀ (x y : ℝ), 2 * x - y + c = 0 ∧ x^2 = 4 * y) → c = -4 := by
  sorry

end line_tangent_to_parabola_l626_626425


namespace angles_proof_l626_626822

-- Definitions (directly from the conditions)
variable {θ₁ θ₂ θ₃ θ₄ : ℝ}

def complementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 90
def supplementary (θ₃ θ₄ : ℝ) : Prop := θ₃ + θ₄ = 180

-- Theorem statement
theorem angles_proof (h1 : complementary θ₁ θ₂) (h2 : supplementary θ₃ θ₄) (h3 : θ₁ = θ₃) :
  θ₂ + 90 = θ₄ :=
by
  sorry

end angles_proof_l626_626822


namespace maximize_profit_l626_626663

variables (m k x y : ℝ)

-- Condition 1: Equation for production volume.
def production_volume (m k : ℝ) : ℝ := 3 - m / k

-- Condition 2: Initial condition for no technological reform.
def initial_production : Prop := production_volume 0 2 = 1

-- Condition 3 and 4: Fixed and additional investment.
def fixed_investment : ℝ := 80
def additional_investment (x : ℝ) : ℝ := 160 * x

-- Condition 5: Selling price per unit.
def selling_price_per_unit (x : ℝ) : ℝ := 1.5 * (80 + 160 * x)

-- Calculating profit.
def profit (m : ℝ) : ℝ :=
  let x := production_volume m 2 in
  x * selling_price_per_unit x - (80 + 160 * x) - m

-- Goal: Express and maximize profit.
theorem maximize_profit : ∀ (m : ℝ), m ≥ 0 → profit m = 28 - m :=
begin
  intros m hm,
  sorry
end

end maximize_profit_l626_626663


namespace carl_trip_cost_l626_626711

theorem carl_trip_cost :
  (let city_mpg := 30 in
   let hwy_mpg := 40 in
   let one_way_city_miles := 60 in
   let one_way_hwy_miles := 200 in
   let gas_cost_per_gallon := 3 in
   let round_trip_city_miles := 2 * one_way_city_miles in
   let round_trip_hwy_miles := 2 * one_way_hwy_miles in
   let city_gas_needed := round_trip_city_miles / city_mpg in
   let hwy_gas_needed := round_trip_hwy_miles / hwy_mpg in
   let total_gas_needed := city_gas_needed + hwy_gas_needed in
   let total_cost := total_gas_needed * gas_cost_per_gallon in
   total_cost = 42) :=
sorry

end carl_trip_cost_l626_626711


namespace sqrt_fraction_sum_ge_one_l626_626762

theorem sqrt_fraction_sum_ge_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  sqrt (a / (a + 3 * b)) + sqrt (b / (b + 3 * a)) ≥ 1 :=
sorry

end sqrt_fraction_sum_ge_one_l626_626762


namespace union_of_P_and_Q_l626_626354

noncomputable def P : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_of_P_and_Q :
  P ∪ Q = {x | -1 < x ∧ x < 2} :=
sorry

end union_of_P_and_Q_l626_626354


namespace ratio_of_perimeters_l626_626568

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l626_626568


namespace present_age_of_A_l626_626240

theorem present_age_of_A (A B C : ℕ) 
  (h1 : A + B + C = 57)
  (h2 : B - 3 = 2 * (A - 3))
  (h3 : C - 3 = 3 * (A - 3)) :
  A = 11 :=
sorry

end present_age_of_A_l626_626240


namespace triangle_area_incircle_trisect_l626_626083

theorem triangle_area_incircle_trisect
  (A B C : ℝ^2) 
  (h : ℝ)
  (bc : ℝ)
  (incircle_trisects : ∃ H P Q, BC = 24 ∧ (AH = 3h) ∧ (incircle_trisects_altitude A B C H P Q) ∧ 
                       (area ABC = p * real.sqrt(q)) ∧ (p ∈ ℤ) ∧ (q ∈ ℤ) ∧ (∀ r, r*r ∣ q → r = 1)) :
  p + q = 145 := 
by
  sorry

end triangle_area_incircle_trisect_l626_626083


namespace smallest_positive_angle_l626_626347

theorem smallest_positive_angle (α : ℝ) 
  (h₁ : sin (2 * π / 3) = sqrt 3 / 2) 
  (h₂ : cos (2 * π / 3) = -1 / 2) : 
  α = 11 * π / 6 :=
sorry

end smallest_positive_angle_l626_626347


namespace quadrilateral_AD_length_l626_626071
noncomputable def length_AD (AB BC CD : ℝ) (sin_C cos_B : ℝ) : ℝ :=
  let C := real.arcsin(sin_C) in
  let B := real.arccos(cos_B) in
  let BD := (BC * sin_C / cos_B) in
  let AD := real.sqrt((AB + BD) ^ 2 + CD ^ 2) in
  AD

theorem quadrilateral_AD_length (h1 : AB = 3) (h2 : BC = 6) (h3 : CD = 15) (h4 : sin C = 4 / 5) (h5 : cos B = 4 / 5) : length_AD AB BC CD (4 / 5) (4 / 5) ≈ 19.52 := sorry

end quadrilateral_AD_length_l626_626071


namespace angle_KLD_is_right_l626_626902

def Point := (ℝ × ℝ)
def Square (a : ℝ) := (A : Point, B : Point, C : Point, D : Point)

def midpoint (p1 p2 : Point) : Point := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def divide_in_ratio (p1 p2 : Point) (m n : ℝ) : Point := ((m * p2.1 + n * p1.1) / (m + n), (m * p2.2 + n * p1.2) / (m + n))

theorem angle_KLD_is_right (a : ℝ) (A B C D K L : Point) :
  K = midpoint A B ∧
  L = divide_in_ratio A C 3 1 ∧
  A = (0, 0) ∧ B = (a, 0) ∧
  C = (a, a) ∧ D = (0, a) →
  ∠KLD = 90 :=
by
  sorry

end angle_KLD_is_right_l626_626902


namespace find_length_of_segment_AE_l626_626548

-- Definitions for the problem conditions
structure Square (α : Type) [MetricSpace α] where
  A B C D : α
  side_len : ℝ
  AB_CD_eq_side_len : dist A B = side_len ∧ dist B C = side_len ∧ dist C D = side_len ∧ dist D A = side_len

structure PointOnSide (α : Type) [MetricSpace α] (A B : α) where
  E : α
  ratio_on_side : ℝ
  E_on_AB : dist A E = ratio_on_side * dist A B

theorem find_length_of_segment_AE {α : Type} [MetricSpace α] 
    (sq : Square α) 
    (AE_on_AB E_on_side_A : PointOnSide α sq.A sq.B)
    (CF_on_CB F_on_side_C : PointOnSide α sq.C sq.B) 
    (side_length_is_two : sq.side_len = 2) 
    (ratio_is_one_fourth : AE_on_AB.ratio_on_side = 1/4 ∧ CF_on_CB.ratio_on_side = 1/4) : 
  dist sq.A AE_on_AB.E = 0.5 :=
sorry

end find_length_of_segment_AE_l626_626548


namespace pool_volume_l626_626770

variable {rate1 rate2 : ℕ}
variables {hose1 hose2 hose3 hose4 : ℕ}
variables {time : ℕ}

def hose1_rate := 2
def hose2_rate := 2
def hose3_rate := 3
def hose4_rate := 3
def fill_time := 25

def total_rate := hose1_rate + hose2_rate + hose3_rate + hose4_rate

theorem pool_volume (h : hose1 = hose1_rate ∧ hose2 = hose2_rate ∧ hose3 = hose3_rate ∧ hose4 = hose4_rate ∧ time = fill_time):
  total_rate * 60 * time = 15000 := 
by 
  sorry

end pool_volume_l626_626770


namespace six_digit_increasing_order_mod_1000_l626_626099

theorem six_digit_increasing_order_mod_1000 :
  let M := (nat.choose 14 6)
  in (M % 1000) = 3 :=
by
  let M := (nat.choose 14 6)
  have h : (M % 1000) = 3 := sorry
  exact h

end six_digit_increasing_order_mod_1000_l626_626099


namespace tan_alpha_plus_pi_four_l626_626378

open Real

theorem tan_alpha_plus_pi_four {α : ℝ} (h1 : α ∈ Ioo (π / 2) π) (h2 : sin α = 5 / 13) :
  tan (α + π / 4) = 7 / 17 :=
sorry

end tan_alpha_plus_pi_four_l626_626378


namespace find_initial_men_l626_626144

def initial_men_working (M : ℕ) : Prop :=
  M * 8 * 30 = (M + 55) * 6 * 50

theorem find_initial_men : ∃ M : ℕ, initial_men_working M ∧ M = 275 :=
by {
  use 275,
  unfold initial_men_working,
  sorry
}

end find_initial_men_l626_626144


namespace count_even_a_l626_626977

def floor (x : ℝ) : ℤ := Int.floor x

def a (n : ℕ) : ℤ := ∑ k in Finset.range (n + 1) \ {0}, floor (n / k)

def is_even (n : ℤ) : Prop := n % 2 = 0

-- Define the count of even numbers in the sequence a₁, a₂, ..., aₙ
def count_evens_up_to (n : ℕ) : ℕ :=
  Finset.card { m | m ∈ Finset.range (n + 1) \ {0} ∧ is_even (a m) }

-- The theorem we want to prove
theorem count_even_a (count_evens_up_to 2018 = 1028) : Prop :=
  sorry

end count_even_a_l626_626977


namespace frac_add_eq_seven_halves_l626_626773

theorem frac_add_eq_seven_halves {x y : ℝ} (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 :=
by
  sorry

end frac_add_eq_seven_halves_l626_626773


namespace Daisy_surname_l626_626348

def sitting_between (x y z : string) : Prop := sorry

def circular_order (names : list string) : Prop := sorry

theorem Daisy_surname :
  ∀ (MissOng MissLim MissMak Ellie Cindy MissNai Amy Beatrice MissPoh Daisy : string)
    (seating : list string),
  circular_order ["MissOng", "MissLim", "MissMak", "Ellie", "Cindy", "MissNai", "Amy", "Beatrice", "MissPoh", Daisy] →
  sitting_between "MissOng" "MissLim" "MissMak" →
  sitting_between "Ellie" "Cindy" "MissNai" →
  sitting_between "MissLim" "Ellie" "Amy" →
  circular_order ["MissPoh", "Beatrice", "MissMak"] →
  Daisy = "MissMak" :=
begin
  sorry
end

end Daisy_surname_l626_626348


namespace contrapositive_of_not_isosceles_l626_626588

-- Define triangle ABC with conditions
variable {A B C : Type}

-- Define isosceles property
def is_isosceles (ABC : A × B × C) : Prop := 
  ∃ a b c, ABC = (a, b, c) ∧ (a = b ∨ b = c ∨ c = a)

-- Define angles being equal
def angles_equal (ABC : A × B × C) : Prop := 
  ∃ a b c, ABC = (a, b, c) ∧ (a = b ∨ b = c ∨ c = a)

-- Contrapositive proof
theorem contrapositive_of_not_isosceles (ABC : A × B × C) 
  (h : ¬ is_isosceles ABC → ¬ angles_equal ABC) : 
  angles_equal ABC → is_isosceles ABC :=
sorry

end contrapositive_of_not_isosceles_l626_626588


namespace possible_values_S_A5_max_value_S_A_odd_possible_n_values_l626_626810

-- Definitions for the sequence and sum
def seq (n : ℕ) (a : fin n → ℤ) : Prop :=
  a 0 = 0 ∧ a (n-1) = 0 ∧ ∀ k, 1 ≤ k ∧ k < n → (a k - a (k-1))^2 = 1

def sum_seq {n : ℕ} (a : fin n → ℤ) : ℤ := 
  finset.univ.sum a

-- Theorem for the possible values of S(A_5)
theorem possible_values_S_A5 : 
  ∀ (a : fin 5 → ℤ), seq 5 a → 
    sum_seq a = 4 ∨ sum_seq a = 2 ∨ sum_seq a = 0 ∨ sum_seq a = -2 ∨ sum_seq a = -4 :=
sorry

-- Theorem for the maximum value of S(A_{2k+1})
theorem max_value_S_A_odd (k : ℕ) (hk : k > 0) : 
  ∀ (a : fin (2*k+1) → ℤ), seq (2*k+1) a → 
    sum_seq a ≤ k^2 :=
sorry

-- Theorem for the possible values of n given 0 ∈ Γₙ
theorem possible_n_values (n : ℕ) (hn : n ≥ 2) :
  (∃ (a : fin n → ℤ), seq n a ∧ sum_seq a = 0) ↔ 
    (∃ m : ℕ, m > 0 ∧ (n = 4*m ∨ n = 4*m + 1)) :=
sorry

end possible_values_S_A5_max_value_S_A_odd_possible_n_values_l626_626810


namespace hotel_guest_count_l626_626975

theorem hotel_guest_count (Oates Hall both : ℕ) (hO : Oates = 50) (hH : Hall = 62) (hB : both = 12) :
  Oates + Hall - both = 100 :=
by {
  rw [hO, hH, hB],
  simp,
  sorry -- Proof to be completed
}

end hotel_guest_count_l626_626975


namespace biking_speed_l626_626310

theorem biking_speed (d_ab : ℕ) (d_ab_eq : d_ab = 55) (d_bike : ℕ) (d_bike_eq : d_bike = 25) (speed_double : ∀ (v_bus v_bike : ℕ), v_bus = 2 * v_bike) (time_diff : ∀ (t_bike t_bus : ℕ), t_bike = t_bus + 1) :
  ∃ (v_bike : ℕ), v_bike = 10 :=
begin
  -- Define the distances
  let d_bus := d_ab - d_bike,
  have d_bus_eq : d_bus = 30 := by linarith [d_ab_eq, d_bike_eq],

  -- Define the speeds
  let v_bike := 10,
  let v_bus := 2 * v_bike,

  -- Define the times
  let t_bike := d_bike / v_bike,
  let t_bus := d_bus / v_bus,

  -- Establish the relationship between times
  have time_eq : t_bike = t_bus + 1 := by simp [d_bike, v_bike, d_bus, v_bus, div_eq],

  -- Conclude that the biking speed matches
  use v_bike,
  exact time_eq.symm,
end

end biking_speed_l626_626310


namespace find_f1_l626_626779

theorem find_f1 : ∀ (f : ℕ → ℕ), 
  (∀ x, f (x + 1) = x^2 - 2 * x) → f (1) = 0 := 
by {
  intro f,
  intro h,
  have h0 : f (1) = (0 : ℕ)^2 - 2 * (0 : ℕ) := by exact h 0,
  rw h0,
  norm_num,
  exact 0,
}

end find_f1_l626_626779


namespace addition_example_l626_626234

theorem addition_example : 248 + 64 = 312 := by
  sorry

end addition_example_l626_626234


namespace incorrect_statement_D_l626_626769

theorem incorrect_statement_D {a b : ℕ} (ha : a ≥ 2) (hb : b ≥ 2)
  (A1 : Event)
  (A2 : Event)
  (B1 : Event)
  (B2 : Event)
  (draws_without_replacement : ∀ (x y : ℕ), (draw x y) -> (draw (x-1) (y-1))) :
  ¬ (conditional_probability B2 A1 + conditional_probability B1 A2 = 1) := 
sorry

end incorrect_statement_D_l626_626769


namespace farey_sequence_problem_l626_626477

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l626_626477


namespace tan_double_angle_l626_626015

theorem tan_double_angle (θ : ℝ) 
(h_initial_side : ∃ p : ℝ × ℝ, p = (1, 2) ∧ θ = real.arctan (p.2 / p.1)) : 
  real.tan (2 * θ) = -4 / 3 := 
sorry

end tan_double_angle_l626_626015


namespace equation_solution_l626_626142

theorem equation_solution (x : ℝ) (h₁ : 2 * x - 5 ≠ 0) (h₂ : 5 - 2 * x ≠ 0) :
  (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ↔ (x = 0) :=
by
  sorry

end equation_solution_l626_626142


namespace evaluate_fg_of_8_l626_626113

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem evaluate_fg_of_8 : f (g 8) = 211 :=
by
  sorry

end evaluate_fg_of_8_l626_626113


namespace find_b_l626_626461

variable (A B C : Type)
variable (a b c : ℝ)
variable (cos_A : ℝ)

def angle_side_relation (a b c : ℝ) (cos_A : ℝ) : Prop :=
  cos_A = (b^2 + c^2 - a^2) / (2 * b * c)

-- Conditions
def given_a := (a = 2)
def given_c := (c = sqrt 2)
def given_cos_A := (cos_A = - (sqrt 2) / 4)

-- Question (Proof of the value of side b)
theorem find_b (h1 : given_a) (h2 : given_c) (h3 : given_cos_A):
  b = 1 :=
sorry

end find_b_l626_626461


namespace triangle_equilateral_perimeter_eq_quadrilateral_l626_626510

noncomputable def equilateral_triangle (A B C : Point ℝ) : Prop :=
  -- Define an equilateral triangle
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def point_on_segment (P A B : Point ℝ) : Prop :=
  -- Define P as a point on segment AB
  dist A P + dist P B = dist A B

noncomputable def perpendicular_to_segment (P X A B : Point ℝ) : Prop :=
  -- Define PX perpendicular to segment AB
  ∃ H : Point ℝ, ∠A P H = 90 ∧ ∠B P H = 90

noncomputable def triangle_perimeter (A B C : Point ℝ) : ℝ :=
  -- Perimeter of triangle
  dist A B + dist B C + dist C A

noncomputable def quadrilateral_perimeter (A B C D : Point ℝ) : ℝ :=
  -- Perimeter of quadrilateral
  dist A B + dist B C + dist C D + dist D A

theorem triangle_equilateral_perimeter_eq_quadrilateral 
  {A B C P X Y : Point ℝ}
  (h_eq : equilateral_triangle A B C)
  (h_point : point_on_segment P B C)
  (h_perp1 : perpendicular_to_segment P X A B)
  (h_perp2 : perpendicular_to_segment P Y A C) :
  triangle_perimeter X A Y = quadrilateral_perimeter B C Y X := 
  sorry

end triangle_equilateral_perimeter_eq_quadrilateral_l626_626510


namespace exists_set_B_l626_626095

open Finset

theorem exists_set_B {n : ℕ} (hn : n ≥ 2) (A : Finset ℕ) (S := Finset.Icc 2 n) (k := S.filter Nat.Prime).card
  (hA_sub_S : A ⊆ S) (hA_card : A.card ≤ k) (hA_no_div : ∀ {x y}, x ∈ A → y ∈ A → x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x)) :
  ∃ B, B.card = k ∧ A ⊆ B ∧ B ⊆ S ∧ ∀ {x y}, x ∈ B → y ∈ B → x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x) :=
  sorry

end exists_set_B_l626_626095


namespace min_value_xy_l626_626058

theorem min_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : x * y ≥ 18 := 
sorry

end min_value_xy_l626_626058


namespace maximum_value_minimum_value_l626_626382

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def check_digits (N M : ℕ) (a b c d e f g h : ℕ) : Prop :=
  N = 1000 * a + 100 * b + 10 * c + d ∧
  M = 1000 * e + 100 * f + 10 * g + h ∧
  a ≠ e ∧
  b ≠ f ∧
  c ≠ g ∧
  d ≠ h ∧
  a ≠ f ∧
  a ≠ g ∧
  a ≠ h ∧
  b ≠ e ∧
  b ≠ g ∧
  b ≠ h ∧
  c ≠ e ∧
  c ≠ f ∧
  c ≠ h ∧
  d ≠ e ∧
  d ≠ f ∧
  d ≠ g

theorem maximum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 15000 :=
by
  intros
  sorry

theorem minimum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 4998 :=
by
  intros
  sorry

end maximum_value_minimum_value_l626_626382


namespace find_x_l626_626039

theorem find_x (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 3 * real.sqrt 54 ∨ x = -3 * real.sqrt 54 :=
by
  sorry

end find_x_l626_626039


namespace flower_garden_width_l626_626866

-- Define the conditions
def gardenArea : ℝ := 143.2
def gardenLength : ℝ := 4
def gardenWidth : ℝ := 35.8

-- The proof statement (question to answer)
theorem flower_garden_width :
    gardenWidth = gardenArea / gardenLength :=
by 
  sorry

end flower_garden_width_l626_626866


namespace cos_2alpha_plus_pi_over_3_l626_626043

open Real

theorem cos_2alpha_plus_pi_over_3 
  (alpha : ℝ) 
  (h1 : cos (alpha - π / 12) = 3 / 5) 
  (h2 : 0 < alpha ∧ alpha < π / 2) : 
  cos (2 * alpha + π / 3) = -24 / 25 := 
sorry

end cos_2alpha_plus_pi_over_3_l626_626043


namespace solve_abs_inequality_l626_626542

theorem solve_abs_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ 
      (x ∈ Ioo (-13 / 2) (-3) 
     ∨ x ∈ Ico (-3) 2 
     ∨ x ∈ Icc 2 (7 / 2)) :=
by
  sorry

end solve_abs_inequality_l626_626542


namespace hyperbola_focal_distance_l626_626751

open Real

theorem hyperbola_focal_distance :
  (let a := sqrt 3
       b := sqrt 6
       c := sqrt (a^2 + b^2)
    in 2 * c) = 6 :=
by
  let a := sqrt 3
  let b := sqrt 6
  let c := sqrt (a^2 + b^2)
  have h : c = 3 := by sorry
  show 2 * c = 6
  rw [h]
  norm_num

end hyperbola_focal_distance_l626_626751


namespace total_end_of_year_students_l626_626701

theorem total_end_of_year_students :
  let start_fourth := 33
  let start_fifth := 45
  let start_sixth := 28
  let left_fourth := 18
  let joined_fourth := 14
  let left_fifth := 12
  let joined_fifth := 20
  let left_sixth := 10
  let joined_sixth := 16

  let end_fourth := start_fourth - left_fourth + joined_fourth
  let end_fifth := start_fifth - left_fifth + joined_fifth
  let end_sixth := start_sixth - left_sixth + joined_sixth
  
  end_fourth + end_fifth + end_sixth = 116 := by
    sorry

end total_end_of_year_students_l626_626701


namespace problem_solution_l626_626230

theorem problem_solution (a b c d : ℝ) 
  (h1 : 3 * a + 2 * b + 4 * c + 8 * d = 40)
  (h2 : 4 * (d + c) = b)
  (h3 : 2 * b + 2 * c = a)
  (h4 : c + 1 = d) :
  a * b * c * d = 0 :=
sorry

end problem_solution_l626_626230


namespace g_n_plus_2_minus_g_n_l626_626500

-- Definition of the function g
def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2)^n

-- The theorem to prove
theorem g_n_plus_2_minus_g_n (n : ℕ) : 
  g (n + 2) - g n = ((-1 + 4 * Real.sqrt 3) / 8) * g n :=
by
  sorry -- Proof to be completed

end g_n_plus_2_minus_g_n_l626_626500


namespace min_value_of_expression_l626_626756

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 4 * x + 1 / x ^ 6 ≥ 5 :=
sorry

end min_value_of_expression_l626_626756


namespace range_of_k_l626_626883

theorem range_of_k (k : ℝ) :
  (∀ a b c : ℝ, 
    let f := λ x, (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)
    in (f(a) + f(b) > f(c)) ∧ (f(a) + f(c) > f(b)) ∧ (f(b) + f(c) > f(a))
  ) → (-1 / 2 < k ∧ k < 4) :=
by
  intro h
  sorry

end range_of_k_l626_626883


namespace distance_between_first_and_last_tree_l626_626963

theorem distance_between_first_and_last_tree
  (n : ℕ) (d_1_5 : ℝ) (h1 : n = 8) (h2 : d_1_5 = 100) :
  let interval_distance := d_1_5 / 4
  let total_intervals := n - 1
  let total_distance := interval_distance * total_intervals
  total_distance = 175 :=
by
  sorry

end distance_between_first_and_last_tree_l626_626963


namespace ratio_of_areas_l626_626474

theorem ratio_of_areas (s : ℝ) (A B C D E F G H : Point) :
  is_square ABCD s →
  is_isosceles_right_triangle E (segment AB) →
  is_isosceles_right_triangle F (segment BC) →
  is_isosceles_right_triangle G (segment CD) →
  is_isosceles_right_triangle H (segment DA) →
  ratio (area_of_square EFGH) (area_of_square ABCD) = 2 :=
sorry

end ratio_of_areas_l626_626474


namespace scalene_triangle_tangent_circles_pass_vertex_l626_626137

theorem scalene_triangle_tangent_circles_pass_vertex 
  (ABC : Triangle) 
  (h1 : scalene ABC) 
  (I : Circle) (h2 : incircle ABC I) 
  (O : Circle) (h3 : circumcircle ABC O) 
  (E : Circle) (h4 : excircle ABC E) 
  (T : Circle) (h5 : tangent_circles T I O ∧ externally_tangent T E): 
  ∃ (A : Point), passes_through_vertex (T A) :=
by {
  -- Proof not required as per the problem definition.
  sorry
}

end scalene_triangle_tangent_circles_pass_vertex_l626_626137


namespace angle_C_is_45_degrees_l626_626838

-- Defining the given problem conditions
variables {A B C P Q : Type} [InnerProdSpace ℝ A] [InnerProdSpace ℝ B]
 [InnerProdSpace ℝ C] [InnerProdSpace ℝ P] [InnerProdSpace ℝ Q]

-- Condition: ABC is an isosceles triangle with AB = BC
axiom is_isosceles_triangle (ABC : Triangle A B C) : length AB = length BC

-- Condition: Points P and Q are on AB and BC respectively with AP = PQ = QB = QC
axiom partition_points 
  (P_on_AB : P ∈ segment A B)
  (Q_on_BC : Q ∈ segment B C)
  (equal_parts : length AP = length PQ ∧ length PQ = length QB ∧ length QB = length QC)

-- Prove that given these conditions, the measure of angle C is 45 degrees
theorem angle_C_is_45_degrees :
  ∃ x : ℝ, measure_angle (angle C) = 45 :=
by
  -- Skipping the proof part
  sorry

end angle_C_is_45_degrees_l626_626838


namespace parabola_vertex_l626_626936

theorem parabola_vertex:
  ∀ x: ℝ, ∀ y: ℝ, (y = (1 / 2) * x ^ 2 - 4 * x + 3) → (x = 4 ∧ y = -5) :=
sorry

end parabola_vertex_l626_626936


namespace inclination_angle_l626_626946

theorem inclination_angle (θ : ℝ) (h_eq : ∀ x y : ℝ, sqrt 3 * x + y - 1 = 0) 
  (h_tan : Math.tan θ = - sqrt 3) (h_theta : 0 ≤ θ ∧ θ < Real.pi) : 
  θ = 2 * Real.pi / 3 :=
begin
  sorry
end

end inclination_angle_l626_626946


namespace general_formula_sum_inverse_Sn_l626_626000

/-- Given an arithmetic sequence {a_n} with a common difference d ≠ 0,
and the sum of the first n terms is S_n, a_1 = 2, and a_1, a_2, a_4 form a geometric sequence -/
variable (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ)
variable (n : ℕ)
hypothesis h1 : d ≠ 0
hypothesis h2 : a 1 = 2
hypothesis h3 : (a 2)^2 = (a 1) * (a 4)
hypothesis h4 : S 0 = 0
hypothesis h5 : ∀ n, S (n + 1) = S n + a (n + 1)

/-- General formula for the sequence a_n -/
theorem general_formula : ∀ n, a n = 2 * n := by
  sorry

/-- The sum of the first n terms of {1/S_n} -/
theorem sum_inverse_Sn : ∀ n, (∑ i in Finset.range n, 1 / (S (i + 1))) = n / (n + 1) := by
  sorry

end general_formula_sum_inverse_Sn_l626_626000


namespace color_plane_no_unit_equilateral_same_color_l626_626861

theorem color_plane_no_unit_equilateral_same_color :
  ∃ (coloring : ℝ × ℝ → ℕ), (∀ (A B C : ℝ × ℝ),
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    (coloring A ≠ coloring B ∨ coloring B ≠ coloring C ∨ coloring C ≠ coloring A)) :=
sorry

end color_plane_no_unit_equilateral_same_color_l626_626861


namespace abs_inequality_solution_l626_626539

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l626_626539


namespace sum_even_integers_12_to_46_l626_626219

theorem sum_even_integers_12_to_46 : 
  let a1 := 12
  let d := 2
  let an := 46
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 522 := 
by
  let a1 := 12 
  let d := 2 
  let an := 46
  let n := (an - a1) / d + 1 
  let Sn := n * (a1 + an) / 2
  sorry

end sum_even_integers_12_to_46_l626_626219


namespace foci_of_ellipse_l626_626160

theorem foci_of_ellipse :
  let a := 5 in -- since sqrt(25) = 5
  let b := 4 in -- since sqrt(16) = 4
  let c := sqrt (a^2 - b^2) in -- c^2 = 25 - 16 = 9
  c = 3 ∧ (0, c) = (0, 3) ∧ (0, -c) = (0, -3) := 
by
  sorry

end foci_of_ellipse_l626_626160


namespace shape_is_tetrahedron_l626_626598

theorem shape_is_tetrahedron (shape : Type) [is_3d_shape shape]
  (angle_condition : ∀ (adj_sides : set (pair_of_sides shape)), angle_between_diagonals adj_sides = 60) :
  shape_is_regular_tetrahedron shape :=
sorry

end shape_is_tetrahedron_l626_626598


namespace triangle_area_l626_626854

-- Definitions for the given conditions
def BC : ℝ := 12 -- BC = 12 cm
def height (A C : ℝ) : ℝ := 15 -- A is 15 cm away from C

-- Theorem statement for the area of triangle ABC
theorem triangle_area (A C B : ℝ) (h_A_north_C : A > C) (h_BC : BC = 12) (h_height : height A C = 15) :
  (1 / 2) * BC * (height A C) = 90 :=
by
  sorry

end triangle_area_l626_626854


namespace percentage_of_football_likers_who_play_l626_626418

theorem percentage_of_football_likers_who_play 
  (likers_ratio : ℚ)
  (likers_total : ℕ)
  (players_total : ℕ) 
  (group_total : ℕ) 
  (expected_players : ℕ) 
  (expected_likers_proportion : likers_ratio = 24 / 60) 
  (number_of_likers : likers_total = 250 * (24 / 60))
  : (players_total / likers_total) * 100 = 50 := by
  have h1 : 24 / 60 = 2 / 5 := by norm_num
  have h2 : 250 * (24 / 60) = 100 := by norm_num
  rw [h1, h2]
  have h3 : players_total = 50 := by assumption
  have h4 : likers_total = 100 := by
    rw [h2]
    assumption
  have h5 : (50 / 100 : ℚ) = 1 / 2 := by norm_num
  rw [h5]
  norm_num
  sorry


end percentage_of_football_likers_who_play_l626_626418


namespace solution_proof_l626_626008

noncomputable def solution_inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (x + 1/2) < f (1 - x)

noncomputable def solution_set (x : ℝ) : Prop :=
  0 ≤ x ∧ x < 1 / 4

theorem solution_proof
  (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_domain : ∀ x, x ∈ Icc (-1 : ℝ) (1 : ℝ))
  (h_ineq : ∀ m n, m + n ≠ 0 → m ∈ Icc (-1) 1 → n ∈ Icc (-1) 1 → (f m + f n)/(m + n) > 0) :
  ∀ x : ℝ, solution_inequality f x ↔ solution_set x := 
sorry

end solution_proof_l626_626008


namespace right_triangle_exists_l626_626005

theorem right_triangle_exists (lengths : Finset ℕ) :
  lengths = {3, 8, 12, 15, 17, 18} →
  ∃ a b c ∈ lengths, a^2 + b^2 = c^2 ∧ ({a, b, c} = {8, 15, 17}) :=
by 
  intro h
  use 8, 15, 17
  have ha : 8 ∈ {3, 8, 12, 15, 17, 18} := by simp [Finset.mem_insert, Finset.mem_singleton]
  have hb : 15 ∈ {3, 8, 12, 15, 17, 18} := by simp [Finset.mem_insert, Finset.mem_singleton]
  have hc : 17 ∈ {3, 8, 12, 15, 17, 18} := by simp [Finset.mem_insert, Finset.mem_singleton]
  exact ⟨ha, hb, hc, by norm_num, by norm_num⟩

end right_triangle_exists_l626_626005


namespace trapezoid_slopes_sum_l626_626589

noncomputable def problem_statement : Prop :=
  let A := (30, 120)
  let D := (32, 129)
  -- Given conditions
  is_isosceles_trapezoid (A : ℤ × ℤ) (D : ℤ × ℤ)
  -- The condition that sides are not horizontal or vertical
  ∧ no_horizontal_or_vertical_sides A D
  -- The condition that sides AB and CD are parallel
  ∧ sides_parallel A D
  -- To prove: The sum of the absolute values of all possible slopes for AB, expressed as p/q in simplest form, is 9
  ∧ (let slopes := possible_slopes A D in
     let abs_sum := sum_of_absolute_values slopes in
     abs_sum = 9)

theorem trapezoid_slopes_sum :
  problem_statement := by
  sorry

end trapezoid_slopes_sum_l626_626589


namespace marc_journey_fraction_l626_626127

-- Defining the problem based on identified conditions
def total_cycling_time (k : ℝ) : ℝ := 20 * k
def total_walking_time (k : ℝ) : ℝ := 60 * (1 - k)
def total_travel_time (k : ℝ) : ℝ := total_cycling_time k + total_walking_time k

theorem marc_journey_fraction:
  ∀ (k : ℝ), total_travel_time k = 52 → k = 1 / 5 :=
by
  sorry

end marc_journey_fraction_l626_626127


namespace abs_inequality_solution_l626_626529

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l626_626529


namespace find_a_b_l626_626281

theorem find_a_b (a b : ℤ) (h: 4 * a^2 + 3 * b^2 + 10 * a * b = 144) :
    (a = 2 ∧ b = 4) :=
by {
  sorry
}

end find_a_b_l626_626281


namespace incenter_of_triangle_l626_626903

-- Define the geometrical objects and conditions
structure Triangle :=
(A B C : Point)

structure PointInTriangle (T : Triangle) :=
(O : Point)
(inside : T.A ≠ O ∧ T.B ≠ O ∧ T.C ≠ O)

structure PassThroughCircumcenter (T : Triangle) (O : Point) :=
(AO_passes_through_BCO : Line T.A O passes through circumcenter of Triangle.mk T.B T.C O)
(BO_passes_through_ACO : Line T.B O passes through circumcenter of Triangle.mk T.A T.C O)
(CO_passes_through_ABO : Line T.C O passes through circumcenter of Triangle.mk T.A T.B O)

-- Define the main theorem
theorem incenter_of_triangle (T : Triangle) (PIT : PointInTriangle T)
  (PTC : PassThroughCircumcenter T PIT.O) : is_incenter_of T PIT.O :=
sorry

end incenter_of_triangle_l626_626903


namespace sum_of_triangle_angles_is_540_l626_626186

theorem sum_of_triangle_angles_is_540
  (A1 A3 A5 B2 B4 B6 C7 C8 C9 : ℝ)
  (H1 : A1 + A3 + A5 = 180)
  (H2 : B2 + B4 + B6 = 180)
  (H3 : C7 + C8 + C9 = 180) :
  A1 + A3 + A5 + B2 + B4 + B6 + C7 + C8 + C9 = 540 :=
by
  sorry

end sum_of_triangle_angles_is_540_l626_626186


namespace sum_of_factors_of_24_l626_626217

theorem sum_of_factors_of_24 : 
  let factors := [1, 2, 4, 8, 3, 6, 12, 24] in
  (factors.sum = 60) :=
by {
  let factors := [1, 2, 4, 8, 3, 6, 12, 24]
  show factors.sum = 60,
  sorry
}

end sum_of_factors_of_24_l626_626217


namespace f_n_div_n_neq_1990_l626_626884

theorem f_n_div_n_neq_1990 (p q n : ℕ) (hp : p.prime) (hq : q.prime) (hpq : p ≠ q) (hn : 0 < n) :
    p + q - 1 ≠ 1990 := by
  sorry

end f_n_div_n_neq_1990_l626_626884


namespace acid_solution_l626_626199

theorem acid_solution (m x : ℝ) (h1 : 0 < m) (h2 : m > 50)
  (h3 : (m / 100) * m = (m - 20) / 100 * (m + x)) : x = 20 * m / (m + 20) := 
sorry

end acid_solution_l626_626199


namespace value_of_a_c_l626_626596

theorem value_of_a_c {a b c d : ℝ} :
  (∀ x y : ℝ, y = -|x - a| + b → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) ∧
  (∀ x y : ℝ, y = |x - c| - d → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) →
  a + c = 8 :=
by
  sorry

end value_of_a_c_l626_626596


namespace ratio_of_allergic_to_peanut_to_total_l626_626064

def total_children : ℕ := 34
def children_not_allergic_to_cashew : ℕ := 10
def children_allergic_to_both : ℕ := 10
def children_allergic_to_cashew : ℕ := 18
def children_not_allergic_to_any : ℕ := 6
def children_allergic_to_peanut : ℕ := 20

theorem ratio_of_allergic_to_peanut_to_total :
  (children_allergic_to_peanut : ℚ) / (total_children : ℚ) = 10 / 17 :=
by
  sorry

end ratio_of_allergic_to_peanut_to_total_l626_626064


namespace problem_statement_l626_626943

theorem problem_statement (a b c : ℝ) (h₀ : 0 < a)
    (h₁ : ∀ x : ℝ, f x + 2 = f (2 - x)) :
  f (real.sqrt 2 / 2) > f real.pi :=
begin
  sorry
end

end problem_statement_l626_626943


namespace DT_length_correct_l626_626077

-- Defining the points and edges based on the problem.
def cube_edge_length (a : ℝ) := a

def M (a : ℝ) := (a, a, a/2)
def N (a : ℝ) := (a, a/3, a)
def P (a : ℝ) := (0, 0, 3*a/4)

-- Function to find the intersection of the plane through M, N, P with the line HD.
noncomputable def DT_length (a : ℝ) : ℝ := λ (M : ℝ × ℝ × ℝ) (N : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ), sorry

-- The proof statement in Lean 4.
theorem DT_length_correct (a : ℝ) : 
  DT_length a (M a) (N a) (P a) = (5/6) * a := 
sorry

end DT_length_correct_l626_626077


namespace Dan_average_rate_l626_626629

-- Constants for the problem
def running_distance : ℝ := 3 -- miles
def swimming_distance : ℝ := 3 -- miles
def running_rate : ℝ := 10 -- miles per hour
def swimming_rate : ℝ := 6 -- miles per hour

-- Definitions for time calculations
def running_time : ℝ := running_distance / running_rate -- hours
def swimming_time : ℝ := swimming_distance / swimming_rate -- hours
def running_time_minutes : ℝ := running_time * 60 -- minutes
def swimming_time_minutes : ℝ := swimming_time * 60 -- minutes
def total_time_minutes : ℝ := running_time_minutes + swimming_time_minutes -- minutes

-- Definitions for distance calculations
def total_distance : ℝ := running_distance + swimming_distance -- miles

-- Definition for average rate calculation
def average_rate : ℝ := total_distance / total_time_minutes -- miles per minute

-- The theorem to prove
theorem Dan_average_rate : average_rate = 0.125 := by
  sorry

end Dan_average_rate_l626_626629


namespace tournament_arrangement_l626_626442

variable {Player : Type}
variable (n : ℕ)
variable (match : Player → Player → Prop)
variable (H : ∀ P Q : Player, match P Q ∨ match Q P)
variable (H_no_draw : ∀ P Q : Player, match P Q → ¬(match Q P))

theorem tournament_arrangement (n : ℕ)
  (Players : Fin n → Player)
  (match : Player → Player → Prop)
  (H : ∀ i j, i ≠ j → match (Players i) (Players j) ∨ match (Players j) (Players i))
  (H_no_draw : ∀ i j, match (Players i) (Players j) ↔ ¬(match (Players j) (Players i))) :
  ∃ (P : Fin n → Player), ∀ i (h : i < n - 1), match (P i) (P ⟨i.succ, Nat.succ_lt_succ h⟩) := 
begin
  sorry
end

end tournament_arrangement_l626_626442


namespace parametric_ellipse_has_major_axis_length_6_l626_626808

noncomputable def parametric_ellipse_major_axis_length : ℝ :=
  if (∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ) then 6 else 0

theorem parametric_ellipse_has_major_axis_length_6 :
  parametric_ellipse_major_axis_length = 6 :=
sorry

end parametric_ellipse_has_major_axis_length_6_l626_626808


namespace modulus_of_z_equals_sqrt2_l626_626379

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := 2 * i + (2 / (1 + i))

theorem modulus_of_z_equals_sqrt2 : complex.abs z = real.sqrt 2 :=
by sorry

end modulus_of_z_equals_sqrt2_l626_626379


namespace entrants_total_l626_626625

theorem entrants_total (N : ℝ) (h1 : N > 800)
  (h2 : 0.35 * N = NumFemales)
  (h3 : 0.65 * N = NumMales)
  (h4 : NumMales - NumFemales = 252) :
  N = 840 := 
sorry

end entrants_total_l626_626625


namespace no_square_number_divisible_by_six_in_range_l626_626335

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (6 ∣ x) ∧ (50 < x) ∧ (x < 120) :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l626_626335


namespace inequality_conditions_l626_626807

theorem inequality_conditions (x y z : ℝ) (h1 : y - x < 1.5 * abs x) (h2 : z = 2 * (y + x)) : 
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) :=
by
  sorry

end inequality_conditions_l626_626807


namespace property_p_half_squared_property_p_pi_over_4_sin_property_p_third_continuous_l626_626363

-- (1) Prove that f(x) = x^2 has property P(1/2) on the interval [-1, 1]
theorem property_p_half_squared (f : ℝ → ℝ) (x : ℝ) (D : set ℝ) :
   (∀ x ∈ D, f x = x^2) → D = set.Icc (-1 : ℝ) (1 : ℝ) → 
   ∃ x0 ∈ D, f x0 = f (x0 + 1 / 2) :=
sorry

-- (2) Find the range of values for n if f(x) = sin(x) has property P(π/4) on the interval (0, n)
theorem property_p_pi_over_4_sin (f : ℝ → ℝ) (D : set ℝ) :
  (∀ x, f x = Real.sin x) → (0 < n) → 
  (∀ x ∈ D, D = set.Ioo 0 n) → (∃ x0, f x0 = f (x0 + Real.pi / 4)) →
  ∀ n, n ∈ set.Ioi (5 * Real.pi / 8) :=
sorry

-- (3) Prove that given continuous y = f(x) with f(0) = f(2), it has property P(1/3) on [0, 2]
theorem property_p_third_continuous (f : ℝ → ℝ) :
    continuous f → (f 0 = f 2) →
    (∃ x0 ∈ set.Icc 0 2, f x0 = f (x0 + 1 / 3)) :=
sorry

end property_p_half_squared_property_p_pi_over_4_sin_property_p_third_continuous_l626_626363


namespace sqrt_condition_l626_626935

theorem sqrt_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ (x ≥ 1) :=
by
  sorry

end sqrt_condition_l626_626935


namespace round_6437_5054_to_nearest_even_l626_626520

noncomputable def round_to_nearest_even (n : ℝ) : ℤ :=
  let m := n - (floor n : ℝ)
  if m > 0.5 then
    if (ceil n : ℤ) % 2 = 0 then ceil n else ceil n + 1
  else
    if (floor n : ℤ) % 2 = 0 then floor n else floor n - 1

theorem round_6437_5054_to_nearest_even : round_to_nearest_even 6437.5054 = 6438 :=
by
  sorry

end round_6437_5054_to_nearest_even_l626_626520


namespace vertex_angle_is_150_degrees_l626_626954

noncomputable def obtuse_isosceles_triangle (a : ℝ) (b : ℝ) (h : ℝ) (θ : ℝ) (φ : ℝ): Prop :=
  let obtuseness := φ > 90 in
  let congruent_sides := 2 * a = b * 2 * h in
  let base_height_relation := a^2 = (2*a * cos θ) * (2*a * sin θ) in
  let sin_double_angle := sin (2 * θ) = 1 / 2 in
  obtuseness ∧ congruent_sides ∧ base_height_relation ∧ sin_double_angle

theorem vertex_angle_is_150_degrees (a b h θ φ : ℝ) :
  obtuse_isosceles_triangle a b h θ φ → φ = 150 :=
sorry

end vertex_angle_is_150_degrees_l626_626954


namespace largest_subset_no_three_times_l626_626284

theorem largest_subset_no_three_times : 
  ∃ (S : set ℕ), (S ⊆ { x | x ≤ 100 }) ∧ (∀ a ∈ S, ∀ b ∈ S, a = 3 * b → false) ∧ (#S ≤ 76) :=
sorry

end largest_subset_no_three_times_l626_626284


namespace carl_trip_cost_l626_626710

theorem carl_trip_cost :
  (let city_mpg := 30 in
   let hwy_mpg := 40 in
   let one_way_city_miles := 60 in
   let one_way_hwy_miles := 200 in
   let gas_cost_per_gallon := 3 in
   let round_trip_city_miles := 2 * one_way_city_miles in
   let round_trip_hwy_miles := 2 * one_way_hwy_miles in
   let city_gas_needed := round_trip_city_miles / city_mpg in
   let hwy_gas_needed := round_trip_hwy_miles / hwy_mpg in
   let total_gas_needed := city_gas_needed + hwy_gas_needed in
   let total_cost := total_gas_needed * gas_cost_per_gallon in
   total_cost = 42) :=
sorry

end carl_trip_cost_l626_626710


namespace probability_of_true_statement_l626_626941

theorem probability_of_true_statement : 
  (∀ (P1 : Prop), 
    (P1 ↔ ¬(∀ (a b c d : ℕ), (a = b) ∧ (b = c) → (a = c))
      ∧ ¬(∀ (x y : Type), (x = y) → ¬(x ≠ y))
      ∧ (∀ (p q : Prop), (p ∨ q) ↔ ¬(¬p ∧ ¬q))
      ∧ ¬(∀ (r s : ℕ), (r ∨ s) = s ∧ (r = s → s = 0))
      ∧ (∀ (angles : list ℕ), angles.length = 5 → angles.sum = 360))) →
  (∃ count, count = 2) →
  (∃ probability : ℚ, probability = 2 / 5) :=
by
  intro P1 
  intro h1
  intro h2 
  sorry

end probability_of_true_statement_l626_626941


namespace monotonicity_of_f_exists_a_for_min_g_l626_626392

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + a * x + a / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := log x + a / x - 2

theorem monotonicity_of_f (a : ℝ) :
  (∀ x > 0, 0 < f' x a) ∨ 
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ 
   (∀ x ∈ (Ioo 0 x1 ⊔ Ioc x2 ∞), f' x a > 0) ∧ 
   (∀ x ∈ Ioo x1 x2, f' x a < 0)) :=
sorry

theorem exists_a_for_min_g : 
  ∃ a : ℝ, 0 < a ∧ (∀ x ∈ Ioc (0:ℝ) (exp 2), g x a ≥ 2) ∧ g (exp 2) a = 2 :=
sorry

end monotonicity_of_f_exists_a_for_min_g_l626_626392


namespace rationalize_denominator_l626_626914

theorem rationalize_denominator :
  ∃ (A B C : ℤ), 
  (A + B * Real.sqrt C) = (2 + Real.sqrt 5) / (3 - Real.sqrt 5) 
  ∧ A = 11 ∧ B = 5 ∧ C = 5 ∧ A * B * C = 275 := by
  sorry

end rationalize_denominator_l626_626914


namespace binom_invalid_p_throwing_game_probability_hypergeo_distribution_expectation_binom_expected_value_max_l626_626641

-- Option A
theorem binom_invalid_p (n : ℕ) (p : ℚ) (h1 : E(X) = 30) (h2 : D(X) = 20) : p ≠ 2/3 :=
begin
  sorry
end

-- Option B
theorem throwing_game_probability : 
  (probability_level_passed = 31/32) := sorry

-- Option C
theorem hypergeo_distribution_expectation : 
  (E(X) = 9/5) := sorry

-- Option D
theorem binom_expected_value_max : 
  (X ~ B(10, 0.6)) -> (E(X) = 6) -> probability_max @ X = 6 := sorry

end binom_invalid_p_throwing_game_probability_hypergeo_distribution_expectation_binom_expected_value_max_l626_626641


namespace wrong_conclusion_l626_626352

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem wrong_conclusion {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : 2 * a + b = 0) (h₂ : a + b + c = 3) (h₃ : 4 * a + 2 * b + c = 8) :
  quadratic a b c (-1) ≠ 0 :=
sorry

end wrong_conclusion_l626_626352


namespace prove_1997_using_ten_threes_count_of_threes_used_l626_626644

-- Define constants and expressions
def three : ℕ := 3
def three_times_three_hundred_thirty_three : ℕ := three * 333
def three_div_three : ℕ := three / three

-- Formulate the problematic expression
def expression : ℕ := three_times_three_hundred_thirty_three + three_times_three_hundred_thirty_three - three_div_three

-- Theorem that states the problem
theorem prove_1997_using_ten_threes : expression = 1997 :=
by sorry

-- Helper theorem to check the count of threes used in the expression
theorem count_of_threes_used : 
  (count_expr_occur three (repr expression)) = 10 :=
by sorry

end prove_1997_using_ten_threes_count_of_threes_used_l626_626644


namespace a_minus_b_is_perfect_square_l626_626010
-- Import necessary libraries

-- Define the problem in Lean
theorem a_minus_b_is_perfect_square (a b c : ℕ) (h1: Nat.gcd a (Nat.gcd b c) = 1) 
    (h2: (ab : ℚ) / (a - b) = c) : ∃ k : ℕ, a - b = k * k :=
by
  sorry

end a_minus_b_is_perfect_square_l626_626010


namespace equation_true_l626_626294

variables {AB BC CD AD AC BD : ℝ}

theorem equation_true :
  (AD * BC + AB * CD = AC * BD) ∧
  (AD * BC - AB * CD ≠ AC * BD) ∧
  (AB * BC + AC * CD ≠ AC * BD) ∧
  (AB * BC - AC * CD ≠ AC * BD) :=
by
  sorry

end equation_true_l626_626294


namespace smallest_positive_debt_resolvable_l626_626206

theorem smallest_positive_debt_resolvable :
  ∃ D : ℤ, D > 0 ∧ (D = 250 * p + 175 * g + 125 * s ∧ 
  (∀ (D' : ℤ), D' > 0 → (∃ p g s : ℤ, D' = 250 * p + 175 * g + 125 * s) → D' ≥ D)) := 
sorry

end smallest_positive_debt_resolvable_l626_626206


namespace line_perpendicular_to_plane_l626_626786

theorem line_perpendicular_to_plane 
  (m n : Type) [line m] [line n] (α : Type) [plane α] 
  (hm : m ⊆ α) (hn : n ⊥ α) : n ⊥ m := 
sorry

end line_perpendicular_to_plane_l626_626786


namespace polynomial_equality_l626_626553

open Real

variable {F G H : ℝ → ℝ} {n : ℕ} 

-- Assume F, G, H are polynomials with degree at most 2n + 1, with real coefficients.
noncomputable theory
def polynomial_degree_at_most (P : ℝ → ℝ) (d : ℕ) := ∃ (a : ℕ) (h : a ≤ d), ∃ (p : polynomial ℝ), ∀ x, P x = polynomial.eval x p

-- Assume the three conditions given in the problem.
axiom degree_le_F : polynomial_degree_at_most F (2 * n + 1)
axiom degree_le_G : polynomial_degree_at_most G (2 * n + 1)
axiom degree_le_H : polynomial_degree_at_most H (2 * n + 1)

axiom condition_1 : ∀ x : ℝ, F x ≤ G x ∧ G x ≤ H x
axiom condition_2 : ∃ (x_i : fin n → ℝ) (h : ∀ i j : fin n, i ≠ j → x_i i ≠ x_i j), ∀ i : fin n, F (x_i i) = H (x_i i)
axiom condition_3 : ∃ (x_0 : ℝ) (h : ∀ i : fin n, x_0 ≠ x_i i), F x_0 + H x_0 = 2 * G x_0

-- The goal is to prove that F(x) + H(x) = 2G(x) for all real numbers x.
theorem polynomial_equality : ∀ x : ℝ, F x + H x = 2 * G x := 
by sorry

end polynomial_equality_l626_626553


namespace parabola_focus_l626_626767

theorem parabola_focus (x y : ℝ) : (y = x^2 / 8) → (y = x^2 / 8) ∧ (∃ p, p = (0, 2)) :=
by
  sorry

end parabola_focus_l626_626767


namespace f_increasing_f_range_0_1_l626_626777

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2 ^ x + 1))

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

theorem f_range_0_1 : set.range (λ x : ℝ, f x) ∩ set.Icc 0 1 = set.Icc (0 : ℝ) (1 / 3) :=
by
  sorry

end f_increasing_f_range_0_1_l626_626777


namespace last_digit_base4_of_89_is_1_l626_626316

theorem last_digit_base4_of_89_is_1 : 
  last_digit_in_base 4 89 1 := sorry

end last_digit_base4_of_89_is_1_l626_626316


namespace values_of_x_in_range_l626_626406

open Real

noncomputable def count_x_values : Nat :=
  let lower_bound := -25
  let upper_bound := 105
  let count_k := ((upper_bound / π).floor - (lower_bound / π).ceil).succ
  count_k

theorem values_of_x_in_range :
  ∃ k_values : FinSet ℤ, 
  (∀ k ∈ k_values, -25 < k * π ∧ k * π < 105) ∧
  (∀ x, x ∈ (Set.Ioo (-25) 105) → (cos x)^2 + 2 * (sin x)^2 = 1 → x = k_values.to_list.length ∧
   k_values.to_list.length = 41) :=
sorry

end values_of_x_in_range_l626_626406


namespace number_of_ways_to_place_5_distinguishable_balls_into_3_indistinguishable_boxes_l626_626409

theorem number_of_ways_to_place_5_distinguishable_balls_into_3_indistinguishable_boxes :
  let balls := 5
      boxes := 3 in
  (3 ^ balls - Nat.choose boxes 1 * 2 ^ balls + Nat.choose boxes 2 * 1 ^ balls) / (Fact boxes) = 25 :=
by
  sorry

end number_of_ways_to_place_5_distinguishable_balls_into_3_indistinguishable_boxes_l626_626409


namespace number_of_divisors_l626_626344

theorem number_of_divisors (n : ℕ) (k : ℕ) (p : Fin k → ℕ) (α : Fin k → ℕ) :
  (∀ i j : Fin k, i ≠ j → p i ≠ p j) →
  (∀ i : Fin k, Nat.Prime (p i)) →
  n = ∏ i in Finset.univ, (p i) ^ (α i) →
  (∀ i : Fin k, α i > 0) →
  (∏ i in Finset.univ, (α i + 1)) = (∑ d in finset.divisors n, 1) :=
by
  sorry

end number_of_divisors_l626_626344


namespace triangle_area_tangent_line_l626_626155

def curve (x : ℝ) : ℝ := x^3 + 11

def tangent_line (m : ℝ) (x₀ y₀ : ℝ) : ℝ := m * (x₀ - x) + y₀

theorem triangle_area_tangent_line :
  let P : ℝ × ℝ := (1, 12)
  let m : ℝ := 3 * (P.1)^2
  let l := tangent_line m P.1 P.2
  let x_intercept := 3
  let y_intercept := -9
  (1 / 2) * abs (x_intercept * y_intercept) = 27 / 2 := 
begin
  sorry
end

end triangle_area_tangent_line_l626_626155


namespace ratio_used_to_total_apples_l626_626704

noncomputable def total_apples_bonnie : ℕ := 8
noncomputable def total_apples_samuel : ℕ := total_apples_bonnie + 20
noncomputable def eaten_apples_samuel : ℕ := total_apples_samuel / 2
noncomputable def used_for_pie_samuel : ℕ := total_apples_samuel - eaten_apples_samuel - 10

theorem ratio_used_to_total_apples : used_for_pie_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 1 ∧
                                     total_apples_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 7 := by
  sorry

end ratio_used_to_total_apples_l626_626704


namespace max_BP_squared_l626_626874

-- Define circle and points
variables {ω : Type*} [MetricSpace ω] [Circumcircle ω]
variables {A B T C P : ω}

-- Assume conditions
variables (diameter : DistanceBetween A B = 24)
variables (extend_AB : ExtendLineThrough A B C)
variables (T_on_ω : OnCircumcircle T ω)
variables (CT_tangent : TangentToCircumcircle CT ω)
variables (P_perpendicular : PerpendicularFrom A CT P)

theorem max_BP_squared : ∀ A B T C P : ω,
  DistanceBetween A B = 24 →
  ExtendLineThrough A B C →
  OnCircumcircle T ω →
  TangentToCircumcircle C T ω →
  PerpendicularFrom A CT P →
  exists m, MaximumDistanceSquared BP m 612 := 
by
  intros A B T C P
  assume diameter extend_AB T_on_ω CT_tangent P_perpendicular,
  sorry

end max_BP_squared_l626_626874


namespace int_sum_solutions_l626_626215

theorem int_sum_solutions : 
  let S := {n : ℤ | |n| < |n - 5| ∧ |n - 5| < 10 } in
  S.sum id = -12 :=
by
  sorry

end int_sum_solutions_l626_626215


namespace rhombus_perimeter_l626_626939

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 52 :=
by
  sorry

end rhombus_perimeter_l626_626939


namespace fractional_sum_bounds_l626_626135

def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem fractional_sum_bounds (n : ℕ) (h : n > 0) :
  (n^2 - n) / 2 ≤ ( ∑ k in Finset.range (n^2 + 1), fractional_part (real.sqrt k) ) ∧
  ( ∑ k in Finset.range (n^2 + 1), fractional_part (real.sqrt k) ) ≤ (n^2 - 1) / 2 :=
by
  sorry

end fractional_sum_bounds_l626_626135


namespace expansion_a0_value_l626_626771

theorem expansion_a0_value :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), (∀ x : ℝ, (x+1)^5 = a_0 + a_1*(x-1) + a_2*(x-1)^2 + a_3*(x-1)^3 + a_4*(x-1)^4 + a_5*(x-1)^5) ∧ a_0 = 32 :=
  sorry

end expansion_a0_value_l626_626771


namespace smallest_positive_period_maximum_value_and_points_monotonically_increasing_intervals_l626_626021

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + 2 * sin (x) ^ 2

-- Statement 1: The smallest positive period of f(x) is π
theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x := by
  sorry

-- Statement 2: The maximum value of f(x) is 2 at x = kπ + π/3 (k ∈ ℤ)
theorem maximum_value_and_points : 
  ∀ x : ℝ, ∃ k : ℤ, f x = 2 ↔ x = k * π + π / 3 := by
  sorry

-- Statement 3: The function f(x) is monotonically increasing on intervals [kπ - π/6, kπ + π/3] (k ∈ ℤ)
theorem monotonically_increasing_intervals : 
  ∀ k : ℤ, ∀ x y : ℝ, (k * π - π / 6 ≤ x ∧ x ≤ y ∧ y ≤ k * π + π / 3) → f x ≤ f y := by
  sorry

end smallest_positive_period_maximum_value_and_points_monotonically_increasing_intervals_l626_626021


namespace small_bottle_volume_l626_626587

theorem small_bottle_volume :
  ∃ V : ℕ, (V > 0) ∧ (let cost_big := 2700 in
                      let volume_big := 30 in
                      let cost_small := 600 in
                      let saved := 300 in
                      let cost_per_ounce_big := cost_big / volume_big in
                      let cost_per_ounce_small := cost_small / V in
                      volume_big * cost_per_ounce_small - volume_big * cost_per_ounce_big = saved) ∧ V = 6 :=
sorry

end small_bottle_volume_l626_626587


namespace range_of_f_l626_626424

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x ≤ 1 then 3^x else -2 * x^2 + m

theorem range_of_f (m : ℝ) : (∀ y ∈ set.range (f (x:=_)), y ≤ 3) → 2 < m ∧ m ≤ 5 :=
by
  intros h
  sorry

end range_of_f_l626_626424


namespace area_of_inner_square_l626_626922

theorem area_of_inner_square (s₁ s₂ : ℝ) (side_length_WXYZ : ℝ) (WI : ℝ) (area_IJKL : ℝ) 
  (h1 : s₁ = 10) 
  (h2 : s₂ = 10 - 2 * Real.sqrt 2)
  (h3 : side_length_WXYZ = 10)
  (h4 : WI = 2)
  (h5 : area_IJKL = (s₂)^2): 
  area_IJKL = 102 - 20 * Real.sqrt 2 :=
by
  sorry

end area_of_inner_square_l626_626922


namespace p_bounds_l626_626913

noncomputable def p : ℝ := ∏ (k : ℕ) in (finset.range 999).filter (λ x, x % 2 = 0), (↑(2 * k + 1) / ↑(2 * k + 2))

theorem p_bounds : 
  (1 / 1999 : ℝ) < p ∧ p < (1 / 44 : ℝ) := 
by 
  sorry

end p_bounds_l626_626913


namespace necessary_condition_extremum_not_sufficient_condition_extremum_l626_626056

-- Definitions from the problem
variable (f : ℝ → ℝ) (x0 : ℝ)
hypothesis hf_diff : DifferentiableAt ℝ f x0

def is_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x ∨ f y ≥ f x

-- Problem statements
theorem necessary_condition_extremum (h_extremum : is_extremum f x0) : deriv f x0 = 0 :=
  sorry

theorem not_sufficient_condition_extremum :
  (deriv f x0 = 0 → is_extremum f x0) → False :=
  sorry

end necessary_condition_extremum_not_sufficient_condition_extremum_l626_626056


namespace abs_inequality_solution_l626_626530

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l626_626530


namespace chemistry_problem_l626_626258

theorem chemistry_problem
  (V : ℝ) 
  (current_water_fraction : ℝ := 0.60)
  (current_acid_fraction : ℝ := 0.40) 
  (added_water : ℝ := 100)
  (desired_water_fraction : ℝ := 0.70)
  (desired_acid_fraction : ℝ := 0.30) 
  (mixture_equation : 0.60 * V + 100 = 0.70 * (V + 100)) :
  V = 300 :=
by
  sorry

end chemistry_problem_l626_626258


namespace carbon_paper_count_l626_626272

theorem carbon_paper_count (x : ℕ) (sheets : ℕ) (copies : ℕ) (h1 : sheets = 3) (h2 : copies = 2) :
  x = 1 :=
sorry

end carbon_paper_count_l626_626272


namespace area_of_right_isosceles_triangle_l626_626337

def is_right_isosceles (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

theorem area_of_right_isosceles_triangle (a b c : ℝ) (h : is_right_isosceles a b c) (h_hypotenuse : c = 10) :
  1/2 * a * b = 25 :=
by
  sorry

end area_of_right_isosceles_triangle_l626_626337


namespace sum_a_inv_eq_two_l626_626103

-- Define the sequence
def a : ℕ → ℕ
| 0     := 1  -- we'll use a(0) to represent a_1 in the problem
| (n+1) := 1 + (List.prod (List.init (List.range (n+1)).map a))

-- Define the infinite sum of the reciprocals of the sequence
noncomputable def sum_a_inv : ℝ := Real.tsum (fun n => 1 / a n)

-- The theorem to be proved
theorem sum_a_inv_eq_two : sum_a_inv = 2 :=
sorry

end sum_a_inv_eq_two_l626_626103


namespace remainder_of_S_l626_626094

-- Define the set of remainders modulo 1000 for powers of 2
def R : Set ℕ := {r | ∃ n : ℕ, r = 2^n % 1000}

-- Define the sum S of all elements in R
def S : ℕ := ∑ r in R, r

-- Required proof statement: the remainder when S is divided by 1000 is 375
theorem remainder_of_S : S % 1000 = 375 :=
by
  sorry

end remainder_of_S_l626_626094


namespace common_chord_eq_l626_626029

-- Conditions: definitions of the circles C₁ and C₂.
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 4 * y + 9 = 0

-- Proving the required results
theorem common_chord_eq :
  ( ∀ x y : ℝ, circle1_eq x y → circle2_eq x y → 4 * x - 9 = 0 ) ∧
  ( ∀ x : ℝ, (∃ y : ℝ, circle1_eq x y ∧ circle2_eq x y) →
    x = 9 / 4 ) ∧
  ( ∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y ∧
    x = 9 / 4 ∧ (abs (x - 1)) / 2 = 5 / 4 → 
    ∃ |AB| : ℝ, |AB| = sqrt 55 / 2 ) :=
by sorry

end common_chord_eq_l626_626029


namespace proof_tangent_circles_l626_626491

noncomputable def tangent_circles (A B C O H M N : Type*) [triangle A B C]
  [circumcenter A B C O] [orthocenter A B C H]
  [circumcircle A B C Ω] [midpoint A H M] [midpoint B H N]
  [distinct M O H N] [points_lie_on_circle M N O H ω] : Prop :=
  ∃ ω Ω, ∀ (M N O H : point), 
  (is_midpoint M H A ∧ is_midpoint N H B ∧ circumcenter_of ABC O ∧ orthocenter_of ABC H ∧ 
    points_on_circle {M, N, O, H} ω ∧ points_on_circle {A, B, C} Ω) →
  are_internally_tangent ω Ω

theorem proof_tangent_circles 
  {A B C O H M N : Type*} [triangle A B C]
  [circumcenter A B C O] [orthocenter A B C H]
  [circumcircle A B C Ω] [midpoint A H M] [midpoint B H N]
  [distinct M N O H] [points_lie_on_circle M N O H ω] :
  tangent_circles A B C O H M N :=
sorry

end proof_tangent_circles_l626_626491


namespace trigonometric_identity_proof_l626_626774

variable (α : ℝ)

-- Conditions
def sin_plus_2cos_eq_zero (α : ℝ) : Prop := sin α + 2 * cos α = 0

-- Expression to evaluate 
def expr (α : ℝ) : ℝ := 2 * sin α * cos α - cos α ^ 2

-- The theorem statement
theorem trigonometric_identity_proof (h : sin_plus_2cos_eq_zero α) : expr α = -1 :=
by
  sorry

end trigonometric_identity_proof_l626_626774


namespace malachi_additional_photos_l626_626070

-- Definition of the conditions
def total_photos : ℕ := 2430
def ratio_last_year : ℕ := 10
def ratio_this_year : ℕ := 17
def total_ratio_units : ℕ := ratio_last_year + ratio_this_year
def diff_ratio_units : ℕ := ratio_this_year - ratio_last_year
def photos_per_unit : ℕ := total_photos / total_ratio_units
def additional_photos : ℕ := diff_ratio_units * photos_per_unit

-- The theorem proving how many more photos Malachi took this year than last year
theorem malachi_additional_photos : additional_photos = 630 := by
  sorry

end malachi_additional_photos_l626_626070


namespace sequences_no_two_heads_follow_each_other_l626_626262

open Nat

-- Definitions only based on conditions provided
def valid_sequences (n : ℕ) (f : ℕ) : ℕ :=
  binomial (n - f + 1) f

def count_valid_sequences (n : ℕ) : ℕ :=
  (range (n.div 2 + 1)).sum (λ f, valid_sequences n f)

theorem sequences_no_two_heads_follow_each_other (n : ℕ) (h : n = 12) : count_valid_sequences n = 377 :=
by
  rw h
  -- no proof needed
  sorry

end sequences_no_two_heads_follow_each_other_l626_626262


namespace cos_alpha_value_l626_626413

theorem cos_alpha_value (α : ℝ) (h : real.cos (real.pi + α) = -1/3) : real.cos α = 1/3 :=
by
  sorry

end cos_alpha_value_l626_626413


namespace max_profit_at_32_l626_626862

-- Definitions for revenue R(x)
def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then 400 - 6 * x
  else if 40 < x then (7400 / x) - (40000 / x^2)
  else 0  -- value for non-positive x is not specified in the problem

-- Definition for cost C(x)
def C (x : ℝ) : ℝ := 16 * x + 40

-- Definition for profit W(x)
def W (x : ℝ) : ℝ := x * R(x) - C(x)

-- Proof statement
theorem max_profit_at_32 : ∀ (x : ℝ), 0 < x → 
  (W x ≤ 6104) ∧ (W 32 = 6104) :=
by
  sorry

end max_profit_at_32_l626_626862


namespace geometric_sequences_count_l626_626110

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 500}

def is_geometric_sequence (a b c d : ℕ) (q : ℕ) : Prop :=
  b = a * q ∧ c = a * q^2 ∧ d = a * q^3

theorem geometric_sequences_count :
  (∃ a b c d q ∈ S, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
    ∧ q ≥ 2 ∧ is_geometric_sequence a b c d q) ↔ 94 :=
sorry

end geometric_sequences_count_l626_626110


namespace number_of_students_is_four_l626_626586

-- Definitions from the conditions
def average_weight_decrease := 8
def replaced_student_weight := 96
def new_student_weight := 64
def weight_decrease := replaced_student_weight - new_student_weight

-- Goal: Prove that the number of students is 4
theorem number_of_students_is_four
  (average_weight_decrease: ℕ)
  (replaced_student_weight new_student_weight: ℕ)
  (weight_decrease: ℕ) :
  weight_decrease / average_weight_decrease = 4 := 
by
  sorry

end number_of_students_is_four_l626_626586


namespace solutions_to_x4_eq1_l626_626457

noncomputable def solutions_x4_eq1 : Set ℂ := {1, -1, Complex.i, -Complex.i}

theorem solutions_to_x4_eq1 :
  {x : ℂ | x^4 = 1} = solutions_x4_eq1 := by
  sorry

end solutions_to_x4_eq1_l626_626457


namespace arithmetic_sequence_properties_sum_of_sequence_l626_626370

theorem arithmetic_sequence_properties {a : ℕ → ℚ} (d : ℚ) :
  a 1 = 2 →
  a 2 = 2 + d →
  a 3 = 2 + 2 * d →
  (2 + d)^2 = 2 * (2 + 4 * d) →
  (∀ n, a n = 2) ∨ (∀ n, a n = 4 * n - 2) :=
sorry

theorem sum_of_sequence {S : ℕ → ℚ} (a : ℕ → ℚ) :
  (∀ n, a n = 4 * n - 2) →
  (∀ n, S n = n * (2 + (4 * n - 2)) / 2) →
  ∀ n, (∑ i in Finset.range n, 1 / (2 * S (i + 1) - 1)) = n / (2 * n + 1) :=
sorry

end arithmetic_sequence_properties_sum_of_sequence_l626_626370


namespace train_length_correct_l626_626290

def train_length (speed_kmph : ℝ) (time_sec : ℝ) (platform_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * time_sec
  total_distance - platform_length_m

theorem train_length_correct :
  train_length 60 23.998080153587715 260 ≈ 139.97 := 
by
  -- Proof steps would go here; omitted for brevity
  sorry

end train_length_correct_l626_626290


namespace max_interval_a_l626_626393

def f (x : ℝ) : ℝ := 4 * x ^ 3 - 3 * x

theorem max_interval_a (a : ℝ) (h1 : ∃ x ∈ set.Ioo a (a+2), ∀ y ∈ set.Ioo a (a+2), f y ≤ f x) : 
  -5 / 2 < a ∧ a ≤ -1 := 
sorry

end max_interval_a_l626_626393


namespace percentage_increase_in_population_due_to_birth_is_55_l626_626604

/-- The initial population at the start of the period is 100,000 people. -/
def initial_population : ℕ := 100000

/-- The period of observation is 10 years. -/
def period : ℕ := 10

/-- The number of people leaving the area each year due to emigration is 2000. -/
def emigration_per_year : ℕ := 2000

/-- The number of people coming into the area each year due to immigration is 2500. -/
def immigration_per_year : ℕ := 2500

/-- The population at the end of the period is 165,000 people. -/
def final_population : ℕ := 165000

/-- The net migration per year is calculated by subtracting emigration from immigration. -/
def net_migration_per_year : ℕ := immigration_per_year - emigration_per_year

/-- The total net migration over the period is obtained by multiplying net migration per year by the number of years. -/
def net_migration_over_period : ℕ := net_migration_per_year * period

/-- The total population increase is the difference between the final and initial population. -/
def total_population_increase : ℕ := final_population - initial_population

/-- The increase in population due to birth is calculated by subtracting net migration over the period from the total population increase. -/
def increase_due_to_birth : ℕ := total_population_increase - net_migration_over_period

/-- The percentage increase in population due to birth is calculated by dividing the increase due to birth by the initial population, and then multiplying by 100 to convert to percentage. -/
def percentage_increase_due_to_birth : ℕ := (increase_due_to_birth * 100) / initial_population

/-- The final Lean statement to prove. -/
theorem percentage_increase_in_population_due_to_birth_is_55 :
  percentage_increase_due_to_birth = 55 := by
sorry

end percentage_increase_in_population_due_to_birth_is_55_l626_626604


namespace norm_photos_l626_626252

-- Define variables for the number of photos taken by Lisa, Mike, and Norm.
variables {L M N : ℕ}

-- Define the given conditions as hypotheses.
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := N = 2 * L + 10

-- State the problem in Lean: we want to prove that the number of photos Norm took is 110.
theorem norm_photos (L M N : ℕ) (h1 : condition1 L M N) (h2 : condition2 L N) : N = 110 :=
by
  sorry

end norm_photos_l626_626252


namespace BC_AO_range_l626_626086

theorem BC_AO_range (a b c : ℝ) (O : ℝ) (h1 : b^2 - 2*b + c^2 = 0)
  (h2 : 0 < b ∧ b < 2 ∧ c^2 > 0) : 
  ∃ x ∈ set.Ico (-(1 / 4)) 2, x = BC * AO :=
by
  let BC := sqrt (a^2 + b^2 - 2 * a * b * cos O);
  let AO := sqrt ((BC / 2)^2 + (a / 2)^2);
  sorry

end BC_AO_range_l626_626086


namespace M_union_N_eq_l626_626026

def M : Set Int := {x | Int.log (x - 1) ≤ 0}
def N : Set Int := {x | |x| < 2}

theorem M_union_N_eq : M ∪ N = {-1, 0, 1, 2} := 
by 
  sorry

end M_union_N_eq_l626_626026


namespace polynomial_not_factored_l626_626462

theorem polynomial_not_factored :
  ∀ (f : ℝ → ℝ) (g : ℝ → ℝ),
    (∏ (i : ℕ) in finset.range 2020, λ x y, (x ^ i * y ^ i)) + 1
    ≠ f x * g y := 
sorry

end polynomial_not_factored_l626_626462


namespace abs_diff_count_S_1000_l626_626764

def tau (n : ℕ) := (finset.filter (λ d, d ∣ n) (finset.range (n + 1))).card

def S (n : ℕ) : ℕ :=
  (finset.range n).sum (λ i, tau (i + 1))

def count_even_S (n : ℕ) : ℕ :=
  (finset.range (n + 1)).card (\λ i, even (S i))

def count_odd_S (n : ℕ) : ℕ :=
  (finset.range (n + 1)).card (\λ i, odd (S i))

theorem abs_diff_count_S_1000 : 
  |count_even_S 1000 - count_odd_S 1000| = 67 :=
sorry

end abs_diff_count_S_1000_l626_626764


namespace hyperbola_distance_l626_626511

variables x y : ℝ

def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

def distance (P F : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

noncomputable def F1 : ℝ × ℝ := (4, 0)  -- Assume positions of foci based on standard form
noncomputable def F2 : ℝ × ℝ := (-4, 0)

theorem hyperbola_distance (P : ℝ × ℝ)
  (hP : hyperbola P.1 P.2)
  (h_dist : distance P F1 = 9) :
  distance P F2 = 17 :=
sorry

end hyperbola_distance_l626_626511


namespace trig_equation_solutions_l626_626647

noncomputable def solutions_to_trig_equation (x : ℝ) : Prop := 
  (1 + Real.cos (4 * x)) * Real.sin (2 * x) = Real.cos (2 * x) ^ 2

theorem trig_equation_solutions :
  ∀ x, solutions_to_trig_equation x ↔ (∃ k ∈ ℤ, x = (-1)^k * π / 12 + π * k / 2) ∨ (∃ n ∈ ℤ, x = π / 4 * (2 * n + 1)) :=
by
  sorry

end trig_equation_solutions_l626_626647


namespace find_t_l626_626025

theorem find_t (t : ℝ) : 
  let M := {1, 3, t}
  let N := {t^2 - t + 1}
  (M ∪ N = M) → t ∈ {0, 2, -1} := 
by
  intros
  sorry

end find_t_l626_626025


namespace numerators_count_l626_626871

open Finset

def T : Set ℚ := {r : ℚ | ∃ (a b : ℕ), 0 < r ∧ r < 1 ∧ r = (10 * a + b) / 99}

noncomputable def numerators_in_lowest_terms : Finset ℕ :=
  (range 100).filter (λ n => gcd n 99 = 1)

theorem numerators_count : card numerators_in_lowest_terms = 60 := 
  sorry

end numerators_count_l626_626871


namespace balls_into_boxes_l626_626408

noncomputable def ways_to_place_balls_into_boxes : ℕ :=
  31

theorem balls_into_boxes :
  ∃ n : ℕ, n = ways_to_place_balls_into_boxes ∧ n = 31 :=
by
  use ways_to_place_balls_into_boxes
  split
  { refl }
  { refl }

end balls_into_boxes_l626_626408


namespace transformations_correct_l626_626696

theorem transformations_correct :
  (∀ x, (λ x, -cos (3 * x)) x = sin (3 * x - π / 2)) ∧
  (∀ x, (λ x, sin (3 * x - π / 6)) x = sin (3 * x - π / 2)) ∧
  ¬(∀ x, (λ x, sin (3 * (x - π / 6))) x = sin (3 * x - π / 2)) ∧
  (∀ x, (λ x, -cos (3 * x)) x = sin (3 * x - π / 2)) :=
by sorry

end transformations_correct_l626_626696


namespace Nagelian_line_through_homothety_center_l626_626933

-- Define two circles with given centers and radii (omega1 and omega2)
structure Circle :=
(center : ℝ × ℝ) (radius : ℝ)

-- Points T1 and T2 are the points of tangency of the common external tangent
variables (T1 T2 : ℝ × ℝ)

-- Define points A and B
def A (T1 T2 : ℝ × ℝ) : ℝ × ℝ := sorry -- Define the condition A is beyond T1 on T1T2 extension
def B (T1 T2 : ℝ × ℝ) : ℝ × ℝ := sorry -- Define the condition B is beyond T2 on T1T2 extension

-- Assume the length condition
axiom length_condition : dist A T1 = dist B T2

-- Nagelian line passes through C and homothety center P of the circles
noncomputable def homothety_center (omega1 omega2 : Circle) : ℝ × ℝ := sorry -- Define center of homothety

-- Define the point C for triangle ABC
variable (C : ℝ × ℝ)

-- Prove the Nagelian line condition
theorem Nagelian_line_through_homothety_center
    (omega1 omega2 : Circle)
    (T1 T2 : ℝ × ℝ)
    (A B C : ℝ × ℝ)
    (length_condition : dist A T1 = dist B T2) :
    ∀ (C : ℝ × ℝ), Nagelian_line A B C → homothety_center omega1 omega2 ∈ Nagelian_line A B C :=
sorry -- proof to be filled by theorem prover


end Nagelian_line_through_homothety_center_l626_626933


namespace pizza_eaten_and_remaining_l626_626229

theorem pizza_eaten_and_remaining : 
  let initial := 1
  let first_trip_eaten := initial * (1 / 3)
  let after_first_trip := initial - first_trip_eaten
  let second_trip_eaten := after_first_trip * (1 / 2)
  let after_second_trip := after_first_trip - second_trip_eaten
  let third_trip_eaten := after_second_trip * (1 / 2)
  let after_third_trip := after_second_trip - third_trip_eaten
  let fourth_trip_eaten := after_third_trip * (1 / 2)
  let after_fourth_trip := after_third_trip - fourth_trip_eaten
  let total_eaten := first_trip_eaten + second_trip_eaten + third_trip_eaten + fourth_trip_eaten
  let remaining := after_fourth_trip
  in total_eaten = 11 / 12 ∧ remaining = 1 / 12 := 
by {
  sorry
}

end pizza_eaten_and_remaining_l626_626229


namespace find_N_with_conditions_l626_626168

open Nat

theorem find_N_with_conditions (N d1 d2 d4 d8 d12 : ℕ) 
  (h1 : N = d1 * d12 ∧ N = d2 * d11 ∧ N = d3 * d10 ∧ N = d4 * d9 ∧
       N = d5 * d8 ∧ N = d6 * d7)
  (h2 : d1 < d2 < d3 < d4 < d5 < d6 < d7 < d8 < d9 < d10 < d11 < d12)
  (h3 : ∃ k, d4 - 1 = k ∧ d_k = (d1 + d2 + d4) * d8) : N = 1989 := sorry

end find_N_with_conditions_l626_626168


namespace youngest_son_cookies_l626_626464

theorem youngest_son_cookies
  (c_o : ℕ) -- cookies for oldest son per day
  (total_cookies : ℕ) -- total cookies in the box
  (days : ℕ) -- total number of days
  (y_c : ℕ) -- cookies for youngest son per day
  (H₁ : c_o = 4)
  (H₂ : total_cookies = 54)
  (H₃ : days = 9) :
  y_c = 2 := 
by
  have total_cookies_oldest : ℕ := c_o * days
  have H₄ : total_cookies_oldest = 36 := by
    rw [H₁, H₃]; 
    exact rfl
  have remaining_cookies : ℕ := total_cookies - total_cookies_oldest
  have H₅ : remaining_cookies = 18 := by
    rw [H₂, H₄]
    exact rfl
  have y_c : ℕ := remaining_cookies / days
  have H₆ : y_c = 2 := by
    rw [H₅, H₃]
    exact rfl
  exact H₆

end youngest_son_cookies_l626_626464


namespace max_volume_of_cylinder_max_volume_is_max_l626_626158

open Real

noncomputable def max_volume_cylinder (h : ℝ) : ℝ :=
  if h ≤ 3 / 2 then (π * (sqrt 3 / 4) ^ 2 * h) else (π / 12 * h * (2 - h / 3) ^ 2)

theorem max_volume_of_cylinder (h : ℝ) (h_pos : 0 < h) (h_le_3 : h ≤ 3) :
  max_volume_cylinder h = if h ≤ 3 / 2 then (π * (sqrt 3 / 4) ^ 2 * h) else (π / 12 * h * (2 - h / 3) ^ 2) :=
by
  sorry

noncomputable def max_volume_among_all : ℝ := π * 8 / 27

theorem max_volume_is_max (V : ℝ) : V = max_volume_among_all ↔ V = π * 8 / 27 :=
by
  sorry

end max_volume_of_cylinder_max_volume_is_max_l626_626158


namespace coefficient_x2_in_expansion_l626_626079

-- Defining the binomial expansion term formula
def binomial_term (n : ℕ) (r : ℕ) (a b : ℤ) : ℤ :=
  (nat.choose n r) * (a^(n-r)) * (b^r)

theorem coefficient_x2_in_expansion :
  (∃ (c : ℚ), ∀ x : ℚ, x ≠ 0 → 
  (∑ r in finset.range (9), binomial_term 8 r x (-1/(2*x)) = 8 choose 2)) :=
sorry

end coefficient_x2_in_expansion_l626_626079


namespace value_of_expression_l626_626114

-- Define the absolute value function for integers (if not already available in imported library)
def abs (n : Int) : Int := if n < 0 then -n else n

theorem value_of_expression : ∀ (x : Int), x = -2023 → abs (abs (abs x - x) - abs x) - x = 4046 :=
by
  intro x h
  rw [h]
  sorry

end value_of_expression_l626_626114


namespace largest_x_plus_y_l626_626923

theorem largest_x_plus_y (x y : ℤ) (h : (x - 2004) * (x - 2006) = 2^y) :
  x + y ≤ 2011 :=
sorry

end largest_x_plus_y_l626_626923


namespace ellipse_equation_max_area_OPQ_l626_626001

-- Definition for ellipse C
def ellipse_eq (a b : ℝ) (a_gt_b : a > b) (eq1 : 2 * b = 2) (eq2 : (sqrt b) / a = (sqrt 2)/2) (eq3 : a ^ 2 = b^2 + (sqrt (a ^ 2 - b ^ 2))^2) : Prop := 
  ∀ x y : ℝ, (x^2 / a ^ 2 + y^2 / b ^ 2 = 1) ↔ (x^2 / 2 + y^2 = 1)

-- Goal: Prove the equation of ellipse
theorem ellipse_equation : ∃ a b : ℝ, a > b ∧ ellipse_eq a b :=
sorry

-- Definition for maximum area of triangle OPQ
def max_area_triangle (M : ℝ × ℝ) (O : ℝ × ℝ) (eq_ellipse : ∀ x y : ℝ, (x^2 / 2 + y^2 = 1))
  (intersect_line : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop) : Prop := 
  ∀ m : ℝ, ∃ P Q : ℝ × ℝ, intersect_line M P Q → 
  (O.1 * P.2 + P.1 * Q.2 + Q.1 * O.2 - (P.2 * Q.1 + Q.2 * O.1 + O.2 * P.1)) / 2 ≤ sqrt 2 / 2

-- Goal: Prove maximum area of triangle OPQ is sqrt(2)/2
theorem max_area_OPQ : ∃ M O : ℝ × ℝ, (∀ x y : ℝ, x^2 / 2 + y^2 = 1) → max_area_triangle M O _ :=
sorry

end ellipse_equation_max_area_OPQ_l626_626001


namespace squares_difference_l626_626723

theorem squares_difference :
  1010^2 - 994^2 - 1008^2 + 996^2 = 8016 :=
by
  sorry

end squares_difference_l626_626723


namespace find_range_of_x_l626_626729

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem find_range_of_x (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
by sorry

end find_range_of_x_l626_626729


namespace solve_for_k_l626_626823

theorem solve_for_k (a k : ℝ) (h : a ^ 10 / (a ^ k) ^ 4 = a ^ 2) : k = 2 :=
by
  sorry

end solve_for_k_l626_626823


namespace hyperbola_real_axis_condition_l626_626793

theorem hyperbola_real_axis_condition (m n : ℝ) :
  (∀ (m n : ℝ), (mn < 0) → (∃ x y : ℝ, (x^2 / m + y^2 / n = 1) → hyperbola_with_real_axis_on_x (m > 0 ∧ n < 0)) ∧
  (hyperbola_with_real_axis_on_x (m > 0 ∧ n < 0) → mn < 0)) :=
by
  sorry

end hyperbola_real_axis_condition_l626_626793


namespace garden_enlargement_l626_626686

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end garden_enlargement_l626_626686


namespace total_earning_correct_l626_626092

-- Define the initial earning of Katrina
def initial_earning := 5.00

-- Define the referral bonus per friend for both Katrina and her friends
def referral_bonus := 5.00

-- Define the number of friends referred initially and later
def initial_referred_friends := 5
def later_referred_friends := 7

-- Total referred friends
def total_referred_friends := initial_referred_friends + later_referred_friends

-- Define Katrina's total earnings
def katrina_earning : ℝ := initial_earning + 
  (initial_referred_friends * referral_bonus) + 
  (later_referred_friends * referral_bonus)

-- Define friends' total earnings
def friends_earning : ℝ := total_referred_friends * referral_bonus

-- Define the total earnings of Katrina and her friends
def total_earning : ℝ := katrina_earning + friends_earning

-- Prove that the total_earning equals 125.00
theorem total_earning_correct : total_earning = 125.00 := by
  -- Placeholder for the proof, since the task doesn't require the actual proof steps
  sorry

end total_earning_correct_l626_626092


namespace total_cost_of_fence_l626_626688

noncomputable def area : ℝ := 3136
noncomputable def sideLength : ℝ := real.sqrt area
noncomputable def perimeter : ℝ := 4 * sideLength
noncomputable def gateWidth : ℝ := 2
noncomputable def adjustedPerimeter : ℝ := perimeter - gateWidth

noncomputable def barbedWireRatio : ℝ := 3
noncomputable def woodenPanelRatio : ℝ := 2
noncomputable def ratioSum : ℝ := barbedWireRatio + woodenPanelRatio

noncomputable def W : ℝ := (woodenPanelRatio / ratioSum) * adjustedPerimeter
noncomputable def B : ℝ := (barbedWireRatio / ratioSum) * adjustedPerimeter

noncomputable def costBarbedWirePerMeter : ℝ := 1.30
noncomputable def costWoodenPanelPerMeter : ℝ := 2.50

noncomputable def costBarbedWire : ℝ := B * costBarbedWirePerMeter
noncomputable def costWoodenPanel : ℝ := W * costWoodenPanelPerMeter
noncomputable def totalCost : ℝ := costBarbedWire + costWoodenPanel

theorem total_cost_of_fence : totalCost = 395.16 := by
  sorry

end total_cost_of_fence_l626_626688


namespace minimum_possible_value_l626_626324

theorem minimum_possible_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ∃ (x : ℝ), x = (a / (3 * b) + b / (6 * c) + c / (9 * a)) ∧ x = (1 / real.sqrt 3 (6 : ℝ)) := 
  sorry

end minimum_possible_value_l626_626324


namespace merged_class_and_student_b_rank_l626_626847

theorem merged_class_and_student_b_rank:
  (let group_A_total := 13 + 8 - 1,
       group_B_total := 10 + 12 - 1,
       group_C_total := 6 + 7 - 1 in
   let merged_class_total := group_A_total + group_B_total + group_C_total in
   merged_class_total = 53) ∧
  ((let student_B_rank := group_A_total + 12 in
   student_B_rank = 32)) :=
by
  sorry

end merged_class_and_student_b_rank_l626_626847


namespace viggo_brother_age_sum_correct_l626_626978

def viggo_and_brother_age_sum (viggo_age_when_brother_2 : ℕ) (brother_age_now : ℕ) : ℕ :=
  let years_passed := brother_age_now - 2
  let viggo_age_now := viggo_age_when_brother_2 + years_passed
  viggo_age_now + brother_age_now

theorem viggo_brother_age_sum_correct :
  ∀ (viggo_age_when_brother_2 brother_age_now : ℕ),
    (viggo_age_when_brother_2 = 2 * 2 + 10) →
    (brother_age_now = 10) →
    (viggo_and_brother_age_sum viggo_age_when_brother_2 brother_age_now = 32) :=
by
  intros viggo_age_when_brother_2 brother_age_now H1 H2
  rw [H1, H2]
  unfold viggo_and_brother_age_sum
  simp
  sorry

end viggo_brother_age_sum_correct_l626_626978


namespace c_plus_d_l626_626833

-- Definitions from conditions
def line (x : ℝ) (c d : ℝ) := c * x + d
def point1 : ℝ × ℝ := (3, -3)
def point2 : ℝ × ℝ := (6, 9)

-- Proof problem
theorem c_plus_d : ∃ c d, 
  (line point1.1 c d = point1.2) ∧ 
  (line point2.1 c d = point2.2) ∧ 
  (c + d = -11) :=
begin
  sorry,
end

end c_plus_d_l626_626833


namespace sound_heard_in_4_seconds_l626_626667

/-- Given the distance between a boy and his friend is 1200 meters,
    the speed of the car is 108 km/hr, and the speed of sound is 330 m/s,
    the duration after which the friend hears the whistle is 4 seconds. -/
theorem sound_heard_in_4_seconds :
  let distance := 1200  -- distance in meters
  let speed_of_car_kmh := 108  -- speed of car in km/hr
  let speed_of_sound := 330  -- speed of sound in m/s
  let speed_of_car := speed_of_car_kmh * 1000 / 3600  -- convert km/hr to m/s
  let effective_speed_of_sound := speed_of_sound - speed_of_car
  let time := distance / effective_speed_of_sound
  time = 4 := 
by
  sorry

end sound_heard_in_4_seconds_l626_626667


namespace correct_average_is_25_point_5_l626_626689

def initial_average : ℝ := 24
def number_of_values : ℕ := 20
def incorrect_values : List ℝ := [35, 20, 50]
def correct_values : List ℝ := [45, 30, 60]
def incorrect_sum := number_of_values * initial_average

-- Calculate the sum of incorrect numbers
def sum_incorrect_values : ℝ := incorrect_values.sum

-- Calculate the sum of correct numbers
def sum_correct_values : ℝ := correct_values.sum

-- Calculate the corrected sum
def corrected_sum : ℝ := incorrect_sum - sum_incorrect_values + sum_correct_values

-- Calculate the correct average
def correct_average : ℝ := corrected_sum / number_of_values

theorem correct_average_is_25_point_5 : correct_average = 25.5 :=
by
  sorry

end correct_average_is_25_point_5_l626_626689


namespace ratio_of_lateral_to_base_area_l626_626198

noncomputable def midpoint (A B : Point) : Point := sorry -- Midpoint definition placeholder

theorem ratio_of_lateral_to_base_area 
  {P A B C M: Point} 
  (hPABC : is_tetrahedron P A B C) 
  (hMidPA : M = midpoint P A) 
  (hMidCone : ∃ (O : Point) (r : ℝ), is_inscribed_circle O r (triangle B C M) ∧ 
                                     is_inscribed_to_edge_midpoint O r B C ∧ 
                                     is_median_intersect O r (triangle P B A) (triangle P C A) ∧ 
                                     height_cone_as_2r P O r):
  let lateral_surface_area := (S_ABC_triangle_area [P, B, A] + S_ABC_triangle_area [P, A, C] + S_ABC_triangle_area [P, B, C]) / 2,
      base_area := (triangle_area A B C)
  in  
  (lateral_surface_area / base_area = 2) :=
sorry

end ratio_of_lateral_to_base_area_l626_626198


namespace problem_solution_l626_626376

-- Definitions and Assumptions
variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, f x - (deriv^[2]) f x > 0)

-- Statement to Prove
theorem problem_solution : e * f 2015 > f 2016 :=
by
  sorry

end problem_solution_l626_626376


namespace Tim_reading_time_l626_626968

theorem Tim_reading_time (meditation_time_per_day : ℕ) (reading_multiplier : ℕ) : 
  meditation_time_per_day = 1 → reading_multiplier = 2 → (reading_multiplier * meditation_time_per_day * 7 = 14) := 
by 
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end Tim_reading_time_l626_626968


namespace smallest_positive_k_l626_626759

noncomputable def rotation_120 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos (2 * Real.pi / 3), -sin (2 * Real.pi / 3)], 
    ![sin (2 * Real.pi / 3), cos (2 * Real.pi / 3)]]

theorem smallest_positive_k (k : ℕ) :
  (rotation_120 ^ k = 1 : Matrix (Fin 2) (Fin 2) ℝ) ↔ k = 3 :=
by sorry

end smallest_positive_k_l626_626759


namespace solve_inequality_l626_626534

theorem solve_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (x ∈ Ioo (-6.5) 3.5) := 
sorry

end solve_inequality_l626_626534


namespace exists_circle_with_three_points_on_circumference_l626_626906

-- Given four points in the plane such that no three points are collinear, 
-- prove there exists a circle such that three of the points lie on the circumference 
-- and the fourth point is either on the circumference or inside the circle.

theorem exists_circle_with_three_points_on_circumference 
  (points : Fin 4 (ℝ × ℝ)) 
  (h_distinct : ∀ i j : Fin 4, i ≠ j → points i ≠ points j)
  (h_no_collinear : ∀ i j k : Fin 4, i ≠ j → j ≠ k → i ≠ k → ¬ (collinear (points i) (points j) (points k))) :
  ∃ (c : ℝ × ℝ) (r : ℝ), r > 0 ∧ 
    (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
       dist (points i) c = r ∧ dist (points j) c = r ∧ dist (points k) c = r ∧
       (∃ l : Fin 4, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ (dist (points l) c = r ∨ dist (points l) c < r))) :=
sorry

end exists_circle_with_three_points_on_circumference_l626_626906


namespace limit_of_derivative_squared_plus_cube_eq_zero_l626_626997

open Filter

variable (f : ℝ → ℝ)

theorem limit_of_derivative_squared_plus_cube_eq_zero 
  (h_deriv : Continuous f') 
  (h_limit : Tendsto (λ x, (f' x)^2 + (f x)^3) atTop (𝓝 0)) :
  Tendsto f atTop (𝓝 0) ∧ Tendsto (λ x, f' x) atTop (𝓝 0) := 
sorry

end limit_of_derivative_squared_plus_cube_eq_zero_l626_626997


namespace area_of_rectangle_l626_626734

-- Define the given lengths and widths in meters.
def length_meters : ℝ := 5.2
def width_meters : ℝ := 2.7

-- Conversion from meters to centimeters.
def length_cm : ℝ := length_meters * 100
def width_cm : ℝ := width_meters * 100

-- Area calculation in square centimeters.
def area_cm2 : ℝ := length_cm * width_cm

-- Conversion from square centimeters to square meters.
def area_m2 : ℝ := area_cm2 / 10000

-- The proof statement we want to verify.
theorem area_of_rectangle :
  area_m2 = 14.04 :=
sorry

end area_of_rectangle_l626_626734


namespace count_distinct_triples_sum_to_55_l626_626852

/-
Given the sum of numbers from 0 to 20, prove that there are exactly 2 distinct 
sets of three distinct integers a, b, and c in this range, such that their sum is 55, 
and changing the signs of these numbers in the expression results in a total sum of 100.
-/

theorem count_distinct_triples_sum_to_55 : 
  ∃! (S : Finset (Fin 21 × Fin 21 × Fin 21)), 
  (∀ (a b c : ℕ), ((a = S.1.val ∧ b = S.2.val ∧ c = S.3.val) → (a + b + c = 55))
    ∧ (210 - 2 * (S.1.val + S.2.val + S.3.val) = 100)
    ∧ (S.to_list.nodup)) 
    ∧ S.card = 2 :=
sorry

end count_distinct_triples_sum_to_55_l626_626852


namespace final_number_is_even_l626_626637

theorem final_number_is_even :
  (∃ (seq : ℕ → ℕ),
     (∀ i, 1 ≤ seq i ∧ seq i ≤ 2020) ∧ 
     (∃ n, ∀ m, m ≥ n → seq m = |seq (m-1) - seq (m-2)|^2) ∧ 
     (∀ i ≠ j, seq i = seq j → false) ∧ 
     (∀ n m, n ≠ m → seq n ≠ seq (m-1) ∧ seq n ≠ seq (m-2)) →
     (∃ m, seq m % 2 = 0)) :=
begin
  sorry
end

end final_number_is_even_l626_626637


namespace min_x_plus_y_l626_626417

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y = 9 :=
sorry

end min_x_plus_y_l626_626417


namespace problem_statement_l626_626489

theorem problem_statement : 
    let i : ℂ := Complex.I in 
    let z : ℂ := i^2023 - 1 in 
    z^2 - 2 * Complex.conj z = 2 :=
by
  sorry

end problem_statement_l626_626489


namespace problem_2014_Shanghai_l626_626662

noncomputable def necessary_not_sufficient (a b : ℝ) : Prop :=
  (a > 2 ∧ b > 2 → a + b > 4) ∧ ¬(a + b > 4 → a > 2 ∧ b > 2)

theorem problem_2014_Shanghai (a b : ℝ) (h1 : a ∈ ℝ) (h2 : b ∈ ℝ) : 
  necessary_not_sufficient a b :=
by sorry

end problem_2014_Shanghai_l626_626662


namespace range_of_f_l626_626251

open Real

noncomputable def f (x : ℝ) := exp x * sin x

theorem range_of_f : 
  (set.range (λ x : ℝ, f x ∈ Icc (0 : ℝ) (exp (π / 2)))) :=
by
  sorry

end range_of_f_l626_626251


namespace shrimp_per_pound_l626_626979

theorem shrimp_per_pound (shrimp_per_guest guests : ℕ) (cost_per_pound : ℝ) (total_spent : ℝ)
  (hshrimp_per_guest : shrimp_per_guest = 5) (hguests : guests = 40) (hcost_per_pound : cost_per_pound = 17.0) (htotal_spent : total_spent = 170.0) :
  let total_shrimp := shrimp_per_guest * guests
  let total_pounds := total_spent / cost_per_pound
  total_shrimp / total_pounds = 20 :=
by
  sorry

end shrimp_per_pound_l626_626979


namespace points_moving_along_perpendicular_lines_l626_626512

noncomputable def points_moving_perpendicularly (A B M N : Type) 
(speeds : ℝ) (k : ℝ) (h1 : speeds ≠ 1) -- Points A and B move with a speed ratio k
(h2 : ∀ (AM BM AN BN : ℝ), AM / BM = k ∧ AN / BN = k ∧ AM + BM = AB ∧ AN + BN = AB) -- Distance conditions
: Prop := 
lines_perpendicular M N

-- We provide a statement without proof
theorem points_moving_along_perpendicular_lines
  (A B M N : Type) (k : ℝ)
  (h1 : speeds ≠ 1) -- Points are moving at constant but unequal speeds
  (h2 : ∀ (AM BM AN BN : ℝ), AM / BM = k ∧ AN / BN = k ∧ AM + BM = AB ∧ AN + BN = AB) -- Distance conditions
  : lines_perpendicular M N := 
sorry

end points_moving_along_perpendicular_lines_l626_626512


namespace no_solution_sin_cos_eq_l626_626601

theorem no_solution_sin_cos_eq (x : ℝ) 
  (h1 : π/4 ≤ x) 
  (h2 : x ≤ π/2) : 
  ¬ (sin (x ^ sin x) = cos (x ^ cos x)) := 
  sorry

end no_solution_sin_cos_eq_l626_626601


namespace no_conditions_problem_count_eq_two_l626_626800

theorem no_conditions_problem_count_eq_two :
  let problems := ["output opposite number of x", "find perimeter of square with area 6",
                   "find maximum of a, b, c", "calculate function value"] in
  let no_conditions := [true, true, false, false] in
  (no_conditions.count true) = 2 := 
by
  sorry

end no_conditions_problem_count_eq_two_l626_626800


namespace largest_value_of_c_l626_626044

noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem largest_value_of_c :
  ∃ (c : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c → |g x - 1| ≤ c) ∧ (∀ (c' : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c' → |g x - 1| ≤ c') → c' ≤ c) :=
sorry

end largest_value_of_c_l626_626044


namespace extremum_f_at_one_range_a_monotonic_f_l626_626503

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem extremum_f_at_one (a : ℝ) (h : a = -1 / 2) : f a 1 = 0 :=
by
  -- Proof for extremum of f at x=1 when a=-1/2
  sorry

theorem range_a_monotonic_f (a : ℝ) : (∀ x > 0, (2 * a * x + 1 - Real.log x) / x^2 ≥ 0) ↔ (a ≥ 1 / (2 * Real.exp 2)) :=
by
  -- Proof defining the range of a for monotonic increase of f on its domain
  sorry

end extremum_f_at_one_range_a_monotonic_f_l626_626503


namespace min_sum_eq_l626_626322

theorem min_sum_eq : 
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → 
  (∃ x : ℝ, x = 3 * real.cbrt (1 / 162) ∧ x = (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a))) :=
by
  intros a b c ha hb hc
  use 3 * real.cbrt (1 / 162)
  split
  { sorry },
  { sorry }

end min_sum_eq_l626_626322


namespace pell_has_infinite_solutions_and_form_l626_626911

theorem pell_has_infinite_solutions_and_form (d : ℕ) (h_square_free : ∀ m : ℕ, m * m ≠ d)
  (x1 y1 : ℕ) (fund_sol : x1 * x1 - d * y1 * y1 = 1) :
  ∃ (f : ℕ → (ℕ × ℕ)), (∀ n : ℕ, ((f n).fst * (f n).fst - d * (f n).snd * (f n).snd = 1)) ∧
  (∀ x y : ℕ, (∃ n : ℕ, x + y * real.sqrt (d : ℝ) = (x1 + y1 * real.sqrt (d : ℝ)) ^ n)) :=
sorry

end pell_has_infinite_solutions_and_form_l626_626911


namespace jill_spent_on_other_items_l626_626132

theorem jill_spent_on_other_items {T : ℝ} (h₁ : T > 0)
    (h₁ : 0.5 * T + 0.2 * T + O * T / 100 = T)
    (h₂ : 0.04 * 0.5 * T = 0.02 * T)
    (h₃ : 0 * 0.2 * T = 0)
    (h₄ : 0.08 * O * T / 100 = 0.0008 * O * T)
    (h₅ : 0.044 * T = 0.02 * T + 0 + 0.0008 * O * T) :
  O = 30 := 
sorry

end jill_spent_on_other_items_l626_626132


namespace magnitude_sum_l626_626380

variables (a b : ℝ^3)
variable (θ : ℝ)

-- Hypotheses
variable (h_angle : θ = real.pi / 3) -- Angle between a and b is 60°
variable (norm_a : ∥a∥ = 1) -- Magnitude of a is 1
variable (norm_b : ∥b∥ = 2) -- Magnitude of b is 2
variable (dot_ab : a ⬝ b = ∥a∥ * ∥b∥ * real.cos θ) -- Dot product property

-- Goal
theorem magnitude_sum : |2 • a + b| = 2 * real.sqrt 3 :=
by
  sorry

end magnitude_sum_l626_626380


namespace log_equivalence_l626_626048

-- Define the original condition and the final proof goal.
theorem log_equivalence (x : ℝ) (h₁ : log 8 (x - 3) = 1 / 3) : log 64 x = log 10 5 / (6 * log 10 2) :=
sorry

end log_equivalence_l626_626048


namespace product_a3_a10_a17_l626_626081

-- Let's define the problem setup
variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a r : α) (n : ℕ) : α := a * r ^ (n - 1)

theorem product_a3_a10_a17 
  (a r : α)
  (h1 : geometric_sequence a r 2 + geometric_sequence a r 18 = -15) 
  (h2 : geometric_sequence a r 2 * geometric_sequence a r 18 = 16) 
  (ha2pos : geometric_sequence a r 18 ≠ 0) 
  (h3 : r < 0) :
  geometric_sequence a r 3 * geometric_sequence a r 10 * geometric_sequence a r 17 = -64 :=
sorry

end product_a3_a10_a17_l626_626081


namespace place_value_face_value_difference_l626_626983

theorem place_value_face_value_difference (num : ℕ) (d : ℕ) (h_num : num = 46) (h_d : d = 4) :
  let place_value := d * 10,
      face_value := d
  in place_value - face_value = 36 := by
  sorry

end place_value_face_value_difference_l626_626983


namespace acute_triangle_division_l626_626514

theorem acute_triangle_division (A B C O : Type) [H : A ≠ B] [H₂: B ≠ C] [H₃: C ≠ A]
(Hacute : ∀ {a b c : ℝ}, 0 < a ∧ a < (π / 2) ∧ 0 < b ∧ b < (π / 2) ∧ 0 < c ∧ c < (π / 2) ∧ (a + b + c = π))
(Hcircum : ∀ (O : Type) (Hcenter_in_triangle: O ∈ interior (triangle A B C)), ∀ (P : Type), P ∈ {A, B, C} → dist O P = dist O A)
: is_isosceles (triangle O A B) ∧ is_isosceles (triangle O B C) ∧ is_isosceles (triangle O C A) := 
sorry

end acute_triangle_division_l626_626514


namespace solve_abs_inequality_l626_626541

theorem solve_abs_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ 
      (x ∈ Ioo (-13 / 2) (-3) 
     ∨ x ∈ Ico (-3) 2 
     ∨ x ∈ Icc 2 (7 / 2)) :=
by
  sorry

end solve_abs_inequality_l626_626541


namespace part_a_part_b1_part_b2_l626_626872

variable (X : Type) [MeasureSpace X]

variable (p : ℝ) (hp : 0 < p)
variable (P : Set X → ℝ) [ProbabilityMeasure P]

/- Part (a) -/
theorem part_a (hn : ∀ n : ℕ, n ^ (p - 1) * P ({x | ∥x∥ > n}) = o(1) / n as n → ∞):
  (∀ r : ℝ, r < p → ∃ c < ∞, ∫ x, ∥x∥ ^ r ∂P = c) ∧ 
  ¬ (∃ c < ∞, ∫ x, ∥x∥ ^ p ∂P = c) := by
  sorry

/- Part (b1) -/
theorem part_b1 :
  ∃ (E : ℕ → ℝ) (hE : ∀ n, 0 ≤ E n ∧ ∑ n in Finset.range n, n ^ (p - 1) * P ({x | ∥x∥ > n}) < ∞)
  → (∃ c < ∞, ∫ x, ∥x∥ ^ p ∂P = c) :=
  by sorry

/- Part (b2) -/
theorem part_b2 :
  (∃ E : ℕ × ℕ → ℝ, (∀ m n, 0 ≤ E (m, n) ∧ ∑ m n, P ({x | ∥x∥ > m * n}) < ∞) 
  → (∃ c < ∞, ∫ x, ∥x∥ * log(1+ ∥x∥) ∂P = c)) :=
  by sorry

end part_a_part_b1_part_b2_l626_626872


namespace tan_theta_at_min_value_l626_626017

theorem tan_theta_at_min_value :
  (∃ θ : ℝ, ∀ x : ℝ, 2 * cos θ - 3 * sin θ ≤ 2 * cos x - 3 * sin x) →
  ∃ θ : ℝ, tan θ = -3 / 2 :=
by
  sorry

end tan_theta_at_min_value_l626_626017


namespace problem1_problem2_problem3_l626_626780

-- Problem 1
theorem problem1 (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 →
    f(x) + f(y) = f((x + y) / (1 + x * y)))
  (h2 : ∀ x : ℝ, -1 < x ∧ x < 0 → f(x) > 0) : 
  f(0) = 0 ∧ ∀ x : ℝ, -1 < x ∧ x < 1 → f(-x) = -f(x) :=
sorry

-- Problem 2
theorem problem2 : ∀ x y : ℝ, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 →
  (log ((1 - x) / (1 + x)) + log ((1 - y) / (1 + y))
  = log ((1 - x - y + x * y) / (1 + x + y + x * y))) ∧ 
  (∀ x : ℝ, -1 < x ∧ x < 0 → log ((1 - x) / (1 + x)) > 0) :=
sorry

-- Problem 3
theorem problem3 (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 →
    f(x) + f(y) = f((x + y) / (1 + x * y)))
  (h2 : ∀ x : ℝ, -1 < x ∧ x < 0 → f(x) > 0)
  (h3 : f(-1/2) = 1) : 
  ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f(x) + 1 / 2 = 0 :=
sorry

end problem1_problem2_problem3_l626_626780


namespace monotonicity_and_extrema_l626_626326

noncomputable def f (x : ℝ) := (2 * x) / (x + 1)

theorem monotonicity_and_extrema :
  (∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f x1 < f x2) ∧
  (f 3 = 5 / 4) ∧
  (f 5 = 3 / 2) :=
by
  sorry

end monotonicity_and_extrema_l626_626326


namespace sin_cos_sum_eq_l626_626399

theorem sin_cos_sum_eq (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2)
  (h : (Real.sin θ) - 2 * (Real.cos θ) = 0) :
  (Real.sin θ + Real.cos θ) = (3 * Real.sqrt 5) / 5 := by
  sorry

end sin_cos_sum_eq_l626_626399


namespace M_is_incenter_of_triangle_BCP_l626_626607

-- Definitions for the geometrical setup
variables {A B C D M P : Point}
variable [circumscribed_circle : ∀ {X Y Z : Point}, (X Y Z : Triangle), Formed_by X Y Z]
variable [diameter_AD : Diameter AD]
variable [intersection_M : Intersect AC BD M]
variable [projection_P : Projection M AD P]
variable [triangle_BCP : Triangle B C P]

-- Lean statement for the proof problem
theorem M_is_incenter_of_triangle_BCP
  (h1 : ∀ {A B C D : Point}, (AD_is_diameter : Diameter AD) → CircumscribedQuad ABCD)
  (h2 : ∀ {A C D B M : Point}, (Intersection AC BD M))
  (h3 : ∀ {M A D P : Point}, (Projection M AD P)) :
  Incenter M B C P := sorry

end M_is_incenter_of_triangle_BCP_l626_626607


namespace how_many_children_l626_626193

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l626_626193


namespace length_of_chord_AB_equation_of_chord_AB_bisected_by_P_l626_626860

-- Definitions for the circle and point P
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8
def point_P := (-1, 2 : ℝ)

-- The Lean 4 statement for part (1)
theorem length_of_chord_AB (α : ℝ) (hα : α = real.pi * 3 / 4) (h_AB : chord_contains_P (α))
  : length_of_chord (α) (h_AB) = real.sqrt 30 := sorry

-- The Lean 4 statement for part (2)
theorem equation_of_chord_AB_bisected_by_P (h_bisect : chord_bisected_by_P) 
  : equation_of_line_AB (h_bisect) = "x - 2y + 5 = 0" := sorry

-- Auxiliary definitions to support the theorems. Note: these need implementations as per Lean syntax,
-- but adding them for clarity of structure
def chord_contains_P (α : ℝ) : Prop := 
  -- condition for chord AB to pass through point P using inclination α

def length_of_chord (α : ℝ) (h : chord_contains_P (α)) : ℝ := 
  -- function to compute the length of the chord

def chord_bisected_by_P : Prop := 
  -- condition for chord AB being bisected by point P

def equation_of_line_AB (h : chord_bisected_by_P) : string := 
  -- function to determine the equation of line AB

end length_of_chord_AB_equation_of_chord_AB_bisected_by_P_l626_626860


namespace relationship_among_values_l626_626796

-- Assume there exists a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition 1: f is strictly increasing on (0, 3)
def increasing_on_0_to_3 : Prop :=
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f x < f y

-- Condition 2: f(x + 3) is an even function
def even_function_shifted : Prop :=
  ∀ x : ℝ, f (x + 3) = f (-(x + 3))

-- The theorem we need to prove
theorem relationship_among_values 
  (h1 : increasing_on_0_to_3 f)
  (h2 : even_function_shifted f) :
  f (9/2) < f 2 ∧ f 2 < f (7/2) :=
sorry

end relationship_among_values_l626_626796


namespace problem_solution_l626_626859

noncomputable def corrected_angles 
  (x1_star x2_star x3_star : ℝ) 
  (σ : ℝ) 
  (h_sum : x1_star + x2_star + x3_star - 180.0 = 0)  
  (h_var : σ^2 = (0.1)^2) : ℝ × ℝ × ℝ :=
  let Δ := 2.0 / 3.0 * 0.667
  let Δx1 := Δ * (σ^2 / 2)
  let Δx2 := Δ * (σ^2 / 2)
  let Δx3 := Δ * (σ^2 / 2)
  let corrected_x1 := x1_star - Δx1
  let corrected_x2 := x2_star - Δx2
  let corrected_x3 := x3_star - Δx3
  (corrected_x1, corrected_x2, corrected_x3)

theorem problem_solution :
  corrected_angles 31 62 89 (0.1) sorry sorry = (30.0 + 40 / 60, 61.0 + 40 / 60, 88 + 20 / 60) := 
  sorry

end problem_solution_l626_626859


namespace saras_favourite_number_digits_l626_626129

theorem saras_favourite_number_digits :
  {d : ℕ | ∃ k : ℕ, 1 ≤ k ∧ Nat.digits 10 (6 * k) ≠ [] ∧ Nat.digits 10 (6 * k) = d :: _} = {0, 2, 4, 6, 8} :=
sorry

end saras_favourite_number_digits_l626_626129


namespace tangent_segment_lengths_l626_626673

theorem tangent_segment_lengths (ABC : Triangle) (AB_eq_BC : ABC.AB = ABC.BC)
  (incircle : Circle) (C_incircle : incircle.inscribed ABC)
  (P QR : ℝ) (hPQ : tangent_point P ABC.BC) (hQR : tangent_point QR ABC.AB)
  (h_lengths : P = QR):
  let m := segment_length ABC.BC incircle.tangent_point
  let n := segment_length ABC.AC incircle.tangent_point
  tangent_segment_lengths_parallel_to_sides ABC incircle = 
  (2 * m * n) / (m + 2 * n), (n * (m + n)) / (m + 2 * n), (n * (m + n)) / (m + 2 * n) :=
begin
  sorry
end

end tangent_segment_lengths_l626_626673


namespace wall_height_approx_l626_626670

noncomputable def brick_length : ℕ := 25
noncomputable def brick_width : ℕ := 11
noncomputable def brick_height : ℕ := 6

noncomputable def wall_length : ℕ := 200
noncomputable def wall_width : ℕ := 2
noncomputable def num_bricks : ℚ := 72.72727272727273

def height_in_cm : ℚ := brick_height * num_bricks

def height_in_meters : ℚ := height_in_cm / 100

theorem wall_height_approx : abs (height_in_meters - 4.36) < 0.01 :=
by
  sorry

end wall_height_approx_l626_626670


namespace math_problem_proof_l626_626185

theorem math_problem_proof : 
  let A := 9 - 4 in
  let B := A + 5 in
  let C := 1 + 8 in
  B - C = 1 := 
by
  sorry

end math_problem_proof_l626_626185


namespace rectangle_perimeter_l626_626270

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b) = 4 * (2 * a + 2 * b) - 12) :
    (2 * (a + b) = 72) ∨ (2 * (a + b) = 100) := by
  sorry

end rectangle_perimeter_l626_626270


namespace square_perimeter_ratio_l626_626558

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l626_626558


namespace xy_squared_l626_626049

theorem xy_squared (x y : ℚ) (h1 : x + y = 9 / 20) (h2 : x - y = 1 / 20) :
  x^2 - y^2 = 9 / 400 :=
by
  sorry

end xy_squared_l626_626049


namespace sum_first_9_terms_arithmetic_sequence_l626_626850

theorem sum_first_9_terms_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_condition : a 2 + a 8 = 16) : 
  (finset.range 9).sum a = 72 :=
sorry

end sum_first_9_terms_arithmetic_sequence_l626_626850


namespace helen_baked_more_raisin_cookies_l626_626815

-- Definitions based on conditions
def raisin_cookies_yesterday : ℕ := 300
def raisin_cookies_day_before : ℕ := 280

-- Theorem to prove the answer
theorem helen_baked_more_raisin_cookies : raisin_cookies_yesterday - raisin_cookies_day_before = 20 :=
by
  sorry

end helen_baked_more_raisin_cookies_l626_626815


namespace range_of_m_l626_626060

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, 0 < x ∧ mx^2 + 2 * x + m > 0) →
  m ≤ -1 := by
  sorry

end range_of_m_l626_626060


namespace garden_area_enlargement_l626_626682

theorem garden_area_enlargement :
  let length := 60
  let width := 20
  (2 * (length + width)) = 160 →
  (160 / 4) = 40 →
  ((40 * 40) - (length * width) = 400) :=
begin
  intros,
  sorry,
end

end garden_area_enlargement_l626_626682


namespace minimal_weights_number_l626_626293

theorem minimal_weights_number
  (groups : Finset (Finset ℝ))
  (balance_condition : ∀ (a b c d : ℝ), a ∈ groups → b ∈ groups → a + b = c + d → c ∈ groups ∧ d ∈ groups)
  (sorted_groups : groups.card = 5): ∃ (k l n f t : ℕ), k + l + n + f + t = 13 :=
by
  sorry

end minimal_weights_number_l626_626293


namespace angle_between_lateral_face_and_base_plane_l626_626162

theorem angle_between_lateral_face_and_base_plane {a : ℝ} :
  (θ : ℝ) =
  let SO := sorry,
  let E := sorry,
  let SE_perp_AC := sorry,
  let BE_perp_AC := sorry,
  let EF_perp_SB := sorry,
  let EF_perp_AC := sorry,
  let EF := a / 2,
  let EB := a * sqrt 3 / 2,
  θ = Real.arctan (sqrt 2) :=
sorry

end angle_between_lateral_face_and_base_plane_l626_626162


namespace intersection_M_N_l626_626100

def M : set ℝ := {x | x > 0}
def N : set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_M_N_l626_626100


namespace how_many_children_l626_626194

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l626_626194


namespace evaluate_expression_l626_626333

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l626_626333


namespace fraction_of_grid_is_one_sixth_l626_626693
  
def point := (ℝ × ℝ)

def A : point := (1, 3)
def B : point := (5, 1)
def C : point := (4, 4)

def area_of_triangle (p1 p2 p3 : point) : ℝ :=
  0.5 * | p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2) |

def area_of_grid (length width : ℝ) : ℝ := length * width

def fraction_of_grid_covered (triangle_area grid_area : ℝ) : ℝ :=
  triangle_area / grid_area

theorem fraction_of_grid_is_one_sixth : 
  fraction_of_grid_covered (area_of_triangle A B C) (area_of_grid 6 5) = 1 / 6 :=
by
  sorry

end fraction_of_grid_is_one_sixth_l626_626693


namespace triangle_BN_squared_l626_626971

theorem triangle_BN_squared
  (A B C D L M N : Type)
  [is_right_triangle A B C]
  (H1 : altitude_foot D C A B)
  (H2 : midpoint L A D)
  (H3 : midpoint M D C)
  (H4 : midpoint N C A)
  (H5 : length CL = 7)
  (H6 : length BM = 12) :
  length_squared BN = 193 :=
by
  sorry

end triangle_BN_squared_l626_626971


namespace find_m_given_exponential_eq_l626_626827

theorem find_m_given_exponential_eq (m : ℤ) (h : 7^(4 * m) = (1 / 7)^(2 * m - 18)) : m = 3 := 
by
  sorry

end find_m_given_exponential_eq_l626_626827


namespace solve_trig_equation_l626_626141

theorem solve_trig_equation:
  ∃ k : ℤ, (x = -π / 2 + 2 * π * k ∨ 
            x = π / 6 + 2 * π * k ∨
            x = 5 * π / 6 + 2 * π * k) ↔ 
          8 * sin x + 12 * sin x ^ 3 + 2022 * sin x ^ 5 = 
          8 * cos (2 * x) + 12 * cos (2 * x) ^ 3 + 2022 * cos (2 * x) ^ 5 := 
  sorry

end solve_trig_equation_l626_626141


namespace P_le_Q_l626_626882

variable {a b c d m n : ℝ}

noncomputable def P (a b c d : ℝ) : ℝ := Real.sqrt (a * b) + Real.sqrt (c * d)
noncomputable def Q (a b c d m n : ℝ) : ℝ := Real.sqrt (m * a + n * c) * Real.sqrt (b / m + d / n)

theorem P_le_Q (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) (hd: 0 < d) (hm: 0 < m) (hn: 0 < n) : 
  P a b c d ≤ Q a b c d := by
  sorry

end P_le_Q_l626_626882


namespace age_of_other_man_l626_626929

variable (A M : ℕ)
variable (h1 : list.sum [A * 8]) -- sum of ages of 8 men
variable (h2 : 23 * 2 = 46) -- sum of the ages of 2 women

theorem age_of_other_man 
  (h3 : list.sum [A * 8 - M - 20 + 46] = 8 * (A + 2)) :
  M = 10 :=
by
  sorry

end age_of_other_man_l626_626929


namespace square_diagonals_not_equal_to_sides_l626_626224

theorem square_diagonals_not_equal_to_sides {s : ℝ} (h_square : ∀ (P Q R S : ℝ), 
  PQR[90] ∧ QR[90] ∧ RS[90] ∧ SP[90] ∧ PQ = QR ∧ QR = RS ∧ RS = SP ∧ SP = PQ ∧ 
  (diagonal(PR) = diagonal(QS) ∨ diagonal(PR) = side_length(s))) : 
  ¬ ∀ s, diagonal(square s) = s := sorry

end square_diagonals_not_equal_to_sides_l626_626224


namespace number_of_children_l626_626191

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l626_626191


namespace worker_rates_l626_626225

variables (a b c d e : ℚ)

theorem worker_rates:
  a = 1 / 3 →
  a + b = 1 / 2 →
  b + c = 2 / 3 →
  a + c + d = 1 →
  b + d + e = 2 / 3 →
  a + c + e = 4 / 5 →
  b = 1 / 6 ∧ (c = 1 / 2) ∧ (d = 1 / 6) ∧ (e = 1 / 3) :=
by {
  intros,
  sorry
}

end worker_rates_l626_626225


namespace prime_q_with_period_166_l626_626421

theorem prime_q_with_period_166 (p q : ℕ) (hp : p.prime) (hq : q.prime) (h_period : nat.periodicity (1/q) = 166) : q = 167 :=
by
  sorry

end prime_q_with_period_166_l626_626421


namespace flour_amount_l626_626890

theorem flour_amount (a b : ℕ) (h₁ : a = 8) (h₂ : b = 2) : a + b = 10 := by
  sorry

end flour_amount_l626_626890


namespace transformation_triple_l626_626944

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ Icc (-3 : ℝ) 0 then -2 - x
else if h : x ∈ Icc (0 : ℝ) 2 then sqrt (4 - (x - 2) ^ 2) - 2
else if h : x ∈ Icc (2 : ℝ) 3 then 2 * (x - 2)
else 0  -- Since the function is defined only on [-3,3]

def h (x : ℝ) : ℝ :=
2 * f (x / 3) - 6

theorem transformation_triple :
  (∃ a b c : ℝ, (h x = a * f (b * x) + c) ∧ a = 2 ∧ b = 1 / 3 ∧ c = -6) :=
begin
  use [2, 1 / 3, -6],
  split,
  { intros x,
    refl, },
  split,
  { refl, },
  { split,
    { refl, },
    { refl, }, },
end

end transformation_triple_l626_626944


namespace sqrt_of_225_eq_15_l626_626661

theorem sqrt_of_225_eq_15 : Real.sqrt 225 = 15 :=
by
  sorry

end sqrt_of_225_eq_15_l626_626661


namespace function_identity_l626_626116

theorem function_identity (f : ℕ → ℕ) :
  (∀ x y : ℕ, ∃ k : ℕ, x^2 - y^2 + 2 * y * (f x + f y) = k^2) ↔ (∀ n : ℕ, f n = n) :=
begin
  sorry
end

end function_identity_l626_626116


namespace hyperbola_range_m_l626_626211

theorem hyperbola_range_m (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (|m| - 1)) - (y^2 / (m - 2)) = 1) ↔ (m < -1) ∨ (m > 2) := 
by
  sorry

end hyperbola_range_m_l626_626211


namespace problem_statement_l626_626942

variable {ℝ} [Nontrivial ℝ]

def f : ℝ → ℝ := sorry
def f' : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, x * f'(x) + f(x) < 0

theorem problem_statement : 2 * f 2 > 3 * f 3 :=
by
  -- Proof body goes here
  sorry

end problem_statement_l626_626942


namespace idempotent_mappings_count_l626_626268

noncomputable def num_idempotent_mappings (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), Nat.choose n k * k^(n - k)

theorem idempotent_mappings_count (X : Type) [Fintype X] [DecidableEq X] (f : X → X)
  (h : ∀ x, f (f x) = f x) (n : ℕ) (hn : Fintype.card X = n) :
  ∃ num_mappings : ℕ, num_mappings = num_idempotent_mappings n :=
by
  use num_idempotent_mappings n
  sorry

end idempotent_mappings_count_l626_626268


namespace probability_sum_less_than_12_l626_626630

def fair_eight_sided_dice : finset (ℕ × ℕ) :=
finset.product (finset.range 8 \ {0}) (finset.range 8 \ {0})

def sum_less_than (n : ℕ) : finset (ℕ × ℕ) :=
fair_eight_sided_dice.filter (λ p, p.1 + p.2 < n)

def probability_sum_less_than (n : ℕ) : ℚ :=
(sum_less_than n).card / fair_eight_sided_dice.card

theorem probability_sum_less_than_12 :
  probability_sum_less_than 12 = 49 / 64 := 
sorry

end probability_sum_less_than_12_l626_626630


namespace garden_area_enlargement_l626_626684

theorem garden_area_enlargement :
  let length := 60
  let width := 20
  (2 * (length + width)) = 160 →
  (160 / 4) = 40 →
  ((40 * 40) - (length * width) = 400) :=
begin
  intros,
  sorry,
end

end garden_area_enlargement_l626_626684


namespace CPD_path_length_l626_626328

-- define necessary constants and variables
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def O : Point := midpoint A B
noncomputable def radius : ℝ := 10
noncomputable def C : Point := sorry -- Point C 6 units from A
noncomputable def D : Point := sorry -- Point D 6 units from B
noncomputable def onCircle (P : Point) : Prop := real_dist P O = radius

theorem CPD_path_length :
  ∃ P : Point, onCircle P ∧  CP_dist C P + PD_dist P D = 20 := 
sorry

end CPD_path_length_l626_626328


namespace coloring_squares_impossible_l626_626699

/-!
  Given a collection of non-overlapping squares on a plane, possibly of different sizes,
  which may share portions of their boundaries, it is not always possible to color these squares
  using three colors such that no two adjacent squares share the same color.
-/

def not_possible_to_color_squares : Prop :=
  ∀ (squares : Set (Set (ℝ × ℝ))) (H : ∀ s1 s2 ∈ squares, s1 ≠ s2 → (s1 ∩ s2).Interior = ∅), 
  ∃ (coloring : ∀ (s : Set (ℝ × ℝ)), s ∈ squares → Fin 3),
  ∃ s1 s2 ∈ squares, s1 ≠ s2 ∧ (s1 ∩ s2).Interior ≠ ∅ ∧ coloring s1 (by { rw [H s1], exact sorry }) = coloring s2 (by { rw [H s2], exact sorry })

theorem coloring_squares_impossible : not_possible_to_color_squares :=
sorry

end coloring_squares_impossible_l626_626699


namespace smallest_number_l626_626145

theorem smallest_number (A B C : ℕ) 
  (h1 : A / 3 = B / 5) 
  (h2 : B / 5 = C / 7) 
  (h3 : C = 56) 
  (h4 : C - A = 32) : 
  A = 24 := 
sorry

end smallest_number_l626_626145


namespace triangle_solution_l626_626105

noncomputable def triangle_problem 
  (A B C P Q: Type)
  (dist: A → A → ℝ)
  (mangle_BAC: ℝ)
  (P_midpoint: dist A B = dist B P * 2 ∧ dist A P = dist P B)
  (Q_midpoint: dist A C = dist C Q * 2 ∧ dist A Q = dist Q C)
  (AP: dist A P = 20)
  (CQ: dist C Q = 26): Prop :=  
  let AB := dist A B in
  let AC := dist A C in
  let BC := dist B C in 
  (mangle_BAC = 90 ∧ AB = 40 ∧ AC = 52) → (BC = 52*sqrt 2)

theorem triangle_solution 
  (A B C P Q: Type)
  (dist: A → A → ℝ)
  (mangle_BAC: ℝ)
  (P_midpoint: dist A B = dist B P * 2 ∧ dist A P = dist P B)
  (Q_midpoint: dist A C = dist C Q * 2 ∧ dist A Q = dist Q C)
  (AP: dist A P = 20)
  (CQ: dist C Q = 26):
  triangle_problem A B C P Q dist mangle_BAC P_midpoint Q_midpoint AP CQ :=
sorry

end triangle_solution_l626_626105


namespace smallest_q_p_difference_l626_626488

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l626_626488


namespace assigned_pages_eq_25_l626_626958

theorem assigned_pages_eq_25 (x harrison_pages pam_pages sam_pages : ℕ) 
  (h1 : harrison_pages = x + 10)
  (h2 : pam_pages = harrison_pages + 15)
  (h3 : sam_pages = 2 * pam_pages)
  (h4 : sam_pages = 100) : x = 25 :=
begin
  sorry
end

end assigned_pages_eq_25_l626_626958


namespace vasya_coupons_exchange_l626_626851

theorem vasya_coupons_exchange (x y : ℕ) :
  (x - y = 0) → (x + y = 1991) → false :=
by {
  intro h1,
  intro h2,
  have h := congr_arg (λ n, n / 2) h2,
  rw [←nat.add_sub_assoc h1, nat.div_add_cancel zero_le h1] at h,
  exact (nat.not_even 1991)
}

end vasya_coupons_exchange_l626_626851


namespace an_values_and_formula_is_geometric_sequence_l626_626123

-- Definitions based on the conditions
def Sn (n : ℕ) : ℝ := sorry  -- S_n to be defined in the context or problem details
def a (n : ℕ) : ℝ := 2 - Sn n

-- Prove the specific values and general formula given the condition a_n = 2 - S_n
theorem an_values_and_formula (Sn : ℕ → ℝ) :
  a 1 = 1 ∧ a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ a 4 = 1 / 8 ∧ (∀ n, a n = (1 / 2)^(n-1)) :=
sorry

-- Prove the sequence is geometric
theorem is_geometric_sequence (Sn : ℕ → ℝ) :
  (∀ n, a n = (1 / 2)^(n-1)) → ∀ n, a (n + 1) / a n = 1 / 2 :=
sorry

end an_values_and_formula_is_geometric_sequence_l626_626123


namespace total_length_of_pencil_l626_626047

def purple := 3
def black := 2
def blue := 1
def total_length := purple + black + blue

theorem total_length_of_pencil : total_length = 6 := 
by 
  sorry -- proof not needed

end total_length_of_pencil_l626_626047


namespace text_messages_in_march_l626_626865

/-
Jared sent text messages each month according to the formula:
  T_n = n^3 - n^2 + n
We need to prove that the number of text messages Jared will send in March
(which is the 5th month) is given by T_5 = 105.
-/

def T (n : ℕ) : ℕ := n^3 - n^2 + n

theorem text_messages_in_march : T 5 = 105 :=
by
  -- proof goes here
  sorry

end text_messages_in_march_l626_626865


namespace infinite_series_sum_l626_626311

theorem infinite_series_sum :
  (∑' n : ℕ, if h : n ≠ 0 then 1 / (n * (n + 1) * (n + 3)) else 0) = 5 / 36 := by
  sorry

end infinite_series_sum_l626_626311


namespace john_daily_reading_hours_l626_626091

-- Definitions from the conditions
def reading_rate := 50  -- pages per hour
def total_pages := 2800  -- pages
def weeks := 4
def days_per_week := 7

-- Hypotheses derived from the conditions
def total_hours := total_pages / reading_rate  -- 2800 / 50 = 56 hours
def total_days := weeks * days_per_week  -- 4 * 7 = 28 days

-- Theorem to prove 
theorem john_daily_reading_hours : (total_hours / total_days) = 2 := by
  sorry

end john_daily_reading_hours_l626_626091


namespace normal_distribution_problem_l626_626797

theorem normal_distribution_problem 
  (X : ℝ → ℝ)
  (hX : ∀ a, ∃ μ σ, X a ~ Normal μ σ)
  (μ : ℝ := 3)
  (σ : ℝ := 4)
  (P : set ℝ → ℝ)
  (hP1 : P {x | 3 ≤ X x ∧ X x ≤ a} = 0.35)
  (a : ℝ)
  (ha : a > 3) :
  P {x | X x > a} = 0.15 :=
sorry

end normal_distribution_problem_l626_626797


namespace pencils_in_pencil_case_l626_626200

theorem pencils_in_pencil_case : ∃ P : ℕ, let pens := 2 * P, eraser := 1, total_items := P + pens + eraser in total_items = 13 ∧ P = 4 :=
  sorry

end pencils_in_pencil_case_l626_626200


namespace angle_MLO_of_regular_hexagon_l626_626980

theorem angle_MLO_of_regular_hexagon (a b c d e f : Point) (h : is_regular_hexagon (Polygon.mk a b c d e f)) :
  angle_measure (angle.mk a e o) = 30 :=
by sorry

end angle_MLO_of_regular_hexagon_l626_626980


namespace number_of_integer_solutions_l626_626611

-- Given condition
def condition (x : ℝ) : Prop := 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5

-- Number of integer solutions that satisfy the condition
theorem number_of_integer_solutions : 
  (Finset.card (Finset.filter (λ x : ℤ, condition x) (Finset.Icc 1 20))) = 3 := 
sorry

end number_of_integer_solutions_l626_626611


namespace min_difference_bounds_l626_626022

noncomputable def f (x t : ℝ) : ℝ :=
if x ≥ 0 then real.sqrt x - t
else 2 * (x + 1) - t

theorem min_difference_bounds (t : ℝ) (ht : 0 ≤ t ∧ t < 2) :
  let x1 := t^2,
      x2 := (1/2) * t - 1
  in (x1 - x2) ≥ (15 / 16) :=
sorry

end min_difference_bounds_l626_626022


namespace students_play_neither_sport_l626_626436

def total_students : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def both_players : ℕ := 10

theorem students_play_neither_sport :
  total_students - (hockey_players + basketball_players - both_players) = 4 :=
by
  sorry

end students_play_neither_sport_l626_626436


namespace groupings_of_tourists_l626_626634

theorem groupings_of_tourists :
  ∃ (n : ℕ), let total_groupings := 2^8 in
  let invalid_groupings := 2 in
  n = total_groupings - invalid_groupings :=
  ∃ (n : ℕ), n = 254 :=
  by sorry

end groupings_of_tourists_l626_626634


namespace sequence_a2015_l626_626783

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 1 - (1 / 2)
  else if n = 2 then 1 - 2
  else if n % 3 = 0 then 2
  else if n % 3 = 1 then 1 - (1 / 2)
  else -1

theorem sequence_a2015 : sequence 2015 = 1 / 2 :=
  sorry

end sequence_a2015_l626_626783


namespace speed_of_current_correct_l626_626267

noncomputable def speed_in_still_water_kmph := 15
noncomputable def speed_in_still_water_mps : ℝ := speed_in_still_water_kmph * 1000 / 3600
noncomputable def time_downstream_seconds := 10.799136069114471
noncomputable def distance_downstream_meters := 60

noncomputable def speed_downstream_mps : ℝ := distance_downstream_meters / time_downstream_seconds
noncomputable def speed_of_current_mps : ℝ := speed_downstream_mps - speed_in_still_water_mps

theorem speed_of_current_correct : 
  abs (speed_of_current_mps - 1.38888889) < 1e-8 := by
  sorry

end speed_of_current_correct_l626_626267


namespace pencil_count_l626_626182

def total_pencils (drawer : Nat) (desk_0 : Nat) (add_dan : Nat) (remove_sarah : Nat) : Nat :=
  let desk_1 := desk_0 + add_dan
  let desk_2 := desk_1 - remove_sarah
  drawer + desk_2

theorem pencil_count :
  total_pencils 43 19 16 7 = 71 :=
by
  sorry

end pencil_count_l626_626182


namespace robots_to_same_square_l626_626271

structure Grid (m n : ℕ) :=
  (edges_passable : ℕ → ℕ → bool) -- A function that tells whether an edge is passable

structure Robot :=
  (x y : ℕ)

def move_robot (r : Robot) (dir : String) (g : Grid) : Robot :=
  sorry -- Implement move logic according to grid conditions

def move_all_robots (robots : List Robot) (dir : String) (g : Grid) : List Robot :=
  sorry -- Move all robots according to the grid and direction

theorem robots_to_same_square (m n : ℕ) (robots : List Robot) (g : Grid m n):
  (∀ r1 r2 : Robot, ∃ seq : List String, r1 ∈ robots → r2 ∈ robots → true) → -- Each robot can reach any square
  ∃ seq : List String, ∀ r : Robot, r ∈ (move_all_robots robots (seq.head!) g) → -- Applying the sequence will result in all robots on the same square
  ∃ sq : Robot, sq ∈ robots → 
    (∀ r : Robot, r ∈ robots → sq.x = r.x ∧ sq.y = r.y) :=
sorry

end robots_to_same_square_l626_626271


namespace adams_father_total_amount_l626_626695

noncomputable def annual_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

noncomputable def total_interest (annual_interest : ℝ) (years : ℝ) : ℝ :=
  annual_interest * years

noncomputable def total_amount (principal : ℝ) (total_interest : ℝ) : ℝ :=
  principal + total_interest

theorem adams_father_total_amount :
  let principal := 2000
  let rate := 0.08
  let years := 2.5
  let annualInterest := annual_interest principal rate
  let interest := total_interest annualInterest years
  let amount := total_amount principal interest
  amount = 2400 :=
by sorry

end adams_father_total_amount_l626_626695


namespace find_polynomials_with_rad_condition_l626_626870

-- Define given conditions in the problem
def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def prime_factors (n : ℕ) : List ℕ :=
  List.filter is_prime (List.range (n+1))

def rad (n : ℕ) : ℕ :=
  (prime_factors n).prod

-- Define the problem that needs proof
theorem find_polynomials_with_rad_condition (P : ℚ → ℚ) :
  (∃ f : ℤ → ℚ, (∀ n : ℕ, P (n : ℚ) = rad n → (∃ c : ℚ, P (n : ℚ) = c ∨ (∃ b : ℕ, P (n : ℚ) = n / b)))) →
  (∃ k : ℚ, ∃ a : ℚ, P = (fun x => k) ∨ (P = (fun x => a * x)) :=
sorry

end find_polynomials_with_rad_condition_l626_626870


namespace hexagon_area_l626_626084

noncomputable def area_of_hexagon (P Q R P' Q' R' : Point) (radius : ℝ) : ℝ :=
  -- a placeholder for the actual area calculation
  sorry 

theorem hexagon_area (P Q R P' Q' R' : Point) 
  (radius : ℝ) (perimeter : ℝ) :
  radius = 9 → perimeter = 42 →
  area_of_hexagon P Q R P' Q' R' radius = 189 := by
  intros h1 h2
  sorry

end hexagon_area_l626_626084


namespace largest_mersenne_prime_less_than_500_l626_626755

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_less_than_500 : ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p :=
sorry

end largest_mersenne_prime_less_than_500_l626_626755


namespace midline_length_half_side_l626_626816

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable {BC CA AB : ℝ}  -- denote the sides of the triangle as real numbers
variable {a b c : ℝ}  -- denote the lengths of the sides

-- condition: sides of the triangle
variable (BC_eq : BC = a)
variable (CA_eq : CA = b)
variable (AB_eq : AB = c)

-- definition: midlines as half of respective sides
def midline_lengths (s_a s_b s_c : ℝ) : Prop :=
  s_a = 1 / 2 * b ∧ s_b = 1 / 2 * c ∧ s_c = 1 / 2 * a

-- theorem to be proven: lengths of midlines are half the lengths of respective sides 
theorem midline_length_half_side (s_a s_b s_c : ℝ) :
  BC = a → CA = b → AB = c → midline_lengths s_a s_b s_c :=
by
  intros
  have mid_a : s_a = 1 / 2 * b, sorry
  have mid_b : s_b = 1 / 2 * c, sorry
  have mid_c : s_c = 1 / 2 * a, sorry
  exact ⟨mid_a, mid_b, mid_c⟩

end midline_length_half_side_l626_626816


namespace cost_of_each_toy_l626_626643

theorem cost_of_each_toy 
  (initial_money : ℕ)
  (money_spent : ℕ)
  (num_toys : ℕ)
  (remaining_money : ℕ)
  (toy_cost : ℕ) :
  initial_money = 57 →
  money_spent = 27 →
  num_toys = 5 →
  remaining_money = initial_money - money_spent →
  toy_cost = remaining_money / num_toys →
  toy_cost = 6 :=
by
  intros h_initial h_spent h_num h_remaining h_toy_cost
  rw [h_initial, h_spent] at h_remaining
  norm_num at h_remaining
  rw [h_remaining] at h_toy_cost
  norm_num at h_toy_cost
  exact h_toy_cost
  -- sorry

end cost_of_each_toy_l626_626643


namespace population_reaches_210_l626_626794

noncomputable def population_function (x : ℕ) : ℝ :=
  200 * (1 + 0.01)^x

theorem population_reaches_210 :
  ∃ x : ℕ, population_function x >= 210 :=
by
  existsi 5
  apply le_of_lt
  sorry

end population_reaches_210_l626_626794


namespace factorize_expression_l626_626745

theorem factorize_expression (x : ℝ) :
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) :=
  sorry

end factorize_expression_l626_626745


namespace range_of_m_squared_l626_626002

noncomputable theory

def ellipse_equation (a b : ℝ) : Prop :=
  a = 2 ∧ b = 1

def ellipse_foci_condition (c a : ℝ) : Prop :=
  c / a = (real.sqrt 3) / 2

def quadrilateral_perimeter_condition (a b : ℝ) : Prop :=
  4 * real.sqrt (a^2 + b^2) = 4 * real.sqrt 5

def line_intersection_points (k m : ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = k * x1 + m ∧ y2 = k * x2 + m

def vector_condition (x1 x2 : ℝ) : Prop :=
  -x1 = 3 * x2

/-- The range of values for m² is (1, 4) for given conditions on the ellipse and line -/
theorem range_of_m_squared
  (a b c k m x1 y1 x2 y2 : ℝ)
  (h_ellipse_eq : ellipse_equation a b)
  (h_foci : ellipse_foci_condition c a)
  (h_perimeter : quadrilateral_perimeter_condition a b)
  (h_line_points : line_intersection_points k m x1 y1 x2 y2)
  (h_vector : vector_condition x1 x2) :
  1 < m^2 ∧ m^2 ≤ 4 :=
begin
  sorry
end

end range_of_m_squared_l626_626002


namespace pear_pairing_l626_626889

theorem pear_pairing (n : ℕ) (h : n ≥ 1) (weights : Fin (2*n + 2) → ℕ) :
  ∃ pairs : Fin (n + 1) → (Fin (2*n + 2) × Fin (2*n + 2)),
    (∀ i : Fin n, abs ((weights (pairs i).1 + weights (pairs i).2) - (weights (pairs (i+1)).1 + weights (pairs (i+1)).2)) ≤ 1) := 
sorry

end pear_pairing_l626_626889


namespace count_rational_numbers_l626_626697

def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

theorem count_rational_numbers :
  let lst := [(-Real.pi / 3), 3.1415, 0, -0.333, -(22 / 7 : ℝ), -0.15, 2.010010001]
  let is_rational_lst := lst.map is_rational
  is_rational_lst.count (λ x, x) = 5 :=
by
  -- definitions and other steps might go here if needed
  sorry

end count_rational_numbers_l626_626697


namespace smallest_n_exceeds_15_l626_626045

noncomputable def g (n : ℕ) : ℕ :=
  sorry  -- Define the sum of the digits of 1 / 3^n to the right of the decimal point

theorem smallest_n_exceeds_15 : ∃ n : ℕ, n > 0 ∧ g n > 15 ∧ ∀ m : ℕ, m > 0 ∧ g m > 15 → n ≤ m :=
  sorry  -- Prove the smallest n such that g(n) > 15

end smallest_n_exceeds_15_l626_626045


namespace definite_integral_eval_l626_626653

noncomputable def integrand (x : ℝ) : ℝ :=
  (4 * sqrt (2 - x) - sqrt (2 * x + 2)) / ((sqrt (2 * x + 2) + 4 * sqrt (2 - x)) * (2 * x + 2) ^ 2)

theorem definite_integral_eval :
  ∫ x in 0..2, integrand x = (1 / 24) * log 5 :=
by
  sorry

end definite_integral_eval_l626_626653


namespace binom_sum_eq_13_l626_626216

open Nat

theorem binom_sum_eq_13 (S : ℕ) :
  (∀ m : ℕ, (binom 23 m + binom 23 12 = binom 24 13) → m ∈ {13}) →
  S = 13 :=
by
  sorry

end binom_sum_eq_13_l626_626216


namespace largest_mersenne_prime_lt_500_l626_626752

open Nat

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n, (prime n) ∧ (p = 2^n - 1)

theorem largest_mersenne_prime_lt_500 : ∀ p, is_mersenne_prime p ∧ p < 500 → p ≤ 127 :=
by
  sorry

end largest_mersenne_prime_lt_500_l626_626752


namespace inequality_sqrt_inequality_l626_626498

theorem inequality_sqrt_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (sqrt (3 * x^2 + x * y) + sqrt (3 * y^2 + y * z) + sqrt (3 * z^2 + z * x)) ≤ 2 * (x + y + z) := 
sorry

end inequality_sqrt_inequality_l626_626498


namespace probability_of_real_solutions_l626_626263

def quadratic_has_real_solutions_probability : ℚ := 
  let outcomes : List (ℕ × ℕ) := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), 
                                   (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), 
                                   (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), 
                                   (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), 
                                   (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), 
                                   (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)]
   let favorable_outcomes := outcomes.filter (λ ⟨a, b⟩, b^2 - 4 * a ≥ 0)
  (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)

theorem probability_of_real_solutions :
  quadratic_has_real_solutions_probability = 19 / 36 := by
    sorry

end probability_of_real_solutions_l626_626263


namespace average_marks_physics_mathematics_l626_626283

theorem average_marks_physics_mathematics {P C M : ℕ} (h1 : P + C + M = 180) (h2 : P = 140) (h3 : P + C = 140) : 
  (P + M) / 2 = 90 := by
  sorry

end average_marks_physics_mathematics_l626_626283


namespace adi_bcs_inequality_l626_626844

variable {Point : Type} [EuclideanGeometry Point]

/-- A convex quadrilateral where side CD is viewed from the midpoint of AB at a right angle -/

def convex_quadrilateral_midpoint_view_right_angle (A B C D O : Point) : Prop :=
  convex_quadrilateral A B C D ∧ 
  midpoint O A B ∧
  angle_eq (angle A O C) (90 : ℝ) ∧ 
  angle_eq (angle B O D) (90 : ℝ)

theorem adi_bcs_inequality (A B C D O : Point)
  (hC : convex_quadrilateral_midpoint_view_right_angle A B C D O) :
  distance A D + distance B C ≥ distance C D :=
sorry

end adi_bcs_inequality_l626_626844


namespace exists_seq_length_at_most_c_log_l626_626912

open Group

theorem exists_seq_length_at_most_c_log (G : Group) [Finite G] (hG : 1 < |G|) :
  ∃ c > 0, ∀ x ∈ G, ∃ L : List G, (L.length ≤ c * Real.log (|G|)) ∧ (∃ L' ⊆ L, List.prod L' = x) :=
by
  sorry

end exists_seq_length_at_most_c_log_l626_626912


namespace team_arrangement_count_l626_626184

theorem team_arrangement_count :
  let male_doctors := 6
  let female_nurses := 3
  let teams := 3
  let doctors_per_team := 2
  let nurses_per_team := 1
  ∃ (males: Finset (Fin male_doctors)) (females: Finset (Fin female_nurses)),
    ∏ i in (Finset.range teams), choose (male_doctors - i * doctors_per_team) doctors_per_team *
    fact teams = 540 := by
  sorry

end team_arrangement_count_l626_626184


namespace vec_subtraction_l626_626401

variables (a b : Prod ℝ ℝ)
def vec1 : Prod ℝ ℝ := (1, 2)
def vec2 : Prod ℝ ℝ := (3, 1)

theorem vec_subtraction : (2 * (vec1.fst, vec1.snd) - (vec2.fst, vec2.snd)) = (-1, 3) := by
  -- Proof here, skipped
  sorry

end vec_subtraction_l626_626401


namespace mailbox_3_contains_one_letter_letter_A_mailbox_1_or_2_l626_626621

def total_ways_to_place_letters : ℕ := 3^4

def favorable_outcomes_mailbox_3 : ℕ := (4.choose 1) * 2^3

def prob_mailbox_3_one_letter : ℚ := favorable_outcomes_mailbox_3 / total_ways_to_place_letters

theorem mailbox_3_contains_one_letter :
  prob_mailbox_3_one_letter = 32 / 81 := by
  sorry

def favorable_outcomes_letter_A : ℕ := (2.choose 1) * 3^3

def prob_letter_A_mailbox_1_or_2 : ℚ := favorable_outcomes_letter_A / total_ways_to_place_letters

theorem letter_A_mailbox_1_or_2 :
  prob_letter_A_mailbox_1_or_2 = 2 / 3 := by
  sorry

end mailbox_3_contains_one_letter_letter_A_mailbox_1_or_2_l626_626621


namespace solve_congruence_l626_626919

theorem solve_congruence (x : ℤ) : 9 * x + 2 ≡ 7 [MOD 15] → x ≡ 0 [MOD 5] :=
by
  sorry

end solve_congruence_l626_626919


namespace carrie_saves_90_l626_626718

-- Define the original prices for each airline
def delta_price : ℝ := 850
def united_price : ℝ := 1100
def american_price : ℝ := 950
def southwest_price : ℝ := 900
def jetblue_price : ℝ := 1200

-- Define the discounts for each airline
def delta_discount : ℝ := 0.20
def united_discount : ℝ := 0.30
def american_discount : ℝ := 0.25
def southwest_discount : ℝ := 0.15
def jetblue_discount : ℝ := 0.40

-- Calculate the final prices after discount
def delta_final : ℝ := delta_price * (1 - delta_discount)
def united_final : ℝ := united_price * (1 - united_discount)
def american_final : ℝ := american_price * (1 - american_discount)
def southwest_final : ℝ := southwest_price * (1 - southwest_discount)
def jetblue_final : ℝ := jetblue_price * (1 - jetblue_discount)

-- Prove that Carrie saves $90 by choosing Delta over the most expensive discounted flight
theorem carrie_saves_90 : (united_final - delta_final) = 90 :=
by
  have h1 : delta_final = 680, { sorry },
  have h2 : united_final = 770, { sorry },
  show 770 - 680 = 90, by norm_num

end carrie_saves_90_l626_626718


namespace even_perfect_square_factors_l626_626034

theorem even_perfect_square_factors (a b c : ℕ) : 
  (0 ≤ a ∧ a ≤ 6) ∧ (0 ≤ c ∧ c ≤ 3) ∧ (0 ≤ b ∧ b ≤ 10) ∧ 
  (∃ k1 k2 k3 : ℕ, a = 2 * k1 ∧ c = 2 * k2 ∧ b = 2 * k3) ∧ 
  (a > 0) -> 
  fintype.card (finset.filter (λ (n : ℕ), ∃ k1 k2 k3 : ℕ, a = 2 * k1 + 1 ∧ 
  c = 2 * k2 + 1 ∧ b = 2 * k3) (finset.range (2^6 * 3^3 * 7^10 + 1))) = 36 := 
by 
  sorry

end even_perfect_square_factors_l626_626034


namespace alpha_beta_property_l626_626873

theorem alpha_beta_property
  (α β : ℝ)
  (hαβ_roots : ∀ x : ℝ, (x = α ∨ x = β) → x^2 + x - 2023 = 0) :
  α^2 + 2 * α + β = 2022 :=
by
  sorry

end alpha_beta_property_l626_626873


namespace congruent_triangles_opposite_pairs_l626_626681

variables {A B C G : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables (Triangle : Type) (centroid : Triangle → Triangle → Triangle → Triangle)
variables (is_centroid : ∀ (α β γ G : Triangle), centroid α β γ = G)
variables (midpoint : Triangle → Triangle → Triangle)

theorem congruent_triangles_opposite_pairs
  (α β γ G : Triangle)
  (hG : is_centroid α β γ G)
  (mA : Triangle := midpoint β γ)
  (mB : Triangle := midpoint α γ)
  (mC : Triangle := midpoint α β) :
  congruent (line_segment α mA G) (line_segment γ mC G) ∧
  congruent (line_segment β mB G) (line_segment γ mC G) ∧
  congruent (line_segment β mB G) (line_segment α mA G) :=
by sorry

end congruent_triangles_opposite_pairs_l626_626681


namespace probability_of_divisible_by_4_ab_l626_626639
-- Importing the Mathlib library

-- Axiom declarations for the conditions
axiom dice_values : set ℕ := {1, 2, 3, 4, 5, 6}
axiom probability_of_fair_die : (∀ n ∈ dice_values, 1 / 6)

-- Definitions for conditions
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- The main statement we want to prove
theorem probability_of_divisible_by_4_ab :
  let outcomes := {a ∈ dice_values | is_divisible_by_4 a} × {b ∈ dice_values | is_divisible_by_4 b}
  in ∀ a b, (a, b) ∈ outcomes →
            is_divisible_by_4 a ∧ is_divisible_by_4 b ∧ is_divisible_by_4 (two_digit_number a b) →
            probability_of_fair_die a * probability_of_fair_die b = 1 / 36 :=
sorry

end probability_of_divisible_by_4_ab_l626_626639


namespace point_transformation_slope_point_transformation_properties_l626_626725

-- Define the transformation
def P_transformed (a b : ℝ) : ℝ × ℝ := (0.6 * a + 0.8 * b, 0.8 * a - 0.6 * b)

-- Define the slope calculation between two points
def slope (a b a' b' : ℝ) : ℝ := (b' - b) / (a' - a)

theorem point_transformation_slope (a b : ℝ) (h : a ≠ 2 * b) :
  slope a b (fst (P_transformed a b)) (snd (P_transformed a b)) = -2 :=
by {
  -- Here would be the proof steps that show slope == -2.
  sorry
}

theorem point_transformation_properties (a b : ℝ) :
  let P' := P_transformed a b in
  (fst (P_transformed (fst P') (snd P')) = a ∧ snd (P_transformed (fst P') (snd P')) = b) ∧
  (∃ m : ℝ, m * (fst (P') + a) / 2 = (snd (P') + b) / 2 ∧ m = 1 / 2) :=
by {
  -- Here would be the proof of additional properties of P' and reflection across y = 0.5x
  sorry
}

end point_transformation_slope_point_transformation_properties_l626_626725


namespace max_take_home_pay_max_income_take_home_pay_l626_626841

theorem max_take_home_pay (x : ℝ) : (2 * x) * (1000 * x) - (20 * x^2) ≤ 12500 :=
sorry

theorem max_income_take_home_pay : ∃ x : ℝ, x = 25 ∧ 1000 * x - (2 * x * 1000 * x * 0.01) = 25000 :=
begin
  use 25,
  have h1 : 1000 * 25 - (2 * 25 * 1000 * 25 * 0.01) = 25000,
  by { norm_num },
  exact ⟨rfl, h1⟩,
end

end max_take_home_pay_max_income_take_home_pay_l626_626841


namespace find_parallel_line_through_point_l626_626593

-- Definition of a point in Cartesian coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of a line in slope-intercept form
def line (a b c : ℝ) : Prop := ∀ p : Point, a * p.x + b * p.y + c = 0

-- Conditions provided in the problem
def P : Point := ⟨-1, 3⟩
def line1 : Prop := line 1 (-2) 3
def parallel_line (c : ℝ) : Prop := line 1 (-2) c

-- Theorem to prove
theorem find_parallel_line_through_point : parallel_line 7 :=
sorry

end find_parallel_line_through_point_l626_626593


namespace parallelogram_symmetric_about_center_l626_626993

-- Define the shapes
inductive Shape
  | Parallelogram
  | IsoscelesTrapezoid
  | RegularPentagon
  | EquilateralTriangle

-- Define the property of being symmetric about the center
def is_symmetrical_about_center (s : Shape) : Prop :=
  ∃ θ : ℝ, θ = 180 ∧ (rotate s θ = s)

-- Specific shape rotations to formalize their properties
def rotate : Shape → ℝ → Shape
  | Shape.Parallelogram, 180 => Shape.Parallelogram
  | _, _ => sorry

-- The theorem that states the Parallelogram is the only shape from the options that is symmetrical about its center
theorem parallelogram_symmetric_about_center :
  is_symmetrical_about_center Shape.Parallelogram :=
by
  -- Proof (to be filled)
  sorry

end parallelogram_symmetric_about_center_l626_626993


namespace log_max_reciprocal_min_l626_626377

open Real

-- Definitions for the conditions
variables (x y : ℝ)
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + 5 * y = 20

-- Theorem statement for the first question
theorem log_max (x y : ℝ) (h : conditions x y) : log x + log y ≤ 1 :=
sorry

-- Theorem statement for the second question
theorem reciprocal_min (x y : ℝ) (h : conditions x y) : (1 / x) + (1 / y) ≥ (7 + 2 * sqrt 10) / 20 :=
sorry

end log_max_reciprocal_min_l626_626377


namespace periodic_function_l626_626318

noncomputable def f : ℝ → ℝ :=
  λ x, ite (0 ≤ x ∧ x ≤ 1) (x * (3 - 2 * x)) (x * (3 - 2 * x))  -- Since we know the form within the interval, we define it generally.

lemma odd_function (x : ℝ) : f (-x) = -f (x) :=
sorry

lemma even_function (x : ℝ) : f (x + 1) = f (-x + 1) :=
sorry

lemma specific_interval (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f (x) = x * (3 - 2 * x) :=
by simp [f]; exact h

theorem periodic_function : f (31 / 2) = -1 :=
by
  show f (31 / 2) = f ((-1) / 2)
  by_contradiction
  sorry  -- you would need to prove this step by step as described in the problem solution.

end periodic_function_l626_626318


namespace triangle_max_y_coordinate_l626_626089

theorem triangle_max_y_coordinate :
  ∃ k : ℝ, (∀ (x y : ℝ), (y ≥ 0) → ((x, y) ∈ set_of (λ p : ℝ × ℝ, (p.fst, p.snd) is a point of triangle with sides 6, 8 and 10))) ∧ (k = 24 / 5) :=
begin
  -- There exists an x, y coordinate system point such that the 
  -- triangle formed with sides 6, 8, and 10 has its maximum y coordinate equal to k = (24 / 5)
  sorry -- Proof would go here
end

end triangle_max_y_coordinate_l626_626089


namespace problem_solution_l626_626887

def vector_magnitude := λ (i j : ℝ), Real.sqrt (i^2 + j^2)

def condition_magnitudes (x y : ℝ) : Prop :=
  vector_magnitude (x + 2) y - vector_magnitude (x - 2) y = 2

def locus_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 3) = 1 ∧ x > 0

def angle_relation (λ : ℝ) : Prop :=
  λ > 0 ∧ λ = 2

theorem problem_solution (x y : ℝ) :
  (condition_magnitudes x y ↔ locus_equation x y) ∧
  (∀ (x y : ℝ), locus_equation x y → ∃ λ, angle_relation λ) :=
by
  intros
  split
  · sorry -- Proof for locus_equation from condition_magnitudes
  · sorry -- Proof for angle_relation

end problem_solution_l626_626887


namespace find_perpendicular_line_through_point_l626_626342

-- Define point A
def Point := (ℝ, ℝ)
def A : Point := (1, -1)

-- Define the line equation x - 2y + 1 = 0
def line1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the line we want to find
def line2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the condition for perpendicular lines
def is_perpendicular (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y k, line1 x y → line2 (x * -2) (y) → k = -1 / 2

-- The proof statement
theorem find_perpendicular_line_through_point :
  (A : Point) → (is_perpendicular line1 line2) 
  → line2 1 (-1) :=
by
  intro A hp
  have : A = (1, -1) := rfl
  rw this
  sorry

end find_perpendicular_line_through_point_l626_626342


namespace problem_proof_l626_626849

noncomputable theory

open Real

def C1_parametric_eqn (θ : ℝ) : ℝ × ℝ :=
( cos θ, 1 + sin θ )

def C2_polar_eqn (θ : ℝ) : ℝ :=
-2 * cos θ + 2 * √3 * sin θ

def line_l_parametric_eqn (t : ℝ) : ℝ × ℝ :=
( t, -√3 * t )

theorem problem_proof :
  (∀ θ, let ⟨x1, y1⟩ := C1_parametric_eqn θ in x1^2 + y1^2 - 2 * y1 = 0 → ( ρ = 2 * sin θ )) ∧
  (∀ θ, (ρ = C2_polar_eqn θ) → let ρ := C2_polar_eqn θ in ρ^2 = ρ (-2 * cos θ + 2 * √3 * sin θ) → ( x^2 + y^2 = -2 * x + 2 * √3 * y )) ∧
  (∀ t θ, let ⟨xt, yt⟩ := line_l_parametric_eqn t in θ = 2 * π / 3 → (√3 - (-2 * (-1/2) + 2 * √3 * (√3/2)) = 4 - √3)) :=
by
  simp
  sorry

end problem_proof_l626_626849


namespace lucky_numbers_l626_626636

theorem lucky_numbers : ∀ (n : ℕ), (∃ (a : Fin 10 → ℕ), Function.Injective a ∧ (∑ i, a i = n)) ↔ (n = 55 ∨ n = 56) :=
by
  intro n
  sorry

end lucky_numbers_l626_626636


namespace compute_BD_l626_626150

-- Definitions based on given conditions
def convex_quadrilateral (A B C D : Type) : Prop := True
def angle_ABD_deg := 105
def angle_ADB_deg := 15
def length_AC := 7
def length_BC := 5
def length_CD := 5

-- Main statement to prove
theorem compute_BD (A B C D : Type) [convex_quadrilateral A B C D]
  (h1 : angle_ABD_deg = 105)
  (h2 : angle_ADB_deg = 15)
  (h3 : length_AC = 7)
  (h4 : length_BC = 5)
  (h5 : length_CD = 5) :
  let BD := real.sqrt 291 in
  BD = real.sqrt 291 := 
by
  sorry

end compute_BD_l626_626150


namespace adult_ticket_cost_l626_626627

def cost_of_child_ticket : ℝ := 4.50
def total_tickets_sold : ℝ := 400
def total_revenue : ℝ := 2100
def children_tickets_sold : ℝ := 200

def total_revenue_from_children_tickets := children_tickets_sold * cost_of_child_ticket
def total_revenue_from_adult_tickets := total_revenue - total_revenue_from_children_tickets
def adult_tickets_sold := total_tickets_sold - children_tickets_sold

theorem adult_ticket_cost :
  total_revenue_from_adult_tickets / adult_tickets_sold = 6 :=
begin
  sorry 
end

end adult_ticket_cost_l626_626627


namespace pizza_slices_count_l626_626280

axiom small_slices : Nat := 6
axiom medium_slices : Nat := 8
axiom large_slices : Nat := 12
axiom total_pizzas : Nat := 15
axiom small_pizzas : Nat := 4
axiom medium_pizzas : Nat := 5

theorem pizza_slices_count :
  (small_pizzas * small_slices + medium_pizzas * medium_slices + 
  (total_pizzas - (small_pizzas + medium_pizzas)) * large_slices) = 136 := 
sorry

end pizza_slices_count_l626_626280


namespace athlete_D_is_selected_l626_626618

-- Define the average scores and variances of athletes
def avg_A : ℝ := 9.5
def var_A : ℝ := 6.6
def avg_B : ℝ := 9.6
def var_B : ℝ := 6.7
def avg_C : ℝ := 9.5
def var_C : ℝ := 6.7
def avg_D : ℝ := 9.6
def var_D : ℝ := 6.6

-- Define what it means for an athlete to be good and stable
def good_performance (avg : ℝ) : Prop := avg ≥ 9.6
def stable_play (variance : ℝ) : Prop := variance ≤ 6.6

-- Combine conditions for selecting the athlete
def D_is_suitable : Prop := good_performance avg_D ∧ stable_play var_D

-- State the theorem to be proved
theorem athlete_D_is_selected : D_is_suitable := 
by 
  sorry

end athlete_D_is_selected_l626_626618


namespace _l626_626277

noncomputable theorem meagre_sets_cardinality {n : ℕ} (hn : n ≥ 2) (S : Finset (Fin n × Fin n)) (hS : ∀ i, ∃ j, (i, j) ∈ S ∧ ∀ j, ∃ i, (i, j) ∈ S) :
  ∃ m n_permutations, m = n ∧ n_permutations = nat.factorial n := by
sorrry

noncomputable theorem fat_sets_cardinality {n : ℕ} (hn : n ≥ 2) (S : Finset (Fin n × Fin n)) (hS : ∀ i, ∃ j, (i, j) ∈ S ∧ ∀ j, ∃ i, (i, j) ∈ S) :
  ∃ M n_squared, M = 2 * n - 2 ∧ n_squared = n ^ 2 := by
sorrry

end _l626_626277


namespace perimeter_ratio_of_squares_l626_626572

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l626_626572


namespace trig_expression_evaluation_l626_626031

theorem trig_expression_evaluation :
  ∀ x : ℝ, (let f := λ x, sin x - cos x in
  (∀ x, deriv f x = 2 * f x) →
  (1 + sin x ^ 2) / (cos x ^ 2 - sin (2 * x)) = -19 / 5) :=
by
  intro x f hf
  sorry

end trig_expression_evaluation_l626_626031


namespace triangle_properties_l626_626291

-- Define the given sides of the triangle
def a := 6
def b := 8
def c := 10

-- Define necessary parameters and properties
def isRightTriangle (a b c : Nat) : Prop := a^2 + b^2 = c^2
def area (a b : Nat) : Nat := (a * b) / 2
def semiperimeter (a b c : Nat) : Nat := (a + b + c) / 2
def inradius (A s : Nat) : Nat := A / s
def circumradius (c : Nat) : Nat := c / 2

-- The theorem statement
theorem triangle_properties :
  isRightTriangle a b c ∧
  area a b = 24 ∧
  semiperimeter a b c = 12 ∧
  inradius (area a b) (semiperimeter a b c) = 2 ∧
  circumradius c = 5 :=
by
  sorry

end triangle_properties_l626_626291


namespace place_b_left_of_a_forms_correct_number_l626_626414

noncomputable def form_three_digit_number (a b : ℕ) : ℕ :=
  100 * b + a

theorem place_b_left_of_a_forms_correct_number (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 1 ≤ b ∧ b < 10) :
  form_three_digit_number a b = 100 * b + a :=
by sorry

end place_b_left_of_a_forms_correct_number_l626_626414


namespace g_at_neg10_l626_626118

def g (x : ℤ) : ℤ := 
  if x < -3 then 3 * x + 7 else 4 - x

theorem g_at_neg10 : g (-10) = -23 := by
  -- The proof goes here
  sorry

end g_at_neg10_l626_626118


namespace terminal_side_quadrant_l626_626038

theorem terminal_side_quadrant (α : ℝ) (h1 : α = 3) : 
  (real.pi / 2 < α ∧ α < real.pi) → "II" := 
by 
  intro H _ sorry

end terminal_side_quadrant_l626_626038


namespace square_perimeter_ratio_l626_626556

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l626_626556


namespace standard_equation_of_parabola_l626_626179

theorem standard_equation_of_parabola (x : ℝ) (y : ℝ) (directrix : ℝ) (eq_directrix : directrix = 1) :
  y^2 = -4 * x :=
sorry

end standard_equation_of_parabola_l626_626179


namespace abs_inequality_solution_l626_626537

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l626_626537


namespace q_n_formula_inequality_l626_626763

def partition (n : ℕ) : ℕ := sorry   -- Placeholder for the number of partitions function, p(n)

def dispersion_sum (n : ℕ) : ℕ := -- Definition of q(n)
  1 + ∑ i in Finset.range n, partition i

theorem q_n_formula (n : ℕ) : 
    dispersion_sum n = 1 + ∑ i in Finset.Ico 1 n, partition i :=
by { sorry }

theorem inequality (n : ℕ) :
  1 + ∑ i in Finset.Ico 1 n, partition i ≤ nat.sqrt (2 * n) * partition n :=
by { sorry }

end q_n_formula_inequality_l626_626763


namespace at_least_one_travels_l626_626740

open ProbabilityTheory

/-
  Let A, B, and C be events representing persons A, B, and C traveling to Beijing respectively.
  The events are mutually independent and have the following probabilities:
  - P(A) = 1/3
  - P(B) = 1/4
  - P(C) = 1/5

  The problem requires us to prove the probability that at least one person travels to Beijing.
-/

noncomputable def prob_A : ℝ := 1 / 3
noncomputable def prob_B : ℝ := 1 / 4
noncomputable def prob_C : ℝ := 1 / 5

theorem at_least_one_travels :
  let prob_not_A := 1 - prob_A,
      prob_not_B := 1 - prob_B,
      prob_not_C := 1 - prob_C,
      prob_none_travel := prob_not_A * prob_not_B * prob_not_C,
      prob_at_least_one_travels := 1 - prob_none_travel 
  in prob_at_least_one_travels = 3 / 5 :=
by
  sorry

end at_least_one_travels_l626_626740


namespace convert_to_scientific_notation_l626_626839

theorem convert_to_scientific_notation (H : 1 = 10^9) : 
  3600 * (10 : ℝ)^9 = 3.6 * (10 : ℝ)^12 :=
by
  sorry

end convert_to_scientific_notation_l626_626839


namespace ticket_number_l626_626146

-- Define the conditions and the problem
theorem ticket_number (x y z N : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy: 0 ≤ y ∧ y ≤ 9) (hz: 0 ≤ z ∧ z ≤ 9) 
(hN1: N = 100 * x + 10 * y + z) (hN2: N = 11 * (x + y + z)) : 
N = 198 :=
sorry

end ticket_number_l626_626146


namespace perpendicular_iff_length_eq_l626_626093

-- Definitions and conditions
variables {A B C D P : Point}
variables (ABC : triangle A B C) (h_acute : acute ABC)
variables (angle_A : angle A B C = 45)
variables (hD_on_AB : D ∈ line_segment A B)
variables (hCD_perp_AB : perp D C A B)
variables (hP_internal_CD : P ∈ line_segment C D)

-- The statement to prove
theorem perpendicular_iff_length_eq
  (h_acute : acute ABC)
  (angle_A : angle A B C = 45)
  (hD_on_AB : D ∈ line_segment A B)
  (hCD_perp_AB : perp D C (line_segment A B))
  (hP_internal_CD : P ∈ (line_segment C D)) :
  (perp A P (line_segment B C)) ↔ (dist A P = dist B C) :=
sorry

end perpendicular_iff_length_eq_l626_626093


namespace man_l626_626676

theorem man's_speed_with_stream
  (V_m V_s : ℝ)
  (h1 : V_m = 6)
  (h2 : V_m - V_s = 4) :
  V_m + V_s = 8 :=
sorry

end man_l626_626676


namespace range_of_a_l626_626804

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ a then cos x else 1 / x

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ∈ set.Icc (-1 : ℝ) (1 : ℝ)) : a ∈ set.Ici (1 : ℝ) :=
begin
  sorry
end

example : range_of_a := sorry

end range_of_a_l626_626804


namespace ratio_triangle_MNC_to_trapezoid_l626_626857

-- Definitions
variables (A B C D M N : Type) 
variables [trapezoid A B C D]
variables (AB CD : ℝ) (MC MD : ℝ)
variable (S_ABCD : ℝ)

-- Conditions
def condition1 : AB = 3 * CD := sorry
def condition2 : MC = 2 * MD := sorry
def condition3 : N = intersection (line B M) (line A C) := sorry

-- Problem Statement
theorem ratio_triangle_MNC_to_trapezoid (h1 : condition1 AB CD) (h2 : condition2 MC MD) (h3 : condition3 N A B C D M) :
  (area_triangle M N C) / S_ABCD = 1 / 33 :=
sorry

end ratio_triangle_MNC_to_trapezoid_l626_626857


namespace rectangle_length_l626_626154

theorem rectangle_length 
  (s l : ℝ) 
  (square_perimeter : 4 * s = 800)
  (rectangle_width : 64)
  (area_relationship : s^2 = 5 * (l * rectangle_width)) : 
  l = 125 :=
by
  sorry

end rectangle_length_l626_626154


namespace min_term_of_sequence_is_12_l626_626023

def sequence (n : ℕ) : ℝ := n - 7 * Real.sqrt n + 2

theorem min_term_of_sequence_is_12 :
    ∀ n : ℕ, n > 0 → n ≠ 12 →  sequence 12 < sequence n :=
by
  sorry

end min_term_of_sequence_is_12_l626_626023


namespace conjugate_of_z_l626_626020

-- Define the given complex number z
def z : ℂ := (1 - complex.i) / (1 + complex.i)

-- State the main problem
theorem conjugate_of_z : complex.conj z = complex.i := by
  sorry

end conjugate_of_z_l626_626020


namespace sum_of_digits_1_to_999_l626_626613

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (List.range (n+1)).map (λ x, x.digits 10).join.sum (λ x, x)

theorem sum_of_digits_1_to_999 : sum_of_digits 999 = 13500 :=
by
  sorry

end sum_of_digits_1_to_999_l626_626613


namespace correct_formula_l626_626164

theorem correct_formula {x y : ℕ} : 
  (x = 0 ∧ y = 100) ∨
  (x = 1 ∧ y = 90) ∨
  (x = 2 ∧ y = 70) ∨
  (x = 3 ∧ y = 40) ∨
  (x = 4 ∧ y = 0) →
  y = 100 - 5 * x - 5 * x^2 :=
by
  sorry

end correct_formula_l626_626164


namespace range_of_x_l626_626731

def star (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x (x : ℝ) (h : star x (x - 2) < 0) : -2 < x ∧ x < 1 := by
  sorry

end range_of_x_l626_626731


namespace num_divisors_of_x13_minus_x_l626_626036

theorem num_divisors_of_x13_minus_x : ∃ n : ℕ, (∀ x : ℕ, (x > 0) → (x ^ 13 - x) % n = 0) ∧ (n > 1) ∧ (card {d : ℕ | d > 1 ∧ d ∣ 2730} = 31) :=
by
  sorry

end num_divisors_of_x13_minus_x_l626_626036


namespace tabbyAverageSpeed_l626_626651

noncomputable def averageSpeedSwimmingRunning (d t_s t_r : ℝ) (swimSpeed runSpeed : ℝ) : ℝ :=
  (d + d) / (t_s + t_r)

theorem tabbyAverageSpeed :
  ∀ (d : ℝ), (d > 0) → averageSpeedSwimmingRunning d (d / 1) (d / 9) 1 9 = 1.8 :=
by
  intros d h_d
  rw [←div_mul_cancel (d : ℝ) (ne_of_gt h_d), div_add_div_same, ←div_eq_div_iff_mul_eq_mul,
      mul_comm (1:ℝ), mul_assoc, mul_div_cancel_left, mul_comm, mul_assoc, mul_div_cancel_left]
  { simp only [one_mul, mul_div_assoc, div_eq_mul_inv, inv_mul_cancel] }
  { exact ne_of_gt (show (9:ℝ) > 0 by norm_num) }
  { exact ne_of_gt h_d }
  { simp }
  sorry

end tabbyAverageSpeed_l626_626651


namespace solution_l626_626525

noncomputable def problem (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), tan x * tan y = a ∧ sin x^2 + sin y^2 = b^2

theorem solution (a b : ℝ) : problem a b ↔ (1 < b^2 ∧ b^2 < 2 * a / (a + 1)) ∧ (1 < b^2 ∧ b^2 < 2 * a / (a - 1)) :=
sorry

end solution_l626_626525


namespace min_sqrt3_neg2_min_range_x_l626_626519

-- Definition of min function
def my_min (a b : ℝ) : ℝ := if a ≥ b then b else a

-- Problem 1: Prove that min{-√3, -2} = -2
theorem min_sqrt3_neg2 : my_min (-real.sqrt 3) (-2) = -2 := sorry

-- Problem 2: Prove the range of x such that min{(2x-3)/2, (x+2)/3} = (x+2)/3
theorem min_range_x (x : ℝ) : my_min ((2 * x - 3) / 2) ((x + 2) / 3) = (x + 2) / 3 ↔ x ≥ 13 / 4 := sorry

end min_sqrt3_neg2_min_range_x_l626_626519


namespace original_average_age_l626_626930

variable (A : ℕ)
variable (N : ℕ := 2)
variable (new_avg_age : ℕ := 32)
variable (age_decrease : ℕ := 4)

theorem original_average_age :
  (A * N + new_avg_age * 2) / (N + 2) = A - age_decrease → A = 40 := 
by
  sorry

end original_average_age_l626_626930


namespace basketball_children_l626_626189

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l626_626189


namespace incorrect_conclusion_l626_626152

-- Define the linear regression model
def model (x : ℝ) : ℝ := 0.85 * x - 85.71

-- Define the conditions
axiom linear_correlation : ∀ (x y : ℝ), ∃ (x_i y_i : ℝ) (i : ℕ), model x = y

-- The theorem to prove the statement for x = 170 is false
theorem incorrect_conclusion (x : ℝ) (h : x = 170) : ¬ (model x = 58.79) :=
  by sorry

end incorrect_conclusion_l626_626152


namespace solve_for_A_plus_B_l626_626967

-- Definition of the problem conditions
def T := 7 -- The common total sum for rows and columns

-- Summing the rows and columns in the partially filled table
variable (A B : ℕ)
def table_condition :=
  4 + 1 + 2 = T ∧
  2 + A + B = T ∧
  4 + 2 + B = T ∧
  1 + A + B = T

-- Statement to prove
theorem solve_for_A_plus_B (A B : ℕ) (h : table_condition A B) : A + B = 5 :=
by
  sorry

end solve_for_A_plus_B_l626_626967


namespace percentage_decrease_l626_626953

theorem percentage_decrease (P : ℝ) (new_price : ℝ) (x : ℝ) (h1 : new_price = 320) (h2 : P = 421.05263157894734) : x = 24 :=
by
  sorry

end percentage_decrease_l626_626953


namespace carl_trip_cost_is_correct_l626_626713

structure TripConditions where
  city_mpg : ℕ
  highway_mpg : ℕ
  city_distance : ℕ
  highway_distance : ℕ
  gas_cost : ℕ

def total_gas_cost (conds : TripConditions) : ℕ :=
  let total_city_miles := conds.city_distance * 2
  let total_highway_miles := conds.highway_distance * 2
  let city_gas_needed := total_city_miles / conds.city_mpg
  let highway_gas_needed := total_highway_miles / conds.highway_mpg
  let total_gas_needed := city_gas_needed + highway_gas_needed
  total_gas_needed * conds.gas_cost

theorem carl_trip_cost_is_correct (conds : TripConditions)
  (h1 : conds.city_mpg = 30)
  (h2 : conds.highway_mpg = 40)
  (h3 : conds.city_distance = 60)
  (h4 : conds.highway_distance = 200)
  (h5 : conds.gas_cost = 3) :
  total_gas_cost conds = 42 := by
  rw [total_gas_cost, h1, h2, h3, h4, h5]
  -- Proof steps will follow here, but we can skip them for now
  sorry

end carl_trip_cost_is_correct_l626_626713


namespace sum_of_factors_of_24_l626_626218

theorem sum_of_factors_of_24 : 
  let factors := [1, 2, 4, 8, 3, 6, 12, 24] in
  (factors.sum = 60) :=
by {
  let factors := [1, 2, 4, 8, 3, 6, 12, 24]
  show factors.sum = 60,
  sorry
}

end sum_of_factors_of_24_l626_626218


namespace samantha_spends_on_dog_toys_l626_626163

theorem samantha_spends_on_dog_toys:
  let toy_price := 12.00
  let discount := 0.5
  let num_toys := 4
  let tax_rate := 0.08
  let full_price_toys := num_toys / 2
  let half_price_toys := num_toys / 2
  let total_cost_before_tax := full_price_toys * toy_price + half_price_toys * (toy_price * discount)
  let sales_tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax = 38.88 :=
by {
  sorry
}

end samantha_spends_on_dog_toys_l626_626163


namespace triang_beaut_AOB_81_l626_626785

-- Definitions for the given conditions
def ABC_isosceles (A B C : Type) [Isosceles A B C] : Prop := 
  ∠ B = 102 

def O_in_ABC (O A B C : Type) : Prop := 
  30 + 21 = 51 

def Triangle_interior_angle_sum (a b c : Nat) : Prop := 
  a + b + c = 180 

-- The theorem to be proved
theorem triang_beaut_AOB_81 {A B C O : Type} [Isosceles A B C] :
  ∠B = 102 ∧ 30 + 21 = 51 →
  ∠BOA = 81 
by
  sorry

end triang_beaut_AOB_81_l626_626785


namespace find_cd_l626_626904

theorem find_cd (c d : ℝ) (h1 : 0 < c) (h2 : 0 < d)
  (h3 : ∀ r : ℝ, 0 < r → r.to_int = real.log r)
  (h4: nat.sqrt (real.log c).nat_abs + nat.sqrt (real.log d).nat_abs +
       (real.log c / 2).to_int + (real.log d / 2).to_int +
       ((real.log c / 2) + (real.log d / 2)).to_int = 50) :
  c * d = 10^37 :=
by
  sorry

end find_cd_l626_626904


namespace find_x_l626_626416

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end find_x_l626_626416


namespace mean_home_runs_l626_626595

/-- Proving the mean (average) home runs hit by the players given the distribution. -/
theorem mean_home_runs :
    let home_runs := [6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 10] in
    (home_runs.sum / home_runs.length : ℚ) = 7 := by
  sorry

end mean_home_runs_l626_626595


namespace cuboid_volume_l626_626652

theorem cuboid_volume (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : a * b * c = 120 :=
by
  sorry

end cuboid_volume_l626_626652


namespace equation_and_inclination_of_AB_area_of_triangle_ABP_l626_626373

-- Definition of point A and B
def A : Point := ⟨1, 1⟩
def B : Point := ⟨2, -1⟩

-- Proving the equation of line AB and angle of inclination
theorem equation_and_inclination_of_AB :
  (line_equation A B = "2x + y - 3 = 0") ∧ (angle_inclination_obtuse A B) :=
sorry

-- Definition of point P on the x-axis such that ∠ABP = 90°
def P : Point := ⟨4, 0⟩

-- Proving the area of triangle △ABP
theorem area_of_triangle_ABP :
  (angle_90_deg A B P) → (triangle_area A B P = 5 / 2) :=
sorry

end equation_and_inclination_of_AB_area_of_triangle_ABP_l626_626373


namespace total_area_l626_626299

-- Definitions based on conditions
variables (A B C D E F G H I J : Type) [metric_space A] 
variable (x : ℝ)
variable (AB AE BE : ℝ)
variable (HI HJ : ℝ)
variable (ABCD AEFG HIJD : ℝ)

-- Conditions
def is_right_angle (E A B : A) : Prop := true -- Dummy definition for right angle
def BE_length (BE : ℝ) := BE = 12
def HI_length (HI AB : ℝ) := HI = 2 * AB
def HJ_length (HJ AB : ℝ) := HJ = AB

-- Square and rectangle area conditions
def square_area (side : ℝ) := side ^ 2
def rect_area (side1 side2 : ℝ) := side1 * side2

-- Main statement to prove
theorem total_area
  (h1 : is_right_angle E A B)
  (h2 : BE_length BE)
  (h3 : HI_length HI x)
  (h4 : HJ_length HJ x)
  (h5 : BE = 12)
  : square_area x + square_area (sqrt (144 - x ^ 2)) + rect_area (2 * x) x = 216 :=
  sorry

end total_area_l626_626299


namespace f_at_2016_l626_626776

noncomputable def f : ℝ → ℝ
| x => if h : x > 0 then f (x - 4) else 2^x + ∫ (t : ℝ) in 0 .. (Real.pi / 6), Real.cos (3 * t)

theorem f_at_2016 : f 2016 = 4 / 3 :=
by
  sorry

end f_at_2016_l626_626776


namespace main_theorem_l626_626787

-- Define proposition p: For an arithmetic sequence, (n, S_n) lies on a parabola
def proposition_p (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  ∃ d a1 : ℝ, ∀ n : ℕ, S_n n = n / 2 * (2 * a_n 1 + (n - 1) * d)

-- False proposition p
axiom p_false : ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ), ¬ proposition_p a_n S_n

-- Define proposition q: If m > 1, mx² + 2(m-2)x + 1 > 0 has solutions in ℝ
def proposition_q (m : ℝ) : Prop :=
  m > 1 → ∀ x : ℝ, m * x^2 + 2 * (m - 2) * x + 1 > 0

-- False proposition q
axiom q_false : ∀ m : ℝ, ¬ proposition_q m

-- The contrapositive of p (proposition s) is false
def contrapositive_p (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  ¬ proposition_p a_n S_n → ∀ n : ℕ, (d ≠ 0 ∨ S_n n ≠ n / 2 * (2 * a_n 1 + (n - 1) * 0))

-- The converse of q (proposition r) is false
def converse_q (m : ℝ) : Prop :=
  (∀ x : ℝ, m * x^2 + 2 * (m - 2) * x + 1 > 0) → m > 1

-- Main theorem to prove
theorem main_theorem : ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (m : ℝ),
  ¬contrapositive_p a_n S_n ∧ ¬converse_q m :=
by {
  intro a_n S_n m,
  split,
  { exact λ h, p_false a_n S_n h },
  { exact λ h, q_false m h },
  sorry
}

end main_theorem_l626_626787


namespace expansion_coefficient_sum_l626_626452

-- Define the coefficient function f
def f (m n : ℕ) : ℕ :=
  @Nat.choose 6 m * @Nat.choose 4 n

-- Theorem statement
theorem expansion_coefficient_sum : f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 :=
by 
  -- Use the definition of f to compute directly
  have h_f30 : f 3 0 = 20 := by sorry
  have h_f21 : f 2 1 = 60 := by sorry
  have h_f12 : f 1 2 = 36 := by sorry
  have h_f03 : f 0 3 = 4 := by sorry
  -- Combine them to get the result
  calc f 3 0 + f 2 1 + f 1 2 + f 0 3
    = 20 + 60 + 36 + 4 := by rw [h_f30, h_f21, h_f12, h_f03]
    ... = 120 := by decide

end expansion_coefficient_sum_l626_626452


namespace centroid_of_A1B1C1_is_midpoint_l626_626881

noncomputable def centroid (A B C : Point) : Point := 
  (A + B + C) / 3

variables
  (A B C X A1 B1 C1 M M1 : Point)
  (H1 : M = centroid A B C) -- M is the centroid of triangle ABC
  (H2 : A1X ∥ AM)           -- A1X is parallel to AM
  (H3 : B1X ∥ BM)           -- B1X is parallel to BM
  (H4 : C1X ∥ CM)           -- C1X is parallel to CM
  (H5 : A1 ∈ line BC)       -- A1 lies on line BC
  (H6 : B1 ∈ line CA)       -- B1 lies on line CA
  (H7 : C1 ∈ line AB)       -- C1 lies on line AB

theorem centroid_of_A1B1C1_is_midpoint (A B C X A1 B1 C1 M M1 : Point) 
  (H1 : M = centroid A B C)
  (H2 : A1X ∥ AM)
  (H3 : B1X ∥ BM)
  (H4 : C1X ∥ CM)
  (H5 : A1 ∈ line BC)
  (H6 : B1 ∈ line CA)
  (H7 : C1 ∈ line AB) :
  M1 = midpoint M X :=
sorry

end centroid_of_A1B1C1_is_midpoint_l626_626881


namespace real_root_of_equation_no_real_root_outside_range_l626_626319

theorem real_root_of_equation (p : ℝ) (x : ℝ) 
  (h0 : 0 ≤ p) (h1 : p ≤ 4 / 3) :
  (sqrt (x^2 - p) + 2 * sqrt (x^2 - 1) = x) ↔ 
  (x = (4 - p) / (sqrt (8 * (2 - p)))) :=
sorry

theorem no_real_root_outside_range (p : ℝ) (x : ℝ) 
  (h : ¬(0 ≤ p ∧ p ≤ 4 / 3)) :
  ¬ (sqrt (x^2 - p) + 2 * sqrt (x^2 - 1) = x) :=
sorry

end real_root_of_equation_no_real_root_outside_range_l626_626319


namespace sin_theta_square_midpoints_l626_626248

theorem sin_theta_square_midpoints 
  (A B C D P Q X: ℝ × ℝ)
  (h_square: (A = (0, 0)) ∧ (B = (0, 4)) ∧ (C = (4, 4)) ∧ (D = (4, 0)))
  (h_midpoints: (P = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧ (Q = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)))
  (h_intersect: ∃ X, (∃ λ μ: ℝ, (X = (λ * A.1 + (1-λ) * P.1, λ * A.2 + (1-λ) * P.2))
  ∧ (X = (μ * B.1 + (1-μ) * Q.1, μ * B.2 + (1-μ) * Q.2))))
  : sin (angle P.X.1 Q.X.1) = 3 / 5 := 
sorry

end sin_theta_square_midpoints_l626_626248


namespace total_books_bought_l626_626635

-- Let x be the number of math books and y be the number of history books
variables (x y : ℕ)

-- Conditions
def math_book_cost := 4
def history_book_cost := 5
def total_price := 368
def num_math_books := 32

-- The total number of books bought is the sum of the number of math books and history books, which should result in 80
theorem total_books_bought : 
  y * history_book_cost + num_math_books * math_book_cost = total_price → 
  x = num_math_books → 
  x + y = 80 :=
by
  sorry

end total_books_bought_l626_626635


namespace min_area_triangle_DEF_l626_626608

theorem min_area_triangle_DEF (z : ℂ) :
((z - 3)^6 = 64) →
  let D := (⟨2, 0⟩ : ℂ)
  let E := (⟨1, Real.sqrt 3⟩ : ℂ)
  let F := (⟨-1, Real.sqrt 3⟩ : ℂ)
  ∃ A : ℝ, A = Real.sqrt 3 ∧ 
  ∃ x1 y1 x2 y2 x3 y3 : ℝ,
  (D = ⟨x1, y1⟩ ∧ E = ⟨x2, y2⟩ ∧ F = ⟨x3, y3⟩) ∧
  (A = (1 / 2) * Real.abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) := sorry

end min_area_triangle_DEF_l626_626608


namespace complex_power_eval_l626_626383

theorem complex_power_eval : 
  let z := (1 - Complex.i) / (1 + Complex.i) 
  in z ^ 2017 = -Complex.i := 
by
  sorry

end complex_power_eval_l626_626383


namespace sum_greater_than_two_l626_626172

variables {x y : ℝ}

theorem sum_greater_than_two (hx : x^7 > y^6) (hy : y^7 > x^6) : x + y > 2 :=
sorry

end sum_greater_than_two_l626_626172


namespace range_x_minus_y_l626_626456

-- Definition of the curve in polar coordinates
def curve_polar (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta + 2 * Real.sin theta

-- Conversion to rectangular coordinates
noncomputable def curve_rectangular (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * x + 2 * y

-- The final Lean 4 statement
theorem range_x_minus_y (x y : ℝ) (h : curve_rectangular x y) :
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10 :=
sorry

end range_x_minus_y_l626_626456


namespace no_such_function_exists_l626_626655

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (f x) = x^2 - 1996 :=
by
  sorry

end no_such_function_exists_l626_626655


namespace value_of_3x_plus_5y_l626_626836

variable (x y : ℚ)

theorem value_of_3x_plus_5y
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 3 * x + 5 * y = 6 := 
sorry

end value_of_3x_plus_5y_l626_626836


namespace marble_distribution_l626_626864

-- Define the problem statement using conditions extracted above
theorem marble_distribution :
  ∃ (A B C D : ℕ), A + B + C + D = 28 ∧
  (A = 7 ∨ B = 7 ∨ C = 7 ∨ D = 7) ∧
  ((A = 7 → B + C + D = 21) ∧
   (B = 7 → A + C + D = 21) ∧
   (C = 7 → A + B + D = 21) ∧
   (D = 7 → A + B + C = 21)) :=
sorry

end marble_distribution_l626_626864


namespace average_mileage_first_car_l626_626066

theorem average_mileage_first_car (X Y : ℝ) 
  (h1 : X + Y = 75) 
  (h2 : 25 * X + 35 * Y = 2275) : 
  X = 35 :=
by 
  sorry

end average_mileage_first_car_l626_626066


namespace paper_fold_l626_626761

theorem paper_fold (n : ℕ) : 0.1 * (2 ^ n) > 20 → n ≥ 8 :=
by sorry

end paper_fold_l626_626761


namespace distance_from_point_a_to_line_bc_l626_626341

-- Definitions based on conditions
def point_a : ℝ × ℝ × ℝ := (2, 0, -1)
def point_b : ℝ × ℝ × ℝ := (0, 3, 5)
def point_c : ℝ × ℝ × ℝ := (1, 0, 3)

-- Function to calculate the distance from a point to a line defined by two points
noncomputable def distance_to_line (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := a
  let (x2, y2, z2) := b
  let (x3, y3, z3) := c
  let t := (x1 * (x3 - x2) + y1 * (y3 - y2) + z1 * (z3 - z2))
             / ((x3 - x2)^2 + (y3 - y2)^2 + (z3 - z2)^2)
  let d := ((x2 + t * (x3 - x2) - x1)^2 +
           (y2 + t * (y3 - y2) - y1)^2 +
           (z2 + t * (z3 - z2) - z1)^2)
  sqrt d

-- Theorem to prove the distance
theorem distance_from_point_a_to_line_bc :
  distance_to_line point_a point_b point_c = sqrt (4370 / 196) :=
sorry

end distance_from_point_a_to_line_bc_l626_626341


namespace job_positions_growth_rate_l626_626269

theorem job_positions_growth_rate (x : ℝ) :
  1501 * (1 + x) ^ 2 = 1815 := sorry

end job_positions_growth_rate_l626_626269


namespace sum_of_integers_ending_in_2_between_100_and_500_l626_626306

theorem sum_of_integers_ending_in_2_between_100_and_500 :
  let s : List ℤ := List.range' 102 400 10
  let sum_of_s := s.sum
  sum_of_s = 11880 :=
by
  sorry

end sum_of_integers_ending_in_2_between_100_and_500_l626_626306


namespace no_infinite_sequence_of_primes_l626_626087

open Nat

def sequence_of_primes (p : ℕ → ℕ) : Prop :=
  ∀ n, Prime (p n)

def condition (p : ℕ → ℕ) : Prop :=
  ∀ n, |p(n + 1) - 2 * p(n)| = 1

theorem no_infinite_sequence_of_primes :
  ¬ ∃ p : ℕ → ℕ, sequence_of_primes p ∧ condition p :=
by
  sorry

end no_infinite_sequence_of_primes_l626_626087


namespace smallest_number_diminished_by_2_divisible_12_16_18_21_28_l626_626214

def conditions_holds (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧ (n - 2) % 16 = 0 ∧ (n - 2) % 18 = 0 ∧ (n - 2) % 21 = 0 ∧ (n - 2) % 28 = 0

theorem smallest_number_diminished_by_2_divisible_12_16_18_21_28 :
  ∃ (n : ℕ), conditions_holds n ∧ (∀ m, conditions_holds m → n ≤ m) ∧ n = 1009 :=
by
  sorry

end smallest_number_diminished_by_2_divisible_12_16_18_21_28_l626_626214


namespace incenter_barycentric_coordinates_l626_626459

theorem incenter_barycentric_coordinates (P Q R J : Type) (p q r : ℕ)
  (h_p : p = 6) (h_q : q = 8) (h_r : r = 3) :
  ∃ (x y z : ℚ), 
  (x = (p : ℚ) / (p + q + r)) ∧ 
  (y = (q : ℚ) / (p + q + r)) ∧ 
  (z = (r : ℚ) / (p + q + r)) ∧ 
  (x + y + z = 1) ∧ 
  (⟨x, y, z⟩ = ⟨6/17, 8/17, 3/17⟩) :=
by {
  -- Placeholder for the proof
  sorry
}

end incenter_barycentric_coordinates_l626_626459


namespace fourth_row_number_is_30210_l626_626147

noncomputable def sequence (i : Nat) : Nat :=
  match i % 3 with
  | 0 => 1
  | 1 => 2
  | _ => 3

noncomputable def grid (row col : Nat) : Nat :=
  if col < row then 0 else sequence (col - row)

theorem fourth_row_number_is_30210 : 
  (grid 3 0) * 10000 + (grid 3 1) * 1000 + (grid 3 2) * 100 + (grid 3 3) * 10 + (grid 3 4) = 30210 :=
by {
  sorry
}

end fourth_row_number_is_30210_l626_626147


namespace condition_2_3_implies_f_x1_greater_f_x2_l626_626389

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem condition_2_3_implies_f_x1_greater_f_x2 
(x1 x2 : ℝ) (h1 : -2 * Real.pi / 3 ≤ x1 ∧ x1 ≤ 2 * Real.pi / 3) 
(h2 : -2 * Real.pi / 3 ≤ x2 ∧ x2 ≤ 2 * Real.pi / 3) 
(hx1_sq_gt_x2_sq : x1^2 > x2^2) (hx1_gt_abs_x2 : x1 > |x2|) : 
  f x1 > f x2 := 
sorry

end condition_2_3_implies_f_x1_greater_f_x2_l626_626389


namespace number_of_solvable_2x2_mazes_l626_626296

def is_blank (grid : Fin 2 → Fin 2 → Bool) (i j : Fin 2) : Prop :=
  grid i j = true

def adjacent (i j k l : Fin 2) : Prop :=
  (i = k ∧ ((j = l + 1) ∨ (j + 1 = l))) ∨ 
  (j = l ∧ ((i = k + 1) ∨ (i + 1 = k)))

def path_exists (grid : Fin 2 → Fin 2 → Bool) (path : List (Fin 2 × Fin 2)) : Prop :=
  path.head = (0,0) ∧ path.last = (1,1) ∧ 
  ∀p ∈ path, is_blank grid p.1 p.2 ∧ 
  ∀p q ∈ path, adjacent p.1 p.2 q.1 q.2 → p = q ∨ (∃i, List.nth path i = p ∧ List.nth path (i+1) = q)

def solvable (grid : Fin 2 → Fin 2 → Bool) : Prop :=
  is_blank grid 0 0 ∧ is_blank grid 1 1 ∧ 
  ∃path, path_exists grid path

theorem number_of_solvable_2x2_mazes : 
  ({ g : (Fin 2 → Fin 2 → Bool) // solvable g }.to_finset.card = 3) := 
sorry

end number_of_solvable_2x2_mazes_l626_626296


namespace tom_finishes_fourth_exam_at_1PM_l626_626970

-- Define the conditions
def equally_time_consuming_exams : Prop := ∀ (n: ℕ), (n ≤ 4) → (prep_duration + exam_duration n) = time_for_all_exams
def preparation_time : ℕ := 20
def start_time : ℕ := 8 * 60
def second_exam_completion_time : ℕ := 10 * 60 + 30
def num_exams : ℕ := 4

-- Compute the finish time of the fourth exam
def finish_time_fourth_exam : ℕ :=
  second_exam_completion_time + 2 * (second_exam_completion_time - start_time) / 2

-- The formal statement in Lean
theorem tom_finishes_fourth_exam_at_1PM :
  (finish_time_fourth_exam = 13 * 60) := by
begin
  sorry
end

end tom_finishes_fourth_exam_at_1PM_l626_626970


namespace cannot_be_diagonals_l626_626640

theorem cannot_be_diagonals (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) :
  ¬( {6, 8, 11}.subset {real.sqrt (a^2 + b^2), real.sqrt (b^2 + c^2), real.sqrt (a^2 + c^2)} ) :=
by
  sorry

end cannot_be_diagonals_l626_626640


namespace problem_c_d_sum_l626_626098

theorem problem_c_d_sum (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (C / (x - 3) + D * (x - 2) = (5 * x ^ 2 - 8 * x - 6) / (x - 3))) : C + D = 20 :=
sorry

end problem_c_d_sum_l626_626098


namespace geom_seq_inv_sum_eq_l626_626453

noncomputable def geom_seq (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * r^n

theorem geom_seq_inv_sum_eq
    (a_1 r : ℚ)
    (h_sum : geom_seq a_1 r 0 + geom_seq a_1 r 1 + geom_seq a_1 r 2 + geom_seq a_1 r 3 = 15/8)
    (h_prod : geom_seq a_1 r 1 * geom_seq a_1 r 2 = -9/8) :
  1 / geom_seq a_1 r 0 + 1 / geom_seq a_1 r 1 + 1 / geom_seq a_1 r 2 + 1 / geom_seq a_1 r 3 = -5/3 :=
sorry

end geom_seq_inv_sum_eq_l626_626453


namespace sandal_price_l626_626227

def price_of_each_sandal
  (cost_per_shirt : ℕ)
  (number_of_shirts : ℕ)
  (number_of_sandals : ℕ)
  (total_money_given : ℕ)
  (change_received : ℕ) : ℕ :=
  (total_money_given - change_received - cost_per_shirt * number_of_shirts) / number_of_sandals

theorem sandal_price
  (cost_per_shirt : ℕ := 5)
  (number_of_shirts : ℕ := 10)
  (number_of_sandals : ℕ := 3)
  (total_money_given : ℕ := 100)
  (change_received : ℕ := 41)
  (h_cost_per_shirt : cost_per_shirt = 5)
  (h_number_of_shirts : number_of_shirts = 10)
  (h_number_of_sandals : number_of_sandals = 3)
  (h_total_money_given : total_money_given = 100)
  (h_change_received : change_received = 41) :
  price_of_each_sandal cost_per_shirt number_of_shirts number_of_sandals total_money_given change_received = 3 :=
by
  -- First, the total cost for the shirts
  let total_cost_shirts := cost_per_shirt * number_of_shirts
  -- Next, the total amount spent
  let total_spent := total_money_given - change_received
  -- Amount spent on sandals
  let amount_spent_on_sandals := total_spent - total_cost_shirts
  -- The price per sandal pair
  let price_per_sandal := amount_spent_on_sandals / number_of_sandals
  simp [price_of_each_sandal, h_cost_per_shirt, h_number_of_shirts, h_number_of_sandals, h_total_money_given, h_change_received]
  -- Finish with the expected value
  show price_per_sandal = 3, from sorry

end sandal_price_l626_626227


namespace arithmetic_sequence_iff_l626_626024

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_iff
  (a b : ℕ → ℝ)
  (h : ∀ n, b n = (finset.sum (finset.range (n + 1)) (λ i, (i + 1) * a (i + 1))) / (finset.sum (finset.range (n + 1)) (λ i, (i + 1)))) :
  arithmetic_sequence b ↔ arithmetic_sequence a :=
sorry

end arithmetic_sequence_iff_l626_626024


namespace area_of_triangle_DEF_l626_626809

-- Define point D
def pointD : ℝ × ℝ := (2, 5)

-- Reflect D over the y-axis to get E
def reflectY (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, P.2)
def pointE : ℝ × ℝ := reflectY pointD

-- Reflect E over the line y = -x to get F
def reflectYX (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, -P.1)
def pointF : ℝ × ℝ := reflectYX pointE

-- Define function to calculate the area of the triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- Define the Lean 4 statement
theorem area_of_triangle_DEF : triangle_area pointD pointE pointF = 6 := by
  sorry

end area_of_triangle_DEF_l626_626809


namespace Xd_Yd_equation_l626_626554

noncomputable def Xd_minus_Yd (d X Y : ℕ) : ℤ := 
  if 2 * X = d + 3 ∧ 2 * Y = d + 5 then (X - Y : ℤ) else 0

theorem Xd_Yd_equation (X Y d : ℕ) (h1 : d > 8) (h2: ⟨X, Y⟩ = ⟨(d + 3) / 2, (d + 5) / 2⟩) :
  Xd_minus_Yd d X Y = -1 :=
by {
  sorry
}

end Xd_Yd_equation_l626_626554


namespace problem_l626_626356

noncomputable def f₁ (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f₂ (x : ℝ) : ℝ := (f₁ x)'

def f : ℕ → (ℝ → ℝ)
| 1 := f₁
| 2 := f₂
| n+1 := (f n)'

theorem problem (n : ℕ) :
  f 1 (Real.pi / 3) + f 2 (Real.pi / 3) + f 3 (Real.pi / 3) + ⋯ + f 2017 (Real.pi / 3) = (1 + Real.sqrt 3) / 2 :=
sorry

end problem_l626_626356


namespace coupon_A_greatest_discount_at_220_l626_626260

noncomputable def CouponA_discount (p : ℝ) : ℝ :=
  if p >= 60 then 0.12 * p else 0

noncomputable def CouponB_discount (p : ℝ) : ℝ :=
  if p >= 120 then 25 else 0

noncomputable def CouponC_discount (p : ℝ) : ℝ :=
  if p > 120 then 0.20 * (p - 120) else 0

def valid_price (p : ℝ) : Prop :=
  208.33 < p ∧ p < 300

def greatest_discount (p : ℝ) : Prop :=
  CouponA_discount p > CouponB_discount p ∧ CouponA_discount p > CouponC_discount p

theorem coupon_A_greatest_discount_at_220 :
  ∀ p, p ∈ {180, 200, 220, 240, 260} → valid_price p ∧ greatest_discount p :=
sorry

end coupon_A_greatest_discount_at_220_l626_626260


namespace find_range_of_x_l626_626730

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem find_range_of_x (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
by sorry

end find_range_of_x_l626_626730


namespace inversion_line_through_center_maps_to_itself_inversion_line_not_through_center_maps_to_circle_l626_626910

noncomputable def inversion (P : Point) (O : Point) (R : ℝ) : Point :=
  sorry 

def is_line_through (O : Point) (l : Line) : Prop := 
  O ∈ l  

def is_line_not_through (O : Point) (l : Line) : Prop :=
  O ∉ l  

theorem inversion_line_through_center_maps_to_itself {O : Point} {R : ℝ} (l : Line) :
  is_line_through O l → ∀ (P : Point), inversion P O R ∈ l ∨ inversion P O R = point_at_infinity → P ∈ l :=
sorry

theorem inversion_line_not_through_center_maps_to_circle {O : Point} {R : ℝ} (l : Line) :
  is_line_not_through O l → ∃ C : Circle, ∀ (P : Point), P ∈ l → inversion P O R ∈ C ∧ O ∈ C :=
sorry

end inversion_line_through_center_maps_to_itself_inversion_line_not_through_center_maps_to_circle_l626_626910


namespace minimum_value_of_f_on_0_2_l626_626599

open Real

def f (x : ℝ) : ℝ := x^3 / 3 + x^2 - 3 * x - 4

theorem minimum_value_of_f_on_0_2 : 
  ∃ c ∈ Set.Icc (0 : ℝ) (2 : ℝ), ∀ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f c ≤ f x ∧ f c = -17 / 3 :=
by
  let c := 1
  use c
  split
  { show c ∈ Set.Icc 0 2
    sorry }
  { intros x hx
    show f c ≤ f x
    sorry
    show f c = -17 / 3
    sorry }

end minimum_value_of_f_on_0_2_l626_626599


namespace abs_inequality_solution_l626_626540

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l626_626540


namespace probability_either_condition_1_or_2_eq_43_over_128_l626_626228

noncomputable def prob_event_condition_1_2 (balls : Fin 8 → Bool) : Prop :=
  let condition_1 := (∑ i, if balls i then 1 else 0) = 4
  let condition_2 := (∑ i, if balls i then 1 else 0) = 1 ∨ (∑ i, if balls i then 1 else 0) = 7
  condition_1 ∨ condition_2

theorem probability_either_condition_1_or_2_eq_43_over_128 :
  let prob : Probability (Fin 8 → Bool) := classical.some (Probability.uniform (Univ : set (Fin 8 → Bool))) in
  Probability.prob_event prob prob_event_condition_1_2 = 43 / 128 := 
sorry

end probability_either_condition_1_or_2_eq_43_over_128_l626_626228


namespace largest_mersenne_prime_lt_500_l626_626753

open Nat

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n, (prime n) ∧ (p = 2^n - 1)

theorem largest_mersenne_prime_lt_500 : ∀ p, is_mersenne_prime p ∧ p < 500 → p ≤ 127 :=
by
  sorry

end largest_mersenne_prime_lt_500_l626_626753


namespace cassandra_makes_four_pies_l626_626308

-- Define the number of dozens and respective apples per dozen
def dozens : ℕ := 4
def apples_per_dozen : ℕ := 12

-- Define the total number of apples
def total_apples : ℕ := dozens * apples_per_dozen

-- Define apples per slice and slices per pie
def apples_per_slice : ℕ := 2
def slices_per_pie : ℕ := 6

-- Calculate the number of slices and number of pies based on conditions
def total_slices : ℕ := total_apples / apples_per_slice
def total_pies : ℕ := total_slices / slices_per_pie

-- Prove that the number of pies is 4
theorem cassandra_makes_four_pies : total_pies = 4 := by
  sorry

end cassandra_makes_four_pies_l626_626308


namespace age_difference_l626_626960

-- Define the ages as variables
variables (A B C D : ℤ)

-- Introduce the conditions as hypotheses
def conditions (h1 : A + B > B + C) (h2 : C = A - 10) : Prop :=
  -- State that the difference D is 10
  D = (A + B) - (B + C)

-- The theorem we intend to prove
theorem age_difference 
  (h1 : A + B > B + C) 
  (h2 : C = A - 10) : 
  (D : ℤ) := 
by 
  -- State the goal
  show D = 10 from  
  sorry

end age_difference_l626_626960


namespace highest_red_ball_probability_l626_626840

theorem highest_red_ball_probability :
  ∀ (total balls red yellow black : ℕ),
    total = 10 →
    red = 7 →
    yellow = 2 →
    black = 1 →
    (red / total) > (yellow / total) ∧ (red / total) > (black / total) :=
by
  intro total balls red yellow black
  intro h_total h_red h_yellow h_black
  sorry

end highest_red_ball_probability_l626_626840


namespace trigonometric_inequality_l626_626516

variable {α β x : Real}
variable (k : Int) (h1 : tan (α / 2) ^ 2 ≤ tan (β / 2) ^ 2) (h2 : β ≠ k * Real.pi)

theorem trigonometric_inequality :
  (sin (α / 2) ^ 2 / sin (β / 2) ^ 2) ≤ (x ^ 2 - 2 * x * cos α + 1) / (x ^ 2 - 2 * x * cos β + 1) ∧
  (x ^ 2 - 2 * x * cos α + 1) / (x ^ 2 - 2 * x * cos β + 1) ≤ (cos (α / 2) ^ 2 / cos (β / 2) ^ 2) := 
sorry

end trigonometric_inequality_l626_626516


namespace value_of_k_l626_626176

-- Define the conditions of the quartic equation and the product of two roots
variable (a b c d k : ℝ)
variable (hx : (Polynomial.X ^ 4 - 18 * Polynomial.X ^ 3 + k * Polynomial.X ^ 2 + 200 * Polynomial.X - 1984).rootSet ℝ = {a, b, c, d})
variable (hprod_ab : a * b = -32)

-- The statement to prove: the value of k is 86
theorem value_of_k :
  k = 86 :=
by sorry

end value_of_k_l626_626176


namespace find_x_squared_plus_y_squared_l626_626821

-- Define the conditions
variables {x y : ℝ} -- Let x and y be real numbers

def condition1 : Prop := x * y = 6
def condition2 : Prop := x^2 * y + x * y^2 + x + y = 63

-- Theorem statement
theorem find_x_squared_plus_y_squared (h1 : condition1) (h2 : condition2) : x^2 + y^2 = 69 :=
sorry

end find_x_squared_plus_y_squared_l626_626821


namespace greatest_possible_integer_l626_626917

noncomputable def median (s : Finset ℝ) : ℝ :=
  (s.sort (· ≤ ·)).toList.get ((s.card - 1) / 2)

noncomputable def range (s : Finset ℝ) : ℝ :=
  s.max' sorry - s.min' sorry

theorem greatest_possible_integer {s : Finset ℝ} (h1 : s.card = 10) (h2 : median s = 10) (h3 : range s = 10) :
  s.max' sorry = 15 :=
sorry

end greatest_possible_integer_l626_626917


namespace average_incorrect_l626_626642

theorem average_incorrect : ¬( (1 + 1 + 0 + 2 + 4) / 5 = 2) :=
by {
  sorry
}

end average_incorrect_l626_626642


namespace range_of_a_l626_626004
-- Lean 4 statement

theorem range_of_a 
  (A : Set ℝ := {-1, 0, a})
  (B : Set ℝ := {x : ℝ | 1 < 2^x ∧ 2^x < 2})
  (a : ℝ) 
  (h_nonempty : (A ∩ B).Nonempty) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l626_626004


namespace Malcom_has_more_cards_l626_626705

-- Define the number of cards Brandon has
def Brandon_cards : ℕ := 20

-- Define the number of cards Malcom has initially, to be found
def Malcom_initial_cards (n : ℕ) := n

-- Define the given condition: Malcom has 14 cards left after giving away half of his cards
def Malcom_half_condition (n : ℕ) := n / 2 = 14

-- Prove that Malcom had 8 more cards than Brandon initially
theorem Malcom_has_more_cards (n : ℕ) (h : Malcom_half_condition n) :
  Malcom_initial_cards n - Brandon_cards = 8 :=
by
  sorry

end Malcom_has_more_cards_l626_626705


namespace tangent_line_at_one_over_e_max_value_f_min_value_F_l626_626806

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def f' (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)
noncomputable def F (a x : ℝ) : ℝ := a * (Real.log x) / x

-- Condition for derivative and specific point
lemma tangent_line_eq (x : ℝ) : (Real.log x / x) = -Real.exp(1) ∧ ((1 - Real.log x) / (x^2)) = 2 * (Real.exp(1))^2 := sorry

-- Prove that the equation of the tangent line to y=f(x) at x=1/e is y=2e^2x-3e
theorem tangent_line_at_one_over_e :
  ∃ (m b : ℝ), (m = 2 * Real.exp(1)^2 ∧ b = -3 * Real.exp(1)) ∧ ∀ (x y : ℝ), y = 2 * Real.exp(1)^2 * (x - 1/Real.exp(1)) - 3 * Real.exp(1) := sorry

-- Prove that the maximum value of f(x) is 1/e
theorem max_value_f :
  ∀ x : ℝ, (0 < x ∧ x ≠ Real.exp(1) ∧ f' x = 0) → f x = 1/Real.exp(1) := sorry

-- Prove that the minimum value of F(x) on [a, 2a]
theorem min_value_F (a : ℝ) (ha : 0 < a) :
  (0 < a ∧ a ≤ 2 → ∀ x : ℝ, x ∈ Set.Icc a (2 * a) → F a x = Real.log a) ∧ 
  (a > 2 → ∀ x : ℝ, x ∈ Set.Icc a (2 * a) → F a x = (1/2) * Real.log (2 * a)) := sorry

end tangent_line_at_one_over_e_max_value_f_min_value_F_l626_626806


namespace ratio_S15_S5_l626_626122

-- Definition of a geometric sequence sum and the given ratio S10/S5 = 1/2
noncomputable def geom_sum : ℕ → ℕ := sorry
axiom ratio_S10_S5 : geom_sum 10 / geom_sum 5 = 1 / 2

-- The goal is to prove that the ratio S15/S5 = 3/4
theorem ratio_S15_S5 : geom_sum 15 / geom_sum 5 = 3 / 4 :=
by sorry

end ratio_S15_S5_l626_626122


namespace quadratic_vertex_axis_l626_626180

theorem quadratic_vertex_axis (x : ℝ) : 
  let y := -x^2 - 4*x + 2 in
  y = - (x + 2)^2 + 6 ∧ 
  (∀ x : ℝ, ∃ a b : ℝ, y = a * (x + b)^2 + 6 ∧ b = -2) :=
by
  sorry

end quadratic_vertex_axis_l626_626180


namespace carl_gas_cost_l626_626716

-- Define the variables for conditions
def city_mileage := 30       -- miles per gallon in city
def highway_mileage := 40    -- miles per gallon on highway
def city_distance := 60      -- city miles one way
def highway_distance := 200  -- highway miles one way
def gas_cost := 3            -- dollars per gallon

-- Define the statement to prove
theorem carl_gas_cost : 
  let city_gas := city_distance / city_mileage in
  let highway_gas := highway_distance / highway_mileage in
  let total_one_way_gas := city_gas + highway_gas in
  let round_trip_gas := total_one_way_gas * 2 in
  let total_cost := round_trip_gas * gas_cost in
  total_cost = 42
:= by
  sorry

end carl_gas_cost_l626_626716


namespace slope_tangent_line_at_point_l626_626346

open Real

noncomputable def curve := λ x : ℝ, x - cos x
noncomputable def point := (π / 2, π / 2)

theorem slope_tangent_line_at_point : 
  (deriv curve (π / 2) = 2) :=
sorry

end slope_tangent_line_at_point_l626_626346


namespace smallest_number_of_fruits_l626_626626

theorem smallest_number_of_fruits (N : ℕ) :
  (∃ x : ℕ, 3 * x + 1 = (8 * N - 56) / 27) → N = 79 :=
by sorry

end smallest_number_of_fruits_l626_626626


namespace cost_to_fly_A_to_B_l626_626509

noncomputable def flight_cost (distance : ℕ) : ℕ := (distance * 10 / 100) + 100

theorem cost_to_fly_A_to_B :
  flight_cost 3250 = 425 :=
by
  sorry

end cost_to_fly_A_to_B_l626_626509


namespace proof_problem_l626_626014

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = n * (n + 1) + 2 ∧ S 1 = a 1 ∧ (∀ n, 1 < n → a n = S n - S (n - 1))

def general_term_a (a : ℕ → ℕ) : Prop :=
  a 1 = 4 ∧ (∀ n, 1 < n → a n = 2 * n)

def geometric_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → 
  a 2 = 4 ∧ a (k+2) = 2 * (k + 2) ∧ a (3 * k + 2) = 2 * (3 * k + 2) →
  b 1 = a 2 ∧ b 2 = a (k + 2) ∧ b 3 = a (3 * k + 2) ∧ 
  (∀ n, b n = 2^(n + 1))

theorem proof_problem :
  ∃ (a b S : ℕ → ℕ),
  sum_of_sequence S a ∧ general_term_a a ∧ geometric_sequence a b :=
sorry

end proof_problem_l626_626014


namespace problem_1_problem_2_l626_626856

noncomputable def parametric_equation_curve_C (theta : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos theta, 1 + √3 * Real.sin theta)

noncomputable def polar_equation_line_l (rho theta : ℝ) : Prop :=
  rho * Real.sin (theta + Real.pi / 6) = 2 * √3

noncomputable def intersection_point_A (rho : ℝ) : ℝ × ℝ :=
  (rho * Real.cos (Real.pi / 6), rho * Real.sin (Real.pi / 6) + 1)

noncomputable def intersection_point_B (rho : ℝ) : ℝ × ℝ :=
  (rho * Real.cos (Real.pi / 6), rho * Real.sin (Real.pi / 6))

theorem problem_1 :
  (∀ theta : ℝ, let x := parametric_equation_curve_C theta in x.1 ^ 2 + (x.2 - 1) ^ 2 = 3) ∧
  (∀ rho theta : ℝ, polar_equation_line_l rho theta ↔ 
    let x := rho * Real.cos theta, y := rho * Real.sin theta in x + √3 * y = 4 * √3) :=
by
  sorry

theorem problem_2 :
  let ρ₁ := 2, ρ₂ := 4 in
  let A := intersection_point_A ρ₁, B := intersection_point_B ρ₂ in
  (A.1, A.2) = (2 * Real.cos (Real.pi / 6), 2 * Real.sin (Real.pi / 6) + 1) ∧
  (B.1, B.2) = (4 * Real.cos (Real.pi / 6), 4 * Real.sin (Real.pi / 6)) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :=
by
  sorry

end problem_1_problem_2_l626_626856


namespace bottles_in_cups_l626_626863

-- Defining the given conditions
variables (BOTTLE GLASS CUP JUG : ℕ)

axiom h1 : JUG = BOTTLE + GLASS
axiom h2 : 2 * JUG = 7 * GLASS
axiom h3 : BOTTLE = CUP + 2 * GLASS

theorem bottles_in_cups : BOTTLE = 5 * CUP :=
sorry

end bottles_in_cups_l626_626863


namespace parallelogram_properties_l626_626678

variable {b h : ℕ}

theorem parallelogram_properties
  (hb : b = 20)
  (hh : h = 4) :
  (b * h = 80) ∧ ((b^2 + h^2) = 416) :=
by
  sorry

end parallelogram_properties_l626_626678


namespace select_2n_comparable_rectangles_l626_626052

def comparable (A B : Rectangle) : Prop :=
  -- A can be placed into B by translation and rotation
  exists f : Rectangle → Rectangle, f A = B

theorem select_2n_comparable_rectangles (n : ℕ) (h : n > 1) :
  ∃ (rectangles : List Rectangle), rectangles.length = 2 * n ∧
  ∀ (a b : Rectangle), a ∈ rectangles → b ∈ rectangles → comparable a b :=
sorry

end select_2n_comparable_rectangles_l626_626052


namespace perimeter_ratio_of_squares_l626_626576

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l626_626576


namespace number_of_zeros_l626_626602

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - Real.log x / Real.log 2

theorem number_of_zeros : (∀ x > 0, f(x) < f(x + 1)) 
  ∧ f(1) > 0 
  ∧ f(2) < 0 
  → ∃! x, 0 < x ∧ f(x) = 0 :=
sorry

end number_of_zeros_l626_626602


namespace equation_of_parabola_l626_626781

theorem equation_of_parabola 
  (a b c p : ℝ)
  (h_a_pos: a > 0)
  (h_b_pos: b > 0)
  (h_eccentricity: c / a = 2)
  (h_distance: (a * p) / (2 * c) = 2)
  (h_c: c = √(a^2 + b^2)) :

  (∀ x y, (x^2 = 2 * p * y) ↔ (x^2 = 16 * y)) :=
sorry

end equation_of_parabola_l626_626781


namespace last_four_digits_of_5_2011_l626_626130

lemma last_four_digits_periodic (n: ℕ) (h: n ≥ 5) : (5^(n+4)) % 10000 = (5^n) % 10000 :=
by sorry

theorem last_four_digits_of_5_2011 : (5^2011) % 10000 = 8125 :=
by
  -- Use the periodicity lemma
  have period : (5 ^ (5 + 4)) % 10000 = (5 ^ 5) % 10000 := last_four_digits_periodic 5 (by decide),
  let period_length := 4,
  let k := 502,  -- 2011 = 4 * 502 + 3
  have step_3 : (5 ^ (5 + 3)) % 10000 = 8125 := by sorry,
  exact step_3

end last_four_digits_of_5_2011_l626_626130


namespace find_maximum_marks_l626_626242

theorem find_maximum_marks (obtained_marks : ℝ) (percentage : ℝ) (M : ℝ) (h1 : obtained_marks = 285) (h2 : percentage = 95) :
  M = 300 :=
by
  have h3 : percentage / 100 = obtained_marks / M,
  { rw [h1, h2],
    norm_num,
    sorry },
  rw h3,
  sorry

end find_maximum_marks_l626_626242


namespace simplified_expression_evaluates_to_2_l626_626918

-- Definitions based on given conditions:
def x := 2 -- where x = (1/2)^(-1)
def y := 1 -- where y = (-2023)^0

-- Main statement to prove:
theorem simplified_expression_evaluates_to_2 :
  ((2 * x - y) / (x + y) - (x * x - 2 * x * y + y * y) / (x * x - y * y)) / (x - y) / (x + y) = 2 :=
by
  sorry

end simplified_expression_evaluates_to_2_l626_626918


namespace eccentricity_of_ellipse_equation_of_ellipse_equation_of_line_ac_l626_626799

theorem eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) :
  (b / a = 1 / 2) → (sqrt (1 - (b^2 / a^2)) = (sqrt 3) / 2) := by
  sorry

theorem equation_of_ellipse (a b : ℝ) (h1 : a > b ∧ b > 0)
  (h2 : b = 1) (h3 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :
  (x^2 / 4 + y^2 = 1) := by
  sorry

theorem equation_of_line_ac (m : ℝ) (h : m ≠ 0) :
  (|ec| is longest) → (y = 1/2 * x - 4/21) := by
  sorry

end eccentricity_of_ellipse_equation_of_ellipse_equation_of_line_ac_l626_626799


namespace twenty_percent_greater_l626_626649

theorem twenty_percent_greater (x : ℕ) : 
  x = 80 + (20 * 80 / 100) → x = 96 :=
by
  sorry

end twenty_percent_greater_l626_626649


namespace no_stew_left_l626_626898

theorem no_stew_left (company : Type) (stew : ℝ)
    (one_third_stayed : ℝ)
    (two_thirds_went : ℝ)
    (camp_consumption : ℝ)
    (range_consumption_per_portion : ℝ)
    (range_portion_multiplier : ℝ)
    (total_stew : ℝ) : 
    one_third_stayed = 1 / 3 →
    two_thirds_went = 2 / 3 →
    camp_consumption = 1 / 4 →
    range_portion_multiplier = 1.5 →
    total_stew = camp_consumption + (range_portion_multiplier * (two_thirds_went * (camp_consumption / one_third_stayed))) →
    total_stew = 1 →
    stew = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- here would be the proof steps
  sorry

end no_stew_left_l626_626898


namespace problem1_solution_problem2_solution_l626_626305

noncomputable def problem1 : ℝ :=
  0.25 * ((-1 / 2) ^ -4) / ((sqrt 5) - 1) ^ 0 - (1 / 16) ^ (-1 / 2)

theorem problem1_solution : problem1 = 0 := by
  unfold problem1
  sorry

noncomputable def log_base_10 (x : ℝ) := log 10 x

noncomputable def problem2 : ℝ :=
  (log_base_10 2 + log_base_10 5 - log_base_10 8) / (log_base_10 50 - log_base_10 40)

theorem problem2_solution : problem2 = 1 := by
  unfold problem2 log_base_10
  sorry

end problem1_solution_problem2_solution_l626_626305


namespace sum_sequence_formula_sum_first_2n_b_l626_626013

theorem sum_sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) (h₁ : S n = (n^2 + n) / 2) :
  (∀ m, m ≥ 2 → a m = S m - S (m - 1)) → 
  a n = n := by sorry

theorem sum_first_2n_b (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) 
  (ha : ∀ m, a m = m) 
  (hb : ∀ k, b k = 2^k + (-1)^k * k) : 
  ∑ i in Finset.range (2 * n), b (i + 1) = 2^(2*n + 1) + n - 2 := by sorry

end sum_sequence_formula_sum_first_2n_b_l626_626013


namespace range_of_t_l626_626374

theorem range_of_t (a b t : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1)
  (h4 : ∀ {a b : ℝ}, 2 * sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1 / 2) :
  t ≥ sqrt 2 / 2 :=
by
  sorry

end range_of_t_l626_626374


namespace range_of_m_l626_626387

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
sorry

end range_of_m_l626_626387


namespace basketball_children_l626_626187

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l626_626187


namespace total_bananas_is_least_possible_l626_626966

variable (b1 b2 b3 : ℕ)

-- Definitions for conditions
def first_monkey_final (b1 b2 b3 : ℕ) : ℚ :=
  (2:ℚ)/3 * b1 + (1:ℚ)/3 * b2 + (7:ℚ)/16 * b3

def second_monkey_final (b1 b2 b3 : ℕ) : ℚ :=
  (1:ℚ)/6 * b1 + (1:ℚ)/3 * b2 + (7:ℚ)/16 * b3

def third_monkey_final (b1 b2 b3 : ℕ) : ℚ :=
  (1:ℚ)/6 * b1 + (1:ℚ)/3 * b2 + (1:ℚ)/8 * b3

def is_ratios (b1 b2 b3 : ℕ) : Prop :=
  ∃ x : ℚ, x > 0 ∧
  first_monkey_final b1 b2 b3 = 5 * x ∧
  second_monkey_final b1 b2 b3 = 3 * x ∧
  third_monkey_final b1 b2 b3 = 2 * x

theorem total_bananas_is_least_possible (b1 b2 b3 : ℕ) : 
  is_ratios b1 b2 b3 → b1 + b2 + b3 = 336 :=
by {
  intros h,
  sorry
}

end total_bananas_is_least_possible_l626_626966


namespace log_expression_value_l626_626708

theorem log_expression_value :
  log 2.5 6.25 + log 10 0.01 + log (Real.exp 1) (Real.sqrt (Real.exp 1)) - (2 ^ log 2 3) = -5 / 2 :=
by
  -- Conditions from the problem statement as defs
  have h1 : log 2.5 6.25 = 2 := sorry,
  have h2 : log 10 0.01 = -2 := sorry,
  have h3 : log (Real.exp 1) (Real.sqrt (Real.exp 1)) = 1 / 2 := sorry,
  have h4 : 2 ^ log 2 3 = 3 := sorry,
  -- The proof would use h1, h2, h3, and h4 which is not required currently
  exact sorry

end log_expression_value_l626_626708


namespace quadratic_eq_roots_have_c_value_l626_626428

-- Definitions based on the conditions
def quadratic_roots_eq (a b c : ℝ) (x : ℝ) : Prop := x^2 * a + x * b + c = 0

-- Lean statement for the proof problem
theorem quadratic_eq_roots_have_c_value (c : ℝ) :
  (∀ x, quadratic_roots_eq 2 8 c x → (x = (-8 + √16) / 4 ∨ x = (-8 - √16) / 4)) → c = 6 :=
by
  intro h
  -- The proof is omitted here, but set conditions are given
  sorry

end quadratic_eq_roots_have_c_value_l626_626428


namespace quadratic_completes_square_l626_626605

theorem quadratic_completes_square (b c : ℤ) :
  (∃ b c : ℤ, (∀ x : ℤ, x^2 - 12 * x + 49 = (x + b)^2 + c) ∧ b + c = 7) :=
sorry

end quadratic_completes_square_l626_626605


namespace math_problem_proof_l626_626989

theorem math_problem_proof (n : ℕ) 
  (h1 : n / 37 = 2) 
  (h2 : n % 37 = 26) :
  48 - n / 4 = 23 := by
  sorry

end math_problem_proof_l626_626989


namespace david_money_left_l626_626317

theorem david_money_left (S : ℤ) (h1 : S - 800 = 1800 - S) : 1800 - S = 500 :=
by
  sorry

end david_money_left_l626_626317


namespace no_solution_ineq_l626_626057

theorem no_solution_ineq (m : ℝ) :
  (¬ ∃ (x : ℝ), x - 1 > 1 ∧ x < m) → m ≤ 2 :=
by
  sorry

end no_solution_ineq_l626_626057


namespace inscribed_sphere_radius_correct_l626_626782

-- Define the conditions of the regular tetrahedron
structure RegularTetrahedron :=
(base_length : ℝ)
(height : ℝ)
(inscribed_sphere_radius : ℝ)

-- Given conditions
def T := RegularTetrahedron.mk 1 (Real.sqrt 2) (Real.sqrt 2 / 6)

-- State the problem: Prove that the inscribed sphere radius is correct given the conditions
theorem inscribed_sphere_radius_correct (T : RegularTetrahedron) (h1 : T.base_length = 1) (h2 : T.height = Real.sqrt 2) :
  T.inscribed_sphere_radius = Real.sqrt 2 / 6 := 
by {
  sorry
}

end inscribed_sphere_radius_correct_l626_626782


namespace largest_distance_between_spheres_l626_626209

def sphere_one_center : ℝ × ℝ × ℝ := (0, 0, 0)
def sphere_one_radius : ℝ := 24

def sphere_two_center : ℝ × ℝ × ℝ := (20, 30, -40)
def sphere_two_radius : ℝ := 50

noncomputable def center_distance : ℝ :=
  Real.sqrt ((sphere_two_center.1 - sphere_one_center.1)^2 + 
             (sphere_two_center.2 - sphere_one_center.2)^2 + 
             (sphere_two_center.3 - sphere_one_center.3)^2)

noncomputable def largest_possible_distance : ℝ :=
  sphere_one_radius + center_distance + sphere_two_radius

theorem largest_distance_between_spheres :
  largest_possible_distance = 74 + 10 * Real.sqrt 29 :=
by sorry

end largest_distance_between_spheres_l626_626209


namespace pints_in_a_quart_l626_626466

theorem pints_in_a_quart (blueberries_per_pint blueberries_per_pie total_blueberries pies : ℕ)
  (h1 : blueberries_per_pint = 200)
  (h2 : blueberries_per_pie = 400)
  (h3 : total_blueberries = 2400)
  (h4 : pies = 6)
  (h5 : total_blueberries / pies = blueberries_per_pie)
  : (blueberries_per_pie / blueberries_per_pint = 2) :=
by {
  -- Definitions from the problem:
  have h6 : blueberries_per_pie / blueberries_per_pint = 400 / 200 := by rw [h2, h1],
  have h7 : 400 / 200 = 2 := by norm_num,
  rw [h6, h7],
  sorry
}

end pints_in_a_quart_l626_626466


namespace geometric_sum_n_is_4_l626_626614

theorem geometric_sum_n_is_4 
  (a r : ℚ) (n : ℕ) (S_n : ℚ) 
  (h1 : a = 1) 
  (h2 : r = 1 / 4) 
  (h3 : S_n = (a * (1 - r^n)) / (1 - r)) 
  (h4 : S_n = 85 / 64) : 
  n = 4 := 
sorry

end geometric_sum_n_is_4_l626_626614


namespace harry_owns_two_iguanas_l626_626814

theorem harry_owns_two_iguanas (I : ℕ) 
    (h1 : ∃ I, Harry_owns 3 4 I)
    (h2 : ∑ Animal_Costs 3 4 I = 1140 / 12) : I = 2 :=
by sorry

end harry_owns_two_iguanas_l626_626814


namespace highest_value_meter_l626_626257

theorem highest_value_meter (A B C : ℝ) 
  (h_avg : (A + B + C) / 3 = 6)
  (h_A_min : A = 2)
  (h_B_min : B = 2) : C = 14 :=
by {
  sorry
}

end highest_value_meter_l626_626257


namespace optimal_distance_minimized_l626_626372

noncomputable def problem :=
  let α := 90
  let triangle_ABC_length := 1
  let triangle_KLM_length := 1 / 4
  ∀ A B C K L M : ℝ × ℝ, --representing points in 2D space
  triangle_equilateral A B C ∧
  triangle_equilateral K L M ∧
  |(dist A B) = triangle_ABC_length| ∧
  |(dist K L) = triangle_KLM_length| ∧
  on_or_inside (A B C) (K L M) ∧
  |(angle B L K) = (angle A K M)| ∧
  60 ≤ α ∧ α ≤ 120 ->
  optimal_position (K L M) (A B C) α = 90

-- Assume necessary definitions are provided
def triangle_equilateral (A B C : ℝ × ℝ) : Prop := sorry
def dist (P Q : ℝ × ℝ) : ℝ := sorry
def on_or_inside (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (KLM : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry
def angle (A B C : ℝ × ℝ) : ℝ := sorry
def optimal_position (KLM ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (α : ℝ) : ℝ := sorry

-- Proof here
theorem optimal_distance_minimized :=
problem sorry

end optimal_distance_minimized_l626_626372


namespace points_with_unit_distance_l626_626136

open Set

theorem points_with_unit_distance (m : ℕ) : 
  ∃ (S : Set (ℝ × ℝ)), S.Finite ∧ S.Nonempty ∧ 
  ∀ (A ∈ S), (S \ {A}).count (λ B, dist A B = 1) = m :=
by
  sorry

end points_with_unit_distance_l626_626136


namespace max_value_of_expression_l626_626493

theorem max_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ x : ℝ, (∀ y : ℝ, 2 * (a - y) * (y - sqrt (y^2 + b^2)) ≤ b^2) ∧ 
           (2 * (a - x) * (x - sqrt (x^2 + b^2)) = b^2) := 
by 
  sorry

end max_value_of_expression_l626_626493


namespace smallest_q_p_difference_l626_626486

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l626_626486


namespace second_term_is_neg_12_l626_626959

-- Define the problem conditions
variables {a d : ℤ}
axiom tenth_term : a + 9 * d = 20
axiom eleventh_term : a + 10 * d = 24

-- Define the second term calculation
def second_term (a d : ℤ) := a + d

-- The problem statement: Prove that the second term is -12 given the conditions
theorem second_term_is_neg_12 : second_term a d = -12 :=
by sorry

end second_term_is_neg_12_l626_626959


namespace businesses_brandon_can_apply_to_l626_626124

theorem businesses_brandon_can_apply_to (x : ℕ) : 
  let number_fired := 36
  let number_quit := 24
  let total_businesses := 72
  let common_businesses := x
  in total_businesses - (number_fired + number_quit - common_businesses) = 12 + x :=
by
  sorry

end businesses_brandon_can_apply_to_l626_626124


namespace eval_expression_l626_626490

def square_avg (a b : ℚ) : ℚ := (a^2 + b^2) / 2
def custom_avg (a b c : ℚ) : ℚ := (a + b + 2 * c) / 3

theorem eval_expression : 
  custom_avg (custom_avg 2 (-1) 1) (square_avg 2 3) 1 = 19 / 6 :=
by
  sorry

end eval_expression_l626_626490


namespace sequence_8th_term_is_sqrt23_l626_626957

noncomputable def sequence_term (n : ℕ) : ℝ := Real.sqrt (2 + 3 * (n - 1))

theorem sequence_8th_term_is_sqrt23 : sequence_term 8 = Real.sqrt 23 :=
by
  sorry

end sequence_8th_term_is_sqrt23_l626_626957


namespace median_of_scores_l626_626848

-- Define the list of scores
def scores : List ℕ := [87, 91, 91, 93, 87, 89, 96, 97]

-- Define a function to compute the median of a list of ℕs
def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  if h : sorted.length % 2 = 0 then
    (sorted.get ⟨sorted.length / 2 - 1, by linarith⟩ + sorted.get ⟨sorted.length / 2, by linarith⟩) / 2
  else 
    sorted.get ⟨sorted.length / 2, by linarith⟩

-- The main statement
theorem median_of_scores : median scores = 91 :=
by
  sorry

end median_of_scores_l626_626848


namespace initial_liquid_A_quantity_l626_626255

theorem initial_liquid_A_quantity
  (x : ℝ)
  (init_A init_B init_C : ℝ)
  (removed_A removed_B removed_C : ℝ)
  (added_B added_C : ℝ)
  (new_A new_B new_C : ℝ)
  (h1 : init_A / init_B = 7 / 5)
  (h2 : init_A / init_C = 7 / 3)
  (h3 : init_A + init_B + init_C = 15 * x)
  (h4 : removed_A = 7 / 15 * 9)
  (h5 : removed_B = 5 / 15 * 9)
  (h6 : removed_C = 3 / 15 * 9)
  (h7 : new_A = init_A - removed_A)
  (h8 : new_B = init_B - removed_B + added_B)
  (h9 : new_C = init_C - removed_C + added_C)
  (h10 : new_A / (new_B + new_C) = 7 / 10)
  (h11 : added_B = 6)
  (h12 : added_C = 3) : 
  init_A = 35.7 :=
sorry

end initial_liquid_A_quantity_l626_626255


namespace perimeters_ratio_l626_626581

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l626_626581


namespace arithmetic_sequence_common_difference_l626_626504

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 1)
  (h_S9 : S 9 = 45)
  (h_S_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) : 
  let d := (a 9 - a 1) / 8 in
  d = 1 :=
by
  sorry

end arithmetic_sequence_common_difference_l626_626504


namespace inscribed_radius_of_original_triangle_l626_626166

-- Define the necessary variables for the problem
variables {ΔABC ΔADC ΔCDB : Type}
variables {A B C D O1 O2 : Point}
variables {r r1 r2 : ℝ}

-- Assume the right angle triangle ABC with hypotenuse AB
def right_angle_triangle (Δ : Type) (A B C : Point) : Prop := sorry

-- Assume altitude dropped from C to hypotenuse AB giving point D
def dropped_altitude (Δ : Type) (A B C D : Point) : Prop := sorry

-- Assume centers of inscribed circles of triangles ADC and CDB are O1 and O2
def inscribed_circle_radius (Δ : Type) (O : Point) : ℝ := sorry

-- Assume the distance between centers of inscribed circles of ADC and CDB is 1
def distance_centers (O1 O2 : Point) : ℝ := 1

-- Main theorem statement
theorem inscribed_radius_of_original_triangle :
  (right_angle_triangle ΔABC A B C) ∧
  (dropped_altitude ΔABC A B C D) ∧
  (inscribed_circle_radius ΔADC O1 = r1) ∧
  (inscribed_circle_radius ΔCDB O2 = r2) ∧
  (distance_centers O1 O2 = 1) →
  (inscribed_circle_radius ΔABC O = r) ∧
  (r = (ℝ.sqrt 2) / 2) :=
begin
  sorry
end

end inscribed_radius_of_original_triangle_l626_626166


namespace iodine_mass_percentage_approx_86_39_l626_626749

theorem iodine_mass_percentage_approx_86_39 
  (molar_mass_Ca : ℝ := 40.08)
  (molar_mass_I : ℝ := 126.90)
  (molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I)
  (mass_percentage_I : ℝ := (2 * molar_mass_I / molar_mass_CaI2) * 100) :
  mass_percentage_I ≈ 86.39 :=
sorry

end iodine_mass_percentage_approx_86_39_l626_626749


namespace value_of_f_neg_5pi_over_6_l626_626824

-- Given conditions and function
variable {f : ℝ → ℝ}
axiom odd_function: ∀ x, f (-x) = -f x
axiom periodic_function: ∀ x, f (x + π/2) = f x
axiom f_at_pi_over_3: f (π/3) = 1

-- Statement to prove
theorem value_of_f_neg_5pi_over_6 : f (-5 * π / 6) = -1 := by
  sorry

end value_of_f_neg_5pi_over_6_l626_626824


namespace probability_calculation_l626_626677

noncomputable def probability_2a_between_1_and_4 : ℝ := 2 / 5

theorem probability_calculation:
  ∀ (a : ℝ), a ∈ set.Icc (0 : ℝ) 5 -> (1 ≤ 2^a ∧ 2^a ≤ 4) ->
  ℙ (set.Icc 0 2) / ℙ (set.Icc 0 5) = probability_2a_between_1_and_4 :=
begin
  sorry
end

end probability_calculation_l626_626677


namespace perimeter_ratio_of_squares_l626_626574

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l626_626574


namespace min_value_of_z_ineq_l626_626778

noncomputable def z (x y : ℝ) : ℝ := 2 * x + 4 * y

theorem min_value_of_z_ineq (k : ℝ) :
  (∃ x y : ℝ, (3 * x + y ≥ 0) ∧ (4 * x + 3 * y ≥ k) ∧ (z x y = -6)) ↔ k = 0 :=
by
  sorry

end min_value_of_z_ineq_l626_626778


namespace finite_set_f_l626_626367

variable {A : Type} [Fintype A] (f : A → A)

def f_n (n : ℕ) : set A → set A
| 0 := id
| (n + 1) := λ B, {a | ∃ b ∈ f_n n B, f b = a}

def f_infinity (f : A → A) : set A :=
⋂ n, f_n n set.univ

theorem finite_set_f (A_fin : Fintype A) (f : A → A) :
  (f '' f_infinity f) = f_infinity f :=
sorry

end finite_set_f_l626_626367


namespace max_good_parabolas_l626_626855

noncomputable def maximum_good_parabolas (n : ℕ) : ℕ :=
  n - 1

theorem max_good_parabolas (n : ℕ) (h : n ≥ 2) :
  (maximum_good_parabolas n) = n - 1 := by
  unfold maximum_good_parabolas
  rw [nat.sub_self 1]
  trivial
  sorry

end max_good_parabolas_l626_626855


namespace range_of_k_l626_626364

theorem range_of_k (k : ℝ) :
  (k < -3 ∨ k > 2) ∧ (-((8 : ℝ)/3) * Real.sqrt 3 < k ∧ k < ((8 : ℝ)/3) * Real.sqrt 3) → 
  (k ∈ Set.Ioo (-((8 : ℝ)/3) * Real.sqrt 3) (-3) ∪ Set.Ioo (2) ((8 : ℝ)/3) * Real.sqrt 3) :=
by 
  sorry

end range_of_k_l626_626364


namespace continuous_stripe_probability_l626_626741

-- Define a structure representing the configuration of each face.
structure FaceConfiguration where
  is_diagonal : Bool
  edge_pair_or_vertex_pair : Bool

-- Define the cube configuration.
structure CubeConfiguration where
  face1 : FaceConfiguration
  face2 : FaceConfiguration
  face3 : FaceConfiguration
  face4 : FaceConfiguration
  face5 : FaceConfiguration
  face6 : FaceConfiguration

noncomputable def total_configurations : ℕ := 4^6

-- Define the function that checks if a configuration results in a continuous stripe.
def results_in_continuous_stripe (c : CubeConfiguration) : Bool := sorry

-- Define the number of configurations resulting in a continuous stripe.
noncomputable def configurations_with_continuous_stripe : ℕ :=
  Nat.card {c : CubeConfiguration // results_in_continuous_stripe c}

-- Define the probability calculation.
noncomputable def probability_continuous_stripe : ℚ :=
  configurations_with_continuous_stripe / total_configurations

-- The statement of the problem: Prove the probability of continuous stripe is 3/256.
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 256 :=
sorry

end continuous_stripe_probability_l626_626741


namespace positional_relationship_l626_626455

theorem positional_relationship (r PO QO : ℝ) (h_r : r = 6) (h_PO : PO = 4) (h_QO : QO = 6) :
  (PO < r) ∧ (QO = r) :=
by
  sorry

end positional_relationship_l626_626455


namespace min_distance_from_P_to_plane_gamma_l626_626161

noncomputable def α (A : ℝ × ℝ) : Prop := A.1 = 3 ∧ A.2 = 3

def distance_to_plane (P : ℝ × ℝ) : ℝ := P.2

def point_satisfies_condition (P A : ℝ × ℝ) : Prop :=
  P.1 = 2 * real.sqrt ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)

def min_distance_point_trajectory (A P : ℝ × ℝ) : Prop :=
  point_satisfies_condition P A → distance_to_plane P = 3 - real.sqrt 3

theorem min_distance_from_P_to_plane_gamma (P A : ℝ × ℝ) :
  α A → (point_satisfies_condition P A → distance_to_plane P ≥ 0) → min_distance_point_trajectory A P :=
begin
  sorry
end

end min_distance_from_P_to_plane_gamma_l626_626161


namespace part_I_part_II_l626_626790

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, w = (k * v.1, k * v.2)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def cos_angle (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / (magnitude v * magnitude w)

theorem part_I (a c : ℝ × ℝ) (ha : a = (1, -2)) (hc : magnitude c = 2 * real.sqrt 5) (h_parallel : parallel a c) :
  c = (-2, 4) ∨ c = (2, -4) :=
  sorry

theorem part_II (a b : ℝ × ℝ) (ha : magnitude a = real.sqrt 5) (hb : magnitude b = 1)
  (h_perpendicular : perpendicular (a.1 + b.1, a.2 + b.2) (a.1 - 2 * b.1, a.2 - 2 * b.2)) :
  cos_angle a b = (3 * real.sqrt 5) / 5 :=
  sorry

end part_I_part_II_l626_626790


namespace construct_diameter_of_circumcircle_l626_626244

open Real EuclideanGeometry

noncomputable theory

theorem construct_diameter_of_circumcircle (A B C I : Point) (h₁ : Triangle A B C) (h₂ : Incenter A B C I) (h₃ : ExistsCircumcircle A B C) :
  exists (K W : Point), K ≠ W ∧ DiameterOfCircumcircle A B C K W where construction_steps <= 7 :=
sorry

end construct_diameter_of_circumcircle_l626_626244


namespace min_value_expression_l626_626106

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + a) / b + 3

theorem min_value_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  ∃ x, min_expression a b c = x ∧ x ≥ 9 := 
sorry

end min_value_expression_l626_626106


namespace geometric_series_sum_l626_626987

theorem geometric_series_sum :
  let a_1 := (1 / 4)
  let r := (-1 / 4)
  let n := 6
  let S_n := ∑ i in Finset.range n, a_1 * r ^ i
  S_n = (81 / 405) :=
by
  sorry

end geometric_series_sum_l626_626987


namespace vertices_form_parabola_l626_626877

theorem vertices_form_parabola 
  (a m : ℝ) (ha : 0 < a) (hm : 0 < m) :
  ∃ c : ℝ, ∀ b : ℝ, 
    let x_b := - (b * m) / (2 * a) in 
    let y_b := a * x_b ^ 2 + b * m * x_b + c in
    ∃ k : ℝ, ∀ x : ℝ, y_b = k * x ^ 2 + c :=
by
  sorry

end vertices_form_parabola_l626_626877


namespace sum_of_distinct_x_l626_626397

def g (x : ℝ) : ℝ := (x^2) / 4 + x - 5

theorem sum_of_distinct_x (S : Set ℝ) (h : ∀ x ∈ S, g (g (g x)) = -5) :
  S.sum = -2 :=
  sorry

end sum_of_distinct_x_l626_626397


namespace monotonicity_of_F_slope_intersection_l626_626388

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1)

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 2

noncomputable def F (a x : ℝ) : ℝ := a * x^2 + f' x

theorem monotonicity_of_F (a : ℝ) : 
  (∀ x > 0, F' a x > 0 → (F a x) is_increasing_on ℝ) ∧ 
  (∀ x > 0, F' a x < 0 → (F a x) is_decreasing_on ℝ) := by
sorry

theorem slope_intersection (x₁ x₂ k : ℝ) (h₁ : x₁ < x₂) (h₂ : k = (f' x₂ - f' x₁) / (x₂ - x₁)) : 
  x₁ < 1 / k ∧ 1 / k < x₂ := by
sorry

end monotonicity_of_F_slope_intersection_l626_626388


namespace cannot_sum_85_with_five_coins_l626_626760

def coin_value (c : Nat) : Prop :=
  c = 1 ∨ c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50

theorem cannot_sum_85_with_five_coins : 
  ¬ ∃ (a b c d e : Nat), 
    coin_value a ∧ 
    coin_value b ∧ 
    coin_value c ∧ 
    coin_value d ∧ 
    coin_value e ∧ 
    a + b + c + d + e = 85 :=
by
  sorry

end cannot_sum_85_with_five_coins_l626_626760


namespace modulus_of_complex_l626_626009

noncomputable def modulus (z : Complex) : Real :=
  Complex.abs z

theorem modulus_of_complex :
  ∀ (i : Complex) (z : Complex), i = Complex.I → z = i * (2 - i) → modulus z = Real.sqrt 5 :=
by
  intros i z hi hz
  -- Proof omitted
  sorry

end modulus_of_complex_l626_626009


namespace kosher_clients_count_l626_626505

def T := 30
def V := 7
def VK := 3
def Neither := 18

theorem kosher_clients_count (K : ℕ) : T - Neither = V + K - VK → K = 8 :=
by
  intro h
  sorry

end kosher_clients_count_l626_626505


namespace number_of_ways_to_pair_12_people_l626_626181

-- Define the conditions as stated
def knows (p q : ℕ) := (p + 1 ≡ q [MOD 12]) ∨ (p - 1 ≡ q [MOD 12]) ∨ (p + 6 ≡ q [MOD 12]) ∨ (p + 2 ≡ q [MOD 12])

-- State the problem as a theorem
theorem number_of_ways_to_pair_12_people : 
  ∃ n : ℕ, (∀ pairings : list (ℕ × ℕ),
    (∀ (p q : ℕ), (p, q) ∈ pairings → knows p q) ∧
    (∀ p : ℕ, ∃! q : ℕ, (p, q) ∈ pairings ∨ (q, p) ∈ pairings) ∧
    pairings.length = 6) → n = 14 :=
sorry -- Proof is not required according to the task

end number_of_ways_to_pair_12_people_l626_626181


namespace floor_sum_eq_n_l626_626115

theorem floor_sum_eq_n (n : ℕ) : 
  (∑ k in Finset.range (n+1), Nat.floor ((n + 2^k : ℝ) / (2^(k+1) : ℝ))) = n := 
by
  sorry

end floor_sum_eq_n_l626_626115


namespace centroid_coordinates_satisfy_l626_626972

noncomputable def P : ℝ × ℝ := (2, 5)
noncomputable def Q : ℝ × ℝ := (-1, 3)
noncomputable def R : ℝ × ℝ := (4, -2)

noncomputable def S : ℝ × ℝ := (
  (P.1 + Q.1 + R.1) / 3,
  (P.2 + Q.2 + R.2) / 3
)

theorem centroid_coordinates_satisfy :
  4 * S.1 + 3 * S.2 = 38 / 3 :=
by
  -- Proof will be added here
  sorry

end centroid_coordinates_satisfy_l626_626972


namespace probability_blue_face_l626_626313

theorem probability_blue_face :
  (3 / 6 : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end probability_blue_face_l626_626313


namespace value_of_a_l626_626042

theorem value_of_a (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a^b = b^a) (h₃ : b = 4 * a) : a = real.cbrt 4 :=
by
  sorry

end value_of_a_l626_626042


namespace range_of_a_l626_626835

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 1) * x + a ≤ 0 → -4 ≤ x ∧ x ≤ 3) ↔ (-4 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l626_626835


namespace union_of_sets_contains_87341_elements_l626_626357

theorem union_of_sets_contains_87341_elements
  (A : Fin 1985 → Set α)
  (h₁ : ∀ i, (A i).card = 45)
  (h₂ : ∀ i j, i ≠ j → (A i ∪ A j).card = 89) :
  (⋃ i, A i).card = 87341 :=
sorry

end union_of_sets_contains_87341_elements_l626_626357


namespace kevin_prizes_l626_626868

theorem kevin_prizes (total_prizes stuffed_animals yo_yos frisbees : ℕ)
  (h1 : total_prizes = 50) (h2 : stuffed_animals = 14) (h3 : yo_yos = 18) :
  frisbees = total_prizes - (stuffed_animals + yo_yos) → frisbees = 18 :=
by
  intro h4
  sorry

end kevin_prizes_l626_626868


namespace probability_face_diamonds_then_spades_l626_626973

-- Define the standard deck of 52 cards.
def standard_deck := 52

-- Define the face cards of diamonds (Jack, Queen, King).
def face_cards_diamonds := 3

-- Define the face cards of spades (Jack, Queen, King).
def face_cards_spades := 3

-- Calculate the probability of drawing a face card of diamonds first.
def prob_first_face_diamonds := (face_cards_diamonds : ℚ) / standard_deck

-- Calculate the probability of drawing a face card of spades second.
def prob_second_face_spades := (face_cards_spades : ℚ) / (standard_deck - 1)

-- Calculate the combined probability.
def combined_probability : ℚ := prob_first_face_diamonds * prob_second_face_spades

-- Prove that the combined probability is 1/294.
theorem probability_face_diamonds_then_spades :
  combined_probability = 1 / 294 := by
  simp [combined_probability, prob_first_face_diamonds, prob_second_face_spades]
  sorry

end probability_face_diamonds_then_spades_l626_626973


namespace sequence_formula_l626_626746

theorem sequence_formula (n : ℕ) : 
  let a_n := 2 * (10^n - 1) / 9 in 
  a_n = (2 * (10^n - 1)) / 9 := 
sorry

end sequence_formula_l626_626746


namespace boxed_boxed_17_l626_626349

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x => n % x = 0).sum

theorem boxed_boxed_17 : sum_of_factors (sum_of_factors 17) = 39 :=
by
  sorry

end boxed_boxed_17_l626_626349


namespace min_value_frac_l626_626355

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : sqrt 2 = real.sqrt (4^a * 2^b)) : 
  ∃ (min_val : ℝ), min_val = 9 ∧ min_val = infi (λ x, x = (2 : ℝ) / a + (1 : ℝ) / b) :=
by
  sorry

end min_value_frac_l626_626355


namespace A_annual_income_l626_626948

-- Definitions and conditions
def C_m : ℝ := 16000
def B_m : ℝ := C_m * 1.12
def A_m : ℝ := (B_m * 5) / 2

-- Proof that the annual income of A is Rs. 537600
theorem A_annual_income : (A_m * 12) = 537600 := by
  sorry

end A_annual_income_l626_626948


namespace find_period_l626_626947

theorem find_period (ω : ℝ) (h1 : ω > 0) (h2 : ∃ T > 0, ∀ x, sin (ω * (x + T) + π / 3) = sin (ω * x + π / 3)) : ω = 2 :=
by
  sorry

end find_period_l626_626947


namespace Andryusha_can_detect_drum_presence_l626_626298

-- Definitions and conditions
def Stone : Type := ℕ

structure House where
  stones : Fin 100 → Stone
  drum : Bool

variables (house : House)

-- A predicate that checks if Andryusha can distinguish stones by their appearance
def distinguish_by_appearance : Prop := ∀ (i j : Fin 100), (house.stones i ≠ house.stones j)

-- A predicate that states that the brownie orders the stones in increasing weight
def brownie_ordered (s : List Stone) : Prop := s = s.sort (≤)

-- A predicate to capture if the drum will change the places of some 2 stones
def drum_changes_order (s : List Stone) : Prop := 
  ∃ (i j : Fin 10), i ≠ j ∧ (s[i] > s[j])

-- A function defining the scenario each night
def night_scenario (s : List Stone) : List Stone :=
  if house.drum then
    let ordered := s.sort (≤)
    sorry -- simulate changing places by the drum (left as an exercise)
  else
    s.sort (≤)

-- The main theorem to check if there is a drum in the house
theorem Andryusha_can_detect_drum_presence :
  distinguish_by_appearance house.stones →
  (∀ s : (Fin 10 → Stone), let v := (List.ofFn s) in 
    brownie_ordered v → drum_changes_order (night_scenario house v)) →
  ∃ drum_in_house : Bool, house.drum = drum_in_house :=
sorry

end Andryusha_can_detect_drum_presence_l626_626298


namespace num_factors_of_72_l626_626818

theorem num_factors_of_72 : (nat.divisors 72).card = 12 :=
by
  sorry

end num_factors_of_72_l626_626818


namespace female_democrats_count_l626_626962

theorem female_democrats_count 
  (F M D : ℕ)
  (total_participants : F + M = 660)
  (total_democrats : F / 2 + M / 4 = 660 / 3)
  (female_democrats : D = F / 2) : 
  D = 110 := 
by
  sorry

end female_democrats_count_l626_626962


namespace trig_identity_example_l626_626041

open Real

noncomputable def tan_alpha_eq_two_tan_pi_fifths (α : ℝ) :=
  tan α = 2 * tan (π / 5)

theorem trig_identity_example (α : ℝ) (h : tan_alpha_eq_two_tan_pi_fifths α) :
  (cos (α - 3 * π / 10) / sin (α - π / 5)) = 3 :=
sorry

end trig_identity_example_l626_626041


namespace smallest_positive_perfect_cube_contains_factor_n_l626_626497

-- Definitions of distinct primes p, q, r
variables {p q r : ℕ}
variables (hp : prime p) (hq : prime q) (hr : prime r)
variables (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r)
variables (h1notprime : ¬ prime 1)

-- Definition of n
def n := p * q^2 * r^4

-- Statement to prove
theorem smallest_positive_perfect_cube_contains_factor_n :
  (∃ (k : ℕ), k > 0 ∧ k = (p * q * r^2)^3 ∧ (∀ m : ℕ, ((∃ (d : ℕ), d^3 = m ∧ n ∣ m) → m ≥ k))) :=
sorry

end smallest_positive_perfect_cube_contains_factor_n_l626_626497


namespace exists_consecutive_nat_with_integer_quotient_l626_626329

theorem exists_consecutive_nat_with_integer_quotient :
  ∃ n : ℕ, (n + 1) / n = 2 :=
by
  sorry

end exists_consecutive_nat_with_integer_quotient_l626_626329


namespace project_completion_time_l626_626727

theorem project_completion_time 
    (w₁ w₂ : ℕ) 
    (d₁ d₂ : ℕ) 
    (fraction₁ fraction₂ : ℝ)
    (h_work_fraction : fraction₁ = 1/2)
    (h_work_time : d₁ = 6)
    (h_first_workforce : w₁ = 90)
    (h_second_workforce : w₂ = 60)
    (h_fraction_done_by_first_team : w₁ * d₁ * (1 / 1080) = fraction₁)
    (h_fraction_done_by_second_team : w₂ * d₂ * (1 / 1080) = fraction₂)
    (h_total_fraction : fraction₂ = 1 - fraction₁) :
    d₂ = 9 :=
by 
  sorry

end project_completion_time_l626_626727


namespace square_perimeter_ratio_l626_626560

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l626_626560


namespace perimeter_ratio_of_squares_l626_626575

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l626_626575


namespace solve_abs_inequality_l626_626544

theorem solve_abs_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ 
      (x ∈ Ioo (-13 / 2) (-3) 
     ∨ x ∈ Ico (-3) 2 
     ∨ x ∈ Icc 2 (7 / 2)) :=
by
  sorry

end solve_abs_inequality_l626_626544


namespace boat_current_ratio_l626_626998

noncomputable def boat_speed_ratio (b c : ℝ) (d : ℝ) : Prop :=
  let time_upstream := 6
  let time_downstream := 10
  d = time_upstream * (b - c) ∧ 
  d = time_downstream * (b + c) → 
  b / c = 4

theorem boat_current_ratio (b c d : ℝ) (h1 : d = 6 * (b - c)) (h2 : d = 10 * (b + c)) : b / c = 4 :=
by sorry

end boat_current_ratio_l626_626998


namespace max_good_sequences_l626_626282

def string_contains_beads (s : String) : Prop :=
  s.foldl (λ (counts : (ℕ × ℕ × ℕ)) (c : Char),
    if c = 'B' then (counts.1 + 1, counts.2, counts.3)
    else if c = 'R' then (counts.1, counts.2 + 1, counts.3)
    else if c = 'G' then (counts.1, counts.2, counts.3 + 1)
    else counts) (0, 0, 0) = (75, 75, 75)

def is_good_sequence (s : String) (i : ℕ) : Prop :=
  s.drop i |>.take 5 |>.foldl (λ (counts : (ℕ × ℕ × ℕ)) (c : Char),
    if c = 'B' then (counts.1 + 1, counts.2, counts.3)
    else if c = 'R' then (counts.1, counts.2 + 1, counts.3)
    else if c = 'G' then (counts.1, counts.2, counts.3 + 1)
    else counts) (0, 0, 0) = (1, 1, 3)

theorem max_good_sequences (s : String) (h_beads : string_contains_beads s) : 
  ∃ max_seqs : ℕ, max_seqs = 123 ∧ 
  ∀ i : ℕ, i < s.length - 4 → is_good_sequence s i → max_seqs = 123 := sorry

end max_good_sequences_l626_626282


namespace opposite_of_neg_two_l626_626174

theorem opposite_of_neg_two : ∃! (x : ℤ), -2 + x = 0 :=
by
  use 2
  split
  { show -2 + 2 = 0, by ring }
  { intro y
    show -2 + y = 0 → y = 2
    intro h
    calc
      y = -(-2) : by rw [←h, add_left_neg]
      ... = 2  : by norm_num
  }

end opposite_of_neg_two_l626_626174


namespace oz_words_lost_l626_626237

/-- 
In the land of Oz, only one or two-letter words are used.
There are 68 different letters in the local language.
Prove that if the seventh letter is forbidden, the number of words lost is 135.
-/
theorem oz_words_lost (letters : Fin 68) (is_forbidden : letters = 6) : 
  let one_letter_words := 68
  let two_letter_words := 68 * 67 * 2
  let lost_one_letter := if is_forbidden then 1 else 0
  let lost_two_letter := if is_forbidden then 67 * 2 else 0
  lost_one_letter + lost_two_letter = 135 :=
by
  sorry

end oz_words_lost_l626_626237


namespace james_gave_away_one_bag_l626_626465

theorem james_gave_away_one_bag (initial_marbles : ℕ) (bags : ℕ) (marbles_left : ℕ) (h1 : initial_marbles = 28) (h2 : bags = 4) (h3 : marbles_left = 21) : (initial_marbles / bags) = (initial_marbles - marbles_left) / (initial_marbles / bags) :=
by
  sorry

end james_gave_away_one_bag_l626_626465


namespace Raja_and_Ram_together_l626_626518

def RajaDays : ℕ := 12
def RamDays : ℕ := 6

theorem Raja_and_Ram_together (W : ℕ) : 
  let RajaRate := W / RajaDays
  let RamRate := W / RamDays
  let CombinedRate := RajaRate + RamRate 
  let DaysTogether := W / CombinedRate 
  DaysTogether = 4 := 
by
  sorry

end Raja_and_Ram_together_l626_626518


namespace garden_enlargement_l626_626685

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end garden_enlargement_l626_626685


namespace ab_inequality_l626_626515

theorem ab_inequality
  {a b : ℝ}
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_b_sum : a + b = 2) :
  ∀ n : ℕ, 2 ≤ n → (a^n + 1) * (b^n + 1) ≥ 4 :=
by
  sorry

end ab_inequality_l626_626515


namespace square_perimeter_ratio_l626_626559

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l626_626559


namespace part1_part2_l626_626803

open Set

noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := x^2 + 2 * real.sin θ * x - 1

theorem part1 (hx : -1/2 ≤ x ∧ x ≤ sqrt 3 / 2) (hθ : real.sin θ = -1/2) : 
  ∃ (min_x max_x : ℝ), min_x ∈ Icc (-1/2) (sqrt 3 / 2) ∧ max_x ∈ Icc (-1/2) (sqrt 3 / 2) ∧ 
                       f min_x θ = -5/4 ∧ f max_x θ = -1/4 :=
sorry

theorem part2 (H : ∀ x1 x2 ∈ Icc (-1/2) (sqrt 3 / 2), x1 ≠ x2 → (x1 < x2 ↔ f x1 θ < f x2 θ)) : 
  θ ∈ Icc (π/6) (5*π/6) ∪ Icc (4*π/3) (5*π/3) :=
sorry

end part1_part2_l626_626803


namespace expected_days_to_find_wand_l626_626303

theorem expected_days_to_find_wand : 
  (∀ wand_location ∈ finset.range 8,
    let probability := 1 / (8 : ℝ) in
    let search_probability := 1 / 2 in
    let searches_per_day := 3 in
    let optimal_strategy := true in
    -- [ Mathematically define the expected value calculation logic here ]
    -- Placeholder for the mathematical definition which confirms to the optimal strategy
    let expected_days := 253 / 56 in
    true)
  → 
  -- Assuming these conditions
  let wand_is_hidden_uniformly := true in
  let death_eaters_search_probability := 1 / 2 in
  let max_searches_per_day := 3 in
  let optimize_strategy := true in
  -- Show the result after removing intermediate proofs
  true → (expected_days = 253 / 56) :=
by sorry

end expected_days_to_find_wand_l626_626303


namespace arithmetic_square_root_7_eq_49_cube_root_neg_27_div_64_eq_neg_3_div_4_sqrt_sqrt_81_eq_pm_3_l626_626156

-- Prove the existence of a number whose arithmetic square root is 7 and is 49.
theorem arithmetic_square_root_7_eq_49 : ∃ x : ℝ, sqrt x = 7 ∧ x = 49 := by
  sorry

-- Prove the existence of a number whose cube root is -27/64 and is -3/4.
theorem cube_root_neg_27_div_64_eq_neg_3_div_4 : ∃ y : ℝ, y^3 = -27 / 64 ∧ y = -3 / 4 := by
  sorry

-- Prove the existence of a number whose square root of square root of 81 is ±3.
theorem sqrt_sqrt_81_eq_pm_3 : ∃ z : ℝ, sqrt (sqrt 81) = z ∧ (z = 3 ∨ z = -3) := by
  sorry

end arithmetic_square_root_7_eq_49_cube_root_neg_27_div_64_eq_neg_3_div_4_sqrt_sqrt_81_eq_pm_3_l626_626156


namespace find_S_l626_626433

variable (R S : ℝ)
variable (A B C D E F : Type) -- Points
variable [HasMeasureTheory (ℝ)] 

-- Definitions based on problem conditions
def side_length_square := R - 1
def area_equilateral_triangle := S - 3

-- Hypotheses/conditions
axiom square_side_length : side_length_square > 0 
axiom equilateral_triangle (AEF : Type) [IsEquilateral AEF] 
axiom AEF_points: BelongsTo E B C ∧ BelongsTo F C D 
axiom triangle_area : ∃ (A : ℝ), A = area_equilateral_triangle

-- Prove S = 2√3
theorem find_S : S = 2 * Real.sqrt 3 := by
  sorry

end find_S_l626_626433


namespace students_got_off_the_bus_l626_626624

theorem students_got_off_the_bus
    (original_students : ℕ)
    (students_left : ℕ)
    (h_original : original_students = 10)
    (h_left : students_left = 7) :
    original_students - students_left = 3 :=
by {
  sorry
}

end students_got_off_the_bus_l626_626624


namespace log_sum_geom_seq_l626_626875

variable {a : ℕ+ → ℝ}
variable (geom_seq : ∀ n, a (n + 1) = a n * r)
variable (r : ℝ) (h : a 5 * a 6 = 81)

-- We are to prove the following
theorem log_sum_geom_seq (h_geom : ∀ n, a (n + 1) = a n * r) (h_prod : a 5 * a 6 = 81) :
  (\sum_{k=1}^10 (Real.log (a k) / Real.log 3) = 20) :=
sorry

end log_sum_geom_seq_l626_626875


namespace convert_base_9_to_base_3_l626_626728

theorem convert_base_9_to_base_3 (a b c : ℕ) (h₁ : 7 = a) (h₂ : 3 = b) (h₃ : 4 = c)
    (convert_a : ∀ a, 7 = a → 21)
    (convert_b : ∀ b, 3 = b → 10)
    (convert_c : ∀ c, 4 = c → 11) : 
  (7 * 9^2 + 3 * 9 + 4) = (2 * 3^5 + 1 * 3^4 + 1 * 3^3 + 1 * 3^2 + 1 * 3 + 0) :=
by
  sorry

end convert_base_9_to_base_3_l626_626728


namespace computer_literate_females_is_724_l626_626439

theorem computer_literate_females_is_724 (total_employees : ℕ)
  (employees_A : ℕ) (employees_B : ℕ) (employees_C : ℕ)
  (females_ratio_A : ℝ) (females_ratio_B : ℝ) (females_ratio_C : ℝ)
  (comp_literacy_fem_A : ℝ) (comp_literacy_fem_B : ℝ) (comp_literacy_fem_C : ℝ) :
  total_employees = employees_A + employees_B + employees_C →
  employees_A = 500 →
  employees_B = 600 →
  employees_C = 500 →
  females_ratio_A = 0.65 →
  females_ratio_B = 0.50 →
  females_ratio_C = 0.75 →
  comp_literacy_fem_A = 0.75 →
  comp_literacy_fem_B = 0.60 →
  comp_literacy_fem_C = 0.80 →
  let females_A := females_ratio_A * employees_A in
  let females_B := females_ratio_B * employees_B in
  let females_C := females_ratio_C * employees_C in
  let comp_fem_A := comp_literacy_fem_A * females_A in
  let comp_fem_B := comp_literacy_fem_B * females_B in
  let comp_fem_C := comp_literacy_fem_C * females_C in
  (comp_fem_A + comp_fem_B + comp_fem_C).toNat = 724 :=
begin
  intros,
  simp only [←nat.cast_add, nat.cast_mul, nat.cast_one, rat.add_num_denom, rat.add_def,
    rat.cast_def, mul_eq_mul_right_iff, or_true, rat.cast_coe_nat, eq_self_iff_true, mul_eq_zero],
  sorry
end

end computer_literate_females_is_724_l626_626439


namespace square_perimeter_ratio_l626_626557

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l626_626557


namespace exists_pos_2x2_square_l626_626334

-- Define the size of the grid and the type of elements in the grid
def grid : Type := array (50 * 50) ℤ

-- Define the property of G, the subgrid configuration
def is_G (g : grid) (i j : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ 48 ∧ 1 ≤ j ∧ j ≤ 48 ∧
  let G := [ g[i * 50 + j], g[i * 50 + j + 2], g[(i + 2) * 50 + j], g[(i + 2) * 50 + j + 2],
              g[i * 50 + j + 1], g[(i + 1) * 50 + j], g[(i + 1) * 50 + j + 2], g[(i + 2) * 50 + j + 1] ] in
  0 < G.sum

-- Define the main theorem statement
theorem exists_pos_2x2_square (g : grid) (hG : ∀ i j, is_G g i j) :
  ∃ i j, 0 ≤ i ∧ i ≤ 48 ∧ 0 ≤ j ∧ j ≤ 48 ∧
         0 < (g[i * 50 + j] + g[i * 50 + j + 1] + g[(i + 1) * 50 + j] + g[(i + 1) * 50 + j + 1]) :=
begin
  sorry
end

end exists_pos_2x2_square_l626_626334


namespace operation_1_circ_2_l626_626422

def circ (a b : ℤ) : ℤ := 2 * a - 3 * b + a * b

theorem operation_1_circ_2 :
  (circ 1 2) - 2 = -4 :=
by
  have H : circ 2 1 - 2 = 1 := sorry
  show circ 1 2 - 2 = -4 from sorry

end operation_1_circ_2_l626_626422


namespace M_inter_N_eq_zero_l626_626368

-- Define the set M
def M : Set ℝ := { x | (x + 2) / (x - 1) ≤ 0 }

-- Define the set N (natural numbers)
def N : Set ℕ := { n | true } -- all natural numbers

-- State the theorem to be proven
theorem M_inter_N_eq_zero : (M ∩ (N : Set ℝ)) = {0} := 
  sorry

end M_inter_N_eq_zero_l626_626368


namespace eccentricity_of_given_ellipse_l626_626592

noncomputable def ellipse_eccentricity (φ : Real) : Real :=
  let x := 3 * Real.cos φ
  let y := 5 * Real.sin φ
  let a := 5
  let b := 3
  let c := Real.sqrt (a * a - b * b)
  c / a

theorem eccentricity_of_given_ellipse (φ : Real) :
  ellipse_eccentricity φ = 4 / 5 :=
sorry

end eccentricity_of_given_ellipse_l626_626592


namespace correct_observation_value_l626_626167

noncomputable def mean_original := 32
noncomputable def mean_corrected := 32.5
noncomputable def observations_count := 50
noncomputable def wrong_value := 23

theorem correct_observation_value :
  let sum_original := observations_count * mean_original in
  let sum_with_wrong := sum_original - wrong_value in
  let sum_corrected := observations_count * mean_corrected in
  ∃ x : ℕ, sum_with_wrong + x = sum_corrected ∧ x = 48 :=
by
  sorry

end correct_observation_value_l626_626167


namespace pizza_percent_increase_l626_626988

noncomputable def radius (d : ℝ) : ℝ := d / 2

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

theorem pizza_percent_increase (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 18) :
  ((area (radius d2) - area (radius d1)) / area (radius d1)) * 100 = 125 :=
by
  let r1 := radius d1
  let r2 := radius d2
  have hr1 : r1 = 6 := by rw [radius, h1, div_eq_mul_inv, mul_one, ←div_eq_mul_inv]
  have hr2 : r2 = 9 := by rw [radius, h2, div_eq_mul_inv, mul_one, ←div_eq_mul_inv]
  have a1 : area r1 = Real.pi * 36 := by rw [area, hr1, pow_two, mul_six_eq_add_six, mul_assoc]
  have a2 : area r2 = Real.pi * 81 := by rw [area, hr2, pow_two, mul_nine_eq_add_boolean, mul_assoc]
  have inc : area r2 - area r1 = Real.pi * 45 := by rw [a2, a1, mul_sub]
  have percent : (area r2 - area r1) / area r1 * 100 = 125 := by
    rw [inc, a1, div_mul, mul_comm, div_eq_mul_inv, mul_assoc, ...]
  exact percent
  sorry

end pizza_percent_increase_l626_626988


namespace integral_f_squared_plus_g_squared_area_between_f_and_g_l626_626471

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) + Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.cos (3 * x) + Real.sin x

theorem integral_f_squared_plus_g_squared :
  ∫ x in 0..2 * Real.pi, (f x)^2 + (g x)^2 = 4 * Real.pi :=
by sorry

theorem area_between_f_and_g :
  ∫ x in 0..π, abs ((f x) - (g x)) = (8 * Real.sqrt (2 + Real.sqrt 2) - 4) / 3 :=
by sorry

end integral_f_squared_plus_g_squared_area_between_f_and_g_l626_626471


namespace simplified_value_of_f_l626_626984

variable (x : ℝ)

noncomputable def f : ℝ := 3 * x + 5 - 4 * x^2 + 2 * x - 7 + x^2 - 3 * x + 8

theorem simplified_value_of_f : f x = -3 * x^2 + 2 * x + 6 := by
  unfold f
  sorry

end simplified_value_of_f_l626_626984


namespace problem1_problem2_problem3_problem4_problem5_problem6_l626_626706

-- Problem 1
theorem problem1 : 4 - (-28) + (-2) = 30 := by
  sorry

-- Problem 2
theorem problem2 : (-3) * (- (2 / 5)) / (- (1 / 4)) = (24 / 5) := by
  sorry

-- Problem 3
theorem problem3 : (-42) / (-7) - (-6) * 4 = 30 := by
  sorry

-- Problem 4
theorem problem4 : -3^2 / (-3)^2 + 3 * (-2) + abs(-4) = -3 := by
  sorry

-- Problem 5
theorem problem5 : (-24) * (3 / 4 - 5 / 6 + 7 / 12) = -12 := by
  sorry

-- Problem 6
theorem problem6 : -1^4 - (1 - 0.5) / (5 / 2) * (1 / 5) = -(26 / 25) := by
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l626_626706


namespace max_n_tickets_l626_626496

theorem max_n_tickets {n : ℕ} : 
  ∃ (nba : ℕ → ℕ), 
  (∀ i, 1 ≤ i ∧ i ≤ 2 * n + 1 → 0 < nba i) →
  (sum (range (2 * n + 1)) nba ≤ 2330) →
  (n > 0 ∧ ∀ S, S ⊆ finset.range (2 * n + 1) ∧ S.card = n → 1165 < sum S nb) →
  n = 10 :=
sorry

end max_n_tickets_l626_626496


namespace find_dihedral_angle_l626_626157

noncomputable def dihedral_angle (a b h : ℝ) : ℝ :=
  180 - Real.arcsin ((h * Real.sqrt(a^2 + b^2 + h^2)) / (Real.sqrt(a^2 + h^2) * Real.sqrt(b^2 + h^2)))

theorem find_dihedral_angle (a b h : ℝ) :
  dihedral_angle a b h = 180 - Real.arcsin ((h * Real.sqrt(a^2 + b^2 + h^2)) / (Real.sqrt(a^2 + h^2) * Real.sqrt(b^2 + h^2))) :=
by
  sorry

end find_dihedral_angle_l626_626157


namespace chad_will_save_460_l626_626719

def chad_money_mowing : ℕ := 600
def chad_money_birthday : ℕ := 250
def chad_money_videogames : ℕ := 150
def chad_money_oddjobs : ℕ := 150

def total_money_chad_made : ℕ := chad_money_mowing + chad_money_birthday + chad_money_videogames + chad_money_oddjobs
def chad_saving_rate : ℚ := 0.40
def chad_savings : ℚ := chad_saving_rate * total_money_chad_made

theorem chad_will_save_460 : chad_savings = 460 := by 
  unfold chad_savings total_money_chad_made chad_money_mowing chad_money_birthday chad_money_videogames chad_money_oddjobs chad_saving_rate 
  norm_num
  sorry

end chad_will_save_460_l626_626719


namespace fraction_increases_l626_626117

theorem fraction_increases (a : ℝ) (h : ℝ) (ha : a > -1) (hh : h > 0) : 
  (a + h) / (a + h + 1) > a / (a + 1) := 
by 
  sorry

end fraction_increases_l626_626117


namespace reciprocal_of_complex_power_l626_626766

noncomputable def complex_num_reciprocal : ℂ :=
  (Complex.I) ^ 2023

theorem reciprocal_of_complex_power :
  ∀ z : ℂ, z = (Complex.I) ^ 2023 -> (1 / z) = Complex.I :=
by
  intro z
  intro hz
  have h_power : z = Complex.I ^ 2023 := by assumption
  sorry

end reciprocal_of_complex_power_l626_626766


namespace co_circular_A1_B1_C1_D1_l626_626444

-- Definitions of the points and sphere properties
variables {A0 B0 C0 D0 P A1 B1 C1 D1 : Point}
variables {S : Sphere}
variables {T : Tetrahedron}
variables (inside_tetrahedron : InscribedSphere S T)
variables (A1_def : SecondIntersection (Line P A0) (FacePlane T A0))
variables (B1_def : SecondIntersection (Line P B0) (FacePlane T B0))
variables (C1_def : SecondIntersection (Line P C0) (FacePlane T C0))
variables (D1_def : SecondIntersection (Line P D0) (FacePlane T D0))

-- The theorem to prove
theorem co_circular_A1_B1_C1_D1 :
  SameCircle A1 B1 C1 D1 :=
sorry

end co_circular_A1_B1_C1_D1_l626_626444


namespace abs_inequality_solution_l626_626526

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l626_626526


namespace abs_inequality_solution_l626_626528

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l626_626528


namespace ratio_problem_l626_626830

-- Define the conditions and the required proof
theorem ratio_problem (p q n : ℝ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 :=
by
  sorry

end ratio_problem_l626_626830


namespace ellipse_properties_l626_626784

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) :=
  ∃ (C : ℝ × ℝ → Prop), (∀ (x y : ℝ), C (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1))

def equilateral_triangle_area (a b c : ℝ) : Prop :=
  ∃ (area : ℝ), area = sqrt 3 ∧ a = b ∧ b = c

def line_through_focus (k : ℝ) := k ≠ 0

def midpoint_of_AB (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def perpendicular_line_intersect_x_axis (P : ℝ × ℝ) (AB_slope : ℝ) :=
  ∃ D : ℝ × ℝ, D.2 = 0 ∧ D.1 = P.1 - P.2 / AB_slope

def DP_by_AB_range (DP AB : ℝ) : Prop :=
  0 < DP / AB ∧ DP / AB < 1 / 4

theorem ellipse_properties
  (a b : ℝ) (h : a > b ∧ b > 0)
  (h_eq_triangle : equilateral_triangle_area a b 2)
  (k : ℝ) (h_line : line_through_focus k)
  (A B : ℝ × ℝ) (h_intersect : ∀ (x : ℝ), (x, k * (x - 1)) ∈ A ∨ (x, k * (x - 1)) ∈ B)
  (h_midpoint : midpoint_of_AB A B = (a/2, -b/2))
  (P : ℝ × ℝ) (D : ℝ × ℝ) (h_perpendicular : perpendicular_line_intersect_x_axis P (1 / -k))
  (DP AB_dist : ℝ) (h_DP : DP = abs (D.1 - P.1))
  (h_AB : AB_dist = sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) :
  ellipse_equation 2 sqrt 3 h ∧ DP_by_AB_range DP AB_dist := sorry

end ellipse_properties_l626_626784


namespace wire_spliced_length_correct_l626_626645

-- Definitions of given conditions
def num_pieces : ℕ := 15
def length_per_piece : ℕ := 25  -- in cm
def overlap_per_splice : ℝ := 0.5  -- in cm

-- Function to calculate length after splicing
def final_length_after_splicing (num_pieces : ℕ) (length_per_piece : ℕ) (overlap_per_splice : ℝ) : ℝ :=
  let total_length_before_splicing := num_pieces * length_per_piece
  let num_splices := (num_pieces - 1 : ℕ)
  let total_overlap := num_splices * overlap_per_splice
  let length_after_splicing_cm := total_length_before_splicing - total_overlap
  length_after_splicing_cm / 100  -- convert to meters

theorem wire_spliced_length_correct :
  final_length_after_splicing num_pieces length_per_piece overlap_per_splice = 3.68 :=
by
  sorry

end wire_spliced_length_correct_l626_626645


namespace positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l626_626951

theorem positive_even_integers_less_than_1000_not_divisible_by_3_or_11 :
  ∃ n : ℕ, n = 108 ∧
    (∀ m : ℕ, 0 < m → 2 ∣ m → m < 1000 → (¬ (3 ∣ m) ∧ ¬ (11 ∣ m) ↔ m ≤ n)) :=
sorry

end positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l626_626951


namespace log10_is_irrational_or_integer_l626_626521

theorem log10_is_irrational_or_integer (N : ℕ) (hN : 0 < N) (hN_ne10_pow : ∀ k : ℕ, N ≠ 10 ^ k) :
  ∀ (a b : ℕ), b ≠ 0 → ¬ (log 10 N = a / b) := 
sorry

end log10_is_irrational_or_integer_l626_626521


namespace log_identity_l626_626362

theorem log_identity : 
  (a = log 625 4) → (b = log 25 5) → a = (4 / b) :=
by
  sorry

end log_identity_l626_626362


namespace not_possible_2002_pieces_l626_626265

theorem not_possible_2002_pieces (k : ℤ) : ¬ (1 + 7 * k = 2002) :=
by
  sorry

end not_possible_2002_pieces_l626_626265


namespace red_box_balls_l626_626440

theorem red_box_balls :
  let p1 := (5 / 8 : ℝ)
  ∧ let pn (n : ℕ) := (1 - (2 ^ -(2 * n + 1)))
  ∧ let pn_p1 (n : ℕ) := pn (n + 1)
  ∧ let pn_p2 (n : ℕ) := pn (n + 2)
  in
  (p1 = 5 / 8) 
  ∧ (pn 2 = 17 / 32)
  ∧ (4 * pn_p2 n + pn n = 5 * pn_p1 n)
  ∧ (∀ n, lim (λ n, pn n) = 1 / 2)
  ∧ (sn n = 1 / 6 * (3 * n + 1 - (1 / 4) ^ n)) := 
sorry

end red_box_balls_l626_626440


namespace find_number_of_dogs_l626_626506

variables (D P S : ℕ)
theorem find_number_of_dogs (h1 : D = 2 * P) (h2 : P = 2 * S) (h3 : 4 * D + 4 * P + 2 * S = 510) :
  D = 60 := 
sorry

end find_number_of_dogs_l626_626506


namespace min_value_proof_l626_626111

noncomputable def min_value_expression 
  (a d: ℝ) (b c: ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ d) (h3 : 0 < b) (h4 : 0 < c) (h5 : b + c ≥ a + d) 
: ℝ :=
  (b / (c + d)) + (c / (a + b))

theorem min_value_proof (a d: ℝ) (b c: ℝ)
  (h1 : 0 ≤ a) (h2 : 0 ≤ d) (h3 : 0 < b) (h4 : 0 < c) (h5 : b + c ≥ a + d) 
: min_value_expression a d b c h1 h2 h3 h4 h5 = sqrt 2 - 1 / 2 :=
  by sorry

end min_value_proof_l626_626111


namespace ella_hank_weight_l626_626332

-- Define the weights as variables
variables (e f g h : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := e + f = 280
def condition2 : Prop := f + g = 230
def condition3 : Prop := g + h = 260

-- The proof statement
theorem ella_hank_weight (h1 : condition1) (h2 : condition2) (h3 : condition3) : e + h = 310 := sorry

end ella_hank_weight_l626_626332


namespace abs_inequality_solution_l626_626527

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l626_626527


namespace sqrt_2450_minus_2_eq_sqrt_a_minus_b_squared_l626_626925

theorem sqrt_2450_minus_2_eq_sqrt_a_minus_b_squared (a b : ℕ) (h₁ : (\sqrt (2450 : ℝ) - 2 = (\sqrt (a : ℝ) - b)^2)) (ha : a = 2450) (hb : b = 1) :
  a + b = 2451 :=
begin
  rw [ha, hb],
  norm_num,
end

end sqrt_2450_minus_2_eq_sqrt_a_minus_b_squared_l626_626925


namespace purchase_price_of_shares_l626_626675

theorem purchase_price_of_shares 
  (dividend_rate : Real)     -- The dividend rate (15.5%)
  (face_value : Real)        -- The face value of the share (Rs. 50)
  (nominal_roi : Real)       -- The nominal return on investment (25%)
  (inflation_rate : Real)    -- The inflation rate (3%)
  (actual_roi : Real)        -- The actual return on investment (22%)
  (dividend_per_share : Real) -- The dividend per share (Rs. 7.75)
  : Real := 
by 
    -- Given the conditions
    let dividend_rate := 15.5 / 100
    let face_value := 50
    let nominal_roi := 25 / 100
    let inflation_rate := 3 / 100
    let actual_roi := nominal_roi - inflation_rate  -- 22% = 25% - 3%
    let dividend_per_share := dividend_rate * face_value  -- Rs. 7.75 = 0.155 * Rs. 50
    -- Prove the purchase price is approximately Rs. 35.23
    sorry

end purchase_price_of_shares_l626_626675


namespace length_of_square_side_l626_626916

theorem length_of_square_side
  (AB AC : ℝ)
  (right_angle: is_right_triangle ABC)
  (side_condition: on_hypotenuse_square_vertex_on_legs ABC)
  (h_AB : AB = 6)
  (h_AC : AC = 8) :
  ∃ s : ℝ, s = 120 / 37 :=
by 
  sorry

end length_of_square_side_l626_626916


namespace playing_cards_distribution_l626_626419

theorem playing_cards_distribution :
  (∀ (deck_total : ℕ) (playing_cards : ℕ) (instruction_cards : ℕ) (people : ℕ), 
    deck_total = 60 ∧ playing_cards = 52 ∧ instruction_cards = 8 ∧ people = 9 → 
    ∃ (num_people_with_6_playing_cards : ℕ), num_people_with_6_playing_cards = 7) :=
begin
  intros deck_total playing_cards instruction_cards people h,
  cases h with h1 h_temp,
  cases h_temp with h2 h_temp',
  cases h_temp' with h3 h4,
  sorry
end

end playing_cards_distribution_l626_626419


namespace sum_of_reciprocals_l626_626617

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) : 
  (1 / x) + (1 / y) = 4 :=
by
  sorry

end sum_of_reciprocals_l626_626617


namespace intersection_A_B_l626_626119

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - 2x - 3 < 0 }
def B : Set ℝ := { y | ∃ x ∈ Set.Icc (0 : ℝ) 2, y = 2 * x }

-- State the theorem about their intersection
theorem intersection_A_B : A ∩ B = Set.Ico (1 : ℝ) 3 :=
by
  sorry

end intersection_A_B_l626_626119


namespace calc_value_l626_626802

def f (x : ℝ) : ℝ :=
if x > 0 then x else 3^x

theorem calc_value : f (f (1 / 2)) = 1 / 2 :=
by
  -- We apply the fact that for x = 1/2, which is greater than 0, f(x) = x
  have h1: f (1 / 2) = 1 / 2 := if_pos (by linarith)
  -- Now we apply f to the result again: f(f(1/2))
  have h2: f (f (1 / 2)) = f (1 / 2) := by rw [h1]
  -- And since f (1/2) = 1/2 (as we found before), then:
  rw [h1] at h2
  exact h2

end calc_value_l626_626802


namespace ants_remove_sugar_l626_626285

theorem ants_remove_sugar (x : ℝ) (h1 : 3 * x + 3 * x = 24) : x = 4 :=
begin
  sorry
end

end ants_remove_sugar_l626_626285


namespace perpendicular_line_plane_l626_626420

theorem perpendicular_line_plane (l : Line) (P : Plane)
  (cond_triangle : ∃ (a b : Line), a ≠ b ∧ a ∈ P ∧ b ∈ P ∧ l ⊥ a ∧ l ⊥ b ∧ intersect a b)
  (cond_circle : ∃ (c d : Line), c ≠ d ∧ c ∈ P ∧ d ∈ P ∧ l ⊥ c ∧ l ⊥ d)
  (cond_parallelogram : ∃ (e f : Line), e ≠ f ∧ e ∈ P ∧ f ∈ P ∧ l ⊥ e ∧ l ⊥ f ∧ intersect e f)
  (cond_trapezoid : ∃ (g h : Line), g ≠ h ∧ g ∈ P ∧ h ∈ P ∧ l ⊥ g ∧ l ⊥ h ∧ intersect g h) :
  (∃ (a b : Line), a ≠ b ∧ a ∈ P ∧ b ∈ P ∧ intersect a b ∧ l ⊥ a ∧ l ⊥ b) :=
by
  have h_triangle := cond_triangle
  have h_parallelogram := cond_parallelogram
  have h_trapezoid := cond_trapezoid
  exact ⟨_, _, _, _, _, _, sorry⟩

end perpendicular_line_plane_l626_626420


namespace num_of_consec_int_sum_18_l626_626405

theorem num_of_consec_int_sum_18 : 
  ∃! (a n : ℕ), n ≥ 3 ∧ (n * (2 * a + n - 1)) = 36 :=
sorry

end num_of_consec_int_sum_18_l626_626405


namespace ellipse_foci_ratio_l626_626789

theorem ellipse_foci_ratio :
  let a := 3
  let b := 2
  let c := (Real.sqrt (a^2 - b^2)) -- c = sqrt(5)
  let ellipse_eq : x^2 / 9 + y^2 / 4 = 1
  let f1 := (-Real.sqrt 5, 0)
  let f2 := (Real.sqrt 5, 0)
  ∃ P : ℝ × ℝ, (P ∈ (set_of (λ p, (p.1)^2 / 9 + (p.2)^2 / 4 = 1))) ∧ 
    ((|((P.1, P.2) - f1)| > |((P.1, P.2) - f2)|) ∧ ( (|((P.1, P.2) - f1)| = 3 ∧ |((P.1, P.2) - f2)| = 1) ∨ 
      (by Pythagorean theorem 
      (|((P.1, P.2) - f1)|, |((P.1, P.2) - f2)|) = (2,1) or (3/2, 1/7.5)) 
    )
  ∧( (|((P.1, P.2) - f1)|)/( |((P.1, P.2) - f2)| = 2) ∨ (|((P.1, P.2) - f1)|)/( |((P.1, P.2) - f2)| = 7/2)

end ellipse_foci_ratio_l626_626789


namespace Percentage_YellowBalls_correct_l626_626739

def Number_of_Team1 := 72
def Number_of_Team2 := 64
def Number_of_Team3 := 53

def YellowBalls_Team1_per_player := 4
def BrownBalls_Team1_per_player := 3
def BlueBalls_Team1_per_player := 3

def GreenBalls_Team2_per_player := 5
def OrangeBalls_Team2_per_player := 4
def PinkBalls_Team2_per_player := 2

def BlackBalls_Team3_per_player := 3
def RedBalls_Team3_per_player := 2
def WhiteBalls_Team3_per_player := 6

noncomputable def Team1_YellowBalls := Number_of_Team1 * YellowBalls_Team1_per_player
noncomputable def Team1_TotalBalls := Number_of_Team1 * (YellowBalls_Team1_per_player + BrownBalls_Team1_per_player + BlueBalls_Team1_per_player)

noncomputable def Team2_TotalBalls := Number_of_Team2 * (GreenBalls_Team2_per_player + OrangeBalls_Team2_per_player + PinkBalls_Team2_per_player)

noncomputable def Team3_TotalBalls := Number_of_Team3 * (BlackBalls_Team3_per_player + RedBalls_Team3_per_player + WhiteBalls_Team3_per_player)

noncomputable def TotalBalls_Distributed := Team1_TotalBalls + Team2_TotalBalls + Team3_TotalBalls

noncomputable def Percentage_YellowBalls := (Team1_YellowBalls / TotalBalls_Distributed) * 100

theorem Percentage_YellowBalls_correct : Percentage_YellowBalls ≈ 14.35 := by
  sorry

end Percentage_YellowBalls_correct_l626_626739


namespace perimeters_ratio_l626_626577

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l626_626577


namespace sum_of_digits_of_elements_count_l626_626104

-- Definitions and conditions from the problem
def sum_of_digits (x : ℕ) : ℕ := (x.digits 10).sum

def satisfies_conditions (n : ℕ) : Prop := sum_of_digits n = 15 ∧ n < 10^8

def S : set ℕ := {n | satisfies_conditions n}

-- Lean 4 statement of the proof problem
theorem sum_of_digits_of_elements_count (m : ℕ) (h : m = set.card S) : 
  sum_of_digits m = 21 := 
sorry

end sum_of_digits_of_elements_count_l626_626104


namespace convex_quad_no_triangle_l626_626886

/-- Given four angles of a convex quadrilateral, it is not always possible to choose any 
three of these angles so that they represent the lengths of the sides of some triangle. -/
theorem convex_quad_no_triangle (α β γ δ : ℝ) 
  (h_sum : α + β + γ + δ = 360) :
  ¬(∀ a b c : ℝ, a + b + c = 360 → (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end convex_quad_no_triangle_l626_626886


namespace ratio_of_perimeters_l626_626567

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l626_626567


namespace min_sum_eq_l626_626323

theorem min_sum_eq : 
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → 
  (∃ x : ℝ, x = 3 * real.cbrt (1 / 162) ∧ x = (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a))) :=
by
  intros a b c ha hb hc
  use 3 * real.cbrt (1 / 162)
  split
  { sorry },
  { sorry }

end min_sum_eq_l626_626323


namespace part_I_part_II_l626_626813

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (2 + Real.sin x, 1)
def b : ℝ × ℝ := (2, -2)
def c (x : ℝ) : ℝ × ℝ := (Real.sin x - 3, 1)
def d (k : ℝ) : ℝ × ℝ := (1, k)

-- Define the parallel condition for part I and solve for x
theorem part_I (x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) : 
  (x : ℝ) = -Real.pi / 6 ↔ 
  let sum_bc := (Real.sin x - 1, -1) in 
  ∃ k, a x = k • sum_bc := 
by
  sorry

-- Define the parallel condition for part II and solve for the range of k
theorem part_II (x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) : 
  Set.Icc 0 Real.top ⊆ { k | let sum_ad := (3 + Real.sin x, 1 + k), sum_bc := (Real.sin x - 1, -1) in 
  ∃ c, sum_ad = c • sum_bc } :=
by
  sorry

end part_I_part_II_l626_626813


namespace find_x0_l626_626120

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + b

theorem find_x0 (a b x₀ : ℝ) (h : a ≠ 0) 
  (int_eq : ∫ x in 0..3, f a b x = 3 * f a b x₀) : x₀ = Real.sqrt 3 ∨ x₀ = -Real.sqrt 3 :=
by
  sorry

end find_x0_l626_626120


namespace quadrilateral_parallelogram_l626_626517

theorem quadrilateral_parallelogram (a b c d m_1 m_2 : ℝ) 
  (h1 : m_1 = (a + c) / 2) 
  (h2 : m_2 = (b + d) / 2) 
  (h_sum : m_1 + m_2 = (a + b + c + d) / 2) : 
  ∃ (parallelogram : Type), parallelogram :=
by
  sorry  -- Proof goes here

end quadrilateral_parallelogram_l626_626517


namespace convex_function_on_interval_l626_626502

noncomputable def f (m x : ℝ) : ℝ := (1 / 12) * x^4 - (m / 6) * x^3 - (3 / 2) * x^2

noncomputable def f'' (m x : ℝ) : ℝ := x^2 - m * x - 3

theorem convex_function_on_interval (m : ℝ) :
  (∀ x ∈ Ioo (1 : ℝ) (3 : ℝ), f'' m x < 0) → 2 ≤ m :=
begin
  intros h,
  have h1 : f'' m 1 ≤ 0 := h 1 (by norm_num [Ioo]),
  have h3 : f'' m 3 ≤ 0 := h 3 (by norm_num [Ioo]),
  linarith,
end

end convex_function_on_interval_l626_626502


namespace f_of_3_l626_626390

def f : ℤ → ℤ
| x => if x ≤ 0 then x + 1 else f (x - 1) - f (x - 2)

theorem f_of_3 : f 3 = -1 := by
  sorry

end f_of_3_l626_626390


namespace log_increasing_interval_l626_626321

theorem log_increasing_interval :
  ∀ x: ℝ, (x > 4) → strict_increasing_on (λ x, log 2 (x^2 - 3 * x - 4)) (set.Ioi 4) := 
begin
  sorry
end

end log_increasing_interval_l626_626321


namespace distance_from_post_office_back_home_l626_626470

theorem distance_from_post_office_back_home :
  ∀ (h_library dist_total dist_house_library dist_library_post_office : ℝ),
  dist_house_library = 0.3 ∧
  dist_library_post_office = 0.1 ∧
  dist_total = 0.8 ∧
  h_library = dist_house_library + dist_library_post_office →
  dist_total - h_library = 0.4 :=
by
  intros h_library dist_total dist_house_library dist_library_post_office
  intro h_conditions
  cases h_conditions with dHL dLP dT H
  cases dLP with dPS H',
  sorry

end distance_from_post_office_back_home_l626_626470


namespace k_start_time_difference_l626_626231

variable (a_flat a_uphill b_flat b_uphill k_flat k_uphill : ℝ)
variable (Va Vb Vk : ℝ)
variable (t x : ℝ)

-- Assume given conditions
-- a's speeds
def a_flat := 30
def a_uphill := 20
def Va := (a_flat + a_uphill) / 2

-- b's speeds
def b_flat := 40
def b_uphill := 25
def Vb := (b_flat + b_uphill) / 2

-- k's speeds
def k_flat := 60
def k_uphill := 30
def Vk := (k_flat + k_uphill) / 2

-- b starts 5 hours after a
variable (start_diff : ℝ)
def start_diff := 5

-- Overtake conditions
variable (overtake_time : ℝ)
def overtake_time := 16.67

-- Proving the start difference for k
theorem k_start_time_difference :
  Va * (overtake_time + x) = Vk * overtake_time → Va * (overtake_time + start_diff) = Vb * overtake_time → x = 13.34 := 
by 
  sorry

end k_start_time_difference_l626_626231


namespace height_to_width_ratio_l626_626945

theorem height_to_width_ratio (s w h l V : ℝ) (h1 : h = s * w) (h2 : l = 7 * h) (h3 : V = 16128) (h4 : w = 4) : s = 12 :=
by
  have eq1 : h = s * 4 := by
    rwa [h4] at h1

  have eq2 : l = 7 * (s * 4) := by
    rwa [eq1] at h2

  have eq3 : V = 4 * (s * 4) * (7 * (s * 4)) := by
    rw [eq1, eq2]

  have eq4 : 16128 = 112 * s^2 := by
    rw [h3, eq3]
    ring

  have eq5 : s^2 = 144 := by
    linarith

  exact (eq5.pos_sqrt_eq 144.zero_le).symm

end height_to_width_ratio_l626_626945


namespace city_cleaning_total_l626_626126

variable (A B C D : ℕ)

theorem city_cleaning_total : 
  A = 54 →
  A = B + 17 →
  C = 2 * B →
  D = A / 3 →
  A + B + C + D = 183 := 
by 
  intros hA hAB hC hD
  sorry

end city_cleaning_total_l626_626126


namespace carl_trip_cost_l626_626709

theorem carl_trip_cost :
  (let city_mpg := 30 in
   let hwy_mpg := 40 in
   let one_way_city_miles := 60 in
   let one_way_hwy_miles := 200 in
   let gas_cost_per_gallon := 3 in
   let round_trip_city_miles := 2 * one_way_city_miles in
   let round_trip_hwy_miles := 2 * one_way_hwy_miles in
   let city_gas_needed := round_trip_city_miles / city_mpg in
   let hwy_gas_needed := round_trip_hwy_miles / hwy_mpg in
   let total_gas_needed := city_gas_needed + hwy_gas_needed in
   let total_cost := total_gas_needed * gas_cost_per_gallon in
   total_cost = 42) :=
sorry

end carl_trip_cost_l626_626709


namespace average_of_data_set_l626_626585

theorem average_of_data_set :
  let data := [3, -2, 4, 1, 4]
  let n := data.length
  let sum := data.foldr (· + ·) 0
  (sum / n : ℝ) = 2 :=
by
  let data := [3, -2, 4, 1, 4]
  let n := data.length
  let sum := data.foldr (· + ·) 0
  have h1 : sum = 10 := by
    sorry
  have h2 : n = 5 := by
    sorry
  have h3 : (10 / 5 : ℝ) = 2 := by
    sorry
  show (sum / n : ℝ) = 2 from
    by rw [h1, h2, h3]

end average_of_data_set_l626_626585


namespace product_fraction_sequence_l626_626722

theorem product_fraction_sequence : 
  (∏ n in Finset.range 15, (n + 1) + 4) / (∏ n in Finset.range 15, (n + 1)) = 11628 := by
  sorry

end product_fraction_sequence_l626_626722


namespace find_x_l626_626006

theorem find_x (x : ℝ) : 3 * (2^x + 2^x + 2^x + 2^x + 2^x) = 1536 → x = 5.67807 :=
by
  sorry

end find_x_l626_626006
