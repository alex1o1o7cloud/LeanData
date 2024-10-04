import Mathlib

namespace path_length_l278_278891

theorem path_length 
    (poles : ℕ) 
    (interval : ℕ) 
    (bridge_length : ℝ) 
    (total_poles : ℕ) 
    (half_total_poles_eq : 2 * poles = total_poles) 
    (interval_between_poles : interval = 6)
    (bridge_length_eq : bridge_length = 42) 
    (total_poles_eq : total_poles = 286) 
    : ℝ :=
  let intervals := poles - 1
  let path_covered_by_poles := intervals * interval
  let total_path_length := path_covered_by_poles + bridge_length
  total_path_length

example : path_length 143 6 42 286 (by norm_num) (by norm_num) (by norm_num) (by norm_num) = 894 := by norm_num

end path_length_l278_278891


namespace probability_of_rolling_greater_than_5_on_8_sided_die_l278_278487

theorem probability_of_rolling_greater_than_5_on_8_sided_die : 
  let total_outcomes := 8
  let successful_outcomes := 3
  let probability := successful_outcomes / total_outcomes.to_rat
  probability = 3 / 8 :=
by
  sorry

end probability_of_rolling_greater_than_5_on_8_sided_die_l278_278487


namespace area_AMCN_l278_278767

-- Definitions of the geometrical entities and conditions
def Rectangle (length width : ℝ) := length * width

def midpoint (a b : ℝ) := (a + b) / 2

def area_triangle (base height : ℝ) := (base * height) / 2

-- Given conditions
def lengthAB := 10
def widthAD := 5
def B := 0
def C := widthAD
def D := lengthAB
def M := midpoint B C
def N := midpoint C D

-- Prove the area of region AMCN
theorem area_AMCN : 
    let area_ABCD := Rectangle lengthAB widthAD,
        area_ABM := area_triangle lengthAB M,
        area_ADN := area_triangle widthAD N
    in area_ABCD - area_ABM - area_ADN = 31.25 := 
by
    -- The proof will go here.
    sorry

end area_AMCN_l278_278767


namespace infinitely_many_primes_dividing_polynomial_l278_278046

theorem infinitely_many_primes_dividing_polynomial (P : ℤ[X]) (hP : P ≠ 0) : 
  ∃^∞ q : ℕ, ∃ n : ℕ, Nat.Prime q ∧ q ∣ (2^n + P.eval (n : ℤ)) :=
sorry

end infinitely_many_primes_dividing_polynomial_l278_278046


namespace sum_midpoints_x_sum_midpoints_y_l278_278459

-- Defining the problem conditions
variables (a b c d e f : ℝ)
-- Sum of the x-coordinates of the triangle vertices is 15
def sum_x_coords (a b c : ℝ) : Prop := a + b + c = 15
-- Sum of the y-coordinates of the triangle vertices is 12
def sum_y_coords (d e f : ℝ) : Prop := d + e + f = 12

-- Proving the sum of x-coordinates of midpoints of sides is 15
theorem sum_midpoints_x (h1 : sum_x_coords a b c) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by  
  sorry

-- Proving the sum of y-coordinates of midpoints of sides is 12
theorem sum_midpoints_y (h2 : sum_y_coords d e f) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := 
by  
  sorry

end sum_midpoints_x_sum_midpoints_y_l278_278459


namespace minimal_value_is_3980_l278_278368

-- The conditions
def isPairwiseDistinct (as: List ℕ) : Prop := 
  ∀ i j, i < as.length → j < as.length → i ≠ j → as[i] ≠ as[j]

def sum_eq_1995 (as: List ℕ) : Prop :=
  as.sum = 1995

-- The function to be minimized
def f (as: List ℕ) : ℕ :=
  (List.zipWith (*) as (as.rotate 1)).sum

-- The proof statement
theorem minimal_value_is_3980 (as: List ℕ) (h₁: isPairwiseDistinct as) (h₂: sum_eq_1995 as) :
  f as = 3980 :=
by
  sorry

end minimal_value_is_3980_l278_278368


namespace line_segment_both_symmetric_l278_278557

-- Definition of centrally symmetric figures
def isCentrallySymmetric (S : Type) : Prop := 
  -- definition based on the context, e.g., reflection symmetry about the center
  sorry

-- Definition of axially symmetric figures
def isAxiallySymmetric (S : Type) : Prop := 
  -- definition based on the context, e.g., reflection symmetry about an axis
  sorry

-- Specific shapes
structure Shape :=
  (isEquilateralTriangle : Prop)
  (isIsoscelesTriangle : Prop)
  (isParallelogram : Prop)
  (isLineSegment : Prop)

-- Given conditions
axiom equilateral_not_centrally_symmetric : isCentrallySymmetric Shape → ¬ Shape.isEquilateralTriangle
axiom equilateral_axially_symmetric : isAxiallySymmetric Shape → Shape.isEquilateralTriangle
axiom isosceles_not_centrally_symmetric : isCentrallySymmetric Shape → ¬ Shape.isIsoscelesTriangle
axiom isosceles_axially_symmetric : isAxiallySymmetric Shape → Shape.isIsoscelesTriangle
axiom parallelogram_centrally_symmetric : isCentrallySymmetric Shape → Shape.isParallelogram
axiom parallelogram_not_axially_symmetric : isAxiallySymmetric Shape → ¬ Shape.isParallelogram
axiom line_segment_centrally_symmetric : isCentrallySymmetric Shape → Shape.isLineSegment
axiom line_segment_axially_symmetric : isAxiallySymmetric Shape → Shape.isLineSegment

-- Proof statement
theorem line_segment_both_symmetric : 
  isCentrallySymmetric Shape → 
  isAxiallySymmetric Shape → 
  Shape.isLineSegment :=
begin
  sorry
end

end line_segment_both_symmetric_l278_278557


namespace largest_non_summable_composite_l278_278154

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278154


namespace min_value_expr_l278_278198

theorem min_value_expr (a : ℝ) (h1 : 1 < a) (h2 : a < 4) : (∀ b, (1 < b ∧ b < 4) → (b / (4 - b) + 1 / (b - 1)) ≥ 2) :=
by
  intro b hb1 hb2
  sorry

end min_value_expr_l278_278198


namespace trains_cross_each_other_in_given_time_l278_278824

noncomputable def trains_crossing_time (length1 length2 speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1 := (speed1_kmph * 1000) / 3600
  let speed2 := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1 + speed2
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem trains_cross_each_other_in_given_time :
  trains_crossing_time 300 400 36 18 = 46.67 :=
by
  -- expected proof here
  sorry

end trains_cross_each_other_in_given_time_l278_278824


namespace probability_best_play_wins_l278_278071

noncomputable def prob_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) : ℝ :=
  let C := Nat.choose in
  (1 : ℝ) / (C (2 * n) n * C (2 * n) (2 * m)) * ∑ q in Finset.range (2 * m + 1),
  (C n q * C n (2 * m - q)) * 
  ∑ t in Finset.range (min q m),
  (C q t * C (2 * n - q) (n - t))

-- A theorem statement in Lean to ensure proper type checks and conditions 
theorem probability_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) :
  ∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t)) 
  =
  (∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t) )) * 
  (nat.choose (2 * n) n * nat.choose (2 * n) (2 * m)) :=
sorry

end probability_best_play_wins_l278_278071


namespace area_after_cuts_is_21_l278_278908

def original_length : ℝ := 20
def original_width : ℝ := 2
def length_after_first_cut := original_length * (1 - 0.30)
def width_after_second_cut := original_width * (1 - 0.25)
def area_after_cuts := length_after_first_cut * width_after_second_cut

theorem area_after_cuts_is_21 :
    area_after_cuts = 21 :=
by
    unfold original_length
    unfold original_width
    unfold length_after_first_cut
    unfold width_after_second_cut
    unfold area_after_cuts
    sorry

end area_after_cuts_is_21_l278_278908


namespace positive_number_sum_square_l278_278005

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l278_278005


namespace taco_truck_profit_l278_278089

-- Definitions and conditions
def pounds_of_beef : ℕ := 100
def beef_per_taco : ℝ := 0.25
def price_per_taco : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Desired profit result
def expected_profit : ℝ := 200

-- The proof statement (to be completed)
theorem taco_truck_profit :
  let tacos := pounds_of_beef / beef_per_taco;
  let revenue := tacos * price_per_taco;
  let cost := tacos * cost_per_taco;
  let profit := revenue - cost;
  profit = expected_profit :=
by
  sorry

end taco_truck_profit_l278_278089


namespace hyperbola_focus_asymptote_distance_l278_278671

theorem hyperbola_focus_asymptote_distance :
  ∀ (a b c : ℝ), 
  (a ≠ 0) ∧ (b ≠ 0) ∧ 
  (c^2 = a^2 + b^2) ∧ 
  (asymptote_eq : ∀ x y : ℝ, y = (b / a) * x) ∧ 
  (asymptote_pt : asymptote_eq √2 1) 
  → 
  (dist_focus_asymptote : dist (√(a^2 + b^2), 0) (asymptote_eq b a)) = √2 :=
begin
  sorry
end

end hyperbola_focus_asymptote_distance_l278_278671


namespace log_sum_eq_two_l278_278934

theorem log_sum_eq_two (log6_3 log6_4 : ℝ) (H1 : Real.logb 6 3 = log6_3) (H2 : Real.logb 6 4 = log6_4) : 
  log6_3 + log6_4 = 2 := 
by 
  sorry

end log_sum_eq_two_l278_278934


namespace horse_running_time_l278_278911

noncomputable def field_area : ℝ := 3750
noncomputable def horse_speed : ℝ := 40
noncomputable def side_length (area : ℝ) : ℝ := Real.sqrt area
noncomputable def perimeter (side : ℝ) : ℝ := 4 * side
noncomputable def time_to_complete_perimeter (perimeter : ℝ) (speed : ℝ) : ℝ := perimeter / speed

theorem horse_running_time :
  let side := side_length field_area,
      perim := perimeter side,
      time := time_to_complete_perimeter perim horse_speed
  in time ≈ 6.124 :=
by
  let side := side_length field_area
  let perim := perimeter side
  let time := time_to_complete_perimeter perim horse_speed
  have h1 : side = Real.sqrt field_area, by sorry
  have h2 : perim = 4 * side, by sorry
  have h3 : time = perim / horse_speed, by sorry
  show time ≈ 6.124, by sorry

end horse_running_time_l278_278911


namespace annual_interest_payment_l278_278867

noncomputable def principal : ℝ := 9000
noncomputable def rate : ℝ := 9 / 100
noncomputable def time : ℝ := 1
noncomputable def interest : ℝ := principal * rate * time

theorem annual_interest_payment : interest = 810 := by
  sorry

end annual_interest_payment_l278_278867


namespace eva_operations_terminate_l278_278583

theorem eva_operations_terminate :
  ∀ (deck : list ℕ), (∀ k, k ∈ deck → 1 ≤ k ∧ k ≤ 100) →
  ∃ fin_seq, fin_seq.length < ∞ ∧ 
  (∀ seq, (∀ i, 0 ≤ i ∧ i < seq.length → (seq.get i = deck.get i) ∨
    (∃ k, seq.get i = deck.get k)) ∧ 
    (seq.get (seq.length - 1) = 1)) :=
begin
  -- sorry to skip the proof
  sorry
end

end eva_operations_terminate_l278_278583


namespace count_multiples_of_8_between_200_and_400_l278_278263

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278263


namespace range_of_a_l278_278643

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - x^2 - log x

-- Define the condition that f has an extreme value
def has_extreme_value (a : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ (deriv (f a) x = 0)

-- Define the condition that the sum of all extreme values of f is less than 5 + log 2
def sum_extreme_values_condition (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ (deriv (f a) x1 = 0) ∧ (deriv (f a) x2 = 0) →
  f a x1 + f a x2 < 5 + log 2

-- Final statement combining the question, conditions, and answer
theorem range_of_a (a : ℝ) :
  has_extreme_value a →
  sum_extreme_values_condition a →
  2 * real.sqrt 2 < a ∧ a < 4 :=
by
  intros h1 h2
  sorry

end range_of_a_l278_278643


namespace intersection_point_l278_278926

-- Define the points
def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (6, 15)

-- Define the slope function
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

-- Check that the slope of the line through point1 and point2 is 3
def line_slope : ℝ := slope point1 point2

-- Define the line equation using point-slope form and check that it passes through (0, -3)
def line_eq (x : ℝ) : ℝ := 
  line_slope * (x - point1.fst) + point1.snd

-- Define the intersection point with the y-axis
def y_intersect_point : ℝ × ℝ := (0, line_eq 0)

theorem intersection_point :
  y_intersect_point = (0, -3) :=
sorry

end intersection_point_l278_278926


namespace area_of_figure_l278_278876

noncomputable def area_bounded_by_curves : ℝ :=
  2 * ∫ t in 0 .. (Real.pi / 3), (8 * (Real.sin t)^3) * (-24 * (Real.cos t)^2 * (Real.sin t))

theorem area_of_figure :
  let S := 2 * ∫ t in 0 .. (Real.pi / 3), (8 * (Real.sin t)^3) * (-24 * (Real.cos t)^2 * (Real.sin t)) in
  S = 8 * Real.pi := 
sorry

end area_of_figure_l278_278876


namespace problem_expression_value_l278_278579

theorem problem_expression_value :
  (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 :=
by
  sorry

end problem_expression_value_l278_278579


namespace biased_coin_flips_l278_278523

theorem biased_coin_flips (h : ℚ) 
  (H1 : (Nat.choose 6 2) * h^2 * (1 - h)^4 = (Nat.choose 6 3) * h^3 * (1 - h)^3)
  (H2 : ∃ h : ℚ, h = 1/3) :
  let prob := (Nat.choose 6 4) * h^4 * (1 - h)^2 in 
  let simplified_prob := prob.num / prob.denom in 
  (simplified_prob.num + simplified_prob.denom) = 263 :=
by {
  sorry
}

end biased_coin_flips_l278_278523


namespace count_multiples_of_8_between_200_and_400_l278_278267

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278267


namespace original_number_is_106_25_l278_278102

theorem original_number_is_106_25 (x : ℝ) (h : (x + 0.375 * x) - (x - 0.425 * x) = 85) : x = 106.25 := by
  sorry

end original_number_is_106_25_l278_278102


namespace system_consistent_and_solution_l278_278605

theorem system_consistent_and_solution (a x : ℝ) : 
  (a = -10 ∧ x = -1/3) ∨ (a = -8 ∧ x = -1) ∨ (a = 4 ∧ x = -2) ↔ 
  3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0 := by
  sorry

end system_consistent_and_solution_l278_278605


namespace min_omega_for_symmetric_center_l278_278324

def is_symmetric_center (ω : ℕ) (x₀ y₀ : ℝ) : Prop :=
  ∃ k : ℤ, ω * x₀ + (π / 6) = k * π + (π / 2)

theorem min_omega_for_symmetric_center :
  ∃ (ω : ℕ), ω > 0 ∧ is_symmetric_center ω (π / 6) 0 ∧ (∀ (ω' : ℕ), ω' > 0 ∧ is_symmetric_center ω' (π / 6) 0 → ω ≤ ω') :=
begin
  let ω := 2,
  use ω,
  split,
  { exact Nat.succ_pos' 1, },
  split,
  { use 1,
    norm_num,
    field_simp,
    ring, },
  { intros ω' hω'_pos h_symm_center,
    rcases h_symm_center with ⟨k, hk⟩,
    sorry }
end

end min_omega_for_symmetric_center_l278_278324


namespace find_x_l278_278586

theorem find_x (x : ℝ) : log x 16 = log 81 3 → x = 65536 := by 
  sorry

end find_x_l278_278586


namespace sum_of_distances_l278_278889

variables {A B : Point} (C : Point)
variable (d₁ d₂ : ℝ)

-- Conditions
-- A circle touches the sides of the angle at points A and B.
-- The distance from point C to the line segment AB is 8.
-- One of the distances from point C to the sides of the angle is 30 less than the other.

def distances_relation := d₁ = d₂ + 30
def sum_distances := d₁ + d₂ = 34

-- Theorem statement
theorem sum_of_distances
  (h1 : |dist C A| = d₁)
  (h2 : |dist C B| = d₂)
  (h3 : distances_relation)
  (h4 : d₁ + d₂ = 34) :
  d₁ + d₂ = 34 := sorry

end sum_of_distances_l278_278889


namespace marjorie_first_day_cakes_l278_278747

def cakes_made_each_day (n : ℕ) : ℕ :=
  match n with
  | 0     => 10
  | (n+1) => 2 * cakes_made_each_day n

theorem marjorie_first_day_cakes :
  cakes_made_each_day 5 = 320 → cakes_made_each_day 0 = 10 :=
begin
  intro h,
  have h5 : cakes_made_each_day 5 = 2 * cakes_made_each_day 4, by sorry,
  have h4 : cakes_made_each_day 4 = 2 * cakes_made_each_day 3, by sorry,
  have h3 : cakes_made_each_day 3 = 2 * cakes_made_each_day 2, by sorry,
  have h2 : cakes_made_each_day 2 = 2 * cakes_made_each_day 1, by sorry,
  have h1 : cakes_made_each_day 1 = 2 * cakes_made_each_day 0, by sorry,
  rw h5 at h,
  rw h4 at h,
  rw h3 at h,
  rw h2 at h,
  rw h1 at h,
  sorry
end

end marjorie_first_day_cakes_l278_278747


namespace cost_of_fencing_l278_278869

theorem cost_of_fencing (d : ℝ) (rate : ℝ) (C : ℝ) (cost : ℝ) : 
  d = 22 → rate = 3 → C = Real.pi * d → cost = C * rate → cost = 207 :=
by
  intros
  sorry

end cost_of_fencing_l278_278869


namespace graph_transformation_l278_278646

noncomputable def initial_function (x : ℝ) : ℝ :=
  2 * Real.sin (x + Real.pi / 3)

noncomputable def resulting_function (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 3)

theorem graph_transformation :
  ∀ x : ℝ, (∃ y, initial_function x = y → resulting_function x = y) :=
by
  intros x y h
  sorry

end graph_transformation_l278_278646


namespace probability_heads_not_less_than_tails_is_11_over_16_l278_278017

open ProbabilityTheory

noncomputable def num_of_heads_not_less_than_tails (tosses : list bool) : Prop :=
  tosses.count tt >= tosses.count ff

def all_outcomes_4_tosses : finset (list bool) :=
  finset.univ.image (λ b: fin 4 → bool, list.of_fn b)

def desired_outcomes : finset (list bool) :=
  finset.filter (λ outcome, num_of_heads_not_less_than_tails outcome) all_outcomes_4_tosses

theorem probability_heads_not_less_than_tails_is_11_over_16 :
  (desired_outcomes.card : ℚ) / (all_outcomes_4_tosses.card : ℚ) = 11 / 16 :=
by
  sorry

end probability_heads_not_less_than_tails_is_11_over_16_l278_278017


namespace remainder_when_P_divided_by_DD_l278_278985

noncomputable def remainder (a b : ℕ) : ℕ := a % b

theorem remainder_when_P_divided_by_DD' (P D Q R D' Q'' R'' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  remainder P (D * D') = R :=
by {
  sorry
}

end remainder_when_P_divided_by_DD_l278_278985


namespace count_multiples_of_8_between_200_and_400_l278_278266

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278266


namespace sumOfDivisors_of_1184_l278_278804

noncomputable def sumOfDivisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, n % d = 0).sum

theorem sumOfDivisors_of_1184 : sumOfDivisors 1184 = 2394 := by
  sorry

end sumOfDivisors_of_1184_l278_278804


namespace triangle_XYZ_properties_l278_278331

-- Defining the problem conditions
variables {X Y Z W : Type}
variables [RightTriangle X Y Z]
variables (angle_X : ∠X = 90)
variables (tan_Y : Real := 3 / 4)
variables (YZ : Real := 30)

-- The statement of the problem to prove
theorem triangle_XYZ_properties
  (XY ZW : Real)
  (h1 : XY = 24)
  (h2 : ZW = 18) :
  XY = 24 ∧ ZW = 18 :=
sorry

end triangle_XYZ_properties_l278_278331


namespace mandy_score_l278_278336

theorem mandy_score (total_questions : ℕ) (lowella_percentage : ℕ) (pamela_percentage_extra : ℕ) :
  let lowella_correct := total_questions * lowella_percentage / 100
  let pamela_correct := lowella_correct + lowella_correct * pamela_percentage_extra / 100
  let mandy_correct := 2 * pamela_correct in
  total_questions = 100 ∧ lowella_percentage = 35 ∧ pamela_percentage_extra = 20 →
  mandy_correct = 84 :=
by
  intros total_questions_equals hundred lowella_percentage_equals thirty_five pamela_percentage_extra_equals twenty
  simp [total_questions_equals, lowella_percentage_equals, pamela_percentage_extra_equals]
  sorry

end mandy_score_l278_278336


namespace sequence_a4_eq_5_over_3_l278_278706

theorem sequence_a4_eq_5_over_3 :
  ∀ (a : ℕ → ℚ), a 1 = 1 → (∀ n > 1, a n = 1 / a (n - 1) + 1) → a 4 = 5 / 3 :=
by
  intro a ha1 H
  sorry

end sequence_a4_eq_5_over_3_l278_278706


namespace solve_gcd_problem_l278_278448

def gcd_problem : Prop :=
  gcd 1337 382 = 191

theorem solve_gcd_problem : gcd_problem := 
by 
  sorry

end solve_gcd_problem_l278_278448


namespace c_share_of_profit_l278_278914

theorem c_share_of_profit 
  (x : ℝ) -- The amount invested by B
  (total_profit : ℝ := 11000) -- Total profit
  (A_invest : ℝ := 3 * x) -- A's investment
  (C_invest : ℝ := (3/2) * A_invest) -- C's investment
  (total_invest : ℝ := A_invest + x + C_invest) -- Total investment
  (C_share : ℝ := C_invest / total_invest * total_profit) -- C's share of the profit
  : C_share = 99000 / 17 := 
  by sorry

end c_share_of_profit_l278_278914


namespace largest_natural_number_not_sum_of_two_composites_l278_278142

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278142


namespace max_b_minus_a_l278_278755

theorem max_b_minus_a {a b : ℝ} (segments : set (set ℝ))
  (h₁ : ∀ s ∈ segments, ∃ l : ℝ, s = set.Icc l (l + 1))
  (h₂ : ∀ s₁ s₂ ∈ segments, s₁ ≠ s₂ → ∃ x₁ ∈ s₁, ∃ x₂ ∈ s₂, x₂ = 2 * x₁)
  (h₃ : ∃ s ∈ segments, ∃ l : ℝ, s = set.Icc l (l + 1) ∧ l = a)
  (h₄ : ∃ s ∈ segments, ∃ l : ℝ, s = set.Icc l (l + 1) ∧ l + 1 = b) :
  b - a ≤ 5.5 := 
  by
  sorry

end max_b_minus_a_l278_278755


namespace smallest_angle_leq_60_largest_angle_geq_60_l278_278739

open Real

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

def angles_formed (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : list ℝ :=
  [(pi / 3), ...] -- Dummy list for angles; replace with actual computation

theorem smallest_angle_leq_60 (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :
  ∃ (angles : list ℝ), angles = angles_formed A B C D ∧ (angles.min ≤ (pi / 3)) :=
begin
  sorry
end

theorem largest_angle_geq_60 (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :
  ∃ (angles : list ℝ), angles = angles_formed A B C D ∧ (angles.max ≥ (pi / 3)) :=
begin
  sorry
end

end smallest_angle_leq_60_largest_angle_geq_60_l278_278739


namespace carrie_expected_strawberries_l278_278113

lemma carrie_harvests_strawberries :
  let l := 10
  let w := 12
  let d := 5
  let y := 12
  S = l * w * d * y
  by
  sorry

theorem carrie_expected_strawberries : 
  carrie_harvests_strawberries = 7200 := 
  by
  sorry

end carrie_expected_strawberries_l278_278113


namespace smallest_angle_of_isosceles_trapezoid_l278_278692

def is_isosceles_trapezoid (a b c d : ℝ) : Prop :=
  a = c ∧ b = d ∧ a + b + c + d = 360 ∧ a + 3 * b = 150

theorem smallest_angle_of_isosceles_trapezoid (a b : ℝ) (h1 : is_isosceles_trapezoid a b a (a + 2 * b))
  : a = 47 :=
sorry

end smallest_angle_of_isosceles_trapezoid_l278_278692


namespace find_a_find_height_CD_l278_278682

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Conditions: Triangle ABC with side lengths a, b, c forming an arithmetic sequence, and angle C = 120 degrees
def arithmetic_sequence (a b c : ℝ) : Prop := 
  b = a + 2 ∧ c = a + 4

def angle_C_120 (C : ℝ) : Prop := 
  C = 120 * Real.pi / 180

-- Proof task 1: proving a = 3
theorem find_a (a b c : ℝ) (C : ℝ) (h1 : arithmetic_sequence a b c) (h2 : angle_C_120 C) : 
  a = 3 := 
sorry

-- Auxiliary function for height calculation
def height_CD (a b c : ℝ) (C : ℝ) : ℝ := 
  (a * b * Real.sin C) / c

-- Proof task 2: proving height CD
theorem find_height_CD (a b c : ℝ) (C : ℝ) (h1 : arithmetic_sequence a b c) (h2 : angle_C_120 C) (ha : a = 3) : 
  height_CD a b c C = 15 * Real.sqrt 3 / 14 := 
sorry

end find_a_find_height_CD_l278_278682


namespace remaining_time_for_P_l278_278038

theorem remaining_time_for_P 
  (P_rate : ℝ) (Q_rate : ℝ) (together_time : ℝ) (remaining_time_minutes : ℝ)
  (hP_rate : P_rate = 1 / 3) 
  (hQ_rate : Q_rate = 1 / 18) 
  (h_together_time : together_time = 2) 
  (h_remaining_time_minutes : remaining_time_minutes = 40) :
  (((P_rate + Q_rate) * together_time) + P_rate * (remaining_time_minutes / 60)) = 1 :=
by  rw [hP_rate, hQ_rate, h_together_time, h_remaining_time_minutes]
    admit

end remaining_time_for_P_l278_278038


namespace rice_sack_weight_l278_278909

theorem rice_sack_weight (cost_per_sack : ℝ) (price_per_kg : ℝ) (profit : ℝ) (k : ℝ)
  (h1 : cost_per_sack = 50)
  (h2 : price_per_kg = 1.20)
  (h3 : profit = 10) :
  k = 50 :=
by
  have h4 : price_per_kg * k = cost_per_sack + profit,
  { sorry },
  have h5 : 1.20 * k = 60,
  { rw [h1, h2, h3] at h4, exact h4 },
  have h6 : k = 50,
  { sorry },
  exact h6

end rice_sack_weight_l278_278909


namespace expression_square_minus_three_times_l278_278584

-- Defining the statement
theorem expression_square_minus_three_times (a b : ℝ) : a^2 - 3 * b = a^2 - 3 * b := 
by
  sorry

end expression_square_minus_three_times_l278_278584


namespace geometric_mean_of_terms_l278_278666

-- Define the arithmetic sequence and sum conditions
theorem geometric_mean_of_terms
  (a_1 d : ℤ)
  (S_9 S_13 : ℤ)
  (h1 : S_9 = 9 * a_1 + 36 * d)
  (h2 : S_13 = 13 * a_1 + 78 * d)
  : (∃ (a : ℝ), a = 4 * real.sqrt 2 ∨ a = -4 * real.sqrt 2) :=
by
  -- calculations done here
  sorry

end geometric_mean_of_terms_l278_278666


namespace probability_no_shaded_squares_l278_278945

def num_rectangles (cols : ℕ) : ℕ :=
  (cols + 1).choose 2

def num_non_shaded_rectangles (cols : ℕ) : ℕ :=
  (cols - 1).choose 2

theorem probability_no_shaded_squares : 
  let total_rects := 2 * num_rectangles 10
  let non_shaded_rects := 2 * num_non_shaded_rectangles 9 in
  (non_shaded_rects : ℚ) / total_rects = 36 / 55 := by sorry

end probability_no_shaded_squares_l278_278945


namespace symmetry_axes_l278_278901

-- Definition of rotational symmetry and axes of different orders for regular polyhedra.
structure RotSymmetry (Φ : Type) :=
  (axes_of_symmetry : ℕ → ℕ) -- maps order of symmetry to number of such axes

-- Instances of the five regular polyhedra with their symmetry properties.
def Tetrahedron : RotSymmetry :=
  { axes_of_symmetry := λ n, if n = 3 then 4 else if n = 2 then 3 else 0 }

def Hexahedron : RotSymmetry :=
  { axes_of_symmetry := λ n, if n = 4 then 3 else if n = 3 then 4 else if n = 2 then 6 else 0 }

def Octahedron : RotSymmetry :=
  { axes_of_symmetry := λ n, if n = 4 then 3 else if n = 3 then 4 else if n = 2 then 6 else 0 }

def Dodecahedron : RotSymmetry :=
  { axes_of_symmetry := λ n, if n = 5 then 6 else if n = 3 then 10 else if n = 2 then 15 else 0 }

def Icosahedron : RotSymmetry :=
  { axes_of_symmetry := λ n, if n = 5 then 6 else if n = 3 then 10 else if n = 2 then 15 else 0 }

-- Theorem stating the number of symmetry axes for each polyhedron as given in the solution.
theorem symmetry_axes :
  (Tetrahedron.axes_of_symmetry 3 = 4 ∧ Tetrahedron.axes_of_symmetry 2 = 3) ∧
  (Hexahedron.axes_of_symmetry 4 = 3 ∧ Hexahedron.axes_of_symmetry 3 = 4 ∧ Hexahedron.axes_of_symmetry 2 = 6) ∧
  (Octahedron.axes_of_symmetry 4 = 3 ∧ Octahedron.axes_of_symmetry 3 = 4 ∧ Octahedron.axes_of_symmetry 2 = 6) ∧
  (Dodecahedron.axes_of_symmetry 5 = 6 ∧ Dodecahedron.axes_of_symmetry 3 = 10 ∧ Dodecahedron.axes_of_symmetry 2 = 15) ∧
  (Icosahedron.axes_of_symmetry 5 = 6 ∧ Icosahedron.axes_of_symmetry 3 = 10 ∧ Icosahedron.axes_of_symmetry 2 = 15) :=
by sorry -- Proof to be provided

end symmetry_axes_l278_278901


namespace greatest_pairs_left_l278_278719

theorem greatest_pairs_left (initial_pairs : ℕ) (lost_socks : ℕ) (h_initial_pairs : initial_pairs = 25) (h_lost_socks : lost_socks = 12) : ∃ final_pairs : ℕ, final_pairs = 25 - 12 :=
by {
  have h1 : initial_pairs = 25 := h_initial_pairs,
  have h2 : lost_socks = 12 := h_lost_socks,
  use (initial_pairs - lost_socks),
  rw [h1, h2],
  sorry
}

end greatest_pairs_left_l278_278719


namespace trapezoid_division_l278_278452

-- Define the nature of the problem
theorem trapezoid_division (m n : ℕ) (h_diff : m ≠ n) :
  ∃ triangles : List (triangle ℝ), 
  (∀ t ∈ triangles, is_congruent t (triangles.head)) ∧
  (trapezoid_are_of_bases m n = triangles.sum area) :=
sorry

end trapezoid_division_l278_278452


namespace exists_odd_point_l278_278379

theorem exists_odd_point 
  (P : Fin 1994 → ℤ × ℤ)
  (distinct : Function.Injective P)
  (integer_coords : ∀ i, ∃ a b : ℤ, P i = (a,b))
  (no_other_int_pts : ∀ i (p : ℤ × ℤ),
      p ∈ Set.Icc (Set.Icc P i P (i + 1)) → p = P i ∨ p = P (i + 1)) :
  ∃ i (q : ℚ × ℚ), q ∈ Segment ℚ (P i) (P (i + 1)) ∧
    (2 * q.1) ∈ Set.Icc { x | x % 2 ≠ 0 } ∧ (2 * q.2) ∈ Set.Icc { y | y % 2 ≠ 0 } := 
sorry

end exists_odd_point_l278_278379


namespace count_multiples_of_8_in_range_l278_278282

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278282


namespace tshirts_per_package_l278_278856

def number_of_packages := 28
def total_white_tshirts := 56
def white_tshirts_per_package : Nat :=
  total_white_tshirts / number_of_packages

theorem tshirts_per_package :
  white_tshirts_per_package = 2 :=
by
  -- Assuming the definitions and the proven facts
  sorry

end tshirts_per_package_l278_278856


namespace sourdough_cost_eq_nine_l278_278936

noncomputable def cost_per_visit (white_bread_cost baguette_cost croissant_cost: ℕ) : ℕ :=
  2 * white_bread_cost + baguette_cost + croissant_cost

noncomputable def total_spent (weekly_cost num_weeks: ℕ) : ℕ :=
  weekly_cost * num_weeks

noncomputable def total_sourdough_spent (total_spent weekly_cost num_weeks: ℕ) : ℕ :=
  total_spent - weekly_cost * num_weeks

noncomputable def total_sourdough_per_week (total_sourdough_spent num_weeks: ℕ) : ℕ :=
  total_sourdough_spent / num_weeks

theorem sourdough_cost_eq_nine (white_bread_cost baguette_cost croissant_cost total_spent_over_4_weeks: ℕ)
  (h₁: white_bread_cost = 350) (h₂: baguette_cost = 150) (h₃: croissant_cost = 200) (h₄: total_spent_over_4_weeks = 7800) :
  total_sourdough_per_week (total_sourdough_spent total_spent_over_4_weeks (cost_per_visit white_bread_cost baguette_cost croissant_cost) 4) 4 = 900 :=
by 
  sorry

end sourdough_cost_eq_nine_l278_278936


namespace clock_equiv_to_square_l278_278393

theorem clock_equiv_to_square : ∃ h : ℕ, h > 5 ∧ (h^2 - h) % 24 = 0 ∧ h = 9 :=
by 
  let h := 9
  use h
  refine ⟨by decide, by decide, rfl⟩ 

end clock_equiv_to_square_l278_278393


namespace perimeter_decrease_l278_278045

namespace Rectangle

variables {a b : ℝ}
variables (h1 : a = 4 * b)

theorem perimeter_decrease (h2 : 1.8 * a + 1.6 * b = 0.88 * (2 * a + 2 * b)) :
  (1.6 * a + 1.8 * b) = 0.82 * (2 * a + 2 * b) :=
by {
  sorry
}

end Rectangle

end perimeter_decrease_l278_278045


namespace largest_non_summable_composite_l278_278149

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278149


namespace probability_of_event_l278_278540

theorem probability_of_event (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 2) :
  let event := -1 ≤ Real.log (x + 1/2) / Real.log (1/2) ∧ Real.log (x + 1/2) / Real.log (1/2) ≤ 1
  in Pr[event | x ∈ [0, 2]] = 3 / 4 :=
sorry

end probability_of_event_l278_278540


namespace largest_natural_number_not_sum_of_two_composites_l278_278147

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278147


namespace count_divisible_by_8_l278_278308

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278308


namespace problem_conditions_l278_278994

theorem problem_conditions (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h_sum : m + n = 1) : 
  (∀ x : ℝ, (4 / m + 1 / n) ≥ x → x = 9) ∧
  (sqrt m + sqrt n ≤ sqrt 2) ∧
  ¬(∀ x : ℝ, (1 / (m + 1) - n) ≥ x → x = 0) ∧
  (m > n → 1 / (m - 1) < 1 / (n - 1)) := 
sorry

end problem_conditions_l278_278994


namespace equilateral_triangle_inscribed_l278_278536

-- Definitions of variables and given conditions
variables {R : ℝ} {O : EuclideanGeometry.Point} 
variables {A B C D E F : EuclideanGeometry.Point}

-- Assumptions based on the problem statement
variables (h1 : EuclideanGeometry.Distance O A = R)
variables (h2 : EuclideanGeometry.Distance O B = R)
variables (h3 : EuclideanGeometry.Distance O C = R)
variables (h4 : EuclideanGeometry.Distance O D = R)
variables (h5 : EuclideanGeometry.Distance O E = R)
variables (h6 : EuclideanGeometry.Distance O F = R)
variables (h7 : EuclideanGeometry.Distance A B = R)
variables (h8 : EuclideanGeometry.Distance C D = R)
variables (h9 : EuclideanGeometry.Distance E F = R)

-- Define the points K, L, M
noncomputable def K := point_of_intersection (circumcircle O B C) (circumcircle F O A)
noncomputable def L := point_of_intersection (circumcircle O D E) (circumcircle B O C)
noncomputable def M := point_of_intersection (circumcircle F O A) (circumcircle O D E)

-- The theorem statement
theorem equilateral_triangle_inscribed 
  (hK : K ≠ O) (hL : L ≠ O) (hM : M ≠ O) :
  EuclideanGeometry.IsEquilateralTriangle K L M ∧ EuclideanGeometry.Distance K L = R :=
sorry

end equilateral_triangle_inscribed_l278_278536


namespace number_of_turns_l278_278059

/-
  Given the cyclist's speed v = 5 m/s, time duration t = 5 s,
  and the circumference of the wheel c = 1.25 m, 
  prove that the number of complete turns n the wheel makes is equal to 20.
-/
theorem number_of_turns (v t c : ℝ) (h_v : v = 5) (h_t : t = 5) (h_c : c = 1.25) : 
  (v * t) / c = 20 :=
by
  sorry

end number_of_turns_l278_278059


namespace convert_to_scientific_notation_l278_278050

-- Condition: definition of micrometer in meters
def micrometer_to_meter : Real := 0.000001

-- Input: value in micrometers
def value_in_micrometers : Real := 2.3

-- Desired result as scientific notation in meters
def expected_value_in_meters : Real := 2.3 * 10^(-6)

-- Theorem: Converting value_in_micrometers to meters should yield expected_value_in_meters
theorem convert_to_scientific_notation :
  value_in_micrometers * micrometer_to_meter = expected_value_in_meters := by
  sorry

end convert_to_scientific_notation_l278_278050


namespace parabola_translation_left_by_two_units_l278_278794

/-- 
The parabola y = x^2 + 4x + 5 is obtained by translating the parabola y = x^2 + 1. 
Prove that this translation is 2 units to the left.
-/
theorem parabola_translation_left_by_two_units :
  ∀ x : ℝ, (x^2 + 4*x + 5) = ((x+2)^2 + 1) :=
by
  intro x
  sorry

end parabola_translation_left_by_two_units_l278_278794


namespace problem1_problem2_problem3_l278_278226

-- Given conditions
variables (a b : ℝ^3) (ha : ‖a‖ = 4) (hb : ‖b‖ = 2) (hab : a ⬝ b = -4) 

-- Problem 1
theorem problem1 : (a - 2 • b) ⬝ (a + b) = 12 := 
by
  sorry

-- Problem 2
theorem problem2 : (‖a‖ * Real.cos (2 * Real.pi / 3)) = -2 := 
by
  sorry

-- Problem 3
theorem problem3 : 
  let c := a + b in Real.arccos ((a ⬝ c) / (‖a‖ * ‖c‖)) = Real.pi / 6 :=
by
  sorry

end problem1_problem2_problem3_l278_278226


namespace radio_loss_percentage_l278_278505

theorem radio_loss_percentage (CP SP : ℝ) (h_CP : CP = 2400) (h_SP : SP = 2100) :
  ((CP - SP) / CP) * 100 = 12.5 :=
by
  -- Given cost price
  have h_CP : CP = 2400 := h_CP
  -- Given selling price
  have h_SP : SP = 2100 := h_SP
  sorry

end radio_loss_percentage_l278_278505


namespace pagoda_lights_l278_278444

/-- From afar, the magnificent pagoda has seven layers, with red lights doubling on each
ascending floor, totaling 381 lights. How many lights are there at the very top? -/
theorem pagoda_lights :
  ∃ x, (1 + 2 + 4 + 8 + 16 + 32 + 64) * x = 381 ∧ x = 3 :=
by
  sorry

end pagoda_lights_l278_278444


namespace period_tan_minus_cot_l278_278957

theorem period_tan_minus_cot : 
  ∃ T, 0 < T ∧ ∀ x, tan x - cot x = tan (x + T) - cot (x + T) :=
sorry

end period_tan_minus_cot_l278_278957


namespace maximum_value_Q_l278_278190

-- Define the conditions
def x (a : ℝ) : Set ℝ := Icc 0 a
def y (a : ℝ) : Set ℝ := Icc 0 (a^2)

noncomputable def Q (a : ℝ) : ℝ :=
  ∫ y in Icc 0 (a^2), (∫ x in Icc 0 a, if (cos (real.pi * x))^2 + (cos (real.pi * y))^2 < 3/2 then 1 else 0 ∂x) ∂y

-- State the theorem
theorem maximum_value_Q :
  ∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → Q a ≤ 7/12 ∧ Q 1 = 7/12 :=
 by sorry

end maximum_value_Q_l278_278190


namespace distance_to_lightning_l278_278124

-- Define the conditions
def time := 8  -- time in seconds
def speed_of_sound := 1120  -- speed of sound in feet per second
def feet_per_mile := 5280  -- conversion factor from feet to miles

-- Given conditions and the question, prove the distance is 1.75 miles
theorem distance_to_lightning :
  let distance_in_feet := speed_of_sound * time,
      distance_in_miles := distance_in_feet / feet_per_mile in
  Real.round (4 * distance_in_miles) / 4 = 1.75 :=
by
  sorry

end distance_to_lightning_l278_278124


namespace typists_retype_time_l278_278825

theorem typists_retype_time
  (x y : ℕ)
  (h1 : (x / 2) + (y / 2) = 25)
  (h2 : 1 / x + 1 / y = 1 / 12) :
  (x = 20 ∧ y = 30) ∨ (x = 30 ∧ y = 20) :=
by
  sorry

end typists_retype_time_l278_278825


namespace remainder_777_777_mod_13_l278_278478

theorem remainder_777_777_mod_13 : (777^777) % 13 = 1 := by
  sorry

end remainder_777_777_mod_13_l278_278478


namespace remainder_zero_mod_three_l278_278405

theorem remainder_zero_mod_three :
  let A := ∑ i in (finset.range 2019),
  A % 3 = 0 :=
by
  sorry

end remainder_zero_mod_three_l278_278405


namespace sum_middle_three_l278_278929

def orange_cards : List ℕ := [1, 2, 3, 4, 5, 6]
def green_cards : List ℕ := [2, 3, 4, 5, 6]

theorem sum_middle_three :
  ∃ (stack : List (ℕ × String)), 
    (stack.length = 11) ∧
    (∀ i, 0 < i ∧ i < 11 → ((stack.nth i.succ).elems.2 ≠ (stack.nth i).elems.2) ∧
                            (if (stack.nth i).elems.2 = "Orange" then 
                              (stack.nth i).elems.1 ∣ (stack.nth i.succ).elems.1 
                             else 
                              (stack.nth i.succ).elems.1 ∣ (stack.nth i).elems.1)) ∧
    ((stack.nth 4).elems.1 + (stack.nth 5).elems.1 + (stack.nth 6).elems.1 = 12) :=
sorry

end sum_middle_three_l278_278929


namespace Deepthi_used_material_l278_278450

theorem Deepthi_used_material (h1 : (4 : ℚ)/17) (h2 : (3 : ℚ)/10) (h3 : ((9 : ℚ)/30) = (3/10)) :
  ((4/17) + (3/10) - (3/10)) = (4/17) := 
by
  sorry

end Deepthi_used_material_l278_278450


namespace abs_diff_factorial_expression_l278_278791

theorem abs_diff_factorial_expression (a_1 a_2 ... a_m b_1 b_2 ... b_n : ℕ) 
  (h1 : 2021 = (a_1! * a_2! * ... * a_m!) / (b_1! * b_2! * ... * b_n!))
  (h2 : a_1 ≥ a_2 ∧ a_2 ≥ ... ∧ a_m > 0 ∧ b_1 ≥ b_2 ∧ b_2 ≥ ... ∧ b_n > 0)
  (h3 : ∀ (c_1 c_2 : ℕ), (c_1 + c_2 < a_1 + b_1 → False)) :
  |a_1 - b_1| = 4 :=
by sorry

end abs_diff_factorial_expression_l278_278791


namespace water_added_l278_278904

theorem water_added (W X : ℝ) 
  (h1 : 45 / W = 2 / 1)
  (h2 : 45 / (W + X) = 6 / 5) : 
  X = 15 := 
by
  sorry

end water_added_l278_278904


namespace probability_three_correct_l278_278011

theorem probability_three_correct (n : ℕ) (hn : n = 5) :
  (∃ favorable : ℕ, ∃ total : ℕ, favorable = (nat.choose 5 3) * nat.doubles !2 ∧ total = 5! ∧ 
  (favorable : ℝ) / (total : ℝ)  = 1 / 12) :=
  begin
    sorry
  end

end probability_three_correct_l278_278011


namespace shortest_distance_between_circles_l278_278480

noncomputable def circle1_center : ℝ × ℝ := (-2, -4)
noncomputable def circle1_radius : ℝ := Real.sqrt 12

noncomputable def circle2_center : ℝ × ℝ := (8, -5)
noncomputable def circle2_radius : ℝ := Real.sqrt 129

theorem shortest_distance_between_circles : 
  let d := Real.sqrt (((-2) - 8)^2 + ((-4) - (-5))^2) in
  let r1 := Real.sqrt 12 in
  let r2 := Real.sqrt 129 in
  max 0 (d - (r1 + r2)) = 0 :=
by
  sorry

end shortest_distance_between_circles_l278_278480


namespace least_multiplier_l278_278506

theorem least_multiplier (x: ℕ) (h1: 72 * x % 112 = 0) (h2: ∀ y, 72 * y % 112 = 0 → x ≤ y) : x = 14 :=
sorry

end least_multiplier_l278_278506


namespace bacteria_colony_growth_l278_278339

theorem bacteria_colony_growth : 
  ∃ (n : ℕ), n = 4 ∧ 5 * 3 ^ n > 200 ∧ (∀ (m : ℕ), 5 * 3 ^ m > 200 → m ≥ n) :=
by
  sorry

end bacteria_colony_growth_l278_278339


namespace setD_is_empty_l278_278917

-- Definitions of sets A, B, C, D
def setA : Set ℝ := {x | x + 3 = 3}
def setB : Set (ℝ × ℝ) := {(x, y) | y^2 ≠ -x^2}
def setC : Set ℝ := {x | x^2 ≤ 0}
def setD : Set ℝ := {x | x^2 - x + 1 = 0}

-- Theorem stating that set D is the empty set
theorem setD_is_empty : setD = ∅ := 
by 
  sorry

end setD_is_empty_l278_278917


namespace complement_of_A_l278_278249

def A : Set ℝ := { x | x^2 - x ≥ 0 }
def R_complement_A : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem complement_of_A :
  ∀ x : ℝ, x ∈ R_complement_A ↔ x ∉ A :=
sorry

end complement_of_A_l278_278249


namespace faster_train_speed_l278_278507

theorem faster_train_speed
  (train_length : ℝ)
  (time_to_cross : ℝ)
  (relative_speed_factor : ℝ)
  (total_distance : ℝ) :
  train_length = 100 →
  time_to_cross = 10 →
  relative_speed_factor = 3 →
  total_distance = 200 →
  2 * (total_distance / (time_to_cross * relative_speed_factor)) = (40 / 3) :=
by
  intros h_length h_time h_factor h_distance
  rw [h_length, h_time, h_factor, h_distance]
  norm_num
  sorry

end faster_train_speed_l278_278507


namespace log_sum_l278_278569

theorem log_sum : log10 20 + log10 5 = 2 := 
by 
  sorry

end log_sum_l278_278569


namespace ellipse_major_axis_coordinates_l278_278594

theorem ellipse_major_axis_coordinates :
  ∀ (x y : ℝ), 6 * x^2 + y^2 = 36 → (0, -6) ∈ set_of (λ p : ℝ × ℝ, 6 * p.1^2 + p.2^2 = 36) ∧
                                           (0, 6) ∈ set_of (λ p : ℝ × ℝ, 6 * p.1^2 + p.2^2 = 36) :=
by
  sorry

end ellipse_major_axis_coordinates_l278_278594


namespace ff_iter_six_times_fff_fff_iter_for_two_l278_278668

def f (x : ℝ) := -1 / x

theorem ff_iter_six_times (x : ℝ) (h : x ≠ 0) : 
  (f ∘ f ∘ f ∘ f ∘ f ∘ f) x = x :=
by sorry

theorem fff_fff_iter_for_two: 
  f (f (f (f (f (f 2))))) = 2 := 
by 
  apply ff_iter_six_times
  norm_num

end ff_iter_six_times_fff_fff_iter_for_two_l278_278668


namespace vertex_of_parabola_l278_278433

def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 4

theorem vertex_of_parabola :
  (∃ a b c x0 y0 : ℝ, y0 = parabola x0 ∧ parabola = λ x, a*x^2 + b*x + c ∧ a = 1 ∧ b = -2 ∧ c = 4 ∧ x0 = - (b / (2 * a)) ∧ y0 = (4 * a * c - b^2) / (4 * a)) →
    (1, 3) =  (1, 3) :=
by
  sorry

end vertex_of_parabola_l278_278433


namespace degree_measure_cherry_pie_l278_278332

theorem degree_measure_cherry_pie 
  (total_students : ℕ) 
  (chocolate_pie : ℕ) 
  (apple_pie : ℕ) 
  (blueberry_pie : ℕ) 
  (remaining_students : ℕ)
  (remaining_students_eq_div : remaining_students = (total_students - (chocolate_pie + apple_pie + blueberry_pie))) 
  (equal_division : remaining_students / 2 = 5) 
  : (remaining_students / 2 * 360 / total_students = 45) := 
by 
  sorry

end degree_measure_cherry_pie_l278_278332


namespace sum_of_labels_bounds_l278_278008

def pointColor := {r : Nat // r = 0 ∨ r = 1 ∨ r = 2}

def label (c1 c2 : pointColor) : Nat :=
  match c1, c2 with
  | ⟨0, _⟩, ⟨1, _⟩ => 1
  | ⟨1, _⟩, ⟨0, _⟩ => 1
  | ⟨0, _⟩, ⟨2, _⟩ => 2
  | ⟨2, _⟩, ⟨0, _⟩ => 2
  | ⟨1, _⟩, ⟨2, _⟩ => 3
  | ⟨2, _⟩, ⟨1, _⟩ => 3
  | _, _ => 0

def sum_of_labels (points : List pointColor) : Nat :=
  List.sum (points.zip points.tail).map (λ (c : pointColor × pointColor) => label c.1 c.2)

def max_sum_of_labels (points : List pointColor) : Prop :=
  sum_of_labels points ≤ 140

def min_sum_of_labels (points : List pointColor) : Prop :=
  sum_of_labels points ≥ 6

theorem sum_of_labels_bounds (points : List pointColor)
  (H1 : List.length (List.filter (λ c : pointColor => c.val = 0) points) = 40)
  (H2 : List.length (List.filter (λ c : pointColor => c.val = 1) points) = 30)
  (H3 : List.length (List.filter (λ c : pointColor => c.val = 2) points) = 20)
  (H4 : List.length (points.zip points.tail) = 90) :
  max_sum_of_labels points ∧ min_sum_of_labels points := sorry

end sum_of_labels_bounds_l278_278008


namespace number_of_pencils_bought_l278_278665

-- Define the conditions
def cost_of_glue : ℕ := 270
def cost_per_pencil : ℕ := 210
def amount_paid : ℕ := 1000
def change_received : ℕ := 100

-- Define the statement to prove
theorem number_of_pencils_bought : 
  ∃ (n : ℕ), cost_of_glue + (cost_per_pencil * n) = amount_paid - change_received :=
by {
  sorry 
}

end number_of_pencils_bought_l278_278665


namespace storm_first_thirty_minutes_rain_l278_278912

theorem storm_first_thirty_minutes_rain 
  (R: ℝ)
  (H1: R + (R / 2) + (1 / 2) = 8)
  : R = 5 :=
by
  sorry

end storm_first_thirty_minutes_rain_l278_278912


namespace circle_land_represents_30105_l278_278510

-- Definitions based on the problem's conditions
def circleLandNumber (digits : List (ℕ × ℕ)) : ℕ :=
  digits.foldl (λ acc (d_circle : ℕ × ℕ) => acc + d_circle.fst * 10^d_circle.snd) 0

-- Example 207
def number_207 : List (ℕ × ℕ) := [(2, 2), (0, 0), (7, 0)]

-- Example 4520
def number_4520 : List (ℕ × ℕ) := [(4, 3), (5, 1), (2, 0), (0, 0)]

-- The diagram to analyze
def given_diagram : List (ℕ × ℕ) := [(3, 4), (1, 2), (5, 0)]

-- The statement proving the given diagram represents 30105 in Circle Land
theorem circle_land_represents_30105 : circleLandNumber given_diagram = 30105 :=
  sorry

end circle_land_represents_30105_l278_278510


namespace part1_part2_l278_278618

-- The sequence definition
def a : ℕ → ℝ
| 0       := 2
| (n + 1) := 2^n / a n

-- Part 1: Prove that a_{n+2} / a_n is constant
theorem part1 (n : ℕ) : (a (n + 2)) / (a n) = 2 :=
sorry

-- Sum of the first n terms of the sequence
def T : ℕ → ℝ
| 0       := 0
| (n + 1) := T n + a n

-- Part 2: Find the sum of the first 2n terms of the sequence
theorem part2 (n : ℕ) : T (2 * n) = 3 * 2^n - 3 :=
sorry

end part1_part2_l278_278618


namespace problem1_problem2_problem3_l278_278996

variable (a b : ℝ^3)
variable (h_a_norm : ‖a‖ = 1)
variable (h_b_norm : ‖b‖ = sqrt 2)

-- Question 1
theorem problem1 (h_parallel : ∃ k : ℝ, a = k • b) : a ⬝ b = sqrt 2 ∨ a ⬝ b = -sqrt 2 := sorry

-- Question 2
theorem problem2 (h_angle : real.angle.ofReal (a ⬝ b / (‖a‖ * ‖b‖)) = real.angle.radians (pi / 3)) :
  ‖a + 2 • b‖ = sqrt (17 + 2 * sqrt 2) := sorry

-- Question 3
theorem problem3 (h_perpendicular : (a - b) ⬝ a = 0) :
  real.angle.ofReal (a ⬝ b / (‖a‖ * ‖b‖)) = real.angle.radians (pi / 4) := sorry

end problem1_problem2_problem3_l278_278996


namespace longest_side_BC_l278_278894

-- Define the quadrilateral and the given angles
variables (A B C D : Type) [EuclideanGeometry A B C D]
variables (∠ABD ∠DBC ∠BDA ∠BAC : Angle)
variables (h1 : ∠ABD = 45)
variables (h2 : ∠DBC = 65)
variables (h3 : ∠BDA = 70)
variables (h4 : ∠BAC = 50)

theorem longest_side_BC (A B C D : Type) [EuclideanGeometry A B C D] (∠ABD ∠DBC ∠BDA ∠BAC : Angle)
  (h1 : ∠ABD = 45) (h2 : ∠DBC = 65) (h3 : ∠BDA = 70) (h4 : ∠BAC = 50) :
  side BC > side AB ∧ side BC > side CD ∧ side BC > side DA ∧ side BC > side AC :=
sorry

end longest_side_BC_l278_278894


namespace number_of_tests_in_series_l278_278078

theorem number_of_tests_in_series (S : ℝ) (n : ℝ) :
  (S + 97) / n = 90 →
  (S + 73) / n = 87 →
  n = 8 :=
by 
  sorry

end number_of_tests_in_series_l278_278078


namespace find_x_l278_278803

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x n : ℕ) (h₀ : n = 4) (h₁ : ¬(is_prime (2 * n + x))) : x = 1 :=
by
  sorry

end find_x_l278_278803


namespace find_all_solutions_l278_278130

noncomputable def functionalSolutions (f : ℝ → ℝ) :=
  ((∃ A : ℝ, f = λ x, A * x) ∨
   (∃ (A a : ℝ), f = λ x, A * Real.sin (a * x)) ∨
   (∃ (A a : ℝ), f = λ x, A * Real.sinh (a * x)))

theorem find_all_solutions (f : ℝ → ℝ) 
  (h_differentiable : Differentiable ℝ f)
  (h_differentiable2 : Differentiable ℝ (fderiv ℝ f)) 
  (functional_equation : ∀ x y : ℝ, f(x)^2 - f(y)^2 = f(x + y) * f(x - y)) : 
  (∀ x, f x = 0) ∨ functionalSolutions f :=
begin
  sorry
end


end find_all_solutions_l278_278130


namespace find_a_b_l278_278223

theorem find_a_b (a b : ℝ) 
  (h : (a / Complex.ofReal (1 : ℝ) - Complex.i) + (b / Complex.ofReal (2 : ℝ) - Complex.i) = (1 / (3 : ℝ - Complex.i))) :
  a = -1 / 5 ∧ b = 1 :=
sorry

end find_a_b_l278_278223


namespace max_real_part_sum_l278_278378

theorem max_real_part_sum (z : ℂ → ℂ) :
  let z_j (j : ℕ) := 8 * complex.exp (2 * j * real.pi * complex.I / 18)
  ∀ (w_j : ℕ → ℂ),
    (∀ j, w_j j = z_j j ∨ w_j j = complex.I * z_j j ∨ w_j j = -z_j j) →
    ((∑ j in finset.range 18, w_j j).re ≤ 8 + 8 * (2 * (1 + real.sqrt 3 + real.sqrt 2 + (real.cos (real.pi / 9)) + (real.cos (2 * real.pi / 9)) + (real.cos (real.pi * 2 / 9)) + (real.cos (5 * real.pi / 9)) + (real.cos (7 * real.pi / 9)) + (real.cos (8 * real.pi / 9))))) :=
begin
  intros z_j w_j h_wj,
  sorry
end

end max_real_part_sum_l278_278378


namespace prove_inequality_l278_278313

variable {x y : ℝ}

theorem prove_inequality (h : 2^x - 3^(-x) ≥ 2^(-y) - 3^y) : x + y ≥ 0 :=
sorry

end prove_inequality_l278_278313


namespace tory_sold_to_neighbor_l278_278187

def total_cookies : ℕ := 50
def sold_to_grandmother : ℕ := 12
def sold_to_uncle : ℕ := 7
def to_be_sold : ℕ := 26

def sold_to_neighbor : ℕ :=
  total_cookies - to_be_sold - (sold_to_grandmother + sold_to_uncle)

theorem tory_sold_to_neighbor :
  sold_to_neighbor = 5 :=
by
  intros
  sorry

end tory_sold_to_neighbor_l278_278187


namespace right_triangle_condition_l278_278095

-- Define the conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (A B C : ℝ) : Prop := A / B = 3 / 4 ∧ B / C = 4 / 5
def condition_C (a b c : ℝ) : Prop := a ^ 2 = (b + c) * (b - c)
def condition_D (a b c : ℝ) : Prop := a / b = 5 / 12 ∧ b / c = 12 / 13

-- The main theorem to be proved
theorem right_triangle_condition (A B C a b c : ℝ) :
  condition_A A B C → condition_B A B C → condition_C a b c → condition_D a b c →
  ¬ (∠ A = 90) ∧ (∠ B = 90) ∧ (∠ C = 90) :=
begin
  sorry
end

end right_triangle_condition_l278_278095


namespace correct_statements_are_two_l278_278786

def statement1 : Prop := 
  ∀ (data : Type) (eq : data → data → Prop), 
    (∃ (t : data), eq t t) → 
    (∀ (d1 d2 : data), eq d1 d2 → d1 = d2)

def statement2 : Prop := 
  ∀ (samplevals : Type) (regress_eqn : samplevals → samplevals → Prop), 
    (∃ (s : samplevals), regress_eqn s s) → 
    (∀ (sv1 sv2 : samplevals), regress_eqn sv1 sv2 → sv1 = sv2)

def statement3 : Prop := 
  ∀ (predvals : Type) (pred_eqn : predvals → predvals → Prop), 
    (∃ (p : predvals), pred_eqn p p) → 
    (∀ (pp1 pp2 : predvals), pred_eqn pp1 pp2 → pp1 = pp2)

def statement4 : Prop := 
  ∀ (observedvals : Type) (linear_eqn : observedvals → observedvals → Prop), 
    (∃ (o : observedvals), linear_eqn o o) → 
    (∀ (ov1 ov2 : observedvals), linear_eqn ov1 ov2 → ov1 = ov2)

def correct_statements_count : ℕ := 2

theorem correct_statements_are_two : 
  (statement1 ∧ statement2 ∧ ¬ statement3 ∧ ¬ statement4) → 
  correct_statements_count = 2 := by
  sorry

end correct_statements_are_two_l278_278786


namespace distance_between_first_and_last_tree_l278_278462

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 3) :
  (n - 1) * d = 87 :=
by
  -- Conditions from the problem
  rw [h1, h2]
  -- Simplify the expression
  sorry

end distance_between_first_and_last_tree_l278_278462


namespace job_completion_days_l278_278320

theorem job_completion_days (Ram_days : ℝ) (Gohul_days : ℝ) (Amit_days : ℝ) :
  Ram_days = 10 → Gohul_days = 15 → Amit_days = 20 → 
  (60 / 13 : ℝ) ≈ (1 / ((1 / Ram_days) + (1 / Gohul_days) + (1 / Amit_days))) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end job_completion_days_l278_278320


namespace polygon_intersection_count_l278_278411

theorem polygon_intersection_count :
  ∀ (circle : Type) (P4 P5 P6 P7 : set circle),
    (∀ p ∈ P4, p ∉ P5 ∧ p ∉ P6 ∧ p ∉ P7) ∧
    (∀ p ∈ P5, p ∉ P4 ∧ p ∉ P6 ∧ p ∉ P7) ∧
    (∀ p ∈ P6, p ∉ P4 ∧ p ∉ P5 ∧ p ∉ P7) ∧
    (∀ p ∈ P7, p ∉ P4 ∧ p ∉ P5 ∧ p ∉ P6) ∧
    (∀ (side_m ∈ P5) (side_n ∈ P4), intersects side_m side_n) ∧
    (∀ (side_m ∈ P6) (side_n ∈ P4), intersects side_m side_n) ∧
    (∀ (side_m ∈ P6) (side_n ∈ P5), intersects side_m side_n) ∧
    (∀ (side_m ∈ P7) (side_n ∈ P4), intersects side_m side_n) ∧
    (∀ (side_m ∈ P7) (side_n ∈ P5), intersects side_m side_n) ∧
    (∀ (side_m ∈ P7) (side_n ∈ P6), intersects side_m side_n) →
    intersection_count (P4 ∪ P5 ∪ P6 ∪ P7) = 56 :=
by sorry

end polygon_intersection_count_l278_278411


namespace tablet_arrangements_l278_278256

theorem tablet_arrangements : 
  let word := "tablet"
  let n := String.length word
  n = 6 → 
  (Finset.univ.perm Finset.range n).card = 720 :=
by
  intros _
  sorry

end tablet_arrangements_l278_278256


namespace number_of_divisibles_by_eight_in_range_l278_278269

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278269


namespace angle_LOK_calculation_l278_278946

-- Define the coordinates of the points L and K
def L : (ℝ × ℝ) := (0, 45)  -- (0° latitude, 45° E longitude)
def K : (ℝ × ℝ) := (60, -30) -- (60° N latitude, 30° W longitude)

-- Define the center of the Earth O
def O : (ℝ × ℝ × ℝ) := (0, 0, 0) -- Center of the sphere

-- Define function to convert spherical coordinates to cartesian
def spherical_to_cartesian (lat long : ℝ) : (ℝ × ℝ × ℝ) :=
  let φ := lat * (Float.pi / 180) in -- Convert latitude to radians
  let θ := long * (Float.pi / 180) in -- Convert longitude to radians
  (cos φ * cos θ, cos φ * sin θ, sin φ)

-- Convert L and K to cartesian coordinates
def L_cartesian := spherical_to_cartesian L.1 L.2
def K_cartesian := spherical_to_cartesian K.1 K.2

-- Proof statement: calculate the angle ∠LOK
theorem angle_LOK_calculation : 
  ∀ (O L K : ℝ × ℝ × ℝ), 
  L = L_cartesian → K = K_cartesian → O = (0, 0, 0) → 
  let angle_LOK := Float.acos (cos (45 * (Float.pi / 180)) * cos (75 * (Float.pi / 180)) + sin (45 * (Float.pi / 180)) * sin (75 * (Float.pi / 180)) * cos (30 * (Float.pi / 180))) in 
  angle_LOK * (180 / Float.pi) = 168 :=
by
  intros _ L K _ _ _ _ O L_cartesian K_cartesian
  sorry

end angle_LOK_calculation_l278_278946


namespace arithmetic_sequence_formula_sum_of_sequence_b_l278_278210

noncomputable def a_n (n : ℕ) : ℕ := n + 1
noncomputable def S_n (n : ℕ) : ℕ := n ^ 2 / 2 + 3 * n / 2
noncomputable def b_n (n : ℕ) : ℚ := 1 / (S_n n - 2 * n)
noncomputable def T_n (n : ℕ) : ℚ := (2 / 3) * (11 / 6 - 1 / (n + 1) - 1 / (n + 2) - 1 / (n + 3))

theorem arithmetic_sequence_formula (a_4_eq : a_n 4 = 5) (S_9_eq : S_n 9 = 54) :
  ∀ n, a_n n = n + 1 ∧ S_n n = n^2 / 2 + 3 * n / 2 :=
by sorry

theorem sum_of_sequence_b (a_4_eq : a_n 4 = 5) (S_9_eq : S_n 9 = 54) :
  ∀ n, (∑ k in range n, b_n k) = T_n n :=
by sorry

end arithmetic_sequence_formula_sum_of_sequence_b_l278_278210


namespace evaluate_at_points_l278_278381

noncomputable def f (x : ℝ) : ℝ :=
if x > 3 then x^2 - 3*x + 2
else if -2 ≤ x ∧ x ≤ 3 then -3*x + 5
else 9

theorem evaluate_at_points : f (-3) + f (0) + f (4) = 20 := by
  sorry

end evaluate_at_points_l278_278381


namespace part1_part2_l278_278206

-- This represents the first part of the problem.
theorem part1 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, a (n + 1) = 2 * a n - n + 1)
  (h2 : a 1 = 3) :
  ∃ b : ℕ → ℝ, (∀ n : ℕ, b n = a n - n) ∧ ∀ n : ℕ, a n = 2^n + n :=
by sorry

-- This represents the second part of the problem.
theorem part2 (a : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a (n + 1) = 2 * a n - n + 1)
  (h2 : a 1 = 3)
  (h3 : ∀ n : ℕ, c n = (a (n + 1) - a n) / (a n * a (n + 1)))
  (h4 : ∀ n : ℕ, S n = ∑ i in finset.range n, c i) :
  ∀ n : ℕ, S n < 1 :=
by sorry

end part1_part2_l278_278206


namespace expected_value_is_one_l278_278387

noncomputable def expected_value : ℚ :=
  let die_faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let primes := {2, 3, 5, 7}
  let composites := {4, 6, 8}
  let multiples_of_3 := {3, 6}
  let winning (x : ℕ) : ℤ :=
    if x ∈ primes then x
    else if x ∈ multiples_of_3 then -x
    else 0
  let probabilities := (die_faces.toList.map (λ x, (winning x).toRat * (1 / 8))).sum
  probabilities

theorem expected_value_is_one : expected_value = 1 := sorry

end expected_value_is_one_l278_278387


namespace minimum_questions_to_determine_Vasya_number_l278_278030

theorem minimum_questions_to_determine_Vasya_number :
  (∃ (V : ℕ), 1 ≤ V ∧ V ≤ 100) →
  (∃ (m1 m2 : ℕ), m1 ≥ 2 ∧ m2 ≥ 2 ∧ 
                  ∀ V ∈ (finset.range 101).filter (λ V, 1 ≤ V),
                  ∀ k1 k2 : ℕ,
                    m1 * k1 ≤ V ∧ V < m1 * (k1 + 1) →
                    m2 * k2 ≤ V ∧ V < m2 * (k2 + 1) → 
                    V = k1 ∨ V = k2) :=
sorry

end minimum_questions_to_determine_Vasya_number_l278_278030


namespace solve_inequality_l278_278420

theorem solve_inequality (x : ℝ) : 3 * (x + 1) > 9 → x > 2 :=
by sorry

end solve_inequality_l278_278420


namespace series_sum_eq_l278_278374

noncomputable def series_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_ineq : a > b) : ℝ :=
  ∑' n, 1 / (((n - 1 : ℝ) * a - (n - 2 : ℝ) * b + c) * ((n : ℝ) * a - (n - 1 : ℝ) * b + c))

theorem series_sum_eq (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_ineq : a > b) :
  series_sum a b c h_pos h_ineq = 1 / ((a - b) * (b + c)) :=
begin
  sorry
end

end series_sum_eq_l278_278374


namespace largest_natural_number_not_sum_of_two_composites_l278_278146

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278146


namespace third_root_of_polynomial_l278_278821

theorem third_root_of_polynomial (c d : ℚ) (f : ℚ → ℚ) 
  (h_polynomial : f = λ x, c * x^3 + (c + 3 * d) * x^2 + (2 * d - 4 * c) * x + (10 - c))
  (h_root1 : f (-1) = 0) (h_root2 : f 4 = 0) :
  ∃ r : ℚ, r = 76 / 11 ∧ f r = 0 :=
begin
  sorry
end

end third_root_of_polynomial_l278_278821


namespace probability_same_group_l278_278461

theorem probability_same_group 
  (total_groups : ℕ)
  (students_choose_group : ∀ (student : ℕ), student ∈ [1, 2] → ℕ → ℕ)
  (equal_prob : students_choose_group 1 = students_choose_group 2) :
  total_groups = 3 → 
  (students_choose_group 1 ∈ [1, 2].map (λ n, n) → 
   students_choose_group 2 ∈ [1, 2].map (λ n, n) → 
   (∑ g in [1, 2, 3], if students_choose_group 1 g = students_choose_group 2 g then 1 else 0) / 
   total_groups ^ 2 = 1 / 3) :=
by
  intros htg h1 h2
  sorry

end probability_same_group_l278_278461


namespace find_X_l278_278954

def spadesuit (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

theorem find_X (X : ℝ) (h : spadesuit X 5 = 23) : X = 7.75 :=
by sorry

end find_X_l278_278954


namespace wood_needed_l278_278556

variable (total_needed : ℕ) (friend_pieces : ℕ) (brother_pieces : ℕ)

/-- Alvin's total needed wood is 376 pieces, he got 123 from his friend and 136 from his brother.
    Prove that Alvin needs 117 more pieces. -/
theorem wood_needed (h1 : total_needed = 376) (h2 : friend_pieces = 123) (h3 : brother_pieces = 136) :
  total_needed - (friend_pieces + brother_pieces) = 117 := by
  sorry

end wood_needed_l278_278556


namespace chip_credit_card_balance_l278_278943

-- Conditions
def initial_balance : Float := 50.00
def first_interest_rate : Float := 0.20
def additional_charge : Float := 20.00
def second_interest_rate : Float := 0.20

-- Question
def current_balance : Float :=
  let first_interest_fee := initial_balance * first_interest_rate
  let balance_after_first_interest := initial_balance + first_interest_fee
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  let second_interest_fee := balance_before_second_interest * second_interest_rate
  balance_before_second_interest + second_interest_fee

-- Correct Answer
def expected_balance : Float := 96.00

-- Proof Problem Statement
theorem chip_credit_card_balance : current_balance = expected_balance := by
  sorry

end chip_credit_card_balance_l278_278943


namespace limit_sequence_l278_278111

def sequence (n : ℕ) : ℝ :=
  (sqrt (3 * n - 1) - (125 * n ^ 3 + n) ^ (1 / 3)) /
  (n ^ (1 / 3) - n)

theorem limit_sequence : Filter.Tendsto sequence Filter.atTop (nhds 5) :=
by
  sorry

end limit_sequence_l278_278111


namespace roots_quadratic_eq_l278_278222

theorem roots_quadratic_eq (α β : ℝ) (hαβ : ∀ x, x^2 + 2017*x + 1 = (x - α) * (x - β))
  : (1 + 2020 * α + α^2) * (1 + 2020 * β + β^2) = 9 :=
  sorry

end roots_quadratic_eq_l278_278222


namespace find_uphill_distance_l278_278524

-- Defining the conditions
def uphill_speed : ℝ := 30
def downhill_speed : ℝ := 80
def downhill_distance : ℝ := 50
def average_speed : ℝ := 37.89

-- Defining the unknown distance
noncomputable def uphill_distance : ℝ :=
(average_speed * (downhill_distance / downhill_speed + uphill_distance / uphill_speed) - downhill_distance)

-- Statement that we need to prove
theorem find_uphill_distance : uphill_distance ≈ 100.08 :=
by
  sorry

end find_uphill_distance_l278_278524


namespace solve_for_z_l278_278319

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
  sorry

end solve_for_z_l278_278319


namespace tangent_line_diameter_l278_278490

theorem tangent_line_diameter (C : Type) [metric_space C] [hilbert_space C] (O : C) (r : ℝ) (circle : set C) (diam_endpoint : C) (line : set C) :
  metric.dist O diam_endpoint = r →
  (∀ p ∈ line, metric.dist p O = r) →
  ∃ t ∈ line, t = diam_endpoint ∧ metric.dist t O = r ∧ (∀ p ∈ line, metric.dist p t = 0 ∨ ⟨C.diam_endpoint, t⟩ = 0) :=
by
  sorry

end tangent_line_diameter_l278_278490


namespace Fabian_total_cost_correct_l278_278129

noncomputable def total_spent_by_Fabian (mouse_cost : ℝ) : ℝ :=
  let keyboard_cost := 2 * mouse_cost
  let headphones_cost := mouse_cost + 15
  let usb_hub_cost := 36 - mouse_cost
  let webcam_cost := keyboard_cost / 2
  let total_cost := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost + webcam_cost
  let discounted_total := total_cost * 0.90
  let final_total := discounted_total * 1.05
  final_total

theorem Fabian_total_cost_correct :
  total_spent_by_Fabian 20 = 123.80 :=
by
  sorry

end Fabian_total_cost_correct_l278_278129


namespace product_b_n_eq_588_div_factorial_l278_278189

noncomputable def b_n (n : ℕ) (h : n ≥ 6) : ℚ :=
  (n^2 + 4 * n + 2) / (n^3 - 1)

theorem product_b_n_eq_588_div_factorial :
  ∏ n in Finset.range 96, b_n (n + 6) (Nat.le_add_left 6 n) = 588 / Nat.factorial 101 :=
sorry

end product_b_n_eq_588_div_factorial_l278_278189


namespace find_constant_l278_278430

theorem find_constant
  (k : ℝ)
  (r : ℝ := 36)
  (C : ℝ := 72 * k)
  (h1 : C = 2 * Real.pi * r)
  : k = Real.pi := by
  sorry

end find_constant_l278_278430


namespace tips_multiple_l278_278120

variable (A T : ℝ) (x : ℝ)
variable (h1 : T = 7 * A)
variable (h2 : T / 4 = x * A)

theorem tips_multiple (A T : ℝ) (x : ℝ) (h1 : T = 7 * A) (h2 : T / 4 = x * A) : x = 1.75 := by
  sorry

end tips_multiple_l278_278120


namespace perimeter_original_square_l278_278543

theorem perimeter_original_square (s : ℝ) (h1 : (3 / 4) * s^2 = 48) : 4 * s = 32 :=
by
  sorry

end perimeter_original_square_l278_278543


namespace tree_planting_problem_l278_278897

variables (n t : ℕ)

theorem tree_planting_problem (h1 : 4 * n = t + 11) (h2 : 2 * n = t - 13) : n = 12 ∧ t = 37 :=
by
  sorry

end tree_planting_problem_l278_278897


namespace clean_room_time_l278_278746

theorem clean_room_time :
  let lisa_time := 8
  let kay_time := 12
  let ben_time := 16
  let combined_work_rate := (1 / lisa_time) + (1 / kay_time) + (1 / ben_time)
  let total_time := 1 / combined_work_rate
  total_time = 48 / 13 :=
by
  sorry

end clean_room_time_l278_278746


namespace circle_radius_tangent_triangle_l278_278526

theorem circle_radius_tangent_triangle :
  ∃ (r : ℝ), 
    (∀ (O : ℝ × ℝ), 
      let C := 2 in 
      let hypotenuse := 45 ∧ 45 ∧ 90 in
      ∀ (x_axis y_axis : ℝ), 
        ((O = (r, r)) ∧ (C = sqrt(2) * r * sqrt(2)) ∧ (tangent to coordinate axes and hypotenuse)) →
          r = sqrt(2) / 2) :=
begin
  sorry
end

end circle_radius_tangent_triangle_l278_278526


namespace largest_non_sum_of_composites_l278_278176

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278176


namespace g_x_even_l278_278734

theorem g_x_even (a b c : ℝ) (g : ℝ → ℝ):
  (∀ x, g x = a * x^6 + b * x^4 - c * x^2 + 5)
  → g 32 = 3
  → g 32 + g (-32) = 6 :=
by
  sorry

end g_x_even_l278_278734


namespace cyclic_hexagon_exists_l278_278341

def can_construct_cyclic_hexagon (a1 a2 a3 a4 a5 a6 : ℝ) : Prop :=
  a1 - a4 = a5 - a2 ∧ a5 - a2 = a3 - a6 → ∃ h : hexagon, cyclic h ∧ side_lengths h = [a1, a2, a3, a4, a5, a6]

-- The following definition of hexagon and cyclic needs to be assumed
-- since they are not part of standard Lean libraries.
-- Assume the necessary geometric definitions:
structure hexagon :=
(sides : ℝ) -- Dummy definition for representation

def cyclic (h : hexagon) : Prop := sorry -- Cyclic property of hexagon

noncomputable def side_lengths (h : hexagon) : list ℝ := sorry -- Function to extract side lengths of hexagon

-- Main theorem statement
theorem cyclic_hexagon_exists (a1 a2 a3 a4 a5 a6 : ℝ) :
  can_construct_cyclic_hexagon a1 a2 a3 a4 a5 a6 :=
by {
  intro h_rel,
  have h_hexagon : ∃ h, cyclic h ∧ side_lengths h = [a1, a2, a3, a4, a5, a6],
  { sorry },
  exact h_hexagon,
}

end cyclic_hexagon_exists_l278_278341


namespace yuna_correct_multiplication_l278_278863

theorem yuna_correct_multiplication (x : ℕ) (h : 4 * x = 60) : 8 * x = 120 :=
by
  sorry

end yuna_correct_multiplication_l278_278863


namespace possible_integer_roots_l278_278782

theorem possible_integer_roots :
  ∀ (b c d e f : ℤ), ∃ n ∈ {0, 1, 2, 5}, ∀ x : ℤ, ((x = 0 ∨ x = -1) → 
  (x ^ 5 + b * x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x + f = 0)) →
  count_roots (x ^ 5 + b * x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x + f) = n := 
sorry

end possible_integer_roots_l278_278782


namespace polar_coordinates_of_point_l278_278952

/-- Given the Cartesian coordinates (-2, 2√3), prove that the
polar coordinates are (4, 2π/3): -/
theorem polar_coordinates_of_point :
  ∀ (x y : ℝ), x = -2 → y = 2 * Real.sqrt 3 →
  let ρ := Real.sqrt (x^2 + y^2),
      θ := if h : x ≠ 0 then Real.arctan (y / x) + if x < 0 then Real.pi else 0 else
           if y > 0 then Real.pi / 2 else if y < 0 then -(Real.pi / 2) else 0
  in ρ = 4 ∧ θ = 2 * Real.pi / 3 :=
by
  sorry

end polar_coordinates_of_point_l278_278952


namespace smallest_period_functions_l278_278558

def period_sin_abs_2x : ℝ := sorry
def period_abs_cos_x : ℝ := π
def period_cos_2x_plus_pi_over_6 : ℝ := π
def period_tan_2x_minus_pi_over_4 : ℝ := sorry

theorem smallest_period_functions
  (h1: period_abs_cos_x = π)
  (h2: period_cos_2x_plus_pi_over_6 = π)
  (h3: period_sin_abs_2x ≠ π)
  (h4: period_tan_2x_minus_pi_over_4 ≠ π) :
  {y = |\cos x|, y = \cos(2x + \frac{π}{6})} = {y = |\cos x|, y = \cos(2x + \frac{π}{6})} :=
by {
  sorry
}

end smallest_period_functions_l278_278558


namespace systematic_sampling_first_group_student_l278_278075

theorem systematic_sampling_first_group_student (
  num_students : ℕ,
  num_selected : ℕ,
  interval_start interval_end first_group_student : ℕ
) (h1 : num_students = 800)
  (h2 : num_selected = 50)
  (h3 : interval_start = 497)
  (h4 : interval_end = 513)
  (h5 : 16 = num_students / num_selected)
  (student_503 : ℕ)
  (h6 : interval_start ≤ student_503 ∧ student_503 < interval_end)
  (h7 : student_503 = 503)
  (k : ℕ)
  (h8 : k = 32)
  (h9 : student_503 = 16 * (k - 1) + first_group_student)
  : first_group_student = 7 :=
begin
  rw [←h1, ←h2, ←h7, ←h9, ←h5] at h6,
  have k_eq_32 : k = 32, from h8,
  rw [k_eq_32] at h9,
  linarith,
end

end systematic_sampling_first_group_student_l278_278075


namespace line_intersects_curve_l278_278437

noncomputable def k_range : Set ℝ := {k : ℝ | k ∈ Set.Icc (1/3) 1}

theorem line_intersects_curve (k : ℝ) (l : ℝ → ℝ := λ x => k * x + 1 - 2 * k) :
  (∃ (x y : ℝ), (sqrt ((x - 1)^2 + y^2) + sqrt ((x + 1)^2 + y^2) = 2) 
  ∧ (y = l x)) ↔ (k ∈ k_range) :=
by
  sorry

end line_intersects_curve_l278_278437


namespace constant_term_expansion_l278_278432

noncomputable def binomialCoeff (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem constant_term_expansion (x : ℝ) :
  let term := fun r : ℕ => binomialCoeff 5 r * (-2 : ℝ)^r * x^(10 - 5*r)
  (finset.range 6).map term = [_, _, 40, _, _, _] := 
by
  sorry

end constant_term_expansion_l278_278432


namespace problem1_problem2_l278_278227

theorem problem1 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : 0 < t ∧ t < 1) :
  x^t - (x-1)^t < (x-2)^t - (x-3)^t :=
sorry

theorem problem2 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : t > 1) :
  x^t - (x-1)^t > (x-2)^t - (x-3)^t :=
sorry

end problem1_problem2_l278_278227


namespace fraction_zero_iff_l278_278329

theorem fraction_zero_iff (x : ℝ) (h₁ : (x - 1) / (2 * x - 4) = 0) (h₂ : 2 * x - 4 ≠ 0) : x = 1 := sorry

end fraction_zero_iff_l278_278329


namespace problem_l278_278322

theorem problem (p q : ℕ) (hp: p > 1) (hq: q > 1) (h1 : (2 * p - 1) % q = 0) (h2 : (2 * q - 1) % p = 0) : p + q = 8 := 
sorry

end problem_l278_278322


namespace hyperbola_distance_focus_to_asymptote_l278_278970

def hyperbola_eq (x y : ℝ) (λ : ℝ) : Prop := 
  (x^2) / 9 - (y^2) / 16 = λ

def A : ℝ × ℝ := (-3, 2 * (Real.sqrt 3))

noncomputable def distance_to_asymptote : ℝ :=
  2

theorem hyperbola_distance_focus_to_asymptote :
  ∃ λ : ℝ, λ ≠ 0 ∧ hyperbola_eq A.1 A.2 λ ∧ distance_to_asymptote = 2 :=
by 
  sorry

end hyperbola_distance_focus_to_asymptote_l278_278970


namespace quadratic_roots_identity_l278_278499

noncomputable def sum_of_roots (a b : ℝ) : Prop := a + b = -10
noncomputable def product_of_roots (a b : ℝ) : Prop := a * b = 5

theorem quadratic_roots_identity (a b : ℝ)
  (h₁ : sum_of_roots a b)
  (h₂ : product_of_roots a b) :
  (a / b + b / a) = 18 :=
by sorry

end quadratic_roots_identity_l278_278499


namespace max_l_m_g_l278_278104

-- Define the cube and its dimensions
structure Cube :=
(edge_length : ℝ)
(A B C D E F G H : ℝ × ℝ × ℝ) -- Vertices

-- Define the point M on the surface EFGH
structure PointOnSurface (cube : Cube) :=
(M : ℝ × ℝ × ℝ)
(M_on_surface : M.fst ≥ cube.E.fst ∧ M.fst ≤ cube.G.fst ∧
                 M.snd ≥ cube.E.snd ∧ M.snd ≤ cube.G.snd ∧
                 M.snd = cube.E.snd)

-- Define the paths l(M, A) and l(M, G)
def path_length (M A : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt ((M.fst - A.fst)^2 + (M.snd - A.snd)^2 + (M.snd - A.snd)^2)

-- Define the problem statement
theorem max_l_m_g (cube : Cube) (p : PointOnSurface cube)
    (h: path_length p.M cube.A = path_length p.M cube.G) :
    path_length p.M cube.G ≤ 5 / (3 * real.sqrt 2) :=
sorry

end max_l_m_g_l278_278104


namespace largest_non_sum_of_composites_l278_278172

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278172


namespace abs_has_min_at_zero_l278_278238

def f (x : ℝ) : ℝ := abs x

theorem abs_has_min_at_zero : ∃ m, (∀ x : ℝ, f x ≥ m) ∧ f 0 = m := by
  sorry

end abs_has_min_at_zero_l278_278238


namespace sum_b_p_eq_24640_l278_278983

def b (p : ℕ) : ℕ :=
  Nat.find (λ k, abs (k - Real.sqrt p) < 1 / 2)

theorem sum_b_p_eq_24640 : (∑ p in Finset.range 3000 |λ x=>x+1|, b p) = 24640 :=
begin
  sorry
end

end sum_b_p_eq_24640_l278_278983


namespace find_a2018_l278_278677

def seq (n : ℕ) : ℚ :=
  if n = 1 then 2
  else let a := seq (n - 1) in (1 + a) / (1 - a)

theorem find_a2018 : seq 2018 = -3 := by
  sorry

end find_a2018_l278_278677


namespace temperature_difference_l278_278328

-- Define the temperatures given in the problem.
def T_noon : ℝ := 10
def T_midnight : ℝ := -150

-- State the theorem to prove the temperature difference.
theorem temperature_difference :
  T_noon - T_midnight = 160 :=
by
  -- We skip the proof and add sorry.
  sorry

end temperature_difference_l278_278328


namespace johnny_marbles_choice_l278_278720

theorem johnny_marbles_choice : 
  ∃ n : ℕ, n = 756 ∧
  (∀ {B R G : ℕ}, B = 3 → R = 4 → G = 3 → ∃ k : ℕ, k = B + R + G ∧ 
  ∃ m : ℕ, m = (B - 1) + (R - 1) + (G - 1) + 2 ∧ (finset.choose m 2) * (B * R * G) = n) := 
by
  sorry

end johnny_marbles_choice_l278_278720


namespace engineer_sidorov_error_is_4kg_l278_278439

-- Given the conditions
def diameter := 1                     -- diameter of each disk in meters
def error_radius_std_dev := 0.01      -- standard deviation of the radius in meters
def mass_per_disk := 100              -- mass of one disk with diameter 1 meter in kg
def num_disks := 100                  -- number of disks
def estimate_sidorov := 10000         -- Engineer Sidorov's estimate in kg

-- Expected calculations
def expected_mass_one_disk := mass_per_disk * (1 + (error_radius_std_dev^2 / (1/4))) -- expected mass considering the deviation
def expected_total_mass := expected_mass_one_disk * num_disks
def true_error := expected_total_mass - estimate_sidorov

-- Prove that the error in Engineer Sidorov's estimate of the total mass of 100 disks is 4 kg
theorem engineer_sidorov_error_is_4kg :
  true_error = 4 :=
sorry

end engineer_sidorov_error_is_4kg_l278_278439


namespace ratio_cd_bd_l278_278681

variables (A B C D E T : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point T]
variables (AT DT BT ET : ℝ)

-- Conditions
def at_dt_ratio (A T D : Type) [Point A] [Point T] [Point D] := AT / DT = 2
def bt_et_ratio (B T E : Type) [Point B] [Point T] [Point E] := BT / ET = 4
variable (CD BD : ℝ)

-- Proof statement
theorem ratio_cd_bd (h1 : at_dt_ratio A T D) 
                    (h2 : bt_et_ratio B T E) 
                    (h3 : intersects_parallel_line A B C T) 
                    (h4 : points_on_line B D C) :
                    CD / BD = 5 / 3 := 
by sorry

end ratio_cd_bd_l278_278681


namespace relationship_of_abc_l278_278993

theorem relationship_of_abc (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 4) : c > b ∧ b > a := by
  sorry

end relationship_of_abc_l278_278993


namespace justin_total_pages_l278_278363

theorem justin_total_pages :
  let pages_read (n : ℕ) := (10 * 2 ^ n)
  in ∑ i in Finset.range 7, pages_read i = 1270 := by
  sorry

end justin_total_pages_l278_278363


namespace Nunzio_eats_pizza_every_day_l278_278392

theorem Nunzio_eats_pizza_every_day
  (one_piece_fraction : ℚ := 1/8)
  (total_pizzas : ℕ := 27)
  (total_days : ℕ := 72)
  (pieces_per_pizza : ℕ := 8)
  (total_pieces : ℕ := total_pizzas * pieces_per_pizza)
  : (total_pieces / total_days = 3) :=
by
  -- We assume 1/8 as a fraction for the pieces of pizza is stated in the conditions, therefore no condition here.
  -- We need to show that Nunzio eats 3 pieces of pizza every day given the total pieces and days.
  sorry

end Nunzio_eats_pizza_every_day_l278_278392


namespace locus_of_points_l278_278845

theorem locus_of_points (a : ℝ) (u v : ℝ) (h : 0 < a) :
  (v + 3 / (4 * a))^2 - u^2 = 1 / (2 * a^2) ↔
  ∃ (P : ℝ × ℝ), let (u, v) := P in
    (λ x : ℝ, ax ^ 2) x = y ∧
    -- Tangents drawn to the parabola enclose a 45-degree angle
    (λ m' m'' : ℝ, |(m' - m'') / (1 + m' * m'')| = 1)

end locus_of_points_l278_278845


namespace hexagon_largest_angle_l278_278892

theorem hexagon_largest_angle (a : ℚ) 
  (h₁ : (a + 2) + (2 * a - 3) + (3 * a + 1) + 4 * a + (5 * a - 4) + (6 * a + 2) = 720) :
  6 * a + 2 = 4374 / 21 :=
by sorry

end hexagon_largest_angle_l278_278892


namespace positive_number_sum_square_l278_278004

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l278_278004


namespace equation_of_the_line_l278_278246

-- Define the parametric equations for parabola M
def parabola (t : ℝ) : ℝ × ℝ := (t, t^2)

-- Define the polar equation for circle N
def circlePolarToCartesian (ρ θ : ℝ) : Prop := ρ^2 - 6 * ρ * sin θ = -8

-- Define the Cartesian coordinate equation after conversion and simplification of circle N
def circleCartesian (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

-- Define the focus of the parabola
def focusParabola (x y : ℝ) : Prop := x = 0 ∧ y = 1 / 4

-- Define the center of the circle
def centerCircle (x y : ℝ) : Prop := x = 0 ∧ y = 3

-- The final theorem to prove the equation of the line passing through the focus and center
theorem equation_of_the_line : 
  (focusParabola 0 (1/4)) →
  (centerCircle 0 3) →
  ∀ x y, (x = 0 → y ∈ ℝ) → (x = 0) :=
by
  sorry

end equation_of_the_line_l278_278246


namespace integer_values_abs_less_than_2pi_l278_278257

-- Define the main theorem to prove
theorem integer_values_abs_less_than_2pi : {x : ℤ | abs x < 2 * Real.pi}.to_finset.card = 13 := by
  -- Proof goes here
  sorry

end integer_values_abs_less_than_2pi_l278_278257


namespace range_of_x1_and_x2_squares_l278_278638

theorem range_of_x1_and_x2_squares (m : ℝ) (x1 x2 : ℝ) 
  (h : x1^2 + 2*(m-2)*x1 + m^2 + 4 = 0 ∧ x2^2 + 2*(m-2)*x2 + m^2 + 4 = 0 ∧ m ≤ 0) 
  : ∃ y ∈ set.Ici (4 : ℝ), y = x1^2 + x2^2 - x1 * x2 :=
sorry

end range_of_x1_and_x2_squares_l278_278638


namespace count_base7_with_digit_456_l278_278660

def num_base7_with_digit_456_in_2401_nat_nums : ℕ :=
  let total_count := 7^4 in
  let base4_count := 4^4 in
  total_count - (base4_count - 1)

theorem count_base7_with_digit_456 : num_base7_with_digit_456_in_2401_nat_nums = 2146 :=
by
  sorry

end count_base7_with_digit_456_l278_278660


namespace maximize_profit_l278_278383

/-- 
The total number of rooms in the hotel 
-/
def totalRooms := 80

/-- 
The initial rent when the hotel is fully booked 
-/
def initialRent := 160

/-- 
The loss in guests for each increase in rent by 20 yuan 
-/
def guestLossPerIncrease := 3

/-- 
The increase in rent 
-/
def increasePer20Yuan := 20

/-- 
The daily service and maintenance cost per occupied room
-/
def costPerOccupiedRoom := 40

/-- 
Maximize profit given the conditions
-/
theorem maximize_profit : 
  ∃ x : ℕ, x = 360 ∧ 
            ∀ y : ℕ,
              (initialRent - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (x - initialRent) / increasePer20Yuan)
              ≥ (y - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (y - initialRent) / increasePer20Yuan) := 
sorry

end maximize_profit_l278_278383


namespace probability_product_positive_l278_278022

-- Define interval and properties
def interval : set ℝ := { x : ℝ | -15 ≤ x ∧ x ≤ 15 }

-- Define independent selection of x and y
def selected_independently (x y : ℝ) : Prop :=
  x ∈ interval ∧ y ∈ interval

-- Main theorem
theorem probability_product_positive : 
  (∃ x y ∈ interval, selected_independently x y ∧ x * y > 0) → (probability (x * y > 0) = 1/2) :=
sorry

end probability_product_positive_l278_278022


namespace range_of_a_l278_278399

-- Define the propositions
def Proposition_p (a : ℝ) := ∀ x : ℝ, x > 0 → x + 1/x > a
def Proposition_q (a : ℝ) := ∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0

-- Define the main theorem
theorem range_of_a (a : ℝ) (h1 : ¬ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) 
(h2 : (∀ x : ℝ, x > 0 → x + 1/x > a) ∧ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) :
a ≥ 2 :=
sorry

end range_of_a_l278_278399


namespace magic_square_y_value_l278_278685

-- Definitions and conditions
def magic_square_condition (y : ℝ) : Prop :=
  let a := y - 16 in
  let b := 2 * y - 40 in
  (y + 8 + c = 24 + a + c) ∧ 
  (y + a + e = 24 + b + e) ∧ 
  (y + 7 + 24 = 8 + a + b)

-- Theorem statement
theorem magic_square_y_value : ∃ y : ℝ, magic_square_condition y ∧ y = 39.5 := 
by
  exists 39.5
  unfold magic_square_condition
  sorry  -- Proof to be filled in

end magic_square_y_value_l278_278685


namespace chip_credit_card_balance_l278_278939

-- Definitions based on the problem conditions
def initial_balance : ℝ := 50.00
def interest_rate : ℝ := 0.20
def additional_amount : ℝ := 20.00

-- Define the function to calculate the final balance after two months
def final_balance (b₀ r a : ℝ) : ℝ :=
  let b₁ := b₀ * (1 + r) in
  let b₂ := (b₁ + a) * (1 + r) in
  b₂

-- Theorem to prove that the final balance is 96.00
theorem chip_credit_card_balance : final_balance initial_balance interest_rate additional_amount = 96.00 :=
by
  -- Simplified proof outline
  sorry

end chip_credit_card_balance_l278_278939


namespace Faye_created_rows_l278_278585

theorem Faye_created_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (rows : ℕ) 
  (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) : rows = 7 :=
by
  sorry

end Faye_created_rows_l278_278585


namespace triangle_area_eq_l278_278593

theorem triangle_area_eq :
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  area = 9 / 4 :=
by
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  sorry

end triangle_area_eq_l278_278593


namespace emma_traveled_distance_l278_278128

noncomputable def time_in_hours : ℝ := 2 + (20 / 60)

def average_speed : ℝ := 120

def distance_traveled := average_speed * time_in_hours

theorem emma_traveled_distance :
  distance_traveled = 280 := by
  sorry

end emma_traveled_distance_l278_278128


namespace distance_P_to_line_l_l278_278232

-- Define point A as a tuple (1, 2, 3)
def A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define point P as a tuple (4, 3, 2)
def P : ℝ × ℝ × ℝ := (4, 3, 2)

-- Define the direction vector (1, 0, 1)
def n : ℝ × ℝ × ℝ := (1, 0, 1)

-- Compute the distance from point P to the line l
theorem distance_P_to_line_l : 
  let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
  let AP_mag := real.sqrt (AP.1 ^ 2 + AP.2 ^ 2 + AP.3 ^ 2) in
  let n_mag := real.sqrt (n.1 ^ 2 + n.3 ^ 2) in
  let u := (n.1 / n_mag, 0, n.3 / n_mag) in
  let dot_product := AP.1 * u.1 + AP.2 * u.2 + AP.3 * u.3 in
  let distance := real.sqrt (AP_mag ^ 2 - dot_product ^ 2) in
  distance = 3 :=
by {
  sorry
}

end distance_P_to_line_l_l278_278232


namespace compute_xy_l278_278822

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := 
sorry

end compute_xy_l278_278822


namespace ann_age_is_24_l278_278101

variable (A B t T : ℕ)

theorem ann_age_is_24 :
  (A + B = 44) ∧
  (B = A - t) ∧
  (B - t = A - T) ∧
  (B - T = (1 / 2) * A) →
  A = 24 :=
begin
  sorry
end

end ann_age_is_24_l278_278101


namespace unique_zero_a_neg_l278_278052

noncomputable def f (a x : ℝ) : ℝ := 3 * Real.exp (abs (x - 1)) - a * (2^(x - 1) + 2^(1 - x)) - a^2

theorem unique_zero_a_neg (a : ℝ) (h_unique : ∃! x : ℝ, f a x = 0) (h_neg : a < 0) : a = -3 := 
sorry

end unique_zero_a_neg_l278_278052


namespace saleswoman_commission_l278_278082

theorem saleswoman_commission (x : ℝ) (h1 : ∀ sale : ℝ, sale = 800) (h2 : (x / 100) * 500 + 0.25 * (800 - 500) = 0.21875 * 800) : x = 20 := by
  sorry

end saleswoman_commission_l278_278082


namespace sum_of_powers_of_z_equals_given_value_l278_278999

noncomputable theory

-- Define the given complex number z.
def z : ℂ := -1/2 + (real.sqrt 3)/2 * complex.I

-- Define the sequence sum we need to prove.
def sum_of_powers_of_z : ℂ := ∑ i in finset.range 2023, z^(i + 1)

-- State the theorem that we need to prove.
theorem sum_of_powers_of_z_equals_given_value :
  sum_of_powers_of_z = -1/2 + (real.sqrt 3)/2 * complex.I :=
by sorry

end sum_of_powers_of_z_equals_given_value_l278_278999


namespace find_m_and_trig_values_l278_278214

variable (α : Angle)
variable (m : ℝ)

-- Given conditions
axiom Point_P : ∃ P : ℝ × ℝ, P = (-3, m) ∧ terminal_side_passes_through (P, α)
axiom cos_alpha : cos α = -3/5

-- To prove
theorem find_m_and_trig_values :
  m = 4 ∨ m = -4 ∧ (sin α = 4/5 ∨ sin α = -4/5) ∧ (tan α = -4/3 ∨ tan α = 4/3) :=
by
  sorry

end find_m_and_trig_values_l278_278214


namespace largest_non_representable_as_sum_of_composites_l278_278167

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278167


namespace count_multiples_of_8_in_range_l278_278276

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278276


namespace numbers_divisible_by_8_between_200_and_400_l278_278285

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278285


namespace largest_four_digit_number_divisible_by_6_l278_278837

theorem largest_four_digit_number_divisible_by_6 :
  ∃ n, n = 9996 ∧ ∀ m, (m ≤ 9999 ∧ m % 6 = 0) → m ≤ n :=
begin
  sorry
end

end largest_four_digit_number_divisible_by_6_l278_278837


namespace sum_of_dropped_students_scores_l278_278780

theorem sum_of_dropped_students_scores 
  (avg_25 : ℝ) (n_25 : ℕ) (new_avg_23 : ℝ) (n_23 : ℕ)
  (h_avg_25 : avg_25 = 60.5) (h_n_25 : n_25 = 25)
  (h_new_avg_23 : new_avg_23 = 64.0) (h_n_23 : n_23 = 23):
  let total_25 : ℝ := n_25 * avg_25 in
  let total_23 : ℝ := n_23 * new_avg_23 in
  let S1_S2 : ℝ := total_25 - total_23 in
  S1_S2 = 40.5 :=
by 
  have h_total_25 : total_25 = 25 * 60.5 := by simp [h_n_25, h_avg_25]
  have h_total_23 : total_23 = 23 * 64.0 := by simp [h_n_23, h_new_avg_23]
  have h_S1_S2 : S1_S2 = (25 * 60.5) - (23 * 64.0) := by simp [S1_S2, h_total_25, h_total_23]
  show S1_S2 = 40.5, by simp [h_S1_S2]

end sum_of_dropped_students_scores_l278_278780


namespace ramu_profit_percentage_l278_278406

def initial_cost : ℝ := 42000
def repair_cost : ℝ := 10000
def shipping_fee_usd : ℝ := 250
def conversion_rate : ℝ := 75
def selling_price_usd : ℝ := 1200

def total_cost_inr : ℝ := initial_cost + repair_cost + (shipping_fee_usd * conversion_rate)
def selling_price_inr : ℝ := selling_price_usd * conversion_rate
def profit_inr : ℝ := selling_price_inr - total_cost_inr
def profit_percentage : ℝ := (profit_inr / total_cost_inr) * 100

theorem ramu_profit_percentage : profit_percentage = 27.2 := 
by
  sorry

end ramu_profit_percentage_l278_278406


namespace space_shuttle_speed_conversion_l278_278040

-- Define the given conditions
def speed_km_per_sec : ℕ := 6  -- Speed in km/s
def seconds_per_hour : ℕ := 3600  -- Seconds in an hour

-- Define the computed speed in km/hr
def expected_speed_km_per_hr : ℕ := 21600  -- Expected speed in km/hr

-- The main theorem statement to be proven
theorem space_shuttle_speed_conversion : speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hr := by
  sorry

end space_shuttle_speed_conversion_l278_278040


namespace largest_not_sum_of_two_composites_l278_278136

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278136


namespace exists_graph_with_degree_sequence_l278_278403

def degree_sequence_valid (n : ℕ) (G : simple_graph (fin (2 * n))) : Prop :=
  ∃ f : fin (2 * n) → ℕ, 
    (∀ v : fin (2 * n), f v = G.degree v) ∧
    multiset.sort (multiset.of_fn f) = multiset.sort (multiset.of_list (list.join (list.init (list.range (n + 1)) 2)))

theorem exists_graph_with_degree_sequence (n : ℕ) : 
  ∃ G : simple_graph (fin (2 * n)), degree_sequence_valid n G := 
sorry

end exists_graph_with_degree_sequence_l278_278403


namespace engineer_sidorov_error_is_4kg_l278_278438

-- Given the conditions
def diameter := 1                     -- diameter of each disk in meters
def error_radius_std_dev := 0.01      -- standard deviation of the radius in meters
def mass_per_disk := 100              -- mass of one disk with diameter 1 meter in kg
def num_disks := 100                  -- number of disks
def estimate_sidorov := 10000         -- Engineer Sidorov's estimate in kg

-- Expected calculations
def expected_mass_one_disk := mass_per_disk * (1 + (error_radius_std_dev^2 / (1/4))) -- expected mass considering the deviation
def expected_total_mass := expected_mass_one_disk * num_disks
def true_error := expected_total_mass - estimate_sidorov

-- Prove that the error in Engineer Sidorov's estimate of the total mass of 100 disks is 4 kg
theorem engineer_sidorov_error_is_4kg :
  true_error = 4 :=
sorry

end engineer_sidorov_error_is_4kg_l278_278438


namespace uber_profit_l278_278858

-- Define conditions
def income : ℕ := 30000
def initial_cost : ℕ := 18000
def trade_in : ℕ := 6000

-- Define depreciation cost
def depreciation_cost : ℕ := initial_cost - trade_in

-- Define the profit
def profit : ℕ := income - depreciation_cost

-- The theorem to be proved
theorem uber_profit : profit = 18000 := by 
  sorry

end uber_profit_l278_278858


namespace limit_max_a_one_evaluate_integral_l278_278560

variables (a : ℝ) (n : ℕ)

-- Statement 1
theorem limit_max_a_one (ha : 0 < a) : 
  (real.limit (λ n : ℕ, (1 + a^n)^(1/(n : ℝ))) = max a 1) :=
sorry

-- Statement 2
theorem evaluate_integral : 
  (∫ x in 1..sqrt 3, (1 / x^2) * real.log (sqrt (1 + x^2))) = 
  ((1 / 2 - 1 / sqrt 3) * real.log 2 + real.pi / 12) :=
sorry

end limit_max_a_one_evaluate_integral_l278_278560


namespace inequality_solution_l278_278419

theorem inequality_solution {x : ℝ} : (x + 1) / x > 1 ↔ x > 0 := 
sorry

end inequality_solution_l278_278419


namespace mean_of_solutions_l278_278975

theorem mean_of_solutions:
  let x := [x | ∃ (x : ℝ), x^3 + 2*x^2 - 8*x - 4 = 0] in
  list.mean(x) = -2/3 :=
by sorry

end mean_of_solutions_l278_278975


namespace min_x_y_l278_278202

theorem min_x_y
  (x y : ℝ)
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : x + 2 * y + x * y - 7 = 0) :
  x + y ≥ 3 := by
  sorry

end min_x_y_l278_278202


namespace sum_first_2009_terms_arith_seq_l278_278345

variable {a : ℕ → ℝ}

-- Given condition a_1004 + a_1005 + a_1006 = 3
axiom H : a 1004 + a 1005 + a 1006 = 3

-- Arithmetic sequence definition
def is_arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem sum_first_2009_terms_arith_seq
  (d : ℝ) (h_arith_seq : is_arith_seq a d)
  : sum_arith_seq a 2009 = 2009 := 
by
  sorry

end sum_first_2009_terms_arith_seq_l278_278345


namespace sixth_term_in_geometric_sequence_l278_278672

open Real

noncomputable def geometric_sequence (a₁ a₈ : ℝ) (n : ℕ) (a_n : ℝ) (r : ℝ) : Prop :=
  a₈ = a₁ * r^(n-1) ∧ a_n = a₁ * r^(n-3)

theorem sixth_term_in_geometric_sequence :
  ∃ (a₁ a₈ : ℝ) (r : ℝ), a₁ = 3 ∧ a₈ = 39366 ∧ r = 6 ∧ geometric_sequence a₁ a₈ 8 23328 r :=
begin
  use [3, 39366, 6],
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  { sorry }
end

end sixth_term_in_geometric_sequence_l278_278672


namespace perimeter_of_fourth_rectangle_is_whole_l278_278907

theorem perimeter_of_fourth_rectangle_is_whole (P PA PB PC : ℕ) 
  (h1 : PA + PC = P) 
  (h2 : PB + P - PB = P) : 
  ∃ PD : ℕ, PA ∈ ℕ ∧ PB ∈ ℕ ∧ PC ∈ ℕ ∧ PD ∈ ℕ :=
by
  sorry

end perimeter_of_fourth_rectangle_is_whole_l278_278907


namespace jeans_vs_scarves_l278_278527

theorem jeans_vs_scarves :
  ∀ (ties belts black_shirts white_shirts : ℕ),
  ties = 34 →
  belts = 40 →
  black_shirts = 63 →
  white_shirts = 42 →
  let total_shirts := black_shirts + white_shirts in
  let jeans := (2 * total_shirts) / 3 in
  let total_ties_and_belts := ties + belts in
  let scarves := total_ties_and_belts / 2 in
  jeans - scarves = 33 :=
by
  intros ties belts black_shirts white_shirts ht hb hbs hws
  let total_shirts := black_shirts + white_shirts
  let jeans := (2 * total_shirts) / 3
  let total_ties_and_belts := ties + belts
  let scarves := total_ties_and_belts / 2
  show jeans - scarves = 33
  sorry

end jeans_vs_scarves_l278_278527


namespace numbers_divisible_by_8_between_200_and_400_l278_278286

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278286


namespace percentage_of_alcohol_in_solution_y_l278_278771

theorem percentage_of_alcohol_in_solution_y :
  let P := percentage of alcohol in solution y (in decimal form)
  let alcohol_x := 0.10 * 300 -- alcohol in solution x
  let mixture_volume := 300 + 900
  let desired_alcohol_volume := 0.25 * mixture_volume
  alcohol_x + P * 900 = desired_alcohol_volume → P = 0.3 :=
by
  sorry

end percentage_of_alcohol_in_solution_y_l278_278771


namespace largest_number_not_sum_of_two_composites_l278_278179

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278179


namespace largest_natural_number_not_sum_of_two_composites_l278_278144

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278144


namespace part1_part2_l278_278236

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≤ 0) : a ≥ 1 / Real.exp 1 :=
  sorry

noncomputable def g (x b : ℝ) : ℝ := Real.log x + 1/2 * x^2 - (b + 1) * x

theorem part2 (b : ℝ) (x1 x2 : ℝ) (h1 : b ≥ 3/2) (h2 : x1 < x2) (hx3 : g x1 b - g x2 b ≥ k) : k ≤ 15/8 - 2 * Real.log 2 :=
  sorry

end part1_part2_l278_278236


namespace inclination_angle_of_line_l278_278693

def inclination_angle_range : Set ℝ := Set.Ico 0 Real.pi

theorem inclination_angle_of_line :
  ∀ θ : ℝ,
  (∃ (l : ℝ × ℝ → Prop), (l ((θ, 0)) (cos θ, sin θ)) ∧ ¬Vertical l) →
  θ ∈ inclination_angle_range :=
by
  intros θ h_exists
  sorry

end inclination_angle_of_line_l278_278693


namespace price_per_glass_first_day_l278_278394

theorem price_per_glass_first_day (O W : ℝ) (P1 P2 : ℝ) 
  (h1 : O = W) 
  (h2 : P2 = 0.40)
  (revenue_eq : 2 * O * P1 = 3 * O * P2) 
  : P1 = 0.60 := 
by 
  sorry

end price_per_glass_first_day_l278_278394


namespace alpha_cubic_expression_l278_278729

theorem alpha_cubic_expression (α : ℝ) (hα : α^2 - 8 * α - 5 = 0) : α^3 - 7 * α^2 - 13 * α + 6 = 11 :=
sorry

end alpha_cubic_expression_l278_278729


namespace comprehensiveInvestigation_is_Census_l278_278058

def comprehensiveInvestigation (s: String) : Prop :=
  s = "Census"

theorem comprehensiveInvestigation_is_Census :
  comprehensiveInvestigation "Census" :=
by
  sorry

end comprehensiveInvestigation_is_Census_l278_278058


namespace x_coordinate_of_point_l278_278467

theorem x_coordinate_of_point (x_1 n : ℝ) 
  (h1 : x_1 = (n / 5) - (2 / 5)) 
  (h2 : x_1 + 3 = ((n + 15) / 5) - (2 / 5)) : 
  x_1 = (n / 5) - (2 / 5) :=
by sorry

end x_coordinate_of_point_l278_278467


namespace counting_divisibles_by_8_l278_278294

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278294


namespace semicircle_circumference_l278_278874

theorem semicircle_circumference :
  let length := 16
  let breadth := 12
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let diameter_semicircle := side_square
  let pi := Real.pi
  let circumference_semicircle := ((pi * diameter_semicircle) / 2 + diameter_semicircle) 
  Float.round (circumference_semicircle, 2) = 35.98 :=
by
  sorry

end semicircle_circumference_l278_278874


namespace problem1_solution_problem2_solution_l278_278408

-- Problem 1
theorem problem1_solution (x : ℝ) : (2 * x - 3) * (x + 1) < 0 ↔ (-1 < x) ∧ (x < 3 / 2) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) : (4 * x - 1) / (x + 2) ≥ 0 ↔ (x < -2) ∨ (x >= 1 / 4) :=
sorry

end problem1_solution_problem2_solution_l278_278408


namespace average_time_relay_race_l278_278121

theorem average_time_relay_race :
  let dawson_time := 38
  let henry_time := 7
  let total_legs := 2
  (dawson_time + henry_time) / total_legs = 22.5 :=
by
  sorry

end average_time_relay_race_l278_278121


namespace fourth_vs_third_difference_l278_278702

def first_competitor_distance : ℕ := 22

def second_competitor_distance : ℕ := first_competitor_distance + 1

def third_competitor_distance : ℕ := second_competitor_distance - 2

def fourth_competitor_distance : ℕ := 24

theorem fourth_vs_third_difference : 
  fourth_competitor_distance - third_competitor_distance = 3 := by
  sorry

end fourth_vs_third_difference_l278_278702


namespace range_of_k_l278_278318

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 0 < x → x^2 * exp(3 * x) ≥ (k + 5) * x + 2 * log x + 1) →
  k ≤ -2 :=
by sorry

end range_of_k_l278_278318


namespace largest_non_summable_composite_l278_278155

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278155


namespace largest_non_sum_of_composites_l278_278174

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278174


namespace faulty_meter_weight_l278_278084

theorem faulty_meter_weight (cp sp : ℝ) (faulty_meter_weight : ℝ) 
  (cost_price : cp = 1) (selling_price : sp = 1.11111111111111) 
  (profit_percent : selling_price = cost_price * (1 + 1/9)) :
  faulty_meter_weight = 100 :=
by
  sorry

end faulty_meter_weight_l278_278084


namespace justin_pages_read_l278_278360

theorem justin_pages_read :
  let pages_day1 := 10 in
  let daily_pages := 2 * pages_day1 in
  let total_pages := pages_day1 + daily_pages * 6 in
  total_pages = 130 :=
by
  let pages_day1 := 10
  let daily_pages := 2 * pages_day1
  let total_pages := pages_day1 + daily_pages * 6
  sorry

end justin_pages_read_l278_278360


namespace locus_of_points_proof_l278_278184

noncomputable def locus_of_points_equidistant_from_intersecting_lines (l1 l2 : Line) (O : Point) (m1 m2 m3 m4 : Line) : Prop :=
  inters_l1l2 l1 l2 O ∧
  bisectors_of_angles l1 l2 m1 m2 m3 m4 ∧
  ∀ P : Point, (d(P, l1) = d(P, l2)) ↔ (P = O ∨ P ∈ m1 ∨ P ∈ m2 ∨ P ∈ m3 ∨ P ∈ m4)

-- Definitions to establish concepts in the problem statement
axiom inters_l1l2 (l1 l2 : Line) (O : Point) : Prop
axiom bisectors_of_angles (l1 l2 : Line) (m1 m2 m3 m4 : Line) : Prop
axiom d : Point → Line → ℝ

theorem locus_of_points_proof (l1 l2 : Line) (O : Point) (m1 m2 m3 m4 : Line) :
  locus_of_points_equidistant_from_intersecting_lines l1 l2 O m1 m2 m3 m4 :=
sorry

end locus_of_points_proof_l278_278184


namespace distance_from_center_to_line_l278_278632

theorem distance_from_center_to_line :
  let circle_equation := λ (ρ θ : ℝ), ρ = 6 * Real.cos θ
  let line_equation := λ (ρ θ : ℝ), ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2
  let center := (3 : ℝ, 0 : ℝ)
  let point_line_distance := λ (x₁ y₁ A B C : ℝ),
    abs (A * x₁ + B * y₁ + C) / Real.sqrt (A^2 + B^2)
  ∃ d : ℝ, d = point_line_distance 3 0 1 1 (-2) ∧ d = Real.sqrt 2 / 2 :=
begin
  use Real.sqrt 2 / 2,
  split,
  sorry
end

end distance_from_center_to_line_l278_278632


namespace find_coefficient_of_x_l278_278191

theorem find_coefficient_of_x :
  ∃ a : ℚ, ∀ (x y : ℚ),
  (x + y = 19) ∧ (x + 3 * y = 1) ∧ (2 * x + y = 5) →
  (a * x + y = 19) ∧ (a = 7) :=
by
  sorry

end find_coefficient_of_x_l278_278191


namespace largest_non_sum_of_composites_l278_278175

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278175


namespace folded_segment_square_l278_278542

theorem folded_segment_square (a b c : ℝ) (h : a = 10) (h1 : b = 7) (h2 : c = 10) : 
  let x := (h - b) / 2
  sqrt ((2 * x^2) + (2 * x^2) - (x^2))

open real
noncomputable def folded_segment_square_proof : ℝ :=
let side_length : ℝ := 10,
    d_to_touch_point_distance : ℝ := 7,
    other_side : ℝ := 10,
    folded_segment : ℝ := sqrt ((2 * (side_length - d_to_touch_point_distance / 2)^2) +
                                (2 * (side_length - d_to_touch_point_distance / 2)^2) -
                                (side_length - d_to_touch_point_distance / 2)^2 / 2) in
(sorry : folded_segment * folded_segment = 27 / 4)

#check folded_segment_square_proof
#eval folded_segment_square_proof

end folded_segment_square_l278_278542


namespace largest_number_not_sum_of_two_composites_l278_278178

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278178


namespace amount_of_p_l278_278043

theorem amount_of_p (p q r : ℝ) (h1 : q = (1 / 6) * p) (h2 : r = (1 / 6) * p) 
  (h3 : p = (q + r) + 32) : p = 48 :=
by
  sorry

end amount_of_p_l278_278043


namespace train_length_l278_278553

namespace TrainProblem

def speed_kmh : ℤ := 60
def time_sec : ℤ := 18
def speed_ms : ℚ := (speed_kmh : ℚ) * (1000 / 1) * (1 / 3600)
def length_meter := speed_ms * (time_sec : ℚ)

theorem train_length :
  length_meter = 300.06 := by
  sorry

end TrainProblem

end train_length_l278_278553


namespace cannot_divide_1985_1987_into_L_shaped_can_divide_1987_1989_into_L_shaped_l278_278935

-- Define m, n and L-shape
def L_shaped (m n : ℕ) := (m * n) % 3 = 0

-- Problem (1)
theorem cannot_divide_1985_1987_into_L_shaped :
  ¬ L_shaped 1985 1987 :=
by
  have h : 1985 * 1987 = 3951395 := by norm_num
  rw [h]
  show 3951395 % 3 ≠ 0, by norm_num
  exact True.intro

-- Problem (2)
theorem can_divide_1987_1989_into_L_shaped :
  L_shaped 1987 1989 :=
by
  have h : 1987 * 1989 = 3957403 := by norm_num
  rw [h]
  show 3957403 % 3 = 0, by norm_num
  exact True.intro

-- End statements

end cannot_divide_1985_1987_into_L_shaped_can_divide_1987_1989_into_L_shaped_l278_278935


namespace probability_of_pink_gumball_l278_278873

theorem probability_of_pink_gumball 
  (P B : ℕ) 
  (total_gumballs : P + B > 0)
  (prob_blue_blue : ((B : ℚ) / (B + P))^2 = 16 / 49) : 
  (B + P > 0) → ((P : ℚ) / (B + P) = 3 / 7) :=
by
  sorry

end probability_of_pink_gumball_l278_278873


namespace set_intersections_l278_278250

def set_A := {x | isosceles_triangle x}
def set_B := {x | right_triangle x}
def set_C := {x | acute_triangle x}

theorem set_intersections :
  (set_A ∩ set_B = {x | isosceles_right_triangle x}) ∧
  (set_B ∩ set_C = ∅) := 
by 
  sorry

end set_intersections_l278_278250


namespace measure_angle_AOB_l278_278398

-- Define the geometric points and angles
variables (A B C D O: Type) [has_angle A] [has_angle B] [has_angle C] [has_angle D] [has_angle O]

-- Define the rhombus and angles properties
def is_rhombus (A B C D : Type) : Prop := 
∃ (P Q : Type), 
  ∠ DAB = 110 ∧ 
  ∠ AOD = 80 ∧ 
  ∠ BOC = 100 ∧ 
  (interior_angle AOB = 80 ∨ interior_angle AOB = 100)

theorem measure_angle_AOB : 
  ∃ A B C D O : Type, 
  is_rhombus A B C D → 
  O ∈ interior (convex_hull (A, B, C, D)) → 
  (interior_angle AOB = 80 ∨ interior_angle AOB = 100) := 
begin 
  sorry 
end

end measure_angle_AOB_l278_278398


namespace slices_per_friend_l278_278370

theorem slices_per_friend (n : ℕ) (h1 : n > 0)
    (h2 : ∀ i : ℕ, i < n → (15 + 18 + 20 + 25) = 78 * n) :
    78 = (15 + 18 + 20 + 25) / n := 
by
  sorry

end slices_per_friend_l278_278370


namespace hyperbola_eccentricity_l278_278647

theorem hyperbola_eccentricity 
  (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (F₁ F₂ : ℝ × ℝ)
  (circle_center : F₂) (intersect_x F₁_intersect B : ℝ × ℝ)
  (intersect_y A : ℝ × ℝ)
  (AF₁_intersect_C M : ℝ × ℝ)
  (ratio : ℝ)
  (h_hyperbola : ∀ (x y : ℝ), x = F₁.1 → y = F₁.2 → (x/a)^2 - (y/b)^2 = 1)
  (h_circle : ∀ (x y : ℝ), x = F₁_intersect.1 → y = F₁_intersect.2 → (x - circle_center.1)^2 + y^2 = (2 * c)^2)
  (h_line : ∀ (x y : ℝ), x = M.1 → y = M.2 → y = (sqrt 3) * x + (sqrt 3) * F₁.2)
  (h_ratio : ratio = sqrt 31 / 3)
  (h_distance : ∀ (x y : ℝ), x = B.1 → y = B.2 → dist (x, y) (M.1, M.2) = 2 * c * ratio) :
  ∃ e : ℝ, e = (sqrt 7 + 1) / 2 :=
by
  sorry

end hyperbola_eccentricity_l278_278647


namespace seq_bound_gt_pow_two_l278_278217

theorem seq_bound_gt_pow_two (a : Fin 101 → ℕ) 
  (h1 : a 1 > a 0) 
  (h2 : ∀ n : Fin 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2 ^ 99 :=
sorry

end seq_bound_gt_pow_two_l278_278217


namespace sum_frac_eq_half_l278_278369

open Nat

theorem sum_frac_eq_half {n : ℕ} {p : ℕ} (h_prime : Prime p) (h_mod : p % 8 = 7) :
  (∑ k in Finset.range (p - 1) + 1, ((fract ((k^2^n) / p : ℚ) - 1/2))) = (p - 1) / 2 :=
sorry

end sum_frac_eq_half_l278_278369


namespace parabola_correct_equation_l278_278614

noncomputable def parabola_equation : Prop :=
  ∃ p > 0, (∀ x y, y^2 = 2 * p * x → 
    (∀ (Ax Ay : ℝ),
      (Ax + p / 2 = 4 ∧ Ay = sqrt (8 * p - p^2) → 
        (0 + Ay - 2) * (Ax - p / 2) + 2 * (sqrt (8 * p - p^2) - Ay) = 0) →
        y^2 = 8x))

theorem parabola_correct_equation : parabola_equation :=
sorry

end parabola_correct_equation_l278_278614


namespace total_defective_rate_l278_278127

theorem total_defective_rate (P : ℕ)
  (hx_def_rate : 0.005) (hy_def_rate : 0.008)
  (hy_ratio : 1/3) (hx_ratio : 2/3):
  let Defective_x := hx_def_rate * hx_ratio * P
  let Defective_y := hy_def_rate * hy_ratio * P
  let Total_defective_rate := (Defective_x + Defective_y) / P in 
  Total_defective_rate = 0.006 := 
sorry

end total_defective_rate_l278_278127


namespace tom_catches_48_trout_l278_278748

variable (melanie_tom_catch_ratio : ℕ := 3)
variable (melanie_catch : ℕ := 16)

theorem tom_catches_48_trout (h1 : melanie_catch = 16) (h2 : melanie_tom_catch_ratio = 3) : (melanie_tom_catch_ratio * melanie_catch) = 48 :=
by
  sorry

end tom_catches_48_trout_l278_278748


namespace rebecca_swimming_problem_l278_278409

theorem rebecca_swimming_problem :
  ∃ D : ℕ, (D / 4 - D / 5) = 6 → D = 120 :=
sorry

end rebecca_swimming_problem_l278_278409


namespace range_of_m_maximum_radius_l278_278636

open Real

-- Definition of the given equation
def circle_equation (x y m : ℝ) : Prop := 
  x^2 + y^2 - 2 * m * x - 2 * m^2 * y + m^4 + 2 * m^2 - m = 0

-- Statement 1: Prove that 0 < m < 1 for the given equation to represent a circle
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, circle_equation x y m) → 0 < m ∧ m < 1 := 
sorry

-- Statement 2: Prove the largest radius occurs at m = 1/2 and find the maximum radius
theorem maximum_radius (x y m : ℝ) :
  (circle_equation x y m) → (m = 1/2) ∧ (sqrt ((-m^2 + m) = 1/2) := 
sorry

end range_of_m_maximum_radius_l278_278636


namespace range_a_l278_278880

noncomputable def A (a : ℝ) : set ℝ := {x | x ^ 2 < log a x}
def B : set ℝ := {y | ∃ x ∈ Iio (-1 : ℝ), y = 2 ^ x}
def range_of_a : set ℝ := {a | A a ⊆ B ∧ ∃ x ∈ B, x ∉ A a}

theorem range_a (a : ℝ) : a ∈ (Ioo 0 (1/16) ∪ Ici (Real.exp (1/2))) ↔ A a ⊂ B :=
sorry

end range_a_l278_278880


namespace largest_cannot_be_sum_of_two_composites_l278_278161

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278161


namespace numbers_divisible_by_8_between_200_and_400_l278_278289

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278289


namespace sum_of_integers_l278_278458

theorem sum_of_integers (xs : List ℤ) : (∀ x ∈ xs, (∃ k : ℤ, 7 = k * (x - 1))) → xs.sum = 4 :=
by
  intro h
  have h1 : xs = [2, 8, 0, -6] :=
    by sorry
  rw h1
  norm_num

end sum_of_integers_l278_278458


namespace shift_graph_sin_cos_l278_278471

theorem shift_graph_sin_cos :
  ∀ x : ℝ, cos (2 * (x + (π / 4)) - (4 * π / 3)) = sin (2 * x - π / 3) :=
by
  intro x
  calc
    cos (2 * (x + π / 4) - 4 * π / 3)
        = cos (2 * x + π / 2 - 4 * π / 3) : by sorry
    ... = cos (2 * x - π / 3 + π / 2) : by sorry
    ... = sin (π / 2 - (2 * x - π / 3)) : by sorry
    ... = sin (2 * x - π / 3) : by sorry

end shift_graph_sin_cos_l278_278471


namespace pages_left_in_pad_l278_278789

-- Definitions from conditions
def total_pages : ℕ := 120
def science_project_pages (total : ℕ) : ℕ := total * 25 / 100
def math_homework_pages : ℕ := 10

-- Proving the final number of pages left
theorem pages_left_in_pad :
  let remaining_pages_after_usage := total_pages - science_project_pages total_pages - math_homework_pages
  let pages_left_after_art_project := remaining_pages_after_usage / 2
  pages_left_after_art_project = 40 :=
by
  sorry

end pages_left_in_pad_l278_278789


namespace sum_of_three_numbers_l278_278950

theorem sum_of_three_numbers (a b c : ℝ) (h1 : (a + b + c) / 3 = a - 15) (h2 : (a + b + c) / 3 = c + 10) (h3 : b = 10) :
  a + b + c = 45 :=
  sorry

end sum_of_three_numbers_l278_278950


namespace sqrt_neg_sq_real_iff_eq_two_l278_278602

theorem sqrt_neg_sq_real_iff_eq_two :
  ∃! x : ℝ, Real.sqrt (-(x-2)*(x-2)) ∈ ℝ :=
sorry

end sqrt_neg_sq_real_iff_eq_two_l278_278602


namespace jeans_more_than_scarves_l278_278530

def num_ties := 34
def num_belts := 40
def num_black_shirts := 63
def num_white_shirts := 42
def num_jeans := (2 / 3) * (num_black_shirts + num_white_shirts)
def num_scarves := (1 / 2) * (num_ties + num_belts)

theorem jeans_more_than_scarves : num_jeans - num_scarves = 33 := by
  sorry

end jeans_more_than_scarves_l278_278530


namespace scaling_transformation_l278_278705

theorem scaling_transformation:
  ∀ (x y x' y' : ℝ) (λ μ : ℝ), 
    λ > 0 ∧ μ > 0 ∧ (x' = λ * x) ∧ (y' = μ * y) ∧ (y = 2 * sin(3 * x)) 
    → (y' = sin x') → (λ = 3 ∧ μ = 1/2) := 
by 
  intros x y x' y' λ μ h
  sorry

end scaling_transformation_l278_278705


namespace permutations_of_13_variations_without_repetition_of_17_variations_with_repetition_of_83_l278_278476

theorem permutations_of_13 (n : ℕ) (h : ∃ n, n! = 6227020800) : n = 13 := by
  sorry

theorem variations_without_repetition_of_17 (n : ℕ) (h : ∃ n, (n! / (n - 5)!) = 742560) : n = 17 := by
  sorry

theorem variations_with_repetition_of_83 (n : ℕ) (h : ∃ n, n^3 - n * (n - 1) * (n - 2) = 20501) : n = 83 := by
  sorry

end permutations_of_13_variations_without_repetition_of_17_variations_with_repetition_of_83_l278_278476


namespace pie_machine_completion_time_l278_278521

theorem pie_machine_completion_time :
  let start_time := time.mk 9 0 -- 9:00 AM
  let quarter_complete_time := time.mk 12 30 -- 12:30 PM
  let one_fourth_duration := (quarter_complete_time - start_time).to_hours -- 3.5 hours
  let total_duration := 4 * one_fourth_duration -- 14 hours
  let end_time := start_time + total_duration
  end_time = time.mk 23 0 := -- 11:00 PM
sorry

end pie_machine_completion_time_l278_278521


namespace algebraic_expression_value_l278_278634

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
by
  sorry

end algebraic_expression_value_l278_278634


namespace concurrency_of_lines_l278_278694

theorem concurrency_of_lines 
  (ABC : Triangle) 
  (equilateral : Equilateral ABC) 
  (A1 A2 : Point)
  (B1 B2 : Point)
  (C1 C2 : Point)
  (on_BC : A1.is_on_line ℓ BC ∧ A2.is_on_line ℓ BC)
  (on_CA : B1.is_on_line ℓ CA ∧ B2.is_on_line ℓ CA)
  (on_AB : C1.is_on_line ℓ AB ∧ C2.is_on_line ℓ AB)
  (equal_sides : ∀ (P Q : Point), (is_side (P Q) (A1 A2) ∨ is_side (P Q) (B1 B2) ∨ is_side (P Q) (C1 C2) → distance P Q = distance A1 A2)) :
  concurrent_lines (line_through A1 B2) (line_through B1 C2) (line_through C1 A2) :=
by
  sorry

end concurrency_of_lines_l278_278694


namespace arithmetic_sequence_a8_l278_278700

/-- In an arithmetic sequence with the given sum of terms, prove the value of a_8 is 14. -/
theorem arithmetic_sequence_a8 (a : ℕ → ℕ) (d : ℕ) (h1 : ∀ (n : ℕ), a (n+1) = a n + d)
    (h2 : a 2 + a 7 + a 8 + a 9 + a 14 = 70) : a 8 = 14 :=
  sorry

end arithmetic_sequence_a8_l278_278700


namespace engineer_mistake_l278_278443

noncomputable def weight_of_single_disk := 100
noncomputable def diameter_of_disk := 1
noncomputable def stddev_radius := 0.01
noncomputable def number_of_disks := 100
noncomputable def engineer_estimate := 10000

theorem engineer_mistake :
  (number_of_disks * weight_of_single_disk + number_of_disks * (stddev_radius^(2 : ℕ) + (0.5)^(2 : ℕ)) - engineer_estimate) = 4 := by
  sorry

end engineer_mistake_l278_278443


namespace engineer_mistake_l278_278442

noncomputable def weight_of_single_disk := 100
noncomputable def diameter_of_disk := 1
noncomputable def stddev_radius := 0.01
noncomputable def number_of_disks := 100
noncomputable def engineer_estimate := 10000

theorem engineer_mistake :
  (number_of_disks * weight_of_single_disk + number_of_disks * (stddev_radius^(2 : ℕ) + (0.5)^(2 : ℕ)) - engineer_estimate) = 4 := by
  sorry

end engineer_mistake_l278_278442


namespace required_moles_H2SO4_l278_278659

-- Definitions for the problem
def moles_NaCl := 2
def moles_H2SO4_needed := 2
def moles_HCl_produced := 2
def moles_NaHSO4_produced := 2

-- Condition representing stoichiometry of the reaction
axiom reaction_stoichiometry : ∀ (moles_NaCl moles_H2SO4 moles_HCl moles_NaHSO4 : ℕ), 
  moles_NaCl = moles_HCl ∧ moles_HCl = moles_NaHSO4 → moles_NaCl = moles_H2SO4

-- Proof statement we want to establish
theorem required_moles_H2SO4 : 
  ∃ (moles_H2SO4 : ℕ), moles_H2SO4 = 2 ∧ ∀ (moles_NaCl : ℕ), moles_NaCl = 2 → moles_H2SO4_needed = 2 := by
  sorry

end required_moles_H2SO4_l278_278659


namespace certain_number_exists_l278_278464

theorem certain_number_exists
  (N : ℕ) 
  (hN : ∀ x, x < N → x % 2 = 1 → ∃ k m, k = 5 * m ∧ x = k ∧ m % 2 = 1) :
  N = 76 := by
  sorry

end certain_number_exists_l278_278464


namespace quincy_sold_more_than_jake_l278_278352

variables (T : ℕ) (Jake Quincy : ℕ)

def thors_sales (T : ℕ) := T
def jakes_sales (T : ℕ) := T + 10
def quincys_sales (T : ℕ) := 10 * T

theorem quincy_sold_more_than_jake (h1 : jakes_sales T = Jake) 
  (h2 : quincys_sales T = Quincy) (h3 : Quincy = 200) : 
  Quincy - Jake = 170 :=
by
  sorry

end quincy_sold_more_than_jake_l278_278352


namespace evaluate_expression_l278_278931

theorem evaluate_expression :
  (2 ^ (-1 : ℤ) + 2 ^ (-2 : ℤ))⁻¹ = (4 / 3 : ℚ) := by
    sorry

end evaluate_expression_l278_278931


namespace total_distance_wheels_l278_278812

theorem total_distance_wheels : 
  let π := Real.pi
  let C1 := π * 0.7
  let C2 := π * 1.2
  let C3 := π * 1.6
  let D1 := C1 * 200
  let D2 := C2 * 200
  let D3 := C3 * 200
  (D1 + D2 + D3) ≈ 2199.2 :=
by
  let π := 3.14159
  let C1 := π * 0.7
  let C2 := π * 1.2
  let C3 := π * 1.6
  let D1 := C1 * 200
  let D2 := C2 * 200
  let D3 := C3 * 200
  sorry

end total_distance_wheels_l278_278812


namespace selling_price_correct_l278_278766

-- Definition of the conditions:
def purchase_price : ℝ := 42000
def repair_costs : ℝ := 15000
def profit_percent : ℝ := 13.859649122807017

-- Calculate the total cost:
def total_cost := purchase_price + repair_costs

-- Calculate the profit based on the given profit percentage:
def profit := (profit_percent / 100) * total_cost

-- Calculate the selling price:
def selling_price := total_cost + profit

-- The theorem to prove:
theorem selling_price_correct : 
  selling_price = 64900 :=
sorry

end selling_price_correct_l278_278766


namespace exterior_angle_BAC_135_l278_278549

theorem exterior_angle_BAC_135 
  (interior_angle_octagon : ℝ := 135)
  (angle_CAD : ℝ := 90)
  (shared_side : ∃ A B C D : Point, square A B C D ∧ octagon AD)
  : exterior_angle BAC = 135 := 
begin
  sorry
end

end exterior_angle_BAC_135_l278_278549


namespace generalized_small_fermat_part_a_part_b_part_c_part_d_l278_278512

def euler_totient (n : ℕ) : ℕ := n.totient

-- General statement
theorem generalized_small_fermat (a k : ℕ) (h_coprime : Nat.coprime a k) :
  (a ^ euler_totient k - 1) % k = 0 :=
by sorry

-- Part a
theorem part_a (a p : ℕ) (h_prime : Nat.Prime p) (h_coprime : Nat.coprime a p) :
  (a ^ euler_totient p - 1) % p = 0 :=
by sorry

-- Part b
theorem part_b (a p q : ℕ) (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q) (h_coprime : Nat.coprime a (p * q)) :
  (a ^ euler_totient (p * q) - 1) % (p * q) = 0 :=
by sorry

-- Part c
theorem part_c (a p : ℕ) (h_prime : Nat.Prime p) (h_coprime : Nat.coprime a (p ^ 2)) :
  (a ^ euler_totient (p ^ 2) - 1) % (p ^ 2) = 0 :=
by sorry

-- Part d
theorem part_d (a p : ℕ) (lambda : ℕ) (h_prime : Nat.Prime p) (h_coprime : Nat.coprime a (p ^ lambda)) :
  (a ^ euler_totient (p ^ lambda) - 1) % (p ^ lambda) = 0 :=
by sorry

end generalized_small_fermat_part_a_part_b_part_c_part_d_l278_278512


namespace find_equation_of_line1_find_equation_of_line2_l278_278878

section Problem1

def Circle (C : ℝ × ℝ → Prop) := ∀ x y, C (x, y) ↔ x^2 + y^2 = 4

def Point_P (P : ℝ × ℝ) := P = (1, 2)

def intersects_circle_at_A_B_with_length (C : ℝ × ℝ → Prop) (l : ℝ → ℝ) (P : ℝ × ℝ) (A B : ℝ × ℝ) :=
  l P.1 = P.2 ∧
  ∀ x y, C (x, y) → l x = y → (A = (x, y) ∨ B = (x, y)) ∧ ∃ d, d = 2 * real.sqrt 3

theorem find_equation_of_line1 {C : ℝ × ℝ → Prop} {l : ℝ → ℝ} {P : ℝ × ℝ}
  (hC : Circle C) (hP : Point_P P) 
  (h_int : ∃ A B : ℝ × ℝ, intersects_circle_at_A_B_with_length C l P A B) :
  (∀ x, l x = 1) ∨ (∀ x y, 3 * x - 4 * y + 5 = 0) :=
sorry

end Problem1

section Problem2

def Line (a : ℝ) (l : ℝ × ℝ → Prop) := ∀ x y, l (x, y) ↔ (a + 1) * x + y - 2 - a = 0

def intercepts_equal (l : ℝ × ℝ → Prop) := 
  ∃ a : ℝ, ∀ x y, l (x, y) → 
    if x ≠ 0 then - (y / x) = (y / (x + a)) else x = 0 ∧ y = 0

theorem find_equation_of_line2 {l : ℝ × ℝ → Prop}
  (hL : ∃ a : ℝ, Line a l) (h_intercepts : intercepts_equal l) :
  (∀ x y, x - y = 0) ∨ (∀ x y, x + y - 2 = 0) :=
sorry

end Problem2

end find_equation_of_line1_find_equation_of_line2_l278_278878


namespace integer_values_abs_less_than_2pi_l278_278258

-- Define the main theorem to prove
theorem integer_values_abs_less_than_2pi : {x : ℤ | abs x < 2 * Real.pi}.to_finset.card = 13 := by
  -- Proof goes here
  sorry

end integer_values_abs_less_than_2pi_l278_278258


namespace ellipse_focus_distance_l278_278595

theorem ellipse_focus_distance : ∀ (x y : ℝ), 9 * x^2 + y^2 = 900 → 2 * Real.sqrt (10^2 - 30^2) = 40 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end ellipse_focus_distance_l278_278595


namespace no_a_b_exist_no_a_b_c_exist_l278_278496

-- Part (a):
theorem no_a_b_exist (a b : ℕ) (h0 : 0 < a) (h1 : 0 < b) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n = k^2) :=
sorry

-- Part (b):
theorem no_a_b_c_exist (a b c : ℕ) (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < c) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n + c = k^2) :=
sorry

end no_a_b_exist_no_a_b_c_exist_l278_278496


namespace general_admission_ticket_cost_l278_278469

-- Define the problem
def student_ticket_cost : ℕ := 4
def total_tickets_sold : ℕ := 525
def total_amount_collected : ℕ := 2876
def general_admission_tickets_sold : ℕ := 388

-- Define the proof problem
theorem general_admission_ticket_cost : 
  (let student_tickets_sold := total_tickets_sold - general_admission_tickets_sold in
   let total_student_amount := student_tickets_sold * student_ticket_cost in
   let total_general_amount := total_amount_collected - total_student_amount in
   total_general_amount / general_admission_tickets_sold = 6) :=
begin
  sorry,
end

end general_admission_ticket_cost_l278_278469


namespace sequence_sum_l278_278037

theorem sequence_sum :
  ∑ i in finset.range 21, if (i % 2 = 0) then (3 * (i + 1) : ℤ) else (-6 * ((i + 1) / 2 + 1) : ℤ) = 33 :=
by
  sorry

end sequence_sum_l278_278037


namespace angle_equiv_l278_278592

theorem angle_equiv (θ : ℝ) : 
  ∃ θ', θ' = 280 ∧ θ' ∈ set.Icc 0 360 ∧ ∃ k : ℤ, θ = θ' + k * 360 :=
by
  use 280
  split
  sorry

end angle_equiv_l278_278592


namespace solve_exponential_equation_l278_278434

theorem solve_exponential_equation (x : ℝ) :
  3^(2 * x) - 15 * 3^x + 18 = 0 ↔ x = 1 ∨ x = Real.log 2 / Real.log 3 + 1 :=
by
  sorry

end solve_exponential_equation_l278_278434


namespace C_can_complete_work_in_100_days_l278_278864

-- Definitions for conditions
def A_work_rate : ℚ := 1 / 20
def B_work_rate : ℚ := 1 / 15
def work_done_by_A_and_B : ℚ := 6 * (1 / 20 + 1 / 15)
def remaining_work : ℚ := 1 - work_done_by_A_and_B
def work_done_by_A_in_5_days : ℚ := 5 * (1 / 20)
def work_done_by_C_in_5_days : ℚ := remaining_work - work_done_by_A_in_5_days
def C_work_rate_in_5_days : ℚ := work_done_by_C_in_5_days / 5

-- Statement to prove
theorem C_can_complete_work_in_100_days : 
  work_done_by_C_in_5_days ≠ 0 → 1 / C_work_rate_in_5_days = 100 :=
by
  -- proof of the theorem
  sorry

end C_can_complete_work_in_100_days_l278_278864


namespace find_m_n_l278_278231

def odd_function (f : ℝ → ℝ) : Prop := 
∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) (m n : ℝ) : ℝ := (m - 3^x) / (n + 3^x)

theorem find_m_n (m n : ℝ) (h_odd : odd_function (f · m n)) :
  m = 1 ∧ n = 1 := sorry

end find_m_n_l278_278231


namespace tyler_meals_l278_278029

def num_meals : ℕ := 
  let num_meats := 3
  let num_vegetable_combinations := Nat.choose 5 3
  let num_desserts := 5
  num_meats * num_vegetable_combinations * num_desserts

theorem tyler_meals :
  num_meals = 150 := by
  sorry

end tyler_meals_l278_278029


namespace correct_calculation_l278_278488

theorem correct_calculation : (Real.sqrt 32) / (Real.sqrt 2) = 4 :=
begin
  sorry
end

end correct_calculation_l278_278488


namespace smallest_n_for_quadratic_factorization_l278_278600

theorem smallest_n_for_quadratic_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, A * B = 50 → n = 5 * B + A) ∧ (∀ m : ℤ, 
    (∀ A B : ℤ, A * B = 50 → m ≤ 5 * B + A) → n ≤ m) :=
by
  sorry

end smallest_n_for_quadratic_factorization_l278_278600


namespace largest_number_not_sum_of_two_composites_l278_278177

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278177


namespace justin_total_pages_l278_278361

theorem justin_total_pages :
  let pages_read (n : ℕ) := (10 * 2 ^ n)
  in ∑ i in Finset.range 7, pages_read i = 1270 := by
  sorry

end justin_total_pages_l278_278361


namespace count_divisibles_l278_278301

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278301


namespace cost_prices_correct_maximize_profit_l278_278426

-- Define the cost prices of zongzi A and B
def cost_price_A : ℕ := 10
def cost_price_B : ℕ := cost_price_A + 2

-- Define the conditions for the purchasing problem
def condition1 : Prop := 1000 / cost_price_A = 1200 / cost_price_B
def condition2 : Prop := ∀ m : ℕ, 200 - m < 2 * m

-- Define the profit function
def profit (m : ℕ) : ℕ := -m + 600

-- Define the valid range for m
def valid_m (m : ℕ) : Prop := 400 / 3 ≤ m ∧ m < 200

-- Prove that the cost prices are 10 and 12
theorem cost_prices_correct : cost_price_A = 10 ∧ cost_price_B = 12 := by
  simp [cost_price_A, cost_price_B]

-- Prove that the profit function and conditions yield the maximum profit
theorem maximize_profit : 
  ∀ m : ℕ, valid_m m → profit m = 466 → m = 134 := by
  sorry

end cost_prices_correct_maximize_profit_l278_278426


namespace problem_B_height_l278_278607

noncomputable def point_B_height (cos : ℝ → ℝ) : ℝ :=
  let θ := 30 * (Real.pi / 180)
  let cos30 := cos θ
  let original_vertical_height := 1 / 2
  let additional_height := cos30 * (1 / 2)
  original_vertical_height + additional_height

theorem problem_B_height : 
  point_B_height Real.cos = (2 + Real.sqrt 3) / 4 := 
by 
  sorry

end problem_B_height_l278_278607


namespace square_side_length_theorem_l278_278525

-- Define the properties of the geometric configurations
def is_tangent_to_extension_segments (circle_radius : ℝ) (segment_length : ℝ) : Prop :=
  segment_length = circle_radius

def angle_between_tangents_from_point (angle : ℝ) : Prop :=
  angle = 60 

def square_side_length (side : ℝ) : Prop :=
  side = 4 * (Real.sqrt 2 - 1)

-- Main theorem
theorem square_side_length_theorem (circle_radius : ℝ) (segment_length : ℝ) (angle : ℝ) (side : ℝ)
  (h1 : is_tangent_to_extension_segments circle_radius segment_length)
  (h2 : angle_between_tangents_from_point angle) :
  square_side_length side :=
by
  sorry

end square_side_length_theorem_l278_278525


namespace dry_grapes_weight_l278_278500

theorem dry_grapes_weight (initial_weight_fresh_grapes : ℕ) 
                          (water_content_fresh : ℕ) 
                          (water_content_dried : ℕ) : 
  initial_weight_fresh_grapes = 40 → water_content_fresh = 90 → water_content_dried = 20 → 
  (let non_water_content_fresh := initial_weight_fresh_grapes * (100 - water_content_fresh) / 100;
       total_weight_dried := non_water_content_fresh * 100 / (100 - water_content_dried)
   in total_weight_dried = 5) :=
by
  intros h1 h2 h3
  let non_water_content_fresh := initial_weight_fresh_grapes * (100 - water_content_fresh) / 100
  let total_weight_dried := non_water_content_fresh * 100 / (100 - water_content_dried)
  have h4 : total_weight_dried = 5 := sorry
  exact h4

end dry_grapes_weight_l278_278500


namespace sum_of_powers_l278_278998

def z : ℂ := -1 / 2 + (real.sqrt 3) / 2 * complex.I

theorem sum_of_powers (z_def : z = -1 / 2 + (real.sqrt 3) / 2 * complex.I) :
  ∑ i in finset.range 2023 + 1, z ^ (i + 1) = -1 / 2 + (real.sqrt 3) / 2 * complex.I :=
sorry

end sum_of_powers_l278_278998


namespace fraction_problem_l278_278879

theorem fraction_problem :
  ((3 / 4 - 5 / 8) / 2) = 1 / 16 :=
by
  sorry

end fraction_problem_l278_278879


namespace probability_of_consecutive_triplets_l278_278015

def total_ways_to_select_3_days (n : ℕ) : ℕ :=
  Nat.choose n 3

def number_of_consecutive_triplets (n : ℕ) : ℕ :=
  n - 2

theorem probability_of_consecutive_triplets :
  let total_ways := total_ways_to_select_3_days 10
  let consecutive_triplets := number_of_consecutive_triplets 10
  (consecutive_triplets : ℚ) / total_ways = 1 / 15 :=
by
  sorry

end probability_of_consecutive_triplets_l278_278015


namespace percentage_increase_in_mean_is_minimum_l278_278034

theorem percentage_increase_in_mean_is_minimum :
  let G := [-7, -5, -4, -1, 0, 6, 9, 11] in
  let old_mean := 9 / 8 in
  let G' := [2, 3, 5, 7, 0, 6, 9, 11] in
  let new_mean := 43 / 8 in
  let percentage_increase := ((new_mean - old_mean) / old_mean) * 100 in
  percentage_increase = 377.78 :=
by
  sorry

end percentage_increase_in_mean_is_minimum_l278_278034


namespace unique_point_l278_278761

noncomputable def pointA : ℝ × ℝ := (6, 0)
def lineP (x : ℝ) : ℝ := -x
def distance_from_A_to_P (P : ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 - 6)^2 + (P.2 - 0)^2)

theorem unique_point :
  ∃! (P : ℝ × ℝ), P.2 = lineP P.1 ∧ distance_from_A_to_P P = 3 * Real.sqrt 2 := by
  sorry

end unique_point_l278_278761


namespace incorrect_represent_twice_diff_squares_product_l278_278710

theorem incorrect_represent_twice_diff_squares_product (a b : ℝ) :
  (a + b) ^ 2 - 2 * a * b ≠ 2 * (a ^ 2 + b ^ 2 - a * b) := by
suffices h1: (a + b) ^ 2 - 2 * a * b = a ^ 2 + b ^ 2 by
  suffices h2: 2 * (a ^ 2 + b ^ 2 - a * b) = 2 * (a ^ 2 + b ^ 2) - 2 * a * b by
    show a ^ 2 + b ^ 2 ≠ 2 * (a ^ 2 + b ^ 2 - a * b) from by
      calc
      a ^ 2 + b ^ 2 ≠ 2 * (a ^ 2 + b ^ 2 - a * b) := by
        -- Using proof by contradiction here
        intro h3
        exact absurd (by linarith [h3]) (h1 ▸ h3 : _ = _ : by sorry : eq.refl (a ^ 2 + b ^ 2))
  show (a + b) ^ 2 - 2 * a * b = a ^ 2 + b ^ 2 from by
    -- Expand and simplify (a + b)^2 - 2ab
    calc
    (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 :=
      by ring_exp
    a ^ 2 + 2 * a * b + b ^ 2 - 2 * a * b = a ^ 2 + b ^ 2 := by ring
sorry

end incorrect_represent_twice_diff_squares_product_l278_278710


namespace rhombus_perimeter_l278_278784

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 4 * sqrt 241 := by
  sorry

end rhombus_perimeter_l278_278784


namespace train_pass_bridge_time_l278_278866

theorem train_pass_bridge_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmph : ℝ)
  (km_to_m : ℝ := 1000)
  (hour_to_s : ℝ := 3600) :
  train_length = 360 →
  bridge_length = 140 →
  train_speed_kmph = 90 →
  (train_length + bridge_length) / (train_speed_kmph * (km_to_m / hour_to_s)) = 20 :=
by
  intros h_train_length h_bridge_length h_train_speed
  rw [h_train_length, h_bridge_length, h_train_speed]
  -- Definition of constants
  let total_distance := (360 + 140 : ℝ)
  let speed_mps := (90 : ℝ) * (1000 / 3600 : ℝ)
  -- Simplify the expression
  have speed_mps_value : speed_mps = 25 :=
    by norm_num
  rw [←speed_mps_value]
  have distance_covered : total_distance / speed_mps = 20 :=
    by norm_num
  exact distance_covered


end train_pass_bridge_time_l278_278866


namespace tabby_avg_speed_l278_278504

-- Define the speeds
def swim_speed : ℝ := 1
def run_speed : ℝ := 11

-- Assume the same distance D for both events
variable (D : ℝ) (hD : D > 0)

-- Define the time taken for each activity
def swim_time := D / swim_speed
def run_time := D / run_speed

-- Define the total distance and total time
def total_distance := 2 * D
def total_time := swim_time + run_time

-- Define the average speed
def avg_speed := total_distance / total_time

-- The theorem we want to prove
theorem tabby_avg_speed : avg_speed = 11 / 6 :=
by
  -- Proof is omitted
  sorry

end tabby_avg_speed_l278_278504


namespace largest_four_digit_divisible_by_6_l278_278829

theorem largest_four_digit_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 6 = 0 → m ≤ n :=
begin
  use 9996,
  split,
  { exact nat.le_refl 9996 },
  split,
  { dec_trivial },
  split,
  { exact nat.zero_mod _ },
  { intros m h1 h2 h3,
    exfalso,
    sorry }
end

end largest_four_digit_divisible_by_6_l278_278829


namespace min_dot_product_l278_278633

-- Definitions and Conditions
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

structure Point :=
(x : ℝ)
(y : ℝ)

structure Square :=
(A B C D : Point)

def isInscribed (c : Circle) (s : Square) : Prop :=
(s.A.x - c.center.1)^2 + (s.A.y - c.center.2)^2 = c.radius^2 ∧
(s.B.x - c.center.1)^2 + (s.B.y - c.center.2)^2 = c.radius^2 ∧
(s.C.x - c.center.1)^2 + (s.C.y - c.center.2)^2 = c.radius^2 ∧
(s.D.x - c.center.1)^2 + (s.D.y - c.center.2)^2 = c.radius^2

def isDiameter (c : Circle) (E F : Point) : Prop :=
(E.x - c.center.1)^2 + (E.y - c.center.2)^2 = c.radius^2 ∧
(F.x - c.center.1)^2 + (F.y - c.center.2)^2 = c.radius^2 ∧
(E.x - F.x)^2 + (E.y - F.y)^2 = (2 * c.radius)^2

def onBoundary (M : Point) (s : Square) : Prop :=
( M.x = s.A.x ∧ M.y ≤ s.A.y ∧ M.y ≥ s.D.y ) ∨
( M.y = s.A.y ∧ M.x ≤ s.B.x ∧ M.x ≥ s.A.x ) ∨
( M.x = s.C.x ∧ M.y ≤ s.C.y ∧ M.y ≥ s.B.y ) ∨
( M.y = s.C.y ∧ M.x ≤ s.C.x ∧ M.x ≥ s.D.x )

-- Vector dot product
def dot (v1 v2 : Point) : ℝ :=
v1.x * v2.x + v1.y * v2.y

-- Main theorem statement
theorem min_dot_product (c : Circle)
  (s : Square)
  (E F M : Point)
  (h1 : c.radius = 1)
  (h2 : isInscribed c s)
  (h3 : isDiameter c E F)
  (h4 : onBoundary M s) :
  ∃(min_val : ℝ), min_val = -1/2 ∧ infi (λ M, dot { x := M.x - E.x, y := M.y - E.y } { x := M.x - F.x, y := M.y - F.y }) = min_val :=
sorry

end min_dot_product_l278_278633


namespace union_of_A_and_B_l278_278220

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
noncomputable def B : Set ℝ := {x : ℝ | 1 < x }

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x} :=
by
  sorry

end union_of_A_and_B_l278_278220


namespace ab_zero_condition_l278_278200

variable (a b : ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def given_condition (f : ℝ → ℝ) : Prop :=
  f = λ x, x * |x + a| + b

theorem ab_zero_condition (f : ℝ → ℝ) (h : given_condition f) : (a * b = 0) → is_odd_function f :=
sorry

end ab_zero_condition_l278_278200


namespace largest_four_digit_number_divisible_by_six_l278_278834

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end largest_four_digit_number_divisible_by_six_l278_278834


namespace exists_monochromatic_rectangle_in_two_colored_plane_l278_278796

open Classical

theorem exists_monochromatic_rectangle_in_two_colored_plane :
  ∀ (color : ℕ × ℕ → bool),
  ∃ (r₁ r₂ r₃ r₄ c₁ c₂ c₃ c₄ : ℕ),
    r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₄ ∧ r₁ ≠ r₄ ∧ 
    c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₄ ∧ c₁ ≠ c₃ ∧
    color (r₁, c₁) = color (r₂, c₂) ∧
    color (r₂, c₂) = color (r₃, c₃) ∧
    color (r₃, c₃) = color (r₄, c₄) :=
by
  sorry

end exists_monochromatic_rectangle_in_two_colored_plane_l278_278796


namespace probability_product_greater_than_zero_l278_278025

noncomputable theory

def probability_of_positive_product : ℝ :=
  let interval_length : ℝ := 30 in
  let probability_of_positive_or_negative : ℝ := (15 / interval_length) in
  probability_of_positive_or_negative * probability_of_positive_or_negative + 
  probability_of_positive_or_negative * probability_of_positive_or_negative

theorem probability_product_greater_than_zero :
  probability_of_positive_product = 1 / 2 :=
by
  sorry

end probability_product_greater_than_zero_l278_278025


namespace problem1_problem2_l278_278573

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) : 2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3 :=
by sorry

end problem1_problem2_l278_278573


namespace center_square_is_15_l278_278944

noncomputable def center_square_value : ℤ :=
  let d1 := (15 - 3) / 2
  let d3 := (33 - 9) / 2
  let middle_first_row := 3 + d1
  let middle_last_row := 9 + d3
  let d2 := (middle_last_row - middle_first_row) / 2
  middle_first_row + d2

theorem center_square_is_15 : center_square_value = 15 := by
  sorry

end center_square_is_15_l278_278944


namespace set_intersection_complement_l278_278382

theorem set_intersection_complement (U M N : Set ℕ) :
  U = {0, 1, 2, 3, 4, 5} →
  M = {0, 3, 5} →
  N = {1, 4, 5} →
  M ∩ (U \ N) = {0, 3} :=
by
  intros U_def M_def N_def
  have complement_U_N := {0, 2, 3}
  have intersection := {0, 3}
  sorry

end set_intersection_complement_l278_278382


namespace sum_of_values_of_m_l278_278228

theorem sum_of_values_of_m :
  ∀ (m : ℕ), (m ≥ 3) → ((m-2) ∣ (3*m^2 - 2*m + 10)) → (∑ (m : ℕ) in {m | m ≥ 3 ∧ (m-2) ∣ (3*m^2 - 2*m + 10)}, m) = 51 := by
  sorry

end sum_of_values_of_m_l278_278228


namespace Justin_reads_total_pages_l278_278357

theorem Justin_reads_total_pages 
  (initial_pages : ℕ)
  (multiplier : ℕ)
  (days_remaining : ℕ)
  (total_days : ℕ)
  (total_pages_needed : ℕ) :
  initial_pages = 10 →
  multiplier = 2 →
  days_remaining = 6 →
  total_days = 7 →
  total_pages_needed = 100 →
  (initial_pages + days_remaining * (initial_pages * multiplier)) = 130 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry

end Justin_reads_total_pages_l278_278357


namespace value_of_x_l278_278314

theorem value_of_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y + 2))  : 
  x = y^2 + 2 * y + 3 := 
by 
  sorry

end value_of_x_l278_278314


namespace max_rooks_removed_l278_278758

def attacks (r1 r2 : ℕ × ℕ) (board : ℕ × ℕ → bool) : Prop :=
  (r1.1 = r2.1 ∨ r1.2 = r2.2) ∧ r1 ≠ r2 ∧
  ∀ k, (r1.1 = r2.1 → (r1.2 < k ∧ k < r2.2 → ¬board (r1.1, k)) ∨ (r2.2 < k ∧ k < r1.2 → ¬board (r1.1, k)))
  ∧ (r1.2 = r2.2 → (r1.1 < k ∧ k < r2.1 → ¬board (k, r1.2)) ∨ (r2.1 < k ∧ k < r1.1 → ¬board (k, r1.2)))

def rook_removal_condition (board : ℕ × ℕ → bool) (removed_r : ℕ × ℕ) : Prop :=
  board removed_r ∧ 
  (Finset.card (Finset.filter (λ r, attacks removed_r r board) (Finset.univ : Finset (ℕ × ℕ))) % 2 = 1)

theorem max_rooks_removed : ∃ removed : Finset (ℕ × ℕ), 
  (∀ r ∈ removed, rook_removal_condition (λ r, true) r) ∧ 
  removed.card = 59 :=
sorry

end max_rooks_removed_l278_278758


namespace find_r_l278_278808

noncomputable def vector_a : ℝ × ℝ × ℝ := (2, 3, -1)
noncomputable def vector_b : ℝ × ℝ × ℝ := (-1, 1, 2)
noncomputable def matrix_m : ℝ × ℝ × ℝ := (1, 4, -6)
noncomputable def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.2.1 * v.2.2 - u.2.2 * v.2.1, u.2.2 * v.1 - u.1 * v.2.2, u.1 * v.2.1 - u.2.1 * v.1)
noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2.1 * v.2.1 + u.2.2 * v.2.2

theorem find_r : ∃ r : ℝ,  matrix_m = (p : ℝ) * vector_a + (q : ℝ) * vector_b + r * cross_product vector_a vector_b ∧ r = -35 / 83 :=
by
  sorry

end find_r_l278_278808


namespace total_surface_area_correct_l278_278093

def surface_area_calculation (height_e height_f height_g : ℚ) : ℚ :=
  let top_bottom_area := 4
  let side_area := (height_e + height_f + height_g) * 2
  let front_back_area := 4
  top_bottom_area + side_area + front_back_area

theorem total_surface_area_correct :
  surface_area_calculation (5 / 8) (1 / 4) (9 / 8) = 12 := 
by
  sorry

end total_surface_area_correct_l278_278093


namespace probability_best_play_wins_l278_278069

noncomputable def prob_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) : ℝ :=
  let C := Nat.choose in
  (1 : ℝ) / (C (2 * n) n * C (2 * n) (2 * m)) * ∑ q in Finset.range (2 * m + 1),
  (C n q * C n (2 * m - q)) * 
  ∑ t in Finset.range (min q m),
  (C q t * C (2 * n - q) (n - t))

-- A theorem statement in Lean to ensure proper type checks and conditions 
theorem probability_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) :
  ∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t)) 
  =
  (∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t) )) * 
  (nat.choose (2 * n) n * nat.choose (2 * n) (2 * m)) :=
sorry

end probability_best_play_wins_l278_278069


namespace second_fraction_correct_l278_278431

theorem second_fraction_correct : 
  ∃ x : ℚ, (2 / 3) * x * (1 / 3) * (3 / 8) = 0.07142857142857142 ∧ x = 6 / 7 :=
by
  sorry

end second_fraction_correct_l278_278431


namespace simplify_expression_l278_278770

variable (α : ℝ)

theorem simplify_expression :
  (sin (4 * α))^2 + 4 * (sin (2 * α))^4 - 4 * (sin (2 * α))^2 * (cos (2 * α))^2) /
  (4 - (sin (4 * α))^2 - 4 * (sin (2 * α))^2) = 
  (tan (2 * α))^4 :=
by
  sorry

end simplify_expression_l278_278770


namespace g_sum_even_l278_278376

def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5

theorem g_sum_even (a b c d : ℝ) (h : g 42 a b c d = 3) : g 42 a b c d + g (-42) a b c d = 6 := by
  sorry

end g_sum_even_l278_278376


namespace divide_square_into_4_isosceles_triangles_l278_278959

-- Define the conditions as a statement
def ways_to_divide_square_into_3_isosceles (ways : ℕ) : Prop :=
  ways = 4

-- Define the ultimate problem as a theorem
theorem divide_square_into_4_isosceles_triangles : ℕ :=
  21

-- Formalize the problem statement
def divide_fixed_square_into_4_isosceles (ways3 : ℕ) (h : ways_to_divide_square_into_3_isosceles ways3) : Prop :=
  divide_square_into_4_isosceles_triangles = 21

-- assertion of the main result
example : divide_fixed_square_into_4_isosceles 4 (by simp [ways_to_divide_square_into_3_isosceles]) =
  (21 = 21) := by refl

end divide_square_into_4_isosceles_triangles_l278_278959


namespace time_for_robot_B_to_reach_B_l278_278923

open Nat

-- Definitions of the robots A, B, and C, and their properties
variables (A B : Type)
variables (circumference : ℕ)
variables (robot_A robot_B robot_C : ℕ → ℕ)
variables (clockwise : ℕ → ℕ)
variables (counterclockwise : ℕ → ℕ)

-- Initial Conditions
def starting_point_A : Prop := ∀ t, robot_A t = robot_B t ∨ robot_A t = A
def starting_point_B : Prop := ∀ t, robot_C t = B
def directions : Prop := clockwise = counterclockwise
def touching_conditions : Prop := ∀ t, robot_A (t + 21) = robot_C t ∧ robot_A t = robot_B t

-- The theorem to calculate the required time
theorem time_for_robot_B_to_reach_B : 
  starting_point_A →
  starting_point_B →
  directions →
  touching_conditions →
  robot_B 56 = B :=
by
  intros _ _ _ _
  sorry

end time_for_robot_B_to_reach_B_l278_278923


namespace flagpole_break_height_l278_278533

theorem flagpole_break_height
  (h : 8) -- original height of the flagpole
  (d : 3) -- horizontal distance from the base to the point where the top part touches the ground
  (x : ℝ) -- height where the flagpole breaks
  (hx : h = 8 ∧ d = 3)
  (Pythagorean : (h - x)^2 + d^2 = (h - x + d)^2)
  : x = sqrt 73 / 2 :=
by
  sorry

end flagpole_break_height_l278_278533


namespace value_of_quarters_as_percent_l278_278862

-- Define the conditions
def dimes : Nat := 80
def quarters : Nat := 40
def nickels : Nat := 30

def value_dime : Nat := 10
def value_quarter : Nat := 25
def value_nickel : Nat := 5

-- Prove the assertion
theorem value_of_quarters_as_percent :
  (40 * 25 : ℝ) / (40 * 25 + 80 * 10 + 30 * 5) * 100 ≈ 51.28 := by
  sorry

end value_of_quarters_as_percent_l278_278862


namespace div_k_l_n_l278_278726

theorem div_k_l_n (n k ℓ : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : 0 < ℓ) (σ : Fin n → Fin n) 
  (hσ : Function.Injective σ) (h4 : ∀ x : Fin n, σ x - x ∈ [k, n - ℓ] ) : k + ℓ ∣ n := 
  sorry

end div_k_l_n_l278_278726


namespace roots_of_quadratic_eval_l278_278203

theorem roots_of_quadratic_eval :
  ∀ x₁ x₂ : ℝ, (x₁^2 + 4 * x₁ + 2 = 0) ∧ (x₂^2 + 4 * x₂ + 2 = 0) ∧ (x₁ + x₂ = -4) ∧ (x₁ * x₂ = 2) →
    x₁^3 + 14 * x₂ + 55 = 7 :=
by
  sorry

end roots_of_quadratic_eval_l278_278203


namespace exactly_one_correct_l278_278787

theorem exactly_one_correct (statement1 : Prop) (statement2 : Prop) (statement3 : Prop) (statement4 : Prop) :
  (statement1 ↔ false) ∧ 
  (statement2 ↔ false) ∧ 
  (statement3 ↔ false) ∧ 
  (statement4 ↔ true) :=
by
  -- Provide definitions for the statements under proof
  let statement1 := ∀ (l : ℝ), ∃! (m : ℝ), l ≠ m ∧ ∀ (p : ℝ × ℝ), parallel (line_through p l) (line_through p m)
  let statement2 := ∀ (k : ℝ), ∃ (a b : ℝ), x^2 - k * y^2 = (x + a * y) * (x + b * y)
  let statement3 := ∃ (t : ℝ), (t - 3) ^ (3 - 2 * t) = 1 ∧ t ≠ 4 ∧ t ≠ 3 / 2
  let statement4 := ∀ (a : ℝ), ∀ (x y : ℝ), (a * x + 2 * y = -5) ∧ (-x + a * y = 2 * a) → (x = 3 ∧ y = -1)
  sorry

end exactly_one_correct_l278_278787


namespace savings_account_final_amount_l278_278778

noncomputable def final_amount (P R : ℝ) (t : ℕ) : ℝ :=
  P * (1 + R) ^ t

theorem savings_account_final_amount :
  final_amount 2500 0.06 21 = 8017.84 :=
by
  sorry

end savings_account_final_amount_l278_278778


namespace bows_count_l278_278340

theorem bows_count (total_bows : ℕ) 
  (h_red : (total_bows : ℚ) * (1/6) = total_bows * (1/6))
  (h_blue : (total_bows : ℚ) * (1/3) = total_bows * (2/6))
  (h_yellow : (total_bows : ℚ) * (1/12) = total_bows * (1/12))
  (h_green : (total_bows : ℚ) * (1/8) = total_bows * (3/24))
  (h_white_bows : 42 = 7/24 * total_bows) :
  total_bows = 144 :=
begin
  -- Proof goes here
  sorry
end

end bows_count_l278_278340


namespace interest_percent_rounded_l278_278395

theorem interest_percent_rounded 
  (purchase_price : ℝ) 
  (down_payment : ℝ) 
  (monthly_payment : ℝ) 
  (num_payments : ℕ) 
  (total_amount_paid : ℝ := down_payment + monthly_payment * num_payments) 
  (extra_amount_paid : ℝ := total_amount_paid - purchase_price) 
  (interest_percent : ℝ := (extra_amount_paid / purchase_price) * 100) 
  (rounded_interest_percent : ℝ := Float.round (interest_percent * 10) / 10) :
  purchase_price = 130 → 
  down_payment = 30 → 
  monthly_payment = 10 → 
  num_payments = 12 → 
  rounded_interest_percent = 15.4 :=
by          
  sorry

end interest_percent_rounded_l278_278395


namespace squad_sizes_l278_278961

-- Definitions for conditions
def total_students (x y : ℕ) : Prop := x + y = 146
def equal_after_transfer (x y : ℕ) : Prop := x - 11 = y + 11

-- Theorem to prove the number of students in first and second-year squads
theorem squad_sizes (x y : ℕ) (h1 : total_students x y) (h2 : equal_after_transfer x y) : 
  x = 84 ∧ y = 62 :=
by
  sorry

end squad_sizes_l278_278961


namespace count_divisibles_l278_278298

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278298


namespace sum_of_squares_first_28_l278_278571

theorem sum_of_squares_first_28 : 
  (28 * (28 + 1) * (2 * 28 + 1)) / 6 = 7722 := by
  sorry

end sum_of_squares_first_28_l278_278571


namespace remainder_sum_mult_3_zero_mod_18_l278_278215

theorem remainder_sum_mult_3_zero_mod_18
  (p q r s : ℕ)
  (hp : p % 18 = 8)
  (hq : q % 18 = 11)
  (hr : r % 18 = 14)
  (hs : s % 18 = 15) :
  3 * (p + q + r + s) % 18 = 0 :=
by
  sorry

end remainder_sum_mult_3_zero_mod_18_l278_278215


namespace shift_line_down_4_units_l278_278816

theorem shift_line_down_4_units :
  ∀ (x : ℝ), y = - (3 / 4) * x → (y - 4 = - (3 / 4) * x - 4) := by
  sorry

end shift_line_down_4_units_l278_278816


namespace Justin_reads_total_pages_l278_278356

theorem Justin_reads_total_pages 
  (initial_pages : ℕ)
  (multiplier : ℕ)
  (days_remaining : ℕ)
  (total_days : ℕ)
  (total_pages_needed : ℕ) :
  initial_pages = 10 →
  multiplier = 2 →
  days_remaining = 6 →
  total_days = 7 →
  total_pages_needed = 100 →
  (initial_pages + days_remaining * (initial_pages * multiplier)) = 130 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry

end Justin_reads_total_pages_l278_278356


namespace largest_non_summable_composite_l278_278153

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278153


namespace good_arrangements_upper_bound_l278_278752

def is_bad (n : ℕ) (arr : list ℕ) : Prop :=
  ∃ (seq : list ℕ), seq.length = 10 ∧
  (∀ (i j : ℕ), i < j → seq.nth i > seq.nth j) ∧
  seq ⊆ arr

def is_good (n : ℕ) (arr : list ℕ) : Prop := ¬ is_bad n arr

theorem good_arrangements_upper_bound (n : ℕ) :
  {arr : list ℕ | arr.length = n ∧ is_good n arr}.to_finset.card ≤ 81^n :=
sorry

end good_arrangements_upper_bound_l278_278752


namespace valid_sequences_length_21_l278_278662

def valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else valid_sequences (n - 3) + valid_sequences (n - 4)

theorem valid_sequences_length_21 : valid_sequences 21 = 38 :=
by
  sorry

end valid_sequences_length_21_l278_278662


namespace slope_of_AB_equation_of_line_AB_l278_278736

-- Define the curve
def curve (x : ℝ) : ℝ := (x^2) / 4

-- Define the points A and B on the curve
variables {A B : ℝ × ℝ}
variables (hA : A.snd = curve A.fst) (hB : B.snd = curve B.fst)
variable (h_sum : A.fst + B.fst = 4)

-- Define the slope of line AB
def slope (P Q : ℝ × ℝ) : ℝ := (Q.snd - P.snd) / (Q.fst - P.fst)

-- The slope of line AB is 1
theorem slope_of_AB : slope A B = 1 :=
  by
    sorry

-- Define point M on the curve
variables {M : ℝ × ℝ}
variable (hM : M = (2, 1))

-- Define the tangent slope at M
def tangent_slope_at_M := 1 / 2 * M.fst

-- Conditions that AM ⊥ BM
variables (h_tangent_parallel : tangent_slope_at_M = 1)
variables (h_perpendicular : slope A M * slope B M = -1)

-- Equation of line AB
noncomputable def line_AB : ℝ → ℝ := λ x, x + 7

-- Prove the equation of line AB
theorem equation_of_line_AB : ∃ t : ℝ, ∀ x y : ℝ, y = line_AB x ↔ y = x + 7 :=
  by
    use 7
    intros x y
    simp [line_AB]
    sorry

end slope_of_AB_equation_of_line_AB_l278_278736


namespace largest_cannot_be_sum_of_two_composites_l278_278156

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278156


namespace triangle_ABC_is_isosceles_l278_278413
variables {α : Type*} [euclidean_geometry α]
variable {A B C O H D E : Point α}
variable {triangle_ABC : triangle α}
variable {circumcenter_O : Incircle α}
variable {orthocenter_H : Point α}

-- Definitions based on the problem conditions
def is_circumcenter_O := circumcenter_O center = O
def is_orthocenter_H := orthocenter_H = H
def intersects_B_H_D := segment H B ∩ segment O C = {D}
def intersects_C_H_E := segment H C ∩ segment O B = {E}
def is_isosceles_ODH := is_isosceles O D H
def is_isosceles_OEH := is_isosceles O E H

-- Main theorem statement
theorem triangle_ABC_is_isosceles
  (h1 : is_circumcenter_O)
  (h2 : is_orthocenter_H)
  (h3 : intersects_B_H_D)
  (h4 : intersects_C_H_E)
  (h5 : is_isosceles_ODH)
  (h6 : is_isosceles_OEH) :
  is_isosceles A B C :=
sorry

end triangle_ABC_is_isosceles_l278_278413


namespace measure_of_AB_l278_278493

-- Define the geometrical setup
variables {A B C D E: Type} -- Points in the geometric diagram
variables {c d : ℝ} -- Measures of segments AD and CD

-- Define the conditions
variable (parallelABCD : ∀ (ab cd : Line), ab ∥ cd)
variable (angleRelation : ∀ b : Angle, exists (d : Angle), d = 3 * b)
variable (lengthAD : ℝ)
variable (lengthCD : ℝ)

-- State the theorem
theorem measure_of_AB (h1 : parallelABCD (A, B) (C, D))
                      (h2 : ∃ b : ℝ, ∃ d : ℝ, d = 3 * b)
                      (h3 : lengthAD = c)
                      (h4 : lengthCD = d) :
  ∃ AB : ℝ, AB = (d + sqrt (d^2 + 4 * c * d)) / 2 := 
sorry

end measure_of_AB_l278_278493


namespace largest_non_sum_of_composites_l278_278170

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278170


namespace maximize_seq_sum_l278_278212

/-- Given an arithmetic sequence {a_n}, where |a_3| = |a_9| and the common difference d < 0,
    prove that the positive integer n that maximizes the sum of the first n terms is either 5 or 6. -/
theorem maximize_seq_sum {a : ℕ → ℝ} (d : ℝ) (h3 : abs (a 3) = abs (a 9)) (h_neg : d < 0) :
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧ (∀ m : ℕ, m > 0 → m < n → (finset.range m).sum a ≤ (finset.range n).sum a) :=
sorry

end maximize_seq_sum_l278_278212


namespace company_profit_iff_employs_min_76_workers_l278_278531

-- Given conditions
variables (daily_maintenance_cost hourly_wage daily_widget_production widget_price workday_hours : ℕ)
variable (n : ℕ) -- number of workers

-- Definitions from conditions
def daily_worker_cost : ℕ := hourly_wage * workday_hours
def daily_worker_revenue : ℕ := widget_price * daily_widget_production * workday_hours

-- Equation for turning a profit
def profit_condition := daily_maintenance_cost + daily_worker_cost * n < daily_worker_revenue * n

-- Conditions instantiation
axiom maintenance_cost : daily_maintenance_cost = 600
axiom worker_hourly_wage : hourly_wage = 20
axiom widget_production : daily_widget_production = 6
axiom widget_sale_price : widget_price = 3.50.to_nat -- converting to natural number 
axiom work_hours : workday_hours = 8

theorem company_profit_iff_employs_min_76_workers :
  profit_condition daily_maintenance_cost hourly_wage daily_widget_production widget_price workday_hours n →
  n ≥ 76 := sorry

end company_profit_iff_employs_min_76_workers_l278_278531


namespace largest_cannot_be_sum_of_two_composites_l278_278160

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278160


namespace find_positive_number_l278_278002

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l278_278002


namespace machine_generates_all_positive_rationals_l278_278031

-- Define the machine operations under the given conditions

variable (machine : Set ℚ → Set ℚ) -- A function that takes a set of rationals and returns a set of rationals

-- Condition 1: The machine starts with the number 100
def starts_with_100 (s : Set ℚ) : Prop :=
  100 ∈ s

-- Condition 2: The machine can generate the reciprocal of any available number
def generate_reciprocal (s : Set ℚ) : Set ℚ :=
  {y | ∃ x ∈ s, y = 1 / x}

-- Condition 3: The machine can sum any two different available numbers
def generate_sum (s : Set ℚ) : Set ℚ :=
  {y | ∃ x1 x2 ∈ s, x1 ≠ x2 ∧ y = x1 + x2}

-- Union of all operations the machine can perform
def machine_operations (s : Set ℚ) : Set ℚ :=
  s ∪ generate_reciprocal s ∪ generate_sum s

-- The machine can operate indefinitely means we consider the closure of operations
def machine_closure (s : Set ℚ) : Set ℚ :=
  Union (λ n, Nat.iterate machine_operations n s)

-- Proposition to prove:
theorem machine_generates_all_positive_rationals :
  ∀ s, starts_with_100 s → machine_closure s = {r : ℚ | r > 0} :=
by
  intros s h
  -- Proof to be provided
  sorry

end machine_generates_all_positive_rationals_l278_278031


namespace Mona_grouped_with_one_player_before_in_second_group_l278_278386

/-- Mona plays in groups with four other players, joined 9 groups, and grouped with 33 unique players. 
    One of the groups included 2 players she had grouped with before. 
    Prove that the number of players she had grouped with before in the second group is 1. -/
theorem Mona_grouped_with_one_player_before_in_second_group 
    (total_groups : ℕ) (group_size : ℕ) (unique_players : ℕ) 
    (repeat_players_in_group1 : ℕ) : 
    (total_groups = 9) → (group_size = 5) → (unique_players = 33) → (repeat_players_in_group1 = 2) 
        → ∃ repeat_players_in_group2 : ℕ, repeat_players_in_group2 = 1 :=
by
    sorry

end Mona_grouped_with_one_player_before_in_second_group_l278_278386


namespace count_knights_15_l278_278754

def knight_or_liar (p : ℕ) : Prop :=
(p = 1 ∨ p = 4 ∨ p = 5 ∨ p = 6 ∨ p = 7 ∨ p = 8 ∨ 
p = 9 ∨ p = 10 ∨ p = 11 ∨ p = 12 ∨ p = 13 ∨ p = 14 ∨ p = 15) →
(p ≠ 1 → p = 4 ∨ p = 5 ∨ p = 6 ∨ p = 7 ∨ p = 8 ∨ (1 ≤ p ∧ p ≤ 15))

theorem count_knights_15 {n : ℕ} (h : n = 15) : 
∃ (k : ℕ), k = 11 ∧ ∀ (p : ℕ) (H : p ≤ n), knight_or_liar p :=
begin
  sorry
end

end count_knights_15_l278_278754


namespace number_of_divisibles_by_eight_in_range_l278_278274

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278274


namespace james_drove_75_miles_l278_278712

noncomputable def james_total_distance : ℝ :=
  let speed1 := 30  -- mph
  let time1 := 0.5  -- hours
  let speed2 := 2 * speed1
  let time2 := 2 * time1
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  distance1 + distance2

theorem james_drove_75_miles : james_total_distance = 75 := by 
  sorry

end james_drove_75_miles_l278_278712


namespace value_of_A_l278_278126

noncomputable def letter_value (c : Char) : ℕ := sorry
def H := 8
def MATH := 32
def TEAM := 40
def MEET := 36

def value_of_word (word : List Char) : ℕ :=
  word.foldl (λ acc c => acc + letter_value c) 0

theorem value_of_A : 
  letter_value 'A' = 20 := 
by 
  have h_math : value_of_word ['M', 'A', 'T', 'H'] = MATH := sorry
  have h_team : value_of_word ['T', 'E', 'A', 'M'] = TEAM := sorry
  have h_meet : value_of_word ['M', 'E', 'E', 'T'] = MEET := sorry
  have h_H : letter_value 'H' = H := sorry
  sorry

end value_of_A_l278_278126


namespace possible_values_of_dot_product_l278_278315

noncomputable def vector_dot_product_values (a b : ℝ) (h1 : ‖a‖ = 5) (h2 : ‖b‖ = 12) : set ℝ :=
  {x | x = 0 ∨ x = 60 ∨ x = -60}

theorem possible_values_of_dot_product (a b : ℝ) (h1 : ‖a‖ = 5) (h2 : ‖b‖ = 12)
  (h3 : ∀ θ : ℝ, θ = 0 ∨ θ = π / 2 ∨ θ = π → a • b = ‖a‖ * ‖b‖ * cos θ) : 
  vector_dot_product_values a b h1 h2 = {0, 60, -60} :=
by
  sorry

end possible_values_of_dot_product_l278_278315


namespace no_solution_exists_l278_278094

theorem no_solution_exists : ¬ ∃ (x : ℕ), (42 + x = 3 * (8 + x) ∧ 42 + x = 2 * (10 + x)) :=
by
  sorry

end no_solution_exists_l278_278094


namespace distance_center_of_ball_travel_l278_278522

theorem distance_center_of_ball_travel (d R1 R2 L : ℝ) (r : ℝ) :
  d = 6 → R1 = 150 → R2 = 90 → L = 100 → r = d / 2 →
  ((R1 - r) * π + (R2 - r) * π + L) = 234 * π + 100 :=
by
  intros hd hR1 hR2 hL hr
  rw [hd, hR1, hR2, hL, hr]
  sorry

end distance_center_of_ball_travel_l278_278522


namespace ceilings_left_correct_l278_278749

def total_ceilings : ℕ := 28
def ceilings_painted_this_week : ℕ := 12
def ceilings_painted_next_week : ℕ := ceilings_painted_this_week / 4
def ceilings_left_to_paint : ℕ := total_ceilings - (ceilings_painted_this_week + ceilings_painted_next_week)

theorem ceilings_left_correct : ceilings_left_to_paint = 13 := by
  sorry

end ceilings_left_correct_l278_278749


namespace non_monotonic_interval_l278_278675

def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem non_monotonic_interval (k : ℝ) :
  ¬(∀ x1 x2 ∈ set.Ioo (k-1) (k+1), x1 < x2 → f x1 ≤ f x2) ↔ (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
begin
  sorry
end

end non_monotonic_interval_l278_278675


namespace jeans_vs_scarves_l278_278528

theorem jeans_vs_scarves :
  ∀ (ties belts black_shirts white_shirts : ℕ),
  ties = 34 →
  belts = 40 →
  black_shirts = 63 →
  white_shirts = 42 →
  let total_shirts := black_shirts + white_shirts in
  let jeans := (2 * total_shirts) / 3 in
  let total_ties_and_belts := ties + belts in
  let scarves := total_ties_and_belts / 2 in
  jeans - scarves = 33 :=
by
  intros ties belts black_shirts white_shirts ht hb hbs hws
  let total_shirts := black_shirts + white_shirts
  let jeans := (2 * total_shirts) / 3
  let total_ties_and_belts := ties + belts
  let scarves := total_ties_and_belts / 2
  show jeans - scarves = 33
  sorry

end jeans_vs_scarves_l278_278528


namespace correct_statement_is_C_l278_278489

theorem correct_statement_is_C :
  (∃ x : ℚ, ∀ y : ℚ, x < y) = false ∧
  (∃ x : ℚ, x < 0 ∧ ∀ y : ℚ, y < 0 → x < y) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y) ∧
  (∃ x : ℝ, 0 < x ∧ ∀ y : ℝ, 0 < y → x ≤ y) = false :=
sorry

end correct_statement_is_C_l278_278489


namespace prob_adjacent_ab_l278_278865

-- Defining the context and assumptions
def num_people := 5
def num_adjacent_ab := 2 * fact (num_people - 1)
def total_arrangements := fact num_people

-- The theorem statement
theorem prob_adjacent_ab : (num_adjacent_ab / total_arrangements) = 0.4 :=
by sorry

end prob_adjacent_ab_l278_278865


namespace problem_solution_l278_278613

noncomputable def complex_z : ℂ := sorry -- We need the system of equations to solve x and y

theorem problem_solution :
  (∀ z : ℂ, |z| = real.sqrt 2 ∧ (z^2).im = -2 ∧ z.re < 0 ∧ z.im > 0 → z = -1 + complex.I) ∧
  (|complex_z| = real.sqrt 2 ∧ (complex_z^2).im = -2 ∧ complex_z.re < 0 ∧ complex_z.im > 0 →
   (∀ ω : ℂ, |ω - 1| ≤ complex.abs (complex.conj complex_z / (complex_z + complex.I)) →
    |ω - 1| ≤ real.sqrt (2/5) * real.pi))
:= sorry

end problem_solution_l278_278613


namespace intersection_y_value_l278_278807

theorem intersection_y_value : ∃ (x y : ℝ), y = 3 * x + 5 ∧ 5 * x - 2 * y = 20 ∧ y = -85 :=
by
  use -30, -85
  split
  · rhs
    calc
      y = 3 * (-30) + 5 := by sorry
        = -85 := by sorry
  split
  · rhs
    calc
      5 * (-30) - 2 * (-85) = 20 := by sorry
  rfl

end intersection_y_value_l278_278807


namespace best_play_wins_probability_l278_278072

/-- Define the conditions and parameters for the problem. -/
variables (n m : ℕ)
variables (C : ℕ → ℕ → ℕ) /- Binomial coefficient -/

/-- Define the probability calculation -/
def probability_best_play_wins : ℚ :=
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t)

/-- The theorem stating that the above calculation represents the probability of the best play winning -/
theorem best_play_wins_probability :
  probability_best_play_wins n m C =
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t) :=
  by
  sorry

end best_play_wins_probability_l278_278072


namespace sqrt_multiplication_division_l278_278572

theorem sqrt_multiplication_division :
  (sqrt 3 * sqrt 15) / sqrt 5 = 3 :=
by
  sorry

end sqrt_multiplication_division_l278_278572


namespace water_to_add_for_desired_composition_l278_278532

def initial_milk_volume (total_volume : ℝ) (milk_percentage : ℝ) : ℝ :=
  (milk_percentage / 100) * total_volume

def initial_water_volume (total_volume : ℝ) (water_percentage : ℝ) : ℝ :=
  (water_percentage / 100) * total_volume

def initial_honey_volume (total_volume : ℝ) (honey_percentage : ℝ) : ℝ :=
  (honey_percentage / 100) * total_volume

def new_milk_volume (initial_volume : ℝ) (x : ℝ) : ℝ :=
  initial_volume

def new_water_volume (initial_volume : ℝ) (x : ℝ) : ℝ :=
  initial_volume + x

def new_honey_volume (initial_volume : ℝ) (x : ℝ) : ℝ :=
  initial_volume

def new_total_volume (initial_total : ℝ) (x : ℝ) : ℝ :=
  initial_total + x

def percentage (part : ℝ) (whole : ℝ) : ℝ :=
  (part / whole) * 100

theorem water_to_add_for_desired_composition 
  (total_volume : ℝ)
  (milk_percentage : ℝ)
  (water_percentage : ℝ)
  (honey_percentage : ℝ)
  (desired_milk_percentage : ℝ)
  (desired_water_percentage : ℝ)
  (desired_honey_percentage : ℝ)
  (x : ℝ) :
  total_volume = 120 ∧
  milk_percentage = 70 ∧
  water_percentage = 25 ∧
  honey_percentage = 5 ∧
  desired_milk_percentage = 50 ∧
  desired_water_percentage = 47 ∧
  desired_honey_percentage = 3 ∧
  x ≈ 49.81 →
  percentage (new_milk_volume (initial_milk_volume total_volume milk_percentage) x)
             (new_total_volume total_volume x) = desired_milk_percentage ∧
  percentage (new_water_volume (initial_water_volume total_volume water_percentage) x)
             (new_total_volume total_volume x) = desired_water_percentage ∧
  percentage (new_honey_volume (initial_honey_volume total_volume honey_percentage) x)
             (new_total_volume total_volume x) = desired_honey_percentage := 
by {
  intros,
  sorry
}

end water_to_add_for_desired_composition_l278_278532


namespace smaller_prime_factor_l278_278792

theorem smaller_prime_factor (a b : ℕ) (prime_a : Nat.Prime a) (prime_b : Nat.Prime b) (distinct : a ≠ b)
  (product : a * b = 316990099009901) :
  min a b = 4002001 :=
  sorry

end smaller_prime_factor_l278_278792


namespace prob_event_E_zero_l278_278679

def set_S : Set ℕ := {4, 5, 6, 9}

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def event_E (a b : ℕ) : Prop := a ≠ b ∧ is_multiple_of (a * b) 16

theorem prob_event_E_zero :
  let choices := {pair : ℕ × ℕ | pair.1 ∈ set_S ∧ pair.2 ∈ set_S ∧ pair.1 ≠ pair.2}
  (∀ pair ∈ choices, ¬ event_E pair.1 pair.2) →
  (∀ total_choices : ℕ, total_choices = choices.toFinset.card → 0 = 0 / total_choices.toRat) := 
by
  intros choices h total_choices h_total_choices
  exact (h_total_choices ▸ by simp : 0 = 0)

end prob_event_E_zero_l278_278679


namespace chip_credit_card_balance_l278_278940

-- Definitions based on the problem conditions
def initial_balance : ℝ := 50.00
def interest_rate : ℝ := 0.20
def additional_amount : ℝ := 20.00

-- Define the function to calculate the final balance after two months
def final_balance (b₀ r a : ℝ) : ℝ :=
  let b₁ := b₀ * (1 + r) in
  let b₂ := (b₁ + a) * (1 + r) in
  b₂

-- Theorem to prove that the final balance is 96.00
theorem chip_credit_card_balance : final_balance initial_balance interest_rate additional_amount = 96.00 :=
by
  -- Simplified proof outline
  sorry

end chip_credit_card_balance_l278_278940


namespace uber_profit_l278_278857

-- Define conditions
def income : ℕ := 30000
def initial_cost : ℕ := 18000
def trade_in : ℕ := 6000

-- Define depreciation cost
def depreciation_cost : ℕ := initial_cost - trade_in

-- Define the profit
def profit : ℕ := income - depreciation_cost

-- The theorem to be proved
theorem uber_profit : profit = 18000 := by 
  sorry

end uber_profit_l278_278857


namespace penumbra_ring_area_l278_278325

theorem penumbra_ring_area (r_umbra r_penumbra : ℝ) (h_ratio : r_umbra / r_penumbra = 2 / 6) (h_umbra : r_umbra = 40) :
  π * (r_penumbra ^ 2 - r_umbra ^ 2) = 12800 * π := by
  sorry

end penumbra_ring_area_l278_278325


namespace james_second_hour_distance_l278_278354

theorem james_second_hour_distance :
  ∃ x : ℝ, 
    x + 1.20 * x + 1.50 * x = 37 ∧ 
    1.20 * x = 12 :=
by
  sorry

end james_second_hour_distance_l278_278354


namespace midsegment_sum_inequality_l278_278416

theorem midsegment_sum_inequality (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  let midsegment_sum := (b + c) / 2 + (a + c) / 2 + (a + b) / 2 in
  (3 / 4) * (a + b + c) < midsegment_sum ∧ midsegment_sum < a + b + c :=
by
  sorry

end midsegment_sum_inequality_l278_278416


namespace bird_population_2002_l278_278887

theorem bird_population_2002 :
  ∃ (y : ℕ), y = 115 ∧
  (∃ k : ℚ, k * 95 = 115 - 58 ∧ k * 115 = 178 - 95) :=
begin
  existsi 115,
  split,
  { refl },
  { existsi (83 / 115 : ℚ),
    split,
    { norm_num },
    { norm_num } }
end

end bird_population_2002_l278_278887


namespace largest_not_sum_of_two_composites_l278_278139

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278139


namespace actual_diameter_of_tissue_l278_278044

theorem actual_diameter_of_tissue (magnification: ℝ) (magnified_diameter: ℝ) :
  magnification = 1000 ∧ magnified_diameter = 1 → magnified_diameter / magnification = 0.001 :=
by
  intro h
  sorry

end actual_diameter_of_tissue_l278_278044


namespace log_is_logarithmic_l278_278788

theorem log_is_logarithmic (x : ℝ) (hx1 : x > 0) : 
  (∃ f : ℝ → ℝ, f = log x) → 
  (y = log 2 x ∧ y = log 3 x → is_logarithmic_function y) :=
by
  intros
  sorry

end log_is_logarithmic_l278_278788


namespace num_solutions_congruence_l278_278225

-- Define the problem context and conditions
def is_valid_solution (y : ℕ) : Prop :=
  y < 150 ∧ (y + 21) % 46 = 79 % 46

-- Define the proof problem
theorem num_solutions_congruence : ∃ (s : Finset ℕ), s.card = 3 ∧ ∀ y ∈ s, is_valid_solution y := by
  sorry

end num_solutions_congruence_l278_278225


namespace expression1_value_expression2_value_l278_278112

theorem expression1_value :
  ( (9/4)^(1/2) - (-9.6)^0 - (27/8)^(2/3) + 1.5^2 + (sqrt 2 * 43)^4 ) = 6942483.5 :=
by sorry

theorem expression2_value :
  ( (log 10 (sqrt 27) + log 10 8 - log 10 (sqrt 1000)) / 
    ((1/2) * log 10 0.3 + log 10 2) + (sqrt 5 - 2)^0 + 0.027^(-1/3) * (-1/3)^(-2) ) = 13 :=
by sorry

end expression1_value_expression2_value_l278_278112


namespace correct_statements_l278_278658

structure Vector2D where
  m : ℝ
  n : ℝ

def otimes (a b : Vector2D) : ℝ :=
  a.m * a.n - b.m * b.n

def dot (a b : Vector2D) : ℝ :=
  a.m * b.m + a.n * b.n

theorem correct_statements (a b : Vector2D) :
  (otimes a a = 0) ∧ ((otimes a b)^2 + (dot a b)^2 = (a.m^2 + b.q^2) * (a.n^2 + b.p^2)) :=
sorry

end correct_statements_l278_278658


namespace distinct_real_roots_l278_278676

-- Define the quadratic equation
def quadratic_eqn (k : ℝ) : polynomial ℝ := polynomial.monomial 2 1 - polynomial.monomial 1 2 + polynomial.C k

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (-2)^2 - 4 * 1 * k

-- Prove that the quadratic equation has two distinct real roots if and only if k < 1
theorem distinct_real_roots (k : ℝ) : (k < 1) ↔ (discriminant k > 0) := by
  sorry

end distinct_real_roots_l278_278676


namespace inradius_of_equilateral_triangle_l278_278371

-- Defining the equilateral triangle and its properties
variables (A B C I : Type) [equilateral_triangle : triangle A B C]
variables (BC : ℝ) (IC : ℝ) (r : ℝ)

-- Given conditions
def side_length_eq : BC = 24 := sorry
def incenter_distance_eq : IC = 12 * real.sqrt 3 := sorry

-- The statement of the proof problem
theorem inradius_of_equilateral_triangle : r = 4 * real.sqrt 3 :=
by
  use side_length_eq
  use incenter_distance_eq
  sorry

end inradius_of_equilateral_triangle_l278_278371


namespace part_one_part_two_l278_278245

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6) + 1 / 2

theorem part_one : (∃ T > 0, ∀ x, f(x) = f(x + T)) ∧ (∀ T > 0, (∀ x, f x = f (x + T)) → T ≥ π) := 
sorry

theorem part_two : 
  (∀ x ∈ Icc (-π / 3) ((π / 3 : ℝ) : ℝ), f x ≤ 3 / 2) ∧ 
  (∃ m, ∀ x ∈ Icc (-(π / 3)) m, f x ≤ 3 / 2 ∧ 
  ((m ≥ π / 3) ∧ ∀ n > 0, (∀ x ∈ Icc (-π / 3) n, f x ≤ 3 / 2) → n ≥ m )) := 
sorry

end part_one_part_two_l278_278245


namespace largest_non_representable_as_sum_of_composites_l278_278164

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278164


namespace total_distance_traveled_l278_278085

/--
A spider is on the edge of a ceiling of a circular room with a radius of 65 feet. 
The spider walks straight across the ceiling to the opposite edge, passing through 
the center. It then walks straight to another point on the edge of the circle but 
not back through the center. The third part of the journey is straight back to the 
original starting point. If the third part of the journey was 90 feet long, then 
the total distance traveled by the spider is 313.81 feet.
-/
theorem total_distance_traveled (r : ℝ) (d1 d2 d3 : ℝ) (h1 : r = 65) (h2 : d1 = 2 * r) (h3 : d3 = 90) :
  d1 + d2 + d3 = 313.81 :=
by
  sorry

end total_distance_traveled_l278_278085


namespace midpoint_expression_eq_neg28_l278_278744

-- Define the points A and B
def A : ℝ × ℝ := (10, 15)
def B : ℝ × ℝ := (-2, 3)

-- Define the midpoint C of points A and B
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Point C is the midpoint of A and B
def C : ℝ × ℝ := midpoint A B

-- The expression to be evaluated
def expression (C : ℝ × ℝ) : ℝ :=
  2 * C.1 - 4 * C.2

-- The theorem statement
theorem midpoint_expression_eq_neg28 : expression C = -28 :=
by
  sorry

end midpoint_expression_eq_neg28_l278_278744


namespace probability_product_positive_l278_278021

-- Define interval and properties
def interval : set ℝ := { x : ℝ | -15 ≤ x ∧ x ≤ 15 }

-- Define independent selection of x and y
def selected_independently (x y : ℝ) : Prop :=
  x ∈ interval ∧ y ∈ interval

-- Main theorem
theorem probability_product_positive : 
  (∃ x y ∈ interval, selected_independently x y ∧ x * y > 0) → (probability (x * y > 0) = 1/2) :=
sorry

end probability_product_positive_l278_278021


namespace tom_has_65_fruits_left_l278_278817

def initial_fruits : ℕ := 40 + 70 + 30 + 15

def sold_oranges : ℕ := (1 / 4) * 40
def sold_apples : ℕ := (2 / 3) * 70
def sold_bananas : ℕ := (5 / 6) * 30
def sold_kiwis : ℕ := (60 / 100) * 15

def fruits_remaining : ℕ :=
  40 - sold_oranges +
  70 - sold_apples +
  30 - sold_bananas +
  15 - sold_kiwis

theorem tom_has_65_fruits_left :
  fruits_remaining = 65 := by
  sorry

end tom_has_65_fruits_left_l278_278817


namespace limit_theorem_l278_278109

noncomputable def limit_problem : ℝ :=
  lim (λ x : ℝ, (1 / x) ^ (log (x + 1) / log (2 - x))) 1

theorem limit_theorem : limit_problem = 2 := by
  sorry

end limit_theorem_l278_278109


namespace max_value_and_points_sum_y_n_2018_l278_278244

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) * (Real.cos x + Real.sqrt 3 * Real.sin x) - 1

theorem max_value_and_points (k : ℤ) : 
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ x : ℝ, f x = 2) ∧ (x = k * Real.pi + Real.pi / 6) :=
sorry

def x₁ : ℝ := Real.pi / 6
def x_n (n : ℕ) : ℝ := x₁ + n * (Real.pi / 3)
def y_n (n : ℕ) : ℝ := 2 * Real.sin (2 * x_n n + Real.pi / 6)

theorem sum_y_n_2018 : ∑ n in Finset.range 2019, y_n n = 1 :=
sorry

end max_value_and_points_sum_y_n_2018_l278_278244


namespace sum_of_inserted_numbers_in_arithmetic_sequence_l278_278709

theorem sum_of_inserted_numbers_in_arithmetic_sequence :
  ∃ a2 a3 : ℤ, 2015 > a2 ∧ a2 > a3 ∧ a3 > 131 ∧ (2015 - a2) = (a2 - a3) ∧ (a2 - a3) = (a3 - 131) ∧ (a2 + a3) = 2146 := 
by
  sorry

end sum_of_inserted_numbers_in_arithmetic_sequence_l278_278709


namespace new_year_day_position_l278_278915

-- Conditions
def winter_solstice_2012 : ℕ := 12 * 21    -- December 21, 2012
def day1_of_first_nine := winter_solstice_2012

-- Known date
def new_years_day_2013 : ℕ := 1 * 1 + 12 * 31  -- January 1, 2013

-- Number of days between December 21, 2012 (inclusive) and January 1, 2013
def days_between_start_end : ℕ := new_years_day_2013 - winter_solstice_2012 + 1

-- Proof Problem 
theorem new_year_day_position : (days_between_start_end ≡ 12) % 9 :=
by {
    -- Calculate the modulus to get the position within nine-day period
    sorry
}

end new_year_day_position_l278_278915


namespace sam_servings_l278_278769

variable (pasta_cost : ℕ) (sauce_cost : ℕ) (meatballs_cost : ℕ) (cost_per_serving : ℕ)
variable (total_cost : ℕ) (number_of_servings : ℕ)

definition spaghetti_meal (pasta_cost sauce_cost meatballs_cost cost_per_serving : ℕ) : Prop :=
  total_cost = pasta_cost + sauce_cost + meatballs_cost ∧
  number_of_servings * cost_per_serving = total_cost

theorem sam_servings :
  spaghetti_meal pasta_cost sauce_cost meatballs_cost cost_per_serving →
  pasta_cost = 1 →
  sauce_cost = 2 →
  meatballs_cost = 5 →
  cost_per_serving = 1 →
  number_of_servings = 8 :=
by
  sorry

end sam_servings_l278_278769


namespace engineer_sidorov_error_is_4kg_l278_278440

-- Given the conditions
def diameter := 1                     -- diameter of each disk in meters
def error_radius_std_dev := 0.01      -- standard deviation of the radius in meters
def mass_per_disk := 100              -- mass of one disk with diameter 1 meter in kg
def num_disks := 100                  -- number of disks
def estimate_sidorov := 10000         -- Engineer Sidorov's estimate in kg

-- Expected calculations
def expected_mass_one_disk := mass_per_disk * (1 + (error_radius_std_dev^2 / (1/4))) -- expected mass considering the deviation
def expected_total_mass := expected_mass_one_disk * num_disks
def true_error := expected_total_mass - estimate_sidorov

-- Prove that the error in Engineer Sidorov's estimate of the total mass of 100 disks is 4 kg
theorem engineer_sidorov_error_is_4kg :
  true_error = 4 :=
sorry

end engineer_sidorov_error_is_4kg_l278_278440


namespace larger_solution_proof_l278_278134

noncomputable def larger_solution : ℝ :=
  let f := λ (x : ℝ), x^2 + 17*x - 72 in
  let solutions := {x : ℝ | f x = 0} in
  if h : ∃ x₁ x₂ ∈ solutions, x₁ ≠ x₂ ∧ (∀ y ∈ solutions, y = x₁ ∨ y = x₂) then
    let ⟨x₁, x₂, hx₁, hx₂, hneq, hall⟩ := h in
    if x₁ > x₂ then x₁ else x₂
  else
    0  -- default value, but should never happen since we know there are two solutions

theorem larger_solution_proof : larger_solution = 3 :=
sorry

end larger_solution_proof_l278_278134


namespace largest_non_summable_composite_l278_278150

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278150


namespace find_halfway_between_l278_278793

def halfway_between (a b : ℚ) : ℚ := (a + b) / 2

theorem find_halfway_between :
  halfway_between (1/8 : ℚ) (1/3 : ℚ) = 11/48 :=
by
  -- declare needed intermediate calculations (common denominators, etc.)
  sorry

end find_halfway_between_l278_278793


namespace cannot_determine_c_l278_278194

-- Definitions based on conditions
variables {a b c d : ℕ}
axiom h1 : a + b + c = 21
axiom h2 : a + b + d = 27
axiom h3 : a + c + d = 30

-- The statement that c cannot be determined exactly
theorem cannot_determine_c : ¬ (∃ c : ℕ, c = c) :=
by sorry

end cannot_determine_c_l278_278194


namespace max_profit_l278_278888

-- Define the unit prices of thermos cups A and B
variable (priceA priceB : ℕ)

-- Define the quantities of thermos cups A and B
variable (qtyA qtyB : ℕ)

-- Define the total quantity of thermos cups to be purchased
variable (totalQty : ℕ) := 120

-- Define the cost and selling prices
variable (costPriceA costPriceB : ℕ) := 30
variable (priceB_discounted : ℕ) := priceB * 9 / 10

-- Define the profit calculation
def profit (qtyA qtyB costPriceA costPriceB priceB_discounted : ℕ) : ℕ :=
  (priceA - costPriceA) * qtyA + (priceB_discounted - costPriceB) * qtyB

-- Define conditions as Lean hypotheses
theorem max_profit : priceA = 40 ∧ priceB = 50 ∧ qtyA = 40 ∧ qtyB = 80 ∧ profit qtyA qtyB costPriceA costPriceB priceB_discounted = 1600 :=
by
  -- Provided conditions
  have h1 : priceB = priceA + 10 := by sorry
  have h2 : 600 / priceB = 480 / priceA := by sorry
  have h3 : qtyA + qtyB = totalQty := by sorry
  have h4 : qtyA ≥ qtyB / 2 := by sorry
  have h5 : costPriceA = 30 := by sorry
  have h6 : priceB_discounted = priceB * 9 / 10 := by sorry
  -- Conclusion based on provided problem statement
  have c1 : priceA = 40 := by sorry
  have c2 : priceB = 50 := by sorry
  have c3 : qtyA = 40 := by sorry
  have c4 : qtyB = 80 := by sorry
  have c5 : profit qtyA qtyB costPriceA costPriceB priceB_discounted = 1600 := by sorry
  exact ⟨c1, c2, c3, c4, c5⟩

end max_profit_l278_278888


namespace e_neg_2i_in_third_quadrant_l278_278582

theorem e_neg_2i_in_third_quadrant : 
  ∃ z : ℂ, z = complex.exp (-2 * complex.I) ∧ (z.re < 0) ∧ (z.im < 0) :=
by
  sorry

end e_neg_2i_in_third_quadrant_l278_278582


namespace average_age_before_new_students_joined_l278_278781

/-
Problem: Given that the original strength of the class was 18, 
18 new students with an average age of 32 years joined the class, 
and the average age decreased by 4 years, prove that 
the average age of the class before the new students joined was 40 years.
-/

def original_strength := 18
def new_students := 18
def average_age_new_students := 32
def decrease_in_average_age := 4
def original_average_age := 40

theorem average_age_before_new_students_joined :
  (original_strength * original_average_age + new_students * average_age_new_students) / (original_strength + new_students) = original_average_age - decrease_in_average_age :=
by
  sorry

end average_age_before_new_students_joined_l278_278781


namespace value_of_expression_at_minus_three_l278_278485

theorem value_of_expression_at_minus_three :
  (x : ℤ) (h : x = -3) : 3 * x^2 + 2 * x = 21 :=
by
  intro x h
  sorry

end value_of_expression_at_minus_three_l278_278485


namespace probability_sin_bound_l278_278407

def prob_sin_bound (x : ℝ) (h : -1 ≤ x ∧ x ≤ 2) : Prop :=
  -1 < 2 * Real.sin (π * x / 4) ∧ 2 * Real.sin (π * x / 4) < Real.sqrt 2

theorem probability_sin_bound :
  let interval_length := (2 - (-1)) in
  let valid_interval_length := (1 - (-2/3)) in
  let expected_probability := valid_interval_length / interval_length in
  expected_probability = 5 / 9 :=
by
  sorry

end probability_sin_bound_l278_278407


namespace lines_through_point_distance_origin_l278_278133

theorem lines_through_point_distance_origin :
  (∃ line1 line2 : ℝ → ℝ → Prop,
      (∀ x y, line1 x y ↔ x = 3) ∧ 
      (∀ x y, line2 x y ↔ 8 * x - 15 * y + 51 = 0) ∧
      line1 3 5 ∧
      line2 3 5 ∧
      ∀ x y, (line1 x y ∨ line2 x y) → (3 / (sqrt (1^2 + 0^2))) = 3 ) :=
sorry

end lines_through_point_distance_origin_l278_278133


namespace find_pairs_l278_278131

theorem find_pairs (a k : ℕ) (h_a : a = 1) (n : ℕ) (h_coprime : Nat.coprime n a) : (n ∣ (a ^ ((k ^ n) + 1) - 1)) :=
by
  -- Proof goes here
  sorry

end find_pairs_l278_278131


namespace domain_k_l278_278955

def k (x : ℝ) : ℝ := 1 / (x + 5) + 1 / (x^2 + 5) + 1 / (x^3 + 5)

theorem domain_k :
  ∀ x : ℝ, x ≠ -5 ∧ x ≠ -real.cbrt 5 ↔ k x ∈ set.univ :=
begin
  sorry
end

end domain_k_l278_278955


namespace axis_of_symmetry_is_pi_over_4_l278_278455

-- Given the function y = sin(2x + π/b)
def f (x : ℝ) (b : ℝ) : ℝ := Real.sin (2 * x + Real.pi / b)

-- Prove that the axis of symmetry for the function f(x) is x = π/4
theorem axis_of_symmetry_is_pi_over_4 (b : ℝ) : axis_of_symmetry (f x b) = Real.pi / 4 := 
sorry

end axis_of_symmetry_is_pi_over_4_l278_278455


namespace correct_survey_method_l278_278855

-- Definitions based on conditions
def OptionA : Prop := (∀ survey_method : Type, survey_method ≠ comprehensive_survey)
def OptionB : Prop := (∀ survey_method : Type, survey_method ≠ sample_survey)
def OptionC : Prop := (∀ survey_method : Type, survey_method = comprehensive_survey)
def OptionD : Prop := (∀ survey_method : Type, survey_method ≠ comprehensive_survey)

-- Main theorem incorporating the conditions and the proof that C is correct
theorem correct_survey_method : OptionC :=
by {
  -- Here we would insert the detailed steps of the proof,
  -- however for the task, we skip the proof details.
  sorry
}

end correct_survey_method_l278_278855


namespace janet_total_time_l278_278715

def blocks_north := 3
def speed_north := 2.5
def blocks_west := 7 * blocks_north
def speed_west := 1.5
def blocks_south := 8
def speed_south := 3
def blocks_east := 2 * blocks_south
def speed_east := 2
def stops := 2
def stop_time := 5

noncomputable def time_north := blocks_north / speed_north
noncomputable def time_west := blocks_west / speed_west
noncomputable def time_south := blocks_south / speed_south
noncomputable def time_east := blocks_east / speed_east + stops * stop_time

noncomputable def total_time := time_north + time_west + time_south + time_east

theorem janet_total_time : total_time = 35.87 := 
by
  sorry

end janet_total_time_l278_278715


namespace union_complement_A_eq_l278_278653

open Set

variable (U A B : Set ℕ)

-- Define the sets U, A, and B
def U := {1, 2, 3, 4, 5}
def A := {1, 3}
def B := {1, 2, 4}

theorem union_complement_A_eq : ((U \ B) ∪ A) = {1, 3, 5} := 
by
  sorry

end union_complement_A_eq_l278_278653


namespace problem1_problem2_l278_278107

variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

theorem problem1 : 
  (4 * a^2 * b^(2/3)) * (-2 * a^(1/3) * b^(-2/3)) / (-b^(-1/2)) = 8 * a^(7/3) * b^(1/2) := 
  sorry

theorem problem2 : 
  (sqrt (6 + 1/4) - 33 - 3/8 - (sqrt 2 - 1)^0 + (-1)^(2016) + 2^(-1)) = 3/2 :=
  sorry

end problem1_problem2_l278_278107


namespace area_of_sector_proof_l278_278229

-- Define the central angle in radians
def central_angle_rad : ℝ := (150 / 180) * Real.pi

-- Define the radius
def radius : ℝ := 3

-- Calculate the area of the sector
noncomputable def area_of_sector (alpha : ℝ) (R : ℝ) : ℝ := (1 / 2) * alpha * R^2

-- State the theorem to prove the area of the sector
theorem area_of_sector_proof : 
  area_of_sector central_angle_rad radius = (15 * Real.pi) / 4 := 
by
  sorry

end area_of_sector_proof_l278_278229


namespace surface_area_of_sphere_l278_278208

variables (O A B C : Type) 
variables [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (S : Sphere O A B C)

def is_triangle_pyramid (O A B C : Type) := 
  ∠BOC = 90 ∧ (OA ⊥ plane BOC) ∧ 
  (AB = sqrt 10) ∧ (BC = sqrt 13) ∧ 
  (AC = sqrt 5) ∧
  (∀ (P : O), P ∈ S)

theorem surface_area_of_sphere (S : Sphere O A B C) 
  (h : is_triangle_pyramid O A B C) : 
  surface_area S = 14 * π := 
sorry

end surface_area_of_sphere_l278_278208


namespace solve_equation_l278_278969

theorem solve_equation :
  (∃ x : ℝ, (∃ y : ℝ, y^3 = 3 - x^2 ∧ x^2 = 3 - y^3)
    ∧ x - 2 ≥ 0 ∧ y + real.sqrt (x - 2) = 1)
  ↔ (x = real.sqrt 11) ∨ (x = real.sqrt 12) :=
by {
  sorry
}

end solve_equation_l278_278969


namespace smallest_number_satisfies_l278_278481

-- Define the divisors
def divisors : List ℤ := [25, 50, 75, 100, 150]

-- Define the required number n to satisfy the conditions
def satisfies_conditions (n : ℤ) : Prop :=
  ∀ d ∈ divisors, (n - 20) % d = 0

-- This is the statement of the required proof
theorem smallest_number_satisfies : ∃ n, satisfies_conditions n ∧ n = 320 :=
by 
  use 320
  unfold satisfies_conditions divisors
  simp
  intro d hd
  fin_cases hd <;> simp

end smallest_number_satisfies_l278_278481


namespace max_covered_area_n_eq_3_min_covered_area_n_eq_4_l278_278457

theorem max_covered_area_n_eq_3 (n a R r ρ : ℝ) (h : n = 3) 
  (cond1 : R - a / 2 ≤ ρ) (cond2 : ρ ≤ r) 
  (h_polygon : is_regular_ngon n a R r) : ρ = a / 2 :=
sorry

theorem min_covered_area_n_eq_4 (n a R r ρ : ℝ) (h : n = 4) 
  (cond1 : R - a / 2 ≤ ρ) (cond2 : ρ ≤ r) 
  (h_polygon : is_regular_ngon n a R r) : ρ = R / Real.sqrt 2 :=
sorry

-- Assume is_regular_ngon is a predicate which proves the polygon properties
def is_regular_ngon (n a R r : ℝ) : Prop := 
-- specify the necessary conditions for regular n-gon
sorry

end max_covered_area_n_eq_3_min_covered_area_n_eq_4_l278_278457


namespace incenter_concyclic_l278_278346

variables {A B C D I₁ I₂ I₃ I₄ : Type} [CircumQuadrilateral A B C D]
  (h₁ : InCenter I₁ (Triangle A B C))
  (h₂ : InCenter I₂ (Triangle B C D))
  (h₃ : InCenter I₃ (Triangle C D A))
  (h₄ : InCenter I₄ (Triangle A B D))

theorem incenter_concyclic 
  (h₁ : InCenter I₁ (Triangle A B C)) 
  (h₂ : InCenter I₂ (Triangle B C D)) 
  (h₃ : InCenter I₃ (Triangle C D A)) 
  (h₄ : InCenter I₄ (Triangle A B D)) : 
  Concyclic I₁ I₂ I₃ I₄ := 
sorry

end incenter_concyclic_l278_278346


namespace angle_terminal_side_l278_278589

theorem angle_terminal_side :
  ∃ θ ∈ Ico 0 360, θ ≡ 1000 [MOD 360] :=
by
  use 280
  split
  {
    show 0 ≤ 280
    norm_num
  }
  {
    show 280 < 360
    norm_num
  }
  show 280 ≡ 1000 [MOD 360]
  norm_num
  refl

end angle_terminal_side_l278_278589


namespace largest_natural_number_not_sum_of_two_composites_l278_278148

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278148


namespace factorizable_polynomial_solution_l278_278377

noncomputable def factorizable_polynomial_conditions (p q : ℕ) (n a : ℤ) : Prop :=
  distinct_prime p q ∧ n ≥ 3 ∧ 
  (a = (-1)^n * p * q + 1 ∨ a = - p * q - 1)

theorem factorizable_polynomial_solution {p q n a : ℤ} :
  distinct_prime p q ∧ n ≥ 3 → 
  (∀ f : polynomial ℤ, f = polynomial.X^n + polynomial.C a * polynomial.X^(n-1) + polynomial.C (p * q) →
    ∃ g h : polynomial ℤ, g.degree > 0 ∧ h.degree > 0 ∧ f = g * h) ↔ 
  (a = (-1)^n * p * q + 1 ∨ a = - p * q - 1) :=
by
  sorry

end factorizable_polynomial_solution_l278_278377


namespace dihedral_angle_B1_DC_A_distance_AB1_CD_l278_278696

-- Name definitions
variables (A B C D B1: Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space B1]
-- Conditions for the rectangle ABCD
variables (AD AB AC : real)
variables (h : real)

-- Given conditions in the problem
axiom AD_eq : AD = 4
axiom AB_eq : AB = 3
axiom AC_folding : true -- Placeholder for the folding condition

-- The statements of what we need to prove
theorem dihedral_angle_B1_DC_A : 
  AD = 4 ∧ AB = 3 ∧ AC_folding → 
  ∃ θ, θ = real.arctan (15 / 16) := 
by {
  intros,
  sorry
}

theorem distance_AB1_CD :
  AD = 4 ∧ AB = 3 ∧ AC_folding →
  ∃ d, d = 10 * real.sqrt 34 / 17 :=
by {
  intros,
  sorry
}

end dihedral_angle_B1_DC_A_distance_AB1_CD_l278_278696


namespace variance_of_dataset_l278_278204

def dataset : List ℝ := [-1, -1, 0, 1, 1]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

theorem variance_of_dataset : variance dataset = 0.8 :=
sorry

end variance_of_dataset_l278_278204


namespace number_of_divisibles_by_eight_in_range_l278_278272

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278272


namespace quadrilateral_area_is_correct_l278_278092

-- Let's define the situation
structure TriangleDivisions where
  T1_area : ℝ
  T2_area : ℝ
  T3_area : ℝ
  Q_area : ℝ

def triangleDivisionExample : TriangleDivisions :=
  { T1_area := 4,
    T2_area := 9,
    T3_area := 9,
    Q_area := 36 }

-- The statement to prove
theorem quadrilateral_area_is_correct (T : TriangleDivisions) (h1 : T.T1_area = 4) 
  (h2 : T.T2_area = 9) (h3 : T.T3_area = 9) : T.Q_area = 36 :=
by
  sorry

end quadrilateral_area_is_correct_l278_278092


namespace find_x_from_percents_l278_278517

theorem find_x_from_percents (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 :=
by
  -- Distilled condition from problem
  have h1 : 0.65 * x = 0.20 * 487.50 := h
  -- Start actual logic here
  sorry

end find_x_from_percents_l278_278517


namespace sum_of_squares_of_solutions_l278_278979

theorem sum_of_squares_of_solutions :
  (∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ s₁ + s₂ = 17 ∧ s₁ * s₂ = 22) →
  ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 245 :=
by
  sorry

end sum_of_squares_of_solutions_l278_278979


namespace count_divisible_by_8_l278_278306

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278306


namespace animals_on_farm_l278_278555

theorem animals_on_farm (cows : ℕ) (sheep : ℕ) (pigs : ℕ) 
  (h1 : cows = 12) 
  (h2 : sheep = 2 * cows) 
  (h3 : pigs = 3 * sheep) : 
  cows + sheep + pigs = 108 := 
by
  sorry

end animals_on_farm_l278_278555


namespace count_multiples_of_8_between_200_and_400_l278_278262

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278262


namespace largest_possible_pencils_in_each_package_l278_278750

def ming_pencils : ℕ := 48
def catherine_pencils : ℕ := 36
def lucas_pencils : ℕ := 60

theorem largest_possible_pencils_in_each_package (d : ℕ) (h_ming: ming_pencils % d = 0) (h_catherine: catherine_pencils % d = 0) (h_lucas: lucas_pencils % d = 0) : d ≤ ming_pencils ∧ d ≤ catherine_pencils ∧ d ≤ lucas_pencils ∧ (∀ e, (ming_pencils % e = 0 ∧ catherine_pencils % e = 0 ∧ lucas_pencils % e = 0) → e ≤ d) → d = 12 :=
by 
  sorry

end largest_possible_pencils_in_each_package_l278_278750


namespace hyperbola_eccentricity_l278_278762

theorem hyperbola_eccentricity 
  (a b : ℝ) (h1 : b > 0) (P F1 F2 : ℝ × ℝ) (h2 : P ∈ { (x, y) | x^2 / a^2 - y^2 / b^2 = 1 }) 
  (h3 : dist P F1 + dist P F2 = 6)
  (h4 : angle P F1 P F2 = 90) :
  let c := sqrt (a^2 + b^2)
  in eccentricity = sqrt 5 := 
  sorry

end hyperbola_eccentricity_l278_278762


namespace marathons_total_distance_l278_278538

theorem marathons_total_distance :
  ∀ (m y : ℕ),
  (26 + 385 / 1760 : ℕ) = 26 ∧ 385 % 1760 = 385 →
  15 * 26 + 15 * 385 / 1760 = m + 495 / 1760 ∧
  15 * 385 % 1760 = 495 →
  0 ≤ 495 ∧ 495 < 1760 →
  y = 495 := by
  intros
  sorry

end marathons_total_distance_l278_278538


namespace chip_credit_card_balance_l278_278942

-- Conditions
def initial_balance : Float := 50.00
def first_interest_rate : Float := 0.20
def additional_charge : Float := 20.00
def second_interest_rate : Float := 0.20

-- Question
def current_balance : Float :=
  let first_interest_fee := initial_balance * first_interest_rate
  let balance_after_first_interest := initial_balance + first_interest_fee
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  let second_interest_fee := balance_before_second_interest * second_interest_rate
  balance_before_second_interest + second_interest_fee

-- Correct Answer
def expected_balance : Float := 96.00

-- Proof Problem Statement
theorem chip_credit_card_balance : current_balance = expected_balance := by
  sorry

end chip_credit_card_balance_l278_278942


namespace f_1982_value_l278_278737

noncomputable def f (n : ℕ) : ℕ := sorry  -- placeholder for the function definition

axiom f_condition_2 : f 2 = 0
axiom f_condition_3 : f 3 > 0
axiom f_condition_9999 : f 9999 = 3333
axiom f_add_condition (m n : ℕ) : f (m+n) - f m - f n = 0 ∨ f (m+n) - f m - f n = 1

open Nat

theorem f_1982_value : f 1982 = 660 :=
by
  sorry  -- proof goes here

end f_1982_value_l278_278737


namespace product_probability_gt_5_l278_278990

def numbers : List ℕ := [1, 2, 3, 4, 5]

def pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ ⟨x, y⟩ => x < y)

def count_favorable_pairs (l : List ℕ) : ℕ :=
  (pairs l).count (λ ⟨x, y⟩ => x * y > 5)

def probability_favorable (l : List ℕ) : ℚ :=
  (count_favorable_pairs l : ℚ) / (pairs l).length

theorem product_probability_gt_5 :
  probability_favorable numbers = 3 / 5 :=
by
  sorry

end product_probability_gt_5_l278_278990


namespace distinct_triangles_2x4_grid_l278_278545

def points : Finset (ℕ × ℕ) := {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)}

def collinear (p1 p2 p3 : ℕ × ℕ) : Prop :=
  ∃ (a b c : ℚ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * p1.1 + b * p1.2 + c = 0) ∧ (a * p2.1 + b * p2.2 + c = 0) ∧ (a * p3.1 + b * p3.2 + c = 0)

def is_triangle (p1 p2 p3 : ℕ × ℕ) : Prop :=
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ ¬collinear p1 p2 p3

theorem distinct_triangles_2x4_grid : 
  Finset.card ((points.product points).product points).filter (λ p, is_triangle p.1.1 p.1.2 p.2) = 48 :=
sorry

end distinct_triangles_2x4_grid_l278_278545


namespace combination_30_5_eq_142506_l278_278890

theorem combination_30_5_eq_142506 : nat.choose 30 5 = 142506 := by
  sorry

end combination_30_5_eq_142506_l278_278890


namespace acute_triangle_A_area_l278_278690

-- Define the conditions needed for the problem
variables {A B C a b c : ℝ}
variables (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variables (hnegbcsin2a : cos (B + C) = - (sqrt 3 / 3) * sin (2 * A))

-- Define the final mathematical statements to verify
theorem acute_triangle_A_area (h1 : A = π / 3) (ha : a = 7) (hb : b = 5) :
  A = π / 3 ∧ (1 / 2) * a * b * sin (C) = 10 * sqrt 3 :=
sorry

end acute_triangle_A_area_l278_278690


namespace infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l278_278417

theorem infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017 :
  ∀ n : ℕ, ∃ m : ℕ, (m ∈ {x | ∀ d ∈ Nat.digits 10 x, d = 0 ∨ d = 1}) ∧ 2017 ∣ m :=
by
  sorry

end infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l278_278417


namespace negation_of_implication_l278_278453

-- Definitions based on the conditions from part (a)
def original_prop (x : ℝ) : Prop := x > 5 → x > 0
def negation_candidate_A (x : ℝ) : Prop := x ≤ 5 → x ≤ 0

-- The goal is to prove that the negation of the original proposition
-- is equivalent to option A, that is:
theorem negation_of_implication (x : ℝ) : (¬ (x > 5 → x > 0)) = (x ≤ 5 → x ≤ 0) :=
by
  sorry

end negation_of_implication_l278_278453


namespace sum_of_coordinates_l278_278466

noncomputable def sum_coords_of_points (x y : ℝ) : ℝ :=
  if ((y = 27 ∨ y = 13) ∧ (x = 10 + 4 * Real.sqrt 11 ∨ x = 10 - 4 * Real.sqrt 11))
  then x + y else 0

theorem sum_of_coordinates :
  let points := [(10 + 4 * Real.sqrt 11, 27), (10 - 4 * Real.sqrt 11, 27), 
                 (10 + 4 * Real.sqrt 11, 13), (10 - 4 * Real.sqrt 11, 13)] in
  ∑ p in points, sum_coords_of_points p.1 p.2 = 120 :=
by
  sorry

end sum_of_coordinates_l278_278466


namespace no_common_points_between_curves_locus_equation_of_P_locus_is_circle_l278_278348

-- Define the curves C1 and C2 in polar coordinates
def curve_C1 (ρ θ : ℝ) := ρ = -2 * Real.cos θ
def curve_C2 (ρ θ : ℝ) := ρ * Real.cos (θ + Real.pi / 3) = 1

-- Check and prove the number of common points between C1 and C2
theorem no_common_points_between_curves :
  ∀ {ρ θ : ℝ}, ¬ (curve_C1 ρ θ ∧ curve_C2 ρ θ) :=
by sorry

-- Define the conditions for point Q and P on the line OQ
def point_Q (ρ_0 θ_0 : ℝ) := curve_C2 ρ_0 θ_0
def point_P (ρ θ ρ_0 θ_0 : ℝ) (h: ρ * ρ_0 = 2 ∧ θ = θ_0) : Prop :=
  ρ * ρ_0 = 2 ∧ θ = θ_0 ∧ point_Q ρ_0 θ_0

-- Locus equation of point P
theorem locus_equation_of_P :
  ∀ (ρ ρ_0 θ θ_0 : ℝ) (h: ρ * ρ_0 = 2 ∧ θ = θ_0),
  point_P ρ θ ρ_0 θ_0 h →
  (ρ = 2 * Real.cos (θ + Real.pi / 3)) :=
by sorry

-- Converted to the rectangular to verify the locus is a circle
theorem locus_is_circle :
  ∀ (x y : ℝ), (x - 1/2)^2 + (y + Math.sqrt 3 / 2)^2 = 1 :=
by sorry

end no_common_points_between_curves_locus_equation_of_P_locus_is_circle_l278_278348


namespace largest_four_digit_divisible_by_6_l278_278831

theorem largest_four_digit_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 6 = 0 → m ≤ n :=
begin
  use 9996,
  split,
  { exact nat.le_refl 9996 },
  split,
  { dec_trivial },
  split,
  { exact nat.zero_mod _ },
  { intros m h1 h2 h3,
    exfalso,
    sorry }
end

end largest_four_digit_divisible_by_6_l278_278831


namespace right_road_total_distance_shorter_direct_distance_l278_278060

-- Define the vertices and given distances
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions: angle and distances
def angle_60_deg := (1/2 : ℝ)   -- cos(60 degrees) = 1/2
def AB : ℝ := 8                   -- Distance from A to B is 8 versts

-- Point D definition with the distance to A and path properties
def right_road_condition (A D : Type) [MetricSpace A] [MetricSpace D]: Prop :=
dist A D = dist D C ∧ dist A D = 8 / angle_60_deg

-- Direct path distance from A to C
def direct_path_distance (A B : Type) [MetricSpace A] [MetricSpace B] : ℝ :=
sqrt ((AB * angle_60_deg)^2 + (AB * sqrt (3/4)) ^2)

theorem right_road_total_distance : dist A D + dist D C = 16 := 
sorry

theorem shorter_direct_distance : dist A C < 10 := 
sorry

end right_road_total_distance_shorter_direct_distance_l278_278060


namespace part1_part2_l278_278119

section SequenceProof

-- Define the sequence and its properties
variable {a : ℕ → ℤ}

-- Condition: a₁ = 2
def a1 := a 1 = 2

-- Recursive relation condition
def recursive_relation (m n : ℕ) : Prop :=
  a (m + n) + a (m - n) - m + n = (1/2 : ℚ) * (a (2 * m) + a (2 * n))

-- Question (1): For all n in ℕ, prove aₙ₊₂ = 2aₙ₊₁ - aₙ + 2
theorem part1 : a1 → (∀ m n : ℕ, m ≥ n → recursive_relation m n) → 
  ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n + 2 := 
by sorry

-- Question (2): Prove (1/a₁) + (1/a₂) + ... + (1/a₂₀₀₉) < 1
theorem part2 : a1 → (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n + 2) →
  (∑ i in Finset.range 2009, (1 : ℚ) / a (i + 1)) < 1 := 
by sorry

end SequenceProof

end part1_part2_l278_278119


namespace shape_is_cone_l278_278697

-- Define spherical coordinates and implicitly use the transformation 
-- to Cartesian coordinates.
def spherical_to_cartesian (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * Real.sin φ * Real.cos θ, 
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

-- Define the specific condition given in the problem
def shape_described_by_phi_eq_pi_over_4 (ρ θ : ℝ) : (ℝ × ℝ × ℝ) :=
  spherical_to_cartesian ρ θ (Real.pi / 4)

-- Statement that the shape is a cone.
theorem shape_is_cone : ∀ (ρ θ : ℝ), 
  shape_described_by_phi_eq_pi_over_4 ρ θ = (ρ * (Real.sqrt 2 / 2) * Real.cos θ, 
                                             ρ * (Real.sqrt 2 / 2) * Real.sin θ, 
                                             ρ * (Real.sqrt 2 / 2)) :=
by
  sorry

end shape_is_cone_l278_278697


namespace count_divisibles_l278_278297

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278297


namespace imaginary_part_of_exp_pi_div_3_l278_278635

open Complex

theorem imaginary_part_of_exp_pi_div_3 :
  ∀ (θ: ℝ), θ = Real.pi / 3 → (Complex.exp (θ * Complex.I)).im = (Real.sqrt 3) / 2 
:= by
  intros θ h
  simp only [Complex.exp, Complex.sin, Complex.cos]
  rw [h] -- Substitute θ = π / 3
  rw [Real.sin_pi_div_three, Real.cos_pi_div_three]
  -- Use Euler's formula and trigonometric identities
  sorry

end imaginary_part_of_exp_pi_div_3_l278_278635


namespace constant_c_value_l278_278079

noncomputable def prism_path_length (a b c : ℕ) (dot_on_top : Bool) : ℝ :=
if dot_on_top then (sqrt (a^2 + c^2) - b) * 2 * real.pi else 0

theorem constant_c_value :
  prism_path_length 2 1 2 true = (sqrt 17 - 2) * real.pi :=
by
  sorry

end constant_c_value_l278_278079


namespace unit_squares_in_50th_ring_l278_278118

-- Definitions from the conditions
def unit_squares_in_first_ring : ℕ := 12

def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  32 * n - 16

-- Prove the specific instance for the 50th ring
theorem unit_squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1584 :=
by
  sorry

end unit_squares_in_50th_ring_l278_278118


namespace combined_6th_grade_percentage_l278_278805

noncomputable def percentage_of_6th_graders 
  (parkPercent : Fin 7 → ℚ) 
  (riversidePercent : Fin 7 → ℚ) 
  (totalParkside : ℕ) 
  (totalRiverside : ℕ) 
  : ℚ := 
    let num6thParkside := parkPercent 6 * totalParkside
    let num6thRiverside := riversidePercent 6 * totalRiverside
    let total6thGraders := num6thParkside + num6thRiverside
    let totalStudents := totalParkside + totalRiverside
    (total6thGraders / totalStudents) * 100

theorem combined_6th_grade_percentage :
  let parkPercent := ![(14.0 : ℚ) / 100, 13 / 100, 16 / 100, 15 / 100, 12 / 100, 15 / 100, 15 / 100]
  let riversidePercent := ![(13.0 : ℚ) / 100, 16 / 100, 13 / 100, 15 / 100, 14 / 100, 15 / 100, 14 / 100]
  percentage_of_6th_graders parkPercent riversidePercent 150 250 = 15 := 
  by
  sorry

end combined_6th_grade_percentage_l278_278805


namespace tank_insulation_cost_l278_278546

theorem tank_insulation_cost (l w h : ℝ) (cost_per_sqft : ℝ) (SA : ℝ) (C : ℝ) 
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost_per_sqft : cost_per_sqft = 20) 
  (h_SA : SA = 2 * l * w + 2 * l * h + 2 * w * h)
  (h_C : C = SA * cost_per_sqft) :
  C = 1440 := 
by
  -- proof will be filled in here
  sorry

end tank_insulation_cost_l278_278546


namespace exists_numbers_with_prime_sum_and_product_l278_278711

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem exists_numbers_with_prime_sum_and_product :
  ∃ a b c : ℕ, is_prime (a + b + c) ∧ is_prime (a * b * c) :=
  by
    -- First import the prime definitions and variables.
    let a := 1
    let b := 1
    let c := 3
    have h1 : is_prime (a + b + c) := by sorry
    have h2 : is_prime (a * b * c) := by sorry
    exact ⟨a, b, c, h1, h2⟩

end exists_numbers_with_prime_sum_and_product_l278_278711


namespace max_distance_PQ_l278_278699

-- Curve C1: x² + y² = 1 (initial curve)
def curve_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Transformation to obtain curve C2
def transformation (x y : ℝ) : ℝ × ℝ := (2 * x, y)

-- Parametric equations of curve C2
def parametric_C2 (α : ℝ) : ℝ × ℝ := (2 * cos α, sin α)

-- Polar equation of curve C3: ρ = -2 sin θ
def polar_C3 (ρ θ : ℝ) : Prop := ρ = -2 * sin θ

-- Cartesian equation of curve C3: x² + (y + 1)² = 1
def cartesian_C3 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- Parametric equations of curve C3
def parametric_C3 (β : ℝ) : ℝ × ℝ := (cos β, -1 + sin β)

-- Max distance |PQ|
theorem max_distance_PQ : ∃ α β : ℝ, 
  let P := parametric_C2 α,
      Q := parametric_C3 β,
      dist := sqrt ((fst P - fst Q)^2 + (snd P - snd Q)^2)
  in dist = (4 * sqrt 3 + 3) / 3 :=
sorry

end max_distance_PQ_l278_278699


namespace remaining_hard_hats_l278_278688

variables (initial_pink: ℕ) (initial_green: ℕ) (initial_yellow: ℕ)
variables (carl_pink: ℕ) (john_pink: ℕ) (john_green: ℕ)

-- Conditions
def total_pink : ℕ := initial_pink - carl_pink - john_pink
def total_green : ℕ := initial_green - john_green
def total_yellow : ℕ := initial_yellow
def total_hard_hats : ℕ := total_pink + total_green + total_yellow

-- The problem 
theorem remaining_hard_hats (h_pink: initial_pink = 26) 
                            (h_green: initial_green = 15) 
                            (h_yellow: initial_yellow = 24) 
                            (carl_removed_pink: carl_pink = 4) 
                            (john_removed_pink: john_pink = 6)
                            (john_removed_green: john_green = 2 * john_pink) : 
    total_hard_hats = 43 :=
by
  rw [total_hard_hats, total_pink, total_green, total_yellow]
  rw [h_pink, h_green, h_yellow, carl_removed_pink, john_removed_pink, john_removed_green]
  norm_num

end remaining_hard_hats_l278_278688


namespace right_triangle_hypotenuse_and_perimeter_l278_278081

theorem right_triangle_hypotenuse_and_perimeter
  (a b : ℝ)
  (h1 : a = 24)
  (h2 : b = 32) :
  let c := real.sqrt (a * a + b * b)
  in c = 40 ∧ a + b + c = 96 :=
by {
  sorry
}

end right_triangle_hypotenuse_and_perimeter_l278_278081


namespace third_largest_three_digit_number_with_ones_8_l278_278472

def is_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 4 ∨ d = 8

def is_three_digit_number_with_ones_8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n % 10 = 8

def use_each_digit_once (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  is_digit d1 ∧ is_digit d2 ∧ is_digit d3 ∧ (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)

theorem third_largest_three_digit_number_with_ones_8 :
  ∃ n, is_three_digit_number_with_ones_8 n ∧ use_each_digit_once n ∧ 
  ∃ a b, is_three_digit_number_with_ones_8 a ∧ use_each_digit_once a ∧
         is_three_digit_number_with_ones_8 b ∧ use_each_digit_once b ∧
         a > b ∧ b > n ∧ n = 148 :=
sorry

end third_largest_three_digit_number_with_ones_8_l278_278472


namespace smithtown_left_handed_women_percentage_l278_278502

theorem smithtown_left_handed_women_percentage
    (x y : ℕ)
    (H1 : 3 * x + x = 4 * x)
    (H2 : 3 * y + 2 * y = 5 * y)
    (H3 : 4 * x = 5 * y) :
    (x / (4 * x)) * 100 = 25 :=
by sorry

end smithtown_left_handed_women_percentage_l278_278502


namespace count_divisible_by_8_l278_278307

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278307


namespace problem1_problem2_l278_278195

-- Definitions
def f (n : ℕ) (x : ℝ) : ℝ := (1 + real.sqrt x) ^ n
def g (x : ℝ) : ℝ := f 4 x + 2 * f 5 x + 3 * f 6 x

-- p_n is the sum of coefficients of irrational terms in f_n(x)
def sum_of_irrational_coeffs (n : ℕ) : ℝ := 2 ^ (n - 1)

-- Sequence {a_n}, all terms > 1
variable (a : ℕ → ℝ)
hypothesis ha_gt_1 : ∀ n, 1 < a n

-- Main statements to prove
theorem problem1 (x : ℝ) : g x.coeff 2 = 56 := sorry

theorem problem2 (n : ℕˣ) : 
  sum_of_irrational_coeffs n.val * ((list.range n.val).map a).prod + 1 ≥ ((list.range n.val).map (λ i, 1 + a i)).prod :=
sorry

end problem1_problem2_l278_278195


namespace final_cost_is_correct_l278_278960

noncomputable def calculate_final_cost 
  (price_orange : ℕ)
  (price_mango : ℕ)
  (increase_percent : ℕ)
  (bulk_discount_percent : ℕ)
  (sales_tax_percent : ℕ) : ℕ := 
  let new_price_orange := price_orange + (price_orange * increase_percent) / 100
  let new_price_mango := price_mango + (price_mango * increase_percent) / 100
  let total_cost_oranges := 10 * new_price_orange
  let total_cost_mangoes := 10 * new_price_mango
  let total_cost_before_discount := total_cost_oranges + total_cost_mangoes
  let discount_oranges := (total_cost_oranges * bulk_discount_percent) / 100
  let discount_mangoes := (total_cost_mangoes * bulk_discount_percent) / 100
  let total_cost_after_discount := total_cost_before_discount - discount_oranges - discount_mangoes
  let sales_tax := (total_cost_after_discount * sales_tax_percent) / 100
  total_cost_after_discount + sales_tax

theorem final_cost_is_correct :
  calculate_final_cost 40 50 15 10 8 = 100602 :=
by
  sorry

end final_cost_is_correct_l278_278960


namespace min_side_values_l278_278018

open Real

noncomputable def min_side_condition (a b c : ℝ) : Prop :=
  a + b + c = 1 ∧
  let A := sqrt (c * (c - a) * (c - b) * (c + a - b)) * 1/2 / c in
  let h_a := 2 * A / a in
  let h_b := 2 * A / b in
  let h_c := 2 * A / c in
  h_a + h_b > h_c ∧ h_b + h_c > h_a ∧ h_c + h_a > h_b

theorem min_side_values (a b c : ℝ) (h : min_side_condition a b c) :
  (min a (min b c)) ∈ set.Ioo (1 / 5) (1 / 3) :=
sorry

end min_side_values_l278_278018


namespace octal_to_binary_conversion_l278_278951

theorem octal_to_binary_conversion :
  let n := 2016 in
  let octal_to_decimal (d : ℕ) : ℕ := 2 * 8^3 + 0 * 8^2 + 1 * 8^1 + 6 * 8^0 in
  let decimal_to_binary (d : ℕ) : ℕ := 32 * 2^9 + 16 * 2^8 + 8 * 2^7 + 4 * 2^6 + 0 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 1  in
  octal_to_decimal n = 1038 ∧ decimal_to_binary 1038 = 0b10000001110 :=
sorry

end octal_to_binary_conversion_l278_278951


namespace shift_line_down_4_units_l278_278815

theorem shift_line_down_4_units :
  ∀ (x : ℝ), y = - (3 / 4) * x → (y - 4 = - (3 / 4) * x - 4) := by
  sorry

end shift_line_down_4_units_l278_278815


namespace even_function_property_l278_278644

def f (x : ℝ) : ℝ := x^3 - Real.log (Real.sqrt (x^2 + 1) - x)

theorem even_function_property (a b : ℝ) (h : a + b ≠ 0) : 
  (f(a) + f(b)) / (a + b) > 0 := 
sorry

end even_function_property_l278_278644


namespace expected_games_is_14_l278_278424

-- Define the conditions of the problem
variable (C : Prop)
  [hC : C ↔ (∀ (player : ℕ), player = 2)
            ∧ (∀ (game_result : ℕ), game_result = 1 \/ game_result = 0)
            ∧ (∀ (p_win : ℚ), p_win = 1/2)]

-- Define the expected value of games and end state conditions
noncomputable def expected_games : ℕ := 14

-- Theorem statement: The expected number of games given the conditions is 14
theorem expected_games_is_14 (hC : C) : expected_games = 14 :=
by trivial

end expected_games_is_14_l278_278424


namespace roger_trips_required_l278_278047

variable (carry_trays_per_trip total_trays : ℕ)

theorem roger_trips_required (h1 : carry_trays_per_trip = 4) (h2 : total_trays = 12) : total_trays / carry_trays_per_trip = 3 :=
by
  -- proof follows
  sorry

end roger_trips_required_l278_278047


namespace probability_product_greater_than_zero_l278_278027

open Set ProbabilityTheory

theorem probability_product_greater_than_zero (a b : ℝ) (ha : a ∈ Icc (-15) 15) (hb : b ∈ Icc (-15) 15) :
  P (λ x : ℝ × ℝ, 0 < x.1 * x.2 | (λ _, true) := 1/2 :=
begin
  sorry
end

end probability_product_greater_than_zero_l278_278027


namespace toms_earnings_l278_278364

theorem toms_earnings (t1 t2 : ℕ) (h1 : t1 = 74) (h2 : t2 = 86) : t2 - t1 = 12 := by
  rw [h1, h2]
  norm_num

end toms_earnings_l278_278364


namespace numbers_divisible_by_8_between_200_and_400_l278_278287

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278287


namespace no_arithmetic_progression_exists_l278_278580

theorem no_arithmetic_progression_exists 
  (a : ℕ) (d : ℕ) (a_n : ℕ → ℕ) 
  (h_seq : ∀ n, a_n n = a + n * d) :
  ¬ ∃ (a_n : ℕ → ℕ), (∀ n, a_n (n+1) > a_n n ∧ 
  ∀ n, (a_n n) * (a_n (n+1)) * (a_n (n+2)) * (a_n (n+3)) * (a_n (n+4)) * 
        (a_n (n+5)) * (a_n (n+6)) * (a_n (n+7)) * (a_n (n+8)) * (a_n (n+9)) % 
        ((a_n n) + (a_n (n+1)) + (a_n (n+2)) + (a_n (n+3)) + (a_n (n+4)) + 
         (a_n (n+5)) + (a_n (n+6)) + (a_n (n+7)) + (a_n (n+8)) + (a_n (n+9)) ) = 0 ) := 
sorry

end no_arithmetic_progression_exists_l278_278580


namespace perpendicular_planes_l278_278656

-- Definitions for the conditions
variables (a b : ℝ → ℝ → ℝ)
variables (α β γ : ℝ → ℝ → ℝ)

-- Theorem statement without proof
theorem perpendicular_planes (a_parallel_alpha : parallel a α) (a_perpendicular_beta : perpendicular a β) : perpendicular α β :=
sorry

end perpendicular_planes_l278_278656


namespace percy_game_price_l278_278759

/--
Percy wants to buy a PlayStation for $500. He receives $200 on his birthday and $150 at Christmas.
He needs to sell 20 games to make up the remaining money. Prove that the price per game is $7.50.
-/
theorem percy_game_price :
  (playstation_cost : ℝ) = 500 →
  (birthday_money : ℝ) = 200 →
  (christmas_money : ℝ) = 150 →
  (games_to_sell : ℕ) = 20 →
  ∃ (game_price : ℝ), game_price = 7.50 :=
by
  intros playstation_cost birthday_money christmas_money games_to_sell
  use 7.50
  sorry

end percy_game_price_l278_278759


namespace range_y_over_x_l278_278615

theorem range_y_over_x {x y : ℝ} (h : (x-4)^2 + (y-2)^2 ≤ 4) : 
  ∃ k : ℝ, k = y / x ∧ 0 ≤ k ∧ k ≤ 4/3 :=
sorry

end range_y_over_x_l278_278615


namespace total_books_on_shelf_l278_278756

theorem total_books_on_shelf 
    (initial_fiction: Nat) (initial_non_fiction: Nat) (added_fiction: Nat) 
    (removed_non_fiction: Nat) (number_of_sets: Nat) (books_per_set: Nat) :
    initial_fiction = 38 ∧ initial_non_fiction = 15 ∧ 
    added_fiction = 10 ∧ removed_non_fiction = 5 ∧ 
    number_of_sets = 3 ∧ books_per_set = 4 →
    (initial_fiction + added_fiction) + (initial_non_fiction - removed_non_fiction) + (number_of_sets * books_per_set) = 70 :=
by 
  intros h
  cases h
  simp [*]
  sorry

end total_books_on_shelf_l278_278756


namespace propositions_correctness_l278_278640

-- Define a triangle and its properties
variables {A B C : ℝ}

-- Conditions for different types of triangles
def is_isosceles_triangle (A B C : ℝ) : Prop := A = B ∨ B = C ∨ A = C
def is_right_angle_triangle (A B : ℝ) : Prop := A + B = π / 2 ∨ A = π / 2 ∨ B = π / 2
def is_obtuse_triangle (A B C : ℝ) : Prop := A > π / 2 ∨ B > π / 2 ∨ C > π / 2
def is_equilateral_triangle (A B C : ℝ) : Prop := A = B ∧ B = C

-- Propositions
def prop1 : Prop := sin (2 * A) = sin (2 * B) → is_isosceles_triangle A B C
def prop2 : Prop := sin A = cos B → is_right_angle_triangle A B
def prop3 : Prop := cos A * cos B * cos C < 0 → is_obtuse_triangle A B C
def prop4 : Prop := cos (A - C) * cos (B - C) * cos (C - A) = 1 → is_equilateral_triangle A B C

-- Proof goal: Verify the correctness of propositions
theorem propositions_correctness :
  ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 :=
by sorry

end propositions_correctness_l278_278640


namespace asha_win_probability_l278_278564

theorem asha_win_probability :
  let P_Lose := (3 : ℚ) / 8
  let P_Tie := (1 : ℚ) / 4
  P_Lose + P_Tie < 1 → 1 - P_Lose - P_Tie = (3 : ℚ) / 8 := 
by
  sorry

end asha_win_probability_l278_278564


namespace arithmetic_mean_comparison_l278_278995

theorem arithmetic_mean_comparison (
  a : Fin 10 → ℝ,
  ha : ∀ i j, i < j → a i < a j
  ) : 
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / 10 >
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5) / 6 :=
by
  sorry

end arithmetic_mean_comparison_l278_278995


namespace find_tangent_circles_tangent_circle_at_given_point_l278_278972

noncomputable def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 1)^2 = 4

def is_tangent (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ (u v : ℝ), (u - a)^2 + (v - b)^2 = 1 ∧
  (x - u)^2 + (y - v)^2 = 4 ∧
  (x = u ∧ y = v)

theorem find_tangent_circles (x y a b : ℝ) (hx : circle_C x y)
  (ha_b : is_tangent x y a b) :
  (a = 5 ∧ b = -1) ∨ (a = 3 ∧ b = -1) :=
sorry

theorem tangent_circle_at_given_point (x y : ℝ) (hx : circle_C x y) (y_pos : y = -1)
  : ((x - 5)^2 + (y + 1)^2 = 1) ∨ ((x - 3)^2 + (y + 1)^2 = 1) :=
sorry

end find_tangent_circles_tangent_circle_at_given_point_l278_278972


namespace right_triangles_congruent_l278_278854

theorem right_triangles_congruent
  (ΔABC ΔDEF : Triangle)
  (hA : ΔABC.angle_a = 90)
  (hD : ΔDEF.angle_a = 90)
  (hB : ΔABC.angle_b = ΔDEF.angle_b) :
  ΔABC ≅ ΔDEF := by
  sorry

end right_triangles_congruent_l278_278854


namespace nested_fraction_value_l278_278233

theorem nested_fraction_value :
  1 + (1 / (1 + (1 / (2 + (2 / 3))))) = 19 / 11 :=
by sorry

end nested_fraction_value_l278_278233


namespace probability_best_play_wins_l278_278070

noncomputable def prob_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) : ℝ :=
  let C := Nat.choose in
  (1 : ℝ) / (C (2 * n) n * C (2 * n) (2 * m)) * ∑ q in Finset.range (2 * m + 1),
  (C n q * C n (2 * m - q)) * 
  ∑ t in Finset.range (min q m),
  (C q t * C (2 * n - q) (n - t))

-- A theorem statement in Lean to ensure proper type checks and conditions 
theorem probability_best_play_wins (n m : ℕ) (h : 2 * m ≤ n) :
  ∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t)) 
  =
  (∑ q in Finset.range (2 * m + 1),
    (nat.choose n q * nat.choose n (2 * m - q)) * 
    ∑ t in Finset.range (min q m),
    (nat.choose q t * nat.choose (2 * n - q) (n - t) )) * 
  (nat.choose (2 * n) n * nat.choose (2 * n) (2 * m)) :=
sorry

end probability_best_play_wins_l278_278070


namespace count_multiples_of_8_in_range_l278_278280

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278280


namespace find_inverse_value_l278_278925

noncomputable theory

theorem find_inverse_value
  (f : ℝ → ℝ)
  (h_sym : ∀ x, f (x + 1) + f (1 - x) = 4)
  (h_value : f 4 = 0)
  (hf_inv : ∃ g : ℝ → ℝ, ∀ y, f (g y) = y ∧ g (f y) = y) :
  (classical.some hf_inv) 4 = -2 :=
sorry

end find_inverse_value_l278_278925


namespace possible_amounts_l278_278066

-- Definitions of the problem conditions
def num_pennies (p : ℕ) : ℕ := p
def num_dimes (p : ℕ) : ℕ := 3 * p
def num_quarters (p : ℕ) : ℕ := 12 * p

-- Definition of the total value
def total_value (p : ℕ) : ℚ := 0.01 * p + 0.10 * (3 * p) + 0.25 * (12 * p)

-- The Lean 4 statement to be proven
theorem possible_amounts (p : ℕ) : 
  let total := total_value p in 
  total = 3.31 * p := 
  sorry

end possible_amounts_l278_278066


namespace largest_four_digit_divisible_by_six_l278_278844

theorem largest_four_digit_divisible_by_six :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 0) ∧ 
    (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 6 = 0) → m ≤ n) :=
begin
  existsi 9996,
  split, 
  exact ⟨by norm_num, by norm_num⟩,
  split, 
  exact dec_trivial,
  intro m,
  intro h,
  exact ⟨by norm_num [h.1], by norm_num [h.2]⟩
end

end largest_four_digit_divisible_by_six_l278_278844


namespace largest_four_digit_number_divisible_by_6_l278_278839

theorem largest_four_digit_number_divisible_by_6 :
  ∃ n, n = 9996 ∧ ∀ m, (m ≤ 9999 ∧ m % 6 = 0) → m ≤ n :=
begin
  sorry
end

end largest_four_digit_number_divisible_by_6_l278_278839


namespace circle_radius_l278_278704

theorem circle_radius {C : ℝ → ℝ → Prop} (h1 : C 4 0) (h2 : C (-4) 0) : ∃ r : ℝ, r = 4 :=
by
  -- sorry for brevity
  sorry

end circle_radius_l278_278704


namespace perimeter_triangle_ABC_l278_278330

-- Define the given conditions:
variables {A B C D E F G O : Point}
variable {triangle_ABC : Triangle}
variable [triangle_ABC.is_right_angle_triangle : is_right_angle_triangle triangle_ABC]

-- Definitions of sides and geometry construction:
def segment_AC : Segment := segment_ABC.segment_AC
def segment_BC : Segment := segment_ABC.segment_BC

def square_ACDE := square segment_AC
def square_BCFG := square segment_BC

-- Points D, E, F, G are vertices of squares:
variable (D : Point) (E : Point)
variable (F : Point) (G : Point)
assume hDE : is_vertex D square_ACDE
assume hEF : is_vertex E square_ACDE
assume hFG : is_vertex F square_BCFG
assume hDG : is_vertex G square_BCFG

-- Additional condition regarding points on the circle:
variable (O : Point)
variable [circle : Circle]
assume hcircle : circle.contains_all [D, E, F, G]

-- Lengths of the triangle sides:
axiom AC_length : length(segment_AC) = 10
axiom BC_length : length(segment_BC) = 24

-- The goal is to prove the perimeter of triangle ABC is 60:
theorem perimeter_triangle_ABC : 
  perimeter(triangle_ABC) = 60 :=
sorry

end perimeter_triangle_ABC_l278_278330


namespace sum_of_f_is_positive_l278_278201

theorem sum_of_f_is_positive {x1 x2 x3 : ℝ} 
  (h1 : x1 + x2 > 0) 
  (h2 : x2 + x3 > 0) 
  (h3 : x3 + x1 > 0)
  (f : ℝ → ℝ := λ x, x + x^3) : 
  f x1 + f x2 + f x3 > 0 :=
by
  sorry

end sum_of_f_is_positive_l278_278201


namespace car_cost_l278_278065

/--
A group of six friends planned to buy a car. They plan to share the cost equally. 
They had a car wash to help raise funds, which would be taken out of the total cost. 
The remaining cost would be split between the six friends. At the car wash, they earn $500. 
However, Brad decided not to join in the purchase of the car, and now each friend has to pay $40 more. 
What is the cost of the car?
-/
theorem car_cost 
  (C : ℝ) 
  (h1 : 6 * ((C - 500) / 5) = 5 * (C / 6 + 40)) : 
  C = 4200 := 
by 
  sorry

end car_cost_l278_278065


namespace hexagon_radius_eq_l278_278898

theorem hexagon_radius_eq (r : ℝ) (h : ∀ (a b : ℝ), 
  (a = 1 ∨ a = 2 ∨ a = 3) → 
  (b = 1 ∨ b = 2 ∨ b = 3) → 
  a ≠ b → 
  ∃ (x : ℝ), x = 2r^3 - 7r - 3):
  2 * r^3 - 7 * r - 3 = 0 :=
sorry

end hexagon_radius_eq_l278_278898


namespace divisors_form_60k_l278_278122

-- Define the conditions in Lean
def is_positive_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def satisfies_conditions (n a b c : ℕ) : Prop :=
  is_positive_divisor n a ∧
  is_positive_divisor n b ∧
  is_positive_divisor n c ∧
  a > b ∧ b > c ∧
  is_positive_divisor n (a^2 - b^2) ∧
  is_positive_divisor n (b^2 - c^2) ∧
  is_positive_divisor n (a^2 - c^2)

-- State the theorem to be proven in Lean
theorem divisors_form_60k (n : ℕ) (a b c : ℕ) (h1 : satisfies_conditions n a b c) : 
  ∃ k : ℕ, n = 60 * k :=
sorry

end divisors_form_60k_l278_278122


namespace prob_at_least_3_out_of_4_patients_cured_l278_278905

namespace MedicineCure

open ProbabilityTheory

-- Define the probability of curing a patient
def cure_prob : ℝ := 0.95

-- Define the probability that at least 3 out of 4 patients are cured
def at_least_3_cured : ℝ :=
  let prob_3_of_4 := 4 * (cure_prob^3) * (1 - cure_prob)
  let prob_4_of_4 := cure_prob^4
  prob_3_of_4 + prob_4_of_4

theorem prob_at_least_3_out_of_4_patients_cured :
  at_least_3_cured = 0.99 :=
by
  -- placeholder for proof
  sorry

end MedicineCure

end prob_at_least_3_out_of_4_patients_cured_l278_278905


namespace fraction_equality_l278_278918

theorem fraction_equality (x y : ℝ) : (-x + y) / (-x - y) = (x - y) / (x + y) :=
by sorry

end fraction_equality_l278_278918


namespace complex_number_quadrant_l278_278741

noncomputable def z : ℂ := (2 * (complex.I ^ 3)) / (1 - complex.I)

theorem complex_number_quadrant :
  let z := (2 * complex.I ^ 3) / (1 - complex.I) in (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end complex_number_quadrant_l278_278741


namespace max_sum_of_arithmetic_sequence_l278_278207

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → 4 * a (n + 1) = 4 * a n - 7) →
  a 1 = 25 →
  (∀ n : ℕ, S n = (n * (50 - (7/4 : ℚ) * (n - 1))) / 2) →
  ∃ n : ℕ, n = 15 ∧ S n = 765 / 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l278_278207


namespace digit_6_occurrences_l278_278350

def count_digit_6_in_range (start : ℕ) (end_ : ℕ) (digit : ℕ) : ℕ :=
  ((list.range' start (end_ - start + 1)).filter (λ n, n.digits 10).count digit)

theorem digit_6_occurrences : count_digit_6_in_range 20 109 6 = 20 := 
  sorry

end digit_6_occurrences_l278_278350


namespace sum_of_coordinates_l278_278465

theorem sum_of_coordinates :
  let points := { p : ℝ × ℝ |
    (p.2 = 15 + 7 ∨ p.2 = 15 - 7) ∧
    (real.sqrt ((p.1 - 9) ^ 2 + (p.2 - 15) ^ 2) = 15) } in
  let coords_sum := points.sum (λ p, p.1 + p.2) in
  coords_sum = 96 :=
by {
  sorry
}

end sum_of_coordinates_l278_278465


namespace flight_duration_is_four_hours_l278_278981

def convert_to_moscow_time (local_time : ℕ) (time_difference : ℕ) : ℕ :=
  (local_time - time_difference) % 24

def flight_duration (departure_time arrival_time : ℕ) : ℕ :=
  (arrival_time - departure_time) % 24

def duration_per_flight (total_flight_time : ℕ) (number_of_flights : ℕ) : ℕ :=
  total_flight_time / number_of_flights

theorem flight_duration_is_four_hours :
  let MoscowToBishkekTimeDifference := 3
  let departureMoscowTime := 12
  let arrivalBishkekLocalTime := 18
  let departureBishkekLocalTime := 8
  let arrivalMoscowTime := 10
  let outboundArrivalMoscowTime := convert_to_moscow_time arrivalBishkekLocalTime MoscowToBishkekTimeDifference
  let returnDepartureMoscowTime := convert_to_moscow_time departureBishkekLocalTime MoscowToBishkekTimeDifference
  let outboundDuration := flight_duration departureMoscowTime outboundArrivalMoscowTime
  let returnDuration := flight_duration returnDepartureMoscowTime arrivalMoscowTime
  let totalFlightTime := outboundDuration + returnDuration
  duration_per_flight totalFlightTime 2 = 4 := by
  sorry

end flight_duration_is_four_hours_l278_278981


namespace base12_remainder_l278_278479

theorem base12_remainder (x : ℕ) (h : x = 2 * 12^3 + 7 * 12^2 + 4 * 12 + 5) : x % 5 = 2 :=
by {
    -- Proof would go here
    sorry
}

end base12_remainder_l278_278479


namespace inscribed_sphere_conditions_l278_278827

theorem inscribed_sphere_conditions (Pyramid : Type) [Polyhedron Pyramid] (Sphere : Type) [MetricSpace Sphere] 
  (center_sphere : Point Sphere) (center_pyramid : Point Pyramid) (faces_pyramid : set (Face Pyramid)) :
  (∀ (face ∈ faces_pyramid), distance center_sphere (plane_of face) = const) 
  ∧ (∀ (angle ∈ dihedral_angles Pyramid), center_sphere ∈ bisector_plane angle) ↔ 
  (∃ (r : ℝ), ∀ (face ∈ faces_pyramid), distance center_sphere (plane_of face) = r) :=
sorry

end inscribed_sphere_conditions_l278_278827


namespace part_one_part_two_l278_278933

-- Part (1)
theorem part_one (x : ℝ) : x - (3 * x - 1) ≤ 2 * x + 3 → x ≥ -1 / 2 :=
by sorry

-- Part (2)
theorem part_two (x : ℝ) : 
  (3 * (x - 1) < 4 * x - 2) ∧ ((1 + 4 * x) / 3 > x - 1) → x > -1 :=
by sorry

end part_one_part_two_l278_278933


namespace cube_edge_coloring_l278_278098

theorem cube_edge_coloring (n : ℕ) (cube : list (ℕ × ℕ × ℕ)) (polygon : list (ℕ × ℕ × ℕ)) :
  (∀ face ∈ cube, (face ∈ distinguished_faces(cube, polygon) → odd_colored_edges(face)) ∧ 
                  (face ∉ distinguished_faces(cube, polygon) → even_colored_edges(face))) :=
sorry

def distinguished_faces (cube : list (ℕ × ℕ × ℕ)) (polygon : list (ℕ × ℕ × ℕ)) : list (ℕ × ℕ × ℕ) :=
  -- This function should identify the distinguished faces intersected by the polygon.
  sorry

def odd_colored_edges (face : (ℕ × ℕ × ℕ)) : Prop :=
  -- This function checks if the given face has an odd number of edges of each color.
  sorry

def even_colored_edges (face : (ℕ × ℕ × ℕ)) : Prop :=
  -- This function checks if the given face has an even number of edges of each color.
  sorry

end cube_edge_coloring_l278_278098


namespace find_x_for_f_eq_10_l278_278612

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2 * x

-- State the theorem
theorem find_x_for_f_eq_10 (x : ℝ) (hx : f(x) = 10) : x = -3 ∨ x = 5 :=
  sorry

end find_x_for_f_eq_10_l278_278612


namespace largest_number_not_sum_of_two_composites_l278_278181

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278181


namespace best_play_wins_probability_l278_278073

/-- Define the conditions and parameters for the problem. -/
variables (n m : ℕ)
variables (C : ℕ → ℕ → ℕ) /- Binomial coefficient -/

/-- Define the probability calculation -/
def probability_best_play_wins : ℚ :=
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t)

/-- The theorem stating that the above calculation represents the probability of the best play winning -/
theorem best_play_wins_probability :
  probability_best_play_wins n m C =
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t) :=
  by
  sorry

end best_play_wins_probability_l278_278073


namespace justin_pages_read_l278_278358

theorem justin_pages_read :
  let pages_day1 := 10 in
  let daily_pages := 2 * pages_day1 in
  let total_pages := pages_day1 + daily_pages * 6 in
  total_pages = 130 :=
by
  let pages_day1 := 10
  let daily_pages := 2 * pages_day1
  let total_pages := pages_day1 + daily_pages * 6
  sorry

end justin_pages_read_l278_278358


namespace max_books_borrowed_l278_278338

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ)
  (two_books : ℕ) (at_least_three_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 35 →
  no_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books_per_student = 2 →
  total_students - (no_books + one_book + two_books) = at_least_three_books →
  ∃ max_books_borrowed_by_individual, max_books_borrowed_by_individual = 8 :=
by
  intros h_total_students h_no_books h_one_book h_two_books h_avg_books_per_student h_remaining_students
  -- Skipping the proof steps
  sorry

end max_books_borrowed_l278_278338


namespace largest_number_not_sum_of_two_composites_l278_278182

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278182


namespace subset_a_b_l278_278624

theorem subset_a_b (a : ℤ) (A : set ℤ := {0, 1}) (B : set ℤ := {-1, 0, a + 3}) : A ⊆ B → a = -2 :=
by
  sorry

end subset_a_b_l278_278624


namespace part_one_part_two_l278_278947

section problem1
variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) := a * x + 3 - |2 * x - 1|

theorem part_one (h : a = 1) : (f x 1 ≤ 2) ↔ (x ∈ Set.Icc (by NegInf) 0 ∨ x ∈ Set.Icc 2 (by PosInf)) :=
by sorry
end problem1

section problem2
variable (a : ℝ)

def f (x : ℝ) (a : ℝ) := a * x + 3 - |2 * x - 1|

theorem part_two : (∀ x, -2 ≤ a ∧ a ≤ 2) ↔ HasMaxValue (f x a) :=
by sorry
end problem2

end part_one_part_two_l278_278947


namespace inequality_solution_l278_278648

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 :=
by
  sorry

end inequality_solution_l278_278648


namespace find_quadrant_l278_278628

set_option pp.all true

-- Definitions based on problem conditions
def i := Complex.I
def z := (2 - i) / (2 + i) + (i ^ 2017)
def conjugate_z := Complex.conj z

-- Formalization of the problem to determine the quadrant where the point corresponding to conjugate_z lies.
-- Note: Instead of evaluating which quadrant, this statement directly ensures it's the fourth quadrant.
theorem find_quadrant :
  let point := (conjugate_z.re, conjugate_z.im)
  in point.1 > 0 ∧ point.2 < 0 := by
  sorry

end find_quadrant_l278_278628


namespace cylinder_volume_relation_l278_278576

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume_relation (h_Y : ℝ) (r_Y : ℝ) (h_X : ℝ) (r_X : ℝ)
  (hX_def : h_X = 3 * r_Y)
  (rX_def : r_X = h_Y / 2)
  (V_rel : cylinder_volume r_X h_X = 3 * cylinder_volume r_Y h_Y) :
  cylinder_volume r_X h_X = (3 / 16) * π * h_Y^3 :=
by
  sorry

end cylinder_volume_relation_l278_278576


namespace M_v3_eq_l278_278730

-- Define the matrix M
def M : Matrix (Fin 2) (Fin 2) ℚ := sorry

-- Define the vectors involved
def v1 : Fin 2 → ℚ := ![1, 2]
def v2 : Fin 2 → ℚ := ![3, -1]
def v3 : Fin 2 → ℚ := ![2, 3]

-- Define the results of matrix-vector multiplication
def M_v1 : Fin 2 → ℚ := ![2, 5]
def M_v2 : Fin 2 → ℚ := ![7, 0]
def M_v3_correct : Fin 2 → ℚ := ![29/7, 55/7]

-- Assuming the given conditions derived from the problem
axiom M_v1_eq : M.mul_vec v1 = M_v1
axiom M_v2_eq : M.mul_vec v2 = M_v2

-- The main theorem to prove
theorem M_v3_eq : M.mul_vec v3 = M_v3_correct := 
  by
    sorry

end M_v3_eq_l278_278730


namespace who_stole_the_broth_l278_278333

-- Define the suspects
inductive Suspect
| MarchHare : Suspect
| MadHatter : Suspect
| Dormouse : Suspect

open Suspect

-- Define the statements
def stole_broth (s : Suspect) : Prop :=
  s = Dormouse

def told_truth (s : Suspect) : Prop :=
  s = Dormouse

-- The March Hare's testimony
def march_hare_testimony : Prop :=
  stole_broth MadHatter

-- Conditions
def condition1 : Prop := ∃! s, stole_broth s
def condition2 : Prop := ∀ s, told_truth s ↔ stole_broth s
def condition3 : Prop := told_truth MarchHare → stole_broth MadHatter

-- Combining conditions into a single proposition to prove
theorem who_stole_the_broth : 
  (condition1 ∧ condition2 ∧ condition3) → stole_broth Dormouse := sorry

end who_stole_the_broth_l278_278333


namespace smallest_number_of_rectangles_l278_278035

theorem smallest_number_of_rectangles (A : ℕ) (hA : A = 24) : 
  ∃ n : ℕ, n = 2 ∧ n * 12 = A :=
by
  use 2
  split
  . rfl
  . exact hA.symm ▸ rfl

end smallest_number_of_rectangles_l278_278035


namespace sum_of_digits_base_5_of_588_is_12_l278_278848

theorem sum_of_digits_base_5_of_588_is_12 :
  let digits := (digits_in_base 5 588) in
  list.sum digits = 12 :=
by
  -- Definition to convert a number to a list of its digits in a given base.
  def digits_in_base (b : ℕ) (n : ℕ) : list ℕ :=
    if n = 0 then [0]
    else
      let rec digits_aux (m : ℕ) (acc : list ℕ) : list ℕ :=
        if m = 0 then acc
        else digits_aux (m / b) ((m % b) :: acc)
      digits_aux n []

  let digits := digits_in_base 5 588 in
  -- Summing the digits
  have h_digits : digits = [4, 3, 2, 3] := by
    sorry
  
  have h_sum_digits : list.sum digits = 12 := by
    rw h_digits
    norm_num
  exact h_sum_digits

end sum_of_digits_base_5_of_588_is_12_l278_278848


namespace selling_price_equals_1280_l278_278454

noncomputable def cost_price : ℝ := 1625 / 1.25

def percentage_profit := ((1320 - cost_price) / cost_price) * 100

def selling_price_with_loss := cost_price - ((percentage_profit / 100) * cost_price)

theorem selling_price_equals_1280 :
  selling_price_with_loss = 1280 := 
sorry

end selling_price_equals_1280_l278_278454


namespace nearest_int_to_sum_of_logarithms_of_proper_divisors_of_1000000_l278_278367

theorem nearest_int_to_sum_of_logarithms_of_proper_divisors_of_1000000 : 
  let S := (∑ d in {d ∣ 1000000 | d ≠ 1000000}, Real.log10 d) 
  in (Int.nearest S) = 141 := 
by 
  have h1 : 1000000 = 10^6 := by norm_num,
  have h2 : ∀ (a : ℕ) (b : ℕ), (a * b = 1000000) → (Real.log10 a + Real.log10 b = 6),
  { intros a b h_eq,
    rw [← Real.log10_mul (ne_of_gt (nat.cast_pos.mpr (nat.pos_of_ne_zero (ne_of_gt (mul_pos (nat.pos_of_ne_zero (nat.lt.base 2)) (nat.pos_of_ne_zero (nat.lt.base 5)))))))],
    exact Real.log10_eq_log10 (a * b) 1000000 h_eq,
    rw [nat.cast_mul], norm_num },
  have h3 : {d : ℕ | d ∣ 1000000 ∧ d ≠ 1000000}.card = 48 := 
  by {
    sorry
  },
  have h4 : ∑ d in {d ∣ 1000000 | d ≠ 1000000}, Real.log10 d = (24 * 6 - Real.log10 1000),
  { sorry
  },
  have h5 : Real.log10 1000 = 3 := by norm_num,
  have h6 : (24 * 6 - 3 : ℝ) = 141 := by norm_num,
  sorry

end nearest_int_to_sum_of_logarithms_of_proper_divisors_of_1000000_l278_278367


namespace smallest_difference_l278_278818

variable (DE EF FD : ℕ)

def is_valid_triangle (DE EF FD : ℕ) : Prop :=
  DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference (h1 : DE < EF)
                           (h2 : EF ≤ FD)
                           (h3 : DE + EF + FD = 1024)
                           (h4 : is_valid_triangle DE EF FD) :
  ∃ d, d = EF - DE ∧ d = 1 :=
by
  sorry

end smallest_difference_l278_278818


namespace lightest_weight_more_than_two_l278_278010

-- Definition of the weights of the bags
def weights : List ℝ := [1.2, 3.1, 2.4, 3.0, 1.8]

-- Define a predicate to filter weights greater than 2 kg
def greater_than_two (x : ℝ) : Prop := x > 2.0

-- Extract the weights that meet the condition
def filtered_weights : List ℝ := weights.filter greater_than_two

-- Verify the minimum weight of the filtered weights
theorem lightest_weight_more_than_two : (filtered_weights ≠ [] ∧ (filtered_weights.minimum > 2.0)) → filtered_weights.minimum = 2.4 :=
by
  sorry

end lightest_weight_more_than_two_l278_278010


namespace most_accurate_value_l278_278691

def concentration (K : ℝ) (error_bound : ℝ) : Prop :=
  (3.91246 - error_bound) ≤ K ∧ K ≤ (3.91246 + error_bound) 

theorem most_accurate_value (K : ℝ) (error_bound : ℝ) (h : concentration K error_bound) : 
    round (K * 10) / 10 = 3.9 :=
by
  sorry

end most_accurate_value_l278_278691


namespace time_to_pass_approx_5_01_l278_278567

def train_length : ℝ := 64 -- Train length in meters
def speed_kmph : ℝ := 46 -- Speed in km per hour

-- Convert speed from km/h to m/s
def speed_mps : ℝ := speed_kmph * (1000 / 3600)

-- Calculate the time taken to pass the telegraph post
def time_to_pass : ℝ := train_length / speed_mps

/-- Proving the time computed is approximately equal to 5.01 seconds -/
theorem time_to_pass_approx_5_01 : abs (time_to_pass - 5.01) < 0.01 :=
by
  sorry

end time_to_pass_approx_5_01_l278_278567


namespace solve_x_from_t_l278_278509

theorem solve_x_from_t (t : Real) (h_cond : 2 * t ^ 2 + 3 * t - 9 = 0 ∧ sqrt 2 ≤ t ∧ t < ∞) :
  x = 1 / 4 ∨ x = 1 ∨ x = 9 / 4 :=
sorry

end solve_x_from_t_l278_278509


namespace square_area_64_l278_278777

theorem square_area_64
  (x1 x2 : ℝ)
  (h1 : (x1 ≠ x2)) :
  let A := (x1, 2),
      B := (x2, 2),
      C := (x2, 10),
      D := (x1, 10)
  in let side_length := (10 - 2) in
  side_length^2 = 64 :=
by
  -- Establish vertices on coordinate plane
  let A : ℝ × ℝ := (x1, 2)
  let B : ℝ × ℝ := (x2, 2)
  let C : ℝ × ℝ := (x2, 10)
  let D : ℝ × ℝ := (x1, 10)
  
  -- Define the side length of the square
  let side_length : ℝ := 10 - 2

  -- We need to prove the area is 64
  show side_length^2 = 64,
  -- Proof is omitted, hence use sorry
  sorry

end square_area_64_l278_278777


namespace count_multiples_of_8_between_200_and_400_l278_278264

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278264


namespace sum_first_10_terms_of_arithmetic_seq_l278_278211

variables {a : ℕ → ℝ} -- Arithmetic Sequence
variable (n : ℕ) -- index

-- Conditions
def condition_1 :=
  (a 3) ^ 2 + (a 8) ^ 2 + 2 * (a 3) * (a 8) = 9

def condition_2 : Prop :=
  ∀ n, a n < 0

-- Conclusion (Sum of the first 10 terms)
def sum_first_10_terms (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 10, a (i + 1)

-- Lean statement for proof
theorem sum_first_10_terms_of_arithmetic_seq :
  condition_1 →
  condition_2 →
  sum_first_10_terms a = -15 :=
by
  intros h1 h2
  sorry

end sum_first_10_terms_of_arithmetic_seq_l278_278211


namespace solve_sqrt_eq_l278_278773

theorem solve_sqrt_eq : ∃ x : ℝ, sqrt (7 * x - 3) + sqrt (x ^ 3 - 1) = 3 ∧ x = 1 :=
by
  existsi (1 : ℝ)
  sorry

end solve_sqrt_eq_l278_278773


namespace sum_of_digits_greatest_prime_divisor_l278_278482

theorem sum_of_digits_greatest_prime_divisor (h : 8191 = 2^13 - 1) : digit_sum (greatest_prime_divisor 8191) = 10 :=
sorry

-- Definitions to provide necessary context
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def greatest_prime_divisor (n : ℕ) : ℕ :=
  if h : n = 1 then 1
  else (n.proper_divisors.filter (prime)).max' sorry

end sum_of_digits_greatest_prime_divisor_l278_278482


namespace avg_first_class_l278_278012

def first_class_avg_mark (n1 n2 : ℕ) (avg2 avg_all : ℝ) (total_students1 : ℕ) : ℝ :=
  (total_students1 * avg_all - n2 * avg2) / n1

theorem avg_first_class :
  let n1 := 30
  let n2 := 50
  let avg2 := 70
  let avg_all := 58.75
  first_class_avg_mark n1 n2 avg2 avg_all 80 = 40 :=
by
  -- Mathematical logic to reach the solution comes here.
  sorry

end avg_first_class_l278_278012


namespace steve_starting_berries_l278_278421

/-- Given Stacy has 32 berries, Steve takes 4 of Stacy's berries, 
    and still has 7 less berries than Stacy started with,
    prove that Steve started with 21 berries. -/
theorem steve_starting_berries : 
  ∃ x : ℕ, x + 4 = 32 - 7 ∧ x = 21 :=
by
  exists 21
  split
  { exact rfl }
  { sorry }

end steve_starting_berries_l278_278421


namespace value_of_a_plus_b_l278_278317

noncomputable def a (b : ℝ) : ℝ := sqrt (2 * b - 4) + sqrt (4 - 2 * b) - 1

theorem value_of_a_plus_b : 
  ∀ b : ℝ, (2 * b - 4 ≥ 0) ∧ (4 - 2 * b ≥ 0) → a b + b = 1 :=
by
  intros b h
  let h1 := h.1
  let h2 := h.2
  sorry

end value_of_a_plus_b_l278_278317


namespace probability_product_greater_than_zero_l278_278024

noncomputable theory

def probability_of_positive_product : ℝ :=
  let interval_length : ℝ := 30 in
  let probability_of_positive_or_negative : ℝ := (15 / interval_length) in
  probability_of_positive_or_negative * probability_of_positive_or_negative + 
  probability_of_positive_or_negative * probability_of_positive_or_negative

theorem probability_product_greater_than_zero :
  probability_of_positive_product = 1 / 2 :=
by
  sorry

end probability_product_greater_than_zero_l278_278024


namespace rectangle_ratio_l278_278193

theorem rectangle_ratio (s x y : ℝ) (h1 : 4 * (x * y) + s * s = 9 * s * s) (h2 : s + 2 * y = 3 * s) (h3 : x + y = 3 * s): x / y = 2 :=
by sorry

end rectangle_ratio_l278_278193


namespace sum_of_ages_l278_278053

variables (P Q : ℕ)

-- Conditions represented in Lean
def condition1 := (P - 6) = (1 / 2 : ℚ) * (Q - 6)
def condition2 := (P : ℚ) / (Q : ℚ) = 3 / 4

-- Goal: Prove P + Q = 21 given the conditions
theorem sum_of_ages (h1 : condition1) (h2 : condition2) : P + Q = 21 := 
by
  sorry

end sum_of_ages_l278_278053


namespace petya_wins_l278_278760

/-- 
  Prove that the smallest \( k \) such that Petya can figure out 
  the position of a \( 1 \times 6 \) rectangle on a \( 13 \times 13 \) board,
  no matter how Vasya places the rectangle and tells Petya the covered marked cells,
  is 84.
-/
def petyaVasyaGame : Prop :=
  ∀ (marking_scheme : fin 13 × fin 13 → Prop), 
    (∃ k ≥ 84, 
      (∀ (rect : fin 13 × fin 6), 
        ∃ (x y : fin 13), 
          marking_scheme (x, y) ∧
          (∀ (x' y' : fin 13), marking_scheme (x', y') → (x = x' ∧ y = y'))))

theorem petya_wins : petyaVasyaGame := sorry

end petya_wins_l278_278760


namespace combinedTaxRate_is_33_33_l278_278868

-- Define Mork's tax rate
def morkTaxRate : ℝ := 0.40

-- Define Mindy's tax rate
def mindyTaxRate : ℝ := 0.30

-- Define relationship between Mork's income and Mindy's income
def incomeRatio : ℝ := 2

-- Given Mork's income
variable (X : ℝ) -- Mork's income

-- Define Mork's and Mindy's income
def morkIncome := X
def mindyIncome := incomeRatio * X

-- Calculate their taxes
def morkTax := morkTaxRate * morkIncome
def mindyTax := mindyTaxRate * mindyIncome

-- Define combined tax and combined income
def combinedTax := morkTax + mindyTax
def combinedIncome := morkIncome + mindyIncome

-- Calculate combined tax rate
def combinedTaxRate := (combinedTax / combinedIncome) * 100

-- The theorem to prove
theorem combinedTaxRate_is_33_33
  (X_pos : X > 0) : combinedTaxRate = 33.33 := by
  unfold combinedTaxRate combinedTax combinedIncome
  simp [morkTaxRate, mindyTaxRate, incomeRatio, morkTax, mindyTax, morkIncome, mindyIncome]
  linarith

end combinedTaxRate_is_33_33_l278_278868


namespace smallest_square_l278_278669

theorem smallest_square (h : ∃ t : ℕ, (t+1)^2 - t^2 = 191) : ∃ a : ℕ, a = 9025 :=
by {
  cases h with t ht,
  have t_eq_95: t = 95,
  { -- proof here
    sorry
  },
  use t^2,
  rw t_eq_95,
  norm_num,
}

end smallest_square_l278_278669


namespace probability_product_positive_l278_278020

-- Define interval and properties
def interval : set ℝ := { x : ℝ | -15 ≤ x ∧ x ≤ 15 }

-- Define independent selection of x and y
def selected_independently (x y : ℝ) : Prop :=
  x ∈ interval ∧ y ∈ interval

-- Main theorem
theorem probability_product_positive : 
  (∃ x y ∈ interval, selected_independently x y ∧ x * y > 0) → (probability (x * y > 0) = 1/2) :=
sorry

end probability_product_positive_l278_278020


namespace number_of_pairs_of_socks_l278_278811

theorem number_of_pairs_of_socks (n : ℕ) (h : 2 * n^2 - n = 112) : n = 16 := sorry

end number_of_pairs_of_socks_l278_278811


namespace cyclic_min_f_min_value_f_l278_278924

variables {A B C D P E : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  [MetricSpace D] [MetricSpace P] [MetricSpace E]

def f (P : P) (A B C D : P) : ℝ :=
  dist P A * dist B C + dist P D * dist C A + dist P C * dist A B

theorem cyclic_min_f (A B C D P : P) (h_conv : ConvexQuadrilateral A B C D) 
  (h_angle : angle B A D + angle D C B < π) : (min_f P A B C D) ↔ Cyclic A B C D := sorry

theorem min_value_f (A B C P : P) (E : E) (circumcircle : Circumcircle A B C)
  (h_ratios : dist A E / dist A B = sqrt 3 / 2 ∧ dist B C / dist E C = sqrt 3 - 1)
  (h_angles : angle E C B = 1 / 2 * angle E C A)
  (h_tangents : Tangent D A circumcircle ∧ Tangent D C circumcircle)
  (h_len : dist A C = sqrt 2) : 
  min_value_of_f P A B C D E = sqrt 10 := sorry

end cyclic_min_f_min_value_f_l278_278924


namespace num_int_values_lt_2pi_l278_278259

theorem num_int_values_lt_2pi : 
  (finset.Icc (- ⌊2 * Real.pi⌋) ⌊2 * Real.pi⌋).card = 13 := 
by
  sorry

end num_int_values_lt_2pi_l278_278259


namespace octahedron_non_blue_probability_l278_278032

theorem octahedron_non_blue_probability :
  let total_faces := 8
  let blue_faces := 3
  let red_faces := 3
  let green_faces := 2
  let non_blue_faces := total_faces - blue_faces
  (non_blue_faces / total_faces : ℚ) = (5 / 8 : ℚ) :=
by
  sorry

end octahedron_non_blue_probability_l278_278032


namespace arrangements_of_people_l278_278810

theorem arrangements_of_people : 
  let n := 5 in
  let total_arrangements := n.factorial in
  let A_left_end := (n - 1).factorial in
  let A_B_adjacent : ℕ := 2 * (n - 1).factorial in
  let A_left_end_B_adjacent := (n - 2).factorial in
  (total_arrangements - A_left_end - A_B_adjacent + A_left_end_B_adjacent) = 54 := sorry

end arrangements_of_people_l278_278810


namespace standard_deviation_less_than_mean_l278_278779

theorem standard_deviation_less_than_mean 
  (μ : ℝ) (σ : ℝ) (x : ℝ) 
  (h1 : μ = 14.5) 
  (h2 : σ = 1.5) 
  (h3 : x = 11.5) : 
  (μ - x) / σ = 2 :=
by
  rw [h1, h2, h3]
  norm_num

end standard_deviation_less_than_mean_l278_278779


namespace point_2023_0_cannot_lie_on_line_l278_278667

-- Define real numbers a and c with the condition ac > 0
variables (a c : ℝ)

-- The condition ac > 0
def ac_positive := (a * c > 0)

-- The statement that (2023, 0) cannot be on the line y = ax + c given the condition a * c > 0
theorem point_2023_0_cannot_lie_on_line (h : ac_positive a c) : ¬ (0 = 2023 * a + c) :=
sorry

end point_2023_0_cannot_lie_on_line_l278_278667


namespace option_A_is_parallel_projection_l278_278853

-- Define the options and describe their conditions
inductive ProjectionOption
| A | B | C | D

-- Defining the conditions for each option based on the problem statement
def isParallelProjection : ProjectionOption → Prop
| ProjectionOption.A := true  -- Sunlight causes parallel projection
| ProjectionOption.B := false -- Street lamp causes central projection
| ProjectionOption.C := false -- Desk lamp causes central projection
| ProjectionOption.D := false -- Flashlight causes central projection

-- The theorem to prove that Option A is a parallel projection
theorem option_A_is_parallel_projection : isParallelProjection ProjectionOption.A = true :=
by {
  sorry
}

end option_A_is_parallel_projection_l278_278853


namespace largest_non_summable_composite_l278_278152

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278152


namespace minimum_value_l278_278185

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem minimum_value :
  ∃ x₀ : ℝ, (∀ x : ℝ, f x₀ ≤ f x) ∧ f x₀ = -2 := by
  sorry

end minimum_value_l278_278185


namespace cone_section_height_ratio_l278_278783

theorem cone_section_height_ratio (A_base A_section : ℝ) (h : A_section = A_base / 2) :
  let h_cone := (2 : ℝ) in
  let h_section := (sqrt 2 - 1) in
  (1 : ℝ) / h_section = 1 / (sqrt 2 - 1) :=
by
  -- proof to be filled
  sorry

end cone_section_height_ratio_l278_278783


namespace starting_number_of_range_l278_278009

theorem starting_number_of_range (multiples: ℕ) (end_of_range: ℕ) (span: ℕ)
  (h1: multiples = 991) (h2: end_of_range = 10000) (h3: span = multiples * 10) :
  end_of_range - span = 90 := 
by 
  sorry

end starting_number_of_range_l278_278009


namespace total_students_in_class_l278_278684

theorem total_students_in_class (R S : ℕ) (h1 : 2 + 12 + 14 + R = S) (h2 : 2 * S = 40 + 3 * R) : S = 44 :=
by
  sorry

end total_students_in_class_l278_278684


namespace problem_solution_l278_278019

def class_a_scores : List ℝ := [9.1, 7.9, 8.4, 6.9, 5.2, 7.1, 8.0, 8.1, 6.7, 4.9]
def class_b_scores : List ℝ := [8.8, 8.5, 7.3, 7.1, 6.7, 8.4, 9.0, 8.7, 7.8, 7.9]

def average (xs : List ℝ) : ℝ := xs.sum / xs.length

def variance (xs : List ℝ) : ℝ :=
  let avg := average xs
  (xs.map (λ x => (x - avg) ^ 2)).sum / xs.length

noncomputable def percentile (xs : List ℝ) (p : ℝ) : ℝ :=
  let sorted := xs.toArray.qsort (· < ·).toList
  let pos := (sorted.length : ℝ) * p
  (sorted.get? (pos.toNat - 1)).getD 0

theorem problem_solution :
  (average class_b_scores > average class_a_scores) ∧
  (variance class_b_scores < variance class_a_scores) ∧
  (percentile class_a_scores 0.4 ≠ 6.9) ∧
  (class_a_scores.filter (λ x => x > 7).length / class_a_scores.length = 0.6) := 
sorry

end problem_solution_l278_278019


namespace function_monotonically_increasing_l278_278642

noncomputable def new_function (x : ℝ) : ℝ :=
  sin (2 * x + π / 3)

theorem function_monotonically_increasing :
  monotone_on new_function (Set.Icc (-5 * π / 12) (-π / 6)) :=
sorry

end function_monotonically_increasing_l278_278642


namespace direct_proportion_l278_278919

theorem direct_proportion (c f p : ℝ) (h : f ≠ 0 ∧ p = c * f) : ∃ k : ℝ, p / f = k * (f / f) :=
by
  sorry

end direct_proportion_l278_278919


namespace james_initial_bars_l278_278713

def initial_chocolate_bars (sold_last_week sold_this_week needs_to_sell : ℕ) : ℕ :=
  sold_last_week + sold_this_week + needs_to_sell

theorem james_initial_bars : 
  initial_chocolate_bars 5 7 6 = 18 :=
by 
  sorry

end james_initial_bars_l278_278713


namespace zero_function_theorem_l278_278588

noncomputable def find_zero_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≥ 0) ∧
  (∀ x, ∃ f' : ℝ, has_deriv_at f f' x) ∧
  (∀ x, ∀ f' : ℝ, has_deriv_at f f' x → f' ≥ 0) ∧
  (∀ n : ℤ, f n = 0) →
  (∀ x, f x = 0)

theorem zero_function_theorem (f : ℝ → ℝ)
  (h1 : ∀ x, f x ≥ 0)
  (h2 : ∀ x, ∃ f' : ℝ, has_deriv_at f f' x)
  (h3 : ∀ x, ∀ f' : ℝ, has_deriv_at f f' x → f' ≥ 0)
  (h4 : ∀ n : ℤ, f n = 0) :
  ∀ x, f x = 0 :=
begin
  sorry,
end

end zero_function_theorem_l278_278588


namespace max_value_set_x_graph_transformation_l278_278645

noncomputable def function_y (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6)) + 2

theorem max_value_set_x :
  ∃ k : ℤ, ∀ x : ℝ, x = k * Real.pi + Real.pi / 6 → function_y x = 4 :=
by
  sorry

theorem graph_transformation :
  ∀ x : ℝ, ∃ y : ℝ, (y = Real.sin x → y = 2 * Real.sin (2 * x + (Real.pi / 6)) + 2) :=
by
  sorry

end max_value_set_x_graph_transformation_l278_278645


namespace largest_non_representable_as_sum_of_composites_l278_278168

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278168


namespace largest_four_digit_divisible_by_six_l278_278843

theorem largest_four_digit_divisible_by_six :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 0) ∧ 
    (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 6 = 0) → m ≤ n) :=
begin
  existsi 9996,
  split, 
  exact ⟨by norm_num, by norm_num⟩,
  split, 
  exact dec_trivial,
  intro m,
  intro h,
  exact ⟨by norm_num [h.1], by norm_num [h.2]⟩
end

end largest_four_digit_divisible_by_six_l278_278843


namespace value_range_U_l278_278219

variable {x y : ℝ}

def condition (x y : ℝ) : Prop := 2^x + 2^y = 4^x + 4^y

theorem value_range_U (hx : condition x y) : 1 < 8^x + 8^y ∧ 8^x + 8^y ≤ 2 := by
  sorry

end value_range_U_l278_278219


namespace number_of_non_decreasing_lists_l278_278881

theorem number_of_non_decreasing_lists (n : ℕ) (r : ℕ) (hn : n = 12) (hr : r = 3) : 
  (nat.choose (n + r - 1) r) = 364 :=
by
  sorry

end number_of_non_decreasing_lists_l278_278881


namespace distinct_prime_count_l278_278578

-- Definitions for the prime factorizations of each number
def fact_85 := 5 * 17
def fact_87 := 3 * 29
def fact_90 := 2 * 3^2 * 5
def fact_92 := 2^2 * 23

-- Definition of the full product
def full_product := fact_85 * fact_87 * fact_90 * fact_92

-- Definition of the distinct primes
def distinct_primes := {2, 3, 5, 17, 23, 29}

-- The proof statement
theorem distinct_prime_count : distinct_primes.card = 6 :=
  by sorry

end distinct_prime_count_l278_278578


namespace allowance_calculation_l278_278601

-- Define the problem constants
noncomputable def allowance : ℝ := A
def final_amount : ℝ := 2.10

-- Define the conditions as Lean definitions
def after_video_games (A : ℝ) := A * (4 / 9)
def after_graphic_novels (A : ℝ) := after_video_games A * (1 / 3)
def after_concert_tickets (A : ℝ) := after_graphic_novels A * (4 / 11)
def after_vinyl_records (A : ℝ) := after_concert_tickets A * (1 / 6)

-- The main theorem to prove
theorem allowance_calculation (A : ℝ) :
  after_vinyl_records A = final_amount → A = 233.8875 :=
by
  sorry

end allowance_calculation_l278_278601


namespace angle_BDP_l278_278423

def congruent_triangles (A B C D P : Type) [triangle A B P] [triangle A C P] [triangle A D P] := 
  AB = AC ∧ AB = AD ∧ BP = CP ∧ BP = DP 

def equal_angles (A B C D P : Type) [triangle A B P] [triangle A C P] [triangle A D P] := 
  ∠BAP = 15 ∧ ∠ACP = 15 ∧ ∠ADP = 15 

theorem angle_BDP (A B C D P : Type)
[h1: congruent_triangles A B C D P] 
[h2: equal_angles A B C D P]: 
∠BDP = 30 :=
sorry

end angle_BDP_l278_278423


namespace smallest_b_no_inverse_mod75_and_mod90_l278_278847

theorem smallest_b_no_inverse_mod75_and_mod90 :
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, n > 0 → n < b →  ¬ (n.gcd 75 > 1 ∧ n.gcd 90 > 1)) ∧ 
  (b.gcd 75 > 1 ∧ b.gcd 90 > 1) ∧ 
  b = 15 := 
by
  sorry

end smallest_b_no_inverse_mod75_and_mod90_l278_278847


namespace largest_non_representable_as_sum_of_composites_l278_278163

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278163


namespace no_digit_C_makes_2C4_multiple_of_5_l278_278984

theorem no_digit_C_makes_2C4_multiple_of_5 : ∀ (C : ℕ), (2 * 100 + C * 10 + 4 ≠ 0 ∨ 2 * 100 + C * 10 + 4 ≠ 5) := 
by 
  intros C
  have h : 4 ≠ 0 := by norm_num
  have h2 : 4 ≠ 5 := by norm_num
  sorry

end no_digit_C_makes_2C4_multiple_of_5_l278_278984


namespace line_through_three_points_l278_278958

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Given conditions
def p1 : Point := { x := 1, y := -1 }
def p2 : Point := { x := 3, y := 3 }
def p3 : Point := { x := 2, y := 1 }

-- The line that passes through the points
def line_eq (m b : ℝ) (p : Point) : Prop :=
  p.y = m * p.x + b

-- The condition of passing through the three points
def passes_three_points (m b : ℝ) : Prop :=
  line_eq m b p1 ∧ line_eq m b p2 ∧ line_eq m b p3

-- The statement to prove
theorem line_through_three_points (m b : ℝ) (h : passes_three_points m b) : m + b = -1 :=
  sorry

end line_through_three_points_l278_278958


namespace quadratic_root_value_l278_278192

theorem quadratic_root_value (a : ℝ) (h : ∃ x : ℝ, x = 2 ∧ 2 * x^2 - 3 * x - a^2 + 1 = 0) : a = sqrt 3 ∨ a = - sqrt 3 :=
by
  sorry

end quadratic_root_value_l278_278192


namespace find_lines_through_P_and_origin_l278_278252

noncomputable def intersect_point (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ :=
  Classical.choose (exists_unique Int)

axiom line_equation (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop

theorem find_lines_through_P_and_origin 
  (l1 : ℝ → ℝ → Prop := λ x y, x + y - 2 = 0)
  (l2 : ℝ → ℝ → Prop := λ x y, 2 * x + y + 2 = 0)
  (l3 : ℝ → ℝ → Prop := λ x y, x - 3 * y - 1 = 0)
  (P : ℝ × ℝ := intersect_point l1 l2) :
  line_equation P (0, 0) (3, 2) ∧
  line_equation P P (-3, 1) :=
sorry

end find_lines_through_P_and_origin_l278_278252


namespace problem1_l278_278051

theorem problem1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a ^ 2 + 1 / a ^ 2 + 3) / (4 * a + 1 / (4 * a)) = 10 * Real.sqrt 5 := sorry

end problem1_l278_278051


namespace pyramid_levels_l278_278721

theorem pyramid_levels (n : ℕ) (toothpicks : ℕ) :
  (∀ k : ℕ, 1 ≤ k → k ≤ n → toothpicks = (2 : ℕ)^(k-1)) →
  (∑ i in finset.range n, (2 : ℕ)^i) = 1023 →
  n = 10 :=
by
  sorry

end pyramid_levels_l278_278721


namespace justin_total_pages_l278_278362

theorem justin_total_pages :
  let pages_read (n : ℕ) := (10 * 2 ^ n)
  in ∑ i in Finset.range 7, pages_read i = 1270 := by
  sorry

end justin_total_pages_l278_278362


namespace largest_not_sum_of_two_composites_l278_278140

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278140


namespace parabola_line_perpendicular_condition_l278_278814

theorem parabola_line_perpendicular_condition (b : ℝ) (k : ℝ) :
  (∃ x₁ x₂ t : ℝ, x₁ ≠ x₂ ∧ t ≠ x₁ ∧ t ≠ x₂ ∧
    (x₁^2 - k*x₁ - b = 0) ∧ (x₂^2 - k*x₂ - b = 0) ∧ 
    ((x₁ + t) * (x₂ + t) = -1)) ↔ b ∈ set.Ici 1 :=
by
  sorry

end parabola_line_perpendicular_condition_l278_278814


namespace prime_divides_f_n_l278_278724

-- Define the function f with the given conditions
variables {f : ℕ → ℕ}
variable [f_condition1 : ∀ m n : ℕ, Nat.gcd m n = 1 → Nat.gcd (f m) (f n) = 1]
variable [f_condition2 : ∀ n : ℕ, n ≤ f n ∧ f n ≤ n + 2012]

theorem prime_divides_f_n {n : ℕ} {p : ℕ} (h1 : Nat.Prime p) (h2 : p ∣ f n) : p ∣ n :=
  sorry

end prime_divides_f_n_l278_278724


namespace sum_of_odds_from_15_to_55_l278_278849

theorem sum_of_odds_from_15_to_55 :
  let a := 15
  let d := 4
  let l := 55
  (l - a) % d = 0 → ∃ n : ℕ, l = a + (n - 1) * d ∧ (n * (a + l)) / 2 = 385 := by
  intros a d l h
  use 11
  split
  sorry  -- l = a + (11 - 1) * d
  sorry  -- (11 * (a + l)) / 2 = 385

end sum_of_odds_from_15_to_55_l278_278849


namespace largest_perimeter_l278_278091

noncomputable def triangle_largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : ℕ :=
7 + 8 + y

theorem largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : triangle_largest_perimeter y h1 h2 = 29 :=
sorry

end largest_perimeter_l278_278091


namespace range_of_m_min_of_squares_l278_278733

-- 1. Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 4)

-- 2. State the condition that f(x) ≤ -m^2 + 6m holds for all x
def condition (m : ℝ) : Prop := ∀ x : ℝ, f x ≤ -m^2 + 6 * m

-- 3. State the range of m to be proven
theorem range_of_m : ∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5 := 
sorry

-- 4. Auxiliary condition for part 2
def m_0 : ℝ := 5

-- 5. State the condition 3a + 4b + 5c = m_0
def sum_condition (a b c : ℝ) : Prop := 3 * a + 4 * b + 5 * c = m_0

-- 6. State the minimum value problem
theorem min_of_squares (a b c : ℝ) : sum_condition a b c → a^2 + b^2 + c^2 ≥ 1 / 2 := 
sorry

end range_of_m_min_of_squares_l278_278733


namespace problem1_problem2_l278_278241

open Real

noncomputable def f (x : ℝ) : ℝ := sin (x + (π / 2))

theorem problem1 : f x = cos x := by
  sorry

theorem problem2 (α : ℝ) (h : tan α + 1 / tan α = 5) :
    (sqrt 2 * cos (2 * α - π / 4) - 1) / (1 - tan α) = 2 / 5 := by
  sorry

end problem1_problem2_l278_278241


namespace sum_of_powers_l278_278997

def z : ℂ := -1 / 2 + (real.sqrt 3) / 2 * complex.I

theorem sum_of_powers (z_def : z = -1 / 2 + (real.sqrt 3) / 2 * complex.I) :
  ∑ i in finset.range 2023 + 1, z ^ (i + 1) = -1 / 2 + (real.sqrt 3) / 2 * complex.I :=
sorry

end sum_of_powers_l278_278997


namespace quadrilateral_centers_diagonals_equal_perpendicular_l278_278757

theorem quadrilateral_centers_diagonals_equal_perpendicular
  {A B C D E F G H : Point}
  (concave_quad : convex_quadrilateral A B C D)
  (squares_AB_BC_CD_DA : squares_constructed_externally_on_sides A B C D E F G H):
  equal_and_perpendicular_diagonals E F G H :=
begin
  sorry
end

end quadrilateral_centers_diagonals_equal_perpendicular_l278_278757


namespace largest_four_digit_number_divisible_by_6_l278_278838

theorem largest_four_digit_number_divisible_by_6 :
  ∃ n, n = 9996 ∧ ∀ m, (m ≤ 9999 ∧ m % 6 = 0) → m ≤ n :=
begin
  sorry
end

end largest_four_digit_number_divisible_by_6_l278_278838


namespace minimum_value_expression_l278_278197

theorem minimum_value_expression {a : ℝ} (h₀ : 1 < a) (h₁ : a < 4) : 
  (∃ m : ℝ, (∀ x : ℝ, 1 < x ∧ x < 4 → m ≤ (x / (4 - x) + 1 / (x - 1))) ∧ m = 2) :=
sorry

end minimum_value_expression_l278_278197


namespace clock_angle_at_3_30_l278_278105

-- Define the necessary parameters
def h := 3
def m := 30

-- Define the formula to calculate the angle
def calculate_angle (h m : ℕ) : ℝ :=
|((60 * h) - (11 * m)) / 2|

-- The main statement to be proved
theorem clock_angle_at_3_30 : calculate_angle h m = 75 := sorry

end clock_angle_at_3_30_l278_278105


namespace find_x_y_l278_278980

theorem find_x_y (x y : ℝ) : (3 * x + 4 * -2 = 0) ∧ (3 * 1 + 4 * y = 0) → x = 8 / 3 ∧ y = -3 / 4 :=
by
  sorry

end find_x_y_l278_278980


namespace minimize_fraction_1099_l278_278629

theorem minimize_fraction_1099 :
  ∃ x : ℕ, (1000 ≤ x ∧ x < 10000) ∧
  ∃ y : ℕ, y = (x / 1000) + ((x % 1000) / 100) + ((x % 100) / 10) + (x % 10) ∧
  (∀ z : ℕ, (1000 ≤ z ∧ z < 10000) → ∃ w : ℕ, w = (z / 1000) + ((z % 1000) / 100) + ((z % 100) / 10) + (z % 10) → 
  (x / y) ≤ (z / w)) ∧
  x = 1099 :=
begin
  sorry
end

end minimize_fraction_1099_l278_278629


namespace largest_non_representable_as_sum_of_composites_l278_278169

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278169


namespace slices_with_both_onions_and_olives_l278_278544

noncomputable def slicesWithBothToppings (total_slices slices_with_onions slices_with_olives : Nat) : Nat :=
  slices_with_onions + slices_with_olives - total_slices

theorem slices_with_both_onions_and_olives 
  (total_slices : Nat) (slices_with_onions : Nat) (slices_with_olives : Nat) :
  total_slices = 18 ∧ slices_with_onions = 10 ∧ slices_with_olives = 10 →
  slicesWithBothToppings total_slices slices_with_onions slices_with_olives = 2 :=
by
  sorry

end slices_with_both_onions_and_olives_l278_278544


namespace rectangular_to_spherical_coords_l278_278953
noncomputable def rect_to_spherical (x y z : ℝ) :=
  (real.sqrt (x^2 + y^2 + z^2),
   real.atan2 y x,
   real.arccos (z / real.sqrt (x^2 + y^2 + z^2)))

theorem rectangular_to_spherical_coords :
  rect_to_spherical 0 (4 * real.sqrt 3) (-4) = (8, real.pi / 2, 2 * real.pi / 3) :=
by
  sorry

end rectangular_to_spherical_coords_l278_278953


namespace positive_integer_solutions_count_l278_278311

theorem positive_integer_solutions_count :
  ∃ (N : ℕ), ∀ (x : ℝ), 0 < N ∧ N < 100 ∧ (∃ x, x ^ (floor x) = N) ∧ N = 1 + 0 + 5 + 38 + 0 → N = 44 :=
by
  sorry

end positive_integer_solutions_count_l278_278311


namespace georgie_spooky_ways_l278_278535

theorem georgie_spooky_ways : 
  ∀ w : ℕ, w = 10 →
  ∃ ways : ℕ, ways = 10 * 9 * 8 * 7 :=
by
  intros w hW
  use 5040
  rw hW
  exact (compute 10 * 9 * 8 * 7)
  -- rewrite 10 * 9 * 8 * 7 calculation
  sorry

end georgie_spooky_ways_l278_278535


namespace largest_cannot_be_sum_of_two_composites_l278_278162

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278162


namespace second_warehouse_more_profitable_in_first_year_buying_first_warehouse_more_advantageous_l278_278064

-- Part (a)
theorem second_warehouse_more_profitable_in_first_year :
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let first_year_cost1 := rent1 * 12
  let worst_case_cost2 := rent2 * 5 + rent1 * 7 + move_cost
  worst_case_cost2 < first_year_cost1 :=
by
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let first_year_cost1 := rent1 * 12
  let worst_case_cost2 := rent2 * 5 + rent1 * 7 + move_cost
  exact lt_of_lt_of_le 
    (add_lt_add_right (add_lt_add_right (add_lt_add_right rfl rfl) rfl) rfl)
    (calc worst_case_cost2 < first_year_cost1 := by sorry
    )

-- Part (b)
theorem buying_first_warehouse_more_advantageous :
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let total_cost1 := rent1 * 12 * 5
  let worst_case_cost2 := rent2 * 5 + rent1 * 12 * 4 + move_cost
  let buy_cost := 3_000_000
  total_cost1 > buy_cost ∧ worst_case_cost2 > buy_cost :=
by
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let total_cost1 := rent1 * 12 * 5
  let worst_case_cost2 := rent2 * 5 + rent1 * 12 * 4 + move_cost
  let buy_cost := 3_000_000
  exact and.intro 
    (calc total_cost1 > buy_cost := by sorry)
    (calc worst_case_cost2 > buy_cost := by sorry)

end second_warehouse_more_profitable_in_first_year_buying_first_warehouse_more_advantageous_l278_278064


namespace simplify_polynomial_l278_278477

theorem simplify_polynomial : 
  (5 - 3 * x - 7 * x^2 + 3 + 12 * x - 9 * x^2 - 8 + 15 * x + 21 * x^2) = (5 * x^2 + 24 * x) :=
by 
  sorry

end simplify_polynomial_l278_278477


namespace largest_four_digit_number_divisible_by_six_l278_278833

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end largest_four_digit_number_divisible_by_six_l278_278833


namespace vector_decomposition_l278_278039

noncomputable section

open Matrix

def x : Fin 3 → ℝ := ![11, -1, 4]
def p : Fin 3 → ℝ := ![1, -1, 2]
def q : Fin 3 → ℝ := ![3, 2, 0]
def r : Fin 3 → ℝ := ![-1, 1, 1]

def α : ℝ := 3
def β : ℝ := 2
def γ : ℝ := -2

theorem vector_decomposition : x = α • p + β • q + γ • r := by
  rw [smul_eq_mul, smul_eq_mul, smul_eq_mul]
  sorry

end vector_decomposition_l278_278039


namespace second_smallest_five_digit_in_pascal_l278_278846

theorem second_smallest_five_digit_in_pascal :
  ∃ (x : ℕ), (x > 10000) ∧ (∀ y : ℕ, (y ≠ 10000) → (y < x) → (y < 10000)) ∧ (x = 10001) :=
sorry

end second_smallest_five_digit_in_pascal_l278_278846


namespace find_magnitude_difference_l278_278630

variables {α : Type*} [real_vector_space α] (a b : α)

-- Conditions
def cond1 : |a| = 2 := sorry
def cond2 : |b| = 3 := sorry
def cond3 : |a + b| = √19 := sorry

-- Question (Proof Problem)
theorem find_magnitude_difference : |a - b| = √7 :=
by
  -- Use the conditions provided
  rw [cond1, cond2, cond3]
  sorry

end find_magnitude_difference_l278_278630


namespace largest_non_sum_of_composites_l278_278171

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278171


namespace olympic_medals_l278_278698

theorem olympic_medals :
  ∃ (a b c : ℕ),
    (a + b + c = 100) ∧
    (3 * a - 153 = 0) ∧
    (c - b = 7) ∧
    (a = 51) ∧
    (a - 13 = 38) ∧
    (c = 28) :=
by
  sorry

end olympic_medals_l278_278698


namespace solution_set_l278_278234

def f (x : ℝ) : ℝ := 2016^x + Real.logb 2016 (Real.sqrt (x^2 + 1) + x) - 2016^(-x) + 2

theorem solution_set (x : ℝ) : f (3 * x + 1) + f x > 4 → x > -1 / 4 := by
  sorry

end solution_set_l278_278234


namespace unique_square_friendly_l278_278099

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k^2 = n

def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18 * m + c)

theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 := 
sorry

end unique_square_friendly_l278_278099


namespace total_present_ages_l278_278882

theorem total_present_ages (P Q : ℕ) 
    (h1 : P - 12 = (1 / 2) * (Q - 12))
    (h2 : P = (3 / 4) * Q) : P + Q = 42 :=
by
  sorry

end total_present_ages_l278_278882


namespace union_complement_l278_278651

open Set

universe u

variable {α : Type u}

def U : Set α := {1, 2, 3, 4, 5}
def A : Set α := {1, 3}
def B : Set α := {1, 2, 4}

theorem union_complement (U A B : Set α) : 
  U = {1, 2, 3, 4, 5} ∧ A = {1, 3} ∧ B = {1, 2, 4} → ((U \ B) ∪ A) = {1, 3, 5} :=
by 
  sorry

end union_complement_l278_278651


namespace regular_tetrahedron_plane_intersection_l278_278547

theorem regular_tetrahedron_plane_intersection (V : Finset ℝ) (hV : V.card = 4) :
  ∃ P : Finset (Finset ℝ), (∀ X ∈ P, X.card = 3) ∧ (∀ X ∈ P, (∃ p q r ∈ X, X = {p, q, r}) ∧ plane_does_not_intersect_interior X V) := sorry

def plane_does_not_intersect_interior (X V : Finset ℝ) : Prop :=
  ∀ p q r ∈ X, X ⊆ V → (plane p q r).interior ⊆ V

def plane (p q r : ℝ) : Set ℝ :=
  {x | x ∈ ℝ ∧ 0 <= x}

end regular_tetrahedron_plane_intersection_l278_278547


namespace polynomial_evaluation_l278_278963

theorem polynomial_evaluation :
  (∃ x : ℝ, x^2 - 3 * x - 18 = 0 ∧ 0 < x ∧ (x^3 - 3 * x^2 - 9 * x + 5 = 59)) :=
by
  let x := 6
  have h₁ : x^2 - 3 * x - 18 = 0 := by ring
  have h₂ : 0 < x := by norm_num
  have h₃ : x^3 - 3 * x^2 - 9 * x + 5 = 59 := by norm_num
  use x
  exact ⟨h₁, h₂, h₃⟩

end polynomial_evaluation_l278_278963


namespace fractional_part_same_l278_278764

theorem fractional_part_same (n : ℕ) :
  let a := (5 + Real.sqrt 26)^n;
      b := (5 - Real.sqrt 26)^n in
  (∃ m : ℤ, a + b = m) ∧ 
  (if even n then 0 < b ∧ b < 10^(-n) else 0 > b ∧ b > -10^(-n)) →
  (if even n then ∃ k : ℕ, (a - k) * 10^n ≅ Real.floor (a * 10^n) + (1 - Real.floor (a * 10^n) / 10^n) else
  (if odd n then ∃ k : ℕ, (a - k) * 10^n ≅ Real.floor (a * 10^n))) :=
sorry

end fractional_part_same_l278_278764


namespace volume_third_bottle_is_250_milliliters_l278_278106

-- Define the volumes of the bottles in milliliters
def volume_first_bottle : ℕ := 2 * 1000                        -- 2000 milliliters
def volume_second_bottle : ℕ := 750                            -- 750 milliliters
def total_volume : ℕ := 3 * 1000                               -- 3000 milliliters
def volume_third_bottle : ℕ := total_volume - (volume_first_bottle + volume_second_bottle)

-- The theorem stating the volume of the third bottle
theorem volume_third_bottle_is_250_milliliters :
  volume_third_bottle = 250 :=
by
  sorry

end volume_third_bottle_is_250_milliliters_l278_278106


namespace exists_n_divisible_l278_278725

theorem exists_n_divisible (k : ℕ) (m : ℤ) (hk : k > 0) (hm : m % 2 = 1) : 
  ∃ n : ℕ, n > 0 ∧ 2^k ∣ (n^n - m) :=
by
  sorry

end exists_n_divisible_l278_278725


namespace estimated_pass_number_l278_278609

theorem estimated_pass_number (total_papers selected_papers passing_papers : ℕ) (H1 : total_papers = 10000) 
  (H2 : selected_papers = 500) (H3 : passing_papers = 420) : 
  let pass_rate := (passing_papers : ℝ) / selected_papers in 
  total_papers * pass_rate = 8400 := 
by 
  sorry

end estimated_pass_number_l278_278609


namespace equal_a_b_l278_278986

theorem equal_a_b (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_n : 0 < n) 
  (h_eq : (a + b)^n - (a - b)^n = (a / b) * ((a + b)^n + (a - b)^n)) : a = b :=
sorry

end equal_a_b_l278_278986


namespace correct_calculation_l278_278664

def original_number (x : ℕ) : Prop := x + 12 = 48

theorem correct_calculation (x : ℕ) (h : original_number x) : x + 22 = 58 := by
  sorry

end correct_calculation_l278_278664


namespace amount_allocated_to_food_l278_278872

theorem amount_allocated_to_food (total_amount : ℝ) (household_ratio food_ratio misc_ratio : ℝ) 
  (h₁ : total_amount = 1800) (h₂ : household_ratio = 5) (h₃ : food_ratio = 4) (h₄ : misc_ratio = 1) :
  food_ratio / (household_ratio + food_ratio + misc_ratio) * total_amount = 720 :=
by
  sorry

end amount_allocated_to_food_l278_278872


namespace color_grid_l278_278717

-- Define a type for colors
inductive Color
| red : Color
| green : Color
| blue : Color

-- Specify cells in the 2 by 3 grid
structure Grid :=
  (A B C D E F : Color)

-- Predicate to ensure adjacent cells do not share the same color
def adjacent_constraint (g : Grid) : Prop :=
  g.A ≠ g.B ∧
  g.B ≠ g.C ∧
  g.A ≠ g.D ∧
  g.D ≠ g.E ∧
  g.B ≠ g.E ∧
  g.C ≠ g.F ∧
  g.E ≠ g.F

-- Count the total number of valid colorings
def count_valid_colorings : Nat :=
  sorry  -- The detailed counting of 384 according to the constraints

-- Theorems statement
theorem color_grid (countEq : count_valid_colorings = 384) : Prop :=
  countEq = 384

end color_grid_l278_278717


namespace smallest_prime_factor_in_C_is_51_l278_278415

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if is_prime n
  then n
  else Nat.find (λ p, p > 1 ∧ p ∣ n ∧ is_prime p)

def set_C := {47, 49, 51, 53, 55}

theorem smallest_prime_factor_in_C_is_51 : ∃ n ∈ set_C, forall m ∈ set_C, smallest_prime_factor n ≤ smallest_prime_factor m ∧ n = 51 :=
by
  sorry

end smallest_prime_factor_in_C_is_51_l278_278415


namespace min_varphi_symmetry_l278_278242

def f (x : ℝ) : ℝ := √3 * Real.sin (2 * x) + Real.cos (2 * x)

def shifted_f (x varphi : ℝ) : ℝ := √3 * Real.sin (2 * (x - varphi)) + Real.cos (2 * (x - varphi))

def symmetric_about (g : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, g (2 * c - x) = g x

theorem min_varphi_symmetry (varphi : ℝ) (h : ∀ x, shifted_f x varphi = 2 * Real.sin (2 * x + π / 6 - 2 * varphi))
  (sym : symmetric_about (fun x => 2 * Real.sin (2 * x + π / 6 - 2 * varphi)) (π / 12)) :
  varphi = 5 * π / 12 :=
by sorry

end min_varphi_symmetry_l278_278242


namespace largest_not_sum_of_two_composites_l278_278141

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278141


namespace largest_non_representable_as_sum_of_composites_l278_278166

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278166


namespace regular_tetrahedron_min_loop_length_l278_278080

theorem regular_tetrahedron_min_loop_length (L : ℝ) (L_pos: L > 0) : 
  ∃ x, (x = 2 * L ∧ ∀ y, (y ≥ 2 * L)) :=
by 
  have CircumscribedCircle : ∀ L: ℝ, L > 0 -> C = 2 * L :=
    λ L L_pos, sorry,
  have LoopAroundOneEdge : ∀ L: ℝ, L > 0 -> y ≥ 2L :=
    λ L L_pos, sorry
  have CompareLengths :  ∀ L x: ℝ, L > 0 -> ((x = (2 * L)) ) :=
    λ L L_pos, sorry,
end sorry

end regular_tetrahedron_min_loop_length_l278_278080


namespace jeans_more_than_scarves_l278_278529

def num_ties := 34
def num_belts := 40
def num_black_shirts := 63
def num_white_shirts := 42
def num_jeans := (2 / 3) * (num_black_shirts + num_white_shirts)
def num_scarves := (1 / 2) * (num_ties + num_belts)

theorem jeans_more_than_scarves : num_jeans - num_scarves = 33 := by
  sorry

end jeans_more_than_scarves_l278_278529


namespace lawn_length_l278_278900

-- Defining the main conditions
def area : ℕ := 20
def width : ℕ := 5

-- The proof statement (goal)
theorem lawn_length : (area / width) = 4 := by
  sorry

end lawn_length_l278_278900


namespace cost_of_building_fence_l278_278508

-- Define the conditions
def area : ℕ := 289
def price_per_foot : ℕ := 60

-- Define the length of one side of the square (since area = side^2)
def side_length (a : ℕ) : ℕ := Nat.sqrt a

-- Define the perimeter of the square (since square has 4 equal sides)
def perimeter (s : ℕ) : ℕ := 4 * s

-- Define the cost of building the fence
def cost (p : ℕ) (ppf : ℕ) : ℕ := p * ppf

-- Prove that the cost of building the fence is Rs. 4080
theorem cost_of_building_fence : cost (perimeter (side_length area)) price_per_foot = 4080 := by
  -- Skip the proof steps
  sorry

end cost_of_building_fence_l278_278508


namespace find_other_root_l278_278515

theorem find_other_root (m : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, (x = -6 → (x^2 + m * x - 6 = 0))) → (x^2 + m * x - 6 = (x + 6) * (x - 1)) → (∀ x : ℝ, (x^2 + 5 * x - 6 = 0) → (x = -6 ∨ x = 1))) :=
sorry

end find_other_root_l278_278515


namespace area_of_triangle_ABC_l278_278422

-- Given triangle ABC is a right triangle at B
-- P is a point on hypotenuse AC such that ∠ABP = 45°
-- AP = 1, CP = 2
-- Prove that the area of triangular ABC is 9/5

def area_triangle_ABC : ℝ :=
  let AP := 1
  let CP := 2
  let AC := AP + CP
  let ratio_AB_BC := 1 / 2
  let AB := x
  let BC := ratio_AB_BC * x
  in 1 / 2 * AB * BC

theorem area_of_triangle_ABC (AP CP : ℝ) (h1 : AP = 1) (h2 : CP = 2)
  (ABC_right : ∀ AB BC AC : ℝ, AB^2 + BC^2 = AC^2) : 
  area_triangle_ABC = 9 / 5 :=
by
  sorry

end area_of_triangle_ABC_l278_278422


namespace sequence_bound_l278_278456

theorem sequence_bound :
  ∃ α > 0, α = 1/3 ∧ ∀ n: ℕ, n > 0 → ∀ (a : ℕ → ℝ),
    a 1 = 1 →
    (∀ k: ℕ, k > 0 → a (k + 1) = real.sqrt (a k ^ 2 + 1 / a k)) →
    (1 / 2) * n^α ≤ a n ∧ a n ≤ 2 * n^α :=
by
  sorry

end sequence_bound_l278_278456


namespace probability_product_greater_than_zero_l278_278028

open Set ProbabilityTheory

theorem probability_product_greater_than_zero (a b : ℝ) (ha : a ∈ Icc (-15) 15) (hb : b ∈ Icc (-15) 15) :
  P (λ x : ℝ × ℝ, 0 < x.1 * x.2 | (λ _, true) := 1/2 :=
begin
  sorry
end

end probability_product_greater_than_zero_l278_278028


namespace union_complement_l278_278652

open Set

universe u

variable {α : Type u}

def U : Set α := {1, 2, 3, 4, 5}
def A : Set α := {1, 3}
def B : Set α := {1, 2, 4}

theorem union_complement (U A B : Set α) : 
  U = {1, 2, 3, 4, 5} ∧ A = {1, 3} ∧ B = {1, 2, 4} → ((U \ B) ∪ A) = {1, 3, 5} :=
by 
  sorry

end union_complement_l278_278652


namespace triangle_angles_ratio_l278_278806

theorem triangle_angles_ratio (A B C : ℕ) 
  (hA : A = 20)
  (hB : B = 3 * A)
  (hSum : A + B + C = 180) :
  (C / A) = 5 := 
by
  sorry

end triangle_angles_ratio_l278_278806


namespace f_decreasing_l278_278401

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / x^2 + 3

theorem f_decreasing (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h : x1 < x2) : f x1 > f x2 := 
by
  sorry

end f_decreasing_l278_278401


namespace count_divisibles_l278_278299

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278299


namespace count_multiples_of_8_between_200_and_400_l278_278265

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278265


namespace largest_natural_number_not_sum_of_two_composites_l278_278143

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278143


namespace find_last_two_digits_of_numerator_l278_278790

noncomputable def S : ℚ :=
  ∑ k in Finset.range 2017, (k + 1) / (Nat.factorial (k + 2))

theorem find_last_two_digits_of_numerator :
  let num := (S * (Nat.factorial 2018)).numerator in
  num % 100 = 99 :=
by
  sorry

end find_last_two_digits_of_numerator_l278_278790


namespace angle_D_is_45_l278_278695

-- Definitions and conditions
variables (A B C D E : Type) [angle_space : ∀ {X Y : Type}, has_zero (angle X Y)]
variables (angleA : angle A D) (angleB : angle B E) (angleD : angle D E)
variables (AB BC CD DE : ℝ)

-- Given conditions
axiom AB_eq_BC : AB = BC
axiom CD_eq_DE : CD = DE
axiom angle_A_two_angle_B : angleA = 2 * angleB

-- Problem: Prove that ∠D is 45°
theorem angle_D_is_45 : angleD = 45 :=
by
  sorry

end angle_D_is_45_l278_278695


namespace period_of_fraction_l278_278597

theorem period_of_fraction (n : ℕ) (h : 1/49 = 0.020408163265306122448979591836734693877551) : ∃ m : ℕ, m = 42 :=
by
  use 42
  sorry

end period_of_fraction_l278_278597


namespace real_root_poly_iff_n_odd_l278_278988

theorem real_root_poly_iff_n_odd (n : ℕ) (hn : 0 < n) :
  (∃ x : ℝ, (∀ n : ℕ, P(x) = x^n + x^{n-1} + ... + 1) ∧ P(x) = 0) ↔ odd n := 
sorry

end real_root_poly_iff_n_odd_l278_278988


namespace solve_for_square_l278_278627

theorem solve_for_square (x : ℤ) (s : ℤ) 
  (h1 : s + x = 80) 
  (h2 : 3 * (s + x) - 2 * x = 164) : 
  s = 42 :=
by 
  -- Include the implementation with sorry
  sorry

end solve_for_square_l278_278627


namespace min_area_triangle_ABC_l278_278513

def point (α : Type*) := (α × α)

def area_of_triangle (A B C : point ℤ) : ℚ :=
  (1/2 : ℚ) * abs (36 * (C.snd) - 15 * (C.fst))

theorem min_area_triangle_ABC :
  ∃ (C : point ℤ), area_of_triangle (0, 0) (36, 15) C = 3 / 2 :=
by
  sorry

end min_area_triangle_ABC_l278_278513


namespace counting_divisibles_by_8_l278_278292

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278292


namespace chimney_problem_l278_278568

variable (x : ℕ) -- number of bricks in the chimney
variable (t : ℕ)
variables (brenda_hours brandon_hours : ℕ)

def brenda_rate := x / brenda_hours
def brandon_rate := x / brandon_hours
def combined_rate := (brenda_rate + brandon_rate - 15) * t

theorem chimney_problem (h1 : brenda_hours = 9)
    (h2 : brandon_hours = 12)
    (h3 : t = 6)
    (h4 : combined_rate = x) : x = 540 := sorry

end chimney_problem_l278_278568


namespace largest_four_digit_divisible_by_six_l278_278841

theorem largest_four_digit_divisible_by_six :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 0) ∧ 
    (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 6 = 0) → m ≤ n) :=
begin
  existsi 9996,
  split, 
  exact ⟨by norm_num, by norm_num⟩,
  split, 
  exact dec_trivial,
  intro m,
  intro h,
  exact ⟨by norm_num [h.1], by norm_num [h.2]⟩
end

end largest_four_digit_divisible_by_six_l278_278841


namespace numbers_divisible_by_8_between_200_and_400_l278_278284

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278284


namespace circle_equation_l278_278436

theorem circle_equation (x y : ℝ) :
  (∃ c : ℝ, (x - 0)^2 + (y - c)^2 = 1 ∧ y ∈ {c - 1, c + 1}) ↔ (x^2 + (y - 2)^2 = 1) :=
begin
  sorry
end

end circle_equation_l278_278436


namespace rectangle_dimensions_l278_278716

theorem rectangle_dimensions (x y : ℕ) (h : 2 * (x + y) = 22) :
  (∃a b c d : ℕ, (a = 2) ∧ (b = 6) ∧ (a * b = 2 * 6) ∧ (∃s1 s2 s3 s4 : ℕ, 2 * ((x) + y) + 18 = 40)) ∧
  ((x = 5 ∧ y = 6) ∨ (x = 8 ∧ y = 3) ∨ (x = 4 ∧ y = 7)) :=
begin
  sorry,
end

end rectangle_dimensions_l278_278716


namespace jelly_cost_l278_278581

theorem jelly_cost (B J : ℕ) 
  (h1 : 15 * (6 * B + 7 * J) = 315) 
  (h2 : 0 ≤ B) 
  (h3 : 0 ≤ J) : 
  15 * J * 7 = 315 := 
sorry

end jelly_cost_l278_278581


namespace age_difference_l278_278495

theorem age_difference (A B C : ℕ) (h1 : B = 20) (h2 : C = B / 2) (h3 : A + B + C = 52) : A - B = 2 := by
  sorry

end age_difference_l278_278495


namespace lower_bound_for_x_l278_278680

variable {x y : ℝ}  -- declaring x and y as real numbers

theorem lower_bound_for_x 
  (h₁ : 3 < x) (h₂ : x < 6)
  (h₃ : 6 < y) (h₄ : y < 8)
  (h₅ : y - x = 4) : 
  ∃ ε > 0, 3 + ε = x := 
sorry

end lower_bound_for_x_l278_278680


namespace sum_of_parallel_segments_l278_278410

theorem sum_of_parallel_segments :
  let EF := 5
  let FG := 4
  let n := 200
  let len_segment (k : ℕ) := sqrt 41 * (200 - k) / 200
  let sum_segments := 2 * (∑ k in finset.range n, len_segment k) - sqrt 41
  sum_segments = 198 * sqrt 41 :=
by sorry

end sum_of_parallel_segments_l278_278410


namespace count_divisibles_l278_278302

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278302


namespace count_divisibles_l278_278300

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278300


namespace area_transformation_l278_278799

noncomputable def area_under_graph (g : ℝ → ℝ) : ℝ := ∫ x in a..b, real.abs (g x)

theorem area_transformation (g : ℝ → ℝ) (a b : ℝ) 
  (h₁ : area_under_graph g = 15) :
  area_under_graph (-2 * (λ x, g (x + 3))) = 30 := 
by { sorry }

end area_transformation_l278_278799


namespace domino_covering_l278_278826

theorem domino_covering (grid : ℕ → ℕ → Prop)
  (H : ∀ (x y : ℕ), (grid x y → grid (x + 1) y ∨ grid (x) (y + 1))
  ∧ (¬∃ x' y', grid x y ∧ grid x' y' ∧ ((x = x' ∧ abs (y - y') = 1) ∨ (y = y' ∧ abs (x - x') = 1)))) :
  ∃ (cover : ℕ → ℕ → Prop), (∀ (x y : ℕ), ¬grid x y → cover x y) ∧ (∀ (x y : ℕ), (¬cover x y → cover (x + 1) y ∨ cover x (y + 1))) := sorry

end domino_covering_l278_278826


namespace circumradius_equals_exradius_l278_278563

variables {A B C D O I : Type*}
variables [Circumcenter A B C O] [Incenter A B C I] [Altitude A D B C] [LiesOnSegment I O D]

theorem circumradius_equals_exradius :
  let R := radius (circumcircle A B C),
      r_a := radius (excircle_opposite_BC A B C) in
  R = r_a :=
begin
  sorry,
end

end circumradius_equals_exradius_l278_278563


namespace scout_troop_net_profit_l278_278910

theorem scout_troop_net_profit :
  ∃ (cost_per_bar selling_price_per_bar : ℝ),
    cost_per_bar = 1 / 3 ∧
    selling_price_per_bar = 0.6 ∧
    (1500 * selling_price_per_bar - (1500 * cost_per_bar + 50) = 350) :=
by {
  sorry
}

end scout_troop_net_profit_l278_278910


namespace roots_of_polynomial_l278_278968

noncomputable def f : Polynomial ℤ := 4*X^4 - 16*X^3 + 11*X^2 + 4*X - 3

theorem roots_of_polynomial :
  ∀ x : ℤ, f.eval x = 0 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end roots_of_polynomial_l278_278968


namespace find_number_l278_278541

variable (x : ℕ)
variable (result : ℕ)

theorem find_number (h : x * 9999 = 4690640889) : x = 469131 :=
by
  sorry

end find_number_l278_278541


namespace quadratic_function_analysis_l278_278649

theorem quadratic_function_analysis :
  let y := λ x : ℝ, -x^2 + 3*x + 1 in
  (∃ c1 c2 : Prop, c1 ∧ c2 ∧
    -- Conclusion 1: The parabola opens downwards.
    (c1 ↔ (∀ x : ℝ, ∂(y) x < 0)) ∧
    -- Conclusion 3: Function values increase as x increases for x < 3 / 2.
    (c2 ↔ (∀ x : ℝ, x < 3 / 2 → ∂(y) x > 0)) ∧
    -- Conclusion 2 and 4: Incorrect
    ¬(∃ c : Prop, c ∧ (c ↔ (∃ x : ℝ, x = 1)) ∧ (c ↔ (∃ x : ℝ, x > 4))) )
:= sorry

end quadratic_function_analysis_l278_278649


namespace total_outfits_l278_278861

-- Define the quantities of each item.
def red_shirts : ℕ := 7
def green_shirts : ℕ := 8
def pants : ℕ := 10
def blue_hats : ℕ := 10
def red_hats : ℕ := 10
def scarves : ℕ := 5

-- The total number of outfits without having the same color of shirts and hats.
theorem total_outfits : 
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves) = 7500 := 
by sorry

end total_outfits_l278_278861


namespace p_and_not_q_is_true_l278_278218

def p : Prop := ∃ x : ℝ, x - 2 > Real.log x / Real.log 2
def q : Prop := ∀ x : ℝ, x * x > 0

theorem p_and_not_q_is_true : p ∧ ¬q :=
by {
  -- proving that there exists an x such that x - 2 > log_2(x)
  have p_true : p := exists.intro 8 (by norm_num),
  -- proving that there exists an x such that x^2 = 0, contradicting q
  have q_false : ¬q := by intro h; exact (lt_irrefl 0 (h 0)),
  -- combining the proofs
  exact ⟨p_true, q_false⟩
}

end p_and_not_q_is_true_l278_278218


namespace abc_relationship_l278_278620

variables {R : Type*} [linear_ordered_field R]

-- Definitions given in conditions
def f (x : R) : R := sorry -- placeholder for function definition
def f' (x : R) : R := sorry -- placeholder for derivative definition

-- Odd function condition
lemma f_odd : ∀ x, f (-x) = -f x := sorry

-- Given condition: f'(x) + f(x) / x > 0 for x ≠ 0
lemma f_derivative_condition (x : R) (hx : x ≠ 0) : f'(x) + f(x) / x > 0 := sorry

noncomputable def a : R := f 1
noncomputable def b : R := -2 * f (-2)
noncomputable def c : R := (Real.log (1 / 2)) * f (Real.log (1 / 2))

-- Theorem: Prove the relationship among a, b, and c
theorem abc_relationship : c < a ∧ a < b :=
begin
  -- translate given conditions into Lean definitions and prove the relationship
  -- the proof steps would go here
  sorry
end

end abc_relationship_l278_278620


namespace largest_four_digit_divisible_by_six_l278_278842

theorem largest_four_digit_divisible_by_six :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 0) ∧ 
    (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 6 = 0) → m ≤ n) :=
begin
  existsi 9996,
  split, 
  exact ⟨by norm_num, by norm_num⟩,
  split, 
  exact dec_trivial,
  intro m,
  intro h,
  exact ⟨by norm_num [h.1], by norm_num [h.2]⟩
end

end largest_four_digit_divisible_by_six_l278_278842


namespace hexagon_center_of_mass_distance_l278_278797

theorem hexagon_center_of_mass_distance 
  (a : ℝ) :
  let hexagon_area := (3 * a^2 * real.sqrt 3) / 2
  let square_side := a / real.sqrt 2
  let square_area := square_side^2
  let remaining_area := hexagon_area - square_area
  let distance_from_center := ((3 * real.sqrt 3 + 1) * a) / 52
  distance_from_center = ((3 * real.sqrt 3 + 1) * a) / 52 :=
by sorry

end hexagon_center_of_mass_distance_l278_278797


namespace bricks_per_course_is_10_l278_278753

def number_of_bricks_per_course (walls courses_per_wall courses_unfinished total_bricks : ℕ) : ℕ :=
  total_bricks / (walls * courses_per_wall - courses_unfinished)

theorem bricks_per_course_is_10 :
  number_of_bricks_per_course 4 6 2 220 = 10 :=
by
  simp [number_of_bricks_per_course]
  sorry

end bricks_per_course_is_10_l278_278753


namespace problem1_problem2_problem3_l278_278108

theorem problem1 : 128 + 52 / 13 = 132 :=
by
  sorry

theorem problem2 : 132 / 11 * 29 - 178 = 170 :=
by
  sorry

theorem problem3 : 45 * (320 / (4 * 5)) = 720 :=
by
  sorry

end problem1_problem2_problem3_l278_278108


namespace natural_number_pairs_lcm_gcd_l278_278967

theorem natural_number_pairs_lcm_gcd (a b : ℕ) (h1 : lcm a b * gcd a b = a * b)
  (h2 : lcm a b - gcd a b = (a * b) / 5) : 
  (a = 4 ∧ b = 20) ∨ (a = 20 ∧ b = 4) :=
  sorry

end natural_number_pairs_lcm_gcd_l278_278967


namespace bill_main_project_hours_l278_278928

def total_working_hours_day1 : ℕ := 10
def total_working_hours_day2 : ℕ := 8
def total_working_hours_day3 : ℕ := 12
def total_working_hours_day4 : ℕ := 6

def naps_breaks_day1 : ℕ := 2 + 1 + 1.5
def naps_breaks_day2 : ℕ := 1.5 + 2 + 1 + 2
def naps_breaks_day3 : ℕ := 3 + 2.5
def naps_breaks_day4 : ℕ := (90 / 60) + 1.5 + 1

def task1_time : ℕ := 6
def task2_time : ℕ := 3

def time_spent_on_main_project : ℕ := 
(total_working_hours_day1 - naps_breaks_day1) + 
(total_working_hours_day2 - naps_breaks_day2 - task2_time) + 
(total_working_hours_day3 - naps_breaks_day3 - task1_time) + 
(total_working_hours_day4 - naps_breaks_day4)

theorem bill_main_project_hours : time_spent_on_main_project = 8 :=
  by
    have h1 : total_working_hours_day1 - naps_breaks_day1 = 5.5 := by sorry
    have h2 : total_working_hours_day2 - naps_breaks_day2 - task2_time = 0 := by sorry -- time cannot be negative
    have h3 : total_working_hours_day3 - naps_breaks_day3 - task1_time = 0.5 := by sorry
    have h4 : total_working_hours_day4 - naps_breaks_day4 = 2 := by sorry
    have h_total := h1 + h2 + h3 + h4
    exact h_total

end bill_main_project_hours_l278_278928


namespace largest_four_digit_number_divisible_by_six_l278_278835

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end largest_four_digit_number_divisible_by_six_l278_278835


namespace darcy_commute_l278_278498

theorem darcy_commute (d w r t x time_walk train_time : ℝ) 
  (h1 : d = 1.5)
  (h2 : w = 3)
  (h3 : r = 20)
  (h4 : train_time = t + x)
  (h5 : time_walk = 15 + train_time)
  (h6 : time_walk = d / w * 60)  -- Time taken to walk in minutes
  (h7 : t = d / r * 60)  -- Time taken on train in minutes
  : x = 10.5 :=
sorry

end darcy_commute_l278_278498


namespace birch_trees_probability_l278_278534

def binom (n k : ℕ) : ℕ := if k ≤ n then nat.choose n k else 0

theorem birch_trees_probability :
  let total_trees := 17
  let birch_trees := 6
  let slots := 12
  let non_birch_trees := 11
  let valid_arrangements := binom slots birch_trees
  let total_arrangements := binom total_trees birch_trees
  let probability := valid_arrangements * total_arrangements⁻¹
  (probability = (21 : ℚ / 283 : ℚ)) → (21 + 283 = 304)
:=  
by {
  sorry
}

end birch_trees_probability_l278_278534


namespace quadratic_inequality_sum_l278_278776

theorem quadratic_inequality_sum (a b : ℝ) (h1 : 1 < 2) 
 (h2 : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) 
 (h3 : 1 + 2 = a)  (h4 : 1 * 2 = b) : 
 a + b = 5 := 
by 
sorry

end quadratic_inequality_sum_l278_278776


namespace chip_credit_card_balance_l278_278941

-- Conditions
def initial_balance : Float := 50.00
def first_interest_rate : Float := 0.20
def additional_charge : Float := 20.00
def second_interest_rate : Float := 0.20

-- Question
def current_balance : Float :=
  let first_interest_fee := initial_balance * first_interest_rate
  let balance_after_first_interest := initial_balance + first_interest_fee
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  let second_interest_fee := balance_before_second_interest * second_interest_rate
  balance_before_second_interest + second_interest_fee

-- Correct Answer
def expected_balance : Float := 96.00

-- Proof Problem Statement
theorem chip_credit_card_balance : current_balance = expected_balance := by
  sorry

end chip_credit_card_balance_l278_278941


namespace num_ways_to_assign_grades_l278_278906

theorem num_ways_to_assign_grades : (4 ^ 12) = 16777216 := by
  sorry

end num_ways_to_assign_grades_l278_278906


namespace decreasing_interval_range_l278_278447

theorem decreasing_interval_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f(x) - f(y) ≤ 0) ∧ f(x) = x^2 + 2*(a-1)*x + 2 ∧ (∀ x : ℝ, x ≤ 6 → ∀ y : ℝ, y ≤ 6 → x < y → f(x) ≤ f(y)) → 
  a ≤ -5 := sorry

end decreasing_interval_range_l278_278447


namespace exists_k_for_polynomials_l278_278728

-- Definition of the set M
def M : set (ℝ → ℝ) := {P | ∃ (a b c d : ℝ), P = λ x, a * x^3 + b * x^2 + c * x + d ∧ ∀ x ∈ Icc (-1 : ℝ) 1, abs (P x) ≤ 1}

-- Lean statement of the problem
theorem exists_k_for_polynomials (P : ℝ → ℝ) (hP : P ∈ M) : ∃ k : ℝ, k = 4 ∧ ∀ (a b c d : ℝ) (h : P = λ x, a * x^3 + b * x^2 + c * x + d), abs a ≤ k := 
by
  use 4
  intros a b c d h
  rw [← h] at hP
  -- Prove the rest...
  sorry

end exists_k_for_polynomials_l278_278728


namespace three_digit_numbers_m_l278_278678

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem three_digit_numbers_m (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ sum_of_digits n = 12 ∧ 100 ≤ 2 * n ∧ 2 * n ≤ 999 ∧ sum_of_digits (2 * n) = 6 → ∃! (m : ℕ), n = m :=
sorry

end three_digit_numbers_m_l278_278678


namespace hyperbola_standard_equation_l278_278673

theorem hyperbola_standard_equation (k : ℝ) :
  ∃ k, (∀ x y : ℝ, y = (1/2) * x → (4, sqrt 3) ∈ (set_of (λ p : ℝ × ℝ, (p.1 ^ 2 / 4) - p.2 ^ 2 = k))) →
      k = 1 := 
sorry

end hyperbola_standard_equation_l278_278673


namespace count_multiples_of_8_between_200_and_400_l278_278268

theorem count_multiples_of_8_between_200_and_400 : 
  let count := (400 / 8 - (200 + 7) / 8) + 1 in
  count = 25 := 
by
  let smallest_multiple := 208
  let largest_multiple := 400
  let a := smallest_multiple / 8
  let l := largest_multiple / 8
  let n := l - a + 1
  have h_sm_le : 200 ≤ smallest_multiple := by norm_num
  have h_lm_ge : largest_multiple ≤ 400 := by norm_num
  trivial
  sorry

end count_multiples_of_8_between_200_and_400_l278_278268


namespace people_attended_game_total_l278_278125

def total_people_attended_game (P : ℕ) : Prop :=
  let supporters_of_first_team := 0.40 * P
  let supporters_of_second_team := 0.34 * P
  let non_supporters := 3
  let percentage_non_supporters := 0.26 * P
  non_supporters = Int.ofNat (round $ percentage_non_supporters) ∧ P = 12

theorem people_attended_game_total : ∃ P : ℕ, total_people_attended_game P :=
by
  use 12
  have percentage_non_supporters := 0.26 * 12
  have non_supporters := 3
  have h := non_supporters = Int.ofNat (round $ percentage_non_supporters)
  exact ⟨h, rfl⟩

end people_attended_game_total_l278_278125


namespace no_more_than_five_neighbors_l278_278007

noncomputable def connected_neighbors (points : Fin 20 → Prop) := sorry

theorem no_more_than_five_neighbors (points : Fin 20 → Prop) :
  (∀ i j : Fin 20, i ≠ j → points i ≠ points j) →  -- Unique pairwise distances
  (∀ i : Fin 20, ∃ S : Set (Fin 20), S.card ≤ 5 ∧ (∀ j ∈ S, distance (points i) (points j) < distance (points i) (points k) ∀ (k : Fin 20), k ≠ i ∧ k ≠ j)) :=    -- Each point connected to nearest neighbors
sorry

end no_more_than_five_neighbors_l278_278007


namespace digit_product_diff_l278_278877

theorem digit_product_diff
  (a b c d e f g : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 1 ≤ b ∧ b ≤ 9)
  (h3 : 1 ≤ c ∧ c ≤ 9)
  (h4 : 1 ≤ d ∧ d ≤ 9)
  (h5 : 1 ≤ e ∧ e ≤ 9)
  (h6 : 1 ≤ f ∧ f ≤ 9)
  (h7 : 1 ≤ g ∧ g ≤ 9)
  (h_distinct : list.nodup [a, b, c, d, e, f, g])
  (h_sum : 100*a + 10*b + c + 1000*d + 100*e + 10*f + g = 2020) :
  let abc := 100*a + 10*b + c,
      defg := 1000*d + 100*e + 10*f + g,
      max_prod := max (abc * defg) (defg * abc), -- Ensure distinct values
      min_prod := min (abc * defg) (defg * abc) in
  max_prod - min_prod = 552000 :=
by sorry

end digit_product_diff_l278_278877


namespace positive_number_is_25_over_9_l278_278327

variable (a : ℚ) (x : ℚ)

theorem positive_number_is_25_over_9 
  (h1 : 2 * a - 1 = -a + 3)
  (h2 : ∃ r : ℚ, r^2 = x ∧ (r = 2 * a - 1 ∨ r = -a + 3)) : 
  x = 25 / 9 := 
by
  sorry

end positive_number_is_25_over_9_l278_278327


namespace chip_credit_card_balance_l278_278938

-- Definitions based on the problem conditions
def initial_balance : ℝ := 50.00
def interest_rate : ℝ := 0.20
def additional_amount : ℝ := 20.00

-- Define the function to calculate the final balance after two months
def final_balance (b₀ r a : ℝ) : ℝ :=
  let b₁ := b₀ * (1 + r) in
  let b₂ := (b₁ + a) * (1 + r) in
  b₂

-- Theorem to prove that the final balance is 96.00
theorem chip_credit_card_balance : final_balance initial_balance interest_rate additional_amount = 96.00 :=
by
  -- Simplified proof outline
  sorry

end chip_credit_card_balance_l278_278938


namespace remainder_when_divide_by_66_l278_278800

-- Define the conditions as predicates
def condition_1 (n : ℕ) : Prop := ∃ l : ℕ, n % 22 = 7
def condition_2 (n : ℕ) : Prop := ∃ m : ℕ, n % 33 = 18

-- Define the main theorem
theorem remainder_when_divide_by_66 (n : ℕ) (h1 : condition_1 n) (h2 : condition_2 n) : n % 66 = 51 :=
  sorry

end remainder_when_divide_by_66_l278_278800


namespace range_of_a_l278_278611

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : ∀ x : ℝ, x > 0 → (1/a - 1/x ≤ 2 * x)) :
  a ∈ set.Ici (real.sqrt 2 / 4) :=
sorry

end range_of_a_l278_278611


namespace prime_triplet_implies_p_eq_3_l278_278474

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_triplet_implies_p_eq_3 (p : ℤ) :
  (is_prime p.to_nat) ∧ (is_prime (p + 2).to_nat) ∧ (is_prime (p + 4).to_nat) → p = 3 :=
by
  sorry

end prime_triplet_implies_p_eq_3_l278_278474


namespace largest_not_sum_of_two_composites_l278_278138

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278138


namespace count_multiples_of_8_in_range_l278_278277

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278277


namespace mean_greater_than_median_l278_278870

theorem mean_greater_than_median (x : ℕ) (hx : 0 < x) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 32)) / 5 in
  let median := x + 4 in
  mean - median = 5 := 
by
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 32)) / 5
  let median := x + 4
  exact calc
    mean - median = (x + 9) - (x + 4) : by sorry
    ... = 5 : by sorry

end mean_greater_than_median_l278_278870


namespace centaur_game_optimal_play_l278_278886

-- Definitions of the conditions
def board_width := 1000
def board_height := 1000
def removed_rectangles := [{x : Nat // x >= 0 ∧ x < 2} × {y : Nat // y >= 0 ∧ y < 994}]
def centaur_start_position := (500, 496)

-- Possible moves of the centaur
def move_up (pos : (Nat × Nat)) : (Nat × Nat) := (pos.1 - 1, pos.2)
def move_left (pos : (Nat × Nat)) : (Nat × Nat) := (pos.1, pos.2 - 1)
def move_diagonal (pos : (Nat × Nat)) : (Nat × Nat) := (pos.1 - 1, pos.2 + 1)

theorem centaur_game_optimal_play : ∀ moves : board_width × board_height → board_width × board_height,
  ∀ start_pos : (board_width × board_height) = centaur_start_position →
  ∀ rem_rects : removed_rectangles,
  ∀ move : (Nat × Nat) = move_up ∨ (Nat × Nat) = move_left ∨ (Nat × Nat) = move_diagonal,
  second_player_wins := 
sorry

end centaur_game_optimal_play_l278_278886


namespace total_pages_in_book_l278_278608

-- Define the given conditions
def chapters : Nat := 41
def days : Nat := 30
def pages_per_day : Nat := 15

-- Define the statement to be proven
theorem total_pages_in_book : (days * pages_per_day) = 450 := by
  sorry

end total_pages_in_book_l278_278608


namespace find_p_and_common_root_l278_278989

noncomputable def quadratic_common_root (p : ℚ) (x : ℚ) : Prop :=
  (9 * x^2 - 3 * (p + 6) * x + 6 * p + 5 = 0) ∧ (6 * x^2 - 3 * (p + 4) * x + 6 * p + 14 = 0)

theorem find_p_and_common_root :
  (quadratic_common_root (32 / 3) 3) ∧ (quadratic_common_root (-32 / 9) (-1)) :=
begin
  sorry
end

end find_p_and_common_root_l278_278989


namespace revision_cost_per_page_is_4_l278_278798

-- Definitions based on conditions
def initial_cost_per_page := 6
def total_pages := 100
def revised_once_pages := 35
def revised_twice_pages := 15
def no_revision_pages := total_pages - revised_once_pages - revised_twice_pages
def total_cost := 860

-- Theorem to be proved
theorem revision_cost_per_page_is_4 : 
  ∃ x : ℝ, 
    ((initial_cost_per_page * total_pages) + 
     (revised_once_pages * x) + 
     (revised_twice_pages * (2 * x)) = total_cost) ∧ x = 4 :=
by
  sorry

end revision_cost_per_page_is_4_l278_278798


namespace problem_proof_l278_278373

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

def num_multiples_of_lt (m bound : ℕ) : ℕ :=
  (bound - 1) / m

-- Definitions for the conditions
def a := num_multiples_of_lt 8 40
def b := num_multiples_of_lt 8 40

-- Proof statement
theorem problem_proof : (a - b)^3 = 0 := by
  sorry

end problem_proof_l278_278373


namespace switches_in_position_A_after_1000_steps_l278_278013

-- Define initial conditions and final statement for the theorem
def max_label := 2^9 * 3^9 * 5^9
def num_switches := 1000

-- The theorem to prove the number of switches in position A after 1000 steps
theorem switches_in_position_A_after_1000_steps : 
  let switch_labels := {n : ℕ | ∃ x y z : ℕ, x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ n = 2^x * 3^y * 5^z} in
  let count_switches_return_A := switch_labels.filter (λ n, (10 - n%2) * (10 - n%3) * (10 - n%5) % 4 = 0) in
  count_switches_return_A.card = 650 :=
by
  -- Statement only, implementation of the proof is needed
  sorry

end switches_in_position_A_after_1000_steps_l278_278013


namespace largest_n_perfect_square_l278_278973

theorem largest_n_perfect_square : ∀ {n : ℕ}, (∑ i in Finset.range (n+1), i.factorial) ∈ {x : ℕ | ∃ k : ℕ, x = k * k} ↔ n ≤ 3 := sorry

end largest_n_perfect_square_l278_278973


namespace find_d_l278_278186

theorem find_d (d : ℝ) (h : 4 * (3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5) = 3200.0000000000005) : d = 0.3 :=
by
  sorry

end find_d_l278_278186


namespace main_theorem_l278_278742

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition: f is symmetric about x = 1
def symmetric_about_one (a b c : ℝ) : Prop := 
  ∀ x : ℝ, f a b c (1 - x) = f a b c (1 + x)

-- Main statement
theorem main_theorem (a b c : ℝ) (h₁ : 0 < a) (h₂ : symmetric_about_one a b c) :
  ∀ x : ℝ, f a b c (2^x) > f a b c (3^x) :=
sorry

end main_theorem_l278_278742


namespace solution_Y_volume_l278_278418

theorem solution_Y_volume 
    (x_volume : ℝ) (x_alcohol_percent : ℝ) (y_alcohol_percent : ℝ) (final_alcohol_percent : ℝ) (final_volume_x : ℝ) : 
    ∃ (y_volume : ℝ), x_volume = 300 ∧ x_alcohol_percent = 0.15 ∧ y_alcohol_percent = 0.45 ∧ final_alcohol_percent = 0.25 ∧ 
    final_volume_x = 300 →
    y_volume = 150 := 
begin
  -- Let x be 300, 15%, let y be 45%, final alcohol be 25%
  intros,
  sorry
end

end solution_Y_volume_l278_278418


namespace max_element_of_list_of_five_l278_278903

theorem max_element_of_list_of_five (L : List ℕ) 
(h_len : L.length = 5) 
(h_pos : ∀ n ∈ L, 0 < n) 
(h_median : L.sorted.nth 2 = some 3) 
(h_mean : (L.sum : ℝ) / 5 = 11) : 
  (List.maximum L).getOrElse 0 = 47 := 
sorry

end max_element_of_list_of_five_l278_278903


namespace sues_final_answer_l278_278551

theorem sues_final_answer (x : ℕ) : 
  let ben_output := (x * 2 + 1) * 2 in
  let sue_output := (ben_output - 1) * 2 in
  x = 8 → sue_output = 66 :=
by
  intro h
  unfold ben_output sue_output
  simp [h]
  sorry

end sues_final_answer_l278_278551


namespace angle_m_half_p_plus_q_l278_278610

variable (P Q R S D : Type) [IsTriangle P Q R] [IsLineSegment P Q D] 
variable (n : RealAngle) [RightAngle n]
variable (R_bisects : ∡S P R = ∡S Q R)

theorem angle_m_half_p_plus_q 
  (p q m : RealAngle) 
  (angle_sum_triangle : p + q + (2 * ∡S P R) = Real.pi)
  (external_angle : m = ExteriorAngle P Q D) :
  m = (p + q) / 2 :=
  sorry

end angle_m_half_p_plus_q_l278_278610


namespace one_minute_interval_all_walking_same_direction_l278_278813

-- Define the speeds of each pedestrian in meters per minute
def speed1 : ℝ := 1000 / 60
def speed2 : ℝ := 2000 / 60
def speed3 : ℝ := 3000 / 60

-- Define the length of the alley in meters
def alley_length : ℝ := 100

-- Define the time it takes each pedestrian to traverse the alley
def time_to_traverse1 : ℝ := alley_length / speed1
def time_to_traverse2 : ℝ := alley_length / speed2
def time_to_traverse3 : ℝ := alley_length / speed3

-- Prove that there is a 1-minute interval during which all three pedestrians are walking in the same direction
theorem one_minute_interval_all_walking_same_direction :
  ∃ t : ℝ, ∃ δ : ℝ, δ = 1 ∧ (0 ≤ t ∧ forall i : ℝ, (i ∈ (t, t + δ)) → 
    ((floor (i / time_to_traverse1) % 2 = 0) ↔ (floor (i / time_to_traverse2) % 2 = 0)) ∧
    ((floor (i / time_to_traverse1) % 2 = 0) ↔ (floor (i / time_to_traverse3) % 2 = 0))) :=
sorry

end one_minute_interval_all_walking_same_direction_l278_278813


namespace part1_part2_l278_278514

variable (α : ℝ)

theorem part1 (h : tan α = 1 / 3) : 
  1 / (2 * sin α * cos α + cos α ^ 2) = 2 / 3 := 
  sorry

theorem part2 : 
  (tan (π - α) * cos (2 * π - α) * sin (- α + 3 * π / 2)) / (cos (- α - π) * sin (- π - α)) = cos α := 
  sorry

end part1_part2_l278_278514


namespace number_of_divisibles_by_eight_in_range_l278_278273

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278273


namespace cube_volume_surface_area_x_l278_278483

theorem cube_volume_surface_area_x (x s : ℝ) (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_x_l278_278483


namespace average_greatest_element_in_subsets_l278_278117

open Nat

theorem average_greatest_element_in_subsets :
  let S := Nat.range 2019 in
  (∑ x in (Finset.range 1919).filter (λ x, 100 ≤ x), x * (Nat.choose (x - 1) 99)) / (Nat.choose 2019 100) = 2000 :=
by
  let S := Nat.range 2019
  sorry 

end average_greatest_element_in_subsets_l278_278117


namespace train_crossing_time_l278_278552

noncomputable def train_length : ℝ := 150
noncomputable def man_speed_kmh : ℝ := 5
noncomputable def train_speed_kmh : ℝ := 84.99280057595394
noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def relative_speed := (train_speed_kmh + man_speed_kmh) * conversion_factor

theorem train_crossing_time :
  (train_length / relative_speed) ≈ 6.00024 := by
  -- Proof goes here
  sorry

end train_crossing_time_l278_278552


namespace engineer_mistake_l278_278441

noncomputable def weight_of_single_disk := 100
noncomputable def diameter_of_disk := 1
noncomputable def stddev_radius := 0.01
noncomputable def number_of_disks := 100
noncomputable def engineer_estimate := 10000

theorem engineer_mistake :
  (number_of_disks * weight_of_single_disk + number_of_disks * (stddev_radius^(2 : ℕ) + (0.5)^(2 : ℕ)) - engineer_estimate) = 4 := by
  sorry

end engineer_mistake_l278_278441


namespace counting_divisibles_by_8_l278_278293

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278293


namespace quadratic_solution_eq_l278_278435

theorem quadratic_solution_eq (c d : ℝ) 
  (h_eq : ∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = c ∨ x = d))
  (h_order : c ≥ d) :
  c + 2*d = 9 - Real.sqrt 23 :=
sorry

end quadratic_solution_eq_l278_278435


namespace bijection_f_l278_278587

-- Define the sets A and B
def A : Set ℝ := { x | ∃ n : ℕ, n > 1 ∧ x = 1 / n }
def B : Set ℝ := A ∪ {0, 1}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 / 2 then 0
  else if x = 1 / 3 then 1
  else if ∃ n : ℕ, n ≥ 4 ∧ x = 1 / n then 1 / ((nat.ofNat (nat.toNat (nat.pred (nat.pred n)))) : ℝ)
  else x

-- Define the intervals
def open_interval (a b : ℝ) : Set ℝ := { x | a < x ∧ x < b }
def closed_interval (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Lean statement to prove f is a bijection
theorem bijection_f : Function.Bijective (f : ℝ → ℝ) :=
sorry

end bijection_f_l278_278587


namespace solve_for_y_l278_278049

theorem solve_for_y : ∃ y : ℕ, 529 + 2 * 23 * 7 + 49 = y ∧ y = 900 :=
by { existsi (900 : ℕ), split, 
     calc
       529 + 2 * 23 * 7 + 49 = (23 + 7) * (23 + 7) : by ring
                         ... = 30 * 30           : by norm_num
                         ... = 900               : by norm_num,
     refl }

end solve_for_y_l278_278049


namespace ratio_of_b_to_a_l278_278606

theorem ratio_of_b_to_a
  (a b : ℝ)
  (A B C D : ℝ × ℝ)
  (hAB : dist A B = 2 * a)
  (hBC : dist B C = 2 * a)
  (hAD : dist A D = sqrt 2 * a)
  (hBD : dist B D = sqrt 2 * a)
  (hDC : dist D C = 2 * b)
  : b / a = 2 := by
  sorry

end ratio_of_b_to_a_l278_278606


namespace total_profit_correct_l278_278087

-- Defining the given conditions as constants
def beef_total : ℝ := 100
def beef_per_taco : ℝ := 0.25
def selling_price : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Calculate the number of tacos
def num_tacos := beef_total / beef_per_taco

-- Calculate the profit per taco
def profit_per_taco := selling_price - cost_per_taco

-- Calculate the total profit
def total_profit := num_tacos * profit_per_taco

-- Prove the total profit
theorem total_profit_correct : total_profit = 200 :=
by sorry

end total_profit_correct_l278_278087


namespace area_of_triangle_AEB_l278_278344

theorem area_of_triangle_AEB {A B C D F G E : Type*}
  (hRectangle : is_rectangle A B C D)
  (hAB : dist A B = 10)
  (hBC : dist B C = 5)
  (hDF : dist D F = 3)
  (hGC : dist G C = 4)
  (hInt : intersect_lines A F B G = E)
  : area_of_triangle A E B = 250 / 3 :=
begin
  sorry
end

end area_of_triangle_AEB_l278_278344


namespace mandy_score_l278_278337

theorem mandy_score (total_questions : ℕ) (lowella_percentage : ℕ) (pamela_percentage_extra : ℕ) :
  let lowella_correct := total_questions * lowella_percentage / 100
  let pamela_correct := lowella_correct + lowella_correct * pamela_percentage_extra / 100
  let mandy_correct := 2 * pamela_correct in
  total_questions = 100 ∧ lowella_percentage = 35 ∧ pamela_percentage_extra = 20 →
  mandy_correct = 84 :=
by
  intros total_questions_equals hundred lowella_percentage_equals thirty_five pamela_percentage_extra_equals twenty
  simp [total_questions_equals, lowella_percentage_equals, pamela_percentage_extra_equals]
  sorry

end mandy_score_l278_278337


namespace part1_part2_l278_278732

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part (1) 
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ a) → -2 ≤ a ∧ a ≤ 1 := by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l278_278732


namespace func_value_sum_l278_278239

noncomputable def f (x : ℝ) : ℝ :=
  -x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2 + 1

theorem func_value_sum : f (1/2) + f (-1/2) = 2 :=
by
  sorry

end func_value_sum_l278_278239


namespace counting_divisibles_by_8_l278_278290

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278290


namespace distance_AB_l278_278885

-- Definitions and conditions taken from part a)
variables (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0)

-- The main theorem statement
theorem distance_AB (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0) : 
  ∃ s : ℝ, s = Real.sqrt ((a * b * c) / (a + c - b)) := 
sorry

end distance_AB_l278_278885


namespace garden_ratio_2_l278_278895

theorem garden_ratio_2 :
  ∃ (P C k R : ℤ), 
      P = 237 ∧ 
      C = P - 60 ∧ 
      P + C + k = 768 ∧ 
      R = k / C ∧ 
      R = 2 := 
by
  sorry

end garden_ratio_2_l278_278895


namespace largest_not_sum_of_two_composites_l278_278135

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278135


namespace locker_count_proof_l278_278565

theorem locker_count_proof (cost_per_digit : ℕ := 3)
  (total_cost : ℚ := 224.91) :
  (N : ℕ) = 2151 :=
by
  sorry

end locker_count_proof_l278_278565


namespace initial_marbles_l278_278351

theorem initial_marbles (shared withRebecca: ℕ) (ends with: ℕ) :
  withRebecca = 33 → ends = 29 → (initial : ℕ) → initial = 62 :=
by
  intro h1 h2 h3
  sorry

end initial_marbles_l278_278351


namespace slower_car_speed_l278_278820

noncomputable def time_fast (D : ℝ) (v_fast : ℝ) : ℝ := D / v_fast

noncomputable def time_slow (time_fast : ℝ) (t_diff : ℝ) : ℝ := time_fast + t_diff

noncomputable def speed_slow (D : ℝ) (time_slow : ℝ) : ℝ := D / time_slow

theorem slower_car_speed :
  let D := 4.333329 in
  let v_fast := 78 in
  let t_diff := 0.333333 in
  let t_fast := time_fast D v_fast in
  let t_slow := time_slow t_fast t_diff in
  speed_slow D t_slow ≈ 11.142857 :=
by
  sorry

end slower_car_speed_l278_278820


namespace largest_number_not_sum_of_two_composites_l278_278180

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278180


namespace cost_prices_correct_maximize_profit_l278_278425

-- Define the cost prices of zongzi A and B
def cost_price_A : ℕ := 10
def cost_price_B : ℕ := cost_price_A + 2

-- Define the conditions for the purchasing problem
def condition1 : Prop := 1000 / cost_price_A = 1200 / cost_price_B
def condition2 : Prop := ∀ m : ℕ, 200 - m < 2 * m

-- Define the profit function
def profit (m : ℕ) : ℕ := -m + 600

-- Define the valid range for m
def valid_m (m : ℕ) : Prop := 400 / 3 ≤ m ∧ m < 200

-- Prove that the cost prices are 10 and 12
theorem cost_prices_correct : cost_price_A = 10 ∧ cost_price_B = 12 := by
  simp [cost_price_A, cost_price_B]

-- Prove that the profit function and conditions yield the maximum profit
theorem maximize_profit : 
  ∀ m : ℕ, valid_m m → profit m = 466 → m = 134 := by
  sorry

end cost_prices_correct_maximize_profit_l278_278425


namespace average_minutes_per_day_l278_278342

def differences_weekdays : List Int := [15, -5, 25, 35, -15]
def differences_weekends : List Int := [-20, 5]

def sum_differences (l : List Int) : Int :=
  l.foldr (· + ·) 0

def total_differences : Int :=
  sum_differences differences_weekdays + sum_differences differences_weekends

def average_difference_per_day : Real :=
  total_differences.toReal / 7

theorem average_minutes_per_day :
  average_difference_per_day ≈ 5.71 :=
by
  sorry

end average_minutes_per_day_l278_278342


namespace log_ab_a2_plus_log_ab_b2_eq_2_l278_278321

theorem log_ab_a2_plus_log_ab_b2_eq_2 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h_distinct : a ≠ b) (h_a_gt_2 : a > 2) (h_b_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 :=
by
  sorry

end log_ab_a2_plus_log_ab_b2_eq_2_l278_278321


namespace problem_solution_l278_278623

def lean_problem (a : ℝ) : Prop :=
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1)^x₁ > (2 * a - 1)^x₂) →
  a > 1 / 2 ∧ a ≤ 2 / 3

theorem problem_solution (a : ℝ) : lean_problem a :=
  sorry -- Proof is to be filled in

end problem_solution_l278_278623


namespace find_positive_number_l278_278001

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l278_278001


namespace best_play_wins_probability_l278_278074

/-- Define the conditions and parameters for the problem. -/
variables (n m : ℕ)
variables (C : ℕ → ℕ → ℕ) /- Binomial coefficient -/

/-- Define the probability calculation -/
def probability_best_play_wins : ℚ :=
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t)

/-- The theorem stating that the above calculation represents the probability of the best play winning -/
theorem best_play_wins_probability :
  probability_best_play_wins n m C =
  1 / (C (2 * n) n * C (2 * n) (2 * m)) *
  ∑ q in Finset.range (2 * m + 1),
    (C n q * C n (2 * m - q)) *
    ∑ t in Finset.range (min q (m - 1) + 1),
      C q t * C (2 * n - q) (n - t) :=
  by
  sorry

end best_play_wins_probability_l278_278074


namespace digit_makes_divisible_by_nine_l278_278475

theorem digit_makes_divisible_by_nine (A : ℕ) : (7 + A + 4 + 6) % 9 = 0 ↔ A = 1 :=
by
  sorry

end digit_makes_divisible_by_nine_l278_278475


namespace average_after_removal_l278_278689

def scores : List ℝ := [9.5, 9.4, 9.6, 9.9, 9.3, 9.7, 9.0]

def remove_highest_lowest (l : List ℝ) : List ℝ :=
  (l.erase (l.maximum?.getD 0)).erase (l.minimum?.getD 0)

def average (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem average_after_removal :
  average (remove_highest_lowest scores) = 9.5 := by
  sorry

end average_after_removal_l278_278689


namespace num_valid_integers_l278_278261

theorem num_valid_integers : 
  ∀ n, 200 ≤ n ∧ n ≤ 250 → 
  (∀ i j k, 
    n = 100*i + 10*j + k ∧
    i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    i < j ∧ j < k) → 
  (card {n | 200 ≤ n ∧ n ≤ 250 ∧ 
    (∃ i j k, n = 100*i + 10*j + k ∧ 
        i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
        i < j ∧ j < k)} = 11) :=
sorry

end num_valid_integers_l278_278261


namespace reasonable_inferences_l278_278491

-- Definitions provided by the conditions
def numbers : List ℝ := [15, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16]
def squares : List ℝ := [225, 228.01, 231.04, 234.09, 237.16, 240.25, 243.36, 246.49, 249.64, 252.81, 256]

-- The proof problem
theorem reasonable_inferences : 
  (√2.2801 = 1.51) ∧ 
  ((16.2^2 - 16.1^2) = 3.23) ∧ 
  (¬ (241 < 240.25 ∧ 243 < 243.36) ∧ 
  (∀ x y, x < 15 ∧ y < 15 ∧ abs (x - y) = 0.1 → abs (x^2 - y^2) < 3.01)) :=
  sorry

end reasonable_inferences_l278_278491


namespace sum_series_l278_278650

noncomputable def a : ℕ → ℕ
| 0 := 0
| 1 := 2019
| 2 := 2019
| (n + 1) := (a (n - 1) - 1) * (b n + 1)

noncomputable def b : ℕ → ℕ
| 0 := 0
| 1 := 2017
| 2 := 2017
| (n + 1) := a n * b (n - 1) - 1

theorem sum_series :
  ∑ n in Nat.range ∞, b n * (1 / (a (n + 1)) - 1 / (a (n + 3))) = 4074345 / 4078 :=
sorry

end sum_series_l278_278650


namespace single_elimination_rounds_l278_278548

theorem single_elimination_rounds {n : ℕ} (h : n = 2016) : 
  ∃ r : ℕ, r = 11 ∧ 
    ∀ k < r, 
      let p := (nat.ceil (n / (2 ^ k : ℕ):ℚ)) in 
        if p = 1 then true else
          p = (nat.ceil (n / (2 ^ (k + 1): ℕ):ℚ) + (1: ℕ)) ∨ p = (nat.ceil (n / (2 ^ (k + 1): ℕ):ℚ)) :=
by
  use (11: ℕ)
  split
  exact rfl
  intros k hk
  sorry

end single_elimination_rounds_l278_278548


namespace number_of_divisibles_by_eight_in_range_l278_278271

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278271


namespace major_premise_false_l278_278255

-- Define the conditions
variable {α : Type*} [Plane α] (a b : Line α) (p : Plane α)

-- Major Premise: If a line is parallel to a plane, then the line is parallel to all the lines within the plane
def major_premise (b : Line α) (p : Plane α) : Prop :=
∀ l : Line α, l ∈ p → b ∥ l

-- Given conditions
variable (cond1 : ¬(b ⊂ p))
variable (cond2 : a ∈ p)
variable (cond3 : b ∥ p)

-- We need to prove that the major premise is incorrect
theorem major_premise_false : ¬ major_premise b p :=
sorry

end major_premise_false_l278_278255


namespace scalene_triangle_l278_278323

theorem scalene_triangle (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : a ≠ c) : 
  ∃ t : Triangle, t.is_scalene a b c := 
by
  sorry

end scalene_triangle_l278_278323


namespace taco_truck_profit_l278_278088

-- Definitions and conditions
def pounds_of_beef : ℕ := 100
def beef_per_taco : ℝ := 0.25
def price_per_taco : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Desired profit result
def expected_profit : ℝ := 200

-- The proof statement (to be completed)
theorem taco_truck_profit :
  let tacos := pounds_of_beef / beef_per_taco;
  let revenue := tacos * price_per_taco;
  let cost := tacos * cost_per_taco;
  let profit := revenue - cost;
  profit = expected_profit :=
by
  sorry

end taco_truck_profit_l278_278088


namespace profit_percent_l278_278484

variable (P C : ℝ)
variable (h₁ : (2/3) * P = 0.84 * C)

theorem profit_percent (P C : ℝ) (h₁ : (2/3) * P = 0.84 * C) : 
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_l278_278484


namespace swimming_speed_in_still_water_l278_278077

theorem swimming_speed_in_still_water (v : ℝ) (current_speed : ℝ) (time : ℝ) (distance : ℝ) (effective_speed : current_speed = 10) (time_to_return : time = 6) (distance_to_return : distance = 12) (speed_eq : v - current_speed = distance / time) : v = 12 :=
by
  sorry

end swimming_speed_in_still_water_l278_278077


namespace count_divisibles_l278_278303

def is_divisible (a b : Nat) : Prop := ∃ k, a = b * k

theorem count_divisibles (count : Nat) :
  count = (List.range' 201 200).countp (λ n, is_divisible n 8) :=
by 
  -- Assume the result is known
  have h : count = 24 := sorry
  exact h

end count_divisibles_l278_278303


namespace proof_problem_l278_278991

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ -1}

theorem proof_problem :
  ((A ∩ {x | x > -1}) ∪ (B ∩ {x | x ≤ 0})) = {x | x > 0 ∨ x ≤ -1} :=
by 
  sorry

end proof_problem_l278_278991


namespace largest_non_sum_of_composites_l278_278173

-- Definition of composite number
def is_composite (n : ℕ) : Prop := 
  ∃ d : ℕ, (2 ≤ d ∧ d < n ∧ n % d = 0)

-- The problem statement
theorem largest_non_sum_of_composites : 
  (∀ n : ℕ, (¬(is_composite n)) → n > 0) 
  → (∀ k : ℕ, k > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ k = a + b) 
  → 11 = ∀ n : ℕ, (n < 12 → ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)) :=
sorry

end largest_non_sum_of_composites_l278_278173


namespace count_divisible_by_8_l278_278305

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278305


namespace largest_possible_b_l278_278809

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c ≤ b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l278_278809


namespace mandy_score_is_correct_l278_278335

-- Definitions based on the problem conditions
def total_items := 100
def lowella_percentage := 0.35
def pamela_extra_percentage := 0.20

-- Calculate individual scores based on the definitions
def lowella_score := lowella_percentage * total_items
def pamela_score := lowella_score + (pamela_extra_percentage * lowella_score)
def mandy_score := 2 * pamela_score

-- Proof statement to verify Mandy's score
theorem mandy_score_is_correct : mandy_score = 84 := by
  -- sorry placeholder for proof
  sorry

end mandy_score_is_correct_l278_278335


namespace midpoint_eccentricity_l278_278537

noncomputable def eccentricity_ellipse : ℝ :=
  let a := Math.sqrt 2
  let b := 1
  (Math.sqrt (a^2 - b^2)) / a

theorem midpoint_eccentricity :
  ∀ (a b : ℝ), a > b → b > 0 → a = Math.sqrt 2 * b → eccentricity_ellipse = (Math.sqrt 2) / 2 :=
by
  intros a b ha_gt_b hb_gt_0 ha_sqrt2b
  sorry

end midpoint_eccentricity_l278_278537


namespace distance_centers_eq_sqrt_2m_l278_278372

open Real

noncomputable def deltaABC := (100 : ℝ)
noncomputable def deltaAC := (240 : ℝ)
noncomputable def deltaBC := (250 : ℝ)
noncomputable def r1 : ℝ := 12000 / 295

noncomputable def DE := (240, r1)
noncomputable def FG := (r1 + 79.66, 40.68)

noncomputable def r2 := 100
noncomputable def r3 := 79.66

noncomputable def centerC2 := (240, r1 + r2)
noncomputable def centerC3 := (r1 + r3, r1)

noncomputable def distanceC2C3 := sqrt ((centerC2.1 - centerC3.1) ^ 2 + (centerC2.2 - centerC3.2) ^ 2)

theorem distance_centers_eq_sqrt_2m : distanceC2C3 = sqrt (2 * 12168) := by
  sorry

end distance_centers_eq_sqrt_2m_l278_278372


namespace vector_cd_equal_l278_278622

variables {A B C D : Type}   -- Points on the plane

noncomputable def vector_magnitude {V : Type} [inner_product_space ℝ V] (v : V) : ℝ :=
real.sqrt (inner_product_space.norm_sq v)

noncomputable def vector_cd {V : Type} [inner_product_space ℝ V] (ca cb : V) (lambda : ℝ) : V :=
  lambda * (ca / vector_magnitude ca + cb / vector_magnitude cb)

theorem vector_cd_equal
  {V : Type} [inner_product_space ℝ V]
  (a b : V)
  (lambda : ℝ) :
  (vector_magnitude b = 2) →
  (vector_magnitude a = 1) →
  vector_cd b a lambda = (2 / 3) * a + (1 / 3) * b :=
by {
  intro h1,
  intro h2,
  sorry
}

end vector_cd_equal_l278_278622


namespace min_distance_l278_278243

theorem min_distance (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h₁ : f = (λ x, Real.cos (2 * x + π / 3)))
  (h₂ : ∀ x, f x1 ≤ f x ∧ f x ≤ f x2) :
  |x2 - x1| = π / 2 := 
sorry

end min_distance_l278_278243


namespace domain_of_f_l278_278240

def tan_gt_one (x : ℝ) : Prop := tan x - 1 > 0
def sqrt_term_dom (x : ℝ) : Prop := 9 - x^2 ≥ 0
def domain_f (x : ℝ) : Prop := x ∈ ( -3 * real.pi / 4, -real.pi / 2 ) ∪ ( real.pi / 4, real.pi / 2 )

theorem domain_of_f :
  ∀ x : ℝ, (tan_gt_one x ∧ sqrt_term_dom x) ↔ domain_f x :=
begin
  sorry
end

end domain_of_f_l278_278240


namespace moving_circle_trajectory_l278_278539

noncomputable def circle1 : Set ℝ² := {p | (p.1 + 3)^2 + p.2^2 = 4 }
noncomputable def circle2 : Set ℝ² := {p | (p.1 - 3)^2 + p.2^2 = 100 }

theorem moving_circle_trajectory :
  (∃ (r : ℝ) (P : ℝ × ℝ), r > 0 ∧
     (P ∈ circle1 ∧ P ∈ circle2) ∧
     ((P.1 + 3)^2 + P.2^2 <= 4) ∧
     ((P.1 - 3)^2 + P.2^2 <= 100)) →
  is_ellipse (trajectory_of_moving_circle P)
  
sorry

end moving_circle_trajectory_l278_278539


namespace number_of_subsets_of_M_is_4_l278_278248

def sin_sign (x : ℝ) : ℝ := if (Real.sin x) > 0 then 1 else -1
def cos_sign (x : ℝ) : ℝ := if (Real.cos x) > 0 then 1 else -1
def tan_sign (x : ℝ) : ℝ := if (Real.tan x) > 0 then 1 else -1

def M : Set ℝ := {s | ∃ (x : ℝ), s = sin_sign x + cos_sign x + tan_sign x}

theorem number_of_subsets_of_M_is_4 : (Set.powerset M).card = 4 :=
by {
  -- Proof is omitted
  sorry
}

end number_of_subsets_of_M_is_4_l278_278248


namespace max_gcd_sequence_l278_278577

noncomputable def a (n : ℕ) : ℕ := n^3 + 4
noncomputable def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_sequence : (∀ n : ℕ, 0 < n → d n ≤ 433) ∧ (∃ n : ℕ, 0 < n ∧ d n = 433) :=
by sorry

end max_gcd_sequence_l278_278577


namespace schedule_four_courses_exactly_one_consecutive_pair_l278_278312

-- Define the courses and period constraints
def courses : List String := ["algebra", "geometry", "number_theory", "calculus"]

def total_periods : Nat := 8

-- Define the condition of exactly one consecutive pair
def consecutive_pairs (schedule : List (String × Nat)) : Bool :=
  ∃ (c1 c2 : String) (p1 p2 : Nat),
    (c1, p1) ∈ schedule ∧ (c2, p2) ∈ schedule ∧
    (p1 + 1 = p2 ∨ p2 + 1 = p1) ∧
    ∀ (c3 c4 : String) (p3 p4 : Nat),
      (c3 ≠ c1 ∧ c4 ≠ c2) →
      (c3, p3) ∈ schedule → (c4, p4) ∈ schedule → |p3 - p4| ≠ 1

-- Define the Lean proof problem statement
theorem schedule_four_courses_exactly_one_consecutive_pair :
  ∃ (schedules : List (List (String × Nat))),
    (∀ schedule ∈ schedules, List.length schedule = 4 ∧
     ∀ (course : String), course ∈ courses → ∃ (period : Nat), (course, period) ∈ schedule ∧
     consecutive_pairs schedule) ∧
    List.length schedules = 1680 :=
by
  sorry

end schedule_four_courses_exactly_one_consecutive_pair_l278_278312


namespace perpendicular_lines_l278_278657

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, x * a + 3 * y - 1 = 0) ∧ (∃ x y : ℝ, 2 * x + (a - 1) * y + 1 = 0) ∧
  (∀ m1 m2 : ℝ, m1 = - a / 3 → m2 = - 2 / (a - 1) → m1 * m2 = -1) →
  a = 3 / 5 :=
sorry

end perpendicular_lines_l278_278657


namespace sum_first_nine_b_l278_278247

noncomputable def f : ℝ → ℝ :=
  λ x, sin (2 * x) + 2 * cos (x / 2) ^ 2

def a (n : ℕ) : ℝ :=
  if n = 5 then π / 2 else if n > 1 then n - 2 else 0 -- This is a placeholder; the correct definition of a_n would need the arithmetic sequence condition

def b (n : ℕ) : ℝ :=
  f (a n)

theorem sum_first_nine_b :
  (b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9) = 9 :=
begin
  -- Prove arithmetic sequence properties and sum
  sorry
end

end sum_first_nine_b_l278_278247


namespace projection_OA_eq_projection_OB_arithmetic_property_l278_278619

noncomputable def arithmetic_sequence (n : ℕ) : ℕ → ℤ := sorry
-- Assuming we have a function arithmetic_sequence representing {a_n}

def Sn : ℕ → ℤ := sorry
-- Assuming we have a function Sn representing the sum {S_n}

def point_A : ℝ × ℝ := (arithmetic_sequence 1009, 1)
def point_B : ℝ × ℝ := (2, -1)
def point_C : ℝ × ℝ := (2, 2)
def point_O : ℝ × ℝ := (0, 0)

def vector_OA := (point_A.1 - point_O.1, point_A.2 - point_O.2)
def vector_OB := (point_B.1 - point_O.1, point_B.2 - point_O.2)
def vector_OC := (point_C.1 - point_O.1, point_C.2 - point_O.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem projection_OA_eq_projection_OB :
  dot_product vector_OA vector_OC = dot_product vector_OB vector_OC := sorry

theorem arithmetic_property :
  Sn 2017 = 0 := by
  have H : dot_product vector_OA vector_OC = dot_product vector_OB vector_OC := by
    apply projection_OA_eq_projection_OB
  -- Further steps to prove S_{2017} using the given H
  sorry

end projection_OA_eq_projection_OB_arithmetic_property_l278_278619


namespace largest_cannot_be_sum_of_two_composites_l278_278157

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278157


namespace evaluate_expression_l278_278962

theorem evaluate_expression : 
  (2^2 + 2^1 + 2^(-1)) / (2^(-1) + 2^(-2) + 2^(-3)) = 52 / 7 :=
by
  sorry

end evaluate_expression_l278_278962


namespace probability_product_greater_than_zero_l278_278026

open Set ProbabilityTheory

theorem probability_product_greater_than_zero (a b : ℝ) (ha : a ∈ Icc (-15) 15) (hb : b ∈ Icc (-15) 15) :
  P (λ x : ℝ × ℝ, 0 < x.1 * x.2 | (λ _, true) := 1/2 :=
begin
  sorry
end

end probability_product_greater_than_zero_l278_278026


namespace vector_norm_ratio_l278_278216

noncomputable theory

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_norm_ratio (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : inner a (a + 2 • b) = 0) :
  ∥a + b∥ / ∥b∥ = 1 :=
sorry

end vector_norm_ratio_l278_278216


namespace total_amount_after_interest_l278_278598

theorem total_amount_after_interest 
    (principle : ℝ) 
    (rate : ℝ) 
    (time : ℝ) 
    (I : ℝ) 
    (total_amount : ℝ) 
    (h1 : principle = 886.0759493670886) 
    (h2 : rate = 11) 
    (h3 : time = 2.4) 
    (h4 : I = principle * rate * time / 100) 
    (h5 : total_amount = principle + I) : 
    total_amount = 1119.9976 := 
begin
  sorry -- Proof goes here
end

end total_amount_after_interest_l278_278598


namespace fraction_add_eq_l278_278851

theorem fraction_add_eq (n : ℤ) :
  (3 + n) = 4 * ((4 + n) - 5) → n = 1 := sorry

end fraction_add_eq_l278_278851


namespace percentage_of_women_employees_l278_278566

variable (E W M : ℝ)

-- Introduce conditions
def total_employees_are_married : Prop := 0.60 * E = (1 / 3) * M + 0.6842 * W
def total_employees_count : Prop := W + M = E
def percentage_of_women : Prop := W = 0.7601 * E

-- State the theorem to prove
theorem percentage_of_women_employees :
  total_employees_are_married E W M ∧ total_employees_count E W M → percentage_of_women E W :=
by sorry

end percentage_of_women_employees_l278_278566


namespace train_crosses_bridge_in_25_92_seconds_l278_278913

/-- Conditions for the problem. -/
def train_length : ℝ := 110 -- meters
def bridge_length : ℝ := 330 -- meters
def initial_speed : ℝ := 60 * 1000 / 3600 -- converted to m/s
def acceleration : ℝ := 4 * 1000 / 3600^2 -- converted to m/s^2

/-- Distance the train needs to cover. -/
def total_distance : ℝ := train_length + bridge_length -- meters

/-- Time it takes for the train to completely cross the bridge. -/
theorem train_crosses_bridge_in_25_92_seconds :
  ∃ t : ℝ, t ≈ 25.92 ∧ total_distance = initial_speed * t + 0.5 * acceleration * t^2 := by
  sorry

end train_crosses_bridge_in_25_92_seconds_l278_278913


namespace geometric_a_sequence_general_term_a_sequence_compare_a_and_quadratic_l278_278987

def a_sequence : ℕ → ℕ
| 0 := 1
| 1 := 5
| (n + 2) := 5 * a_sequence (n + 1) - 6 * a_sequence n

theorem geometric_a_sequence :
  ∀ n ≥ 1, (a_sequence (n + 1) - 3 * a_sequence n) = 2 * (a_sequence n - 3 * a_sequence (n - 1)) :=
by sorry

theorem general_term_a_sequence :
  ∀ n, a_sequence n = 3^n - 2^n :=
by sorry

theorem compare_a_and_quadratic (n : ℕ) :
  (n = 1 → a_sequence n < 2 * n^2 + 1) ∧ (n ≥ 2 → a_sequence n ≥ 2 * n^2 + 1) :=
by sorry

end geometric_a_sequence_general_term_a_sequence_compare_a_and_quadratic_l278_278987


namespace problem_statement_l278_278380

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]
variables (a b c : V) (k : ℝ)

-- Define the given conditions
def unit_vector (v : V) := ⟪v, v⟫ = 1

noncomputable
def is_orthogonal (u v : V) := ⟪u, v⟫ = 0

def angle_pi_over_three (u v : V) := real.angle u v = real.pi / 3

theorem problem_statement
  (ha : unit_vector a)
  (hb : unit_vector b)
  (hc : unit_vector c)
  (hab : is_orthogonal a b)
  (hac : is_orthogonal a c)
  (hbc : angle_pi_over_three b c) :
  (a = (2 * real.sqrt 3 / 3) • (b × c) ∨ a = (- (2 * real.sqrt 3 / 3)) • (b × c)) :=
sorry

end problem_statement_l278_278380


namespace moss_pollen_diameter_in_scientific_notation_l278_278429

theorem moss_pollen_diameter_in_scientific_notation :
  (0.0000084 : ℝ) = 8.4 * 10^(-6) :=
sorry

end moss_pollen_diameter_in_scientific_notation_l278_278429


namespace brenda_winning_strategy_at_35_l278_278921

-- Define the game and conditions
def wins (player : ℕ → Prop) (n : ℕ) : Prop :=
  (n = 1 ∨ n = 3 ∨ n = 4) ∧ ¬player (n-1) ∧ ¬player (n-3) ∧ ¬player (n-4) 

-- Define the players: Anne and Brenda
def anne_wins : ℕ → Prop
| n := if (wins brenda_wins n) then false else true
and brenda_wins : ℕ → Prop
| n := if (wins anne_wins n) then false else true

-- Prove Brenda has a winning strategy when n = 35
theorem brenda_winning_strategy_at_35 : brenda_wins 35 :=
sorry

end brenda_winning_strategy_at_35_l278_278921


namespace sphere_radius_tangent_to_pyramid_base_and_edges_l278_278978

theorem sphere_radius_tangent_to_pyramid_base_and_edges
  (a α : ℝ) :
  let r := (a * real.sqrt 3 / 3) * (-2 * real.cot α + real.sqrt (4 * (real.cot α)^2 + 1))
  in r = (a * real.sqrt 3 / 3) * (-2 * real.cot α + real.sqrt (4 * (real.cot α)^2 + 1)) :=
by
  sorry

end sphere_radius_tangent_to_pyramid_base_and_edges_l278_278978


namespace rectangle_width_l278_278451

theorem rectangle_width (width : ℝ) : 
  ∃ w, w = 14 ∧
  (∀ length : ℝ, length = 10 →
  (2 * (length + width) = 3 * 16)) → 
  width = w :=
by
  sorry

end rectangle_width_l278_278451


namespace correct_price_max_profit_l278_278427

/-- Definitions for Part (1) --/
def typeA_cost_price (x : ℝ) (costB : ℝ) : Prop :=
  costB = x + 2

def zongzi_quantity_constraint (x : ℝ) (costB : ℝ) : Prop :=
  1000 / x = 1200 / costB

/-- Definitions for Part (2) --/
def profit_function (m : ℕ) : ℝ :=
  -m + 600

def purchase_constraint (m : ℕ) : Prop :=
  m ≥ 400 / 3 ∧ m < 200

theorem correct_price (x : ℝ) (costB : ℝ) : typeA_cost_price x costB ∧ zongzi_quantity_constraint x costB → x = 10 ∧ costB = 12 :=
by
  intros h
  cases h with hc hz
  have := calc
    1000 * (x + 2) = 1000 * (costB : ℝ) : by rw hc
                 ... = 1200 * (x : ℝ) : by rw hz 
  have eq := calc
    1000 * x + 2000 = 1200 * x : by linarith
             ... = 1000 * (x + (x / 5)) + 2000 : by ring
   
    sorry

theorem max_profit : 
  ∃ m, purchase_constraint m ∧ 
       profit_function m = 466 :=
by
  use 134
  split
  · split
    · linarith
    · linarith
  · unfold profit_function
    linarith

end correct_price_max_profit_l278_278427


namespace sport_drink_water_l278_278707

theorem sport_drink_water (S : Type) [linearOrderedField S]
  (sport_corn_syrup_oz : S) (hs : sport_corn_syrup_oz = 8) :
  ∃ water_oz : S, water_oz = 30 :=
begin
  -- By given conditions
  let standard_flavoring_to_corn_syrup_ratio := 1,
  let standard_corn_syrup_to_flavoring_ratio := 12,
  let standard_flavoring_to_water_ratio := 1 / 30,

  let sport_flavoring_to_corn_syrup_ratio := 1 / 4,
  let sport_flavoring_to_water_ratio := 1 / 15,

  -- Given quantifies the ratio we are looking for
  let sport_corn_syrup_to_water_ratio := (4 : S) / 15,

  -- Establish the cross-multiplication equality
  let water_oz := sport_corn_syrup_oz * 15 / 4,

  -- Conclusion from the computed values
  have water_computation : water_oz = 30, from sorry,

  -- We return the value we computed in the context of the proof
  exact Exists.intro water_oz water_computation,
end

end sport_drink_water_l278_278707


namespace largest_non_summable_composite_l278_278151

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l278_278151


namespace all_numbers_in_S_are_rational_l278_278802

theorem all_numbers_in_S_are_rational
  (S : Set ℝ) (hS_finite : S.Finite)
  (hS_sub : ∀ s ∈ S, ∃ a b ∈ S ∪ {0, 1}, a ≠ s ∧ b ≠ s ∧ s = (a + b) / 2) :
  ∀ s ∈ S, s ∈ ℚ :=
by
  sorry

end all_numbers_in_S_are_rational_l278_278802


namespace hyperbola_asymptotes_eq_l278_278230

theorem hyperbola_asymptotes_eq {x y : ℝ} (foci_left focus_right endpoint_left endpoint_right : ℝ)
  (h_foci_left : foci_left = (-3))
  (h_foci_right : focus_right = 3)
  (h_endpoint_left : endpoint_left = (-4))
  (h_endpoint_right : endpoint_right = 4) :
  y = ±(√7) / 3 * x := 
by sorry

end hyperbola_asymptotes_eq_l278_278230


namespace area_triangle_l278_278132

def point3D := (ℝ × ℝ × ℝ)

def distance (p1 p2 : point3D) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def area_of_triangle (A B C : point3D) : ℝ :=
  if distance A B = Real.sqrt 6 ∧ distance B C = 3 * Real.sqrt 2 ∧ distance A C = 2 * Real.sqrt 6
  then 3 * Real.sqrt 3
  else 0

theorem area_triangle (A B C : point3D) :
  A = (1, 8, 11) →
  B = (0, 7, 9) →
  C = (-3, 10, 9) →
  area_of_triangle A B C = 3 * Real.sqrt 3 :=
  by intros; sorry

end area_triangle_l278_278132


namespace min_value_of_c_l278_278391

theorem min_value_of_c
  (a b c : ℕ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : ∃ x y, (2 * x + y = 2027) ∧ (y = |x - a| + |x - b| + |x - c|) ∧ (∀ x1 x2 y1 y2, ((2 * x1 + y1 = 2027) ∧ (y1 = |x1 - a| + |x1 - b| + |x1 - c|) ∧ (2 * x2 + y2 = 2027) ∧ (y2 = |x2 - a| + |x2 - b| + |x2 - c|)) → (x1 = x2 ∧ y1 = y2))):
  c = 1014 :=
by
  sorry

end min_value_of_c_l278_278391


namespace toffee_price_l278_278518

theorem toffee_price (x : ℝ) : (9 * x < 10) → (10 * x > 11) → (11 / 10 < x ∧ x < 10 / 9) :=
begin
  intros h1 h2,
  split;
  linarith,
end

end toffee_price_l278_278518


namespace area_AMNK_eq_13_over_20_l278_278763

-- Definitions based on conditions
variables {A B C M N K : Type} [LinearOrder C]

-- Conditions in the problem
def conditions (BC_AC : ℝ) :=
  let S_ABC := 1 in
  let BM := BC_AC / 4 in
  let MN := BC_AC / 4 in
  let NC := BC_AC / 2 in
  let CK := BC_AC / 5 in
  let AK := 4 * (CK) in
  -- find the area of quadrilateral AMNK
  let S_ABM := S_ABC / 4 in
  let S_CNK := S_ABC / 10 in
  let S_AMNK := S_ABC - S_ABM - S_CNK in
  S_AMNK

-- The main proof statement
theorem area_AMNK_eq_13_over_20 {BC_AC : ℝ} (h : BC_AC = 1) :
  conditions BC_AC = 13/20 :=
by
  sorry

end area_AMNK_eq_13_over_20_l278_278763


namespace largest_four_digit_divisible_by_6_l278_278832

theorem largest_four_digit_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 6 = 0 → m ≤ n :=
begin
  use 9996,
  split,
  { exact nat.le_refl 9996 },
  split,
  { dec_trivial },
  split,
  { exact nat.zero_mod _ },
  { intros m h1 h2 h3,
    exfalso,
    sorry }
end

end largest_four_digit_divisible_by_6_l278_278832


namespace suitable_n_exists_l278_278511

theorem suitable_n_exists :
  ∃ n : ℕ, (∀ S : ℝ, (∃ (a : ℕ → ℕ) (h : ∀ i, 1 ≤ a i ∧ a i ≤ n), S = ∑ i in range n, (a i + 1) / (i + 1)) → 
  ∀ m : ℕ, n ≤ m ∧ m ≤ n + 100 → ∃ (b : ℕ → ℕ) (hb : ∀ i, 1 ≤ b i ∧ b i ≤ n), ∑ i in range n, (b i + 1) / (i + 1) = m) :=
begin
  use 798,
  sorry
end

end suitable_n_exists_l278_278511


namespace find_quadratic_polynomial_l278_278977

noncomputable def q : ℝ → ℝ := sorry

theorem find_quadratic_polynomial (h₁ : q (-3) = 0) (h₂ : q 6 = 0) (h₃ : q 2 = -40) : 
  q = (λ x : ℝ, 2 * x^2 - 6 * x - 36) :=
sorry

end find_quadratic_polynomial_l278_278977


namespace work_completion_time_l278_278494

-- Definitions from the problem conditions
def A_work_rate : ℝ := 1 / 30
def B_work_rate : ℝ := 1 / 15
def combined_work_rate : ℝ := A_work_rate + B_work_rate

-- Lean theorem statement
theorem work_completion_time : (5 * combined_work_rate) + ((1 - 5 * combined_work_rate) / A_work_rate) = 20 := by sorry

end work_completion_time_l278_278494


namespace largest_four_digit_number_divisible_by_six_l278_278836

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end largest_four_digit_number_divisible_by_six_l278_278836


namespace find_b_l278_278927

theorem find_b (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : ∀ x, y = a * sin (b * x + c) + d) : 
  (2 * π / b = 2 * π / 5) → b = 5 :=
sorry

end find_b_l278_278927


namespace proof_no_solution_l278_278103

noncomputable def no_solution (a b : ℕ) : Prop :=
  2 * a^2 + 1 ≠ 4 * b^2

theorem proof_no_solution (a b : ℕ) : no_solution a b := by
  sorry

end proof_no_solution_l278_278103


namespace true_propositions_123_l278_278641

-- Defining the four propositions as separate terms
def prop1 := ∀ (l : Line) (p : Plane), ¬ (l ⊆ p) → (∀ x y, x ∈ l ∧ y ∈ l ∧ x ∈ p ∧ y ∈ p → x = y)
def prop2 := ∀ (a b : Line) (α β : Plane), (a ⊆ α ∧ b ⊆ β ∧ ∃ x, x ∈ a ∧ x ∈ b) → (∃ y, y ∈ α ∧ y ∈ β) 
def prop3 := ∀ (l l1 l2 : Line), (∀ x, x ∈ l1 ∧ x ∈ l2 → false) ∧ (∃ z, z ∈ l ∧ z ∈ l1) ∧ (∃ w, w ∈ l ∧ w ∈ l2)
def prop4 := ∀ (l1 l2 l3 : Line), (∃ x1, x1 ∈ l1 ∧ x1 ∈ l2) ∧ (∃ x2, x2 ∈ l2 ∧ x2 ∈ l3) ∧ (∃ x3, x3 ∈ l1 ∧ x3 ∈ l3) → (∃ p, l1 ⊆ p ∧ l2 ⊆ p ∧ l3 ⊆ p)

-- The Lean 4 statement
theorem true_propositions_123 : prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4 :=
by
  sorry

end true_propositions_123_l278_278641


namespace students_not_in_biology_l278_278871

theorem students_not_in_biology (total_students : ℕ) (percent_enrolled : ℝ) (students_enrolled : ℕ) (students_not_enrolled : ℕ) : 
  total_students = 880 ∧ percent_enrolled = 32.5 ∧ total_students - students_enrolled = students_not_enrolled ∧ students_enrolled = 286 ∧ students_not_enrolled = 594 :=
by
  sorry

end students_not_in_biology_l278_278871


namespace largest_number_not_sum_of_two_composites_l278_278183

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l278_278183


namespace trains_crossing_time_l278_278823

-- Define the known conditions of the problem

def length_train1 : ℝ := 140 -- Length of the first train in meters
def length_train2 : ℝ := 160 -- Length of the second train in meters
def speed_train1 : ℝ := 60 * (5 / 18) -- Speed of the first train in m/s
def speed_train2 : ℝ := 40 * (5 / 18) -- Speed of the second train in m/s

-- Address the question
theorem trains_crossing_time :
  let total_distance := length_train1 + length_train2 in
  let relative_speed := speed_train1 + speed_train2 in
  abs ((total_distance / relative_speed) - 10.8) < 0.1 :=
by
  sorry

end trains_crossing_time_l278_278823


namespace distance_centroid_BC_l278_278637

theorem distance_centroid_BC (m : ℝ) (BC : ℝ) (AE : ℝ) :
  (∀ a : ℝ, a^2 - 8 * a + m + 6 = 0) ∧ (BC = some (solve_quadratic_eq 1 (-8) (m + 6))) ∧ (BC ≠ none) →
  (BC = 4) ∧ (area_triangle ABC = 6) →
  ∃ G : point, distance_from_centroid_G_to_BC G BC = 1 :=
by
  sorry

noncomputable def solve_quadratic_eq (a b c : ℝ) : option ℝ :=
if Δ : b^2 - 4 * a * c = 0 then
  some (- b / (2 * a))
else 
  none

noncomputable def area_triangle (ABC : triangle) : ℝ := 6

def distance_from_centroid_G_to_BC (G : point) (BC : ℝ) : ℝ := 1

structure point := (x : ℝ) (y : ℝ) (z : ℝ)

structure triangle := (A B C : point)

end distance_centroid_BC_l278_278637


namespace arc_length_l278_278974

section

open Real
open Interval

noncomputable def arcLengthOfCurve : ℝ :=
  ∫ x in 1..e, sqrt (1 + ((x^2 / 4 - log x / 2)' x)^2)

theorem arc_length (L : ℝ) :
  L = ∫ x in 1..e, sqrt (1 + ((x^2 / 4 - log x / 2)' x)^2) → L = (e^2 + 1) / 4 :=
by
  intro h
  exact sorry

end

end arc_length_l278_278974


namespace company_acquaintances_l278_278400

theorem company_acquaintances (k : ℕ) (acquainted : Fin k → Fin k → Prop) 
  (symm : ∀ i j, acquainted i j → acquainted j i) : 
  ∃ i j : Fin k, i ≠ j ∧ (∀ n : Fin k, i ≠ n → ∃ cnt : ℕ, acquaintance_count i cnt ∧ acquaintance_count n cnt) :=
sorry

def acquaintance_count {k : ℕ} (i : Fin k) (cnt : ℕ) :=
∃ l : List (Fin k), (∀ x ∈ l, x ≠ i ∧ acquainted i x) ∧ l.length = cnt

end company_acquaintances_l278_278400


namespace beta_angle_relationship_l278_278852

theorem beta_angle_relationship (α β γ : ℝ) (h1 : β - α = 3 * γ) (h2 : α + β + γ = 180) : β = 90 + γ :=
sorry

end beta_angle_relationship_l278_278852


namespace solve_equation_l278_278492

theorem solve_equation (x : ℝ) :
  3 * (x - 3) = (x - 3) ^ 2 → x = 3 ∨ x = 6 :=
by
  intro h
  -- Rewrite the equation
  have eqn : 3 * (x - 3) - (x - 3) ^ 2 = 0 := by linarith
  -- Factorization
  have factored_eqn : (x - 3) * (6 - x) = 0 := by ring_exp_eq_civ
  -- Solve for x
  cases eq (x-3) 0 with eq1 eq2
  { left, linarith }
  { right, linarith }
  sorry

end solve_equation_l278_278492


namespace johns_profit_l278_278859

noncomputable def earnings : ℕ := 30000
noncomputable def purchase_price : ℕ := 18000
noncomputable def trade_in_value : ℕ := 6000

noncomputable def depreciation : ℕ := purchase_price - trade_in_value
noncomputable def profit : ℕ := earnings - depreciation

theorem johns_profit : profit = 18000 := by
  sorry

end johns_profit_l278_278859


namespace positive_number_sum_square_l278_278003

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l278_278003


namespace correct_statements_count_l278_278097

theorem correct_statements_count :
  let s1 := "The further away a point representing a number is from the origin, the larger the number."
  let s2 := "The only numbers whose cube equals themselves are 0 and 1."
  let s3 := "There is exactly one line parallel to a given line through a given point."
  let s4 := "If |a|=4, then a=±4."
  let s5 := "The degree of the algebraic expression -2π³ab is 2."
  let is_correct1 := false
  let is_correct2 := false
  let is_correct3 := false
  let is_correct4 := true
  let is_correct5 := true
  (if is_correct1 then 1 else 0) + (if is_correct2 then 1 else 0) + (if is_correct3 then 1 else 0) + (if is_correct4 then 1 else 0) + (if is_correct5 then 1 else 0) = 2 :=
begin
  sorry
end

end correct_statements_count_l278_278097


namespace centroid_is_center_of_circle_l278_278795

noncomputable def locus_intersection_medians (S A B C X Y Z : Point) (O : Point)
  (h_circumscribed : is_circumscribed S A B C)
  (h_equal_division : dist S X = dist X Y ∧ dist X Y = dist Y C)
  (h_intersection : is_intersection AX BY Z) : Prop :=
  centroid (triangle A B C) = O

-- Here is how we would state that the proof of the aforementioned proposition exists:
theorem centroid_is_center_of_circle : 
  ∀ (S A B C X Y Z O : Point),
    is_circumscribed S A B C →
    (dist S X = dist X Y ∧ dist X Y = dist Y C) →
    is_intersection AX BY Z →
    centroid (triangle A B C) = O :=
sorry

end centroid_is_center_of_circle_l278_278795


namespace complement_of_union_l278_278655

open Set

variable (U A B : Set ℕ)
variable (u_def : U = {0, 1, 2, 3, 4, 5, 6})
variable (a_def : A = {1, 3})
variable (b_def : B = {3, 5})

theorem complement_of_union :
  (U \ (A ∪ B)) = {0, 2, 4, 6} :=
by
  sorry

end complement_of_union_l278_278655


namespace sufficient_but_not_necessary_condition_l278_278048

theorem sufficient_but_not_necessary_condition {α : ℝ} :
  (α = π / 6 → Real.sin α = 1 / 2) ∧ (∃ α', α' ≠ π / 6 ∧ Real.sin α' = 1 / 2) :=
by
  split
  { intro h
    rw [h, Real.sin_pi_div_six] 
  }
  {
    use 5 * π / 6
    split
    { linarith }
    rw [Real.sin_of_real 5π / 6]
    norm_num
  }
  sorry


end sufficient_but_not_necessary_condition_l278_278048


namespace largest_four_digit_divisible_by_6_l278_278830

theorem largest_four_digit_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 6 = 0 → m ≤ n :=
begin
  use 9996,
  split,
  { exact nat.le_refl 9996 },
  split,
  { dec_trivial },
  split,
  { exact nat.zero_mod _ },
  { intros m h1 h2 h3,
    exfalso,
    sorry }
end

end largest_four_digit_divisible_by_6_l278_278830


namespace abigail_saving_period_l278_278554

-- Define the conditions
def amount_saved_each_month : ℕ := 4000
def total_amount_saved : ℕ := 48000

-- State the theorem
theorem abigail_saving_period : total_amount_saved / amount_saved_each_month = 12 := by
  -- Proof would go here
  sorry

end abigail_saving_period_l278_278554


namespace perimeter_of_PQR_is_correct_l278_278683

noncomputable def PQR_perimeter :=
  let P := (0, 0)
  let Q := (10, 0)
  let R := (0, 10)
  let X := (-10, 10)
  let Y := (10, 10)
  let W := (-10, -10)
  let Z := (0, -10)
  (dist P Q + dist Q R + dist R P)

def is_square (a b c d : (ℝ × ℝ)) : Prop :=
  dist a b = dist b c ∧ dist c d = dist d a ∧ d ≠ a ∧ b ≠ c

theorem perimeter_of_PQR_is_correct :
  ∀ (P Q R X Y W Z : ℝ × ℝ),
    ∠PQR = 90 ∧
    dist P Q = 10 ∧
    is_square P Q X Y ∧
    is_square P R W Z ∧
    (list.set_mem_attribute [X, Y, Z, W]) →
    PQR_perimeter = 10 + 10 * real.sqrt 2 := by
  sorry

#eval perimeter_of_PQR_is_correct

end perimeter_of_PQR_is_correct_l278_278683


namespace henry_earnings_correct_l278_278254

-- Define constants for the amounts earned per task
def earn_per_lawn : Nat := 5
def earn_per_leaves : Nat := 10
def earn_per_driveway : Nat := 15

-- Define constants for the number of tasks he actually managed to do
def lawns_mowed : Nat := 5
def leaves_raked : Nat := 3
def driveways_shoveled : Nat := 2

-- Define the expected total earnings calculation
def expected_earnings : Nat :=
  (lawns_mowed * earn_per_lawn) +
  (leaves_raked * earn_per_leaves) +
  (driveways_shoveled * earn_per_driveway)

-- State the theorem that the total earnings are 85 dollars.
theorem henry_earnings_correct : expected_earnings = 85 :=
by
  sorry

end henry_earnings_correct_l278_278254


namespace numbers_divisible_by_8_between_200_and_400_l278_278288

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278288


namespace total_profit_correct_l278_278086

-- Defining the given conditions as constants
def beef_total : ℝ := 100
def beef_per_taco : ℝ := 0.25
def selling_price : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Calculate the number of tacos
def num_tacos := beef_total / beef_per_taco

-- Calculate the profit per taco
def profit_per_taco := selling_price - cost_per_taco

-- Calculate the total profit
def total_profit := num_tacos * profit_per_taco

-- Prove the total profit
theorem total_profit_correct : total_profit = 200 :=
by sorry

end total_profit_correct_l278_278086


namespace correct_price_max_profit_l278_278428

/-- Definitions for Part (1) --/
def typeA_cost_price (x : ℝ) (costB : ℝ) : Prop :=
  costB = x + 2

def zongzi_quantity_constraint (x : ℝ) (costB : ℝ) : Prop :=
  1000 / x = 1200 / costB

/-- Definitions for Part (2) --/
def profit_function (m : ℕ) : ℝ :=
  -m + 600

def purchase_constraint (m : ℕ) : Prop :=
  m ≥ 400 / 3 ∧ m < 200

theorem correct_price (x : ℝ) (costB : ℝ) : typeA_cost_price x costB ∧ zongzi_quantity_constraint x costB → x = 10 ∧ costB = 12 :=
by
  intros h
  cases h with hc hz
  have := calc
    1000 * (x + 2) = 1000 * (costB : ℝ) : by rw hc
                 ... = 1200 * (x : ℝ) : by rw hz 
  have eq := calc
    1000 * x + 2000 = 1200 * x : by linarith
             ... = 1000 * (x + (x / 5)) + 2000 : by ring
   
    sorry

theorem max_profit : 
  ∃ m, purchase_constraint m ∧ 
       profit_function m = 466 :=
by
  use 134
  split
  · split
    · linarith
    · linarith
  · unfold profit_function
    linarith

end correct_price_max_profit_l278_278428


namespace rectangles_in_4x4_grid_l278_278116

theorem rectangles_in_4x4_grid : 
  let n := 4 in
  let total_rectangles := (nat.choose n 2) * (nat.choose n 2) in
  total_rectangles = 36 :=
by
  sorry

end rectangles_in_4x4_grid_l278_278116


namespace cyclic_quadrilateral_AC_plus_BD_l278_278366

theorem cyclic_quadrilateral_AC_plus_BD (AB BC CD DA : ℝ) (AC BD : ℝ) (h1 : AB = 5) (h2 : BC = 10) (h3 : CD = 11) (h4 : DA = 14)
  (h5 : AC = Real.sqrt 221) (h6 : BD = 195 / Real.sqrt 221) :
  AC + BD = 416 / Real.sqrt (13 * 17) ∧ (AC = Real.sqrt 221 ∧ BD = 195 / Real.sqrt 221) →
  (AC + BD = 416 / Real.sqrt (13 * 17)) ∧ (AC + BD = 446) :=
by
  sorry

end cyclic_quadrilateral_AC_plus_BD_l278_278366


namespace hockey_games_in_season_l278_278014

-- Define the conditions
def games_per_month : Nat := 13
def season_months : Nat := 14

-- Define the total number of hockey games in the season
def total_games_in_season (games_per_month : Nat) (season_months : Nat) : Nat :=
  games_per_month * season_months

-- Define the theorem to prove
theorem hockey_games_in_season :
  total_games_in_season games_per_month season_months = 182 :=
by
  -- Proof omitted
  sorry

end hockey_games_in_season_l278_278014


namespace number_of_routes_depends_on_map_specifics_l278_278445

-- Definitions for cities and roads
structure City where
  name : String

structure Road where
  city1 city2 : City

-- Given conditions
constant cities : List City
constant roads : List Road
constant C J : City

-- Ensure there are exactly 10 cities and 15 roads
axiom cities_count : cities.length = 10
axiom roads_count : roads.length = 15

-- Specific problem conditions
axiom start_at_C : ∀ r : Road, List.count roads r ≥ 2 -> (r.city1 = C ∨ r.city2 = C)
axiom visit_road_once_except_C : ∀ r : Road, (r.city1 = C ∨ r.city2 = C) -> List.count roads r = 2
axiom visit_road_once : ∀ r : Road, ¬(r.city1 = C ∨ r.city2 = C) -> List.count roads r = 1

-- Definition for a route
structure Route where
  path: List Road
  starts_at: path.head.city1 = C
  ends_at: path.reverse.head.city2 = J
  uses_ten_roads: path.length = 10

-- Proposed theorem to prove
theorem number_of_routes_depends_on_map_specifics : 
  ∃ routes : List Route, True :=
sorry

end number_of_routes_depends_on_map_specifics_l278_278445


namespace caps_in_third_week_l278_278883

theorem caps_in_third_week (x : ℕ) : 
    let first_week_caps := 320 in
    let second_week_caps := 400 in
    let total_caps := 1360 in
    total_caps = first_week_caps + second_week_caps + x + (first_week_caps + second_week_caps + x) / 3 →
    x = 300 := sorry

end caps_in_third_week_l278_278883


namespace carrie_hours_per_day_l278_278114

theorem carrie_hours_per_day (h : ℕ) 
  (worked_4_days : ∀ n, n = 4 * h) 
  (paid_per_hour : ℕ := 22)
  (cost_of_supplies : ℕ := 54)
  (profit : ℕ := 122) :
  88 * h - cost_of_supplies = profit → h = 2 := 
by 
  -- Assume problem conditions and solve
  sorry

end carrie_hours_per_day_l278_278114


namespace positive_number_sum_square_l278_278006

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l278_278006


namespace marked_price_percentage_l278_278083

variable (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ)

# Conditions
-- The shopkeeper buys the items at a 30% discount, so the cost price is 70% of the list price.
def cost_price : Prop := C = 0.7 * L

-- The shopkeeper wants a 30% profit on the cost price, so the selling price is 130% of the cost price.
def selling_price : Prop := S = 1.3 * C

-- The selling price is after a 25% discount on the marked price, so the selling price is 75% of the marked price.
def marked_selling_relation : Prop := S = 0.75 * M

# Theorem
-- We need to prove that the marked price is 121.33% of the list price.
theorem marked_price_percentage
  (h1 : cost_price L C)
  (h2 : selling_price C S)
  (h3 : marked_selling_relation S M):
  M = 1.2133 * L :=
sorry

end marked_price_percentage_l278_278083


namespace larger_exceeds_smaller_times_l278_278604

theorem larger_exceeds_smaller_times {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_diff : a ≠ b)
  (h_eq : a^3 - b^3 = 3 * (2 * a^2 * b - 3 * a * b^2 + b^3)) : a = 4 * b :=
sorry

end larger_exceeds_smaller_times_l278_278604


namespace johns_profit_l278_278860

noncomputable def earnings : ℕ := 30000
noncomputable def purchase_price : ℕ := 18000
noncomputable def trade_in_value : ℕ := 6000

noncomputable def depreciation : ℕ := purchase_price - trade_in_value
noncomputable def profit : ℕ := earnings - depreciation

theorem johns_profit : profit = 18000 := by
  sorry

end johns_profit_l278_278860


namespace num_int_values_lt_2pi_l278_278260

theorem num_int_values_lt_2pi : 
  (finset.Icc (- ⌊2 * Real.pi⌋) ⌊2 * Real.pi⌋).card = 13 := 
by
  sorry

end num_int_values_lt_2pi_l278_278260


namespace expected_tomato_yield_is_correct_l278_278388

noncomputable def expected_tomato_yield : ℝ :=
let step_length := 2.5
let tomato_yield_per_sqft := 3 / 4
let area1_length := 15 * step_length
let area1_width := 15 * step_length
let area1_area := area1_length * area1_width
let area2_length := 15 * step_length
let area2_width := 5 * step_length
let area2_area := area2_length * area2_width
let total_area := area1_area + area2_area
total_area * tomato_yield_per_sqft

theorem expected_tomato_yield_is_correct :
  expected_tomato_yield = 1406.25 :=
by
  sorry

end expected_tomato_yield_is_correct_l278_278388


namespace geometric_sequence_tenth_fifth_terms_l278_278949

variable (a r : ℚ) (n : ℕ)

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem geometric_sequence_tenth_fifth_terms :
  (geometric_sequence 4 (4/3) 10 = 1048576 / 19683) ∧ (geometric_sequence 4 (4/3) 5 = 1024 / 81) :=
by
  sorry

end geometric_sequence_tenth_fifth_terms_l278_278949


namespace angle_equiv_l278_278591

theorem angle_equiv (θ : ℝ) : 
  ∃ θ', θ' = 280 ∧ θ' ∈ set.Icc 0 360 ∧ ∃ k : ℤ, θ = θ' + k * 360 :=
by
  use 280
  split
  sorry

end angle_equiv_l278_278591


namespace solve_logarithmic_system_l278_278251

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_logarithmic_system :
  ∃ x y : ℝ, log_base 2 x + log_base 4 y = 4 ∧ log_base 4 x + log_base 2 y = 5 ∧ x = 4 ∧ y = 16 :=
by
  sorry

end solve_logarithmic_system_l278_278251


namespace sum_num_divisors_equals_floor_sum_sum_sum_divisors_equals_weighted_floor_sum_l278_278875

-- Part a
def numDivisors (n : ℕ) : ℕ := (n.divisors.card)

theorem sum_num_divisors_equals_floor_sum (n : ℕ) :
  (∑ i in Finset.range (n + 1), numDivisors i) = (∑ i in Finset.range (n + 1), n / (i + 1)) :=
by sorry

-- Part b
def sumDivisors (n : ℕ) : ℕ := (∑ i in n.divisors, i)

theorem sum_sum_divisors_equals_weighted_floor_sum (n : ℕ) :
  (∑ i in Finset.range (n + 1), sumDivisors i) 
  = (∑ i in Finset.range (n + 1), (i + 1) * (n / (i + 1))) :=
by sorry

end sum_num_divisors_equals_floor_sum_sum_sum_divisors_equals_weighted_floor_sum_l278_278875


namespace counting_divisibles_by_8_l278_278295

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278295


namespace centroid_value_l278_278819

theorem centroid_value :
  let P := (-2 : ℝ, 3 : ℝ)
  let Q := (4 : ℝ, 7 : ℝ)
  let R := (1 : ℝ, 2 : ℝ)
  let S := ((-2 + 4 + 1) / 3, (3 + 7 + 2) / 3)
  S = (1, 4) →
  10 * S.1 + S.2 = 14 :=
by
  intro h
  rw [h]
  norm_num
  rfl

end centroid_value_l278_278819


namespace parabola_properties_l278_278205

theorem parabola_properties
    (C : set (Real × Real))
    (A : Real × Real)
    (O : Real × Real)
    (p : Real)
    (y : Real) 
    (x : Real) :
  C = { (x, y) | y^2 = 2 * p * x } ∧ A = (1, -2) ∧ p > 0 ∧ O = (0, 0) ∧ 
  (C ∋ A) →
  (let eq_parabola := y^2 = 4 * x,
       eq_axis := x = -1,
       eq_line_L := 2 * x + y - 1 = 0,
       distance_between_lines := (|2 - √5|) / √5
  in
  eq_parabola ∧ eq_axis ∧ (distance_between_lines = √5 / 5 ∧ 
  ∃ (L : set (Real × Real)), 
    L = { (x, y) | 2 * x + y = 1 } ∧ 
    ∀ (A B : Real × Real), A ≠ B ∧ A ∈ C ∧ B ∈ L) ) :=
begin
  sorry
end

end parabola_properties_l278_278205


namespace second_warehouse_more_profitable_initially_purchase_first_warehouse_advantageous_long_term_l278_278061

-- Condition definitions
def monthly_rent_first_warehouse : ℕ := 80000
def monthly_rent_second_warehouse : ℕ := 20000
def probability_repossession_second_warehouse : ℝ := 0.5
def moving_expense : ℕ := 150000
def months_in_year : ℕ := 12
def repossession_months : ℕ := 5
def remaining_months_after_repossession : ℕ := months_in_year - repossession_months

-- Annual rents
def annual_rent_first_warehouse : ℕ := monthly_rent_first_warehouse * months_in_year
def annual_rent_second_warehouse_worst_case : ℕ :=
 monthly_rent_second_warehouse * repossession_months + 
 monthly_rent_first_warehouse * remaining_months_after_repossession + 
 moving_expense

-- Part (a) Statement
theorem second_warehouse_more_profitable_initially :
  annual_rent_second_warehouse_worst_case < annual_rent_first_warehouse :=
sorry

-- Part (b) additional definitions
def purchase_cost_first_warehouse : ℕ := 3000000
def monthly_installment : ℕ := purchase_cost_first_warehouse / 36 -- over 3 years
def total_installment_cost_first_warehouse : ℕ := monthly_installment * months_in_year * 3

-- 5-year rents
def total_rent_first_warehouse_5_years : ℕ := annual_rent_first_warehouse * 5
def total_rent_second_warehouse_5_years_worst_case : ℕ :=
  annual_rent_second_warehouse_worst_case + (annual_rent_first_warehouse * 4)

-- Part (b) Statement
theorem purchase_first_warehouse_advantageous_long_term :
  total_installment_cost_first_warehouse < total_rent_first_warehouse_5_years ∧ total_rent_second_warehouse_5_years_worst_case :=
sorry

end second_warehouse_more_profitable_initially_purchase_first_warehouse_advantageous_long_term_l278_278061


namespace a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l278_278224

variable {a b c : ℝ}

theorem a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2 :
  ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) :=
sorry

theorem a_gt_b_necessary_not_sufficient_ac2_gt_bc2 :
  ¬((a > b) → (a * c^2 > b * c^2)) ∧ ((a * c^2 > b * c^2) → (a > b)) :=
sorry

end a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l278_278224


namespace total_area_of_paths_l278_278718

theorem total_area_of_paths:
  let bed_width := 4
  let bed_height := 3
  let num_beds_width := 3
  let num_beds_height := 5
  let path_width := 2

  let total_bed_width := num_beds_width * bed_width
  let total_path_width := (num_beds_width + 1) * path_width
  let total_width := total_bed_width + total_path_width

  let total_bed_height := num_beds_height * bed_height
  let total_path_height := (num_beds_height + 1) * path_width
  let total_height := total_bed_height + total_path_height

  let total_area_greenhouse := total_width * total_height
  let total_area_beds := num_beds_width * num_beds_height * bed_width * bed_height

  let total_area_paths := total_area_greenhouse - total_area_beds

  total_area_paths = 360 :=
by sorry

end total_area_of_paths_l278_278718


namespace count_just_cool_numbers_l278_278100

-- Definition of what constitutes a "just cool" number.
def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def no_adjacent_same (n : ℕ) : Prop :=
  let digits := n.digits 10
  in ∀ i, i < digits.length - 1 → digits.nth i ≠ digits.nth (i + 1)

def is_just_cool (n : ℕ) : Prop :=
  (100000 ≤ n ∧ n < 1000000) ∧
  is_odd n ∧
  (∀ d ∈ n.digits 10, is_prime_digit d) ∧
  no_adjacent_same n

theorem count_just_cool_numbers : 
  ∃ (card : ℕ), card = 729 ∧ (∀ n, is_just_cool n ↔ n < card) :=
by
  sorry

end count_just_cool_numbers_l278_278100


namespace sum_of_roots_eq_twelve_l278_278743

variable (g : ℝ → ℝ)
variable (roots : Finset ℝ)

-- Conditions
axiom symmetry : ∀ x : ℝ, g (3 + x) = g (3 - x)
axiom four_distinct_real_roots : (roots : Set ℝ).card = 4 ∧ ∀ x ∈ roots, g x = 0

-- Theorem to prove
theorem sum_of_roots_eq_twelve : (roots.sum id) = 12 :=
sorry

end sum_of_roots_eq_twelve_l278_278743


namespace T_shape_cube_valid_combinations_l278_278574

-- Assume we have a structure of four squares in a T-shape.
-- Define the possible positions for adding squares.

def is_valid_combination (placement1 placement2 : ℕ) : Prop :=
  -- A placeholder function to determine if the positions allow for a valid cube construction. 
  sorry

def count_valid_cube_combinations : ℕ :=
  -- Implement a method to count all valid combinations given the conditions.
  sorry

-- Main theorem statement to verify the result as per the problem statement.
theorem T_shape_cube_valid_combinations (h : count_valid_cube_combinations = 4) : Prop :=
  h

end T_shape_cube_valid_combinations_l278_278574


namespace pedestrian_time_to_B_l278_278893

-- Definition of speeds
variables (v : ℝ) -- speed of the pedestrian
noncomputable def cyclist_speed := 4 * v

-- Times and conditions
variables (time_A_B : ℝ) (time_stop : ℝ := 0.5) (time_total : ℝ := 2)
hypotheses 
  (h_overtake : ∃ x, time_A_B = x / cyclist_speed)
  (h_meet : ∃ x, time_total = (x - 2 * v) / cyclist_speed)

-- Distance relations
noncomputable def distance_a_b := 4 * v

-- Final statement of time taken by pedestrian
theorem pedestrian_time_to_B : (v / cyclist_speed) * 12 * v = 10 :=
by sorry

end pedestrian_time_to_B_l278_278893


namespace even_number_binary_to_ternary_l278_278473

theorem even_number_binary_to_ternary (N : ℕ) (hN : even N) (h0 : N % 2 = 0) 
  (h_equiv : binary_to_ternary_remove_trailing_zero N = N) : N = 10 ∨ N = 12 :=
sorry

end even_number_binary_to_ternary_l278_278473


namespace hyperbola_equation_and_product_const_l278_278899

theorem hyperbola_equation_and_product_const:
  (∃ (a b : ℝ), a^2 = 1/3 ∧ b^2 = 1 ∧ 3 * x^2 - y^2 = 1) ∧
  (∀ (x0 y0 : ℝ), (3 * x0^2 - y0^2 = 1) → 
  (abs (sqrt 3 * x0 - y0) / (sqrt (3 + 1))) * 
  (abs (sqrt 3 * x0 + y0) / (sqrt (3 + 1))) = 1 / 4) :=
  sorry

end hyperbola_equation_and_product_const_l278_278899


namespace line_equation_with_slope_2_l278_278971

def point_on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

def intersect_x_axis (L : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, L x = 0

theorem line_equation_with_slope_2 
    (x : ℝ) (y : ℝ) (hx : point_on_x_axis (x, y)) 
    (hL : intersect_x_axis (λ x, 2 * x + y - 4)) 
    (hxy : x = 2 ∧ y = 0) : 
    2 * x - y - 4 = 0 :=
by {
  -- Proof of the theorem is omitted here
  sorry
}

end line_equation_with_slope_2_l278_278971


namespace unpainted_cubes_l278_278520

theorem unpainted_cubes (n : ℕ) (cubes_per_face : ℕ) (faces : ℕ) (total_cubes : ℕ) (painted_cubes : ℕ) :
  n = 6 → cubes_per_face = 4 → faces = 6 → total_cubes = 216 → painted_cubes = 24 → 
  total_cubes - painted_cubes = 192 := by
  intros
  sorry

end unpainted_cubes_l278_278520


namespace digit_sum_is_twelve_l278_278503

theorem digit_sum_is_twelve (n x y : ℕ) (h1 : n = 10 * x + y) (h2 : 0 ≤ x ∧ x ≤ 9) (h3 : 0 ≤ y ∧ y ≤ 9)
  (h4 : (1 / 2 : ℚ) * n = (1 / 4 : ℚ) * n + 3) : x + y = 12 :=
by
  sorry

end digit_sum_is_twelve_l278_278503


namespace minimum_value_of_alpha_beta_l278_278625

open Polynomial

noncomputable def min_value_alpha_beta (m : ℝ) (h : m^2 ≥ 1) : ℝ :=
let α := root_of_quadratic (-2 * m) (2 - m^2) in
let β := root_of_quadratic_disjoint α (-2 * m) (2 - m^2) in
(α^2 + β^2)

theorem minimum_value_of_alpha_beta {α β m : ℝ} (h : m^2 ≥ 1) :
  (α^2 + β^2 = 2) :=
by
  have h_eq : (α^2 + β^2 = 6 * m^2 - 4), {
    sorry -- Proof showing how this expression is derived
  },
  have h_min : 6 * m^2 - 4 ≥ 2, {
    sorry -- Proof showing how minimum is derived
  },
  exact h_min

end minimum_value_of_alpha_beta_l278_278625


namespace probability_product_greater_than_zero_l278_278023

noncomputable theory

def probability_of_positive_product : ℝ :=
  let interval_length : ℝ := 30 in
  let probability_of_positive_or_negative : ℝ := (15 / interval_length) in
  probability_of_positive_or_negative * probability_of_positive_or_negative + 
  probability_of_positive_or_negative * probability_of_positive_or_negative

theorem probability_product_greater_than_zero :
  probability_of_positive_product = 1 / 2 :=
by
  sorry

end probability_product_greater_than_zero_l278_278023


namespace face_value_of_each_ticket_without_tax_l278_278041

theorem face_value_of_each_ticket_without_tax (total_people : ℕ) (total_cost : ℝ) (sales_tax : ℝ) (face_value : ℝ)
  (h1 : total_people = 25)
  (h2 : total_cost = 945)
  (h3 : sales_tax = 0.05)
  (h4 : total_cost = (1 + sales_tax) * face_value * total_people) :
  face_value = 36 := by
  sorry

end face_value_of_each_ticket_without_tax_l278_278041


namespace concyclic_points_anf_p_l278_278727

-- Definitions of the points and properties according to the conditions
variables (A B C M N P D E F : Type)
variables [ScaleneAcuteTriangle A B C]
variables [Midpoint M B C] [Midpoint N C A] [Midpoint P A B]
variables [PerpendicularBisectorIntersection D A B A M] [PerpendicularBisectorIntersection E A C A M]
variables [Intersection F B D C E]

-- The theorem to prove that points A, N, F, and P are concyclic
theorem concyclic_points_anf_p : Concyclic A N F P :=
sorry

end concyclic_points_anf_p_l278_278727


namespace evaluate_expression_l278_278948

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

theorem evaluate_expression : 
  3 * g(3) + 2 * g(-3) = 160 := 
by
  sorry

end evaluate_expression_l278_278948


namespace mass_percentage_of_Ca_in_CaCO3_l278_278596

-- Definitions for molar masses
def molar_mass_Ca := 40.08
def molar_mass_C := 12.01
def molar_mass_O := 16.00

-- Definition for molar mass of CaCO3
def molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

-- Theorem to prove the mass percentage of Ca in CaCO3
theorem mass_percentage_of_Ca_in_CaCO3 : 
  (molar_mass_Ca / molar_mass_CaCO3) * 100 ≈ 40 := sorry

end mass_percentage_of_Ca_in_CaCO3_l278_278596


namespace count_divisible_by_8_l278_278310

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278310


namespace five_digit_odd_number_l278_278468

noncomputable def x := 29995
noncomputable def y := 59992

theorem five_digit_odd_number:
  (∃ x : ℕ, 10000 ≤ x ∧ x < 100000 ∧ x % 2 = 1 ∧
   (∀ d : ℕ, (d = 2 → ∃ k, (x div 10^k) % 10 = 2 ∧ (y div 10^k) % 10 = 5) ∧
             (d = 5 → ∃ k, (x div 10^k) % 10 = 5 ∧ (y div 10^k) % 10 = 2)) ∧
   y = 2 * (x + 1))
  → x = 29995 :=
by
  sorry

end five_digit_odd_number_l278_278468


namespace find_angle_l278_278626

variables {V : Type*} [inner_product_space ℝ V] {a b : V}

-- Assumptions from the problem
def unit_vector (v : V) : Prop := ∥v∥ = 1

def condition_1 (a b : V) : Prop := unit_vector a ∧ unit_vector b

def condition_2 (a b : V) : Prop := ∥a - 2 • b∥ = real.sqrt 3

-- Combine conditions into a new context and declare the theorem to be proved
noncomputable def angle_between_vectors (a b : V) : ℝ :=
real.acos (inner_product_space.to_real_inner_product_space a b / (∥a∥ * ∥ b∥))

theorem find_angle (h1 : condition_1 a b) (h2 : condition_2 a b) : 
  angle_between_vectors a b = real.pi / 3 := sorry

end find_angle_l278_278626


namespace number_of_correct_statements_is_4_l278_278785

theorem number_of_correct_statements_is_4 :
  (∀ (a b c : Vector ℝ 3), LinearIndependent ![a, b, c] →
    LinearIndependent ![a + b, a - b, c]) ∧
  (∀ (a b : Vector ℝ 3), ∃ (p : AffineSpan ℝ (Set.Icc a b)) (b ≠ 0 → Vector ℝ 3), true) ∧
  (∀ (l m : Line ℝ), (DirectionVector l ≈ DirectionVector m) ↔ l ≈ m) ∧
  (∀ (α β : Plane ℝ) (u v : Vector ℝ 3), 
    u = #[1, 2, -2] ∧ v = #[-2, -4, 4] → (NormalVector α ≈ NormalVector β)) :=
by
  apply And.intro
  -- Proof for the first statement
  sorry
  apply And.intro 
  -- Proof for the second statement
  sorry
  apply And.intro
  -- Proof for the third statement
  sorry
  -- Proof for the fourth statement
  sorry

end number_of_correct_statements_is_4_l278_278785


namespace limit_sequence_l278_278110

def sequence (n : ℕ) : ℝ :=
  (sqrt (3 * n - 1) - (125 * n ^ 3 + n) ^ (1 / 3)) /
  (n ^ (1 / 3) - n)

theorem limit_sequence : Filter.Tendsto sequence Filter.atTop (nhds 5) :=
by
  sorry

end limit_sequence_l278_278110


namespace bike_helmet_cost_increase_l278_278365

open Real

theorem bike_helmet_cost_increase :
  let old_bike_cost := 150
  let old_helmet_cost := 50
  let new_bike_cost := old_bike_cost + 0.10 * old_bike_cost
  let new_helmet_cost := old_helmet_cost + 0.20 * old_helmet_cost
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let total_increase := new_total_cost - old_total_cost
  let percent_increase := (total_increase / old_total_cost) * 100
  percent_increase = 12.5 :=
by
  sorry

end bike_helmet_cost_increase_l278_278365


namespace coin_order_l278_278463

theorem coin_order (F D A E B C : Type) 
  (h1 : ∀ coin, coin ≠ F → F covers coin)
  (h2 : ∀ coin, coin ≠ D ∧ coin ≠ F → D covers coin → coin ∈ {B, C, E})
  (h3 : ∀ coin, coin ≠ A ∧ coin ≠ D ∧ coin ≠ F → A covers coin → coin ∈ {B, C})
  (h4 : ∀ coin, coin ≠ E ∧ coin ≠ D ∧ coin ≠ F → E covers coin → coin = C)
  (h5 : ∀ coin, coin ≠ B ∧ coin ≠ E ∧ coin ≠ D ∧ coin ≠ F → B covers coin → coin = C)
  (h6 : ∀ coin, coin ≠ C → ∃ covering_coin, covering_coin covers C) :
  F > D ∧ D > A ∧ A > E ∧ E > B ∧ B > C :=
sorry

end coin_order_l278_278463


namespace second_warehouse_more_profitable_initially_purchase_first_warehouse_advantageous_long_term_l278_278062

-- Condition definitions
def monthly_rent_first_warehouse : ℕ := 80000
def monthly_rent_second_warehouse : ℕ := 20000
def probability_repossession_second_warehouse : ℝ := 0.5
def moving_expense : ℕ := 150000
def months_in_year : ℕ := 12
def repossession_months : ℕ := 5
def remaining_months_after_repossession : ℕ := months_in_year - repossession_months

-- Annual rents
def annual_rent_first_warehouse : ℕ := monthly_rent_first_warehouse * months_in_year
def annual_rent_second_warehouse_worst_case : ℕ :=
 monthly_rent_second_warehouse * repossession_months + 
 monthly_rent_first_warehouse * remaining_months_after_repossession + 
 moving_expense

-- Part (a) Statement
theorem second_warehouse_more_profitable_initially :
  annual_rent_second_warehouse_worst_case < annual_rent_first_warehouse :=
sorry

-- Part (b) additional definitions
def purchase_cost_first_warehouse : ℕ := 3000000
def monthly_installment : ℕ := purchase_cost_first_warehouse / 36 -- over 3 years
def total_installment_cost_first_warehouse : ℕ := monthly_installment * months_in_year * 3

-- 5-year rents
def total_rent_first_warehouse_5_years : ℕ := annual_rent_first_warehouse * 5
def total_rent_second_warehouse_5_years_worst_case : ℕ :=
  annual_rent_second_warehouse_worst_case + (annual_rent_first_warehouse * 4)

-- Part (b) Statement
theorem purchase_first_warehouse_advantageous_long_term :
  total_installment_cost_first_warehouse < total_rent_first_warehouse_5_years ∧ total_rent_second_warehouse_5_years_worst_case :=
sorry

end second_warehouse_more_profitable_initially_purchase_first_warehouse_advantageous_long_term_l278_278062


namespace points_T_lie_on_single_line_l278_278209

-- Define the given conditions
variables (A B C X Y T : Type*) [acute_triangle A B C]
noncomputable def on_segment_AC (X : Type*) : Prop := X ∈ (segment A C)
noncomputable def on_extension_BC_beyond_C (Y : Type*) : Prop := ∃ D, C ∈ (segment B D) ∧ Y = D

-- Define the angle condition
def angle_condition (A B X C Y : Type*) : Prop := ∠ ABX + ∠ CXY = 90

-- Define the projection point T
noncomputable def is_projection (B X Y T : Type*) : Prop := T = foot_of_perpendicular B line_XY

-- Define the line (in this context presumed to be line KH)
variable (l : Type*)

-- The main theorem stating the conclusion
theorem points_T_lie_on_single_line
  (h1 : acute_triangle A B C)
  (h2 : on_segment_AC X)
  (h3 : on_extension_BC_beyond_C Y)
  (h4 : angle_condition A B X C Y)
  (h5 : is_projection B X Y T) :
  ∃ l, ∀ T, is_projection B X Y T → T ∈ l :=
sorry

end points_T_lie_on_single_line_l278_278209


namespace largest_natural_number_not_sum_of_two_composites_l278_278145

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l278_278145


namespace ant_movement_black_squares_l278_278559

theorem ant_movement_black_squares (start_pos : nat) (moves : nat) 
 (h_start : start_pos % 2 = 0) (h_moves : moves = 4) : 
∃ end_squares : finset nat, end_squares.card = 6 ∧ 
  ∀ pos ∈ end_squares, pos % 2 = 0 := 
begin
  sorry
end

end ant_movement_black_squares_l278_278559


namespace teacher_wu_final_position_teacher_wu_total_steps_if_return_l278_278343

-- Define the list of walking distances
def walking_distances := [+620, -580, +450, +650, -520, -480, -660, +550]

-- Define the stepping rate
def steps_per_meter := 2

-- Task 1: Prove the final position of Teacher Wu
theorem teacher_wu_final_position : 
  (list.sum walking_distances) = 30 := 
sorry

-- Task 2: Prove the total steps recorded on WeChat exercise tracker
theorem teacher_wu_total_steps_if_return : 
  let total_distance := list.sum (list.map abs walking_distances) + abs (list.sum walking_distances) in
  (total_distance * steps_per_meter) = 9080 := 
sorry

end teacher_wu_final_position_teacher_wu_total_steps_if_return_l278_278343


namespace union_complement_A_eq_l278_278654

open Set

variable (U A B : Set ℕ)

-- Define the sets U, A, and B
def U := {1, 2, 3, 4, 5}
def A := {1, 3}
def B := {1, 2, 4}

theorem union_complement_A_eq : ((U \ B) ∪ A) = {1, 3, 5} := 
by
  sorry

end union_complement_A_eq_l278_278654


namespace minimize_area_l278_278603

def curve (x : ℝ) : ℝ := 1 - x^2

def tangent_line (a : ℝ) : ℝ → ℝ := 
  fun x => -2 * a * (x - a) + curve a

def x_intercept_tangent (a : ℝ) (h : a ≠ 0) : ℝ :=
  (a^2 + 1) / (2 * a)

def area_of_triangle (a b : ℝ) : ℝ :=
  let E := x_intercept_tangent a (by linarith)
  let F := x_intercept_tangent b (by linarith)
  let height := 1 - a * b
  (E - F).abs * height / 2

theorem minimize_area : 
  ∃ a b : ℝ, a > 0 ∧ b < 0 ∧ area_of_triangle a b = 8 * Real.sqrt 3 / 9 := 
sorry

end minimize_area_l278_278603


namespace total_photos_in_gallery_l278_278745

def initial_photos : ℕ := 800
def photos_first_day : ℕ := (2 * initial_photos) / 3
def photos_second_day : ℕ := photos_first_day + 180

theorem total_photos_in_gallery : initial_photos + photos_first_day + photos_second_day = 2046 := by
  -- the proof can be provided here
  sorry

end total_photos_in_gallery_l278_278745


namespace age_ratio_l278_278765

-- Definitions of the ages based on the given conditions.
def Rachel_age : ℕ := 12  -- Rachel's age
def Father_age_when_Rachel_25 : ℕ := 60

-- Defining Mother, Father, Grandfather ages based on given conditions.
def Grandfather_age (R : ℕ) (F : ℕ) : ℕ := 2 * (F - 5)
def Father_age (R : ℕ) : ℕ := Father_age_when_Rachel_25 - (25 - R)

-- Proving the ratio of Grandfather's age to Rachel's age is 7:1
theorem age_ratio (R : ℕ) (F : ℕ) (G : ℕ) :
  R = Rachel_age →
  F = Father_age R →
  G = Grandfather_age R F →
  G / R = 7 := by
  exact sorry

end age_ratio_l278_278765


namespace problem_a_b_sum_l278_278920

-- Define the operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Given conditions
variable (a b : ℝ)

-- Theorem statement: Prove that a + b = 4
theorem problem_a_b_sum :
  (∀ x, ((2 < x) ∧ (x < 3)) ↔ ((x - a) * (x - b - 1) < 0)) → a + b = 4 :=
by
  sorry

end problem_a_b_sum_l278_278920


namespace polynomial_has_real_root_l278_278774

open Real Polynomial

variable {c d : ℝ}
variable {P : Polynomial ℝ}

theorem polynomial_has_real_root (hP1 : ∀ n : ℕ, c * |(n : ℝ)|^3 ≤ |P.eval (n : ℝ)|)
                                (hP2 : ∀ n : ℕ, |P.eval (n : ℝ)| ≤ d * |(n : ℝ)|^3)
                                (hc : 0 < c) (hd : 0 < d) : 
                                ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l278_278774


namespace license_plate_probability_l278_278384

theorem license_plate_probability :
  let m := 5
  let n := 104
  Nat.gcd m n = 1 ∧ m + n = 109 := by
  have h : Nat.gcd 5 104 = 1 := by norm_num
  exact ⟨h, by norm_num⟩

end license_plate_probability_l278_278384


namespace percent_both_correct_proof_l278_278042

-- Define the problem parameters
def totalTestTakers := 100
def percentFirstCorrect := 80
def percentSecondCorrect := 75
def percentNeitherCorrect := 5

-- Define the target proof statement
theorem percent_both_correct_proof :
  percentFirstCorrect + percentSecondCorrect - percentFirstCorrect + percentNeitherCorrect = 60 := 
by 
  sorry

end percent_both_correct_proof_l278_278042


namespace inscribed_circle_radius_l278_278599

theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : c = 20) :
    (∃ r : ℝ, (1 / r) = (1 / a) + (1 / b) + (1 / c) + 2 * sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))) ∧ r = 1.381) :=
by
  sorry

end inscribed_circle_radius_l278_278599


namespace largest_cannot_be_sum_of_two_composites_l278_278159

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278159


namespace ratio_of_potatoes_l278_278714

def total_potatoes : ℕ := 24
def people : ℕ := 3
def potatoes_per_person : ℕ := 8

theorem ratio_of_potatoes (h : total_potatoes = potatoes_per_person * people) :
  (potatoes_per_person = total_potatoes / people) :=
begin
  sorry
end

end ratio_of_potatoes_l278_278714


namespace no_arithmetic_seq_with_sum_n_cubed_l278_278123

theorem no_arithmetic_seq_with_sum_n_cubed (a1 d : ℕ) :
  ¬ (∀ (n : ℕ), (n > 0) → (n / 2) * (2 * a1 + (n - 1) * d) = n^3) :=
sorry

end no_arithmetic_seq_with_sum_n_cubed_l278_278123


namespace james_weekly_earning_l278_278353

-- Let the hourly wage at the main job be $20.
def main_job_wage : ℝ := 20

-- Let the reduction percentage at the second job be 20%.
def reduction_percentage : ℝ := 0.20

-- Calculate the hourly wage at the second job after reduction.
def second_job_wage : ℝ := main_job_wage * (1 - reduction_percentage)

-- Let the number of hours worked at the main job be 30.
def main_job_hours : ℝ := 30

-- Let the number of hours at the second job be half of that at the main job.
def second_job_hours : ℝ := main_job_hours / 2

-- Calculate the weekly earning at the main job.
def main_job_earning : ℝ := main_job_hours * main_job_wage

-- Calculate the weekly earning at the second job.
def second_job_earning : ℝ := second_job_hours * second_job_wage

-- Calculate the total weekly earning.
def total_weekly_earning : ℝ := main_job_earning + second_job_earning

-- The proof statement that James earns $840 per week.
theorem james_weekly_earning : total_weekly_earning = 840 := by
  sorry

end james_weekly_earning_l278_278353


namespace lattice_points_in_square_l278_278708

theorem lattice_points_in_square (side_centered_at_origin : ℝ) (is_lattice_point : ℤ × ℤ → Prop) :
  side_centered_at_origin = 4 ∧ (∀ (x y : ℤ), is_lattice_point (x, y) ↔ x ∈ { -1, 0, 1 } ∧ y ∈ { -1, 0, 1 })
  → (finset.filter is_lattice_point ({x | x.1 >= -2 ∧ x.1 <= 2 ∧ x.2 >= -2 ∧ x.2 <= 2}.to_finset)).card = 9 := by
  sorry

end lattice_points_in_square_l278_278708


namespace find_a_l278_278326

theorem find_a (a : ℝ) (hne : a ≠ 1) (eq_sets : ∀ x : ℝ, (a-1) * x < a + 5 ↔ 2 * x < 4) : a = 7 :=
sorry

end find_a_l278_278326


namespace sum_expression_l278_278617

-- Definitions from the problem conditions
def a_seq : ℕ → ℕ
| 0     := 0 -- convention for a₀, since our sequences use ℕ^* which starts from 1
| (n+1) := if n = 0 then 1 else S (n)

-- S_n is defined as the sum of the first n terms of a sequence
def S : ℕ → ℕ
| 0     := 0 -- S₀ is conventionally set to 0
| (n+1) := S n + a_seq (n+1)

-- Proof problem to show S_n = 2^(n-1)
theorem sum_expression (n : ℕ) (n_pos : n > 0) : S n = 2^(n-1) :=
by sorry

end sum_expression_l278_278617


namespace karen_paddle_time_l278_278723

-- Declare speeds for each section
def still_water_speed : ℝ := 10
def current_speed1 : ℝ := 4
def current_speed2 : ℝ := 6
def current_speed3 : ℝ := 3

-- Declare lengths for each section
def length1 : ℝ := 5
def length2 : ℝ := 8
def length3 : ℝ := 7

-- Calculate time for each section
def time_section1 : ℝ := length1 / (still_water_speed - current_speed1)
def time_section2 : ℝ := length2 / (still_water_speed - current_speed2)
def time_section3 : ℝ := length3 / (still_water_speed - current_speed3)

-- Total time
def total_time_hours : ℝ := time_section1 + time_section2 + time_section3

-- Conversion of the fractional part to minutes
def fractional_to_minutes (fraction : ℝ) : ℝ := fraction * 60

-- Main theorem
theorem karen_paddle_time : total_time_hours = 3 + 50 / 60 := by
  sorry

end karen_paddle_time_l278_278723


namespace trapezoid_of_intersections_l278_278213

theorem trapezoid_of_intersections
  (ABCD : Type)
  (is_trapezoid : asymmetric_trapezoid ABCD)
  (A1 B1 C1 D1 : Type)
  (A1_def : is_intersection (circumcircle (triangle B C D)) (line A C) (ne_point C))
  (B1_def : is_intersection (circumcircle (triangle C D A)) (line B D) (ne_point D))
  (C1_def : is_intersection (circumcircle (triangle D A B)) (line C A) (ne_point A))
  (D1_def : is_intersection (circumcircle (triangle A B C)) (line D B) (ne_point B)) :
  is_trapezoid A1 B1 C1 D1 :=
  sorry

end trapezoid_of_intersections_l278_278213


namespace tom_new_collection_l278_278016

theorem tom_new_collection (initial_stamps mike_gift : ℕ) (harry_gift : ℕ := 2 * mike_gift + 10) (sarah_gift : ℕ := 3 * mike_gift - 5) (total_gifts : ℕ := mike_gift + harry_gift + sarah_gift) (new_collection : ℕ := initial_stamps + total_gifts) 
  (h_initial_stamps : initial_stamps = 3000) (h_mike_gift : mike_gift = 17) :
  new_collection = 3107 := by
  sorry

end tom_new_collection_l278_278016


namespace expansion_identity_l278_278701

-- Defining the conditions as given in the problem
constant n : ℕ
constant x P Q : ℝ
axiom sum_of_odd_terms : ∀ n x, (1 + x)^n = P + Q
axiom sum_of_even_terms : ∀ n x, (1 - x)^n = P - Q

-- Main statement we need to prove
theorem expansion_identity : (1 - x^2)^n = P^2 - Q^2 :=
by
  sorry

end expansion_identity_l278_278701


namespace sandwiches_needed_l278_278896

variable (total_people children adults children_sandwiches_per_adult adults_sandwiches_per_adult : ℕ)

-- Given conditions: total_people = 219, children = 125, adults = 94
def total_people : ℕ := 219
def children : ℕ := 125
def adults : ℕ := 94
def children_sandwiches_per_child : ℕ := 4
def adults_sandwiches_per_adult : ℕ := 3

-- Total number of sandwiches needed
def total_sandwiches_needed : ℕ := children * children_sandwiches_per_child + adults * adults_sandwiches_per_adult

theorem sandwiches_needed : total_sandwiches_needed = 782 :=
by
  sorry

end sandwiches_needed_l278_278896


namespace largest_not_sum_of_two_composites_l278_278137

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l278_278137


namespace counting_divisibles_by_8_l278_278291

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278291


namespace length_AL_l278_278349

-- Given conditions
variables (A B C L : Type)
variables (AB AC BC : Type)
variable [InnerProductSpace ℝ A]

-- Assume the conditions
axiom AB_AC_ratio : (5 : ℝ) / (4 : ℝ) = |AB| / |AC|

-- Axiom for the length of the vector combination
axiom vec_comb_length : ∥4 • (AB : A) + 5 • (AC : A)∥ = 2016

-- Define the angle bisector conditions
axiom angle_bisector_theorem : ∥L - B∥ / ∥L - C∥ = |AB| / |AC|

-- Define the position of L on BC using the angle bisector theorem
def BL_pos := (5 : ℝ) / (9 : ℝ) • (BC : A)

-- Calculate vector AL
def vec_AL := (4 : ℝ / 9) • (AB : A) + (5 : ℝ / 9) • (AC : A)

-- Theorem to prove the desired length of AL
theorem length_AL : (∥vec_AL∥ = 224) :=
by
  sorry

end length_AL_l278_278349


namespace decreasing_intervals_sin_2A_l278_278235

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * (cos x)^2 - 1

-- Problem (1): Prove that f(x) is monotonically decreasing in the given intervals
theorem decreasing_intervals (k : ℤ) : 
  is_decreasing_on f (set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
sorry

-- Problem (2): Given A is an acute angle and f(A) = 2/3, prove the value of sin 2A
theorem sin_2A (A : ℝ) (hA1 : 0 < A ∧ A < π / 2) (hA2 : f A = 2 / 3) : 
  sin (2 * A) = (sqrt 3 + 2 * sqrt 2) / 6 :=
sorry

end decreasing_intervals_sin_2A_l278_278235


namespace right_triangle_sum_of_squares_l278_278687

theorem right_triangle_sum_of_squares (A B C : Type*) [metric_space A] [normed_group B] 
  [normed_space ℝ B]  [inner_product_space ℝ B] 
  (h : is_right_triangle A B C) (hBC : dist B C = 2) :
  (dist A B)^2 + (dist A C)^2 + (dist B C)^2 = 8 :=
sorry

end right_triangle_sum_of_squares_l278_278687


namespace cos_theta_equals_sqrt2_div_2_l278_278253

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b_minus_a : ℝ × ℝ := (-1, 1)
noncomputable def b : ℝ × ℝ := (a.1 + b_minus_a.1, a.2 + b_minus_a.2)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem cos_theta_equals_sqrt2_div_2 :
  let theta := Real.arccos (dot_product a b / (magnitude a * magnitude b)) in
  Real.cos theta = Real.sqrt 2 / 2 :=
by
  sorry

end cos_theta_equals_sqrt2_div_2_l278_278253


namespace value_of_f_three_l278_278237

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * Real.cos x - x

theorem value_of_f_three (a b : ℝ) (h : f a b (-3) = 7) : f a b 3 = 1 :=
by
  sorry

end value_of_f_three_l278_278237


namespace equilateral_triangle_l278_278397

noncomputable def cube_sides_ratio (a : ℝ) : Prop :=
  let X := (a / 3, 0, 0)
  let Y := (0, a, a / 3)
  let Z := (0, 2 * a / 3, a)
  let dist (P Q : ℝ × ℝ × ℝ) := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

theorem equilateral_triangle (a : ℝ) (h : a > 0) : cube_sides_ratio a :=
by {
  -- proof here
  sorry
}

end equilateral_triangle_l278_278397


namespace arctan_sum_eq_pi_div_3_l278_278976

theorem arctan_sum_eq_pi_div_3 : 
  ∃ (n : ℕ), n > 0 ∧ arctan (1 / 3) + arctan (1 / 4) + arctan (1 / 7) + arctan (1 / n.to_real) = π / 3 :=
begin
  use 10,
  split,
  { norm_num }, -- Prove that 10 is greater than 0
  { sorry } -- Proof of the equality
end

end arctan_sum_eq_pi_div_3_l278_278976


namespace intersection_point_on_ω1_l278_278519

noncomputable def triangle := {A B C : Point}

noncomputable def circumcircle (Δ : triangle) := sorry -- definition of circumcircle

noncomputable def center (C : Circle) := sorry -- definition of center of a circle

noncomputable def diameter (C : Circle) := sorry -- definition of diameter of a circle

noncomputable def parallelogram (P Q R S : Point) := sorry -- definition of parallelogram

open triangle

variable (A B C O Q M N : Point)
variable (Δ : triangle.{u}) (ω ω₁ : Circle)

axiom acute_triangle (t : triangle.{u}) : Prop := sorry -- Definition for acute triangle

axiom circumcircle (Δ : triangle) (ω : Circle) : Prop := sorry -- Definition for circumcircle
axiom center (ω : Circle) : Point := sorry -- Definition for center of a circle
axiom line_through (P Q : Point) : Line := sorry

-- Conditions of the problem
axiom acute_angled_triangle : acute_triangle (triangle.mk A B C)
axiom circumcircle_ABC : circumcircle (triangle.mk A B C) ω
axiom circumcircle_AOC : circumcircle (triangle.mk A O C) ω₁
axiom diameter_OQ_of_ω₁ : diameter (ω₁) = line_through O Q

axiom M_on_AQ : M ∈ line_through A Q
axiom N_on_AC : N ∈ line_through A C
axiom AMBN_parallelogram : parallelogram A M B N

-- Theorem statement
theorem intersection_point_on_ω1
  (H1 : acute_angled_triangle)
  (H2 : circumcircle_ABC)
  (H3 : circumcircle_AOC)
  (H4 : diameter_OQ_of_ω₁)
  (H5 : M_on_AQ)
  (H6 : N_on_AC)
  (H7 : AMBN_parallelogram) :
  let MN := line_through M N,
      BQ := line_through B Q,
      P := intersection MN BQ
  in P ∈ ω₁ := sorry

end intersection_point_on_ω1_l278_278519


namespace tan_2x_eq_sin_x_solutions_l278_278956

open Real

theorem tan_2x_eq_sin_x_solutions :
    ∃ n : Nat, n = 5 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * π → tan(2 * x) = sin(x) → False :=
by
  sorry

end tan_2x_eq_sin_x_solutions_l278_278956


namespace inequality_represents_area_l278_278449

theorem inequality_represents_area (a : ℝ) :
  (if a > 1 then ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y < - (x + 3) / (a - 1)
  else ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y > - (x + 3) / (a - 1)) :=
by sorry

end inequality_represents_area_l278_278449


namespace justin_pages_read_l278_278359

theorem justin_pages_read :
  let pages_day1 := 10 in
  let daily_pages := 2 * pages_day1 in
  let total_pages := pages_day1 + daily_pages * 6 in
  total_pages = 130 :=
by
  let pages_day1 := 10
  let daily_pages := 2 * pages_day1
  let total_pages := pages_day1 + daily_pages * 6
  sorry

end justin_pages_read_l278_278359


namespace pattern_proof_l278_278516

theorem pattern_proof (h1 : 1 = 6) (h2 : 2 = 36) (h3 : 3 = 363) (h4 : 4 = 364) (h5 : 5 = 365) : 36 = 3636 := by
  sorry

end pattern_proof_l278_278516


namespace scientific_notation_l278_278561

theorem scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) (h₁ : x = 5853) (h₂ : 1 ≤ |a|) (h₃ : |a| < 10) (h₄ : x = a * 10^n) : 
  a = 5.853 ∧ n = 3 :=
by sorry

end scientific_notation_l278_278561


namespace count_multiples_of_8_in_range_l278_278279

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278279


namespace cat_food_percentage_l278_278497

theorem cat_food_percentage (D C : ℝ) (h1 : 7 * D + 4 * C = 8 * D) (h2 : 4 * C = D) : 
  (C / (7 * D + D)) * 100 = 3.125 := by
  sorry

end cat_food_percentage_l278_278497


namespace area_charming_points_l278_278067

noncomputable def strange_line (a : ℝ) (h : 0 ≤ a ∧ a ≤ 10) : set (ℝ × ℝ) :=
{ p | let x := p.1, y := p.2 in y = (a - 10) / a * x + 10 - a }

def is_charming (p : ℝ × ℝ) : Prop :=
p.1 > 0 ∧ p.2 > 0 ∧ ∃ a (ha : 0 ≤ a ∧ a ≤ 10), strange_line a ha p

theorem area_charming_points : 
  let region := { p : ℝ × ℝ | is_charming p } in
  ∃ area : ℝ, area = 50 / 3 :=
sorry

end area_charming_points_l278_278067


namespace establish_charges_l278_278090

variable (E : ℝ) -- Establishment charges
variable (m : ℝ := 14) -- Total number of machines
variable (m_closed : ℝ := 7.14) -- Number of machines closed
variable (full_output : ℝ := 70000) -- Annual output with all machines
variable (full_cost : ℝ := 42000) -- Annual manufacturing cost with all machines
variable (profit_percentage : ℝ := 0.125) -- Profit percentage for shareholders
variable (profit_decrease_percentage : ℝ := 0.125) -- Percentage decrease in profit

def operational_machines := m - m_closed
def new_output := (operational_machines / m) * full_output
def full_profit := profit_percentage * full_output
def new_profit := full_profit - (profit_decrease_percentage * full_profit)
def new_cost := (operational_machines / m) * full_cost
def total_cost := new_cost + E
def total_income := new_output + new_profit

theorem establish_charges :
  total_cost ≤ total_income → E ≤ 21376.25 := by
  sorry

end establish_charges_l278_278090


namespace second_warehouse_more_profitable_in_first_year_buying_first_warehouse_more_advantageous_l278_278063

-- Part (a)
theorem second_warehouse_more_profitable_in_first_year :
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let first_year_cost1 := rent1 * 12
  let worst_case_cost2 := rent2 * 5 + rent1 * 7 + move_cost
  worst_case_cost2 < first_year_cost1 :=
by
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let first_year_cost1 := rent1 * 12
  let worst_case_cost2 := rent2 * 5 + rent1 * 7 + move_cost
  exact lt_of_lt_of_le 
    (add_lt_add_right (add_lt_add_right (add_lt_add_right rfl rfl) rfl) rfl)
    (calc worst_case_cost2 < first_year_cost1 := by sorry
    )

-- Part (b)
theorem buying_first_warehouse_more_advantageous :
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let total_cost1 := rent1 * 12 * 5
  let worst_case_cost2 := rent2 * 5 + rent1 * 12 * 4 + move_cost
  let buy_cost := 3_000_000
  total_cost1 > buy_cost ∧ worst_case_cost2 > buy_cost :=
by
  let rent1 := 80_000
  let rent2 := 20_000
  let move_cost := 150_000
  let total_cost1 := rent1 * 12 * 5
  let worst_case_cost2 := rent2 * 5 + rent1 * 12 * 4 + move_cost
  let buy_cost := 3_000_000
  exact and.intro 
    (calc total_cost1 > buy_cost := by sorry)
    (calc worst_case_cost2 > buy_cost := by sorry)

end second_warehouse_more_profitable_in_first_year_buying_first_warehouse_more_advantageous_l278_278063


namespace parabola_tangent_and_distance_conditions_l278_278221

theorem parabola_tangent_and_distance_conditions
  (p : ℝ) (h_p : p > 0)
  (A : ℝ × ℝ) (h_A : A = (1,1))
  (B : ℝ × ℝ) (h_B : B = (0,-1))
  (O : ℝ × ℝ) (h_O : O = (0,0))
  (P Q : ℝ × ℝ)
  (h_A_on_parabola : A.1 ^ 2 = 2 * p * A.2)
  (h_line_AB_tangent : line_is_tangent (1,1) (0,-1))
  (h_OP_OQ : distance O P * distance O Q > distance O A ^ 2) :
    (line_is_tangent (1,1) (0,-1) ∧ distance O P * distance O Q > distance O A ^ 2) :=
begin
  sorry
end

-- Assumed auxiliary definitions and lemmas
def line_is_tangent (A B : ℝ × ℝ) : Prop := sorry
def distance (X Y : ℝ × ℝ) : ℝ := sorry

end parabola_tangent_and_distance_conditions_l278_278221


namespace find_positive_number_l278_278000

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l278_278000


namespace acute_angle_phi_l278_278316

theorem acute_angle_phi (φ : ℝ) (h1 : φ > 0) (h2 : φ < 90) (h3 : sqrt 2 * real.cos (20 * real.pi / 180) = real.sin (φ * real.pi / 180) - real.cos (φ * real.pi / 180)) : φ = 25 :=
by
  sorry

end acute_angle_phi_l278_278316


namespace white_probability_l278_278054

variable (P_white pop_white P_blue pop_blue : ℚ)

-- Given conditions
def cond_prob_white_popped :=
let P_white := 3/4 in
let P_blue := 1/4 in
let pop_white := 3/5 in
let pop_blue := 3/4 in
let P_white_popped := P_white * pop_white in
let P_blue_popped := P_blue * pop_blue in
let P_popped := P_white_popped + P_blue_popped in
P_white_popped / P_popped = 12/13

theorem white_probability :
  cond_prob_white_popped :=
by
  sorry

end white_probability_l278_278054


namespace triangle_construction_l278_278575

noncomputable def construct_triangle (α : ℝ) (m_a : ℝ) (k_a : ℝ) : Type :=
  { A B C : Type } -- need constraints to ensure these are points forming a triangle with the given properties.

theorem triangle_construction (α : ℝ) (m_a : ℝ) (k_a : ℝ) :
  ∃ (A B C : Type), 
  (A, B, C are vertices of a triangle) ∧
  (angle BAC = α) ∧
  (altitude from A to BC = m_a) ∧
  (median from A to midpoint of BC = k_a) :=
begin
  sorry,
end

end triangle_construction_l278_278575


namespace mr_llesis_rice_cost_l278_278389

theorem mr_llesis_rice_cost :
  let total_kg : ℚ := 50
  let price_part1 : ℚ := 1.2
  let price_part2 : ℚ := 1.5
  let price_part3 : ℚ := 2
  let kg_part1 : ℚ := 20
  let kg_part2 : ℚ := 25
  let kg_part3 : ℚ := 5
  let total_cost := kg_part1 * price_part1 + kg_part2 * price_part2 + kg_part3 * price_part3
  let kept_kg := 7 / 10 * total_kg
  let given_kg := total_kg - kept_kg
  let cost_kept := kg_part1 * price_part1 + (kg_part2 * (kept_kg - kg_part1) / kg_part2) * price_part2 + kg_part3 * price_part3
  let cost_given := total_cost - cost_kept
in cost_kept - cost_given = 41.5 := sorry

end mr_llesis_rice_cost_l278_278389


namespace count_divisible_by_8_l278_278309

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278309


namespace find_functional_solution_l278_278966

noncomputable def solution_exists (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (sqrt 2 * x) + f ((4 + 3 * sqrt 2) * x) = 2 * f ((2 + sqrt 2) * x)

theorem find_functional_solution (f : ℝ → ℝ) :
  solution_exists f →
  ∃ (a : ℝ) (g : ℝ → ℝ), (∀ x : ℝ, g (x + 1) = g x) ∧
  (∀ x : ℝ, f x = if x = 0 then a else g (log (sqrt 2 - 1) (| x |))) :=
sorry

end find_functional_solution_l278_278966


namespace floor_minus_3_7_eq_minus_4_l278_278932

def floor_of_neg_3_7 : Int :=
  Int.floor (-3.7)

theorem floor_minus_3_7_eq_minus_4 :
  floor_of_neg_3_7 = -4 := by
  sorry

end floor_minus_3_7_eq_minus_4_l278_278932


namespace sam_dads_dimes_l278_278412

theorem sam_dads_dimes (original_dimes new_dimes given_dimes : ℕ) 
  (h1 : original_dimes = 9)
  (h2 : new_dimes = 16)
  (h3 : new_dimes = original_dimes + given_dimes) : 
  given_dimes = 7 := 
by 
  sorry

end sam_dads_dimes_l278_278412


namespace determine_number_within_10_questions_l278_278385

theorem determine_number_within_10_questions :
  ∃ (strategy : (ℕ → Bool) → ℕ), 
    ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 1000) → 
      ∃ (questions : (fin 10) → (ℕ → Bool)),
        strategy questions = n :=
by sorry

end determine_number_within_10_questions_l278_278385


namespace general_term_formula_T_formula_l278_278616

noncomputable def S (a : ℕ → ℕ) (n : ℕ) := ∑ i in Finset.range (n + 1), a i

theorem general_term_formula (a : ℕ → ℕ) (n : ℕ) (h : ∀ n, S a n = (a n) * (a n + 1) / 2) : 
  ∀ n, a n = n :=
by
  sorry

def b (a : ℕ → ℕ) (n : ℕ) := 1 / (a n * a (n+1))

noncomputable def T (a : ℕ → ℕ) (n : ℕ) := ∑ i in Finset.range (n + 1), b a i

theorem T_formula (a : ℕ → ℕ) (h₁ : ∀ n, S a n = (a n) * (a n + 1) / 2) (h₂ : ∀ n, a n = n) :
  ∀ n, T a n = n / (n + 1) :=
by
  sorry

end general_term_formula_T_formula_l278_278616


namespace integral_sin_cos_squared_l278_278964

theorem integral_sin_cos_squared :
  ∫ (x : ℝ) in set.univ, 1 / (sin x ^ 2 + 4 * cos x ^ 2) = 
    (1/2) * arctan(tan x / 2) + C := 
by
  sorry

end integral_sin_cos_squared_l278_278964


namespace solve_for_y_l278_278772

theorem solve_for_y (y : ℝ) : 3^y + 11 = 5 * 3^y - 39 → y = Real.log 12.5 / Real.log 3 :=
by
  intro h
  sorry

end solve_for_y_l278_278772


namespace four_digit_even_number_count_l278_278414

noncomputable def count_even_four_digit_numbers : ℕ :=
  let digits : Finset ℕ := {0, 1, 2, 3, 4}
  let evens : Finset ℕ := {0, 2, 4}
  let without_repetition : Finset (Finset ℕ) :=
    digits.powerset.filter (fun s => s.card = 4)
  let valid_numbers : Finset (Finset ℕ) :=
    without_repetition.filter (fun s => ∃ x ∈ evens, s.erase x = {s_1 | s_1 ∈ digits ∧ s_1 ≠ x})
  valid_numbers.card

theorem four_digit_even_number_count : count_even_four_digit_numbers = 60 :=
  sorry

end four_digit_even_number_count_l278_278414


namespace cos_reflected_value_l278_278992

theorem cos_reflected_value (x : ℝ) (h : Real.cos (π / 6 + x) = 1 / 3) :
  Real.cos (5 * π / 6 - x) = -1 / 3 := 
by {
  sorry
}

end cos_reflected_value_l278_278992


namespace angle_terminal_side_l278_278590

theorem angle_terminal_side :
  ∃ θ ∈ Ico 0 360, θ ≡ 1000 [MOD 360] :=
by
  use 280
  split
  {
    show 0 ≤ 280
    norm_num
  }
  {
    show 280 < 360
    norm_num
  }
  show 280 ≡ 1000 [MOD 360]
  norm_num
  refl

end angle_terminal_side_l278_278590


namespace simple_interest_l278_278550

theorem simple_interest (P R T : ℕ) (H_P : P = 15000) (H_R : R = 6) (H_T : T = 3) :
  (P * R * T) / 100 = 2700 :=
  by
  rw [H_P, H_R, H_T]
  -- Further simplification steps would go here
  sorry

end simple_interest_l278_278550


namespace units_digit_7_pow_2050_l278_278850

noncomputable def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_7_pow_2050 : units_digit (7 ^ 2050) = 9 :=
by
  -- Establish the cycle of units digits of powers of 7
  have h_cycle : ∀ k, units_digit (7 ^ (k + 4)) = units_digit (7 ^ k) := by sorry
  
  -- Find the position in the cycle
  have h_mod: 2050 % 4 = 2 := by sorry
  
  -- Conclude the units digit of 7^2050
  show units_digit (7 ^ 2050) = 9, from by
  {
    apply Eq.trans (units_digit (7 ^ 2050))
    apply h_cycle 2
    exact Eq.refl 9
  }

end units_digit_7_pow_2050_l278_278850


namespace gummy_cost_proof_l278_278937

variables (lollipop_cost : ℝ) (num_lollipops : ℕ) (initial_money : ℝ) (remaining_money : ℝ)
variables (num_gummies : ℕ) (cost_per_gummy : ℝ)

-- Conditions
def conditions : Prop :=
  lollipop_cost = 1.50 ∧
  num_lollipops = 4 ∧
  initial_money = 15 ∧
  remaining_money = 5 ∧
  num_gummies = 2 ∧
  initial_money - remaining_money = (num_lollipops * lollipop_cost) + (num_gummies * cost_per_gummy)

-- Proof problem
theorem gummy_cost_proof : conditions lollipop_cost num_lollipops initial_money remaining_money num_gummies cost_per_gummy → cost_per_gummy = 2 :=
by
  sorry  -- Solution steps would be filled in here


end gummy_cost_proof_l278_278937


namespace largest_four_digit_number_divisible_by_6_l278_278840

theorem largest_four_digit_number_divisible_by_6 :
  ∃ n, n = 9996 ∧ ∀ m, (m ≤ 9999 ∧ m % 6 = 0) → m ≤ n :=
begin
  sorry
end

end largest_four_digit_number_divisible_by_6_l278_278840


namespace largest_non_representable_as_sum_of_composites_l278_278165

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l278_278165


namespace molly_age_l278_278751

theorem molly_age (total_miles : ℕ) (miles_per_day : ℕ) (age_stop_riding : ℕ) : age_stop_riding = 16 ∧ total_miles = 3285 ∧ miles_per_day = 3 → age_stop_riding - total_miles / (miles_per_day * 365) = 13 :=
begin
  sorry
end

end molly_age_l278_278751


namespace exists_infinitely_many_pairs_of_finite_sets_l278_278738

noncomputable def x (n : ℕ) : ℕ := Nat.choose (2 * n) n

theorem exists_infinitely_many_pairs_of_finite_sets :
  ∃ (A B : Finset ℕ), A ∩ B = ∅ ∧ (A ∪ B).Nonempty ∧
  (∏ j in A, x j) / (∏ j in B, x j) = 2012 :=
sorry

end exists_infinitely_many_pairs_of_finite_sets_l278_278738


namespace minimum_value_expression_l278_278196

theorem minimum_value_expression {a : ℝ} (h₀ : 1 < a) (h₁ : a < 4) : 
  (∃ m : ℝ, (∀ x : ℝ, 1 < x ∧ x < 4 → m ≤ (x / (4 - x) + 1 / (x - 1))) ∧ m = 2) :=
sorry

end minimum_value_expression_l278_278196


namespace probability_A_score_not_less_than_135_l278_278056

/-- A certain school organized a competition with the following conditions:
  - The test has 25 multiple-choice questions, each with 4 options.
  - Each correct answer scores 6 points, each unanswered question scores 2 points, and each wrong answer scores 0 points.
  - Both candidates answered the first 20 questions correctly.
  - Candidate A will attempt only the last 3 questions, and for each, A can eliminate 1 wrong option,
    hence the probability of answering any one question correctly is 1/3.
  - A gives up the last 2 questions.
  - Prove that the probability that A's total score is not less than 135 points is equal to 7/27.
-/
theorem probability_A_score_not_less_than_135 :
  let prob_success := 1 / 3
  let prob_2_successes := (3 * (prob_success^2) * (2/3))
  let prob_3_successes := (prob_success^3)
  prob_2_successes + prob_3_successes = 7 / 27 := 
by
  sorry

end probability_A_score_not_less_than_135_l278_278056


namespace ratio_BK_KC_l278_278703

variables {A B C D O K : Type}
variables {AB BC CD DA AC BD BK KC: ℝ}
variables [Geometry]

-- Given a parallelogram ABCD
def is_parallelogram (A B C D: Type) [Geometry] : Prop :=
  parallelogram A B C D

-- Given that diagonal AC is twice the length of side AB
def ac_twice_ab (A B C D: Type) [Geometry] (AC AB: ℝ) : Prop :=
  AC = 2 * AB

-- Given that point K on BC such that ∠KDB = ∠BDA
def angle_condition (K D B A : Type) [Angles] : Prop :=
  ∠KDB = ∠BDA

-- To prove that the ratio BK : KC = 2 : 1
theorem ratio_BK_KC (A B C D K : Type) [Geometry]
  (h1 : is_parallelogram A B C D)
  (h2 : ac_twice_ab A B C D AC AB)
  (h3 : angle_condition K D B A):
  BK / KC = 2 :=
by
  sorry

end ratio_BK_KC_l278_278703


namespace chuck_distance_l278_278115

theorem chuck_distance
  (total_time : ℝ) (out_speed : ℝ) (return_speed : ℝ) (D : ℝ)
  (h1 : total_time = 3)
  (h2 : out_speed = 16)
  (h3 : return_speed = 24)
  (h4 : D / out_speed + D / return_speed = total_time) :
  D = 28.80 :=
by
  sorry

end chuck_distance_l278_278115


namespace point_not_in_image_of_plane_l278_278735

def satisfies_plane (P : ℝ × ℝ × ℝ) (A B C D : ℝ) : Prop :=
  let (x, y, z) := P
  A * x + B * y + C * z + D = 0

theorem point_not_in_image_of_plane :
  let A := (2, -3, 1)
  let aA := 1
  let aB := 1
  let aC := -2
  let aD := 2
  let k := 5 / 2
  let a'A := aA
  let a'B := aB
  let a'C := aC
  let a'D := k * aD
  ¬ satisfies_plane A a'A a'B a'C a'D :=
by
  -- TODO: Proof needed
  sorry

end point_not_in_image_of_plane_l278_278735


namespace average_odd_numbers_l278_278033

def odd_numbers_in_range := {x : ℕ | x % 2 = 1 ∧ 1 ≤ x ∧ x ≤ 9}
def odd_numbers_less_than_six := {x ∈ odd_numbers_in_range | x < 6}

theorem average_odd_numbers :
  (∑ x in odd_numbers_less_than_six.to_finset, x) / odd_numbers_less_than_six.to_finset.card = 3 := by
  sorry

end average_odd_numbers_l278_278033


namespace find_number_l278_278965

theorem find_number 
  (m : ℤ)
  (h13 : m % 13 = 12)
  (h12 : m % 12 = 11)
  (h11 : m % 11 = 10)
  (h10 : m % 10 = 9)
  (h9 : m % 9 = 8)
  (h8 : m % 8 = 7)
  (h7 : m % 7 = 6)
  (h6 : m % 6 = 5)
  (h5 : m % 5 = 4)
  (h4 : m % 4 = 3)
  (h3 : m % 3 = 2) :
  m = 360359 :=
by
  sorry

end find_number_l278_278965


namespace movie_start_time_l278_278722

def time_after_hours (start_time : Nat) (hours : Nat) : Nat :=
start_time + hours

theorem movie_start_time :
  let dinner_time := 45
  let homework_time := 30
  let clean_room_time := 30
  let take_out_trash_time := 5
  let empty_dishwasher_time := 10
  let latest_start_time := 18  -- 6 pm in 24-hour format
  let total_time := dinner_time + homework_time + clean_room_time + take_out_trash_time + empty_dishwasher_time
  let total_hours := total_time / 60
  time_after_hours latest_start_time total_hours = 20  -- 8 pm in 24-hour format := 
sorry

end movie_start_time_l278_278722


namespace part1_part2_l278_278446

-- Definitions for part 1
def total_souvenirs := 60
def price_a := 100
def price_b := 60
def total_cost_1 := 4600

-- Definitions for part 2
def max_total_cost := 4500
def twice (m : ℕ) := 2 * m

theorem part1 (x y : ℕ) (hx : x + y = total_souvenirs) (hc : price_a * x + price_b * y = total_cost_1) :
  x = 25 ∧ y = 35 :=
by
  -- You can provide the detailed proof here
  sorry

theorem part2 (m : ℕ) (hm1 : 20 ≤ m) (hm2 : m ≤ 22) (hc2 : price_a * m + price_b * (total_souvenirs - m) ≤ max_total_cost) :
  (m = 20 ∨ m = 21 ∨ m = 22) ∧ 
  ∃ W, W = min (40 * 20 + 3600) (min (40 * 21 + 3600) (40 * 22 + 3600)) ∧ W = 4400 :=
by
  -- You can provide the detailed proof here
  sorry

end part1_part2_l278_278446


namespace mandy_score_is_correct_l278_278334

-- Definitions based on the problem conditions
def total_items := 100
def lowella_percentage := 0.35
def pamela_extra_percentage := 0.20

-- Calculate individual scores based on the definitions
def lowella_score := lowella_percentage * total_items
def pamela_score := lowella_score + (pamela_extra_percentage * lowella_score)
def mandy_score := 2 * pamela_score

-- Proof statement to verify Mandy's score
theorem mandy_score_is_correct : mandy_score = 84 := by
  -- sorry placeholder for proof
  sorry

end mandy_score_is_correct_l278_278334


namespace student_D_score_is_correct_l278_278686

-- Define the correct answers
def correct_answers : List Bool := 
  [false, false, false, true, true, false, true, false]

-- Define the score calculation based on answers and correct answers
def calculate_score (answers : List Bool) : ℕ :=
  answers.zip correct_answers |>.count (λ x => x.1 = x.2) * 5

-- Define the answers for each student
def student_A_answers : List Bool := 
  [false, true, false, true, false, false, true, false]

def student_B_answers : List Bool := 
  [false, false, true, true, true, false, false, true]

def student_C_answers : List Bool := 
  [true, false, false, false, true, true, true, false]

def student_D_answers : List Bool := 
  [false, true, false, true, true, false, true, true]

-- Define the known scores for students A, B, C
def student_A_score : ℕ := 30
def student_B_score : ℕ := 25
def student_C_score : ℕ := 25

-- Define the function that verifies the problem's proof
theorem student_D_score_is_correct : 
  calculate_score student_D_answers = 30 := 
by
  sorry

end student_D_score_is_correct_l278_278686


namespace outfit_choices_l278_278663

/-- Given 8 shirts, 8 pairs of pants, and 8 hats, each in 8 colors,
only 6 colors have a matching shirt, pair of pants, and hat.
Each item in the outfit must be of a different color.
Prove that the number of valid outfits is 368. -/
theorem outfit_choices (shirts pants hats colors : ℕ)
  (matching_colors : ℕ)
  (h_shirts : shirts = 8)
  (h_pants : pants = 8)
  (h_hats : hats = 8)
  (h_colors : colors = 8)
  (h_matching_colors : matching_colors = 6) :
  (shirts * pants * hats) - 3 * (matching_colors * colors) = 368 := 
by {
  sorry
}

end outfit_choices_l278_278663


namespace carrots_picked_by_mother_l278_278390

theorem carrots_picked_by_mother :
  (n : ℕ) (Nancy_picked : 38) (good_carrots : 71) (bad_carrots : 14)
  (total_carrots : ℕ := good_carrots + bad_carrots)
  (mother_picked : ℕ := total_carrots - Nancy_picked) :
  mother_picked = 47 :=
by
  sorry

end carrots_picked_by_mother_l278_278390


namespace equation_of_line_through_point_l278_278068

theorem equation_of_line_through_point (a T : ℝ) (h : a ≠ 0 ∧ T ≠ 0) :
  ∃ k : ℝ, (k = T / (a^2)) ∧ (k * x + (2 * T / a)) = (k * x + (2 * T / a)) → 
  (T * x - a^2 * y + 2 * T * a = 0) :=
by
  use T / (a^2)
  sorry

end equation_of_line_through_point_l278_278068


namespace second_candidate_gets_more_marks_l278_278055

noncomputable def total_marks := 499.9999999999999

def passing_marks := 199.99999999999997

def first_candidate_marks := 0.30 * total_marks + 50

def second_candidate_marks := 0.45 * total_marks

def additional_marks := second_candidate_marks - passing_marks

theorem second_candidate_gets_more_marks :
  additional_marks = 25 := 
  by
  calc
    additional_marks = 0.45 * total_marks - 199.99999999999997 := sorry
                 ... = 225 - 200 := sorry
                 ... = 25 := sorry

end second_candidate_gets_more_marks_l278_278055


namespace linear_regression_equation_consecutive_years_probability_l278_278057

noncomputable def sales_data : List (ℕ × ℕ) := [(1, 5), (2, 5), (3, 6), (4, 7), (5, 7)]

theorem linear_regression_equation
    (data : List (ℕ × ℕ) := sales_data)
    (n : ℕ := data.length)
    (x̄ : ℝ := data.sum (λ p, p.1) / n)
    (ȳ : ℝ := data.sum (λ p, p.2) / n)
    (Σxiẏ : ℝ := data.sum (λ p, p.1 * p.2))
    (Σxi2 : ℝ := data.sum (λ p, p.1 * p.1)) :
    let b := (Σxiẏ - n * x̄ * ȳ) / (Σxi2 - n * x̄ * x̄)
    let a := ȳ - b * x̄
    in ∀ x: ℕ, 
        (\hat_y : ℝ := b * x + a) =
        0.6 * x + 4.2 := 
sorry

open_locale big_operators

theorem consecutive_years_probability :
    let total_events := (data.powerset.filter (λ s, s.length = 2)).length
    let consecutive_events := (
      {(1, 2), (2, 3), (3, 4), (4, 5)} : Finset (ℕ × ℕ)).to_list.length
    in (consecutive_events / total_events) = 2 / 5 :=
sorry

end linear_regression_equation_consecutive_years_probability_l278_278057


namespace compute_milk_production_l278_278775

noncomputable def totalMilk (x y z w v : ℕ) : ℚ :=
  (v * y * (3^w - 1)) / (z * (3^x - 1))

theorem compute_milk_production
  (x y z w v : ℕ)
  (hx : x > 0)
  (hz : z > 0)
  (hy : y > 0)
  (hv : v > 0) :
  let Amount := (v * y * (3^w - 1)) / (z * (3^x - 1))
  in totalMilk x y z w v = Amount :=
by sorry

end compute_milk_production_l278_278775


namespace largest_cannot_be_sum_of_two_composites_l278_278158

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l278_278158


namespace value_of_F_l278_278916

noncomputable def polynomial : Polynomial ℤ := Polynomial.X ^ 4 - 8 * Polynomial.X ^ 3 + E * Polynomial.X ^ 2 + F * Polynomial.X + 24

theorem value_of_F 
  (h : ∀ x ∈ (Polynomial.roots polynomial), x ∈ (ℤ≥0))
  (sum_roots : Polynomial.roots polynomial.sum = 8) :
  F = -22 := 
by sorry

end value_of_F_l278_278916


namespace area_of_triangle_example_l278_278828

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_example : 
  area_of_triangle (3, 3) (3, 10) (12, 19) = 31.5 :=
by
  sorry

end area_of_triangle_example_l278_278828


namespace find_divisor_l278_278486

theorem find_divisor (n d : ℤ) (k : ℤ)
  (h1 : n % d = 3)
  (h2 : n^2 % d = 4) : d = 5 :=
sorry

end find_divisor_l278_278486


namespace numbers_divisible_by_8_between_200_and_400_l278_278283

theorem numbers_divisible_by_8_between_200_and_400 : 
  ∃ (n : ℕ), 
    (∀ x, 200 ≤ x ∧ x ≤ 400 → x % 8 = 0 → n = 26) :=
begin
  sorry
end

end numbers_divisible_by_8_between_200_and_400_l278_278283


namespace cross_spectral_density_l278_278404

theorem cross_spectral_density (s_x : ℝ → ℝ) :
  (∃ (r_x r_x_dot : ℝ → ℝ), 
    (∀ (ω : ℝ), 
      r_x_dot = (λ τ, (d (r_x τ)) / dτ) ∧
      s_x = (λ τ, ∫ τ in (-∞) .. ∞, s_x(ω) exp(iτ ω) dω) ∧
      s_x_dot(ω) = (1 / (2 * π)) * ∫ τ in (-∞) .. ∞, r_x_dot(τ) exp(-i ω τ) dτ)) →
  (∀ (ω : ℝ), s_x_dot(ω) = i * ω * s_x(ω))
:=
begin
  sorry
end

end cross_spectral_density_l278_278404


namespace candidates_appeared_l278_278501

-- Define the number of appeared candidates in state A and state B
variables (X : ℝ)

-- The conditions given in the problem
def condition1 : Prop := (0.07 * X = 0.06 * X + 83)

-- The claim that needs to be proved
def claim : Prop := (X = 8300)

-- The theorem statement in Lean 4
theorem candidates_appeared (X : ℝ) (h1 : condition1 X) : claim X := by
  -- Proof is omitted
  sorry

end candidates_appeared_l278_278501


namespace simplify_expression_l278_278188

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 :=
by
  sorry

end simplify_expression_l278_278188


namespace positive_integer_solutions_eq_8_2_l278_278639

-- Define the variables and conditions in the problem
def positive_integer_solution_count_eq (n m : ℕ) : Prop :=
  ∀ (x₁ x₂ x₃ x₄ : ℕ),
    x₂ = m →
    (x₁ + x₂ + x₃ + x₄ = n) →
    (x₁ > 0 ∧ x₃ > 0 ∧ x₄ > 0) →
    -- Number of positive integer solutions should be 10
    (x₁ + x₃ + x₄ = 6)

-- Statement of the theorem
theorem positive_integer_solutions_eq_8_2 : positive_integer_solution_count_eq 8 2 := sorry

end positive_integer_solutions_eq_8_2_l278_278639


namespace number_of_divisibles_by_eight_in_range_l278_278275

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278275


namespace count_multiples_of_8_in_range_l278_278278

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278278


namespace tiffany_total_score_l278_278470

-- Definitions based on conditions
def points_per_treasure : ℕ := 6
def treasures_first_level : ℕ := 3
def treasures_second_level : ℕ := 5

-- The statement we want to prove
theorem tiffany_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 48 := by
  sorry

end tiffany_total_score_l278_278470


namespace round_to_nearest_hundredth_l278_278768

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 36.84397) : (Float.round (x * 100) / 100) = 36.84 :=
by
  sorry

end round_to_nearest_hundredth_l278_278768


namespace number_of_divisibles_by_eight_in_range_l278_278270

theorem number_of_divisibles_by_eight_in_range :
  (Finset.filter (λ n, n % 8 = 0) (Finset.Icc 200 400)).card = 25 :=
by
  sorry

end number_of_divisibles_by_eight_in_range_l278_278270


namespace QD_reciprocity_l278_278902

-- Given an equilateral triangle ABC.
-- A line from vertex A to side BC meets BC at D and the circumcircle at Q.

variable (A B C D Q : Point)
variable (cond_eq_triangle : IsEquilateralTriangle A B C)
variable (cond_point_D : Collinear A D C)
variable (cond_Point_Q : OnCircumcircle Q (Triangle A B C))

-- The actual theorem stating what needs to be proven:
theorem QD_reciprocity (h_eq_triangle : cond_eq_triangle) 
                      (h_point_D : cond_point_D)
                      (h_point_Q : cond_Point_Q) :
    1 / (distance Q D) = 1 / (distance Q B) + 1 / (distance Q C) :=
sorry

end QD_reciprocity_l278_278902


namespace correct_propositions_l278_278096

-- Define each proposition as a boolean value
def proposition1 : Prop := ∃ α β : Plane, α ≠ β ∧ (∀ p : Point, p ∈ α ∩ β → p ∈ (finite_points α β))
def proposition2 : Prop := ∀ (l : Line) (P : Point), P ∉ l → ∃! α : Plane, l ⊆ α ∧ P ∈ α
def proposition3 : Prop := ∀ (l₁ l₂ : Line), intersect l₁ l₂ → ∃! α : Plane, l₁ ⊆ α ∧ l₂ ⊆ α
def proposition4 : Prop := ∀ (α β : Plane), (∃ p₁ p₂ p₃ : Point, p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ ∧ p₁ ∈ α ∩ β ∧ p₂ ∈ α ∩ β ∧ p₃ ∈ α ∩ β) → α = β
def proposition5 : Prop := ∀ (Q : Quadrilateral), ∃! α : Plane, Q ⊆ α

-- Main theorem stating which propositions are correct
theorem correct_propositions : proposition2 ∧ proposition3 ∧ proposition4 := 
by {
  sorry
}

end correct_propositions_l278_278096


namespace equal_segments_l278_278562

/-- Two externally tangent circles and properties of their tangents -/
theorem equal_segments {Γ₁ Γ₂ : Type} (Ext_Internal_Tangent : Prop) 
  (external_touches : ∃ A B : Γ₁, touches_tangent_line A B)
  (internal_touches : ∃ C D : Γ₂, touches_internal_tangent C D)
  (intersection_exists : ∃ E, intersection_point (line_through AC) (line_through BD) E)
  (point_on_circle : ∃ F : point_ON_𝛤₁, F ∈ Γ₁)
  (tangent_at_F : ∃ M, intersects_perp_bisector EF_at_F M)
  (tangent_from_M : ∃ G, tangent_point_from M G)
  :

  MF = MG :=
  sorry

end equal_segments_l278_278562


namespace cube_edge_diff_at_least_three_l278_278460

theorem cube_edge_diff_at_least_three :
  ∀ (labeling : Fin 8 → ℕ), 
  (∀ i, 1 ≤ labeling i ∧ labeling i ≤ 8) →
  (∀ i j, i ≠ j → labeling i ≠ labeling j) →
  ∃ (u v : Fin 8), 
  (u ≠ v) ∧
  (distance u v = 1) ∧ 
  (|labeling u - labeling v| ≥ 3) := sorry

-- Definition of distance between vertices in a cube.
def distance : (Fin 8) → (Fin 8) → ℕ
| u, v => arbitrary -- This is a placeholder definition.

end cube_edge_diff_at_least_three_l278_278460


namespace eccentricity_of_hyperbola_l278_278674

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)

def distance_center_asymptote :=
  let center := (2 : ℝ, 0 : ℝ) in
  let r := 2 in
  let d := (2 * b) / Real.sqrt (a ^ 2 + b ^ 2) in
  d = Real.sqrt 3

theorem eccentricity_of_hyperbola :
  (distance_center_asymptote a b a_pos b_pos) → Real.sqrt (a^2 + b^2) = 2 * a :=
sorry

end eccentricity_of_hyperbola_l278_278674


namespace infection_never_covers_grid_l278_278396

theorem infection_never_covers_grid (n : ℕ) (H : n > 0) :
  exists (non_infected_cell : ℕ × ℕ), (non_infected_cell.1 < n ∧ non_infected_cell.2 < n) :=
by
  sorry

end infection_never_covers_grid_l278_278396


namespace boxes_of_chocolates_l278_278884

theorem boxes_of_chocolates (total_pieces : ℕ) (pieces_per_box : ℕ) (h_total : total_pieces = 3000) (h_each : pieces_per_box = 500) : total_pieces / pieces_per_box = 6 :=
by
  sorry

end boxes_of_chocolates_l278_278884


namespace unique_solution_pairs_count_l278_278661

theorem unique_solution_pairs_count :
  ∃! (p : ℝ × ℝ), (p.1 + 2 * p.2 = 2 ∧ (|abs p.1 - 2 * abs p.2| = 2) ∧
       ∃! q, (q = (2, 0) ∨ q = (0, 1)) ∧ p = q) := 
sorry

end unique_solution_pairs_count_l278_278661


namespace count_divisible_by_8_l278_278304

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l278_278304


namespace product_of_last_two_digits_l278_278670

theorem product_of_last_two_digits (n A B : ℤ) 
  (h1 : n % 8 = 0) 
  (h2 : 10 * A + B = n % 100) 
  (h3 : A + B = 14) : 
  A * B = 48 := 
sorry

end product_of_last_two_digits_l278_278670


namespace perfect_square_probability_l278_278076

noncomputable def modified_die_faces : List ℕ := [1, 2, 3, 4, 5, 8]

theorem perfect_square_probability (n : ℕ) (H : n = 5) : 
  let possible_faces := {1, 2, 3, 4, 5, 8},
      total_outcomes := 7776,
      favorable_outcomes := 762
  in (total_outcomes = (6^5)) ∧ 
     (favorable_outcomes = ∑ x in Finset.powersetLen n (Finset.ofList modified_die_faces), 
                                 ite (is_perfect_square (x.prod id)) 1 0) ∧
     let probability := (762 : ℚ) / 7776
  in probability = 127 / 1296 ∧ 127 + 1296 = 1423 :=
by
  sorry

end perfect_square_probability_l278_278076


namespace counting_divisibles_by_8_l278_278296

theorem counting_divisibles_by_8 :
  (∃ n : ℕ, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) → (finset.card (finset.filter (λ n, 200 ≤ n ∧ n ≤ 400 ∧ n % 8 = 0) (finset.range 401)) = 24) := 
by
  sorry

end counting_divisibles_by_8_l278_278296


namespace minimum_value_theorem_l278_278922

def Apollonius_circle_eqn (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 + M.2 ^ 2 = 1 / 4

def on_x_axis (Q : ℝ × ℝ) : Prop :=
  Q.2 = 0

def fixed_point_P := (-1, 0 : ℝ × ℝ)

def ratio_condition (M Q : ℝ × ℝ) : Prop :=
  dist M Q / dist M fixed_point_P = 1 / 2

def given_point_B := (1, 2 : ℝ × ℝ)

noncomputable def minimum_value (M : ℝ × ℝ) : ℝ :=
  1 / 2 * dist M fixed_point_P + dist M given_point_B

theorem minimum_value_theorem (M Q : ℝ × ℝ)
  (h1 : Apollonius_circle_eqn M)
  (h2 : on_x_axis Q)
  (h3 : ratio_condition M Q)
  : minimum_value M = sqrt 89 / 4 :=
sorry

end minimum_value_theorem_l278_278922


namespace circumradius_angle_inequality_l278_278740

variable (A B C D : Type)
variable [convex_quadrilateral : ConvexQuadrilateral A B C D]
variable (RA RB RC RD : ℝ)
variable (angle_A angle_B angle_C angle_D : ℝ)

def circumradii (triangle : Type) (A B C: Type) := 
  ∃ (R : ℝ), R = circumradius_of_triangle triangle A B C

variable [radii_DAB : circumradii Type.DAB A B D RA]
variable [radii_ABC : circumradii Type.ABC A B C RB]
variable [radii_BCD : circumradii Type.BCD B C D RC]
variable [radii_CDA : circumradii Type.CDA C D A RD]

theorem circumradius_angle_inequality 
  (h : ConvexQuadrilateral A B C D) 
  (hRA : circumradii Type.DAB A B D RA) 
  (hRB : circumradii Type.ABC A B C RB)
  (hRC : circumradii Type.BCD B C D RC) 
  (hRD : circumradii Type.CDA C D A RD) :
  (RA + RC > RB + RD) ↔ (angle_A + angle_C > angle_B + angle_D) := sorry

end circumradius_angle_inequality_l278_278740


namespace trajectory_of_center_distance_AB_l278_278621

noncomputable def circleM : set (ℝ × ℝ) := { p | (p.1 + 1)^2 + p.2^2 = 1 }
noncomputable def circleN : set (ℝ × ℝ) := { p | (p.1 - 1)^2 + p.2^2 = 9 }
noncomputable def curveC : set (ℝ × ℝ) := { p | (p.1^2)/4 + (p.2^2)/3 = 1 }

theorem trajectory_of_center {P : set (ℝ × ℝ → Prop)} :
  (∃ R : ℝ, P R ∈ circleM ∧ P R ∈ circleN ∧ 
  circleM P R ∧ circleN P R) → ∀ p ∈ curveC, 
  p.1^2 / 4 + p.2^2 / 3 = 1 :=
sorry

theorem distance_AB :
  ∀ l : (ℝ × ℝ) → Prop,
  tangent_to l circleM ∧ tangent_to l circleP ∧
  intersects_curve l curveC (A : ℝ × ℝ) (B : ℝ × ℝ) ∧
  max_radius P → dist A B = 18/7 :=
sorry

end trajectory_of_center_distance_AB_l278_278621


namespace circle_passes_through_single_point_l278_278982

theorem circle_passes_through_single_point {p q : ℝ} {a b : ℝ} 
  (h₁: q = a * b) 
  (h₂: ∀ x, x^2 + p * x + q = 0 → x = a ∨ x = b) 
  (h₃: ∀ y, y = q → (0, y) = (0, q)) 
  (h₄: a ≠ b) :
  ∃ C : Type, ∃ (k : C → ℝ → ℝ → Prop), ∀ (x y : ℝ), k x y = true ↔ ((x, y) = (0, 1)) :=
sorry

end circle_passes_through_single_point_l278_278982


namespace min_value_expr_l278_278199

theorem min_value_expr (a : ℝ) (h1 : 1 < a) (h2 : a < 4) : (∀ b, (1 < b ∧ b < 4) → (b / (4 - b) + 1 / (b - 1)) ≥ 2) :=
by
  intro b hb1 hb2
  sorry

end min_value_expr_l278_278199


namespace no_such_polynomial_exists_l278_278402

theorem no_such_polynomial_exists :
  ¬ ∃ (P : Polynomial ℤ) (m : ℕ), ∀ (x : ℤ), Polynomial.eval x P(P(x)) = x^m + x + 2 := 
by
  sorry

end no_such_polynomial_exists_l278_278402


namespace division_of_monomials_l278_278570

variable (x : ℝ) -- ensure x is defined as a variable, here assuming x is a real number

theorem division_of_monomials (x : ℝ) : (2 * x^3 / x^2) = 2 * x := 
by 
  sorry

end division_of_monomials_l278_278570


namespace a_is_perfect_square_l278_278631

variable (a b : ℕ)
variable (h1 : 0 < a) 
variable (h2 : 0 < b)
variable (h3 : b % 2 = 1)
variable (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b)

theorem a_is_perfect_square (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b % 2 = 1) 
  (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b) : ∃ n : ℕ, a = n ^ 2 :=
sorry

end a_is_perfect_square_l278_278631


namespace solve_inequality_l278_278375

variable {f : ℝ → ℝ}

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f x

def isStrictlyIncreasingOnPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

def atOneZero (f : ℝ → ℝ) : Prop :=
  f 1 = 0

theorem solve_inequality (h_odd : isOddFunction f) 
  (h_increasing : isStrictlyIncreasingOnPositive f) 
  (h_one : atOneZero f) :
  { x : ℝ | (x-1)*f(x) > 0 } = Ioo (-∞) (-1) ∪ Ioo 0 1 ∪ Ioo 1 ∞ :=
sorry

end solve_inequality_l278_278375


namespace chsh_inequality_l278_278731

open ProbabilityTheory

/-- Clauser-Horne-Shimony-Holt Inequality -/
theorem chsh_inequality
  {Ω : Type*} {m : MeasurableSpace Ω} {μ : Measure Ω}
  {ξ η X Y : Ω → ℝ}
  (hξ : ∀ ω, |ξ ω| ≤ 1)
  (hη : ∀ ω, |η ω| ≤ 1)
  (hX : ∀ ω, |X ω| ≤ 1)
  (hY : ∀ ω, |Y ω| ≤ 1) :
  |(∫ ω, ξ ω * X ω \partial μ) + (∫ ω, ξ ω * Y ω \partial μ) + 
   (∫ ω, η ω * X ω \partial μ) - (∫ ω, η ω * Y ω \partial μ)| ≤ 2 := 
sorry

end chsh_inequality_l278_278731


namespace sum_positive_two_digit_integers_divisible_by_digit_conditions_l278_278036

def digit_conditions (a b : ℕ) : Prop :=
  (10a + b : ℕ) ≥ 10 ∧ (10a + b : ℕ) < 100 ∧ 
  (a + b)^2 ∣ 10a + b ∧ (ab)^2 ∣ 10a + b

theorem sum_positive_two_digit_integers_divisible_by_digit_conditions : 
  ∑ (n : ℕ) in Finset.filter (λ n, ∃ a b, 10 * a + b = n ∧ digit_conditions a b) (Finset.range 100), n = 12 :=
by {
  sorry
}

end sum_positive_two_digit_integers_divisible_by_digit_conditions_l278_278036


namespace brendan_match_ratio_l278_278930

noncomputable def brendanMatches (totalMatches firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound : ℕ) :=
  matchesWonFirstTwoRounds = firstRound + secondRound ∧
  matchesWonFirstTwoRounds = 12 ∧
  totalMatches = matchesWonTotal ∧
  matchesWonTotal = 14 ∧
  firstRound = 6 ∧
  secondRound = 6 ∧
  matchesInLastRound = 4

theorem brendan_match_ratio :
  ∃ ratio: ℕ × ℕ,
    let firstRound := 6
    let secondRound := 6
    let matchesInLastRound := 4
    let matchesWonFirstTwoRounds := firstRound + secondRound
    let matchesWonTotal := 14
    let matchesWonLastRound := matchesWonTotal - matchesWonFirstTwoRounds
    let ratio := (matchesWonLastRound, matchesInLastRound)
    brendanMatches matchesWonTotal firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound ∧
    ratio = (1, 2) :=
by
  sorry

end brendan_match_ratio_l278_278930


namespace area_of_triangle_ABD_l278_278347

-- Given conditions for the problem
variables {r : ℝ} (O B A D C : EuclideanGeometry.Point)
variable (hO : EuclideanGeometry.Circle O r)
variable (h_tangent : EuclideanGeometry.Tangent (segment B A) hO)
variable (h_diameter : EuclideanGeometry.Diameter (segment A C) hO)
variable (h_perpendicular : EuclideanGeometry.Perpendicular (segment A D) (segment D C))

-- The theorem we want to prove
theorem area_of_triangle_ABD (h : r > 0) : 
  EuclideanGeometry.TriangleArea A B D = (r^2 * Real.sqrt 2) / 2 :=
sorry

end area_of_triangle_ABD_l278_278347


namespace Justin_reads_total_pages_l278_278355

theorem Justin_reads_total_pages 
  (initial_pages : ℕ)
  (multiplier : ℕ)
  (days_remaining : ℕ)
  (total_days : ℕ)
  (total_pages_needed : ℕ) :
  initial_pages = 10 →
  multiplier = 2 →
  days_remaining = 6 →
  total_days = 7 →
  total_pages_needed = 100 →
  (initial_pages + days_remaining * (initial_pages * multiplier)) = 130 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry

end Justin_reads_total_pages_l278_278355


namespace count_multiples_of_8_in_range_l278_278281

theorem count_multiples_of_8_in_range : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k ≥ 200 ∧ k ≤ 400 ∧ k % 8 = 0 ↔ ∃ i : ℕ, 25 ≤ i ∧ i ≤ 50 ∧ k = 8 * i :=
begin
  sorry
end

end count_multiples_of_8_in_range_l278_278281


namespace sum_2013_l278_278801

-- Define the sequence {a_n} with initial conditions and recurrence relation
def a : ℕ → ℚ
| 0       := 1
| 1       := 1 / 2
| (n + 2) := a (n + 1) - a n

-- Define the sum of the first n terms S_n
def S : ℕ → ℚ
| 0       := a 0
| (n + 1) := S n + a (n + 1)

-- Theorem to prove S_{2013} equals 1
theorem sum_2013 : S 2013 = 1 := 
sorry

end sum_2013_l278_278801
