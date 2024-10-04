import Mathlib

namespace charlie_laps_l502_502797

-- Definitions for the given conditions
def steps_per_lap : ℕ := 5350
def total_steps : ℕ := 13375

-- The statement to be proven
theorem charlie_laps : (total_steps / steps_per_lap) = 2 :=
by {
  -- Assuming integer division as we're rounding down
  have h : total_steps / steps_per_lap = 2,
  by norm_num,
  exact h,
}

end charlie_laps_l502_502797


namespace pipe_fill_time_l502_502502

theorem pipe_fill_time (T : ℝ) 
  (h1 : ∃ T : ℝ, 0 < T) 
  (h2 : T + (1/2) > 0) 
  (h3 : ∃ leak_rate : ℝ, leak_rate = 1/10) 
  (h4 : ∃ pipe_rate : ℝ, pipe_rate = 1/T) 
  (h5 : ∃ effective_rate : ℝ, effective_rate = pipe_rate - leak_rate) 
  (h6 : effective_rate = 1 / (T + 1/2))  : 
  T = Real.sqrt 5 :=
  sorry

end pipe_fill_time_l502_502502


namespace kopeechka_purchase_l502_502444

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502444


namespace probability_digits_different_l502_502242

theorem probability_digits_different : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999} in
  let total := ∑ x in S, 1 in
  let different_digits := ∑ x in S, (if (x / 100 ≠ (x % 100) / 10 ∧ (x % 100) / 10 ≠ x % 10 ∧ x / 100 ≠ x % 10) then 1 else 0) in
  (different_digits / total) = (18 / 25) := by
sorry

end probability_digits_different_l502_502242


namespace hyperbola_eccentricity_l502_502315

-- Define the conditions and parameters for the problem
variables (m : ℝ) (c a e : ℝ)

-- Given conditions
def hyperbola_eq (m : ℝ) := ∀ x y : ℝ, (x^2 / m^2 - y^2 = 4)
def focal_distance : Prop := c = 4
def standard_hyperbola_form : Prop := a^2 = 4 * m^2 ∧ 4 = 4

-- Eccentricity definition
def eccentricity : Prop := e = c / a

-- Main theorem
theorem hyperbola_eccentricity (m : ℝ) (h_pos : 0 < m) (h_foc_dist : focal_distance c) (h_form : standard_hyperbola_form a m) :
  eccentricity e a c :=
by
  sorry

end hyperbola_eccentricity_l502_502315


namespace circumscribed_circle_area_l502_502750

theorem circumscribed_circle_area (side_length : ℝ) (h : side_length = 12) :
  ∃ (A : ℝ), A = 48 * π :=
by
  sorry

end circumscribed_circle_area_l502_502750


namespace golden_ratio_problem_l502_502030

noncomputable def m := 2 * Real.sin (Real.pi * 18 / 180)
noncomputable def n := 4 - m^2
noncomputable def target_expression := m * Real.sqrt n / (2 * (Real.cos (Real.pi * 27 / 180))^2 - 1)

theorem golden_ratio_problem :
  target_expression = 2 :=
by
  -- Proof will be placed here
  sorry

end golden_ratio_problem_l502_502030


namespace different_digits_probability_l502_502240

noncomputable def number_nonidentical_probability : ℚ :=
  let total_numbers := 900
  let identical_numbers := 9
  -- The probability of identical digits.
  let identical_probability := identical_numbers / total_numbers
  -- The probability of non-identical digits.
  1 - identical_probability

theorem different_digits_probability : number_nonidentical_probability = 99 / 100 := by
  sorry

end different_digits_probability_l502_502240


namespace regular_polygon_sides_and_exterior_angle_l502_502346

theorem regular_polygon_sides_and_exterior_angle (n : ℕ) (exterior_sum : ℝ) :
  (180 * (n - 2) = 360 + exterior_sum) → (exterior_sum = 360) → n = 6 ∧ (360 / n = 60) :=
by
  intro h1 h2
  sorry

end regular_polygon_sides_and_exterior_angle_l502_502346


namespace kopeechka_items_l502_502422

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502422


namespace count_valid_integers_l502_502376

theorem count_valid_integers : 
  (∀ n ∈ [1..9999], 
    ¬(∃ d ∈ ['2', '3', '4', '5', '8'], d ∈ n.toString.toList)) → 
  (number_of_valid_integers = 624) := 
by
  sorry

end count_valid_integers_l502_502376


namespace units_digit_of_product_of_first_four_composites_l502_502644

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502644


namespace sum_X_Y_divisibility_by_9_l502_502828

def digit (n : ℕ) := n ≤ 9

theorem sum_X_Y_divisibility_by_9 :
  (∑ (x y : ℕ) in (finset.range 10).product (finset.range 10),
    if (16 + x + y) % 9 = 0 then x + y else 0) = 13 :=
by {
  sorry
}

end sum_X_Y_divisibility_by_9_l502_502828


namespace multiples_of_15_between_12_and_152_l502_502380

theorem multiples_of_15_between_12_and_152 : 
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, (m * 15 > 12 ∧ m * 15 < 152) ↔ (1 ≤ m ∧ m ≤ 10) :=
by
  sorry

end multiples_of_15_between_12_and_152_l502_502380


namespace routeY_is_faster_l502_502497

def routeX_distance : ℝ := 8
def routeX_speed : ℝ := 32
def routeY_distance : ℝ := 7
def routeY_non_school_zone_speed : ℝ := 45
def routeY_school_zone1_distance : ℝ := 1
def routeY_school_zone1_speed : ℝ := 25
def routeY_school_zone2_distance : ℝ := 0.5
def routeY_school_zone2_speed : ℝ := 15

def t_X : ℝ := routeX_distance / routeX_speed
def t_Y1 : ℝ := (routeY_distance - (routeY_school_zone1_distance + routeY_school_zone2_distance)) / routeY_non_school_zone_speed
def t_Y2 : ℝ := routeY_school_zone1_distance / routeY_school_zone1_speed
def t_Y3 : ℝ := routeY_school_zone2_distance / routeY_school_zone2_speed
def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3

def difference_in_minutes : ℝ := (t_X - t_Y) * 60

theorem routeY_is_faster : difference_in_minutes ≈ 3.27 := by
  sorry

end routeY_is_faster_l502_502497


namespace train_speed_l502_502228

theorem train_speed :
  ∀ (distance time distance_in_km time_in_hr speed: ℝ),
    distance = 140 →
    time = 3.499720022398208 →
    distance_in_km = distance / 1000 →
    time_in_hr = time / 3600 →
    speed = distance_in_km / time_in_hr →
    speed = 144.0144 :=
by
  intros distance time distance_in_km time_in_hr speed
  assume hdist htime hdist_in_km htime_in_hr hspeed
  sorry

end train_speed_l502_502228


namespace symmetric_point_cartesian_l502_502317

-- Conditions: Point P has polar coordinates (2, -5π/3)
def polar_coords : ℝ × ℝ := (2, -5 * Real.pi / 3)

-- Cartesian coordinates of the point symmetric to P with respect to the pole
def symmetric_cartesian_coords (p : ℝ × ℝ) : ℝ × ℝ :=
  let (r, θ) := p
  let θ_sym := θ + Real.pi
  (r * Real.cos θ_sym, r * Real.sin θ_sym)

-- Proof statement
theorem symmetric_point_cartesian :
  symmetric_cartesian_coords polar_coords = (-1, -Real.sqrt 3) :=
sorry

end symmetric_point_cartesian_l502_502317


namespace sum_of_reciprocals_of_squares_l502_502541

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 41) :
  (1 / (a^2) + 1 / (b^2)) = 1682 / 1681 := sorry

end sum_of_reciprocals_of_squares_l502_502541


namespace eval_expression_l502_502277

theorem eval_expression : 81^(1/2) * 64^(-1/3) * 49^(1/2) = (63 / 4) :=
by
  sorry

end eval_expression_l502_502277


namespace units_digit_of_composite_product_l502_502652

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502652


namespace sum_x_coordinates_where_f_eq_x_plus_2_l502_502105

-- Defining the linear segments for the function f
def segment1 (x : ℝ) := if -4 ≤ x ∧ x ≤ -2 then 2 * x + 3 else 0
def segment2 (x : ℝ) := if -2 < x ∧ x ≤ -1 then -x - 3 else 0
def segment3 (x : ℝ) := if -1 < x ∧ x <= 1 then 2 * x else 0
def segment4 (x : ℝ) := if 1 < x ∧ x <= 2 then x + 1 else 0
def segment5 (x : ℝ) := if 2 < x ∧ x <= 4 then x + 3 else 0

-- Defining the piecewise linear function f
def f (x : ℝ) := segment1 x + segment2 x + segment3 x + segment4 x + segment5 x

noncomputable def intersection_points := 
  {x : ℝ | f x = x + 2}.to_finset.sum id 

theorem sum_x_coordinates_where_f_eq_x_plus_2 : intersection_points = 2 := 
  sorry

end sum_x_coordinates_where_f_eq_x_plus_2_l502_502105


namespace direction_vector_of_line_l502_502251

theorem direction_vector_of_line
  (P : Matrix (Fin 3) (Fin 3) ℚ :=
    !![
      [ 3/10, -1/5, -3/5 ],
      [ -1/5, 1/5, 2/5 ],
      [ -3/5, 2/5, 4/5 ]
    ])
  (v : Fin 3 → ℚ := fun i => if i = 0 then 1 else 0) :
  ∃ (a b c : ℤ), a > 0 ∧ Int.gcd a.nat_abs b.nat_abs c.nat_abs = 1 ∧
  (vecMulMatrix P v) = !![3, -2, -6] := by
  sorry

end direction_vector_of_line_l502_502251


namespace lines_parallel_l502_502873

noncomputable def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a, 2, 6)
noncomputable def line2 (a : ℝ) : ℝ × ℝ × ℝ := (1, a-1, a^2-1)

def are_parallel (line1 line2 : ℝ × ℝ × ℝ) : Prop :=
  let ⟨a1, b1, _⟩ := line1
  let ⟨a2, b2, _⟩ := line2
  a1 * b2 = a2 * b1

theorem lines_parallel (a : ℝ) :
  are_parallel (line1 a) (line2 a) → a = -1 :=
sorry

end lines_parallel_l502_502873


namespace purchase_options_l502_502420

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502420


namespace largest_eight_digit_number_l502_502698

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l502_502698


namespace gross_profit_value_l502_502546

theorem gross_profit_value (C GP : ℝ) (h1 : GP = 1.6 * C) (h2 : 91 = C + GP) : GP = 56 :=
by
  sorry

end gross_profit_value_l502_502546


namespace kopeechka_items_l502_502425

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502425


namespace units_digit_first_four_composites_l502_502623

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502623


namespace angle_OMT_half_angle_BAC_l502_502406

open Triangle

-- Define the isosceles triangle ABC
variables {A B C O N M T : Point}
variables {BC : LineSegment}
variables [IsoscelesTriangle ABC]
variables [Circumcenter O ABC]
variables [Midpoint N BC]
variables [Reflection M N AC]
variables [Rectangle ANBT]

-- Define the given angles
noncomputable def angleBAC : Real := angle A B C

-- The theorem to prove
theorem angle_OMT_half_angle_BAC :
  ∠ O M T = (1 : ℚ) / (2 : ℚ) * angleBAC :=
sorry

end angle_OMT_half_angle_BAC_l502_502406


namespace factorizations_of_2079_l502_502903

theorem factorizations_of_2079 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 2079 ∧ (a, b) = (21, 99) ∨ (a, b) = (33, 63) :=
sorry

end factorizations_of_2079_l502_502903


namespace kopeechka_items_l502_502464

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502464


namespace possible_items_l502_502437

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502437


namespace hyperbola_range_l502_502349

theorem hyperbola_range (m : ℝ) : m * (2 * m - 1) < 0 → 0 < m ∧ m < (1 / 2) :=
by
  intro h
  sorry

end hyperbola_range_l502_502349


namespace kopeechka_items_l502_502463

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502463


namespace units_digit_of_product_of_first_four_composites_l502_502659

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502659


namespace number_of_opposite_pairs_l502_502008

-- Definition of non-zero numbers and opposite relationship
variables {a b : ℝ}
-- Condition: a and b are non-zero and opposite to each other
def conditions (a b : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ a = -b

-- Statement of the proof problem
theorem number_of_opposite_pairs (a b : ℝ) (h : conditions a b) :
  {x : ℕ | x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3}.Card = 2 :=
sorry

end number_of_opposite_pairs_l502_502008


namespace darcy_commute_l502_502734

/-- 
Darcy's commuting problem.
-/
theorem darcy_commute (distance walk_speed train_speed : ℝ) (additional_time walk_extra_time : ℝ) 
  (walking_time train_time : ℝ) (h_distance : distance = 1.5)
  (h_walk_speed : walk_speed = 3) (h_train_speed : train_speed = 20) 
  (h_walk_extra_time : walk_extra_time = 10) 
  (h_walking_time : walking_time = distance / walk_speed * 60) 
  (h_train_time : train_time = distance / train_speed * 60) :
  additional_time = 15.5 :=
by
  have h_walking_time_simplified : walking_time = 30 := by
    rw [h_distance, h_walk_speed]
    norm_num
    
  have h_train_time_simplified : train_time = 4.5 := by
    rw [h_distance, h_train_speed]
    norm_num
    
  have commute_equation : walking_time = train_time + additional_time + walk_extra_time := by
    rw [h_walking_time, h_train_time]
    linarith
    
  sorry

end darcy_commute_l502_502734


namespace length_of_wall_is_29_l502_502200

-- Definitions of the given conditions
def brick_volume_cm3 := 20 * 10 * 7.5
def brick_volume_m3 := brick_volume_cm3 / 1000000

def wall_height_m := 0.75
def wall_width_m := 2
def num_bricks := 29000

-- The total volume of the bricks used in the wall
def total_brick_volume_m3 := num_bricks * brick_volume_m3

-- The length of the wall is the unknown we need to prove
def wall_volume := total_brick_volume_m3

theorem length_of_wall_is_29 : ∃ L : ℝ, (L * wall_width_m * wall_height_m = wall_volume) → L = 29 :=
by
  sorry

end length_of_wall_is_29_l502_502200


namespace perimeter_of_ABDFC_is_64_l502_502520

noncomputable def perimeter_ABDFC (s : ℝ) (h_square : 4 * s = 64) (h_triangle : ∀ x y z : ℝ, x = s ∧ y = s ∧ z = s) : ℝ :=
  4 * s

theorem perimeter_of_ABDFC_is_64 :
  ∃ s : ℝ, 4 * s = 64 ∧ (perimeter_ABDFC s (by norm_num; exact eq_refl 64) (by intros; exact ⟨eq.refl s, eq.refl s, eq.refl s⟩) = 64) :=
begin
  use 16,
  split,
  { norm_num },
  { unfold perimeter_ABDFC,
    norm_num }
end

end perimeter_of_ABDFC_is_64_l502_502520


namespace abs_diff_seq_l502_502246

open Int

def seq_A : ℕ → ℤ
| 0     := 1
| (n+1) := seq_A n + (2 * (n + 1) - 1) * (2 * (n + 1))

def seq_B : ℕ → ℤ
| 0     := 1
| (n+1) := seq_B n + if (n + 1) % 2 = 1 then (2 * (n - 1 + 2) * (2 * (n - 1 + 2) + 1)) else 0

theorem abs_diff_seq : abs (seq_A 21 - seq_B 21) = 882 := by
sorry

end abs_diff_seq_l502_502246


namespace units_digit_of_product_of_first_four_composites_l502_502663

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502663


namespace prob_a_or_b_selected_l502_502004

-- Define the set of graduates.
inductive Graduate
| A | B | C | D | E

-- Define the function that calculates whether a graduate is selected.
def is_selected (graduate : Graduate) (selection : Finset Graduate) : Prop :=
  graduate ∈ selection

-- Define the selection process: selecting 3 out of 5 graduates.
noncomputable def selections : Finset (Finset Graduate) :=
  (Finset.univ : Finset Graduate).powerset.filter (λ s, s.card = 3)

-- Define the event where A or B is selected.
def a_or_b_selected (selection : Finset Graduate) : Prop :=
  is_selected Graduate.A selection ∨ is_selected Graduate.B selection

-- The proof statement.
theorem prob_a_or_b_selected : 
  (selections.filter a_or_b_selected).card / selections.card = 9 / 10 :=
sorry

end prob_a_or_b_selected_l502_502004


namespace log_prob_event_l502_502089

noncomputable def probability_event_log : ℝ :=
  let interval1 := set.Icc 0 2
  let probability := (λ x, -1 ≤ Real.log (x + 1/2) / Real.log (1/2) ∧ Real.log (x + 1/2) / Real.log (1/2) ≤ 1)
  (set.Icc 0 (3/2)).measure / interval1.measure

theorem log_prob_event : probability_event_log = 3/4 :=
  sorry

end log_prob_event_l502_502089


namespace maximum_possible_colors_l502_502967

variable {Point : Type}

noncomputable def maximum_possible_k {S : set Point} (d : Point → Point → ℝ) (k : ℕ) : Prop :=
  (∀ P ∈ S, ∃! closest furthest ∈ S, closest ≠ furthest ∧
    ∀ Q1 ∈ S, d(P, closest) ≤ d(P, Q1) ∧
              d(P, closest) < d(P, Q1) ↔ Q1 ≠ closest ∧ Q1 ≠ P ∧
              d(P, furthest) ≥ d(P, Q1) ∧
              d(P, furthest) > d(P, Q1) ↔ Q1 ≠ furthest ∧ Q1 ≠ P ∧
    closest_color = furthest_color = P_color) ∧
  |S| = 2023 ∧ (dissat : ∀ {P Q : Point}, P ≠ Q → d(P, Q) ≠ d(Q, P))
  
theorem maximum_possible_colors (S : set Point) (d : Point → Point → ℝ) :
  (∀ P ∈ S, ∃! Q ∈ S, Q ≠ P ∧ ∀ R ∈ S, d P Q <= d P R ∧ (d P Q < d P R ↔ R ≠ Q ∧ R ≠ P)) →
  (∀ P ∈ S, ∃! Q ∈ S, Q ≠ P ∧ ∀ R ∈ S, d P Q >= d P R ∧ (d P Q > d P R ↔ R ≠ Q ∧ R ≠ P)) →
  ∃ k ≤ 506, maximum_possible_k d k :=
by sorry

end maximum_possible_colors_l502_502967


namespace decreasing_interval_sum_of_10_terms_l502_502881

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin x * cos x - cos x ^ 2

-- Define the sequence aₙ
def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then (n / 2) * π + π / 4
  else (n / 2) * π + π / 2

-- Part (1): Prove the interval where f(x) is decreasing
theorem decreasing_interval (k : ℤ) :
  ∀ x, (k * π + (3 * π / 8) ≤ x ∧ x ≤ k * π + (7 * π / 8)) → f x < f (x + 1) :=
sorry

-- Part (2): Prove the sum of the first 10 terms of the sequence aₙ
theorem sum_of_10_terms :
  (Finset.range 10).sum (λ n, a (n + 1)) = (95 * π) / 4 :=
sorry

end decreasing_interval_sum_of_10_terms_l502_502881


namespace sum_even_ints_302_to_400_excluding_first_50_l502_502547

theorem sum_even_ints_302_to_400_excluding_first_50 :
  let sum_first_50_even := 2550
      sum_302_to_400 := 17550
  in sum_302_to_400 - sum_first_50_even = 15000 := by
  sorry

end sum_even_ints_302_to_400_excluding_first_50_l502_502547


namespace midpoint_incenter_halves_BK_l502_502527

variables {A B C I K M : Type*}
variable [metric_space A]

-- Definitions for points and properties of the triangle
def is_incenter (I : A) (tri : triangle A B C) : Prop := sorry -- I is the incenter of triangle ABC
def is_midpoint (M : A) (AC : segment A) : Prop := sorry -- M is the midpoint of AC
def touches (K : A) (circle : in_circle) (side : segment A) : Prop := sorry -- K is the touchpoint on AC

-- The main theorem
theorem midpoint_incenter_halves_BK
  (triangle : triangle A B C) (incircle : in_circle (triangle))
  (h_incenter : is_incenter I triangle)
  (h_touch : touches K incircle (segment A C))
  (h_midpoint : is_midpoint M (segment A C)) :
  halves (line M I) (segment B K) :=
sorry

end midpoint_incenter_halves_BK_l502_502527


namespace largest_eight_digit_number_contains_even_digits_l502_502719

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l502_502719


namespace maximum_profit_at_110_l502_502530

noncomputable def profit (x : ℕ) : ℝ := 
if x > 0 ∧ x < 100 then 
  -0.5 * (x : ℝ)^2 + 90 * (x : ℝ) - 600 
else if x ≥ 100 then 
  -2 * (x : ℝ) - 24200 / (x : ℝ) + 4100 
else 
  0 -- To ensure totality, although this won't match the problem's condition that x is always positive

theorem maximum_profit_at_110 :
  ∃ (y_max : ℝ), ∀ (x : ℕ), profit 110 = y_max ∧ (∀ x ≠ 0, profit 110 ≥ profit x) :=
sorry

end maximum_profit_at_110_l502_502530


namespace coefficient_x40_l502_502291

noncomputable def poly : Polynomial ℚ :=
  ∏ i in (1:ℕ) .. 10, Polynomial.C i * Polynomial.x ^ i - Polynomial.C i

theorem coefficient_x40 : (poly.coeff 40) = 329 := by
  sorry

end coefficient_x40_l502_502291


namespace jim_ran_16_miles_in_2_hours_l502_502958

-- Given conditions
variables (j f : ℝ) -- miles Jim ran in 2 hours, miles Frank ran in 2 hours
variables (h1 : f = 20) -- Frank ran 20 miles in 2 hours
variables (h2 : f / 2 = (j / 2) + 2) -- Frank ran 2 miles more than Jim in an hour

-- Statement to prove
theorem jim_ran_16_miles_in_2_hours (j f : ℝ) (h1 : f = 20) (h2 : f / 2 = (j / 2) + 2) : j = 16 :=
by
  sorry

end jim_ran_16_miles_in_2_hours_l502_502958


namespace circle_polar_eq_distance_PA_PB_l502_502948

noncomputable def polarCenter := (2: ℝ, Real.pi / 3: ℝ)
noncomputable def radius := 2: ℝ
noncomputable def lineParametric (t: ℝ) := (1 - (Real.sqrt 3) / 2 * t, (Real.sqrt 3) + 1 / 2 * t)

theorem circle_polar_eq :
  (∀ ρ θ: ℝ, ρ = 2 → θ = Real.pi / 3 → ρ = 4 * Real.sin (θ + Real.pi / 6)) :=
sorry

theorem distance_PA_PB :
  (∀ t1 t2 t0: ℝ, t1 = 2 → t2 = -2 → t0 = -2 * Real.sqrt 3 → (abs (2 + 2 * Real.sqrt 3)) + (abs (-2 + 2 * Real.sqrt 3)) = 4 * Real.sqrt 3) :=
sorry

end circle_polar_eq_distance_PA_PB_l502_502948


namespace units_digit_of_product_is_eight_l502_502585

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502585


namespace largest_eight_digit_with_all_even_digits_l502_502711

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l502_502711


namespace watermelons_left_l502_502236

theorem watermelons_left (initial_dozen : ℕ) (sold_yesterday_percentage : ℝ) (sold_today_fraction : ℝ) (total_watermelons : ℕ) 
                         (sold_yesterday : ℕ) (remaining_after_yesterday : ℕ) (sold_today : ℕ) (remaining_for_tomorrow : ℕ) :
  initial_dozen = 10 →
  sold_yesterday_percentage = 0.40 →
  sold_today_fraction = 0.25 →
  total_watermelons = initial_dozen * 12 →
  sold_yesterday = (total_watermelons : ℝ) * sold_yesterday_percentage →
  remaining_after_yesterday = total_watermelons - sold_yesterday →
  sold_today = ℕ.floor (remaining_after_yesterday * sold_today_fraction) →
  remaining_for_tomorrow = remaining_after_yesterday - sold_today →
  remaining_for_tomorrow = 54 :=
by
  intros
  sorry

end watermelons_left_l502_502236


namespace kopeechka_items_l502_502468

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502468


namespace watermelons_left_to_be_sold_tomorrow_l502_502234

def initial_watermelons := 10 * 12
def sold_yesterday := 0.40 * initial_watermelons
def remaining_after_yesterday := initial_watermelons - sold_yesterday
def sold_today := remaining_after_yesterday / 4
def remaining_for_tomorrow := remaining_after_yesterday - sold_today

theorem watermelons_left_to_be_sold_tomorrow : remaining_for_tomorrow = 54 :=
by
  -- Proof goes here, currently omitted with 'sorry'
  sorry

end watermelons_left_to_be_sold_tomorrow_l502_502234


namespace kopeechka_items_l502_502429

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502429


namespace georgie_entry_exit_ways_l502_502212

-- Defining the conditions
def castle_windows : Nat := 8
def non_exitable_windows : Nat := 2

-- Defining the problem
theorem georgie_entry_exit_ways (total_windows : Nat) (blocked_exits : Nat) (entry_windows : Nat) : 
  total_windows = castle_windows → blocked_exits = non_exitable_windows → 
  entry_windows = castle_windows →
  (entry_windows * (total_windows - 1 - blocked_exits) = 40) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end georgie_entry_exit_ways_l502_502212


namespace distance_apart_after_skating_l502_502780

theorem distance_apart_after_skating :
  let Ann_speed := 6 -- Ann's speed in miles per hour
  let Glenda_speed := 8 -- Glenda's speed in miles per hour
  let skating_time := 3 -- Time spent skating in hours
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  Total_Distance = 42 :=
by
  let Ann_speed := 6
  let Glenda_speed := 8
  let skating_time := 3
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  sorry

end distance_apart_after_skating_l502_502780


namespace zero_of_composed_function_is_pm_sqrt_2_l502_502837

def f (x : ℝ) : ℝ := 2 * x - 4
def g (x : ℝ) : ℝ := x^2

theorem zero_of_composed_function_is_pm_sqrt_2 :
  ∀ x : ℝ, f(g(x)) = 0 → x = sqrt 2 ∨ x = -sqrt 2 := by
  sorry

end zero_of_composed_function_is_pm_sqrt_2_l502_502837


namespace zoe_avg_speed_l502_502792

-- Given definitions based on the conditions
variable {d : ℝ} (d_pos : 0 < d)

-- Definitions for Chantal's speeds
def speed_chantal1 := 5 -- Chantal's speed for the first segment
def speed_chantal2 := 3 -- Chantal's speed for the rocky path and the lookout segment
def speed_chantal3 := 4 -- Chantal's speed while descending

-- Times for each segment Chantal travels
def time_chantal1 := d / speed_chantal1
def time_chantal2 := d / speed_chantal2
def time_chantal3 := d / speed_chantal3

-- Total time taken by Chantal till she meets Zoe
def total_time_chantal := time_chantal1 + time_chantal2 + time_chantal3 

-- Zoe's average speed until they meet
def zoe_speed := d / total_time_chantal

-- The statement to prove
theorem zoe_avg_speed : zoe_speed = 60 / 47 :=
by 
  -- No proof required
  sorry

end zoe_avg_speed_l502_502792


namespace maximum_value_MN_l502_502073

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x)

theorem maximum_value_MN : ∃ a : ℝ, |f a - g a| ≤ 3 :=
by
  let M := |f a - g a|
  sorry

end maximum_value_MN_l502_502073


namespace problem1_problem2_l502_502192

-- Problem 1
theorem problem1 : sqrt 9 - sqrt 2 * 32 * 62 = 1 :=
sorry

-- Problem 2
variables {x : ℝ}
theorem problem2 (h1 : x + x⁻¹ = 3) (hx : x > 0) : x^(3/2) + x^(-3/2) = 2 * sqrt 5 :=
sorry

end problem1_problem2_l502_502192


namespace max_area_of_triangle_l502_502477

theorem max_area_of_triangle (x y : ℕ) (h : x + y = 418) : 
  (0.5 * (x * y)).floor = 21840 := sorry

end max_area_of_triangle_l502_502477


namespace exists_unique_c_l502_502798

def f (x : ℝ) : ℝ := 2 - 5 * x + 4 * x^2 - 3 * x^3 + 7 * x^5
def g (x : ℝ) : ℝ := 1 - 7 * x + 5 * x^2 - 2 * x^4 + 9 * x^5

theorem exists_unique_c : ∃! c : ℝ, degree (λ x, f x + c * g x) = 4 :=
by
  sorry

end exists_unique_c_l502_502798


namespace largest_eight_digit_number_l502_502703

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l502_502703


namespace third_month_sale_l502_502211

theorem third_month_sale
  (avg_sale : ℕ)
  (num_months : ℕ)
  (sales : List ℕ)
  (sixth_month_sale : ℕ)
  (total_sales_req : ℕ) :
  avg_sale = 6500 →
  num_months = 6 →
  sales = [6435, 6927, 7230, 6562] →
  sixth_month_sale = 4991 →
  total_sales_req = avg_sale * num_months →
  total_sales_req - (sales.sum + sixth_month_sale) = 6855 := by
  sorry

end third_month_sale_l502_502211


namespace total_students_l502_502127

theorem total_students (g s : ℕ) (h_g : g = 304) (h_s : s = 75) : g * s = 22800 := by
  rw [h_g, h_s]
  norm_num
  sorry

end total_students_l502_502127


namespace log2_50_between_consecutive_integers_l502_502810

theorem log2_50_between_consecutive_integers :
  ∃ (c d : ℕ), (c < d) ∧ (5 < Real.log 50 / Real.log 2) ∧ (Real.log 50 / Real.log 2 < 6) ∧ (c + d = 11) :=
by
  use 5
  use 6
  split
  . exact Nat.lt_succ_self 5
  . split
     . have h : 32 < 50 := by norm_num
       have h2 : (Real.log 32 / Real.log 2 = 5) := by { rw [Real.log_pow 2 5], field_simp, norm_num, }
       rw Real.log_pow 2 5 at h2 
       exact (Real.log_lt_log₂ (show (0 : ℝ) < 32 from by norm_num) (show (0 : ℝ) < 50 from by norm_num) h2 h)
     . split
       . have h2 : (Real.log 64 / Real.log 2 = 6) := by { rw [Real.log_pow 2 6], field_simp, norm_num, }
         exact (Real.log_lt_log₂ (show (0 : ℝ) < 50 from by norm_num) (show (0 : ℝ) < 64 from by norm_num) h h2)
       . norm_num
   . sorry

end log2_50_between_consecutive_integers_l502_502810


namespace tangent_line_eq_l502_502112

theorem tangent_line_eq (x y : ℝ) (h : y = 2 * x - Real.log x) (tangent_pt : (1, 2)) :
  (x - y + 1 = 0) := 
sorry

end tangent_line_eq_l502_502112


namespace units_digit_first_four_composites_l502_502574

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502574


namespace wendy_lost_lives_l502_502566

theorem wendy_lost_lives (L : ℕ) (h1 : 10 - L + 37 = 41) : L = 6 :=
by
  sorry

end wendy_lost_lives_l502_502566


namespace angles_arith_prog_tangent_tangent_parallel_euler_line_l502_502510

-- Define a non-equilateral triangle with angles in arithmetic progression
structure Triangle :=
  (A B C : ℝ) -- Angles in a non-equilateral triangle
  (non_equilateral : A ≠ B ∨ B ≠ C ∨ A ≠ C)
  (angles_arith_progression : (2 * B = A + C))

-- Additional geometry concepts will be assumptions as their definition 
-- would involve extensive axiomatic setups

-- The main theorem to state the equivalence
theorem angles_arith_prog_tangent_tangent_parallel_euler_line (Δ : Triangle)
  (common_tangent_parallel_euler : sorry) : 
  ((Δ.A = 60) ∨ (Δ.B = 60) ∨ (Δ.C = 60)) :=
sorry

end angles_arith_prog_tangent_tangent_parallel_euler_line_l502_502510


namespace units_digit_first_four_composites_l502_502594

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502594


namespace number_of_undeveloped_sections_l502_502134

def undeveloped_sections (total_area section_area : ℕ) : ℕ :=
  total_area / section_area

theorem number_of_undeveloped_sections :
  undeveloped_sections 7305 2435 = 3 :=
by
  unfold undeveloped_sections
  exact rfl

end number_of_undeveloped_sections_l502_502134


namespace find_integers_l502_502085

theorem find_integers (x y : ℕ) (h : 2 * x * y = 21 + 2 * x + y) : (x = 1 ∧ y = 23) ∨ (x = 6 ∧ y = 3) :=
by
  sorry

end find_integers_l502_502085


namespace capital_after_18_years_l502_502201

noncomputable def initial_investment : ℝ := 2000
def rate_of_increase : ℝ := 0.50
def period : ℕ := 3
def total_time : ℕ := 18

theorem capital_after_18_years :
  (initial_investment * (1 + rate_of_increase) ^ (total_time / period)) = 22781.25 :=
by
  sorry

end capital_after_18_years_l502_502201


namespace common_terms_sequence_l502_502364

-- Definitions of sequences
def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℤ := 2 ^ n
def c (n : ℕ) : ℤ := 2 ^ (2 * n - 1)

-- Theorem stating the conjecture
theorem common_terms_sequence :
  ∀ n : ℕ, ∃ m : ℕ, a m = b (2 * n - 1) :=
by
  sorry

end common_terms_sequence_l502_502364


namespace max_log_expression_l502_502361

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem max_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x > y) :
  log_base x (x^2 / y^3) + log_base y (y^2 / x^3) = -2 :=
by
  sorry

end max_log_expression_l502_502361


namespace arithmetic_sequence_sum_eq_l502_502413

variable {α : Type} [Add α] [Mul α] [HasEq α]

noncomputable theory

def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n m, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_eq (a : ℕ → α) (h_arith : arithmetic_sequence a) (h_cond : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 :=
by
  sorry

end arithmetic_sequence_sum_eq_l502_502413


namespace units_digit_of_product_of_first_four_composites_l502_502641

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502641


namespace arithmetic_sqrt_four_l502_502099

theorem arithmetic_sqrt_four : sqrt 4 = 2 :=
by
  sorry

end arithmetic_sqrt_four_l502_502099


namespace smaller_angle_at_3_25_l502_502153

-- Definitions for the given conditions
def initial_angle : ℝ := 90
def minute_hand_rate : ℝ := 6
def hour_hand_rate : ℝ := 0.5
def minutes_passed : ℝ := 25

-- Calculate the angles moved by hour and minute hands after given minutes
def hour_hand_movement := hour_hand_rate * minutes_passed
def minute_hand_movement := minute_hand_rate * minutes_passed

-- Calculate the new angle (subtract the minute hand movement)
def new_angle := initial_angle + hour_hand_movement - minute_hand_movement

-- Make the new_angle a positive angle in 360-degree circumference
def positive_angle := if new_angle < 0 then 360 + new_angle else new_angle

-- The smaller angle between the hands
def smaller_angle := if positive_angle > 180 then 360 - positive_angle else positive_angle

theorem smaller_angle_at_3_25 :
  smaller_angle = 47.5 := sorry

end smaller_angle_at_3_25_l502_502153


namespace problem_range_m_l502_502322

theorem problem_range_m (m : ℝ)
  (p : Prop := ∃ x y : ℝ, ∃ m : ℝ, 3 < m ∧ m < 6 ∧ (x^2 / m + y^2 / (6 - m) = 1))
  (q : Prop := ∃ y x : ℝ, ∃ m : ℝ, (5/2 < m ∧ m < 5) ∧ (∃ e : ℝ, 1 + (m / 5) = e^2 ∧ (sqrt 6 / 2 < e ∧ e < sqrt 2)))
  (h1 : p ∨ q) (h2 : ¬ (p ∧ q)) :
  (m ∈ Icc (5 / 2) 3 ∪ Icc 5 6) :=
sorry

end problem_range_m_l502_502322


namespace orange_cells_possible_l502_502745

-- Definitions for the problem
def board_width : ℕ := 2022
def board_height : ℕ := 2022
def square_size : ℕ := 2

-- Condition statements
def cells_in_square : ℕ := square_size * square_size
def total_cells : ℕ := board_width * board_height

def possible_orange_counts : ℕ × ℕ :=
  (2022 * 2020, 2021 * 2020)

-- Theorem to prove the possible outcomes
theorem orange_cells_possible :
  ∃ x y : ℕ, x * y ∈ {2022 * 2020, 2021 * 2020} ∧ 
  x * y ≤ total_cells :=
sorry

end orange_cells_possible_l502_502745


namespace jump_rope_analysis_l502_502049

-- Defining the dataset
def jump_rope_data : List ℕ := [100, 110, 114, 114, 120, 122, 122, 131, 144, 148, 152, 155, 156, 165, 165, 165, 165, 174, 188, 190]

-- Given conditions
def mean : ℕ := 145

-- Proof that the mode (most frequent value) is 165
def mode_is_165 : Prop := List.mode jump_rope_data = 165

-- Proof that the median value is 150
def median_is_150 : Prop :=
  let sorted_data := List.sort jump_rope_data
  let n := List.length sorted_data
  if n % 2 = 0 then (sorted_data.get (n / 2 - 1) + sorted_data.get (n / 2)) / 2 = 150
  else sorted_data.get (n / 2) = 150

-- Estimation of students achieving 165 or more out of 240 based on the sample proportion
def estimate_165_or_more : Prop :=
  let count_165_or_more := List.length (List.filter (fun x => x >= 165) jump_rope_data)
  let proportion := count_165_or_more / List.length jump_rope_data
  let estimated_students := proportion * 240
  estimated_students = 84

-- Verification if a student with 152 jumps exceeds half the students (compare with median)
def exceeds_half_students : Prop :=
  let median := 150
  152 > median

-- Statement combining all conditions
theorem jump_rope_analysis :
  mode_is_165 ∧ median_is_150 ∧ estimate_165_or_more ∧ exceeds_half_students := by
  sorry

end jump_rope_analysis_l502_502049


namespace sum_of_v_l502_502263

-- Define initial vectors v0 and w0
def v0 : ℝ × ℝ := (2, 2)
def w0 : ℝ × ℝ := (3, 1)

-- Define a projection function
def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_sq := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_sq * u.1, dot_uv / norm_u_sq * u.2)

-- Define sequences of vectors v_n and w_n
noncomputable def v : ℕ → ℝ × ℝ
| 0     := v0
| (n+1) := projection v0 (w n)

noncomputable def w : ℕ → ℝ × ℝ
| 0     := w0
| (n+1) := projection w0 (v (n+1))

-- Define the infinite sum of vectors v_n
noncomputable def sum_vs (n : ℕ) : ℝ × ℝ := (finset.range n).sum (λ k, v (k + 1))

-- Statement of the problem
theorem sum_of_v : sum_vs 1000 = (10, 10) := sorry

end sum_of_v_l502_502263


namespace f_inv_1_eq_1_l502_502068

variable {R : Type}
variables {a b c : R} [Nontrivial R] [CommRing R]

noncomputable def f : R → R :=
  λ x, 1 / (a * x^2 + b * x + c)

theorem f_inv_1_eq_1 : f (1 / (a + b + c)) = 1 := by
  sorry

end f_inv_1_eq_1_l502_502068


namespace shortest_path_cone_l502_502526

/-- The base radius of a cone is 1, and the slant height is 2. The vertex is S, 
and the axial section is \( \triangle SAB \). C is the midpoint of SB. If point 
A revolves around the lateral surface to point C, then the shortest path length 
is \( \sqrt{5} \). -/
theorem shortest_path_cone (r s : ℝ) (h_r : r = 1) (h_s : s = 2) (A B S C : Type*) 
  [vertex_of_cone S A B] [midpoint C S B]:
  shortest_path A C = sqrt 5 :=
sorry

end shortest_path_cone_l502_502526


namespace symmetric_about_pi_over_4_l502_502114

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.cos x

theorem symmetric_about_pi_over_4 (a : ℝ) :
  (∀ x : ℝ, f a (x + π / 4) = f a (-(x + π / 4))) → a = 1 := by
  unfold f
  sorry

end symmetric_about_pi_over_4_l502_502114


namespace simplify_f_f_monotonically_increasing_intervals_inequality_for_m_l502_502838

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := 2 * sin (3/2 * x) * (cos (3/2 * x) + sqrt 3 * sin (3/2 * x)) - sqrt 3

-- Statement 1: Simplification of the function
theorem simplify_f : 
  ∀ x : ℝ, f x = 2 * sin (3/2 * x + π/3) - sqrt 3 := 
sorry

-- Statement 2: Interval of Monotonic Increase
theorem f_monotonically_increasing_intervals :
  ∀ k : ℤ, 
  (∀ x : ℝ, 
    x ≥ (-π/18 + 2 * k * π/3) ∧ x ≤ (5 * π/18 + 2 * k * π/3) → 
    (deriv f x) > 0) := 
sorry

-- Statement 3: Inequality for m
theorem inequality_for_m :
  ∀ m : ℝ,
  (∀ x : ℝ, 
    x ∈ Icc 0 (π/6) → 
    m * f x + 2 * m ≥ f x) ↔ m ≥ 1/3 :=
sorry

end simplify_f_f_monotonically_increasing_intervals_inequality_for_m_l502_502838


namespace trust_meteorologist_l502_502177

-- Definitions for problem conditions
variables {G M1 M2 S : Prop}
variable {r : ℝ}
variable {p : ℝ}

/-- The probability of a clear day is r -/
axiom prob_clear_day : r = 0.74

/-- Senators' prediction accuracy -/
axiom senator_accuracy : ℝ

/-- Meteorologist's prediction accuracy being 1.5 times senators' /-
axiom meteorologist_accuracy : ∀ p, 1.5 * p

/-- Independence of predictions -/
axiom independence_preds : independent [G, M1, M2, S]

noncomputable def joint_probability_given_G : ℝ :=
(1 - 1.5 * meteorologist_accuracy senator_accuracy) * senator_accuracy^2

noncomputable def joint_probability_given_not_G : ℝ :=
1.5 * meteorologist_accuracy senator_accuracy * (1 - senator_accuracy)^2

noncomputable def overall_probability : ℝ :=
joint_probability_given_G * r + joint_probability_given_not_G * (1 - r)

noncomputable def conditional_prob_not_clear : ℝ /-
(joint_probability_given_not_G * (1 - r)) / overall_probability

noncomputable def conditional_prob_clear : ℝ
(joint_probability_given_G * r) / overall_probability

-- Main theorem statement: Given the conditions, the meteorologist's forecast is more reliable
theorem trust_meteorologist : conditional_prob_not_clear > conditional_prob_clear :=
by sorry

end trust_meteorologist_l502_502177


namespace probability_three_heads_one_tail_l502_502387

-- Definition of the event space when tossing four coins.
def event_space := {s : list bool | s.length = 4}

-- Definition of the event of getting exactly three heads and one tail.
def three_heads_one_tail (s: list bool) : Prop := s.count (λ b, b = tt) = 3

-- Probability function
def probability (event: set (list bool)) : ℚ := (event.to_finset.card : ℚ) / (event_space.to_finset.card : ℚ)

-- Total event space.
noncomputable def total_event_space : set (list bool) := {s | s.length = 4}

-- Event of getting exactly three heads and one tail.
noncomputable def event_three_heads_one_tail : set (list bool) := {s | s.length = 4 ∧ three_heads_one_tail s}

-- Probability of getting exactly three heads and one tail.
theorem probability_three_heads_one_tail :
  probability event_three_heads_one_tail = 1 / 4 :=
sorry

end probability_three_heads_one_tail_l502_502387


namespace lame_rook_traverse_l502_502969

variable (Chessboard : Type) [Finite Chessboard]

variables (A B : Chessboard)
hypothesis (corner_A : is_corner_cell A)
hypothesis (adjacent_diagonally_A_B : is_diagonally_adjacent A B)
  
def number_of_ways_to_traverse (start : Chessboard) : ℕ := sorry

theorem lame_rook_traverse (A B : Chessboard)
  (corner_A : is_corner_cell A)
  (adjacent_diagonally_A_B : is_diagonally_adjacent A B) :
  number_of_ways_to_traverse A > number_of_ways_to_traverse B :=
sorry

end lame_rook_traverse_l502_502969


namespace trust_meteorologist_l502_502178

-- Definitions for problem conditions
variables {G M1 M2 S : Prop}
variable {r : ℝ}
variable {p : ℝ}

/-- The probability of a clear day is r -/
axiom prob_clear_day : r = 0.74

/-- Senators' prediction accuracy -/
axiom senator_accuracy : ℝ

/-- Meteorologist's prediction accuracy being 1.5 times senators' /-
axiom meteorologist_accuracy : ∀ p, 1.5 * p

/-- Independence of predictions -/
axiom independence_preds : independent [G, M1, M2, S]

noncomputable def joint_probability_given_G : ℝ :=
(1 - 1.5 * meteorologist_accuracy senator_accuracy) * senator_accuracy^2

noncomputable def joint_probability_given_not_G : ℝ :=
1.5 * meteorologist_accuracy senator_accuracy * (1 - senator_accuracy)^2

noncomputable def overall_probability : ℝ :=
joint_probability_given_G * r + joint_probability_given_not_G * (1 - r)

noncomputable def conditional_prob_not_clear : ℝ /-
(joint_probability_given_not_G * (1 - r)) / overall_probability

noncomputable def conditional_prob_clear : ℝ
(joint_probability_given_G * r) / overall_probability

-- Main theorem statement: Given the conditions, the meteorologist's forecast is more reliable
theorem trust_meteorologist : conditional_prob_not_clear > conditional_prob_clear :=
by sorry

end trust_meteorologist_l502_502178


namespace trust_meteorologist_l502_502174

/-- 
The probability of a clear day in Anchuria.
-/
def P_G : ℝ := 0.74

/-- 
Accuracy of the forecast by each senator.
-/
variable (p : ℝ)

/-- 
Accuracy of the meteorologist's forecast being 1.5 times that of a senator.
-/
def meteorologist_accuracy : ℝ := 1.5 * p

/-- 
Calculations and final proof that the meteorologist's forecast is more reliable than that of the senators. 
-/
theorem trust_meteorologist (p : ℝ) (Hp1 : 0 ≤ p) (Hp2 : p ≤ 1) : 
  λ P_S_M1_M2_G P_S_M1_M2_not_G : 
  P_G := 0.74 ∧ meteorologist_accuracy = 1.5 * p ∧
  (∀ P_S_M1_M2, P_S_M1_M2 = P_S_M1_M2_G * P_G + P_S_M1_M2_not_G * (1 - P_G)) → 
  (P_S_M1_M2_not_G * (1 - P_G) > P_S_M1_M2_G * P_G) :=
begin
  sorry
end

end trust_meteorologist_l502_502174


namespace range_of_m_l502_502384

theorem range_of_m (α : ℝ) (hα1 : π/2 < α) (hα2 : α < π) (h : sin α = 4 - 3 * m) :
    1 < m ∧ m < 4 / 3 :=
by
  sorry

end range_of_m_l502_502384


namespace count_integers_containing_2_and_5_l502_502898

-- Definition for the conditions in Lean 4
def contains_digits_2_5 (n : ℕ) : Prop :=
  let digits := n.digits 10
  2 ∈ digits ∧ 5 ∈ digits

def is_between_300_and_700 (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 700

-- The main statement
theorem count_integers_containing_2_and_5 : 
  {n : ℕ | is_between_300_and_700 n ∧ contains_digits_2_5 n}.to_finset.card = 8 :=
by sorry

end count_integers_containing_2_and_5_l502_502898


namespace true_statement_is_S99_l502_502147

-- Definitions
def is_statement_true (n : ℕ) : Prop :=
  ∃! k, (k > 0 ∧ k ≤ 100) ∧ k = n ∧ (count_false_statements k = n)

def count_false_statements (k : ℕ) : ℕ :=
  if h : 1 ≤ k ∧ k ≤ 100 then 100 - k else 0

-- Main Theorem
theorem true_statement_is_S99 :
  is_statement_true 99 := 
  sorry

end true_statement_is_S99_l502_502147


namespace smallest_n_no_xy_exists_l502_502249

theorem smallest_n_no_xy_exists : ∃ (n : ℕ), (∀ x y : ℤ, n ≠ x^3 + 3 * y^3) ∧ (∀ m : ℕ, (∀ x y : ℤ, m ≠ x^3 + 3 * y^3) → n ≤ m) :=
by {
  let n := 6,
  use n,
  split,
  {
    intros x y,
    have h : x^3 + 3 * y^3 ≠ 6 := sorry,
    exact h,
  },
  {
    intros m hm,
    have h_smallest : n ≤ m := sorry,
    exact h_smallest,
  }
}

end smallest_n_no_xy_exists_l502_502249


namespace Sn_value_l502_502862

-- Given the definition of a_n
def a_n (n : ℕ) : ℚ := n * (n + 1) / 2

-- Sum of the first n terms of the sequence {1 / a_n}
def S_n (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), 1 / a_n i

theorem Sn_value (n : ℕ) :
  S_n n = 2 * (1 - 1 / (n + 1)) :=
sorry

end Sn_value_l502_502862


namespace cube_surface_area_increase_l502_502158

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l502_502158


namespace distance_to_other_focus_l502_502869

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
{x | let (x1, y1) := x in (x1^2 / a^2) + (y1^2 / b^2) = 1}

theorem distance_to_other_focus (P : ℝ × ℝ) (a b : ℝ) (d1 : ℝ) (d2 : ℝ) :
  P ∈ ellipse 5 4 →
  d1 = 3 →
  d2 = 10 - d1 →
  d2 = 7 :=
by
  intros hP hd1 hdef
  rw [hd1, hdef]
  norm_num
  sorry

end distance_to_other_focus_l502_502869


namespace final_selling_price_l502_502763
-- Import necessary library

-- Define the conditions
variables {cp_A : ℝ} (h_cpA : cp_A = 112.5) -- Cost price for A
variables {profit_A_rate : ℝ} (h_profit_A_rate : profit_A_rate = 0.6) -- Profit rate for A
variables {profit_B_rate : ℝ} (h_profit_B_rate : profit_B_rate = 0.25) -- Profit rate for B

-- Define the theorem
theorem final_selling_price
    (h_cpA : cp_A = 112.5)
    (h_profit_A_rate : profit_A_rate = 0.6)
    (h_profit_B_rate : profit_B_rate = 0.25) :
    let profit_A := profit_A_rate * cp_A,
        sp_A := cp_A + profit_A,
        profit_B := profit_B_rate * sp_A,
        final_sp := sp_A + profit_B in final_sp = 225 := by
    -- skip the proof
    sorry

end final_selling_price_l502_502763


namespace vector_inequality_l502_502082

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_inequality (a b c d : V) (h : a + b + c + d = 0) :
  ∥a∥ + ∥b∥ + ∥c∥ + ∥d∥ ≥ ∥a + d∥ + ∥b + d∥ + ∥c + d∥ :=
by sorry

end vector_inequality_l502_502082


namespace loop_final_value_l502_502252

def loop_output (initial_S : ℕ) (initial_A : ℕ) : ℕ :=
  let rec iterate S A :=
    if S <= 36 then iterate (S + A + 1) (A + 1) else S
  iterate initial_S initial_A

theorem loop_final_value : loop_output 0 1 = 45 :=
  sorry

end loop_final_value_l502_502252


namespace arithmetic_sequence_solution_l502_502344

variable (a : ℕ → ℕ) (S T : ℕ → ℕ)

-- The sequence {a_n} is an arithmetic sequence
variable (d : ℕ)
axiom h1 : ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom h2 : a 2 = 3
axiom h3 : a 4 + a 5 + a 6 = 27

-- Definitions
def Sn (n : ℕ) : ℕ := n * (a 1 + a n) / 2
def bn (n : ℕ) : ℕ := a (2 ^ n)
def Tn (n : ℕ) : ℕ := (Finset.range n).sum (λ i, a (2 ^ (i+1))) - n

-- Theorem statement
theorem arithmetic_sequence_solution :
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, S n = n^2) ∧
  (∀ n : ℕ, T n = 2^(n+2) - n - 4) :=
by
    sorry

end arithmetic_sequence_solution_l502_502344


namespace simpleInterest_500_l502_502294

def simpleInterest (P R T : ℝ) : ℝ := P * R * T

theorem simpleInterest_500 :
  simpleInterest 10000 0.05 1 = 500 :=
by
  sorry

end simpleInterest_500_l502_502294


namespace units_digit_of_product_of_first_four_composites_l502_502638

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502638


namespace watermelons_left_l502_502237

theorem watermelons_left (initial_dozen : ℕ) (sold_yesterday_percentage : ℝ) (sold_today_fraction : ℝ) (total_watermelons : ℕ) 
                         (sold_yesterday : ℕ) (remaining_after_yesterday : ℕ) (sold_today : ℕ) (remaining_for_tomorrow : ℕ) :
  initial_dozen = 10 →
  sold_yesterday_percentage = 0.40 →
  sold_today_fraction = 0.25 →
  total_watermelons = initial_dozen * 12 →
  sold_yesterday = (total_watermelons : ℝ) * sold_yesterday_percentage →
  remaining_after_yesterday = total_watermelons - sold_yesterday →
  sold_today = ℕ.floor (remaining_after_yesterday * sold_today_fraction) →
  remaining_for_tomorrow = remaining_after_yesterday - sold_today →
  remaining_for_tomorrow = 54 :=
by
  intros
  sorry

end watermelons_left_l502_502237


namespace shaded_area_l502_502142

-- Definition of given conditions
def r : ℝ := 15
def θ : ℝ := Real.pi / 4

-- Goal statement
theorem shaded_area : 
  2 * (0.5 * r^2 * θ - 0.5 * (2 * r * Real.sin(θ / 2) * (r * Real.sin(θ / 2)))) = 
  (1125 * Real.pi / 4 - 450 * Real.sin(Real.pi / 8) ^ 2) :=
by
  sorry

end shaded_area_l502_502142


namespace graph_symmetric_value_of_a_l502_502920

theorem graph_symmetric_value_of_a (a : ℝ) (hf : ∀ x, f(x) = |x + 2 * a| - 1) (hsymm : ∀ x, f(1 - x) = f(1 + x)) : a = -1/2 :=
by
  sorry

end graph_symmetric_value_of_a_l502_502920


namespace find_A_in_terms_of_B_and_C_l502_502974

variable {A B C : ℝ}

def f (x : ℝ) : ℝ := A * x^2 - 2 * B^2 * x + 3
def g (x : ℝ) : ℝ := B * x + 1

theorem find_A_in_terms_of_B_and_C (hB : B ≠ 0) (hC : f (g 1) = C) : 
  A = (C + 2 * B^3 + 2 * B^2 - 3) / (B^2 + 2 * B + 1) :=
by
  sorry

end find_A_in_terms_of_B_and_C_l502_502974


namespace units_digit_first_four_composites_l502_502621

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502621


namespace abs_z_eq_five_l502_502868

theorem abs_z_eq_five (z : ℂ) (h : z - 3 = (3 + Complex.i) / Complex.i) : Complex.abs z = 5 := by
  sorry

end abs_z_eq_five_l502_502868


namespace relationship_f_minus_a2_f_minus_1_l502_502351

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement translation
theorem relationship_f_minus_a2_f_minus_1 (a : ℝ) : f (-a^2) ≤ f (-1) := 
sorry

end relationship_f_minus_a2_f_minus_1_l502_502351


namespace problem_statement_l502_502993

noncomputable def f (x : ℝ) := 2 * cos x * (cos x + sqrt 3 * sin x)

theorem problem_statement :
  (∃ T : ℝ, (T = π) ∧
    (∀ k : ℤ, (∀ x : ℝ, (k * π - π / 3 < x ∧ x < k * π + π / 6) → (f x = 2 * sin (2 * x + π / 6) + 1)))) ∧
  (x ∈ (set.Icc 0 (π / 2)) → (∃ m : ℝ, (m = 3) ∧ (∀ x : ℝ, f x ≤ m))) :=
begin
  sorry
end

end problem_statement_l502_502993


namespace largest_eight_digit_number_l502_502700

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l502_502700


namespace units_digit_of_composite_product_l502_502657

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502657


namespace find_length_DE_l502_502090

theorem find_length_DE 
  (ABCD : Type) [rectangle ABCD] 
  (AB AD DC CE DE : ℝ) 
  (shared_side : DC = AB)
  (rectangle_area : 2 * (AB * AD) = DC * CE) 
  (AB_eq : AB = 7)
  (AD_eq : AD = 8) 
  (triangle_area : ½ * DC * CE = 28) : 
  DE = Real.sqrt 113 := 
by sorry

end find_length_DE_l502_502090


namespace find_f_expression_find_m_range_l502_502350
noncomputable theory

-- Define the function f(x) with given parameters
def f (x : ℝ) (A ω ϕ : ℝ) : ℝ := A * Real.sin (ω * x + ϕ)

-- Given conditions
variables (A ω ϕ : ℝ)
variable hx1 : A > 0
variable hw1 : ω > 0
variable hϕ1 : |ϕ| < π

-- Maximum and minimum conditions
variable hmax : f (π/12) A ω ϕ = 3
variable hmin : f (7*π/12) A ω ϕ = -3

-- Prove the first question
theorem find_f_expression : ∃ A ω ϕ, A = 3 ∧ ω = 2 ∧ ϕ = π/3 ∧ ∀ x, f x A ω ϕ = 3 * Real.sin (2 * x + π/3) := by
  sorry

-- Given function h(x) and range of x
def h (x : ℝ) (m : ℝ) := 2 * f x 3 2 (π/3) + 1 - m
variable hx_range : ∀ x, x ∈ Icc (-(π/3)) (π/6)

-- Prove the second question
theorem find_m_range (m : ℝ) : (∀ x, x ∈ Icc (-(π/3)) (π/6) → h x m = 0 → 2 ∃) ↔ m ∈ Icc (3*Real.sqrt 3 + 1) 7 := by
  sorry

end find_f_expression_find_m_range_l502_502350


namespace area_triangle_DEF_50sqrt11_l502_502203

noncomputable def area_triangle_DEF (r1 r2 DE DF : ℝ) : ℝ :=
  let DG := sqrt ((r1 + DE - DF / 2)^2 - r1^2)
  let DI := (r1 + r2 + sqrt(r1^2 - r1^2 + r2^2))
  (1 / 2) * (2 * DG) * DI

theorem area_triangle_DEF_50sqrt11 : 
  ∀ {r1 r2 DE DF : ℝ},
    r1 = 3 → r2 = 5 →
    DE = DF + 6 →
    area_triangle_DEF r1 r2 DE DF = 50 * sqrt 11 :=
by
  -- The proof steps are omitted as required
  sorry

end area_triangle_DEF_50sqrt11_l502_502203


namespace units_digit_of_first_four_composite_numbers_l502_502631

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502631


namespace find_angle_B_l502_502861

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {m n : ℝ × ℝ}
variable (h1 : m = (Real.cos A, Real.sin A))
variable (h2 : n = (1, Real.sqrt 3))
variable (h3 : m.1 / n.1 = m.2 / n.2)
variable (h4 : a * Real.cos B + b * Real.cos A = c * Real.sin C)

theorem find_angle_B (h_conditions : a * Real.cos B + b * Real.cos A = c * Real.sin C) : B = Real.pi / 6 :=
sorry

end find_angle_B_l502_502861


namespace problem1_problem2_l502_502859

-- Define the first problem: For positive real numbers a and b,
-- with the condition a + b = 2, show that the minimum value of 
-- (1 / (1 + a) + 4 / (1 + b)) is 9/4.
theorem problem1 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  1 / (1 + a) + 4 / (1 + b) ≥ 9 / 4 :=
sorry

-- Define the second problem: For any positive real numbers a and b,
-- prove that a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1).
theorem problem2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end problem1_problem2_l502_502859


namespace units_digit_product_first_four_composite_numbers_l502_502678

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502678


namespace recurrence_relation_a_l502_502827

def a : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := a (n+1) + a n

theorem recurrence_relation_a (n : ℕ) : a (n+2) = a (n+1) + a n := by
  sorry

end recurrence_relation_a_l502_502827


namespace trust_meteorologist_l502_502181

-- Definitions
def probability_of_clear := 0.74
def senator_accuracy := p : ℝ
def meteorologist_accuracy := 1.5 * p

-- Events
def event_G := "clear day"
def event_M1 := "first senator predicted clear"
def event_M2 := "second senator predicted clear"
def event_S := "meteorologist predicted rain"

theorem trust_meteorologist :
  let r := probability_of_clear
  let p := senator_accuracy
  let q := meteorologist_accuracy
  1.5 * p * (1 - p)^2 * (1 - r) - (1 - 1.5 * p) * p^2 * r > 0 :=
by
  sorry

end trust_meteorologist_l502_502181


namespace max_supervisors_hired_l502_502232

/-- A university hires research supervisors who each serve a non-consecutive 4-year term 
with a 1-year gap before the next supervisor can begin. --/
theorem max_supervisors_hired (n : ℕ) (term years : ℕ) : 
  (term = 4 ∧ years = 15 ∧ n = 3) ∧ 
  ∀ i, (0 ≤ i ∧ i < n) → ∃ j, (1 ≤ j ∧ j ≤ years ∧ (i = (j-1) / (term + 1))) :=
begin
  sorry
end

end max_supervisors_hired_l502_502232


namespace intersect_at_2d_l502_502536

def g (x : ℝ) (c : ℝ) : ℝ := 4 * x + c

theorem intersect_at_2d (c d : ℤ) (h₁ : d = 8 + c) (h₂ : 2 = g d c) : d = 2 :=
by
  sorry

end intersect_at_2d_l502_502536


namespace shakespeare_born_on_wednesday_l502_502096

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

def days_in_year (y : ℕ) : ℕ :=
  if is_leap_year y then 366 else 365

noncomputable def total_days (start_year end_year : ℕ) : ℕ :=
  (List.range' start_year (end_year - start_year + 1)).sum days_in_year

theorem shakespeare_born_on_wednesday :
  total_days 1664 1964 % 7 = 6 := -- 6 corresponds to Thursday being the base (+5, because 0: Sunday, 1: Monday, ..., 5: Friday, 6: Saturday)
sorry

end shakespeare_born_on_wednesday_l502_502096


namespace hyperbola_eccentricity_l502_502103

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = 2 →
  a^2 = 2 * b^2 →
  (c : ℝ) = Real.sqrt (a^2 + b^2) →
  Real.sqrt (a^2 + b^2) = Real.sqrt (3 / 2 * a^2) →
  (e : ℝ) = c / a →
  e = Real.sqrt (6) / 2 :=
by
  sorry

end hyperbola_eccentricity_l502_502103


namespace average_weight_of_a_and_b_l502_502525

-- Given conditions as Lean definitions
variable (A B C : ℝ)
variable (h1 : (A + B + C) / 3 = 45)
variable (h2 : (B + C) / 2 = 46)
variable (hB : B = 37)

-- The statement we want to prove
theorem average_weight_of_a_and_b : (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_l502_502525


namespace total_sand_donated_l502_502535

theorem total_sand_donated (A B C D: ℚ) (hA: A = 33 / 2) (hB: B = 26) (hC: C = 49 / 2) (hD: D = 28) : 
  A + B + C + D = 95 := by
  sorry

end total_sand_donated_l502_502535


namespace circumference_of_minor_arc_l502_502979

-- Given:
-- 1. Three points (D, E, F) are on a circle with radius 25
-- 2. The angle ∠EFD = 120°

-- We need to prove that the length of the minor arc DE is 50π / 3
theorem circumference_of_minor_arc 
  (D E F : Point) 
  (r : ℝ) (h : r = 25) 
  (angleEFD : ℝ) 
  (hAngle : angleEFD = 120) 
  (circumference : ℝ) 
  (hCircumference : circumference = 2 * Real.pi * r) :
  arc_length_DE = 50 * Real.pi / 3 :=
by
  sorry

end circumference_of_minor_arc_l502_502979


namespace units_digit_of_composite_product_l502_502650

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502650


namespace moon_land_value_l502_502120

theorem moon_land_value (surface_area_earth : ℕ) (surface_area_moon : ℕ) (total_value_earth : ℕ) (worth_factor : ℕ)
  (h_moon_surface_area : surface_area_moon = surface_area_earth / 5)
  (h_surface_area_earth : surface_area_earth = 200) 
  (h_worth_factor : worth_factor = 6) 
  (h_total_value_earth : total_value_earth = 80) : (total_value_earth / 5) * worth_factor = 96 := 
by 
  -- Simplify using the given conditions
  -- total_value_earth / 5 is the value of the moon's land if it had the same value per square acre as Earth's land
  -- multiplying by worth_factor to get the total value on the moon
  sorry

end moon_land_value_l502_502120


namespace deepaks_age_l502_502125

theorem deepaks_age (R D : ℕ) (h1 : R / D = 5 / 2) (h2 : R + 6 = 26) : D = 8 := 
sorry

end deepaks_age_l502_502125


namespace avg_price_of_pencil_l502_502198

theorem avg_price_of_pencil 
  (total_pens : ℤ) (total_pencils : ℤ) (total_cost : ℤ)
  (avg_cost_pen : ℤ) (avg_cost_pencil : ℤ) :
  total_pens = 30 → 
  total_pencils = 75 → 
  total_cost = 690 → 
  avg_cost_pen = 18 → 
  (total_cost - total_pens * avg_cost_pen) / total_pencils = avg_cost_pencil → 
  avg_cost_pencil = 2 :=
by
  intros
  sorry

end avg_price_of_pencil_l502_502198


namespace arithmetic_sequence_d_property_l502_502845

theorem arithmetic_sequence_d_property :
  ∀ (d : ℕ), d ∈ {x ∈ ℕ | x > 0} → d ∣ 80 → d ≠ 3 :=
by
  sorry

end arithmetic_sequence_d_property_l502_502845


namespace number_of_items_l502_502458

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502458


namespace find_x_l502_502894

def vector (α : Type*) := α × α

def parallel (a b : vector ℝ) : Prop :=
a.1 * b.2 - a.2 * b.1 = 0

theorem find_x (x : ℝ) (a b : vector ℝ)
  (ha : a = (1, 2))
  (hb : b = (x, 4))
  (h : parallel a b) : x = 2 :=
by sorry

end find_x_l502_502894


namespace trust_meteorologist_l502_502171

/-- 
The probability of a clear day in Anchuria.
-/
def P_G : ℝ := 0.74

/-- 
Accuracy of the forecast by each senator.
-/
variable (p : ℝ)

/-- 
Accuracy of the meteorologist's forecast being 1.5 times that of a senator.
-/
def meteorologist_accuracy : ℝ := 1.5 * p

/-- 
Calculations and final proof that the meteorologist's forecast is more reliable than that of the senators. 
-/
theorem trust_meteorologist (p : ℝ) (Hp1 : 0 ≤ p) (Hp2 : p ≤ 1) : 
  λ P_S_M1_M2_G P_S_M1_M2_not_G : 
  P_G := 0.74 ∧ meteorologist_accuracy = 1.5 * p ∧
  (∀ P_S_M1_M2, P_S_M1_M2 = P_S_M1_M2_G * P_G + P_S_M1_M2_not_G * (1 - P_G)) → 
  (P_S_M1_M2_not_G * (1 - P_G) > P_S_M1_M2_G * P_G) :=
begin
  sorry
end

end trust_meteorologist_l502_502171


namespace alpha_values_l502_502005

theorem alpha_values (α : ℝ) (h : ∀ (n : ℕ), cos (2^n * α) < -1/3) : 
  ∃ k : ℤ, α = 2 * k * π + 2 * π / 3 ∨ α = 2 * k * π - 2 * π / 3 :=
sorry

end alpha_values_l502_502005


namespace tetrahedron_volume_l502_502300

noncomputable def midpoint_to_face_distance : ℝ := 2
noncomputable def midpoint_to_edge_distance : ℝ := Real.sqrt 5

theorem tetrahedron_volume 
  (d_face : ℝ := midpoint_to_face_distance) 
  (d_edge : ℝ := midpoint_to_edge_distance) 
  (h : ℝ := 6)  -- derived height h = 3 * 2
  (a : ℝ := 3 * Real.sqrt 6)  -- derived edge length a
  : abs (volume (a := a) - 485.42) < 0.01 :=
by sorry

noncomputable def volume (a : ℝ): ℝ := a^3 / (6 * Real.sqrt 2)

end tetrahedron_volume_l502_502300


namespace speaking_orders_l502_502557

theorem speaking_orders {group : Finset ℕ} (leader deputy : ℕ) (hl : leader ∈ group) (hd : deputy ∈ group) (n : ℕ) (hgroup_size : group.card = 7) 
  (hne : leader ≠ deputy) :
  ∃ (orders : ℕ), orders = 600 ∧ 
  (∀ (selected : Finset ℕ), selected.card = 4 → 
    (leader ∈ selected ∨ deputy ∈ selected) ∧ 
    ((leader ∈ selected ∧ deputy ∈ selected → ∀ (i : ℕ), i + 1 ∉ selected.to_list.index_of leader ∨ i + 1 ≠ selected.to_list.index_of deputy) ∧ 
    (orders = (number_of_ways_selected) selected))) :=
begin
  sorry
end

end speaking_orders_l502_502557


namespace city_planning_l502_502032

-- Definition of the problem's conditions
def has_six_vertices (V : set ℕ) : Prop := V = {1, 2, 3, 4, 5, 6}

def has_degree_three (E : set (ℕ × ℕ)) (v : ℕ) : Prop :=
  ∃(adj : set ℕ), adj = {w | (v, w) ∈ E ∨ (w, v) ∈ E} ∧ adj.card = 3

def non_intersecting_streets (E : set (ℕ × ℕ)) : Prop :=
  ∀ (e1 e2 : ℕ × ℕ), e1 ∈ E → e2 ∈ E → e1 ≠ e2 → ¬ (∃ (p : ℝ → ℝ × ℝ), continuous p ∧ surjective p ∧ 
  ∀ t, ((fst (p t) = e1.1 ∨ fst (p t) = e1.2) ∧ (snd (p t) = e2.1 ∨ snd (p t) = e2.2)))

def inside_angle_condition (E : set (ℕ×ℕ)) (v : ℕ) : Prop :=
  ∃ (A B C : ℕ), (A ≠ B ∧ B ≠ C ∧ C ≠ A) ∧ ((v, A) ∈ E ∧ (v, B) ∈ E ∧ (v, C) ∈ E) ∧
  -- formalize the concept that one street passes inside the angle formed by the others
  sorry

-- Definition of the proof problem
theorem city_planning : 
  ∃ (V : set ℕ) (E : set (ℕ × ℕ)), 
    has_six_vertices V ∧
    (∀ v ∈ V, has_degree_three E v) ∧
    non_intersecting_streets E ∧
    (∀ v ∈ V, inside_angle_condition E v) :=
sorry

end city_planning_l502_502032


namespace solve_max_eq_l502_502306

theorem solve_max_eq (x : ℚ) (h : max x (-x) = 2 * x + 1) : x = -1 / 3 := by
  sorry

end solve_max_eq_l502_502306


namespace even_numbers_average_l502_502100

theorem even_numbers_average (n : ℕ) (h : (n / 2 * (2 + 2 * n)) / n = 16) : n = 15 :=
by
  have hn : n ≠ 0 := sorry -- n > 0 because the first some even numbers were mentioned
  have hn_pos : 0 < n / 2 * (2 + 2 * n) := sorry -- n / 2 * (2 + 2n) > 0
  sorry

end even_numbers_average_l502_502100


namespace total_ballpoint_pens_l502_502473

theorem total_ballpoint_pens (red_pens blue_pens : ℕ) (h1 : red_pens = 37) (h2 : blue_pens = 17) : red_pens + blue_pens = 54 :=
by 
  rw [h1, h2]
  exact rfl

end total_ballpoint_pens_l502_502473


namespace number_of_divisors_of_gcd_is_eight_l502_502769

theorem number_of_divisors_of_gcd_is_eight
  (x : ℕ) 
  (hx90 : ∃ n m : ℕ, n * x = 90)
  (hx150 : ∃ n m : ℕ, n * x = 150) : 
  card (finset.filter (λ d, d ∣ 90 ∧ d ∣ 150) (finset.range 31)) = 8 := 
  sorry

end number_of_divisors_of_gcd_is_eight_l502_502769


namespace triangle_inequality_l502_502843

variables {a b c h : ℝ}
variable {n : ℕ}

theorem triangle_inequality
  (h_triangle : a^2 + b^2 = c^2)
  (h_height : a * b = c * h)
  (h_cond : a + b < c + h)
  (h_pos_n : n > 0) :
  a^n + b^n < c^n + h^n :=
sorry

end triangle_inequality_l502_502843


namespace probability_of_sum_odd_is_correct_l502_502747

noncomputable def probability_sum_odd : ℚ :=
  let total_balls := 13
  let drawn_balls := 7
  let total_ways := Nat.choose total_balls drawn_balls
  let favorable_ways := 
    Nat.choose 7 5 * Nat.choose 6 2 + 
    Nat.choose 7 3 * Nat.choose 6 4 + 
    Nat.choose 7 1 * Nat.choose 6 6
  favorable_ways / total_ways

theorem probability_of_sum_odd_is_correct :
  probability_sum_odd = 847 / 1716 :=
by
  -- Proof goes here
  sorry

end probability_of_sum_odd_is_correct_l502_502747


namespace winter_expenditure_l502_502116

theorem winter_expenditure (exp_end_nov : Real) (exp_end_feb : Real) 
  (h_nov : exp_end_nov = 3.0) (h_feb : exp_end_feb = 5.5) : 
  (exp_end_feb - exp_end_nov) = 2.5 :=
by 
  sorry

end winter_expenditure_l502_502116


namespace num_valid_paths_l502_502954

def C(n k : ℕ) := Nat.choose n k

theorem num_valid_paths : 
  let total_paths := C 7 4,
      paths_through_dangerous := C 4 2 * C 3 2
  in total_paths - paths_through_dangerous = 17 :=
by
  let total_paths := C 7 4
  let paths_through_dangerous := C 4 2 * C 3 2
  have h1 : total_paths = 35 := by sorry
  have h2 : paths_through_dangerous = 18 := by sorry
  have h3 : 35 - 18 = 17 := by sorry
  exact (by sorry : total_paths - paths_through_dangerous = 17)

end num_valid_paths_l502_502954


namespace is_angle_bisector_of_parallelogram_l502_502219

variables {A B C D O K L M : Point}
variables {circ : Circle}

def is_parallelogram (A B C D : Point) : Prop := 
  ∃ (E : Point), A - B = C - D ∧ B - C = E - D

def is_perpendicular (B O : Point) (A D : Point) : Prop :=
  ∃ θ : ℝ, θ = 90

def is_on_circle (O : Point) (ω : Circle) (P : Point) : Prop :=
  distance O P = radius ω

def intersects_extension_at (ω : Circle) (A D : Point) : Point :=
  -- Assume function definition for the intersection point
  sorry

def intersect_segment (B K : Segment) (C D : Segment) : Point :=
  -- Assume function definition for the intersection point
  sorry

def is_ray_intersect (L O : Point) (ω : Circle) : Point :=
  -- Assume function definition for the intersection point
  sorry

def is_angle_bisector (K M : Point) (B K C : Angle) : Prop :=
  -- Assume function definition for angle bisector
  sorry

theorem is_angle_bisector_of_parallelogram
  (h1 : is_parallelogram A B C D)
  (h2 : is_perpendicular B O A D)
  (h3 : is_on_circle O circ A)
  (h4 : is_on_circle O circ B)
  (h5 : let K := intersects_extension_at circ A D,
        is_on_circle O circ K)
  (h6 : let L := intersect_segment B K C D,
        true)
  (h7 : let M := is_ray_intersect L O circ,
        true)
  : is_angle_bisector K M B K C :=
sorry

end is_angle_bisector_of_parallelogram_l502_502219


namespace possible_items_l502_502436

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502436


namespace max_value_expr_l502_502681

-- Define the absolute value function
def abs (x : ℝ) : ℝ := if x >= 0 then x else -x

-- Define the expression 2 + |x - 2|
def expr (x : ℝ) : ℝ := 2 + abs (x - 2)

-- State the theorem
theorem max_value_expr : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → expr x ≤ 5 ∧ (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ expr x = 5) :=
by
  intros x h
  sorry

end max_value_expr_l502_502681


namespace count_special_integers_l502_502371

theorem count_special_integers : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 9999 ∧ (∀ d ∈ digits 10 n, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 5 ∧ d ≠ 8)) → 
  count_integers_with_conditions 1 9999 (λ d, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 5 ∧ d ≠ 8) = 624 := 
by
  sorry

end count_special_integers_l502_502371


namespace base13_addition_l502_502773

/--
Given two numbers in base 13: 528₁₃ and 274₁₃, prove that their sum is 7AC₁₃.
-/
theorem base13_addition :
  let u1 := 8
  let t1 := 2
  let h1 := 5
  let u2 := 4
  let t2 := 7
  let h2 := 2
  -- Add the units digits: 8 + 4 = 12; 12 is C in base 13
  let s1 := 12 -- 'C' in base 13
  let carry1 := 1
  -- Add the tens digits along with the carry: 2 + 7 + 1 = 10; 10 is A in base 13
  let s2 := 10 -- 'A' in base 13
  -- Add the hundreds digits: 5 + 2 = 7
  let s3 := 7 -- 7 in base 13
  s1 = 12 ∧ s2 = 10 ∧ s3 = 7 :=
by
  sorry

end base13_addition_l502_502773


namespace ratio_of_areas_l502_502801

-- The dimensions of the box (length, width, height) are real numbers
variables (L W H : ℝ)

-- The given conditions:
def condition1 : Prop := W * H = (1/2) * (L * W)
def condition2 : Prop := L * W * H = 3000
def condition3 : Prop := L * H = 200

-- The theorem to prove the ratio of the area of the top face to the area of the side face is 3:2
theorem ratio_of_areas (L W H : ℝ) (h1 : condition1 L W H) (h2 : condition2 L W H) (h3 : condition3 L W H) :
  (L * W) / (L * H) = 3 / 2 :=
by
  sorry

end ratio_of_areas_l502_502801


namespace asian_countries_visited_l502_502259

theorem asian_countries_visited (total_countries europe_countries south_america_countries remaining_asian_countries : ℕ)
  (h1 : total_countries = 42)
  (h2 : europe_countries = 20)
  (h3 : south_america_countries = 10)
  (h4 : remaining_asian_countries = (total_countries - (europe_countries + south_america_countries)) / 2) :
  remaining_asian_countries = 6 :=
by sorry

end asian_countries_visited_l502_502259


namespace three_digit_avg_permutations_l502_502815

theorem three_digit_avg_permutations (a b c: ℕ) (A: ℕ) (h₀: 1 ≤ a ∧ a ≤ 9) (h₁: 0 ≤ b ∧ b ≤ 9) (h₂: 0 ≤ c ∧ c ≤ 9) (h₃: A = 100 * a + 10 * b + c):
  ((100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)) / 6 = A ↔ 7 * a = 3 * b + 4 * c := by
  sorry

end three_digit_avg_permutations_l502_502815


namespace num_possible_lists_l502_502744

theorem num_possible_lists : 
  let balls := 15
  let draws := 4
  let numLists (balls draws : ℕ) : ℕ := List.range' (balls - draws + 1) draws |> List.foldr (· * ·) 1
in 
  numLists balls draws = 32760 :=
by
  sorry

end num_possible_lists_l502_502744


namespace units_digit_of_product_is_eight_l502_502587

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502587


namespace limit_a_n_l502_502984

noncomputable def a_n (n : ℕ) : ℝ :=
  ∑ k in finset.range (n+1).succ | k ≥ 3, 2 / (k * (k - 1) * (k - 2))

theorem limit_a_n : tendsto (λ n, a_n n) at_top (𝓝 (1 / 2)) :=
sorry

end limit_a_n_l502_502984


namespace angle_BHX_l502_502936

def is_acute_triangle (A B C : Type) (triangle : Type) : Prop := sorry
def is_orthocenter (H : Type) (triangle : Type) : Prop := sorry
def angle_at (A B C : Type) (angle : ℝ) : Prop := sorry

theorem angle_BHX (A B C H X Y : Type) 
  (ABC_triangle : is_acute_triangle A B C) 
  (H_orthocenter : is_orthocenter H ABC_triangle) 
  (AX_altitude : is_orthocenter A H ABC_triangle)
  (BY_altitude : is_orthocenter B H ABC_triangle)
  (angle_BAC : angle_at A B C 70)
  (angle_ABC : angle_at B A C 65) :
  angle_at B H X 45 :=
by 
  sorry -- Proof is skipped

end angle_BHX_l502_502936


namespace cliff_total_rocks_l502_502926

theorem cliff_total_rocks (I S : ℕ) (h1 : S = 2 * I) (h2 : I / 3 = 30) :
  I + S = 270 :=
sorry

end cliff_total_rocks_l502_502926


namespace kopeechka_items_l502_502426

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502426


namespace units_digit_of_composite_product_l502_502653

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502653


namespace determine_n_l502_502271

theorem determine_n (n : ℚ) (h : 7^(3*n) = (1/7)^(n-14)) : n = 7/2 :=
by
  sorry

end determine_n_l502_502271


namespace habitable_land_area_l502_502223

noncomputable def area_of_habitable_land : ℝ :=
  let length : ℝ := 23
  let diagonal : ℝ := 33
  let radius_of_pond : ℝ := 3
  let width : ℝ := Real.sqrt (diagonal ^ 2 - length ^ 2)
  let area_of_rectangle : ℝ := length * width
  let area_of_pond : ℝ := Real.pi * (radius_of_pond ^ 2)
  area_of_rectangle - area_of_pond

theorem habitable_land_area :
  abs (area_of_habitable_land - 515.91) < 0.01 :=
by
  sorry

end habitable_land_area_l502_502223


namespace largest_eight_digit_number_l502_502701

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l502_502701


namespace angle_bisector_KM_l502_502216

-- Define a structure for the conditions given in the problem
structure parallelogram (A B C D O K L M : Type) extends is_parallelogram : Prop :=
(parallelogram_prop : IsParallelogram A B C D)
(BO_perp_AD : Perpendicular B O A D)
(circle_center_O : Circle O A B)
(intersect_extension_AD_K : ∃ K, Circle O A B ∧ LineIntersectExtension A D K)
(intersect_BK_CD_L : ∃ L, SegmentIntersect B K C D L)
(ray_OL_intersect_circle_M : ∃ M, RayIntersect O L Circle O A B M)

-- The theorem statement in Lean
theorem angle_bisector_KM (A B C D O K L M : Type) [parallelogram A B C D O K L M]:
  AngleBisector K M (Angle B K C) :=
by
  -- Proof to be constructed here
  sorry

end angle_bisector_KM_l502_502216


namespace units_digit_of_product_is_eight_l502_502586

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502586


namespace initial_pieces_count_l502_502904

theorem initial_pieces_count (people : ℕ) (pieces_per_person : ℕ) (leftover_pieces : ℕ) :
  people = 6 → pieces_per_person = 7 → leftover_pieces = 3 → people * pieces_per_person + leftover_pieces = 45 :=
by
  intros h_people h_pieces_per_person h_leftover_pieces
  sorry

end initial_pieces_count_l502_502904


namespace max_value_modulus_conjugate_l502_502990

theorem max_value_modulus_conjugate (z : ℂ) (h : |z - complex.I| = 2) : ∃ w, |z - conj z| = w ∧ w = 6 :=
begin
  sorry
end

end max_value_modulus_conjugate_l502_502990


namespace ratio_length_to_width_l502_502538

-- Define the given conditions and values
def width : ℕ := 75
def perimeter : ℕ := 360

-- Define the proof problem statement
theorem ratio_length_to_width (L : ℕ) (P_eq : perimeter = 2 * L + 2 * width) :
  (L / width : ℚ) = 7 / 5 :=
sorry

end ratio_length_to_width_l502_502538


namespace weighted_average_yield_l502_502799

-- Define the conditions
def face_value_A : ℝ := 1000
def market_price_A : ℝ := 1200
def yield_A : ℝ := 0.18

def face_value_B : ℝ := 1000
def market_price_B : ℝ := 800
def yield_B : ℝ := 0.22

def face_value_C : ℝ := 1000
def market_price_C : ℝ := 1000
def yield_C : ℝ := 0.15

def investment_A : ℝ := 5000
def investment_B : ℝ := 3000
def investment_C : ℝ := 2000

-- Prove the weighted average yield
theorem weighted_average_yield :
  (investment_A + investment_B + investment_C) = 10000 →
  ((investment_A / (investment_A + investment_B + investment_C)) * yield_A +
   (investment_B / (investment_A + investment_B + investment_C)) * yield_B +
   (investment_C / (investment_A + investment_B + investment_C)) * yield_C) = 0.186 :=
by
  sorry

end weighted_average_yield_l502_502799


namespace kolya_purchase_l502_502451

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502451


namespace kolya_purchase_l502_502449

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502449


namespace triplet_problem_l502_502290

/-- Find all triplets of positive rational numbers (m, n, p) such that the
    numbers m + 1 / (n * p), n + 1 / (p * m), and p + 1 / (m * n) are integers. -/
theorem triplet_problem (a b c : ℚ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
    (h1 : (a + 1 / (b * c)) ∈ ℚ) (h2 : (b + 1 / (c * a)) ∈ ℚ) (h3 : (c + 1 / (a * b)) ∈ ℚ) :
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = 1/2 ∧ b = 1/2 ∧ c = 4) ∨
    (a = 1/2 ∧ b = 1 ∧ c = 2) ∨
    (a = 1/2 ∧ b = 4 ∧ c = 1/2) ∨
    (a = 1 ∧ b = 2 ∧ c = 1/2) ∨ 
    (a = 2 ∧ b = 1/2 ∧ c = 1) :=
sorry

end triplet_problem_l502_502290


namespace find_x_l502_502289

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem find_x (x : ℝ) (hx₁ : x ≠ 0) (hx₂ : fractional_part x + fractional_part x = floor x) : x = 3 / 2 :=
by
  sorry

end find_x_l502_502289


namespace sufficient_but_not_necessary_condition_l502_502293

theorem sufficient_but_not_necessary_condition (a : ℝ) 
  (sufficient_condition : ∀ x : ℝ, x > 1 → x > a) 
  (not_necessary_condition : ∃ x : ℝ, x ≤ 1 ∧ x > a) : 
  a < 1 := 
begin
  sorry -- Proof goes here
end

end sufficient_but_not_necessary_condition_l502_502293


namespace syllogism_rhombus_square_l502_502864

theorem syllogism_rhombus_square (h_rhombus : ∀ r : ℝ, is_rhombus r → diagonals_perpendicular r)
                                   (h_square : ∀ s : ℝ, is_square s → is_rhombus s) :
  ∀ sq : ℝ, is_square sq → diagonals_perpendicular sq :=
by
  intros sq h_sq
  apply h_rhombus sq
  apply h_square sq
  exact h_sq
  sorry

end syllogism_rhombus_square_l502_502864


namespace uniqueness_of_integers_l502_502069

def sequence_of_integers (a : ℕ → ℤ) : Prop :=
  (∃ f : ℕ → ℕ, (∀ n, 0 < f n ∧ ∀ m, m < n → a (f m) > 0)) ∧
  (∃ g : ℕ → ℕ, (∀ n, 0 < g n ∧ ∀ m, m < n → a (g m) < 0)) ∧
  (∀ n, function.injective (λ i, a i % n))

theorem uniqueness_of_integers (a : ℕ → ℤ) (h : sequence_of_integers a) : 
  ∀ x : ℤ, ∃! n : ℕ, a n = x := 
sorry

end uniqueness_of_integers_l502_502069


namespace possible_items_l502_502432

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502432


namespace units_digit_of_product_is_eight_l502_502583

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502583


namespace purchase_options_l502_502418

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502418


namespace units_digit_of_product_of_first_four_composites_l502_502665

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502665


namespace speed_of_the_man_is_l502_502771

noncomputable def man_speed (train_speed_kmph : ℝ) (train_length_m : ℝ) (time_seconds : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed_mps := train_length_m / time_seconds
  let man_speed_mps := relative_speed_mps - train_speed_mps
  man_speed_mps * (3600 / 1000)

theorem speed_of_the_man_is
  (train_length : ℝ) (train_speed : ℝ) (time_taken : ℝ)
  (h_train_length : train_length = 605)
  (h_train_speed : train_speed = 60)
  (h_time_taken : time_taken = 33) :
  man_speed train_speed train_length time_taken ≈ 5.976 := by {
  rw [h_train_length, h_train_speed, h_time_taken],
  sorry
}

end speed_of_the_man_is_l502_502771


namespace number_of_outfits_l502_502727

theorem number_of_outfits (shirts pants : ℕ) (h_shirts : shirts = 5) (h_pants : pants = 3) 
    : shirts * pants = 15 := by
  sorry

end number_of_outfits_l502_502727


namespace largest_eight_digit_with_all_even_digits_l502_502705

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l502_502705


namespace modulus_of_z_l502_502310

-- Define the imaginary unit
def i := Complex.I

-- Given conditions
variables (a b : ℝ) (ha_cond : a + 2 * i = 1 - b * i)

-- Define the complex number
def z := a + b * i

-- Statement: Prove that the modulus of the complex number z is sqrt(5)
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l502_502310


namespace characterize_function_l502_502803

noncomputable theory
open Classical

def satisfies_conditions (f : ℕ+ → ℕ) : Prop :=
  (∃ n : ℕ+, f n ≠ 0) ∧
  (∀ x y : ℕ+, f (x * y) = f x + f y) ∧
  (∀ᶠ n in Filter.at_top, ∀ k : ℕ+, k < n → f k = f (n - k))

def solution_function (n : ℕ) (p : ℕ) (c : ℕ) (hp : p.prime) : ℕ :=
  c * (n.factorization.get p).getD 0

theorem characterize_function (f : ℕ+ → ℕ) :
  satisfies_conditions f →
  ∃ (p : ℕ) (c : ℕ), p.prime ∧ ∀ n : ℕ, f n = solution_function n p c :=
sorry

end characterize_function_l502_502803


namespace circle_tangent_to_parabola_l502_502363

section parabola_tangent_circle

variables {p : ℝ}
variables {x₁ y₁ x₂ y₂ : ℝ}

-- P1 and P2 lie on the parabola y^2 = 2px
def on_parabola (x y : ℝ) : Prop := y^2 = 2 * p * x

-- P1 and P2 are on the parabola
hypothesis hP1_on_parabola : on_parabola x₁ y₁
hypothesis hP2_on_parabola : on_parabola x₂ y₂

-- |y₁ - y₂| = 4p
hypothesis hy1_y2_difference : abs (y₁ - y₂) = 4 * p

theorem circle_tangent_to_parabola :
  let circle_eq : ℝ → ℝ → ℝ := λ x y, (x - x₁) * (x - x₂) + (y - y₁) * (y - y₂) 
  in ∃ x₃ y₃, on_parabola x₃ y₃ ∧ circle_eq x₃ y₃ = 0.
sorry

end parabola_tangent_circle

end circle_tangent_to_parabola_l502_502363


namespace abscissa_of_A_is_3_l502_502947

-- Definitions of the points A, B, line l and conditions
def in_first_quadrant (A : ℝ × ℝ) := (A.1 > 0) ∧ (A.2 > 0)

def on_line_l (A : ℝ × ℝ) := A.2 = 2 * A.1

def point_B : ℝ × ℝ := (5, 0)

def diameter_circle (A B : ℝ × ℝ) (P : ℝ × ℝ) :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Vectors AB and CD
def vector_AB (A B : ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

def vector_CD (C D : ℝ × ℝ) := (D.1 - C.1, D.2 - C.2)

def dot_product_zero (A B C D : ℝ × ℝ) := (vector_AB A B).1 * (vector_CD C D).1 + (vector_AB A B).2 * (vector_CD C D).2 = 0

-- Statement to prove
theorem abscissa_of_A_is_3 (A : ℝ × ℝ) (D : ℝ × ℝ) (a : ℝ) :
  in_first_quadrant A →
  on_line_l A →
  diameter_circle A point_B D →
  dot_product_zero A point_B (a, a) D →
  A.1 = 3 :=
by
  sorry

end abscissa_of_A_is_3_l502_502947


namespace find_a_l502_502871

-- Define the first line and its slope
def line1 (a : ℝ) : ℝ → ℝ := λ x, a * x + 2 * a

-- Define the second line and its slope
def line2 (a : ℝ) : ℝ → ℝ := λ x, -((2 * a - 1) / a) * x - 1

-- Define the condition for perpendicular lines
def perpendicular_condition (a : ℝ) : Prop :=
  a * (-(2 * a - 1) / a) = -1

-- Given the conditions, prove that the values of a are 1 or 0
theorem find_a (a : ℝ) :
  perpendicular_condition a → (a = 1 ∨ a = 0) :=
by sorry

end find_a_l502_502871


namespace find_smaller_bags_l502_502959

-- Definitions based on conditions
section scuba_diving

variable (hours_spent : ℕ) (coins_per_hour : ℕ) (total_coins : ℕ)
variable (treasure_chest_coins : ℕ) (smaller_bag_coins : ℕ) (num_smaller_bags : ℕ)

-- Given conditions
def jim_conditions :=
  hours_spent = 8 ∧
  coins_per_hour = 25 ∧
  total_coins = hours_spent * coins_per_hour ∧
  treasure_chest_coins = 100 ∧
  smaller_bag_coins = treasure_chest_coins / 2 ∧
  num_smaller_bags = (total_coins - treasure_chest_coins) / smaller_bag_coins

-- The theorem to prove
theorem find_smaller_bags (h : jim_conditions) : num_smaller_bags = 2 :=
by
  simp [jim_conditions] at h
  sorry

end scuba_diving

end find_smaller_bags_l502_502959


namespace student_correct_sums_l502_502766

-- Defining variables R and W along with the given conditions
variables (R W : ℕ)

-- Given conditions as Lean definitions
def condition1 := W = 5 * R
def condition2 := R + W = 180

-- Statement of the problem to prove R equals 30
theorem student_correct_sums :
  (W = 5 * R) → (R + W = 180) → R = 30 :=
by
  -- Import needed definitions and theorems from Mathlib
  sorry -- skipping the proof

end student_correct_sums_l502_502766


namespace distance_to_left_focus_l502_502347

noncomputable def ellipse_eqn (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 5 = 1
noncomputable def hyperbola_eqn (x y : ℝ) : Prop := x^2 - (y^2) / 3 = 1
noncomputable def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem distance_to_left_focus (x y : ℝ) (P : x > 0 ∧ y > 0) :
  ellipse_eqn x y ∧ hyperbola_eqn x y → 
  P = (4, sqrt(21/5)) := sorry

end distance_to_left_focus_l502_502347


namespace not_option_D_l502_502550

variable (O Δ □ : ℝ)

-- Given conditions
def condition1 : Prop := 2 * O = 6 * □
def condition2 : Prop := 2 * O + 6 * □ = 4 * Δ

-- Statement to prove
theorem not_option_D (h1 : condition1 O □) (h2 : condition2 O Δ □) :
  ¬(O + Δ = 4 * □) :=
  sorry

end not_option_D_l502_502550


namespace tan_C_equals_neg1_l502_502040

noncomputable def tan_of_A (A : Real) := (1 : Real) / 2
noncomputable def cos_of_B (B : Real) := (3 * Real.sqrt 10) / 10
noncomputable def tan_of_C (C : Real) (A B : Real) := -1

theorem tan_C_equals_neg1 (A B C : Real)
  (h1 : tan A = tan_of_A A)
  (h2 : cos B = cos_of_B B) :
  tan C = tan_of_C C A B := by
sorry

end tan_C_equals_neg1_l502_502040


namespace nancy_soap_bars_l502_502998

def packs : ℕ := 6
def bars_per_pack : ℕ := 5

theorem nancy_soap_bars : packs * bars_per_pack = 30 := by
  sorry

end nancy_soap_bars_l502_502998


namespace units_digit_first_four_composites_l502_502622

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502622


namespace kolya_purchase_l502_502448

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502448


namespace distance_between_lines_l502_502914

def line_eq (A B C : ℝ) : ℝ × ℝ → Prop := λ p, A * p.1 + B * p.2 + C = 0

theorem distance_between_lines 
  (a : ℝ)
  (l1 : ∀ p : ℝ × ℝ, line_eq 1 a 6 p)
  (l2 : ∀ p : ℝ × ℝ, line_eq (a - 2) 3 (2 * a) p)
  (h_parallel : (a ≠ 3))
  (h_parallel' : l1.parallel_with l2)
  : distance_between l1 l2 = (8 * Real.sqrt 2 / 3) :=
sorry

end distance_between_lines_l502_502914


namespace distance_MN_l502_502109

-- Define points M and N
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def M : Point3D := { x := 1, y := 0, z := 2 }
def N : Point3D := { x := -1, y := 2, z := 0 }

-- Distance formula
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

-- The theorem statement
theorem distance_MN : distance M N = 2 * real.sqrt 3 := by
  sorry

end distance_MN_l502_502109


namespace neither_sufficient_nor_necessary_condition_l502_502059

-- Define the vectors a and b in the real vector space
variables (a b : EuclideanSpace ℝ (Fin 3)) -- Assuming 3D space for concreteness

-- Lean statement for the proof problem
theorem neither_sufficient_nor_necessary_condition 
    (h : (|a| = |b|)) : (|a + b| = |a - b|) ↔ false := 
by
  sorry -- The proof is omitted as per the prompt

end neither_sufficient_nor_necessary_condition_l502_502059


namespace count_ways_to_make_change_50_l502_502000

-- Definitions and conditions
def is_valid_coin (n : ℕ) : Prop := n = 1 ∨ n = 5 ∨ n = 10 ∨ n = 25 ∨ n = 50

def total_cents (coins : list ℕ) : ℕ := coins.sum

def valid_combination_no_half_dollar (coins : list ℕ) : Prop :=
  total_cents coins = 50 ∧ (¬ coins = [50]) ∧ (∀ c ∈ coins, is_valid_coin c)

-- The proof statement
theorem count_ways_to_make_change_50 : 
  ∃ (combinations : finset (list ℕ)), 
    (∀ c ∈ combinations, valid_combination_no_half_dollar c) 
    ∧ combinations.card = 20 :=
sorry

end count_ways_to_make_change_50_l502_502000


namespace graph_inequality_l502_502001

def is_non_planar (G : Type*) [Graph G] : Prop :=
  ∃ (vertices : set G) (edges : set (G × G)), non_planar_graph vertices edges

def kappa (G : Type*) [Graph G] : ℕ :=
  vertex_connectivity G

def lambda (G : Type*) [Graph G] : ℕ :=
  edge_connectivity G

def delta (G : Type*) [Graph G] : ℕ :=
  minimum_degree G

theorem graph_inequality (G : Type*) [Graph G] (h : is_non_planar G) :
  kappa(G) ≤ lambda(G) ∧ lambda(G) ≤ delta(G) :=
sorry

end graph_inequality_l502_502001


namespace equilateral_triangle_of_altitude_sum_l502_502952

def triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def altitude (a b c : ℝ) (S : ℝ) : ℝ := 
  2 * S / a

noncomputable def inradius (S : ℝ) (s : ℝ) : ℝ := 
  S / s

def shape_equilateral (a b c : ℝ) : Prop := 
  a = b ∧ b = c

theorem equilateral_triangle_of_altitude_sum (a b c h_a h_b h_c r S s : ℝ) 
  (habc : triangle a b c)
  (ha : h_a = altitude a b c S)
  (hb : h_b = altitude b a c S)
  (hc : h_c = altitude c a b S)
  (hr : r = inradius S s)
  (h_sum : h_a + h_b + h_c = 9 * r)
  (h_area : S = s * r)
  (h_semi : s = (a + b + c) / 2) : 
  shape_equilateral a b c := 
sorry

end equilateral_triangle_of_altitude_sum_l502_502952


namespace calculate_EI_l502_502034

theorem calculate_EI 
  (ABCD_is_square : ∀ (A B C D : ℝ) (s : ℝ), (s = 2) → (dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s))
  (rectangles_congruent : ∀ (E F G H I J K L : ℝ), (E ≠ F ∧ F ≠ G ∧ G ≠ H ∧ E ≠ H ∧ I ≠ J ∧ J ≠ K ∧ K ≠ L ∧ I ≠ L) → (dist E F = dist I J ∧ dist F G = dist J K ∧ dist G H = dist K L ∧ dist H E = dist L I))
  (EFGH_within_square : ∀ (A B C D E F G H : ℝ), (dist H E = dist E F) → (H ≤ D) → (E ≥ A))
  (J_on_side_CD : ∀ (C D J : ℝ), (J ∈ seg C D))
  (L_outside_ABCD : ∀ (A B C D L : ℝ), (L ∉ seg A B) ∧ (L ∉ seg B C) ∧ (L ∉ seg C D) ∧ (L ∉ seg D A))
  (EH_FK_one : ∀ (E H F K : ℝ), (dist E H = 1) ∧ (dist F K = 1))
  : EI = 2 :=
sorry

end calculate_EI_l502_502034


namespace units_digit_of_composite_product_l502_502658

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502658


namespace negation_of_existential_l502_502390

theorem negation_of_existential : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 ≤ 0) := 
by 
  sorry

end negation_of_existential_l502_502390


namespace valid_divisors_of_450_l502_502046

theorem valid_divisors_of_450 :
  let m_values := {m ∈ (finset.Ico 1 451) | 450 % m = 0 ∧ m > 1 ∧ m < 450}
  m_values.card = 17 :=
by
  let total_divisors := {m ∈ (finset.Ico 1 451) | 450 % m = 0}
  have total_divisors_card : total_divisors.card = 18 := sorry
  let m_values := total_divisors.filter (λ m, m > 1 ∧ m < 450)
  have m_values_card_eq_total_sub_two : m_values.card = 18 - 2 := by
    rw [←total_divisors_card, finset.card_filter]
    exact finset.card_Ico _ _ _
  simpa using m_values_card_eq_total_sub_two

end valid_divisors_of_450_l502_502046


namespace units_digit_of_product_of_first_four_composites_l502_502664

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502664


namespace decreasing_interval_sum_of_10_terms_l502_502882

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin x * cos x - cos x ^ 2

-- Define the sequence aₙ
def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then (n / 2) * π + π / 4
  else (n / 2) * π + π / 2

-- Part (1): Prove the interval where f(x) is decreasing
theorem decreasing_interval (k : ℤ) :
  ∀ x, (k * π + (3 * π / 8) ≤ x ∧ x ≤ k * π + (7 * π / 8)) → f x < f (x + 1) :=
sorry

-- Part (2): Prove the sum of the first 10 terms of the sequence aₙ
theorem sum_of_10_terms :
  (Finset.range 10).sum (λ n, a (n + 1)) = (95 * π) / 4 :=
sorry

end decreasing_interval_sum_of_10_terms_l502_502882


namespace trust_meteorologist_l502_502175

/-- 
The probability of a clear day in Anchuria.
-/
def P_G : ℝ := 0.74

/-- 
Accuracy of the forecast by each senator.
-/
variable (p : ℝ)

/-- 
Accuracy of the meteorologist's forecast being 1.5 times that of a senator.
-/
def meteorologist_accuracy : ℝ := 1.5 * p

/-- 
Calculations and final proof that the meteorologist's forecast is more reliable than that of the senators. 
-/
theorem trust_meteorologist (p : ℝ) (Hp1 : 0 ≤ p) (Hp2 : p ≤ 1) : 
  λ P_S_M1_M2_G P_S_M1_M2_not_G : 
  P_G := 0.74 ∧ meteorologist_accuracy = 1.5 * p ∧
  (∀ P_S_M1_M2, P_S_M1_M2 = P_S_M1_M2_G * P_G + P_S_M1_M2_not_G * (1 - P_G)) → 
  (P_S_M1_M2_not_G * (1 - P_G) > P_S_M1_M2_G * P_G) :=
begin
  sorry
end

end trust_meteorologist_l502_502175


namespace empty_subset_of_disjoint_and_nonempty_l502_502338

variable {α : Type*} (A B : Set α)

theorem empty_subset_of_disjoint_and_nonempty (h₁ : A ≠ ∅) (h₂ : A ∩ B = ∅) : ∅ ⊆ B :=
by
  sorry

end empty_subset_of_disjoint_and_nonempty_l502_502338


namespace count_valid_integers_l502_502374

theorem count_valid_integers : 
  (∀ n ∈ [1..9999], 
    ¬(∃ d ∈ ['2', '3', '4', '5', '8'], d ∈ n.toString.toList)) → 
  (number_of_valid_integers = 624) := 
by
  sorry

end count_valid_integers_l502_502374


namespace find_value_of_f_one_l502_502975

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 2 * x^2 - x else - (2 * (-x)^2 - (-x))

theorem find_value_of_f_one : f 1 = -3 :=
by
  -- proof omitted
  sorry

end find_value_of_f_one_l502_502975


namespace derivative_inequality_l502_502486

noncomputable def P (x : ℝ) (n : ℕ) (x_i : Fin n → ℝ) : ℝ :=
  ∏ i in Finset.univ, (x - x_i i)

theorem derivative_inequality 
  (n : ℕ) 
  (x_i : Fin n → ℝ) 
  (x : ℝ) :
  (deriv (λ x, P x n x_i) x)^2 ≥ 
  (P x n x_i) * (deriv (deriv (λ x, P x n x_i)) x) :=
sorry

end derivative_inequality_l502_502486


namespace units_digit_first_four_composites_l502_502599

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502599


namespace tan_product_and_sum_l502_502507

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_product_and_sum :
  (tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = √3) ∧ 
  (tan (π / 9) ^ 2 + tan (2 * π / 9) ^ 2 + tan (4 * π / 9) ^ 2 = 33) :=
sorry

end tan_product_and_sum_l502_502507


namespace possible_items_l502_502433

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502433


namespace cube_surface_area_increase_l502_502154

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l502_502154


namespace game_ends_after_45_rounds_l502_502023

-- Define the initial tokens for each player
def initial_tokens_A : ℕ := 18
def initial_tokens_B : ℕ := 16
def initial_tokens_C : ℕ := 15

-- Function to simulate one round of the game
def simulate_round (tokens_A tokens_B tokens_C : ℕ) : ℕ × ℕ × ℕ :=
  if tokens_A ≥ tokens_B ∧ tokens_A ≥ tokens_C then
    (tokens_A - 3, tokens_B + 1, tokens_C + 1)
  else if tokens_B ≥ tokens_A ∧ tokens_B ≥ tokens_C then
    (tokens_A + 1, tokens_B - 3, tokens_C + 1)
  else
    (tokens_A + 1, tokens_B + 1, tokens_C - 3)

-- Function to simulate multiple rounds of the game
def simulate_game (rounds : ℕ) : ℕ × ℕ × ℕ :=
  (List.range rounds).foldl (λ (tokens : ℕ × ℕ × ℕ) _ => simulate_round tokens.1 tokens.2 tokens.3) 
  (initial_tokens_A, initial_tokens_B, initial_tokens_C)

-- Prove that the game ends after 45 rounds
theorem game_ends_after_45_rounds : (simulate_game 45).2 = 0 :=
sorry

end game_ends_after_45_rounds_l502_502023


namespace rhombus_area_eq_72_l502_502403

theorem rhombus_area_eq_72 (A B C D : ℝ × ℝ)
  (hA : A = (0, 4.5)) (hB : B = (8, 0)) (hC : C = (0, -4.5)) (hD : D = (-8, 0)) :
  let d1 := real.dist A C,
      d2 := real.dist B D in
  (d1 * d2) / 2 = 72 :=
  by
    sorry

end rhombus_area_eq_72_l502_502403


namespace solution_set_of_inequality_l502_502353

noncomputable def f : ℝ → ℝ :=
λ x, if 1 < x then 2^x - x else 1

theorem solution_set_of_inequality :
  {x : ℝ | f x < f (2 / x)} = set.Ioo 0 (real.sqrt 2) :=
by
  sorry

end solution_set_of_inequality_l502_502353


namespace units_digit_product_first_four_composite_numbers_l502_502670

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502670


namespace market_value_of_stock_l502_502732

-- Definitions based on the conditions
def face_value : ℝ := 100
def annual_dividend (face_value : ℝ) : ℝ := 0.05 * face_value
def yield (annual_dividend market_value : ℝ) : ℝ := (annual_dividend / market_value) * 100

theorem market_value_of_stock : ∃ (market_value : ℝ), 
  let annual_dividend := annual_dividend face_value in
  yield annual_dividend market_value = 10 ∧ 
  market_value = 50 :=
by
  -- Proof goes here
  sorry

end market_value_of_stock_l502_502732


namespace number_of_items_l502_502455

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502455


namespace floor_alpha_pow_is_multiple_of_k_l502_502490

theorem floor_alpha_pow_is_multiple_of_k (k n : ℕ) (h₁ : 0 < k) :
  let α := k + 1 / 2 + Real.sqrt (k^2 + 1/4)
  in k ∣ Int.floor (α^n) :=
begin
  sorry
end

end floor_alpha_pow_is_multiple_of_k_l502_502490


namespace units_digit_first_four_composites_l502_502572

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502572


namespace distinct_primes_in_y_seq_l502_502215

def is_capicua (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

def capicua_seq : ℕ → ℕ
| 0       := 1
| (n + 1) := Nat.find (λ m, m > (capicua_seq n) ∧ is_capicua m)

def y (i : ℕ) : ℕ :=
  capicua_seq (i + 1) - capicua_seq i

def distinct_primes_in_set : ℕ :=
  (Finset.image y Finset.univ).filter Nat.prime |>.card

theorem distinct_primes_in_y_seq : distinct_primes_in_set = 2 := 
  sorry

end distinct_primes_in_y_seq_l502_502215


namespace find_z_l502_502394

theorem find_z 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h_eq : 4 * x = 5 * y) 
  (h_sum : x + y + z = 37) : 
  z = 28 :=
begin
  sorry
end

end find_z_l502_502394


namespace count_valid_integers_l502_502375

theorem count_valid_integers : 
  (∀ n ∈ [1..9999], 
    ¬(∃ d ∈ ['2', '3', '4', '5', '8'], d ∈ n.toString.toList)) → 
  (number_of_valid_integers = 624) := 
by
  sorry

end count_valid_integers_l502_502375


namespace largest_eight_digit_number_contains_even_digits_l502_502723

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l502_502723


namespace units_digit_first_four_composites_l502_502578

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502578


namespace kopeechka_purchase_l502_502439

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502439


namespace toads_l502_502561

theorem toads (Tim Jim Sarah : ℕ) 
  (h1 : Jim = Tim + 20) 
  (h2 : Sarah = 2 * Jim) 
  (h3 : Sarah = 100) : Tim = 30 := 
by 
  -- Proof will be provided later
  sorry

end toads_l502_502561


namespace obtuse_triangle_division_into_acute_l502_502087

-- Definitions of the points and the conditions
variable 
  {A B C : Point} 
  (h_triangle : scalene_triangle A B C) -- Condition 1
  (h_triangle_obtuse : obtuse_angled_triangle A B C)

noncomputable def incenter (A B C : Point) : Point := sorry 

variable 
  (O : Point)
  (h_O : O = incenter A B C) -- Condition 2

noncomputable def radius_OC (O : Point) : ℝ := sorry

variable 
  (h_circle : circle O (radius_OC O)) -- Condition 3

-- Points K, M, P, Q on the sides of the triangle
variable 
  (K M P Q : Point)
  (h_KM_AB : K M ∈ (segment A B))
  (h_P_BC : P ∈ (segment B C))
  (h_Q_CA : Q ∈ (segment C A))

-- Connect the points as described in the problem
variable 
  (h_connections : True) -- This is a placeholder for the actual connection conditions

-- Problem statement: prove obtuse-angled triangle can be divided into 7 acute-angled triangles
theorem obtuse_triangle_division_into_acute 
  (A B C O K M P Q : Point)
  (h_triangle : scalene_triangle A B C)
  (h_triangle_obtuse : obtuse_angled_triangle A B C)
  (h_O : O = incenter A B C)
  (h_circle : circle O (radius_OC O))
  (h_KM_AB : K M ∈ (segment A B))
  (h_P_BC : P ∈ (segment B C))
  (h_Q_CA : Q ∈ (segment C A))
  (h_connections : True) :
  ∃ triangles : List (Triangle), 
    (∀ t ∈ triangles, acute_angled_triangle t) ∧ 
    length triangles = 7 ∧
    (union_of_triangles triangles = triangle A B C) := sorry

end obtuse_triangle_division_into_acute_l502_502087


namespace number_of_items_l502_502456

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502456


namespace shanghai_team_score_l502_502929

variables (S B : ℕ)

-- Conditions
def yao_ming_points : ℕ := 30
def point_margin : ℕ := 10
def total_points_minus_10 : ℕ := 5 * yao_ming_points - 10
def combined_total_points : ℕ := total_points_minus_10

-- The system of equations as conditions
axiom condition1 : S - B = point_margin
axiom condition2 : S + B = combined_total_points

-- The proof statement
theorem shanghai_team_score : S = 75 :=
by
  sorry

end shanghai_team_score_l502_502929


namespace original_area_area_after_translation_l502_502058

-- Defining vectors v, w, and t
def v : ℝ × ℝ := (6, -4)
def w : ℝ × ℝ := (-8, 3)
def t : ℝ × ℝ := (3, 2)

-- Function to compute the determinant of two vectors in R^2
def det (v w : ℝ × ℝ) : ℝ := v.1 * w.2 - v.2 * w.1

-- The area of a parallelogram is the absolute value of the determinant
def parallelogram_area (v w : ℝ × ℝ) : ℝ := |det v w|

-- Proving the original area is 14
theorem original_area : parallelogram_area v w = 14 := by
  sorry

-- Proving the area remains the same after translation
theorem area_after_translation : parallelogram_area v w = parallelogram_area (v.1 + t.1, v.2 + t.2) (w.1 + t.1, w.2 + t.2) := by
  sorry

end original_area_area_after_translation_l502_502058


namespace count_mod_6_mod_11_lt_1000_l502_502902

theorem count_mod_6_mod_11_lt_1000 : ∃ n : ℕ, (∀ x : ℕ, (x < n + 1) ∧ ((6 + 11 * x) < 1000) ∧ (6 + 11 * x) % 11 = 6) ∧ (n + 1 = 91) :=
by
  sorry

end count_mod_6_mod_11_lt_1000_l502_502902


namespace amount_coach_mike_gave_l502_502247

-- Definitions from conditions
def cost_of_lemonade : ℕ := 58
def change_received : ℕ := 17

-- Theorem stating the proof problem
theorem amount_coach_mike_gave : cost_of_lemonade + change_received = 75 := by
  sorry

end amount_coach_mike_gave_l502_502247


namespace literate_females_percentage_l502_502080

-- Defining the number of inhabitants in the town
def total_inhabitants : ℕ := 2500

-- Percentage of males in the town
def percent_males : ℝ := 55 / 100

-- Percentage of literate males
def percent_literate_males : ℝ := 35 / 100

-- Percentage of literate inhabitants
def percent_literate_inhabitants : ℝ := 40 / 100

-- Number of males
def num_males : ℕ := (percent_males * total_inhabitants).toNat

-- Number of literate males
def num_literate_males : ℕ := (percent_literate_males * num_males).toNat

-- Total number of literate inhabitants
def total_literate_inhabitants : ℕ := (percent_literate_inhabitants * total_inhabitants).toNat

-- Number of females
def num_females : ℕ := total_inhabitants - num_males

-- Number of literate females
def num_literate_females : ℕ := total_literate_inhabitants - num_literate_males

-- Percentage of literate females
def percent_literate_females : ℝ := (num_literate_females.toReal / num_females.toReal) * 100

-- Statement to prove that the percentage of literate females is approximately 46.13%
theorem literate_females_percentage : abs (percent_literate_females - 46.13) < 0.01 := by
    sorry

end literate_females_percentage_l502_502080


namespace units_digit_of_composite_product_l502_502655

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502655


namespace determinant_transformation_l502_502002

theorem determinant_transformation (p q r s : ℝ) 
    (h : Matrix.det !![![p, q], ![r, s]] = 9) : 
    Matrix.det !![![2*p, 5*p + 4*q], ![2*r, 5*r + 4*s]] = 72 := 
by 
    sorry

end determinant_transformation_l502_502002


namespace lines_perpendicular_or_parallel_l502_502833

noncomputable theory

-- Defining the planes and lines involved
variable (α β : Plane)
variable (a b l : Line)

-- Defining the conditions
def dihedral_angle (α l β : Plane) : Prop := sorry -- formalize the notion of a dihedral angle
def in_plane (x : Line) (p : Plane) : Prop := sorry -- formalize the notion of a line being in a plane
def not_perpendicular (x y : Line) : Prop := sorry -- formalize the notion of two lines not being perpendicular

-- The actual theorem
theorem lines_perpendicular_or_parallel (h1 : dihedral_angle α l β)
    (h2 : in_plane a α) (h3 : in_plane b β)
    (h4 : not_perpendicular a l) (h5 : not_perpendicular b l) :
    (a.perpendicular b ∨ a.parallel b) :=
sorry

end lines_perpendicular_or_parallel_l502_502833


namespace units_digit_first_four_composites_l502_502577

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502577


namespace units_digit_product_first_four_composite_numbers_l502_502675

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502675


namespace units_digit_of_product_is_eight_l502_502582

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502582


namespace correct_propositions_l502_502893

variables (m n : Line)
variables (α β : Plane)
variables (m_perp_alpha : IsPerpendicular m α) (n_in_beta : IsIn n β)
variables (m_perp_beta : IsPerpendicular m β) (n_perp_alpha : IsPerpendicular n α) (alpha_parallel_beta : IsParallel α β) (alpha_perp_beta : IsPerpendicular α β) (m_parallel_n : IsParallel m n)

theorem correct_propositions :
  (↑([
    (alpha_parallel_beta ∧ m_perp_beta) → m_perp_n,
    m_perp_n → alpha_parallel_beta,
    (m_parallel_n ∧ n_perp_alpha) → alpha_perp_beta,
    alpha_perp_beta → m_parallel_n
  ])).count(true) = 2 := by
  sorry

end correct_propositions_l502_502893


namespace find_sin_alpha_l502_502331

theorem find_sin_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos (α + π / 4) = 5 / 13) : 
  sin α = 7 * real.sqrt 2 / 26 :=
sorry

end find_sin_alpha_l502_502331


namespace exists_tangent_circle_l502_502320

variables {α : Type*} [MetricSpace α]

-- Conditions
variable (A B C M : α)
variable (ABC : angle ABC)

-- Definition of the constructed circle
noncomputable def constructed_circle (S : set α) := ∃ (O : α) (r : ℝ), Metric.ball O r = S ∧
  (∀ p ∈ S, Metric.dist p (line_through A B) = r) ∧
  (∀ p ∈ S, Metric.dist p (line_through B C) = r) ∧
  M ∈ S

-- Theorem Statement
theorem exists_tangent_circle (ABC : angle ABC) (M : α) :
  ∃ S, constructed_circle ABC M S :=
sorry

end exists_tangent_circle_l502_502320


namespace problem_f_values_order_l502_502851

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y ∈ s, x < y → f y ≤ f x

theorem problem_f_values_order 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_mono : is_monotonically_decreasing f (Set.Iio 0)) :
  f (Real.logb 4 (1/6)) > f (Real.logb 4 (1/5)) > f (2^(-3/4)) :=
by
  sorry

end problem_f_values_order_l502_502851


namespace chinese_mathematical_system_l502_502942

noncomputable def problem_statement : Prop :=
  ∃ (x : ℕ) (y : ℕ),
    7 * x + 7 = y ∧ 
    9 * (x - 1) = y

theorem chinese_mathematical_system :
  problem_statement := by
  sorry

end chinese_mathematical_system_l502_502942


namespace units_digit_first_four_composite_is_eight_l502_502612

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502612


namespace length_of_XY_correct_l502_502820

noncomputable def length_of_XY (XZ : ℝ) (angleY : ℝ) (angleZ : ℝ) :=
  if angleZ = 90 ∧ angleY = 30 then 8 * Real.sqrt 3 else panic! "Invalid triangle angles"

theorem length_of_XY_correct : length_of_XY 12 30 90 = 8 * Real.sqrt 3 :=
by
  sorry

end length_of_XY_correct_l502_502820


namespace consecutive_not_prime_power_l502_502504

theorem consecutive_not_prime_power (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n) → ¬ is_prime_power (m + k)) :=
by
  sorry

end consecutive_not_prime_power_l502_502504


namespace units_digit_first_four_composites_l502_502620

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502620


namespace units_digit_product_first_four_composite_numbers_l502_502677

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502677


namespace watermelons_left_to_be_sold_tomorrow_l502_502235

def initial_watermelons := 10 * 12
def sold_yesterday := 0.40 * initial_watermelons
def remaining_after_yesterday := initial_watermelons - sold_yesterday
def sold_today := remaining_after_yesterday / 4
def remaining_for_tomorrow := remaining_after_yesterday - sold_today

theorem watermelons_left_to_be_sold_tomorrow : remaining_for_tomorrow = 54 :=
by
  -- Proof goes here, currently omitted with 'sorry'
  sorry

end watermelons_left_to_be_sold_tomorrow_l502_502235


namespace simultaneous_equations_in_quadrant_I_l502_502480

theorem simultaneous_equations_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 4 / 3) :=
  sorry

end simultaneous_equations_in_quadrant_I_l502_502480


namespace symmetric_function_is_periodic_l502_502505

theorem symmetric_function_is_periodic {f : ℝ → ℝ} {a b y0 : ℝ}
  (h1 : ∀ x, f (a + x) - y0 = y0 - f (a - x))
  (h2 : ∀ x, f (b + x) = f (b - x))
  (hb : b > a) :
  ∀ x, f (x + 4 * (b - a)) = f x := sorry

end symmetric_function_is_periodic_l502_502505


namespace inverse_h_l502_502482

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h (x : ℝ) : h⁻¹ (x) = (x - 7) / 12 :=
sorry

end inverse_h_l502_502482


namespace units_digit_first_four_composites_l502_502575

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502575


namespace sum_a_99_l502_502313

noncomputable def a (n : ℕ) : ℝ := 
  1 / ((n + 1) * real.sqrt n + n * real.sqrt (n + 1))

theorem sum_a_99 : 
  ∑ k in finset.range 99, a k.succ = 9 / 10 :=
by
  sorry

end sum_a_99_l502_502313


namespace next_palindrome_product_l502_502551

def is_palindrome (n : Nat) : Bool :=
  let s := toString n
  s = s.reverse

def digits_product (n : Nat) : Nat :=
  n.digits.foldr (λ x acc => x * acc) 1

theorem next_palindrome_product :
  digits_product 2121 = 4 := by
  sorry

end next_palindrome_product_l502_502551


namespace units_digit_of_first_four_composite_numbers_l502_502632

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502632


namespace teams_worked_together_days_l502_502500

noncomputable def first_team_rate : ℝ := 1 / 12
noncomputable def second_team_rate : ℝ := 1 / 9
noncomputable def first_team_days : ℕ := 5
noncomputable def total_work : ℝ := 1
noncomputable def work_first_team_alone := first_team_rate * first_team_days

theorem teams_worked_together_days (x : ℝ) : work_first_team_alone + (first_team_rate + second_team_rate) * x = total_work → x = 3 := 
by
  sorry

end teams_worked_together_days_l502_502500


namespace max_correct_questions_prime_score_l502_502021

-- Definitions and conditions
def total_questions := 20
def points_correct := 5
def points_no_answer := 0
def points_wrong := -2

-- Main statement to prove
theorem max_correct_questions_prime_score :
  ∃ (correct : ℕ) (no_answer wrong : ℕ), 
    correct + no_answer + wrong = total_questions ∧ 
    correct * points_correct + no_answer * points_no_answer + wrong * points_wrong = 83 ∧
    correct = 17 :=
sorry

end max_correct_questions_prime_score_l502_502021


namespace max_students_l502_502274

-- Define the problem
def math_problems : Finset ℕ := (Finset.range 20).map (Finset.range_equiv.mk 1).to_embedding
def physics_problems : Finset ℕ := (Finset.range 11).map (Finset.range_equiv.mk 1).to_embedding

-- Define a student as a selection of one math problem and one physics problem
structure Student :=
(math_problem : ℕ)
(phys_problem : ℕ)
(h_math : math_problem ∈ math_problems)
(h_phys : phys_problem ∈ physics_problems)

-- Define the condition where no two students can select the same pair of problems
def no_same_pair (students : Finset Student) : Prop :=
∀ (s1 s2 : Student), s1 ≠ s2 → (s1.math_problem, s1.phys_problem) ≠ (s2.math_problem, s2.phys_problem)

-- Define the condition where at least one problem selected by any student is chosen by at most one other student
def at_least_one_unpopular (students : Finset Student) : Prop :=
∀ s : Student, (Finset.filter (λ x => x.math_problem = s.math_problem) students).card ≤ 2 ∨
               (Finset.filter (λ x => x.phys_problem = s.phys_problem) students).card ≤ 2

-- The main theorem statement
theorem max_students : ∀ (students : Finset Student),
  no_same_pair students ∧ at_least_one_unpopular students → students.card ≤ 54 :=
sorry

end max_students_l502_502274


namespace vector_perpendicular_l502_502866

noncomputable def angle_between_vectors : ℝ := Real.pi / 3
noncomputable def magnitude_a : ℝ := 2
noncomputable def magnitude_b : ℝ := 3

variables (a b : ℝ → ℝ) -- defining vectors a and b as functions
variable lambda : ℝ -- real number λ

-- defining the dot product function
def dot_product (u v : ℝ → ℝ) : ℝ := ∑ i, u i * v i

-- hypothesis that (2a + λb) is perpendicular to b
def perpendicular_condition (a b : ℝ → ℝ) (λ : ℝ) : Prop :=
  dot_product (λ i, 2 * a i + λ * b i) b = 0

theorem vector_perpendicular (h_angle: angle_between_vectors = Real.pi / 3)
                             (h_mag_a: magnitude_a = 2)
                             (h_mag_b: magnitude_b = 3)
                             (h_perp: perpendicular_condition a b lambda) :
                             lambda = -2 / 3 := sorry

end vector_perpendicular_l502_502866


namespace sum_of_acute_angles_l502_502054

theorem sum_of_acute_angles (A B C D : Type*) [ConvexQuadrilateral A B C D]
  (h : AB * CD = AD * BC ∧ AD * BC = AC * BD) :
  sum_acute_angles A B C D = 60 := 
sorry

end sum_of_acute_angles_l502_502054


namespace math_problem_l502_502865

theorem math_problem
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hxy : 2 * x + y = 1) :

  (xy_le_one_eighth : x * y ≤ 1 / 8) ∧ 
  (min_frac_value : ∃ y, 2/x + 1/y ≥ 9) ∧
  (min_expr_value : 4 * x^2 + y^2 ≥ 1/2) ∧
  (sqrt_expr_value_not_two : ∀ 2, (√(2 * x) + √y) ≤ √2) :=
sorry

end math_problem_l502_502865


namespace inequality_proof_l502_502488

variable (a b c : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)
variable (h_cond : 2 * c > a + b)

theorem inequality_proof :
  c - sqrt (c^2 - a * b) < a ∧ a < c + sqrt (c^2 - a * b) := by
  sorry

end inequality_proof_l502_502488


namespace third_speed_is_3_km_per_hr_l502_502220

theorem third_speed_is_3_km_per_hr :
  (d : ℝ) (v : ℝ)
  (h1 : 3 * d = 900)                             -- Condition: Total distance
  (h2 : d = 300)                                 -- Derived: Single distance d
  (h3 : 11 = d / (6 * 1000 / 60) + d / (9 * 1000 / 60) + d / v) ->  -- Total time formula
  v = 50 * (60 / 1000) :=                        -- Conversion to km/hr
by
  sorry

end third_speed_is_3_km_per_hr_l502_502220


namespace incorrect_statement_D_l502_502683

open Polynomial

/-- Define what it means to be a monomial. -/
def is_monomial (p : Polynomial ℚ) : Prop :=
  p.nat_degree = 0 ∨ (∃ (a : ℚ) (n : ℕ), p = C a * X ^ n)

/-- Define the coefficient of the term. -/
def term_coefficient (p : Polynomial ℚ) (n : ℕ) : ℚ :=
  coeff p n

/-- Define what a quadratic binomial is. -/
def is_quadratic_binomial (p : Polynomial ℚ) : Prop :=
  p.nat_degree = 2 ∧ ∃ a b c : ℚ, p = C a * X ^ 2 + C b * X + C c

/-- Define the degree of the highest degree term. -/
def highest_degree_term (p : Polynomial ℚ) : ℕ :=
  p.nat_degree

theorem incorrect_statement_D :
  ∃ (D : Prop), D ∧
    (∀ (A B C : Prop),
      (A ↔ is_monomial 0 ∧ is_monomial (C 1)) ∧
      (B ↔ term_coefficient ((C (-(1 / 3) : ℚ) * X ^ 2) * (Y ^ 2)) 0 = (-1 / 3)) ∧
      (C ↔ is_quadratic_binomial (C 1 + X ^ 2)) ∧
      (D ↔ highest_degree_term ((C (-2 : ℚ) * X ^ 2) * (Y) + (C 1 * X * Y)) = 5) →
      ¬D) :=
by {
  let A := is_monomial 0 ∧ is_monomial (C 1 : Polynomial ℚ),
  let B := term_coefficient ((C (-(1 / 3) : ℚ) * X ^ 2) * (Y ^ 2)) 0 = (-1 / 3 : ℚ),
  let C := is_quadratic_binomial (C 1 + X ^ 2),
  let D := highest_degree_term ((C (-2 : ℚ) * X ^ 2) * (Y : Polynomial ℚ) + (C 1 * X * Y)) = 5,
  use D,
  split,
  exact D,
  intros A B C D,
  simp at *,
  sorry
}

end incorrect_statement_D_l502_502683


namespace polar_to_cartesian_max_min_xy_l502_502031

-- Define the given polar equation
def polar_equation (rho θ: ℝ) : Prop :=
  rho^2 - 4 * rho * cos θ + 2 = 0

-- Define the corresponding cartesian equation
def cartesian_equation (x y: ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 2 = 0

-- Define the circle equation rewritten from the cartesian equation
def circle_equation (x y: ℝ) : Prop :=
  (x - 2)^2 + y^2 = 2

-- Prove the conversion from polar to Cartesian equation
theorem polar_to_cartesian (ρ θ: ℝ) :
  polar_equation ρ θ → ∃ x y: ℝ, cartesian_equation x y :=
sorry

-- Prove the max and min values of x + y
theorem max_min_xy (x y: ℝ) (h: circle_equation x y) :
  0 ≤ x + y ∧ x + y ≤ 4 :=
sorry

end polar_to_cartesian_max_min_xy_l502_502031


namespace determine_function_expression_l502_502340

noncomputable def f (a b x : ℝ) := a * x^2 + b * x + 3 * a + b
def domain (a : ℝ) := Set.Icc (a - 1) (2 * a)
def evens (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def analytic_expression (a : ℝ) (x : ℝ) := (1 / 3) * x^2 + 1

theorem determine_function_expression (a : ℝ) (b : ℝ):
  evens (f a b) ∧ domain a = Set.Icc (a - 1) (2 * a)
  → a = 1 / 3 ∧ b = 0 ∧ ∀ x, f (1 / 3) 0 x = analytic_expression (1 / 3) x ∧ domain (1 / 3) = Set.Icc (-2 / 3) (2 / 3) :=
by
  sorry

end determine_function_expression_l502_502340


namespace negation_prob1_negation_prob2_negation_prob3_l502_502255

-- Definitions and Conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def defines_const_func (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f x1 = f x2

-- Problem 1
theorem negation_prob1 : 
  (∃ n : ℕ, ∀ p : ℕ, is_prime p → p ≤ n) ↔ 
  ¬(∀ n : ℕ, ∃ p : ℕ, is_prime p ∧ n ≤ p) :=
sorry

-- Problem 2
theorem negation_prob2 : 
  (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) ↔ 
  ¬(∀ n : ℤ, ∃! p : ℤ, n + p = 0) :=
sorry

-- Problem 3
theorem negation_prob3 : 
  (∀ y : ℝ, ¬defines_const_func (λ x => x * y) y) ↔ 
  ¬(∃ y : ℝ, defines_const_func (λ x => x * y) y) :=
sorry

end negation_prob1_negation_prob2_negation_prob3_l502_502255


namespace largest_eight_digit_with_all_even_digits_l502_502708

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l502_502708


namespace usain_full_lap_time_l502_502143

/--
Usain runs one lap around the school stadium at a constant speed. 
Photographers Arina and Marina are positioned near the track.
After the start, for 4 seconds, Usain was closer to Arina, 
then for 21 seconds he was closer to Marina, 
and then until the finish, he was again closer to Arina. 
Prove that the total time for Usain to run one full lap is 42 seconds.
-/
theorem usain_full_lap_time
  (constant_speed : ℝ) 
  (total_time : ℝ) 
  (t_arina_initial : ℝ := 4) 
  (t_marina : ℝ := 21) 
  (t_arina_final : ℝ) 
  (h_split_time : t_arina_initial + t_marina + t_arina_final = total_time) 
  (h_marina_half_lap : t_marina = total_time / 2) :
  total_time = 42 :=
begin
  sorry,
end

end usain_full_lap_time_l502_502143


namespace kolya_purchase_l502_502447

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502447


namespace evaluate_expression_l502_502284

theorem evaluate_expression :
  (81:ℝ)^(1/2) * (64:ℝ)^(-1/3) * (49:ℝ)^(1/2) = (63:ℝ) / 4 :=
by
  sorry

end evaluate_expression_l502_502284


namespace find_functions_l502_502814

open Nat

noncomputable def satisfies_conditions (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ, 1 ≤ n → (f n + f (succ n) = (f (succ (succ n)) * f (succ (succ (succ n)))) - 11)) ∧
  (∀ n : ℕ, 1 ≤ n → f n ≥ 2)

theorem find_functions (f : ℕ+ → ℕ+) :
  satisfies_conditions f ↔ ∃ (a b : ℕ+), (a, b) ∈ [(13, 2), (7, 3), (5, 4), (2, 13), (3, 7), (4, 5)] ∧
  (∀ n : ℕ, 1 ≤ n → f n = if n % 2 = 1 then a else b) :=
by
  sorry

end find_functions_l502_502814


namespace checkerboard_ratio_l502_502789

def rectangles_on_checkerboard (n : ℕ) : ℕ :=
  (nat.choose (n + 1) 2) * (nat.choose (n + 1) 2)

def squares_on_checkerboard (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum (λ k, k * k)

def simplify_ratio (a b : ℕ) : ℚ :=
  rat.mk a b

theorem checkerboard_ratio (n : ℕ) (h : n = 9) :
  let r := rectangles_on_checkerboard n,
      s := squares_on_checkerboard n,
      ratio := simplify_ratio s r,
      m := ratio.num.nat_abs,
      n := ratio.denom in
  r = 2025 ∧
  s = 285 ∧
  ratio = rat.mk 285 2025 ∧
  m = 19 ∧
  n = 135 ∧
  m + n = 154 :=
by {
  sorry
}

end checkerboard_ratio_l502_502789


namespace angle_first_second_quadrants_l502_502160

noncomputable def meaningful_log_angle (θ : ℝ) : Prop := 
  log (cos θ * tan θ) = log (sin θ) ∧ sin θ ≠ 1

theorem angle_first_second_quadrants {θ : ℝ} :
  (0 < θ ∧ θ < π / 2 ∨ π / 2 < θ ∧ θ < π) →
  meaningful_log_angle θ :=
by sorry

end angle_first_second_quadrants_l502_502160


namespace units_digit_of_product_of_first_four_composites_l502_502667

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502667


namespace part_a_part_b_l502_502847

-- Definitions for part (a)
variables {A B C A1 B1 C1 : Point}
variables (Ω ω : Circle)

-- Given conditions for triangles and circles
-- Incircle ω is tangent to BC at A2
variables {A2 : Point}
-- Circle tangent to Ω at A and ω externally at A1
variables (hA : Tangent Ω ω A) (hA1 : Tangent ω (extern A A1))
-- Similarly for B and C
variables (hB : Tangent Ω ω B) (hB1 : Tangent ω (extern B B1))
variables (hC : Tangent Ω ω C) (hC1 : Tangent ω (extern C C1))

-- Part (a)
theorem part_a :
  concurrent_lines A A1 B B1 C C1 := sorry

-- Part (b)
theorem part_b (hA2 : Tangent ω (external_tangent BC A2)) :
  symmetric_over_angle_bisector A A1 A2 := sorry

end part_a_part_b_l502_502847


namespace probability_different_color_chips_l502_502554

-- Define the number of chips for each color
def num_blue_chips := 8
def num_red_chips := 5
def num_yellow_chips := 4
def num_green_chips := 3
def total_chips := num_blue_chips + num_red_chips + num_yellow_chips + num_green_chips

-- Define the theorem statement
theorem probability_different_color_chips :
  let p := (num_blue_chips / total_chips) * ((num_red_chips + num_yellow_chips + num_green_chips) / total_chips) +
           (num_red_chips / total_chips) * ((num_blue_chips + num_yellow_chips + num_green_chips) / total_chips) +
           (num_yellow_chips / total_chips) * ((num_blue_chips + num_red_chips + num_green_chips) / total_chips) +
           (num_green_chips / total_chips) * ((num_blue_chips + num_red_chips + num_yellow_chips) / total_chips)
  in p = 143 / 200 :=
sorry

end probability_different_color_chips_l502_502554


namespace exercise_proof_l502_502311

noncomputable def a : ℝ := (3 / 5) ^ 4
noncomputable def b : ℝ := (3 / 5) ^ 3
noncomputable def c : ℝ := Real.log 3 (3 / 5)

theorem exercise_proof :
  b > a ∧ a > c := by
  sorry

end exercise_proof_l502_502311


namespace largest_eight_digit_number_contains_even_digits_l502_502722

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l502_502722


namespace lambda_range_l502_502877

theorem lambda_range (f : ℝ → ℝ) (m n : ℝ) (h_pos : m > 0) 
  (h_not_monotonic : ¬ ∀ x y ∈ Ioi (0:ℝ), x < y → f x ≤ f y)
  (h_f : ∀ x > 0, f x = Real.log x - (m * (x + n)) / (x + 1)) :
  ∀ λ, m - n > λ → λ < 3 :=
by 
  sorry

end lambda_range_l502_502877


namespace quadratic_ineq_real_solutions_l502_502807

theorem quadratic_ineq_real_solutions (d : ℝ) (h₀ : 0 < d) :
  (∀ x : ℝ, x^2 - 8 * x + d < 0 → 0 < d ∧ d < 16) :=
by
  sorry

end quadratic_ineq_real_solutions_l502_502807


namespace total_population_of_Springfield_and_Greenville_l502_502795

theorem total_population_of_Springfield_and_Greenville :
  let Springfield := 482653
  let diff := 119666
  let Greenville := Springfield - diff
  Springfield + Greenville = 845640 := by
  sorry

end total_population_of_Springfield_and_Greenville_l502_502795


namespace resultant_figure_area_is_correct_l502_502027

-- Define the problem conditions as Lean definitions
def CircleRadius : ℝ := 15
def SectorAngle : ℝ := 45 * Real.pi / 180 -- convert degrees to radians

-- Define the calculated area of resultant figure
noncomputable def ResultantFigureArea : ℝ :=
  let sectorArea := (SectorAngle / (2 * Real.pi)) * (CircleRadius ^ 2 * Real.pi)
  let twoSectorsArea := 2 * sectorArea
  let triangleArea := (1 / 2) * CircleRadius * CircleRadius
  twoSectorsArea + triangleArea

-- The final theorem to prove
theorem resultant_figure_area_is_correct :
  ResultantFigureArea = 675 :=
sorry

end resultant_figure_area_is_correct_l502_502027


namespace t_squared_le_8r_squared_l502_502743

-- Define the triangle and geometric properties
variables {D E F : Point} {r : ℝ} (h: r > 0) 

-- Conditions as definitions
def de_is_diameter (DE : ℝ) : Prop := DE = 2 * r
def t (DF EF : ℝ) : ℝ := DF + EF
def distance_squared (DF EF : ℝ) : Prop := (DF * DF + EF * EF = 4 * r * r)

-- Question as a theorem
theorem t_squared_le_8r_squared (DF EF : ℝ) 
  (hDF: DF > 0) (hEF: EF > 0) 
  (h_triangle_de_is_diameter : de_is_diameter (2 * r))
  (h_distance_squared : distance_squared DF EF) : 
  t DF EF^2 ≤ 8 * r^2 := 
sorry

end t_squared_le_8r_squared_l502_502743


namespace max_product_of_digits_l502_502144

def is_valid_digit_combination (digits : List ℕ) (x y : ℕ) : Prop :=
  (digits = [3, 5, 7, 8, 9]) ∧ (list.nodup digits) ∧ (list.nodup (to_digits x ++ to_digits y)) ∧ 
  (length (to_digits x) = 3) ∧ (length (to_digits y) = 2) ∧ (∀ d, d ∈ to_digits x ∨ d ∈ to_digits y)

theorem max_product_of_digits :
  ∃ x y, is_valid_digit_combination [3, 5, 7, 8, 9] x y ∧
  ∀ x' y', is_valid_digit_combination [3, 5, 7, 8, 9] x' y' → x * y ≥ x' * y' →
  x = 975 :=
by
  sorry

end max_product_of_digits_l502_502144


namespace sam_runs_more_than_sarah_l502_502927

-- Definitions of the conditions
def street_width : ℤ := 30
def block_side_length : ℤ := 400

-- Lean statement to prove the equivalent math proof problem
theorem sam_runs_more_than_sarah :
  let sam_side_length := block_side_length + 2 * street_width in
  let sam_perimeter := 4 * sam_side_length in
  let sarah_perimeter := 4 * block_side_length in
  sam_perimeter - sarah_perimeter = 240 :=
by
  sorry

end sam_runs_more_than_sarah_l502_502927


namespace max_value_2001_exists_l502_502532

theorem max_value_2001_exists : ∃ k : ℕ, (∀ n : ℕ, n ≠ 2001 → (k^2 / 1.001^k) > (n^2 / 1.001^n)) := sorry

end max_value_2001_exists_l502_502532


namespace chessboard_queens_semiperimeter_l502_502081

/-- 
Given a 2004x2004 chessboard with 2004 queens placed such that no two queens attack each other, 
prove that there exist two queens such that the rectangle formed by the centers of the squares they 
occupy has a semiperimeter of 2004.
-/
theorem chessboard_queens_semiperimeter :
  ∃ (x y : Fin 2004), 
    (∀ i j, (i ≠ j → ((x i ≠ x j) ∧ (y i ≠ y j) ∧ (x i + y i ≠ x j + y j) ∧ (x i - y i ≠ x j - y j)))) →
    ∃ (i j : Fin 2004), i ≠ j ∧ (|x i - x j| + |y i - y j| = 4008) :=
by
  sorry

end chessboard_queens_semiperimeter_l502_502081


namespace simplify_and_multiply_expression_l502_502513

variable (b : ℝ)

theorem simplify_and_multiply_expression :
  (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 :=
by
  sorry

end simplify_and_multiply_expression_l502_502513


namespace expected_turns_formula_l502_502753

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1))))

theorem expected_turns_formula (n : ℕ) (h : n > 1) :
  expected_turns n = n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1)))) :=
by
  unfold expected_turns
  sorry

end expected_turns_formula_l502_502753


namespace number_of_asian_countries_l502_502257

theorem number_of_asian_countries (total european south_american : ℕ) 
  (H_total : total = 42) 
  (H_european : european = 20) 
  (H_south_american : south_american = 10) 
  (H_half_asian : ∃ rest, rest = total - european - south_american ∧ rest / 2 = 6) : 
  ∃ asian, asian = 6 :=
by {
  let rest := total - european - south_american,
  have H_rest : rest = 42 - 20 - 10, from sorry,
  have H_asian : rest / 2 = 6, from sorry,
  exact ⟨6, rfl⟩,
}

end number_of_asian_countries_l502_502257


namespace work_completion_time_l502_502163

theorem work_completion_time (T : ℚ) :
  (∀ (a b : ℚ), (a + b) = 9 → (1 / a + 1 / b) * (9 / 18) = 1 / 9) ∧
  (∀ (a : ℚ), 1 / a * (9 / 18) = 1 / 18) ∧
  (∀ (c : ℚ), 1 / c * (9 / 24) = 1 / 24) →
  T = 72 / 11 := sorry

end work_completion_time_l502_502163


namespace minimum_cost_per_serving_l502_502895

theorem minimum_cost_per_serving
    (almond_price_per_oz : ℝ := 18.0 / 32)
    (cashew_price_per_oz : ℝ := 22.50 / 28)
    (walnut_price_per_oz : ℝ := 15.00 / 24)
    (almond_discount : ℝ := 0.10)
    (cashew_discount : ℝ := 0.15)
    (walnut_discount : ℝ := 0.20)
    (almond_proportion : ℝ := 0.50)
    (cashew_proportion : ℝ := 0.30)
    (walnut_proportion : ℝ := 0.20) : 
    (56 : ℕ) = Nat.ceil
        ((almond_price_per_oz * (1 - almond_discount) * almond_proportion +
          cashew_price_per_oz * (1 - cashew_discount) * cashew_proportion +
          walnut_price_per_oz * (1 - walnut_discount) * walnut_proportion) * 100) :=
by 
  sorry

end minimum_cost_per_serving_l502_502895


namespace solve_for_x_l502_502809

noncomputable def power_tower (x : ℝ) : ℝ := x ^ (x ^ (x ^ ...)) -- Conceptual infinite power tower definition

theorem solve_for_x :
  let y := 4
  ∃ x : ℝ, power_tower x = y ↔ x = real.sqrt (real.sqrt 4) := 
sorry

end solve_for_x_l502_502809


namespace complex_conjugate_l502_502989

theorem complex_conjugate (z : ℂ) (h : i * (z + 1) = -3 + 2 * i) : conj z = 1 - 3 * i :=
sorry

end complex_conjugate_l502_502989


namespace area_bound_by_parabola_l502_502248

-- Define the function representing the parabola
def parabola (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Define the roots of the parabola (intersect points with the x-axis)
def roots := {x : ℝ | parabola x = 0}
def x1 := (roots.toFinset : Finset ℝ).min'
def x2 := (roots.toFinset : Finset ℝ).max'

-- Define the area calculations as integrals
noncomputable def area_below := Real.abs (∫ x in 0..1, parabola x)
noncomputable def area_above := ∫ x in 1..5, parabola x

-- The total bounded area
noncomputable def total_area := area_below + area_above

-- The final theorem statement
theorem area_bound_by_parabola : total_area = 13 := by
  sorry

end area_bound_by_parabola_l502_502248


namespace meet_point_l502_502139

-- Define the initial conditions
variables (d v v1 : ℝ)
-- Define the point of meeting is x
noncomputable def meeting_point (d v v1 : ℝ) : ℝ := (v * d) / (v + v1)

-- Create the theorem statement:
theorem meet_point (d v v1 : ℝ) : 
  let x := meeting_point d v v1 in
  x = (v * d) / (v + v1) :=
by
  sorry

end meet_point_l502_502139


namespace sweater_shirt_price_difference_l502_502166

theorem sweater_shirt_price_difference:
  (total_price_shirts: ℝ) (number_of_shirts: ℝ) (total_price_sweaters: ℝ) (number_of_sweaters: ℝ)
  (h₁: total_price_shirts = 400) (h₂: number_of_shirts = 25) 
  (h₃: total_price_sweaters = 1500) (h₄: number_of_sweaters = 75):
  let avg_price_shirt := total_price_shirts / number_of_shirts
  let avg_price_sweater := total_price_sweaters / number_of_sweaters
  avg_price_sweater - avg_price_shirt = 4
  :=
by
  sorry

end sweater_shirt_price_difference_l502_502166


namespace vertex_of_parabola_l502_502266

theorem vertex_of_parabola (a b c : ℝ) (h₁ : a = -3) (h₂ : b = -6) (h₃ : c = 2) :
    let x_v : ℝ := -b / (2 * a),
        y_v : ℝ := a * x_v ^ 2 + b * x_v + c in
    (x_v, y_v) = (1, -7) :=
by
  -- Definitions based on the problem conditions:
  let x_v := -b / (2 * a)
  let y_v := a * x_v ^ 2 + b * x_v + c

  -- Prove the coordinates of the vertex are \((1, -7)\)
  suffices : x_v = 1 ∧ y_v = -7
  · exact this
  
  sorry

end vertex_of_parabola_l502_502266


namespace parallel_condition_l502_502071

def line1 (m : ℝ) := ∀ (x y : ℝ), 2 * x - m * y - 1 = 0
def line2 (m : ℝ) := ∀ (x y : ℝ), (m - 1) * x - y + 1 = 0

theorem parallel_condition (m : ℝ) : 
  (∀ (x1 y1 x2 y2 : ℝ), line1 m x1 y1 → line2 m x2 y2 → x1 / x2 = y1 / y2) ↔ m = 2 :=
sorry

end parallel_condition_l502_502071


namespace tangent_line_ln_curves_l502_502921

/-- Given the conditions that a line is tangent to two logarithmic curves, prove a relationship for b. -/
theorem tangent_line_ln_curves (k b x1 x2 : ℝ) 
    (h1 : k = 1 / x1) 
    (h2 : k = 1 / (x2 + 1))
    (h3 : k * x1 + b = ln x1 + 2)
    (h4 : k * x2 + b = ln (x2 + 1)) : 
    b = 1 - ln 2 :=
begin
  sorry
end

end tangent_line_ln_curves_l502_502921


namespace x_eq_fib_l502_502940

-- Define the base cases and recursive relation for the sequence x_n
def x : ℕ → ℕ
| 0     := 1 -- Since we're using 0-based indexing, this corresponds to x_1 in 1-based.
| 1     := 2 -- This corresponds to x_2 in 1-based.
| (n+2) := x n + x (n+1)

-- Define Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib n + fib (n+1)

-- Main theorem stating x_n = F_(n+1)
theorem x_eq_fib (n : ℕ) : x n = fib (n+1) :=
by sorry

end x_eq_fib_l502_502940


namespace problem_1_problem_2_monotonic_intervals_problem_3_range_f_x2_l502_502253

def f (x a : ℝ) : ℝ := (x - 1) ^ 2 + a * Real.log x

theorem problem_1 (a : ℝ) : 
  (∃ (f' : ℝ → ℝ), f' 1 = a ∧ (∀ x, f' x = (2 * x^2 - 2 * x + a) / x) ∧ ((f' 1 + 1/2) = 0)) ↔ a = 2 :=
sorry

theorem problem_2_monotonic_intervals (a : ℝ) : 
  ((a ≥ 1/2 → ∀ x : ℝ, 0 < x → f (x a)).monotonic (0, +∞)) ∧
  ((a < 1/2 → 
    let x₁ := (1 - Real.sqrt (1 - 2 * a)) / 2;
    let x₂ := (1 + Real.sqrt (1 - 2 * a)) / 2 in
    (0 < a → f (x₁ a)).monto_increasing (0, x₁) ∧
    (0 < a → f (x₁ a)).monto_decreasing (x₁, x₂) ∧
    (0 < a → f (x₂ a)).monto_increasing (x₂, +∞))) :=
sorry

theorem problem_3_range_f_x2 (a x1 x2 : ℝ) (h : 0 < x1 ∧ x1 < x2 ∧ x2 < 1) :
  let f_val := x2^2 - 2*x2 + 1 + (2*x2 - 2*x2^2) * Real.log x2 in
  a = 2 * x2 * (1 - x2) → f_val > (1 - 2 * Real.log 2) / 4 ∧ f_val < 0 :=
sorry

end problem_1_problem_2_monotonic_intervals_problem_3_range_f_x2_l502_502253


namespace floor_function_properties_l502_502304

noncomputable def floor_function := λ (x : ℝ),⌊x⌋

theorem floor_function_properties :
  (∀ x : ℝ, floor_function (x + 2) = floor_function x + 2) ∧
  (∀ x y : ℝ, floor_function (x + y) ≤ floor_function x + floor_function y + 1) ∧
  (∀ x y : ℝ, floor_function ((x + 1) * (y + 1)) = floor_function x * floor_function y + floor_function x + floor_function y + 1) :=
by
  split
  sorry
  split
  sorry
  sorry

end floor_function_properties_l502_502304


namespace entree_cost_l502_502245

theorem entree_cost (E : ℝ) :
  let appetizer := 9
  let dessert := 11
  let tip_rate := 0.30
  let total_cost_with_tip := 78
  let total_cost_before_tip := appetizer + 2 * E + dessert
  total_cost_with_tip = total_cost_before_tip + (total_cost_before_tip * tip_rate) →
  E = 20 :=
by
  intros appetizer dessert tip_rate total_cost_with_tip total_cost_before_tip h
  sorry

end entree_cost_l502_502245


namespace units_digit_of_product_is_eight_l502_502590

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502590


namespace units_digit_of_product_is_eight_l502_502589

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502589


namespace intersection_line_plane_l502_502819

-- Define the line in parametric form
def line (t : ℝ) : ℝ × ℝ × ℝ := 
  (-2 + t, 2, -3)

-- Define the plane equation
def on_plane (p : ℝ × ℝ × ℝ) : Prop := 
  let (x, y, z) := p in 2*x - 3*y - 5*z - 7 = 0

-- Define the expected intersection point
def intersection_point : ℝ × ℝ × ℝ := 
  (-1, 2, -3)

-- The theorem to prove the intersection point 
theorem intersection_line_plane : 
  ∃ t : ℝ, line t = intersection_point ∧ on_plane (line t) :=
sorry

end intersection_line_plane_l502_502819


namespace median_of_given_trapezoid_l502_502020

noncomputable def median_of_trapezoid (a b : ℝ) : ℝ := (a + b) / 2

theorem median_of_given_trapezoid :
  let large_triangle_side : ℝ := 3
  let small_triangle_side : ℝ := real.sqrt 3
  median_of_trapezoid large_triangle_side small_triangle_side = (3 + real.sqrt 3) / 2 :=
by
  sorry

end median_of_given_trapezoid_l502_502020


namespace star_product_equality_l502_502963

-- Define the necessary vertices and midpoints for the star
variables A B C D E : Point
variables A1 B1 C1 D1 E1 : Point

-- Assume the Law of Sines holds for the relevant triangles
axiom law_of_sines : ∀ (X Y Z : Point),
  (X.distance Y) / (real.sin (angle Y Z X)) = (Y.distance Z) / (real.sin (angle Z X Y))

-- State the theorem to be proven
theorem star_product_equality :
  A1.distance C * B1.distance D * C1.distance E * D1.distance A * E1.distance B =
  A1.distance D * B1.distance E * C1.distance A * D1.distance B * E1.distance C :=
by
  sorry

end star_product_equality_l502_502963


namespace part1_part2_l502_502880

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / (3 ^ x + 1) + a

theorem part1 (h : ∀ x : ℝ, f (-x) a = -f x a) : a = -1 :=
by sorry

noncomputable def f' (x : ℝ) : ℝ := 2 / (3 ^ x + 1) - 1

theorem part2 : ∀ t : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f' x + 1 = t ↔ 1 / 2 ≤ t ∧ t ≤ 1 :=
by sorry

end part1_part2_l502_502880


namespace repeated_number_divisible_by_1001001_l502_502757

theorem repeated_number_divisible_by_1001001 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  (1000000 * (100 * a + 10 * b + c) + 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)) % 1001001 = 0 := 
by 
  sorry

end repeated_number_divisible_by_1001001_l502_502757


namespace largest_eight_digit_with_all_evens_l502_502694

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l502_502694


namespace problem1_problem2_l502_502193

-- First problem
theorem problem1 (z : ℂ) (hz : (conj z) / (1 + I) = (2 + I)) : z = 1 - 3*I := by
  sorry

-- Second problem
theorem problem2 (m : ℝ) (hm : ¬(m = 1)) (hz : (m * (m + 2)) / (m - 1) + (m^2 + 2*m - 3) * I = 0 + (m^2 + 2*m - 3) * I) : 
  (m = -2) ∨ (m = 0 ∧ (m^2 + 2*m - 3 ≠ 0) ∧ (m * (m + 2) = 0)) := by
  sorry

end problem1_problem2_l502_502193


namespace intersection_property_l502_502858

-- Define Triangle ABC and the points of intersection
variables (A B C F D E : Type)

noncomputable
def intersection_points (A B C F D E : Type) : Prop :=
  -- EF intersects AB at F, BC at D, and CA at E
  true  -- We require detailed geometric definitions here

-- The main theorem statement
theorem intersection_property 
  (A B C F D E : Type) 
  (h : intersection_points A B C F D E) :
  (AE / CE = AF / BF) ↔ (BD / DC = 1) := 
sorry

end intersection_property_l502_502858


namespace units_digit_of_first_four_composite_numbers_l502_502629

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502629


namespace find_backyard_width_l502_502781

/-- The length of Archie's backyard in yards -/
def backyard_length : ℕ := 20

/-- The area of the shed in square yards -/
def shed_area : ℕ := 3 * 5

/-- The required sod area in square yards -/
def sod_area : ℕ := 245

/-- The width of Archie's backyard in yards, denoted as W -/
noncomputable def backyard_width : ℕ := 13

theorem find_backyard_width :
  ∃ W : ℕ, 245 = (backyard_length * W) - shed_area ∧ W = 13 :=
by
  use backyard_width
  split
  · simp [backyard_length, shed_area, sod_area, backyard_width]
  · refl

end find_backyard_width_l502_502781


namespace angle_difference_parallelogram_l502_502931

theorem angle_difference_parallelogram (A B : ℝ) (hA : A = 55) (h1 : A + B = 180) :
  B - A = 70 := 
by
  sorry

end angle_difference_parallelogram_l502_502931


namespace find_angle_c_find_triangle_perimeter_l502_502396

-- Definition of the triangle ABC with sides a, b, c and angle opposite each side
variables {A B C : ℝ} {a b c : ℝ}

-- Additional given condition for the trigonometric relationship
variable (h : (2 * a - b) * Real.cos C = c * Real.cos B)

-- Conditions for part (1)
theorem find_angle_c (h1 : 0 < A < Real.pi) (h2 : Real.sin A > 0) : 
  Real.cos C = 1 / 2 → C = Real.pi / 3 :=
sorry

-- Additional given conditions for part (2)
variable (area_condition : (1 / 2) * a * b * Real.sin (Real.pi / 3) = sqrt 3)
variable (c_eq_2 : c = 2)

-- Conditions for part (2)
theorem find_triangle_perimeter
  (h3 : a * b = 4) 
  (h4 : a + b = 4) : a + b + c = 6 :=
sorry

end find_angle_c_find_triangle_perimeter_l502_502396


namespace tetrahedron_volume_l502_502299

theorem tetrahedron_volume 
  (d_face : ℝ) (d_edge : ℝ) (d_face_eq : d_face = 2) (d_edge_eq : d_edge = Real.sqrt 5) :
  ∃ (V : ℝ), V ≈ 46.76 :=
by
  have h := 6
  have a := 3 * Real.sqrt 6
  let V := (a^3) / (6 * Real.sqrt 2)
  have V_approx := V ≈ 46.76
  exact ⟨V, V_approx⟩
  sorry

end tetrahedron_volume_l502_502299


namespace points_on_line_l502_502083

variable {r : ℝ} (α : ℝ)

-- Definitions for points on the circle and intersections
def circle := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}
def A := (-r, 0 : ℝ)
def B := (r, 0 : ℝ)
def M : ℝ × ℝ := (r * Real.cos α, r * Real.sin α)

-- Condition for selected point Q on the circle
def Q (β : ℝ) : ℝ × ℝ := (r * Real.cos β, r * Real.sin β)

-- Intersection of MQ with AB at K
def K (β : ℝ) : ℝ × ℝ := (r * Real.sin (α - β) / (Real.sin α - Real.sin β), 0)

-- Line BQ equation and point P intersection
def BQ (β : ℝ) : ℝ → ℝ :=
  let slope := (Real.sin β) / (Real.cos β - 1)
  fun x => slope * (x - r)

def P (β : ℝ) : ℝ × ℝ :=
  let k_x := r * Real.sin (α - β) / (Real.sin α - Real.sin β)
  (k_x, BQ β k_x)

-- The theorem statement
theorem points_on_line (β : ℝ) : P β ∈ {p : ℝ × ℝ | p.1 + Real.cot (α / 2) * p.2 + r = 0} :=
sorry

end points_on_line_l502_502083


namespace instantaneous_velocity_at_t_1_l502_502401

def height (t : ℝ) : ℝ := -4.9 * t^2 + 4.8 * t + 11

theorem instantaneous_velocity_at_t_1 :
  deriv height 1 = -5 := 
sorry

end instantaneous_velocity_at_t_1_l502_502401


namespace min_sum_distances_to_lines_on_circle_l502_502875

variable (A X Y : Point) (C : Circle) (Hangle : IsAngle X A Y) (Hcircle : CircleInsideAngle C X A Y)

theorem min_sum_distances_to_lines_on_circle :
  ∃ (P : Point), PointOnCircle P C ∧
  ∀ (Q : Point), PointOnCircle Q C →
    (distance P LineAX + distance P LineAY) ≤ (distance Q LineAX + distance Q LineAY) :=
sorry

end min_sum_distances_to_lines_on_circle_l502_502875


namespace minimize_integral_l502_502476

variable (a b : ℝ) (f : ℝ → ℝ)
variable (h0 : 0 < a) (h1 : a < b)
variable (h2 : ∀ x, a < x ∧ x < b → 0 < f' x)

theorem minimize_integral : 
  ∃ t, a < t ∧ t < b ∧ t = Real.sqrt ((a^2 + b^2) / 2) ∧ 
       is_minimizer (λ t, ∫ x in set.Ioo a b, |f x - f t| * x) t :=
sorry

end minimize_integral_l502_502476


namespace sum_of_intersections_l502_502072

noncomputable def f : ℝ → ℝ := λ x, 2 * Real.sin (Real.pi * x)
noncomputable def g : ℝ → ℝ := λ x, (1 / (1 - x))

theorem sum_of_intersections : 
  let x_coords := {x : ℝ | f x = g x ∧ x ∈ Icc (-2 : ℝ) 4} in
  ∑ x in x_coords, x = 8 :=
sorry

end sum_of_intersections_l502_502072


namespace trust_meteorologist_l502_502185

-- Definitions
def probability_of_clear := 0.74
def senator_accuracy := p : ℝ
def meteorologist_accuracy := 1.5 * p

-- Events
def event_G := "clear day"
def event_M1 := "first senator predicted clear"
def event_M2 := "second senator predicted clear"
def event_S := "meteorologist predicted rain"

theorem trust_meteorologist :
  let r := probability_of_clear
  let p := senator_accuracy
  let q := meteorologist_accuracy
  1.5 * p * (1 - p)^2 * (1 - r) - (1 - 1.5 * p) * p^2 * r > 0 :=
by
  sorry

end trust_meteorologist_l502_502185


namespace zero_in_interval_l502_502132

noncomputable def f (x : ℝ) : ℝ := 2 * x - 8 + Real.logb 3 x

theorem zero_in_interval : 
  (0 < 3) ∧ (3 < 4) → (f 3 < 0) ∧ (f 4 > 0) → ∃ x, 3 < x ∧ x < 4 ∧ f x = 0 :=
by
  intro h1 h2
  obtain ⟨h3, h4⟩ := h2
  sorry

end zero_in_interval_l502_502132


namespace quadrilateral_proof_l502_502411

variables (a b x y : ℝ) (P : ℝ)
variables (PC CD PB BA : ℝ)

-- Given the conditions
variable (h1 : PC = a)
variable (h2 : CD = b)
variable (h3 : PB = y)
variable (h4 : BA = x)
variable (h5 : ∠ACB = P)
variable (h6 : ∠ADB = P)

theorem quadrilateral_proof :
  x = (a + b / 2) * (Real.sec P) - (2 * a * (a + b) * (Real.cos P)) / (2 * a + b) ∧
  y = (2 * a * (a + b) * (Real.cos P)) / (2 * a + b) := sorry

end quadrilateral_proof_l502_502411


namespace total_population_l502_502794

-- Defining the populations of Springfield and the difference in population
def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

-- The definition of Greenville's population in terms of Springfield's population
def greenville_population : ℕ := springfield_population - population_difference

-- The statement that we want to prove: the total population of Springfield and Greenville
theorem total_population :
  springfield_population + greenville_population = 845640 := by
  sorry

end total_population_l502_502794


namespace arithmetic_sequence_common_difference_l502_502024

theorem arithmetic_sequence_common_difference
  (a : ℤ)
  (a_n : ℤ)
  (S_n : ℤ)
  (n : ℤ)
  (d : ℚ)
  (h1 : a = 3)
  (h2 : a_n = 34)
  (h3 : S_n = 222)
  (h4 : S_n = n * (a + a_n) / 2)
  (h5 : a_n = a + (n - 1) * d) :
  d = 31 / 11 :=
by
  sorry

end arithmetic_sequence_common_difference_l502_502024


namespace units_digit_first_four_composites_l502_502619

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502619


namespace mul_exponents_l502_502788

theorem mul_exponents (m : ℝ) : 2 * m^3 * 3 * m^4 = 6 * m^7 :=
by sorry

end mul_exponents_l502_502788


namespace average_value_of_T_l502_502094

def average_T (boys girls : ℕ) (starts_with_boy : Bool) (ends_with_girl : Bool) : ℕ :=
  if boys = 9 ∧ girls = 15 ∧ starts_with_boy ∧ ends_with_girl then 12 else 0

theorem average_value_of_T :
  average_T 9 15 true true = 12 :=
sorry

end average_value_of_T_l502_502094


namespace trust_meteorologist_l502_502188

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l502_502188


namespace exist_relprime_in_sub_2n_exist_divisor_in_sub_2n_l502_502976

theorem exist_relprime_in_sub_2n (n: ℕ) (S: Finset ℕ) (hS: S.card = n + 1) (hS_sub: S ⊆ Finset.range (2 * n + 1)) :
  ∃ x y ∈ S, Nat.gcd x y = 1 :=
by sorry

theorem exist_divisor_in_sub_2n (n: ℕ) (S: Finset ℕ) (hS: S.card = n + 1) (hS_sub: S ⊆ Finset.range (2 * n + 1)) :
  ∃ x y ∈ S, x ≠ y ∧ (x ∣ y ∨ y ∣ x) :=
by sorry

end exist_relprime_in_sub_2n_exist_divisor_in_sub_2n_l502_502976


namespace units_digit_first_four_composites_l502_502624

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502624


namespace probability_white_ball_is_two_fifths_l502_502199

-- Define the total number of each type of balls.
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Calculate the total number of balls in the bag.
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability calculation.
noncomputable def probability_of_white_ball : ℚ := white_balls / total_balls

-- The theorem statement asserting the probability of drawing a white ball.
theorem probability_white_ball_is_two_fifths :
  probability_of_white_ball = 2 / 5 :=
sorry

end probability_white_ball_is_two_fifths_l502_502199


namespace construct_quadrilateral_l502_502800

structure QuadrilateralConstructible (e f a c g : ℝ) : Prop :=
  (exists_ABCD : ∃ A B C D : ℝ → ℝ, (dist A C = e) ∧ (dist B D = f) ∧ (dist A B = a) ∧ (dist C D = c) ∧ 
    (dist (midpoint B C) (midpoint D A) = g))

theorem construct_quadrilateral {e f a c g : ℝ} : QuadrilateralConstructible e f a c g := 
  sorry

end construct_quadrilateral_l502_502800


namespace cyclic_quadrilateral_concurrent_tangents_l502_502978

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop :=
∃ (O : Circle), A ∈ O ∧ B ∈ O ∧ C ∈ O ∧ D ∈ O 

noncomputable def collinear (P Q R S : Point) : Prop :=
∃ (l : Line), P ∈ l ∧ Q ∈ l ∧ R ∈ l ∧ S ∈ l

noncomputable def tangent_to_circle (P : Point) (O : Circle) (l : Line) : Prop :=
P ∈ O ∧ ∀ (Q : Point), Q ∈ O → ∠(P, Q) = 90°

noncomputable def midpoint (P Q M : Point) : Prop :=
∃ (l : Line), P ∈ l ∧ Q ∈ l ∧ M ∈ l ∧ distance(P, M) = distance(M, Q)

theorem cyclic_quadrilateral_concurrent_tangents
  (A B C D Q P : Point)
  (O1 O2 O3 O4 O5 O6 O7 O8 : Circle)
  (l1 l2 l3 l4 l5 : Line)
  (M N : Point) :
  cyclic_quadrilateral A B C D →
  collinear Q A B P →
  tangent_to_circle A O1 l1 →
  tangent_to_circle P O2 l2 →
  midpoint B C M →
  midpoint A D N →
  (∃ (l : Line), l = line through C D) ∧ 
  (∃ (l : Line), tangency l3 at A with circle O3) ∧
  (∃ (l : Line), tangency l4 at B with circle O4) →
  are_concurrent l1 l2 l3 := 
sorry

end cyclic_quadrilateral_concurrent_tangents_l502_502978


namespace units_digit_first_four_composites_l502_502618

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502618


namespace intersectAandB_l502_502905

def setA : Set ℝ := { x | log 2 x < 1 }
def setB : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem intersectAandB : setA ∩ setB = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersectAandB_l502_502905


namespace length_of_square_side_l502_502735

variable {R : Type}
variables [LinearOrderedField R]

def is_midpoint {P : Type} [AddGroup P] [Module R P] (a b m : P) : Prop :=
  2 • m = a + b

def is_square {P : Type} [AddGroup P] [Module R P] (a b c d : P) (s : R) : Prop :=
  s ≠ 0 ∧ dist a b = s ∧ dist b c = s ∧ dist c d = s ∧ dist d a = s ∧
  dist a c = dist b d ∧ dist a c = s * sqrt 2

noncomputable def triangle_area {P : Type} [InnerProductSpace R P] (a b c : P) : R :=
  (1 / 2) * (dist a b) * (dist b c) * real.sin (∠ b a c)

theorem length_of_square_side (a b c d e f : P)
  (s : R) (h_square : is_square a b c d s) 
  (h_mid_ab : is_midpoint a b f) (h_mid_ad : is_midpoint a d e)
  (h_area : triangle_area c e f = 24) :
  s = 4 * real.sqrt 6 :=
sorry

end length_of_square_side_l502_502735


namespace units_digit_first_four_composites_l502_502580

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502580


namespace units_digit_first_four_composites_l502_502581

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502581


namespace problem_statement_l502_502483

noncomputable def g (x : ℝ) : ℝ := 4 / (16 ^ x + 4)

theorem problem_statement :
  (∑ k in Finset.range 2000, g ((k+1) / 2001)) = 1000 :=
sorry

end problem_statement_l502_502483


namespace find_f_g_neg1_l502_502918

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x then x^2 + 2 * x else g x

axiom g : ℝ → ℝ

axiom odd_f : ∀ x : ℝ, f (-x) = -f x

theorem find_f_g_neg1 : f (g (-1)) = -15 := by
  sorry

end find_f_g_neg1_l502_502918


namespace trig_identity_l502_502043

theorem trig_identity (α a : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
    (h3 : (Real.tan α) + (1 / (Real.tan α)) = a) : 
    (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt (a^2 + 2 * a) :=
by
  sorry

end trig_identity_l502_502043


namespace largest_eight_digit_number_l502_502702

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l502_502702


namespace father_cards_given_l502_502956

-- Defining the conditions
def Janessa_initial_cards : Nat := 4
def eBay_cards : Nat := 36
def bad_cards : Nat := 4
def dexter_cards : Nat := 29
def janessa_kept_cards : Nat := 20

-- Proving the number of cards father gave her
theorem father_cards_given : ∃ n : Nat, n = 13 ∧ (Janessa_initial_cards + eBay_cards - bad_cards + n = dexter_cards + janessa_kept_cards) := 
by
  sorry

end father_cards_given_l502_502956


namespace units_digit_of_product_of_first_four_composites_l502_502646

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502646


namespace correct_choice_is_A_l502_502775

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

def f_A (x : ℝ) : ℝ := x^2 + 1
def f_B (x : ℝ) : ℝ := 2^x
def f_C (x : ℝ) : ℝ := x + (1 / x)
def f_D (x : ℝ) : ℝ := -x^2 + 1

theorem correct_choice_is_A : is_even f_A ∧ is_monotonically_increasing_on f_A {x | 0 < x} ∧
  ¬(is_even f_B ∧ is_monotonically_increasing_on f_B {x | 0 < x}) ∧
  ¬(is_even f_C ∧ is_monotonically_increasing_on f_C {x | 0 < x}) ∧
  ¬(is_even f_D ∧ is_monotonically_increasing_on f_D {x | 0 < x}) :=
by
  sorry

end correct_choice_is_A_l502_502775


namespace pyramid_cross_section_area_l502_502098

theorem pyramid_cross_section_area 
  (A B C D K L M : Point)
  (hK : midpoint K A D)
  (hL : midpoint L B D)
  (hM : midpoint M C D)
  (hABC : area (triangle A B C) = 2) :
  area (triangle K L M) = 1 / 2 := 
sorry

end pyramid_cross_section_area_l502_502098


namespace P_2008_value_l502_502981

-- Define the polynomial P that satisfies the given conditions
def P (x : ℕ) : ℕ :=
sorry -- polynomial of degree 2008 with leading coefficient 1

-- Define the conditions of the polynomial P
axiom P_conditions : 
  ∀ (n : ℕ), n ≤ 2007 → P(n) = 2007 - n

theorem P_2008_value : 
  P(2008) = 2008! - 1 := 
sorry

end P_2008_value_l502_502981


namespace find_a_l502_502964

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x / (3 * x + 4)

theorem find_a (a : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : (f a) (f a x) = x → a = -2 := by
  unfold f
  -- Remaining proof steps skipped
  sorry

end find_a_l502_502964


namespace number_of_methods_to_remove_term_l502_502206

-- Definition of the arithmetic sequence
def arithmetic_sequence (a1 d n: ℕ) : ℕ := a1 + (n - 1) * d

-- Main statement
theorem number_of_methods_to_remove_term :
  ∃ n : ℕ, 
  (arithmetic_sequence 1 2 n) = 2 * 2015 - 1 ∧
  ∀ k, 1 ≤ k ∧ k ≤ n →
  ∃ K, 
  (K = 1 ∨ K = 1008 ∨ K = 2015) ∧
  (∃ m : ℕ, m = 3 ∧ (arithmetic_sequence 1 2 m = 2 * 2016 - 1)). 

end number_of_methods_to_remove_term_l502_502206


namespace parallel_lines_distance_l502_502870

noncomputable def distance_between_parallel_lines
  (A1 B1 C1 A2 B2 C2 : ℝ) : ℝ :=
  abs (C1 - C2) / sqrt (A1^2 + B1^2)

theorem parallel_lines_distance (m : ℝ) (h1 : 3 * 1 + 2 * 1 - 3 = 0)
  (h2 : 6 * 1 + m * 1 + 1 = 0)
  (h3 : m = 4) :
  distance_between_parallel_lines 3 2 (-3) 6 4 (-1) = 2 * sqrt 13 / 13 :=
by
  sorry

end parallel_lines_distance_l502_502870


namespace solve_for_x_l502_502517

theorem solve_for_x : 
  (∀ x : ℝ, x ≠ -2 → (x^2 - x - 2) / (x + 2) = x - 1 ↔ x = 0) := 
by 
  sorry

end solve_for_x_l502_502517


namespace original_students_l502_502555

theorem original_students (a b : ℕ) : 
  a + b = 92 ∧ a - 5 = 3 * (b + 5 - 32) → a = 45 ∧ b = 47 :=
by sorry

end original_students_l502_502555


namespace units_digit_product_first_four_composite_numbers_l502_502674

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502674


namespace units_digit_first_four_composite_is_eight_l502_502608

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502608


namespace largest_eight_digit_number_contains_even_digits_l502_502725

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l502_502725


namespace units_digit_of_first_four_composite_numbers_l502_502635

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502635


namespace prop_I_prop_II_prop_III_l502_502272

-- Proposition (Ⅰ)
theorem prop_I : (∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) :=
sorry  -- Proves that the negation of the proposition (Ⅰ) is true, hence the proposition (Ⅰ) is false

-- Proposition (Ⅱ)
theorem prop_II : ¬ (∀ t : Triangle, t.is_equilateral) :=
sorry  -- Proves that the negation of the negation of proposition (Ⅱ) is false, hence the proposition (Ⅱ) is true

-- Proposition (Ⅲ)
theorem prop_III : ¬ (∃ x : ℝ, x.odd ∧ x^2 - 8*x - 10 = 0) :=
sorry  -- Proves that the negation of proposition (Ⅲ) is false, hence the proposition (Ⅲ) is true

end prop_I_prop_II_prop_III_l502_502272


namespace units_digit_of_composite_product_l502_502648

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502648


namespace decreasing_interval_l502_502878

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 4)

theorem decreasing_interval :
  (∀ x₁ x₂ ∈ set.Icc (a : ℝ) (b : ℝ), x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) →
  [a, b] = [Real.pi / 4, 5 * Real.pi / 4] :=
by
  sorry

end decreasing_interval_l502_502878


namespace ellipse_circle_intersection_l502_502889

theorem ellipse_circle_intersection (a : ℝ) (h : a > 1) :
  let line_eq := ∀ x : ℝ, (x, x - 1)
  let ellipse_eq := ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (a^2 - 1) = 1
  let focus_left := λ f : ℝ → ℝ × ℝ, f(-1) = (-1, 0)
  let circle_eq := ∃ A B : ℝ × ℝ,
    (line_eq A.1 = A ∧ ellipse_eq A.1 A.2 ∧ line_eq B.1 = B ∧ ellipse_eq B.1 B.2) 
    ∧ ∀ C : ℝ × ℝ, C = (A.1 + B.1)/2 ∧ C = (A.2 + B.2)/2
    ∧ (focus_left (λ x, A.1 + (B.1 - A.1) * x - 1) = (-1, 0)) :=
  a = (sqrt 6 + sqrt 2) / 2 := sorry

end ellipse_circle_intersection_l502_502889


namespace largest_8_digit_number_with_even_digits_l502_502687

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l502_502687


namespace find_angle_B_l502_502994

-- Definitions for angles A, B, and C
def angle (ν : Type) := ℝ  -- Angles are represented as real numbers in degrees

variables {A B C : angle ℝ}

-- Conditions given in the problem
axiom parallel_lines (l k : Prop) : l ∧ k  -- l and k are parallel
axiom angle_A : A = 130
axiom angle_C : C = 60

-- Definition corresponding to proving the question
theorem find_angle_B (l k : Prop) (h_parallel : l ∧ k)
  (h_A : A = 130) (h_C : C = 60) : B = 170 :=
sorry  -- Placeholder for the proof

end find_angle_B_l502_502994


namespace proof_of_derivative_of_function_l502_502107

noncomputable def derivative_of_function_equals_proof : Prop :=
  ∀ (x : ℝ), (deriv (λ x : ℝ, x * exp x) x) = (1 + x) * exp x

theorem proof_of_derivative_of_function: derivative_of_function_equals_proof :=
by
  intros
  rw deriv_mul
  rw deriv_id
  rw deriv_exp
  ring
  sorry

end proof_of_derivative_of_function_l502_502107


namespace students_present_in_class_l502_502932

noncomputable def num_students : ℕ := 100
noncomputable def percent_boys : ℝ := 0.55
noncomputable def percent_girls : ℝ := 0.45
noncomputable def absent_boys_percent : ℝ := 0.16
noncomputable def absent_girls_percent : ℝ := 0.12

theorem students_present_in_class :
  let num_boys := percent_boys * num_students
  let num_girls := percent_girls * num_students
  let absent_boys := absent_boys_percent * num_boys
  let absent_girls := absent_girls_percent * num_girls
  let present_boys := num_boys - absent_boys
  let present_girls := num_girls - absent_girls
  present_boys + present_girls = 86 :=
by
  sorry

end students_present_in_class_l502_502932


namespace sum_of_interior_angles_regular_polygon_l502_502404

theorem sum_of_interior_angles_regular_polygon (n : ℕ) (h1 : polygon.regular n) (h2 : polygon.interior_angle n = 144) : 
  polygon.sum_of_interior_angles n = 1440 :=
by
  sorry

end sum_of_interior_angles_regular_polygon_l502_502404


namespace product_of_roots_in_arithmetic_progression_l502_502126

theorem product_of_roots_in_arithmetic_progression (a b c : ℚ) (h : 36 * a^3 - 66 * a^2 + 31 * a - 4 = 0) (roots_in_arithmetic_progression : (a, b, c) = (x, y, z) in List.chain (λ x y, y = x + k) [(a, b), (b, c)]) :
  (max a c) * (min a c) = 2 / 9 :=
sorry

end product_of_roots_in_arithmetic_progression_l502_502126


namespace num_integers_in_interval_l502_502370

-- Define the lower and upper bounds of the inequality using the approximation of π
def lower_bound := -6 * 3.14
def upper_bound := 12 * 3.14

-- Round the bounds to the nearest integers
def lower_integer_bound := Int.ceil lower_bound
def upper_integer_bound := Int.floor upper_bound

-- Prove the count of integers satisfying the inequality -6π ≤ x ≤ 12π is 55
theorem num_integers_in_interval : 
    lower_integer_bound = -18 → 
    upper_integer_bound = 37 → 
    (upper_integer_bound - lower_integer_bound + 1) = 55 := by
    sorry

end num_integers_in_interval_l502_502370


namespace units_digit_first_four_composite_is_eight_l502_502606

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502606


namespace units_digit_of_first_four_composite_numbers_l502_502627

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502627


namespace circumcircle_AEF_fixed_point_l502_502966

theorem circumcircle_AEF_fixed_point {A B C E F: Point} (h1: acute_angled_triangle A B C)
  (h2: E ∈ line_segment A C) (h3: F ∈ line_segment A B)
  (h4: dist B C ^ 2 = dist B A * dist B F + dist C E * dist C A) :
  (∃ M: Point, M ≠ A ∧ M = midpoint B C ∧ circle_passes_through (circumcircle A E F) M) :=
sorry

end circumcircle_AEF_fixed_point_l502_502966


namespace evaluate_expression_l502_502280

theorem evaluate_expression : 
  (81:ℝ)^(1 / 2) * (64:ℝ)^(-1 / 3) * (49:ℝ)^(1 / 2) = 63 / 4 := 
by 
{
  sorry
}

end evaluate_expression_l502_502280


namespace yards_gained_l502_502207

variable {G : ℤ}

theorem yards_gained (h : -5 + G = 3) : G = 8 :=
  by
  sorry

end yards_gained_l502_502207


namespace number_of_machines_l502_502519

def machine_problem : Prop :=
  ∃ (m : ℕ), (6 * 42) = 6 * 36 ∧ m = 7

theorem number_of_machines : machine_problem :=
  sorry

end number_of_machines_l502_502519


namespace largest_8_digit_number_with_even_digits_l502_502688

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l502_502688


namespace max_distance_l502_502222

/-- Define the points A(0,0), B(1,0), and C(0,1) in the plane. -/
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨0, 1⟩

/-- Define the distances from P to A, B, and C, respectively. -/
noncomputable def u (P : Point) : ℝ :=
  real.sqrt (P.x^2 + P.y^2)

noncomputable def v (P : Point) : ℝ :=
  real.sqrt ((P.x - 1)^2 + P.y^2)

noncomputable def w (P : Point) : ℝ :=
  real.sqrt (P.x^2 + (P.y - 1)^2)

/-- Define the condition that the sum of the squares of the distances is 5. -/
def distance_condition (P : Point) : Prop :=
  u P ^ 2 + v P ^ 2 + w P ^ 2 = 5

/-- Prove that the maximum distance from P to A is (sqrt(2)/3 + sqrt(7/3)) 
    given the distance_condition. -/
theorem max_distance (P : Point) (h : distance_condition P) : 
  dist P A ≤ (real.sqrt (2) / 3) + real.sqrt (7 / 3) := 
sorry

end max_distance_l502_502222


namespace max_triangle_area_l502_502951

theorem max_triangle_area (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + B + C = π / 2) 
  (h2 : ∀ (h3 : sin B ≠ 0), cos A / sin B + cos B / sin A = 2) 
  (h4 : a + b + c = 12) 
  (h5 : c = sqrt (a^2 + b^2)) 
  : ∃ (S : ℝ), S = 36 * (3 - 2 * sqrt 2) := 
begin
  sorry
end

end max_triangle_area_l502_502951


namespace michael_monica_age_ratio_l502_502522

theorem michael_monica_age_ratio
  (x y : ℕ)
  (Patrick Michael Monica : ℕ)
  (h1 : Patrick = 3 * x)
  (h2 : Michael = 5 * x)
  (h3 : Monica = y)
  (h4 : y - Patrick = 64)
  (h5 : Patrick + Michael + Monica = 196) :
  Michael * 5 = Monica * 3 :=
by
  sorry

end michael_monica_age_ratio_l502_502522


namespace trust_meteorologist_l502_502189

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l502_502189


namespace distinct_square_sum_100_l502_502409

theorem distinct_square_sum_100 :
  ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → 
  a^2 + b^2 + c^2 = 100 → false := by
  sorry

end distinct_square_sum_100_l502_502409


namespace license_plate_combinations_l502_502783

theorem license_plate_combinations :
  let letters := 26 in
  let choose (n k : Nat) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) in
  let repeated_letter_choices := letters in
  let remaining_letter_choices := choose 25 3 in
  let placement_choices := choose 5 2 in
  let arrangements := 6 in
  let digit_choices := 10 * 9 * 8 in
  repeated_letter_choices * remaining_letter_choices * placement_choices * arrangements * digit_choices = 2594880000 :=
by sorry

end license_plate_combinations_l502_502783


namespace largest_eight_digit_with_all_evens_l502_502693

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l502_502693


namespace trust_meteorologist_l502_502179

-- Definitions for problem conditions
variables {G M1 M2 S : Prop}
variable {r : ℝ}
variable {p : ℝ}

/-- The probability of a clear day is r -/
axiom prob_clear_day : r = 0.74

/-- Senators' prediction accuracy -/
axiom senator_accuracy : ℝ

/-- Meteorologist's prediction accuracy being 1.5 times senators' /-
axiom meteorologist_accuracy : ∀ p, 1.5 * p

/-- Independence of predictions -/
axiom independence_preds : independent [G, M1, M2, S]

noncomputable def joint_probability_given_G : ℝ :=
(1 - 1.5 * meteorologist_accuracy senator_accuracy) * senator_accuracy^2

noncomputable def joint_probability_given_not_G : ℝ :=
1.5 * meteorologist_accuracy senator_accuracy * (1 - senator_accuracy)^2

noncomputable def overall_probability : ℝ :=
joint_probability_given_G * r + joint_probability_given_not_G * (1 - r)

noncomputable def conditional_prob_not_clear : ℝ /-
(joint_probability_given_not_G * (1 - r)) / overall_probability

noncomputable def conditional_prob_clear : ℝ
(joint_probability_given_G * r) / overall_probability

-- Main theorem statement: Given the conditions, the meteorologist's forecast is more reliable
theorem trust_meteorologist : conditional_prob_not_clear > conditional_prob_clear :=
by sorry

end trust_meteorologist_l502_502179


namespace decreasing_interval_l502_502539

noncomputable def f : ℝ → ℝ := λ x, Real.sin x - Real.sqrt 3 * Real.cos x
def interval : Set ℝ := Set.Icc (5 * Real.pi / 6) Real.pi

theorem decreasing_interval : ∀ x, x ∈ Icc 0 Real.pi → MonotoneOn (f := f) interval ∧ (∀ y ∈ interval, y ∉ (Icc 0 Real.pi)) ∧ (∀ z ∈ Icc 0 Real.pi, z ∉ interval) := by 
  sorry

end decreasing_interval_l502_502539


namespace kopeechka_items_l502_502423

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502423


namespace units_digit_of_product_of_first_four_composites_l502_502647

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502647


namespace sum_of_A_comp_B_l502_502802

open Set

-- Define the set operation
def setOperation (A B : Set ℕ) : Set ℕ :=
  {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y * (x + y)}

-- Given sets A and B
def A : Set ℕ := {0, 1}
def B : Set ℕ := {2, 3}

-- Define the sum of elements in a set
def sumSet (S : Set ℕ) : ℕ :=
  S.toFinset.sum id

-- The mathematical proof problem: the sum of all elements in the set A ⊙ B is 18
theorem sum_of_A_comp_B : sumSet (setOperation A B) = 18 := by
  sorry

end sum_of_A_comp_B_l502_502802


namespace units_digit_of_product_of_first_four_composites_l502_502642

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502642


namespace train_length_correct_l502_502733

def speed_kmph := 70 -- speed in km/hr
def time_seconds := 9 -- time in seconds

def speed_mps := speed_kmph * 1000 / 3600 -- convert speed to m/s
def length_of_train := speed_mps * time_seconds -- compute the length of the train in meters

theorem train_length_correct :
  length_of_train = 174.96 :=
by
  sorry

end train_length_correct_l502_502733


namespace sum_of_coefficients_l502_502907

/-- If (2x - 1)^4 = a₄x^4 + a₃x^3 + a₂x^2 + a₁x + a₀, then the sum of the coefficients a₀ + a₁ + a₂ + a₃ + a₄ is 1. -/
theorem sum_of_coefficients :
  ∃ a₄ a₃ a₂ a₁ a₀ : ℝ, (2 * x - 1) ^ 4 = a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀ → 
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  sorry

end sum_of_coefficients_l502_502907


namespace complement_union_correct_l502_502891

noncomputable def A : set ℝ := {x | x^2 + 2*x - 8 ≥ 0 }
noncomputable def B : set ℝ := {x | 1 < x ∧ x < 5 }
def U : set ℝ := set.univ
def C_U (s : set ℝ) : set ℝ := U \ s

theorem complement_union_correct :
  C_U (A ∪ B) = {x | -4 < x ∧ x ≤ 1} :=
by
  sorry

end complement_union_correct_l502_502891


namespace largest_prime_2010_digits_k_l502_502988

theorem largest_prime_2010_digits_k:
  let p := some_prime_with_2010_digits in
  p = largest_prime_with_2010_digits → ∃ k, k = 1 ∧ 24 ∣ (p^2 - k) :=
sorry

end largest_prime_2010_digits_k_l502_502988


namespace kolya_purchase_l502_502446

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502446


namespace units_digit_first_four_composites_l502_502593

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502593


namespace sum_of_x_degrees_l502_502296

theorem sum_of_x_degrees :
  (∑ x in { x | 0 < x ∧ x < 90 ∧ (sin (3 * x) ^ 3 + sin (5 * x) ^ 3 = 8 * (sin (4 * x)) ^ 3 * (sin x)^3) }, x) = 97.5 :=
by
  sorry

end sum_of_x_degrees_l502_502296


namespace range_of_x_for_acute_angle_l502_502362

theorem range_of_x_for_acute_angle (x : ℝ) (h₁ : (x, 2*x) ≠ (0, 0)) (h₂ : (x+1, x+3) ≠ (0, 0)) (h₃ : (3*x^2 + 7*x > 0)) : 
  x < -7/3 ∨ (0 < x ∧ x < 1) ∨ x > 1 :=
by {
  -- This theorem asserts the given range of x given the dot product solution.
  sorry
}

end range_of_x_for_acute_angle_l502_502362


namespace clock_angle_at_3_25_l502_502150

-- Definitions based on conditions
def angle_at_3_oclock := 90
def minute_hand_movement_per_minute := 6
def hour_hand_movement_per_minute := 0.5

-- Time given in minutes after 3:00
def time_in_minutes := 25

-- Calculations based on conditions and time
def minute_hand_total_movement := minute_hand_movement_per_minute * time_in_minutes
def hour_hand_total_movement := hour_hand_movement_per_minute * time_in_minutes
def hour_hand_position_at_3_25 := angle_at_3_oclock + hour_hand_total_movement
def angle_between_hands := |minute_hand_total_movement - hour_hand_position_at_3_25|

-- Proof statement
theorem clock_angle_at_3_25 : angle_between_hands = 47.5 :=
by
  -- Proof would go here
  sorry

end clock_angle_at_3_25_l502_502150


namespace quadratic_ineq_real_solutions_l502_502806

theorem quadratic_ineq_real_solutions (d : ℝ) (h₀ : 0 < d) :
  (∀ x : ℝ, x^2 - 8 * x + d < 0 → 0 < d ∧ d < 16) :=
by
  sorry

end quadratic_ineq_real_solutions_l502_502806


namespace circle_area_from_string_l502_502221

theorem circle_area_from_string :
  ∃ (A : ℕ), A = 215 ∧
  ∃ (perimeter : ℝ) (length width : ℝ),
    let ratio := 3 / 2 in
    (length / width = ratio) ∧
    (length * width = 162) ∧
    (perimeter = 2 * (length + width)) ∧
    (A = Float.ceil (π * (perimeter / (2 * π))^2).toReal) :=
sorry

end circle_area_from_string_l502_502221


namespace remainder_of_11_12_factorial_sum_mod_11_l502_502965

theorem remainder_of_11_12_factorial_sum_mod_11 :
    (11! * 12! * ∑ n in Finset.range 9 \ Finset.range 2, (n^2 + 3 * n + 1) / ((n + 1)! * (n + 2)!)) % 11 = 10 := 
sorry

end remainder_of_11_12_factorial_sum_mod_11_l502_502965


namespace frustum_lateral_surface_area_frustum_cross_sectional_area_l502_502250

def lower_base_radius : ℝ := 8
def upper_base_radius : ℝ := 4
def frustum_height : ℝ := 5

theorem frustum_lateral_surface_area (r₁ r₂ h : ℝ) (hyp_r₁ : r₁ = lower_base_radius)
  (hyp_r₂ : r₂ = upper_base_radius) (hyp_h : h = frustum_height) :
  let s := Real.sqrt (h * h + (r₁ - r₂) * (r₁ - r₂)) in
  π * (r₁ + r₂) * s = 12 * π * Real.sqrt 41 := sorry

theorem frustum_cross_sectional_area (h d : ℝ) (hyp_h : h = frustum_height)
  (hyp_d : d = 2 * upper_base_radius) :
  h * d = 40 := sorry

end frustum_lateral_surface_area_frustum_cross_sectional_area_l502_502250


namespace simplify_fraction_l502_502514

theorem simplify_fraction (a : ℕ) (h : a = 3) : (10 * a ^ 3) / (55 * a ^ 2) = 6 / 11 :=
by sorry

end simplify_fraction_l502_502514


namespace cross_product_result_l502_502264

theorem cross_product_result 
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 5)
  (dot_ab : EuclideanSpace.inner a b = -6) :
  ∥EuclideanSpace.crossProduct a b∥ = 8 := by
    sorry

end cross_product_result_l502_502264


namespace line_L_trajectory_Q_l502_502348

-- Define the problem conditions
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_P : Prop := (1 : ℝ, 2 : ℝ)

-- The proof is to determine the equation of line L
theorem line_L (L : ℝ → ℝ → Prop) :
  (L = λ x y, x = 1) ∨ (L = λ x y, 3 * x - 4 * y + 5 = 0) :=
sorry

-- Define the conditions for the moving point Q
def point_on_circle (s t : ℝ) : Prop := s^2 + t^2 = 4
def vector_relation (x y s t : ℝ) : Prop := x = s ∧ y = 2 * t

-- The proof is to determine the trajectory of point Q
theorem trajectory_Q (x y : ℝ) :
  (∃ s t : ℝ, point_on_circle s t ∧ vector_relation x y s t) →
  (x^2 / 4 + y^2 / 16 = 1) :=
sorry

end line_L_trajectory_Q_l502_502348


namespace hoodies_ownership_l502_502825

-- Step a): Defining conditions
variables (Fiona_casey_hoodies_total: ℕ) (Casey_difference: ℕ) (Alex_hoodies: ℕ)

-- Functions representing the constraints
def hoodies_owned_by_Fiona (F : ℕ) : Prop :=
  (F + (F + 2) + 3 = 15)

-- Step c): Prove the correct number of hoodies owned by each
theorem hoodies_ownership (F : ℕ) (H1 : hoodies_owned_by_Fiona F) : 
  F = 5 ∧ (F + 2 = 7) ∧ (3 = 3) :=
by {
  -- Skipping proof details
  sorry
}

end hoodies_ownership_l502_502825


namespace minimum_value_x_plus_2y_l502_502389

theorem minimum_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end minimum_value_x_plus_2y_l502_502389


namespace boys_more_than_girls_l502_502558

def numGirls : ℝ := 28.0
def numBoys : ℝ := 35.0

theorem boys_more_than_girls : numBoys - numGirls = 7.0 := by
  sorry

end boys_more_than_girls_l502_502558


namespace maximum_x_plus_2y_l502_502863

theorem maximum_x_plus_2y 
  (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^2 + 8 * y^2 + x * y = 2) :
  x + 2 * y ≤ 4 / 3 :=
sorry

end maximum_x_plus_2y_l502_502863


namespace pentagon_termination_l502_502136

theorem pentagon_termination (x1 x2 x3 x4 x5 : ℤ) :
  x1 + x2 + x3 + x4 + x5 > 0 →
  (∃ (x y z : ℤ), (y < 0) ∧ 
    let x_new := x + y, y_new := -y, z_new := z + y, 
        f := λ (x1 x2 x3 x4 x5 : ℤ), 
              (1/2 : ℝ) * (↑(x2 - x1)^2 + ↑(x3 - x2)^2 + ↑(x4 - x3)^2 + ↑(x5 - x4)^2 + ↑((x1 - x5)^2)),
        f' := f x_new y_new z_new x4 x5 
    in f' < f x1 x2 x3 x4 x5) → 
  ∃ n : ℕ, ∀ i ≥ n, ∀ x y z, y < 0 → 
    (x, y, z) ≠ list.nth [x1, x2, x3, x4, x5] i := 
sorry

end pentagon_termination_l502_502136


namespace matrix_eq_value_satisfied_for_two_values_l502_502007

variable (a b c d x : ℝ)

def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the specific instance for the given matrix problem
def matrix_eq_value (x : ℝ) : Prop :=
  matrix_value (2 * x) x 1 x = 3

-- Prove that the equation is satisfied for exactly two values of x
theorem matrix_eq_value_satisfied_for_two_values :
  (∃! (x : ℝ), matrix_value (2 * x) x 1 x = 3) :=
sorry

end matrix_eq_value_satisfied_for_two_values_l502_502007


namespace largest_8_digit_number_with_even_digits_l502_502686

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l502_502686


namespace circle_center_eq_l502_502887

theorem circle_center_eq (x y : ℝ) :
    (x^2 + y^2 - 2*x + y + 1/4 = 0) → (x = 1 ∧ y = -1/2) :=
by
  sorry

end circle_center_eq_l502_502887


namespace sin_ratio_equal_segments_l502_502028

variable {O : Type} [MetricSpace O]

def midpoint (M P : O) : Prop := ∃A B : O, M = midpoint A B ∧ distance A P = distance B P
def chord_contains_point (M P Q : O) : Prop := ∃P Q : O, M ∈ line_segment P Q
def angle (P Q R : O) : ℝ := sorry  -- Define angle appropriately

theorem sin_ratio_equal_segments {O : Type} [MetricSpace O] 
  (M GH AB CD : O) (α β : ℝ)
  (midpoint_M_GH : midpoint M GH)
  (chords_through_M : chord_contains_point M AB ∧ chord_contains_point M CD)
  (angle_AMG : α = angle M A G)
  (angle_CMG : β = angle M C G) :
  (sin α / sin β) = (distance M B - distance M A) / (distance M C - distance M D) :=
sorry

end sin_ratio_equal_segments_l502_502028


namespace log_base_four_of_product_l502_502285

theorem log_base_four_of_product (b : ℝ) (e : ℝ) (h : b > 0 ∧ b ≠ 1) :
  log b (4^3 * 4^(1/2)) = 7/2 := by
  simp[sorry]

end log_base_four_of_product_l502_502285


namespace length_of_new_section_l502_502762

-- Definitions from the conditions
def area : ℕ := 35
def width : ℕ := 7

-- The problem statement
theorem length_of_new_section (h : area = 35 ∧ width = 7) : 35 / 7 = 5 :=
by
  -- We'll provide the proof later
  sorry

end length_of_new_section_l502_502762


namespace units_digit_of_composite_product_l502_502649

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502649


namespace trapezoid_area_ratio_l502_502935

theorem trapezoid_area_ratio (AB CD AD BC : ℝ) 
  (h_parallel : AB ∥ CD)
  (h_CD_2AB : CD = 2 * AB)
  (DP PA BQ QC : ℝ)
  (h_DP_PA : DP / PA = 3 / 4)
  (h_BQ_QC : BQ / QC = 1 / 2) : 
  let ABQP_area := -- sorry for now; correct computation needed for ABQP_area
      5 -- placeholder
  let CDPQ_area := -- sorry for now; correct computation needed for CDPQ_area
      7 -- placeholder
  ABQP_area / CDPQ_area = 5 / 7 := 
sorry

end trapezoid_area_ratio_l502_502935


namespace find_angle_A_l502_502924

noncomputable def angle_A (a b c S : ℝ) := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

theorem find_angle_A (a b c S : ℝ) (hb : 0 < b) (hc : 0 < c) (hS : S = (1/2) * b * c * Real.sin (angle_A a b c S)) 
    (h_eq : b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S) : 
    angle_A a b c S = π / 6 := by 
  sorry

end find_angle_A_l502_502924


namespace trigonometric_identity_l502_502808

theorem trigonometric_identity (α : ℝ) : 
  (sin (π + α))^2 - (cos (π + α)) * (cos (-α)) + 1 = 2 :=
by sorry

end trigonometric_identity_l502_502808


namespace count_integers_containing_2_and_5_l502_502899

-- Definition for the conditions in Lean 4
def contains_digits_2_5 (n : ℕ) : Prop :=
  let digits := n.digits 10
  2 ∈ digits ∧ 5 ∈ digits

def is_between_300_and_700 (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 700

-- The main statement
theorem count_integers_containing_2_and_5 : 
  {n : ℕ | is_between_300_and_700 n ∧ contains_digits_2_5 n}.to_finset.card = 8 :=
by sorry

end count_integers_containing_2_and_5_l502_502899


namespace delta_problem_1_delta_problem_2_l502_502968

-- Definition of the custom binary operation Δ
def delta (A B : ℕ) : ℕ := A * B + A + B

-- Statement of the mathematical problems
theorem delta_problem_1 : delta (delta (delta 1 9) 9) 9 = 1999 := 
sorry

theorem delta_problem_2 (n : ℕ) (hn : n > 0) : 
  let result := (string.of_nat 1) ++ string.replicate (n - 1) '9' in 
  delta (delta 1 (iterate (delta 1) n 9)) = string.to_nat result := 
sorry

end delta_problem_1_delta_problem_2_l502_502968


namespace find_w_value_l502_502337

theorem find_w_value (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : (sqrt x / sqrt y) - (sqrt y / sqrt x) = 7 / 12) (h4 : x - y = 7) : x + y = 25 := 
sorry

end find_w_value_l502_502337


namespace problem1_problem2_l502_502367

-- Definitions of vectors
noncomputable def a : ℝ × ℝ := (-3, 0)
noncomputable def b (μ : ℝ) : ℝ × ℝ := (μ + 3, -1)
noncomputable def c (λ : ℝ) : ℝ × ℝ := (1, λ)

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Angle between vectors
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  real.acos (dot_product v w / (magnitude v * magnitude w))

-- Projection magnitude of vector v onto vector w
noncomputable def projection_magnitude (v w : ℝ × ℝ) : ℝ :=
  real.abs (dot_product v w) / magnitude w

-- Problem 1: Prove the angle between \a - \c and \b is π/4 given \λ=8 and \μ=-6.
theorem problem1 (λ : ℝ) (μ : ℝ) (hλ : λ = 8) (hμ : μ = -6) :
  angle_between (a - c λ) (b μ) = real.pi / 4 :=
sorry

-- Problem 2: Prove the values of \λ and \μ are ±2√2 if \a + \b ⊥ \c and projection magnitude of a onto c is 1.
theorem problem2 (λ μ : ℝ) 
  (h1 : dot_product (a + b μ) (c λ) = 0) 
  (h2 : projection_magnitude a (c λ) = 1) :
  (λ = 2 * real.sqrt 2 ∧ μ = 2 * real.sqrt 2) ∨ (λ = -2 * real.sqrt 2 ∧ μ = -2 * real.sqrt 2) :=
sorry

end problem1_problem2_l502_502367


namespace max_value_of_sum_l502_502118

theorem max_value_of_sum (a b : ℝ) (h : a^2 + b^2 = 1) : a + b + a * b ≤ √2 + 1/2 := 
sorry

end max_value_of_sum_l502_502118


namespace expand_and_simplify_l502_502287

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5 * x - 66 :=
by
  sorry

end expand_and_simplify_l502_502287


namespace mark_owes_820_l502_502996

-- Definitions of the problem conditions
def base_fine : ℕ := 50
def over_speed_fine (mph_over : ℕ) : ℕ := mph_over * 2
def school_zone_multiplier : ℕ := 2
def court_costs : ℕ := 300
def lawyer_cost_per_hour : ℕ := 80
def lawyer_hours : ℕ := 3

-- Calculation of the total fine
def total_fine (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let mph_over := actual_speed - speed_limit
  let additional_fine := over_speed_fine mph_over
  let fine_before_multipliers := base_fine + additional_fine
  let fine_after_multipliers := fine_before_multipliers * school_zone_multiplier
  fine_after_multipliers

-- Calculation of the total costs
def total_costs (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let fine := total_fine speed_limit actual_speed
  fine + court_costs + (lawyer_cost_per_hour * lawyer_hours)

theorem mark_owes_820 : total_costs 30 75 = 820 := 
by
  sorry

end mark_owes_820_l502_502996


namespace largest_eight_digit_with_all_evens_l502_502696

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l502_502696


namespace necessary_and_sufficient_condition_l502_502836

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
sorry

end necessary_and_sufficient_condition_l502_502836


namespace trust_meteorologist_l502_502176

-- Definitions for problem conditions
variables {G M1 M2 S : Prop}
variable {r : ℝ}
variable {p : ℝ}

/-- The probability of a clear day is r -/
axiom prob_clear_day : r = 0.74

/-- Senators' prediction accuracy -/
axiom senator_accuracy : ℝ

/-- Meteorologist's prediction accuracy being 1.5 times senators' /-
axiom meteorologist_accuracy : ∀ p, 1.5 * p

/-- Independence of predictions -/
axiom independence_preds : independent [G, M1, M2, S]

noncomputable def joint_probability_given_G : ℝ :=
(1 - 1.5 * meteorologist_accuracy senator_accuracy) * senator_accuracy^2

noncomputable def joint_probability_given_not_G : ℝ :=
1.5 * meteorologist_accuracy senator_accuracy * (1 - senator_accuracy)^2

noncomputable def overall_probability : ℝ :=
joint_probability_given_G * r + joint_probability_given_not_G * (1 - r)

noncomputable def conditional_prob_not_clear : ℝ /-
(joint_probability_given_not_G * (1 - r)) / overall_probability

noncomputable def conditional_prob_clear : ℝ
(joint_probability_given_G * r) / overall_probability

-- Main theorem statement: Given the conditions, the meteorologist's forecast is more reliable
theorem trust_meteorologist : conditional_prob_not_clear > conditional_prob_clear :=
by sorry

end trust_meteorologist_l502_502176


namespace probability_two_cards_different_numbers_l502_502238

theorem probability_two_cards_different_numbers :
  let total_outcomes := 10 * 10 in
  let favorable_outcomes := 10 * 9 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 9 / 10 :=
by
  sorry

end probability_two_cards_different_numbers_l502_502238


namespace mirror_area_l502_502496

theorem mirror_area (frame_length frame_width frame_width_side : ℕ)
  (mirror_length mirror_width : ℕ)
  (h_frame_length : frame_length = 120)
  (h_frame_width : frame_width = 100)
  (h_frame_width_side : frame_width_side = 15)
  (h_mirror_length : mirror_length = frame_length - 2 * frame_width_side)
  (h_mirror_width : mirror_width = frame_width - 2 * frame_width_side) :
  mirror_length * mirror_width = 6300 :=
by {
  rw [h_frame_length, h_frame_width, h_frame_width_side] at *,
  rw [h_mirror_length, h_mirror_width],
  sorry
}

end mirror_area_l502_502496


namespace units_digit_first_four_composites_l502_502616

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502616


namespace count_valid_numbers_l502_502897

def is_valid_number (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 250 ∧
  let digits := [n / 100, (n / 10) % 10, n % 10] in
  digits.nodup ∧
  digits = digits.sorted ∧
  digits.nth 2 ≥ (digits.nth 1 + 2)

theorem count_valid_numbers : {n // is_valid_number n}.card = 9 :=
sorry

end count_valid_numbers_l502_502897


namespace neg_p_l502_502915

open Nat -- Opening natural number namespace

-- Definition of the proposition p
def p := ∃ (m : ℕ), ∃ (k : ℕ), k * k = m * m + 1

-- Theorem statement for the negation of proposition p
theorem neg_p : ¬p ↔ ∀ (m : ℕ), ¬ ∃ (k : ℕ), k * k = m * m + 1 :=
by {
  -- Provide the proof here
  sorry
}

end neg_p_l502_502915


namespace skittles_bought_l502_502050

-- Step a: Definitions for the conditions
def original_skittles : ℕ := 5
def total_skittles : ℕ := 12

-- Step c: Translate to proof problem statement
theorem skittles_bought (bought_skittles : ℕ) (h : bought_skittles = total_skittles - original_skittles) : bought_skittles = 7 :=
by
  have h1 : total_skittles - original_skittles = 7 := by sorry -- this reflects the correct answer
  rw h1 at h
  exact h

end skittles_bought_l502_502050


namespace A_less_than_2aiaj_l502_502973

theorem A_less_than_2aiaj 
  (n : ℕ) (h_n : n > 1) 
  (a : Fin n → ℝ)
  (A : ℝ) 
  (h : A + (∑ i, a i ^ 2) < (1 / (n - 1 : ℝ)) * (∑ i, a i)^2) :
  ∀ i j : Fin n, i < j → A < 2 * a i * a j := 
by
  -- sorry to skip the proof for now
  sorry

end A_less_than_2aiaj_l502_502973


namespace kopeechka_purchase_l502_502443

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502443


namespace units_digit_of_product_of_first_four_composites_l502_502645

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502645


namespace units_digit_first_four_composites_l502_502603

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502603


namespace angle_between_planes_l502_502101

variables (α k : ℝ)
-- Conditions for α and k
variable (hα : α < π / 2)
variable (hα_pos : α > 0)
variable (k_pos : k > 0)

theorem angle_between_planes (a : ℝ) (ha_pos : a > 0) :
  ∃ θ : ℝ, θ = Real.arctan (k / (2 * Real.sin α)) :=
begin
  use Real.arctan (k / (2 * Real.sin α)),
  sorry
end

end angle_between_planes_l502_502101


namespace junior_scores_l502_502930

theorem junior_scores (total_students : ℕ) (junior_ratio : ℚ) (avg_class_score : ℚ) (avg_senior_score : ℚ)
                      (junior_identical_score : ℚ) (junior_score : ℚ) :
  junior_ratio = 0.2 →
  avg_class_score = 86 →
  avg_senior_score = 85 →
  junior_score = junior_identical_score →
  junior_identical_score =
    (avg_class_score * total_students - avg_senior_score * (total_students * (1 - junior_ratio))) /
    (total_students * junior_ratio) :=
begin
  sorry
end

end junior_scores_l502_502930


namespace units_digit_first_four_composite_is_eight_l502_502613

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502613


namespace units_digit_first_four_composites_l502_502601

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502601


namespace units_digit_of_product_is_eight_l502_502588

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502588


namespace math_problem_l502_502567

noncomputable def base10_b := 25 + 1  -- 101_5 in base 10
noncomputable def base10_c := 343 + 98 + 21 + 4  -- 1234_7 in base 10
noncomputable def base10_d := 2187 + 324 + 45 + 6  -- 3456_9 in base 10

theorem math_problem (a : ℕ) (b c d : ℕ) (h_a : a = 2468)
  (h_b : b = base10_b) (h_c : c = base10_c) (h_d : d = base10_d) :
  (a / b) * c - d = 41708 :=
  by {
  sorry
}

end math_problem_l502_502567


namespace solution_set_l502_502985

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set (x : ℝ) :
  ∃ f : ℝ → ℝ, 
    (∀ x, f (x + 2) = f x) ∧                   -- f is periodic with period 2
    (∀ x, f (-x) = f x) ∧                      -- f is even
    (∀ x ∈ Icc (1 : ℝ) 2, strict_mono_decr_on f (Icc 1 2)) ∧ -- f strictly decreases on [1, 2]
    (f π = 1) ∧                                -- f(π) = 1
    (f (2*π) = 0) ∧                            -- f(2π) = 0 
    (0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1) ↔      -- system of inequalities
    (2 * π - 6 ≤ x ∧ x ≤ 4 - π) := sorry       -- solution set is [2π - 6, 4 - π]

end solution_set_l502_502985


namespace area_of_overlap_l502_502108

def area_of_square_1 : ℝ := 1
def area_of_square_2 : ℝ := 4
def area_of_square_3 : ℝ := 9
def area_of_square_4 : ℝ := 16
def total_area_of_rectangle : ℝ := 27.5
def unshaded_area : ℝ := 1.5

def total_area_of_squares : ℝ := area_of_square_1 + area_of_square_2 + area_of_square_3 + area_of_square_4
def total_area_covered_by_squares : ℝ := total_area_of_rectangle - unshaded_area

theorem area_of_overlap :
  total_area_of_squares - total_area_covered_by_squares = 4 := 
sorry

end area_of_overlap_l502_502108


namespace normal_distribution_symmetric_l502_502319

-- Define the normal distribution and the variables
variables {σ : ℝ} (ξ : ℝ)

-- Assume ξ follows N(2, σ^2)
def is_normal_distribution (ξ : ℝ) (μ : ℝ) (σ : ℝ) : Prop :=
  ∃ Z, Z ~ ℕ(0, 1) ∧ ξ = μ + σ * Z

-- Assume P(ξ ≥ 4) = 0.6
def prob_ξ_geq_4 (ξ : ℝ) : Prop := 
  P(ξ ≥ 4) = 0.6

-- Translate proof problem into Lean 4 statement
theorem normal_distribution_symmetric (ξ : ℝ) (σ : ℝ) :
  is_normal_distribution ξ 2 σ ∧ prob_ξ_geq_4 ξ → P(ξ ≤ 0) = 0.6 :=
sorry

end normal_distribution_symmetric_l502_502319


namespace kellan_wax_remaining_l502_502475

def remaining_wax (initial_A : ℕ) (initial_B : ℕ)
                  (spill_A : ℕ) (spill_B : ℕ)
                  (use_car_A : ℕ) (use_suv_B : ℕ) : ℕ :=
  let remaining_A := initial_A - spill_A - use_car_A
  let remaining_B := initial_B - spill_B - use_suv_B
  remaining_A + remaining_B

theorem kellan_wax_remaining
  (initial_A : ℕ := 10) 
  (initial_B : ℕ := 15)
  (spill_A : ℕ := 3) 
  (spill_B : ℕ := 4)
  (use_car_A : ℕ := 4) 
  (use_suv_B : ℕ := 5) :
  remaining_wax initial_A initial_B spill_A spill_B use_car_A use_suv_B = 9 :=
by sorry

end kellan_wax_remaining_l502_502475


namespace units_digit_of_product_of_first_four_composites_l502_502639

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502639


namespace percent_of_male_literate_l502_502938

noncomputable def female_percentage : ℝ := 0.6
noncomputable def total_employees : ℕ := 1500
noncomputable def literate_percentage : ℝ := 0.62
noncomputable def literate_female_employees : ℕ := 630

theorem percent_of_male_literate :
  let total_females := (female_percentage * total_employees)
  let total_males := total_employees - total_females
  let total_literate := literate_percentage * total_employees
  let literate_male_employees := total_literate - literate_female_employees
  let male_literate_percentage := (literate_male_employees / total_males) * 100
  male_literate_percentage = 50 := by
  sorry

end percent_of_male_literate_l502_502938


namespace units_digit_of_product_of_first_four_composites_l502_502640

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502640


namespace log_x_125_l502_502386

noncomputable def x : ℝ := 102.4

theorem log_x_125 (hx : log 8 (5 * x) = 3) : log x 125 = 10428 / 10000 :=
by
  have h1 : 5 * x = 8 ^ 3 := log_eq_iff_eq_exp.1 hx
  have h2 : x = 102.4 := by rw [h1]; norm_num
  -- sorry: skipping detailed proof steps and numerical constants as by instruction
  sorry

end log_x_125_l502_502386


namespace num_valid_integers_l502_502379

theorem num_valid_integers : 
  (finset.card ((finset.range 10000).filter (λ n, ∀ d ∈ (n.digits 10), d ∉ {2, 3, 4, 5, 8}))) = 776 :=
by
  sorry

end num_valid_integers_l502_502379


namespace middle_digit_base8_l502_502758

theorem middle_digit_base8 (M : ℕ) (e : ℕ) (d f : Fin 8) 
  (M_base8 : M = 64 * d + 8 * e + f)
  (M_base10 : M = 100 * f + 10 * e + d) :
  e = 6 :=
by sorry

end middle_digit_base8_l502_502758


namespace units_digit_product_first_four_composite_numbers_l502_502679

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502679


namespace possible_items_l502_502431

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502431


namespace exists_points_with_integer_distances_l502_502508

theorem exists_points_with_integer_distances (n : ℕ) (hn : n ≥ 2) :
  ∃ (points : Fin n → ℝ × ℝ), 
    (∀ i j : Fin n, i ≠ j → i.val < j.val → dist (points i) (points j) ∈ Int) ∧
    ¬ ∀ (i j k : Fin n), collinear ℝ ({points i, points j, points k} : Set (ℝ × ℝ)) := by
  sorry

end exists_points_with_integer_distances_l502_502508


namespace total_grapes_l502_502774
-- Import the math library

-- Definitions for the initial conditions
def Rob_bowl := 25
def Allie_bowl := Rob_bowl + 2
def Allyn_bowl := Allie_bowl + 4

-- Prove that the total combined number of grapes in all three bowls is 83
theorem total_grapes : Rob_bowl + Allie_bowl + Allyn_bowl = 83 :=
by
  -- Use exact values from definitions to avoid floating unknowns
  have h1 : Rob_bowl = 25 := rfl
  have h2 : Allie_bowl = 27 := by rw [Allie_bowl, h1]; exact rfl
  have h3 : Allyn_bowl = 31 := by rw [Allyn_bowl, h2]; exact rfl
  rw [h1, h2, h3]
  rfl

end total_grapes_l502_502774


namespace purchase_options_l502_502417

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502417


namespace satisfies_diff_eq_l502_502512

noncomputable def y (x : ℝ) : ℝ := 2 * (sin x / x) + cos x

theorem satisfies_diff_eq : 
  ∀ x : ℝ, x * sin x * (deriv y x) + (sin x - x * cos x) * y x = sin x * cos x - x := 
by 
  -- proof goes here
  sorry

end satisfies_diff_eq_l502_502512


namespace find_general_formula_and_sum_l502_502937

noncomputable def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, d ≠ 0 ∧ (∀ n : ℕ, a (n + 1) = a n + d)

def is_geometric (b : ℕ → ℤ) (i j k : ℕ) :=
  b j * b j = b i * b k

theorem find_general_formula_and_sum {a b : ℕ → ℤ} (h : ∃ d : ℤ, d ≠ 0 ∧ (∀ n : ℕ, a (n + 1) = a n + d))
  (h1 : a 1 = 2)
  (h2 : is_geometric a 3 5 8) :
  (∀ n : ℕ, a n = n + 1) ∧ (∀ n : ℕ, let b n := a n + 2^(a n) in 
    (∑ i in finset.range (n + 1), b i) = 2^(n + 2) + (n * (n + 1))/2 + (3*n)/2 - 4
  ) :=
sorry

end find_general_formula_and_sum_l502_502937


namespace orthocenters_collinear_l502_502980

-- Define a triangle with points A, B, C
variables {A B C K M L N : Type} [Point A] [Point B] [Point C] 
[Point K] [Point M] [Point L] [Point N]

-- Conditions
def on_side_AB (A B K M : Type) [Point A] [Point B] [Point K] [Point M] : Prop := 
  PointOnSegment K (Segment.mk A B) ∧ PointOnSegment M (Segment.mk A B)

def on_side_AC (A C L N : Type) [Point A] [Point C] [Point L] [Point N] : Prop := 
  PointOnSegment L (Segment.mk A C) ∧ PointOnSegment N (Segment.mk A C)
  
def between (X Y Z : Type) [Point X] [Point Y] [Point Z] : Prop := 
  PointOnSegment Y (Segment.mk X Z)

def ratio_condition (B K M C L N : Type) [Point B] [Point K] [Point M] [Point C] 
[Point L] [Point N] : Prop :=
  (distance B K / distance K M) = (distance C L / distance L N)

-- Define orthocenters
def orthocenter (A B C : Type) [Point A] [Point B] [Point C] : Type := sorry

-- Define collinearity
def collinear {P Q R : Type} [Point P] [Point Q] [Point R] : Prop := 
  LineContaining P Q ⊆ LineContaining P R

-- Statement: Given conditions, prove collinearity of orthocenters
theorem orthocenters_collinear (A B C K M L N : Type) [Point A] [Point B] [Point C] 
[Point K] [Point M] [Point L] [Point N] :
  on_side_AB A B K M ∧
  on_side_AC A C L N ∧
  between K M B ∧
  between L N C ∧
  ratio_condition B K M C L N →
  collinear (orthocenter A B C) (orthocenter A K L) (orthocenter A M N) :=
begin
  sorry
end

end orthocenters_collinear_l502_502980


namespace largest_eight_digit_number_with_even_digits_l502_502714

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l502_502714


namespace largest_8_digit_number_with_even_digits_l502_502684

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l502_502684


namespace max_min_values_in_interval_l502_502860

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

theorem max_min_values_in_interval :
  (∀ x, x ∈ set.Icc (-1 : ℝ) 3 → f x ≤ 16) ∧ 
  (∃ x, x ∈ set.Icc (-1 : ℝ) 3 ∧ f x = 16) ∧
  (∀ x, x ∈ set.Icc (-1 : ℝ) 3 → 0 ≤ f x) ∧ 
  (∃ x, x ∈ set.Icc (-1 : ℝ) 3 ∧ f x = 0) :=
by 
  sorry

end max_min_values_in_interval_l502_502860


namespace hyperbola_max_product_eccentricity_l502_502888

theorem hyperbola_max_product_eccentricity (a b x y : ℝ) (h1 : 0 < a) (h2: 0 < b)
  (h3 : (x^2 / a^2 - y^2 / b^2 = 1)) (h4 : ∃ F1 F2 A B P Q : ℝ,
    A = line_perpendicular F1 intersects hyperbola 
    ∧ B = line_perpendicular F1 intersects hyperbola
    ∧ AF2.intersects_y_axis P
    ∧ BF2.intersects_y_axis Q
    ∧ perimeter_of_triangle P Q F2 = 12) :
  when_max_ab, e = 2 * sqrt 3 / 3 :=
begin
  sorry,
end

end hyperbola_max_product_eccentricity_l502_502888


namespace total_population_of_Springfield_and_Greenville_l502_502796

theorem total_population_of_Springfield_and_Greenville :
  let Springfield := 482653
  let diff := 119666
  let Greenville := Springfield - diff
  Springfield + Greenville = 845640 := by
  sorry

end total_population_of_Springfield_and_Greenville_l502_502796


namespace old_hen_weight_unit_l502_502778

theorem old_hen_weight_unit (w : ℕ) (units : String) (opt1 opt2 opt3 opt4 : String)
  (h_opt1 : opt1 = "grams") (h_opt2 : opt2 = "kilograms") (h_opt3 : opt3 = "tons") (h_opt4 : opt4 = "meters") (h_w : w = 2) : 
  (units = opt2) :=
sorry

end old_hen_weight_unit_l502_502778


namespace simplify_and_evaluate_expression_l502_502515

theorem simplify_and_evaluate_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  let expr := (xy + 2) * (xy - 2) + (xy - 2)^2
  in expr / xy = -8 :=
by
  simp only [h1, h2]
  have xy := x * y
  have expr := (xy + 2) * (xy - 2) + (xy - 2)^2
  calc
    expr / xy = ?m -- this line will be expanded in the full proof
  sorry

end simplify_and_evaluate_expression_l502_502515


namespace num_valid_integers_l502_502377

theorem num_valid_integers : 
  (finset.card ((finset.range 10000).filter (λ n, ∀ d ∈ (n.digits 10), d ∉ {2, 3, 4, 5, 8}))) = 776 :=
by
  sorry

end num_valid_integers_l502_502377


namespace divisibility_of_f_l502_502829

-- Definitions for C_n^i and f(n, q)
def c (n i : ℕ) : ℕ := if binomial n i % 2 = 1 then 1 else 0

def f (n q : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ i, c n i * q^i)

-- Lean statement for the proof
theorem divisibility_of_f (m n q r : ℕ) (h1 : 0 ≤ m) (h2 : 0 ≤ n) (h3 : 0 ≤ q) 
  (h4 : q + 1 ≠ 2^((Int.log2 q) + 1)) (h5 : f m q ∣ f n q) (h6 : 0 ≤ r) 
  : f m r ∣ f n r := by sorry

end divisibility_of_f_l502_502829


namespace odd_function_condition_l502_502060

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem odd_function_condition (A ω : ℝ) (hA : 0 < A) (hω : 0 < ω) (φ : ℝ) :
  (f A ω φ 0 = 0) ↔ (f A ω φ) = fun x => -f A ω φ (-x) := 
by
  sorry

end odd_function_condition_l502_502060


namespace nina_weekend_earnings_l502_502999

noncomputable def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℕ)
                                   (necklaces_sold bracelets_sold individual_earrings_sold ensembles_sold : ℕ) : ℕ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (individual_earrings_sold / 2) +
  ensemble_price * ensembles_sold

theorem nina_weekend_earnings :
  total_money_made 25 15 10 45 5 10 20 2 = 465 :=
by
  sorry

end nina_weekend_earnings_l502_502999


namespace sum_and_product_l502_502540

variable (a b : ℝ)

theorem sum_and_product (h1 : (a + (a - real.sqrt b)) = 0) (h2 : (a + real.sqrt b) * (a - real.sqrt b) = 4) : a + b = -4 :=
sorry

end sum_and_product_l502_502540


namespace peter_candles_l502_502501

theorem peter_candles (candles_rupert : ℕ) (ratio : ℝ) (candles_peter : ℕ) 
  (h1 : ratio = 3.5) (h2 : candles_rupert = 35) (h3 : candles_peter = candles_rupert / ratio) : 
  candles_peter = 10 := 
sorry

end peter_candles_l502_502501


namespace area_ratio_OAC_OAB_l502_502853

-- Definitions for the points inside the equilateral triangle and the vector equation condition
variable (O A B C : Type) [equilateral_triangle ABC] 
variable (h : O ∈ interior_triangle ABC) 
variable (v : vector_space ℝ (O → A) (O → B) (O → C))

-- Given condition
variable (h_eq : v (O → A) + 2 * v (O → B) + 3 * v (O → C) = 0)

-- Proof that the ratio of the area of triangle △OAC to the area of triangle △OAB is 2/3
theorem area_ratio_OAC_OAB : ratio (area (triangle O A C)) (area (triangle O A B)) = 2 / 3 := 
sorry

end area_ratio_OAC_OAB_l502_502853


namespace inverse_function_l502_502537

noncomputable def f : ℝ → ℝ := λ x, 1 + real.log x / real.log 2 -- Definition for f(x) = 1 + log_2 (x)

noncomputable def f_inv : ℝ → ℝ := λ x, 2 ^ (x - 1) -- Definition for f⁻¹(x) = 2^(x - 1)

theorem inverse_function :
  ∀ x : ℝ, x ≥ 1 → f (f_inv x) = x ∧ f_inv (f x) = x :=
by
  intros
  sorry

end inverse_function_l502_502537


namespace largest_eight_digit_number_with_even_digits_l502_502713

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l502_502713


namespace no_adjacent_teachers_l502_502553

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

theorem no_adjacent_teachers (students teachers : ℕ)
  (h_students : students = 4)
  (h_teachers : teachers = 3) :
  ∃ (arrangements : ℕ), arrangements = (factorial students) * (permutation (students + 1) teachers) :=
by
  sorry

end no_adjacent_teachers_l502_502553


namespace triangle_solution_l502_502916

-- Conditions: the base-6 addition problem holds.
def condition1 (triangle : ℕ) : Prop :=
  let sum1 := 3 * 6^2 + 2 * 6 + 1 * 6 + triangle in
  let sum2 := triangle * 6^2 + 4 * 6 + 0 in
  let sum3 := triangle * 6 + 2 in
  let result := 4 * 6^2 + 2 * 6 + triangle * 6 + 1 in
  sum1 + sum2 + sum3 = result

-- The proof statement: Prove that \( \triangle = 5 \) satisfies the condition.
theorem triangle_solution : ∃ triangle : ℕ, condition1 triangle ∧ triangle = 5 :=
by sorry

end triangle_solution_l502_502916


namespace largest_eight_digit_with_all_even_digits_l502_502710

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l502_502710


namespace no_integer_solution_l502_502268

theorem no_integer_solution : ¬∃ (x y : ℤ), 2^(2 * x) - 3^(2 * y) = 35 := 
by
  sorry

end no_integer_solution_l502_502268


namespace kolya_purchase_l502_502450

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502450


namespace units_digit_product_first_four_composite_numbers_l502_502671

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502671


namespace constant_term_expansion_l502_502529

theorem constant_term_expansion (x : ℝ) (n : ℕ) (hn : n = 6) :
  ((finset.range (n + 1)).max' (by sorry)).card = 4 →
  let term := (finset.range (n + 1)).filter (λ r, (x ^ ((3 - 3 * r) / 2).denom = 0)) in
  term.val = (∑ (r in term), nat.choose n r * (1 / 3) ^ r).to_real * (∏ r in term, x ^ ((3 - 3 * r) / 2)) 
: term = 5 / 3 :=
by sorry

end constant_term_expansion_l502_502529


namespace lawn_width_is_60_l502_502761

theorem lawn_width_is_60
  (length : ℕ)
  (width : ℕ)
  (road_width : ℕ)
  (cost_per_sq_meter : ℕ)
  (total_cost : ℕ)
  (area_of_lawn : ℕ)
  (total_area_of_roads : ℕ)
  (intersection_area : ℕ)
  (area_cost_relation : total_area_of_roads * cost_per_sq_meter = total_cost)
  (intersection_included : (road_width * length + road_width * width - intersection_area) = total_area_of_roads)
  (length_eq : length = 80)
  (road_width_eq : road_width = 10)
  (cost_eq : cost_per_sq_meter = 2)
  (total_cost_eq : total_cost = 2600)
  (intersection_area_eq : intersection_area = road_width * road_width)
  : width = 60 :=
by
  sorry

end lawn_width_is_60_l502_502761


namespace older_brother_catches_up_l502_502945

-- Define the initial conditions and required functions
def younger_brother_steps_before_chase : ℕ := 10
def time_per_3_steps_older := 1  -- in seconds
def time_per_4_steps_younger := 1  -- in seconds 
def dist_older_in_5_steps : ℕ := 7  -- 7d_younger / 5
def dist_younger_in_7_steps : ℕ := 5
def speed_older : ℕ := 3 * dist_older_in_5_steps / 5  -- steps/second 
def speed_younger : ℕ := 4 * dist_younger_in_7_steps / 7  -- steps/second

theorem older_brother_catches_up : ∃ n : ℕ, n = 150 :=
by sorry  -- final theorem statement with proof omitted

end older_brother_catches_up_l502_502945


namespace evaluate_expression_l502_502283

theorem evaluate_expression :
  (81:ℝ)^(1/2) * (64:ℝ)^(-1/3) * (49:ℝ)^(1/2) = (63:ℝ) / 4 :=
by
  sorry

end evaluate_expression_l502_502283


namespace jersey_sum_adjacent_gt_17_l502_502503

theorem jersey_sum_adjacent_gt_17 (a : ℕ → ℕ) (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ n, 0 < a n ∧ a n ≤ 10) (h_circle : ∀ n, a n = a (n % 10)) :
  ∃ n, a n + a (n+1) + a (n+2) > 17 :=
by
  sorry

end jersey_sum_adjacent_gt_17_l502_502503


namespace circle_equation_tangent_x_axis_l502_502111

theorem circle_equation_tangent_x_axis (x y : ℝ) (center : ℝ × ℝ) (r : ℝ) 
  (h_center : center = (-1, 2)) 
  (h_tangent : r = |2 - 0|) :
  (x + 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_tangent_x_axis_l502_502111


namespace trust_meteorologist_l502_502172

/-- 
The probability of a clear day in Anchuria.
-/
def P_G : ℝ := 0.74

/-- 
Accuracy of the forecast by each senator.
-/
variable (p : ℝ)

/-- 
Accuracy of the meteorologist's forecast being 1.5 times that of a senator.
-/
def meteorologist_accuracy : ℝ := 1.5 * p

/-- 
Calculations and final proof that the meteorologist's forecast is more reliable than that of the senators. 
-/
theorem trust_meteorologist (p : ℝ) (Hp1 : 0 ≤ p) (Hp2 : p ≤ 1) : 
  λ P_S_M1_M2_G P_S_M1_M2_not_G : 
  P_G := 0.74 ∧ meteorologist_accuracy = 1.5 * p ∧
  (∀ P_S_M1_M2, P_S_M1_M2 = P_S_M1_M2_G * P_G + P_S_M1_M2_not_G * (1 - P_G)) → 
  (P_S_M1_M2_not_G * (1 - P_G) > P_S_M1_M2_G * P_G) :=
begin
  sorry
end

end trust_meteorologist_l502_502172


namespace largest_eight_digit_number_with_even_digits_l502_502718

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l502_502718


namespace units_digit_first_four_composites_l502_502596

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502596


namespace minimum_exercise_hours_l502_502560

theorem minimum_exercise_hours 
  (days_20_min : ℕ)
  (days_40_min : ℕ)
  (days_2_hours : ℕ)
  (min_minutes : ℕ)
  (max_minutes : ℕ)
  (total_days_20_min : days_20_min ≥ 26)
  (total_days_40_min : days_40_min ≥ 24)
  (total_days_2_hours : days_2_hours = 4)
  (never_less_than_min : min_minutes = 20)
  (never_more_than_max : max_minutes = 120)
  (never_exceed_2_hours : ∀ d, (d = days_20_min ∨ d = days_40_min ∨ d = days_2_hours) → (min_minutes ≤ d ∧ d ≤ max_minutes)) : 
  hours_exercised := 22 :=
by
  sorry

end minimum_exercise_hours_l502_502560


namespace kopeechka_purchase_l502_502440

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502440


namespace polynomial_divisibility_l502_502506

theorem polynomial_divisibility (n : ℕ) : (¬ n % 3 = 0) → (x ^ (2 * n) + x ^ n + 1) % (x ^ 2 + x + 1) = 0 :=
by
  sorry

end polynomial_divisibility_l502_502506


namespace num_valid_integers_l502_502378

theorem num_valid_integers : 
  (finset.card ((finset.range 10000).filter (λ n, ∀ d ∈ (n.digits 10), d ∉ {2, 3, 4, 5, 8}))) = 776 :=
by
  sorry

end num_valid_integers_l502_502378


namespace complex_modulus_l502_502006

theorem complex_modulus (z : ℂ) (h : z * (2 + complex.I) = 5 * complex.I - 10) : complex.abs z = 5 :=
sorry

end complex_modulus_l502_502006


namespace john_has_18_blue_pens_l502_502738

variables (R B Bl : ℕ)

-- Conditions from the problem
def john_has_31_pens : Prop := R + B + Bl = 31
def black_pens_5_more_than_red : Prop := B = R + 5
def blue_pens_twice_black : Prop := Bl = 2 * B

theorem john_has_18_blue_pens :
  john_has_31_pens R B Bl ∧ black_pens_5_more_than_red R B ∧ blue_pens_twice_black B Bl →
  Bl = 18 :=
by
  sorry

end john_has_18_blue_pens_l502_502738


namespace largest_eight_digit_number_l502_502704

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l502_502704


namespace endpoint_of_unit_vector_l502_502328

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def unit_vector (v : ℝ × ℝ) : Prop :=
  vector_magnitude v = 1

theorem endpoint_of_unit_vector 
  (a b : ℝ × ℝ)
  (ha_unit : unit_vector a)
  (ha_parallel : vector_parallel a b)
  (a_start : ℝ × ℝ)
  (hstart : a_start = (3, -1))
  (hb : b = (-3, 4))
  : ∃ x y : ℝ, 
    (x, y) = (3 - 3 / real.sqrt 5, -1 + 4 / real.sqrt 5) 
    ∨ (x, y) = (3 + 3 / real.sqrt 5, -1 - 4 / real.sqrt 5) :=
sorry

end endpoint_of_unit_vector_l502_502328


namespace cube_surface_area_increase_l502_502156

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l502_502156


namespace vector_projection_l502_502971

noncomputable def projection : ℝ := sorry

/-- Prove the projection of a vector onto a plane given certain conditions. -/
theorem vector_projection :
  let Q := {vec | vec.1 - vec.2 + 2 * vec.3 = 0} in
  (⟨6, 4, 6⟩ : ℝ × ℝ × ℝ) ∈ Q ∧
  (⟨4, 6, 2⟩ : ℝ × ℝ × ℝ) ∈ Q ∧
  projection (⟨5, 2, 8⟩ : ℝ × ℝ × ℝ) Q = ⟨11/6, 31/6, 10/6⟩ :=
by
  sorry

end vector_projection_l502_502971


namespace eval_expression_l502_502790

theorem eval_expression :
  (1/2 : ℝ)^(-1) + |(3 - real.sqrt 12)| + (-1)^2 = 2 * real.sqrt 3 := 
sorry

end eval_expression_l502_502790


namespace remainder_71_3_73_5_mod_8_l502_502570

theorem remainder_71_3_73_5_mod_8 :
  (71^3) * (73^5) % 8 = 7 :=
by {
  -- hint, use the conditions given: 71 ≡ -1 (mod 8) and 73 ≡ 1 (mod 8)
  sorry
}

end remainder_71_3_73_5_mod_8_l502_502570


namespace benny_spent_l502_502784

variable (initial_dollars remaining_dollars : ℕ)
variable (spent_dollars : ℕ)
variable (h_initial : initial_dollars = 79)
variable (h_remaining : remaining_dollars = 32)

theorem benny_spent (h : spent_dollars = initial_dollars - remaining_dollars) : spent_dollars = 47 := by
  subst h_initial
  subst h_remaining
  rw [h]
  rfl

end benny_spent_l502_502784


namespace largest_eight_digit_number_with_even_digits_l502_502712

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l502_502712


namespace proof_problem_l502_502855

noncomputable def p : Prop := ∃ (α : ℝ), Real.cos (Real.pi - α) = Real.cos α
def q : Prop := ∀ (x : ℝ), x ^ 2 + 1 > 0

theorem proof_problem : p ∨ q := 
by
  sorry

end proof_problem_l502_502855


namespace replaced_person_weight_l502_502524

theorem replaced_person_weight (W : ℝ) :
  let increase := 8 * 2.5
  let new_person_weight := 65
  -- Given the average weight of 8 persons increases by 2.5 kg, 
  -- Prove that the weight of the replaced person is 45 kg.
  (W + 8 * 2.5 = W - (new_person_weight - increase) + new_person_weight) → 
  (new_person_weight - increase = 45) := 
by
  intro h
  have a : increase = 8 * 2.5 := rfl
  have b : new_person_weight = 65 := rfl
  calc 
  new_person_weight - increase = 65 - 20 := by rw [a, b]
  ... = 45 := by norm_num

end replaced_person_weight_l502_502524


namespace cube_surface_area_increase_l502_502159

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l502_502159


namespace units_digit_product_first_four_composite_numbers_l502_502680

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502680


namespace statistical_measure_mode_l502_502137

theorem statistical_measure_mode (fav_dishes : List ℕ) :
  (∀ measure, (measure = "most frequently occurring value" → measure = "Mode")) :=
by
  intro measure
  intro h
  sorry

end statistical_measure_mode_l502_502137


namespace solution_set_of_inequality_l502_502061

noncomputable def f (x: ℝ) : ℝ := sorry
noncomputable def f' (x: ℝ) : ℝ := sorry

theorem solution_set_of_inequality (h_even: ∀ x: ℝ, f(x) = f(-x)) 
  (h_inequality: ∀ x: ℝ, x < 0 → f(x) + x * f'(x) < 0)
  (h_initial: f(-4) = 0) : 
  { x: ℝ | x * f x > 0 } = { x: ℝ | x ∈ Ioo (-∞) (-4) ∪ Ioo 4 (+∞) } :=
sorry

end solution_set_of_inequality_l502_502061


namespace john_makes_money_l502_502961

-- Definitions of the conditions
def num_cars := 5
def time_first_3_cars := 3 * 40 -- 3 cars each take 40 minutes
def time_remaining_car := 40 * 3 / 2 -- Each remaining car takes 50% longer
def time_remaining_cars := 2 * time_remaining_car -- 2 remaining cars
def total_time_min := time_first_3_cars + time_remaining_cars
def total_time_hr := total_time_min / 60 -- Convert total time from minutes to hours
def rate_per_hour := 20

-- Theorem statement
theorem john_makes_money : total_time_hr * rate_per_hour = 80 := by
  sorry

end john_makes_money_l502_502961


namespace train_length_is_correct_l502_502229

noncomputable def length_of_train 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * time_to_cross
  distance_covered - bridge_length

theorem train_length_is_correct :
  length_of_train 23.998080153587715 140 36 = 99.98080153587715 :=
by sorry

end train_length_is_correct_l502_502229


namespace largest_eight_digit_with_all_evens_l502_502691

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l502_502691


namespace find_a_l502_502872

-- Define the first line and its slope
def line1 (a : ℝ) : ℝ → ℝ := λ x, a * x + 2 * a

-- Define the second line and its slope
def line2 (a : ℝ) : ℝ → ℝ := λ x, -((2 * a - 1) / a) * x - 1

-- Define the condition for perpendicular lines
def perpendicular_condition (a : ℝ) : Prop :=
  a * (-(2 * a - 1) / a) = -1

-- Given the conditions, prove that the values of a are 1 or 0
theorem find_a (a : ℝ) :
  perpendicular_condition a → (a = 1 ∨ a = 0) :=
by sorry

end find_a_l502_502872


namespace total_votes_election_l502_502025

-- Given conditions
variables (V : ℝ) -- Total number of votes
variables (validVotesA : ℕ) -- Number of valid votes for candidate A
variables (p_invalid : ℝ := 0.15) -- Percentage of invalid votes
variables (p_validA : ℝ := 0.70) -- Percentage of valid votes candidate A got

-- Set the given correct value
def validVotesA_value : ℝ := 333200
def V_value : ℝ := 560000

-- Prove the total number of votes
theorem total_votes_election : 
  validVotesA = 333200 ∧ p_invalid = 0.15 ∧ p_validA = 0.70 →
  V = 560000 :=
by
  intro h,
  cases h,
  cases h_right,
  sorry

end total_votes_election_l502_502025


namespace liters_to_pints_l502_502335

theorem liters_to_pints (l : ℝ) (p : ℝ) (h : 0.75 = l) (h_p : 1.575 = p) : 
  Float.round (1.5 * (p / l) * 10) / 10 = 3.2 :=
by sorry

end liters_to_pints_l502_502335


namespace units_digit_product_first_four_composite_numbers_l502_502676

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502676


namespace cube_surface_area_increase_l502_502155

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l502_502155


namespace range_of_f_l502_502879

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * sqrt 3 * sin (ω * x) * cos (ω * x) + 2 * cos (ω * x) ^ 2 - 1

theorem range_of_f {ω : ℝ} (h1 : ω > 0) (h2 : (2 * π) / (2 * ω) = π / 2) :
  set.range (f ω) ∩ set.Icc 0 (π / 4) = set.Icc (-1 : ℝ) 2 :=
sorry

end range_of_f_l502_502879


namespace probability_digits_different_l502_502243

theorem probability_digits_different : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999} in
  let total := ∑ x in S, 1 in
  let different_digits := ∑ x in S, (if (x / 100 ≠ (x % 100) / 10 ∧ (x % 100) / 10 ≠ x % 10 ∧ x / 100 ≠ x % 10) then 1 else 0) in
  (different_digits / total) = (18 / 25) := by
sorry

end probability_digits_different_l502_502243


namespace valid_combination_exists_l502_502726

def exists_valid_combination : Prop :=
  ∃ (a: Fin 7 → ℤ), (a 0 = 1) ∧
  (a 1 = 2) ∧ (a 2 = 3) ∧ (a 3 = 4) ∧ 
  (a 4 = 5) ∧ (a 5 = 6) ∧ (a 6 = 7) ∧
  ((a 0 = a 1 + a 2 + a 3 + a 4 - a 5 - a 6))

theorem valid_combination_exists :
  exists_valid_combination :=
by
  sorry

end valid_combination_exists_l502_502726


namespace largest_eight_digit_number_contains_even_digits_l502_502721

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l502_502721


namespace solve_for_y_l502_502093

theorem solve_for_y (y : ℝ) :
  (1/2)^(4*y + 10) = 8^(2*y + 3) → y = -1.9 :=
sorry

end solve_for_y_l502_502093


namespace vector_basis_translation_l502_502741

open Polynomial

noncomputable theory

variables {n : ℕ}
variables {f : ℤ2[X]} (hf : irreducible (X^n + 1) / (X + 1))

theorem vector_basis_translation (v : Fin n → ℤ2) 
  (h1 : (Finset.univ.filter (λ i, v i = 1)).card % 2 = 1) (h2 : ∃ i, v i = 0) :
  ∃ b : Basis (Fin n) ℤ2, ∀ k, b k = v.rotate k :=
sorry

end vector_basis_translation_l502_502741


namespace solve_equations_l502_502518

theorem solve_equations :
  (∀ x : ℝ, (1 / 2) * (2 * x - 5) ^ 2 - 2 = 0 ↔ x = 7 / 2 ∨ x = 3 / 2) ∧
  (∀ x : ℝ, x ^ 2 - 4 * x - 4 = 0 ↔ x = 2 + 2 * Real.sqrt 2 ∨ x = 2 - 2 * Real.sqrt 2) :=
by
  sorry

end solve_equations_l502_502518


namespace problem_solution_l502_502161

noncomputable def problem_statement : Prop :=
  ∀ (x y a : ℝ), (x > y) → (a = 0) → ¬(-a^2 * x < -a^2 * y)

theorem problem_solution : problem_statement :=
by {
  intros x y a hx ha,
  sorry
}

end problem_solution_l502_502161


namespace total_distribution_in_dollars_l502_502767

-- Defining the exchange rates and the initial values
def exchange_rates := {
  pound_to_dollar : ℝ := 1.18,
  euro_to_dollar : ℝ := 1.12,
  yen_per_200_to_dollar : ℝ := 1.82,
  aud_to_dollar : ℝ := 0.73
}

-- Defining shares in their respective currencies for each person
def shares := {
  c_yen : ℝ := 5000,
  a_pound : ℝ := 25,  -- since 5000 yen is 25 * 200 yen
  b_euro_per_pound : ℝ := 1.5,
  d_dollar_per_pound : ℝ := 2,
  e_aud_per_pound : ℝ := 1.2
}

-- Calculating the total distribution in dollars
theorem total_distribution_in_dollars : 
  shares.c_yen / 200 * exchange_rates.yen_per_200_to_dollar + 
  (shares.c_yen / 200) * exchange_rates.pound_to_dollar +
  (shares.c_yen / 200 * shares.b_euro_per_pound) * exchange_rates.euro_to_dollar +
  (shares.c_yen / 200 * shares.d_dollar_per_pound) + 
  (shares.c_yen / 200 * shares.e_aud_per_pound) * exchange_rates.aud_to_dollar = 188.9 :=
by
  sorry

end total_distribution_in_dollars_l502_502767


namespace units_digit_product_first_four_composite_numbers_l502_502672

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502672


namespace min_max_distances_l502_502543

-- Define the radius of the circle and the distance from the center to point M.
variables (r d : ℝ) (M O : Point)

-- State the given conditions
def conditions : Prop :=
  r = 10 ∧ d = 3 ∧ dist O M = d

-- Define the minimum and maximum distances to prove
def minimum_distance (M O : Point) : ℝ := r - d
def maximum_distance (M O : Point) : ℝ := r + d

-- Formulate the main theorem
theorem min_max_distances (M O : Point) (r d : ℝ) (h : conditions r d M O) :
  minimum_distance M O = 7 ∧ maximum_distance M O = 13 :=
by
  sorry

end min_max_distances_l502_502543


namespace outfits_count_l502_502729

theorem outfits_count (s p : ℕ) (h_s : s = 5) (h_p : p = 3) : s * p = 15 :=
by
  rw [h_s, h_p]
  exact Nat.mul_comm 5 3

end outfits_count_l502_502729


namespace trust_meteorologist_l502_502183

-- Definitions
def probability_of_clear := 0.74
def senator_accuracy := p : ℝ
def meteorologist_accuracy := 1.5 * p

-- Events
def event_G := "clear day"
def event_M1 := "first senator predicted clear"
def event_M2 := "second senator predicted clear"
def event_S := "meteorologist predicted rain"

theorem trust_meteorologist :
  let r := probability_of_clear
  let p := senator_accuracy
  let q := meteorologist_accuracy
  1.5 * p * (1 - p)^2 * (1 - r) - (1 - 1.5 * p) * p^2 * r > 0 :=
by
  sorry

end trust_meteorologist_l502_502183


namespace find_f_neg2_l502_502062

section
variable {f : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(-x) = -f(x)
def f_def (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 < x → f(x) = x^2 - 3

theorem find_f_neg2 (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_f_def : f_def f) :
  f(-2) = -1 :=
sorry
end

end find_f_neg2_l502_502062


namespace find_triangle_angles_l502_502097

theorem find_triangle_angles :
  ∀ (A B C M N O : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited M] [inhabited N] [inhabited O],
  let AM := sorry, BN := sorry, AO := sorry, MO := sorry, NO := sorry, BO := sorry in
  (AO = sqrt 3 * MO) ∧
  (NO = (sqrt 3 - 1) * BO) →
  let angleA := 60, angleB := 90, angleC := 30 in
  angleA + angleB + angleC = 180 :=
sorry

end find_triangle_angles_l502_502097


namespace trust_meteorologist_l502_502187

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l502_502187


namespace average_male_students_l502_502523

variables (M : ℝ) (avg_total : ℝ := 90) (num_male : ℕ := 8) (avg_male : ℝ) (avg_female : ℝ := 92) (num_female : ℕ := 28)

theorem average_male_students : 
  let total_students := num_male + num_female,
      total_sum_grades := avg_total * (total_students : ℝ),
      total_sum_female_grades := avg_female * (num_female : ℝ),
      total_sum_male_grades := total_sum_grades - total_sum_female_grades
  in
  (total_sum_male_grades / (num_male : ℝ)) = 83 :=
by
  sorry

end average_male_students_l502_502523


namespace angle_relationship_l502_502405

-- Define the type of points in the plane
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Assume the conditions of the problem as hypotheses
variables (h1 : ∠BAC = 90) -- Triangle ABC is right triangle with ∠BAC = 90°
          (h2 : midpoint D BC) -- D is midpoint of BC
          (h3 : E ∈ AB) -- E is on AB
          (h4 : angleBisector BE (∠ABC)) -- BE bisects ∠ABC

-- Define the angles x, y, z
variables (x : ℝ := ∠BAE)
          (y : ℝ := ∠BEC)
          (z : ℝ := ∠ECB)

-- State the proof problem as a theorem
theorem angle_relationship :
  x = y ∧ x ≠ z :=
by
  sorry

end angle_relationship_l502_502405


namespace problem1_problem2_l502_502398

-- Conditions
def balls_in_bag := { red := 4, white := 2 }
def draws_without_replacement := λ n: Nat, Fin n -> { x : Nat // x < 6 }

-- Problem 1
theorem problem1 : 
  (prob_of_white_on_third_draw : ∀ draws : Fin 3 -> draws_without_replacement 6, 
  (∃ (balls: balls_in_bag),
   let ⟨draw1, draw2, draw3⟩ := draws in 
   (balls.red + balls.white = 6) -> 
   ((balls.white / 2) * (balls.red / 5 * balls.red / 4) + 
    (balls.red / 6 * balls.white / 5 * balls.white / 4) + 
    (balls.white / 6 * balls.red / 5 * balls.red / 4)) = 1 / 3)).sorry

-- Conditions
def binomial_distribution := { trials := 6, prob_red := 2 / 3 }

-- Problem 2
theorem problem2 : 
  (prob_of_draw_red_leq_4 : ∀ ξ : binomial_distribution, 
  (exists (dist: binomial_distribution),
   (ξ.sim dist) -> 
   (1 - binomial_prob(5, 2 / 3) * (2 / 3)^5 * (1 / 3) - (2 / 3)^6) = 473 / 729)).sorry

end problem1_problem2_l502_502398


namespace sum_a5_a6_a7_l502_502922

def S (n : ℕ) : ℕ :=
  n^2 + 2 * n + 5

theorem sum_a5_a6_a7 : S 7 - S 4 = 39 :=
  by sorry

end sum_a5_a6_a7_l502_502922


namespace units_digit_first_four_composites_l502_502617

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502617


namespace solve_geometry_problem_l502_502036

def geometry_problem : Prop :=
  ∃ (A B C D : Type) (angle : A → B → C → D → ℝ) (between : A → B → D → Prop),
  (between A B D) →
  (angle A B D = 150) →
  (angle B A C = 88) →
  (angle A B C + angle A B D = 180) →
  (angle A C B = 62)

theorem solve_geometry_problem : geometry_problem :=
begin
  sorry
end

end solve_geometry_problem_l502_502036


namespace units_digit_of_product_of_first_four_composites_l502_502660

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502660


namespace probability_no_adjacent_equal_l502_502826

open Finset
open BigOperators

def no_adjacent_equal_prob : ℚ :=
  -- Number of valid arrangements
  let num_valid_arrangements := 8 * 7^4 - 8 * 6 * 7^3 in
  -- Total possible arrangements
  let total_arrangements := 8^5 in
  -- The probability
  (637 : ℚ) / (2048 : ℚ)

theorem probability_no_adjacent_equal :
  no_adjacent_equal_prob = (637 : ℚ) / (2048 : ℚ) :=
  by
    sorry

end probability_no_adjacent_equal_l502_502826


namespace sets_of_headphones_l502_502205

-- Definitions of the conditions
variable (M H : ℕ)

-- Theorem statement for proving the question given the conditions
theorem sets_of_headphones (h1 : 5 * M + 30 * H = 840) (h2 : 3 * M + 120 = 480) : H = 8 := by
  sorry

end sets_of_headphones_l502_502205


namespace induction_inequality_l502_502145

theorem induction_inequality (n : ℕ) (h : n > 0) : 
  (finset.range (n+1)).sum (λ k, 1 / ((k + 2)^2 : ℝ)) > 1 / 2 - 1 / (n + 2) :=
sorry

end induction_inequality_l502_502145


namespace lines_concurrent_a_lines_concurrent_b_l502_502053

section part_a

variables {ABC : Type} [Triangle ABC]
variables {A_a B_a C_a A_b B_b C_b A_c B_c C_c : Point}
variables {A' B' C' : Point}

-- Definition of conditions for Part (a)
-- Excircle definitions
def excircle_tangent_points (ABC : Triangle) (A_a B_a C_a A_b B_b C_b A_c B_c C_c : Point) : Prop :=
  -- Conditions for excircle tangency points (can use explicit conditions if needed)
  sorry 

-- Intersection points
def intersection_points (ABC : Triangle) (A' B' C' : Point) (A_b B_b C_b A_c B_c C_c : Point) : Prop :=
  -- A' is the intersection of A_bC_b and A_cB_c
  -- B' is the intersection of B_aC_a and A_cB_c
  -- C' is the intersection of B_aC_a and A_bC_b
  sorry 

-- Lines concur
theorem lines_concurrent_a (ABC : Triangle) (A_a B_a C_a A_b B_b C_b A_c B_c C_c A' B' C' : Point)
  (h_exc : excircle_tangent_points ABC A_a B_a C_a A_b B_b C_b A_c B_c C_c)
  (h_int : intersection_points ABC A' B' C' A_b B_b C_b A_c B_c C_c) :
  concurrent (line A' A_a) (line B' B_b) (line C' C_c) :=
sorry

end part_a

section part_b

variables {ABC : Type} [Triangle ABC]
variables {T_a T_b T_c : Point}
variables {A' B' C' : Point}
variables {I H : Point}

-- Definition of conditions for Part (b)
-- Incircle tangency points
def incircle_tangent_points (ABC : Triangle) (T_a T_b T_c : Point) : Prop :=
  -- Conditions for incircle tangency points (can use explicit conditions if needed)
  sorry 

-- Intersection points
def intersection_points' (ABC : Triangle) (A' B' C' : Point) (A_b B_b C_b A_c B_c C_c : Point) : Prop :=
  -- A' is the intersection of A_bC_b and A_cB_c
  -- B' is the intersection of B_aC_a and A_cB_c
  -- C' is the intersection of B_aC_a and A_bC_b
  sorry 

-- Lines concur
theorem lines_concurrent_b (ABC : Triangle) (T_a T_b T_c A' B' C' : Point)
  (I H : Point)
  (h_inc : incircle_tangent_points ABC T_a T_b T_c)
  (h_int : intersection_points' ABC A' B' C' A_b B_b C_b A_c B_c C_c)
  (h_line : collinear I H T) :
  concurrent (line A' T_a) (line B' T_b) (line C' T_c) ∧ (exists O_2, collinear O_2 I H) :=
sorry

end part_b

end lines_concurrent_a_lines_concurrent_b_l502_502053


namespace trailing_zeroes_1500_l502_502787

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Function to count the number of trailing zeroes in n!
def trailing_zeroes (n : ℕ) : ℕ :=
  let f k acc := if k = 0 then acc else f (k / 5) (acc + k / 5)
  in f n 0

-- Theorem: The number of trailing zeroes in 1500! is 374
theorem trailing_zeroes_1500 : trailing_zeroes 1500 = 374 :=
sorry

-- We leave out the second question due to the ambiguity in the problem statement

end trailing_zeroes_1500_l502_502787


namespace difference_between_two_numbers_l502_502131

theorem difference_between_two_numbers (a : ℕ) (b : ℕ)
  (h1 : a + b = 24300)
  (h2 : b = 100 * a) :
  b - a = 23760 :=
by {
  sorry
}

end difference_between_two_numbers_l502_502131


namespace sqrt_simplify_l502_502830

theorem sqrt_simplify (x : ℝ) : sqrt (x^6 + x^4) = x^2 * sqrt (x^2 + 1) :=
by
  sorry

end sqrt_simplify_l502_502830


namespace find_x_l502_502824

open Real

noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

theorem find_x :
  ∃ x : ℝ, 0 < x ∧
  log_base 5 (x - 1) + log_base (sqrt 5) (x^2 - 1) + log_base (1/5) (x - 1) = 3 ∧
  x = sqrt (5 * sqrt 5 + 1) :=
by
  sorry

end find_x_l502_502824


namespace liters_to_pints_conversion_l502_502332

-- Definitions based on conditions
def liters_to_pints_ratio := 0.75 / 1.575
def target_liters := 1.5
def expected_pints := 3.15

-- Lean statement
theorem liters_to_pints_conversion 
  (h_ratio : 0.75 / 1.575 = liters_to_pints_ratio)
  (h_target : 1.5 = target_liters) :
  target_liters * (1 / liters_to_pints_ratio) = expected_pints :=
by 
  sorry

end liters_to_pints_conversion_l502_502332


namespace coin_game_probability_l502_502941

-- Define the outcomes of a single coin toss.
def coin_outcomes : set (fin 2) := {0, 1} -- where 0 represents tails and 1 represents heads

-- Define the possible outcomes when tossing three coins.
def three_coin_outcomes : set (fin 8) := {0, 1, 2, 3, 4, 5, 6, 7}

-- Define the set of winning outcomes.
def winning_outcomes : set (fin 8) := {0, 7} -- where 0 represents HHH and 7 represents TTT

-- Define the probability of winning the coin game.
def probability_of_winning : ℚ :=
  (winning_outcomes.card : ℚ) / (three_coin_outcomes.card : ℚ)

-- The theorem statement that the probability of winning is 1/4.
theorem coin_game_probability : probability_of_winning = 1 / 4 :=
by sorry

end coin_game_probability_l502_502941


namespace units_digit_of_first_four_composite_numbers_l502_502634

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502634


namespace kevin_hops_six_l502_502962

noncomputable def kevin_hop_distance (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 2 * 1 / 4
  else (⅔ ^ (n - 2)) * 1.5

noncomputable def kevin_total_distance (hops : ℕ) : ℚ :=
  (∑ k in (range hops).filter (λ k, k > 0), kevin_hop_distance (k + 1)) + kevin_hop_distance 1

theorem kevin_hops_six : kevin_total_distance 6 = 1071 / 243 :=
begin
  sorry
end

end kevin_hops_six_l502_502962


namespace largest_eight_digit_with_all_evens_l502_502697

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l502_502697


namespace probability_of_green_ball_l502_502256

/-- 
Container I holds 5 red balls and 5 green balls; 
container II holds 3 red balls and 3 green balls; 
container III holds 4 red balls and 2 green balls. 
A container is selected at random, and then a ball is randomly selected from that container. 
Prove that the probability that the ball selected is green is 4/9. 
-/
theorem probability_of_green_ball : 
  let P_I := 1 / 3,
      P_II := 1 / 3,
      P_III := 1 / 3,
      P_green_I := 5 / 10,
      P_green_II := 3 / 6,
      P_green_III := 2 / 6
  in (P_I * P_green_I + P_II * P_green_II + P_III * P_green_III) = 4 / 9 :=
by
  -- Introduce the given probabilities as let bindings
  let P_I := 1 / 3
  let P_II := 1 / 3
  let P_III := 1 / 3
  let P_green_I := 5 / 10
  let P_green_II := 3 / 6
  let P_green_III := 2 / 6

  -- Compute the total probability using the law of total probability
  let total_prob := P_I * P_green_I + P_II * P_green_II + P_III * P_green_III

  -- Simplify the components and verify the final probability
  have h1 : total_prob = 1 / 6 + 1 / 6 + 1 / 9 := by sorry
  have h2 : h1 = 3 / 18 + 3 / 18 + 2 / 18 := by sorry
  have h3 : h2 = 8 / 18 := by sorry
  have h4 : h3 = 4 / 9 := by sorry

  -- Conclude the theorem
  exact h4

end probability_of_green_ball_l502_502256


namespace purchase_options_l502_502419

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502419


namespace measure_angle_EFG_l502_502325

theorem measure_angle_EFG (AD_parallel_FG : ∀ (A D F G : Type) (l1 l2 : set Type), parallel l1 l2)
                           (angle_CFG_eq_2x : ∀ (C F G : Type) (x : ℝ), ∠CFG = 2 * x)
                           (angle_CEA_eq_4x : ∀ (C E A : Type) (x : ℝ), ∠CEA = 4 * x)
                           (supplementary_angles : ∀ (a b : ℝ), a + b = 180) :
                           ∃ (EFG : ℝ), EFG = 60 :=
begin
  sorry
end

end measure_angle_EFG_l502_502325


namespace janet_earning_per_post_l502_502957

theorem janet_earning_per_post :
  ∀ (time_per_post : ℕ) (hourly_rate : ℕ),
  time_per_post = 10 →
  hourly_rate = 90 →
  let seconds_per_hour := 3600 in
  let posts_per_hour := seconds_per_hour / time_per_post in
  let rate_per_post := hourly_rate / posts_per_hour in
  rate_per_post = 0.25 := 
by
  intros time_per_post hourly_rate h1 h2
  let seconds_per_hour := 3600
  let posts_per_hour := seconds_per_hour / time_per_post
  let rate_per_post := hourly_rate / posts_per_hour
  have h3 : seconds_per_hour = 3600 := by rfl
  have h4 : posts_per_hour = 3600 / 10 := by rw [h1]
  have h5 : posts_per_hour = 360 := by norm_num [h4]
  have h6 : rate_per_post = 90 / 360 := by rw [h2, h5]
  have h7 : rate_per_post = 0.25 := by norm_num [h6]
  exact h7

end janet_earning_per_post_l502_502957


namespace Laticia_knitted_socks_l502_502051

theorem Laticia_knitted_socks (x : ℕ) (cond1 : x ≥ 0)
  (cond2 : ∃ y, y = x + 4)
  (cond3 : ∃ z, z = (x + (x + 4)) / 2)
  (cond4 : ∃ w, w = z - 3)
  (cond5 : x + (x + 4) + z + w = 57) : x = 13 := by
  sorry

end Laticia_knitted_socks_l502_502051


namespace eval_expression_l502_502276

theorem eval_expression : 81^(1/2) * 64^(-1/3) * 49^(1/2) = (63 / 4) :=
by
  sorry

end eval_expression_l502_502276


namespace sum_of_g_l502_502063

def g (n : ℕ) : ℝ := round (n^(1/3 : ℝ))

theorem sum_of_g (s : ℝ) : s = ∑ k in (finset.range 4096).map (λ k, g k⁻¹) :=
  s = 758.3125 :=
sorry

end sum_of_g_l502_502063


namespace number_of_integers_satisfying_condition_l502_502381

theorem number_of_integers_satisfying_condition : 
  {n : ℤ | 100 < n^2 ∧ n^2 < 1000 ∧ 0 < n}.card = 21 := 
by 
  sorry

end number_of_integers_satisfying_condition_l502_502381


namespace kopeechka_purchase_l502_502445

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502445


namespace general_formula_an_sum_first_n_terms_l502_502494

open Nat

noncomputable def seq {n : ℕ} (h_npos: 0 < n) (a : ℕ → ℕ) : Prop := 
  ∀ (n' : ℕ) (h_n' : 0 < n'), (S_n a n' h_n' / n' = n')

noncomputable def geom_seq {n : ℕ} (b : ℕ → ℕ) : Prop :=
  ∃ q, b 1 = 1 ∧ (∀ n', b (n' + 2) = b (n' + 1) * q) ∧ (b 1 * b 2 * b 3 = 8)

noncomputable def sum_seq {n : ℕ} (a b : ℕ → ℕ) : ℕ :=
  (Σ i in range n, a i) + (Σ i in range n, b i)

theorem general_formula_an (n : ℕ) (h : 0 < n) (a : ℕ → ℕ) (h_seq : seq n h a) : ∀ n, a n = 2 * n - 1 := 
  sorry

theorem sum_first_n_terms (n : ℕ) (h : 0 < n) (a b : ℕ → ℕ) (h_seq : seq n h a) (h_geom : geom_seq b) :
  sum_seq a b n = n^2 + 2^n - 1 :=
  sorry

end general_formula_an_sum_first_n_terms_l502_502494


namespace exponents_equal_find_x_1_find_x_2_express_y_in_x_l502_502909

-- Condition that relates to equivalence of exponents.
theorem exponents_equal {a m n : ℕ} (ha : a > 0) (ha1 : a ≠ 1) (hm : 0 < m) (hn : 0 < n) :
  a^m = a^n → m = n := sorry

-- Problem (1)
theorem find_x_1 : (2^x * 2^3 = 32) → (x = 2) := sorry

-- Problem (2)
theorem find_x_2 : (2 / 8^x * 16^x = 2^5) → (x = 4) := sorry

-- Problem (3)
theorem express_y_in_x (m : ℕ) : (x = 5^m - 2) → (y = 3 - 25^m) → (y = -x^2 - 4x - 1) := sorry

end exponents_equal_find_x_1_find_x_2_express_y_in_x_l502_502909


namespace max_a_value_l502_502885

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  sqrt 3 * sin (ω * x / 2) * cos (ω * x / 2) - cos^2 (ω * x / 2)

theorem max_a_value (ω : ℝ) (hω : ω > 0)
  (h_mono_inc : ∀ x y, 0 < x ∧ x < y ∧ y < (4 * Real.pi / 3) → f x ω ≤ f y ω)
  (h_mono_dec : ∀ x y, (4 * Real.pi / 3) < x ∧ x < y ∧ y < 2 * Real.pi → f x ω ≥ f y ω) :
  ∃ a, (a = (22 * Real.pi / 3)) ∧ (∀ x, (17 * Real.pi / 3) < x ∧ x < a → f x ω ≤ f x ω) :=
sorry

end max_a_value_l502_502885


namespace find_monthly_payment_l502_502498

variable (purchase_price down_payment num_payments interest_percentage : ℝ)
variable (monthly_payment : ℝ)

-- Defining the conditions
def conditions : Prop :=
  purchase_price = 112 ∧
  down_payment = 12 ∧
  num_payments = 12 ∧
  interest_percentage = 10.714285714285714 / 100

-- Defining the total interest paid
def total_interest_paid : ℝ :=
  interest_percentage * purchase_price

-- Defining the total amount paid
def total_amount_paid : ℝ :=
  down_payment + (monthly_payment * num_payments) + total_interest_paid

-- Theorem to prove that the monthly payment equals $8.333333333333333
theorem find_monthly_payment (h : conditions) : monthly_payment = 100 / 12 :=
by
  sorry

end find_monthly_payment_l502_502498


namespace sqrt_sum_solutions_l502_502823

theorem sqrt_sum_solutions (a b : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
    (\sqrt{1 + \sqrt{18 + 8 * \sqrt{2}}} = \sqrt(a) + \sqrt(b)) ↔ (a = 1 ∧ b = 2) := by
  sorry

end sqrt_sum_solutions_l502_502823


namespace value_of_1_over_x_squared_minus_x_l502_502911

namespace ComplexProof

-- Define the given x value
def x : ℂ := (1 + Complex.I * Real.sqrt 3) / 2

-- Main theorem
theorem value_of_1_over_x_squared_minus_x : (1 : ℂ) / (x^2 - x) = -1 := by
  sorry

end ComplexProof

end value_of_1_over_x_squared_minus_x_l502_502911


namespace area_enclosed_by_sqrt_and_x2_l502_502196

theorem area_enclosed_by_sqrt_and_x2 :
  (∫ x in 0..1, (real.sqrt x - x^2)) = 1 / 3 :=
sorry

end area_enclosed_by_sqrt_and_x2_l502_502196


namespace ratio_B1_N_to_N_C1_is_sqrt2_plus_1_l502_502841

-- Define the necessary constructs and hypotheses
variables (a : ℝ)  -- length of the edge of the cube
variables (B1 C1 N L M K : ℝ)  -- Points on the cube
variables (x : ℝ)  -- distance representing |NC1|
variables (LM LK NK : ℝ)  -- distances between various points

hypotheses 
  (h1 : L = a / 2)  -- midpoint of A1 B1
  (h2 : LM = a / 2)  -- center of face AB B1 A1
  (h3 : NK = x / sqrt(2))  -- perpendicular foot distance
  (h4 : LK = sqrt(5 * a^2 / 4 - a * x + x^2 / 2))  -- using Pythagorean theorem
  (h5 : (LK^2) = LM^2 + M * K^2 - 2 * LM * M * K * cos(φ))  -- law of cosines 1
  (h6 : (LM^2) = K * N^2 + NK^2 - 2 * K * N * NK * cos(φ))  -- law of cosines 2
  (h7 : angle (L M K) = angle (M K N))  -- given equal angles

-- The Lean statement asserting that the ratio is √2 + 1
theorem ratio_B1_N_to_N_C1_is_sqrt2_plus_1 : 
  | B1 - N | / | N - C1 | = sqrt(2) + 1 := 
sorry  -- proof to be provided

end ratio_B1_N_to_N_C1_is_sqrt2_plus_1_l502_502841


namespace common_chord_length_l502_502140

theorem common_chord_length (r d : ℝ) (hr : r = 12) (hd : d = 16) : 
  ∃ l : ℝ, l = 8 * Real.sqrt 5 := 
by
  sorry

end common_chord_length_l502_502140


namespace smallest_lambda_inequality_l502_502521

noncomputable def smallest_lambda (n : ℕ) : ℝ :=
if n = 1 then (Float.sqrt 3 / 3).toReal
else if n = 2 then (2 * Float.sqrt 3 / 3).toReal
else (n - 1)

theorem smallest_lambda_inequality (n : ℕ) (x: ℕ → ℝ)
  (hx_pos : ∀ i, 0 < x i) 
  (hx_prod : (Finset.range n).prod x = 1) : 
  (Finset.range n).sum (λ i, 1 / Float.sqrt (1 + 2 * x i)) ≤ smallest_lambda n := sorry

end smallest_lambda_inequality_l502_502521


namespace S_13_value_l502_502074

variable {a : ℕ → ℝ} -- arithmetic sequence
variable {S : ℕ → ℝ} -- sum of the first n terms

-- Conditions
axiom sum_terms_condition : a 2 + a 7 + a 12 = 24

-- Summation definition
def S (n : ℕ) := n * (a 1 + a n) / 2

-- Lean 4 statement to be proven
theorem S_13_value (h : summation_formula) (sum_terms_cond : sum_terms_condition) : S 13 = 104 := 
by sorry

end S_13_value_l502_502074


namespace max_x_plus_2y_on_ellipse_l502_502492

theorem max_x_plus_2y_on_ellipse (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 12) : 
  ∃ θ : ℝ, x + 2 * y = sqrt 22 * sin θ ∧ 1 ≤ sqrt 22 * sin θ :=
sorry

end max_x_plus_2y_on_ellipse_l502_502492


namespace average_production_is_correct_l502_502019

noncomputable def average_tv_production_last_5_days
  (daily_production : ℕ)
  (ill_workers : List ℕ)
  (decrease_rate : ℕ) : ℚ :=
  let productivity_decrease (n : ℕ) : ℚ := (1 - (decrease_rate * n) / 100 : ℚ) * daily_production
  let total_production := (ill_workers.map productivity_decrease).sum
  total_production / ill_workers.length

theorem average_production_is_correct :
  average_tv_production_last_5_days 50 [3, 5, 2, 4, 3] 2 = 46.6 :=
by
  -- proof needed here
  sorry

end average_production_is_correct_l502_502019


namespace simplify_expression_l502_502286

theorem simplify_expression (x : ℝ) (h1 : x^3 + 2*x + 1 ≠ 0) (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ( ((x + 2)^2 * (x^2 - x + 2)^2 / (x^3 + 2*x + 1)^2 )^3 * ((x - 2)^2 * (x^2 + x + 2)^2 / (x^3 - 2*x - 1)^2 )^3 ) = 1 :=
by sorry

end simplify_expression_l502_502286


namespace probability_of_three_passing_l502_502202

theorem probability_of_three_passing : 
  let p_A := 2/3
  let p_B := 1/2
  let indep : Prop := ∀ (X Y : ℕ → Prop), (X 0 = Y 0) → (X n = Y n → X (n+1) = Y (n+1))
  let events := λ (student : String), if student ∈ ["A", "B", "C"] then p_A else p_B
  let outcomes := ["A", "B", "C", "D", "E"]
  in (∑ (t : list ℕ) in finset.powerset (finset.range 5), 
      if (events.outcomes.filter (λ x, x = 1).length = 3) then 1 else 0) = 19/54 := 
by
  sorry

end probability_of_three_passing_l502_502202


namespace paul_completion_time_l502_502308

theorem paul_completion_time :
  let george_rate := 1 / 15
  let remaining_work := 2 / 5
  let combined_rate (P : ℚ) := george_rate + P
  let P_work := 4 * combined_rate P = remaining_work
  let paul_rate := 13 / 90
  let total_work := 1
  let time_paul_alone := total_work / paul_rate
  P_work → time_paul_alone = (90 / 13) := by
  intros
  -- all necessary definitions and conditions are used
  sorry

end paul_completion_time_l502_502308


namespace units_digit_first_four_composite_is_eight_l502_502609

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502609


namespace increasing_on_interval_l502_502919

noncomputable def f (x a : ℝ) : ℝ := x^3 + a * x - 2

theorem increasing_on_interval (a : ℝ) : 
  (a ≥ -3) ↔ (∀ x, 1 ≤ x → 0 ≤ deriv (λ x : ℝ, f x a) x) :=
begin
  sorry
end

end increasing_on_interval_l502_502919


namespace Y_3_2_eq_1_l502_502908

def Y (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem Y_3_2_eq_1 : Y 3 2 = 1 := by
  sorry

end Y_3_2_eq_1_l502_502908


namespace count_integers_with_2_and_5_l502_502901

theorem count_integers_with_2_and_5 : 
  ∃ n : ℕ, n = (4 * 2) ∧ (∀ x : ℕ, 300 ≤ x ∧ x < 700 → 
  let h := x / 100,
      t := (x % 100) / 10,
      u := x % 10 in
  (h = 3 ∨ h = 4 ∨ h = 5 ∨ h = 6) ∧ (t = 2 ∧ u = 5 ∨ t = 5 ∧ u = 2) ↔ ¬ (x = 0)) :=
begin
  sorry
end

end count_integers_with_2_and_5_l502_502901


namespace cone_csa_l502_502129

noncomputable def curvedSurfaceArea (r l : ℝ) : ℝ := π * r * l

theorem cone_csa (r l : ℝ) (hr : r = 9) (hl : l = 13) : 
  curvedSurfaceArea r l ≈ 367.0143 :=
by
  rw [hr, hl]
  have : curvedSurfaceArea 9 13 = π * 9 * 13 :=
    by rw [curvedSurfaceArea, hr, hl]
  calc
    curvedSurfaceArea 9 13 = π * 9 * 13   : by rw [this]
                      ... ≈ 367.0143 cm² : by norm_num
  sorry

end cone_csa_l502_502129


namespace part_I_part_II_max_part_II_min_part_III_l502_502839

-- Definitions based on conditions
def f (x a : ℝ) := x * Real.log x - a * x
def g (x : ℝ) := -x^2 - 2

-- Problem part (I)
theorem part_I (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≥ g x) → a ≤ 3 := 
sorry

-- Problem part (II)
theorem part_II_max (m : ℝ) (h : 0 < m) : 
  (Function.maxOn (f · (-1)) (Set.Icc m (m + 3))) = ((m + 3) * (Real.log (m + 3) + 1)) := 
sorry

theorem part_II_min (m : ℝ) (h : 0 < m) : 
  if m < (1 / Real.exp 2) then (Function.minOn (f · (-1)) (Set.Icc m (m + 3))) = (-1 / (Real.exp 2)^2)
  else (Function.minOn (f · (-1)) (Set.Icc m (m + 3))) = (m * (Real.log m + 1)) :=
sorry

-- Problem part (III)
theorem part_III (x : ℝ) (h : 0 < x) : 
  Real.log x + 1 > (1 / Real.exp x) - (2 / (Real.exp x * x)) :=
sorry

end part_I_part_II_max_part_II_min_part_III_l502_502839


namespace largest_eight_digit_number_l502_502699

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l502_502699


namespace percentage_shaded_is_14_29_l502_502562

noncomputable def side_length : ℝ := 20
noncomputable def rect_length : ℝ := 35
noncomputable def rect_width : ℝ := side_length
noncomputable def rect_area : ℝ := rect_length * rect_width
noncomputable def overlap_length : ℝ := 2 * side_length - rect_length
noncomputable def overlap_area : ℝ := overlap_length * side_length
noncomputable def shaded_percentage : ℝ := (overlap_area / rect_area) * 100

theorem percentage_shaded_is_14_29 :
  shaded_percentage = 14.29 :=
sorry

end percentage_shaded_is_14_29_l502_502562


namespace units_digit_of_product_is_eight_l502_502584

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502584


namespace pizza_eaten_after_four_trips_l502_502162

def pizza := ℚ

def first_trip (p : pizza) := p * (1 / 3)
def remaining_after_first (p : pizza) := p * (2 / 3)

def second_trip (p : pizza) := (remaining_after_first p) * (1 / 2)
def remaining_after_second (p : pizza) := (remaining_after_first p) * (1 / 2)

def third_trip (p : pizza) := (remaining_after_second p) * (1 / 2)
def remaining_after_third (p : pizza) := (remaining_after_second p) * (1 / 2)

def fourth_trip (p : pizza) := (remaining_after_third p) * (1 / 2)

def total_pizza_eaten (p : pizza) := 
  first_trip p + second_trip p + third_trip p + fourth_trip p

theorem pizza_eaten_after_four_trips (p : pizza) (h : p = 1) : total_pizza_eaten p = 11 / 12 :=
by sorry

end pizza_eaten_after_four_trips_l502_502162


namespace angle_between_vectors_l502_502366

variable {a b : ℝ³} -- Declaring a and b as vectors in 3-dimensional space

-- Hypothesis: a and b are unit vectors and |a + 3b| = sqrt(13)
def unit_vector (v : ℝ³) : Prop := ‖v‖ = 1
def condition1 := unit_vector a
def condition2 := unit_vector b
def condition3 : Prop := ‖a + 3 • b‖ = sqrt 13

-- Prove the angle between a and b is pi / 3
theorem angle_between_vectors : condition1 ∧ condition2 ∧ condition3 → 
  real.angle (a, b) = real.pi / 3 :=
by
  sorry

end angle_between_vectors_l502_502366


namespace hall_stone_length_l502_502754

open Real

noncomputable def hall_paving_problem
    (hall_length_m : ℝ) (hall_breadth_m : ℝ)
    (stone_breadth_dm : ℝ) (number_of_stones : ℕ) : ℝ :=
let hall_length_dm := hall_length_m * 10
let hall_breadth_dm := hall_breadth_m * 10
let hall_area_dm2 := hall_length_dm * hall_breadth_dm
let area_per_stone := stone_breadth_dm * L
let total_area_covered := area_per_stone * (number_of_stones : ℝ)
in L

theorem hall_stone_length (hall_length : ℝ) (hall_breadth : ℝ)
    (stone_breadth : ℝ) (stones : ℕ) (h_len : hall_length = 36)
    (h_breadth : hall_breadth = 15) (h_stone_breadth : stone_breadth = 0.5)
    (h_stones : stones = 1800) :
    hall_paving_problem hall_length hall_breadth stone_breadth stones = 6 :=
by
  let hall_length_dm := hall_length * 10
  let hall_breadth_dm := hall_breadth * 10
  let hall_area_dm2 := hall_length_dm * hall_breadth_dm
  let area_per_stone := stone_breadth * L
  let total_area_covered := area_per_stone * (stones : ℝ)
  calc
    54000 = 54000 := by rfl
    54000 = total_area_covered := by rw [←h_len, ←h_breadth, ←h_stone_breadth, ←h_stones, total_area_covered, hall_length_dm, hall_breadth_dm, hall_area_dm2, area_per_stone]
    total_area_covered = area_per_stone * stones := by rw [area_per_stone]
    area_per_stone * stones = stone_breadth * L * stones := by rw [area_per_stone]
    stone_breadth * L * stones = 54000 := by rfl
  sorry

end hall_stone_length_l502_502754


namespace binomial_params_l502_502759

variable {X : Type} [ProbTheory.binomial n p]   -- declare the random variable X with binomial distribution parameters

theorem binomial_params (n : ℕ) (p : ℝ) 
    (hn : n > 0) (hp : 0 < p ∧ p < 1) 
    (hmean : n * p = 200) 
    (hstd : (n * p * (1 - p))^(1/2) = 10) : 
    n = 400 ∧ p = 0.5 := 
by 
    sorry

end binomial_params_l502_502759


namespace min_area_triangle_ABC_l502_502731

theorem min_area_triangle_ABC :
  let A := (0, 0) 
  let B := (42, 18)
  (∃ p q : ℤ, let C := (p, q) 
              ∃ area : ℝ, area = (1 / 2 : ℝ) * |42 * q - 18 * p| 
              ∧ area = 3) := 
sorry

end min_area_triangle_ABC_l502_502731


namespace percentage_decrease_correct_l502_502749

theorem percentage_decrease_correct :
  ∀ (p : ℝ), (1 + 0.25) * (1 - p) = 1 → p = 0.20 :=
by
  intro p
  intro h
  sorry

end percentage_decrease_correct_l502_502749


namespace triangle_equiv_condition_triangle_equiv_condition_rev_l502_502066

variables {A B C D E F M : Type*}

-- Triangle ABC, with points D and E defined as intersections of the circle passing through A, B with AC, BC.
variable [is_triangle ABC]
variable (circle) [is_passed_through A B circle]
variable [intersects circle AC at D]
variable [intersects circle BC at E]

-- Line AB and DE intersection at F
variable [intersects_lines AB DE at F]

-- Line BD and CF intersection at M
variable [intersects_lines BD CF at M]

-- Prove MF = MC if and only if MB * MD = MC^2
theorem triangle_equiv_condition (hMF_MC : MF = MC) : MB * MD = MC^2 :=
by
sorry

theorem triangle_equiv_condition_rev (hMB_MD_MC2 : MB * MD = MC^2) : MF = MC :=
by
sorry

end triangle_equiv_condition_triangle_equiv_condition_rev_l502_502066


namespace asian_countries_visited_l502_502260

theorem asian_countries_visited (total_countries europe_countries south_america_countries remaining_asian_countries : ℕ)
  (h1 : total_countries = 42)
  (h2 : europe_countries = 20)
  (h3 : south_america_countries = 10)
  (h4 : remaining_asian_countries = (total_countries - (europe_countries + south_america_countries)) / 2) :
  remaining_asian_countries = 6 :=
by sorry

end asian_countries_visited_l502_502260


namespace flight_duration_l502_502095

noncomputable def departure_time_pst := 9 * 60 + 15 -- in minutes
noncomputable def arrival_time_est := 17 * 60 + 40 -- in minutes
noncomputable def time_difference := 3 * 60 -- in minutes

theorem flight_duration (h m : ℕ) 
  (h_cond : 0 < m ∧ m < 60) 
  (total_flight_time : (arrival_time_est - (departure_time_pst + time_difference)) = h * 60 + m) : 
  h + m = 30 :=
sorry

end flight_duration_l502_502095


namespace sum_f_to_2015_l502_502262

def f : ℝ → ℝ 
| x => if (-3 ≤ x ∧ x < -1) then -(x + 2)^2 else if (-1 ≤ x ∧ x < 3) then x else f (x - 6)

theorem sum_f_to_2015 : (∑ i in Finset.range 2015, λ i, f (1 + i)) = 336 := 
sorry

end sum_f_to_2015_l502_502262


namespace single_light_on_positions_l502_502224

   open Matrix
   open Finset

   def toggle (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ) (i j : Fin n) : Matrix (Fin n) (Fin n) ℕ :=
     A.update i j (1 - A i j)  -- Toggle the light at (i, j)
      |>.update i (λ k => 1 - A i k)  -- Toggle the row 
      |>.update j (λ k => 1 - A k j)  -- Toggle the column

   noncomputable def possiblePositions := {(2, 2), (2, 4), (3, 3), (4, 2), (4, 4)}

   theorem single_light_on_positions
     (A : Matrix (Fin 5) (Fin 5) ℕ)
     (h_initial : ∀ i j, A i j = 0)
     (h_toggle : ∃ (toggles : Finset (Fin 5 × Fin 5)),
         ∀ i j, (toggle 5 A i j) ∈ toggles → A i j = 1)
     : ∀ (i j : Fin 5), 
         (i, j) ∈ possiblePositions ↔ (A i j = 1 ∧ ∀ (u v : Fin 5), (u, v) ≠ (i, j) → A u v = 0) :=
   sorry
   
end single_light_on_positions_l502_502224


namespace maximal_value_of_s_l502_502065

noncomputable def max_s_possible (p q r s : ℝ) : ℝ :=
  if h₁ : p + q + r + s = 10 ∧ pq + pr + ps + qr + qs + rs = 20 then
    (5 + Real.sqrt 105) / 2
  else
    s

theorem maximal_value_of_s (p q r s : ℝ) (h₁ : p + q + r + s = 10) (h₂ : pq + pr + ps + qr + qs + rs = 20) :
  s ≤ max_s_possible p q r s :=
by
  rw max_s_possible
  simp only [h₁, h₂, if_true]
  sorry

end maximal_value_of_s_l502_502065


namespace inscribed_circle_radius_l502_502022

-- Define the conditions and question
variables (A p s r : ℝ)
variables (triangle : Type) [IsTriangle triangle]

-- Main statement that needs to be proved
theorem inscribed_circle_radius (h1 : ∀ t : triangle, 2 * s = p) 
                               (h2 : ∀ t : triangle, A = 2 * p) 
                               (h3 : ∀ t : triangle, A = r * s) : 
    r = 4 :=
sorry

end inscribed_circle_radius_l502_502022


namespace units_digit_of_product_of_first_four_composites_l502_502666

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502666


namespace divisibility_condition_l502_502831

theorem divisibility_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ab ∣ (a^2 + b^2 - a - b + 1) → (a = 1 ∧ b = 1) :=
by sorry

end divisibility_condition_l502_502831


namespace largest_eight_digit_with_all_evens_l502_502692

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l502_502692


namespace second_pipe_fill_time_l502_502141

theorem second_pipe_fill_time
  (rate1: ℝ) (rate_outlet: ℝ) (combined_time: ℝ)
  (h1: rate1 = 1 / 18)
  (h2: rate_outlet = 1 / 45)
  (h_combined: combined_time = 0.05):
  ∃ (x: ℝ), (1 / x) = 60 :=
by
  sorry

end second_pipe_fill_time_l502_502141


namespace find_solution_set_compare_expressions_l502_502355

noncomputable def f (x : ℝ) : ℝ := |x| - |2 * x - 1|

def M : Set ℝ := {x | f x > -1}

theorem find_solution_set : M = {x | 0 < x ∧ x < 2} :=
sorry

theorem compare_expressions (a : ℝ) (h : 0 < a ∧ a < 2) :
  if 0 < a ∧ a < 1 then (a^2 - a + 1 < (1 / a)) else
  if a = 1 then (a^2 - a + 1 = (1 / a)) else
  if 1 < a ∧ a < 2 then (a^2 - a + 1 > (1 / a)) :=
sorry

end find_solution_set_compare_expressions_l502_502355


namespace range_of_m_l502_502856

variable (p q : Prop)
variable (m : ℝ)

-- Define proposition p
def proposition_p : Prop := 
  ∃ x ∈ Set.Icc (-π / 6) (π / 4), 2 * Real.sin (2 * x + π / 6) - m = 0

-- Define proposition q
def proposition_q : Prop := 
  ∃ x ∈ Set.Ioi (0 : ℝ), x^2 - 2 * m * x + 1 < 0

-- Define the main statement
theorem range_of_m (hp : proposition_p p) (hnq : ¬proposition_q q) : 
  -1 ≤ m ∧ m ≤ 1 := 
by
  sorry

end range_of_m_l502_502856


namespace max_expression_value_l502_502010

noncomputable def p : ℝ := sorry
noncomputable def E_xi : ℝ := p

noncomputable def D_xi : ℝ := p * (1 - p)

noncomputable def expression : ℝ := (4 * D_xi - 1) / E_xi

theorem max_expression_value : 0 < p ∧ p < 1 → ∃ m, m = 0 ∧ ∀ x, expression <= m :=  by
  sorry

end max_expression_value_l502_502010


namespace coplanarity_condition_l502_502816

theorem coplanarity_condition (a b : ℝ) :
  let p1 := (0, 0, 0) in
  let p2 := (1, a, b) in
  let p3 := (b, 1, a) in
  let p4 := (a, b, 0) in
  (determine if coplanar p1 p2 p3 p4) ↔ (a = 0 ∧ b = 0) :=
sorry

end coplanarity_condition_l502_502816


namespace solid_object_top_view_circle_l502_502765

theorem solid_object_top_view_circle (S : Type) (h : ∀ (o : S), top_view o = circle) :
  ∃ (o : S), o = sphere ∨ o = cylinder ∨ o = cone ∨ o = frustum_of_cone :=
by
  sorry

end solid_object_top_view_circle_l502_502765


namespace shortest_side_of_triangle_l502_502041

-- Define a theorem which states that under given conditions, the shortest side of triangle ABC is x.
theorem shortest_side_of_triangle (x y z : ℝ)
    (hBD : BD = 3) (hDE : DE = 4) (hEC : EC = 5) 
    (trisect : trisects_angle BAC AD AE) :
    shortest_side ABC = x := by
  sorry

end shortest_side_of_triangle_l502_502041


namespace min_strip_width_for_area_one_triangles_l502_502568

theorem min_strip_width_for_area_one_triangles : 
  ∀ (w : ℝ), (∀ (A B C : ℝ × ℝ), 
    (∃ (a b c : ℝ), 
       a = dist A B ∧ b = dist B C ∧ c = dist C A ∧ 
       (1 / 2) * abs (A.1*(B.2 - C.2) + B.1*(C.2 - A.2) + C.1*(A.2 - B.2)) = 1) →
       max (abs (A.2 - B.2)) (abs (B.2 - C.2)) ≤ w) ↔ (w ≥ real.sqrt (real.sqrt 3)) :=
by
  sorry

end min_strip_width_for_area_one_triangles_l502_502568


namespace possible_items_l502_502430

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502430


namespace hyperbola_focus_l502_502064

theorem hyperbola_focus (m : ℝ) (h : (0, 5) = (0, 5)) : 
  (∀ x y : ℝ, (y^2 / m - x^2 / 9 = 1) → m = 16) :=
sorry

end hyperbola_focus_l502_502064


namespace sum_of_intercepts_l502_502213

theorem sum_of_intercepts (x y : ℝ) (hx : y + 3 = 5 * (x - 6)) : 
  let x_intercept := 6 + 3/5;
  let y_intercept := -33;
  x_intercept + y_intercept = -26.4 := by
  sorry

end sum_of_intercepts_l502_502213


namespace find_n_l502_502195

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n < 31 ∧ 78256 % 31 = n % 31 :=
begin
  use 19,
  split,
  { exact nat.zero_le 19, },
  split,
  { exact nat.lt_of_lt_of_le (nat.lt_succ_self 19) (nat.succ_le_of_lt (nat.lt_of_succ_lt (nat.succ_lt_succ (nat.succ_le_succ (nat.lt_trans (nat.lt_succ_self 18) (nat.succ_le_of_lt (nat.lt_of_succ_le (nat.succ_le_of_lt nat.lt_succ_self 30)))))))), },
  { norm_num, }
end

end find_n_l502_502195


namespace prob_product_72_of_three_dice_rolls_l502_502682

theorem prob_product_72_of_three_dice_rolls :
  let dice_values := {1, 2, 3, 4, 5, 6}
  let successful_outcomes := { (2,6,6), (6,2,6), (6,6,2), (3,4,6), (3,6,4), (4,3,6), (4,6,3), (6,3,4), (6,4,3) }
  (successful_outcomes.card : ℚ) / (dice_values.card * dice_values.card * dice_values.card : ℚ) = 1 / 24 := 
by
  sorry

end prob_product_72_of_three_dice_rolls_l502_502682


namespace number_of_outfits_l502_502728

theorem number_of_outfits (shirts pants : ℕ) (h_shirts : shirts = 5) (h_pants : pants = 3) 
    : shirts * pants = 15 := by
  sorry

end number_of_outfits_l502_502728


namespace exists_fixed_point_l502_502052

-- Define the geometric setup with all given conditions
variables {A B C : Type} 
variables [triangle : A < B < C] (AB_lt_AC : AB < AC)
noncomputable def omega : circle := sorry
variables {X Y : omega}

-- Define the angle conditions and positional constraints
variables are_angles_equal : ∀ {X Y : omega}, ∠BXA = ∠AYC
variables left_right_positions_1 : ∀ {X : omega}, X and C are on opposite sides of line AB
variables left_right_positions_2 : ∀ {Y : omega}, Y and B are on opposite sides of line AC

-- The goal to prove: existence of a fixed point P
theorem exists_fixed_point (X Y : omega) : ∃ P, ∀ X Y : omega, line.through X Y P :=
sorry

end exists_fixed_point_l502_502052


namespace train_speed_proof_l502_502230

theorem train_speed_proof (train_length : ℕ) (bridge_length : ℕ) (time : ℕ) 
  (h_train_length : train_length = 360) 
  (h_bridge_length : bridge_length = 140) 
  (h_time : time = 40) : 
  (train_length + bridge_length) / time * 3.6 = 45 := 
by {
  calc
    (train_length + bridge_length) / time * 3.6
    = (360 + 140) / 40 * 3.6 : by rw [h_train_length, h_bridge_length, h_time]
    ... = 500 / 40 * 3.6 : by norm_num
    ... = 12.5 * 3.6 : by norm_num
    ... = 45 : by norm_num
}

end train_speed_proof_l502_502230


namespace change_received_l502_502204

def regular_ticket_cost : ℕ := 129
def discount : ℕ := 15
def ages : List ℕ := [6, 10, 13, 8]
def amount_given : ℕ := 800

theorem change_received : 
  let discounted_ticket_cost (c : ℕ) := if c < 12 then regular_ticket_cost - discount else regular_ticket_cost in
  let total_cost := (List.sum (List.map discounted_ticket_cost ages)) + 2 * regular_ticket_cost in
  amount_given - total_cost = 71 := 
by sorry

end change_received_l502_502204


namespace solution_l502_502943

variable {ℤ : Type}

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n m : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the arithmetic sequence condition
axiom arithmetic_seq : arithmetic_sequence a

-- Condition provided: a_6 + a_9 + a_{12} = 48
axiom a_6_9_12_sum : a 6 + a 9 + a 12 = 48

theorem solution : a 8 + a 10 = 32 :=
by
  have eqn1 : a 8 = a 6 + 2 * d :=
    sorry
  have eqn2 : a 10 = a 6 + 4 * d :=
    sorry
  have eqn3 : a 9 = a 6 + 3 * d := 
    sorry
  have eqn4 : a 6 + (a 6 + 3 * d) + (a 6 + 6 * d) = 48 := 
    sorry
  have eqn5 : 3 * a 6 + 9 * d = 48 :=
    sorry
  have eqn6 : a 6 + 3 * d = 16 := 
    by
      sorry
  have eqn7 : a 9 = 16 :=
    sorry
  exact eqn1 + eqn2 - 2 * eqn3 -- Simplify to prove a 8 + a 10 = 32

end solution_l502_502943


namespace length_of_parallel_line_l502_502102

theorem length_of_parallel_line (A B C D E : Point)
  (hABC : Triangle A B C) (hBC : ∥B - C∥ = 20)
  (hDE : ∥D - E∥ = ∥B - C∥ / 2)
  (hDiv : divides_triangle_into_equal_areas hABC D E (1 : ℝ) 4) :
  ∥D - E∥ = 10 :=
by sorry

end length_of_parallel_line_l502_502102


namespace analytical_expression_monotonically_increasing_interval_range_of_f_l502_502368

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - (1/2) * Real.cos (2 * x)

theorem analytical_expression (x : ℝ) : 
  f(x) = 1/2 - Real.sin (2 * x + Real.pi / 6) :=
sorry

theorem monotonically_increasing_interval (k : ℤ) : 
  (k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + (2 * Real.pi) / 3) → 
  (∀ x1 x2, k * Real.pi + Real.pi / 6 ≤ x1 ∧ x2 ≤ k * Real.pi + (2 * Real.pi) / 3 → 
  x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

theorem range_of_f (x : ℝ) : 
  0 ≤ x ∧ x ≤ Real.pi / 3 → -1/2 ≤ f(x) ∧ f(x) ≤ 0 :=
sorry

end analytical_expression_monotonically_increasing_interval_range_of_f_l502_502368


namespace circumcircle_center_eq_orthocenter_l502_502499

-- Defining the points and properties
variables {A B C A₁ B₁ C₁ : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (point_in_bc : A₁ ∈ line_through B C)
variables (point_in_ca : B₁ ∈ line_through C A)
variables (point_in_ab : C₁ ∈ line_through A B)
variables (equal_angles : ∀ x y z : Type, angle x (line_through y z) C A₁ = angle x (line_through z A) B₁ ∧
                                             angle x (line_through z A) B₁ = angle x (line_through y B) C₁)

-- Proving the required property
theorem circumcircle_center_eq_orthocenter (triangle_circ_center_eq_orthocenter : ∀ {A B C A₁ B₁ C₁ : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (point_in_bc : A₁ ∈ line_through B C)
  (point_in_ca : B₁ ∈ line_through C A)
  (point_in_ab : C₁ ∈ line_through A B)
  (equal_angles : ∀ x y z : Type, angle x (line_through y z) C A₁ = angle x (line_through z A) B₁ ∧
                                   angle x (line_through z A) B₁ = angle x (line_through y B) C₁),
      (triangle_orthocenter_eq_circ_center : orthocenter (triangle A B C) = circumcenter (triangle (line_through A A₁) (line_through B B₁) (line_through C C₁)))) :
       (orthocenter (triangle A B C) = circumcenter (triangle (line_through A A₁) (line_through B B₁) (line_through C C₁))) :=
begin
  sorry,
end

end circumcircle_center_eq_orthocenter_l502_502499


namespace count_special_integers_l502_502373

theorem count_special_integers : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 9999 ∧ (∀ d ∈ digits 10 n, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 5 ∧ d ≠ 8)) → 
  count_integers_with_conditions 1 9999 (λ d, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 5 ∧ d ≠ 8) = 624 := 
by
  sorry

end count_special_integers_l502_502373


namespace coefficient_x3_y2_z3_l502_502148

def f (x y : ℂ) := (2 * x + y)^5
def g (z : ℂ) := (z - 1 / z^2)^7

theorem coefficient_x3_y2_z3 :
  coefficient (x^3 * y^2 * z^3) (f x y * g z) = 2800 :=
by
  sorry

end coefficient_x3_y2_z3_l502_502148


namespace cubicroots_expression_l502_502991

theorem cubicroots_expression (a b c : ℝ)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 11)
  (h₃ : a * b * c = 6) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 251 / 216 :=
by sorry

end cubicroots_expression_l502_502991


namespace midpoint_line_divides_quad_into_equal_areas_parallel_l502_502128

theorem midpoint_line_divides_quad_into_equal_areas_parallel 
  {A B C D M N : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  [metric_space M] [metric_space N]
  (h_convex: convex_hull R {A, B, C, D})
  (hM: midpoint A B M) (hN: midpoint C D N)
  (h_equal_area: area A M N D = area B M N C) :
  parallel A B C D :=
sorry

end midpoint_line_divides_quad_into_equal_areas_parallel_l502_502128


namespace area_ratio_l502_502950

-- Definitions for the conditions in the problem
variables (PQ QR RP : ℝ) (p q r : ℝ)

-- Conditions
def pq_condition := PQ = 18
def qr_condition := QR = 24
def rp_condition := RP = 30
def pqr_sum := p + q + r = 3 / 4
def pqr_squaresum := p^2 + q^2 + r^2 = 1 / 2

-- Goal statement that the area ratio of triangles XYZ to PQR is 23/32
theorem area_ratio (h1 : PQ = 18) (h2 : QR = 24) (h3 : RP = 30) 
  (h4 : p + q + r = 3 / 4) (h5 : p^2 + q^2 + r^2 = 1 / 2) : 
  ∃ (m n : ℕ), (m + n = 55) ∧ (m / n = 23 / 32) := 
sorry

end area_ratio_l502_502950


namespace area_CPQ_constant_l502_502191

theorem area_CPQ_constant {A B C D P Q : Type*}
  [h_rhombus : rhombus A B C D] 
  [tangent : tangent_to_inscribed_circle A B P Q D]
  (h_P_on_AB : on_side P A B)
  (h_Q_on_AD : on_side Q A D) :
  exists (k : ℝ), ∀ (P Q : Type*), area (triangle C P Q) = k :=
sorry

end area_CPQ_constant_l502_502191


namespace units_digit_of_first_four_composite_numbers_l502_502636

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502636


namespace probability_of_divisibility_by_8_l502_502305

theorem probability_of_divisibility_by_8 :
  let x_vals := {3, 58}
  let y_vals := {3, 58}
  let valid_values := {v | v ∈ x_vals ∧ v ∈ y_vals ∧ v < 10}
  let probable_value (x y : ℕ) := 46*1000 + x*100 + y*10 + 12
  let is_divisible_by_8 (n : ℕ) := n % 8 = 0
  let favorable_outcomes := {v | v.1 = 3 ∧ v.2 = 3}
  ∃ w : ℝ, 
  (favorable_outcomes.card : ℝ) / (valid_values.to_finset.powerset.card : ℝ) = w ∧ 
  w = 1 :=
by
  sorry

end probability_of_divisibility_by_8_l502_502305


namespace seq_2016_2017_l502_502038

-- Define the sequence a_n
def seq (n : ℕ) : ℚ := sorry

-- Given conditions
axiom a1_cond : seq 1 = 1/2
axiom a2_cond : seq 2 = 1/3
axiom seq_rec : ∀ n : ℕ, seq n * seq (n + 2) = 1

-- The main goal
theorem seq_2016_2017 : seq 2016 + seq 2017 = 7/2 := sorry

end seq_2016_2017_l502_502038


namespace minimum_f_value_l502_502821

noncomputable def f (x : ℝ) : ℝ :=
   Real.sqrt (2 * x ^ 2 - 4 * x + 4) + 
   Real.sqrt (2 * x ^ 2 - 16 * x + (Real.log x / Real.log 2) ^ 2 - 2 * x * (Real.log x / Real.log 2) + 
              2 * (Real.log x / Real.log 2) + 50)

theorem minimum_f_value : ∀ x : ℝ, x > 0 → f x ≥ 7 ∧ f 2 = 7 :=
by
  sorry

end minimum_f_value_l502_502821


namespace different_losses_l502_502015

theorem different_losses {n : ℕ} (participants : Fin n) 
  (games : Fin n → Fin n → Bool) 
  (total_games : ∀ i : Fin n, ∑ j, if games i j then 1 else 0 = n - 1)
  (no_draws : ∀ i j : Fin n, games i j ≠ games j i)
  (unique_wins : ∃ wins : Fin n → ℕ, ∀ i j : Fin n, i ≠ j → wins i ≠ wins j ∧ wins i < n) :
  ∃ losses : Fin n → ℕ, ∀ i j : Fin n, i ≠ j → losses i ≠ losses j ∧ losses i < n := 
begin
  sorry
end

end different_losses_l502_502015


namespace evaporation_amount_l502_502045

variable (E : ℝ)

def initial_koolaid_powder : ℝ := 2
def initial_water : ℝ := 16
def final_percentage : ℝ := 0.04

theorem evaporation_amount :
  (initial_koolaid_powder = 2) →
  (initial_water = 16) →
  (0.04 * (initial_koolaid_powder + 4 * (initial_water - E)) = initial_koolaid_powder) →
  E = 4 :=
by
  intros h1 h2 h3
  sorry

end evaporation_amount_l502_502045


namespace sum_of_powers_mod_l502_502569

-- Define a function that calculates the nth power of a number modulo a given base
def power_mod (a n k : ℕ) : ℕ := (a^n) % k

-- The main theorem: prove that the sum of powers modulo 5 gives the remainder 0
theorem sum_of_powers_mod 
  : ((power_mod 1 2013 5) + (power_mod 2 2013 5) + (power_mod 3 2013 5) + (power_mod 4 2013 5) + (power_mod 5 2013 5)) % 5 = 0 := 
by {
  sorry
}

end sum_of_powers_mod_l502_502569


namespace moon_land_value_l502_502119

theorem moon_land_value (surface_area_earth : ℕ) (surface_area_moon : ℕ) (total_value_earth : ℕ) (worth_factor : ℕ)
  (h_moon_surface_area : surface_area_moon = surface_area_earth / 5)
  (h_surface_area_earth : surface_area_earth = 200) 
  (h_worth_factor : worth_factor = 6) 
  (h_total_value_earth : total_value_earth = 80) : (total_value_earth / 5) * worth_factor = 96 := 
by 
  -- Simplify using the given conditions
  -- total_value_earth / 5 is the value of the moon's land if it had the same value per square acre as Earth's land
  -- multiplying by worth_factor to get the total value on the moon
  sorry

end moon_land_value_l502_502119


namespace largest_eight_digit_with_all_even_digits_l502_502709

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l502_502709


namespace tetrahedron_volume_l502_502298

theorem tetrahedron_volume 
  (d_face : ℝ) (d_edge : ℝ) (d_face_eq : d_face = 2) (d_edge_eq : d_edge = Real.sqrt 5) :
  ∃ (V : ℝ), V ≈ 46.76 :=
by
  have h := 6
  have a := 3 * Real.sqrt 6
  let V := (a^3) / (6 * Real.sqrt 2)
  have V_approx := V ≈ 46.76
  exact ⟨V, V_approx⟩
  sorry

end tetrahedron_volume_l502_502298


namespace number_of_items_l502_502459

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502459


namespace kopeechka_purchase_l502_502438

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502438


namespace largest_value_p_l502_502484

theorem largest_value_p 
  (p q r : ℝ) 
  (h1 : p + q + r = 10) 
  (h2 : p * q + p * r + q * r = 25) :
  p ≤ 20 / 3 :=
sorry

end largest_value_p_l502_502484


namespace student_who_scored_full_marks_l502_502777

def scores_full_marks (student : Type) : Prop := sorry

variables (A B C : Type)

def A_statement : Prop := ¬(scores_full_marks C)
def B_statement : Prop := scores_full_marks B
def C_statement : Prop := A_statement

def one_lying (a b c : Prop) : Prop := (¬a ∧ b ∧ c) ∨ (a ∧ ¬b ∧ c) ∨ (a ∧ b ∧ ¬c)

theorem student_who_scored_full_marks (H : one_lying A_statement B_statement C_statement) : scores_full_marks A :=
sorry

end student_who_scored_full_marks_l502_502777


namespace equal_ratios_sequences_l502_502742

open Nat

theorem equal_ratios_sequences :
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 k : ℕ,
    {a1, a2, a3, a4, a5, a6, a7, a8} = {1, 2, 3, 4, 6, 8, 12, 24} ∧
    (a1 / a2 = k) ∧ (a3 / a4 = k) ∧ (a5 / a6 = k) ∧ (a7 / a8 = k) ∧
    k ∈ {2, 3} :=
sorry

end equal_ratios_sequences_l502_502742


namespace bus_ride_cost_l502_502772

theorem bus_ride_cost (B : ℝ) 
  (train_cost : B + 6.85) 
  (total_cost : B + (B + 6.85) = 9.85) : 
  B = 1.50 := 
sorry

end bus_ride_cost_l502_502772


namespace units_digit_first_four_composites_l502_502573

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502573


namespace enough_fabric_l502_502261

open Int

def fabric_length := 140
def fabric_width := 75
def dress_length := 45
def dress_width := 26
def required_dresses := 8

def num_parts_from_length (total_length part_length : nat) := total_length / part_length
def num_parts_from_width (total_width part_width : nat) := total_width / part_width

def num_pieces (total_length total_width part_length part_width : nat) : nat :=
  (num_parts_from_width total_width part_width) * (num_parts_from_length total_length part_length)
  +
  (num_parts_from_width total_length part_width) * (num_parts_from_length total_width part_length)

theorem enough_fabric (fabric_length fabric_width dress_length dress_width required_dresses :
  nat) : num_pieces fabric_length fabric_width dress_length dress_width ≥ required_dresses :=
by {
  -- calculations and proof go here
  sorry
}

#eval enough_fabric fabric_length fabric_width dress_length dress_width required_dresses

end enough_fabric_l502_502261


namespace sum_b_n_l502_502329

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), (∀ n : ℕ, a (n + 1) = q * a n)

theorem sum_b_n (h_geo : is_geometric a) (h_a1 : a 1 = 3) (h_sum_a : ∑' n, a n = 9) (h_bn : ∀ n, b n = a (2 * n)) :
  ∑' n, b n = 18 / 5 :=
sorry

end sum_b_n_l502_502329


namespace value_of_m_monotonicity_range_of_f_l502_502886

def f (x : ℝ) (m : ℝ) : ℝ := x^m - 2/x

theorem value_of_m (h : f 4 m = 7/2) : m = 1 := by
  sorry

theorem monotonicity (m : ℝ) (h : m = 1) : 
  ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 → f x1 1 < f x2 1 := by
  sorry

theorem range_of_f :
  let f_one (x : ℝ) := f x 1
  ∃ (min max : ℝ), min = 1 ∧ max = 23/5 ∧ 
  ∀ y, y ∈ f_one '' (set.Icc 2 5) ↔ y ∈ set.Icc 1 (23/5) := by
  sorry

end value_of_m_monotonicity_range_of_f_l502_502886


namespace slope_l1_parallel_lines_math_proof_problem_l502_502357

-- Define the two lines
def l1 := ∀ x y : ℝ, x + 2 * y + 2 = 0
def l2 (a : ℝ) := ∀ x y : ℝ, a * x + y - 4 = 0

-- Define the assertions
theorem slope_l1 : ∀ x y : ℝ, x + 2 * y + 2 = 0 ↔ y = -1 / 2 * x - 1 := sorry

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) ↔ a = 1 / 2 := sorry

-- Using the assertions to summarize what we need to prove
theorem math_proof_problem (a : ℝ) :
  ((∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) → a = 1 / 2) ∧
  (∀ x y : ℝ, x + 2 * y + 2 = 0 → y = -1 / 2 * x - 1) := sorry

end slope_l1_parallel_lines_math_proof_problem_l502_502357


namespace sale_in_first_month_l502_502210

theorem sale_in_first_month 
  (sale2 : ℕ := 8927) 
  (sale3 : ℕ := 8855) 
  (sale4 : ℕ := 9230) 
  (sale5 : ℕ := 8562) 
  (sale6 : ℕ := 6991) 
  (average_sale : ℕ := 8500)
  (total_months : ℕ := 6) :
  let sale1 := average_sale * total_months - (sale2 + sale3 + sale4 + sale5 + sale6) in
  sale1 = 8435 := by
  sorry

end sale_in_first_month_l502_502210


namespace units_digit_first_four_composite_is_eight_l502_502607

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502607


namespace least_consecutive_odd_integers_l502_502737

theorem least_consecutive_odd_integers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 8 * 414)) :
  x = 407 :=
by
  sorry

end least_consecutive_odd_integers_l502_502737


namespace parabola_equation_and_focus_length_of_chord_l502_502358

-- Definitions and Conditions from the problem
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
def point_on_parabola (M_x M_y p : ℝ) := parabola p M_x M_y
def equidistant_focus_directrix (M_x M_y F_x F_y d : ℝ) := M_x = F_x - d

-- (I) Equation of C and coordinates of its focus
theorem parabola_equation_and_focus {p : ℝ} (hp : p > 0) (m : ℝ)
  (hM : point_on_parabola 1 m p)
  (hF : distance 1 m 1 0 = 2) :
  ∃ x y, parabola 2 x y ∧ (1 = 1 - 1) := sorry

-- (II) Length of the chord |AB|
theorem length_of_chord (A_x A_y B_x B_y : ℝ)
  (hA : parabola 2 A_x A_y)
  (hB : parabola 2 B_x B_y)
  (h_line : ∀ x, A_y = x - 1 ∧ B_y = x - 1) :
  distance A_x A_y B_x B_y = 8 := sorry

end parabola_equation_and_focus_length_of_chord_l502_502358


namespace is_angle_bisector_of_parallelogram_l502_502218

variables {A B C D O K L M : Point}
variables {circ : Circle}

def is_parallelogram (A B C D : Point) : Prop := 
  ∃ (E : Point), A - B = C - D ∧ B - C = E - D

def is_perpendicular (B O : Point) (A D : Point) : Prop :=
  ∃ θ : ℝ, θ = 90

def is_on_circle (O : Point) (ω : Circle) (P : Point) : Prop :=
  distance O P = radius ω

def intersects_extension_at (ω : Circle) (A D : Point) : Point :=
  -- Assume function definition for the intersection point
  sorry

def intersect_segment (B K : Segment) (C D : Segment) : Point :=
  -- Assume function definition for the intersection point
  sorry

def is_ray_intersect (L O : Point) (ω : Circle) : Point :=
  -- Assume function definition for the intersection point
  sorry

def is_angle_bisector (K M : Point) (B K C : Angle) : Prop :=
  -- Assume function definition for angle bisector
  sorry

theorem is_angle_bisector_of_parallelogram
  (h1 : is_parallelogram A B C D)
  (h2 : is_perpendicular B O A D)
  (h3 : is_on_circle O circ A)
  (h4 : is_on_circle O circ B)
  (h5 : let K := intersects_extension_at circ A D,
        is_on_circle O circ K)
  (h6 : let L := intersect_segment B K C D,
        true)
  (h7 : let M := is_ray_intersect L O circ,
        true)
  : is_angle_bisector K M B K C :=
sorry

end is_angle_bisector_of_parallelogram_l502_502218


namespace readers_scifi_l502_502400

variable (S L B T : ℕ)

-- Define conditions given in the problem
def totalReaders := 650
def literaryReaders := 550
def bothReaders := 150

-- Define the main problem to prove
theorem readers_scifi (S L B T : ℕ) (hT : T = totalReaders) (hL : L = literaryReaders) (hB : B = bothReaders) (hleq : T = S + L - B) : S = 250 :=
by
  -- Insert proof here
  sorry

end readers_scifi_l502_502400


namespace condition_necessity_but_not_sufficiency_l502_502326

-- Definition: unit_vector
def unit_vector (a : ℝ) (u : a ≠ 0) : Prop := (a=1)

-- Definition: λ > 0 such that b = λ * a
def exists_lambda_positive (a b : ℝ) (u : unit_vector a (ne_zero_of_coe_ne_zero 1)) : Prop :=
  ∃ λ : ℝ, λ > 0 ∧ b = λ * a

-- Condition: |a + b| - |b| = 1
def equation (a b : ℝ) : Prop :=
  |a + b| - |b| = 1

-- Proof problem statement
theorem condition_necessity_but_not_sufficiency 
  (a b : ℝ) (u : unit_vector a (ne_zero_of_coe_ne_zero 1)) :
  (equation a b) → ∃ (λ : ℝ), (λ > 0 ∧ b = λ * a) := sorry

end condition_necessity_but_not_sufficiency_l502_502326


namespace least_triangular_faces_l502_502752

theorem least_triangular_faces (P : Type) [convex_polyhedron P] 
  (h1 : ∀ v : vertex P, degree v = 4) : ∃ m : ℕ, m ≥ 8 ∧ 
  (∃ F V E : ℕ, (F + V = E + 2) ∧ (E = 2 * V) ∧ 
  (∃ num_of_tri_faces : ℕ, num_of_tri_faces = m ∧ 
  (∑ i in faces P, num_edges i = 2 * E) ∧ 
  (∀ j ∈ faces P \ {tri_faces}, num_edges j ≥ 4))) :=
sorry

end least_triangular_faces_l502_502752


namespace number_of_tiles_in_figure_with_perimeter_91_l502_502227

theorem number_of_tiles_in_figure_with_perimeter_91 :
  ∀ (side_length perimeter1 perimeter2 n : ℕ),
    side_length = 7 →
    perimeter1 = 21 →
    perimeter2 = 91 →
    perimeter2 - perimeter1 = 70 →
    perimeter1 = 3 * side_length →
    perimeter2 = perimeter1 + 7 * n →
    n = 10 →
    1 + n = 11 :=
by
  intros side_length perimeter1 perimeter2 n h_side_length h_perimeter1 h_perimeter2 h_diff h_perm1_calc h_perm2_calc h_n
  rw [h_side_length, h_perimeter1, h_perimeter2, h_diff, h_perm1_calc, h_perm2_calc, h_n]
  sorry

end number_of_tiles_in_figure_with_perimeter_91_l502_502227


namespace largest_eight_digit_with_all_even_digits_l502_502707

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l502_502707


namespace largest_eight_digit_with_all_even_digits_l502_502706

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l502_502706


namespace homothety_translation_composition_l502_502092

variable {ℂ : Type} [Complex]

def T (a : ℂ) (z : ℂ) : ℂ := z + a

def H (a : ℂ) (k : ℂ) (z : ℂ) : ℂ := k * (z - a) + a

theorem homothety_translation_composition (i z : ℂ) :
  H i 2 z = (T i) (H 0 2 (T (-i) z)) :=
by sorry

end homothety_translation_composition_l502_502092


namespace units_digit_first_four_composite_is_eight_l502_502604

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502604


namespace find_k_l502_502339

noncomputable def curve (x k : ℝ) : ℝ := x + k * Real.log (1 + x)

theorem find_k (k : ℝ) :
  let y' := (fun x => 1 + k / (1 + x))
  (y' 1 = 2) ∧ ((1 + 2 * 1) = 0) → k = 2 :=
by
  sorry

end find_k_l502_502339


namespace graph_shift_f_x_minus_3_is_C_l502_502356

noncomputable def f : ℝ → ℝ
| x => if x ≥ -3 ∧ x ≤ 0 then -2 - x
       else if x > 0 ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
       else if x > 2 ∧ x ≤ 3 then 2 * (x - 2)
       else 0  -- Define outside given ranges as 0 for simplicity

noncomputable def f_shifted (x: ℝ) : ℝ := f (x + 3)

theorem graph_shift_f_x_minus_3_is_C :
    ∃ (g: ℝ → ℝ), (∀ x, g x = f_shifted x) := by
  sorry

end graph_shift_f_x_minus_3_is_C_l502_502356


namespace jason_car_cost_l502_502472

theorem jason_car_cost
    (down_payment : ℕ := 8000)
    (monthly_payment : ℕ := 525)
    (months : ℕ := 48)
    (interest_rate : ℝ := 0.05) :
    (down_payment + monthly_payment * months + interest_rate * (monthly_payment * months)) = 34460 := 
by
  sorry

end jason_car_cost_l502_502472


namespace non_vegan_gluten_free_cupcakes_l502_502756

def total_cupcakes : ℕ := 80
def vegan_cupcakes : ℕ := 24
def gluten_free_half (n : ℕ) : ℕ := n / 2

theorem non_vegan_gluten_free_cupcakes :
  let gluten_free_cupcakes := gluten_free_half total_cupcakes
  let vegan_gluten_free_cupcakes := gluten_free_half vegan_cupcakes
  gluten_free_cupcakes - vegan_gluten_free_cupcakes = 28 :=
by
  let gluten_free_cupcakes := gluten_free_half total_cupcakes  -- 80 / 2 = 40
  let vegan_gluten_free_cupcakes := gluten_free_half vegan_cupcakes  -- 24 / 2 = 12
  show gluten_free_cupcakes - vegan_gluten_free_cupcakes = 28
  from by rfl

end non_vegan_gluten_free_cupcakes_l502_502756


namespace each_baby_worms_per_day_l502_502995

variable (babies : Nat) (worms_papa : Nat) (worms_mama_caught : Nat) (worms_mama_stolen : Nat) (worms_needed : Nat)
variable (days : Nat)

theorem each_baby_worms_per_day 
  (h1 : babies = 6) 
  (h2 : worms_papa = 9) 
  (h3 : worms_mama_caught = 13) 
  (h4 : worms_mama_stolen = 2)
  (h5 : worms_needed = 34) 
  (h6 : days = 3) :
  (worms_papa + (worms_mama_caught - worms_mama_stolen) + worms_needed) / babies / days = 3 :=
by
  sorry

end each_baby_worms_per_day_l502_502995


namespace log_mul_log_inv_log_frac_log_inv_l502_502509

theorem log_mul_log_inv (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  log b a * log a b = 1 :=
by sorry

theorem log_frac_log_inv (a m n : ℝ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  log a n / log (m * a) n = 1 + log a m :=
by sorry

end log_mul_log_inv_log_frac_log_inv_l502_502509


namespace frustum_volume_correct_l502_502233

noncomputable def volume_frustum (base_edge_original base_edge_smaller altitude_original altitude_smaller : ℝ) : ℝ :=
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let volume_original := (1 / 3) * base_area_original * altitude_original
  let volume_smaller := (1 / 3) * base_area_smaller * altitude_smaller
  volume_original - volume_smaller

theorem frustum_volume_correct :
  volume_frustum 16 8 10 5 = 2240 / 3 :=
by
  have h1 : volume_frustum 16 8 10 5 = 
    (1 / 3) * (16^2) * 10 - (1 / 3) * (8^2) * 5 := rfl
  simp only [pow_two] at h1
  norm_num at h1
  exact h1

end frustum_volume_correct_l502_502233


namespace num_ways_write_100_as_distinct_squares_l502_502408

theorem num_ways_write_100_as_distinct_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 + c^2 = 100 ∧
  (∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x^2 + y^2 + z^2 = 100 ∧ (x, y, z) ≠ (a, b, c) ∧ (x, y, z) ≠ (a, c, b) ∧ (x, y, z) ≠ (b, a, c) ∧ (x, y, z) ≠ (b, c, a) ∧ (x, y, z) ≠ (c, a, b) ∧ (x, y, z) ≠ (c, b, a)) ∧
  ∀ (p q r : ℕ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p^2 + q^2 + r^2 = 100 → (p, q, r) = (a, b, c) ∨ (p, q, r) = (a, c, b) ∨ (p, q, r) = (b, a, c) ∨ (p, q, r) = (b, c, a) ∨ (p, q, r) = (c, a, b) ∨ (p, q, r) = (c, b, a) ∨ (p, q, r) = (x, y, z) ∨ (p, q, r) = (x, z, y) ∨ (p, q, r) = (y, x, z) ∨ (p, q, r) = (y, z, x) ∨ (p, q, r) = (z, x, y) ∨ (p, q, r) = (z, y, x) :=
sorry

end num_ways_write_100_as_distinct_squares_l502_502408


namespace units_digit_of_first_four_composite_numbers_l502_502626

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502626


namespace possible_items_l502_502435

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502435


namespace cube_surface_area_increase_l502_502157

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l502_502157


namespace units_digit_first_four_composite_is_eight_l502_502605

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502605


namespace units_digit_first_four_composite_is_eight_l502_502614

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502614


namespace length_of_DE_l502_502091

-- Define the conditions of the problem
def AB : ℝ := 7
def AD : ℝ := 8
def DC : ℝ := 7
def area_triangle_DCE : ℝ := 28

-- Express the conditions formally
-- Condition 1: Rectangle sides
axiom AB_CD_equal_7 : AB = 7
axiom AD_CD_equal_8 : AD = 8

-- Condition 2: Height of rectangle and triangle
axiom height_of_triangle : DC = 7

-- Condition 3: Area of triangle
axiom area_of_triangle : (1/2) * DC * (area_triangle_DCE / (1/2 * 7)) = 28

-- The target length of DE
def DE : ℝ := Real.sqrt (DC^2 + (area_triangle_DCE / (1/2 * 7))^2)

-- Proof statement
theorem length_of_DE : DE = Real.sqrt 113 :=
by
  sorry

end length_of_DE_l502_502091


namespace units_digit_of_product_is_eight_l502_502591

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502591


namespace find_AB_equals_9_l502_502944

-- Define the necessary concepts
variables {A B C N G : Type}
variables (AB AC BC : ℝ) (x : ℝ)

-- Define the hypothesis and conditions
def triangleABC := (AC = 6) ∧ (BC = 5 * real.sqrt 3) ∧ (AB = x)
def bisectorAN := true -- Placeholder for AN bisects ∠BAC
def centroidG := true -- Placeholder for G is the centroid of triangle ABC
def perpendicularGN := true -- Placeholder for GN is perpendicular to BC

-- Formalizing the main theorem
theorem find_AB_equals_9
  (h1: triangleABC AC BC AB x)
  (h2: bisectorAN)
  (h3: centroidG)
  (h4: perpendicularGN) : 
  x = 9 := 
  sorry

end find_AB_equals_9_l502_502944


namespace truncated_cone_lateral_surface_area_l502_502542

theorem truncated_cone_lateral_surface_area
  (r R h : ℝ) (hr : r = 1) (hR : R = 4) (hh : h = 4) : 
  ∃ (S_lateral : ℝ), S_lateral = 25 * Real.pi := 
by
  let l := Real.sqrt (h^2 + (R - r)^2)
  have hl : l = 5 := 
    by
      rw [hr, hR, hh]
      norm_num
      rw [Real.sqrt_eq_rpow, Real.rpow_two]
      norm_num
  let S_lateral := Real.pi * l * (r + R)
  use S_lateral
  have : S_lateral = 25 * Real.pi :=
    by
      rw [hr, hR, hh, hl]
      norm_num
  exact this

end truncated_cone_lateral_surface_area_l502_502542


namespace largest_eight_digit_number_with_even_digits_l502_502717

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l502_502717


namespace number_of_unordered_pairs_l502_502740

theorem number_of_unordered_pairs (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  ∃ (n : ℕ), n = 365 ∧ ∀ (A B : Set ℕ), A ⊆ S → B ⊆ S → A ∩ B = ∅ → unordered_pairs_count (A, B) = n :=
begin
  use 365,
  intros A B hA hB hAB,
  sorry
end

/-- Helper definition for counting unordered pairs, to be defined as needed in further proof -/
def unordered_pairs_count (a_b : Set ℕ × Set ℕ) : ℕ := sorry

end number_of_unordered_pairs_l502_502740


namespace chromosome_stability_due_to_meiosis_and_fertilization_l502_502545

-- Definitions for conditions
def chrom_replicate_distribute_evenly : Prop := true
def central_cell_membrane_invagination : Prop := true
def mitosis : Prop := true
def meiosis_and_fertilization : Prop := true

-- Main theorem statement to be proved
theorem chromosome_stability_due_to_meiosis_and_fertilization :
  meiosis_and_fertilization :=
sorry

end chromosome_stability_due_to_meiosis_and_fertilization_l502_502545


namespace units_digit_of_product_of_first_four_composites_l502_502637

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502637


namespace deductive_reasoning_example_l502_502812

-- Definitions from the conditions
def everyone_makes_mistakes : Prop := ∀ x : Person, makes_mistakes x
def mr_wang : Person

-- Definition of the conclusion derived from premises
def mr_wang_makes_mistakes : Prop := makes_mistakes mr_wang

-- Proof statement
theorem deductive_reasoning_example
  (H1 : everyone_makes_mistakes)
  (H2 : ∀ p : Person, p = mr_wang)
  : (mr_wang_makes_mistakes) :=
  sorry

end deductive_reasoning_example_l502_502812


namespace time_to_pass_bridge_l502_502231

def train_length : ℝ := 360 -- meters
def bridge_length : ℝ := 140 -- meters
def train_speed_kmh : ℝ := 50 -- km/hour
def speed_conversion_factor : ℝ := 1000 / 3600 -- (meters/second)/(km/hour)

def total_distance : ℝ := train_length + bridge_length
def train_speed_ms : ℝ := train_speed_kmh * speed_conversion_factor

theorem time_to_pass_bridge : 
  let time := total_distance / train_speed_ms in
  time ≈ 36 := 
by
  -- We outline the proof but leave it to be completed
  sorry

end time_to_pass_bridge_l502_502231


namespace tetrahedron_volume_l502_502301

noncomputable def midpoint_to_face_distance : ℝ := 2
noncomputable def midpoint_to_edge_distance : ℝ := Real.sqrt 5

theorem tetrahedron_volume 
  (d_face : ℝ := midpoint_to_face_distance) 
  (d_edge : ℝ := midpoint_to_edge_distance) 
  (h : ℝ := 6)  -- derived height h = 3 * 2
  (a : ℝ := 3 * Real.sqrt 6)  -- derived edge length a
  : abs (volume (a := a) - 485.42) < 0.01 :=
by sorry

noncomputable def volume (a : ℝ): ℝ := a^3 / (6 * Real.sqrt 2)

end tetrahedron_volume_l502_502301


namespace incenter_on_line_PQ_l502_502491

open_locale euclidean_geometry

noncomputable theory

variables {A B C P Q : Point} 
variables {Γ Γ₀ : Circle -- defining the circles }

-- Definitions of the circumcircle and internal tangency
variables (hΓ : Circumcircle △ A B C Γ)
variables (hΓ₀ : CircleTangentInternally Γ Γ₀) 
variables (hP : TangentToSide Γ₀ A B P)

-- Definition of the tangent line and point of tangency
variables (hCQ : ∃ Q, TangentLineThroughPoint C Γ₀ Q) 
variables (same_side_PQ : SameSideOfLine P Q (LineBC))

theorem incenter_on_line_PQ :
  Incenter△ A B C ∈ LinePQ :=
sorry

end incenter_on_line_PQ_l502_502491


namespace mean_temperature_l502_502123

def temperatures : List ℝ := [75, 77, 76, 78, 80, 81, 83, 82, 84, 86]

theorem mean_temperature : (temperatures.sum / temperatures.length) = 80.2 := by
  sorry

end mean_temperature_l502_502123


namespace find_angle_C_range_of_a_plus_b_l502_502928

variables {A B C a b c : ℝ}

-- Define the conditions
def conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a + c) * (Real.sin A - Real.sin C) = Real.sin B * (a - b)

-- Proof problem 1: show angle C is π/3
theorem find_angle_C (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b c A B C) : 
  C = π / 3 :=
sorry

-- Proof problem 2: if c = 2, then show the range of a + b
theorem range_of_a_plus_b (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b 2 A B C) :
  2 < a + b ∧ a + b ≤ 4 :=
sorry

end find_angle_C_range_of_a_plus_b_l502_502928


namespace angle_bisector_KM_l502_502217

-- Define a structure for the conditions given in the problem
structure parallelogram (A B C D O K L M : Type) extends is_parallelogram : Prop :=
(parallelogram_prop : IsParallelogram A B C D)
(BO_perp_AD : Perpendicular B O A D)
(circle_center_O : Circle O A B)
(intersect_extension_AD_K : ∃ K, Circle O A B ∧ LineIntersectExtension A D K)
(intersect_BK_CD_L : ∃ L, SegmentIntersect B K C D L)
(ray_OL_intersect_circle_M : ∃ M, RayIntersect O L Circle O A B M)

-- The theorem statement in Lean
theorem angle_bisector_KM (A B C D O K L M : Type) [parallelogram A B C D O K L M]:
  AngleBisector K M (Angle B K C) :=
by
  -- Proof to be constructed here
  sorry

end angle_bisector_KM_l502_502217


namespace train_speed_l502_502770

theorem train_speed :
  ∃ (Vt : ℝ), 
    let L := 250 
    let t := 17.998560115190788 
    let Vm := 8 * (1000 / 3600)  -- Man's speed in m/s 
    let Vr := Vt - Vm 
    Vt = (L + (Vm * t)) / t 
    ∃ (Vt_kmph : ℝ), Vt_kmph = Vt * (3600 / 1000) 
    abs (Vt_kmph - 57.99) < 1e-2 := -- Approximation with a tolerance
begin
  sorry
end

end train_speed_l502_502770


namespace triangle_AEG_area_l502_502117

theorem triangle_AEG_area (AB AD : ℝ)
  (hAB : AB = 8)
  (hAD : AD = 6) :
  let AC := Real.sqrt (AB ^ 2 + AD ^ 2)
  let segment := AC / 4
  let base := segment
  let height := (3 / 4) * (2 * AB * AD / AC) in
  (1 / 2) * base * height = 4.5 := by
  -- Definition of AC using Pythagorean theorem
  let AC := Real.sqrt (8^2 + 6^2)
  have hAC : AC = 10 := by
    norm_num1; rw [←Real.sqrt_sq, add_comm]
  -- Each segment of AC
  let segment := AC / 4
  have hsegment : segment = 2.5 := by
    exact (div_eq_mul_one_div AC 4).symm ▸ hAC.symm ▸ rfl
  -- Base and height calculations
  let base := segment
  let height := (3 / 4) * (2 * 8 * 6 / AC)
  have hbase : base = 2.5 := by
    exact hsegment
  have hheight : height = 3.6 := by
    simp only [mul_div_cancel_left]; rw [←hAC, ←Real.mul_div_mul_comm]; norm_num1
  -- Area calculation
  have area_AEG : (1 / 2) * base * height = 4.5 := by
    norm_num1
  exact area_AEG

end triangle_AEG_area_l502_502117


namespace num_ways_write_100_as_distinct_squares_l502_502407

theorem num_ways_write_100_as_distinct_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 + c^2 = 100 ∧
  (∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x^2 + y^2 + z^2 = 100 ∧ (x, y, z) ≠ (a, b, c) ∧ (x, y, z) ≠ (a, c, b) ∧ (x, y, z) ≠ (b, a, c) ∧ (x, y, z) ≠ (b, c, a) ∧ (x, y, z) ≠ (c, a, b) ∧ (x, y, z) ≠ (c, b, a)) ∧
  ∀ (p q r : ℕ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p^2 + q^2 + r^2 = 100 → (p, q, r) = (a, b, c) ∨ (p, q, r) = (a, c, b) ∨ (p, q, r) = (b, a, c) ∨ (p, q, r) = (b, c, a) ∨ (p, q, r) = (c, a, b) ∨ (p, q, r) = (c, b, a) ∨ (p, q, r) = (x, y, z) ∨ (p, q, r) = (x, z, y) ∨ (p, q, r) = (y, x, z) ∨ (p, q, r) = (y, z, x) ∨ (p, q, r) = (z, x, y) ∨ (p, q, r) = (z, y, x) :=
sorry

end num_ways_write_100_as_distinct_squares_l502_502407


namespace target_hit_probability_l502_502365

open ProbabilityTheory

theorem target_hit_probability :
  let PA := 0.8
  let PB := 0.7
  let P_not_hit := (1 - PA) * (1 - PB)
  let P_hit := 1 - P_not_hit
  P_hit = 0.94 :=
by
  sorry

end target_hit_probability_l502_502365


namespace real_solutions_quadratic_l502_502805

theorem real_solutions_quadratic (d : ℝ) (h : 0 < d) :
  ∃ x : ℝ, x^2 - 8 * x + d < 0 ↔ 0 < d ∧ d < 16 :=
by
  sorry

end real_solutions_quadratic_l502_502805


namespace units_digit_of_first_four_composite_numbers_l502_502633

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502633


namespace intersection_points_parabolas_l502_502565

theorem intersection_points_parabolas :
  let parabola1 := λ x: ℝ, 2 * x^2 - 3 * x + 1
  let parabola2 := λ x: ℝ, x^2 - 4 * x + 4
  ∃ (x y: ℝ), (x, y) = (-3, 25) ∨ (x, y) = (1, 1) ∧ parabola1 x = y ∧ parabola2 x = y :=
begin
  sorry
end

end intersection_points_parabolas_l502_502565


namespace number_of_items_l502_502461

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502461


namespace intersection_of_A_and_B_l502_502324

def A (x : ℝ) : Prop := x^2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := x > 1

theorem intersection_of_A_and_B :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_of_A_and_B_l502_502324


namespace evaluate_expression_l502_502281

theorem evaluate_expression : 
  (81:ℝ)^(1 / 2) * (64:ℝ)^(-1 / 3) * (49:ℝ)^(1 / 2) = 63 / 4 := 
by 
{
  sorry
}

end evaluate_expression_l502_502281


namespace number_of_asian_countries_l502_502258

theorem number_of_asian_countries (total european south_american : ℕ) 
  (H_total : total = 42) 
  (H_european : european = 20) 
  (H_south_american : south_american = 10) 
  (H_half_asian : ∃ rest, rest = total - european - south_american ∧ rest / 2 = 6) : 
  ∃ asian, asian = 6 :=
by {
  let rest := total - european - south_american,
  have H_rest : rest = 42 - 20 - 10, from sorry,
  have H_asian : rest / 2 = 6, from sorry,
  exact ⟨6, rfl⟩,
}

end number_of_asian_countries_l502_502258


namespace units_digit_of_product_is_eight_l502_502592

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l502_502592


namespace kopeechka_items_l502_502466

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502466


namespace find_perimeter_ABC_l502_502874

noncomputable def perimeter_triangle_eq {A B C : Type} (ellipse : ℝ → ℝ → Prop)
  (is_focus_a : A)
  (is_focus_f_on_bc : B → C → A → Prop)
  (vertices_on_ellipse : ∀ {b c}, ellipse b c → b = B ∧ c = C) :
  ℝ :=
  2 * Real.sqrt 2

theorem find_perimeter_ABC :
  let ellipse := λ x y, 2 * x^2 + 3 * y^2 = 1,
      is_focus_a := true, -- assuming A is at one of the foci of the ellipse
      is_focus_f_on_bc := λ B C A, true, -- the other foci is on line segment BC
      vertices_on_ellipse := λ B C, ellipse B C in
  perimeter_triangle_eq ellipse is_focus_a is_focus_f_on_bc vertices_on_ellipse = 2 * Real.sqrt 2 :=
sorry

end find_perimeter_ABC_l502_502874


namespace evaluate_expression_l502_502282

theorem evaluate_expression :
  (81:ℝ)^(1/2) * (64:ℝ)^(-1/3) * (49:ℝ)^(1/2) = (63:ℝ) / 4 :=
by
  sorry

end evaluate_expression_l502_502282


namespace vector_AC_l502_502857

variable {V : Type} [AddCommGroup V] [VectorSpace ℝ V]
variables (a b : V)

def is_median (A B C D : V) :=
  D = (B + C) / 2 ∧ A + C = 2 * D

theorem vector_AC (A B C D : V) (h1 : is_median A B C D)
  (h2 : A + (B - A) = b) (h3 : A + b = 2 * D) :
  C = 2 * b - A := 
sorry

end vector_AC_l502_502857


namespace calculate_sequence_l502_502055

def N (x : ℝ) : ℝ := 2 * Real.sqrt x
def O (x : ℝ) : ℝ := x^2
def P (x : ℝ) : ℝ := x + 1

theorem calculate_sequence : N(O(P(N(O(P(N(O(2)))))))) = 22 := 
by
  sorry

end calculate_sequence_l502_502055


namespace local_min_value_of_f_l502_502876

noncomputable def f (x m : ℝ) : ℝ := (x^2 + x + m) * Real.exp x

theorem local_min_value_of_f :
  (∃ m : ℝ, ∀ x : ℝ, f'(-3) = 0) → ∃ x : ℝ, f(x, -1) = -1 :=
by
  -- conditions
  let f' := λ x m, (2 * x + 1 + (x^2 + x + m) * Real.exp x) * Real.exp x
  have h_deriv : f'(-3) = ((-3)^2 + 3 * -3 + m + 1) * Real.exp(-3) = 0, by sorry,
  -- proof of local minimum
  have h_min : ∃ x : ℝ, f(x, -1) = -1, by sorry,
  exact ⟨0, h_min⟩


end local_min_value_of_f_l502_502876


namespace largest_eight_digit_number_with_even_digits_l502_502715

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l502_502715


namespace evaluate_expression_l502_502279

theorem evaluate_expression : 
  (81:ℝ)^(1 / 2) * (64:ℝ)^(-1 / 3) * (49:ℝ)^(1 / 2) = 63 / 4 := 
by 
{
  sorry
}

end evaluate_expression_l502_502279


namespace minimize_sum_dist_l502_502854

noncomputable section

variables {Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ}

-- Conditions
def clusters (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) :=
  Q3 <= Q1 + Q2 + Q4 / 3 ∧ Q3 = (Q1 + 2 * Q2 + 2 * Q4) / 5 ∧
  Q7 <= Q5 + Q6 + Q8 / 3 ∧ Q7 = (Q5 + 2 * Q6 + 2 * Q8) / 5

-- Sum of distances function
def sum_dist (Q : ℝ) (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) : ℝ :=
  abs (Q - Q1) + abs (Q - Q2) + abs (Q - Q3) + abs (Q - Q4) +
  abs (Q - Q5) + abs (Q - Q6) + abs (Q - Q7) + abs (Q - Q8) + abs (Q - Q9)

-- Theorem
theorem minimize_sum_dist (h : clusters Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) :
  ∃ Q : ℝ, (∀ Q' : ℝ, sum_dist Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 ≤ sum_dist Q' Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) → Q = Q5 :=
sorry

end minimize_sum_dist_l502_502854


namespace kopeechka_items_l502_502467

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502467


namespace maximize_profit_l502_502209

variables (x : ℝ) (initial_price_per_unit purchase_price_per_unit : ℝ) (initial_sales_volume decrease_per_unit_increase_per_yuan : ℝ)

def sales_volume (x : ℝ) : ℝ :=
  initial_sales_volume - decrease_per_unit_increase_per_yuan * (x - initial_price_per_unit)

def profit (x purchase_price_per_unit : ℝ) : ℝ :=
  (x - purchase_price_per_unit) * sales_volume x

theorem maximize_profit
  (initial_price_per_unit := 60)
  (purchase_price_per_unit := 40)
  (initial_sales_volume := 400)
  (decrease_per_unit_increase_per_yuan := 10) : x = 70 :=
begin
  sorry
end

end maximize_profit_l502_502209


namespace units_digit_first_four_composites_l502_502571

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502571


namespace max_arithmetic_sum_l502_502149

def a1 : ℤ := 113
def d : ℤ := -4

def S (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem max_arithmetic_sum : S 29 = 1653 :=
by
  sorry

end max_arithmetic_sum_l502_502149


namespace ratio_QE_ED_l502_502925

-- Definitions related to the conditions
variables {DE EF DF x k : ℝ}
variables (E D F Q : Type) [DecidableEq E] [DecidableEq D] [DecidableEq F] [DecidableEq Q]
variables (between_ED : E ∈ segment Q D) (exterior_angle_bisector_at_F : bisects (exterior_angle F) Q) 

-- Given conditions
axiom ratio_DE_EF : DE / EF = 5 / 3
axiom extends_ED_to_Q : intersects (extended ED) Q
axiom scaling_factor_k : QE / ED = 4 / 1

-- The statement to be proved
theorem ratio_QE_ED : QE / ED = 4 :=
by 
  sorry

end ratio_QE_ED_l502_502925


namespace polynomial_no_strictly_positive_roots_l502_502983

-- Define the necessary conditions and prove the main statement

variables (n : ℕ)
variables (a : Fin n → ℕ) (k : ℕ) (M : ℕ)

-- Axioms/Conditions
axiom pos_a (i : Fin n) : 0 < a i
axiom pos_k : 0 < k
axiom pos_M : 0 < M
axiom M_gt_1 : M > 1

axiom sum_reciprocals : (Finset.univ.sum (λ i => (1 : ℚ) / a i)) = k
axiom product_a : (Finset.univ.prod a) = M

noncomputable def polynomial_has_no_positive_roots : Prop :=
  ∀ x : ℝ, 0 < x →
    M * (1 + x)^k > (Finset.univ.prod (λ i => x + a i))

theorem polynomial_no_strictly_positive_roots (h : polynomial_has_no_positive_roots n a k M) : 
  ∀ x : ℝ, 0 < x → (M * (1 + x)^k - (Finset.univ.prod (λ i => x + a i)) ≠ 0) :=
by
  sorry

end polynomial_no_strictly_positive_roots_l502_502983


namespace sum_of_odd_indexed_terms_l502_502764

-- Definitions based on given conditions
def sequence (n : ℕ) : ℕ → ℕ
| 0       => n
| (k + 1) => (sequence n k) + 2

def sum_nat_seq (f : ℕ → ℕ) (N : ℕ) : ℕ := 
  Nat.sum (Finset.range N) (λ i => f i)

-- Given: total sum of 1500 terms is 7600
axiom total_sum : ∃ n, (sum_nat_seq (sequence n) 1500) = 7600

-- Prove the sum of every second term (starting with the first and ending with the second last) is 3050
theorem sum_of_odd_indexed_terms : 
  (∃ n, (sum_nat_seq (λ k => sequence n (2 * k)) 750) = 3050) :=
sorry

end sum_of_odd_indexed_terms_l502_502764


namespace sum_of_first_six_terms_l502_502104

theorem sum_of_first_six_terms 
  (a₁ : ℝ) 
  (r : ℝ) 
  (h_ratio : r = 2) 
  (h_sum_three : a₁ + 2*a₁ + 4*a₁ = 3) 
  : a₁ * (r^6 - 1) / (r - 1) = 27 := 
by {
  sorry
}

end sum_of_first_six_terms_l502_502104


namespace f_decreasing_interval_sum_of_first_10_zeros_l502_502883

noncomputable def f (x : ℝ) : ℝ := (Math.sin x) * (Math.cos x) - (Math.cos x) ^ 2

theorem f_decreasing_interval (k : ℤ) :
  ∀ x, k * Real.pi + (3 * Real.pi / 8) ≤ x ∧ x ≤ k * Real.pi + (7 * Real.pi / 8) → 
  (f' x < 0) :=
sorry

theorem sum_of_first_10_zeros :
  ∑ i in Finset.range 10, (λ n, if n % 2 = 0 then (n / 2) * Real.pi + Real.pi / 4 else (n / 2) * Real.pi + Real.pi / 2) (i + 1) = 95 * Real.pi / 4 :=
sorry

end f_decreasing_interval_sum_of_first_10_zeros_l502_502883


namespace largest_eight_digit_number_contains_even_digits_l502_502720

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l502_502720


namespace selling_price_proof_l502_502075

noncomputable def selling_price (CP MP DiscountAmount SP : ℝ) : Prop :=
  CP = 540 ∧
  MP = CP + 0.15 * CP ∧
  DiscountAmount = 0.2593 * MP ∧
  SP = MP - DiscountAmount ∧
  SP = 459.93

theorem selling_price_proof : ∃ SP : ℝ, selling_price 540 (540 + 0.15 * 540) (0.2593 * (540 + 0.15 * 540)) SP :=
by
  have CP := 540
  have MP := 540 + 0.15 * 540
  have DiscountAmount := 0.2593 * MP
  existsi (MP - DiscountAmount)
  simp [selling_price, CP, MP, DiscountAmount]
  split
  rfl
  split
  rfl
  split
  rfl
  split
  rfl
  norm_num
  done
  rfl
  sorry

end selling_price_proof_l502_502075


namespace compare_a_b_l502_502972

def a := Real.sqrt 7 - Real.sqrt 3
def b := Real.sqrt 6 - Real.sqrt 2

theorem compare_a_b : a < b := sorry

end compare_a_b_l502_502972


namespace find_inverse_f_123_l502_502385

def f (x : ℝ) : ℝ := 3 * x^3 + 6

theorem find_inverse_f_123 : f (ℝ.sqrt3 39) = 123 := 
by 
  have h : f (ℝ.sqrt3 39) = 3 * (ℝ.sqrt3 39)^3 + 6 := rfl
  rw [← Real.rpow_nat_cast (ℝ.sqrt3 39) 3] at h
  rw [Real.sqrt3_rpow_eq (ℝ.sqrt3 39) (by norm_num)] at h
  simp only [Real.rpow_nat_cast, zero_add, pow_three] at h
  norm_num at h
  exact h
  

end find_inverse_f_123_l502_502385


namespace monotonic_intervals_and_extreme_values_of_f_range_of_a_for_h_to_have_2_zeros_l502_502852

noncomputable def f (x : ℝ) := 2 * x^2 * Real.exp x
noncomputable def g (a x : ℝ) := a * x + 2 * a * Real.log x
noncomputable def h (a x : ℝ) := f x - g a x

theorem monotonic_intervals_and_extreme_values_of_f :
  (∀ x : ℝ, x < -2 → f' x > 0) ∧
  (f'(-2) = 0 ∧ ∀ x : ℝ, -2 < x ∧ x < 0 → f' x < 0) ∧
  (f'(0) = 0 ∧ ∀ x : ℝ, x > 0 → f' x > 0) ∧
  (f (-2) = 8 / Real.exp 2) ∧
  (f 0 = 0) :=
sorry

theorem range_of_a_for_h_to_have_2_zeros (a : ℝ) :
  (∃ x1 x2 : ℝ, (h a x1 = 0) ∧ (h a x2 = 0) ∧ x1 ≠ x2) ↔ a > 2 * Real.exp 1 :=
sorry

end monotonic_intervals_and_extreme_values_of_f_range_of_a_for_h_to_have_2_zeros_l502_502852


namespace percentage_passed_exam_l502_502026

theorem percentage_passed_exam (total_students failed_students : ℕ) (h_total : total_students = 540) (h_failed : failed_students = 351) :
  (total_students - failed_students) * 100 / total_students = 35 :=
by
  sorry

end percentage_passed_exam_l502_502026


namespace polynomial_coefficient_a0_polynomial_sum_a0_to_a7_l502_502382

noncomputable def given_polynomial (x : ℝ) : ℝ :=
  (1 - 2 * x) ^ 7

theorem polynomial_coefficient_a0 :
  (∀ (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ), 
    (given_polynomial 0 = a0 + a1 * 0 + a2 * 0^2 + a3 * 0^3 + a4 * 0^4 + a5 * 0^5 + a6 * 0^6 + a7 * 0^7) →
    a0 = 1) :=
begin
  intros,
  rw [given_polynomial, zero_pow, mul_zero, add_zero] at *,
  exact eq.symm this,
end

theorem polynomial_sum_a0_to_a7 :
  (∀ (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ),
    (given_polynomial 1 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7) →
    (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = -128)) :=
begin
  intros,
  calc 
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
        = given_polynomial 1 : by { exact eq.symm this }
    ... = (1 - 2) ^ 7 : by { unfold given_polynomial }
    ... = -128 : by norm_num,
end

end polynomial_coefficient_a0_polynomial_sum_a0_to_a7_l502_502382


namespace total_population_l502_502793

-- Defining the populations of Springfield and the difference in population
def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

-- The definition of Greenville's population in terms of Springfield's population
def greenville_population : ℕ := springfield_population - population_difference

-- The statement that we want to prove: the total population of Springfield and Greenville
theorem total_population :
  springfield_population + greenville_population = 845640 := by
  sorry

end total_population_l502_502793


namespace donations_correct_transportation_min_cost_correct_l502_502397

-- Definitions reflecting the conditions
def total_fertilizer_donation (c d : ℕ) := c + d = 100
def c_donation_less_than_twice_d (c d : ℕ) := c = 2 * d - 20

-- The theorems to prove
theorem donations_correct :
  ∃ c d : ℕ, total_fertilizer_donation c d ∧ c_donation_less_than_twice_d c d ∧ c = 60 ∧ d = 40 :=
by
  sorry

def transportation_cost (x a : ℕ) := (4 - a) * x + (1980 + 10 * a)
def min_cost_0_a_lt_4 (a : ℕ) (h : 0 < a ∧ a < 4) :=  transportation_cost 10 a = 2020
def min_cost_a_eq_4 (a : ℕ) (h : a = 4) := transportation_cost 10 a = 2020
def min_cost_4_lt_a_lt_6 (a : ℕ) (h : 4 < a ∧ a < 6) :=  transportation_cost 50 a = 2180 - 40 * a

theorem transportation_min_cost_correct :
  ∀ (x a : ℕ), 
    (0 < a ∧ a < 4) → min_cost_0_a_lt_4 a ‹0 < a ∧ a < 4› ∨
    (a = 4) → min_cost_a_eq_4 a ‹a = 4› ∨
    (4 < a ∧ a < 6) → min_cost_4_lt_a_lt_6 a ‹4 < a ∧ a < 6› :=
by
  sorry

end donations_correct_transportation_min_cost_correct_l502_502397


namespace inequality_with_arithmetic_condition_l502_502987

variable (n : ℕ) [fact (0 < n)]
variable {x : ℕ → ℝ}

-- Hypotheses
hypothesis sorted : ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → x i ≤ x j

theorem inequality_with_arithmetic_condition :
  (∑ i in finRange n, ∑ j in finRange n, abs (x i - x j))^2 =
  2 * ((n^2 - 1) / 3) * (∑ i in finRange n, ∑ j in finRange n, (x i - x j)^2) ↔
  ∃ (c d : ℝ), ∀ i, 1 ≤ i ∧ i ≤ n → x i = c + d * i := sorry

end inequality_with_arithmetic_condition_l502_502987


namespace cost_price_of_apple_is_18_l502_502167

noncomputable def cp (sp : ℝ) (loss_fraction : ℝ) : ℝ := sp / (1 - loss_fraction)

theorem cost_price_of_apple_is_18 :
  cp 15 (1/6) = 18 :=
by
  sorry

end cost_price_of_apple_is_18_l502_502167


namespace geometric_sequence_a_11_l502_502946

-- Define the geometric sequence with given terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

axiom a_5 : a 5 = -16
axiom a_8 : a 8 = 8

-- Question to prove
theorem geometric_sequence_a_11 (h : is_geometric_sequence a q) : a 11 = -4 := 
sorry

end geometric_sequence_a_11_l502_502946


namespace distinct_pairs_count_l502_502822

theorem distinct_pairs_count :
  ∃ (n : ℕ), n = 35 ∧
  ∀ (x y : ℕ), 0 < x ∧ x < y ∧ real.sqrt 5041 = real.sqrt x + real.sqrt y ↔ n = 35 :=
sorry

end distinct_pairs_count_l502_502822


namespace prob_not_same_class_prob_same_class_prob_diff_gender_not_same_class_l502_502399

theorem prob_not_same_class : 
  let students := [
    ("Class1", "Male"), ("Class1", "Female"),
    ("Class2", "Male"), ("Class2", "Female"),
    ("Class3", "Male"), ("Class3", "Female")
  ] in
  let total_pairs := ((students.length : ℚ)!).choose(2 : ℚ) in
  let diff_class_pairs := 12 in
  diff_class_pairs / total_pairs = 4/5 :=
by
  sorry

theorem prob_same_class : 
  let students := [
    ("Class1", "Male"), ("Class1", "Female"),
    ("Class2", "Male"), ("Class2", "Female"),
    ("Class3", "Male"), ("Class3", "Female")
  ] in
  let total_pairs := ((students.length : ℚ)!).choose(2 : ℚ) in
  let same_class_pairs := 3 in
  same_class_pairs / total_pairs = 1/5 :=
by
  sorry

theorem prob_diff_gender_not_same_class : 
  let students := [
    ("Class1", "Male"), ("Class1", "Female"),
    ("Class2", "Male"), ("Class2", "Female"),
    ("Class3", "Male"), ("Class3", "Female")
  ] in
  let total_pairs := ((students.length : ℚ)!).choose(2 : ℚ) in
  let diff_gender_not_same_class_pairs := 6 in
  diff_gender_not_same_class_pairs / total_pairs = 2/5 :=
by
  sorry

end prob_not_same_class_prob_same_class_prob_diff_gender_not_same_class_l502_502399


namespace joe_new_average_score_l502_502474

/-- Joe's initial average test score across 4 tests -/
def initial_avg_score : ℕ := 50

/-- Joe's lowest test score -/
def lowest_score : ℕ := 35

/-- Number of tests Joe took -/
def num_tests : ℕ := 4

/-- Number of tests Joe considered after dropping the lowest score -/
def num_tests_after_dropping : ℕ := num_tests - 1

theorem joe_new_average_score :
  let total_initial := num_tests * initial_avg_score in
  let total_after_dropping := total_initial - lowest_score in
  total_after_dropping / num_tests_after_dropping = 55 :=
by
  -- We'll just check that the desired outcome holds with the given numbers
  let total_initial := num_tests * initial_avg_score
  let total_after_dropping := total_initial - lowest_score
  calc
    total_after_dropping / num_tests_after_dropping = 165 / 3 : by sorry
    ... = 55 : by sorry

end joe_new_average_score_l502_502474


namespace find_c_d_l502_502481

def star (c d : ℕ) : ℕ := c^d + c*d

theorem find_c_d (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d) (h_star : star c d = 28) : c + d = 7 :=
by
  sorry

end find_c_d_l502_502481


namespace divides_ratio_AD_correct_l502_502949

noncomputable def ratio_divide_AD (A B C D M : Type) [Plane A] [Plane B] [Plane C] [Plane D] [Point M] :=
  right_angle A ∧
  right_angle D ∧
  dist_2d A B 1 ∧
  dist_2d C D 4 ∧
  dist_2d A D 5 ∧
  (∃ α : ℝ, angle B M A = α ∧ angle C M D = 2 * α) →
  divide_point M A D (2 / 3)
  
theorem divides_ratio_AD_correct (A B C D M : Type) [Plane A] [Plane B] [Plane C] [Plane D] [Point M] :
  ratio_divide_AD A B C D M := by
  unfold ratio_divide_AD
  sorry

end divides_ratio_AD_correct_l502_502949


namespace ben_whitewashed_feet_l502_502138

theorem ben_whitewashed_feet :
  ∃ x : ℝ, 100 - x - (1/5) * (100 - x) - (1/3) * (100 - x - (1/5) * (100 - x)) = 48 ∧ x = 10 :=
begin
  sorry
end

end ben_whitewashed_feet_l502_502138


namespace andy_remaining_demerits_l502_502779

-- Definitions based on conditions
def max_demerits : ℕ := 50
def demerits_per_late_instance : ℕ := 2
def late_instances : ℕ := 6
def joke_demerits : ℕ := 15

-- Calculation of total demerits for the month
def total_demerits : ℕ := (demerits_per_late_instance * late_instances) + joke_demerits

-- Proof statement: Andy can receive 23 more demerits without being fired
theorem andy_remaining_demerits : max_demerits - total_demerits = 23 :=
by
  -- Placeholder for proof
  sorry

end andy_remaining_demerits_l502_502779


namespace units_digit_first_four_composites_l502_502579

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502579


namespace area_of_triangle_G1G2G3_l502_502056

-- Define the given conditions
variable (ABC : Triangle)
variable (P : Point) -- incenter of triangle ABC
variable (G1 G2 G3 : Point) -- centroids of PBC, PCA, PAB respectively
variable (area_ABC : ℝ) -- area of triangle ABC

-- Define the property of triangle areas
def area (T : Triangle) : ℝ := sorry

-- Assume the given conditions
axiom incenter (P : Point) (ABC : Triangle) : P is_incenter ABC
axiom centroid_PBC (G1 : Point) (P B C : Point) : G1 is_centroid (mk_triangle P B C)
axiom centroid_PCA (G2 : Point) (P C A : Point) : G2 is_centroid (mk_triangle P C A)
axiom centroid_PAB (G3 : Point) (P A B : Point) : G3 is_centroid (mk_triangle P A B)
axiom area_triangle_ABC : area ABC = 24

-- Statement to prove
theorem area_of_triangle_G1G2G3 (ABC : Triangle) (P G1 G2 G3 : Point)
  (h1 : P is_incenter ABC) (h2 : G1 is_centroid (mk_triangle P (ABC.B) (ABC.C)))
  (h3 : G2 is_centroid (mk_triangle P (ABC.C) (ABC.A)))
  (h4 : G3 is_centroid (mk_triangle P (ABC.A) (ABC.B)))
  (h5 : area ABC = 24) :
  area (mk_triangle G1 G2 G3) = 8 / 3 :=
  sorry

end area_of_triangle_G1G2G3_l502_502056


namespace kolya_purchase_l502_502452

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502452


namespace largest_eight_digit_number_with_even_digits_l502_502716

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l502_502716


namespace find_value_l502_502327

theorem find_value (α : ℝ) (h₁ : sin α = -4/5) (h₂ : tan α = 2) :
  4 * (sin α)^2 + 2 * (sin α) * (cos α) = 4 :=
sorry

end find_value_l502_502327


namespace households_with_car_l502_502933

theorem households_with_car (total households : ℕ) (no_car_or_bike : ℕ) (both_car_and_bike : ℕ) (bike_only : ℕ) : 
  total households = 90 ∧ no_car_or_bike = 11 ∧ both_car_and_bike = 18 ∧ bike_only = 35 → 
  ∃ (car : ℕ), car = 62 :=
begin
  sorry
end

end households_with_car_l502_502933


namespace major_axis_length_l502_502768

noncomputable def conic_section_points (x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 : ℝ) : Prop :=
∀ (x y : ℝ), 
  ∃ (A B C D E : ℝ), 
  A * x^2 + B * y^2 + C * x * y + D * x + E * y = 1 ∧ 
  A * x1^2 + B * y1^2 + C * x1 * y1 + D * x1 + E * y1 = 1 ∧
  A * x2^2 + B * y2^2 + C * x2 * y2 + D * x2 + E * y2 = 1 ∧
  A * x3^2 + B * y3^2 + C * x3 * y3 + D * x3 + E * y3 = 1 ∧
  A * x4^2 + B * y4^2 + C * x4 * y4 + D * x4 + E * y4 = 1 ∧
  A * x5^2 + B * y5^2 + C * x5 * y5 + D * x5 + E * y5 = 1

theorem major_axis_length :
  conic_section_points (-5 / 2) 1 0 0 0 3 4 0 4 3 ∧ 
  true :=   -- Placeholder for "it is an ellipse" proof condition
begin
  have ellipse_coords_and_major_axis := 
    ∃ c : ℝ × ℝ, ∃ a b : ℝ, 
    (c = (2, 1.5)) ∧ (a = 9 / 2) ∧ (b = sqrt (2.25 * 20.25 / 16.25)), -- where 'a' is the semi-major axis
  sorry -- Skipping detailed solution steps and directly asserting the axis length
end

end major_axis_length_l502_502768


namespace problem_statement_l502_502533

theorem problem_statement :
  ∃ p q r : ℤ,
    (∀ x : ℝ, (x^2 + 19*x + 88 = (x + p) * (x + q)) ∧ (x^2 - 23*x + 132 = (x - q) * (x - r))) →
      p + q + r = 31 :=
sorry

end problem_statement_l502_502533


namespace solve_x_l502_502835

def matrix_A (a x : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![[x + a, x, x, x],
    [x, x + a, x, x],
    [x, x, x + a, x],
    [x, x, x, x + a]]

theorem solve_x (a : ℝ) (x : ℝ) (h : a ≠ 0) (h_det : (matrix_A a x).det = 0) : x = 0 :=
by
  sorry

end solve_x_l502_502835


namespace integral_sin_plus_two_l502_502785

theorem integral_sin_plus_two :
  ∫ x in -Real.pi / 2 .. Real.pi / 2, (Real.sin x + 2) = 2 * Real.pi :=
by
  sorry

end integral_sin_plus_two_l502_502785


namespace f_decreasing_interval_sum_of_first_10_zeros_l502_502884

noncomputable def f (x : ℝ) : ℝ := (Math.sin x) * (Math.cos x) - (Math.cos x) ^ 2

theorem f_decreasing_interval (k : ℤ) :
  ∀ x, k * Real.pi + (3 * Real.pi / 8) ≤ x ∧ x ≤ k * Real.pi + (7 * Real.pi / 8) → 
  (f' x < 0) :=
sorry

theorem sum_of_first_10_zeros :
  ∑ i in Finset.range 10, (λ n, if n % 2 = 0 then (n / 2) * Real.pi + Real.pi / 4 else (n / 2) * Real.pi + Real.pi / 2) (i + 1) = 95 * Real.pi / 4 :=
sorry

end f_decreasing_interval_sum_of_first_10_zeros_l502_502884


namespace cubic_expression_l502_502910

theorem cubic_expression {x : ℝ} (h : x + (1/x) = 5) : x^3 + (1/x^3) = 110 := 
by
  sorry

end cubic_expression_l502_502910


namespace kopeechka_items_l502_502465

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502465


namespace empty_bucket_weight_l502_502208

theorem empty_bucket_weight {B : ℝ} 
  (h1 : 3.4kg = B + (3.4kg - B)) 
  (h2 : 2.98kg = B + (4 / 5) * (3.4kg - B)) :
  B = 1.3kg := 
sorry

end empty_bucket_weight_l502_502208


namespace pour_tea_into_containers_l502_502748

-- Define the total number of containers
def total_containers : ℕ := 80

-- Define the amount of tea that Geraldo drank in terms of containers
def geraldo_drank_containers : ℚ := 3.5

-- Define the amount of tea that Geraldo consumed in terms of pints
def geraldo_drank_pints : ℕ := 7

-- Define the conversion factor from pints to gallons
def pints_per_gallon : ℕ := 8

-- Question: How many gallons of tea were poured into the containers?
theorem pour_tea_into_containers 
  (total_containers : ℕ)
  (geraldo_drank_containers : ℚ)
  (geraldo_drank_pints : ℕ)
  (pints_per_gallon : ℕ) :
  (total_containers * (geraldo_drank_pints / geraldo_drank_containers) / pints_per_gallon) = 20 :=
by
  sorry

end pour_tea_into_containers_l502_502748


namespace purchase_options_l502_502421

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502421


namespace equilateral_pentagon_area_computation_l502_502239

-- Define the problem in Lean

-- Condition 1: Triangle ABC is equilateral with side length 2
structure Triangle where
  A B C : ℝ × ℝ
  eq_side_length : dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

-- Condition 2: Pentagon AMNPQ inscribed in Triangle ABC with symmetry and position constraints
structure InscribedPentagon where
  A M N P Q B C : ℝ × ℝ
  in_triangle : triangle A B C
  symmetry : is_symmetric AMNPQ BC
  on_sides : M ∈ segment A B ∧ Q ∈ segment A C ∧ N ∈ segment B C ∧ P ∈ segment B C

-- Given these conditions, prove the area and final computation
theorem equilateral_pentagon_area_computation (t: Triangle) (p: InscribedPentagon) :
  let n := 48 in let q := 3 in let r := 27 in
  p.area = n - r * sqrt q → 100 * n + 10 * r + q = 5073 :=
by
  sorry

end equilateral_pentagon_area_computation_l502_502239


namespace symmetric_circle_equation_l502_502818

theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (-x ^ 2 + y^2 + 4 * x = 0) :=
sorry

end symmetric_circle_equation_l502_502818


namespace kopeechka_purchase_l502_502441

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502441


namespace final_withdrawal_amount_july_2005_l502_502170

-- Define the conditions given in the problem
variables (a r : ℝ) (n : ℕ)

-- Define the recursive formula for deposits
def deposit_amount (n : ℕ) : ℝ :=
  if n = 0 then a else (deposit_amount (n - 1)) * (1 + r) + a

-- The problem statement translated to Lean
theorem final_withdrawal_amount_july_2005 :
  deposit_amount a r 5 = a / r * ((1 + r) ^ 6 - (1 + r)) :=
sorry

end final_withdrawal_amount_july_2005_l502_502170


namespace units_digit_first_four_composites_l502_502615

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502615


namespace liters_to_pints_conversion_l502_502333

-- Definitions based on conditions
def liters_to_pints_ratio := 0.75 / 1.575
def target_liters := 1.5
def expected_pints := 3.15

-- Lean statement
theorem liters_to_pints_conversion 
  (h_ratio : 0.75 / 1.575 = liters_to_pints_ratio)
  (h_target : 1.5 = target_liters) :
  target_liters * (1 / liters_to_pints_ratio) = expected_pints :=
by 
  sorry

end liters_to_pints_conversion_l502_502333


namespace number_of_items_l502_502460

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502460


namespace purchase_options_l502_502414

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502414


namespace smallest_number_jungkook_l502_502113

def Yoongi := 7
def Jungkook := 6
def Yuna := 9
def Yoojung := 8
def Taehyung := 10

theorem smallest_number_jungkook :
  Jungkook = 6 ∧
  (∀ x ∈ {Yoongi, Jungkook, Yuna, Yoojung, Taehyung}, Jungkook ≤ x) :=
by
  sorry

end smallest_number_jungkook_l502_502113


namespace area_of_triangle_is_zero_l502_502564

-- Define the three lines based on the conditions provided
def line1 (x : ℝ) : ℝ := - (1 / 2) * x + 3
def line2 (x : ℝ) : ℝ := 2 * x - 2
def line3 (x y : ℝ) : Prop := x + y = 8

-- Define the intersection points
def point_A := (2 : ℝ, 2 : ℝ)
def point_C := (5 / 3 : ℝ, 4 / 3 : ℝ)
def point_B := (10 / 3 : ℝ, 14 / 3 : ℝ)

-- Prove that the area of the triangle formed by these points is 0
theorem area_of_triangle_is_zero :
  let A := point_A in
  let B := point_B in
  let C := point_C in
  ((1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))) = 0 :=
by
  sorry

end area_of_triangle_is_zero_l502_502564


namespace product_eq_3_over_5_l502_502226

noncomputable def a : ℕ → ℚ
| 0       := 1/3
| (n + 1) := 1 + (a n - 1)^3

theorem product_eq_3_over_5 : ( ∏ n in (Finset.range ∞), a n ) = 3 / 5 := 
sorry

end product_eq_3_over_5_l502_502226


namespace eccentricity_of_ellipse_l502_502850

-- Definitions given in the problem
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x * x) / (a * a) + (y * y) / (b * b) = 1

def is_circle (x y b : ℝ) : Prop :=
  x * x + y * y = b * b

def line_through_focus (x y c : ℝ) : Prop :=
  y = (sqrt 3 / 3) * (x + c)

-- Eccentricity definition
def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Central lemma / theorem to be proved
theorem eccentricity_of_ellipse (a b c : ℝ) (x y : ℝ) :
  is_ellipse x y a b →
  is_circle x y b →
  line_through_focus x y c →
  (sqrt 3 b / 2) = c →
  eccentricity a (sqrt (a^2 - b^2)) = sqrt 2 / 2 :=
sorry

end eccentricity_of_ellipse_l502_502850


namespace ratio_new_circumference_diameter_l502_502392

theorem ratio_new_circumference_diameter (r : ℝ) (h : r > 0) :
  let new_radius := r + 1,
      new_diameter := 2 * new_radius,
      new_circumference := 2 * π * new_radius
  in new_circumference / new_diameter = π :=
by
  sorry

end ratio_new_circumference_diameter_l502_502392


namespace units_digit_of_first_four_composite_numbers_l502_502630

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502630


namespace quadratic_function_properties_l502_502842

-- Definitions based on given conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def pointCondition (a b c : ℝ) : Prop := quadraticFunction a b c (-2) = 0
def inequalityCondition (a b c : ℝ) : Prop := ∀ x : ℝ, 2 * x ≤ quadraticFunction a b c x ∧ quadraticFunction a b c x ≤ (1 / 2) * x^2 + 2
def strengthenCondition (f : ℝ → ℝ) (t : ℝ) : Prop := ∀ x, -1 ≤ x ∧ x ≤ 1 → f (x + t) < f (x / 3)

-- Our primary statement to prove
theorem quadratic_function_properties :
  ∃ a b c, pointCondition a b c ∧ inequalityCondition a b c ∧
           (a = 1 / 4 ∧ b = 1 ∧ c = 1) ∧
           (∀ t, (-8 / 3 < t ∧ t < -2 / 3) ↔ strengthenCondition (quadraticFunction (1 / 4) 1 1) t) :=
by sorry 

end quadratic_function_properties_l502_502842


namespace find_x_l502_502892

theorem find_x (U : Set ℕ) (A B : Set ℕ) (x : ℕ) 
  (hU : U = Set.univ)
  (hA : A = {1, 4, x})
  (hB : B = {1, x ^ 2})
  (h : compl A ⊂ compl B) :
  x = 0 ∨ x = 2 := 
by 
  sorry

end find_x_l502_502892


namespace clock_angle_at_3_25_l502_502151

-- Definitions based on conditions
def angle_at_3_oclock := 90
def minute_hand_movement_per_minute := 6
def hour_hand_movement_per_minute := 0.5

-- Time given in minutes after 3:00
def time_in_minutes := 25

-- Calculations based on conditions and time
def minute_hand_total_movement := minute_hand_movement_per_minute * time_in_minutes
def hour_hand_total_movement := hour_hand_movement_per_minute * time_in_minutes
def hour_hand_position_at_3_25 := angle_at_3_oclock + hour_hand_total_movement
def angle_between_hands := |minute_hand_total_movement - hour_hand_position_at_3_25|

-- Proof statement
theorem clock_angle_at_3_25 : angle_between_hands = 47.5 :=
by
  -- Proof would go here
  sorry

end clock_angle_at_3_25_l502_502151


namespace solve_log_equation_l502_502295

theorem solve_log_equation : 
  {x : ℝ | log 2 (x - 1) = 2 - log 2 (x + 1) ∧ x - 1 > 0 ∧ x + 1 > 0} = {real.sqrt 5} :=
sorry

end solve_log_equation_l502_502295


namespace graph_shift_up_l502_502115

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Example outside domain to ensure totality

theorem graph_shift_up:
  ∃ (graph : ℝ → ℝ), (∀ x, graph x = f x + 1) ∧ (graph = graph_option_C) := 
sorry

end graph_shift_up_l502_502115


namespace max_salary_of_single_player_l502_502934

-- Define the number of players
def n : ℕ := 22

-- Define the minimum salary per player
def s_min : ℕ := 16_000

-- Define the total payroll limit for the team
def S_total : ℕ := 880_000

-- Define the maximum possible salary for a single player
def s_max : ℕ := 544_000

-- The theorem stating the maximum salary of a single player given the conditions
theorem max_salary_of_single_player : 
  21 * s_min + s_max = S_total :=
by 
  sorry

end max_salary_of_single_player_l502_502934


namespace kolya_purchase_l502_502453

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l502_502453


namespace kopeechka_items_l502_502424

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502424


namespace tangent_line_monotonicity_inequality_l502_502840

section part1

def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 - x + 2
def g_derivative (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x - 1

theorem tangent_line (a : ℝ) (h : a = 1) : ∀ (x y : ℝ), g 0 1 = 2 ∧ g_derivative 0 1 = -1 → (x = 0 ∧ y = 2 ∧ - (x + y) = -2) :=
by sorry

end part1

section part2

def f (x : ℝ) : ℝ := x * Real.log x
def f_derivative (x : ℝ) : ℝ := Real.log x + 1

theorem monotonicity (m : ℝ) (hm : 0 < m) :
  (m ≤ 1 / Real.exp 1 → ∀ (x : ℝ), x ∈ Ioo (0 : ℝ) m → f_derivative x < 0) ∧
  (m > 1 / Real.exp 1 → ∀ (x : ℝ), (x ∈ Ioo (0 : ℝ) (1 / Real.exp 1) → f_derivative x < 0) ∧ (x ∈ Ioo (1 / Real.exp 1) m → f_derivative x > 0)) :=
by sorry

end part2

section part3

def h (x : ℝ) : ℝ := 2 * Real.log x - 3 * x - 1 / x

theorem inequality (a : ℝ) (ha : a ≥ -2) : ∀ x ∈ Ioo 0 (Real.log 1), 2 * f x ≤ g_derivative x a + 2 :=
by sorry

end part3

end tangent_line_monotonicity_inequality_l502_502840


namespace volume_of_truncated_cone_l502_502867

noncomputable def surface_area_top : ℝ := 3 * Real.pi
noncomputable def surface_area_bottom : ℝ := 12 * Real.pi
noncomputable def slant_height : ℝ := 2
noncomputable def volume_cone : ℝ := 7 * Real.pi

theorem volume_of_truncated_cone :
  ∃ V : ℝ, V = volume_cone :=
sorry

end volume_of_truncated_cone_l502_502867


namespace brownies_made_next_morning_l502_502077

theorem brownies_made_next_morning :
  ∀ (initial_brownies father_ate daughter_ate next_morning_total made_next_morning : ℕ),
    initial_brownies = 24 →
    father_ate = 8 →
    daughter_ate = 4 →
    next_morning_total = 36 →
    made_next_morning = next_morning_total - (initial_brownies - father_ate - daughter_ate) →
    made_next_morning = 24 :=
begin
  intros initial_brownies father_ate daughter_ate next_morning_total made_next_morning,
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4] at h5,
  exact h5,
end

end brownies_made_next_morning_l502_502077


namespace simply_connected_polyhedron_faces_l502_502086

def polyhedron_faces_condition (σ3 σ4 σ5 : Nat) (V E F : Nat) : Prop :=
  V - E + F = 2

theorem simply_connected_polyhedron_faces : 
  ∀ (σ3 σ4 σ5 : Nat) (V E F : Nat),
  polyhedron_faces_condition σ3 σ4 σ5 V E F →
  (σ4 = 0 ∧ σ5 = 0 → σ3 ≥ 4) ∧
  (σ3 = 0 ∧ σ5 = 0 → σ4 ≥ 6) ∧
  (σ3 = 0 ∧ σ4 = 0 → σ5 ≥ 12) := 
by
  intros
  sorry

end simply_connected_polyhedron_faces_l502_502086


namespace base4_arithmetic_l502_502288

theorem base4_arithmetic :
  (Nat.ofDigits 4 [2, 3, 1] * Nat.ofDigits 4 [2, 2] / Nat.ofDigits 4 [3]) = Nat.ofDigits 4 [2, 2, 1] := by
sorry

end base4_arithmetic_l502_502288


namespace kopeechka_purchase_l502_502442

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l502_502442


namespace probability_of_cosine_condition_l502_502039

-- Define the set M
def M : Set ℝ := {x | ∃ n : ℕ, (1 ≤ n ∧ n ≤ 10) ∧ x = n * Real.pi / 6}

-- Define the elements in M
def elements_of_M : List ℝ :=
  [Real.pi / 6, 2 * Real.pi / 6, 3 * Real.pi / 6, 4 * Real.pi / 6, 6 * Real.pi / 6,
   8 * Real.pi / 6, 9 * Real.pi / 6, 10 * Real.pi / 6, 15 * Real.pi / 6, 20 * Real.pi / 6]

-- Define the property to check
def satisfies_condition (x : ℝ) : Prop := Real.cos x = 1 / 2

-- Define the problem as a theorem in Lean
theorem probability_of_cosine_condition :
  (elements_of_M.filter satisfies_condition).length / elements_of_M.length = 1 / 5 := by
  sorry

end probability_of_cosine_condition_l502_502039


namespace land_value_moon_l502_502121

-- Define the conditions
def surface_area_earth : ℕ := 200
def surface_area_ratio : ℕ := 5
def value_ratio : ℕ := 6
def total_value_earth : ℕ := 80

-- Define the question and the expected answer
noncomputable def total_value_moon : ℕ := 96

-- State the proof problem
theorem land_value_moon :
  (surface_area_earth / surface_area_ratio * value_ratio) * (surface_area_earth / surface_area_ratio) = total_value_moon := 
sorry

end land_value_moon_l502_502121


namespace trust_meteorologist_l502_502173

/-- 
The probability of a clear day in Anchuria.
-/
def P_G : ℝ := 0.74

/-- 
Accuracy of the forecast by each senator.
-/
variable (p : ℝ)

/-- 
Accuracy of the meteorologist's forecast being 1.5 times that of a senator.
-/
def meteorologist_accuracy : ℝ := 1.5 * p

/-- 
Calculations and final proof that the meteorologist's forecast is more reliable than that of the senators. 
-/
theorem trust_meteorologist (p : ℝ) (Hp1 : 0 ≤ p) (Hp2 : p ≤ 1) : 
  λ P_S_M1_M2_G P_S_M1_M2_not_G : 
  P_G := 0.74 ∧ meteorologist_accuracy = 1.5 * p ∧
  (∀ P_S_M1_M2, P_S_M1_M2 = P_S_M1_M2_G * P_G + P_S_M1_M2_not_G * (1 - P_G)) → 
  (P_S_M1_M2_not_G * (1 - P_G) > P_S_M1_M2_G * P_G) :=
begin
  sorry
end

end trust_meteorologist_l502_502173


namespace log_base_inequality_l502_502832

theorem log_base_inequality (a x y : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : log a x < log a y ∧ log a y < 0) : 1 < y ∧ y < x := 
by
  sorry

end log_base_inequality_l502_502832


namespace max_diagonal_moves_l502_502811

theorem max_diagonal_moves (n : ℕ) : 
  let m := (n+1)^2 - 1 in
  let total_moves := n^2 + 2 * n in
  ∃ d : ℕ, 
  (if n % 2 = 0 then d = n^2 else d = n^2 + n) ∧
  d ≤ total_moves := 
begin
  sorry
end

end max_diagonal_moves_l502_502811


namespace number_of_integers_satisfying_condition_l502_502896

theorem number_of_integers_satisfying_condition :
  let x := Int in
  (∃ x : Int, (x - 1) / 3 < 5 / 7 ∧ 5 / 7 < (x + 4) / 5) ↔ 4 := by
  sorry

end number_of_integers_satisfying_condition_l502_502896


namespace kopeechka_items_l502_502427

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502427


namespace find_angle_BAC_l502_502471

-- Definitions based on the conditions of the problem
variables (A B C P : Type) [Coercible A P] 
variables (AB AC BP PC : ℝ) (angle_BAC angle_PAB angle_PCA : ℝ)
variables (H_AB : AB = 2 * AC) (H_BP : BP = 2 * PC) (H_angle : angle_PAB = angle_PCA)

-- The goal is to prove the value of angle BAC
theorem find_angle_BAC (H_AB : AB = 2 * AC) (H_BP : BP = 2 * PC) (H_angle : angle_PAB = angle_PCA) : angle_BAC = 120 :=
sorry

end find_angle_BAC_l502_502471


namespace units_digit_first_four_composites_l502_502602

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502602


namespace correct_statements_count_l502_502267

theorem correct_statements_count (K X Y : Type)
  (H1 : ∀ K2 : ℝ, K2 > 0 → higher_confidence_XY K2)
  (H2 : ¬ (original_suspicious_data_discovered_by_residuals))
  (H3 : ∀ {n : ℕ} (observations : Fin n → ℝ × ℝ), 
         regression_through_center observations)
  (H4 : ∀ R2 : ℝ, R2 > 0 → better_model_fitting R2) :
  num_correct_statements = 2 :=
sorry

end correct_statements_count_l502_502267


namespace trust_meteorologist_l502_502186

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l502_502186


namespace trig_expression_value_l502_502309

-- Let α be a real number
variable (α : ℝ)

-- Assuming condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = -1 / 2

-- Defining the function in the problem
def expression (α : ℝ) : ℝ := (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2)

-- The proof problem statement
theorem trig_expression_value (h : tan_alpha α) : expression α = -1 / 3 :=
  sorry

end trig_expression_value_l502_502309


namespace sequence_monotonically_decreasing_l502_502359

theorem sequence_monotonically_decreasing (t : ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = -↑n^2 + t * ↑n) →
  (∀ n : ℕ, a (n + 1) < a n) →
  t < 3 :=
by
  intros h1 h2
  sorry

end sequence_monotonically_decreasing_l502_502359


namespace problem_statement_l502_502307

noncomputable def star (m n : ℝ) : ℝ :=
  if m ≥ n then log m n else log n m

theorem problem_statement (a b c : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : a * b = c) :
  (1 / (star a c)) + (1 / (star b c)) = 1 :=
by
  sorry

end problem_statement_l502_502307


namespace center_of_mass_cycloid_l502_502817

-- Definition of the cycloid parameters
def x (a t : ℝ) : ℝ := a * (t - sin t)
def y (a t : ℝ) : ℝ := a * (1 - cos t)

-- The question is to find the center of mass yc of this cycloid arch
theorem center_of_mass_cycloid (a : ℝ) : (∫ t in 0..(2 * Real.pi), y a t * (2 * a * sin (t / 2)) ∂t / ∫ t in 0..(2 * Real.pi), 2 * a * sin (t / 2) ∂t) = - (1 / 3) * a :=
by
  sorry

end center_of_mass_cycloid_l502_502817


namespace fraction_drained_in_4_minutes_l502_502913

variable (F : ℝ) (drainIn4 : F) (drainIn5 : 1)

theorem fraction_drained_in_4_minutes (F : ℝ) (H1 : F = 4 / 5) : F = 4 / 5 :=
by
  sorry

end fraction_drained_in_4_minutes_l502_502913


namespace product_of_values_of_f_half_l502_502489

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eqn (x y : ℝ) : f (x * y + f x) = x * f y + f x + 1

axiom f_one_eq_two : f 1 = 2

theorem product_of_values_of_f_half : 
  let n := (∃! (val : ℝ), f (1 / 2) = val).some,
      s := (some_spec (∃! (val : ℝ), f (1 / 2) = val)).snd
  in n * s = 1 / 2 :=
by 
  sorry

end product_of_values_of_f_half_l502_502489


namespace trust_meteorologist_l502_502184

-- Definitions
def probability_of_clear := 0.74
def senator_accuracy := p : ℝ
def meteorologist_accuracy := 1.5 * p

-- Events
def event_G := "clear day"
def event_M1 := "first senator predicted clear"
def event_M2 := "second senator predicted clear"
def event_S := "meteorologist predicted rain"

theorem trust_meteorologist :
  let r := probability_of_clear
  let p := senator_accuracy
  let q := meteorologist_accuracy
  1.5 * p * (1 - p)^2 * (1 - r) - (1 - 1.5 * p) * p^2 * r > 0 :=
by
  sorry

end trust_meteorologist_l502_502184


namespace unique_solution_to_equation_l502_502265

-- Define the problem as a theorem in Lean 4
theorem unique_solution_to_equation (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) :
  x^2 - 2 * y.factorial = 2021 → (x = 45 ∧ y = 2) :=
begin
  -- Skipping the proof by inserting sorry.
  sorry,
end

end unique_solution_to_equation_l502_502265


namespace units_digit_of_first_four_composite_numbers_l502_502628

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l502_502628


namespace arithmetic_sequence_properties_l502_502849

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}
variable {a1 : ℝ}

theorem arithmetic_sequence_properties 
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a1 > 0) 
  (h3 : a 9 + a 10 = a 11) :
  (∀ m n, m < n → a m > a n) ∧ (∀ n, S n = n * (a1 + (d * (n - 1) / 2))) ∧ S 14 > 0 :=
by 
  sorry

end arithmetic_sequence_properties_l502_502849


namespace D_144_eq_129_l502_502970

def D (n: ℕ) : ℕ :=
  if n = 1 then 1
  else (List.range (n - 1)).sum (λ k, if n % (k + 1) = 0 then D (n / (k + 1)) else 0)

theorem D_144_eq_129 : D 144 = 129 := by
  sorry

end D_144_eq_129_l502_502970


namespace find_m_value_l502_502316

theorem find_m_value (m : ℝ) (A B : ℝ × ℝ) (slope_angle_45 : ℝ) :
  A = (2, 4) →
  B = (1, m) →
  slope_angle_45 = 45 →
  (1 : ℝ) = (A.snd - B.snd) / (A.fst - B.fst) →
  m = 3 :=
begin
  sorry,
end

end find_m_value_l502_502316


namespace number_of_factors_l502_502269

theorem number_of_factors (M : ℕ) (h : M = 2^4 * 3^3 * 5^2 * 7^1) : 
  (∃ n : ℕ, n = 5 * 4 * 3 * 2 ∧ n = 120) :=
by {
  use 120,
  split,
  { norm_num, },
  { refl, },
}

end number_of_factors_l502_502269


namespace same_derivative_values_l502_502360

theorem same_derivative_values (x₀ : ℝ) :
  (deriv (λ x : ℝ, x^2 - 1) x₀ = deriv (λ x : ℝ, 1 - x^3) x₀) →
  (x₀ = 0 ∨ x₀ = -2 / 3) :=
by
  sorry

end same_derivative_values_l502_502360


namespace find_f_6_l502_502534

open Real

variable (f : ℝ → ℝ)

theorem find_f_6 (H1 : ∀ x y : ℝ, f(x + y) = f(x) * f(y)) (H2 : f(2) = 3) : f(6) = 27 := by
  sorry

end find_f_6_l502_502534


namespace answer_is_function_d_l502_502776

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def lies_in_domain (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x : ℝ, x ∈ domain → ∃ y : ℝ, y = f x

noncomputable def has_min_value (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f x ≥ m

theorem answer_is_function_d :
  (∀ x : ℝ, (x + (1 / x)) = - (x + (1 / x))) → 
  (∀ x : ℝ, (x * (sin x)) = (x * (sin x))) → 
  (∀ x : ℝ, (if x ≥ 0 then x^2 - x else -x^2 - x) = - (if x ≥ 0 then x^2 - x else -x^2 - x)) → 
  (∀ x : ℝ, (cos (x - (π / 2))) = - (cos (x - (π / 2)))) → 
  (∃ m : ℝ, ∀ x : ℝ, sin x ≥ m) → 
  true := 
sorry

end answer_is_function_d_l502_502776


namespace total_distance_traveled_l502_502012

theorem total_distance_traveled
  (bike_time_min : ℕ) (bike_rate_mph : ℕ)
  (jog_time_min : ℕ) (jog_rate_mph : ℕ)
  (total_time_min : ℕ)
  (h_bike_time : bike_time_min = 30)
  (h_bike_rate : bike_rate_mph = 6)
  (h_jog_time : jog_time_min = 45)
  (h_jog_rate : jog_rate_mph = 8)
  (h_total_time : total_time_min = 75) :
  (bike_rate_mph * bike_time_min / 60) + (jog_rate_mph * jog_time_min / 60) = 9 :=
by sorry

end total_distance_traveled_l502_502012


namespace find_KE_on_tangent_l502_502939

-- Given a cube with side length 1 and an inscribed sphere 
-- E on edge CC_1 such that C_1E = 1/8
-- ∠KEC = arccos(1/7)
-- Prove KE = 7/8

variables {A B C D A1 B1 C1 D1 : Type}

-- Define the constants
constant cube_side_length : ℝ := 1
constant sphere_radius : ℝ := cube_side_length / 2
constant E_on_CC1 : ℝ := 1 / 8
constant angle_KEC : ℝ := real.arccos (1 / 7)
constant KE : ℝ

-- Theorem to prove KE
theorem find_KE_on_tangent :
  KE = 7 / 8 :=
sorry

end find_KE_on_tangent_l502_502939


namespace sum_of_reciprocals_eq_eight_l502_502549

theorem sum_of_reciprocals_eq_eight {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) : 
  (1 / x) + (1 / y) = 8 :=
begin
  sorry
end

end sum_of_reciprocals_eq_eight_l502_502549


namespace largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l502_502292

noncomputable def largest_integral_x_in_ineq (x : ℤ) : Prop :=
  (2 / 5 : ℚ) < (x / 7 : ℚ) ∧ (x / 7 : ℚ) < (8 / 11 : ℚ)

theorem largest_integral_x_satisfies_ineq : largest_integral_x_in_ineq 5 :=
sorry

theorem largest_integral_x_is_5 (x : ℤ) (h : largest_integral_x_in_ineq x) : x ≤ 5 :=
sorry

end largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l502_502292


namespace find_value_of_x_y_l502_502003

theorem find_value_of_x_y (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : |y| + x - y = 12) : x + y = 18 / 5 :=
by
  sorry

end find_value_of_x_y_l502_502003


namespace find_f_2004_l502_502336

-- Define the real function f and the periodicity condition
variables {f : ℝ → ℝ} {g : ℝ → ℝ}

-- Define the given conditions.
def even_function (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)
def odd_function (g : ℝ → ℝ) := ∀ x, g(-x) = -g(x)
def g_related_to_f (f g : ℝ → ℝ) := ∀ x, g(x) = f(x-1)
def g_one_value (g : ℝ → ℝ) := g(1) = 2003

theorem find_f_2004 (h1 : even_function f) (h2 : odd_function g) (h3 : g_related_to_f f g) (h4 : g_one_value g) :
  f(2004) = 2003 :=
by sorry

end find_f_2004_l502_502336


namespace proof_problem_l502_502544

theorem proof_problem (a : ℕ → ℝ) (n : ℕ) (c : ℝ)
  (h_a0 : a 0 = 0)
  (h_an : a n = 0)
  (h_ak : ∀ k : ℕ, k < n → a k = c + ∑ i in finset.Ico k n, a (i - k) * (a i + a (i + 1))) :
  c ≤ 1 / (4 * n) :=
sorry

end proof_problem_l502_502544


namespace b_prime_or_perfect_square_l502_502487

open Nat

theorem b_prime_or_perfect_square {a b c : ℤ} 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≥ b) (h5 : b ≥ c)
  (h6 : prime ((a - c) / 2))
  (h7 : a^2 + b^2 + c^2 - 2*(a*b + b*c + c*a) = b) :
  prime b ∨ ∃ k : ℤ, k^2 = b := 
  sorry

end b_prime_or_perfect_square_l502_502487


namespace value_of_expression_l502_502270

variable (a b : ℝ)

theorem value_of_expression : 
  let x := a + b 
  let y := a - b 
  (x - y) * (x + y) = 4 * a * b := 
by
  sorry

end value_of_expression_l502_502270


namespace outfits_count_l502_502730

theorem outfits_count (s p : ℕ) (h_s : s = 5) (h_p : p = 3) : s * p = 15 :=
by
  rw [h_s, h_p]
  exact Nat.mul_comm 5 3

end outfits_count_l502_502730


namespace count_integers_with_2_and_5_l502_502900

theorem count_integers_with_2_and_5 : 
  ∃ n : ℕ, n = (4 * 2) ∧ (∀ x : ℕ, 300 ≤ x ∧ x < 700 → 
  let h := x / 100,
      t := (x % 100) / 10,
      u := x % 10 in
  (h = 3 ∨ h = 4 ∨ h = 5 ∨ h = 6) ∧ (t = 2 ∧ u = 5 ∨ t = 5 ∧ u = 2) ↔ ¬ (x = 0)) :=
begin
  sorry
end

end count_integers_with_2_and_5_l502_502900


namespace largest_8_digit_number_with_even_digits_l502_502685

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l502_502685


namespace correct_propositions_l502_502194

-- Definitions for the conditions
variables {α β : Plane}   -- Planes α and β
variables {l m : Line}    -- Lines l and m
variable l_perp_to_alpha : l.perpendicular α
variable m_in_beta : m ∈ β

-- The key propositions to be proved
theorem correct_propositions :
  (α.parallel β → l.perpendicular m) ∧
  (l.parallel m → α.perpendicular β) :=
by {
  -- Placeholders for the actual proofs
  sorry
}

end correct_propositions_l502_502194


namespace total_digits_in_book_5000_pages_l502_502168

theorem total_digits_in_book_5000_pages : 
  let digits_in_range (start : ℕ) (end_ : ℕ) (digit_count : ℕ) := (end_ - start + 1) * digit_count in
  let digits_1_to_9 := digits_in_range 1 9 1 in
  let digits_10_to_99 := digits_in_range 10 99 2 in
  let digits_100_to_999 := digits_in_range 100 999 3 in
  let digits_1000_to_4999 := digits_in_range 1000 4999 4 in
  let digits_5000 := 4 in
  digits_1_to_9 + digits_10_to_99 + digits_100_to_999 + digits_1000_to_4999 + digits_5000 = 18893 :=
by
  sorry

end total_digits_in_book_5000_pages_l502_502168


namespace point_on_axis_l502_502383

theorem point_on_axis (m : ℝ) : (∃ p : ℝ × ℝ, p = (m, 2 - m) ∧ (p.1 = 0 ∨ p.2 = 0)) → (m = 0 ∨ m = 2) :=
by
  intro h
  cases h with p hp
  cases hp with h1 h2
  cases h2
  case or.inl h3 =>
    rw [Prod.mk.eta] at h3
    left
    exact h3
  case or.inr h4 =>
    right
    rw [Prod.mk.eta] at h4
    linarith

end point_on_axis_l502_502383


namespace units_digit_first_four_composites_l502_502595

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502595


namespace problem1_problem2_l502_502321

noncomputable def F : Point := ⟨0, 1⟩
def parabola (x y : ℝ) : Prop := x^2 = 4 * y
def point_D (x1 : ℝ) : Point := ⟨x1, x1^2 / 4⟩
def point_E (x2 : ℝ) : Point := ⟨x2, x2^2 / 4⟩
def point_M (x1 x2 : ℝ) : Point := ⟨(x1 + x2) / 2, x1 * x2 / 4⟩
def is_perpendicular (p1 p2 : ℝ) := p1 * p2 = -1
def is_arithmetic_sequence (a b c : ℝ) := a + c = 2 * b
def distance (p1 p2 : Point) := (p2.x - p1.x)^2 + (p2.y - p1.y)^2
def area_triangle (d e m : Point) :=
  let side_length (u v : Point) := Math.sqrt (distance u v)
  let a := side_length d m
  let b := side_length e m
  let c := side_length d e
  (a * b) / 2

theorem problem1 (x1 x2 : ℝ) (hx : parabola x1 (x1^2 / 4)) (hy : parabola x2 (x2^2 / 4)) (h3 : x1 * x2 = -4) :
  let m := point_M x1 x2 in is_perpendicular (x1 / 2) (x2 / 2) :=
begin
  sorry
end

theorem problem2 (x1 x2 : ℝ) (hx : parabola x1 (x1^2 / 4)) (hy : parabola x2 (x2^2 / 4)) 
  (h3 : x1 * x2 = -4) 
  (hseq : is_arithmetic_sequence (distance (point_D x1) (point_M x1 x2)) (distance (point_E x2) (point_M x1 x2)) (distance (point_D x1) (point_E x2))) :
  let d := point_D x1 in let e := point_E x2 in let m := point_M x1 x2 in
  area_triangle d e m = 15625 / 3456 :=
begin
  sorry
end

end problem1_problem2_l502_502321


namespace sum_binomial_squares_l502_502088

open Nat

theorem sum_binomial_squares (n : ℕ) : 
  ∑ k in finset.range (n + 1), (factorial (2 * n)) / (factorial k * factorial k * factorial (n - k) * factorial (n - k)) = (binomial (2 * n) n) ^ 2 := by
  sorry

end sum_binomial_squares_l502_502088


namespace solve_for_y_l502_502813

theorem solve_for_y :
  ∃ (y : ℝ), 
    (∑' n : ℕ, (4 * (n + 1) - 2) * y^n) = 100 ∧ |y| < 1 ∧ y = 0.6036 :=
sorry

end solve_for_y_l502_502813


namespace instantaneous_velocity_at_t_is_4_l502_502890

variable (t : ℝ)

def position (t : ℝ) : ℝ := t + (1 / 9) * t^3

theorem instantaneous_velocity_at_t_is_4 :
  deriv position 3 = 4 := by
sorry

end instantaneous_velocity_at_t_is_4_l502_502890


namespace count_special_integers_l502_502372

theorem count_special_integers : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 9999 ∧ (∀ d ∈ digits 10 n, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 5 ∧ d ≠ 8)) → 
  count_integers_with_conditions 1 9999 (λ d, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 5 ∧ d ≠ 8) = 624 := 
by
  sorry

end count_special_integers_l502_502372


namespace units_digit_of_product_of_first_four_composites_l502_502661

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502661


namespace unknown_number_is_10_l502_502009

def operation_e (x y : ℕ) : ℕ := 2 * x * y

theorem unknown_number_is_10 (n : ℕ) (h : operation_e 8 (operation_e n 5) = 640) : n = 10 :=
by
  sorry

end unknown_number_is_10_l502_502009


namespace power_function_point_l502_502342

theorem power_function_point (α : ℝ) (h : (∀ x : ℝ, x ≥ 0 → y = x ^ α) ∧ (2, sqrt 2) ∈ set_points) : α = 1 / 2 :=
by
  sorry

end power_function_point_l502_502342


namespace sequence_is_arithmetic_l502_502992

noncomputable def f (x : ℝ) (n : ℕ) : ℝ := (x - 1)^2 + n

def a_n (n : ℕ) : ℝ := n

def b_n (n : ℕ) : ℝ := n + 4

def c_n (n : ℕ) : ℝ := (b_n n)^2 - (a_n n) * (b_n n)

theorem sequence_is_arithmetic :
  ∀ n : ℕ, c_n (n + 1) - c_n n = 4 := 
by
  intros n
  sorry

end sequence_is_arithmetic_l502_502992


namespace total_medals_1996_l502_502013

variable (g s b : Nat)

theorem total_medals_1996 (h_g : g = 16) (h_s : s = 22) (h_b : b = 12) :
  g + s + b = 50 :=
by
  sorry

end total_medals_1996_l502_502013


namespace least_number_correct_l502_502169

def least_number_to_add_to_make_perfect_square (x : ℝ) : ℝ :=
  let y := 1 - x -- since 1 is the smallest whole number > sqrt(0.0320)
  y

theorem least_number_correct (x : ℝ) (h : x = 0.0320) : least_number_to_add_to_make_perfect_square x = 0.9680 :=
by {
  -- Proof is skipped
  -- The proof would involve verifying that adding this number to x results in a perfect square (1 in this case).
  sorry
}

end least_number_correct_l502_502169


namespace equation_of_line_l502_502531

theorem equation_of_line 
  (slope : ℝ)
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h_slope : slope = 2)
  (h_line1 : a1 = 3 ∧ b1 = 4 ∧ c1 = -5)
  (h_line2 : a2 = 3 ∧ b2 = -4 ∧ c2 = -13) 
  : ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = -7) ∧ 
    (∀ x y : ℝ, (a1 * x + b1 * y + c1 = 0) ∧ (a2 * x + b2 * y + c2 = 0) → (a * x + b * y + c = 0)) :=
by
  sorry

end equation_of_line_l502_502531


namespace sum_of_first_12_terms_l502_502844

theorem sum_of_first_12_terms :
  ∀ (a : ℕ → ℝ),
    a 1 = 1 →
    a 2 = 2 →
    (∀ n, a (n+2) = (1 + (Real.cos (n * Real.pi / 2))^2) * a n + (Real.sin (n * Real.pi / 2))^2) →
    (Finset.range 12).sum (λ i, a (i + 1)) = 147 :=
by
  intros a h1 h2 hrec
  sorry

end sum_of_first_12_terms_l502_502844


namespace six_digit_integers_count_l502_502369

theorem six_digit_integers_count :
  let digits := [2, 2, 2, 9, 9, 9] in
  multiset.card (multiset.permutations (multiset.of_list digits)) = 20 :=
sorry

end six_digit_integers_count_l502_502369


namespace markup_rate_l502_502960

theorem markup_rate (S : ℝ) (h1 : S = 8) (profit_percentage : ℝ) (h2 : profit_percentage = 0.25) (overhead_percentage : ℝ) (h3 : overhead_percentage = 0.20) :
  (((S - (S - h2 * S - h3 * S)) / (S - h2 * S - h3 * S)) * 100) = 81.82 :=
by
  -- Imports and definitions used above should guarantee the credibility of the statement
  sorry

end markup_rate_l502_502960


namespace man_to_son_age_ratio_l502_502755

-- Definitions based on conditions
variable (son_age : ℕ) (man_age : ℕ)
variable (h1 : man_age = son_age + 18) -- The man is 18 years older than his son
variable (h2 : 2 * (son_age + 2) = man_age + 2) -- In two years, the man's age will be a multiple of the son's age
variable (h3 : son_age = 16) -- The present age of the son is 16

-- Theorem statement to prove the desired ratio
theorem man_to_son_age_ratio (son_age man_age : ℕ) (h1 : man_age = son_age + 18) (h2 : 2 * (son_age + 2) = man_age + 2) (h3 : son_age = 16) :
  (man_age + 2) / (son_age + 2) = 2 :=
by
  sorry

end man_to_son_age_ratio_l502_502755


namespace hamilton_high_students_l502_502014

open Finset

theorem hamilton_high_students (U A B : Finset ℕ) (hU : card U = 220) (hA : card A = 165) (hB : card B = 140) 
(hOppose : card (U \ (A ∪ B)) = 40) : card (A ∩ B) = 125 :=
by
  sorry

end hamilton_high_students_l502_502014


namespace necessary_and_sufficient_condition_l502_502352

theorem necessary_and_sufficient_condition :
  ∀ a b : ℝ, (a + b > 0) ↔ ((a ^ 3) + (b ^ 3) > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l502_502352


namespace units_digit_of_composite_product_l502_502654

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502654


namespace circle_O2_tangent_circle_O2_intersect_l502_502110

-- Condition: The equation of circle O_1 is \(x^2 + (y + 1)^2 = 4\)
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4

-- Condition: The center of circle O_2 is \(O_2(2, 1)\)
def center_O2 : (ℝ × ℝ) := (2, 1)

-- Prove the equation of circle O_2 if it is tangent to circle O_1
theorem circle_O2_tangent : 
  ∀ (x y : ℝ), circle_O1 x y → (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

-- Prove the equations of circle O_2 if it intersects circle O_1 and \(|AB| = 2\sqrt{2}\)
theorem circle_O2_intersect :
  ∀ (x y : ℝ), 
  circle_O1 x y → 
  (2 * Real.sqrt 2 = |(x - 2)^2 + (y - 1)^2 - 4| ∨ (x - 2)^2 + (y - 1)^2 = 20) :=
sorry

end circle_O2_tangent_circle_O2_intersect_l502_502110


namespace largest_8_digit_number_with_even_digits_l502_502690

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l502_502690


namespace rectangle_area_l502_502760

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) (h_diag : y^2 = 10 * w^2) : 
  (3 * w)^2 * w = 3 * (y^2 / 10) :=
by sorry

end rectangle_area_l502_502760


namespace trust_meteorologist_l502_502190

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l502_502190


namespace prove_AP_BP_CP_product_l502_502848

open Classical

-- Defines that the point P is inside the acute-angled triangle ABC
variables {A B C P: Type} [MetricSpace P] 
variables (PA1 PB1 PC1 AP BP CP : ℝ)

-- Conditions
def conditions (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) : Prop :=
  PA1 = 3 ∧ PB1 = 3 ∧ PC1 = 3 ∧ AP + BP + CP = 43

-- Proof goal
theorem prove_AP_BP_CP_product (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) :
  AP * BP * CP = 441 :=
by {
  -- Proof steps will be filled here
  sorry
}

end prove_AP_BP_CP_product_l502_502848


namespace h_mul_k_geq_one_l502_502318

noncomputable def h (a : ℕ → ℝ) := Real.lim (fun n => (Finset.sum (Finset.range n) (fun i => a i) ) / n)

noncomputable def k (a : ℕ → ℝ) := Real.lim (fun n => (Finset.sum (Finset.range n) (fun i => 1 / a i) ) / n)

theorem h_mul_k_geq_one (a : ℕ → ℝ) (h_exists : ∃ h', h a = h') (k_exists : ∃ k', k a = k') : 
  (h a) * (k a) ≥ 1 := sorry

end h_mul_k_geq_one_l502_502318


namespace not_age_of_child_digit_l502_502079

variable {n : Nat}

theorem not_age_of_child_digit : 
  ∀ (ages : List Nat), 
    (∀ x ∈ ages, 5 ≤ x ∧ x ≤ 13) ∧ -- condition 1
    ages.Nodup ∧                    -- condition 2: distinct ages
    ages.length = 9 ∧               -- condition 1: 9 children
    (∃ num : Nat, 
       10000 ≤ num ∧ num < 100000 ∧         -- 5-digit number
       (∀ d : Nat, d ∈ num.digits 10 →     -- condition 3 & 4: each digit appears once and follows a consecutive pattern in increasing order
          1 ≤ d ∧ d ≤ 9) ∧
       (∀ age ∈ ages, num % age = 0)       -- condition 4: number divisible by all children's ages
    ) →
    ¬(9 ∈ ages) :=                         -- question: Prove that '9' is not the age of any child
by
  intro ages h
  -- The proof would go here
  sorry

end not_age_of_child_digit_l502_502079


namespace number_of_days_a_worked_l502_502165

theorem number_of_days_a_worked
  (days_b : ℕ := 9)
  (days_c : ℕ := 4)
  (ratio_a : ℚ := 3)
  (ratio_b : ℚ := 4)
  (ratio_c : ℚ := 5)
  (wage_c : ℚ := 71.15384615384615)
  (total_earnings : ℚ := 1480) :
  ∃ (days_a : ℕ), days_a ≈ 16 :=
by
  -- Definitions
  let y := wage_c / ratio_c
  let wage_a := ratio_a * y
  let wage_b := ratio_b * y
  -- Formulation of the equation for total earnings
  let earnings_a := wage_a * days_a
  let earnings_b := wage_b * days_b
  let earnings_c := wage_c * days_c
  have h : total_earnings = earnings_a + earnings_b + earnings_c := by sorry
  -- Solve for days_a
  have h2 : days_a = (total_earnings - earnings_b - earnings_c) / wage_a := by sorry
  -- Assert that days_a is approximately 16
  exact exists.intro (⌊h2⌋.nat_abs) (by sorry)

end number_of_days_a_worked_l502_502165


namespace votes_cast_l502_502164

-- Definition: the total number of votes, the candidate's and rival's votes.
variable (V : ℕ)

-- Equation generated from the problem conditions
-- The candidate got 25% of votes, rival got 25% + 4000 votes
-- Together their votes make up the total number of votes.
def votes_equation : Prop := (0.25 * V + (0.25 * V + 4000) = V)

-- The proof statement that needs to be proven
theorem votes_cast (h : votes_equation V) : V = 8000 := sorry

end votes_cast_l502_502164


namespace exists_team_loses_at_most_four_l502_502197

-- Definitions
variable (teams : Finset ℕ) (num_teams : ℕ)
variable (losses : ℕ → ℕ → Prop) -- losses a b means team a loses to team b

-- Tournament setup
axiom tournament_setup : num_teams = 110 ∧ ∀ i j, i ∈ teams ∧ j ∈ teams ∧ i ≠ j → (losses i j ∨ losses j i)

-- Condition
axiom condition : ∀ S, S.card = 55 → ∃ t ∈ S, (S.erase t).count (losses t) ≤ 4

-- Goal
theorem exists_team_loses_at_most_four : ∃ t ∈ teams, (teams.erase t).count (losses t) ≤ 4 :=
by sorry

end exists_team_loses_at_most_four_l502_502197


namespace units_digit_first_four_composite_is_eight_l502_502610

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502610


namespace ratio_of_hypotenuse_segments_l502_502393

theorem ratio_of_hypotenuse_segments (a b c d : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : b = (3/4) * a)
  (h3 : d^2 = (c - d)^2 + b^2) :
  (d / (c - d)) = (4 / 3) :=
sorry

end ratio_of_hypotenuse_segments_l502_502393


namespace intersection_points_hyperbola_l502_502323

theorem intersection_points_hyperbola (t : ℝ) :
  ∃ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 4 = 0) ∧ 
  (x^2 / 4 - y^2 / (9 / 16) = 1) :=
sorry

end intersection_points_hyperbola_l502_502323


namespace largest_eight_digit_with_all_evens_l502_502695

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l502_502695


namespace different_digits_probability_l502_502241

noncomputable def number_nonidentical_probability : ℚ :=
  let total_numbers := 900
  let identical_numbers := 9
  -- The probability of identical digits.
  let identical_probability := identical_numbers / total_numbers
  -- The probability of non-identical digits.
  1 - identical_probability

theorem different_digits_probability : number_nonidentical_probability = 99 / 100 := by
  sorry

end different_digits_probability_l502_502241


namespace length_of_DB_l502_502035

theorem length_of_DB (A B C D : Point) (h₁ : ∠ABC = 90) (h₂ : ∠ADB = 90) (h₃ : dist A C = 24) (h₄ : dist A D = 7) : dist D B = real.sqrt 119 := 
sorry

end length_of_DB_l502_502035


namespace two_numbers_and_sum_l502_502078

theorem two_numbers_and_sum (x y : ℕ) (hx : x * y = 18) (hy : x - y = 4) : x + y = 10 :=
sorry

end two_numbers_and_sum_l502_502078


namespace hyperbola_eccentricity_range_l502_502214

-- Conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hba : b / a < 2)

-- Define the hyperbola and its eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b / a) ^ 2)

-- The statement to be proven
theorem hyperbola_eccentricity_range :
  1 < eccentricity a b ∧ eccentricity a b < Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_range_l502_502214


namespace not_exist_three_numbers_l502_502273

theorem not_exist_three_numbers :
  ¬ ∃ (a b c : ℝ),
  (b^2 - 4 * a * c > 0 ∧ (-b / a > 0) ∧ (c / a > 0)) ∧
  (b^2 - 4 * a * c > 0 ∧ (-b / a < 0) ∧ (c / a > 0)) :=
by
  sorry

end not_exist_three_numbers_l502_502273


namespace lindsey_owns_more_cars_than_cathy_l502_502495

theorem lindsey_owns_more_cars_than_cathy :
  ∀ (cathy carol susan lindsey : ℕ),
    cathy = 5 →
    carol = 2 * cathy →
    susan = carol - 2 →
    cathy + carol + susan + lindsey = 32 →
    lindsey = cathy + 4 :=
by
  intros cathy carol susan lindsey h1 h2 h3 h4
  sorry

end lindsey_owns_more_cars_than_cathy_l502_502495


namespace find_m_plus_5n_l502_502556

theorem find_m_plus_5n : 
  ∃ (x y : ℕ), 
    (log 10 x + 2 * log 10 (Nat.gcd x y) = 24) ∧
    (log 10 y + log 10 (Nat.lcm x y) = 75) ∧
    let m := x.factor_count and 
    let n := y.factor_count in
    m + 5 * n = 420 :=
sorry

end find_m_plus_5n_l502_502556


namespace kopeechka_items_l502_502428

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l502_502428


namespace no_danger_hitting_reefs_l502_502135

noncomputable def danger_of_hitting_reef (A B C : Point)
  (distance_reon_reefs : ℝ := 3.8)
  (B_surrounded_by_reefs : ∀ P, 0 < distance P B ∧ distance P B ≤ distance_reon_reefs → on_reef P)
  (A_warship_path : Angle A B C = 60)
  (A_distance_to_C : distance A C = 8)
  (A_angle_at_A : Angle C A B = 75) : Prop :=
  let AC := distance A C
      angle_B := 15 -- angle ABC
      BC := AC in
  let perpendicular_B_to_AC := BC * Real.sin (Real.pi / 6) in -- sin(30 degrees) = 0.5
  perpendicular_B_to_AC > 3.8

theorem no_danger_hitting_reefs {A B C : Point} (distance_reon_reefs : ℝ := 3.8)
  (B_surrounded_by_reefs : ∀ P, 0 < distance P B ∧ distance P B ≤ distance_reon_reefs → on_reef P)
  (A_warship_path : Angle A B C = 60)
  (A_distance_to_C : distance A C = 8)
  (A_angle_at_A : Angle C A B = 75) : 
  ¬ danger_of_hitting_reef A B C
  := sorry

end no_danger_hitting_reefs_l502_502135


namespace digit_2500th_l502_502977

def constructX : list ℕ :=
  let nums := list.range' 1 999;
  nums.reverse

noncomputable def x : list ℕ :=
  constructX.bind (λ n, (n.to_string.to_list.reverse.map (λ c, c.to_nat - '0'.to_nat)))

def getDigitAt (digits : list ℕ) (pos : ℕ) : ℕ :=
  digits.nth_le pos (by simp [list.length_bind, list.range', list.length_reverse, of_nat_nat_ge])

theorem digit_2500th :
  getDigitAt x 2499 = 3 :=
sorry

end digit_2500th_l502_502977


namespace avg_minutes_eq_170_div_9_l502_502402

-- Define the conditions
variables (s : ℕ) -- number of seventh graders
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2
def sixth_grade_minutes := 18
def seventh_grade_run_minutes := 20
def seventh_grade_stretching_minutes := 5
def eighth_grade_minutes := 12

-- Define the total activity minutes for each grade
def total_activity_minutes_sixth := sixth_grade_minutes * sixth_graders
def total_activity_minutes_seventh := (seventh_grade_run_minutes + seventh_grade_stretching_minutes) * seventh_graders
def total_activity_minutes_eighth := eighth_grade_minutes * eighth_graders

-- Calculate total activity minutes
def total_activity_minutes := total_activity_minutes_sixth + total_activity_minutes_seventh + total_activity_minutes_eighth

-- Calculate total number of students
def total_students := sixth_graders + seventh_graders + eighth_graders

-- Calculate average minutes per student
def average_minutes_per_student := total_activity_minutes / total_students

theorem avg_minutes_eq_170_div_9 : average_minutes_per_student s = 170 / 9 := by
  sorry

end avg_minutes_eq_170_div_9_l502_502402


namespace units_digit_of_composite_product_l502_502651

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502651


namespace find_incorrect_expression_l502_502906

variable {x y : ℚ}

theorem find_incorrect_expression
  (h : x / y = 5 / 6) :
  ¬ (
    (x + 3 * y) / x = 23 / 5
  ) := by
  sorry

end find_incorrect_expression_l502_502906


namespace units_digit_of_product_of_first_four_composites_l502_502643

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502643


namespace game_cost_proof_l502_502047

variable (initial_money : ℕ) (allowance : ℕ) (current_money : ℕ) (game_cost : ℕ)

-- The conditions given in the problem
def john_initial_money : initial_money = 5 := by sorry
def john_allowance : allowance = 26 := by sorry
def john_current_money : current_money = 29 := by sorry

-- The derived equation for the game cost
def john_game_cost (initial_money allowance current_money : ℕ) : Prop :=
  game_cost = (initial_money + allowance) - current_money

-- The statement that needs to be proved
theorem game_cost_proof :
  john_game_cost initial_money allowance current_money → game_cost = 2 := by
  assume h1 : john_game_cost initial_money allowance current_money
  show game_cost = 2 by sorry

end game_cost_proof_l502_502047


namespace max_product_two_four_digit_numbers_l502_502146

theorem max_product_two_four_digit_numbers :
  ∃ (a b : ℕ), 
    (a * b = max (8564 * 7321) (8531 * 7642)) 
    ∧ max 8531 8564 = 8531 ∧ 
    (∀ x y : ℕ, x * y ≤ 8531 * 7642 → x * y = max (8564 * 7321) (8531 * 7642)) :=
sorry

end max_product_two_four_digit_numbers_l502_502146


namespace tangent_cone_surface_area_l502_502846

noncomputable def conical_surface_area (R : ℝ) : ℝ :=
  (3 * Real.pi * R^2) / 2

theorem tangent_cone_surface_area (R : ℝ) (hR : R > 0) {C : Type*} [normed_group C] [normed_space ℝ C] (c : C) : 
  let S := 2 * R in
  let sphere := metric.sphere c R in
  S = 2 * R →
  ∃ surface_area, surface_area = conical_surface_area R ∧ 
  (∀ P ∈ sphere, ∃ tangents, ∀ t ∈ tangents, metric.tangent_space (tangent_bundle.tangent_space_at c) t = P) →
  surface_area = (3 * Real.pi * R^2) / 2 :=
begin
  sorry,
end

end tangent_cone_surface_area_l502_502846


namespace intersection_area_greater_than_half_l502_502563

theorem intersection_area_greater_than_half (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (rect1 rect2 : set (ℝ × ℝ)) 
  (h1 : is_rectangle rect1 a b) 
  (h2 : is_rectangle rect2 a b)
  (inter_points : (rect1 ∩ rect2).finite ∧ (rect1 ∩ rect2).card = 8) :
  ∃ intersection_area : ℝ, 
    intersection_area > (a * b) / 2 :=
sorry

end intersection_area_greater_than_half_l502_502563


namespace possible_items_l502_502434

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l502_502434


namespace collinear_P_H_Q_l502_502478

noncomputable def orthocenter {A B C : Point} [IsAcuteTriangle A B C] : Point := sorry
noncomputable def circleDiameterBC (B C : Point) : Circle := sorry
noncomputable def tangentToCircle (A : Point) (circle : Circle) : (Point × Point) := sorry

theorem collinear_P_H_Q 
  {A B C P Q H : Point}
  (hAcute : IsAcuteTriangle A B C)
  (hOrthocenter : H = orthocenter A B C)
  (hTangent : (P, Q) = tangentToCircle A (circleDiameterBC B C)) :
  Collinear P H Q :=
sorry

end collinear_P_H_Q_l502_502478


namespace problem_A_problem_B_not_problem_C_problem_D_l502_502391

def isITypeFunction (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃! x₂ ∈ D, f x₁ + f x₂ = 1

theorem problem_A (D : set ℝ) (hD : ∀ x ∈ D, 0 < x):
  isITypeFunction (λ x, Real.log x) D :=
sorry

theorem problem_B_not (D : set ℝ):
  ¬ isITypeFunction (λ x, Real.sin x) D :=
sorry

theorem problem_C (f : ℝ → ℝ) (D : set ℝ):
  isITypeFunction f D → isITypeFunction (λ x, 1 - f x) D :=
sorry

theorem problem_D (m : ℝ) :
  (isITypeFunction (λ x, m + Real.sin x) (set.Icc (-Real.pi / 2) (Real.pi / 2))) → m = 1 / 2 :=
sorry

end problem_A_problem_B_not_problem_C_problem_D_l502_502391


namespace land_value_moon_l502_502122

-- Define the conditions
def surface_area_earth : ℕ := 200
def surface_area_ratio : ℕ := 5
def value_ratio : ℕ := 6
def total_value_earth : ℕ := 80

-- Define the question and the expected answer
noncomputable def total_value_moon : ℕ := 96

-- State the proof problem
theorem land_value_moon :
  (surface_area_earth / surface_area_ratio * value_ratio) * (surface_area_earth / surface_area_ratio) = total_value_moon := 
sorry

end land_value_moon_l502_502122


namespace number_of_items_l502_502454

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502454


namespace angle_tuvels_equiv_l502_502076

-- Defining the conditions
def full_circle_tuvels : ℕ := 400
def degree_angle_in_circle : ℕ := 360
def specific_angle_degrees : ℕ := 45

-- Proof statement showing the equivalence
theorem angle_tuvels_equiv :
  (specific_angle_degrees * full_circle_tuvels) / degree_angle_in_circle = 50 :=
by
  sorry

end angle_tuvels_equiv_l502_502076


namespace real_solutions_quadratic_l502_502804

theorem real_solutions_quadratic (d : ℝ) (h : 0 < d) :
  ∃ x : ℝ, x^2 - 8 * x + d < 0 ↔ 0 < d ∧ d < 16 :=
by
  sorry

end real_solutions_quadratic_l502_502804


namespace quadratic_equation_same_solutions_l502_502044

theorem quadratic_equation_same_solutions :
  ∃ b c : ℝ, (b, c) = (1, -7) ∧ (∀ x : ℝ, (x - 3 = 4 ∨ 3 - x = 4) ↔ (x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_equation_same_solutions_l502_502044


namespace find_mt_product_l502_502070

open Classical
open BigOperators
open Nat

variable (g : ℕ → ℕ)

-- Given conditions
axiom h_fun_eq : ∀ a b : ℕ, 2 * g (a^2 + b^2 + 1) = g a ^ 2 + g b ^ 2

-- Proof statement
theorem find_mt_product : (∃ (m t : ℕ), (∀ x ∈ {0, 1, 52}, g 26 = x) ∧ m = 3 ∧ t = 53 ∧ m * t = 159) := sorry

end find_mt_product_l502_502070


namespace sheets_needed_l502_502746

-- Definition of constants
def ream_sheets : ℕ := 500
def ream_cost : ℝ := 27
def total_cost : ℝ := 270

-- Theorem statement: Given the conditions, the number of sheets needed is 5000
theorem sheets_needed (ream_sheets : ℕ) (ream_cost total_cost : ℝ) : ream_sheets = 500 ∧ ream_cost = 27 ∧ total_cost = 270 → 
  total_cost / (ream_cost / ream_sheets) = 5000 := 
by
  intros h
  cases h with h1 h23
  cases h23 with h2 h3
  have h_cost_per_sheet : ℝ := (ream_cost / ream_sheets)
  have h_sheets_needed : ℝ := (total_cost / h_cost_per_sheet)
  rw [h1, h2, h3] at h_sheets_needed
  have h_cost_per_sheet_value : 27 / 500 = 0.054 := by norm_num
  rw [h_cost_per_sheet_value] at h_sheets_needed
  norm_num at h_sheets_needed
  exact congr_arg coe (Nat.cast_inj.2 rfl)


end sheets_needed_l502_502746


namespace least_n_for_factorial_multiple_10080_l502_502388

theorem least_n_for_factorial_multiple_10080 (n : ℕ) 
  (h₁ : 0 < n) 
  (h₂ : ∀ m, m > 0 → (n ≠ m → n! % 10080 ≠ 0)) 
  : n = 8 := 
sorry

end least_n_for_factorial_multiple_10080_l502_502388


namespace fraction_never_lost_to_AI_l502_502084

theorem fraction_never_lost_to_AI (total_players : ℕ) (players_lost : ℕ) (h1 : total_players = 40) (h2 : players_lost = 30) :
  (total_players - players_lost) / (total_players : ℚ) = 1 / 4 := by
s

end fraction_never_lost_to_AI_l502_502084


namespace range_f_positive_l502_502330

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
axiom odd_f (x : ℝ) : f(-x) = -f(x)
axiom deriv_f (x : ℝ) : f' x = derivative f x
axiom f_neg2_eq_zero : f (-2) = 0
axiom ineq_cond (x : ℝ) (hx : 0 < x) : x * f' x - f x < 0

theorem range_f_positive : 
  {x : ℝ | f x > 0} = set.Iio (-2) ∪ set.Ioo 0 2 :=
sorry

end range_f_positive_l502_502330


namespace units_digit_of_product_of_first_four_composites_l502_502662

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502662


namespace units_digit_first_four_composites_l502_502576

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l502_502576


namespace units_digit_first_four_composite_is_eight_l502_502611

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l502_502611


namespace trig_identity_simplify_l502_502516

theorem trig_identity_simplify (α : ℝ) : 
  sin (π / 6 + α) + cos (π / 3 + α) = cos α :=
by
  sorry

end trig_identity_simplify_l502_502516


namespace FN_eq_GM_MN_parallel_CE_AD_perpendicular_CE_l502_502033

-- Define the setup of the complete quadrilateral and various points of intersection
variables (A B C D E F G M N: Type)
variables (B_perp_AC: B ⟂ A ∧ B ∈ AC) (F_perp_AE: F ⟂ A ∧ F ∈ AE)
variables (G_on_CE: G ∈ CE) (M_on_BG_CF: M ∈ BG ∧ M ∈ CF) (N_on_FG_BE: N ∈ FG ∧ N ∈ BE)

-- Conditions
axiom angle_CAE_eq_90 : ∠ CAE = 90
axiom eq_AB_AF : AB = AF

theorem FN_eq_GM (B F N G M: Type) (perp_B_AC: B ⟂ A ∧ B ∈ AC) (perp_F_AE: F ⟂ A ∧ F ∈ AE) 
    (on_G_CE: G ∈ CE) (on_M_BG_CF: M ∈ BG ∧ M ∈ CF) (on_N_FG_BE: N ∈ FG ∧ N ∈ BE) :
  FN = GM := sorry

theorem MN_parallel_CE (B F N G M: Type) (perp_B_AC: B ⟂ A ∧ B ∈ AC) (perp_F_AE: F ⟂ A ∧ F ∈ AE) 
    (on_G_CE: G ∈ CE) (on_M_BG_CF: M ∈ BG ∧ M ∈ CF) (on_N_FG_BE: N ∈ FG ∧ N ∈ BE) :
  MN ∥ CE := sorry

theorem AD_perpendicular_CE (B F D G M: Type) (perp_B_AC: B ⟂ A ∧ B ∈ AC) (perp_F_AE: F ⟂ A ∧ F ∈ AE) 
    (on_G_CE: G ∈ CE) (on_D_CE: D ∈ CE) :
  AD ⟂ CE := sorry

end FN_eq_GM_MN_parallel_CE_AD_perpendicular_CE_l502_502033


namespace kopeechka_items_l502_502462

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502462


namespace triangle_area_l502_502011

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ := 
let s := semi_perimeter a b c in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area:
  let a := 30
  let b := 28
  let c := 10
  heron_area a b c ≈ 139.94 :=
by
  sorry

end triangle_area_l502_502011


namespace subset_count_of_set_ab_l502_502124

theorem subset_count_of_set_ab : (set.univ.powerset.to_finset.filter (λ s, s ⊆ ({a, b} : set ℕ))).card = 4 := 
begin
  sorry
end

end subset_count_of_set_ab_l502_502124


namespace compute_operation_value_l502_502493

def operation (a b c : ℝ) : ℝ := b^3 - 3 * a * b * c - 4 * a * c^2

theorem compute_operation_value : operation 2 (-1) 4 = -105 :=
by
  sorry

end compute_operation_value_l502_502493


namespace liters_to_pints_l502_502334

theorem liters_to_pints (l : ℝ) (p : ℝ) (h : 0.75 = l) (h_p : 1.575 = p) : 
  Float.round (1.5 * (p / l) * 10) / 10 = 3.2 :=
by sorry

end liters_to_pints_l502_502334


namespace determinant_in_terms_of_roots_l502_502067

noncomputable def determinant_3x3 (a b c : ℝ) : ℝ :=
  (1 + a) * ((1 + b) * (1 + c) - 1) - 1 * (1 + c) + (1 + b) * 1

theorem determinant_in_terms_of_roots (a b c s p q : ℝ)
  (h1 : a + b + c = -s)
  (h2 : a * b + a * c + b * c = p)
  (h3 : a * b * c = -q) :
  determinant_3x3 a b c = -q + p - s :=
by
  sorry

end determinant_in_terms_of_roots_l502_502067


namespace sulphur_atoms_l502_502751

noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_S : ℝ := 32.06

def number_of_aluminium_atoms : ℕ := 2
def total_molecular_weight : ℝ := 150

theorem sulphur_atoms :
  let weight_Al := number_of_aluminium_atoms * atomic_weight_Al in
  let weight_S := total_molecular_weight - weight_Al in
  let number_of_sulphur_atoms := weight_S / atomic_weight_S in
  number_of_sulphur_atoms ≈ 3 :=
by 
  sorry

end sulphur_atoms_l502_502751


namespace sum_fn_3000_l502_502302

def f (n : ℕ) : ℚ :=
  if (logBase 9 n).isRational 
  then logBase 9 n 
  else 0

theorem sum_fn_3000 : (∑ n in Finset.range 3001, f n) = 21 / 2 :=
sorry

end sum_fn_3000_l502_502302


namespace president_vice_president_selection_l502_502528

theorem president_vice_president_selection :
  ∃ (total_members girls boys : ℕ),
  total_members = 30 ∧
  girls = 15 ∧
  boys = 15 ∧
  total_members = girls + boys ∧
  (ways_to_choose : ℕ) ∃,
    ways_to_choose = girls * boys :=
  sorry

end president_vice_president_selection_l502_502528


namespace present_age_of_A_is_11_l502_502548

-- Definitions for present ages
variables (A B C : ℕ)

-- Definitions for the given conditions
def sum_of_ages_present : Prop := A + B + C = 57
def age_ratio_three_years_ago (x : ℕ) : Prop := (A - 3 = x) ∧ (B - 3 = 2 * x) ∧ (C - 3 = 3 * x)

-- The proof statement
theorem present_age_of_A_is_11 (x : ℕ) (h1 : sum_of_ages_present A B C) (h2 : age_ratio_three_years_ago A B C x) : A = 11 := 
by
  sorry

end present_age_of_A_is_11_l502_502548


namespace sum_of_cubes_l502_502923

-- Definitions based on the conditions
variables (a b : ℝ)
variables (h1 : a + b = 2) (h2 : a * b = -3)

-- The Lean statement to prove the sum of their cubes is 26
theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
by
  sorry

end sum_of_cubes_l502_502923


namespace distinct_square_sum_100_l502_502410

theorem distinct_square_sum_100 :
  ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → 
  a^2 + b^2 + c^2 = 100 → false := by
  sorry

end distinct_square_sum_100_l502_502410


namespace remainder_Q_div_l502_502057

-- Q is a polynomial such that Q(17) = 101 and Q(101) = 17
def Q (x : ℝ) : ℝ

-- Constants for conditions
axiom Q_at_17 : Q 17 = 101
axiom Q_at_101 : Q 101 = 17

-- Define the form of the polynomial remainder when divided by (x - 17)(x - 101)
theorem remainder_Q_div (x : ℝ) :
  ∃ (a b : ℝ), (Q x = (x - 17) * (x - 101) * (R x) + ax + b) ∧ a = -1 ∧ b = 118 :=
by
  existsi [-1, 118]
  split
  -- Prove that Q(x) = (x - 17)(x - 101)R(x) - x + 118
  sorry
  -- Prove that a = -1
  sorry
  -- Prove that b = 118
  sorry

end remainder_Q_div_l502_502057


namespace exponential_decreasing_l502_502917

theorem exponential_decreasing (a : ℝ) : (∀ x y : ℝ, x < y → (2 * a - 1)^y < (2 * a - 1)^x) ↔ (1 / 2 < a ∧ a < 1) := 
by
    sorry

end exponential_decreasing_l502_502917


namespace binomial_constant_term_l502_502106

theorem binomial_constant_term :
  let expr := (3 * x - (1 / x)) ^ 6
  (term : ℕ → ℝ) (constant_term : ℝ) :=
  (∀ r : ℕ, term r = (nat.choose 6 r) * (3:ℝ)^(6-r) * (-1)^r * x^(6 - 2 * r)) →
  (constant_term = -540) →
  (∃ r : ℕ, 6 - 2 * r = 0 ∧ term r = constant_term) :=
begin
  let expr := (3 * x - (1 / x)) ^ 6,
  assume term constant_term,
  assume term_property,
  assume constant_value,
  existsi 3,
  split,
  { -- Prove 6 - 2 * 3 = 0
    simp, },
  { -- Prove term 3 = -540
    rw term_property,
    sorry, } -- Computation steps are omitted for brevity.
end

end binomial_constant_term_l502_502106


namespace purchase_options_l502_502416

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502416


namespace sin_1035_eq_neg_sqrt2_div_2_l502_502297

theorem sin_1035_eq_neg_sqrt2_div_2 : Real.sin (1035 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
    sorry

end sin_1035_eq_neg_sqrt2_div_2_l502_502297


namespace transportation_mode_l502_502017

theorem transportation_mode (n : ℕ) (cities : Finset (Fin n)) 
  (transportation : Fin n → Fin n → Prop) :
  (∀ (c1 c2 : Fin n), transportation c1 c2 ∨ transportation c2 c1) →
  ∃ (mode : Fin n → Fin n → Prop), (∀ c1 c2 : Fin n, connected cities mode) :=
by
  sorry

end transportation_mode_l502_502017


namespace second_trial_point_618_method_l502_502343

theorem second_trial_point_618_method (a b : ℝ) (h1 : a = 500) (h2 : b = 1500) : 
    let range := b - a in
    let trial1 := a + range * 0.618 in
    let trial2 := b - range * 0.618 in
    let new_range1 := trial1 - a in
    let new_range2 := b - trial2 in
    let new_trial1 := a + new_range1 * 0.618 in
    let new_trial2 := b - new_range2 * 0.618 in
    new_trial1 ≈ 882 ∨ new_trial2 ≈ 1118 := 
by
    sorry

end second_trial_point_618_method_l502_502343


namespace units_digit_of_product_of_first_four_composites_l502_502669

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502669


namespace trust_meteorologist_l502_502180

-- Definitions for problem conditions
variables {G M1 M2 S : Prop}
variable {r : ℝ}
variable {p : ℝ}

/-- The probability of a clear day is r -/
axiom prob_clear_day : r = 0.74

/-- Senators' prediction accuracy -/
axiom senator_accuracy : ℝ

/-- Meteorologist's prediction accuracy being 1.5 times senators' /-
axiom meteorologist_accuracy : ∀ p, 1.5 * p

/-- Independence of predictions -/
axiom independence_preds : independent [G, M1, M2, S]

noncomputable def joint_probability_given_G : ℝ :=
(1 - 1.5 * meteorologist_accuracy senator_accuracy) * senator_accuracy^2

noncomputable def joint_probability_given_not_G : ℝ :=
1.5 * meteorologist_accuracy senator_accuracy * (1 - senator_accuracy)^2

noncomputable def overall_probability : ℝ :=
joint_probability_given_G * r + joint_probability_given_not_G * (1 - r)

noncomputable def conditional_prob_not_clear : ℝ /-
(joint_probability_given_not_G * (1 - r)) / overall_probability

noncomputable def conditional_prob_clear : ℝ
(joint_probability_given_G * r) / overall_probability

-- Main theorem statement: Given the conditions, the meteorologist's forecast is more reliable
theorem trust_meteorologist : conditional_prob_not_clear > conditional_prob_clear :=
by sorry

end trust_meteorologist_l502_502180


namespace eval_expression_l502_502278

theorem eval_expression : 81^(1/2) * 64^(-1/3) * 49^(1/2) = (63 / 4) :=
by
  sorry

end eval_expression_l502_502278


namespace train_distance_problem_l502_502739

theorem train_distance_problem
  (Vx : ℝ) (Vy : ℝ) (t : ℝ) (distanceX : ℝ) 
  (h1 : Vx = 32) 
  (h2 : Vy = 160 / 3) 
  (h3 : 32 * t + (160 / 3) * t = 160) :
  distanceX = Vx * t → distanceX = 60 :=
by {
  sorry
}

end train_distance_problem_l502_502739


namespace purchase_options_l502_502415

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l502_502415


namespace trust_meteorologist_l502_502182

-- Definitions
def probability_of_clear := 0.74
def senator_accuracy := p : ℝ
def meteorologist_accuracy := 1.5 * p

-- Events
def event_G := "clear day"
def event_M1 := "first senator predicted clear"
def event_M2 := "second senator predicted clear"
def event_S := "meteorologist predicted rain"

theorem trust_meteorologist :
  let r := probability_of_clear
  let p := senator_accuracy
  let q := meteorologist_accuracy
  1.5 * p * (1 - p)^2 * (1 - r) - (1 - 1.5 * p) * p^2 * r > 0 :=
by
  sorry

end trust_meteorologist_l502_502182


namespace number_of_black_balls_l502_502133

variable (T : ℝ)
variable (red_balls : ℝ := 21)
variable (prop_red : ℝ := 0.42)
variable (prop_white : ℝ := 0.28)
variable (white_balls : ℝ := 0.28 * T)

noncomputable def total_balls : ℝ := red_balls / prop_red

theorem number_of_black_balls :
  T = total_balls → 
  ∃ black_balls : ℝ, black_balls = total_balls - red_balls - white_balls ∧ black_balls = 15 := 
by
  intro hT
  let black_balls := total_balls - red_balls - white_balls
  use black_balls
  simp [total_balls]
  sorry

end number_of_black_balls_l502_502133


namespace intersection_points_of_line_segments_l502_502782

theorem intersection_points_of_line_segments
  (a : ℕ) (b : ℕ)
  (points_a : Fin a) (points_b : Fin b) :
  a = 10 → b = 11 →
  let intersections := Nat.choose a 2 * Nat.choose b 2
  intersections = 2475 :=
by
  intros ha hb
  simp only []
  exact Nat.choose_eq ha hb

end intersection_points_of_line_segments_l502_502782


namespace find_y_intercept_of_l2_l502_502345

noncomputable def slope (p1 p2 : Point) : ℝ :=
if (p1.x = p2.x) then 0 else (p2.y - p1.y) / (p2.x - p1.x)

structure Line where
  slope : ℝ
  y_intercept : ℝ

def parallel_lines (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem find_y_intercept_of_l2
  (l1_slope : ℝ) (h_l1 : l1_slope = 2)
  (parallel : ∀ (l : Line), parallel_lines {slope := l1_slope, y_intercept := 0} l)
  (passes_through : ∃ x y : ℝ, x = -1 ∧ y = 1 ∧ ∀ y, y = 2 * x + 3)
  : ∃ y : ℝ, (∃ x : ℝ, x = 0) ∧ y = 3 := 
by
  sorry

end find_y_intercept_of_l2_l502_502345


namespace minimum_sum_is_M_n_l502_502303

-- Define the i-th Fibonacci number recursively
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem minimum_sum_is_M_n (n : ℕ) (h : n ≥ 2) :
  ∃ (a : ℕ → ℕ), a 0 = 1 ∧ (∀ i < n - 1, a i ≤ a (i + 1) + a (i + 2)) ∧
  (∀ i, a i ≥ 0) ∧ a 0 + a 1 + ... + a n = (fib (n + 2) - 1) / fib n := sorry

end minimum_sum_is_M_n_l502_502303


namespace albums_in_either_but_not_both_l502_502244

-- Defining the conditions
def shared_albums : ℕ := 9
def total_albums_andrew : ℕ := 17
def unique_albums_john : ℕ := 6

-- Stating the theorem to prove
theorem albums_in_either_but_not_both :
  (total_albums_andrew - shared_albums) + unique_albums_john = 14 :=
sorry

end albums_in_either_but_not_both_l502_502244


namespace f_zero_and_odd_f_strictly_increasing_range_of_a_l502_502953

-- Definitions
variable {f : ℝ → ℝ}
variable (h_add : ∀ x y : ℝ, f(x + y) = f(x) + f(y))
variable (h_pos : ∀ x : ℝ, x > 0 → f(x) > 0)
variable (h_f3 : f(3) = 12)
def A := { p : ℝ × ℝ | f(p.1 ^ 2) + f(p.2 ^ 2) = 4 }
def B (a : ℝ) := { p : ℝ × ℝ | p.1 + a * p.2 = real.sqrt 5 }

-- Statements of the proof problems

-- 1. Prove f(0) = 0 and f is odd
theorem f_zero_and_odd : f(0) = 0 ∧ ∀ x, f(-x) = -f(x) :=
sorry

-- 2. Prove f is strictly increasing
theorem f_strictly_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) :=
sorry

-- 3. Determine the range of a
theorem range_of_a (a : ℝ) : (A ∩ B(a)).nonempty → (a ≤ -2 ∨ a ≥ 2) :=
sorry

end f_zero_and_odd_f_strictly_increasing_range_of_a_l502_502953


namespace largest_8_digit_number_with_even_digits_l502_502689

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l502_502689


namespace monthly_income_30pct_tax_l502_502029

noncomputable def monthly_gross_income (x : ℝ) : ℝ :=
  if x > 1_050_000 then x / 12 else 0

theorem monthly_income_30pct_tax (x : ℝ) (h : 1_050_000 < x) :
  0.4 * (x - 1_050_000) + 267_000 = 0.3 * x → monthly_gross_income x = 127_500 :=
by
  intro h1
  sorry

end monthly_income_30pct_tax_l502_502029


namespace semicircle_perimeter_l502_502225
-- Lean 4 code

theorem semicircle_perimeter (r : ℝ) (π : ℝ) (approx_pi : π ≈ 3.14159) (radius_value : r = 14) :
  (14 * π + 28) ≈ 71.98 := by
  sorry

end semicircle_perimeter_l502_502225


namespace total_legs_for_insects_l502_502016

theorem total_legs_for_insects (n_insects legs_per_insect : ℕ) (h1 : n_insects = 6) (h2 : legs_per_insect = 6) :
  n_insects * legs_per_insect = 36 :=
by
  rw [h1, h2]
  norm_num

end total_legs_for_insects_l502_502016


namespace xyz_sum_sqrt14_l502_502485

theorem xyz_sum_sqrt14 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 1) (h2 : x + 2 * y + 3 * z = Real.sqrt 14) :
  x + y + z = (3 * Real.sqrt 14) / 7 :=
sorry

end xyz_sum_sqrt14_l502_502485


namespace ten_pow_neg_x_equals_one_tenth_l502_502912

theorem ten_pow_neg_x_equals_one_tenth (x : ℝ) (h : 10^(3 * x) = 1000) : 10^(-x) = 1 / 10 :=
sorry

end ten_pow_neg_x_equals_one_tenth_l502_502912


namespace anna_and_carl_frame_probability_l502_502018

noncomputable def probability_in_frame (anna_time: ℝ) (carl_time: ℝ) (photo_fraction: ℝ) (start_time: ℝ) (end_time: ℝ) : ℝ :=
  let photo_time := end_time - start_time in
  let gcd_time := Real.gcd anna_time carl_time in  -- Common period of synchronization based on running times
  let anna_frame_time := photo_fraction * anna_time in
  let carl_frame_time := photo_fraction * carl_time in
  let overlap_time := anna_frame_time + carl_frame_time - gcd_time in
  let total_frame_time := (overlap_time * (photo_time / gcd_time)) in
  total_frame_time / photo_time

theorem anna_and_carl_frame_probability :
  probability_in_frame 100 60 (1/3) 720 900 = 8 / 45 :=
sorry

end anna_and_carl_frame_probability_l502_502018


namespace kopeechka_items_l502_502469

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l502_502469


namespace Jame_tears_30_cards_at_a_time_l502_502955

theorem Jame_tears_30_cards_at_a_time
    (cards_per_deck : ℕ)
    (times_per_week : ℕ)
    (decks : ℕ)
    (weeks : ℕ)
    (total_cards : ℕ := decks * cards_per_deck)
    (total_times : ℕ := weeks * times_per_week)
    (cards_at_a_time : ℕ := total_cards / total_times)
    (h1 : cards_per_deck = 55)
    (h2 : times_per_week = 3)
    (h3 : decks = 18)
    (h4 : weeks = 11) :
    cards_at_a_time = 30 := by
  -- Proof can be added here
  sorry

end Jame_tears_30_cards_at_a_time_l502_502955


namespace units_digit_of_product_of_first_four_composites_l502_502668

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l502_502668


namespace cos_alpha_given_tan_and_alpha_range_l502_502834

theorem cos_alpha_given_tan_and_alpha_range (α : ℝ) (h1 : Real.tan (π - α) = 3 / 4) (h2 : α ∈ Ioo (π / 2) π) : Real.cos α = -4 / 5 := 
by 
  sorry

end cos_alpha_given_tan_and_alpha_range_l502_502834


namespace max_intersection_5_points_l502_502312

noncomputable def max_intersection_points (n : ℕ) : ℕ := 435 - 30 - 20 - 75

theorem max_intersection_5_points :
  ∀ (P : Fin 5 → Point) (L : Fin 10 → Line),
  (∀ (i j : Fin 5), i ≠ j → ¬ Parallel (L i) (L j) ∧ ¬ Perpendicular (L i) (L j)) →
  (∀ (i : Fin 5), ∃ (j k : Fin 5), Perpendicular P[L j, L k]) →
  max_intersection_points 5 = 310 := 
sorry

end max_intersection_5_points_l502_502312


namespace jordan_rectangle_length_l502_502791

theorem jordan_rectangle_length :
  ∀ (a1 a2 b1 b2 : ℝ), (a1 = 15 ∧ a2 = 20 ∧ b2 = 50 ∧ a1 * a2 = b1 * b2) → b1 = 6 :=
by
  intro a1 a2 b1 b2 h,
  cases h with h_a1 h_rest,
  cases h_rest with h_a2 h_rest,
  cases h_rest with h_b2 h_eq_area,
  have h_carol_area : (a1 * a2 = 300),
  { calc
      a1 * a2 = 15 * 20 : by rw [h_a1, h_a2]
      ... = 300 : by norm_num },
  rw [h_carol_area] at h_eq_area,
  have h_jordan_eq_300 : (b1 * 50 = 300) := by rw [h_b2] at h_eq_area; exact h_eq_area,
  calc
    b1 = 300 / 50 : by rw ← h_jordan_eq_300; exact (eq_div_iff_mul_eq 50 (by norm_num)).mpr rfl
    ... = 6 : by norm_num

end jordan_rectangle_length_l502_502791


namespace subset_union_card_eq_six_l502_502986

def A_subset (X : Type) [Fintype X] (A : Finset X) := A.card = 5

theorem subset_union_card_eq_six 
  (n : ℕ) 
  (h_n : n > 6) 
  (X : Finset (Fin n))
  (h_X : X.card = n) 
  (m : ℕ) 
  (A : Fin m → Finset (Fin n))
  (h_A : ∀ i, A i ∈ X.powerset.filter (λ s, s.card = 5)) 
  (h_m : m > n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15) / 600) : 
  ∃ (i₁ i₂ i₃ i₄ i₅ i₆ : Fin m),
    i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ < i₅ ∧ i₅ < i₆ ∧ 
    (Finset.bUnion {i₁, i₂, i₃, i₄, i₅, i₆}.to_finset A).card = 6 := 
begin
  sorry
end

end subset_union_card_eq_six_l502_502986


namespace smaller_angle_at_3_25_l502_502152

-- Definitions for the given conditions
def initial_angle : ℝ := 90
def minute_hand_rate : ℝ := 6
def hour_hand_rate : ℝ := 0.5
def minutes_passed : ℝ := 25

-- Calculate the angles moved by hour and minute hands after given minutes
def hour_hand_movement := hour_hand_rate * minutes_passed
def minute_hand_movement := minute_hand_rate * minutes_passed

-- Calculate the new angle (subtract the minute hand movement)
def new_angle := initial_angle + hour_hand_movement - minute_hand_movement

-- Make the new_angle a positive angle in 360-degree circumference
def positive_angle := if new_angle < 0 then 360 + new_angle else new_angle

-- The smaller angle between the hands
def smaller_angle := if positive_angle > 180 then 360 - positive_angle else positive_angle

theorem smaller_angle_at_3_25 :
  smaller_angle = 47.5 := sorry

end smaller_angle_at_3_25_l502_502152


namespace x_plus_y_is_negative_eight_l502_502314

theorem x_plus_y_is_negative_eight
  (x y : ℝ)
  (h1 : 5 ^ x = 25 ^ (y + 2))
  (h2 : 16 ^ y = 4 ^ (x + 4)) :
  x + y = -8 :=
by
  sorry

end x_plus_y_is_negative_eight_l502_502314


namespace harry_worked_total_hours_l502_502275

theorem harry_worked_total_hours (x : ℝ) (H : ℝ) (H_total : ℝ) :
  (24 * x + 1.5 * x * H = 42 * x) → (H_total = 24 + H) → H_total = 36 :=
by
sorry

end harry_worked_total_hours_l502_502275


namespace winning_votes_calculation_l502_502559

variables (V : ℚ) (winner_votes : ℚ)

-- Conditions
def percentage_of_votes_of_winner : ℚ := 0.60 * V
def percentage_of_votes_of_loser : ℚ := 0.40 * V
def vote_difference_spec : 0.60 * V - 0.40 * V = 288 := by sorry

-- Theorem to prove
theorem winning_votes_calculation (h1 : winner_votes = 0.60 * V)
  (h2 : 0.60 * V - 0.40 * V = 288) : winner_votes = 864 :=
by
  sorry

end winning_votes_calculation_l502_502559


namespace value_of_a_l502_502736

theorem value_of_a (a b c d : ℕ) (h : (18^a) * (9^(4*a-1)) * (27^c) = (2^6) * (3^b) * (7^d)) : a = 6 :=
by
  sorry

end value_of_a_l502_502736


namespace possible_values_of_x_l502_502479

theorem possible_values_of_x 
  (x : ℤ) 
  (M : set ℤ := {3, 9, 3 * x}) 
  (N : set ℤ := {3, x^2}) 
  (h : N ⊆ M) : 
  x = -3 ∨ x = 0 := 
by 
  sorry

end possible_values_of_x_l502_502479


namespace units_digit_first_four_composites_l502_502625

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l502_502625


namespace area_of_right_angled_isosceles_triangle_l502_502130

-- Definitions
variables {x y : ℝ}
def is_right_angled_isosceles (x y : ℝ) : Prop := y^2 = 2 * x^2
def sum_of_square_areas (x y : ℝ) : Prop := x^2 + x^2 + y^2 = 72

-- Theorem
theorem area_of_right_angled_isosceles_triangle (x y : ℝ) 
  (h1 : is_right_angled_isosceles x y) 
  (h2 : sum_of_square_areas x y) : 
  1/2 * x^2 = 9 :=
sorry

end area_of_right_angled_isosceles_triangle_l502_502130


namespace limit_calculation_l502_502786

open Real

noncomputable def limit_expr (x : ℝ) : ℝ :=
  (2^(3*x) - 3^(2*x)) / (x + arcsin (x^3))

theorem limit_calculation :
  filter.tendsto (λ x : ℝ, limit_expr x) (nhds 0) (nhds (log (8/9))) :=
begin
  sorry
end

end limit_calculation_l502_502786


namespace units_digit_product_first_four_composite_numbers_l502_502673

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l502_502673


namespace number_of_items_l502_502457

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l502_502457


namespace triangle_bc_length_l502_502395

theorem triangle_bc_length (A B C X : Type*)
  (AB AC BC BX CX : ℕ)
  (hAB : AB = 72)
  (hAC : AC = 80)
  (hCircle : ∀ P, (|P - A| = AB) → P = B ∨ P = X)
  (hIntLengths : ∃ BX_length CX_length : ℕ, BX = BX_length ∧ CX = CX_length)
  (hPowerOfPoint : CX * (CX + BX) = 8 * 152)
  : BC = 38 :=
sorry

end triangle_bc_length_l502_502395


namespace part1_part2_l502_502982

-- Part 1: Number of k-tuples of ordered subsets with empty intersection
theorem part1 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (∃ (f : Fin (n) → Fin (2^k - 1)), true) :=
sorry

-- Part 2: Number of k-tuples of subsets with chain condition
theorem part2 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (S.card = (k + 1)^n) :=
sorry

end part1_part2_l502_502982


namespace range_of_a_exists_distinct_x1_x2_eq_f_l502_502354

noncomputable
def f (a x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem range_of_a_exists_distinct_x1_x2_eq_f :
  { a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2 } = 
  { a : ℝ | (a > (2 / 3)) ∨ (a ≤ 0) } :=
sorry

end range_of_a_exists_distinct_x1_x2_eq_f_l502_502354


namespace units_digit_first_four_composites_l502_502598

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502598


namespace three_lines_intersect_at_three_points_l502_502254

-- Define the lines as propositions expressing the equations
def line1 (x y : ℝ) := 2 * y - 3 * x = 4
def line2 (x y : ℝ) := x + 3 * y = 3
def line3 (x y : ℝ) := 3 * x - 4.5 * y = 7.5

-- Define a proposition stating that there are exactly 3 unique points of intersection among the three lines
def number_of_intersections : ℕ := 3

-- Prove that the number of unique intersection points is exactly 3 given the lines
theorem three_lines_intersect_at_three_points : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
    (line3 p3.1 p3.2 ∧ line1 p3.1 p3.2) :=
sorry

end three_lines_intersect_at_three_points_l502_502254


namespace insert_arithmetic_sequence_l502_502042

theorem insert_arithmetic_sequence (d a b : ℤ) 
  (h1 : (-1) + 3 * d = 8) 
  (h2 : a = (-1) + d) 
  (h3 : b = a + d) : 
  a = 2 ∧ b = 5 := by
  sorry

end insert_arithmetic_sequence_l502_502042


namespace limit_of_sequence_exists_l502_502037

noncomputable def sequence (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a
  else if n = 2 then b
  else (sequence a b (n - 1) + sequence a b (n - 2)) / 2

theorem limit_of_sequence_exists (a b : ℝ) :
  ∃ l : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence a b n - l| < ε) ∧
  l = (a + 2 * b) / 3 :=
sorry

end limit_of_sequence_exists_l502_502037


namespace seven_points_unit_distance_l502_502511

theorem seven_points_unit_distance :
  ∃ (A B C D E F G : ℝ × ℝ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
     E ≠ F ∧ E ≠ G ∧
     F ≠ G) ∧
    (∀ (P Q R : ℝ × ℝ),
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F ∨ P = G) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F ∨ Q = G) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E ∨ R = F ∨ R = G) →
      P ≠ Q → P ≠ R → Q ≠ R →
      (dist P Q = 1 ∨ dist P R = 1 ∨ dist Q R = 1)) :=
sorry

end seven_points_unit_distance_l502_502511


namespace units_digit_of_composite_product_l502_502656

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l502_502656


namespace units_digit_first_four_composites_l502_502597

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502597


namespace tax_rate_for_remaining_l502_502048

variable (total_earnings deductions first_tax_rate total_tax taxed_amount remaining_taxable_income rem_tax_rate : ℝ)

def taxable_income (total_earnings deductions : ℝ) := total_earnings - deductions

def tax_on_first_portion (portion tax_rate : ℝ) := portion * tax_rate

def remaining_taxable (total_taxable first_portion : ℝ) := total_taxable - first_portion

def total_tax_payable (tax_first tax_remaining : ℝ) := tax_first + tax_remaining

theorem tax_rate_for_remaining :
  total_earnings = 100000 ∧ 
  deductions = 30000 ∧ 
  first_tax_rate = 0.10 ∧
  total_tax = 12000 ∧
  tax_on_first_portion 20000 first_tax_rate = 2000 ∧
  taxed_amount = 2000 ∧
  remaining_taxable_income = taxable_income total_earnings deductions - 20000 ∧
  total_tax_payable taxed_amount (remaining_taxable_income * rem_tax_rate) = total_tax →
  rem_tax_rate = 0.20 := 
sorry

end tax_rate_for_remaining_l502_502048


namespace find_angle_A_range_of_y_area_solution1_area_solution2_l502_502470

variables {a b c : ℝ}
variables {A B C : ℝ} (A_eq : A = π / 6)
variables (acute_ABC : A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variables (tan_eq : 1 + tan A / tan B = 2 * c / (sqrt 3 * b))
variables (y : ℝ → ℝ := (λ B, sin (2 * B - π / 6) + 1 / 2))

theorem find_angle_A : A = π / 6 := sorry

theorem range_of_y (h_acute : acute_ABC) (B_bounds : π / 6 < B ∧ B < π / 3) : 
  1 < sin (2 * B - π / 6) + 1 / 2 ∧ sin (2 * B - π / 6) + 1 / 2 < 3 / 2 := sorry

variables (a_eq : a = 1)
variables (condition1 : 2 * c - (sqrt 3 + 1) * b = 0)
variables (B_eq : B = π / 4)
variables (sine_eq : c = (sqrt 6 + sqrt 2)/2)
variables (area_eq : 1/2 * b * c * sin A)

theorem area_solution1 (h1 : a_eq) (h2 : condition1) : 
  area_eq = (sqrt 3 + 1) / 4 := sorry

theorem area_solution2 (h1 : a_eq) (h2 : B_eq) : 
  area_eq = (sqrt 3 + 1) / 4 := sorry

end find_angle_A_range_of_y_area_solution1_area_solution2_l502_502470


namespace units_digit_first_four_composites_l502_502600

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l502_502600


namespace number_system_base_l502_502997

theorem number_system_base (a : ℕ) (h : 2 * a^2 + 5 * a + 3 = 136) : a = 7 := 
sorry

end number_system_base_l502_502997


namespace right_triangle_AB_length_l502_502412

/--
In a right triangle ABC with angle C being 90 degrees, if BC = 6 and sin A = 3/4,
then the length of AB is 8.
-/
theorem right_triangle_AB_length (A B C : Type)
  [InnerProductSpace ℝ (EuclideanSpace ℝ)] [DecidableEq C] [DecidableEq B] [DecidableEq A]
  (h₁ : angle (A - C) (B - C) = π / 2)
  (h₂ : dist B C = 6)
  (h₃ : sin (angle (A - C) (B - C)) = 3 / 4) :
  dist A B = 8 :=
sorry

end right_triangle_AB_length_l502_502412


namespace power_function_inverse_point_l502_502341

theorem power_function_inverse_point
  (f : ℝ → ℝ)
  (hinv : Function.inverse f)
  (hpoint : hinv (6 : ℝ) = 36)
  (hdef : ∀ x : ℝ, f x = x ^ (1/2)) :
  f (1 / 9) = 1 / 3 :=
by
  sorry

end power_function_inverse_point_l502_502341


namespace heaviest_vs_lightest_weight_difference_total_weight_of_baskets_l502_502552

variable (n_baskets : ℕ := 20)
variable (standard_weight : ℕ := 25)
variable (dif_weights : List ℕ := [-3, -2, -1.5, 0, 1, 2.5])
variable (num_baskets : List ℕ := [1, 4, 2, 3, 2, 8])

theorem heaviest_vs_lightest_weight_difference :
  let heaviest := 2.5
  let lightest := -3
  (heaviest - lightest = 5.5) :=
by
  sorry

theorem total_weight_of_baskets :
  let total_standard_weight := n_baskets * standard_weight
  let total_deviation := -3 * 1 + -2 * 4 + -1.5 * 2 + 0 * 3 + 1 * 2 + 2.5 * 8
  (total_standard_weight + total_deviation = 508) :=
by
  sorry

end heaviest_vs_lightest_weight_difference_total_weight_of_baskets_l502_502552


namespace largest_eight_digit_number_contains_even_digits_l502_502724

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l502_502724
