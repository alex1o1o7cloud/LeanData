import Mathlib

namespace NUMINAMATH_GPT_monotonic_quadratic_range_l644_64410

-- Define a quadratic function
noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- The theorem
theorem monotonic_quadratic_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≤ quadratic a x₂) ∨
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≥ quadratic a x₂) →
  (a ≤ 2 ∨ 3 ≤ a) :=
sorry

end NUMINAMATH_GPT_monotonic_quadratic_range_l644_64410


namespace NUMINAMATH_GPT_truck_loading_time_l644_64448

theorem truck_loading_time :
  let worker1_rate := (1:ℝ) / 6
  let worker2_rate := (1:ℝ) / 5
  let combined_rate := worker1_rate + worker2_rate
  (combined_rate != 0) → 
  (1 / combined_rate = (30:ℝ) / 11) :=
by
  sorry

end NUMINAMATH_GPT_truck_loading_time_l644_64448


namespace NUMINAMATH_GPT_proof_of_problem_l644_64491

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, (x + 2) ^ (x + 3) = 1 ↔ (x = -1 ∨ x = -3)

theorem proof_of_problem : proof_problem :=
by
  sorry

end NUMINAMATH_GPT_proof_of_problem_l644_64491


namespace NUMINAMATH_GPT_least_number_remainder_l644_64434

theorem least_number_remainder (N k : ℕ) (h : N = 18 * k + 4) : N = 256 :=
by
  sorry

end NUMINAMATH_GPT_least_number_remainder_l644_64434


namespace NUMINAMATH_GPT_difference_of_squares_l644_64419

variable (x y : ℚ)

theorem difference_of_squares (h1 : x + y = 3 / 8) (h2 : x - y = 1 / 8) : x^2 - y^2 = 3 / 64 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l644_64419


namespace NUMINAMATH_GPT_commercial_duration_l644_64417

/-- Michael was watching a TV show, which was aired for 1.5 hours. 
    During this time, there were 3 commercials. 
    The TV show itself, not counting commercials, was 1 hour long. 
    Prove that each commercial lasted 10 minutes. -/
theorem commercial_duration (total_time : ℝ) (num_commercials : ℕ) (show_time : ℝ)
  (h1 : total_time = 1.5) (h2 : num_commercials = 3) (h3 : show_time = 1) :
  (total_time - show_time) / num_commercials * 60 = 10 := 
sorry

end NUMINAMATH_GPT_commercial_duration_l644_64417


namespace NUMINAMATH_GPT_value_of_m_l644_64443

theorem value_of_m (m : ℝ) (h : m ≠ 0)
  (h_roots : ∀ x, m * x^2 + 8 * m * x + 60 = 0 ↔ x = -5 ∨ x = -3) :
  m = 4 :=
sorry

end NUMINAMATH_GPT_value_of_m_l644_64443


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l644_64485

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 4) (h2 : b = 9) (h3 : c = 9) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : a + b + c = 22 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l644_64485


namespace NUMINAMATH_GPT_quadrilateral_inequality_l644_64439

theorem quadrilateral_inequality (A C : ℝ) (AB AC AD BC CD : ℝ) (h1 : A + C < 180) (h2 : A > 0) (h3 : C > 0) (h4 : AB > 0) (h5 : AC > 0) (h6 : AD > 0) (h7 : BC > 0) (h8 : CD > 0) : 
  AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end NUMINAMATH_GPT_quadrilateral_inequality_l644_64439


namespace NUMINAMATH_GPT_river_width_l644_64493

theorem river_width
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) (flow_rate_m_per_min : ℝ)
  (H_depth : depth = 5)
  (H_flow_rate_kmph : flow_rate_kmph = 4)
  (H_volume_per_minute : volume_per_minute = 6333.333333333333)
  (H_flow_rate_m_per_min : flow_rate_m_per_min = 66.66666666666667) :
  volume_per_minute / (depth * flow_rate_m_per_min) = 19 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_river_width_l644_64493


namespace NUMINAMATH_GPT_total_marks_calculation_l644_64425

def average (total_marks : ℕ) (num_candidates : ℕ) : ℕ := total_marks / num_candidates
def total_marks (average : ℕ) (num_candidates : ℕ) : ℕ := average * num_candidates

theorem total_marks_calculation
  (num_candidates : ℕ)
  (average_marks : ℕ)
  (range_min : ℕ)
  (range_max : ℕ)
  (h1 : num_candidates = 250)
  (h2 : average_marks = 42)
  (h3 : range_min = 10)
  (h4 : range_max = 80) :
  total_marks average_marks num_candidates = 10500 :=
by 
  sorry

end NUMINAMATH_GPT_total_marks_calculation_l644_64425


namespace NUMINAMATH_GPT_marble_prob_l644_64454

theorem marble_prob (a c x y p q : ℕ) (h1 : 2 * a + c = 36) 
    (h2 : (x / a) * (x / a) * (y / c) = 1 / 3) 
    (h3 : (a - x) / a * (a - x) / a * (c - y) / c = p / q) 
    (hpq_rel_prime : Nat.gcd p q = 1) : p + q = 65 := by
  sorry

end NUMINAMATH_GPT_marble_prob_l644_64454


namespace NUMINAMATH_GPT_largest_divisor_same_remainder_l644_64466

theorem largest_divisor_same_remainder 
  (d : ℕ) (r : ℕ)
  (a b c : ℕ) 
  (h13511 : 13511 = a * d + r) 
  (h13903 : 13903 = b * d + r)
  (h14589 : 14589 = c * d + r) :
  d = 98 :=
by 
  sorry

end NUMINAMATH_GPT_largest_divisor_same_remainder_l644_64466


namespace NUMINAMATH_GPT_max_value_is_one_sixteenth_l644_64469

noncomputable def max_value_expression (t : ℝ) : ℝ :=
  (3^t - 4 * t) * t / 9^t

theorem max_value_is_one_sixteenth : 
  ∃ t : ℝ, max_value_expression t = 1 / 16 :=
sorry

end NUMINAMATH_GPT_max_value_is_one_sixteenth_l644_64469


namespace NUMINAMATH_GPT_company_workers_l644_64453

theorem company_workers (W : ℕ) (H1 : (1/3 : ℚ) * W = ((1/3 : ℚ) * W)) 
  (H2 : 0.20 * ((1/3 : ℚ) * W) = ((1/15 : ℚ) * W)) 
  (H3 : 0.40 * ((2/3 : ℚ) * W) = ((4/15 : ℚ) * W)) 
  (H4 : (4/15 : ℚ) * W + (4/15 : ℚ) * W = 160)
  : (W - 160 = 140) :=
by
  sorry

end NUMINAMATH_GPT_company_workers_l644_64453


namespace NUMINAMATH_GPT_percent_motorists_exceeding_speed_limit_l644_64459

-- Definitions based on conditions:
def total_motorists := 100
def percent_receiving_tickets := 10
def percent_exceeding_no_ticket := 50

-- The Lean 4 statement to prove the question
theorem percent_motorists_exceeding_speed_limit :
  (percent_receiving_tickets + (percent_receiving_tickets * percent_exceeding_no_ticket / 100)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_motorists_exceeding_speed_limit_l644_64459


namespace NUMINAMATH_GPT_lcm_18_24_30_eq_360_l644_64458

-- Define the three numbers in the condition
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 30

-- State the theorem to prove
theorem lcm_18_24_30_eq_360 : Nat.lcm a (Nat.lcm b c) = 360 :=
by 
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_lcm_18_24_30_eq_360_l644_64458


namespace NUMINAMATH_GPT_jill_trips_to_fill_tank_l644_64471

-- Definitions as per the conditions specified
def tank_capacity : ℕ := 600
def bucket_capacity : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_trips_ratio : ℕ := 3
def jill_trips_ratio : ℕ := 2
def leak_per_trip : ℕ := 2

-- Prove that the number of trips Jill will make = 20 given the above conditions
theorem jill_trips_to_fill_tank : 
  (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) * (tank_capacity / ((jack_trips_ratio + jill_trips_ratio) * (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) / (jack_trips_ratio + jill_trips_ratio)))  = 20 := 
sorry

end NUMINAMATH_GPT_jill_trips_to_fill_tank_l644_64471


namespace NUMINAMATH_GPT_sin_of_7pi_over_6_l644_64412

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_of_7pi_over_6_l644_64412


namespace NUMINAMATH_GPT_maximize_revenue_l644_64409

-- Define the problem conditions
def is_valid (x y : ℕ) : Prop :=
  x + y ≤ 60 ∧ 6 * x + 30 * y ≤ 600

-- Define the objective function
def revenue (x y : ℕ) : ℚ :=
  2.5 * x + 7.5 * y

-- State the theorem with the given conditions
theorem maximize_revenue : 
  (∃ x y : ℕ, is_valid x y ∧ ∀ a b : ℕ, is_valid a b → revenue x y >= revenue a b) ∧
  ∃ x y, is_valid x y ∧ revenue x y = revenue 50 10 := 
sorry

end NUMINAMATH_GPT_maximize_revenue_l644_64409


namespace NUMINAMATH_GPT_total_distance_walked_l644_64429

def distance_to_fountain : ℕ := 30
def number_of_trips : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain

theorem total_distance_walked : (number_of_trips * round_trip_distance) = 240 := by
  sorry

end NUMINAMATH_GPT_total_distance_walked_l644_64429


namespace NUMINAMATH_GPT_number_of_black_balls_l644_64430

theorem number_of_black_balls
  (total_balls : ℕ)  -- define the total number of balls
  (B : ℕ)            -- define B as the number of black balls
  (prob_red : ℚ := 1/4) -- define the probability of drawing a red ball as 1/4
  (red_balls : ℕ := 3)  -- define the number of red balls as 3
  (h1 : total_balls = red_balls + B) -- total balls is the sum of red and black balls
  (h2 : red_balls / total_balls = prob_red) -- given probability
  : B = 9 :=              -- we need to prove that B is 9
by
  sorry

end NUMINAMATH_GPT_number_of_black_balls_l644_64430


namespace NUMINAMATH_GPT_bread_slices_per_loaf_l644_64455

theorem bread_slices_per_loaf (friends: ℕ) (total_loaves : ℕ) (slices_per_friend: ℕ) (total_slices: ℕ)
  (h1 : friends = 10) (h2 : total_loaves = 4) (h3 : slices_per_friend = 6) (h4 : total_slices = friends * slices_per_friend):
  total_slices / total_loaves = 15 :=
by
  sorry

end NUMINAMATH_GPT_bread_slices_per_loaf_l644_64455


namespace NUMINAMATH_GPT_union_complement_correctness_l644_64486

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complement_correctness : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 4} →
  A ∪ (U \ B) = {1, 2, 3, 5} :=
by
  intro hU hA hB
  sorry

end NUMINAMATH_GPT_union_complement_correctness_l644_64486


namespace NUMINAMATH_GPT_find_x_for_set_6_l644_64483

theorem find_x_for_set_6 (x : ℝ) (h : 6 ∈ ({2, 4, x^2 - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_for_set_6_l644_64483


namespace NUMINAMATH_GPT_consecutive_sum_impossible_l644_64480

theorem consecutive_sum_impossible (n : ℕ) :
  (¬ (∃ (a b : ℕ), a < b ∧ n = (b - a + 1) * (a + b) / 2)) ↔ ∃ s : ℕ, n = 2 ^ s :=
sorry

end NUMINAMATH_GPT_consecutive_sum_impossible_l644_64480


namespace NUMINAMATH_GPT_area_of_triangle_l644_64405

-- Definitions of the conditions
def hypotenuse_AC (a b c : ℝ) : Prop := c = 50
def sum_of_legs (a b : ℝ) : Prop := a + b = 70
def pythagorean_theorem (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- The main theorem statement
theorem area_of_triangle (a b c : ℝ) (h1 : hypotenuse_AC a b c)
  (h2 : sum_of_legs a b) (h3 : pythagorean_theorem a b c) : 
  (1/2) * a * b = 300 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l644_64405


namespace NUMINAMATH_GPT_correct_value_l644_64474

-- Given condition
def incorrect_calculation (x : ℝ) : Prop := (x + 12) / 8 = 8

-- Theorem to prove the correct value
theorem correct_value (x : ℝ) (h : incorrect_calculation x) : (x - 12) * 9 = 360 :=
by
  sorry

end NUMINAMATH_GPT_correct_value_l644_64474


namespace NUMINAMATH_GPT_game_returns_to_A_after_three_rolls_l644_64452

theorem game_returns_to_A_after_three_rolls :
  (∃ i j k : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ 1 ≤ k ∧ k ≤ 6 ∧ (i + j + k) % 12 = 0) → 
  true :=
by
  sorry

end NUMINAMATH_GPT_game_returns_to_A_after_three_rolls_l644_64452


namespace NUMINAMATH_GPT_probability_male_monday_female_tuesday_l644_64426

structure Volunteers where
  men : ℕ
  women : ℕ
  total : ℕ

def group : Volunteers := {men := 2, women := 2, total := 4}

def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_male_monday_female_tuesday :
  let n := permutations group.total 2
  let m := combinations group.men 1 * combinations group.women 1
  (m / n : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_male_monday_female_tuesday_l644_64426


namespace NUMINAMATH_GPT_boat_distance_ratio_l644_64421

theorem boat_distance_ratio :
  ∀ (D_u D_d : ℝ),
  (3.6 = (D_u + D_d) / ((D_u / 4) + (D_d / 6))) →
  D_u / D_d = 4 :=
by
  intros D_u D_d h
  sorry

end NUMINAMATH_GPT_boat_distance_ratio_l644_64421


namespace NUMINAMATH_GPT_tilly_bag_cost_l644_64423

noncomputable def cost_per_bag (n s P τ F : ℕ) : ℕ :=
  let revenue := n * s
  let total_sales_tax := n * (s * τ / 100)
  let total_additional_expenses := total_sales_tax + F
  (revenue - (P + total_additional_expenses)) / n

theorem tilly_bag_cost :
  let n := 100
  let s := 10
  let P := 300
  let τ := 5
  let F := 50
  cost_per_bag n s P τ F = 6 :=
  by
    let n := 100
    let s := 10
    let P := 300
    let τ := 5
    let F := 50
    have : cost_per_bag n s P τ F = 6 := sorry
    exact this

end NUMINAMATH_GPT_tilly_bag_cost_l644_64423


namespace NUMINAMATH_GPT_longest_side_of_triangle_l644_64472

-- Defining variables and constants
variables (x : ℕ)

-- Defining the side lengths of the triangle
def side1 := 7
def side2 := x + 4
def side3 := 2 * x + 1

-- Defining the perimeter of the triangle
def perimeter := side1 + side2 + side3

-- Statement of the main theorem
theorem longest_side_of_triangle (h : perimeter x = 36) : max side1 (max (side2 x) (side3 x)) = 17 :=
by sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l644_64472


namespace NUMINAMATH_GPT_austin_more_apples_than_dallas_l644_64463

-- Conditions as definitions
def dallas_apples : ℕ := 14
def dallas_pears : ℕ := 9
def austin_pears : ℕ := dallas_pears - 5
def austin_total_fruit : ℕ := 24

-- The theorem statement
theorem austin_more_apples_than_dallas 
  (austin_apples : ℕ) (h1 : austin_apples + austin_pears = austin_total_fruit) :
  austin_apples - dallas_apples = 6 :=
sorry

end NUMINAMATH_GPT_austin_more_apples_than_dallas_l644_64463


namespace NUMINAMATH_GPT_ratio_of_girls_l644_64451

theorem ratio_of_girls (total_julian_friends : ℕ) (percent_julian_girls : ℚ)
  (percent_julian_boys : ℚ) (total_boyd_friends : ℕ) (percent_boyd_boys : ℚ) :
  total_julian_friends = 80 →
  percent_julian_girls = 0.40 →
  percent_julian_boys = 0.60 →
  total_boyd_friends = 100 →
  percent_boyd_boys = 0.36 →
  (0.64 * total_boyd_friends : ℚ) / (0.40 * total_julian_friends : ℚ) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_l644_64451


namespace NUMINAMATH_GPT_find_smallest_number_l644_64496

theorem find_smallest_number
  (a1 a2 a3 a4 : ℕ)
  (h1 : (a1 + a2 + a3 + a4) / 4 = 30)
  (h2 : a2 = 28)
  (h3 : a2 = 35 - 7) :
  a1 = 27 :=
sorry

end NUMINAMATH_GPT_find_smallest_number_l644_64496


namespace NUMINAMATH_GPT_Antoinette_weight_l644_64495

-- Define weights for Antoinette and Rupert
variables (A R : ℕ)

-- Define the given conditions
def condition1 := A = 2 * R - 7
def condition2 := A + R = 98

-- The theorem to prove under the given conditions
theorem Antoinette_weight : condition1 A R → condition2 A R → A = 63 := 
by {
  -- The proof is omitted
  sorry
}

end NUMINAMATH_GPT_Antoinette_weight_l644_64495


namespace NUMINAMATH_GPT_part1_part2_l644_64473

open Real

noncomputable def condition1 (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a^2 + 3 * b^2 = 3

theorem part1 {a b : ℝ} (h : condition1 a b) : sqrt 5 * a + b ≤ 4 := 
sorry

theorem part2 {x a b : ℝ} (h₁ : condition1 a b) (h₂ : 2 * abs (x - 1) + abs x ≥ 4) : 
x ≤ -2/3 ∨ x ≥ 2 := 
sorry

end NUMINAMATH_GPT_part1_part2_l644_64473


namespace NUMINAMATH_GPT_carol_blocks_l644_64464

theorem carol_blocks (x : ℕ) (h : x - 25 = 17) : x = 42 :=
sorry

end NUMINAMATH_GPT_carol_blocks_l644_64464


namespace NUMINAMATH_GPT_hyperbola_iff_m_lt_0_l644_64402

theorem hyperbola_iff_m_lt_0 (m : ℝ) : (m < 0) ↔ (∃ x y : ℝ,  x^2 + m * y^2 = m) :=
by sorry

end NUMINAMATH_GPT_hyperbola_iff_m_lt_0_l644_64402


namespace NUMINAMATH_GPT_distinct_ratios_zero_l644_64407

theorem distinct_ratios_zero (p q r : ℝ) (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) 
  (h : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 :=
sorry

end NUMINAMATH_GPT_distinct_ratios_zero_l644_64407


namespace NUMINAMATH_GPT_cubic_poly_real_roots_l644_64415

theorem cubic_poly_real_roots (a b c d : ℝ) (h : a ≠ 0) : 
  ∃ (min_roots max_roots : ℕ), 1 ≤ min_roots ∧ max_roots ≤ 3 ∧ min_roots = 1 ∧ max_roots = 3 :=
by
  sorry

end NUMINAMATH_GPT_cubic_poly_real_roots_l644_64415


namespace NUMINAMATH_GPT_fencing_rate_3_rs_per_meter_l644_64413

noncomputable def rate_per_meter (A_hectares : ℝ) (total_cost : ℝ) : ℝ := 
  let A_m2 := A_hectares * 10000
  let r := Real.sqrt (A_m2 / Real.pi)
  let C := 2 * Real.pi * r
  total_cost / C

theorem fencing_rate_3_rs_per_meter : rate_per_meter 17.56 4456.44 = 3.00 :=
by 
  sorry

end NUMINAMATH_GPT_fencing_rate_3_rs_per_meter_l644_64413


namespace NUMINAMATH_GPT_T_number_square_l644_64401

theorem T_number_square (a b : ℤ) : ∃ c d : ℤ, (a^2 + a * b + b^2)^2 = c^2 + c * d + d^2 := by
  sorry

end NUMINAMATH_GPT_T_number_square_l644_64401


namespace NUMINAMATH_GPT_abs_diff_between_sequences_l644_64478

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 + (-8) * (n - 1)

theorem abs_diff_between_sequences :
  |sequence_C 31 - sequence_D 31| = 600 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_between_sequences_l644_64478


namespace NUMINAMATH_GPT_circle_radius_tangents_l644_64411

theorem circle_radius_tangents
  (AB CD EF r : ℝ)
  (circle_tangent_AB : AB = 5)
  (circle_tangent_CD : CD = 11)
  (circle_tangent_EF : EF = 15) :
  r = 2.5 := by
  sorry

end NUMINAMATH_GPT_circle_radius_tangents_l644_64411


namespace NUMINAMATH_GPT_total_toys_l644_64449

theorem total_toys (toys_kamari : ℕ) (toys_anais : ℕ) (h1 : toys_kamari = 65) (h2 : toys_anais = toys_kamari + 30) :
  toys_kamari + toys_anais = 160 :=
by 
  sorry

end NUMINAMATH_GPT_total_toys_l644_64449


namespace NUMINAMATH_GPT_max_sundays_in_84_days_l644_64450

-- Define constants
def days_in_week : ℕ := 7
def total_days : ℕ := 84

-- Theorem statement
theorem max_sundays_in_84_days : (total_days / days_in_week) = 12 :=
by sorry

end NUMINAMATH_GPT_max_sundays_in_84_days_l644_64450


namespace NUMINAMATH_GPT_find_x_in_inches_l644_64440

theorem find_x_in_inches (x : ℝ) :
  let area_smaller_square := 9 * x^2
  let area_larger_square := 36 * x^2
  let area_triangle := 9 * x^2
  area_smaller_square + area_larger_square + area_triangle = 1950 → x = (5 * Real.sqrt 13) / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_inches_l644_64440


namespace NUMINAMATH_GPT_ducks_counted_l644_64403

theorem ducks_counted (x y : ℕ) (h1 : x + y = 300) (h2 : 2 * x + 4 * y = 688) : x = 256 :=
by
  sorry

end NUMINAMATH_GPT_ducks_counted_l644_64403


namespace NUMINAMATH_GPT_female_officers_on_duty_percentage_l644_64441

   def percentage_of_females_on_duty (total_on_duty : ℕ) (female_on_duty : ℕ) (total_females : ℕ) : ℕ :=
   (female_on_duty * 100) / total_females
  
   theorem female_officers_on_duty_percentage
     (total_on_duty : ℕ) (h1 : total_on_duty = 180)
     (female_on_duty : ℕ) (h2 : female_on_duty = total_on_duty / 2)
     (total_females : ℕ) (h3 : total_females = 500) :
     percentage_of_females_on_duty total_on_duty female_on_duty total_females = 18 :=
   by
     rw [h1, h2, h3]
     sorry
   
end NUMINAMATH_GPT_female_officers_on_duty_percentage_l644_64441


namespace NUMINAMATH_GPT_how_many_times_faster_l644_64477

theorem how_many_times_faster (A B : ℝ) (h1 : A = 1 / 32) (h2 : A + B = 1 / 24) : A / B = 3 := by
  sorry

end NUMINAMATH_GPT_how_many_times_faster_l644_64477


namespace NUMINAMATH_GPT_inequality_solution_l644_64487

theorem inequality_solution :
  {x : ℝ | -x^2 - |x| + 6 > 0} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_inequality_solution_l644_64487


namespace NUMINAMATH_GPT_min_value_fraction_expr_l644_64456

theorem min_value_fraction_expr : ∀ (x : ℝ), x > 0 → (4 + x) * (1 + x) / x ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_expr_l644_64456


namespace NUMINAMATH_GPT_difference_between_eights_l644_64435

theorem difference_between_eights (value_tenths : ℝ) (value_hundredths : ℝ) (h1 : value_tenths = 0.8) (h2 : value_hundredths = 0.08) : 
  value_tenths - value_hundredths = 0.72 :=
by 
  sorry

end NUMINAMATH_GPT_difference_between_eights_l644_64435


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_divisible_by_5_l644_64460

theorem sum_of_squares_of_consecutive_integers_divisible_by_5 (n : ℤ) :
  (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_divisible_by_5_l644_64460


namespace NUMINAMATH_GPT_smallest_k_value_for_screws_packs_l644_64484

theorem smallest_k_value_for_screws_packs :
  ∃ k : ℕ, k = 60 ∧ (∃ x y : ℕ, (k = 10 * x ∧ k = 12 * y) ∧ x ≠ y) := sorry

end NUMINAMATH_GPT_smallest_k_value_for_screws_packs_l644_64484


namespace NUMINAMATH_GPT_vincent_earnings_l644_64492

def fantasy_book_cost : ℕ := 6
def literature_book_cost : ℕ := fantasy_book_cost / 2
def mystery_book_cost : ℕ := 4

def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def mystery_books_sold_per_day : ℕ := 3

def daily_earnings : ℕ :=
  (fantasy_books_sold_per_day * fantasy_book_cost) +
  (literature_books_sold_per_day * literature_book_cost) +
  (mystery_books_sold_per_day * mystery_book_cost)

def total_earnings_after_seven_days : ℕ :=
  daily_earnings * 7

theorem vincent_earnings : total_earnings_after_seven_days = 462 :=
by
  sorry

end NUMINAMATH_GPT_vincent_earnings_l644_64492


namespace NUMINAMATH_GPT_find_y_in_terms_of_x_l644_64445

variable (x y : ℝ)

theorem find_y_in_terms_of_x (hx : x = 5) (hy : y = -4) (hp : ∃ k, y = k * (x - 3)) :
  y = -2 * x + 6 := by
sorry

end NUMINAMATH_GPT_find_y_in_terms_of_x_l644_64445


namespace NUMINAMATH_GPT_modulus_product_l644_64428

open Complex -- to open the complex namespace

-- Define the complex numbers
def z1 : ℂ := 10 - 5 * Complex.I
def z2 : ℂ := 7 + 24 * Complex.I

-- State the theorem to prove
theorem modulus_product : abs (z1 * z2) = 125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_modulus_product_l644_64428


namespace NUMINAMATH_GPT_angle_C_in_triangle_l644_64481

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 90) (h2 : A + B + C = 180) : C = 90 :=
sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l644_64481


namespace NUMINAMATH_GPT_mindy_earns_k_times_more_than_mork_l644_64422

-- Given the following conditions:
-- Mork's tax rate: 0.45
-- Mindy's tax rate: 0.25
-- Combined tax rate: 0.29
-- Mindy earns k times more than Mork

theorem mindy_earns_k_times_more_than_mork (M : ℝ) (k : ℝ) (hM : M > 0) :
  (0.45 * M + 0.25 * k * M) / (M * (1 + k)) = 0.29 → k = 4 :=
by
  sorry

end NUMINAMATH_GPT_mindy_earns_k_times_more_than_mork_l644_64422


namespace NUMINAMATH_GPT_find_value_of_a_l644_64400

theorem find_value_of_a 
  (P : ℝ × ℝ)
  (a : ℝ)
  (α : ℝ)
  (point_on_terminal_side : P = (-4, a))
  (sin_cos_condition : Real.sin α * Real.cos α = Real.sqrt 3 / 4) : 
  a = -4 * Real.sqrt 3 ∨ a = - (4 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_GPT_find_value_of_a_l644_64400


namespace NUMINAMATH_GPT_score_of_juniors_correct_l644_64465

-- Let the total number of students be 20
def total_students : ℕ := 20

-- 20% of the students are juniors
def juniors_percent : ℝ := 0.20

-- Total number of juniors
def number_of_juniors : ℕ := 4 -- 20% of 20

-- The remaining are seniors
def number_of_seniors : ℕ := 16 -- 80% of 20

-- Overall average score of all students
def overall_average_score : ℝ := 85

-- Average score of the seniors
def seniors_average_score : ℝ := 84

-- Calculate the total score of all students
def total_score : ℝ := overall_average_score * total_students

-- Calculate the total score of the seniors
def total_score_of_seniors : ℝ := seniors_average_score * number_of_seniors

-- We need to prove that the score of each junior
def score_of_each_junior : ℝ := 89

theorem score_of_juniors_correct :
  (total_score - total_score_of_seniors) / number_of_juniors = score_of_each_junior :=
by
  sorry

end NUMINAMATH_GPT_score_of_juniors_correct_l644_64465


namespace NUMINAMATH_GPT_conversion_base8_to_base10_l644_64488

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end NUMINAMATH_GPT_conversion_base8_to_base10_l644_64488


namespace NUMINAMATH_GPT_plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l644_64447

theorem plan_Y_cheaper_than_X (x : ℕ) : 
  ∃ x, 2500 + 7 * x < 15 * x ∧ ∀ y, y < x → ¬ (2500 + 7 * y < 15 * y) := 
sorry

theorem plan_Z_cheaper_than_X (x : ℕ) : 
  ∃ x, 3000 + 6 * x < 15 * x ∧ ∀ y, y < x → ¬ (3000 + 6 * y < 15 * y) := 
sorry

end NUMINAMATH_GPT_plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l644_64447


namespace NUMINAMATH_GPT_K9_le_89_K9_example_171_l644_64424

section weights_proof

def K (n : ℕ) (P : ℕ) : ℕ := sorry -- Assume the definition of K given by the problem

theorem K9_le_89 : ∀ P, K 9 P ≤ 89 := by
  sorry -- Proof to be filled

def example_weight : ℕ := 171

theorem K9_example_171 : K 9 example_weight = 89 := by
  sorry -- Proof to be filled

end weights_proof

end NUMINAMATH_GPT_K9_le_89_K9_example_171_l644_64424


namespace NUMINAMATH_GPT_opposite_of_2023_l644_64479

/-- The opposite of a number n is defined as the number that, when added to n, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l644_64479


namespace NUMINAMATH_GPT_solve_linear_system_l644_64404

theorem solve_linear_system (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = -10 - 4 * y)
  (h3 : x + y = 14 - 4 * z)
  : 2 * x + 2 * y + 2 * z = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l644_64404


namespace NUMINAMATH_GPT_ratio_of_side_lengths_l644_64438

theorem ratio_of_side_lengths (t s : ℕ) (ht : 2 * t + (20 - 2 * t) = 20) (hs : 4 * s = 20) :
  t / s = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_side_lengths_l644_64438


namespace NUMINAMATH_GPT_expression_evaluation_l644_64406

theorem expression_evaluation :
  (0.15)^3 - (0.06)^3 / (0.15)^2 + 0.009 + (0.06)^2 = 0.006375 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l644_64406


namespace NUMINAMATH_GPT_decreasing_interval_l644_64490

noncomputable def f (x : ℝ) := x^2 * Real.exp x

theorem decreasing_interval : ∀ x : ℝ, x > -2 ∧ x < 0 → deriv f x < 0 := 
by
  intro x h
  sorry

end NUMINAMATH_GPT_decreasing_interval_l644_64490


namespace NUMINAMATH_GPT_complement_union_eq_l644_64420

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end NUMINAMATH_GPT_complement_union_eq_l644_64420


namespace NUMINAMATH_GPT_total_cost_l644_64461

-- Defining the prices based on the given conditions
def price_smartphone : ℕ := 300
def price_pc : ℕ := price_smartphone + 500
def price_tablet : ℕ := price_smartphone + price_pc

-- The theorem to prove the total cost of buying one of each product
theorem total_cost : price_smartphone + price_pc + price_tablet = 2200 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l644_64461


namespace NUMINAMATH_GPT_score_order_l644_64494

variable (A B C D : ℕ)

theorem score_order (h1 : A + B = C + D) (h2 : C + A > B + D) (h3 : C > A + B) :
  (C > A ∧ A > B ∧ B > D) :=
by
  sorry

end NUMINAMATH_GPT_score_order_l644_64494


namespace NUMINAMATH_GPT_average_speed_round_trip_l644_64444

/--
Let \( d = 150 \) miles be the distance from City \( X \) to City \( Y \).
Let \( v1 = 50 \) mph be the speed from \( X \) to \( Y \).
Let \( v2 = 30 \) mph be the speed from \( Y \) to \( X \).
Then the average speed for the round trip is 37.5 mph.
-/
theorem average_speed_round_trip :
  let d := 150
  let v1 := 50
  let v2 := 30
  (2 * d) / ((d / v1) + (d / v2)) = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l644_64444


namespace NUMINAMATH_GPT_financial_outcome_l644_64408

theorem financial_outcome :
  let initial_value : ℝ := 12000
  let selling_price : ℝ := initial_value * 1.20
  let buying_price : ℝ := selling_price * 0.85
  let financial_outcome : ℝ := buying_price - initial_value
  financial_outcome = 240 :=
by
  sorry

end NUMINAMATH_GPT_financial_outcome_l644_64408


namespace NUMINAMATH_GPT_find_numbers_l644_64497

theorem find_numbers (x y : ℝ) (h₁ : x + y = x * y) (h₂ : x * y = x / y) :
  (x = 1 / 2) ∧ (y = -1) := by
  sorry

end NUMINAMATH_GPT_find_numbers_l644_64497


namespace NUMINAMATH_GPT_vacant_seats_l644_64431

theorem vacant_seats (total_seats : ℕ) (filled_percent vacant_percent : ℚ) 
  (h_total : total_seats = 600)
  (h_filled_percent : filled_percent = 75)
  (h_vacant_percent : vacant_percent = 100 - filled_percent)
  (h_vacant_percent_25 : vacant_percent = 25) :
  (25 / 100) * 600 = 150 :=
by 
  -- this is the final answer we want to prove, replace with sorry to skip the proof just for statement validation
  sorry

end NUMINAMATH_GPT_vacant_seats_l644_64431


namespace NUMINAMATH_GPT_value_of_f_5_l644_64418

theorem value_of_f_5 (f : ℕ → ℕ) (y : ℕ)
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : f 5 = 62 :=
sorry

end NUMINAMATH_GPT_value_of_f_5_l644_64418


namespace NUMINAMATH_GPT_sum_of_legs_eq_40_l644_64462

theorem sum_of_legs_eq_40
  (x : ℝ)
  (h1 : x > 0)
  (h2 : x^2 + (x + 2)^2 = 29^2) :
  x + (x + 2) = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_legs_eq_40_l644_64462


namespace NUMINAMATH_GPT_solution_set_quadratic_l644_64432

-- Define the quadratic equation as a function
def quadratic_eq (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- The theorem to prove
theorem solution_set_quadratic :
  {x : ℝ | quadratic_eq x = 0} = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_quadratic_l644_64432


namespace NUMINAMATH_GPT_girls_tried_out_l644_64436

-- Definitions for conditions
def boys_trying_out : ℕ := 4
def students_called_back : ℕ := 26
def students_did_not_make_cut : ℕ := 17

-- Definition to calculate total students who tried out
def total_students_who_tried_out : ℕ := students_called_back + students_did_not_make_cut

-- Proof statement
theorem girls_tried_out : ∀ (G : ℕ), G + boys_trying_out = total_students_who_tried_out → G = 39 :=
by
  intro G
  intro h
  rw [total_students_who_tried_out, boys_trying_out] at h
  sorry

end NUMINAMATH_GPT_girls_tried_out_l644_64436


namespace NUMINAMATH_GPT_relationship_a_b_c_l644_64433

open Real

theorem relationship_a_b_c (x : ℝ) (hx1 : e < x) (hx2 : x < e^2)
  (a : ℝ) (ha : a = log x)
  (b : ℝ) (hb : b = (1 / 2) ^ log x)
  (c : ℝ) (hc : c = exp (log x)) :
  c > a ∧ a > b :=
by {
  -- we state the theorem without providing the proof for now
  sorry
}

end NUMINAMATH_GPT_relationship_a_b_c_l644_64433


namespace NUMINAMATH_GPT_total_emails_675_l644_64476

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end NUMINAMATH_GPT_total_emails_675_l644_64476


namespace NUMINAMATH_GPT_exists_pos_integer_n_l644_64427

theorem exists_pos_integer_n (n : ℕ) (hn_pos : n > 0) (h : ∃ m : ℕ, m * m = 1575 * n) : n = 7 :=
sorry

end NUMINAMATH_GPT_exists_pos_integer_n_l644_64427


namespace NUMINAMATH_GPT_train_speed_kmph_l644_64446

theorem train_speed_kmph (length time : ℝ) (h_length : length = 90) (h_time : time = 8.999280057595392) :
  (length / time) * 3.6 = 36.003 :=
by
  rw [h_length, h_time]
  norm_num
  sorry -- the norm_num tactic might simplify this enough, otherwise further steps would be added here.

end NUMINAMATH_GPT_train_speed_kmph_l644_64446


namespace NUMINAMATH_GPT_abs_expression_value_l644_64442

theorem abs_expression_value : (abs (2 * Real.pi - abs (Real.pi - 9))) = 3 * Real.pi - 9 := 
by
  sorry

end NUMINAMATH_GPT_abs_expression_value_l644_64442


namespace NUMINAMATH_GPT_set_in_quadrant_I_l644_64416

theorem set_in_quadrant_I (x y : ℝ) (h1 : y ≥ 3 * x) (h2 : y ≥ 5 - x) (h3 : y < 7) : 
  x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_set_in_quadrant_I_l644_64416


namespace NUMINAMATH_GPT_JessieScore_l644_64470

-- Define the conditions as hypotheses
variables (correct_answers : ℕ) (incorrect_answers : ℕ) (unanswered_questions : ℕ)
variables (points_per_correct : ℕ) (points_deducted_per_incorrect : ℤ)

-- Define the values for the specific problem instance
def JessieCondition := correct_answers = 16 ∧ incorrect_answers = 4 ∧ unanswered_questions = 10 ∧
                       points_per_correct = 2 ∧ points_deducted_per_incorrect = -1 / 2

-- Define the statement that Jessie's score is 30 given the conditions
theorem JessieScore (h : JessieCondition correct_answers incorrect_answers unanswered_questions points_per_correct points_deducted_per_incorrect) :
  (correct_answers * points_per_correct : ℤ) + (incorrect_answers * points_deducted_per_incorrect) = 30 :=
by
  sorry

end NUMINAMATH_GPT_JessieScore_l644_64470


namespace NUMINAMATH_GPT_age_problem_l644_64437

theorem age_problem (S F : ℕ) (h1 : F = S + 27) (h2 : F + 2 = 2 * (S + 2)) :
  S = 25 := by
  sorry

end NUMINAMATH_GPT_age_problem_l644_64437


namespace NUMINAMATH_GPT_initial_percentage_salt_l644_64475

theorem initial_percentage_salt :
  ∀ (P : ℝ),
  let Vi := 64 
  let Vf := 80
  let target_percent := 0.08
  (Vi * P = Vf * target_percent) → P = 0.1 :=
by
  intros P Vi Vf target_percent h
  have h1 : Vi = 64 := rfl
  have h2 : Vf = 80 := rfl
  have h3 : target_percent = 0.08 := rfl
  rw [h1, h2, h3] at h
  sorry

end NUMINAMATH_GPT_initial_percentage_salt_l644_64475


namespace NUMINAMATH_GPT_infinite_solutions_l644_64498

theorem infinite_solutions (x : ℕ) :
  15 < 2 * x + 10 ↔ ∃ n : ℕ, x = n + 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_infinite_solutions_l644_64498


namespace NUMINAMATH_GPT_smallest_even_n_for_reducible_fraction_l644_64467

theorem smallest_even_n_for_reducible_fraction : 
  ∃ (N: ℕ), (N > 2013) ∧ (N % 2 = 0) ∧ (Nat.gcd (15 * N - 7) (22 * N - 5) > 1) ∧ N = 2144 :=
sorry

end NUMINAMATH_GPT_smallest_even_n_for_reducible_fraction_l644_64467


namespace NUMINAMATH_GPT_find_power_l644_64457

theorem find_power (some_power : ℕ) (k : ℕ) :
  k = 8 → (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → some_power = 16 :=
by
  intro h1 h2
  rw [h1] at h2
  sorry

end NUMINAMATH_GPT_find_power_l644_64457


namespace NUMINAMATH_GPT_find_f_neg_two_l644_64468

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : 3 * f (1 / x) + (2 * f x) / x = x ^ 2

theorem find_f_neg_two : f (-2) = 67 / 20 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_two_l644_64468


namespace NUMINAMATH_GPT_problem_correct_l644_64414

noncomputable def S : Set ℕ := {x | x^2 - x = 0}
noncomputable def T : Set ℕ := {x | x ∈ Set.univ ∧ 6 % (x - 2) = 0}

theorem problem_correct : S ∩ T = ∅ :=
by sorry

end NUMINAMATH_GPT_problem_correct_l644_64414


namespace NUMINAMATH_GPT_f_2016_eq_one_third_l644_64489

noncomputable def f (x : ℕ) : ℝ := sorry

axiom f_one : f 1 = 2
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = (1 + f x) / (1 - f x)

theorem f_2016_eq_one_third : f 2016 = 1 / 3 := sorry

end NUMINAMATH_GPT_f_2016_eq_one_third_l644_64489


namespace NUMINAMATH_GPT_greatest_cars_with_ac_not_racing_stripes_l644_64499

-- Definitions
def total_cars : ℕ := 100
def cars_without_ac : ℕ := 47
def cars_with_ac : ℕ := total_cars - cars_without_ac
def at_least_racing_stripes : ℕ := 53

-- Prove that the greatest number of cars that could have air conditioning but not racing stripes is 53
theorem greatest_cars_with_ac_not_racing_stripes :
  ∃ maximum_cars_with_ac_not_racing_stripes, 
    maximum_cars_with_ac_not_racing_stripes = cars_with_ac - 0 ∧
    maximum_cars_with_ac_not_racing_stripes = 53 := 
by
  sorry

end NUMINAMATH_GPT_greatest_cars_with_ac_not_racing_stripes_l644_64499


namespace NUMINAMATH_GPT_unloading_time_relationship_l644_64482

-- Conditions
def loading_speed : ℝ := 30
def loading_time : ℝ := 8
def total_tonnage : ℝ := loading_speed * loading_time
def unloading_speed (x : ℝ) : ℝ := x

-- Proof statement
theorem unloading_time_relationship (x : ℝ) (hx : x ≠ 0) : 
  ∀ y : ℝ, y = 240 / x :=
by 
  sorry

end NUMINAMATH_GPT_unloading_time_relationship_l644_64482
