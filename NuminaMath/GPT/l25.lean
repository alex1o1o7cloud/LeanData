import Mathlib

namespace circle_repr_eq_l25_2585

theorem circle_repr_eq (a : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 + a = 0) ↔ a < 4 :=
by
  sorry

end circle_repr_eq_l25_2585


namespace mary_age_l25_2512

theorem mary_age (x : ℤ) (n m : ℤ) : (x - 2 = n^2) ∧ (x + 2 = m^3) → x = 6 := by
  sorry

end mary_age_l25_2512


namespace find_xyz_l25_2552

theorem find_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end find_xyz_l25_2552


namespace find_a_parallel_find_a_perpendicular_l25_2591

open Real

def line_parallel (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 = k2

def line_perpendicular (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 * k2 = -1

theorem find_a_parallel (a : ℝ) :
  line_parallel (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 1 ∨ a = 6 :=
by sorry

theorem find_a_perpendicular (a : ℝ) :
  line_perpendicular (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 3 ∨ a = -4 :=
by sorry

end find_a_parallel_find_a_perpendicular_l25_2591


namespace intersection_ab_correct_l25_2544

noncomputable def set_A : Set ℝ := { x : ℝ | x > 1/3 }
def set_B : Set ℝ := { x : ℝ | ∃ y : ℝ, x^2 + y^2 = 4 ∧ y ≥ -2 ∧ y ≤ 2 }
def intersection_AB : Set ℝ := { x : ℝ | 1/3 < x ∧ x ≤ 2 }

theorem intersection_ab_correct : set_A ∩ set_B = intersection_AB := 
by 
  -- proof omitted
  sorry

end intersection_ab_correct_l25_2544


namespace greatest_three_digit_multiple_of_17_l25_2535

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l25_2535


namespace smallest_c_for_inverse_l25_2510

def f (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c, (∀ x₁ x₂, (c ≤ x₁ ∧ c ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) ∧
       (∀ d, (∀ x₁ x₂, (d ≤ x₁ ∧ d ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) → c ≤ d) ∧
       c = 3 := sorry

end smallest_c_for_inverse_l25_2510


namespace probability_of_selecting_at_least_one_female_l25_2537

open BigOperators

noncomputable def prob_at_least_one_female_selected : ℚ :=
  let total_choices := Nat.choose 10 3
  let all_males_choices := Nat.choose 6 3
  1 - (all_males_choices / total_choices : ℚ)

theorem probability_of_selecting_at_least_one_female :
  prob_at_least_one_female_selected = 5 / 6 := by
  sorry

end probability_of_selecting_at_least_one_female_l25_2537


namespace count_prime_digit_sums_less_than_10_l25_2507

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ := n / 10 + n % 10

def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem count_prime_digit_sums_less_than_10 :
  ∃ count : ℕ, count = 17 ∧
  ∀ n : ℕ, is_two_digit_number n →
  (is_prime (sum_of_digits n) ∧ sum_of_digits n < 10) ↔
  n ∈ [11, 20, 12, 21, 30, 14, 23, 32, 41, 50, 16, 25, 34, 43, 52, 61, 70] :=
sorry

end count_prime_digit_sums_less_than_10_l25_2507


namespace ganesh_average_speed_l25_2571

noncomputable def averageSpeed (D : ℝ) : ℝ :=
  let time_uphill := D / 60
  let time_downhill := D / 36
  let total_time := time_uphill + time_downhill
  let total_distance := 2 * D
  total_distance / total_time

theorem ganesh_average_speed (D : ℝ) (hD : D > 0) : averageSpeed D = 45 := by
  sorry

end ganesh_average_speed_l25_2571


namespace geometric_sequence_sum_is_9_l25_2528

theorem geometric_sequence_sum_is_9 {a : ℕ → ℝ} (q : ℝ) 
  (h3a7 : a 3 * a 7 = 8) 
  (h4a6 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a (n + 1) = a n * q) : a 2 + a 8 = 9 :=
sorry

end geometric_sequence_sum_is_9_l25_2528


namespace news_spread_time_l25_2532

theorem news_spread_time (n : ℕ) (m : ℕ) :
  (2^m < n ∧ n < 2^(m+k+1) ∧ (n % 2 = 1) ∧ n % 2 = 1) →
  ∃ t : ℕ, t = (if n % 2 = 1 then m+2 else m+1) := 
sorry

end news_spread_time_l25_2532


namespace line_does_not_pass_second_quadrant_l25_2550

theorem line_does_not_pass_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0 → ¬(x < 0 ∧ y > 0)) ↔ a ≤ -1 :=
by
  sorry

end line_does_not_pass_second_quadrant_l25_2550


namespace ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l25_2506

theorem ln_sqrt2_lt_sqrt2_div2 : Real.log (Real.sqrt 2) < Real.sqrt 2 / 2 :=
sorry

theorem ln_sin_cos_sum : 2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1 / 4 :=
sorry

end ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l25_2506


namespace total_seconds_eq_250200_l25_2596

def bianca_hours : ℝ := 12.5
def celeste_hours : ℝ := 2 * bianca_hours
def mcclain_hours : ℝ := celeste_hours - 8.5
def omar_hours : ℝ := bianca_hours + 3

def total_hours : ℝ := bianca_hours + celeste_hours + mcclain_hours + omar_hours
def hour_to_seconds : ℝ := 3600
def total_seconds : ℝ := total_hours * hour_to_seconds

theorem total_seconds_eq_250200 : total_seconds = 250200 := by
  sorry

end total_seconds_eq_250200_l25_2596


namespace find_common_difference_find_possible_a1_l25_2511

structure ArithSeq :=
  (a : ℕ → ℤ) -- defining the sequence
  
noncomputable def S (n : ℕ) (a : ArithSeq) : ℤ :=
  (n * (2 * a.a 0 + (n - 1) * (a.a 1 - a.a 0))) / 2

axiom a4 (a : ArithSeq) : a.a 3 = 10

axiom S20 (a : ArithSeq) : S 20 a = 590

theorem find_common_difference (a : ArithSeq) (d : ℤ) : 
  (a.a 1 - a.a 0 = d) →
  d = 3 :=
sorry

theorem find_possible_a1 (a : ArithSeq) : 
  (∃a1: ℤ, a1 ∈ Set.range a.a) →
  (∀n : ℕ, S n a ≤ S 7 a) →
  Set.range a.a ∩ {n | 18 ≤ n ∧ n ≤ 20} = {18, 19, 20} :=
sorry

end find_common_difference_find_possible_a1_l25_2511


namespace silvia_order_total_cost_l25_2598

theorem silvia_order_total_cost :
  let quiche_price : ℝ := 15
  let croissant_price : ℝ := 3
  let biscuit_price : ℝ := 2
  let quiche_count : ℝ := 2
  let croissant_count : ℝ := 6
  let biscuit_count : ℝ := 6
  let discount_rate : ℝ := 0.10
  let pre_discount_total : ℝ := (quiche_price * quiche_count) + (croissant_price * croissant_count) + (biscuit_price * biscuit_count)
  let discount_amount : ℝ := pre_discount_total * discount_rate
  let post_discount_total : ℝ := pre_discount_total - discount_amount
  pre_discount_total > 50 → post_discount_total = 54 :=
by
  sorry

end silvia_order_total_cost_l25_2598


namespace range_of_a_l25_2558

def line_intersects_circle (a : ℝ) : Prop :=
  let distance_from_center_to_line := |1 - a| / Real.sqrt 2
  distance_from_center_to_line ≤ Real.sqrt 2

theorem range_of_a :
  {a : ℝ | line_intersects_circle a} = {a : ℝ | -1 ≤ a ∧ a ≤ 3} :=
by
  sorry

end range_of_a_l25_2558


namespace cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l25_2580

-- Problem 1
theorem cross_fraction_eq1 (x : ℝ) : (x + 12 / x = -7) → 
  ∃ (x₁ x₂ : ℝ), (x₁ = -3 ∧ x₂ = -4 ∧ x = x₁ ∨ x = x₂) :=
sorry

-- Problem 2
theorem cross_fraction_eq2 (a b : ℝ) 
    (h1 : a * b = -6) 
    (h2 : a + b = -5) : (a ≠ 0 ∧ b ≠ 0) →
    (b / a + a / b + 1 = -31 / 6) :=
sorry

-- Problem 3
theorem cross_fraction_eq3 (k x₁ x₂ : ℝ)
    (hk : k > 2)
    (hx1 : x₁ = 2022 * k - 2022)
    (hx2 : x₂ = k + 1) :
    (x₁ > x₂) →
    (x₁ + 4044) / x₂ = 2022 :=
sorry

end cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l25_2580


namespace apples_in_each_bag_l25_2586

variable (x : ℕ)
variable (total_children : ℕ)
variable (eaten_apples : ℕ)
variable (sold_apples : ℕ)
variable (remaining_apples : ℕ)

theorem apples_in_each_bag
  (h1 : total_children = 5)
  (h2 : eaten_apples = 2 * 4)
  (h3 : sold_apples = 7)
  (h4 : remaining_apples = 60)
  (h5 : total_children * x - eaten_apples - sold_apples = remaining_apples) :
  x = 15 :=
by
  sorry

end apples_in_each_bag_l25_2586


namespace intersection_A_B_l25_2561

-- Define the sets A and B based on the given conditions
def A := { x : ℝ | (1 / 9) ≤ (3:ℝ)^x ∧ (3:ℝ)^x ≤ 1 }
def B := { x : ℝ | x^2 < 1 }

-- State the theorem for the intersection of sets A and B
theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x ≤ 0 } :=
by
  sorry

end intersection_A_B_l25_2561


namespace find_smaller_number_l25_2520

theorem find_smaller_number (x y : ℕ) (h1 : x = 2 * y - 3) (h2 : x + y = 51) : y = 18 :=
sorry

end find_smaller_number_l25_2520


namespace inequality_proof_l25_2587

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / Real.sqrt (a^2 + 8 * b * c)) + 
  (b / Real.sqrt (b^2 + 8 * c * a)) + 
  (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proof_l25_2587


namespace find_two_digit_number_l25_2551

theorem find_two_digit_number (n s p : ℕ) (h1 : n = 4 * s) (h2 : n = 3 * p) : n = 24 := 
  sorry

end find_two_digit_number_l25_2551


namespace average_interest_rate_l25_2524

theorem average_interest_rate 
  (total : ℝ)
  (rate1 rate2 yield1 yield2 : ℝ)
  (amount1 amount2 : ℝ)
  (h_total : total = amount1 + amount2)
  (h_rate1 : rate1 = 0.03)
  (h_rate2 : rate2 = 0.07)
  (h_yield_equal : yield1 = yield2)
  (h_yield1 : yield1 = rate1 * amount1)
  (h_yield2 : yield2 = rate2 * amount2) :
  (yield1 + yield2) / total = 0.042 :=
by
  sorry

end average_interest_rate_l25_2524


namespace problem_statement_l25_2503

theorem problem_statement (a b c d n : Nat) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < n) (h_eq : 7 * 4^n = a^2 + b^2 + c^2 + d^2) : 
  a ≥ 2^(n-1) ∧ b ≥ 2^(n-1) ∧ c ≥ 2^(n-1) ∧ d ≥ 2^(n-1) :=
sorry

end problem_statement_l25_2503


namespace units_digit_7_pow_103_l25_2509

theorem units_digit_7_pow_103 : Nat.mod (7 ^ 103) 10 = 3 := sorry

end units_digit_7_pow_103_l25_2509


namespace g_2002_value_l25_2523

noncomputable def g : ℕ → ℤ := sorry

theorem g_2002_value :
  (∀ a b n : ℕ, a + b = 2^n → g a + g b = n^3) →
  (g 2 + g 46 = 180) →
  g 2002 = 1126 := 
by
  intros h1 h2
  sorry

end g_2002_value_l25_2523


namespace hockey_games_in_season_l25_2554

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

end hockey_games_in_season_l25_2554


namespace income_is_20000_l25_2513

-- Definitions from conditions
def income (x : ℕ) : ℕ := 4 * x
def expenditure (x : ℕ) : ℕ := 3 * x
def savings : ℕ := 5000

-- Theorem to prove the income
theorem income_is_20000 (x : ℕ) (h : income x - expenditure x = savings) : income x = 20000 :=
by
  sorry

end income_is_20000_l25_2513


namespace area_difference_of_squares_l25_2502

theorem area_difference_of_squares (d1 d2 : ℝ) (h1 : d1 = 19) (h2 : d2 = 17) : 
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let area1 := s1 * s1
  let area2 := s2 * s2
  (area1 - area2) = 36 :=
by
  sorry

end area_difference_of_squares_l25_2502


namespace return_trip_time_is_15_or_67_l25_2514

variable (d p w : ℝ)

-- Conditions
axiom h1 : (d / (p - w)) = 100
axiom h2 : ∃ t : ℝ, t = d / p ∧ (d / (p + w)) = t - 15

-- Correct answer to prove: time for the return trip is 15 minutes or 67 minutes
theorem return_trip_time_is_15_or_67 : (d / (p + w)) = 15 ∨ (d / (p + w)) = 67 := 
by 
  sorry

end return_trip_time_is_15_or_67_l25_2514


namespace find_x_l25_2569

theorem find_x (x : ℝ) 
  (a : ℝ × ℝ := (2*x - 1, x + 3)) 
  (b : ℝ × ℝ := (x, 2*x + 1))
  (c : ℝ × ℝ := (1, 2))
  (h : (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0) :
  x = 3 :=
  sorry

end find_x_l25_2569


namespace range_of_f_l25_2573

def f (x : Int) : Int :=
  x + 1

def domain : Set Int :=
  {-1, 1, 2}

theorem range_of_f :
  Set.image f domain = {0, 2, 3} :=
by
  sorry

end range_of_f_l25_2573


namespace border_area_is_198_l25_2557

-- We define the dimensions of the picture and the border width
def picture_height : ℝ := 12
def picture_width : ℝ := 15
def border_width : ℝ := 3

-- We compute the entire framed height and width
def framed_height : ℝ := picture_height + 2 * border_width
def framed_width : ℝ := picture_width + 2 * border_width

-- We compute the area of the picture and framed area
def picture_area : ℝ := picture_height * picture_width
def framed_area : ℝ := framed_height * framed_width

-- We compute the area of the border
def border_area : ℝ := framed_area - picture_area

-- Now we pose the theorem to prove the area of the border is 198 square inches
theorem border_area_is_198 : border_area = 198 := by
  sorry

end border_area_is_198_l25_2557


namespace algebraic_expression_value_l25_2549

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = -2) 
  (h2 : 2 * x + y = -1) : 
  (x - y)^2 - (x - 2 * y) * (x + 2 * y) = 7 :=
by {
  sorry
}

end algebraic_expression_value_l25_2549


namespace intersection_of_sets_l25_2530

-- Define the sets M and N
def M : Set ℝ := { x | 2 < x ∧ x < 3 }
def N : Set ℝ := { x | 2 < x ∧ x ≤ 5 / 2 }

-- State the theorem to prove
theorem intersection_of_sets : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by 
  sorry

end intersection_of_sets_l25_2530


namespace binary_division_correct_l25_2546

def b1100101 := 0b1100101
def b1101 := 0b1101
def b101 := 0b101
def expected_result := 0b11111010

theorem binary_division_correct : ((b1100101 * b1101) / b101) = expected_result :=
by {
  sorry
}

end binary_division_correct_l25_2546


namespace r_minus_p_value_l25_2518

theorem r_minus_p_value (p q r : ℝ)
  (h₁ : (p + q) / 2 = 10)
  (h₂ : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end r_minus_p_value_l25_2518


namespace quadratic_solution_eq_l25_2542

noncomputable def p : ℝ :=
  (8 + Real.sqrt 364) / 10

noncomputable def q : ℝ :=
  (8 - Real.sqrt 364) / 10

theorem quadratic_solution_eq (p q : ℝ) (h₁ : 5 * p^2 - 8 * p - 15 = 0) (h₂ : 5 * q^2 - 8 * q - 15 = 0) : 
  (p - q) ^ 2 = 14.5924 :=
sorry

end quadratic_solution_eq_l25_2542


namespace kids_waiting_for_swings_l25_2541

theorem kids_waiting_for_swings (x : ℕ) (h1 : 2 * 60 = 120) 
  (h2 : ∀ y, y = 2 → (y * x = 2 * x)) 
  (h3 : 15 * (2 * x) = 30 * x)
  (h4 : 120 * x - 30 * x = 270) : x = 3 :=
sorry

end kids_waiting_for_swings_l25_2541


namespace ascending_order_l25_2539

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb1 : -1 < b) (hb2 : b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end ascending_order_l25_2539


namespace line_is_tangent_to_circle_l25_2597

theorem line_is_tangent_to_circle
  (θ : Real)
  (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop)
  (h_l : ∀ x y, l x y ↔ x * Real.sin θ + 2 * y * Real.cos θ = 1)
  (h_C : ∀ x y, C x y ↔ x^2 + y^2 = 1) :
  (∀ x y, l x y ↔ x = 1 ∨ x = -1) ↔
  (∃ x y, C x y ∧ ∀ x y, l x y → Real.sqrt ((x * Real.sin θ + 2 * y * Real.cos θ - 1)^2 / (Real.sin θ^2 + 4 * Real.cos θ^2)) = 1) :=
sorry

end line_is_tangent_to_circle_l25_2597


namespace john_total_replacement_cost_l25_2540

def cost_to_replace_all_doors
  (num_bedroom_doors : ℕ)
  (num_outside_doors : ℕ)
  (cost_outside_door : ℕ)
  (cost_bedroom_door : ℕ) : ℕ :=
  let total_cost_outside_doors := num_outside_doors * cost_outside_door
  let total_cost_bedroom_doors := num_bedroom_doors * cost_bedroom_door
  total_cost_outside_doors + total_cost_bedroom_doors

theorem john_total_replacement_cost :
  let num_bedroom_doors := 3
  let num_outside_doors := 2
  let cost_outside_door := 20
  let cost_bedroom_door := cost_outside_door / 2
  cost_to_replace_all_doors num_bedroom_doors num_outside_doors cost_outside_door cost_bedroom_door = 70 := by
  sorry

end john_total_replacement_cost_l25_2540


namespace point_in_second_quadrant_l25_2590

variable (m : ℝ)

-- Defining the conditions
def x_negative (m : ℝ) := 3 - m < 0
def y_positive (m : ℝ) := m - 1 > 0

theorem point_in_second_quadrant (h1 : x_negative m) (h2 : y_positive m) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l25_2590


namespace expected_balls_in_original_position_proof_l25_2568

-- Define the problem conditions as Lean definitions
def n_balls : ℕ := 10

def probability_not_moved_by_one_rotation : ℚ := 7 / 10

def probability_not_moved_by_two_rotations : ℚ := (7 / 10) * (7 / 10)

def expected_balls_in_original_position : ℚ := n_balls * probability_not_moved_by_two_rotations

-- The statement representing the proof problem
theorem expected_balls_in_original_position_proof :
  expected_balls_in_original_position = 4.9 :=
  sorry

end expected_balls_in_original_position_proof_l25_2568


namespace majka_numbers_product_l25_2508

/-- Majka created a three-digit funny and a three-digit cheerful number.
    - The funny number starts with an odd digit and alternates between odd and even.
    - The cheerful number starts with an even digit and alternates between even and odd.
    - All digits are distinct and nonzero.
    - The sum of these two numbers is 1617.
    - The product of these two numbers ends in 40.
    Prove that the product of these numbers is 635040.
-/
theorem majka_numbers_product :
  ∃ (a b c : ℕ) (D E F : ℕ),
    -- Define 3-digit funny number as (100 * a + 10 * b + c)
    -- with a and c odd, b even, and all distinct and nonzero
    (a % 2 = 1) ∧ (c % 2 = 1) ∧ (b % 2 = 0) ∧
    -- Define 3-digit cheerful number as (100 * D + 10 * E + F)
    -- with D and F even, E odd, and all distinct and nonzero
    (D % 2 = 0) ∧ (F % 2 = 0) ∧ (E % 2 = 1) ∧
    -- All digits are distinct and nonzero
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ D ∧ a ≠ E ∧ a ≠ F ∧
     b ≠ c ∧ b ≠ D ∧ b ≠ E ∧ b ≠ F ∧
     c ≠ D ∧ c ≠ E ∧ c ≠ F ∧
     D ≠ E ∧ D ≠ F ∧ E ≠ F) ∧
    (100 * a + 10 * b + c + 100 * D + 10 * E + F = 1617) ∧
    ((100 * a + 10 * b + c) * (100 * D + 10 * E + F) = 635040) := sorry

end majka_numbers_product_l25_2508


namespace price_of_rice_packet_l25_2582

-- Definitions based on conditions
def initial_amount : ℕ := 500
def wheat_flour_price : ℕ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℕ := 150
def remaining_balance : ℕ := 235
def total_spending (P : ℕ) : ℕ := initial_amount - remaining_balance

-- Theorem to prove
theorem price_of_rice_packet (P : ℕ) (h: 2 * P + wheat_flour_quantity * wheat_flour_price + soda_price = total_spending P) : P = 20 :=
sorry

end price_of_rice_packet_l25_2582


namespace fraction_simplification_l25_2588

theorem fraction_simplification :
  (3 / (2 - (3 / 4))) = 12 / 5 := 
by
  sorry

end fraction_simplification_l25_2588


namespace relationship_withdrawn_leftover_l25_2500

-- Definitions based on the problem conditions
def pie_cost : ℝ := 6
def sandwich_cost : ℝ := 3
def book_cost : ℝ := 10
def book_discount : ℝ := 0.2 * book_cost
def book_price_with_discount : ℝ := book_cost - book_discount
def total_spent_before_tax : ℝ := pie_cost + sandwich_cost + book_price_with_discount
def sales_tax_rate : ℝ := 0.05
def sales_tax : ℝ := sales_tax_rate * total_spent_before_tax
def total_spent_with_tax : ℝ := total_spent_before_tax + sales_tax

-- Given amount withdrawn and amount left after shopping
variables (X Y : ℝ)

-- Theorem statement
theorem relationship_withdrawn_leftover :
  Y = X - total_spent_with_tax :=
sorry

end relationship_withdrawn_leftover_l25_2500


namespace sin_double_angle_of_tangent_l25_2553

theorem sin_double_angle_of_tangent (α : ℝ) (h : Real.tan (π + α) = 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_of_tangent_l25_2553


namespace matrix_addition_l25_2599

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![-1, 2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 4], ![1, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, -1]]

theorem matrix_addition : A + B = C := by
    sorry

end matrix_addition_l25_2599


namespace average_sqft_per_person_texas_l25_2527

theorem average_sqft_per_person_texas :
  let population := 17000000
  let area_sqmiles := 268596
  let usable_land_percentage := 0.8
  let sqfeet_per_sqmile := 5280 * 5280
  let total_sqfeet := area_sqmiles * sqfeet_per_sqmile
  let usable_sqfeet := usable_land_percentage * total_sqfeet
  let avg_sqfeet_per_person := usable_sqfeet / population
  352331 <= avg_sqfeet_per_person ∧ avg_sqfeet_per_person < 500000 :=
by
  sorry

end average_sqft_per_person_texas_l25_2527


namespace problem1_asymptotes_problem2_equation_l25_2559

-- Problem 1: Asymptotes of a hyperbola
theorem problem1_asymptotes (a : ℝ) (x y : ℝ) (hx : (y + a) ^ 2 - (x - a) ^ 2 = 2 * a)
  (hpt : 3 = x ∧ 1 = y) : 
  (y = x - 2 * a) ∨ (y = - x) := 
by 
  sorry

-- Problem 2: Equation of a hyperbola
theorem problem2_equation (a b c : ℝ) (x y : ℝ) 
  (hasymptote : y = x + 1 ∨ y = - (x + 1))  (hfocal : 2 * c = 4)
  (hc_squared : c ^ 2 = a ^ 2 + b ^ 2) (ha_eq_b : a = b): 
  y^2 - (x + 1)^2 = 2 := 
by 
  sorry

end problem1_asymptotes_problem2_equation_l25_2559


namespace hyperbola_focal_distance_distance_focus_to_asymptote_l25_2556

theorem hyperbola_focal_distance :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  (2 * c = 4) :=
by sorry

theorem distance_focus_to_asymptote :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let focus := (c, 0)
  let A := -Real.sqrt 3
  let B := 1
  let C := 0
  let distance := (|A * focus.fst + B * focus.snd + C|) / Real.sqrt (A ^ 2 + B ^ 2)
  (distance = Real.sqrt 3) :=
by sorry

end hyperbola_focal_distance_distance_focus_to_asymptote_l25_2556


namespace smallest_m_plus_n_l25_2516

theorem smallest_m_plus_n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_lt : m < n)
    (h_eq : 1978^m % 1000 = 1978^n % 1000) : m + n = 26 :=
sorry

end smallest_m_plus_n_l25_2516


namespace device_records_720_instances_in_one_hour_l25_2515

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l25_2515


namespace box_volume_l25_2563

theorem box_volume
  (l w h : ℝ)
  (A1 : l * w = 36)
  (A2 : w * h = 18)
  (A3 : l * h = 8) :
  l * w * h = 102 := 
sorry

end box_volume_l25_2563


namespace cost_per_metre_of_carpet_l25_2525

theorem cost_per_metre_of_carpet :
  (length_of_room = 18) →
  (breadth_of_room = 7.5) →
  (carpet_width = 0.75) →
  (total_cost = 810) →
  (cost_per_metre = 4.5) :=
by
  intros length_of_room breadth_of_room carpet_width total_cost
  sorry

end cost_per_metre_of_carpet_l25_2525


namespace cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l25_2593

/-- 
Vasiliy has 2019 coins, one of which is counterfeit (differing in weight). 
Using balance scales without weights and immediately paying out identified genuine coins, 
it is impossible to determine whether the counterfeit coin is lighter or heavier.
-/
theorem cannot_determine_if_counterfeit_coin_is_lighter_or_heavier 
  (num_coins : ℕ)
  (num_counterfeit : ℕ)
  (balance_scale : Bool → Bool → Bool)
  (immediate_payment : Bool → Bool) :
  num_coins = 2019 →
  num_counterfeit = 1 →
  (∀ coins_w1 coins_w2, balance_scale coins_w1 coins_w2 = (coins_w1 = coins_w2)) →
  (∀ coin_p coin_q, (immediate_payment coin_p = true) → ¬ coin_p = coin_q) →
  ¬ ∃ (is_lighter_or_heavier : Bool), true :=
by
  intro h1 h2 h3 h4
  sorry

end cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l25_2593


namespace monotonic_increase_l25_2579

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2)

theorem monotonic_increase : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 < f x2 :=
by
  sorry

end monotonic_increase_l25_2579


namespace cupboard_slots_l25_2536

theorem cupboard_slots (shelves_from_top shelves_from_bottom slots_from_left slots_from_right : ℕ)
  (h_top : shelves_from_top = 1)
  (h_bottom : shelves_from_bottom = 3)
  (h_left : slots_from_left = 0)
  (h_right : slots_from_right = 6) :
  (shelves_from_top + 1 + shelves_from_bottom) * (slots_from_left + 1 + slots_from_right) = 35 := by
  sorry

end cupboard_slots_l25_2536


namespace range_of_slope_angle_l25_2562

theorem range_of_slope_angle (l : ℝ → ℝ) (theta : ℝ) 
    (h_line_eqn : ∀ x y, l x = y ↔ x - y * Real.sin theta + 2 = 0) : 
    ∃ α : ℝ, α ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) :=
sorry

end range_of_slope_angle_l25_2562


namespace ratio_of_female_to_male_officers_on_duty_l25_2572

theorem ratio_of_female_to_male_officers_on_duty 
    (p : ℝ) (T : ℕ) (F : ℕ) 
    (hp : p = 0.19) (hT : T = 152) (hF : F = 400) : 
    (76 / 76) = 1 :=
by
  sorry

end ratio_of_female_to_male_officers_on_duty_l25_2572


namespace AndrewAge_l25_2504

noncomputable def AndrewAgeProof : Prop :=
  ∃ (a g : ℕ), g = 10 * a ∧ g - a = 45 ∧ a = 5

-- Proof is not required, so we use sorry to skip the proof.
theorem AndrewAge : AndrewAgeProof := by
  sorry

end AndrewAge_l25_2504


namespace parametric_curve_intersects_l25_2595

noncomputable def curve_crosses_itself : Prop :=
  let t1 := Real.sqrt 11
  let t2 := -Real.sqrt 11
  let x (t : ℝ) := t^3 - t + 1
  let y (t : ℝ) := t^3 - 11*t + 11
  (x t1 = 10 * Real.sqrt 11 + 1) ∧ (y t1 = 11) ∧
  (x t2 = 10 * Real.sqrt 11 + 1) ∧ (y t2 = 11)

theorem parametric_curve_intersects : curve_crosses_itself :=
by
  sorry

end parametric_curve_intersects_l25_2595


namespace add_sub_decimals_l25_2583

theorem add_sub_decimals :
  (0.513 + 0.0067 - 0.048 = 0.4717) :=
by
  sorry

end add_sub_decimals_l25_2583


namespace intersection_of_S_and_T_l25_2531

-- Define S and T based on given conditions
def S : Set ℝ := { x | x^2 + 2 * x = 0 }
def T : Set ℝ := { x | x^2 - 2 * x = 0 }

-- Prove the intersection of S and T
theorem intersection_of_S_and_T : S ∩ T = {0} :=
sorry

end intersection_of_S_and_T_l25_2531


namespace real_solutions_to_system_l25_2517

theorem real_solutions_to_system :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x y z w : ℝ), 
    (x = z + w + 2*z*w*x) ∧ 
    (y = w + x + 2*w*x*y) ∧ 
    (z = x + y + 2*x*y*z) ∧ 
    (w = y + z + 2*y*z*w) ↔ 
    (x, y, z, w) ∈ s) ∧
    (s.card = 15) :=
sorry

end real_solutions_to_system_l25_2517


namespace line_equation_l25_2555

theorem line_equation (x y : ℝ) : 
  (∃ (m c : ℝ), m = 3 ∧ c = 4 ∧ y = m * x + c) ↔ 3 * x - y + 4 = 0 := by
  sorry

end line_equation_l25_2555


namespace luke_total_points_l25_2565

theorem luke_total_points (rounds : ℕ) (points_per_round : ℕ) (total_points : ℕ) 
  (h1 : rounds = 177) (h2 : points_per_round = 46) : 
  total_points = 8142 := by
  have h : total_points = rounds * points_per_round := by sorry
  rw [h1, h2] at h
  exact h

end luke_total_points_l25_2565


namespace find_number_x_l25_2564

theorem find_number_x (x : ℝ) (h : 2500 - x / 20.04 = 2450) : x = 1002 :=
by
  -- Proof can be written here, but skipped by using sorry
  sorry

end find_number_x_l25_2564


namespace ground_beef_sold_ratio_l25_2560

variable (beef_sold_Thursday : ℕ) (beef_sold_Saturday : ℕ) (avg_sold_per_day : ℕ) (days : ℕ)

theorem ground_beef_sold_ratio (h₁ : beef_sold_Thursday = 210)
                             (h₂ : beef_sold_Saturday = 150)
                             (h₃ : avg_sold_per_day = 260)
                             (h₄ : days = 3) :
  let total_sold := avg_sold_per_day * days
  let beef_sold_Friday := total_sold - beef_sold_Thursday - beef_sold_Saturday
  (beef_sold_Friday : ℕ) / (beef_sold_Thursday : ℕ) = 2 := by
  sorry

end ground_beef_sold_ratio_l25_2560


namespace find_c_l25_2547

theorem find_c (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) : c = (n * a) / (n - 2 * a * b) :=
by
  sorry

end find_c_l25_2547


namespace mass_percentage_ca_in_compound_l25_2522

noncomputable def mass_percentage_ca_in_cac03 : ℝ :=
  let mm_ca := 40.08
  let mm_c := 12.01
  let mm_o := 16.00
  let mm_caco3 := mm_ca + mm_c + 3 * mm_o
  (mm_ca / mm_caco3) * 100

theorem mass_percentage_ca_in_compound (mp : ℝ) (h : mp = mass_percentage_ca_in_cac03) : mp = 40.04 := by
  sorry

end mass_percentage_ca_in_compound_l25_2522


namespace three_Z_five_l25_2575

def Z (a b : ℤ) : ℤ := b + 7 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end three_Z_five_l25_2575


namespace roots_quadratic_sum_product_l25_2534

theorem roots_quadratic_sum_product :
  (∀ x1 x2 : ℝ, (∀ x, x^2 - 4 * x + 3 = 0 → x = x1 ∨ x = x2) → (x1 + x2 - x1 * x2 = 1)) :=
by
  sorry

end roots_quadratic_sum_product_l25_2534


namespace emily_points_l25_2545

theorem emily_points (r1 r2 r3 r4 r5 m4 m5 l : ℤ)
  (h1 : r1 = 16)
  (h2 : r2 = 33)
  (h3 : r3 = 21)
  (h4 : r4 = 10)
  (h5 : r5 = 4)
  (hm4 : m4 = 2)
  (hm5 : m5 = 3)
  (hl : l = 48) :
  r1 + r2 + r3 + r4 * m4 + r5 * m5 - l = 54 := by
  sorry

end emily_points_l25_2545


namespace GouguPrinciple_l25_2574

-- Definitions according to conditions
def volumes_not_equal (A B : Type) : Prop := sorry -- p: volumes of A and B are not equal
def cross_sections_not_equal (A B : Type) : Prop := sorry -- q: cross-sectional areas of A and B are not always equal

-- The theorem to be proven
theorem GouguPrinciple (A B : Type) (h1 : volumes_not_equal A B) : cross_sections_not_equal A B :=
sorry

end GouguPrinciple_l25_2574


namespace arithmetic_sequence_30th_term_l25_2529

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l25_2529


namespace triangle_inequality_difference_l25_2519

theorem triangle_inequality_difference :
  ∀ (x : ℕ), (x + 8 > 10) → (x + 10 > 8) → (8 + 10 > x) →
    (17 - 3 = 14) :=
by
  intros x hx1 hx2 hx3
  sorry

end triangle_inequality_difference_l25_2519


namespace domain_h_l25_2589

noncomputable def h (x : ℝ) : ℝ := (3 * x - 1) / Real.sqrt (x - 5)

theorem domain_h (x : ℝ) : h x = (3 * x - 1) / Real.sqrt (x - 5) → (x > 5) :=
by
  intro hx
  have hx_nonneg : x - 5 >= 0 := sorry
  have sqrt_nonzero : Real.sqrt (x - 5) ≠ 0 := sorry
  sorry

end domain_h_l25_2589


namespace evaluate_g_at_4_l25_2543

def g (x : ℕ) := 5 * x + 2

theorem evaluate_g_at_4 : g 4 = 22 := by
  sorry

end evaluate_g_at_4_l25_2543


namespace largest_value_of_n_l25_2592

noncomputable def largest_n_under_200000 : ℕ :=
  if h : 199999 < 200000 ∧ (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 then 199999 else 0

theorem largest_value_of_n (n : ℕ) :
  n < 200000 → (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0 → n = 199999 :=
by sorry

end largest_value_of_n_l25_2592


namespace cat_mouse_position_258_l25_2576

-- Define the cycle positions for the cat
def cat_position (n : ℕ) : String :=
  match n % 4 with
  | 0 => "top left"
  | 1 => "top right"
  | 2 => "bottom right"
  | _ => "bottom left"

-- Define the cycle positions for the mouse
def mouse_position (n : ℕ) : String :=
  match n % 8 with
  | 0 => "top middle"
  | 1 => "top right"
  | 2 => "right middle"
  | 3 => "bottom right"
  | 4 => "bottom middle"
  | 5 => "bottom left"
  | 6 => "left middle"
  | _ => "top left"

theorem cat_mouse_position_258 : 
  cat_position 258 = "top right" ∧ mouse_position 258 = "top right" := by
  sorry

end cat_mouse_position_258_l25_2576


namespace find_x_l25_2501

-- Define the known values
def a := 6
def b := 16
def c := 8
def desired_average := 13

-- Define the target number we need to find
def target_x := 22

-- Prove that the number we need to add to get the desired average is 22
theorem find_x : (a + b + c + target_x) / 4 = desired_average :=
by
  -- The proof itself is omitted as per instructions
  sorry

end find_x_l25_2501


namespace sum_is_220_l25_2567

def second_number := 60
def first_number := 2 * second_number
def third_number := first_number / 3
def sum_of_numbers := first_number + second_number + third_number

theorem sum_is_220 : sum_of_numbers = 220 :=
by
  sorry

end sum_is_220_l25_2567


namespace compare_sums_of_sines_l25_2581

theorem compare_sums_of_sines {A B C : ℝ} 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π) :
  (if A < π / 2 ∧ B < π / 2 ∧ C < π / 2 then
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      ≥ 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))
  else
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      < 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))) :=
sorry

end compare_sums_of_sines_l25_2581


namespace perpendicular_lines_solve_a_l25_2577

theorem perpendicular_lines_solve_a (a : ℝ) :
  (3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0 → a = 0 ∨ a = 12 / 11 :=
by 
  sorry

end perpendicular_lines_solve_a_l25_2577


namespace sin_pow_cos_pow_sum_l25_2594

namespace ProofProblem

-- Define the condition
def trig_condition (x : ℝ) : Prop :=
  3 * (Real.sin x)^3 + (Real.cos x)^3 = 3

-- State the theorem
theorem sin_pow_cos_pow_sum (x : ℝ) (h : trig_condition x) : Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 :=
by
  sorry

end ProofProblem

end sin_pow_cos_pow_sum_l25_2594


namespace hillside_camp_boys_percentage_l25_2526

theorem hillside_camp_boys_percentage (B G : ℕ) 
  (h1 : B + G = 60) 
  (h2 : G = 6) : (B: ℕ) / 60 * 100 = 90 :=
by
  sorry

end hillside_camp_boys_percentage_l25_2526


namespace abs_x_plus_abs_y_eq_one_area_l25_2578

theorem abs_x_plus_abs_y_eq_one_area : 
  (∃ (A : ℝ), ∀ (x y : ℝ), |x| + |y| = 1 → A = 2) :=
sorry

end abs_x_plus_abs_y_eq_one_area_l25_2578


namespace tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l25_2584
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem tangent_line_through_origin (x y : ℝ) :
  (∃ a : ℝ, (x, y) = (a, f a) ∧ (0, 0) = (0, 0) ∧ y - f a = ((2 * a - a^2) / Real.exp a) * (x - a)) →
  y = x / Real.exp 1 :=
sorry

theorem max_value_on_interval : ∃ (x : ℝ), x = 9 / Real.exp 3 :=
  sorry

theorem min_value_on_interval : ∃ (x : ℝ), x = 0 :=
  sorry

end tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l25_2584


namespace amount_returned_l25_2533

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end amount_returned_l25_2533


namespace find_distance_l25_2505

-- Definitions based on the given conditions
def speed_of_boat := 16 -- in kmph
def speed_of_stream := 2 -- in kmph
def total_time := 960 -- in hours
def downstream_speed := speed_of_boat + speed_of_stream
def upstream_speed := speed_of_boat - speed_of_stream

-- Prove that the distance D is 7590 km given the total time and speeds
theorem find_distance (D : ℝ) :
  (D / downstream_speed + D / upstream_speed = total_time) → D = 7590 :=
by
  sorry

end find_distance_l25_2505


namespace line_segments_cannot_form_triangle_l25_2538

theorem line_segments_cannot_form_triangle (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 7 = 21)
    (h3 : ∀ n, a n < a (n+1)) (h4 : ∀ i j k, a i + a j ≤ a k) :
    a 6 = 13 :=
    sorry

end line_segments_cannot_form_triangle_l25_2538


namespace probability_point_in_sphere_eq_2pi_div_3_l25_2570

open Real Topology

noncomputable def volume_of_region := 4 * 2 * 2

noncomputable def volume_of_sphere_radius_2 : ℝ :=
  (4 / 3) * π * (2 ^ 3)

noncomputable def probability_in_sphere : ℝ :=
  volume_of_sphere_radius_2 / volume_of_region

theorem probability_point_in_sphere_eq_2pi_div_3 :
  probability_in_sphere = (2 * π) / 3 :=
by
  sorry

end probability_point_in_sphere_eq_2pi_div_3_l25_2570


namespace original_number_of_laborers_l25_2566

theorem original_number_of_laborers 
(L : ℕ) (h1 : L * 15 = (L - 5) * 20) : L = 15 :=
sorry

end original_number_of_laborers_l25_2566


namespace find_f1_plus_g1_l25_2548

variable (f g : ℝ → ℝ)

-- Conditions
def even_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = h (-x)
def odd_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = -h (-x)
def function_relation := ∀ x : ℝ, f x - g x = x^3 + x^2 + 1

-- Mathematically equivalent proof problem
theorem find_f1_plus_g1
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (h_relation : function_relation f g) :
  f 1 + g 1 = 1 := by
  sorry

end find_f1_plus_g1_l25_2548


namespace peanut_butter_candy_count_l25_2521

theorem peanut_butter_candy_count (B G P : ℕ) 
  (hB : B = 43)
  (hG : G = B + 5)
  (hP : P = 4 * G) :
  P = 192 := by
  sorry

end peanut_butter_candy_count_l25_2521
