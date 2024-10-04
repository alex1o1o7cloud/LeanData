import Mathlib

namespace handshakes_minimum_l276_276766

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l276_276766


namespace number_divisible_by_11_l276_276051

theorem number_divisible_by_11 (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by
  sorry

end number_divisible_by_11_l276_276051


namespace num_ways_to_tile_3x5_is_40_l276_276926

-- Definition of the problem
def numTilings (tiles : List (ℕ × ℕ)) (m n : ℕ) : ℕ :=
  sorry -- Placeholder for actual tiling computation

-- Condition specific to this problem
def specificTiles : List (ℕ × ℕ) :=
  [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

-- Problem statement in Lean 4
theorem num_ways_to_tile_3x5_is_40 :
  numTilings specificTiles 3 5 = 40 :=
sorry

end num_ways_to_tile_3x5_is_40_l276_276926


namespace ratio_of_dinner_to_lunch_l276_276449

theorem ratio_of_dinner_to_lunch
  (dinner: ℕ) (lunch: ℕ) (breakfast: ℕ) (k: ℕ)
  (h1: dinner = 240)
  (h2: dinner = k * lunch)
  (h3: dinner = 6 * breakfast)
  (h4: breakfast + lunch + dinner = 310) :
  dinner / lunch = 8 :=
by
  -- Proof to be completed
  sorry

end ratio_of_dinner_to_lunch_l276_276449


namespace number_of_girls_l276_276181

theorem number_of_girls (total_students : ℕ) (sample_size : ℕ) (girls_sampled_minus : ℕ) (girls_sampled_ratio : ℚ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_sampled_minus = 20 →
  girls_sampled_ratio = 90 / 200 →
  (∃ x, x / (total_students : ℚ) = girls_sampled_ratio ∧ x = 720) :=
by intros _ _ _ _; sorry

end number_of_girls_l276_276181


namespace average_age_new_students_l276_276539

theorem average_age_new_students (O A_old A_new_avg A_new : ℕ) 
  (hO : O = 8) 
  (hA_old : A_old = 40) 
  (hA_new_avg : A_new_avg = 36)
  (h_total_age_before : O * A_old = 8 * 40)
  (h_total_age_after : (O + 8) * A_new_avg = 16 * 36)
  (h_age_new_students : (16 * 36) - (8 * 40) = A_new * 8) :
  A_new = 32 := 
by 
  sorry

end average_age_new_students_l276_276539


namespace average_screen_time_per_player_l276_276862

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l276_276862


namespace largest_divisible_by_6_ending_in_4_l276_276721

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l276_276721


namespace daily_rental_cost_l276_276295

theorem daily_rental_cost (x : ℝ) (total_cost miles : ℝ)
  (cost_per_mile : ℝ) (daily_cost : ℝ) :
  total_cost = daily_cost + cost_per_mile * miles →
  total_cost = 46.12 →
  miles = 214 →
  cost_per_mile = 0.08 →
  daily_cost = 29 :=
by
  sorry

end daily_rental_cost_l276_276295


namespace num_positive_divisors_36_l276_276984

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l276_276984


namespace a5_value_S8_value_l276_276021

-- Definitions based on the conditions
def seq (n : ℕ) : ℕ :=
if n = 0 then 0
else if n = 1 then 1
else 2 * seq (n - 1)

noncomputable def S (n : ℕ) : ℕ :=
(1 - 2^n) / (1 - 2)

-- Proof statements
theorem a5_value : seq 5 = 16 := sorry

theorem S8_value : S 8 = 255 := sorry

end a5_value_S8_value_l276_276021


namespace halloween_candy_l276_276332

theorem halloween_candy : 23 - 7 + 21 = 37 :=
by
  sorry

end halloween_candy_l276_276332


namespace part_a_part_b_l276_276899

theorem part_a (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ∃ (a b c d : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) := sorry

theorem part_b (a b c d e : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) : 
  ∃ (a b c d e : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) + 1 / (e : ℝ) := sorry

end part_a_part_b_l276_276899


namespace sum_base6_l276_276918

theorem sum_base6 (a b : ℕ) (h₁ : a = 5) (h₂ : b = 23) : 
  let sum := Nat.ofDigits 6 [2, 3] + Nat.ofDigits 6 [5]
  in Nat.digits 6 sum = [2, 3] :=
by
  sorry

end sum_base6_l276_276918


namespace validate_equation_l276_276122

variable (x : ℝ)

def price_of_notebook : ℝ := x - 2
def price_of_pen : ℝ := x

def total_cost (x : ℝ) : ℝ := 5 * price_of_notebook x + 3 * price_of_pen x

theorem validate_equation (x : ℝ) : total_cost x = 14 :=
by
  unfold total_cost
  unfold price_of_notebook
  unfold price_of_pen
  sorry

end validate_equation_l276_276122


namespace average_speed_l276_276439

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem average_speed (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) = (2 * b * a) / (a + b) :=
by
  sorry

end average_speed_l276_276439


namespace at_least_three_same_mistakes_l276_276555

structure Student :=
  (mistakes : Fin 13)

theorem at_least_three_same_mistakes (students : List Student)
  (h1 : students.length = 30)
  (h2 : ∃ s : Student, (s.mistakes = 12) ∧ (students.count s = 1)) 
  (h3 : ∀ s : Student, s.mistakes < 13 → (students.filter (λ t, t.mistakes = s.mistakes)).length ≤ 2) :
  ∃ n : Fin 13, 3 ≤ (students.filter (λ s, s.mistakes = n)).length :=
by
  sorry

end at_least_three_same_mistakes_l276_276555


namespace determine_c_l276_276457

noncomputable def c_floor : ℤ := -3
noncomputable def c_frac : ℝ := (25 - Real.sqrt 481) / 8

theorem determine_c : c_floor + c_frac = -2.72 := by
  have h1 : 3 * (c_floor : ℝ)^2 + 19 * (c_floor : ℝ) - 63 = 0 := by
    sorry
  have h2 : 4 * c_frac^2 - 25 * c_frac + 9 = 0 := by
    sorry
  sorry

end determine_c_l276_276457


namespace probability_four_green_marbles_l276_276919

theorem probability_four_green_marbles :
  let number_of_green := 10
  let number_of_purple := 5
  let total_marbles := number_of_green + number_of_purple
  let trials := 8
  let desired_green_marbles := 4
  (finset.card (finset.range(total_marbles).filter (λ x, x < number_of_green)) / total_marbles)^desired_green_marbles * 
  (finset.card (finset.range(total_marbles).filter (λ x, x >= number_of_green)) / total_marbles)^(trials - desired_green_marbles) *
  nat.choose(trials, desired_green_marbles) = 0.171 :=
by
  let number_of_green := 10
  let number_of_purple := 5
  let total_marbles := number_of_green + number_of_purple
  let trials := 8
  let desired_green_marbles := 4
  have h1 : (number_of_green / total_marbles : ℝ)^desired_green_marbles * 
            (number_of_purple / total_marbles : ℝ)^(trials - desired_green_marbles) * 
            nat.choose(trials, desired_green_marbles) = 0.171 := sorry
  exact h1

end probability_four_green_marbles_l276_276919


namespace gasoline_reduction_l276_276141

theorem gasoline_reduction (P Q : ℝ) :
  let new_price := 1.25 * P
  let new_budget := 1.10 * (P * Q)
  let new_quantity := new_budget / new_price
  let percent_reduction := 1 - (new_quantity / Q)
  percent_reduction = 0.12 :=
by
  sorry

end gasoline_reduction_l276_276141


namespace final_answer_is_d_l276_276472

-- Definitions of the propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x > 1
def q : Prop := false  -- since the distance between focus and directrix is not 1/6 but 3/2

-- The statement to be proven
theorem final_answer_is_d : p ∧ ¬ q := by sorry

end final_answer_is_d_l276_276472


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l276_276705

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l276_276705


namespace ten_degrees_below_zero_l276_276315

theorem ten_degrees_below_zero :
  (∀ (n : ℤ), n > 0 → (n.to_nat : ℤ) = n ∧ (-n.to_nat : ℤ) = -n) →
  (∀ t : ℤ, t = 10 → (t.above_zero = 10) → (10.below_zero = -10)) :=
begin
  intro h,
  have h1 : ∀ t : ℤ, t = 10 → (t * 1 : ℤ) = 10,
  { intro t,
    intro h2,
    rw h2,
    simp,
  },
  apply h1,
  sorry
end

end ten_degrees_below_zero_l276_276315


namespace intervals_of_monotonicity_range_of_k_l276_276343

noncomputable def f (a b x : ℝ) : ℝ := a * x - b * log x

theorem intervals_of_monotonicity :
  ∀ {a b : ℝ}, 
  (f a b =λx:ℝ, a * x - b * log x) ∧
  ((∃ a b, f a b 1 = 1 + 1) ∧
  (∃ a b, deriv (f a b) 1 = 1)) → 
  intervals_of_monotonicity (f 2 1) =
  {inc: (1 / 2, ∞), dec: (0, 1 / 2) } := sorry

theorem range_of_k :
  ∀ k : ℝ,
  (∀ x ≥ 1, k ≤ (2 - (log x / x)) →
  k ≤ 2 - 1 / real.exp 1) := sorry

end intervals_of_monotonicity_range_of_k_l276_276343


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l276_276709

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l276_276709


namespace only_n_equal_one_l276_276172

theorem only_n_equal_one (n : ℕ) (hn : 0 < n) : 
  (5 ^ (n - 1) + 3 ^ (n - 1)) ∣ (5 ^ n + 3 ^ n) → n = 1 := by
  intro h_div
  sorry

end only_n_equal_one_l276_276172


namespace kindergarten_classes_l276_276835

theorem kindergarten_classes :
  ∃ (j a m : ℕ), j + a + m = 32 ∧
                  j > 0 ∧ a > 0 ∧ m > 0 ∧
                  j / 2 + a / 4 + m / 8 = 6 ∧
                  (j = 4 ∧ a = 4 ∧ m = 24) :=
by {
  sorry
}

end kindergarten_classes_l276_276835


namespace geometric_sequence_common_ratio_l276_276195

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_arith : -a 5 + a 6 = 2 * a 4) :
  q = -1 ∨ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l276_276195


namespace central_angle_of_sector_l276_276057

theorem central_angle_of_sector (r A θ : ℝ) (hr : r = 2) (hA : A = 4) :
  θ = 2 :=
by
  sorry

end central_angle_of_sector_l276_276057


namespace number_of_cows_l276_276903

theorem number_of_cows (C H : ℕ) (L : ℕ) (h1 : L = 4 * C + 2 * H) (h2 : L = 2 * (C + H) + 20) : C = 10 :=
by
  sorry

end number_of_cows_l276_276903


namespace number_of_divisors_of_36_l276_276967

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l276_276967


namespace largest_number_among_four_l276_276137

theorem largest_number_among_four :
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  max a (max b (max c d)) = b := 
sorry

end largest_number_among_four_l276_276137


namespace range_of_m_l276_276072

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y ≤ f x) (m : ℝ) (h : f (m-1) > f (2*m-1)) : 0 < m :=
by
  sorry

end range_of_m_l276_276072


namespace pencils_given_l276_276374

theorem pencils_given (pencils_original pencils_left pencils_given : ℕ)
  (h1 : pencils_original = 142)
  (h2 : pencils_left = 111)
  (h3 : pencils_given = pencils_original - pencils_left) :
  pencils_given = 31 :=
by
  sorry

end pencils_given_l276_276374


namespace find_unknown_number_l276_276120

theorem find_unknown_number : 
  ∃ x : ℚ, (x * 7) / (10 * 17) = 10000 ∧ x = 1700000 / 7 :=
by
  sorry

end find_unknown_number_l276_276120


namespace relationship_y1_y2_y3_l276_276526

def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variable (m : ℝ) (y1 y2 y3 : ℝ)

-- Given conditions
axiom m_gt_zero : m > 0
axiom point1_on_graph : y1 = quadratic m (-1)
axiom point2_on_graph : y2 = quadratic m (5 / 2)
axiom point3_on_graph : y3 = quadratic m 6

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 : y3 > y1 ∧ y1 > y2 :=
by sorry

end relationship_y1_y2_y3_l276_276526


namespace pigeon_count_correct_l276_276907

def initial_pigeon_count : ℕ := 1
def new_pigeon_count : ℕ := 1
def total_pigeon_count : ℕ := 2

theorem pigeon_count_correct : initial_pigeon_count + new_pigeon_count = total_pigeon_count :=
by
  sorry

end pigeon_count_correct_l276_276907


namespace ratio_of_terms_l276_276269

theorem ratio_of_terms (a_n b_n : ℕ → ℕ) (S_n T_n : ℕ → ℕ) :
  (∀ n, S_n n = (n * (2 * a_n n - (n - 1))) / 2) → 
  (∀ n, T_n n = (n * (2 * b_n n - (n - 1))) / 2) → 
  (∀ n, S_n n / T_n n = (n + 3) / (2 * n + 1)) → 
  S_n 6 / T_n 6 = 14 / 23 :=
by
  sorry

end ratio_of_terms_l276_276269


namespace int_power_sum_is_integer_l276_276064

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem int_power_sum_is_integer {x : ℝ} (h : is_integer (x + 1/x)) (n : ℤ) : is_integer (x^n + 1/x^n) :=
by
  sorry

end int_power_sum_is_integer_l276_276064


namespace bananas_per_truck_l276_276896

theorem bananas_per_truck (total_apples total_bananas apples_per_truck : ℝ) 
  (h_total_apples: total_apples = 132.6)
  (h_apples_per_truck: apples_per_truck = 13.26)
  (h_total_bananas: total_bananas = 6.4) :
  (total_bananas / (total_apples / apples_per_truck)) = 0.64 :=
by
  sorry

end bananas_per_truck_l276_276896


namespace number_of_machines_in_first_group_l276_276741

-- Define the initial conditions
def first_group_production_rate (x : ℕ) : ℚ :=
  20 / (x * 10)

def second_group_production_rate : ℚ :=
  180 / (20 * 22.5)

-- The theorem we aim to prove
theorem number_of_machines_in_first_group (x : ℕ) (h1 : first_group_production_rate x = second_group_production_rate) :
  x = 5 :=
by
  -- Placeholder for the proof steps
  sorry

end number_of_machines_in_first_group_l276_276741


namespace linear_function_difference_l276_276546

variables {R : Type*} [LinearOrderedField R]

noncomputable def g (x : R) : R := sorry

theorem linear_function_difference (g : R → R) (h_linear : ∀ x y, g (x + y) = g x + g y) (h_diff : ∀ d : R, g (d + 1) - g d = 5) :
  g 0 - g 10 = -50 :=
by {
  sorry
}

end linear_function_difference_l276_276546


namespace a_takes_30_minutes_more_l276_276363

noncomputable def speed_ratio := 3 / 4
noncomputable def time_A := 2 -- 2 hours
noncomputable def time_diff (b_time : ℝ) := time_A - b_time

theorem a_takes_30_minutes_more (b_time : ℝ) 
  (h_ratio : speed_ratio = 3 / 4)
  (h_a : time_A = 2) :
  time_diff b_time = 0.5 →  -- because 0.5 hours = 30 minutes
  time_diff b_time * 60 = 30 :=
by sorry

end a_takes_30_minutes_more_l276_276363


namespace f_has_two_zeros_iff_l276_276198

open Real

noncomputable def f (x a : ℝ) : ℝ := (x - 2) * exp x + a * (x - 1)^2

theorem f_has_two_zeros_iff (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 0 < a :=
sorry

end f_has_two_zeros_iff_l276_276198


namespace compute_LM_length_l276_276058

-- Definitions of lengths and equidistant property
variables (GH JK LM : ℝ) 
variables (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
variables (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK)

-- State the theorem to prove lengths
theorem compute_LM_length (GH JD LM : ℝ) (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
  (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK) :
  LM = (2 / 3) * 80 := 
sorry

end compute_LM_length_l276_276058


namespace vector_sum_eq_l276_276633

variables (x y : ℝ)
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, 3)
def c : ℝ × ℝ := (7, 8)

theorem vector_sum_eq :
  ∃ (x y : ℝ), c = (x • a.1 + y • b.1, x • a.2 + y • b.2) ∧ x + y = 8 / 3 :=
by
  have h1 : 7 = 2 * x + 3 * y := sorry
  have h2 : 8 = 3 * x + 3 * y := sorry
  sorry

end vector_sum_eq_l276_276633


namespace aaron_walking_speed_l276_276782

-- Definitions of the conditions
def distance_jog : ℝ := 3 -- in miles
def speed_jog : ℝ := 2 -- in miles/hour
def total_time : ℝ := 3 -- in hours

-- The problem statement
theorem aaron_walking_speed :
  ∃ (v : ℝ), v = (distance_jog / (total_time - (distance_jog / speed_jog))) ∧ v = 2 :=
by
  sorry

end aaron_walking_speed_l276_276782


namespace even_divisors_8_factorial_l276_276037

theorem even_divisors_8_factorial : 
  let n := (2^7) * (3^2) * 5 * 7 in
  ∃ (count : ℕ), even_divisors_count n = 84 := 
sorry

end even_divisors_8_factorial_l276_276037


namespace angle_D_in_pentagon_l276_276496

theorem angle_D_in_pentagon (A B C D E : ℝ) 
  (h1 : A = B) (h2 : B = C) (h3 : D = E) (h4 : A + 40 = D) 
  (h5 : A + B + C + D + E = 540) : D = 132 :=
by
  -- Add proof here if needed
  sorry

end angle_D_in_pentagon_l276_276496


namespace sequence_periodic_l276_276217

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = (1 + a n) / (1 - a n)

theorem sequence_periodic :
  ∃ a : ℕ → ℝ, sequence a ∧ a 2016 = 3 :=
by
  sorry

end sequence_periodic_l276_276217


namespace base_length_of_parallelogram_l276_276746

theorem base_length_of_parallelogram (area : ℝ) (base altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base) :
  base = 7 :=
by
  sorry

end base_length_of_parallelogram_l276_276746


namespace arithmetic_seq_sum_l276_276366

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 3 + a 4 + a 5 + a 6 + a 7 = 250) : a 2 + a 8 = 100 :=
sorry

end arithmetic_seq_sum_l276_276366


namespace expression_for_f_general_formula_a_n_sum_S_n_l276_276621

-- Definitions for conditions
def f (x : ℝ) : ℝ := x^2 + x

-- Given conditions
axiom f_zero : f 0 = 0
axiom f_recurrence : ∀ x : ℝ, f (x + 1) - f x = x + 1

-- Statements to prove
theorem expression_for_f (x : ℝ) : f x = x^2 + x := 
sorry

theorem general_formula_a_n (t : ℝ) (n : ℕ) (H : 0 < t) : 
    ∃ a_n : ℕ → ℝ, a_n n = t^n := 
sorry

theorem sum_S_n (t : ℝ) (n : ℕ) (H : 0 < t) :
    ∃ S_n : ℕ → ℝ, (S_n n = if t = 1 then ↑n else (t * (t^n - 1)) / (t - 1)) := 
sorry

end expression_for_f_general_formula_a_n_sum_S_n_l276_276621


namespace annual_decrease_rate_l276_276119

theorem annual_decrease_rate (P : ℕ) (P2 : ℕ) (r : ℝ) : 
  (P = 10000) → (P2 = 8100) → (P2 = P * (1 - r / 100)^2) → (r = 10) :=
by
  intro hP hP2 hEq
  sorry

end annual_decrease_rate_l276_276119


namespace last_integer_in_sequence_is_one_l276_276585

theorem last_integer_in_sequence_is_one :
  ∀ seq : ℕ → ℕ, (seq 0 = 37) ∧ (∀ n, seq (n + 1) = seq n / 2) → (∃ n, seq (n + 1) = 0 ∧ seq n = 1) :=
by
  sorry

end last_integer_in_sequence_is_one_l276_276585


namespace find_f_neg_one_l276_276511

def f (x : ℝ) : ℝ := if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m)
def m : ℝ := -1

theorem find_f_neg_one (m : ℝ) (h_m : m = -1) (h_odd : ∀ x : ℝ, f (-x) = -f x) : f (-1) = -3 := 
by
  have h_def : f x = if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m),
  from sorry,
  sorry

end find_f_neg_one_l276_276511


namespace total_carrots_l276_276245

theorem total_carrots (sally_carrots fred_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : fred_carrots = 4) : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l276_276245


namespace solve_equation_l276_276740

theorem solve_equation (x : ℝ) (h : (x^2 + x + 1) / (x + 1) = x + 2) : x = -1/2 :=
by sorry

end solve_equation_l276_276740


namespace avg_weight_of_22_boys_l276_276493

theorem avg_weight_of_22_boys:
  let total_boys := 30
  let avg_weight_8 := 45.15
  let avg_weight_total := 48.89
  let total_weight_8 := 8 * avg_weight_8
  let total_weight_all := total_boys * avg_weight_total
  ∃ A : ℝ, A = 50.25 ∧ 22 * A + total_weight_8 = total_weight_all :=
by {
  sorry 
}

end avg_weight_of_22_boys_l276_276493


namespace find_functions_l276_276008

noncomputable def pair_of_functions_condition (f g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, g (f (x + y)) = f x + (2 * x + y) * g y

theorem find_functions (f g : ℝ → ℝ) :
  pair_of_functions_condition f g →
  (∃ c d : ℝ, ∀ x : ℝ, f x = c * (x + d)) :=
sorry

end find_functions_l276_276008


namespace concurrency_of_median_circle_intersections_l276_276275

theorem concurrency_of_median_circle_intersections
  (A B C : EuclideanGeometry.Point)
  (M1 M2 M3 : EuclideanGeometry.Point)
  (MA MB MC : EuclideanGeometry.Line)
  (circ1 circ2 circ3 : EuclideanGeometry.Circle) :
  EuclideanGeometry.is_median A M1 B C ∧
  EuclideanGeometry.is_median B M2 A C ∧
  EuclideanGeometry.is_median C M3 A B ∧
  EuclideanGeometry.is_circle_diameter circ1 A B ∧
  EuclideanGeometry.is_circle_diameter circ2 B C ∧
  EuclideanGeometry.is_circle_diameter circ3 C A ∧
  EuclideanGeometry.intersects_pairwise circ1 circ2 circ3 →
  let C1 := EuclideanGeometry.intersection_point circ1 circ2
  let A1 := EuclideanGeometry.intersection_point circ2 circ3
  let B1 := EuclideanGeometry.intersection_point circ3 circ1 
  in EuclideanGeometry.are_concurrent (EuclideanGeometry.line_through A A1)
                                      (EuclideanGeometry.line_through B B1)
                                      (EuclideanGeometry.line_through C C1) := 
begin
  sorry
end

end concurrency_of_median_circle_intersections_l276_276275


namespace num_br_atoms_l276_276301

theorem num_br_atoms (num_br : ℕ) : 
  (1 * 1 + num_br * 80 + 3 * 16 = 129) → num_br = 1 :=
  by
    intro h
    sorry

end num_br_atoms_l276_276301


namespace minimum_handshakes_l276_276771

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l276_276771


namespace fraction_check_l276_276134

variable (a b x y : ℝ)
noncomputable def is_fraction (expr : ℝ) : Prop :=
∃ n d : ℝ, d ≠ 0 ∧ expr = n / d ∧ ∃ var : ℝ, d = var

theorem fraction_check :
  is_fraction ((x + 3) / x) :=
sorry

end fraction_check_l276_276134


namespace muffin_cost_l276_276801

theorem muffin_cost (m : ℝ) :
  let fruit_cup_cost := 3
  let francis_cost := 2 * m + 2 * fruit_cup_cost
  let kiera_cost := 2 * m + 1 * fruit_cup_cost
  let total_cost := 17
  (francis_cost + kiera_cost = total_cost) → m = 2 :=
by
  intro h
  sorry

end muffin_cost_l276_276801


namespace number_of_correct_propositions_l276_276348

/-
Definitions corresponding to the propositions.
-/

def proposition1 := ∀ (A B C D : ℝ × ℝ), collinear A B ∧ collinear C D → collinear_all A B C D

def proposition2 := ∀ (u v : vector ℝ 2), unit_vector u → unit_vector v → u = v

def proposition3 := ∀ (a b c : vector ℝ 2), a = b → b = c → a = c

def proposition4 := ∀ (a : vector ℝ 2), magnitude a = 0 → ∀ (b : vector ℝ 2), parallel a b

def proposition5 := ∀ (a b c : vector ℝ 2), collinear a b → collinear b c → collinear a c

def proposition6 := ∀ (n : ℕ), 
  let Sn := ∑ k in finset.range n, real.sin (k * real.pi / 7) 
  in number_of_positive_terms Sn < 100 → count_positive_terms Sn = 72

/-
Main statement.
-/
theorem number_of_correct_propositions: 
  (proposition1 = false) ∨
  (proposition2 = false) ∧
  (proposition3 = true) ∧
  (proposition4 = true) ∧
  (proposition5 = false) ∧
  (proposition6 = false) ↔ 
  2 = 2 
:= by
  sorry

end number_of_correct_propositions_l276_276348


namespace circumference_of_jogging_track_l276_276106

noncomputable def trackCircumference (Deepak_speed : ℝ) (Wife_speed : ℝ) (meet_time_minutes : ℝ) : ℝ :=
  let relative_speed := Deepak_speed + Wife_speed
  let meet_time_hours := meet_time_minutes / 60
  relative_speed * meet_time_hours

theorem circumference_of_jogging_track :
  trackCircumference 20 17 37 = 1369 / 60 :=
by
  sorry

end circumference_of_jogging_track_l276_276106


namespace number_of_even_divisors_of_factorial_eight_l276_276038

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l276_276038


namespace decagon_diagonals_l276_276175

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l276_276175


namespace inequality_solution_l276_276475

theorem inequality_solution (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1 / x + 4 / y) ≥ 9 / 4 := 
sorry

end inequality_solution_l276_276475


namespace solve_abs_inequality_l276_276534

theorem solve_abs_inequality (x : ℝ) : x + |2 * x + 3| ≥ 2 ↔ (x ≤ -5 ∨ x ≥ -1/3) :=
by {
  sorry
}

end solve_abs_inequality_l276_276534


namespace polynomial_division_l276_276398

variable (x : ℝ)

theorem polynomial_division :
  ((3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 - x^2 + 6 * x - 8)) / (x + 1) 
  = (x^2 + 4 * x - 15 + 25 / (x+1)) :=
by sorry

end polynomial_division_l276_276398


namespace paula_cans_used_l276_276088

/-- 
  Paula originally had enough paint to cover 42 rooms. 
  Unfortunately, she lost 4 cans of paint on her way, 
  and now she can only paint 34 rooms. 
  Prove the number of cans she used for these 34 rooms is 17.
-/
theorem paula_cans_used (R L P C : ℕ) (hR : R = 42) (hL : L = 4) (hP : P = 34)
    (hRooms : R - ((R - P) / L) * L = P) :
  C = 17 :=
by
  sorry

end paula_cans_used_l276_276088


namespace quadratic_inequality_solution_l276_276203

theorem quadratic_inequality_solution
  (a b : ℝ)
  (h1 : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -2 ∨ -1/2 < x)) :
  ∀ x : ℝ, b * x^2 + a * x + 1 < 0 ↔ -2 < x ∧ x < -1/2 :=
by
  sorry

end quadratic_inequality_solution_l276_276203


namespace path_counts_l276_276843

    noncomputable def x : ℝ := 2 + Real.sqrt 2
    noncomputable def y : ℝ := 2 - Real.sqrt 2

    theorem path_counts (n : ℕ) :
      ∃ α : ℕ → ℕ, (α (2 * n - 1) = 0) ∧ (α (2 * n) = (1 / Real.sqrt 2) * ((x ^ (n - 1)) - (y ^ (n - 1)))) :=
    by
      sorry
    
end path_counts_l276_276843


namespace non_negative_integer_solutions_of_inequality_system_l276_276251

theorem non_negative_integer_solutions_of_inequality_system :
  (∀ x : ℚ, 3 * (x - 1) < 5 * x + 1 → (x - 1) / 2 ≥ 2 * x - 4 → (x = 0 ∨ x = 1 ∨ x = 2)) :=
by
  sorry

end non_negative_integer_solutions_of_inequality_system_l276_276251


namespace max_even_integers_with_odd_product_l276_276442

theorem max_even_integers_with_odd_product (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) (h_odd_product : (a * b * c * d * e * f) % 2 = 1) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) := 
sorry

end max_even_integers_with_odd_product_l276_276442


namespace annie_journey_time_l276_276921

noncomputable def total_time_journey (walk_speed1 bus_speed train_speed walk_speed2 blocks_walk1 blocks_bus blocks_train blocks_walk2 : ℝ) : ℝ :=
  let time_walk1 := blocks_walk1 / walk_speed1
  let time_bus := blocks_bus / bus_speed
  let time_train := blocks_train / train_speed
  let time_walk2 := blocks_walk2 / walk_speed2
  let time_back := time_walk2
  time_walk1 + time_bus + time_train + time_walk2 + time_back + time_train + time_bus + time_walk1

theorem annie_journey_time :
  total_time_journey 2 4 5 2 5 7 10 4 = 16.5 := by 
  sorry

end annie_journey_time_l276_276921


namespace volume_rectangular_prism_space_diagonal_rectangular_prism_surface_area_rectangular_prism_surface_area_circumscribed_sphere_l276_276109

-- Define the conditions of the rectangular prism
def length := 4
def width := 2
def height := 1

-- 1. Prove that the volume of the rectangular prism is 8
theorem volume_rectangular_prism : length * width * height = 8 := sorry

-- 2. Prove that the length of the space diagonal is √21
theorem space_diagonal_rectangular_prism : Real.sqrt (length^2 + width^2 + height^2) = Real.sqrt 21 := sorry

-- 3. Prove that the surface area of the rectangular prism is 28
theorem surface_area_rectangular_prism : 2 * (length * width + length * height + width * height) = 28 := sorry

-- 4. Prove that the surface area of the circumscribed sphere is 21π
theorem surface_area_circumscribed_sphere : 4 * Real.pi * (Real.sqrt (length^2 + width^2 + height^2) / 2)^2 = 21 * Real.pi := sorry

end volume_rectangular_prism_space_diagonal_rectangular_prism_surface_area_rectangular_prism_surface_area_circumscribed_sphere_l276_276109


namespace percent_reduction_l276_276266

def original_price : ℕ := 500
def reduction_amount : ℕ := 400

theorem percent_reduction : (reduction_amount * 100) / original_price = 80 := by
  sorry

end percent_reduction_l276_276266


namespace num_pos_divisors_36_l276_276981

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l276_276981


namespace min_handshakes_30_people_3_each_l276_276762

theorem min_handshakes_30_people_3_each : 
  ∃ (H : ℕ), (∀ (n k : ℕ), n = 30 ∧ k = 3 → H = (n * k) / 2) := 
by {
  use 45,
  intros n k h,
  rw [← h.1, ← h.2],
  norm_num,
  sorry
}

end min_handshakes_30_people_3_each_l276_276762


namespace megs_cat_weight_l276_276885

/-- The ratio of the weight of Meg's cat to Anne's cat is 5:7 and Anne's cat weighs 8 kg more than Meg's cat. Prove that the weight of Meg's cat is 20 kg. -/
theorem megs_cat_weight
  (M A : ℝ)
  (h1 : M / A = 5 / 7)
  (h2 : A = M + 8) :
  M = 20 :=
sorry

end megs_cat_weight_l276_276885


namespace number_chosen_div_8_sub_100_eq_6_l276_276434

variable (n : ℤ)

theorem number_chosen_div_8_sub_100_eq_6 (h : (n / 8) - 100 = 6) : n = 848 := 
by
  sorry

end number_chosen_div_8_sub_100_eq_6_l276_276434


namespace value_ne_one_l276_276208

theorem value_ne_one (a b: ℝ) (h : a * b ≠ 0) : (|a| / a) + (|b| / b) ≠ 1 := 
by 
  sorry

end value_ne_one_l276_276208


namespace total_cost_of_roads_l276_276780

/-- A rectangular lawn with dimensions 150 m by 80 m with two roads running 
through the middle, one parallel to the length and one parallel to the breadth. 
The first road has a width of 12 m, a base cost of Rs. 4 per sq m, and an additional section 
through a hill costing 25% more for a section of length 60 m. The second road has a width 
of 8 m and a cost of Rs. 5 per sq m. Prove that the total cost for both roads is Rs. 14000. -/
theorem total_cost_of_roads :
  let lawn_length := 150
  let lawn_breadth := 80
  let road1_width := 12
  let road2_width := 8
  let road1_base_cost := 4
  let road1_hill_length := 60
  let road1_hill_cost := road1_base_cost + (road1_base_cost / 4)
  let road2_cost := 5
  let road1_length := lawn_length
  let road2_length := lawn_breadth

  let road1_area_non_hill := road1_length * road1_width
  let road1_area_hill := road1_hill_length * road1_width
  let road1_cost_non_hill := road1_area_non_hill * road1_base_cost
  let road1_cost_hill := road1_area_hill * road1_hill_cost

  let total_road1_cost := road1_cost_non_hill + road1_cost_hill

  let road2_area := road2_length * road2_width
  let road2_total_cost := road2_area * road2_cost

  let total_cost := total_road1_cost + road2_total_cost

  total_cost = 14000 := by sorry

end total_cost_of_roads_l276_276780


namespace part1_part2_l276_276842

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem part2 (a x : ℝ) : 
  (f a x < a + 2) ↔ 
    (a = 0 ∧ x < 1) ∨ 
    (a > 0 ∧ -2 / a < x ∧ x < 1) ∨ 
    (-2 < a ∧ a < 0 ∧ (x < 1 ∨ x > -2 / a)) ∨ 
    (a = -2) ∨ 
    (a < -2 ∧ (x < -2 / a ∨ x > 1)) := sorry

end part1_part2_l276_276842


namespace log_product_log_expression_l276_276923

theorem log_product (h1 : log 2 25 = log 2 5 ^ 2) 
                    (h2 : log 3 4 = 2 * log 3 2)
                    (h3 : log 5 9 = log 5 3 ^ 2)
                    (h4 : log 2 5 * log 5 3 * log 3 2 = 1) :
  log 2 25 * log 3 4 * log 5 9 = 8 :=
by sorry

theorem log_expression (h1 : lg (32 / 49) = lg 32 - lg 49) 
                       (h2 : lg 32 = 5 * lg 2)
                       (h3 : lg 49 = 2 * lg 7)
                       (h4 : lg (sqrt 8) = (1/2) * lg 8)
                       (h5 : lg 8 = 3 * lg 2)
                       (h6 : lg (sqrt 245) = (1/2) * lg 245)
                       (h7 : lg 245 = lg 5 + 2 * lg 7)
                       (h8 : (1/2) * lg 2 + (1/2) * lg 5 = (1/2) * (lg 2 + lg 5)) :
  (1/2) * (lg (32 / 49)) - (4/3) * (lg (sqrt 8)) + lg (sqrt 245) = (1/2) :=
by sorry

end log_product_log_expression_l276_276923


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l276_276713

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l276_276713


namespace john_avg_speed_l276_276219

/-- John's average speed problem -/
theorem john_avg_speed (d : ℕ) (total_time : ℕ) (time1 : ℕ) (speed1 : ℕ) 
  (time2 : ℕ) (speed2 : ℕ) (time3 : ℕ) (x : ℕ) :
  d = 144 ∧ total_time = 120 ∧ time1 = 40 ∧ speed1 = 64 
  ∧ time2 = 40 ∧ speed2 = 70 ∧ time3 = 40 
  → (d = time1 * speed1 + time2 * speed2 + time3 * x / 60)
  → x = 82 := 
by
  intros h1 h2
  sorry

end john_avg_speed_l276_276219


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l276_276706

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l276_276706


namespace find_denomination_l276_276389

def denomination_of_bills (num_tumblers : ℕ) (cost_per_tumbler change num_bills amount_paid bill_denomination : ℤ) : Prop :=
  num_tumblers * cost_per_tumbler + change = amount_paid ∧
  amount_paid = num_bills * bill_denomination

theorem find_denomination :
  denomination_of_bills
    10    -- num_tumblers
    45    -- cost_per_tumbler
    50    -- change
    5     -- num_bills
    500   -- amount_paid
    100   -- bill_denomination
:=
by
  sorry

end find_denomination_l276_276389


namespace average_temperature_correct_l276_276130

-- Definition of the daily temperatures
def daily_temperatures : List ℕ := [51, 64, 61, 59, 48, 63, 55]

-- Define the number of days
def number_of_days : ℕ := 7

-- Prove the average temperature calculation
theorem average_temperature_correct :
  ((List.sum daily_temperatures : ℚ) / number_of_days : ℚ) = 57.3 :=
by
  sorry

end average_temperature_correct_l276_276130


namespace perpendicular_d_to_BC_l276_276634

def vector := (ℝ × ℝ)

noncomputable def AB : vector := (1, 1)
noncomputable def AC : vector := (2, 3)

noncomputable def BC : vector := (AC.1 - AB.1, AC.2 - AB.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

noncomputable def d : vector := (-6, 3)

theorem perpendicular_d_to_BC : is_perpendicular d BC :=
by
  sorry

end perpendicular_d_to_BC_l276_276634


namespace moles_of_Br2_combined_l276_276463

-- Definition of the reaction relation
def chemical_reaction (CH4 Br2 CH3Br HBr : ℕ) : Prop :=
  CH4 = 1 ∧ HBr = 1

-- Statement of the proof problem
theorem moles_of_Br2_combined (CH4 Br2 CH3Br HBr : ℕ) (h : chemical_reaction CH4 Br2 CH3Br HBr) : Br2 = 1 :=
by
  sorry

end moles_of_Br2_combined_l276_276463


namespace factorize_expression_l276_276185

theorem factorize_expression (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x - 1)^2 := 
sorry

end factorize_expression_l276_276185


namespace find_x_l276_276482

theorem find_x (x y : ℝ) (hx : x ≠ 0) (h1 : x / 2 = y^2) (h2 : x / 4 = 4 * y) : x = 128 :=
by
  sorry

end find_x_l276_276482


namespace pages_can_be_copied_l276_276665

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l276_276665


namespace solve_system_l276_276535

theorem solve_system : ∃ x y : ℝ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_l276_276535


namespace candy_eaten_l276_276016

/--
Given:
- Faye initially had 47 pieces of candy
- Faye ate x pieces the first night
- Faye's sister gave her 40 more pieces
- Now Faye has 62 pieces of candy

We need to prove:
- Faye ate 25 pieces of candy the first night.
-/
theorem candy_eaten (x : ℕ) (h1 : 47 - x + 40 = 62) : x = 25 :=
by
  sorry

end candy_eaten_l276_276016


namespace heads_matching_probability_l276_276502

-- Define the experiment and outcomes
def keiko_outcomes := {tt, ff} -- Keiko can get heads (tt) or tails (ff)
def ephraim_outcomes := { -- Ephraim's three coin outcomes
  (tt, tt, tt), (tt, tt, ff), (tt, ff, tt), (tt, ff, ff),
  (ff, tt, tt), (ff, tt, ff), (ff, ff, tt), (ff, ff, ff)
}
def linda_outcomes := {tt, ff} -- Linda can get heads (tt) or tails (ff)

-- Calculate the combined outcomes
def combined_outcomes : Set (Bool × (Bool × Bool × Bool) × Bool) :=
  { (k, (e1, e2, e3), l) | k ∈ keiko_outcomes ∧ (e1, e2, e3) ∈ ephraim_outcomes ∧ l ∈ linda_outcomes }

-- Calculate the number of matching outcomes
def count_matching_outcomes : ℕ :=
  combined_outcomes.count (λ (k, (e1, e2, e3), l) => (k.b2n = (e1.b2n + e2.b2n + e3.b2n + l.b2n)))

-- Total number of outcomes
def total_outcomes : ℕ := keiko_outcomes.card * ephraim_outcomes.card * linda_outcomes.card

-- Calculate the probability
def probability := (count_matching_outcomes.to_rat / total_outcomes.to_rat)

-- The proof statement
theorem heads_matching_probability : probability = 5 / 32 :=
by
  sorry

end heads_matching_probability_l276_276502


namespace sum_of_fractions_l276_276894

theorem sum_of_fractions : (1 / 6) + (2 / 9) + (1 / 3) = 13 / 18 := by
  sorry

end sum_of_fractions_l276_276894


namespace no_one_is_always_largest_l276_276253

theorem no_one_is_always_largest (a b c d : ℝ) :
  a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5 →
  ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → (x ≤ c ∨ x ≤ a) :=
by
  -- The proof requires assuming the conditions and showing that no variable is always the largest.
  intro h cond
  sorry

end no_one_is_always_largest_l276_276253


namespace max_area_of_rectangle_with_perimeter_40_l276_276112

theorem max_area_of_rectangle_with_perimeter_40 :
  (∃ (x y : ℝ), (2 * x + 2 * y = 40) ∧
                (∀ (a b : ℝ), (2 * a + 2 * b = 40) → (a * b ≤ x * y)) ∧
                (x * y = 100)) :=
begin
  -- Definitions of x and y satisfying the perimeter and maximizing the area.
  have h1 : ∀ (x y : ℝ), 2 * x + 2 * y = 40 → x * (20 - x) = -(x - 10)^2 + 100,
  { intro x, intro y, intro hper,
    have hy : y = 20 - x, by linarith,
    rw hy,
    ring },
  use 10,
  use 10,
  split,
  { -- Perimeter condition
    linarith },
  { split,
    { -- Maximum area condition
      intros a b hper,
      have hab : b = 20 - a, by linarith,
      rw hab,
      specialize h1 a (20 - a),
      linarith },
    { -- Maximum area is 100
      exact (by ring) } }
end

end max_area_of_rectangle_with_perimeter_40_l276_276112


namespace joe_eats_at_least_two_kinds_of_fruit_l276_276596

noncomputable def joe_probability_at_least_two_kinds_fruit : ℚ := 
  1 - (4 * (1 / 4) ^ 3)

theorem joe_eats_at_least_two_kinds_of_fruit : 
  joe_probability_at_least_two_kinds_fruit = 15 / 16 := by
  sorry

end joe_eats_at_least_two_kinds_of_fruit_l276_276596


namespace largest_two_digit_divisible_by_6_ending_in_4_l276_276732

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l276_276732


namespace coprime_probability_l276_276336

open Nat

def is_coprime (a b : ℕ) : Prop := gcd a b = 1

def selected_numbers := {x : Fin 9 // 2 ≤ x.val ∧ x.val ≤ 8}

theorem coprime_probability:
  let S := {x : ℕ | 2 ≤ x ∧ x ≤ 8} in
  let pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2} in
  let coprime_pairs := {p : ℕ × ℕ | p ∈ pairs ∧ is_coprime p.1 p.2} in
  (Fintype.card coprime_pairs : ℚ) / (Fintype.card pairs : ℚ) = 2 / 3 :=
by
  sorry

end coprime_probability_l276_276336


namespace tan_identity_l276_276485

theorem tan_identity (A B : ℝ) (hA : A = 30) (hB : B = 30) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = (4 + 2 * Real.sqrt 3)/3 := by
  sorry

end tan_identity_l276_276485


namespace min_area_bounded_l276_276612

noncomputable def intersection_points (a : ℝ) (ha : 0 < a) : set ℝ :=
    { x | a^3 * x^2 - a^4 * x = x }

noncomputable def area_bounded (a : ℝ) (ha : 0 < a) : ℝ :=
    ∫ x in 0..(a^4 + 1) / a^3, x - (a^3 * x^2 - a^4 * x)

theorem min_area_bounded (a : ℝ) (ha : 0 < a) : area_bounded a ha = 4 / 3 :=
begin
    sorry
end

end min_area_bounded_l276_276612


namespace min_value_expression_l276_276327

theorem min_value_expression (x : ℝ) (h : x > 10) : (x^2) / (x - 10) ≥ 40 :=
sorry

end min_value_expression_l276_276327


namespace loss_equates_to_balls_l276_276853

theorem loss_equates_to_balls
    (SP_20 : ℕ) (CP_1: ℕ) (Loss: ℕ) (x: ℕ)
    (h1 : SP_20 = 720)
    (h2 : CP_1 = 48)
    (h3 : Loss = (20 * CP_1 - SP_20))
    (h4 : Loss = x * CP_1) :
    x = 5 :=
by
  sorry

end loss_equates_to_balls_l276_276853


namespace simplify_expression_l276_276880

theorem simplify_expression : (Real.sqrt (9 / 4) - Real.sqrt (4 / 9)) = 5 / 6 :=
by
  sorry

end simplify_expression_l276_276880


namespace sin_pi_minus_alpha_l276_276025

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 5/13) : Real.sin (π - α) = 5/13 :=
by
  sorry

end sin_pi_minus_alpha_l276_276025


namespace kelly_gave_away_games_l276_276840

theorem kelly_gave_away_games (initial_games : ℕ) (remaining_games : ℕ) (given_away_games : ℕ) 
  (h1 : initial_games = 183) 
  (h2 : remaining_games = 92) 
  (h3 : given_away_games = initial_games - remaining_games) : 
  given_away_games = 91 := 
by 
  sorry

end kelly_gave_away_games_l276_276840


namespace largestValidNumberIs84_l276_276693

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l276_276693


namespace find_common_difference_l276_276215

theorem find_common_difference (a a_n S_n : ℝ) (h1 : a = 3) (h2 : a_n = 50) (h3 : S_n = 318) : 
  ∃ d n, (a + (n - 1) * d = a_n) ∧ (n / 2 * (a + a_n) = S_n) ∧ (d = 47 / 11) := 
by
  sorry

end find_common_difference_l276_276215


namespace angle_ABC_tangent_circle_l276_276550

theorem angle_ABC_tangent_circle 
  (BAC ACB : ℝ)
  (h1 : BAC = 70)
  (h2 : ACB = 45)
  (D : Type)
  (incenter : ∀ D : Type, Prop)  -- Represent the condition that D is the incenter
  : ∃ ABC : ℝ, ABC = 65 :=
by
  sorry

end angle_ABC_tangent_circle_l276_276550


namespace diagonals_of_decagon_l276_276180

theorem diagonals_of_decagon : 
  let n := 10 in 
  (n * (n - 3)) / 2 = 35 := 
by
  let n := 10
  show (n * (n - 3)) / 2 = 35
  sorry

end diagonals_of_decagon_l276_276180


namespace find_values_l276_276191

-- Define the conditions as Lean hypotheses
variables (A B : ℝ)

-- State the problem conditions
def condition1 := 30 - (4 * A + 5) = 3 * B
def condition2 := B = 2 * A

-- State the main theorem to be proved
theorem find_values (h1 : condition1 A B) (h2 : condition2 A B) : A = 2.5 ∧ B = 5 :=
by { sorry }

end find_values_l276_276191


namespace number_of_divisors_36_l276_276976

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l276_276976


namespace dad_eyes_l276_276218

def mom_eyes : ℕ := 1
def kids_eyes : ℕ := 3 * 4
def total_eyes : ℕ := 16

theorem dad_eyes :
  mom_eyes + kids_eyes + (total_eyes - (mom_eyes + kids_eyes)) = total_eyes :=
by 
  -- The proof part is omitted as per instructions
  sorry

example : (total_eyes - (mom_eyes + kids_eyes)) = 3 :=
by 
  -- The proof part is omitted as per instructions
  sorry

end dad_eyes_l276_276218


namespace matrix_sum_correct_l276_276608

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 0],
  ![1, 2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-5, -7],
  ![4, -9]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, -7],
  ![5, -7]
]

theorem matrix_sum_correct : A + B = C := by 
  sorry

end matrix_sum_correct_l276_276608


namespace max_f_on_interval_l276_276011

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x

theorem max_f_on_interval : 
  ∃ x ∈ Set.Icc (2 * Real.pi / 5) (3 * Real.pi / 4), f x = (1 + Real.sqrt 2) / 2 :=
by
  sorry

end max_f_on_interval_l276_276011


namespace points_per_win_is_5_l276_276254

-- Definitions based on conditions
def rounds_played : ℕ := 30
def vlad_points : ℕ := 64
def taro_points (T : ℕ) : ℕ := (3 * T) / 5 - 4
def total_points (T : ℕ) : ℕ := taro_points T + vlad_points

-- Theorem statement to prove the number of points per win
theorem points_per_win_is_5 (T : ℕ) (H : total_points T = T) : T / rounds_played = 5 := sorry

end points_per_win_is_5_l276_276254


namespace rationalize_denominator_l276_276691

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem rationalize_denominator :
  let a := cbrt 2
  let b := cbrt 27
  b = 3 -> ( 1 / (a + b)) = (cbrt 4 / (2 + 3 * cbrt 4))
:= by
  intro a
  intro b
  sorry

end rationalize_denominator_l276_276691


namespace eccentricity_of_ellipse_l276_276027

theorem eccentricity_of_ellipse (a b : ℝ) (h_ab : a > b) (h_b : b > 0) :
  (∀ x y : ℝ, (y = -2 * x + 1 → ∃ x₁ y₁ x₂ y₂ : ℝ, (y₁ = -2 * x₁ + 1 ∧ y₂ = -2 * x₂ + 1) ∧ 
    (x₁ / a * x₁ / a + y₁ / b * y₁ / b = 1) ∧ (x₂ / a * x₂ / a + y₂ / b * y₂ / b = 1) ∧ 
    ((x₁ + x₂) / 2 = 4 * (y₁ + y₂) / 2)) → (x / a)^2 + (y / b)^2 = 1) →
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = (Real.sqrt 2) / 2 :=
sorry

end eccentricity_of_ellipse_l276_276027


namespace percentage_increase_l276_276158

theorem percentage_increase 
    (P : ℝ)
    (buying_price : ℝ) (h1 : buying_price = 0.80 * P)
    (selling_price : ℝ) (h2 : selling_price = 1.24 * P) :
    ((selling_price - buying_price) / buying_price) * 100 = 55 := by 
  sorry

end percentage_increase_l276_276158


namespace chicken_cost_l276_276145
noncomputable def chicken_cost_per_plate
  (plates : ℕ) 
  (rice_cost_per_plate : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_rice_cost := plates * rice_cost_per_plate
  let total_chicken_cost := total_cost - total_rice_cost
  total_chicken_cost / plates

theorem chicken_cost
  (hplates : plates = 100)
  (hrice_cost_per_plate : rice_cost_per_plate = 0.10)
  (htotal_cost : total_cost = 50) :
  chicken_cost_per_plate 100 0.10 50 = 0.40 :=
by
  sorry

end chicken_cost_l276_276145


namespace archie_touchdown_passes_l276_276594

-- Definitions based on the conditions
def richard_avg_first_14_games : ℕ := 6
def richard_avg_last_2_games : ℕ := 3
def richard_games_first : ℕ := 14
def richard_games_last : ℕ := 2

-- Total touchdowns Richard made in the first 14 games
def touchdowns_first_14 := richard_games_first * richard_avg_first_14_games

-- Total touchdowns Richard needs in the final 2 games
def touchdowns_last_2 := richard_games_last * richard_avg_last_2_games

-- Total touchdowns Richard made in the season
def richard_touchdowns_season := touchdowns_first_14 + touchdowns_last_2

-- Archie's record is one less than Richard's total touchdowns for the season
def archie_record := richard_touchdowns_season - 1

-- Proposition to prove Archie's touchdown passes in a season
theorem archie_touchdown_passes : archie_record = 89 := by
  sorry

end archie_touchdown_passes_l276_276594


namespace correct_value_l276_276139

theorem correct_value (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 5/4 :=
sorry

end correct_value_l276_276139


namespace mod_equiv_pow_five_l276_276096

theorem mod_equiv_pow_five (m : ℤ) (hm : 0 ≤ m ∧ m < 11) (h : 12^5 ≡ m [ZMOD 11]) : m = 1 :=
by
  sorry

end mod_equiv_pow_five_l276_276096


namespace pages_copied_for_15_dollars_l276_276661

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l276_276661


namespace quadratic_union_nonempty_l276_276201

theorem quadratic_union_nonempty (a : ℝ) :
  (∃ x : ℝ, x^2 - (a-2)*x - 2*a + 4 = 0) ∨ (∃ y : ℝ, y^2 + (2*a-3)*y + 2*a^2 - a - 3 = 0) ↔
    a ≤ -6 ∨ (-7/2) ≤ a ∧ a ≤ (3/2) ∨ a ≥ 2 :=
sorry

end quadratic_union_nonempty_l276_276201


namespace planned_pencils_is_49_l276_276688

def pencils_planned (x : ℕ) : ℕ := x
def pencils_bought (x : ℕ) : ℕ := x + 12
def total_pencils_bought (x : ℕ) : ℕ := 61

theorem planned_pencils_is_49 (x : ℕ) :
  pencils_bought (pencils_planned x) = total_pencils_bought x → x = 49 :=
sorry

end planned_pencils_is_49_l276_276688


namespace find_a_and_b_l276_276344

theorem find_a_and_b :
  ∃ a b : ℝ, 
    (∀ x : ℝ, (x^3 + 3*x^2 + 2*x > 0) ↔ (x > 0 ∨ -2 < x ∧ x < -1)) ∧
    (∀ x : ℝ, (x^2 + a*x + b ≤ 0) ↔ (-2 < x ∧ x ≤ 0 ∨ 0 < x ∧ x ≤ 2)) ∧ 
    a = -1 ∧ b = -2 := 
  sorry

end find_a_and_b_l276_276344


namespace find_k_l276_276944

theorem find_k (k n m : ℕ) (hk : k > 0) (hn : n > 0) (hm : m > 0) 
  (h : (1 / (n ^ 2 : ℝ) + 1 / (m ^ 2 : ℝ)) = (k : ℝ) / (n ^ 2 + m ^ 2)) : k = 4 :=
sorry

end find_k_l276_276944


namespace total_time_of_four_sets_of_stairs_l276_276450

def time_first : ℕ := 15
def time_increment : ℕ := 10
def num_sets : ℕ := 4

theorem total_time_of_four_sets_of_stairs :
  let a := time_first
  let d := time_increment
  let n := num_sets
  let l := a + (n - 1) * d
  let S := n / 2 * (a + l)
  S = 120 :=
by
  sorry

end total_time_of_four_sets_of_stairs_l276_276450


namespace problem_l276_276624

variable (x y : ℝ)

-- Define the given condition
def condition : Prop := |x + 5| + (y - 4)^2 = 0

-- State the theorem we need to prove
theorem problem (h : condition x y) : (x + y)^99 = -1 := sorry

end problem_l276_276624


namespace no_solution_system_l276_276798

theorem no_solution_system (v : ℝ) :
  (∀ x y z : ℝ, ¬(x + y + z = v ∧ x + v * y + z = v ∧ x + y + v^2 * z = v^2)) ↔ (v = -1) :=
  sorry

end no_solution_system_l276_276798


namespace range_of_a_l276_276188

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 → 
    (x + 3 + 2 * (Real.sin θ) * (Real.cos θ))^2 + (x + a * (Real.sin θ) + a * (Real.cos θ))^2 ≥ 1 / 8) → 
  a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end range_of_a_l276_276188


namespace relationship_y1_y2_y3_l276_276525

def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variable (m : ℝ) (y1 y2 y3 : ℝ)

-- Given conditions
axiom m_gt_zero : m > 0
axiom point1_on_graph : y1 = quadratic m (-1)
axiom point2_on_graph : y2 = quadratic m (5 / 2)
axiom point3_on_graph : y3 = quadratic m 6

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 : y3 > y1 ∧ y1 > y2 :=
by sorry

end relationship_y1_y2_y3_l276_276525


namespace trader_gain_percentage_l276_276598

theorem trader_gain_percentage 
  (C : ℝ) -- cost of each pen
  (h1 : 250 * C ≠ 0) -- ensure the cost of 250 pens is non-zero
  (h2 : 65 * C > 0) -- ensure the gain is positive
  (h3 : 250 * C + 65 * C > 0) -- ensure the selling price is positive
  : (65 / 250) * 100 = 26 := 
sorry

end trader_gain_percentage_l276_276598


namespace actual_distance_map_l276_276233

theorem actual_distance_map (scale : ℕ) (map_distance : ℕ) (actual_distance_km : ℕ) (h1 : scale = 500000) (h2 : map_distance = 4) :
  actual_distance_km = 20 :=
by
  -- definitions and assumptions
  let actual_distance_cm := map_distance * scale
  have cm_to_km_conversion : actual_distance_km = actual_distance_cm / 100000 := sorry
  -- calculation
  have actual_distance_sol : actual_distance_cm = 4 * 500000 := sorry
  have actual_distance_eq : actual_distance_km = (4 * 500000) / 100000 := sorry
  -- final answer
  have answer_correct : actual_distance_km = 20 := sorry
  exact answer_correct

end actual_distance_map_l276_276233


namespace tree_height_after_two_years_l276_276160

theorem tree_height_after_two_years :
  (∀ (f : ℕ → ℝ), (∀ (n : ℕ), f (n + 1) = 3 * f n) → f 4 = 81 → f 2 = 9) :=
begin
  sorry
end

end tree_height_after_two_years_l276_276160


namespace books_sum_l276_276225

theorem books_sum (darryl_books lamont_books loris_books danielle_books : ℕ) 
  (h1 : darryl_books = 20)
  (h2 : lamont_books = 2 * darryl_books)
  (h3 : lamont_books = loris_books + 3)
  (h4 : danielle_books = lamont_books + darryl_books + 10) : 
  darryl_books + lamont_books + loris_books + danielle_books = 167 := 
by
  sorry

end books_sum_l276_276225


namespace solve_system_of_equations_l276_276816

theorem solve_system_of_equations :
  ∀ (x y : ℝ),
  (3 * x - 2 * y = 7) →
  (2 * x + 3 * y = 8) →
  x = 37 / 13 :=
by
  intros x y h1 h2
  -- to prove x = 37 / 13 from the given system of equations
  sorry

end solve_system_of_equations_l276_276816


namespace bead_count_l276_276077

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end bead_count_l276_276077


namespace isosceles_triangle_three_times_ce_l276_276837

/-!
# Problem statement
In the isosceles triangle \( ABC \) with \( \overline{AC} = \overline{BC} \), 
\( D \) is the foot of the altitude through \( C \) and \( M \) is 
the midpoint of segment \( CD \). The line \( BM \) intersects \( AC \) 
at \( E \). Prove that \( AC \) is three times as long as \( CE \).
-/

-- Definition of isosceles triangle and related points
variables {A B C D E M : Type} 

-- Assume necessary conditions
variables (triangle_isosceles : A = B)
variables (D_foot : true) -- Placeholder, replace with proper definition if needed
variables (M_midpoint : true) -- Placeholder, replace with proper definition if needed
variables (BM_intersects_AC : true) -- Placeholder, replace with proper definition if needed

-- Main statement to prove
theorem isosceles_triangle_three_times_ce (h1 : A = B)
    (h2 : true) (h3 : true) (h4 : true) : 
    AC = 3 * CE :=
by
  sorry

end isosceles_triangle_three_times_ce_l276_276837


namespace linda_age_l276_276518

theorem linda_age 
  (J : ℕ)  -- Jane's current age
  (H1 : ∃ J, 2 * J + 3 = 13) -- Linda is 3 more than 2 times the age of Jane
  (H2 : (J + 5) + ((2 * J + 3) + 5) = 28) -- In 5 years, the sum of their ages will be 28
  : 2 * J + 3 = 13 :=
by {
  sorry
}

end linda_age_l276_276518


namespace determine_m_type_l276_276799

theorem determine_m_type (m : ℝ) :
  ((m^2 + 2*m - 8 = 0) ↔ (m = -4)) ∧
  ((m^2 - 2*m = 0) ↔ (m = 0 ∨ m = 2)) ∧
  ((m^2 - 2*m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 2)) :=
by sorry

end determine_m_type_l276_276799


namespace aquarium_water_l276_276391

theorem aquarium_water (T1 T2 T3 T4 : ℕ) (g w : ℕ) (hT1 : T1 = 8) (hT2 : T2 = 8) (hT3 : T3 = 6) (hT4 : T4 = 6):
  (g = T1 + T2 + T3 + T4) → (w = g * 4) → w = 112 :=
by
  sorry

end aquarium_water_l276_276391


namespace closest_point_to_origin_l276_276086

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l276_276086


namespace coprime_probability_is_two_thirds_l276_276335

-- Define the set of integers
def intSet : Finset ℕ := {2, 3, 4, 5, 6, 7, 8}

-- Define what it means to be coprime
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Number of ways to choose 2 different numbers from the set
def totalWays : ℕ := intSet.card.choose 2

-- Number of coprime pairs in the set
def coprimePairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ p : ℕ × ℕ, coprime p.fst p.snd) (intSet.product intSet)

-- Calculating the probability
def coprimeProbability : ℚ :=
  (coprimePairs.card.toRat / totalWays.toRat)

-- Theorem statement
theorem coprime_probability_is_two_thirds : coprimeProbability = 2/3 :=
  sorry

end coprime_probability_is_two_thirds_l276_276335


namespace add_in_base6_l276_276916

def add_base6 (a b : ℕ) : ℕ := (a + b) % 6 + (((a + b) / 6) * 10)

theorem add_in_base6 (x y : ℕ) (h1 : x = 5) (h2 : y = 23) : add_base6 x y = 32 :=
by
  rw [h1, h2]
  -- Explanation: here add_base6 interprets numbers as base 6 and then performs addition,
  -- taking care of the base conversion automatically. This avoids directly involving steps of the given solution.
  sorry

end add_in_base6_l276_276916


namespace white_stones_count_l276_276412

/-- We define the total number of stones as a constant. -/
def total_stones : ℕ := 120

/-- We define the difference between white and black stones as a constant. -/
def white_minus_black : ℕ := 36

/-- The theorem states that if there are 120 go stones in total and 
    36 more white go stones than black go stones, then there are 78 white go stones. -/
theorem white_stones_count (W B : ℕ) (h1 : W = B + white_minus_black) (h2 : B + W = total_stones) : W = 78 := 
sorry

end white_stones_count_l276_276412


namespace eccentricity_of_ellipse_l276_276053
-- Import the Mathlib library for mathematical tools and structures

-- Define the condition for the ellipse and the arithmetic sequence
variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = a + c) (h4 : b^2 = a^2 - c^2)

-- State the theorem to prove
theorem eccentricity_of_ellipse : ∃ e : ℝ, e = 3 / 5 :=
by
  -- Proof would go here
  sorry

end eccentricity_of_ellipse_l276_276053


namespace dragon_legs_l276_276908

variable {x y n : ℤ}

theorem dragon_legs :
  (x = 40) ∧
  (y = 9) ∧
  (220 = 40 * x + n * y) →
  n = 4 :=
by
  sorry

end dragon_legs_l276_276908


namespace triangle_sum_of_squares_not_right_l276_276214

noncomputable def is_right_triangle (a b c : ℝ) : Prop := 
  (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2) ∨ (c^2 + a^2 = b^2)

theorem triangle_sum_of_squares_not_right
  (a b r : ℝ) :
  a^2 + b^2 = (2 * r)^2 → ¬ ∃ (c : ℝ), is_right_triangle a b c := 
sorry

end triangle_sum_of_squares_not_right_l276_276214


namespace right_triangle_ratio_segments_l276_276649

theorem right_triangle_ratio_segments (a b c r s : ℝ) (h : a^2 + b^2 = c^2) (h_drop : r + s = c) (a_to_b_ratio : 2 * b = 5 * a) : r / s = 4 / 25 :=
sorry

end right_triangle_ratio_segments_l276_276649


namespace weight_differences_correct_l276_276513

-- Define the weights of Heather, Emily, Elizabeth, and Emma
def H : ℕ := 87
def E1 : ℕ := 58
def E2 : ℕ := 56
def E3 : ℕ := 64

-- Proof problem statement
theorem weight_differences_correct :
  (H - E1 = 29) ∧ (H - E2 = 31) ∧ (H - E3 = 23) :=
by
  -- Note: 'sorry' is used to skip the proof itself
  sorry

end weight_differences_correct_l276_276513


namespace union_sets_M_N_l276_276032

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- The proof statement: the union of M and N should be x > -3
theorem union_sets_M_N : (M ∪ N) = {x | x > -3} :=
sorry

end union_sets_M_N_l276_276032


namespace equations_not_equivalent_l276_276595

theorem equations_not_equivalent :
  ∀ x : ℝ, (x + 7 + 10 / (2 * x - 1) = 8 - x + 10 / (2 * x - 1)) ↔ false :=
by
  intro x
  sorry

end equations_not_equivalent_l276_276595


namespace max_area_rect_40_perimeter_l276_276114

noncomputable def max_rect_area (P : ℕ) (hP : P = 40) : ℕ :=
  let w : ℕ → ℕ := id
  let l : ℕ → ℕ := λ w, P / 2 - w
  let area : ℕ → ℕ := λ w, w * (P / 2 - w)
  find_max_value area sorry

theorem max_area_rect_40_perimeter : max_rect_area 40 40 = 100 := 
sorry

end max_area_rect_40_perimeter_l276_276114


namespace least_positive_integer_x_l276_276010

theorem least_positive_integer_x (x : ℕ) (h : x + 5683 ≡ 420 [MOD 17]) : x = 7 :=
sorry

end least_positive_integer_x_l276_276010


namespace total_numbers_is_eight_l276_276878

theorem total_numbers_is_eight
  (avg_all : ∀ n : ℕ, (total_sum : ℝ) / n = 25)
  (avg_first_two : ∀ a₁ a₂ : ℝ, (a₁ + a₂) / 2 = 20)
  (avg_next_three : ∀ a₃ a₄ a₅ : ℝ, (a₃ + a₄ + a₅) / 3 = 26)
  (h_sixth : ∀ a₆ a₇ a₈ : ℝ, a₆ + 4 = a₇ ∧ a₆ + 6 = a₈)
  (last_num : ∀ a₈ : ℝ, a₈ = 30) :
  ∃ n : ℕ, n = 8 :=
by
  sorry

end total_numbers_is_eight_l276_276878


namespace john_spent_15_dollars_on_soap_l276_276499

-- Define the number of soap bars John bought
def num_bars : ℕ := 20

-- Define the weight of each bar of soap in pounds
def weight_per_bar : ℝ := 1.5

-- Define the cost per pound of soap in dollars
def cost_per_pound : ℝ := 0.5

-- Total weight of the soap in pounds
def total_weight : ℝ := num_bars * weight_per_bar

-- Total cost of the soap in dollars
def total_cost : ℝ := total_weight * cost_per_pound

-- Statement to prove
theorem john_spent_15_dollars_on_soap : total_cost = 15 :=
by sorry

end john_spent_15_dollars_on_soap_l276_276499


namespace num_ways_is_20_l276_276133

-- Define the set of students
def students := { "Jungkook", "Jimin", "Seokjin", "Taehyung", "Namjoon" }

-- Define a representative and a vice-president as a permutation of two students
def num_ways :=
  fintype.card (equiv.perm (fin 2))

-- Theorem: Number of ways to select representative and vice-president
theorem num_ways_is_20 : num_ways = 20 := by
  sorry

end num_ways_is_20_l276_276133


namespace num_divisors_360_l276_276207

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l276_276207


namespace triangle_side_ineq_l276_276486

theorem triangle_side_ineq (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 :=
  sorry

end triangle_side_ineq_l276_276486


namespace directrix_of_parabola_l276_276325

theorem directrix_of_parabola (y : ℝ) : 
  (∃ y : ℝ, x = 1) ↔ (x = (1 / 4 : ℝ) * y^2) := 
sorry

end directrix_of_parabola_l276_276325


namespace fruit_salad_total_l276_276494

def fruit_salad_problem (R_red G R_rasp total_fruit : ℕ) : Prop :=
  R_red = 67 ∧ (3 * G + 7 = 67) ∧ (R_rasp = G - 5) ∧ (total_fruit = R_red + G + R_rasp)

theorem fruit_salad_total (R_red G R_rasp : ℕ) (total_fruit : ℕ) :
  fruit_salad_problem R_red G R_rasp total_fruit → total_fruit = 102 :=
by
  intro h
  sorry

end fruit_salad_total_l276_276494


namespace three_digit_integers_count_l276_276044

theorem three_digit_integers_count (N : ℕ) :
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
            n % 7 = 4 ∧ 
            n % 8 = 3 ∧ 
            n % 10 = 2) → N = 3 :=
by
  sorry

end three_digit_integers_count_l276_276044


namespace fractional_part_of_students_who_walk_home_l276_276164

theorem fractional_part_of_students_who_walk_home 
  (students_by_bus : ℚ)
  (students_by_car : ℚ)
  (students_by_bike : ℚ)
  (students_by_skateboard : ℚ)
  (h_bus : students_by_bus = 1/3)
  (h_car : students_by_car = 1/5)
  (h_bike : students_by_bike = 1/8)
  (h_skateboard : students_by_skateboard = 1/15)
  : 1 - (students_by_bus + students_by_car + students_by_bike + students_by_skateboard) = 11/40 := 
by
  sorry

end fractional_part_of_students_who_walk_home_l276_276164


namespace intersection_A_B_l276_276478

-- Definition of sets A and B
def A : Set ℤ := {0, 1, 2, 3}
def B : Set ℤ := { x | -1 ≤ x ∧ x < 3 }

-- Statement to prove
theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := 
sorry

end intersection_A_B_l276_276478


namespace add_in_base6_l276_276915

def add_base6 (a b : ℕ) : ℕ := (a + b) % 6 + (((a + b) / 6) * 10)

theorem add_in_base6 (x y : ℕ) (h1 : x = 5) (h2 : y = 23) : add_base6 x y = 32 :=
by
  rw [h1, h2]
  -- Explanation: here add_base6 interprets numbers as base 6 and then performs addition,
  -- taking care of the base conversion automatically. This avoids directly involving steps of the given solution.
  sorry

end add_in_base6_l276_276915


namespace fraction_product_eq_l276_276419

theorem fraction_product_eq : (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end fraction_product_eq_l276_276419


namespace flowchart_correct_option_l276_276898

-- Definitions based on conditions
def typical_flowchart (start_points end_points : ℕ) : Prop :=
  start_points = 1 ∧ end_points ≥ 1

-- Theorem to prove
theorem flowchart_correct_option :
  ∃ (start_points end_points : ℕ), typical_flowchart start_points end_points ∧ "Option C" = "Option C" :=
by {
  sorry -- This part skips the proof itself,
}

end flowchart_correct_option_l276_276898


namespace f_ln2_add_f_ln_half_l276_276350

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_ln2_add_f_ln_half :
  f (Real.log 2) + f (Real.log (1 / 2)) = 2 :=
by
  sorry

end f_ln2_add_f_ln_half_l276_276350


namespace discriminant_of_quadratic_eq_l276_276456

-- Define the coefficients of the quadratic equation
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

-- State the theorem that we want to prove
theorem discriminant_of_quadratic_eq : discriminant a b c = 61 := by
  sorry

end discriminant_of_quadratic_eq_l276_276456


namespace analytic_expression_of_f_range_of_k_l276_276815

noncomputable def quadratic_function_minimum (a b : ℝ) : ℝ :=
a * (-1) ^ 2 + b * (-1) + 1

theorem analytic_expression_of_f (a b : ℝ) (ha : quadratic_function_minimum a b = 0)
  (hmin: -1 = -b / (2 * a)) : a = 1 ∧ b = 2 :=
by sorry

theorem range_of_k (k : ℝ) : ∃ k : ℝ, (k ∈ Set.Ici 3 ∨ k = 13 / 4) :=
by sorry

end analytic_expression_of_f_range_of_k_l276_276815


namespace economy_value_after_two_years_l276_276996

/--
Given an initial amount A₀ = 3200,
that increases annually by 1/8th of itself,
with an inflation rate of 3% in the first year and 4% in the second year,
prove that the value of the amount after two years is 3771.36
-/
theorem economy_value_after_two_years :
  let A₀ := 3200 
  let increase_rate := 1 / 8
  let inflation_rate_year_1 := 0.03
  let inflation_rate_year_2 := 0.04
  let A₁ := A₀ * (1 + increase_rate)
  let V₁ := A₁ * (1 - inflation_rate_year_1)
  let A₂ := V₁ * (1 + increase_rate)
  let V₂ := A₂ * (1 - inflation_rate_year_2)
  V₂ = 3771.36 :=
by
  simp only []
  sorry

end economy_value_after_two_years_l276_276996


namespace min_cylinder_surface_area_l276_276588

noncomputable def h := Real.sqrt (5^2 - 4^2)
noncomputable def V_cone := (1 / 3) * Real.pi * 4^2 * h
noncomputable def V_cylinder (r h': ℝ) := Real.pi * r^2 * h'
noncomputable def h' (r: ℝ) := 16 / r^2
noncomputable def S (r: ℝ) := 2 * Real.pi * r^2 + (32 * Real.pi) / r

theorem min_cylinder_surface_area : 
  ∃ r, r = 2 ∧ ∀ r', r' ≠ 2 → S r' > S 2 := sorry

end min_cylinder_surface_area_l276_276588


namespace repeating_decimal_as_fraction_l276_276004

theorem repeating_decimal_as_fraction : (3 + 167 / 999 : ℚ) = 3164 / 999 := 
by sorry

end repeating_decimal_as_fraction_l276_276004


namespace coprime_probability_is_two_thirds_l276_276338

/-- Problem statement -/
def S : Finset ℕ := {2, 3, 4, 5, 6, 7, 8}

/-- Two numbers are coprime if their gcd is 1. -/
def coprime_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S.product S.filter (λ (p : ℕ × ℕ), p.fst < p.snd ∧ Nat.gcd p.fst p.snd = 1)

/-- The set of all pairs of different numbers from S -/
def all_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S.product S.filter (λ (p : ℕ × ℕ), p.fst < p.snd)

/-- Probability calculation -/
noncomputable def coprime_probability (S : Finset ℕ) : ℚ :=
  (coprime_pairs S).card / (all_pairs S).card

theorem coprime_probability_is_two_thirds :
  coprime_probability S = 2 / 3 :=
  sorry -- Proof goes here.

end coprime_probability_is_two_thirds_l276_276338


namespace tiling_problem_l276_276277

theorem tiling_problem (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ n = 4 * k) 
  ↔ (∃ (L_tile T_tile : ℕ), n * n = 3 * L_tile + 4 * T_tile) :=
by
  sorry

end tiling_problem_l276_276277


namespace area_of_square_plot_l276_276895

theorem area_of_square_plot (price_per_foot : ℕ) (total_cost : ℕ) (h_price : price_per_foot = 58) (h_cost : total_cost = 2088) :
  ∃ s : ℕ, s^2 = 81 := by
  sorry

end area_of_square_plot_l276_276895


namespace percent_gain_correct_l276_276303

theorem percent_gain_correct :
  ∀ (x : ℝ), (900 * x + 50 * (900 * x / 850) - 900 * x) / (900 * x) * 100 = 58.82 :=
by sorry

end percent_gain_correct_l276_276303


namespace new_average_weight_l276_276749

theorem new_average_weight (avg_weight_19_students : ℝ) (new_student_weight : ℝ) (num_students_initial : ℕ) : 
  avg_weight_19_students = 15 → new_student_weight = 7 → num_students_initial = 19 → 
  let total_weight_with_new_student := (avg_weight_19_students * num_students_initial + new_student_weight) 
  let new_num_students := num_students_initial + 1 
  let new_avg_weight := total_weight_with_new_student / new_num_students 
  new_avg_weight = 14.6 :=
by
  intros h1 h2 h3
  let total_weight := avg_weight_19_students * num_students_initial
  let total_weight_with_new_student := total_weight + new_student_weight
  let new_num_students := num_students_initial + 1
  let new_avg_weight := total_weight_with_new_student / new_num_students
  have h4 : total_weight = 285 := by sorry
  have h5 : total_weight_with_new_student = 292 := by sorry
  have h6 : new_num_students = 20 := by sorry
  have h7 : new_avg_weight = 292 / 20 := by sorry
  have h8 : new_avg_weight = 14.6 := by sorry
  exact h8

end new_average_weight_l276_276749


namespace intersection_M_N_l276_276200

-- Define set M
def M : Set Int := {-2, -1, 0, 1}

-- Define set N using the given condition
def N : Set Int := {n : Int | -1 <= n ∧ n <= 3}

-- State that the intersection of M and N is the set {-1, 0, 1}
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l276_276200


namespace height_difference_zero_l276_276298

-- Define the problem statement and conditions
theorem height_difference_zero (a b : ℝ) (h1 : ∀ x, y = 2 * x^2)
  (h2 : b - a^2 = 1 / 4) : 
  ( b - 2 * a^2) = 0 :=
by
  sorry

end height_difference_zero_l276_276298


namespace boat_distance_along_stream_l276_276651

-- Define the conditions
def speed_of_boat_still_water : ℝ := 9
def distance_against_stream_per_hour : ℝ := 7

-- Define the speed of the stream
def speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour

-- Define the speed of the boat along the stream
def speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream

-- Theorem statement
theorem boat_distance_along_stream (speed_of_boat_still_water : ℝ)
                                    (distance_against_stream_per_hour : ℝ)
                                    (effective_speed_against_stream : ℝ := speed_of_boat_still_water - speed_of_stream)
                                    (speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour)
                                    (speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream)
                                    (one_hour : ℝ := 1) :
  speed_of_boat_along_stream = 11 := 
  by
    sorry

end boat_distance_along_stream_l276_276651


namespace age_of_participant_who_left_l276_276067

theorem age_of_participant_who_left
  (avg_age_first_room : ℕ)
  (num_people_first_room : ℕ)
  (avg_age_second_room : ℕ)
  (num_people_second_room : ℕ)
  (increase_in_avg_age : ℕ)
  (total_num_people : ℕ)
  (final_avg_age : ℕ)
  (initial_avg_age : ℕ)
  (sum_ages : ℕ)
  (person_left : ℕ) :
  avg_age_first_room = 20 ∧ 
  num_people_first_room = 8 ∧
  avg_age_second_room = 45 ∧
  num_people_second_room = 12 ∧
  increase_in_avg_age = 1 ∧
  total_num_people = num_people_first_room + num_people_second_room ∧
  final_avg_age = initial_avg_age + increase_in_avg_age ∧
  initial_avg_age = (sum_ages) / total_num_people ∧
  sum_ages = (avg_age_first_room * num_people_first_room + avg_age_second_room * num_people_second_room) ∧
  19 * final_avg_age = sum_ages - person_left
  → person_left = 16 :=
by sorry

end age_of_participant_who_left_l276_276067


namespace min_a_decreasing_range_a_condition_l276_276197

noncomputable def f (x a : ℝ) : ℝ := x / Real.log x - a * x
noncomputable def f' (x a : ℝ) : ℝ := (Real.log x - 1) / (Real.log x)^2 - a

theorem min_a_decreasing (a : ℝ) (h : ∀ x > 1, f' x a ≤ 0) : 
  a ≥ 1 / 4 :=
sorry

theorem range_a_condition (a : ℝ) 
  (h : ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc Real.exp (Real.exp 1 2) ∧ x₂ ∈ Set.Icc Real.exp (Real.exp 1 2) ∧ f x₁ a ≤ f' x₂ a + a) : 
  a ≥ (1 / 2) - (1 / (4 * Real.exp 1 2)) :=
sorry

end min_a_decreasing_range_a_condition_l276_276197


namespace num_pos_divisors_36_l276_276962

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l276_276962


namespace jelly_bean_remaining_l276_276273

theorem jelly_bean_remaining (J : ℕ) (P : ℕ) (taken_last_4_each : ℕ) (taken_first_each : ℕ) 
 (taken_last_total : ℕ) (taken_first_total : ℕ) (taken_total : ℕ) (remaining : ℕ) :
  J = 8000 →
  P = 10 →
  taken_last_4_each = 400 →
  taken_first_each = 2 * taken_last_4_each →
  taken_last_total = 4 * taken_last_4_each →
  taken_first_total = 6 * taken_first_each →
  taken_total = taken_last_total + taken_first_total →
  remaining = J - taken_total →
  remaining = 1600 :=
by
  intros
  sorry  

end jelly_bean_remaining_l276_276273


namespace sugar_fill_count_l276_276310

noncomputable def sugar_needed_for_one_batch : ℚ := 3 + 1/2
noncomputable def total_batches : ℕ := 2
noncomputable def cup_capacity : ℚ := 1/3
noncomputable def total_sugar_needed : ℚ := total_batches * sugar_needed_for_one_batch

theorem sugar_fill_count : (total_sugar_needed / cup_capacity) = 21 :=
by
  -- Assuming necessary preliminary steps already defined, we just check the equality directly
  sorry

end sugar_fill_count_l276_276310


namespace jordyn_total_payment_l276_276644

theorem jordyn_total_payment :
  let price_cherries := 5
  let price_olives := 7
  let price_grapes := 11
  let num_cherries := 50
  let num_olives := 75
  let num_grapes := 25
  let discount_cherries := 0.12
  let discount_olives := 0.08
  let discount_grapes := 0.15
  let sales_tax := 0.05
  let service_charge := 0.02
  let total_cherries := num_cherries * price_cherries
  let total_olives := num_olives * price_olives
  let total_grapes := num_grapes * price_grapes
  let discounted_cherries := total_cherries * (1 - discount_cherries)
  let discounted_olives := total_olives * (1 - discount_olives)
  let discounted_grapes := total_grapes * (1 - discount_grapes)
  let subtotal := discounted_cherries + discounted_olives + discounted_grapes
  let taxed_amount := subtotal * (1 + sales_tax)
  let final_amount := taxed_amount * (1 + service_charge)
  final_amount = 1002.32 :=
by
  sorry

end jordyn_total_payment_l276_276644


namespace ratio_of_cans_l276_276683

theorem ratio_of_cans (martha_cans : ℕ) (total_required : ℕ) (remaining_cans : ℕ) (diego_cans : ℕ) (ratio : ℚ) 
  (h1 : martha_cans = 90) 
  (h2 : total_required = 150) 
  (h3 : remaining_cans = 5) 
  (h4 : martha_cans + diego_cans = total_required - remaining_cans) 
  (h5 : ratio = (diego_cans : ℚ) / martha_cans) : 
  ratio = 11 / 18 := 
by
  sorry

end ratio_of_cans_l276_276683


namespace cannot_use_square_difference_formula_l276_276897

theorem cannot_use_square_difference_formula (x y : ℝ) :
  ¬ ∃ a b : ℝ, (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) :=
sorry

end cannot_use_square_difference_formula_l276_276897


namespace Kendra_weekly_words_not_determined_without_weeks_l276_276504

def Kendra_goal : Nat := 60
def Kendra_already_learned : Nat := 36
def Kendra_needs_to_learn : Nat := 24

theorem Kendra_weekly_words_not_determined_without_weeks (weeks : Option Nat) : weeks = none → Kendra_needs_to_learn / weeks.getD 1 = 24 -> False := by
  sorry

end Kendra_weekly_words_not_determined_without_weeks_l276_276504


namespace janet_counts_total_birds_l276_276674

theorem janet_counts_total_birds :
  let crows := 30
  let hawks := crows + (60 / 100) * crows
  hawks + crows = 78 :=
by
  sorry

end janet_counts_total_birds_l276_276674


namespace exist_non_quadratic_residues_sum_l276_276516

noncomputable section

def is_quadratic_residue_mod (p a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 ≡ a [ZMOD p]

theorem exist_non_quadratic_residues_sum {p : ℤ} (hp : p > 5) (hp_modeq : p ≡ 1 [ZMOD 4]) (a : ℤ) : 
  ∃ b c : ℤ, a = b + c ∧ ¬is_quadratic_residue_mod p b ∧ ¬is_quadratic_residue_mod p c :=
sorry

end exist_non_quadratic_residues_sum_l276_276516


namespace simplify_expression_l276_276241

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x ≠ 3) :
  ((x - 5) / (x - 3) - ((x^2 + 2 * x + 1) / (x^2 + x)) / ((x + 1) / (x - 2)) = 
  -6 / (x^2 - 3 * x)) :=
by
  sorry

end simplify_expression_l276_276241


namespace value_of_x_plus_2y_l276_276636

theorem value_of_x_plus_2y 
  (x y : ℝ) 
  (h : (x + 5)^2 = -(|y - 2|)) : 
  x + 2 * y = -1 :=
sorry

end value_of_x_plus_2y_l276_276636


namespace exists_unique_xy_l276_276238

theorem exists_unique_xy (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end exists_unique_xy_l276_276238


namespace number_of_second_grade_students_l276_276882

theorem number_of_second_grade_students 
    (students_first_grade students_second_grade students_third_grade total_students total_volunteers : ℕ)
    (h_first : students_first_grade = 1200)
    (h_second : students_second_grade = 1000)
    (h_third : students_third_grade = 800)
    (h_total : total_students = students_first_grade + students_second_grade + students_third_grade)
    (h_volunteers : total_volunteers = 30) :
    (total_volunteers * students_second_grade / total_students) = 10 :=
by
  have h_total_eq : total_students = 3000 := by 
    rw [h_total, h_first, h_second, h_third]
    norm_num
  have h_proportion : (total_volunteers * students_second_grade / total_students : ℝ) = 
                      (30 * 1000 / 3000 : ℝ) := by
    rw [h_first, h_second, h_third]
  norm_num at h_proportion
  exact_mod_cast h_proportion

end number_of_second_grade_students_l276_276882


namespace copy_pages_15_dollars_l276_276668

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l276_276668


namespace sunset_time_l276_276686

theorem sunset_time (length_of_daylight : Nat := 11 * 60 + 18) -- length of daylight in minutes
    (sunrise : Nat := 6 * 60 + 32) -- sunrise time in minutes after midnight
    : (sunrise + length_of_daylight) % (24 * 60) = 17 * 60 + 50 := -- sunset time calculation
by
  sorry

end sunset_time_l276_276686


namespace parallel_lines_a_value_l276_276812

theorem parallel_lines_a_value (a : ℝ) 
  (h1 : ∀ x y : ℝ, x + a * y - 1 = 0 → x = a * (-4 * y - 2)) 
  : a = 2 :=
sorry

end parallel_lines_a_value_l276_276812


namespace perimeter_of_equilateral_triangle_l276_276876

theorem perimeter_of_equilateral_triangle (s : ℝ) 
  (h1 : (s ^ 2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_equilateral_triangle_l276_276876


namespace terminal_side_in_third_quadrant_l276_276952

open Real

theorem terminal_side_in_third_quadrant (θ : ℝ) (h1 : sin θ < 0) (h2 : cos θ < 0) : 
    θ ∈ Set.Ioo (π : ℝ) (3 * π / 2) := 
sorry

end terminal_side_in_third_quadrant_l276_276952


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l276_276716

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l276_276716


namespace minimum_handshakes_l276_276769

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l276_276769


namespace find_number_l276_276460

noncomputable def S (x : ℝ) : ℝ :=
  -- Assuming S(x) is a non-trivial function that sums the digits
  sorry

theorem find_number (x : ℝ) (hx_nonzero : x ≠ 0) (h_cond : x = (S x) / 5) : x = 1.8 :=
by
  sorry

end find_number_l276_276460


namespace fraction_product_eq_l276_276420

theorem fraction_product_eq : (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end fraction_product_eq_l276_276420


namespace find_side_a_l276_276063

theorem find_side_a (a b c : ℝ) (B : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 120) :
  a = Real.sqrt 2 :=
sorry

end find_side_a_l276_276063


namespace problem1_problem2_l276_276630

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * log (x + 1)

theorem problem1 (a : ℝ) : (∀ x : ℝ, (1 ≤ x → (2 * x^2 + 2 * x + a) / (x + 1) ≥ 0)) ↔ a ≥ -4 :=
sorry

theorem problem2 (a : ℝ) (x1 x2 : ℝ) (h0 : 0 < a) (h1 : a < 1 / 2) (h2 : x1 < x2)
  (h3 : ∃ x : ℝ, 2 * x ^ 2 + 2 * x + a = 0) :
  0 < (f x2 a) / x1 ∧ (f x2 a) / x1 < -1 / 2 + log 2 :=
sorry

end problem1_problem2_l276_276630


namespace floor_e_minus_3_eq_negative_one_l276_276002

theorem floor_e_minus_3_eq_negative_one 
  (e : ℝ) 
  (h : 2 < e ∧ e < 3) : 
  (⌊e - 3⌋ = -1) :=
by
  sorry

end floor_e_minus_3_eq_negative_one_l276_276002


namespace average_score_for_entire_class_l276_276831

theorem average_score_for_entire_class (n x y : ℕ) (a b : ℝ) (hn : n = 100) (hx : x = 70) (hy : y = 30) (ha : a = 0.65) (hb : b = 0.95) :
    ((x * a + y * b) / n) = 0.74 := by
  sorry

end average_score_for_entire_class_l276_276831


namespace smallest_three_digit_perfect_square_l276_276331

theorem smallest_three_digit_perfect_square :
  ∀ (a : ℕ), 100 ≤ a ∧ a < 1000 → (∃ (n : ℕ), 1001 * a + 1 = n^2) → a = 183 :=
by
  sorry

end smallest_three_digit_perfect_square_l276_276331


namespace michelle_silver_beads_l276_276073

theorem michelle_silver_beads :
  ∀ (total_beads blue_beads red_beads white_beads silver_beads : ℕ),
    total_beads = 40 →
    blue_beads = 5 →
    red_beads = 2 * blue_beads →
    white_beads = blue_beads + red_beads →
    silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
    silver_beads = 10 :=
by {
  intros total_beads blue_beads red_beads white_beads silver_beads,
  assume h1 h2 h3 h4 h5,
  sorry
}

end michelle_silver_beads_l276_276073


namespace derivative_at_one_l276_276467

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem derivative_at_one :
  deriv f 1 = -1 / 4 :=
by
  sorry

end derivative_at_one_l276_276467


namespace tangent_line_hyperbola_l276_276542

variable {a b x x₀ y y₀ : ℝ}
variable (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (he : x₀^2 / a^2 + y₀^2 / b^2 = 1)
variable (hh : x₀^2 / a^2 - y₀^2 / b^2 = 1)

theorem tangent_line_hyperbola
  (h_tangent_ellipse : (x₀ * x / a^2 + y₀ * y / b^2 = 1)) :
  (x₀ * x / a^2 - y₀ * y / b^2 = 1) :=
sorry

end tangent_line_hyperbola_l276_276542


namespace NutsInThirdBox_l276_276123

variable (x y z : ℝ)

theorem NutsInThirdBox (h1 : x = (y + z) - 6) (h2 : y = (x + z) - 10) : z = 16 := 
sorry

end NutsInThirdBox_l276_276123


namespace number_of_white_balls_l276_276997

theorem number_of_white_balls (a : ℕ) (h1 : 3 + a ≠ 0) (h2 : (3 : ℚ) / (3 + a) = 3 / 7) : a = 4 :=
sorry

end number_of_white_balls_l276_276997


namespace reading_time_per_week_l276_276561

variable (meditation_time_per_day : ℕ)
variable (reading_factor : ℕ)

theorem reading_time_per_week (h1 : meditation_time_per_day = 1) (h2 : reading_factor = 2) : 
  (reading_factor * meditation_time_per_day * 7) = 14 :=
by
  sorry

end reading_time_per_week_l276_276561


namespace large_painting_area_l276_276171

theorem large_painting_area :
  ∃ (large_painting : ℕ),
  (3 * (6 * 6) + 4 * (2 * 3) + large_painting = 282) → large_painting = 150 := by
  sorry

end large_painting_area_l276_276171


namespace intersection_of_A_and_B_l276_276948

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := sorry

end intersection_of_A_and_B_l276_276948


namespace Cornelia_current_age_l276_276645

theorem Cornelia_current_age (K : ℕ) (C : ℕ) (h1 : K = 20) (h2 : C + 10 = 3 * (K + 10)) : C = 80 :=
by
  sorry

end Cornelia_current_age_l276_276645


namespace meters_of_cloth_l276_276839

variable (total_cost cost_per_meter : ℝ)
variable (h1 : total_cost = 434.75)
variable (h2 : cost_per_meter = 47)

theorem meters_of_cloth : 
  total_cost / cost_per_meter = 9.25 := 
by
  sorry

end meters_of_cloth_l276_276839


namespace smallest_x_solution_l276_276189

def smallest_x_condition (x : ℝ) : Prop :=
  (x^2 - 5 * x - 84 = (x - 12) * (x + 7)) ∧
  (x ≠ 9) ∧
  (x ≠ -7) ∧
  ((x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 7))

theorem smallest_x_solution :
  ∃ x : ℝ, smallest_x_condition x ∧ ∀ y : ℝ, smallest_x_condition y → x ≤ y :=
sorry

end smallest_x_solution_l276_276189


namespace sum_ab_system_1_l276_276396

theorem sum_ab_system_1 {a b : ℝ} 
  (h1 : a^3 - a^2 + a - 5 = 0) 
  (h2 : b^3 - 2*b^2 + 2*b + 4 = 0) : 
  a + b = 1 := 
by 
  sorry

end sum_ab_system_1_l276_276396


namespace num_pos_divisors_36_l276_276968

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l276_276968


namespace find_a_l276_276811

-- Define the conditions
def parabola_equation (a : ℝ) (x : ℝ) : ℝ := a * x^2
def axis_of_symmetry : ℝ := -2

-- The main theorem: proving the value of a
theorem find_a (a : ℝ) : (axis_of_symmetry = - (1 / (4 * a))) → a = 1/8 :=
by
  intro h
  sorry

end find_a_l276_276811


namespace plane_through_point_and_line_l276_276326

noncomputable def plane_equation (x y z : ℝ) : Prop :=
  12 * x + 67 * y + 23 * z - 26 = 0

theorem plane_through_point_and_line :
  ∃ (A B C D : ℤ), 
  (A > 0) ∧ (Int.gcd (abs A) (Int.gcd (abs B) (Int.gcd (abs C) (abs D))) = 1) ∧
  (plane_equation 1 4 (-6)) ∧  
  ∀ t : ℝ, (plane_equation (4 * t + 2)  (-t - 1) (5 * t + 3)) :=
sorry

end plane_through_point_and_line_l276_276326


namespace max_area_of_rectangle_l276_276118

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l276_276118


namespace geometric_sequence_a4_l276_276495

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the geometric sequence

axiom a_2 : a 2 = -2
axiom a_6 : a 6 = -32
axiom geom_seq (n : ℕ) : a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geometric_sequence_a4 : a 4 = -8 := 
by
  sorry

end geometric_sequence_a4_l276_276495


namespace dilation_image_l276_276262

theorem dilation_image :
  let z_0 := (1 : ℂ) + 2 * I
  let k := (2 : ℂ)
  let z_1 := (3 : ℂ) + I
  let z := z_0 + k * (z_1 - z_0)
  z = 5 :=
by
  sorry

end dilation_image_l276_276262


namespace div_c_a_l276_276827

theorem div_c_a (a b c : ℚ) (h1 : a = 3 * b) (h2 : b = 2 / 5 * c) :
  c / a = 5 / 6 := 
by
  sorry

end div_c_a_l276_276827


namespace min_value_of_reciprocal_powers_l276_276322

theorem min_value_of_reciprocal_powers (t q a b : ℝ) (h1 : a + b = t)
  (h2 : a^2 + b^2 = t) (h3 : a^3 + b^3 = t) (h4 : a^4 + b^4 = t) :
  (a^2 = b^2) ∧ (a * b = q) ∧ ((1 / a^5) + (1 / b^5) = 128 * Real.sqrt 3 / 45) :=
by
  sorry

end min_value_of_reciprocal_powers_l276_276322


namespace quadratic_roots_distinct_real_l276_276174

theorem quadratic_roots_distinct_real (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = 0)
    (Δ : ℝ := b^2 - 4 * a * c) (hΔ : Δ > 0) :
    (∀ r1 r2 : ℝ, r1 ≠ r2) :=
by
  sorry

end quadratic_roots_distinct_real_l276_276174


namespace unique_solutions_xy_l276_276936

theorem unique_solutions_xy (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end unique_solutions_xy_l276_276936


namespace distinct_numbers_on_board_l276_276079

def count_distinct_numbers (Mila_divisors : ℕ) (Zhenya_divisors : ℕ) (common : ℕ) : ℕ :=
  Mila_divisors + Zhenya_divisors - (common - 1)

theorem distinct_numbers_on_board :
  count_distinct_numbers 10 9 2 = 13 := by
  sorry

end distinct_numbers_on_board_l276_276079


namespace largestValidNumberIs84_l276_276694

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l276_276694


namespace number_of_positive_divisors_360_l276_276206

theorem number_of_positive_divisors_360 : 
  let n := 360 
  in let prime_factors := [(2, 3), (3, 2), (5, 1)]
  in (∀ (p : ℕ) (a : ℕ), (p, a) ∈ prime_factors → p.prime) →
     (∀ m ∈ prime_factors, ∃ (p a : ℕ), m = (p, a) ∧ n = (p ^ a) * (prime_factors.filter (λ m', m ≠ m')).prod (λ m', (m'.fst ^ m'.snd))) →
     (prime_factors.foldr (λ (m : ℕ × ℕ) acc, (m.snd + 1) * acc) 1) = 24 := 
begin
  sorry
end

end number_of_positive_divisors_360_l276_276206


namespace expected_value_best_of_seven_games_correct_l276_276098

noncomputable def expected_value_best_of_seven_games : ℚ :=
  4 * (1 / 8) + 5 * (1 / 4) + 6 * (5 / 16) + 7 * (5 / 16)

theorem expected_value_best_of_seven_games_correct :
  expected_value_best_of_seven_games = 93 / 16 :=
by
  sorry

end expected_value_best_of_seven_games_correct_l276_276098


namespace max_area_of_rectangle_with_perimeter_40_l276_276111

theorem max_area_of_rectangle_with_perimeter_40 :
  (∃ (x y : ℝ), (2 * x + 2 * y = 40) ∧
                (∀ (a b : ℝ), (2 * a + 2 * b = 40) → (a * b ≤ x * y)) ∧
                (x * y = 100)) :=
begin
  -- Definitions of x and y satisfying the perimeter and maximizing the area.
  have h1 : ∀ (x y : ℝ), 2 * x + 2 * y = 40 → x * (20 - x) = -(x - 10)^2 + 100,
  { intro x, intro y, intro hper,
    have hy : y = 20 - x, by linarith,
    rw hy,
    ring },
  use 10,
  use 10,
  split,
  { -- Perimeter condition
    linarith },
  { split,
    { -- Maximum area condition
      intros a b hper,
      have hab : b = 20 - a, by linarith,
      rw hab,
      specialize h1 a (20 - a),
      linarith },
    { -- Maximum area is 100
      exact (by ring) } }
end

end max_area_of_rectangle_with_perimeter_40_l276_276111


namespace carrots_total_l276_276243

def carrots_grown_by_sally := 6
def carrots_grown_by_fred := 4
def total_carrots := carrots_grown_by_sally + carrots_grown_by_fred

theorem carrots_total : total_carrots = 10 := 
by 
  sorry  -- proof to be filled in

end carrots_total_l276_276243


namespace find_f2_l276_276544

-- Define the function f and the condition it satisfies
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def condition : Prop := ∀ x, x ≠ 1 / 3 → f x + f ((x + 1) / (1 - 3 * x)) = x

-- State the theorem to prove the value of f(2)
theorem find_f2 (h : condition f) : f 2 = 48 / 35 := 
by
  sorry

end find_f2_l276_276544


namespace lines_intersect_at_same_points_l276_276479

-- Definitions of linear equations in system 1 and system 2
def line1 (a1 b1 c1 x y : ℝ) := a1 * x + b1 * y = c1
def line2 (a2 b2 c2 x y : ℝ) := a2 * x + b2 * y = c2
def line3 (a3 b3 c3 x y : ℝ) := a3 * x + b3 * y = c3
def line4 (a4 b4 c4 x y : ℝ) := a4 * x + b4 * y = c4

-- Equivalence condition of the systems
def systems_equivalent (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :=
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y)

-- Proof statement that the four lines intersect at the same set of points
theorem lines_intersect_at_same_points (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :
  systems_equivalent a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 →
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y) :=
by
  intros h_equiv x y
  exact h_equiv x y

end lines_intersect_at_same_points_l276_276479


namespace range_of_a_l276_276805

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, (2 * a - 1 ≤ x ∧ x ≤ a + 3)) →
  (-1 ≤ a ∧ a ≤ 0) :=
by
  -- Prove the theorem
  sorry

end range_of_a_l276_276805


namespace hockey_pads_cost_l276_276654

variable (x h p remaining : ℝ)

theorem hockey_pads_cost :
  x = 150 ∧ h = x / 2 ∧ remaining = x - h - p ∧ remaining = 25 → p = 50 :=
by
  intro h₁
  cases h₁ with hx hh
  cases hh with hh₁ hh₂
  cases hh₂ with hr hr₁
  sorry

end hockey_pads_cost_l276_276654


namespace min_value_of_expression_l276_276132

theorem min_value_of_expression :
  ∀ (x y : ℝ), ∃ a b : ℝ, x = 5 ∧ y = -3 ∧ (x^2 + y^2 - 10*x + 6*y + 25) = -9 := 
by
  sorry

end min_value_of_expression_l276_276132


namespace number_of_divisors_of_36_l276_276966

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l276_276966


namespace angle_quadrant_l276_276828

theorem angle_quadrant (θ : Real) (P : Real × Real) (h : P = (Real.sin θ * Real.cos θ, 2 * Real.cos θ) ∧ P.1 < 0 ∧ P.2 < 0) :
  π / 2 < θ ∧ θ < π :=
by
  sorry

end angle_quadrant_l276_276828


namespace average_screen_time_per_player_l276_276861

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l276_276861


namespace volume_hemisphere_from_sphere_l276_276554

theorem volume_hemisphere_from_sphere (r : ℝ) (V_sphere : ℝ) (V_hemisphere : ℝ) 
  (h1 : V_sphere = 150 * Real.pi) 
  (h2 : V_sphere = (4 / 3) * Real.pi * r^3) : 
  V_hemisphere = 75 * Real.pi :=
by
  sorry

end volume_hemisphere_from_sphere_l276_276554


namespace probability_letter_in_mathematics_l276_276639

/-- 
Given that Lisa picks one letter randomly from the alphabet, 
prove that the probability that Lisa picks a letter in "MATHEMATICS" is 4/13.
-/
theorem probability_letter_in_mathematics :
  (8 : ℚ) / 26 = 4 / 13 :=
by
  sorry

end probability_letter_in_mathematics_l276_276639


namespace intersection_complement_R_M_and_N_l276_276271

open Set

def universalSet := ℝ
def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def complementR (S : Set ℝ) := {x : ℝ | x ∉ S}
def N := {x : ℝ | x < 1}

theorem intersection_complement_R_M_and_N:
  (complementR M ∩ N) = {x : ℝ | x < -2} := by
  sorry

end intersection_complement_R_M_and_N_l276_276271


namespace can_reach_4_white_l276_276655

/-
We define the possible states and operations on the urn as described.
-/

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def operation1 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation2 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation3 (u : Urn) : Urn :=
  { white := u.white - 1, black := u.black - 1 }

def operation4 (u : Urn) : Urn :=
  { white := u.white - 2, black := u.black + 1 }

theorem can_reach_4_white : ∃ (u : Urn), u.white = 4 ∧ u.black > 0 :=
  sorry

end can_reach_4_white_l276_276655


namespace glasses_in_smaller_box_l276_276165

variable (x : ℕ)

theorem glasses_in_smaller_box (h : (x + 16) / 2 = 15) : x = 14 :=
by
  sorry

end glasses_in_smaller_box_l276_276165


namespace kate_jenna_sticker_ratio_l276_276501

theorem kate_jenna_sticker_ratio :
  let k := 21
  let j := 12
  Nat.gcd k j = 3 ∧ k / Nat.gcd k j = 7 ∧ j / Nat.gcd k j = 4 :=
by
  let k := 21
  let j := 12
  have g : Nat.gcd k j = 3 := by sorry
  have hr : k / Nat.gcd k j = 7 := by sorry
  have jr : j / Nat.gcd k j = 4 := by sorry
  exact ⟨g, hr, jr⟩

end kate_jenna_sticker_ratio_l276_276501


namespace bus_speed_excluding_stoppages_l276_276184

theorem bus_speed_excluding_stoppages (v : ℝ) 
  (speed_including_stoppages : ℝ := 45) 
  (stoppage_time : ℝ := 1/6) 
  (h : v * (1 - stoppage_time) = speed_including_stoppages) : 
  v = 54 := 
by 
  sorry

end bus_speed_excluding_stoppages_l276_276184


namespace total_distance_to_run_l276_276681

theorem total_distance_to_run
  (track_length : ℕ)
  (initial_laps : ℕ)
  (additional_laps : ℕ)
  (total_laps := initial_laps + additional_laps) :
  track_length = 150 →
  initial_laps = 6 →
  additional_laps = 4 →
  total_laps * track_length = 1500 := by
  sorry

end total_distance_to_run_l276_276681


namespace num_pos_divisors_36_l276_276969

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l276_276969


namespace largestValidNumberIs84_l276_276697

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l276_276697


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276728

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276728


namespace tenth_term_arithmetic_seq_l276_276418

theorem tenth_term_arithmetic_seq :
  let a₁ : ℚ := 1 / 2
  let a₂ : ℚ := 5 / 6
  let d : ℚ := a₂ - a₁
  let a₁₀ : ℚ := a₁ + 9 * d
  a₁₀ = 7 / 2 :=
by
  sorry

end tenth_term_arithmetic_seq_l276_276418


namespace fraction_relationship_l276_276988

theorem fraction_relationship (a b c : ℚ)
  (h1 : a / b = 3 / 5)
  (h2 : b / c = 2 / 7) :
  c / a = 35 / 6 :=
by
  sorry

end fraction_relationship_l276_276988


namespace bubble_sort_probability_r10_r25_l276_276537

theorem bubble_sort_probability_r10_r25 (n : ℕ) (r : ℕ → ℕ) :
  n = 50 ∧ (∀ i, 1 ≤ i ∧ i ≤ 50 → r i ≠ r (i + 1)) ∧ (∀ i j, i ≠ j → r i ≠ r j) →
  let p := 1
  let q := 650
  p + q = 651 :=
by
  intros h
  sorry

end bubble_sort_probability_r10_r25_l276_276537


namespace remaining_value_subtract_70_percent_from_4500_l276_276049

theorem remaining_value_subtract_70_percent_from_4500 (num : ℝ) 
  (h : 0.36 * num = 2376) : 4500 - 0.70 * num = -120 :=
by
  sorry

end remaining_value_subtract_70_percent_from_4500_l276_276049


namespace trig_equation_solution_l276_276285

open Real

theorem trig_equation_solution (x : ℝ) (k n : ℤ) :
  (sin (2 * x)) ^ 4 + (sin (2 * x)) ^ 3 * (cos (2 * x)) -
  8 * (sin (2 * x)) * (cos (2 * x)) ^ 3 - 8 * (cos (2 * x)) ^ 4 = 0 ↔
  (∃ k : ℤ, x = -π / 8 + (π * k) / 2) ∨ 
  (∃ n : ℤ, x = (1 / 2) * arctan 2 + (π * n) / 2) := sorry

end trig_equation_solution_l276_276285


namespace find_f_neg_one_l276_276510

variable {R : Type} [LinearOrderedField R]

noncomputable def f (x : R) (m : R) : R :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : R) : (2^0 + 2 * 0 + m = 0) → f (-1) (-1) = -3 :=
by
  intro h1
  have h2 : m = -1 := by linarith
  rw [f, if_neg (show -1 >= 0, by linarith)]
  simp only [f, h2]
  norm_num
  sorry

end find_f_neg_one_l276_276510


namespace circle_area_from_circumference_l276_276260

theorem circle_area_from_circumference (C : ℝ) (A : ℝ) (hC : C = 36) (hCircumference : ∀ r, C = 2 * Real.pi * r) (hAreaFormula : ∀ r, A = Real.pi * r^2) :
  A = 324 / Real.pi :=
by
  sorry

end circle_area_from_circumference_l276_276260


namespace robot_trajectory_no_intersection_l276_276155

noncomputable def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def line_equation (x y k : ℝ) : Prop := y = k * (x + 1)

theorem robot_trajectory_no_intersection (k : ℝ) :
  (∀ x y : ℝ, parabola_equation x y → ¬ line_equation x y k) →
  (k > 1 ∨ k < -1) :=
by
  sorry

end robot_trajectory_no_intersection_l276_276155


namespace room_area_in_square_meters_l276_276436

theorem room_area_in_square_meters :
  ∀ (length_ft width_ft : ℝ), 
  (length_ft = 15) → 
  (width_ft = 8) → 
  (1 / 9 * 0.836127 = 0.092903) → 
  (length_ft * width_ft * 0.092903 = 11.14836) :=
by
  intros length_ft width_ft h_length h_width h_conversion
  -- sorry to skip the proof steps.
  sorry

end room_area_in_square_meters_l276_276436


namespace root_intervals_l276_276758

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem root_intervals (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ r1 r2 : ℝ, (a < r1 ∧ r1 < b ∧ f a b c r1 = 0) ∧ (b < r2 ∧ r2 < c ∧ f a b c r2 = 0) :=
sorry

end root_intervals_l276_276758


namespace inequality_solution_set_l276_276994

theorem inequality_solution_set (a : ℝ) : (∀ x : ℝ, x > 5 ∧ x > a ↔ x > 5) → a ≤ 5 :=
by
  sorry

end inequality_solution_set_l276_276994


namespace largest_three_digit_number_divisible_by_six_l276_276131

theorem largest_three_digit_number_divisible_by_six : ∃ n : ℕ, (∃ m < 1000, m ≥ 100 ∧ m % 6 = 0 ∧ m = n) ∧ (∀ k < 1000, k ≥ 100 ∧ k % 6 = 0 → k ≤ n) ∧ n = 996 :=
by sorry

end largest_three_digit_number_divisible_by_six_l276_276131


namespace simplify_log_expression_l276_276094

theorem simplify_log_expression : 
  (2 * log 4 3 + log 8 3) * (log 3 2 + log 9 2) = 2 := 
by 
  sorry

end simplify_log_expression_l276_276094


namespace simplify_expression_l276_276737

theorem simplify_expression :
  (360 / 24) * (10 / 240) * (6 / 3) * (9 / 18) = 5 / 8 := by
  sorry

end simplify_expression_l276_276737


namespace number_of_even_divisors_of_factorial_eight_l276_276039

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l276_276039


namespace molecular_weight_one_mole_of_AlOH3_l276_276570

variable (MW_7_moles : ℕ) (MW : ℕ)

theorem molecular_weight_one_mole_of_AlOH3 (h : MW_7_moles = 546) : MW = 78 :=
by
  sorry

end molecular_weight_one_mole_of_AlOH3_l276_276570


namespace num_pos_divisors_36_l276_276978

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l276_276978


namespace walking_distance_l276_276435

theorem walking_distance (a b : ℝ) (h1 : 10 * a + 45 * b = a * 15)
(h2 : x * (a + 9 * b) = 10 * a + 45 * b) : x = 13.5 :=
by
  sorry

end walking_distance_l276_276435


namespace integral_value_l276_276553

theorem integral_value : ∫ x in (1:ℝ)..(2:ℝ), (x^2 + 1) / x = (3 / 2) + Real.log 2 :=
by sorry

end integral_value_l276_276553


namespace solve_for_a_l276_276211

theorem solve_for_a (S P Q R : Type) (a b c d : ℝ) 
  (h1 : a + b + c + d = 360)
  (h2 : ∀ (PSQ : Type), d = 90) :
  a = 270 - b - c :=
by
  sorry

end solve_for_a_l276_276211


namespace product_of_three_numbers_l276_276409

theorem product_of_three_numbers (p q r m : ℝ) (h1 : p + q + r = 180) (h2 : m = 8 * p)
  (h3 : m = q - 10) (h4 : m = r + 10) : p * q * r = 90000 := by
  sorry

end product_of_three_numbers_l276_276409


namespace suki_bags_l276_276742

theorem suki_bags (bag_weight_suki : ℕ) (bag_weight_jimmy : ℕ) (containers : ℕ) 
  (container_weight : ℕ) (num_bags_jimmy : ℝ) (num_containers : ℕ)
  (h1 : bag_weight_suki = 22) 
  (h2 : bag_weight_jimmy = 18) 
  (h3 : container_weight = 8) 
  (h4 : num_bags_jimmy = 4.5)
  (h5 : num_containers = 28) : 
  6 = ⌊(num_containers * container_weight - num_bags_jimmy * bag_weight_jimmy) / bag_weight_suki⌋ :=
by
  sorry

end suki_bags_l276_276742


namespace cylinder_surface_area_is_128pi_l276_276910

noncomputable def cylinder_total_surface_area (h r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem cylinder_surface_area_is_128pi :
  cylinder_total_surface_area 12 4 = 128 * Real.pi :=
by
  sorry

end cylinder_surface_area_is_128pi_l276_276910


namespace mixing_paint_l276_276682

theorem mixing_paint (total_parts : ℕ) (blue_parts : ℕ) (red_parts : ℕ) (white_parts : ℕ) (blue_ounces : ℕ) (max_mixture : ℕ) (ounces_per_part : ℕ) :
  total_parts = blue_parts + red_parts + white_parts →
  blue_parts = 7 →
  red_parts = 2 →
  white_parts = 1 →
  blue_ounces = 140 →
  max_mixture = 180 →
  ounces_per_part = blue_ounces / blue_parts →
  max_mixture / ounces_per_part = 9 →
  white_ounces = white_parts * ounces_per_part →
  white_ounces = 20 :=
sorry

end mixing_paint_l276_276682


namespace solve_inequalities_l276_276250

/-- Solve the inequality system and find all non-negative integer solutions. -/
theorem solve_inequalities :
  { x : ℤ | 0 ≤ x ∧ 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {0, 1, 2} :=
by
  sorry

end solve_inequalities_l276_276250


namespace logan_passengers_count_l276_276901

noncomputable def passengers_used_Kennedy_Airport : ℝ := (1 / 3) * 38.3
noncomputable def passengers_used_Miami_Airport : ℝ := (1 / 2) * passengers_used_Kennedy_Airport
noncomputable def passengers_used_Logan_Airport : ℝ := passengers_used_Miami_Airport / 4

theorem logan_passengers_count : abs (passengers_used_Logan_Airport - 1.6) < 0.01 := by
  sorry

end logan_passengers_count_l276_276901


namespace solve_system_eqs_l276_276095

theorem solve_system_eqs : 
    ∃ (x y z : ℚ), 
    4 * x - 3 * y + z = -10 ∧
    3 * x + 5 * y - 2 * z = 8 ∧
    x - 2 * y + 7 * z = 5 ∧
    x = -51 / 61 ∧ 
    y = 378 / 61 ∧ 
    z = 728 / 61 := by
  sorry

end solve_system_eqs_l276_276095


namespace focus_of_parabola_y_eq_9x2_plus_6_l276_276173

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, b + (1 / (4 * a)))

theorem focus_of_parabola_y_eq_9x2_plus_6 :
  focus_of_parabola 9 6 = (0, 217 / 36) :=
by
  sorry

end focus_of_parabola_y_eq_9x2_plus_6_l276_276173


namespace find_ratio_of_a_b_l276_276626

noncomputable def slope_of_tangent_to_curve_at_P := 3 * 1^2 + 1

noncomputable def perpendicular_slope (a b : ℝ) : Prop :=
  slope_of_tangent_to_curve_at_P * (a / b) = -1

noncomputable def line_slope_eq_slope_of_tangent (a b : ℝ) : Prop := 
  perpendicular_slope a b

theorem find_ratio_of_a_b (a b : ℝ) 
  (h1 : a - b * 2 = 0) 
  (h2 : line_slope_eq_slope_of_tangent a b) : 
  a / b = -1 / 4 :=
by
  sorry

end find_ratio_of_a_b_l276_276626


namespace regular_polygon_sides_l276_276838

theorem regular_polygon_sides 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : B = 3 * A)
  (h₃ : C = 6 * A) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end regular_polygon_sides_l276_276838


namespace equilateral_triangle_perimeter_twice_side_area_l276_276875

noncomputable def triangle_side_length (s : ℝ) :=
  s * s * Real.sqrt 3 / 4 = 2 * s

noncomputable def triangle_perimeter (s : ℝ) := 3 * s

theorem equilateral_triangle_perimeter_twice_side_area (s : ℝ) (h : triangle_side_length s) : 
  triangle_perimeter s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_twice_side_area_l276_276875


namespace amount_transferred_l276_276600

def original_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : original_balance - remaining_balance = 69 :=
by
  sorry

end amount_transferred_l276_276600


namespace arith_seq_common_diff_l276_276992

/-
Given:
- an arithmetic sequence {a_n} with common difference d,
- the sum of the first n terms S_n = n * a_1 + n * (n - 1) / 2 * d,
- b_n = S_n / n,

Prove that the common difference of the sequence {a_n - b_n} is d/2.
-/

theorem arith_seq_common_diff (a b : ℕ → ℚ) (a1 d : ℚ) 
  (h1 : ∀ n, a n = a1 + n * d) 
  (h2 : ∀ n, b n = (a1 + n - 1 * d + n * (n - 1) / 2 * d) / n) : 
  ∀ n, (a n - b n) - (a (n + 1) - b (n + 1)) = d / 2 := 
    sorry

end arith_seq_common_diff_l276_276992


namespace pages_copied_for_15_dollars_l276_276662

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l276_276662


namespace set_intersection_l276_276033

theorem set_intersection (M N : Set ℝ) (hM : M = {x | x < 3}) (hN : N = {x | x > 2}) :
  M ∩ N = {x | 2 < x ∧ x < 3} :=
sorry

end set_intersection_l276_276033


namespace gwen_walked_time_l276_276458

-- Definition of given conditions
def time_jogged : ℕ := 15
def ratio_jogged_to_walked (j w : ℕ) : Prop := j * 3 = w * 5

-- Definition to state the exact time walked with given ratio
theorem gwen_walked_time (j w : ℕ) (h1 : j = time_jogged) (h2 : ratio_jogged_to_walked j w) : w = 9 :=
by
  sorry

end gwen_walked_time_l276_276458


namespace base_conversion_subtraction_l276_276599

def base6_to_base10 (n : ℕ) : ℕ :=
3 * (6^2) + 2 * (6^1) + 5 * (6^0)

def base5_to_base10 (m : ℕ) : ℕ :=
2 * (5^2) + 3 * (5^1) + 1 * (5^0)

theorem base_conversion_subtraction : 
  base6_to_base10 325 - base5_to_base10 231 = 59 :=
by
  sorry

end base_conversion_subtraction_l276_276599


namespace find_a_and_monotonicity_range_of_k_l276_276950

noncomputable def f (x : ℝ) : ℝ := (- 2 ^ x + 1) / (2 ^ x + 1)

theorem find_a_and_monotonicity (a : ℝ) :
  (∀ x : ℝ, f x = (a * 2^x + 1) / (2^x + 1)) →
  (∀ x : ℝ, f (-x) = - f x) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) :=
sorry

theorem range_of_k (k : ℝ) :
  (∃ t : ℝ, t ∈ set.Icc 1 2 ∧ f (t^2 - 2*t) + f (2*t^2 - k) > 0) →
  k ∈ set.Ioi 1 :=
sorry

end find_a_and_monotonicity_range_of_k_l276_276950


namespace total_carrots_l276_276244

theorem total_carrots (sally_carrots fred_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : fred_carrots = 4) : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l276_276244


namespace abs_of_neg_square_add_l276_276738

theorem abs_of_neg_square_add (a b : ℤ) : |-a^2 + b| = 10 :=
by
  sorry

end abs_of_neg_square_add_l276_276738


namespace Michelle_silver_beads_count_l276_276075

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end Michelle_silver_beads_count_l276_276075


namespace find_x_value_l276_276347

-- Define the conditions and the proof problem as Lean 4 statement
theorem find_x_value 
  (k : ℚ)
  (h1 : ∀ (x y : ℚ), (2 * x - 3) / (2 * y + 10) = k)
  (h2 : (2 * 4 - 3) / (2 * 5 + 10) = k)
  : (∃ x : ℚ, (2 * x - 3) / (2 * 10 + 10) = k) ↔ x = 5.25 :=
by
  sorry

end find_x_value_l276_276347


namespace min_handshakes_l276_276763

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l276_276763


namespace circumradius_of_triangle_l276_276773

theorem circumradius_of_triangle (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 6) (h₃ : c = 10) 
  (h₄ : a^2 + b^2 = c^2) : 
  (c : ℝ) / 2 = 5 := 
by {
  -- proof goes here
  sorry
}

end circumradius_of_triangle_l276_276773


namespace closest_point_on_graph_l276_276083

theorem closest_point_on_graph (x y : ℝ) (h1 : x > 0) (h2 : y = x + 1/x) :
  (x = 1/real.root 4 2) ∧ (y = (1 + real.sqrt 2) / real.root 4 2) :=
sorry

end closest_point_on_graph_l276_276083


namespace total_flowers_sold_l276_276304

-- Definitions for conditions
def roses_per_bouquet : ℕ := 12
def daisies_per_bouquet : ℕ := 12  -- Assuming each daisy bouquet contains the same number of daisies as roses
def total_bouquets : ℕ := 20
def rose_bouquets_sold : ℕ := 10
def daisy_bouquets_sold : ℕ := 10

-- Statement of the equivalent Lean theorem
theorem total_flowers_sold :
  (rose_bouquets_sold * roses_per_bouquet) + (daisy_bouquets_sold * daisies_per_bouquet) = 240 :=
by
  sorry

end total_flowers_sold_l276_276304


namespace simplify_and_evaluate_expression_l276_276529

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 3) : 
  ((x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4))) = (5 : ℝ) / 3 := 
by
  sorry

end simplify_and_evaluate_expression_l276_276529


namespace points_distance_le_sqrt5_l276_276569

theorem points_distance_le_sqrt5 :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 4) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end points_distance_le_sqrt5_l276_276569


namespace Alyssa_puppies_l276_276162

theorem Alyssa_puppies (initial_puppies : ℕ) (given_puppies : ℕ)
  (h_initial : initial_puppies = 7) (h_given : given_puppies = 5) :
  initial_puppies - given_puppies = 2 :=
by
  sorry

end Alyssa_puppies_l276_276162


namespace billy_distance_l276_276446

-- Definitions
def distance_billy_spit (b : ℝ) : ℝ := b
def distance_madison_spit (m : ℝ) (b : ℝ) : Prop := m = 1.20 * b
def distance_ryan_spit (r : ℝ) (m : ℝ) : Prop := r = 0.50 * m

-- Conditions
variables (m : ℝ) (b : ℝ) (r : ℝ)
axiom madison_farther: distance_madison_spit m b
axiom ryan_shorter: distance_ryan_spit r m
axiom ryan_distance: r = 18

-- Proof problem
theorem billy_distance : b = 30 := by
  sorry

end billy_distance_l276_276446


namespace number_of_divisors_36_l276_276971

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l276_276971


namespace num_pos_divisors_36_l276_276963

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l276_276963


namespace sum_of_prime_factors_is_prime_l276_276267

/-- Define the specific number in question -/
def num := 30030

/-- List the prime factors of the number -/
def prime_factors := [2, 3, 5, 7, 11, 13]

/-- Sum of the prime factors -/
def sum_prime_factors := prime_factors.sum

theorem sum_of_prime_factors_is_prime :
  sum_prime_factors = 41 ∧ Prime 41 := 
by
  -- The conditions are encapsulated in the definitions above
  -- Now, establish the required proof goal using these conditions
  sorry

end sum_of_prime_factors_is_prime_l276_276267


namespace handshakes_minimum_l276_276768

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l276_276768


namespace copy_pages_15_dollars_l276_276667

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l276_276667


namespace volume_of_dug_earth_l276_276299

theorem volume_of_dug_earth :
  let r := 2
  let h := 14
  ∃ V : ℝ, V = Real.pi * r^2 * h ∧ V = 56 * Real.pi :=
by
  sorry

end volume_of_dug_earth_l276_276299


namespace product_of_invertible_labels_l276_276105

def f1 (x : ℤ) : ℤ := x^3 - 2 * x
def f2 (x : ℤ) : ℤ := x - 2
def f3 (x : ℤ) : ℤ := 2 - x

theorem product_of_invertible_labels :
  (¬ ∃ inv : ℤ → ℤ, f1 (inv 0) = 0 ∧ ∀ x : ℤ, f1 (inv (f1 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f2 (inv 0) = 0 ∧ ∀ x : ℤ, f2 (inv (f2 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f3 (inv 0) = 0 ∧ ∀ x : ℤ, f3 (inv (f3 x)) = x) →
  (2 * 3 = 6) :=
by sorry

end product_of_invertible_labels_l276_276105


namespace determinant_sum_is_34_l276_276925

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![5, -2],
  ![3, 4]
]

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 3],
  ![-1, 2]
]

-- Prove the determinant of the sum of A and B is 34
theorem determinant_sum_is_34 : Matrix.det (A + B) = 34 := by
  sorry

end determinant_sum_is_34_l276_276925


namespace tetrahedron_solution_l276_276469

noncomputable def num_triangles (a : ℝ) (E F G : ℝ → ℝ → ℝ) : ℝ :=
  if a > 3 then 3 else 0

theorem tetrahedron_solution (a : ℝ) (E F G : ℝ → ℝ → ℝ) :
  a > 3 → num_triangles a E F G = 3 := by
  sorry

end tetrahedron_solution_l276_276469


namespace relationship_between_y_values_l276_276524

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variables (m : ℝ) (y1 y2 y3 : ℝ)
variables (h : m > 0)
variables (h1 : y1 = quadratic_function m (-1))
variables (h2 : y2 = quadratic_function m (5 / 2))
variables (h3 : y3 = quadratic_function m 6)

theorem relationship_between_y_values : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end relationship_between_y_values_l276_276524


namespace train_speed_is_45_kmph_l276_276437

noncomputable def speed_of_train_kmph (train_length bridge_length total_time : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := total_distance / total_time
  let speed_kmph := speed_mps * 36 / 10
  speed_kmph

theorem train_speed_is_45_kmph :
  speed_of_train_kmph 150 225 30 = 45 :=
  sorry

end train_speed_is_45_kmph_l276_276437


namespace jackie_phil_probability_l276_276065

noncomputable def probability_same_heads : ℚ :=
  let fair_coin := (1 + 1: ℚ)
  let p3_coin := (2 + 3: ℚ)
  let p2_coin := (1 + 2: ℚ)
  let generating_function := fair_coin * p3_coin * p2_coin
  let sum_of_coefficients := 30
  let sum_of_squares_of_coefficients := 290
  sum_of_squares_of_coefficients / (sum_of_coefficients ^ 2)

theorem jackie_phil_probability : probability_same_heads = 29 / 90 := by
  sorry

end jackie_phil_probability_l276_276065


namespace susan_gave_sean_8_apples_l276_276735

theorem susan_gave_sean_8_apples (initial_apples total_apples apples_given : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : total_apples = 17)
  (h3 : apples_given = total_apples - initial_apples) : 
  apples_given = 8 :=
by
  sorry

end susan_gave_sean_8_apples_l276_276735


namespace lemon_ratio_l276_276517

variable (Levi Jayden Eli Ian : ℕ)

theorem lemon_ratio (h1: Levi = 5)
    (h2: Jayden = Levi + 6)
    (h3: Jayden = Eli / 3)
    (h4: Levi + Jayden + Eli + Ian = 115) :
    Eli = Ian / 2 :=
by
  sorry

end lemon_ratio_l276_276517


namespace largest_two_digit_divisible_by_6_ending_in_4_l276_276734

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l276_276734


namespace bar_weight_calc_l276_276444

variable (blue_weight green_weight num_blue_weights num_green_weights bar_weight total_weight : ℕ)

theorem bar_weight_calc
  (h1 : blue_weight = 2)
  (h2 : green_weight = 3)
  (h3 : num_blue_weights = 4)
  (h4 : num_green_weights = 5)
  (h5 : total_weight = 25)
  (weights_total := num_blue_weights * blue_weight + num_green_weights * green_weight)
  : bar_weight = total_weight - weights_total :=
by
  sorry

end bar_weight_calc_l276_276444


namespace positive_integers_dividing_10n_l276_276946

-- Define the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the predicate that checks if the sum of the first n positive integers evenly divides 10n
def evenly_divides_10n (n : ℕ) : Prop :=
  10 * n % sum_first_n n = 0

-- Define the proof statement that there are exactly 5 positive integers n where the sum evenly divides 10n
theorem positive_integers_dividing_10n : (finset.range 20).filter (λ n, n > 0 ∧ evenly_divides_10n n).card = 5 :=
by
  sorry

end positive_integers_dividing_10n_l276_276946


namespace closest_point_l276_276087

noncomputable def closest_point_to_origin : ℝ × ℝ :=
  let x := (1 : ℝ) / Real.root 2 4 in
  let y := x + 1 / x in
  (x, y)

theorem closest_point (x y : ℝ) (h : y = x + 1 / x) (hx : x > 0) :
  (x, y) = closest_point_to_origin :=
begin
  sorry
end

end closest_point_l276_276087


namespace dentist_age_considered_years_ago_l276_276234

theorem dentist_age_considered_years_ago (A : ℕ) (X : ℕ) (H1 : A = 32) (H2 : (1/6 : ℚ) * (A - X) = (1/10 : ℚ) * (A + 8)) : X = 8 :=
sorry

end dentist_age_considered_years_ago_l276_276234


namespace speed_of_man_l276_276438

theorem speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ)
  (relative_speed_km_h : ℝ)
  (h_train_length : train_length = 440)
  (h_train_speed : train_speed_kmph = 60)
  (h_time : time_seconds = 24)
  (h_relative_speed : relative_speed_km_h = (train_length / time_seconds) * 3.6):
  (relative_speed_km_h - train_speed_kmph) = 6 :=
by sorry

end speed_of_man_l276_276438


namespace bisector_intersects_varies_l276_276647

open Real EuclideanGeometry Circle Segment

-- Define the problem conditions
variables {O A B C D : Point} 
variables {r : ℝ}  -- radius

-- Fixed chord AB of the circle
axiom fixed_chord : Segment A B

-- Point C is any point on the circle with center O and radius r
axiom point_on_circle : Circle O r.contains C

-- Chord CD forms a 30º angle with chord AB
axiom angle_condition : ∃ D, Segment C D ∧ Angle (C, O, D) = 30°

-- Define quadrant traversal for point C
axiom quadrant_traversal : ∀ θ, (0 ≤ θ) ∧ (θ ≤ π/2) → Circle O r.radius_point θ = C

-- State the theorem to be proved
theorem bisector_intersects_varies :
  ∃ P, (∉ {A, B}) ∧ (is_bisector_point P (Circle O r) (Segment O C) (Segment C D)) → varies :=
sorry

end bisector_intersects_varies_l276_276647


namespace sin_alpha_trig_expression_l276_276028

theorem sin_alpha {α : ℝ} (hα : ∃ P : ℝ × ℝ, P = (4/5, -3/5)) :
  Real.sin α = -3/5 :=
sorry

theorem trig_expression {α : ℝ} 
  (hα : Real.sin α = -3/5) : 
  (Real.sin (π / 2 - α) / Real.sin (α + π)) - 
  (Real.tan (α - π) / Real.cos (3 * π - α)) = 19 / 48 :=
sorry

end sin_alpha_trig_expression_l276_276028


namespace jason_initial_cards_l276_276373

theorem jason_initial_cards (a : ℕ) (b : ℕ) (x : ℕ) : 
  a = 224 → 
  b = 452 → 
  x = a + b → 
  x = 676 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jason_initial_cards_l276_276373


namespace inequality_solution_maximum_expression_l276_276142

-- Problem 1: Inequality for x
theorem inequality_solution (x : ℝ) : |x + 1| + 2 * |x - 1| < 3 * x + 5 ↔ x > -1/2 :=
by
  sorry

-- Problem 2: Maximum value for expression within [0, 1]
theorem maximum_expression (a b : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) : 
  ab + (1 - a - b) * (a + b) ≤ 1/3 :=
by
  sorry

end inequality_solution_maximum_expression_l276_276142


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l276_276712

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l276_276712


namespace unique_triple_solution_l276_276937

theorem unique_triple_solution (x y z : ℝ) :
  (1 + x^4 ≤ 2 * (y - z)^2) →
  (1 + y^4 ≤ 2 * (z - x)^2) →
  (1 + z^4 ≤ 2 * (x - y)^2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
sorry

end unique_triple_solution_l276_276937


namespace sum_of_coefficients_is_60_l276_276543

theorem sum_of_coefficients_is_60 :
  ∀ (a b c d e : ℤ), (∀ x : ℤ, 512 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) →
  a + b + c + d + e = 60 :=
by
  intros a b c d e h
  sorry

end sum_of_coefficients_is_60_l276_276543


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l276_276714

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l276_276714


namespace arithmetic_sequence_20th_term_l276_276602

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 3
  let n := 20
  (a + (n - 1) * d) = 59 :=
by 
  sorry

end arithmetic_sequence_20th_term_l276_276602


namespace average_velocity_instantaneous_velocity_l276_276151

noncomputable def s (t : ℝ) : ℝ := 8 - 3 * t^2

theorem average_velocity {Δt : ℝ} (h : Δt ≠ 0) :
  (s (1 + Δt) - s 1) / Δt = -6 - 3 * Δt :=
sorry

theorem instantaneous_velocity :
  deriv s 1 = -6 :=
sorry

end average_velocity_instantaneous_velocity_l276_276151


namespace part1_part2_part3_l276_276631

noncomputable def f (x m : ℝ) : ℝ :=
  -x^2 + m*x - m

-- Part (1)
theorem part1 (m : ℝ) : (∀ x, f x m ≤ 0) → (m = 0 ∨ m = 4) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x, -1 ≤ x ∧ x ≤ 0 → f x m ≤ f (-1) m) → (m ≤ -2) :=
sorry

-- Part (3)
theorem part3 : ∃ (m : ℝ), (∀ x, 2 ≤ x ∧ x ≤ 3 → (2 ≤ f x m ∧ f x m ≤ 3)) → m = 6 :=
sorry

end part1_part2_part3_l276_276631


namespace hockey_pads_cost_l276_276653

theorem hockey_pads_cost
  (initial_money : ℕ)
  (cost_hockey_skates : ℕ)
  (remaining_money : ℕ)
  (h : initial_money = 150)
  (h1 : cost_hockey_skates = initial_money / 2)
  (h2 : remaining_money = 25) :
  initial_money - cost_hockey_skates - 50 = remaining_money :=
by sorry

end hockey_pads_cost_l276_276653


namespace initial_amount_celine_had_l276_276154

-- Define the costs and quantities
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def change_received : ℕ := 200

-- Calculate costs and total amount
def cost_laptops : ℕ := num_laptops * laptop_cost
def cost_smartphones : ℕ := num_smartphones * smartphone_cost
def total_cost : ℕ := cost_laptops + cost_smartphones
def initial_amount : ℕ := total_cost + change_received

-- The statement to prove
theorem initial_amount_celine_had : initial_amount = 3000 := by
  sorry

end initial_amount_celine_had_l276_276154


namespace integer_roots_l276_276508

noncomputable def is_quadratic_root (p q x : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem integer_roots (p q x1 x2 : ℝ)
  (hq1 : is_quadratic_root p q x1)
  (hq2 : is_quadratic_root p q x2)
  (hd : x1 ≠ x2)
  (hx : |x1 - x2| = 1)
  (hpq : |p - q| = 1) :
  (∃ (p_int q_int x1_int x2_int : ℤ), 
      p = p_int ∧ q = q_int ∧ x1 = x1_int ∧ x2 = x2_int) :=
sorry

end integer_roots_l276_276508


namespace smallest_q_p_l276_276069

noncomputable def q_p_difference : ℕ := 3

theorem smallest_q_p (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : 5 * q < 9 * p) (h2 : 9 * p < 5 * q) : q - p = q_p_difference → q = 7 :=
by
  sorry

end smallest_q_p_l276_276069


namespace solveExpression_l276_276906

noncomputable def evaluateExpression : ℝ := (Real.sqrt 3) / Real.sin (Real.pi / 9) - 1 / Real.sin (7 * Real.pi / 18)

theorem solveExpression : evaluateExpression = 4 :=
by sorry

end solveExpression_l276_276906


namespace necessary_but_not_sufficient_l276_276468

variable (p q : Prop)

theorem necessary_but_not_sufficient (hp : p) : p ∧ q ↔ p ∧ (p ∧ q → q) :=
  sorry

end necessary_but_not_sufficient_l276_276468


namespace four_digit_numbers_l276_276432

open Finset

theorem four_digit_numbers :
  let digits := {1, 2, 3, 5}
  let digit_list := (digits : Finset ℕ).to_list
  let permutations := digit_list.permutations
  ∃ total_numbers odd_numbers even_numbers numbers_gt_2000,
  (total_numbers = permutations.length) ∧
  (odd_numbers = permutations.count (λ l, l[3] ∈ {1, 3, 5})) ∧
  (even_numbers = permutations.count (λ l, l[3] = 2)) ∧
  (numbers_gt_2000 = permutations.count (λ l, l[0] ∈ {2, 3, 5})) ∧
  (total_numbers = 24) ∧
  (odd_numbers = 18) ∧
  (even_numbers = 6) ∧
  (numbers_gt_2000 = 18) :=
by
  sorry

end four_digit_numbers_l276_276432


namespace no_integer_solution_exists_l276_276787

theorem no_integer_solution_exists :
  ¬ ∃ m n : ℤ, m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_integer_solution_exists_l276_276787


namespace circle_area_from_circumference_l276_276258

theorem circle_area_from_circumference (r : ℝ) (π : ℝ) (h1 : 2 * π * r = 36) : (π * (r^2) = 324 / π) := by
  sorry

end circle_area_from_circumference_l276_276258


namespace fourth_ball_black_probability_l276_276423

noncomputable def prob_fourth_is_black : Prop :=
  let total_balls := 8
  let black_balls := 4
  let prob_black := black_balls / total_balls
  prob_black = 1 / 2

theorem fourth_ball_black_probability :
  prob_fourth_is_black :=
sorry

end fourth_ball_black_probability_l276_276423


namespace decagon_diagonals_l276_276176

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l276_276176


namespace Michelle_silver_beads_count_l276_276076

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end Michelle_silver_beads_count_l276_276076


namespace prob_ending_game_after_five_distribution_and_expectation_l276_276297

-- Define the conditions
def shooting_accuracy_rate : ℚ := 2 / 3
def game_clear_coupon : ℕ := 9
def game_fail_coupon : ℕ := 3
def game_no_clear_no_fail_coupon : ℕ := 6

-- Define the probabilities for ending the game after 5 shots
def ending_game_after_five : ℚ := (shooting_accuracy_rate^2 * (1 - shooting_accuracy_rate)^3 * 2) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate))

-- Define the distribution table
def P_clear : ℚ := (shooting_accuracy_rate^3) + (shooting_accuracy_rate^3 * (1 - shooting_accuracy_rate)) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate) * 2)
def P_fail : ℚ := ((1 - shooting_accuracy_rate)^2) + ((1 - shooting_accuracy_rate)^2 * shooting_accuracy_rate * 2) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^2 * 3) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^3)
def P_neither : ℚ := 1 - P_clear - P_fail

-- Expected value calculation
def expectation : ℚ := (P_fail * game_fail_coupon) + (P_neither * game_no_clear_no_fail_coupon) + (P_clear * game_clear_coupon)

-- The Part I proof statement
theorem prob_ending_game_after_five : ending_game_after_five = 8 / 81 :=
by
  sorry

-- The Part II proof statement
theorem distribution_and_expectation (X : ℕ → ℚ) :
  (X game_fail_coupon = 233 / 729) ∧
  (X game_no_clear_no_fail_coupon = 112 / 729) ∧
  (X game_clear_coupon = 128 / 243) ∧
  (expectation = 1609 / 243) :=
by
  sorry

end prob_ending_game_after_five_distribution_and_expectation_l276_276297


namespace sqrt_meaningful_range_l276_276642

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x = Real.sqrt (a + 2)) ↔ a ≥ -2 := 
sorry

end sqrt_meaningful_range_l276_276642


namespace div_identity_l276_276825

theorem div_identity (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := by
  sorry

end div_identity_l276_276825


namespace min_value_expression_l276_276071

theorem min_value_expression {x y z w : ℝ} 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) 
  (hw : 0 ≤ w ∧ w ≤ 1) : 
  ∃ m, m = 2 ∧ ∀ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) →
  m ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) :=
by
  sorry

end min_value_expression_l276_276071


namespace minimum_value_of_expression_l276_276070

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) :
  ∃ P, (P = (x / y + y / z + z / x) * (y / x + z / y + x / z)) ∧ P = 25 := 
by sorry

end minimum_value_of_expression_l276_276070


namespace total_actions_135_l276_276388

theorem total_actions_135
  (y : ℕ) -- represents the total number of actions
  (h1 : y ≥ 10) -- since there are at least 10 initial comments
  (h2 : ∀ (likes dislikes : ℕ), likes + dislikes = y - 10) -- total votes exclude neutral comments
  (score_eq : ∀ (likes dislikes : ℕ), 70 * dislikes = 30 * likes)
  (score_50 : ∀ (likes dislikes : ℕ), 50 = likes - dislikes) :
  y = 135 :=
by {
  sorry
}

end total_actions_135_l276_276388


namespace f_divisible_by_k2_k1_l276_276385

noncomputable def f (n : ℕ) (x : ℤ) : ℤ :=
  x^(n + 2) + (x + 1)^(2 * n + 1)

theorem f_divisible_by_k2_k1 (n : ℕ) (k : ℤ) (hn : n > 0) : 
  ((k^2 + k + 1) ∣ f n k) :=
sorry

end f_divisible_by_k2_k1_l276_276385


namespace find_num_pennies_l276_276986

def total_value (nickels : ℕ) (dimes : ℕ) (pennies : ℕ) : ℕ :=
  5 * nickels + 10 * dimes + pennies

def num_pennies (nickels_value: ℕ) (dimes_value: ℕ) (total: ℕ): ℕ :=
  total - (nickels_value + dimes_value)

theorem find_num_pennies : 
  ∀ (total : ℕ) (num_nickels : ℕ) (num_dimes: ℕ),
  total = 59 → num_nickels = 4 → num_dimes = 3 → num_pennies (5 * num_nickels) (10 * num_dimes) total = 9 :=
by
  intros
  sorry

end find_num_pennies_l276_276986


namespace goods_train_length_is_280_meters_l276_276149

def speed_of_man_train_kmph : ℝ := 80
def speed_of_goods_train_kmph : ℝ := 32
def time_to_pass_seconds : ℝ := 9

theorem goods_train_length_is_280_meters :
  let relative_speed_kmph := speed_of_man_train_kmph + speed_of_goods_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let length_of_goods_train := relative_speed_mps * time_to_pass_seconds
  abs (length_of_goods_train - 280) < 1 :=
by
  -- skipping the proof
  sorry

end goods_train_length_is_280_meters_l276_276149


namespace SUCCESSOR_arrangement_count_l276_276930

theorem SUCCESSOR_arrangement_count :
  (Nat.factorial 9) / (Nat.factorial 3 * Nat.factorial 2) = 30240 :=
by
  sorry

end SUCCESSOR_arrangement_count_l276_276930


namespace average_of_four_given_conditions_l276_276099

noncomputable def average_of_four_integers : ℕ × ℕ × ℕ × ℕ → ℚ :=
  λ ⟨a, b, c, d⟩ => (a + b + c + d : ℚ) / 4

theorem average_of_four_given_conditions :
  ∀ (A B C D : ℕ), 
    (A + B) / 2 = 35 → 
    C = 130 → 
    D = 1 → 
    average_of_four_integers (A, B, C, D) = 50.25 := 
by
  intros A B C D hAB hC hD
  unfold average_of_four_integers
  sorry

end average_of_four_given_conditions_l276_276099


namespace original_price_per_kg_of_salt_l276_276584

variable {P X : ℝ}

theorem original_price_per_kg_of_salt (h1 : 400 / (0.8 * P) = X + 10)
    (h2 : 400 / P = X) : P = 10 :=
by
  sorry

end original_price_per_kg_of_salt_l276_276584


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276704

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276704


namespace seokjin_rank_l276_276370

-- Define the ranks and the people between them as given conditions in the problem
def jimin_rank : Nat := 4
def people_between : Nat := 19

-- The goal is to prove that Seokjin's rank is 24
theorem seokjin_rank : jimin_rank + people_between + 1 = 24 := 
by
  sorry

end seokjin_rank_l276_276370


namespace sum_possible_values_l276_276193

def abs_eq_2023 (a : ℤ) : Prop := abs a = 2023
def abs_eq_2022 (b : ℤ) : Prop := abs b = 2022
def greater_than (a b : ℤ) : Prop := a > b

theorem sum_possible_values (a b : ℤ) (h1 : abs_eq_2023 a) (h2 : abs_eq_2022 b) (h3 : greater_than a b) :
  a + b = 1 ∨ a + b = 4045 := 
sorry

end sum_possible_values_l276_276193


namespace abs_sum_of_factors_of_quadratic_l276_276050

variable (h b c d : ℤ)

theorem abs_sum_of_factors_of_quadratic :
  (∀ x : ℤ, 6 * x * x + x - 12 = (h * x + b) * (c * x + d)) →
  (|h| + |b| + |c| + |d| = 12) :=
by
  sorry

end abs_sum_of_factors_of_quadratic_l276_276050


namespace problem_31_36_l276_276515

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem problem_31_36 (p k : ℕ) (hp : is_prime (4 * k + 1)) :
  (∃ x y m : ℕ, x^2 + y^2 = m * p) ∧ (∀ m > 1, ∃ x y m1 : ℕ, x^2 + y^2 = m * p ∧ 0 < m1 ∧ m1 < m) :=
by sorry

end problem_31_36_l276_276515


namespace smallest_three_digit_number_with_property_l276_276328

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a < 1000 ∧ ∃ n : ℕ, 1000 * a + (a + 1) = n^2 ∧ a = 183 :=
sorry

end smallest_three_digit_number_with_property_l276_276328


namespace probability_third_winning_l276_276803

-- Definitions based on the conditions provided
def num_tickets : ℕ := 10
def num_winning_tickets : ℕ := 3
def num_non_winning_tickets : ℕ := num_tickets - num_winning_tickets

-- Define the probability function
def probability_of_third_draw_winning : ℚ :=
  (num_non_winning_tickets / num_tickets) * 
  ((num_non_winning_tickets - 1) / (num_tickets - 1)) * 
  (num_winning_tickets / (num_tickets - 2))

-- The theorem to prove
theorem probability_third_winning : probability_of_third_draw_winning = 7 / 40 :=
  by sorry

end probability_third_winning_l276_276803


namespace maximum_area_of_rectangle_with_fixed_perimeter_l276_276115

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l276_276115


namespace white_surface_fraction_l276_276580

-- Definition of the problem conditions
def larger_cube_surface_area : ℕ := 54
def white_cubes : ℕ := 6
def white_surface_area_minimized : ℕ := 5

-- Theorem statement proving the fraction of white surface area
theorem white_surface_fraction : (white_surface_area_minimized / larger_cube_surface_area : ℚ) = 5 / 54 := 
by
  sorry

end white_surface_fraction_l276_276580


namespace decreasing_power_function_l276_276054

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x ^ k

theorem decreasing_power_function (k : ℝ) : 
  (∀ x : ℝ, 0 < x → (f k x) ≤ 0) ↔ k < 0 ∧ k ≠ 0 := sorry

end decreasing_power_function_l276_276054


namespace four_digit_number_l276_276581

theorem four_digit_number (x : ℕ) (hx : 100 ≤ x ∧ x < 1000) (unit_digit : ℕ) (hu : unit_digit = 2) :
    (10 * x + unit_digit) - (2000 + x) = 108 → 10 * x + unit_digit = 2342 :=
by
  intros h
  sorry


end four_digit_number_l276_276581


namespace find_temp_M_l276_276256

section TemperatureProof

variables (M T W Th F : ℕ)

-- Conditions
def avg_temp_MTWT := (M + T + W + Th) / 4 = 48
def avg_temp_TWThF := (T + W + Th + F) / 4 = 40
def temp_F := F = 10

-- Proof
theorem find_temp_M (h1 : avg_temp_MTWT M T W Th)
                    (h2 : avg_temp_TWThF T W Th F)
                    (h3 : temp_F F)
                    : M = 42 :=
sorry

end TemperatureProof

end find_temp_M_l276_276256


namespace condition_necessary_but_not_sufficient_l276_276222

variable (a : ℝ)

theorem condition_necessary_but_not_sufficient (h : a^2 < 1) : (a < 1) ∧ (¬(a < 1 → a^2 < 1)) := sorry

end condition_necessary_but_not_sufficient_l276_276222


namespace range_of_a_l276_276643

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
by 
  sorry

end range_of_a_l276_276643


namespace diagonals_in_decagon_l276_276177

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end diagonals_in_decagon_l276_276177


namespace pages_can_be_copied_l276_276664

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l276_276664


namespace pages_can_be_copied_l276_276666

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l276_276666


namespace equation_completing_square_l276_276400

theorem equation_completing_square :
  ∃ (a b c : ℤ), 64 * x^2 + 80 * x - 81 = 0 → 
  (a > 0) ∧ (2 * a * b = 80) ∧ (a^2 = 64) ∧ (a + b + c = 119) :=
sorry

end equation_completing_square_l276_276400


namespace felix_trees_chopped_l276_276933

theorem felix_trees_chopped (trees_per_sharpen : ℕ) (cost_per_sharpen : ℕ) (total_spent : ℕ) (trees_chopped : ℕ) :
  trees_per_sharpen = 13 →
  cost_per_sharpen = 5 →
  total_spent = 35 →
  trees_chopped = (total_spent / cost_per_sharpen) * trees_per_sharpen →
  trees_chopped ≥ 91 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have : (35 / 5) * 13 = 91 := sorry
  rw this at h4
  exact le_of_eq h4

end felix_trees_chopped_l276_276933


namespace unknown_number_l276_276107

theorem unknown_number (n : ℕ) (h1 : Nat.lcm 24 n = 168) (h2 : Nat.gcd 24 n = 4) : n = 28 :=
by
  sorry

end unknown_number_l276_276107


namespace simplify_expression_l276_276247

variable (x y : ℕ)
variable (h_x : x = 5)
variable (h_y : y = 2)

theorem simplify_expression : (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end simplify_expression_l276_276247


namespace time_reading_per_week_l276_276563

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end time_reading_per_week_l276_276563


namespace number_of_cows_l276_276902

theorem number_of_cows (C H : ℕ) (hcnd : 4 * C + 2 * H = 2 * (C + H) + 18) :
  C = 9 :=
sorry

end number_of_cows_l276_276902


namespace max_quotient_l276_276822

theorem max_quotient (a b : ℝ) (h1 : 300 ≤ a) (h2 : a ≤ 500) (h3 : 900 ≤ b) (h4 : b ≤ 1800) :
  ∃ (q : ℝ), q = 5 / 9 ∧ (∀ (x y : ℝ), (300 ≤ x ∧ x ≤ 500) ∧ (900 ≤ y ∧ y ≤ 1800) → (x / y ≤ q)) :=
by
  use 5 / 9
  sorry

end max_quotient_l276_276822


namespace mistaken_multiplier_is_34_l276_276911

-- Define the main conditions
def correct_number : ℕ := 135
def correct_multiplier : ℕ := 43
def difference : ℕ := 1215

-- Define what we need to prove
theorem mistaken_multiplier_is_34 :
  (correct_number * correct_multiplier - correct_number * x = difference) →
  x = 34 :=
by
  sorry

end mistaken_multiplier_is_34_l276_276911


namespace addition_amount_first_trial_l276_276372

theorem addition_amount_first_trial :
  ∀ (a b : ℝ),
  20 ≤ a ∧ a ≤ 30 ∧ 20 ≤ b ∧ b ≤ 30 → (a = 20 + (30 - 20) * 0.618 ∨ b = 30 - (30 - 20) * 0.618) :=
by {
  sorry
}

end addition_amount_first_trial_l276_276372


namespace triangle_arithmetic_geometric_equilateral_l276_276360

theorem triangle_arithmetic_geometric_equilateral :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ (∃ d, β = α + d ∧ γ = α + 2 * d) ∧ (∃ r, β = α * r ∧ γ = α * r^2) →
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by
  sorry

end triangle_arithmetic_geometric_equilateral_l276_276360


namespace jelly_beans_remaining_l276_276274

theorem jelly_beans_remaining :
  let initial_jelly_beans := 8000
  let num_first_group := 6
  let num_last_group := 4
  let last_group_took := 400
  let first_group_took := 2 * last_group_took
  let last_group_total := last_group_took * num_last_group
  let first_group_total := first_group_took * num_first_group
  let remaining_jelly_beans := initial_jelly_beans - (first_group_total + last_group_total)
  remaining_jelly_beans = 1600 :=
by {
  -- Define the initial number of jelly beans
  let initial_jelly_beans := 8000
  -- Number of people in first and last groups
  let num_first_group := 6
  let num_last_group := 4
  -- Jelly beans taken by last group and first group per person
  let last_group_took := 400
  let first_group_took := 2 * last_group_took
  -- Calculate total jelly beans taken by last group and first group
  let last_group_total := last_group_took * num_last_group
  let first_group_total := first_group_took * num_first_group
  -- Jelly beans remaining
  let remaining_jelly_beans := initial_jelly_beans - (first_group_total + last_group_total)
  -- Proof of the theorem
  show remaining_jelly_beans = 1600, from sorry
}

end jelly_beans_remaining_l276_276274


namespace age_difference_l276_276752

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : C + 11 = A :=
by {
  sorry
}

end age_difference_l276_276752


namespace card_draw_prob_correct_l276_276124

-- Define the probability calculation for the card drawing scenario
def card_drawing_probability : ℚ :=
  9 / 385

theorem card_draw_prob_correct :
  (n : ℕ) (cards : Finset ℕ) (h : cards.cardinality = 12) (pairs : Finset (Finset ℕ)) :
  pairs = (Finset.range 6).bind (λ x, Finset.pair x) →
  ∀ d ≤ pairs.cardinality, (draw_probability_with_conditions cards pairs d).terminal = true →
  card_drawing_probability = 9 / 385 :=
begin
  sorry
end

end card_draw_prob_correct_l276_276124


namespace smallest_possible_value_of_c_l276_276597

theorem smallest_possible_value_of_c
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (H : ∀ x : ℝ, (a * Real.sin (b * x + c)) ≤ (a * Real.sin (b * 0 + c))) :
  c = Real.pi / 2 :=
by
  sorry

end smallest_possible_value_of_c_l276_276597


namespace expand_product_l276_276003

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 :=
by sorry

end expand_product_l276_276003


namespace vegetarian_gluten_free_fraction_l276_276851

theorem vegetarian_gluten_free_fraction :
  ∀ (total_dishes meatless_dishes gluten_free_meatless_dishes : ℕ),
  meatless_dishes = 4 →
  meatless_dishes = total_dishes / 5 →
  gluten_free_meatless_dishes = meatless_dishes - 3 →
  gluten_free_meatless_dishes / total_dishes = 1 / 20 :=
by sorry

end vegetarian_gluten_free_fraction_l276_276851


namespace rent_percentage_increase_l276_276507

theorem rent_percentage_increase 
  (E : ℝ) 
  (h1 : ∀ (E : ℝ), rent_last_year = 0.25 * E)
  (h2 : ∀ (E : ℝ), earnings_this_year = 1.45 * E)
  (h3 : ∀ (E : ℝ), rent_this_year = 0.35 * earnings_this_year) :
  (rent_this_year / rent_last_year) * 100 = 203 := 
by 
  sorry

end rent_percentage_increase_l276_276507


namespace geometric_construction_l276_276951

open Set

structure Point := (x : ℝ) (y : ℝ)

structure Line := (a b c : ℝ)

def is_on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

def on_opposite_sides (A B : Point) (l : Line) : Prop :=
  (l.a * A.x + l.b * A.y + l.c) * (l.a * B.x + l.b * B.y + l.c) < 0

def tangent_point (A : Point) (l : Line) : Point := sorry -- Point of tangency K

def circle_center (A : Point) (radius : ℝ) (l : Line) : Point := sorry -- Circle center at A with given radius

def tangent_from_point (B : Point) (circle_center : Point) : Point := sorry -- Tangent from B to the circle with center A touches at N

def intersection_with_line (from : Point) (to : Point) (l : Line) : Point := sorry -- Intersection point of BN with line l

def angle (P Q R : Point) : ℝ := sorry -- Angle PQS in degrees or radians; to be defined

theorem geometric_construction (l : Line) (A B : Point)
(h1 : on_opposite_sides A B l) : 
∃ M : Point,
  let K := tangent_point A l in
  let N := tangent_from_point B (circle_center A (sqrt (l.a ^ 2 + l.b ^ 2)) l) in
  M = intersection_with_line B N l ∧
  (angle A M K) = (1 / 2) * (angle B M K):=
sorry


end geometric_construction_l276_276951


namespace keun_bae_jumps_fourth_day_l276_276066

def jumps (n : ℕ) : ℕ :=
  match n with
  | 0 => 15
  | n + 1 => 2 * jumps n

theorem keun_bae_jumps_fourth_day : jumps 3 = 120 :=
by
  sorry

end keun_bae_jumps_fourth_day_l276_276066


namespace sticks_difference_l276_276455

def sticks_picked_up : ℕ := 14
def sticks_left : ℕ := 4

theorem sticks_difference : (sticks_picked_up - sticks_left) = 10 := by
  sorry

end sticks_difference_l276_276455


namespace abs_neg_eight_l276_276872

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l276_276872


namespace solve_for_a_l276_276817

theorem solve_for_a (x y a : ℝ) (h1 : 2 * x + y = 2 * a + 1) 
                    (h2 : x + 2 * y = a - 1) 
                    (h3 : x - y = 4) : a = 2 :=
by
  sorry

end solve_for_a_l276_276817


namespace cherries_eaten_l276_276852

-- Define the number of cherries Oliver had initially
def initial_cherries : ℕ := 16

-- Define the number of cherries Oliver had left after eating some
def left_cherries : ℕ := 6

-- Prove that the difference between the initial and left cherries is 10
theorem cherries_eaten : initial_cherries - left_cherries = 10 := by
  sorry

end cherries_eaten_l276_276852


namespace rainy_days_last_week_l276_276082

-- All conditions in Lean definitions
def even_integer (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def cups_of_tea_n (n : ℤ) : ℤ := 3
def total_drinks (R NR : ℤ) (m : ℤ) : Prop := 2 * m * R + 3 * NR = 36
def more_tea_than_hot_chocolate (R NR : ℤ) (m : ℤ) : Prop := 3 * NR - 2 * m * R = 12
def odd_number_of_rainy_days (R : ℤ) : Prop := R % 2 = 1
def total_days_in_week (R NR : ℤ) : Prop := R + NR = 7

-- Main statement
theorem rainy_days_last_week : ∃ R m NR : ℤ, 
  odd_number_of_rainy_days R ∧ 
  total_days_in_week R NR ∧ 
  total_drinks R NR m ∧ 
  more_tea_than_hot_chocolate R NR m ∧
  R = 3 :=
by
  sorry

end rainy_days_last_week_l276_276082


namespace no_fractional_linear_function_l276_276690

noncomputable def fractional_linear_function (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem no_fractional_linear_function (a b c d : ℝ) :
  ∀ x : ℝ, c ≠ 0 → 
  (fractional_linear_function a b c d x + fractional_linear_function b (-d) c (-a) x ≠ -2) :=
by
  sorry

end no_fractional_linear_function_l276_276690


namespace find_width_of_brick_l276_276190

theorem find_width_of_brick (l h : ℝ) (SurfaceArea : ℝ) (w : ℝ) :
  l = 8 → h = 2 → SurfaceArea = 152 → 2*l*w + 2*l*h + 2*w*h = SurfaceArea → w = 6 :=
by
  intro l_value
  intro h_value
  intro SurfaceArea_value
  intro surface_area_equation
  sorry

end find_width_of_brick_l276_276190


namespace scientific_notation_142000_l276_276582

theorem scientific_notation_142000 : (142000 : ℝ) = 1.42 * 10^5 := sorry

end scientific_notation_142000_l276_276582


namespace relationship_of_magnitudes_l276_276616

noncomputable def is_ordered (x : ℝ) (A B C : ℝ) : Prop :=
  0 < x ∧ x < Real.pi / 4 ∧
  A = Real.cos (x ^ Real.sin (x ^ Real.sin x)) ∧
  B = Real.sin (x ^ Real.cos (x ^ Real.sin x)) ∧
  C = Real.cos (x ^ Real.sin (x * (x ^ Real.cos x))) ∧
  B < A ∧ A < C

theorem relationship_of_magnitudes (x A B C : ℝ) : 
  is_ordered x A B C := 
sorry

end relationship_of_magnitudes_l276_276616


namespace goldfish_count_15_weeks_l276_276227

def goldfish_count_after_weeks (initial : ℕ) (weeks : ℕ) : ℕ :=
  let deaths := λ n => 10 + 2 * (n - 1)
  let purchases := λ n => 5 + 2 * (n - 1)
  let rec update_goldfish (current : ℕ) (week : ℕ) :=
    if week = 0 then current
    else 
      let new_count := current - deaths week + purchases week
      update_goldfish new_count (week - 1)
  update_goldfish initial weeks

theorem goldfish_count_15_weeks : goldfish_count_after_weeks 35 15 = 15 :=
  by
  sorry

end goldfish_count_15_weeks_l276_276227


namespace average_minutes_per_player_l276_276855

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l276_276855


namespace darcy_folded_shorts_l276_276454

-- Define the conditions
def total_shirts : Nat := 20
def total_shorts : Nat := 8
def folded_shirts : Nat := 12
def remaining_pieces : Nat := 11

-- Expected result to prove
def folded_shorts : Nat := 5

-- The statement to prove
theorem darcy_folded_shorts : total_shorts - (remaining_pieces - (total_shirts - folded_shirts)) = folded_shorts :=
by
  sorry

end darcy_folded_shorts_l276_276454


namespace find_the_number_l276_276128

-- Statement
theorem find_the_number (x : ℤ) (h : 2 * x = 3 * x - 25) : x = 25 :=
  sorry

end find_the_number_l276_276128


namespace spider_moves_away_from_bee_l276_276428

noncomputable def bee : ℝ × ℝ := (14, 5)
noncomputable def spider_line (x : ℝ) : ℝ := -3 * x + 25
noncomputable def perpendicular_line (x : ℝ) : ℝ := (1 / 3) * x + 14 / 3

theorem spider_moves_away_from_bee : ∃ (c d : ℝ), 
  (d = spider_line c) ∧ (d = perpendicular_line c) ∧ c + d = 13.37 := 
sorry

end spider_moves_away_from_bee_l276_276428


namespace sunflower_mix_is_50_percent_l276_276147

-- Define the proportions and percentages given in the problem
def prop_A : ℝ := 0.60 -- 60% of the mix is Brand A
def prop_B : ℝ := 0.40 -- 40% of the mix is Brand B
def sf_A : ℝ := 0.60 -- Brand A is 60% sunflower
def sf_B : ℝ := 0.35 -- Brand B is 35% sunflower

-- Define the final percentage of sunflower in the mix
noncomputable def sunflower_mix_percentage : ℝ :=
  (sf_A * prop_A) + (sf_B * prop_B)

-- Statement to prove that the percentage of sunflower in the mix is 50%
theorem sunflower_mix_is_50_percent : sunflower_mix_percentage = 0.50 :=
by
  sorry

end sunflower_mix_is_50_percent_l276_276147


namespace max_height_of_ball_l276_276144

def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

theorem max_height_of_ball : ∃ t₀, h t₀ = 81.25 ∧ ∀ t, h t ≤ 81.25 :=
by
  sorry

end max_height_of_ball_l276_276144


namespace binary_multiplication_l276_276924

/-- 
Calculate the product of two binary numbers and validate the result.
Given:
  a = 1101 in base 2,
  b = 111 in base 2,
Prove:
  a * b = 1011110 in base 2. 
-/
theorem binary_multiplication : 
  let a := 0b1101
  let b := 0b111
  a * b = 0b1011110 :=
by
  sorry

end binary_multiplication_l276_276924


namespace simplify_expr1_simplify_expr2_l276_276248

variable {a b : ℝ} -- Assume a and b are arbitrary real numbers

-- Part 1: Prove that 2a - [-3b - 3(3a - b)] = 11a
theorem simplify_expr1 : (2 * a - (-3 * b - 3 * (3 * a - b))) = 11 * a :=
by
  sorry

-- Part 2: Prove that 12ab^2 - [7a^2b - (ab^2 - 3a^2b)] = 13ab^2 - 10a^2b
theorem simplify_expr2 : (12 * a * b^2 - (7 * a^2 * b - (a * b^2 - 3 * a^2 * b))) = (13 * a * b^2 - 10 * a^2 * b) :=
by
  sorry

end simplify_expr1_simplify_expr2_l276_276248


namespace seq_inequality_l276_276841

noncomputable def sequence_of_nonneg_reals (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, a (n + m) ≤ a n + a m

theorem seq_inequality
  (a : ℕ → ℝ)
  (h : sequence_of_nonneg_reals a)
  (h_nonneg : ∀ n, 0 ≤ a n) :
  ∀ n m : ℕ, m > 0 → n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := 
by
  sorry

end seq_inequality_l276_276841


namespace parabola_shift_right_by_3_l276_276110

theorem parabola_shift_right_by_3 :
  ∀ (x : ℝ), (∃ y₁ y₂ : ℝ, y₁ = 2 * x^2 ∧ y₂ = 2 * (x - 3)^2) →
  (∃ (h : ℝ), h = 3) :=
sorry

end parabola_shift_right_by_3_l276_276110


namespace winnie_balloons_rem_l276_276421

theorem winnie_balloons_rem (r w g c : ℕ) (h_r : r = 17) (h_w : w = 33) (h_g : g = 65) (h_c : c = 83) :
  (r + w + g + c) % 8 = 6 := 
by 
  sorry

end winnie_balloons_rem_l276_276421


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276702

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276702


namespace number_of_new_students_l276_276540

variable (O N : ℕ)
variable (H1 : 48 * O + 32 * N = 44 * 160)
variable (H2 : O + N = 160)

theorem number_of_new_students : N = 40 := sorry

end number_of_new_students_l276_276540


namespace trig_expression_eq_zero_l276_276620

theorem trig_expression_eq_zero (α : ℝ) (h1 : Real.sin α = -2 / Real.sqrt 5) (h2 : Real.cos α = 1 / Real.sqrt 5) :
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 0 := by
  sorry

end trig_expression_eq_zero_l276_276620


namespace minimum_p_for_required_profit_l276_276883

noncomputable def profit (x p : ℝ) : ℝ := p * x - (0.5 * x^2 - 2 * x - 10)
noncomputable def max_profit (p : ℝ) : ℝ := (p + 2)^2 / 2 + 10

theorem minimum_p_for_required_profit : ∀ (p : ℝ), 3 * max_profit p >= 126 → p >= 6 :=
by
  intro p
  unfold max_profit
  -- Given:  3 * ((p + 2)^2 / 2 + 10) >= 126
  sorry

end minimum_p_for_required_profit_l276_276883


namespace min_value_one_over_a_plus_two_over_b_l276_276958

/-- Given a > 0, b > 0, 2a + b = 1, prove that the minimum value of (1/a) + (2/b) is 8 --/
theorem min_value_one_over_a_plus_two_over_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a) + (2 / b) ≥ 8 :=
sorry

end min_value_one_over_a_plus_two_over_b_l276_276958


namespace smallest_number_of_cubes_filling_box_l276_276772
open Nat

theorem smallest_number_of_cubes_filling_box (L W D : ℕ) (hL : L = 27) (hW : W = 15) (hD : D = 6) :
  let gcd := 3
  let cubes_along_length := L / gcd
  let cubes_along_width := W / gcd
  let cubes_along_depth := D / gcd
  cubes_along_length * cubes_along_width * cubes_along_depth = 90 :=
by
  sorry

end smallest_number_of_cubes_filling_box_l276_276772


namespace negation_of_cube_of_every_odd_number_is_odd_l276_276547

theorem negation_of_cube_of_every_odd_number_is_odd:
  ¬ (∀ n : ℤ, (n % 2 = 1 → (n^3 % 2 = 1))) ↔ ∃ n : ℤ, (n % 2 = 1 ∧ ¬ (n^3 % 2 = 1)) := 
by
  sorry

end negation_of_cube_of_every_odd_number_is_odd_l276_276547


namespace bricks_needed_l276_276900

noncomputable def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ := length * width * height

theorem bricks_needed
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (hl : brick_length = 40)
  (hw : brick_width = 11.25)
  (hh : brick_height = 6)
  (wl : wall_length = 800)
  (wh : wall_height = 600)
  (wt : wall_thickness = 22.5) :
  (volume wall_length wall_height wall_thickness / volume brick_length brick_width brick_height) = 4000 := by
  sorry

end bricks_needed_l276_276900


namespace Sara_has_3194_quarters_in_the_end_l276_276527

theorem Sara_has_3194_quarters_in_the_end
  (initial_quarters : ℕ)
  (borrowed_quarters : ℕ)
  (initial_quarters_eq : initial_quarters = 4937)
  (borrowed_quarters_eq : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 := by
  sorry

end Sara_has_3194_quarters_in_the_end_l276_276527


namespace inequality_solution_l276_276408

theorem inequality_solution (x : ℝ) : (2 * x - 3 < x + 1) -> (x < 4) :=
by
  intro h
  sorry

end inequality_solution_l276_276408


namespace find_complex_number_purely_imaginary_l276_276935

theorem find_complex_number_purely_imaginary :
  ∃ z : ℂ, (∃ b : ℝ, b ≠ 0 ∧ z = 1 + b * I) ∧ (∀ a b : ℝ, z = a + b * I → a^2 - b^2 + 3 = 0) :=
by
  -- Proof will go here
  sorry

end find_complex_number_purely_imaginary_l276_276935


namespace max_element_sum_l276_276676

-- Definitions based on conditions
def S : Set ℚ :=
  {r | ∃ (p q : ℕ), r = p / q ∧ q ≤ 2009 ∧ p / q < 1257/2009}

-- Maximum element of S in reduced form
def max_element_S (r : ℚ) : Prop := r ∈ S ∧ ∀ s ∈ S, r ≥ s

-- Main statement to be proven
theorem max_element_sum : 
  ∃ p0 q0 : ℕ, max_element_S (p0 / q0) ∧ Nat.gcd p0 q0 = 1 ∧ p0 + q0 = 595 := 
sorry

end max_element_sum_l276_276676


namespace div_identity_l276_276824

theorem div_identity (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := by
  sorry

end div_identity_l276_276824


namespace ratio_of_cubes_l276_276431

/-- A cubical block of metal weighs 7 pounds. Another cube of the same metal, with sides of a certain ratio longer, weighs 56 pounds. Prove that the ratio of the side length of the second cube to the first cube is 2:1. --/
theorem ratio_of_cubes (s r : ℝ) (weight1 weight2 : ℝ)
  (h1 : weight1 = 7) (h2 : weight2 = 56)
  (h_vol1 : weight1 = s^3)
  (h_vol2 : weight2 = (r * s)^3) :
  r = 2 := 
sorry

end ratio_of_cubes_l276_276431


namespace circle_area_from_circumference_l276_276259

theorem circle_area_from_circumference (C : ℝ) (A : ℝ) (hC : C = 36) (hCircumference : ∀ r, C = 2 * Real.pi * r) (hAreaFormula : ∀ r, A = Real.pi * r^2) :
  A = 324 / Real.pi :=
by
  sorry

end circle_area_from_circumference_l276_276259


namespace divides_polynomial_l276_276383

theorem divides_polynomial (n : ℕ) (x : ℤ) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(n+2) + (x+1)^(2*n+1)) :=
sorry

end divides_polynomial_l276_276383


namespace product_of_numbers_in_given_ratio_l276_276566

theorem product_of_numbers_in_given_ratio :
  ∃ (x y : ℝ), (x - y) ≠ 0 ∧ (x + y) / (x - y) = 9 ∧ (x * y) / (x - y) = 40 ∧ (x * y) = 80 :=
by {
  sorry
}

end product_of_numbers_in_given_ratio_l276_276566


namespace Matias_sales_l276_276228

def books_sold (Tuesday Wednesday Thursday : Nat) : Prop :=
  Tuesday = 7 ∧ 
  Wednesday = 3 * Tuesday ∧ 
  Thursday = 3 * Wednesday ∧ 
  Tuesday + Wednesday + Thursday = 91

theorem Matias_sales
  (Tuesday Wednesday Thursday : Nat) :
  books_sold Tuesday Wednesday Thursday := by
  sorry

end Matias_sales_l276_276228


namespace range_of_a_l276_276055

theorem range_of_a (a : ℝ) (A : Set ℝ) (h : A = {x | a * x^2 - 3 * x + 1 = 0} ∧ ∃ (n : ℕ), 2 ^ n - 1 = 3) :
  a ∈ Set.Ioo (-(1:ℝ)/0) 0 ∪ Set.Ioo 0 (9 / 4) :=
sorry

end range_of_a_l276_276055


namespace surface_area_of_circumscribed_sphere_l276_276477

theorem surface_area_of_circumscribed_sphere (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : 
  ∃ S : ℝ, S = 29 * Real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_l276_276477


namespace cylinder_volume_eq_l276_276778

variable (α β l : ℝ)

theorem cylinder_volume_eq (hα_pos : 0 < α ∧ α < π/2) (hβ_pos : 0 < β ∧ β < π/2) (hl_pos : 0 < l) :
  let V := (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2)
  V = (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2) :=
by 
  sorry

end cylinder_volume_eq_l276_276778


namespace find_a_l276_276810

open Real

def point_in_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x + 4 * y + 4 = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 3 = 0

theorem find_a (a : ℝ) :
  point_in_circle 1 a →
  line_equation 1 a →
  a = -2 :=
by
  intro h1 h2
  sorry

end find_a_l276_276810


namespace equalize_expenses_l276_276800

def total_expenses := 130 + 160 + 150 + 180
def per_person_share := total_expenses / 4
def tom_owes := per_person_share - 130
def dorothy_owes := per_person_share - 160
def sammy_owes := per_person_share - 150
def alice_owes := per_person_share - 180
def t := tom_owes
def d := dorothy_owes

theorem equalize_expenses : t - dorothy_owes = 30 := by
  sorry

end equalize_expenses_l276_276800


namespace Gerald_toy_cars_l276_276341

theorem Gerald_toy_cars :
  let initial_toy_cars := 20
  let fraction_donated := 1 / 4
  let donated_toy_cars := initial_toy_cars * fraction_donated
  let remaining_toy_cars := initial_toy_cars - donated_toy_cars
  remaining_toy_cars = 15 := 
by
  sorry

end Gerald_toy_cars_l276_276341


namespace neighbor_to_johnson_yield_ratio_l276_276500

-- Definitions
def johnsons_yield (months : ℕ) : ℕ := 80 * (months / 2)
def neighbors_yield_per_hectare (x : ℕ) (months : ℕ) : ℕ := 80 * x * (months / 2)
def total_neighor_yield (x : ℕ) (months : ℕ) : ℕ := 2 * neighbors_yield_per_hectare x months

-- Theorem statement
theorem neighbor_to_johnson_yield_ratio
  (x : ℕ)
  (h1 : johnsons_yield 6 = 240)
  (h2 : total_neighor_yield x 6 = 480 * x)
  (h3 : johnsons_yield 6 + total_neighor_yield x 6 = 1200)
  : x = 2 := by
sorry

end neighbor_to_johnson_yield_ratio_l276_276500


namespace min_handshakes_30_people_3_each_l276_276760

theorem min_handshakes_30_people_3_each : 
  ∃ (H : ℕ), (∀ (n k : ℕ), n = 30 ∧ k = 3 → H = (n * k) / 2) := 
by {
  use 45,
  intros n k h,
  rw [← h.1, ← h.2],
  norm_num,
  sorry
}

end min_handshakes_30_people_3_each_l276_276760


namespace divisor_of_100_by_quotient_9_and_remainder_1_l276_276424

theorem divisor_of_100_by_quotient_9_and_remainder_1 :
  ∃ d : ℕ, 100 = d * 9 + 1 ∧ d = 11 :=
by
  sorry

end divisor_of_100_by_quotient_9_and_remainder_1_l276_276424


namespace distance_between_points_l276_276035

theorem distance_between_points (A B : ℝ) (hA : |A| = 2) (hB : |B| = 7) :
  |A - B| = 5 ∨ |A - B| = 9 := 
sorry

end distance_between_points_l276_276035


namespace sam_distance_when_meeting_l276_276017

theorem sam_distance_when_meeting :
  ∃ t : ℝ, (35 = 2 * t + 5 * t) ∧ (5 * t = 25) :=
by
  sorry

end sam_distance_when_meeting_l276_276017


namespace part1_part2_l276_276473

theorem part1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

theorem part2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) ≥ (2 / (1 + a)) + (2 / (1 + b)) + (2 / (1 + c)) :=
sorry

end part1_part2_l276_276473


namespace range_of_m_l276_276353

theorem range_of_m (m : ℝ) :
  (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ m > 0 ∧ (15 - m > 0) ∧ (15 - m > 2 * m))
  ∨ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)) →
  (¬ (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)))) →
  (0 < m ∧ m ≤ 2) ∨ (5 ≤ m ∧ m < 16/3) :=
by
  sorry

end range_of_m_l276_276353


namespace min_value_of_exp_l276_276957

noncomputable def minimum_value_of_expression (a b : ℝ) : ℝ :=
  (1 - a)^2 + (1 - 2 * b)^2 + (a - 2 * b)^2

theorem min_value_of_exp (a b : ℝ) (h : a^2 ≥ 8 * b) : minimum_value_of_expression a b = 9 / 8 :=
by
  sorry

end min_value_of_exp_l276_276957


namespace geometric_seq_sum_first_4_terms_l276_276797

theorem geometric_seq_sum_first_4_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * 2)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 4 = 15 :=
by
  -- The actual proof would go here.
  sorry

end geometric_seq_sum_first_4_terms_l276_276797


namespace sum_of_roots_l276_276018

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem sum_of_roots (m : ℝ) (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 < 2 * Real.pi)
  (h3 : 0 ≤ x2) (h4 : x2 < 2 * Real.pi) (h_distinct : x1 ≠ x2)
  (h_eq1 : f x1 = m) (h_eq2 : f x2 = m) : x1 + x2 = Real.pi / 2 ∨ x1 + x2 = 5 * Real.pi / 2 :=
by
  sorry

end sum_of_roots_l276_276018


namespace find_a_l276_276351

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a

theorem find_a :
  (∀ x : ℝ, 0 ≤ f x a) ∧ (∀ y : ℝ, ∃ x : ℝ, y = f x a) ↔ a = 1 := by
  sorry

end find_a_l276_276351


namespace point_on_coordinate_axes_l276_276990

theorem point_on_coordinate_axes (x y : ℝ) (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by sorry

end point_on_coordinate_axes_l276_276990


namespace remainder_of_division_l276_276281

theorem remainder_of_division :
  Nat.mod 4536 32 = 24 :=
sorry

end remainder_of_division_l276_276281


namespace triple_solution_exists_and_unique_l276_276939

theorem triple_solution_exists_and_unique:
  ∀ (x y z : ℝ), (1 + x^4 ≤ 2 * (y - z) ^ 2) ∧ (1 + y^4 ≤ 2 * (z - x) ^ 2) ∧ (1 + z^4 ≤ 2 * (x - y) ^ 2)
  → (x = 1 ∧ y = 0 ∧ z = -1) :=
by
  sorry

end triple_solution_exists_and_unique_l276_276939


namespace find_m_l276_276627

theorem find_m (y x m : ℝ) (h1 : 2 - 3 * (1 - y) = 2 * y) (h2 : y = x) (h3 : m * (x - 3) - 2 = -8) : m = 3 :=
sorry

end find_m_l276_276627


namespace p_twice_q_in_future_years_l276_276286

-- We define the ages of p and q
def p_current_age : ℕ := 33
def q_current_age : ℕ := 11

-- Third condition that is redundant given the values we already defined
def age_relation : Prop := (p_current_age = 3 * q_current_age)

-- Number of years in the future when p will be twice as old as q
def future_years_when_twice : ℕ := 11

-- Prove that in future_years_when_twice years, p will be twice as old as q
theorem p_twice_q_in_future_years :
  ∀ t : ℕ, t = future_years_when_twice → (p_current_age + t = 2 * (q_current_age + t)) := by
  sorry

end p_twice_q_in_future_years_l276_276286


namespace sum_of_positive_x_and_y_is_ten_l276_276509

theorem sum_of_positive_x_and_y_is_ten (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^3 + y^3 + (x + y)^3 + 30 * x * y = 2000) : 
  x + y = 10 :=
sorry

end sum_of_positive_x_and_y_is_ten_l276_276509


namespace initial_percentage_liquid_X_l276_276587

theorem initial_percentage_liquid_X (P : ℝ) :
  let original_solution_kg := 8
  let evaporated_water_kg := 2
  let added_solution_kg := 2
  let remaining_solution_kg := original_solution_kg - evaporated_water_kg
  let new_solution_kg := remaining_solution_kg + added_solution_kg
  let new_solution_percentage := 0.25
  let initial_liquid_X_kg := (P / 100) * original_solution_kg
  let final_liquid_X_kg := initial_liquid_X_kg + (P / 100) * added_solution_kg
  let final_liquid_X_kg' := new_solution_percentage * new_solution_kg
  (final_liquid_X_kg = final_liquid_X_kg') → 
  P = 20 :=
by
  intros
  let original_solution_kg_p0 := 8
  let evaporated_water_kg_p1 := 2
  let added_solution_kg_p2 := 2
  let remaining_solution_kg_p3 := (original_solution_kg_p0 - evaporated_water_kg_p1)
  let new_solution_kg_p4 := (remaining_solution_kg_p3 + added_solution_kg_p2)
  let new_solution_percentage : ℝ := 0.25
  let initial_liquid_X_kg_p6 := ((P / 100) * original_solution_kg_p0)
  let final_liquid_X_kg_p7 := initial_liquid_X_kg_p6 + ((P / 100) * added_solution_kg_p2)
  let final_liquid_X_kg_p8 := (new_solution_percentage * new_solution_kg_p4)
  exact sorry

end initial_percentage_liquid_X_l276_276587


namespace blue_pens_count_l276_276413

-- Definitions based on the conditions
def total_pens (B R : ℕ) : Prop := B + R = 82
def more_blue_pens (B R : ℕ) : Prop := B = R + 6

-- The theorem to prove
theorem blue_pens_count (B R : ℕ) (h1 : total_pens B R) (h2 : more_blue_pens B R) : B = 44 :=
by {
  -- This is where the proof steps would normally go.
  sorry
}

end blue_pens_count_l276_276413


namespace roots_of_star_equation_l276_276319

def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

theorem roots_of_star_equation :
  ∀ x : ℝ, (star 1 x = 0) → (∃ a b : ℝ, a ≠ b ∧ x = a ∨ x = b) := 
by
  sorry

end roots_of_star_equation_l276_276319


namespace find_f_7_l276_276381

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  dsimp [f] at *
  sorry

end find_f_7_l276_276381


namespace polynomial_divisibility_l276_276224

theorem polynomial_divisibility (
  p q r s : ℝ
) :
  (x^5 + 5 * x^4 + 10 * p * x^3 + 10 * q * x^2 + 5 * r * x + s) % (x^4 + 4 * x^3 + 6 * x^2 + 4 * x + 1) = 0 ->
  (p + q + r) * s = -2 :=
by {
  sorry
}

end polynomial_divisibility_l276_276224


namespace students_like_both_l276_276648

theorem students_like_both (total_students French_fries_likers burger_likers neither_likers : ℕ)
(H1 : total_students = 25)
(H2 : French_fries_likers = 15)
(H3 : burger_likers = 10)
(H4 : neither_likers = 6)
: (French_fries_likers + burger_likers + neither_likers - total_students) = 12 :=
by sorry

end students_like_both_l276_276648


namespace arithmetic_sequence_diff_l276_276026

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (a : ℕ → ℤ)
variable (h1 : is_arithmetic_sequence a 2)

-- Prove that a_5 - a_2 = 6
theorem arithmetic_sequence_diff : a 5 - a 2 = 6 :=
by sorry

end arithmetic_sequence_diff_l276_276026


namespace average_weight_proof_l276_276100

variables (W_A W_B W_C W_D W_E : ℝ)

noncomputable def final_average_weight (W_A W_B W_C W_D W_E : ℝ) : ℝ := (W_B + W_C + W_D + W_E) / 4

theorem average_weight_proof
  (h1 : (W_A + W_B + W_C) / 3 = 84)
  (h2 : W_A = 77)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (h4 : W_E = W_D + 5) :
  final_average_weight W_A W_B W_C W_D W_E = 97.25 :=
by
  sorry

end average_weight_proof_l276_276100


namespace divide_group_among_boats_l276_276272
noncomputable def number_of_ways_divide_group 
  (boatA_capacity : ℕ) 
  (boatB_capacity : ℕ) 
  (boatC_capacity : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : ℕ := 
    sorry

theorem divide_group_among_boats 
  (boatA_capacity : ℕ := 3) 
  (boatB_capacity : ℕ := 2) 
  (boatC_capacity : ℕ := 1) 
  (num_adults : ℕ := 2) 
  (num_children : ℕ := 2) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : 
  number_of_ways_divide_group boatA_capacity boatB_capacity boatC_capacity num_adults num_children constraint = 8 := 
sorry

end divide_group_among_boats_l276_276272


namespace f_divisible_by_k2_k1_l276_276384

noncomputable def f (n : ℕ) (x : ℤ) : ℤ :=
  x^(n + 2) + (x + 1)^(2 * n + 1)

theorem f_divisible_by_k2_k1 (n : ℕ) (k : ℤ) (hn : n > 0) : 
  ((k^2 + k + 1) ∣ f n k) :=
sorry

end f_divisible_by_k2_k1_l276_276384


namespace setA_times_setB_equals_desired_l276_276987

def setA : Set ℝ := { x | abs (x - 1/2) < 1 }
def setB : Set ℝ := { x | 1/x ≥ 1 }
def setAB : Set ℝ := { x | (x ∈ setA ∪ setB) ∧ (x ∉ setA ∩ setB) }

theorem setA_times_setB_equals_desired :
  setAB = { x | (-1/2 < x ∧ x ≤ 0) ∨ (1 < x ∧ x < 3/2) } :=
by
  sorry

end setA_times_setB_equals_desired_l276_276987


namespace sum_of_bases_l276_276998

theorem sum_of_bases (R_1 R_2 : ℕ) 
  (hF1 : (4 * R_1 + 8) / (R_1 ^ 2 - 1) = (3 * R_2 + 6) / (R_2 ^ 2 - 1))
  (hF2 : (8 * R_1 + 4) / (R_1 ^ 2 - 1) = (6 * R_2 + 3) / (R_2 ^ 2 - 1)) : 
  R_1 + R_2 = 21 :=
sorry

end sum_of_bases_l276_276998


namespace x_coordinate_incenter_eq_l276_276779

theorem x_coordinate_incenter_eq {x y : ℝ} :
  (y = 0 → x + y = 3 → x = 0) → 
  (y = x → y = -x + 3 → x = 3 / 2) :=
by
  sorry

end x_coordinate_incenter_eq_l276_276779


namespace calculate_E_l276_276378

variables {α : Type*} [Field α] {a b c d : α → α → α}

noncomputable def E (a b c : α → α) : α :=
  Matrix.det ![a, b, c]

noncomputable def E' (a b c d : α → α) : α :=
  Matrix.det ![a × b, b × c, c × d]

theorem calculate_E' (a b c d : α → α) :
  let E := Matrix.det ![a, b, c] in
  E' a b c d = E^2 * ((b × c) • d) :=
sorry

end calculate_E_l276_276378


namespace number_of_divisors_of_36_l276_276965

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l276_276965


namespace num_pos_divisors_36_l276_276982

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l276_276982


namespace inequality_proof_l276_276386

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z ≥ 1/x + 1/y + 1/z) : 
  x/y + y/z + z/x ≥ 1/(x * y) + 1/(y * z) + 1/(z * x) :=
by
  sorry

end inequality_proof_l276_276386


namespace oak_trees_cut_down_l276_276414

   def number_of_cuts (initial: ℕ) (remaining: ℕ) : ℕ :=
     initial - remaining

   theorem oak_trees_cut_down : number_of_cuts 9 7 = 2 :=
   by
     -- Based on the conditions, we start with 9 and after workers finished, there are 7 oak trees.
     -- We calculate the number of trees cut down:
     -- 9 - 7 = 2
     sorry
   
end oak_trees_cut_down_l276_276414


namespace find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l276_276387

theorem find_zeros_of_quadratic {a b : ℝ} (h_a : a = 1) (h_b : b = -2) :
  ∀ x, (a * x^2 + b * x + b - 1 = 0) ↔ (x = 3 ∨ x = -1) := sorry

theorem range_of_a_for_two_distinct_zeros :
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + b - 1 = 0 ∧ a * x2^2 + b * x2 + b - 1 = 0) ↔ (0 < a ∧ a < 1) := sorry

end find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l276_276387


namespace calculate_height_l276_276101

def base_length : ℝ := 2 -- in cm
def base_width : ℝ := 5 -- in cm
def volume : ℝ := 30 -- in cm^3

theorem calculate_height: base_length * base_width * 3 = volume :=
by
  -- base_length * base_width = 10
  -- 10 * 3 = 30
  sorry

end calculate_height_l276_276101


namespace value_at_minus_two_l276_276364

def f (x : ℝ) : ℝ := x^2 + 3 * x - 5

theorem value_at_minus_two : f (-2) = -7 := by
  sorry

end value_at_minus_two_l276_276364


namespace complex_division_l276_276808

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (7 - i) / (3 + i) = 2 - i := by
  sorry

end complex_division_l276_276808


namespace find_value_of_expression_l276_276163

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry

def condition1 : Prop := x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 + 11 * x6 = 2
def condition2 : Prop := 3 * x1 + 5 * x2 + 7 * x3 + 9 * x4 + 11 * x5 + 13 * x6 = 15
def condition3 : Prop := 5 * x1 + 7 * x2 + 9 * x3 + 11 * x4 + 13 * x5 + 15 * x6 = 52

theorem find_value_of_expression : condition1 → condition2 → condition3 → (7 * x1 + 9 * x2 + 11 * x3 + 13 * x4 + 15 * x5 + 17 * x6 = 65) :=
by
  intros h1 h2 h3
  sorry

end find_value_of_expression_l276_276163


namespace area_of_triangle_DEF_eq_480_l276_276995

theorem area_of_triangle_DEF_eq_480 (DE EF DF : ℝ) (h1 : DE = 20) (h2 : EF = 48) (h3 : DF = 52) :
  let s := (DE + EF + DF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF))
  area = 480 :=
by
  sorry

end area_of_triangle_DEF_eq_480_l276_276995


namespace nicholas_bottle_caps_l276_276232

theorem nicholas_bottle_caps (initial : ℕ) (additional : ℕ) (final : ℕ) (h1 : initial = 8) (h2 : additional = 85) :
  final = 93 :=
by
  sorry

end nicholas_bottle_caps_l276_276232


namespace average_player_time_l276_276859

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l276_276859


namespace non_negative_integer_solutions_of_inequality_system_l276_276252

theorem non_negative_integer_solutions_of_inequality_system :
  (∀ x : ℚ, 3 * (x - 1) < 5 * x + 1 → (x - 1) / 2 ≥ 2 * x - 4 → (x = 0 ∨ x = 1 ∨ x = 2)) :=
by
  sorry

end non_negative_integer_solutions_of_inequality_system_l276_276252


namespace paper_needed_l276_276091

theorem paper_needed : 26 + 26 + 10 = 62 := by
  sorry

end paper_needed_l276_276091


namespace cemc_basketball_team_l276_276402

theorem cemc_basketball_team (t g : ℕ) (h_t : t = 6)
  (h1 : 40 * t + 20 * g = 28 * (g + 4)) :
  g = 16 := by
  -- Start your proof here

  sorry

end cemc_basketball_team_l276_276402


namespace ratio_A_BC_1_to_4_l276_276296

/-
We will define the conditions and prove the ratio.
-/

def A := 20
def total := 100

-- defining the conditions
variables (B C : ℝ)
def condition1 := A + B + C = total
def condition2 := B = 3 / 5 * (A + C)

-- the theorem to prove
theorem ratio_A_BC_1_to_4 (h1 : condition1 B C) (h2 : condition2 B C) : A / (B + C) = 1 / 4 :=
by
  sorry

end ratio_A_BC_1_to_4_l276_276296


namespace find_f_neg_one_l276_276512

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : ℝ) (h_m : f 0 m = 0) : f (-1) (-1) = -3 :=
by
  sorry

end find_f_neg_one_l276_276512


namespace temperature_representation_l276_276316

-- Defining the temperature representation problem
def posTemp := 10 -- $10^\circ \mathrm{C}$ above zero
def negTemp := -10 -- $10^\circ \mathrm{C}$ below zero
def aboveZero (temp : Int) : Prop := temp > 0
def belowZero (temp : Int) : Prop := temp < 0

-- The proof statement to be proved using the given conditions
theorem temperature_representation : 
  (aboveZero posTemp → posTemp = 10) ∧ (belowZero negTemp → negTemp = -10) := 
  by
    sorry -- Proof would go here

end temperature_representation_l276_276316


namespace largest_circle_radius_l276_276586

theorem largest_circle_radius 
  (h H : ℝ) (h_pos : h > 0) (H_pos : H > 0) :
  ∃ R, R = (h * H) / (h + H) :=
sorry

end largest_circle_radius_l276_276586


namespace egyptian_fraction_decomposition_l276_276650

theorem egyptian_fraction_decomposition (n : ℕ) (hn : 0 < n) : 
  (2 : ℚ) / (2 * n + 1) = (1 : ℚ) / (n + 1) + (1 : ℚ) / ((n + 1) * (2 * n + 1)) := 
by {
  sorry
}

end egyptian_fraction_decomposition_l276_276650


namespace angle_A_in_triangle_l276_276491

noncomputable def is_angle_A (a b : ℝ) (B A: ℝ) : Prop :=
  a = 2 * Real.sqrt 3 ∧ b = 2 * Real.sqrt 2 ∧ B = Real.pi / 4 ∧
  (A = Real.pi / 3 ∨ A = 2 * Real.pi / 3)

theorem angle_A_in_triangle (a b A B : ℝ) (h : is_angle_A a b B A) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end angle_A_in_triangle_l276_276491


namespace x_eq_1_iff_quadratic_eq_zero_l276_276102

theorem x_eq_1_iff_quadratic_eq_zero :
  ∀ x : ℝ, (x = 1) ↔ (x^2 - 2 * x + 1 = 0) := by
  sorry

end x_eq_1_iff_quadratic_eq_zero_l276_276102


namespace line_passing_through_M_l276_276776

-- Define the point M
def M : ℝ × ℝ := (-3, 4)

-- Define the predicate for a line equation having equal intercepts and passing through point M
def line_eq (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ c : ℝ, ((a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) 

theorem line_passing_through_M (x y : ℝ) (a b : ℝ) (h₀ : (-3, 4) = M) (h₁ : ∃ c : ℝ, (a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) :
  (4 * x + 3 * y = 0) ∨ (x + y = 1) :=
by
  -- We add 'sorry' to skip the proof
  sorry

end line_passing_through_M_l276_276776


namespace probability_of_selecting_male_volunteer_l276_276129

variables {UnitA UnitB : Type}
variables (maleA femaleA maleB femaleB: ℕ)

-- Define the conditions given in the problem
def conditions (h1 : maleA = 5) (h2 : femaleA = 7) 
  (h3 : maleB = 4) (h4 : femaleB = 2) 
  (h5 : (1 : ℝ) / 2 = (1 : ℝ) / 2) : Prop := 
maleA + femaleA + maleB + femaleB ≤ 24

-- The theorem stating the probability of selecting a male volunteer is 13/24
theorem probability_of_selecting_male_volunteer : 
  ∀ (h1 : maleA = 5) (h2 : femaleA = 7) 
   (h3 : maleB = 4) (h4 : femaleB = 2),
  (1 : ℝ) / 2 * (maleA / (maleA + femaleA) : ℝ) 
  + (1 : ℝ) / 2 * (maleB / (maleB + femaleB) : ℝ) = 13 / 24 := 
begin
  intros,
  rw [h1, h2, h3, h4],
  norm_num,
end

end probability_of_selecting_male_volunteer_l276_276129


namespace gcd_7392_15015_l276_276461

-- Define the two numbers
def num1 : ℕ := 7392
def num2 : ℕ := 15015

-- State the theorem and use sorry to omit the proof
theorem gcd_7392_15015 : Nat.gcd num1 num2 = 1 := 
  by sorry

end gcd_7392_15015_l276_276461


namespace largest_divisible_by_6_ending_in_4_l276_276717

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l276_276717


namespace michelle_silver_beads_l276_276074

theorem michelle_silver_beads :
  ∀ (total_beads blue_beads red_beads white_beads silver_beads : ℕ),
    total_beads = 40 →
    blue_beads = 5 →
    red_beads = 2 * blue_beads →
    white_beads = blue_beads + red_beads →
    silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
    silver_beads = 10 :=
by {
  intros total_beads blue_beads red_beads white_beads silver_beads,
  assume h1 h2 h3 h4 h5,
  sorry
}

end michelle_silver_beads_l276_276074


namespace rectangle_length_l276_276575

theorem rectangle_length {width length : ℝ} (h1 : (3 : ℝ) * 3 = 9) (h2 : width = 3) (h3 : width * length = 9) : 
  length = 3 :=
by
  sorry

end rectangle_length_l276_276575


namespace bowling_ball_weight_l276_276080

-- Definitions for the conditions
def kayak_weight : ℕ := 36
def total_weight_of_two_kayaks := 2 * kayak_weight
def total_weight_of_nine_bowling_balls (ball_weight : ℕ) := 9 * ball_weight  

theorem bowling_ball_weight (w : ℕ) (h1 : total_weight_of_two_kayaks = total_weight_of_nine_bowling_balls w) : w = 8 :=
by
  -- Proof goes here
  sorry

end bowling_ball_weight_l276_276080


namespace soccer_league_fraction_female_proof_l276_276235

variable (m f : ℝ)

def soccer_league_fraction_female : Prop :=
  let males_last_year := m
  let females_last_year := f
  let males_this_year := 1.05 * m
  let females_this_year := 1.2 * f
  let total_this_year := 1.1 * (m + f)
  (1.05 * m + 1.2 * f = 1.1 * (m + f)) → ((0.6 * m) / (1.65 * m) = 4 / 11)

theorem soccer_league_fraction_female_proof (m f : ℝ) : soccer_league_fraction_female m f :=
by {
  sorry
}

end soccer_league_fraction_female_proof_l276_276235


namespace group_photo_arrangements_l276_276138

theorem group_photo_arrangements :
  ∃ (arrangements : ℕ), arrangements = 36 ∧
    ∀ (M G H P1 P2 : ℕ),
    (M = G + 1 ∨ M + 1 = G) ∧ (M ≠ H - 1 ∧ M ≠ H + 1) →
    arrangements = 36 :=
by {
  sorry
}

end group_photo_arrangements_l276_276138


namespace geometric_sequence_a3_l276_276062

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = a 1 * q ^ 3) (h2 : a 2 = a 1 * q) (h3 : a 5 = a 1 * q ^ 4) 
    (h4 : a 4 - a 2 = 6) (h5 : a 5 - a 1 = 15) : a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end geometric_sequence_a3_l276_276062


namespace solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l276_276352

variable (a : ℝ) (x : ℝ)

def inequality (a x : ℝ) : Prop :=
  a * x^2 - (a + 2) * x + 2 < 0

theorem solve_inequality_when_a_lt_2 (h : a < 2) :
  (a = 0 → ∀ x, x > 1 → inequality a x) ∧
  (a < 0 → ∀ x, x < 2 / a ∨ x > 1 → inequality a x) ∧
  (0 < a ∧ a < 2 → ∀ x, 1 < x ∧ x < 2 / a → inequality a x) := 
sorry

theorem find_a_range_when_x_in_2_3 :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → inequality a x) → a < 2 / 3 :=
sorry

end solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l276_276352


namespace gift_arrangement_l276_276216

theorem gift_arrangement (n k : ℕ) (h_n : n = 5) (h_k : k = 4) : 
  (n * Nat.factorial k) = 120 :=
by
  sorry

end gift_arrangement_l276_276216


namespace fifth_term_in_geometric_sequence_l276_276786

variable (y : ℝ)

def geometric_sequence : ℕ → ℝ
| 0       => 3
| (n + 1) => geometric_sequence n * (3 * y)

theorem fifth_term_in_geometric_sequence (y : ℝ) : 
  geometric_sequence y 4 = 243 * y^4 :=
sorry

end fifth_term_in_geometric_sequence_l276_276786


namespace math_problem_l276_276871

variable (a b c : ℝ)

theorem math_problem (h1 : -10 ≤ a ∧ a < 0) (h2 : 0 < a ∧ a < b ∧ b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end math_problem_l276_276871


namespace num_positive_divisors_36_l276_276985

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l276_276985


namespace cos_double_angle_l276_276357

open Real

-- Define the given conditions
variables {θ : ℝ}
axiom θ_in_interval : 0 < θ ∧ θ < π / 2
axiom sin_minus_cos : sin θ - cos θ = sqrt 2 / 2

-- Create a theorem that reflects the proof problem
theorem cos_double_angle : cos (2 * θ) = - sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_l276_276357


namespace hyperbola_real_axis_length_correct_l276_276541

noncomputable def hyperbola_real_axis_length : ℝ :=
  let λ := 1
  let real_axis_length := 2 * λ
  real_axis_length

theorem hyperbola_real_axis_length_correct :
  ∀ (C : Hyperbola),
    C.center = (0, 0) →
    C.foci_on_x_axis →
    C.intersects_parabola_directrix (y^2 = 8 * x) (A, B) →
    dist A B = 2 * sqrt 3 →
    hyperbola_real_axis_length = 2 :=
by
  intros
  sorry

end hyperbola_real_axis_length_correct_l276_276541


namespace sin_cos_15_sin_cos_18_l276_276426

theorem sin_cos_15 (h45sin : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h45cos : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h30sin : Real.sin (30 * Real.pi / 180) = 1 / 2)
                  (h30cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2) :
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 ∧
  Real.cos (15 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

theorem sin_cos_18 (h18sin : Real.sin (18 * Real.pi / 180) = (-1 + Real.sqrt 5) / 4)
                   (h36cos : Real.cos (36 * Real.pi / 180) = (Real.sqrt 5 + 1) / 4) :
  Real.cos (18 * Real.pi / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4 := by
  sorry

end sin_cos_15_sin_cos_18_l276_276426


namespace color_of_face_opposite_silver_is_yellow_l276_276183

def Face : Type := String

def Color : Type := String

variable (B Y O Bl S V : Color)

-- Conditions based on views
variable (cube : Face → Color)
variable (top front_right_1 right_1 front_right_2 front_right_3 : Face)
variable (back : Face)

axiom view1 : cube top = B ∧ cube front_right_1 = Y ∧ cube right_1 = O
axiom view2 : cube top = B ∧ cube front_right_2 = Bl ∧ cube right_1 = O
axiom view3 : cube top = B ∧ cube front_right_3 = V ∧ cube right_1 = O

-- Additional axiom based on the fact that S is not visible and deduced to be on the back face
axiom silver_back : cube back = S

-- The problem: Prove that the color of the face opposite the silver face is yellow.
theorem color_of_face_opposite_silver_is_yellow :
  (∃ front : Face, cube front = Y) :=
by
  sorry

end color_of_face_opposite_silver_is_yellow_l276_276183


namespace handshakes_minimum_l276_276767

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l276_276767


namespace razorback_tshirts_sold_l276_276744

variable (T : ℕ) -- Number of t-shirts sold
variable (price_per_tshirt : ℕ := 62) -- Price of each t-shirt
variable (total_revenue : ℕ := 11346) -- Total revenue from t-shirts

theorem razorback_tshirts_sold :
  (price_per_tshirt * T = total_revenue) → T = 183 :=
by
  sorry

end razorback_tshirts_sold_l276_276744


namespace probability_red_bean_l276_276656

section ProbabilityRedBean

-- Initially, there are 5 red beans and 9 black beans in a bag.
def initial_red_beans : ℕ := 5
def initial_black_beans : ℕ := 9
def initial_total_beans : ℕ := initial_red_beans + initial_black_beans

-- Then, 3 red beans and 3 black beans are added to the bag.
def added_red_beans : ℕ := 3
def added_black_beans : ℕ := 3
def final_red_beans : ℕ := initial_red_beans + added_red_beans
def final_black_beans : ℕ := initial_black_beans + added_black_beans
def final_total_beans : ℕ := final_red_beans + final_black_beans

-- The probability of drawing a red bean should be 2/5
theorem probability_red_bean :
  (final_red_beans : ℚ) / final_total_beans = 2 / 5 := by
  sorry

end ProbabilityRedBean

end probability_red_bean_l276_276656


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l276_276710

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l276_276710


namespace sqrt_expression_eq_neg_one_l276_276447

theorem sqrt_expression_eq_neg_one : 
  Real.sqrt ((-2)^2) + (Real.sqrt 3)^2 - (Real.sqrt 12 * Real.sqrt 3) = -1 :=
sorry

end sqrt_expression_eq_neg_one_l276_276447


namespace gcd_8p_18q_l276_276989

theorem gcd_8p_18q (p q : ℕ) (hp : p > 0) (hq : q > 0) (hg : Nat.gcd p q = 9) : Nat.gcd (8 * p) (18 * q) = 18 := 
sorry

end gcd_8p_18q_l276_276989


namespace geometric_sequence_property_l276_276498

theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a 1 / a 0) (h₁ : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end geometric_sequence_property_l276_276498


namespace g_of_50_eq_zero_l276_276263

theorem g_of_50_eq_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - 3 * y * g x = g (x / y)) : g 50 = 0 :=
sorry

end g_of_50_eq_zero_l276_276263


namespace min_reciprocal_sum_l276_276379

theorem min_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a) + (1 / b) ≥ 2 := by
  sorry

end min_reciprocal_sum_l276_276379


namespace ratio_adults_children_is_one_l276_276745

theorem ratio_adults_children_is_one (a c : ℕ) (ha : a ≥ 1) (hc : c ≥ 1) (h : 30 * a + 15 * c = 2475) : a / c = 1 :=
by
  sorry

end ratio_adults_children_is_one_l276_276745


namespace inequality_solution_l276_276192

section
variables (a x : ℝ)

theorem inequality_solution (h : a < 0) :
  (ax^2 + (1 - a) * x - 1 > 0 ↔
     (-1 < a ∧ a < 0 ∧ 1 < x ∧ x < -1/a) ∨
     (a = -1 ∧ false) ∨
     (a < -1 ∧ -1/a < x ∧ x < 1)) :=
by sorry

end inequality_solution_l276_276192


namespace sum_of_first_70_odd_integers_l276_276751

theorem sum_of_first_70_odd_integers : 
  let sum_even := 70 * (70 + 1)
  let sum_odd := 70 ^ 2
  let diff := sum_even - sum_odd
  diff = 70 → sum_odd = 4900 :=
by
  intros
  sorry

end sum_of_first_70_odd_integers_l276_276751


namespace kim_total_ounces_l276_276675

def quarts_to_ounces (q : ℚ) : ℚ := q * 32

def bottle_quarts : ℚ := 1.5
def can_ounces : ℚ := 12
def bottle_ounces : ℚ := quarts_to_ounces bottle_quarts

def total_ounces : ℚ := bottle_ounces + can_ounces

theorem kim_total_ounces : total_ounces = 60 :=
by
  -- Proof will go here
  sorry

end kim_total_ounces_l276_276675


namespace ryan_correct_percentage_l276_276866

theorem ryan_correct_percentage :
  let problems1 := 25
  let correct1 := 0.8 * problems1
  let problems2 := 40
  let correct2 := 0.9 * problems2
  let problems3 := 10
  let correct3 := 0.7 * problems3
  let total_problems := problems1 + problems2 + problems3
  let total_correct := correct1 + correct2 + correct3
  (total_correct / total_problems) = 0.84 :=
by 
  sorry

end ryan_correct_percentage_l276_276866


namespace sequence_finite_l276_276470

def sequence_terminates (a_0 : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (a 0 = a_0) ∧ 
                  (∀ n, ((a n > 5) ∧ (a n % 10 ≤ 5) → a (n + 1) = a n / 10)) ∧
                  (∀ n, ((a n > 5) ∧ (a n % 10 > 5) → a (n + 1) = 9 * a n)) → 
                  ∃ n, a n ≤ 5 

theorem sequence_finite (a_0 : ℕ) : sequence_terminates a_0 :=
sorry

end sequence_finite_l276_276470


namespace find_x_range_l276_276356

def tight_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, 0 < n → 1/2 ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

theorem find_x_range
  (a : ℕ → ℝ)
  (h_tight : tight_sequence a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3 / 2)
  (h3 : ∃ x, a 3 = x)
  (h4 : a 4 = 4) :
  ∃ x, (2 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) :=
sorry

end find_x_range_l276_276356


namespace slope_of_parallel_line_l276_276571

theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) : 
  (5 * x - 3 * y = 12) → m = 5 / 3 → (∃ b : ℝ, y = (5 / 3) * x + b) :=
by
  intro h_eqn h_slope
  use -4 / 3
  sorry

end slope_of_parallel_line_l276_276571


namespace evaluate_expression_l276_276001

variable (b : ℝ)

theorem evaluate_expression : ( ( (b^(16/8))^(1/4) )^3 * ( (b^(16/4))^(1/8) )^3 ) = b^3 := by
  sorry

end evaluate_expression_l276_276001


namespace solve_for_x_l276_276399

theorem solve_for_x (x : ℚ) (h₁ : (7 * x + 2) / (x - 4) = -6 / (x - 4)) (h₂ : x ≠ 4) :
  x = -8 / 7 := 
  sorry

end solve_for_x_l276_276399


namespace problem_solution_l276_276615

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem problem_solution (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end problem_solution_l276_276615


namespace quadratic_inequality_solution_l276_276791

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - x - 12 < 0) ↔ (-3 < x ∧ x < 4) :=
by sorry

end quadratic_inequality_solution_l276_276791


namespace minimum_banks_needed_l276_276783

-- Condition definitions
def total_amount : ℕ := 10000000
def max_insurance_payout_per_bank : ℕ := 1400000

-- Theorem statement
theorem minimum_banks_needed :
  ∃ n : ℕ, n * max_insurance_payout_per_bank ≥ total_amount ∧ n = 8 :=
sorry

end minimum_banks_needed_l276_276783


namespace pages_copied_l276_276671

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l276_276671


namespace num_even_divisors_of_8_l276_276042

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l276_276042


namespace rachel_math_homework_l276_276394

/-- Rachel had to complete some pages of math homework. 
Given:
- 4 more pages of math homework than reading homework
- 3 pages of reading homework
Prove that Rachel had to complete 7 pages of math homework.
--/
theorem rachel_math_homework
  (r : ℕ) (h_r : r = 3)
  (m : ℕ) (h_m : m = r + 4) :
  m = 7 := by
  sorry

end rachel_math_homework_l276_276394


namespace baseball_singles_percentage_l276_276606

theorem baseball_singles_percentage :
  let total_hits := 50
  let home_runs := 2
  let triples := 3
  let doubles := 8
  let non_single_hits := home_runs + triples + doubles
  let singles := total_hits - non_single_hits
  let singles_percentage := (singles / total_hits) * 100
  singles = 37 ∧ singles_percentage = 74 :=
by
  sorry

end baseball_singles_percentage_l276_276606


namespace symmetric_point_coordinates_l276_276052

theorem symmetric_point_coordinates (a b : ℝ) (hp : (3, 4) = (a + 3, b + 4)) :
  (a, b) = (5, 2) :=
  sorry

end symmetric_point_coordinates_l276_276052


namespace sufficient_but_not_necessary_l276_276481

theorem sufficient_but_not_necessary {a b : ℝ} (h₁ : a < b) (h₂ : b < 0) : 
  (a^2 > b^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by
  sorry

end sufficient_but_not_necessary_l276_276481


namespace largest_divisible_by_6_ending_in_4_l276_276719

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l276_276719


namespace cost_of_balls_max_basketball_count_l276_276652

-- Define the prices of basketball and soccer ball
variables (x y : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := 2 * x + 3 * y = 310
def condition2 : Prop := 5 * x + 2 * y = 500

-- Proving the cost of each basketball and soccer ball
theorem cost_of_balls (h1 : condition1 x y) (h2 : condition2 x y) : x = 80 ∧ y = 50 :=
sorry

-- Define the total number of balls and the inequality constraint
variable (m : ℕ)
def total_balls_condition : Prop := m + (60 - m) = 60
def cost_constraint : Prop := 80 * m + 50 * (60 - m) ≤ 4000

-- Proving the maximum number of basketballs
theorem max_basketball_count (hc : cost_constraint m) (ht : total_balls_condition m) : m ≤ 33 :=
sorry

end cost_of_balls_max_basketball_count_l276_276652


namespace sufficient_but_not_necessary_l276_276677

theorem sufficient_but_not_necessary (a b : ℝ) :
  ((a - b) ^ 3 * b ^ 2 > 0 → a > b) ∧ ¬(a > b → (a - b) ^ 3 * b ^ 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_l276_276677


namespace volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l276_276108

namespace RectangularPrism

def length := 4
def width := 2
def height := 1

theorem volume_eq_eight : length * width * height = 8 := sorry

theorem space_diagonal_eq_sqrt21 :
  Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) = Real.sqrt 21 := sorry

theorem surface_area_neq_24 :
  2 * (length * width + width * height + height * length) ≠ 24 := sorry

theorem circumscribed_sphere_area_eq_21pi :
  4 * Real.pi * ((Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) / 2) ^ 2) = 21 * Real.pi := sorry

end RectangularPrism

end volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l276_276108


namespace solve_for_x_l276_276870

theorem solve_for_x (x : ℝ) (h : (5 * x - 3) / (6 * x - 6) = (4 / 3)) : x = 5 / 3 :=
sorry

end solve_for_x_l276_276870


namespace prime_sum_l276_276679

theorem prime_sum (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : 2 * p + 3 * q = 6 * r) : 
  p + q + r = 7 := 
sorry

end prime_sum_l276_276679


namespace equilateral_triangle_perimeter_twice_side_area_l276_276874

noncomputable def triangle_side_length (s : ℝ) :=
  s * s * Real.sqrt 3 / 4 = 2 * s

noncomputable def triangle_perimeter (s : ℝ) := 3 * s

theorem equilateral_triangle_perimeter_twice_side_area (s : ℝ) (h : triangle_side_length s) : 
  triangle_perimeter s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_twice_side_area_l276_276874


namespace suitable_high_jump_athlete_l276_276909

structure Athlete where
  average : ℕ
  variance : ℝ

def A : Athlete := ⟨169, 6.0⟩
def B : Athlete := ⟨168, 17.3⟩
def C : Athlete := ⟨169, 5.0⟩
def D : Athlete := ⟨168, 19.5⟩

def isSuitableCandidate (athlete: Athlete) (average_threshold: ℕ) : Prop :=
  athlete.average = average_threshold

theorem suitable_high_jump_athlete : isSuitableCandidate C 169 ∧
  (∀ a, isSuitableCandidate a 169 → a.variance ≥ C.variance) := by
  sorry

end suitable_high_jump_athlete_l276_276909


namespace number_of_divisors_36_l276_276974

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l276_276974


namespace average_minutes_per_player_l276_276854

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l276_276854


namespace find_a_l276_276568

-- Define the necessary variables
variables (a b : ℝ) (t : ℝ)

-- Given conditions
def b_val : ℝ := 2120
def t_val : ℝ := 0.5

-- The statement we need to prove
theorem find_a (h: b = b_val) (h2: t = t_val) (h3: t = a / b) : a = 1060 := by
  -- Placeholder for proof
  sorry

end find_a_l276_276568


namespace vertex_not_neg2_2_l276_276321

theorem vertex_not_neg2_2 (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : a * 1^2 + b * 1 + c = 0)
  (hsymm : ∀ x y, y = a * x^2 + b * x + c → y = a * (4 - x)^2 + b * (4 - x) + c) :
  ¬ ((-b) / (2 * a) = -2 ∧ a * (-2)^2 + b * (-2) + c = 2) :=
by
  sorry

end vertex_not_neg2_2_l276_276321


namespace sarah_numbers_sum_l276_276090

-- Definition of x and y being integers with their respective ranges
def isTwoDigit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def isThreeDigit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999

-- The condition relating x and y
def formedNumber (x y : ℕ) : Prop := 1000 * x + y = 7 * x * y

-- The Lean 4 statement for the proof problem
theorem sarah_numbers_sum (x y : ℕ) (H1 : isTwoDigit x) (H2 : isThreeDigit y) (H3 : formedNumber x y) : x + y = 1074 :=
  sorry

end sarah_numbers_sum_l276_276090


namespace largest_two_digit_divisible_by_6_ending_in_4_l276_276729

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l276_276729


namespace carries_jellybeans_l276_276333

/-- Bert's box holds 150 jellybeans. --/
def bert_jellybeans : ℕ := 150

/-- Carrie's box is three times as high, three times as wide, and three times as long as Bert's box. --/
def volume_ratio : ℕ := 27

/-- Given that Carrie's box dimensions are three times those of Bert's and Bert's box holds 150 jellybeans, 
    we need to prove that Carrie's box holds 4050 jellybeans. --/
theorem carries_jellybeans : bert_jellybeans * volume_ratio = 4050 := 
by sorry

end carries_jellybeans_l276_276333


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l276_276711

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l276_276711


namespace roots_of_star_eqn_l276_276318

def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

theorem roots_of_star_eqn :
  ∀ x : ℝ, ∃ a b c : ℝ, a = 1 ∧ b = -1 ∧ c = -1 ∧ star 1 x = a * x^2 + b * x + c ∧
    (b^2 - 4 * a * c > 0) :=
by
  intro x
  use [1, -1, -1]
  simp [star]
  sorry

end roots_of_star_eqn_l276_276318


namespace suresh_borrowed_amount_l276_276287

theorem suresh_borrowed_amount 
  (P: ℝ)
  (i1 i2 i3: ℝ)
  (t1 t2 t3: ℝ)
  (total_interest: ℝ)
  (h1 : i1 = 0.12) 
  (h2 : t1 = 3)
  (h3 : i2 = 0.09)
  (h4 : t2 = 5)
  (h5 : i3 = 0.13)
  (h6 : t3 = 3)
  (h_total : total_interest = 8160) 
  (h_interest_eq : total_interest = P * i1 * t1 + P * i2 * t2 + P * i3 * t3)
  : P = 6800 :=
by
  sorry

end suresh_borrowed_amount_l276_276287


namespace largest_two_digit_divisible_by_6_ending_in_4_l276_276730

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l276_276730


namespace find_a_l276_276623

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
sorry

end find_a_l276_276623


namespace sin_squared_plus_one_l276_276807

theorem sin_squared_plus_one (x : ℝ) (hx : Real.tan x = 2) : Real.sin x ^ 2 + 1 = 9 / 5 := 
by 
  sorry

end sin_squared_plus_one_l276_276807


namespace number_of_divisors_36_l276_276975

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l276_276975


namespace probability_at_least_one_woman_selected_l276_276487

theorem probability_at_least_one_woman_selected:
  let men := 10
  let women := 5
  let totalPeople := men + women
  let totalSelections := Nat.choose totalPeople 4
  let menSelections := Nat.choose men 4
  let noWomenProbability := (menSelections : ℚ) / (totalSelections : ℚ)
  let atLeastOneWomanProbability := 1 - noWomenProbability
  atLeastOneWomanProbability = 11 / 13 :=
by
  sorry

end probability_at_least_one_woman_selected_l276_276487


namespace slices_per_pizza_l276_276684

def num_pizzas : ℕ := 2
def total_slices : ℕ := 16

theorem slices_per_pizza : total_slices / num_pizzas = 8 := by
  sorry

end slices_per_pizza_l276_276684


namespace complete_task_in_3_days_l276_276157

theorem complete_task_in_3_days (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0)
  (h1 : 1 / x + 1 / y + 1 / z = 1 / 7.5)
  (h2 : 1 / x + 1 / z + 1 / v = 1 / 5)
  (h3 : 1 / x + 1 / z + 1 / w = 1 / 6)
  (h4 : 1 / y + 1 / w + 1 / v = 1 / 4) :
  1 / (1 / x + 1 / z + 1 / v + 1 / w + 1 / y) = 3 :=
sorry

end complete_task_in_3_days_l276_276157


namespace green_peaches_per_basket_l276_276411

-- Definitions based on given conditions
def total_peaches : ℕ := 10
def red_peaches_per_basket : ℕ := 4

-- Theorem statement based on the question and correct answer
theorem green_peaches_per_basket :
  (total_peaches - red_peaches_per_basket) = 6 := 
by
  sorry

end green_peaches_per_basket_l276_276411


namespace simplify_and_find_ratio_l276_276868

theorem simplify_and_find_ratio (k : ℤ) : (∃ (c d : ℤ), (∀ x y : ℤ, c = 1 ∧ d = 2 ∧ x = c ∧ y = d → ((6 * k + 12) / 6 = k + 2) ∧ (c / d = 1 / 2))) :=
by
  use 1
  use 2
  sorry

end simplify_and_find_ratio_l276_276868


namespace melissa_points_per_game_l276_276519

theorem melissa_points_per_game (total_points : ℕ) (games : ℕ) (h1 : total_points = 81) 
(h2 : games = 3) : total_points / games = 27 :=
by
  sorry

end melissa_points_per_game_l276_276519


namespace min_transport_cost_l276_276567

/- Definitions for the problem conditions -/
def villageA_vegetables : ℕ := 80
def villageB_vegetables : ℕ := 60
def destinationX_requirement : ℕ := 65
def destinationY_requirement : ℕ := 75

def cost_A_to_X : ℕ := 50
def cost_A_to_Y : ℕ := 30
def cost_B_to_X : ℕ := 60
def cost_B_to_Y : ℕ := 45

def W (x : ℕ) : ℕ :=
  cost_A_to_X * x +
  cost_A_to_Y * (villageA_vegetables - x) +
  cost_B_to_X * (destinationX_requirement - x) +
  cost_B_to_Y * (x - 5) + 6075 - 225

/- Prove that the minimum total cost W is 6100 -/
theorem min_transport_cost : ∃ (x : ℕ), 5 ≤ x ∧ x ≤ 65 ∧ W x = 6100 :=
by sorry

end min_transport_cost_l276_276567


namespace texas_california_plate_diff_l276_276401

def california_plates := 26^3 * 10^3
def texas_plates := 26^3 * 10^4
def plates_difference := texas_plates - california_plates

theorem texas_california_plate_diff :
  plates_difference = 158184000 :=
by sorry

end texas_california_plate_diff_l276_276401


namespace smallest_divisor_after_323_l276_276848

-- Let n be an even 4-digit number such that 323 is a divisor of n.
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisor (d n : ℕ) : Prop :=
  n % d = 0

theorem smallest_divisor_after_323 (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : is_divisor 323 n) : ∃ k, k > 323 ∧ is_divisor k n ∧ k = 340 :=
by
  sorry

end smallest_divisor_after_323_l276_276848


namespace find_taller_tree_height_l276_276888

-- Define the known variables and conditions
variables (H : ℕ) (ratio : ℚ) (difference : ℕ)

-- Specify the conditions from the problem
def taller_tree_height (H difference : ℕ) := H
def shorter_tree_height (H difference : ℕ) := H - difference
def height_ratio (H : ℕ) (ratio : ℚ) (difference : ℕ) :=
  (shorter_tree_height H difference : ℚ) / (taller_tree_height H difference : ℚ) = ratio

-- Prove the height of the taller tree given the conditions
theorem find_taller_tree_height (H : ℕ) (h_ratio : height_ratio H (2/3) 20) : 
  taller_tree_height H 20 = 60 :=
  sorry

end find_taller_tree_height_l276_276888


namespace tree_height_at_two_years_l276_276159

variable (h : ℕ → ℕ)

-- Given conditions
def condition1 := h 4 = 81
def condition2 := ∀ t : ℕ, h (t + 1) = 3 * h t

theorem tree_height_at_two_years
  (h_tripled : ∀ t : ℕ, h (t + 1) = 3 * h t)
  (h_at_four : h 4 = 81) :
  h 2 = 9 :=
by
  -- Formal proof will be provided here
  sorry

end tree_height_at_two_years_l276_276159


namespace num_pos_divisors_36_l276_276970

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l276_276970


namespace closest_point_to_origin_l276_276085

def y (x : ℝ) := x + 1 / x

theorem closest_point_to_origin : ∃ x : ℝ, x > 0 ∧ (x, y x) = (1 / 2^(1/4 : ℝ), (1 + real.sqrt 2) / 2^(1/4 : ℝ)) :=
by
  sorry

end closest_point_to_origin_l276_276085


namespace pooja_speed_l276_276692

theorem pooja_speed (v : ℝ) 
  (roja_speed : ℝ := 5)
  (distance : ℝ := 32)
  (time : ℝ := 4)
  (h : distance = (roja_speed + v) * time) : v = 3 :=
by
  sorry

end pooja_speed_l276_276692


namespace triple_solution_exists_and_unique_l276_276940

theorem triple_solution_exists_and_unique:
  ∀ (x y z : ℝ), (1 + x^4 ≤ 2 * (y - z) ^ 2) ∧ (1 + y^4 ≤ 2 * (z - x) ^ 2) ∧ (1 + z^4 ≤ 2 * (x - y) ^ 2)
  → (x = 1 ∧ y = 0 ∧ z = -1) :=
by
  sorry

end triple_solution_exists_and_unique_l276_276940


namespace distance_between_first_and_last_stop_in_km_l276_276294

-- Define the total number of stops
def num_stops := 12

-- Define the distance between the third and sixth stops in meters
def dist_3_to_6 := 3300

-- The distance between consecutive stops is the same
def distance_between_first_and_last_stop : ℕ := (num_stops - 1) * (dist_3_to_6 / 3)

-- The distance in kilometers (1 kilometer = 1000 meters)
noncomputable def distance_km : ℝ := distance_between_first_and_last_stop / 1000

-- Statement to prove
theorem distance_between_first_and_last_stop_in_km : distance_km = 12.1 :=
by
  -- Theorem proof should go here
  sorry

end distance_between_first_and_last_stop_in_km_l276_276294


namespace balance_difference_l276_276443

variables (P_A P_B n : ℕ) (r_A r_B : ℚ)

noncomputable def angela_balance (P_A : ℕ) (r_A : ℚ) (n : ℕ) : ℚ :=
P_A * (1 + r_A) ^ n

noncomputable def bob_balance (P_B : ℕ) (r_B : ℚ) (n : ℕ) : ℚ :=
P_B * (1 + r_B) ^ n

theorem balance_difference :
    angela_balance 9000 (5 / 100) 25 - bob_balance 10000 (45 / 1000) 25 ≈ 852 := by
  sorry 

end balance_difference_l276_276443


namespace minimize_theta_abs_theta_val_l276_276323

noncomputable def theta (k : ℤ) : ℝ := -11 / 4 * Real.pi + 2 * k * Real.pi

theorem minimize_theta_abs (k : ℤ) :
  ∃ θ : ℝ, (θ = -11 / 4 * Real.pi + 2 * k * Real.pi) ∧
           (∀ η : ℝ, (η = -11 / 4 * Real.pi + 2 * (k + 1) * Real.pi) →
             |θ| ≤ |η|) :=
  sorry

theorem theta_val : ∃ θ : ℝ, θ = -3 / 4 * Real.pi :=
  ⟨ -3 / 4 * Real.pi, rfl ⟩

end minimize_theta_abs_theta_val_l276_276323


namespace average_player_time_l276_276857

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l276_276857


namespace eggs_at_park_l276_276640

-- Define the number of eggs found at different locations
def eggs_at_club_house : Nat := 40
def eggs_at_town_hall : Nat := 15
def total_eggs_found : Nat := 80

-- Prove that the number of eggs found at the park is 25
theorem eggs_at_park :
  ∃ P : Nat, eggs_at_club_house + P + eggs_at_town_hall = total_eggs_found ∧ P = 25 := 
by
  sorry

end eggs_at_park_l276_276640


namespace convert_50_to_base_3_l276_276603

-- Define a function to convert decimal to ternary (base-3)
def convert_to_ternary (n : ℕ) : ℕ := sorry

-- Main theorem statement
theorem convert_50_to_base_3 : convert_to_ternary 50 = 1212 :=
sorry

end convert_50_to_base_3_l276_276603


namespace cos_2theta_l276_276480

theorem cos_2theta (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2*n) = 8) : 
  Real.cos (2 * θ) = 3 / 4 := 
sorry

end cos_2theta_l276_276480


namespace geometric_sequence_ratio_l276_276884

theorem geometric_sequence_ratio (a : ℕ → ℤ) (q : ℤ) (n : ℕ) (i : ℕ → ℕ) (ε : ℕ → ℤ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → a k = a 1 * q ^ (k - 1)) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n → ε k * a (i k) = 0) ∧
  (∀ m, 1 ≤ i m ∧ i m ≤ n) → q = -1 := 
sorry

end geometric_sequence_ratio_l276_276884


namespace betty_needs_more_flies_l276_276465

def flies_per_day := 2
def days_per_week := 7
def flies_needed_per_week := flies_per_day * days_per_week

def flies_caught_morning := 5
def flies_caught_afternoon := 6
def fly_escaped := 1

def flies_caught_total := flies_caught_morning + flies_caught_afternoon - fly_escaped

theorem betty_needs_more_flies : 
  flies_needed_per_week - flies_caught_total = 4 := by
  sorry

end betty_needs_more_flies_l276_276465


namespace count_squares_containing_A_l276_276750

-- Given conditions
def figure_with_squares : Prop := ∃ n : ℕ, n = 20

-- The goal is to prove that the number of squares containing A is 13
theorem count_squares_containing_A (h : figure_with_squares) : ∃ k : ℕ, k = 13 :=
by 
  sorry

end count_squares_containing_A_l276_276750


namespace reading_time_per_week_l276_276562

variable (meditation_time_per_day : ℕ)
variable (reading_factor : ℕ)

theorem reading_time_per_week (h1 : meditation_time_per_day = 1) (h2 : reading_factor = 2) : 
  (reading_factor * meditation_time_per_day * 7) = 14 :=
by
  sorry

end reading_time_per_week_l276_276562


namespace remainder_2519_div_7_l276_276777

theorem remainder_2519_div_7 : 2519 % 7 = 6 :=
by
  sorry

end remainder_2519_div_7_l276_276777


namespace guzman_boxes_l276_276850

noncomputable def total_doughnuts : Nat := 48
noncomputable def doughnuts_per_box : Nat := 12

theorem guzman_boxes :
  ∃ (N : Nat), N = total_doughnuts / doughnuts_per_box ∧ N = 4 :=
by
  use 4
  sorry

end guzman_boxes_l276_276850


namespace returning_players_count_l276_276886

def total_players_in_team (groups : ℕ) (players_per_group : ℕ): ℕ := groups * players_per_group
def returning_players (total_players : ℕ) (new_players : ℕ): ℕ := total_players - new_players

theorem returning_players_count
    (new_players : ℕ)
    (groups : ℕ)
    (players_per_group : ℕ)
    (total_players : ℕ := total_players_in_team groups players_per_group)
    (returning_players_count : ℕ := returning_players total_players new_players):
    new_players = 4 ∧
    groups = 2 ∧
    players_per_group = 5 → 
    returning_players_count = 6 := by
    intros h
    sorry

end returning_players_count_l276_276886


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276725

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276725


namespace find_f_4500_l276_276015

noncomputable def f : ℕ → ℕ
| 0 => 1
| (n + 3) => f n + 2 * n + 3
| n => sorry  -- This handles all other cases, but should not be called.

theorem find_f_4500 : f 4500 = 6750001 :=
by
  sorry

end find_f_4500_l276_276015


namespace diagonals_in_decagon_l276_276178

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end diagonals_in_decagon_l276_276178


namespace novel_writing_time_l276_276311

theorem novel_writing_time :
  ∀ (total_words : ℕ) (first_half_speed second_half_speed : ℕ),
    total_words = 50000 →
    first_half_speed = 600 →
    second_half_speed = 400 →
    (total_words / 2 / first_half_speed + total_words / 2 / second_half_speed : ℚ) = 104.17 :=
by
  -- No proof is required, placeholder using sorry
  sorry

end novel_writing_time_l276_276311


namespace angle_in_third_quadrant_l276_276045

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
sorry

end angle_in_third_quadrant_l276_276045


namespace complex_division_l276_276196

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 - i) = -1 + i :=
by sorry

end complex_division_l276_276196


namespace simple_interest_rate_l276_276613

theorem simple_interest_rate (P T SI : ℝ) (hP : P = 10000) (hT : T = 1) (hSI : SI = 400) :
    (SI = P * 0.04 * T) := by
  rw [hP, hT, hSI]
  sorry

end simple_interest_rate_l276_276613


namespace height_of_shorter_tree_l276_276270

theorem height_of_shorter_tree (H h : ℝ) (h_difference : H = h + 20) (ratio : h / H = 5 / 7) : h = 50 := 
by
  sorry

end height_of_shorter_tree_l276_276270


namespace surface_area_increase_l276_276452

structure RectangularSolid (length : ℝ) (width : ℝ) (height : ℝ) where
  surface_area : ℝ := 2 * (length * width + length * height + width * height)

def cube_surface_contributions (side : ℝ) : ℝ := side ^ 2 * 3

theorem surface_area_increase
  (original : RectangularSolid 4 3 5)
  (cube_side : ℝ := 1) :
  let new_cube_contribution := cube_surface_contributions cube_side
  let removed_face : ℝ := cube_side ^ 2
  let original_surface_area := original.surface_area
  original_surface_area + new_cube_contribution - removed_face = original_surface_area + 2 :=
by
  sorry

end surface_area_increase_l276_276452


namespace probability_product_zero_l276_276565

open Finset

theorem probability_product_zero :
  let s := \{ -3, -1, 0, 2, 4, 6, 7\} : Finset ℤ,
  n := s.card,
  total := nat.choose n 2,
  favorable := 6
  in  favorable / total = (2:ℚ) / 7 :=
by
  let s := \{ -3, -1, 0, 2, 4, 6, 7\} : Finset ℤ
  let n := s.card
  let total := nat.choose n 2
  let favorable := 6
  have h : favorable / total = (2: ℚ) / 7 := sorry
  exact h

end probability_product_zero_l276_276565


namespace find_m_l276_276955

variables (AB AC AD : ℝ × ℝ)
variables (m : ℝ)

-- Definitions of vectors
def vector_AB : ℝ × ℝ := (-1, 2)
def vector_AC : ℝ × ℝ := (2, 3)
def vector_AD (m : ℝ) : ℝ × ℝ := (m, -3)

-- Conditions
def collinear (B C D : ℝ × ℝ) : Prop := ∃ k : ℝ, B = k • C ∨ C = k • D ∨ D = k • B

-- Main statement to prove
theorem find_m (h1 : vector_AB = (-1, 2))
               (h2 : vector_AC = (2, 3))
               (h3 : vector_AD m = (m, -3))
               (h4 : collinear vector_AB vector_AC (vector_AD m)) :
  m = -16 :=
sorry

end find_m_l276_276955


namespace rhombus_area_l276_276476

theorem rhombus_area (side_length : ℝ) (d1_diff_d2 : ℝ) 
  (h_side_length : side_length = Real.sqrt 104) 
  (h_d1_diff_d2 : d1_diff_d2 = 10) : 
  (1 / 2) * (2 * Real.sqrt 104 - d1_diff_d2) * (d1_diff_d2 + 2 * Real.sqrt 104) = 79.17 :=
by
  sorry

end rhombus_area_l276_276476


namespace angle_B_is_pi_div_3_sin_C_value_l276_276829

-- Definitions and conditions
variable (A B C a b c : ℝ)
variable (cos_cos_eq : (2 * a - c) * Real.cos B = b * Real.cos C)
variable (triangle_ineq : 0 < A ∧ A < Real.pi)
variable (sin_positive : Real.sin A > 0)
variable (a_eq_2 : a = 2)
variable (c_eq_3 : c = 3)

-- Proving B = π / 3 under given conditions
theorem angle_B_is_pi_div_3 : B = Real.pi / 3 := sorry

-- Proving sin C under given additional conditions
theorem sin_C_value : Real.sin C = 3 * Real.sqrt 14 / 14 := sorry

end angle_B_is_pi_div_3_sin_C_value_l276_276829


namespace determine_unique_row_weight_free_l276_276292

theorem determine_unique_row_weight_free (t : ℝ) (rows : Fin 10 → ℝ) (unique_row : Fin 10)
  (h_weights_same : ∀ i : Fin 10, i ≠ unique_row → rows i = t) :
  0 = 0 := by
  sorry

end determine_unique_row_weight_free_l276_276292


namespace eccentricity_of_ellipse_l276_276927

theorem eccentricity_of_ellipse (p q : ℕ) (hp : Nat.Coprime p q) (z : ℂ) :
  ((z - 2) * (z^2 + 3 * z + 5) * (z^2 + 5 * z + 8) = 0) →
  (∃ p q : ℕ, Nat.Coprime p q ∧ (∃ e : ℝ, e^2 = p / q ∧ p + q = 16)) :=
by
  sorry

end eccentricity_of_ellipse_l276_276927


namespace divides_polynomial_l276_276382

theorem divides_polynomial (n : ℕ) (x : ℤ) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(n+2) + (x+1)^(2*n+1)) :=
sorry

end divides_polynomial_l276_276382


namespace largest_two_digit_divisible_by_6_ending_in_4_l276_276733

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l276_276733


namespace minimum_value_of_f_l276_276462

noncomputable def f (x : ℝ) : ℝ := x^2 + 10*x + (100 / x^3)

theorem minimum_value_of_f :
  ∃ x > 0, f x = 40 ∧ ∀ y > 0, f y ≥ 40 :=
sorry

end minimum_value_of_f_l276_276462


namespace solve_system_eqns_l276_276533

theorem solve_system_eqns (x y z : ℝ) (h1 : x^3 + y^3 + z^3 = 8)
  (h2 : x^2 + y^2 + z^2 = 22)
  (h3 : 1/x + 1/y + 1/z + z/(x * y) = 0) :
  (x = 3 ∧ y = 2 ∧ z = -3) ∨ (x = -3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = -3) ∨ (x = 2 ∧ y = -3 ∧ z = 3) :=
by
  sorry

end solve_system_eqns_l276_276533


namespace smallest_three_digit_number_with_property_l276_276329

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a < 1000 ∧ ∃ n : ℕ, 1000 * a + (a + 1) = n^2 ∧ a = 183 :=
sorry

end smallest_three_digit_number_with_property_l276_276329


namespace bead_count_l276_276078

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end bead_count_l276_276078


namespace number_of_problems_l276_276390

/-- Given the conditions of the problem, prove that the number of problems I did is exactly 140.-/
theorem number_of_problems (p t : ℕ) (h1 : p > 12) (h2 : p * t = (p + 6) * (t - 3)) : p * t = 140 :=
by
  sorry

end number_of_problems_l276_276390


namespace truth_probability_l276_276484

theorem truth_probability (P_A : ℝ) (P_A_and_B : ℝ) (P_B : ℝ) 
  (hA : P_A = 0.70) (hA_and_B : P_A_and_B = 0.42) : 
  P_A * P_B = P_A_and_B → P_B = 0.6 :=
by
  sorry

end truth_probability_l276_276484


namespace fermats_little_theorem_l276_276427

theorem fermats_little_theorem 
  (a n : ℕ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < n) 
  (h₃ : Nat.gcd a n = 1) 
  (phi : ℕ := (Nat.totient n)) 
  : n ∣ (a ^ phi - 1) := sorry

end fermats_little_theorem_l276_276427


namespace median_of_first_twelve_even_integers_l276_276417

open Real

-- Define the first twelve positive integers that are even
def even_integers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Define the sixth and seventh integers of that list
def sixth_integer : ℕ := even_integers.nthLe 5 (by decide)
def seventh_integer : ℕ := even_integers.nthLe 6 (by decide)

-- Define the median calculation formula using sixth and seventh integers
def calculate_median : ℝ := (sixth_integer + seventh_integer) / 2

-- The theorem to prove
theorem median_of_first_twelve_even_integers : calculate_median = 13.0 := by
  sorry

end median_of_first_twelve_even_integers_l276_276417


namespace Kelsey_watched_537_videos_l276_276125

-- Definitions based on conditions
def total_videos : ℕ := 1222
def delilah_videos : ℕ := 78

-- Declaration of variables representing the number of videos each friend watched
variables (Kelsey Ekon Uma Ivan Lance : ℕ)

-- Conditions from the problem
def cond1 : Kelsey = 3 * Ekon := sorry
def cond2 : Ekon = Uma - 23 := sorry
def cond3 : Uma = 2 * Ivan := sorry
def cond4 : Lance = Ivan + 19 := sorry
def cond5 : delilah_videos = 78 := sorry
def cond6 := Kelsey + Ekon + Uma + Ivan + Lance + delilah_videos = total_videos

-- The theorem to prove
theorem Kelsey_watched_537_videos : Kelsey = 537 :=
  by
  sorry

end Kelsey_watched_537_videos_l276_276125


namespace hyperbola_eccentricity_l276_276538

-- Definitions of conditions
def asymptotes_of_hyperbola (a b x y : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  (b * x + a * y = 0) ∨ (b * x - a * y = 0)

def circle_tangent_to_asymptotes (x y a b : ℝ) : Prop :=
  ∀ x1 y1 : ℝ, 
  (x1, y1) = (0, 4) → 
  (Real.sqrt (b^2 + a^2) = 2 * a)

-- Main statement
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (h_asymptotes : ∀ (x y : ℝ), asymptotes_of_hyperbola a b x y h_a h_b) 
  (h_tangent : circle_tangent_to_asymptotes 0 4 a b) : 
  ∃ e : ℝ, e = 2 := 
sorry

end hyperbola_eccentricity_l276_276538


namespace fraction_addition_l276_276284

variable (d : ℝ)

theorem fraction_addition (d : ℝ) : (5 + 2 * d) / 8 + 3 = (29 + 2 * d) / 8 := 
sorry

end fraction_addition_l276_276284


namespace find_perimeter_l276_276748

-- Define the area function of an equilateral triangle
def area_equilateral (a : ℝ) : ℝ := (sqrt 3 / 4) * a^2

-- Define the given area
def given_area : ℝ := 50 * sqrt 12

-- Define the perimeter function
def perimeter_equilateral (a : ℝ) : ℝ := 3 * a

-- Proof statement
theorem find_perimeter (a : ℝ) (h : area_equilateral a = given_area) : perimeter_equilateral a = 60 := sorry

end find_perimeter_l276_276748


namespace residents_rent_contribution_l276_276548

theorem residents_rent_contribution (x R : ℝ) (hx1 : 10 * x + 88 = R) (hx2 : 10.80 * x = 1.025 * R) :
  R / x = 10.54 :=
by sorry

end residents_rent_contribution_l276_276548


namespace f_lt_2_l276_276618

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f (x + 2) = f (-x + 2)

axiom f_ge_2 (x : ℝ) (h : x ≥ 2) : f x = x^2 - 6 * x + 4

theorem f_lt_2 (x : ℝ) (h : x < 2) : f x = x^2 - 2 * x - 4 :=
by
  sorry

end f_lt_2_l276_276618


namespace find_f_log2_20_l276_276928

noncomputable def f (x : ℝ) : ℝ :=
if -1 < x ∧ x < 0 then 2^x + 1 else sorry

lemma f_periodic (x : ℝ) : f (x - 2) = f (x + 2) :=
sorry

lemma f_odd (x : ℝ) : f (-x) = -f (x) :=
sorry

theorem find_f_log2_20 : f (Real.log 20 / Real.log 2) = -1 :=
sorry

end find_f_log2_20_l276_276928


namespace fraction_of_widgets_second_shift_l276_276922

theorem fraction_of_widgets_second_shift (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let first_shift_widgets := x * y
  let second_shift_widgets := (2 / 3) * x * (4 / 3) * y
  let total_widgets := first_shift_widgets + second_shift_widgets
  let fraction_second_shift := second_shift_widgets / total_widgets
  fraction_second_shift = 8 / 17 :=
by
  sorry

end fraction_of_widgets_second_shift_l276_276922


namespace quadratic_inequality_solution_l276_276790

theorem quadratic_inequality_solution (x : ℝ) : -3 < x ∧ x < 4 → x^2 - x - 12 < 0 := by
  sorry

end quadratic_inequality_solution_l276_276790


namespace negation_of_universal_l276_276405

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry    -- Proof is not required, just the statement.

end negation_of_universal_l276_276405


namespace complement_set_l276_276034

open Set

theorem complement_set (U M : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hM : M = {1, 2, 4}) :
  compl M ∩ U = {3, 5, 6} := 
by
  rw [compl, hU, hM]
  sorry

end complement_set_l276_276034


namespace triangle_base_length_l276_276873

theorem triangle_base_length (base : ℝ) (h1 : ∃ (side : ℝ), side = 6 ∧ (side^2 = (base * 12) / 2)) : base = 6 :=
sorry

end triangle_base_length_l276_276873


namespace solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l276_276954

theorem solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c ≤ 0 ↔ x ≤ -1 ∨ x ≥ 3) →
  b = -2*a →
  c = -3*a →
  a < 0 →
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) := 
by 
  intro h_root_set h_b_eq h_c_eq h_a_lt_0 
  sorry

end solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l276_276954


namespace zero_point_interval_l276_276048

noncomputable def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem zero_point_interval: 
  ∃ x₀ : ℝ, f x₀ = 0 → 1 < x₀ ∧ x₀ < 2 :=
sorry

end zero_point_interval_l276_276048


namespace odd_times_even_is_even_l276_276592

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_times_even_is_even (a b : ℤ) (h₁ : is_odd a) (h₂ : is_even b) : is_even (a * b) :=
by sorry

end odd_times_even_is_even_l276_276592


namespace problem1_problem2_l276_276293

-- Problem 1:
theorem problem1 (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  (∃ m b, (∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0)) ∧ ∀ x y, (x = -1 → y = 0 → y = m * x + b)) → 
  ∃ m b, ∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0) :=
sorry

-- Problem 2:
theorem problem2 (L1 : ℝ → ℝ → Prop) (hL1 : ∀ x y, L1 x y ↔ 3 * x + 4 * y - 12 = 0) (d : ℝ) (hd : d = 7) :
  (∃ c, ∀ x y, (3 * x + 4 * y + c = 0 ∨ 3 * x + 4 * y - 47 = 0) ↔ L1 x y ∧ d = 7) :=
sorry

end problem1_problem2_l276_276293


namespace min_handshakes_l276_276765

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l276_276765


namespace water_needed_four_weeks_l276_276392

theorem water_needed_four_weeks :
  ∀ (n : ℕ) (water_first_two tanks : Σ (t1 t2 t3 t4 : ℕ), (t1 = t2 ∧ t1 = 8 ∧ t3 = t4 ∧ t3 = t1 - 2)) (water_per_week : Σ (w : ℕ), (w = 28)),
  water_first_two.1 = 8 →
  water_first_two.2.1 = 8 →
  water_first_two.2.2.1 = water_first_two.1 - 2 →
  water_first_two.2.2.2.1 = water_first_two.2.2.1 →
  water_per_week.1 = water_first_two.1 * 2 + water_first_two.2.2.1 * 2 →
  n = 4 →
  water_per_week.1 * n = 112 := 
begin
  sorry
end

end water_needed_four_weeks_l276_276392


namespace minimum_handshakes_l276_276770

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l276_276770


namespace money_given_to_last_set_l276_276520

theorem money_given_to_last_set (total first second third fourth last : ℝ) 
  (h_total : total = 4500) 
  (h_first : first = 725) 
  (h_second : second = 1100) 
  (h_third : third = 950) 
  (h_fourth : fourth = 815) 
  (h_sum: total = first + second + third + fourth + last) : 
  last = 910 :=
sorry

end money_given_to_last_set_l276_276520


namespace nora_muffin_price_l276_276685

theorem nora_muffin_price
  (cases : ℕ)
  (packs_per_case : ℕ)
  (muffins_per_pack : ℕ)
  (total_money : ℕ)
  (total_cases : ℕ)
  (h1 : total_money = 120)
  (h2 : packs_per_case = 3)
  (h3 : muffins_per_pack = 4)
  (h4 : total_cases = 5) :
  (total_money / (total_cases * packs_per_case * muffins_per_pack) = 2) :=
by
  sorry

end nora_muffin_price_l276_276685


namespace sum_base6_l276_276917

theorem sum_base6 (a b : ℕ) (h₁ : a = 5) (h₂ : b = 23) : 
  let sum := Nat.ofDigits 6 [2, 3] + Nat.ofDigits 6 [5]
  in Nat.digits 6 sum = [2, 3] :=
by
  sorry

end sum_base6_l276_276917


namespace simplest_radical_l276_276136

theorem simplest_radical (r1 r2 r3 r4 : ℝ) 
  (h1 : r1 = Real.sqrt 3) 
  (h2 : r2 = Real.sqrt 4)
  (h3 : r3 = Real.sqrt 8)
  (h4 : r4 = Real.sqrt (1 / 2)) : r1 = Real.sqrt 3 :=
  by sorry

end simplest_radical_l276_276136


namespace statement_B_statement_D_l276_276573

variable {a b c d : ℝ}

theorem statement_B (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : (c / a) > (c / b) := 
by sorry

theorem statement_D (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : (a * c) < (b * d) := 
by sorry

end statement_B_statement_D_l276_276573


namespace initial_mixture_volume_l276_276150

variable (p q : ℕ) (x : ℕ)

theorem initial_mixture_volume :
  (3 * x) + (2 * x) = 5 * x →
  (3 * x) / (2 * x + 12) = 3 / 4 →
  5 * x = 30 :=
by
  sorry

end initial_mixture_volume_l276_276150


namespace copy_pages_l276_276660

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l276_276660


namespace probability_at_least_one_woman_selected_l276_276488

theorem probability_at_least_one_woman_selected:
  let men := 10
  let women := 5
  let totalPeople := men + women
  let totalSelections := Nat.choose totalPeople 4
  let menSelections := Nat.choose men 4
  let noWomenProbability := (menSelections : ℚ) / (totalSelections : ℚ)
  let atLeastOneWomanProbability := 1 - noWomenProbability
  atLeastOneWomanProbability = 11 / 13 :=
by
  sorry

end probability_at_least_one_woman_selected_l276_276488


namespace percent_profit_l276_276641

theorem percent_profit (C S : ℝ) (h : 60 * C = 50 * S):
  (((S - C) / C) * 100) = 20 :=
by 
  sorry

end percent_profit_l276_276641


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l276_276708

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l276_276708


namespace negation_seated_l276_276406

variable (Person : Type) (in_room : Person → Prop) (seated : Person → Prop)

theorem negation_seated :
  ¬ (∀ x, in_room x → seated x) ↔ ∃ x, in_room x ∧ ¬ seated x :=
by sorry

end negation_seated_l276_276406


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276700

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276700


namespace sum_two_numbers_l276_276422

theorem sum_two_numbers :
  let X := (2 * 10) + 6
  let Y := (4 * 10) + 1
  X + Y = 67 :=
by
  sorry

end sum_two_numbers_l276_276422


namespace train_speed_45_kmph_l276_276914

variable (length_train length_bridge time_passed : ℕ)

def total_distance (length_train length_bridge : ℕ) : ℕ :=
  length_train + length_bridge

def speed_m_per_s (length_train length_bridge time_passed : ℕ) : ℚ :=
  (total_distance length_train length_bridge) / time_passed

def speed_km_per_h (length_train length_bridge time_passed : ℕ) : ℚ :=
  (speed_m_per_s length_train length_bridge time_passed) * 3.6

theorem train_speed_45_kmph :
  length_train = 360 → length_bridge = 140 → time_passed = 40 → speed_km_per_h length_train length_bridge time_passed = 45 := 
by
  sorry

end train_speed_45_kmph_l276_276914


namespace cube_colorings_distinguishable_l276_276774

-- Define the problem
def cube_construction_distinguishable_ways : Nat :=
  30

-- The theorem we need to prove
theorem cube_colorings_distinguishable :
  ∃ (ways : Nat), ways = cube_construction_distinguishable_ways :=
by
  sorry

end cube_colorings_distinguishable_l276_276774


namespace purely_imaginary_z_point_on_line_z_l276_276814

-- Proof problem for (I)
theorem purely_imaginary_z (a : ℝ) (z : ℂ) (h : z = Complex.mk 0 (a+2)) 
: a = 2 :=
sorry

-- Proof problem for (II)
theorem point_on_line_z (a : ℝ) (x y : ℝ) (h1 : x = a^2-4) (h2 : y = a+2) (h3 : x + 2*y + 1 = 0) 
: a = -1 :=
sorry

end purely_imaginary_z_point_on_line_z_l276_276814


namespace average_minutes_per_player_l276_276856

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l276_276856


namespace threshold_mu_l276_276796

/-- 
Find threshold values μ₁₀₀ and μ₁₀₀₀₀₀ such that 
F = m * n * sin (π / m) * sqrt (1 / n² + sin⁴ (π / m)) 
is definitely greater than 100 and 1,000,000 respectively for all m greater than μ₁₀₀ and μ₁₀₀₀₀₀, 
assuming n = m³. -/
theorem threshold_mu : 
  (∃ (μ₁₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 100) ∧ 
  (∃ (μ₁₀₀₀₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀₀₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 1000000) :=
sorry

end threshold_mu_l276_276796


namespace at_least_one_negative_l276_276514

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a^2 + 1 / b = b^2 + 1 / a) : a < 0 ∨ b < 0 :=
by
  sorry

end at_least_one_negative_l276_276514


namespace remainder_a52_div_52_l276_276068

def a_n (n : ℕ) : ℕ := 
  (List.range (n + 1)).foldl (λ acc x => acc * 10 ^ (Nat.digits 10 x).length + x) 0

theorem remainder_a52_div_52 : (a_n 52) % 52 = 28 := 
  by
  sorry

end remainder_a52_div_52_l276_276068


namespace largestValidNumberIs84_l276_276698

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l276_276698


namespace d_in_N_l276_276376

def M := {x : ℤ | ∃ n : ℤ, x = 3 * n}
def N := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def P := {x : ℤ | ∃ n : ℤ, x = 3 * n - 1}

theorem d_in_N (a b c d : ℤ) (ha : a ∈ M) (hb : b ∈ N) (hc : c ∈ P) (hd : d = a - b + c) : d ∈ N :=
by sorry

end d_in_N_l276_276376


namespace largestValidNumberIs84_l276_276696

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l276_276696


namespace find_p_q_l276_276483

theorem find_p_q (p q : ℚ) : 
    (∀ x, x^5 - x^4 + x^3 - p*x^2 + q*x + 9 = 0 → (x = -3 ∨ x = 2)) →
    (p, q) = (-19.5, -55.5) :=
by {
  sorry
}

end find_p_q_l276_276483


namespace ellipse_chord_through_focus_l276_276591

theorem ellipse_chord_through_focus (x y : ℝ) (a b : ℝ := 6) (c : ℝ := 3 * Real.sqrt 3)
  (F : ℝ × ℝ := (3 * Real.sqrt 3, 0)) (AF BF : ℝ) :
  (x^2 / 36) + (y^2 / 9) = 1 ∧ ((x - 3 * Real.sqrt 3)^2 + y^2 = (3/2)^2) ∧
  (AF = 3 / 2) ∧ F.1 = 3 * Real.sqrt 3 ∧ F.2 = 0 →
  BF = 3 / 2 :=
sorry

end ellipse_chord_through_focus_l276_276591


namespace roots_of_polynomial_l276_276942

theorem roots_of_polynomial :
  ∀ x : ℝ, (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 → x = 1 :=
by
  sorry

end roots_of_polynomial_l276_276942


namespace floor_div_of_M_l276_276024

open BigOperators

theorem floor_div_of_M {M : ℕ} 
  (h : ∑ k in finset.range(8) (3 + k).fact * (22 - (3 + k)).fact = 21! * M) : 
  M = 95290 → floor (M / 100) = 952 :=
by
  sorry

end floor_div_of_M_l276_276024


namespace Matias_sales_l276_276229

def books_sold (Tuesday Wednesday Thursday : Nat) : Prop :=
  Tuesday = 7 ∧ 
  Wednesday = 3 * Tuesday ∧ 
  Thursday = 3 * Wednesday ∧ 
  Tuesday + Wednesday + Thursday = 91

theorem Matias_sales
  (Tuesday Wednesday Thursday : Nat) :
  books_sold Tuesday Wednesday Thursday := by
  sorry

end Matias_sales_l276_276229


namespace geometric_sequence_sum_l276_276148

theorem geometric_sequence_sum (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/2) (S_n : ℚ) (h_S_n : S_n = 80/243) : ∃ n : ℕ, S_n = a * ((1 - r^n) / (1 - r)) ∧ n = 4 := by
  sorry

end geometric_sequence_sum_l276_276148


namespace probability_coprime_integers_l276_276340

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem probability_coprime_integers :
  let S := {2, 3, 4, 5, 6, 7, 8}
  let pairs := (Set.to_finset S).pairs
  let coprime_pairs := pairs.filter (λ (p : ℕ × ℕ), is_coprime p.1 p.2)
  (coprime_pairs.card : ℚ) / pairs.card = 2 / 3 :=
by
  sorry

end probability_coprime_integers_l276_276340


namespace afternoon_pear_sales_l276_276153

theorem afternoon_pear_sales (morning_sales afternoon_sales total_sales : ℕ)
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : total_sales = morning_sales + afternoon_sales)
  (h3 : total_sales = 420) : 
  afternoon_sales = 280 :=
by {
  -- placeholders for the proof
  sorry 
}

end afternoon_pear_sales_l276_276153


namespace max_area_of_rectangle_l276_276117

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l276_276117


namespace second_set_parallel_lines_l276_276638

theorem second_set_parallel_lines (n : ℕ) (h : 7 * (n - 1) = 784) : n = 113 := 
by
  sorry

end second_set_parallel_lines_l276_276638


namespace sixth_equation_l276_276081

theorem sixth_equation :
  (6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 = 121) :=
by
  sorry

end sixth_equation_l276_276081


namespace traders_fabric_sales_l276_276127

theorem traders_fabric_sales (x y : ℕ) : 
  x + y = 85 ∧
  x = y + 5 ∧
  60 = x * (60 / y) ∧
  30 = y * (30 / x) →
  (x, y) = (25, 20) :=
by {
  sorry
}

end traders_fabric_sales_l276_276127


namespace f_neg_2_f_monotonically_decreasing_l276_276345

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 4
axiom f_2 : f 2 = 0
axiom f_pos_2 (x : ℝ) : x > 2 → f x < 0

-- Statement to prove f(-2) = 8
theorem f_neg_2 : f (-2) = 8 := sorry

-- Statement to prove that f(x) is monotonically decreasing on ℝ
theorem f_monotonically_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := sorry

end f_neg_2_f_monotonically_decreasing_l276_276345


namespace positive_integer_divisors_of_sum_l276_276947

theorem positive_integer_divisors_of_sum (n : ℕ) :
  (∃ n_values : Finset ℕ, 
    (∀ n ∈ n_values, n > 0 
      ∧ (n * (n + 1)) ∣ (2 * 10 * n)) 
      ∧ n_values.card = 5) :=
by
  sorry

end positive_integer_divisors_of_sum_l276_276947


namespace perimeter_of_square_l276_276747

theorem perimeter_of_square (s : ℝ) (area : s^2 = 468) : 4 * s = 24 * Real.sqrt 13 := 
by
  sorry

end perimeter_of_square_l276_276747


namespace time_brushing_each_cat_l276_276819

theorem time_brushing_each_cat :
  ∀ (t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats : ℕ),
  t_total_free_time = 3 * 60 →
  t_vacuum = 45 →
  t_dust = 60 →
  t_mop = 30 →
  t_cats = 3 →
  t_free_left_after_cleaning = 30 →
  ((t_total_free_time - t_free_left_after_cleaning) - (t_vacuum + t_dust + t_mop)) / t_cats = 5
 := by
  intros t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats
  intros h_total_free_time h_vacuum h_dust h_mop h_cats h_free_left
  sorry

end time_brushing_each_cat_l276_276819


namespace inequality_proof_l276_276863

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end inequality_proof_l276_276863


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l276_276707

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l276_276707


namespace square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l276_276342

variable {a b : ℝ}

theorem square_inequality_not_sufficient_nor_necessary_for_cube_inequality (a b : ℝ) :
  (a^2 > b^2) ↔ (a^3 > b^3) = false :=
sorry

end square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l276_276342


namespace kendalls_total_distance_l276_276503

-- Definitions of the conditions
def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5

-- The theorem to prove the total distance
theorem kendalls_total_distance : distance_with_mother + distance_with_father = 0.67 :=
by
  sorry

end kendalls_total_distance_l276_276503


namespace product_roots_cos_pi_by_9_cos_2pi_by_9_l276_276743

theorem product_roots_cos_pi_by_9_cos_2pi_by_9 :
  ∀ (d e : ℝ), (∀ x, x^2 + d * x + e = (x - Real.cos (π / 9)) * (x - Real.cos (2 * π / 9))) → 
    d * e = -5 / 64 :=
by
  sorry

end product_roots_cos_pi_by_9_cos_2pi_by_9_l276_276743


namespace new_class_mean_l276_276056

theorem new_class_mean (n1 n2 : ℕ) (mean1 mean2 : ℝ) (h1 : n1 = 45) (h2 : n2 = 5) (h3 : mean1 = 0.85) (h4 : mean2 = 0.90) : 
(n1 + n2 = 50) → 
((n1 * mean1 + n2 * mean2) / (n1 + n2) = 0.855) := 
by
  intro total_students
  sorry

end new_class_mean_l276_276056


namespace graph_of_f_does_not_pass_through_second_quadrant_l276_276104

def f (x : ℝ) : ℝ := x - 2

theorem graph_of_f_does_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = f x ∧ x < 0 ∧ y > 0 :=
sorry

end graph_of_f_does_not_pass_through_second_quadrant_l276_276104


namespace no_real_roots_of_quadratic_eq_l276_276358

theorem no_real_roots_of_quadratic_eq (k : ℝ) (h : k < -1) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - k = 0 :=
by
  sorry

end no_real_roots_of_quadratic_eq_l276_276358


namespace angle_bisector_inequality_l276_276657

theorem angle_bisector_inequality {a b c fa fb fc : ℝ} 
  (h_triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (1 / fa + 1 / fb + 1 / fc > 1 / a + 1 / b + 1 / c) :=
by
  sorry

end angle_bisector_inequality_l276_276657


namespace largest_divisible_by_6_ending_in_4_l276_276722

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l276_276722


namespace zero_points_of_f_l276_276410

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem zero_points_of_f : (f (-1/2) = 0) ∧ (f (-1) = 0) :=
by
  sorry

end zero_points_of_f_l276_276410


namespace circumradius_of_A_l276_276589

open EuclideanGeometry

variables {A B C A' B' C' : Point}
variable {r : ℝ}

-- Conditions
variable (ABC_inradius : ∀ {ABC : Triangle}, inradius ABC = r)
variable (A_circle_orthogonal : orthogonal_circle_through A C)
variable (B_circle_orthogonal : orthogonal_circle_through A B)
variable (C_def : ∀ {A B C : Point}, ∃ A', circle A C ∩ circle A B = {A, A'})
variable (B'_def : ∀ {A B C : Point}, ∃ B', circle B A ∩ circle B C = {B, B'})
variable (C'_def : ∀ {A B C : Point}, ∃ C', circle C A ∩ circle C B = {C, C'})

-- Goal
theorem circumradius_of_A'B'C' (h1 : ABC_inradius) (h2 : A_circle_orthogonal)
  (h3 : B_circle_orthogonal) (h4 : C_def) (h5 : B'_def) (h6 : C'_def): circumradius (triangle A' B' C') = r / 2 := by
  sorry

end circumradius_of_A_l276_276589


namespace calculate_fourth_quarter_shots_l276_276182

-- Definitions based on conditions
def first_quarters_shots : ℕ := 20
def first_quarters_successful_shots : ℕ := 12
def third_quarter_shots : ℕ := 10
def overall_accuracy : ℚ := 46 / 100
def total_shots (n : ℕ) : ℕ := first_quarters_shots + third_quarter_shots + n
def total_successful_shots (n : ℕ) : ℚ := first_quarters_successful_shots + 3 + (4 / 10 * n)


-- Main theorem to prove
theorem calculate_fourth_quarter_shots (n : ℕ) (h : (total_successful_shots n) / (total_shots n) = overall_accuracy) : 
  n = 20 :=
by {
  sorry
}

end calculate_fourth_quarter_shots_l276_276182


namespace face_value_stock_l276_276578

-- Given conditions
variables (F : ℝ) (yield quoted_price dividend_rate : ℝ)
variables (h_yield : yield = 20) (h_quoted_price : quoted_price = 125)
variables (h_dividend_rate : dividend_rate = 0.25)

--Theorem to prove the face value of the stock is 100
theorem face_value_stock : (dividend_rate * F / quoted_price) * 100 = yield ↔ F = 100 :=
by
  sorry

end face_value_stock_l276_276578


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276726

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276726


namespace even_divisors_8_factorial_l276_276036

theorem even_divisors_8_factorial : 
  let n := (2^7) * (3^2) * 5 * 7 in
  ∃ (count : ℕ), even_divisors_count n = 84 := 
sorry

end even_divisors_8_factorial_l276_276036


namespace probability_coprime_selected_integers_l276_276339

-- Define the range of integers from 2 to 8
def integers := {i : ℕ | 2 ≤ i ∧ i ≤ 8}

-- Define a function to check if two numbers are coprime
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Statement of the problem in Lean
theorem probability_coprime_selected_integers :
  (∃ (s : finset ℕ) (hs : s ⊆ integers) (h_card : s.card = 2),
    ∃ (a b : ℕ) (ha : a ∈ s) (hb : b ∈ s) (hab : a ≠ b), coprime a b) →
  (finset.card {s : finset (ℕ × ℕ) // s ⊆ (finset.product integers integers) ∧ (∃ x y, (x = s.1) ∧ (y = s.2) ∧ x ≠ y ∧ coprime x y)}.1.to_finset).to_float /
  (finset.card {s : finset (ℕ × ℕ) // s ⊆ (finset.product integers integers) ∧ (∃ x y, (x = s.1) ∧ (y = s.2) ∧ x ≠ y)}.1.to_finset).to_float = 2 / 3 :=
sorry

end probability_coprime_selected_integers_l276_276339


namespace number_of_books_from_second_shop_l276_276864

theorem number_of_books_from_second_shop (books_first_shop : ℕ) (cost_first_shop : ℕ)
    (books_second_shop : ℕ) (cost_second_shop : ℕ) (average_price : ℕ) :
    books_first_shop = 50 →
    cost_first_shop = 1000 →
    cost_second_shop = 800 →
    average_price = 20 →
    average_price * (books_first_shop + books_second_shop) = cost_first_shop + cost_second_shop →
    books_second_shop = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_books_from_second_shop_l276_276864


namespace point_in_third_quadrant_coordinates_l276_276237

theorem point_in_third_quadrant_coordinates :
  ∀ (P : ℝ × ℝ), (P.1 < 0) ∧ (P.2 < 0) ∧ (|P.2| = 2) ∧ (|P.1| = 3) -> P = (-3, -2) :=
by
  intros P h
  sorry

end point_in_third_quadrant_coordinates_l276_276237


namespace paula_remaining_money_l276_276521

-- Definitions based on the conditions
def initialMoney : ℕ := 1000
def shirtCost : ℕ := 45
def pantsCost : ℕ := 85
def jacketCost : ℕ := 120
def shoeCost : ℕ := 95
def jeansOriginalPrice : ℕ := 140
def jeansDiscount : ℕ := 30 / 100  -- 30%

-- Using definitions to compute the spending and remaining money
def totalShirtCost : ℕ := 6 * shirtCost
def totalPantsCost : ℕ := 2 * pantsCost
def totalShoeCost : ℕ := 3 * shoeCost
def jeansDiscountValue : ℕ := jeansDiscount * jeansOriginalPrice
def jeansDiscountedPrice : ℕ := jeansOriginalPrice - jeansDiscountValue
def totalSpent : ℕ := totalShirtCost + totalPantsCost + jacketCost + totalShoeCost
def remainingMoney : ℕ := initialMoney - totalSpent - jeansDiscountedPrice

-- Proof problem statement
theorem paula_remaining_money : remainingMoney = 57 := by
  sorry

end paula_remaining_money_l276_276521


namespace part_I_part_II_l276_276846

noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) - (2 * x) / (x + 2)
noncomputable def g (x : ℝ) : ℝ := f x - (4 / (x + 2))

theorem part_I (x : ℝ) (h₀ : 0 < x) : f x > 0 := sorry

theorem part_II (a : ℝ) (h : ∀ x, g x < x + a) : -2 < a := sorry

end part_I_part_II_l276_276846


namespace no_real_roots_ffx_l276_276905

noncomputable def quadratic_f (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem no_real_roots_ffx (a b c : ℝ) (h : (b - 1)^2 < 4 * a * c) :
  ∀ x : ℝ, quadratic_f a b c (quadratic_f a b c x) ≠ x :=
by
  sorry

end no_real_roots_ffx_l276_276905


namespace largest_three_digit_perfect_square_and_cube_l276_276280

theorem largest_three_digit_perfect_square_and_cube :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a : ℕ), n = a^6) ∧ ∀ (m : ℕ), ((100 ≤ m ∧ m ≤ 999) ∧ (∃ (b : ℕ), m = b^6)) → m ≤ n := 
by 
  sorry

end largest_three_digit_perfect_square_and_cube_l276_276280


namespace least_positive_integer_special_property_l276_276795

theorem least_positive_integer_special_property : ∃ (N : ℕ) (a b c : ℕ), 
  N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 10 * b + c = N / 29 ∧ N = 725 :=
by
  sorry

end least_positive_integer_special_property_l276_276795


namespace num_pos_divisors_36_l276_276961

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l276_276961


namespace vector_MN_l276_276471

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

theorem vector_MN :
  vector_sub N M = (-2, -4) :=
by
  sorry

end vector_MN_l276_276471


namespace monochromatic_triangle_in_K17_l276_276788

theorem monochromatic_triangle_in_K17 :
  ∀ (V : Type) (E : V → V → ℕ), (∀ v1 v2, 0 ≤ E v1 v2 ∧ E v1 v2 < 3) →
    (∃ (v1 v2 v3 : V), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ (E v1 v2 = E v2 v3 ∧ E v2 v3 = E v1 v3)) :=
by
  intro V E Hcl
  sorry

end monochromatic_triangle_in_K17_l276_276788


namespace pages_copied_l276_276670

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l276_276670


namespace prime_p_perfect_cube_l276_276012

theorem prime_p_perfect_cube (p : ℕ) (hp : Nat.Prime p) (h : ∃ n : ℕ, 13 * p + 1 = n^3) :
  p = 2 ∨ p = 211 :=
by
  sorry

end prime_p_perfect_cube_l276_276012


namespace cot_trig_identity_l276_276221

noncomputable def cot (x : Real) : Real :=
  Real.cos x / Real.sin x

theorem cot_trig_identity (a b c α β γ : Real) 
  (habc : a^2 + b^2 = 2021 * c^2) 
  (hα : α = Real.arcsin (a / c)) 
  (hβ : β = Real.arcsin (b / c)) 
  (hγ : γ = Real.arccos ((2021 * c^2 - a^2 - b^2) / (2 * 2021 * c^2))) 
  (h_triangle : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  cot α / (cot β + cot γ) = 1010 :=
by
  sorry

end cot_trig_identity_l276_276221


namespace linear_function_does_not_pass_third_quadrant_l276_276489

/-
Given an inverse proportion function \( y = \frac{a^2 + 1}{x} \), where \( a \) is a constant, and given two points \( (x_1, y_1) \) and \( (x_2, y_2) \) on the same branch of this function, 
with \( b = (x_1 - x_2)(y_1 - y_2) \), prove that the graph of the linear function \( y = bx - b \) does not pass through the third quadrant.
-/

theorem linear_function_does_not_pass_third_quadrant 
  (a x1 x2 : ℝ) 
  (y1 y2 : ℝ)
  (h1 : y1 = (a^2 + 1) / x1) 
  (h2 : y2 = (a^2 + 1) / x2) 
  (h3 : b = (x1 - x2) * (y1 - y2)) : 
  ∃ b, ∀ x y : ℝ, (y = b * x - b) → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) :=
by 
  sorry

end linear_function_does_not_pass_third_quadrant_l276_276489


namespace two_pow_start_digits_l276_276689

theorem two_pow_start_digits (A : ℕ) : 
  ∃ (m n : ℕ), 10^m * A < 2^n ∧ 2^n < 10^m * (A + 1) :=
  sorry

end two_pow_start_digits_l276_276689


namespace smallest_s_for_347_l276_276261

open Nat

theorem smallest_s_for_347 (r s : ℕ) (hr_pos : 0 < r) (hs_pos : 0 < s) 
  (h_rel_prime : Nat.gcd r s = 1) (h_r_lt_s : r < s) 
  (h_contains_347 : ∃ k : ℕ, ∃ y : ℕ, 10 ^ k * r - s * y = 347): 
  s = 653 := 
by sorry

end smallest_s_for_347_l276_276261


namespace polynomial_independent_of_m_l276_276030

theorem polynomial_independent_of_m (m : ℝ) (x : ℝ) (h : 6 * x^2 + (1 - 2 * m) * x + 7 * m = 6 * x^2 + x) : 
  x = 7 / 2 :=
by
  sorry

end polynomial_independent_of_m_l276_276030


namespace was_not_speeding_l276_276754

theorem was_not_speeding (x s : ℝ) (s_obs : ℝ := 26.5) (x_limit : ℝ := 120)
  (brake_dist_eq : s = 0.01 * x + 0.002 * x^2) : s_obs < 30 → x ≤ x_limit :=
sorry

end was_not_speeding_l276_276754


namespace units_digit_of_n_l276_276014

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_of_n 
  (m n : ℕ) 
  (h1 : m * n = 21 ^ 6) 
  (h2 : units_digit m = 7) : 
  units_digit n = 3 := 
sorry

end units_digit_of_n_l276_276014


namespace find_x_l276_276557

def side_of_square_eq_twice_radius_of_larger_circle (s: ℝ) (r_l: ℝ) : Prop :=
  s = 2 * r_l

def radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle (r_l: ℝ) (x: ℝ) (r_s: ℝ) : Prop :=
  r_l = x - (1 / 3) * r_s

def circumference_of_smaller_circle_eq (r_s: ℝ) (circumference: ℝ) : Prop :=
  2 * Real.pi * r_s = circumference

def side_squared_eq_area (s: ℝ) (area: ℝ) : Prop :=
  s^2 = area

noncomputable def value_of_x (r_s r_l: ℝ) : ℝ :=
  14 + 4 / (3 * Real.pi)

theorem find_x 
  (s r_l r_s x: ℝ)
  (h1: side_squared_eq_area s 784)
  (h2: side_of_square_eq_twice_radius_of_larger_circle s r_l)
  (h3: radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle r_l x r_s)
  (h4: circumference_of_smaller_circle_eq r_s 8) :
  x = value_of_x r_s r_l :=
sorry

end find_x_l276_276557


namespace numerical_value_expression_l276_276464

theorem numerical_value_expression (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (ab + 1)) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (ab + 1) = 2 := 
by 
  -- Proof outline provided in the solution section, but actual proof is omitted
  sorry

end numerical_value_expression_l276_276464


namespace prop1_prop2_prop3_l276_276818

variables (a b c d : ℝ)

-- Proposition 1: ab > 0 ∧ bc - ad > 0 → (c/a - d/b > 0)
theorem prop1 (h1 : a * b > 0) (h2 : b * c - a * d > 0) : c / a - d / b > 0 :=
sorry

-- Proposition 2: ab > 0 ∧ (c/a - d/b > 0) → bc - ad > 0
theorem prop2 (h1 : a * b > 0) (h2 : c / a - d / b > 0) : b * c - a * d > 0 :=
sorry

-- Proposition 3: (bc - ad > 0) ∧ (c/a - d/b > 0) → ab > 0
theorem prop3 (h1 : b * c - a * d > 0) (h2 : c / a - d / b > 0) : a * b > 0 :=
sorry

end prop1_prop2_prop3_l276_276818


namespace sum_of_numbers_l276_276282

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := sorry

end sum_of_numbers_l276_276282


namespace num_pos_divisors_36_l276_276980

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l276_276980


namespace percent_decaffeinated_second_batch_l276_276305

theorem percent_decaffeinated_second_batch :
  ∀ (initial_stock : ℝ) (initial_percent : ℝ) (additional_stock : ℝ) (total_percent : ℝ) (second_batch_percent : ℝ),
  initial_stock = 400 →
  initial_percent = 0.20 →
  additional_stock = 100 →
  total_percent = 0.26 →
  (initial_percent * initial_stock + second_batch_percent * additional_stock = total_percent * (initial_stock + additional_stock)) →
  second_batch_percent = 0.50 :=
by
  intros initial_stock initial_percent additional_stock total_percent second_batch_percent
  intros h1 h2 h3 h4 h5
  sorry

end percent_decaffeinated_second_batch_l276_276305


namespace non_zero_digits_of_fraction_l276_276046

def fraction : ℚ := 80 / (2^4 * 5^9)

def decimal_expansion (x : ℚ) : String :=
  -- some function to compute the decimal expansion of a fraction as a string
  "0.00000256" -- placeholder

def non_zero_digits_to_right (s : String) : ℕ :=
  -- some function to count the number of non-zero digits to the right of the decimal point in the string
  3 -- placeholder

theorem non_zero_digits_of_fraction : non_zero_digits_to_right (decimal_expansion fraction) = 3 := by
  sorry

end non_zero_digits_of_fraction_l276_276046


namespace simplest_quadratic_radical_l276_276135

theorem simplest_quadratic_radical :
  ∀ x ∈ {sqrt 3, sqrt 4, sqrt 8, sqrt (1 / 2)}, (x = sqrt 3) :=
by
  sorry

end simplest_quadratic_radical_l276_276135


namespace bed_length_l276_276309

noncomputable def volume (length width height : ℝ) : ℝ :=
  length * width * height

theorem bed_length
  (width height : ℝ)
  (bags_of_soil soil_volume_per_bag total_volume : ℝ)
  (needed_bags : ℝ)
  (L : ℝ) :
  width = 4 →
  height = 1 →
  needed_bags = 16 →
  soil_volume_per_bag = 4 →
  total_volume = needed_bags * soil_volume_per_bag →
  total_volume = 2 * volume L width height →
  L = 8 :=
by
  intros
  sorry

end bed_length_l276_276309


namespace eccentricity_of_ellipse_l276_276194

noncomputable def ellipse_eccentricity (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse:
  ∀ (a b : ℝ) (c : ℝ), 
    0 < b ∧ b < a ∧ a = 3 * c → 
    ellipse_eccentricity a b c = 1/3 := by
  intros a b c h
  let e := ellipse_eccentricity a b c
  have h1 : 0 < b := h.1
  have h2 : b < a := h.2.left
  have h3 : a = 3 * c := h.2.right
  simp [ellipse_eccentricity, h3]
  sorry

end eccentricity_of_ellipse_l276_276194


namespace determine_b_div_a_l276_276019

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem determine_b_div_a
  (a b : ℝ)
  (hf_deriv : ∀ x : ℝ, (deriv (f a b)) x = 3 * x^2 + 2 * a * x + b)
  (hf_max : f a b 1 = 10)
  (hf_deriv_at_1 : (deriv (f a b)) 1 = 0) :
  b / a = -3 / 2 :=
sorry

end determine_b_div_a_l276_276019


namespace correct_factorization_A_l276_276283

-- Define the polynomial expressions
def expression_A : Prop :=
  (x : ℝ) → x^2 - x - 6 = (x + 2) * (x - 3)

def expression_B : Prop :=
  (x : ℝ) → x^2 - 1 = x * (x - 1 / x)

def expression_C : Prop :=
  (x y : ℝ) → 7 * x^2 * y^5 = x * y * 7 * x * y^4

def expression_D : Prop :=
  (x : ℝ) → x^2 + 4 * x + 4 = x * (x + 4) + 4

-- The correct factorization from left to right
theorem correct_factorization_A : expression_A := 
by 
  -- Proof omitted
  sorry

end correct_factorization_A_l276_276283


namespace scrambled_eggs_count_l276_276505

-- Definitions based on the given conditions
def num_sausages := 3
def time_per_sausage := 5
def time_per_egg := 4
def total_time := 39

-- Prove that Kira scrambled 6 eggs
theorem scrambled_eggs_count : (total_time - num_sausages * time_per_sausage) / time_per_egg = 6 := by
  sorry

end scrambled_eggs_count_l276_276505


namespace total_money_l276_276821

theorem total_money 
  (n_pennies n_nickels n_dimes n_quarters n_half_dollars : ℝ) 
  (h_pennies : n_pennies = 9) 
  (h_nickels : n_nickels = 4) 
  (h_dimes : n_dimes = 3) 
  (h_quarters : n_quarters = 7) 
  (h_half_dollars : n_half_dollars = 5) : 
  0.01 * n_pennies + 0.05 * n_nickels + 0.10 * n_dimes + 0.25 * n_quarters + 0.50 * n_half_dollars = 4.84 :=
by 
  sorry

end total_money_l276_276821


namespace find_FC_l276_276802

theorem find_FC
  (DC : ℝ) (CB : ℝ) (AD : ℝ)
  (hDC : DC = 9) (hCB : CB = 10)
  (hAB : ∃ (k1 : ℝ), k1 = 1/5 ∧ AB = k1 * AD)
  (hED : ∃ (k2 : ℝ), k2 = 3/4 ∧ ED = k2 * AD) :
  ∃ FC : ℝ, FC = 11.025 :=
by
  sorry

end find_FC_l276_276802


namespace simplify_expression_l276_276531

theorem simplify_expression(x : ℝ) : 2 * x * (4 * x^2 - 3 * x + 1) - 7 * (2 * x^2 - 3 * x + 4) = 8 * x^3 - 20 * x^2 + 23 * x - 28 :=
by
  sorry

end simplify_expression_l276_276531


namespace find_second_half_profit_l276_276300

variable (P : ℝ)
variable (profit_difference total_annual_profit : ℝ)
variable (h_difference : profit_difference = 2750000)
variable (h_total : total_annual_profit = 3635000)

theorem find_second_half_profit (h_eq : P + (P + profit_difference) = total_annual_profit) : 
  P = 442500 :=
by
  rw [h_difference, h_total] at h_eq
  sorry

end find_second_half_profit_l276_276300


namespace ratio_of_ages_ten_years_ago_l276_276126

theorem ratio_of_ages_ten_years_ago (A T : ℕ) 
    (h1: A = 30) 
    (h2: T = A - 15) : 
    (A - 10) / (T - 10) = 4 :=
by
  sorry

end ratio_of_ages_ten_years_ago_l276_276126


namespace wizard_collection_value_l276_276440

theorem wizard_collection_value :
  let crystal_ball := nat.of_digits 7 [3, 4, 2, 6]
  let wand := nat.of_digits 7 [0, 5, 6, 1]
  let book_of_spells := nat.of_digits 7 [2, 0, 2]
  crystal_ball + wand + book_of_spells = 2959 :=
by
  let crystal_ball := nat.of_digits 7 [3, 4, 2, 6]
  let wand := nat.of_digits 7 [0, 5, 6, 1]
  let book_of_spells := nat.of_digits 7 [2, 0, 2]
  sorry

end wizard_collection_value_l276_276440


namespace even_divisors_of_8fac_l276_276040

theorem even_divisors_of_8fac : 
  let num_even_divisors := ∏ x in {a | 1 ≤ a ∧ a ≤ 7}.card * 
                                      {b | 0 ≤ b ∧ b ≤ 2}.card *
                                      {c | 0 ≤ c ∧ c ≤ 1}.card *
                                      {d | 0 ≤ d ∧ d ≤ 1}.card
  in num_even_divisors = 84 := by
  sorry

end even_divisors_of_8fac_l276_276040


namespace share_of_y_l276_276156

theorem share_of_y (A y z : ℝ)
  (hx : y = 0.45 * A)
  (hz : z = 0.30 * A)
  (h_total : A + y + z = 140) :
  y = 36 := by
  sorry

end share_of_y_l276_276156


namespace div_c_a_l276_276826

theorem div_c_a (a b c : ℚ) (h1 : a = 3 * b) (h2 : b = 2 / 5 * c) :
  c / a = 5 / 6 := 
by
  sorry

end div_c_a_l276_276826


namespace geom_sequence_sum_l276_276497

theorem geom_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, a n > 0)
  (h_geom : ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q)
  (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 :=
sorry

end geom_sequence_sum_l276_276497


namespace largest_crate_dimension_l276_276430

def largest_dimension_of_crate : ℝ := 10

theorem largest_crate_dimension (length width : ℝ) (r : ℝ) (h : ℝ) 
  (h_length : length = 5) (h_width : width = 8) (h_radius : r = 5) (h_height : h >= 10) :
  h = largest_dimension_of_crate :=
by 
  sorry

end largest_crate_dimension_l276_276430


namespace alpha_div_3_range_l276_276617

theorem alpha_div_3_range (α : ℝ) (k : ℤ) 
  (h1 : Real.sin α > 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k : ℤ, (2 * k * Real.pi + Real.pi / 4 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi / 3) ∨ 
            (2 * k * Real.pi + 5 * Real.pi / 6 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi) :=
sorry

end alpha_div_3_range_l276_276617


namespace correct_scientific_notation_representation_l276_276590

-- Defining the given number of visitors in millions
def visitors_in_millions : Float := 8.0327
-- Converting this number to an integer and expressing in scientific notation
def rounded_scientific_notation (num : Float) : String :=
  if num == 8.0327 then "8.0 × 10^6" else "incorrect"

-- The mathematical proof statement
theorem correct_scientific_notation_representation :
  rounded_scientific_notation visitors_in_millions = "8.0 × 10^6" :=
by
  sorry

end correct_scientific_notation_representation_l276_276590


namespace arithmetic_and_geometric_mean_l276_276865

theorem arithmetic_and_geometric_mean (x y : ℝ) (h₁ : (x + y) / 2 = 20) (h₂ : Real.sqrt (x * y) = Real.sqrt 150) : x^2 + y^2 = 1300 :=
by
  sorry

end arithmetic_and_geometric_mean_l276_276865


namespace probability_of_winning_game_l276_276361

theorem probability_of_winning_game :
  let wheel1 := {1, 2, 3, 4, 5, 6}
  let wheel2 := {1, 1, 2, 2}
  let isWinning (x y : ℕ) := x + y < 5
  let totalOutcomes := (Finset.card wheel1) * (Finset.card wheel2)
  let winningOutcomes := finset.card ((finset.product wheel1 wheel2).filter (λ (p : ℕ × ℕ), isWinning p.1 p.2))
  winningOutcomes.toR / totalOutcomes.toR = 1 / 3 := by sorry

end probability_of_winning_game_l276_276361


namespace closest_point_to_origin_on_graph_l276_276084

theorem closest_point_to_origin_on_graph :
  ∃ x : ℝ, x > 0 ∧ (y = x + 1/x ∧ (x, y) = (1/real.root 4 2, (1 + real.sqrt 2)/real.root 4 2)) := sorry

end closest_point_to_origin_on_graph_l276_276084


namespace max_value_of_xy_l276_276536

theorem max_value_of_xy (x y : ℝ) (h₁ : x + y = 40) (h₂ : x > 0) (h₃ : y > 0) : xy ≤ 400 :=
sorry

end max_value_of_xy_l276_276536


namespace grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l276_276775

def can_jump (x : Int) : Prop :=
  ∃ (k m : Int), x = k * 36 + m * 14

theorem grasshopper_cannot_move_3_cm :
  ¬ can_jump 3 :=
by
  sorry

theorem grasshopper_can_move_2_cm :
  can_jump 2 :=
by
  sorry

theorem grasshopper_can_move_1234_cm :
  can_jump 1234 :=
by
  sorry

end grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l276_276775


namespace math_proof_problem_l276_276290

open Set

noncomputable def alpha : ℝ := (3 - Real.sqrt 5) / 2

theorem math_proof_problem (α_pos : 0 < α) (α_lt_delta : α < alpha) :
  ∃ n p : ℕ, p > α * 2^n ∧ ∃ S T : Finset (Fin n) → Finset (Fin n), (∀ i j, (S i) ∩ (T j) ≠ ∅) :=
  sorry

end math_proof_problem_l276_276290


namespace find_positive_integer_l276_276289

theorem find_positive_integer (x : ℕ) (h1 : (10 * x + 4) % (x + 4) = 0) (h2 : (10 * x + 4) / (x + 4) = x - 23) : x = 32 :=
by
  sorry

end find_positive_integer_l276_276289


namespace range_of_p_l276_276632

def A (x : ℝ) : Prop := -2 < x ∧ x < 5
def B (p : ℝ) (x : ℝ) : Prop := p + 1 < x ∧ x < 2 * p - 1

theorem range_of_p (p : ℝ) :
  (∀ x, A x ∨ B p x → A x) ↔ p ≤ 3 :=
by
  sorry

end range_of_p_l276_276632


namespace largest_number_in_set_l276_276920

theorem largest_number_in_set :
  ∀ (a b c d : ℤ), (a ∈ [0, 2, -1, -2]) → (b ∈ [0, 2, -1, -2]) → (c ∈ [0, 2, -1, -2]) → (d ∈ [0, 2, -1, -2])
  → (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  → max (max a b) (max c d) = 2
  := 
by
  sorry

end largest_number_in_set_l276_276920


namespace num_pos_divisors_36_l276_276960

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l276_276960


namespace max_area_of_rectangle_with_perimeter_40_l276_276949

theorem max_area_of_rectangle_with_perimeter_40 :
  ∃ (A : ℝ), (A = 100) ∧ (∀ (length width : ℝ), 2 * (length + width) = 40 → length * width ≤ A) :=
by
  sorry

end max_area_of_rectangle_with_perimeter_40_l276_276949


namespace maximize_profit_l276_276146

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

theorem maximize_profit :
  let max_price := 14
  let max_profit := 360
  (∀ x > 10, profit x ≤ profit max_price) ∧ profit max_price = max_profit :=
by
  let max_price := 14
  let max_profit := 360
  sorry

end maximize_profit_l276_276146


namespace min_handshakes_l276_276764

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l276_276764


namespace total_bricks_used_l276_276393

-- Definitions for conditions
def num_courses_per_wall : Nat := 10
def num_bricks_per_course : Nat := 20
def num_complete_walls : Nat := 5
def incomplete_wall_missing_courses : Nat := 3

-- Lean statement to prove the mathematically equivalent problem
theorem total_bricks_used : 
  (num_complete_walls * (num_courses_per_wall * num_bricks_per_course) + 
  ((num_courses_per_wall - incomplete_wall_missing_courses) * num_bricks_per_course)) = 1140 :=
by
  sorry

end total_bricks_used_l276_276393


namespace loss_percentage_is_nine_percent_l276_276143

theorem loss_percentage_is_nine_percent
    (C S : ℝ)
    (h1 : 15 * C = 20 * S)
    (discount_rate : ℝ := 0.10)
    (tax_rate : ℝ := 0.08) :
    (((0.9 * C) - (1.08 * S)) / C) * 100 = 9 :=
by
  sorry

end loss_percentage_is_nine_percent_l276_276143


namespace odd_expression_is_odd_l276_276097

theorem odd_expression_is_odd (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : (4 * p * q + 1) % 2 = 1 :=
sorry

end odd_expression_is_odd_l276_276097


namespace unique_triple_solution_l276_276938

theorem unique_triple_solution (x y z : ℝ) :
  (1 + x^4 ≤ 2 * (y - z)^2) →
  (1 + y^4 ≤ 2 * (z - x)^2) →
  (1 + z^4 ≤ 2 * (x - y)^2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
sorry

end unique_triple_solution_l276_276938


namespace a6_is_32_l276_276622

namespace arithmetic_sequence

variables {a : ℕ → ℝ} -- {aₙ} is an arithmetic sequence with positive terms
variables (q : ℝ) -- Common ratio

-- Conditions as definitions
def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a1_is_1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2_times_a4_is_16 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 4 = 16

-- The ultimate goal is to prove a₆ = 32
theorem a6_is_32 (h_arith : is_arithmetic_sequence a q) 
  (h_a1 : a1_is_1 a) (h_product : a2_times_a4_is_16 a q) : 
  a 6 = 32 := 
sorry

end arithmetic_sequence

end a6_is_32_l276_276622


namespace cost_price_of_product_l276_276579

theorem cost_price_of_product (x y : ℝ)
  (h1 : 0.8 * y - x = 120)
  (h2 : 0.6 * y - x = -20) :
  x = 440 := sorry

end cost_price_of_product_l276_276579


namespace max_sum_e3_f3_g3_h3_i3_l276_276845

theorem max_sum_e3_f3_g3_h3_i3 (e f g h i : ℝ) (h_cond : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  e^3 + f^3 + g^3 + h^3 + i^3 ≤ 5^(3/4) :=
sorry

end max_sum_e3_f3_g3_h3_i3_l276_276845


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276723

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276723


namespace value_of_a_l276_276359

theorem value_of_a (a : ℝ) (A : Set ℝ) (h : ∀ x, x ∈ A ↔ |x - a| < 1) : A = Set.Ioo 1 3 → a = 2 :=
by
  intro ha
  have : Set.Ioo 1 3 = {x | ∃ y, y ∈ Set.Ioi (1 : ℝ) ∧ y ∈ Set.Iio (3 : ℝ)} := by sorry
  sorry

end value_of_a_l276_276359


namespace intersection_A_B_l276_276202

def is_log2 (y x : ℝ) : Prop := y = Real.log x / Real.log 2

def set_A (y : ℝ) : Set ℝ := { x | ∃ y, is_log2 y x}
def set_B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_A_B : (set_A 1) ∩ set_B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_A_B_l276_276202


namespace value_2_std_dev_less_than_mean_l276_276288

-- Define the mean and standard deviation as constants
def mean : ℝ := 14.5
def std_dev : ℝ := 1.5

-- State the theorem (problem)
theorem value_2_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.5 := by
  sorry

end value_2_std_dev_less_than_mean_l276_276288


namespace largest_divisible_by_6_ending_in_4_l276_276720

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l276_276720


namespace geometric_sequence_ratio_l276_276619

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

-- Definitions based on given conditions
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement
theorem geometric_sequence_ratio :
  is_geometric_seq a q →
  q = -1/3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros
  sorry

end geometric_sequence_ratio_l276_276619


namespace rope_length_after_100_cuts_l276_276307

noncomputable def rope_cut (initial_length : ℝ) (num_cuts : ℕ) (cut_fraction : ℝ) : ℝ :=
  initial_length * (1 - cut_fraction) ^ num_cuts

theorem rope_length_after_100_cuts :
  rope_cut 1 100 (3 / 4) = (1 / 4) ^ 100 :=
by
  sorry

end rope_length_after_100_cuts_l276_276307


namespace value_of_m_l276_276223

noncomputable def has_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

noncomputable def has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

theorem value_of_m (m : ℝ) :
  (has_distinct_real_roots 1 m 1 ∧ has_no_real_roots 4 (4 * (m + 2)) 1) ↔ (-3 < m ∧ m < -2) :=
by
  sorry

end value_of_m_l276_276223


namespace point_B_coordinates_l276_276522

variable (A : ℝ × ℝ)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

theorem point_B_coordinates : 
  (move_left (move_up (-3, -5) 4) 3) = (-6, -1) :=
by
  sorry

end point_B_coordinates_l276_276522


namespace total_num_birds_l276_276673

-- Definitions for conditions
def num_crows := 30
def percent_more_hawks := 0.60

-- Theorem to prove the total number of birds
theorem total_num_birds : num_crows + num_crows * percent_more_hawks + num_crows = 78 := 
sorry

end total_num_birds_l276_276673


namespace even_divisors_of_8fac_l276_276041

theorem even_divisors_of_8fac : 
  let num_even_divisors := ∏ x in {a | 1 ≤ a ∧ a ≤ 7}.card * 
                                      {b | 0 ≤ b ∧ b ≤ 2}.card *
                                      {c | 0 ≤ c ∧ c ≤ 1}.card *
                                      {d | 0 ≤ d ∧ d ≤ 1}.card
  in num_even_divisors = 84 := by
  sorry

end even_divisors_of_8fac_l276_276041


namespace sum_of_roots_l276_276013

theorem sum_of_roots : ∀ x : ℝ, x^2 - 2004 * x + 2021 = 0 → x = 2004 := by
  sorry

end sum_of_roots_l276_276013


namespace determine_q_l276_276103

-- Lean 4 statement
theorem determine_q (a : ℝ) (q : ℝ → ℝ) :
  (∀ x, q x = a * (x + 2) * (x - 3)) ∧ q 1 = 8 →
  q x = - (4 / 3) * x ^ 2 + (4 / 3) * x + 8 := 
sorry

end determine_q_l276_276103


namespace odd_function_behavior_l276_276474

variable {f : ℝ → ℝ}

theorem odd_function_behavior (h1 : ∀ x : ℝ, f (-x) = -f x) 
                             (h2 : ∀ x : ℝ, 0 < x → f x = x * (1 + x)) 
                             (x : ℝ)
                             (hx : x < 0) : 
  f x = x * (1 - x) :=
by
  -- Insert proof here
  sorry

end odd_function_behavior_l276_276474


namespace infinite_series_converges_l276_276451

open BigOperators

noncomputable def problem : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0

theorem infinite_series_converges : problem = 61 / 24 :=
sorry

end infinite_series_converges_l276_276451


namespace cost_of_each_shirt_l276_276943

theorem cost_of_each_shirt
  (x : ℝ) 
  (h : 3 * x + 2 * 20 = 85) : x = 15 :=
sorry

end cost_of_each_shirt_l276_276943


namespace area_of_ABCD_l276_276999

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l276_276999


namespace find_volume_of_12_percent_solution_l276_276756

variable (x y : ℝ)

theorem find_volume_of_12_percent_solution
  (h1 : x + y = 60)
  (h2 : 0.02 * x + 0.12 * y = 3) :
  y = 18 := 
sorry

end find_volume_of_12_percent_solution_l276_276756


namespace repeating_decimal_sum_l276_276005

noncomputable def repeating_decimal_0_3 : ℚ := 1 / 3
noncomputable def repeating_decimal_0_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_0_2 : ℚ := 2 / 9

theorem repeating_decimal_sum :
  repeating_decimal_0_3 + repeating_decimal_0_6 - repeating_decimal_0_2 = 7 / 9 :=
by
  sorry

end repeating_decimal_sum_l276_276005


namespace find_positive_integers_l276_276793

theorem find_positive_integers
  (a b c : ℕ) 
  (h : a ≥ b ∧ b ≥ c ∧ a ≥ c)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) :
  (1 + 1 / (a : ℚ)) * (1 + 1 / (b : ℚ)) * (1 + 1 / (c : ℚ)) = 2 →
  (a, b, c) ∈ [(15, 4, 2), (9, 5, 2), (7, 6, 2), (8, 3, 3), (5, 4, 3)] :=
by
  sorry

end find_positive_integers_l276_276793


namespace track_circumference_l276_276757

theorem track_circumference (x : ℕ) 
  (A_B_uniform_speeds_opposite : True) 
  (diametrically_opposite_start : True) 
  (same_start_time : True) 
  (first_meeting_B_150_yards : True) 
  (second_meeting_A_90_yards_before_complete_lap : True) : 
  2 * x = 720 :=
by
  sorry

end track_circumference_l276_276757


namespace cos_sin_sum_l276_276355

open Real

theorem cos_sin_sum (α : ℝ) (h : (cos (2 * α)) / (sin (α - π / 4)) = -sqrt 2 / 2) : cos α + sin α = 1 / 2 := by
  sorry

end cos_sin_sum_l276_276355


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276701

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276701


namespace find_r_l276_276161

noncomputable def parabola_vertex : (ℝ × ℝ) := (0, -1)

noncomputable def intersection_points (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (r - Real.sqrt (r^2 + 4)) / 2
  let y1 := r * x1
  let x2 := (r + Real.sqrt (r^2 + 4)) / 2
  let y2 := r * x2
  ((x1, y1), (x2, y2))

noncomputable def triangle_area (r : ℝ) : ℝ :=
  let base := Real.sqrt (r^2 + 4)
  let height := 2
  1/2 * base * height

theorem find_r (r : ℝ) (h : r > 0) : triangle_area r = 32 → r = Real.sqrt 1020 := 
by
  sorry

end find_r_l276_276161


namespace correct_exponent_operation_l276_276572

theorem correct_exponent_operation (a b : ℝ) : 
  a^2 * a^3 = a^5 := 
by sorry

end correct_exponent_operation_l276_276572


namespace value_of_expression_l276_276991

theorem value_of_expression (x y : ℤ) (h1 : x = 1) (h2 : y = 630) : 
  2019 * x - 3 * y - 9 = 120 := 
by
  sorry

end value_of_expression_l276_276991


namespace sum_of_three_digit_numbers_l276_276614

theorem sum_of_three_digit_numbers :
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  let Sum := n / 2 * (first_term + last_term)
  Sum = 494550 :=
by {
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  have n_def : n = 900 := by norm_num [n]
  let Sum := n / 2 * (first_term + last_term)
  have sum_def : Sum = 450 * (100 + 999) := by norm_num [Sum, first_term, last_term, n_def]
  have final_sum : Sum = 494550 := by norm_num [sum_def]
  exact final_sum
}

end sum_of_three_digit_numbers_l276_276614


namespace zeros_in_square_of_nines_l276_276806

def num_zeros (n : ℕ) (m : ℕ) : ℕ :=
  -- Count the number of zeros in the decimal representation of m
sorry

theorem zeros_in_square_of_nines :
  num_zeros 6 ((10^6 - 1)^2) = 5 :=
sorry

end zeros_in_square_of_nines_l276_276806


namespace value_of_fraction_l276_276823

theorem value_of_fraction (x y : ℤ) (h1 : x = 3) (h2 : y = 4) : (x^5 + 3 * y^3) / 9 = 48 :=
by
  sorry

end value_of_fraction_l276_276823


namespace smallest_integer_value_l276_276931

theorem smallest_integer_value (n : ℤ) : ∃ (n : ℤ), n = 5 ∧ n^2 - 11*n + 28 < 0 :=
by
  use 5
  sorry

end smallest_integer_value_l276_276931


namespace martin_and_martina_ages_l276_276226

-- Conditions
def martin_statement (x y : ℕ) : Prop := x = 3 * (2 * y - x)
def martina_statement (x y : ℕ) : Prop := 3 * x - y = 77

-- Proof problem
theorem martin_and_martina_ages :
  ∃ (x y : ℕ), martin_statement x y ∧ martina_statement x y ∧ x = 33 ∧ y = 22 :=
by {
  -- No proof required, just the statement
  sorry
}

end martin_and_martina_ages_l276_276226


namespace total_books_sold_l276_276231

theorem total_books_sold (tuesday_books wednesday_books thursday_books : Nat) 
  (h1 : tuesday_books = 7) 
  (h2 : wednesday_books = 3 * tuesday_books) 
  (h3 : thursday_books = 3 * wednesday_books) : 
  tuesday_books + wednesday_books + thursday_books = 91 := 
by 
  sorry

end total_books_sold_l276_276231


namespace find_three_numbers_l276_276276

theorem find_three_numbers :
  ∃ (a₁ a₄ a₂₅ : ℕ), a₁ + a₄ + a₂₅ = 114 ∧
    ( ∃ r ≠ 1, a₄ = a₁ * r ∧ a₂₅ = a₄ * r * r ) ∧
    ( ∃ d, a₄ = a₁ + 3 * d ∧ a₂₅ = a₁ + 24 * d ) ∧
    a₁ = 2 ∧ a₄ = 14 ∧ a₂₅ = 98 :=
by
  sorry

end find_three_numbers_l276_276276


namespace sum_central_square_l276_276830

noncomputable def table_sum : ℕ := 10200
noncomputable def a : ℕ := 1200
noncomputable def central_sum : ℕ := 720

theorem sum_central_square :
  ∃ (a : ℕ), table_sum = a * (1 + (1 / 3) + (1 / 9) + (1 / 27)) * (1 + (1 / 4) + (1 / 16) + (1 / 64)) ∧ 
              central_sum = (a / 3) + (a / 12) + (a / 9) + (a / 36) :=
by
  sorry

end sum_central_square_l276_276830


namespace campers_in_two_classes_l276_276313

-- Definitions of the sets and conditions
variable (S A R : Finset ℕ)
variable (n : ℕ)
variable (x : ℕ)

-- Given conditions
axiom hyp1 : S.card = 20
axiom hyp2 : A.card = 20
axiom hyp3 : R.card = 20
axiom hyp4 : (S ∩ A ∩ R).card = 4
axiom hyp5 : (S \ (A ∪ R)).card + (A \ (S ∪ R)).card + (R \ (S ∪ A)).card = 24

-- The hypothesis that n = |S ∪ A ∪ R|
axiom hyp6 : n = (S ∪ A ∪ R).card

-- Statement to be proven in Lean
theorem campers_in_two_classes : x = 12 :=
by
  sorry

end campers_in_two_classes_l276_276313


namespace find_c_l276_276209

def f (x : ℤ) : ℤ := x - 2

def F (x y : ℤ) : ℤ := y^2 + x

theorem find_c : ∃ c, c = F 3 (f 16) ∧ c = 199 :=
by
  use F 3 (f 16)
  sorry

end find_c_l276_276209


namespace largest_variable_l276_276956

theorem largest_variable {x y z w : ℤ} 
  (h1 : x + 3 = y - 4)
  (h2 : x + 3 = z + 2)
  (h3 : x + 3 = w - 1) :
  y > x ∧ y > z ∧ y > w :=
by sorry

end largest_variable_l276_276956


namespace all_items_weight_is_8040_l276_276849

def weight_of_all_items : Real :=
  let num_tables := 15
  let settings_per_table := 8
  let backup_percentage := 0.25

  let weight_fork := 3.5
  let weight_knife := 4.0
  let weight_spoon := 4.5
  let weight_large_plate := 14.0
  let weight_small_plate := 10.0
  let weight_wine_glass := 7.0
  let weight_water_glass := 9.0
  let weight_table_decoration := 16.0

  let total_settings := (num_tables * settings_per_table) * (1 + backup_percentage)
  let weight_per_setting := (weight_fork + weight_knife + weight_spoon) + (weight_large_plate + weight_small_plate) + (weight_wine_glass + weight_water_glass)
  let total_weight_decorations := num_tables * weight_table_decoration

  let total_weight := total_settings * weight_per_setting + total_weight_decorations
  total_weight

theorem all_items_weight_is_8040 :
  weight_of_all_items = 8040 := sorry

end all_items_weight_is_8040_l276_276849


namespace find_r_k_l276_276881

theorem find_r_k :
  ∃ r k : ℚ, (∀ t : ℚ, (∃ x y : ℚ, (x = r + 3 * t ∧ y = 2 + k * t) ∧ y = 5 * x - 7)) ∧ 
            r = 9 / 5 ∧ k = -4 :=
by {
  sorry
}

end find_r_k_l276_276881


namespace independence_iff_expectation_condition_l276_276092

open MeasureTheory

variable {α : Type*} {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ξ : Ω → α) (𝒢 : MeasurableSpace Ω)
variable (g : α → ℝ)

def is_independent (ξ : Ω → α) (𝒢 : MeasurableSpace Ω) : Prop :=
∀ A ∈ 𝒢.sets, ∀ B ∈ borel α, P (A ∩ ξ ⁻¹' B) = P A * P (ξ ⁻¹' B)

def expectation_condition (ξ : Ω → α) (𝒢 : MeasurableSpace Ω) (g : α → ℝ) : Prop :=
integrable g (pmap ξ 𝒢) ∧ 
∀ B ∈ borel α, ∀ A ∈ 𝒢.sets, 
integral A (g ∘ ξ) = P A * integral (g ∘ ξ)

theorem independence_iff_expectation_condition (ξ : Ω → α) (𝒢 : MeasurableSpace Ω) :
is_independent ξ 𝒢 ↔ expectation_condition ξ 𝒢 g :=
sorry

end independence_iff_expectation_condition_l276_276092


namespace number_of_white_balls_l276_276060

theorem number_of_white_balls (total_balls : ℕ) (red_prob black_prob : ℝ)
  (h_total : total_balls = 50)
  (h_red_prob : red_prob = 0.15)
  (h_black_prob : black_prob = 0.45) :
  ∃ (white_balls : ℕ), white_balls = 20 :=
by
  sorry

end number_of_white_balls_l276_276060


namespace solve_inequalities_l276_276249

/-- Solve the inequality system and find all non-negative integer solutions. -/
theorem solve_inequalities :
  { x : ℤ | 0 ≤ x ∧ 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {0, 1, 2} :=
by
  sorry

end solve_inequalities_l276_276249


namespace simplify_expression_l276_276736

-- Define the given expressions
def numerator : ℕ := 5^5 + 5^3 + 5
def denominator : ℕ := 5^4 - 2 * 5^2 + 5

-- Define the simplified fraction
def simplified_fraction : ℚ := numerator / denominator

-- Prove that the simplified fraction is equivalent to 651 / 116
theorem simplify_expression : simplified_fraction = 651 / 116 := by
  sorry

end simplify_expression_l276_276736


namespace find_constants_l276_276941

theorem find_constants :
  ∃ A B C D : ℚ,
    (∀ x : ℚ,
      x ≠ 2 → x ≠ 3 → x ≠ 5 → x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1)) ∧
  A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 :=
by
  sorry

end find_constants_l276_276941


namespace neg_all_cups_full_l276_276265

variable (x : Type) (cup : x → Prop) (full : x → Prop)

theorem neg_all_cups_full :
  ¬ (∀ x, cup x → full x) = ∃ x, cup x ∧ ¬ full x := by
sorry

end neg_all_cups_full_l276_276265


namespace min_max_value_expression_l276_276844

theorem min_max_value_expression
  (x1 x2 x3 : ℝ) 
  (hx : x1 + x2 + x3 = 1)
  (hx1 : 0 ≤ x1)
  (hx2 : 0 ≤ x2)
  (hx3 : 0 ≤ x3) :
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5) = 1 := 
sorry

end min_max_value_expression_l276_276844


namespace find_k_for_min_value_zero_l276_276890

theorem find_k_for_min_value_zero :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0 ∧
                         ∃ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) →
  k = 3 / 2 :=
sorry

end find_k_for_min_value_zero_l276_276890


namespace find_ab_pairs_l276_276609

open Set

-- Definitions
def f (a b x : ℝ) : ℝ := a * x + b

-- Main theorem
theorem find_ab_pairs (a b : ℝ) :
  (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → 
    f a b x * f a b y + f a b (x + y - x * y) ≤ 0) ↔ 
  (-1 ≤ b ∧ b ≤ 0 ∧ -(b + 1) ≤ a ∧ a ≤ -b) :=
by sorry

end find_ab_pairs_l276_276609


namespace bowling_tournament_l276_276869

def num_possible_orders : ℕ := 32

theorem bowling_tournament : num_possible_orders = 2 * 2 * 2 * 2 * 2 := by
  -- The structure of the playoff with 2 choices per match until all matches are played,
  -- leading to a total of 5 rounds and 2 choices per round, hence 2^5 = 32.
  sorry

end bowling_tournament_l276_276869


namespace calculation_result_l276_276549

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by
  sorry

end calculation_result_l276_276549


namespace simple_interest_years_l276_276121

theorem simple_interest_years (r1 r2 t2 P1 P2 S : ℝ) (hP1: P1 = 3225) (hP2: P2 = 8000) (hr1: r1 = 0.08) (hr2: r2 = 0.15) (ht2: t2 = 2) (hCI : S = 2580) :
    S / 2 = (P1 * r1 * t) / 100 → t = 5 :=
by
  sorry

end simple_interest_years_l276_276121


namespace sqrt_of_16_is_4_l276_276552

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 :=
sorry

end sqrt_of_16_is_4_l276_276552


namespace winding_clock_available_time_l276_276891

theorem winding_clock_available_time
    (minute_hand_restriction_interval: ℕ := 5) -- Each interval the minute hand restricts
    (hour_hand_restriction_interval: ℕ := 60) -- Each interval the hour hand restricts
    (intervals_per_12_hours: ℕ := 2) -- Number of restricted intervals in each 12-hour cycle
    (minutes_in_day: ℕ := 24 * 60) -- Total minutes in 24 hours
    : (minutes_in_day - ((minute_hand_restriction_interval * intervals_per_12_hours * 12) + 
                         (hour_hand_restriction_interval * intervals_per_12_hours * 2))) = 1080 :=
by
  -- Skipping the proof steps
  sorry

end winding_clock_available_time_l276_276891


namespace problem1_problem2_problem3_problem4_l276_276558

-- Question 1
theorem problem1 (a b : ℝ) (h : 5 * a + 3 * b = -4) : 2 * (a + b) + 4 * (2 * a + b) = -8 :=
by
  sorry

-- Question 2
theorem problem2 (a : ℝ) (h : a^2 + a = 3) : 2 * a^2 + 2 * a + 2023 = 2029 :=
by
  sorry

-- Question 3
theorem problem3 (a b : ℝ) (h : a - 2 * b = -3) : 3 * (a - b) - 7 * a + 11 * b + 2 = 14 :=
by
  sorry

-- Question 4
theorem problem4 (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) 
  (h2 : a * b - 2 * b^2 = -3) : a^2 + a * b + 2 * b^2 = -2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l276_276558


namespace sum_of_coefficients_l276_276551

theorem sum_of_coefficients (x : ℝ) : 
  (1 - 2 * x) ^ 10 = 1 :=
sorry

end sum_of_coefficients_l276_276551


namespace product_of_undefined_x_l276_276187

-- Define the quadratic equation condition
def quad_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The main theorem to prove the product of all x such that the expression is undefined
theorem product_of_undefined_x :
  (∃ x₁ x₂ : ℝ, quad_eq 1 4 3 x₁ ∧ quad_eq 1 4 3 x₂ ∧ x₁ * x₂ = 3) :=
by
  sorry

end product_of_undefined_x_l276_276187


namespace fraction_evaluation_l276_276932

theorem fraction_evaluation : (1 - (1 / 4)) / (1 - (1 / 3)) = (9 / 8) :=
by
  sorry

end fraction_evaluation_l276_276932


namespace divisible_expressions_l276_276240

theorem divisible_expressions (x y : ℤ) (n : ℤ) 
  (h : 2 * x + 3 * y = 17 * n) : 17 ∣ (9 * x + 5 * y) :=
begin
  -- proof goes here
  sorry
end

end divisible_expressions_l276_276240


namespace simplify_and_evaluate_expression_l276_276530

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 / (x + 2) + x - 2) / ((x^2 - 2*x + 1) / (x + 2))

theorem simplify_and_evaluate_expression (x : ℝ) (hx : |x| = 2) (h_ne : x ≠ -2) :
  given_expression x = 3 :=
by
  sorry

end simplify_and_evaluate_expression_l276_276530


namespace reducible_iff_form_l276_276166

def isReducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem reducible_iff_form (a : ℕ) : isReducible a ↔ ∃ k : ℕ, a = 7 * k + 1 := by
  sorry

end reducible_iff_form_l276_276166


namespace problem_part1_problem_part2_l276_276809

variable (a m : ℝ)

def prop_p (a m : ℝ) : Prop := (m - a) * (m - 3 * a) ≤ 0
def prop_q (m : ℝ) : Prop := (m + 2) * (m + 1) < 0

theorem problem_part1 (h₁ : a = -1) (h₂ : prop_p a m ∨ prop_q m) : -3 ≤ m ∧ m ≤ -1 :=
sorry

theorem problem_part2 (h₁ : ∀ m, prop_p a m → ¬prop_q m) :
  -1 / 3 ≤ a ∧ a < 0 ∨ a ≤ -2 :=
sorry

end problem_part1_problem_part2_l276_276809


namespace num_even_divisors_of_8_l276_276043

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l276_276043


namespace probability_right_angled_triangle_l276_276210

-- Definition of valid pairs that form a right-angled triangle with points (0,0) and (1,-1).
def is_valid_pair (m n : ℕ) : Prop :=
  (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 4 ∧ n = 4) ∨
  (m = 5 ∧ n = 5) ∨ (m = 6 ∧ n = 6) ∨ (m = 3 ∧ n = 1) ∨ (m = 4 ∧ n = 2) ∨
  (m = 5 ∧ n = 3) ∨ (m = 6 ∧ n = 4)

-- Set of all possible pairs (m, n) where m, n ∈ {1, 2, ..., 6}
def all_pairs : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 6).map (λ x, x + 1) (Finset.range 6).map (λ x, x + 1)

-- Set of valid pairs (m, n) that form a right-angled triangle with points (0,0) and (1,-1)
def valid_pairs : Finset (ℕ × ℕ) := all_pairs.filter (λ ⟨m, n⟩, is_valid_pair m n)

-- Probability of forming a right-angled triangle
def triangle_probability : ℝ := ((valid_pairs.card : ℝ) / (all_pairs.card : ℝ))

theorem probability_right_angled_triangle : triangle_probability = 5 / 18 := by
  sorry

end probability_right_angled_triangle_l276_276210


namespace exists_long_segment_between_parabolas_l276_276453

def parabola1 (x : ℝ) : ℝ :=
  x ^ 2

def parabola2 (x : ℝ) : ℝ :=
  x ^ 2 - 1

def in_between_parabolas (x y : ℝ) : Prop :=
  (parabola2 x) ≤ y ∧ y ≤ (parabola1 x)

theorem exists_long_segment_between_parabolas :
  ∃ (M1 M2: ℝ × ℝ), in_between_parabolas M1.1 M1.2 ∧ in_between_parabolas M2.1 M2.2 ∧ dist M1 M2 > 10^6 :=
sorry

end exists_long_segment_between_parabolas_l276_276453


namespace num_pos_divisors_36_l276_276959

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l276_276959


namespace angle_C_max_sum_of_sides_l276_276371

theorem angle_C (a b c : ℝ) (S : ℝ) (h1 : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.pi / 3 :=
by
  sorry

theorem max_sum_of_sides (a b : ℝ) (c : ℝ) (hC : c = Real.sqrt 3) :
  (a + b) ≤ 2 * Real.sqrt 3 :=
by
  sorry

end angle_C_max_sum_of_sides_l276_276371


namespace number_of_positive_divisors_of_360_is_24_l276_276205

theorem number_of_positive_divisors_of_360_is_24 :
  ∀ n : ℕ, n = 360 → n = 2^3 * 3^2 * 5^1 → 
  (n_factors : {p : ℕ × ℕ // p.1 ∈ [2, 3, 5] ∧ p.2 ∈ [3, 2, 1]} )
    → (n_factors.val.snd + 1).prod = 24 :=
by
  intro n hn h_factors
  rw hn at *
  have factors := h_factors.val
  cases factors with p_k q_l r_m
  have hpq : p_k.1 = 2 ∧ p_k.2 = 3 :=
    And.intro sorry sorry,
  have hqr : q_l.1 = 3 ∧ q_l.2 = 2 :=
    And.intro sorry sorry,
  have hr : r_m.1 = 5 ∧ r_m.2 = 1 :=
    And.intro sorry sorry,
  -- The proof would continue, but we'll skip it
  sorry

end number_of_positive_divisors_of_360_is_24_l276_276205


namespace repeating_decimals_sum_l276_276168

theorem repeating_decimals_sum :
  let x := (246 : ℚ) / 999
  let y := (135 : ℚ) / 999
  let z := (579 : ℚ) / 999
  x - y + z = (230 : ℚ) / 333 :=
by
  sorry

end repeating_decimals_sum_l276_276168


namespace find_m_sum_terms_l276_276576

theorem find_m (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) 
  (h2 : a 3 + a 6 + a 10 + a 13 = 32) (hm : a m = 8) : m = 8 :=
sorry

theorem sum_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) (hS3 : S 3 = 9) (hS6 : S 6 = 36) 
  (a_def : ∀ n, S n = n * (a 1 + a n) / 2) : a 7 + a 8 + a 9 = 45 :=
sorry

end find_m_sum_terms_l276_276576


namespace num_pos_divisors_36_l276_276964

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l276_276964


namespace verify_tin_amount_l276_276759

def ratio_to_fraction (part1 part2 : ℕ) : ℚ :=
  part2 / (part1 + part2 : ℕ)

def tin_amount_in_alloy (total_weight : ℚ) (ratio : ℚ) : ℚ :=
  total_weight * ratio

def alloy_mixture_tin_weight_is_correct
    (weight_A weight_B : ℚ)
    (ratio_A_lead ratio_A_tin : ℕ)
    (ratio_B_tin ratio_B_copper : ℕ) : Prop :=
  let tin_ratio_A := ratio_to_fraction ratio_A_lead ratio_A_tin
  let tin_ratio_B := ratio_to_fraction ratio_B_tin ratio_B_copper
  let tin_weight_A := tin_amount_in_alloy weight_A tin_ratio_A
  let tin_weight_B := tin_amount_in_alloy weight_B tin_ratio_B
  tin_weight_A + tin_weight_B = 146.57

theorem verify_tin_amount :
    alloy_mixture_tin_weight_is_correct 130 160 2 3 3 4 :=
by
  sorry

end verify_tin_amount_l276_276759


namespace vertex_angle_is_130_8_l276_276059

-- Define the given conditions
variables {a b h : ℝ}

def is_isosceles_triangle (a b h : ℝ) : Prop :=
  a^2 = b * 3 * h ∧ b = 2 * h

-- Define the obtuse condition on the vertex angle
def vertex_angle_obtuse (a b h : ℝ) : Prop :=
  ∃ θ : ℝ, 120 < θ ∧ θ < 180 ∧ θ = (130.8 : ℝ)

-- The formal proof statement using Lean 4
theorem vertex_angle_is_130_8 (a b h : ℝ) 
  (h1 : is_isosceles_triangle a b h)
  (h2 : vertex_angle_obtuse a b h) : 
  ∃ (φ : ℝ), φ = 130.8 :=
sorry

end vertex_angle_is_130_8_l276_276059


namespace magnification_factor_l276_276312

variable (diameter_magnified : ℝ)
variable (diameter_actual : ℝ)
variable (M : ℝ)

theorem magnification_factor
    (h_magnified : diameter_magnified = 0.3)
    (h_actual : diameter_actual = 0.0003) :
    M = diameter_magnified / diameter_actual ↔ M = 1000 := by
  sorry

end magnification_factor_l276_276312


namespace max_area_rect_40_perimeter_l276_276113

noncomputable def max_rect_area (P : ℕ) (hP : P = 40) : ℕ :=
  let w : ℕ → ℕ := id
  let l : ℕ → ℕ := λ w, P / 2 - w
  let area : ℕ → ℕ := λ w, w * (P / 2 - w)
  find_max_value area sorry

theorem max_area_rect_40_perimeter : max_rect_area 40 40 = 100 := 
sorry

end max_area_rect_40_perimeter_l276_276113


namespace compare_fractions_l276_276169

theorem compare_fractions : (- (4 / 5) < - (2 / 3)) :=
by
  sorry

end compare_fractions_l276_276169


namespace compute_expression_l276_276317

theorem compute_expression (y : ℕ) (h : y = 3) : 
  (y^8 + 18 * y^4 + 81) / (y^4 + 9) = 90 :=
by
  sorry

end compute_expression_l276_276317


namespace oranges_count_l276_276574

theorem oranges_count (N : ℕ) (k : ℕ) (m : ℕ) (j : ℕ) :
  (N ≡ 2 [MOD 10]) ∧ (N ≡ 0 [MOD 12]) → N = 72 :=
by
  sorry

end oranges_count_l276_276574


namespace solution_set_l276_276887

variable (x : ℝ)

def condition_1 : Prop := 2 * x - 4 ≤ 0
def condition_2 : Prop := -x + 1 < 0

theorem solution_set : (condition_1 x ∧ condition_2 x) ↔ (1 < x ∧ x ≤ 2) := by
sorry

end solution_set_l276_276887


namespace max_a_squared_b_l276_276022

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) : a^2 * b ≤ 54 :=
sorry

end max_a_squared_b_l276_276022


namespace slips_with_number_three_l276_276213

theorem slips_with_number_three : 
  ∀ (total_slips : ℕ) (number3 number8 : ℕ) (E : ℚ), 
  total_slips = 15 → 
  E = 5.6 → 
  number3 + number8 = total_slips → 
  (number3 : ℚ) / total_slips * 3 + (number8 : ℚ) / total_slips * 8 = E →
  number3 = 8 :=
by
  intros total_slips number3 number8 E h1 h2 h3 h4
  sorry

end slips_with_number_three_l276_276213


namespace maximum_area_of_rectangle_with_fixed_perimeter_l276_276116

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l276_276116


namespace shelves_used_l276_276913

def coloring_books := 87
def sold_books := 33
def books_per_shelf := 6

theorem shelves_used (h1: coloring_books - sold_books = 54) : 54 / books_per_shelf = 9 :=
by
  sorry

end shelves_used_l276_276913


namespace max_plates_l276_276314

/-- Bill can buy pans, pots, and plates for 3, 5, and 10 dollars each, respectively.
    What is the maximum number of plates he can purchase if he must buy at least
    two of each item and will spend exactly 100 dollars? -/
theorem max_plates (x y z : ℕ) (hx : x ≥ 2) (hy : y ≥ 2) (hz : z ≥ 2) 
  (h_cost : 3 * x + 5 * y + 10 * z = 100) : z = 8 :=
sorry

end max_plates_l276_276314


namespace det_A_l276_276601

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![2, -4, 5],
  ![0, 6, -2],
  ![3, -1, 2]
]

theorem det_A : A.det = -46 := by
  sorry

end det_A_l276_276601


namespace probability_heart_then_king_of_clubs_l276_276833

theorem probability_heart_then_king_of_clubs : 
  let deck := 52
  let hearts := 13
  let remaining_cards := deck - 1
  let king_of_clubs := 1
  let first_card_heart_probability := (hearts : ℝ) / deck
  let second_card_king_of_clubs_probability := (king_of_clubs : ℝ) / remaining_cards
  first_card_heart_probability * second_card_king_of_clubs_probability = 1 / 204 :=
by
  sorry

end probability_heart_then_king_of_clubs_l276_276833


namespace andrew_current_age_l276_276593

-- Definitions based on conditions.
def initial_age := 11  -- Andrew started donating at age 11
def donation_per_year := 7  -- Andrew donates 7k each year on his birthday
def total_donation := 133  -- Andrew has donated a total of 133k till now

-- The theorem stating the problem and the conclusion.
theorem andrew_current_age : 
  ∃ (A : ℕ), donation_per_year * (A - initial_age) = total_donation :=
by {
  sorry
}

end andrew_current_age_l276_276593


namespace multiply_by_11_l276_276089

theorem multiply_by_11 (A B k : ℕ) (h1 : 10 * A + B < 100) (h2 : A + B = 10 + k) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * k + B :=
by 
  sorry

end multiply_by_11_l276_276089


namespace actual_cost_of_article_l276_276755

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 760) : x = 1000 :=
by 
  sorry

end actual_cost_of_article_l276_276755


namespace simplified_expression_correct_l276_276246

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) :=
  sorry

end simplified_expression_correct_l276_276246


namespace g_at_0_eq_1_l276_276403

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x * g y
axiom g_deriv_at_0 : deriv g 0 = 2

theorem g_at_0_eq_1 : g 0 = 1 :=
by
  sorry

end g_at_0_eq_1_l276_276403


namespace average_screen_time_per_player_l276_276860

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l276_276860


namespace line_tangent_to_parabola_l276_276264

theorem line_tangent_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → 16 - 16 * c = 0) → c = 1 :=
by
  intros h
  sorry

end line_tangent_to_parabola_l276_276264


namespace total_eggs_collected_by_all_four_l276_276445

def benjamin_eggs := 6
def carla_eggs := 3 * benjamin_eggs
def trisha_eggs := benjamin_eggs - 4
def david_eggs := 2 * trisha_eggs

theorem total_eggs_collected_by_all_four :
  benjamin_eggs + carla_eggs + trisha_eggs + david_eggs = 30 := by
  sorry

end total_eggs_collected_by_all_four_l276_276445


namespace carrots_total_l276_276242

def carrots_grown_by_sally := 6
def carrots_grown_by_fred := 4
def total_carrots := carrots_grown_by_sally + carrots_grown_by_fred

theorem carrots_total : total_carrots = 10 := 
by 
  sorry  -- proof to be filled in

end carrots_total_l276_276242


namespace largestValidNumberIs84_l276_276695

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l276_276695


namespace gray_region_area_l276_276152

noncomputable def area_of_gray_region (length width : ℝ) (angle_deg : ℝ) : ℝ :=
  if (length = 55 ∧ width = 44 ∧ angle_deg = 45) then 10 else 0

theorem gray_region_area :
  area_of_gray_region 55 44 45 = 10 :=
by sorry

end gray_region_area_l276_276152


namespace largest_divisible_by_6_ending_in_4_l276_276718

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l276_276718


namespace copy_pages_l276_276659

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l276_276659


namespace contractor_absent_days_l276_276140

theorem contractor_absent_days
    (total_days : ℤ) (work_rate : ℤ) (fine_rate : ℤ) (total_amount : ℤ)
    (x y : ℤ)
    (h1 : total_days = 30)
    (h2 : work_rate = 25)
    (h3 : fine_rate = 75) -- fine_rate here is multiplied by 10 to avoid decimals
    (h4 : total_amount = 4250) -- total_amount multiplied by 10 for the same reason
    (h5 : x + y = total_days)
    (h6 : work_rate * x - fine_rate * y = total_amount) :
  y = 10 := 
by
  -- Here, we would provide the proof steps.
  sorry

end contractor_absent_days_l276_276140


namespace student_marks_l276_276781

theorem student_marks
(M P C : ℕ) -- the marks of Mathematics, Physics, and Chemistry are natural numbers
(h1 : C = P + 20)  -- Chemistry is 20 marks more than Physics
(h2 : (M + C) / 2 = 30)  -- The average marks in Mathematics and Chemistry is 30
: M + P = 40 := 
sorry

end student_marks_l276_276781


namespace sequence_arith_l276_276268

theorem sequence_arith {a : ℕ → ℕ} (h_initial : a 2 = 2) (h_recursive : ∀ n ≥ 2, a (n + 1) = a n + 1) :
  ∀ n ≥ 2, a n = n :=
by
  sorry

end sequence_arith_l276_276268


namespace total_spent_on_pens_l276_276000

/-- Dorothy, Julia, and Robert go to the store to buy school supplies.
    Dorothy buys half as many pens as Julia.
    Julia buys three times as many pens as Robert.
    Robert buys 4 pens.
    The cost of one pen is $1.50.
    Prove that the total amount of money spent on pens by the three friends is $33. 
-/
theorem total_spent_on_pens :
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen = 33 := 
by
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  sorry

end total_spent_on_pens_l276_276000


namespace range_of_a_l276_276628

theorem range_of_a (a : ℝ) : (1 < a) → 
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 
  (1 / (x1 + 2) = a * |x1| ∧ 1 / (x2 + 2) = a * |x2| ∧ 1 / (x3 + 2) = a * |x3|) :=
sorry

end range_of_a_l276_276628


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276727

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276727


namespace tim_reading_hours_per_week_l276_276560

theorem tim_reading_hours_per_week :
  (meditation_hours_per_day = 1) →
  (reading_hours_per_day = 2 * meditation_hours_per_day) →
  (reading_hours_per_week = reading_hours_per_day * 7) →
  reading_hours_per_week = 14 :=
by
  intros h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  exact h3

end tim_reading_hours_per_week_l276_276560


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l276_276715

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l276_276715


namespace perimeter_of_equilateral_triangle_l276_276877

theorem perimeter_of_equilateral_triangle (s : ℝ) 
  (h1 : (s ^ 2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_equilateral_triangle_l276_276877


namespace num_pos_divisors_36_l276_276979

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l276_276979


namespace determine_a_value_l276_276490

-- Define the initial equation and conditions
def fractional_equation (x a : ℝ) : Prop :=
  (x - a) / (x - 1) - 3 / x = 1

-- Define the existence of a positive root
def has_positive_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ fractional_equation x a

-- The main theorem stating the correct value of 'a' for the given condition
theorem determine_a_value (x : ℝ) : has_positive_root 1 :=
sorry

end determine_a_value_l276_276490


namespace probability_of_coprime_pairs_l276_276337

def numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8]

def pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.bind l (λ x => List.map (λ y => (x, y)) (List.filter (λ y => x < y) l))

def gcd (x y : ℕ) : ℕ := Nat.gcd x y

def coprime_pairs_count (pairs : List (ℕ × ℕ)) : ℕ :=
  List.length (List.filter (λ p : ℕ × ℕ => gcd p.1 p.2 = 1) pairs)

def total_pairs_count (pairs : List (ℕ × ℕ)) : ℕ := List.length pairs

theorem probability_of_coprime_pairs :
  let ps := pairs numbers
  in (coprime_pairs_count ps) / (total_pairs_count ps) = 2 / 3 := by
  sorry

end probability_of_coprime_pairs_l276_276337


namespace arithmetic_mean_a8_a11_l276_276368

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_mean_a8_a11 {a : ℕ → ℝ} (h1 : geometric_sequence a (-2)) 
    (h2 : a 2 * a 6 = 4 * a 3) :
  ((a 7 + a 10) / 2) = -56 :=
sorry

end arithmetic_mean_a8_a11_l276_276368


namespace unique_value_not_in_range_l276_276545

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_value_not_in_range (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  g p q r s 11 = 11 ∧ g p q r s 41 = 41 ∧ (∀ x, x ≠ -s / r → g p q r s (g p q r s x) = x) → ∃! y, ¬ ∃ x, g p q r s x = y :=
begin
  intros h,
  use 30,
  split,
  {
    intro hy,
    sorry -- Proof omitted
  },
  {
    intros z hz,
    have hneq : z ≠ 30 := sorry, -- Proof omitted
    assumption
  }
end

end unique_value_not_in_range_l276_276545


namespace solve_diophantine_eq_l276_276929

theorem solve_diophantine_eq (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^2 = b * (b + 7) ↔ (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) := 
by 
  sorry

end solve_diophantine_eq_l276_276929


namespace range_of_m_l276_276945

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + |x - 1| > m) → m < 1 :=
by
  sorry

end range_of_m_l276_276945


namespace avg_cost_of_6_toys_l276_276320

-- Define the given conditions
def dhoni_toys_count : ℕ := 5
def dhoni_toys_avg_cost : ℝ := 10
def sixth_toy_cost : ℝ := 16
def sales_tax_rate : ℝ := 0.10

-- Define the supposed answer
def supposed_avg_cost : ℝ := 11.27

-- Define the problem in Lean 4 statement
theorem avg_cost_of_6_toys :
  (dhoni_toys_count * dhoni_toys_avg_cost + sixth_toy_cost * (1 + sales_tax_rate)) / (dhoni_toys_count + 1) = supposed_avg_cost :=
by
  -- Proof goes here, replace with actual proof
  sorry

end avg_cost_of_6_toys_l276_276320


namespace min_cubes_required_l276_276912

/--
A lady builds a box with dimensions 10 cm length, 18 cm width, and 4 cm height using 12 cubic cm cubes. Prove that the minimum number of cubes required to build the box is 60.
-/
def min_cubes_for_box (length width height volume_cube : ℕ) : ℕ :=
  (length * width * height) / volume_cube

theorem min_cubes_required :
  min_cubes_for_box 10 18 4 12 = 60 :=
by
  -- The proof details are omitted.
  sorry

end min_cubes_required_l276_276912


namespace avg_age_all_l276_276255

-- Define the conditions
def avg_age_seventh_graders (n₁ : Nat) (a₁ : Nat) : Prop :=
  n₁ = 40 ∧ a₁ = 13

def avg_age_parents (n₂ : Nat) (a₂ : Nat) : Prop :=
  n₂ = 50 ∧ a₂ = 40

-- Define the problem to prove
def avg_age_combined (n₁ n₂ a₁ a₂ : Nat) : Prop :=
  (n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 28

-- The main theorem
theorem avg_age_all (n₁ n₂ a₁ a₂ : Nat):
  avg_age_seventh_graders n₁ a₁ → avg_age_parents n₂ a₂ → avg_age_combined n₁ n₂ a₁ a₂ :=
by 
  intros h1 h2
  sorry

end avg_age_all_l276_276255


namespace passes_through_point_l276_276635

theorem passes_through_point (a : ℝ) (h : a > 0) (h2 : a ≠ 1) : 
  (2, 1) ∈ {p : ℝ × ℝ | ∃ a, p.snd = a * p.fst - 2} :=
sorry

end passes_through_point_l276_276635


namespace average_player_time_l276_276858

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l276_276858


namespace new_batting_average_l276_276577

def initial_runs (A : ℕ) := 16 * A
def additional_runs := 85
def increased_average := 3
def runs_in_5_innings := 100 + 120 + 45 + 75 + 65
def total_runs_17_innings (A : ℕ) := 17 * (A + increased_average)
def A : ℕ := 34
def total_runs_22_innings := total_runs_17_innings A + runs_in_5_innings
def number_of_innings := 22
def new_average := total_runs_22_innings / number_of_innings

theorem new_batting_average : new_average = 47 :=
by sorry

end new_batting_average_l276_276577


namespace unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l276_276629

noncomputable def f (x k : ℝ) : ℝ := (Real.log x) - k * x + k

theorem unique_solution_f_geq_0 {k : ℝ} :
  (∃! x : ℝ, 0 < x ∧ f x k ≥ 0) ↔ k = 1 :=
sorry

theorem inequality_hold_for_a_leq_1 {a x : ℝ} (h₀ : a ≤ 1) :
  x * (f x 1 + x - 1) < Real.exp x - a * x^2 - 1 :=
sorry

end unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l276_276629


namespace min_handshakes_30_people_3_each_l276_276761

theorem min_handshakes_30_people_3_each : 
  ∃ (H : ℕ), (∀ (n k : ℕ), n = 30 ∧ k = 3 → H = (n * k) / 2) := 
by {
  use 45,
  intros n k h,
  rw [← h.1, ← h.2],
  norm_num,
  sorry
}

end min_handshakes_30_people_3_each_l276_276761


namespace coin_stack_count_l276_276365

theorem coin_stack_count
  (TN : ℝ := 1.95)
  (TQ : ℝ := 1.75)
  (SH : ℝ := 20)
  (n q : ℕ) :
  (n*Tℕ + q*TQ = SH) → (n + q = 10) :=
sorry

end coin_stack_count_l276_276365


namespace simplify_expression_l276_276678

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

noncomputable def x : ℝ := (b / c) * (c / b)
noncomputable def y : ℝ := (a / c) * (c / a)
noncomputable def z : ℝ := (a / b) * (b / a)

theorem simplify_expression : x^2 + y^2 + z^2 + x^2 * y^2 * z^2 = 4 := 
by {
  sorry
}

end simplify_expression_l276_276678


namespace num_positive_divisors_36_l276_276983

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l276_276983


namespace find_f_l276_276349

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f (x - 1/x) = x^2 + 1/x^2 - 4) :
  ∀ x : ℝ, f x = x^2 - 2 :=
by
  intros x
  sorry

end find_f_l276_276349


namespace find_mn_solutions_l276_276007

theorem find_mn_solutions :
  ∀ (m n : ℤ), m^5 - n^5 = 16 * m * n →
  (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
by
  sorry

end find_mn_solutions_l276_276007


namespace polynomial_characterization_l276_276610

theorem polynomial_characterization (P : ℝ → ℝ) :
  (∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) →
  ∃ (α β : ℝ), ∀ x : ℝ, P x = α * x^4 + β * x^2 :=
by
  sorry

end polynomial_characterization_l276_276610


namespace average_percentage_decrease_l276_276739

-- Given definitions
def original_price : ℝ := 10000
def final_price : ℝ := 6400
def num_reductions : ℕ := 2

-- The goal is to prove the average percentage decrease per reduction
theorem average_percentage_decrease (x : ℝ) (h : (original_price * (1 - x)^num_reductions = final_price)) : x = 0.2 :=
sorry

end average_percentage_decrease_l276_276739


namespace compare_fractions_l276_276170

theorem compare_fractions : (- (4 / 5) < - (2 / 3)) :=
by
  sorry

end compare_fractions_l276_276170


namespace pages_copied_l276_276672

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l276_276672


namespace quadrilateral_area_l276_276306

theorem quadrilateral_area (c d : ℤ) (h1 : 0 < d) (h2 : d < c) (h3 : 2 * ((c : ℝ) ^ 2 - (d : ℝ) ^ 2) = 18) : 
  c + d = 9 :=
by
  sorry

end quadrilateral_area_l276_276306


namespace diagonals_of_decagon_l276_276179

theorem diagonals_of_decagon : 
  let n := 10 in 
  (n * (n - 3)) / 2 = 35 := 
by
  let n := 10
  show (n * (n - 3)) / 2 = 35
  sorry

end diagonals_of_decagon_l276_276179


namespace imaginary_part_div_z1_z2_l276_276020

noncomputable def z1 := 1 - 3 * Complex.I
noncomputable def z2 := 3 + Complex.I

theorem imaginary_part_div_z1_z2 : 
  Complex.im ((1 + 3 * Complex.I) / (3 + Complex.I)) = 4 / 5 := 
by 
  sorry

end imaginary_part_div_z1_z2_l276_276020


namespace serenity_total_shoes_l276_276397

def pairs_of_shoes : ℕ := 3
def shoes_per_pair : ℕ := 2

theorem serenity_total_shoes : pairs_of_shoes * shoes_per_pair = 6 := by
  sorry

end serenity_total_shoes_l276_276397


namespace train_length_l276_276904

noncomputable def length_of_each_train (L : ℝ) : Prop :=
  let v1 := 46 -- speed of faster train in km/hr
  let v2 := 36 -- speed of slower train in km/hr
  let relative_speed := (v1 - v2) * (5/18) -- converting relative speed to m/s
  let time := 72 -- time in seconds
  2 * L = relative_speed * time -- distance equation

theorem train_length : ∃ (L : ℝ), length_of_each_train L ∧ L = 100 :=
by
  use 100
  unfold length_of_each_train
  sorry

end train_length_l276_276904


namespace tim_reading_hours_per_week_l276_276559

theorem tim_reading_hours_per_week :
  (meditation_hours_per_day = 1) →
  (reading_hours_per_day = 2 * meditation_hours_per_day) →
  (reading_hours_per_week = reading_hours_per_day * 7) →
  reading_hours_per_week = 14 :=
by
  intros h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  exact h3

end tim_reading_hours_per_week_l276_276559


namespace time_reading_per_week_l276_276564

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end time_reading_per_week_l276_276564


namespace total_coin_tosses_l276_276367

variable (heads : ℕ) (tails : ℕ)

theorem total_coin_tosses (h_head : heads = 9) (h_tail : tails = 5) : heads + tails = 14 := by
  sorry

end total_coin_tosses_l276_276367


namespace focus_of_curve_is_4_0_l276_276794

noncomputable def is_focus (p : ℝ × ℝ) (curve : ℝ × ℝ → Prop) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, curve (x, y) ↔ (y^2 = -16 * c * (x - 4))

def curve (p : ℝ × ℝ) : Prop := p.2^2 = -16 * p.1 + 64

theorem focus_of_curve_is_4_0 : is_focus (4, 0) curve :=
by
sorry

end focus_of_curve_is_4_0_l276_276794


namespace original_numbers_l276_276556

theorem original_numbers (a b c d : ℝ) (h1 : a + b + c + d = 45)
    (h2 : ∃ x : ℝ, a + 2 = x ∧ b - 2 = x ∧ 2 * c = x ∧ d / 2 = x) : 
    a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 :=
by
  sorry

end original_numbers_l276_276556


namespace kathryn_more_pints_than_annie_l276_276607

-- Definitions for conditions
def annie_pints : ℕ := 8
def ben_pints (kathryn_pints : ℕ) : ℕ := kathryn_pints - 3
def total_pints (annie_pints kathryn_pints ben_pints : ℕ) : ℕ := annie_pints + kathryn_pints + ben_pints

-- The problem statement
theorem kathryn_more_pints_than_annie (k : ℕ) (h1 : total_pints annie_pints k (ben_pints k) = 25) : k - annie_pints = 2 :=
sorry

end kathryn_more_pints_than_annie_l276_276607


namespace derivative_at_3_l276_276466

def f (x : ℝ) : ℝ := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l276_276466


namespace smallest_three_digit_perfect_square_l276_276330

theorem smallest_three_digit_perfect_square :
  ∀ (a : ℕ), 100 ≤ a ∧ a < 1000 → (∃ (n : ℕ), 1001 * a + 1 = n^2) → a = 183 :=
by
  sorry

end smallest_three_digit_perfect_square_l276_276330


namespace circle_radius_squared_l276_276429

open Real

/-- Prove that the square of the radius of a circle is 200 given the conditions provided. -/

theorem circle_radius_squared {r : ℝ}
  (AB CD : ℝ)
  (BP : ℝ) 
  (APD : ℝ) 
  (hAB : AB = 12)
  (hCD : CD = 9)
  (hBP : BP = 10)
  (hAPD : APD = 45) :
  r^2 = 200 := 
sorry

end circle_radius_squared_l276_276429


namespace copy_pages_15_dollars_l276_276669

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l276_276669


namespace number_in_tenth_group_l276_276646

-- Number of students
def students : ℕ := 1000

-- Number of groups
def groups : ℕ := 100

-- Interval between groups
def interval : ℕ := students / groups

-- First number drawn
def first_number : ℕ := 6

-- Number drawn from n-th group given first_number and interval
def number_in_group (n : ℕ) : ℕ := first_number + interval * (n - 1)

-- Statement to prove
theorem number_in_tenth_group :
  number_in_group 10 = 96 :=
by
  sorry

end number_in_tenth_group_l276_276646


namespace total_third_graders_l276_276416

theorem total_third_graders (num_girls : ℕ) (num_boys : ℕ) (h1 : num_girls = 57) (h2 : num_boys = 66) : num_girls + num_boys = 123 :=
by
  sorry

end total_third_graders_l276_276416


namespace debate_club_girls_l276_276302

theorem debate_club_girls (B G : ℕ) 
  (h1 : B + G = 22)
  (h2 : B + (1/3 : ℚ) * G = 14) : G = 12 :=
sorry

end debate_club_girls_l276_276302


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276724

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l276_276724


namespace sum_of_four_primes_is_prime_l276_276407

theorem sum_of_four_primes_is_prime
    (A B : ℕ)
    (hA_prime : Prime A)
    (hB_prime : Prime B)
    (hA_minus_B_prime : Prime (A - B))
    (hA_plus_B_prime : Prime (A + B)) :
    Prime (A + B + (A - B) + A) :=
by
  sorry

end sum_of_four_primes_is_prime_l276_276407


namespace smallest_prime_factor_of_setC_l276_276528

def setC : Set ℕ := {51, 53, 54, 56, 57}

def prime_factors (n : ℕ) : Set ℕ :=
  { p | p.Prime ∧ p ∣ n }

theorem smallest_prime_factor_of_setC :
  (∃ n ∈ setC, ∀ m ∈ setC, ∀ p ∈ prime_factors n, ∀ q ∈ prime_factors m, p ≤ q) ∧
  (∃ m ∈ setC, ∀ p ∈ prime_factors 54, ∀ q ∈ prime_factors m, p = q) := 
sorry

end smallest_prime_factor_of_setC_l276_276528


namespace felix_chopped_at_least_91_trees_l276_276934

def cost_to_sharpen := 5
def total_spent := 35
def trees_per_sharpen := 13

theorem felix_chopped_at_least_91_trees :
  (total_spent / cost_to_sharpen) * trees_per_sharpen = 91 := by
  sorry

end felix_chopped_at_least_91_trees_l276_276934


namespace construct_trihedral_angle_l276_276354

-- Define the magnitudes of dihedral angles
variables (α β γ : ℝ)

-- Problem statement
theorem construct_trihedral_angle (h₀ : 0 < α) (h₁ : 0 < β) (h₂ : 0 < γ) :
  ∃ (trihedral_angle : Type), true := 
sorry

end construct_trihedral_angle_l276_276354


namespace binary_multiplication_correct_l276_276186

theorem binary_multiplication_correct :
  (0b1101 : ℕ) * (0b1011 : ℕ) = (0b10011011 : ℕ) :=
by
  sorry

end binary_multiplication_correct_l276_276186


namespace complex_numbers_equation_l276_276680

theorem complex_numbers_equation {a b : ℂ} (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := 
by sorry

end complex_numbers_equation_l276_276680


namespace cubic_meter_to_cubic_centimeters_l276_276820

theorem cubic_meter_to_cubic_centimeters :
  (1 : ℝ) ^ 3 = (100 : ℝ) ^ 3 := by
  sorry

end cubic_meter_to_cubic_centimeters_l276_276820


namespace number_of_divisors_36_l276_276972

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l276_276972


namespace avg_primes_between_30_and_50_l276_276009

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

def sum_primes : ℕ := primes_between_30_and_50.sum

def count_primes : ℕ := primes_between_30_and_50.length

def average_primes : ℚ := (sum_primes : ℚ) / (count_primes : ℚ)

theorem avg_primes_between_30_and_50 : average_primes = 39.8 := by
  sorry

end avg_primes_between_30_and_50_l276_276009


namespace shirt_price_is_150_l276_276459

def price_of_shirt (X C : ℝ) : Prop :=
  (X + C = 600) ∧ (X = C / 3)

theorem shirt_price_is_150 :
  ∃ X C : ℝ, price_of_shirt X C ∧ X = 150 :=
by {
  use [150, 450],
  dsimp [price_of_shirt],
  split,
  { norm_num, },
  { norm_num, },
}

end shirt_price_is_150_l276_276459


namespace quadratic_inequality_solution_l276_276792

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - x - 12 < 0) ↔ (-3 < x ∧ x < 4) :=
by sorry

end quadratic_inequality_solution_l276_276792


namespace find_m_l276_276993

theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) : m = 2 :=
sorry

end find_m_l276_276993


namespace copy_pages_l276_276658

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l276_276658


namespace expression_value_l276_276893

theorem expression_value : 4 * (8 - 2) ^ 2 - 6 = 138 :=
by
  sorry

end expression_value_l276_276893


namespace relationship_between_y_values_l276_276523

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variables (m : ℝ) (y1 y2 y3 : ℝ)
variables (h : m > 0)
variables (h1 : y1 = quadratic_function m (-1))
variables (h2 : y2 = quadratic_function m (5 / 2))
variables (h3 : y3 = quadratic_function m 6)

theorem relationship_between_y_values : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end relationship_between_y_values_l276_276523


namespace triangle_base_l276_276369

noncomputable def side_length_square (p : ℕ) : ℕ := p / 4

noncomputable def area_square (s : ℕ) : ℕ := s * s

noncomputable def area_triangle (h b : ℕ) : ℕ := (h * b) / 2

theorem triangle_base (p h a b : ℕ) (hp : p = 80) (hh : h = 40) (ha : a = (side_length_square p)^2) (eq_areas : area_square (side_length_square p) = area_triangle h b) : b = 20 :=
by {
  -- Here goes the proof which we are omitting
  sorry
}

end triangle_base_l276_276369


namespace simplify_expression_l276_276093

theorem simplify_expression :
  18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 :=
by
  sorry

end simplify_expression_l276_276093


namespace largest_two_digit_divisible_by_6_ending_in_4_l276_276731

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l276_276731


namespace arithmetic_sequence_a2_value_l276_276813

theorem arithmetic_sequence_a2_value :
  ∃ (a : ℕ) (d : ℕ), (a = 3) ∧ (a + d + (a + 2 * d) = 12) ∧ (a + d = 5) :=
by
  sorry

end arithmetic_sequence_a2_value_l276_276813


namespace part_I_intersection_part_I_union_complements_part_II_range_l276_276023

namespace MathProof

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Prove that the intersection of A and B is {x | 3 < x ∧ x < 6}
theorem part_I_intersection : A ∩ B = {x | 3 < x ∧ x < 6} := sorry

-- Prove that the union of the complements of A and B is {x | x ≤ 3 ∨ x ≥ 6}
theorem part_I_union_complements : (Aᶜ ∪ Bᶜ) = {x | x ≤ 3 ∨ x ≥ 6} := sorry

-- Prove the range of a such that C is a subset of B and B union C equals B
theorem part_II_range (a : ℝ) : B ∪ C a = B → (a ≤ 1 ∨ 2 ≤ a ∧ a ≤ 5) := sorry

end MathProof

end part_I_intersection_part_I_union_complements_part_II_range_l276_276023


namespace right_triangle_hypotenuse_length_l276_276308

theorem right_triangle_hypotenuse_length (a b c : ℝ) (h₀ : a = 7) (h₁ : b = 24) (h₂ : a^2 + b^2 = c^2) : c = 25 :=
by
  rw [h₀, h₁] at h₂
  -- This step will simplify the problem
  sorry

end right_triangle_hypotenuse_length_l276_276308


namespace garden_perimeter_is_64_l276_276892

theorem garden_perimeter_is_64 :
    ∀ (width_garden length_garden width_playground length_playground : ℕ),
    width_garden = 24 →
    width_playground = 12 →
    length_playground = 16 →
    width_playground * length_playground = width_garden * length_garden →
    2 * length_garden + 2 * width_garden = 64 :=
by
  intros width_garden length_garden width_playground length_playground
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end garden_perimeter_is_64_l276_276892


namespace quadratic_inequality_solution_l276_276789

theorem quadratic_inequality_solution (x : ℝ) : -3 < x ∧ x < 4 → x^2 - x - 12 < 0 := by
  sorry

end quadratic_inequality_solution_l276_276789


namespace circle_area_from_circumference_l276_276257

theorem circle_area_from_circumference (r : ℝ) (π : ℝ) (h1 : 2 * π * r = 36) : (π * (r^2) = 324 / π) := by
  sorry

end circle_area_from_circumference_l276_276257


namespace women_in_village_l276_276362

theorem women_in_village (W : ℕ) (men_present : ℕ := 150) (p : ℝ := 140.78099890167377) 
    (men_reduction_per_year: ℝ := 0.10) (year1_men : ℝ := men_present * (1 - men_reduction_per_year)) 
    (year2_men : ℝ := year1_men * (1 - men_reduction_per_year)) 
    (formula : ℝ := (year2_men^2 + W^2).sqrt) 
    (h : formula = p) : W = 71 := 
by
  sorry

end women_in_village_l276_276362


namespace expression_divisibility_l276_276239

theorem expression_divisibility (x y : ℤ) (k_1 k_2 : ℤ) (h1 : 2 * x + 3 * y = 17 * k_1) :
    ∃ k_2 : ℤ, 9 * x + 5 * y = 17 * k_2 :=
by
  sorry

end expression_divisibility_l276_276239


namespace trajectory_equation_l276_276836

noncomputable def A : ℝ × ℝ := (0, -1)
noncomputable def B (x_b : ℝ) : ℝ × ℝ := (x_b, -3)
noncomputable def M (x y : ℝ) : ℝ × ℝ := (x, y)

-- Conditions as definitions in Lean 4
def MB_parallel_OA (x y x_b : ℝ) : Prop :=
  ∃ k : ℝ, (x_b - x) = k * 0 ∧ (-3 - y) = k * (-1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def condition (x y x_b : ℝ) : Prop :=
  let MA := (0 - x, -1 - y)
  let AB := (x_b - 0, -3 - (-1))
  let MB := (x_b - x, -3 - y)
  let BA := (-x_b, 2)

  dot_product MA AB = dot_product MB BA

theorem trajectory_equation : ∀ x y, (∀ x_b, MB_parallel_OA x y x_b) → condition x y x_b → y = (1 / 4) * x^2 - 2 :=
by
  intros
  sorry

end trajectory_equation_l276_276836


namespace number_of_ways_to_form_team_l276_276889

noncomputable def binomial : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial n k + binomial n (k + 1)

theorem number_of_ways_to_form_team :
  let total_selections := binomial 11 5
  let all_boys_selections := binomial 8 5
  total_selections - all_boys_selections = 406 :=
by 
  sorry

end number_of_ways_to_form_team_l276_276889


namespace number_of_oarsmen_l276_276879

-- Define the conditions
variables (n : ℕ)
variables (W : ℕ)
variables (h_avg_increase : (W + 40) / n = W / n + 2)

-- Lean 4 statement without the proof
theorem number_of_oarsmen : n = 20 :=
by
  sorry

end number_of_oarsmen_l276_276879


namespace sum_of_segments_l276_276753

noncomputable def segment_sum (AB_len CB_len FG_len : ℕ) : ℝ :=
  199 * (Real.sqrt (AB_len * AB_len + CB_len * CB_len) +
         Real.sqrt (AB_len * AB_len + FG_len * FG_len))

theorem sum_of_segments : segment_sum 5 6 8 = 199 * (Real.sqrt 61 + Real.sqrt 89) :=
by
  sorry

end sum_of_segments_l276_276753


namespace number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l276_276375

def Jungkook_cards : Real := 0.8
def Yoongi_cards : Real := 0.5

theorem number_of_people_with_cards_leq_0_point_3 : 
  (Jungkook_cards <= 0.3 ∨ Yoongi_cards <= 0.3) = False := 
by 
  -- neither Jungkook nor Yoongi has number cards less than or equal to 0.3
  sorry

theorem number_of_people_with_cards_leq_0_point_3_count :
  (if (Jungkook_cards <= 0.3) then 1 else 0) + (if (Yoongi_cards <= 0.3) then 1 else 0) = 0 :=
by 
  -- calculate number of people with cards less than or equal to 0.3
  sorry

end number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l276_276375


namespace total_books_sold_l276_276230

theorem total_books_sold (tuesday_books wednesday_books thursday_books : Nat) 
  (h1 : tuesday_books = 7) 
  (h2 : wednesday_books = 3 * tuesday_books) 
  (h3 : thursday_books = 3 * wednesday_books) : 
  tuesday_books + wednesday_books + thursday_books = 91 := 
by 
  sorry

end total_books_sold_l276_276230


namespace fraction_equality_l276_276605

def at_op (a b : ℕ) : ℕ := a * b - b^2 + b^3
def hash_op (a b : ℕ) : ℕ := a + b - a * b^2 + a * b^3

theorem fraction_equality : 
  ∀ (a b : ℕ), a = 7 → b = 3 → (at_op a b : ℚ) / (hash_op a b : ℚ) = 39 / 136 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  sorry

end fraction_equality_l276_276605


namespace tan_trig_identity_l276_276953

noncomputable def given_condition (α : ℝ) : Prop :=
  Real.tan (α + Real.pi / 3) = 2

theorem tan_trig_identity (α : ℝ) (h : given_condition α) :
  (Real.sin (α + (4 * Real.pi / 3)) + Real.cos ((2 * Real.pi / 3) - α)) /
  (Real.cos ((Real.pi / 6) - α) - Real.sin (α + (5 * Real.pi / 6))) = -3 :=
sorry

end tan_trig_identity_l276_276953


namespace handshakes_at_convention_l276_276415

theorem handshakes_at_convention (num_gremlins : ℕ) (num_imps : ℕ) 
  (H_gremlins_shake : num_gremlins = 25) (H_imps_shake_gremlins : num_imps = 20) : 
  let handshakes_among_gremlins := num_gremlins * (num_gremlins - 1) / 2
  let handshakes_between_imps_and_gremlins := num_imps * num_gremlins
  let total_handshakes := handshakes_among_gremlins + handshakes_between_imps_and_gremlins
  total_handshakes = 800 := 
by 
  sorry

end handshakes_at_convention_l276_276415


namespace sqrt_multiplication_division_l276_276448

theorem sqrt_multiplication_division :
  Real.sqrt 27 * Real.sqrt (8 / 3) / Real.sqrt (1 / 2) = 18 :=
by
  sorry

end sqrt_multiplication_division_l276_276448


namespace division_proof_l276_276785

def dividend : ℕ := 144
def inner_divisor_num : ℕ := 12
def inner_divisor_denom : ℕ := 2
def final_divisor : ℕ := inner_divisor_num / inner_divisor_denom
def expected_result : ℕ := 24

theorem division_proof : (dividend / final_divisor) = expected_result := by
  sorry

end division_proof_l276_276785


namespace updated_mean_of_observations_l276_276404

theorem updated_mean_of_observations
    (number_of_observations : ℕ)
    (initial_mean : ℝ)
    (decrement_per_observation : ℝ)
    (h1 : number_of_observations = 50)
    (h2 : initial_mean = 200)
    (h3 : decrement_per_observation = 15) :
    (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 185 :=
by {
    sorry
}

end updated_mean_of_observations_l276_276404


namespace num_pos_divisors_36_l276_276977

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l276_276977


namespace problem_a_problem_b_problem_c_l276_276291

variables {x y z t : ℝ}

-- Variables are positive
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom pos_t : 0 < t

-- Problem a)
theorem problem_a : x^4 * y^2 * z + y^4 * x^2 * z + y^4 * z^2 * x + z^4 * y^2 * x + x^4 * z^2 * y + z^4 * x^2 * y
  ≥ 2 * (x^3 * y^2 * z^2 + x^2 * y^3 * z^2 + x^2 * y^2 * z^3) :=
sorry

-- Problem b)
theorem problem_b : x^5 + y^5 + z^5 ≥ x^2 * y^2 * z + x^2 * y * z^2 + x * y^2 * z^2 :=
sorry

-- Problem c)
theorem problem_c : x^3 + y^3 + z^3 + t^3 ≥ x * y * z + x * y * t + x * z * t + y * z * t :=
sorry

end problem_a_problem_b_problem_c_l276_276291


namespace simplify_expression_l276_276867

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 :=
by
  sorry

end simplify_expression_l276_276867


namespace quotient_base_6_l276_276006

noncomputable def base_6_to_base_10 (n : ℕ) : ℕ := 
  match n with
  | 2314 => 2 * 6^3 + 3 * 6^2 + 1 * 6^1 + 4
  | 14 => 1 * 6^1 + 4
  | _ => 0

noncomputable def base_10_to_base_6 (n : ℕ) : ℕ := 
  match n with
  | 55 => 1 * 6^2 + 3 * 6^1 + 5
  | _ => 0

theorem quotient_base_6 :
  base_10_to_base_6 ((base_6_to_base_10 2314) / (base_6_to_base_10 14)) = 135 :=
by
  sorry

end quotient_base_6_l276_276006


namespace sequence_sum_l276_276834

theorem sequence_sum {A B C D E F G H I J : ℤ} (hD : D = 8)
    (h_sum1 : A + B + C + D = 45)
    (h_sum2 : B + C + D + E = 45)
    (h_sum3 : C + D + E + F = 45)
    (h_sum4 : D + E + F + G = 45)
    (h_sum5 : E + F + G + H = 45)
    (h_sum6 : F + G + H + I = 45)
    (h_sum7 : G + H + I + J = 45)
    (h_sum8 : H + I + J + A = 45)
    (h_sum9 : I + J + A + B = 45)
    (h_sum10 : J + A + B + C = 45) :
  A + J = 0 := 
sorry

end sequence_sum_l276_276834


namespace greater_solution_of_quadratic_l276_276279

theorem greater_solution_of_quadratic :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 5 * x₁ - 84 = 0) ∧ (x₂^2 - 5 * x₂ - 84 = 0) ∧ (max x₁ x₂ = 12) :=
by
  sorry

end greater_solution_of_quadratic_l276_276279


namespace angle_SR_XY_is_70_l276_276492

-- Define the problem conditions
variables (X Y Z V H S R : Type) 
variables (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ)

-- Set the conditions
def triangleXYZ (X Y Z V H S R : Type) (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ) : Prop :=
  angleX = 40 ∧ angleY = 70 ∧ XY = 12 ∧ XV = 2 ∧ YH = 2 ∧
  ∃ S R, S = (XY / 2) ∧ R = ((XV + YH) / 2)

-- Construct the theorem to be proven
theorem angle_SR_XY_is_70 {X Y Z V H S R : Type} 
  {angleX angleY angleZ angleSRXY : ℝ} 
  {XY XV YH : ℝ} : 
  triangleXYZ X Y Z V H S R angleX angleY angleZ angleSRXY XY XV YH →
  angleSRXY = 70 :=
by
  -- Placeholder proof steps
  sorry

end angle_SR_XY_is_70_l276_276492


namespace max_sin_a_l276_276220

theorem max_sin_a (a b : ℝ)
  (h1 : b = Real.pi / 2 - a)
  (h2 : Real.cos (a + b) = Real.cos a + Real.cos b) :
  Real.sin a ≤ Real.sqrt 2 / 2 :=
sorry

end max_sin_a_l276_276220


namespace distribution_methods_l276_276061

theorem distribution_methods (n m k : Nat) (h : n = 23) (h1 : m = 10) (h2 : k = 2) :
  (∃ d : Nat, d = Nat.choose m 1 + 2 * Nat.choose m 2 + Nat.choose m 3) →
  ∃ x : Nat, x = 220 :=
by
  sorry

end distribution_methods_l276_276061


namespace solve_inequalities_l276_276324

theorem solve_inequalities :
  {x : ℝ // 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 8} = {x : ℝ // 3 < x ∧ x < 4} :=
sorry

end solve_inequalities_l276_276324


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276699

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276699


namespace number_of_divisors_360_l276_276204

theorem number_of_divisors_360 : 
  ∃ (e1 e2 e3 : ℕ), e1 = 3 ∧ e2 = 2 ∧ e3 = 1 ∧ (∏ e in [e1, e2, e3], e + 1) = 24 := by
    use 3, 2, 1
    split
    { exact rfl }
    split
    { exact rfl }
    split
    { exact rfl }
    simp
    norm_num

end number_of_divisors_360_l276_276204


namespace absolute_value_condition_necessary_non_sufficient_l276_276637

theorem absolute_value_condition_necessary_non_sufficient (x : ℝ) :
  (abs (x - 1) < 2 → x^2 < x) ∧ ¬ (x^2 < x → abs (x - 1) < 2) := sorry

end absolute_value_condition_necessary_non_sufficient_l276_276637


namespace recipe_serves_correctly_l276_276687

theorem recipe_serves_correctly:
  ∀ (cream_fat_per_cup : ℝ) (cream_amount_cup : ℝ) (fat_per_serving : ℝ) (total_servings: ℝ),
    cream_fat_per_cup = 88 →
    cream_amount_cup = 0.5 →
    fat_per_serving = 11 →
    total_servings = (cream_amount_cup * cream_fat_per_cup) / fat_per_serving →
    total_servings = 4 :=
by
  intros cream_fat_per_cup cream_amount_cup fat_per_serving total_servings
  intros hcup hccup hfserv htserv
  sorry

end recipe_serves_correctly_l276_276687


namespace volume_ratio_l276_276278

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) : 
  C / (A + B) = 23 / 12 :=
sorry

end volume_ratio_l276_276278


namespace bottles_more_than_apples_l276_276433

-- Definitions given in the conditions
def apples : ℕ := 36
def regular_soda_bottles : ℕ := 80
def diet_soda_bottles : ℕ := 54

-- Theorem statement representing the question
theorem bottles_more_than_apples : (regular_soda_bottles + diet_soda_bottles) - apples = 98 :=
by
  sorry

end bottles_more_than_apples_l276_276433


namespace g_half_l276_276604

noncomputable def g : ℝ → ℝ := sorry

axiom g0 : g 0 = 0
axiom g1 : g 1 = 1
axiom g_non_decreasing : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom g_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom g_fraction : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

theorem g_half : g (1 / 2) = 1 / 2 := sorry

end g_half_l276_276604


namespace solution_set_of_inequality_l276_276047

theorem solution_set_of_inequality (a x : ℝ) (h : 1 < a) :
  (x - a) * (x - (1 / a)) > 0 ↔ x < 1 / a ∨ x > a :=
by
  sorry

end solution_set_of_inequality_l276_276047


namespace find_ravish_marks_l276_276395

-- Define the data according to the conditions.
def max_marks : ℕ := 200
def passing_percentage : ℕ := 40
def failed_by : ℕ := 40

-- The main theorem we need to prove.
theorem find_ravish_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) 
  (passing_marks := (max_marks * passing_percentage) / 100)
  (ravish_marks := passing_marks - failed_by) 
  : ravish_marks = 40 := by sorry

end find_ravish_marks_l276_276395


namespace sequence_formula_l276_276031

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) :
  ∀ n : ℕ, a n = 2 ^ n - 1 :=
sorry

end sequence_formula_l276_276031


namespace expenditure_should_increase_by_21_percent_l276_276425

noncomputable def old_income := 100.0
noncomputable def ratio_exp_sav := (3 : ℝ) / (2 : ℝ)
noncomputable def income_increase_percent := 15.0 / 100.0
noncomputable def savings_increase_percent := 6.0 / 100.0
noncomputable def old_expenditure := old_income * (3 / (3 + 2))
noncomputable def old_savings := old_income * (2 / (3 + 2))
noncomputable def new_income := old_income * (1 + income_increase_percent)
noncomputable def new_savings := old_savings * (1 + savings_increase_percent)
noncomputable def new_expenditure := new_income - new_savings
noncomputable def expenditure_increase_percent := ((new_expenditure - old_expenditure) / old_expenditure) * 100

theorem expenditure_should_increase_by_21_percent :
  expenditure_increase_percent = 21 :=
sorry

end expenditure_should_increase_by_21_percent_l276_276425


namespace serum_prevents_colds_l276_276583

noncomputable def hypothesis_preventive_effect (H : Prop) : Prop :=
  let K2 := 3.918
  let critical_value := 3.841
  let P_threshold := 0.05
  K2 >= critical_value ∧ P_threshold = 0.05 → H

theorem serum_prevents_colds (H : Prop) : hypothesis_preventive_effect H → H :=
by
  -- Proof will be added here
  sorry

end serum_prevents_colds_l276_276583


namespace largest_vs_smallest_circles_l276_276380

variable (M : Type) [MetricSpace M] [MeasurableSpace M]

def non_overlapping_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

def covering_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

theorem largest_vs_smallest_circles (M : Type) [MetricSpace M] [MeasurableSpace M] :
  non_overlapping_circles M ≥ covering_circles M :=
sorry

end largest_vs_smallest_circles_l276_276380


namespace train_ride_time_in_hours_l276_276847

-- Definition of conditions
def lukes_total_trip_time_hours : ℕ := 8
def bus_ride_minutes : ℕ := 75
def walk_to_train_center_minutes : ℕ := 15
def wait_time_minutes : ℕ := 2 * walk_to_train_center_minutes

-- Convert total trip time to minutes
def lukes_total_trip_time_minutes : ℕ := lukes_total_trip_time_hours * 60

-- Calculate the total time spent on bus, walking, and waiting
def bus_walk_wait_time_minutes : ℕ :=
  bus_ride_minutes + walk_to_train_center_minutes + wait_time_minutes

-- Calculate the train ride time in minutes
def train_ride_time_minutes : ℕ :=
  lukes_total_trip_time_minutes - bus_walk_wait_time_minutes

-- Prove the train ride time in hours
theorem train_ride_time_in_hours : train_ride_time_minutes / 60 = 6 :=
by
  sorry

end train_ride_time_in_hours_l276_276847


namespace max_constant_inequality_l276_276611

theorem max_constant_inequality (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha1 : a ≤ 1)
    (hb : 0 ≤ b) (hb1 : b ≤ 1)
    (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (hd : 0 ≤ d) (hd1 : d ≤ 1) 
    : a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3) :=
sorry

end max_constant_inequality_l276_276611


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276703

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l276_276703


namespace pages_copied_for_15_dollars_l276_276663

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l276_276663


namespace age_ratio_l276_276506

def Kul : ℕ := 22
def Saras : ℕ := 33

theorem age_ratio : (Saras / Kul : ℚ) = 3 / 2 := by
  sorry

end age_ratio_l276_276506


namespace parabola_vertex_coordinates_l276_276346

theorem parabola_vertex_coordinates {a b c : ℝ} (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 3)
  (h_root : a * 2^2 + b * 2 + c = 3) (h_symm : ∀ x : ℝ, a * (2 - x)^2 + b * (2 - x) + c = a * x^2 + b * x + c) :
  (2, 3) = (2, 3) :=
by
  sorry

end parabola_vertex_coordinates_l276_276346


namespace expected_points_correct_l276_276441

noncomputable def expected_points : ℕ :=
  let n := 13 in
  let p := 5 / 12 in
  (n - 1) * p

theorem expected_points_correct : expected_points = 5 := by
  -- Define the number of rolls
  let n := 13
  -- Define the probability of gaining a point on a single roll after the first
  let p := 5 / 12
  -- Calculate the expected number of points
  have : expected_points = (n - 1) * p := rfl
  -- Simplify and verify the result
  have : (n - 1) * p = 12 * (5 / 12) := by
    simp [n, p]
  show expected_points = 5 from by
    rw this
    norm_num
    sorry

end expected_points_correct_l276_276441


namespace problem1_problem2_l276_276532

-- Define conditions for Problem 1
def problem1_cond (x : ℝ) : Prop :=
  x ≠ 0 ∧ 2 * x ≠ 1

-- Statement for Problem 1
theorem problem1 (x : ℝ) (h : problem1_cond x) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by
  sorry

-- Define conditions for Problem 2
def problem2_cond (x : ℝ) : Prop :=
  x ≠ 2 

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h : problem2_cond x) :
  ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ↔ x = 1 := by
  sorry

end problem1_problem2_l276_276532


namespace geometric_series_sum_l276_276167

-- Define the geometric series
def geometricSeries (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

-- Define the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 5

-- Define the sum of the first n terms using the provided formula
def S_n := geometricSeries a r n

-- State the theorem: the sum S_5 equals the given answer
theorem geometric_series_sum :
  S_n = 1023 / 3072 :=
by
  sorry

end geometric_series_sum_l276_276167


namespace number_of_divisors_36_l276_276973

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l276_276973


namespace find_value_of_n_l276_276804

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_value_of_n
  (a b c n : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (hc : is_prime c)
  (h1 : 2 * a + 3 * b = c)
  (h2 : 4 * a + c + 1 = 4 * b)
  (h3 : n = a * b * c)
  (h4 : n < 10000) :
  n = 1118 :=
by
  sorry

end find_value_of_n_l276_276804


namespace number_of_B_is_14_l276_276832

-- Define the problem conditions
variable (num_students : ℕ)
variable (num_A num_B num_C num_D : ℕ)
variable (h1 : num_A = 8 * num_B / 10)
variable (h2 : num_C = 13 * num_B / 10)
variable (h3 : num_D = 5 * num_B / 10)
variable (h4 : num_students = 50)
variable (h5 : num_A + num_B + num_C + num_D = num_students)

-- Formalize the statement to be proved
theorem number_of_B_is_14 :
  num_B = 14 := by
  sorry

end number_of_B_is_14_l276_276832


namespace coprime_probability_is_correct_l276_276334

-- Define the set of integers from 2 to 8
def set_of_integers : List ℕ := [2, 3, 4, 5, 6, 7, 8]

-- Define a function to verify if two numbers are coprime
def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Calculate all possible pairs of integers
def pairs (l : List ℕ) : List (ℕ × ℕ) :=
  l.bind (λ x, l.map (λ y, (x, y))).filter (λ pair, pair.fst < pair.snd)

-- Select pairs and verify if they are coprime
def coprime_pairs : List (ℕ × ℕ) :=
  (pairs set_of_integers).filter (λ pair, are_coprime pair.fst pair.snd)

-- Calculate the probability as the ratio of coprime pairs to total pairs
def coprime_probability : ℚ :=
  coprime_pairs.length / (pairs set_of_integers).length

-- Prove that the coprime probability is 2/3
theorem coprime_probability_is_correct : coprime_probability = 2 / 3 := by
  sorry

end coprime_probability_is_correct_l276_276334


namespace hyperbola_eccentricity_range_l276_276029

theorem hyperbola_eccentricity_range (a b e : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_upper : b / a < 2) :
  e = Real.sqrt (1 + (b / a) ^ 2) → 1 < e ∧ e < Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_range_l276_276029


namespace intersection_M_N_l276_276377

def M (x : ℝ) : Prop := x^2 + 2*x - 15 < 0
def N (x : ℝ) : Prop := x^2 + 6*x - 7 ≥ 0

theorem intersection_M_N : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_M_N_l276_276377


namespace sales_tax_is_8_percent_l276_276236

-- Define the conditions
def total_before_tax : ℝ := 150
def total_with_tax : ℝ := 162

-- Define the relationship to find the sales tax percentage
noncomputable def sales_tax_percent (before_tax after_tax : ℝ) : ℝ :=
  ((after_tax - before_tax) / before_tax) * 100

-- State the theorem to prove the sales tax percentage is 8%
theorem sales_tax_is_8_percent :
  sales_tax_percent total_before_tax total_with_tax = 8 :=
by
  -- skipping the proof
  sorry

end sales_tax_is_8_percent_l276_276236


namespace cafeteria_sales_comparison_l276_276212

theorem cafeteria_sales_comparison
  (S : ℝ) -- initial sales
  (a : ℝ) -- monthly increment for Cafeteria A
  (p : ℝ) -- monthly percentage increment for Cafeteria B
  (h1 : S > 0) -- initial sales are positive
  (h2 : a > 0) -- constant increment for Cafeteria A is positive
  (h3 : p > 0) -- constant percentage increment for Cafeteria B is positive
  (h4 : S + 8 * a = S * (1 + p) ^ 8) -- sales are equal in September 2013
  (h5 : S = S) -- sales are equal in January 2013 (trivially true)
  : S + 4 * a > S * (1 + p) ^ 4 := 
sorry

end cafeteria_sales_comparison_l276_276212


namespace power_function_increasing_is_3_l276_276625

theorem power_function_increasing_is_3 (m : ℝ) :
  (∀ x : ℝ, x > 0 → (m^2 - m - 5) * (x^(m)) > 0) ∧ (m^2 - m - 5 = 1) → m = 3 :=
by
  sorry

end power_function_increasing_is_3_l276_276625


namespace part1_part2_l276_276199

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1 (a : ℝ) (h : a = 2) : 
  {x : ℝ | f x a ≥ 4 - abs (x - 4)} = {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 5} :=
by
  sorry

theorem part2 (set_is : {x : ℝ | 1 ≤ x ∧ x ≤ 2}) : 
  ∃ a : ℝ, 
    (∀ x : ℝ, abs (f (2*x + a) a - 2*f x a) ≤ 2 → (1 ≤ x ∧ x ≤ 2)) ∧ 
    a = 3 :=
by
  sorry

end part1_part2_l276_276199


namespace total_nuggets_ordered_l276_276784

noncomputable def Alyssa_nuggets : ℕ := 20
noncomputable def Keely_nuggets : ℕ := 2 * Alyssa_nuggets
noncomputable def Kendall_nuggets : ℕ := 2 * Alyssa_nuggets

theorem total_nuggets_ordered : Alyssa_nuggets + Keely_nuggets + Kendall_nuggets = 100 := by
  sorry -- Proof is intentionally omitted

end total_nuggets_ordered_l276_276784
