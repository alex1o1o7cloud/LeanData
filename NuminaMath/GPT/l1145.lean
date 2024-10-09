import Mathlib

namespace rahim_books_l1145_114562

/-- 
Rahim bought some books for Rs. 6500 from one shop and 35 books for Rs. 2000 from another. 
The average price he paid per book is Rs. 85. 
Prove that Rahim bought 65 books from the first shop. 
-/
theorem rahim_books (x : ℕ) 
  (h1 : 6500 + 2000 = 8500) 
  (h2 : 85 * (x + 35) = 8500) : 
  x = 65 := 
sorry

end rahim_books_l1145_114562


namespace cosine_range_l1145_114501

theorem cosine_range {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.cos x ≤ 1 / 2) : 
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end cosine_range_l1145_114501


namespace smallest_n_terminating_decimal_l1145_114570

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (n > 0) ∧
           (∃ (k: ℕ), (n + 150) = 2^k ∧ k < 150) ∨ 
           (∃ (k m: ℕ), (n + 150) = 2^k * 5^m ∧ m < 150) ∧ 
           ∀ m : ℕ, ((m > 0 ∧ (∃ (j: ℕ), (m + 150) = 2^j ∧ j < 150) ∨ 
           (∃ (j l: ℕ), (m + 150) = 2^j * 5^l ∧ l < 150)) → m ≥ n)
:= ⟨10, by {
  sorry
}⟩

end smallest_n_terminating_decimal_l1145_114570


namespace increase_in_cases_second_day_l1145_114592

-- Define the initial number of cases.
def initial_cases : ℕ := 2000

-- Define the number of recoveries on the second day.
def recoveries_day2 : ℕ := 50

-- Define the number of new cases on the third day and the recoveries on the third day.
def new_cases_day3 : ℕ := 1500
def recoveries_day3 : ℕ := 200

-- Define the total number of positive cases after the third day.
def total_cases_day3 : ℕ := 3750

-- Lean statement to prove the increase in cases on the second day is 750.
theorem increase_in_cases_second_day : 
  ∃ x : ℕ, initial_cases + x - recoveries_day2 + new_cases_day3 - recoveries_day3 = total_cases_day3 ∧ x = 750 :=
by
  sorry

end increase_in_cases_second_day_l1145_114592


namespace sum_of_roots_of_quadratic_eq_l1145_114546

theorem sum_of_roots_of_quadratic_eq (x : ℝ) (hx : x^2 = 8 * x + 15) :
  ∃ S : ℝ, S = 8 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l1145_114546


namespace bus_speed_excluding_stoppages_l1145_114523

theorem bus_speed_excluding_stoppages :
  ∀ (S : ℝ), (45 = (3 / 4) * S) → (S = 60) :=
by 
  intros S h
  sorry

end bus_speed_excluding_stoppages_l1145_114523


namespace number_of_distinct_rationals_l1145_114593

theorem number_of_distinct_rationals (L : ℕ) :
  L = 26 ↔
  (∃ (k : ℚ), |k| < 100 ∧ (∃ (x : ℤ), 7 * x^2 + k * x + 20 = 0)) :=
sorry

end number_of_distinct_rationals_l1145_114593


namespace required_speed_l1145_114568

-- The car covers 504 km in 6 hours initially.
def distance : ℕ := 504
def initial_time : ℕ := 6
def initial_speed : ℕ := distance / initial_time

-- The time that is 3/2 times the initial time.
def factor : ℚ := 3 / 2
def new_time : ℚ := initial_time * factor

-- The speed required to cover the same distance in the new time.
def new_speed : ℚ := distance / new_time

-- The proof statement
theorem required_speed : new_speed = 56 := by
  sorry

end required_speed_l1145_114568


namespace who_next_to_boris_l1145_114565

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l1145_114565


namespace at_least_one_is_zero_l1145_114509

theorem at_least_one_is_zero (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : false := by sorry

end at_least_one_is_zero_l1145_114509


namespace initial_distance_between_stations_l1145_114566

theorem initial_distance_between_stations
  (speedA speedB distanceA : ℝ)
  (rateA rateB : speedA = 40 ∧ speedB = 30)
  (dist_travelled : distanceA = 200) :
  (distanceA / speedA) * speedB + distanceA = 350 := by
  sorry

end initial_distance_between_stations_l1145_114566


namespace find_x_l1145_114534

theorem find_x (x : ℚ) (h : (3 * x - 6 + 4) / 7 = 15) : x = 107 / 3 :=
by
  sorry

end find_x_l1145_114534


namespace faye_complete_bouquets_l1145_114526

theorem faye_complete_bouquets :
  let roses_initial := 48
  let lilies_initial := 40
  let tulips_initial := 76
  let sunflowers_initial := 34
  let roses_wilted := 24
  let lilies_wilted := 10
  let tulips_wilted := 14
  let sunflowers_wilted := 7
  let roses_remaining := roses_initial - roses_wilted
  let lilies_remaining := lilies_initial - lilies_wilted
  let tulips_remaining := tulips_initial - tulips_wilted
  let sunflowers_remaining := sunflowers_initial - sunflowers_wilted
  let bouquets_roses := roses_remaining / 2
  let bouquets_lilies := lilies_remaining
  let bouquets_tulips := tulips_remaining / 3
  let bouquets_sunflowers := sunflowers_remaining
  let bouquets := min (min bouquets_roses bouquets_lilies) (min bouquets_tulips bouquets_sunflowers)
  bouquets = 12 :=
by
  sorry

end faye_complete_bouquets_l1145_114526


namespace sum_between_9p5_and_10_l1145_114516

noncomputable def sumMixedNumbers : ℚ :=
  (29 / 9) + (11 / 4) + (81 / 20)

theorem sum_between_9p5_and_10 :
  9.5 < sumMixedNumbers ∧ sumMixedNumbers < 10 :=
by
  sorry

end sum_between_9p5_and_10_l1145_114516


namespace jogger_distance_ahead_l1145_114543

def speed_jogger_kmph : ℕ := 9
def speed_train_kmph : ℕ := 45
def length_train_m : ℕ := 120
def time_to_pass_jogger_s : ℕ := 36

theorem jogger_distance_ahead :
  let relative_speed_mps := (speed_train_kmph - speed_jogger_kmph) * 1000 / 3600
  let distance_covered_m := relative_speed_mps * time_to_pass_jogger_s
  let jogger_distance_ahead : ℕ := distance_covered_m - length_train_m
  jogger_distance_ahead = 240 :=
by
  sorry

end jogger_distance_ahead_l1145_114543


namespace hyperbola_constants_sum_l1145_114514

noncomputable def hyperbola_asymptotes_equation (x y : ℝ) : Prop :=
  (y = 2 * x + 5) ∨ (y = -2 * x + 1)

noncomputable def hyperbola_passing_through (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 7)

theorem hyperbola_constants_sum
  (a b h k : ℝ) (ha : a > 0) (hb : b > 0)
  (H1 : ∀ x y : ℝ, hyperbola_asymptotes_equation x y)
  (H2 : hyperbola_passing_through 0 7)
  (H3 : h = -1)
  (H4 : k = 3)
  (H5 : a = 2 * b)
  (H6 : b = Real.sqrt 3) :
  a + h = 2 * Real.sqrt 3 - 1 :=
sorry

end hyperbola_constants_sum_l1145_114514


namespace player_B_wins_in_least_steps_l1145_114553

noncomputable def least_steps_to_win (n : ℕ) : ℕ :=
  n

theorem player_B_wins_in_least_steps (n : ℕ) (h_n : n > 0) :
  ∃ k, k = least_steps_to_win n ∧ k = n := by
  sorry

end player_B_wins_in_least_steps_l1145_114553


namespace range_g_minus_2x_l1145_114536

variable (g : ℝ → ℝ)
variable (x : ℝ)

axiom g_values : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → 
  (g x = x ∨ g x = x - 1 ∨ g x = x - 2 ∨ g x = x - 3 ∨ g x = x - 4)

axiom g_le_2x : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → g x ≤ 2 * x

theorem range_g_minus_2x : 
  Set.range (fun x => g x - 2 * x) = Set.Icc (-5 : ℝ) 0 :=
sorry

end range_g_minus_2x_l1145_114536


namespace general_formula_expression_of_k_l1145_114507

noncomputable def sequence_a : ℕ → ℤ
| 0     => 0 
| 1     => 0 
| 2     => -6
| n + 2 => 2 * (sequence_a (n + 1)) - (sequence_a n)

theorem general_formula :
  ∀ n, sequence_a n = 2 * n - 10 := sorry

def sequence_k : ℕ → ℕ
| 0     => 0 
| 1     => 8 
| n + 1 => 3 * 2 ^ n + 5

theorem expression_of_k (n : ℕ) :
  sequence_k (n + 1) = 3 * 2 ^ n + 5 := sorry

end general_formula_expression_of_k_l1145_114507


namespace Rachel_money_left_l1145_114574

theorem Rachel_money_left 
  (money_earned : ℕ)
  (lunch_fraction : ℚ)
  (clothes_percentage : ℚ)
  (dvd_cost : ℚ)
  (supplies_percentage : ℚ)
  (money_left : ℚ) :
  money_earned = 200 →
  lunch_fraction = 1 / 4 →
  clothes_percentage = 15 / 100 →
  dvd_cost = 24.50 →
  supplies_percentage = 10.5 / 100 →
  money_left = 74.50 :=
by
  intros h_money h_lunch h_clothes h_dvd h_supplies
  sorry

end Rachel_money_left_l1145_114574


namespace no_integer_solutions_l1145_114518

theorem no_integer_solutions (m n : ℤ) (h1 : m ^ 3 + n ^ 4 + 130 * m * n = 42875) (h2 : m * n ≥ 0) :
  false :=
sorry

end no_integer_solutions_l1145_114518


namespace loaves_per_hour_in_one_oven_l1145_114502

-- Define the problem constants and variables
def loaves_in_3_weeks : ℕ := 1740
def ovens : ℕ := 4
def weekday_hours : ℕ := 5
def weekend_hours : ℕ := 2
def weekdays_per_week : ℕ := 5
def weekends_per_week : ℕ := 2
def weeks : ℕ := 3

-- Calculate the total hours per week
def hours_per_week : ℕ := (weekdays_per_week * weekday_hours) + (weekends_per_week * weekend_hours)

-- Calculate the total oven-hours for 3 weeks
def total_oven_hours : ℕ := hours_per_week * ovens * weeks

-- Provide the proof statement
theorem loaves_per_hour_in_one_oven : (loaves_in_3_weeks = 5 * total_oven_hours) :=
by
  sorry -- Proof omitted

end loaves_per_hour_in_one_oven_l1145_114502


namespace large_bottle_water_amount_l1145_114548

noncomputable def sport_drink_water_amount (C V : ℝ) (prop_e : ℝ) : ℝ :=
  let F := C / 4
  let W := (C * 15)
  W

theorem large_bottle_water_amount (C V : ℝ) (prop_e : ℝ) (hc : C = 7) (hprop_e : prop_e = 0.05) : sport_drink_water_amount C V prop_e = 105 := by
  sorry

end large_bottle_water_amount_l1145_114548


namespace trip_duration_l1145_114598

/--
Given:
1. The car averages 30 miles per hour for the first 5 hours of the trip.
2. The car averages 42 miles per hour for the rest of the trip.
3. The average speed for the entire trip is 34 miles per hour.

Prove: 
The total duration of the trip is 7.5 hours.
-/
theorem trip_duration (t T : ℝ) (h1 : 150 + 42 * t = 34 * T) (h2 : T = 5 + t) : T = 7.5 :=
by
  sorry

end trip_duration_l1145_114598


namespace june_time_to_bernard_l1145_114521

theorem june_time_to_bernard (distance_Julia : ℝ) (time_Julia : ℝ) (distance_Bernard_June : ℝ) (time_Bernard : ℝ) (distance_June_Bernard : ℝ)
  (h1 : distance_Julia = 2) (h2 : time_Julia = 6) (h3 : distance_Bernard_June = 5) (h4 : time_Bernard = 15) (h5 : distance_June_Bernard = 7) :
  distance_June_Bernard / (distance_Julia / time_Julia) = 21 := by
    sorry

end june_time_to_bernard_l1145_114521


namespace total_sections_l1145_114500

theorem total_sections (boys girls : ℕ) (h_boys : boys = 408) (h_girls : girls = 240) :
  let gcd_boys_girls := Nat.gcd boys girls
  let sections_boys := boys / gcd_boys_girls
  let sections_girls := girls / gcd_boys_girls
  sections_boys + sections_girls = 27 :=
by
  sorry

end total_sections_l1145_114500


namespace part1_solution_part2_solution_part3_solution_l1145_114515

-- Part (1): Prove the solution of the system of equations 
theorem part1_solution (x y : ℝ) (h1 : x - y - 1 = 0) (h2 : 4 * (x - y) - y = 5) : 
  x = 0 ∧ y = -1 := 
sorry

-- Part (2): Prove the solution of the system of equations 
theorem part2_solution (x y : ℝ) (h1 : 2 * x - 3 * y - 2 = 0) 
  (h2 : (2 * x - 3 * y + 5) / 7 + 2 * y = 9) : 
  x = 7 ∧ y = 4 := 
sorry

-- Part (3): Prove the range of the parameter m
theorem part3_solution (m : ℕ) (h1 : 2 * (2 : ℝ) * x + y = (-3 : ℝ) * ↑m + 2) 
  (h2 : x + 2 * y = 7) (h3 : x + y > -5 / 6) : 
  m = 1 ∨ m = 2 ∨ m = 3 :=
sorry

end part1_solution_part2_solution_part3_solution_l1145_114515


namespace calculate_polynomial_value_l1145_114505

theorem calculate_polynomial_value :
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := 
by 
  sorry

end calculate_polynomial_value_l1145_114505


namespace min_value_frac_l1145_114599

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) : 
  (2 / x) + (1 / y) ≥ 9 / 2 :=
by
  sorry

end min_value_frac_l1145_114599


namespace solve_for_x_l1145_114577

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.8) : x = 71.7647 := 
by 
  sorry

end solve_for_x_l1145_114577


namespace least_pos_int_div_by_3_5_7_l1145_114533

/-
  Prove that the least positive integer divisible by the primes 3, 5, and 7 is 105.
-/

theorem least_pos_int_div_by_3_5_7 : ∃ (n : ℕ), n > 0 ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ n = 105 :=
by 
  sorry

end least_pos_int_div_by_3_5_7_l1145_114533


namespace evaluate_expression_l1145_114588

theorem evaluate_expression 
  (a c : ℝ)
  (h : a + c = 9) :
  (a * (-1)^2 + (-1) + c) = 8 := 
by 
  sorry

end evaluate_expression_l1145_114588


namespace variance_of_arithmetic_sequence_common_diff_3_l1145_114547

noncomputable def variance (ξ : List ℝ) : ℝ :=
  let n := ξ.length
  let mean := ξ.sum / n
  let var_sum := (ξ.map (fun x => (x - mean) ^ 2)).sum
  var_sum / n

def arithmetic_sequence (a1 : ℝ) (d : ℝ) (n : ℕ) : List ℝ :=
  List.range n |>.map (fun i => a1 + i * d)

theorem variance_of_arithmetic_sequence_common_diff_3 :
  ∀ (a1 : ℝ),
    variance (arithmetic_sequence a1 3 9) = 60 :=
by
  sorry

end variance_of_arithmetic_sequence_common_diff_3_l1145_114547


namespace range_of_m_range_of_x_l1145_114559

-- Define the function f(x) = m*x^2 - m*x - 6 + m
def f (m x : ℝ) : ℝ := m*x^2 - m*x - 6 + m

-- Proof for the first statement
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f m x < 0) ↔ m < 6 / 7 := 
sorry

-- Proof for the second statement
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end range_of_m_range_of_x_l1145_114559


namespace angle_terminal_side_l1145_114532

theorem angle_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 →
  α = 330 :=
by
  sorry

end angle_terminal_side_l1145_114532


namespace largest_unrepresentable_l1145_114596

theorem largest_unrepresentable (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1)
  : ¬ ∃ (x y z : ℕ), x * b * c + y * c * a + z * a * b = 2 * a * b * c - a * b - b * c - c * a :=
by
  -- The proof is omitted
  sorry

end largest_unrepresentable_l1145_114596


namespace sufficient_condition_implies_true_l1145_114589

variable {p q : Prop}

theorem sufficient_condition_implies_true (h : p → q) : (p → q) = true :=
by
  sorry

end sufficient_condition_implies_true_l1145_114589


namespace percentage_corresponding_to_120_l1145_114573

variable (x p : ℝ)

def forty_percent_eq_160 := (0.4 * x = 160)
def p_times_x_eq_120 := (p * x = 120)

theorem percentage_corresponding_to_120 (h₁ : forty_percent_eq_160 x) (h₂ : p_times_x_eq_120 x p) :
  p = 0.30 :=
sorry

end percentage_corresponding_to_120_l1145_114573


namespace problem_statement_l1145_114578

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem problem_statement : f (1 / 2) + f (-1 / 2) = 2 := sorry

end problem_statement_l1145_114578


namespace difference_of_one_third_and_five_l1145_114556

theorem difference_of_one_third_and_five (n : ℕ) (h : n = 45) : (n / 3) - 5 = 10 :=
by
  sorry

end difference_of_one_third_and_five_l1145_114556


namespace range_of_a_l1145_114554

noncomputable def f (x : ℝ) : ℝ := (1 / (1 + x^2)) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) ↔
  (1 / Real.exp 1 ≤ a ∧ a ≤ (2 + Real.log 3) / 3) :=
sorry

end range_of_a_l1145_114554


namespace total_viewing_time_l1145_114583

theorem total_viewing_time (video_length : ℕ) (num_videos : ℕ) (lila_speed_factor : ℕ) :
  video_length = 100 ∧ num_videos = 6 ∧ lila_speed_factor = 2 →
  (num_videos * (video_length / lila_speed_factor) + num_videos * video_length) = 900 :=
by
  sorry

end total_viewing_time_l1145_114583


namespace circle_center_and_radius_l1145_114517

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * y - 1 = 0 ↔ (x, y) = (0, 2) ∧ 5 = (0 - x)^2 + (2 - y)^2 :=
by sorry

end circle_center_and_radius_l1145_114517


namespace trajectory_equation_l1145_114527

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

end trajectory_equation_l1145_114527


namespace field_area_l1145_114535

def length : ℝ := 80 -- Length of the uncovered side
def total_fencing : ℝ := 97 -- Total fencing required

theorem field_area : ∃ (W L : ℝ), L = length ∧ 2 * W + L = total_fencing ∧ L * W = 680 := by
  sorry

end field_area_l1145_114535


namespace intersection_eq_l1145_114513

open Set

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 5 }
def B : Set ℝ := { -1, 2, 3, 6 }

-- State the proof problem
theorem intersection_eq : A ∩ B = {2, 3} := 
by 
-- placeholder for the proof steps
sorry

end intersection_eq_l1145_114513


namespace exist_monochromatic_equilateral_triangle_l1145_114530

theorem exist_monochromatic_equilateral_triangle 
  (color : ℝ × ℝ → ℕ) 
  (h_color : ∀ p : ℝ × ℝ, color p = 0 ∨ color p = 1) : 
  ∃ (A B C : ℝ × ℝ), (dist A B = dist B C) ∧ (dist B C = dist C A) ∧ (color A = color B ∧ color B = color C) :=
sorry

end exist_monochromatic_equilateral_triangle_l1145_114530


namespace prove_inequality_l1145_114538

-- Define the function properties
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Function properties as given in the problem
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def decreasing_on_nonneg (f : ℝ → ℝ) := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- The main theorem statement
theorem prove_inequality (h_even : even_function f) (h_dec : decreasing_on_nonneg f) :
  f (-3 / 4) ≥ f (a^2 - a + 1) :=
sorry

end prove_inequality_l1145_114538


namespace number_of_valid_three_digit_numbers_l1145_114584

theorem number_of_valid_three_digit_numbers : 
  (∃ A B C : ℕ, 
      (100 * A + 10 * B + C + 297 = 100 * C + 10 * B + A) ∧ 
      (0 ≤ A ∧ A ≤ 9) ∧ 
      (0 ≤ B ∧ B ≤ 9) ∧ 
      (0 ≤ C ∧ C ≤ 9)) 
    ∧ (number_of_such_valid_numbers = 70) :=
by
  sorry

def number_of_such_valid_numbers : ℕ := 
  sorry

end number_of_valid_three_digit_numbers_l1145_114584


namespace minimum_value_inequality_l1145_114591

theorem minimum_value_inequality (x y z : ℝ) (hx : 2 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 5) :
    (x - 2)^2 + (y / x - 2)^2 + (z / y - 2)^2 + (5 / z - 2)^2 ≥ 4 * (Real.sqrt (Real.sqrt 5) - 2)^2 := 
    sorry

end minimum_value_inequality_l1145_114591


namespace perimeter_of_region_is_70_l1145_114528

-- Define the given conditions
def area_of_region (total_area : ℝ) (num_squares : ℕ) : Prop :=
  total_area = 392 ∧ num_squares = 8

def side_length_of_square (area : ℝ) (side_length : ℝ) : Prop :=
  area = side_length^2 ∧ side_length = 7

def perimeter_of_region (num_squares : ℕ) (side_length : ℝ) (perimeter : ℝ) : Prop :=
  perimeter = 8 * side_length + 2 * side_length ∧ perimeter = 70

-- Statement to prove
theorem perimeter_of_region_is_70 :
  ∀ (total_area : ℝ) (num_squares : ℕ), 
    area_of_region total_area num_squares →
    ∃ (side_length : ℝ) (perimeter : ℝ), 
      side_length_of_square (total_area / num_squares) side_length ∧
      perimeter_of_region num_squares side_length perimeter :=
by {
  sorry
}

end perimeter_of_region_is_70_l1145_114528


namespace distance_from_A_to_B_l1145_114555

theorem distance_from_A_to_B (D : ℝ) :
  (∃ D, (∀ tC, tC = D / 30) 
      ∧ (∀ tD, tD = D / 48 ∧ tD < (D / 30 - 1.5))
      ∧ D = 120) :=
by
  sorry

end distance_from_A_to_B_l1145_114555


namespace problem_statement_l1145_114529

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

def candidate_function (x : ℝ) : ℝ :=
  x * |x|

theorem problem_statement : is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end problem_statement_l1145_114529


namespace car_trip_distance_l1145_114531

theorem car_trip_distance (speed_first_car speed_second_car : ℝ) (time_first_car time_second_car distance_first_car distance_second_car : ℝ) 
  (h_speed_first : speed_first_car = 30)
  (h_time_first : time_first_car = 1.5)
  (h_speed_second : speed_second_car = 60)
  (h_time_second : time_second_car = 1.3333)
  (h_distance_first : distance_first_car = speed_first_car * time_first_car)
  (h_distance_second : distance_second_car = speed_second_car * time_second_car) :
  distance_first_car = 45 :=
by
  sorry

end car_trip_distance_l1145_114531


namespace daniel_spent_2290_l1145_114567

theorem daniel_spent_2290 (total_games: ℕ) (price_12_games count_price_12: ℕ) 
  (price_7_games frac_price_7: ℕ) (price_3_games: ℕ) 
  (count_price_7: ℕ) (h1: total_games = 346)
  (h2: count_price_12 = 80) (h3: price_12_games = 12)
  (h4: frac_price_7 = 50) (h5: price_7_games = 7)
  (h6: price_3_games = 3) (h7: count_price_7 = (frac_price_7 * (total_games - count_price_12)) / 100):
  (count_price_12 * price_12_games) + (count_price_7 * price_7_games) + ((total_games - count_price_12 - count_price_7) * price_3_games) = 2290 := 
by
  sorry

end daniel_spent_2290_l1145_114567


namespace stayed_days_calculation_l1145_114558

theorem stayed_days_calculation (total_cost : ℕ) (charge_1st_week : ℕ) (charge_additional_week : ℕ) (first_week_days : ℕ) :
  total_cost = 302 ∧ charge_1st_week = 18 ∧ charge_additional_week = 11 ∧ first_week_days = 7 →
  ∃ D : ℕ, D = 23 :=
by {
  sorry
}

end stayed_days_calculation_l1145_114558


namespace inequality_not_hold_l1145_114581

theorem inequality_not_hold (x y : ℝ) (h : x > y) : ¬ (1 - x > 1 - y) :=
by
  -- condition and given statements
  sorry

end inequality_not_hold_l1145_114581


namespace induction_proof_l1145_114575

-- Given conditions and definitions
def plane_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

-- The induction hypothesis for k ≥ 2
def induction_step (k : ℕ) (h : 2 ≤ k) : Prop :=
  plane_parts (k + 1) - plane_parts k = k + 1

-- The complete statement we want to prove
theorem induction_proof (k : ℕ) (h : 2 ≤ k) : induction_step k h := by
  sorry

end induction_proof_l1145_114575


namespace equilateral_triangle_side_length_l1145_114595
noncomputable def equilateral_triangle_side (r R : ℝ) (h : R > r) : ℝ :=
  r * R * Real.sqrt 3 / (Real.sqrt (r ^ 2 - r * R + R ^ 2))

theorem equilateral_triangle_side_length
  (r R : ℝ) (hRgr : R > r) :
  ∃ a, a = equilateral_triangle_side r R hRgr :=
sorry

end equilateral_triangle_side_length_l1145_114595


namespace statement_A_statement_A_statement_C_statement_D_l1145_114512

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem statement_A (x : ℝ) (hx : x > 1) : f x > 0 := sorry

theorem statement_A' (x : ℝ) (hx : 0 < x ∧ x < 1) : f x < 0 := sorry

theorem statement_C : Set.range f = Set.Ici (-1 / (2 * Real.exp 1)) := sorry

theorem statement_D (x : ℝ) : f x ≥ x - 1 := sorry

end statement_A_statement_A_statement_C_statement_D_l1145_114512


namespace triangle_area_l1145_114560

theorem triangle_area (h : ℝ) (hypotenuse : h = 12) (angle : ∃θ : ℝ, θ = 30 ∧ θ = 30) :
  ∃ (A : ℝ), A = 18 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l1145_114560


namespace mean_of_observations_decreased_l1145_114580

noncomputable def original_mean : ℕ := 200

theorem mean_of_observations_decreased (S' : ℕ) (M' : ℕ) (n : ℕ) (d : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : M' = 185)
  (h4 : S' = M' * n)
  : original_mean = (S' + d * n) / n :=
by
  rw [original_mean]
  sorry

end mean_of_observations_decreased_l1145_114580


namespace Emily_cleaning_time_in_second_room_l1145_114522

/-
Lilly, Fiona, Jack, and Emily are cleaning 3 rooms.
For the first room: Lilly and Fiona together: 1/4 of the time, Jack: 1/3 of the time, Emily: the rest of the time.
In the second room: Jack: 25%, Emily: 25%, Lilly and Fiona: the remaining 50%.
In the third room: Emily: 40%, Lilly: 20%, Jack: 20%, Fiona: 20%.
Total time for all rooms: 12 hours.

Prove that the total time Emily spent cleaning in the second room is 60 minutes.
-/

theorem Emily_cleaning_time_in_second_room :
  let total_time := 12 -- total time in hours
  let time_per_room := total_time / 3 -- time per room in hours
  let time_per_room_minutes := time_per_room * 60 -- time per room in minutes
  let emily_cleaning_percentage := 0.25 -- Emily's cleaning percentage in the second room
  let emily_cleaning_time := emily_cleaning_percentage * time_per_room_minutes -- cleaning time in minutes
  emily_cleaning_time = 60 := by
  sorry

end Emily_cleaning_time_in_second_room_l1145_114522


namespace find_k_for_line_l1145_114571

theorem find_k_for_line (k : ℝ) : (2 * k * (-1/2) + 1 = -7 * 3) → k = 22 :=
by
  intro h
  sorry

end find_k_for_line_l1145_114571


namespace first_player_wins_l1145_114590

theorem first_player_wins :
  ∀ (sticks : ℕ), (sticks = 1) →
  (∀ (break_rule : ℕ → ℕ → Prop),
  (∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z → break_rule x y → break_rule x z)
  → (∃ n : ℕ, n % 3 = 0 ∧ break_rule n (n + 1) → ∃ t₁ t₂ t₃ : ℕ, t₁ = t₂ ∧ t₂ = t₃ ∧ t₁ + t₂ + t₃ = n))
  → (∃ w : ℕ, w = 1) := sorry

end first_player_wins_l1145_114590


namespace original_quantity_l1145_114524

theorem original_quantity (x : ℕ) : 
  (532 * x - 325 * x = 1065430) -> x = 5148 := 
by
  intro h
  sorry

end original_quantity_l1145_114524


namespace proof_problem1_proof_problem2_proof_problem3_proof_problem4_l1145_114582

noncomputable def problem1 : Prop := 
  2500 * (1/10000) = 0.25

noncomputable def problem2 : Prop := 
  20 * (1/100) = 0.2

noncomputable def problem3 : Prop := 
  45 * (1/60) = 3/4

noncomputable def problem4 : Prop := 
  1250 * (1/10000) = 0.125

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

theorem proof_problem3 : problem3 := by
  sorry

theorem proof_problem4 : problem4 := by
  sorry

end proof_problem1_proof_problem2_proof_problem3_proof_problem4_l1145_114582


namespace international_sales_correct_option_l1145_114511

theorem international_sales_correct_option :
  (∃ (A B C D : String),
     A = "who" ∧
     B = "what" ∧
     C = "whoever" ∧
     D = "whatever" ∧
     (∃ x, x = C → "Could I speak to " ++ x ++ " is in charge of International Sales please?" = "Could I speak to whoever is in charge of International Sales please?")) :=
sorry

end international_sales_correct_option_l1145_114511


namespace infinite_series_sum_l1145_114557

theorem infinite_series_sum :
  (∑' n : ℕ, (3:ℝ)^n / (1 + (3:ℝ)^n + (3:ℝ)^(n+1) + (3:ℝ)^(2*n+2))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l1145_114557


namespace arithmetic_contains_geometric_l1145_114540

theorem arithmetic_contains_geometric {a b : ℚ} (h : a^2 + b^2 ≠ 0) :
  ∃ (c q : ℚ) (f : ℕ → ℚ), (∀ n, f n = c * q^n) ∧ (∀ n, f n = a + b * n) := 
sorry

end arithmetic_contains_geometric_l1145_114540


namespace jason_total_payment_l1145_114550

def total_cost (shorts jacket shoes socks tshirts : ℝ) : ℝ :=
  shorts + jacket + shoes + socks + tshirts

def discount_amount (total : ℝ) (discount_rate : ℝ) : ℝ :=
  total * discount_rate

def total_after_discount (total discount : ℝ) : ℝ :=
  total - discount

def sales_tax_amount (total : ℝ) (tax_rate : ℝ) : ℝ :=
  total * tax_rate

def final_amount (total after_discount tax : ℝ) : ℝ :=
  after_discount + tax

theorem jason_total_payment :
  let shorts := 14.28
  let jacket := 4.74
  let shoes := 25.95
  let socks := 6.80
  let tshirts := 18.36
  let discount_rate := 0.15
  let tax_rate := 0.07
  let total := total_cost shorts jacket shoes socks tshirts
  let discount := discount_amount total discount_rate
  let after_discount := total_after_discount total discount
  let tax := sales_tax_amount after_discount tax_rate
  let final := final_amount total after_discount tax
  final = 63.78 :=
by
  sorry

end jason_total_payment_l1145_114550


namespace james_after_paying_debt_l1145_114549

variables (L J A : Real)

-- Define the initial conditions
def total_money : Real := 300
def debt : Real := 25
def total_with_debt : Real := total_money + debt

axiom h1 : J = A + 40
axiom h2 : J + A = total_with_debt

-- Prove that James owns $170 after paying off half of Lucas' debt
theorem james_after_paying_debt (h1 : J = A + 40) (h2 : J + A = total_with_debt) :
  (J - (debt / 2)) = 170 :=
  sorry

end james_after_paying_debt_l1145_114549


namespace ants_first_group_count_l1145_114508

theorem ants_first_group_count :
    ∃ x : ℕ, 
        (∀ (w1 c1 a1 t1 w2 c2 a2 t2 : ℕ),
          w1 = 10 ∧ c1 = 600 ∧ a1 = x ∧ t1 = 5 ∧
          w2 = 5 ∧ c2 = 960 ∧ a2 = 20 ∧ t2 = 3 ∧ 
          (w1 * c1) / t1 = 1200 / a1 ∧ (w2 * c2) / t2 = 1600 / 20 →
             x = 15)
:= sorry

end ants_first_group_count_l1145_114508


namespace new_average_daily_production_l1145_114597

theorem new_average_daily_production (n : ℕ) (avg_past_n_days : ℕ) (today_production : ℕ) (h1 : avg_past_n_days = 50) (h2 : today_production = 90) (h3 : n = 9) : 
  (avg_past_n_days * n + today_production) / (n + 1) = 54 := 
by
  sorry

end new_average_daily_production_l1145_114597


namespace lincoln_one_way_fare_l1145_114576

-- Define the given conditions as assumptions
variables (x : ℝ) (days : ℝ) (total_cost : ℝ) (trips_per_day : ℝ)

-- State the conditions
axiom condition1 : days = 9
axiom condition2 : total_cost = 288
axiom condition3 : trips_per_day = 2

-- The theorem we want to prove based on the conditions
theorem lincoln_one_way_fare (h1 : total_cost = days * trips_per_day * x) : x = 16 :=
by
  -- We skip the proof for the sake of this exercise
  sorry

end lincoln_one_way_fare_l1145_114576


namespace wave_propagation_l1145_114569

def accum (s : String) : String :=
  String.join (List.intersperse "-" (s.data.enum.map (λ (i : Nat × Char) =>
    String.mk [i.2.toUpper] ++ String.mk (List.replicate i.1 i.2.toLower))))

theorem wave_propagation (s : String) :
  s = "dremCaheя" → accum s = "D-Rr-Eee-Mmmm-Ccccc-Aaaaaa-Hhhhhhh-Eeeeeeee-Яяяяяяяяя" :=
  by
  intro h
  rw [h]
  sorry

end wave_propagation_l1145_114569


namespace total_number_of_people_l1145_114551

def total_people_at_park(hikers bike_riders : Nat) : Nat :=
  hikers + bike_riders

theorem total_number_of_people 
  (bike_riders : Nat)
  (hikers : Nat)
  (hikers_eq_bikes_plus_178 : hikers = bike_riders + 178)
  (bikes_eq_249 : bike_riders = 249) :
  total_people_at_park hikers bike_riders = 676 :=
by
  sorry

end total_number_of_people_l1145_114551


namespace find_s_l1145_114542

theorem find_s (n r s c d : ℚ) 
  (h1 : Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 3 = 0) 
  (h2 : c * d = 3)
  (h3 : Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s = 
        Polynomial.C (c + d⁻¹) * Polynomial.C (d + c⁻¹)) : 
  s = 16 / 3 := 
by
  sorry

end find_s_l1145_114542


namespace warmup_puzzle_time_l1145_114563

theorem warmup_puzzle_time (W : ℕ) (H : W + 3 * W + 3 * W = 70) : W = 10 :=
by
  sorry

end warmup_puzzle_time_l1145_114563


namespace three_digit_number_l1145_114586

theorem three_digit_number (m : ℕ) : (300 * m + 10 * m + (m - 1)) = (311 * m - 1) :=
by 
  sorry

end three_digit_number_l1145_114586


namespace least_add_to_divisible_least_subtract_to_divisible_l1145_114561

theorem least_add_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (a : ℤ) : 
  n = 1100 → d = 37 → r = n % d → a = d - r → (n + a) % d = 0 :=
by sorry

theorem least_subtract_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (s : ℤ) : 
  n = 1100 → d = 37 → r = n % d → s = r → (n - s) % d = 0 :=
by sorry

end least_add_to_divisible_least_subtract_to_divisible_l1145_114561


namespace find_m_value_l1145_114587

-- Define the points P and Q and the condition of perpendicularity
def points_PQ (m : ℝ) : Prop := 
  let P := (-2, m)
  let Q := (m, 4)
  let slope_PQ := (m - 4) / (-2 - m)
  slope_PQ * (-1) = -1

-- Problem statement: Find the value of m such that the above condition holds
theorem find_m_value : ∃ (m : ℝ), points_PQ m ∧ m = 1 :=
by sorry

end find_m_value_l1145_114587


namespace cost_price_of_article_l1145_114552

theorem cost_price_of_article (M : ℝ) (SP : ℝ) (C : ℝ) 
  (hM : M = 65)
  (hSP : SP = 0.95 * M)
  (hProfit : SP = 1.30 * C) : 
  C = 47.50 :=
by 
  sorry

end cost_price_of_article_l1145_114552


namespace fill_time_of_three_pipes_l1145_114539

def rate (hours : ℕ) : ℚ := 1 / hours

def combined_rate : ℚ :=
  rate 12 + rate 15 + rate 20

def time_to_fill (rate : ℚ) : ℚ :=
  1 / rate

theorem fill_time_of_three_pipes :
  time_to_fill combined_rate = 5 := by
  sorry

end fill_time_of_three_pipes_l1145_114539


namespace find_a_l1145_114520

theorem find_a (a b c : ℝ) (h1 : ∀ x, x = 2 → y = 5) (h2 : ∀ x, x = 3 → y = 7) :
  a = 2 :=
sorry

end find_a_l1145_114520


namespace cows_total_l1145_114510

theorem cows_total (M F : ℕ) 
  (h1 : F = 2 * M) 
  (h2 : F / 2 = M / 2 + 50) : 
  M + F = 300 :=
by
  sorry

end cows_total_l1145_114510


namespace adrianna_gum_pieces_l1145_114594

-- Definitions based on conditions
def initial_gum_pieces : ℕ := 10
def additional_gum_pieces : ℕ := 3
def friends_count : ℕ := 11

-- Expression to calculate the final pieces of gum
def total_gum_pieces : ℕ := initial_gum_pieces + additional_gum_pieces
def gum_left : ℕ := total_gum_pieces - friends_count

-- Lean statement we want to prove
theorem adrianna_gum_pieces: gum_left = 2 := 
by 
  sorry

end adrianna_gum_pieces_l1145_114594


namespace intersection_A_B_l1145_114506

variable (x : ℝ)

def setA : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersection_A_B :
  { x | x^2 - 4*x - 5 < 0 } ∩ { x | -2 < x ∧ x < 2 } = { x | -1 < x ∧ x < 2 } :=
by
  -- Here would be the proof, but we use sorry to skip it
  sorry

end intersection_A_B_l1145_114506


namespace find_x_l1145_114572

theorem find_x (h : ℝ → ℝ)
  (H1 : ∀x, h (3*x - 2) = 5*x + 6) :
  (∀x, h x = 2*x - 1) → x = 31 :=
by
  sorry

end find_x_l1145_114572


namespace quadratic_polynomials_perfect_square_l1145_114579

variables {x y p q a b c : ℝ}

theorem quadratic_polynomials_perfect_square (h1 : ∃ a, x^2 + p * x + q = (x + a) * (x + a))
  (h2 : ∃ a b, a^2 * x^2 + 2 * b^2 * x * y + c^2 * y^2 = (a * x + b * y) * (a * x + b * y)) :
  q = (p^2 / 4) ∧ b^2 = a * c :=
by
  sorry

end quadratic_polynomials_perfect_square_l1145_114579


namespace selection_including_both_genders_is_34_l1145_114564

def count_ways_to_select_students_with_conditions (total_students boys girls select_students : ℕ) : ℕ :=
  if total_students = 7 ∧ boys = 4 ∧ girls = 3 ∧ select_students = 4 then
    (Nat.choose total_students select_students) - 1
  else
    0

theorem selection_including_both_genders_is_34 :
  count_ways_to_select_students_with_conditions 7 4 3 4 = 34 :=
by
  -- The proof would go here
  sorry

end selection_including_both_genders_is_34_l1145_114564


namespace revenue_decrease_percent_l1145_114545

theorem revenue_decrease_percent (T C : ℝ) (hT_pos : T > 0) (hC_pos : C > 0) :
  let new_T := 0.75 * T
  let new_C := 1.10 * C
  let original_revenue := T * C
  let new_revenue := new_T * new_C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 17.5 := 
by {
  sorry
}

end revenue_decrease_percent_l1145_114545


namespace relationship_between_a_b_c_l1145_114541

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) - 1
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2)
noncomputable def c : ℝ := f (Real.log (1/4) / Real.log 2)

theorem relationship_between_a_b_c : a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l1145_114541


namespace original_board_is_120_l1145_114585

-- Define the two given conditions
def S : ℕ := 35
def L : ℕ := 2 * S + 15

-- Define the length of the original board
def original_board_length : ℕ := S + L

-- The theorem we want to prove
theorem original_board_is_120 : original_board_length = 120 :=
by
  -- Skipping the actual proof
  sorry

end original_board_is_120_l1145_114585


namespace min_colors_needed_l1145_114537

theorem min_colors_needed (n : ℕ) (h : n + n.choose 2 ≥ 12) : n = 5 :=
sorry

end min_colors_needed_l1145_114537


namespace number_of_triangles_l1145_114503

theorem number_of_triangles (m : ℕ) (h : m > 0) :
  ∃ n : ℕ, n = (m * (m + 1)) / 2 :=
by sorry

end number_of_triangles_l1145_114503


namespace find_first_number_l1145_114519

theorem find_first_number (n : ℝ) (h1 : n / 14.5 = 175) :
  n = 2537.5 :=
by 
  sorry

end find_first_number_l1145_114519


namespace exterior_angle_of_parallel_lines_l1145_114544

theorem exterior_angle_of_parallel_lines (A B C x y : ℝ) (hAx : A = 40) (hBx : B = 90) (hCx : C = 40)
  (h_parallel : true)
  (h_triangle : x = 180 - A - C)
  (h_exterior_angle : y = 180 - x) :
  y = 80 := 
by
  sorry

end exterior_angle_of_parallel_lines_l1145_114544


namespace place_value_diff_7669_l1145_114504

theorem place_value_diff_7669 :
  let a := 6 * 10
  let b := 6 * 100
  b - a = 540 :=
by
  let a := 6 * 10
  let b := 6 * 100
  have h : b - a = 540 := by sorry
  exact h

end place_value_diff_7669_l1145_114504


namespace pieces_per_package_l1145_114525

-- Define Robin's packages
def numGumPackages := 28
def numCandyPackages := 14

-- Define total number of pieces
def totalPieces := 7

-- Define the total number of packages
def totalPackages := numGumPackages + numCandyPackages

-- Define the expected number of pieces per package as the theorem to prove
theorem pieces_per_package : (totalPieces / totalPackages) = 1/6 := by
  sorry

end pieces_per_package_l1145_114525
