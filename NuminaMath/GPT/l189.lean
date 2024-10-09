import Mathlib

namespace perpendicular_condition_l189_18987

noncomputable def line := ℝ → (ℝ × ℝ × ℝ)
noncomputable def plane := (ℝ × ℝ × ℝ) → Prop

variable {l m : line}
variable {α : plane}

-- l and m are two different lines
axiom lines_are_different : l ≠ m

-- m is parallel to the plane α
axiom m_parallel_alpha : ∀ t : ℝ, α (m t)

-- Prove that l perpendicular to α is a sufficient but not necessary condition for l perpendicular to m
theorem perpendicular_condition :
  (∀ t : ℝ, ¬ α (l t)) → (∀ t₁ t₂ : ℝ, (l t₁) ≠ (m t₂)) ∧ ¬ (∀ t : ℝ, ¬ α (l t)) :=
by 
  sorry

end perpendicular_condition_l189_18987


namespace value_of_x_l189_18989

theorem value_of_x (x y : ℝ) :
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 :=
by
  intro h
  sorry

end value_of_x_l189_18989


namespace dog_speed_correct_l189_18921

-- Definitions of the conditions
def football_field_length_yards : ℕ := 200
def total_football_fields : ℕ := 6
def yards_to_feet_conversion : ℕ := 3
def time_to_fetch_minutes : ℕ := 9

-- The goal is to find the dog's speed in feet per minute
def dog_speed_feet_per_minute : ℕ :=
  (total_football_fields * football_field_length_yards * yards_to_feet_conversion) / time_to_fetch_minutes

-- Statement for the proof
theorem dog_speed_correct : dog_speed_feet_per_minute = 400 := by
  sorry

end dog_speed_correct_l189_18921


namespace number_of_new_players_l189_18941

variable (returning_players : ℕ)
variable (groups : ℕ)
variable (players_per_group : ℕ)

theorem number_of_new_players
  (h1 : returning_players = 6)
  (h2 : groups = 9)
  (h3 : players_per_group = 6) :
  (groups * players_per_group - returning_players = 48) := 
sorry

end number_of_new_players_l189_18941


namespace basketball_team_starters_l189_18929

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_team_starters :
  choose 4 2 * choose 14 4 = 6006 := by
  sorry

end basketball_team_starters_l189_18929


namespace binom_2024_1_l189_18976

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_2024_1 : binomial 2024 1 = 2024 := by
  sorry

end binom_2024_1_l189_18976


namespace side_lengths_of_triangle_l189_18900

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

theorem side_lengths_of_triangle (m : ℝ) (a b c : ℝ) 
  (h1 : f m a > 0) 
  (h2 : f m b > 0) 
  (h3 : f m c > 0) 
  (h4 : f m a + f m b > f m c)
  (h5 : f m a + f m c > f m b)
  (h6 : f m b + f m c > f m a) :
  m ∈ Set.Ioo (7/5 : ℝ) 5 :=
sorry

end side_lengths_of_triangle_l189_18900


namespace closest_number_to_fraction_l189_18931

theorem closest_number_to_fraction (x : ℝ) : 
  (abs (x - 2000) < abs (x - 1500)) ∧ 
  (abs (x - 2000) < abs (x - 2500)) ∧ 
  (abs (x - 2000) < abs (x - 3000)) ∧ 
  (abs (x - 2000) < abs (x - 3500)) :=
by
  let x := 504 / 0.252
  sorry

end closest_number_to_fraction_l189_18931


namespace max_value_of_a_max_value_reached_l189_18974

theorem max_value_of_a (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  a ≤ Real.sqrt 6 / 3 :=
by
  sorry

theorem max_value_reached (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  ∃ a, a = Real.sqrt 6 / 3 :=
by
  sorry

end max_value_of_a_max_value_reached_l189_18974


namespace xyz_value_l189_18932

theorem xyz_value
  (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 14 / 3 :=
  sorry

end xyz_value_l189_18932


namespace proportion_solution_l189_18935

theorem proportion_solution (x: ℕ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_solution_l189_18935


namespace roots_of_equation_l189_18975

theorem roots_of_equation (x : ℝ) : ((x - 5) ^ 2 = 2 * (x - 5)) ↔ (x = 5 ∨ x = 7) := by
sorry

end roots_of_equation_l189_18975


namespace four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l189_18928

-- Number of four-digit numbers greater than 3999 such that the product of the middle two digits > 12 is 4260
theorem four_digit_numbers_greater_3999_with_middle_product_exceeding_12
  {d1 d2 d3 d4 : ℕ}
  (h1 : 4 ≤ d1 ∧ d1 ≤ 9)
  (h2 : 0 ≤ d4 ∧ d4 ≤ 9)
  (h3 : 1 ≤ d2 ∧ d2 ≤ 9)
  (h4 : 1 ≤ d3 ∧ d3 ≤ 9)
  (h5 : d2 * d3 > 12) :
  (6 * 71 * 10 = 4260) :=
by
  sorry

end four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l189_18928


namespace smallest_integer_mod_conditions_l189_18920

theorem smallest_integer_mod_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 3) ∧ (x % 3 = 2) ∧ (∀ y : ℕ, (y % 4 = 3) ∧ (y % 3 = 2) → x ≤ y) ∧ x = 11 :=
by
  sorry

end smallest_integer_mod_conditions_l189_18920


namespace value_of_m_l189_18925

theorem value_of_m
  (m : ℝ)
  (a : ℝ × ℝ := (-1, 3))
  (b : ℝ × ℝ := (m, m - 2))
  (collinear : a.1 * b.2 = a.2 * b.1) :
  m = 1 / 2 :=
sorry

end value_of_m_l189_18925


namespace convex_power_function_l189_18954

theorem convex_power_function (n : ℕ) (h : 0 < n) : 
  (∀ x : ℝ, 0 < x → 0 ≤ (↑n * (↑n - 1) * x ^ (↑n - 2))) ↔ (n = 1 ∨ ∃ k : ℕ, n = 2 * k) :=
by
  sorry

end convex_power_function_l189_18954


namespace problem1_problem2_l189_18946

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then (1/2)^x - 2 else (x - 2) * (|x| - 1)

theorem problem1 : f (f (-2)) = 0 := by 
  sorry

theorem problem2 (x : ℝ) (h : f x ≥ 2) : x ≥ 3 ∨ x = 0 := by
  sorry

end problem1_problem2_l189_18946


namespace solve_for_n_l189_18960

theorem solve_for_n (n : ℕ) : 4^8 = 16^n → n = 4 :=
by
  sorry

end solve_for_n_l189_18960


namespace animal_fish_consumption_l189_18996

-- Definitions for the daily consumption of each animal
def daily_trout_polar1 := 0.2
def daily_salmon_polar1 := 0.4

def daily_trout_polar2 := 0.3
def daily_salmon_polar2 := 0.5

def daily_trout_polar3 := 0.25
def daily_salmon_polar3 := 0.45

def daily_trout_sealion1 := 0.1
def daily_salmon_sealion1 := 0.15

def daily_trout_sealion2 := 0.2
def daily_salmon_sealion2 := 0.25

-- Calculate total daily consumption
def total_daily_trout :=
  daily_trout_polar1 + daily_trout_polar2 + daily_trout_polar3 + daily_trout_sealion1 + daily_trout_sealion2

def total_daily_salmon :=
  daily_salmon_polar1 + daily_salmon_polar2 + daily_salmon_polar3 + daily_salmon_sealion1 + daily_salmon_sealion2

-- Calculate total monthly consumption
def total_monthly_trout := total_daily_trout * 30
def total_monthly_salmon := total_daily_salmon * 30

-- Total monthly fish bucket consumption
def total_monthly_fish := total_monthly_trout + total_monthly_salmon

-- The statement to prove the total consumption
theorem animal_fish_consumption : total_monthly_fish = 84 := by
  sorry

end animal_fish_consumption_l189_18996


namespace problem_I_problem_II_l189_18913

-- Problem (I)
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) : 
  (f (x + 8) ≥ 10 - f x) ↔ (x ≤ -10 ∨ x ≥ 0) :=
sorry

-- Problem (II)
theorem problem_II (x y : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) 
(h_abs_x : |x| > 1) (h_abs_y : |y| < 1) :
  f y < |x| * f (y / x^2) :=
sorry

end problem_I_problem_II_l189_18913


namespace solution_set_of_inequality_l189_18906

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x | 3 * a < x ∧ x < -a} :=
sorry

end solution_set_of_inequality_l189_18906


namespace t_shirts_in_two_hours_l189_18973

-- Definitions for the conditions
def first_hour_rate : Nat := 12
def second_hour_rate : Nat := 6

-- Main statement to prove
theorem t_shirts_in_two_hours : 
  (60 / first_hour_rate + 60 / second_hour_rate) = 15 := by
  sorry

end t_shirts_in_two_hours_l189_18973


namespace value_of_a_l189_18959

theorem value_of_a
    (a b : ℝ)
    (h₁ : 0 < a ∧ 0 < b)
    (h₂ : a + b = 1)
    (h₃ : 21 * a^5 * b^2 = 35 * a^4 * b^3) :
    a = 5 / 8 :=
by
  sorry

end value_of_a_l189_18959


namespace find_number_satisfy_equation_l189_18903

theorem find_number_satisfy_equation (x : ℝ) :
  9 - x / 7 * 5 + 10 = 13.285714285714286 ↔ x = -20 := sorry

end find_number_satisfy_equation_l189_18903


namespace find_a1_and_d_l189_18979

-- Given conditions
variables {a : ℕ → ℤ} 
variables {a1 d : ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem find_a1_and_d 
  (h1 : is_arithmetic_sequence a a1 d)
  (h2 : (a 3) * (a 7) = -16)
  (h3 : (a 4) + (a 6) = 0)
  : (a1 = -8 ∧ d = 2) ∨ (a1 = 8 ∧ d = -2) :=
sorry

end find_a1_and_d_l189_18979


namespace ball_box_distribution_l189_18971

theorem ball_box_distribution : (∃ (f : Fin 4 → Fin 2), true) ∧ (∀ (f : Fin 4 → Fin 2), true) → ∃ (f : Fin 4 → Fin 2), true ∧ f = 16 :=
by sorry

end ball_box_distribution_l189_18971


namespace abs_diff_of_solutions_l189_18968

theorem abs_diff_of_solutions (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
sorry

end abs_diff_of_solutions_l189_18968


namespace max_marks_l189_18942

theorem max_marks (M : ℝ) (pass_percent : ℝ) (obtained_marks : ℝ) (failed_by : ℝ) (pass_marks : ℝ) 
  (h1 : pass_percent = 0.40) 
  (h2 : obtained_marks = 150) 
  (h3 : failed_by = 50) 
  (h4 : pass_marks = 200) 
  (h5 : pass_marks = obtained_marks + failed_by) 
  : M = 500 :=
by 
  -- Placeholder for the proof
  sorry

end max_marks_l189_18942


namespace calculate_gf3_l189_18981

def f (x : ℕ) : ℕ := x^3 - 1
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem calculate_gf3 : g (f 3) = 2056 := by
  sorry

end calculate_gf3_l189_18981


namespace cottage_cost_per_hour_l189_18948

-- Define the conditions
def jack_payment : ℝ := 20
def jill_payment : ℝ := 20
def total_payment : ℝ := jack_payment + jill_payment
def rental_duration : ℝ := 8

-- Define the theorem to be proved
theorem cottage_cost_per_hour : (total_payment / rental_duration) = 5 := by
  sorry

end cottage_cost_per_hour_l189_18948


namespace mid_point_between_fractions_l189_18915

theorem mid_point_between_fractions : (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end mid_point_between_fractions_l189_18915


namespace solve_for_x_l189_18933

theorem solve_for_x (x : ℝ) (h : (15 - 2 + (x / 1)) / 2 * 8 = 77) : x = 6.25 :=
by
  sorry

end solve_for_x_l189_18933


namespace probability_sum_3_or_7_or_10_l189_18961

-- Definitions of the faces of each die
def die_1_faces : List ℕ := [1, 2, 2, 5, 5, 6]
def die_2_faces : List ℕ := [1, 2, 4, 4, 5, 6]

-- Probability of a sum being 3 (valid_pairs: (1, 2))
def probability_sum_3 : ℚ :=
  (1 / 6) * (1 / 6)

-- Probability of a sum being 7 (valid pairs: (1, 6), (2, 5))
def probability_sum_7 : ℚ :=
  ((1 / 6) * (1 / 6)) + ((1 / 3) * (1 / 6))

-- Probability of a sum being 10 (valid pairs: (5, 5))
def probability_sum_10 : ℚ :=
  (1 / 3) * (1 / 6)

-- Total probability for sums being 3, 7, or 10
def total_probability : ℚ :=
  probability_sum_3 + probability_sum_7 + probability_sum_10

-- The proof statement
theorem probability_sum_3_or_7_or_10 : total_probability = 1 / 6 :=
  sorry

end probability_sum_3_or_7_or_10_l189_18961


namespace find_value_of_x_l189_18947

theorem find_value_of_x (w : ℕ) (x y z : ℕ) (h₁ : x = y / 3) (h₂ : y = z / 6) (h₃ : z = 2 * w) (hw : w = 45) : x = 5 :=
by
  sorry

end find_value_of_x_l189_18947


namespace vector_addition_l189_18924

def a : ℝ × ℝ := (5, -3)
def b : ℝ × ℝ := (-6, 4)

theorem vector_addition : a + b = (-1, 1) := by
  rw [a, b]
  sorry

end vector_addition_l189_18924


namespace sum_last_two_digits_7_13_23_l189_18972

theorem sum_last_two_digits_7_13_23 :
  (7 ^ 23 + 13 ^ 23) % 100 = 40 :=
by 
-- Proof goes here
sorry

end sum_last_two_digits_7_13_23_l189_18972


namespace jason_seashells_initial_count_l189_18999

variable (initialSeashells : ℕ) (seashellsGivenAway : ℕ)
variable (seashellsNow : ℕ) (initialSeashells := 49)
variable (seashellsGivenAway := 13) (seashellsNow := 36)

theorem jason_seashells_initial_count :
  initialSeashells - seashellsGivenAway = seashellsNow → initialSeashells = 49 := by
  sorry

end jason_seashells_initial_count_l189_18999


namespace b_divisible_by_a_l189_18950

theorem b_divisible_by_a (a b c : ℕ) (ha : a > 1) (hbc : b > c ∧ c > 1) (hdiv : (abc + 1) % (ab - b + 1) = 0) : a ∣ b :=
  sorry

end b_divisible_by_a_l189_18950


namespace time_spent_in_park_is_76_19_percent_l189_18957

noncomputable def total_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (t, _, _) => acc + t) 0

noncomputable def total_walking_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (_, w1, w2) => acc + (w1 + w2)) 0

noncomputable def total_trip_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  total_time_in_park trip_times + total_walking_time trip_times

noncomputable def percentage_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℚ :=
  (total_time_in_park trip_times : ℚ) / (total_trip_time trip_times : ℚ) * 100

theorem time_spent_in_park_is_76_19_percent (trip_times : List (ℕ × ℕ × ℕ)) :
  trip_times = [(120, 20, 25), (90, 15, 15), (150, 10, 20), (180, 30, 20), (120, 20, 10), (60, 15, 25)] →
  percentage_time_in_park trip_times = 76.19 :=
by
  intro h
  rw [h]  
  simp
  sorry

end time_spent_in_park_is_76_19_percent_l189_18957


namespace maximum_value_sum_l189_18910

theorem maximum_value_sum (a b c d : ℕ) (h1 : a + c = 1000) (h2 : b + d = 500) :
  ∃ a b c d, a + c = 1000 ∧ b + d = 500 ∧ (a = 1 ∧ c = 999 ∧ b = 499 ∧ d = 1) ∧ 
  ((a : ℝ) / b + (c : ℝ) / d = (1 / 499) + 999) := 
  sorry

end maximum_value_sum_l189_18910


namespace equivalent_annual_rate_approx_l189_18911

noncomputable def annual_rate : ℝ := 0.045
noncomputable def days_in_year : ℝ := 365
noncomputable def daily_rate : ℝ := annual_rate / days_in_year
noncomputable def equivalent_annual_rate : ℝ := (1 + daily_rate) ^ days_in_year - 1

theorem equivalent_annual_rate_approx :
  abs (equivalent_annual_rate - 0.0459) < 0.0001 :=
by sorry

end equivalent_annual_rate_approx_l189_18911


namespace necessary_but_not_sufficient_l189_18939

-- Define the sets A and B
def A (x : ℝ) : Prop := x > 2
def B (x : ℝ) : Prop := x > 1

-- Prove that B (necessary condition x > 1) does not suffice for A (x > 2)
theorem necessary_but_not_sufficient (x : ℝ) (h : B x) : A x ∨ ¬A x :=
by
  -- B x is a necessary condition for A x
  have h1 : x > 1 := h
  -- A x is not necessarily implied by B x
  sorry

end necessary_but_not_sufficient_l189_18939


namespace apples_mass_left_l189_18997

theorem apples_mass_left (initial_kidney golden canada fuji granny : ℕ)
                         (sold_kidney golden canada fuji granny : ℕ)
                         (left_kidney golden canada fuji granny : ℕ) :
  initial_kidney = 26 → sold_kidney = 15 → left_kidney = 11 →
  initial_golden = 42 → sold_golden = 28 → left_golden = 14 →
  initial_canada = 19 → sold_canada = 12 → left_canada = 7 →
  initial_fuji = 35 → sold_fuji = 20 → left_fuji = 15 →
  initial_granny = 22 → sold_granny = 18 → left_granny = 4 →
  left_kidney = initial_kidney - sold_kidney ∧
  left_golden = initial_golden - sold_golden ∧
  left_canada = initial_canada - sold_canada ∧
  left_fuji = initial_fuji - sold_fuji ∧
  left_granny = initial_granny - sold_granny := by sorry

end apples_mass_left_l189_18997


namespace one_in_B_neg_one_not_in_B_B_roster_l189_18991

open Set Int

def B : Set ℤ := {x | ∃ n : ℕ, 6 = n * (3 - x)}

theorem one_in_B : 1 ∈ B :=
by sorry

theorem neg_one_not_in_B : (-1 ∉ B) :=
by sorry

theorem B_roster : B = {2, 1, 0, -3} :=
by sorry

end one_in_B_neg_one_not_in_B_B_roster_l189_18991


namespace second_player_wins_l189_18905

noncomputable def is_winning_position (n : ℕ) : Prop :=
  n % 4 = 0

theorem second_player_wins (n : ℕ) (h : n = 100) :
  ∃ f : ℕ → ℕ, (∀ k, 0 < k → k ≤ n → (k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 5) → is_winning_position (n - k)) ∧ is_winning_position n := 
sorry

end second_player_wins_l189_18905


namespace find_x_l189_18951

theorem find_x (x : ℕ) (h1 : 8^x = 2^9) (h2 : 8 = 2^3) : x = 3 := by
  sorry

end find_x_l189_18951


namespace proof_problem_l189_18994

theorem proof_problem (x : ℝ) (h1 : x = 3) (h2 : 2 * x ≠ 5) (h3 : x + 5 ≠ 3) 
                      (h4 : 7 - x ≠ 2) (h5 : 6 + 2 * x ≠ 14) :
    3 * x - 1 = 8 :=
by 
  sorry

end proof_problem_l189_18994


namespace part_one_part_two_l189_18937

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem part_one :
  ∀ x m : ℕ, f x ≤ -m^2 + 6 * m → 1 ≤ m ∧ m ≤ 5 := 
by
  sorry

theorem part_two (a b c : ℝ) (h : 3 * a + 4 * b + 5 * c = 1) :
  (a^2 + b^2 + c^2) ≥ (1 / 50) :=
by
  sorry

end part_one_part_two_l189_18937


namespace quadratic_polynomial_AT_BT_l189_18902

theorem quadratic_polynomial_AT_BT (p s : ℝ) :
  ∃ (AT BT : ℝ), (AT + BT = p + 3) ∧ (AT * BT = s^2) ∧ (∀ (x : ℝ), (x^2 - (p+3) * x + s^2) = (x - AT) * (x - BT)) := 
sorry

end quadratic_polynomial_AT_BT_l189_18902


namespace selection_ways_l189_18952

-- Step a): Define the conditions
def number_of_boys := 26
def number_of_girls := 24

-- Step c): State the problem
theorem selection_ways :
  number_of_boys + number_of_girls = 50 := by
  sorry

end selection_ways_l189_18952


namespace geometric_concepts_cases_l189_18907

theorem geometric_concepts_cases :
  (∃ x y, x = "rectangle" ∧ y = "rhombus") ∧ 
  (∃ x y z, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "acute_triangle") ∧ 
  (∃ x y z u, x = "parallelogram" ∧ y = "rectangle" ∧ z = "square" ∧ u = "acute_angled_rhombus") ∧ 
  (∃ x y z u t, x = "polygon" ∧ y = "triangle" ∧ z = "isosceles_triangle" ∧ u = "equilateral_triangle" ∧ t = "right_triangle") ∧ 
  (∃ x y z u, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "obtuse_triangle" ∧ u = "scalene_triangle") :=
by {
  sorry
}

end geometric_concepts_cases_l189_18907


namespace find_f_at_3_l189_18977

variable (f : ℝ → ℝ)

-- Conditions
-- 1. f is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
-- 2. f(-1) = 1/2
axiom f_neg_one : f (-1) = 1 / 2
-- 3. f(x+2) = f(x) + 2 for all x
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + 2

-- The target value to prove
theorem find_f_at_3 : f 3 = 3 / 2 := by
  sorry

end find_f_at_3_l189_18977


namespace opposite_of_one_l189_18909

theorem opposite_of_one (a : ℤ) (h : a = -1) : a = -1 := 
by 
  exact h

end opposite_of_one_l189_18909


namespace robotics_club_problem_l189_18998

theorem robotics_club_problem 
    (total_students cs_students eng_students both_students : ℕ)
    (h1 : total_students = 120)
    (h2 : cs_students = 75)
    (h3 : eng_students = 50)
    (h4 : both_students = 10) :
    total_students - (cs_students - both_students + eng_students - both_students + both_students) = 5 := by
  sorry

end robotics_club_problem_l189_18998


namespace rise_in_water_level_l189_18967

theorem rise_in_water_level : 
  let edge_length : ℝ := 15
  let volume_cube : ℝ := edge_length ^ 3
  let length : ℝ := 20
  let width : ℝ := 15
  let base_area : ℝ := length * width
  let rise_in_level : ℝ := volume_cube / base_area
  rise_in_level = 11.25 :=
by
  sorry

end rise_in_water_level_l189_18967


namespace find_point_B_coordinates_l189_18926

theorem find_point_B_coordinates (a : ℝ) : 
  (∀ (x y : ℝ), x^2 - 4*x + y^2 = 0 → (x - a)^2 + y^2 = 4 * ((x - 1)^2 + y^2)) →
  a = -2 :=
by
  sorry

end find_point_B_coordinates_l189_18926


namespace simplify_expr_l189_18934

theorem simplify_expr : 
  (576:ℝ)^(1/4) * (216:ℝ)^(1/2) = 72 := 
by 
  have h1 : 576 = (2^4 * 36 : ℝ) := by norm_num
  have h2 : 36 = (6^2 : ℝ) := by norm_num
  have h3 : 216 = (6^3 : ℝ) := by norm_num
  sorry

end simplify_expr_l189_18934


namespace range_of_a_l189_18970

def p (a : ℝ) : Prop :=
(∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0)

def q (a : ℝ) : Prop :=
0 < a ∧ a < 1

theorem range_of_a (a : ℝ) : ((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
  sorry

end range_of_a_l189_18970


namespace chosen_number_is_129_l189_18955

theorem chosen_number_is_129 (x : ℕ) (h : 2 * x - 148 = 110) : x = 129 :=
by
  sorry

end chosen_number_is_129_l189_18955


namespace cone_surface_area_is_correct_l189_18964

noncomputable def cone_surface_area (central_angle_degrees : ℝ) (sector_area : ℝ) : ℝ :=
  if central_angle_degrees = 120 ∧ sector_area = 3 * Real.pi then 4 * Real.pi else 0

theorem cone_surface_area_is_correct :
  cone_surface_area 120 (3 * Real.pi) = 4 * Real.pi :=
by
  -- proof would go here
  sorry

end cone_surface_area_is_correct_l189_18964


namespace lion_to_leopard_ratio_l189_18953

variable (L P E : ℕ)

axiom lion_count : L = 200
axiom total_population : L + P + E = 450
axiom elephants_relation : E = (1 / 2 : ℚ) * (L + P)

theorem lion_to_leopard_ratio : L / P = 2 :=
by
  sorry

end lion_to_leopard_ratio_l189_18953


namespace unicorn_witch_ratio_l189_18965

theorem unicorn_witch_ratio (W D U : ℕ) (h1 : W = 7) (h2 : D = W + 25) (h3 : U + W + D = 60) :
  U / W = 3 := by
  sorry

end unicorn_witch_ratio_l189_18965


namespace number_of_distinct_configurations_l189_18945

-- Define the conditions
def numConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 else n + 1

-- Theorem statement
theorem number_of_distinct_configurations (n : ℕ) : 
  numConfigurations n = if n % 2 = 1 then 2 else n + 1 :=
by
  sorry -- Proof intentionally left out

end number_of_distinct_configurations_l189_18945


namespace route_B_no_quicker_l189_18982

noncomputable def time_route_A (distance_A : ℕ) (speed_A : ℕ) : ℕ :=
(distance_A * 60) / speed_A

noncomputable def time_route_B (distance_B : ℕ) (speed_B1 : ℕ) (speed_B2 : ℕ) : ℕ :=
  let distance_B1 := distance_B - 1
  let distance_B2 := 1
  (distance_B1 * 60) / speed_B1 + (distance_B2 * 60) / speed_B2

theorem route_B_no_quicker : time_route_A 8 40 = time_route_B 6 50 10 :=
by
  sorry

end route_B_no_quicker_l189_18982


namespace desiree_age_l189_18983

theorem desiree_age (D C G Gr : ℕ) 
  (h1 : D = 2 * C)
  (h2 : D + 30 = (2 * (C + 30)) / 3 + 14)
  (h3 : G = D + C)
  (h4 : G + 20 = 3 * (D - C))
  (h5 : Gr = (D + 10) * (C + 10) / 2) : 
  D = 6 := 
sorry

end desiree_age_l189_18983


namespace fraction_product_eq_six_l189_18986

theorem fraction_product_eq_six : (2/5) * (3/4) * (1/6) * (120 : ℚ) = 6 := by
  sorry

end fraction_product_eq_six_l189_18986


namespace prob_bashers_win_at_least_4_out_of_5_l189_18956

-- Define the probability p that the Bashers win a single game.
def p := 4 / 5

-- Define the number of games n.
def n := 5

-- Define the random trial outcome space.
def trials : Type := Fin n → Bool

-- Define the number of wins (true means a win, false means a loss).
def wins (t : trials) : ℕ := (Finset.univ.filter (λ i => t i = true)).card

-- Define winning exactly k games.
def win_exactly (t : trials) (k : ℕ) : Prop := wins t = k

-- Define the probability of winning exactly k games.
noncomputable def prob_win_exactly (k : ℕ) : ℚ :=
  (Nat.descFactorial n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the event of winning at least 4 out of 5 games.
def event_win_at_least (t : trials) := (wins t ≥ 4)

-- Define the probability of winning at least k out of n games.
noncomputable def prob_win_at_least (k : ℕ) : ℚ :=
  prob_win_exactly k + prob_win_exactly (k + 1)

-- Theorem to prove: Probability of winning at least 4 out of 5 games is 3072/3125.
theorem prob_bashers_win_at_least_4_out_of_5 :
  prob_win_at_least 4 = 3072 / 3125 :=
by
  sorry

end prob_bashers_win_at_least_4_out_of_5_l189_18956


namespace max_3x_4y_eq_73_l189_18988

theorem max_3x_4y_eq_73 :
  (∀ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73) ∧
  (∃ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 ∧ 3 * x + 4 * y = 73) :=
by sorry

end max_3x_4y_eq_73_l189_18988


namespace parabola_focus_line_slope_intersect_l189_18978

theorem parabola_focus (p : ℝ) (hp : 0 < p) 
  (focus : (1/2 : ℝ) = p/2) : p = 1 :=
by sorry

theorem line_slope_intersect (t : ℝ)
  (intersects_parabola : ∃ A B : ℝ × ℝ, A ≠ (0, 0) ∧ B ≠ (0, 0) ∧
    A ≠ B ∧ A.2 = 2 * A.1 + t ∧ B.2 = 2 * B.1 + t ∧ 
    A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
    A.1 * B.1 + A.2 * B.2 = 0) : 
  t = -4 :=
by sorry

end parabola_focus_line_slope_intersect_l189_18978


namespace exists_unique_circle_l189_18940

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

def diametrically_opposite_points (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (cx, cy) := C.center
  let (px, py) := P
  (px - cx) ^ 2 + (py - cy) ^ 2 = (C.radius ^ 2)

def intersects_at_diametrically_opposite_points (K A : Circle) : Prop :=
  ∃ P₁ P₂ : ℝ × ℝ, diametrically_opposite_points A P₁ ∧ diametrically_opposite_points A P₂ ∧
  P₁ ≠ P₂ ∧ diametrically_opposite_points K P₁ ∧ diametrically_opposite_points K P₂

theorem exists_unique_circle (A B C : Circle) :
  ∃! K : Circle, intersects_at_diametrically_opposite_points K A ∧
  intersects_at_diametrically_opposite_points K B ∧
  intersects_at_diametrically_opposite_points K C := sorry

end exists_unique_circle_l189_18940


namespace cos_2beta_value_l189_18969

theorem cos_2beta_value (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5) 
  (h2 : Real.cos (α + β) = -3/5) 
  (h3 : α - β ∈ Set.Ioo (π/2) π) 
  (h4 : α + β ∈ Set.Ioo (π/2) π) : 
  Real.cos (2 * β) = 24/25 := 
sorry

end cos_2beta_value_l189_18969


namespace complementary_three_card_sets_l189_18923

-- Definitions for the problem conditions
inductive Shape | circle | square | triangle | star
inductive Color | red | blue | green | yellow
inductive Shade | light | medium | dark | very_dark

-- Definition of a Card as a combination of shape, color, shade
structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

-- Definition of a set being complementary
def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape = c2.shape ∧ c2.shape = c3.shape) ∨ (c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape)) ∧
  ((c1.color = c2.color ∧ c2.color = c3.color) ∨ (c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color)) ∧
  ((c1.shade = c2.shade ∧ c2.shade = c3.shade) ∨ (c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade))

-- Definition of the problem statement
def complementary_three_card_sets_count : Nat :=
  360

-- The theorem to be proved
theorem complementary_three_card_sets : ∃ (n : Nat), n = complementary_three_card_sets_count :=
  by
    use 360
    sorry

end complementary_three_card_sets_l189_18923


namespace train_speed_is_correct_l189_18912

-- Definitions for conditions
def train_length : ℝ := 150  -- length of the train in meters
def time_to_cross_pole : ℝ := 3  -- time to cross the pole in seconds

-- Proof statement
theorem train_speed_is_correct : (train_length / time_to_cross_pole) = 50 := by
  sorry

end train_speed_is_correct_l189_18912


namespace joel_laps_count_l189_18993

def yvonne_laps : ℕ := 10

def younger_sister_laps : ℕ := yvonne_laps / 2

def joel_laps : ℕ := younger_sister_laps * 3

theorem joel_laps_count : joel_laps = 15 := by
  -- The proof is not required as per instructions
  sorry

end joel_laps_count_l189_18993


namespace silver_cube_price_l189_18949

theorem silver_cube_price
  (price_2inch_cube : ℝ := 300) (side_length_2inch : ℝ := 2) (side_length_4inch : ℝ := 4) : 
  price_4inch_cube = 2400 := 
by 
  sorry

end silver_cube_price_l189_18949


namespace range_of_a_l189_18995

theorem range_of_a 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x, f x = -x^2 + 2*(a - 1)*x + 2)
  (increasing_on : ∀ x < 4, deriv f x > 0) : a ≥ 5 :=
sorry

end range_of_a_l189_18995


namespace sum_integers_neg40_to_60_l189_18980

theorem sum_integers_neg40_to_60 : 
  (Finset.sum (Finset.range (60 + 40 + 1)) (λ x => x - 40)) = 1010 := sorry

end sum_integers_neg40_to_60_l189_18980


namespace nathan_subtracts_79_l189_18962

theorem nathan_subtracts_79 (a b : ℤ) (h₁ : a = 40) (h₂ : b = 1) :
  (a - b) ^ 2 = a ^ 2 - 79 := 
by
  sorry

end nathan_subtracts_79_l189_18962


namespace probability_relationship_l189_18922

def total_outcomes : ℕ := 36

def P1 : ℚ := 1 / total_outcomes
def P2 : ℚ := 2 / total_outcomes
def P3 : ℚ := 3 / total_outcomes

theorem probability_relationship :
  P1 < P2 ∧ P2 < P3 :=
by
  sorry

end probability_relationship_l189_18922


namespace solve_problem_l189_18901

-- Declare the variables n and m
variables (n m : ℤ)

-- State the theorem with given conditions and prove that 2n + m = 36
theorem solve_problem
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 :=
sorry

end solve_problem_l189_18901


namespace nods_per_kilometer_l189_18930

theorem nods_per_kilometer
  (p q r s t u : ℕ)
  (h1 : p * q = q * p)
  (h2 : r * s = s * r)
  (h3 : t * u = u * t) : 
  (1 : ℕ) = qts/pru :=
by
  sorry

end nods_per_kilometer_l189_18930


namespace ethanol_total_amount_l189_18938

-- Definitions based on Conditions
def total_tank_capacity : ℕ := 214
def fuel_A_volume : ℕ := 106
def fuel_B_volume : ℕ := total_tank_capacity - fuel_A_volume
def ethanol_in_fuel_A : ℚ := 0.12
def ethanol_in_fuel_B : ℚ := 0.16

-- Theorem Statement
theorem ethanol_total_amount :
  (fuel_A_volume * ethanol_in_fuel_A + fuel_B_volume * ethanol_in_fuel_B) = 30 := 
sorry

end ethanol_total_amount_l189_18938


namespace seongjun_ttakji_count_l189_18963

variable (S A : ℕ)

theorem seongjun_ttakji_count (h1 : (3/4 : ℚ) * S - 25 = 7 * (A - 50)) (h2 : A = 100) : S = 500 :=
sorry

end seongjun_ttakji_count_l189_18963


namespace largest_possible_n_base10_l189_18958

theorem largest_possible_n_base10 :
  ∃ (n A B C : ℕ),
    n = 25 * A + 5 * B + C ∧ 
    n = 81 * C + 9 * B + A ∧ 
    A < 5 ∧ B < 5 ∧ C < 5 ∧ 
    n = 69 :=
by {
  sorry
}

end largest_possible_n_base10_l189_18958


namespace percent_of_a_is_4b_l189_18927

variables (a b : ℝ)
theorem percent_of_a_is_4b (h : a = 2 * b) : 4 * b / a = 2 :=
by 
  sorry

end percent_of_a_is_4b_l189_18927


namespace range_of_x_l189_18904

theorem range_of_x (θ : ℝ) (h0 : 0 < θ) (h1 : θ < Real.pi / 2) (h2 : ∀ θ, (0 < θ) → (θ < Real.pi / 2) → (1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2 ≥ abs (2 * x - 1))) :
  -4 ≤ x ∧ x ≤ 5 := sorry

end range_of_x_l189_18904


namespace cylinder_volume_l189_18966

theorem cylinder_volume (length width : ℝ) (h₁ h₂ : ℝ) (radius1 radius2 : ℝ) (V1 V2 : ℝ) (π : ℝ)
  (h_length : length = 12) (h_width : width = 8) 
  (circumference1 : circumference1 = length)
  (circumference2 : circumference2 = width)
  (h_radius1 : radius1 = 6 / π) (h_radius2 : radius2 = 4 / π)
  (h_height1 : h₁ = width) (h_height2 : h₂ = length)
  (h_V1 : V1 = π * radius1^2 * h₁) (h_V2 : V2 = π * radius2^2 * h₂) :
  V1 = 288 / π ∨ V2 = 192 / π :=
sorry


end cylinder_volume_l189_18966


namespace total_eyes_l189_18908

def boys := 23
def girls := 18
def cats := 10
def spiders := 5

def boy_eyes := 2
def girl_eyes := 2
def cat_eyes := 2
def spider_eyes := 8

theorem total_eyes : (boys * boy_eyes) + (girls * girl_eyes) + (cats * cat_eyes) + (spiders * spider_eyes) = 142 := by
  sorry

end total_eyes_l189_18908


namespace evaluate_expression_l189_18917

variable (x y z : ℤ)

theorem evaluate_expression :
  x = 3 → y = 2 → z = 4 → 3 * x - 4 * y + 5 * z = 21 :=
by
  intros hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l189_18917


namespace evaluate_f_at_1_l189_18992

noncomputable def f (x : ℝ) : ℝ := 2^x + 2

theorem evaluate_f_at_1 : f 1 = 4 :=
by {
  -- proof goes here
  sorry
}

end evaluate_f_at_1_l189_18992


namespace verify_b_c_sum_ten_l189_18919

theorem verify_b_c_sum_ten (a b c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hb : 1 ≤ b ∧ b < 10) (hc : 1 ≤ c ∧ c < 10) 
    (h_eq : (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a ^ 2) : b + c = 10 :=
by
  sorry

end verify_b_c_sum_ten_l189_18919


namespace consecutive_integers_product_divisible_l189_18990

theorem consecutive_integers_product_divisible (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∀ n : ℕ, ∃ (x y : ℕ), (n ≤ x) ∧ (x < n + b) ∧ (n ≤ y) ∧ (y < n + b) ∧ (x ≠ y) ∧ (a * b ∣ x * y) :=
by
  sorry

end consecutive_integers_product_divisible_l189_18990


namespace problem_1_l189_18984

theorem problem_1 (a b : ℝ) (h : b < a ∧ a < 0) : 
  (a + b < a * b) ∧ (¬ (abs a > abs b)) ∧ (¬ (1 / b > 1 / a ∧ 1 / a > 0)) ∧ (¬ (b / a + a / b > 2)) := sorry

end problem_1_l189_18984


namespace solve_rational_eq_l189_18985

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 3 * x - 18) + 1 / (x^2 - 15 * x - 12) = 0) →
  (x = 1 ∨ x = -1 ∨ x = 12 ∨ x = -12) :=
by
  intro h
  sorry

end solve_rational_eq_l189_18985


namespace sequence_value_2023_l189_18914

theorem sequence_value_2023 (a : ℕ → ℕ) (h₁ : a 1 = 3)
  (h₂ : ∀ m n : ℕ, a (m + n) = a m + a n) : a 2023 = 6069 := by
  sorry

end sequence_value_2023_l189_18914


namespace fraction_addition_l189_18918

theorem fraction_addition : (3 / 4 : ℚ) + (5 / 6) = 19 / 12 :=
by
  sorry

end fraction_addition_l189_18918


namespace taxi_fare_l189_18936

theorem taxi_fare (x : ℝ) (h : x > 3) : 
  let starting_price := 6
  let additional_fare_per_km := 1.4
  let fare := starting_price + additional_fare_per_km * (x - 3)
  fare = 1.4 * x + 1.8 :=
by
  sorry

end taxi_fare_l189_18936


namespace gum_ratio_correct_l189_18916

variable (y : ℝ)
variable (cherry_pieces : ℝ := 30)
variable (grape_pieces : ℝ := 40)
variable (pieces_per_pack : ℝ := y)

theorem gum_ratio_correct:
  ((cherry_pieces - 2 * pieces_per_pack) / grape_pieces = cherry_pieces / (grape_pieces + 4 * pieces_per_pack)) ↔ y = 5 :=
by
  sorry

end gum_ratio_correct_l189_18916


namespace problem_solution_l189_18943

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem problem_solution (x1 x2 : ℝ) 
  (hx1 : x1 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (hx2 : x2 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (h : f x1 < f x2) : x1^2 > x2^2 := 
sorry

end problem_solution_l189_18943


namespace solution_set_of_inequality_l189_18944

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x - 2 else if x < 0 then -(x - 2) else 0

theorem solution_set_of_inequality :
  {x : ℝ | f x < 1 / 2} =
  {x : ℝ | (0 ≤ x ∧ x < 5 / 2) ∨ x < -3 / 2} :=
by
  sorry

end solution_set_of_inequality_l189_18944
