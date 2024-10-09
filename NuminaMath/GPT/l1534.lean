import Mathlib

namespace stickers_per_student_l1534_153413

theorem stickers_per_student : 
  ∀ (gold silver bronze total : ℕ), 
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total = gold + silver + bronze →
    total / 5 = 46 :=
by
  intros
  sorry

end stickers_per_student_l1534_153413


namespace painted_cube_problem_l1534_153456

theorem painted_cube_problem (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2)^2 = (n - 2)^3) : n = 8 :=
by {
  sorry
}

end painted_cube_problem_l1534_153456


namespace num_students_second_grade_l1534_153447

structure School :=
(total_students : ℕ)
(prob_male_first_grade : ℝ)

def stratified_sampling (school : School) : ℕ := sorry

theorem num_students_second_grade (school : School) (total_selected : ℕ) : 
    school.total_students = 4000 →
    school.prob_male_first_grade = 0.2 →
    total_selected = 100 →
    stratified_sampling school = 30 :=
by
  intros
  sorry

end num_students_second_grade_l1534_153447


namespace total_days_spent_l1534_153466

theorem total_days_spent {weeks_to_days : ℕ → ℕ} : 
  (weeks_to_days 3 + weeks_to_days 1) + 
  (weeks_to_days (weeks_to_days 3 + weeks_to_days 2) + 3) + 
  (2 * (weeks_to_days (weeks_to_days 3 + weeks_to_days 2))) + 
  (weeks_to_days 5 - weeks_to_days 1) + 
  (weeks_to_days ((weeks_to_days 5 - weeks_to_days 1) - weeks_to_days 3) + 6) + 
  (weeks_to_days (weeks_to_days 5 - weeks_to_days 1) + 4) = 230 :=
by
  sorry

end total_days_spent_l1534_153466


namespace temperature_range_l1534_153448

theorem temperature_range (t : ℕ) : (21 ≤ t ∧ t ≤ 29) :=
by
  sorry

end temperature_range_l1534_153448


namespace largest_garden_is_candace_and_difference_is_100_l1534_153485

-- Define the dimensions of the gardens
def area_alice : Nat := 30 * 50
def area_bob : Nat := 35 * 45
def area_candace : Nat := 40 * 40

-- The proof goal
theorem largest_garden_is_candace_and_difference_is_100 :
  area_candace > area_alice ∧ area_candace > area_bob ∧ area_candace - area_alice = 100 := by
    sorry

end largest_garden_is_candace_and_difference_is_100_l1534_153485


namespace hourly_wage_difference_l1534_153479

theorem hourly_wage_difference (P Q: ℝ) (H_p: ℝ) (H_q: ℝ) (h1: P = 1.5 * Q) (h2: H_q = H_p + 10) (h3: P * H_p = 420) (h4: Q * H_q = 420) : P - Q = 7 := by
  sorry

end hourly_wage_difference_l1534_153479


namespace train_length_l1534_153473

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def distance_ahead : ℝ := 270
noncomputable def time_to_pass : ℝ := 39

noncomputable def jogger_speed_mps := jogger_speed_kmph * (1000 / 1) * (1 / 3600)
noncomputable def train_speed_mps := train_speed_kmph * (1000 / 1) * (1 / 3600)

noncomputable def relative_speed_mps := train_speed_mps - jogger_speed_mps

theorem train_length :
  let jogger_speed := 9 * (1000 / 3600)
  let train_speed := 45 * (1000 / 3600)
  let relative_speed := train_speed - jogger_speed
  let distance := 270
  let time := 39
  distance + relative_speed * time = 390 → relative_speed * time = 120 := by
  sorry

end train_length_l1534_153473


namespace find_positive_number_l1534_153419

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l1534_153419


namespace range_of_a_l1534_153499

variable {x a : ℝ}

theorem range_of_a (hx : 1 ≤ x ∧ x ≤ 2) (h : 2 * x > a - x^2) : a < 8 :=
by sorry

end range_of_a_l1534_153499


namespace total_matches_l1534_153426

noncomputable def matches_in_tournament (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_matches :
  matches_in_tournament 5 + matches_in_tournament 7 + matches_in_tournament 4 = 37 := 
by 
  sorry

end total_matches_l1534_153426


namespace mathland_transport_l1534_153423

theorem mathland_transport (n : ℕ) (h : n ≥ 2) (transport : Fin n -> Fin n -> Prop) :
(∀ i j, transport i j ∨ transport j i) →
(∃ tr : Fin n -> Fin n -> Prop, 
  (∀ i j, transport i j → tr i j) ∨
  (∀ i j, transport j i → tr i j)) :=
by
  sorry

end mathland_transport_l1534_153423


namespace xyz_identity_l1534_153438

theorem xyz_identity (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : xy + xz + yz = 32) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 1400 := 
by 
  -- Proof steps will be placed here, use sorry for now
  sorry

end xyz_identity_l1534_153438


namespace regular_decagon_interior_angle_l1534_153421

theorem regular_decagon_interior_angle {n : ℕ} (h1 : n = 10) (h2 : ∀ (k : ℕ), k = 10 → (180 * (k - 2)) / 10 = 144) : 
  (∃ θ : ℕ, θ = 180 * (n - 2) / n ∧ n = 10 ∧ θ = 144) :=
by
  sorry

end regular_decagon_interior_angle_l1534_153421


namespace altitude_on_hypotenuse_l1534_153425

theorem altitude_on_hypotenuse (a b : ℝ) (h₁ : a = 5) (h₂ : b = 12) (c : ℝ) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  ∃ h : ℝ, h = (a * b) / c ∧ h = 60 / 13 :=
by
  use (5 * 12) / 13
  -- proof that (60 / 13) is indeed the altitude will be done by verifying calculations
  sorry

end altitude_on_hypotenuse_l1534_153425


namespace probability_multiple_choice_and_essay_correct_l1534_153400

noncomputable def probability_multiple_choice_and_essay (C : ℕ → ℕ → ℕ) : ℚ :=
    (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3)

theorem probability_multiple_choice_and_essay_correct (C : ℕ → ℕ → ℕ) :
    probability_multiple_choice_and_essay C = (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3) :=
by
  sorry

end probability_multiple_choice_and_essay_correct_l1534_153400


namespace solution_set_part1_solution_set_part2_l1534_153414

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

theorem solution_set_part1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem solution_set_part2 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end solution_set_part1_solution_set_part2_l1534_153414


namespace participants_initial_count_l1534_153442

theorem participants_initial_count (initial_participants remaining_after_first_round remaining_after_second_round : ℝ) 
  (h1 : remaining_after_first_round = 0.4 * initial_participants)
  (h2 : remaining_after_second_round = (1/4) * remaining_after_first_round)
  (h3 : remaining_after_second_round = 15) : 
  initial_participants = 150 :=
sorry

end participants_initial_count_l1534_153442


namespace sum_roots_quadratic_l1534_153472

theorem sum_roots_quadratic (a b c : ℝ) (P : ℝ → ℝ) 
  (hP : ∀ x : ℝ, P x = a * x^2 + b * x + c)
  (h : ∀ x : ℝ, P (2 * x^5 + 3 * x) ≥ P (3 * x^4 + 2 * x^2 + 1)) : 
  -b / a = 6 / 5 :=
sorry

end sum_roots_quadratic_l1534_153472


namespace total_water_output_l1534_153486

theorem total_water_output (flow_rate: ℚ) (time_duration: ℕ) (total_water: ℚ) :
  flow_rate = 2 + 2 / 3 → time_duration = 9 → total_water = 24 →
  flow_rate * time_duration = total_water :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_water_output_l1534_153486


namespace initial_avg_height_l1534_153471

-- Lean 4 statement for the given problem
theorem initial_avg_height (A : ℝ) (n : ℕ) (wrong_height correct_height actual_avg init_diff : ℝ)
  (h_class_size : n = 35)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106)
  (h_actual_avg : actual_avg = 183)
  (h_init_diff : init_diff = wrong_height - correct_height)
  (h_total_height_actual : n * actual_avg = 35 * 183)
  (h_total_height_wrong : n * A = 35 * actual_avg - init_diff) :
  A = 181 :=
by {
  -- The problem and conditions are correctly stated. The proof is skipped with sorry.
  sorry
}

end initial_avg_height_l1534_153471


namespace trig_expression_value_l1534_153449

theorem trig_expression_value (θ : ℝ)
  (h1 : Real.sin (Real.pi + θ) = 1/4) :
  (Real.cos (Real.pi + θ) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) + 
  Real.sin (Real.pi / 2 - θ) / (Real.cos (θ + 2 * Real.pi) * Real.cos (Real.pi + θ) + Real.cos (-θ))) = 32 :=
by
  sorry

end trig_expression_value_l1534_153449


namespace map_distance_scaled_l1534_153437

theorem map_distance_scaled (d_map : ℝ) (scale : ℝ) (d_actual : ℝ) :
  d_map = 8 ∧ scale = 1000000 → d_actual = 80 :=
by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end map_distance_scaled_l1534_153437


namespace angles_with_same_terminal_side_pi_div_3_l1534_153412

noncomputable def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 2 * k * Real.pi

theorem angles_with_same_terminal_side_pi_div_3 :
  { α : ℝ | same_terminal_side α (Real.pi / 3) } =
  { α : ℝ | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3 } :=
by
  sorry

end angles_with_same_terminal_side_pi_div_3_l1534_153412


namespace Rachel_and_Mike_l1534_153476

theorem Rachel_and_Mike :
  ∃ b c : ℤ,
    (∀ x : ℝ, |x - 3| = 4 ↔ (x = 7 ∨ x = -1)) ∧
    (∀ x : ℝ, (x - 7) * (x + 1) = 0 ↔ x * x + b * x + c = 0) ∧
    (b, c) = (-6, -7) := by
sorry

end Rachel_and_Mike_l1534_153476


namespace percentage_problem_l1534_153402

theorem percentage_problem (P : ℕ) (n : ℕ) (h_n : n = 16)
  (h_condition : (40: ℚ) = 0.25 * n + 2) : P = 250 :=
by
  sorry

end percentage_problem_l1534_153402


namespace average_after_17th_inning_l1534_153488

-- Define the conditions.
variable (A : ℚ) -- The initial average after 16 innings

-- Define the score in the 17th inning and the increment in the average.
def runs_in_17th_inning : ℚ := 87
def increment_in_average : ℚ := 3

-- Define the equation derived from the conditions.
theorem average_after_17th_inning :
  (16 * A + runs_in_17th_inning) / 17 = A + increment_in_average →
  A + increment_in_average = 39 :=
sorry

end average_after_17th_inning_l1534_153488


namespace product_of_real_solutions_of_t_cubed_eq_216_l1534_153459

theorem product_of_real_solutions_of_t_cubed_eq_216 : 
  (∃ t : ℝ, t^3 = 216) →
  (∀ t₁ t₂, (t₁ = t₂) → (t₁^3 = 216 → t₂^3 = 216) → (t₁ * t₂ = 6)) :=
by
  sorry

end product_of_real_solutions_of_t_cubed_eq_216_l1534_153459


namespace children_left_l1534_153451

-- Define the initial problem constants and conditions
def totalGuests := 50
def halfGuests := totalGuests / 2
def numberOfMen := 15
def numberOfWomen := halfGuests
def numberOfChildren := totalGuests - (numberOfWomen + numberOfMen)
def proportionMenLeft := numberOfMen / 5
def totalPeopleStayed := 43
def totalPeopleLeft := totalGuests - totalPeopleStayed

-- Define the proposition to prove
theorem children_left : 
  totalPeopleLeft - proportionMenLeft = 4 := by 
    sorry

end children_left_l1534_153451


namespace part1_exists_infinite_rationals_part2_rationals_greater_bound_l1534_153436

theorem part1_exists_infinite_rationals 
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2):
  ∀ ε > 0, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ abs (q / p - sqrt5_minus1_div2) < 1 / p ^ 2 :=
by sorry

theorem part2_rationals_greater_bound
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2)
  (sqrt5_plus1_inv := 1 / (Real.sqrt 5 + 1)):
  ∀ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 → abs (q / p - sqrt5_minus1_div2) > sqrt5_plus1_inv / p ^ 2 :=
by sorry

end part1_exists_infinite_rationals_part2_rationals_greater_bound_l1534_153436


namespace acute_angle_10_10_l1534_153416

noncomputable def clock_angle_proof : Prop :=
  let minute_hand_position := 60
  let hour_hand_position := 305
  let angle_diff := hour_hand_position - minute_hand_position
  let acute_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  acute_angle = 115

theorem acute_angle_10_10 : clock_angle_proof := by
  sorry

end acute_angle_10_10_l1534_153416


namespace cos_double_angle_l1534_153409

-- Definition of the terminal condition
def terminal_side_of_angle (α : ℝ) (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = Real.sqrt 3) ∧ (x^2 + y^2 = 1)

-- Prove the required statement
theorem cos_double_angle (α : ℝ) :
  (terminal_side_of_angle α 1 (Real.sqrt 3)) →
  Real.cos (2 * α + Real.pi / 2) = - Real.sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_l1534_153409


namespace linear_independent_vectors_p_value_l1534_153440

theorem linear_independent_vectors_p_value (p : ℝ) :
  (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ a * (2 : ℝ) + b * (5 : ℝ) = 0 ∧ a * (4 : ℝ) + b * p = 0) ↔ p = 10 :=
by
  sorry

end linear_independent_vectors_p_value_l1534_153440


namespace wall_length_to_height_ratio_l1534_153461

theorem wall_length_to_height_ratio (W H L V : ℝ) (h1 : H = 6 * W) (h2 : V = W * H * L) (h3 : W = 4) (h4 : V = 16128) :
  L / H = 7 :=
by
  -- Note: The proof steps are omitted as per the problem's instructions.
  sorry

end wall_length_to_height_ratio_l1534_153461


namespace range_of_f_on_interval_l1534_153407

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem range_of_f_on_interval :
  Set.Icc (-1 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = y} :=
by
  sorry

end range_of_f_on_interval_l1534_153407


namespace find_first_type_cookies_l1534_153405

section CookiesProof

variable (x : ℕ)

-- Conditions
def box_first_type_cookies : ℕ := x
def box_second_type_cookies : ℕ := 20
def box_third_type_cookies : ℕ := 16
def boxes_first_type_sold : ℕ := 50
def boxes_second_type_sold : ℕ := 80
def boxes_third_type_sold : ℕ := 70
def total_cookies_sold : ℕ := 3320

-- Theorem to prove
theorem find_first_type_cookies 
  (h1 : 50 * x + 80 * box_second_type_cookies + 70 * box_third_type_cookies = total_cookies_sold) :
  x = 12 := by
    sorry

end CookiesProof

end find_first_type_cookies_l1534_153405


namespace problem_l1534_153434

noncomputable def f (a b x : ℝ) := a * x^2 - b * x + 1

theorem problem (a b : ℝ) (h1 : 4 * a - b^2 = 3)
                (h2 : ∀ x : ℝ, f a b (x + 1) = f a b (-x))
                (h3 : b = a + 1) 
                (h4 : 0 ≤ a ∧ a ≤ 1) 
                (h5 : ∀ x ∈ Set.Icc 0 2, ∃ m : ℝ, m ≥ abs (f a b x)) :
  (∀ x : ℝ, f a b x = x^2 - x + 1) ∧ (∃ m : ℝ, m = 1 ∧ ∀ x ∈ Set.Icc 0 2, m ≥ abs (f a b x)) :=
  sorry

end problem_l1534_153434


namespace average_percent_score_l1534_153475

theorem average_percent_score (num_students : ℕ)
    (students_95 students_85 students_75 students_65 students_55 students_45 : ℕ)
    (h : students_95 + students_85 + students_75 + students_65 + students_55 + students_45 = 120) :
  ((95 * students_95 + 85 * students_85 + 75 * students_75 + 65 * students_65 + 55 * students_55 + 45 * students_45) / 120 : ℚ) = 72.08 := 
by {
  sorry
}

end average_percent_score_l1534_153475


namespace snowdrift_depth_end_of_third_day_l1534_153477

theorem snowdrift_depth_end_of_third_day :
  let depth_ninth_day := 40
  let d_before_eighth_night_snowfall := depth_ninth_day - 10
  let d_before_eighth_day_melting := d_before_eighth_night_snowfall * 4 / 3
  let depth_seventh_day := d_before_eighth_day_melting
  let d_before_sixth_day_snowfall := depth_seventh_day - 20
  let d_before_fifth_day_snowfall := d_before_sixth_day_snowfall - 15
  let d_before_fourth_day_melting := d_before_fifth_day_snowfall * 3 / 2
  depth_ninth_day = 40 →
  d_before_eighth_night_snowfall = depth_ninth_day - 10 →
  d_before_eighth_day_melting = d_before_eighth_night_snowfall * 4 / 3 →
  depth_seventh_day = d_before_eighth_day_melting →
  d_before_sixth_day_snowfall = depth_seventh_day - 20 →
  d_before_fifth_day_snowfall = d_before_sixth_day_snowfall - 15 →
  d_before_fourth_day_melting = d_before_fifth_day_snowfall * 3 / 2 →
  d_before_fourth_day_melting = 7.5 :=
by
  intros
  sorry

end snowdrift_depth_end_of_third_day_l1534_153477


namespace product_of_sums_of_two_squares_l1534_153452

theorem product_of_sums_of_two_squares
  (a b a1 b1 : ℤ) :
  ((a^2 + b^2) * (a1^2 + b1^2)) = ((a * a1 - b * b1)^2 + (a * b1 + b * a1)^2) := 
sorry

end product_of_sums_of_two_squares_l1534_153452


namespace total_coin_value_l1534_153403

theorem total_coin_value (total_coins : ℕ) (two_dollar_coins : ℕ) (one_dollar_value : ℕ)
  (two_dollar_value : ℕ) (h_total_coins : total_coins = 275)
  (h_two_dollar_coins : two_dollar_coins = 148)
  (h_one_dollar_value : one_dollar_value = 1)
  (h_two_dollar_value : two_dollar_value = 2) :
  total_coins - two_dollar_coins = 275 - 148
  ∧ ((total_coins - two_dollar_coins) * one_dollar_value + two_dollar_coins * two_dollar_value) = 423 :=
by
  sorry

end total_coin_value_l1534_153403


namespace union_P_Q_l1534_153439

open Set

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5, 6}

theorem union_P_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} :=
by 
  -- Proof goes here
  sorry

end union_P_Q_l1534_153439


namespace no_arithmetic_mean_l1534_153467

def eight_thirteen : ℚ := 8 / 13
def eleven_seventeen : ℚ := 11 / 17
def five_eight : ℚ := 5 / 8

-- Define the function to calculate the arithmetic mean of two rational numbers
def arithmetic_mean (a b : ℚ) : ℚ :=
(a + b) / 2

-- The theorem statement
theorem no_arithmetic_mean :
  eight_thirteen ≠ arithmetic_mean eleven_seventeen five_eight ∧
  eleven_seventeen ≠ arithmetic_mean eight_thirteen five_eight ∧
  five_eight ≠ arithmetic_mean eight_thirteen eleven_seventeen :=
sorry

end no_arithmetic_mean_l1534_153467


namespace probability_five_digit_palindrome_divisible_by_11_l1534_153410

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  n % 100 = 100*a + 10*b + c

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem probability_five_digit_palindrome_divisible_by_11 :
  let count_palindromes := 9 * 10 * 10
  let count_divisible_by_11 := 165
  (count_divisible_by_11 : ℚ) / count_palindromes = 11 / 60 :=
by
  sorry

end probability_five_digit_palindrome_divisible_by_11_l1534_153410


namespace calculate_expression_l1534_153404

theorem calculate_expression : 
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 :=
by
  sorry

end calculate_expression_l1534_153404


namespace pipe_B_time_l1534_153483

theorem pipe_B_time (C : ℝ) (T : ℝ) 
    (h1 : 2 / 3 * C + C / 3 = C)
    (h2 : C / 36 + C / (3 * T) = C / 14.4) 
    (h3 : T > 0) : 
    T = 8 := 
sorry

end pipe_B_time_l1534_153483


namespace min_dot_product_l1534_153411

variable {α : Type}
variables {a b : α}

noncomputable def dot (x y : α) : ℝ := sorry

axiom condition (a b : α) : abs (3 * dot a b) ≤ 4

theorem min_dot_product : dot a b = -4 / 3 :=
by
  sorry

end min_dot_product_l1534_153411


namespace quiz_answer_key_combinations_l1534_153427

noncomputable def num_ways_answer_key : ℕ :=
  let true_false_combinations := 2^4
  let valid_true_false_combinations := true_false_combinations - 2
  let multi_choice_combinations := 4 * 4
  valid_true_false_combinations * multi_choice_combinations

theorem quiz_answer_key_combinations : num_ways_answer_key = 224 := 
by
  sorry

end quiz_answer_key_combinations_l1534_153427


namespace difference_of_roots_of_quadratic_l1534_153408

theorem difference_of_roots_of_quadratic :
  (∃ (r1 r2 : ℝ), 3 * r1 ^ 2 + 4 * r1 - 15 = 0 ∧
                  3 * r2 ^ 2 + 4 * r2 - 15 = 0 ∧
                  r1 + r2 = -4 / 3 ∧
                  r1 * r2 = -5 ∧
                  r1 - r2 = 14 / 3) :=
sorry

end difference_of_roots_of_quadratic_l1534_153408


namespace find_n_l1534_153490

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 :=
by
  sorry

end find_n_l1534_153490


namespace new_figure_perimeter_equals_5_l1534_153480

-- Defining the side length of the square and the equilateral triangle
def side_length : ℝ := 1

-- Defining the perimeter of the new figure
def new_figure_perimeter : ℝ := 3 * side_length + 2 * side_length

-- Statement: The perimeter of the new figure equals 5
theorem new_figure_perimeter_equals_5 :
  new_figure_perimeter = 5 := by
  sorry

end new_figure_perimeter_equals_5_l1534_153480


namespace initial_contribution_amount_l1534_153496

variable (x : ℕ)
variable (workers : ℕ := 1200)
variable (total_with_extra_contribution: ℕ := 360000)
variable (extra_contribution_each: ℕ := 50)

theorem initial_contribution_amount :
  (workers * x = total_with_extra_contribution - workers * extra_contribution_each) →
  workers * x = 300000 :=
by
  intro h
  sorry

end initial_contribution_amount_l1534_153496


namespace polynomial_evaluation_l1534_153493

theorem polynomial_evaluation 
  (x : ℝ) (h : x^2 - 3*x - 10 = 0 ∧ x > 0) :
  x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 :=
sorry

end polynomial_evaluation_l1534_153493


namespace select_team_ways_l1534_153420

-- Definitions of the conditions and question
def boys := 7
def girls := 10
def boys_needed := 2
def girls_needed := 3
def total_team := 5

-- Theorem statement to prove the number of selecting the team
theorem select_team_ways : (Nat.choose boys boys_needed) * (Nat.choose girls girls_needed) = 2520 := 
by
  -- Place holder for proof
  sorry

end select_team_ways_l1534_153420


namespace valid_sentence_count_is_208_l1534_153433

def four_words := ["splargh", "glumph", "amr", "flark"]

def valid_sentence (sentence : List String) : Prop :=
  ¬(sentence.contains "glumph amr")

def count_valid_sentences : Nat :=
  let total_sentences := 4^4
  let invalid_sentences := 3 * 4 * 4
  total_sentences - invalid_sentences

theorem valid_sentence_count_is_208 :
  count_valid_sentences = 208 := by
  sorry

end valid_sentence_count_is_208_l1534_153433


namespace find_g_x_f_y_l1534_153443

-- Definition of the functions and conditions
variable (f g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1)

-- The theorem to prove
theorem find_g_x_f_y (x y : ℝ) : g (x + f y) = -x + y - 1 := 
sorry

end find_g_x_f_y_l1534_153443


namespace largest_int_lt_100_remainder_3_div_by_8_l1534_153424

theorem largest_int_lt_100_remainder_3_div_by_8 : 
  ∃ n, n < 100 ∧ n % 8 = 3 ∧ ∀ m, m < 100 ∧ m % 8 = 3 → m ≤ 99 := by
  sorry

end largest_int_lt_100_remainder_3_div_by_8_l1534_153424


namespace megan_folders_l1534_153495

def filesOnComputer : Nat := 93
def deletedFiles : Nat := 21
def filesPerFolder : Nat := 8

theorem megan_folders:
  let remainingFiles := filesOnComputer - deletedFiles
  (remainingFiles / filesPerFolder) = 9 := by
    sorry

end megan_folders_l1534_153495


namespace square_area_fraction_shaded_l1534_153455

theorem square_area_fraction_shaded (s : ℝ) :
  let R := (s / 2, s)
  let S := (s, s / 2)
  -- Area of triangle RSV
  let area_RSV := (1 / 2) * (s / 2) * (s * Real.sqrt 2 / 4)
  -- Non-shaded area
  let non_shaded_area := area_RSV
  -- Total area of the square
  let total_area := s^2
  -- Shaded area
  let shaded_area := total_area - non_shaded_area
  -- Fraction shaded
  (shaded_area / total_area) = 1 - Real.sqrt 2 / 16 :=
by
  sorry

end square_area_fraction_shaded_l1534_153455


namespace arrange_p_q_r_l1534_153470

theorem arrange_p_q_r (p : ℝ) (h : 1 < p ∧ p < 1.1) : p < p^p ∧ p^p < p^(p^p) :=
by
  sorry

end arrange_p_q_r_l1534_153470


namespace find_c_l1534_153453

theorem find_c
  (c d : ℝ)
  (h1 : ∀ (x : ℝ), 7 * x^3 + 3 * c * x^2 + 6 * d * x + c = 0)
  (h2 : ∀ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
        7 * p^3 + 3 * c * p^2 + 6 * d * p + c = 0 ∧ 
        7 * q^3 + 3 * c * q^2 + 6 * d * q + c = 0 ∧ 
        7 * r^3 + 3 * c * r^2 + 6 * d * r + c = 0 ∧ 
        Real.log (p * q * r) / Real.log 3 = 3) :
  c = -189 :=
sorry

end find_c_l1534_153453


namespace compare_constants_l1534_153482

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 2 / 2
noncomputable def c := Real.log 3 / 3

theorem compare_constants : b < c ∧ c < a := by
  sorry

end compare_constants_l1534_153482


namespace find_x_weeks_l1534_153417

-- Definition of the problem conditions:
def archibald_first_two_weeks_apples : Nat := 14
def archibald_next_x_weeks_apples (x : Nat) : Nat := 14
def archibald_last_two_weeks_apples : Nat := 42
def total_weeks : Nat := 7
def weekly_average : Nat := 10

-- Statement of the theorem to prove that x = 2 given the conditions
theorem find_x_weeks :
  ∃ x : Nat, (archibald_first_two_weeks_apples + archibald_next_x_weeks_apples x + archibald_last_two_weeks_apples = total_weeks * weekly_average) 
  ∧ (archibald_next_x_weeks_apples x / x = 7) 
  → x = 2 :=
by
  sorry

end find_x_weeks_l1534_153417


namespace discount_percentage_l1534_153492

theorem discount_percentage 
    (original_price : ℝ) 
    (total_paid : ℝ) 
    (sales_tax_rate : ℝ) 
    (sale_price_before_tax : ℝ) 
    (discount_amount : ℝ) 
    (discount_percentage : ℝ) :
    original_price = 200 → total_paid = 165 → sales_tax_rate = 0.10 →
    total_paid = sale_price_before_tax * (1 + sales_tax_rate) →
    sale_price_before_tax = original_price - discount_amount →
    discount_percentage = (discount_amount / original_price) * 100 →
    discount_percentage = 25 :=
by
  intros h_original h_total h_tax h_eq1 h_eq2 h_eq3
  sorry

end discount_percentage_l1534_153492


namespace additional_regular_gift_bags_needed_l1534_153415

-- Defining the conditions given in the question
def confirmed_guests : ℕ := 50
def additional_guests_70pc : ℕ := 30
def additional_guests_40pc : ℕ := 15
def probability_70pc : ℚ := 0.7
def probability_40pc : ℚ := 0.4
def extravagant_bags_prepared : ℕ := 10
def special_bags_prepared : ℕ := 25
def regular_bags_prepared : ℕ := 20

-- Defining the expected number of additional guests based on probabilities
def expected_guests_70pc : ℚ := additional_guests_70pc * probability_70pc
def expected_guests_40pc : ℚ := additional_guests_40pc * probability_40pc

-- Defining the total expected guests including confirmed guests and expected additional guests
def total_expected_guests : ℚ := confirmed_guests + expected_guests_70pc + expected_guests_40pc

-- Defining the problem statement in Lean, proving the additional regular gift bags needed
theorem additional_regular_gift_bags_needed : 
  total_expected_guests = 77 → regular_bags_prepared = 20 → 22 = 22 :=
by
  sorry

end additional_regular_gift_bags_needed_l1534_153415


namespace part1_extreme_value_at_2_part2_increasing_function_l1534_153491

noncomputable def f (a x : ℝ) := a * x - a / x - 2 * Real.log x

theorem part1_extreme_value_at_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y : ℝ, f a x ≥ f a y) → a = 4 / 5 ∧ f a 1/2 = 2 * Real.log 2 - 6 / 5 := by
  sorry

theorem part2_increasing_function (a : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (f a) x ≥ 0) → a ≥ 1 := by
  sorry

end part1_extreme_value_at_2_part2_increasing_function_l1534_153491


namespace min_value_expr_l1534_153487

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 :=
sorry

end min_value_expr_l1534_153487


namespace fruit_shop_problem_l1534_153465

variable (x y z : ℝ)

theorem fruit_shop_problem
  (h1 : x + 4 * y + 2 * z = 27.2)
  (h2 : 2 * x + 6 * y + 2 * z = 32.4) :
  x + 2 * y = 5.2 :=
by
  sorry

end fruit_shop_problem_l1534_153465


namespace largest_of_5_consecutive_odd_integers_l1534_153450

theorem largest_of_5_consecutive_odd_integers (n : ℤ) (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 235) :
  n + 8 = 51 :=
sorry

end largest_of_5_consecutive_odd_integers_l1534_153450


namespace farthings_in_a_pfennig_l1534_153430

theorem farthings_in_a_pfennig (x : ℕ) (h : 54 - 2 * x = 7 * x) : x = 6 :=
by
  sorry

end farthings_in_a_pfennig_l1534_153430


namespace factorize_expression_l1534_153469

theorem factorize_expression : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := 
by sorry

end factorize_expression_l1534_153469


namespace remainder_2007_div_81_l1534_153446

theorem remainder_2007_div_81 : 2007 % 81 = 63 :=
by
  sorry

end remainder_2007_div_81_l1534_153446


namespace library_books_difference_l1534_153458

theorem library_books_difference :
  let books_old_town := 750
  let books_riverview := 1240
  let books_downtown := 1800
  let books_eastside := 1620
  books_downtown - books_old_town = 1050 :=
by
  sorry

end library_books_difference_l1534_153458


namespace sum_of_possible_values_l1534_153422

variable (a b c d : ℝ)

theorem sum_of_possible_values
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) :
  (b - a) * (d - c) / ((c - b) * (a - d)) = -4 / 3 :=
sorry

end sum_of_possible_values_l1534_153422


namespace work_increase_percent_l1534_153435

theorem work_increase_percent (W p : ℝ) (p_pos : p > 0) :
  (1 / 3 * p) * W / ((2 / 3) * p) - (W / p) = 0.5 * (W / p) :=
by
  sorry

end work_increase_percent_l1534_153435


namespace no_real_solution_for_x_l1534_153478

theorem no_real_solution_for_x
  (y : ℝ)
  (x : ℝ)
  (h1 : y = (x^3 - 8) / (x - 2))
  (h2 : y = 3 * x) :
  ¬ ∃ x : ℝ, y = 3*x ∧ y = (x^3 - 8) / (x - 2) :=
by {
  sorry
}

end no_real_solution_for_x_l1534_153478


namespace container_capacity_l1534_153406

theorem container_capacity (C : ℝ) (h1 : C > 0) (h2 : 0.40 * C + 14 = 0.75 * C) : C = 40 := 
by 
  -- Would contain the proof here
  sorry

end container_capacity_l1534_153406


namespace exponential_equivalence_l1534_153474

theorem exponential_equivalence (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end exponential_equivalence_l1534_153474


namespace sum_of_terms_l1534_153441

noncomputable def arithmetic_sequence : Type :=
  {a : ℕ → ℤ // ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d}

theorem sum_of_terms (a : arithmetic_sequence) (h1 : a.val 1 + a.val 3 = 2) (h2 : a.val 3 + a.val 5 = 4) :
  a.val 5 + a.val 7 = 6 :=
by
  sorry

end sum_of_terms_l1534_153441


namespace chickens_at_stacy_farm_l1534_153481
-- Importing the necessary library

-- Defining the provided conditions and correct answer in Lean 4.
theorem chickens_at_stacy_farm (C : ℕ) (piglets : ℕ) (goats : ℕ) : 
  piglets = 40 → 
  goats = 34 → 
  (C + piglets + goats) = 2 * 50 → 
  C = 26 :=
by
  intros h_piglets h_goats h_animals
  sorry

end chickens_at_stacy_farm_l1534_153481


namespace hand_towels_in_set_l1534_153464

theorem hand_towels_in_set {h : ℕ}
  (hand_towel_sets : ℕ)
  (bath_towel_sets : ℕ)
  (hand_towel_sold : h * hand_towel_sets = 102)
  (bath_towel_sold : 6 * bath_towel_sets = 102)
  (same_sets_sold : hand_towel_sets = bath_towel_sets) :
  h = 17 := 
sorry

end hand_towels_in_set_l1534_153464


namespace sum_binomials_eq_l1534_153418

theorem sum_binomials_eq : 
  (Nat.choose 6 1) + (Nat.choose 6 2) + (Nat.choose 6 3) + (Nat.choose 6 4) + (Nat.choose 6 5) = 62 :=
by
  sorry

end sum_binomials_eq_l1534_153418


namespace compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l1534_153484

def a_n (p n : ℕ) : ℕ := (2 * n + 1) ^ p
def b_n (p n : ℕ) : ℕ := (2 * n) ^ p + (2 * n - 1) ^ p

theorem compare_magnitude_p2_for_n1 :
  b_n 2 1 < a_n 2 1 := sorry

theorem compare_magnitude_p2_for_n2 :
  b_n 2 2 = a_n 2 2 := sorry

theorem compare_magnitude_p2_for_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  b_n 2 n > a_n 2 n := sorry

theorem compare_magnitude_p_eq_n_for_all_n (n : ℕ) :
  a_n n n ≥ b_n n n := sorry

end compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l1534_153484


namespace combinations_of_painting_options_l1534_153445

theorem combinations_of_painting_options : 
  let colors := 6
  let methods := 3
  let finishes := 2
  colors * methods * finishes = 36 := by
  sorry

end combinations_of_painting_options_l1534_153445


namespace crayons_left_l1534_153428

def initial_green_crayons : ℝ := 5
def initial_blue_crayons : ℝ := 8
def initial_yellow_crayons : ℝ := 7
def given_green_crayons : ℝ := 3.5
def given_blue_crayons : ℝ := 1.25
def given_yellow_crayons : ℝ := 2.75
def broken_yellow_crayons : ℝ := 0.5

theorem crayons_left (initial_green_crayons initial_blue_crayons initial_yellow_crayons given_green_crayons given_blue_crayons given_yellow_crayons broken_yellow_crayons : ℝ) :
  initial_green_crayons - given_green_crayons + 
  initial_blue_crayons - given_blue_crayons + 
  initial_yellow_crayons - given_yellow_crayons - broken_yellow_crayons = 12 :=
by
  sorry

end crayons_left_l1534_153428


namespace fan_working_time_each_day_l1534_153444

theorem fan_working_time_each_day
  (airflow_per_second : ℝ)
  (total_airflow_week : ℝ)
  (seconds_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (airy_sector: airflow_per_second = 10)
  (flow_week : total_airflow_week = 42000)
  (sec_per_hr : seconds_per_hour = 3600)
  (hrs_per_day : hours_per_day = 24)
  (days_week : days_per_week = 7) :
  let airflow_per_hour := airflow_per_second * seconds_per_hour
  let total_hours_week := total_airflow_week / airflow_per_hour
  let hours_per_day_given := total_hours_week / days_per_week
  let minutes_per_day := hours_per_day_given * 60
  minutes_per_day = 10 := 
by
  sorry

end fan_working_time_each_day_l1534_153444


namespace project_completion_time_l1534_153401

theorem project_completion_time (x : ℕ) :
  (∀ (B_days : ℕ), B_days = 40 →
  (∀ (combined_work_days : ℕ), combined_work_days = 10 →
  (∀ (total_days : ℕ), total_days = 20 →
  10 * (1 / (x : ℚ) + 1 / 40) + 10 * (1 / 40) = 1))) →
  x = 20 :=
by
  sorry

end project_completion_time_l1534_153401


namespace sqrt_eight_simplify_l1534_153498

theorem sqrt_eight_simplify : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_simplify_l1534_153498


namespace num_female_managers_l1534_153494

-- Definitions based on the conditions
def total_employees : ℕ := 250
def female_employees : ℕ := 90
def total_managers : ℕ := 40
def male_associates : ℕ := 160

-- Proof statement that computes the number of female managers
theorem num_female_managers : 
  (total_managers - (total_employees - female_employees - male_associates)) = 40 := 
by 
  sorry

end num_female_managers_l1534_153494


namespace doberman_puppies_count_l1534_153429

theorem doberman_puppies_count (D : ℝ) (S : ℝ) (h1 : S = 55) (h2 : 3 * D - 5 + (D - S) = 90) : D = 37.5 :=
by
  sorry

end doberman_puppies_count_l1534_153429


namespace eval_f_at_4_l1534_153460

def f (x : ℕ) : ℕ := 5 * x + 2

theorem eval_f_at_4 : f 4 = 22 :=
by
  sorry

end eval_f_at_4_l1534_153460


namespace xy_sum_eq_16_l1534_153468

theorem xy_sum_eq_16 (x y : ℕ) (h1: x > 0) (h2: y > 0) (h3: x < 20) (h4: y < 20) (h5: x + y + x * y = 76) : x + y = 16 :=
  sorry

end xy_sum_eq_16_l1534_153468


namespace adjacent_girl_pairs_l1534_153497

variable (boyCount girlCount : ℕ) 
variable (adjacentBoyPairs adjacentGirlPairs: ℕ)

theorem adjacent_girl_pairs
  (h1 : boyCount = 10)
  (h2 : girlCount = 15)
  (h3 : adjacentBoyPairs = 5) :
  adjacentGirlPairs = 10 :=
sorry

end adjacent_girl_pairs_l1534_153497


namespace max_puzzle_sets_l1534_153432

theorem max_puzzle_sets 
  (total_logic : ℕ) (total_visual : ℕ) (total_word : ℕ)
  (h1 : total_logic = 36) (h2 : total_visual = 27) (h3 : total_word = 15)
  (x y : ℕ)
  (h4 : 7 ≤ 4 * x + 3 * x + y ∧ 4 * x + 3 * x + y ≤ 12)
  (h5 : 4 * x / 3 * x = 4 / 3)
  (h6 : y ≥ 3 * x / 2) :
  5 ≤ total_logic / (4 * x) ∧ 5 ≤ total_visual / (3 * x) ∧ 5 ≤ total_word / y :=
sorry

end max_puzzle_sets_l1534_153432


namespace central_angle_measure_l1534_153463

-- Constants representing the arc length and the area of the sector.
def arc_length : ℝ := 5
def sector_area : ℝ := 5

-- Variables representing the central angle in radians and the radius.
variable (α r : ℝ)

-- Conditions given in the problem.
axiom arc_length_eq : arc_length = α * r
axiom sector_area_eq : sector_area = 1 / 2 * α * r^2

-- The goal to prove that the radian measure of the central angle α is 5 / 2.
theorem central_angle_measure : α = 5 / 2 := by sorry

end central_angle_measure_l1534_153463


namespace range_of_a_l1534_153457

noncomputable def f (a x : ℝ) : ℝ := Real.log x + 2 * a * (1 - x)

theorem range_of_a (a : ℝ) :
  (∀ x, x > 2 → f a x > f a 2) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici (1 / 4)) :=
by
  sorry

end range_of_a_l1534_153457


namespace lcm_technicians_schedule_l1534_153431

theorem lcm_technicians_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := 
sorry

end lcm_technicians_schedule_l1534_153431


namespace inequality_proof_l1534_153454

theorem inequality_proof 
  (a b c d : ℝ) (n : ℕ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_n : 9 ≤ n) :
  a^n + b^n + c^n + d^n ≥ a^(n-9)*b^4*c^3*d^2 + b^(n-9)*c^4*d^3*a^2 + c^(n-9)*d^4*a^3*b^2 + d^(n-9)*a^4*b^3*c^2 :=
by
  sorry

end inequality_proof_l1534_153454


namespace percentage_of_black_marbles_l1534_153489

variable (T : ℝ) -- Total number of marbles
variable (C : ℝ) -- Number of clear marbles
variable (B : ℝ) -- Number of black marbles
variable (O : ℝ) -- Number of other colored marbles

-- Conditions
def condition1 := C = 0.40 * T
def condition2 := O = (2 / 5) * T
def condition3 := C + B + O = T

-- Proof statement
theorem percentage_of_black_marbles :
  C = 0.40 * T → O = (2 / 5) * T → C + B + O = T → B = 0.20 * T :=
by
  intros hC hO hTotal
  -- Intermediate steps would go here, but we use sorry to skip the proof.
  sorry

end percentage_of_black_marbles_l1534_153489


namespace evaluate_f_3_minus_f_neg3_l1534_153462

def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

theorem evaluate_f_3_minus_f_neg3 : f 3 - f (-3) = 210 := by
  sorry

end evaluate_f_3_minus_f_neg3_l1534_153462
