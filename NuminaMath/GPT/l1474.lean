import Mathlib

namespace sequence_is_geometric_and_general_formula_l1474_147496

theorem sequence_is_geometric_and_general_formula (a : ℕ → ℝ) (h0 : a 1 = 2 / 3)
  (h1 : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (a (n + 1) + 1)) :
  ∃ r : ℝ, (0 < r ∧ r < 1 ∧ (∀ n : ℕ, a (n + 1) = (2:ℝ)^n / (1 + (2:ℝ)^n)) ∧
  ∀ n : ℕ, (1 / a (n + 1) - 1) = (1 / 2) * (1 / a n - 1)) := sorry

end sequence_is_geometric_and_general_formula_l1474_147496


namespace find_a_if_parallel_l1474_147444

-- Define the parallel condition for the given lines
def is_parallel (a : ℝ) : Prop :=
  let slope1 := -a / 2
  let slope2 := 3
  slope1 = slope2

-- Prove that a = -6 under the parallel condition
theorem find_a_if_parallel (a : ℝ) (h : is_parallel a) : a = -6 := by
  sorry

end find_a_if_parallel_l1474_147444


namespace marbles_per_friend_l1474_147479

variable (initial_marbles remaining_marbles given_marbles_per_friend : ℕ)

-- conditions in a)
def condition_initial_marbles := initial_marbles = 500
def condition_remaining_marbles := 4 * remaining_marbles = 720
def condition_total_given_marbles := initial_marbles - remaining_marbles = 320
def condition_given_marbles_per_friend := given_marbles_per_friend * 4 = 320

-- question proof goal
theorem marbles_per_friend (initial_marbles: ℕ) (remaining_marbles: ℕ) (given_marbles_per_friend: ℕ) :
  (condition_initial_marbles initial_marbles) →
  (condition_remaining_marbles remaining_marbles) →
  (condition_total_given_marbles initial_marbles remaining_marbles) →
  (condition_given_marbles_per_friend given_marbles_per_friend) →
  given_marbles_per_friend = 80 :=
by
  intros hinitial hremaining htotal_given hgiven_per_friend
  sorry

end marbles_per_friend_l1474_147479


namespace angle_measure_l1474_147436

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l1474_147436


namespace one_inch_cubes_with_red_paint_at_least_two_faces_l1474_147405

theorem one_inch_cubes_with_red_paint_at_least_two_faces
  (number_of_one_inch_cubes : ℕ)
  (cubes_with_three_faces : ℕ)
  (cubes_with_two_faces : ℕ)
  (total_cubes_with_at_least_two_faces : ℕ) :
  number_of_one_inch_cubes = 64 →
  cubes_with_three_faces = 8 →
  cubes_with_two_faces = 24 →
  total_cubes_with_at_least_two_faces = cubes_with_three_faces + cubes_with_two_faces →
  total_cubes_with_at_least_two_faces = 32 :=
by
  sorry

end one_inch_cubes_with_red_paint_at_least_two_faces_l1474_147405


namespace find_EF_squared_l1474_147473

noncomputable def square_side := 15
noncomputable def BE := 6
noncomputable def DF := 6
noncomputable def AE := 14
noncomputable def CF := 14

theorem find_EF_squared (A B C D E F : ℝ) (AB BC CD DA : ℝ := square_side) :
  (BE = 6) → (DF = 6) → (AE = 14) → (CF = 14) → EF^2 = 72 :=
by
  -- Definitions and conditions usage according to (a)
  sorry

end find_EF_squared_l1474_147473


namespace necessary_but_not_sufficient_condition_l1474_147454

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (m < 1) → (∀ x y : ℝ, (x - m) ^ 2 + y ^ 2 = m ^ 2 → (x, y) ≠ (1, 1)) :=
sorry

end necessary_but_not_sufficient_condition_l1474_147454


namespace probability_of_two_sunny_days_l1474_147402

def prob_two_sunny_days (prob_sunny prob_rain : ℚ) (days : ℕ) : ℚ :=
  (days.choose 2) * (prob_sunny^2 * prob_rain^(days-2))

theorem probability_of_two_sunny_days :
  prob_two_sunny_days (2/5) (3/5) 3 = 36/125 :=
by 
  sorry

end probability_of_two_sunny_days_l1474_147402


namespace Mary_ends_with_31_eggs_l1474_147461

theorem Mary_ends_with_31_eggs (a b : ℕ) (h1 : a = 27) (h2 : b = 4) : a + b = 31 := by
  sorry

end Mary_ends_with_31_eggs_l1474_147461


namespace expression_for_f_pos_f_monotone_on_pos_l1474_147485

section

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_neg : ∀ x, -1 ≤ x ∧ x < 0 → f x = 2 * x + 1 / x^2)

-- Part 1: Prove the expression for f(x) when x ∈ (0,1]
theorem expression_for_f_pos (x : ℝ) (hx : 0 < x ∧ x ≤ 1) : 
  f x = 2 * x - 1 / x^2 :=
sorry

-- Part 2: Prove the monotonicity of f(x) on (0,1]
theorem f_monotone_on_pos : 
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y ≤ 1 → f x < f y :=
sorry

end

end expression_for_f_pos_f_monotone_on_pos_l1474_147485


namespace refund_amount_l1474_147423

def income_tax_paid : ℝ := 156000
def education_expenses : ℝ := 130000
def medical_expenses : ℝ := 10000
def tax_rate : ℝ := 0.13

def eligible_expenses : ℝ := education_expenses + medical_expenses
def max_refund : ℝ := tax_rate * eligible_expenses

theorem refund_amount : min (max_refund) (income_tax_paid) = 18200 := by
  sorry

end refund_amount_l1474_147423


namespace eight_letter_good_words_l1474_147430

-- Definition of a good word sequence (only using A, B, and C)
inductive Letter
| A | B | C

-- Define the restriction condition for a good word
def is_valid_transition (a b : Letter) : Prop :=
  match a, b with
  | Letter.A, Letter.B => False
  | Letter.B, Letter.C => False
  | Letter.C, Letter.A => False
  | _, _ => True

-- Count the number of 8-letter good words
def count_good_words : ℕ :=
  let letters := [Letter.A, Letter.B, Letter.C]
  -- Initial 3 choices for the first letter
  let first_choices := letters.length
  -- Subsequent 7 letters each have 2 valid previous choices
  let subsequent_choices := 2 ^ 7
  first_choices * subsequent_choices

theorem eight_letter_good_words : count_good_words = 384 :=
by
  sorry

end eight_letter_good_words_l1474_147430


namespace curve_equation_with_params_l1474_147407

theorem curve_equation_with_params (a m x y : ℝ) (ha : a > 0) (hm : m ≠ 0) :
    (y^2) = m * (x^2 - a^2) ↔ mx^2 - y^2 = ma^2 := by
  sorry

end curve_equation_with_params_l1474_147407


namespace andy_late_minutes_l1474_147493

theorem andy_late_minutes (school_starts_at : Nat) (normal_travel_time : Nat) 
  (stop_per_light : Nat) (red_lights : Nat) (construction_wait : Nat) 
  (left_house_at : Nat) : 
  let total_delay := (stop_per_light * red_lights) + construction_wait
  let total_travel_time := normal_travel_time + total_delay
  let arrive_time := left_house_at + total_travel_time
  let late_time := arrive_time - school_starts_at
  late_time = 7 :=
by
  sorry

end andy_late_minutes_l1474_147493


namespace Felipe_time_to_build_house_l1474_147453

variables (E F : ℝ)
variables (Felipe_building_time_months : ℝ) (Combined_time : ℝ := 7.5) (Half_time_relation : F = 1 / 2 * E)

-- Felipe finished his house in 30 months
theorem Felipe_time_to_build_house :
  (F = 1 / 2 * E) →
  (F + E = Combined_time) →
  (Felipe_building_time_months = F * 12) →
  Felipe_building_time_months = 30 :=
by
  intros h1 h2 h3
  -- Combining the given conditions to prove the statement
  sorry

end Felipe_time_to_build_house_l1474_147453


namespace socks_combinations_correct_l1474_147495

noncomputable def num_socks_combinations (colors patterns pairs : ℕ) : ℕ :=
  colors * (colors - 1) * patterns * (patterns - 1)

theorem socks_combinations_correct :
  num_socks_combinations 5 4 20 = 240 :=
by
  sorry

end socks_combinations_correct_l1474_147495


namespace range_of_m_l1474_147400

theorem range_of_m (m : ℝ) :
  (∀ P : ℝ × ℝ, P.2 = 2 * P.1 + m → (abs (P.1^2 + (P.2 - 1)^2) = (1/2) * abs (P.1^2 + (P.2 - 4)^2)) → (-2 * Real.sqrt 5) ≤ m ∧ m ≤ (2 * Real.sqrt 5)) :=
sorry

end range_of_m_l1474_147400


namespace find_highest_score_l1474_147425

theorem find_highest_score (average innings : ℕ) (avg_excl_two innings_excl_two H L : ℕ)
  (diff_high_low total_runs total_excl_two : ℕ)
  (h1 : diff_high_low = 150)
  (h2 : total_runs = average * innings)
  (h3 : total_excl_two = avg_excl_two * innings_excl_two)
  (h4 : total_runs - total_excl_two = H + L)
  (h5 : H - L = diff_high_low)
  (h6 : average = 62)
  (h7 : innings = 46)
  (h8 : avg_excl_two = 58)
  (h9 : innings_excl_two = 44)
  (h10 : total_runs = 2844)
  (h11 : total_excl_two = 2552) :
  H = 221 :=
by
  sorry

end find_highest_score_l1474_147425


namespace fifth_number_in_pascals_triangle_l1474_147437

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l1474_147437


namespace probability_10_coins_at_most_3_heads_l1474_147480

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l1474_147480


namespace pasta_ratio_l1474_147492

theorem pasta_ratio (total_students : ℕ) (spaghetti : ℕ) (manicotti : ℕ) 
  (h1 : total_students = 650) 
  (h2 : spaghetti = 250) 
  (h3 : manicotti = 100) : 
  (spaghetti : ℤ) / (manicotti : ℤ) = 5 / 2 :=
by
  sorry

end pasta_ratio_l1474_147492


namespace part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l1474_147469

theorem part_a_max_cells_crossed (m n : ℕ) : 
  ∃ max_cells : ℕ, max_cells = m + n - 1 := sorry

theorem part_b_max_cells_crossed_by_needle : 
  ∃ max_cells : ℕ, max_cells = 285 := sorry

end part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l1474_147469


namespace lisa_and_robert_total_photos_l1474_147441

def claire_photos : Nat := 10
def lisa_photos (c : Nat) : Nat := 3 * c
def robert_photos (c : Nat) : Nat := c + 20

theorem lisa_and_robert_total_photos :
  let c := claire_photos
  let l := lisa_photos c
  let r := robert_photos c
  l + r = 60 :=
by
  sorry

end lisa_and_robert_total_photos_l1474_147441


namespace total_coins_l1474_147487
-- Import the necessary library

-- Defining the conditions
def quarters := 22
def dimes := quarters + 3
def nickels := quarters - 6

-- Main theorem statement
theorem total_coins : (quarters + dimes + nickels) = 63 := by
  sorry

end total_coins_l1474_147487


namespace log_base_equal_l1474_147488

noncomputable def logx (b x : ℝ) := Real.log x / Real.log b

theorem log_base_equal {x : ℝ} (h : 0 < x ∧ x ≠ 1) :
  logx 81 x = logx 16 2 → x = 3 :=
by
  intro h1
  sorry

end log_base_equal_l1474_147488


namespace unique_value_expression_l1474_147404

theorem unique_value_expression (m n : ℤ) : 
  (mn + 13 * m + 13 * n - m^2 - n^2 = 169) → 
  ∃! (m n : ℤ), mn + 13 * m + 13 * n - m^2 - n^2 = 169 := 
by
  sorry

end unique_value_expression_l1474_147404


namespace factorization_correct_l1474_147464

theorem factorization_correct (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end factorization_correct_l1474_147464


namespace highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l1474_147483

theorem highest_power_of_two_factor_13_pow_4_minus_11_pow_4 :
  ∃ n : ℕ, n = 5 ∧ (2 ^ n ∣ (13 ^ 4 - 11 ^ 4)) ∧ ¬ (2 ^ (n + 1) ∣ (13 ^ 4 - 11 ^ 4)) :=
sorry

end highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l1474_147483


namespace quadratic_sum_solutions_l1474_147446

theorem quadratic_sum_solutions {a b : ℝ} (h : a ≥ b) (h1: a = 1 + Real.sqrt 17) (h2: b = 1 - Real.sqrt 17) :
  3 * a + 2 * b = 5 + Real.sqrt 17 := by
  sorry

end quadratic_sum_solutions_l1474_147446


namespace can_form_triangle_l1474_147448

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle :
  (is_triangle 3 5 7) ∧ ¬(is_triangle 3 3 7) ∧ ¬(is_triangle 4 4 8) ∧ ¬(is_triangle 4 5 9) :=
by
  -- Proof steps will be added here
  sorry

end can_form_triangle_l1474_147448


namespace calc_log_expression_l1474_147457

theorem calc_log_expression : 2 * Real.log 5 + Real.log 4 = 2 :=
by
  sorry

end calc_log_expression_l1474_147457


namespace necessary_but_not_sufficient_condition_l1474_147418

-- Prove that x^2 ≥ -x is a necessary but not sufficient condition for |x| = x
theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 ≥ -x → |x| = x ↔ x ≥ 0 := 
sorry

end necessary_but_not_sufficient_condition_l1474_147418


namespace number_of_real_roots_l1474_147456

theorem number_of_real_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b^2 + 1 = 0) :
  (c > 0 → ∃ x1 x2 x3 : ℝ, 
    (x1 = b * Real.sqrt c ∨ x1 = -b * Real.sqrt c ∨ x1 = -c / b) ∧
    (x2 = b * Real.sqrt c ∨ x2 = -b * Real.sqrt c ∨ x2 = -c / b) ∧
    (x3 = b * Real.sqrt c ∨ x3 = -b * Real.sqrt c ∨ x3 = -c / b)) ∧
  (c < 0 → ∃ x1 : ℝ, x1 = -c / b) :=
by
  sorry

end number_of_real_roots_l1474_147456


namespace distance_light_travels_100_years_l1474_147440

def distance_light_travels_one_year : ℝ := 5870e9 * 10^3

theorem distance_light_travels_100_years : distance_light_travels_one_year * 100 = 587 * 10^12 :=
by
  rw [distance_light_travels_one_year]
  sorry

end distance_light_travels_100_years_l1474_147440


namespace b_squared_gt_4ac_l1474_147455

theorem b_squared_gt_4ac (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
by
  sorry

end b_squared_gt_4ac_l1474_147455


namespace solve_for_x_l1474_147484

theorem solve_for_x (x : ℤ) (h : x + 1 = 10) : x = 9 := 
by 
  sorry

end solve_for_x_l1474_147484


namespace annual_interest_rate_is_10_percent_l1474_147463

noncomputable def principal (P : ℝ) := P = 1500
noncomputable def total_amount (A : ℝ) := A = 1815
noncomputable def time_period (t : ℝ) := t = 2
noncomputable def compounding_frequency (n : ℝ) := n = 1
noncomputable def interest_rate_compound_interest_formula (P A t n : ℝ) (r : ℝ) := 
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate_is_10_percent : 
  ∀ (P A t n : ℝ) (r : ℝ), principal P → total_amount A → time_period t → compounding_frequency n → 
  interest_rate_compound_interest_formula P A t n r → r = 0.1 :=
by
  intros P A t n r hP hA ht hn h_formula
  sorry

end annual_interest_rate_is_10_percent_l1474_147463


namespace rabbitAgeOrder_l1474_147415

-- Define the ages of the rabbits as variables
variables (blue black red gray : ℕ)

-- Conditions based on the problem statement
noncomputable def rabbitConditions := 
  (blue ≠ max blue (max black (max red gray))) ∧  -- The blue-eyed rabbit is not the eldest
  (gray ≠ min blue (min black (min red gray))) ∧  -- The gray rabbit is not the youngest
  (red ≠ min blue (min black (min red gray))) ∧  -- The red-eyed rabbit is not the youngest
  (black > red) ∧ (gray > black)  -- The black rabbit is older than the red-eyed rabbit and younger than the gray rabbit

-- Required proof statement
theorem rabbitAgeOrder : rabbitConditions blue black red gray → gray > black ∧ black > red ∧ red > blue :=
by
  intro h
  sorry

end rabbitAgeOrder_l1474_147415


namespace digit_in_452nd_place_l1474_147442

def repeating_sequence : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]
def repeat_length : Nat := 18

theorem digit_in_452nd_place :
  (repeating_sequence.get ⟨(452 % repeat_length) - 1, sorry⟩ = 6) :=
sorry

end digit_in_452nd_place_l1474_147442


namespace ratio_of_a_over_3_to_b_over_2_l1474_147476

theorem ratio_of_a_over_3_to_b_over_2 (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
by
  sorry

end ratio_of_a_over_3_to_b_over_2_l1474_147476


namespace tom_mileage_per_gallon_l1474_147494

-- Definitions based on the given conditions
def daily_mileage : ℕ := 75
def cost_per_gallon : ℕ := 3
def amount_spent_in_10_days : ℕ := 45
def days : ℕ := 10

-- Main theorem to prove
theorem tom_mileage_per_gallon : 
  (amount_spent_in_10_days / cost_per_gallon) * 75 * days = 50 :=
by
  sorry

end tom_mileage_per_gallon_l1474_147494


namespace units_digit_17_times_29_l1474_147431

theorem units_digit_17_times_29 :
  (17 * 29) % 10 = 3 :=
by
  sorry

end units_digit_17_times_29_l1474_147431


namespace number_of_girls_l1474_147458

-- Given conditions
def ratio_girls_boys_teachers (girls boys teachers : ℕ) : Prop :=
  3 * (girls + boys + teachers) = 3 * girls + 2 * boys + 1 * teachers

def total_people (total girls boys teachers : ℕ) : Prop :=
  total = girls + boys + teachers

-- Define the main theorem
theorem number_of_girls 
  (k total : ℕ)
  (h1 : ratio_girls_boys_teachers (3 * k) (2 * k) k)
  (h2 : total_people total (3 * k) (2 * k) k)
  (h_total : total = 60) : 
  3 * k = 30 :=
  sorry

end number_of_girls_l1474_147458


namespace part_a_part_b_l1474_147472

def balanced (V : Finset (ℝ × ℝ)) : Prop :=
  ∀ (A B : ℝ × ℝ), A ∈ V → B ∈ V → A ≠ B → ∃ C : ℝ × ℝ, C ∈ V ∧ (dist C A = dist C B)

def center_free (V : Finset (ℝ × ℝ)) : Prop :=
  ¬ ∃ (A B C P : ℝ × ℝ), A ∈ V → B ∈ V → C ∈ V → P ∈ V →
                         A ≠ B ∧ B ≠ C ∧ A ≠ C →
                         (dist P A = dist P B ∧ dist P B = dist P C)

theorem part_a (n : ℕ) (hn : 3 ≤ n) :
  ∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V :=
by sorry

theorem part_b : ∀ n : ℕ, 3 ≤ n →
  (∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V ∧ center_free V ↔ n % 2 = 1) :=
by sorry

end part_a_part_b_l1474_147472


namespace cookies_difference_l1474_147432

-- Define the initial conditions
def initial_cookies : ℝ := 57
def cookies_eaten : ℝ := 8.5
def cookies_bought : ℝ := 125.75

-- Problem statement
theorem cookies_difference (initial_cookies cookies_eaten cookies_bought : ℝ) : 
  cookies_bought - cookies_eaten = 117.25 := 
sorry

end cookies_difference_l1474_147432


namespace value_of_x_and_z_l1474_147490

theorem value_of_x_and_z (x y z : ℤ) (h1 : x / y = 7 / 3) (h2 : y = 21) (h3 : z = 3 * y) : x = 49 ∧ z = 63 :=
by
  sorry

end value_of_x_and_z_l1474_147490


namespace mask_price_reduction_l1474_147467

theorem mask_price_reduction 
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (additional_sales_factor : ℝ)
  (desired_profit : ℝ)
  (x : ℝ)
  (h_initial_sales : initial_sales = 500)
  (h_initial_profit : initial_profit = 0.6)
  (h_additional_sales_factor : additional_sales_factor = 100 / 0.1)
  (h_desired_profit : desired_profit = 240) :
  (initial_profit - x) * (initial_sales + additional_sales_factor * x) = desired_profit → x = 0.3 :=
sorry

end mask_price_reduction_l1474_147467


namespace hyperbola_foci_coordinates_l1474_147422

theorem hyperbola_foci_coordinates :
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 → (x, y) = (4, 0) ∨ (x, y) = (-4, 0) :=
by
  -- We assume the given equation of the hyperbola
  intro x y h
  -- sorry is used to skip the actual proof steps
  sorry

end hyperbola_foci_coordinates_l1474_147422


namespace tan_angle_PAB_correct_l1474_147460

noncomputable def tan_angle_PAB (AB BC CA : ℝ) (P inside ABC : Prop) (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop) : ℝ :=
  180 / 329

theorem tan_angle_PAB_correct :
  ∀ (AB BC CA : ℝ)
    (P_inside_ABC : Prop)
    (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop),
    AB = 12 → BC = 15 → CA = 17 →
    (tan_angle_PAB AB BC CA P_inside_ABC PAB_angle_eq_PBC_angle_eq_PCA_angle) = 180 / 329 :=
by
  intros
  sorry

end tan_angle_PAB_correct_l1474_147460


namespace fill_pipe_fraction_l1474_147428

theorem fill_pipe_fraction (t : ℕ) (f : ℝ) (h : t = 30) (h' : f = 1) : f = 1 :=
by
  sorry

end fill_pipe_fraction_l1474_147428


namespace largest_divisor_is_15_l1474_147497

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def largest_divisor (n : ℕ) : ℕ :=
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)

theorem largest_divisor_is_15 : ∀ (n : ℕ), n > 0 → is_even n → 15 ∣ largest_divisor n ∧ (∀ m, m ∣ largest_divisor n → m ≤ 15) :=
by
  intros n pos even
  sorry

end largest_divisor_is_15_l1474_147497


namespace min_value_of_f_l1474_147478

noncomputable def f (x : ℝ) : ℝ := x^2 + 8 * x + 3

theorem min_value_of_f : ∃ x₀ : ℝ, (∀ x : ℝ, f x ≥ f x₀) ∧ f x₀ = -13 :=
by
  sorry

end min_value_of_f_l1474_147478


namespace f_f_2_eq_2_l1474_147466

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 2 * Real.exp (x - 1)
else Real.log (x ^ 2 - 1) / Real.log 3

theorem f_f_2_eq_2 : f (f 2) = 2 :=
by
  sorry

end f_f_2_eq_2_l1474_147466


namespace find_temperature_on_friday_l1474_147459

variable (M T W Th F : ℕ)

def problem_conditions : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 46 ∧
  M = 44

theorem find_temperature_on_friday (h : problem_conditions M T W Th F) : F = 36 := by
  sorry

end find_temperature_on_friday_l1474_147459


namespace g_at_3_l1474_147475

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) - x * g (3 ^ (-x)) = x) : g 3 = 0 :=
by
  sorry

end g_at_3_l1474_147475


namespace contrapositive_of_original_l1474_147414

theorem contrapositive_of_original (a b : ℝ) :
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
by
  sorry

end contrapositive_of_original_l1474_147414


namespace part1_part2_l1474_147403

open Set

variable (A B : Set ℝ) (m : ℝ)

def setA : Set ℝ := {x | x ^ 2 - 2 * x - 8 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | x ^ 2 - (2 * m - 3) * x + m ^ 2 - 3 * m ≤ 0}

theorem part1 (h : (setA ∩ setB 5) = Icc 2 4) : m = 5 := sorry

theorem part2 (h : setA ⊆ compl (setB m)) :
  m ∈ Iio (-2) ∪ Ioi 7 := sorry

end part1_part2_l1474_147403


namespace train_length_is_correct_l1474_147447

noncomputable def speed_kmhr : ℝ := 45
noncomputable def time_sec : ℝ := 30
noncomputable def bridge_length_m : ℝ := 235

noncomputable def speed_ms : ℝ := (speed_kmhr * 1000) / 3600
noncomputable def total_distance_m : ℝ := speed_ms * time_sec
noncomputable def train_length_m : ℝ := total_distance_m - bridge_length_m

theorem train_length_is_correct : train_length_m = 140 :=
by
  -- Placeholder to indicate that a proof should go here
  -- Proof is omitted as per the instructions
  sorry

end train_length_is_correct_l1474_147447


namespace hamburgers_left_over_l1474_147416

theorem hamburgers_left_over (h_made : ℕ) (h_served : ℕ) (h_total : h_made = 9) (h_served_count : h_served = 3) : h_made - h_served = 6 :=
by
  sorry

end hamburgers_left_over_l1474_147416


namespace largest_even_digit_multiple_of_five_l1474_147438

theorem largest_even_digit_multiple_of_five : ∃ n : ℕ, n = 8860 ∧ n < 10000 ∧ (∀ digit ∈ (n.digits 10), digit % 2 = 0) ∧ n % 5 = 0 :=
by
  sorry

end largest_even_digit_multiple_of_five_l1474_147438


namespace no_valid_triples_l1474_147489

theorem no_valid_triples (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : 6 * (a * b + b * c + c * a) = a * b * c) : false :=
by
  sorry

end no_valid_triples_l1474_147489


namespace sum_of_coefficients_l1474_147406

theorem sum_of_coefficients :
  ∃ a b c d e : ℤ, 
    27 * (x : ℝ)^3 + 64 = (a * x + b) * (c * x^2 + d * x + e) ∧ 
    a + b + c + d + e = 20 :=
by
  sorry

end sum_of_coefficients_l1474_147406


namespace carols_rectangle_length_l1474_147445

theorem carols_rectangle_length :
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  carol_length = 5 :=
by
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  show carol_length = 5
  sorry

end carols_rectangle_length_l1474_147445


namespace juice_fraction_left_l1474_147420

theorem juice_fraction_left (initial_juice : ℝ) (given_juice : ℝ) (remaining_juice : ℝ) : 
  initial_juice = 5 → given_juice = 18/4 → remaining_juice = initial_juice - given_juice → remaining_juice = 1/2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end juice_fraction_left_l1474_147420


namespace participants_who_drank_neither_l1474_147427

-- Conditions
variables (total_participants : ℕ) (coffee_drinkers : ℕ) (juice_drinkers : ℕ) (both_drinkers : ℕ)

-- Initial Facts from the Conditions
def conditions := total_participants = 30 ∧ coffee_drinkers = 15 ∧ juice_drinkers = 18 ∧ both_drinkers = 7

-- The statement to prove
theorem participants_who_drank_neither : conditions total_participants coffee_drinkers juice_drinkers both_drinkers → 
  (total_participants - (coffee_drinkers + juice_drinkers - both_drinkers)) = 4 :=
by
  intros
  sorry

end participants_who_drank_neither_l1474_147427


namespace min_sticks_to_avoid_rectangles_l1474_147452

noncomputable def min_stick_deletions (n : ℕ) : ℕ :=
  if n = 8 then 43 else 0 -- we define 43 as the minimum for an 8x8 chessboard

theorem min_sticks_to_avoid_rectangles : min_stick_deletions 8 = 43 :=
  by
    sorry

end min_sticks_to_avoid_rectangles_l1474_147452


namespace largest_possible_sum_l1474_147409

theorem largest_possible_sum (clubsuit heartsuit : ℕ) (h₁ : clubsuit * heartsuit = 48) (h₂ : Even clubsuit) : 
  clubsuit + heartsuit ≤ 26 :=
sorry

end largest_possible_sum_l1474_147409


namespace certain_positive_integer_value_l1474_147417

theorem certain_positive_integer_value :
  ∃ (i m p : ℕ), (x = 2 ^ i * 3 ^ 2 * 5 ^ m * 7 ^ p) ∧ (i + 2 + m + p = 11) :=
by
  let x := 40320 -- 8!
  sorry

end certain_positive_integer_value_l1474_147417


namespace surface_area_of_large_cube_l1474_147410

theorem surface_area_of_large_cube (l w h : ℕ) (cube_side : ℕ) 
  (volume_cuboid : ℕ := l * w * h) 
  (n_cubes := volume_cuboid / (cube_side ^ 3))
  (side_length_large_cube : ℕ := cube_side * (n_cubes^(1/3 : ℕ))) 
  (surface_area_large_cube : ℕ := 6 * (side_length_large_cube ^ 2)) :
  l = 25 → w = 10 → h = 4 → cube_side = 1 → surface_area_large_cube = 600 :=
by
  intros hl hw hh hcs
  subst hl
  subst hw
  subst hh
  subst hcs
  sorry

end surface_area_of_large_cube_l1474_147410


namespace mrs_hilt_baked_pecan_pies_l1474_147474

def total_pies (rows : ℕ) (pies_per_row : ℕ) : ℕ :=
  rows * pies_per_row

def pecan_pies (total_pies : ℕ) (apple_pies : ℕ) : ℕ :=
  total_pies - apple_pies

theorem mrs_hilt_baked_pecan_pies :
  let apple_pies := 14
  let rows := 6
  let pies_per_row := 5
  let total := total_pies rows pies_per_row
  pecan_pies total apple_pies = 16 :=
by
  sorry

end mrs_hilt_baked_pecan_pies_l1474_147474


namespace maximize_profit_at_200_l1474_147471

noncomputable def cost (q : ℝ) : ℝ := 50000 + 200 * q
noncomputable def price (q : ℝ) : ℝ := 24200 - (1/5) * q^2
noncomputable def profit (q : ℝ) : ℝ := (price q) * q - (cost q)

theorem maximize_profit_at_200 : ∃ (q : ℝ), q = 200 ∧ ∀ (x : ℝ), x ≥ 0 → profit q ≥ profit x :=
by
  sorry

end maximize_profit_at_200_l1474_147471


namespace max_remainder_when_divided_by_7_l1474_147450

theorem max_remainder_when_divided_by_7 (y : ℕ) (r : ℕ) (h : r = y % 7) : r ≤ 6 ∧ ∃ k, y = 7 * k + r :=
by
  sorry

end max_remainder_when_divided_by_7_l1474_147450


namespace UnionMathInstitute_students_l1474_147401

theorem UnionMathInstitute_students :
  ∃ n : ℤ, n < 500 ∧ 
    n % 17 = 15 ∧ 
    n % 19 = 18 ∧ 
    n % 16 = 7 ∧ 
    n = 417 :=
by
  -- Problem setup and constraints
  sorry

end UnionMathInstitute_students_l1474_147401


namespace intersection_of_P_and_Q_l1474_147468

def P (x : ℝ) : Prop := 2 ≤ x ∧ x < 4
def Q (x : ℝ) : Prop := 3 * x - 7 ≥ 8 - 2 * x

theorem intersection_of_P_and_Q :
  ∀ x, P x ∧ Q x ↔ 3 ≤ x ∧ x < 4 :=
by
  sorry

end intersection_of_P_and_Q_l1474_147468


namespace number_of_blue_balls_l1474_147421

theorem number_of_blue_balls (T : ℕ) (h1 : (1 / 4) * T = green) (h2 : (1 / 8) * T = blue)
    (h3 : (1 / 12) * T = yellow) (h4 : 26 = white) (h5 : green + blue + yellow + white = T) :
    blue = 6 :=
by
  sorry

end number_of_blue_balls_l1474_147421


namespace find_digits_l1474_147443

-- Define the digits range
def is_digit (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 9

-- Define the five-digit numbers
def num_abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 111 * c
def num_abbbb (a b : ℕ) : ℕ := 10000 * a + 1111 * b

-- Problem statement
theorem find_digits (a b c : ℕ) (h_da : is_digit a) (h_db : is_digit b) (h_dc : is_digit c) :
  (num_abccc a b c) + 1 = (num_abbbb a b) ↔
  (a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0) :=
sorry

end find_digits_l1474_147443


namespace simplify_fraction_l1474_147465

theorem simplify_fraction (x : ℤ) : 
    (2 * x + 3) / 4 + (5 - 4 * x) / 3 = (-10 * x + 29) / 12 := 
by
  sorry

end simplify_fraction_l1474_147465


namespace study_tour_arrangement_l1474_147470

def number_of_arrangements (classes routes : ℕ) (max_selected_route : ℕ) : ℕ :=
  if classes = 4 ∧ routes = 4 ∧ max_selected_route = 2 then 240 else 0

theorem study_tour_arrangement :
  number_of_arrangements 4 4 2 = 240 :=
by sorry

end study_tour_arrangement_l1474_147470


namespace least_subtract_to_divisible_by_14_l1474_147486

theorem least_subtract_to_divisible_by_14 (n : ℕ) (h : n = 7538): 
  (n % 14 = 6) -> ∃ m, (m = 6) ∧ ((n - m) % 14 = 0) :=
by
  sorry

end least_subtract_to_divisible_by_14_l1474_147486


namespace plates_difference_l1474_147481

def num_plates_sunshine := 26^3 * 10^3
def num_plates_prairie := 26^2 * 10^4
def difference := num_plates_sunshine - num_plates_prairie

theorem plates_difference :
  difference = 10816000 := by sorry

end plates_difference_l1474_147481


namespace sum_consecutive_integers_l1474_147408

theorem sum_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_consecutive_integers_l1474_147408


namespace min_value_of_a_l1474_147451

noncomputable def x (t a : ℝ) : ℝ :=
  5 * (t + 1)^2 + a / (t + 1)^5

theorem min_value_of_a (a : ℝ) :
  (∀ t : ℝ, t ≥ 0 → x t a ≥ 24) ↔ a ≥ 2 * Real.sqrt ((24 / 7)^7) :=
sorry

end min_value_of_a_l1474_147451


namespace zach_needs_more_money_l1474_147482

theorem zach_needs_more_money
  (bike_cost : ℕ) (allowance : ℕ) (mowing_payment : ℕ) (babysitting_rate : ℕ) 
  (current_savings : ℕ) (babysitting_hours : ℕ) :
  bike_cost = 100 →
  allowance = 5 →
  mowing_payment = 10 →
  babysitting_rate = 7 →
  current_savings = 65 →
  babysitting_hours = 2 →
  (bike_cost - (current_savings + (allowance + mowing_payment + babysitting_hours * babysitting_rate))) = 6 :=
by
  sorry

end zach_needs_more_money_l1474_147482


namespace ratio_problem_l1474_147439

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l1474_147439


namespace sum_of_reciprocals_of_roots_l1474_147413

open Real

-- Define the polynomial and its properties using Vieta's formulas
theorem sum_of_reciprocals_of_roots :
  ∀ p q : ℝ, 
  (p + q = 16) ∧ (p * q = 9) → 
  (1 / p + 1 / q = 16 / 9) :=
by
  intros p q h
  let ⟨h1, h2⟩ := h
  sorry

end sum_of_reciprocals_of_roots_l1474_147413


namespace rectangular_prism_cut_l1474_147433

theorem rectangular_prism_cut
  (x y : ℕ)
  (original_volume : ℕ := 15 * 5 * 4) 
  (remaining_volume : ℕ := 120) 
  (cut_out_volume_eq : original_volume - remaining_volume = 5 * x * y) 
  (x_condition : 1 < x) 
  (x_condition_2 : x < 4) 
  (y_condition : 1 < y) 
  (y_condition_2 : y < 15) : 
  x + y = 15 := 
sorry

end rectangular_prism_cut_l1474_147433


namespace min_value_of_D_l1474_147491

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a) ^ 2 + (Real.exp x - 2 * Real.sqrt a) ^ 2) + a + 2

theorem min_value_of_D (e : ℝ) (h_e : e = 2.71828) :
  ∀ a : ℝ, ∃ x : ℝ, D x a = Real.sqrt 2 + 1 :=
sorry

end min_value_of_D_l1474_147491


namespace exists_xy_binom_eq_l1474_147435

theorem exists_xy_binom_eq (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x + y).choose 2 = a * x + b * y :=
by
  sorry

end exists_xy_binom_eq_l1474_147435


namespace largest_solution_achieves_largest_solution_l1474_147434

theorem largest_solution (x : ℝ) (hx : ⌊x⌋ = 5 + 100 * (x - ⌊x⌋)) : x ≤ 104.99 :=
by
  -- Placeholder for the proof
  sorry

theorem achieves_largest_solution : ∃ (x : ℝ), ⌊x⌋ = 5 + 100 * (x - ⌊x⌋) ∧ x = 104.99 :=
by
  -- Placeholder for the proof
  sorry

end largest_solution_achieves_largest_solution_l1474_147434


namespace unique_triangle_constructions_l1474_147419

structure Triangle :=
(a b c : ℝ) (A B C : ℝ)

-- Definitions for the conditions
def SSS (t : Triangle) : Prop := 
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def SAS (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180

def ASA (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.c > 0 ∧ t.A + t.B < 180

def SSA (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180 

-- The formally stated proof goal
theorem unique_triangle_constructions (t : Triangle) :
  (SSS t ∨ SAS t ∨ ASA t) ∧ ¬(SSA t) :=
by
  sorry

end unique_triangle_constructions_l1474_147419


namespace determine_N_l1474_147499

theorem determine_N (N : ℕ) : (Nat.choose N 5 = 3003) ↔ (N = 15) :=
by
  sorry

end determine_N_l1474_147499


namespace wrongly_entered_mark_l1474_147429

theorem wrongly_entered_mark (x : ℕ) 
    (h1 : x - 33 = 52) : x = 85 :=
by
  sorry

end wrongly_entered_mark_l1474_147429


namespace complement_of_P_with_respect_to_U_l1474_147411

universe u

def U : Set ℤ := {-1, 0, 1, 2}

def P : Set ℤ := {x | x * x < 2}

theorem complement_of_P_with_respect_to_U : U \ P = {2} :=
by
  sorry

end complement_of_P_with_respect_to_U_l1474_147411


namespace sufficient_but_not_necessary_condition_l1474_147424

theorem sufficient_but_not_necessary_condition (A B : Set ℝ) :
  (A = {x : ℝ | 1 < x ∧ x < 3}) →
  (B = {x : ℝ | x > -1}) →
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1474_147424


namespace find_natural_numbers_l1474_147498

theorem find_natural_numbers (n : ℕ) (h : n > 1) : 
  ((n - 1) ∣ (n^3 - 3)) ↔ (n = 2 ∨ n = 3) := 
by 
  sorry

end find_natural_numbers_l1474_147498


namespace abs_ineq_solution_l1474_147462

theorem abs_ineq_solution (x : ℝ) : abs (x - 2) + abs (x - 3) < 9 ↔ -2 < x ∧ x < 7 :=
sorry

end abs_ineq_solution_l1474_147462


namespace part_a_part_b_l1474_147477

-- Define what it means for a coloring to be valid.
def valid_coloring (n : ℕ) (colors : Fin n → Fin 3) : Prop :=
  ∀ (i : Fin n),
  ∃ j k : Fin n, 
  ((i + 1) % n = j ∧ (i + 2) % n = k ∧ colors i ≠ colors j ∧ colors i ≠ colors k ∧ colors j ≠ colors k)

-- Part (a)
theorem part_a (n : ℕ) (hn : 3 ∣ n) : ∃ (colors : Fin n → Fin 3), valid_coloring n colors :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) : (∃ (colors : Fin n → Fin 3), valid_coloring n colors) → 3 ∣ n :=
by sorry

end part_a_part_b_l1474_147477


namespace remaining_grandchild_share_l1474_147426

theorem remaining_grandchild_share 
  (total : ℕ) 
  (half_share : ℕ) 
  (remaining : ℕ) 
  (n : ℕ) 
  (total_eq : total = 124600)
  (half_share_eq : half_share = total / 2)
  (remaining_eq : remaining = total - half_share)
  (n_eq : n = 10) 
  : remaining / n = 6230 := 
by sorry

end remaining_grandchild_share_l1474_147426


namespace find_resistance_x_l1474_147412

theorem find_resistance_x (y r x : ℝ) (h₁ : y = 5) (h₂ : r = 1.875) (h₃ : 1/r = 1/x + 1/y) : x = 3 :=
by
  sorry

end find_resistance_x_l1474_147412


namespace second_field_area_percent_greater_l1474_147449

theorem second_field_area_percent_greater (r1 r2 : ℝ) (h : r1 / r2 = 2 / 5) : 
  (π * (r2^2) - π * (r1^2)) / (π * (r1^2)) * 100 = 525 := 
by
  sorry

end second_field_area_percent_greater_l1474_147449
