import Mathlib

namespace wilma_garden_rows_l137_13791

theorem wilma_garden_rows :
  ∃ (rows : ℕ),
    (∃ (yellow green red total : ℕ),
      yellow = 12 ∧
      green = 2 * yellow ∧
      red = 42 ∧
      total = yellow + green + red ∧
      total / 13 = rows ∧
      rows = 6) :=
sorry

end wilma_garden_rows_l137_13791


namespace susan_remaining_money_l137_13781

theorem susan_remaining_money :
  let initial_amount := 90
  let food_spent := 20
  let game_spent := 3 * food_spent
  let total_spent := food_spent + game_spent
  initial_amount - total_spent = 10 :=
by 
  sorry

end susan_remaining_money_l137_13781


namespace smallest_n_l137_13730

theorem smallest_n (n : ℕ) (h : 0 < n) : 
  (1 / (n : ℝ)) - (1 / (n + 1 : ℝ)) < 1 / 15 → n = 4 := sorry

end smallest_n_l137_13730


namespace segment_length_is_15_l137_13756

theorem segment_length_is_15 : 
  ∀ (x : ℝ), 
  ∀ (y1 y2 : ℝ), 
  x = 3 → 
  y1 = 5 → 
  y2 = 20 → 
  abs (y2 - y1) = 15 := by 
sorry

end segment_length_is_15_l137_13756


namespace breadth_of_rectangular_plot_is_18_l137_13728

/-- Problem statement:
The length of a rectangular plot is thrice its breadth. 
If the area of the rectangular plot is 972 sq m, 
this theorem proves that the breadth of the rectangular plot is 18 meters.
-/
theorem breadth_of_rectangular_plot_is_18 (b l : ℝ) (h_length : l = 3 * b) (h_area : l * b = 972) : b = 18 :=
by
  sorry

end breadth_of_rectangular_plot_is_18_l137_13728


namespace last_three_digits_of_3_pow_5000_l137_13722

theorem last_three_digits_of_3_pow_5000 : (3 ^ 5000) % 1000 = 1 := 
by
  -- skip the proof
  sorry

end last_three_digits_of_3_pow_5000_l137_13722


namespace prime_square_minus_one_divisible_by_24_l137_13771

theorem prime_square_minus_one_divisible_by_24 (n : ℕ) (h_prime : Prime n) (h_n_neq_2 : n ≠ 2) (h_n_neq_3 : n ≠ 3) : 24 ∣ (n^2 - 1) :=
sorry

end prime_square_minus_one_divisible_by_24_l137_13771


namespace sequence_sum_l137_13721

theorem sequence_sum (a b : ℤ) (h1 : ∃ d, d = 5 ∧ (∀ n : ℕ, (3 + n * d) = a ∨ (3 + (n-1) * d) = b ∨ (3 + (n-2) * d) = 33)) : 
  a + b = 51 :=
by
  sorry

end sequence_sum_l137_13721


namespace domain_of_f_comp_l137_13777

theorem domain_of_f_comp (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ x^2 - 2 ∧ x^2 - 2 ≤ -1) →
  (∀ x, - (4 : ℝ) / 3 ≤ x ∧ x ≤ -1 → -2 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ -1) :=
by
  sorry

end domain_of_f_comp_l137_13777


namespace total_pastries_l137_13748

variable (P x : ℕ)

theorem total_pastries (h1 : P = 28 * (10 + x)) (h2 : P = 49 * (4 + x)) : P = 392 := 
by 
  sorry

end total_pastries_l137_13748


namespace distance_between_centers_of_externally_tangent_circles_l137_13708

noncomputable def external_tangent_distance (R r : ℝ) (hR : R = 2) (hr : r = 3) (tangent : R > 0 ∧ r > 0) : ℝ :=
  R + r

theorem distance_between_centers_of_externally_tangent_circles :
  external_tangent_distance 2 3 (by rfl) (by rfl) (by norm_num) = 5 :=
sorry

end distance_between_centers_of_externally_tangent_circles_l137_13708


namespace solve_equation_l137_13788

theorem solve_equation :
  ∃ x : ℝ, x = (Real.sqrt (x - 1/x)) + (Real.sqrt (1 - 1/x)) ∧ x = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end solve_equation_l137_13788


namespace probability_of_selecting_one_male_and_one_female_l137_13755

noncomputable def probability_one_male_one_female : ℚ :=
  let total_ways := (Nat.choose 6 2) -- Total number of ways to select 2 out of 6
  let ways_one_male_one_female := (Nat.choose 3 1) * (Nat.choose 3 1) -- Ways to select 1 male and 1 female
  ways_one_male_one_female / total_ways

theorem probability_of_selecting_one_male_and_one_female :
  probability_one_male_one_female = 3 / 5 := by
  sorry

end probability_of_selecting_one_male_and_one_female_l137_13755


namespace probability_of_blue_candy_l137_13704

theorem probability_of_blue_candy (green blue red : ℕ) (h1 : green = 5) (h2 : blue = 3) (h3 : red = 4) :
  (blue : ℚ) / (green + blue + red : ℚ) = 1 / 4 :=
by
  rw [h1, h2, h3]
  norm_num


end probability_of_blue_candy_l137_13704


namespace count_big_boxes_l137_13767

theorem count_big_boxes (B : ℕ) (h : 7 * B + 4 * 9 = 71) : B = 5 :=
sorry

end count_big_boxes_l137_13767


namespace expression_equals_sqrt2_l137_13779

theorem expression_equals_sqrt2 :
  (1 + Real.pi)^0 + 2 - abs (-3) + 2 * Real.sin (Real.pi / 4) = Real.sqrt 2 := by
  sorry

end expression_equals_sqrt2_l137_13779


namespace average_age_of_team_l137_13726

/--
The captain of a cricket team of 11 members is 26 years old and the wicket keeper is 
3 years older. If the ages of these two are excluded, the average age of the remaining 
players is one year less than the average age of the whole team. Prove that the average 
age of the whole team is 32 years.
-/
theorem average_age_of_team 
  (captain_age : Nat) (wicket_keeper_age : Nat) (remaining_9_average_age : Nat)
  (team_size : Nat) (total_team_age : Nat) (remaining_9_total_age : Nat)
  (A : Nat) :
  captain_age = 26 →
  wicket_keeper_age = captain_age + 3 →
  team_size = 11 →
  total_team_age = team_size * A →
  total_team_age = remaining_9_total_age + captain_age + wicket_keeper_age →
  remaining_9_total_age = 9 * (A - 1) →
  A = 32 :=
by
  sorry

end average_age_of_team_l137_13726


namespace value_of_m_l137_13762

theorem value_of_m (m : ℚ) : 
  (m = - -(-(1/3) : ℚ) → m = -1/3) :=
by
  sorry

end value_of_m_l137_13762


namespace bowling_ball_weight_l137_13747

theorem bowling_ball_weight :
  (∀ b c : ℝ, 9 * b = 2 * c → c = 35 → b = 70 / 9) :=
by
  intros b c h1 h2
  sorry

end bowling_ball_weight_l137_13747


namespace minimum_elements_union_l137_13776

open Set

def A : Finset ℕ := sorry
def B : Finset ℕ := sorry

variable (size_A : A.card = 25)
variable (size_B : B.card = 18)
variable (at_least_10_not_in_A : (B \ A).card ≥ 10)

theorem minimum_elements_union : (A ∪ B).card = 35 :=
by
  sorry

end minimum_elements_union_l137_13776


namespace extinction_prob_one_l137_13701

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l137_13701


namespace part1_part2_l137_13749

variable {a b : ℝ}

noncomputable def in_interval (x: ℝ) : Prop :=
  -1/2 < x ∧ x < 1/2

theorem part1 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1/3 * a + 1/6 * b) < 1/4 := 
by sorry

theorem part2 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1 - 4 * a * b) > 2 * abs (a - b) := 
by sorry

end part1_part2_l137_13749


namespace impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l137_13787

theorem impossible_to_get_60_pieces :
  ¬ ∃ (n m : ℕ), 1 + 7 * n + 11 * m = 60 :=
sorry

theorem possible_to_get_more_than_60_pieces :
  ∀ k > 60, ∃ (n m : ℕ), 1 + 7 * n + 11 * m = k :=
sorry

end impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l137_13787


namespace alloy_mixing_l137_13792

theorem alloy_mixing (x : ℕ) :
  (2 / 5) * 60 + (1 / 5) * x = 44 → x = 100 :=
by
  intros h1
  sorry

end alloy_mixing_l137_13792


namespace chickens_egg_production_l137_13737

/--
Roberto buys 4 chickens for $20 each. The chickens cost $1 in total per week to feed.
Roberto used to buy 1 dozen eggs (12 eggs) a week, spending $2 per dozen.
After 81 weeks, the total cost of raising chickens will be cheaper than buying the eggs.
Prove that each chicken produces 3 eggs per week.
-/
theorem chickens_egg_production:
  let chicken_cost := 20
  let num_chickens := 4
  let weekly_feed_cost := 1
  let weekly_eggs_cost := 2
  let dozen_eggs := 12
  let weeks := 81

  -- Cost calculations
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weekly_feed_cost * weeks
  let total_raising_cost := total_chicken_cost + total_feed_cost
  let total_buying_cost := weekly_eggs_cost * weeks

  -- Ensure cost condition
  (total_raising_cost <= total_buying_cost) →
  
  -- Egg production calculation
  (dozen_eggs / num_chickens) = 3 :=
by
  intros
  sorry

end chickens_egg_production_l137_13737


namespace evaluate_f_neg_a_l137_13774

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem evaluate_f_neg_a (a : ℝ) (h : f a = 1 / 3) : f (-a) = 5 / 3 :=
by sorry

end evaluate_f_neg_a_l137_13774


namespace regular_polygon_sides_l137_13790

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l137_13790


namespace no_solution_inequality_l137_13724

theorem no_solution_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| < 4 * x - 1 ∧ x < a) ↔ a ≤ (2/3) := by sorry

end no_solution_inequality_l137_13724


namespace stationery_cost_l137_13740

theorem stationery_cost (cost_per_pencil cost_per_pen : ℕ)
    (boxes : ℕ)
    (pencils_per_box pens_offset : ℕ)
    (total_cost : ℕ) :
    cost_per_pencil = 4 →
    boxes = 15 →
    pencils_per_box = 80 →
    pens_offset = 300 →
    cost_per_pen = 5 →
    total_cost = (boxes * pencils_per_box * cost_per_pencil) +
                 ((2 * (boxes * pencils_per_box + pens_offset)) * cost_per_pen) →
    total_cost = 18300 :=
by
  intros
  sorry

end stationery_cost_l137_13740


namespace no_hikers_in_morning_l137_13784

-- Given Conditions
def morning_rowers : ℕ := 13
def afternoon_rowers : ℕ := 21
def total_rowers : ℕ := 34

-- Statement to be proven
theorem no_hikers_in_morning : (total_rowers - afternoon_rowers = morning_rowers) →
                              (total_rowers - afternoon_rowers = morning_rowers) →
                              0 = 34 - 21 - morning_rowers :=
by
  intros h1 h2
  sorry

end no_hikers_in_morning_l137_13784


namespace no_int_solutions_for_equation_l137_13780

theorem no_int_solutions_for_equation : 
  ∀ x y : ℤ, x ^ 2022 + y^2 = 2 * y + 2 → false := 
by
  -- By the given steps in the solution, we can conclude that no integer solutions exist
  sorry

end no_int_solutions_for_equation_l137_13780


namespace range_of_a_l137_13714

-- Definitions for the conditions
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0

-- Main theorem
theorem range_of_a (a : ℝ) (h : a < 0) : (¬ (∃ x, prop_p a x)) → (¬ (∃ x, ¬ prop_q x)) :=
sorry

end range_of_a_l137_13714


namespace find_radius_l137_13745

theorem find_radius :
  ∃ (r : ℝ), 
  (∀ (x : ℝ), y = x^2 + r) ∧ 
  (∀ (x : ℝ), y = x) ∧ 
  (∀ (x : ℝ), x^2 + r = x) ∧ 
  (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 0) → 
  r = 1 / 4 :=
by
  sorry

end find_radius_l137_13745


namespace find_m_l137_13786

theorem find_m (m l : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, m)) (h_b : b = (l, -2))
  (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ a = k • (a + 2 • b)) :
  m = -4 :=
by
  sorry

end find_m_l137_13786


namespace michael_laps_to_pass_donovan_l137_13710

theorem michael_laps_to_pass_donovan (track_length : ℕ) (donovan_lap_time : ℕ) (michael_lap_time : ℕ) 
  (h1 : track_length = 400) (h2 : donovan_lap_time = 48) (h3 : michael_lap_time = 40) : 
  michael_lap_time * 6 = donovan_lap_time * (michael_lap_time * 6 / track_length * michael_lap_time) :=
by
  sorry

end michael_laps_to_pass_donovan_l137_13710


namespace goose_eggs_laid_l137_13751

theorem goose_eggs_laid (E : ℕ) 
    (H1 : ∃ h, h = (2 / 5) * E)
    (H2 : ∃ m, m = (11 / 15) * h)
    (H3 : ∃ s, s = (1 / 4) * m)
    (H4 : ∃ y, y = (2 / 7) * s)
    (H5 : y = 150) : 
    E = 7160 := 
sorry

end goose_eggs_laid_l137_13751


namespace arithmetic_sequence_ratios_l137_13707

theorem arithmetic_sequence_ratios
  (a : ℕ → ℝ) (b : ℕ → ℝ) (A : ℕ → ℝ) (B : ℕ → ℝ)
  (d1 d2 a1 b1 : ℝ)
  (hA_sum : ∀ n : ℕ, A n = n * a1 + (n * (n - 1)) * d1 / 2)
  (hB_sum : ∀ n : ℕ, B n = n * b1 + (n * (n - 1)) * d2 / 2)
  (h_ratio : ∀ n : ℕ, B n ≠ 0 → A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b n ≠ 0 → a n / b n = (4 * n - 3) / (6 * n - 2) := sorry

end arithmetic_sequence_ratios_l137_13707


namespace range_of_x_l137_13734

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) : x > 1/3 :=
by
  sorry

end range_of_x_l137_13734


namespace train_length_l137_13715

theorem train_length (v : ℝ) (t : ℝ) (l_b : ℝ) (v_r : v = 52) (t_r : t = 34.61538461538461) (l_b_r : l_b = 140) : 
  ∃ l_t : ℝ, l_t = 360 :=
by
  have speed_ms := v * (1000 / 3600)
  have total_distance := speed_ms * t
  have length_train := total_distance - l_b
  use length_train
  sorry

end train_length_l137_13715


namespace value_of_m_l137_13760

theorem value_of_m : 5^2 + 7 = 4^3 + m → m = -32 :=
by
  intro h
  sorry

end value_of_m_l137_13760


namespace problem_statement_l137_13789

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def has_minimum_value_at (f : ℝ → ℝ) (a : ℝ) := ∀ x : ℝ, f a ≤ f x
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_statement : is_even_function f4 ∧ has_minimum_value_at f4 0 :=
by
  sorry

end problem_statement_l137_13789


namespace intersection_A_complement_B_l137_13727

def A := { x : ℝ | x ≥ -1 }
def B := { x : ℝ | x > 2 }
def complement_B := { x : ℝ | x ≤ 2 }

theorem intersection_A_complement_B :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_A_complement_B_l137_13727


namespace queenie_daily_earnings_l137_13712

/-- Define the overtime earnings per hour. -/
def overtime_pay_per_hour : ℤ := 5

/-- Define the total amount received. -/
def total_received : ℤ := 770

/-- Define the number of days worked. -/
def days_worked : ℤ := 5

/-- Define the number of overtime hours. -/
def overtime_hours : ℤ := 4

/-- State the theorem to find out Queenie's daily earnings. -/
theorem queenie_daily_earnings :
  ∃ D : ℤ, days_worked * D + overtime_hours * overtime_pay_per_hour = total_received ∧ D = 150 :=
by
  use 150
  sorry

end queenie_daily_earnings_l137_13712


namespace sum_of_squares_base_case_l137_13706

theorem sum_of_squares_base_case : 1^2 + 2^2 = (1 * 3 * 5) / 3 := by sorry

end sum_of_squares_base_case_l137_13706


namespace ned_games_l137_13729

theorem ned_games (F: ℕ) (bought_from_friend garage_sale non_working good total_games: ℕ) 
  (h₁: bought_from_friend = F)
  (h₂: garage_sale = 27)
  (h₃: non_working = 74)
  (h₄: good = 3)
  (h₅: total_games = non_working + good)
  (h₆: total_games = bought_from_friend + garage_sale) :
  F = 50 :=
by
  sorry

end ned_games_l137_13729


namespace ratio_of_drinking_speeds_l137_13752

def drinking_ratio(mala_portion usha_portion : ℚ) (same_time: Bool) (usha_fraction: ℚ) : ℚ :=
if same_time then mala_portion / usha_portion else 0

theorem ratio_of_drinking_speeds
  (mala_portion : ℚ)
  (usha_portion : ℚ)
  (same_time : Bool)
  (usha_fraction : ℚ)
  (usha_drank : usha_fraction = 2 / 10)
  (mala_drank : mala_portion = 1 - usha_fraction)
  (equal_time : same_time = tt)
  (ratio : drinking_ratio mala_portion usha_portion same_time usha_fraction = 4) :
  mala_portion / usha_portion = 4 :=
by
  sorry

end ratio_of_drinking_speeds_l137_13752


namespace what_percent_of_y_l137_13766

-- Given condition
axiom y_pos : ℝ → Prop

noncomputable def math_problem (y : ℝ) (h : y_pos y) : Prop :=
  (8 * y / 20 + 3 * y / 10 = 0.7 * y)

-- The theorem to be proved
theorem what_percent_of_y (y : ℝ) (h : y > 0) : 8 * y / 20 + 3 * y / 10 = 0.7 * y :=
by
  sorry

end what_percent_of_y_l137_13766


namespace october_profit_condition_l137_13757

noncomputable def calculate_profit (price_reduction : ℝ) : ℝ :=
  (50 - price_reduction) * (500 + 20 * price_reduction)

theorem october_profit_condition (x : ℝ) (h : calculate_profit x = 28000) : x = 10 ∨ x = 15 := 
by
  sorry

end october_profit_condition_l137_13757


namespace overlapping_area_is_correct_l137_13716

-- Defining the coordinates of the grid points
def topLeft : (ℝ × ℝ) := (0, 2)
def topMiddle : (ℝ × ℝ) := (1.5, 2)
def topRight : (ℝ × ℝ) := (3, 2)
def middleLeft : (ℝ × ℝ) := (0, 1)
def center : (ℝ × ℝ) := (1.5, 1)
def middleRight : (ℝ × ℝ) := (3, 1)
def bottomLeft : (ℝ × ℝ) := (0, 0)
def bottomMiddle : (ℝ × ℝ) := (1.5, 0)
def bottomRight : (ℝ × ℝ) := (3, 0)

-- Defining the vertices of the triangles
def triangle1_points : List (ℝ × ℝ) := [topLeft, middleRight, bottomMiddle]
def triangle2_points : List (ℝ × ℝ) := [bottomLeft, topMiddle, middleRight]

-- Function to calculate the area of a polygon given the vertices -- placeholder here
noncomputable def area_of_overlapped_region (tr1 tr2 : List (ℝ × ℝ)) : ℝ := 
  -- Placeholder for the actual computation of the overlapped area
  1.2

-- Statement to prove
theorem overlapping_area_is_correct : 
  area_of_overlapped_region triangle1_points triangle2_points = 1.2 := sorry

end overlapping_area_is_correct_l137_13716


namespace billboards_color_schemes_is_55_l137_13772

def adjacent_color_schemes (n : ℕ) : ℕ :=
  if h : n = 8 then 55 else 0

theorem billboards_color_schemes_is_55 :
  adjacent_color_schemes 8 = 55 :=
sorry

end billboards_color_schemes_is_55_l137_13772


namespace envelope_width_l137_13731

theorem envelope_width (L W A : ℝ) (hL : L = 4) (hA : A = 16) (hArea : A = L * W) : W = 4 := 
by
  -- We state the problem
  sorry

end envelope_width_l137_13731


namespace simplify_and_evaluate_expression_l137_13705

theorem simplify_and_evaluate_expression (x y : ℚ) (h_x : x = -2) (h_y : y = 1/2) :
  (x + 2 * y)^2 - (x + y) * (x - y) = -11/4 := by
  sorry

end simplify_and_evaluate_expression_l137_13705


namespace greatest_mass_l137_13769

theorem greatest_mass (V : ℝ) (h : ℝ) (l : ℝ) 
    (ρ_Hg ρ_H2O ρ_Oil : ℝ) 
    (V1 V2 V3 : ℝ) 
    (m_Hg m_H2O m_Oil : ℝ)
    (ρ_Hg_val : ρ_Hg = 13.59) 
    (ρ_H2O_val : ρ_H2O = 1) 
    (ρ_Oil_val : ρ_Oil = 0.915) 
    (height_layers_equal : h = l) :
    ∀ V1 V2 V3 m_Hg m_H2O m_Oil, 
    V1 + V2 + V3 = 27 * (l^3) → 
    V2 = 7 * V1 → 
    V3 = 19 * V1 → 
    m_Hg = ρ_Hg * V1 → 
    m_H2O = ρ_H2O * V2 → 
    m_Oil = ρ_Oil * V3 → 
    m_Oil > m_Hg ∧ m_Oil > m_H2O := 
by 
    intros
    sorry

end greatest_mass_l137_13769


namespace solve_proportion_l137_13765

noncomputable def x : ℝ := 0.6

theorem solve_proportion (x : ℝ) (h : 0.75 / x = 10 / 8) : x = 0.6 :=
by
  sorry

end solve_proportion_l137_13765


namespace part1_part2_l137_13761

noncomputable def f (a x : ℝ) : ℝ := a * x + x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x, x ≥ Real.exp 1 → (a + 1 + Real.log x) ≥ 0) →
  a ≥ -2 :=
by
  sorry

theorem part2 (k : ℤ) :
  (∀ x, 1 < x → (k : ℝ) * (x - 1) < f 1 x) →
  k ≤ 3 :=
by
  sorry

end part1_part2_l137_13761


namespace claire_initial_balloons_l137_13725

theorem claire_initial_balloons (B : ℕ) (h : B - 12 - 9 + 11 = 39) : B = 49 :=
by sorry

end claire_initial_balloons_l137_13725


namespace initial_bottle_caps_correct_l137_13799

-- Defining the variables based on the conditions
def bottle_caps_found : ℕ := 7
def total_bottle_caps_now : ℕ := 32
def initial_bottle_caps : ℕ := 25

-- Statement of the theorem
theorem initial_bottle_caps_correct:
  total_bottle_caps_now - bottle_caps_found = initial_bottle_caps :=
sorry

end initial_bottle_caps_correct_l137_13799


namespace maximum_value_of_expression_l137_13770

-- Define the given condition
def condition (a b c : ℝ) : Prop := a + 3 * b + c = 5

-- Define the objective function
def objective (a b c : ℝ) : ℝ := a * b + a * c + b * c

-- Main theorem statement
theorem maximum_value_of_expression (a b c : ℝ) (h : condition a b c) : 
  ∃ (a b c : ℝ), condition a b c ∧ objective a b c = 25 / 3 :=
sorry

end maximum_value_of_expression_l137_13770


namespace joyce_initial_eggs_l137_13778

theorem joyce_initial_eggs :
  ∃ E : ℕ, (E + 6 = 14) ∧ E = 8 :=
sorry

end joyce_initial_eggs_l137_13778


namespace inequality_proof_l137_13732

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) :=
by
  sorry

end inequality_proof_l137_13732


namespace anne_find_bottle_caps_l137_13739

theorem anne_find_bottle_caps 
  (n_i n_f : ℕ) (h_initial : n_i = 10) (h_final : n_f = 15) : n_f - n_i = 5 :=
by
  sorry

end anne_find_bottle_caps_l137_13739


namespace difference_of_interchanged_digits_l137_13753

theorem difference_of_interchanged_digits (X Y : ℕ) (h1 : X - Y = 3) :
  (10 * X + Y) - (10 * Y + X) = 27 := by
  sorry

end difference_of_interchanged_digits_l137_13753


namespace depth_of_first_hole_l137_13744

theorem depth_of_first_hole (n1 t1 n2 t2 : ℕ) (D : ℝ) (r : ℝ) 
  (h1 : n1 = 45) (h2 : t1 = 8) (h3 : n2 = 90) (h4 : t2 = 6) 
  (h5 : r = 1 / 12) (h6 : D = n1 * t1 * r) (h7 : n2 * t2 * r = 45) : 
  D = 30 := 
by 
  sorry

end depth_of_first_hole_l137_13744


namespace inequality_proof_l137_13795

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a / (a + 2 * b)^(1/3) + b / (b + 2 * c)^(1/3) + c / (c + 2 * a)^(1/3)) ≥ 1 := 
by
  sorry

end inequality_proof_l137_13795


namespace mixture_solution_l137_13711

theorem mixture_solution (x y : ℝ) :
  (0.30 * x + 0.40 * y = 32) →
  (x + y = 100) →
  (x = 80) :=
by
  intros h₁ h₂
  sorry

end mixture_solution_l137_13711


namespace f_continuous_on_interval_f_not_bounded_variation_l137_13773

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else x * Real.sin (1 / x)

theorem f_continuous_on_interval : ContinuousOn f (Set.Icc 0 1) :=
sorry

theorem f_not_bounded_variation : ¬ BoundedVariationOn f (Set.Icc 0 1) :=
sorry

end f_continuous_on_interval_f_not_bounded_variation_l137_13773


namespace find_y_when_x_is_minus_2_l137_13743

theorem find_y_when_x_is_minus_2 :
  ∀ (x y t : ℝ), (x = 3 - 2 * t) → (y = 5 * t + 6) → (x = -2) → y = 37 / 2 :=
by
  intros x y t h1 h2 h3
  sorry

end find_y_when_x_is_minus_2_l137_13743


namespace necessary_not_sufficient_condition_l137_13782

theorem necessary_not_sufficient_condition (x : ℝ) : (x < 2) → (x^2 - x - 2 < 0) :=
by {
  sorry
}

end necessary_not_sufficient_condition_l137_13782


namespace coefficients_square_sum_l137_13763

theorem coefficients_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1000 * x ^ 3 + 27 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end coefficients_square_sum_l137_13763


namespace frequency_of_group_5_l137_13754

theorem frequency_of_group_5 (total_students freq1 freq2 freq3 freq4 : ℕ)
  (h_total: total_students = 50) 
  (h_freq1: freq1 = 7) 
  (h_freq2: freq2 = 12) 
  (h_freq3: freq3 = 13) 
  (h_freq4: freq4 = 8) :
  (50 - (7 + 12 + 13 + 8)) / 50 = 0.2 :=
by
  sorry

end frequency_of_group_5_l137_13754


namespace problem_statement_l137_13775

theorem problem_statement (pi : ℝ) (h : pi = 4 * Real.sin (52 * Real.pi / 180)) :
  (2 * pi * Real.sqrt (16 - pi ^ 2) - 8 * Real.sin (44 * Real.pi / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * (Real.sin (22 * Real.pi / 180)) ^ 2) = 8 * Real.sqrt 3 := 
  sorry

end problem_statement_l137_13775


namespace line_perpendicular_intersection_l137_13797

noncomputable def line_equation (x y : ℝ) := 3 * x + y + 2 = 0

def is_perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

theorem line_perpendicular_intersection (x y : ℝ) :
  (x - y + 2 = 0) →
  (2 * x + y + 1 = 0) →
  is_perpendicular (1 / 3) (-3) →
  line_equation x y := 
sorry

end line_perpendicular_intersection_l137_13797


namespace abs_diff_61st_term_l137_13700

-- Define sequences C and D
def seqC (n : ℕ) : ℤ := 20 + 15 * (n - 1)
def seqD (n : ℕ) : ℤ := 20 - 15 * (n - 1)

-- Prove the absolute value of the difference between the 61st terms is 1800
theorem abs_diff_61st_term : (abs (seqC 61 - seqD 61) = 1800) :=
by
  sorry

end abs_diff_61st_term_l137_13700


namespace part1_a_range_part2_x_range_l137_13758
open Real

-- Definitions based on given conditions
def quad_func (a b x : ℝ) : ℝ :=
  a * x^2 + b * x + 2

def y_at_x1 (a b : ℝ) : Prop :=
  quad_func a b 1 = 1

def pos_on_interval (a b l r : ℝ) (x : ℝ) : Prop :=
  l < x ∧ x < r → 0 < quad_func a b x

-- Part 1 proof statement in Lean 4
theorem part1_a_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ x : ℝ, pos_on_interval a b 2 5 x) :
  a > 3 - 2 * sqrt 2 :=
sorry

-- Part 2 proof statement in Lean 4
theorem part2_x_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ a' : ℝ, -2 ≤ a' ∧ a' ≤ -1 → 0 < quad_func a' b x) :
  (1 - sqrt 17) / 4 < x ∧ x < (1 + sqrt 17) / 4 :=
sorry

end part1_a_range_part2_x_range_l137_13758


namespace count_solutions_inequalities_l137_13735

theorem count_solutions_inequalities :
  {x : ℤ | -5 * x ≥ 2 * x + 10} ∩ {x : ℤ | -3 * x ≤ 15} ∩ {x : ℤ | -6 * x ≥ 3 * x + 21} = {x : ℤ | x = -5 ∨ x = -4 ∨ x = -3} :=
by 
  sorry

end count_solutions_inequalities_l137_13735


namespace matt_climbing_speed_l137_13720

theorem matt_climbing_speed :
  ∃ (x : ℝ), (12 * 7 = 7 * x + 42) ∧ x = 6 :=
by {
  sorry
}

end matt_climbing_speed_l137_13720


namespace simplify_expression_l137_13702

theorem simplify_expression (y : ℝ) : (y - 2)^2 + 2 * (y - 2) * (5 + y) + (5 + y)^2 = (2*y + 3)^2 := 
by sorry

end simplify_expression_l137_13702


namespace episodes_per_season_l137_13785

theorem episodes_per_season (S : ℕ) (E : ℕ) (H1 : S = 12) (H2 : 2/3 * E = 160) : E / S = 20 :=
by
  sorry

end episodes_per_season_l137_13785


namespace one_eighth_of_two_pow_36_eq_two_pow_y_l137_13719

theorem one_eighth_of_two_pow_36_eq_two_pow_y (y : ℕ) : (2^36 / 8 = 2^y) → (y = 33) :=
by
  sorry

end one_eighth_of_two_pow_36_eq_two_pow_y_l137_13719


namespace arithmetic_sequence_sum_l137_13718

variable (a_n : ℕ → ℕ)

theorem arithmetic_sequence_sum (h1: a_n 1 + a_n 2 = 5) (h2 : a_n 3 + a_n 4 = 7) (arith : ∀ n, a_n (n + 1) - a_n n = a_n 2 - a_n 1) :
  a_n 5 + a_n 6 = 9 := 
sorry

end arithmetic_sequence_sum_l137_13718


namespace hyperbola_asymptote_l137_13709

theorem hyperbola_asymptote (a : ℝ) (h_cond : 0 < a)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2 - y^2 / 9 = 1) → (y = (3 / 5) * x))
  : a = 5 :=
sorry

end hyperbola_asymptote_l137_13709


namespace ratio_of_sides_l137_13783

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l137_13783


namespace rectangle_ratio_constant_l137_13738

theorem rectangle_ratio_constant (length width : ℝ) (d k : ℝ)
  (h1 : length/width = 5/2)
  (h2 : 2 * (length + width) = 28)
  (h3 : d^2 = length^2 + width^2)
  (h4 : (length * width) = k * d^2) :
  k = (10/29) := by
  sorry

end rectangle_ratio_constant_l137_13738


namespace range_of_m_l137_13759

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l137_13759


namespace geometric_sequence_first_term_l137_13742

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 162) : a = 2 := by
  sorry

end geometric_sequence_first_term_l137_13742


namespace number_of_terms_is_13_l137_13764

-- Define sum of first three terms
def sum_first_three (a d : ℤ) : ℤ := a + (a + d) + (a + 2 * d)

-- Define sum of last three terms when the number of terms is n
def sum_last_three (a d : ℤ) (n : ℕ) : ℤ := (a + (n - 3) * d) + (a + (n - 2) * d) + (a + (n - 1) * d)

-- Define sum of all terms in the sequence
def sum_all_terms (a d : ℤ) (n : ℕ) : ℤ := n / 2 * (2 * a + (n - 1) * d)

-- Given conditions
def condition_one (a d : ℤ) : Prop := sum_first_three a d = 34
def condition_two (a d : ℤ) (n : ℕ) : Prop := sum_last_three a d n = 146
def condition_three (a d : ℤ) (n : ℕ) : Prop := sum_all_terms a d n = 390

-- Theorem to prove that n = 13
theorem number_of_terms_is_13 (a d : ℤ) (n : ℕ) :
  condition_one a d →
  condition_two a d n →
  condition_three a d n →
  n = 13 :=
by sorry

end number_of_terms_is_13_l137_13764


namespace minimum_choir_size_l137_13796

theorem minimum_choir_size : ∃ (choir_size : ℕ), 
  (choir_size % 9 = 0) ∧ 
  (choir_size % 11 = 0) ∧ 
  (choir_size % 13 = 0) ∧ 
  (choir_size % 10 = 0) ∧ 
  (choir_size = 12870) :=
by
  sorry

end minimum_choir_size_l137_13796


namespace scalene_triangle_angle_difference_l137_13798

def scalene_triangle (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem scalene_triangle_angle_difference (x y : ℝ) :
  (x + y = 100) → scalene_triangle x y 80 → (x - y = 80) :=
by
  intros h1 h2
  sorry

end scalene_triangle_angle_difference_l137_13798


namespace original_ratio_l137_13703

theorem original_ratio (x y : ℕ) (h1 : y = 15) (h2 : x + 10 = y) : x / y = 1 / 3 :=
by
  sorry

end original_ratio_l137_13703


namespace find_n_l137_13768

theorem find_n (n : ℕ) :
  (2^n - 1) % 3 = 0 ∧ (∃ m : ℤ, (2^n - 1) / 3 ∣ 4 * m^2 + 1) →
  ∃ j : ℕ, n = 2^j :=
by
  sorry

end find_n_l137_13768


namespace sales_tax_difference_l137_13793

-- Definitions for the conditions
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.075
def tax_rate2 : ℝ := 0.05

-- Calculations based on the conditions
def tax1 := item_price * tax_rate1
def tax2 := item_price * tax_rate2

-- The proof statement
theorem sales_tax_difference :
  tax1 - tax2 = 1.25 :=
by
  sorry

end sales_tax_difference_l137_13793


namespace num_tables_l137_13794

theorem num_tables (T : ℕ) : 
  (6 * T = (17 / 3) * T) → 
  T = 6 :=
sorry

end num_tables_l137_13794


namespace tom_four_times_cindy_years_ago_l137_13733

variables (t c x : ℕ)

-- Conditions
axiom cond1 : t + 5 = 2 * (c + 5)
axiom cond2 : t - 13 = 3 * (c - 13)

-- Question to prove
theorem tom_four_times_cindy_years_ago :
  t - x = 4 * (c - x) → x = 19 :=
by
  intros h
  -- simply skip the proof for now
  sorry

end tom_four_times_cindy_years_ago_l137_13733


namespace find_length_of_BC_l137_13736

-- Define the geometrical objects and lengths
variable {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variable (AB AC AM BC : ℝ)
variable (is_midpoint : Midpoint M B C)
variable (known_AB : AB = 7)
variable (known_AC : AC = 6)
variable (known_AM : AM = 4)

theorem find_length_of_BC : BC = Real.sqrt 106 := by
  sorry

end find_length_of_BC_l137_13736


namespace triangle_angle_opposite_c_l137_13741

theorem triangle_angle_opposite_c (a b c : ℝ) (x : ℝ) 
  (ha : a = 2) (hb : b = 2) (hc : c = 4) : x = 180 :=
by 
  -- proof steps are not required as per the instruction
  sorry

end triangle_angle_opposite_c_l137_13741


namespace loga_increasing_loga_decreasing_l137_13746

noncomputable def loga (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem loga_increasing (a : ℝ) (h₁ : a > 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a x < loga a y := by
  sorry 

theorem loga_decreasing (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a y < loga a x := by
  sorry

end loga_increasing_loga_decreasing_l137_13746


namespace find_point_C_find_area_triangle_ABC_l137_13717

noncomputable section

-- Given points and equations
def point_B : ℝ × ℝ := (4, 4)
def eq_angle_bisector : ℝ × ℝ → Prop := λ p => p.2 = 0
def eq_altitude : ℝ × ℝ → Prop := λ p => p.1 - 2 * p.2 + 2 = 0

-- Target coordinates of point C
def point_C : ℝ × ℝ := (10, -8)

-- Coordinates of point A derived from given conditions
def point_A : ℝ × ℝ := (-2, 0)

-- Line equations derived from conditions
def eq_line_BC : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 - 12 = 0
def eq_line_AC : ℝ × ℝ → Prop := λ p => 2 * p.1 + 3 * p.2 + 4 = 0

-- Prove the coordinates of point C
theorem find_point_C : ∃ C : ℝ × ℝ, eq_line_BC C ∧ eq_line_AC C ∧ C = point_C := by
  sorry

-- Prove the area of triangle ABC.
theorem find_area_triangle_ABC : ∃ S : ℝ, S = 48 := by
  sorry

end find_point_C_find_area_triangle_ABC_l137_13717


namespace total_profit_is_27_l137_13750

noncomputable def total_profit : ℕ :=
  let natasha_money := 60
  let carla_money := natasha_money / 3
  let cosima_money := carla_money / 2
  let sergio_money := 3 * cosima_money / 2

  let natasha_spent := 4 * 15
  let carla_spent := 6 * 10
  let cosima_spent := 5 * 8
  let sergio_spent := 3 * 12

  let natasha_profit := natasha_spent * 10 / 100
  let carla_profit := carla_spent * 15 / 100
  let cosima_profit := cosima_spent * 12 / 100
  let sergio_profit := sergio_spent * 20 / 100

  natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_is_27 : total_profit = 27 := by
  sorry

end total_profit_is_27_l137_13750


namespace equal_white_black_balls_l137_13723

theorem equal_white_black_balls (b w n x : ℕ) 
(h1 : x = n - x)
: (x = b + w - n + x - w) := sorry

end equal_white_black_balls_l137_13723


namespace find_expression_l137_13713

theorem find_expression (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end find_expression_l137_13713
