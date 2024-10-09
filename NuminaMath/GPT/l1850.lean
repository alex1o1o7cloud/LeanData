import Mathlib

namespace quadrilateral_angle_E_l1850_185037

theorem quadrilateral_angle_E (E F G H : ℝ)
  (h1 : E = 3 * F)
  (h2 : E = 4 * G)
  (h3 : E = 6 * H)
  (h_sum : E + F + G + H = 360) :
  E = 206 :=
by
  sorry

end quadrilateral_angle_E_l1850_185037


namespace total_number_of_feet_l1850_185032

theorem total_number_of_feet 
  (H C F : ℕ)
  (h1 : H + C = 44)
  (h2 : H = 24)
  (h3 : F = 2 * H + 4 * C) : 
  F = 128 :=
by
  sorry

end total_number_of_feet_l1850_185032


namespace problem_l1850_185034

theorem problem (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
  sorry

end problem_l1850_185034


namespace ratio_of_squares_l1850_185094

noncomputable def right_triangle : Type := sorry -- Placeholder for the right triangle type

variables (a b c : ℕ)

-- Given lengths of the triangle sides
def triangle_sides (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2

-- Define x and y based on the conditions in the problem
def side_length_square_x (x : ℝ) : Prop :=
  0 < x ∧ x < 5 ∧ x < 12

def side_length_square_y (y : ℝ) : Prop :=
  0 < y ∧ y < 13

-- The main theorem to prove
theorem ratio_of_squares (x y : ℝ) :
  ∀ a b c, triangle_sides a b c →
  side_length_square_x x →
  side_length_square_y y →
  x / y = 1 :=
sorry

end ratio_of_squares_l1850_185094


namespace exists_two_elements_l1850_185046

variable (F : Finset (Finset ℕ))
variable (h1 : ∀ (A B : Finset ℕ), A ∈ F → B ∈ F → (A ∪ B) ∈ F)
variable (h2 : ∀ (A : Finset ℕ), A ∈ F → ¬ (3 ∣ A.card))

theorem exists_two_elements : ∃ (x y : ℕ), ∀ (A : Finset ℕ), A ∈ F → x ∈ A ∨ y ∈ A :=
by
  sorry

end exists_two_elements_l1850_185046


namespace divisor_is_three_l1850_185017

theorem divisor_is_three (n d q p : ℕ) (h1 : n = d * q + 3) (h2 : n^2 = d * p + 3) : d = 3 := 
sorry

end divisor_is_three_l1850_185017


namespace bowling_average_before_last_match_l1850_185003

theorem bowling_average_before_last_match
  (wickets_before_last : ℕ)
  (wickets_last_match : ℕ)
  (runs_last_match : ℕ)
  (decrease_in_average : ℝ)
  (average_before_last : ℝ) :

  wickets_before_last = 115 →
  wickets_last_match = 6 →
  runs_last_match = 26 →
  decrease_in_average = 0.4 →
  (average_before_last - decrease_in_average) = 
  ((wickets_before_last * average_before_last + runs_last_match) / 
  (wickets_before_last + wickets_last_match)) →
  average_before_last = 12.4 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end bowling_average_before_last_match_l1850_185003


namespace real_root_solution_l1850_185040

theorem real_root_solution (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ∃ x1 x2 : ℝ, 
    (x1 < b ∧ b < x2) ∧
    (1 / (x1 - a) + 1 / (x1 - b) + 1 / (x1 - c) = 0) ∧ 
    (1 / (x2 - a) + 1 / (x2 - b) + 1 / (x2 - c) = 0) :=
by
  sorry

end real_root_solution_l1850_185040


namespace lives_per_player_l1850_185012

theorem lives_per_player (num_players total_lives : ℕ) (h1 : num_players = 8) (h2 : total_lives = 64) :
  total_lives / num_players = 8 := by
  sorry

end lives_per_player_l1850_185012


namespace negation_of_proposition_l1850_185011

theorem negation_of_proposition (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by
  sorry

end negation_of_proposition_l1850_185011


namespace second_investment_value_l1850_185069

theorem second_investment_value
  (a : ℝ) (r1 r2 rt : ℝ) (x : ℝ)
  (h1 : a = 500)
  (h2 : r1 = 0.07)
  (h3 : r2 = 0.09)
  (h4 : rt = 0.085)
  (h5 : r1 * a + r2 * x = rt * (a + x)) :
  x = 1500 :=
by 
  -- The proof will go here
  sorry

end second_investment_value_l1850_185069


namespace fruit_salad_total_l1850_185039

def fruit_salad_problem (R_red G R_rasp total_fruit : ℕ) : Prop :=
  R_red = 67 ∧ (3 * G + 7 = 67) ∧ (R_rasp = G - 5) ∧ (total_fruit = R_red + G + R_rasp)

theorem fruit_salad_total (R_red G R_rasp : ℕ) (total_fruit : ℕ) :
  fruit_salad_problem R_red G R_rasp total_fruit → total_fruit = 102 :=
by
  intro h
  sorry

end fruit_salad_total_l1850_185039


namespace sally_spent_total_l1850_185086

section SallySpending

def peaches : ℝ := 12.32
def cherries : ℝ := 11.54
def total_spent : ℝ := peaches + cherries

theorem sally_spent_total :
  total_spent = 23.86 := by
  sorry

end SallySpending

end sally_spent_total_l1850_185086


namespace volume_of_stone_l1850_185097

def width := 16
def length := 14
def full_height := 9
def initial_water_height := 4
def final_water_height := 9

def volume_before := length * width * initial_water_height
def volume_after := length * width * final_water_height

def volume_stone := volume_after - volume_before

theorem volume_of_stone : volume_stone = 1120 := by
  unfold volume_stone
  unfold volume_after volume_before
  unfold final_water_height initial_water_height width length
  sorry

end volume_of_stone_l1850_185097


namespace correct_average_is_18_l1850_185028

theorem correct_average_is_18 (incorrect_avg : ℕ) (incorrect_num : ℕ) (true_num : ℕ) (n : ℕ) 
  (h1 : incorrect_avg = 16) (h2 : incorrect_num = 25) (h3 : true_num = 45) (h4 : n = 10) : 
  (incorrect_avg * n + (true_num - incorrect_num)) / n = 18 :=
by
  sorry

end correct_average_is_18_l1850_185028


namespace smallest_positive_multiple_of_6_and_5_l1850_185009

theorem smallest_positive_multiple_of_6_and_5 : ∃ (n : ℕ), (n > 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
  sorry

end smallest_positive_multiple_of_6_and_5_l1850_185009


namespace clock_correction_calculation_l1850_185036

noncomputable def clock_correction : ℝ :=
  let daily_gain := 5/4
  let hourly_gain := daily_gain / 24
  let total_hours := (9 * 24) + 9
  let total_gain := total_hours * hourly_gain
  total_gain

theorem clock_correction_calculation : clock_correction = 11.72 := by
  sorry

end clock_correction_calculation_l1850_185036


namespace no_real_solutions_for_eqn_l1850_185015

theorem no_real_solutions_for_eqn :
  ¬ ∃ x : ℝ, (x + 4) ^ 2 = 3 * (x - 2) := 
by 
  sorry

end no_real_solutions_for_eqn_l1850_185015


namespace cos_double_angle_zero_l1850_185005

variable (θ : ℝ)

-- Conditions
def tan_eq_one : Prop := Real.tan θ = 1

-- Objective
theorem cos_double_angle_zero (h : tan_eq_one θ) : Real.cos (2 * θ) = 0 :=
sorry

end cos_double_angle_zero_l1850_185005


namespace longer_side_of_new_rectangle_l1850_185055

theorem longer_side_of_new_rectangle {z : ℕ} (h : ∃x : ℕ, 9 * 16 = 144 ∧ x * z = 144 ∧ z ≠ 9 ∧ z ≠ 16) : z = 18 :=
sorry

end longer_side_of_new_rectangle_l1850_185055


namespace sum_of_f_l1850_185079

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) / (3^x + (Real.sqrt 3))

theorem sum_of_f :
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6) + 
   f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f (0) + f (1) + f (2) + 
   f (3) + f (4) + f (5) + f (6) + f (7) + f (8) + f (9) + f (10) + 
   f (11) + f (12) + f (13)) = 13 :=
sorry

end sum_of_f_l1850_185079


namespace find_n_l1850_185053

noncomputable def condition (n : ℕ) : Prop :=
  (1/5)^n * (1/4)^18 = 1 / (2 * 10^35)

theorem find_n (n : ℕ) (h : condition n) : n = 35 :=
by
  sorry

end find_n_l1850_185053


namespace value_of_expression_at_x_eq_2_l1850_185047

theorem value_of_expression_at_x_eq_2 :
  (2 * (2: ℕ)^2 - 3 * 2 + 4 = 6) := 
by sorry

end value_of_expression_at_x_eq_2_l1850_185047


namespace rectangular_field_area_eq_l1850_185078

-- Definitions based on the problem's conditions
def length (x : ℝ) := x
def width (x : ℝ) := 60 - x
def area (x : ℝ) := x * (60 - x)

-- The proof statement
theorem rectangular_field_area_eq (x : ℝ) (h₀ : x + (60 - x) = 60) (h₁ : area x = 864) :
  x * (60 - x) = 864 :=
by
  -- Using the provided conditions and definitions, we aim to prove the equation.
  sorry

end rectangular_field_area_eq_l1850_185078


namespace fraction_to_percentage_l1850_185052

theorem fraction_to_percentage (x : ℝ) (hx : 0 < x) : 
  (x / 50 + x / 25) = 0.06 * x := 
sorry

end fraction_to_percentage_l1850_185052


namespace binom_arithmetic_sequence_l1850_185025

noncomputable def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_arithmetic_sequence {n : ℕ} (h : 2 * binom n 5 = binom n 4 + binom n 6) (n_eq : n = 14) : binom n 12 = 91 := by
  sorry

end binom_arithmetic_sequence_l1850_185025


namespace total_wasted_time_is_10_l1850_185059

-- Define the time Martin spends waiting in traffic
def waiting_time : ℕ := 2

-- Define the constant for the multiplier
def multiplier : ℕ := 4

-- Define the time spent trying to get off the freeway
def off_freeway_time : ℕ := waiting_time * multiplier

-- Define the total wasted time
def total_wasted_time : ℕ := waiting_time + off_freeway_time

-- Theorem stating that the total time wasted is 10 hours
theorem total_wasted_time_is_10 : total_wasted_time = 10 :=
by
  sorry

end total_wasted_time_is_10_l1850_185059


namespace find_n_l1850_185085

/-- Given: 
1. The second term in the expansion of (x + a)^n is binom n 1 * x^(n-1) * a = 210.
2. The third term in the expansion of (x + a)^n is binom n 2 * x^(n-2) * a^2 = 840.
3. The fourth term in the expansion of (x + a)^n is binom n 3 * x^(n-3) * a^3 = 2520.
We are to prove that n = 10. -/
theorem find_n (x a : ℕ) (n : ℕ)
  (h1 : Nat.choose n 1 * x^(n-1) * a = 210)
  (h2 : Nat.choose n 2 * x^(n-2) * a^2 = 840)
  (h3 : Nat.choose n 3 * x^(n-3) * a^3 = 2520) : 
  n = 10 := by sorry

end find_n_l1850_185085


namespace initial_walking_speed_l1850_185056

theorem initial_walking_speed
  (t : ℝ) -- Time in minutes for bus to reach the bus stand from when the person starts walking
  (h₁ : 5 = 5 * ((t - 5) / 60)) -- When walking at 5 km/h, person reaches 5 minutes early
  (h₂ : 5 = v * ((t + 10) / 60)) -- At initial speed v, person misses the bus by 10 minutes
  : v = 4 := 
by
  sorry

end initial_walking_speed_l1850_185056


namespace remainders_are_distinct_l1850_185004

theorem remainders_are_distinct (a : ℕ → ℕ) (H1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i ≠ a (i % 100 + 1))
  (H2 : ∃ r1 r2 : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i % a (i % 100 + 1) = r1 ∨ a i % a (i % 100 + 1) = r2) :
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 100 → (a (i % 100 + 1) % a i) ≠ (a (j % 100 + 1) % a j) :=
by
  sorry

end remainders_are_distinct_l1850_185004


namespace max_sum_red_green_balls_l1850_185045

theorem max_sum_red_green_balls (total_balls : ℕ) (green_balls : ℕ) (max_red_balls : ℕ) 
  (h1 : total_balls = 28) (h2 : green_balls = 12) (h3 : max_red_balls ≤ 11) : 
  (max_red_balls + green_balls) = 23 := 
sorry

end max_sum_red_green_balls_l1850_185045


namespace kayla_total_items_l1850_185043

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l1850_185043


namespace dealer_selling_price_above_cost_l1850_185038

variable (cost_price : ℝ := 100)
variable (discount_percent : ℝ := 20)
variable (profit_percent : ℝ := 20)

theorem dealer_selling_price_above_cost :
  ∀ (x : ℝ), 
  (0.8 * x = 1.2 * cost_price) → 
  x = cost_price * (1 + profit_percent / 100) :=
by
  sorry

end dealer_selling_price_above_cost_l1850_185038


namespace find_natural_numbers_l1850_185084

theorem find_natural_numbers (x y z : ℕ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_ordered : x < y ∧ y < z)
  (h_reciprocal_sum_nat : ∃ a : ℕ, 1/x + 1/y + 1/z = a) : (x, y, z) = (2, 3, 6) := 
sorry

end find_natural_numbers_l1850_185084


namespace shorter_side_length_l1850_185077

variables (x y : ℝ)
variables (h1 : 2 * x + 2 * y = 60)
variables (h2 : x * y = 200)

theorem shorter_side_length :
  min x y = 10 :=
by
  sorry

end shorter_side_length_l1850_185077


namespace distinct_strings_after_operations_l1850_185002

def valid_strings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else valid_strings (n-1) + valid_strings (n-2)

theorem distinct_strings_after_operations :
  valid_strings 10 = 144 := by
  sorry

end distinct_strings_after_operations_l1850_185002


namespace problem1_problem2_l1850_185087

theorem problem1 (x : ℝ) : 2 * (x - 1) ^ 2 = 18 ↔ x = 4 ∨ x = -2 := by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem1_problem2_l1850_185087


namespace minimum_distance_AB_l1850_185041

-- Definitions of the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 - y + 1 = 0
def C2 (x y : ℝ) : Prop := y^2 - x + 1 = 0

theorem minimum_distance_AB :
  ∃ (A B : ℝ × ℝ), C1 A.1 A.2 ∧ C2 B.1 B.2 ∧ dist A B = 3*Real.sqrt 2 / 4 := sorry

end minimum_distance_AB_l1850_185041


namespace purchase_total_cost_l1850_185065

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end purchase_total_cost_l1850_185065


namespace initial_solution_weight_100kg_l1850_185066

theorem initial_solution_weight_100kg
  (W : ℝ)
  (initial_salt_percentage : ℝ)
  (added_salt : ℝ)
  (final_salt_percentage : ℝ)
  (H1 : initial_salt_percentage = 0.10)
  (H2 : added_salt = 12.5)
  (H3 : final_salt_percentage = 0.20)
  (H4 : 0.20 * (W + 12.5) = 0.10 * W + 12.5) :
  W = 100 :=   
by 
  sorry

end initial_solution_weight_100kg_l1850_185066


namespace integer_pairs_solution_l1850_185068

theorem integer_pairs_solution (k : ℕ) (h : k ≠ 1) : 
  ∃ (m n : ℤ), 
    ((m - n) ^ 2 = 4 * m * n / (m + n - 1)) ∧ 
    (m = k^2 + k / 2 ∧ n = k^2 - k / 2) ∨ 
    (m = k^2 - k / 2 ∧ n = k^2 + k / 2) :=
sorry

end integer_pairs_solution_l1850_185068


namespace greatest_third_side_of_triangle_l1850_185061

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l1850_185061


namespace quadratic_roots_condition_l1850_185092

theorem quadratic_roots_condition (m : ℝ) :
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1^2 - 3 * x1 + 2 * m = 0 ∧ x2^2 - 3 * x2 + 2 * m = 0) →
  m < 9 / 8 :=
by
  sorry

end quadratic_roots_condition_l1850_185092


namespace grace_earnings_in_september_l1850_185007

theorem grace_earnings_in_september
  (hours_mowing : ℕ) (hours_pulling_weeds : ℕ) (hours_putting_mulch : ℕ)
  (rate_mowing : ℕ) (rate_pulling_weeds : ℕ) (rate_putting_mulch : ℕ)
  (total_hours_mowing : hours_mowing = 63) (total_hours_pulling_weeds : hours_pulling_weeds = 9) (total_hours_putting_mulch : hours_putting_mulch = 10)
  (rate_for_mowing : rate_mowing = 6) (rate_for_pulling_weeds : rate_pulling_weeds = 11) (rate_for_putting_mulch : rate_putting_mulch = 9) :
  hours_mowing * rate_mowing + hours_pulling_weeds * rate_pulling_weeds + hours_putting_mulch * rate_putting_mulch = 567 :=
by
  intros
  sorry

end grace_earnings_in_september_l1850_185007


namespace lydia_ate_24_ounces_l1850_185074

theorem lydia_ate_24_ounces (total_fruit_pounds : ℕ) (mario_oranges_ounces : ℕ) (nicolai_peaches_pounds : ℕ) (total_fruit_ounces mario_oranges_ounces_in_ounces nicolai_peaches_ounces_in_ounces : ℕ) :
  total_fruit_pounds = 8 →
  mario_oranges_ounces = 8 →
  nicolai_peaches_pounds = 6 →
  total_fruit_ounces = total_fruit_pounds * 16 →
  mario_oranges_ounces_in_ounces = mario_oranges_ounces →
  nicolai_peaches_ounces_in_ounces = nicolai_peaches_pounds * 16 →
  (total_fruit_ounces - mario_oranges_ounces_in_ounces - nicolai_peaches_ounces_in_ounces) = 24 :=
by
  sorry

end lydia_ate_24_ounces_l1850_185074


namespace radius_of_circle_l1850_185000

noncomputable def circle_radius (x y : ℝ) : ℝ := 
  let lhs := x^2 - 8 * x + y^2 - 4 * y + 16
  if lhs = 0 then 2 else 0

theorem radius_of_circle : circle_radius 0 0 = 2 :=
sorry

end radius_of_circle_l1850_185000


namespace compute_large_expression_l1850_185072

theorem compute_large_expression :
  ( (11^4 + 484) * (23^4 + 484) * (35^4 + 484) * (47^4 + 484) * (59^4 + 484) ) / 
  ( (5^4 + 484) * (17^4 + 484) * (29^4 + 484) * (41^4 + 484) * (53^4 + 484) ) = 552.42857 := 
sorry

end compute_large_expression_l1850_185072


namespace interior_angles_sum_l1850_185051

def sum_of_interior_angles (sides : ℕ) : ℕ :=
  180 * (sides - 2)

theorem interior_angles_sum (n : ℕ) (h : sum_of_interior_angles n = 1800) :
  sum_of_interior_angles (n + 4) = 2520 :=
sorry

end interior_angles_sum_l1850_185051


namespace inequality_proof_l1850_185083

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)

theorem inequality_proof :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) >= 3 / 2 :=
by
  sorry

end inequality_proof_l1850_185083


namespace christopher_more_than_karen_l1850_185080

-- Define the number of quarters Karen and Christopher have
def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64

-- Define the value of a quarter in dollars
def value_of_quarter : ℚ := 0.25

-- Define the amount of money Christopher has more than Karen in dollars
def christopher_more_money : ℚ := (christopher_quarters - karen_quarters) * value_of_quarter

-- Theorem to prove that Christopher has $8.00 more than Karen
theorem christopher_more_than_karen : christopher_more_money = 8 := by
  sorry

end christopher_more_than_karen_l1850_185080


namespace hyperbola_represents_l1850_185018

theorem hyperbola_represents (k : ℝ) : 
  (k - 2) * (5 - k) < 0 ↔ (k < 2 ∨ k > 5) :=
by
  sorry

end hyperbola_represents_l1850_185018


namespace TotalLaddersClimbedInCentimeters_l1850_185093

def keaton_ladder_height := 50  -- height of Keaton's ladder in meters
def keaton_climbs := 30  -- number of times Keaton climbs the ladder

def reece_ladder_height := keaton_ladder_height - 6  -- height of Reece's ladder in meters
def reece_climbs := 25  -- number of times Reece climbs the ladder

def total_meters_climbed := (keaton_ladder_height * keaton_climbs) + (reece_ladder_height * reece_climbs)

def total_cm_climbed := total_meters_climbed * 100

theorem TotalLaddersClimbedInCentimeters :
  total_cm_climbed = 260000 :=
by
  sorry

end TotalLaddersClimbedInCentimeters_l1850_185093


namespace cost_comparison_cost_effectiveness_47_l1850_185006

section
variable (x : ℕ)

-- Conditions
def price_teapot : ℕ := 25
def price_teacup : ℕ := 5
def quantity_teapots : ℕ := 4
def discount_scheme_2 : ℝ := 0.94

-- Total cost for Scheme 1
def cost_scheme_1 (x : ℕ) : ℕ :=
  (quantity_teapots * price_teapot) + (price_teacup * (x - quantity_teapots))

-- Total cost for Scheme 2
def cost_scheme_2 (x : ℕ) : ℝ :=
  (quantity_teapots * price_teapot + price_teacup * x : ℝ) * discount_scheme_2

-- The proof problem
theorem cost_comparison (x : ℕ) (h : x ≥ 4) :
  cost_scheme_1 x = 5 * x + 80 ∧ cost_scheme_2 x = 4.7 * x + 94 :=
sorry

-- When x = 47
theorem cost_effectiveness_47 : cost_scheme_2 47 < cost_scheme_1 47 :=
sorry

end

end cost_comparison_cost_effectiveness_47_l1850_185006


namespace f_10_equals_1_l1850_185049

noncomputable def f : ℝ → ℝ 
| x => sorry 

axiom odd_f_x_minus_1 : ∀ x : ℝ, f (-x-1) = -f (x-1)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x+1) = f (x+1)
axiom f_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 2^x

theorem f_10_equals_1 : f 10 = 1 :=
by
  sorry -- The actual proof goes here.

end f_10_equals_1_l1850_185049


namespace solve_equation_l1850_185075

theorem solve_equation (x : ℚ) : 3 * (x - 2) = 2 - 5 * (x - 2) ↔ x = 9 / 4 := by
  sorry

end solve_equation_l1850_185075


namespace least_positive_whole_number_divisible_by_five_primes_l1850_185073

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l1850_185073


namespace equation_of_line_projection_l1850_185019

theorem equation_of_line_projection (x y : ℝ) (m : ℝ) (x1 x2 : ℝ) (d : ℝ)
  (h1 : (5, 3) ∈ {(x, y) | y = 3 + m * (x - 5)})
  (h2 : x1 = (16 + 20 * m - 12) / (4 * m + 3))
  (h3 : x2 = (1 + 20 * m - 12) / (4 * m + 3))
  (h4 : abs (x1 - x2) = 1) :
  (y = 3 * x - 12 ∨ y = -4.5 * x + 25.5) :=
sorry

end equation_of_line_projection_l1850_185019


namespace part1_part2_l1850_185050

theorem part1 (x : ℝ) : 3 + 2 * x > - x - 6 ↔ x > -3 := by
  sorry

theorem part2 (x : ℝ) : 2 * x + 1 ≤ x + 3 ∧ (2 * x + 1) / 3 > 1 ↔ 1 < x ∧ x ≤ 2 := by
  sorry

end part1_part2_l1850_185050


namespace solve_for_x_l1850_185030

noncomputable def avg (a b : ℝ) := (a + b) / 2

noncomputable def B (t : List ℝ) : List ℝ :=
  match t with
  | [a, b, c, d, e] => [avg a b, avg b c, avg c d, avg d e]
  | _ => []

noncomputable def B_iter (m : ℕ) (t : List ℝ) : List ℝ :=
  match m with
  | 0 => t
  | k + 1 => B (B_iter k t)

theorem solve_for_x (x : ℝ) (h1 : 0 < x) (h2 : B_iter 4 [1, x, x^2, x^3, x^4] = [1/4]) :
  x = Real.sqrt 2 - 1 :=
sorry

end solve_for_x_l1850_185030


namespace simple_interest_for_2_years_l1850_185035

noncomputable def calculate_simple_interest (P r t : ℝ) : ℝ :=
  (P * r * t) / 100

theorem simple_interest_for_2_years (CI P r t : ℝ) (hCI : CI = P * (1 + r / 100)^t - P)
  (hCI_value : CI = 615) (r_value : r = 5) (t_value : t = 2) : 
  calculate_simple_interest P r t = 600 :=
by
  sorry

end simple_interest_for_2_years_l1850_185035


namespace arithmetic_mean_six_expressions_l1850_185081

theorem arithmetic_mean_six_expressions (x : ℝ) :
  (x + 10 + 17 + 2 * x + 15 + 2 * x + 6 + 3 * x - 5) / 6 = 30 →
  x = 137 / 8 :=
by
  sorry

end arithmetic_mean_six_expressions_l1850_185081


namespace total_expense_l1850_185044

noncomputable def sandys_current_age : ℕ := 36 - 2
noncomputable def sandys_monthly_expense : ℕ := 10 * sandys_current_age
noncomputable def alexs_current_age : ℕ := sandys_current_age / 2
noncomputable def alexs_next_month_expense : ℕ := 2 * sandys_monthly_expense

theorem total_expense : 
  sandys_monthly_expense + alexs_next_month_expense = 1020 := 
by 
  sorry

end total_expense_l1850_185044


namespace Sara_team_wins_l1850_185029

theorem Sara_team_wins (total_games losses wins : ℕ) (h1 : total_games = 12) (h2 : losses = 4) (h3 : wins = total_games - losses) :
  wins = 8 :=
by
  sorry

end Sara_team_wins_l1850_185029


namespace initial_nickels_l1850_185010

variable (q0 n0 : Nat)
variable (d_nickels : Nat := 3) -- His dad gave him 3 nickels
variable (final_nickels : Nat := 12) -- Tim has now 12 nickels

theorem initial_nickels (q0 : Nat) (n0 : Nat) (d_nickels : Nat) (final_nickels : Nat) :
  final_nickels = n0 + d_nickels → n0 = 9 :=
by
  sorry

end initial_nickels_l1850_185010


namespace slope_of_line_l1850_185031

-- Definition of the line equation
def lineEquation (x y : ℝ) : Prop := 4 * x - 7 * y = 14

-- The statement that we need to prove
theorem slope_of_line : ∀ x y, lineEquation x y → ∃ m, m = 4 / 7 :=
by {
  sorry
}

end slope_of_line_l1850_185031


namespace problem1_problem2_l1850_185063

-- Define the given sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x < a + 4 }
def setB : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Problem 1: Prove A ∩ B = { x | -3 < x ∧ x < -1 } when a = 1
theorem problem1 (a : ℝ) (h : a = 1) : 
  (setA a ∩ setB) = { x : ℝ | -3 < x ∧ x < -1 } := sorry

-- Problem 2: Prove range of a given A ∪ B = ℝ is (1, 3)
theorem problem2 (a : ℝ) : 
  (forall x : ℝ, x ∈ (setA a ∪ setB)) ↔ (1 < a ∧ a < 3) := sorry

end problem1_problem2_l1850_185063


namespace books_from_library_l1850_185095

def initial_books : ℝ := 54.5
def additional_books_1 : ℝ := 23.7
def returned_books_1 : ℝ := 12.3
def additional_books_2 : ℝ := 15.6
def returned_books_2 : ℝ := 9.1
def additional_books_3 : ℝ := 7.2

def total_books : ℝ :=
  initial_books + additional_books_1 - returned_books_1 + additional_books_2 - returned_books_2 + additional_books_3

theorem books_from_library : total_books = 79.6 := by
  sorry

end books_from_library_l1850_185095


namespace cole_round_trip_time_l1850_185088

theorem cole_round_trip_time :
  ∀ (speed_to_work speed_return : ℝ) (time_to_work_minutes : ℝ),
  speed_to_work = 75 ∧ speed_return = 105 ∧ time_to_work_minutes = 210 →
  (time_to_work_minutes / 60 + (speed_to_work * (time_to_work_minutes / 60)) / speed_return) = 6 := 
by
  sorry

end cole_round_trip_time_l1850_185088


namespace polygon_sides_given_interior_angle_l1850_185014

theorem polygon_sides_given_interior_angle
  (h : ∀ (n : ℕ), (n > 2) → ((n - 2) * 180 = n * 140)): n = 9 := by
  sorry

end polygon_sides_given_interior_angle_l1850_185014


namespace product_of_xy_l1850_185089

theorem product_of_xy : 
  ∃ (x y : ℝ), 3 * x + 4 * y = 60 ∧ 6 * x - 4 * y = 12 ∧ x * y = 72 :=
by
  sorry

end product_of_xy_l1850_185089


namespace sales_in_second_month_l1850_185076

-- Given conditions:
def sales_first_month : ℕ := 6400
def sales_third_month : ℕ := 6800
def sales_fourth_month : ℕ := 7200
def sales_fifth_month : ℕ := 6500
def sales_sixth_month : ℕ := 5100
def average_sales : ℕ := 6500

-- Statement to prove:
theorem sales_in_second_month :
  ∃ (sales_second_month : ℕ), 
    average_sales * 6 = sales_first_month + sales_second_month + sales_third_month 
    + sales_fourth_month + sales_fifth_month + sales_sixth_month 
    ∧ sales_second_month = 7000 :=
  sorry

end sales_in_second_month_l1850_185076


namespace correct_propositions_l1850_185064

-- Define propositions
def proposition1 : Prop :=
  ∀ x, 2 * (Real.cos (1/3 * x + Real.pi / 4))^2 - 1 = -Real.sin (2 * x / 3)

def proposition2 : Prop :=
  ∃ α : ℝ, Real.sin α + Real.cos α = 3 / 2

def proposition3 : Prop :=
  ∀ α β : ℝ, (0 < α ∧ α < Real.pi / 2) → (0 < β ∧ β < Real.pi / 2) → α < β → Real.tan α < Real.tan β

def proposition4 : Prop :=
  ∀ x, x = Real.pi / 8 → Real.sin (2 * x + 5 * Real.pi / 4) = -1

def proposition5 : Prop :=
  Real.sin ( 2 * (Real.pi / 12) + Real.pi / 3 ) = 0

-- Define the main theorem combining correct propositions
theorem correct_propositions : 
  proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ proposition4 ∧ ¬proposition5 :=
  by
  -- Since we only need to state the theorem, we use sorry.
  sorry

end correct_propositions_l1850_185064


namespace walt_part_time_job_l1850_185048

theorem walt_part_time_job (x : ℝ) 
  (h1 : 0.09 * x + 0.08 * 4000 = 770) : 
  x + 4000 = 9000 := by
  sorry

end walt_part_time_job_l1850_185048


namespace find_real_m_of_purely_imaginary_z_l1850_185090

theorem find_real_m_of_purely_imaginary_z (m : ℝ) 
  (h1 : m^2 - 8 * m + 15 = 0) 
  (h2 : m^2 - 9 * m + 18 ≠ 0) : 
  m = 5 := 
by 
  sorry

end find_real_m_of_purely_imaginary_z_l1850_185090


namespace ski_price_l1850_185054

variable {x y : ℕ}

theorem ski_price (h1 : 2 * x + y = 340) (h2 : 3 * x + 2 * y = 570) : x = 110 ∧ y = 120 := by
  sorry

end ski_price_l1850_185054


namespace fourth_person_height_l1850_185082

theorem fourth_person_height (H : ℝ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 77) : 
  H + 10 = 83 :=
sorry

end fourth_person_height_l1850_185082


namespace Chad_savings_l1850_185022

theorem Chad_savings :
  let earnings_mowing := 600
  let earnings_birthday := 250
  let earnings_video_games := 150
  let earnings_odd_jobs := 150
  let total_earnings := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := 0.40
  let savings := savings_rate * total_earnings
  savings = 460 :=
by
  -- Definitions
  let earnings_mowing : ℤ := 600
  let earnings_birthday : ℤ := 250
  let earnings_video_games : ℤ := 150
  let earnings_odd_jobs : ℤ := 150
  let total_earnings : ℤ := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := (40:ℚ) / 100
  let savings : ℚ := savings_rate * total_earnings
  -- Proof (to be completed by sorry)
  exact sorry

end Chad_savings_l1850_185022


namespace graph_of_y_eq_neg2x_passes_quadrant_II_IV_l1850_185013

-- Definitions
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

def is_in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The main statement
theorem graph_of_y_eq_neg2x_passes_quadrant_II_IV :
  ∀ (x : ℝ), (is_in_quadrant_II x (linear_function (-2) x) ∨ 
               is_in_quadrant_IV x (linear_function (-2) x)) :=
by
  sorry

end graph_of_y_eq_neg2x_passes_quadrant_II_IV_l1850_185013


namespace trajectory_of_M_is_ellipse_l1850_185033

def circle_eq (x y : ℝ) : Prop := ((x + 3)^2 + y^2 = 100)

def point_B (x y : ℝ) : Prop := (x = 3 ∧ y = 0)

def point_on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, P = (x, y) ∧ circle_eq x y

def perpendicular_bisector_intersects_CQ_at_M (B P M : ℝ × ℝ) : Prop :=
  (B.fst = 3 ∧ B.snd = 0) ∧
  point_on_circle P ∧
  ∃ r : ℝ, (P.fst + B.fst) / 2 = M.fst ∧ r = (M.snd - P.snd) / (M.fst - P.fst) ∧ 
  r = -(P.fst - B.fst) / (P.snd - B.snd)

theorem trajectory_of_M_is_ellipse (M : ℝ × ℝ) 
  (hC : ∀ x y, circle_eq x y)
  (hB : point_B 3 0)
  (hP : ∃ P : ℝ × ℝ, point_on_circle P)
  (hM : ∃ B P : ℝ × ℝ, perpendicular_bisector_intersects_CQ_at_M B P M) 
: (M.fst^2 / 25 + M.snd^2 / 16 = 1) := 
sorry

end trajectory_of_M_is_ellipse_l1850_185033


namespace exchange_rmb_ways_l1850_185057

theorem exchange_rmb_ways : 
  {n : ℕ // ∃ (x y z : ℕ), x + 2 * y + 5 * z = 10 ∧ n = 10} :=
sorry

end exchange_rmb_ways_l1850_185057


namespace geese_count_l1850_185020

-- Define the number of ducks in the marsh
def number_of_ducks : ℕ := 37

-- Define the total number of birds in the marsh
def total_number_of_birds : ℕ := 95

-- Define the number of geese in the marsh
def number_of_geese : ℕ := total_number_of_birds - number_of_ducks

-- Theorem stating the number of geese in the marsh is 58
theorem geese_count : number_of_geese = 58 := by
  sorry

end geese_count_l1850_185020


namespace number_of_distinct_collections_l1850_185070

def mathe_matical_letters : Multiset Char :=
  {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

def vowels : Multiset Char :=
  {'A', 'A', 'A', 'E', 'I'}

def consonants : Multiset Char :=
  {'M', 'T', 'H', 'M', 'T', 'C', 'L', 'C'}

def indistinguishable (s : Multiset Char) :=
  (s.count 'A' = s.count 'A' ∧
   s.count 'E' = 1 ∧
   s.count 'I' = 1 ∧
   s.count 'M' = 2 ∧
   s.count 'T' = 2 ∧
   s.count 'H' = 1 ∧
   s.count 'C' = 2 ∧
   s.count 'L' = 1)

theorem number_of_distinct_collections :
  5 * 16 = 80 :=
by
  -- proof would go here
  sorry

end number_of_distinct_collections_l1850_185070


namespace liars_count_l1850_185026

inductive Person
| Knight
| Liar
| Eccentric

open Person

def isLiarCondition (p : Person) (right : Person) : Prop :=
  match p with
  | Knight => right = Liar
  | Liar => right ≠ Liar
  | Eccentric => True

theorem liars_count (people : Fin 100 → Person) (h : ∀ i, isLiarCondition (people i) (people ((i + 1) % 100))) :
  (∃ n : ℕ, n = 0 ∨ n = 50) :=
sorry

end liars_count_l1850_185026


namespace guitar_center_discount_is_correct_l1850_185023

-- Define the suggested retail price
def retail_price : ℕ := 1000

-- Define the shipping fee of Guitar Center
def shipping_fee : ℕ := 100

-- Define the discount percentage offered by Sweetwater
def sweetwater_discount_rate : ℕ := 10

-- Define the amount saved by buying from the cheaper store
def savings : ℕ := 50

-- Define the discount offered by Guitar Center
def guitar_center_discount : ℕ :=
  retail_price - ((retail_price * (100 - sweetwater_discount_rate) / 100) + savings - shipping_fee)

-- Theorem: Prove that the discount offered by Guitar Center is $150
theorem guitar_center_discount_is_correct : guitar_center_discount = 150 :=
  by
    -- The proof will be filled in based on the given conditions
    sorry

end guitar_center_discount_is_correct_l1850_185023


namespace boyfriend_picks_up_correct_l1850_185058

-- Define the initial condition
def init_pieces : ℕ := 60

-- Define the amount swept by Anne
def swept_pieces (n : ℕ) : ℕ := n / 2

-- Define the number of pieces stolen by the cat
def stolen_pieces : ℕ := 3

-- Define the remaining pieces after the cat steals
def remaining_pieces (n : ℕ) : ℕ := n - stolen_pieces

-- Define how many pieces the boyfriend picks up
def boyfriend_picks_up (n : ℕ) : ℕ := n / 3

-- The main theorem
theorem boyfriend_picks_up_correct : boyfriend_picks_up (remaining_pieces (init_pieces - swept_pieces init_pieces)) = 9 :=
by
  sorry

end boyfriend_picks_up_correct_l1850_185058


namespace operations_equivalent_l1850_185042

theorem operations_equivalent (x : ℚ) : 
  ((x * (5 / 6)) / (2 / 3) - 2) = (x * (5 / 4) - 2) :=
sorry

end operations_equivalent_l1850_185042


namespace parallelogram_area_increase_l1850_185024

theorem parallelogram_area_increase (b h : ℕ) :
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  (A2 - A1) * 100 / A1 = 300 :=
by
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  sorry

end parallelogram_area_increase_l1850_185024


namespace geometric_sequence_problem_l1850_185096

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 * a 7 = 8)
  (h2 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1)):
  a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l1850_185096


namespace area_under_arccos_cos_l1850_185099

noncomputable def func (x : ℝ) : ℝ := Real.arccos (Real.cos x)

theorem area_under_arccos_cos :
  ∫ x in (0:ℝ)..3 * Real.pi, func x = 3 * Real.pi ^ 2 / 2 :=
by
  sorry

end area_under_arccos_cos_l1850_185099


namespace smallest_solution_of_equation_l1850_185016

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (x^4 - 26 * x^2 + 169 = 0) ∧ x = -Real.sqrt 13 :=
by
  sorry

end smallest_solution_of_equation_l1850_185016


namespace multiplication_result_l1850_185027

theorem multiplication_result :
  3^2 * 5^2 * 7 * 11^2 = 190575 :=
by sorry

end multiplication_result_l1850_185027


namespace slope_of_line_l1850_185062

theorem slope_of_line :
  ∃ (m : ℝ), (∃ b : ℝ, ∀ x y : ℝ, y = m * x + b) ∧
             (b = 2 ∧ ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ = 0 ∧ x₂ = 269 ∧ y₁ = 2 ∧ y₂ = 540 ∧ 
             m = (y₂ - y₁) / (x₂ - x₁)) ∧
             m = 2 :=
by {
  sorry
}

end slope_of_line_l1850_185062


namespace largest_int_less_than_100_rem_5_by_7_l1850_185098

theorem largest_int_less_than_100_rem_5_by_7 :
  ∃ k : ℤ, (7 * k + 5 = 96) ∧ ∀ n : ℤ, (7 * n + 5 < 100) → (n ≤ k) :=
sorry

end largest_int_less_than_100_rem_5_by_7_l1850_185098


namespace ab_cardinals_l1850_185067

open Set

/-- a|A| = b|B| given the conditions.
1. a and b are positive integers.
2. A and B are finite sets of integers such that:
   a. A and B are disjoint.
   b. If an integer i belongs to A or to B, then i + a ∈ A or i - b ∈ B.
-/
theorem ab_cardinals 
  (a b : ℕ) (A B : Finset ℤ) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (disjoint_AB : Disjoint A B)
  (condition_2 : ∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := 
sorry

end ab_cardinals_l1850_185067


namespace cylinder_volume_scaling_l1850_185060

theorem cylinder_volume_scaling (r h : ℝ) (V : ℝ) (V' : ℝ) 
  (h_original : V = π * r^2 * h) 
  (h_new : V' = π * (1.5 * r)^2 * (3 * h)) :
  V' = 6.75 * V := by
  sorry

end cylinder_volume_scaling_l1850_185060


namespace sector_radius_l1850_185008

theorem sector_radius (A L : ℝ) (hA : A = 240 * Real.pi) (hL : L = 20 * Real.pi) : 
  ∃ r : ℝ, r = 24 :=
by
  sorry

end sector_radius_l1850_185008


namespace prop_B_contrapositive_correct_l1850_185071

/-
Proposition B: The contrapositive of the proposition 
"If x^2 < 1, then -1 < x < 1" is 
"If x ≥ 1 or x ≤ -1, then x^2 ≥ 1".
-/
theorem prop_B_contrapositive_correct :
  (∀ (x : ℝ), x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ (x : ℝ), (x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
sorry

end prop_B_contrapositive_correct_l1850_185071


namespace minimum_people_who_like_both_l1850_185091

open Nat

theorem minimum_people_who_like_both (total : ℕ) (mozart : ℕ) (bach : ℕ)
  (h_total: total = 100) (h_mozart: mozart = 87) (h_bach: bach = 70) :
  ∃ x, x = mozart + bach - total ∧ x ≥ 57 :=
by
  sorry

end minimum_people_who_like_both_l1850_185091


namespace fill_up_minivans_l1850_185001

theorem fill_up_minivans (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ)
  (mini_van_liters : ℝ) (truck_percent_bigger : ℝ) (num_trucks : ℕ) (num_minivans : ℕ) :
  service_cost = 2.3 ∧ fuel_cost_per_liter = 0.7 ∧ total_cost = 396 ∧
  mini_van_liters = 65 ∧ truck_percent_bigger = 1.2 ∧ num_trucks = 2 →
  num_minivans = 4 :=
by
  sorry

end fill_up_minivans_l1850_185001


namespace equilateral_triangle_area_decrease_l1850_185021

theorem equilateral_triangle_area_decrease (A : ℝ) (A' : ℝ) (s s' : ℝ) 
  (h1 : A = 121 * Real.sqrt 3) 
  (h2 : A = (s^2 * Real.sqrt 3) / 4) 
  (h3 : s' = s - 8) 
  (h4 : A' = (s'^2 * Real.sqrt 3) / 4) :
  A - A' = 72 * Real.sqrt 3 := 
by sorry

end equilateral_triangle_area_decrease_l1850_185021
