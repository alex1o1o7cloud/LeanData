import Mathlib

namespace part1_part2_l1075_107591

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x m : ℝ) : ℝ := -|x + 3| + m

def solution_set_ineq_1 (a : ℝ) : Set ℝ :=
  if a = 1 then {x | x < 2 ∨ x > 2}
  else if a > 1 then Set.univ
  else {x | x < 1 + a ∨ x > 3 - a}

theorem part1 (a : ℝ) : 
  ∃ S : Set ℝ, S = solution_set_ineq_1 a ∧ ∀ x : ℝ, (f x + a - 1 > 0) ↔ x ∈ S := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := sorry

end part1_part2_l1075_107591


namespace lab_techs_share_l1075_107586

theorem lab_techs_share (u c t : ℕ) 
  (h1 : c = 6 * u)
  (h2 : t = u / 2)
  (h3 : u = 12) : 
  (c + u) / t = 14 := 
by 
  sorry

end lab_techs_share_l1075_107586


namespace hyperbola_equation_l1075_107554

noncomputable def distance_between_vertices : ℝ := 8
noncomputable def eccentricity : ℝ := 5 / 4

theorem hyperbola_equation :
  ∃ a b c : ℝ, 2 * a = distance_between_vertices ∧ 
               c = a * eccentricity ∧ 
               b^2 = c^2 - a^2 ∧ 
               (a = 4 ∧ c = 5 ∧ b^2 = 9) ∧ 
               ∀ x y : ℝ, (x^2 / (a:ℝ)^2) - (y^2 / (b:ℝ)^2) = 1 :=
by 
  sorry

end hyperbola_equation_l1075_107554


namespace compare_abc_l1075_107501

noncomputable def a : ℝ := (0.6)^(2/5)
noncomputable def b : ℝ := (0.4)^(2/5)
noncomputable def c : ℝ := (0.4)^(3/5)

theorem compare_abc : a > b ∧ b > c := 
by
  sorry

end compare_abc_l1075_107501


namespace cricket_team_members_l1075_107546

theorem cricket_team_members (n : ℕ) (captain_age wicket_keeper_age average_whole_age average_remaining_age : ℕ) :
  captain_age = 24 →
  wicket_keeper_age = 31 →
  average_whole_age = 23 →
  average_remaining_age = 22 →
  n * average_whole_age - captain_age - wicket_keeper_age = (n - 2) * average_remaining_age →
  n = 11 :=
by
  intros h_cap_age h_wk_age h_avg_whole h_avg_remain h_eq
  sorry

end cricket_team_members_l1075_107546


namespace domino_trick_l1075_107550

theorem domino_trick (x y : ℕ) (h1 : x ≤ 6) (h2 : y ≤ 6)
  (h3 : 10 * x + y + 30 = 62) : x = 3 ∧ y = 2 :=
by
  sorry

end domino_trick_l1075_107550


namespace problem1_div_expr_problem2_div_expr_l1075_107563

-- Problem 1
theorem problem1_div_expr : (1 / 30) / ((2 / 3) - (1 / 10) + (1 / 6) - (2 / 5)) = 1 / 10 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

-- Problem 2
theorem problem2_div_expr : (-1 / 20) / (-(1 / 4) - (2 / 5) + (9 / 10) - (3 / 2)) = 1 / 25 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

end problem1_div_expr_problem2_div_expr_l1075_107563


namespace friends_meeting_distance_l1075_107581

theorem friends_meeting_distance (R_q : ℝ) (t : ℝ) (D_p D_q trail_length : ℝ) :
  trail_length = 36 ∧ D_p = 1.25 * R_q * t ∧ D_q = R_q * t ∧ D_p + D_q = trail_length → D_p = 20 := by
  sorry

end friends_meeting_distance_l1075_107581


namespace dubblefud_red_balls_zero_l1075_107594

theorem dubblefud_red_balls_zero
  (R B G : ℕ)
  (H1 : 2^R * 4^B * 5^G = 16000)
  (H2 : B = G) : R = 0 :=
sorry

end dubblefud_red_balls_zero_l1075_107594


namespace smallest_n_satisfies_conditions_l1075_107572

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l1075_107572


namespace unique_solution_m_l1075_107513

theorem unique_solution_m (m : ℝ) :
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end unique_solution_m_l1075_107513


namespace burger_meal_cost_l1075_107524

-- Define the conditions
variables (B S : ℝ)
axiom cost_of_soda : S = (1 / 3) * B
axiom total_cost : B + S + 2 * (B + S) = 24

-- Prove that the cost of the burger meal is $6
theorem burger_meal_cost : B = 6 :=
by {
  -- We'll use both the axioms provided to show B equals 6
  sorry
}

end burger_meal_cost_l1075_107524


namespace Will_worked_on_Tuesday_l1075_107562

variable (HourlyWage MondayHours TotalEarnings : ℝ)

-- Given conditions
def Wage : ℝ := 8
def Monday_worked_hours : ℝ := 8
def Total_two_days_earnings : ℝ := 80

theorem Will_worked_on_Tuesday (HourlyWage_eq : HourlyWage = Wage)
  (MondayHours_eq : MondayHours = Monday_worked_hours)
  (TotalEarnings_eq : TotalEarnings = Total_two_days_earnings) :
  let MondayEarnings := MondayHours * HourlyWage
  let TuesdayEarnings := TotalEarnings - MondayEarnings
  let TuesdayHours := TuesdayEarnings / HourlyWage
  TuesdayHours = 2 :=
by
  sorry

end Will_worked_on_Tuesday_l1075_107562


namespace union_of_A_and_B_l1075_107573

section
variable {A B : Set ℝ}
variable (a b : ℝ)

def setA := {x : ℝ | x^2 - 3 * x + a = 0}
def setB := {x : ℝ | x^2 + b = 0}

theorem union_of_A_and_B:
  setA a ∩ setB b = {2} →
  setA a ∪ setB b = ({-2, 1, 2} : Set ℝ) := by
  sorry
end

end union_of_A_and_B_l1075_107573


namespace new_ratio_cooks_waiters_l1075_107534

theorem new_ratio_cooks_waiters
  (initial_ratio : ℕ → ℕ → Prop)
  (cooks waiters : ℕ) :
  initial_ratio 9 24 → 
  12 + waiters = 36 →
  initial_ratio 3 8 →
  9 * 4 = 36 :=
by
  intros h1 h2 h3
  sorry

end new_ratio_cooks_waiters_l1075_107534


namespace jack_needs_more_money_l1075_107547

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l1075_107547


namespace solve_ab_sum_l1075_107503

theorem solve_ab_sum (x a b : ℝ) (ha : ℕ) (hb : ℕ)
  (h1 : a = ha)
  (h2 : b = hb)
  (h3 : x = a + Real.sqrt b)
  (h4 : x^2 + 3 * x + 3 / x + 1 / x^2 = 26) :
  (ha + hb = 5) :=
sorry

end solve_ab_sum_l1075_107503


namespace abc_inequality_l1075_107597

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) :=
sorry

end abc_inequality_l1075_107597


namespace james_pays_per_episode_l1075_107533

-- Conditions
def minor_characters : ℕ := 4
def major_characters : ℕ := 5
def pay_per_minor_character : ℕ := 15000
def multiplier_major_payment : ℕ := 3

-- Theorems and Definitions needed
def pay_per_major_character : ℕ := pay_per_minor_character * multiplier_major_payment
def total_pay_minor : ℕ := minor_characters * pay_per_minor_character
def total_pay_major : ℕ := major_characters * pay_per_major_character
def total_pay_per_episode : ℕ := total_pay_minor + total_pay_major

-- Main statement to prove
theorem james_pays_per_episode : total_pay_per_episode = 285000 := by
  sorry

end james_pays_per_episode_l1075_107533


namespace sum_of_digits_of_7_pow_1974_l1075_107500

-- Define the number \(7^{1974}\)
def num := 7^1974

-- Function to extract the last two digits
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Function to compute the sum of the tens and units digits
def sum_tens_units (n : ℕ) : ℕ :=
  let last_two := last_two_digits n
  (last_two / 10) + (last_two % 10)

theorem sum_of_digits_of_7_pow_1974 : sum_tens_units num = 9 := by
  sorry

end sum_of_digits_of_7_pow_1974_l1075_107500


namespace height_of_boxes_l1075_107544

-- Conditions
def total_volume : ℝ := 1.08 * 10^6
def cost_per_box : ℝ := 0.2
def total_monthly_cost : ℝ := 120

-- Target height of the boxes
def target_height : ℝ := 12.2

-- Problem: Prove that the height of each box is 12.2 inches
theorem height_of_boxes : 
  (total_monthly_cost / cost_per_box) * ((total_volume / (total_monthly_cost / cost_per_box))^(1/3)) = target_height := 
sorry

end height_of_boxes_l1075_107544


namespace domain_lg_tan_minus_sqrt3_l1075_107583

open Real

theorem domain_lg_tan_minus_sqrt3 :
  {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} =
    {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} :=
by
  sorry

end domain_lg_tan_minus_sqrt3_l1075_107583


namespace find_genuine_coin_in_three_weighings_l1075_107519

theorem find_genuine_coin_in_three_weighings (coins : Fin 15 → ℝ)
  (even_number_of_counterfeit : ∃ n : ℕ, 2 * n < 15 ∧ (∀ i, coins i = 1) ∨ (∃ j, coins j = 0.5)) : 
  ∃ i, coins i = 1 :=
by sorry

end find_genuine_coin_in_three_weighings_l1075_107519


namespace max_books_borrowed_l1075_107576

noncomputable def max_books_per_student : ℕ := 14

theorem max_books_borrowed (students_borrowed_0 : ℕ)
                           (students_borrowed_1 : ℕ)
                           (students_borrowed_2 : ℕ)
                           (total_students : ℕ)
                           (average_books : ℕ)
                           (remaining_students_borrowed_at_least_3 : ℕ)
                           (total_books : ℕ)
                           (max_books : ℕ) 
  (h1 : students_borrowed_0 = 2)
  (h2 : students_borrowed_1 = 10)
  (h3 : students_borrowed_2 = 5)
  (h4 : total_students = 20)
  (h5 : average_books = 2)
  (h6 : remaining_students_borrowed_at_least_3 = total_students - students_borrowed_0 - students_borrowed_1 - students_borrowed_2)
  (h7 : total_books = total_students * average_books)
  (h8 : total_books = (students_borrowed_1 * 1 + students_borrowed_2 * 2) + remaining_students_borrowed_at_least_3 * 3 + (max_books - 6))
  (h_max : max_books = max_books_per_student) :
  max_books ≤ max_books_per_student := 
sorry

end max_books_borrowed_l1075_107576


namespace square_free_even_less_than_200_count_l1075_107529

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

theorem square_free_even_less_than_200_count : ∃ (count : ℕ), count = 38 ∧ (∀ n : ℕ, n < 200 ∧ is_multiple_of_2 n ∧ is_square_free n → count = 38) :=
by
  sorry

end square_free_even_less_than_200_count_l1075_107529


namespace num_ordered_triples_l1075_107522

-- Given constants
def b : ℕ := 2024
def constant_value : ℕ := 4096576

-- Number of ordered triples (a, b, c) meeting the conditions
theorem num_ordered_triples (h : b = 2024 ∧ constant_value = 2024 * 2024) :
  ∃ (n : ℕ), n = 10 ∧ ∀ (a c : ℕ), a * c = constant_value → a ≤ c → n = 10 :=
by
  -- Translation of the mathematical conditions into the theorem
  sorry

end num_ordered_triples_l1075_107522


namespace men_seated_count_l1075_107578

theorem men_seated_count (total_passengers : ℕ) (two_thirds_women : total_passengers * 2 / 3 = women)
                         (one_eighth_standing : total_passengers / 3 / 8 = standing_men) :
  total_passengers = 48 →
  women = 32 →
  standing_men = 2 →
  men_seated = (total_passengers - women) - standing_men →
  men_seated = 14 :=
by
  intros
  sorry

end men_seated_count_l1075_107578


namespace find_xyz_l1075_107595

theorem find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 198) (h5 : y * (z + x) = 216) (h6 : z * (x + y) = 234) :
  x * y * z = 1080 :=
sorry

end find_xyz_l1075_107595


namespace two_digit_integer_one_less_than_lcm_of_3_4_7_l1075_107548

theorem two_digit_integer_one_less_than_lcm_of_3_4_7 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n + 1) % (Nat.lcm (Nat.lcm 3 4) 7) = 0 ∧ n = 83 := by
  sorry

end two_digit_integer_one_less_than_lcm_of_3_4_7_l1075_107548


namespace plane_equation_rewriting_l1075_107506

theorem plane_equation_rewriting (A B C D x y z p q r : ℝ)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (eq1 : A * x + B * y + C * z + D = 0)
  (hp : p = -D / A) (hq : q = -D / B) (hr : r = -D / C) :
  x / p + y / q + z / r = 1 :=
by
  sorry

end plane_equation_rewriting_l1075_107506


namespace pigeonhole_divisible_l1075_107568

theorem pigeonhole_divisible (n : ℕ) (a : Fin (n + 1) → ℕ) (h : ∀ i, 1 ≤ a i ∧ a i ≤ 2 * n) :
  ∃ i j, i ≠ j ∧ a i ∣ a j :=
by
  sorry

end pigeonhole_divisible_l1075_107568


namespace multiples_of_5_with_units_digit_0_l1075_107521

theorem multiples_of_5_with_units_digit_0 (h1 : ∀ n : ℕ, n % 5 = 0 → (n % 10 = 0 ∨ n % 10 = 5))
  (h2 : ∀ m : ℕ, m < 200 → m % 5 = 0) :
  ∃ k : ℕ, k = 19 ∧ (∀ x : ℕ, (x < 200) ∧ (x % 5 = 0) → (x % 10 = 0) → k = (k - 1) + 1) := sorry

end multiples_of_5_with_units_digit_0_l1075_107521


namespace inequality_positive_numbers_l1075_107558

theorem inequality_positive_numbers (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := 
sorry

end inequality_positive_numbers_l1075_107558


namespace trajectory_of_circle_center_is_ellipse_l1075_107580

theorem trajectory_of_circle_center_is_ellipse 
    (a b : ℝ) (θ : ℝ) 
    (h1 : a ≠ b)
    (h2 : 0 < a)
    (h3 : 0 < b)
    : ∃ (x y : ℝ), 
    (x, y) = (a * Real.cos θ, b * Real.sin θ) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end trajectory_of_circle_center_is_ellipse_l1075_107580


namespace message_channels_encryption_l1075_107531

theorem message_channels_encryption :
  ∃ (assign_key : Fin 105 → Fin 105 → Fin 100),
  ∀ (u v w x : Fin 105), 
  u ≠ v → u ≠ w → u ≠ x → v ≠ w → v ≠ x → w ≠ x →
  (assign_key u v = assign_key u w ∧ assign_key u v = assign_key u x ∧ 
   assign_key u v = assign_key v w ∧ assign_key u v = assign_key v x ∧ 
   assign_key u v = assign_key w x) → False :=
by
  sorry

end message_channels_encryption_l1075_107531


namespace largest_and_smallest_A_l1075_107509

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end largest_and_smallest_A_l1075_107509


namespace price_reduction_l1075_107532

theorem price_reduction (x : ℝ) :
  (20 + 2 * x) * (40 - x) = 1200 → x = 20 :=
by
  sorry

end price_reduction_l1075_107532


namespace susan_total_distance_l1075_107587

theorem susan_total_distance (a b : ℕ) (r : ℝ) (h1 : a = 15) (h2 : b = 25) (h3 : r = 3) :
  (r * ((a + b) / 60)) = 2 :=
by
  sorry

end susan_total_distance_l1075_107587


namespace max_profit_at_300_l1075_107553

-- Define the cost and revenue functions and total profit function

noncomputable def cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 400 * x else 90090

noncomputable def profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 300 * x - 20000 else -100 * x + 70090

-- The Lean statement for proving maximum profit occurs at x = 300
theorem max_profit_at_300 : ∀ x : ℝ, profit x ≤ profit 300 :=
sorry

end max_profit_at_300_l1075_107553


namespace side_length_of_inscribed_square_l1075_107536

theorem side_length_of_inscribed_square
  (S1 S2 S3 : ℝ)
  (hS1 : S1 = 1) (hS2 : S2 = 3) (hS3 : S3 = 1) :
  ∃ (x : ℝ), S1 = 1 ∧ S2 = 3 ∧ S3 = 1 ∧ x = 2 := 
by
  sorry

end side_length_of_inscribed_square_l1075_107536


namespace slower_speed_l1075_107575

theorem slower_speed (x : ℝ) :
  (50 / x = 70 / 14) → x = 10 := by
  sorry

end slower_speed_l1075_107575


namespace chip_exits_from_A2_l1075_107545

noncomputable def chip_exit_cell (grid_size : ℕ) (initial_cell : ℕ × ℕ) (move_direction : ℕ × ℕ → ℕ × ℕ) : ℕ × ℕ :=
(1, 2) -- A2; we assume the implementation of function movement follows the solution as described

theorem chip_exits_from_A2 :
  chip_exit_cell 4 (3, 2) move_direction = (1, 2) :=
sorry  -- Proof omitted

end chip_exits_from_A2_l1075_107545


namespace cars_in_section_H_l1075_107526

theorem cars_in_section_H
  (rows_G : ℕ) (cars_per_row_G : ℕ) (rows_H : ℕ)
  (cars_per_minute : ℕ) (minutes_spent : ℕ)  
  (total_cars_walked_past : ℕ) :
  rows_G = 15 →
  cars_per_row_G = 10 →
  rows_H = 20 →
  cars_per_minute = 11 →
  minutes_spent = 30 →
  total_cars_walked_past = (rows_G * cars_per_row_G) + ((cars_per_minute * minutes_spent) - (rows_G * cars_per_row_G)) →
  (total_cars_walked_past - (rows_G * cars_per_row_G)) / rows_H = 9 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end cars_in_section_H_l1075_107526


namespace probability_of_distinct_dice_numbers_l1075_107502

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l1075_107502


namespace tangent_line_on_x_axis_l1075_107538

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1/4

theorem tangent_line_on_x_axis (x0 a : ℝ) (h1: f x0 a = 0) (h2: (3 * x0^2 + a) = 0) : a = -3/4 :=
by sorry

end tangent_line_on_x_axis_l1075_107538


namespace amount_spent_per_sibling_l1075_107598

-- Definitions and conditions
def total_spent := 150
def amount_per_parent := 30
def num_parents := 2
def num_siblings := 3

-- Claim
theorem amount_spent_per_sibling :
  (total_spent - (amount_per_parent * num_parents)) / num_siblings = 30 :=
by
  sorry

end amount_spent_per_sibling_l1075_107598


namespace num_digits_difference_l1075_107518

-- Define the two base-10 integers
def n1 : ℕ := 150
def n2 : ℕ := 950

-- Find the number of digits in the base-2 representation of these numbers.
def num_digits_base2 (n : ℕ) : ℕ :=
  Nat.log2 n + 1

-- State the theorem
theorem num_digits_difference :
  num_digits_base2 n2 - num_digits_base2 n1 = 2 :=
by
  sorry

end num_digits_difference_l1075_107518


namespace correct_sampling_methods_l1075_107570

-- Definitions for different sampling methods
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

-- Conditions from the problem
def situation1 (students_selected_per_class : Nat) : Prop :=
  students_selected_per_class = 2

def situation2 (students_above_110 : Nat) (students_between_90_and_100 : Nat) (students_below_90 : Nat) : Prop :=
  students_above_110 = 10 ∧ students_between_90_and_100 = 40 ∧ students_below_90 = 12

def situation3 (tracks_arranged_for_students : Nat) : Prop :=
  tracks_arranged_for_students = 6

-- Theorem
theorem correct_sampling_methods :
  ∀ (students_selected_per_class students_above_110 students_between_90_and_100 students_below_90 tracks_arranged_for_students: Nat),
  situation1 students_selected_per_class →
  situation2 students_above_110 students_between_90_and_100 students_below_90 →
  situation3 tracks_arranged_for_students →
  (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) = (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) :=
by
  intros
  rfl

end correct_sampling_methods_l1075_107570


namespace problem_statement_l1075_107569

-- Conditions
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a ^ x > 0
def q (x : ℝ) : Prop := x > 0 ∧ x ≠ 1 ∧ (Real.log 2 / Real.log x + Real.log x / Real.log 2 ≥ 2)

-- Theorem statement
theorem problem_statement (a x : ℝ) : ¬p a ∨ ¬q x :=
by sorry

end problem_statement_l1075_107569


namespace number_of_roots_eq_seven_l1075_107523

noncomputable def problem_function (x : ℝ) : ℝ :=
  (21 * x - 11 + (Real.sin x) / 100) * Real.sin (6 * Real.arcsin x) * Real.sqrt ((Real.pi - 6 * x) * (Real.pi + x))

theorem number_of_roots_eq_seven :
  (∃ xs : List ℝ, (∀ x ∈ xs, problem_function x = 0) ∧ (∀ x ∈ xs, -1 ≤ x ∧ x ≤ 1) ∧ xs.length = 7) :=
sorry

end number_of_roots_eq_seven_l1075_107523


namespace required_barrels_of_pitch_l1075_107508

def total_road_length : ℕ := 16
def bags_of_gravel_per_truckload : ℕ := 2
def barrels_of_pitch_per_truckload (bgt : ℕ) : ℚ := bgt / 5
def truckloads_per_mile : ℕ := 3

def miles_paved_day1 : ℕ := 4
def miles_paved_day2 : ℕ := (miles_paved_day1 * 2) - 1
def total_miles_paved_first_two_days : ℕ := miles_paved_day1 + miles_paved_day2
def remaining_miles_paved_day3 : ℕ := total_road_length - total_miles_paved_first_two_days

def truckloads_needed (miles : ℕ) : ℕ := miles * truckloads_per_mile
def barrels_of_pitch_needed (truckloads : ℕ) (bgt : ℕ) : ℚ := truckloads * barrels_of_pitch_per_truckload bgt

theorem required_barrels_of_pitch : 
  barrels_of_pitch_needed (truckloads_needed remaining_miles_paved_day3) bags_of_gravel_per_truckload = 6 := 
by
  sorry

end required_barrels_of_pitch_l1075_107508


namespace probability_first_genuine_on_third_test_l1075_107540

noncomputable def probability_of_genuine : ℚ := 3 / 4
noncomputable def probability_of_defective : ℚ := 1 / 4
noncomputable def probability_X_eq_3 := probability_of_defective * probability_of_defective * probability_of_genuine

theorem probability_first_genuine_on_third_test :
  probability_X_eq_3 = 3 / 64 :=
by
  sorry

end probability_first_genuine_on_third_test_l1075_107540


namespace equation_solution_unique_or_not_l1075_107565

theorem equation_solution_unique_or_not (a b : ℝ) :
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x - a) / (x - 2) + (x - b) / (x - 3) = 2) ↔ 
  (a = 2 ∧ b = 3) ∨ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
by
  sorry

end equation_solution_unique_or_not_l1075_107565


namespace bonus_percentage_is_correct_l1075_107515

theorem bonus_percentage_is_correct (kills total_points enemies_points bonus_threshold bonus_percentage : ℕ) 
  (h1 : enemies_points = 10) 
  (h2 : kills = 150) 
  (h3 : total_points = 2250) 
  (h4 : bonus_threshold = 100) 
  (h5 : kills >= bonus_threshold) 
  (h6 : bonus_percentage = (total_points - kills * enemies_points) * 100 / (kills * enemies_points)) : 
  bonus_percentage = 50 := 
by
  sorry

end bonus_percentage_is_correct_l1075_107515


namespace election_majority_l1075_107514

theorem election_majority
  (total_votes : ℕ)
  (winning_percent : ℝ)
  (other_percent : ℝ)
  (votes_cast : total_votes = 700)
  (winning_share : winning_percent = 0.84)
  (other_share : other_percent = 0.16) :
  ∃ majority : ℕ, majority = 476 := by
  sorry

end election_majority_l1075_107514


namespace eval_nabla_l1075_107566

namespace MathProblem

-- Definition of the operation
def nabla (a b : ℕ) : ℕ :=
  3 + b ^ a

-- Theorem statement
theorem eval_nabla : nabla (nabla 2 3) 4 = 16777219 :=
by
  sorry

end MathProblem

end eval_nabla_l1075_107566


namespace quartic_root_sum_l1075_107567

theorem quartic_root_sum (a n l : ℝ) (h : ∃ (r1 r2 r3 r4 : ℝ), 
  r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧ 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ 
  r1 + r2 + r3 + r4 = 10 ∧
  r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4 = a ∧
  r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 = n ∧
  r1 * r2 * r3 * r4 = l) : 
  a + n + l = 109 :=
sorry

end quartic_root_sum_l1075_107567


namespace log_diff_condition_l1075_107564

theorem log_diff_condition (a : ℕ → ℝ) (d e : ℝ) (H1 : ∀ n : ℕ, n > 1 → a n = Real.log n / Real.log 3003)
  (H2 : d = a 2 + a 3 + a 4 + a 5 + a 6) (H3 : e = a 15 + a 16 + a 17 + a 18 + a 19) :
  d - e = -Real.log 1938 / Real.log 3003 := by
  sorry

end log_diff_condition_l1075_107564


namespace milk_after_three_operations_l1075_107574

-- Define the initial amount of milk and the proportion replaced each step
def initial_milk : ℝ := 100
def proportion_replaced : ℝ := 0.2

-- Define the amount of milk after each replacement operation
noncomputable def milk_after_n_operations (n : ℕ) (milk : ℝ) : ℝ :=
  if n = 0 then milk
  else (1 - proportion_replaced) * milk_after_n_operations (n - 1) milk

-- Define the statement about the amount of milk after three operations
theorem milk_after_three_operations : milk_after_n_operations 3 initial_milk = 51.2 :=
by
  sorry

end milk_after_three_operations_l1075_107574


namespace slopes_product_of_tangents_l1075_107542

theorem slopes_product_of_tangents 
  (x₀ y₀ : ℝ) 
  (h_hyperbola : (2 * x₀^2) / 3 - y₀^2 / 6 = 1) 
  (h_outside_circle : x₀^2 + y₀^2 > 2) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ * k₂ = 4 ∧ 
    (y₀ - k₁ * x₀)^2 + k₁^2 = 2 ∧ 
    (y₀ - k₂ * x₀)^2 + k₂^2 = 2 :=
by {
  -- this proof will use the properties of tangents to a circle and the constraints given
  -- we don't need to implement it now, but we aim to show the correct relationship
  sorry
}

end slopes_product_of_tangents_l1075_107542


namespace find_f_2015_l1075_107590

variables (f : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) = f x + f 3

theorem find_f_2015 (h1 : is_even_function f) (h2 : satisfies_condition f) (h3 : f 1 = 2) : f 2015 = 2 :=
by
  sorry

end find_f_2015_l1075_107590


namespace first_method_of_exhaustion_l1075_107599

-- Define the names
inductive Names where
  | ZuChongzhi
  | LiuHui
  | ZhangHeng
  | YangHui
  deriving DecidableEq

-- Statement of the problem
def method_of_exhaustion_author : Names :=
  Names.LiuHui

-- Main theorem to state the result
theorem first_method_of_exhaustion : method_of_exhaustion_author = Names.LiuHui :=
by 
  sorry

end first_method_of_exhaustion_l1075_107599


namespace simplify_expression_l1075_107556

theorem simplify_expression :
  ( (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) ) / ( (2 * 3) * (3 * 4) * (4 * 5) * (5 * 6) ) = 1 / 5 :=
by
  sorry

end simplify_expression_l1075_107556


namespace time_to_cross_signal_post_l1075_107560

-- Definition of the conditions
def length_of_train : ℝ := 600  -- in meters
def time_to_cross_bridge : ℝ := 8  -- in minutes
def length_of_bridge : ℝ := 7200  -- in meters

-- Equivalent statement
theorem time_to_cross_signal_post (constant_speed : ℝ) (t : ℝ) 
  (h1 : constant_speed * t = length_of_train) 
  (h2 : constant_speed * time_to_cross_bridge = length_of_train + length_of_bridge) : 
  t * 60 = 36.9 := 
sorry

end time_to_cross_signal_post_l1075_107560


namespace acid_solution_l1075_107539

theorem acid_solution (n y : ℝ) (h : n > 30) (h1 : y = 15 * n / (n - 15)) :
  (n / 100) * n = ((n - 15) / 100) * (n + y) :=
by
  sorry

end acid_solution_l1075_107539


namespace find_point_A_coordinates_l1075_107592

theorem find_point_A_coordinates :
  ∃ (A : ℝ × ℝ), (A.2 = 0) ∧ 
  (dist A (-3, 2) = dist A (4, -5)) →
  A = (2, 0) :=
by
-- We'll provide the explicit exact proof later
-- Proof steps would go here
sorry 

end find_point_A_coordinates_l1075_107592


namespace Ray_wrote_35_l1075_107530

theorem Ray_wrote_35 :
  ∃ (x y : ℕ), (10 * x + y = 35) ∧ (10 * x + y = 4 * (x + y) + 3) ∧ (10 * x + y + 18 = 10 * y + x) :=
by
  sorry

end Ray_wrote_35_l1075_107530


namespace ball_color_arrangement_l1075_107520

-- Definitions for the conditions
variable (balls_in_red_box balls_in_white_box balls_in_yellow_box : Nat)
variable (red_balls white_balls yellow_balls : Nat)

-- Conditions as assumptions
axiom more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls
axiom different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls
axiom fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box

-- The main theorem to prove
theorem ball_color_arrangement
  (more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls)
  (different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls)
  (fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box) :
  (balls_in_red_box, balls_in_white_box, balls_in_yellow_box) = (yellow_balls, red_balls, white_balls) :=
sorry

end ball_color_arrangement_l1075_107520


namespace correct_choice_is_B_l1075_107588

def draw_ray := "Draw ray OP=3cm"
def connect_points := "Connect points A and B"
def draw_midpoint := "Draw the midpoint of points A and B"
def draw_distance := "Draw the distance between points A and B"

-- Mathematical function to identify the correct statement about drawing
def correct_drawing_statement (s : String) : Prop :=
  s = connect_points

theorem correct_choice_is_B :
  correct_drawing_statement connect_points :=
by
  sorry

end correct_choice_is_B_l1075_107588


namespace cosine_of_A_l1075_107551

theorem cosine_of_A (a b : ℝ) (A B : ℝ) (h1 : b = (5 / 8) * a) (h2 : A = 2 * B) :
  Real.cos A = 7 / 25 :=
by
  sorry

end cosine_of_A_l1075_107551


namespace mean_greater_than_median_by_two_l1075_107596

theorem mean_greater_than_median_by_two (x : ℕ) (h : x > 0) :
  ((x + (x + 2) + (x + 4) + (x + 7) + (x + 17)) / 5 - (x + 4)) = 2 :=
sorry

end mean_greater_than_median_by_two_l1075_107596


namespace domain_of_f_l1075_107504

noncomputable def f (x : ℝ) := 2 ^ (Real.sqrt (3 - x)) + 1 / (x - 1)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y = f x) ↔ (x ≤ 3 ∧ x ≠ 1) :=
by
  sorry

end domain_of_f_l1075_107504


namespace combined_percentage_of_students_preferring_tennis_is_39_l1075_107516

def total_students_north : ℕ := 1800
def percentage_tennis_north : ℚ := 25 / 100
def total_students_south : ℕ := 3000
def percentage_tennis_south : ℚ := 50 / 100
def total_students_valley : ℕ := 800
def percentage_tennis_valley : ℚ := 30 / 100

def students_prefer_tennis_north : ℚ := total_students_north * percentage_tennis_north
def students_prefer_tennis_south : ℚ := total_students_south * percentage_tennis_south
def students_prefer_tennis_valley : ℚ := total_students_valley * percentage_tennis_valley

def total_students : ℕ := total_students_north + total_students_south + total_students_valley
def total_students_prefer_tennis : ℚ := students_prefer_tennis_north + students_prefer_tennis_south + students_prefer_tennis_valley

def percentage_students_prefer_tennis : ℚ := (total_students_prefer_tennis / total_students) * 100

theorem combined_percentage_of_students_preferring_tennis_is_39 :
  percentage_students_prefer_tennis = 39 := by
  sorry

end combined_percentage_of_students_preferring_tennis_is_39_l1075_107516


namespace smallest_composite_no_prime_factors_less_than_20_l1075_107541

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l1075_107541


namespace calculate_expression_l1075_107552

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end calculate_expression_l1075_107552


namespace candidate_lost_by_2460_votes_l1075_107535

noncomputable def total_votes : ℝ := 8199.999999999998
noncomputable def candidate_percentage : ℝ := 0.35
noncomputable def rival_percentage : ℝ := 1 - candidate_percentage
noncomputable def candidate_votes := candidate_percentage * total_votes
noncomputable def rival_votes := rival_percentage * total_votes
noncomputable def votes_lost_by := rival_votes - candidate_votes

theorem candidate_lost_by_2460_votes : votes_lost_by = 2460 := by
  sorry

end candidate_lost_by_2460_votes_l1075_107535


namespace distance_between_city_centers_l1075_107517

theorem distance_between_city_centers :
  let distance_on_map_cm := 55
  let scale_cm_to_km := 30
  let km_to_m := 1000
  (distance_on_map_cm * scale_cm_to_km * km_to_m) = 1650000 :=
by
  sorry

end distance_between_city_centers_l1075_107517


namespace n_is_power_of_three_l1075_107507

theorem n_is_power_of_three {n : ℕ} (hn_pos : 0 < n) (p : Nat.Prime (4^n + 2^n + 1)) :
  ∃ (a : ℕ), n = 3^a :=
by
  sorry

end n_is_power_of_three_l1075_107507


namespace log_expression_evaluation_l1075_107593

theorem log_expression_evaluation (log2 log5 : ℝ) (h : log2 + log5 = 1) :
  log2 * (log5 + log10) + 2 * log5 - log5 * log20 = 1 := by
  sorry

end log_expression_evaluation_l1075_107593


namespace geometric_series_evaluation_l1075_107579

theorem geometric_series_evaluation (c d : ℝ) (h : (∑' n : ℕ, c / d^(n + 1)) = 3) :
  (∑' n : ℕ, c / (c + 2 * d)^(n + 1)) = (3 * d - 3) / (5 * d - 4) :=
sorry

end geometric_series_evaluation_l1075_107579


namespace bread_slices_remaining_l1075_107585

-- Conditions
def total_slices : ℕ := 12
def fraction_eaten_for_breakfast : ℕ := total_slices / 3
def slices_used_for_lunch : ℕ := 2

-- Mathematically Equivalent Proof Problem
theorem bread_slices_remaining : total_slices - fraction_eaten_for_breakfast - slices_used_for_lunch = 6 :=
by
  sorry

end bread_slices_remaining_l1075_107585


namespace probability_between_C_and_D_l1075_107561

theorem probability_between_C_and_D :
  ∀ (A B C D : ℝ) (AB AD BC : ℝ),
    AB = 3 * AD ∧ AB = 6 * BC ∧ D - A = AD ∧ C - A = AD + BC ∧ B - A = AB →
    (C < D) →
    ∃ p : ℝ, p = 1 / 2 := by
  sorry

end probability_between_C_and_D_l1075_107561


namespace correct_calculation_l1075_107577

theorem correct_calculation (a : ℝ) : a^3 / a^2 = a := by
  sorry

end correct_calculation_l1075_107577


namespace person_B_processes_components_l1075_107510

theorem person_B_processes_components (x : ℕ) (h1 : ∀ x, x > 0 → x + 2 > 0) 
(h2 : ∀ x, x > 0 → (25 / (x + 2)) = (20 / x)) :
  x = 8 := sorry

end person_B_processes_components_l1075_107510


namespace tenth_student_solved_six_l1075_107543

theorem tenth_student_solved_six : 
  ∀ (n : ℕ), 
    (∀ (i : ℕ) (j : ℕ), 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ n → (∀ k : ℕ, k ≤ n → ∃ s : ℕ, s = 7)) → 
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 9 → ∃ p : ℕ, p = 4) → ∃ m : ℕ, m = 6 := 
by
  sorry

end tenth_student_solved_six_l1075_107543


namespace triangle_BC_value_l1075_107527

theorem triangle_BC_value (B C A : ℝ) (AB AC BC : ℝ) 
  (hB : B = 45) 
  (hAB : AB = 100)
  (hAC : AC = 100)
  (h_deg : A ≠ 0) :
  BC = 100 * Real.sqrt 2 := 
by 
  sorry

end triangle_BC_value_l1075_107527


namespace frogs_seen_in_pond_l1075_107537

-- Definitions from the problem conditions
def initial_frogs_on_lily_pads : ℕ := 5
def frogs_on_logs : ℕ := 3
def baby_frogs_on_rock : ℕ := 2 * 12  -- Two dozen

-- The statement of the proof
theorem frogs_seen_in_pond : initial_frogs_on_lily_pads + frogs_on_logs + baby_frogs_on_rock = 32 :=
by sorry

end frogs_seen_in_pond_l1075_107537


namespace teal_more_blue_l1075_107589

def numSurveyed : ℕ := 150
def numGreen : ℕ := 90
def numBlue : ℕ := 50
def numBoth : ℕ := 40
def numNeither : ℕ := 20

theorem teal_more_blue : 40 + (numSurveyed - (numBoth + (numGreen - numBoth) + numNeither)) = 80 :=
by
  -- Here we simplify numerically until we get the required answer
  -- start with calculating the total accounted and remaining
  sorry

end teal_more_blue_l1075_107589


namespace circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l1075_107525

-- Circle 1 with center (8, -3) and passing through point (5, 1)
theorem circle_centered_at_8_neg3_passing_through_5_1 :
  ∃ r : ℝ, (r = 5) ∧ ((x - 8: ℝ)^2 + (y + 3)^2 = r^2) := by
  sorry

-- Circle passing through points A(-1, 5), B(5, 5), and C(6, -2)
theorem circle_passing_through_ABC :
  ∃ D E F : ℝ, (D = -4) ∧ (E = -2) ∧ (F = -20) ∧
    ( ∀ (x : ℝ) (y : ℝ), (x = -1 ∧ y = 5) 
      ∨ (x = 5 ∧ y = 5) 
      ∨ (x = 6 ∧ y = -2) 
      → (x^2 + y^2 + D*x + E*y + F = 0)) := by
  sorry

end circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l1075_107525


namespace parabola_vertex_l1075_107512

theorem parabola_vertex (x y : ℝ) : 
  y^2 + 10 * y + 3 * x + 9 = 0 → 
  (∃ v_x v_y, v_x = 16/3 ∧ v_y = -5 ∧ ∀ (y' : ℝ), (x, y) = (v_x, v_y) ↔ (x, y) = (-1 / 3 * ((y' + 5)^2 - 16), y')) :=
by
  sorry

end parabola_vertex_l1075_107512


namespace maria_total_distance_l1075_107582

-- Definitions
def total_distance (D : ℝ) : Prop :=
  let d1 := D/2   -- Distance traveled before first stop
  let r1 := D - d1 -- Distance remaining after first stop
  let d2 := r1/4  -- Distance traveled before second stop
  let r2 := r1 - d2 -- Distance remaining after second stop
  let d3 := r2/3  -- Distance traveled before third stop
  let r3 := r2 - d3 -- Distance remaining after third stop
  r3 = 270 -- Remaining distance after third stop equals 270 miles

-- Theorem statement
theorem maria_total_distance : ∃ D : ℝ, total_distance D ∧ D = 1080 :=
sorry

end maria_total_distance_l1075_107582


namespace p_sufficient_not_necessary_for_q_l1075_107584

open Real

def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

theorem p_sufficient_not_necessary_for_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l1075_107584


namespace smallest_solution_l1075_107511

noncomputable def equation (x : ℝ) := x^4 - 40 * x^2 + 400

theorem smallest_solution : ∃ x : ℝ, equation x = 0 ∧ ∀ y : ℝ, equation y = 0 → -2 * Real.sqrt 5 ≤ y :=
by
  sorry

end smallest_solution_l1075_107511


namespace regular_polygon_perimeter_l1075_107528

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l1075_107528


namespace maximize_f_l1075_107505

noncomputable def f (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximize_f (x : ℝ) (h : x < 5 / 4): ∃ M, (∀ y, (y < 5 / 4) → f y ≤ M) ∧ M = 1 := by
  sorry

end maximize_f_l1075_107505


namespace arun_weight_lower_limit_l1075_107571

theorem arun_weight_lower_limit :
  ∃ (w : ℝ), w > 60 ∧ w <= 64 ∧ (∀ (a : ℝ), 60 < a ∧ a <= 64 → ((a + 64) / 2 = 63) → a = 62) :=
by
  sorry

end arun_weight_lower_limit_l1075_107571


namespace scalene_triangle_minimum_altitude_l1075_107557

theorem scalene_triangle_minimum_altitude (a b c : ℕ) (h : ℕ) 
  (h₁ : a ≠ b ∧ b ≠ c ∧ c ≠ a) -- scalene condition
  (h₂ : ∃ k : ℕ, ∃ m : ℕ, k * m = a ∧ m = 6) -- first altitude condition
  (h₃ : ∃ k : ℕ, ∃ n : ℕ, k * n = b ∧ n = 8) -- second altitude condition
  (h₄ : c = (7 : ℕ) * b / (3 : ℕ)) -- third side condition given inequalities and area relations
  : h = 2 := 
sorry

end scalene_triangle_minimum_altitude_l1075_107557


namespace second_flower_shop_groups_l1075_107549

theorem second_flower_shop_groups (n : ℕ) (h1 : n ≠ 0) (h2 : n ≠ 9) (h3 : Nat.lcm 9 n = 171) : n = 19 := 
by
  sorry

end second_flower_shop_groups_l1075_107549


namespace total_worth_of_stock_l1075_107555

theorem total_worth_of_stock (x y : ℕ) (cheap_cost expensive_cost : ℝ) 
  (h1 : y = 21) (h2 : x + y = 22)
  (h3 : expensive_cost = 10) (h4 : cheap_cost = 2.5) :
  (x * expensive_cost + y * cheap_cost) = 62.5 :=
by
  sorry

end total_worth_of_stock_l1075_107555


namespace original_price_l1075_107559

theorem original_price (P : ℝ) (profit : ℝ) (profit_percentage : ℝ)
  (h1 : profit = 675) (h2 : profit_percentage = 0.35) :
  P = 1928.57 :=
by
  -- The proof is skipped using sorry
  sorry

end original_price_l1075_107559
