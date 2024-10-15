import Mathlib

namespace NUMINAMATH_GPT_find_number_l675_67594

theorem find_number (x : ℤ) (h : 3 * x - 6 = 2 * x) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l675_67594


namespace NUMINAMATH_GPT_rainfall_thursday_l675_67558

theorem rainfall_thursday : 
  let monday_rain := 0.9
  let tuesday_rain := monday_rain - 0.7
  let wednesday_rain := tuesday_rain * 1.5
  let thursday_rain := wednesday_rain * 0.8
  thursday_rain = 0.24 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_thursday_l675_67558


namespace NUMINAMATH_GPT_savings_calculation_l675_67512

theorem savings_calculation (x : ℕ) (h1 : 15 * x = 15000) : (15000 - 8 * x = 7000) :=
sorry

end NUMINAMATH_GPT_savings_calculation_l675_67512


namespace NUMINAMATH_GPT_abs_diff_eq_0_5_l675_67565

noncomputable def x : ℝ := 3.7
noncomputable def y : ℝ := 4.2

theorem abs_diff_eq_0_5 (hx : ⌊x⌋ + (y - ⌊y⌋) = 3.2) (hy : (x - ⌊x⌋) + ⌊y⌋ = 4.7) :
  |x - y| = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_eq_0_5_l675_67565


namespace NUMINAMATH_GPT_smallest_product_of_non_factors_l675_67542

theorem smallest_product_of_non_factors (a b : ℕ) (h_a : a ∣ 48) (h_b : b ∣ 48) (h_distinct : a ≠ b) (h_prod_non_factor : ¬ (a * b ∣ 48)) : a * b = 18 :=
sorry

end NUMINAMATH_GPT_smallest_product_of_non_factors_l675_67542


namespace NUMINAMATH_GPT_team_c_score_l675_67508

theorem team_c_score (points_A points_B total_points : ℕ) (hA : points_A = 2) (hB : points_B = 9) (hTotal : total_points = 15) :
  total_points - (points_A + points_B) = 4 :=
by
  sorry

end NUMINAMATH_GPT_team_c_score_l675_67508


namespace NUMINAMATH_GPT_inequality_direction_change_l675_67519

theorem inequality_direction_change :
  ∃ (a b c : ℝ), (a < b) ∧ (c < 0) ∧ (a * c > b * c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_direction_change_l675_67519


namespace NUMINAMATH_GPT_total_people_served_l675_67544

variable (total_people : ℕ)
variable (people_not_buy_coffee : ℕ := 10)

theorem total_people_served (H : (2 / 5 : ℚ) * total_people = people_not_buy_coffee) : total_people = 25 := 
by
  sorry

end NUMINAMATH_GPT_total_people_served_l675_67544


namespace NUMINAMATH_GPT_eval_polynomial_at_2_l675_67511

theorem eval_polynomial_at_2 : 
  ∃ a b c d : ℝ, (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 18) :=
by
  sorry

end NUMINAMATH_GPT_eval_polynomial_at_2_l675_67511


namespace NUMINAMATH_GPT_product_xyz_l675_67554

theorem product_xyz {x y z a b c : ℝ} 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = b^2) 
  (h3 : x^3 + y^3 + z^3 = c^3) : 
  x * y * z = (a^3 - 3 * a * b^2 + 2 * c^3) / 6 :=
by
  sorry

end NUMINAMATH_GPT_product_xyz_l675_67554


namespace NUMINAMATH_GPT_solve_system_of_equations_l675_67596

theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x - 2 * y = 1) (h2 : x + y = 2) : x^2 - 2 * y^2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l675_67596


namespace NUMINAMATH_GPT_route_B_is_faster_by_7_5_minutes_l675_67525

def distance_A := 10  -- miles
def normal_speed_A := 30  -- mph
def construction_distance_A := 2  -- miles
def construction_speed_A := 15  -- mph
def distance_B := 8  -- miles
def normal_speed_B := 40  -- mph
def school_zone_distance_B := 1  -- miles
def school_zone_speed_B := 10  -- mph

noncomputable def time_for_normal_speed_A : ℝ := (distance_A - construction_distance_A) / normal_speed_A * 60  -- minutes
noncomputable def time_for_construction_A : ℝ := construction_distance_A / construction_speed_A * 60  -- minutes
noncomputable def total_time_A : ℝ := time_for_normal_speed_A + time_for_construction_A

noncomputable def time_for_normal_speed_B : ℝ := (distance_B - school_zone_distance_B) / normal_speed_B * 60  -- minutes
noncomputable def time_for_school_zone_B : ℝ := school_zone_distance_B / school_zone_speed_B * 60  -- minutes
noncomputable def total_time_B : ℝ := time_for_normal_speed_B + time_for_school_zone_B

theorem route_B_is_faster_by_7_5_minutes : total_time_B + 7.5 = total_time_A := by
  sorry

end NUMINAMATH_GPT_route_B_is_faster_by_7_5_minutes_l675_67525


namespace NUMINAMATH_GPT_tim_kittens_count_l675_67531

def initial_kittens : Nat := 6
def kittens_given_to_jessica : Nat := 3
def kittens_received_from_sara : Nat := 9

theorem tim_kittens_count : initial_kittens - kittens_given_to_jessica + kittens_received_from_sara = 12 :=
by
  sorry

end NUMINAMATH_GPT_tim_kittens_count_l675_67531


namespace NUMINAMATH_GPT_inequality_problem_l675_67523

variable {a b : ℕ}

theorem inequality_problem (a : ℕ) (b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq_1_a : a ≠ 1) (h_neq_1_b : b ≠ 1) :
  ((a^5 - 1:ℚ) / (a^4 - 1)) * ((b^5 - 1) / (b^4 - 1)) > (25 / 64 : ℚ) * (a + 1) * (b + 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l675_67523


namespace NUMINAMATH_GPT_expense_of_three_yuan_l675_67582

def isIncome (x : Int) : Prop := x > 0
def isExpense (x : Int) : Prop := x < 0
def incomeOfTwoYuan : Int := 2

theorem expense_of_three_yuan : isExpense (-3) :=
by
  -- Assuming the conditions:
  -- Income is positive: isIncome incomeOfTwoYuan (which is 2)
  -- Expenses are negative
  -- Expenses of 3 yuan should be denoted as -3 yuan
  sorry

end NUMINAMATH_GPT_expense_of_three_yuan_l675_67582


namespace NUMINAMATH_GPT_solution_set_l675_67546

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set :
  {x | f x ≥ x^2 - 8 * x + 15} = {2} ∪ {x | x > 6} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l675_67546


namespace NUMINAMATH_GPT_john_total_amount_l675_67571

def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount
def aunt_amount : ℕ := 3 / 2 * grandpa_amount
def uncle_amount : ℕ := 2 / 3 * grandma_amount

def total_amount : ℕ :=
  grandpa_amount + grandma_amount + aunt_amount + uncle_amount

theorem john_total_amount : total_amount = 225 := by sorry

end NUMINAMATH_GPT_john_total_amount_l675_67571


namespace NUMINAMATH_GPT_sqrt_529000_pow_2_5_l675_67577

theorem sqrt_529000_pow_2_5 : (529000 ^ (1 / 2) ^ (5 / 2)) = 14873193 := by
  sorry

end NUMINAMATH_GPT_sqrt_529000_pow_2_5_l675_67577


namespace NUMINAMATH_GPT_quadratic_real_roots_l675_67568

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x1 x2 : ℝ, k * x1^2 - 6 * x1 - 1 = 0 ∧ k * x2^2 - 6 * x2 - 1 = 0 ∧ x1 ≠ x2) ↔ k ≥ -9 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l675_67568


namespace NUMINAMATH_GPT_positive_integers_divisible_by_4_5_and_6_less_than_300_l675_67510

open Nat

theorem positive_integers_divisible_by_4_5_and_6_less_than_300 : 
    ∃ n : ℕ, n = 5 ∧ ∀ m, m < 300 → (m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0) → (m % 60 = 0) :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_divisible_by_4_5_and_6_less_than_300_l675_67510


namespace NUMINAMATH_GPT_smallest_altitude_le_3_l675_67521

theorem smallest_altitude_le_3 (a b c h_a h_b h_c : ℝ) (r : ℝ) (h_r : r = 1)
    (h_a_ge_b : a ≥ b) (h_b_ge_c : b ≥ c) 
    (area_eq1 : (a + b + c) / 2 * r = (a * h_a) / 2) 
    (area_eq2 : (a + b + c) / 2 * r = (b * h_b) / 2) 
    (area_eq3 : (a + b + c) / 2 * r = (c * h_c) / 2) : 
    min h_a (min h_b h_c) ≤ 3 := 
by
  sorry

end NUMINAMATH_GPT_smallest_altitude_le_3_l675_67521


namespace NUMINAMATH_GPT_isosceles_right_triangle_leg_hypotenuse_ratio_l675_67504

theorem isosceles_right_triangle_leg_hypotenuse_ratio (a d k : ℝ) 
  (h_iso : d = a * Real.sqrt 2)
  (h_ratio : k = a / d) : 
  k^2 = 1 / 2 := by sorry

end NUMINAMATH_GPT_isosceles_right_triangle_leg_hypotenuse_ratio_l675_67504


namespace NUMINAMATH_GPT_function_graph_second_quadrant_l675_67576

theorem function_graph_second_quadrant (b : ℝ) (h : ∀ x, 2 ^ x + b - 1 ≥ 0): b ≤ 0 :=
sorry

end NUMINAMATH_GPT_function_graph_second_quadrant_l675_67576


namespace NUMINAMATH_GPT_Jason_spent_correct_amount_l675_67543

def flute_cost : ℝ := 142.46
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7.00
def total_cost : ℝ := 158.35

theorem Jason_spent_correct_amount :
  flute_cost + music_stand_cost + song_book_cost = total_cost :=
by
  sorry

end NUMINAMATH_GPT_Jason_spent_correct_amount_l675_67543


namespace NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l675_67545

theorem line_does_not_pass_through_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l675_67545


namespace NUMINAMATH_GPT_initial_number_of_balls_l675_67527

theorem initial_number_of_balls (T B : ℕ) (P : ℚ) (after3_blue : ℕ) (prob : ℚ) :
  B = 7 → after3_blue = B - 3 → prob = after3_blue / T → prob = 1/3 → T = 15 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_balls_l675_67527


namespace NUMINAMATH_GPT_find_principal_amount_l675_67537

theorem find_principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (h1 : SI = 4034.25)
  (h2 : R = 9)
  (h3 : T = 5) :
  P = 8965 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_amount_l675_67537


namespace NUMINAMATH_GPT_first_pipe_fill_time_l675_67599

theorem first_pipe_fill_time 
  (T : ℝ)
  (h1 : 48 * (1 / T - 1 / 24) + 18 * (1 / T) = 1) :
  T = 22 :=
by
  sorry

end NUMINAMATH_GPT_first_pipe_fill_time_l675_67599


namespace NUMINAMATH_GPT_range_of_m_l675_67549

noncomputable def problem_statement
  (x y m : ℝ) : Prop :=
  (x - 2 * y + 5 ≥ 0) ∧
  (3 - x ≥ 0) ∧
  (x + y ≥ 0) ∧
  (m > 0)

theorem range_of_m (x y m : ℝ) :
  problem_statement x y m →
  ((∀ x y, problem_statement x y m → x^2 + y^2 ≤ m^2) ↔ m ≥ 3 * Real.sqrt 2) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l675_67549


namespace NUMINAMATH_GPT_parabola_ord_l675_67503

theorem parabola_ord {M : ℝ × ℝ} (h1 : M.1 = (M.2 * M.2) / 8) (h2 : dist M (2, 0) = 4) : M.2 = 4 ∨ M.2 = -4 := 
sorry

end NUMINAMATH_GPT_parabola_ord_l675_67503


namespace NUMINAMATH_GPT_factorial_equation_solution_unique_l675_67535

theorem factorial_equation_solution_unique :
  ∀ a b c : ℕ, (0 < a ∧ 0 < b ∧ 0 < c) →
  (a.factorial * b.factorial = a.factorial + b.factorial + c.factorial) →
  (a = 3 ∧ b = 3 ∧ c = 4) := 
by
  intros a b c h_positive h_eq
  sorry

end NUMINAMATH_GPT_factorial_equation_solution_unique_l675_67535


namespace NUMINAMATH_GPT_number_of_correct_answers_is_95_l675_67561

variable (x y : ℕ) -- Define x as the number of correct answers and y as the number of wrong answers

-- Define the conditions
axiom h1 : x + y = 150
axiom h2 : 5 * x - 2 * y = 370

-- State the goal we want to prove
theorem number_of_correct_answers_is_95 : x = 95 :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_answers_is_95_l675_67561


namespace NUMINAMATH_GPT_ratio_of_adults_to_children_l675_67591

-- Definitions based on conditions
def adult_ticket_price : ℝ := 5.50
def child_ticket_price : ℝ := 2.50
def total_receipts : ℝ := 1026
def number_of_adults : ℝ := 152

-- Main theorem to prove ratio of adults to children is 2:1
theorem ratio_of_adults_to_children : 
  ∃ (number_of_children : ℝ), adult_ticket_price * number_of_adults + child_ticket_price * number_of_children = total_receipts ∧ 
  number_of_adults / number_of_children = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_adults_to_children_l675_67591


namespace NUMINAMATH_GPT_division_remainder_correct_l675_67567

theorem division_remainder_correct :
  ∃ q r, 987670 = 128 * q + r ∧ 0 ≤ r ∧ r < 128 ∧ r = 22 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_correct_l675_67567


namespace NUMINAMATH_GPT_slices_per_pack_l675_67548

theorem slices_per_pack (sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_of_bread : ℕ) (total_slices : ℕ) 
  (h1 : sandwiches = 8) (h2 : slices_per_sandwich = 2) (h3 : packs_of_bread = 4) : 
  total_slices = 4 :=
by
  sorry

end NUMINAMATH_GPT_slices_per_pack_l675_67548


namespace NUMINAMATH_GPT_roots_of_quadratic_l675_67513

theorem roots_of_quadratic (p q : ℝ) (h1 : 3 * p^2 + 9 * p - 21 = 0) (h2 : 3 * q^2 + 9 * q - 21 = 0) :
  (3 * p - 4) * (6 * q - 8) = 122 := by
  -- We don't need to provide the proof here, only the statement
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l675_67513


namespace NUMINAMATH_GPT_sequence_problem_l675_67560

/-- Given sequence a_n with specific values for a_2 and a_4 and the assumption that a_(n+1)
    is a geometric sequence, prove that a_6 equals 63. -/
theorem sequence_problem 
  {a : ℕ → ℝ} 
  (h1 : a 2 = 3) 
  (h2 : a 4 = 15) 
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n ∧ q^2 = 4) : 
  a 6 = 63 := by
  sorry

end NUMINAMATH_GPT_sequence_problem_l675_67560


namespace NUMINAMATH_GPT_angle_is_20_l675_67564

theorem angle_is_20 (x : ℝ) (h : 180 - x = 2 * (90 - x) + 20) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_angle_is_20_l675_67564


namespace NUMINAMATH_GPT_minimum_value_of_expression_l675_67550

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z))

theorem minimum_value_of_expression : ∀ (x y z : ℝ), -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ -1 < z ∧ z < 0 → 
  minimum_value_expression x y z ≥ 2 := 
by
  intro x y z h
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l675_67550


namespace NUMINAMATH_GPT_charlotte_age_l675_67536

theorem charlotte_age : 
  ∀ (B C E : ℝ), 
    (B = 4 * C) → 
    (E = C + 5) → 
    (B = E) → 
    C = 5 / 3 :=
by
  intros B C E h1 h2 h3
  /- start of the proof -/
  sorry

end NUMINAMATH_GPT_charlotte_age_l675_67536


namespace NUMINAMATH_GPT_take_home_pay_correct_l675_67579

noncomputable def faith_take_home_pay : Float :=
  let regular_hourly_rate := 13.50
  let regular_hours_per_day := 8
  let days_per_week := 5
  let regular_hours_per_week := regular_hours_per_day * days_per_week
  let regular_earnings_per_week := regular_hours_per_week * regular_hourly_rate

  let overtime_rate_multiplier := 1.5
  let overtime_hourly_rate := regular_hourly_rate * overtime_rate_multiplier
  let overtime_hours_per_day := 2
  let overtime_hours_per_week := overtime_hours_per_day * days_per_week
  let overtime_earnings_per_week := overtime_hours_per_week * overtime_hourly_rate

  let total_sales := 3200.0
  let commission_rate := 0.10
  let commission := total_sales * commission_rate

  let total_earnings_before_deductions := regular_earnings_per_week + overtime_earnings_per_week + commission

  let deduction_rate := 0.25
  let amount_withheld := total_earnings_before_deductions * deduction_rate
  let amount_withheld_rounded := (amount_withheld * 100).round / 100

  let take_home_pay := total_earnings_before_deductions - amount_withheld_rounded
  take_home_pay

theorem take_home_pay_correct : faith_take_home_pay = 796.87 :=
by
  /- Proof omitted -/
  sorry

end NUMINAMATH_GPT_take_home_pay_correct_l675_67579


namespace NUMINAMATH_GPT_rhombus_area_from_roots_l675_67592

-- Definition of the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 10 * x + 24 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b

-- Final mathematical statement to prove
theorem rhombus_area_from_roots (a b : ℝ) (h : roots a b) :
  a * b = 24 → (1 / 2) * a * b = 12 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_area_from_roots_l675_67592


namespace NUMINAMATH_GPT_range_of_a_l675_67590

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l675_67590


namespace NUMINAMATH_GPT_gaussian_solutions_count_l675_67575

noncomputable def solve_gaussian (x : ℝ) : ℕ :=
  if h : x^2 = 2 * (⌊x⌋ : ℝ) + 1 then 
    1 
  else
    0

theorem gaussian_solutions_count :
  ∀ x : ℝ, solve_gaussian x = 2 :=
sorry

end NUMINAMATH_GPT_gaussian_solutions_count_l675_67575


namespace NUMINAMATH_GPT_machine_transportation_l675_67574

theorem machine_transportation (x y : ℕ) 
  (h1 : x + 6 - y = 10) 
  (h2 : 400 * x + 800 * (20 - x) + 300 * (6 - y) + 500 * y = 16000) : 
  x = 5 ∧ y = 1 := 
sorry

end NUMINAMATH_GPT_machine_transportation_l675_67574


namespace NUMINAMATH_GPT_product_of_three_consecutive_integers_divisible_by_six_l675_67518

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end NUMINAMATH_GPT_product_of_three_consecutive_integers_divisible_by_six_l675_67518


namespace NUMINAMATH_GPT_ratio_of_shares_l675_67593

-- Definitions
variable (A B C : ℝ)   -- Representing the shares of a, b, and c
variable (x : ℝ)       -- Fraction

-- Conditions
axiom h1 : A = 80
axiom h2 : A + B + C = 200
axiom h3 : A = x * (B + C)
axiom h4 : B = (6 / 9) * (A + C)

-- Statement to prove
theorem ratio_of_shares : A / (B + C) = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_shares_l675_67593


namespace NUMINAMATH_GPT_coffee_remaining_after_shrink_l675_67500

-- Definitions of conditions in the problem
def shrink_factor : ℝ := 0.5
def cups_before_shrink : ℕ := 5
def ounces_per_cup_before_shrink : ℝ := 8

-- Definition of the total ounces of coffee remaining after shrinking
def ounces_per_cup_after_shrink : ℝ := ounces_per_cup_before_shrink * shrink_factor
def total_ounces_after_shrink : ℝ := cups_before_shrink * ounces_per_cup_after_shrink

-- The proof statement
theorem coffee_remaining_after_shrink :
  total_ounces_after_shrink = 20 :=
by
  -- Omitting the proof as only the statement is needed
  sorry

end NUMINAMATH_GPT_coffee_remaining_after_shrink_l675_67500


namespace NUMINAMATH_GPT_max_value_of_expr_l675_67562

-- Define the initial conditions and expression 
def initial_ones (n : ℕ) := List.replicate n 1

-- Given that we place "+" or ")(" between consecutive ones
def max_possible_value (n : ℕ) : ℕ := sorry

theorem max_value_of_expr : max_possible_value 2013 = 3 ^ 671 := 
sorry

end NUMINAMATH_GPT_max_value_of_expr_l675_67562


namespace NUMINAMATH_GPT_intersection_A_B_l675_67580

-- Definition of set A based on the given inequality
def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

-- Definition of set B
def B : Set ℝ := {-3, -1, 1, 3}

-- Prove the intersection A ∩ B equals the expected set {-1, 1, 3}
theorem intersection_A_B : A ∩ B = {-1, 1, 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l675_67580


namespace NUMINAMATH_GPT_sequence_arithmetic_mean_l675_67538

theorem sequence_arithmetic_mean (a b c d e f g : ℝ)
  (h1 : b = (a + c) / 2)
  (h2 : c = (b + d) / 2)
  (h3 : d = (c + e) / 2)
  (h4 : e = (d + f) / 2)
  (h5 : f = (e + g) / 2) :
  d = (a + g) / 2 :=
sorry

end NUMINAMATH_GPT_sequence_arithmetic_mean_l675_67538


namespace NUMINAMATH_GPT_new_average_after_doubling_l675_67595

theorem new_average_after_doubling
  (avg : ℝ) (num_students : ℕ) (h_avg : avg = 40) (h_num_students : num_students = 10) :
  let total_marks := avg * num_students
  let new_total_marks := total_marks * 2
  let new_avg := new_total_marks / num_students
  new_avg = 80 :=
by
  sorry

end NUMINAMATH_GPT_new_average_after_doubling_l675_67595


namespace NUMINAMATH_GPT_imaginary_unit_multiplication_l675_67569

theorem imaginary_unit_multiplication (i : ℂ) (h1 : i * i = -1) : i * (1 + i) = i - 1 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_unit_multiplication_l675_67569


namespace NUMINAMATH_GPT_diagonals_in_nine_sided_polygon_l675_67588

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end NUMINAMATH_GPT_diagonals_in_nine_sided_polygon_l675_67588


namespace NUMINAMATH_GPT_max_ratio_three_digit_sum_l675_67597

theorem max_ratio_three_digit_sum (N a b c : ℕ) (hN : N = 100 * a + 10 * b + c) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) :
  (∀ (N' a' b' c' : ℕ), N' = 100 * a' + 10 * b' + c' → 1 ≤ a' → b' ≤ 9 → c' ≤ 9 → (N' : ℚ) / (a' + b' + c') ≤ 100) :=
sorry

end NUMINAMATH_GPT_max_ratio_three_digit_sum_l675_67597


namespace NUMINAMATH_GPT_find_a_range_l675_67541

def f (a x : ℝ) : ℝ := x^2 + a * x

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, f a (f a x) ≤ f a x) → (a ≤ 0 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_range_l675_67541


namespace NUMINAMATH_GPT_average_annual_growth_rate_l675_67501

-- Definitions of the provided conditions
def initial_amount : ℝ := 200
def final_amount : ℝ := 338
def periods : ℝ := 2

-- Statement of the goal
theorem average_annual_growth_rate :
  (final_amount / initial_amount)^(1 / periods) - 1 = 0.3 := 
sorry

end NUMINAMATH_GPT_average_annual_growth_rate_l675_67501


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l675_67502

-- Defining each problem as a theorem statement
theorem problem1 : 20 + 3 - (-27) + (-5) = 45 :=
by sorry

theorem problem2 : (-7) - (-6 + 5 / 6) + abs (-3) + 1 + 1 / 6 = 4 :=
by sorry

theorem problem3 : (1 / 4 + 3 / 8 - 7 / 12) / (1 / 24) = 1 :=
by sorry

theorem problem4 : -1 ^ 4 - (1 - 0.4) + 1 / 3 * ((-2) ^ 2 - 6) = -2 - 4 / 15 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l675_67502


namespace NUMINAMATH_GPT_total_pounds_of_food_l675_67598

-- Conditions
def chicken := 16
def hamburgers := chicken / 2
def hot_dogs := hamburgers + 2
def sides := hot_dogs / 2

-- Define the total pounds of food
def total_food := chicken + hamburgers + hot_dogs + sides

-- Theorem statement that corresponds to the problem, showing the final result
theorem total_pounds_of_food : total_food = 39 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_pounds_of_food_l675_67598


namespace NUMINAMATH_GPT_initial_volume_shampoo_l675_67547

theorem initial_volume_shampoo (V : ℝ) 
  (replace_rate : ℝ)
  (use_rate : ℝ)
  (t : ℝ) 
  (hot_sauce_fraction : ℝ) 
  (hot_sauce_amount : ℝ) : 
  replace_rate = 1/2 → 
  use_rate = 1 → 
  t = 4 → 
  hot_sauce_fraction = 0.25 → 
  hot_sauce_amount = t * replace_rate → 
  hot_sauce_amount = hot_sauce_fraction * V → 
  V = 8 :=
by 
  intro h_replace_rate h_use_rate h_t h_hot_sauce_fraction h_hot_sauce_amount h_hot_sauce_amount_eq
  sorry

end NUMINAMATH_GPT_initial_volume_shampoo_l675_67547


namespace NUMINAMATH_GPT_min_moves_l675_67584

theorem min_moves (n : ℕ) : (n * (n + 1)) / 2 > 100 → n = 15 :=
by
  sorry

end NUMINAMATH_GPT_min_moves_l675_67584


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l675_67540

theorem geometric_sequence_third_term (a1 a5 a3 : ℕ) (r : ℝ) 
  (h1 : a1 = 4) 
  (h2 : a5 = 1296) 
  (h3 : a5 = a1 * r^4)
  (h4 : a3 = a1 * r^2) : 
  a3 = 36 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l675_67540


namespace NUMINAMATH_GPT_find_n_l675_67551

theorem find_n (n : ℤ) : -180 ≤ n ∧ n ≤ 180 ∧ (Real.sin (n * Real.pi / 180) = Real.cos (690 * Real.pi / 180)) → n = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_n_l675_67551


namespace NUMINAMATH_GPT_complement_of_M_in_U_l675_67585

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}

theorem complement_of_M_in_U : compl M ∩ U = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l675_67585


namespace NUMINAMATH_GPT_statement_C_correct_l675_67589

theorem statement_C_correct (a b c d : ℝ) (h_ab : a > b) (h_cd : c > d) : a + c > b + d :=
by
  sorry

end NUMINAMATH_GPT_statement_C_correct_l675_67589


namespace NUMINAMATH_GPT_minimum_handshakes_l675_67559

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_minimum_handshakes_l675_67559


namespace NUMINAMATH_GPT_pet_shop_legs_l675_67539

theorem pet_shop_legs :
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  birds * bird_legs + dogs * dog_legs + snakes * snake_legs + spiders * spider_legs = 34 := 
by
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  sorry

end NUMINAMATH_GPT_pet_shop_legs_l675_67539


namespace NUMINAMATH_GPT_travel_allowance_increase_20_l675_67529

def employees_total : ℕ := 480
def employees_no_increase : ℕ := 336
def employees_salary_increase_percentage : ℕ := 10

def employees_salary_increase : ℕ :=
(employees_salary_increase_percentage * employees_total) / 100

def employees_travel_allowance_increase : ℕ :=
employees_total - (employees_salary_increase + employees_no_increase)

def travel_allowance_increase_percentage : ℕ :=
(employees_travel_allowance_increase * 100) / employees_total

theorem travel_allowance_increase_20 :
  travel_allowance_increase_percentage = 20 :=
by sorry

end NUMINAMATH_GPT_travel_allowance_increase_20_l675_67529


namespace NUMINAMATH_GPT_problem_statement_l675_67509

-- Define the problem parameters with the constraints
def numberOfWaysToDistributeBalls (totalBalls : Nat) (initialDistribution : List Nat) : Nat :=
  -- Compute the number of remaining balls after the initial distribution
  let remainingBalls := totalBalls - initialDistribution.foldl (· + ·) 0
  -- Use the stars and bars formula to compute the number of ways to distribute remaining balls
  Nat.choose (remainingBalls + initialDistribution.length - 1) (initialDistribution.length - 1)

-- The boxes are to be numbered 1, 2, and 3, and each must contain at least its number of balls
def answer : Nat := numberOfWaysToDistributeBalls 9 [1, 2, 3]

-- Statement of the theorem
theorem problem_statement : answer = 10 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l675_67509


namespace NUMINAMATH_GPT_avg_weight_of_13_children_l675_67533

-- Definitions based on conditions:
def boys_avg_weight := 160
def boys_count := 8
def girls_avg_weight := 130
def girls_count := 5

-- Calculation to determine the total weights
def boys_total_weight := boys_avg_weight * boys_count
def girls_total_weight := girls_avg_weight * girls_count

-- Combined total weight
def total_weight := boys_total_weight + girls_total_weight

-- Average weight calculation
def children_count := boys_count + girls_count
def avg_weight := total_weight / children_count

-- The theorem to prove:
theorem avg_weight_of_13_children : avg_weight = 148 := by
  sorry

end NUMINAMATH_GPT_avg_weight_of_13_children_l675_67533


namespace NUMINAMATH_GPT_christina_total_payment_l675_67586

def item1_ticket_price : ℝ := 200
def item1_discount1 : ℝ := 0.25
def item1_discount2 : ℝ := 0.15
def item1_tax_rate : ℝ := 0.07

def item2_ticket_price : ℝ := 150
def item2_discount : ℝ := 0.30
def item2_tax_rate : ℝ := 0.10

def item3_ticket_price : ℝ := 100
def item3_discount : ℝ := 0.20
def item3_tax_rate : ℝ := 0.05

def expected_total : ℝ := 335.93

theorem christina_total_payment :
  let item1_final_price :=
    (item1_ticket_price * (1 - item1_discount1) * (1 - item1_discount2)) * (1 + item1_tax_rate)
  let item2_final_price :=
    (item2_ticket_price * (1 - item2_discount)) * (1 + item2_tax_rate)
  let item3_final_price :=
    (item3_ticket_price * (1 - item3_discount)) * (1 + item3_tax_rate)
  item1_final_price + item2_final_price + item3_final_price = expected_total :=
by
  sorry

end NUMINAMATH_GPT_christina_total_payment_l675_67586


namespace NUMINAMATH_GPT_product_third_fourth_term_l675_67534

theorem product_third_fourth_term (a d : ℝ) : 
  (a + 7 * d = 20) → (d = 2) → 
  ( (a + 2 * d) * (a + 3 * d) = 120 ) := 
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_product_third_fourth_term_l675_67534


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l675_67581

theorem number_of_ordered_pairs (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / 24)) : 
  ∃ n : ℕ, n = 41 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l675_67581


namespace NUMINAMATH_GPT_Lennon_total_reimbursement_l675_67522

def mileage_reimbursement (industrial_weekday: ℕ → ℕ) (commercial_weekday: ℕ → ℕ) (weekend: ℕ → ℕ) : ℕ :=
  let industrial_rate : ℕ := 36
  let commercial_weekday_rate : ℕ := 42
  let weekend_rate : ℕ := 45
  (industrial_weekday 1 * industrial_rate + commercial_weekday 1 * commercial_weekday_rate)    -- Monday
  + (industrial_weekday 2 * industrial_rate + commercial_weekday 2 * commercial_weekday_rate + commercial_weekday 3 * commercial_weekday_rate)  -- Tuesday
  + (industrial_weekday 3 * industrial_rate + commercial_weekday 3 * commercial_weekday_rate)    -- Wednesday
  + (commercial_weekday 4 * commercial_weekday_rate + commercial_weekday 5 * commercial_weekday_rate)  -- Thursday
  + (industrial_weekday 5 * industrial_rate + commercial_weekday 6 * commercial_weekday_rate + industrial_weekday 6 * industrial_rate)    -- Friday
  + (weekend 1 * weekend_rate)                                       -- Saturday

def monday_industrial_miles : ℕ := 10
def monday_commercial_miles : ℕ := 8

def tuesday_industrial_miles : ℕ := 12
def tuesday_commercial_miles_1 : ℕ := 9
def tuesday_commercial_miles_2 : ℕ := 5

def wednesday_industrial_miles : ℕ := 15
def wednesday_commercial_miles : ℕ := 5

def thursday_commercial_miles_1 : ℕ := 10
def thursday_commercial_miles_2 : ℕ := 10

def friday_industrial_miles_1 : ℕ := 5
def friday_commercial_miles : ℕ := 8
def friday_industrial_miles_2 : ℕ := 3

def saturday_commercial_miles : ℕ := 12

def reimbursement_total :=
  mileage_reimbursement
    (fun day => if day = 1 then monday_industrial_miles else if day = 2 then tuesday_industrial_miles else if day = 3 then wednesday_industrial_miles else if day = 5 then friday_industrial_miles_1 + friday_industrial_miles_2 else 0)
    (fun day => if day = 1 then monday_commercial_miles else if day = 2 then tuesday_commercial_miles_1 + tuesday_commercial_miles_2 else if day = 3 then wednesday_commercial_miles else if day = 4 then thursday_commercial_miles_1 + thursday_commercial_miles_2 else if day = 6 then friday_commercial_miles else 0)
    (fun day => if day = 1 then saturday_commercial_miles else 0)

theorem Lennon_total_reimbursement : reimbursement_total = 4470 := 
by sorry

end NUMINAMATH_GPT_Lennon_total_reimbursement_l675_67522


namespace NUMINAMATH_GPT_Problem_l675_67573

theorem Problem (x y : ℝ) (h1 : 2*x + 2*y = 10) (h2 : x*y = -15) : 4*(x^2) + 4*(y^2) = 220 := 
by
  sorry

end NUMINAMATH_GPT_Problem_l675_67573


namespace NUMINAMATH_GPT_train_passes_platform_in_43_2_seconds_l675_67520

open Real

noncomputable def length_of_train : ℝ := 360
noncomputable def length_of_platform : ℝ := 180
noncomputable def speed_of_train_kmph : ℝ := 45
noncomputable def speed_of_train_mps : ℝ := (45 * 1000) / 3600  -- Converting km/hr to m/s

noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_of_train_mps

theorem train_passes_platform_in_43_2_seconds :
  time_to_pass_platform = 43.2 := by
  sorry

end NUMINAMATH_GPT_train_passes_platform_in_43_2_seconds_l675_67520


namespace NUMINAMATH_GPT_factorial_sum_power_of_two_l675_67530

theorem factorial_sum_power_of_two (a b c : ℕ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) :
  a! + b! = 2 ^ c! ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) :=
by
  sorry

end NUMINAMATH_GPT_factorial_sum_power_of_two_l675_67530


namespace NUMINAMATH_GPT_solve_ab_c_eq_l675_67532

theorem solve_ab_c_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_eq : 11^a + 3^b = c^2) :
  a = 4 ∧ b = 5 ∧ c = 122 :=
by
  sorry

end NUMINAMATH_GPT_solve_ab_c_eq_l675_67532


namespace NUMINAMATH_GPT_width_of_hall_l675_67526

variable (L W H : ℕ) -- Length, Width, Height of the hall
variable (expenditure cost : ℕ) -- Expenditure and cost per square meter

-- Given conditions
def hall_length : L = 20 := by sorry
def hall_height : H = 5 := by sorry
def total_expenditure : expenditure = 28500 := by sorry
def cost_per_sq_meter : cost = 30 := by sorry

-- Derived value
def total_area_to_cover (W : ℕ) : ℕ :=
  (2 * (L * W) + 2 * (L * H) + 2 * (W * H))

theorem width_of_hall (W : ℕ) (h: total_area_to_cover L W H * cost = expenditure) : W = 15 := by
  sorry

end NUMINAMATH_GPT_width_of_hall_l675_67526


namespace NUMINAMATH_GPT_women_stockbrokers_2005_l675_67563

-- Define the context and conditions
def women_stockbrokers_2000 : ℕ := 10000
def percent_increase_2005 : ℕ := 100

-- Statement to prove the number of women stockbrokers in 2005
theorem women_stockbrokers_2005 : women_stockbrokers_2000 + women_stockbrokers_2000 * percent_increase_2005 / 100 = 20000 := by
  sorry

end NUMINAMATH_GPT_women_stockbrokers_2005_l675_67563


namespace NUMINAMATH_GPT_new_acute_angle_ACB_l675_67583

-- Define the initial condition: the measure of angle ACB is 50 degrees.
def measure_ACB_initial : ℝ := 50

-- Define the rotation: ray CA is rotated by 540 degrees clockwise.
def rotation_CW_degrees : ℝ := 540

-- Theorem statement: The positive measure of the new acute angle ACB.
theorem new_acute_angle_ACB : 
  ∃ (new_angle : ℝ), new_angle = 50 ∧ new_angle < 90 := 
by
  sorry

end NUMINAMATH_GPT_new_acute_angle_ACB_l675_67583


namespace NUMINAMATH_GPT_part1_part2_l675_67557

noncomputable def A_m (m : ℕ) (k : ℕ) : ℕ := (2 * k - 1) * m + k

theorem part1 (m : ℕ) (hm : m ≥ 2) :
  ∃ a : ℕ, 1 ≤ a ∧ a < m ∧ (∃ k : ℕ, 2^a = A_m m k) ∨ (∃ k : ℕ, 2^a + 1 = A_m m k) :=
sorry

theorem part2 {m : ℕ} (hm : m ≥ 2) 
  (a b : ℕ) (ha : ∃ k, 2^a = A_m m k) (hb : ∃ k, 2^b + 1 = A_m m k)
  (hmin_a : ∀ x, (∃ k, 2^x = A_m m k) → a ≤ x) 
  (hmin_b : ∀ y, (∃ k, 2^y + 1 = A_m m k) → b ≤ y) :
  a = 2 * b + 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l675_67557


namespace NUMINAMATH_GPT_ending_number_is_54_l675_67572

def first_even_after_15 : ℕ := 16
def evens_between (a b : ℕ) : ℕ := (b - first_even_after_15) / 2 + 1

theorem ending_number_is_54 (n : ℕ) (h : evens_between 15 n = 20) : n = 54 :=
by {
  sorry
}

end NUMINAMATH_GPT_ending_number_is_54_l675_67572


namespace NUMINAMATH_GPT_sum_of_solutions_l675_67515

theorem sum_of_solutions :
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    ((x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  ((∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (1 + 1 = 3 ∨ true)) → 
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  (-1) + 0 + 2 + 3 + 7 + 2 = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l675_67515


namespace NUMINAMATH_GPT_smallest_A_divided_by_6_has_third_of_original_factors_l675_67566

theorem smallest_A_divided_by_6_has_third_of_original_factors:
  ∃ A: ℕ, A > 0 ∧ (∃ a b: ℕ, A = 2^a * 3^b ∧ (a + 1) * (b + 1) = 3 * a * b) ∧ A = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_A_divided_by_6_has_third_of_original_factors_l675_67566


namespace NUMINAMATH_GPT_pumps_fill_time_l675_67570

def fill_time {X Y Z : ℝ} (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : Prop :=
  1 / (X + Y + Z) = 36 / 13

theorem pumps_fill_time (X Y Z : ℝ) (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : 
  1 / (X + Y + Z) = 36 / 13 :=
by
  sorry

end NUMINAMATH_GPT_pumps_fill_time_l675_67570


namespace NUMINAMATH_GPT_eval_P_at_4_over_3_eval_P_at_2_l675_67514

noncomputable def P (a : ℚ) : ℚ := (6 * a^2 - 14 * a + 5) * (3 * a - 4)

theorem eval_P_at_4_over_3 : P (4 / 3) = 0 :=
by sorry

theorem eval_P_at_2 : P 2 = 2 :=
by sorry

end NUMINAMATH_GPT_eval_P_at_4_over_3_eval_P_at_2_l675_67514


namespace NUMINAMATH_GPT_driving_distance_l675_67553

def miles_per_gallon : ℕ := 20
def gallons_of_gas : ℕ := 5

theorem driving_distance :
  miles_per_gallon * gallons_of_gas = 100 :=
  sorry

end NUMINAMATH_GPT_driving_distance_l675_67553


namespace NUMINAMATH_GPT_calc_fraction_l675_67505

theorem calc_fraction : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end NUMINAMATH_GPT_calc_fraction_l675_67505


namespace NUMINAMATH_GPT_original_cost_price_of_car_l675_67578

theorem original_cost_price_of_car
    (S_m S_f C : ℝ)
    (h1 : S_m = 0.86 * C)
    (h2 : S_f = 54000)
    (h3 : S_f = 1.20 * S_m) :
    C = 52325.58 :=
by
    sorry

end NUMINAMATH_GPT_original_cost_price_of_car_l675_67578


namespace NUMINAMATH_GPT_initial_numbers_count_l675_67587

theorem initial_numbers_count (n : ℕ) (S : ℝ)
  (h1 : S / n = 56)
  (h2 : (S - 100) / (n - 2) = 56.25) :
  n = 50 :=
sorry

end NUMINAMATH_GPT_initial_numbers_count_l675_67587


namespace NUMINAMATH_GPT_horners_rule_correct_l675_67507

open Classical

variables (x : ℤ) (poly_val : ℤ)

def original_polynomial (x : ℤ) : ℤ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_evaluation (x : ℤ) : ℤ := ((7 * x + 3) * x - 5) * x + 11

theorem horners_rule_correct : (poly_val = horner_evaluation 23) ↔ (poly_val = original_polynomial 23) :=
by {
  sorry
}

end NUMINAMATH_GPT_horners_rule_correct_l675_67507


namespace NUMINAMATH_GPT_powers_of_two_l675_67516

theorem powers_of_two (n : ℕ) (h : ∀ n, ∃ m, (2^n - 1) ∣ (m^2 + 9)) : ∃ s, n = 2^s :=
sorry

end NUMINAMATH_GPT_powers_of_two_l675_67516


namespace NUMINAMATH_GPT_rabbit_fraction_l675_67517

theorem rabbit_fraction
  (initial_rabbits : ℕ) (added_rabbits : ℕ) (total_rabbits_seen : ℕ)
  (h_initial : initial_rabbits = 13)
  (h_added : added_rabbits = 7)
  (h_seen : total_rabbits_seen = 60) :
  (initial_rabbits + added_rabbits) / total_rabbits_seen = 1 / 3 :=
by
  -- we will prove this
  sorry

end NUMINAMATH_GPT_rabbit_fraction_l675_67517


namespace NUMINAMATH_GPT_three_digit_numbers_l675_67552

theorem three_digit_numbers (n : ℕ) :
  n = 4 ↔ ∃ (x y : ℕ), 
  (100 ≤ 101 * x + 10 * y ∧ 101 * x + 10 * y < 1000) ∧ 
  (x ≠ 0 ∧ x ≠ 5) ∧ 
  (2 * x + y = 15) ∧ 
  (y < 10) :=
by { sorry }

end NUMINAMATH_GPT_three_digit_numbers_l675_67552


namespace NUMINAMATH_GPT_original_price_of_cycle_l675_67506

theorem original_price_of_cycle (P : ℝ) (h1 : P * 0.85 = 1190) : P = 1400 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_cycle_l675_67506


namespace NUMINAMATH_GPT_sum_of_consecutive_even_integers_l675_67524

theorem sum_of_consecutive_even_integers
  (a1 a2 a3 a4 : ℤ)
  (h1 : a2 = a1 + 2)
  (h2 : a3 = a1 + 4)
  (h3 : a4 = a1 + 6)
  (h_sum : a1 + a3 = 146) :
  a1 + a2 + a3 + a4 = 296 :=
by sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_integers_l675_67524


namespace NUMINAMATH_GPT_total_numbers_l675_67528

theorem total_numbers (n : ℕ) (a : ℕ → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 25)
  (h2 : (a (n - 3) + a (n - 2) + a (n - 1)) / 3 = 35)
  (h3 : a 3 = 25)
  (h4 : (Finset.sum (Finset.range n) a) / n = 30) :
  n = 6 :=
sorry

end NUMINAMATH_GPT_total_numbers_l675_67528


namespace NUMINAMATH_GPT_lottery_probability_correct_l675_67556

/-- The binomial coefficient function -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of matching MegaBall and WinnerBalls in the lottery -/
noncomputable def lottery_probability : ℚ :=
  let megaBall_prob := (1 : ℚ) / 30
  let winnerBalls_prob := (1 : ℚ) / binom 45 6
  megaBall_prob * winnerBalls_prob

theorem lottery_probability_correct : lottery_probability = (1 : ℚ) / 244351800 := by
  sorry

end NUMINAMATH_GPT_lottery_probability_correct_l675_67556


namespace NUMINAMATH_GPT_proof_ab_lt_1_l675_67555

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem proof_ab_lt_1 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a > f b) : a * b < 1 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_proof_ab_lt_1_l675_67555
