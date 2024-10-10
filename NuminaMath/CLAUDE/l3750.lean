import Mathlib

namespace ball_travel_distance_l3750_375048

/-- The distance traveled by a ball rolling down a ramp -/
def ballDistance (initialDistance : ℕ) (increase : ℕ) (time : ℕ) : ℕ :=
  let lastTerm := initialDistance + (time - 1) * increase
  time * (initialDistance + lastTerm) / 2

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_travel_distance :
  ballDistance 10 8 25 = 2650 := by
  sorry

end ball_travel_distance_l3750_375048


namespace nonIntersectingPolylines_correct_l3750_375052

/-- The number of ways to connect n points on a circle with a non-self-intersecting polyline -/
def nonIntersectingPolylines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n-3)
  else 0

theorem nonIntersectingPolylines_correct (n : ℕ) (h : n > 1) :
  nonIntersectingPolylines n =
    if n = 2 then 1
    else n * 2^(n-3) := by
  sorry

end nonIntersectingPolylines_correct_l3750_375052


namespace game_cost_is_two_l3750_375098

/-- Calculates the cost of a new game based on initial money, allowance, and final amount. -/
def game_cost (initial_money : ℝ) (allowance : ℝ) (final_amount : ℝ) : ℝ :=
  initial_money + allowance - final_amount

/-- Proves that the cost of the new game is $2 given the specific amounts in the problem. -/
theorem game_cost_is_two :
  game_cost 5 5 8 = 2 := by
  sorry

end game_cost_is_two_l3750_375098


namespace number_of_girls_in_college_l3750_375074

theorem number_of_girls_in_college (total_students : ℕ) (boys_to_girls_ratio : ℚ) 
  (h1 : total_students = 416) 
  (h2 : boys_to_girls_ratio = 8 / 5) : 
  ∃ (girls : ℕ), girls = 160 ∧ 
    (girls : ℚ) * (1 + boys_to_girls_ratio) = total_students := by
  sorry

end number_of_girls_in_college_l3750_375074


namespace correct_quotient_after_error_l3750_375072

theorem correct_quotient_after_error (dividend : ℕ) (incorrect_divisor correct_divisor incorrect_quotient : ℕ) :
  incorrect_divisor = 48 →
  correct_divisor = 36 →
  incorrect_quotient = 24 →
  dividend = incorrect_divisor * incorrect_quotient →
  dividend / correct_divisor = 32 :=
by sorry

end correct_quotient_after_error_l3750_375072


namespace conic_eccentricity_l3750_375029

/-- The eccentricity of a conic section given by x^2 + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) → 
  (∃ (x y : ℝ), x^2 + y^2/m = 1) → 
  (∃ (e : ℝ), e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) :=
sorry

end conic_eccentricity_l3750_375029


namespace james_calories_per_minute_l3750_375060

/-- Represents the number of calories burned per minute in a spinning class -/
def calories_per_minute (classes_per_week : ℕ) (hours_per_class : ℚ) (total_calories_per_week : ℕ) : ℚ :=
  let minutes_per_week : ℚ := classes_per_week * hours_per_class * 60
  total_calories_per_week / minutes_per_week

/-- Proves that James burns 7 calories per minute in his spinning class -/
theorem james_calories_per_minute :
  calories_per_minute 3 (3/2) 1890 = 7 := by
sorry

end james_calories_per_minute_l3750_375060


namespace unique_relative_minimum_l3750_375092

/-- The function f(x) = x^4 - x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - x^2 + a*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 - 3*x^2 - 2*x + a

theorem unique_relative_minimum (a : ℝ) :
  (∃ (x : ℝ), f a x = x ∧ 
    ∀ (y : ℝ), y ≠ x → f a y > f a x) ↔ a = 1 := by
  sorry

end unique_relative_minimum_l3750_375092


namespace p_sufficient_not_necessary_for_q_l3750_375025

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by
  sorry

end p_sufficient_not_necessary_for_q_l3750_375025


namespace fudge_difference_is_14_ounces_l3750_375051

/-- Conversion factor from pounds to ounces -/
def poundsToOunces : ℚ := 16

/-- Marina's fudge in pounds -/
def marinaFudgePounds : ℚ := 4.5

/-- Amount of fudge Lazlo has less than 4 pounds, in ounces -/
def lazloFudgeDifference : ℚ := 6

/-- Calculates the difference in ounces of fudge between Marina and Lazlo -/
def fudgeDifferenceInOunces : ℚ :=
  marinaFudgePounds * poundsToOunces - (4 * poundsToOunces - lazloFudgeDifference)

theorem fudge_difference_is_14_ounces :
  fudgeDifferenceInOunces = 14 := by
  sorry

end fudge_difference_is_14_ounces_l3750_375051


namespace students_per_group_l3750_375065

theorem students_per_group (total : ℕ) (not_picked : ℕ) (groups : ℕ) 
  (h1 : total = 17) 
  (h2 : not_picked = 5) 
  (h3 : groups = 3) :
  (total - not_picked) / groups = 4 := by
sorry

end students_per_group_l3750_375065


namespace mod_equivalence_problem_l3750_375013

theorem mod_equivalence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end mod_equivalence_problem_l3750_375013


namespace unique_k_value_l3750_375036

theorem unique_k_value : ∃! k : ℝ, ∀ x : ℝ, 
  (x * (2 * x + 3) < k ↔ -5/2 < x ∧ x < 1) := by
  sorry

end unique_k_value_l3750_375036


namespace sector_central_angle_l3750_375040

theorem sector_central_angle (circumference : ℝ) (area : ℝ) :
  circumference = 6 →
  area = 2 →
  ∃ (r l : ℝ),
    l + 2*r = 6 ∧
    (1/2) * l * r = 2 ∧
    (l / r = 1 ∨ l / r = 4) :=
by sorry

end sector_central_angle_l3750_375040


namespace craigs_remaining_apples_l3750_375023

/-- Calculates the number of apples Craig has after sharing -/
def craigs_apples_after_sharing (initial_apples : ℕ) (shared_apples : ℕ) : ℕ :=
  initial_apples - shared_apples

theorem craigs_remaining_apples :
  craigs_apples_after_sharing 20 7 = 13 := by
  sorry

end craigs_remaining_apples_l3750_375023


namespace negation_of_universal_proposition_l3750_375068

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 0) := by sorry

end negation_of_universal_proposition_l3750_375068


namespace horse_value_is_240_l3750_375007

/-- Represents the payment terms and actual service of a soldier --/
structure SoldierPayment where
  total_payment : ℕ  -- Total payment promised for full service in florins
  service_period : ℕ  -- Full service period in months
  actual_service : ℕ  -- Actual service period in months
  cash_payment : ℕ   -- Cash payment given at the end of actual service

/-- Calculates the value of a horse given to a soldier as part of payment --/
def horse_value (p : SoldierPayment) : ℕ :=
  p.total_payment - (p.total_payment / p.service_period * p.actual_service + p.cash_payment)

/-- Theorem stating the value of the horse in the given problem --/
theorem horse_value_is_240 (p : SoldierPayment) 
  (h1 : p.total_payment = 300)
  (h2 : p.service_period = 36)
  (h3 : p.actual_service = 17)
  (h4 : p.cash_payment = 15) :
  horse_value p = 240 := by
  sorry

end horse_value_is_240_l3750_375007


namespace ice_cream_scoop_arrangements_l3750_375071

theorem ice_cream_scoop_arrangements (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end ice_cream_scoop_arrangements_l3750_375071


namespace solution_replacement_fraction_l3750_375058

theorem solution_replacement_fraction (V : ℝ) (x : ℝ) 
  (h1 : V > 0)
  (h2 : 0 ≤ x ∧ x ≤ 1)
  (h3 : (0.80 * V - 0.80 * x * V) + 0.25 * x * V = 0.35 * V) :
  x = 9 / 11 := by
sorry

end solution_replacement_fraction_l3750_375058


namespace quadratic_roots_sums_l3750_375057

theorem quadratic_roots_sums (p q x₁ x₂ : ℝ) 
  (hq : q ≠ 0)
  (hroots : x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) :
  (1/x₁ + 1/x₂ = -p/q) ∧
  (1/x₁^2 + 1/x₂^2 = (p^2 - 2*q)/q^2) ∧
  (1/x₁^3 + 1/x₂^3 = (p/q^3)*(3*q - p^2)) := by
  sorry


end quadratic_roots_sums_l3750_375057


namespace no_solutions_sqrt_1452_l3750_375015

theorem no_solutions_sqrt_1452 : 
  ¬ ∃ (x y : ℕ), 0 < x ∧ x < y ∧ Real.sqrt 1452 = Real.sqrt x + Real.sqrt y := by
  sorry

end no_solutions_sqrt_1452_l3750_375015


namespace polynomial_expansion_l3750_375066

theorem polynomial_expansion (z : ℝ) :
  (3 * z^2 - 4 * z + 1) * (2 * z^3 + 3 * z^2 - 5 * z + 2) =
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2 := by
  sorry

end polynomial_expansion_l3750_375066


namespace rectangular_prism_diagonal_l3750_375090

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 3) (hw : w = 4) (hh : h = 5) :
  Real.sqrt (l^2 + w^2 + h^2) = 5 * Real.sqrt 2 := by
  sorry

end rectangular_prism_diagonal_l3750_375090


namespace number_divisibility_l3750_375064

theorem number_divisibility :
  (∀ a : ℕ, 100 ≤ a ∧ a < 1000 → (7 ∣ 1001 * a) ∧ (11 ∣ 1001 * a) ∧ (13 ∣ 1001 * a)) ∧
  (∀ b : ℕ, 1000 ≤ b ∧ b < 10000 → (73 ∣ 10001 * b) ∧ (137 ∣ 10001 * b)) :=
by sorry

end number_divisibility_l3750_375064


namespace blue_to_red_ratio_l3750_375067

theorem blue_to_red_ratio (n : ℕ) (h : n = 13) : 
  (6 * n^3 - 6 * n^2) / (6 * n^2) = 12 := by
  sorry

end blue_to_red_ratio_l3750_375067


namespace gilbert_herb_count_l3750_375016

/-- Represents the number of herb plants Gilbert has at different stages of spring -/
structure HerbGarden where
  initial_basil : ℕ
  initial_parsley : ℕ
  initial_mint : ℕ
  extra_basil : ℕ
  eaten_mint : ℕ

/-- Calculates the final number of herb plants in Gilbert's garden -/
def final_herb_count (garden : HerbGarden) : ℕ :=
  garden.initial_basil + garden.initial_parsley + garden.initial_mint + garden.extra_basil - garden.eaten_mint

/-- Theorem stating that Gilbert had 5 herb plants at the end of spring -/
theorem gilbert_herb_count :
  ∀ (garden : HerbGarden),
    garden.initial_basil = 3 →
    garden.initial_parsley = 1 →
    garden.initial_mint = 2 →
    garden.extra_basil = 1 →
    garden.eaten_mint = 2 →
    final_herb_count garden = 5 := by
  sorry


end gilbert_herb_count_l3750_375016


namespace new_employee_age_l3750_375019

theorem new_employee_age 
  (initial_employees : ℕ) 
  (initial_avg_age : ℝ) 
  (final_employees : ℕ) 
  (final_avg_age : ℝ) : 
  initial_employees = 13 → 
  initial_avg_age = 35 → 
  final_employees = initial_employees + 1 → 
  final_avg_age = 34 → 
  (final_employees * final_avg_age - initial_employees * initial_avg_age : ℝ) = 21 := by
  sorry

end new_employee_age_l3750_375019


namespace decimal_expansion_415th_digit_l3750_375041

/-- The decimal expansion of 17/29 -/
def decimal_expansion : ℚ := 17 / 29

/-- The length of the repeating cycle in the decimal expansion of 17/29 -/
def cycle_length : ℕ := 87

/-- The position of the 415th digit within the repeating cycle -/
def position_in_cycle : ℕ := 415 % cycle_length

/-- The 415th digit in the decimal expansion of 17/29 -/
def digit_415 : ℕ := 8

/-- Theorem stating that the 415th digit to the right of the decimal point
    in the decimal expansion of 17/29 is 8 -/
theorem decimal_expansion_415th_digit :
  digit_415 = 8 :=
sorry

end decimal_expansion_415th_digit_l3750_375041


namespace power_of_256_l3750_375055

theorem power_of_256 : (256 : ℝ) ^ (5/8 : ℝ) = 32 := by
  sorry

end power_of_256_l3750_375055


namespace sqrt_sum_sin_equals_sqrt_two_minus_cos_l3750_375002

theorem sqrt_sum_sin_equals_sqrt_two_minus_cos (α : Real) 
  (h : 5 * Real.pi / 2 ≤ α ∧ α ≤ 7 * Real.pi / 2) : 
  Real.sqrt (1 + Real.sin α) + Real.sqrt (1 - Real.sin α) = Real.sqrt (2 - Real.cos α) := by
  sorry

end sqrt_sum_sin_equals_sqrt_two_minus_cos_l3750_375002


namespace intersection_of_M_and_N_l3750_375011

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l3750_375011


namespace inspection_team_selection_l3750_375088

theorem inspection_team_selection 
  (total_employees : ℕ) 
  (men : ℕ) 
  (women : ℕ) 
  (team_size : ℕ) 
  (h1 : total_employees = 15)
  (h2 : men = 10)
  (h3 : women = 5)
  (h4 : team_size = 6)
  (h5 : men + women = total_employees)
  (h6 : 2 * women = men) : 
  Nat.choose men 4 * Nat.choose women 2 = 
  (number_of_ways_to_select_team : ℕ) := by
  sorry

end inspection_team_selection_l3750_375088


namespace simplify_expression_l3750_375050

theorem simplify_expression (x y : ℝ) : (5 - 4*y) - (6 + 5*y - 2*x) = -1 - 9*y + 2*x := by
  sorry

end simplify_expression_l3750_375050


namespace expression_evaluation_l3750_375032

theorem expression_evaluation :
  let a : ℤ := 2
  let b : ℤ := -1
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -14 :=
by sorry

end expression_evaluation_l3750_375032


namespace absolute_value_six_point_five_l3750_375096

theorem absolute_value_six_point_five (x : ℝ) : |x| = 6.5 ↔ x = 6.5 ∨ x = -6.5 := by
  sorry

end absolute_value_six_point_five_l3750_375096


namespace mini_train_length_l3750_375077

/-- The length of a mini-train given its speed and time to cross a pole -/
theorem mini_train_length (speed_kmph : ℝ) (time_seconds : ℝ) : 
  speed_kmph = 75 → time_seconds = 3 → 
  (speed_kmph * 1000 / 3600) * time_seconds = 62.5 := by
  sorry

end mini_train_length_l3750_375077


namespace triangle_area_l3750_375017

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (3, 5)

theorem triangle_area : 
  (1/2 : ℝ) * |a.1 * b.2 - a.2 * b.1| = 23/2 := by sorry

end triangle_area_l3750_375017


namespace triangle_problem_l3750_375082

noncomputable def f (x θ : Real) : Real :=
  2 * Real.sin x * (Real.cos (θ / 2))^2 + Real.cos x * Real.sin θ - Real.sin x

theorem triangle_problem (θ : Real) (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : ∀ x, f x θ ≥ f π θ) :
  ∃ (A B C : Real),
    θ = π / 2 ∧
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    Real.sin B / Real.sin A = Real.sqrt 2 ∧
    f A (π / 2) = Real.sqrt 3 / 2 ∧
    (C = 7 * π / 12 ∨ C = π / 12) := by
  sorry

end triangle_problem_l3750_375082


namespace line_tangent_to_ellipse_l3750_375009

/-- Theorem: If a line y = mx + 3 is tangent to the ellipse x² + 9y² = 9, then m² = 8/9 -/
theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃! x y : ℝ, y = m * x + 3 ∧ x^2 + 9 * y^2 = 9) → m^2 = 8/9 := by
  sorry

end line_tangent_to_ellipse_l3750_375009


namespace sam_tuna_change_sam_change_proof_l3750_375024

/-- Calculates the change Sam received when buying tuna cans. -/
theorem sam_tuna_change (num_cans : ℕ) (num_coupons : ℕ) (coupon_value : ℕ) 
  (can_cost : ℕ) (paid_amount : ℕ) : ℕ :=
  let total_discount := num_coupons * coupon_value
  let total_cost := num_cans * can_cost
  let actual_paid := total_cost - total_discount
  paid_amount - actual_paid

/-- Proves that Sam received $5.50 in change. -/
theorem sam_change_proof : 
  sam_tuna_change 9 5 25 175 2000 = 550 := by
  sorry

end sam_tuna_change_sam_change_proof_l3750_375024


namespace unique_prime_with_next_square_is_three_l3750_375043

theorem unique_prime_with_next_square_is_three :
  ∀ p : ℕ, Prime p → (∃ n : ℕ, p + 1 = n^2) → p = 3 :=
by
  sorry

end unique_prime_with_next_square_is_three_l3750_375043


namespace fast_site_selection_probability_l3750_375008

theorem fast_site_selection_probability (total : ℕ) (guizhou : ℕ) (selected : ℕ)
  (h1 : total = 8)
  (h2 : guizhou = 3)
  (h3 : selected = 2)
  (h4 : guizhou ≤ total) :
  (Nat.choose guizhou 1 * Nat.choose (total - guizhou) 1 + Nat.choose guizhou 2) / Nat.choose total selected = 9 / 14 :=
by sorry

end fast_site_selection_probability_l3750_375008


namespace compute_custom_op_l3750_375021

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem compute_custom_op : custom_op (custom_op 8 6) 2 = 9 / 5 := by
  sorry

end compute_custom_op_l3750_375021


namespace smithtown_left_handed_women_percentage_l3750_375083

theorem smithtown_left_handed_women_percentage
  (total : ℕ)
  (right_handed : ℕ)
  (left_handed : ℕ)
  (men : ℕ)
  (women : ℕ)
  (h1 : right_handed = 3 * left_handed)
  (h2 : men = 3 * (men + women) / 5)
  (h3 : women = 2 * (men + women) / 5)
  (h4 : total = right_handed + left_handed)
  (h5 : total = men + women)
  (h6 : men ≤ right_handed) :
  left_handed * 100 / total = 25 := by
sorry

end smithtown_left_handed_women_percentage_l3750_375083


namespace tammys_climbing_speed_l3750_375042

/-- Tammy's mountain climbing problem -/
theorem tammys_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) :
  ∃ (speed_day1 speed_day2 time_day1 time_day2 : ℝ),
    speed_day2 = speed_day1 + speed_difference ∧
    time_day2 = time_day1 - time_difference ∧
    time_day1 + time_day2 = total_time ∧
    speed_day1 * time_day1 + speed_day2 * time_day2 = total_distance ∧
    speed_day2 = 4 := by
  sorry

end tammys_climbing_speed_l3750_375042


namespace intersection_M_N_l3750_375020

def M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
def N : Set ℝ := {x | ∃ y, y = x^2 - 1}

theorem intersection_M_N : M ∩ N = Set.Icc (-1) (Real.sqrt 2) := by sorry

end intersection_M_N_l3750_375020


namespace fourth_game_score_l3750_375033

def game_scores (game1 game2 game3 game4 total : ℕ) : Prop :=
  game1 = 10 ∧ game2 = 14 ∧ game3 = 6 ∧ game1 + game2 + game3 + game4 = total

theorem fourth_game_score (game1 game2 game3 game4 total : ℕ) :
  game_scores game1 game2 game3 game4 total → total = 40 → game4 = 10 := by
  sorry

end fourth_game_score_l3750_375033


namespace sqrt_fifteen_div_sqrt_three_eq_sqrt_five_l3750_375045

theorem sqrt_fifteen_div_sqrt_three_eq_sqrt_five : 
  Real.sqrt 15 / Real.sqrt 3 = Real.sqrt 5 := by
  sorry

end sqrt_fifteen_div_sqrt_three_eq_sqrt_five_l3750_375045


namespace base6_addition_correct_l3750_375056

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The first number in base 6 -/
def num1 : List Nat := [3, 4, 2, 1]

/-- The second number in base 6 -/
def num2 : List Nat := [4, 5, 2, 5]

/-- The expected sum in base 6 -/
def expectedSum : List Nat := [1, 2, 3, 5, 0]

theorem base6_addition_correct :
  decimalToBase6 (base6ToDecimal num1 + base6ToDecimal num2) = expectedSum := by
  sorry

end base6_addition_correct_l3750_375056


namespace triangle_area_inequality_l3750_375005

/-- Triangle type with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  valid : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b

/-- Theorem statement for triangle inequality -/
theorem triangle_area_inequality (t : Triangle) :
  t.S ≤ (t.a^2 + t.b^2 + t.c^2) / (4 * Real.sqrt 3) ∧
  (t.S = (t.a^2 + t.b^2 + t.c^2) / (4 * Real.sqrt 3) ↔ t.a = t.b ∧ t.b = t.c) :=
by sorry

end triangle_area_inequality_l3750_375005


namespace fourth_root_of_polynomial_l3750_375022

theorem fourth_root_of_polynomial (a b : ℝ) : 
  (∀ x : ℝ, a * x^4 + (a + 2*b) * x^3 + (b - 3*a) * x^2 + (2*a - 6) * x + (7 - a) = 0 ↔ 
    x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2) → 
  ∃ x : ℝ, x = -2 ∧ a * x^4 + (a + 2*b) * x^3 + (b - 3*a) * x^2 + (2*a - 6) * x + (7 - a) = 0 :=
by sorry

end fourth_root_of_polynomial_l3750_375022


namespace smallest_n_with_2323_divisible_l3750_375089

def count_divisible (n : ℕ) : ℕ :=
  (n / 2) + (n / 23) - 2 * (n / 46)

theorem smallest_n_with_2323_divisible : ∃ (n : ℕ), n > 0 ∧ count_divisible n = 2323 ∧ ∀ m < n, count_divisible m ≠ 2323 :=
sorry

end smallest_n_with_2323_divisible_l3750_375089


namespace book_arrangement_proof_l3750_375034

def arrange_books (geometry_copies : ℕ) (algebra_copies : ℕ) : ℕ :=
  Nat.choose (geometry_copies + algebra_copies - 2) (algebra_copies - 2)

theorem book_arrangement_proof : 
  arrange_books 4 5 = 35 :=
by sorry

end book_arrangement_proof_l3750_375034


namespace smallest_possible_a_l3750_375091

theorem smallest_possible_a (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b →
  b ≤ c →
  c ≥ 25 →
  ∃ (a' b' c' : ℕ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (a' + b' + c') / 3 = 20 ∧
    a' ≤ b' ∧
    b' ≤ c' ∧
    c' ≥ 25 ∧
    a' = 1 ∧
    ∀ (a'' : ℕ), a'' > 0 → 
      (∃ (b'' c'' : ℕ), b'' > 0 ∧ c'' > 0 ∧
        (a'' + b'' + c'') / 3 = 20 ∧
        a'' ≤ b'' ∧
        b'' ≤ c'' ∧
        c'' ≥ 25) →
      a'' ≥ a' := by
  sorry


end smallest_possible_a_l3750_375091


namespace diet_soda_bottles_l3750_375073

theorem diet_soda_bottles (regular_soda : ℕ) (lite_soda : ℕ) (total_bottles : ℕ) 
  (h1 : regular_soda = 57)
  (h2 : lite_soda = 27)
  (h3 : total_bottles = 110) :
  total_bottles - (regular_soda + lite_soda) = 26 := by
  sorry

end diet_soda_bottles_l3750_375073


namespace f_inequality_range_l3750_375095

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) - 1 / (x^2 + 2)

theorem f_inequality_range (x : ℝ) : 
  f x > f (2 * x - 1) ↔ 1/3 < x ∧ x < 1 :=
by sorry

end f_inequality_range_l3750_375095


namespace polynomial_value_l3750_375044

theorem polynomial_value (x y : ℝ) (h : x + 2*y = 6) : 2*x + 4*y - 5 = 7 := by
  sorry

end polynomial_value_l3750_375044


namespace candy_distribution_l3750_375010

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) (h1 : total_candy = 30) (h2 : num_friends = 4) :
  total_candy - (total_candy / num_friends) * num_friends = 2 :=
by
  sorry

end candy_distribution_l3750_375010


namespace larger_circle_radius_l3750_375014

theorem larger_circle_radius (r : ℝ) (h1 : r = 2) : ∃ R : ℝ,
  (∀ i j : Fin 4, i ≠ j → (∃ c₁ c₂ : ℝ × ℝ, 
    dist c₁ c₂ = 2 * r ∧ 
    (∀ x : ℝ × ℝ, dist x c₁ ≤ r ∨ dist x c₂ ≤ r))) →
  (∃ C : ℝ × ℝ, ∀ i : Fin 4, ∃ c : ℝ × ℝ, 
    dist C c = R - r ∧ 
    (∀ x : ℝ × ℝ, dist x c ≤ r → dist x C ≤ R)) →
  R = 4 * Real.sqrt 2 + 2 :=
sorry

end larger_circle_radius_l3750_375014


namespace smallest_sum_of_sequence_l3750_375030

theorem smallest_sum_of_sequence (X Y Z W : ℤ) : 
  X > 0 → Y > 0 → Z > 0 →  -- X, Y, Z are positive integers
  (∃ d : ℤ, Y - X = d ∧ Z - Y = d) →  -- X, Y, Z form an arithmetic sequence
  (∃ r : ℚ, Z = Y * r ∧ W = Z * r) →  -- Y, Z, W form a geometric sequence
  Z = (7 * Y) / 4 →  -- Z/Y = 7/4
  (∀ X' Y' Z' W' : ℤ, 
    X' > 0 → Y' > 0 → Z' > 0 →
    (∃ d : ℤ, Y' - X' = d ∧ Z' - Y' = d) →
    (∃ r : ℚ, Z' = Y' * r ∧ W' = Z' * r) →
    Z' = (7 * Y') / 4 →
    X + Y + Z + W ≤ X' + Y' + Z' + W') →
  X + Y + Z + W = 97 :=
by sorry

end smallest_sum_of_sequence_l3750_375030


namespace negative_nine_less_than_negative_sqrt_80_l3750_375028

theorem negative_nine_less_than_negative_sqrt_80 : -9 < -Real.sqrt 80 := by
  sorry

end negative_nine_less_than_negative_sqrt_80_l3750_375028


namespace mr_blue_bean_yield_l3750_375076

/-- Calculates the expected bean yield for a rectangular terrain --/
def expected_bean_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length * yield_per_sqft

/-- Proves that the expected bean yield for Mr. Blue's terrain is 5906.25 pounds --/
theorem mr_blue_bean_yield :
  expected_bean_yield 25 35 3 0.75 = 5906.25 := by
  sorry

#eval expected_bean_yield 25 35 3 0.75

end mr_blue_bean_yield_l3750_375076


namespace work_completion_time_l3750_375099

/-- If P people can complete a job in 20 days, then 2P people can complete half of the job in 5 days -/
theorem work_completion_time 
  (P : ℕ) -- number of people
  (full_work_time : ℕ := 20) -- time to complete full work with P people
  (h : P > 0) -- ensure P is positive
  : (2 * P) * 5 = P * full_work_time / 2 := by
  sorry

end work_completion_time_l3750_375099


namespace pear_price_l3750_375093

/-- Proves that the price of a pear is $60 given the conditions from the problem -/
theorem pear_price (orange pear banana : ℚ) 
  (h1 : orange - pear = banana)
  (h2 : orange + pear = 120)
  (h3 : 200 * banana + 400 * orange = 24000) : 
  pear = 60 := by
  sorry

end pear_price_l3750_375093


namespace quadratic_expression_value_l3750_375049

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 9) 
  (eq2 : x + 4 * y = 16) : 
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 := by
sorry

end quadratic_expression_value_l3750_375049


namespace matrix_equation_solution_l3750_375026

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 0],
    ![0, 1, 2],
    ![2, 0, 1]]

theorem matrix_equation_solution :
  ∃ (p q r : ℚ), B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 ∧ 
  p = -3 ∧ q = 3 ∧ r = -9 := by
  sorry

end matrix_equation_solution_l3750_375026


namespace algebraic_expression_equality_l3750_375035

theorem algebraic_expression_equality (a b : ℝ) (h1 : a = 3) (h2 : a - b = 1) :
  a^2 - a*b = a*(a - b) := by sorry

end algebraic_expression_equality_l3750_375035


namespace geometric_sequence_middle_term_l3750_375097

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (geom_seq : b^2 = a * c)
  (value_a : a = 5 + 2 * Real.sqrt 3)
  (value_c : c = 5 - 2 * Real.sqrt 3) : 
  b = Real.sqrt 13 := by
  sorry

end geometric_sequence_middle_term_l3750_375097


namespace tanning_salon_revenue_l3750_375080

/-- Calculate the revenue of a tanning salon for a calendar month -/
theorem tanning_salon_revenue 
  (first_visit_cost : ℕ) 
  (subsequent_visit_cost : ℕ) 
  (total_customers : ℕ) 
  (second_visit_customers : ℕ) 
  (third_visit_customers : ℕ)
  (h1 : first_visit_cost = 10)
  (h2 : subsequent_visit_cost = 8)
  (h3 : total_customers = 100)
  (h4 : second_visit_customers = 30)
  (h5 : third_visit_customers = 10)
  (h6 : second_visit_customers ≤ total_customers)
  (h7 : third_visit_customers ≤ second_visit_customers) :
  first_visit_cost * total_customers + 
  subsequent_visit_cost * second_visit_customers + 
  subsequent_visit_cost * third_visit_customers = 1320 :=
by sorry

end tanning_salon_revenue_l3750_375080


namespace grandma_olga_grandchildren_l3750_375063

/-- Represents the number of grandchildren Grandma Olga has. -/
def total_grandchildren : ℕ :=
  let daughters := 5
  let sons := 4
  let children_per_daughter := 8 + 7
  let children_per_son := 6 + 3
  daughters * children_per_daughter + sons * children_per_son

/-- Proves that Grandma Olga has 111 grandchildren. -/
theorem grandma_olga_grandchildren : total_grandchildren = 111 := by
  sorry

end grandma_olga_grandchildren_l3750_375063


namespace apple_cost_calculation_l3750_375086

/-- Given that two dozen apples cost $15.60, prove that four dozen apples at the same rate will cost $31.20. -/
theorem apple_cost_calculation (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) :
  let cost_per_dozen : ℝ := cost_two_dozen / 2
  let cost_four_dozen : ℝ := 4 * cost_per_dozen
  cost_four_dozen = 31.20 := by
sorry

end apple_cost_calculation_l3750_375086


namespace remainder_theorem_l3750_375069

theorem remainder_theorem (P D D' Q Q' R R' : ℤ) 
  (h1 : P = Q * D + R) 
  (h2 : 0 ≤ R ∧ R < D) 
  (h3 : Q = Q' * D' + R') 
  (h4 : 0 ≤ R' ∧ R' < D') : 
  P % (D * D') = R + R' * D :=
by sorry

end remainder_theorem_l3750_375069


namespace average_monthly_balance_l3750_375084

def monthly_balances : List ℝ := [120, 150, 180, 150, 210, 180]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 165 := by sorry

end average_monthly_balance_l3750_375084


namespace reflection_maps_correctly_l3750_375053

-- Define points in 2D space
def C : Prod ℝ ℝ := (-3, 2)
def D : Prod ℝ ℝ := (-2, 5)
def C' : Prod ℝ ℝ := (3, -2)
def D' : Prod ℝ ℝ := (2, -5)

-- Define reflection across y = -x
def reflect_across_y_eq_neg_x (p : Prod ℝ ℝ) : Prod ℝ ℝ :=
  (-p.2, -p.1)

-- Theorem statement
theorem reflection_maps_correctly :
  reflect_across_y_eq_neg_x C = C' ∧
  reflect_across_y_eq_neg_x D = D' := by
  sorry

end reflection_maps_correctly_l3750_375053


namespace integer_fractions_l3750_375018

theorem integer_fractions (x : ℤ) : 
  (∃ k : ℤ, (5 * x^3 - x + 17) = 15 * k) ∧ 
  (∃ m : ℤ, (2 * x^2 + x - 3) = 7 * m) ↔ 
  (∃ t : ℤ, x = 105 * t + 22 ∨ x = 105 * t + 37) :=
sorry

end integer_fractions_l3750_375018


namespace hyperbola_intersection_theorem_l3750_375006

/-- Represents a hyperbola with vertices on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a line with slope k passing through (3,0) -/
structure Line where
  k : ℝ

/-- Defines when a line intersects a hyperbola at exactly one point -/
def intersects_at_one_point (h : Hyperbola) (l : Line) : Prop :=
  ∃! x y : ℝ, x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ y = l.k * (x - 3)

/-- The main theorem to be proved -/
theorem hyperbola_intersection_theorem (h : Hyperbola) (l : Line) :
  h.a = 4 ∧ h.b = 3 →
  intersects_at_one_point h l ↔ 
    l.k = 3/4 ∨ l.k = -3/4 ∨ l.k = 3*Real.sqrt 7/7 ∨ l.k = -3*Real.sqrt 7/7 := by
  sorry

end hyperbola_intersection_theorem_l3750_375006


namespace rectangular_solid_surface_area_l3750_375031

-- Define a structure for a rectangular solid
structure RectangularSolid where
  a : ℕ
  b : ℕ
  c : ℕ

-- Define primality
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define volume
def volume (solid : RectangularSolid) : ℕ := solid.a * solid.b * solid.c

-- Define surface area
def surfaceArea (solid : RectangularSolid) : ℕ :=
  2 * (solid.a * solid.b + solid.b * solid.c + solid.c * solid.a)

-- The main theorem
theorem rectangular_solid_surface_area :
  ∀ solid : RectangularSolid,
    isPrime solid.a ∧ isPrime solid.b ∧ isPrime solid.c →
    volume solid = 221 →
    surfaceArea solid = 502 := by
  sorry

end rectangular_solid_surface_area_l3750_375031


namespace root_difference_range_l3750_375054

/-- Given a quadratic function f(x) = ax² + (b-a)x + (c-b) where a > b > c and a + b + c = 0,
    the absolute difference between its roots |x₁ - x₂| lies in the open interval (3/2, 2√3). -/
theorem root_difference_range (a b c : ℝ) (ha : a > b) (hb : b > c) (hsum : a + b + c = 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + (b - a) * x + (c - b)
  let x₁ := (-(b - a) + Real.sqrt ((b - a)^2 - 4 * a * (c - b))) / (2 * a)
  let x₂ := (-(b - a) - Real.sqrt ((b - a)^2 - 4 * a * (c - b))) / (2 * a)
  3/2 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3 := by
  sorry

end root_difference_range_l3750_375054


namespace exists_zero_sum_assignment_l3750_375037

/-- A regular 2n-gon -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- An arrow assignment for a regular 2n-gon -/
def ArrowAssignment (n : ℕ) := 
  (Fin (2*n) × Fin (2*n)) → ℝ × ℝ

/-- The sum of vectors in an arrow assignment -/
def sumVectors (n : ℕ) (assignment : ArrowAssignment n) : ℝ × ℝ := sorry

/-- Theorem stating the existence of a zero-sum arrow assignment -/
theorem exists_zero_sum_assignment (n : ℕ) (polygon : RegularPolygon n) :
  ∃ (assignment : ArrowAssignment n), sumVectors n assignment = (0, 0) := by sorry

end exists_zero_sum_assignment_l3750_375037


namespace doll_count_l3750_375085

theorem doll_count (vera sophie aida : ℕ) : 
  vera = 20 → 
  sophie = 2 * vera → 
  aida = 2 * sophie → 
  vera + sophie + aida = 140 := by
sorry

end doll_count_l3750_375085


namespace billboard_average_l3750_375027

theorem billboard_average (h1 h2 h3 : ℕ) (h1_val : h1 = 17) (h2_val : h2 = 20) (h3_val : h3 = 23) :
  (h1 + h2 + h3) / 3 = 20 := by
  sorry

end billboard_average_l3750_375027


namespace perpendicular_lines_l3750_375012

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line (m b : ℝ) : ℝ := m

/-- The slope of a line in the form ax + y + c = 0 is -a -/
def slope_of_general_line (a c : ℝ) : ℝ := -a

theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope_of_general_line a (-5)) (slope_of_line 7 (-2)) → a = 1/7 := by
  sorry

end perpendicular_lines_l3750_375012


namespace fraction_equality_l3750_375001

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + 2 * y) / (2 * x - 5 * y) = 3) : 
  (x + 3 * y) / (3 * x - y) = 2 / 5 := by
  sorry

end fraction_equality_l3750_375001


namespace more_girls_than_boys_l3750_375038

theorem more_girls_than_boys (boys girls : ℕ) : 
  boys = 40 →
  girls * 5 = boys * 13 →
  girls > boys →
  girls - boys = 64 :=
by
  sorry

end more_girls_than_boys_l3750_375038


namespace cos_seven_pi_fourth_l3750_375079

theorem cos_seven_pi_fourth : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := by
  sorry

end cos_seven_pi_fourth_l3750_375079


namespace tournament_boxes_needed_l3750_375078

/-- A single-elimination tennis tournament -/
structure TennisTournament where
  participants : ℕ
  boxes_per_match : ℕ

/-- The number of boxes needed for a single-elimination tournament -/
def boxes_needed (t : TennisTournament) : ℕ :=
  t.participants - 1

/-- Theorem: A single-elimination tournament with 199 participants needs 198 boxes -/
theorem tournament_boxes_needed :
  ∀ t : TennisTournament, t.participants = 199 ∧ t.boxes_per_match = 1 →
  boxes_needed t = 198 :=
by sorry

end tournament_boxes_needed_l3750_375078


namespace song_duration_l3750_375000

theorem song_duration (initial_songs : ℕ) (added_songs : ℕ) (total_time : ℕ) :
  initial_songs = 25 →
  added_songs = 10 →
  total_time = 105 →
  (initial_songs + added_songs) * (total_time / (initial_songs + added_songs)) = total_time →
  total_time / (initial_songs + added_songs) = 3 :=
by sorry

end song_duration_l3750_375000


namespace leonardo_earnings_l3750_375039

/-- Calculates the total earnings for Leonardo over two weeks given the following conditions:
  * Leonardo worked 18 hours in the second week
  * Leonardo worked 13 hours in the first week
  * Leonardo earned $65.70 more in the second week than in the first week
  * His hourly wage remained the same throughout both weeks
-/
def total_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) : ℚ :=
  let hourly_wage := extra_earnings / (hours_week2 - hours_week1 : ℚ)
  (hours_week1 + hours_week2 : ℚ) * hourly_wage

/-- The theorem states that given the specific conditions in the problem,
    Leonardo's total earnings for the two weeks is $407.34. -/
theorem leonardo_earnings :
  total_earnings 13 18 65.70 = 407.34 := by
  sorry

end leonardo_earnings_l3750_375039


namespace determine_contents_l3750_375062

-- Define the colors of balls
inductive Color
| White
| Black

-- Define the types of boxes
inductive BoxType
| TwoWhite
| TwoBlack
| OneWhiteOneBlack

-- Define a box with a label and contents
structure Box where
  label : BoxType
  contents : BoxType

-- Define the problem setup
def problem_setup : Prop :=
  ∃ (box1 box2 box3 : Box),
    -- Three boxes with different labels
    box1.label ≠ box2.label ∧ box2.label ≠ box3.label ∧ box1.label ≠ box3.label ∧
    -- Contents don't match labels
    box1.contents ≠ box1.label ∧ box2.contents ≠ box2.label ∧ box3.contents ≠ box3.label ∧
    -- One box has two white balls, one has two black balls, and one has one of each
    (box1.contents = BoxType.TwoWhite ∧ box2.contents = BoxType.TwoBlack ∧ box3.contents = BoxType.OneWhiteOneBlack) ∨
    (box1.contents = BoxType.TwoWhite ∧ box2.contents = BoxType.OneWhiteOneBlack ∧ box3.contents = BoxType.TwoBlack) ∨
    (box1.contents = BoxType.TwoBlack ∧ box2.contents = BoxType.TwoWhite ∧ box3.contents = BoxType.OneWhiteOneBlack) ∨
    (box1.contents = BoxType.TwoBlack ∧ box2.contents = BoxType.OneWhiteOneBlack ∧ box3.contents = BoxType.TwoWhite) ∨
    (box1.contents = BoxType.OneWhiteOneBlack ∧ box2.contents = BoxType.TwoWhite ∧ box3.contents = BoxType.TwoBlack) ∨
    (box1.contents = BoxType.OneWhiteOneBlack ∧ box2.contents = BoxType.TwoBlack ∧ box3.contents = BoxType.TwoWhite)

-- Define the theorem
theorem determine_contents (setup : problem_setup) :
  ∃ (box : Box) (c : Color),
    box.label = BoxType.OneWhiteOneBlack →
    (c = Color.White → box.contents = BoxType.TwoWhite) ∧
    (c = Color.Black → box.contents = BoxType.TwoBlack) :=
sorry

end determine_contents_l3750_375062


namespace f_composition_value_l3750_375059

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_composition_value : f (3 * f (-1)) = 6 := by
  sorry

end f_composition_value_l3750_375059


namespace system_solutions_l3750_375081

theorem system_solutions (x y z : ℤ) : 
  x^2 - 9*y^2 - z^2 = 0 ∧ z = x - 3*y →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 3 ∧ y = 1 ∧ z = 0) ∨
  (x = 9 ∧ y = 3 ∧ z = 0) := by
sorry

end system_solutions_l3750_375081


namespace total_rainfall_sum_l3750_375004

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℝ := 0.17

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℝ := 0.42

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℝ := 0.08

/-- The total rainfall recorded over the three days -/
def total_rainfall : ℝ := monday_rainfall + tuesday_rainfall + wednesday_rainfall

/-- Theorem stating that the total rainfall is equal to 0.67 cm -/
theorem total_rainfall_sum : total_rainfall = 0.67 := by sorry

end total_rainfall_sum_l3750_375004


namespace exponential_function_fixed_point_l3750_375094

/-- The function f(x) = a^(x-2) + 1 passes through the point (2, 2) for any a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end exponential_function_fixed_point_l3750_375094


namespace cubic_roots_sum_cubes_l3750_375087

theorem cubic_roots_sum_cubes (r s t : ℝ) : 
  (8 * r^3 + 1001 * r + 2008 = 0) →
  (8 * s^3 + 1001 * s + 2008 = 0) →
  (8 * t^3 + 1001 * t + 2008 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := by
  sorry

end cubic_roots_sum_cubes_l3750_375087


namespace lawn_width_is_30_l3750_375003

/-- Represents the dimensions and properties of a rectangular lawn with roads --/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  gravel_cost_per_sqm : ℝ
  total_gravel_cost : ℝ

/-- Calculates the total area of the roads on the lawn --/
def road_area (l : LawnWithRoads) : ℝ :=
  l.length * l.road_width + (l.width - l.road_width) * l.road_width

/-- Theorem stating that the width of the lawn is 30 meters --/
theorem lawn_width_is_30 (l : LawnWithRoads) 
  (h1 : l.length = 70)
  (h2 : l.road_width = 5)
  (h3 : l.gravel_cost_per_sqm = 4)
  (h4 : l.total_gravel_cost = 1900)
  : l.width = 30 := by
  sorry

end lawn_width_is_30_l3750_375003


namespace max_ratio_x_y_l3750_375047

theorem max_ratio_x_y (x y a b : ℝ) : 
  x ≥ y ∧ y > 0 →
  0 ≤ a ∧ a ≤ x →
  0 ≤ b ∧ b ≤ y →
  (x - a)^2 + (y - b)^2 = x^2 + b^2 →
  (x - a)^2 + (y - b)^2 = y^2 + a^2 →
  ∃ (c : ℝ), c = x / y ∧ c ≤ 2 * Real.sqrt 3 / 3 ∧
  ∀ (d : ℝ), d = x / y → d ≤ c :=
by sorry

end max_ratio_x_y_l3750_375047


namespace binary_110011_equals_51_l3750_375061

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_equals_51_l3750_375061


namespace tangent_line_to_logarithmic_curve_l3750_375075

theorem tangent_line_to_logarithmic_curve : ∃ (n : ℕ+) (a : ℝ), 
  (n : ℝ) < a ∧ a < (n : ℝ) + 1 ∧
  (∃ (x : ℝ), x > 0 ∧ x + 1 = a * Real.log x ∧ 1 = a / x) ∧
  n = 3 := by
  sorry

end tangent_line_to_logarithmic_curve_l3750_375075


namespace unique_two_digit_number_l3750_375070

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_are_different (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens + ones

theorem unique_two_digit_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            digits_are_different n ∧ 
            n^2 = (sum_of_digits n)^3 ∧
            n = 27 :=
sorry

end unique_two_digit_number_l3750_375070


namespace existence_of_special_integers_l3750_375046

/-- There exist positive integers a and b with a > b > 1 such that
    for all positive integers k, there exists a positive integer n
    where an + b is a k-th power of a positive integer. -/
theorem existence_of_special_integers : ∃ (a b : ℕ), 
  a > b ∧ b > 1 ∧ 
  ∀ (k : ℕ), k > 0 → 
    ∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ a * n + b = m ^ k :=
sorry

end existence_of_special_integers_l3750_375046
