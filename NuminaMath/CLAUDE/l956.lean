import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l956_95623

/-- If three lines ax + y + 1 = 0, y = 3x, and x + y = 4 intersect at one point, then a = -4 -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, a * p.1 + p.2 + 1 = 0 ∧ p.2 = 3 * p.1 ∧ p.1 + p.2 = 4) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l956_95623


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l956_95644

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l956_95644


namespace NUMINAMATH_CALUDE_f_properties_l956_95665

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 1/a|

theorem f_properties :
  (∀ x : ℝ, (f 1 x < x + 3) ↔ (x > -3/4 ∧ x < 3/2)) ∧
  (∀ a : ℝ, a > 0 → ∀ x : ℝ, f a x ≥ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l956_95665


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l956_95648

theorem rectangle_area_with_hole (x : ℝ) (h : x > 1.5) :
  (x + 10) * (x + 8) - (2 * x) * (x + 1) = -x^2 + 16*x + 80 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l956_95648


namespace NUMINAMATH_CALUDE_kindergarten_distribution_l956_95663

def apples : ℕ := 270
def pears : ℕ := 180
def oranges : ℕ := 235

def is_valid_distribution (n : ℕ) : Prop :=
  n ≠ 0 ∧
  (apples - n * (apples / n) : ℤ) = 3 * (oranges - n * (oranges / n)) ∧
  (pears - n * (pears / n) : ℤ) = 2 * (oranges - n * (oranges / n))

theorem kindergarten_distribution :
  ∃ (n : ℕ), is_valid_distribution n ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_distribution_l956_95663


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l956_95684

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l956_95684


namespace NUMINAMATH_CALUDE_inequality_equivalence_l956_95631

theorem inequality_equivalence :
  ∀ y : ℝ, (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ ((7 ≤ y ∧ y ≤ 11) ∨ (-3 ≤ y ∧ y ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l956_95631


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l956_95614

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  3 * (apples / 12)

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l956_95614


namespace NUMINAMATH_CALUDE_first_time_below_397_l956_95661

def countingOff (n : ℕ) : ℕ := n - (n / 3)

def remainingStudents (initialCount : ℕ) (rounds : ℕ) : ℕ :=
  match rounds with
  | 0 => initialCount
  | n + 1 => countingOff (remainingStudents initialCount n)

theorem first_time_below_397 (initialCount : ℕ) (h : initialCount = 2010) :
  remainingStudents initialCount 5 ≤ 397 ∧
  ∀ k < 5, remainingStudents initialCount k > 397 :=
sorry

end NUMINAMATH_CALUDE_first_time_below_397_l956_95661


namespace NUMINAMATH_CALUDE_quadratic_trinomials_sum_l956_95629

theorem quadratic_trinomials_sum (p q : ℝ) : 
  (∃! x, 2 * x^2 + (p + q) * x + (p + q) = 0) →
  (2 + (p + q) + (p + q) = 2 ∨ 2 + (p + q) + (p + q) = 18) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomials_sum_l956_95629


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_factorization_4_factorization_5_l956_95621

-- 1. a³ - 9a = a(a + 3)(a - 3)
theorem factorization_1 (a : ℝ) : a^3 - 9*a = a*(a + 3)*(a - 3) := by sorry

-- 2. 3x² - 6xy + x = x(3x - 6y + 1)
theorem factorization_2 (x y : ℝ) : 3*x^2 - 6*x*y + x = x*(3*x - 6*y + 1) := by sorry

-- 3. n²(m - 2) + n(2 - m) = n(m - 2)(n - 1)
theorem factorization_3 (m n : ℝ) : n^2*(m - 2) + n*(2 - m) = n*(m - 2)*(n - 1) := by sorry

-- 4. -4x² + 4xy + y² = [(2 + 2√2)x + y][(2 - 2√2)x + y]
theorem factorization_4 (x y : ℝ) : 
  -4*x^2 + 4*x*y + y^2 = ((2 + 2*Real.sqrt 2)*x + y)*((2 - 2*Real.sqrt 2)*x + y) := by sorry

-- 5. a² + 2a - 8 = (a - 2)(a + 4)
theorem factorization_5 (a : ℝ) : a^2 + 2*a - 8 = (a - 2)*(a + 4) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_factorization_4_factorization_5_l956_95621


namespace NUMINAMATH_CALUDE_actual_miles_traveled_l956_95618

/-- Represents an odometer that skips the digit 7 -/
structure FaultyOdometer where
  current_reading : Nat
  skipped_digit : Nat

/-- Calculates the number of skipped readings up to a given number -/
def count_skipped_readings (n : Nat) : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem: The actual miles traveled when the faulty odometer reads 003008 is 2194 -/
theorem actual_miles_traveled (o : FaultyOdometer) 
  (h1 : o.current_reading = 3008)
  (h2 : o.skipped_digit = 7) : 
  o.current_reading - count_skipped_readings o.current_reading = 2194 := by
  sorry

end NUMINAMATH_CALUDE_actual_miles_traveled_l956_95618


namespace NUMINAMATH_CALUDE_magic_card_profit_l956_95610

/-- Calculates the profit from selling a Magic card that triples in value -/
theorem magic_card_profit (initial_cost : ℝ) : 
  initial_cost > 0 → 2 * initial_cost = (3 * initial_cost) - initial_cost := by
  sorry

#check magic_card_profit

end NUMINAMATH_CALUDE_magic_card_profit_l956_95610


namespace NUMINAMATH_CALUDE_function_properties_l956_95659

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- Theorem statement
theorem function_properties :
  ∀ (a b : ℝ),
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  (∃ (c : ℝ), ∀ x : ℝ, -3 * x^2 + 5 * x + c ≤ 0) →
  (∃ (y : ℝ), ∀ x > -1, (f (-3) 5 x - 21) / (x + 1) ≤ y ∧ 
    ∃ x₀ > -1, (f (-3) 5 x₀ - 21) / (x₀ + 1) = y) →
  (∀ x : ℝ, f a b x = -3 * x^2 + 3 * x + 18) ∧
  (∀ c : ℝ, (∀ x : ℝ, -3 * x^2 + 5 * x + c ≤ 0) → c ≤ -25/12) ∧
  (∀ x > -1, (f (-3) 5 x - 21) / (x + 1) ≤ -3 ∧ 
    ∃ x₀ > -1, (f (-3) 5 x₀ - 21) / (x₀ + 1) = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l956_95659


namespace NUMINAMATH_CALUDE_original_number_proof_l956_95626

/-- Given a three-digit number abc and N = 3194, where N is the sum of acb, bac, bca, cab, and cba, prove that abc = 358 -/
theorem original_number_proof (a b c : ℕ) (h1 : a ≠ 0) 
  (h2 : a * 100 + b * 10 + c < 1000) 
  (h3 : 3194 = (a * 100 + c * 10 + b) + (b * 100 + a * 10 + c) + 
               (b * 100 + c * 10 + a) + (c * 100 + a * 10 + b) + 
               (c * 100 + b * 10 + a)) : 
  a * 100 + b * 10 + c = 358 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l956_95626


namespace NUMINAMATH_CALUDE_geometric_series_sum_quarter_five_terms_l956_95679

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_quarter_five_terms :
  geometric_series_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_quarter_five_terms_l956_95679


namespace NUMINAMATH_CALUDE_oil_bill_ratio_l956_95620

/-- The oil bill problem -/
theorem oil_bill_ratio : 
  ∀ (feb_bill jan_bill : ℚ),
  jan_bill = 180 →
  (feb_bill + 45) / jan_bill = 3 / 2 →
  feb_bill / jan_bill = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_l956_95620


namespace NUMINAMATH_CALUDE_correct_product_after_reversal_error_l956_95624

-- Define a function to reverse digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

-- Define the theorem
theorem correct_product_after_reversal_error (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverseDigits a * b = 221) →  -- erroneous product is 221
  (a * b = 923) :=  -- correct product is 923
by sorry

end NUMINAMATH_CALUDE_correct_product_after_reversal_error_l956_95624


namespace NUMINAMATH_CALUDE_probability_rain_given_strong_winds_l956_95681

theorem probability_rain_given_strong_winds 
  (p_strong_winds : ℝ) 
  (p_rain : ℝ) 
  (p_both : ℝ) 
  (h1 : p_strong_winds = 0.4) 
  (h2 : p_rain = 0.5) 
  (h3 : p_both = 0.3) : 
  p_both / p_strong_winds = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_given_strong_winds_l956_95681


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l956_95602

/-- Given an ellipse and a hyperbola with the same foci, prove that the semi-major axis of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 4 = 1) →  -- Ellipse equation
  (∀ x y : ℝ, x^2 / 9 - y^2 / 3 = 1) →   -- Hyperbola equation
  (a > 0) →                              -- a is positive
  (a^2 - 4 = 12) →                       -- Same foci condition
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l956_95602


namespace NUMINAMATH_CALUDE_houses_with_both_pets_l956_95641

theorem houses_with_both_pets (total : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h_total : total = 60) 
  (h_dogs : dogs = 40) 
  (h_cats : cats = 30) : 
  dogs + cats - total = 10 := by
  sorry

end NUMINAMATH_CALUDE_houses_with_both_pets_l956_95641


namespace NUMINAMATH_CALUDE_hanoi_theorem_l956_95657

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks -/
def hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks
    when direct movement between pegs 1 and 3 is prohibited -/
def hanoi_moves_restricted (n : ℕ) : ℕ := 3^n - 1

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks
    when the smallest disk cannot be placed on peg 2 -/
def hanoi_moves_no_small_on_middle (n : ℕ) : ℕ := 2 * 3^(n-1) - 1

theorem hanoi_theorem (n : ℕ) :
  (hanoi_moves n = 2^n - 1) ∧
  (hanoi_moves_restricted n = 3^n - 1) ∧
  (hanoi_moves_no_small_on_middle n = 2 * 3^(n-1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_hanoi_theorem_l956_95657


namespace NUMINAMATH_CALUDE_second_quadrant_complex_l956_95653

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem second_quadrant_complex :
  let z : ℂ := -1
  is_in_second_quadrant ((2 - Complex.I) * z) := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_complex_l956_95653


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l956_95643

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  (a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) →
  (c = Real.sqrt 7) →
  (b = 2) →
  -- Conclusions
  (C = 2 * Real.pi / 3) ∧
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l956_95643


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l956_95667

/-- Given a principal amount and an interest rate, if the simple interest for 2 years is 660
    and the compound interest for 2 years is 696.30, then the interest rate is 11%. -/
theorem interest_rate_calculation (P R : ℝ) : 
  P * R * 2 / 100 = 660 →
  P * ((1 + R / 100)^2 - 1) = 696.30 →
  R = 11 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l956_95667


namespace NUMINAMATH_CALUDE_gcd_18_30_l956_95686

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l956_95686


namespace NUMINAMATH_CALUDE_basketball_win_rate_l956_95654

theorem basketball_win_rate (initial_wins initial_games remaining_games : ℕ) 
  (h1 : initial_wins = 45)
  (h2 : initial_games = 60)
  (h3 : remaining_games = 50) :
  ∃ (remaining_wins : ℕ), 
    (initial_wins + remaining_wins : ℚ) / (initial_games + remaining_games) = 3/4 ∧ 
    remaining_wins = 38 := by
  sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l956_95654


namespace NUMINAMATH_CALUDE_problem_solution_l956_95608

theorem problem_solution (a b c x : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) ≠ 0)
  (h2 : a^2 / (a + b) = a^2 / (a + c) + 20)
  (h3 : b^2 / (b + c) = b^2 / (b + a) + 14)
  (h4 : c^2 / (c + a) = c^2 / (c + b) + x) :
  x = -34 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l956_95608


namespace NUMINAMATH_CALUDE_coordinates_sum_of_X_l956_95666

def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := (3, 1)
def Z : ℝ × ℝ := (-1, 5)

theorem coordinates_sum_of_X :
  (X.1 + X.2 = 4) ∧
  (‖Z - X‖ / ‖Y - X‖ = 1/2) ∧
  (‖Y - Z‖ / ‖Y - X‖ = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_X_l956_95666


namespace NUMINAMATH_CALUDE_no_solution_iff_m_geq_three_l956_95628

theorem no_solution_iff_m_geq_three (m : ℝ) :
  (∀ x : ℝ, ¬(x - m ≥ 0 ∧ (1/2) * x + (1/2) < 2)) ↔ m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_geq_three_l956_95628


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l956_95688

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area_ratio : ℝ
  total_length : ℝ
  h : AB > 0
  i : CD > 0
  j : area_ratio = 5 / 2
  k : AB + CD = total_length

/-- Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to ADC is 5:2,
    and AB + CD = 280, then AB = 200 -/
theorem trapezoid_segment_length (t : Trapezoid) (h : t.total_length = 280) : t.AB = 200 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l956_95688


namespace NUMINAMATH_CALUDE_sine_graph_shift_l956_95693

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x + π/8)) = 3 * Real.sin (2 * x + π/4) :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l956_95693


namespace NUMINAMATH_CALUDE_copy_machine_rate_l956_95605

/-- Given two copy machines working together for 30 minutes to produce 2850 copies,
    with one machine producing 55 copies per minute, prove that the other machine
    produces 40 copies per minute. -/
theorem copy_machine_rate : ∀ (rate1 : ℕ),
  (30 * rate1 + 30 * 55 = 2850) → rate1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_copy_machine_rate_l956_95605


namespace NUMINAMATH_CALUDE_unpaired_numbers_mod_6_l956_95646

theorem unpaired_numbers_mod_6 (n : ℕ) (hn : n = 800) : 
  ¬ (∃ (f : ℕ → ℕ), 
    (∀ x ∈ Finset.range n, f (f x) = x ∧ x ≠ f x) ∧ 
    (∀ x ∈ Finset.range n, (x + f x) % 6 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_unpaired_numbers_mod_6_l956_95646


namespace NUMINAMATH_CALUDE_total_expense_calculation_l956_95673

/-- Sandy's current age -/
def sandy_age : ℕ := 34

/-- Kim's current age -/
def kim_age : ℕ := 10

/-- Alex's current age -/
def alex_age : ℕ := sandy_age / 2

/-- Sandy's monthly phone bill expense -/
def sandy_expense : ℕ := 10 * sandy_age

/-- Alex's monthly expense next month -/
def alex_expense : ℕ := 2 * sandy_expense

theorem total_expense_calculation :
  sandy_age = 34 ∧
  kim_age = 10 ∧
  alex_age = sandy_age / 2 ∧
  sandy_expense = 10 * sandy_age ∧
  alex_expense = 2 * sandy_expense ∧
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_expense + alex_expense = 1020 := by
  sorry

end NUMINAMATH_CALUDE_total_expense_calculation_l956_95673


namespace NUMINAMATH_CALUDE_cakes_served_at_lunch_l956_95613

theorem cakes_served_at_lunch (total : ℕ) (dinner : ℕ) (yesterday : ℕ) 
  (h1 : total = 14) 
  (h2 : dinner = 6) 
  (h3 : yesterday = 3) : 
  total - dinner - yesterday = 5 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_at_lunch_l956_95613


namespace NUMINAMATH_CALUDE_chess_game_probabilities_l956_95670

/-- The probability of winning a single game -/
def prob_win : ℝ := 0.4

/-- The probability of not losing a single game -/
def prob_not_lose : ℝ := 0.9

/-- The probability of a draw in a single game -/
def prob_draw : ℝ := prob_not_lose - prob_win

/-- The probability of winning at least one game out of two independent games -/
def prob_win_at_least_one : ℝ := 1 - (1 - prob_win) ^ 2

theorem chess_game_probabilities :
  prob_draw = 0.5 ∧ prob_win_at_least_one = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probabilities_l956_95670


namespace NUMINAMATH_CALUDE_simplify_expression_l956_95647

theorem simplify_expression (a b : ℝ) : (15*a + 45*b) + (21*a + 32*b) - (12*a + 40*b) = 24*a + 37*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l956_95647


namespace NUMINAMATH_CALUDE_tan_negative_255_degrees_l956_95649

theorem tan_negative_255_degrees : Real.tan (-(255 * π / 180)) = Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_255_degrees_l956_95649


namespace NUMINAMATH_CALUDE_det_A_eq_neg_46_l956_95650

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 5; 0, 6, -2; 3, -1, 2]

theorem det_A_eq_neg_46 : Matrix.det A = -46 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_neg_46_l956_95650


namespace NUMINAMATH_CALUDE_average_monthly_bill_l956_95698

theorem average_monthly_bill (first_four_months_avg : ℝ) (last_two_months_avg : ℝ) :
  first_four_months_avg = 30 →
  last_two_months_avg = 24 →
  (4 * first_four_months_avg + 2 * last_two_months_avg) / 6 = 28 :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_bill_l956_95698


namespace NUMINAMATH_CALUDE_roque_walking_time_l956_95638

/-- The time it takes Roque to walk to work -/
def walking_time : ℝ := sorry

/-- The time it takes Roque to bike to work -/
def biking_time : ℝ := 1

/-- Number of times Roque walks to and from work per week -/
def walks_per_week : ℕ := 3

/-- Number of times Roque bikes to and from work per week -/
def bikes_per_week : ℕ := 2

/-- Total commuting time in a week -/
def total_commute_time : ℝ := 16

theorem roque_walking_time :
  walking_time = 2 :=
by
  have h1 : (2 * walking_time * walks_per_week) + (2 * biking_time * bikes_per_week) = total_commute_time := by sorry
  sorry

end NUMINAMATH_CALUDE_roque_walking_time_l956_95638


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_13_4_l956_95617

theorem floor_plus_x_eq_13_4 :
  ∃! x : ℝ, ⌊x⌋ + x = 13.4 ∧ x = 6.4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_13_4_l956_95617


namespace NUMINAMATH_CALUDE_total_cookies_l956_95601

theorem total_cookies (chris kenny glenn : ℕ) : 
  chris = kenny / 2 →
  glenn = 4 * kenny →
  glenn = 24 →
  chris + kenny + glenn = 33 := by
sorry

end NUMINAMATH_CALUDE_total_cookies_l956_95601


namespace NUMINAMATH_CALUDE_prime_between_30_and_40_with_specific_remainder_l956_95680

theorem prime_between_30_and_40_with_specific_remainder : 
  {n : ℕ | 30 ≤ n ∧ n ≤ 40 ∧ Prime n ∧ 1 ≤ n % 7 ∧ n % 7 ≤ 6} = {31, 37} := by
  sorry

end NUMINAMATH_CALUDE_prime_between_30_and_40_with_specific_remainder_l956_95680


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l956_95611

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 11 ∧ x₂ = 2 - Real.sqrt 11 ∧
    x₁^2 - 4*x₁ - 7 = 0 ∧ x₂^2 - 4*x₂ - 7 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -2 ∧
    3*y₁^2 + 5*y₁ - 2 = 0 ∧ 3*y₂^2 + 5*y₂ - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l956_95611


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l956_95691

theorem positive_integer_solutions_count : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ p.1 > 0 ∧ p.2 > 0 ∧ (4 : ℚ) / p.1 + (2 : ℚ) / p.2 = 1) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l956_95691


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_12_l956_95635

theorem circle_area_with_diameter_12 (π : Real) (diameter : Real) (area : Real) :
  diameter = 12 →
  area = π * (diameter / 2)^2 →
  area = π * 36 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_12_l956_95635


namespace NUMINAMATH_CALUDE_find_N_l956_95615

theorem find_N : ∃ N : ℕ, 
  (87^2 - 78^2) % N = 0 ∧ 
  45 < N ∧ 
  N < 100 ∧ 
  (N = 55 ∨ N = 99) := by
sorry

end NUMINAMATH_CALUDE_find_N_l956_95615


namespace NUMINAMATH_CALUDE_green_peaches_count_l956_95633

/-- The number of green peaches in a basket, given the total number of peaches and the number of red peaches. -/
def num_green_peaches (total : ℕ) (red : ℕ) : ℕ :=
  total - red

/-- Theorem stating that there are 3 green peaches in the basket. -/
theorem green_peaches_count :
  let total := 16
  let red := 13
  num_green_peaches total red = 3 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l956_95633


namespace NUMINAMATH_CALUDE_excess_meat_sold_proof_l956_95640

/-- Calculates the excess meat sold beyond the original plan. -/
def excess_meat_sold (thursday_sales : ℕ) (saturday_sales : ℕ) (original_plan : ℕ) : ℕ :=
  let friday_sales := 2 * thursday_sales
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  total_sales - original_plan

/-- Proves that the excess meat sold beyond the original plan is 325 kg. -/
theorem excess_meat_sold_proof :
  excess_meat_sold 210 130 500 = 325 := by
  sorry

end NUMINAMATH_CALUDE_excess_meat_sold_proof_l956_95640


namespace NUMINAMATH_CALUDE_length_A_l956_95694

-- Define the points
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 7)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the condition that A' and B' are on the line y = x
axiom A'_on_line : ∃ A' : ℝ × ℝ, line_y_eq_x A'
axiom B'_on_line : ∃ B' : ℝ × ℝ, line_y_eq_x B'

-- Define the condition that AA' and BB' intersect at C
axiom AA'_BB'_intersect_at_C : 
  ∃ A' B' : ℝ × ℝ, line_y_eq_x A' ∧ line_y_eq_x B' ∧
  (∃ t₁ t₂ : ℝ, A + t₁ • (A' - A) = C ∧ B + t₂ • (B' - B) = C)

-- State the theorem
theorem length_A'B'_is_4_sqrt_2 : 
  ∃ A' B' : ℝ × ℝ, line_y_eq_x A' ∧ line_y_eq_x B' ∧
  (∃ t₁ t₂ : ℝ, A + t₁ • (A' - A) = C ∧ B + t₂ • (B' - B) = C) ∧
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_length_A_l956_95694


namespace NUMINAMATH_CALUDE_fonzie_payment_l956_95672

/-- Proves that Fonzie's payment for the treasure map is $7000 -/
theorem fonzie_payment (fonzie_payment : ℝ) : 
  (∀ total_payment : ℝ, 
    total_payment = fonzie_payment + 8000 + 9000 ∧ 
    9000 / total_payment = 337500 / 900000) →
  fonzie_payment = 7000 := by
sorry

end NUMINAMATH_CALUDE_fonzie_payment_l956_95672


namespace NUMINAMATH_CALUDE_student_arrangement_count_l956_95682

/-- The number of ways to arrange n elements -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of ways to arrange 5 students in a line -/
def totalArrangements : ℕ := factorial 5

/-- The number of ways to arrange 3 students together and 2 separately -/
def restrictedArrangements : ℕ := factorial 3 * factorial 3

/-- The number of valid arrangements where 3 students are not next to each other -/
def validArrangements : ℕ := totalArrangements - restrictedArrangements

theorem student_arrangement_count :
  validArrangements = 84 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l956_95682


namespace NUMINAMATH_CALUDE_susan_strawberry_eating_l956_95627

theorem susan_strawberry_eating (basket_capacity : ℕ) (total_picked : ℕ) (handful_size : ℕ) :
  basket_capacity = 60 →
  total_picked = 75 →
  handful_size = 5 →
  (total_picked - basket_capacity) / (total_picked / handful_size) = 1 := by
  sorry

end NUMINAMATH_CALUDE_susan_strawberry_eating_l956_95627


namespace NUMINAMATH_CALUDE_determinant_equal_polynomial_l956_95662

variable (x : ℝ)

def matrix : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
  match i, j with
  | 0, 0 => 2*x + 3
  | 0, 1 => x
  | 0, 2 => x
  | 1, 0 => 2*x
  | 1, 1 => 2*x + 3
  | 1, 2 => x
  | 2, 0 => 2*x
  | 2, 1 => x
  | 2, 2 => 2*x + 3

theorem determinant_equal_polynomial (x : ℝ) :
  Matrix.det (matrix x) = 2*x^3 + 27*x^2 + 27*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equal_polynomial_l956_95662


namespace NUMINAMATH_CALUDE_seminar_invitations_count_l956_95636

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 6 teachers out of 10 for a seminar,
    where two specific teachers (A and B) cannot attend together -/
def seminar_invitations : ℕ :=
  2 * binomial 8 5 + binomial 8 6

theorem seminar_invitations_count : seminar_invitations = 140 := by
  sorry

end NUMINAMATH_CALUDE_seminar_invitations_count_l956_95636


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_thirteen_thirds_l956_95690

theorem floor_plus_self_eq_thirteen_thirds (x : ℝ) : 
  (⌊x⌋ : ℝ) + x = 13/3 → x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_thirteen_thirds_l956_95690


namespace NUMINAMATH_CALUDE_circle_center_l956_95695

/-- Given a circle with polar equation ρ = 2cos(θ), its center in the Cartesian coordinate system is at (1,0) -/
theorem circle_center (ρ θ : ℝ) : ρ = 2 * Real.cos θ → ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ (x - 1)^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l956_95695


namespace NUMINAMATH_CALUDE_expression_evaluation_l956_95677

theorem expression_evaluation (x y k : ℤ) 
  (hx : x = 7) (hy : y = 3) (hk : k = 10) : 
  (x - y) * (x + y) + k = 50 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l956_95677


namespace NUMINAMATH_CALUDE_area_with_holes_formula_l956_95637

/-- The area of a rectangle with holes -/
def area_with_holes (x : ℝ) : ℝ :=
  let large_rectangle_area := (x + 8) * (x + 6)
  let hole_area := (2 * x - 4) * (x - 3)
  let total_hole_area := 2 * hole_area
  large_rectangle_area - total_hole_area

/-- Theorem: The area of the rectangle with holes is equal to -3x^2 + 34x + 24 -/
theorem area_with_holes_formula (x : ℝ) :
  area_with_holes x = -3 * x^2 + 34 * x + 24 := by
  sorry

#check area_with_holes_formula

end NUMINAMATH_CALUDE_area_with_holes_formula_l956_95637


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l956_95656

/-- Given two vectors a and b in ℝ², where a is parallel to (b - a), prove that the x-coordinate of a is -2. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (h : a.1 = m ∧ a.2 = 1 ∧ b.1 = 2 ∧ b.2 = -1) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (b - a)) : m = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l956_95656


namespace NUMINAMATH_CALUDE_tv_screen_width_l956_95607

/-- Given a rectangular TV screen with area 21 square feet and height 7 feet, its width is 3 feet. -/
theorem tv_screen_width (area : ℝ) (height : ℝ) (width : ℝ) : 
  area = 21 → height = 7 → area = width * height → width = 3 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_width_l956_95607


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l956_95687

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The fifth term of an arithmetic sequence is 5, given the sum of the first and ninth terms is 10 -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l956_95687


namespace NUMINAMATH_CALUDE_cubic_root_sum_l956_95639

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 4*p^2 + 6*p - 3 = 0 ∧ 
  q^3 - 4*q^2 + 6*q - 3 = 0 ∧ 
  r^3 - 4*r^2 + 6*r - 3 = 0 → 
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l956_95639


namespace NUMINAMATH_CALUDE_equation_solution_l956_95625

theorem equation_solution : ∃! y : ℚ, 7 * (4 * y - 3) + 5 = 3 * (-2 + 8 * y) :=
  by sorry

end NUMINAMATH_CALUDE_equation_solution_l956_95625


namespace NUMINAMATH_CALUDE_daily_earnings_of_c_l956_95685

theorem daily_earnings_of_c (A B C : ℕ) 
  (h1 : A + B + C = 600)
  (h2 : A + C = 400)
  (h3 : B + C = 300) :
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_daily_earnings_of_c_l956_95685


namespace NUMINAMATH_CALUDE_same_ending_squares_l956_95630

theorem same_ending_squares (N : ℕ) (h1 : N > 0) 
  (h2 : ∃ (a b c d e : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    N % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (N * N) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  N % 100000 = 69969 := by
sorry

end NUMINAMATH_CALUDE_same_ending_squares_l956_95630


namespace NUMINAMATH_CALUDE_cereal_expense_per_year_l956_95604

def boxes_per_week : ℕ := 2
def cost_per_box : ℚ := 3
def weeks_per_year : ℕ := 52

theorem cereal_expense_per_year :
  (boxes_per_week * weeks_per_year * cost_per_box : ℚ) = 312 := by
  sorry

end NUMINAMATH_CALUDE_cereal_expense_per_year_l956_95604


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_l956_95645

noncomputable def inscribed_sphere_radius : ℝ := Real.sqrt 6 - 1

theorem circumscribed_sphere_radius
  (inscribed_radius : ℝ)
  (h_inscribed_radius : inscribed_radius = inscribed_sphere_radius)
  (h_touching : inscribed_radius > 0) :
  ∃ (circumscribed_radius : ℝ),
    circumscribed_radius = 5 * (Real.sqrt 2 + 1) * inscribed_radius :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_l956_95645


namespace NUMINAMATH_CALUDE_characterize_function_l956_95678

/-- A strictly increasing function from positive integers to positive integers -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m < n → f m < f n

/-- The main theorem about the structure of functions satisfying f(f(n)) = f(n)^2 -/
theorem characterize_function (f : ℕ+ → ℕ+) 
  (h_incr : StrictlyIncreasing f) 
  (h_eq : ∀ n : ℕ+, f (f n) = (f n)^2) :
  ∃ c : ℕ+, 
    (∀ n : ℕ+, n ≥ 2 → f n = c * n) ∧
    (f 1 = 1 ∨ f 1 = c) :=
sorry

end NUMINAMATH_CALUDE_characterize_function_l956_95678


namespace NUMINAMATH_CALUDE_unique_m_value_l956_95622

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, HasPeriod f q ∧ q > 0 → p ≤ q

theorem unique_m_value (f : ℝ → ℝ) (m : ℝ) :
  IsOdd f →
  SmallestPositivePeriod f 4 →
  f 1 > 1 →
  f 2 = m^2 - 2*m →
  f 3 = (2*m - 5)/(m + 1) →
  m = 0 := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l956_95622


namespace NUMINAMATH_CALUDE_sin_cos_equality_implies_pi_quarter_l956_95619

theorem sin_cos_equality_implies_pi_quarter (x : Real) :
  x ∈ Set.Icc 0 Real.pi →
  Real.sin (x + Real.sin x) = Real.cos (x - Real.cos x) →
  x = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_equality_implies_pi_quarter_l956_95619


namespace NUMINAMATH_CALUDE_adams_money_l956_95642

/-- Adam's money problem --/
theorem adams_money (initial_amount spent allowance : ℕ) :
  initial_amount = 5 →
  spent = 2 →
  allowance = 5 →
  initial_amount - spent + allowance = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_money_l956_95642


namespace NUMINAMATH_CALUDE_initial_persons_count_l956_95664

/-- The initial number of persons in a group where:
    1. The average weight increases by 4.5 kg when a new person joins.
    2. The person being replaced weighs 65 kg.
    3. The new person weighs 101 kg. -/
def initialPersons : ℕ := 8

theorem initial_persons_count :
  let avgWeightIncrease : ℚ := 4.5
  let replacedPersonWeight : ℕ := 65
  let newPersonWeight : ℕ := 101
  let totalWeightIncrease : ℚ := avgWeightIncrease * initialPersons
  totalWeightIncrease = (newPersonWeight - replacedPersonWeight) →
  initialPersons = 8 := by sorry

end NUMINAMATH_CALUDE_initial_persons_count_l956_95664


namespace NUMINAMATH_CALUDE_min_xy_value_l956_95683

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (3 / (2 + x)) + (3 / (2 + y)) = 1) : 
  x * y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (3 / (2 + x)) + (3 / (2 + y)) = 1 ∧ x * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l956_95683


namespace NUMINAMATH_CALUDE_square_root_x_minus_y_l956_95632

theorem square_root_x_minus_y (x y : ℝ) (h : Real.sqrt (x - 2) + (y + 1)^2 = 0) : 
  (Real.sqrt (x - y))^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_x_minus_y_l956_95632


namespace NUMINAMATH_CALUDE_positive_interval_l956_95616

theorem positive_interval (x : ℝ) : (x + 3) * (x - 2) > 0 ↔ x < -3 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_interval_l956_95616


namespace NUMINAMATH_CALUDE_cannot_divide_rectangle_l956_95658

theorem cannot_divide_rectangle : ¬ ∃ (m n : ℕ), 55 = m * 5 ∧ 39 = n * 11 := by
  sorry

end NUMINAMATH_CALUDE_cannot_divide_rectangle_l956_95658


namespace NUMINAMATH_CALUDE_find_number_l956_95634

theorem find_number : ∃ x : ℤ, x + 5 = 9 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_find_number_l956_95634


namespace NUMINAMATH_CALUDE_equation_solution_l956_95696

theorem equation_solution : ∃! x : ℝ, (16 : ℝ)^(x - 1) / (8 : ℝ)^(x - 1) = (64 : ℝ)^(x + 2) ∧ x = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l956_95696


namespace NUMINAMATH_CALUDE_corporate_event_handshakes_eq_430_l956_95609

/-- Represents the number of handshakes at a corporate event --/
def corporate_event_handshakes : ℕ :=
  let total_people : ℕ := 40
  let group_a_size : ℕ := 15
  let group_b_size : ℕ := 20
  let group_c_size : ℕ := 5
  let group_b_knowing_a : ℕ := 5
  let group_b_knowing_none : ℕ := 15

  let handshakes_a_b : ℕ := group_a_size * group_b_knowing_none
  let handshakes_within_b : ℕ := (group_b_knowing_none * (group_b_knowing_none - 1)) / 2
  let handshakes_b_c : ℕ := group_b_size * group_c_size

  handshakes_a_b + handshakes_within_b + handshakes_b_c

/-- Theorem stating that the number of handshakes at the corporate event is 430 --/
theorem corporate_event_handshakes_eq_430 : corporate_event_handshakes = 430 := by
  sorry

end NUMINAMATH_CALUDE_corporate_event_handshakes_eq_430_l956_95609


namespace NUMINAMATH_CALUDE_equality_proof_l956_95600

theorem equality_proof : 2017 - 1 / 2017 = (2018 * 2016) / 2017 := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l956_95600


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l956_95655

/-- Given a geometric sequence {a_n} where all terms are positive and 
    (a₁, ½a₃, 2a₂) forms an arithmetic sequence, 
    prove that (a₉ + a₁₀) / (a₇ + a₈) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →
  (a 1 + 2 * a 2 = a 3) →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l956_95655


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l956_95676

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) : 
  Nat.lcm X Y = 180 → 
  (X : ℚ) / (Y : ℚ) = 2 / 5 → 
  Nat.gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l956_95676


namespace NUMINAMATH_CALUDE_bedroom_wall_area_l956_95674

/-- Calculates the total paintable wall area for multiple identical bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - non_paintable_area
  num_bedrooms * paintable_area

/-- Proves that the total paintable wall area of 4 bedrooms with given dimensions is 1860 square feet -/
theorem bedroom_wall_area : total_paintable_area 4 15 12 10 75 = 1860 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_wall_area_l956_95674


namespace NUMINAMATH_CALUDE_large_circle_diameter_l956_95675

theorem large_circle_diameter (r : ℝ) (h : r = 4) :
  let small_circles_radius := r
  let small_circles_count := 8
  let inner_octagon_side := 2 * small_circles_radius
  let inner_octagon_radius := inner_octagon_side / Real.sqrt 2
  let large_circle_radius := inner_octagon_radius + small_circles_radius
  large_circle_radius * 2 = 8 * Real.sqrt 2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_large_circle_diameter_l956_95675


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_is_tight_l956_95652

theorem min_value_of_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 * Real.sqrt 5 / 5 := by sorry

theorem lower_bound_is_tight : 
  ∃ (x : ℝ), (x^2 + 9) / Real.sqrt (x^2 + 5) = 9 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_is_tight_l956_95652


namespace NUMINAMATH_CALUDE_min_b_value_l956_95689

theorem min_b_value (a b : ℤ) (h1 : 6 < a ∧ a < 17) (h2 : b < 29) 
  (h3 : (16 : ℚ) / b - (7 : ℚ) / 28 = 15/4) : 4 ≤ b := by
  sorry

end NUMINAMATH_CALUDE_min_b_value_l956_95689


namespace NUMINAMATH_CALUDE_inequality_proof_l956_95699

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l956_95699


namespace NUMINAMATH_CALUDE_vectors_are_collinear_l956_95669

/-- Two 2D vectors are collinear if their cross product is zero -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vectors_are_collinear : 
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (6, -4)
  are_collinear a b :=
by sorry

end NUMINAMATH_CALUDE_vectors_are_collinear_l956_95669


namespace NUMINAMATH_CALUDE_product_of_powers_equals_hundred_l956_95697

theorem product_of_powers_equals_hundred : 
  (10 ^ 0.6) * (10 ^ 0.2) * (10 ^ 0.1) * (10 ^ 0.3) * (10 ^ 0.7) * (10 ^ 0.1) = 100 := by
sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_hundred_l956_95697


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l956_95651

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3) ∨ f a x ≥ f a (-3)) →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l956_95651


namespace NUMINAMATH_CALUDE_exist_three_similar_numbers_l956_95668

/-- A function that repeats a given 3-digit number to form a 1995-digit number -/
def repeat_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a 1995-digit number -/
def is_1995_digit (n : ℕ) : Prop := sorry

/-- Predicate to check if two numbers use the same set of digits -/
def same_digit_set (a b : ℕ) : Prop := sorry

/-- Predicate to check if a number contains the digit 0 -/
def contains_zero (n : ℕ) : Prop := sorry

theorem exist_three_similar_numbers :
  ∃ (A B C : ℕ),
    is_1995_digit A ∧
    is_1995_digit B ∧
    is_1995_digit C ∧
    same_digit_set A B ∧
    same_digit_set B C ∧
    ¬contains_zero A ∧
    ¬contains_zero B ∧
    ¬contains_zero C ∧
    A + B = C :=
  sorry

end NUMINAMATH_CALUDE_exist_three_similar_numbers_l956_95668


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l956_95671

theorem paper_strip_dimensions 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Positive dimensions
  (h2 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43)  -- Sum of areas is 43
  : a = 1 ∧ b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l956_95671


namespace NUMINAMATH_CALUDE_function_properties_l956_95692

open Real

theorem function_properties (f : ℝ → ℝ) (k : ℝ) (hf : Differentiable ℝ f) 
  (h0 : f 0 = -1) (hk : k > 1) (hf' : ∀ x, deriv f x > k) :
  (f (1/k) > 1/k - 1) ∧ 
  (f (1/(k-1)) > 1/(k-1)) ∧ 
  (f (1/k) < f (1/(k-1))) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l956_95692


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l956_95660

theorem least_value_x_minus_y_plus_z (x y z : ℕ+) (h : (3 : ℕ) * x.val = (4 : ℕ) * y.val ∧ (4 : ℕ) * y.val = (7 : ℕ) * z.val) :
  (x.val - y.val + z.val : ℤ) ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℕ+), (3 : ℕ) * x₀.val = (4 : ℕ) * y₀.val ∧ (4 : ℕ) * y₀.val = (7 : ℕ) * z₀.val ∧ (x₀.val - y₀.val + z₀.val : ℤ) = 19 :=
sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l956_95660


namespace NUMINAMATH_CALUDE_max_ben_cookies_proof_l956_95603

/-- The maximum number of cookies Ben can eat when sharing with Beth -/
def max_ben_cookies : ℕ := 12

/-- The total number of cookies shared between Ben and Beth -/
def total_cookies : ℕ := 36

/-- Predicate to check if a given number of cookies for Ben is valid -/
def valid_ben_cookies (ben : ℕ) : Prop :=
  (ben + 2 * ben = total_cookies) ∨ (ben + 3 * ben = total_cookies)

theorem max_ben_cookies_proof :
  (∀ ben : ℕ, valid_ben_cookies ben → ben ≤ max_ben_cookies) ∧
  valid_ben_cookies max_ben_cookies :=
sorry

end NUMINAMATH_CALUDE_max_ben_cookies_proof_l956_95603


namespace NUMINAMATH_CALUDE_local_extrema_of_f_l956_95606

def f (x : ℝ) : ℝ := 1 + 3*x - x^3

theorem local_extrema_of_f :
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo (-1 - δ₁) (-1 + δ₁), f x ≥ f (-1)) ∧
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (1 - δ₂) (1 + δ₂), f x ≤ f 1) ∧
  f (-1) = -1 ∧
  f 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_local_extrema_of_f_l956_95606


namespace NUMINAMATH_CALUDE_daisy_rose_dogs_pool_l956_95612

/-- The number of legs/paws in a pool with humans and dogs -/
def legs_paws_in_pool (num_humans : ℕ) (num_dogs : ℕ) : ℕ :=
  num_humans * 2 + num_dogs * 4

/-- Theorem: The number of legs/paws in the pool with Daisy, Rose, and their 5 dogs is 24 -/
theorem daisy_rose_dogs_pool : legs_paws_in_pool 2 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_daisy_rose_dogs_pool_l956_95612
