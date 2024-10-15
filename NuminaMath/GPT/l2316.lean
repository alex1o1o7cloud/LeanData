import Mathlib

namespace NUMINAMATH_GPT_smallest_possible_value_l2316_231629

-- Definitions and conditions provided
def x_plus_4_y_minus_4_eq_zero (x y : ℝ) : Prop := (x + 4) * (y - 4) = 0

-- Main theorem to state
theorem smallest_possible_value (x y : ℝ) (h : x_plus_4_y_minus_4_eq_zero x y) : x^2 + y^2 = 32 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l2316_231629


namespace NUMINAMATH_GPT_people_dislike_both_radio_and_music_l2316_231663

theorem people_dislike_both_radio_and_music :
  let total_people := 1500
  let dislike_radio_percent := 0.35
  let dislike_both_percent := 0.20
  let dislike_radio := dislike_radio_percent * total_people
  let dislike_both := dislike_both_percent * dislike_radio
  dislike_both = 105 :=
by
  sorry

end NUMINAMATH_GPT_people_dislike_both_radio_and_music_l2316_231663


namespace NUMINAMATH_GPT_find_k_l2316_231657

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x + 3
def q (k : ℝ) (x : ℝ) : ℝ := k * x + k
def intersection (x y : ℝ) : Prop := y = p x ∧ ∃ k, y = q k x

-- Proof that based on the intersection at (1, 5), k evaluates to 5/2
theorem find_k : ∃ k : ℝ, intersection 1 5 → k = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_find_k_l2316_231657


namespace NUMINAMATH_GPT_marcus_sees_7_l2316_231637

variable (marcus humphrey darrel : ℕ)
variable (humphrey_sees : humphrey = 11)
variable (darrel_sees : darrel = 9)
variable (average_is_9 : (marcus + humphrey + darrel) / 3 = 9)

theorem marcus_sees_7 : marcus = 7 :=
by
  -- Needs proof
  sorry

end NUMINAMATH_GPT_marcus_sees_7_l2316_231637


namespace NUMINAMATH_GPT_possible_m_value_l2316_231649

variable (a b m t : ℝ)
variable (h_a : a ≠ 0)
variable (h1 : ∃ t, ∀ x, ax^2 - bx ≥ -1 ↔ (x ≤ t - 1 ∨ x ≥ -3 - t))
variable (h2 : a * m^2 - b * m = 2)

theorem possible_m_value : m = 1 :=
sorry

end NUMINAMATH_GPT_possible_m_value_l2316_231649


namespace NUMINAMATH_GPT_max_value_of_quadratic_l2316_231676

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the problem formally in Lean
theorem max_value_of_quadratic : ∀ x : ℝ, quadratic x ≤ quadratic 0 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l2316_231676


namespace NUMINAMATH_GPT_min_value_l2316_231647

theorem min_value (a b c : ℤ) (h : a > b ∧ b > c) :
  ∃ x, x = (a + b + c) / (a - b - c) ∧ 
       x + (a - b - c) / (a + b + c) = 2 := sorry

end NUMINAMATH_GPT_min_value_l2316_231647


namespace NUMINAMATH_GPT_inequality_must_be_true_l2316_231643

theorem inequality_must_be_true (a b : ℝ) (h : a > b ∧ b > 0) :
  a + 1 / b > b + 1 / a :=
sorry

end NUMINAMATH_GPT_inequality_must_be_true_l2316_231643


namespace NUMINAMATH_GPT_two_digit_sum_reverse_l2316_231630

theorem two_digit_sum_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_sum_reverse_l2316_231630


namespace NUMINAMATH_GPT_balls_left_l2316_231648

-- Define the conditions
def initial_balls : ℕ := 10
def removed_balls : ℕ := 3

-- The main statement to prove
theorem balls_left : initial_balls - removed_balls = 7 := by sorry

end NUMINAMATH_GPT_balls_left_l2316_231648


namespace NUMINAMATH_GPT_percentage_meetings_correct_l2316_231605

def work_day_hours : ℕ := 10
def minutes_in_hour : ℕ := 60
def total_work_day_minutes := work_day_hours * minutes_in_hour

def lunch_break_minutes : ℕ := 30
def effective_work_day_minutes := total_work_day_minutes - lunch_break_minutes

def first_meeting_minutes : ℕ := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

def percentage_of_day_spent_in_meetings := (total_meeting_minutes * 100) / effective_work_day_minutes

theorem percentage_meetings_correct : percentage_of_day_spent_in_meetings = 42 := 
by
  sorry

end NUMINAMATH_GPT_percentage_meetings_correct_l2316_231605


namespace NUMINAMATH_GPT_calculate_expression_l2316_231641

-- Definitions based on the conditions
def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℝ) : Prop := c * d = 1
def negative_abs_two (m : ℝ) : Prop := m = -2

-- The main statement to be proved
theorem calculate_expression (a b : ℤ) (c d m : ℝ) 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : negative_abs_two m) : 
  m + c * d + a + b + (c * d) ^ 2010 = 0 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2316_231641


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2316_231674

theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, 0 < n → (a n - 2 * a (n + 1) + a (n + 2) = 0)) : ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2316_231674


namespace NUMINAMATH_GPT_minimize_sum_first_n_terms_l2316_231666

noncomputable def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

noncomputable def sum_first_n_terms (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n-1) / 2) * d

theorem minimize_sum_first_n_terms (a₁ : ℤ) (a₃_plus_a₅ : ℤ) (n_min : ℕ) :
  a₁ = -9 → a₃_plus_a₅ = -6 → n_min = 5 := by
  sorry

end NUMINAMATH_GPT_minimize_sum_first_n_terms_l2316_231666


namespace NUMINAMATH_GPT_sum_prime_factors_of_77_l2316_231697

theorem sum_prime_factors_of_77 : ∃ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 ∧ p1 + p2 = 18 := by
  sorry

end NUMINAMATH_GPT_sum_prime_factors_of_77_l2316_231697


namespace NUMINAMATH_GPT_brown_eyed_brunettes_l2316_231687

theorem brown_eyed_brunettes (total_girls blondes brunettes blue_eyed_blondes brown_eyed_girls : ℕ) 
    (h1 : total_girls = 60) 
    (h2 : blondes + brunettes = total_girls) 
    (h3 : blue_eyed_blondes = 20) 
    (h4 : brunettes = 35) 
    (h5 : brown_eyed_girls = 22) 
    (h6 : blondes = total_girls - brunettes) 
    (h7 : brown_eyed_blondes = blondes - blue_eyed_blondes) :
  brunettes - (brown_eyed_girls - brown_eyed_blondes) = 17 :=
by sorry  -- Proof is not required

end NUMINAMATH_GPT_brown_eyed_brunettes_l2316_231687


namespace NUMINAMATH_GPT_sphere_triangle_distance_l2316_231607

theorem sphere_triangle_distance
  (P X Y Z : Type)
  (radius : ℝ)
  (h1 : radius = 15)
  (dist_XY : ℝ)
  (h2 : dist_XY = 6)
  (dist_YZ : ℝ)
  (h3 : dist_YZ = 8)
  (dist_ZX : ℝ)
  (h4 : dist_ZX = 10)
  (distance_from_P_to_triangle : ℝ)
  (h5 : distance_from_P_to_triangle = 10 * Real.sqrt 2) :
  let a := 10
  let b := 2
  let c := 1
  let result := a + b + c
  result = 13 :=
by
  sorry

end NUMINAMATH_GPT_sphere_triangle_distance_l2316_231607


namespace NUMINAMATH_GPT_num_three_digit_integers_divisible_by_12_l2316_231660

theorem num_three_digit_integers_divisible_by_12 : 
  (∃ (count : ℕ), count = 3 ∧ 
    (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 
      (∀ d : ℕ, d ∈ [n / 100, (n / 10) % 10, n % 10] → 4 < d) ∧ 
      n % 12 = 0 → 
      count = count + 1)) := 
sorry

end NUMINAMATH_GPT_num_three_digit_integers_divisible_by_12_l2316_231660


namespace NUMINAMATH_GPT_train_length_approx_90_l2316_231610

noncomputable def speed_in_m_per_s := (124 : ℝ) * (1000 / 3600)

noncomputable def time_in_s := (2.61269421026963 : ℝ)

noncomputable def length_of_train := speed_in_m_per_s * time_in_s

theorem train_length_approx_90 : abs (length_of_train - 90) < 1e-9 :=
  by
  sorry

end NUMINAMATH_GPT_train_length_approx_90_l2316_231610


namespace NUMINAMATH_GPT_nextSimultaneousRingingTime_l2316_231603

-- Define the intervals
def townHallInterval := 18
def universityTowerInterval := 24
def fireStationInterval := 30

-- Define the start time (in minutes from 00:00)
def startTime := 8 * 60 -- 8:00 AM

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Prove the next simultaneous ringing time
theorem nextSimultaneousRingingTime : 
  let lcmIntervals := lcm (lcm townHallInterval universityTowerInterval) fireStationInterval 
  startTime + lcmIntervals = 14 * 60 := -- 14:00 equals 2:00 PM in minutes
by
  -- You can replace the proof with the actual detailed proof.
  sorry

end NUMINAMATH_GPT_nextSimultaneousRingingTime_l2316_231603


namespace NUMINAMATH_GPT_profit_difference_is_50_l2316_231671

-- Given conditions
def materials_cost_cars : ℕ := 100
def cars_made_per_month : ℕ := 4
def price_per_car : ℕ := 50

def materials_cost_motorcycles : ℕ := 250
def motorcycles_made_per_month : ℕ := 8
def price_per_motorcycle : ℕ := 50

-- Definitions based on the conditions
def profit_from_cars : ℕ := (cars_made_per_month * price_per_car) - materials_cost_cars
def profit_from_motorcycles : ℕ := (motorcycles_made_per_month * price_per_motorcycle) - materials_cost_motorcycles

-- The difference in profit
def profit_difference : ℕ := profit_from_motorcycles - profit_from_cars

-- The proof goal
theorem profit_difference_is_50 :
  profit_difference = 50 := by
  sorry

end NUMINAMATH_GPT_profit_difference_is_50_l2316_231671


namespace NUMINAMATH_GPT_smallest_n_l2316_231656

theorem smallest_n (n : ℕ) : 
  (∃ k : ℕ, 4 * n = k^2) ∧ (∃ l : ℕ, 5 * n = l^5) ↔ n = 625 :=
by sorry

end NUMINAMATH_GPT_smallest_n_l2316_231656


namespace NUMINAMATH_GPT_chord_length_l2316_231670

noncomputable def circle_eq (θ : ℝ) : ℝ × ℝ :=
  (2 + 5 * Real.cos θ, 1 + 5 * Real.sin θ)

noncomputable def line_eq (t : ℝ) : ℝ × ℝ :=
  (-2 + 4 * t, -1 - 3 * t)

theorem chord_length :
  let center := (2, 1)
  let radius := 5
  let line_dist := |3 * center.1 + 4 * center.2 + 10| / Real.sqrt (3^2 + 4^2)
  let chord_len := 2 * Real.sqrt (radius^2 - line_dist^2)
  chord_len = 6 := 
by
  sorry

end NUMINAMATH_GPT_chord_length_l2316_231670


namespace NUMINAMATH_GPT_rate_of_current_l2316_231640

theorem rate_of_current (c : ℝ) (h1 : 7.5 = (20 + c) * 0.3) : c = 5 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_current_l2316_231640


namespace NUMINAMATH_GPT_overall_gain_percentage_correct_l2316_231652

structure Transaction :=
  (buy_prices : List ℕ)
  (sell_prices : List ℕ)

def overallGainPercentage (trans : Transaction) : ℚ :=
  let total_cost := (trans.buy_prices.foldl (· + ·) 0 : ℚ)
  let total_sell := (trans.sell_prices.foldl (· + ·) 0 : ℚ)
  (total_sell - total_cost) / total_cost * 100

theorem overall_gain_percentage_correct
  (trans : Transaction)
  (h_buy_prices : trans.buy_prices = [675, 850, 920])
  (h_sell_prices : trans.sell_prices = [1080, 1100, 1000]) :
  overallGainPercentage trans = 30.06 := by
  sorry

end NUMINAMATH_GPT_overall_gain_percentage_correct_l2316_231652


namespace NUMINAMATH_GPT_stored_bales_correct_l2316_231604

theorem stored_bales_correct :
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  stored_bales = 26 :=
by
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  show stored_bales = 26
  sorry

end NUMINAMATH_GPT_stored_bales_correct_l2316_231604


namespace NUMINAMATH_GPT_problem1_problem2_l2316_231613

def f (x a : ℝ) := x^2 + 2 * a * x + 2

theorem problem1 (a : ℝ) (h : a = -1) : 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≤ 37) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 37) ∧
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≥ 1) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 1) :=
by
  sorry

theorem problem2 (a : ℝ) : 
  (∀ x1 x2 : ℝ, -5 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 5 → f x1 a > f x2 a) ↔ a ≤ -5 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2316_231613


namespace NUMINAMATH_GPT_hypotenuse_of_triangle_PQR_l2316_231689

theorem hypotenuse_of_triangle_PQR (PA PB PC QR : ℝ) (h1: PA = 2) (h2: PB = 3) (h3: PC = 2)
  (h4: PA + PB + PC = QR) (h5: QR = PA + 3 + 2 * PA): QR = 5 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_hypotenuse_of_triangle_PQR_l2316_231689


namespace NUMINAMATH_GPT_circle_radius_increase_l2316_231632

variable (r n : ℝ) -- declare variables r and n as real numbers

theorem circle_radius_increase (h : 2 * π * (r + n) = 2 * (2 * π * r)) : r = n :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_increase_l2316_231632


namespace NUMINAMATH_GPT_algebraic_expression_value_l2316_231616

noncomputable def a : ℝ := 1 + Real.sqrt 2
noncomputable def b : ℝ := 1 - Real.sqrt 2

theorem algebraic_expression_value :
  let a := 1 + Real.sqrt 2
  let b := 1 - Real.sqrt 2
  a^2 - a * b + b^2 = 7 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2316_231616


namespace NUMINAMATH_GPT_certain_number_is_sixteen_l2316_231698

theorem certain_number_is_sixteen (x : ℝ) (h : x ^ 5 = 4 ^ 10) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_sixteen_l2316_231698


namespace NUMINAMATH_GPT_solve_for_x_l2316_231658

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 9) * x = 14) : x = 220.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2316_231658


namespace NUMINAMATH_GPT_eval_fraction_l2316_231696

theorem eval_fraction (a b : ℕ) : (40 : ℝ) = 2^3 * 5 → (10 : ℝ) = 2 * 5 → (40^56 / 10^28) = 160^28 :=
by 
  sorry

end NUMINAMATH_GPT_eval_fraction_l2316_231696


namespace NUMINAMATH_GPT_at_least_three_double_marked_l2316_231654

noncomputable def grid := Matrix (Fin 10) (Fin 20) ℕ -- 10x20 matrix with natural numbers

def is_red_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 20), k₁ ≠ k₂ ∧ (g i k₁) ≤ g i j ∧ (g i k₂) ≤ g i j ∧ ∀ (k : Fin 20), (k ≠ k₁ ∧ k ≠ k₂) → g i k ≤ g i j

def is_blue_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 10), k₁ ≠ k₂ ∧ (g k₁ j) ≤ g i j ∧ (g k₂ j) ≤ g i j ∧ ∀ (k : Fin 10), (k ≠ k₁ ∧ k ≠ k₂) → g k j ≤ g i j

def is_double_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  is_red_marked g i j ∧ is_blue_marked g i j

theorem at_least_three_double_marked (g : grid) :
  (∃ (i₁ i₂ i₃ : Fin 10) (j₁ j₂ j₃ : Fin 20), i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₃ ≠ i₁ ∧ 
    j₁ ≠ j₂ ∧ j₂ ≠ j₃ ∧ j₃ ≠ j₁ ∧ is_double_marked g i₁ j₁ ∧ is_double_marked g i₂ j₂ ∧ is_double_marked g i₃ j₃) :=
sorry

end NUMINAMATH_GPT_at_least_three_double_marked_l2316_231654


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l2316_231664

open Real

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x ^ 2 - 2 * x - 1 = 0 ∧ k * y ^ 2 - 2 * y - 1 = 0) ↔ k > -1 ∧ k ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l2316_231664


namespace NUMINAMATH_GPT_gcd_odd_multiple_1187_l2316_231672

theorem gcd_odd_multiple_1187 (b: ℤ) (h1: b % 2 = 1) (h2: ∃ k: ℤ, b = 1187 * k) :
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_odd_multiple_1187_l2316_231672


namespace NUMINAMATH_GPT_crayons_total_cost_l2316_231651

theorem crayons_total_cost :
  let packs_initial := 4
  let packs_to_buy := 2
  let cost_per_pack := 2.5
  let total_packs := packs_initial + packs_to_buy
  let total_cost := total_packs * cost_per_pack
  total_cost = 15 :=
by
  sorry

end NUMINAMATH_GPT_crayons_total_cost_l2316_231651


namespace NUMINAMATH_GPT_inequality_satisfaction_l2316_231606

theorem inequality_satisfaction (k n : ℕ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 + y^n / x^k) ≥ ((1 + y)^n / (1 + x)^k) ↔ 
    (k = 0) ∨ (n = 0) ∨ (0 = k ∧ 0 = n) ∨ (k ≥ n - 1 ∧ n ≥ 1) :=
by sorry

end NUMINAMATH_GPT_inequality_satisfaction_l2316_231606


namespace NUMINAMATH_GPT_percentage_students_receive_valentine_l2316_231618

/-- Given the conditions:
  1. There are 30 students.
  2. Mo wants to give a Valentine to some percentage of them.
  3. Each Valentine costs $2.
  4. Mo has $40.
  5. Mo will spend 90% of his money on Valentines.
Prove that the percentage of students receiving a Valentine is 60%.
-/
theorem percentage_students_receive_valentine :
  let total_students := 30
  let valentine_cost := 2
  let total_money := 40
  let spent_percentage := 0.90
  ∃ (cards : ℕ), 
    let money_spent := total_money * spent_percentage
    let cards_bought := money_spent / valentine_cost
    let percentage_students := (cards_bought / total_students) * 100
    percentage_students = 60 := 
by
  sorry

end NUMINAMATH_GPT_percentage_students_receive_valentine_l2316_231618


namespace NUMINAMATH_GPT_computer_table_cost_price_l2316_231665

theorem computer_table_cost_price (CP SP : ℝ) (h1 : SP = CP * (124 / 100)) (h2 : SP = 8091) :
  CP = 6525 :=
by
  sorry

end NUMINAMATH_GPT_computer_table_cost_price_l2316_231665


namespace NUMINAMATH_GPT_ellipse_foci_distance_l2316_231631

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 8) :
  2 * Real.sqrt (a^2 - b^2) = 12 :=
by
  rw [ha, hb]
  -- Proof follows here, but we skip it using sorry.
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l2316_231631


namespace NUMINAMATH_GPT_average_age_increase_l2316_231692

variable (A B C : ℕ)

theorem average_age_increase (A : ℕ) (B : ℕ) (C : ℕ) (h1 : 21 < B) (h2 : 23 < C) (h3 : A + B + C > A + 21 + 23) :
  (B + C) / 2 > 22 := by
  sorry

end NUMINAMATH_GPT_average_age_increase_l2316_231692


namespace NUMINAMATH_GPT_division_value_l2316_231679

theorem division_value (x : ℝ) (h1 : 2976 / x - 240 = 8) : x = 12 := 
by
  sorry

end NUMINAMATH_GPT_division_value_l2316_231679


namespace NUMINAMATH_GPT_chris_money_l2316_231615

-- Define conditions
def grandmother_gift : Nat := 25
def aunt_uncle_gift : Nat := 20
def parents_gift : Nat := 75
def total_after_birthday : Nat := 279

-- Define the proof problem to show Chris had $159 before his birthday
theorem chris_money (x : Nat) (h : x + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_birthday) :
  x = 159 :=
by
  -- Leave the proof blank
  sorry

end NUMINAMATH_GPT_chris_money_l2316_231615


namespace NUMINAMATH_GPT_canal_depth_l2316_231686

theorem canal_depth (A : ℝ) (w_top w_bottom : ℝ) (h : ℝ) 
    (hA : A = 10290) 
    (htop : w_top = 6) 
    (hbottom : w_bottom = 4) 
    (harea : A = 1 / 2 * (w_top + w_bottom) * h) : 
    h = 2058 :=
by
  -- here goes the proof steps
  sorry

end NUMINAMATH_GPT_canal_depth_l2316_231686


namespace NUMINAMATH_GPT_incorrect_statement_l2316_231628

def population : ℕ := 13000
def sample_size : ℕ := 500
def academic_performance (n : ℕ) : Type := sorry

def statement_A (ap : Type) : Prop := 
  ap = academic_performance population

def statement_B (ap : Type) : Prop := 
  ∀ (u : ℕ), u ≤ population → ap = academic_performance 1

def statement_C (ap : Type) : Prop := 
  ap = academic_performance sample_size

def statement_D : Prop := 
  sample_size = 500

theorem incorrect_statement : ¬ (statement_B (academic_performance 1)) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_l2316_231628


namespace NUMINAMATH_GPT_find_x_parallel_find_x_perpendicular_l2316_231669

def a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def b : ℝ × ℝ := (1, 2)

-- Given that a vector is proportional to another
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Given that the dot product is zero
def are_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_x_parallel (x : ℝ) (h : are_parallel (a x) b) : x = 2 :=
by sorry

theorem find_x_perpendicular (x : ℝ) (h : are_perpendicular (a x - b) b) : x = (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_GPT_find_x_parallel_find_x_perpendicular_l2316_231669


namespace NUMINAMATH_GPT_possible_degrees_of_remainder_l2316_231662

theorem possible_degrees_of_remainder (p : Polynomial ℝ) (h : p = 3 * X^3 - 5 * X^2 + 2 * X - 8) :
  ∃ d : Finset ℕ, d = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_possible_degrees_of_remainder_l2316_231662


namespace NUMINAMATH_GPT_negation_statement_l2316_231675

theorem negation_statement (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_GPT_negation_statement_l2316_231675


namespace NUMINAMATH_GPT_tree_shadow_length_l2316_231685

theorem tree_shadow_length (jane_shadow : ℝ) (jane_height : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h₁ : jane_shadow = 0.5)
  (h₂ : jane_height = 1.5)
  (h₃ : tree_height = 30)
  (h₄ : jane_height / jane_shadow = tree_height / tree_shadow)
  : tree_shadow = 10 :=
by
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_tree_shadow_length_l2316_231685


namespace NUMINAMATH_GPT_range_of_a_l2316_231695

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 + a) * x^2 + a * x + a < x^2 + 1) : a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2316_231695


namespace NUMINAMATH_GPT_range_of_m_l2316_231677

def point_P := (1, 1)
def circle_C1 (x y m : ℝ) := x^2 + y^2 + 2*x - m = 0

theorem range_of_m (m : ℝ) :
  (1 + 1)^2 + 1^2 > m + 1 → -1 < m ∧ m < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2316_231677


namespace NUMINAMATH_GPT_shorter_stick_length_l2316_231691

variable (L S : ℝ)

theorem shorter_stick_length
  (h1 : L - S = 12)
  (h2 : (2 / 3) * L = S) :
  S = 24 := by
  sorry

end NUMINAMATH_GPT_shorter_stick_length_l2316_231691


namespace NUMINAMATH_GPT_A_union_B_subset_B_A_intersection_B_subset_B_l2316_231614

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3 * x - 10 <= 0}
def B (m : ℝ) : Set ℝ := {x | m - 4 <= x ∧ x <= 3 * m + 2}

-- Problem 1: Prove the range of m if A ∪ B = B
theorem A_union_B_subset_B (m : ℝ) : (A ∪ B m = B m) → (1 ≤ m ∧ m ≤ 2) :=
by
  sorry

-- Problem 2: Prove the range of m if A ∩ B = B
theorem A_intersection_B_subset_B (m : ℝ) : (A ∩ B m = B m) → (m < -3) :=
by
  sorry

end NUMINAMATH_GPT_A_union_B_subset_B_A_intersection_B_subset_B_l2316_231614


namespace NUMINAMATH_GPT_math_problem_l2316_231682

theorem math_problem
  (a b c x1 x2 : ℝ)
  (h1 : a > 0)
  (h2 : a^2 = 4 * b)
  (h3 : |x1 - x2| = 4)
  (h4 : x1 < x2) :
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (c = 4) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2316_231682


namespace NUMINAMATH_GPT_select_more_stable_athlete_l2316_231684

-- Define the problem conditions
def athlete_average_score : ℝ := 9
def athlete_A_variance : ℝ := 1.2
def athlete_B_variance : ℝ := 2.4

-- Define what it means to have more stable performance
def more_stable (variance_A variance_B : ℝ) : Prop := variance_A < variance_B

-- The theorem to prove
theorem select_more_stable_athlete :
  more_stable athlete_A_variance athlete_B_variance →
  "A" = "A" :=
by
  sorry

end NUMINAMATH_GPT_select_more_stable_athlete_l2316_231684


namespace NUMINAMATH_GPT_geometric_sequence_term_6_l2316_231617

-- Define the geometric sequence conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

variables 
  (a : ℕ → ℝ) -- the geometric sequence
  (r : ℝ) -- common ratio, which is 2
  (h_r : r = 2)
  (h_pos : ∀ n, 0 < a n)
  (h_condition : a 4 * a 10 = 16)

-- The proof statement
theorem geometric_sequence_term_6 :
  a 6 = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_term_6_l2316_231617


namespace NUMINAMATH_GPT_max_value_of_f_l2316_231645

noncomputable def f (x a b : ℝ) := (1 - x ^ 2) * (x ^ 2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (x : ℝ) 
  (h1 : (∀ x, f x 8 15 = (1 - x^2) * (x^2 + 8*x + 15)))
  (h2 : ∀ x, f x a b = f (-(x + 4)) a b) :
  ∃ m, (∀ x, f x 8 15 ≤ m) ∧ m = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2316_231645


namespace NUMINAMATH_GPT_triangle_area_l2316_231688

noncomputable def area_of_triangle := 
  let a := 4
  let b := 5
  let c := 6
  let cosA := 3 / 4
  let sinA := Real.sqrt (1 - cosA ^ 2)
  (1 / 2) * b * c * sinA

theorem triangle_area :
  ∃ (a b c : ℝ), a = 4 ∧ b = 5 ∧ c = 6 ∧ 
  a < b ∧ b < c ∧ 
  -- Additional conditions
  (∃ A B C : ℝ, C = 2 * A ∧ 
   Real.cos A = 3 / 4 ∧ 
   Real.sin A * Real.cos A = sinA * cosA ∧ 
   0 < A ∧ A < Real.pi ∧ 
   (1 / 2) * b * c * sinA = (15 * Real.sqrt 7) / 4) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l2316_231688


namespace NUMINAMATH_GPT_prob_single_trial_l2316_231678

theorem prob_single_trial (P : ℝ) : 
  (1 - (1 - P)^4) = 65 / 81 → P = 1 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_prob_single_trial_l2316_231678


namespace NUMINAMATH_GPT_angle_ACE_is_38_l2316_231650

noncomputable def measure_angle_ACE (A B C D E : Type) : Prop :=
  let angle_ABC := 55
  let angle_BCA := 38
  let angle_BAC := 87
  let angle_ABD := 125
  (angle_ABC + angle_ABD = 180) → -- supplementary condition
  (angle_BAC = 87) → -- given angle at BAC
  (let angle_ACB := 180 - angle_BAC - angle_ABC;
   angle_ACB = angle_BCA ∧  -- derived angle at BCA
   angle_ACB = 38) → -- target angle
  (angle_BCA = 38) -- final result that needs to be proven

theorem angle_ACE_is_38 {A B C D E : Type} :
  measure_angle_ACE A B C D E :=
by
  sorry

end NUMINAMATH_GPT_angle_ACE_is_38_l2316_231650


namespace NUMINAMATH_GPT_cube_volume_in_cubic_yards_l2316_231642

def volume_in_cubic_feet := 64
def cubic_feet_per_cubic_yard := 27

theorem cube_volume_in_cubic_yards : 
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 64 / 27 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_in_cubic_yards_l2316_231642


namespace NUMINAMATH_GPT_find_S17_l2316_231661

-- Definitions based on the conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (a1 : ℝ) (d : ℝ)

-- Conditions from the problem restated in Lean
axiom arithmetic_sequence : ∀ n, a n = a1 + (n - 1) * d
axiom sum_of_n_terms : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)
axiom arithmetic_subseq : 2 * a 7 = a 5 + 3

-- Theorem to prove
theorem find_S17 : S 17 = 51 :=
by sorry

end NUMINAMATH_GPT_find_S17_l2316_231661


namespace NUMINAMATH_GPT_cows_count_l2316_231622

theorem cows_count (D C : ℕ) (h_legs : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end NUMINAMATH_GPT_cows_count_l2316_231622


namespace NUMINAMATH_GPT_elaine_earnings_increase_l2316_231602

variable (E : ℝ) -- Elaine's earnings last year
variable (P : ℝ) -- Percentage increase in earnings

-- Conditions
variable (rent_last_year : ℝ := 0.20 * E)
variable (earnings_this_year : ℝ := E * (1 + P / 100))
variable (rent_this_year : ℝ := 0.30 * earnings_this_year)
variable (multiplied_rent_last_year : ℝ := 1.875 * rent_last_year)

-- Theorem to be proven
theorem elaine_earnings_increase (h : rent_this_year = multiplied_rent_last_year) : P = 25 :=
by
  sorry

end NUMINAMATH_GPT_elaine_earnings_increase_l2316_231602


namespace NUMINAMATH_GPT_pencils_more_than_200_on_saturday_l2316_231609

theorem pencils_more_than_200_on_saturday 
    (p : ℕ → ℕ) 
    (h_start : p 1 = 3)
    (h_next_day : ∀ n, p (n + 1) = (p n + 2) * 2) 
    : p 6 > 200 :=
by
  -- Proof steps can be filled in here.
  sorry

end NUMINAMATH_GPT_pencils_more_than_200_on_saturday_l2316_231609


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_l2316_231638

open Real

theorem area_of_inscribed_rectangle (r l w : ℝ) (h_radius : r = 7) 
  (h_ratio : l / w = 3) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_l2316_231638


namespace NUMINAMATH_GPT_smallest_x_for_cubic_1890_l2316_231683

theorem smallest_x_for_cubic_1890 (x : ℕ) (N : ℕ) (hx : 1890 * x = N ^ 3) : x = 4900 :=
sorry

end NUMINAMATH_GPT_smallest_x_for_cubic_1890_l2316_231683


namespace NUMINAMATH_GPT_least_pawns_required_l2316_231600

theorem least_pawns_required (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : 2 * k > n) (h4 : 3 * k ≤ 2 * n) : 
  ∃ (m : ℕ), m = 4 * (n - k) :=
sorry

end NUMINAMATH_GPT_least_pawns_required_l2316_231600


namespace NUMINAMATH_GPT_cannot_contain_point_1997_0_l2316_231619

variable {m b : ℝ}

theorem cannot_contain_point_1997_0 (h : m * b > 0) : ¬ (0 = 1997 * m + b) := sorry

end NUMINAMATH_GPT_cannot_contain_point_1997_0_l2316_231619


namespace NUMINAMATH_GPT_correct_equation_l2316_231653

theorem correct_equation:
  (∀ x y : ℝ, -5 * (x - y) = -5 * x + 5 * y) ∧ 
  (∀ a c : ℝ, ¬ (-2 * (-a + c) = -2 * a - 2 * c)) ∧ 
  (∀ x y z : ℝ, ¬ (3 - (x + y + z) = -x + y - z)) ∧ 
  (∀ a b : ℝ, ¬ (3 * (a + 2 * b) = 3 * a + 2 * b)) :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l2316_231653


namespace NUMINAMATH_GPT_average_without_ivan_l2316_231639

theorem average_without_ivan
  (total_friends : ℕ := 5)
  (avg_all : ℝ := 55)
  (ivan_amount : ℝ := 43)
  (remaining_friends : ℕ := total_friends - 1)
  (total_amount : ℝ := total_friends * avg_all)
  (remaining_amount : ℝ := total_amount - ivan_amount)
  (new_avg : ℝ := remaining_amount / remaining_friends) :
  new_avg = 58 := 
sorry

end NUMINAMATH_GPT_average_without_ivan_l2316_231639


namespace NUMINAMATH_GPT_baker_remaining_cakes_l2316_231646

def initial_cakes : ℝ := 167.3
def sold_cakes : ℝ := 108.2
def remaining_cakes : ℝ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 59.1 := by
  sorry

end NUMINAMATH_GPT_baker_remaining_cakes_l2316_231646


namespace NUMINAMATH_GPT_average_apples_per_guest_l2316_231659

theorem average_apples_per_guest
  (servings_per_pie : ℕ)
  (pies : ℕ)
  (apples_per_serving : ℚ)
  (total_guests : ℕ)
  (red_delicious_proportion : ℚ)
  (granny_smith_proportion : ℚ)
  (total_servings := pies * servings_per_pie)
  (total_apples := total_servings * apples_per_serving)
  (total_red_delicious := (red_delicious_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (total_granny_smith := (granny_smith_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (average_apples_per_guest := total_apples / total_guests) :
  servings_per_pie = 8 →
  pies = 3 →
  apples_per_serving = 1.5 →
  total_guests = 12 →
  red_delicious_proportion = 2 →
  granny_smith_proportion = 1 →
  average_apples_per_guest = 3 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_average_apples_per_guest_l2316_231659


namespace NUMINAMATH_GPT_how_many_years_younger_is_C_compared_to_A_l2316_231625

variables (a b c d : ℕ)

def condition1 : Prop := a + b = b + c + 13
def condition2 : Prop := b + d = c + d + 7
def condition3 : Prop := a + d = 2 * c - 12

theorem how_many_years_younger_is_C_compared_to_A
  (h1 : condition1 a b c)
  (h2 : condition2 b c d)
  (h3 : condition3 a c d) : a = c + 13 :=
sorry

end NUMINAMATH_GPT_how_many_years_younger_is_C_compared_to_A_l2316_231625


namespace NUMINAMATH_GPT_solve_x_plus_Sx_eq_2001_l2316_231699

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem solve_x_plus_Sx_eq_2001 (x : ℕ) (h : x + sum_of_digits x = 2001) : x = 1977 :=
  sorry

end NUMINAMATH_GPT_solve_x_plus_Sx_eq_2001_l2316_231699


namespace NUMINAMATH_GPT_probability_white_black_l2316_231608

variable (a b : ℕ)

theorem probability_white_black (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  (2 * a * b) / (a + b) / (a + b - 1) = (2 * (a * b) : ℝ) / ((a + b) * (a + b - 1): ℝ) :=
by sorry

end NUMINAMATH_GPT_probability_white_black_l2316_231608


namespace NUMINAMATH_GPT_difference_of_squares_650_550_l2316_231624

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 :=
by sorry

end NUMINAMATH_GPT_difference_of_squares_650_550_l2316_231624


namespace NUMINAMATH_GPT_factorize_expression_l2316_231673

theorem factorize_expression (x : ℝ) : x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2316_231673


namespace NUMINAMATH_GPT_a_perp_a_minus_b_l2316_231690

noncomputable def a : ℝ × ℝ := (-2, 1)
noncomputable def b : ℝ × ℝ := (-1, 3)
noncomputable def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem a_perp_a_minus_b : (a.1 * a_minus_b.1 + a.2 * a_minus_b.2) = 0 := by
  sorry

end NUMINAMATH_GPT_a_perp_a_minus_b_l2316_231690


namespace NUMINAMATH_GPT_a_plus_b_eq_2_l2316_231693

theorem a_plus_b_eq_2 (a b : ℝ) 
  (h₁ : 2 = a + b) 
  (h₂ : 4 = a + b / 4) : a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_b_eq_2_l2316_231693


namespace NUMINAMATH_GPT_michael_truck_meetings_2_times_l2316_231621

/-- Michael walks at a rate of 6 feet per second on a straight path. 
Trash pails are placed every 240 feet along the path. 
A garbage truck traveling at 12 feet per second in the same direction stops for 40 seconds at each pail. 
When Michael passes a pail, he sees the truck, which is 240 feet ahead, just leaving the next pail. 
Prove that Michael and the truck will meet exactly 2 times. -/

def michael_truck_meetings (v_michael v_truck d_pail t_stop init_michael init_truck : ℕ) : ℕ := sorry

theorem michael_truck_meetings_2_times :
  michael_truck_meetings 6 12 240 40 0 240 = 2 := 
  sorry

end NUMINAMATH_GPT_michael_truck_meetings_2_times_l2316_231621


namespace NUMINAMATH_GPT_ratatouille_cost_per_quart_l2316_231627

theorem ratatouille_cost_per_quart:
  let eggplant_weight := 5.5
  let eggplant_price := 2.20
  let zucchini_weight := 3.8
  let zucchini_price := 1.85
  let tomatoes_weight := 4.6
  let tomatoes_price := 3.75
  let onions_weight := 2.7
  let onions_price := 1.10
  let basil_weight := 1.0
  let basil_price_per_quarter := 2.70
  let bell_peppers_weight := 0.75
  let bell_peppers_price := 3.15
  let yield_quarts := 4.5
  let eggplant_cost := eggplant_weight * eggplant_price
  let zucchini_cost := zucchini_weight * zucchini_price
  let tomatoes_cost := tomatoes_weight * tomatoes_price
  let onions_cost := onions_weight * onions_price
  let basil_cost := basil_weight * (basil_price_per_quarter * 4)
  let bell_peppers_cost := bell_peppers_weight * bell_peppers_price
  let total_cost := eggplant_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost + bell_peppers_cost
  let cost_per_quart := total_cost / yield_quarts
  cost_per_quart = 11.67 :=
by
  sorry

end NUMINAMATH_GPT_ratatouille_cost_per_quart_l2316_231627


namespace NUMINAMATH_GPT_simplify_expression_l2316_231626

theorem simplify_expression : (90 / 150) * (35 / 21) = 1 :=
by
  -- Insert proof here 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2316_231626


namespace NUMINAMATH_GPT_max_pencils_thrown_out_l2316_231611

theorem max_pencils_thrown_out (n : ℕ) : (n % 7 ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_max_pencils_thrown_out_l2316_231611


namespace NUMINAMATH_GPT_compute_expression_l2316_231601

variables (a b c : ℝ)

theorem compute_expression (h1 : a - b = 2) (h2 : a + c = 6) : 
  (2 * a + b + c) - 2 * (a - b - c) = 12 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2316_231601


namespace NUMINAMATH_GPT_value_of_m_l2316_231612

theorem value_of_m (m : ℝ) :
  (∀ A B : ℝ × ℝ, A = (m + 1, -2) → B = (3, m - 1) → (A.snd = B.snd) → m = -1) :=
by
  intros A B hA hB h_parallel
  -- Apply the given conditions and assumptions to prove the value of m.
  sorry

end NUMINAMATH_GPT_value_of_m_l2316_231612


namespace NUMINAMATH_GPT_rational_comparison_correct_l2316_231620

-- Definitions based on conditions 
def positive_gt_zero (a : ℚ) : Prop := 0 < a
def negative_lt_zero (a : ℚ) : Prop := a < 0
def positive_gt_negative (a b : ℚ) : Prop := positive_gt_zero a ∧ negative_lt_zero b ∧ a > b
def negative_comparison (a b : ℚ) : Prop := negative_lt_zero a ∧ negative_lt_zero b ∧ abs a > abs b ∧ a < b

-- Theorem to prove
theorem rational_comparison_correct :
  (0 < - (1 / 2)) = false ∧
  ((4 / 5) < - (6 / 7)) = false ∧
  ((9 / 8) > (8 / 9)) = true ∧
  (-4 > -3) = false :=
by
  -- Mark the proof as unfinished.
  sorry

end NUMINAMATH_GPT_rational_comparison_correct_l2316_231620


namespace NUMINAMATH_GPT_number_of_rows_seating_exactly_9_students_l2316_231635

theorem number_of_rows_seating_exactly_9_students (x : ℕ) : 
  ∀ y z, x * 9 + y * 5 + z * 8 = 55 → x % 5 = 1 ∧ x % 8 = 7 → x = 3 :=
by sorry

end NUMINAMATH_GPT_number_of_rows_seating_exactly_9_students_l2316_231635


namespace NUMINAMATH_GPT_num_pairs_eq_12_l2316_231667

theorem num_pairs_eq_12 :
  ∃ (n : ℕ), (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧
    (a + 1/b : ℚ) / (1/a + b : ℚ) = 7 ↔ (7 * b = a)) ∧ n = 12 :=
sorry

end NUMINAMATH_GPT_num_pairs_eq_12_l2316_231667


namespace NUMINAMATH_GPT_ratio_problem_l2316_231668

theorem ratio_problem (a b c d : ℚ) (h1 : a / b = 5 / 4) (h2 : c / d = 4 / 1) (h3 : d / b = 1 / 8) :
  a / c = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_problem_l2316_231668


namespace NUMINAMATH_GPT_find_x_l2316_231644

theorem find_x (x : ℤ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2316_231644


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_eq_neg_3_l2316_231681

theorem tan_alpha_minus_pi_over_4_eq_neg_3
  (α : ℝ)
  (h1 : True) -- condition to ensure we define α in ℝ, "True" is just a dummy
  (a : ℝ × ℝ := (Real.cos α, -2))
  (b : ℝ × ℝ := (Real.sin α, 1))
  (h2 : ∃ k : ℝ, a = k • b) : 
  Real.tan (α - Real.pi / 4) = -3 :=
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_eq_neg_3_l2316_231681


namespace NUMINAMATH_GPT_M_inter_N_is_1_2_l2316_231634

-- Definitions based on given conditions
def M : Set ℝ := { y | ∃ x : ℝ, x > 0 ∧ y = 2^x }
def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Prove intersection of M and N is (1, 2]
theorem M_inter_N_is_1_2 :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_M_inter_N_is_1_2_l2316_231634


namespace NUMINAMATH_GPT_radius_of_circle_l2316_231633

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end NUMINAMATH_GPT_radius_of_circle_l2316_231633


namespace NUMINAMATH_GPT_angela_more_marbles_l2316_231636

/--
Albert has three times as many marbles as Angela.
Allison has 28 marbles.
Albert and Allison have 136 marbles together.
Prove that Angela has 8 more marbles than Allison.
-/
theorem angela_more_marbles 
  (albert_angela : ℕ) 
  (angela: ℕ) 
  (albert: ℕ) 
  (allison: ℕ) 
  (h_albert_is_three_times_angela : albert = 3 * angela) 
  (h_allison_is_28 : allison = 28) 
  (h_albert_allison_is_136 : albert + allison = 136) 
  : angela - allison = 8 := 
by
  sorry

end NUMINAMATH_GPT_angela_more_marbles_l2316_231636


namespace NUMINAMATH_GPT_cricket_bat_profit_percentage_l2316_231680

theorem cricket_bat_profit_percentage 
  (selling_price profit : ℝ) 
  (h_sp: selling_price = 850) 
  (h_p: profit = 230) : 
  (profit / (selling_price - profit) * 100) = 37.10 :=
by
  sorry

end NUMINAMATH_GPT_cricket_bat_profit_percentage_l2316_231680


namespace NUMINAMATH_GPT_min_value_sin_cos_l2316_231655

open Real

theorem min_value_sin_cos (x : ℝ) : 
  ∃ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 = 2 / 3 ∧ ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_sin_cos_l2316_231655


namespace NUMINAMATH_GPT_compressor_stations_valid_l2316_231694

def compressor_stations : Prop :=
  ∃ (x y z a : ℝ),
    x + y = 3 * z ∧  -- condition 1
    z + y = x + a ∧  -- condition 2
    x + z = 60 ∧     -- condition 3
    0 < a ∧ a < 60 ∧ -- condition 4
    a = 42 ∧         -- specific value for a
    x = 33 ∧         -- expected value for x
    y = 48 ∧         -- expected value for y
    z = 27           -- expected value for z

theorem compressor_stations_valid : compressor_stations := 
  by sorry

end NUMINAMATH_GPT_compressor_stations_valid_l2316_231694


namespace NUMINAMATH_GPT_false_props_l2316_231623

-- Definitions for conditions
def prop1 :=
  ∀ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ (a * d = b * c) → 
  (a / b = b / c ∧ b / c = c / d)

def prop2 :=
  ∀ (a : ℕ), (∃ k : ℕ, a = 2 * k) → (a % 2 = 0)

def prop3 :=
  ∀ (A : ℝ), (A > 30) → (Real.sin (A * Real.pi / 180) > 1 / 2)

-- Theorem statement
theorem false_props : (¬ prop1) ∧ (¬ prop3) :=
by sorry

end NUMINAMATH_GPT_false_props_l2316_231623
