import Mathlib

namespace NUMINAMATH_CALUDE_five_mondays_in_september_l2779_277900

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Represents a month with its dates -/
structure Month where
  dates : List Date
  numDays : Nat

def August : Month := sorry
def September : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- Determines the day of the week for the first day of the next month -/
def nextMonthFirstDay (m : Month) : DayOfWeek := sorry

theorem five_mondays_in_september 
  (h1 : August.numDays = 31)
  (h2 : September.numDays = 30)
  (h3 : countDayOccurrences August DayOfWeek.Sunday = 5) :
  countDayOccurrences September DayOfWeek.Monday = 5 := by sorry

end NUMINAMATH_CALUDE_five_mondays_in_september_l2779_277900


namespace NUMINAMATH_CALUDE_desk_rearrangement_combinations_l2779_277989

/-- The number of choices for each day of the week --/
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 4
def friday_choices : ℕ := 1

/-- The total number of combinations --/
def total_combinations : ℕ := 
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

/-- Theorem stating that the total number of combinations is 60 --/
theorem desk_rearrangement_combinations : total_combinations = 60 := by
  sorry

end NUMINAMATH_CALUDE_desk_rearrangement_combinations_l2779_277989


namespace NUMINAMATH_CALUDE_max_discount_rate_l2779_277909

/-- Represents the maximum discount rate problem --/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1) :
  ∃ (max_discount : ℝ),
    max_discount = 0.12 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (selling_price * (1 - discount) - cost_price) / cost_price ≥ min_profit_margin :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l2779_277909


namespace NUMINAMATH_CALUDE_correct_factorization_l2779_277906

theorem correct_factorization (a b : ℤ) :
  (∃ k : ℤ, (X + 6) * (X - 2) = X^2 + k*X + b) ∧
  (∃ m : ℤ, (X - 8) * (X + 4) = X^2 + a*X + m) →
  (X + 2) * (X - 6) = X^2 + a*X + b :=
by sorry

end NUMINAMATH_CALUDE_correct_factorization_l2779_277906


namespace NUMINAMATH_CALUDE_student_a_score_l2779_277931

/-- Calculates the score for a test based on the given grading method -/
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℕ :=
  let incorrectResponses := totalQuestions - correctResponses
  correctResponses - 2 * incorrectResponses

theorem student_a_score :
  calculateScore 100 90 = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_a_score_l2779_277931


namespace NUMINAMATH_CALUDE_probability_of_perfect_square_l2779_277997

theorem probability_of_perfect_square :
  ∀ (P : ℝ),
  (P * 50 + 3 * P * 50 = 1) →
  (∃ (perfect_squares_le_50 perfect_squares_gt_50 : ℕ),
    perfect_squares_le_50 = 7 ∧ perfect_squares_gt_50 = 3) →
  (perfect_squares_le_50 * P + perfect_squares_gt_50 * 3 * P) / 100 = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_perfect_square_l2779_277997


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2779_277915

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2779_277915


namespace NUMINAMATH_CALUDE_shopkeeper_total_cards_l2779_277978

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define the number of complete decks the shopkeeper has
def complete_decks : ℕ := 3

-- Define the number of additional cards
def additional_cards : ℕ := 4

-- Theorem to prove
theorem shopkeeper_total_cards : 
  complete_decks * standard_deck_size + additional_cards = 160 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_cards_l2779_277978


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_is_72_l2779_277939

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 8*x - 20

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x + a) * q x + f (-a) :=
sorry

theorem remainder_is_72 : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x + 2) * q x + 72 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_is_72_l2779_277939


namespace NUMINAMATH_CALUDE_range_of_a_l2779_277929

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) ∧ 
  (∃ x_0 : ℝ, x_0^2 + 2*a*x_0 + 2 - a = 0) → 
  a = 1 ∨ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2779_277929


namespace NUMINAMATH_CALUDE_series_divergent_l2779_277984

open Complex

/-- The series ∑_{n=1}^{∞} (e^(iπ/n))/n is divergent -/
theorem series_divergent : 
  ¬ Summable (fun n : ℕ => (exp (I * π / n : ℂ)) / n) :=
sorry

end NUMINAMATH_CALUDE_series_divergent_l2779_277984


namespace NUMINAMATH_CALUDE_line_circle_separation_l2779_277916

theorem line_circle_separation (a b : ℝ) (h_inside : a^2 + b^2 < 1) (h_not_origin : (a, b) ≠ (0, 0)) :
  ∀ x y : ℝ, (x^2 + y^2 = 1) → (a*x + b*y ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_separation_l2779_277916


namespace NUMINAMATH_CALUDE_second_player_wins_l2779_277949

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a move in the game --/
inductive Move
  | Up : Nat → Move
  | Left : Nat → Move

/-- Applies a move to a position --/
def applyMove (pos : Position) (move : Move) : Position :=
  match move with
  | Move.Up n => ⟨pos.x, pos.y + n⟩
  | Move.Left n => ⟨pos.x - n, pos.y⟩

/-- Checks if a position is valid on the 8x8 board --/
def isValidPosition (pos : Position) : Prop :=
  1 ≤ pos.x ∧ pos.x ≤ 8 ∧ 1 ≤ pos.y ∧ pos.y ≤ 8

/-- Checks if a move is valid from a given position --/
def isValidMove (pos : Position) (move : Move) : Prop :=
  isValidPosition (applyMove pos move)

/-- Represents the game state --/
structure GameState :=
  (position : Position)
  (currentPlayer : Bool)  -- True for first player, False for second player

/-- The winning strategy for the second player --/
def secondPlayerWinningStrategy : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.position = ⟨1, 1⟩ →
      initialState.currentPlayer = true →
      ∀ (game : ℕ → GameState),
        game 0 = initialState →
        (∀ n : ℕ, 
          (game (n+1)).position = 
            if (game n).currentPlayer
            then applyMove (game n).position (strategy (game n))
            else applyMove (game n).position (strategy (game n))) →
        ∃ (n : ℕ), ¬isValidMove (game n).position (strategy (game n))

theorem second_player_wins : secondPlayerWinningStrategy :=
  sorry

end NUMINAMATH_CALUDE_second_player_wins_l2779_277949


namespace NUMINAMATH_CALUDE_smallest_a1_l2779_277911

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 11 * a (n - 1) - n)

theorem smallest_a1 (a : ℕ → ℝ) (h : is_valid_sequence a) :
  ∀ ε > 0, a 1 ≥ 21 / 100 - ε :=
sorry

end NUMINAMATH_CALUDE_smallest_a1_l2779_277911


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2779_277996

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℝ)
  (h₁ : a₁ = 5)
  (h₂ : a₅ = 10240)
  (h₃ : ∃ (r : ℝ), ∀ (n : ℕ), n ≤ 5 → a₁ * r^(n-1) = a₅^((n-1)/4) * a₁^(1-(n-1)/4)) :
  ∃ (a₄ : ℝ), a₄ = 2560 ∧ a₄ = a₁ * (a₅ / a₁)^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2779_277996


namespace NUMINAMATH_CALUDE_range_of_m_l2779_277926

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > (1/2) → x^2 - m*x + 4 > 0) → m < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2779_277926


namespace NUMINAMATH_CALUDE_factorization_of_9a_minus_6b_l2779_277960

theorem factorization_of_9a_minus_6b (a b : ℝ) : 9*a - 6*b = 3*(3*a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_9a_minus_6b_l2779_277960


namespace NUMINAMATH_CALUDE_smallest_w_l2779_277998

theorem smallest_w (w : ℕ+) : 
  (∃ k : ℕ, 936 * w.val = k * 2^5) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 3^3) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 14^2) → 
  w ≥ 1764 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l2779_277998


namespace NUMINAMATH_CALUDE_combined_boys_avg_is_70_8_l2779_277999

/-- Represents a school with average scores for boys, girls, and combined -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Calculates the combined average score for boys across two schools -/
def combined_boys_avg (school1 school2 : School) : ℝ :=
  sorry

/-- Theorem stating that the combined average score for boys is 70.8 -/
theorem combined_boys_avg_is_70_8
  (chs : School)
  (dhs : School)
  (h_chs_boys : chs.boys_avg = 68)
  (h_chs_girls : chs.girls_avg = 73)
  (h_chs_combined : chs.combined_avg = 70)
  (h_dhs_boys : dhs.boys_avg = 75)
  (h_dhs_girls : dhs.girls_avg = 85)
  (h_dhs_combined : dhs.combined_avg = 80) :
  combined_boys_avg chs dhs = 70.8 := by
  sorry

end NUMINAMATH_CALUDE_combined_boys_avg_is_70_8_l2779_277999


namespace NUMINAMATH_CALUDE_bicycle_distance_l2779_277937

/-- Given a bicycle traveling b/2 feet in t seconds, prove it travels 50b/t yards in 5 minutes -/
theorem bicycle_distance (b t : ℝ) (h : b > 0) (h' : t > 0) : 
  (b / 2) / t * (5 * 60) / 3 = 50 * b / t := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_l2779_277937


namespace NUMINAMATH_CALUDE_number_difference_proof_l2779_277948

theorem number_difference_proof (x : ℚ) : x - (3/5) * x = 50 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l2779_277948


namespace NUMINAMATH_CALUDE_fixed_point_l2779_277920

-- Define the function f(x) = kx - k + 2
def f (k : ℝ) (x : ℝ) : ℝ := k * x - k + 2

-- Theorem stating that f(x) passes through (1, 2) for any k
theorem fixed_point (k : ℝ) : f k 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_l2779_277920


namespace NUMINAMATH_CALUDE_watch_selling_prices_l2779_277971

/-- Calculates the selling price given the cost price and profit percentage -/
def sellingPrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

theorem watch_selling_prices :
  let watch1CP : ℚ := 1400
  let watch1Profit : ℚ := 5
  let watch2CP : ℚ := 1800
  let watch2Profit : ℚ := 8
  let watch3CP : ℚ := 2500
  let watch3Profit : ℚ := 12
  (sellingPrice watch1CP watch1Profit = 1470) ∧
  (sellingPrice watch2CP watch2Profit = 1944) ∧
  (sellingPrice watch3CP watch3Profit = 2800) :=
by sorry

end NUMINAMATH_CALUDE_watch_selling_prices_l2779_277971


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l2779_277975

theorem complex_arithmetic_proof : ((2 : ℂ) + 5*I + (3 : ℂ) - 6*I) * ((1 : ℂ) + 2*I) = (7 : ℂ) + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l2779_277975


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2779_277972

theorem min_value_expression (x y : ℝ) : x^2 + 8*x*Real.sin y - 16*(Real.cos y)^2 ≥ -16 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + 8*x*Real.sin y - 16*(Real.cos y)^2 = -16 := by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2779_277972


namespace NUMINAMATH_CALUDE_one_third_percentage_l2779_277992

-- Define the given numbers
def total : ℚ := 1206
def divisor : ℚ := 3
def base : ℚ := 134

-- Define one-third of the total
def one_third : ℚ := total / divisor

-- Define the percentage calculation
def percentage : ℚ := (one_third / base) * 100

-- Theorem to prove
theorem one_third_percentage : percentage = 300 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percentage_l2779_277992


namespace NUMINAMATH_CALUDE_eight_faucets_fill_time_l2779_277962

/-- The time (in seconds) it takes for a given number of faucets to fill a tub of a given volume -/
def fill_time (num_faucets : ℕ) (volume : ℝ) : ℝ :=
  -- Definition to be filled based on the problem conditions
  sorry

theorem eight_faucets_fill_time :
  -- Given conditions
  (fill_time 4 200 = 8 * 60) →  -- 4 faucets fill 200 gallons in 8 minutes (converted to seconds)
  (∀ n v, fill_time n v = fill_time 1 v / n) →  -- All faucets dispense water at the same rate
  -- Conclusion
  (fill_time 8 50 = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_eight_faucets_fill_time_l2779_277962


namespace NUMINAMATH_CALUDE_max_quotient_value_l2779_277922

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 900 ≤ b ∧ b ≤ 1800) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 900 ≤ y ∧ y ≤ 1800 → x / y ≤ a / b) →
  a / b = 5 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l2779_277922


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2779_277957

theorem factorization_of_cubic (x : ℝ) : 3 * x^3 - 27 * x = 3 * x * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2779_277957


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l2779_277932

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

theorem extreme_values_of_f :
  ∃ (a b : ℝ), (∀ x : ℝ, f x ≤ f a ∨ f x ≥ f b) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≤ f c) → c = a) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≥ f c) → c = b) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l2779_277932


namespace NUMINAMATH_CALUDE_bus_driver_rate_l2779_277955

/-- Represents the bus driver's compensation structure and work details -/
structure BusDriverCompensation where
  regularHours : ℕ := 40
  totalHours : ℕ
  overtimeMultiplier : ℚ
  totalCompensation : ℚ

/-- Calculates the regular hourly rate given the compensation structure -/
def calculateRegularRate (bdc : BusDriverCompensation) : ℚ :=
  let overtimeHours := bdc.totalHours - bdc.regularHours
  bdc.totalCompensation / (bdc.regularHours + overtimeHours * bdc.overtimeMultiplier)

/-- Theorem stating that the bus driver's regular rate is $16 per hour -/
theorem bus_driver_rate : 
  let bdc : BusDriverCompensation := {
    totalHours := 65,
    overtimeMultiplier := 1.75,
    totalCompensation := 1340
  }
  calculateRegularRate bdc = 16 := by sorry

end NUMINAMATH_CALUDE_bus_driver_rate_l2779_277955


namespace NUMINAMATH_CALUDE_kim_morning_routine_time_l2779_277969

/-- Represents Kim's morning routine and calculates the total time taken. -/
def morning_routine_time (total_employees : ℕ) (senior_employees : ℕ) (overtime_employees : ℕ)
  (coffee_time : ℕ) (regular_status_time : ℕ) (senior_status_extra_time : ℕ)
  (overtime_payroll_time : ℕ) (regular_payroll_time : ℕ)
  (email_time : ℕ) (task_allocation_time : ℕ) : ℕ :=
  let regular_employees := total_employees - senior_employees
  let non_overtime_employees := total_employees - overtime_employees
  coffee_time +
  (regular_employees * regular_status_time) +
  (senior_employees * (regular_status_time + senior_status_extra_time)) +
  (overtime_employees * overtime_payroll_time) +
  (non_overtime_employees * regular_payroll_time) +
  email_time +
  task_allocation_time

/-- Theorem stating that Kim's morning routine takes 60 minutes given the specified conditions. -/
theorem kim_morning_routine_time :
  morning_routine_time 9 3 4 5 2 1 3 1 10 7 = 60 := by
  sorry

end NUMINAMATH_CALUDE_kim_morning_routine_time_l2779_277969


namespace NUMINAMATH_CALUDE_function_inequality_l2779_277986

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, (x - 1) * deriv f x ≤ 0) : 
  f 0 + f 2 ≤ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2779_277986


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2779_277934

/-- Three points are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

/-- The theorem states that if the points (4,7), (0,k), and (-8,5) are collinear, then k = 19/3. -/
theorem collinear_points_k_value :
  collinear (4, 7) (0, k) (-8, 5) → k = 19/3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2779_277934


namespace NUMINAMATH_CALUDE_bill_calculation_l2779_277933

theorem bill_calculation (a b c : ℤ) 
  (h1 : a - (b - c) = 13)
  (h2 : (b - c) - a = -9)
  (h3 : a - b - c = 1) : 
  b - c = 1 := by
sorry

end NUMINAMATH_CALUDE_bill_calculation_l2779_277933


namespace NUMINAMATH_CALUDE_only_5_12_13_is_pythagorean_triple_l2779_277905

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem only_5_12_13_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 1 3 5 ∧
  is_pythagorean_triple 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_only_5_12_13_is_pythagorean_triple_l2779_277905


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2779_277938

/-- The angle between two vectors given their components and projection. -/
theorem angle_between_vectors (a b : ℝ × ℝ) (h : Real.sqrt 3 * (3 : ℝ) = (b.2 : ℝ)) 
  (proj : (3 : ℝ) * (1 : ℝ) + Real.sqrt 3 * b.2 = 3 * Real.sqrt ((1 : ℝ)^2 + (Real.sqrt 3)^2)) :
  let angle := Real.arccos ((3 : ℝ) * (1 : ℝ) + Real.sqrt 3 * b.2) / 
    (Real.sqrt ((1 : ℝ)^2 + (Real.sqrt 3)^2) * Real.sqrt ((3 : ℝ)^2 + b.2^2))
  angle = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2779_277938


namespace NUMINAMATH_CALUDE_negative_integer_problem_l2779_277965

theorem negative_integer_problem (n : ℤ) : 
  n < 0 → n * (-8) + 5 = 93 → n = -11 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_problem_l2779_277965


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l2779_277983

theorem complex_sum_to_polar : 15 * Complex.exp (Complex.I * Real.pi / 6) + 15 * Complex.exp (Complex.I * 5 * Real.pi / 6) = 15 * Complex.exp (Complex.I * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l2779_277983


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2779_277914

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 1000 * x^3 + 27 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2779_277914


namespace NUMINAMATH_CALUDE_frac_2023rd_digit_l2779_277953

-- Define the fraction
def frac : ℚ := 7 / 26

-- Define the length of the repeating decimal
def repeat_length : ℕ := 6

-- Define the position we're interested in
def position : ℕ := 2023

-- Define the function that returns the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem frac_2023rd_digit :
  nth_digit position = 5 :=
sorry

end NUMINAMATH_CALUDE_frac_2023rd_digit_l2779_277953


namespace NUMINAMATH_CALUDE_complex_number_additive_inverse_parts_l2779_277927

theorem complex_number_additive_inverse_parts (b : ℝ) : 
  let z := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_additive_inverse_parts_l2779_277927


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2779_277958

theorem concentric_circles_area_ratio : 
  let d₁ : ℝ := 2  -- diameter of smallest circle
  let d₂ : ℝ := 4  -- diameter of middle circle
  let d₃ : ℝ := 6  -- diameter of largest circle
  let r₁ := d₁ / 2  -- radius of smallest circle
  let r₂ := d₂ / 2  -- radius of middle circle
  let r₃ := d₃ / 2  -- radius of largest circle
  let A₁ := π * r₁^2  -- area of smallest circle
  let A₂ := π * r₂^2  -- area of middle circle
  let A₃ := π * r₃^2  -- area of largest circle
  let blue_area := A₂ - A₁  -- area between smallest and middle circles
  let green_area := A₃ - A₂  -- area between middle and largest circles
  (green_area / blue_area : ℝ) = 5/3
  := by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2779_277958


namespace NUMINAMATH_CALUDE_garden_sprinkler_morning_usage_garden_sprinkler_conditions_l2779_277970

/-- A sprinkler system that waters a desert garden twice daily -/
structure SprinklerSystem where
  morning_usage : ℝ
  evening_usage : ℝ
  days : ℕ
  total_usage : ℝ

/-- The specific sprinkler system described in the problem -/
def garden_sprinkler : SprinklerSystem where
  morning_usage := 4  -- This is what we want to prove
  evening_usage := 6
  days := 5
  total_usage := 50

/-- Theorem stating that the morning usage of the garden sprinkler is 4 liters -/
theorem garden_sprinkler_morning_usage :
  garden_sprinkler.morning_usage = 4 :=
by sorry

/-- Theorem proving that the given conditions are satisfied by the garden sprinkler -/
theorem garden_sprinkler_conditions :
  garden_sprinkler.evening_usage = 6 ∧
  garden_sprinkler.days = 5 ∧
  garden_sprinkler.total_usage = 50 ∧
  garden_sprinkler.days * (garden_sprinkler.morning_usage + garden_sprinkler.evening_usage) = garden_sprinkler.total_usage :=
by sorry

end NUMINAMATH_CALUDE_garden_sprinkler_morning_usage_garden_sprinkler_conditions_l2779_277970


namespace NUMINAMATH_CALUDE_count_eight_to_thousand_l2779_277994

/-- Count of digit 8 in a single integer -/
def count_eight (n : ℕ) : ℕ := sorry

/-- Sum of count_eight for integers from 1 to n -/
def sum_count_eight (n : ℕ) : ℕ := sorry

/-- The count of digit 8 in integers from 1 to 1000 is 300 -/
theorem count_eight_to_thousand : sum_count_eight 1000 = 300 := by sorry

end NUMINAMATH_CALUDE_count_eight_to_thousand_l2779_277994


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2779_277912

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2779_277912


namespace NUMINAMATH_CALUDE_system_inequalities_solution_range_l2779_277979

theorem system_inequalities_solution_range (a : ℚ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (((2 * x + 5 : ℚ) / 3 > x - 5) ∧ ((x + 3 : ℚ) / 2 < x + a)))) →
  (-6 < a ∧ a ≤ -11/2) :=
sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_range_l2779_277979


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2779_277930

/-- Given a rectangle with specific properties, prove its perimeter is 92 cm -/
theorem rectangle_perimeter (width length : ℕ) : 
  width = 34 ∧ 
  width % 4 = 2 ∧ 
  (width / 4) * length = 24 → 
  2 * (width + length) = 92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2779_277930


namespace NUMINAMATH_CALUDE_multiply_preserves_inequality_l2779_277942

theorem multiply_preserves_inequality (a b c : ℝ) : a > b → c > 0 → a * c > b * c := by
  sorry

end NUMINAMATH_CALUDE_multiply_preserves_inequality_l2779_277942


namespace NUMINAMATH_CALUDE_dog_bath_time_l2779_277918

/-- Represents the time spent on various activities with a dog -/
structure DogCareTime where
  total : ℝ
  walking : ℝ
  bath : ℝ
  blowDry : ℝ

/-- Represents the walking parameters -/
structure WalkingParams where
  distance : ℝ
  speed : ℝ

/-- Theorem stating the bath time given the conditions -/
theorem dog_bath_time (t : DogCareTime) (w : WalkingParams) : 
  t.total = 60 ∧ 
  w.distance = 3 ∧ 
  w.speed = 6 ∧ 
  t.blowDry = t.bath / 2 ∧ 
  t.total = t.walking + t.bath + t.blowDry ∧ 
  t.walking = w.distance / w.speed * 60 →
  t.bath = 20 := by
  sorry


end NUMINAMATH_CALUDE_dog_bath_time_l2779_277918


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l2779_277967

theorem absolute_value_of_z (r : ℝ) (z : ℂ) 
  (hr : |r| > 2) 
  (hz : z - 1/z = r) : 
  Complex.abs z = Real.sqrt ((r^2 / 2) + 1) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l2779_277967


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2779_277974

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 551245 → ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    m + 5 = 9 * k₁ ∧ 
    m + 5 = 70 * k₂ ∧ 
    m + 5 = 25 * k₃ ∧ 
    m + 5 = 21 * k₄ ∧ 
    m + 5 = 49 * k₅)) ∧ 
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    551245 + 5 = 9 * k₁ ∧ 
    551245 + 5 = 70 * k₂ ∧ 
    551245 + 5 = 25 * k₃ ∧ 
    551245 + 5 = 21 * k₄ ∧ 
    551245 + 5 = 49 * k₅) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2779_277974


namespace NUMINAMATH_CALUDE_problem_solution_l2779_277923

theorem problem_solution : 2^(0^(1^9)) + ((2^0)^1)^9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2779_277923


namespace NUMINAMATH_CALUDE_sarah_trucks_l2779_277977

theorem sarah_trucks (trucks_to_jeff trucks_to_ashley trucks_remaining : ℕ) 
  (h1 : trucks_to_jeff = 13)
  (h2 : trucks_to_ashley = 21)
  (h3 : trucks_remaining = 38) :
  trucks_to_jeff + trucks_to_ashley + trucks_remaining = 72 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_l2779_277977


namespace NUMINAMATH_CALUDE_krishans_money_l2779_277913

theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 490 →
  krishan = 2890 := by
sorry

end NUMINAMATH_CALUDE_krishans_money_l2779_277913


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l2779_277910

theorem ordered_pairs_count : 
  ∃! (pairs : List (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ 6 / m + 3 / n = 1) ∧
    pairs.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l2779_277910


namespace NUMINAMATH_CALUDE_angle_triple_complement_l2779_277901

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l2779_277901


namespace NUMINAMATH_CALUDE_container_fullness_l2779_277928

theorem container_fullness 
  (capacity : ℝ) 
  (initial_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : capacity = 120)
  (h2 : initial_percentage = 0.3)
  (h3 : added_water = 54) :
  (initial_percentage * capacity + added_water) / capacity = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_container_fullness_l2779_277928


namespace NUMINAMATH_CALUDE_negative_integer_solutions_l2779_277963

def inequality_system (x : ℤ) : Prop :=
  2 * x + 9 ≥ 3 ∧ (1 + 2 * x) / 3 + 1 > x

def is_negative_integer (x : ℤ) : Prop :=
  x < 0

theorem negative_integer_solutions :
  {x : ℤ | inequality_system x ∧ is_negative_integer x} = {-3, -2, -1} :=
sorry

end NUMINAMATH_CALUDE_negative_integer_solutions_l2779_277963


namespace NUMINAMATH_CALUDE_sticker_distribution_ways_l2779_277925

def distribute_stickers (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem sticker_distribution_ways :
  distribute_stickers 10 5 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_ways_l2779_277925


namespace NUMINAMATH_CALUDE_simplify_fraction_l2779_277919

theorem simplify_fraction : (15^30) / (45^15) = 5^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2779_277919


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2779_277902

theorem gcd_of_specific_numbers : Nat.gcd 333333333 666666666 = 333333333 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2779_277902


namespace NUMINAMATH_CALUDE_marble_arrangement_mod_l2779_277954

/-- The number of blue marbles -/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that maintains the balance -/
def yellow_marbles : ℕ := 18

/-- The total number of marbles -/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of different arrangements -/
def N : ℕ := (Nat.choose total_marbles blue_marbles)

theorem marble_arrangement_mod :
  N % 1000 = 564 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_mod_l2779_277954


namespace NUMINAMATH_CALUDE_common_root_sum_k_l2779_277964

theorem common_root_sum_k : ∃ (k₁ k₂ : ℝ),
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k₁ = 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k₂ = 0) ∧
  k₁ ≠ k₂ ∧
  k₁ + k₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_common_root_sum_k_l2779_277964


namespace NUMINAMATH_CALUDE_sum_of_edges_l2779_277941

/-- A rectangular solid with given properties -/
structure RectangularSolid where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  volume_eq : a * b * c = 8
  surface_area_eq : 2 * (a * b + b * c + c * a) = 32
  width_sq_eq : b ^ 2 = a * c

/-- The sum of all edges of the rectangular solid is 32 -/
theorem sum_of_edges (solid : RectangularSolid) :
  4 * (solid.a + solid.b + solid.c) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edges_l2779_277941


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l2779_277993

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original current : ℕ) : ℕ :=
  current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l2779_277993


namespace NUMINAMATH_CALUDE_square_garden_area_l2779_277946

theorem square_garden_area (p : ℝ) (s : ℝ) : 
  p = 28 →                   -- The perimeter is 28 feet
  p = 4 * s →                -- Perimeter of a square is 4 times the side length
  s^2 = p + 21 →             -- Area is equal to perimeter plus 21
  s^2 = 49 :=                -- The area of the garden is 49 square feet
by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_l2779_277946


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2779_277936

theorem right_triangle_hypotenuse (L M N : ℝ) : 
  -- LMN is a right triangle with right angle at M
  -- sin N = 3/5
  -- LM = 18
  Real.sin N = 3/5 → LM = 18 → LN = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2779_277936


namespace NUMINAMATH_CALUDE_tony_winnings_l2779_277907

/-- Calculates the winnings for a single lottery ticket -/
def ticket_winnings (winning_numbers : ℕ) : ℕ :=
  if winning_numbers ≤ 2 then
    15 * winning_numbers
  else
    30 + 20 * (winning_numbers - 2)

/-- Represents Tony's lottery tickets and calculates total winnings -/
def total_winnings : ℕ :=
  ticket_winnings 3 + ticket_winnings 5 + ticket_winnings 2 + ticket_winnings 4

/-- Theorem stating that Tony's total winnings are $240 -/
theorem tony_winnings : total_winnings = 240 := by
  sorry

end NUMINAMATH_CALUDE_tony_winnings_l2779_277907


namespace NUMINAMATH_CALUDE_derivative_f_at_half_l2779_277991

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- State the theorem
theorem derivative_f_at_half : 
  deriv f (1/2) = -2 :=
sorry

end NUMINAMATH_CALUDE_derivative_f_at_half_l2779_277991


namespace NUMINAMATH_CALUDE_tenfold_largest_two_digit_l2779_277903

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit : 10 * largest_two_digit_number = 990 := by
  sorry

end NUMINAMATH_CALUDE_tenfold_largest_two_digit_l2779_277903


namespace NUMINAMATH_CALUDE_joseph_decks_l2779_277961

/-- The number of complete decks given a total number of cards and cards per deck -/
def number_of_decks (total_cards : ℕ) (cards_per_deck : ℕ) : ℕ :=
  total_cards / cards_per_deck

/-- Proof that Joseph has 4 complete decks of cards -/
theorem joseph_decks :
  number_of_decks 208 52 = 4 := by
  sorry

end NUMINAMATH_CALUDE_joseph_decks_l2779_277961


namespace NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l2779_277980

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  a : ℕ  -- Side length of the smallest square
  b : ℕ  -- Side length of the second smallest square
  length : ℕ  -- Length of the rectangle
  width : ℕ   -- Width of the rectangle

/-- The perimeter of a rectangle -/
def perimeter (rect : NineSquareRectangle) : ℕ :=
  2 * (rect.length + rect.width)

/-- Conditions for a valid NineSquareRectangle configuration -/
def is_valid_configuration (rect : NineSquareRectangle) : Prop :=
  rect.b = 3 * rect.a ∧
  rect.length = 2 * rect.a + rect.b + 3 * rect.a + rect.b ∧
  rect.width = 12 * rect.a - 2 * rect.b + 8 * rect.a - rect.b

/-- Theorem stating the smallest possible perimeter of a NineSquareRectangle -/
theorem min_perimeter_nine_square_rectangle :
  ∀ rect : NineSquareRectangle, is_valid_configuration rect →
  perimeter rect ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l2779_277980


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2779_277904

/-- Given a line in vector form, prove its slope-intercept form and find (m, b) -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y + 4) = 0 →
  y = 2 * x - 10 ∧ (2, -10) = (2, -10) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2779_277904


namespace NUMINAMATH_CALUDE_expression_value_l2779_277956

theorem expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x*(x - 3) + (x + 1)*(x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2779_277956


namespace NUMINAMATH_CALUDE_wedge_volume_l2779_277973

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h : ℝ) (θ : ℝ) : 
  d = 20 → θ = 30 → (π * (d/2)^2 * h * θ) / 360 = (500/3) * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l2779_277973


namespace NUMINAMATH_CALUDE_peach_trees_count_l2779_277987

/-- The number of peach trees in an orchard. -/
def number_of_peach_trees (apple_trees : ℕ) (apple_yield : ℕ) (peach_yield : ℕ) (total_yield : ℕ) : ℕ :=
  (total_yield - apple_trees * apple_yield) / peach_yield

/-- Theorem stating the number of peach trees in the orchard. -/
theorem peach_trees_count : number_of_peach_trees 30 150 65 7425 = 45 := by
  sorry

end NUMINAMATH_CALUDE_peach_trees_count_l2779_277987


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2779_277945

/-- Calculates the number of pages left to be read in a book --/
theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (daily_reading : ℕ) 
  (days : ℕ) 
  (h1 : total_pages = 381) 
  (h2 : pages_read = 149) 
  (h3 : daily_reading = 20) 
  (h4 : days = 7) :
  total_pages - (pages_read + daily_reading * days) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l2779_277945


namespace NUMINAMATH_CALUDE_total_ants_employed_l2779_277943

/-- The total number of ants employed for all construction tasks -/
def total_ants (carrying_red carrying_black digging_red digging_black assembling_red assembling_black : ℕ) : ℕ :=
  carrying_red + carrying_black + digging_red + digging_black + assembling_red + assembling_black

/-- Theorem stating that the total number of ants employed is 2464 -/
theorem total_ants_employed :
  total_ants 413 487 356 518 298 392 = 2464 := by
  sorry

#eval total_ants 413 487 356 518 298 392

end NUMINAMATH_CALUDE_total_ants_employed_l2779_277943


namespace NUMINAMATH_CALUDE_g_geq_one_l2779_277966

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

-- Define the function g
def g (x : ℝ) : ℝ := Real.exp (x - 1) + 3 * x^2 + 4 - f x

-- Theorem statement
theorem g_geq_one (x : ℝ) (h : x > 0) : g x ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_g_geq_one_l2779_277966


namespace NUMINAMATH_CALUDE_bella_needs_twelve_beads_l2779_277908

/-- Given the number of friends, beads per bracelet, and beads on hand,
    calculate the number of additional beads needed. -/
def additional_beads_needed (friends : ℕ) (beads_per_bracelet : ℕ) (beads_on_hand : ℕ) : ℕ :=
  max 0 (friends * beads_per_bracelet - beads_on_hand)

/-- Proof that Bella needs 12 more beads to make bracelets for her friends. -/
theorem bella_needs_twelve_beads :
  additional_beads_needed 6 8 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bella_needs_twelve_beads_l2779_277908


namespace NUMINAMATH_CALUDE_martha_has_19_butterflies_l2779_277976

/-- The number of butterflies in Martha's collection -/
structure ButterflyCollection where
  blue : ℕ
  yellow : ℕ
  black : ℕ

/-- Martha's butterfly collection satisfies the given conditions -/
def marthasCollection : ButterflyCollection where
  blue := 6
  yellow := 3
  black := 10

/-- The total number of butterflies in a collection -/
def totalButterflies (c : ButterflyCollection) : ℕ :=
  c.blue + c.yellow + c.black

/-- Theorem stating that Martha's collection has 19 butterflies in total -/
theorem martha_has_19_butterflies :
  totalButterflies marthasCollection = 19 ∧
  marthasCollection.blue = 2 * marthasCollection.yellow :=
by
  sorry


end NUMINAMATH_CALUDE_martha_has_19_butterflies_l2779_277976


namespace NUMINAMATH_CALUDE_paper_airplane_competition_l2779_277952

theorem paper_airplane_competition
  (a b h v m : ℝ)
  (total : a + b + h + v + m = 41)
  (matyas_least : m ≤ a ∧ m ≤ b ∧ m ≤ h ∧ m ≤ v)
  (andelka_matyas : a = m + 0.9)
  (vlada_andelka : v = a + 0.6)
  (honzik_furthest : h > a ∧ h > b ∧ h > v ∧ h > m)
  (honzik_whole : ∃ n : ℕ, h = n)
  (avg_difference : (a + v + m) / 3 = (a + b + h + v + m) / 5 - 0.2) :
  a = 8.1 ∧ b = 8 ∧ h = 9 ∧ v = 8.7 ∧ m = 7.2 := by
sorry

end NUMINAMATH_CALUDE_paper_airplane_competition_l2779_277952


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2779_277951

def f (a : ℝ) (x : ℝ) := x^2 - 2*a*x + 1

theorem necessary_but_not_sufficient (a : ℝ) :
  (a ≤ 0 → ∀ x y, 1 ≤ x → x < y → f a x < f a y) ∧
  (∃ a > 0, ∀ x y, 1 ≤ x → x < y → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2779_277951


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2779_277981

theorem sum_of_coefficients_zero (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (x^2 + x + 1) * (2*x - a)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₀ = -32 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2779_277981


namespace NUMINAMATH_CALUDE_team_a_remaining_days_l2779_277944

/-- The number of days Team A needs to complete the project alone initially -/
def team_a_initial_days : ℚ := 24

/-- The number of days Team B needs to complete the project alone initially -/
def team_b_initial_days : ℚ := 18

/-- The number of days Team A needs to complete the project after receiving 6 people from Team B -/
def team_a_after_transfer_days : ℚ := 18

/-- The number of days Team B needs to complete the project after transferring 6 people to Team A -/
def team_b_after_transfer_days : ℚ := 24

/-- The number of days Team B works alone -/
def team_b_alone_days : ℚ := 6

/-- The number of days both teams work together -/
def teams_together_days : ℚ := 4

/-- The efficiency of one person per day -/
def efficiency_per_person : ℚ := 1 / 432

/-- The number of people in Team A -/
def team_a_people : ℚ := team_a_initial_days / efficiency_per_person

/-- The number of people in Team B -/
def team_b_people : ℚ := team_b_initial_days / efficiency_per_person

/-- The theorem stating that Team A needs 26/3 more days to complete the project -/
theorem team_a_remaining_days : 
  ∃ (m : ℚ), 
    (team_a_people * efficiency_per_person * (team_b_alone_days + teams_together_days) + 
     team_b_people * teams_together_days * efficiency_per_person + 
     team_a_people * m * efficiency_per_person = 1) ∧ 
    m = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_team_a_remaining_days_l2779_277944


namespace NUMINAMATH_CALUDE_transportation_charges_l2779_277985

theorem transportation_charges 
  (purchase_price : ℕ) 
  (repair_cost : ℕ) 
  (profit_percentage : ℚ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 13000)
  (h2 : repair_cost = 5000)
  (h3 : profit_percentage = 1/2)
  (h4 : selling_price = 28500) :
  ∃ (transportation_charges : ℕ),
    selling_price = (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) ∧
    transportation_charges = 1000 :=
by sorry

end NUMINAMATH_CALUDE_transportation_charges_l2779_277985


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2779_277995

theorem quadratic_function_property (a b : ℝ) : 
  let f := λ x : ℝ => x^2 + a*x + b
  (f 1 = 0) → (f 2 = 0) → (f (-1) = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2779_277995


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2779_277921

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics --/
structure Survey where
  population : ℕ
  sample_size : ℕ
  has_groups : Bool

/-- Determines the most appropriate sampling method for a given survey --/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_groups then SamplingMethod.Stratified
  else if s.population > 100 then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The yogurt box survey --/
def yogurt_survey : Survey :=
  { population := 10, sample_size := 3, has_groups := false }

/-- The audience survey --/
def audience_survey : Survey :=
  { population := 1280, sample_size := 32, has_groups := false }

/-- The school staff survey --/
def staff_survey : Survey :=
  { population := 160, sample_size := 20, has_groups := true }

theorem correct_sampling_methods :
  best_sampling_method yogurt_survey = SamplingMethod.SimpleRandom ∧
  best_sampling_method audience_survey = SamplingMethod.Systematic ∧
  best_sampling_method staff_survey = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2779_277921


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2779_277950

theorem closest_integer_to_cube_root_150 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (150 : ℝ)^(1/3)| ≤ |m - (150 : ℝ)^(1/3)| ∧ n = 5 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2779_277950


namespace NUMINAMATH_CALUDE_tulip_bouquet_combinations_l2779_277959

theorem tulip_bouquet_combinations (n : ℕ) (max_tulips : ℕ) (total_money : ℕ) (tulip_cost : ℕ) : 
  n = 11 → 
  max_tulips = 11 → 
  total_money = 550 → 
  tulip_cost = 49 → 
  (Finset.filter (fun k => k % 2 = 1 ∧ k ≤ max_tulips) (Finset.range (n + 1))).card = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_tulip_bouquet_combinations_l2779_277959


namespace NUMINAMATH_CALUDE_divisibility_problem_l2779_277935

theorem divisibility_problem (a b c : ℤ) 
  (h1 : a ∣ b * c - 1) 
  (h2 : b ∣ c * a - 1) 
  (h3 : c ∣ a * b - 1) : 
  a * b * c ∣ a * b + b * c + c * a - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2779_277935


namespace NUMINAMATH_CALUDE_square_of_opposites_are_equal_l2779_277924

theorem square_of_opposites_are_equal (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ 0) : a^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_opposites_are_equal_l2779_277924


namespace NUMINAMATH_CALUDE_mountain_hike_l2779_277947

theorem mountain_hike (rate_up : ℝ) (time : ℝ) (rate_down_factor : ℝ) : 
  rate_up = 3 →
  time = 2 →
  rate_down_factor = 1.5 →
  (rate_up * time) * rate_down_factor = 9 := by
  sorry

end NUMINAMATH_CALUDE_mountain_hike_l2779_277947


namespace NUMINAMATH_CALUDE_absolute_value_properties_problem_solutions_l2779_277917

theorem absolute_value_properties :
  (∀ x y : ℝ, |x - y| = |y - x|) ∧
  (∀ x : ℝ, |x| ≥ 0) ∧
  (∀ x : ℝ, |x| = 0 ↔ x = 0) ∧
  (∀ x y : ℝ, |x + y| ≤ |x| + |y|) :=
sorry

theorem problem_solutions :
  (|3 - (-2)| = 5) ∧
  (∀ x : ℝ, |x + 2| = 3 → (x = 1 ∨ x = -5)) ∧
  (∃ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 3| ≥ m) ∧ (∃ x : ℝ, |x - 1| + |x + 3| = m) ∧ m = 4) ∧
  (∃ m : ℝ, (∀ x : ℝ, |x + 1| + |x - 2| + |x - 4| ≥ m) ∧ (|2 + 1| + |2 - 2| + |2 - 4| = m) ∧ m = 5) ∧
  (∀ x y z : ℝ, (|x + 1| + |x - 2|) * (|y - 2| + |y + 1|) * (|z - 3| + |z + 1|) = 36 →
    (-3 ≤ x + y + z ∧ x + y + z ≤ 7)) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_properties_problem_solutions_l2779_277917


namespace NUMINAMATH_CALUDE_largest_decimal_l2779_277940

theorem largest_decimal : ∀ (a b c d e : ℝ), 
  a = 0.997 → b = 0.9797 → c = 0.97 → d = 0.979 → e = 0.9709 →
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l2779_277940


namespace NUMINAMATH_CALUDE_shopkeeper_payment_l2779_277988

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_successive_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem shopkeeper_payment (porcelain_price crystal_price : ℝ)
  (porcelain_discounts crystal_discounts : List ℝ) :
  porcelain_price = 8500 →
  crystal_price = 1500 →
  porcelain_discounts = [0.25, 0.15, 0.05] →
  crystal_discounts = [0.30, 0.10, 0.05] →
  (apply_successive_discounts porcelain_price porcelain_discounts +
   apply_successive_discounts crystal_price crystal_discounts) = 6045.56 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_payment_l2779_277988


namespace NUMINAMATH_CALUDE_second_third_ratio_l2779_277982

theorem second_third_ratio (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B = 30 →
  B / C = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_second_third_ratio_l2779_277982


namespace NUMINAMATH_CALUDE_exists_natural_number_with_seventh_eighth_root_natural_l2779_277968

theorem exists_natural_number_with_seventh_eighth_root_natural :
  ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7/8) = m := by
  sorry

end NUMINAMATH_CALUDE_exists_natural_number_with_seventh_eighth_root_natural_l2779_277968


namespace NUMINAMATH_CALUDE_sine_graph_shift_l2779_277990

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x + π/8)) = 3 * Real.sin (2 * x + π/4) :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l2779_277990
