import Mathlib

namespace NUMINAMATH_CALUDE_min_ticket_cost_l3906_390646

theorem min_ticket_cost (total_tickets : ℕ) (price_low price_high : ℕ) 
  (h_total : total_tickets = 140)
  (h_price_low : price_low = 6)
  (h_price_high : price_high = 10)
  (h_constraint : ∀ x : ℕ, x ≤ total_tickets → total_tickets - x ≥ 2 * x → x ≤ 46) :
  ∃ (low_count high_count : ℕ),
    low_count + high_count = total_tickets ∧
    high_count ≥ 2 * low_count ∧
    low_count = 46 ∧
    high_count = 94 ∧
    low_count * price_low + high_count * price_high = 1216 ∧
    (∀ (a b : ℕ), a + b = total_tickets → b ≥ 2 * a → 
      a * price_low + b * price_high ≥ 1216) :=
by sorry

end NUMINAMATH_CALUDE_min_ticket_cost_l3906_390646


namespace NUMINAMATH_CALUDE_woman_work_days_l3906_390657

-- Define the work rates
def man_rate : ℚ := 1 / 6
def boy_rate : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 3

-- Define the woman's work rate
def woman_rate : ℚ := combined_rate - man_rate - boy_rate

-- Theorem to prove
theorem woman_work_days : (1 : ℚ) / woman_rate = 12 := by
  sorry


end NUMINAMATH_CALUDE_woman_work_days_l3906_390657


namespace NUMINAMATH_CALUDE_two_y_squared_over_x_is_fraction_l3906_390613

/-- A fraction is an expression with a variable in the denominator -/
def is_fraction (numerator denominator : ℚ) : Prop :=
  ∃ (x : ℚ), denominator = x

/-- The expression 2y^2/x is a fraction -/
theorem two_y_squared_over_x_is_fraction (x y : ℚ) :
  is_fraction (2 * y^2) x :=
sorry

end NUMINAMATH_CALUDE_two_y_squared_over_x_is_fraction_l3906_390613


namespace NUMINAMATH_CALUDE_trig_simplification_l3906_390653

theorem trig_simplification :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3906_390653


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3906_390647

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, x) acc => acc + if x then 2^i else 0) 0

def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.enum.foldr (fun (i, x) acc => acc + x * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary_num := [true, false, true, true]
  let ternary_num := [2, 1, 2]
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 299 := by
sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3906_390647


namespace NUMINAMATH_CALUDE_cube_of_negative_double_l3906_390677

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_double_l3906_390677


namespace NUMINAMATH_CALUDE_connie_marbles_l3906_390610

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℝ := 183.0

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.0

/-- The initial number of marbles Connie had -/
def initial_marbles : ℝ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776.0 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l3906_390610


namespace NUMINAMATH_CALUDE_product_sum_relation_l3906_390623

theorem product_sum_relation (a b : ℤ) : 
  (a * b = 2 * (a + b) + 11) → (b = 7) → (b - a = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3906_390623


namespace NUMINAMATH_CALUDE_trains_crossing_time_l3906_390628

/-- Time for two trains to cross each other -/
theorem trains_crossing_time
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : length1 = 140)
  (h2 : length2 = 160)
  (h3 : speed1 = 60)
  (h4 : speed2 = 48)
  : (length1 + length2) / (speed1 + speed2) * (1000 / 3600) = 10 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l3906_390628


namespace NUMINAMATH_CALUDE_triangle_inradius_l3906_390634

/-- Given a triangle with perimeter 60 cm and area 75 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) :
  perimeter = 60 ∧ area = 75 → inradius = 2.5 := by
  sorry

#check triangle_inradius

end NUMINAMATH_CALUDE_triangle_inradius_l3906_390634


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_twelve_zeroes_l3906_390603

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem fifteen_factorial_base_twelve_zeroes :
  ∃ k : ℕ, k = 5 ∧ 12^k ∣ factorial 15 ∧ ¬(12^(k+1) ∣ factorial 15) :=
by sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_twelve_zeroes_l3906_390603


namespace NUMINAMATH_CALUDE_pond_width_l3906_390681

/-- The width of a rectangular pond, given its length, depth, and volume -/
theorem pond_width (length : ℝ) (depth : ℝ) (volume : ℝ) : 
  length = 28 → depth = 5 → volume = 1400 → volume = length * depth * 10 := by
  sorry

end NUMINAMATH_CALUDE_pond_width_l3906_390681


namespace NUMINAMATH_CALUDE_system_solution_relation_l3906_390639

theorem system_solution_relation (a₁ a₂ c₁ c₂ : ℝ) :
  (2 * a₁ + 3 = c₁ ∧ 2 * a₂ + 3 = c₂) →
  (-1 * a₁ + (-3) = a₁ - c₁ ∧ -1 * a₂ + (-3) = a₂ - c₂) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_relation_l3906_390639


namespace NUMINAMATH_CALUDE_function_characterization_l3906_390675

theorem function_characterization (f : ℕ → ℕ) : 
  (∀ m n : ℕ, 2 * f (m * n) ≥ f (m^2 + n^2) - f m^2 - f n^2 ∧ 
               f (m^2 + n^2) - f m^2 - f n^2 ≥ 2 * f m * f n) → 
  (∀ n : ℕ, f n = n^2) := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l3906_390675


namespace NUMINAMATH_CALUDE_ps_length_l3906_390696

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (PQRS : Quadrilateral) : Prop :=
  let (px, py) := PQRS.P
  let (qx, qy) := PQRS.Q
  let (rx, ry) := PQRS.R
  let (sx, sy) := PQRS.S
  -- PQ = 6
  (px - qx)^2 + (py - qy)^2 = 36 ∧
  -- QR = 10
  (qx - rx)^2 + (qy - ry)^2 = 100 ∧
  -- RS = 25
  (rx - sx)^2 + (ry - sy)^2 = 625 ∧
  -- Angle Q is right angle
  (px - qx) * (rx - qx) + (py - qy) * (ry - qy) = 0 ∧
  -- Angle R is right angle
  (qx - rx) * (sx - rx) + (qy - ry) * (sy - ry) = 0

-- Theorem statement
theorem ps_length (PQRS : Quadrilateral) (h : is_valid_quadrilateral PQRS) :
  (PQRS.P.1 - PQRS.S.1)^2 + (PQRS.P.2 - PQRS.S.2)^2 = 461 :=
by sorry

end NUMINAMATH_CALUDE_ps_length_l3906_390696


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l3906_390617

/-- The time when a ball thrown upward hits the ground -/
theorem ball_hit_ground_time : ∃ t : ℚ, t = 10 / 7 ∧ -4.9 * t^2 + 4 * t + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l3906_390617


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l3906_390645

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l3906_390645


namespace NUMINAMATH_CALUDE_inequality_range_l3906_390671

theorem inequality_range :
  {a : ℝ | ∀ x : ℝ, a * (4 - Real.sin x)^4 - 3 + (Real.cos x)^2 + a > 0} = {a : ℝ | a > 3/82} := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3906_390671


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_seven_fifths_l3906_390658

def h (x : ℝ) := 5*x + 7

theorem h_zero_iff_b_eq_neg_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = -7/5 := by sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_seven_fifths_l3906_390658


namespace NUMINAMATH_CALUDE_correct_experimental_procedure_l3906_390694

-- Define the type for experimental procedures
inductive ExperimentalProcedure
| MicroorganismIsolation
| WineFermentation
| CellObservation
| EcoliCounting

-- Define the properties of each procedure
def requiresLight (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.MicroorganismIsolation => false
  | _ => true

def requiresOpenBottle (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.WineFermentation => false
  | _ => true

def adjustAperture (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.CellObservation => false
  | _ => true

def ensureDilution (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.EcoliCounting => true
  | _ => false

-- Theorem stating that E. coli counting is the correct procedure
theorem correct_experimental_procedure :
  ∀ p : ExperimentalProcedure,
    (p = ExperimentalProcedure.EcoliCounting) ↔
    (¬requiresLight p ∧ ¬requiresOpenBottle p ∧ ¬adjustAperture p ∧ ensureDilution p) :=
by sorry

end NUMINAMATH_CALUDE_correct_experimental_procedure_l3906_390694


namespace NUMINAMATH_CALUDE_systematic_sampling_l3906_390622

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (included_number : Nat) (group_number : Nat) : 
  total_students = 50 →
  sample_size = 10 →
  included_number = 46 →
  group_number = 7 →
  (included_number - (3 * (total_students / sample_size))) = 31 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3906_390622


namespace NUMINAMATH_CALUDE_perimeter_of_square_d_l3906_390620

/-- Given two squares C and D, where C has a perimeter of 32 cm and D has an area that is half the area of C,
    prove that the perimeter of D is 16√2 cm. -/
theorem perimeter_of_square_d (C D : Real) : 
  (C = 32) →  -- Perimeter of square C is 32 cm
  (D^2 = (C/4)^2 / 2) →  -- Area of D is half the area of C
  (4 * D = 16 * Real.sqrt 2) :=  -- Perimeter of D is 16√2 cm
by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_d_l3906_390620


namespace NUMINAMATH_CALUDE_worker_savings_proof_l3906_390650

/-- Represents the fraction of take-home pay saved each month -/
def savings_fraction : ℚ := 2 / 5

theorem worker_savings_proof (monthly_pay : ℝ) (h1 : monthly_pay > 0) :
  let yearly_savings := 12 * savings_fraction * monthly_pay
  let monthly_not_saved := (1 - savings_fraction) * monthly_pay
  yearly_savings = 8 * monthly_not_saved :=
by sorry

end NUMINAMATH_CALUDE_worker_savings_proof_l3906_390650


namespace NUMINAMATH_CALUDE_abc_inequality_l3906_390699

/-- Given a = √2, b = √7 - √3, and c = √6 - √2, prove that a > c > b -/
theorem abc_inequality :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 7 - Real.sqrt 3
  let c : ℝ := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3906_390699


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l3906_390632

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem matrix_inverse_proof :
  A⁻¹ = !![(-1), (-3); (-2), (-5)] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l3906_390632


namespace NUMINAMATH_CALUDE_mean_median_difference_l3906_390621

/-- Represents the score distribution on a math test -/
structure ScoreDistribution where
  score65 : ℝ
  score75 : ℝ
  score85 : ℝ
  score92 : ℝ
  score98 : ℝ
  sum_to_one : score65 + score75 + score85 + score92 + score98 = 1

/-- Calculates the mean score given a score distribution -/
def mean_score (sd : ScoreDistribution) : ℝ :=
  65 * sd.score65 + 75 * sd.score75 + 85 * sd.score85 + 92 * sd.score92 + 98 * sd.score98

/-- Determines the median score given a score distribution -/
noncomputable def median_score (sd : ScoreDistribution) : ℝ :=
  if sd.score65 + sd.score75 > 0.5 then 75
  else if sd.score65 + sd.score75 + sd.score85 > 0.5 then 85
  else if sd.score65 + sd.score75 + sd.score85 + sd.score92 > 0.5 then 92
  else 98

/-- Theorem stating that the absolute difference between mean and median is 1.05 -/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.score65 = 0.15)
  (h2 : sd.score75 = 0.20)
  (h3 : sd.score85 = 0.30)
  (h4 : sd.score92 = 0.10) :
  |mean_score sd - median_score sd| = 1.05 := by
  sorry


end NUMINAMATH_CALUDE_mean_median_difference_l3906_390621


namespace NUMINAMATH_CALUDE_power_function_inequality_l3906_390684

/-- A power function that passes through the point (2,√2) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem stating the inequality for any two points on the graph of f -/
theorem power_function_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  x₂ * f x₁ > x₁ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l3906_390684


namespace NUMINAMATH_CALUDE_martin_trip_distance_is_1185_l3906_390625

/-- Calculates the total distance traveled during Martin's business trip --/
def martin_trip_distance : ℝ :=
  let segment1 := 70 * 3
  let segment2 := 80 * 4
  let segment3 := 65 * 3
  let segment4 := 50 * 2
  let segment5 := 90 * 4
  segment1 + segment2 + segment3 + segment4 + segment5

/-- Theorem stating that Martin's total trip distance is 1185 km --/
theorem martin_trip_distance_is_1185 :
  martin_trip_distance = 1185 := by sorry

end NUMINAMATH_CALUDE_martin_trip_distance_is_1185_l3906_390625


namespace NUMINAMATH_CALUDE_cheese_fries_cost_is_eight_l3906_390698

/-- Represents the cost of items and money brought by Jim and his cousin --/
structure RestaurantScenario where
  cheeseburger_cost : ℚ
  milkshake_cost : ℚ
  jim_money : ℚ
  cousin_money : ℚ
  spent_percentage : ℚ

/-- Calculates the cost of cheese fries given a RestaurantScenario --/
def cheese_fries_cost (scenario : RestaurantScenario) : ℚ :=
  let total_money := scenario.jim_money + scenario.cousin_money
  let total_spent := scenario.spent_percentage * total_money
  let burger_shake_cost := 2 * (scenario.cheeseburger_cost + scenario.milkshake_cost)
  total_spent - burger_shake_cost

/-- Theorem stating that the cost of cheese fries is 8 given the specific scenario --/
theorem cheese_fries_cost_is_eight :
  let scenario := {
    cheeseburger_cost := 3,
    milkshake_cost := 5,
    jim_money := 20,
    cousin_money := 10,
    spent_percentage := 4/5
  }
  cheese_fries_cost scenario = 8 := by
  sorry


end NUMINAMATH_CALUDE_cheese_fries_cost_is_eight_l3906_390698


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3906_390655

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ 2 < x ∧ x < 3) 
  (h2 : a < 0) :
  ∀ x, c*x^2 - b*x + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3906_390655


namespace NUMINAMATH_CALUDE_arithmetic_equation_l3906_390663

theorem arithmetic_equation : 12 - 11 + (9 * 8) + 7 - (6 * 5) + 4 - 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l3906_390663


namespace NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l3906_390614

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard -/
structure Checkerboard where
  square_size : ℝ

/-- Calculates the maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered_2inch_card (card : Card) (board : Checkerboard) :
  card.side_length = 2 → board.square_size = 1 → max_squares_covered card board = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l3906_390614


namespace NUMINAMATH_CALUDE_not_divisible_by_100_l3906_390608

theorem not_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_100_l3906_390608


namespace NUMINAMATH_CALUDE_min_value_theorem_l3906_390686

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 12) :
  2 * x + 3 * y + 6 * z ≥ 18 * Real.rpow 2 (1/3) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 12 ∧
  2 * x₀ + 3 * y₀ + 6 * z₀ = 18 * Real.rpow 2 (1/3) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3906_390686


namespace NUMINAMATH_CALUDE_roger_lawn_mowing_l3906_390609

/-- The number of lawns Roger had to mow -/
def total_lawns : ℕ := 14

/-- The amount Roger earns per lawn -/
def earnings_per_lawn : ℕ := 9

/-- The number of lawns Roger forgot to mow -/
def forgotten_lawns : ℕ := 8

/-- The total amount Roger actually earned -/
def actual_earnings : ℕ := 54

/-- Theorem stating that the total number of lawns Roger had to mow is 14 -/
theorem roger_lawn_mowing :
  total_lawns = (actual_earnings / earnings_per_lawn) + forgotten_lawns :=
by sorry

end NUMINAMATH_CALUDE_roger_lawn_mowing_l3906_390609


namespace NUMINAMATH_CALUDE_experts_win_probability_l3906_390601

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expertsScore : ℕ := 3

/-- The current score of Viewers -/
def viewersScore : ℕ := 4

/-- The number of rounds needed to win the game -/
def winningScore : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
def expertsWinProbability : ℝ := p^4 + 4 * p^3 * q

theorem experts_win_probability : 
  expertsWinProbability = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_l3906_390601


namespace NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l3906_390689

/-- Calculates the number of logs left in a cookfire after a given number of hours. -/
def logs_left (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℕ :=
  initial_logs + add_rate * hours - burn_rate * hours

/-- Theorem: After 3 hours, a cookfire that starts with 6 logs, burns 3 logs per hour, 
    and receives 2 logs at the end of each hour will have 3 logs left. -/
theorem cookfire_logs_after_three_hours :
  logs_left 6 3 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l3906_390689


namespace NUMINAMATH_CALUDE_parallel_condition_l3906_390643

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + y = 1
def l2 (a x y : ℝ) : Prop := 9 * x + a * y = 1

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ l2 a x y

-- Theorem statement
theorem parallel_condition (a : ℝ) :
  (a + 3 = 0 → parallel a) ∧ ¬(parallel a → a + 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3906_390643


namespace NUMINAMATH_CALUDE_three_digit_number_rearrangement_l3906_390667

theorem three_digit_number_rearrangement (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a ≠ 0 →
  (100 * a + 10 * b + c) + 
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 4422 →
  a + b + c ≥ 18 →
  100 * a + 10 * b + c = 785 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_rearrangement_l3906_390667


namespace NUMINAMATH_CALUDE_lyras_initial_budget_l3906_390627

-- Define the cost of the chicken bucket
def chicken_cost : ℕ := 12

-- Define the cost per pound of beef
def beef_cost_per_pound : ℕ := 3

-- Define the number of pounds of beef bought
def beef_pounds : ℕ := 5

-- Define the amount left in the budget
def amount_left : ℕ := 53

-- Theorem to prove
theorem lyras_initial_budget :
  chicken_cost + beef_cost_per_pound * beef_pounds + amount_left = 80 := by
  sorry

end NUMINAMATH_CALUDE_lyras_initial_budget_l3906_390627


namespace NUMINAMATH_CALUDE_product_sum_multiple_l3906_390682

theorem product_sum_multiple (a b m : ℤ) : 
  b = 9 → b - a = 5 → a * b = m * (a + b) + 10 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_multiple_l3906_390682


namespace NUMINAMATH_CALUDE_people_who_left_line_l3906_390691

theorem people_who_left_line (initial : ℕ) (joined : ℕ) (final : ℕ) : 
  initial = 7 → joined = 8 → final = 11 → initial - (initial - final + joined) = 4 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_line_l3906_390691


namespace NUMINAMATH_CALUDE_negative_number_identification_l3906_390606

theorem negative_number_identification (a b c d : ℝ) : 
  a = -6 ∧ b = 0 ∧ c = 0.2 ∧ d = 3 →
  (a < 0 ∧ b ≥ 0 ∧ c > 0 ∧ d > 0) := by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l3906_390606


namespace NUMINAMATH_CALUDE_certain_number_sum_l3906_390683

theorem certain_number_sum : ∃ x : ℝ, x = 5.46 - 3.97 ∧ x + 5.46 = 6.95 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_sum_l3906_390683


namespace NUMINAMATH_CALUDE_probability_blue_red_white_l3906_390693

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  blue : ℕ
  red : ℕ
  white : ℕ

/-- Calculates the probability of drawing a specific sequence of marbles -/
def probability_of_sequence (counts : MarbleCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.white
  (counts.blue : ℚ) / total *
  (counts.red : ℚ) / (total - 1) *
  (counts.white : ℚ) / (total - 2)

/-- The main theorem stating the probability of drawing blue, red, then white -/
theorem probability_blue_red_white :
  probability_of_sequence ⟨4, 3, 6⟩ = 6 / 143 := by
  sorry

end NUMINAMATH_CALUDE_probability_blue_red_white_l3906_390693


namespace NUMINAMATH_CALUDE_sum_of_r_values_l3906_390662

/-- Given two quadratic equations with a common real root, prove the sum of possible values of r -/
theorem sum_of_r_values (r : ℝ) : 
  (∃ x : ℝ, x^2 + (r-1)*x + 6 = 0 ∧ x^2 + (2*r+1)*x + 22 = 0) → 
  (∃ r1 r2 : ℝ, (r = r1 ∨ r = r2) ∧ r1 + r2 = 12/5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_r_values_l3906_390662


namespace NUMINAMATH_CALUDE_ellipse_touches_hyperbola_l3906_390676

/-- An ellipse touches a hyperbola if they share a common point and have the same tangent at that point -/
def touches (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x ∧
    (-(b / a) * (x / Real.sqrt (a ^ 2 - x ^ 2))) = -1 / x ^ 2

/-- If an ellipse with equation (x/a)^2 + (y/b)^2 = 1 touches a hyperbola with equation y = 1/x, then ab = 2 -/
theorem ellipse_touches_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  touches a b → a * b = 2 := by
  sorry

#check ellipse_touches_hyperbola

end NUMINAMATH_CALUDE_ellipse_touches_hyperbola_l3906_390676


namespace NUMINAMATH_CALUDE_roger_actual_earnings_l3906_390615

/-- Calculates Roger's earnings from mowing lawns --/
def roger_earnings (small_price medium_price large_price : ℕ)
                   (total_small total_medium total_large : ℕ)
                   (forgot_small forgot_medium forgot_large : ℕ) : ℕ :=
  (small_price * (total_small - forgot_small)) +
  (medium_price * (total_medium - forgot_medium)) +
  (large_price * (total_large - forgot_large))

/-- Theorem: Roger's actual earnings are $69 --/
theorem roger_actual_earnings :
  roger_earnings 9 12 15 5 4 5 2 3 3 = 69 := by
  sorry

end NUMINAMATH_CALUDE_roger_actual_earnings_l3906_390615


namespace NUMINAMATH_CALUDE_algebraic_expression_proof_l3906_390654

theorem algebraic_expression_proof (a b : ℝ) : 
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2) = 2 * b^2 ∧
  3 * a^2 + (4 * a * (-2) - a^2) - 2 * (a^2 + 2 * a * (-2) - (-2)^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_proof_l3906_390654


namespace NUMINAMATH_CALUDE_unique_prime_polynomial_l3906_390607

theorem unique_prime_polynomial : ∃! (n : ℕ), n > 0 ∧ Nat.Prime (n^3 - 9*n^2 + 27*n - 28) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_polynomial_l3906_390607


namespace NUMINAMATH_CALUDE_subtraction_inequality_l3906_390618

theorem subtraction_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_inequality_l3906_390618


namespace NUMINAMATH_CALUDE_solutions_for_14_solutions_for_0_1_solutions_for_neg_0_0544_l3906_390630

-- Define the equation
def f (x : ℝ) := (x - 1) * (2 * x - 3) * (3 * x - 4) * (6 * x - 5)

-- Theorem for a = 14
theorem solutions_for_14 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 14 ∧ f x₂ = 14 ∧
  ∀ x : ℝ, f x = 14 → x = x₁ ∨ x = x₂ :=
sorry

-- Theorem for a = 0.1
theorem solutions_for_0_1 :
  ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  f x₁ = 0.1 ∧ f x₂ = 0.1 ∧ f x₃ = 0.1 ∧ f x₄ = 0.1 ∧
  ∀ x : ℝ, f x = 0.1 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

-- Theorem for a = -0.0544
theorem solutions_for_neg_0_0544 :
  ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  f x₁ = -0.0544 ∧ f x₂ = -0.0544 ∧ f x₃ = -0.0544 ∧ f x₄ = -0.0544 ∧
  ∀ x : ℝ, f x = -0.0544 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

end NUMINAMATH_CALUDE_solutions_for_14_solutions_for_0_1_solutions_for_neg_0_0544_l3906_390630


namespace NUMINAMATH_CALUDE_simplify_expression_l3906_390690

theorem simplify_expression (y : ℝ) : (3*y)^3 - (2*y)*(y^2) + y^4 = 25*y^3 + y^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3906_390690


namespace NUMINAMATH_CALUDE_chlorine_atomic_weight_l3906_390673

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 68

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := total_weight - hydrogen_weight - 2 * oxygen_weight

theorem chlorine_atomic_weight : chlorine_weight = 35 := by
  sorry

end NUMINAMATH_CALUDE_chlorine_atomic_weight_l3906_390673


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l3906_390652

theorem smallest_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m → m ≥ n) ∧
  n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l3906_390652


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l3906_390602

theorem arithmetic_equalities : 
  (-(2^3) / 8 - 1/4 * (-2)^2 = -2) ∧ 
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l3906_390602


namespace NUMINAMATH_CALUDE_sn_double_angle_l3906_390642

-- Define the cosine and sine functions
noncomputable def cs : Real → Real := Real.cos
noncomputable def sn : Real → Real := Real.sin

-- Define the theorem
theorem sn_double_angle (α : Real) 
  (h1 : cs (α + Real.pi/2) = 3/5) 
  (h2 : -Real.pi/2 < α ∧ α < Real.pi/2) : 
  sn (2*α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sn_double_angle_l3906_390642


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3906_390688

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 6
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3906_390688


namespace NUMINAMATH_CALUDE_prime_sum_of_composites_l3906_390648

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem prime_sum_of_composites :
  (∃ p : ℕ, Nat.Prime p ∧ p = 13 ∧ 
    ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b) ∧
  (∀ p : ℕ, Nat.Prime p → p > 13 → 
    ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_of_composites_l3906_390648


namespace NUMINAMATH_CALUDE_area_invariant_under_opposite_vertex_translation_l3906_390611

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Moves a point by a given vector -/
def movePoint (p : Point2D) (v : Vector2D) : Point2D :=
  { x := p.x + v.x, y := p.y + v.y }

/-- Theorem: The area of a quadrilateral remains unchanged when two opposite vertices
    are moved by the same vector -/
theorem area_invariant_under_opposite_vertex_translation (q : Quadrilateral) (v : Vector2D) :
  let q' := { q with
    A := movePoint q.A v,
    C := movePoint q.C v
  }
  area q = area q' :=
sorry

end NUMINAMATH_CALUDE_area_invariant_under_opposite_vertex_translation_l3906_390611


namespace NUMINAMATH_CALUDE_equal_numbers_product_l3906_390678

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 22 →
  c = d →
  c * d = 529 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l3906_390678


namespace NUMINAMATH_CALUDE_complex_square_ratio_real_l3906_390641

theorem complex_square_ratio_real (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  ∃ (r : ℝ), (z₁ / z₂)^2 = r := by
sorry

end NUMINAMATH_CALUDE_complex_square_ratio_real_l3906_390641


namespace NUMINAMATH_CALUDE_kids_wearing_socks_l3906_390604

theorem kids_wearing_socks (total : ℕ) (wearing_shoes : ℕ) (wearing_both : ℕ) (barefoot : ℕ) :
  total = 22 →
  wearing_shoes = 8 →
  wearing_both = 6 →
  barefoot = 8 →
  total - barefoot - (wearing_shoes - wearing_both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_kids_wearing_socks_l3906_390604


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3906_390619

theorem sum_of_coefficients (x y : ℝ) : (2*x + 3*y)^12 = 244140625 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3906_390619


namespace NUMINAMATH_CALUDE_snail_distance_is_31_l3906_390624

def snail_path : List ℤ := [3, -5, 10, 2]

def distance (a b : ℤ) : ℕ := Int.natAbs (b - a)

def total_distance (path : List ℤ) : ℕ :=
  match path with
  | [] => 0
  | [_] => 0
  | x :: y :: rest => distance x y + total_distance (y :: rest)

theorem snail_distance_is_31 : total_distance snail_path = 31 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_is_31_l3906_390624


namespace NUMINAMATH_CALUDE_non_juniors_playing_instruments_l3906_390651

theorem non_juniors_playing_instruments (total_students : ℕ) 
  (junior_play_percent : ℚ) (non_junior_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) :
  total_students = 600 →
  junior_play_percent = 30 / 100 →
  non_junior_not_play_percent = 35 / 100 →
  total_not_play_percent = 40 / 100 →
  ∃ (non_juniors_playing : ℕ), non_juniors_playing = 334 :=
by sorry

end NUMINAMATH_CALUDE_non_juniors_playing_instruments_l3906_390651


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l3906_390665

/-- The number of wheels on a car -/
def car_wheels : ℕ := 4

/-- The number of wheels on a bike -/
def bike_wheels : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 2

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * car_wheels + num_bikes * bike_wheels

theorem parking_lot_wheels : total_wheels = 44 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l3906_390665


namespace NUMINAMATH_CALUDE_not_equal_to_eighteen_fifths_other_options_equal_l3906_390616

theorem not_equal_to_eighteen_fifths : (18 + 1) / (5 + 1) ≠ 18 / 5 := by
  sorry

theorem other_options_equal :
  6^2 / 10 = 18 / 5 ∧
  (1 / 5) * (6 * 3) = 18 / 5 ∧
  3.6 = 18 / 5 ∧
  Real.sqrt (324 / 25) = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_not_equal_to_eighteen_fifths_other_options_equal_l3906_390616


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3906_390661

theorem complex_modulus_problem (z : ℂ) : z * (1 + Complex.I) = 1 - Complex.I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3906_390661


namespace NUMINAMATH_CALUDE_positive_rational_solution_condition_l3906_390670

theorem positive_rational_solution_condition 
  (a b : ℚ) (x y : ℚ) 
  (h_product : x * y = a) 
  (h_sum : x + y = b) : 
  (∃ (k : ℚ), k > 0 ∧ b^2 / 4 - a = k^2) ↔ 
  (x > 0 ∧ y > 0 ∧ ∃ (m n : ℕ), x = m / n) :=
sorry

end NUMINAMATH_CALUDE_positive_rational_solution_condition_l3906_390670


namespace NUMINAMATH_CALUDE_function_equation_l3906_390672

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_equation (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
  f = fun x ↦ x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_equation_l3906_390672


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_36_seconds_l3906_390631

/-- Time for a train to pass a jogger --/
theorem train_passing_jogger (jogger_speed : Real) (train_speed : Real) 
  (train_length : Real) (initial_distance : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The train passes the jogger in 36 seconds --/
theorem train_passes_jogger_in_36_seconds : 
  train_passing_jogger 9 45 120 240 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_36_seconds_l3906_390631


namespace NUMINAMATH_CALUDE_new_person_weight_l3906_390685

def initial_group_size : ℕ := 4
def average_weight_increase : ℝ := 3
def replaced_person_weight : ℝ := 70

theorem new_person_weight :
  let total_weight_increase : ℝ := initial_group_size * average_weight_increase
  let new_person_weight : ℝ := replaced_person_weight + total_weight_increase
  new_person_weight = 82 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3906_390685


namespace NUMINAMATH_CALUDE_golden_ratio_geometric_sequence_l3906_390668

theorem golden_ratio_geometric_sequence : 
  let x : ℝ := (1 + Real.sqrt 5) / 2
  let int_part := ⌊x⌋
  let frac_part := x - int_part
  (frac_part * x = int_part * int_part) ∧ (int_part * x = x * x) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_geometric_sequence_l3906_390668


namespace NUMINAMATH_CALUDE_solve_equation_l3906_390666

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3906_390666


namespace NUMINAMATH_CALUDE_find_x_l3906_390640

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 14 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_find_x_l3906_390640


namespace NUMINAMATH_CALUDE_bales_in_barn_l3906_390633

/-- The number of bales in the barn after stacking more bales -/
def total_bales (initial : ℕ) (stacked : ℕ) : ℕ := initial + stacked

/-- Theorem: Given 22 initial bales and 67 newly stacked bales, the total is 89 bales -/
theorem bales_in_barn : total_bales 22 67 = 89 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l3906_390633


namespace NUMINAMATH_CALUDE_basketball_team_callbacks_l3906_390637

theorem basketball_team_callbacks (girls boys cut : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : cut = 39) :
  girls + boys - cut = 10 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_callbacks_l3906_390637


namespace NUMINAMATH_CALUDE_point_translation_coordinates_equal_l3906_390636

theorem point_translation_coordinates_equal (m : ℝ) : 
  let A : ℝ × ℝ := (m, 2)
  let B : ℝ × ℝ := (m + 1, 5)
  (B.1 = B.2) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_translation_coordinates_equal_l3906_390636


namespace NUMINAMATH_CALUDE_average_of_numbers_l3906_390679

def numbers : List ℕ := [12, 13, 14, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 125781 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3906_390679


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3906_390649

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def sequence_condition (a : ℕ → ℤ) : Prop :=
  a 2 - a 3 - a 7 - a 11 - a 13 + a 16 = 8

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sequence_condition a) :
  a 9 = -4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3906_390649


namespace NUMINAMATH_CALUDE_point_three_units_from_negative_four_l3906_390605

theorem point_three_units_from_negative_four (x : ℝ) : 
  (x = -4 - 3 ∨ x = -4 + 3) ↔ |x - (-4)| = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_three_units_from_negative_four_l3906_390605


namespace NUMINAMATH_CALUDE_all_students_same_classroom_l3906_390664

/-- The probability that all three students choose the same classroom when randomly selecting between two classrooms with equal probability. -/
theorem all_students_same_classroom (num_classrooms : ℕ) (num_students : ℕ) : 
  num_classrooms = 2 → num_students = 3 → (1 : ℚ) / 4 = 
    (1 : ℚ) / num_classrooms^num_students + (1 : ℚ) / num_classrooms^num_students :=
by sorry

end NUMINAMATH_CALUDE_all_students_same_classroom_l3906_390664


namespace NUMINAMATH_CALUDE_star_value_for_specific_conditions_l3906_390692

-- Define the operation * for non-zero integers
def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Theorem statement
theorem star_value_for_specific_conditions 
  (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 15) 
  (h4 : a * b = 36) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_for_specific_conditions_l3906_390692


namespace NUMINAMATH_CALUDE_max_red_points_l3906_390600

/-- Represents a point on the circle -/
structure Point where
  color : Bool  -- True for red, False for blue
  connections : Nat

/-- Represents the circle with its points -/
structure Circle where
  points : Finset Point
  total_points : Nat
  red_points : Nat
  blue_points : Nat
  valid_connections : Bool

/-- The main theorem statement -/
theorem max_red_points (c : Circle) : 
  c.total_points = 25 ∧ 
  c.red_points + c.blue_points = c.total_points ∧
  c.valid_connections ∧
  (∀ p q : Point, p ∈ c.points → q ∈ c.points → 
    p.color = true → q.color = true → p ≠ q → p.connections ≠ q.connections) →
  c.red_points ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_red_points_l3906_390600


namespace NUMINAMATH_CALUDE_range_of_f_l3906_390635

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3906_390635


namespace NUMINAMATH_CALUDE_original_number_proof_l3906_390674

theorem original_number_proof (x y : ℝ) : 
  x = 19 ∧ 8 * x + 3 * y = 203 → x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3906_390674


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l3906_390687

theorem arcsin_sqrt3_div2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l3906_390687


namespace NUMINAMATH_CALUDE_unicorn_flowers_theorem_l3906_390638

/-- The number of flowers that bloom per unicorn step -/
def flowers_per_step (num_unicorns : ℕ) (total_distance : ℕ) (step_length : ℕ) (total_flowers : ℕ) : ℚ :=
  total_flowers / (num_unicorns * (total_distance * 1000 / step_length))

/-- Theorem: Given the conditions, 4 flowers bloom per unicorn step -/
theorem unicorn_flowers_theorem (num_unicorns : ℕ) (total_distance : ℕ) (step_length : ℕ) (total_flowers : ℕ)
  (h1 : num_unicorns = 6)
  (h2 : total_distance = 9)
  (h3 : step_length = 3)
  (h4 : total_flowers = 72000) :
  flowers_per_step num_unicorns total_distance step_length total_flowers = 4 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_flowers_theorem_l3906_390638


namespace NUMINAMATH_CALUDE_ratio_calculations_l3906_390659

theorem ratio_calculations (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 ∧ 
  (A + C) / (2 * B + A) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculations_l3906_390659


namespace NUMINAMATH_CALUDE_max_baggies_count_l3906_390626

def cookies_per_bag : ℕ := 3
def chocolate_chip_cookies : ℕ := 2
def oatmeal_cookies : ℕ := 16

theorem max_baggies_count : 
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_baggies_count_l3906_390626


namespace NUMINAMATH_CALUDE_inverse_g_at_124_l3906_390697

noncomputable def g (x : ℝ) : ℝ := 5 * x^3 - 4 * x + 1

theorem inverse_g_at_124 : g⁻¹ 124 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_124_l3906_390697


namespace NUMINAMATH_CALUDE_expression_simplification_l3906_390656

theorem expression_simplification :
  (∀ a : ℝ, 2 * (a - 1) - (2 * a - 3) + 3 = 4) ∧
  (∀ x : ℝ, 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = x^2 - 3 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3906_390656


namespace NUMINAMATH_CALUDE_smallest_square_with_five_interior_points_l3906_390644

/-- A lattice point in 2D space -/
def LatticePoint := ℤ × ℤ

/-- The number of interior lattice points in a square with side length s -/
def interiorLatticePoints (s : ℕ) : ℕ := (s - 1) ^ 2

/-- The smallest square side length with exactly 5 interior lattice points -/
def smallestSquareSide : ℕ := 4

theorem smallest_square_with_five_interior_points :
  (∀ n < smallestSquareSide, interiorLatticePoints n ≠ 5) ∧
  interiorLatticePoints smallestSquareSide = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_with_five_interior_points_l3906_390644


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3906_390629

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equality : i / (2 + i) = (1 + 2*i) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3906_390629


namespace NUMINAMATH_CALUDE_workbook_arrangement_count_l3906_390680

/-- The number of ways to arrange 2 Korean and 2 English workbooks in a row with English workbooks side by side -/
def arrange_workbooks : ℕ :=
  let korean_books := 2
  let english_books := 2
  let total_units := korean_books + 1  -- English books count as one unit
  let unit_arrangements := Nat.factorial total_units
  let english_arrangements := Nat.factorial english_books
  unit_arrangements * english_arrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem workbook_arrangement_count : arrange_workbooks = 12 := by
  sorry

end NUMINAMATH_CALUDE_workbook_arrangement_count_l3906_390680


namespace NUMINAMATH_CALUDE_symmetric_points_l3906_390669

/-- Given a line ax + by + c = 0 and two points (x₁, y₁) and (x₂, y₂), 
    this function returns true if the points are symmetric with respect to the line -/
def are_symmetric (a b c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The midpoint of the two points lies on the line
  a * ((x₁ + x₂) / 2) + b * ((y₁ + y₂) / 2) + c = 0 ∧
  -- The line connecting the two points is perpendicular to the given line
  (y₂ - y₁) * a = -(x₂ - x₁) * b

/-- Theorem stating that (-5, -4) is symmetric to (3, 4) with respect to the line x + y + 1 = 0 -/
theorem symmetric_points : are_symmetric 1 1 1 3 4 (-5) (-4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l3906_390669


namespace NUMINAMATH_CALUDE_max_probability_dice_difference_l3906_390612

def roll_dice : Finset (ℕ × ℕ) := Finset.product (Finset.range 6) (Finset.range 6)

def difference (roll : ℕ × ℕ) : ℤ := (roll.1 : ℤ) - (roll.2 : ℤ)

def target_differences : Finset ℤ := {-2, -1, 0, 1, 2}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  roll_dice.filter (λ roll => difference roll ∈ target_differences)

theorem max_probability_dice_difference :
  (favorable_outcomes.card : ℚ) / roll_dice.card = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_max_probability_dice_difference_l3906_390612


namespace NUMINAMATH_CALUDE_passing_percentage_l3906_390660

def total_marks : ℕ := 400
def student_marks : ℕ := 100
def failing_margin : ℕ := 40

theorem passing_percentage :
  (student_marks + failing_margin) * 100 / total_marks = 35 := by
sorry

end NUMINAMATH_CALUDE_passing_percentage_l3906_390660


namespace NUMINAMATH_CALUDE_valid_sequence_count_is_840_l3906_390695

/-- Represents a coin toss sequence --/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a given subsequence in a coin sequence --/
def countSubsequence (seq : CoinSequence) (subseq : CoinSequence) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions --/
def satisfiesConditions (seq : CoinSequence) : Prop :=
  seq.length = 16 ∧
  countSubsequence seq [true, true] = 2 ∧
  countSubsequence seq [false, false] = 6 ∧
  countSubsequence seq [true, false] = 4 ∧
  countSubsequence seq [false, true] = 4

/-- The number of valid coin sequences --/
def validSequenceCount : Nat :=
  sorry

theorem valid_sequence_count_is_840 :
  validSequenceCount = 840 :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_count_is_840_l3906_390695
