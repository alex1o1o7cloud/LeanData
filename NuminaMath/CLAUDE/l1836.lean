import Mathlib

namespace NUMINAMATH_CALUDE_church_attendance_female_adults_l1836_183656

theorem church_attendance_female_adults
  (total : ℕ) (children : ℕ) (male_adults : ℕ)
  (h1 : total = 200)
  (h2 : children = 80)
  (h3 : male_adults = 60) :
  total - children - male_adults = 60 :=
by sorry

end NUMINAMATH_CALUDE_church_attendance_female_adults_l1836_183656


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1836_183617

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  total_runs : Nat
  average : Rat

/-- Calculates the increase in average for a batsman -/
def average_increase (b : Batsman) (new_runs : Nat) (new_average : Rat) : Rat :=
  new_average - b.average

/-- Theorem: The increase in the batsman's average is 3 -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
  b.innings = 16 →
  average_increase b 56 8 = 3 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_increase_l1836_183617


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1836_183606

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 3*a - 2010 = 0) → 
  (b^2 + 3*b - 2010 = 0) → 
  (a^2 - a - 4*b = 2022) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1836_183606


namespace NUMINAMATH_CALUDE_mrs_hilt_remaining_cents_l1836_183632

/-- Given that Mrs. Hilt had 15 cents initially and spent 11 cents on a pencil, 
    prove that she was left with 4 cents. -/
theorem mrs_hilt_remaining_cents 
  (initial_cents : ℕ) 
  (pencil_cost : ℕ) 
  (h1 : initial_cents = 15)
  (h2 : pencil_cost = 11) :
  initial_cents - pencil_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_remaining_cents_l1836_183632


namespace NUMINAMATH_CALUDE_ratio_A_to_B_in_X_l1836_183674

/-- Represents a compound with two elements -/
structure Compound where
  totalWeight : ℝ
  weightB : ℝ

/-- Calculates the ratio of element A to element B in a compound -/
def ratioAtoB (c : Compound) : ℝ × ℝ :=
  let weightA := c.totalWeight - c.weightB
  (weightA, c.weightB)

/-- Theorem: The ratio of A to B in compound X is 1:5 -/
theorem ratio_A_to_B_in_X :
  let compoundX : Compound := { totalWeight := 300, weightB := 250 }
  let (a, b) := ratioAtoB compoundX
  a / b = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_A_to_B_in_X_l1836_183674


namespace NUMINAMATH_CALUDE_two_digit_reverse_divisible_by_11_l1836_183687

theorem two_digit_reverse_divisible_by_11 (a b : ℕ) 
  (ha : a ≤ 9) (hb : b ≤ 9) : 
  (1000 * a + 100 * b + 10 * b + a) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_divisible_by_11_l1836_183687


namespace NUMINAMATH_CALUDE_borrowed_amount_l1836_183652

theorem borrowed_amount (P : ℝ) 
  (h1 : (P * 12 / 100 * 3) + (P * 9 / 100 * 5) + (P * 13 / 100 * 3) = 8160) : 
  P = 6800 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l1836_183652


namespace NUMINAMATH_CALUDE_tangency_point_difference_l1836_183671

/-- A quadrilateral inscribed in a circle with an inscribed circle --/
structure InscribedQuadrilateral where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Conditions
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  cyclic : True  -- Represents that the quadrilateral is cyclic
  inscribed : True  -- Represents that there's an inscribed circle

/-- The specific quadrilateral from the problem --/
def specificQuad : InscribedQuadrilateral where
  a := 60
  b := 110
  c := 140
  d := 90
  positive := by simp
  cyclic := True.intro
  inscribed := True.intro

/-- The point of tangency divides the side of length 140 into m and n --/
def tangencyPoint (q : InscribedQuadrilateral) : ℝ × ℝ :=
  sorry

/-- The theorem to prove --/
theorem tangency_point_difference (q : InscribedQuadrilateral) 
  (h : q = specificQuad) : 
  let (m, n) := tangencyPoint q
  |m - n| = 120 := by
  sorry

end NUMINAMATH_CALUDE_tangency_point_difference_l1836_183671


namespace NUMINAMATH_CALUDE_range_of_a_l1836_183624

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a ≤ 0

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, p a ↔ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1836_183624


namespace NUMINAMATH_CALUDE_sequence_problem_l1836_183660

def arithmetic_sequence (a b c d : ℚ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d e : ℚ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_problem (a₁ a₂ b₁ b₂ b₃ : ℚ) 
  (h1 : arithmetic_sequence (-1) a₁ a₂ (-4))
  (h2 : geometric_sequence (-1) b₁ b₂ b₃ (-4)) :
  (a₂ - a₁) / b₂ = 1/2 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l1836_183660


namespace NUMINAMATH_CALUDE_ruth_math_class_hours_l1836_183649

/-- Represents Ruth's weekly school schedule and math class time --/
structure RuthSchedule where
  hours_per_day : ℝ
  days_per_week : ℝ
  math_class_percentage : ℝ

/-- Calculates the number of hours Ruth spends in math class per week --/
def math_class_hours (schedule : RuthSchedule) : ℝ :=
  schedule.hours_per_day * schedule.days_per_week * schedule.math_class_percentage

/-- Theorem stating that Ruth spends 10 hours per week in math class --/
theorem ruth_math_class_hours :
  let schedule := RuthSchedule.mk 8 5 0.25
  math_class_hours schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_ruth_math_class_hours_l1836_183649


namespace NUMINAMATH_CALUDE_two_number_problem_l1836_183600

theorem two_number_problem : ∃ x y : ℕ, 
  x ≠ y ∧ 
  x ≥ 10 ∧ 
  y ≥ 10 ∧ 
  (x + y) + (max x y - min x y) + (x * y) + (max x y / min x y) = 576 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l1836_183600


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1836_183664

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  n * (n - 1) = 90 → 2 * (n * (n - 1)) = 180 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l1836_183664


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l1836_183667

theorem mark_and_carolyn_money_sum : 
  (5 : ℚ) / 6 + (2 : ℚ) / 5 = (37 : ℚ) / 30 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l1836_183667


namespace NUMINAMATH_CALUDE_spinner_prime_probability_l1836_183678

def spinner : List Nat := [2, 7, 9, 11, 15, 17]

def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

def countPrimes (l : List Nat) : Nat :=
  (l.filter isPrime).length

theorem spinner_prime_probability :
  (countPrimes spinner : Rat) / (spinner.length : Rat) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_spinner_prime_probability_l1836_183678


namespace NUMINAMATH_CALUDE_sum_factorials_mod_15_l1836_183672

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_factorials_mod_15 : sum_factorials 50 % 15 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_15_l1836_183672


namespace NUMINAMATH_CALUDE_game_tie_fraction_l1836_183602

theorem game_tie_fraction (max_win_rate sara_win_rate postponed_rate : ℚ)
  (h_max : max_win_rate = 2/5)
  (h_sara : sara_win_rate = 1/4)
  (h_postponed : postponed_rate = 5/100) :
  let total_win_rate := max_win_rate + sara_win_rate
  let non_postponed_rate := 1 - postponed_rate
  let win_rate_of_non_postponed := total_win_rate / non_postponed_rate
  1 - win_rate_of_non_postponed = 6/19 := by
sorry

end NUMINAMATH_CALUDE_game_tie_fraction_l1836_183602


namespace NUMINAMATH_CALUDE_parabola_tangent_lines_l1836_183622

/-- A parabola defined by y^2 = 8x -/
def parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- A line that has exactly one common point with the parabola and passes through P -/
def tangent_line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - P.2 = m * (p.1 - P.1)}

/-- The number of lines passing through P and having exactly one common point with the parabola -/
def num_tangent_lines : ℕ := 2

theorem parabola_tangent_lines :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
  (∀ m : ℝ, (tangent_line m ∩ parabola).Nonempty → m = m₁ ∨ m = m₂) ∧
  (tangent_line m₁ ∩ parabola).Nonempty ∧
  (tangent_line m₂ ∩ parabola).Nonempty :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_lines_l1836_183622


namespace NUMINAMATH_CALUDE_car_sale_profit_percentage_l1836_183692

/-- Calculate the net profit percentage for a car sale --/
theorem car_sale_profit_percentage 
  (purchase_price : ℝ) 
  (repair_cost_percentage : ℝ) 
  (sales_tax_percentage : ℝ) 
  (registration_fee_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 42000)
  (h2 : repair_cost_percentage = 0.35)
  (h3 : sales_tax_percentage = 0.08)
  (h4 : registration_fee_percentage = 0.06)
  (h5 : selling_price = 64900) :
  let total_cost := purchase_price * (1 + repair_cost_percentage + sales_tax_percentage + registration_fee_percentage)
  let net_profit := selling_price - total_cost
  let net_profit_percentage := (net_profit / total_cost) * 100
  ∃ ε > 0, |net_profit_percentage - 3.71| < ε :=
by sorry

end NUMINAMATH_CALUDE_car_sale_profit_percentage_l1836_183692


namespace NUMINAMATH_CALUDE_cycling_distance_is_four_point_five_l1836_183638

/-- Represents the cycling scenario with given conditions -/
structure CyclingScenario where
  speed : ℝ  -- Original speed in miles per hour
  time : ℝ   -- Original time taken in hours
  distance : ℝ -- Distance cycled in miles

/-- The conditions of the cycling problem -/
def cycling_conditions (scenario : CyclingScenario) : Prop :=
  -- Distance is speed multiplied by time
  scenario.distance = scenario.speed * scenario.time ∧
  -- Faster speed condition
  scenario.distance = (scenario.speed + 1/4) * (3/4 * scenario.time) ∧
  -- Slower speed condition
  scenario.distance = (scenario.speed - 1/4) * (scenario.time + 3)

/-- The theorem to be proved -/
theorem cycling_distance_is_four_point_five :
  ∀ (scenario : CyclingScenario), cycling_conditions scenario → scenario.distance = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_cycling_distance_is_four_point_five_l1836_183638


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1836_183653

theorem sum_of_fractions (a b c : ℝ) (h : a * b * c = 1) :
  a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1836_183653


namespace NUMINAMATH_CALUDE_coefficient_x3y5_proof_l1836_183634

/-- The coefficient of x^3y^5 in the expansion of (x+y)(x-y)^7 -/
def coefficient_x3y5 : ℤ := 14

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_x3y5_proof :
  coefficient_x3y5 = (binomial 7 4 : ℤ) - (binomial 7 5 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_proof_l1836_183634


namespace NUMINAMATH_CALUDE_saving_time_proof_l1836_183621

def down_payment : ℝ := 108000
def monthly_savings : ℝ := 3000
def months_in_year : ℝ := 12

theorem saving_time_proof : 
  (down_payment / monthly_savings) / months_in_year = 3 := by
sorry

end NUMINAMATH_CALUDE_saving_time_proof_l1836_183621


namespace NUMINAMATH_CALUDE_total_profit_is_54000_l1836_183655

/-- Calculates the total profit given the investments and Jose's profit share -/
def calculate_total_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_ratio : ℕ := tom_investment * tom_months
  let jose_ratio : ℕ := jose_investment * jose_months
  let total_ratio : ℕ := tom_ratio + jose_ratio
  (jose_profit * total_ratio) / jose_ratio

/-- The total profit for Tom and Jose's business venture -/
theorem total_profit_is_54000 :
  calculate_total_profit 30000 12 45000 10 30000 = 54000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_54000_l1836_183655


namespace NUMINAMATH_CALUDE_proposition_a_proposition_d_l1836_183625

-- Proposition A
theorem proposition_a (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) :
  -4 < a - b ∧ a - b < 2 := by sorry

-- Proposition D
theorem proposition_d : ∃ a : ℝ, a + 1 / a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_proposition_a_proposition_d_l1836_183625


namespace NUMINAMATH_CALUDE_product_equality_l1836_183608

theorem product_equality : 50 * 29.96 * 2.996 * 500 = 2244004 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1836_183608


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l1836_183691

/-- 
Given a quadratic equation kx^2 - 2x - 1 = 0, this theorem states that
for the equation to have two distinct real roots, k must be greater than -1
and not equal to 0.
-/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l1836_183691


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1836_183685

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ  -- Average before the 17th inning
  runsScored : ℕ      -- Runs scored in the 17th inning
  averageIncrease : ℝ -- Increase in average after the 17th inning

/-- Calculates the new average after the 17th inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem: The batsman's average after the 17th inning is 140 runs -/
theorem batsman_average_after_17th_inning (b : Batsman)
  (h1 : b.runsScored = 300)
  (h2 : b.averageIncrease = 10)
  : newAverage b = 140 := by
  sorry

#check batsman_average_after_17th_inning

end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1836_183685


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1836_183658

/-- The set of available digits --/
def available_digits : Finset Nat := {0, 2, 4, 6}

/-- A function to check if a number is a valid three-digit number formed from the available digits --/
def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : Nat), a ∈ available_digits ∧ b ∈ available_digits ∧ c ∈ available_digits ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = 100 * a + 10 * b + c

/-- The largest valid number --/
def largest_number : Nat := 642

/-- The smallest valid number --/
def smallest_number : Nat := 204

/-- Theorem: The sum of the largest and smallest valid numbers is 846 --/
theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : Nat, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : Nat, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 846 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1836_183658


namespace NUMINAMATH_CALUDE_clock_angle_at_3_45_l1836_183669

/-- The smaller angle between the hour hand and minute hand on a 12-hour analog clock at 3:45 --/
theorem clock_angle_at_3_45 :
  let full_rotation : ℝ := 360
  let hour_marks : ℕ := 12
  let degrees_per_hour : ℝ := full_rotation / hour_marks
  let minute_hand_angle : ℝ := 270
  let hour_hand_angle : ℝ := 3 * degrees_per_hour + 3/4 * degrees_per_hour
  let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
  min angle_diff (full_rotation - angle_diff) = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_45_l1836_183669


namespace NUMINAMATH_CALUDE_min_value_of_g_l1836_183643

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Define the property that f is decreasing on (-∞, 1]
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 1 → f a x ≥ f a y

-- Define the function g
def g (a : ℝ) : ℝ := f a (a + 1) - f a 1

-- State the theorem
theorem min_value_of_g :
  ∀ a : ℝ, is_decreasing_on_interval a → g a ≥ 1 ∧ ∃ a₀, g a₀ = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l1836_183643


namespace NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_equality_condition_l1836_183651

theorem min_value_sqrt_and_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 4 / x ≥ 4 * Real.sqrt 2 :=
sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  ∃ x > 0, 3 * Real.sqrt x + 4 / x = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_equality_condition_l1836_183651


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1836_183601

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, (21 ^ k ∣ n) → 7 ^ k - k ^ 7 = 1) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, (21 ^ k ∣ m) → 7 ^ k - k ^ 7 = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1836_183601


namespace NUMINAMATH_CALUDE_solve_equation_l1836_183694

theorem solve_equation (x y : ℤ) 
  (h1 : x^2 - 3*x + 6 = y + 2) 
  (h2 : x = -8) : 
  y = 92 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1836_183694


namespace NUMINAMATH_CALUDE_paper_tearing_l1836_183635

theorem paper_tearing (n : ℕ) : 
  (∃ k : ℕ, 1 + 2 * k = 503) ∧ 
  (¬ ∃ k : ℕ, 1 + 2 * k = 2020) := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_l1836_183635


namespace NUMINAMATH_CALUDE_knight_freedom_guaranteed_l1836_183696

/-- Represents a pile of coins -/
structure Pile :=
  (total : ℕ)
  (magical : ℕ)

/-- Represents the state of the coins -/
structure CoinState :=
  (pile1 : Pile)
  (pile2 : Pile)

/-- Checks if the piles have equal magical or ordinary coins -/
def isEqualDistribution (state : CoinState) : Prop :=
  state.pile1.magical = state.pile2.magical ∨ 
  (state.pile1.total - state.pile1.magical) = (state.pile2.total - state.pile2.magical)

/-- Represents a division strategy -/
def DivisionStrategy := ℕ → CoinState

/-- The theorem to be proved -/
theorem knight_freedom_guaranteed :
  ∃ (strategy : DivisionStrategy),
    (∀ (n : ℕ), n ≤ 25 → 
      (strategy n).pile1.total + (strategy n).pile2.total = 100 ∧
      (strategy n).pile1.magical + (strategy n).pile2.magical = 50) →
    ∃ (day : ℕ), day ≤ 25 ∧ isEqualDistribution (strategy day) :=
sorry

end NUMINAMATH_CALUDE_knight_freedom_guaranteed_l1836_183696


namespace NUMINAMATH_CALUDE_tunneled_cube_surface_area_l1836_183673

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure TunneledCube where
  sideLength : ℝ
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculates the surface area of a tunneled cube -/
def surfaceArea (cube : TunneledCube) : ℝ := sorry

/-- Checks if a number is square-free (not divisible by the square of any prime) -/
def isSquareFree (n : ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem tunneled_cube_surface_area :
  ∃ (cube : TunneledCube) (u v w : ℕ),
    cube.sideLength = 10 ∧
    surfaceArea cube = u + v * Real.sqrt w ∧
    isSquareFree w ∧
    u + v + w = 472 := by sorry

end NUMINAMATH_CALUDE_tunneled_cube_surface_area_l1836_183673


namespace NUMINAMATH_CALUDE_oven_temperature_l1836_183666

/-- Represents the temperature of the steak at time t -/
noncomputable def T (t : ℝ) : ℝ := sorry

/-- The constant oven temperature -/
def T_o : ℝ := sorry

/-- The initial temperature of the steak -/
def T_i : ℝ := 5

/-- The constant of proportionality in Newton's Law of Cooling -/
noncomputable def k : ℝ := sorry

/-- Newton's Law of Cooling: The rate of change of the steak's temperature
    is proportional to the difference between the steak's temperature and the oven temperature -/
axiom newtons_law_cooling : ∀ t, (deriv T t) = k * (T_o - T t)

/-- The solution to Newton's Law of Cooling -/
axiom cooling_solution : ∀ t, T t = T_o + (T_i - T_o) * Real.exp (-k * t)

/-- The temperature after 15 minutes is 45°C -/
axiom temp_at_15 : T 15 = 45

/-- The temperature after 30 minutes is 77°C -/
axiom temp_at_30 : T 30 = 77

/-- The theorem stating that the oven temperature is 205°C -/
theorem oven_temperature : T_o = 205 := by sorry

end NUMINAMATH_CALUDE_oven_temperature_l1836_183666


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l1836_183613

theorem sphere_volume_surface_area_relation (r₁ r₂ : ℝ) (h : r₁ > 0) :
  (4 / 3 * Real.pi * r₂^3) = 8 * (4 / 3 * Real.pi * r₁^3) →
  (4 * Real.pi * r₂^2) = 4 * (4 * Real.pi * r₁^2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l1836_183613


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1836_183648

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 3)
  f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1836_183648


namespace NUMINAMATH_CALUDE_max_xy_on_circle_l1836_183681

theorem max_xy_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 4 → a * b ≤ max) ∧ (∃ c d : ℝ, c^2 + d^2 = 4 ∧ c * d = max) ∧ max = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_on_circle_l1836_183681


namespace NUMINAMATH_CALUDE_S_formula_l1836_183636

/-- g(k) is the largest odd factor of k -/
def g (k : ℕ+) : ℕ+ :=
  sorry

/-- Sn is the sum of g(k) for k from 1 to 2^n -/
def S (n : ℕ) : ℚ :=
  sorry

/-- The main theorem: Sn = (1/3)(4^n + 2) for all natural numbers n -/
theorem S_formula (n : ℕ) : S n = (1/3) * (4^n + 2) :=
  sorry

end NUMINAMATH_CALUDE_S_formula_l1836_183636


namespace NUMINAMATH_CALUDE_intersection_point_unique_l1836_183680

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (54/5, -26/5)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3*y = -2*x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 7*y = -3*x - 4

theorem intersection_point_unique :
  (∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l1836_183680


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1836_183644

/-- Given a point P with coordinates (3a-6, 1-a) that lies on the x-axis, 
    prove that its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 3*a - 6 ∧ P.2 = 1 - a ∧ P.2 = 0) → 
  (∃ P : ℝ × ℝ, P = (-3, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1836_183644


namespace NUMINAMATH_CALUDE_inverse_36_mod_53_l1836_183604

theorem inverse_36_mod_53 (h : (17⁻¹ : ZMod 53) = 26) : (36⁻¹ : ZMod 53) = 27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_36_mod_53_l1836_183604


namespace NUMINAMATH_CALUDE_max_consecutive_irreducible_five_digit_l1836_183661

/-- A number is irreducible if it cannot be expressed as a product of two three-digit numbers -/
def irreducible (n : ℕ) : Prop :=
  ∀ a b : ℕ, 100 ≤ a ∧ a ≤ 999 ∧ 100 ≤ b ∧ b ≤ 999 → n ≠ a * b

/-- The set of five-digit numbers -/
def five_digit_numbers : Set ℕ := {n | 10000 ≤ n ∧ n ≤ 99999}

/-- A function that returns the length of the longest sequence of consecutive irreducible numbers in a set -/
def max_consecutive_irreducible (s : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating that the maximum number of consecutive irreducible five-digit numbers is 99 -/
theorem max_consecutive_irreducible_five_digit :
  max_consecutive_irreducible five_digit_numbers = 99 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_irreducible_five_digit_l1836_183661


namespace NUMINAMATH_CALUDE_shifted_line_equation_l1836_183642

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift * l.slope }

/-- The original line y = x -/
def original_line : Line := { slope := 1, intercept := 0 }

theorem shifted_line_equation :
  let shifted := shift_line original_line (-1)
  shifted.slope = 1 ∧ shifted.intercept = 1 :=
sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l1836_183642


namespace NUMINAMATH_CALUDE_simplify_and_compare_l1836_183676

theorem simplify_and_compare : 
  1.82 * (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_compare_l1836_183676


namespace NUMINAMATH_CALUDE_smallest_solution_l1836_183603

-- Define the equation
def equation (t : ℝ) : Prop :=
  (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2

-- Define the set of all t that satisfy the equation
def solution_set : Set ℝ := {t | equation t}

-- Theorem statement
theorem smallest_solution :
  ∃ (t_min : ℝ), t_min ∈ solution_set ∧ t_min = 3/4 ∧ ∀ (t : ℝ), t ∈ solution_set → t_min ≤ t :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l1836_183603


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1836_183611

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1836_183611


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l1836_183605

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of the line segment with endpoints (5, π/4) and (5, 3π/4) in polar coordinates is (5√2/2, π/2) -/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 5 (π/4) 5 (3*π/4)
  r = 5 * Real.sqrt 2 / 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
by sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l1836_183605


namespace NUMINAMATH_CALUDE_james_vegetable_consumption_l1836_183683

/-- Calculates the final weekly vegetable consumption based on initial daily consumption and changes --/
def final_weekly_consumption (initial_daily : ℚ) (kale_addition : ℚ) : ℚ :=
  (initial_daily * 2 * 7) + kale_addition

/-- Proves that James' final weekly vegetable consumption is 10 pounds --/
theorem james_vegetable_consumption :
  let initial_daily := (1/4 : ℚ) + (1/4 : ℚ)
  let kale_addition := (3 : ℚ)
  final_weekly_consumption initial_daily kale_addition = 10 := by
  sorry

#eval final_weekly_consumption ((1/4 : ℚ) + (1/4 : ℚ)) 3

end NUMINAMATH_CALUDE_james_vegetable_consumption_l1836_183683


namespace NUMINAMATH_CALUDE_tau_fraction_values_l1836_183618

/-- The number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The number of positive divisors of n which have remainders 1 when divided by 3 -/
def τ₁ (n : ℕ+) : ℕ := sorry

/-- A number is composite if it's greater than 1 and not prime -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

/-- The set of possible values for τ(10n) / τ₁(10n) -/
def possibleValues : Set ℕ := {n | n % 2 = 0 ∨ isComposite n}

/-- The main theorem -/
theorem tau_fraction_values (n : ℕ+) : 
  ∃ (k : ℕ), k ∈ possibleValues ∧ (τ (10 * n) : ℚ) / τ₁ (10 * n) = k := by sorry

end NUMINAMATH_CALUDE_tau_fraction_values_l1836_183618


namespace NUMINAMATH_CALUDE_locus_of_circle_center_l1836_183662

/-
  Define the points M and N, and the circle passing through them with center P.
  Then prove that the locus of vertex P satisfies the given equation.
-/

-- Define the points M and N
def M : ℝ × ℝ := (0, -5)
def N : ℝ × ℝ := (0, 5)

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  passes_through_M : (center.1 - M.1)^2 + (center.2 - M.2)^2 = (center.1 - N.1)^2 + (center.2 - N.2)^2

-- Define the locus equation
def locus_equation (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ (P.2^2 / 169 + P.1^2 / 144 = 1)

-- Theorem statement
theorem locus_of_circle_center (c : Circle) : locus_equation c.center :=
  sorry


end NUMINAMATH_CALUDE_locus_of_circle_center_l1836_183662


namespace NUMINAMATH_CALUDE_total_albums_l1836_183663

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina

/-- The theorem to prove -/
theorem total_albums (a : Albums) (h : album_conditions a) : 
  a.adele + a.bridget + a.katrina + a.miriam = 585 := by
  sorry

#check total_albums

end NUMINAMATH_CALUDE_total_albums_l1836_183663


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1836_183609

-- Define the quadratic function
def f (k x : ℝ) : ℝ := 2*k*x^2 - 2*x - 3*k - 2

-- Define the property of having two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0

-- Define the property of roots being on opposite sides of 1
def roots_around_one (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ f k x₁ = 0 ∧ f k x₂ = 0

-- Theorem statement
theorem quadratic_roots_range (k : ℝ) :
  has_two_real_roots k ∧ roots_around_one k ↔ k < -4 ∨ k > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1836_183609


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l1836_183641

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 4, 5, 6; 7, 8, 9]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 1; 1, 1, 0; 0, 1, 1]
  A * B = !![3, 5, 4; 9, 11, 10; 15, 17, 16] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l1836_183641


namespace NUMINAMATH_CALUDE_joels_dads_age_l1836_183629

theorem joels_dads_age :
  ∀ (joel_current_age joel_future_age dads_current_age : ℕ),
    joel_current_age = 5 →
    joel_future_age = 27 →
    dads_current_age + (joel_future_age - joel_current_age) = 2 * joel_future_age →
    dads_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_joels_dads_age_l1836_183629


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l1836_183679

/-- Given a sinusoidal function y = A*sin(ω*x + φ) with specific properties,
    prove its exact form and characteristics. -/
theorem sinusoidal_function_properties
  (A ω φ : ℝ)
  (h_A_pos : A > 0)
  (h_ω_pos : ω > 0)
  (h_passes_through : A * Real.sin (ω * (π / 12) + φ) = 0)
  (h_highest_point : A * Real.sin (ω * (π / 3) + φ) = 5) :
  ∃ (f : ℝ → ℝ),
    (∀ x, f x = 5 * Real.sin (2 * x - π / 6)) ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 3) (k * π + 5 * π / 6))) ∧
    (∀ x ∈ Set.Icc 0 π, f x ≤ 5 ∧ f x ≥ -5) ∧
    (f (π / 3) = 5 ∧ f (5 * π / 6) = -5) ∧
    (∀ k : ℤ, ∀ x, (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12) → f x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l1836_183679


namespace NUMINAMATH_CALUDE_charlie_has_31_pennies_l1836_183684

/-- The number of pennies Charlie has -/
def charlie_pennies : ℕ := 31

/-- The number of pennies Alex has -/
def alex_pennies : ℕ := 9

/-- Condition 1: If Alex gives Charlie a penny, Charlie will have four times as many pennies as Alex has -/
axiom condition1 : charlie_pennies + 1 = 4 * (alex_pennies - 1)

/-- Condition 2: If Charlie gives Alex a penny, Charlie will have three times as many pennies as Alex has -/
axiom condition2 : charlie_pennies - 1 = 3 * (alex_pennies + 1)

theorem charlie_has_31_pennies : charlie_pennies = 31 := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_31_pennies_l1836_183684


namespace NUMINAMATH_CALUDE_exists_projective_map_three_points_l1836_183689

-- Define the necessary structures
structure ProjectivePlane where
  Point : Type
  Line : Type
  incidence : Point → Line → Prop

-- Define a projective map
def ProjectiveMap (π : ProjectivePlane) := π.Point → π.Point

-- State the theorem
theorem exists_projective_map_three_points 
  (π : ProjectivePlane) 
  (l₀ l : π.Line) 
  (A₀ B₀ C₀ A B C : π.Point)
  (on_l₀ : π.incidence A₀ l₀ ∧ π.incidence B₀ l₀ ∧ π.incidence C₀ l₀)
  (on_l : π.incidence A l ∧ π.incidence B l ∧ π.incidence C l) :
  ∃ (f : ProjectiveMap π), 
    f A₀ = A ∧ f B₀ = B ∧ f C₀ = C := by
  sorry

end NUMINAMATH_CALUDE_exists_projective_map_three_points_l1836_183689


namespace NUMINAMATH_CALUDE_clock_problem_l1836_183645

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Represents duration in hours, minutes, and seconds -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Converts total seconds to Time structure -/
def secondsToTime (totalSeconds : Nat) : Time :=
  let hours := totalSeconds / 3600
  let minutes := (totalSeconds % 3600) / 60
  let seconds := totalSeconds % 60
  { hours := hours % 12, minutes := minutes, seconds := seconds }

/-- Adds a Duration to a Time, wrapping around 12-hour clock -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalSeconds := 
    (t.hours * 3600 + t.minutes * 60 + t.seconds) +
    (d.hours * 3600 + d.minutes * 60 + d.seconds)
  secondsToTime totalSeconds

/-- Calculates the sum of digits in a Time -/
def sumDigits (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem clock_problem (initialTime : Time) (elapsedTime : Duration) : 
  initialTime.hours = 3 ∧ 
  initialTime.minutes = 15 ∧ 
  initialTime.seconds = 20 ∧
  elapsedTime.hours = 305 ∧ 
  elapsedTime.minutes = 45 ∧ 
  elapsedTime.seconds = 56 →
  let finalTime := addDuration initialTime elapsedTime
  finalTime.hours = 9 ∧ 
  finalTime.minutes = 1 ∧ 
  finalTime.seconds = 16 ∧
  sumDigits finalTime = 26 := by
  sorry

end NUMINAMATH_CALUDE_clock_problem_l1836_183645


namespace NUMINAMATH_CALUDE_triangles_equality_l1836_183646

-- Define the points
variable (A K L M N G G' : ℝ × ℝ)

-- Define the angle α
variable (α : ℝ)

-- Define similarity of triangles
def similar_triangles (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define isosceles triangle
def isosceles_triangle (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define angle at vertex
def angle_at_vertex (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (v : ℝ × ℝ) (θ : ℝ) : Prop := sorry

-- State the theorem
theorem triangles_equality (h1 : similar_triangles (A, K, L) (A, M, N))
                           (h2 : isosceles_triangle (A, K, L))
                           (h3 : isosceles_triangle (A, M, N))
                           (h4 : angle_at_vertex (A, K, L) A α)
                           (h5 : angle_at_vertex (A, M, N) A α)
                           (h6 : similar_triangles (G, N, K) (G', L, M))
                           (h7 : isosceles_triangle (G, N, K))
                           (h8 : isosceles_triangle (G', L, M))
                           (h9 : angle_at_vertex (G, N, K) G (π - α))
                           (h10 : angle_at_vertex (G', L, M) G' (π - α)) :
  G = G' := by sorry

end NUMINAMATH_CALUDE_triangles_equality_l1836_183646


namespace NUMINAMATH_CALUDE_no_negative_exponents_l1836_183630

theorem no_negative_exponents (a b c d : ℤ) (h : (5 : ℝ)^a + (5 : ℝ)^b = (2 : ℝ)^c + (2 : ℝ)^d + 17) :
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d :=
by sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l1836_183630


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l1836_183640

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_a : a = 4) 
  (hypotenuse : c = 5) : 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l1836_183640


namespace NUMINAMATH_CALUDE_percentage_increase_l1836_183607

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 80 → final = 120 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1836_183607


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_is_145_greatest_l1836_183623

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 150 ∧ Nat.gcd n 30 = 5 → n ≤ 145 := by
  sorry

theorem gcf_of_145_and_30 : Nat.gcd 145 30 = 5 := by
  sorry

theorem is_145_greatest : ∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ 145 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_is_145_greatest_l1836_183623


namespace NUMINAMATH_CALUDE_train_crossing_time_l1836_183639

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 160 ∧ 
  train_speed_kmh = 72 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 8 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1836_183639


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1836_183628

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + a^n) + 1 / (1 + b^n)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1836_183628


namespace NUMINAMATH_CALUDE_third_segment_length_l1836_183654

/-- Represents the lengths of interview segments in a radio show. -/
structure InterviewSegments where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Checks if the given segment lengths satisfy the radio show conditions. -/
def validSegments (s : InterviewSegments) : Prop :=
  s.first = 2 * (s.second + s.third) ∧
  s.third = s.second / 2 ∧
  s.first + s.second + s.third = 90

theorem third_segment_length :
  ∀ s : InterviewSegments, validSegments s → s.third = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_segment_length_l1836_183654


namespace NUMINAMATH_CALUDE_travel_time_for_a_l1836_183699

/-- Represents the travel time of a person given their relative speed and time difference from a reference traveler -/
def travelTime (relativeSpeed : ℚ) (timeDiff : ℚ) : ℚ :=
  (4 : ℚ) / 3 * ((3 : ℚ) / 2 + timeDiff)

theorem travel_time_for_a (speedRatio : ℚ) (timeDiffHours : ℚ) 
    (h1 : speedRatio = 3 / 4) 
    (h2 : timeDiffHours = 1 / 2) : 
  travelTime speedRatio timeDiffHours = 2 := by
  sorry

#eval travelTime (3/4) (1/2)

end NUMINAMATH_CALUDE_travel_time_for_a_l1836_183699


namespace NUMINAMATH_CALUDE_pizzas_per_person_is_30_l1836_183682

/-- The number of croissants each person eats -/
def croissants_per_person : ℕ := 7

/-- The number of cakes each person eats -/
def cakes_per_person : ℕ := 18

/-- The total number of items consumed by both people -/
def total_items : ℕ := 110

/-- The number of people -/
def num_people : ℕ := 2

theorem pizzas_per_person_is_30 :
  ∃ (pizzas_per_person : ℕ),
    pizzas_per_person = 30 ∧
    num_people * (croissants_per_person + cakes_per_person + pizzas_per_person) = total_items :=
by sorry

end NUMINAMATH_CALUDE_pizzas_per_person_is_30_l1836_183682


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1836_183620

/-- The volume of a cylinder formed by rotating a rectangle about its shorter side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (h_length : length = 30) (h_width : width = 16) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  volume = 1920 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1836_183620


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_sequence_l1836_183631

/-- Given a geometric sequence with first term a, common ratio r, and n-th term defined as a * r^(n-1) -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- The fifth term of a geometric sequence with first term 25 and common ratio -2 is 400 -/
theorem fifth_term_of_specific_geometric_sequence :
  let a := 25
  let r := -2
  geometric_sequence a r 5 = 400 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_sequence_l1836_183631


namespace NUMINAMATH_CALUDE_workday_end_time_l1836_183626

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

def workday_duration : Duration := ⟨8, 0⟩
def start_time : Time := ⟨7, 0, by sorry, by sorry⟩
def lunch_start : Time := ⟨11, 30, by sorry, by sorry⟩
def lunch_duration : Duration := ⟨0, 30⟩

/-- Adds a duration to a time -/
def add_duration (t : Time) (d : Duration) : Time :=
  sorry

/-- Subtracts two times to get a duration -/
def time_difference (t1 t2 : Time) : Duration :=
  sorry

theorem workday_end_time : 
  let time_before_lunch := time_difference lunch_start start_time
  let lunch_end := add_duration lunch_start lunch_duration
  let remaining_work := 
    ⟨workday_duration.hours - time_before_lunch.hours, 
     workday_duration.minutes - time_before_lunch.minutes⟩
  let end_time := add_duration lunch_end remaining_work
  end_time = ⟨15, 30, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_workday_end_time_l1836_183626


namespace NUMINAMATH_CALUDE_water_height_after_transfer_l1836_183657

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of water in a rectangular tank given its dimensions and water height -/
def waterVolume (tank : TankDimensions) (waterHeight : ℝ) : ℝ :=
  tank.length * tank.width * waterHeight

/-- Theorem: The height of water in Tank A after transfer -/
theorem water_height_after_transfer (tankA : TankDimensions) (transferredVolume : ℝ) :
  tankA.length = 3 →
  tankA.width = 2 →
  tankA.height = 4 →
  transferredVolume = 12 →
  (waterVolume tankA (transferredVolume / (tankA.length * tankA.width))) = transferredVolume :=
by sorry

end NUMINAMATH_CALUDE_water_height_after_transfer_l1836_183657


namespace NUMINAMATH_CALUDE_power_15000_mod_1000_l1836_183697

theorem power_15000_mod_1000 (h : 7^500 ≡ 1 [ZMOD 1000]) :
  7^15000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_15000_mod_1000_l1836_183697


namespace NUMINAMATH_CALUDE_quadratic_intersection_at_one_point_l1836_183698

theorem quadratic_intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 5 * x + 3 = -2 * x - 2) ↔ b = 49 / 20 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_at_one_point_l1836_183698


namespace NUMINAMATH_CALUDE_work_completion_time_a_completion_time_l1836_183695

/-- The time it takes for worker b to complete the work alone -/
def b_time : ℝ := 6

/-- The time it takes for worker b to complete the remaining work after both workers work for 1 day -/
def b_remaining_time : ℝ := 2.0000000000000004

/-- The time it takes for worker a to complete the work alone -/
def a_time : ℝ := 2

theorem work_completion_time :
  (1 / a_time + 1 / b_time) + b_remaining_time / b_time = 1 := by sorry

theorem a_completion_time : a_time = 2 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_a_completion_time_l1836_183695


namespace NUMINAMATH_CALUDE_second_race_outcome_l1836_183612

/-- Represents the speeds of Katie and Sarah -/
structure RunnerSpeeds where
  katie : ℝ
  sarah : ℝ

/-- The problem setup -/
def race_problem (speeds : RunnerSpeeds) : Prop :=
  speeds.katie > 0 ∧ 
  speeds.sarah > 0 ∧
  speeds.katie * 95 = speeds.sarah * 100

/-- The theorem to prove -/
theorem second_race_outcome (speeds : RunnerSpeeds) 
  (h : race_problem speeds) : 
  speeds.katie * 105 = speeds.sarah * 99.75 := by
  sorry

#check second_race_outcome

end NUMINAMATH_CALUDE_second_race_outcome_l1836_183612


namespace NUMINAMATH_CALUDE_lg_sum_equals_one_l1836_183677

theorem lg_sum_equals_one (a b : ℝ) 
  (ha : a + Real.log a = 10) 
  (hb : b + (10 : ℝ)^b = 10) : 
  Real.log (a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_sum_equals_one_l1836_183677


namespace NUMINAMATH_CALUDE_exists_good_placement_l1836_183637

def RegularPolygon (n : ℕ) := Fin n → ℕ

def IsGoodPlacement (p : RegularPolygon 1983) : Prop :=
  ∀ (axis : Fin 1983), 
    ∀ (i : Fin 991), 
      p ((axis + i) % 1983) > p ((axis - i) % 1983)

theorem exists_good_placement : 
  ∃ (p : RegularPolygon 1983), IsGoodPlacement p :=
sorry

end NUMINAMATH_CALUDE_exists_good_placement_l1836_183637


namespace NUMINAMATH_CALUDE_john_years_taking_pictures_l1836_183668

/-- Calculates the number of years John has been taking pictures given the following conditions:
  * John takes 10 pictures every day
  * Each memory card can store 50 images
  * Each memory card costs $60
  * John spent $13,140 on memory cards
-/
def years_taking_pictures (
  pictures_per_day : ℕ)
  (images_per_card : ℕ)
  (card_cost : ℕ)
  (total_spent : ℕ)
  : ℕ :=
  let cards_bought := total_spent / card_cost
  let total_images := cards_bought * images_per_card
  let days_taking_pictures := total_images / pictures_per_day
  days_taking_pictures / 365

theorem john_years_taking_pictures :
  years_taking_pictures 10 50 60 13140 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_years_taking_pictures_l1836_183668


namespace NUMINAMATH_CALUDE_friend_contribution_is_eleven_l1836_183650

/-- The amount each friend should contribute when splitting the cost of movie tickets, popcorn, and milk tea. -/
def friend_contribution : ℚ :=
  let num_friends : ℕ := 3
  let ticket_price : ℚ := 7
  let num_tickets : ℕ := 3
  let popcorn_price : ℚ := 3/2  -- $1.5 as a rational number
  let num_popcorn : ℕ := 2
  let milk_tea_price : ℚ := 3
  let num_milk_tea : ℕ := 3
  let total_cost : ℚ := ticket_price * num_tickets + popcorn_price * num_popcorn + milk_tea_price * num_milk_tea
  total_cost / num_friends

theorem friend_contribution_is_eleven :
  friend_contribution = 11 := by
  sorry

end NUMINAMATH_CALUDE_friend_contribution_is_eleven_l1836_183650


namespace NUMINAMATH_CALUDE_f_above_x_axis_iff_valid_a_range_l1836_183627

/-- The function f(x) = (a^2 - 3a + 2)x^2 + (a - 1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 3*a + 2)*x^2 + (a - 1)*x + 2

/-- The graph of f(x) is above the x-axis -/
def above_x_axis (a : ℝ) : Prop := ∀ x, f a x > 0

/-- The range of values for a -/
def valid_a_range (a : ℝ) : Prop := a > 15/7 ∨ a ≤ 1

theorem f_above_x_axis_iff_valid_a_range :
  ∀ a : ℝ, above_x_axis a ↔ valid_a_range a := by sorry

end NUMINAMATH_CALUDE_f_above_x_axis_iff_valid_a_range_l1836_183627


namespace NUMINAMATH_CALUDE_ratio_problem_l1836_183610

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : 
  x / y = 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1836_183610


namespace NUMINAMATH_CALUDE_prime_factorization_of_9600_l1836_183616

theorem prime_factorization_of_9600 : 9600 = 2^6 * 3 * 5^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_of_9600_l1836_183616


namespace NUMINAMATH_CALUDE_conditional_prob_one_jiuzhaigou_l1836_183659

/-- The number of attractions available for choice. -/
def num_attractions : ℕ := 5

/-- The probability that two people choose different attractions. -/
def prob_different_attractions : ℚ := 4 / 5

/-- The probability that exactly one person chooses Jiuzhaigou and they choose different attractions. -/
def prob_one_jiuzhaigou_different : ℚ := 8 / 25

/-- The conditional probability that exactly one person chooses Jiuzhaigou given that they choose different attractions. -/
theorem conditional_prob_one_jiuzhaigou (h : num_attractions = 5) :
  prob_one_jiuzhaigou_different / prob_different_attractions = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_prob_one_jiuzhaigou_l1836_183659


namespace NUMINAMATH_CALUDE_range_of_a_l1836_183614

def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

theorem range_of_a (a : ℝ) : 
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1836_183614


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l1836_183688

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the line segment MN
structure LineSegment :=
  (M N : ℝ × ℝ)

-- Define the parallel relation
def parallel (l1 l2 : LineSegment) : Prop := sorry

-- Define the length of a line segment
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_similarity_theorem (XYZ : Triangle) (MN : LineSegment) :
  parallel MN (LineSegment.mk XYZ.X XYZ.Y) →
  length XYZ.X (MN.M) = 5 →
  length (MN.M) XYZ.Y = 8 →
  length (MN.N) XYZ.Z = 7 →
  length XYZ.X XYZ.Z = 18.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l1836_183688


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l1836_183633

/-- The height of a right circular cylinder inscribed in a hemisphere -/
theorem cylinder_height_in_hemisphere (r_cylinder : ℝ) (r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7) :
  let h := Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  h = 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l1836_183633


namespace NUMINAMATH_CALUDE_odd_power_sum_divisibility_l1836_183665

theorem odd_power_sum_divisibility (k : ℕ) (x y : ℤ) (h_odd : Odd k) (h_pos : k > 0) :
  (x^k + y^k) % (x + y) = 0 → (x^(k+2) + y^(k+2)) % (x + y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisibility_l1836_183665


namespace NUMINAMATH_CALUDE_common_divisors_9240_13860_l1836_183690

/-- The number of positive divisors that two natural numbers have in common -/
def common_divisors_count (a b : ℕ) : ℕ := (Nat.divisors (Nat.gcd a b)).card

/-- Theorem stating that 9240 and 13860 have 48 positive divisors in common -/
theorem common_divisors_9240_13860 :
  common_divisors_count 9240 13860 = 48 := by sorry

end NUMINAMATH_CALUDE_common_divisors_9240_13860_l1836_183690


namespace NUMINAMATH_CALUDE_infinite_linear_combinations_l1836_183670

/-- An infinite sequence of strictly positive integers with a_k < a_{k+1} for all k -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that a_m can be written as x * a_p + y * a_q -/
def CanBeWrittenAs (a : ℕ → ℕ) (m p q x y : ℕ) : Prop :=
  a m = x * a p + y * a q ∧ 0 < x ∧ 0 < y ∧ p ≠ q

theorem infinite_linear_combinations (a : ℕ → ℕ) 
  (h : StrictlyIncreasingSequence a) :
  ∀ n : ℕ, ∃ m p q x y, m > n ∧ CanBeWrittenAs a m p q x y :=
sorry

end NUMINAMATH_CALUDE_infinite_linear_combinations_l1836_183670


namespace NUMINAMATH_CALUDE_roots_equal_opposite_sign_l1836_183615

theorem roots_equal_opposite_sign (a b c d m : ℝ) : 
  (∀ x, (x^2 - 2*b*x + d) / (3*a*x - 4*c) = (m - 2) / (m + 2)) →
  (∃ r : ℝ, (r^2 - 2*b*r + d = 0) ∧ ((-r)^2 - 2*b*(-r) + d = 0)) →
  m = 4*b / (3*a - 2*b) :=
sorry

end NUMINAMATH_CALUDE_roots_equal_opposite_sign_l1836_183615


namespace NUMINAMATH_CALUDE_parallelogram_longer_side_length_l1836_183619

/-- Given a parallelogram with adjacent sides in the ratio 3:2 and perimeter 20,
    prove that the length of the longer side is 6. -/
theorem parallelogram_longer_side_length
  (a b : ℝ)
  (ratio : a / b = 3 / 2)
  (perimeter : 2 * (a + b) = 20)
  : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_longer_side_length_l1836_183619


namespace NUMINAMATH_CALUDE_aquarium_species_count_l1836_183647

theorem aquarium_species_count 
  (sharks : ℕ) (eels : ℕ) (whales : ℕ) (dolphins : ℕ) (rays : ℕ) (octopuses : ℕ)
  (shark_pairs : ℕ) (eel_pairs : ℕ) (whale_pairs : ℕ) (octopus_split : ℕ)
  (h1 : sharks = 48)
  (h2 : eels = 21)
  (h3 : whales = 7)
  (h4 : dolphins = 16)
  (h5 : rays = 9)
  (h6 : octopuses = 30)
  (h7 : shark_pairs = 3)
  (h8 : eel_pairs = 2)
  (h9 : whale_pairs = 1)
  (h10 : octopus_split = 1) :
  sharks + eels + whales + dolphins + rays + octopuses 
  - (shark_pairs + eel_pairs + whale_pairs) 
  + octopus_split = 126 :=
by sorry

end NUMINAMATH_CALUDE_aquarium_species_count_l1836_183647


namespace NUMINAMATH_CALUDE_fraction_product_one_l1836_183693

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem fraction_product_one : 
  ∃ (a b c d e f : ℕ), 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1 ∧
    (a * c * e : ℚ) / (b * d * f : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_one_l1836_183693


namespace NUMINAMATH_CALUDE_initial_production_was_200_l1836_183686

/-- The number of doors per car -/
def doors_per_car : ℕ := 5

/-- The number of cars cut from production due to metal shortages -/
def cars_cut : ℕ := 50

/-- The fraction of remaining production after pandemic cuts -/
def production_fraction : ℚ := 1/2

/-- The final number of doors produced -/
def final_doors : ℕ := 375

/-- Theorem stating that the initial planned production was 200 cars -/
theorem initial_production_was_200 : 
  ∃ (initial_cars : ℕ), 
    (doors_per_car : ℚ) * production_fraction * (initial_cars - cars_cut) = final_doors ∧ 
    initial_cars = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_production_was_200_l1836_183686


namespace NUMINAMATH_CALUDE_rectangle_folding_l1836_183675

/-- Rectangle ABCD with given side lengths -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AD_length : dist A D = 4)
  (AB_length : dist A B = 3)

/-- Point B₁ after folding along diagonal AC -/
def B₁ (rect : Rectangle) : ℝ × ℝ := sorry

/-- Dihedral angle between two planes -/
def dihedral_angle (p₁ p₂ p₃ : ℝ × ℝ) (q₁ q₂ q₃ : ℝ × ℝ) : ℝ := sorry

/-- Distance between two skew lines -/
def skew_line_distance (p₁ p₂ q₁ q₂ : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem rectangle_folding (rect : Rectangle) :
  let b₁ := B₁ rect
  dihedral_angle b₁ rect.D rect.C rect.A rect.C rect.D = Real.arctan (15/16) ∧
  skew_line_distance rect.A b₁ rect.C rect.D = 10 * Real.sqrt 34 / 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_folding_l1836_183675
