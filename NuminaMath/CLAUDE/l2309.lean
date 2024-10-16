import Mathlib

namespace NUMINAMATH_CALUDE_resort_tips_fraction_l2309_230930

theorem resort_tips_fraction (average_tips : ℝ) (h : average_tips > 0) :
  let other_months_total := 6 * average_tips
  let august_tips := 6 * average_tips
  let total_tips := other_months_total + august_tips
  august_tips / total_tips = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_resort_tips_fraction_l2309_230930


namespace NUMINAMATH_CALUDE_man_money_problem_l2309_230995

theorem man_money_problem (x : ℝ) : 
  (((2 * (2 * (2 * (2 * x - 50) - 60) - 70) - 80) = 0) ↔ (x = 53.75)) := by
  sorry

end NUMINAMATH_CALUDE_man_money_problem_l2309_230995


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2309_230992

def A : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
def B : Set (ℝ × ℝ) := {p | 2 ≤ p.1 ∧ p.1 ≤ 3 ∧ 1 ≤ p.2 ∧ p.2 ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {(2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2309_230992


namespace NUMINAMATH_CALUDE_smallest_sum_a_b_l2309_230928

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_sum_a_b (a b : ℕ) 
  (h1 : a^a % b^b = 0)
  (h2 : ¬(a % b = 0))
  (h3 : is_coprime b 210) :
  (∀ (x y : ℕ), x^x % y^y = 0 → ¬(x % y = 0) → is_coprime y 210 → a + b ≤ x + y) →
  a + b = 374 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_a_b_l2309_230928


namespace NUMINAMATH_CALUDE_nursery_school_students_l2309_230956

theorem nursery_school_students (T : ℕ) 
  (h1 : T / 8 + T / 4 + T / 3 + 40 + 60 = T) 
  (h2 : T / 8 + T / 4 + T / 3 = 100) : T = 142 := by
  sorry

end NUMINAMATH_CALUDE_nursery_school_students_l2309_230956


namespace NUMINAMATH_CALUDE_max_value_parabola_l2309_230918

theorem max_value_parabola :
  ∀ x : ℝ, 0 < x → x < 6 → x * (6 - x) ≤ 9 ∧ ∃ y : ℝ, 0 < y ∧ y < 6 ∧ y * (6 - y) = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_parabola_l2309_230918


namespace NUMINAMATH_CALUDE_M_equals_N_l2309_230993

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | 1/x < 1}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l2309_230993


namespace NUMINAMATH_CALUDE_bowling_ball_weight_bowling_ball_weight_proof_l2309_230970

theorem bowling_ball_weight : ℝ → ℝ → Prop :=
  fun (bowling_ball_weight kayak_weight : ℝ) =>
    (8 * bowling_ball_weight = 4 * kayak_weight) ∧
    (3 * kayak_weight = 84) →
    bowling_ball_weight = 14

-- Proof
theorem bowling_ball_weight_proof : bowling_ball_weight 14 28 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_bowling_ball_weight_proof_l2309_230970


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2309_230955

theorem p_sufficient_not_necessary_for_q :
  ∃ (a : ℝ), (a = 1 → abs a = 1) ∧ (abs a = 1 → a = 1 → False) := by
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2309_230955


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l2309_230977

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 34567 * k) → 
  Nat.gcd ((3*x+4)*(8*x+3)*(15*x+11)*(x+15) : ℤ).natAbs x.natAbs = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l2309_230977


namespace NUMINAMATH_CALUDE_largest_valid_pair_l2309_230976

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ is_integer ((a + b) * (a + b + 1) / (a * b : ℚ))

theorem largest_valid_pair :
  ∀ a b : ℕ, valid_pair a b →
    b ≤ 90 ∧
    (b = 90 → a ≤ 35) ∧
    valid_pair 35 90
  := by sorry

end NUMINAMATH_CALUDE_largest_valid_pair_l2309_230976


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2309_230981

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧
  (∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2309_230981


namespace NUMINAMATH_CALUDE_hundred_million_composition_l2309_230907

-- Define the decimal counting system progression rate
def decimal_progression_rate : ℕ := 10

-- Define the units
def one_million : ℕ := 1000000
def ten_million : ℕ := 10000000
def hundred_million : ℕ := 100000000

-- Theorem statement
theorem hundred_million_composition :
  hundred_million = decimal_progression_rate * ten_million ∧
  hundred_million = (decimal_progression_rate * decimal_progression_rate) * one_million :=
by sorry

end NUMINAMATH_CALUDE_hundred_million_composition_l2309_230907


namespace NUMINAMATH_CALUDE_compare_expressions_l2309_230990

theorem compare_expressions : (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l2309_230990


namespace NUMINAMATH_CALUDE_donation_problem_l2309_230936

theorem donation_problem (day1_amount day2_amount : ℕ) 
  (day2_extra_donors : ℕ) (h1 : day1_amount = 4800) 
  (h2 : day2_amount = 6000) (h3 : day2_extra_donors = 50) : 
  ∃ (day1_donors : ℕ), 
    (day1_donors > 0 ∧ day1_donors + day2_extra_donors > 0) ∧
    (day1_amount : ℚ) / day1_donors = (day2_amount : ℚ) / (day1_donors + day2_extra_donors) ∧
    day1_donors + (day1_donors + day2_extra_donors) = 450 ∧
    (day1_amount : ℚ) / day1_donors = 24 :=
by
  sorry

#check donation_problem

end NUMINAMATH_CALUDE_donation_problem_l2309_230936


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l2309_230917

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : 2 ≤ a + b ∧ a + b ≤ 5) 
  (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 3*a - b ≤ x → x ≤ 7) ∧ 
  (∀ y, -2 ≤ y → y ≤ 3*a - b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l2309_230917


namespace NUMINAMATH_CALUDE_range_of_m_l2309_230952

theorem range_of_m (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b - a * b = 0)
  (h_log : ∀ m : ℝ, Real.log ((m^2) / (a + b)) ≤ 0) :
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2309_230952


namespace NUMINAMATH_CALUDE_range_of_m_l2309_230985

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2) →
  (∀ x, g x = 2^x - m) →
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂) →
  m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2309_230985


namespace NUMINAMATH_CALUDE_range_of_a_l2309_230998

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1

def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∃ x₁ > 0, ∃ x₂, f a x₁ ≥ g a x₂) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2309_230998


namespace NUMINAMATH_CALUDE_stick_markings_l2309_230935

theorem stick_markings (stick_length : ℝ) (red_mark : ℝ) (blue_mark : ℝ) : 
  stick_length = 12 →
  red_mark = stick_length / 2 →
  blue_mark = red_mark / 2 →
  red_mark - blue_mark = 3 := by
sorry

end NUMINAMATH_CALUDE_stick_markings_l2309_230935


namespace NUMINAMATH_CALUDE_steve_has_four_friends_l2309_230926

/-- The number of friends Steve has, given the initial number of gold bars,
    the number of lost gold bars, and the number of gold bars each friend receives. -/
def number_of_friends (initial_bars : ℕ) (lost_bars : ℕ) (bars_per_friend : ℕ) : ℕ :=
  (initial_bars - lost_bars) / bars_per_friend

/-- Theorem stating that Steve has 4 friends given the problem conditions. -/
theorem steve_has_four_friends :
  number_of_friends 100 20 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_steve_has_four_friends_l2309_230926


namespace NUMINAMATH_CALUDE_no_groups_of_six_l2309_230948

theorem no_groups_of_six (x y z : ℕ) : 
  (2*x + 6*y + 10*z) / (x + y + z : ℚ) = 5 →
  (2*x + 30*y + 90*z) / (2*x + 6*y + 10*z : ℚ) = 7 →
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_no_groups_of_six_l2309_230948


namespace NUMINAMATH_CALUDE_shoes_selection_ways_l2309_230939

/-- The number of pairs of distinct shoes in the bag -/
def total_pairs : ℕ := 10

/-- The number of shoes taken out -/
def shoes_taken : ℕ := 4

/-- The number of ways to select 4 shoes from 10 pairs such that
    exactly two form a pair and the other two don't form a pair -/
def ways_to_select : ℕ := 1440

/-- Theorem stating the number of ways to select 4 shoes from 10 pairs
    such that exactly two form a pair and the other two don't form a pair -/
theorem shoes_selection_ways (n : ℕ) (h : n = total_pairs) :
  ways_to_select = Nat.choose n 1 * Nat.choose (n - 1) 2 * 2^2 :=
sorry

end NUMINAMATH_CALUDE_shoes_selection_ways_l2309_230939


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2309_230921

theorem solve_exponential_equation :
  ∃ t : ℝ, 4 * (4^t) + Real.sqrt (16 * (16^t)) = 32 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2309_230921


namespace NUMINAMATH_CALUDE_geometric_arithmetic_mean_sum_l2309_230964

theorem geometric_arithmetic_mean_sum (a b c x y : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric sequence condition
  (h2 : x ≠ 0)
  (h3 : y ≠ 0)
  (h4 : 2 * x = a + b)  -- arithmetic mean condition
  (h5 : 2 * y = b + c)  -- arithmetic mean condition
  : a / x + c / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_mean_sum_l2309_230964


namespace NUMINAMATH_CALUDE_river_width_river_width_example_l2309_230967

/-- Calculates the width of a river given its depth, flow rate, and discharge volume. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (discharge_volume : ℝ) : ℝ :=
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  let width := discharge_volume / (flow_rate_mpm * depth)
  width

/-- The width of a river with given parameters is 45 meters. -/
theorem river_width_example : river_width 2 6 9000 = 45 := by
  sorry

end NUMINAMATH_CALUDE_river_width_river_width_example_l2309_230967


namespace NUMINAMATH_CALUDE_log5_of_125_l2309_230984

-- Define the logarithm function for base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log5_of_125 : log5 125 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log5_of_125_l2309_230984


namespace NUMINAMATH_CALUDE_power_result_l2309_230957

theorem power_result (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(2*m - 3*n) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_power_result_l2309_230957


namespace NUMINAMATH_CALUDE_solve_for_y_l2309_230929

theorem solve_for_y (x y n : ℝ) (h : x ≠ y) (h_n : n = (3 * x * y) / (x - y)) :
  y = (n * x) / (3 * x + n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2309_230929


namespace NUMINAMATH_CALUDE_unique_solution_l2309_230974

/-- Represents the number of photos taken by each person -/
structure PhotoCounts where
  C : ℕ  -- Claire
  L : ℕ  -- Lisa
  R : ℕ  -- Robert
  D : ℕ  -- David
  E : ℕ  -- Emma

/-- Checks if the given photo counts satisfy all the conditions -/
def satisfiesConditions (p : PhotoCounts) : Prop :=
  p.L = 3 * p.C ∧
  p.R = p.C + 10 ∧
  p.D = 2 * p.C - 5 ∧
  p.E = 2 * p.R ∧
  p.L + p.R + p.C + p.D + p.E = 350

/-- The unique solution to the photo counting problem -/
def solution : PhotoCounts :=
  { C := 36, L := 108, R := 46, D := 67, E := 93 }

/-- Theorem stating that the solution is unique and satisfies all conditions -/
theorem unique_solution :
  satisfiesConditions solution ∧
  ∀ p : PhotoCounts, satisfiesConditions p → p = solution :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2309_230974


namespace NUMINAMATH_CALUDE_spherical_equation_describes_cone_l2309_230969

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Definition of a cone in 3D space -/
structure Cone where
  vertex : ℝ × ℝ × ℝ
  axis : ℝ × ℝ × ℝ
  openingAngle : ℝ

/-- The equation φ = c in spherical coordinates describes a cone -/
theorem spherical_equation_describes_cone (c : ℝ) :
  ∃ (cone : Cone), ∀ (p : SphericalCoord),
    p.φ = c →
    cone.vertex = (0, 0, 0) ∧
    cone.axis = (0, 0, 1) ∧
    cone.openingAngle = c :=
  sorry

end NUMINAMATH_CALUDE_spherical_equation_describes_cone_l2309_230969


namespace NUMINAMATH_CALUDE_epidemic_test_analysis_l2309_230991

/-- Represents a class of students with their test scores and statistics -/
structure ClassData where
  scores : List Nat
  frequency_table : List (Nat × Nat)
  mean : Nat
  mode : Nat
  median : Nat
  variance : Float

/-- The data for the entire school -/
structure SchoolData where
  total_students : Nat
  class_a : ClassData
  class_b : ClassData

/-- Definition of excellent performance -/
def excellent_score : Nat := 90

/-- The given school data -/
def school_data : SchoolData := {
  total_students := 600,
  class_a := {
    scores := [78, 83, 89, 97, 98, 85, 100, 94, 87, 90, 93, 92, 99, 95, 100],
    frequency_table := [(1, 75), (1, 80), (3, 85), (4, 90), (6, 95)],
    mean := 92,
    mode := 100,
    median := 93,
    variance := 41.07
  },
  class_b := {
    scores := [91, 92, 94, 90, 93],
    frequency_table := [(1, 75), (2, 80), (3, 85), (5, 90), (4, 95)],
    mean := 90,
    mode := 87,
    median := 91,
    variance := 50.2
  }
}

theorem epidemic_test_analysis (data : SchoolData := school_data) :
  (data.class_a.mode = 100) ∧
  (data.class_b.median = 91) ∧
  (((data.class_a.frequency_table.filter (λ x => x.2 ≥ 90)).map (λ x => x.1)).sum +
   ((data.class_b.frequency_table.filter (λ x => x.2 ≥ 90)).map (λ x => x.1)).sum) * 20 = 380 ∧
  (data.class_a.mean > data.class_b.mean ∧ data.class_a.variance < data.class_b.variance) := by
  sorry

end NUMINAMATH_CALUDE_epidemic_test_analysis_l2309_230991


namespace NUMINAMATH_CALUDE_ivanov_net_worth_is_2300000_l2309_230971

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℤ :=
  let apartment_value : ℤ := 3000000
  let car_value : ℤ := 900000
  let bank_deposit : ℤ := 300000
  let securities_value : ℤ := 200000
  let liquid_cash : ℤ := 100000
  let mortgage_balance : ℤ := 1500000
  let car_loan_balance : ℤ := 500000
  let relatives_debt : ℤ := 200000
  let total_assets : ℤ := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities : ℤ := mortgage_balance + car_loan_balance + relatives_debt
  total_assets - total_liabilities

theorem ivanov_net_worth_is_2300000 : ivanov_net_worth = 2300000 := by
  sorry

end NUMINAMATH_CALUDE_ivanov_net_worth_is_2300000_l2309_230971


namespace NUMINAMATH_CALUDE_arun_weight_estimation_l2309_230949

/-- Arun's weight estimation problem -/
theorem arun_weight_estimation (x : ℝ) 
  (h1 : 65 < x)  -- Arun's lower bound
  (h2 : 60 < x ∧ x < 70)  -- Brother's estimation
  (h3 : x ≤ 68)  -- Mother's estimation
  (h4 : (65 + x) / 2 = 67)  -- Average of probable weights
  : x = 68 := by
  sorry

end NUMINAMATH_CALUDE_arun_weight_estimation_l2309_230949


namespace NUMINAMATH_CALUDE_problem_statement_l2309_230933

theorem problem_statement (a b c : ℝ) 
  (h1 : a * b * c * (a + b) * (b + c) * (c + a) ≠ 0)
  (h2 : (a + b + c) * (1 / a + 1 / b + 1 / c) = 1007 / 1008) :
  a * b / ((a + c) * (b + c)) + b * c / ((b + a) * (c + a)) + c * a / ((c + b) * (a + b)) = 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2309_230933


namespace NUMINAMATH_CALUDE_max_b_value_l2309_230945

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := true

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

-- Define the condition for not passing through lattice points
def no_lattice_intersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → is_lattice_point x y → line_equation m x ≠ y

-- State the theorem
theorem max_b_value :
  ∀ b : ℚ, (∀ m : ℚ, 1/3 < m ∧ m < b → no_lattice_intersection m) →
  b ≤ 68/203 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l2309_230945


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2309_230958

theorem quadratic_root_value (a : ℝ) : 
  a^2 - 2*a - 3 = 0 → 2*a^2 - 4*a + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2309_230958


namespace NUMINAMATH_CALUDE_equation_solution_range_l2309_230904

theorem equation_solution_range (x m : ℝ) : 
  x + 3 = 3 * x - m → x ≥ 0 → m ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2309_230904


namespace NUMINAMATH_CALUDE_smallest_m_dividing_power_minus_one_l2309_230927

theorem smallest_m_dividing_power_minus_one :
  ∃ (m : ℕ+), (2^1990 : ℕ) ∣ (1989^(m : ℕ) - 1) ∧
    ∀ (k : ℕ+), (2^1990 : ℕ) ∣ (1989^(k : ℕ) - 1) → m ≤ k :=
by
  use 2^1988
  sorry

end NUMINAMATH_CALUDE_smallest_m_dividing_power_minus_one_l2309_230927


namespace NUMINAMATH_CALUDE_teacher_periods_per_day_l2309_230994

/-- Represents the number of periods a teacher teaches per day -/
def periods_per_day : ℕ := 5

/-- Represents the number of working days per month -/
def days_per_month : ℕ := 24

/-- Represents the payment per period in dollars -/
def payment_per_period : ℕ := 5

/-- Represents the number of months worked -/
def months_worked : ℕ := 6

/-- Represents the total earnings in dollars -/
def total_earnings : ℕ := 3600

/-- Theorem stating that given the conditions, the teacher teaches 5 periods per day -/
theorem teacher_periods_per_day :
  periods_per_day * days_per_month * months_worked * payment_per_period = total_earnings :=
sorry

end NUMINAMATH_CALUDE_teacher_periods_per_day_l2309_230994


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2309_230911

def P : Set (ℝ × ℝ) := {(x, y) | x + y = 0}
def Q : Set (ℝ × ℝ) := {(x, y) | x - y = 2}

theorem intersection_of_P_and_Q : P ∩ Q = {(1, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2309_230911


namespace NUMINAMATH_CALUDE_triangle_base_value_l2309_230912

theorem triangle_base_value (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * x * triangle_height →
  x = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_base_value_l2309_230912


namespace NUMINAMATH_CALUDE_prime_power_equation_solutions_l2309_230931

theorem prime_power_equation_solutions :
  ∀ (p x y : ℕ),
    Prime p →
    x > 0 →
    y > 0 →
    p^x = y^3 + 1 →
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_equation_solutions_l2309_230931


namespace NUMINAMATH_CALUDE_system_solution_l2309_230923

theorem system_solution : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + x₁*y₁ + y₁ = 1 ∧ y₁^2 + x₁*y₁ + x₁ = 5) ∧
    (x₂^2 + x₂*y₂ + y₂ = 1 ∧ y₂^2 + x₂*y₂ + x₂ = 5) ∧
    x₁ = -1 ∧ y₁ = 3 ∧ x₂ = -1 ∧ y₂ = -2 ∧
    ∀ (x y : ℝ), (x^2 + x*y + y = 1 ∧ y^2 + x*y + x = 5) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2309_230923


namespace NUMINAMATH_CALUDE_sector_area_from_arc_and_angle_l2309_230982

/-- Given an arc length of 28 cm and a central angle of 240°, 
    the area of the sector is 294/π cm² -/
theorem sector_area_from_arc_and_angle 
  (arc_length : ℝ) 
  (central_angle : ℝ) 
  (h1 : arc_length = 28) 
  (h2 : central_angle = 240) : 
  (1/2) * arc_length * (arc_length / (central_angle * (π / 180))) = 294 / π :=
by sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_and_angle_l2309_230982


namespace NUMINAMATH_CALUDE_parabola_intersection_l2309_230980

theorem parabola_intersection (k α β : ℝ) : 
  (∀ x, x^2 - (k-1)*x - 3*k - 2 = 0 ↔ x = α ∨ x = β) →
  α^2 + β^2 = 17 →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2309_230980


namespace NUMINAMATH_CALUDE_equation_satisfied_l2309_230959

theorem equation_satisfied (x y z : ℤ) (h1 : x = z + 1) (h2 : y = z) :
  x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l2309_230959


namespace NUMINAMATH_CALUDE_expression_simplification_l2309_230915

theorem expression_simplification :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (1 / (Real.sqrt 5 - 2)))) = 
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2309_230915


namespace NUMINAMATH_CALUDE_twenty_four_game_solvable_l2309_230924

/-- Represents the basic arithmetic operations -/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Represents an expression in the 24 Game -/
inductive Expr
  | Num (n : ℕ)
  | Op (op : Operation) (e1 e2 : Expr)

/-- Evaluates an expression -/
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
  | Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2
  | Expr.Op Operation.Divide e1 e2 => eval e1 / eval e2

/-- Checks if an expression uses all given numbers exactly once -/
def usesAllNumbers (e : Expr) (numbers : List ℕ) : Prop := sorry

/-- The 24 Game theorem -/
theorem twenty_four_game_solvable (numbers : List ℕ := [2, 5, 11, 12]) :
  ∃ e : Expr, usesAllNumbers e numbers ∧ eval e = 24 := by sorry

end NUMINAMATH_CALUDE_twenty_four_game_solvable_l2309_230924


namespace NUMINAMATH_CALUDE_simplify_expression_l2309_230932

theorem simplify_expression (y : ℝ) : (3*y)^3 + (4*y)*(y^2) - 2*y^3 = 29*y^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2309_230932


namespace NUMINAMATH_CALUDE_triangle_altitude_l2309_230943

/-- Given a triangle with area 720 square feet and base 36 feet, prove its altitude is 40 feet -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 720 →
  base = 36 →
  area = (1/2) * base * altitude →
  altitude = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l2309_230943


namespace NUMINAMATH_CALUDE_range_of_x_l2309_230908

theorem range_of_x (a b x : ℝ) (h_a : a ≠ 0) :
  (∀ a b, |a + b| + |a - b| ≥ |a| * (|x - 1| + |x - 2|)) →
  x ∈ Set.Icc (1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2309_230908


namespace NUMINAMATH_CALUDE_f_of_3_equals_5_l2309_230951

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem f_of_3_equals_5 : f 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_5_l2309_230951


namespace NUMINAMATH_CALUDE_article_selling_price_l2309_230966

theorem article_selling_price (CP : ℝ) (SP : ℝ) : 
  CP = 12500 → 
  0.9 * SP = 1.08 * CP → 
  SP = 15000 := by
sorry

end NUMINAMATH_CALUDE_article_selling_price_l2309_230966


namespace NUMINAMATH_CALUDE_plane_points_theorem_l2309_230963

def connecting_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem plane_points_theorem (n₁ n₂ : ℕ) : 
  (connecting_lines n₁ = connecting_lines n₂ + 27) →
  (connecting_lines n₁ + connecting_lines n₂ = 171) →
  (n₁ = 11 ∧ n₂ = 8) :=
by sorry

end NUMINAMATH_CALUDE_plane_points_theorem_l2309_230963


namespace NUMINAMATH_CALUDE_pred_rohem_30_more_pred_rohem_triple_total_sold_is_60_l2309_230996

/-- The number of alarm clocks sold at "Za Rohem" -/
def za_rohem : ℕ := 15

/-- The number of alarm clocks sold at "Před Rohem" -/
def pred_rohem : ℕ := za_rohem + 30

/-- The claim that "Před Rohem" sold 30 more alarm clocks than "Za Rohem" -/
theorem pred_rohem_30_more : pred_rohem = za_rohem + 30 := by sorry

/-- The claim that "Před Rohem" sold three times as many alarm clocks as "Za Rohem" -/
theorem pred_rohem_triple : pred_rohem = 3 * za_rohem := by sorry

/-- The total number of alarm clocks sold at both shops -/
def total_sold : ℕ := za_rohem + pred_rohem

/-- Proof that the total number of alarm clocks sold at both shops is 60 -/
theorem total_sold_is_60 : total_sold = 60 := by sorry

end NUMINAMATH_CALUDE_pred_rohem_30_more_pred_rohem_triple_total_sold_is_60_l2309_230996


namespace NUMINAMATH_CALUDE_existence_of_uv_l2309_230910

theorem existence_of_uv (m n X : ℕ) (hm : X ≥ m) (hn : X ≥ n) :
  ∃ u v : ℤ,
    (|u| + |v| > 0) ∧
    (|u| ≤ Real.sqrt X) ∧
    (|v| ≤ Real.sqrt X) ∧
    (0 ≤ m * u + n * v) ∧
    (m * u + n * v ≤ 2 * Real.sqrt X) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_uv_l2309_230910


namespace NUMINAMATH_CALUDE_books_sold_on_thursday_l2309_230987

theorem books_sold_on_thursday (initial_stock : ℕ) (sold_monday : ℕ) (sold_tuesday : ℕ)
  (sold_wednesday : ℕ) (sold_friday : ℕ) (unsold : ℕ) :
  initial_stock = 800 →
  sold_monday = 60 →
  sold_tuesday = 10 →
  sold_wednesday = 20 →
  sold_friday = 66 →
  unsold = 600 →
  initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_friday + unsold) = 44 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_on_thursday_l2309_230987


namespace NUMINAMATH_CALUDE_minimize_sqrt_difference_l2309_230950

theorem minimize_sqrt_difference (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (x y : ℕ), 
    (x > 0 ∧ y > 0) ∧
    (x ≤ y) ∧
    (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0) ∧
    (∀ (a b : ℕ), (a > 0 ∧ b > 0) → (a ≤ b) → 
      (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) →
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b)) ∧
    (x = (p - 1) / 2) ∧
    (y = (p + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_minimize_sqrt_difference_l2309_230950


namespace NUMINAMATH_CALUDE_simplify_fraction_l2309_230941

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2309_230941


namespace NUMINAMATH_CALUDE_min_tablets_for_both_types_l2309_230946

/-- Given a box with tablets of two types of medicine, this theorem proves
    the minimum number of tablets needed to ensure at least one of each type
    when extracting a specific total number. -/
theorem min_tablets_for_both_types 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extract_total : ℕ) 
  (h1 : total_A = 10) 
  (h2 : total_B = 16) 
  (h3 : extract_total = 18) :
  extract_total = min (total_A + total_B) extract_total := by
sorry

end NUMINAMATH_CALUDE_min_tablets_for_both_types_l2309_230946


namespace NUMINAMATH_CALUDE_triangle_is_acute_l2309_230961

-- Define the triangle and its angles
def Triangle (a1 a2 a3 : ℝ) : Prop :=
  a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a1 + a2 + a3 = 180

-- Define an acute triangle
def AcuteTriangle (a1 a2 a3 : ℝ) : Prop :=
  Triangle a1 a2 a3 ∧ a1 < 90 ∧ a2 < 90 ∧ a3 < 90

-- Theorem statement
theorem triangle_is_acute (a2 : ℝ) :
  Triangle (2 * a2) a2 (1.5 * a2) → AcuteTriangle (2 * a2) a2 (1.5 * a2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_is_acute_l2309_230961


namespace NUMINAMATH_CALUDE_fish_added_calculation_james_added_eight_fish_l2309_230925

theorem fish_added_calculation (initial_fish : ℕ) (fish_eaten_per_day : ℕ) 
  (days_before_adding : ℕ) (days_after_adding : ℕ) (final_fish : ℕ) : ℕ :=
  let total_days := days_before_adding + days_after_adding
  let total_fish_eaten := total_days * fish_eaten_per_day
  let expected_remaining := initial_fish - total_fish_eaten
  final_fish - expected_remaining
  
-- The main theorem
theorem james_added_eight_fish : 
  fish_added_calculation 60 2 14 7 26 = 8 := by
sorry

end NUMINAMATH_CALUDE_fish_added_calculation_james_added_eight_fish_l2309_230925


namespace NUMINAMATH_CALUDE_jellybean_distribution_l2309_230965

theorem jellybean_distribution (total_jellybeans : ℕ) (nephews : ℕ) (nieces : ℕ) 
  (h1 : total_jellybeans = 70)
  (h2 : nephews = 3)
  (h3 : nieces = 2) :
  total_jellybeans / (nephews + nieces) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_distribution_l2309_230965


namespace NUMINAMATH_CALUDE_northton_capsule_depth_l2309_230968

/-- The depth of Southton's time capsule in feet -/
def southton_depth : ℝ := 15

/-- The depth of Northton's time capsule in feet -/
def northton_depth : ℝ := 4 * southton_depth - 12

/-- Theorem stating the depth of Northton's time capsule -/
theorem northton_capsule_depth : northton_depth = 48 := by
  sorry

end NUMINAMATH_CALUDE_northton_capsule_depth_l2309_230968


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2309_230940

-- Define the arithmetic sequence
def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmeticSequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2309_230940


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l2309_230900

/-- Given a circle with radius 3 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    the length of the longer side of the rectangle is 4.5π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ) :
  circle_radius = 3 →
  rectangle_area = 3 * circle_area →
  circle_area = π * circle_radius^2 →
  (4.5 * π : ℝ) * (2 * circle_radius) = rectangle_area :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l2309_230900


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l2309_230914

/-- Given an isosceles triangle and a similar triangle, calculates the perimeter of the larger triangle -/
theorem similar_triangle_perimeter (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a = b →
  c > a →
  c > b →
  d > c →
  (a + b + c) * (d / c) = 100 :=
by
  sorry

#check similar_triangle_perimeter

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l2309_230914


namespace NUMINAMATH_CALUDE_square_side_increase_percentage_l2309_230999

theorem square_side_increase_percentage (a : ℝ) (x : ℝ) :
  (a > 0) →
  (x > 0) →
  (a * (1 + x / 100) * 1.8)^2 = 2.592 * (a^2 + (a * (1 + x / 100))^2) →
  x = 100 := by sorry

end NUMINAMATH_CALUDE_square_side_increase_percentage_l2309_230999


namespace NUMINAMATH_CALUDE_inequality_two_integer_solutions_l2309_230919

def has_exactly_two_integer_solutions (a : ℝ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧
    (x : ℝ)^2 - (a + 1) * (x : ℝ) + a < 0 ∧
    (y : ℝ)^2 - (a + 1) * (y : ℝ) + a < 0 ∧
    ∀ z : ℤ, z ≠ x → z ≠ y → (z : ℝ)^2 - (a + 1) * (z : ℝ) + a ≥ 0

theorem inequality_two_integer_solutions :
  {a : ℝ | has_exactly_two_integer_solutions a} = {a : ℝ | (3 < a ∧ a ≤ 4) ∨ (-2 ≤ a ∧ a < -1)} :=
by sorry

end NUMINAMATH_CALUDE_inequality_two_integer_solutions_l2309_230919


namespace NUMINAMATH_CALUDE_fourth_power_representation_l2309_230909

/-- For any base N ≥ 6, (N-1)^4 in base N can be represented as (N-4)5(N-4)1 -/
theorem fourth_power_representation (N : ℕ) (h : N ≥ 6) :
  ∃ (a b c d : ℕ), (N - 1)^4 = a * N^3 + b * N^2 + c * N + d ∧
                    a = N - 4 ∧
                    b = 5 ∧
                    c = N - 4 ∧
                    d = 1 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_representation_l2309_230909


namespace NUMINAMATH_CALUDE_min_lines_proof_l2309_230997

/-- The number of regions created by n lines in a plane -/
def regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- The minimum number of lines needed to divide a plane into at least 1000 regions -/
def min_lines_for_1000_regions : ℕ := 45

theorem min_lines_proof :
  (∀ k < min_lines_for_1000_regions, regions k < 1000) ∧
  regions min_lines_for_1000_regions ≥ 1000 := by
  sorry

#eval regions min_lines_for_1000_regions

end NUMINAMATH_CALUDE_min_lines_proof_l2309_230997


namespace NUMINAMATH_CALUDE_greatest_ba_value_l2309_230938

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem greatest_ba_value (a b : ℕ) :
  is_prime a →
  is_prime b →
  a < 10 →
  b < 10 →
  is_divisible_by (110 * 10 + a * 10 + b) 55 →
  (∀ a' b' : ℕ, 
    is_prime a' → 
    is_prime b' → 
    a' < 10 → 
    b' < 10 → 
    is_divisible_by (110 * 10 + a' * 10 + b') 55 → 
    b * a ≥ b' * a') →
  b * a = 15 := by
sorry

end NUMINAMATH_CALUDE_greatest_ba_value_l2309_230938


namespace NUMINAMATH_CALUDE_guaranteed_matches_l2309_230913

/-- Represents a card in a deck -/
structure Card :=
  (suit : Fin 4)
  (rank : Fin 9)

/-- A deck of cards -/
def Deck := List Card

/-- A function to split a deck into two halves -/
def split_deck (d : Deck) : Deck × Deck :=
  sorry

/-- A function to count matching pairs between two sets of cards -/
def count_matches (d1 d2 : Deck) : Nat :=
  sorry

/-- The theorem stating that the second player can always guarantee at least 15 matching pairs -/
theorem guaranteed_matches (d : Deck) : 
  d.length = 36 → ∀ (d1 d2 : Deck), split_deck d = (d1, d2) → count_matches d1 d2 ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_guaranteed_matches_l2309_230913


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2309_230979

theorem fraction_equation_solution (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (2 : ℚ) / 7 = 1 / (a : ℚ) + 1 / (b : ℚ) → a = 28 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2309_230979


namespace NUMINAMATH_CALUDE_union_contains_1980_l2309_230901

/-- An arithmetic progression of integers -/
def ArithmeticProgression (a₀ d : ℤ) : Set ℤ :=
  {n : ℤ | ∃ k : ℕ, n = a₀ + k * d}

theorem union_contains_1980
  (A B C : Set ℤ)
  (hA : ∃ a₀ d : ℤ, A = ArithmeticProgression a₀ d)
  (hB : ∃ a₀ d : ℤ, B = ArithmeticProgression a₀ d)
  (hC : ∃ a₀ d : ℤ, C = ArithmeticProgression a₀ d)
  (h_union : {1, 2, 3, 4, 5, 6, 7, 8} ⊆ A ∪ B ∪ C) :
  1980 ∈ A ∪ B ∪ C :=
sorry

end NUMINAMATH_CALUDE_union_contains_1980_l2309_230901


namespace NUMINAMATH_CALUDE_max_pieces_theorem_l2309_230934

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.eraseDups.length

def max_pieces : ℕ := 7

theorem max_pieces_theorem :
  ∀ n : ℕ, n > max_pieces →
    ¬∃ (A B : ℕ), is_five_digit A ∧ is_five_digit B ∧ has_distinct_digits A ∧ A = B * n :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_theorem_l2309_230934


namespace NUMINAMATH_CALUDE_range_of_a_l2309_230960

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2309_230960


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l2309_230903

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  (∀ x y, y = x^2 + b*x + c → 
    ((x = 2 ∧ y = 5) ∨ (x = 4 ∧ y = 9))) → 
  c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l2309_230903


namespace NUMINAMATH_CALUDE_final_bacteria_count_l2309_230937

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 50

-- Define the doubling interval in minutes
def doubling_interval : ℕ := 4

-- Define the total time elapsed in minutes
def total_time : ℕ := 15

-- Define the number of complete doubling intervals
def complete_intervals : ℕ := total_time / doubling_interval

-- Function to calculate the bacteria population after a given number of intervals
def bacteria_population (intervals : ℕ) : ℕ :=
  initial_bacteria * (2 ^ intervals)

-- Theorem stating the final bacteria count
theorem final_bacteria_count :
  bacteria_population complete_intervals = 400 := by
  sorry

end NUMINAMATH_CALUDE_final_bacteria_count_l2309_230937


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l2309_230947

theorem line_satisfies_conditions : ∃! k : ℝ,
  let f (x : ℝ) := x^2 + 8*x + 7
  let g (x : ℝ) := 19.5*x - 32
  let p1 := (k, f k)
  let p2 := (k, g k)
  (g 2 = 7) ∧
  (abs (f k - g k) = 6) ∧
  (-32 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l2309_230947


namespace NUMINAMATH_CALUDE_derivative_at_alpha_l2309_230942

theorem derivative_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ α^2 - Real.cos x
  deriv f α = Real.sin α := by
sorry

end NUMINAMATH_CALUDE_derivative_at_alpha_l2309_230942


namespace NUMINAMATH_CALUDE_trig_simplification_l2309_230916

theorem trig_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2309_230916


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2309_230989

def divisors : List ℕ := [12, 16, 18, 21, 28, 35, 40, 45, 55]

theorem smallest_number_divisible (n : ℕ) : 
  (∀ d ∈ divisors, (n - 10) % d = 0) →
  (∀ m < n, ∃ d ∈ divisors, (m - 10) % d ≠ 0) →
  n = 55450 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2309_230989


namespace NUMINAMATH_CALUDE_range_of_a_l2309_230975

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| < 1}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 2}

-- Define the complement of B
def complementB : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  A a ⊆ complementB → 3 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2309_230975


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l2309_230983

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : 0 < n

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle (in degrees) for rotational symmetry -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ (p : RegularPolygon 17),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l2309_230983


namespace NUMINAMATH_CALUDE_six_eight_ten_right_triangle_l2309_230905

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem statement
theorem six_eight_ten_right_triangle :
  is_right_triangle 6 8 10 :=
sorry

end NUMINAMATH_CALUDE_six_eight_ten_right_triangle_l2309_230905


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2309_230906

/-- The number of apples initially in the cafeteria -/
def initial_apples : ℕ := 50

/-- The number of oranges initially in the cafeteria -/
def initial_oranges : ℕ := 40

/-- The cost of an apple in dollars -/
def apple_cost : ℚ := 4/5

/-- The cost of an orange in dollars -/
def orange_cost : ℚ := 1/2

/-- The number of apples left after selling -/
def remaining_apples : ℕ := 10

/-- The number of oranges left after selling -/
def remaining_oranges : ℕ := 6

/-- The total earnings from selling apples and oranges in dollars -/
def total_earnings : ℚ := 49

theorem cafeteria_apples :
  apple_cost * (initial_apples - remaining_apples : ℚ) +
  orange_cost * (initial_oranges - remaining_oranges : ℚ) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2309_230906


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l2309_230922

theorem min_value_sum_of_squares (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l2309_230922


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2309_230962

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2309_230962


namespace NUMINAMATH_CALUDE_false_proposition_l2309_230973

def p1 : Prop := ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0

def p2 : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

theorem false_proposition : ¬((¬p1) ∧ (¬p2)) := by
  sorry

end NUMINAMATH_CALUDE_false_proposition_l2309_230973


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2309_230944

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) (h_s : s = 8) (h_r : r = 3) :
  s^2 - 4 * Real.pi * r^2 = 64 - 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2309_230944


namespace NUMINAMATH_CALUDE_find_k_value_l2309_230988

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l2309_230988


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2309_230902

theorem factorization_of_cubic (b : ℝ) : 2*b^3 - 4*b^2 + 2*b = 2*b*(b-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2309_230902


namespace NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l2309_230986

theorem least_positive_integer_for_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 10) ∧ k ∣ (9*m + 11))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 10) ∧ k ∣ (9*n + 11)) ∧
  n = 111 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l2309_230986


namespace NUMINAMATH_CALUDE_equation_solution_l2309_230954

theorem equation_solution : ∃ (x : ℝ), 45 - (28 - (37 - (x - 19))) = 58 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2309_230954


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_sum_162_l2309_230920

theorem smallest_consecutive_even_sum_162 (n : ℤ) : 
  (∃ (a b c : ℤ), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ a + b + c = 162) → n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_sum_162_l2309_230920


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_x_in_range_f_always_negative_implies_x_in_range_l2309_230953

/-- The function f(x) = ax^2 - (2a+1)x + a+1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

theorem f_nonnegative_iff_x_in_range (a : ℝ) (x : ℝ) :
  a = 2 → (f a x ≥ 0 ↔ x ≥ 3/2 ∨ x ≤ 1) := by sorry

theorem f_always_negative_implies_x_in_range (a : ℝ) (x : ℝ) :
  a ∈ Set.Icc (-2) 2 → (∀ y, f a y < 0) → x ∈ Set.Ioo 1 (3/2) := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_x_in_range_f_always_negative_implies_x_in_range_l2309_230953


namespace NUMINAMATH_CALUDE_sqrt_five_multiplication_l2309_230978

theorem sqrt_five_multiplication : 2 * Real.sqrt 5 * (3 * Real.sqrt 5) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_multiplication_l2309_230978


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l2309_230972

theorem sum_a_b_equals_one (a b : ℝ) : 
  Real.sqrt (a - b - 3) + abs (2 * a - 4) = 0 → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l2309_230972
