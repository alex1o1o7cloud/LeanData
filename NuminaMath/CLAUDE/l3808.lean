import Mathlib

namespace NUMINAMATH_CALUDE_gcd_equality_implies_equal_l3808_380879

theorem gcd_equality_implies_equal (a b c : ℕ+) :
  a + Nat.gcd a b = b + Nat.gcd b c ∧
  b + Nat.gcd b c = c + Nat.gcd c a →
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_gcd_equality_implies_equal_l3808_380879


namespace NUMINAMATH_CALUDE_car_rental_per_mile_rate_l3808_380818

theorem car_rental_per_mile_rate (daily_rate : ℝ) (daily_budget : ℝ) (distance : ℝ) :
  daily_rate = 30 →
  daily_budget = 76 →
  distance = 200 →
  (daily_budget - daily_rate) / distance * 100 = 23 := by
sorry

end NUMINAMATH_CALUDE_car_rental_per_mile_rate_l3808_380818


namespace NUMINAMATH_CALUDE_fractional_parts_inequality_l3808_380876

theorem fractional_parts_inequality (q : ℕ+) (hq : ¬ ∃ (m : ℕ), m^3 = q) :
  ∃ (c : ℝ), c > 0 ∧ ∀ (n : ℕ+),
    (nq^(1/3:ℝ) - ⌊nq^(1/3:ℝ)⌋) + (nq^(2/3:ℝ) - ⌊nq^(2/3:ℝ)⌋) ≥ c * n^(-1/2:ℝ) :=
by sorry

end NUMINAMATH_CALUDE_fractional_parts_inequality_l3808_380876


namespace NUMINAMATH_CALUDE_lines_do_not_intersect_l3808_380846

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

theorem lines_do_not_intersect (k : ℝ) : 
  (parallel 
    { point := (1, 3), direction := (2, -5) }
    { point := (-1, 4), direction := (3, k) }) ↔ 
  k = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_lines_do_not_intersect_l3808_380846


namespace NUMINAMATH_CALUDE_dividend_calculation_l3808_380842

/-- Calculates the total dividend paid to a shareholder --/
def total_dividend (expected_earnings : ℚ) (actual_earnings : ℚ) (base_dividend_ratio : ℚ) 
  (additional_dividend_rate : ℚ) (additional_earnings_threshold : ℚ) (num_shares : ℕ) : ℚ :=
  let base_dividend := expected_earnings * base_dividend_ratio
  let earnings_difference := actual_earnings - expected_earnings
  let additional_dividend := 
    if earnings_difference > 0 
    then (earnings_difference / additional_earnings_threshold).floor * additional_dividend_rate
    else 0
  let total_dividend_per_share := base_dividend + additional_dividend
  total_dividend_per_share * num_shares

/-- Theorem stating the total dividend paid to a shareholder with given conditions --/
theorem dividend_calculation : 
  total_dividend 0.80 1.10 (1/2) 0.04 0.10 100 = 52 := by
  sorry

#eval total_dividend 0.80 1.10 (1/2) 0.04 0.10 100

end NUMINAMATH_CALUDE_dividend_calculation_l3808_380842


namespace NUMINAMATH_CALUDE_inequality_always_holds_l3808_380870

theorem inequality_always_holds (a b c : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) :
  a + c > b + c := by sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l3808_380870


namespace NUMINAMATH_CALUDE_total_harvest_earnings_l3808_380822

/-- Lewis's weekly earnings during the harvest -/
def weekly_earnings : ℕ := 2

/-- Duration of the harvest in weeks -/
def harvest_duration : ℕ := 89

/-- Theorem stating the total earnings for the harvest -/
theorem total_harvest_earnings :
  weekly_earnings * harvest_duration = 178 := by sorry

end NUMINAMATH_CALUDE_total_harvest_earnings_l3808_380822


namespace NUMINAMATH_CALUDE_church_cookie_baking_l3808_380807

theorem church_cookie_baking (members : ℕ) (cookies_per_sheet : ℕ) (total_cookies : ℕ) 
  (h1 : members = 100)
  (h2 : cookies_per_sheet = 16)
  (h3 : total_cookies = 16000) :
  total_cookies / (members * cookies_per_sheet) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_church_cookie_baking_l3808_380807


namespace NUMINAMATH_CALUDE_all_truth_probability_l3808_380833

def alice_truth_prob : ℝ := 0.7
def bob_truth_prob : ℝ := 0.6
def carol_truth_prob : ℝ := 0.8
def david_truth_prob : ℝ := 0.5

theorem all_truth_probability :
  alice_truth_prob * bob_truth_prob * carol_truth_prob * david_truth_prob = 0.168 := by
  sorry

end NUMINAMATH_CALUDE_all_truth_probability_l3808_380833


namespace NUMINAMATH_CALUDE_inequality_proof_l3808_380859

theorem inequality_proof (x y : ℝ) : x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3808_380859


namespace NUMINAMATH_CALUDE_grunters_win_all_games_l3808_380868

/-- The probability of the Grunters winning a single game -/
def p : ℚ := 3/4

/-- The number of games in the series -/
def n : ℕ := 6

/-- The theorem stating the probability of the Grunters winning all games -/
theorem grunters_win_all_games : p ^ n = 729/4096 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_all_games_l3808_380868


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l3808_380865

theorem quadratic_equations_common_root (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_common_root1 : ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0)
  (h_common_root2 : ∃! x : ℝ, x^2 + b*x + c = 0 ∧ x^2 + c*x + a = 0)
  (h_common_root3 : ∃! x : ℝ, x^2 + c*x + a = 0 ∧ x^2 + a*x + b = 0) :
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l3808_380865


namespace NUMINAMATH_CALUDE_intersection_not_roots_l3808_380813

theorem intersection_not_roots : ∀ x : ℝ, 
  (x = x - 3 → x^2 - 3*x ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_not_roots_l3808_380813


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3808_380850

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 - 4*Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3808_380850


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3808_380828

theorem divisibility_of_sum_of_squares (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p → 
  (x + y + z) ∣ (x^2 + y^2 + z^2) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3808_380828


namespace NUMINAMATH_CALUDE_rabbit_population_l3808_380872

theorem rabbit_population (breeding_rabbits : ℕ) (first_spring_ratio : ℕ) 
  (second_spring_kittens : ℕ) (second_spring_adopted : ℕ) (total_rabbits : ℕ) :
  breeding_rabbits = 10 →
  second_spring_kittens = 60 →
  second_spring_adopted = 4 →
  total_rabbits = 121 →
  breeding_rabbits + (first_spring_ratio * breeding_rabbits / 2 + 5) + 
    (second_spring_kittens - second_spring_adopted) = total_rabbits →
  first_spring_ratio = 10 := by
sorry

end NUMINAMATH_CALUDE_rabbit_population_l3808_380872


namespace NUMINAMATH_CALUDE_age_puzzle_l3808_380852

theorem age_puzzle (A : ℕ) (N : ℚ) (h1 : A = 24) (h2 : (A + 3) * N - (A - 3) * N = A) : N = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3808_380852


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3808_380878

theorem binomial_coefficient_congruence (p a b : ℕ) : 
  Nat.Prime p → a ≥ b → b ≥ 0 → 
  (Nat.choose (p * a) (p * b)) ≡ (Nat.choose a b) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3808_380878


namespace NUMINAMATH_CALUDE_x_divisibility_l3808_380857

def x : ℤ := 64 + 96 + 128 + 160 + 288 + 352 + 3232

theorem x_divisibility :
  (∃ k : ℤ, x = 4 * k) ∧
  (∃ k : ℤ, x = 8 * k) ∧
  (∃ k : ℤ, x = 16 * k) ∧
  (∃ k : ℤ, x = 32 * k) :=
by sorry

end NUMINAMATH_CALUDE_x_divisibility_l3808_380857


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l3808_380898

theorem imaginary_unit_sum (i : ℂ) : i * i = -1 → Complex.abs (-i) + i^2018 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l3808_380898


namespace NUMINAMATH_CALUDE_hierarchy_combinations_l3808_380809

def society_size : ℕ := 12
def num_dukes : ℕ := 3
def knights_per_duke : ℕ := 2

def choose_hierarchy : ℕ := 
  society_size * 
  (society_size - 1) * 
  (society_size - 2) * 
  (society_size - 3) * 
  (Nat.choose (society_size - 4) knights_per_duke) * 
  (Nat.choose (society_size - 4 - knights_per_duke) knights_per_duke) * 
  (Nat.choose (society_size - 4 - 2 * knights_per_duke) knights_per_duke)

theorem hierarchy_combinations : 
  choose_hierarchy = 907200 :=
by sorry

end NUMINAMATH_CALUDE_hierarchy_combinations_l3808_380809


namespace NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l3808_380886

/-- Represents a rectangular board --/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a rectangle that can be placed on the board --/
structure Rectangle :=
  (length : Nat)
  (width : Nat)

/-- Function to calculate the minimum number of cells to mark --/
def min_marked_cells (b : Board) (r : Rectangle) : Nat :=
  sorry

/-- Theorem stating the minimum number of cells to mark for the given problem --/
theorem min_marked_cells_13x13_board (b : Board) (r : Rectangle) :
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  min_marked_cells b r = 84 :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l3808_380886


namespace NUMINAMATH_CALUDE_sqrt_3_plus_2_times_sqrt_3_minus_2_l3808_380881

theorem sqrt_3_plus_2_times_sqrt_3_minus_2 : (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_plus_2_times_sqrt_3_minus_2_l3808_380881


namespace NUMINAMATH_CALUDE_fraction_equals_negative_one_l3808_380826

theorem fraction_equals_negative_one (a b : ℝ) (h : a + b ≠ 0) :
  (-a - b) / (a + b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_negative_one_l3808_380826


namespace NUMINAMATH_CALUDE_repeating_decimal_56_l3808_380840

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56 :
  RepeatingDecimal 5 6 = 56 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_l3808_380840


namespace NUMINAMATH_CALUDE_point_M_coordinates_l3808_380863

-- Define the points A, B, C, and M
def A : ℝ × ℝ := (2, -4)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (3, 4)
def M : ℝ × ℝ := (-11, -15)

-- Define vectors
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- State the theorem
theorem point_M_coordinates :
  vec C M = (2 : ℝ) • (vec C A) + (3 : ℝ) • (vec C B) → M = (-11, -15) := by
  sorry


end NUMINAMATH_CALUDE_point_M_coordinates_l3808_380863


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l3808_380871

theorem rectangular_box_dimensions (X Y Z : ℝ) 
  (h1 : X * Y = 32)
  (h2 : X * Z = 50)
  (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l3808_380871


namespace NUMINAMATH_CALUDE_sams_money_l3808_380808

-- Define the value of a penny and a quarter in cents
def penny_value : ℚ := 1
def quarter_value : ℚ := 25

-- Define the number of pennies and quarters Sam has
def num_pennies : ℕ := 9
def num_quarters : ℕ := 7

-- Calculate the total value in cents
def total_value : ℚ := num_pennies * penny_value + num_quarters * quarter_value

-- Theorem to prove
theorem sams_money : total_value = 184 := by
  sorry

end NUMINAMATH_CALUDE_sams_money_l3808_380808


namespace NUMINAMATH_CALUDE_total_beignets_in_16_weeks_l3808_380854

/-- The number of beignets Sandra eats each morning -/
def daily_beignets : ℕ := 3

/-- The number of weeks we're considering -/
def weeks : ℕ := 16

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating the total number of beignets Sandra will eat in 16 weeks -/
theorem total_beignets_in_16_weeks : 
  daily_beignets * days_per_week * weeks = 336 := by
  sorry

end NUMINAMATH_CALUDE_total_beignets_in_16_weeks_l3808_380854


namespace NUMINAMATH_CALUDE_tangent_line_proof_l3808_380896

def circle_center : ℝ × ℝ := (6, 3)
def circle_radius : ℝ := 2
def point_p : ℝ × ℝ := (10, 0)

def is_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

def is_on_line (p : ℝ × ℝ) : Prop :=
  4 * p.1 - 3 * p.2 = 19

theorem tangent_line_proof :
  ∃ (q : ℝ × ℝ),
    is_on_circle q ∧
    is_on_line q ∧
    is_on_line point_p ∧
    ∀ (r : ℝ × ℝ), is_on_circle r ∧ is_on_line r → r = q :=
  sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l3808_380896


namespace NUMINAMATH_CALUDE_two_intersections_iff_m_values_l3808_380803

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

theorem two_intersections_iff_m_values (m : ℝ) : 
  (∃! (p q : ℝ × ℝ), (p.1 = 0 ∨ p.2 = 0) ∧ (q.1 = 0 ∨ q.2 = 0) ∧ 
    p ≠ q ∧ f m p.1 = p.2 ∧ f m q.1 = q.2) ↔ 
  (m = 1 ∨ m = -5/4) := by
sorry

end NUMINAMATH_CALUDE_two_intersections_iff_m_values_l3808_380803


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l3808_380894

theorem largest_multiple_of_seven_below_negative_85 :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -85 → n ≤ -91 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l3808_380894


namespace NUMINAMATH_CALUDE_three_valid_plans_l3808_380823

/-- Represents the cost and construction details of parking spaces -/
structure ParkingProject where
  aboveGroundCost : ℚ
  undergroundCost : ℚ
  totalSpaces : ℕ
  minInvestment : ℚ
  maxInvestment : ℚ

/-- Calculates the number of valid construction plans -/
def validConstructionPlans (project : ParkingProject) : ℕ :=
  (project.totalSpaces + 1).fold
    (λ count aboveGround =>
      let underground := project.totalSpaces - aboveGround
      let cost := project.aboveGroundCost * aboveGround + project.undergroundCost * underground
      if project.minInvestment < cost ∧ cost ≤ project.maxInvestment then
        count + 1
      else
        count)
    0

/-- Theorem stating that there are exactly 3 valid construction plans -/
theorem three_valid_plans (project : ParkingProject)
  (h1 : project.aboveGroundCost + project.undergroundCost = 0.6)
  (h2 : 3 * project.aboveGroundCost + 2 * project.undergroundCost = 1.3)
  (h3 : project.totalSpaces = 50)
  (h4 : project.minInvestment = 12)
  (h5 : project.maxInvestment = 13) :
  validConstructionPlans project = 3 := by
  sorry

#eval validConstructionPlans {
  aboveGroundCost := 0.1,
  undergroundCost := 0.5,
  totalSpaces := 50,
  minInvestment := 12,
  maxInvestment := 13
}

end NUMINAMATH_CALUDE_three_valid_plans_l3808_380823


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3808_380829

theorem inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3808_380829


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3808_380832

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 4x^2 - 6x + 9 has discriminant -108 -/
theorem quadratic_discriminant :
  discriminant 4 (-6) 9 = -108 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3808_380832


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l3808_380889

-- Define the quadrilateral ABCD and point M
variable (A B C D M : ℝ × ℝ)

-- Define the convexity of ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the condition that M is inside ABCD
def is_inside (M A B C D : ℝ × ℝ) : Prop := sorry

-- Define angle measure
def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_inequality
  (h_convex : is_convex_quadrilateral A B C D)
  (h_inside : is_inside M A B C D)
  (h_angle1 : angle A M B = angle A D M + angle B C M)
  (h_angle2 : angle A M D = angle A B M + angle D C M) :
  distance A M * distance C M + distance B M * distance D M ≥ 
  Real.sqrt (distance A B * distance B C * distance C D * distance D A) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l3808_380889


namespace NUMINAMATH_CALUDE_loan_problem_l3808_380869

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- Theorem stating the conditions and conclusion of the loan problem -/
theorem loan_problem (loan : SimpleLoan) 
  (h1 : loan.time = loan.rate)
  (h2 : loan.interest = 108)
  (h3 : loan.rate = 0.03)
  (h4 : loan.interest = loan.principal * loan.rate * loan.time) :
  loan.principal = 1200 := by
  sorry

#check loan_problem

end NUMINAMATH_CALUDE_loan_problem_l3808_380869


namespace NUMINAMATH_CALUDE_childrens_bikes_count_l3808_380867

theorem childrens_bikes_count (regular_bikes : ℕ) (regular_wheels : ℕ) (childrens_wheels : ℕ) (total_wheels : ℕ) :
  regular_bikes = 7 →
  regular_wheels = 2 →
  childrens_wheels = 4 →
  total_wheels = 58 →
  regular_bikes * regular_wheels + childrens_wheels * (total_wheels - regular_bikes * regular_wheels) / childrens_wheels = 11 :=
by
  sorry

#check childrens_bikes_count

end NUMINAMATH_CALUDE_childrens_bikes_count_l3808_380867


namespace NUMINAMATH_CALUDE_composite_numbers_equal_if_same_main_divisors_l3808_380814

/-- Main divisors of a natural number -/
def main_divisors (n : ℕ) : Set ℕ :=
  {d ∈ Nat.divisors n | d ≠ n ∧ d > 1 ∧ ∀ e ∈ Nat.divisors n, e ≠ n → e ≤ d}

/-- Two largest elements of a finite set of natural numbers -/
def two_largest (s : Set ℕ) : Set ℕ :=
  {x ∈ s | ∀ y ∈ s, y ≤ x ∨ ∃ z ∈ s, z ≠ x ∧ z ≠ y ∧ y ≤ z}

theorem composite_numbers_equal_if_same_main_divisors
  (a b : ℕ) (ha : ¬Nat.Prime a) (hb : ¬Nat.Prime b)
  (h : two_largest (main_divisors a) = two_largest (main_divisors b)) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_composite_numbers_equal_if_same_main_divisors_l3808_380814


namespace NUMINAMATH_CALUDE_badminton_tournament_l3808_380839

theorem badminton_tournament (n : ℕ) : 
  (∃ (x : ℕ), 
    (5 * n * (5 * n - 1)) / 2 = 7 * x ∧ 
    4 * x = (2 * n * (2 * n - 1)) / 2 + 2 * n * 3 * n) → 
  n = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_badminton_tournament_l3808_380839


namespace NUMINAMATH_CALUDE_distinct_values_count_l3808_380849

-- Define the expression
def base : ℕ := 3
def expr := base^(base^(base^base))

-- Define the possible parenthesizations
def p1 := base^(base^(base^base))
def p2 := base^((base^base)^base)
def p3 := ((base^base)^base)^base
def p4 := (base^(base^base))^base
def p5 := (base^base)^(base^base)

-- Theorem statement
theorem distinct_values_count :
  ∃ (s : Finset ℕ), (∀ x : ℕ, x ∈ s ↔ (x = p1 ∨ x = p2 ∨ x = p3 ∨ x = p4 ∨ x = p5)) ∧ Finset.card s = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_count_l3808_380849


namespace NUMINAMATH_CALUDE_proposition_and_equivalents_l3808_380875

def IsDecreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem proposition_and_equivalents (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n ↔ IsDecreasing a) ∧
  (IsDecreasing a ↔ ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) ∧
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 ≥ a n ↔ ¬IsDecreasing a) ∧
  (¬IsDecreasing a ↔ ∀ n : ℕ+, (a n + a (n + 1)) / 2 ≥ a n) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_equivalents_l3808_380875


namespace NUMINAMATH_CALUDE_max_value_expression_l3808_380838

theorem max_value_expression (a b c d : ℝ) 
  (ha : -13.5 ≤ a ∧ a ≤ 13.5)
  (hb : -13.5 ≤ b ∧ b ≤ 13.5)
  (hc : -13.5 ≤ c ∧ c ≤ 13.5)
  (hd : -13.5 ≤ d ∧ d ≤ 13.5) :
  (∀ x y z w, -13.5 ≤ x ∧ x ≤ 13.5 → 
              -13.5 ≤ y ∧ y ≤ 13.5 → 
              -13.5 ≤ z ∧ z ≤ 13.5 → 
              -13.5 ≤ w ∧ w ≤ 13.5 → 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 756) ∧ 
  (∃ x y z w, -13.5 ≤ x ∧ x ≤ 13.5 ∧ 
              -13.5 ≤ y ∧ y ≤ 13.5 ∧ 
              -13.5 ≤ z ∧ z ≤ 13.5 ∧ 
              -13.5 ≤ w ∧ w ≤ 13.5 ∧ 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 756) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3808_380838


namespace NUMINAMATH_CALUDE_limit_inequality_l3808_380800

theorem limit_inequality : 12.37 * (3/2 - 1/3) > Real.cos (π/10) := by
  sorry

end NUMINAMATH_CALUDE_limit_inequality_l3808_380800


namespace NUMINAMATH_CALUDE_complex_imaginary_condition_l3808_380847

theorem complex_imaginary_condition (m : ℝ) :
  (∃ z : ℂ, z = Complex.mk (3*m - 2) (m - 1) ∧ z.re = 0) → m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_imaginary_condition_l3808_380847


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3808_380853

theorem fraction_sum_equality : (20 : ℚ) / 24 + (20 : ℚ) / 25 = 49 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3808_380853


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3808_380812

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume : ℝ := cube_side_length ^ 3
  let clear_tetrahedron_volume : ℝ := (1 / 6) * cube_side_length ^ 3
  let colored_tetrahedron_volume : ℝ := cube_volume - 4 * clear_tetrahedron_volume
  colored_tetrahedron_volume = 172 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3808_380812


namespace NUMINAMATH_CALUDE_weights_division_l3808_380841

theorem weights_division (n : ℕ) : 
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔ 
  (n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2)) :=
sorry

end NUMINAMATH_CALUDE_weights_division_l3808_380841


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l3808_380824

theorem right_triangle_side_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensure positive lengths
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  a = 6 →                  -- Given side length
  c = 10 →                 -- Given hypotenuse length
  b = 8 := by              -- Prove other side length is 8
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l3808_380824


namespace NUMINAMATH_CALUDE_emails_left_theorem_l3808_380892

/-- Given an initial number of emails, calculate the number of emails left in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
def emails_left_in_inbox (initial_emails : ℕ) : ℕ :=
  let after_trash := initial_emails / 2
  let to_work_folder := (after_trash * 40) / 100
  after_trash - to_work_folder

/-- Theorem stating that given 400 initial emails, 120 emails are left in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
theorem emails_left_theorem : emails_left_in_inbox 400 = 120 := by
  sorry

#eval emails_left_in_inbox 400

end NUMINAMATH_CALUDE_emails_left_theorem_l3808_380892


namespace NUMINAMATH_CALUDE_triangle_tangent_relations_l3808_380888

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- State the theorem
theorem triangle_tangent_relations (t : Triangle) 
  (h1 : cos (2 * t.C) = 1 - 8 * t.b^2 / t.a^2)
  (h2 : tan t.B = 8/15) :
  (1 / tan t.A + 1 / tan t.C = 1/2) ∧ 
  (tan t.A = 4) ∧ 
  (tan t.C = 4) := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_relations_l3808_380888


namespace NUMINAMATH_CALUDE_al_mass_percentage_l3808_380801

theorem al_mass_percentage (mass_percentage : ℝ) (h : mass_percentage = 20.45) :
  mass_percentage = 20.45 := by
sorry

end NUMINAMATH_CALUDE_al_mass_percentage_l3808_380801


namespace NUMINAMATH_CALUDE_sin_cos_shift_l3808_380837

open Real

theorem sin_cos_shift (x : ℝ) : 
  sin (2 * x) + Real.sqrt 3 * cos (2 * x) = 2 * sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l3808_380837


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3808_380856

/-- A binary operation on nonzero real numbers satisfying certain properties -/
def diamond (a b : ℝ) : ℝ := sorry

/-- The binary operation satisfies a ♢ (b ♢ c) = (a ♢ b) · c -/
axiom diamond_assoc (a b c : ℝ) : a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c

/-- The binary operation satisfies a ♢ a = 1 -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- The equation 1008 ♢ (12 ♢ x) = 50 is satisfied when x = 25/42 -/
theorem diamond_equation_solution :
  1008 ≠ 0 → 12 ≠ 0 → (25 : ℝ) / 42 ≠ 0 → diamond 1008 (diamond 12 ((25 : ℝ) / 42)) = 50 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3808_380856


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3808_380827

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 1, b = √2, and A = 30°, then B = 45° or B = 135°. -/
theorem triangle_angle_calculation (a b c A B C : Real) : 
  a = 1 → b = Real.sqrt 2 → A = π / 6 → 
  (B = π / 4 ∨ B = 3 * π / 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_calculation_l3808_380827


namespace NUMINAMATH_CALUDE_circle_area_in_square_l3808_380883

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 - 16*y + 65 = 0

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | 3 ≤ p.1 ∧ p.1 ≤ 8 ∧ 8 ≤ p.2 ∧ p.2 ≤ 13}

-- Theorem statement
theorem circle_area_in_square :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (∀ x y, circle_equation x y → (x, y) ∈ square) ∧
    (π * radius^2 = 24 * π) :=
sorry

end NUMINAMATH_CALUDE_circle_area_in_square_l3808_380883


namespace NUMINAMATH_CALUDE_book_price_l3808_380843

def original_price : ℝ → Prop :=
  fun price =>
    let first_discount := price * (1 - 1/5)
    let second_discount := first_discount * (1 - 1/5)
    second_discount = 32

theorem book_price : original_price 50 := by
  sorry

end NUMINAMATH_CALUDE_book_price_l3808_380843


namespace NUMINAMATH_CALUDE_regular_16gon_symmetry_sum_l3808_380862

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_16gon_symmetry_sum :
  ∀ (p : RegularPolygon 16),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_regular_16gon_symmetry_sum_l3808_380862


namespace NUMINAMATH_CALUDE_triangle_point_distance_l3808_380804

-- Define the triangle and points
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 4 ∧ d B C = 5 ∧ d C A = 6

def OnRay (A B D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

def CircumCircle (A B C P : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A P = d B P ∧ d B P = d C P

-- State the theorem
theorem triangle_point_distance (A B C D E F : ℝ × ℝ) :
  Triangle A B C →
  OnRay A B D →
  OnRay A B E →
  CircumCircle A C D F →
  CircumCircle E B C F →
  F ≠ C →
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d D F = 2 →
  d E F = 7 →
  d B E = (5 + 21 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_distance_l3808_380804


namespace NUMINAMATH_CALUDE_product_equals_sum_exists_percentage_calculation_l3808_380844

-- Problem 1
theorem product_equals_sum_exists : ∃ (a b c : ℤ), a * b * c = a + b + c := by
  sorry

-- Problem 2
theorem percentage_calculation : (12.5 / 100) * 44 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_exists_percentage_calculation_l3808_380844


namespace NUMINAMATH_CALUDE_regular_tetrahedron_height_l3808_380873

/-- Given a regular tetrahedron with an inscribed sphere, 
    prove that its height is 4 times the radius of the inscribed sphere -/
theorem regular_tetrahedron_height (h r : ℝ) : h = 4 * r :=
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_height_l3808_380873


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3808_380891

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3808_380891


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_l3808_380816

/-- Given vectors a and b in ℝ², if a is perpendicular to (2a - b), then the second component of b is 14. -/
theorem perpendicular_vectors_imply_k (a b : ℝ × ℝ) (h : a.1 = 2 ∧ a.2 = 1 ∧ b.1 = -2) :
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2) = 0) → b.2 = 14 := by
  sorry

#check perpendicular_vectors_imply_k

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_l3808_380816


namespace NUMINAMATH_CALUDE_B_profit_share_l3808_380860

def investment_A : ℕ := 8000
def investment_B : ℕ := 10000
def investment_C : ℕ := 12000
def profit_difference_AC : ℕ := 560

theorem B_profit_share :
  let total_investment := investment_A + investment_B + investment_C
  let profit_ratio_A := investment_A / total_investment
  let profit_ratio_B := investment_B / total_investment
  let profit_ratio_C := investment_C / total_investment
  let total_profit := profit_difference_AC * total_investment / (profit_ratio_C - profit_ratio_A)
  profit_ratio_B * total_profit = 1400 := by sorry

end NUMINAMATH_CALUDE_B_profit_share_l3808_380860


namespace NUMINAMATH_CALUDE_vanya_can_always_win_l3808_380877

/-- Represents a sequence of signs (+1 for "+", -1 for "-") -/
def SignSequence := List Int

/-- Represents a move that swaps two adjacent signs -/
def Move := Nat

/-- Applies a move to a sign sequence -/
def applyMove (seq : SignSequence) (m : Move) : SignSequence :=
  sorry

/-- Evaluates the expression given a sign sequence -/
def evaluateExpression (seq : SignSequence) : Int :=
  sorry

/-- Checks if a number is divisible by 7 -/
def isDivisibleBy7 (n : Int) : Prop :=
  n % 7 = 0

/-- The main theorem: Vanya can always achieve a sum divisible by 7 -/
theorem vanya_can_always_win (initialSeq : SignSequence) :
  ∃ (moves : List Move), isDivisibleBy7 (evaluateExpression (moves.foldl applyMove initialSeq)) :=
sorry

end NUMINAMATH_CALUDE_vanya_can_always_win_l3808_380877


namespace NUMINAMATH_CALUDE_road_trip_gas_cost_l3808_380855

/-- Calculates the total cost of filling a car's gas tank at multiple stations -/
def total_gas_cost (tank_capacity : ℝ) (gas_prices : List ℝ) : ℝ :=
  (gas_prices.map (· * tank_capacity)).sum

/-- Proves that the total cost of filling a 12-gallon tank at 4 stations with given prices is $180 -/
theorem road_trip_gas_cost : 
  let tank_capacity : ℝ := 12
  let gas_prices : List ℝ := [3, 3.5, 4, 4.5]
  total_gas_cost tank_capacity gas_prices = 180 := by
  sorry

#eval total_gas_cost 12 [3, 3.5, 4, 4.5]

end NUMINAMATH_CALUDE_road_trip_gas_cost_l3808_380855


namespace NUMINAMATH_CALUDE_triangle_angle_and_vector_dot_product_l3808_380899

theorem triangle_angle_and_vector_dot_product 
  (A B C : ℝ) (a b c : ℝ) (k : ℝ) :
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  (2 * a - c) * Real.cos B = b * Real.cos C ∧
  k > 1 ∧
  (∀ t : ℝ, 0 < t ∧ t ≤ 1 → -2 * t^2 + 4 * k * t + 1 ≤ 5) ∧
  -2 + 4 * k + 1 = 5 →
  B = π / 3 ∧ k = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_and_vector_dot_product_l3808_380899


namespace NUMINAMATH_CALUDE_hyperbola_right_directrix_l3808_380836

/-- Given a parabola and a hyperbola with a shared focus, this theorem proves 
    the equation of the right directrix of the hyperbola. -/
theorem hyperbola_right_directrix 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_focus : ∀ x y : ℝ, y^2 = 8*x → x^2/a^2 - y^2/3 = 1 → x = 2 ∧ y = 0) :
  ∃ x : ℝ, x = 1/2 ∧ 
    ∀ y : ℝ, (∃ t : ℝ, t^2/a^2 - y^2/3 = 1 ∧ t > x) → 
      x = a^2 / (2 * (a^2 + 3)^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_right_directrix_l3808_380836


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l3808_380805

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Define the tangent line segment
def is_tangent (P Q : ℝ × ℝ) : Prop :=
  C1 P.1 P.2 ∧ C2 Q.1 Q.2 ∧ 
  ∀ R : ℝ × ℝ, (C1 R.1 R.2 ∨ C2 R.1 R.2) → 
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) ≥ 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent P Q ∧ 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 15 ∧
    ∀ P' Q' : ℝ × ℝ, is_tangent P' Q' → 
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l3808_380805


namespace NUMINAMATH_CALUDE_ratio_sum_difference_l3808_380884

theorem ratio_sum_difference (a b c : ℝ) : 
  (a : ℝ) / 1 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 6 →
  a + b + c = 30 →
  c - b - a = 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_difference_l3808_380884


namespace NUMINAMATH_CALUDE_tree_leaves_theorem_l3808_380831

/-- Calculates the number of leaves remaining on a tree after three weeks of shedding --/
def leaves_remaining (initial_leaves : ℕ) : ℕ :=
  let first_week_remaining := initial_leaves - (2 * initial_leaves / 5)
  let second_week_shed := (40 * first_week_remaining) / 100
  let second_week_remaining := first_week_remaining - second_week_shed
  let third_week_shed := (3 * second_week_shed) / 4
  second_week_remaining - third_week_shed

/-- Theorem stating that a tree with 1000 initial leaves will have 180 leaves remaining after three weeks of shedding --/
theorem tree_leaves_theorem : leaves_remaining 1000 = 180 := by
  sorry

end NUMINAMATH_CALUDE_tree_leaves_theorem_l3808_380831


namespace NUMINAMATH_CALUDE_pen_distribution_l3808_380821

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem pen_distribution (num_pencils : ℕ) (num_pens : ℕ) 
  (h1 : num_pencils = 1203)
  (h2 : is_prime num_pencils)
  (h3 : ∀ (students : ℕ), students > 1 → ¬(num_pencils % students = 0 ∧ num_pens % students = 0)) :
  ∃ (n : ℕ), num_pens = n :=
sorry

end NUMINAMATH_CALUDE_pen_distribution_l3808_380821


namespace NUMINAMATH_CALUDE_problem_solution_l3808_380893

def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| + |m - x|

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4)) ∧
  (∀ m : ℝ, (∀ x : ℝ, f m x ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3808_380893


namespace NUMINAMATH_CALUDE_circle_center_sum_l3808_380834

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (x + y = -1) := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3808_380834


namespace NUMINAMATH_CALUDE_smallest_area_increase_l3808_380866

theorem smallest_area_increase (l w : ℕ) (hl : l > 0) (hw : w > 0) :
  ∃ (x : ℕ), x > 0 ∧ (w + 1) * (l - 1) - w * l = x ∧
  ∀ (y : ℕ), y > 0 → (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (b + 1) * (a - 1) - b * a = y) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_increase_l3808_380866


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3808_380845

def N : ℕ := 48 * 48 * 55 * 125 * 81

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3808_380845


namespace NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_13_l3808_380851

theorem number_divided_by_6_multiplied_by_12_equals_13 : ∃ x : ℚ, (x / 6) * 12 = 13 ∧ x = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_13_l3808_380851


namespace NUMINAMATH_CALUDE_cube_lateral_surface_area_l3808_380815

/-- The lateral surface area of a cube with side length 12 meters is 576 square meters. -/
theorem cube_lateral_surface_area :
  let side_length : ℝ := 12
  let lateral_surface_area : ℝ := 4 * side_length * side_length
  lateral_surface_area = 576 := by sorry

end NUMINAMATH_CALUDE_cube_lateral_surface_area_l3808_380815


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3808_380810

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3808_380810


namespace NUMINAMATH_CALUDE_smallest_number_with_million_divisors_l3808_380848

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_million_divisors (N : ℕ) :
  divisor_count N = 1000000 →
  N ≥ 2^9 * (3 * 5 * 7 * 11 * 13)^4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_million_divisors_l3808_380848


namespace NUMINAMATH_CALUDE_root_conditions_imply_a_range_l3808_380885

theorem root_conditions_imply_a_range (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ > 1 ∧ x₂ < 1 ∧ 
   x₁^2 + a*x₁ + a^2 - a - 2 = 0 ∧
   x₂^2 + a*x₂ + a^2 - a - 2 = 0) →
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_a_range_l3808_380885


namespace NUMINAMATH_CALUDE_max_value_parabola_l3808_380864

theorem max_value_parabola :
  (∀ x : ℝ, -x^2 + 5 ≤ 5) ∧ (∃ x : ℝ, -x^2 + 5 = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_parabola_l3808_380864


namespace NUMINAMATH_CALUDE_larger_number_problem_l3808_380887

theorem larger_number_problem (x y : ℤ) 
  (sum_is_62 : x + y = 62) 
  (y_is_larger : y = x + 12) : 
  y = 37 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3808_380887


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_9_18_36_l3808_380835

theorem gcf_lcm_sum_9_18_36 : 
  let A := Nat.gcd 9 (Nat.gcd 18 36)
  let B := Nat.lcm 9 (Nat.lcm 18 36)
  A + B = 45 := by sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_9_18_36_l3808_380835


namespace NUMINAMATH_CALUDE_square_area_from_octagon_l3808_380817

theorem square_area_from_octagon (side_length : ℝ) (octagon_area : ℝ) : 
  side_length > 0 →
  octagon_area = 7 * (side_length / 3)^2 →
  octagon_area = 105 →
  side_length^2 = 135 :=
by
  sorry

#check square_area_from_octagon

end NUMINAMATH_CALUDE_square_area_from_octagon_l3808_380817


namespace NUMINAMATH_CALUDE_diesel_cost_approximation_l3808_380874

/-- Calculates the approximate average cost of diesel per litre over three years -/
def average_diesel_cost (price1 price2 price3 yearly_spend : ℚ) : ℚ :=
  let litres1 := yearly_spend / price1
  let litres2 := yearly_spend / price2
  let litres3 := yearly_spend / price3
  let total_litres := litres1 + litres2 + litres3
  let total_spent := 3 * yearly_spend
  total_spent / total_litres

/-- Theorem stating that the average diesel cost is approximately 8.98 given the specified conditions -/
theorem diesel_cost_approximation :
  let price1 : ℚ := 8.5
  let price2 : ℚ := 9
  let price3 : ℚ := 9.5
  let yearly_spend : ℚ := 5000
  let result := average_diesel_cost price1 price2 price3 yearly_spend
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |result - 8.98| < ε :=
sorry

end NUMINAMATH_CALUDE_diesel_cost_approximation_l3808_380874


namespace NUMINAMATH_CALUDE_triangle_extension_l3808_380806

/-- Triangle extension theorem -/
theorem triangle_extension (n : ℕ) (a b c t S : ℝ) 
  (h_n : n > 0)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_area : t > 0)
  (h_S : S = a^2 + b^2 + c^2)
  (t_i : Fin (n-1) → ℝ)
  (S_i : Fin (n-1) → ℝ)
  (h_t_i : ∀ i, t_i i > 0)
  (h_S_i : ∀ i, S_i i > 0) :
  (∃ k : ℝ, 
    (S + (Finset.sum Finset.univ S_i) = n^3 * S) ∧ 
    (t + (Finset.sum Finset.univ t_i) = n^3 * t) ∧ 
    (∀ i : Fin (n-1), S_i i / t_i i = k) ∧ 
    (S / t = k)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_extension_l3808_380806


namespace NUMINAMATH_CALUDE_table_height_proof_l3808_380880

/-- Given two configurations of stacked blocks on a table, prove that the table height is 34 inches -/
theorem table_height_proof (r s b : ℝ) (hr : r = 40) (hs : s = 34) (hb : b = 6) :
  ∃ (h l w : ℝ), h = 34 ∧ l + h - w = r ∧ w + h - l + b = s := by
  sorry

end NUMINAMATH_CALUDE_table_height_proof_l3808_380880


namespace NUMINAMATH_CALUDE_external_roads_different_colors_l3808_380825

/-- Represents a city with colored streets and intersections -/
structure ColoredCity where
  /-- The number of intersections in the city -/
  n : ℕ
  /-- The number of external roads of each color -/
  c : Fin 3 → ℕ
  /-- The total number of half-streets of each color -/
  half_streets : Fin 3 → ℕ
  /-- Each intersection connects exactly three streets of different colors -/
  intersection_property : ∀ i : Fin 3, half_streets i = n + c i
  /-- The sum of external roads is exactly three -/
  external_roads_sum : c 0 + c 1 + c 2 = 3

/-- Theorem stating that the three external roads have different colors -/
theorem external_roads_different_colors (city : ColoredCity) : 
  city.c 0 = 1 ∧ city.c 1 = 1 ∧ city.c 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_external_roads_different_colors_l3808_380825


namespace NUMINAMATH_CALUDE_currency_comparisons_l3808_380882

-- Define the conversion rate from jiao to yuan
def jiao_to_yuan (jiao : ℚ) : ℚ := jiao / 10

-- Define the theorem
theorem currency_comparisons :
  (2.3 < 3.2) ∧
  (10 > 9.9) ∧
  (1 + jiao_to_yuan 6 = 1.6) ∧
  (15 * 4 < 14 * 5) :=
by sorry

end NUMINAMATH_CALUDE_currency_comparisons_l3808_380882


namespace NUMINAMATH_CALUDE_total_absent_students_l3808_380861

def total_students : ℕ := 280

def absent_third_day (total : ℕ) : ℕ := total / 7

def absent_second_day (absent_third : ℕ) : ℕ := 2 * absent_third

def present_first_day (total : ℕ) (absent_second : ℕ) : ℕ := total - absent_second

def absent_first_day (total : ℕ) (present_first : ℕ) : ℕ := total - present_first

theorem total_absent_students :
  let absent_third := absent_third_day total_students
  let absent_second := absent_second_day absent_third
  let present_first := present_first_day total_students absent_second
  let absent_first := absent_first_day total_students present_first
  absent_first + absent_second + absent_third = 200 := by sorry

end NUMINAMATH_CALUDE_total_absent_students_l3808_380861


namespace NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l3808_380858

theorem goose_egg_hatch_fraction (total_eggs : ℕ) (survived_year : ℕ) 
  (h1 : total_eggs = 550)
  (h2 : survived_year = 110)
  (h3 : ∀ x : ℚ, x * total_eggs * (3/4 : ℚ) * (2/5 : ℚ) = survived_year → x = 2/3) :
  ∃ x : ℚ, x * total_eggs = (total_eggs : ℚ) * (2/3 : ℚ) := by sorry

end NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l3808_380858


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l3808_380890

/-- Systematic sampling function that returns the nth sample number given the start number and interval -/
def systematicSample (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * interval

/-- Theorem: In a systematic sampling of 56 students with sample size 4,
    if 6, 20, and 48 are in the sample, then 34 is the fourth sample number -/
theorem systematic_sampling_problem :
  let totalStudents : ℕ := 56
  let sampleSize : ℕ := 4
  let interval : ℕ := totalStudents / sampleSize
  let sample1 : ℕ := 6
  let sample2 : ℕ := 20
  let sample3 : ℕ := 48
  let sample4 : ℕ := 34
  (systematicSample sample1 interval 1 = sample1) ∧
  (systematicSample sample1 interval 2 = sample2) ∧
  (systematicSample sample1 interval 3 = sample3) ∧
  (systematicSample sample1 interval 4 = sample4) :=
by
  sorry

#check systematic_sampling_problem

end NUMINAMATH_CALUDE_systematic_sampling_problem_l3808_380890


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l3808_380802

theorem max_value_of_linear_combination (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  ∃ (M : ℝ), M = 3 * Real.sqrt 14 ∧ 
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 9 → a + 2*b + 3*c ≤ M) ∧
  (∃ (u v w : ℝ), u^2 + v^2 + w^2 = 9 ∧ u + 2*v + 3*w = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l3808_380802


namespace NUMINAMATH_CALUDE_square_1849_product_l3808_380820

theorem square_1849_product (y : ℤ) (h : y^2 = 1849) : (y+2)*(y-2) = 1845 := by
  sorry

end NUMINAMATH_CALUDE_square_1849_product_l3808_380820


namespace NUMINAMATH_CALUDE_final_solid_properties_l3808_380830

/-- Represents a solid shape with faces, edges, and vertices -/
structure Solid where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Represents a pyramid attached to a face -/
structure Pyramid where
  base_edges : ℕ

/-- Attaches a pyramid to a solid, updating its properties -/
def attach_pyramid (s : Solid) (p : Pyramid) : Solid :=
  { faces := s.faces + p.base_edges - 1
  , edges := s.edges + p.base_edges
  , vertices := s.vertices + 1 }

/-- The initial triangular prism -/
def initial_prism : Solid :=
  { faces := 5, edges := 9, vertices := 6 }

/-- Pyramid attached to triangular face -/
def triangular_pyramid : Pyramid :=
  { base_edges := 3 }

/-- Pyramid attached to quadrilateral face -/
def quadrilateral_pyramid : Pyramid :=
  { base_edges := 4 }

theorem final_solid_properties :
  let s1 := attach_pyramid initial_prism triangular_pyramid
  let final_solid := attach_pyramid s1 quadrilateral_pyramid
  final_solid.faces = 10 ∧
  final_solid.edges = 16 ∧
  final_solid.vertices = 8 ∧
  final_solid.faces + final_solid.edges + final_solid.vertices = 34 := by
  sorry


end NUMINAMATH_CALUDE_final_solid_properties_l3808_380830


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3808_380811

def expression : ℕ := 18^3 + 15^4 - 3^7

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p ∧ p = 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3808_380811


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3808_380895

/-- The curve y = x^2 - 3x -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem tangent_parallel_to_x_axis :
  let P : ℝ × ℝ := (3/2, -9/4)
  (f P.1 = P.2) ∧ (f' P.1 = 0) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3808_380895


namespace NUMINAMATH_CALUDE_magpies_in_park_l3808_380819

/-- The number of magpies in a park with blackbirds and magpies -/
theorem magpies_in_park (trees : ℕ) (blackbirds_per_tree : ℕ) (total_birds : ℕ) 
  (h1 : trees = 7)
  (h2 : blackbirds_per_tree = 3)
  (h3 : total_birds = 34) :
  total_birds - trees * blackbirds_per_tree = 13 := by
  sorry

end NUMINAMATH_CALUDE_magpies_in_park_l3808_380819


namespace NUMINAMATH_CALUDE_unique_solution_x2024_y3_3y_l3808_380897

theorem unique_solution_x2024_y3_3y :
  ∀ x y : ℤ, x^2024 + y^3 = 3*y ↔ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x2024_y3_3y_l3808_380897
