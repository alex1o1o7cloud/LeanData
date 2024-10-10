import Mathlib

namespace unique_positive_solution_l1419_141926

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 7 = 5 / (x - 7) := by
  sorry

end unique_positive_solution_l1419_141926


namespace complex_purely_imaginary_l1419_141909

theorem complex_purely_imaginary (m : ℝ) : 
  let z : ℂ := m + 2*I
  (∃ (y : ℝ), (2 + I) * z = y * I) → m = 1 := by
sorry

end complex_purely_imaginary_l1419_141909


namespace partially_symmetric_iff_l1419_141965

/-- A function is partially symmetric if it satisfies three specific conditions. -/
def PartiallySymmetric (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ x : ℝ, x ≠ 0 → x * (deriv f x) > 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ abs x₁ < abs x₂ → f x₁ < f x₂)

/-- Theorem: A function is partially symmetric if and only if it satisfies the three conditions. -/
theorem partially_symmetric_iff (f : ℝ → ℝ) :
  PartiallySymmetric f ↔
    (f 0 = 0) ∧
    (∀ x : ℝ, x ≠ 0 → x * (deriv f x) > 0) ∧
    (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ abs x₁ < abs x₂ → f x₁ < f x₂) :=
by
  sorry

end partially_symmetric_iff_l1419_141965


namespace quadratic_root_in_unit_interval_l1419_141929

/-- Given real numbers a, b, c, and a positive number m satisfying the condition,
    the quadratic equation has a root between 0 and 1. -/
theorem quadratic_root_in_unit_interval (a b c m : ℝ) (hm : m > 0) 
    (h : a / (m + 2) + b / (m + 1) + c / m = 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end quadratic_root_in_unit_interval_l1419_141929


namespace simplify_rational_expression_l1419_141970

theorem simplify_rational_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  ((x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4)) / ((x - 4) / (x^2 - 2*x)) = 1 / (x - 2) :=
by sorry

end simplify_rational_expression_l1419_141970


namespace rational_solution_exists_l1419_141993

theorem rational_solution_exists : ∃ (a b : ℚ), (a ≠ 0) ∧ (a + b ≠ 0) ∧ ((a + b) / a + a / (a + b) = b) := by
  sorry

end rational_solution_exists_l1419_141993


namespace max_value_a_l1419_141978

theorem max_value_a (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a : ℝ, a ≤ 1/x + 9/y) → (∃ a : ℝ, a = 16 ∧ ∀ b : ℝ, b ≤ 1/x + 9/y → b ≤ a) :=
by sorry

end max_value_a_l1419_141978


namespace multiplication_addition_equality_l1419_141971

theorem multiplication_addition_equality : 26 * 43 + 57 * 26 = 2600 := by
  sorry

end multiplication_addition_equality_l1419_141971


namespace square_area_l1419_141934

/-- The area of a square with side length 13 cm is 169 square centimeters. -/
theorem square_area (side_length : ℝ) (h : side_length = 13) : side_length ^ 2 = 169 := by
  sorry

end square_area_l1419_141934


namespace tan_plus_3sin_30_deg_l1419_141942

theorem tan_plus_3sin_30_deg :
  Real.tan (30 * π / 180) + 3 * Real.sin (30 * π / 180) = (1 + 3 * Real.sqrt 3) / 2 :=
by sorry

end tan_plus_3sin_30_deg_l1419_141942


namespace prob_sum_less_than_one_l1419_141944

theorem prob_sum_less_than_one (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) : 
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 := by
  sorry

end prob_sum_less_than_one_l1419_141944


namespace airplane_passengers_l1419_141905

theorem airplane_passengers (total : ℕ) (children : ℕ) : 
  total = 80 → children = 20 → ∃ (men women : ℕ), 
    men = women ∧ 
    men + women + children = total ∧ 
    men = 30 := by
  sorry

end airplane_passengers_l1419_141905


namespace fractional_equation_solution_l1419_141901

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 3 ∧ x ≠ 1 ∧ (x / (x - 3) = (x + 1) / (x - 1)) ↔ x = -3 :=
by sorry

end fractional_equation_solution_l1419_141901


namespace triangle_abc_proof_l1419_141979

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  a > c →
  b = 3 →
  (a * c * (1 / 3) = 2) →  -- Equivalent to BA · BC = 2 and cos B = 1/3
  a + c = 5 →              -- From the solution, but derivable from given conditions
  (a = 3 ∧ c = 2) ∧ (Real.cos C = 7 / 9) := by
  sorry

end triangle_abc_proof_l1419_141979


namespace E_equals_F_l1419_141935

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_equals_F : E = F := by
  sorry

end E_equals_F_l1419_141935


namespace point_7_8_numbered_72_l1419_141999

def first_quadrant_numbering (x y : ℕ) : ℕ :=
  sorry

theorem point_7_8_numbered_72 :
  first_quadrant_numbering 7 8 = 72 := by
  sorry

end point_7_8_numbered_72_l1419_141999


namespace equation_properties_l1419_141966

def p (x : ℝ) := x^4 - x^3 - 1

theorem equation_properties :
  (∃ (r₁ r₂ : ℝ), p r₁ = 0 ∧ p r₂ = 0 ∧ r₁ ≠ r₂ ∧
    (∀ (r : ℝ), p r = 0 → r = r₁ ∨ r = r₂)) ∧
  (∀ (r₁ r₂ : ℝ), p r₁ = 0 → p r₂ = 0 → r₁ + r₂ > 6/11) ∧
  (∀ (r₁ r₂ : ℝ), p r₁ = 0 → p r₂ = 0 → r₁ * r₂ < -11/10) :=
by sorry

end equation_properties_l1419_141966


namespace packing_theorem_l1419_141985

/-- Represents the types of boxes that can be packed. -/
inductive BoxType
  | Large
  | Medium
  | Small

/-- Represents the types of packing tape. -/
inductive TapeType
  | A
  | B

/-- Calculates the amount of tape needed for a given box type. -/
def tapeNeeded (b : BoxType) : ℕ :=
  match b with
  | BoxType.Large => 5
  | BoxType.Medium => 3
  | BoxType.Small => 2

/-- Calculates the total tape used for packing a list of boxes. -/
def totalTapeUsed (boxes : List (BoxType × ℕ)) : ℕ :=
  boxes.foldl (fun acc (b, n) => acc + n * tapeNeeded b) 0

/-- Represents the packing scenario for Debbie and Mike. -/
structure PackingScenario where
  debbieBoxes : List (BoxType × ℕ)
  mikeBoxes : List (BoxType × ℕ)
  tapeARollLength : ℕ
  tapeBRollLength : ℕ

/-- Calculates the remaining tape for Debbie and Mike. -/
def remainingTape (scenario : PackingScenario) : TapeType → ℕ
  | TapeType.A => scenario.tapeARollLength - totalTapeUsed scenario.debbieBoxes
  | TapeType.B => 
      let usedTapeB := totalTapeUsed scenario.mikeBoxes
      scenario.tapeBRollLength - (usedTapeB % scenario.tapeBRollLength)

/-- The main theorem stating the remaining tape for Debbie and Mike. -/
theorem packing_theorem (scenario : PackingScenario) 
    (h1 : scenario.debbieBoxes = [(BoxType.Large, 2), (BoxType.Medium, 8), (BoxType.Small, 5)])
    (h2 : scenario.mikeBoxes = [(BoxType.Large, 3), (BoxType.Medium, 6), (BoxType.Small, 10)])
    (h3 : scenario.tapeARollLength = 50)
    (h4 : scenario.tapeBRollLength = 40) :
    remainingTape scenario TapeType.A = 6 ∧ remainingTape scenario TapeType.B = 27 := by
  sorry

end packing_theorem_l1419_141985


namespace factoring_expression_l1419_141976

theorem factoring_expression (x y : ℝ) : 
  72 * x^4 * y^2 - 180 * x^8 * y^5 = 36 * x^4 * y^2 * (2 - 5 * x^4 * y^3) := by
  sorry

end factoring_expression_l1419_141976


namespace gingerbreads_in_unknown_tray_is_20_l1419_141904

/-- The number of gingerbreads in each of the first four trays -/
def gingerbreads_per_tray : ℕ := 25

/-- The number of trays with a known number of gingerbreads -/
def known_trays : ℕ := 4

/-- The number of trays with an unknown number of gingerbreads -/
def unknown_trays : ℕ := 3

/-- The total number of gingerbreads baked -/
def total_gingerbreads : ℕ := 160

/-- The number of gingerbreads in each of the unknown trays -/
def gingerbreads_in_unknown_tray : ℕ := (total_gingerbreads - known_trays * gingerbreads_per_tray) / unknown_trays

theorem gingerbreads_in_unknown_tray_is_20 :
  gingerbreads_in_unknown_tray = 20 := by
  sorry

end gingerbreads_in_unknown_tray_is_20_l1419_141904


namespace probability_exactly_one_instrument_l1419_141947

/-- Given a group of people, calculate the probability that a randomly selected person plays exactly one instrument. -/
theorem probability_exactly_one_instrument 
  (total_people : ℕ) 
  (at_least_one_fraction : ℚ) 
  (two_or_more : ℕ) 
  (h1 : total_people = 800)
  (h2 : at_least_one_fraction = 1 / 5)
  (h3 : two_or_more = 128) :
  (total_people : ℚ) * at_least_one_fraction - two_or_more / total_people = 1 / 25 := by
sorry

end probability_exactly_one_instrument_l1419_141947


namespace unique_base_perfect_square_l1419_141938

theorem unique_base_perfect_square : 
  ∃! n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end unique_base_perfect_square_l1419_141938


namespace lost_to_remaining_ratio_l1419_141922

def initial_amount : ℕ := 5000
def motorcycle_cost : ℕ := 2800
def final_amount : ℕ := 825

def amount_after_motorcycle : ℕ := initial_amount - motorcycle_cost
def concert_ticket_cost : ℕ := amount_after_motorcycle / 2
def amount_after_concert : ℕ := amount_after_motorcycle - concert_ticket_cost
def amount_lost : ℕ := amount_after_concert - final_amount

theorem lost_to_remaining_ratio :
  (amount_lost : ℚ) / (amount_after_concert : ℚ) = 1 / 4 := by sorry

end lost_to_remaining_ratio_l1419_141922


namespace batsman_average_l1419_141975

/-- Represents a batsman's performance over multiple innings -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  latestScore : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman after their latest innings -/
def calculateAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.latestScore) / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after the 12th innings is 58 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.latestScore = 80)
  (h3 : calculateAverage b = calculateAverage { b with innings := b.innings - 1 } + 2) :
  calculateAverage b = 58 := by
  sorry

end batsman_average_l1419_141975


namespace sherman_weekly_driving_time_l1419_141967

/-- Calculates the total weekly driving time for Sherman given his commute and weekend driving schedules. -/
theorem sherman_weekly_driving_time 
  (weekday_commute_minutes : ℕ) -- Daily commute time (round trip) in minutes
  (weekdays : ℕ) -- Number of weekdays
  (weekend_driving_hours : ℕ) -- Daily weekend driving time in hours
  (weekend_days : ℕ) -- Number of weekend days
  (h1 : weekday_commute_minutes = 60) -- 30 minutes to office + 30 minutes back home
  (h2 : weekdays = 5) -- 5 weekdays in a week
  (h3 : weekend_driving_hours = 2) -- 2 hours of driving each weekend day
  (h4 : weekend_days = 2) -- 2 days in a weekend
  : (weekday_commute_minutes * weekdays) / 60 + weekend_driving_hours * weekend_days = 9 :=
by sorry

end sherman_weekly_driving_time_l1419_141967


namespace boys_participation_fraction_l1419_141972

-- Define the total number of students
def total_students : ℕ := 800

-- Define the number of participating students
def participating_students : ℕ := 550

-- Define the number of participating girls
def participating_girls : ℕ := 150

-- Define the fraction of girls who participated
def girls_participation_fraction : ℚ := 3/4

-- Theorem to prove
theorem boys_participation_fraction :
  let total_girls : ℕ := participating_girls * 4 / 3
  let total_boys : ℕ := total_students - total_girls
  let participating_boys : ℕ := participating_students - participating_girls
  (participating_boys : ℚ) / total_boys = 2/3 :=
by sorry

end boys_participation_fraction_l1419_141972


namespace maximize_x_cubed_y_fourth_l1419_141921

theorem maximize_x_cubed_y_fourth :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 27 →
  x^3 * y^4 ≤ (81/7)^3 * (108/7)^4 :=
by sorry

end maximize_x_cubed_y_fourth_l1419_141921


namespace sum_equals_336_l1419_141939

theorem sum_equals_336 : 237 + 45 + 36 + 18 = 336 := by
  sorry

end sum_equals_336_l1419_141939


namespace lee_soccer_game_probability_l1419_141959

theorem lee_soccer_game_probability (p : ℚ) (h : p = 5/9) :
  1 - p = 4/9 := by sorry

end lee_soccer_game_probability_l1419_141959


namespace tan_negative_1140_degrees_l1419_141924

theorem tan_negative_1140_degrees : Real.tan (-(1140 * π / 180)) = -Real.sqrt 3 := by
  sorry

end tan_negative_1140_degrees_l1419_141924


namespace number_of_grades_l1419_141954

theorem number_of_grades (students_per_grade : ℕ) (total_students : ℕ) : 
  students_per_grade = 75 → total_students = 22800 → total_students / students_per_grade = 304 := by
  sorry

end number_of_grades_l1419_141954


namespace quadratic_sequence_exists_l1419_141992

def is_quadratic_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ n → |a i - a (i-1)| = (i : ℤ)^2

theorem quadratic_sequence_exists (h k : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = h ∧ a n = k ∧ is_quadratic_sequence a n :=
sorry

end quadratic_sequence_exists_l1419_141992


namespace min_value_of_fraction_l1419_141913

theorem min_value_of_fraction (x : ℝ) (h : x > 9) :
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end min_value_of_fraction_l1419_141913


namespace square_sum_equality_l1419_141915

theorem square_sum_equality (x y z : ℝ) 
  (h1 : x^2 + 4*y^2 + 16*z^2 = 48) 
  (h2 : x*y + 4*y*z + 2*z*x = 24) : 
  x^2 + y^2 + z^2 = 21 := by
  sorry

end square_sum_equality_l1419_141915


namespace maia_daily_requests_l1419_141962

/-- The number of client requests Maia works on each day -/
def requests_per_day : ℕ := 4

/-- The number of days Maia works -/
def days_worked : ℕ := 5

/-- The number of client requests remaining after the working period -/
def remaining_requests : ℕ := 10

/-- The number of client requests Maia gets every day -/
def daily_requests : ℕ := 6

theorem maia_daily_requests : 
  days_worked * daily_requests = days_worked * requests_per_day + remaining_requests :=
by sorry

end maia_daily_requests_l1419_141962


namespace square_equality_implies_four_l1419_141969

theorem square_equality_implies_four (x : ℝ) : (8 - x)^2 = x^2 → x = 4 := by
  sorry

end square_equality_implies_four_l1419_141969


namespace fraction_simplification_l1419_141911

theorem fraction_simplification : (27 : ℚ) / 25 * 20 / 33 * 55 / 54 = 25 / 3 := by
  sorry

end fraction_simplification_l1419_141911


namespace suit_cost_ratio_l1419_141925

theorem suit_cost_ratio (off_rack_cost tailoring_cost total_cost : ℝ) 
  (h1 : off_rack_cost = 300)
  (h2 : tailoring_cost = 200)
  (h3 : total_cost = 1400)
  (h4 : ∃ x : ℝ, total_cost = off_rack_cost + (x * off_rack_cost + tailoring_cost)) :
  ∃ x : ℝ, x = 3 ∧ total_cost = off_rack_cost + (x * off_rack_cost + tailoring_cost) :=
by sorry

end suit_cost_ratio_l1419_141925


namespace retailer_profit_percent_l1419_141995

/-- Calculates the profit percent given purchase price, overhead expenses, and selling price -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem: The profit percent for the given values is 25% -/
theorem retailer_profit_percent :
  profit_percent 225 15 300 = 25 := by
  sorry

end retailer_profit_percent_l1419_141995


namespace exists_integer_sqrt_8m_l1419_141980

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℝ).sqrt = k := by
  sorry

end exists_integer_sqrt_8m_l1419_141980


namespace equation_solution_l1419_141982

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (3 + Real.sqrt 89) / 40 ∧ 
     x₂ = (3 - Real.sqrt 89) / 40) ∧ 
    (∀ x y : ℝ, y = 3 * x → 
      (4 * y^2 + y + 5 = 2 * (8 * x^2 + y + 3) ↔ 
       (x = x₁ ∨ x = x₂))) := by
  sorry

end equation_solution_l1419_141982


namespace x_equals_n_l1419_141968

def x : ℕ → ℚ
  | 0 => 0
  | n + 1 => ((n^2 + n + 1) * x n + 1) / (n^2 + n + 1 - x n)

theorem x_equals_n (n : ℕ) : x n = n := by
  sorry

end x_equals_n_l1419_141968


namespace cubic_factorization_l1419_141914

theorem cubic_factorization (m : ℝ) : m^3 - 4*m^2 + 4*m = m*(m-2)^2 := by
  sorry

end cubic_factorization_l1419_141914


namespace triangle_problem_l1419_141943

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  (2 * (Real.cos (A / 2))^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1) →
  (c = 2) →
  (1/2 * a * b * Real.sin C = Real.sqrt 3) →
  -- Conclusions to prove
  (C = π/3) ∧ (a = 2) ∧ (b = 2) := by
  sorry


end triangle_problem_l1419_141943


namespace divisibility_implies_seven_divides_l1419_141923

theorem divisibility_implies_seven_divides (n : ℕ) : 
  n ≥ 2 → (n ∣ 3^n + 4^n) → (7 ∣ n) := by sorry

end divisibility_implies_seven_divides_l1419_141923


namespace smallest_soldier_arrangement_l1419_141981

theorem smallest_soldier_arrangement : ∃ (n : ℕ), n > 0 ∧
  (∀ k ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), n % k = k - 1) ∧
  (∀ m : ℕ, m > 0 → (∀ k ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), m % k = k - 1) → m ≥ n) ∧
  n = 2519 := by
  sorry

end smallest_soldier_arrangement_l1419_141981


namespace line_intersects_ellipse_l1419_141918

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def PossibleSlopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/5) ∨ m ≥ Real.sqrt (1/5)}

/-- The equation of the line with slope m and y-intercept (0, -3) -/
def LineEquation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def EllipseEquation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) :
  m ∈ PossibleSlopes ↔
  ∃ x : ℝ, EllipseEquation x (LineEquation m x) :=
sorry

end line_intersects_ellipse_l1419_141918


namespace max_value_of_function_l1419_141903

theorem max_value_of_function (x : ℝ) : 
  (∀ x, -1 ≤ Real.cos x ∧ Real.cos x ≤ 1) →
  (∃ M : ℝ, M = 3 ∧ ∀ x, (2 + Real.cos x) / (2 - Real.cos x) ≤ M) :=
by sorry

end max_value_of_function_l1419_141903


namespace least_integer_square_quadruple_l1419_141961

theorem least_integer_square_quadruple (x : ℤ) : x^2 = 4*x + 56 → x ≥ -7 :=
by sorry

end least_integer_square_quadruple_l1419_141961


namespace gcd_of_B_is_two_l1419_141945

def B : Set ℕ := {n : ℕ | ∃ y : ℕ+, n = 4 * y + 2}

theorem gcd_of_B_is_two : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l1419_141945


namespace length_a_prime_b_prime_l1419_141917

/-- Given points A, B, C, and the line y = x, prove that the length of A'B' is 4√2 -/
theorem length_a_prime_b_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 10) →
  C = (3, 7) →
  (A'.1 = A'.2 ∧ B'.1 = B'.2) →  -- A' and B' are on the line y = x
  (∃ t : ℝ, A + t • (A' - A) = C) →  -- AA' passes through C
  (∃ s : ℝ, B + s • (B' - B) = C) →  -- BB' passes through C
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
by sorry

end length_a_prime_b_prime_l1419_141917


namespace simplify_expression_l1419_141960

theorem simplify_expression : (1 / ((-8^4)^2)) * (-8)^11 = -512 := by
  sorry

end simplify_expression_l1419_141960


namespace frog_hop_probability_l1419_141996

/-- Represents the possible positions on a 3x3 grid -/
inductive Position
  | Center
  | Edge
  | Corner

/-- Represents a single hop of the frog -/
def hop (pos : Position) : Position :=
  match pos with
  | Position.Center => Position.Edge
  | Position.Edge => sorry  -- Randomly choose between Center, Edge, or Corner
  | Position.Corner => Position.Corner

/-- Calculates the probability of landing on a corner exactly once in at most four hops -/
def prob_corner_once (hops : Nat) : ℚ :=
  sorry  -- Implement the probability calculation

/-- The main theorem stating the probability of landing on a corner exactly once in at most four hops -/
theorem frog_hop_probability : 
  prob_corner_once 4 = 25 / 32 := by
  sorry


end frog_hop_probability_l1419_141996


namespace distance_to_school_l1419_141933

/-- Represents the travel conditions to Jeremy's school -/
structure TravelConditions where
  normal_time : ℝ  -- Normal travel time in hours
  fast_time : ℝ    -- Travel time when speed is increased in hours
  slow_time : ℝ    -- Travel time when speed is decreased in hours
  speed_increase : ℝ  -- Speed increase in mph
  speed_decrease : ℝ  -- Speed decrease in mph

/-- Calculates the distance to Jeremy's school given the travel conditions -/
def calculateDistance (tc : TravelConditions) : ℝ :=
  -- Implementation not required for the statement
  sorry

/-- Theorem stating that the distance to Jeremy's school is 15 miles -/
theorem distance_to_school :
  let tc : TravelConditions := {
    normal_time := 1/2,  -- 30 minutes in hours
    fast_time := 3/10,   -- 18 minutes in hours
    slow_time := 2/3,    -- 40 minutes in hours
    speed_increase := 15,
    speed_decrease := 10
  }
  calculateDistance tc = 15 := by
  sorry


end distance_to_school_l1419_141933


namespace stone_number_150_l1419_141936

/-- Represents the counting pattern for each round -/
def countingPattern : List Nat := [12, 10, 8, 6, 4, 2]

/-- Calculates the sum of a list of natural numbers -/
def sumList (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

/-- Represents the total count in one complete cycle -/
def cycleCount : Nat := sumList countingPattern

/-- Calculates the number of complete cycles before reaching the target count -/
def completeCycles (target : Nat) : Nat :=
  target / cycleCount

/-- Calculates the remaining count after complete cycles -/
def remainingCount (target : Nat) : Nat :=
  target % cycleCount

/-- Finds the original stone number corresponding to the target count -/
def findStoneNumber (target : Nat) : Nat :=
  let remainingCount := remainingCount target
  let rec findInPattern (count : Nat) (pattern : List Nat) : Nat :=
    match pattern with
    | [] => 0  -- Should not happen if the input is valid
    | h :: t =>
      if count <= h then
        12 - (h - count) - (6 - pattern.length) * 2
      else
        findInPattern (count - h) t
  findInPattern remainingCount countingPattern

theorem stone_number_150 :
  findStoneNumber 150 = 4 := by sorry

end stone_number_150_l1419_141936


namespace complex_equation_solution_l1419_141990

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 11 + 7 * I → z = 3 + 5 * I := by
  sorry

end complex_equation_solution_l1419_141990


namespace imaginary_part_of_z_l1419_141920

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I + 2) :
  z.im = -1/2 := by
  sorry

end imaginary_part_of_z_l1419_141920


namespace vector_dot_product_and_magnitude_l1419_141998

theorem vector_dot_product_and_magnitude :
  ∀ (t : ℝ),
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2, t]
  (a 0 * b 0 + a 1 * b 1 = 0) →
  Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = Real.sqrt 5 :=
by sorry

end vector_dot_product_and_magnitude_l1419_141998


namespace distance_AB_distance_AB_value_l1419_141900

def path_north : ℝ := 30 - 15 + 10
def path_east : ℝ := 80 - 30

theorem distance_AB : ℝ :=
  let north_south_distance := path_north
  let east_west_distance := path_east
  Real.sqrt (north_south_distance ^ 2 + east_west_distance ^ 2)

theorem distance_AB_value : distance_AB = 25 * Real.sqrt 5 := by sorry

end distance_AB_distance_AB_value_l1419_141900


namespace min_odd_integers_l1419_141949

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_3 : a + b + c = 36)
  (sum_5 : a + b + c + d + e = 59)
  (sum_6 : a + b + c + d + e + f = 78) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    odds.card = 2 ∧
    (∀ x ∈ odds, Odd x) ∧
    (∀ (odds' : Finset ℤ), odds' ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ odds', Odd x) → odds'.card ≥ 2) :=
by sorry

end min_odd_integers_l1419_141949


namespace common_internal_tangent_length_l1419_141948

/-- Given two circles with centers 25 inches apart, where one circle has a radius of 7 inches
    and the other has a radius of 10 inches, the length of their common internal tangent
    is √336 inches. -/
theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 25)
  (h2 : small_radius = 7)
  (h3 : large_radius = 10) :
  Real.sqrt (center_distance ^ 2 - (small_radius + large_radius) ^ 2) = Real.sqrt 336 :=
by sorry

end common_internal_tangent_length_l1419_141948


namespace weight_difference_l1419_141941

/-- Given the combined weights of Annette and Caitlin, and Caitlin and Sara,
    prove that Annette weighs 8 pounds more than Sara. -/
theorem weight_difference (a c s : ℝ) 
  (h1 : a + c = 95) 
  (h2 : c + s = 87) : 
  a - s = 8 := by
  sorry

end weight_difference_l1419_141941


namespace quadratic_roots_when_positive_discriminant_l1419_141974

theorem quadratic_roots_when_positive_discriminant
  (a b c : ℝ) (h_a : a ≠ 0) (h_disc : b^2 - 4*a*c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end quadratic_roots_when_positive_discriminant_l1419_141974


namespace givenPointInSecondQuadrant_l1419_141955

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point we want to prove is in the second quadrant -/
def givenPoint : Point :=
  { x := -1, y := 2 }

/-- Theorem stating that the given point is in the second quadrant -/
theorem givenPointInSecondQuadrant : isInSecondQuadrant givenPoint := by
  sorry

end givenPointInSecondQuadrant_l1419_141955


namespace five_balls_four_boxes_l1419_141997

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by sorry

end five_balls_four_boxes_l1419_141997


namespace discount_problem_l1419_141916

theorem discount_problem (original_price : ℝ) : 
  original_price > 0 → 
  0.7 * original_price + 0.8 * original_price = 50 → 
  original_price = 100 / 3 := by
sorry

end discount_problem_l1419_141916


namespace textbook_savings_l1419_141940

/-- Calculates the savings when buying a textbook from an external bookshop instead of the school bookshop -/
def calculate_savings (school_price : ℚ) (discount_percent : ℚ) : ℚ :=
  school_price * discount_percent / 100

/-- Represents the prices and discounts for the three textbooks -/
structure TextbookPrices where
  math_price : ℚ
  math_discount : ℚ
  science_price : ℚ
  science_discount : ℚ
  literature_price : ℚ
  literature_discount : ℚ

/-- Calculates the total savings for all three textbooks -/
def total_savings (prices : TextbookPrices) : ℚ :=
  calculate_savings prices.math_price prices.math_discount +
  calculate_savings prices.science_price prices.science_discount +
  calculate_savings prices.literature_price prices.literature_discount

/-- Theorem stating that the total savings is $29.25 -/
theorem textbook_savings :
  let prices : TextbookPrices := {
    math_price := 45,
    math_discount := 20,
    science_price := 60,
    science_discount := 25,
    literature_price := 35,
    literature_discount := 15
  }
  total_savings prices = 29.25 := by
  sorry

end textbook_savings_l1419_141940


namespace stating_max_squares_specific_cases_l1419_141953

/-- 
Given a rectangular grid of dimensions m × n, this function calculates 
the maximum number of squares that can be cut along the grid lines.
-/
def max_squares (m n : ℕ) : ℕ := sorry

/--
Theorem stating that for specific grid dimensions (8, 11) and (8, 12),
the maximum number of squares that can be cut is 5.
-/
theorem max_squares_specific_cases : 
  (max_squares 8 11 = 5) ∧ (max_squares 8 12 = 5) := by sorry

end stating_max_squares_specific_cases_l1419_141953


namespace min_value_sum_of_reciprocals_l1419_141952

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h_sum : a + b + c + d + e + f = 10) :
  2/a + 3/b + 9/c + 16/d + 25/e + 36/f ≥ (329 + 38 * Real.sqrt 6) / 10 := by
  sorry

end min_value_sum_of_reciprocals_l1419_141952


namespace remaining_pills_l1419_141973

/-- Calculates the total number of pills left after using supplements for a specified number of days. -/
def pillsLeft (largeBottles smallBottles : ℕ) (largePillCount smallPillCount daysUsed : ℕ) : ℕ :=
  (largeBottles * (largePillCount - daysUsed)) + (smallBottles * (smallPillCount - daysUsed))

/-- Theorem stating that given the specific supplement configuration and usage, 350 pills remain. -/
theorem remaining_pills :
  pillsLeft 3 2 120 30 14 = 350 := by
  sorry

end remaining_pills_l1419_141973


namespace ratio_of_amounts_l1419_141946

theorem ratio_of_amounts (total : ℕ) (r_amount : ℕ) (h1 : total = 5000) (h2 : r_amount = 2000) :
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by sorry

end ratio_of_amounts_l1419_141946


namespace a_minus_2ab_plus_b_eq_zero_l1419_141987

theorem a_minus_2ab_plus_b_eq_zero 
  (a b : ℝ) 
  (h1 : a + b = 2) 
  (h2 : a * b = 1) : 
  a - 2 * a * b + b = 0 := by
sorry

end a_minus_2ab_plus_b_eq_zero_l1419_141987


namespace coefficient_of_monomial_l1419_141983

def monomial : ℤ × ℤ × ℤ → ℤ
| (a, m, n) => a * m * n^3

theorem coefficient_of_monomial :
  ∃ (m n : ℤ), monomial (-5, m, n) = -5 * m * n^3 :=
by sorry

end coefficient_of_monomial_l1419_141983


namespace male_cattle_percentage_l1419_141950

/-- Represents the farmer's cattle statistics -/
structure CattleStats where
  total_milk : ℕ
  milk_per_cow : ℕ
  male_count : ℕ

/-- Calculates the percentage of male cattle -/
def male_percentage (stats : CattleStats) : ℚ :=
  let female_count := stats.total_milk / stats.milk_per_cow
  let total_cattle := stats.male_count + female_count
  (stats.male_count : ℚ) / (total_cattle : ℚ) * 100

/-- Theorem stating that the percentage of male cattle is 40% -/
theorem male_cattle_percentage (stats : CattleStats) 
  (h1 : stats.total_milk = 150)
  (h2 : stats.milk_per_cow = 2)
  (h3 : stats.male_count = 50) :
  male_percentage stats = 40 := by
  sorry

#eval male_percentage { total_milk := 150, milk_per_cow := 2, male_count := 50 }

end male_cattle_percentage_l1419_141950


namespace greatest_3digit_base9_divisible_by_7_l1419_141958

/-- Converts a base 9 number to base 10 --/
def base9To10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 --/
def base10To9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 9 number --/
def isThreeDigitBase9 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase9 n ∧ 
             (base9To10 n) % 7 = 0 ∧
             ∀ (m : ℕ), isThreeDigitBase9 m ∧ (base9To10 m) % 7 = 0 → m ≤ n ∧
             n = 888 := by
  sorry

end greatest_3digit_base9_divisible_by_7_l1419_141958


namespace winning_scores_count_l1419_141951

-- Define the number of teams and runners per team
def num_teams : Nat := 3
def runners_per_team : Nat := 3

-- Define the total number of runners
def total_runners : Nat := num_teams * runners_per_team

-- Define the sum of all positions
def total_points : Nat := (total_runners * (total_runners + 1)) / 2

-- Define the maximum possible winning score
def max_winning_score : Nat := total_points / 2

-- Define the minimum possible winning score
def min_winning_score : Nat := 1 + 2 + 3

-- Theorem statement
theorem winning_scores_count :
  (∃ (winning_scores : Finset Nat),
    (∀ s ∈ winning_scores, min_winning_score ≤ s ∧ s ≤ max_winning_score) ∧
    (∀ s ∈ winning_scores, ∃ (a b c : Nat),
      a < b ∧ b < c ∧ c ≤ total_runners ∧ s = a + b + c) ∧
    winning_scores.card = 17) :=
by sorry

end winning_scores_count_l1419_141951


namespace not_multiple_of_121_l1419_141930

theorem not_multiple_of_121 (n : ℤ) : ¬(121 ∣ (n^2 + 2*n + 12)) := by
  sorry

end not_multiple_of_121_l1419_141930


namespace interior_angle_sum_l1419_141989

/-- 
Given a convex polygon where the sum of interior angles is 1800°,
prove that the sum of interior angles of a polygon with 3 fewer sides is 1260°.
-/
theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n - 3) - 2) = 1260) := by
  sorry

end interior_angle_sum_l1419_141989


namespace phase_shift_cos_l1419_141927

theorem phase_shift_cos (b c : ℝ) : 
  let phase_shift := -c / b
  b = 2 ∧ c = π / 2 → phase_shift = -π / 4 := by
sorry

end phase_shift_cos_l1419_141927


namespace arithmetic_sequence_sum_l1419_141932

/-- An arithmetic sequence with its first term and sum function -/
structure ArithmeticSequence where
  a₁ : ℤ
  S : ℕ → ℤ

/-- The specific arithmetic sequence from the problem -/
def problemSequence : ArithmeticSequence where
  a₁ := -2012
  S := sorry  -- Definition of S is left as sorry as it's not explicitly given in the conditions

/-- The main theorem to prove -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h : seq.a₁ = -2012)
  (h_sum_diff : seq.S 2012 / 2012 - seq.S 10 / 10 = 2002) :
  seq.S 2017 = 2017 := by
  sorry

end arithmetic_sequence_sum_l1419_141932


namespace perpendicular_lines_intersection_l1419_141964

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The equation of a line in the form ax + by = c -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y = c

theorem perpendicular_lines_intersection (a b c : ℝ) :
  line_equation a (-2) c 1 (-5) ∧
  line_equation 2 b (-c) 1 (-5) ∧
  perpendicular (a / 2) (-2 / b) →
  c = 13 := by sorry

end perpendicular_lines_intersection_l1419_141964


namespace point_in_second_quadrant_l1419_141902

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := 3 - m

/-- The y-coordinate of point P -/
def y_coord (m : ℝ) : ℝ := m - 1

/-- If point P(3-m, m-1) is in the second quadrant, then m > 3 -/
theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (x_coord m) (y_coord m) → m > 3 := by
  sorry

end point_in_second_quadrant_l1419_141902


namespace trig_identity_l1419_141963

theorem trig_identity : 
  Real.sin (20 * π / 180) * Real.sin (80 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end trig_identity_l1419_141963


namespace multiples_of_10_average_l1419_141957

theorem multiples_of_10_average : 
  let first := 10
  let last := 600
  let step := 10
  let count := (last - first) / step + 1
  let sum := count * (first + last) / 2
  sum / count = 305 := by
sorry

end multiples_of_10_average_l1419_141957


namespace angle_tangent_sum_zero_l1419_141931

theorem angle_tangent_sum_zero :
  ∃ θ : Real,
    0 < θ ∧ θ < π / 6 ∧
    Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) + Real.tan (5 * θ) = 0 ∧
    Real.tan θ = 1 / Real.sqrt 5 := by
  sorry

end angle_tangent_sum_zero_l1419_141931


namespace min_toothpicks_removal_for_48_l1419_141977

/-- Represents a hexagonal grid structure --/
structure HexagonalGrid where
  toothpicks : ℕ
  small_hexagons : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all triangles --/
def min_toothpicks_to_remove (grid : HexagonalGrid) : ℕ :=
  sorry

/-- Theorem stating the minimum number of toothpicks to remove for a specific grid --/
theorem min_toothpicks_removal_for_48 :
  ∀ (grid : HexagonalGrid),
    grid.toothpicks = 48 →
    min_toothpicks_to_remove grid = 6 :=
by
  sorry

end min_toothpicks_removal_for_48_l1419_141977


namespace quadratic_equation_properties_l1419_141908

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 + m*x + m - 2
  (f (-2) = 0) →
  (∃ x, x ≠ -2 ∧ f x = 0 ∧ x = 0) ∧
  (∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0) :=
by sorry


end quadratic_equation_properties_l1419_141908


namespace orange_juice_consumption_l1419_141986

theorem orange_juice_consumption (initial_amount : ℚ) (alex_fraction : ℚ) (pat_fraction : ℚ) :
  initial_amount = 3/4 →
  alex_fraction = 1/2 →
  pat_fraction = 1/3 →
  pat_fraction * (initial_amount - alex_fraction * initial_amount) = 1/8 := by
  sorry

end orange_juice_consumption_l1419_141986


namespace arccos_sum_eq_pi_half_l1419_141906

theorem arccos_sum_eq_pi_half (x : ℝ) :
  Real.arccos (3 * x) + Real.arccos x = π / 2 →
  x = 1 / Real.sqrt 10 ∨ x = -1 / Real.sqrt 10 :=
by sorry

end arccos_sum_eq_pi_half_l1419_141906


namespace triangle_trigonometric_identities_l1419_141988

theorem triangle_trigonometric_identities (A B C : ℝ) 
  (h : A + B + C = π) : 
  (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 2 * (1 + Real.cos A * Real.cos B * Real.cos C) ∧
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = 1 - 2 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end triangle_trigonometric_identities_l1419_141988


namespace tan_value_from_ratio_l1419_141956

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan α = -3 := by
  sorry

end tan_value_from_ratio_l1419_141956


namespace max_sum_distances_l1419_141994

-- Define the points A, B, and O
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the incircle of triangle AOB
def incircle (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 1 + Real.cos θ ∧ P.2 = 4/3 + Real.sin θ

-- Define the distance function
def dist_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- State the theorem
theorem max_sum_distances :
  ∀ P : ℝ × ℝ, incircle P →
    dist_squared P A + dist_squared P B + dist_squared P O ≤ 22 ∧
    ∃ P : ℝ × ℝ, incircle P ∧
      dist_squared P A + dist_squared P B + dist_squared P O = 22 := by
  sorry


end max_sum_distances_l1419_141994


namespace composite_shape_area_l1419_141928

/-- The total area of a composite shape consisting of three rectangles -/
def composite_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height

/-- Theorem stating that the area of the given composite shape is 68 square units -/
theorem composite_shape_area : composite_area 7 6 3 2 4 5 = 68 := by
  sorry

end composite_shape_area_l1419_141928


namespace f_range_l1419_141907

/-- The function f(x) = (x^2-1)(x^2-12x+35) -/
def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 12*x + 35)

/-- The graph of f(x) is symmetric about the line x=3 -/
axiom f_symmetry (x : ℝ) : f (6 - x) = f x

theorem f_range : Set.range f = Set.Ici (-36) := by sorry

end f_range_l1419_141907


namespace system_solution_l1419_141912

theorem system_solution : ∃ (x y : ℚ), (4 * x - 3 * y = -13) ∧ (5 * x + 3 * y = -14) ∧ (x = -3) ∧ (y = 1/3) := by
  sorry

end system_solution_l1419_141912


namespace triangle_sine_value_l1419_141984

theorem triangle_sine_value (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- Side lengths are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) ∧
  -- Given conditions
  (C = π / 6) ∧
  (a = 3) ∧
  (c = 4) →
  Real.sin A = 3 / 8 := by
sorry

end triangle_sine_value_l1419_141984


namespace units_digit_of_17_to_2107_l1419_141937

theorem units_digit_of_17_to_2107 :
  (17^2107 : ℕ) % 10 = 3 :=
sorry

end units_digit_of_17_to_2107_l1419_141937


namespace permutation_problem_l1419_141919

-- Define permutation function
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then (n - r + 1).factorial / (n - r).factorial else 0

-- Theorem statement
theorem permutation_problem : 
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * Nat.factorial 0 = 4 :=
by sorry

end permutation_problem_l1419_141919


namespace simplify_expression_l1419_141910

theorem simplify_expression : (1024 : ℝ) ^ (1/5 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end simplify_expression_l1419_141910


namespace pyramid_surface_area_l1419_141991

/-- The total surface area of a pyramid with a regular hexagonal base -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  let base_area := (3 * a^2 * Real.sqrt 3) / 2
  let perp_edge_length := a
  let side_triangle_area := a^2 / 2
  let side_triangle_area2 := a^2
  let side_triangle_area3 := (a^2 * Real.sqrt 7) / 4
  base_area + 2 * side_triangle_area + 2 * side_triangle_area2 + 2 * side_triangle_area3 =
    (a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7)) / 2 := by
  sorry

end pyramid_surface_area_l1419_141991
