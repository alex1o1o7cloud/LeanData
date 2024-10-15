import Mathlib

namespace NUMINAMATH_CALUDE_total_students_l3517_351749

/-- Represents the setup of students in lines -/
structure StudentLines where
  total_lines : ℕ
  students_per_line : ℕ
  left_position : ℕ
  right_position : ℕ

/-- Theorem stating the total number of students given the conditions -/
theorem total_students (setup : StudentLines) 
  (h1 : setup.total_lines = 5)
  (h2 : setup.left_position = 4)
  (h3 : setup.right_position = 9)
  (h4 : setup.students_per_line = setup.left_position + setup.right_position - 1) :
  setup.total_lines * setup.students_per_line = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l3517_351749


namespace NUMINAMATH_CALUDE_least_sum_m_n_l3517_351774

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m.val + n.val) 330 = 1) ∧ 
  (∃ (k : ℕ), m.val ^ m.val = k * (n.val ^ n.val)) ∧ 
  (∀ (l : ℕ), m.val ≠ l * n.val) ∧
  (m.val + n.val = 390) ∧
  (∀ (p q : ℕ+), 
    (Nat.gcd (p.val + q.val) 330 = 1) → 
    (∃ (k : ℕ), p.val ^ p.val = k * (q.val ^ q.val)) → 
    (∀ (l : ℕ), p.val ≠ l * q.val) → 
    (p.val + q.val ≥ 390)) := by
  sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l3517_351774


namespace NUMINAMATH_CALUDE_inequality_proof_l3517_351765

theorem inequality_proof (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3517_351765


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3517_351752

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity 2,
    prove that the equation of its asymptotes is y = ± √3 x. -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let eccentricity := Real.sqrt ((a^2 + b^2) / a^2)
  eccentricity = 2 →
  ∃ k : ℝ, k = Real.sqrt 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ C → y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3517_351752


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l3517_351723

theorem matrix_sum_theorem (a b c : ℝ) 
  (h : a^4 + b^4 + c^4 - a^2*b^2 - a^2*c^2 - b^2*c^2 = 0) :
  (a^2 / (b^2 + c^2)) + (b^2 / (a^2 + c^2)) + (c^2 / (a^2 + b^2)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l3517_351723


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3517_351761

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * 3 + Real.sqrt 3) * z = Complex.I * 3 →
  z = 3 / 4 + Complex.I * (Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3517_351761


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l3517_351788

theorem max_value_sum_of_fractions (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 / (1 + x^2) + y^2 / (1 + y^2) + z^2 / (1 + z^2) = 2) :
  x / (1 + x^2) + y / (1 + y^2) + z / (1 + z^2) ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l3517_351788


namespace NUMINAMATH_CALUDE_equivalent_coin_value_l3517_351781

theorem equivalent_coin_value : ∀ (quarter_value dime_value : ℕ),
  quarter_value = 25 →
  dime_value = 10 →
  30 * quarter_value + 20 * dime_value = 15 * quarter_value + 58 * dime_value :=
by
  sorry

end NUMINAMATH_CALUDE_equivalent_coin_value_l3517_351781


namespace NUMINAMATH_CALUDE_company_profit_and_assignment_l3517_351756

/-- Represents the profit calculation for a company with two products. -/
def CompanyProfit (totalWorkers : ℕ) (profitA profitB : ℚ) (decreaseRate : ℚ) : Prop :=
  ∃ x : ℕ,
    x ≤ totalWorkers ∧
    let workersA := totalWorkers - x
    let outputA := 2 * workersA
    let outputB := x
    let profitPerUnitB := profitB - decreaseRate * x
    let totalProfitA := profitA * outputA
    let totalProfitB := profitPerUnitB * outputB
    totalProfitA = totalProfitB + 650 ∧
    totalProfitA + totalProfitB = 2650

/-- Represents the optimal worker assignment when introducing a third product. -/
def OptimalAssignment (totalWorkers : ℕ) (profitA profitB profitC : ℚ) (decreaseRate : ℚ) : Prop :=
  ∃ m : ℕ,
    m ≤ totalWorkers ∧
    let workersA := m
    let workersC := 2 * m
    let workersB := totalWorkers - workersA - workersC
    workersA + workersB + workersC = totalWorkers ∧
    let outputA := 2 * workersA
    let outputB := workersB
    let outputC := workersC
    outputA = outputC ∧
    let profitPerUnitB := profitB - decreaseRate * workersB
    let totalProfit := profitA * outputA + profitPerUnitB * outputB + profitC * outputC
    totalProfit = 2650 ∧
    m = 10

/-- Theorem stating the company's profit and optimal assignment. -/
theorem company_profit_and_assignment :
  CompanyProfit 65 15 120 2 ∧
  OptimalAssignment 65 15 120 30 2 :=
sorry

end NUMINAMATH_CALUDE_company_profit_and_assignment_l3517_351756


namespace NUMINAMATH_CALUDE_circus_tickets_l3517_351714

theorem circus_tickets (ticket_cost : ℕ) (total_spent : ℕ) (h1 : ticket_cost = 44) (h2 : total_spent = 308) :
  total_spent / ticket_cost = 7 :=
sorry

end NUMINAMATH_CALUDE_circus_tickets_l3517_351714


namespace NUMINAMATH_CALUDE_equivalent_statements_l3517_351733

theorem equivalent_statements (A B : Prop) :
  ((A ∧ B) → ¬(A ∨ B)) ↔ ((A ∨ B) → ¬(A ∧ B)) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l3517_351733


namespace NUMINAMATH_CALUDE_prob_select_all_cocaptains_l3517_351792

/-- Represents a math team with a given number of students and co-captains -/
structure MathTeam where
  num_students : ℕ
  num_cocaptains : ℕ

/-- Calculates the probability of selecting all co-captains from a given team -/
def prob_select_cocaptains (team : MathTeam) : ℚ :=
  1 / (team.num_students.choose 3)

/-- The set of math teams in the area -/
def math_teams : List MathTeam := [
  { num_students := 6, num_cocaptains := 3 },
  { num_students := 7, num_cocaptains := 3 },
  { num_students := 8, num_cocaptains := 3 },
  { num_students := 9, num_cocaptains := 3 }
]

/-- Theorem stating the probability of selecting all co-captains -/
theorem prob_select_all_cocaptains : 
  (1 / (math_teams.length : ℚ)) * (math_teams.map prob_select_cocaptains).sum = 91 / 6720 := by
  sorry


end NUMINAMATH_CALUDE_prob_select_all_cocaptains_l3517_351792


namespace NUMINAMATH_CALUDE_wang_lei_pastries_l3517_351780

/-- Represents the number of pastries in a large box -/
def large_box_pastries : ℕ := 32

/-- Represents the number of pastries in a small box -/
def small_box_pastries : ℕ := 15

/-- Represents the cost of a large box in yuan -/
def large_box_cost : ℚ := 85.6

/-- Represents the cost of a small box in yuan -/
def small_box_cost : ℚ := 46.8

/-- Represents the total amount spent by Wang Lei in yuan -/
def total_spent : ℚ := 654

/-- Represents the total number of boxes bought by Wang Lei -/
def total_boxes : ℕ := 9

/-- Theorem stating that Wang Lei got 237 pastries -/
theorem wang_lei_pastries : 
  ∃ (large_boxes small_boxes : ℕ), 
    large_boxes + small_boxes = total_boxes ∧
    large_box_cost * large_boxes + small_box_cost * small_boxes = total_spent ∧
    large_box_pastries * large_boxes + small_box_pastries * small_boxes = 237 :=
by sorry

end NUMINAMATH_CALUDE_wang_lei_pastries_l3517_351780


namespace NUMINAMATH_CALUDE_sum_equals_1332_l3517_351736

/-- Converts a base 4 number (represented as a list of digits) to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a decimal number to its base 4 representation (as a list of digits) -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The sum of 232₄, 121₄, and 313₄ in base 4 -/
def sumInBase4 : List Nat :=
  decimalToBase4 (base4ToDecimal [2,3,2] + base4ToDecimal [1,2,1] + base4ToDecimal [3,1,3])

theorem sum_equals_1332 : sumInBase4 = [1,3,3,2] := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_1332_l3517_351736


namespace NUMINAMATH_CALUDE_log_inequality_l3517_351760

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x^2) < x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3517_351760


namespace NUMINAMATH_CALUDE_percy_swimming_weeks_l3517_351739

/-- Represents Percy's swimming schedule and calculates the number of weeks to swim a given total hours -/
def swimming_schedule (weekday_hours_per_day : ℕ) (weekday_days : ℕ) (weekend_hours : ℕ) (total_hours : ℕ) : ℕ :=
  let hours_per_week := weekday_hours_per_day * weekday_days + weekend_hours
  total_hours / hours_per_week

/-- Proves that Percy's swimming schedule over 52 hours covers 4 weeks -/
theorem percy_swimming_weeks : swimming_schedule 2 5 3 52 = 4 := by
  sorry

#eval swimming_schedule 2 5 3 52

end NUMINAMATH_CALUDE_percy_swimming_weeks_l3517_351739


namespace NUMINAMATH_CALUDE_seating_arrangement_l3517_351715

theorem seating_arrangement (n m : ℕ) (h1 : n = 6) (h2 : m = 4) : 
  (n.factorial / (n - m).factorial) = 360 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3517_351715


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3517_351746

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + n/8 + 1/8 < 1 ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3517_351746


namespace NUMINAMATH_CALUDE_max_value_expression_l3517_351751

theorem max_value_expression (a b c d : ℝ) 
  (ha : -4.5 ≤ a ∧ a ≤ 4.5)
  (hb : -4.5 ≤ b ∧ b ≤ 4.5)
  (hc : -4.5 ≤ c ∧ c ≤ 4.5)
  (hd : -4.5 ≤ d ∧ d ≤ 4.5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    (-4.5 ≤ a' ∧ a' ≤ 4.5) ∧
    (-4.5 ≤ b' ∧ b' ≤ 4.5) ∧
    (-4.5 ≤ c' ∧ c' ≤ 4.5) ∧
    (-4.5 ≤ d' ∧ d' ≤ 4.5) ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 90 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3517_351751


namespace NUMINAMATH_CALUDE_spiders_went_loose_l3517_351729

theorem spiders_went_loose (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (birds_sold puppies_adopted animals_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  initial_spiders = 15 →
  birds_sold = initial_birds / 2 →
  puppies_adopted = 3 →
  animals_left = 25 →
  initial_spiders - (animals_left - ((initial_birds - birds_sold) + (initial_puppies - puppies_adopted) + initial_cats)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_spiders_went_loose_l3517_351729


namespace NUMINAMATH_CALUDE_triangle_angle_B_l3517_351772

theorem triangle_angle_B (a b : ℝ) (A : ℝ) (h1 : a = 2) (h2 : b = 2 * Real.sqrt 3) (h3 : A = π / 6) :
  ∃ B : ℝ, (B = π / 3 ∨ B = 2 * π / 3) ∧
    a / Real.sin A = b / Real.sin B :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l3517_351772


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3517_351778

/-- If point P with coordinates (4-a, 3a+9) lies on the x-axis, then its coordinates are (7, 0) -/
theorem point_on_x_axis (a : ℝ) :
  let P : ℝ × ℝ := (4 - a, 3 * a + 9)
  (P.2 = 0) → P = (7, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3517_351778


namespace NUMINAMATH_CALUDE_cosine_in_special_triangle_l3517_351711

/-- Given a triangle ABC where the sides a, b, and c are in the ratio 2:3:4, 
    prove that cos C = -1/4 -/
theorem cosine_in_special_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (ratio : ∃ (x : ℝ), x > 0 ∧ a = 2*x ∧ b = 3*x ∧ c = 4*x) : 
    (a^2 + b^2 - c^2) / (2*a*b) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_in_special_triangle_l3517_351711


namespace NUMINAMATH_CALUDE_solve_luncheon_problem_l3517_351796

def luncheon_problem (no_shows : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) : Prop :=
  let attendees := tables_needed * table_capacity
  let total_invited := no_shows + attendees
  total_invited = 18

theorem solve_luncheon_problem :
  luncheon_problem 12 3 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_luncheon_problem_l3517_351796


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3517_351779

/-- The distance between foci of an ellipse with given parameters -/
theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : a > b) :
  2 * Real.sqrt (a^2 - b^2) = 12 := by
  sorry

#check ellipse_foci_distance

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3517_351779


namespace NUMINAMATH_CALUDE_sand_remaining_l3517_351731

/-- Given a truck with an initial amount of sand and an amount of sand lost during transit,
    prove that the remaining amount of sand is equal to the initial amount minus the lost amount. -/
theorem sand_remaining (initial_sand : ℝ) (sand_lost : ℝ) :
  initial_sand - sand_lost = initial_sand - sand_lost :=
by sorry

end NUMINAMATH_CALUDE_sand_remaining_l3517_351731


namespace NUMINAMATH_CALUDE_factorial_ratio_squared_l3517_351707

theorem factorial_ratio_squared : (Nat.factorial 10 / Nat.factorial 9) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_squared_l3517_351707


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3517_351720

-- Define the geometric sequence and its sum
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)
def S (n : ℕ) : ℝ := (3^n - 1)

-- State the theorem
theorem geometric_sequence_properties :
  (a 1 + a 2 + a 3 = 26) ∧ 
  (S 6 = 728) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * 3^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → S (n + 1)^2 - S n * S (n + 2) = 4 * 3^n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3517_351720


namespace NUMINAMATH_CALUDE_pascal_triangle_15th_row_4th_number_l3517_351794

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem pascal_triangle_15th_row_4th_number : binomial 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_15th_row_4th_number_l3517_351794


namespace NUMINAMATH_CALUDE_f_max_value_l3517_351775

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n) / ((n + 32) * (S (n + 1)))

theorem f_max_value : ∀ n : ℕ, f n ≤ 1 / 50 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l3517_351775


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3517_351713

theorem unique_integer_solution : 
  ∀ m n : ℕ+, 
    (m : ℚ) + n - (3 * m * n) / (m + n) = 2011 / 3 ↔ 
    ((m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144)) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3517_351713


namespace NUMINAMATH_CALUDE_brand_comparison_l3517_351703

/-- Distribution of timing errors for brand A -/
def dist_A : List (ℝ × ℝ) := [(-1, 0.1), (0, 0.8), (1, 0.1)]

/-- Distribution of timing errors for brand B -/
def dist_B : List (ℝ × ℝ) := [(-2, 0.1), (-1, 0.2), (0, 0.4), (1, 0.2), (2, 0.1)]

/-- Expected value of a discrete random variable -/
def expected_value (dist : List (ℝ × ℝ)) : ℝ :=
  (dist.map (fun (x, p) => x * p)).sum

/-- Variance of a discrete random variable -/
def variance (dist : List (ℝ × ℝ)) : ℝ :=
  (dist.map (fun (x, p) => x^2 * p)).sum - (expected_value dist)^2

/-- Theorem stating the properties of brands A and B -/
theorem brand_comparison :
  expected_value dist_A = 0 ∧
  expected_value dist_B = 0 ∧
  variance dist_A = 0.2 ∧
  variance dist_B = 1.2 ∧
  variance dist_A < variance dist_B := by
  sorry

#check brand_comparison

end NUMINAMATH_CALUDE_brand_comparison_l3517_351703


namespace NUMINAMATH_CALUDE_youngbin_line_position_l3517_351719

/-- Given a line of students with Youngbin in it, calculate the number of students in front of Youngbin. -/
def students_in_front (total : ℕ) (behind : ℕ) : ℕ :=
  total - behind - 1

/-- Theorem: There are 11 students in front of Youngbin given the problem conditions. -/
theorem youngbin_line_position : students_in_front 25 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_youngbin_line_position_l3517_351719


namespace NUMINAMATH_CALUDE_makeup_palette_cost_l3517_351737

/-- The cost of a makeup palette given the following conditions:
  * There are 3 makeup palettes
  * 4 lipsticks cost $2.50 each
  * 3 boxes of hair color cost $4 each
  * The total cost is $67
-/
theorem makeup_palette_cost :
  let num_palettes : ℕ := 3
  let num_lipsticks : ℕ := 4
  let lipstick_cost : ℚ := 5/2
  let num_hair_color : ℕ := 3
  let hair_color_cost : ℚ := 4
  let total_cost : ℚ := 67
  (total_cost - (num_lipsticks * lipstick_cost + num_hair_color * hair_color_cost)) / num_palettes = 15 := by
  sorry

end NUMINAMATH_CALUDE_makeup_palette_cost_l3517_351737


namespace NUMINAMATH_CALUDE_pure_imaginary_z_l3517_351771

theorem pure_imaginary_z (a : ℝ) : 
  let z : ℂ := a^2 + 2*a - 2 + (2*Complex.I)/(1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 ∨ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_l3517_351771


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l3517_351716

theorem larger_number_of_pair (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 156) (h3 : x > y) :
  x = (5 + Real.sqrt 649) / 2 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l3517_351716


namespace NUMINAMATH_CALUDE_system_solution_l3517_351791

theorem system_solution (x y z : ℝ) : 
  (2 * x^2 + 3 * y + 5 = 2 * Real.sqrt (2 * z + 5)) ∧
  (2 * y^2 + 3 * z + 5 = 2 * Real.sqrt (2 * x + 5)) ∧
  (2 * z^2 + 3 * x + 5 = 2 * Real.sqrt (2 * y + 5)) →
  x = -1/2 ∧ y = -1/2 ∧ z = -1/2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3517_351791


namespace NUMINAMATH_CALUDE_season_games_count_l3517_351701

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of baseball games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games_count : total_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l3517_351701


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3517_351705

/-- Given a rectangular metallic sheet with length 48 m and width w,
    if squares of 8 m are cut from each corner to form an open box with volume 5120 m³,
    then the width w of the original sheet is 36 m. -/
theorem metallic_sheet_width (w : ℝ) : 
  w > 0 →  -- Ensuring positive width
  (48 - 2 * 8) * (w - 2 * 8) * 8 = 5120 →  -- Volume equation
  w = 36 := by
sorry


end NUMINAMATH_CALUDE_metallic_sheet_width_l3517_351705


namespace NUMINAMATH_CALUDE_part_one_part_two_l3517_351721

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part_one (t : Triangle) (h1 : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b) 
    (h2 : 0 < t.A ∧ t.A < Real.pi / 2) : t.A = Real.pi / 3 := by
  sorry

-- Part 2
theorem part_two (t : Triangle) (h1 : t.b = 5) (h2 : t.c = Real.sqrt 5) 
    (h3 : Real.cos t.C = 9/10) : t.a = 4 ∨ t.a = 5 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3517_351721


namespace NUMINAMATH_CALUDE_yellow_mms_added_l3517_351741

/-- Represents the number of M&Ms of each color in the jar -/
structure MandMs where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- The initial state of the jar -/
def initial_jar : MandMs :=
  { green := 20, red := 20, yellow := 0 }

/-- The state of the jar after Carter eats 12 green M&Ms -/
def after_carter_eats (jar : MandMs) : MandMs :=
  { jar with green := jar.green - 12 }

/-- The state of the jar after Carter's sister eats half the red M&Ms -/
def after_sister_eats (jar : MandMs) : MandMs :=
  { jar with red := jar.red / 2 }

/-- The final state of the jar after yellow M&Ms are added -/
def final_jar (jar : MandMs) (yellow_added : ℕ) : MandMs :=
  { jar with yellow := jar.yellow + yellow_added }

/-- The probability of picking a green M&M from the jar -/
def prob_green (jar : MandMs) : ℚ :=
  jar.green / (jar.green + jar.red + jar.yellow)

/-- The theorem stating the number of yellow M&Ms added -/
theorem yellow_mms_added : 
  ∃ yellow_added : ℕ,
    let jar1 := after_carter_eats initial_jar
    let jar2 := after_sister_eats jar1
    let jar3 := final_jar jar2 yellow_added
    prob_green jar3 = 1/4 ∧ yellow_added = 14 := by
  sorry

end NUMINAMATH_CALUDE_yellow_mms_added_l3517_351741


namespace NUMINAMATH_CALUDE_binary_decimal_base7_conversion_l3517_351700

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_decimal_base7_conversion :
  let binary := [true, false, true, true, false, true]
  binary_to_decimal binary = 45 ∧
  decimal_to_base7 45 = [6, 3] :=
by sorry

end NUMINAMATH_CALUDE_binary_decimal_base7_conversion_l3517_351700


namespace NUMINAMATH_CALUDE_min_dot_product_on_W_l3517_351702

/-- The trajectory W of point P -/
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 2 ∧ p.1 ≥ Real.sqrt 2}

/-- The dot product of two vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

/-- The origin point -/
def O : ℝ × ℝ := (0, 0)

theorem min_dot_product_on_W :
  ∀ A B : ℝ × ℝ, A ∈ W → B ∈ W → A ≠ B →
  ∀ C D : ℝ × ℝ, C ∈ W → D ∈ W →
  dot_product (C.1 - O.1, C.2 - O.2) (D.1 - O.1, D.2 - O.2) ≥
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) →
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_W_l3517_351702


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l3517_351795

/-- A polynomial of the form x^n + 5x^(n-1) + 3 where n > 1 is irreducible over the integers -/
theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.monomial n 1 + Polynomial.monomial (n-1) 5 + Polynomial.C 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l3517_351795


namespace NUMINAMATH_CALUDE_three_km_to_meters_four_kg_to_grams_l3517_351770

-- Define the conversion factors
def meters_per_kilometer : ℝ := 1000
def grams_per_kilogram : ℝ := 1000

-- Theorem for kilometer to meter conversion
theorem three_km_to_meters :
  3 * meters_per_kilometer = 3000 := by sorry

-- Theorem for kilogram to gram conversion
theorem four_kg_to_grams :
  4 * grams_per_kilogram = 4000 := by sorry

end NUMINAMATH_CALUDE_three_km_to_meters_four_kg_to_grams_l3517_351770


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3517_351786

theorem fraction_product_simplification :
  (2 / 3) * (3 / 7) * (7 / 4) * (4 / 5) * (5 / 6) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3517_351786


namespace NUMINAMATH_CALUDE_decimal_place_150_is_3_l3517_351732

/-- The decimal representation of 7/11 repeats every 2 digits -/
def repeat_length : ℕ := 2

/-- The repeating decimal representation of 7/11 -/
def decimal_rep : List ℕ := [6, 3]

/-- The 150th decimal place of 7/11 -/
def decimal_place_150 : ℕ := 
  decimal_rep[(150 - 1) % repeat_length]

theorem decimal_place_150_is_3 : decimal_place_150 = 3 := by sorry

end NUMINAMATH_CALUDE_decimal_place_150_is_3_l3517_351732


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l3517_351799

/-- Calculates the total cost of apples for Irene and her dog for 2 weeks -/
def apple_cost (apple_weight : Real) (red_price : Real) (green_price : Real) 
  (red_increase : Real) (green_decrease : Real) : Real :=
  let apples_needed := 14 * 0.5
  let pounds_needed := apples_needed * apple_weight
  let week1_cost := pounds_needed * red_price
  let week2_cost := pounds_needed * (green_price * (1 - green_decrease))
  week1_cost + week2_cost

theorem apple_cost_calculation :
  apple_cost (1/4) 2 2.5 0.1 0.05 = 7.65625 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l3517_351799


namespace NUMINAMATH_CALUDE_power_sum_equals_six_l3517_351766

theorem power_sum_equals_six (a x : ℝ) (h : a^x - a^(-x) = 2) : 
  a^(2*x) + a^(-2*x) = 6 := by
sorry

end NUMINAMATH_CALUDE_power_sum_equals_six_l3517_351766


namespace NUMINAMATH_CALUDE_distance_P_to_y_axis_l3517_351789

/-- The distance from a point to the y-axis in a Cartesian coordinate system --/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- The point P --/
def P : ℝ × ℝ := (-3, 4)

/-- Theorem: The distance from P(-3, 4) to the y-axis is 3 --/
theorem distance_P_to_y_axis :
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_P_to_y_axis_l3517_351789


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l3517_351755

theorem rectangular_prism_width 
  (l h d : ℝ) 
  (hl : l = 6) 
  (hh : h = 8) 
  (hd : d = 15) 
  (h_diagonal : d^2 = l^2 + w^2 + h^2) : 
  w = 5 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l3517_351755


namespace NUMINAMATH_CALUDE_opposite_of_two_opposite_definition_l3517_351709

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 2 is -2
theorem opposite_of_two : opposite 2 = -2 := by
  -- The proof goes here
  sorry

-- Theorem proving the definition of opposite
theorem opposite_definition (x : ℝ) : x + opposite x = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_two_opposite_definition_l3517_351709


namespace NUMINAMATH_CALUDE_lilith_cap_collection_l3517_351734

/-- Calculates the total number of caps Lilith has collected over 5 years -/
def total_caps_collected : ℕ :=
  let caps_first_year := 3 * 12
  let caps_after_first_year := 5 * 12 * 4
  let caps_from_christmas := 40 * 5
  let caps_lost := 15 * 5
  caps_first_year + caps_after_first_year + caps_from_christmas - caps_lost

/-- Theorem stating that the total number of caps Lilith has collected is 401 -/
theorem lilith_cap_collection : total_caps_collected = 401 := by
  sorry

end NUMINAMATH_CALUDE_lilith_cap_collection_l3517_351734


namespace NUMINAMATH_CALUDE_second_number_proof_l3517_351782

theorem second_number_proof (x : ℕ) : 
  (∃ k₁ k₂ : ℕ, 690 = 170 * k₁ + 10 ∧ x = 170 * k₂ + 25) ∧
  (∀ d : ℕ, d > 170 → ¬(∃ m₁ m₂ : ℕ, 690 = d * m₁ + 10 ∧ x = d * m₂ + 25)) →
  x = 875 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l3517_351782


namespace NUMINAMATH_CALUDE_nonzero_real_solution_l3517_351706

theorem nonzero_real_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 1 / y = 12) (eq2 : y + 1 / x = 7 / 15) :
  x = 6 + 3 * Real.sqrt (8 / 7) ∨ x = 6 - 3 * Real.sqrt (8 / 7) :=
by sorry

end NUMINAMATH_CALUDE_nonzero_real_solution_l3517_351706


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3517_351740

theorem isosceles_triangle_quadratic_roots (m n : ℝ) (k : ℝ) : 
  (m > 0 ∧ n > 0) →  -- positive side lengths
  (m = n ∨ m = 4 ∨ n = 4) →  -- isosceles condition
  (m ≠ n ∨ m ≠ 4) →  -- not equilateral
  (m + n > 4 ∧ m + 4 > n ∧ n + 4 > m) →  -- triangle inequality
  (m^2 - 6*m + k + 2 = 0) →  -- m is a root
  (n^2 - 6*n + k + 2 = 0) →  -- n is a root
  (k = 6 ∨ k = 7) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3517_351740


namespace NUMINAMATH_CALUDE_f_domain_l3517_351763

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (3 - Real.tan x ^ 2) + Real.sqrt (x * (Real.pi - x))

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem f_domain :
  domain f = Set.Icc 0 (Real.pi / 3) ∪ Set.Ioc (2 * Real.pi / 3) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_f_domain_l3517_351763


namespace NUMINAMATH_CALUDE_sixteen_even_numbers_l3517_351773

/-- Represents a card with two numbers -/
structure Card where
  front : Nat
  back : Nat

/-- Counts the number of three-digit even numbers that can be formed from the given cards -/
def countEvenNumbers (cards : List Card) : Nat :=
  cards.foldl (fun acc card => 
    acc + (if card.front % 2 == 0 then 1 else 0) + 
          (if card.back % 2 == 0 then 1 else 0)
  ) 0

/-- The main theorem stating that 16 different three-digit even numbers can be formed -/
theorem sixteen_even_numbers : 
  let cards := [Card.mk 0 1, Card.mk 2 3, Card.mk 4 5]
  countEvenNumbers cards = 16 := by
  sorry


end NUMINAMATH_CALUDE_sixteen_even_numbers_l3517_351773


namespace NUMINAMATH_CALUDE_sixth_is_wednesday_l3517_351754

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week for a given date in a month starting with Friday -/
def dayOfWeek (date : Nat) : DayOfWeek :=
  match (date - 1) % 7 with
  | 0 => DayOfWeek.Friday
  | 1 => DayOfWeek.Saturday
  | 2 => DayOfWeek.Sunday
  | 3 => DayOfWeek.Monday
  | 4 => DayOfWeek.Tuesday
  | 5 => DayOfWeek.Wednesday
  | _ => DayOfWeek.Thursday

theorem sixth_is_wednesday 
  (h1 : ∃ (x : Nat), x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75) 
  : dayOfWeek 6 = DayOfWeek.Wednesday := by
  sorry

end NUMINAMATH_CALUDE_sixth_is_wednesday_l3517_351754


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3517_351797

/-- Sum of interior angles of a polygon with n sides -/
def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

theorem polygon_interior_angles :
  (∀ n : ℕ, n ≥ 3 → sumInteriorAngles n = (n - 2) * 180) ∧
  sumInteriorAngles 6 = 720 ∧
  (∃ n : ℕ, n ≥ 3 ∧ (1/3) * sumInteriorAngles n = 300 ∧ n = 7) :=
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3517_351797


namespace NUMINAMATH_CALUDE_equal_probability_same_different_color_l3517_351768

theorem equal_probability_same_different_color (t : ℤ) :
  let n := t * (t + 1) / 2
  let k := t * (t - 1) / 2
  let total := n + k
  total ≥ 2 →
  (n * (n - 1) + k * (k - 1)) / (total * (total - 1)) = 
  (2 * n * k) / (total * (total - 1)) := by
sorry

end NUMINAMATH_CALUDE_equal_probability_same_different_color_l3517_351768


namespace NUMINAMATH_CALUDE_apples_in_box_l3517_351712

/-- The number of apples in a box -/
def apples_per_box : ℕ := 14

/-- The number of people eating apples -/
def num_people : ℕ := 2

/-- The number of weeks spent eating apples -/
def num_weeks : ℕ := 3

/-- The number of boxes of apples -/
def num_boxes : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of apples eaten per person per day -/
def apples_per_person_per_day : ℕ := 1

theorem apples_in_box :
  apples_per_box * num_boxes = num_people * apples_per_person_per_day * num_weeks * days_per_week :=
by sorry

end NUMINAMATH_CALUDE_apples_in_box_l3517_351712


namespace NUMINAMATH_CALUDE_min_ticket_cost_is_800_l3517_351717

/-- Represents the ticket pricing structure and group composition --/
structure TicketPricing where
  adultPrice : ℕ
  childPrice : ℕ
  groupPrice : ℕ
  groupMinSize : ℕ
  numAdults : ℕ
  numChildren : ℕ

/-- Calculates the minimum cost for tickets given the pricing structure --/
def minTicketCost (pricing : TicketPricing) : ℕ :=
  sorry

/-- Theorem stating that the minimum cost for the given scenario is 800 yuan --/
theorem min_ticket_cost_is_800 :
  let pricing : TicketPricing := {
    adultPrice := 100,
    childPrice := 50,
    groupPrice := 70,
    groupMinSize := 10,
    numAdults := 8,
    numChildren := 4
  }
  minTicketCost pricing = 800 := by sorry

end NUMINAMATH_CALUDE_min_ticket_cost_is_800_l3517_351717


namespace NUMINAMATH_CALUDE_decimal_to_binary_98_l3517_351748

theorem decimal_to_binary_98 : 
  (98 : ℕ).digits 2 = [0, 1, 0, 0, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_98_l3517_351748


namespace NUMINAMATH_CALUDE_ayen_exercise_time_l3517_351730

/-- Represents the total exercise time in minutes for a week -/
def weekly_exercise (
  weekday_jog : ℕ
  ) (tuesday_extra : ℕ) (friday_extra : ℕ) (saturday_jog : ℕ) (sunday_swim : ℕ) : ℚ :=
  let weekday_total := 3 * weekday_jog + (weekday_jog + tuesday_extra) + (weekday_jog + friday_extra)
  let jogging_total := weekday_total + saturday_jog
  let swimming_equivalent := (3 / 2) * sunday_swim
  (jogging_total + swimming_equivalent) / 60

/-- The theorem stating Ayen's total exercise time for the week -/
theorem ayen_exercise_time : 
  weekly_exercise 30 5 25 45 60 = (23 / 4) := by sorry

end NUMINAMATH_CALUDE_ayen_exercise_time_l3517_351730


namespace NUMINAMATH_CALUDE_linear_term_coefficient_l3517_351710

/-- The coefficient of the linear term in the expansion of (x-1)(1/x + x)^6 is 20 -/
theorem linear_term_coefficient : ℕ :=
  20

#check linear_term_coefficient

end NUMINAMATH_CALUDE_linear_term_coefficient_l3517_351710


namespace NUMINAMATH_CALUDE_tan_two_implications_l3517_351787

theorem tan_two_implications (θ : Real) (h : Real.tan θ = 2) : 
  (Real.cos θ)^2 = 1/5 ∧ (Real.sin θ)^2 = 4/5 ∧ 
  (4 * Real.sin θ - 3 * Real.cos θ) / (6 * Real.cos θ + 2 * Real.sin θ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implications_l3517_351787


namespace NUMINAMATH_CALUDE_max_axes_of_symmetry_l3517_351758

/-- A type representing a configuration of segments on a plane -/
structure SegmentConfiguration where
  k : ℕ+  -- number of segments (positive natural number)

/-- The number of axes of symmetry for a given segment configuration -/
def axesOfSymmetry (config : SegmentConfiguration) : ℕ := sorry

/-- Theorem stating that the maximum number of axes of symmetry is 2k -/
theorem max_axes_of_symmetry (config : SegmentConfiguration) :
  ∃ (arrangement : SegmentConfiguration), 
    arrangement.k = config.k ∧ 
    axesOfSymmetry arrangement = 2 * config.k.val ∧
    ∀ (other : SegmentConfiguration), 
      other.k = config.k → 
      axesOfSymmetry other ≤ axesOfSymmetry arrangement :=
by sorry

end NUMINAMATH_CALUDE_max_axes_of_symmetry_l3517_351758


namespace NUMINAMATH_CALUDE_canteen_to_bathroom_ratio_l3517_351724

/-- Represents the number of tables in the classroom -/
def num_tables : ℕ := 6

/-- Represents the number of students currently sitting at each table -/
def students_per_table : ℕ := 3

/-- Represents the number of girls who went to the bathroom -/
def girls_in_bathroom : ℕ := 3

/-- Represents the number of new groups added to the class -/
def new_groups : ℕ := 2

/-- Represents the number of students in each new group -/
def students_per_new_group : ℕ := 4

/-- Represents the number of countries from which foreign exchange students came -/
def num_countries : ℕ := 3

/-- Represents the number of foreign exchange students from each country -/
def students_per_country : ℕ := 3

/-- Represents the total number of students supposed to be in the class -/
def total_students : ℕ := 47

/-- Theorem stating the ratio of students who went to the canteen to girls who went to the bathroom -/
theorem canteen_to_bathroom_ratio :
  let students_present := num_tables * students_per_table
  let new_group_students := new_groups * students_per_new_group
  let foreign_students := num_countries * students_per_country
  let missing_students := girls_in_bathroom + new_group_students + foreign_students
  let canteen_students := total_students - students_present - missing_students
  (canteen_students : ℚ) / girls_in_bathroom = 3 := by
  sorry

end NUMINAMATH_CALUDE_canteen_to_bathroom_ratio_l3517_351724


namespace NUMINAMATH_CALUDE_cubic_real_root_existence_l3517_351722

theorem cubic_real_root_existence (a₀ a₁ a₂ a₃ : ℝ) (ha₀ : a₀ ≠ 0) :
  ∃ x : ℝ, a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_real_root_existence_l3517_351722


namespace NUMINAMATH_CALUDE_angle_twice_complement_l3517_351704

theorem angle_twice_complement (x : ℝ) : 
  (x = 2 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_twice_complement_l3517_351704


namespace NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l3517_351793

theorem smallest_bound_for_cubic_inequality :
  ∃ (M : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 :=
sorry

end NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l3517_351793


namespace NUMINAMATH_CALUDE_horner_method_v2_l3517_351759

def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

def horner_v2 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := 2*x + 1
  v1 * x

theorem horner_method_v2 : horner_v2 2 = 10 := by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l3517_351759


namespace NUMINAMATH_CALUDE_exactly_two_valid_sets_l3517_351757

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)  -- The first integer in the set
  (length : ℕ) -- The number of integers in the set

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set according to our conditions -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive s = 18

/-- The main theorem to prove -/
theorem exactly_two_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ ∀ s ∈ sets, is_valid_set s :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_sets_l3517_351757


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3517_351742

theorem absolute_value_inequality (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3517_351742


namespace NUMINAMATH_CALUDE_logarithm_equation_l3517_351767

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_equation : log10 5 * log10 50 - log10 2 * log10 20 - log10 625 = -2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_l3517_351767


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l3517_351718

/-- 
Given a triangle with two sides of 36 units and 45 units, and the third side being an integer,
the least possible perimeter is 91 units.
-/
theorem least_perimeter_triangle : 
  ∀ (x : ℕ), 
  x > 0 → 
  x + 36 > 45 → 
  x + 45 > 36 → 
  36 + 45 > x → 
  (∀ y : ℕ, y > 0 → y + 36 > 45 → y + 45 > 36 → 36 + 45 > y → x + 36 + 45 ≤ y + 36 + 45) →
  x + 36 + 45 = 91 := by
sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l3517_351718


namespace NUMINAMATH_CALUDE_max_value_quadratic_expression_l3517_351753

theorem max_value_quadratic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  x^2 + 2*x*y + 3*y^2 ≤ 10*(45 + 42*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_expression_l3517_351753


namespace NUMINAMATH_CALUDE_rectangle_product_of_b_values_l3517_351783

theorem rectangle_product_of_b_values :
  ∀ (b₁ b₂ : ℝ),
    (∀ x y : ℝ, (y = 1 ∨ y = 4 ∨ x = 2 ∨ x = b₁) → (y = 1 ∨ y = 4 ∨ x = 2 ∨ x = b₂)) →
    (abs (2 - b₁) = 2 * abs (4 - 1)) →
    (abs (2 - b₂) = 2 * abs (4 - 1)) →
    b₁ * b₂ = -32 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_product_of_b_values_l3517_351783


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3517_351764

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 22

theorem total_cost_calculation : 
  (10 * mango_cost = 24 * rice_cost) → 
  (flour_cost = 2 * rice_cost) → 
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 248.6) := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3517_351764


namespace NUMINAMATH_CALUDE_min_value_of_expression_exists_min_value_l3517_351769

theorem min_value_of_expression (x : ℚ) : (2*x - 5)^2 + 18 ≥ 18 :=
sorry

theorem exists_min_value : ∃ x : ℚ, (2*x - 5)^2 + 18 = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_exists_min_value_l3517_351769


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l3517_351708

def A1 : ℝ × ℝ × ℝ := (3, 10, -1)
def A2 : ℝ × ℝ × ℝ := (-2, 3, -5)
def A3 : ℝ × ℝ × ℝ := (-6, 0, -3)
def A4 : ℝ × ℝ × ℝ := (1, -1, 2)

def tetrahedron_volume (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ := sorry

def tetrahedron_height (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  tetrahedron_volume A1 A2 A3 A4 = 45.5 ∧
  tetrahedron_height A1 A2 A3 A4 = 7 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l3517_351708


namespace NUMINAMATH_CALUDE_mariams_neighborhood_homes_l3517_351728

/-- The number of homes in Mariam's neighborhood -/
def total_homes (homes_one_side : ℕ) (multiplier : ℕ) : ℕ :=
  homes_one_side + multiplier * homes_one_side

/-- Theorem stating the total number of homes in Mariam's neighborhood -/
theorem mariams_neighborhood_homes :
  total_homes 40 3 = 160 := by
  sorry

end NUMINAMATH_CALUDE_mariams_neighborhood_homes_l3517_351728


namespace NUMINAMATH_CALUDE_constant_phi_forms_cone_l3517_351727

/-- A point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The set of points satisfying φ = d -/
def ConstantPhiSet (d : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ = d}

/-- Definition of a cone in spherical coordinates -/
def IsCone (s : Set SphericalPoint) : Prop :=
  ∃ d : ℝ, s = ConstantPhiSet d

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_forms_cone (d : ℝ) :
  IsCone (ConstantPhiSet d) := by
  sorry

end NUMINAMATH_CALUDE_constant_phi_forms_cone_l3517_351727


namespace NUMINAMATH_CALUDE_field_trip_absentees_prove_girls_absent_l3517_351735

/-- Given a field trip scenario, calculate the number of girls who couldn't join. -/
theorem field_trip_absentees (total_students : ℕ) (boys : ℕ) (girls_present : ℕ) : ℕ :=
  let girls_assigned := total_students - boys
  girls_assigned - girls_present

/-- Prove the number of girls who couldn't join the field trip. -/
theorem prove_girls_absent : field_trip_absentees 18 8 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_absentees_prove_girls_absent_l3517_351735


namespace NUMINAMATH_CALUDE_percent_relation_l3517_351743

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.14 * a) 
  (h2 : b = 0.35 * a) : 
  c = 0.4 * b := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3517_351743


namespace NUMINAMATH_CALUDE_fifth_number_15th_row_l3517_351798

def pascal_triangle (n k : ℕ) : ℕ := Nat.choose n k

theorem fifth_number_15th_row : pascal_triangle 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_15th_row_l3517_351798


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3517_351784

/-- If 4x^2 + 12x + a is the square of a binomial, then a = 9 -/
theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 + 12*x + a = (2*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3517_351784


namespace NUMINAMATH_CALUDE_carrie_pants_count_l3517_351747

/-- The number of pairs of pants Carrie bought -/
def pants_count : ℕ := 2

/-- The cost of a single shirt in dollars -/
def shirt_cost : ℕ := 8

/-- The cost of a single pair of pants in dollars -/
def pants_cost : ℕ := 18

/-- The cost of a single jacket in dollars -/
def jacket_cost : ℕ := 60

/-- The number of shirts Carrie bought -/
def shirts_count : ℕ := 4

/-- The number of jackets Carrie bought -/
def jackets_count : ℕ := 2

/-- The amount Carrie paid in dollars -/
def carrie_payment : ℕ := 94

theorem carrie_pants_count :
  shirts_count * shirt_cost + pants_count * pants_cost + jackets_count * jacket_cost = 2 * carrie_payment :=
by sorry

end NUMINAMATH_CALUDE_carrie_pants_count_l3517_351747


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_zero_or_one_l3517_351777

/-- Given a real number a, define the set A as the solutions to ax^2 + 2x + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

/-- Theorem: If A(a) has exactly one element, then a = 0 or a = 1 -/
theorem unique_solution_implies_a_zero_or_one (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_zero_or_one_l3517_351777


namespace NUMINAMATH_CALUDE_horner_method_v3_l3517_351745

def horner_polynomial (x : ℝ) : ℝ := 10 + 25*x - 8*x^2 + x^4 + 6*x^5 + 2*x^6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 6
  let v2 := v1 * x + 1
  v2 * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -36 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3517_351745


namespace NUMINAMATH_CALUDE_chocolate_vanilla_survey_l3517_351790

theorem chocolate_vanilla_survey (total : ℕ) (chocolate : ℕ) (vanilla : ℕ) 
  (h_total : total = 120)
  (h_chocolate : chocolate = 95)
  (h_vanilla : vanilla = 85) :
  (chocolate + vanilla - total : ℕ) ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_vanilla_survey_l3517_351790


namespace NUMINAMATH_CALUDE_sunflower_seed_distribution_l3517_351776

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) 
  (h1 : total_seeds = 54)
  (h2 : num_cans = 9)
  (h3 : total_seeds = num_cans * seeds_per_can) :
  seeds_per_can = 6 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_distribution_l3517_351776


namespace NUMINAMATH_CALUDE_high_school_sample_senior_count_is_160_l3517_351750

theorem high_school_sample (total : ℕ) (junior_percent : ℚ) (not_sophomore_percent : ℚ) 
  (freshman_sophomore_diff : ℕ) : ℕ :=
  let junior_count : ℕ := (junior_percent * total).num.toNat
  let sophomore_count : ℕ := ((1 - not_sophomore_percent) * total).num.toNat
  let freshman_count : ℕ := sophomore_count + freshman_sophomore_diff
  total - (junior_count + sophomore_count + freshman_count)

theorem senior_count_is_160 :
  high_school_sample 800 (27/100) (75/100) 24 = 160 := by
  sorry

end NUMINAMATH_CALUDE_high_school_sample_senior_count_is_160_l3517_351750


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_l3517_351785

theorem jenny_easter_eggs (n : ℕ) : 
  n ∣ 30 ∧ n ∣ 45 ∧ n ≥ 5 → n ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_l3517_351785


namespace NUMINAMATH_CALUDE_negative_root_implies_a_less_than_neg_three_l3517_351726

theorem negative_root_implies_a_less_than_neg_three (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
by sorry

end NUMINAMATH_CALUDE_negative_root_implies_a_less_than_neg_three_l3517_351726


namespace NUMINAMATH_CALUDE_parabola_focus_line_l3517_351762

/-- Given a parabola and a line passing through its focus, prove the value of p -/
theorem parabola_focus_line (p : ℝ) (A B : ℝ × ℝ) : 
  p > 0 →  -- p is positive
  (∀ x y, y = x^2 / (2*p)) →  -- equation of parabola
  (A.1^2 = 2*p*A.2) →  -- A is on the parabola
  (B.1^2 = 2*p*B.2) →  -- B is on the parabola
  (A.1 + B.1 = 2) →  -- midpoint of AB has x-coordinate 1
  ((A.2 + B.2) / 2 = 1) →  -- midpoint of AB has y-coordinate 1
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 →  -- length of AB is 6
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_line_l3517_351762


namespace NUMINAMATH_CALUDE_total_miles_run_l3517_351725

theorem total_miles_run (xavier katie cole lily joe : ℝ) : 
  xavier = 3 * katie → 
  katie = 4 * cole → 
  lily = 5 * cole → 
  joe = 2 * lily → 
  xavier = 84 → 
  lily = 0.85 * joe → 
  xavier + katie + cole + lily + joe = 168.875 := by
sorry

end NUMINAMATH_CALUDE_total_miles_run_l3517_351725


namespace NUMINAMATH_CALUDE_selection_plans_l3517_351738

theorem selection_plans (n m : ℕ) (h1 : n = 6) (h2 : m = 3) : 
  (n.choose m) * m.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_selection_plans_l3517_351738


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_8_l3517_351744

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

theorem largest_even_digit_multiple_of_8 :
  ∃ (n : ℕ), n = 8888 ∧
  has_only_even_digits n ∧
  n < 10000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, has_only_even_digits m → m < 10000 → m % 8 = 0 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_8_l3517_351744
