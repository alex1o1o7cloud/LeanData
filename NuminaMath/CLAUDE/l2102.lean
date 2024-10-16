import Mathlib

namespace NUMINAMATH_CALUDE_select_with_boys_l2102_210252

theorem select_with_boys (num_boys num_girls : ℕ) : 
  num_boys = 6 → num_girls = 4 → 
  (2^(num_boys + num_girls) - 2^num_girls) = 1008 := by
  sorry

end NUMINAMATH_CALUDE_select_with_boys_l2102_210252


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2102_210236

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 - I) / (1 - I)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2102_210236


namespace NUMINAMATH_CALUDE_percent_democrat_voters_l2102_210228

theorem percent_democrat_voters (D R : ℝ) : 
  D + R = 100 →
  0.85 * D + 0.20 * R = 59 →
  D = 60 :=
by sorry

end NUMINAMATH_CALUDE_percent_democrat_voters_l2102_210228


namespace NUMINAMATH_CALUDE_fraction_comparison_l2102_210224

theorem fraction_comparison (x y : ℝ) (h1 : x > y) (h2 : y > 2) :
  y / (y^2 - y + 1) > x / (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2102_210224


namespace NUMINAMATH_CALUDE_irrational_approximation_l2102_210295

theorem irrational_approximation (ξ : ℝ) (h_irrational : Irrational ξ) :
  Set.Infinite {q : ℚ | ∃ (m : ℤ) (n : ℕ), q = m / n ∧ |ξ - (m / n)| < 1 / (Real.sqrt 5 * m^2)} := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l2102_210295


namespace NUMINAMATH_CALUDE_min_max_non_triangle_sequence_l2102_210276

/-- A sequence of 8 integer lengths where no three can form a triangle -/
def NonTriangleSequence : Type := 
  { seq : Fin 8 → ℕ // 
    (∀ i j k, i < j → j < k → k < 8 → seq i + seq j ≤ seq k) ∧
    (∀ i < 8, seq i > 0) }

/-- The minimum of the maximum value in any NonTriangleSequence is 21 -/
theorem min_max_non_triangle_sequence : 
  (⨅ (seq : NonTriangleSequence), ⨆ (i : Fin 8), seq.val i) = 21 := by
  sorry

end NUMINAMATH_CALUDE_min_max_non_triangle_sequence_l2102_210276


namespace NUMINAMATH_CALUDE_solution_set_R_solution_set_m_lower_bound_l2102_210220

-- Define the inequality
def inequality (x m : ℝ) : Prop := x^2 - 2*(m+1)*x + 4*m ≥ 0

-- Statement 1
theorem solution_set_R (m : ℝ) : 
  (∀ x, inequality x m) ↔ m = 1 := by sorry

-- Statement 2
theorem solution_set (m : ℝ) :
  (m = 1 ∧ ∀ x, inequality x m) ∨
  (m > 1 ∧ ∀ x, inequality x m ↔ (x ≤ 2 ∨ x ≥ 2*m)) ∨
  (m < 1 ∧ ∀ x, inequality x m ↔ (x ≤ 2*m ∨ x ≥ 2)) := by sorry

-- Statement 3
theorem m_lower_bound :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → inequality x m) → m ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_R_solution_set_m_lower_bound_l2102_210220


namespace NUMINAMATH_CALUDE_number_problem_l2102_210203

theorem number_problem (A B : ℤ) 
  (h1 : A - B = 144) 
  (h2 : A = 3 * B - 14) : 
  A = 223 := by sorry

end NUMINAMATH_CALUDE_number_problem_l2102_210203


namespace NUMINAMATH_CALUDE_similar_squares_side_length_l2102_210265

theorem similar_squares_side_length 
  (area_ratio : ℚ) 
  (small_side : ℝ) 
  (similar : Bool) 
  (h1 : area_ratio = 1 / 9)
  (h2 : small_side = 4)
  (h3 : similar = true) : 
  ∃ (large_side : ℝ), large_side = 12 := by
sorry

end NUMINAMATH_CALUDE_similar_squares_side_length_l2102_210265


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2102_210249

/-- For a hyperbola with equation x²/9 - y²/m = 1 and eccentricity e = 2, m = 27 -/
theorem hyperbola_eccentricity (x y m : ℝ) (e : ℝ) 
  (h1 : x^2 / 9 - y^2 / m = 1)
  (h2 : e = 2)
  (h3 : e = Real.sqrt (1 + m / 9)) : 
  m = 27 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2102_210249


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2102_210212

theorem reciprocal_sum_theorem (a b c : ℕ+) : 
  a < b ∧ b < c → 
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 → 
  (a : ℕ) + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2102_210212


namespace NUMINAMATH_CALUDE_searchlight_probability_l2102_210256

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℚ := 2

/-- The time in seconds for one complete revolution of the searchlight -/
def revolution_time : ℚ := 60 / revolutions_per_minute

/-- The minimum time in seconds a man needs to stay in the dark -/
def min_dark_time : ℚ := 10

/-- The probability of a man staying in the dark for at least the minimum time -/
def dark_probability : ℚ := min_dark_time / revolution_time

theorem searchlight_probability :
  dark_probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_searchlight_probability_l2102_210256


namespace NUMINAMATH_CALUDE_min_value_theorem_l2102_210290

theorem min_value_theorem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → 1/a' + 4/b' + 9/c' ≥ 9) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 4/b' + 9/c' = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2102_210290


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l2102_210217

theorem polynomial_equation_solution (x : ℝ) : 
  let q : ℝ → ℝ := λ t => 12 * t^3 - 4
  q (x^3) - q (x^3 - 4) = (q x)^2 + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l2102_210217


namespace NUMINAMATH_CALUDE_largest_domain_of_g_l2102_210243

def g_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → g x + g (1 / x) = x^2

theorem largest_domain_of_g :
  ∃! (S : Set ℝ), S.Nonempty ∧
    (∀ T : Set ℝ, (∃ g : ℝ → ℝ, (∀ x ∈ T, x ≠ 0 ∧ g_condition g) → T ⊆ S)) ∧
    S = {-1, 1} :=
  sorry

end NUMINAMATH_CALUDE_largest_domain_of_g_l2102_210243


namespace NUMINAMATH_CALUDE_lineup_count_l2102_210201

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def excluded_players : ℕ := 3

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of valid lineups -/
def valid_lineups : ℕ :=
  3 * choose (team_size - excluded_players) (lineup_size - 1) + 
  choose (team_size - excluded_players) lineup_size

theorem lineup_count : valid_lineups = 2277 := by
  sorry

end NUMINAMATH_CALUDE_lineup_count_l2102_210201


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_seven_l2102_210246

/-- The repeating decimal 0.777... as a rational number -/
def repeating_seven : ℚ := 7 / 9

/-- The reciprocal of the repeating decimal 0.777... -/
def reciprocal_repeating_seven : ℚ := 9 / 7

/-- Theorem stating that the reciprocal of 0.777... is 9/7 -/
theorem reciprocal_of_repeating_seven :
  (repeating_seven)⁻¹ = reciprocal_repeating_seven :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_seven_l2102_210246


namespace NUMINAMATH_CALUDE_jerome_solution_l2102_210211

def jerome_problem (initial_money : ℕ) (given_to_meg : ℕ) (given_to_bianca : ℕ) (money_left : ℕ) : Prop :=
  initial_money / 2 = 43 ∧
  given_to_meg = 8 ∧
  money_left = 54 ∧
  initial_money = given_to_meg + given_to_bianca + money_left ∧
  given_to_bianca / given_to_meg = 3

theorem jerome_solution :
  ∃ (initial_money given_to_meg given_to_bianca money_left : ℕ),
    jerome_problem initial_money given_to_meg given_to_bianca money_left :=
by
  sorry

end NUMINAMATH_CALUDE_jerome_solution_l2102_210211


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l2102_210237

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  2/a + 1/b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l2102_210237


namespace NUMINAMATH_CALUDE_original_paper_sheets_l2102_210250

-- Define the number of sheets per book
def sheets_per_book : ℕ := sorry

-- Define the total number of sheets
def total_sheets : ℕ := 18000

-- Theorem statement
theorem original_paper_sheets :
  (120 * sheets_per_book = (60 : ℕ) * total_sheets / 100) ∧
  (185 * sheets_per_book + 1350 = total_sheets) :=
by sorry

end NUMINAMATH_CALUDE_original_paper_sheets_l2102_210250


namespace NUMINAMATH_CALUDE_most_accurate_reading_is_10_45_l2102_210287

/-- Represents a scientific weighing scale --/
structure ScientificScale where
  smallest_division : ℝ
  lower_bound : ℝ
  upper_bound : ℝ
  marker_position : ℝ

/-- Determines if a given reading is the most accurate for a scientific scale --/
def is_most_accurate_reading (s : ScientificScale) (reading : ℝ) : Prop :=
  s.lower_bound < reading ∧ 
  reading < s.upper_bound ∧ 
  reading % s.smallest_division = 0 ∧
  ∀ r, s.lower_bound < r ∧ r < s.upper_bound ∧ r % s.smallest_division = 0 → 
    |s.marker_position - reading| ≤ |s.marker_position - r|

/-- The theorem stating the most accurate reading for the given scale --/
theorem most_accurate_reading_is_10_45 (s : ScientificScale) 
  (h_division : s.smallest_division = 0.01)
  (h_lower : s.lower_bound = 10.41)
  (h_upper : s.upper_bound = 10.55)
  (h_marker : s.lower_bound < s.marker_position ∧ s.marker_position < (s.lower_bound + s.upper_bound) / 2) :
  is_most_accurate_reading s 10.45 :=
sorry

end NUMINAMATH_CALUDE_most_accurate_reading_is_10_45_l2102_210287


namespace NUMINAMATH_CALUDE_sum_and_opposites_l2102_210240

theorem sum_and_opposites : 
  let a := -5
  let b := -2
  let c := abs b
  let d := 0
  (a + b + c + d = -5) ∧ 
  (- a = 5) ∧ 
  (- b = 2) ∧ 
  (- c = -2) ∧ 
  (- d = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_opposites_l2102_210240


namespace NUMINAMATH_CALUDE_symmetry_line_probability_l2102_210255

/-- Represents a point on a grid --/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a rectangle with a uniform grid --/
structure GridRectangle where
  width : Nat
  height : Nat

/-- The total number of points in the grid rectangle --/
def totalPoints (rect : GridRectangle) : Nat :=
  rect.width * rect.height

/-- The center point of the rectangle --/
def centerPoint (rect : GridRectangle) : GridPoint :=
  { x := rect.width / 2, y := rect.height / 2 }

/-- Checks if a given point is on a line of symmetry --/
def isOnSymmetryLine (p : GridPoint) (center : GridPoint) (rect : GridRectangle) : Bool :=
  p.x = center.x ∨ p.y = center.y

/-- Counts the number of points on lines of symmetry, excluding the center --/
def countSymmetryPoints (rect : GridRectangle) : Nat :=
  rect.width + rect.height - 2

/-- The main theorem --/
theorem symmetry_line_probability (rect : GridRectangle) : 
  rect.width = 10 ∧ rect.height = 10 →
  (countSymmetryPoints rect : Rat) / ((totalPoints rect - 1 : Nat) : Rat) = 2 / 11 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_line_probability_l2102_210255


namespace NUMINAMATH_CALUDE_number_of_boys_l2102_210222

/-- The number of boys in a school, given the number of girls and the difference between boys and girls. -/
theorem number_of_boys (girls : ℕ) (difference : ℕ) : girls = 1225 → difference = 1750 → girls + difference = 2975 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l2102_210222


namespace NUMINAMATH_CALUDE_A_3_2_equals_5_l2102_210261

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 2
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_5 : A 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_A_3_2_equals_5_l2102_210261


namespace NUMINAMATH_CALUDE_algebraic_expressions_values_l2102_210263

noncomputable def a : ℝ := Real.sqrt 5 + 1
noncomputable def b : ℝ := Real.sqrt 5 - 1

theorem algebraic_expressions_values :
  (a^2 * b + a * b^2 = 8 * Real.sqrt 5) ∧ (a^2 - a * b + b^2 = 8) := by sorry

end NUMINAMATH_CALUDE_algebraic_expressions_values_l2102_210263


namespace NUMINAMATH_CALUDE_range_of_a_l2102_210210

def P : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def M (a : ℝ) : Set ℝ := {-a, a}

theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ P := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2102_210210


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2102_210242

theorem solve_linear_equation :
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 :=
by
  use -30
  constructor
  · -- Prove that x = -30 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check solve_linear_equation

end NUMINAMATH_CALUDE_solve_linear_equation_l2102_210242


namespace NUMINAMATH_CALUDE_multiply_ones_seven_l2102_210266

theorem multiply_ones_seven : 1111111 * 1111111 = 1234567654321 := by
  sorry

end NUMINAMATH_CALUDE_multiply_ones_seven_l2102_210266


namespace NUMINAMATH_CALUDE_second_warehouse_more_profitable_l2102_210235

/-- Represents the monthly rent in thousands of rubles -/
def monthly_rent_first : ℝ := 80

/-- Represents the monthly rent in thousands of rubles -/
def monthly_rent_second : ℝ := 20

/-- Represents the probability of the bank repossessing the second warehouse -/
def repossession_probability : ℝ := 0.5

/-- Represents the number of months after which repossession might occur -/
def repossession_month : ℕ := 5

/-- Represents the moving expenses in thousands of rubles -/
def moving_expenses : ℝ := 150

/-- Represents the lease duration in months -/
def lease_duration : ℕ := 12

/-- Calculates the expected cost of renting the second warehouse for one year -/
def expected_cost_second : ℝ :=
  let cost_no_repossession := monthly_rent_second * lease_duration
  let cost_repossession := monthly_rent_second * repossession_month +
                           monthly_rent_first * (lease_duration - repossession_month) +
                           moving_expenses
  (1 - repossession_probability) * cost_no_repossession +
  repossession_probability * cost_repossession

/-- Calculates the cost of renting the first warehouse for one year -/
def cost_first : ℝ := monthly_rent_first * lease_duration

theorem second_warehouse_more_profitable :
  expected_cost_second < cost_first :=
sorry

end NUMINAMATH_CALUDE_second_warehouse_more_profitable_l2102_210235


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2102_210227

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_second_term : a 2 = 1/3)
  (h_eighth_term : a 8 = 27) :
  a 5 = 3 ∨ a 5 = -3 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2102_210227


namespace NUMINAMATH_CALUDE_total_amount_received_l2102_210209

def lottery_winnings : ℚ := 555850
def num_students : ℕ := 500
def fraction : ℚ := 3 / 10000

theorem total_amount_received :
  (lottery_winnings * fraction * num_students : ℚ) = 833775 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_received_l2102_210209


namespace NUMINAMATH_CALUDE_equation_solution_l2102_210244

theorem equation_solution : ∃ x : ℝ, (2 / (x + 3) = 1) ∧ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2102_210244


namespace NUMINAMATH_CALUDE_inequality_proof_l2102_210234

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2102_210234


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l2102_210206

theorem no_real_solutions_for_equation : 
  ¬∃ y : ℝ, (10 - y)^2 = 4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l2102_210206


namespace NUMINAMATH_CALUDE_six_stairs_ways_l2102_210230

/-- The number of ways to climb n stairs, taking 1, 2, or 3 stairs at a time -/
def stairClimbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => stairClimbWays (n + 2) + stairClimbWays (n + 1) + stairClimbWays n

theorem six_stairs_ways :
  stairClimbWays 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_six_stairs_ways_l2102_210230


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2102_210208

def complex_number : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem complex_number_in_second_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2102_210208


namespace NUMINAMATH_CALUDE_zeros_after_one_in_8000_to_50_l2102_210293

theorem zeros_after_one_in_8000_to_50 :
  let n : ℕ := 8000
  let k : ℕ := 50
  let base_ten_factor : ℕ := 3
  n = 8 * (10 ^ base_ten_factor) →
  (∃ m : ℕ, n^k = m * 10^(base_ten_factor * k) ∧ m % 10 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_8000_to_50_l2102_210293


namespace NUMINAMATH_CALUDE_heptagon_sum_l2102_210284

/-- Represents a heptagon with numbers distributed on its sides -/
structure NumberedHeptagon where
  /-- Total number of circles in the heptagon -/
  total_circles : Nat
  /-- Number of circles on each side of the heptagon -/
  circles_per_side : Nat
  /-- Total number of sides in the heptagon -/
  sides : Nat
  /-- The sum of all numbers distributed in the heptagon -/
  total_sum : Nat
  /-- The sum of numbers 1 to 7 -/
  sum_1_to_7 : Nat
  /-- Condition: Total circles is the product of circles per side and number of sides -/
  h_total : total_circles = circles_per_side * sides
  /-- Condition: The heptagon has 7 sides -/
  h_heptagon : sides = 7
  /-- Condition: Each side has 3 circles -/
  h_three_per_side : circles_per_side = 3
  /-- Condition: The total sum is the sum of numbers 1 to 14 plus the sum of 1 to 7 -/
  h_sum : total_sum = (14 * 15) / 2 + sum_1_to_7
  /-- Condition: The sum of numbers 1 to 7 -/
  h_sum_1_to_7 : sum_1_to_7 = (7 * 8) / 2

/-- Theorem: The sum of numbers in each line of three circles is 19 -/
theorem heptagon_sum (h : NumberedHeptagon) : h.total_sum / h.sides = 19 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_sum_l2102_210284


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l2102_210270

theorem divisibility_implies_multiple_of_three (a b : ℤ) :
  (9 : ℤ) ∣ (a^2 + a*b + b^2) → (3 : ℤ) ∣ a ∧ (3 : ℤ) ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l2102_210270


namespace NUMINAMATH_CALUDE_multiplication_equation_solution_l2102_210264

theorem multiplication_equation_solution (x : ℚ) : x * (-2/3) = 2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_solution_l2102_210264


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l2102_210239

theorem salary_increase_percentage (initial_salary final_salary : ℝ) 
  (h1 : initial_salary = 1000.0000000000001)
  (h2 : final_salary = 1045) :
  ∃ P : ℝ, 
    (P = 10) ∧ 
    (final_salary = initial_salary * (1 + P / 100) * (1 - 5 / 100)) := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l2102_210239


namespace NUMINAMATH_CALUDE_rachel_problem_solving_time_l2102_210273

/-- The number of minutes Rachel spent solving math problems before bed -/
def minutes_before_bed : ℕ := 12

/-- The number of problems Rachel solved per minute before bed -/
def problems_per_minute : ℕ := 5

/-- The number of problems Rachel solved the next day -/
def problems_next_day : ℕ := 16

/-- The total number of problems Rachel solved -/
def total_problems : ℕ := 76

/-- Theorem stating that Rachel spent 12 minutes solving problems before bed -/
theorem rachel_problem_solving_time :
  minutes_before_bed * problems_per_minute + problems_next_day = total_problems :=
by sorry

end NUMINAMATH_CALUDE_rachel_problem_solving_time_l2102_210273


namespace NUMINAMATH_CALUDE_perfect_square_values_l2102_210225

theorem perfect_square_values (a n : ℕ) : 
  (a ^ 2 + a + 1589 = n ^ 2) ↔ (a = 1588 ∨ a = 28 ∨ a = 316 ∨ a = 43) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_values_l2102_210225


namespace NUMINAMATH_CALUDE_differential_of_y_l2102_210282

noncomputable section

open Real

-- Define the function y
def y (x : ℝ) : ℝ := x * (sin (log x) - cos (log x))

-- State the theorem
theorem differential_of_y (x : ℝ) (h : x > 0) :
  deriv y x = 2 * sin (log x) :=
by sorry

end

end NUMINAMATH_CALUDE_differential_of_y_l2102_210282


namespace NUMINAMATH_CALUDE_smores_group_size_l2102_210299

/-- Given the conditions for S'mores supplies, prove the number of people in the group. -/
theorem smores_group_size :
  ∀ (smores_per_person : ℕ) 
    (cost_per_set : ℕ) 
    (smores_per_set : ℕ) 
    (total_cost : ℕ),
  smores_per_person = 3 →
  cost_per_set = 3 →
  smores_per_set = 4 →
  total_cost = 18 →
  (total_cost / cost_per_set) * smores_per_set / smores_per_person = 8 :=
by sorry

end NUMINAMATH_CALUDE_smores_group_size_l2102_210299


namespace NUMINAMATH_CALUDE_darwin_money_left_l2102_210285

theorem darwin_money_left (initial_amount : ℚ) (gas_fraction : ℚ) (food_fraction : ℚ) 
  (h1 : initial_amount = 600)
  (h2 : gas_fraction = 1/3)
  (h3 : food_fraction = 1/4) :
  initial_amount - (gas_fraction * initial_amount) - (food_fraction * (initial_amount - (gas_fraction * initial_amount))) = 300 := by
  sorry

end NUMINAMATH_CALUDE_darwin_money_left_l2102_210285


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2102_210271

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2102_210271


namespace NUMINAMATH_CALUDE_sqrt_seven_bounds_l2102_210202

theorem sqrt_seven_bounds : 2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_bounds_l2102_210202


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l2102_210200

theorem sum_abcd_equals_negative_ten_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 6) : 
  a + b + c + d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l2102_210200


namespace NUMINAMATH_CALUDE_students_left_in_classroom_l2102_210207

theorem students_left_in_classroom 
  (total_students : ℕ) 
  (painting_fraction : ℚ) 
  (playing_fraction : ℚ) 
  (h1 : total_students = 50) 
  (h2 : painting_fraction = 3/5) 
  (h3 : playing_fraction = 1/5) : 
  total_students - (painting_fraction * total_students + playing_fraction * total_students) = 10 := by
sorry

end NUMINAMATH_CALUDE_students_left_in_classroom_l2102_210207


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2102_210277

/-- Two vectors in ℝ² -/
def Vector2 := ℝ × ℝ

/-- The dot product of two vectors in ℝ² -/
def dot_product (v w : Vector2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2) : Prop :=
  dot_product v w = 0

theorem perpendicular_lines (b : ℝ) :
  perpendicular (4, -5) (b, 3) → b = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2102_210277


namespace NUMINAMATH_CALUDE_geometric_sequence_150th_term_l2102_210269

/-- Given a geometric sequence with first term 5 and second term -10,
    the 150th term is equal to -5 * 2^149 -/
theorem geometric_sequence_150th_term :
  let a₁ : ℝ := 5
  let a₂ : ℝ := -10
  let r : ℝ := a₂ / a₁
  let a₁₅₀ : ℝ := a₁ * r^149
  a₁₅₀ = -5 * 2^149 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_150th_term_l2102_210269


namespace NUMINAMATH_CALUDE_sum_of_sides_l2102_210205

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The sum of sides of a triangle, square, and hexagon is 13 -/
theorem sum_of_sides : triangle_sides + square_sides + hexagon_sides = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sides_l2102_210205


namespace NUMINAMATH_CALUDE_completing_square_transformation_l2102_210283

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 8*x + 2 = 0) ↔ ((x - 4)^2 = 14) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l2102_210283


namespace NUMINAMATH_CALUDE_library_books_total_l2102_210247

theorem library_books_total (initial_books : ℕ) (additional_books : ℕ) : 
  initial_books = 54 → additional_books = 23 → initial_books + additional_books = 77 := by
sorry

end NUMINAMATH_CALUDE_library_books_total_l2102_210247


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2102_210268

/-- The line (2k-1)x-(k+3)y-(k-11)=0 passes through the point (2, 3) for all values of k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2102_210268


namespace NUMINAMATH_CALUDE_total_amount_to_pay_l2102_210288

def original_balance : ℝ := 150
def finance_charge_percentage : ℝ := 0.02

theorem total_amount_to_pay : 
  original_balance * (1 + finance_charge_percentage) = 153 := by sorry

end NUMINAMATH_CALUDE_total_amount_to_pay_l2102_210288


namespace NUMINAMATH_CALUDE_convention_handshakes_l2102_210229

/-- The number of handshakes at the Annual Mischief Convention --/
def annual_mischief_convention_handshakes (num_gremlins : ℕ) (num_imps : ℕ) : ℕ :=
  let gremlin_handshakes := num_gremlins.choose 2
  let imp_gremlin_handshakes := num_imps * num_gremlins
  gremlin_handshakes + imp_gremlin_handshakes

/-- Theorem stating the number of handshakes at the Annual Mischief Convention --/
theorem convention_handshakes :
  annual_mischief_convention_handshakes 25 20 = 800 := by
  sorry

#eval annual_mischief_convention_handshakes 25 20

end NUMINAMATH_CALUDE_convention_handshakes_l2102_210229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2102_210275

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_6th : a 6 = 9)
  (h_3rd : a 3 = 3 * a 2) :
  a 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2102_210275


namespace NUMINAMATH_CALUDE_max_speed_is_four_l2102_210226

/-- Represents the scenario of two pedestrians traveling between points A and B. -/
structure PedestrianScenario where
  route1_length : ℝ
  route2_length : ℝ
  first_section_length : ℝ
  time_difference : ℝ
  speed_difference : ℝ

/-- Calculates the maximum average speed of the first pedestrian on the second section. -/
def max_average_speed (scenario : PedestrianScenario) : ℝ :=
  4 -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the maximum average speed is 4 km/h given the scenario conditions. -/
theorem max_speed_is_four (scenario : PedestrianScenario) 
  (h1 : scenario.route1_length = 19)
  (h2 : scenario.route2_length = 12)
  (h3 : scenario.first_section_length = 11)
  (h4 : scenario.time_difference = 2)
  (h5 : scenario.speed_difference = 0.5) :
  max_average_speed scenario = 4 := by
  sorry

#check max_speed_is_four

end NUMINAMATH_CALUDE_max_speed_is_four_l2102_210226


namespace NUMINAMATH_CALUDE_magnitude_2a_minus_b_l2102_210297

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem magnitude_2a_minus_b : ‖(2 • a) - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_2a_minus_b_l2102_210297


namespace NUMINAMATH_CALUDE_casino_game_max_guaranteed_money_l2102_210291

/-- Represents the outcome of a single bet -/
inductive BetOutcome
| Win
| Lose

/-- Represents the state of the game after each bet -/
structure GameState :=
  (money : ℕ)
  (bets_made : ℕ)
  (consecutive_losses : ℕ)

/-- The betting strategy function type -/
def BettingStrategy := GameState → ℕ

/-- The game rules function type -/
def GameRules := GameState → BetOutcome → GameState

theorem casino_game_max_guaranteed_money 
  (initial_money : ℕ) 
  (max_bets : ℕ) 
  (max_bet_amount : ℕ) 
  (consolation_win_threshold : ℕ) 
  (strategy : BettingStrategy) 
  (rules : GameRules) :
  initial_money = 100 →
  max_bets = 5 →
  max_bet_amount = 17 →
  consolation_win_threshold = 4 →
  ∃ (final_money : ℕ), final_money ≥ 98 ∧
    ∀ (outcomes : List BetOutcome), 
      outcomes.length = max_bets →
      let final_state := outcomes.foldl rules { money := initial_money, bets_made := 0, consecutive_losses := 0 }
      final_state.money ≥ final_money :=
by sorry

end NUMINAMATH_CALUDE_casino_game_max_guaranteed_money_l2102_210291


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l2102_210286

/-- Represents the profit share calculation for a business partnership. -/
structure ProfitShare where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of A and C. -/
def profit_share_difference (ps : ProfitShare) : ℕ :=
  let total_ratio := ps.a_investment + ps.b_investment + ps.c_investment
  let unit_profit := ps.b_profit * total_ratio / ps.b_investment
  let a_profit := unit_profit * ps.a_investment / total_ratio
  let c_profit := unit_profit * ps.c_investment / total_ratio
  c_profit - a_profit

/-- Theorem stating the difference in profit shares for the given scenario. -/
theorem profit_share_difference_example :
  profit_share_difference ⟨8000, 10000, 12000, 2000⟩ = 800 := by
  sorry


end NUMINAMATH_CALUDE_profit_share_difference_example_l2102_210286


namespace NUMINAMATH_CALUDE_eight_T_three_equals_fifty_l2102_210294

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem statement
theorem eight_T_three_equals_fifty : T 8 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_eight_T_three_equals_fifty_l2102_210294


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2102_210260

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y - z = 0 →
  4 * x + 2 * k * y + 3 * z = 0 →
  3 * x + 6 * y + 2 * z = 0 →
  x * z / (y^2) = 1368 / 25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2102_210260


namespace NUMINAMATH_CALUDE_move_right_2_units_l2102_210232

/-- Moving a point to the right in a 2D coordinate system -/
def move_right (x y dx : ℝ) : ℝ × ℝ :=
  (x + dx, y)

theorem move_right_2_units :
  let A : ℝ × ℝ := (1, 2)
  let A' : ℝ × ℝ := move_right A.1 A.2 2
  A' = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_move_right_2_units_l2102_210232


namespace NUMINAMATH_CALUDE_book_club_unique_books_l2102_210278

theorem book_club_unique_books (tony dean breanna piper asher : ℕ)
  (tony_dean breanna_piper_asher dean_piper_tony asher_breanna_tony all_five : ℕ)
  (h_tony : tony = 23)
  (h_dean : dean = 20)
  (h_breanna : breanna = 30)
  (h_piper : piper = 26)
  (h_asher : asher = 25)
  (h_tony_dean : tony_dean = 5)
  (h_breanna_piper_asher : breanna_piper_asher = 6)
  (h_dean_piper_tony : dean_piper_tony = 4)
  (h_asher_breanna_tony : asher_breanna_tony = 3)
  (h_all_five : all_five = 2) :
  tony + dean + breanna + piper + asher -
  ((tony_dean - all_five) + (breanna_piper_asher - all_five) +
   (dean_piper_tony - all_five) + (asher_breanna_tony - all_five) + all_five) = 112 :=
by sorry

end NUMINAMATH_CALUDE_book_club_unique_books_l2102_210278


namespace NUMINAMATH_CALUDE_arithmetic_operations_correctness_l2102_210221

theorem arithmetic_operations_correctness :
  ((-2 : ℤ) + 8 ≠ 10) ∧
  ((-1 : ℤ) - 3 = -4) ∧
  ((-2 : ℤ) * 2 ≠ 4) ∧
  ((-8 : ℚ) / (-1) ≠ -1/8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_correctness_l2102_210221


namespace NUMINAMATH_CALUDE_hyperbola_point_comparison_l2102_210213

theorem hyperbola_point_comparison 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2023 / x₁) 
  (h2 : y₂ = 2023 / x₂) 
  (h3 : y₁ > y₂) 
  (h4 : y₂ > 0) : 
  x₁ < x₂ := by
sorry

end NUMINAMATH_CALUDE_hyperbola_point_comparison_l2102_210213


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2102_210253

theorem sum_of_roots_equals_one :
  ∀ x y : ℝ, (x + 3) * (x - 4) = 18 ∧ (y + 3) * (y - 4) = 18 → x + y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2102_210253


namespace NUMINAMATH_CALUDE_museum_trip_ratio_l2102_210289

theorem museum_trip_ratio : 
  let total_people : ℕ := 123
  let num_boys : ℕ := 50
  let num_staff : ℕ := 3  -- driver, assistant, and teacher
  let num_girls : ℕ := total_people - num_boys - num_staff
  (num_girls > num_boys) →
  (num_girls - num_boys : ℚ) / num_boys = 21 / 50 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_trip_ratio_l2102_210289


namespace NUMINAMATH_CALUDE_total_oil_needed_l2102_210267

def oil_for_wheels : ℕ := 2 * 15
def oil_for_chain : ℕ := 10
def oil_for_pedals : ℕ := 5
def oil_for_brakes : ℕ := 8

theorem total_oil_needed : 
  oil_for_wheels + oil_for_chain + oil_for_pedals + oil_for_brakes = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_needed_l2102_210267


namespace NUMINAMATH_CALUDE_power_multiplication_l2102_210262

theorem power_multiplication (x : ℝ) : x^4 * x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2102_210262


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2102_210258

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = π ∧  -- sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  a = 1 ∧  -- given condition
  a * Real.cos C + 0.5 * c = b →  -- given condition
  let l := a + b + c  -- perimeter definition
  2 < l ∧ l ≤ 3 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2102_210258


namespace NUMINAMATH_CALUDE_carnation_bouquets_l2102_210274

theorem carnation_bouquets (b1 b2 b3 : ℝ) (total_bouquets : ℕ) (avg : ℝ) :
  b1 = 9.5 →
  b2 = 14.25 →
  b3 = 18.75 →
  total_bouquets = 6 →
  avg = 16 →
  ∃ b4 b5 b6 : ℝ, b4 + b5 + b6 = total_bouquets * avg - (b1 + b2 + b3) ∧
                  b4 + b5 + b6 = 53.5 :=
by sorry

end NUMINAMATH_CALUDE_carnation_bouquets_l2102_210274


namespace NUMINAMATH_CALUDE_circle_symmetry_l2102_210248

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y + 19 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 5 = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (C1 C2 l : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), C1 x1 y1 ∧ C2 x2 y2 ∧
  (x1 + x2) / 2 + 2 * ((y1 + y2) / 2) - 5 = 0 ∧
  (y2 - y1) / (x2 - x1) * (-1/2) = -1

-- Define circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_line C1 C2 l → ∀ x y, C2 x y ↔ x^2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2102_210248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2102_210280

theorem arithmetic_sequence_length (a₁ l d : ℕ) (h : l = a₁ + (n - 1) * d) :
  a₁ = 4 → l = 205 → d = 3 → n = 68 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2102_210280


namespace NUMINAMATH_CALUDE_orthocenter_property_l2102_210292

/-- An acute-angled triangle with its orthocenter properties -/
structure AcuteTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Altitude lengths
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  -- Distances from vertices to orthocenter
  d_a : ℝ
  d_b : ℝ
  d_c : ℝ
  -- Conditions
  acute : a > 0 ∧ b > 0 ∧ c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angles : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- The orthocenter property for acute-angled triangles -/
theorem orthocenter_property (t : AcuteTriangle) :
  t.m_a * t.d_a + t.m_b * t.d_b + t.m_c * t.d_c = (t.a^2 + t.b^2 + t.c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_property_l2102_210292


namespace NUMINAMATH_CALUDE_sum_f_positive_l2102_210296

def f (x : ℝ) : ℝ := x^5 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l2102_210296


namespace NUMINAMATH_CALUDE_bank_account_balances_l2102_210257

/-- Calculates the final balances of two bank accounts after a series of transactions --/
theorem bank_account_balances
  (primary_initial : ℝ)
  (secondary_initial : ℝ)
  (primary_deposit : ℝ)
  (secondary_deposit : ℝ)
  (primary_spend : ℝ)
  (save_percentage : ℝ)
  (h1 : primary_initial = 3179.37)
  (h2 : secondary_initial = 1254.12)
  (h3 : primary_deposit = 21.85)
  (h4 : secondary_deposit = 150)
  (h5 : primary_spend = 87.41)
  (h6 : save_percentage = 0.15)
  : ∃ (primary_available secondary_final : ℝ),
    primary_available = 2646.74 ∧
    secondary_final = 1404.12 := by
  sorry


end NUMINAMATH_CALUDE_bank_account_balances_l2102_210257


namespace NUMINAMATH_CALUDE_calcium_bromide_weight_l2102_210233

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def bromine_weight : ℝ := 79.904

/-- The number of moles of calcium bromide -/
def moles : ℝ := 4

/-- The molecular weight of calcium bromide (CaBr2) in g/mol -/
def molecular_weight_CaBr2 : ℝ := calcium_weight + 2 * bromine_weight

/-- The total weight of the given number of moles of calcium bromide in grams -/
def total_weight : ℝ := moles * molecular_weight_CaBr2

theorem calcium_bromide_weight : total_weight = 799.552 := by
  sorry

end NUMINAMATH_CALUDE_calcium_bromide_weight_l2102_210233


namespace NUMINAMATH_CALUDE_income_comparison_l2102_210215

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : mart = 1.6 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.96 * juan := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l2102_210215


namespace NUMINAMATH_CALUDE_quality_difference_confidence_l2102_210272

/-- Data for machine production --/
structure MachineData where
  first_class : ℕ
  second_class : ℕ

/-- Calculate K² statistic --/
def calculate_k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the confidence level in quality difference --/
theorem quality_difference_confidence
  (machine_a machine_b : MachineData)
  (h_total : machine_a.first_class + machine_a.second_class = 200)
  (h_total_b : machine_b.first_class + machine_b.second_class = 200)
  (h_a_first : machine_a.first_class = 150)
  (h_b_first : machine_b.first_class = 120) :
  calculate_k_squared machine_a.first_class machine_a.second_class
                      machine_b.first_class machine_b.second_class > 6635 / 1000 :=
sorry

end NUMINAMATH_CALUDE_quality_difference_confidence_l2102_210272


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l2102_210218

/-- Given that (1-2x)^5 = a₀ + a₁x + a₂x² + a₃x³ + a₄x⁴ + a₅x⁵,
    prove the following statements about the coefficients. -/
theorem polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ = 1 ∧ 
   a₁ + a₂ + a₃ + a₄ + a₅ = -2 ∧
   a₁ + a₃ + a₅ = -122) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l2102_210218


namespace NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l2102_210241

theorem opposite_signs_abs_sum_less_abs_diff (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l2102_210241


namespace NUMINAMATH_CALUDE_factor_expression_l2102_210254

theorem factor_expression (x : ℝ) : 2*x*(x+3) + (x+3) = (2*x+1)*(x+3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2102_210254


namespace NUMINAMATH_CALUDE_function_maximum_condition_l2102_210245

open Real

theorem function_maximum_condition (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (1/2) * Real.exp (2*x) + (a - Real.exp 1) * Real.exp x - a * Real.exp 1 + b
  (∀ x, f x ≤ f 1) → a < -Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_function_maximum_condition_l2102_210245


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2102_210279

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2102_210279


namespace NUMINAMATH_CALUDE_sum_of_digits_N_l2102_210281

/-- The sum of digits function for natural numbers -/
noncomputable def sumOfDigits (n : ℕ) : ℕ := sorry

/-- N is defined as the positive integer whose square is 36^49 * 49^36 * 81^25 -/
noncomputable def N : ℕ := sorry

/-- Theorem stating that the sum of digits of N is 21 -/
theorem sum_of_digits_N : sumOfDigits N = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_N_l2102_210281


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l2102_210214

theorem power_of_three_mod_eight : 3^1234 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l2102_210214


namespace NUMINAMATH_CALUDE_geometric_sequence_a9_l2102_210251

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a9 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = 3 → a 6 = 9 → a 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a9_l2102_210251


namespace NUMINAMATH_CALUDE_G_n_planarity_l2102_210216

/-- A graph G_n where vertices are integers from 1 to n -/
def G_n (n : ℕ) := {v : ℕ // v ≤ n}

/-- Two vertices are connected if and only if their sum is prime -/
def connected (n : ℕ) (a b : G_n n) : Prop :=
  Nat.Prime (a.val + b.val)

/-- The graph G_n is planar -/
def is_planar (n : ℕ) : Prop :=
  ∃ (f : G_n n → ℝ × ℝ), ∀ (a b c d : G_n n),
    a ≠ b ∧ c ≠ d ∧ connected n a b ∧ connected n c d →
    (f a ≠ f c ∨ f b ≠ f d) ∧ (f a ≠ f d ∨ f b ≠ f c)

/-- The main theorem: G_n is planar if and only if n ≤ 8 -/
theorem G_n_planarity (n : ℕ) : is_planar n ↔ n ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_G_n_planarity_l2102_210216


namespace NUMINAMATH_CALUDE_sum_of_distances_less_than_diagonal_l2102_210219

-- Define the quadrilateral ABCD and point P
variable (A B C D P : ℝ × ℝ)

-- Define the conditions
variable (h1 : IsConvex A B C D)
variable (h2 : dist A B = dist C D)
variable (h3 : IsInside P A B C D)
variable (h4 : angle P B A + angle P C D = π)

-- State the theorem
theorem sum_of_distances_less_than_diagonal :
  dist P B + dist P C < dist A D :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_less_than_diagonal_l2102_210219


namespace NUMINAMATH_CALUDE_smallest_number_with_same_factors_l2102_210223

def alice_number : Nat := 30

-- Bob's number must have all prime factors of Alice's number
def has_all_prime_factors (m n : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → (p ∣ n → p ∣ m)

-- The theorem to prove
theorem smallest_number_with_same_factors (n : Nat) (h : n = alice_number) :
  ∃ m : Nat, has_all_prime_factors m n ∧ 
  (∀ k : Nat, has_all_prime_factors k n → m ≤ k) ∧
  m = n :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_factors_l2102_210223


namespace NUMINAMATH_CALUDE_smaug_hoard_theorem_l2102_210204

/-- Calculates the total value of Smaug's hoard in copper coins -/
def smaug_hoard_value : ℕ :=
  let gold_coins : ℕ := 100
  let silver_coins : ℕ := 60
  let copper_coins : ℕ := 33
  let silver_to_copper : ℕ := 8
  let gold_to_silver : ℕ := 3
  
  let gold_value : ℕ := gold_coins * gold_to_silver * silver_to_copper
  let silver_value : ℕ := silver_coins * silver_to_copper
  let total_value : ℕ := gold_value + silver_value + copper_coins
  
  total_value

theorem smaug_hoard_theorem : smaug_hoard_value = 2913 := by
  sorry

end NUMINAMATH_CALUDE_smaug_hoard_theorem_l2102_210204


namespace NUMINAMATH_CALUDE_prob_white_after_transfer_l2102_210238

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  black : ℕ

/-- The probability of drawing a white ball from a bag -/
def prob_white (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.black)

theorem prob_white_after_transfer : 
  let bag_a := Bag.mk 4 6
  let bag_b := Bag.mk 4 5
  let new_bag_b := Bag.mk (bag_b.white + 1) bag_b.black
  prob_white new_bag_b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_after_transfer_l2102_210238


namespace NUMINAMATH_CALUDE_probability_AC_adjacent_given_AB_adjacent_l2102_210298

def num_students : ℕ := 5

def total_arrangements_AB_adjacent : ℕ := 48

def arrangements_ABC_adjacent : ℕ := 12

theorem probability_AC_adjacent_given_AB_adjacent :
  (arrangements_ABC_adjacent : ℚ) / total_arrangements_AB_adjacent = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_AC_adjacent_given_AB_adjacent_l2102_210298


namespace NUMINAMATH_CALUDE_unsprinkled_bricks_count_l2102_210231

/-- Represents a rectangular solid pile of bricks -/
structure BrickPile where
  length : Nat
  width : Nat
  height : Nat

/-- Calculates the number of bricks not sprinkled with lime water -/
def unsprinkledBricks (pile : BrickPile) : Nat :=
  pile.length * pile.width * pile.height - 
  (pile.length - 2) * (pile.width - 2) * (pile.height - 2)

/-- Theorem stating that the number of unsprinkled bricks in a 30x20x10 pile is 4032 -/
theorem unsprinkled_bricks_count :
  let pile : BrickPile := { length := 30, width := 20, height := 10 }
  unsprinkledBricks pile = 4032 := by
  sorry

end NUMINAMATH_CALUDE_unsprinkled_bricks_count_l2102_210231


namespace NUMINAMATH_CALUDE_minimal_polynomial_with_roots_l2102_210259

/-- The polynomial we're proving is correct -/
def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 14*x^2 + 8*x - 3

/-- A root of a polynomial -/
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

/-- A polynomial with rational coefficients -/
def has_rational_coeffs (p : ℝ → ℝ) : Prop := 
  ∃ (a b c d e : ℚ), ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e

theorem minimal_polynomial_with_roots : 
  (is_root f (2 + Real.sqrt 3)) ∧ 
  (is_root f (2 + Real.sqrt 5)) ∧ 
  (has_rational_coeffs f) ∧
  (∀ g : ℝ → ℝ, has_rational_coeffs g → is_root g (2 + Real.sqrt 3) → 
    is_root g (2 + Real.sqrt 5) → (∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * g x) → 
    (∃ n : ℕ, ∀ x, g x = (f x) * x^n)) := 
sorry

end NUMINAMATH_CALUDE_minimal_polynomial_with_roots_l2102_210259
