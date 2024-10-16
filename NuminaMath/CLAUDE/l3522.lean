import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_remainder_l3522_352207

theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 6) (h2 : p 4 = 10) :
  ∃ (q r : ℝ → ℝ), (∀ x, p x = q x * ((x - 2) * (x - 4)) + r x) ∧
                    (∃ a b : ℝ, ∀ x, r x = a * x + b) ∧
                    (∀ x, r x = 2 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3522_352207


namespace NUMINAMATH_CALUDE_solve_equation_l3522_352274

theorem solve_equation : ∃ x : ℝ, 10 * x - (2 * 1.5 / 0.3) = 50 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3522_352274


namespace NUMINAMATH_CALUDE_satisfying_polynomial_form_l3522_352212

/-- A polynomial that satisfies the given equation for all real numbers a, b, c 
    such that ab + bc + ca = 0 -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a * b + b * c + c * a = 0 → 
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- Theorem stating the form of polynomials satisfying the equation -/
theorem satisfying_polynomial_form (P : ℝ → ℝ) :
  SatisfyingPolynomial P →
  ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^4 + b * x^2 := by
  sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_form_l3522_352212


namespace NUMINAMATH_CALUDE_parkway_elementary_boys_l3522_352234

theorem parkway_elementary_boys (total_students : ℕ) (soccer_players : ℕ) (girls_not_playing : ℕ)
  (h1 : total_students = 470)
  (h2 : soccer_players = 250)
  (h3 : girls_not_playing = 135)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑⌊(86 : ℚ) / 100 * soccer_players⌋) :
  total_students - (girls_not_playing + (soccer_players - ⌊(86 : ℚ) / 100 * soccer_players⌋)) = 300 :=
by sorry

end NUMINAMATH_CALUDE_parkway_elementary_boys_l3522_352234


namespace NUMINAMATH_CALUDE_domain_of_g_l3522_352209

-- Define the function f with domain [-3, 1]
def f : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

-- Define the function g(x) = f(x) + f(-x)
def g (x : ℝ) : Prop := x ∈ f ∧ (-x) ∈ f

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3522_352209


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3522_352240

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 + 2 * x = 3528 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3522_352240


namespace NUMINAMATH_CALUDE_new_average_rent_l3522_352294

/-- Calculates the new average rent per person after one person's rent is increased -/
theorem new_average_rent (num_friends : ℕ) (initial_average : ℚ) (increased_rent : ℚ) (increase_percentage : ℚ) : 
  num_friends = 4 →
  initial_average = 800 →
  increased_rent = 1250 →
  increase_percentage = 16 / 100 →
  (num_friends * initial_average - increased_rent + increased_rent * (1 + increase_percentage)) / num_friends = 850 :=
by sorry

end NUMINAMATH_CALUDE_new_average_rent_l3522_352294


namespace NUMINAMATH_CALUDE_triangle_in_circle_and_polygon_l3522_352218

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the regular polygon
structure RegularPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ

def is_inscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

def are_adjacent_vertices (p1 p2 : ℝ × ℝ) (poly : RegularPolygon) : Prop :=
  sorry

theorem triangle_in_circle_and_polygon (t : Triangle) (c : Circle) (poly : RegularPolygon) :
  is_inscribed t c →
  angle t.B t.A t.C = angle t.C t.A t.B →
  angle t.B t.A t.C = 3 * angle t.A t.B t.C →
  are_adjacent_vertices t.B t.C poly →
  is_inscribed (Triangle.mk t.A t.B t.C) c →
  poly.n = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_in_circle_and_polygon_l3522_352218


namespace NUMINAMATH_CALUDE_problem_statement_l3522_352244

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) :
  (x - 3)^2 + 36/(x - 3)^2 = 12.375 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3522_352244


namespace NUMINAMATH_CALUDE_probability_of_five_consecutive_heads_l3522_352297

/-- Represents a sequence of 8 coin flips -/
def CoinFlipSequence := Fin 8 → Bool

/-- Returns true if the given sequence has at least 5 consecutive heads -/
def hasAtLeastFiveConsecutiveHeads (seq : CoinFlipSequence) : Bool :=
  sorry

/-- The total number of possible outcomes when flipping a coin 8 times -/
def totalOutcomes : Nat := 2^8

/-- The number of outcomes with at least 5 consecutive heads -/
def successfulOutcomes : Nat := 13

theorem probability_of_five_consecutive_heads :
  (Nat.card {seq : CoinFlipSequence | hasAtLeastFiveConsecutiveHeads seq} : ℚ) / totalOutcomes = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_five_consecutive_heads_l3522_352297


namespace NUMINAMATH_CALUDE_expression_simplification_l3522_352290

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3522_352290


namespace NUMINAMATH_CALUDE_turtle_distribution_theorem_l3522_352287

/-- The ratio of turtles received by Marion, Martha, and Martin -/
def turtle_ratio : Fin 3 → ℕ
| 0 => 3  -- Marion
| 1 => 2  -- Martha
| 2 => 1  -- Martin

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The total number of turtles received by all three -/
def total_turtles : ℕ := martha_turtles * (turtle_ratio 0 + turtle_ratio 1 + turtle_ratio 2) / turtle_ratio 1

theorem turtle_distribution_theorem : total_turtles = 120 := by
  sorry

end NUMINAMATH_CALUDE_turtle_distribution_theorem_l3522_352287


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3522_352203

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3522_352203


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3522_352277

theorem tangent_perpendicular_to_line (a b : ℝ) : 
  b = a^3 →                             -- point (a, b) is on the curve y = x^3
  (3 * a^2) * (-1/3) = -1 →             -- tangent is perpendicular to x + 3y + 1 = 0
  a = 1 ∨ a = -1 :=                     -- conclusion: a = 1 or a = -1
by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3522_352277


namespace NUMINAMATH_CALUDE_min_value_ratio_l3522_352243

theorem min_value_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ d₁ : ℝ, ∀ n, a (n + 1) = a n + d₁) →
  (∃ d₂ : ℝ, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d₂) →
  (∀ n, S n = (n * (a 1 + a n)) / 2) →
  (∀ n, (S (n + 10)) / (a n) ≥ 21) ∧
  (∃ n, (S (n + 10)) / (a n) = 21) :=
sorry

end NUMINAMATH_CALUDE_min_value_ratio_l3522_352243


namespace NUMINAMATH_CALUDE_smallest_c_value_l3522_352257

/-- The smallest possible value of c in a sequence satisfying specific conditions -/
theorem smallest_c_value : ∃ (a b c : ℤ),
  (a < b ∧ b < c) ∧                    -- a < b < c are integers
  (2 * b = a + c) ∧                    -- arithmetic progression
  (a * a = c * b) ∧                    -- geometric progression
  (∃ (m n p : ℤ), a = 5 * m ∧ b = 5 * n ∧ c = 5 * p) ∧  -- multiples of 5
  (0 < a ∧ 0 < b ∧ 0 < c) ∧            -- all numbers are positive
  (c = 20) ∧                           -- c equals 20
  (∀ (a' b' c' : ℤ),                   -- for any other triple satisfying the conditions
    (a' < b' ∧ b' < c') →
    (2 * b' = a' + c') →
    (a' * a' = c' * b') →
    (∃ (m' n' p' : ℤ), a' = 5 * m' ∧ b' = 5 * n' ∧ c' = 5 * p') →
    (0 < a' ∧ 0 < b' ∧ 0 < c') →
    (c ≤ c')) :=                       -- c is the smallest possible value
by sorry


end NUMINAMATH_CALUDE_smallest_c_value_l3522_352257


namespace NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l3522_352293

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ := m * 100

/-- The dimensions of the large wooden box in meters -/
def largeBoxDimensionsMeters : BoxDimensions := {
  length := 8,
  width := 10,
  height := 6
}

/-- The dimensions of the large wooden box in centimeters -/
def largeBoxDimensionsCm : BoxDimensions := {
  length := metersToCentimeters largeBoxDimensionsMeters.length,
  width := metersToCentimeters largeBoxDimensionsMeters.width,
  height := metersToCentimeters largeBoxDimensionsMeters.height
}

/-- The dimensions of the small rectangular box in centimeters -/
def smallBoxDimensions : BoxDimensions := {
  length := 4,
  width := 5,
  height := 6
}

/-- Theorem: The maximum number of small boxes that can fit in the large box is 4,000,000 -/
theorem max_small_boxes_in_large_box :
  (boxVolume largeBoxDimensionsCm) / (boxVolume smallBoxDimensions) = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l3522_352293


namespace NUMINAMATH_CALUDE_probability_two_red_one_blue_is_11_70_l3522_352271

def total_marbles : ℕ := 16
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 4

def probability_two_red_one_blue : ℚ :=
  (red_marbles * (red_marbles - 1) * blue_marbles) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem probability_two_red_one_blue_is_11_70 :
  probability_two_red_one_blue = 11 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_one_blue_is_11_70_l3522_352271


namespace NUMINAMATH_CALUDE_arrangement_exists_linear_not_circular_l3522_352291

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def validLinearArrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 16 ∧
  arrangement.toFinset = Finset.range 16 ∧
  ∀ i : ℕ, i < 15 → isPerfectSquare (arrangement[i]! + arrangement[i+1]!)

def validCircularArrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 16 ∧
  arrangement.toFinset = Finset.range 16 ∧
  ∀ i : ℕ, i < 16 → isPerfectSquare (arrangement[i]! + arrangement[(i+1) % 16]!)

theorem arrangement_exists_linear_not_circular :
  (∃ arrangement : List ℕ, validLinearArrangement arrangement) ∧
  (¬ ∃ arrangement : List ℕ, validCircularArrangement arrangement) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_exists_linear_not_circular_l3522_352291


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3522_352286

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) → m < -13/11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3522_352286


namespace NUMINAMATH_CALUDE_room_width_l3522_352217

/-- Proves that a rectangular room with given volume, length, and height has a specific width -/
theorem room_width (volume : ℝ) (length : ℝ) (height : ℝ) (width : ℝ) 
  (h_volume : volume = 10000)
  (h_length : length = 100)
  (h_height : height = 10)
  (h_relation : volume = length * width * height) :
  width = 10 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l3522_352217


namespace NUMINAMATH_CALUDE_farm_animals_l3522_352239

theorem farm_animals (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 4 * initial_cows →
  (initial_horses - 15) / (initial_cows + 15) = 13 / 7 →
  (initial_horses - 15) - (initial_cows + 15) = 30 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l3522_352239


namespace NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l3522_352259

theorem lcm_ratio_implies_gcd (X Y : ℕ+) : 
  Nat.lcm X Y = 180 → X * 6 = Y * 5 → Nat.gcd X Y = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l3522_352259


namespace NUMINAMATH_CALUDE_average_book_width_l3522_352200

def book_widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

theorem average_book_width : 
  (book_widths.sum / book_widths.length : ℝ) = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l3522_352200


namespace NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3522_352263

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_exactly_five_prime_factors (n : ℕ) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅ ∧
    ∀ (q : ℕ), is_prime q → q ∣ n → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄ ∨ q = p₅)

theorem smallest_odd_with_five_prime_factors :
  (∀ n : ℕ, n < 15015 → ¬(n % 2 = 1 ∧ has_exactly_five_prime_factors n)) ∧
  15015 % 2 = 1 ∧ has_exactly_five_prime_factors 15015 := by
  sorry

end NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3522_352263


namespace NUMINAMATH_CALUDE_min_brown_eyes_and_lunch_box_l3522_352276

theorem min_brown_eyes_and_lunch_box 
  (total_students : ℕ) 
  (brown_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 25) 
  (h2 : brown_eyes = 15) 
  (h3 : lunch_box = 18) :
  (brown_eyes + lunch_box - total_students : ℕ) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_brown_eyes_and_lunch_box_l3522_352276


namespace NUMINAMATH_CALUDE_total_height_difference_l3522_352255

/-- Given the height relationships between family members, calculate the total height difference --/
theorem total_height_difference (anne bella cathy daisy ellie : ℝ) : 
  anne = 2 * cathy ∧ 
  bella = 3 * anne ∧ 
  daisy = 1.5 * cathy ∧ 
  ellie = 1.75 * bella ∧ 
  anne = 80 → 
  |bella - cathy| + |bella - daisy| + |bella - ellie| + 
  |cathy - daisy| + |cathy - ellie| + |daisy - ellie| = 1320 := by
sorry

end NUMINAMATH_CALUDE_total_height_difference_l3522_352255


namespace NUMINAMATH_CALUDE_rectangle_area_l3522_352248

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 49 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 147 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3522_352248


namespace NUMINAMATH_CALUDE_power_sum_inequality_l3522_352245

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) 
    (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hn : n > 0) :
  a^n + b^n + c^n ≥ a*b^(n-1) + b*c^(n-1) + c*a^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l3522_352245


namespace NUMINAMATH_CALUDE_coefficient_of_sixth_power_l3522_352237

theorem coefficient_of_sixth_power (x : ℝ) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (2 - x)^6 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 + a₆*(1+x)^6 ∧
    a₆ = 1 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_sixth_power_l3522_352237


namespace NUMINAMATH_CALUDE_unique_solution_l3522_352238

/-- Represents a cell in the 5x5 grid --/
structure Cell :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the 5x5 grid --/
def Grid := Cell → Fin 5

/-- Check if two cells are in the same row --/
def same_row (c1 c2 : Cell) : Prop := c1.row = c2.row

/-- Check if two cells are in the same column --/
def same_column (c1 c2 : Cell) : Prop := c1.col = c2.col

/-- Check if two cells are in the same block --/
def same_block (c1 c2 : Cell) : Prop :=
  (c1.row / 3 = c2.row / 3) ∧ (c1.col / 3 = c2.col / 3)

/-- Check if two cells are diagonally adjacent --/
def diag_adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row + 1 ∧ c1.col = c2.col + 1) ∨
  (c1.row = c2.row + 1 ∧ c1.col = c2.col - 1) ∨
  (c1.row = c2.row - 1 ∧ c1.col = c2.col + 1) ∨
  (c1.row = c2.row - 1 ∧ c1.col = c2.col - 1)

/-- Check if a grid is valid according to the rules --/
def valid_grid (g : Grid) : Prop :=
  ∀ c1 c2 : Cell, c1 ≠ c2 →
    (same_row c1 c2 ∨ same_column c1 c2 ∨ same_block c1 c2 ∨ diag_adjacent c1 c2) →
    g c1 ≠ g c2

/-- The unique solution to the puzzle --/
theorem unique_solution (g : Grid) (h : valid_grid g) :
  (g ⟨0, 0⟩ = 5) ∧ (g ⟨0, 1⟩ = 3) ∧ (g ⟨0, 2⟩ = 1) ∧ (g ⟨0, 3⟩ = 2) ∧ (g ⟨0, 4⟩ = 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3522_352238


namespace NUMINAMATH_CALUDE_new_person_weight_l3522_352246

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3522_352246


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3522_352233

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3522_352233


namespace NUMINAMATH_CALUDE_fraction_product_value_l3522_352282

/-- The product of fractions from 8/4 to 2008/2004 following the pattern (4n+4)/(4n) -/
def fraction_product : ℚ :=
  (2008 : ℚ) / 4

theorem fraction_product_value : fraction_product = 502 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_value_l3522_352282


namespace NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l3522_352268

theorem lottery_not_guaranteed_win (total_tickets : ℕ) (winning_rate : ℝ) (bought_tickets : ℕ) : 
  total_tickets = 1000000 →
  winning_rate = 0.001 →
  bought_tickets = 1000 →
  ∃ p : ℝ, p > 0 ∧ p = (1 - winning_rate) ^ bought_tickets := by
  sorry

end NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l3522_352268


namespace NUMINAMATH_CALUDE_jenny_peanut_butter_cookies_l3522_352256

theorem jenny_peanut_butter_cookies :
  ∀ (jenny_pb : ℕ) (jenny_cc marcus_pb marcus_lemon : ℕ),
    jenny_cc = 50 →
    marcus_pb = 30 →
    marcus_lemon = 20 →
    jenny_pb + marcus_pb = jenny_cc + marcus_lemon →
    jenny_pb = 40 := by
  sorry

end NUMINAMATH_CALUDE_jenny_peanut_butter_cookies_l3522_352256


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3522_352204

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3522_352204


namespace NUMINAMATH_CALUDE_W_lower_bound_l3522_352252

/-- W(k,2) is the smallest number such that if n ≥ W(k,2), 
    for each coloring of the set {1,2,...,n} with two colors, 
    there exists a monochromatic arithmetic progression of length k -/
def W (k : ℕ) : ℕ := sorry

/-- The main theorem stating that W(k,2) = Ω(2^(k/2)) -/
theorem W_lower_bound : ∃ (c : ℝ) (k₀ : ℕ), c > 0 ∧ ∀ k ≥ k₀, (W k : ℝ) ≥ c * 2^(k/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_W_lower_bound_l3522_352252


namespace NUMINAMATH_CALUDE_modulus_of_complex_l3522_352242

theorem modulus_of_complex (z : ℂ) : (1 - Complex.I) * z = 3 - Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l3522_352242


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l3522_352201

structure Ball :=
  (color : String)

def Bag : Finset Ball := sorry

axiom bag_composition : 
  (Bag.filter (λ b => b.color = "red")).card = 2 ∧ 
  (Bag.filter (λ b => b.color = "black")).card = 2

def Draw : Finset Ball := sorry

axiom draw_size : Draw.card = 2

def exactly_one_black : Prop :=
  (Draw.filter (λ b => b.color = "black")).card = 1

def exactly_two_black : Prop :=
  (Draw.filter (λ b => b.color = "black")).card = 2

theorem mutually_exclusive_not_contradictory :
  (¬(exactly_one_black ∧ exactly_two_black)) ∧
  (∃ draw : Finset Ball, draw.card = 2 ∧ ¬exactly_one_black ∧ ¬exactly_two_black) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l3522_352201


namespace NUMINAMATH_CALUDE_f_inequality_range_l3522_352270

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : f (2*x) > f (x - 3) ↔ x < -3 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l3522_352270


namespace NUMINAMATH_CALUDE_inverse_matrices_product_l3522_352278

def inverse_matrices (x y z w : ℝ) : Prop :=
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![x, 3; 4, y]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, z; w, -5]
  A * B = 1

theorem inverse_matrices_product (x y z w : ℝ) 
  (h : inverse_matrices x y z w) : x * y * z * w = -5040/49 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_product_l3522_352278


namespace NUMINAMATH_CALUDE_parking_lot_cars_l3522_352225

theorem parking_lot_cars (car_wheels : ℕ) (motorcycle_wheels : ℕ) (num_motorcycles : ℕ) (total_wheels : ℕ) :
  car_wheels = 5 →
  motorcycle_wheels = 2 →
  num_motorcycles = 11 →
  total_wheels = 117 →
  ∃ num_cars : ℕ, num_cars * car_wheels + num_motorcycles * motorcycle_wheels = total_wheels ∧ num_cars = 19 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l3522_352225


namespace NUMINAMATH_CALUDE_factoring_expression_l3522_352249

theorem factoring_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) = (y + 2) * (5 * y + 9) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3522_352249


namespace NUMINAMATH_CALUDE_ac_over_b_squared_eq_one_l3522_352285

/-- A quadratic equation with real coefficients -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  has_imaginary_roots : ∃ (x₁ x₂ : ℂ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ ∧ x₁.im ≠ 0 ∧ x₂.im ≠ 0
  x₁_cubed_real : ∃ (x₁ : ℂ), (a * x₁^2 + b * x₁ + c = 0) ∧ (∃ (r : ℝ), x₁^3 = r)

/-- Theorem stating that ac/b^2 = 1 for a quadratic equation satisfying the given conditions -/
theorem ac_over_b_squared_eq_one (eq : QuadraticEquation) : eq.a * eq.c / eq.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ac_over_b_squared_eq_one_l3522_352285


namespace NUMINAMATH_CALUDE_circumradius_inradius_ratio_irrational_l3522_352272

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define a triangle with lattice points as vertices
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

-- Define a square-free natural number
def SquareFree (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m * m ∣ n → m = 1

-- Define the property that one side of the triangle has length √n
def HasSqrtNSide (t : LatticeTriangle) (n : ℕ) : Prop :=
  SquareFree n ∧
  (((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 : ℚ) = n ∨
   ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 : ℚ) = n ∨
   ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 : ℚ) = n)

-- Define the circumradius and inradius of a triangle
noncomputable def circumradius (t : LatticeTriangle) : ℝ := sorry
noncomputable def inradius (t : LatticeTriangle) : ℝ := sorry

-- The main theorem
theorem circumradius_inradius_ratio_irrational (t : LatticeTriangle) (n : ℕ) :
  HasSqrtNSide t n → ¬ (∃ q : ℚ, (circumradius t / inradius t : ℝ) = q) :=
sorry

end NUMINAMATH_CALUDE_circumradius_inradius_ratio_irrational_l3522_352272


namespace NUMINAMATH_CALUDE_kevins_weekly_revenue_l3522_352247

/-- Calculates the total weekly revenue for a fruit vendor --/
def total_weekly_revenue (total_crates price_grapes price_mangoes price_passion 
                          crates_grapes crates_mangoes : ℕ) : ℕ :=
  let crates_passion := total_crates - crates_grapes - crates_mangoes
  let revenue_grapes := crates_grapes * price_grapes
  let revenue_mangoes := crates_mangoes * price_mangoes
  let revenue_passion := crates_passion * price_passion
  revenue_grapes + revenue_mangoes + revenue_passion

/-- Theorem stating that Kevin's total weekly revenue is $1020 --/
theorem kevins_weekly_revenue : 
  total_weekly_revenue 50 15 20 25 13 20 = 1020 := by
  sorry

#eval total_weekly_revenue 50 15 20 25 13 20

end NUMINAMATH_CALUDE_kevins_weekly_revenue_l3522_352247


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l3522_352206

theorem polynomial_root_implies_coefficients : 
  ∀ (p q : ℝ), 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + p * (2 - 3 * Complex.I : ℂ) ^ 2 - 5 * (2 - 3 * Complex.I : ℂ) + q = 0 →
  p = 1/2 ∧ q = 117/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l3522_352206


namespace NUMINAMATH_CALUDE_cone_base_radius_l3522_352295

/-- Given a sector with radius 5 and central angle 144°, prove that when wrapped into a cone, 
    the radius of the base of the cone is 2. -/
theorem cone_base_radius (r : ℝ) (θ : ℝ) : 
  r = 5 → θ = 144 → (θ / 360) * (2 * π * r) = 2 * π * 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3522_352295


namespace NUMINAMATH_CALUDE_min_value_of_a_l3522_352215

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3522_352215


namespace NUMINAMATH_CALUDE_chloe_score_l3522_352280

/-- The score for each treasure found in the game -/
def points_per_treasure : ℕ := 9

/-- The number of treasures found on the first level -/
def treasures_level_1 : ℕ := 6

/-- The number of treasures found on the second level -/
def treasures_level_2 : ℕ := 3

/-- Chloe's total score in the game -/
def total_score : ℕ := points_per_treasure * (treasures_level_1 + treasures_level_2)

/-- Theorem stating that Chloe's total score is 81 points -/
theorem chloe_score : total_score = 81 := by
  sorry

end NUMINAMATH_CALUDE_chloe_score_l3522_352280


namespace NUMINAMATH_CALUDE_a_work_days_l3522_352253

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 16

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 6

/-- The theorem stating that A can finish the work alone in 4 days -/
theorem a_work_days : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    together_days * (1/x + 1/b_days) + b_alone_days * (1/b_days) = 1 ∧ 
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_work_days_l3522_352253


namespace NUMINAMATH_CALUDE_difference_of_powers_l3522_352289

theorem difference_of_powers (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4) 
  (h2 : c ^ 3 = d ^ 2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by
sorry

end NUMINAMATH_CALUDE_difference_of_powers_l3522_352289


namespace NUMINAMATH_CALUDE_envelope_difference_l3522_352269

theorem envelope_difference (blue_envelopes : ℕ) (total_envelopes : ℕ) (yellow_envelopes : ℕ) :
  blue_envelopes = 10 →
  total_envelopes = 16 →
  yellow_envelopes < blue_envelopes →
  yellow_envelopes + blue_envelopes = total_envelopes →
  blue_envelopes - yellow_envelopes = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_envelope_difference_l3522_352269


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l3522_352292

/-- The probability of selecting one marble of each color from a bag with 3 red, 3 blue, and 3 green marbles -/
theorem one_of_each_color_probability : 
  let total_marbles : ℕ := 3 + 3 + 3
  let marbles_per_color : ℕ := 3
  let selected_marbles : ℕ := 3
  (marbles_per_color ^ selected_marbles : ℚ) / (Nat.choose total_marbles selected_marbles) = 9 / 28 :=
by sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l3522_352292


namespace NUMINAMATH_CALUDE_inverse_proportional_cube_root_l3522_352284

theorem inverse_proportional_cube_root (x y : ℝ) (k : ℝ) : 
  (x ^ 2 * y ^ (1/3) = k) →  -- x² and ³√y are inversely proportional
  (3 ^ 2 * 216 ^ (1/3) = k) →  -- x = 3 when y = 216
  (x * y = 54) →  -- xy = 54
  y = 18 * 4 ^ (1/3) :=  -- y = 18 ³√4
by sorry

end NUMINAMATH_CALUDE_inverse_proportional_cube_root_l3522_352284


namespace NUMINAMATH_CALUDE_average_daily_attendance_l3522_352258

def monday_attendance : ℕ := 10
def tuesday_attendance : ℕ := 15
def wednesday_to_friday_attendance : ℕ := 10
def total_days : ℕ := 5

def total_attendance : ℕ := monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance

theorem average_daily_attendance :
  (total_attendance : ℚ) / total_days = 11 := by sorry

end NUMINAMATH_CALUDE_average_daily_attendance_l3522_352258


namespace NUMINAMATH_CALUDE_angle_BDC_is_20_l3522_352298

-- Define the angles in degrees
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- Theorem to prove
theorem angle_BDC_is_20 : 
  ∃ (angle_BDC : ℝ), angle_BDC = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_BDC_is_20_l3522_352298


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3522_352296

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 3 * x + 1 > 0}
def B : Set ℝ := {x : ℝ | |x - 1| < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-1/3 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3522_352296


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3522_352235

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3522_352235


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3522_352228

/-- Given a line y = kx + 3 intersecting the circle (x-1)^2 + (y-2)^2 = 4 at points M and N,
    if |MN| ≥ 2√3, then k ≤ 0. -/
theorem intersection_line_circle (k : ℝ) (M N : ℝ × ℝ) : 
  (∀ x y, y = k * x + 3 → (x - 1)^2 + (y - 2)^2 = 4) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12 →
  k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l3522_352228


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l3522_352279

theorem parabola_x_intercepts :
  let f (x : ℝ) := -3 * x^2 + 4 * x - 1
  (∃ a b : ℝ, a ≠ b ∧ f a = 0 ∧ f b = 0) ∧
  (∀ x y z : ℝ, f x = 0 → f y = 0 → f z = 0 → x = y ∨ x = z ∨ y = z) := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l3522_352279


namespace NUMINAMATH_CALUDE_composite_function_ratio_l3522_352224

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composite_function_ratio : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 35 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_ratio_l3522_352224


namespace NUMINAMATH_CALUDE_exists_ratio_eq_rational_l3522_352265

def u : ℕ → ℚ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then u (n / 2) + u ((n - 1) / 2) else u (n / 2)

theorem exists_ratio_eq_rational (k : ℚ) (hk : k > 0) :
  ∃ n : ℕ, u n / u (n + 1) = k :=
by sorry

end NUMINAMATH_CALUDE_exists_ratio_eq_rational_l3522_352265


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_not_equal_not_vertical_l3522_352241

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the property of being vertical angles
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define the property of angles being equal
def are_equal (a b : Angle) : Prop := a = b

-- Theorem 1: If two angles are vertical angles, then they are equal
theorem vertical_angles_are_equal (a b : Angle) :
  are_vertical_angles a b → are_equal a b := by sorry

-- Theorem 2: If two angles are not equal, then they are not vertical angles
theorem not_equal_not_vertical (a b : Angle) :
  ¬(are_equal a b) → ¬(are_vertical_angles a b) := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_not_equal_not_vertical_l3522_352241


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3522_352288

/-- Given a cubic function f with three distinct roots, prove properties about its values at 0, 1, and 3 -/
theorem cubic_function_properties (a b c : ℝ) (abc : ℝ) :
  a < b → b < c →
  let f : ℝ → ℝ := fun x ↦ x^3 - 6*x^2 + 9*x - abc
  f a = 0 → f b = 0 → f c = 0 →
  (f 0) * (f 1) < 0 ∧ (f 0) * (f 3) > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3522_352288


namespace NUMINAMATH_CALUDE_average_score_two_classes_l3522_352227

theorem average_score_two_classes (n1 n2 : ℕ) (s1 s2 : ℝ) :
  n1 > 0 → n2 > 0 →
  s1 = 80 → s2 = 70 →
  n1 = 20 → n2 = 30 →
  (n1 * s1 + n2 * s2) / (n1 + n2 : ℝ) = 74 := by
  sorry

end NUMINAMATH_CALUDE_average_score_two_classes_l3522_352227


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3522_352299

theorem smallest_x_absolute_value_equation : 
  (∃ x : ℝ, |4*x - 5| = 29) ∧ 
  (∀ x : ℝ, |4*x - 5| = 29 → x ≥ -6) ∧ 
  |4*(-6) - 5| = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3522_352299


namespace NUMINAMATH_CALUDE_pencil_length_l3522_352202

theorem pencil_length (length1 length2 total_length : ℕ) : 
  length1 = length2 → 
  length1 + length2 = 24 → 
  length1 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l3522_352202


namespace NUMINAMATH_CALUDE_gender_related_to_reading_l3522_352208

-- Define the survey data
def survey_data : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![30, 20],
    ![40, 10]]

-- Define the total number of observations
def N : ℕ := 100

-- Define the formula for calculating k^2
def calculate_k_squared (data : Matrix (Fin 2) (Fin 2) ℕ) (total : ℕ) : ℚ :=
  let O11 := data 0 0
  let O12 := data 0 1
  let O21 := data 1 0
  let O22 := data 1 1
  (total * (O11 * O22 - O12 * O21)^2 : ℚ) / 
  ((O11 + O12) * (O21 + O22) * (O11 + O21) * (O12 + O22) : ℚ)

-- Define the critical values
def critical_value_005 : ℚ := 3841 / 1000
def critical_value_001 : ℚ := 6635 / 1000

-- State the theorem
theorem gender_related_to_reading :
  let k_squared := calculate_k_squared survey_data N
  k_squared > critical_value_005 ∧ k_squared < critical_value_001 := by
  sorry

end NUMINAMATH_CALUDE_gender_related_to_reading_l3522_352208


namespace NUMINAMATH_CALUDE_e_percentage_of_d_l3522_352266

-- Define the variables
variable (a b c d e : ℝ)

-- Define the relationships between the variables
def relationship_d : Prop := d = 0.4 * a ∧ d = 0.35 * b
def relationship_e : Prop := e = 0.5 * b ∧ e = 0.2 * c
def relationship_c : Prop := c = 0.3 * a ∧ c = 0.25 * b

-- Theorem statement
theorem e_percentage_of_d 
  (hd : relationship_d a b d)
  (he : relationship_e b c e)
  (hc : relationship_c a b c) :
  e / d = 0.15 := by sorry

end NUMINAMATH_CALUDE_e_percentage_of_d_l3522_352266


namespace NUMINAMATH_CALUDE_marbles_distribution_l3522_352281

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 80 →
  num_boys = 8 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l3522_352281


namespace NUMINAMATH_CALUDE_combined_stickers_l3522_352226

theorem combined_stickers (june_initial : ℕ) (bonnie_initial : ℕ) (birthday_gift : ℕ) :
  june_initial = 76 →
  bonnie_initial = 63 →
  birthday_gift = 25 →
  june_initial + bonnie_initial + 2 * birthday_gift = 189 :=
by sorry

end NUMINAMATH_CALUDE_combined_stickers_l3522_352226


namespace NUMINAMATH_CALUDE_halving_r_problem_l3522_352261

theorem halving_r_problem (r : ℝ) (n : ℝ) (a : ℝ) :
  a = (2 * r) ^ n →
  ((r / 2) ^ n = 0.125 * a) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_halving_r_problem_l3522_352261


namespace NUMINAMATH_CALUDE_problem_statement_l3522_352216

theorem problem_statement : 
  ∃ d : ℝ, 5^(Real.log 30) * (1/3)^(Real.log 0.5) = d ∧ d = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3522_352216


namespace NUMINAMATH_CALUDE_odd_m_triple_g_35_l3522_352262

def g (n : Int) : Int :=
  if n % 2 = 1 then n + 5 else n / 2

theorem odd_m_triple_g_35 (m : Int) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_35_l3522_352262


namespace NUMINAMATH_CALUDE_regular_polygon_with_20_degree_exterior_angle_l3522_352223

theorem regular_polygon_with_20_degree_exterior_angle (n : ℕ) : 
  n > 2 → (360 : ℝ) / n = 20 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_20_degree_exterior_angle_l3522_352223


namespace NUMINAMATH_CALUDE_line_through_points_l3522_352230

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line passing through two given points -/
def line_equation (p₀ p₁ : Point) : ℝ → Prop :=
  fun y => y = p₀.y

/-- The theorem states that the line equation y = 2 passes through the given points -/
theorem line_through_points :
  let p₀ : Point := ⟨1, 2⟩
  let p₁ : Point := ⟨3, 2⟩
  let eq := line_equation p₀ p₁
  (eq 2) ∧ (p₀.y = 2) ∧ (p₁.y = 2) := by sorry

end NUMINAMATH_CALUDE_line_through_points_l3522_352230


namespace NUMINAMATH_CALUDE_decimal_to_binary_38_l3522_352214

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_38_l3522_352214


namespace NUMINAMATH_CALUDE_range_of_x_l3522_352273

def is_monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_x (f : ℝ → ℝ) 
  (h1 : is_monotone_decreasing f)
  (h2 : is_odd_function f)
  (h3 : f 1 = -1)
  (h4 : ∀ x, -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1) :
  ∀ x, -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1 → 1 ≤ x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l3522_352273


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3522_352236

theorem solution_set_inequality (x : ℝ) : 
  (Set.Icc (-3 : ℝ) 6 : Set ℝ) = {x | (x + 3) * (6 - x) ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3522_352236


namespace NUMINAMATH_CALUDE_base13_representation_of_234_l3522_352219

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a natural number to its base 13 representation -/
def toBase13 (n : ℕ) : List Base13Digit := sorry

/-- Converts a list of Base13Digits to its decimal (base 10) value -/
def fromBase13 (digits : List Base13Digit) : ℕ := sorry

theorem base13_representation_of_234 :
  toBase13 234 = [Base13Digit.D1, Base13Digit.D5] := by sorry

end NUMINAMATH_CALUDE_base13_representation_of_234_l3522_352219


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l3522_352220

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, f(x)) in that quadrant. -/
def passes_through_quadrant (m b : ℝ) (quad : ℕ) : Prop :=
  ∃ x y : ℝ, y = m * x + b ∧
  match quad with
  | 1 => x > 0 ∧ y > 0
  | 2 => x < 0 ∧ y > 0
  | 3 => x < 0 ∧ y < 0
  | 4 => x > 0 ∧ y < 0
  | _ => False

/-- The slope of the linear function -/
def m : ℝ := -5

/-- The y-intercept of the linear function -/
def b : ℝ := 3

/-- Theorem stating that the linear function f(x) = -5x + 3 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant m b 1 ∧
  passes_through_quadrant m b 2 ∧
  passes_through_quadrant m b 4 ∧
  ¬passes_through_quadrant m b 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l3522_352220


namespace NUMINAMATH_CALUDE_max_digits_product_four_digit_numbers_l3522_352211

theorem max_digits_product_four_digit_numbers :
  ∀ a b : ℕ, 1000 ≤ a ∧ a ≤ 9999 → 1000 ≤ b ∧ b ≤ 9999 →
  ∃ n : ℕ, n ≤ 8 ∧ a * b < 10^n :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_four_digit_numbers_l3522_352211


namespace NUMINAMATH_CALUDE_proposition_relationship_l3522_352205

theorem proposition_relationship (a b : ℝ) :
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ ¬(a > b ∧ a⁻¹ > b⁻¹)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l3522_352205


namespace NUMINAMATH_CALUDE_semi_annual_annuity_payment_l3522_352275

/-- Calculates the semi-annual annuity payment given the following conditions:
  * Initial annual payment of 2500 HUF
  * Payment duration of 15 years
  * No collection for first 5 years
  * Convert to semi-annual annuity lasting 20 years, starting at beginning of 6th year
  * Annual interest rate of 4.75%
-/
def calculate_semi_annual_annuity (
  initial_payment : ℝ
  ) (payment_duration : ℕ
  ) (no_collection_years : ℕ
  ) (annuity_duration : ℕ
  ) (annual_interest_rate : ℝ
  ) : ℝ :=
  sorry

/-- The semi-annual annuity payment is approximately 2134.43 HUF -/
theorem semi_annual_annuity_payment :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |calculate_semi_annual_annuity 2500 15 5 20 0.0475 - 2134.43| < ε :=
sorry

end NUMINAMATH_CALUDE_semi_annual_annuity_payment_l3522_352275


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3522_352221

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (2 : ℝ)^2 = a^2 + b^2 →
  (∀ (x y : ℝ), (b*x = a*y ∨ b*x = -a*y) → (x - 2)^2 + y^2 = 3) →
  (∀ (x y : ℝ), x^2 - y^2 / 3 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3522_352221


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3522_352250

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  initial_price = 150 →
  first_discount = 20 →
  final_price = 108 →
  ∃ (second_discount : ℝ),
    second_discount = 10 ∧
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3522_352250


namespace NUMINAMATH_CALUDE_real_estate_problem_l3522_352260

-- Define the constants
def total_sets : ℕ := 80
def cost_A : ℕ := 90
def price_A : ℕ := 102
def cost_B : ℕ := 60
def price_B : ℕ := 70
def min_funds : ℕ := 5700
def max_A : ℕ := 32

-- Define the variables
variable (x : ℕ) -- number of Type A sets
variable (W : ℕ → ℕ) -- profit function
variable (a : ℚ) -- price reduction for Type A

-- Define the theorem
theorem real_estate_problem :
  (∀ x, W x = 2 * x + 800) ∧
  (x ≥ 30 ∧ x ≤ 32) ∧
  (∀ a, 0 < a ∧ a ≤ 3 →
    (0 < a ∧ a < 2 → x = 32) ∧
    (a = 2 → true) ∧
    (2 < a ∧ a ≤ 3 → x = 30)) :=
sorry

end NUMINAMATH_CALUDE_real_estate_problem_l3522_352260


namespace NUMINAMATH_CALUDE_rational_fraction_representation_l3522_352264

def is_rational (f : ℕ+ → ℚ) : Prop :=
  ∀ x : ℕ+, ∃ p q : ℤ, f x = p / q ∧ q ≠ 0

theorem rational_fraction_representation
  (a b : ℚ) (h : is_rational (λ x : ℕ+ => (a * x + b) / x)) :
  ∃ A B C : ℤ, ∀ x : ℕ+, (a * x + b) / x = (A * x + B) / (C * x) :=
sorry

end NUMINAMATH_CALUDE_rational_fraction_representation_l3522_352264


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3522_352254

theorem possible_values_of_a (a b c d : ℕ) : 
  a > b ∧ b > c ∧ c > d ∧ 
  a + b + c + d = 2010 ∧ 
  a^2 - b^2 + c^2 - d^2 = 2010 →
  (∃ (s : Finset ℕ), s.card = 501 ∧ ∀ x, x ∈ s ↔ (∃ b' c' d' : ℕ, 
    x > b' ∧ b' > c' ∧ c' > d' ∧ 
    x + b' + c' + d' = 2010 ∧ 
    x^2 - b'^2 + c'^2 - d'^2 = 2010)) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3522_352254


namespace NUMINAMATH_CALUDE_ellipse_theorem_proof_l3522_352267

noncomputable def ellipse_theorem (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  let e := Real.sqrt (a^2 - b^2)
  ∀ (M : ℝ × ℝ),
    M.1^2 / a^2 + M.2^2 / b^2 = 1 →
    let F₁ : ℝ × ℝ := (-e, 0)
    let F₂ : ℝ × ℝ := (e, 0)
    ∃ (A B : ℝ × ℝ),
      A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
      B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
      (∃ t : ℝ, A = M + t • (M - F₁)) ∧
      (∃ s : ℝ, B = M + s • (M - F₂)) →
      (b^2 / a^2) * (‖M - F₁‖ / ‖F₁ - A‖ + ‖M - F₂‖ / ‖F₂ - B‖ + 2) = 4

theorem ellipse_theorem_proof (a b : ℝ) (h : a > b ∧ b > 0) :
  ellipse_theorem a b h := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_proof_l3522_352267


namespace NUMINAMATH_CALUDE_simplify_expression_l3522_352283

theorem simplify_expression (b : ℝ) : (2 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 720 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3522_352283


namespace NUMINAMATH_CALUDE_digit_of_fraction_l3522_352213

/-- The fraction we're considering -/
def f : ℚ := 66 / 1110

/-- The index of the digit we're looking for (0-indexed) -/
def n : ℕ := 221

/-- The function that returns the nth digit after the decimal point
    in the decimal representation of a rational number -/
noncomputable def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem digit_of_fraction :
  nth_digit_after_decimal f n = 5 := by sorry

end NUMINAMATH_CALUDE_digit_of_fraction_l3522_352213


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3522_352210

theorem complex_equation_solution :
  ∃ x : ℤ, x - (28 - (37 - (15 - 17))) = 56 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3522_352210


namespace NUMINAMATH_CALUDE_vector_sum_of_squares_l3522_352251

/-- Given vectors a and b, with n as their midpoint, prove that ‖a‖² + ‖b‖² = 48 -/
theorem vector_sum_of_squares (a b : ℝ × ℝ) (n : ℝ × ℝ) : 
  n = (4, -1) → n = (a + b) / 2 → a • b = 10 → ‖a‖^2 + ‖b‖^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_of_squares_l3522_352251


namespace NUMINAMATH_CALUDE_quadratic_form_j_value_l3522_352232

/-- Given a quadratic expression px^2 + qx + r that can be expressed as 5(x - 3)^2 + 15,
    prove that when 4px^2 + 4qx + 4r is expressed as m(x - j)^2 + k, j = 3. -/
theorem quadratic_form_j_value (p q r : ℝ) :
  (∃ m j k : ℝ, ∀ x : ℝ, 
    px^2 + q*x + r = 5*(x - 3)^2 + 15 ∧ 
    4*p*x^2 + 4*q*x + 4*r = m*(x - j)^2 + k) →
  (∃ m k : ℝ, ∀ x : ℝ, 4*p*x^2 + 4*q*x + 4*r = m*(x - 3)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_j_value_l3522_352232


namespace NUMINAMATH_CALUDE_first_part_segments_second_part_segments_l3522_352229

/-- Number of segments after cutting loops in a Chinese knot --/
def segments_after_cutting (loops : ℕ) (wings : ℕ := 1) : ℕ :=
  (loops * 2 * wings + wings) / wings

/-- Theorem for the first part of the problem --/
theorem first_part_segments : segments_after_cutting 5 = 6 := by sorry

/-- Theorem for the second part of the problem --/
theorem second_part_segments : segments_after_cutting 7 2 = 15 := by sorry

end NUMINAMATH_CALUDE_first_part_segments_second_part_segments_l3522_352229


namespace NUMINAMATH_CALUDE_total_followers_after_one_month_l3522_352222

/-- Represents the number of followers on various social media platforms -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ
  pinterest : ℕ
  snapchat : ℕ

/-- Calculates the total number of followers across all platforms -/
def total_followers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube + f.pinterest + f.snapchat

/-- Represents the initial number of followers -/
def initial_followers : Followers := {
  instagram := 240,
  facebook := 500,
  twitter := (240 + 500) / 2,
  tiktok := 3 * ((240 + 500) / 2),
  youtube := 3 * ((240 + 500) / 2) + 510,
  pinterest := 120,
  snapchat := 120 / 2
}

/-- Represents the number of followers after one month -/
def followers_after_one_month : Followers := {
  instagram := initial_followers.instagram + (initial_followers.instagram * 15 / 100),
  facebook := initial_followers.facebook + (initial_followers.facebook * 20 / 100),
  twitter := initial_followers.twitter + 30,
  tiktok := initial_followers.tiktok + 45,
  youtube := initial_followers.youtube,
  pinterest := initial_followers.pinterest,
  snapchat := initial_followers.snapchat - 10
}

/-- Theorem stating that the total number of followers after one month is 4221 -/
theorem total_followers_after_one_month : 
  total_followers followers_after_one_month = 4221 := by
  sorry


end NUMINAMATH_CALUDE_total_followers_after_one_month_l3522_352222


namespace NUMINAMATH_CALUDE_phantom_ink_problem_l3522_352231

/-- The cost of a single black printer ink -/
def black_ink_cost : ℕ := 11

/-- The amount Phantom's mom gave him -/
def initial_amount : ℕ := 50

/-- The number of black printer inks bought -/
def black_ink_count : ℕ := 2

/-- The number of red printer inks bought -/
def red_ink_count : ℕ := 3

/-- The cost of each red printer ink -/
def red_ink_cost : ℕ := 15

/-- The number of yellow printer inks bought -/
def yellow_ink_count : ℕ := 2

/-- The cost of each yellow printer ink -/
def yellow_ink_cost : ℕ := 13

/-- The additional amount Phantom needs -/
def additional_amount : ℕ := 43

theorem phantom_ink_problem :
  black_ink_cost * black_ink_count +
  red_ink_cost * red_ink_count +
  yellow_ink_cost * yellow_ink_count =
  initial_amount + additional_amount :=
sorry

end NUMINAMATH_CALUDE_phantom_ink_problem_l3522_352231
