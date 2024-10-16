import Mathlib

namespace NUMINAMATH_CALUDE_linear_function_property_l2632_263229

/-- A linear function is a function of the form f(x) = mx + b, where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hLinear : LinearFunction g) 
  (hDiff : g 10 - g 0 = 20) : 
  g 20 - g 0 = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2632_263229


namespace NUMINAMATH_CALUDE_seven_square_side_length_l2632_263258

/-- Represents a shape composed of seven equal squares -/
structure SevenSquareShape :=
  (side_length : ℝ)

/-- Represents a line that divides the shape into two equal areas -/
structure DividingLine :=
  (shape : SevenSquareShape)
  (divides_equally : Bool)

/-- Represents the intersection points of the dividing line with the shape -/
structure IntersectionPoints :=
  (line : DividingLine)
  (cf_length : ℝ)
  (ae_length : ℝ)
  (sum_cf_ae : cf_length + ae_length = 91)

/-- Theorem: The side length of each small square is 26 cm -/
theorem seven_square_side_length
  (shape : SevenSquareShape)
  (line : DividingLine)
  (points : IntersectionPoints)
  (h1 : line.shape = shape)
  (h2 : line.divides_equally = true)
  (h3 : points.line = line)
  : shape.side_length = 26 :=
by sorry

end NUMINAMATH_CALUDE_seven_square_side_length_l2632_263258


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_after_increase_l2632_263268

/-- The perimeter of a semi-circle with radius 7.68 cm is approximately 39.50 cm. -/
theorem semi_circle_perimeter_after_increase : 
  let r : ℝ := 7.68
  let π : ℝ := 3.14159
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, |perimeter - 39.50| < ε :=
by sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_after_increase_l2632_263268


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l2632_263261

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ, n ≥ 12 → ∃ z ∈ T, z ^ n = 1) ∧ 
  (∀ m : ℕ, m < 12 → ∃ n : ℕ, n ≥ m ∧ ∀ z ∈ T, z ^ n ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l2632_263261


namespace NUMINAMATH_CALUDE_age_difference_proof_l2632_263265

theorem age_difference_proof (n : ℕ) 
  (h1 : 2 * n + 8 = 3 * (n + 8)) : 
  2 * n - n = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2632_263265


namespace NUMINAMATH_CALUDE_cuboid_edge_sum_l2632_263232

/-- The sum of the lengths of the edges of a cuboid -/
def sumOfEdges (width length height : ℝ) : ℝ :=
  4 * (width + length + height)

/-- Theorem: The sum of the lengths of the edges of a cuboid with
    width 10 cm, length 8 cm, and height 5 cm is equal to 92 cm -/
theorem cuboid_edge_sum :
  sumOfEdges 10 8 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_sum_l2632_263232


namespace NUMINAMATH_CALUDE_first_day_exceeding_2000_l2632_263249

def algae_population (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_2000 :
  ∃ n : ℕ, n > 0 ∧ algae_population n > 2000 ∧ ∀ m : ℕ, m < n → algae_population m ≤ 2000 :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_first_day_exceeding_2000_l2632_263249


namespace NUMINAMATH_CALUDE_least_k_cubed_divisible_by_336_l2632_263296

theorem least_k_cubed_divisible_by_336 :
  ∃ (k : ℕ), k > 0 ∧ k^3 % 336 = 0 ∧ ∀ (m : ℕ), m > 0 → m^3 % 336 = 0 → k ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_k_cubed_divisible_by_336_l2632_263296


namespace NUMINAMATH_CALUDE_natural_equation_example_natural_equation_condition_l2632_263247

-- Definition of a natural equation
def is_natural_equation (a b c : ℤ) : Prop :=
  ∃ x₁ x₂ : ℤ, (a * x₁^2 + b * x₁ + c = 0) ∧ 
              (a * x₂^2 + b * x₂ + c = 0) ∧ 
              (abs (x₁ - x₂) = 1) ∧
              (a ≠ 0)

-- Theorem 1: x² + 3x + 2 = 0 is a natural equation
theorem natural_equation_example : is_natural_equation 1 3 2 := by
  sorry

-- Theorem 2: x² - (m+1)x + m = 0 is a natural equation iff m = 0 or m = 2
theorem natural_equation_condition (m : ℤ) : 
  is_natural_equation 1 (-(m+1)) m ↔ m = 0 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_natural_equation_example_natural_equation_condition_l2632_263247


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2632_263219

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | x^2 + y^2 = x + y + 2} = 
  {(-1, 0), (-1, 1), (0, -1), (0, 2), (1, -1), (1, 2), (2, 0), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2632_263219


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l2632_263283

theorem coefficient_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 - 4 * X^2 + 3 * X + 2
  let p₂ : Polynomial ℤ := -X^2 + 3 * X - 5
  (p₁ * p₂).coeff 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l2632_263283


namespace NUMINAMATH_CALUDE_volume_three_triangular_pyramids_l2632_263254

/-- The volume of three identical triangular pyramids -/
theorem volume_three_triangular_pyramids 
  (base_measurement : ℝ) 
  (base_height : ℝ) 
  (pyramid_height : ℝ) 
  (h1 : base_measurement = 40) 
  (h2 : base_height = 20) 
  (h3 : pyramid_height = 30) : 
  3 * (1/3 * (1/2 * base_measurement * base_height) * pyramid_height) = 12000 := by
  sorry

#check volume_three_triangular_pyramids

end NUMINAMATH_CALUDE_volume_three_triangular_pyramids_l2632_263254


namespace NUMINAMATH_CALUDE_solve_equation_l2632_263260

theorem solve_equation (x y : ℚ) : 
  y = 2 / (4 * x + 2) → y = 1/2 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2632_263260


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2632_263217

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2632_263217


namespace NUMINAMATH_CALUDE_coefficient_expansion_l2632_263275

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def coefficient_x2y3z2 (a b c : ℕ → ℕ → ℕ → ℕ) : ℕ :=
  2^3 * binomial 6 3 * binomial 3 2 - 2^2 * binomial 6 4 * binomial 4 2

theorem coefficient_expansion :
  ∀ (a b c : ℕ → ℕ → ℕ → ℕ),
  (∀ x y z, a x y z = x - y) →
  (∀ x y z, b x y z = x + 2*y + z) →
  (∀ x y z, c x y z = a x y z * (b x y z)^6) →
  coefficient_x2y3z2 a b c = 120 :=
sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l2632_263275


namespace NUMINAMATH_CALUDE_equations_solutions_l2632_263257

-- Define the equations
def equation1 (x : ℝ) : Prop :=
  (x - 3) / (x - 2) + 1 = 3 / (2 - x)

def equation2 (x : ℝ) : Prop :=
  (x - 2) / (x + 2) - (x + 2) / (x - 2) = 16 / (x^2 - 4)

-- Theorem statement
theorem equations_solutions :
  equation1 1 ∧ equation2 (-4) :=
by sorry

end NUMINAMATH_CALUDE_equations_solutions_l2632_263257


namespace NUMINAMATH_CALUDE_square_area_is_25_l2632_263240

-- Define the points
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (5, 6)

-- Define the square area function
def square_area (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  (dx * dx + dy * dy)

-- Theorem statement
theorem square_area_is_25 : square_area point1 point2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_25_l2632_263240


namespace NUMINAMATH_CALUDE_probability_both_selected_l2632_263228

theorem probability_both_selected (prob_X prob_Y : ℚ) 
  (h1 : prob_X = 1/3) (h2 : prob_Y = 2/5) : 
  prob_X * prob_Y = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l2632_263228


namespace NUMINAMATH_CALUDE_smallest_n_mod_20_sum_l2632_263226

theorem smallest_n_mod_20_sum (n : ℕ) : n ≥ 9 ↔ 
  ∀ (S : Finset ℤ), S.card = n → 
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      (a + b) % 20 = (c + d) % 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_mod_20_sum_l2632_263226


namespace NUMINAMATH_CALUDE_area_between_curves_l2632_263266

-- Define the functions f and g
def f (x : ℝ) : ℝ := -2 * x^2 + 7 * x - 6
def g (x : ℝ) : ℝ := -x

-- Define the intersection points
def x1 : ℝ := 1
def x2 : ℝ := 3

-- State the theorem
theorem area_between_curves : 
  (∫ (x : ℝ) in x1..x2, f x - g x) = 8/3 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l2632_263266


namespace NUMINAMATH_CALUDE_football_season_games_l2632_263201

/-- Represents a football team's season statistics -/
structure SeasonStats where
  totalGames : ℕ
  tieGames : ℕ
  firstHundredWins : ℕ
  remainingWins : ℕ
  maxConsecutiveLosses : ℕ
  minConsecutiveWins : ℕ

/-- Calculates the total wins for a season -/
def totalWins (stats : SeasonStats) : ℕ :=
  stats.firstHundredWins + stats.remainingWins

/-- Calculates the win percentage for a season, excluding tie games -/
def winPercentage (stats : SeasonStats) : ℚ :=
  (totalWins stats : ℚ) / ((stats.totalGames - stats.tieGames) : ℚ)

/-- Theorem stating the total number of games played in the season -/
theorem football_season_games (stats : SeasonStats) 
  (h1 : 150 ≤ stats.totalGames ∧ stats.totalGames ≤ 200)
  (h2 : stats.firstHundredWins = 63)
  (h3 : stats.remainingWins = (stats.totalGames - 100) * 48 / 100)
  (h4 : stats.tieGames = 5)
  (h5 : winPercentage stats = 58 / 100)
  (h6 : stats.minConsecutiveWins ≥ 20)
  (h7 : stats.maxConsecutiveLosses ≤ 10) :
  stats.totalGames = 179 := by
  sorry


end NUMINAMATH_CALUDE_football_season_games_l2632_263201


namespace NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l2632_263293

/-- The set of positive integers of the form a^2 + 2b^2, where a and b are integers and b ≠ 0 -/
def A : Set ℕ+ :=
  {n : ℕ+ | ∃ (a b : ℤ), (b ≠ 0) ∧ (n : ℤ) = a^2 + 2*b^2}

/-- Theorem: If p is a prime number and p^2 is in A, then p is in A -/
theorem prime_square_in_A_implies_prime_in_A (p : ℕ+) (hp : Nat.Prime p) 
    (h_p_sq : (p^2 : ℕ+) ∈ A) : p ∈ A := by
  sorry

end NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l2632_263293


namespace NUMINAMATH_CALUDE_solution_set_equality_l2632_263220

-- Define the set S
def S : Set ℝ := {x : ℝ | |x + 3| - |x - 2| ≥ 3}

-- State the theorem
theorem solution_set_equality : S = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2632_263220


namespace NUMINAMATH_CALUDE_amusement_park_groups_l2632_263237

theorem amusement_park_groups (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_groups_l2632_263237


namespace NUMINAMATH_CALUDE_tan_difference_pi_4_minus_theta_l2632_263272

theorem tan_difference_pi_4_minus_theta (θ : Real) (h : Real.tan θ = 1/2) :
  Real.tan (π/4 - θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_pi_4_minus_theta_l2632_263272


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l2632_263286

theorem power_sum_equals_two : (-1 : ℝ)^2 + (1/3 : ℝ)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l2632_263286


namespace NUMINAMATH_CALUDE_lucy_disproves_tom_l2632_263298

-- Define the visible sides of the cards
def visible_numbers : List ℕ := [2, 4, 5, 7]
def visible_letters : List Char := ['B', 'C', 'D']

-- Define primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define consonant
def is_consonant (c : Char) : Prop := c ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

-- Tom's claim
def toms_claim (n : ℕ) (c : Char) : Prop := is_prime n → is_consonant c

-- Lucy's action
def lucy_flips_5 : Prop := ∃ (c : Char), c ∉ visible_letters ∧ ¬(is_consonant c)

-- Theorem to prove
theorem lucy_disproves_tom : 
  (∀ n ∈ visible_numbers, is_prime n → n ≠ 5 → ∃ c ∈ visible_letters, toms_claim n c) →
  lucy_flips_5 →
  ¬(∀ n c, toms_claim n c) :=
by sorry

end NUMINAMATH_CALUDE_lucy_disproves_tom_l2632_263298


namespace NUMINAMATH_CALUDE_number_percentage_l2632_263250

theorem number_percentage (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (40/100 : ℝ) * N = 192 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_l2632_263250


namespace NUMINAMATH_CALUDE_alternating_color_probability_l2632_263231

def total_balls : ℕ := 10
def white_balls : ℕ := 5
def black_balls : ℕ := 5

def alternating_sequences : ℕ := 2

def total_arrangements : ℕ := Nat.choose total_balls white_balls

theorem alternating_color_probability :
  (alternating_sequences : ℚ) / total_arrangements = 1 / 126 :=
sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l2632_263231


namespace NUMINAMATH_CALUDE_bus_speed_problem_l2632_263213

theorem bus_speed_problem (distance : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) :
  distance = 660 ∧ 
  speed_increase = 5 ∧ 
  time_reduction = 1 →
  ∃ (v : ℝ), 
    v > 0 ∧
    distance / v - time_reduction = distance / (v + speed_increase) ∧
    v = 55 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l2632_263213


namespace NUMINAMATH_CALUDE_equation_solution_range_l2632_263224

theorem equation_solution_range (x a : ℝ) : 
  (2 * x - 1 = x + a) → (x > 0) → (a > -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2632_263224


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2632_263281

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.cos (80 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2632_263281


namespace NUMINAMATH_CALUDE_line_intersects_and_passes_through_point_l2632_263221

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y + 2 * m - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 8 = 0

-- Theorem statement
theorem line_intersects_and_passes_through_point :
  ∀ m : ℝ,
  (∃ x y : ℝ, line_l m x y ∧ circle_C x y) ∧
  (line_l m (-2) 2) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_and_passes_through_point_l2632_263221


namespace NUMINAMATH_CALUDE_triangle_height_proof_l2632_263262

/-- Given a square with side length s, a rectangle with base s and height h,
    and an isosceles triangle with base s and height h,
    prove that h = 2s/3 when the combined area of the rectangle and triangle
    equals the area of the square. -/
theorem triangle_height_proof (s : ℝ) (h : ℝ) : 
  s > 0 → h > 0 → s * h + (s * h) / 2 = s^2 → h = 2 * s / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_proof_l2632_263262


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2632_263203

/-- A line passing through the point (-1, 2) and perpendicular to 2x - 3y + 4 = 0 has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  let point : ℝ × ℝ := (-1, 2)
  let given_line (x y : ℝ) := 2 * x - 3 * y + 4 = 0
  let perpendicular_line (x y : ℝ) := 3 * x + 2 * y - 1 = 0
  (∀ x y : ℝ, perpendicular_line x y ↔ 
    (x = point.1 ∧ y = point.2 ∨ 
     ∃ t : ℝ, x = point.1 + 3 * t ∧ y = point.2 + 2 * t)) ∧
  (∀ x y : ℝ, given_line x y → 
    ∀ x' y' : ℝ, perpendicular_line x' y' → 
      (x - x') * 2 + (y - y') * 3 = 0) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l2632_263203


namespace NUMINAMATH_CALUDE_sum_in_base7_l2632_263253

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem sum_in_base7 :
  toBase7 (toDecimal [4, 2, 3, 1] + toDecimal [1, 3, 5, 2, 6]) = [6, 0, 0, 6, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base7_l2632_263253


namespace NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l2632_263277

/-- Given two successive price reductions, calculates the second reduction percentage. -/
theorem second_reduction_percentage 
  (first_reduction : ℝ) 
  (final_price_percentage : ℝ) : ℝ :=
let remaining_after_first := 1 - first_reduction
let second_reduction := 1 - (final_price_percentage / remaining_after_first)
second_reduction

/-- Proves that for the given conditions, the second reduction percentage is 23.5%. -/
theorem store_price_reduction : 
  second_reduction_percentage 0.15 0.765 = 0.235 := by
sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l2632_263277


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2632_263270

theorem similar_triangles_leg_sum (A₁ A₂ : ℝ) (h : ℝ) (s : ℝ) :
  A₁ = 18 →
  A₂ = 288 →
  h = 9 →
  (A₂ / A₁ = (s / h) ^ 2) →
  s = 4 * Real.sqrt 153 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2632_263270


namespace NUMINAMATH_CALUDE_sum_of_odds_15_to_51_l2632_263227

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ := 
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem sum_of_odds_15_to_51 : 
  arithmetic_sum 15 51 2 = 627 := by sorry

end NUMINAMATH_CALUDE_sum_of_odds_15_to_51_l2632_263227


namespace NUMINAMATH_CALUDE_combination_equality_implies_seven_l2632_263294

theorem combination_equality_implies_seven (n : ℕ) : 
  (n.choose 3) = ((n-1).choose 3) + ((n-1).choose 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_seven_l2632_263294


namespace NUMINAMATH_CALUDE_log_identity_l2632_263289

theorem log_identity (a b : ℝ) (h1 : 0 < 3) (h2 : 0 < 2) (h3 : 0 < 7) 
  (ha : Real.log 2 / Real.log 3 = a) 
  (hb : Real.log 7 / Real.log 2 = b) : 
  Real.log 7 / Real.log 3 = a * b := by
sorry

end NUMINAMATH_CALUDE_log_identity_l2632_263289


namespace NUMINAMATH_CALUDE_problem_1_l2632_263287

theorem problem_1 (x : ℝ) (a : ℝ) : x - 1/x = 3 → a = x^2 + 1/x^2 → a = 11 := by
  sorry


end NUMINAMATH_CALUDE_problem_1_l2632_263287


namespace NUMINAMATH_CALUDE_tire_repair_cost_is_seven_l2632_263276

/-- The cost of repairing one tire without sales tax, given the total cost for 4 tires and the sales tax per tire. -/
def tire_repair_cost (total_cost : ℚ) (sales_tax : ℚ) : ℚ :=
  (total_cost - 4 * sales_tax) / 4

/-- Theorem stating that the cost of repairing one tire without sales tax is $7,
    given a total cost of $30 for 4 tires and a sales tax of $0.50 per tire. -/
theorem tire_repair_cost_is_seven :
  tire_repair_cost 30 0.5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tire_repair_cost_is_seven_l2632_263276


namespace NUMINAMATH_CALUDE_M_inter_N_empty_l2632_263208

/-- Set M of complex numbers -/
def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = (t / (1 + t) : ℂ) + Complex.I * ((1 + t) / t : ℂ)}

/-- Set N of complex numbers -/
def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = (Real.sqrt 2 : ℂ) * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

/-- Theorem stating that the intersection of M and N is empty -/
theorem M_inter_N_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_empty_l2632_263208


namespace NUMINAMATH_CALUDE_locus_not_hyperbola_ellipse_intersection_l2632_263297

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent (c1 c2 : Circle) : Prop :=
  dist c1.center c2.center = c1.radius + c2.radius ∨
  dist c1.center c2.center = |c1.radius - c2.radius|

def locus (O₁ O₂ : Circle) : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), tangent O₁ ⟨P, r⟩ ∧ tangent O₂ ⟨P, r⟩}

def hyperbola (f₁ f₂ : ℝ × ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {P | |dist P f₁ - dist P f₂| = 2 * a}

def ellipse (f₁ f₂ : ℝ × ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {P | dist P f₁ + dist P f₂ = 2 * a}

theorem locus_not_hyperbola_ellipse_intersection
  (O₁ O₂ : Circle) (f₁ f₂ g₁ g₂ : ℝ × ℝ) (a b : ℝ) :
  locus O₁ O₂ ≠ hyperbola f₁ f₂ a ∩ ellipse g₁ g₂ b :=
sorry

end NUMINAMATH_CALUDE_locus_not_hyperbola_ellipse_intersection_l2632_263297


namespace NUMINAMATH_CALUDE_tomato_soup_cans_l2632_263252

/-- Proves the number of cans of tomato soup sold for every 4 cans of chili beans -/
theorem tomato_soup_cans (total_cans : ℕ) (chili_beans_cans : ℕ) 
  (h1 : total_cans = 12)
  (h2 : chili_beans_cans = 8)
  (h3 : ∃ (n : ℕ), n * 4 = total_cans - chili_beans_cans) :
  4 = total_cans - chili_beans_cans :=
by sorry

end NUMINAMATH_CALUDE_tomato_soup_cans_l2632_263252


namespace NUMINAMATH_CALUDE_cars_to_sell_l2632_263271

/-- The number of cars each client selected -/
def cars_per_client : ℕ := 3

/-- The number of times each car was selected -/
def selections_per_car : ℕ := 3

/-- The number of clients who visited the garage -/
def num_clients : ℕ := 15

/-- The number of cars the seller has to sell -/
def num_cars : ℕ := 15

theorem cars_to_sell :
  num_cars * selections_per_car = num_clients * cars_per_client :=
by sorry

end NUMINAMATH_CALUDE_cars_to_sell_l2632_263271


namespace NUMINAMATH_CALUDE_simplify_expression_l2632_263273

theorem simplify_expression (x : ℝ) :
  3 * x + 10 * x^2 - 7 - (1 + 5 * x - 10 * x^2) = 20 * x^2 - 2 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2632_263273


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_greater_than_one_l2632_263267

/-- A linear function y = mx + b decreases if and only if its slope m is negative -/
axiom decreasing_linear_function (m b : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b > m * x₂ + b) ↔ m < 0

/-- For the function y = (1-a)x + 2, if it decreases as x increases, then a > 1 -/
theorem decreasing_function_implies_a_greater_than_one (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (1 - a) * x₁ + 2 > (1 - a) * x₂ + 2) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_greater_than_one_l2632_263267


namespace NUMINAMATH_CALUDE_squirrel_count_l2632_263205

theorem squirrel_count (total_acorns : ℕ) (needed_acorns : ℕ) (shortage : ℕ) : 
  total_acorns = 575 →
  needed_acorns = 130 →
  shortage = 15 →
  (total_acorns / (needed_acorns - shortage) : ℕ) = 5 := by
sorry

end NUMINAMATH_CALUDE_squirrel_count_l2632_263205


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l2632_263255

theorem smallest_divisible_by_18_and_24 : 
  ∃ n : ℕ, (n > 0 ∧ n % 18 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 18 = 0 ∧ m % 24 = 0) → n ≤ m) ∧ n = 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l2632_263255


namespace NUMINAMATH_CALUDE_committee_formation_count_l2632_263280

def schoolchildren : ℕ := 12
def teachers : ℕ := 3
def committee_size : ℕ := 9

theorem committee_formation_count :
  (Nat.choose (schoolchildren + teachers) committee_size) -
  (Nat.choose schoolchildren committee_size) = 4785 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2632_263280


namespace NUMINAMATH_CALUDE_irrational_zero_one_sequence_exists_l2632_263209

/-- A sequence representing the decimal digits of a number -/
def DecimalSequence := ℕ → Fin 10

/-- Checks if a decimal sequence contains only 0 and 1 -/
def OnlyZeroOne (s : DecimalSequence) : Prop :=
  ∀ n, s n = 0 ∨ s n = 1

/-- Checks if a decimal sequence has no two adjacent 1s -/
def NoAdjacentOnes (s : DecimalSequence) : Prop :=
  ∀ n, ¬(s n = 1 ∧ s (n + 1) = 1)

/-- Checks if a decimal sequence has no more than two adjacent 0s -/
def NoMoreThanTwoZeros (s : DecimalSequence) : Prop :=
  ∀ n, ¬(s n = 0 ∧ s (n + 1) = 0 ∧ s (n + 2) = 0)

/-- Checks if a decimal sequence represents an irrational number -/
def IsIrrational (s : DecimalSequence) : Prop :=
  ∀ k p, ∃ n ≥ k, s n ≠ s (n + p)

/-- There exists an irrational number whose decimal representation
    contains only 0 and 1, with no two adjacent 1s and no more than two adjacent 0s -/
theorem irrational_zero_one_sequence_exists : 
  ∃ s : DecimalSequence, 
    OnlyZeroOne s ∧ 
    NoAdjacentOnes s ∧ 
    NoMoreThanTwoZeros s ∧ 
    IsIrrational s := by
  sorry

end NUMINAMATH_CALUDE_irrational_zero_one_sequence_exists_l2632_263209


namespace NUMINAMATH_CALUDE_shooting_probabilities_shooting_probabilities_alt_l2632_263239

-- Define the probabilities given in the problem
def p_9_or_more : ℝ := 0.56
def p_8 : ℝ := 0.22
def p_7 : ℝ := 0.12

-- Theorem statement
theorem shooting_probabilities :
  let p_less_than_8 := 1 - p_9_or_more - p_8
  let p_at_least_7 := p_7 + p_8 + p_9_or_more
  (p_less_than_8 = 0.22) ∧ (p_at_least_7 = 0.9) := by
  sorry

-- Alternative formulation using complement for p_at_least_7
theorem shooting_probabilities_alt :
  let p_less_than_7 := 1 - p_9_or_more - p_8 - p_7
  let p_less_than_8 := p_less_than_7 + p_7
  let p_at_least_7 := 1 - p_less_than_7
  (p_less_than_8 = 0.22) ∧ (p_at_least_7 = 0.9) := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_shooting_probabilities_alt_l2632_263239


namespace NUMINAMATH_CALUDE_imaginary_part_reciprocal_l2632_263216

theorem imaginary_part_reciprocal (a : ℝ) : Complex.im (1 / (a - Complex.I)) = 1 / (1 + a^2) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_reciprocal_l2632_263216


namespace NUMINAMATH_CALUDE_cube_layer_removal_l2632_263264

/-- Represents a cube composed of smaller cubes -/
structure Cube where
  side_length : Nat
  total_cubes : Nat

/-- Calculates the number of cubes in one layer -/
def layer_size (c : Cube) : Nat :=
  c.side_length * c.side_length

/-- Calculates the number of remaining cubes after removing one layer -/
def remaining_cubes (c : Cube) : Nat :=
  c.total_cubes - layer_size c

/-- Theorem stating that for a 10x10x10 cube, 900 cubes remain after removing one layer -/
theorem cube_layer_removal :
  ∀ c : Cube, c.side_length = 10 ∧ c.total_cubes = 1000 →
  remaining_cubes c = 900 := by
  sorry

end NUMINAMATH_CALUDE_cube_layer_removal_l2632_263264


namespace NUMINAMATH_CALUDE_snowball_theorem_l2632_263248

/-- Represents the action of throwing a snowball -/
def throws (n : Nat) (m : Nat) : Prop := sorry

/-- The number of children -/
def num_children : Nat := 43

theorem snowball_theorem :
  (∀ i : Nat, i > 0 ∧ i ≤ num_children → ∃! j : Nat, j > 0 ∧ j ≤ num_children ∧ throws i j) ∧
  (∀ j : Nat, j > 0 ∧ j ≤ num_children → ∃! i : Nat, i > 0 ∧ i ≤ num_children ∧ throws i j) ∧
  (∃ x : Nat, x > 0 ∧ x ≤ num_children ∧ throws 1 x ∧ throws x 2) ∧
  (∃ y : Nat, y > 0 ∧ y ≤ num_children ∧ throws 2 y ∧ throws y 3) ∧
  (∃ z : Nat, z > 0 ∧ z ≤ num_children ∧ throws num_children z ∧ throws z 1) →
  ∃ w : Nat, w = 24 ∧ throws w 3 := by sorry

end NUMINAMATH_CALUDE_snowball_theorem_l2632_263248


namespace NUMINAMATH_CALUDE_sqrt_square_abs_two_div_sqrt_two_l2632_263241

-- Theorem 1: For any real number x, sqrt(x^2) = |x|
theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

-- Theorem 2: 2 / sqrt(2) = sqrt(2)
theorem two_div_sqrt_two : 2 / Real.sqrt 2 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_two_div_sqrt_two_l2632_263241


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_evaluation_l2632_263210

theorem complex_arithmetic_expression_evaluation : 
  let a := (8/7 - 23/49) / (22/147)
  let b := (0.6 / (15/4)) * (5/2)
  let c := 3.75 / (3/2)
  ((a - b + c) / 2.2) = 3 := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_evaluation_l2632_263210


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l2632_263246

def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem intersection_nonempty_implies_a_geq_neg_one (a : ℝ) :
  (M ∩ N a).Nonempty → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l2632_263246


namespace NUMINAMATH_CALUDE_playground_area_l2632_263214

theorem playground_area (perimeter width length : ℝ) : 
  perimeter = 120 →
  length = 3 * width →
  2 * length + 2 * width = perimeter →
  length * width = 675 :=
by
  sorry

end NUMINAMATH_CALUDE_playground_area_l2632_263214


namespace NUMINAMATH_CALUDE_croissant_making_time_l2632_263279

-- Define the constants
def fold_time : ℕ := 5
def fold_count : ℕ := 4
def rest_time : ℕ := 75
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

-- Define the theorem
theorem croissant_making_time :
  (fold_time * fold_count + 
   rest_time * fold_count + 
   mixing_time + 
   baking_time) / 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_croissant_making_time_l2632_263279


namespace NUMINAMATH_CALUDE_rectangle_region_perimeter_l2632_263202

/-- Given a region formed by four congruent rectangles with a total area of 360 square centimeters
    and each rectangle having a width to length ratio of 3:4, the perimeter of the region is 14√7.5 cm. -/
theorem rectangle_region_perimeter (total_area : ℝ) (width : ℝ) (length : ℝ) : 
  total_area = 360 →
  width / length = 3 / 4 →
  width * length = total_area / 4 →
  2 * (2 * width + 2 * length) = 14 * Real.sqrt 7.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_region_perimeter_l2632_263202


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_is_9_6_l2632_263204

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : std_dev > 0

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 12 and standard deviation 1.2,
    the value that is exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_below_mean_is_9_6 :
  let d : NormalDistribution := ⟨12, 1.2, by norm_num⟩
  value_n_std_dev_below_mean d 2 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_is_9_6_l2632_263204


namespace NUMINAMATH_CALUDE_max_ratio_squared_l2632_263299

theorem max_ratio_squared (a b c y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≥ b) (hbc : b ≥ c)
  (hy : 0 ≤ y ∧ y < a) (hz : 0 ≤ z ∧ z < c)
  (heq : a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2) :
  (a / c)^2 ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l2632_263299


namespace NUMINAMATH_CALUDE_least_multiple_36_with_digit_product_multiple_9_l2632_263292

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let digits := n.digits 10
  digits.foldl (· * ·) 1

theorem least_multiple_36_with_digit_product_multiple_9 :
  ∀ n : ℕ, n > 0 →
    is_multiple_of n 36 →
    is_multiple_of (digit_product n) 9 →
    n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_36_with_digit_product_multiple_9_l2632_263292


namespace NUMINAMATH_CALUDE_equal_split_probability_eight_dice_l2632_263278

theorem equal_split_probability_eight_dice (n : ℕ) (p : ℝ) : 
  n = 8 →
  p = 1 / 2 →
  (n.choose (n / 2)) * p^n = 35 / 128 :=
by sorry

end NUMINAMATH_CALUDE_equal_split_probability_eight_dice_l2632_263278


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l2632_263200

theorem sum_of_squared_coefficients : 
  let expression := fun x : ℝ => 5 * (x^3 - x) - 3 * (x^2 - 4*x + 3)
  let simplified := fun x : ℝ => 5*x^3 - 3*x^2 + 7*x - 9
  (∀ x : ℝ, expression x = simplified x) →
  (5^2 + (-3)^2 + 7^2 + (-9)^2 = 164) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l2632_263200


namespace NUMINAMATH_CALUDE_triangles_in_polygon_l2632_263235

/-- The number of triangles formed by diagonals passing through one vertex of an n-sided polygon -/
def triangles_from_diagonals (n : ℕ) : ℕ :=
  n - 2

/-- Theorem stating that the number of triangles formed by diagonals passing through one vertex
    of an n-sided polygon is equal to (n-2) -/
theorem triangles_in_polygon (n : ℕ) (h : n ≥ 3) :
  triangles_from_diagonals n = n - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_polygon_l2632_263235


namespace NUMINAMATH_CALUDE_actor_stage_time_l2632_263236

theorem actor_stage_time (actors_at_once : ℕ) (total_actors : ℕ) (show_duration : ℕ) : 
  actors_at_once = 5 → total_actors = 20 → show_duration = 60 → 
  (show_duration / (total_actors / actors_at_once) : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_actor_stage_time_l2632_263236


namespace NUMINAMATH_CALUDE_last_digit_is_square_of_second_l2632_263245

/-- Represents a 4-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  is_four_digit : d1 ≠ 0 ∧ d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10

/-- The specific 4-digit number 1349 -/
def number : FourDigitNumber where
  d1 := 1
  d2 := 3
  d3 := 4
  d4 := 9
  is_four_digit := by sorry

theorem last_digit_is_square_of_second :
  (number.d1 = number.d2 / 3) →
  (number.d3 = number.d1 + number.d2) →
  (number.d4 = number.d2 * number.d2) := by sorry

end NUMINAMATH_CALUDE_last_digit_is_square_of_second_l2632_263245


namespace NUMINAMATH_CALUDE_upper_limit_of_b_l2632_263206

theorem upper_limit_of_b (a b : ℤ) (h1 : 9 ≤ a ∧ a ≤ 14) (h2 : b ≥ 7) 
  (h3 : (14 : ℚ) / 7 - (9 : ℚ) / b = 1.55) : b ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_of_b_l2632_263206


namespace NUMINAMATH_CALUDE_moses_esther_difference_l2632_263295

theorem moses_esther_difference (total : ℝ) (moses_percentage : ℝ) : 
  total = 50 ∧ moses_percentage = 0.4 → 
  let moses_share := moses_percentage * total
  let remainder := total - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by
sorry

end NUMINAMATH_CALUDE_moses_esther_difference_l2632_263295


namespace NUMINAMATH_CALUDE_prob_heart_or_king_two_draws_l2632_263288

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 52)
  (hearts : ℕ := 13)
  (kings : ℕ := 4)
  (heart_king_overlap : ℕ := 1)

/-- Calculates the probability of drawing at least one heart or king in two draws with replacement -/
def prob_at_least_one_heart_or_king (d : Deck) : ℚ :=
  1 - (((d.total_cards - (d.hearts + d.kings - d.heart_king_overlap)) / d.total_cards) ^ 2)

/-- Theorem stating that the probability of drawing at least one heart or king
    in two draws with replacement from a standard deck is 88/169 -/
theorem prob_heart_or_king_two_draws (d : Deck) :
  prob_at_least_one_heart_or_king d = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_or_king_two_draws_l2632_263288


namespace NUMINAMATH_CALUDE_log_equality_l2632_263223

theorem log_equality : Real.log 16 / Real.log 4096 = Real.log 4 / Real.log 64 := by
  sorry

#check log_equality

end NUMINAMATH_CALUDE_log_equality_l2632_263223


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2632_263238

theorem largest_angle_in_triangle : ∀ x : ℝ,
  x + 35 + 70 = 180 →
  max x (max 35 70) = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2632_263238


namespace NUMINAMATH_CALUDE_unique_solution_for_class_representatives_l2632_263215

theorem unique_solution_for_class_representatives (m n : ℕ) : 
  10 ≥ m ∧ m > n ∧ n ≥ 4 →
  ((m - n)^2 = m + n) ↔ (m = 10 ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_class_representatives_l2632_263215


namespace NUMINAMATH_CALUDE_boatman_distance_against_current_l2632_263207

/-- Represents the speed of a boat in different water conditions -/
structure BoatSpeed where
  stillWater : ℝ
  current : ℝ

/-- Calculates the distance traveled given speed and time -/
def distanceTraveled (speed time : ℝ) : ℝ := speed * time

/-- Represents the problem of a boatman traveling in a stream -/
theorem boatman_distance_against_current 
  (boat : BoatSpeed)
  (h1 : distanceTraveled (boat.stillWater + boat.current) (1/3) = 1)
  (h2 : distanceTraveled boat.stillWater 3 = 6)
  (h3 : boat.stillWater > boat.current)
  (h4 : boat.current > 0) :
  distanceTraveled (boat.stillWater - boat.current) 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boatman_distance_against_current_l2632_263207


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2632_263218

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 41 →
  a 4 * a 8 = 5 →
  a 4 + a 8 = Real.sqrt 51 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2632_263218


namespace NUMINAMATH_CALUDE_man_twice_son_age_l2632_263234

/-- Represents the number of years until a man's age is twice his son's age. -/
def yearsUntilTwiceAge (sonAge : ℕ) (ageDifference : ℕ) : ℕ :=
  2

theorem man_twice_son_age (sonAge : ℕ) (ageDifference : ℕ) 
  (h1 : sonAge = 25) 
  (h2 : ageDifference = 27) : 
  yearsUntilTwiceAge sonAge ageDifference = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_twice_son_age_l2632_263234


namespace NUMINAMATH_CALUDE_intersection_A_B_l2632_263244

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2632_263244


namespace NUMINAMATH_CALUDE_performances_distribution_l2632_263285

/-- The number of ways to distribute performances among classes -/
def distribute_performances (total : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  Nat.choose (total - num_classes * min_per_class + num_classes - 1) (num_classes - 1)

/-- Theorem stating the number of ways to distribute 14 performances among 3 classes -/
theorem performances_distribution :
  distribute_performances 14 3 3 = 21 := by sorry

end NUMINAMATH_CALUDE_performances_distribution_l2632_263285


namespace NUMINAMATH_CALUDE_ones_digit_of_17_power_l2632_263291

theorem ones_digit_of_17_power : ∃ n : ℕ, 17^(17*(13^13)) ≡ 7 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_17_power_l2632_263291


namespace NUMINAMATH_CALUDE_basketball_handshakes_l2632_263263

theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l2632_263263


namespace NUMINAMATH_CALUDE_range_of_a_when_quadratic_nonnegative_l2632_263274

theorem range_of_a_when_quadratic_nonnegative (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a ≥ 0) → a ∈ Set.Icc 0 4 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_when_quadratic_nonnegative_l2632_263274


namespace NUMINAMATH_CALUDE_find_number_l2632_263233

theorem find_number : ∃ x : ℝ, x + 0.303 + 0.432 = 5.485 ∧ x = 4.750 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2632_263233


namespace NUMINAMATH_CALUDE_batsman_average_l2632_263251

theorem batsman_average (total_matches : ℕ) (first_set_matches : ℕ) (first_set_average : ℝ) (total_average : ℝ) :
  total_matches = 30 →
  first_set_matches = 20 →
  first_set_average = 30 →
  total_average = 25 →
  let second_set_matches := total_matches - first_set_matches
  let second_set_average := (total_average * total_matches - first_set_average * first_set_matches) / second_set_matches
  second_set_average = 15 := by sorry

end NUMINAMATH_CALUDE_batsman_average_l2632_263251


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l2632_263242

theorem bowling_ball_volume :
  let sphere_diameter : ℝ := 40
  let hole1_depth : ℝ := 10
  let hole1_diameter : ℝ := 5
  let hole2_depth : ℝ := 12
  let hole2_diameter : ℝ := 4
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let hole1_volume := π * (hole1_diameter / 2)^2 * hole1_depth
  let hole2_volume := π * (hole2_diameter / 2)^2 * hole2_depth
  sphere_volume - hole1_volume - hole2_volume = 10556.17 * π :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l2632_263242


namespace NUMINAMATH_CALUDE_max_c_value_l2632_263211

theorem max_c_value (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) :
  c ≤ 47 ∧ ∃ d₀, 5 * 47 + (d₀ - 12)^2 = 235 := by
  sorry

end NUMINAMATH_CALUDE_max_c_value_l2632_263211


namespace NUMINAMATH_CALUDE_problem_solution_l2632_263282

noncomputable section

def f (x : ℝ) := 3 - 2 * Real.log x / Real.log 2
def g (x : ℝ) := Real.log x / Real.log 2
def h (x : ℝ) := (f x + 1) * g x
def M (x : ℝ) := max (g x) (f x)

theorem problem_solution :
  (∀ x ∈ Set.Icc 1 8, h x ∈ Set.Icc (-6) 2) ∧
  (∀ x > 0, M x ≤ 1) ∧
  (∃ x > 0, M x = 1) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Icc 1 8, f (x^2) * f (Real.sqrt x) ≥ k * g x) → k ≤ -3) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2632_263282


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2632_263259

/-- A line passing through (2,4) and tangent to (x-1)^2 + (y-2)^2 = 1 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 2 ∨ (3 * p.1 - 4 * p.2 + 10 = 0)}

/-- The circle (x-1)^2 + (y-2)^2 = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 1}

/-- The point (2,4) -/
def Point : ℝ × ℝ := (2, 4)

theorem tangent_line_equation :
  ∀ (L : Set (ℝ × ℝ)),
    (Point ∈ L) →
    (∃! p, p ∈ L ∩ Circle) →
    L = TangentLine :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2632_263259


namespace NUMINAMATH_CALUDE_marching_band_members_l2632_263225

theorem marching_band_members :
  ∃! n : ℕ, 100 < n ∧ n < 200 ∧
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 7 = 3 ∧
  n = 157 := by sorry

end NUMINAMATH_CALUDE_marching_band_members_l2632_263225


namespace NUMINAMATH_CALUDE_cos_square_sum_simplification_l2632_263212

theorem cos_square_sum_simplification (x y : ℝ) : 
  (Real.cos x)^2 + (Real.cos (x + y))^2 - 2 * (Real.cos x) * (Real.cos y) * (Real.cos (x + y)) = (Real.sin y)^2 := by
  sorry

end NUMINAMATH_CALUDE_cos_square_sum_simplification_l2632_263212


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l2632_263290

/-- Represents Cherry's delivery service earnings --/
def cherry_earnings : ℕ → ℚ
| 5 => 2.5  -- $2.50 for 5 kg cargo
| 8 => 4    -- $4 for 8 kg cargo
| _ => 0    -- Default case

/-- Calculates Cherry's daily earnings --/
def daily_earnings : ℚ :=
  4 * cherry_earnings 5 + 2 * cherry_earnings 8

/-- Theorem: Cherry's weekly earnings are $126 --/
theorem cherry_weekly_earnings :
  7 * daily_earnings = 126 := by
  sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l2632_263290


namespace NUMINAMATH_CALUDE_max_cross_pattern_sum_l2632_263243

/-- Represents the cross-shaped pattern -/
structure CrossPattern where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 6, 9, 11, 14}

/-- Checks if the pattern satisfies the sum conditions -/
def isValidPattern (p : CrossPattern) : Prop :=
  p.a + p.b + p.e = p.a + p.c + p.e ∧
  p.a + p.c + p.e = p.b + p.d + p.e ∧
  p.a + p.d = p.b + p.c

/-- Checks if the pattern uses all available numbers exactly once -/
def usesAllNumbers (p : CrossPattern) : Prop :=
  {p.a, p.b, p.c, p.d, p.e} = availableNumbers

/-- The sum of any row, column, or diagonal in a valid pattern -/
def patternSum (p : CrossPattern) : ℕ := p.a + p.b + p.e

/-- Theorem: The maximum sum in a valid cross pattern is 31 -/
theorem max_cross_pattern_sum :
  ∀ p : CrossPattern,
    isValidPattern p →
    usesAllNumbers p →
    patternSum p ≤ 31 :=
sorry

end NUMINAMATH_CALUDE_max_cross_pattern_sum_l2632_263243


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2632_263284

theorem complex_number_in_first_quadrant (z : ℂ) : z = 2 + I → z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2632_263284


namespace NUMINAMATH_CALUDE_bird_nest_problem_l2632_263256

theorem bird_nest_problem (first_bird_initial : Nat) (first_bird_additional : Nat)
                          (second_bird_initial : Nat) (second_bird_additional : Nat)
                          (third_bird_initial : Nat) (third_bird_additional : Nat)
                          (first_bird_carry_capacity : Nat) (tree_drop_fraction : Nat) :
  first_bird_initial = 12 →
  first_bird_additional = 6 →
  second_bird_initial = 15 →
  second_bird_additional = 8 →
  third_bird_initial = 10 →
  third_bird_additional = 4 →
  first_bird_carry_capacity = 3 →
  tree_drop_fraction = 3 →
  (first_bird_initial * first_bird_additional +
   second_bird_initial * second_bird_additional +
   third_bird_initial * third_bird_additional = 232) ∧
  (((first_bird_initial * first_bird_additional) -
    (first_bird_initial * first_bird_additional / tree_drop_fraction)) /
    first_bird_carry_capacity = 16) :=
by sorry

end NUMINAMATH_CALUDE_bird_nest_problem_l2632_263256


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l2632_263222

theorem quadratic_square_completion (x : ℝ) : 
  (x^2 + 10*x + 9 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l2632_263222


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l2632_263269

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  probHeads : ℚ
  probTails : ℚ
  payoffHeads : ℚ
  payoffTails : ℚ
  prob_sum : probHeads + probTails = 1

/-- The expected value of a flip of the unfair coin -/
def expectedValue (coin : UnfairCoin) : ℚ :=
  coin.probHeads * coin.payoffHeads + coin.probTails * coin.payoffTails

/-- Theorem: The expected worth of a specific unfair coin flip -/
theorem expected_worth_unfair_coin :
  ∃ (coin : UnfairCoin),
    coin.probHeads = 2/3 ∧
    coin.probTails = 1/3 ∧
    coin.payoffHeads = 5 ∧
    coin.payoffTails = -9 ∧
    expectedValue coin = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l2632_263269


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l2632_263230

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) 
  (container_capacity : ℕ) (min_total_eggs : ℕ) :
  total_containers > 10 ∧ 
  deficient_containers = 3 ∧ 
  container_capacity = 15 ∧ 
  min_total_eggs = 150 →
  (total_containers * container_capacity - deficient_containers = 
    (total_containers - deficient_containers) * container_capacity + 
    deficient_containers * (container_capacity - 1)) ∧
  (total_containers * container_capacity - deficient_containers > min_total_eggs) ∧
  ∀ n : ℕ, n < total_containers → 
    n * container_capacity - deficient_containers ≤ min_total_eggs :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l2632_263230
