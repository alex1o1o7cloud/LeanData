import Mathlib

namespace NUMINAMATH_CALUDE_line_equations_l272_27280

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def linesParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Function to check if two lines are perpendicular
def linesPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equations :
  let p1 : Point2D := ⟨1, 2⟩
  let p2 : Point2D := ⟨1, 1⟩
  let l1 : Line2D := ⟨1, 1, -1⟩  -- x + y - 1 = 0
  let l2 : Line2D := ⟨3, 1, -1⟩  -- 3x + y - 1 = 0
  let result1 : Line2D := ⟨1, 1, -3⟩  -- x + y - 3 = 0
  let result2 : Line2D := ⟨1, -3, 2⟩  -- x - 3y + 2 = 0
  (pointOnLine p1 result1 ∧ linesParallel result1 l1) ∧
  (pointOnLine p2 result2 ∧ linesPerpendicular result2 l2) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l272_27280


namespace NUMINAMATH_CALUDE_shooter_probabilities_l272_27260

def hit_probability : ℚ := 4/5

def exactly_eight_hits (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1-p)^(n-k)

def at_least_eight_hits (n : ℕ) (p : ℚ) : ℚ :=
  exactly_eight_hits n 8 p + exactly_eight_hits n 9 p + p^n

theorem shooter_probabilities :
  (exactly_eight_hits 10 8 hit_probability = 
    Nat.choose 10 8 * (4/5)^8 * (1/5)^2) ∧
  (at_least_eight_hits 10 hit_probability = 
    Nat.choose 10 8 * (4/5)^8 * (1/5)^2 + 
    Nat.choose 10 9 * (4/5)^9 * (1/5) + 
    (4/5)^10) := by
  sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l272_27260


namespace NUMINAMATH_CALUDE_product_lmn_equals_one_l272_27262

theorem product_lmn_equals_one 
  (p q r l m n : ℂ)
  (distinct_pqr : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (distinct_lmn : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (nonzero_lmn : l ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0)
  (eq1 : p / (1 - q) = l)
  (eq2 : q / (1 - r) = m)
  (eq3 : r / (1 - p) = n) :
  l * m * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_lmn_equals_one_l272_27262


namespace NUMINAMATH_CALUDE_linear_equation_natural_solution_l272_27238

theorem linear_equation_natural_solution (m : ℤ) : 
  (∃ x : ℕ, m * (x : ℤ) - 6 = x) ↔ m ∈ ({2, 3, 4, 7} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_linear_equation_natural_solution_l272_27238


namespace NUMINAMATH_CALUDE_exponential_function_first_quadrant_l272_27240

theorem exponential_function_first_quadrant (m : ℝ) :
  (∀ x : ℝ, x > 0 → (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -1/5 := by sorry

end NUMINAMATH_CALUDE_exponential_function_first_quadrant_l272_27240


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_min_value_l272_27293

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set (x : ℝ) : f x ≤ x + 1 ↔ 1 ≤ x ∧ x ≤ 5 := by sorry

-- Theorem 2: Inequality proof
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 / (a + 1) + b^2 / (b + 1) ≥ 1 := by sorry

-- Theorem to show that the minimum value of f(x) is 2
theorem min_value : ∀ x : ℝ, f x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_min_value_l272_27293


namespace NUMINAMATH_CALUDE_opposite_signs_for_positive_solution_l272_27203

theorem opposite_signs_for_positive_solution (a b : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ x : ℝ, x > 0 ∧ a * x + b = 0) : a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_for_positive_solution_l272_27203


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l272_27200

theorem complex_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  1 / ((x - 2) * (x - 4)) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l272_27200


namespace NUMINAMATH_CALUDE_min_box_height_l272_27206

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- States that the surface area is at least 150 square units -/
def surface_area_constraint : ℝ → Prop := λ x => surface_area x ≥ 150

theorem min_box_height :
  ∃ x : ℝ, x > 0 ∧ surface_area_constraint x ∧
    box_height x = 10 ∧
    ∀ y : ℝ, y > 0 ∧ surface_area_constraint y → surface_area x ≤ surface_area y :=
by sorry

end NUMINAMATH_CALUDE_min_box_height_l272_27206


namespace NUMINAMATH_CALUDE_circle_area_l272_27278

/-- The area of the circle defined by 3x^2 + 3y^2 - 12x + 18y + 27 = 0 is 4π. -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ 
    π * radius^2 = 4 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l272_27278


namespace NUMINAMATH_CALUDE_m_union_n_eq_n_l272_27276

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x^2 < 2}

-- State the theorem
theorem m_union_n_eq_n : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_m_union_n_eq_n_l272_27276


namespace NUMINAMATH_CALUDE_annual_decrease_rate_l272_27237

/-- Proves that the annual decrease rate is 20% for a town with given population changes. -/
theorem annual_decrease_rate (initial_population : ℝ) (population_after_two_years : ℝ) 
  (h1 : initial_population = 15000)
  (h2 : population_after_two_years = 9600) :
  ∃ (r : ℝ), r = 20 ∧ population_after_two_years = initial_population * (1 - r / 100)^2 := by
  sorry

end NUMINAMATH_CALUDE_annual_decrease_rate_l272_27237


namespace NUMINAMATH_CALUDE_absolute_value_and_exponentiation_calculation_l272_27225

theorem absolute_value_and_exponentiation_calculation : 
  |1 - 3| * ((-12) - 2^3) = -40 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponentiation_calculation_l272_27225


namespace NUMINAMATH_CALUDE_base_89_multiple_of_13_l272_27209

theorem base_89_multiple_of_13 (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : (142536472 : ℤ) ≡ b [ZMOD 13]) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_89_multiple_of_13_l272_27209


namespace NUMINAMATH_CALUDE_money_calculation_l272_27277

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalMoney (n50 : ℕ) (n500 : ℕ) : ℕ :=
  50 * n50 + 500 * n500

/-- Theorem stating that given 90 notes in total, with 77 being 50 rupee notes,
    the total amount of money is 10350 rupees -/
theorem money_calculation :
  let total_notes : ℕ := 90
  let n50 : ℕ := 77
  let n500 : ℕ := total_notes - n50
  totalMoney n50 n500 = 10350 := by
sorry

end NUMINAMATH_CALUDE_money_calculation_l272_27277


namespace NUMINAMATH_CALUDE_square_number_divisible_by_nine_between_40_and_90_l272_27263

theorem square_number_divisible_by_nine_between_40_and_90 :
  ∃ x : ℕ, x^2 = x ∧ x % 9 = 0 ∧ 40 < x ∧ x < 90 → x = 81 :=
by sorry

end NUMINAMATH_CALUDE_square_number_divisible_by_nine_between_40_and_90_l272_27263


namespace NUMINAMATH_CALUDE_count_without_one_between_1_and_2000_l272_27250

/-- Count of numbers without digit 1 in a given range -/
def count_without_digit_one (lower : Nat) (upper : Nat) : Nat :=
  sorry

/-- The main theorem -/
theorem count_without_one_between_1_and_2000 :
  count_without_digit_one 1 2000 = 1457 := by sorry

end NUMINAMATH_CALUDE_count_without_one_between_1_and_2000_l272_27250


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l272_27247

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance_covered := train_speed_ms * crossing_time
  distance_covered - train_length

/-- The length of the bridge is 215 meters -/
theorem bridge_length_problem : bridge_length 160 45 30 = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l272_27247


namespace NUMINAMATH_CALUDE_original_fraction_l272_27287

theorem original_fraction (x y : ℚ) : 
  (x > 0) → (y > 0) → 
  ((6/5 * x) / (9/10 * y) = 20/21) → 
  (x / y = 10/21) := by
sorry

end NUMINAMATH_CALUDE_original_fraction_l272_27287


namespace NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l272_27208

/-- Converts cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (yards : ℝ) : ℝ :=
  yards * 27

theorem five_cubic_yards_equals_135_cubic_feet :
  cubic_yards_to_cubic_feet 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l272_27208


namespace NUMINAMATH_CALUDE_round_trip_time_l272_27233

/-- Calculates the total time for a round trip boat journey -/
theorem round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 9)
  (h2 : stream_speed = 6)
  (h3 : distance = 170) :
  (distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed)) = 68 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_time_l272_27233


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l272_27249

-- Define the function f(m, x)
def f (m x : ℝ) : ℝ := m * (x^2 - 1) - 1 - 8*x

-- State the theorem
theorem x_range_for_inequality :
  (∀ x : ℝ, (∀ m : ℝ, -1 ≤ m ∧ m ≤ 4 → f m x < 0) ↔ 0 < x ∧ x < 5/2) :=
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l272_27249


namespace NUMINAMATH_CALUDE_nine_team_league_games_l272_27256

/-- The number of games played in a league where each team plays every other team once -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 9 teams, where each team plays every other team exactly once,
    the total number of games played is 36. -/
theorem nine_team_league_games :
  numGames 9 = 36 := by
  sorry


end NUMINAMATH_CALUDE_nine_team_league_games_l272_27256


namespace NUMINAMATH_CALUDE_smallest_median_l272_27204

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 4, 3, 6}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_median :
  ∀ x : ℤ, ∃ m : ℤ, is_median m (number_set x) ∧ 
  (∀ m' : ℤ, is_median m' (number_set x) → m ≤ m') ∧
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_median_l272_27204


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_five_l272_27236

/-- Given a function f(x) = 4x^3 - ax^2 - 2x + b with an extremum at x = 1, prove that a = 5 --/
theorem extremum_implies_a_equals_five (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => 4*x^3 - a*x^2 - 2*x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_five_l272_27236


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l272_27265

/-- A pyramid with a parallelogram base and specific dimensions --/
structure Pyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_diagonal : ℝ
  lateral_edge : ℝ

/-- The volume of the pyramid --/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific pyramid is 200 --/
theorem specific_pyramid_volume :
  let p : Pyramid := {
    base_side1 := 9,
    base_side2 := 10,
    base_diagonal := 11,
    lateral_edge := Real.sqrt 10
  }
  pyramid_volume p = 200 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l272_27265


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l272_27239

theorem min_value_theorem (x : ℝ) (h : x > 0) : 2*x + 1/(2*x) + 1 ≥ 3 := by
  sorry

theorem equality_exists : ∃ x : ℝ, x > 0 ∧ 2*x + 1/(2*x) + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l272_27239


namespace NUMINAMATH_CALUDE_sandy_has_144_marbles_l272_27284

-- Define the number of red marbles Jessica has
def jessica_marbles : ℕ := 3 * 12

-- Define the relationship between Sandy's and Jessica's marbles
def sandy_marbles : ℕ := 4 * jessica_marbles

-- Theorem to prove
theorem sandy_has_144_marbles : sandy_marbles = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_has_144_marbles_l272_27284


namespace NUMINAMATH_CALUDE_log_comparison_l272_27222

theorem log_comparison : Real.log 80 / Real.log 20 < Real.log 640 / Real.log 80 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l272_27222


namespace NUMINAMATH_CALUDE_petrol_price_increase_l272_27231

theorem petrol_price_increase (original_price original_consumption : ℝ) 
  (h : original_price > 0) (h2 : original_consumption > 0) :
  let new_consumption := original_consumption * (1 - 1/6)
  let price_increase_factor := (original_price * original_consumption) / (original_price * new_consumption)
  price_increase_factor = 1.2 := by
sorry

end NUMINAMATH_CALUDE_petrol_price_increase_l272_27231


namespace NUMINAMATH_CALUDE_sum_of_21st_group_l272_27286

/-- The first number in the n-th group -/
def first_number (n : ℕ) : ℕ := n * (n - 1) / 2 + 1

/-- The last number in the n-th group -/
def last_number (n : ℕ) : ℕ := first_number n + (n - 1)

/-- The sum of numbers in the n-th group -/
def group_sum (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

/-- Theorem: The sum of numbers in the 21st group is 4641 -/
theorem sum_of_21st_group : group_sum 21 = 4641 := by sorry

end NUMINAMATH_CALUDE_sum_of_21st_group_l272_27286


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_six_l272_27246

theorem largest_three_digit_divisible_by_six :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 6 = 0 → n ≤ 996 ∧ 996 % 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_six_l272_27246


namespace NUMINAMATH_CALUDE_constant_k_value_l272_27298

/-- Given that -x^2 - (k + 10)x - 8 = -(x - 2)(x - 4) for all real x, prove that k = -16 -/
theorem constant_k_value (k : ℝ) 
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 := by sorry

end NUMINAMATH_CALUDE_constant_k_value_l272_27298


namespace NUMINAMATH_CALUDE_power_quotient_plus_five_l272_27268

theorem power_quotient_plus_five : 23^12 / 23^5 + 5 = 148035894 := by
  sorry

end NUMINAMATH_CALUDE_power_quotient_plus_five_l272_27268


namespace NUMINAMATH_CALUDE_circle_in_square_l272_27264

theorem circle_in_square (r : ℝ) (h : r = 6) :
  let square_side := 2 * r
  let square_area := square_side ^ 2
  let smaller_square_side := square_side - 2
  let smaller_square_area := smaller_square_side ^ 2
  (square_area = 144 ∧ square_area - smaller_square_area = 44) := by
  sorry

end NUMINAMATH_CALUDE_circle_in_square_l272_27264


namespace NUMINAMATH_CALUDE_expression_defined_iff_l272_27245

-- Define the set of real numbers for which the expression is defined
def valid_x : Set ℝ := {x | x ∈ Set.Ioo (-Real.sqrt 5) 1 ∪ Set.Ioo 3 (Real.sqrt 5)}

-- Define the conditions for the expression to be defined
def conditions (x : ℝ) : Prop :=
  x^2 - 4*x + 3 > 0 ∧ 5 - x^2 > 0

-- Theorem statement
theorem expression_defined_iff (x : ℝ) :
  conditions x ↔ x ∈ valid_x := by sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l272_27245


namespace NUMINAMATH_CALUDE_expand_and_simplify_expression_l272_27259

theorem expand_and_simplify_expression (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_expression_l272_27259


namespace NUMINAMATH_CALUDE_area_triangle_PAB_l272_27254

/-- Given points A(-1, 2), B(3, 4), and P on the x-axis such that |PA| = |PB|,
    the area of triangle PAB is 15/2. -/
theorem area_triangle_PAB :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (3, 4)
  ∀ P : ℝ × ℝ,
    P.2 = 0 →  -- P is on the x-axis
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →  -- |PA| = |PB|
    abs ((B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1)) / 2 = 15/2 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_PAB_l272_27254


namespace NUMINAMATH_CALUDE_problem_solution_l272_27234

theorem problem_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^4)
  (h2 : z^5 = w^2)
  (h3 : z - x = 31) :
  (w : ℤ) - y = -759439 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l272_27234


namespace NUMINAMATH_CALUDE_difference_of_unit_vectors_with_sum_magnitude_one_l272_27252

/-- Given two unit vectors a and b in a real inner product space such that
    the magnitude of their sum is 1, prove that the magnitude of their
    difference is √3. -/
theorem difference_of_unit_vectors_with_sum_magnitude_one
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ‖a + b‖ = 1) :
  ‖a - b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_unit_vectors_with_sum_magnitude_one_l272_27252


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_sum_counts_l272_27213

/-- A function that returns the number of four-digit natural numbers with a given digit sum -/
def countFourDigitNumbersWithSum (sum : Nat) : Nat :=
  sorry

/-- The theorem stating the correct counts for digit sums 5, 6, and 7 -/
theorem four_digit_numbers_with_sum_counts :
  (countFourDigitNumbersWithSum 5 = 35) ∧
  (countFourDigitNumbersWithSum 6 = 56) ∧
  (countFourDigitNumbersWithSum 7 = 84) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_sum_counts_l272_27213


namespace NUMINAMATH_CALUDE_f_equals_g_l272_27289

/-- Given two functions f and g defined on real numbers,
    where f(x) = x^2 and g(x) = ∛(x^6),
    prove that f and g are equal for all real x. -/
theorem f_equals_g : ∀ x : ℝ, (fun x => x^2) x = (fun x => (x^6)^(1/3)) x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l272_27289


namespace NUMINAMATH_CALUDE_factorial_equation_l272_27215

theorem factorial_equation (x : ℕ) : 6 * 8 * 3 * x = Nat.factorial 10 → x = 75600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l272_27215


namespace NUMINAMATH_CALUDE_tan_4290_degrees_l272_27224

theorem tan_4290_degrees : Real.tan (4290 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_4290_degrees_l272_27224


namespace NUMINAMATH_CALUDE_fraction_simplification_l272_27271

theorem fraction_simplification :
  (252 : ℚ) / 18 * 7 / 189 * 9 / 4 = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l272_27271


namespace NUMINAMATH_CALUDE_average_adjacent_pairs_l272_27267

/-- Represents a row of people --/
structure Row where
  boys : ℕ
  girls : ℕ

/-- Calculates the expected number of boy-girl or girl-boy pairs in a row --/
def expectedPairs (r : Row) : ℚ :=
  let total := r.boys + r.girls
  let prob := (r.boys : ℚ) * r.girls / (total * (total - 1))
  2 * prob * (total - 1)

/-- The problem statement --/
theorem average_adjacent_pairs (row1 row2 : Row)
  (h1 : row1 = ⟨10, 12⟩)
  (h2 : row2 = ⟨15, 5⟩) :
  expectedPairs row1 + expectedPairs row2 = 2775 / 154 := by
  sorry

#eval expectedPairs ⟨10, 12⟩ + expectedPairs ⟨15, 5⟩

end NUMINAMATH_CALUDE_average_adjacent_pairs_l272_27267


namespace NUMINAMATH_CALUDE_equal_playing_time_l272_27273

theorem equal_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) % total_players = 0 →
  (players_on_field * match_duration) / total_players = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_playing_time_l272_27273


namespace NUMINAMATH_CALUDE_point_A_l272_27258

def point_A : ℝ × ℝ := (-2, 4)

def move_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

def move_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 - units, p.2)

def point_A' : ℝ × ℝ :=
  move_left (move_up point_A 2) 3

theorem point_A'_coordinates :
  point_A' = (-5, 6) := by
  sorry

end NUMINAMATH_CALUDE_point_A_l272_27258


namespace NUMINAMATH_CALUDE_count_special_numbers_l272_27290

/-- A function that returns the set of all divisors of a natural number -/
def divisors (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that counts the number of divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ :=
  sorry

/-- A function that counts the number of divisors of a natural number that are less than or equal to 10 -/
def divisors_leq_10_count (n : ℕ) : ℕ :=
  sorry

/-- The set of natural numbers from 1 to 100 with exactly four divisors, 
    at least three of which do not exceed 10 -/
def special_numbers : Finset ℕ :=
  sorry

theorem count_special_numbers : special_numbers.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l272_27290


namespace NUMINAMATH_CALUDE_track_length_l272_27226

/-- The length of a circular track given specific meeting conditions of two runners -/
theorem track_length : ∀ (x : ℝ), 
  (∃ (v_brenda v_sally : ℝ), v_brenda > 0 ∧ v_sally > 0 ∧
    -- First meeting condition
    80 / v_brenda = (x/2 - 80) / v_sally ∧
    -- Second meeting condition
    (x/2 - 100) / v_brenda = (x/2 + 100) / v_sally) →
  x = 520 :=
by sorry

end NUMINAMATH_CALUDE_track_length_l272_27226


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_eight_l272_27274

theorem three_digit_divisible_by_eight :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ (n / 100) % 10 = 5 ∧ n % 8 = 0 ∧ n = 533 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_eight_l272_27274


namespace NUMINAMATH_CALUDE_numbering_system_base_l272_27285

theorem numbering_system_base : ∃! (n : ℕ), n > 0 ∧ n^2 = 5*n + 6 := by sorry

end NUMINAMATH_CALUDE_numbering_system_base_l272_27285


namespace NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l272_27269

theorem sum_of_squares_of_solutions : ∃ (s₁ s₂ : ℝ), 
  (s₁^2 - 17*s₁ + 22 = 0) ∧ 
  (s₂^2 - 17*s₂ + 22 = 0) ∧ 
  (s₁^2 + s₂^2 = 245) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l272_27269


namespace NUMINAMATH_CALUDE_skew_lines_and_tetrahedron_l272_27229

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the relation for a point lying on a line
variable (lies_on : Point → Line → Prop)

-- Define the property of lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of lines being perpendicular
variable (perpendicular : Line → Line → Prop)

-- Define the property of points forming a regular tetrahedron
variable (form_regular_tetrahedron : Point → Point → Point → Point → Prop)

-- State the theorem
theorem skew_lines_and_tetrahedron 
  (A B C D : Point) (a b : Line) :
  lies_on A a → lies_on B a → lies_on C b → lies_on D b →
  skew a b →
  ¬perpendicular a b →
  (∃ (AC BD : Line), lies_on A AC ∧ lies_on C AC ∧ lies_on B BD ∧ lies_on D BD ∧ skew AC BD) ∧
  ¬form_regular_tetrahedron A B C D :=
sorry

end NUMINAMATH_CALUDE_skew_lines_and_tetrahedron_l272_27229


namespace NUMINAMATH_CALUDE_initial_apps_equal_final_apps_l272_27296

/-- Proves that the initial number of apps is equal to the final number of apps -/
theorem initial_apps_equal_final_apps 
  (initial_files : ℕ) 
  (final_files : ℕ) 
  (deleted_files : ℕ) 
  (final_apps : ℕ) 
  (h1 : initial_files = 21)
  (h2 : final_files = 7)
  (h3 : deleted_files = 14)
  (h4 : final_apps = 3)
  (h5 : initial_files = final_files + deleted_files) :
  initial_files - final_files = deleted_files ∧ final_apps = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_apps_equal_final_apps_l272_27296


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l272_27279

theorem lcm_gcd_problem :
  let a₁ := 5^2 * 7^4
  let b₁ := 490 * 175
  let a₂ := 2^5 * 3 * 7
  let b₂ := 3^4 * 5^4 * 7^2
  let c₂ := 10000
  (Nat.gcd a₁ b₁ = 8575 ∧ Nat.lcm a₁ b₁ = 600250) ∧
  (Nat.gcd a₂ (Nat.gcd b₂ c₂) = 1 ∧ Nat.lcm a₂ (Nat.lcm b₂ c₂) = 793881600) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l272_27279


namespace NUMINAMATH_CALUDE_twice_x_minus_y_negative_l272_27230

theorem twice_x_minus_y_negative (x y : ℝ) : 
  (2 * x - y < 0) ↔ (∃ z : ℝ, z < 0 ∧ 2 * x - y = z) :=
sorry

end NUMINAMATH_CALUDE_twice_x_minus_y_negative_l272_27230


namespace NUMINAMATH_CALUDE_equation_one_solution_l272_27211

theorem equation_one_solution (x : ℝ) : 
  (x - 1)^2 - 4 = 0 ↔ x = -1 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_l272_27211


namespace NUMINAMATH_CALUDE_tank_filling_time_l272_27282

/-- Given two pipes that can fill a tank in 18 and 20 minutes respectively,
    and an outlet pipe that can empty the tank in 45 minutes,
    prove that when all pipes are opened simultaneously on an empty tank,
    it will take 12 minutes to fill the tank. -/
theorem tank_filling_time
  (pipe1 : ℝ → ℝ)
  (pipe2 : ℝ → ℝ)
  (outlet : ℝ → ℝ)
  (h1 : ∀ t, pipe1 t = t / 18)
  (h2 : ∀ t, pipe2 t = t / 20)
  (h3 : ∀ t, outlet t = t / 45)
  : ∃ t, t > 0 ∧ pipe1 t + pipe2 t - outlet t = 1 ∧ t = 12 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l272_27282


namespace NUMINAMATH_CALUDE_sequence_formula_l272_27201

theorem sequence_formula (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 2 → a n - a (n-1) = (1/3)^(n-1)) →
  ∀ n : ℕ, n ≥ 1 → a n = (3/2) * (1 - (1/3)^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l272_27201


namespace NUMINAMATH_CALUDE_range_of_x_l272_27232

theorem range_of_x (x : ℝ) : 
  0 ≤ x → x < 2 * Real.pi → Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x →
  π / 4 ≤ x ∧ x ≤ 5 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l272_27232


namespace NUMINAMATH_CALUDE_black_duck_count_l272_27270

/-- Represents the number of fish per duck of each color --/
structure FishPerDuck where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- Represents the number of ducks of each color --/
structure DuckCounts where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- The theorem stating the number of black ducks --/
theorem black_duck_count 
  (fish_per_duck : FishPerDuck)
  (duck_counts : DuckCounts)
  (total_fish : ℕ)
  (h1 : fish_per_duck.white = 5)
  (h2 : fish_per_duck.black = 10)
  (h3 : fish_per_duck.multicolor = 12)
  (h4 : duck_counts.white = 3)
  (h5 : duck_counts.multicolor = 6)
  (h6 : total_fish = 157)
  (h7 : total_fish = 
    fish_per_duck.white * duck_counts.white + 
    fish_per_duck.black * duck_counts.black + 
    fish_per_duck.multicolor * duck_counts.multicolor) :
  duck_counts.black = 7 := by
  sorry

end NUMINAMATH_CALUDE_black_duck_count_l272_27270


namespace NUMINAMATH_CALUDE_no_perfect_square_19xx99_l272_27212

theorem no_perfect_square_19xx99 : ¬ ∃ (n : ℕ), 
  (n * n ≥ 1900000) ∧ 
  (n * n < 2000000) ∧ 
  (n * n % 100 = 99) := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_19xx99_l272_27212


namespace NUMINAMATH_CALUDE_min_magnitude_vector_sum_l272_27241

/-- The minimum magnitude of the vector sum of two specific unit vectors -/
theorem min_magnitude_vector_sum :
  let a : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (25 * π / 180))
  let b : ℝ × ℝ := (Real.sin (20 * π / 180), Real.cos (20 * π / 180))
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 / 2 ∧
    ∀ (t : ℝ), Real.sqrt ((a.1 + t * b.1)^2 + (a.2 + t * b.2)^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_magnitude_vector_sum_l272_27241


namespace NUMINAMATH_CALUDE_max_remainder_two_digit_div_sum_digits_l272_27253

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the maximum remainder when dividing a two-digit number
    by the sum of its digits is 15 -/
theorem max_remainder_two_digit_div_sum_digits :
  ∃ (n : ℕ), TwoDigitNumber n ∧
    ∀ (m : ℕ), TwoDigitNumber m →
      n % (sumOfDigits n) ≥ m % (sumOfDigits m) ∧
      n % (sumOfDigits n) = 15 :=
sorry

end NUMINAMATH_CALUDE_max_remainder_two_digit_div_sum_digits_l272_27253


namespace NUMINAMATH_CALUDE_sum_formula_and_difference_l272_27272

def f (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (3 * n - 2)

theorem sum_formula_and_difference (n k : ℕ) (h : n > 0) (h' : k > 0) : 
  f n = (2 * n - 1)^2 ∧ f (k + 1) - f k = 8 * k := by sorry

end NUMINAMATH_CALUDE_sum_formula_and_difference_l272_27272


namespace NUMINAMATH_CALUDE_zachary_pushups_count_l272_27248

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + 22

/-- The number of push-ups John did -/
def john_pushups : ℕ := 69

theorem zachary_pushups_count : zachary_pushups = 51 := by
  have h1 : david_pushups = zachary_pushups + 22 := rfl
  have h2 : john_pushups = david_pushups - 4 := by sorry
  have h3 : john_pushups = 69 := rfl
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_count_l272_27248


namespace NUMINAMATH_CALUDE_characterize_superinvariant_sets_l272_27227

/-- A set S is superinvariant if for any stretching A, there exists a translation B
    such that the images of S under A and B agree -/
def IsSuperinvariant (S : Set ℝ) : Prop :=
  ∀ (x₀ a : ℝ) (ha : a > 0),
    ∃ (b : ℝ),
      (∀ x ∈ S, ∃ y ∈ S, x₀ + a * (x - x₀) = y + b) ∧
      (∀ t ∈ S, ∃ u ∈ S, t + b = x₀ + a * (u - x₀))

/-- The set of all superinvariant subsets of ℝ -/
def SuperinvariantSets : Set (Set ℝ) :=
  {S | IsSuperinvariant S}

theorem characterize_superinvariant_sets :
  SuperinvariantSets =
    {∅} ∪ {Set.univ} ∪ {{p} | p : ℝ} ∪ {Set.univ \ {p} | p : ℝ} ∪
    {Set.Ioi p | p : ℝ} ∪ {Set.Ici p | p : ℝ} ∪
    {Set.Iio p | p : ℝ} ∪ {Set.Iic p | p : ℝ} :=
  sorry

#check characterize_superinvariant_sets

end NUMINAMATH_CALUDE_characterize_superinvariant_sets_l272_27227


namespace NUMINAMATH_CALUDE_inequality_proof_l272_27244

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hab : a + b ≥ 1) 
  (hbc : b + c ≥ 1) 
  (hca : c + a ≥ 1) : 
  1 ≤ (1 - a)^2 + (1 - b)^2 + (1 - c)^2 + (2 * Real.sqrt 2 * a * b * c) / Real.sqrt (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l272_27244


namespace NUMINAMATH_CALUDE_probability_is_one_third_l272_27295

/-- A standard die with six faces -/
def StandardDie : Type := Fin 6

/-- The total number of dots on all faces of a standard die -/
def totalDots : ℕ := 21

/-- The number of favorable outcomes (faces with 1 or 2 dots) -/
def favorableOutcomes : ℕ := 2

/-- The total number of possible outcomes (total faces) -/
def totalOutcomes : ℕ := 6

/-- The probability of the sum of dots on five faces being at least 19 -/
def probability : ℚ := favorableOutcomes / totalOutcomes

theorem probability_is_one_third :
  probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l272_27295


namespace NUMINAMATH_CALUDE_limestone_cost_proof_l272_27223

/-- The cost of limestone per pound -/
def limestone_cost : ℝ := 3

/-- The total weight of the compound in pounds -/
def total_weight : ℝ := 100

/-- The total cost of the compound in dollars -/
def total_cost : ℝ := 425

/-- The weight of limestone used in the compound in pounds -/
def limestone_weight : ℝ := 37.5

/-- The weight of shale mix used in the compound in pounds -/
def shale_weight : ℝ := 62.5

/-- The cost of shale mix per pound in dollars -/
def shale_cost_per_pound : ℝ := 5

/-- The total cost of shale mix in the compound in dollars -/
def total_shale_cost : ℝ := 312.5

theorem limestone_cost_proof :
  limestone_cost * limestone_weight + total_shale_cost = total_cost ∧
  limestone_weight + shale_weight = total_weight ∧
  shale_cost_per_pound * shale_weight = total_shale_cost :=
by sorry

end NUMINAMATH_CALUDE_limestone_cost_proof_l272_27223


namespace NUMINAMATH_CALUDE_distance_between_P_and_Q_l272_27205

theorem distance_between_P_and_Q : ∀ (pq : ℝ),
  (∃ (x : ℝ),
    -- A walks 30 km each day
    30 * x = pq ∧
    -- B starts after A has walked 72 km
    72 + 30 * (pq / 80) = x ∧
    -- B walks 1/10 of the total distance each day
    (pq / 10) * (pq / 80) = pq - x ∧
    -- B meets A after walking for 1/8 of the daily km covered
    (pq / 10) * (1 / 8) = pq / 80) →
  pq = 320 ∨ pq = 180 := by
sorry

end NUMINAMATH_CALUDE_distance_between_P_and_Q_l272_27205


namespace NUMINAMATH_CALUDE_max_value_theorem_l272_27288

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 5 * a + 2 * b < 100) :
  a * b * (100 - 5 * a - 2 * b) ≤ 78125 / 36 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 5 * a₀ + 2 * b₀ < 100 ∧
    a₀ * b₀ * (100 - 5 * a₀ - 2 * b₀) = 78125 / 36 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l272_27288


namespace NUMINAMATH_CALUDE_not_odd_function_iff_exists_neq_l272_27257

theorem not_odd_function_iff_exists_neq (f : ℝ → ℝ) :
  (¬ ∀ x, f (-x) = -f x) ↔ ∃ x₀, f (-x₀) ≠ -f x₀ :=
sorry

end NUMINAMATH_CALUDE_not_odd_function_iff_exists_neq_l272_27257


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l272_27216

theorem added_number_after_doubling (initial_number : ℕ) (x : ℕ) : 
  initial_number = 8 → 
  3 * (2 * initial_number + x) = 75 → 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l272_27216


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l272_27202

theorem min_sum_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l272_27202


namespace NUMINAMATH_CALUDE_smallest_multiple_l272_27210

theorem smallest_multiple (x : ℕ) : x = 32 ↔ 
  (x > 0 ∧ 
   1152 ∣ (900 * x) ∧ 
   ∀ y : ℕ, (y > 0 ∧ y < x) → ¬(1152 ∣ (900 * y))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l272_27210


namespace NUMINAMATH_CALUDE_chess_games_count_l272_27261

/-- The number of combinations of k items from a set of n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of players in the chess group -/
def num_players : ℕ := 50

/-- The number of players in each game -/
def players_per_game : ℕ := 2

theorem chess_games_count : combinations num_players players_per_game = 1225 := by
  sorry

end NUMINAMATH_CALUDE_chess_games_count_l272_27261


namespace NUMINAMATH_CALUDE_total_present_age_l272_27255

-- Define the present ages of p and q
def p : ℕ := sorry
def q : ℕ := sorry

-- Define the conditions
axiom age_relation : p - 12 = (q - 12) / 2
axiom present_ratio : p * 4 = q * 3

-- Theorem to prove
theorem total_present_age : p + q = 42 := by sorry

end NUMINAMATH_CALUDE_total_present_age_l272_27255


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l272_27228

theorem proportional_function_quadrants (k : ℝ) :
  let f : ℝ → ℝ := λ x => (-k^2 - 2) * x
  (∀ x y, f x = y → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l272_27228


namespace NUMINAMATH_CALUDE_stating_judge_assignment_count_l272_27219

/-- Represents the number of judges from each grade -/
def judges_per_grade : ℕ := 2

/-- Represents the number of grades -/
def num_grades : ℕ := 3

/-- Represents the number of courts -/
def num_courts : ℕ := 3

/-- Represents the number of judges per court -/
def judges_per_court : ℕ := 2

/-- 
Theorem stating that the number of ways to assign judges to courts 
under the given conditions is 48
-/
theorem judge_assignment_count : 
  (judges_per_grade ^ num_courts) * (Nat.factorial num_courts) = 48 := by
  sorry


end NUMINAMATH_CALUDE_stating_judge_assignment_count_l272_27219


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l272_27207

theorem sqrt_product_equals_sqrt_of_product : 
  Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l272_27207


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l272_27218

/-- Given a circle inscribed in a square, if the circle's area is 314 square inches,
    then the square's area is 400 square inches. -/
theorem inscribed_circle_square_area :
  ∀ (circle_radius square_side : ℝ),
  circle_radius > 0 →
  square_side > 0 →
  circle_radius * 2 = square_side →
  π * circle_radius^2 = 314 →
  square_side^2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l272_27218


namespace NUMINAMATH_CALUDE_investment_problem_l272_27283

theorem investment_problem (x : ℝ) : 
  (0.07 * x + 0.19 * 1500 = 0.16 * (x + 1500)) → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l272_27283


namespace NUMINAMATH_CALUDE_four_children_probability_l272_27217

theorem four_children_probability (p_boy p_girl : ℚ) : 
  p_boy = 2/3 → 
  p_girl = 1/3 → 
  (1 - (p_boy^4 + p_girl^4)) = 64/81 := by
sorry

end NUMINAMATH_CALUDE_four_children_probability_l272_27217


namespace NUMINAMATH_CALUDE_anna_meal_cost_difference_l272_27294

theorem anna_meal_cost_difference : 
  let bagel_cost : ℚ := 95/100
  let orange_juice_cost : ℚ := 85/100
  let sandwich_cost : ℚ := 465/100
  let milk_cost : ℚ := 115/100
  let breakfast_cost := bagel_cost + orange_juice_cost
  let lunch_cost := sandwich_cost + milk_cost
  lunch_cost - breakfast_cost = 4
  := by sorry

end NUMINAMATH_CALUDE_anna_meal_cost_difference_l272_27294


namespace NUMINAMATH_CALUDE_book_selection_combinations_l272_27235

theorem book_selection_combinations :
  let mystery_count : ℕ := 5
  let fantasy_count : ℕ := 4
  let biography_count : ℕ := 6
  mystery_count * fantasy_count * biography_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l272_27235


namespace NUMINAMATH_CALUDE_smallest_cookie_containers_six_satisfies_condition_smallest_n_is_six_l272_27292

theorem smallest_cookie_containers (n : ℕ) : (∃ k : ℕ, 15 * n - 2 = 11 * k) → n ≥ 6 := by
  sorry

theorem six_satisfies_condition : ∃ k : ℕ, 15 * 6 - 2 = 11 * k := by
  sorry

theorem smallest_n_is_six : (∃ n : ℕ, (∃ k : ℕ, 15 * n - 2 = 11 * k) ∧ (∀ m : ℕ, m < n → ¬(∃ k : ℕ, 15 * m - 2 = 11 * k))) ∧ (∃ k : ℕ, 15 * 6 - 2 = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_containers_six_satisfies_condition_smallest_n_is_six_l272_27292


namespace NUMINAMATH_CALUDE_proportional_relationship_l272_27220

-- Define the proportionality constant
def k : ℝ := 2

-- Define the functional relationship
def f (x : ℝ) : ℝ := k * x + 3

-- State the theorem
theorem proportional_relationship (x y : ℝ) :
  (∀ x, y - 3 = k * x) →  -- (y-3) is directly proportional to x
  (f 2 = 7) →             -- when x=2, y=7
  (∀ x, f x = 2 * x + 3) ∧ -- functional relationship
  (f 4 = 11) ∧            -- when x=4, y=11
  (f⁻¹ 4 = 1/2)           -- when y=4, x=1/2
  := by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l272_27220


namespace NUMINAMATH_CALUDE_complex_division_l272_27281

theorem complex_division (i : ℂ) : i^2 = -1 → (2 + 4*i) / i = 4 - 2*i := by sorry

end NUMINAMATH_CALUDE_complex_division_l272_27281


namespace NUMINAMATH_CALUDE_regular_heptagon_diagonal_relation_l272_27297

/-- Regular heptagon with side length a, diagonal spanning two sides c, and diagonal spanning three sides d -/
structure RegularHeptagon where
  a : ℝ  -- side length
  c : ℝ  -- length of diagonal spanning two sides
  d : ℝ  -- length of diagonal spanning three sides

/-- Theorem: In a regular heptagon, d^2 = c^2 + a^2 -/
theorem regular_heptagon_diagonal_relation (h : RegularHeptagon) : h.d^2 = h.c^2 + h.a^2 := by
  sorry

end NUMINAMATH_CALUDE_regular_heptagon_diagonal_relation_l272_27297


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l272_27291

/-- The distance between two cities given a map distance and scale -/
def real_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem: The distance between Stockholm and Uppsala is 1200 km -/
theorem stockholm_uppsala_distance :
  let map_distance : ℝ := 120
  let scale : ℝ := 10
  real_distance map_distance scale = 1200 := by
sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l272_27291


namespace NUMINAMATH_CALUDE_coefficient_of_minus_five_ab_l272_27242

/-- The coefficient of a monomial is the numerical factor multiplying the variables. -/
def coefficient (m : ℤ) (x : String) : ℤ :=
  m

/-- A monomial is represented as an integer multiplied by a string of variables. -/
def Monomial := ℤ × String

theorem coefficient_of_minus_five_ab :
  let m : Monomial := (-5, "ab")
  coefficient m.1 m.2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_minus_five_ab_l272_27242


namespace NUMINAMATH_CALUDE_smallest_common_shelving_count_l272_27243

theorem smallest_common_shelving_count : Nat.lcm 6 17 = 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_shelving_count_l272_27243


namespace NUMINAMATH_CALUDE_students_with_all_pets_l272_27266

theorem students_with_all_pets (total_students : ℕ) 
  (dog_fraction : ℚ) (cat_fraction : ℚ)
  (other_pet_count : ℕ) (no_pet_count : ℕ)
  (only_dog_count : ℕ) (only_other_count : ℕ)
  (cat_and_other_count : ℕ) :
  total_students = 40 →
  dog_fraction = 5 / 8 →
  cat_fraction = 1 / 4 →
  other_pet_count = 8 →
  no_pet_count = 6 →
  only_dog_count = 12 →
  only_other_count = 3 →
  cat_and_other_count = 10 →
  (∃ (all_pets_count : ℕ),
    all_pets_count = 0 ∧
    total_students * dog_fraction = only_dog_count + all_pets_count + cat_and_other_count ∧
    total_students * cat_fraction = cat_and_other_count + all_pets_count ∧
    other_pet_count = only_other_count + all_pets_count + cat_and_other_count ∧
    total_students - no_pet_count = only_dog_count + only_other_count + all_pets_count + cat_and_other_count) :=
by
  sorry

end NUMINAMATH_CALUDE_students_with_all_pets_l272_27266


namespace NUMINAMATH_CALUDE_running_gender_related_l272_27275

structure RunningData where
  total_students : Nat
  male_students : Nat
  female_like_running : Nat
  male_dislike_running : Nat

def chi_square (data : RunningData) : Rat :=
  let female_students := data.total_students - data.male_students
  let male_like_running := data.male_students - data.male_dislike_running
  let female_dislike_running := female_students - data.female_like_running
  let n := data.total_students
  let a := male_like_running
  let b := data.male_dislike_running
  let c := data.female_like_running
  let d := female_dislike_running
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def is_gender_related (data : RunningData) : Prop :=
  chi_square data > 6635 / 1000

theorem running_gender_related (data : RunningData) 
  (h1 : data.total_students = 200)
  (h2 : data.male_students = 120)
  (h3 : data.female_like_running = 30)
  (h4 : data.male_dislike_running = 50) :
  is_gender_related data := by
  sorry

#eval chi_square { total_students := 200, male_students := 120, female_like_running := 30, male_dislike_running := 50 }

end NUMINAMATH_CALUDE_running_gender_related_l272_27275


namespace NUMINAMATH_CALUDE_all_solutions_are_valid_l272_27299

/-- A quadruple of real numbers satisfying the given conditions -/
structure Quadruple where
  x : ℝ
  y : ℝ
  z : ℝ
  w : ℝ
  sum_zero : x + y + z + w = 0
  sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0

/-- Definition of a valid solution -/
def is_valid_solution (q : Quadruple) : Prop :=
  (q.x = 0 ∧ q.y = 0 ∧ q.z = 0 ∧ q.w = 0) ∨
  (q.x = -q.y ∧ q.z = -q.w) ∨
  (q.x = -q.z ∧ q.y = -q.w) ∨
  (q.x = -q.w ∧ q.y = -q.z)

/-- Main theorem: All solutions are valid -/
theorem all_solutions_are_valid (q : Quadruple) : is_valid_solution q := by
  sorry

end NUMINAMATH_CALUDE_all_solutions_are_valid_l272_27299


namespace NUMINAMATH_CALUDE_simple_interest_principal_l272_27251

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h1 : interest = 2500)
  (h2 : time = 5)
  (h3 : rate = 10)
  : interest = (5000 * rate * time) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l272_27251


namespace NUMINAMATH_CALUDE_y_equation_solution_l272_27221

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 4*y + 4/y + 1/y^2 = 30)
  (h2 : y = c + Real.sqrt d) :
  c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_y_equation_solution_l272_27221


namespace NUMINAMATH_CALUDE_sum_of_squared_roots_l272_27214

theorem sum_of_squared_roots (p q r : ℝ) : 
  (3 * p^3 + 2 * p^2 - 3 * p - 8 = 0) →
  (3 * q^3 + 2 * q^2 - 3 * q - 8 = 0) →
  (3 * r^3 + 2 * r^2 - 3 * r - 8 = 0) →
  p^2 + q^2 + r^2 = -14/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_roots_l272_27214
