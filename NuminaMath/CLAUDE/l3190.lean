import Mathlib

namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3190_319035

theorem repeating_decimal_to_fraction :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (n : ℚ) / d = 7 + (789 : ℚ) / 10000 / (1 - 1 / 10000) :=
by
  -- The fraction 365/85 satisfies this property
  use 365, 85
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3190_319035


namespace NUMINAMATH_CALUDE_min_xy_value_l3190_319048

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z : ℝ, x * y ≥ z → z ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l3190_319048


namespace NUMINAMATH_CALUDE_album_time_calculation_l3190_319089

/-- Calculates the total time to finish all songs in an album -/
def total_album_time (initial_songs : ℕ) (song_duration : ℕ) (added_songs : ℕ) : ℕ :=
  (initial_songs + added_songs) * song_duration

/-- Theorem: Given an initial album of 25 songs, each 3 minutes long, and adding 10 more songs
    of the same duration, the total time to finish all songs in the album is 105 minutes. -/
theorem album_time_calculation :
  total_album_time 25 3 10 = 105 := by
  sorry

end NUMINAMATH_CALUDE_album_time_calculation_l3190_319089


namespace NUMINAMATH_CALUDE_product_of_cosines_l3190_319054

theorem product_of_cosines (π : Real) : 
  (1 + Real.cos (π / 9)) * (1 + Real.cos (2 * π / 9)) * 
  (1 + Real.cos (4 * π / 9)) * (1 + Real.cos (5 * π / 9)) = 
  (1 / 2) * (Real.sin (π / 9))^4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l3190_319054


namespace NUMINAMATH_CALUDE_car_catchup_l3190_319073

/-- The time (in hours) it takes for the second car to catch up with the first car -/
def catchup_time : ℝ :=
  1.5

/-- The speed of the first car in km/h -/
def speed_first : ℝ :=
  60

/-- The speed of the second car in km/h -/
def speed_second : ℝ :=
  80

/-- The head start of the first car in hours -/
def head_start : ℝ :=
  0.5

theorem car_catchup :
  speed_second * catchup_time = speed_first * (catchup_time + head_start) :=
sorry

end NUMINAMATH_CALUDE_car_catchup_l3190_319073


namespace NUMINAMATH_CALUDE_line_parabola_single_intersection_l3190_319092

theorem line_parabola_single_intersection (a : ℝ) :
  (∃! x : ℝ, a * x - 6 = x^2 + 4*x + 3) ↔ (a = -2 ∨ a = 10) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_single_intersection_l3190_319092


namespace NUMINAMATH_CALUDE_bill_score_l3190_319053

theorem bill_score (john sue bill : ℕ) 
  (score_diff : bill = john + 20)
  (bill_half_sue : bill * 2 = sue)
  (total_score : john + bill + sue = 160) :
  bill = 45 := by
sorry

end NUMINAMATH_CALUDE_bill_score_l3190_319053


namespace NUMINAMATH_CALUDE_cone_volume_not_equal_base_height_product_l3190_319051

/-- The volume of a cone is not equal to the product of its base area and height. -/
theorem cone_volume_not_equal_base_height_product (S h : ℝ) (S_pos : S > 0) (h_pos : h > 0) :
  ∃ V : ℝ, V = (1/3) * S * h ∧ V ≠ S * h := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_not_equal_base_height_product_l3190_319051


namespace NUMINAMATH_CALUDE_factorial_division_l3190_319005

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3190_319005


namespace NUMINAMATH_CALUDE_existence_of_number_with_four_prime_factors_l3190_319050

theorem existence_of_number_with_four_prime_factors : ∃ N : ℕ,
  (∃ p₁ p₂ p₃ p₄ : ℕ, 
    (Nat.Prime p₁) ∧ (Nat.Prime p₂) ∧ (Nat.Prime p₃) ∧ (Nat.Prime p₄) ∧
    (p₁ ≠ p₂) ∧ (p₁ ≠ p₃) ∧ (p₁ ≠ p₄) ∧ (p₂ ≠ p₃) ∧ (p₂ ≠ p₄) ∧ (p₃ ≠ p₄) ∧
    (1 < p₁) ∧ (p₁ ≤ 100) ∧
    (1 < p₂) ∧ (p₂ ≤ 100) ∧
    (1 < p₃) ∧ (p₃ ≤ 100) ∧
    (1 < p₄) ∧ (p₄ ≤ 100) ∧
    (N = p₁ * p₂ * p₃ * p₄) ∧
    (∀ q : ℕ, Nat.Prime q → q ∣ N → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄))) ∧
  N = 210 :=
by
  sorry


end NUMINAMATH_CALUDE_existence_of_number_with_four_prime_factors_l3190_319050


namespace NUMINAMATH_CALUDE_race_distance_l3190_319062

/-- 
Given a race with two contestants A and B, where:
- The ratio of speeds of A and B is 3:4
- A has a start of 140 meters
- A wins by 20 meters

Prove that the total distance of the race is 480 meters.
-/
theorem race_distance (speed_A speed_B : ℝ) (total_distance : ℝ) : 
  speed_A / speed_B = 3 / 4 →
  total_distance - (total_distance - 140 + 20) = speed_A / speed_B * total_distance →
  total_distance = 480 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3190_319062


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l3190_319052

/-- Two vectors in R² are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2) ∨ w = (k * v.1, k * v.2)

/-- The problem statement -/
theorem collinear_vectors_x_value :
  ∀ (x : ℝ), collinear (x, 1) (4, x) → x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l3190_319052


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3190_319076

def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  isGeometricSequence a →
  a 1 = 2 →
  (∀ n, a (n + 2)^2 + 4 * a n^2 = 4 * a (n + 1)^2) →
  ∀ n, a n = 2^((n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3190_319076


namespace NUMINAMATH_CALUDE_smallest_p_value_l3190_319077

theorem smallest_p_value (p q : ℕ+) 
  (h1 : (5 : ℚ) / 8 < p / q)
  (h2 : p / q < (7 : ℚ) / 8)
  (h3 : p + q = 2005) : 
  p.val ≥ 772 ∧ (∀ m : ℕ+, m < p → ¬((5 : ℚ) / 8 < m / (2005 - m) ∧ m / (2005 - m) < (7 : ℚ) / 8)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_p_value_l3190_319077


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3190_319013

theorem arctan_equation_solution :
  ∃ x : ℝ, 2 * Real.arctan (1/5) + 2 * Real.arctan (1/10) + Real.arctan (1/x) = π/2 ∧ x = 120/119 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3190_319013


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l3190_319043

/-- Given an isosceles triangle with base 2s and height h, and a rectangle with side length s,
    if their areas are equal, then the height of the triangle equals the side length of the rectangle. -/
theorem isosceles_triangle_rectangle_equal_area
  (s h : ℝ) -- s: side length of rectangle, h: height of triangle
  (h_positive : s > 0) -- Ensure s is positive
  (area_equal : s * h = s^2) -- Areas are equal
  : h = s := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l3190_319043


namespace NUMINAMATH_CALUDE_no_simultaneous_age_ratio_l3190_319083

theorem no_simultaneous_age_ratio : ¬∃ (x : ℝ), x ≥ 0 ∧ 
  (85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_age_ratio_l3190_319083


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3190_319082

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to (a - b), then the x-coordinate of b is 1/2. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b.2 = 4) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) → b.1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3190_319082


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l3190_319018

theorem complex_expression_evaluation : (3 * (3 * (4 * (3 * (4 * (2 + 1) + 1) + 2) + 1) + 2) + 1) = 1492 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l3190_319018


namespace NUMINAMATH_CALUDE_nail_trimming_customers_l3190_319075

/-- The number of nails per person -/
def nails_per_person : ℕ := 20

/-- The total number of sounds produced by the nail cutter -/
def total_sounds : ℕ := 100

/-- The number of customers whose nails were trimmed -/
def num_customers : ℕ := total_sounds / nails_per_person

theorem nail_trimming_customers :
  num_customers = 5 :=
sorry

end NUMINAMATH_CALUDE_nail_trimming_customers_l3190_319075


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l3190_319023

/-- Represents an arrangement of numbers satisfying the given conditions -/
def ValidArrangement (n : ℕ) := List ℕ

/-- Checks if the arrangement is valid for a given n -/
def isValidArrangement (n : ℕ) (arr : ValidArrangement n) : Prop :=
  (arr.length = 2*n + 1) ∧
  (arr.count 0 = 1) ∧
  (∀ m : ℕ, m ≥ 1 → m ≤ n → arr.count m = 2) ∧
  (∀ m : ℕ, m ≥ 1 → m ≤ n → 
    ∃ i j : ℕ, i < j ∧ 
    (arr.get! i = m) ∧ 
    (arr.get! j = m) ∧ 
    (j - i - 1 = m))

/-- Theorem stating that a valid arrangement exists for any natural number n -/
theorem valid_arrangement_exists (n : ℕ) : ∃ arr : ValidArrangement n, isValidArrangement n arr :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l3190_319023


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l3190_319001

theorem gcd_linear_combination (a b d : ℕ) :
  d = Nat.gcd a b →
  d = Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) := by
  sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l3190_319001


namespace NUMINAMATH_CALUDE_quadratic_sum_l3190_319020

/-- Given a quadratic function f(x) = 10x^2 + 100x + 1000, 
    proves that when written in the form a(x+b)^2 + c, 
    the sum of a, b, and c is 765 -/
theorem quadratic_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
    (∀ x, 10 * x^2 + 100 * x + 1000 = a * (x + b)^2 + c) ∧
    a + b + c = 765 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3190_319020


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3190_319084

theorem complex_equation_solution (z : ℂ) : 
  z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) →
  z = (1/2 : ℂ) - Complex.I * ((Real.sqrt 3)/2) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3190_319084


namespace NUMINAMATH_CALUDE_train_length_l3190_319034

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 27 → ∃ length : ℝ, abs (length - 299.97) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3190_319034


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l3190_319003

/-- Calculates the money made from selling chocolate bars --/
def money_made (cost_per_bar : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * cost_per_bar

/-- Proves that Olivia made $9 from selling chocolate bars --/
theorem olivia_chocolate_sales : money_made 3 7 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l3190_319003


namespace NUMINAMATH_CALUDE_cricket_average_l3190_319032

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 10 → 
  next_runs = 81 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase → 
  current_average = 37 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l3190_319032


namespace NUMINAMATH_CALUDE_village_population_l3190_319059

theorem village_population (initial_population : ℝ) : 
  (initial_population * 1.2 * 0.8 = 9600) → initial_population = 10000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3190_319059


namespace NUMINAMATH_CALUDE_quiz_team_payment_l3190_319022

/-- The set of possible values for B in the number 2B5 -/
def possible_B : Set Nat :=
  {b | b ∈ Finset.range 10 ∧ (200 + 10 * b + 5) % 15 = 0}

/-- The theorem stating that the only possible values for B are 2, 5, and 8 -/
theorem quiz_team_payment :
  possible_B = {2, 5, 8} := by sorry

end NUMINAMATH_CALUDE_quiz_team_payment_l3190_319022


namespace NUMINAMATH_CALUDE_inequality_solution_l3190_319074

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 45) / (x + 7) < 0 ↔ (x > -7 ∧ x < -5) ∨ (x > -5 ∧ x < 9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3190_319074


namespace NUMINAMATH_CALUDE_square_region_perimeter_l3190_319029

theorem square_region_perimeter (area : ℝ) (num_squares : ℕ) (perimeter : ℝ) :
  area = 392 →
  num_squares = 8 →
  (area / num_squares).sqrt * (2 * num_squares + 2) = perimeter →
  perimeter = 70 := by
  sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l3190_319029


namespace NUMINAMATH_CALUDE_divisible_by_41_l3190_319009

theorem divisible_by_41 (n : ℕ) : ∃ k : ℤ, 5 * 7^(2*(n+1)) + 2^(3*n) = 41 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_41_l3190_319009


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3190_319087

/-- Represents an ellipse with equation x²/(m-2) + y²/(10-m) = 1 -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (m - 2) + y^2 / (10 - m) = 1

/-- Represents the focal distance of an ellipse -/
def focalDistance (e : Ellipse m) := 4

/-- Represents that the foci of the ellipse are on the x-axis -/
def fociOnXAxis (e : Ellipse m) := True

theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : focalDistance e = 4) (h2 : fociOnXAxis e) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3190_319087


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3190_319044

theorem complex_equation_solution (a : ℝ) : (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3190_319044


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3190_319025

theorem average_of_remaining_numbers 
  (total : ℝ) 
  (group1 : ℝ) 
  (group2 : ℝ) 
  (h1 : total = 6 * 3.95) 
  (h2 : group1 = 2 * 4.2) 
  (h3 : group2 = 2 * 3.85) : 
  (total - group1 - group2) / 2 = 3.8 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3190_319025


namespace NUMINAMATH_CALUDE_textile_firm_profit_decrease_l3190_319066

/-- Represents the decrease in profit due to loom breakdowns -/
def decrease_in_profit (
  total_looms : ℕ)
  (monthly_sales : ℝ)
  (monthly_manufacturing_expenses : ℝ)
  (monthly_establishment_charges : ℝ)
  (breakdown_days : List ℕ)
  (repair_cost_per_loom : ℝ)
  : ℝ :=
  sorry

/-- Theorem stating the decrease in profit for the given scenario -/
theorem textile_firm_profit_decrease :
  decrease_in_profit 70 1000000 150000 75000 [10, 5, 15] 2000 = 20285.70 :=
sorry

end NUMINAMATH_CALUDE_textile_firm_profit_decrease_l3190_319066


namespace NUMINAMATH_CALUDE_tournament_outcomes_l3190_319015

/-- Represents a knockout tournament with 6 players -/
structure Tournament :=
  (num_players : Nat)
  (num_games : Nat)

/-- The number of possible outcomes for each game -/
def outcomes_per_game : Nat := 2

/-- Theorem stating that the number of possible prize orders is 32 -/
theorem tournament_outcomes (t : Tournament) (h1 : t.num_players = 6) (h2 : t.num_games = 5) : 
  outcomes_per_game ^ t.num_games = 32 := by
  sorry

#eval outcomes_per_game ^ 5

end NUMINAMATH_CALUDE_tournament_outcomes_l3190_319015


namespace NUMINAMATH_CALUDE_red_balls_count_l3190_319000

/-- The number of times 18 balls are taken out after the initial 60 balls -/
def x : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 60 + 18 * x

/-- The total number of red balls in the bag -/
def red_balls : ℕ := 56 + 14 * x

/-- The proportion of red balls to total balls is 4/5 -/
axiom proportion_axiom : (red_balls : ℚ) / total_balls = 4 / 5

theorem red_balls_count : red_balls = 336 := by sorry

end NUMINAMATH_CALUDE_red_balls_count_l3190_319000


namespace NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_l3190_319008

variable (p q : Prop)

theorem either_false_sufficient_not_necessary :
  (((¬p ∨ ¬q) → ¬p) ∧ ¬(¬p → (¬p ∨ ¬q))) := by sorry

end NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_l3190_319008


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l3190_319056

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ (N.prod Set.univ) = ∅ := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l3190_319056


namespace NUMINAMATH_CALUDE_find_M_l3190_319068

theorem find_M : ∃ M : ℕ, (992 + 994 + 996 + 998 + 1000 = 5000 - M) ∧ (M = 20) := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3190_319068


namespace NUMINAMATH_CALUDE_base4_division_theorem_l3190_319041

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent. -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

/-- Represents a number in base 4. -/
structure Base4 where
  digits : List Nat
  valid : ∀ d ∈ digits.toFinset, d < 4

/-- The dividend in base 4. -/
def dividend : Base4 := {
  digits := [0, 2, 3, 2, 1]
  valid := by sorry
}

/-- The divisor in base 4. -/
def divisor : Base4 := {
  digits := [2, 1]
  valid := by sorry
}

/-- The quotient in base 4. -/
def quotient : Base4 := {
  digits := [1, 2, 1, 1]
  valid := by sorry
}

/-- Theorem stating that the division of the dividend by the divisor equals the quotient in base 4. -/
theorem base4_division_theorem :
  (base4ToDecimal dividend.digits) / (base4ToDecimal divisor.digits) = base4ToDecimal quotient.digits := by
  sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l3190_319041


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3190_319047

/-- The x-coordinate of the intersection point of two lines -/
def intersection_x (m₁ b₁ a₂ b₂ c₂ : ℚ) : ℚ :=
  (c₂ + 2 * b₁) / (2 * m₁ + a₂)

theorem intersection_of_lines :
  let line1 : ℚ → ℚ := λ x => 3 * x - 24
  let line2 : ℚ → ℚ → Prop := λ x y => 5 * x + 2 * y = 102
  ∃ x y : ℚ, line2 x y ∧ y = line1 x ∧ x = 150 / 11 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3190_319047


namespace NUMINAMATH_CALUDE_ratio_equality_l3190_319070

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3190_319070


namespace NUMINAMATH_CALUDE_negation_of_exists_prop_l3190_319081

theorem negation_of_exists_prop :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_prop_l3190_319081


namespace NUMINAMATH_CALUDE_even_function_theorem_l3190_319080

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_theorem (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_positive : ∀ x > 0, f x = (1 - x) * x) : 
  ∀ x < 0, f x = -x^2 - x := by
sorry

end NUMINAMATH_CALUDE_even_function_theorem_l3190_319080


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3190_319017

/-- Calculates the total amount after a given period using simple interest -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, prove that the total amount after 7 years is $595 -/
theorem simple_interest_problem :
  ∃ (rate : ℝ),
    (totalAmount 350 rate 2 = 420) →
    (totalAmount 350 rate 7 = 595) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3190_319017


namespace NUMINAMATH_CALUDE_inequality_proof_l3190_319093

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a * b + b * c + c * a = 1/3) : 
  1 / (a^2 - b*c + 1) + 1 / (b^2 - c*a + 1) + 1 / (c^2 - a*b + 1) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3190_319093


namespace NUMINAMATH_CALUDE_sqrt_40_simplification_l3190_319058

theorem sqrt_40_simplification : Real.sqrt 40 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_40_simplification_l3190_319058


namespace NUMINAMATH_CALUDE_garden_area_is_855_l3190_319094

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  posts : ℕ
  post_distance : ℝ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the garden given the specifications -/
def garden_area (g : Garden) : ℝ :=
  let shorter_side_posts := (g.posts / 2) / (g.longer_side_post_ratio + 1)
  let longer_side_posts := g.longer_side_post_ratio * shorter_side_posts
  let shorter_side_length := (shorter_side_posts - 1) * g.post_distance
  let longer_side_length := (longer_side_posts - 1) * g.post_distance
  shorter_side_length * longer_side_length

/-- Theorem stating that the garden with given specifications has an area of 855 square yards -/
theorem garden_area_is_855 (g : Garden) 
    (h1 : g.posts = 24)
    (h2 : g.post_distance = 6)
    (h3 : g.longer_side_post_ratio = 3) : 
  garden_area g = 855 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_is_855_l3190_319094


namespace NUMINAMATH_CALUDE_vector_b_value_l3190_319006

theorem vector_b_value (a b : ℝ × ℝ × ℝ) :
  a = (4, 0, -2) →
  a - b = (0, 1, -2) →
  b = (4, -1, 0) := by
sorry

end NUMINAMATH_CALUDE_vector_b_value_l3190_319006


namespace NUMINAMATH_CALUDE_average_value_equals_combination_l3190_319031

def average_value (n : ℕ) : ℚ :=
  (n + 1) * n * (n - 1) / 6

theorem average_value_equals_combination (n : ℕ) (h : n > 0) :
  average_value n = Nat.choose (n + 1) 3 := by sorry

end NUMINAMATH_CALUDE_average_value_equals_combination_l3190_319031


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_value_l3190_319028

def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

def v_3 (x : ℝ) : ℝ := ((7*x + 6)*x + 5)*x + 4

theorem qin_jiushao_v3_value : v_3 3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_value_l3190_319028


namespace NUMINAMATH_CALUDE_count_integers_with_at_most_three_divisors_cubic_plus_eight_l3190_319065

def has_at_most_three_divisors (x : ℤ) : Prop :=
  (∃ p : ℕ, Prime p ∧ x = p^2) ∨ (∃ p : ℕ, Prime p ∧ x = p) ∨ x = 1

theorem count_integers_with_at_most_three_divisors_cubic_plus_eight :
  ∃! (S : Finset ℤ), ∀ n : ℤ, n ∈ S ↔ has_at_most_three_divisors (n^3 + 8) ∧ Finset.card S = 2 :=
sorry

end NUMINAMATH_CALUDE_count_integers_with_at_most_three_divisors_cubic_plus_eight_l3190_319065


namespace NUMINAMATH_CALUDE_trig_identity_l3190_319079

theorem trig_identity (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3190_319079


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3190_319057

/-- Given two parallel vectors a and b in R², if a = (4, 2) and b = (x, 3), then x = 6 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (4, 2)) 
  (h2 : b = (x, 3)) 
  (h3 : ∃ (k : ℝ), b = k • a) : 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3190_319057


namespace NUMINAMATH_CALUDE_employee_payment_l3190_319040

theorem employee_payment (total : ℝ) (a_multiplier : ℝ) (b_payment : ℝ) :
  total = 580 →
  a_multiplier = 1.5 →
  total = b_payment + a_multiplier * b_payment →
  b_payment = 232 := by
sorry

end NUMINAMATH_CALUDE_employee_payment_l3190_319040


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3190_319002

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Main theorem -/
theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 5 = 16)
  (h_fourth : a 4 = 8) :
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3190_319002


namespace NUMINAMATH_CALUDE_mike_seeds_count_mike_total_seeds_l3190_319038

theorem mike_seeds_count : ℕ → Prop :=
  fun total_seeds =>
    let seeds_left : ℕ := 20
    let seeds_right : ℕ := 2 * seeds_left
    let seeds_joining : ℕ := 30
    let seeds_remaining : ℕ := 30
    total_seeds = seeds_left + seeds_right + seeds_joining + seeds_remaining

theorem mike_total_seeds :
  ∃ (total_seeds : ℕ), mike_seeds_count total_seeds ∧ total_seeds = 120 := by
  sorry

end NUMINAMATH_CALUDE_mike_seeds_count_mike_total_seeds_l3190_319038


namespace NUMINAMATH_CALUDE_odd_tau_tau_count_l3190_319061

/-- The number of positive integer divisors of n -/
def τ (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The count of integers n between 1 and 50 (inclusive) such that τ(τ(n)) is odd -/
def countOddTauTau : ℕ := sorry

theorem odd_tau_tau_count : countOddTauTau = 17 := by sorry

end NUMINAMATH_CALUDE_odd_tau_tau_count_l3190_319061


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3190_319049

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its possible cuts --/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Generates all possible ways to cut the plywood into congruent pieces --/
def possible_cuts (p : Plywood) : List Rectangle :=
  sorry -- Implementation details omitted

/-- Finds the maximum perimeter from a list of rectangles --/
def max_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

/-- Finds the minimum perimeter from a list of rectangles --/
def min_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

theorem plywood_cut_perimeter_difference :
  let p : Plywood := { length := 9, width := 6, num_pieces := 6 }
  let cuts := possible_cuts p
  max_perimeter cuts - min_perimeter cuts = 10 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3190_319049


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l3190_319045

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 500 -/
def product : ℕ := 45 * 500

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l3190_319045


namespace NUMINAMATH_CALUDE_x_range_for_f_l3190_319036

-- Define the function f
def f (x : ℝ) := x^3 + 3*x

-- State the theorem
theorem x_range_for_f (x : ℝ) :
  (∀ m ∈ Set.Icc (-2 : ℝ) 2, f (m*x - 2) + f x < 0) →
  x ∈ Set.Ioo (-2 : ℝ) (2/3) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_f_l3190_319036


namespace NUMINAMATH_CALUDE_percentage_to_fraction_l3190_319042

theorem percentage_to_fraction (p : ℚ) : p = 166 / 1000 → p = 83 / 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_fraction_l3190_319042


namespace NUMINAMATH_CALUDE_incorrect_reasoning_l3190_319010

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_on_plane : Line → Plane → Prop)

-- Define the theorem
theorem incorrect_reasoning 
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ l α A, ¬(line_on_plane l α) → on_line A l → ¬(on_plane A α)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_reasoning_l3190_319010


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l3190_319024

/-- Given two right circular cylinders with identical volumes, where the radius of the second cylinder
    is 20% more than the radius of the first, prove that the height of the first cylinder
    is 44% more than the height of the second cylinder. -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) : 
  r₁ > 0 → h₁ > 0 → r₂ > 0 → h₂ > 0 →
  (π * r₁^2 * h₁ = π * r₂^2 * h₂) →  -- Volumes are equal
  (r₂ = 1.2 * r₁) →                  -- Second radius is 20% more than the first
  (h₁ = 1.44 * h₂) :=                -- First height is 44% more than the second
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l3190_319024


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3190_319071

theorem average_of_remaining_numbers 
  (total : ℝ) 
  (group1 : ℝ) 
  (group2 : ℝ) 
  (h1 : total = 6 * 3.95) 
  (h2 : group1 = 2 * 3.8) 
  (h3 : group2 = 2 * 3.85) : 
  (total - group1 - group2) / 2 = 4.2 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3190_319071


namespace NUMINAMATH_CALUDE_integral_sqrt_rational_equals_pi_sixth_l3190_319033

theorem integral_sqrt_rational_equals_pi_sixth :
  ∫ x in (2 : ℝ)..3, Real.sqrt ((3 - 2*x) / (2*x - 7)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_rational_equals_pi_sixth_l3190_319033


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3190_319046

theorem least_positive_integer_with_remainders : ∃ M : ℕ, 
  (M > 0) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (∀ n : ℕ, n > 0 ∧ 
    n % 6 = 5 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 ∧
    n % 10 = 9 ∧
    n % 11 = 10 → n ≥ M) ∧
  M = 27719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3190_319046


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3190_319085

open Set Real

-- Define the universal set I as the set of real numbers
def I : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x + 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 2 ≥ 0}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Bᶜ) = {x : ℝ | -1 ≤ x ∧ x < sqrt 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3190_319085


namespace NUMINAMATH_CALUDE_dans_balloons_l3190_319086

theorem dans_balloons (sam_initial : Real) (fred_given : Real) (total : Real) : Real :=
  let sam_remaining := sam_initial - fred_given
  let dan_balloons := total - sam_remaining
  dan_balloons

#check dans_balloons 46.0 10.0 52.0

end NUMINAMATH_CALUDE_dans_balloons_l3190_319086


namespace NUMINAMATH_CALUDE_problem_statement_l3190_319090

theorem problem_statement (p q r : ℝ) 
  (h1 : p * q / (p + r) + q * r / (q + p) + r * p / (r + q) = -7)
  (h2 : p * r / (p + r) + q * p / (q + p) + r * q / (r + q) = 8) :
  q / (p + q) + r / (q + r) + p / (r + p) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3190_319090


namespace NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l3190_319060

theorem largest_divisor_of_polynomial (n : ℤ) : 
  ∃ (k : ℕ), k = 120 ∧ (k : ℤ) ∣ (n^5 - 5*n^3 + 4*n) ∧ 
  ∀ (m : ℕ), m > k → ¬((m : ℤ) ∣ (n^5 - 5*n^3 + 4*n)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l3190_319060


namespace NUMINAMATH_CALUDE_erasers_per_box_l3190_319019

theorem erasers_per_box 
  (num_boxes : ℕ) 
  (price_per_eraser : ℚ) 
  (total_money : ℚ) 
  (h1 : num_boxes = 48)
  (h2 : price_per_eraser = 3/4)
  (h3 : total_money = 864) : 
  (total_money / price_per_eraser) / num_boxes = 24 := by
sorry

end NUMINAMATH_CALUDE_erasers_per_box_l3190_319019


namespace NUMINAMATH_CALUDE_lcm_of_primes_l3190_319027

theorem lcm_of_primes : 
  let p₁ : Nat := 1223
  let p₂ : Nat := 1399
  let p₃ : Nat := 2687
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ →
  Nat.lcm p₁ (Nat.lcm p₂ p₃) = 4583641741 :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_primes_l3190_319027


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3190_319088

theorem parallelogram_side_length 
  (s : ℝ) 
  (area : ℝ) 
  (h1 : area = 27 * Real.sqrt 3) 
  (h2 : area = 3 * s^2 * (1/2)) : 
  s = 3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3190_319088


namespace NUMINAMATH_CALUDE_count_four_digit_integers_eq_sixteen_l3190_319098

/-- The number of four-digit positive integers composed only of digits 2 and 5 -/
def count_four_digit_integers : ℕ :=
  let digit_choices := 2  -- number of choices for each digit (2 or 5)
  let num_digits := 4     -- number of digits in the integer
  digit_choices ^ num_digits

/-- Theorem stating that the count of four-digit positive integers
    composed only of digits 2 and 5 is equal to 16 -/
theorem count_four_digit_integers_eq_sixteen :
  count_four_digit_integers = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_eq_sixteen_l3190_319098


namespace NUMINAMATH_CALUDE_min_value_3a_minus_2ab_l3190_319030

theorem min_value_3a_minus_2ab :
  ∀ a b : ℕ+, a < 8 → b < 8 → (3 * a - 2 * a * b : ℤ) ≥ -77 ∧
  ∃ a₀ b₀ : ℕ+, a₀ < 8 ∧ b₀ < 8 ∧ (3 * a₀ - 2 * a₀ * b₀ : ℤ) = -77 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3a_minus_2ab_l3190_319030


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l3190_319096

theorem factorial_equation_solution : 
  ∃ (n : ℕ), n > 0 ∧ (Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 1320) ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l3190_319096


namespace NUMINAMATH_CALUDE_f_is_odd_iff_l3190_319037

-- Define the function f
def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

-- State the theorem
theorem f_is_odd_iff (a b : ℝ) :
  (∀ x, f a b (-x) = -f a b x) ↔ a^2 + b^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_is_odd_iff_l3190_319037


namespace NUMINAMATH_CALUDE_circle_rolling_in_triangle_l3190_319007

/-- The distance traveled by the center of a circle rolling inside a right triangle -/
theorem circle_rolling_in_triangle (a b c : ℝ) (r : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) (h_radius : r = 2) : 
  (a - 2*r) + (b - 2*r) + (c - 2*r) = 24 := by
sorry


end NUMINAMATH_CALUDE_circle_rolling_in_triangle_l3190_319007


namespace NUMINAMATH_CALUDE_norbs_age_l3190_319039

def guesses : List Nat := [26, 30, 35, 39, 42, 43, 45, 47, 49, 52, 53]

def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

def countLowerGuesses (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (· < age)).length

def countOffByOne (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (fun g => g = age - 1 || g = age + 1)).length

theorem norbs_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    isPrime age ∧
    countLowerGuesses age guesses ≥ guesses.length / 2 ∧
    countOffByOne age guesses = 3 ∧
    age = 47 :=
  by sorry

end NUMINAMATH_CALUDE_norbs_age_l3190_319039


namespace NUMINAMATH_CALUDE_systematic_sampling_l3190_319097

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (groups : ℕ) 
  (interval : ℕ) 
  (group_15_num : ℕ) 
  (h1 : total_students = 160) 
  (h2 : sample_size = 20) 
  (h3 : groups = 20) 
  (h4 : interval = 8) 
  (h5 : group_15_num = 116) :
  ∃ (first_group_num : ℕ), 
    first_group_num + interval * (15 - 1) = group_15_num ∧ 
    first_group_num = 4 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3190_319097


namespace NUMINAMATH_CALUDE_cos_seven_arccos_two_fifths_l3190_319055

theorem cos_seven_arccos_two_fifths (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, abs (Real.cos (7 * Real.arccos (2/5)) - x) < ε ∧ abs (x + 0.2586) < ε :=
sorry

end NUMINAMATH_CALUDE_cos_seven_arccos_two_fifths_l3190_319055


namespace NUMINAMATH_CALUDE_merry_and_brother_lambs_l3190_319091

/-- The number of lambs Merry and her brother have -/
theorem merry_and_brother_lambs :
  let merry_lambs : ℕ := 10
  let brother_lambs : ℕ := merry_lambs + 3
  merry_lambs + brother_lambs = 23 :=
by sorry

end NUMINAMATH_CALUDE_merry_and_brother_lambs_l3190_319091


namespace NUMINAMATH_CALUDE_unique_intersection_point_l3190_319063

theorem unique_intersection_point (m : ℤ) : 
  (∃ (x : ℕ+), -3 * x + 2 = m * (x^2 - x + 1)) ↔ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l3190_319063


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_m_value_for_intersection_l3190_319011

-- Define set A
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}

-- Define set B with parameter m
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem complement_B_intersect_A :
  (Set.compl (B 3) ∩ A) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem m_value_for_intersection :
  ∃ m : ℝ, (A ∩ B m) = {x | -1 < x ∧ x < 4} → m = 8 := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_m_value_for_intersection_l3190_319011


namespace NUMINAMATH_CALUDE_factorization_equality_l3190_319072

theorem factorization_equality (x : ℝ) : (x + 2) * x - (x + 2) = (x + 2) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3190_319072


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_4_A_subset_B_condition_l3190_319014

-- Define the sets A and B
def A : Set ℝ := {x | (1 - x) / (x - 7) > 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}

-- Theorem 1: Intersection of A and B when a = 4
theorem intersection_A_B_when_a_4 : A ∩ B 4 = {x | 1 < x ∧ x < 6} := by sorry

-- Theorem 2: Condition for A to be a subset of B
theorem A_subset_B_condition (a : ℝ) : A ⊆ B a ↔ a ≤ -7 ∨ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_4_A_subset_B_condition_l3190_319014


namespace NUMINAMATH_CALUDE_line_through_two_points_l3190_319095

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem stating that the equation represents a line through two points
theorem line_through_two_points (M N : Point2D) (h : M ≠ N) :
  ∃! l : Line2D, pointOnLine M l ∧ pointOnLine N l ∧
  ∀ P : Point2D, pointOnLine P l ↔ (P.x - M.x) / (N.x - M.x) = (P.y - M.y) / (N.y - M.y) :=
sorry

end NUMINAMATH_CALUDE_line_through_two_points_l3190_319095


namespace NUMINAMATH_CALUDE_donald_oranges_l3190_319078

theorem donald_oranges (initial : ℕ) (total : ℕ) (found : ℕ) : 
  initial = 4 → total = 9 → found = total - initial → found = 5 := by sorry

end NUMINAMATH_CALUDE_donald_oranges_l3190_319078


namespace NUMINAMATH_CALUDE_grace_and_henry_weight_l3190_319012

/-- Given the weights of pairs of people, prove that Grace and Henry weigh 250 pounds together. -/
theorem grace_and_henry_weight
  (e f g h : ℝ)  -- Weights of Ella, Finn, Grace, and Henry
  (h1 : e + f = 280)  -- Ella and Finn weigh 280 pounds together
  (h2 : f + g = 230)  -- Finn and Grace weigh 230 pounds together
  (h3 : e + h = 300)  -- Ella and Henry weigh 300 pounds together
  : g + h = 250 := by
  sorry

end NUMINAMATH_CALUDE_grace_and_henry_weight_l3190_319012


namespace NUMINAMATH_CALUDE_sin_beta_value_l3190_319069

-- Define acute angles
def is_acute (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem sin_beta_value (α β : Real) 
  (h_acute_α : is_acute α) (h_acute_β : is_acute β)
  (h_sin_α : Real.sin α = (4/7) * Real.sqrt 3)
  (h_cos_sum : Real.cos (α + β) = -11/14) :
  Real.sin β = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l3190_319069


namespace NUMINAMATH_CALUDE_arctan_identity_l3190_319026

theorem arctan_identity (x : Real) : 
  Real.arctan (Real.tan (70 * π / 180) - 2 * Real.tan (35 * π / 180)) = 20 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_identity_l3190_319026


namespace NUMINAMATH_CALUDE_reflect_point_over_x_axis_l3190_319004

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflect_point_over_x_axis :
  let P : Point := { x := -6, y := -9 }
  reflectOverXAxis P = { x := -6, y := 9 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_over_x_axis_l3190_319004


namespace NUMINAMATH_CALUDE_cos_of_tan_in_third_quadrant_l3190_319099

/-- Prove that for an angle α in the third quadrant with tan α = 4/3, cos α = -3/5 -/
theorem cos_of_tan_in_third_quadrant (α : Real) 
  (h1 : π < α ∧ α < 3*π/2)  -- α is in the third quadrant
  (h2 : Real.tan α = 4/3) : 
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_of_tan_in_third_quadrant_l3190_319099


namespace NUMINAMATH_CALUDE_investor_purchase_price_l3190_319016

/-- The dividend rate paid by the company -/
def dividend_rate : ℚ := 185 / 1000

/-- The face value of each share -/
def face_value : ℚ := 50

/-- The return on investment received by the investor -/
def roi : ℚ := 1 / 4

/-- The purchase price per share -/
def purchase_price : ℚ := 37

theorem investor_purchase_price : 
  dividend_rate * face_value / purchase_price = roi := by sorry

end NUMINAMATH_CALUDE_investor_purchase_price_l3190_319016


namespace NUMINAMATH_CALUDE_gumballs_eaten_l3190_319021

/-- The number of gumballs in each package -/
def gumballs_per_package : ℝ := 5.0

/-- The number of packages Nathan ate -/
def packages_eaten : ℝ := 20.0

/-- The total number of gumballs Nathan ate -/
def total_gumballs : ℝ := gumballs_per_package * packages_eaten

theorem gumballs_eaten :
  total_gumballs = 100.0 := by sorry

end NUMINAMATH_CALUDE_gumballs_eaten_l3190_319021


namespace NUMINAMATH_CALUDE_lana_total_pages_l3190_319067

/-- Calculate the total number of pages Lana will have after receiving pages from Duane and Alexa -/
theorem lana_total_pages
  (lana_initial : ℕ)
  (duane_pages : ℕ)
  (duane_percentage : ℚ)
  (alexa_pages : ℕ)
  (alexa_percentage : ℚ)
  (h1 : lana_initial = 8)
  (h2 : duane_pages = 42)
  (h3 : duane_percentage = 70 / 100)
  (h4 : alexa_pages = 48)
  (h5 : alexa_percentage = 25 / 100)
  : ℕ := by
  sorry

#check lana_total_pages

end NUMINAMATH_CALUDE_lana_total_pages_l3190_319067


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3190_319064

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 210 * r = b ∧ b * r = 140 / 60) → b = 7 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3190_319064
