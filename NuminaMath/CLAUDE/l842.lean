import Mathlib

namespace NUMINAMATH_CALUDE_bicycle_has_four_wheels_l842_84223

-- Define the universe of objects
variable (Object : Type)

-- Define predicates
variable (isCar : Object → Prop)
variable (hasFourWheels : Object → Prop)

-- Define a specific object
variable (bicycle : Object)

-- Theorem statement
theorem bicycle_has_four_wheels 
  (all_cars_have_four_wheels : ∀ x, isCar x → hasFourWheels x)
  (bicycle_is_car : isCar bicycle) :
  hasFourWheels bicycle :=
by
  sorry


end NUMINAMATH_CALUDE_bicycle_has_four_wheels_l842_84223


namespace NUMINAMATH_CALUDE_vasyas_numbers_l842_84281

theorem vasyas_numbers (x y : ℝ) : 
  x + y = x * y ∧ x + y = x / y → x = (1 : ℝ) / 2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l842_84281


namespace NUMINAMATH_CALUDE_scores_mode_and_median_l842_84244

def scores : List ℕ := [95, 97, 96, 97, 99, 98]

/-- The mode of a list of natural numbers -/
def mode (l : List ℕ) : ℕ := sorry

/-- The median of a list of natural numbers -/
def median (l : List ℕ) : ℚ := sorry

theorem scores_mode_and_median :
  mode scores = 97 ∧ median scores = 97 := by sorry

end NUMINAMATH_CALUDE_scores_mode_and_median_l842_84244


namespace NUMINAMATH_CALUDE_profit_percentage_is_twenty_percent_l842_84220

/-- Calculates the profit percentage given wholesale price, retail price, and discount percentage. -/
def profit_percentage (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that under the given conditions, the profit percentage is 20%. -/
theorem profit_percentage_is_twenty_percent :
  profit_percentage 90 120 (10/100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_twenty_percent_l842_84220


namespace NUMINAMATH_CALUDE_art_arrangement_count_l842_84249

/-- Represents the number of calligraphy works -/
def calligraphy_count : ℕ := 2

/-- Represents the number of painting works -/
def painting_count : ℕ := 2

/-- Represents the number of architectural designs -/
def architecture_count : ℕ := 1

/-- Represents the total number of art pieces -/
def total_art_pieces : ℕ := calligraphy_count + painting_count + architecture_count

/-- Calculates the number of arrangements of art pieces -/
def calculate_arrangements : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 24 -/
theorem art_arrangement_count : calculate_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_art_arrangement_count_l842_84249


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l842_84294

def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l842_84294


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l842_84225

theorem rectangular_prism_diagonal (length width height : ℝ) 
  (h_length : length = 12) 
  (h_width : width = 15) 
  (h_height : height = 8) : 
  Real.sqrt (length^2 + width^2 + height^2) = Real.sqrt 433 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l842_84225


namespace NUMINAMATH_CALUDE_train_distance_l842_84228

/-- Represents the speed of a train in miles per minute -/
def train_speed : ℚ := 3 / 2.25

/-- Represents the duration of the journey in minutes -/
def journey_duration : ℚ := 120

/-- Theorem stating that the train will travel 160 miles in 2 hours -/
theorem train_distance : train_speed * journey_duration = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l842_84228


namespace NUMINAMATH_CALUDE_negative_correlation_implies_negative_slope_l842_84290

/-- A linear regression model with two variables -/
structure LinearRegressionModel where
  x : ℝ → ℝ  -- Independent variable
  y : ℝ → ℝ  -- Dependent variable
  a : ℝ       -- Intercept
  b : ℝ       -- Slope

/-- Definition of negative correlation between two variables -/
def NegativelyCorrelated (model : LinearRegressionModel) : Prop :=
  ∀ x1 x2, x1 < x2 → model.y x1 > model.y x2

/-- Theorem: In a linear regression model, if two variables are negatively correlated, then the slope b is negative -/
theorem negative_correlation_implies_negative_slope (model : LinearRegressionModel) 
  (h : NegativelyCorrelated model) : model.b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_correlation_implies_negative_slope_l842_84290


namespace NUMINAMATH_CALUDE_g_definition_l842_84234

-- Define the function f
def f (x : ℝ) : ℝ := 5 - 2*x

-- Define the function g
def g (x : ℝ) : ℝ := 4 - 3*x

-- Theorem statement
theorem g_definition (x : ℝ) : 
  (∀ y, f (y + 1) = 3 - 2*y) ∧ (f (g x) = 6*x - 3) → g x = 4 - 3*x :=
by
  sorry

end NUMINAMATH_CALUDE_g_definition_l842_84234


namespace NUMINAMATH_CALUDE_garden_area_l842_84280

/-- Given a square garden with perimeter 48 meters and a pond of area 20 square meters inside,
    the area of the garden not taken up by the pond is 124 square meters. -/
theorem garden_area (garden_perimeter : ℝ) (pond_area : ℝ) : 
  garden_perimeter = 48 → 
  pond_area = 20 → 
  (garden_perimeter / 4) ^ 2 - pond_area = 124 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l842_84280


namespace NUMINAMATH_CALUDE_march_first_is_friday_l842_84246

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the number of occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

theorem march_first_is_friday (m : Month) : 
  m.days = 31 ∧ 
  countDayOccurrences m DayOfWeek.Friday = 5 ∧ 
  countDayOccurrences m DayOfWeek.Sunday = 4 → 
  m.firstDay = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_march_first_is_friday_l842_84246


namespace NUMINAMATH_CALUDE_geometric_series_sum_five_terms_quarter_l842_84262

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_five_terms_quarter :
  geometric_series_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_five_terms_quarter_l842_84262


namespace NUMINAMATH_CALUDE_circle_area_not_tripled_l842_84257

/-- Tripling the radius of a circle does not triple its area -/
theorem circle_area_not_tripled (r : ℝ) (h : r > 0) : π * (3 * r)^2 ≠ 3 * (π * r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_not_tripled_l842_84257


namespace NUMINAMATH_CALUDE_zeros_product_less_than_one_l842_84269

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem zeros_product_less_than_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂ → x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_less_than_one_l842_84269


namespace NUMINAMATH_CALUDE_original_decimal_l842_84259

theorem original_decimal (x : ℝ) : (100 * x = x + 29.7) → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_l842_84259


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l842_84298

/-- The number of solutions to the system of equations -/
def num_solutions : ℕ := 2

/-- A solution to the system of equations is a triple of positive integers (x, y, z) -/
def is_solution (x y z : ℕ+) : Prop :=
  x * y + x * z = 255 ∧ x * z - y * z = 224

/-- The theorem stating that there are exactly two solutions -/
theorem exactly_two_solutions :
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = num_solutions ∧ 
    ∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ is_solution x y z) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l842_84298


namespace NUMINAMATH_CALUDE_min_guesses_bound_one_guess_sufficient_two_guesses_necessary_l842_84210

/-- Given positive integers n and k with n > k, this function returns the minimum number
    of guesses required to determine a binary string of length n, given all binary strings
    that differ from it in exactly k positions. -/
def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

/-- Theorem stating that the minimum number of guesses is at most 2 and at least 1. -/
theorem min_guesses_bound (n k : ℕ) (h : n > k) :
  min_guesses n k = max 1 2 := by
  sorry

/-- Theorem stating that when n ≠ 2k, one guess is sufficient. -/
theorem one_guess_sufficient (n k : ℕ) (h1 : n > k) (h2 : n ≠ 2 * k) :
  min_guesses n k = 1 := by
  sorry

/-- Theorem stating that when n = 2k, two guesses are necessary and sufficient. -/
theorem two_guesses_necessary (n k : ℕ) (h1 : n > k) (h2 : n = 2 * k) :
  min_guesses n k = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_guesses_bound_one_guess_sufficient_two_guesses_necessary_l842_84210


namespace NUMINAMATH_CALUDE_cos_A_value_c_value_l842_84207

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = 2 * t.A ∧ Real.cos t.C = 1/8

-- Theorem 1: Prove cos A = 3/4
theorem cos_A_value (t : Triangle) (h : triangle_conditions t) : Real.cos t.A = 3/4 := by
  sorry

-- Theorem 2: Prove c = 6 when a = 4
theorem c_value (t : Triangle) (h1 : triangle_conditions t) (h2 : t.a = 4) : t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_value_c_value_l842_84207


namespace NUMINAMATH_CALUDE_yoongis_answer_l842_84202

theorem yoongis_answer : ∃ x : ℝ, 5 * x = 100 ∧ x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_answer_l842_84202


namespace NUMINAMATH_CALUDE_base8_subtraction_l842_84236

-- Define a function to convert base 8 numbers to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction :
  natToBase8 (base8ToNat 546 - base8ToNat 321 - base8ToNat 105) = 120 := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_l842_84236


namespace NUMINAMATH_CALUDE_justin_tim_games_count_l842_84209

/-- The number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of players to choose after Justin and Tim are already selected -/
def players_to_choose : ℕ := players_per_game - 2

/-- The number of remaining players after Justin and Tim are excluded -/
def remaining_players : ℕ := total_players - 2

/-- Theorem stating that the number of games Justin and Tim play together
    is equal to the number of ways to choose the remaining players -/
theorem justin_tim_games_count :
  Nat.choose remaining_players players_to_choose = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_tim_games_count_l842_84209


namespace NUMINAMATH_CALUDE_non_increasing_iff_exists_greater_l842_84212

open Set

-- Define the property of being an increasing function
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the property of being a non-increasing function
def IsNonIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ f x > f y

-- Theorem statement
theorem non_increasing_iff_exists_greater (f : ℝ → ℝ) (a b : ℝ) :
  IsNonIncreasing f a b ↔ ¬(IsIncreasing f a b) :=
sorry

end NUMINAMATH_CALUDE_non_increasing_iff_exists_greater_l842_84212


namespace NUMINAMATH_CALUDE_f_equals_g_l842_84240

-- Define the two functions
def f (x : ℝ) : ℝ := (x ^ (1/3)) ^ 3
def g (x : ℝ) : ℝ := x

-- Theorem statement
theorem f_equals_g : ∀ (x : ℝ), f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l842_84240


namespace NUMINAMATH_CALUDE_ratio_value_l842_84255

theorem ratio_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_value_l842_84255


namespace NUMINAMATH_CALUDE_town_literacy_distribution_l842_84273

theorem town_literacy_distribution :
  ∀ (T : ℝ) (M F : ℝ),
    T > 0 →
    M + F = 100 →
    0.20 * M * T + 0.325 * F * T = 0.25 * T →
    M = 60 ∧ F = 40 := by
  sorry

end NUMINAMATH_CALUDE_town_literacy_distribution_l842_84273


namespace NUMINAMATH_CALUDE_xy_sum_theorem_l842_84204

theorem xy_sum_theorem (x y : ℤ) (h : 2*x*y + x + y = 83) : 
  x + y = 83 ∨ x + y = -85 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_theorem_l842_84204


namespace NUMINAMATH_CALUDE_min_value_linear_program_l842_84286

theorem min_value_linear_program :
  ∀ x y : ℝ,
  (2 * x + y - 2 ≥ 0) →
  (x - 2 * y + 4 ≥ 0) →
  (x - 1 ≤ 0) →
  ∃ (z : ℝ), z = 3 * x + 2 * y ∧ z ≥ 3 ∧ (∀ x' y' : ℝ, 
    (2 * x' + y' - 2 ≥ 0) →
    (x' - 2 * y' + 4 ≥ 0) →
    (x' - 1 ≤ 0) →
    3 * x' + 2 * y' ≥ z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_linear_program_l842_84286


namespace NUMINAMATH_CALUDE_sixth_diagram_shaded_fraction_l842_84213

/-- Represents the number of shaded triangles in the nth diagram -/
def shaded_triangles (n : ℕ) : ℕ := (n - 1) ^ 2

/-- Represents the total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := n ^ 2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded_triangles n / total_triangles n

theorem sixth_diagram_shaded_fraction :
  shaded_fraction 6 = 25 / 36 := by sorry

end NUMINAMATH_CALUDE_sixth_diagram_shaded_fraction_l842_84213


namespace NUMINAMATH_CALUDE_prime_triplet_l842_84226

theorem prime_triplet (p : ℤ) : 
  Prime p ∧ Prime (p + 2) ∧ Prime (p + 4) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_l842_84226


namespace NUMINAMATH_CALUDE_common_tangents_of_circles_l842_84253

/-- Circle C1 with equation x² + y² - 2x - 4y - 4 = 0 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- Circle C2 with equation x² + y² - 6x - 10y - 2 = 0 -/
def C2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 10*y - 2 = 0

/-- The number of common tangents between C1 and C2 -/
def num_common_tangents : ℕ := 2

theorem common_tangents_of_circles :
  num_common_tangents = 2 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_of_circles_l842_84253


namespace NUMINAMATH_CALUDE_beka_flew_more_than_jackson_l842_84235

/-- The difference in miles flown between Beka and Jackson -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating that Beka flew 310 miles more than Jackson -/
theorem beka_flew_more_than_jackson :
  miles_difference 873 563 = 310 := by
  sorry

end NUMINAMATH_CALUDE_beka_flew_more_than_jackson_l842_84235


namespace NUMINAMATH_CALUDE_sandals_sold_l842_84265

theorem sandals_sold (shoes : ℕ) (sandals : ℕ) : 
  (9 : ℚ) / 5 = shoes / sandals → shoes = 72 → sandals = 40 := by
  sorry

end NUMINAMATH_CALUDE_sandals_sold_l842_84265


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l842_84211

theorem tenth_term_of_sequence (a : ℕ → ℚ) :
  (∀ n : ℕ, a n = (-1)^(n+1) * (2*n) / (2*n+1)) →
  a 10 = -20 / 21 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l842_84211


namespace NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l842_84241

theorem percentage_subtraction_equivalence (a : ℝ) : 
  a - (0.05 * a) = 0.95 * a := by sorry

end NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l842_84241


namespace NUMINAMATH_CALUDE_sin_120_degrees_l842_84252

theorem sin_120_degrees : Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l842_84252


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l842_84299

theorem gcd_n_cube_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l842_84299


namespace NUMINAMATH_CALUDE_parallelogram_EFGH_area_l842_84221

/-- Represents a parallelogram with a base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.base * p.height

/-- Theorem: The area of parallelogram EFGH is 18 square units -/
theorem parallelogram_EFGH_area :
  let p : Parallelogram := { base := 6, height := 3 }
  area p = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_EFGH_area_l842_84221


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l842_84215

/-- A circle passing through three points -/
structure Circle where
  D : ℝ
  E : ℝ

/-- Check if a point lies on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y = 0

/-- The specific circle we're interested in -/
def our_circle : Circle := { D := -4, E := -6 }

/-- Theorem stating that our_circle passes through the given points -/
theorem circle_passes_through_points : 
  (our_circle.contains 0 0) ∧ 
  (our_circle.contains 4 0) ∧ 
  (our_circle.contains (-1) 1) := by
  sorry


end NUMINAMATH_CALUDE_circle_passes_through_points_l842_84215


namespace NUMINAMATH_CALUDE_quadratic_inequality_l842_84230

theorem quadratic_inequality (x : ℝ) : 2 * x^2 - 6 * x - 56 > 0 ↔ x < -4 ∨ x > 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l842_84230


namespace NUMINAMATH_CALUDE_meeting_2015_same_as_first_l842_84277

/-- Represents a point on a line segment --/
structure Point :=
  (position : ℝ)

/-- Represents a person moving on a line segment --/
structure Person :=
  (speed : ℝ)
  (startPosition : Point)
  (startTime : ℝ)

/-- Represents a meeting between two people --/
structure Meeting :=
  (position : Point)
  (time : ℝ)

/-- The theorem stating that the 2015th meeting occurs at the same point as the first meeting --/
theorem meeting_2015_same_as_first 
  (a b : Person) 
  (segment : Set Point) 
  (first_meeting last_meeting : Meeting) :
  first_meeting.position = last_meeting.position :=
sorry

end NUMINAMATH_CALUDE_meeting_2015_same_as_first_l842_84277


namespace NUMINAMATH_CALUDE_event_probability_l842_84238

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_event_probability_l842_84238


namespace NUMINAMATH_CALUDE_unique_solution_l842_84260

theorem unique_solution : ∃! n : ℕ, n > 0 ∧ Nat.lcm n 150 = Nat.gcd n 150 + 600 ∧ n = 675 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l842_84260


namespace NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_solution_system_equations_l842_84203

-- Problem 1
theorem solution_equation_one (x : ℝ) : 4 - 3 * x = 6 - 5 * x ↔ x = 1 := by sorry

-- Problem 2
theorem solution_equation_two (x : ℝ) : (x + 1) / 2 - 1 = (2 - x) / 3 ↔ x = 7 / 5 := by sorry

-- Problem 3
theorem solution_system_equations (x y : ℝ) : 3 * x - y = 7 ∧ x + 3 * y = -1 ↔ x = 2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_solution_system_equations_l842_84203


namespace NUMINAMATH_CALUDE_cube_coloring_l842_84200

theorem cube_coloring (n : ℕ) (h : n > 0) : 
  (∃ (W B : ℕ), W + B = n^3 ∧ 
   3 * W = 3 * B ∧ 
   2 * W = n^3) → 
  ∃ k : ℕ, n = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_cube_coloring_l842_84200


namespace NUMINAMATH_CALUDE_three_numbers_product_sum_l842_84289

theorem three_numbers_product_sum (x y z : ℝ) : 
  (x * y + x + y = 8) ∧ 
  (y * z + y + z = 15) ∧ 
  (x * z + x + z = 24) → 
  x = 209 / 25 ∧ y = 7 ∧ z = 17 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_product_sum_l842_84289


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l842_84237

theorem quadruple_equation_solutions :
  ∀ (a b c d : ℝ),
  (b + c + d)^2010 = 3 * a ∧
  (a + c + d)^2010 = 3 * b ∧
  (a + b + d)^2010 = 3 * c ∧
  (a + b + c)^2010 = 3 * d →
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
   (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l842_84237


namespace NUMINAMATH_CALUDE_kingdom_cats_and_hogs_l842_84291

theorem kingdom_cats_and_hogs (num_hogs : ℕ) (num_cats : ℕ) : 
  num_hogs = 630 → 
  num_hogs = 7 * num_cats → 
  15 < (0.8 * (num_cats^2 : ℝ)) → 
  (0.8 * (num_cats^2 : ℝ)) - 15 = 6465 := by
sorry

end NUMINAMATH_CALUDE_kingdom_cats_and_hogs_l842_84291


namespace NUMINAMATH_CALUDE_product_evaluation_l842_84233

theorem product_evaluation (n : ℤ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l842_84233


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l842_84275

def N : ℕ := sorry

theorem highest_power_of_three_dividing_N : 
  (∃ m : ℕ, N = 3 * m) ∧ ¬(∃ m : ℕ, N = 9 * m) := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l842_84275


namespace NUMINAMATH_CALUDE_max_side_length_triangle_l842_84266

theorem max_side_length_triangle (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Different side lengths
  a + b + c = 24 →  -- Perimeter is 24
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 →  -- Maximum side length is 11
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
  ∃ (x y z : ℕ), x + y + z = 24 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x ≤ 11 ∧ y ≤ 11 ∧ z ≤ 11 ∧ 
    (x + y > z ∧ y + z > x ∧ x + z > y) ∧
    (∀ w : ℕ, w > 11 → ¬(∃ u v : ℕ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
      u + v + w = 24 ∧ u + v > w ∧ v + w > u ∧ u + w > v)) :=
by sorry

end NUMINAMATH_CALUDE_max_side_length_triangle_l842_84266


namespace NUMINAMATH_CALUDE_three_student_committees_from_eight_l842_84296

theorem three_student_committees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_student_committees_from_eight_l842_84296


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l842_84217

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_18 : 
  ∀ n : ℕ, n < 10 → 
    (is_divisible_by (3140 + n) 18 ↔ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l842_84217


namespace NUMINAMATH_CALUDE_jills_salary_l842_84268

theorem jills_salary (discretionary_income : ℝ) (net_salary : ℝ) : 
  discretionary_income = net_salary / 5 →
  discretionary_income * 0.15 = 105 →
  net_salary = 3500 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l842_84268


namespace NUMINAMATH_CALUDE_new_city_building_count_l842_84261

/-- Represents the number of buildings of each type in Pittsburgh -/
structure PittsburghBuildings where
  stores : Nat
  hospitals : Nat
  schools : Nat
  police_stations : Nat

/-- Calculates the total number of buildings for the new city based on Pittsburgh's data -/
def new_city_buildings (p : PittsburghBuildings) : Nat :=
  p.stores / 2 + p.hospitals * 2 + (p.schools - 50) + (p.police_stations + 5)

/-- The theorem stating that given Pittsburgh's building numbers, the new city will require 2175 buildings -/
theorem new_city_building_count (p : PittsburghBuildings) 
  (h1 : p.stores = 2000)
  (h2 : p.hospitals = 500)
  (h3 : p.schools = 200)
  (h4 : p.police_stations = 20) :
  new_city_buildings p = 2175 := by
  sorry

end NUMINAMATH_CALUDE_new_city_building_count_l842_84261


namespace NUMINAMATH_CALUDE_samantha_routes_l842_84231

/-- The number of ways to arrange n blocks in two directions --/
def arrangeBlocks (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The number of diagonal paths through the park --/
def diagonalPaths : ℕ := 2

/-- The total number of routes Samantha can take --/
def totalRoutes : ℕ := arrangeBlocks 3 * diagonalPaths * arrangeBlocks 3

theorem samantha_routes :
  totalRoutes = 800 := by
  sorry

end NUMINAMATH_CALUDE_samantha_routes_l842_84231


namespace NUMINAMATH_CALUDE_gcd_problem_l842_84247

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ b = k * 1177) :
  Nat.gcd (Int.natAbs (2 * b^2 + 31 * b + 71)) (Int.natAbs (b + 15)) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l842_84247


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l842_84201

theorem factorization_of_cubic (x : ℝ) : 6 * x^3 - 24 = 6 * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l842_84201


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l842_84245

/-- The probability of a randomly selected point in a square with side length 6 
    being within 2 units of the center is π/9. -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 → circle_radius = 2 → 
  (π * circle_radius^2) / (square_side^2) = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l842_84245


namespace NUMINAMATH_CALUDE_intersection_point_l842_84272

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -2

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 2

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The y-intercept of the perpendicular line -/
def b₂ : ℚ := y₀ - m₂ * x₀

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := (b₂ - b₁) / (m₁ - m₂)

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := m₁ * x_intersect + b₁

theorem intersection_point :
  (x_intersect = 7/5) ∧ (y_intersect = 11/5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l842_84272


namespace NUMINAMATH_CALUDE_sum_equals_222_l842_84263

theorem sum_equals_222 : 148 + 35 + 17 + 13 + 9 = 222 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_222_l842_84263


namespace NUMINAMATH_CALUDE_product_repeating_decimal_9_and_8_l842_84251

/-- The repeating decimal 0.999... -/
def repeating_decimal_9 : ℝ := 0.999999

/-- Theorem: The product of 0.999... and 8 is equal to 8 -/
theorem product_repeating_decimal_9_and_8 : repeating_decimal_9 * 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_9_and_8_l842_84251


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l842_84248

theorem two_digit_number_problem : ∃ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) ∧  -- two-digit number
  (n % 10 = (n / 10) + 3) ∧  -- units digit is 3 greater than tens digit
  ((n % 10)^2 + (n / 10)^2 = (n % 10) + (n / 10) + 18) ∧  -- sum of squares condition
  n = 47 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l842_84248


namespace NUMINAMATH_CALUDE_johns_cost_per_minute_l842_84216

/-- Calculates the cost per minute for long distance calls -/
def cost_per_minute (monthly_fee : ℚ) (total_bill : ℚ) (minutes_billed : ℚ) : ℚ :=
  (total_bill - monthly_fee) / minutes_billed

/-- Theorem stating that John's cost per minute for long distance calls is $0.25 -/
theorem johns_cost_per_minute :
  let monthly_fee : ℚ := 5
  let total_bill : ℚ := 12.02
  let minutes_billed : ℚ := 28.08
  cost_per_minute monthly_fee total_bill minutes_billed = 0.25 := by
sorry

end NUMINAMATH_CALUDE_johns_cost_per_minute_l842_84216


namespace NUMINAMATH_CALUDE_cake_cross_section_is_rectangle_l842_84271

/-- A cylindrical cake -/
structure Cake where
  base_diameter : ℝ
  height : ℝ

/-- The cross-section of a cake when cut along its diameter -/
inductive CrossSection
  | Rectangle
  | Circle
  | Square
  | Undetermined

/-- The shape of the cross-section when a cylindrical cake is cut along its diameter -/
def cross_section_shape (c : Cake) : CrossSection :=
  CrossSection.Rectangle

/-- Theorem: The cross-section of a cylindrical cake with base diameter 3 cm and height 9 cm, 
    when cut along its diameter, is a rectangle -/
theorem cake_cross_section_is_rectangle :
  let c : Cake := { base_diameter := 3, height := 9 }
  cross_section_shape c = CrossSection.Rectangle := by
  sorry

end NUMINAMATH_CALUDE_cake_cross_section_is_rectangle_l842_84271


namespace NUMINAMATH_CALUDE_transaction_result_l842_84224

def initial_x : ℝ := 15000
def initial_y : ℝ := 18000
def painting_value : ℝ := 15000
def first_sale_price : ℝ := 20000
def second_sale_price : ℝ := 14000
def commission_rate : ℝ := 0.05

def first_transaction_x (initial : ℝ) (sale_price : ℝ) (commission : ℝ) : ℝ :=
  initial + sale_price * (1 - commission)

def first_transaction_y (initial : ℝ) (purchase_price : ℝ) : ℝ :=
  initial - purchase_price

def second_transaction_x (cash : ℝ) (purchase_price : ℝ) : ℝ :=
  cash - purchase_price

def second_transaction_y (cash : ℝ) (sale_price : ℝ) (commission : ℝ) : ℝ :=
  cash + sale_price * (1 - commission)

theorem transaction_result :
  let x_final := second_transaction_x (first_transaction_x initial_x first_sale_price commission_rate) second_sale_price
  let y_final := second_transaction_y (first_transaction_y initial_y first_sale_price) second_sale_price commission_rate
  (x_final - initial_x = 5000) ∧ (y_final - initial_y = -6700) :=
by sorry

end NUMINAMATH_CALUDE_transaction_result_l842_84224


namespace NUMINAMATH_CALUDE_part_i_part_ii_l842_84206

-- Define the function f
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

-- Part I
theorem part_i : 
  let m : ℝ := -1
  let a : ℝ := -1
  {x : ℝ | f x a m ≥ x} = {x : ℝ | x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 2)} := by sorry

-- Part II
theorem part_ii (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2) 
  (h4 : a ≤ -3 ∨ a ≥ 3) : 
  m = 1/3 := by sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l842_84206


namespace NUMINAMATH_CALUDE_five_single_beds_weight_l842_84274

/-- The weight of a single bed in kg -/
def single_bed_weight : ℝ := sorry

/-- The weight of a double bed in kg -/
def double_bed_weight : ℝ := sorry

/-- A double bed is 10 kg heavier than a single bed -/
axiom double_bed_heavier : double_bed_weight = single_bed_weight + 10

/-- The total weight of 2 single beds and 4 double beds is 100 kg -/
axiom total_weight : 2 * single_bed_weight + 4 * double_bed_weight = 100

theorem five_single_beds_weight :
  5 * single_bed_weight = 50 := by sorry

end NUMINAMATH_CALUDE_five_single_beds_weight_l842_84274


namespace NUMINAMATH_CALUDE_water_consumption_difference_l842_84205

/-- The yearly water consumption difference between two schools -/
theorem water_consumption_difference 
  (chunlei_daily : ℕ) -- Daily water consumption of Chunlei Central Elementary School
  (days_per_year : ℕ) -- Number of days in a year
  (h1 : chunlei_daily = 111) -- Chunlei's daily consumption is 111 kg
  (h2 : days_per_year = 365) -- A year has 365 days
  : 
  chunlei_daily * days_per_year - (chunlei_daily / 3) * days_per_year = 26910 :=
by sorry

end NUMINAMATH_CALUDE_water_consumption_difference_l842_84205


namespace NUMINAMATH_CALUDE_butterfat_percentage_in_cream_l842_84285

/-- The percentage of butterfat in cream when mixed with skim milk to achieve a target butterfat percentage -/
theorem butterfat_percentage_in_cream 
  (cream_volume : ℝ) 
  (skim_milk_volume : ℝ) 
  (skim_milk_butterfat : ℝ) 
  (final_mixture_butterfat : ℝ) 
  (h1 : cream_volume = 1)
  (h2 : skim_milk_volume = 3)
  (h3 : skim_milk_butterfat = 5.5)
  (h4 : final_mixture_butterfat = 6.5)
  (h5 : cream_volume + skim_milk_volume = 4) :
  ∃ (cream_butterfat : ℝ), 
    cream_butterfat = 9.5 ∧ 
    cream_butterfat * cream_volume + skim_milk_butterfat * skim_milk_volume = 
    final_mixture_butterfat * (cream_volume + skim_milk_volume) := by
  sorry


end NUMINAMATH_CALUDE_butterfat_percentage_in_cream_l842_84285


namespace NUMINAMATH_CALUDE_tv_watching_time_l842_84208

/-- Given children watch 6 hours of television in 2 weeks and are allowed to watch 4 days a week,
    prove they spend 45 minutes each day watching television. -/
theorem tv_watching_time (hours_per_two_weeks : ℕ) (days_per_week : ℕ) 
    (h1 : hours_per_two_weeks = 6) 
    (h2 : days_per_week = 4) : 
  (hours_per_two_weeks * 60) / (days_per_week * 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_l842_84208


namespace NUMINAMATH_CALUDE_simplify_expression_l842_84222

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l842_84222


namespace NUMINAMATH_CALUDE_coordinates_of_point_E_l842_84239

/-- Given points A, B, C, D, and E in the plane, where D lies on line AB and E is on the extension of DC,
    prove that E has specific coordinates. -/
theorem coordinates_of_point_E (A B C D E : ℝ × ℝ) : 
  A = (-2, 1) →
  B = (1, 4) →
  C = (4, -3) →
  (∃ t : ℝ, D = (1 - t) • A + t • B ∧ t = 2/3) →
  (∃ s : ℝ, E = (1 + s) • D - s • C ∧ s = 5) →
  E = (-8/3, 11/3) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_E_l842_84239


namespace NUMINAMATH_CALUDE_max_bananas_purchase_l842_84278

def apple_cost : ℕ := 3
def orange_cost : ℕ := 5
def banana_cost : ℕ := 8
def total_budget : ℕ := 100

def is_valid_purchase (apples oranges bananas : ℕ) : Prop :=
  apples ≥ 1 ∧ oranges ≥ 1 ∧ bananas ≥ 1 ∧
  apple_cost * apples + orange_cost * oranges + banana_cost * bananas ≤ total_budget

theorem max_bananas_purchase :
  ∃ (apples oranges : ℕ),
    is_valid_purchase apples oranges 11 ∧
    ∀ (a o b : ℕ), is_valid_purchase a o b → b ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_bananas_purchase_l842_84278


namespace NUMINAMATH_CALUDE_tuesday_is_only_valid_start_day_l842_84218

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (advance_days d m)

def voucher_days (start : DayOfWeek) : List DayOfWeek :=
  List.map (fun i => advance_days start (i * 7)) [0, 1, 2, 3, 4]

theorem tuesday_is_only_valid_start_day :
  ∀ (start : DayOfWeek),
    (∀ (d : DayOfWeek), d ∈ voucher_days start → d ≠ DayOfWeek.Monday) ↔
    start = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_tuesday_is_only_valid_start_day_l842_84218


namespace NUMINAMATH_CALUDE_range_of_a_l842_84256

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, x^a ≥ 2 * Real.exp (2*x) * f a x + Real.exp (2*x)) →
  a ≤ 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l842_84256


namespace NUMINAMATH_CALUDE_zeros_of_continuous_function_l842_84250

theorem zeros_of_continuous_function 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (a b c : ℝ) (hab : a < b) (hbc : b < c)
  (hab_sign : f a * f b < 0) (hbc_sign : f b * f c < 0) :
  ∃ (n : ℕ), n > 0 ∧ Even n ∧ 
  (∃ (S : Finset ℝ), S.card = n ∧ 
    (∀ x ∈ S, a < x ∧ x < c ∧ f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_continuous_function_l842_84250


namespace NUMINAMATH_CALUDE_sample_size_proof_l842_84270

theorem sample_size_proof (n : ℕ) : 
  (∃ (x : ℚ), 
    x > 0 ∧ 
    2*x + 3*x + 4*x + 6*x + 4*x + x = 1 ∧ 
    (2*x + 3*x + 4*x) * n = 27) → 
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_proof_l842_84270


namespace NUMINAMATH_CALUDE_cos_135_degrees_l842_84282

theorem cos_135_degrees : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l842_84282


namespace NUMINAMATH_CALUDE_somu_age_problem_l842_84243

/-- Proves that Somu was one-fifth of his father's age 6 years ago -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 12 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 6 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l842_84243


namespace NUMINAMATH_CALUDE_total_accidents_across_highways_l842_84288

/-- Represents the accident rate and traffic data for a highway -/
structure HighwayData where
  accidents : ℕ
  vehicles : ℕ
  totalTraffic : ℕ

/-- Calculates the number of accidents for a given highway -/
def calculateAccidents (data : HighwayData) : ℕ :=
  (data.accidents * data.totalTraffic) / data.vehicles

/-- The data for Highway A -/
def highwayA : HighwayData :=
  { accidents := 75, vehicles := 100000000, totalTraffic := 2500000000 }

/-- The data for Highway B -/
def highwayB : HighwayData :=
  { accidents := 50, vehicles := 80000000, totalTraffic := 1600000000 }

/-- The data for Highway C -/
def highwayC : HighwayData :=
  { accidents := 90, vehicles := 200000000, totalTraffic := 1900000000 }

/-- Theorem stating that the total number of accidents across all three highways is 3730 -/
theorem total_accidents_across_highways :
  calculateAccidents highwayA + calculateAccidents highwayB + calculateAccidents highwayC = 3730 :=
by
  sorry

end NUMINAMATH_CALUDE_total_accidents_across_highways_l842_84288


namespace NUMINAMATH_CALUDE_dress_designs_count_l842_84214

/-- The number of different fabric colors available -/
def num_colors : ℕ := 5

/-- The number of different patterns available -/
def num_patterns : ℕ := 4

/-- The number of different sizes available -/
def num_sizes : ℕ := 3

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns * num_sizes

/-- Theorem stating that the total number of possible dress designs is 60 -/
theorem dress_designs_count : total_designs = 60 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l842_84214


namespace NUMINAMATH_CALUDE_sam_carrots_l842_84297

/-- Given that Sandy grew 6 carrots and the total number of carrots grown is 9,
    prove that Sam grew 3 carrots. -/
theorem sam_carrots (sandy_carrots : ℕ) (total_carrots : ℕ) (sam_carrots : ℕ) :
  sandy_carrots = 6 → total_carrots = 9 → sam_carrots = total_carrots - sandy_carrots →
  sam_carrots = 3 := by
  sorry

#check sam_carrots

end NUMINAMATH_CALUDE_sam_carrots_l842_84297


namespace NUMINAMATH_CALUDE_parallel_vectors_result_symmetric_function_range_l842_84219

-- Part 1
theorem parallel_vectors_result (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, Real.cos x)
  let b : ℝ × ℝ := (3, -1)
  (∃ (k : ℝ), a = k • b) →
  2 * (Real.sin x)^2 - 3 * (Real.cos x)^2 = 3/2 := by sorry

-- Part 2
theorem symmetric_function_range (x m : ℝ) :
  let a : ℝ → ℝ × ℝ := λ t => (Real.sin t, m * Real.cos t)
  let b : ℝ × ℝ := (3, -1)
  let f : ℝ → ℝ := λ t => (a t).1 * b.1 + (a t).2 * b.2
  (∀ t, f (2*π/3 - t) = f (2*π/3 + t)) →
  ∃ y ∈ Set.Icc (-Real.sqrt 3) (2 * Real.sqrt 3),
    ∃ x ∈ Set.Icc (π/8) (2*π/3), f (2*x) = y := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_result_symmetric_function_range_l842_84219


namespace NUMINAMATH_CALUDE_working_partner_receives_6000_l842_84267

/-- Calculates the amount received by the working partner in a business partnership --/
def amount_received_by_working_partner (total_profit management_fee_percentage a_capital b_capital : ℚ) : ℚ :=
  let management_fee := management_fee_percentage * total_profit
  let remaining_profit := total_profit - management_fee
  let total_capital := a_capital + b_capital
  let a_share := (a_capital / total_capital) * remaining_profit
  management_fee + a_share

/-- Theorem stating that the working partner receives 6000 Rs given the specified conditions --/
theorem working_partner_receives_6000 :
  let total_profit : ℚ := 9600
  let management_fee_percentage : ℚ := 1/10
  let a_capital : ℚ := 3500
  let b_capital : ℚ := 2500
  amount_received_by_working_partner total_profit management_fee_percentage a_capital b_capital = 6000 := by
  sorry

end NUMINAMATH_CALUDE_working_partner_receives_6000_l842_84267


namespace NUMINAMATH_CALUDE_nested_fraction_equation_l842_84254

theorem nested_fraction_equation (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 53/16 → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equation_l842_84254


namespace NUMINAMATH_CALUDE_factorial_calculation_l842_84295

theorem factorial_calculation : (Nat.factorial 11) / (Nat.factorial 10) * 12 = 132 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l842_84295


namespace NUMINAMATH_CALUDE_sqrt_product_equals_product_l842_84292

theorem sqrt_product_equals_product : Real.sqrt (4 * 9) = 2 * 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_product_l842_84292


namespace NUMINAMATH_CALUDE_loss_percentage_proof_l842_84258

def calculate_loss_percentage (cost_prices selling_prices : List ℚ) : ℚ :=
  let total_cp := cost_prices.sum
  let total_sp := selling_prices.sum
  let loss := total_cp - total_sp
  (loss / total_cp) * 100

theorem loss_percentage_proof (cost_prices selling_prices : List ℚ) :
  cost_prices = [1200, 1500, 1800] →
  selling_prices = [800, 1300, 1500] →
  calculate_loss_percentage cost_prices selling_prices = 20 := by
  sorry

#eval calculate_loss_percentage [1200, 1500, 1800] [800, 1300, 1500]

end NUMINAMATH_CALUDE_loss_percentage_proof_l842_84258


namespace NUMINAMATH_CALUDE_diagonal_length_range_l842_84293

/-- Represents a quadrilateral with given side lengths and an integer diagonal -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℤ)

/-- The theorem stating the possible values for the diagonal EG -/
theorem diagonal_length_range (q : Quadrilateral)
  (h1 : q.EF = 7)
  (h2 : q.FG = 12)
  (h3 : q.GH = 7)
  (h4 : q.HE = 15) :
  9 ≤ q.EG ∧ q.EG ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_range_l842_84293


namespace NUMINAMATH_CALUDE_inequality_equivalence_l842_84242

theorem inequality_equivalence (x : ℝ) : 
  3/20 + |2*x - 5/40| < 9/40 ↔ 1/40 < x ∧ x < 1/10 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l842_84242


namespace NUMINAMATH_CALUDE_function_inequality_l842_84229

/-- Given a continuous function f: ℝ → ℝ such that xf'(x) < 0 for all x in ℝ,
    prove that f(-1) + f(1) < 2f(0). -/
theorem function_inequality (f : ℝ → ℝ) 
    (hf_cont : Continuous f) 
    (hf_deriv : ∀ x : ℝ, x * (deriv f x) < 0) : 
    f (-1) + f 1 < 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l842_84229


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l842_84279

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascalRowElements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascalTriangleSum (n : ℕ) : ℕ := 
  (n + 1) * (n + 2) / 2

theorem pascal_triangle_30_rows_sum :
  pascalTriangleSum 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l842_84279


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l842_84264

theorem system_of_equations_solution (x y m : ℝ) : 
  (3 * x - y = 4 * m + 1) → 
  (x + y = 2 * m - 5) → 
  (x - y = 4) → 
  (m = 1) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l842_84264


namespace NUMINAMATH_CALUDE_not_red_ball_percentage_is_52_5_percent_l842_84232

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  cube_percentage : ℝ
  red_ball_percentage : ℝ

/-- Calculates the percentage of objects in the urn that are not red balls -/
def not_red_ball_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.cube_percentage) * (1 - urn.red_ball_percentage)

/-- Theorem stating that the percentage of objects in the urn that are not red balls is 52.5% -/
theorem not_red_ball_percentage_is_52_5_percent (urn : UrnComposition)
  (h1 : urn.cube_percentage = 0.3)
  (h2 : urn.red_ball_percentage = 0.25) :
  not_red_ball_percentage urn = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_not_red_ball_percentage_is_52_5_percent_l842_84232


namespace NUMINAMATH_CALUDE_max_profit_is_4900_l842_84284

/-- A transportation problem with two types of trucks --/
structure TransportProblem where
  driversAvailable : ℕ
  workersAvailable : ℕ
  typeATrucks : ℕ
  typeBTrucks : ℕ
  typeATruckCapacity : ℕ
  typeBTruckCapacity : ℕ
  minTonsToTransport : ℕ
  typeAWorkersRequired : ℕ
  typeBWorkersRequired : ℕ
  typeAProfit : ℕ
  typeBProfit : ℕ

/-- The solution to the transportation problem --/
structure TransportSolution where
  typeATrucksUsed : ℕ
  typeBTrucksUsed : ℕ

/-- Calculate the profit for a given solution --/
def calculateProfit (p : TransportProblem) (s : TransportSolution) : ℕ :=
  p.typeAProfit * s.typeATrucksUsed + p.typeBProfit * s.typeBTrucksUsed

/-- Check if a solution is valid for a given problem --/
def isValidSolution (p : TransportProblem) (s : TransportSolution) : Prop :=
  s.typeATrucksUsed ≤ p.typeATrucks ∧
  s.typeBTrucksUsed ≤ p.typeBTrucks ∧
  s.typeATrucksUsed * p.typeAWorkersRequired + s.typeBTrucksUsed * p.typeBWorkersRequired ≤ p.workersAvailable ∧
  s.typeATrucksUsed * p.typeATruckCapacity + s.typeBTrucksUsed * p.typeBTruckCapacity ≥ p.minTonsToTransport

/-- The main theorem stating that the maximum profit is 4900 yuan --/
theorem max_profit_is_4900 (p : TransportProblem)
  (h1 : p.driversAvailable = 12)
  (h2 : p.workersAvailable = 19)
  (h3 : p.typeATrucks = 8)
  (h4 : p.typeBTrucks = 7)
  (h5 : p.typeATruckCapacity = 10)
  (h6 : p.typeBTruckCapacity = 6)
  (h7 : p.minTonsToTransport = 72)
  (h8 : p.typeAWorkersRequired = 2)
  (h9 : p.typeBWorkersRequired = 1)
  (h10 : p.typeAProfit = 450)
  (h11 : p.typeBProfit = 350) :
  ∃ (s : TransportSolution), isValidSolution p s ∧ 
  calculateProfit p s = 4900 ∧ 
  ∀ (s' : TransportSolution), isValidSolution p s' → calculateProfit p s' ≤ 4900 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_is_4900_l842_84284


namespace NUMINAMATH_CALUDE_numerator_increase_l842_84276

theorem numerator_increase (x y a : ℝ) : 
  x / y = 2 / 5 → 
  x + y = 5.25 → 
  (x + a) / (2 * y) = 1 / 3 → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_numerator_increase_l842_84276


namespace NUMINAMATH_CALUDE_complex_equation_solution_l842_84287

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (3 - 2 * i * z = 1 + 4 * i * z) ∧ (z = -i / 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l842_84287


namespace NUMINAMATH_CALUDE_distance_is_sqrt_152_l842_84227

/-- The distance between two adjacent parallel lines intersecting a circle -/
def distance_between_lines (r : ℝ) (d : ℝ) : Prop :=
  ∃ (chord1 chord2 chord3 : ℝ),
    chord1 = 40 ∧ chord2 = 36 ∧ chord3 = 34 ∧
    40 * r^2 = 800 + 10 * d^2 ∧
    36 * r^2 = 648 + 9 * d^2 ∧
    d = Real.sqrt 152

/-- Theorem stating that the distance between two adjacent parallel lines is √152 -/
theorem distance_is_sqrt_152 :
  ∃ (r : ℝ), distance_between_lines r (Real.sqrt 152) :=
sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_152_l842_84227


namespace NUMINAMATH_CALUDE_inequality_proof_l842_84283

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l842_84283
