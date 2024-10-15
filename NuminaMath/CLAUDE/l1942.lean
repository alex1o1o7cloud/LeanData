import Mathlib

namespace NUMINAMATH_CALUDE_bob_sister_time_relation_l1942_194226

/-- Bob's current time for a mile in seconds -/
def bob_current_time : ℝ := 640

/-- The percentage improvement Bob needs to make -/
def improvement_percentage : ℝ := 9.062499999999996

/-- Bob's sister's time for a mile in seconds -/
def sister_time : ℝ := 582

/-- Theorem stating the relationship between Bob's current time, improvement percentage, and his sister's time -/
theorem bob_sister_time_relation :
  sister_time = bob_current_time * (1 - improvement_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_bob_sister_time_relation_l1942_194226


namespace NUMINAMATH_CALUDE_regression_not_exact_l1942_194270

-- Define the linear regression model
def linear_regression (x : ℝ) : ℝ := 0.5 * x - 85

-- Define the specific x value we're interested in
def x_value : ℝ := 200

-- Theorem stating that y is not necessarily exactly 15 when x = 200
theorem regression_not_exact : 
  ∃ (ε : ℝ), ε ≠ 0 ∧ linear_regression x_value + ε = 15 := by
  sorry

end NUMINAMATH_CALUDE_regression_not_exact_l1942_194270


namespace NUMINAMATH_CALUDE_psychology_majors_percentage_l1942_194256

/-- Given a college with the following properties:
  * 40% of total students are freshmen
  * 50% of freshmen are enrolled in the school of liberal arts
  * 10% of total students are freshmen psychology majors in the school of liberal arts
  Prove that 50% of freshmen in the school of liberal arts are psychology majors -/
theorem psychology_majors_percentage 
  (total_students : ℕ) 
  (freshmen_percent : ℚ) 
  (liberal_arts_percent : ℚ) 
  (psych_majors_percent : ℚ) 
  (h1 : freshmen_percent = 40 / 100) 
  (h2 : liberal_arts_percent = 50 / 100) 
  (h3 : psych_majors_percent = 10 / 100) : 
  (psych_majors_percent * total_students) / (freshmen_percent * liberal_arts_percent * total_students) = 50 / 100 := by
  sorry

end NUMINAMATH_CALUDE_psychology_majors_percentage_l1942_194256


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l1942_194298

/-- For a complex number z = 2x + 3iy, |z|^2 = 4x^2 + 9y^2 -/
theorem complex_magnitude_squared (x y : ℝ) : 
  let z : ℂ := 2*x + 3*y*Complex.I
  Complex.normSq z = 4*x^2 + 9*y^2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l1942_194298


namespace NUMINAMATH_CALUDE_shells_added_l1942_194212

/-- The amount of shells added to Jovana's bucket -/
theorem shells_added (initial_amount final_amount : ℝ) 
  (h1 : initial_amount = 5.75)
  (h2 : final_amount = 28.3) : 
  final_amount - initial_amount = 22.55 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_l1942_194212


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l1942_194248

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l1942_194248


namespace NUMINAMATH_CALUDE_coloring_four_cells_six_colors_l1942_194241

def ColoringMethods (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  let twoColorMethods := (Nat.choose n 2) * 2
  let threeColorMethods := (Nat.choose n 3) * (3 * 2^3 - Nat.choose 3 2 * 2)
  twoColorMethods + threeColorMethods

theorem coloring_four_cells_six_colors :
  ColoringMethods 6 4 3 = 390 :=
sorry

end NUMINAMATH_CALUDE_coloring_four_cells_six_colors_l1942_194241


namespace NUMINAMATH_CALUDE_twelve_sided_polygon_equilateral_triangles_l1942_194202

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Function to check if a triangle is equilateral -/
def isEquilateral (t : EquilateralTriangle) : Prop := sorry

/-- Function to check if a triangle has at least two vertices from a given set -/
def hasAtLeastTwoVerticesFrom (t : EquilateralTriangle) (s : Set (ℝ × ℝ)) : Prop := sorry

/-- The main theorem -/
theorem twelve_sided_polygon_equilateral_triangles 
  (p : RegularPolygon 12) : 
  ∃ (ts : Finset EquilateralTriangle), 
    (∀ t ∈ ts, isEquilateral t ∧ 
      hasAtLeastTwoVerticesFrom t (Set.range p.vertices)) ∧ 
    ts.card ≥ 12 := by sorry

end NUMINAMATH_CALUDE_twelve_sided_polygon_equilateral_triangles_l1942_194202


namespace NUMINAMATH_CALUDE_cubic_function_zeros_l1942_194284

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

theorem cubic_function_zeros (c : ℝ) :
  (∀ a : ℝ, (a < -3 ∨ (1 < a ∧ a < 3/2) ∨ a > 3/2) →
    ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f a (c - a) x₁ = 0 ∧ f a (c - a) x₂ = 0 ∧ f a (c - a) x₃ = 0) →
  c = 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_zeros_l1942_194284


namespace NUMINAMATH_CALUDE_gcd_1037_425_l1942_194299

theorem gcd_1037_425 : Nat.gcd 1037 425 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1037_425_l1942_194299


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l1942_194243

/-- Represents the number of ways to make 50 cents using 5 cent and 7 cent stamps -/
def stamp_combinations : Set (ℕ × ℕ) :=
  {(s, t) | 5 * s + 7 * t = 50 ∧ s ≥ 0 ∧ t ≥ 0}

/-- The total number of stamps used in a combination -/
def total_stamps (combination : ℕ × ℕ) : ℕ :=
  combination.1 + combination.2

theorem min_stamps_for_50_cents :
  ∃ (combination : ℕ × ℕ),
    combination ∈ stamp_combinations ∧
    (∀ other ∈ stamp_combinations, total_stamps combination ≤ total_stamps other) ∧
    total_stamps combination = 8 :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l1942_194243


namespace NUMINAMATH_CALUDE_cookies_eaten_l1942_194292

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem cookies_eaten :
  initial_cookies - remaining_cookies = 9 :=
by sorry

end NUMINAMATH_CALUDE_cookies_eaten_l1942_194292


namespace NUMINAMATH_CALUDE_min_matches_25_players_l1942_194272

/-- Represents a chess tournament. -/
structure ChessTournament where
  numPlayers : ℕ
  skillLevels : Fin numPlayers → ℕ
  uniqueSkills : ∀ i j, i ≠ j → skillLevels i ≠ skillLevels j

/-- The minimum number of matches required to determine the two strongest players. -/
def minMatchesForTopTwo (tournament : ChessTournament) : ℕ :=
  -- Definition to be proved
  28

/-- Theorem stating the minimum number of matches for a 25-player tournament. -/
theorem min_matches_25_players (tournament : ChessTournament) 
  (h_players : tournament.numPlayers = 25) :
  minMatchesForTopTwo tournament = 28 := by
  sorry

#check min_matches_25_players

end NUMINAMATH_CALUDE_min_matches_25_players_l1942_194272


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_390_l1942_194228

theorem sin_n_eq_cos_390 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (390 * π / 180) →
  n = 60 ∨ n = 120 := by
sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_390_l1942_194228


namespace NUMINAMATH_CALUDE_right_triangle_circles_l1942_194218

theorem right_triangle_circles (a b : ℝ) (R r : ℝ) : 
  a = 16 → b = 30 → 
  R = (a^2 + b^2).sqrt / 2 → 
  r = (a * b) / (a + b + (a^2 + b^2).sqrt) → 
  R + r = 23 := by sorry

end NUMINAMATH_CALUDE_right_triangle_circles_l1942_194218


namespace NUMINAMATH_CALUDE_valentines_day_problem_l1942_194249

theorem valentines_day_problem (boys girls : ℕ) : 
  boys * girls = boys + girls + 52 → boys * girls = 108 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_problem_l1942_194249


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1942_194242

/-- Given three non-overlapping circles with radii r₁, r₂, r₃ where r₁ > r₂ and r₁ > r₃,
    the quadrilateral formed by their external common tangents has an inscribed circle
    with radius r = (r₁r₂r₃) / (r₁r₂ - r₁r₃ - r₂r₃) -/
theorem inscribed_circle_radius
  (r₁ r₂ r₃ : ℝ)
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0)
  (h₄ : r₁ > r₂) (h₅ : r₁ > r₃)
  (h_non_overlap : r₁ < r₂ + r₃) :
  ∃ r : ℝ, r > 0 ∧ r = (r₁ * r₂ * r₃) / (r₁ * r₂ - r₁ * r₃ - r₂ * r₃) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1942_194242


namespace NUMINAMATH_CALUDE_amy_albums_count_l1942_194229

/-- The number of photos Amy uploaded to Facebook -/
def total_photos : ℕ := 180

/-- The number of photos in each album -/
def photos_per_album : ℕ := 20

/-- The number of albums Amy created -/
def num_albums : ℕ := total_photos / photos_per_album

theorem amy_albums_count : num_albums = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_albums_count_l1942_194229


namespace NUMINAMATH_CALUDE_solve_system_of_equations_no_solution_for_inequalities_l1942_194281

-- Part 1: System of equations
theorem solve_system_of_equations :
  ∃! (x y : ℝ), x - 3 * y = -5 ∧ 2 * x + 2 * y = 6 ∧ x = 1 ∧ y = 2 :=
by sorry

-- Part 2: System of inequalities
theorem no_solution_for_inequalities :
  ¬∃ (x : ℝ), 2 * x < -4 ∧ (1/2) * x - 5 > 1 - (3/2) * x :=
by sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_no_solution_for_inequalities_l1942_194281


namespace NUMINAMATH_CALUDE_daughters_age_l1942_194208

theorem daughters_age (mother_age : ℕ) (daughter_age : ℕ) : 
  mother_age = 42 → 
  (mother_age + 9) = 3 * (daughter_age + 9) → 
  daughter_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_daughters_age_l1942_194208


namespace NUMINAMATH_CALUDE_find_number_from_announcements_l1942_194259

def circle_number_game (numbers : Fin 15 → ℝ) (announcements : Fin 15 → ℝ) : Prop :=
  ∀ i : Fin 15, announcements i = (numbers (i - 1) + numbers (i + 1)) / 2

theorem find_number_from_announcements 
  (numbers : Fin 15 → ℝ) (announcements : Fin 15 → ℝ)
  (h_circle : circle_number_game numbers announcements)
  (h_8th : announcements 7 = 10)
  (h_exists_5 : ∃ j : Fin 15, announcements j = 5) :
  ∃ k : Fin 15, announcements k = 5 ∧ numbers k = 0 := by
sorry

end NUMINAMATH_CALUDE_find_number_from_announcements_l1942_194259


namespace NUMINAMATH_CALUDE_range_of_a_l1942_194282

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4*x - 3) ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- State the theorem
theorem range_of_a : 
  (∀ x a : ℝ, q x a → p x) ∧ 
  (∃ x : ℝ, p x ∧ ∀ a : ℝ, ¬(q x a)) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∃ x : ℝ, q x a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1942_194282


namespace NUMINAMATH_CALUDE_square_area_difference_l1942_194278

-- Define the sides of the squares
def a : ℕ := 12
def b : ℕ := 9
def c : ℕ := 7
def d : ℕ := 3

-- Define the theorem
theorem square_area_difference : a ^ 2 + c ^ 2 - b ^ 2 - d ^ 2 = 103 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l1942_194278


namespace NUMINAMATH_CALUDE_school_garden_flowers_l1942_194271

theorem school_garden_flowers (total : ℕ) (yellow : ℕ) : 
  total = 96 → yellow = 12 → ∃ (green : ℕ), 
    green + 3 * green + (total / 2) + yellow = total ∧ green = 9 := by
  sorry

end NUMINAMATH_CALUDE_school_garden_flowers_l1942_194271


namespace NUMINAMATH_CALUDE_rose_difference_after_changes_l1942_194293

/-- Calculates the difference in red roses between two people after changes -/
def rose_difference (santiago_initial : ℕ) (garrett_initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  (santiago_initial - given_away + received) - (garrett_initial - given_away + received)

theorem rose_difference_after_changes :
  rose_difference 58 24 10 5 = 34 := by
  sorry

end NUMINAMATH_CALUDE_rose_difference_after_changes_l1942_194293


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1942_194247

/-- Two lines intersect in the fourth quadrant if and only if m > -2/3 -/
theorem intersection_in_fourth_quadrant (m : ℝ) :
  (∃ x y : ℝ, 3 * x + 2 * y - 2 * m - 1 = 0 ∧
               2 * x + 4 * y - m = 0 ∧
               x > 0 ∧ y < 0) ↔
  m > -2/3 := by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1942_194247


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l1942_194279

theorem sqrt_product_equals_sqrt_of_product : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l1942_194279


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l1942_194225

-- Define the initial and final counts of roses and orchids
def initial_roses : ℕ := 7
def initial_orchids : ℕ := 12
def final_roses : ℕ := 11
def final_orchids : ℕ := 20

-- Theorem to prove
theorem orchid_rose_difference :
  final_orchids - final_roses = 9 :=
by sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l1942_194225


namespace NUMINAMATH_CALUDE_train_length_l1942_194254

theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 700) :
  ∃ (train_length : ℝ),
    train_length = 600 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l1942_194254


namespace NUMINAMATH_CALUDE_cow_profit_calculation_l1942_194246

def cow_profit (purchase_price : ℕ) (daily_food_cost : ℕ) (vaccination_cost : ℕ) (days : ℕ) (selling_price : ℕ) : ℕ :=
  selling_price - (purchase_price + daily_food_cost * days + vaccination_cost)

theorem cow_profit_calculation :
  cow_profit 600 20 500 40 2500 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cow_profit_calculation_l1942_194246


namespace NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l1942_194280

theorem cinnamon_swirls_distribution (total_pieces : Real) (num_people : Real) (jane_pieces : Real) : 
  total_pieces = 12.0 → num_people = 3.0 → jane_pieces = total_pieces / num_people → jane_pieces = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l1942_194280


namespace NUMINAMATH_CALUDE_repeated_number_divisible_by_91_l1942_194268

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundred_nonzero : hundreds ≠ 0
  digit_bounds : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Represents the six-digit number formed by repeating a three-digit number -/
def repeated_number (n : ThreeDigitNumber) : Nat :=
  1000000 * n.hundreds + 100000 * n.tens + 10000 * n.ones +
  1000 * n.hundreds + 100 * n.tens + 10 * n.ones

/-- Theorem stating that the repeated number is divisible by 91 -/
theorem repeated_number_divisible_by_91 (n : ThreeDigitNumber) :
  (repeated_number n) % 91 = 0 := by
  sorry

end NUMINAMATH_CALUDE_repeated_number_divisible_by_91_l1942_194268


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l1942_194266

def is_single_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    is_single_digit d ∧
    is_single_digit e ∧
    is_prime d ∧
    is_prime e ∧
    is_prime (10 * d + e) ∧
    n = d * e * (10 * d + e) ∧
    (∀ (m : ℕ), m = d' * e' * (10 * d' + e') →
      is_single_digit d' →
      is_single_digit e' →
      is_prime d' →
      is_prime e' →
      is_prime (10 * d' + e') →
      m ≤ n) ∧
    sum_of_digits n = 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l1942_194266


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1942_194203

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SequenceCondition a) : 
  a 6 = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1942_194203


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l1942_194262

theorem triangle_inequality_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l1942_194262


namespace NUMINAMATH_CALUDE_equation_solution_l1942_194219

theorem equation_solution (a b c d p : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |p| = 3) :
  ∃! x : ℝ, (a + b) * x^2 + 4 * c * d * x + p^2 = x ∧ x = -3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1942_194219


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l1942_194206

/-- Given a principal amount and an interest rate, if increasing the rate by 1%
    results in an additional interest of 63 over 3 years, then the principal amount is 2100. -/
theorem principal_amount_calculation (P R : ℝ) (h : P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 63) :
  P = 2100 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l1942_194206


namespace NUMINAMATH_CALUDE_competition_result_l1942_194201

structure Athlete where
  longJump : ℝ
  tripleJump : ℝ
  highJump : ℝ

def totalDistance (a : Athlete) : ℝ :=
  a.longJump + a.tripleJump + a.highJump

def isWinner (a : Athlete) : Prop :=
  totalDistance a = 22 * 3

theorem competition_result (x : ℝ) :
  let athlete1 := Athlete.mk x 30 7
  let athlete2 := Athlete.mk 24 34 8
  isWinner athlete2 ∧ ¬∃y, y = x ∧ isWinner (Athlete.mk y 30 7) := by
  sorry

end NUMINAMATH_CALUDE_competition_result_l1942_194201


namespace NUMINAMATH_CALUDE_range_of_a_l1942_194275

/-- Proposition p: There exists x ∈ ℝ such that x^2 - 2x + a^2 = 0 -/
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

/-- Proposition q: For all x ∈ ℝ, ax^2 - ax + 1 > 0 -/
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

/-- The range of a given p ∧ (¬q) is true -/
theorem range_of_a (a : ℝ) (h : p a ∧ ¬(q a)) : -1 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1942_194275


namespace NUMINAMATH_CALUDE_f_derivative_lower_bound_and_range_l1942_194291

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_derivative_lower_bound_and_range :
  (∀ x : ℝ, (deriv f) x ≥ 2) ∧
  (∀ x : ℝ, x ≥ 0 → f (x^2 - 1) < Real.exp 1 - Real.exp (-1) → 0 ≤ x ∧ x < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_lower_bound_and_range_l1942_194291


namespace NUMINAMATH_CALUDE_root_range_implies_k_range_l1942_194237

theorem root_range_implies_k_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1) :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_k_range_l1942_194237


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1942_194235

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f x a ≥ 2*a - 1} = {a : ℝ | a ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1942_194235


namespace NUMINAMATH_CALUDE_largest_x_value_l1942_194257

theorem largest_x_value (x : ℝ) : 
  (3 * x / 7 + 2 / (9 * x) = 1) → 
  x ≤ (63 + Real.sqrt 2457) / 54 ∧ 
  ∃ y : ℝ, (3 * y / 7 + 2 / (9 * y) = 1) ∧ y = (63 + Real.sqrt 2457) / 54 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l1942_194257


namespace NUMINAMATH_CALUDE_train_speed_problem_l1942_194222

theorem train_speed_problem (v : ℝ) : 
  v > 0 → -- The speed of the second train is positive
  (∃ t : ℝ, t > 0 ∧ -- There exists a positive time t
    16 * t + v * t = 444 ∧ -- Total distance traveled equals the distance between stations
    v * t = 16 * t + 60) -- The second train travels 60 km more than the first
  → v = 21 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1942_194222


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1942_194285

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

-- Define the condition x = 2
def condition (x : ℝ) : Prop := x = 2

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ x : ℝ, quadratic_equation x ↔ (x = 2 ∨ x = 3)) →
  (∀ x : ℝ, condition x → quadratic_equation x) ∧
  ¬(∀ x : ℝ, quadratic_equation x → condition x) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1942_194285


namespace NUMINAMATH_CALUDE_pentagonal_prism_volume_l1942_194287

/-- The volume of a pentagonal prism with specific dimensions -/
theorem pentagonal_prism_volume : 
  let square_side : ℝ := 2
  let prism_height : ℝ := 2
  let triangle_leg : ℝ := 1
  let base_area : ℝ := square_side ^ 2 - (1 / 2 * triangle_leg * triangle_leg)
  let volume : ℝ := base_area * prism_height
  volume = 7 := by sorry

end NUMINAMATH_CALUDE_pentagonal_prism_volume_l1942_194287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1942_194269

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def increasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ) (h : d > 0) (h_arith : arithmeticSequence a d) :
  increasingSequence a ∧
  increasingSequence (fun n ↦ a n + 3 * n * d) ∧
  (¬ ∀ d, arithmeticSequence a d → increasingSequence (fun n ↦ n * a n)) ∧
  (¬ ∀ d, arithmeticSequence a d → increasingSequence (fun n ↦ a n / n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1942_194269


namespace NUMINAMATH_CALUDE_return_flight_speed_l1942_194210

/-- Proves that given a round trip flight with specified conditions, the return flight speed is 500 mph -/
theorem return_flight_speed 
  (total_distance : ℝ) 
  (outbound_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 3000) 
  (h2 : outbound_speed = 300) 
  (h3 : total_time = 8) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / outbound_speed) = 500 := by
  sorry

#check return_flight_speed

end NUMINAMATH_CALUDE_return_flight_speed_l1942_194210


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1942_194231

theorem quadratic_equation_solution (x : ℝ) : x^2 + 2*x - 8 = 0 ↔ x = -4 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1942_194231


namespace NUMINAMATH_CALUDE_sum_of_factors_36_l1942_194244

/-- The sum of positive factors of 36 is 91. -/
theorem sum_of_factors_36 : (Finset.filter (· ∣ 36) (Finset.range 37)).sum id = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_36_l1942_194244


namespace NUMINAMATH_CALUDE_initial_flour_amount_l1942_194260

theorem initial_flour_amount (initial : ℕ) : 
  initial + 2 = 10 → initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_flour_amount_l1942_194260


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l1942_194223

/-- The number of digits in a and b -/
def n : ℕ := 1984

/-- The integer a consisting of n nines in base 10 -/
def a : ℕ := (10^n - 1) / 9

/-- The integer b consisting of n fives in base 10 -/
def b : ℕ := (5 * (10^n - 1)) / 9

/-- Function to calculate the sum of digits of a natural number in base 10 -/
def sumOfDigits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sumOfDigits (k / 10)

/-- Theorem stating that the sum of digits of 9ab is 27779 -/
theorem sum_of_digits_9ab : sumOfDigits (9 * a * b) = 27779 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l1942_194223


namespace NUMINAMATH_CALUDE_pages_difference_l1942_194251

/-- The number of pages Person A reads per day -/
def pages_per_day_A : ℕ := 8

/-- The number of pages Person B reads per day (when not resting) -/
def pages_per_day_B : ℕ := 13

/-- The total number of days -/
def total_days : ℕ := 7

/-- The number of days in Person B's reading cycle -/
def cycle_days : ℕ := 3

/-- The number of days Person B reads in a cycle -/
def reading_days_per_cycle : ℕ := 2

/-- Calculate the number of pages read by Person A -/
def pages_read_A : ℕ := total_days * pages_per_day_A

/-- Calculate the number of full cycles in the total days -/
def full_cycles : ℕ := total_days / cycle_days

/-- Calculate the number of days Person B reads -/
def reading_days_B : ℕ := full_cycles * reading_days_per_cycle + (total_days % cycle_days)

/-- Calculate the number of pages read by Person B -/
def pages_read_B : ℕ := reading_days_B * pages_per_day_B

/-- The theorem to prove -/
theorem pages_difference : pages_read_B - pages_read_A = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l1942_194251


namespace NUMINAMATH_CALUDE_grid_arrangement_theorem_l1942_194295

/-- A type representing the grid arrangement of digits -/
def GridArrangement := Fin 8 → Fin 9

/-- Function to check if a three-digit number is a multiple of k -/
def isMultipleOfK (n : ℕ) (k : ℕ) : Prop :=
  n % k = 0

/-- Function to extract a three-digit number from the grid -/
def extractNumber (g : GridArrangement) (start : Fin 8) : ℕ :=
  100 * (g start).val + 10 * (g ((start + 2) % 8)).val + (g ((start + 4) % 8)).val

/-- Predicate to check if all four numbers in the grid are multiples of k -/
def allMultiplesOfK (g : GridArrangement) (k : ℕ) : Prop :=
  ∀ i : Fin 4, isMultipleOfK (extractNumber g (2 * i)) k

/-- Predicate to check if a grid arrangement is valid (uses all digits 1 to 8 once) -/
def isValidArrangement (g : GridArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → g i ≠ g j

/-- The main theorem stating for which values of k a valid arrangement exists -/
theorem grid_arrangement_theorem :
  ∀ k : ℕ, 2 ≤ k → k ≤ 6 →
    (∃ g : GridArrangement, isValidArrangement g ∧ allMultiplesOfK g k) ↔ (k = 2 ∨ k = 3) :=
sorry

end NUMINAMATH_CALUDE_grid_arrangement_theorem_l1942_194295


namespace NUMINAMATH_CALUDE_range_of_a_l1942_194276

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 ≤ 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, ¬(p x a) → ¬(q x))
  (h3 : ∃ x, ¬(p x a) ∧ q x) :
  0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1942_194276


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l1942_194277

theorem paint_usage_fraction (total_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1 / 9 →
  total_used = 104 →
  let remaining_paint := total_paint - first_week_fraction * total_paint
  let second_week_usage := total_used - first_week_fraction * total_paint
  second_week_usage / remaining_paint = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l1942_194277


namespace NUMINAMATH_CALUDE_marble_probability_difference_l1942_194217

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 2000

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 2000

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1)) / (total_marbles * (total_marbles - 1))

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (2 * red_marbles * black_marbles) / (total_marbles * (total_marbles - 1))

/-- The theorem stating the absolute difference between P_s and P_d -/
theorem marble_probability_difference : |P_s - P_d| = 1 / 3999 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l1942_194217


namespace NUMINAMATH_CALUDE_inequalities_solution_l1942_194232

theorem inequalities_solution :
  (∀ x : ℝ, x * (9 - x) > 0 ↔ 0 < x ∧ x < 9) ∧
  (∀ x : ℝ, 16 - x^2 ≤ 0 ↔ x ≤ -4 ∨ x ≥ 4) := by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l1942_194232


namespace NUMINAMATH_CALUDE_quadrilateral_max_area_and_angles_l1942_194253

/-- A quadrilateral with two sides of length 3 and two sides of length 4 -/
structure Quadrilateral :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)
  (side1_eq_3 : side1 = 3)
  (side2_eq_4 : side2 = 4)
  (side3_eq_3 : side3 = 3)
  (side4_eq_4 : side4 = 4)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The angles of a quadrilateral -/
def angles (q : Quadrilateral) : Fin 4 → ℝ := sorry

/-- The sum of two opposite angles in a quadrilateral -/
def opposite_angles_sum (q : Quadrilateral) : ℝ := 
  angles q 0 + angles q 2

theorem quadrilateral_max_area_and_angles (q : Quadrilateral) : 
  (∀ q' : Quadrilateral, area q' ≤ area q) → 
  (area q = 12 ∧ opposite_angles_sum q = 180) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_max_area_and_angles_l1942_194253


namespace NUMINAMATH_CALUDE_basketball_scoring_l1942_194288

/-- Basketball game scoring problem -/
theorem basketball_scoring
  (alex_points : ℕ)
  (sam_points : ℕ)
  (jon_points : ℕ)
  (jack_points : ℕ)
  (tom_points : ℕ)
  (h1 : jon_points = 2 * sam_points + 3)
  (h2 : sam_points = alex_points / 2)
  (h3 : alex_points = jack_points - 7)
  (h4 : jack_points = jon_points + 5)
  (h5 : tom_points = jon_points + jack_points - 4)
  (h6 : alex_points = 18) :
  jon_points + jack_points + tom_points + sam_points + alex_points = 115 := by
sorry

end NUMINAMATH_CALUDE_basketball_scoring_l1942_194288


namespace NUMINAMATH_CALUDE_compound_interest_with_contributions_l1942_194227

theorem compound_interest_with_contributions
  (initial_amount : ℝ)
  (interest_rate : ℝ)
  (annual_contribution : ℝ)
  (years : ℕ)
  (h1 : initial_amount = 76800)
  (h2 : interest_rate = 0.125)
  (h3 : annual_contribution = 5000)
  (h4 : years = 2) :
  let amount_after_first_year := initial_amount * (1 + interest_rate)
  let total_after_first_year := amount_after_first_year + annual_contribution
  let amount_after_second_year := total_after_first_year * (1 + interest_rate)
  let final_amount := amount_after_second_year + annual_contribution
  final_amount = 107825 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_with_contributions_l1942_194227


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l1942_194211

theorem initial_markup_percentage (initial_price : ℝ) (additional_increase : ℝ) : 
  initial_price = 45 →
  additional_increase = 5 →
  initial_price + additional_increase = 2 * (initial_price - (initial_price - (initial_price / (1 + 8)))) →
  (initial_price - (initial_price / (1 + 8))) / (initial_price / (1 + 8)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l1942_194211


namespace NUMINAMATH_CALUDE_polar_curve_length_l1942_194234

noncomputable def curve_length (ρ : Real → Real) (φ₁ φ₂ : Real) : Real :=
  ∫ x in φ₁..φ₂, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem polar_curve_length :
  let ρ : Real → Real := fun φ ↦ 2 * (1 - Real.cos φ)
  curve_length ρ (-Real.pi) (-Real.pi/2) = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_curve_length_l1942_194234


namespace NUMINAMATH_CALUDE_sum_in_base_6_l1942_194286

/-- Converts a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The sum of 453₆, 436₆, and 42₆ in base 6 is 1415₆ --/
theorem sum_in_base_6 :
  to_base_6 (to_base_10 [3, 5, 4] + to_base_10 [6, 3, 4] + to_base_10 [2, 4]) = [5, 1, 4, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base_6_l1942_194286


namespace NUMINAMATH_CALUDE_square_ending_four_identical_digits_l1942_194215

theorem square_ending_four_identical_digits (n : ℕ) (d : ℕ) 
  (h1 : d ≤ 9) 
  (h2 : ∃ k : ℕ, n^2 = 10000 * k + d * 1111) : 
  d = 0 := by
sorry

end NUMINAMATH_CALUDE_square_ending_four_identical_digits_l1942_194215


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l1942_194209

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 3. -/
theorem sum_of_common_ratios_is_three
  (k p r : ℝ)
  (h_nonconstant : k ≠ 0)
  (h_different_ratios : p ≠ r)
  (h_condition : k * p^2 - k * r^2 = 3 * (k * p - k * r)) :
  p + r = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l1942_194209


namespace NUMINAMATH_CALUDE_percentage_in_quarters_calculation_l1942_194273

/-- Given a collection of coins, calculate the percentage of the total value that is in quarters. -/
def percentageInQuarters (dimes nickels quarters : ℕ) : ℚ :=
  let dimesValue : ℕ := dimes * 10
  let nickelsValue : ℕ := nickels * 5
  let quartersValue : ℕ := quarters * 25
  let totalValue : ℕ := dimesValue + nickelsValue + quartersValue
  (quartersValue : ℚ) / (totalValue : ℚ) * 100

theorem percentage_in_quarters_calculation :
  percentageInQuarters 70 40 30 = 750 / 1650 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_quarters_calculation_l1942_194273


namespace NUMINAMATH_CALUDE_same_root_value_l1942_194240

theorem same_root_value (a b c d : ℝ) (h : a ≠ c) :
  ∀ α : ℝ, (α^2 + a*α + b = 0 ∧ α^2 + c*α + d = 0) → α = (d - b) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_same_root_value_l1942_194240


namespace NUMINAMATH_CALUDE_total_weight_chromic_acid_sodium_hydroxide_l1942_194250

/-- The total weight of Chromic acid and Sodium hydroxide in a neutralization reaction -/
theorem total_weight_chromic_acid_sodium_hydroxide 
  (moles_chromic_acid : ℝ) 
  (moles_sodium_hydroxide : ℝ) 
  (molar_mass_chromic_acid : ℝ) 
  (molar_mass_sodium_hydroxide : ℝ) : 
  moles_chromic_acid = 17.3 →
  moles_sodium_hydroxide = 8.5 →
  molar_mass_chromic_acid = 118.02 →
  molar_mass_sodium_hydroxide = 40.00 →
  moles_chromic_acid * molar_mass_chromic_acid + 
  moles_sodium_hydroxide * molar_mass_sodium_hydroxide = 2381.746 := by
  sorry

#check total_weight_chromic_acid_sodium_hydroxide

end NUMINAMATH_CALUDE_total_weight_chromic_acid_sodium_hydroxide_l1942_194250


namespace NUMINAMATH_CALUDE_sum_of_possible_p_values_l1942_194265

theorem sum_of_possible_p_values : ∃ (S : Finset Nat), 
  (∀ p ∈ S, ∃ q : Nat, 
    Nat.Prime p ∧ 
    p > 0 ∧ 
    q > 0 ∧ 
    p ∣ (q - 1) ∧ 
    (p + q) ∣ (p^2 + 2020*q^2)) ∧
  (∀ p : Nat, 
    (∃ q : Nat, 
      Nat.Prime p ∧ 
      p > 0 ∧ 
      q > 0 ∧ 
      p ∣ (q - 1) ∧ 
      (p + q) ∣ (p^2 + 2020*q^2)) → 
    p ∈ S) ∧
  S.sum id = 35 := by
sorry


end NUMINAMATH_CALUDE_sum_of_possible_p_values_l1942_194265


namespace NUMINAMATH_CALUDE_lottery_theorem_l1942_194261

-- Define the lottery setup
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Define the probability of drawing a red ball first given a white ball second
def prob_red_given_white : ℚ := 5/11

-- Define the probabilities for the distribution of red balls drawn
def prob_zero_red : ℚ := 27/125
def prob_one_red : ℚ := 549/1000
def prob_two_red : ℚ := 47/200

-- Define the expected number of red balls drawn
def expected_red_balls : ℚ := 1019/1000

-- Theorem statement
theorem lottery_theorem :
  (total_balls = red_balls + white_balls) →
  (prob_red_given_white = 5/11) ∧
  (prob_zero_red + prob_one_red + prob_two_red = 1) ∧
  (expected_red_balls = 0 * prob_zero_red + 1 * prob_one_red + 2 * prob_two_red) :=
by sorry

end NUMINAMATH_CALUDE_lottery_theorem_l1942_194261


namespace NUMINAMATH_CALUDE_f_composition_one_sixteenth_l1942_194274

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 3^x

theorem f_composition_one_sixteenth : f (f (1/16)) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_sixteenth_l1942_194274


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1942_194296

/-- Represents the seven points on the circle -/
inductive CirclePoint
  | one | two | three | four | five | six | seven

/-- Determines if a CirclePoint is prime -/
def isPrime : CirclePoint → Bool
  | CirclePoint.two => true
  | CirclePoint.three => true
  | CirclePoint.five => true
  | CirclePoint.seven => true
  | _ => false

/-- Determines if a CirclePoint is composite -/
def isComposite : CirclePoint → Bool
  | CirclePoint.four => true
  | CirclePoint.six => true
  | _ => false

/-- Moves the bug according to the jumping rule -/
def move (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.one => CirclePoint.two
  | CirclePoint.two => CirclePoint.three
  | CirclePoint.three => CirclePoint.four
  | CirclePoint.four => CirclePoint.seven
  | CirclePoint.five => CirclePoint.six
  | CirclePoint.six => CirclePoint.two
  | CirclePoint.seven => CirclePoint.one

/-- Performs n jumps starting from a given point -/
def jumpN (start : CirclePoint) (n : Nat) : CirclePoint :=
  match n with
  | 0 => start
  | n + 1 => move (jumpN start n)

theorem bug_position_after_2023_jumps :
  jumpN CirclePoint.seven 2023 = CirclePoint.two := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1942_194296


namespace NUMINAMATH_CALUDE_students_not_enrolled_l1942_194258

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 69) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l1942_194258


namespace NUMINAMATH_CALUDE_ammonium_hydroxide_formation_l1942_194221

-- Define the chemicals involved in the reaction
inductive Chemical
| NH4Cl
| NaOH
| NH4OH
| NaCl

-- Define a function to represent the reaction
def reaction (nh4cl : ℚ) (naoh : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  (nh4cl - min nh4cl naoh, naoh - min nh4cl naoh, min nh4cl naoh, min nh4cl naoh)

-- Theorem stating the result of the reaction
theorem ammonium_hydroxide_formation 
  (nh4cl_moles naoh_moles : ℚ) 
  (h1 : nh4cl_moles = 1) 
  (h2 : naoh_moles = 1) : 
  (reaction nh4cl_moles naoh_moles).2.1 = 1 := by
  sorry

-- Note: The theorem states that the third component of the reaction result
-- (which represents NH4OH) is equal to 1 when both input moles are 1.

end NUMINAMATH_CALUDE_ammonium_hydroxide_formation_l1942_194221


namespace NUMINAMATH_CALUDE_bird_families_count_l1942_194255

theorem bird_families_count (africa asia left : ℕ) (h1 : africa = 23) (h2 : asia = 37) (h3 : left = 25) :
  africa + asia + left = 85 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_count_l1942_194255


namespace NUMINAMATH_CALUDE_project_selection_count_l1942_194294

def num_key_projects : ℕ := 4
def num_general_projects : ℕ := 6
def projects_to_select : ℕ := 3

def select_projects (n k : ℕ) : ℕ := Nat.choose n k

theorem project_selection_count : 
  (select_projects (num_general_projects - 1) (projects_to_select - 1) * 
   select_projects (num_key_projects - 1) (projects_to_select - 1)) +
  (select_projects (num_key_projects - 1) 1 * 
   select_projects (num_general_projects - 1) 1) = 45 := by sorry

end NUMINAMATH_CALUDE_project_selection_count_l1942_194294


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_37_l1942_194239

def polynomial (x : ℝ) : ℝ := -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2) - 2 * (x^6 - 5)

theorem sum_of_coefficients_is_37 : 
  (polynomial 1) = 37 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_37_l1942_194239


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1942_194216

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 5)*x - k + 8 > 0) ↔ k > -1 ∧ k < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1942_194216


namespace NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l1942_194205

/-- The number of people that can fit in one teacup -/
def people_per_teacup : ℕ := 9

/-- The number of teacups on the ride -/
def number_of_teacups : ℕ := 7

/-- The total number of people that can ride at a time -/
def total_riders : ℕ := people_per_teacup * number_of_teacups

theorem twirly_tea_cups_capacity :
  total_riders = 63 := by sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l1942_194205


namespace NUMINAMATH_CALUDE_first_fibonacci_exceeding_3000_l1942_194236

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem first_fibonacci_exceeding_3000 :
  (∀ k < 19, fibonacci k ≤ 3000) ∧ fibonacci 19 > 3000 := by sorry

end NUMINAMATH_CALUDE_first_fibonacci_exceeding_3000_l1942_194236


namespace NUMINAMATH_CALUDE_fraction_simplification_l1942_194207

theorem fraction_simplification :
  (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1942_194207


namespace NUMINAMATH_CALUDE_incenter_is_circumcenter_of_A₁B₁C₁_l1942_194263

-- Define a structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_acute_angled (t : Triangle) : Prop := sorry
def is_non_equilateral (t : Triangle) : Prop := sorry

-- Define the circumradius
def circumradius (t : Triangle) : ℝ := sorry

-- Define the heights of the triangle
def height_A (t : Triangle) : ℝ × ℝ := sorry
def height_B (t : Triangle) : ℝ × ℝ := sorry
def height_C (t : Triangle) : ℝ × ℝ := sorry

-- Define points A₁, B₁, C₁ on the heights
def A₁ (t : Triangle) : ℝ × ℝ := sorry
def B₁ (t : Triangle) : ℝ × ℝ := sorry
def C₁ (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- The main theorem
theorem incenter_is_circumcenter_of_A₁B₁C₁ (t : Triangle) 
  (h_acute : is_acute_angled t) 
  (h_non_equilateral : is_non_equilateral t) 
  (h_A₁ : A₁ t = height_A t + (0, circumradius t))
  (h_B₁ : B₁ t = height_B t + (0, circumradius t))
  (h_C₁ : C₁ t = height_C t + (0, circumradius t)) :
  incenter t = circumcenter { A := A₁ t, B := B₁ t, C := C₁ t } := by
  sorry

end NUMINAMATH_CALUDE_incenter_is_circumcenter_of_A₁B₁C₁_l1942_194263


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l1942_194283

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_seq : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l1942_194283


namespace NUMINAMATH_CALUDE_cylinder_views_l1942_194238

-- Define the cylinder and its orientation
structure Cylinder where
  upright : Bool
  on_horizontal_plane : Bool

-- Define the possible view shapes
inductive ViewShape
  | Rectangle
  | Circle

-- Define the function to get the view of the cylinder
def get_cylinder_view (c : Cylinder) (view : String) : ViewShape :=
  match view with
  | "front" => ViewShape.Rectangle
  | "side" => ViewShape.Rectangle
  | "top" => ViewShape.Circle
  | _ => ViewShape.Rectangle  -- Default case, though not needed for our problem

-- Theorem statement
theorem cylinder_views (c : Cylinder) 
  (h1 : c.upright = true) 
  (h2 : c.on_horizontal_plane = true) : 
  (get_cylinder_view c "front" = ViewShape.Rectangle) ∧ 
  (get_cylinder_view c "side" = ViewShape.Rectangle) ∧ 
  (get_cylinder_view c "top" = ViewShape.Circle) := by
  sorry


end NUMINAMATH_CALUDE_cylinder_views_l1942_194238


namespace NUMINAMATH_CALUDE_sarahs_brother_apples_l1942_194224

theorem sarahs_brother_apples (sarah_apples : ℝ) (ratio : ℝ) (brother_apples : ℝ) : 
  sarah_apples = 45.0 →
  sarah_apples = ratio * brother_apples →
  ratio = 5 →
  brother_apples = 9.0 := by
sorry

end NUMINAMATH_CALUDE_sarahs_brother_apples_l1942_194224


namespace NUMINAMATH_CALUDE_vertical_shift_graph_l1942_194204

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define vertical shift operation
def verticalShift (f : RealFunction) (k : ℝ) : RealFunction :=
  λ x => f x + k

-- Theorem statement
theorem vertical_shift_graph (f : RealFunction) (k : ℝ) :
  ∀ x y, y = f x ↔ (y + k) = (verticalShift f k) x :=
sorry

end NUMINAMATH_CALUDE_vertical_shift_graph_l1942_194204


namespace NUMINAMATH_CALUDE_residential_ratio_is_half_l1942_194230

/-- Represents a building with residential, office, and restaurant units. -/
structure Building where
  total_units : ℕ
  restaurant_units : ℕ
  office_units : ℕ
  residential_units : ℕ

/-- The ratio of residential units to total units in a building. -/
def residential_ratio (b : Building) : ℚ :=
  b.residential_units / b.total_units

/-- Theorem stating the residential ratio for a specific building configuration. -/
theorem residential_ratio_is_half (b : Building) 
    (h1 : b.total_units = 300)
    (h2 : b.restaurant_units = 75)
    (h3 : b.office_units = b.restaurant_units)
    (h4 : b.residential_units = b.total_units - (b.restaurant_units + b.office_units)) :
    residential_ratio b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_residential_ratio_is_half_l1942_194230


namespace NUMINAMATH_CALUDE_asterisk_replacement_l1942_194290

theorem asterisk_replacement : ∃ x : ℚ, (x / 18) * (36 / 72) = 1 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1942_194290


namespace NUMINAMATH_CALUDE_x_value_theorem_l1942_194252

theorem x_value_theorem (x n : ℕ) :
  x = 2^n - 32 ∧
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧ 
    (∀ r : ℕ, Prime r → r ∣ x ↔ r = 3 ∨ r = p ∨ r = q)) →
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l1942_194252


namespace NUMINAMATH_CALUDE_book_selection_problem_l1942_194297

theorem book_selection_problem (total_books : ℕ) (novels : ℕ) (to_choose : ℕ) :
  total_books = 15 →
  novels = 5 →
  to_choose = 3 →
  (Nat.choose total_books to_choose) - (Nat.choose (total_books - novels) to_choose) = 335 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_problem_l1942_194297


namespace NUMINAMATH_CALUDE_square_x_plus_2y_l1942_194267

theorem square_x_plus_2y (x y : ℝ) 
  (h1 : x * (x + y) = 40) 
  (h2 : y * (x + y) = 90) : 
  (x + 2*y)^2 = 310 + 8100/130 := by
  sorry

end NUMINAMATH_CALUDE_square_x_plus_2y_l1942_194267


namespace NUMINAMATH_CALUDE_min_value_ab_l1942_194220

theorem min_value_ab (a b : ℝ) (h : (4 / a) + (1 / b) = Real.sqrt (a * b)) : 
  ∀ x y : ℝ, ((4 / x) + (1 / y) = Real.sqrt (x * y)) → (a * b) ≤ (x * y) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l1942_194220


namespace NUMINAMATH_CALUDE_compound_interest_rate_equation_l1942_194245

/-- Proves that the given compound interest scenario results in the specified equation for the interest rate. -/
theorem compound_interest_rate_equation (P r : ℝ) 
  (h1 : P * (1 + r)^3 = 310) 
  (h2 : P * (1 + r)^8 = 410) : 
  (1 + r)^5 = 410/310 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_equation_l1942_194245


namespace NUMINAMATH_CALUDE_one_time_cost_correct_l1942_194200

/-- The one-time product cost for editing and printing --/
def one_time_cost : ℝ := 56430

/-- The variable cost per book --/
def variable_cost : ℝ := 8.25

/-- The selling price per book --/
def selling_price : ℝ := 21.75

/-- The number of books at the break-even point --/
def break_even_books : ℕ := 4180

/-- Theorem stating that the one-time cost is correct given the conditions --/
theorem one_time_cost_correct :
  one_time_cost = (selling_price - variable_cost) * break_even_books :=
by sorry

end NUMINAMATH_CALUDE_one_time_cost_correct_l1942_194200


namespace NUMINAMATH_CALUDE_quadrant_restriction_l1942_194264

theorem quadrant_restriction (θ : Real) :
  1 + Real.sin θ * Real.sqrt (Real.sin θ * Real.sin θ) + 
  Real.cos θ * Real.sqrt (Real.cos θ * Real.cos θ) = 0 →
  (Real.sin θ > 0 ∧ Real.cos θ > 0) ∨ 
  (Real.sin θ > 0 ∧ Real.cos θ < 0) ∨ 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) → False := by
  sorry

end NUMINAMATH_CALUDE_quadrant_restriction_l1942_194264


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1942_194289

theorem smallest_number_divisible (n : ℕ) : n = 44398 ↔ 
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 12 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 30 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 48 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 74 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 100 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, (n + 2) = 12 * k₁ ∧ 
                         (n + 2) = 30 * k₂ ∧ 
                         (n + 2) = 48 * k₃ ∧ 
                         (n + 2) = 74 * k₄ ∧ 
                         (n + 2) = 100 * k₅) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1942_194289


namespace NUMINAMATH_CALUDE_decimal_89_to_binary_l1942_194214

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_89_to_binary :
  decimal_to_binary 89 = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_89_to_binary_l1942_194214


namespace NUMINAMATH_CALUDE_complex_2_minus_3i_in_fourth_quadrant_l1942_194213

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_2_minus_3i_in_fourth_quadrant :
  is_in_fourth_quadrant (2 - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_2_minus_3i_in_fourth_quadrant_l1942_194213


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_1_l1942_194233

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 1| + |x - a|

-- Statement 1
theorem solution_set_when_a_is_2 :
  let f2 := f 2
  {x : ℝ | f2 x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 8/3} := by sorry

-- Statement 2
theorem range_of_a_when_f_geq_1 :
  {a : ℝ | a > 0 ∧ ∀ x, f a x ≥ 1} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_1_l1942_194233
