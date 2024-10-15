import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_hidden_numbers_l3781_378180

/-- Represents a standard six-sided die with faces numbered 1 through 6 -/
def Die := Fin 6

/-- The sum of all numbers on a standard die -/
def dieTotalSum : ℕ := 21

/-- The visible numbers on the seven sides of the stacked dice -/
def visibleNumbers : List ℕ := [2, 3, 4, 4, 5, 5, 6]

/-- The number of dice stacked -/
def numberOfDice : ℕ := 3

/-- Theorem stating that the sum of numbers not visible on the stacked dice is 34 -/
theorem sum_of_hidden_numbers :
  (numberOfDice * dieTotalSum) - (visibleNumbers.sum) = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_hidden_numbers_l3781_378180


namespace NUMINAMATH_CALUDE_correct_remaining_money_l3781_378136

/-- Calculates the remaining money after shopping --/
def remaining_money (initial_amount : ℕ) (banana_price : ℕ) (banana_quantity : ℕ) 
  (pear_price : ℕ) (asparagus_price : ℕ) (chicken_price : ℕ) : ℕ :=
  initial_amount - (banana_price * banana_quantity + pear_price + asparagus_price + chicken_price)

/-- Proves that the remaining money is correct given the initial amount and purchases --/
theorem correct_remaining_money :
  remaining_money 55 4 2 2 6 11 = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_money_l3781_378136


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l3781_378179

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that g satisfies the given conditions -/
def SatisfiesConditions (g : ThirdDegreePolynomial) : Prop :=
  ∀ x : ℝ, x ∈ ({-1, 0, 2, 4, 5, 8} : Set ℝ) → |g x| = 10

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : SatisfiesConditions g) : |g 3| = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l3781_378179


namespace NUMINAMATH_CALUDE_comic_book_collections_l3781_378159

def kymbrea_initial : ℕ := 50
def kymbrea_rate : ℕ := 1
def lashawn_initial : ℕ := 20
def lashawn_rate : ℕ := 7
def months : ℕ := 33

theorem comic_book_collections : 
  (lashawn_initial + lashawn_rate * months) = 
  3 * (kymbrea_initial + kymbrea_rate * months) :=
by sorry

end NUMINAMATH_CALUDE_comic_book_collections_l3781_378159


namespace NUMINAMATH_CALUDE_special_numbers_l3781_378140

/-- A two-digit number is equal to three times the product of its digits -/
def is_special_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ (a b : ℕ), n = 10 * a + b ∧ n = 3 * a * b

/-- The only two-digit numbers that are equal to three times the product of their digits are 15 and 24 -/
theorem special_numbers : ∀ n : ℕ, is_special_number n ↔ (n = 15 ∨ n = 24) :=
sorry

end NUMINAMATH_CALUDE_special_numbers_l3781_378140


namespace NUMINAMATH_CALUDE_min_tangent_length_l3781_378187

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the symmetry line
def symmetry_line (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

-- Define the tangent point
def tangent_point (a b : ℝ) : Prop := ∃ x y : ℝ, circle_C x y ∧ symmetry_line a b x y

-- Theorem statement
theorem min_tangent_length (a b : ℝ) : 
  tangent_point a b → 
  (∃ t : ℝ, t ≥ 0 ∧ 
    (∀ s : ℝ, s ≥ 0 → 
      (∃ x y : ℝ, circle_C x y ∧ (x - a)^2 + (y - b)^2 = s^2) → 
      t ≤ s) ∧ 
    t = 4) := 
sorry

end NUMINAMATH_CALUDE_min_tangent_length_l3781_378187


namespace NUMINAMATH_CALUDE_shortest_path_on_specific_floor_l3781_378113

/-- Represents a rectangular floor with a missing tile -/
structure RectangularFloor :=
  (width : Nat)
  (length : Nat)
  (missingTileX : Nat)
  (missingTileY : Nat)

/-- Calculates the shortest path length for a bug traversing the floor -/
def shortestPathLength (floor : RectangularFloor) : Nat :=
  floor.width + floor.length - Nat.gcd floor.width floor.length + 1

/-- Theorem stating the shortest path length for the given floor configuration -/
theorem shortest_path_on_specific_floor :
  let floor : RectangularFloor := {
    width := 12,
    length := 20,
    missingTileX := 6,
    missingTileY := 10
  }
  shortestPathLength floor = 29 := by
  sorry


end NUMINAMATH_CALUDE_shortest_path_on_specific_floor_l3781_378113


namespace NUMINAMATH_CALUDE_kate_change_l3781_378141

-- Define the prices of items
def gum_price : ℚ := 89 / 100
def chocolate_price : ℚ := 125 / 100
def chips_price : ℚ := 249 / 100

-- Define the sales tax rate
def sales_tax_rate : ℚ := 6 / 100

-- Define the amount Kate gave to the clerk
def amount_given : ℚ := 10

-- Theorem statement
theorem kate_change (gum : ℚ) (chocolate : ℚ) (chips : ℚ) (tax_rate : ℚ) (given : ℚ) :
  gum = gum_price →
  chocolate = chocolate_price →
  chips = chips_price →
  tax_rate = sales_tax_rate →
  given = amount_given →
  ∃ (change : ℚ), change = 509 / 100 ∧ 
    change = given - (gum + chocolate + chips + (gum + chocolate + chips) * tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_kate_change_l3781_378141


namespace NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l3781_378177

/-- A permutation of numbers from 1 to n satisfies the average condition if for any three indices
    i < j < k, the average of the i-th and k-th elements is not equal to the j-th element. -/
def satisfies_average_condition (n : ℕ) (perm : Fin n → ℕ) : Prop :=
  ∀ i j k : Fin n, i < j → j < k →
    (perm i + perm k) / 2 ≠ perm j

/-- For any positive integer n, there exists a permutation of the numbers 1 to n
    that satisfies the average condition. -/
theorem exists_permutation_satisfying_average_condition (n : ℕ+) :
  ∃ perm : Fin n → ℕ, Function.Injective perm ∧ Set.range perm = Finset.range n ∧
    satisfies_average_condition n perm :=
sorry

end NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l3781_378177


namespace NUMINAMATH_CALUDE_circles_are_separate_l3781_378153

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x + 3)^2 + (y - 2)^2 = 4

-- Define the centers and radii
def center₁ : ℝ × ℝ := (1, 0)
def center₂ : ℝ × ℝ := (-3, 2)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 2

-- Theorem statement
theorem circles_are_separate :
  let d := Real.sqrt ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2)
  d > radius₁ + radius₂ :=
by sorry

end NUMINAMATH_CALUDE_circles_are_separate_l3781_378153


namespace NUMINAMATH_CALUDE_newspaper_recycling_profit_l3781_378170

/-- Calculates the amount of money made from recycling stolen newspapers over a period of time. -/
def recycling_profit (weekday_paper_weight : ℚ) (sunday_paper_weight : ℚ) 
  (papers_per_day : ℕ) (num_weeks : ℕ) (recycling_rate : ℚ) : ℚ :=
  let weekly_weight := (6 * weekday_paper_weight + sunday_paper_weight) * papers_per_day
  let total_weight := weekly_weight * num_weeks
  let total_tons := total_weight / 2000
  total_tons * recycling_rate

/-- Theorem stating that under the given conditions, the profit from recycling stolen newspapers is $100. -/
theorem newspaper_recycling_profit :
  recycling_profit (8/16) (16/16) 250 10 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_recycling_profit_l3781_378170


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3781_378191

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3781_378191


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3781_378116

/-- Given two vectors a and b in a 2D plane with an angle of 120° between them,
    |a| = 1, and |b| = 3, prove that |a - b| = √13 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  (a.fst * b.fst + a.snd * b.snd = -3/2) →  -- Dot product for 120° angle
  (a.fst^2 + a.snd^2 = 1) →  -- |a| = 1
  (b.fst^2 + b.snd^2 = 9) →  -- |b| = 3
  ((a.fst - b.fst)^2 + (a.snd - b.snd)^2 = 13) :=
by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3781_378116


namespace NUMINAMATH_CALUDE_lucy_money_ratio_l3781_378181

theorem lucy_money_ratio : 
  ∀ (initial_amount spent remaining : ℚ),
    initial_amount = 30 →
    remaining = 15 →
    spent + remaining = initial_amount * (2/3) →
    spent / (initial_amount * (2/3)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_lucy_money_ratio_l3781_378181


namespace NUMINAMATH_CALUDE_equation_solution_l3781_378138

theorem equation_solution : 
  let equation := fun x : ℝ => 3 * x * (x - 2) = x - 2
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/3 ∧ equation x₁ ∧ equation x₂ ∧ 
  ∀ x : ℝ, equation x → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3781_378138


namespace NUMINAMATH_CALUDE_range_of_a_for_two_negative_roots_l3781_378121

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + |a|

-- Define the condition for two negative roots
def has_two_negative_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic a x₁ = 0 ∧ quadratic a x₂ = 0

-- State the theorem
theorem range_of_a_for_two_negative_roots :
  ∃ l u : ℝ, ∀ a : ℝ, has_two_negative_roots a ↔ l < a ∧ a < u :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_negative_roots_l3781_378121


namespace NUMINAMATH_CALUDE_simplify_radical_product_l3781_378152

theorem simplify_radical_product (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (27 * x) = 54 * x * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l3781_378152


namespace NUMINAMATH_CALUDE_decrement_value_proof_l3781_378106

theorem decrement_value_proof (n : ℕ) (original_mean updated_mean : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : updated_mean = 153) :
  (n : ℚ) * original_mean - n * updated_mean = n * 47 := by
  sorry

end NUMINAMATH_CALUDE_decrement_value_proof_l3781_378106


namespace NUMINAMATH_CALUDE_square_center_sum_l3781_378165

-- Define the square ABCD
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def square_conditions (s : Square) : Prop :=
  -- Square is in the first quadrant
  s.A.1 ≥ 0 ∧ s.A.2 ≥ 0 ∧
  s.B.1 ≥ 0 ∧ s.B.2 ≥ 0 ∧
  s.C.1 ≥ 0 ∧ s.C.2 ≥ 0 ∧
  s.D.1 ≥ 0 ∧ s.D.2 ≥ 0 ∧
  -- Points on the lines
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (4, 0) = s.A + t • (s.B - s.A)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (7, 0) = s.C + t • (s.D - s.C)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (9, 0) = s.B + t • (s.C - s.B)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (15, 0) = s.D + t • (s.A - s.D))

-- Theorem statement
theorem square_center_sum (s : Square) (h : square_conditions s) :
  (s.A.1 + s.B.1 + s.C.1 + s.D.1) / 4 + (s.A.2 + s.B.2 + s.C.2 + s.D.2) / 4 = 27 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_center_sum_l3781_378165


namespace NUMINAMATH_CALUDE_odd_function_value_l3781_378198

theorem odd_function_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^(2-m)
  (∀ x ∈ Set.Icc (-3-m) (m^2-m), f (-x) = -f x) →
  f m = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l3781_378198


namespace NUMINAMATH_CALUDE_hamburger_combinations_l3781_378139

theorem hamburger_combinations (num_condiments : ℕ) (num_patty_options : ℕ) :
  num_condiments = 10 →
  num_patty_options = 3 →
  (2^num_condiments) * num_patty_options = 3072 :=
by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l3781_378139


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3781_378185

theorem complex_product_magnitude : 
  Complex.abs ((-6 * Real.sqrt 3 + 6 * Complex.I) * (2 * Real.sqrt 2 - 2 * Complex.I)) = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3781_378185


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3781_378183

theorem max_value_trig_expression :
  ∀ x y z : ℝ,
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) *
  (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 4.5 ∧
  ∃ a b c : ℝ,
  (Real.sin (3 * a) + Real.sin (2 * b) + Real.sin c) *
  (Real.cos (3 * a) + Real.cos (2 * b) + Real.cos c) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3781_378183


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3781_378175

theorem complex_fraction_simplification :
  let z₁ : ℂ := 5 + 7 * I
  let z₂ : ℂ := 2 + 3 * I
  z₁ / z₂ = (31 : ℚ) / 13 - (1 : ℚ) / 13 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3781_378175


namespace NUMINAMATH_CALUDE_shaded_area_of_intersecting_diameters_l3781_378173

theorem shaded_area_of_intersecting_diameters (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = π / 3 → 2 * (θ / (2 * π)) * (π * r^2) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_intersecting_diameters_l3781_378173


namespace NUMINAMATH_CALUDE_tiling_remainder_l3781_378115

/-- Represents a tiling of an 8x1 board -/
structure Tiling :=
  (pieces : ℕ)
  (red_used : Bool)
  (blue_used : Bool)
  (green_used : Bool)

/-- The number of valid tilings of an 8x1 board -/
def M : ℕ := sorry

/-- Theorem stating the result of the tiling problem -/
theorem tiling_remainder : M % 1000 = 336 := by sorry

end NUMINAMATH_CALUDE_tiling_remainder_l3781_378115


namespace NUMINAMATH_CALUDE_football_team_progress_l3781_378104

theorem football_team_progress (yards_lost yards_gained : ℤ) : 
  yards_lost = 5 → yards_gained = 9 → yards_gained - yards_lost = 4 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l3781_378104


namespace NUMINAMATH_CALUDE_cube_root_function_l3781_378158

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, 
    prove that y = 2√3 when x = 8 -/
theorem cube_root_function (k : ℝ) :
  (∀ x : ℝ, x > 0 → k * x^(1/3) = 4 * Real.sqrt 3 → x = 64) →
  k * 8^(1/3) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_function_l3781_378158


namespace NUMINAMATH_CALUDE_movie_book_difference_l3781_378131

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 47

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 23

/-- Theorem: The difference between the number of movies and books in the 'crazy silly school' series is 24 -/
theorem movie_book_difference : num_movies - num_books = 24 := by
  sorry

end NUMINAMATH_CALUDE_movie_book_difference_l3781_378131


namespace NUMINAMATH_CALUDE_largest_n_unique_k_l3781_378144

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n = 112 ∧
  (∃! (k : ℤ), (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) ∧
  (∀ (m : ℕ), m > n → ¬∃! (k : ℤ), (8 : ℚ)/15 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 7/13) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_unique_k_l3781_378144


namespace NUMINAMATH_CALUDE_seven_lines_twenty_two_regions_l3781_378168

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  total_lines : ℕ
  parallel_lines : ℕ
  non_parallel_lines : ℕ
  no_concurrency : Prop
  no_other_parallel : Prop

/-- Calculate the number of regions formed by a given line configuration -/
def number_of_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- The theorem stating that the specific configuration of 7 lines creates 22 regions -/
theorem seven_lines_twenty_two_regions :
  ∀ (config : LineConfiguration),
    config.total_lines = 7 ∧
    config.parallel_lines = 2 ∧
    config.non_parallel_lines = 5 ∧
    config.no_concurrency ∧
    config.no_other_parallel →
    number_of_regions config = 22 :=
by sorry

end NUMINAMATH_CALUDE_seven_lines_twenty_two_regions_l3781_378168


namespace NUMINAMATH_CALUDE_perfect_square_existence_l3781_378178

theorem perfect_square_existence (k : ℕ+) :
  ∃ (n m : ℕ+), n * 2^(k : ℕ) - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_existence_l3781_378178


namespace NUMINAMATH_CALUDE_absolute_value_difference_l3781_378111

theorem absolute_value_difference : |(8-(3^2))| - |((4^2) - (6*3))| = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l3781_378111


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l3781_378192

structure Plane where
  -- Define a plane

structure Line where
  -- Define a line

def perpendicular (l : Line) (p : Plane) : Prop :=
  -- Define what it means for a line to be perpendicular to a plane
  sorry

def parallel (p1 p2 : Plane) : Prop :=
  -- Define what it means for two planes to be parallel
  sorry

def contains (p : Plane) (l : Line) : Prop :=
  -- Define what it means for a plane to contain a line
  sorry

def perpendicular_lines (l1 l2 : Line) : Prop :=
  -- Define what it means for two lines to be perpendicular
  sorry

theorem perpendicular_line_parallel_planes 
  (m : Line) (n : Line) (α β : Plane) :
  perpendicular m α → contains β n → parallel α β → perpendicular_lines m n :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l3781_378192


namespace NUMINAMATH_CALUDE_poor_people_distribution_l3781_378128

theorem poor_people_distribution (x : ℕ) : 
  (120 / (x - 10) - 120 / x = 120 / x - 120 / (x + 20)) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_poor_people_distribution_l3781_378128


namespace NUMINAMATH_CALUDE_curve_C_and_m_range_l3781_378155

/-- The curve C defined by the arithmetic sequence property -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) = 4}

/-- The line l₁ that intersects C -/
def l₁ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               x - y + m = 0}

/-- Predicate for the obtuse angle condition -/
def isObtuseAngle (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ * x₂ + y₁ * y₂ < 0

/-- The main theorem stating the properties of C and the range of m -/
theorem curve_C_and_m_range :
  ∃ (a b : ℝ),
    (a = 2 ∧ b = Real.sqrt 3) ∧
    (C = {p : ℝ × ℝ | let (x, y) := p
                      x^2 / a^2 + y^2 / b^2 = 1}) ∧
    (∀ m : ℝ,
      (∃ M N : ℝ × ℝ, M ∈ C ∧ N ∈ C ∧ M ∈ l₁ m ∧ N ∈ l₁ m ∧ M ≠ N ∧ isObtuseAngle M N) ↔
      -2 * Real.sqrt 42 / 7 < m ∧ m < 2 * Real.sqrt 42 / 7) := by
  sorry

end NUMINAMATH_CALUDE_curve_C_and_m_range_l3781_378155


namespace NUMINAMATH_CALUDE_hexagon_area_l3781_378101

-- Define the hexagon points
def hexagon_points : List (ℤ × ℤ) := [(0, 0), (1, 2), (2, 3), (4, 2), (3, 0), (0, 0)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (points : List (ℤ × ℤ)) : ℚ :=
  sorry

-- Theorem statement
theorem hexagon_area : polygon_area hexagon_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l3781_378101


namespace NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l3781_378137

theorem three_fourths_to_fifth_power : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l3781_378137


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_three_l3781_378105

theorem absolute_value_sqrt_three : 
  |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_three_l3781_378105


namespace NUMINAMATH_CALUDE_third_number_divisible_by_seven_l3781_378167

theorem third_number_divisible_by_seven (n : ℕ) : 
  (Nat.gcd 35 91 = 7) → (Nat.gcd (Nat.gcd 35 91) n = 7) → (n % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_third_number_divisible_by_seven_l3781_378167


namespace NUMINAMATH_CALUDE_counterexample_exists_negative_four_is_counterexample_l3781_378100

theorem counterexample_exists : ∃ a : ℝ, a < 3 ∧ a^2 ≥ 9 :=
  by
  use -4
  constructor
  · -- Prove -4 < 3
    sorry
  · -- Prove (-4)^2 ≥ 9
    sorry

theorem negative_four_is_counterexample : -4 < 3 ∧ (-4)^2 ≥ 9 :=
  by
  constructor
  · -- Prove -4 < 3
    sorry
  · -- Prove (-4)^2 ≥ 9
    sorry

end NUMINAMATH_CALUDE_counterexample_exists_negative_four_is_counterexample_l3781_378100


namespace NUMINAMATH_CALUDE_eight_brown_boxes_contain_480_sticks_l3781_378119

/-- Calculates the number of sticks of gum in a given number of brown boxes. -/
def sticksInBrownBoxes (numBoxes : ℕ) : ℕ :=
  let packsPerCarton : ℕ := 5
  let sticksPerPack : ℕ := 3
  let cartonsPerBox : ℕ := 4
  numBoxes * cartonsPerBox * packsPerCarton * sticksPerPack

/-- Theorem stating that 8 brown boxes contain 480 sticks of gum. -/
theorem eight_brown_boxes_contain_480_sticks :
  sticksInBrownBoxes 8 = 480 := by
  sorry


end NUMINAMATH_CALUDE_eight_brown_boxes_contain_480_sticks_l3781_378119


namespace NUMINAMATH_CALUDE_dinner_lunch_difference_l3781_378130

-- Define the number of cakes served during lunch
def lunch_cakes : ℕ := 6

-- Define the number of cakes served during dinner
def dinner_cakes : ℕ := 9

-- Theorem stating the difference between dinner and lunch cakes
theorem dinner_lunch_difference : dinner_cakes - lunch_cakes = 3 := by
  sorry

end NUMINAMATH_CALUDE_dinner_lunch_difference_l3781_378130


namespace NUMINAMATH_CALUDE_circle_center_is_zero_one_l3781_378114

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition of circle passing through a point
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define the condition of circle being tangent to parabola at a point
def tangent_to_parabola (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y = parabola x ∧ passes_through c p ∧
  ∀ q : ℝ × ℝ, q ≠ p → parabola q.1 = q.2 → ¬passes_through c q

theorem circle_center_is_zero_one :
  ∃ c : Circle,
    passes_through c (0, 2) ∧
    tangent_to_parabola c (1, 1) ∧
    c.center = (0, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_zero_one_l3781_378114


namespace NUMINAMATH_CALUDE_positive_values_of_f_l3781_378157

open Set

noncomputable def f (a : ℝ) : ℝ := a + (-1 + 9*a + 4*a^2) / (a^2 - 3*a - 10)

theorem positive_values_of_f :
  {a : ℝ | f a > 0} = Ioo (-2 : ℝ) (-1) ∪ Ioo (-1 : ℝ) 1 ∪ Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_positive_values_of_f_l3781_378157


namespace NUMINAMATH_CALUDE_trig_identity_l3781_378108

theorem trig_identity : 
  let tan30 := Real.sqrt 3 / 3
  let tan60 := Real.sqrt 3
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let cot45 := 1
  3 * tan30^2 + tan60^2 - cos30 * sin60 * cot45 = 7/4 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l3781_378108


namespace NUMINAMATH_CALUDE_project_time_allocation_l3781_378156

theorem project_time_allocation (worker1 worker2 worker3 : ℚ) 
  (h1 : worker1 = 1/2)
  (h2 : worker3 = 1/3)
  (h_total : worker1 + worker2 + worker3 = 1) :
  worker2 = 1/6 := by
sorry

end NUMINAMATH_CALUDE_project_time_allocation_l3781_378156


namespace NUMINAMATH_CALUDE_fraction_inequality_l3781_378176

theorem fraction_inequality (x : ℝ) : 
  x ∈ Set.Icc (-2 : ℝ) 2 → 
  (6 * x + 1 > 7 - 4 * x ↔ 3 / 5 < x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3781_378176


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3781_378102

theorem junk_mail_distribution (total_mail : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ)
  (h1 : total_mail = 48)
  (h2 : total_houses = 8)
  (h3 : white_houses = 2)
  (h4 : red_houses = 3)
  (h5 : total_houses > 0) :
  let colored_houses := white_houses + red_houses
  let mail_per_house := total_mail / total_houses
  mail_per_house = 6 ∧ colored_houses * mail_per_house = colored_houses * 6 :=
by sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3781_378102


namespace NUMINAMATH_CALUDE_max_value_of_f_l3781_378194

def f (a b : ℕ) : ℚ :=
  (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b)

theorem max_value_of_f :
  ∀ a b : ℕ,
  a ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ) →
  b ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ) →
  f a b ≤ 89 / 287 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3781_378194


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3781_378122

/-- Given a quadratic function f(x) = ax² + bx + c with specific properties,
    prove statements about its coefficients and roots. -/
theorem quadratic_function_properties (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^2 + b * x + c)
    (h2 : f 1 = -a / 2)
    (h3 : 3 * a > 2 * c)
    (h4 : 2 * c > 2 * b) : 
  (a > 0 ∧ -3 < b / a ∧ b / a < -3 / 4) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → 
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3781_378122


namespace NUMINAMATH_CALUDE_fifth_month_sale_l3781_378135

def sales_problem (sales1 sales2 sales3 sales4 sales6 average : ℕ) : Prop :=
  let total_sales := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales4 + sales6
  total_sales - known_sales = 3560

theorem fifth_month_sale :
  sales_problem 3435 3920 3855 4230 2000 3500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l3781_378135


namespace NUMINAMATH_CALUDE_solve_equation_l3781_378164

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (a : ℝ) : Prop := (a - i : ℂ) ^ 2 = 2 * i

-- Theorem statement
theorem solve_equation : ∃! (a : ℝ), equation a :=
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3781_378164


namespace NUMINAMATH_CALUDE_expression_evaluation_inequality_system_solution_l3781_378184

-- Part 1
theorem expression_evaluation :
  Real.sqrt 12 + |Real.sqrt 3 - 2| - 2 * Real.tan (60 * π / 180) + (1/3)⁻¹ = 5 - Real.sqrt 3 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (x + 3 * (x - 2) ≥ 2 ∧ (1 + 2 * x) / 3 > x - 1) ↔ (2 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_inequality_system_solution_l3781_378184


namespace NUMINAMATH_CALUDE_max_value_f_in_interval_l3781_378134

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧ f c = 2 ∧ ∀ x ∈ Set.Icc (-1) 1, f x ≤ f c :=
sorry

end NUMINAMATH_CALUDE_max_value_f_in_interval_l3781_378134


namespace NUMINAMATH_CALUDE_tan_identities_l3781_378117

theorem tan_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_identities_l3781_378117


namespace NUMINAMATH_CALUDE_inequality_empty_solution_set_l3781_378147

theorem inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2*a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_empty_solution_set_l3781_378147


namespace NUMINAMATH_CALUDE_aquarium_width_l3781_378199

theorem aquarium_width (length height : ℝ) (volume_final : ℝ) : 
  length = 4 → height = 3 → volume_final = 54 → 
  ∃ (width : ℝ), 3 * ((length * width * height) / 4) = volume_final ∧ width = 6 := by
sorry

end NUMINAMATH_CALUDE_aquarium_width_l3781_378199


namespace NUMINAMATH_CALUDE_negation_equivalence_l3781_378172

theorem negation_equivalence (f g : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x * g x = 0) ↔ (∀ x : ℝ, f x ≠ 0 ∧ g x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3781_378172


namespace NUMINAMATH_CALUDE_pen_price_proof_l3781_378186

/-- Represents the regular price of a pen in dollars -/
def regular_price : ℝ := 2

/-- Represents the total number of pens bought -/
def total_pens : ℕ := 20

/-- Represents the total cost paid by the customer in dollars -/
def total_cost : ℝ := 30

/-- Represents the number of pens at regular price -/
def regular_price_pens : ℕ := 10

/-- Represents the number of pens at half price -/
def half_price_pens : ℕ := 10

theorem pen_price_proof :
  regular_price * regular_price_pens + 
  (regular_price / 2) * half_price_pens = total_cost ∧
  regular_price_pens + half_price_pens = total_pens := by
  sorry

end NUMINAMATH_CALUDE_pen_price_proof_l3781_378186


namespace NUMINAMATH_CALUDE_tetrahedron_inference_is_logical_l3781_378195

/-- Represents the concept of logical reasoning -/
def LogicalReasoning : Type := Unit

/-- Represents the concept of analogical reasoning -/
def AnalogicalReasoning : Type := Unit

/-- Represents the act of inferring properties of a spatial tetrahedron from a plane triangle -/
def InferTetrahedronFromTriangle : Type := Unit

/-- Analogical reasoning is a type of logical reasoning -/
axiom analogical_is_logical : AnalogicalReasoning → LogicalReasoning

/-- Inferring tetrahedron properties from triangle properties is analogical reasoning -/
axiom tetrahedron_inference_is_analogical : InferTetrahedronFromTriangle → AnalogicalReasoning

/-- Theorem: Inferring properties of a spatial tetrahedron from properties of a plane triangle
    is a kind of logical reasoning -/
theorem tetrahedron_inference_is_logical : InferTetrahedronFromTriangle → LogicalReasoning := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inference_is_logical_l3781_378195


namespace NUMINAMATH_CALUDE_larger_number_problem_l3781_378142

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1395)
  (h2 : L = 6 * S + 15) :
  L = 1671 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3781_378142


namespace NUMINAMATH_CALUDE_nonagon_dissection_l3781_378149

/-- Represents a rhombus with unit side length and a specific angle -/
structure Rhombus :=
  (angle : ℝ)

/-- Represents an isosceles triangle with unit side length and a specific vertex angle -/
structure IsoscelesTriangle :=
  (vertex_angle : ℝ)

/-- Represents a regular polygon with a specific number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- The original 9-gon composed of specific shapes -/
def original_nonagon : RegularPolygon :=
  { sides := 9 }

/-- The set of rhombuses with 40° angles -/
def rhombuses_40 : Finset Rhombus :=
  sorry

/-- The set of rhombuses with 80° angles -/
def rhombuses_80 : Finset Rhombus :=
  sorry

/-- The set of isosceles triangles with 120° vertex angles -/
def triangles_120 : Finset IsoscelesTriangle :=
  sorry

/-- Represents the dissection of the original nonagon into three congruent regular nonagons -/
def dissection (original : RegularPolygon) (parts : Finset RegularPolygon) : Prop :=
  sorry

/-- The theorem stating that the original nonagon can be dissected into three congruent regular nonagons -/
theorem nonagon_dissection :
  ∃ (parts : Finset RegularPolygon),
    (parts.card = 3) ∧
    (∀ p ∈ parts, p.sides = 9) ∧
    (dissection original_nonagon parts) :=
sorry

end NUMINAMATH_CALUDE_nonagon_dissection_l3781_378149


namespace NUMINAMATH_CALUDE_system_solution_value_l3781_378143

theorem system_solution_value (x y a b : ℝ) : 
  3 * x - 2 * y + 20 = 0 →
  2 * x + 15 * y - 3 = 0 →
  a * x - b * y = 3 →
  6 * a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_value_l3781_378143


namespace NUMINAMATH_CALUDE_irrational_sqrt_7_rational_others_l3781_378169

theorem irrational_sqrt_7_rational_others : 
  (Irrational (Real.sqrt 7)) ∧ 
  (¬ Irrational 3.1415) ∧ 
  (¬ Irrational 3) ∧ 
  (¬ Irrational (1/3 : ℚ)) := by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_7_rational_others_l3781_378169


namespace NUMINAMATH_CALUDE_inequality_proof_l3781_378118

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3781_378118


namespace NUMINAMATH_CALUDE_exist_five_naturals_sum_product_ten_l3781_378112

theorem exist_five_naturals_sum_product_ten : 
  ∃ (a b c d e : ℕ), a + b + c + d + e = 10 ∧ a * b * c * d * e = 10 :=
by sorry

end NUMINAMATH_CALUDE_exist_five_naturals_sum_product_ten_l3781_378112


namespace NUMINAMATH_CALUDE_sin_cos_sum_13_17_l3781_378132

theorem sin_cos_sum_13_17 :
  Real.sin (13 * π / 180) * Real.cos (17 * π / 180) +
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_13_17_l3781_378132


namespace NUMINAMATH_CALUDE_units_digit_problem_l3781_378160

theorem units_digit_problem : ∃ n : ℕ, (7 * 27 * 1977 + 9) - 7^3 ≡ 9 [ZMOD 10] ∧ n < 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3781_378160


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3781_378124

theorem quadratic_inequality_properties (a b c : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) → 
  (a < 0 ∧ a - b + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3781_378124


namespace NUMINAMATH_CALUDE_sequence_eleventh_term_l3781_378163

/-- Given a sequence a₁, a₂, ..., where a₁ = 3 and aₙ₊₁ - aₙ = n for n ≥ 1,
    prove that a₁₁ = 58. -/
theorem sequence_eleventh_term (a : ℕ → ℕ) 
  (h1 : a 1 = 3)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = n) : 
  a 11 = 58 := by
  sorry

end NUMINAMATH_CALUDE_sequence_eleventh_term_l3781_378163


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3781_378146

theorem complex_equation_solution (z : ℂ) :
  (3 + 4*I) * z = 1 - 2*I → z = -1/5 - 2/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3781_378146


namespace NUMINAMATH_CALUDE_orange_boxes_l3781_378189

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 42) (h2 : oranges_per_box = 6) :
  total_oranges / oranges_per_box = 7 :=
by sorry

end NUMINAMATH_CALUDE_orange_boxes_l3781_378189


namespace NUMINAMATH_CALUDE_negative_solution_existence_l3781_378162

/-- The inequality x^2 < 4 - |x - a| has at least one negative solution if and only if a ∈ [-17/4, 4). -/
theorem negative_solution_existence (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a|) ↔ -17/4 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_solution_existence_l3781_378162


namespace NUMINAMATH_CALUDE_complex_sum_equals_two_l3781_378196

def z : ℂ := 1 - Complex.I

theorem complex_sum_equals_two : (2 / z) + z = 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_two_l3781_378196


namespace NUMINAMATH_CALUDE_tank_fill_time_l3781_378150

-- Define the rates of the pipes
def input_pipe_rate : ℚ := 1 / 15
def outlet_pipe_rate : ℚ := 1 / 45

-- Define the combined rate of all pipes
def combined_rate : ℚ := 2 * input_pipe_rate - outlet_pipe_rate

-- State the theorem
theorem tank_fill_time :
  (1 : ℚ) / combined_rate = 9 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3781_378150


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l3781_378123

theorem reciprocal_equation_solution (x : ℝ) :
  (2 - (1 / (1 - x)) = 1 / (1 - x)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l3781_378123


namespace NUMINAMATH_CALUDE_units_digit_of_4539_pow_201_l3781_378166

theorem units_digit_of_4539_pow_201 : (4539^201) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_4539_pow_201_l3781_378166


namespace NUMINAMATH_CALUDE_tan_beta_value_l3781_378133

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3)
  (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3781_378133


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3781_378197

theorem inequality_system_solution (x : ℤ) : 
  (2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1) ↔ x ∈ ({3, 4, 5} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3781_378197


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l3781_378129

theorem x_range_for_quadratic_inequality :
  (∀ m : ℝ, |m| ≤ 2 → ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0) →
  ∀ x : ℝ, (-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l3781_378129


namespace NUMINAMATH_CALUDE_total_hamburgers_bought_l3781_378110

/-- Proves that the total number of hamburgers bought is 50 given the specified conditions. -/
theorem total_hamburgers_bought (total_spent : ℚ) (single_cost : ℚ) (double_cost : ℚ) (double_count : ℕ) : ℕ :=
  if total_spent = 70.5 ∧ single_cost = 1 ∧ double_cost = 1.5 ∧ double_count = 41 then
    50
  else
    0

#check total_hamburgers_bought

end NUMINAMATH_CALUDE_total_hamburgers_bought_l3781_378110


namespace NUMINAMATH_CALUDE_max_value_2a_plus_b_l3781_378107

theorem max_value_2a_plus_b (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  2 * a + b ≤ 5 ∧ ∃ (a' b' : ℝ), 4 * a' + 3 * b' ≤ 10 ∧ 3 * a' + 6 * b' ≤ 12 ∧ 2 * a' + b' = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_2a_plus_b_l3781_378107


namespace NUMINAMATH_CALUDE_boat_distance_downstream_l3781_378145

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
def distanceDownstream (boatSpeed streamSpeed time : ℝ) : ℝ :=
  (boatSpeed + streamSpeed) * time

/-- Proves that the distance traveled downstream is 54 km under the given conditions -/
theorem boat_distance_downstream :
  let boatSpeed : ℝ := 10
  let streamSpeed : ℝ := 8
  let time : ℝ := 3
  distanceDownstream boatSpeed streamSpeed time = 54 := by
sorry

#eval distanceDownstream 10 8 3

end NUMINAMATH_CALUDE_boat_distance_downstream_l3781_378145


namespace NUMINAMATH_CALUDE_power_function_domain_and_oddness_l3781_378103

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_real_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ y, f x = y

theorem power_function_domain_and_oddness (a : ℝ) :
  a ∈ ({-1, 0, 1/2, 1, 2, 3} : Set ℝ) →
  (has_real_domain (fun x ↦ x^a) ∧ is_odd_function (fun x ↦ x^a)) ↔ (a = 1 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_power_function_domain_and_oddness_l3781_378103


namespace NUMINAMATH_CALUDE_complex_exp_eleven_pi_over_two_equals_neg_i_l3781_378109

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_eleven_pi_over_two_equals_neg_i :
  cexp (11 * Real.pi / 2 * Complex.I) = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_exp_eleven_pi_over_two_equals_neg_i_l3781_378109


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_two_l3781_378193

theorem sum_of_solutions_is_two :
  ∃ (x y : ℤ), x^2 = x + 224 ∧ y^2 = y + 224 ∧ x + y = 2 ∧
  ∀ (z : ℤ), z^2 = z + 224 → z = x ∨ z = y :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_two_l3781_378193


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_twice_C_x_plus_y_equals_76_l3781_378148

-- Define the angles
def angle_A : ℝ := 34
def angle_B : ℝ := 80
def angle_C : ℝ := 38

-- Define x and y as real numbers (representing angle measures)
variable (x y : ℝ)

-- State the theorem
theorem sum_of_x_and_y_equals_twice_C :
  x + y = 2 * angle_C := by sorry

-- Prove that x + y equals 76
theorem x_plus_y_equals_76 :
  x + y = 76 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_twice_C_x_plus_y_equals_76_l3781_378148


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l3781_378126

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define a positive relationship between two variables
def positive_relationship (x y : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → x a < x b → y a < y b

-- Define a perfect linear relationship between two variables
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ m b, ∀ t, y t = m * x t + b

-- Theorem statement
theorem correlation_coefficient_properties
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → positive_relationship x y) ∧
  (r = 1 ∨ r = -1 → perfect_linear_relationship x y) :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l3781_378126


namespace NUMINAMATH_CALUDE_cars_distance_theorem_l3781_378154

/-- The distance between two cars on a straight road -/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - (car1_distance + car2_distance)

/-- Theorem: The distance between two cars is 28 km -/
theorem cars_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 113)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 35) :
  distance_between_cars initial_distance car1_distance car2_distance = 28 := by
  sorry

#eval distance_between_cars 113 50 35

end NUMINAMATH_CALUDE_cars_distance_theorem_l3781_378154


namespace NUMINAMATH_CALUDE_sets_and_domains_l3781_378188

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| ≥ 1}
def B : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 1}

-- State the theorem
theorem sets_and_domains (a : ℝ) (h : a < 1) :
  (A ∩ B = {x | x < -1 ∨ x ≥ 2}) ∧
  ((Set.univ \ (A ∪ B)) = {x | 0 < x ∧ x < 1}) ∧
  (C a ⊆ B → (a ≤ -2 ∨ (1/2 ≤ a ∧ a < 1))) :=
by sorry

end NUMINAMATH_CALUDE_sets_and_domains_l3781_378188


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3781_378161

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), (6 * s^2 = 294) → (s^3 = 343) := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3781_378161


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l3781_378190

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_conversion :
  ∀ (x y : ℝ), x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 4 ∧ θ = π / 4 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry


end NUMINAMATH_CALUDE_rect_to_polar_conversion_l3781_378190


namespace NUMINAMATH_CALUDE_river_flow_volume_l3781_378182

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate. -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 12) 
  (h_width : width = 35) 
  (h_flow_rate : flow_rate_kmph = 8) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 56000 := by
  sorry

end NUMINAMATH_CALUDE_river_flow_volume_l3781_378182


namespace NUMINAMATH_CALUDE_cyclist_speed_solution_l3781_378174

/-- Represents the speeds and distance of two cyclists traveling in opposite directions. -/
structure CyclistProblem where
  slower_speed : ℝ
  time : ℝ
  distance_apart : ℝ
  speed_difference : ℝ

/-- Calculates the total distance traveled by both cyclists. -/
def total_distance (p : CyclistProblem) : ℝ :=
  p.time * (2 * p.slower_speed + p.speed_difference)

/-- Theorem stating the conditions and solution for the cyclist problem. -/
theorem cyclist_speed_solution (p : CyclistProblem) 
  (h1 : p.time = 6)
  (h2 : p.distance_apart = 246)
  (h3 : p.speed_difference = 5) :
  p.slower_speed = 18 ∧ p.slower_speed + p.speed_difference = 23 :=
by
  sorry

#check cyclist_speed_solution

end NUMINAMATH_CALUDE_cyclist_speed_solution_l3781_378174


namespace NUMINAMATH_CALUDE_man_speed_against_current_is_10_l3781_378171

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def man_speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current_is_10 :
  man_speed_against_current 15 2.5 = 10 := by
  sorry

#eval man_speed_against_current 15 2.5

end NUMINAMATH_CALUDE_man_speed_against_current_is_10_l3781_378171


namespace NUMINAMATH_CALUDE_margarets_mean_score_l3781_378127

def scores : List ℝ := [82, 85, 88, 90, 95, 97, 98, 100]

theorem margarets_mean_score 
  (h1 : scores.length = 8)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        margaret_scores.length = 4 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 91) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 4 ∧ 
    margaret_scores.sum / margaret_scores.length = 92.75 := by
  sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l3781_378127


namespace NUMINAMATH_CALUDE_number_difference_l3781_378151

theorem number_difference (x y : ℝ) (h1 : x + y = 147) (h2 : x - 0.375 * y = 4) (h3 : x ≥ y) : x - 0.375 * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3781_378151


namespace NUMINAMATH_CALUDE_hydrochloric_acid_percentage_l3781_378120

/-- Calculates the percentage of hydrochloric acid in a solution after adding water -/
theorem hydrochloric_acid_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (initial_acid_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 300)
  (h2 : initial_water_percentage = 0.60)
  (h3 : initial_acid_percentage = 0.40)
  (h4 : added_water = 100)
  (h5 : initial_water_percentage + initial_acid_percentage = 1) :
  let initial_water := initial_volume * initial_water_percentage
  let initial_acid := initial_volume * initial_acid_percentage
  let final_volume := initial_volume + added_water
  let final_water := initial_water + added_water
  let final_acid := initial_acid
  final_acid / final_volume = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_percentage_l3781_378120


namespace NUMINAMATH_CALUDE_common_root_cubics_theorem_l3781_378125

/-- Two cubic equations with two common roots -/
structure CommonRootCubics where
  A : ℝ
  B : ℝ
  C : ℝ
  root1 : ℝ
  root2 : ℝ
  eq1_holds : ∀ x : ℝ, x^3 + A*x^2 + 20*x + C = 0 ↔ x = root1 ∨ x = root2 ∨ x = -A - root1 - root2
  eq2_holds : ∀ x : ℝ, x^3 + B*x^2 + 100 = 0 ↔ x = root1 ∨ x = root2 ∨ x = -B - root1 - root2

theorem common_root_cubics_theorem (cubics : CommonRootCubics) :
  cubics.C = 100 ∧ cubics.root1 * cubics.root2 = 5 * Real.rpow 5 (1/3) := by sorry

end NUMINAMATH_CALUDE_common_root_cubics_theorem_l3781_378125
