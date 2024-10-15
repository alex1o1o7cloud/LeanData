import Mathlib

namespace NUMINAMATH_CALUDE_hemisphere_center_of_mass_l347_34794

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a hemisphere -/
structure Hemisphere where
  radius : ℝ

/-- Density function for the hemisphere -/
def density (p : Point3D) : ℝ :=
  sorry

/-- Center of mass of a hemisphere -/
def centerOfMass (h : Hemisphere) : Point3D :=
  sorry

/-- Theorem: The center of mass of a hemisphere with radius R and volume density
    proportional to the distance from the origin is located at (0, 0, 2R/5) -/
theorem hemisphere_center_of_mass (h : Hemisphere) :
  let com := centerOfMass h
  com.x = 0 ∧ com.y = 0 ∧ com.z = 2 * h.radius / 5 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_center_of_mass_l347_34794


namespace NUMINAMATH_CALUDE_cross_area_l347_34757

-- Define the grid size
def gridSize : Nat := 6

-- Define the center point of the cross
def centerPoint : (Nat × Nat) := (3, 3)

-- Define the arm length of the cross
def armLength : Nat := 1

-- Define the boundary points of the cross
def boundaryPoints : List (Nat × Nat) := [(3, 1), (1, 3), (3, 3), (3, 5), (5, 3)]

-- Define the interior points of the cross
def interiorPoints : List (Nat × Nat) := [(3, 2), (2, 3), (4, 3), (3, 4)]

-- Theorem: The area of the cross is 6 square units
theorem cross_area : Nat := by
  sorry

end NUMINAMATH_CALUDE_cross_area_l347_34757


namespace NUMINAMATH_CALUDE_franks_reading_average_l347_34755

/-- Calculates the average pages read per day given the pages and days for three books --/
def average_pages_per_day (pages1 pages2 pages3 : ℕ) (days1 days2 days3 : ℕ) : ℚ :=
  (pages1 + pages2 + pages3 : ℚ) / (days1 + days2 + days3)

/-- Theorem stating that the average pages per day for Frank's reading is as calculated --/
theorem franks_reading_average :
  average_pages_per_day 249 379 480 3 5 6 = 79.14 := by
  sorry

end NUMINAMATH_CALUDE_franks_reading_average_l347_34755


namespace NUMINAMATH_CALUDE_classroom_benches_l347_34704

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of benches needed given the number of students and students per bench -/
def benchesNeeded (students : ℕ) (studentsPerBench : ℕ) : ℕ := sorry

theorem classroom_benches :
  let studentsBase5 : ℕ := 312
  let studentsPerBench : ℕ := 3
  let studentsBase10 : ℕ := base5ToBase10 studentsBase5
  benchesNeeded studentsBase10 studentsPerBench = 28 := by sorry

end NUMINAMATH_CALUDE_classroom_benches_l347_34704


namespace NUMINAMATH_CALUDE_cards_per_page_l347_34768

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 3) 
  (h2 : old_cards = 9) 
  (h3 : pages = 4) : 
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l347_34768


namespace NUMINAMATH_CALUDE_both_questions_correct_percentage_l347_34771

theorem both_questions_correct_percentage
  (p_first : ℝ)
  (p_second : ℝ)
  (p_neither : ℝ)
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.60 :=
by
  sorry

end NUMINAMATH_CALUDE_both_questions_correct_percentage_l347_34771


namespace NUMINAMATH_CALUDE_problem_solution_l347_34782

theorem problem_solution (x : ℝ) (h : Real.sqrt x = 6000 * (1/1000)) :
  (600 - Real.sqrt x)^2 + x = 352872 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l347_34782


namespace NUMINAMATH_CALUDE_polynomial_remainder_l347_34736

theorem polynomial_remainder (x : ℝ) : (x^13 + 1) % (x - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l347_34736


namespace NUMINAMATH_CALUDE_min_team_size_is_six_l347_34718

/-- Represents the job parameters and conditions -/
structure JobParameters where
  totalDays : ℕ
  initialDays : ℕ
  initialWorkCompleted : ℚ
  initialTeamSize : ℕ
  rateIncreaseDay : ℕ
  rateIncreaseFactor : ℚ

/-- Calculates the minimum team size required from the rate increase day -/
def minTeamSizeAfterRateIncrease (params : JobParameters) : ℕ :=
  sorry

/-- Theorem stating that the minimum team size after rate increase is 6 -/
theorem min_team_size_is_six (params : JobParameters)
  (h1 : params.totalDays = 40)
  (h2 : params.initialDays = 10)
  (h3 : params.initialWorkCompleted = 1/4)
  (h4 : params.initialTeamSize = 12)
  (h5 : params.rateIncreaseDay = 20)
  (h6 : params.rateIncreaseFactor = 2) :
  minTeamSizeAfterRateIncrease params = 6 :=
sorry

end NUMINAMATH_CALUDE_min_team_size_is_six_l347_34718


namespace NUMINAMATH_CALUDE_part1_part2_l347_34777

-- Define the inequality function
def f (k x : ℝ) : ℝ := (k^2 - 2*k - 3)*x^2 - (k + 1)*x - 1

-- Define the solution set M
def M (k : ℝ) : Set ℝ := {x : ℝ | f k x < 0}

-- Part 1: Range of positive integer k when 1 ∈ M
theorem part1 : 
  (∀ k : ℕ+, 1 ∈ M k ↔ k ∈ ({1, 2, 3, 4} : Set ℕ+)) :=
sorry

-- Part 2: Range of real k when M = ℝ
theorem part2 : 
  (∀ k : ℝ, M k = Set.univ ↔ k ∈ Set.Icc (-1) (11/5)) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l347_34777


namespace NUMINAMATH_CALUDE_complex_subtraction_l347_34779

theorem complex_subtraction : (5 * Complex.I) - (2 + 2 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l347_34779


namespace NUMINAMATH_CALUDE_factorization_equality_l347_34763

theorem factorization_equality (x : ℝ) : 6 * x^2 + 5 * x - 1 = (6 * x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l347_34763


namespace NUMINAMATH_CALUDE_smallest_a_with_single_digit_sum_l347_34744

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is single-digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- The property we want to prove -/
def has_single_digit_sum (a : ℕ) : Prop :=
  is_single_digit (sum_of_digits (10^a - 74))

theorem smallest_a_with_single_digit_sum :
  (∀ k < 2, ¬ has_single_digit_sum k) ∧ has_single_digit_sum 2 := by sorry

end NUMINAMATH_CALUDE_smallest_a_with_single_digit_sum_l347_34744


namespace NUMINAMATH_CALUDE_largest_number_with_digits_3_2_sum_11_l347_34784

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digits_3_2_sum_11 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 11 → n ≤ 32222 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digits_3_2_sum_11_l347_34784


namespace NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_when_always_ge_three_l347_34781

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1: Solution set when a = -1
theorem solution_set_when_a_neg_one :
  {x : ℝ | f (-1) x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Part 2: Range of a when f(x) ≥ 3 for all x
theorem range_of_a_when_always_ge_three :
  {a : ℝ | ∀ x, f a x ≥ 3} = {a : ℝ | a ≤ -2 ∨ a ≥ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_when_always_ge_three_l347_34781


namespace NUMINAMATH_CALUDE_solve_for_q_l347_34773

theorem solve_for_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p*q = 9) :
  q = (9 + 3*Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l347_34773


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l347_34780

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (2, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (vector_a x) vector_b → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l347_34780


namespace NUMINAMATH_CALUDE_line_intersects_extension_l347_34787

/-- Given a line l: Ax + By + C = 0 and two points P₁ and P₂, 
    prove that l intersects with the extension of P₁P₂ under certain conditions. -/
theorem line_intersects_extension (A B C x₁ y₁ x₂ y₂ : ℝ) 
  (hAB : A ≠ 0 ∨ B ≠ 0)
  (hSameSide : (A * x₁ + B * y₁ + C) * (A * x₂ + B * y₂ + C) > 0)
  (hDistance : |A * x₁ + B * y₁ + C| > |A * x₂ + B * y₂ + C|) :
  ∃ (t : ℝ), t > 1 ∧ A * (x₁ + t * (x₂ - x₁)) + B * (y₁ + t * (y₂ - y₁)) + C = 0 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_extension_l347_34787


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l347_34759

theorem square_perimeter_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 / b^2 = 16 / 25) → ((4 * a) / (4 * b) = 4 / 5) :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l347_34759


namespace NUMINAMATH_CALUDE_total_cookies_count_l347_34770

/-- Given 286 bags of cookies with 452 cookies in each bag, 
    prove that the total number of cookies is 129,272. -/
theorem total_cookies_count (bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : bags = 286) (h2 : cookies_per_bag = 452) : 
  bags * cookies_per_bag = 129272 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_count_l347_34770


namespace NUMINAMATH_CALUDE_ABC_equality_l347_34788

variables (u v w : ℝ)
variables (A B C : ℝ)

def A_def : A = u * v + u + 1 := by sorry
def B_def : B = v * w + v + 1 := by sorry
def C_def : C = w * u + w + 1 := by sorry
def uvw_condition : u * v * w = 1 := by sorry

theorem ABC_equality : A * B * C = A * B + B * C + C * A := by sorry

end NUMINAMATH_CALUDE_ABC_equality_l347_34788


namespace NUMINAMATH_CALUDE_third_dog_summer_avg_distance_proof_l347_34719

/-- Represents the average daily distance walked by the third dog in summer -/
def third_dog_summer_avg_distance : ℝ := 2.2

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the number of weekend days in a month -/
def weekend_days : ℕ := 8

/-- Represents the distance walked by the third dog on a summer weekday -/
def third_dog_summer_distance : ℝ := 3

theorem third_dog_summer_avg_distance_proof :
  third_dog_summer_avg_distance = 
    (third_dog_summer_distance * (days_in_month - weekend_days)) / days_in_month :=
by sorry

end NUMINAMATH_CALUDE_third_dog_summer_avg_distance_proof_l347_34719


namespace NUMINAMATH_CALUDE_max_temp_range_l347_34775

/-- Given 5 temperatures with an average of 40 and a minimum of 30,
    the maximum possible range is 50. -/
theorem max_temp_range (temps : Fin 5 → ℝ) 
    (avg : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 40)
    (min : ∀ i, temps i ≥ 30) 
    (exists_min : ∃ i, temps i = 30) : 
    (∀ i j, temps i - temps j ≤ 50) ∧ 
    (∃ i j, temps i - temps j = 50) := by
  sorry

end NUMINAMATH_CALUDE_max_temp_range_l347_34775


namespace NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l347_34739

theorem m_equals_one_sufficient_not_necessary (m : ℝ) :
  (m = 1 → |m| = 1) ∧ (∃ m : ℝ, |m| = 1 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l347_34739


namespace NUMINAMATH_CALUDE_min_boxes_fit_l347_34715

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- The large box dimensions -/
def largeBox : BoxDimensions := ⟨12, 14, 16⟩

/-- The approximate dimensions of the small irregular boxes -/
def smallBox : BoxDimensions := ⟨3, 7, 2⟩

/-- Theorem stating that at least 64 small boxes can fit into the large box -/
theorem min_boxes_fit (irreg_shape : Prop) : ∃ n : ℕ, n ≥ 64 ∧ n * boxVolume smallBox ≤ boxVolume largeBox := by
  sorry

end NUMINAMATH_CALUDE_min_boxes_fit_l347_34715


namespace NUMINAMATH_CALUDE_certain_value_calculation_l347_34795

theorem certain_value_calculation (x : ℝ) (v : ℝ) (h1 : x = 100) (h2 : 0.8 * x + v = x) : v = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_calculation_l347_34795


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_increasing_l347_34792

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The condition a₁ < a₂ < a₃ -/
def FirstThreeIncreasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_increasing_iff_first_three_increasing
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  IncreasingSequence a ↔ FirstThreeIncreasing a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_increasing_l347_34792


namespace NUMINAMATH_CALUDE_vertex_in_second_quadrant_l347_34735

/-- The quadratic function f(x) = -(x+1)^2 + 2 -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 2

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := -1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := f vertex_x

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The vertex of f(x) = -(x+1)^2 + 2 is in the second quadrant -/
theorem vertex_in_second_quadrant : is_in_second_quadrant vertex_x vertex_y := by
  sorry

end NUMINAMATH_CALUDE_vertex_in_second_quadrant_l347_34735


namespace NUMINAMATH_CALUDE_movie_screening_guests_l347_34710

theorem movie_screening_guests :
  ∀ G : ℕ,
  G / 2 + 15 + (G - (G / 2 + 15)) = G →  -- Total guests = women + men + children
  G - (15 / 5 + 4) = 43 →                -- Guests who stayed
  G = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_screening_guests_l347_34710


namespace NUMINAMATH_CALUDE_no_integer_solutions_l347_34717

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l347_34717


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l347_34774

theorem complex_on_imaginary_axis (a : ℝ) :
  let z : ℂ := (a^2 - 2*a) + (a^2 - a - 2)*I
  (z.re = 0) ↔ (a = 2 ∨ a = 0) := by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l347_34774


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l347_34793

def has_solution (n : ℕ) : Prop :=
  ∃ (a b c : ℤ), a^n + b^n = c^n + n

theorem diophantine_equation_solutions :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → (has_solution n ↔ n = 1 ∨ n = 2 ∨ n = 3)) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l347_34793


namespace NUMINAMATH_CALUDE_elises_initial_money_l347_34709

/-- Proves that Elise's initial amount of money was $8 --/
theorem elises_initial_money :
  ∀ (initial savings comic_cost puzzle_cost final : ℕ),
  savings = 13 →
  comic_cost = 2 →
  puzzle_cost = 18 →
  final = 1 →
  initial + savings - comic_cost - puzzle_cost = final →
  initial = 8 := by
sorry

end NUMINAMATH_CALUDE_elises_initial_money_l347_34709


namespace NUMINAMATH_CALUDE_tetris_arrangement_exists_l347_34705

/-- Represents a Tetris piece type -/
inductive TetrisPiece
  | O | I | T | S | Z | L | J

/-- Represents a position on the 6x6 grid -/
structure Position where
  x : Fin 6
  y : Fin 6

/-- Represents a placed Tetris piece on the grid -/
structure PlacedPiece where
  piece : TetrisPiece
  positions : List Position

/-- Checks if a list of placed pieces forms a valid arrangement -/
def isValidArrangement (pieces : List PlacedPiece) : Prop :=
  -- Each position on the 6x6 grid is covered exactly once
  ∀ (x y : Fin 6), ∃! p : PlacedPiece, p ∈ pieces ∧ Position.mk x y ∈ p.positions

/-- Checks if all piece types are used at least once -/
def allPiecesUsed (pieces : List PlacedPiece) : Prop :=
  ∀ t : TetrisPiece, ∃ p : PlacedPiece, p ∈ pieces ∧ p.piece = t

/-- Main theorem: There exists a valid arrangement of Tetris pieces -/
theorem tetris_arrangement_exists : 
  ∃ (pieces : List PlacedPiece), isValidArrangement pieces ∧ allPiecesUsed pieces :=
sorry

end NUMINAMATH_CALUDE_tetris_arrangement_exists_l347_34705


namespace NUMINAMATH_CALUDE_one_approval_probability_l347_34720

/-- The probability of a voter approving the council's measures -/
def approval_rate : ℝ := 0.6

/-- The number of voters polled -/
def num_polled : ℕ := 4

/-- The probability of exactly one voter approving out of the polled voters -/
def prob_one_approval : ℝ := 4 * (approval_rate * (1 - approval_rate)^3)

/-- Theorem stating that the probability of exactly one voter approving is 0.1536 -/
theorem one_approval_probability : prob_one_approval = 0.1536 := by
  sorry

end NUMINAMATH_CALUDE_one_approval_probability_l347_34720


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l347_34701

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 5 ≥ 0 ∧ x < 7) ↔ (5 ≤ x ∧ x < 7) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l347_34701


namespace NUMINAMATH_CALUDE_double_base_exponent_l347_34741

theorem double_base_exponent (a b x : ℝ) (hb : b ≠ 0) :
  (2 * a)^(2 * b) = a^b * x^b → x = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_l347_34741


namespace NUMINAMATH_CALUDE_slope_point_relation_l347_34778

theorem slope_point_relation (m : ℝ) : 
  m > 0 → 
  ((m + 1 - 4) / (2 - m) = Real.sqrt 5) → 
  m = (10 - Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_slope_point_relation_l347_34778


namespace NUMINAMATH_CALUDE_equation_solution_l347_34708

theorem equation_solution : 
  ∃! x : ℝ, x > 0 ∧ Real.sqrt (3 * x - 2) + 9 / Real.sqrt (3 * x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l347_34708


namespace NUMINAMATH_CALUDE_cucumbers_for_twenty_apples_l347_34760

/-- The number of cucumbers that can be bought for the price of 20 apples,
    given the cost equivalences between apples, bananas, and cucumbers. -/
theorem cucumbers_for_twenty_apples :
  -- Condition 1: Ten apples cost the same as five bananas
  ∀ (apple_cost banana_cost : ℝ),
  10 * apple_cost = 5 * banana_cost →
  -- Condition 2: Three bananas cost the same as four cucumbers
  ∀ (cucumber_cost : ℝ),
  3 * banana_cost = 4 * cucumber_cost →
  -- Conclusion: 20 apples are equivalent in cost to 13 cucumbers
  20 * apple_cost = 13 * cucumber_cost :=
by
  sorry

end NUMINAMATH_CALUDE_cucumbers_for_twenty_apples_l347_34760


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l347_34785

-- Define the common ratios and terms of the geometric sequences
variables {k p r : ℝ} {a₂ a₃ b₂ b₃ : ℝ}

-- Define the geometric sequences
def is_geometric_sequence (k p a₂ a₃ : ℝ) : Prop :=
  a₂ = k * p ∧ a₃ = k * p^2

-- State the theorem
theorem sum_of_common_ratios
  (h₁ : is_geometric_sequence k p a₂ a₃)
  (h₂ : is_geometric_sequence k r b₂ b₃)
  (h₃ : p ≠ r)
  (h₄ : k ≠ 0)
  (h₅ : a₃ - b₃ = 4 * (a₂ - b₂)) :
  p + r = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l347_34785


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l347_34723

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) : 
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / (2 * x' + y') + 1 / (y' + 1) = 1 → x + 2 * y ≤ x' + 2 * y') ∧ 
  x + 2 * y = Real.sqrt 3 + 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l347_34723


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_l347_34764

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- Theorem statement
theorem x_negative_necessary_not_sufficient :
  (∀ x : ℝ, ln (x + 1) < 0 → x < 0) ∧
  ¬(∀ x : ℝ, x < 0 → ln (x + 1) < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_l347_34764


namespace NUMINAMATH_CALUDE_thirteenth_on_monday_l347_34799

/-- Represents a day of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents a month of the year -/
inductive Month
  | january
  | february
  | march
  | april
  | may
  | june
  | july
  | august
  | september
  | october
  | november
  | december

/-- Returns the number of days in a given month -/
def daysInMonth (m : Month) : Nat :=
  match m with
  | .january => 31
  | .february => 28  -- Assuming non-leap year for simplicity
  | .march => 31
  | .april => 30
  | .may => 31
  | .june => 30
  | .july => 31
  | .august => 31
  | .september => 30
  | .october => 31
  | .november => 30
  | .december => 31

/-- Calculates the day of the week for the 13th of a given month, 
    given the day of the week for the 13th of the previous month -/
def dayOf13th (prevDay : DayOfWeek) (m : Month) : DayOfWeek :=
  sorry

/-- Theorem: In any year, there exists at least one month where the 13th falls on a Monday -/
theorem thirteenth_on_monday :
  ∀ (startDay : DayOfWeek), 
    ∃ (m : Month), dayOf13th startDay m = DayOfWeek.monday :=
  sorry

end NUMINAMATH_CALUDE_thirteenth_on_monday_l347_34799


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l347_34761

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 4*x - 12 = 0) :
  x^3 - 4*x^2 - 12*x + 16 = 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l347_34761


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_rectangular_solid_l347_34711

/-- The surface area of a sphere circumscribing a rectangular solid -/
theorem sphere_surface_area_of_rectangular_solid (l w h : ℝ) (S : ℝ) :
  l = 2 →
  w = 2 →
  h = 1 →
  S = 4 * Real.pi * ((l^2 + w^2 + h^2) / 4) →
  S = 9 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_rectangular_solid_l347_34711


namespace NUMINAMATH_CALUDE_cubic_root_sum_l347_34743

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 24*r^2 + 50*r - 24 = 0 →
  s^3 - 24*s^2 + 50*s - 24 = 0 →
  t^3 - 24*t^2 + 50*t - 24 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 19.04 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l347_34743


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l347_34758

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define set B (domain of log(4x - x^2))
def B : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 4 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l347_34758


namespace NUMINAMATH_CALUDE_rhombus_area_l347_34729

/-- A rhombus with side length √113 and diagonals differing by 8 units has an area of 194 square units. -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 → 
  diag_diff = 8 → 
  area = 194 → 
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ d₂ - d₁ = diag_diff ∧ d₁ * d₂ / 2 = area ∧ 
    d₁^2 / 4 + d₂^2 / 4 = side^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l347_34729


namespace NUMINAMATH_CALUDE_triangle_area_l347_34706

/-- The area of a triangle with base 2t and height 3t + 2, where t = 6 -/
theorem triangle_area (t : ℝ) (h : t = 6) : (1/2 : ℝ) * (2*t) * (3*t + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l347_34706


namespace NUMINAMATH_CALUDE_clock_problem_l347_34738

/-- Represents a time on a 12-hour digital clock -/
structure Time where
  hour : Nat
  minute : Nat
  second : Nat
  deriving Repr

/-- Adds a duration to a given time -/
def addDuration (t : Time) (hours minutes seconds : Nat) : Time :=
  sorry

/-- Calculates the sum of hour, minute, and second values of a time -/
def timeSum (t : Time) : Nat :=
  sorry

theorem clock_problem :
  let initial_time : Time := ⟨3, 0, 0⟩
  let final_time := addDuration initial_time 85 58 30
  final_time = ⟨4, 58, 30⟩ ∧ timeSum final_time = 92 := by sorry

end NUMINAMATH_CALUDE_clock_problem_l347_34738


namespace NUMINAMATH_CALUDE_mark_born_in_1978_l347_34786

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1985

/-- The year Mark took the ninth AMC 8 -/
def ninth_amc8_year : ℕ := first_amc8_year + 8

/-- Mark's age when he took the ninth AMC 8 -/
def marks_age : ℕ := 15

/-- Mark's birth year -/
def marks_birth_year : ℕ := ninth_amc8_year - marks_age

theorem mark_born_in_1978 : marks_birth_year = 1978 := by sorry

end NUMINAMATH_CALUDE_mark_born_in_1978_l347_34786


namespace NUMINAMATH_CALUDE_doll_price_is_five_l347_34703

/-- Represents the inventory and financial data of Stella's antique shop --/
structure AntiqueShop where
  num_dolls : ℕ
  num_clocks : ℕ
  num_glasses : ℕ
  clock_price : ℕ
  glass_price : ℕ
  total_cost : ℕ
  total_profit : ℕ

/-- Calculates the price of each doll given the shop's data --/
def calculate_doll_price (shop : AntiqueShop) : ℕ :=
  let total_revenue := shop.total_cost + shop.total_profit
  let clock_revenue := shop.num_clocks * shop.clock_price
  let glass_revenue := shop.num_glasses * shop.glass_price
  let doll_revenue := total_revenue - clock_revenue - glass_revenue
  doll_revenue / shop.num_dolls

/-- Theorem stating that the doll price is $5 given Stella's shop data --/
theorem doll_price_is_five (shop : AntiqueShop) 
  (h1 : shop.num_dolls = 3)
  (h2 : shop.num_clocks = 2)
  (h3 : shop.num_glasses = 5)
  (h4 : shop.clock_price = 15)
  (h5 : shop.glass_price = 4)
  (h6 : shop.total_cost = 40)
  (h7 : shop.total_profit = 25) :
  calculate_doll_price shop = 5 := by
  sorry

#eval calculate_doll_price {
  num_dolls := 3,
  num_clocks := 2,
  num_glasses := 5,
  clock_price := 15,
  glass_price := 4,
  total_cost := 40,
  total_profit := 25
}

end NUMINAMATH_CALUDE_doll_price_is_five_l347_34703


namespace NUMINAMATH_CALUDE_box_surface_area_l347_34737

/-- Calculates the surface area of the interior of a box formed by removing square corners from a rectangular sheet and folding up the remaining flaps. -/
def interior_surface_area (sheet_length sheet_width corner_size : ℕ) : ℕ :=
  let base_length := sheet_length - 2 * corner_size
  let base_width := sheet_width - 2 * corner_size
  let base_area := base_length * base_width
  let side_area1 := 2 * (base_length * corner_size)
  let side_area2 := 2 * (base_width * corner_size)
  base_area + side_area1 + side_area2

/-- The surface area of the interior of the box is 812 square units. -/
theorem box_surface_area :
  interior_surface_area 28 36 7 = 812 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_l347_34737


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l347_34754

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  S : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- Given conditions for the arithmetic sequence -/
def given_conditions (seq : ArithmeticSequence) : Prop :=
  seq.a 5 + seq.a 9 = -2 ∧ seq.S 3 = 57

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : given_conditions seq) :
  (∀ n : ℕ, seq.a n = 27 - 4 * n) ∧
  (∃ m : ℕ, ∀ n : ℕ, seq.S n ≤ m ∧ seq.S n = m ↔ n = 6) ∧ 
  (∃ m : ℕ, m = 78 ∧ ∀ n : ℕ, seq.S n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l347_34754


namespace NUMINAMATH_CALUDE_magic_rectangle_unique_z_l347_34712

/-- Represents a 3x3 magic rectangle with some fixed values --/
structure MagicRectangle where
  x : ℕ+
  y : ℕ+
  u : ℕ+
  z : ℕ+

/-- The sum of each row and column in the magic rectangle --/
def row_col_sum (m : MagicRectangle) : ℕ :=
  3 + m.x + 21

/-- The magic rectangle property: all rows and columns have the same sum --/
def is_magic_rectangle (m : MagicRectangle) : Prop :=
  (row_col_sum m = m.y + 25 + m.z) ∧
  (row_col_sum m = 15 + m.u + 4) ∧
  (row_col_sum m = 3 + m.y + 15) ∧
  (row_col_sum m = m.x + 25 + m.u) ∧
  (row_col_sum m = 21 + m.z + 4)

theorem magic_rectangle_unique_z :
  ∀ m : MagicRectangle, is_magic_rectangle m → m.z = 20 :=
by sorry

end NUMINAMATH_CALUDE_magic_rectangle_unique_z_l347_34712


namespace NUMINAMATH_CALUDE_ducks_in_other_flock_other_flock_size_l347_34721

/-- Calculates the number of ducks in the other flock given the conditions of the problem -/
theorem ducks_in_other_flock (original_flock : ℕ) (net_increase_per_year : ℕ) (years : ℕ) (combined_flock : ℕ) : ℕ :=
  let final_original_flock := original_flock + net_increase_per_year * years
  combined_flock - final_original_flock

/-- Proves that the number of ducks in the other flock is 150 given the problem conditions -/
theorem other_flock_size :
  ducks_in_other_flock 100 10 5 300 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_other_flock_other_flock_size_l347_34721


namespace NUMINAMATH_CALUDE_two_red_or_blue_marbles_probability_l347_34772

/-- The probability of drawing two marbles consecutively where both are either red or blue
    from a bag containing 5 red, 3 blue, and 7 yellow marbles, with replacement. -/
theorem two_red_or_blue_marbles_probability :
  let red_marbles : ℕ := 5
  let blue_marbles : ℕ := 3
  let yellow_marbles : ℕ := 7
  let total_marbles : ℕ := red_marbles + blue_marbles + yellow_marbles
  let prob_red_or_blue : ℚ := (red_marbles + blue_marbles : ℚ) / total_marbles
  (prob_red_or_blue * prob_red_or_blue) = 64 / 225 := by
  sorry

end NUMINAMATH_CALUDE_two_red_or_blue_marbles_probability_l347_34772


namespace NUMINAMATH_CALUDE_quadratic_expression_minimum_l347_34749

theorem quadratic_expression_minimum :
  ∀ x y : ℝ, 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 ≥ 4 ∧
  ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 3 * y₀^2 - 12 * x₀ + 6 * y₀ + 25 = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_minimum_l347_34749


namespace NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l347_34732

theorem sufficient_condition_for_quadratic_inequality :
  ∀ x : ℝ, x ≥ 3 → x^2 - 2*x - 3 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l347_34732


namespace NUMINAMATH_CALUDE_divisors_of_8_factorial_l347_34783

theorem divisors_of_8_factorial : Nat.card (Nat.divisors (Nat.factorial 8)) = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8_factorial_l347_34783


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l347_34726

/-- Proves that the rate of interest is 7% given the problem conditions -/
theorem interest_rate_calculation (loan_amount interest_paid : ℚ) : 
  loan_amount = 1500 →
  interest_paid = 735 →
  ∃ (rate : ℚ), 
    (interest_paid = loan_amount * rate * rate / 100) ∧
    (rate = 7) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l347_34726


namespace NUMINAMATH_CALUDE_cow_value_increase_l347_34752

/-- Calculates the increase in value of a cow after weight gain -/
theorem cow_value_increase (initial_weight : ℝ) (weight_factor : ℝ) (price_per_pound : ℝ)
  (h1 : initial_weight = 400)
  (h2 : weight_factor = 1.5)
  (h3 : price_per_pound = 3) :
  (initial_weight * weight_factor - initial_weight) * price_per_pound = 600 := by
  sorry

#check cow_value_increase

end NUMINAMATH_CALUDE_cow_value_increase_l347_34752


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l347_34725

/-- Given a geometric sequence {a_n} where a_3 = 9 and a_6 = 243, 
    prove that the general term formula is a_n = 3^(n-1) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a3 : a 3 = 9) 
  (h_a6 : a 6 = 243) : 
  ∀ n : ℕ, a n = 3^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l347_34725


namespace NUMINAMATH_CALUDE_inequality_theorem_l347_34769

theorem inequality_theorem (a b : ℝ) : 
  |2*a - 2| < |a - 4| → |2*b - 2| < |b - 4| → 2*|a + b| < |4 + a*b| := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l347_34769


namespace NUMINAMATH_CALUDE_volume_removed_tetrahedra_l347_34747

/-- The volume of tetrahedra removed from a cube when slicing corners to form octagonal faces -/
theorem volume_removed_tetrahedra (cube_edge : ℝ) (h : cube_edge = 2) :
  let octagon_side := 2 * (Real.sqrt 2 - 1)
  let tetrahedron_height := 2 / Real.sqrt 2
  let base_area := 2 * (3 - 2 * Real.sqrt 2)
  let single_tetrahedron_volume := (1 / 3) * base_area * tetrahedron_height
  8 * single_tetrahedron_volume = (32 * (3 - 2 * Real.sqrt 2)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_volume_removed_tetrahedra_l347_34747


namespace NUMINAMATH_CALUDE_jogger_difference_l347_34767

def jogger_problem (tyson martha alexander christopher natasha : ℕ) : Prop :=
  martha = max 0 (tyson - 15) ∧
  alexander = tyson + 22 ∧
  christopher = 20 * tyson ∧
  natasha = 2 * (martha + alexander) ∧
  christopher = 80

theorem jogger_difference (tyson martha alexander christopher natasha : ℕ) 
  (h : jogger_problem tyson martha alexander christopher natasha) : 
  christopher - natasha = 28 := by
sorry

end NUMINAMATH_CALUDE_jogger_difference_l347_34767


namespace NUMINAMATH_CALUDE_roses_kept_l347_34766

/-- Given that Ian had 20 roses and gave away specific numbers to different people,
    prove that he kept exactly 1 rose. -/
theorem roses_kept (total : ℕ) (mother grandmother sister : ℕ)
    (h1 : total = 20)
    (h2 : mother = 6)
    (h3 : grandmother = 9)
    (h4 : sister = 4) :
    total - (mother + grandmother + sister) = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_kept_l347_34766


namespace NUMINAMATH_CALUDE_range_of_m_l347_34790

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x ≥ 2 → x^2 - 2*x + 1 ≥ m) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l347_34790


namespace NUMINAMATH_CALUDE_direction_vector_form_l347_34789

/-- Given a line passing through two points, prove that its direction vector
    has a specific form. -/
theorem direction_vector_form (p₁ p₂ : ℝ × ℝ) (b : ℝ) : 
  p₁ = (-3, 4) →
  p₂ = (2, -1) →
  (p₂.1 - p₁.1, p₂.2 - p₁.2) = (b * (p₂.2 - p₁.2), p₂.2 - p₁.2) →
  b = 1 := by
  sorry

#check direction_vector_form

end NUMINAMATH_CALUDE_direction_vector_form_l347_34789


namespace NUMINAMATH_CALUDE_inequality_solution_set_l347_34745

theorem inequality_solution_set (x : ℝ) :
  (x - 3) / (x^2 - 2*x + 11) ≥ 0 ↔ x ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l347_34745


namespace NUMINAMATH_CALUDE_unique_number_l347_34750

theorem unique_number : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_l347_34750


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l347_34753

/-- A triangle with marked points on two sides -/
structure MarkedTriangle where
  -- The number of points marked on side BC
  pointsOnBC : ℕ
  -- The number of points marked on side AB
  pointsOnAB : ℕ
  -- Ensure the points are distinct from vertices
  distinctPoints : pointsOnBC > 0 ∧ pointsOnAB > 0

/-- The number of intersection points formed by connecting marked points -/
def intersectionPoints (t : MarkedTriangle) : ℕ := t.pointsOnBC * t.pointsOnAB

/-- Theorem: The number of intersection points in a triangle with 60 points on BC and 50 points on AB is 3000 -/
theorem intersection_points_theorem (t : MarkedTriangle) 
  (h1 : t.pointsOnBC = 60) (h2 : t.pointsOnAB = 50) : 
  intersectionPoints t = 3000 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l347_34753


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l347_34733

/-- A regular polygon with side length 8 and exterior angle 90 degrees has a perimeter of 32 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 90 ∧ 
  (n : ℝ) * exterior_angle = 360 →
  n * side_length = 32 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l347_34733


namespace NUMINAMATH_CALUDE_train_arrangement_count_l347_34728

/-- Represents the number of trains -/
def total_trains : ℕ := 8

/-- Represents the number of trains in each group -/
def trains_per_group : ℕ := 4

/-- Calculates the number of ways to arrange the trains according to the given conditions -/
def train_arrangements : ℕ := sorry

/-- Theorem stating that the number of train arrangements is 720 -/
theorem train_arrangement_count : train_arrangements = 720 := by sorry

end NUMINAMATH_CALUDE_train_arrangement_count_l347_34728


namespace NUMINAMATH_CALUDE_geometric_mean_sqrt3_plus_minus_one_l347_34727

theorem geometric_mean_sqrt3_plus_minus_one : 
  ∃ (x : ℝ), x^2 = (Real.sqrt 3 - 1) * (Real.sqrt 3 + 1) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_sqrt3_plus_minus_one_l347_34727


namespace NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l347_34730

/-- The minimum volume of a cone containing a sphere of radius 1 that touches the base of the cone -/
theorem min_cone_volume_with_sphere (h r : ℝ) : 
  h > 0 → r > 0 → (1 : ℝ) ≤ h →
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x^2 + (y - 1)^2 = r^2 ∧ y = h - 1) →
  (1/3 * π * r^2 * h) ≥ 8*π/3 :=
by sorry

end NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l347_34730


namespace NUMINAMATH_CALUDE_solution_set_rational_inequality_l347_34702

theorem solution_set_rational_inequality :
  ∀ x : ℝ, x ≠ 0 → ((x - 1) / x ≥ 2 ↔ -1 ≤ x ∧ x < 0) := by sorry

end NUMINAMATH_CALUDE_solution_set_rational_inequality_l347_34702


namespace NUMINAMATH_CALUDE_problem_statement_l347_34762

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l347_34762


namespace NUMINAMATH_CALUDE_grapes_purchased_l347_34700

/-- The problem of calculating the amount of grapes purchased -/
theorem grapes_purchased (grape_cost mango_cost total_paid : ℕ) (mango_amount : ℕ) : 
  grape_cost = 70 →
  mango_amount = 9 →
  mango_cost = 65 →
  total_paid = 1145 →
  ∃ (grape_amount : ℕ), grape_amount * grape_cost + mango_amount * mango_cost = total_paid ∧ grape_amount = 8 :=
by sorry

end NUMINAMATH_CALUDE_grapes_purchased_l347_34700


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l347_34724

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + 2*x - 8 ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l347_34724


namespace NUMINAMATH_CALUDE_point_c_not_in_region_point_a_in_region_point_b_in_region_point_d_in_region_main_result_l347_34740

/-- Defines the plane region x + y - 1 ≤ 0 -/
def in_plane_region (x y : ℝ) : Prop := x + y - 1 ≤ 0

/-- The point (-1,3) is not in the plane region -/
theorem point_c_not_in_region : ¬ in_plane_region (-1) 3 := by sorry

/-- Point A (0,0) is in the plane region -/
theorem point_a_in_region : in_plane_region 0 0 := by sorry

/-- Point B (-1,1) is in the plane region -/
theorem point_b_in_region : in_plane_region (-1) 1 := by sorry

/-- Point D (2,-3) is in the plane region -/
theorem point_d_in_region : in_plane_region 2 (-3) := by sorry

/-- The main theorem combining all results -/
theorem main_result : 
  ¬ in_plane_region (-1) 3 ∧ 
  in_plane_region 0 0 ∧ 
  in_plane_region (-1) 1 ∧ 
  in_plane_region 2 (-3) := by sorry

end NUMINAMATH_CALUDE_point_c_not_in_region_point_a_in_region_point_b_in_region_point_d_in_region_main_result_l347_34740


namespace NUMINAMATH_CALUDE_special_integers_characterization_l347_34756

/-- The set of integers that are divisible by all integers not exceeding their square root -/
def SpecialIntegers : Set ℕ :=
  {n : ℕ | ∀ m : ℕ, m ≤ Real.sqrt n → n % m = 0}

/-- Theorem stating that SpecialIntegers is equal to the set {2, 4, 6, 8, 12, 24} -/
theorem special_integers_characterization :
  SpecialIntegers = {2, 4, 6, 8, 12, 24} := by
  sorry


end NUMINAMATH_CALUDE_special_integers_characterization_l347_34756


namespace NUMINAMATH_CALUDE_true_discount_example_l347_34734

/-- Given a banker's discount and sum due, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / sum_due)

/-- Theorem stating that for a banker's discount of 18 and sum due of 90, the true discount is 15 -/
theorem true_discount_example : true_discount 18 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_example_l347_34734


namespace NUMINAMATH_CALUDE_absolute_difference_60th_terms_l347_34748

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem absolute_difference_60th_terms : 
  let C := arithmetic_sequence 25 15
  let D := arithmetic_sequence 40 (-15)
  |C 60 - D 60| = 1755 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_60th_terms_l347_34748


namespace NUMINAMATH_CALUDE_shortest_handspan_l347_34751

def sangwon_handspan : ℝ := 19 + 0.8
def doyoon_handspan : ℝ := 18.9
def changhyeok_handspan : ℝ := 19.3

theorem shortest_handspan :
  doyoon_handspan < sangwon_handspan ∧ doyoon_handspan < changhyeok_handspan :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_handspan_l347_34751


namespace NUMINAMATH_CALUDE_unique_perfect_between_primes_l347_34796

/-- A number is perfect if the sum of its positive divisors equals twice the number. -/
def IsPerfect (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 2 * n

/-- The theorem stating that 6 is the only perfect number n such that n-1 and n+1 are prime. -/
theorem unique_perfect_between_primes :
  ∀ n : ℕ, IsPerfect n ∧ Nat.Prime (n - 1) ∧ Nat.Prime (n + 1) → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_perfect_between_primes_l347_34796


namespace NUMINAMATH_CALUDE_extra_eyes_percentage_l347_34713

def total_frogs : ℕ := 150
def extra_eyes : ℕ := 5

def percentage_with_extra_eyes : ℚ :=
  (extra_eyes : ℚ) / (total_frogs : ℚ) * 100

def rounded_percentage : ℕ := 
  (percentage_with_extra_eyes + 1/2).floor.toNat

theorem extra_eyes_percentage :
  rounded_percentage = 3 :=
sorry

end NUMINAMATH_CALUDE_extra_eyes_percentage_l347_34713


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l347_34765

/-- Given a parabola and two points on it, prove the y-intercept of the line through these points -/
theorem parabola_line_intersection (A B : ℝ × ℝ) (a : ℝ) : 
  (A.1^2 = A.2) →  -- A is on the parabola
  (B.1^2 = B.2) →  -- B is on the parabola
  (A.1 < 0) →  -- A is on the left side of y-axis
  (B.1 > 0) →  -- B is on the right side of y-axis
  (∃ k : ℝ, A.2 = k * A.1 + a ∧ B.2 = k * B.1 + a) →  -- Line AB has equation y = kx + a
  (A.1 * B.1 + A.2 * B.2 > 0) →  -- ∠AOB is acute
  a > 1 := by
sorry


end NUMINAMATH_CALUDE_parabola_line_intersection_l347_34765


namespace NUMINAMATH_CALUDE_fraction_decomposition_l347_34798

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 ≠ -1) :
  (-x^3 + 4*x^2 - 5*x + 3) / (x^4 + x^2) = 3/x^2 + (4*x + 1)/(x^2 + 1) - 5/x :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l347_34798


namespace NUMINAMATH_CALUDE_ceiling_of_negative_fraction_squared_l347_34797

theorem ceiling_of_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_fraction_squared_l347_34797


namespace NUMINAMATH_CALUDE_travel_time_difference_proof_l347_34722

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The lengths of the four detours on Telegraph Road in kilometers -/
def telegraph_detours : List ℝ := [5.2, 2.7, 3.8, 4.4]

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- The increase in length of Pardee Road due to road work in kilometers -/
def pardee_road_increase : ℝ := 2.5

/-- The constant speed of travel in kilometers per hour -/
def travel_speed : ℝ := 80

/-- The difference in travel time between Telegraph Road and Pardee Road in minutes -/
def travel_time_difference : ℝ := 122.7

theorem travel_time_difference_proof :
  let telegraph_total := telegraph_road_length + (telegraph_detours.sum)
  let pardee_total := (pardee_road_length / 1000) + pardee_road_increase
  let telegraph_time := (telegraph_total / travel_speed) * 60
  let pardee_time := (pardee_total / travel_speed) * 60
  telegraph_time - pardee_time = travel_time_difference := by
  sorry

end NUMINAMATH_CALUDE_travel_time_difference_proof_l347_34722


namespace NUMINAMATH_CALUDE_parabola_comparison_l347_34714

theorem parabola_comparison : ∀ x : ℝ, x^2 - x + 3 < x^2 - x + 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_comparison_l347_34714


namespace NUMINAMATH_CALUDE_car_average_speed_l347_34716

/-- Given a car traveling at different speeds for two hours, 
    calculate its average speed. -/
theorem car_average_speed 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 60) : 
  (speed1 + speed2) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l347_34716


namespace NUMINAMATH_CALUDE_solution_set_inequality_l347_34746

theorem solution_set_inequality (x : ℝ) :
  x * (x - 1) < 0 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l347_34746


namespace NUMINAMATH_CALUDE_f_has_unique_zero_l347_34731

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x + (a / 2) * x^2

theorem f_has_unique_zero (a : ℝ) (h : a ∈ Set.Icc (-Real.exp 1) 0) :
  ∃! x, f a x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_l347_34731


namespace NUMINAMATH_CALUDE_equation_solution_l347_34707

theorem equation_solution : ∃! x : ℝ, (x^2 + x)^2 + Real.sqrt (x^2 - 1) = 0 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l347_34707


namespace NUMINAMATH_CALUDE_remainder_sum_modulo_l347_34776

theorem remainder_sum_modulo (p q : ℤ) 
  (hp : p % 98 = 84) 
  (hq : q % 126 = 117) : 
  (p + q) % 42 = 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_modulo_l347_34776


namespace NUMINAMATH_CALUDE_triangle_properties_l347_34742

/-- Given a triangle ABC where sides a and b are roots of x^2 - 2√3x + 2 = 0,
    and cos(A + B) = 1/2, prove the following properties -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (∃ x y : ℝ, x^2 - 2 * Real.sqrt 3 * x + 2 = 0 ∧ y^2 - 2 * Real.sqrt 3 * y + 2 = 0 ∧ x = a ∧ y = b) →
  Real.cos (A + B) = 1/2 →
  C = Real.pi * 2/3 ∧
  (a^2 + b^2 - 2*a*b*Real.cos C) = 10 ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l347_34742


namespace NUMINAMATH_CALUDE_smallest_b_for_even_polynomial_l347_34791

theorem smallest_b_for_even_polynomial : ∃ (b : ℕ+), 
  (∀ (x : ℤ), ∃ (k : ℤ), x^4 + (b : ℤ)^3 + (b : ℤ)^2 = 2 * k) ∧ 
  (∀ (b' : ℕ+), b' < b → ∃ (x : ℤ), ∀ (k : ℤ), x^4 + (b' : ℤ)^3 + (b' : ℤ)^2 ≠ 2 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_even_polynomial_l347_34791
