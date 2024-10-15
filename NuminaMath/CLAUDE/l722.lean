import Mathlib

namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l722_72219

def angle : ℝ := 2017

theorem point_in_third_quadrant :
  let x := Real.cos (angle * π / 180)
  let y := Real.sin (angle * π / 180)
  x < 0 ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l722_72219


namespace NUMINAMATH_CALUDE_soccer_team_games_l722_72293

/-- Calculates the number of games played by a soccer team based on pizza slices and goals scored. -/
theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (goals_per_game : ℕ) 
  (h1 : pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : goals_per_game = 9)
  (h4 : pizzas * slices_per_pizza = goals_per_game * (pizzas * slices_per_pizza / goals_per_game)) :
  pizzas * slices_per_pizza / goals_per_game = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_l722_72293


namespace NUMINAMATH_CALUDE_det_equality_l722_72220

theorem det_equality (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 7 →
  Matrix.det !![x - 2*z, y - 2*w; z, w] = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_equality_l722_72220


namespace NUMINAMATH_CALUDE_number_division_sum_l722_72204

theorem number_division_sum (x : ℝ) : (x / 2 + x + 2 = 62) ↔ (x = 40) := by
  sorry

end NUMINAMATH_CALUDE_number_division_sum_l722_72204


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l722_72224

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number d such that ax^2 + bx + c = (dx + e)^2 for all x -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ d e : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (d * x + e)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, is_perfect_square_trinomial 1 (-4) m → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l722_72224


namespace NUMINAMATH_CALUDE_lara_today_cans_l722_72270

/-- The number of cans collected by Sarah and Lara over two days -/
structure CanCollection where
  sarah_yesterday : ℕ
  lara_yesterday : ℕ
  sarah_today : ℕ
  lara_today : ℕ

/-- The conditions of the can collection problem -/
def can_collection_problem (c : CanCollection) : Prop :=
  c.sarah_yesterday = 50 ∧
  c.lara_yesterday = c.sarah_yesterday + 30 ∧
  c.sarah_today = 40 ∧
  c.sarah_today + c.lara_today = c.sarah_yesterday + c.lara_yesterday - 20

/-- The theorem stating Lara collected 70 cans today -/
theorem lara_today_cans (c : CanCollection) :
  can_collection_problem c → c.lara_today = 70 := by
  sorry

end NUMINAMATH_CALUDE_lara_today_cans_l722_72270


namespace NUMINAMATH_CALUDE_martha_reading_challenge_l722_72254

def pages_read : List Nat := [12, 18, 14, 20, 11, 13, 19, 15, 17]
def total_days : Nat := 10
def target_average : Nat := 15

theorem martha_reading_challenge :
  ∃ (x : Nat), 
    (List.sum pages_read + x) / total_days = target_average ∧
    x = 11 := by
  sorry

end NUMINAMATH_CALUDE_martha_reading_challenge_l722_72254


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l722_72292

/-- Rounds a number to the nearest multiple of 5, rounding 5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  ((n + 2) / 5) * 5

/-- Sums all integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums all integers from 1 to n after rounding each to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  List.sum (List.map roundToNearestFive (List.range n))

theorem sum_difference_theorem :
  sumToN 100 - sumRoundedToN 100 = 4750 :=
sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l722_72292


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l722_72259

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : Real
  length : Real

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : Real :=
  d.width * d.length

/-- Calculates the perimeter of a rectangular garden -/
def gardenPerimeter (d : GardenDimensions) : Real :=
  2 * (d.width + d.length)

/-- Theorem: Optimal dimensions for a rectangular garden with minimum fencing -/
theorem optimal_garden_dimensions :
  ∃ (d : GardenDimensions),
    d.length = 2 * d.width ∧
    gardenArea d ≥ 500 ∧
    (∀ (d' : GardenDimensions),
      d'.length = 2 * d'.width →
      gardenArea d' ≥ 500 →
      gardenPerimeter d ≤ gardenPerimeter d') ∧
    d.width = 5 * Real.sqrt 10 ∧
    d.length = 10 * Real.sqrt 10 ∧
    gardenPerimeter d = 30 * Real.sqrt 10 :=
  sorry


end NUMINAMATH_CALUDE_optimal_garden_dimensions_l722_72259


namespace NUMINAMATH_CALUDE_problem_statement_l722_72297

/-- Given real numbers a and b satisfying a + 2b = 9, prove:
    1. If |9 - 2b| + |a + 1| < 3, then -2 < a < 1.
    2. If a > 0, b > 0, and z = ab^2, then the maximum value of z is 27. -/
theorem problem_statement (a b : ℝ) (h1 : a + 2*b = 9) :
  (|9 - 2*b| + |a + 1| < 3 → -2 < a ∧ a < 1) ∧
  (a > 0 → b > 0 → ∃ z : ℝ, z = a*b^2 ∧ ∀ w : ℝ, w = a*b^2 → w ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l722_72297


namespace NUMINAMATH_CALUDE_base_subtraction_l722_72291

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The theorem statement --/
theorem base_subtraction :
  to_base_10 [3, 2, 5] 9 - to_base_10 [2, 3, 1] 6 = 175 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l722_72291


namespace NUMINAMATH_CALUDE_square_root_of_two_l722_72250

theorem square_root_of_two : Real.sqrt 2 = (Real.sqrt 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_two_l722_72250


namespace NUMINAMATH_CALUDE_mitchell_gum_packets_l722_72214

theorem mitchell_gum_packets (pieces_per_packet : ℕ) (pieces_chewed : ℕ) (pieces_left : ℕ) : 
  pieces_per_packet = 7 →
  pieces_left = 2 →
  pieces_chewed = 54 →
  (pieces_chewed + pieces_left) / pieces_per_packet = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mitchell_gum_packets_l722_72214


namespace NUMINAMATH_CALUDE_distribute_five_three_l722_72261

/-- The number of ways to distribute n distinct elements into k distinct groups,
    where each group must contain at least one element. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinct elements into 3 distinct groups,
    where each group must contain at least one element. -/
theorem distribute_five_three : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_three_l722_72261


namespace NUMINAMATH_CALUDE_min_nSn_value_l722_72217

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (2 * a 0 + (n - 1) * (a 1 - a 0)) / 2

/-- The main theorem stating the minimum value of nS_n -/
theorem min_nSn_value (seq : ArithmeticSequence) 
    (h1 : seq.S 10 = 0) 
    (h2 : seq.S 15 = 25) : 
  ∃ n : ℕ, ∀ m : ℕ, n * seq.S n ≤ m * seq.S m ∧ n * seq.S n = -49 := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_value_l722_72217


namespace NUMINAMATH_CALUDE_max_triangle_area_l722_72244

/-- Ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle type -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to check if a point is on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop := sorry

/-- Function to check if a line intersects an ellipse at two distinct points -/
def intersectsAtTwoPoints (l : Line) (e : Ellipse) : Prop := sorry

/-- Theorem statement -/
theorem max_triangle_area 
  (e : Ellipse) 
  (h_eccentricity : e.a^2 - e.b^2 = e.a^2 / 2)
  (A : Point)
  (h_A_on_ellipse : isOnEllipse A e)
  (h_A_coords : A.x = 1 ∧ A.y = Real.sqrt 2)
  (l : Line)
  (h_l_slope : l.slope = Real.sqrt 2)
  (h_intersects : intersectsAtTwoPoints l e) :
  ∃ (B C : Point), 
    isOnEllipse B e ∧ 
    isOnEllipse C e ∧ 
    B ≠ C ∧
    ∀ (B' C' : Point), 
      isOnEllipse B' e → 
      isOnEllipse C' e → 
      B' ≠ C' →
      triangleArea ⟨A, B', C'⟩ ≤ Real.sqrt 2 ∧
      triangleArea ⟨A, B, C⟩ = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_max_triangle_area_l722_72244


namespace NUMINAMATH_CALUDE_rayden_vs_lily_birds_l722_72231

theorem rayden_vs_lily_birds (lily_ducks lily_geese lily_chickens lily_pigeons : ℕ)
  (rayden_ducks rayden_geese rayden_chickens rayden_pigeons : ℕ)
  (h1 : lily_ducks = 20)
  (h2 : lily_geese = 10)
  (h3 : lily_chickens = 5)
  (h4 : lily_pigeons = 30)
  (h5 : rayden_ducks = 3 * lily_ducks)
  (h6 : rayden_geese = 4 * lily_geese)
  (h7 : rayden_chickens = 5 * lily_chickens)
  (h8 : lily_pigeons = 2 * rayden_pigeons) :
  (rayden_ducks + rayden_geese + rayden_chickens + rayden_pigeons) -
  (lily_ducks + lily_geese + lily_chickens + lily_pigeons) = 75 :=
by sorry

end NUMINAMATH_CALUDE_rayden_vs_lily_birds_l722_72231


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l722_72274

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l722_72274


namespace NUMINAMATH_CALUDE_expected_count_in_sample_l722_72242

/-- 
Given a population where 1/4 of the members have a certain characteristic,
prove that the expected number of individuals with that characteristic
in a random sample of 300 is 75.
-/
theorem expected_count_in_sample 
  (population_probability : ℚ) 
  (sample_size : ℕ) 
  (h1 : population_probability = 1 / 4) 
  (h2 : sample_size = 300) : 
  population_probability * sample_size = 75 := by
sorry

end NUMINAMATH_CALUDE_expected_count_in_sample_l722_72242


namespace NUMINAMATH_CALUDE_counterexample_exists_l722_72236

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l722_72236


namespace NUMINAMATH_CALUDE_barn_paint_area_l722_72235

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular opening (door or window) -/
structure OpeningDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a barn -/
def totalPaintArea (barn : BarnDimensions) (doors : List OpeningDimensions) (windows : List OpeningDimensions) : ℝ :=
  let wallArea := 2 * (barn.width * barn.height + barn.length * barn.height)
  let floorCeilingArea := 2 * (barn.width * barn.length)
  let doorArea := doors.map (fun d => d.width * d.height) |>.sum
  let windowArea := windows.map (fun w => w.width * w.height) |>.sum
  2 * (wallArea - doorArea - windowArea) + floorCeilingArea

/-- Theorem stating that the total area to be painted is 1588 sq yd -/
theorem barn_paint_area :
  let barn := BarnDimensions.mk 15 20 8
  let doors := [OpeningDimensions.mk 3 7, OpeningDimensions.mk 3 7]
  let windows := [OpeningDimensions.mk 2 4, OpeningDimensions.mk 2 4, OpeningDimensions.mk 2 4]
  totalPaintArea barn doors windows = 1588 := by
  sorry

end NUMINAMATH_CALUDE_barn_paint_area_l722_72235


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l722_72206

/-- A quadratic function with a positive constant term -/
def f (a b k : ℝ) (hk : k > 0) (x : ℝ) : ℝ := a * x^2 + b * x + k

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_sum_of_coefficients 
  (a b k : ℝ) (hk : k > 0) : 
  (f' a b 0 = 0) → 
  (f' a b 1 = 2) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l722_72206


namespace NUMINAMATH_CALUDE_even_odd_square_sum_l722_72211

theorem even_odd_square_sum (a b : ℕ) :
  (Even (a * b) → ∃ c d : ℕ, a^2 + b^2 + c^2 = d^2) ∧
  (Odd (a * b) → ¬∃ c d : ℕ, a^2 + b^2 + c^2 = d^2) :=
by sorry

end NUMINAMATH_CALUDE_even_odd_square_sum_l722_72211


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l722_72227

/-- Represents the allocation of a budget in a circle graph --/
def BudgetAllocation (total : ℝ) (allocated : ℝ) (degreesPerPercent : ℝ) : Prop :=
  total = 100 ∧ 
  allocated = 95 ∧ 
  degreesPerPercent = 360 / 100

/-- Theorem: The number of degrees representing the remaining budget (basic astrophysics) is 18 --/
theorem basic_astrophysics_degrees 
  (total allocated remaining : ℝ) 
  (degreesPerPercent : ℝ) 
  (h : BudgetAllocation total allocated degreesPerPercent) :
  remaining = 18 :=
sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l722_72227


namespace NUMINAMATH_CALUDE_football_practice_missed_days_l722_72212

/-- Calculates the number of days a football team missed practice due to rain. -/
theorem football_practice_missed_days
  (daily_hours : ℕ)
  (total_hours : ℕ)
  (days_in_week : ℕ)
  (h1 : daily_hours = 5)
  (h2 : total_hours = 30)
  (h3 : days_in_week = 7) :
  days_in_week - (total_hours / daily_hours) = 1 :=
by sorry

end NUMINAMATH_CALUDE_football_practice_missed_days_l722_72212


namespace NUMINAMATH_CALUDE_area_enclosed_by_cosine_curve_l722_72260

theorem area_enclosed_by_cosine_curve : 
  let f (x : ℝ) := Real.cos x
  let area := ∫ x in (0)..(π/2), f x - ∫ x in (π/2)..(3*π/2), f x
  area = 3 := by
sorry

end NUMINAMATH_CALUDE_area_enclosed_by_cosine_curve_l722_72260


namespace NUMINAMATH_CALUDE_nested_rectangles_exist_l722_72237

/-- Represents a rectangle with integer sides --/
structure Rectangle where
  width : Nat
  height : Nat
  width_bound : width ≤ 100
  height_bound : height ≤ 100

/-- Checks if rectangle a can be nested inside rectangle b --/
def can_nest (a b : Rectangle) : Prop :=
  a.width ≤ b.width ∧ a.height ≤ b.height

theorem nested_rectangles_exist (rectangles : Finset Rectangle) 
  (h : rectangles.card = 101) :
  ∃ (A B C : Rectangle), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles ∧
    can_nest A B ∧ can_nest B C := by
  sorry

end NUMINAMATH_CALUDE_nested_rectangles_exist_l722_72237


namespace NUMINAMATH_CALUDE_average_weight_increase_l722_72269

theorem average_weight_increase (initial_count : ℕ) (replaced_weight original_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  original_weight = 85 →
  (original_weight - replaced_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l722_72269


namespace NUMINAMATH_CALUDE_smallest_multiple_l722_72247

theorem smallest_multiple : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (∃ (k : ℕ), a = 5 * k) ∧ 
  (∃ (m : ℕ), a + 1 = 7 * m) ∧ 
  (∃ (n : ℕ), a + 2 = 9 * n) ∧ 
  (∃ (p : ℕ), a + 3 = 11 * p) ∧ 
  (∀ (b : ℕ), 
    (b > 0) ∧ 
    (∃ (k : ℕ), b = 5 * k) ∧ 
    (∃ (m : ℕ), b + 1 = 7 * m) ∧ 
    (∃ (n : ℕ), b + 2 = 9 * n) ∧ 
    (∃ (p : ℕ), b + 3 = 11 * p) 
    → b ≥ a) ∧
  a = 1735 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l722_72247


namespace NUMINAMATH_CALUDE_score_difference_l722_72240

/-- Represents the test scores of three students -/
structure TestScores where
  meghan : ℕ
  jose : ℕ
  alisson : ℕ

/-- The properties of the test and scores -/
def ValidTestScores (s : TestScores) : Prop :=
  let totalQuestions : ℕ := 50
  let marksPerQuestion : ℕ := 2
  let maxScore : ℕ := totalQuestions * marksPerQuestion
  let wrongQuestions : ℕ := 5
  (s.jose = maxScore - wrongQuestions * marksPerQuestion) ∧ 
  (s.jose = s.alisson + 40) ∧
  (s.meghan + s.jose + s.alisson = 210) ∧
  (s.meghan < s.jose)

/-- The theorem stating the difference between Jose's and Meghan's scores -/
theorem score_difference (s : TestScores) (h : ValidTestScores s) : 
  s.jose - s.meghan = 20 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l722_72240


namespace NUMINAMATH_CALUDE_sum_of_generated_numbers_eq_5994_l722_72207

/-- The sum of all three-digit natural numbers created using digits 1, 2, and 3 -/
def sum_three_digit_numbers : ℕ := 5994

/-- The set of digits that can be used -/
def valid_digits : Finset ℕ := {1, 2, 3}

/-- A function to generate all possible three-digit numbers using the valid digits -/
def generate_numbers : Finset ℕ := sorry

/-- Theorem stating that the sum of all generated numbers equals sum_three_digit_numbers -/
theorem sum_of_generated_numbers_eq_5994 : 
  (generate_numbers.sum id) = sum_three_digit_numbers := by sorry

end NUMINAMATH_CALUDE_sum_of_generated_numbers_eq_5994_l722_72207


namespace NUMINAMATH_CALUDE_paul_initial_books_l722_72200

/-- The number of books Paul sold in the garage sale -/
def books_sold : ℕ := 78

/-- The number of books Paul has left after the sale -/
def books_left : ℕ := 37

/-- The initial number of books Paul had -/
def initial_books : ℕ := books_sold + books_left

theorem paul_initial_books : initial_books = 115 := by
  sorry

end NUMINAMATH_CALUDE_paul_initial_books_l722_72200


namespace NUMINAMATH_CALUDE_yellow_bags_count_l722_72294

/-- Represents the number of marbles in each type of bag -/
def marbles_per_bag : Fin 3 → ℕ
  | 0 => 10  -- Red bags
  | 1 => 50  -- Blue bags
  | 2 => 100 -- Yellow bags
  | _ => 0   -- This case is unreachable due to Fin 3

/-- The total number of bags -/
def total_bags : ℕ := 12

/-- The total number of marbles -/
def total_marbles : ℕ := 500

theorem yellow_bags_count :
  ∃ (red blue yellow : ℕ),
    red + blue + yellow = total_bags ∧
    red * marbles_per_bag 0 + blue * marbles_per_bag 1 + yellow * marbles_per_bag 2 = total_marbles ∧
    red = blue ∧
    yellow = 2 := by sorry

end NUMINAMATH_CALUDE_yellow_bags_count_l722_72294


namespace NUMINAMATH_CALUDE_william_journey_time_l722_72287

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Calculates the difference between two times in hours -/
def timeDifferenceInHours (t1 t2 : Time) : ℚ :=
  (t2.hours - t1.hours : ℚ) + (t2.minutes - t1.minutes : ℚ) / 60

/-- Represents a journey with stops and delays -/
structure Journey where
  departureTime : Time
  arrivalTime : Time
  timeZoneDifference : ℕ
  stops : List ℕ
  trafficDelay : ℕ

theorem william_journey_time (j : Journey) 
  (h1 : j.departureTime = ⟨7, 0, by norm_num⟩)
  (h2 : j.arrivalTime = ⟨20, 0, by norm_num⟩)
  (h3 : j.timeZoneDifference = 2)
  (h4 : j.stops = [25, 10, 25])
  (h5 : j.trafficDelay = 45) :
  timeDifferenceInHours j.departureTime ⟨18, 0, by norm_num⟩ + 
  (j.stops.sum / 60 : ℚ) + (j.trafficDelay / 60 : ℚ) = 12.75 := by
  sorry

#check william_journey_time

end NUMINAMATH_CALUDE_william_journey_time_l722_72287


namespace NUMINAMATH_CALUDE_vacant_seats_l722_72215

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 45/100) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l722_72215


namespace NUMINAMATH_CALUDE_six_fold_application_of_f_on_four_l722_72299

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem six_fold_application_of_f_on_four (h : ∀ (x : ℝ), x ≠ 0 → f x = -1 / x) :
  f (f (f (f (f (f 4))))) = 4 :=
by sorry

end NUMINAMATH_CALUDE_six_fold_application_of_f_on_four_l722_72299


namespace NUMINAMATH_CALUDE_quartic_equation_integer_roots_l722_72202

theorem quartic_equation_integer_roots :
  let f (x : ℤ) (a : ℤ) := x^4 - 16*x^3 + (81-2*a)*x^2 + (16*a-142)*x + a^2 - 21*a + 68
  ∃ (a : ℤ), a = -4 ∧ (∀ x : ℤ, f x a = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 7) := by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_integer_roots_l722_72202


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l722_72226

theorem tangent_line_to_circle (r : ℝ) (h_pos : r > 0) :
  (∀ x y : ℝ, x + y = r → x^2 + y^2 = 4*r → 
    ∀ ε > 0, ∃ x' y' : ℝ, x' + y' = r ∧ (x' - x)^2 + (y' - y)^2 < ε^2 ∧ x'^2 + y'^2 ≠ 4*r) →
  r = 8 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l722_72226


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l722_72241

-- System of equations (1)
theorem system_one_solution (x y : ℝ) : 
  3*x - 2*y = 6 ∧ 2*x + 3*y = 17 → x = 4 ∧ y = 3 := by
sorry

-- System of equations (2)
theorem system_two_solution (x y : ℝ) :
  x + 4*y = 14 ∧ (x-3)/4 - (y-3)/3 = 1/12 → x = 3 ∧ y = 11/4 := by
sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l722_72241


namespace NUMINAMATH_CALUDE_maximize_sum_with_constraint_l722_72271

theorem maximize_sum_with_constraint (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_constraint : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2*b + 3*c ≤ 91/3 :=
by sorry

end NUMINAMATH_CALUDE_maximize_sum_with_constraint_l722_72271


namespace NUMINAMATH_CALUDE_odd_digits_in_560_base9_l722_72284

/-- Converts a natural number from base 10 to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers --/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem odd_digits_in_560_base9 :
  countOddDigits (toBase9 560) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_in_560_base9_l722_72284


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l722_72278

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (geom_seq : b^2 = a * c) 
  (def_a : a = 5 + 2 * Real.sqrt 3) 
  (def_c : c = 5 - 2 * Real.sqrt 3) : 
  b = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l722_72278


namespace NUMINAMATH_CALUDE_trajectory_is_parallel_plane_l722_72248

-- Define the type for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the set of points P satisfying y = 3
def TrajectorySet : Set Point3D :=
  {p : Point3D | p.y = 3}

-- Define a plane parallel to xOz plane
def ParallelPlane (h : ℝ) : Set Point3D :=
  {p : Point3D | p.y = h}

-- Theorem statement
theorem trajectory_is_parallel_plane :
  ∃ h : ℝ, TrajectorySet = ParallelPlane h := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parallel_plane_l722_72248


namespace NUMINAMATH_CALUDE_chairs_moved_by_pat_l722_72251

theorem chairs_moved_by_pat (total_chairs : ℕ) (careys_chairs : ℕ) (chairs_left : ℕ) 
  (h1 : total_chairs = 74)
  (h2 : careys_chairs = 28)
  (h3 : chairs_left = 17) :
  total_chairs - careys_chairs - chairs_left = 29 := by
  sorry

end NUMINAMATH_CALUDE_chairs_moved_by_pat_l722_72251


namespace NUMINAMATH_CALUDE_spinner_divisible_by_three_probability_l722_72281

/-- Represents the possible outcomes of the spinner -/
inductive SpinnerOutcome
  | One
  | Two
  | Four

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinnerOutcome
  tens : SpinnerOutcome
  units : SpinnerOutcome

/-- Converts a SpinnerOutcome to its numerical value -/
def spinnerValue (outcome : SpinnerOutcome) : Nat :=
  match outcome with
  | SpinnerOutcome.One => 1
  | SpinnerOutcome.Two => 2
  | SpinnerOutcome.Four => 4

/-- Checks if a ThreeDigitNumber is divisible by 3 -/
def isDivisibleByThree (n : ThreeDigitNumber) : Bool :=
  (spinnerValue n.hundreds + spinnerValue n.tens + spinnerValue n.units) % 3 = 0

/-- Calculates the probability of getting a number divisible by 3 -/
def probabilityDivisibleByThree : ℚ :=
  let totalOutcomes := 27  -- 3^3
  let favorableOutcomes := 6  -- Counted from the problem
  favorableOutcomes / totalOutcomes

/-- Main theorem: The probability of getting a number divisible by 3 is 2/9 -/
theorem spinner_divisible_by_three_probability :
  probabilityDivisibleByThree = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_spinner_divisible_by_three_probability_l722_72281


namespace NUMINAMATH_CALUDE_points_per_game_l722_72210

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 24 → 
  num_games = 6 → 
  total_points = num_games * points_per_game → 
  points_per_game = 4 := by
sorry

end NUMINAMATH_CALUDE_points_per_game_l722_72210


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l722_72298

/-- Given a sphere O with radius R and a plane perpendicular to a radius OP at its midpoint M,
    intersecting the sphere to form a circle O₁, the volume ratio of the sphere with O₁ as its
    great circle to sphere O is 3/8 * √3. -/
theorem sphere_volume_ratio (R : ℝ) (h : R > 0) : 
  let r := R * (Real.sqrt 3 / 2)
  (4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * R^3) = 3 / 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l722_72298


namespace NUMINAMATH_CALUDE_peter_additional_miles_l722_72218

/-- The number of additional miles Peter runs compared to Andrew each day -/
def additional_miles : ℝ := sorry

/-- Andrew's daily miles -/
def andrew_miles : ℝ := 2

/-- Number of days they run -/
def days : ℕ := 5

/-- Total miles run by both after 5 days -/
def total_miles : ℝ := 35

theorem peter_additional_miles :
  additional_miles = 3 ∧
  days * (andrew_miles + additional_miles) + days * andrew_miles = total_miles :=
sorry

end NUMINAMATH_CALUDE_peter_additional_miles_l722_72218


namespace NUMINAMATH_CALUDE_sale_result_l722_72239

/-- Represents the total number of cases of cat food sold during a sale. -/
def total_cases_sold (first_group : Nat) (second_group : Nat) (third_group : Nat) 
  (first_group_cases : Nat) (second_group_cases : Nat) (third_group_cases : Nat) : Nat :=
  first_group * first_group_cases + second_group * second_group_cases + third_group * third_group_cases

/-- Theorem stating that the total number of cases sold is 40 given the specific customer purchase patterns. -/
theorem sale_result : 
  total_cases_sold 8 4 8 3 2 1 = 40 := by
  sorry

#check sale_result

end NUMINAMATH_CALUDE_sale_result_l722_72239


namespace NUMINAMATH_CALUDE_ratio_equality_l722_72255

theorem ratio_equality : ∃ x : ℝ, (12 : ℝ) / 8 = x / 240 ∧ x = 360 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l722_72255


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l722_72208

theorem rectangle_area_increase (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  let original_area := x * y
  let new_length := 1.2 * x
  let new_width := 1.1 * y
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.32 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l722_72208


namespace NUMINAMATH_CALUDE_quadratic_factorization_l722_72295

theorem quadratic_factorization (a : ℤ) : 
  (∃ m n p q : ℤ, (15 : ℤ) * x^2 + a * x + (15 : ℤ) = (m * x + n) * (p * x + q) ∧ 
   Nat.Prime m.natAbs ∧ Nat.Prime p.natAbs) → 
  ∃ k : ℤ, a = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l722_72295


namespace NUMINAMATH_CALUDE_function_upper_bound_l722_72286

theorem function_upper_bound
  (f : ℝ → ℝ)
  (h1 : ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2))
  (h2 : ∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f x| ≤ M)
  : ∀ (x : ℝ), x ≥ 0 → f x ≤ x^2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_l722_72286


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_2x2_minus_8x_plus_6_l722_72205

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_2x2_minus_8x_plus_6 :
  let f : ℝ → ℝ := λ x => 2*x^2 - 8*x + 6
  let r₁ := (-(-8) + Real.sqrt ((-8)^2 - 4*2*6)) / (2*2)
  let r₂ := (-(-8) - Real.sqrt ((-8)^2 - 4*2*6)) / (2*2)
  r₁ + r₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_2x2_minus_8x_plus_6_l722_72205


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l722_72263

/-- A polynomial that is always a perfect square for integer inputs can be expressed as (dx + e)^2 -/
theorem perfect_square_polynomial
  (a b c : ℤ)
  (h : ∀ (x : ℤ), ∃ (y : ℤ), a * x^2 + b * x + c = y^2) :
  ∃ (d e : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (d * x + e)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l722_72263


namespace NUMINAMATH_CALUDE_square_of_one_forty_four_l722_72275

/-- Represents a number in a given base -/
def BaseRepresentation (n : ℕ) (b : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : ℕ), d₁ < b ∧ d₂ < b ∧ d₃ < b ∧ n = d₁ * b^2 + d₂ * b + d₃

/-- The number 144 in base b -/
def OneFortyFour (b : ℕ) : ℕ := b^2 + 4*b + 4

theorem square_of_one_forty_four (b : ℕ) :
  b > 4 →
  BaseRepresentation (OneFortyFour b) b →
  ∃ k : ℕ, OneFortyFour b = k^2 :=
sorry

end NUMINAMATH_CALUDE_square_of_one_forty_four_l722_72275


namespace NUMINAMATH_CALUDE_jill_age_l722_72280

/-- Represents the ages of individuals in the problem -/
structure Ages where
  gina : ℕ
  helen : ℕ
  ian : ℕ
  jill : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.gina + 4 = ages.helen ∧
  ages.helen = ages.ian + 5 ∧
  ages.jill = ages.ian + 2 ∧
  ages.gina = 18

/-- The theorem stating Jill's age -/
theorem jill_age (ages : Ages) (h : problem_conditions ages) : ages.jill = 19 := by
  sorry

#check jill_age

end NUMINAMATH_CALUDE_jill_age_l722_72280


namespace NUMINAMATH_CALUDE_no_five_points_configuration_l722_72273

-- Define a type for points in space
variable (Point : Type)

-- Define a distance function between two points
variable (dist : Point → Point → ℝ)

-- Define the congruence transformation type
def CongruenceTransformation (Point : Type) := Point → Point

-- First congruence transformation
variable (t1 : CongruenceTransformation Point)

-- Second congruence transformation
variable (t2 : CongruenceTransformation Point)

-- Theorem statement
theorem no_five_points_configuration 
  (A B C D E : Point)
  (h1 : t1 A = B ∧ t1 B = A ∧ t1 C = C ∧ t1 D = D ∧ t1 E = E)
  (h2 : t2 A = B ∧ t2 B = C ∧ t2 C = D ∧ t2 D = E ∧ t2 E = A)
  (h3 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h4 : ∀ X Y : Point, dist (t1 X) (t1 Y) = dist X Y)
  (h5 : ∀ X Y : Point, dist (t2 X) (t2 Y) = dist X Y) :
  False :=
sorry

end NUMINAMATH_CALUDE_no_five_points_configuration_l722_72273


namespace NUMINAMATH_CALUDE_max_log_sum_l722_72213

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 4*y = 40) :
  ∃ (max : ℝ), max = 8 * Real.log 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l722_72213


namespace NUMINAMATH_CALUDE_f_inequality_l722_72272

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_increasing : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y
axiom f_even_shifted : ∀ x, f (x + 2) = f (-x + 2)

-- State the theorem to be proved
theorem f_inequality : f (5/2) > f 1 ∧ f 1 > f (7/2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l722_72272


namespace NUMINAMATH_CALUDE_milk_production_l722_72262

/-- Given x cows with efficiency α producing y gallons in z days,
    calculate the milk production of w cows with efficiency β in v days -/
theorem milk_production
  (x y z w v : ℝ) (α β : ℝ) (hx : x > 0) (hz : z > 0) (hα : α > 0) :
  let production := (β * y * w * v) / (α^2 * x * z)
  production = (β * y * w * v) / (α^2 * x * z) := by
  sorry

end NUMINAMATH_CALUDE_milk_production_l722_72262


namespace NUMINAMATH_CALUDE_line_contains_point_l722_72243

theorem line_contains_point (j : ℝ) : 
  (∀ x y : ℝ, -2 - 3*j*x = 7*y → x = 1/3 ∧ y = -3) → j = 19 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l722_72243


namespace NUMINAMATH_CALUDE_triangle_side_length_l722_72245

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if A = 60°, b = 4, and the area is 2√3, then a = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (A : Real) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  b = 4 →
  S = 2 * Real.sqrt 3 →
  S = 1 / 2 * b * c * Real.sin A →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l722_72245


namespace NUMINAMATH_CALUDE_bills_age_l722_72229

/-- Bill's current age -/
def b : ℕ := 24

/-- Tracy's current age -/
def t : ℕ := 18

/-- Bill's age is one third larger than Tracy's age -/
axiom bill_tracy_relation : b = (4 * t) / 3

/-- In 30 years, Bill's age will be one eighth larger than Tracy's age -/
axiom future_relation : b + 30 = (9 * (t + 30)) / 8

/-- Theorem: Given the age relations between Bill and Tracy, Bill's current age is 24 -/
theorem bills_age : b = 24 := by sorry

end NUMINAMATH_CALUDE_bills_age_l722_72229


namespace NUMINAMATH_CALUDE_square_difference_identity_l722_72289

theorem square_difference_identity : 287 * 287 + 269 * 269 - 2 * 287 * 269 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l722_72289


namespace NUMINAMATH_CALUDE_largest_house_number_l722_72233

def phone_number : List Nat := [4, 3, 1, 7, 8, 2]

def digit_sum (num : List Nat) : Nat :=
  num.sum

def is_distinct (num : List Nat) : Prop :=
  num.length = num.toFinset.card

theorem largest_house_number :
  ∃ (house : List Nat),
    house.length = 5 ∧
    is_distinct house ∧
    digit_sum house = digit_sum phone_number ∧
    (∀ other : List Nat,
      other.length = 5 →
      is_distinct other →
      digit_sum other = digit_sum phone_number →
      house.foldl (fun acc d => acc * 10 + d) 0 ≥
      other.foldl (fun acc d => acc * 10 + d) 0) ∧
    house = [9, 8, 7, 1, 0] :=
sorry

end NUMINAMATH_CALUDE_largest_house_number_l722_72233


namespace NUMINAMATH_CALUDE_intersection_points_correct_l722_72285

/-- The number of intersection points of segments joining m distinct points 
    on the positive x-axis to n distinct points on the positive y-axis, 
    where no three segments are concurrent. -/
def intersectionPoints (m n : ℕ) : ℕ :=
  m * n * (m - 1) * (n - 1) / 4

/-- Theorem stating that the number of intersection points is correct. -/
theorem intersection_points_correct (m n : ℕ) :
  intersectionPoints m n = m * n * (m - 1) * (n - 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_correct_l722_72285


namespace NUMINAMATH_CALUDE_division_problem_l722_72228

theorem division_problem :
  ∃ (dividend : ℕ), 
    dividend = 11889708 ∧ 
    dividend / 12 = 990809 ∧ 
    dividend % 12 = 0 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l722_72228


namespace NUMINAMATH_CALUDE_remainder_101_47_mod_100_l722_72246

theorem remainder_101_47_mod_100 : 101^47 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_47_mod_100_l722_72246


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l722_72276

theorem angle_in_third_quadrant (α : Real) : 
  (π / 2 < α ∧ α < π) → (π < π / 2 + α ∧ π / 2 + α < 3 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l722_72276


namespace NUMINAMATH_CALUDE_triangle_angle_sum_max_l722_72277

theorem triangle_angle_sum_max (A C : Real) (h1 : 0 < A) (h2 : A < 2 * π / 3) (h3 : A + C = 2 * π / 3) :
  let S := (Real.sqrt 3 / 3) * Real.sin A * Real.sin C
  ∃ (max_S : Real), ∀ (A' C' : Real), 
    0 < A' → A' < 2 * π / 3 → A' + C' = 2 * π / 3 → 
    (Real.sqrt 3 / 3) * Real.sin A' * Real.sin C' ≤ max_S ∧
    max_S = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_max_l722_72277


namespace NUMINAMATH_CALUDE_inequality_always_true_l722_72282

theorem inequality_always_true (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l722_72282


namespace NUMINAMATH_CALUDE_melanie_plums_count_l722_72234

/-- The number of plums Melanie picked initially -/
def melanie_picked : ℝ := 7.0

/-- The number of plums Sam gave to Melanie -/
def sam_gave : ℝ := 3.0

/-- The total number of plums Melanie has now -/
def total_plums : ℝ := melanie_picked + sam_gave

theorem melanie_plums_count : total_plums = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_count_l722_72234


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l722_72267

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos (x + y) ≥ Real.cos x + Real.cos y - 1) ↔ 
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l722_72267


namespace NUMINAMATH_CALUDE_probability_one_doctor_one_nurse_l722_72296

/-- The probability of selecting exactly 1 doctor and 1 nurse from a group of 3 doctors and 2 nurses, when choosing 2 people randomly. -/
theorem probability_one_doctor_one_nurse :
  let total_people : ℕ := 5
  let doctors : ℕ := 3
  let nurses : ℕ := 2
  let selection : ℕ := 2
  Nat.choose total_people selection ≠ 0 →
  (Nat.choose doctors 1 * Nat.choose nurses 1 : ℚ) / Nat.choose total_people selection = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_doctor_one_nurse_l722_72296


namespace NUMINAMATH_CALUDE_wash_time_proof_l722_72222

/-- The number of weeks between each wash -/
def wash_interval : ℕ := 4

/-- The time in minutes it takes to wash the pillowcases -/
def wash_time : ℕ := 30

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Calculates the total time spent washing pillowcases in a year -/
def total_wash_time_per_year : ℕ :=
  (weeks_per_year / wash_interval) * wash_time

theorem wash_time_proof :
  total_wash_time_per_year = 390 :=
by sorry

end NUMINAMATH_CALUDE_wash_time_proof_l722_72222


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l722_72232

theorem surrounding_circles_radius (r : ℝ) : r = 2 * (Real.sqrt 2 + 1) :=
  let central_radius := 2
  let square_side := 2 * r
  let square_diagonal := square_side * Real.sqrt 2
  let total_diagonal := 2 * central_radius + 2 * r
by
  sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l722_72232


namespace NUMINAMATH_CALUDE_max_donuts_is_17_seventeen_donuts_possible_l722_72253

-- Define the prices and budget
def single_price : ℕ := 1
def pack4_price : ℕ := 3
def pack8_price : ℕ := 5
def budget : ℕ := 11

-- Define a function to calculate the number of donuts for a given combination
def donut_count (singles pack4 pack8 : ℕ) : ℕ :=
  singles + 4 * pack4 + 8 * pack8

-- Define a function to calculate the total cost for a given combination
def total_cost (singles pack4 pack8 : ℕ) : ℕ :=
  singles * single_price + pack4 * pack4_price + pack8 * pack8_price

-- Theorem stating that 17 is the maximum number of donuts that can be purchased
theorem max_donuts_is_17 :
  ∀ (singles pack4 pack8 : ℕ),
    total_cost singles pack4 pack8 ≤ budget →
    donut_count singles pack4 pack8 ≤ 17 :=
by
  sorry

-- Theorem stating that 17 donuts can actually be purchased
theorem seventeen_donuts_possible :
  ∃ (singles pack4 pack8 : ℕ),
    total_cost singles pack4 pack8 ≤ budget ∧
    donut_count singles pack4 pack8 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_max_donuts_is_17_seventeen_donuts_possible_l722_72253


namespace NUMINAMATH_CALUDE_odd_product_remainder_l722_72290

def odd_product : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then odd_product n else (2 * n + 1) * odd_product n

theorem odd_product_remainder :
  odd_product 1002 % 1000 = 875 :=
sorry

end NUMINAMATH_CALUDE_odd_product_remainder_l722_72290


namespace NUMINAMATH_CALUDE_cone_base_circumference_l722_72249

/-- 
Given a right circular cone with volume 24π cubic centimeters and height 6 cm,
prove that the circumference of its base is 4√3π cm.
-/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 24 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l722_72249


namespace NUMINAMATH_CALUDE_max_sides_1950_gon_l722_72265

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here
  sides : n > 2

/-- The result of drawing all diagonals in a convex polygon --/
def drawAllDiagonals (p : ConvexPolygon n) : Set (ConvexPolygon m) :=
  sorry

/-- The maximum number of sides among the resulting polygons after drawing all diagonals --/
def maxResultingSides (p : ConvexPolygon n) : ℕ :=
  sorry

theorem max_sides_1950_gon :
  ∀ (p : ConvexPolygon 1950),
  maxResultingSides p = 1949 :=
sorry

end NUMINAMATH_CALUDE_max_sides_1950_gon_l722_72265


namespace NUMINAMATH_CALUDE_f_monotone_intervals_cos_alpha_value_l722_72257

noncomputable def f (x : ℝ) : ℝ := 
  1/2 * (Real.sin x + Real.cos x) * (Real.sin x - Real.cos x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_monotone_intervals (k : ℤ) : 
  StrictMonoOn f (Set.Icc (-Real.pi/6 + k * Real.pi) (Real.pi/3 + k * Real.pi)) := by sorry

theorem cos_alpha_value (α : ℝ) 
  (h1 : f (α/2 + Real.pi/4) = Real.sqrt 3 / 3) 
  (h2 : -Real.pi/2 < α) 
  (h3 : α < 0) : 
  Real.cos α = (3 + Real.sqrt 6) / 6 := by sorry

end NUMINAMATH_CALUDE_f_monotone_intervals_cos_alpha_value_l722_72257


namespace NUMINAMATH_CALUDE_triangle_area_sides_circumradius_l722_72252

/-- The area of a triangle in terms of its sides and circumradius -/
theorem triangle_area_sides_circumradius (a b c R S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = (a * b * c) / (4 * S))
  (h_area : S > 0) :
  S = (a * b * c) / (4 * R) := by
sorry


end NUMINAMATH_CALUDE_triangle_area_sides_circumradius_l722_72252


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l722_72268

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) →
  (c = b + 1) →
  (a + c = 140) →
  (b - a = 2) →
  (a + b + c = 210) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l722_72268


namespace NUMINAMATH_CALUDE_gavin_blue_shirts_l722_72209

/-- The number of blue shirts Gavin has -/
def blue_shirts (total : ℕ) (green : ℕ) : ℕ := total - green

theorem gavin_blue_shirts :
  let total_shirts : ℕ := 23
  let green_shirts : ℕ := 17
  blue_shirts total_shirts green_shirts = 6 := by
sorry

end NUMINAMATH_CALUDE_gavin_blue_shirts_l722_72209


namespace NUMINAMATH_CALUDE_side_x_must_be_green_l722_72230

-- Define the possible colors
inductive Color
  | Red
  | Green
  | Blue

-- Define a triangle with three sides
structure Triangle where
  side1 : Color
  side2 : Color
  side3 : Color

-- Define the condition that each triangle must have one of each color
def validTriangle (t : Triangle) : Prop :=
  t.side1 ≠ t.side2 ∧ t.side2 ≠ t.side3 ∧ t.side1 ≠ t.side3

-- Define the configuration of five triangles
structure Configuration where
  t1 : Triangle
  t2 : Triangle
  t3 : Triangle
  t4 : Triangle
  t5 : Triangle

-- Define the given colored sides
def givenColoring (c : Configuration) : Prop :=
  c.t1.side1 = Color.Green ∧
  c.t2.side1 = Color.Blue ∧
  c.t3.side3 = Color.Green ∧
  c.t5.side2 = Color.Blue

-- Define the shared sides
def sharedSides (c : Configuration) : Prop :=
  c.t1.side2 = c.t2.side3 ∧
  c.t1.side3 = c.t3.side1 ∧
  c.t2.side2 = c.t3.side2 ∧
  c.t3.side3 = c.t4.side1 ∧
  c.t4.side2 = c.t5.side1 ∧
  c.t4.side3 = c.t5.side3

-- Theorem statement
theorem side_x_must_be_green (c : Configuration) 
  (h1 : givenColoring c)
  (h2 : sharedSides c)
  (h3 : ∀ t, t ∈ [c.t1, c.t2, c.t3, c.t4, c.t5] → validTriangle t) :
  c.t4.side3 = Color.Green :=
sorry

end NUMINAMATH_CALUDE_side_x_must_be_green_l722_72230


namespace NUMINAMATH_CALUDE_decimal_123_to_binary_l722_72258

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem decimal_123_to_binary :
  decimal_to_binary 123 = [true, true, false, true, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_decimal_123_to_binary_l722_72258


namespace NUMINAMATH_CALUDE_sandwich_combinations_l722_72256

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) : 
  n_meat = 10 → n_cheese = 9 → n_meat * (n_cheese.choose 2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l722_72256


namespace NUMINAMATH_CALUDE_rectangular_field_area_l722_72264

/-- Represents a rectangular field with given properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ
  length_constraint : length = breadth + 30
  perimeter_constraint : perimeter = 2 * (length + breadth)

/-- Theorem: Area of the rectangular field with given constraints is 18000 square meters -/
theorem rectangular_field_area (field : RectangularField) (h : field.perimeter = 540) :
  field.length * field.breadth = 18000 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l722_72264


namespace NUMINAMATH_CALUDE_point_on_line_l722_72283

/-- 
Given two points (m, n) and (m + 2, n + some_value) that lie on the line x = (y/2) - (2/5),
prove that some_value must equal 4.
-/
theorem point_on_line (m n some_value : ℝ) : 
  (m = n / 2 - 2 / 5) ∧ (m + 2 = (n + some_value) / 2 - 2 / 5) → some_value = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l722_72283


namespace NUMINAMATH_CALUDE_initial_average_weight_l722_72225

/-- Proves the initially calculated average weight given the conditions of the problem -/
theorem initial_average_weight (n : ℕ) (misread_weight correct_weight : ℝ) (correct_avg : ℝ) :
  n = 20 ∧ 
  misread_weight = 56 ∧
  correct_weight = 61 ∧
  correct_avg = 58.65 →
  ∃ initial_avg : ℝ, 
    initial_avg * n + (correct_weight - misread_weight) = correct_avg * n ∧
    initial_avg = 58.4 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l722_72225


namespace NUMINAMATH_CALUDE_acute_angle_range_l722_72223

/-- Given two vectors a and b in ℝ², prove that the angle between them is acute
    if and only if x is in the specified range. -/
theorem acute_angle_range (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∀ i, a i * b i > 0) ↔ x ∈ Set.Ioo (-8) 2 ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_range_l722_72223


namespace NUMINAMATH_CALUDE_stream_speed_l722_72221

theorem stream_speed (swim_speed : ℝ) (upstream_time downstream_time : ℝ) :
  swim_speed = 12 ∧ 
  upstream_time = 2 * downstream_time ∧ 
  upstream_time > 0 ∧ 
  downstream_time > 0 →
  ∃ stream_speed : ℝ,
    stream_speed = 4 ∧
    (swim_speed - stream_speed) * upstream_time = (swim_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l722_72221


namespace NUMINAMATH_CALUDE_sharon_journey_distance_l722_72279

/-- Represents the journey from Sharon's house to her mother's house -/
structure Journey where
  distance : ℝ
  normalTime : ℝ
  reducedTime : ℝ
  speedReduction : ℝ

/-- The specific journey with given conditions -/
def sharonJourney : Journey where
  distance := 140  -- to be proved
  normalTime := 240
  reducedTime := 330
  speedReduction := 15

theorem sharon_journey_distance :
  ∀ j : Journey,
  j.normalTime = 240 ∧
  j.reducedTime = 330 ∧
  j.speedReduction = 15 ∧
  (j.distance / j.normalTime - j.speedReduction / 60) * (j.reducedTime - j.normalTime / 2) = j.distance / 2 →
  j.distance = 140 := by
  sorry

#check sharon_journey_distance

end NUMINAMATH_CALUDE_sharon_journey_distance_l722_72279


namespace NUMINAMATH_CALUDE_rectangle_24_60_parts_l722_72288

/-- The number of parts a rectangle is divided into when split into unit squares and its diagonal is drawn -/
def rectangle_parts (width : ℕ) (length : ℕ) : ℕ :=
  width * length + width + length - Nat.gcd width length

/-- Theorem stating that a 24 × 60 rectangle divided into unit squares and with its diagonal drawn is divided into 1512 parts -/
theorem rectangle_24_60_parts :
  rectangle_parts 24 60 = 1512 := by
  sorry

#eval rectangle_parts 24 60

end NUMINAMATH_CALUDE_rectangle_24_60_parts_l722_72288


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l722_72216

/-- Given a parallelogram with opposite vertices (2, -3) and (14, 9),
    the diagonals intersect at the point (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  midpoint = (8, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l722_72216


namespace NUMINAMATH_CALUDE_science_club_election_theorem_l722_72203

def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_to_fill : ℕ := 6

def elections_with_at_least_two_past_officers : ℕ :=
  Nat.choose total_candidates positions_to_fill -
  (Nat.choose (total_candidates - past_officers) positions_to_fill +
   Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions_to_fill - 1))

theorem science_club_election_theorem :
  elections_with_at_least_two_past_officers = 31500 := by
  sorry

end NUMINAMATH_CALUDE_science_club_election_theorem_l722_72203


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l722_72201

theorem no_linear_term_condition (p q : ℝ) : 
  (∀ x : ℝ, (x^2 - p*x + q)*(x - 3) = x^3 + (-p-3)*x^2 + 0*x + (-3*q)) → 
  q + 3*p = 0 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l722_72201


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l722_72238

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Two lines are distinct -/
def distinct_lines (l1 l2 : Line3D) : Prop := sorry

/-- Two planes are distinct -/
def distinct_planes (p1 p2 : Plane3D) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perpendicular (p1 p2 : Plane3D) : Prop := sorry

theorem line_plane_perpendicularity 
  (m n : Line3D) (α β : Plane3D) 
  (h1 : distinct_lines m n)
  (h2 : distinct_planes α β)
  (h3 : line_parallel_to_plane m α)
  (h4 : line_perpendicular_to_plane n β)
  (h5 : lines_parallel m n) :
  planes_perpendicular α β := by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l722_72238


namespace NUMINAMATH_CALUDE_fraction_of_25_comparison_l722_72266

theorem fraction_of_25_comparison : ∃ x : ℚ, 
  (x * 25 = 80 / 100 * 60 - 28) ∧ 
  (x = 4 / 5) := by
sorry

end NUMINAMATH_CALUDE_fraction_of_25_comparison_l722_72266
