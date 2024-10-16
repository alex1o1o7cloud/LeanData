import Mathlib

namespace NUMINAMATH_CALUDE_undefined_fraction_roots_product_l452_45258

theorem undefined_fraction_roots_product : ∃ (r₁ r₂ : ℝ), 
  (r₁^2 - 4*r₁ - 12 = 0) ∧ 
  (r₂^2 - 4*r₂ - 12 = 0) ∧ 
  (r₁ ≠ r₂) ∧
  (r₁ * r₂ = -12) := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_roots_product_l452_45258


namespace NUMINAMATH_CALUDE_parentheses_removal_l452_45244

theorem parentheses_removal (a b c d : ℝ) : a - (b - c + d) = a - b + c - d := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l452_45244


namespace NUMINAMATH_CALUDE_volunteer_selection_l452_45216

/-- The number of ways to select exactly one person to serve both days
    given 5 volunteers and 2 days of service where 2 people are selected each day. -/
theorem volunteer_selection (n : ℕ) (d : ℕ) (s : ℕ) (p : ℕ) : 
  n = 5 → d = 2 → s = 2 → p = 1 →
  (n.choose p) * ((n - p).choose (s - p)) * ((n - s).choose (s - p)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_l452_45216


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l452_45261

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/8))^(1/4))^2 * (((b^16)^(1/4))^(1/8))^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l452_45261


namespace NUMINAMATH_CALUDE_expression_evaluation_l452_45285

theorem expression_evaluation :
  |(-Real.sqrt 2)| + (-2023)^(0 : ℕ) - 2 * Real.sin (45 * π / 180) - (1/2)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l452_45285


namespace NUMINAMATH_CALUDE_sin_330_degrees_l452_45217

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l452_45217


namespace NUMINAMATH_CALUDE_students_per_row_l452_45222

theorem students_per_row (S R x : ℕ) : 
  S = x * R + 6 →  -- First scenario
  S = 12 * (R - 3) →  -- Second scenario
  S = 6 * R →  -- Third condition
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_row_l452_45222


namespace NUMINAMATH_CALUDE_book_cost_price_l452_45287

theorem book_cost_price (selling_price profit_percentage : ℝ)
  (h1 : profit_percentage = 0.10)
  (h2 : selling_price = (1 + profit_percentage) * 2800)
  (h3 : selling_price + 140 = (1 + 0.15) * 2800) :
  2800 = (selling_price - (1 + profit_percentage) * 2800) / profit_percentage :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l452_45287


namespace NUMINAMATH_CALUDE_shelves_needed_l452_45257

/-- Given a total of 14 books, with 2 taken by a librarian and 3 books fitting on each shelf,
    prove that 4 shelves are needed to store the remaining books. -/
theorem shelves_needed (total_books : ℕ) (taken_books : ℕ) (books_per_shelf : ℕ) :
  total_books = 14 →
  taken_books = 2 →
  books_per_shelf = 3 →
  ((total_books - taken_books) / books_per_shelf : ℕ) = 4 := by
  sorry

#check shelves_needed

end NUMINAMATH_CALUDE_shelves_needed_l452_45257


namespace NUMINAMATH_CALUDE_two_triangles_max_parts_two_rectangles_max_parts_two_n_gons_max_parts_l452_45253

/-- The maximum number of parts into which two polygons can divide a plane -/
def max_parts (sides : ℕ) : ℕ := 2 * sides + 2

/-- Two triangles can divide a plane into at most 8 parts -/
theorem two_triangles_max_parts : max_parts 3 = 8 := by sorry

/-- Two rectangles can divide a plane into at most 10 parts -/
theorem two_rectangles_max_parts : max_parts 4 = 10 := by sorry

/-- Two convex n-gons can divide a plane into at most 2n + 2 parts -/
theorem two_n_gons_max_parts (n : ℕ) : max_parts n = 2 * n + 2 := by sorry

end NUMINAMATH_CALUDE_two_triangles_max_parts_two_rectangles_max_parts_two_n_gons_max_parts_l452_45253


namespace NUMINAMATH_CALUDE_xanths_are_yelps_and_wicks_l452_45264

-- Define the sets
variable (U : Type) -- Universe set
variable (Zorb Yelp Xanth Wick : Set U)

-- State the given conditions
variable (h1 : Zorb ⊆ Yelp)
variable (h2 : Xanth ⊆ Zorb)
variable (h3 : Xanth ⊆ Wick)

-- State the theorem to be proved
theorem xanths_are_yelps_and_wicks : Xanth ⊆ Yelp ∩ Wick := by
  sorry

end NUMINAMATH_CALUDE_xanths_are_yelps_and_wicks_l452_45264


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l452_45278

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def chemistry_marks : ℕ := 97
def biology_marks : ℕ := 95
def average_marks : ℚ := 93
def num_subjects : ℕ := 5

theorem physics_marks_calculation :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 82 :=
by sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l452_45278


namespace NUMINAMATH_CALUDE_bottle_arrangement_l452_45273

theorem bottle_arrangement (x : ℕ) : 
  (x^2 + 36 = (x + 1)^2 + 3) → (x^2 + 36 = 292) :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_arrangement_l452_45273


namespace NUMINAMATH_CALUDE_choose_one_book_from_specific_shelf_l452_45283

/-- Represents a bookshelf with Chinese books on the upper shelf and math books on the lower shelf -/
structure Bookshelf :=
  (chinese_books : ℕ)
  (math_books : ℕ)

/-- Calculates the number of ways to choose one book from the bookshelf -/
def ways_to_choose_one_book (shelf : Bookshelf) : ℕ :=
  shelf.chinese_books + shelf.math_books

/-- Theorem stating that for a bookshelf with 5 Chinese books and 4 math books,
    the number of ways to choose one book is 9 -/
theorem choose_one_book_from_specific_shelf :
  let shelf : Bookshelf := ⟨5, 4⟩
  ways_to_choose_one_book shelf = 9 := by sorry

end NUMINAMATH_CALUDE_choose_one_book_from_specific_shelf_l452_45283


namespace NUMINAMATH_CALUDE_pentagon_fraction_sum_l452_45265

theorem pentagon_fraction_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h₅ : a₅ > 0) : 
  let s := a₁ + a₂ + a₃ + a₄ + a₅
  (a₁ / (s - a₁)) + (a₂ / (s - a₂)) + (a₃ / (s - a₃)) + (a₄ / (s - a₄)) + (a₅ / (s - a₅)) < 2 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_fraction_sum_l452_45265


namespace NUMINAMATH_CALUDE_triangle_base_length_l452_45298

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Perimeter of the triangle -/
  perimeter : ℝ
  /-- Length of the segment of the tangent to the inscribed circle, drawn parallel to the base and contained between the sides of the triangle -/
  tangent_segment : ℝ

/-- Theorem stating that for a triangle with perimeter 20 cm and an inscribed circle, 
    if the segment of the tangent to the circle drawn parallel to the base and 
    contained between the sides of the triangle is 2.4 cm, 
    then the base of the triangle is either 4 cm or 6 cm -/
theorem triangle_base_length (t : TriangleWithInscribedCircle) 
  (h_perimeter : t.perimeter = 20)
  (h_tangent : t.tangent_segment = 2.4) :
  ∃ (base : ℝ), (base = 4 ∨ base = 6) ∧ 
  (∃ (side1 side2 : ℝ), side1 + side2 + base = t.perimeter) :=
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l452_45298


namespace NUMINAMATH_CALUDE_not_always_true_parallel_transitivity_l452_45200

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_true_parallel_transitivity 
  (a b : Line) (α : Plane) : 
  ¬(∀ (a b : Line) (α : Plane), 
    parallel_lines a b → 
    parallel_line_plane a α → 
    parallel_line_plane b α) :=
by
  sorry


end NUMINAMATH_CALUDE_not_always_true_parallel_transitivity_l452_45200


namespace NUMINAMATH_CALUDE_matrix_det_times_two_l452_45235

def matrix_det (a b c d : ℤ) : ℤ := a * d - b * c

theorem matrix_det_times_two :
  2 * (matrix_det 5 7 2 3) = 2 := by sorry

end NUMINAMATH_CALUDE_matrix_det_times_two_l452_45235


namespace NUMINAMATH_CALUDE_abs_a_plus_inv_a_geq_two_l452_45282

theorem abs_a_plus_inv_a_geq_two (a : ℝ) (h : a ≠ 0) : |a + 1/a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_plus_inv_a_geq_two_l452_45282


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l452_45291

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l452_45291


namespace NUMINAMATH_CALUDE_exists_valid_sequence_l452_45204

-- Define the sequence type
def IncreasingSequence := ℕ → ℕ

-- Define the properties of the sequence
def IsValidSequence (a : IncreasingSequence) : Prop :=
  a 0 = 0 ∧
  (∀ n m, n < m → a n < a m) ∧
  (∀ m : ℕ, ∃ i j, m = a i + a j) ∧
  (∀ n : ℕ, n > 0 → a n > n^2 / 16)

-- The theorem to be proved
theorem exists_valid_sequence : ∃ a : IncreasingSequence, IsValidSequence a := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_sequence_l452_45204


namespace NUMINAMATH_CALUDE_melissa_shoe_repair_time_l452_45202

/-- The time Melissa spends repairing her shoes -/
theorem melissa_shoe_repair_time :
  ∀ (buckle_time heel_time : ℕ) (num_shoes : ℕ),
  buckle_time = 5 →
  heel_time = 10 →
  num_shoes = 2 →
  buckle_time * num_shoes + heel_time * num_shoes = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_melissa_shoe_repair_time_l452_45202


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l452_45233

def taxi_cost (initial_charge : ℚ) (additional_charge : ℚ) (passenger_fee : ℚ) (luggage_fee : ℚ)
              (distance : ℚ) (passengers : ℕ) (luggage : ℕ) : ℚ :=
  let distance_quarters := (distance - 1/4).ceil * 4
  let distance_charge := initial_charge + additional_charge * (distance_quarters - 1)
  let passenger_charge := passenger_fee * (passengers - 1)
  let luggage_charge := luggage_fee * luggage
  distance_charge + passenger_charge + luggage_charge

theorem taxi_ride_cost :
  taxi_cost 5 0.6 1 2 12.4 3 2 = 39.8 := by
  sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l452_45233


namespace NUMINAMATH_CALUDE_ellipse_and_circle_intersection_l452_45260

/-- The ellipse C₁ -/
def C₁ (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- The parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle C₃ -/
def C₃ (x y x₀ y₀ r : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The point P -/
def P (x y : ℝ) : Prop := C₁ x y 2 (Real.sqrt 3) ∧ C₂ x y ∧ x > 0 ∧ y > 0

/-- The point T -/
def T (x y : ℝ) : Prop := C₂ x y

theorem ellipse_and_circle_intersection 
  (a b : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : ∃ x y, P x y ∧ (x - 1)^2 + y^2 = (5/3)^2) 
  (h₄ : ∀ x₀ y₀ r, T x₀ y₀ → C₃ 0 2 x₀ y₀ r → C₃ 0 (-2) x₀ y₀ r → r^2 = 4 + x₀^2) :
  (∀ x y, C₁ x y a b ↔ C₁ x y 2 (Real.sqrt 3)) ∧ 
  (∀ x₀ y₀ r, T x₀ y₀ → C₃ 2 0 x₀ y₀ r) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_intersection_l452_45260


namespace NUMINAMATH_CALUDE_equal_expressions_l452_45214

theorem equal_expressions : 
  (2^3 ≠ 2 * 3) ∧ 
  (-(-2)^2 ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l452_45214


namespace NUMINAMATH_CALUDE_log_equality_l452_45284

theorem log_equality : Real.log 16 / Real.log 4096 = Real.log 4 / Real.log 64 := by sorry

end NUMINAMATH_CALUDE_log_equality_l452_45284


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_math_majors_consecutive_probability_proof_l452_45269

/-- The probability of all math majors sitting consecutively around a circular table -/
theorem math_majors_consecutive_probability : ℚ :=
  let total_people : ℕ := 12
  let math_majors : ℕ := 5
  let physics_majors : ℕ := 4
  let biology_majors : ℕ := 3
  1 / 330

/-- Proof that the probability of all math majors sitting consecutively is 1/330 -/
theorem math_majors_consecutive_probability_proof :
  math_majors_consecutive_probability = 1 / 330 := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_math_majors_consecutive_probability_proof_l452_45269


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l452_45255

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 17th term is 12 and the 18th term is 15, the 3rd term is -30. -/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℤ) 
  (h_arithmetic : isArithmeticSequence a) 
  (h_17th : a 17 = 12) 
  (h_18th : a 18 = 15) : 
  a 3 = -30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l452_45255


namespace NUMINAMATH_CALUDE_function_satisfying_cross_ratio_is_linear_l452_45263

/-- A function satisfying the given cross-ratio condition is linear -/
theorem function_satisfying_cross_ratio_is_linear (f : ℝ → ℝ) :
  (∀ (a b c d : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ (k m : ℝ), ∀ x, f x = k * x + m :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_cross_ratio_is_linear_l452_45263


namespace NUMINAMATH_CALUDE_conference_games_sports_conference_season_l452_45268

theorem conference_games (total_teams : ℕ) (divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let total_games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  let total_games_counted_twice := total_teams * total_games_per_team
  total_games_counted_twice / 2

theorem sports_conference_season :
  conference_games 16 2 8 2 1 = 176 := by
sorry

end NUMINAMATH_CALUDE_conference_games_sports_conference_season_l452_45268


namespace NUMINAMATH_CALUDE_song_guessing_game_theorem_l452_45230

/-- The Song Guessing Game -/
structure SongGuessingGame where
  /-- Probability of correctly guessing a song from group A -/
  probA : ℝ
  /-- Probability of correctly guessing a song from group B -/
  probB : ℝ
  /-- Number of songs played from group A -/
  numA : ℕ
  /-- Number of songs played from group B -/
  numB : ℕ
  /-- Points earned for correctly guessing a song from group A -/
  pointsA : ℕ
  /-- Points earned for correctly guessing a song from group B -/
  pointsB : ℕ

/-- The probability of guessing at least 2 song titles correctly -/
def probAtLeastTwo (game : SongGuessingGame) : ℝ := sorry

/-- The expectation of the total score -/
def expectedScore (game : SongGuessingGame) : ℝ := sorry

/-- Main theorem about the Song Guessing Game -/
theorem song_guessing_game_theorem (game : SongGuessingGame) 
  (h1 : game.probA = 2/3)
  (h2 : game.probB = 1/2)
  (h3 : game.numA = 2)
  (h4 : game.numB = 2)
  (h5 : game.pointsA = 1)
  (h6 : game.pointsB = 2) :
  probAtLeastTwo game = 29/36 ∧ expectedScore game = 10/3 := by sorry

end NUMINAMATH_CALUDE_song_guessing_game_theorem_l452_45230


namespace NUMINAMATH_CALUDE_infinitely_many_non_squares_l452_45211

theorem infinitely_many_non_squares (a b c : ℕ+) :
  Set.Infinite {n : ℕ+ | ∃ k : ℕ, (n.val : ℤ)^3 + (a.val : ℤ) * (n.val : ℤ)^2 + (b.val : ℤ) * (n.val : ℤ) + (c.val : ℤ) ≠ (k : ℤ)^2} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_non_squares_l452_45211


namespace NUMINAMATH_CALUDE_range_of_m_solution_set_correct_l452_45203

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (2*m + 1) * x + 2

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ x y, x > 1 ∧ y < 1 ∧ f m x = 0 ∧ f m y = 0) → -1 < m ∧ m < 0 :=
sorry

-- Define the solution set for f(x) ≤ 0
def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then { x | x ≤ -2 }
  else if m < 0 then { x | -2 ≤ x ∧ x ≤ -1/m }
  else if 0 < m ∧ m < 1/2 then { x | -1/m ≤ x ∧ x ≤ -2 }
  else if m = 1/2 then { -2 }
  else { x | -2 ≤ x ∧ x ≤ -1/m }

-- Theorem for the solution set
theorem solution_set_correct (m : ℝ) (x : ℝ) : 
  x ∈ solution_set m ↔ f m x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_set_correct_l452_45203


namespace NUMINAMATH_CALUDE_faster_walking_speed_l452_45250

theorem faster_walking_speed 
  (actual_distance : ℝ) 
  (original_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 60) 
  (h2 : original_speed = 12) 
  (h3 : additional_distance = 20) : 
  let time := actual_distance / original_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 16 := by sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l452_45250


namespace NUMINAMATH_CALUDE_functional_equation_solution_l452_45201

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * f y - y * f x = x * y * f (x / y)

/-- Theorem stating that for any function satisfying the functional equation, f(50) = 0 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l452_45201


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l452_45295

theorem quadratic_completing_square (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l452_45295


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_l452_45205

theorem rational_coefficient_terms_count :
  let expansion := (fun (x y : ℝ) => (x * Real.rpow 2 (1/3) + y * Real.sqrt 3) ^ 500)
  let total_terms := 501
  let is_rational_coeff (k : ℕ) := (k % 3 = 0) ∧ ((500 - k) % 2 = 0)
  (Finset.filter is_rational_coeff (Finset.range total_terms)).card = 84 :=
sorry

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_l452_45205


namespace NUMINAMATH_CALUDE_water_fraction_in_mixture_l452_45297

theorem water_fraction_in_mixture (alcohol_to_water_ratio : ℚ) :
  alcohol_to_water_ratio = 2/3 →
  (water_volume / (water_volume + alcohol_volume) = 3/5) :=
by
  sorry

end NUMINAMATH_CALUDE_water_fraction_in_mixture_l452_45297


namespace NUMINAMATH_CALUDE_pears_picked_total_l452_45279

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 8

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 7

/-- The total number of pears picked -/
def total_pears : ℕ := mike_pears + jason_pears

theorem pears_picked_total : total_pears = 15 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l452_45279


namespace NUMINAMATH_CALUDE_figure_to_square_l452_45219

/-- Represents a figure on a graph paper -/
structure Figure where
  area : ℕ

/-- Represents a cut of the figure -/
structure Cut where
  parts : ℕ

/-- Represents the result of reassembling cut parts -/
structure Reassembly where
  isSquare : Bool

/-- A function that cuts a figure into parts -/
def cutFigure (f : Figure) (c : Cut) : Cut :=
  c

/-- A function that reassembles cut parts -/
def reassemble (c : Cut) : Reassembly :=
  { isSquare := true }

/-- Theorem stating that a figure with area 18 can be cut into 3 parts
    and reassembled into a square -/
theorem figure_to_square (f : Figure) (h : f.area = 18) :
  ∃ (c : Cut), c.parts = 3 ∧ (reassemble (cutFigure f c)).isSquare = true := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l452_45219


namespace NUMINAMATH_CALUDE_tan_value_for_given_point_l452_45223

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) (h : ∃ (r : Real), r * (Real.cos θ) = -Real.sqrt 3 / 2 ∧ r * (Real.sin θ) = 1 / 2) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_point_l452_45223


namespace NUMINAMATH_CALUDE_complex_equation_solution_l452_45248

theorem complex_equation_solution (z : ℂ) :
  z * Complex.I = 1 → z = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l452_45248


namespace NUMINAMATH_CALUDE_power_seven_mod_eight_l452_45221

theorem power_seven_mod_eight : 7^51 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_eight_l452_45221


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l452_45237

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 2 → a 8 = 32 → a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l452_45237


namespace NUMINAMATH_CALUDE_min_dot_product_of_tangents_l452_45226

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_M (x y θ : ℝ) : Prop := (x - 5 * Real.cos θ)^2 + (y - 5 * Real.sin θ)^2 = 1

-- Define a point on circle M
def point_on_M (P : ℝ × ℝ) (θ : ℝ) : Prop := circle_M P.1 P.2 θ

-- Define tangent lines from P to circle O
def tangent_to_O (P E : ℝ × ℝ) : Prop := 
  circle_O E.1 E.2 ∧ (P.1 - E.1) * E.1 + (P.2 - E.2) * E.2 = 0

-- Statement of the theorem
theorem min_dot_product_of_tangents :
  ∀ (P : ℝ × ℝ) (θ : ℝ),
  point_on_M P θ →
  ∃ (E F : ℝ × ℝ),
  tangent_to_O P E ∧ tangent_to_O P F →
  (∀ (E' F' : ℝ × ℝ), tangent_to_O P E' ∧ tangent_to_O P F' →
    ((P.1 - E.1) * (P.1 - F.1) + (P.2 - E.2) * (P.2 - F.2)) ≤
    ((P.1 - E'.1) * (P.1 - F'.1) + (P.2 - E'.2) * (P.2 - F'.2))) ∧
  ((P.1 - E.1) * (P.1 - F.1) + (P.2 - E.2) * (P.2 - F.2)) = 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_of_tangents_l452_45226


namespace NUMINAMATH_CALUDE_brenda_lead_after_turn3_l452_45294

/-- Represents the score of a player in a Scrabble game -/
structure ScrabbleScore where
  turn1 : ℕ
  turn2 : ℕ
  turn3 : ℕ

/-- Calculates the total score for a player -/
def totalScore (score : ScrabbleScore) : ℕ :=
  score.turn1 + score.turn2 + score.turn3

/-- Represents the Scrabble game between Brenda and David -/
structure ScrabbleGame where
  brenda : ScrabbleScore
  david : ScrabbleScore
  brenda_lead_before_turn3 : ℕ

/-- The Scrabble game instance based on the given problem -/
def game : ScrabbleGame :=
  { brenda := { turn1 := 18, turn2 := 25, turn3 := 15 }
  , david := { turn1 := 10, turn2 := 35, turn3 := 32 }
  , brenda_lead_before_turn3 := 22 }

/-- Theorem stating that Brenda is ahead by 5 points after the third turn -/
theorem brenda_lead_after_turn3 (g : ScrabbleGame) : 
  totalScore g.brenda - totalScore g.david = 5 :=
sorry

end NUMINAMATH_CALUDE_brenda_lead_after_turn3_l452_45294


namespace NUMINAMATH_CALUDE_cone_cross_section_area_l452_45210

/-- Given a cone with surface area 36π and sector central angle 2π/3 when unfolded,
    the area of the cross-section along its axis is 18√2. -/
theorem cone_cross_section_area (R a h : ℝ) : 
  R > 0 ∧ a > 0 ∧ h > 0 →
  π * R^2 + (2/3) * π * a^2 = 36 * π →
  (2/3) * 2 * π * R = 2 * π * a →
  a^2 = h^2 + R^2 →
  R * h = 18 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cone_cross_section_area_l452_45210


namespace NUMINAMATH_CALUDE_foreign_trade_income_l452_45274

/-- Foreign trade income problem -/
theorem foreign_trade_income 
  (m : ℝ) -- Foreign trade income in 2001 (billion yuan)
  (x : ℝ) -- Percentage increase in 2002
  (n : ℝ) -- Foreign trade income in 2003 (billion yuan)
  (h1 : x > 0) -- Ensure x is positive
  (h2 : m > 0) -- Ensure initial income is positive
  : n = m * (1 + x / 100) * (1 + 2 * x / 100) :=
by sorry

end NUMINAMATH_CALUDE_foreign_trade_income_l452_45274


namespace NUMINAMATH_CALUDE_horse_cow_pricing_system_l452_45286

theorem horse_cow_pricing_system (x y : ℝ) :
  (4 * x + 6 * y = 48 ∧ 3 * x + 5 * y = 38) ↔
  (∃ (horse_price cow_price : ℝ),
    horse_price = x ∧
    cow_price = y ∧
    4 * horse_price + 6 * cow_price = 48 ∧
    3 * horse_price + 5 * cow_price = 38) :=
by sorry

end NUMINAMATH_CALUDE_horse_cow_pricing_system_l452_45286


namespace NUMINAMATH_CALUDE_circle_M_properties_l452_45275

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x + 3*y - 2 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

-- Theorem statement
theorem circle_M_properties :
  (∃ (c_x c_y r : ℝ), ∀ (x y : ℝ), circle_M x y ↔ (x - c_x)^2 + (y - c_y)^2 = r^2) ∧
  (∀ (x y : ℝ), circle_M x y → circle_M (2*(2-x) - x) (2*(2-y) - y)) ∧
  (∃ (t_x t_y : ℝ), circle_M t_x t_y ∧ tangent_line t_x t_y ∧
    ∀ (x y : ℝ), circle_M x y → (x - t_x)^2 + (y - t_y)^2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_properties_l452_45275


namespace NUMINAMATH_CALUDE_perpendicular_slope_l452_45229

/-- The slope of a line perpendicular to the line containing points (2, -3) and (-4, -8) is -6/5 -/
theorem perpendicular_slope : 
  let p₁ : ℚ × ℚ := (2, -3)
  let p₂ : ℚ × ℚ := (-4, -8)
  let m : ℚ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  (-1 / m) = -6/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l452_45229


namespace NUMINAMATH_CALUDE_bill_selling_price_l452_45224

theorem bill_selling_price (P : ℝ) 
  (h1 : P + 0.1 * P = 1.1 * P)  -- Original selling price
  (h2 : 0.9 * P + 0.3 * (0.9 * P) = 1.17 * P)  -- New selling price with 30% profit
  (h3 : 1.17 * P = 1.1 * P + 42)  -- Equation relating the two selling prices
  : 1.1 * P = 660 := by
  sorry

end NUMINAMATH_CALUDE_bill_selling_price_l452_45224


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l452_45262

def M : Set ℤ := {0, 1, 2}

def A : Set ℤ := {y | ∃ x ∈ M, y = 2 * x}

def B : Set ℤ := {y | ∃ x ∈ M, y = 2 * x - 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l452_45262


namespace NUMINAMATH_CALUDE_log_equation_solution_l452_45228

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution (x k : ℝ) :
  log k x * log 3 k = 4 → k = 9 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l452_45228


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l452_45241

-- Define the quadrant type
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define the function to determine the quadrant of an angle
def angle_quadrant (θ : Real) : Quadrant :=
  sorry

-- Theorem statement
theorem angle_in_second_quadrant (θ : Real) 
  (h1 : Real.sin θ > Real.cos θ) 
  (h2 : Real.tan θ < 0) : 
  angle_quadrant θ = Quadrant.second :=
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l452_45241


namespace NUMINAMATH_CALUDE_measure_of_angle_ABC_l452_45212

-- Define the angles
def angle_ABC : ℝ := sorry
def angle_ABD : ℝ := 30
def angle_CBD : ℝ := 90

-- State the theorem
theorem measure_of_angle_ABC :
  angle_ABC = 60 ∧ 
  angle_CBD = 90 ∧ 
  angle_ABD = 30 ∧ 
  angle_ABC + angle_ABD + angle_CBD = 180 :=
by sorry

end NUMINAMATH_CALUDE_measure_of_angle_ABC_l452_45212


namespace NUMINAMATH_CALUDE_more_girls_than_boys_in_class_l452_45296

theorem more_girls_than_boys_in_class (num_students : ℕ) (num_teachers : ℕ) 
  (h_students : num_students = 42)
  (h_teachers : num_teachers = 6)
  (h_ratio : ∃ (x : ℕ), num_students = 7 * x ∧ 3 * x = num_boys ∧ 4 * x = num_girls) :
  num_girls - num_boys = 6 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_in_class_l452_45296


namespace NUMINAMATH_CALUDE_product_plus_245_divisible_by_5_l452_45239

theorem product_plus_245_divisible_by_5 : ∃ k : ℤ, (1250 * 1625 * 1830 * 2075 + 245 : ℤ) = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_product_plus_245_divisible_by_5_l452_45239


namespace NUMINAMATH_CALUDE_smaller_part_is_eleven_l452_45236

theorem smaller_part_is_eleven (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : 
  min x y = 11 := by
  sorry

end NUMINAMATH_CALUDE_smaller_part_is_eleven_l452_45236


namespace NUMINAMATH_CALUDE_mary_bike_rental_hours_l452_45276

/-- Calculates the number of hours a bike was rented given the total payment, fixed fee, and hourly rate. -/
def rent_hours (total_payment fixed_fee hourly_rate : ℚ) : ℚ :=
  (total_payment - fixed_fee) / hourly_rate

/-- Proves that Mary rented the bike for 9 hours given the specified conditions. -/
theorem mary_bike_rental_hours :
  let fixed_fee : ℚ := 17
  let hourly_rate : ℚ := 7
  let total_payment : ℚ := 80
  rent_hours total_payment fixed_fee hourly_rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_mary_bike_rental_hours_l452_45276


namespace NUMINAMATH_CALUDE_power_of_128_l452_45227

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by sorry

end NUMINAMATH_CALUDE_power_of_128_l452_45227


namespace NUMINAMATH_CALUDE_range_of_m_l452_45266

/-- Given an equation with a non-negative solution, prove the range of m -/
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 2) + 1 = x / (2 - x)) → 
  m ≤ 2 ∧ m ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l452_45266


namespace NUMINAMATH_CALUDE_food_production_growth_rate_l452_45249

theorem food_production_growth_rate 
  (initial_production : ℝ) 
  (a b x : ℝ) 
  (h1 : initial_production = 5000)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : x > 0)
  (h5 : initial_production * (1 + a) * (1 + b) = initial_production * (1 + x)^2) :
  x ≤ (a + b) / 2 := by
sorry

end NUMINAMATH_CALUDE_food_production_growth_rate_l452_45249


namespace NUMINAMATH_CALUDE_teacher_age_l452_45252

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 50 →
  student_avg_age = 14 →
  new_avg_age = 15 →
  (num_students * student_avg_age + (65 : ℝ)) / (num_students + 1) = new_avg_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l452_45252


namespace NUMINAMATH_CALUDE_white_to_brown_dog_weight_ratio_l452_45247

def brown_dog_weight : ℝ := 4
def black_dog_weight : ℝ := brown_dog_weight + 1
def grey_dog_weight : ℝ := black_dog_weight - 2
def average_weight : ℝ := 5
def num_dogs : ℕ := 4

def white_dog_weight : ℝ := average_weight * num_dogs - (brown_dog_weight + black_dog_weight + grey_dog_weight)

theorem white_to_brown_dog_weight_ratio :
  white_dog_weight / brown_dog_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_white_to_brown_dog_weight_ratio_l452_45247


namespace NUMINAMATH_CALUDE_bert_equals_kameron_in_40_days_l452_45242

/-- The number of kangaroos Kameron has -/
def kameron_kangaroos : ℕ := 100

/-- The initial number of kangaroos Bert has -/
def bert_initial_kangaroos : ℕ := 20

/-- The number of kangaroos Bert buys per day -/
def bert_daily_increase : ℕ := 2

/-- The number of days it takes for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal : ℕ := (kameron_kangaroos - bert_initial_kangaroos) / bert_daily_increase

theorem bert_equals_kameron_in_40_days :
  days_to_equal = 40 :=
by sorry

end NUMINAMATH_CALUDE_bert_equals_kameron_in_40_days_l452_45242


namespace NUMINAMATH_CALUDE_min_value_of_max_function_l452_45240

theorem min_value_of_max_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > 2*y) :
  ∃ (t : ℝ), t = max (x^2/2) (4/(y*(x-2*y))) ∧ t ≥ 4 ∧ 
  (∃ (x0 y0 : ℝ), x0 > 0 ∧ y0 > 0 ∧ x0 > 2*y0 ∧ 
    max (x0^2/2) (4/(y0*(x0-2*y0))) = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_max_function_l452_45240


namespace NUMINAMATH_CALUDE_article_cost_price_l452_45213

/-- The cost price of an article that satisfies the given conditions -/
def cost_price : ℝ := 1600

/-- The selling price of the article with a 5% gain -/
def selling_price (c : ℝ) : ℝ := 1.05 * c

/-- The new cost price if bought at 5% less -/
def new_cost_price (c : ℝ) : ℝ := 0.95 * c

/-- The new selling price if sold for 8 less -/
def new_selling_price (c : ℝ) : ℝ := selling_price c - 8

theorem article_cost_price :
  selling_price cost_price = 1.05 * cost_price ∧
  new_cost_price cost_price = 0.95 * cost_price ∧
  new_selling_price cost_price = selling_price cost_price - 8 ∧
  new_selling_price cost_price = 1.1 * new_cost_price cost_price :=
by sorry

end NUMINAMATH_CALUDE_article_cost_price_l452_45213


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l452_45271

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x * y = 3) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + a * b = 3 → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l452_45271


namespace NUMINAMATH_CALUDE_total_marbles_l452_45208

theorem total_marbles (x : ℝ) : (4*x + 2) + 2*x + (3*x + 1) = 9*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l452_45208


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l452_45246

/-- Definition of the sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Definition of the sequence terms -/
def a (n : ℕ) : ℕ := S (n + 1) - S n

/-- Proposition: The sequence {a_n} is arithmetic -/
theorem sequence_is_arithmetic : ∃ (d : ℕ), ∀ (n : ℕ), a (n + 1) = a n + d := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l452_45246


namespace NUMINAMATH_CALUDE_percentage_sum_l452_45243

theorem percentage_sum : (28 / 100) * 400 + (45 / 100) * 250 = 224.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l452_45243


namespace NUMINAMATH_CALUDE_quadratic_greater_than_zero_l452_45245

theorem quadratic_greater_than_zero (x : ℝ) :
  (x + 2) * (x - 3) - 4 > 0 ↔ x < (1 - Real.sqrt 41) / 2 ∨ x > (1 + Real.sqrt 41) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_greater_than_zero_l452_45245


namespace NUMINAMATH_CALUDE_light_green_yellow_percentage_l452_45232

-- Define the variables
def light_green_volume : ℝ := 5
def dark_green_volume : ℝ := 1.66666666667
def dark_green_yellow_percentage : ℝ := 0.4
def mixture_yellow_percentage : ℝ := 0.25

-- Define the theorem
theorem light_green_yellow_percentage :
  ∃ x : ℝ, 
    x * light_green_volume + dark_green_yellow_percentage * dark_green_volume = 
    mixture_yellow_percentage * (light_green_volume + dark_green_volume) ∧ 
    x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_light_green_yellow_percentage_l452_45232


namespace NUMINAMATH_CALUDE_cookie_count_l452_45277

/-- The number of cookies Paul and Paula have altogether -/
def total_cookies (paul_cookies : ℕ) (paula_difference : ℕ) : ℕ :=
  paul_cookies + (paul_cookies - paula_difference)

/-- Theorem: Paul and Paula have 87 cookies altogether -/
theorem cookie_count : total_cookies 45 3 = 87 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l452_45277


namespace NUMINAMATH_CALUDE_sum_binary_digits_350_1350_l452_45281

/-- The number of digits in the binary representation of a positive integer -/
def binaryDigits (n : ℕ+) : ℕ :=
  Nat.log2 n + 1

/-- The sum of binary digits for 350 and 1350 -/
def sumBinaryDigits : ℕ := binaryDigits 350 + binaryDigits 1350

theorem sum_binary_digits_350_1350 : sumBinaryDigits = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_350_1350_l452_45281


namespace NUMINAMATH_CALUDE_snail_track_time_equivalence_l452_45225

theorem snail_track_time_equivalence (clockwise_time : Real) (counterclockwise_time : Real) : 
  clockwise_time = 1.5 → counterclockwise_time = 90 → clockwise_time * 60 = counterclockwise_time :=
by
  sorry

end NUMINAMATH_CALUDE_snail_track_time_equivalence_l452_45225


namespace NUMINAMATH_CALUDE_curve_properties_l452_45251

/-- The curve equation -/
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0

theorem curve_properties :
  (∀ x y : ℝ, curve x y 1 ↔ x = 1 ∧ y = 1) ∧
  (∀ a : ℝ, a ≠ 1 → curve 1 1 a) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l452_45251


namespace NUMINAMATH_CALUDE_perpendicular_construction_l452_45288

/-- A two-sided ruler with parallel edges -/
structure TwoSidedRuler :=
  (width : ℝ)
  (width_pos : width > 0)

/-- A line in a plane -/
structure Line :=
  (a b c : ℝ)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0)

/-- A point in a plane -/
structure Point :=
  (x y : ℝ)

/-- Checks if a point is on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem perpendicular_construction 
  (l : Line) (M : Point) (h : M.on_line l) :
  ∃ (P : Point), ∃ (n : Line), 
    M.on_line n ∧ P.on_line n ∧ n.perpendicular l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_construction_l452_45288


namespace NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l452_45220

def hotdog_sequence (n : ℕ) : ℕ := 10 + 2 * (n - 1)

theorem total_hotdogs_by_wednesday :
  (hotdog_sequence 1) + (hotdog_sequence 2) + (hotdog_sequence 3) = 36 :=
by sorry

end NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l452_45220


namespace NUMINAMATH_CALUDE_cone_height_from_sector_l452_45270

/-- Given a sector with radius 7 cm and area 21π cm², when used to form the lateral surface of a cone, 
    the height of the cone is 2√10 cm. -/
theorem cone_height_from_sector (r : ℝ) (area : ℝ) (h : ℝ) : 
  r = 7 → 
  area = 21 * Real.pi → 
  area = (1/2) * (2 * Real.pi) * 3 * r → 
  h = Real.sqrt (r^2 - 3^2) → 
  h = 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_cone_height_from_sector_l452_45270


namespace NUMINAMATH_CALUDE_two_x_minus_one_gt_zero_is_linear_inequality_l452_45299

/-- Definition of a linear inequality in one variable -/
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x ↔ a * x + b > 0 ∨ a * x + b < 0 ∨ a * x + b = 0

/-- The inequality 2x - 1 > 0 is a linear inequality in one variable -/
theorem two_x_minus_one_gt_zero_is_linear_inequality :
  is_linear_inequality_one_var (λ x : ℝ => 2 * x - 1 > 0) :=
sorry

end NUMINAMATH_CALUDE_two_x_minus_one_gt_zero_is_linear_inequality_l452_45299


namespace NUMINAMATH_CALUDE_nested_average_equals_seven_eighteenths_l452_45207

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem -/
theorem nested_average_equals_seven_eighteenths :
  avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_equals_seven_eighteenths_l452_45207


namespace NUMINAMATH_CALUDE_amount_spent_first_shop_l452_45259

/-- The amount spent on books from the first shop -/
def amount_first_shop : ℕ := 1500

/-- The number of books bought from the first shop -/
def books_first_shop : ℕ := 55

/-- The number of books bought from the second shop -/
def books_second_shop : ℕ := 60

/-- The amount spent on books from the second shop -/
def amount_second_shop : ℕ := 340

/-- The average price per book -/
def average_price : ℕ := 16

/-- Theorem stating that the amount spent on the first shop is 1500,
    given the conditions of the problem -/
theorem amount_spent_first_shop :
  amount_first_shop = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_amount_spent_first_shop_l452_45259


namespace NUMINAMATH_CALUDE_square_inequality_l452_45272

theorem square_inequality (a b c A B C : ℝ) 
  (h1 : b^2 < a*c) 
  (h2 : a*C - 2*b*B + c*A = 0) : 
  B^2 ≥ A*C := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l452_45272


namespace NUMINAMATH_CALUDE_product_in_base_9_l452_45267

/-- Converts a base-9 number to base-10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

/-- Converts a base-10 number to base-9 --/
def to_base_9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Theorem: The product of 237₉ and 17₉ is 4264₉ --/
theorem product_in_base_9 : 
  to_base_9 (to_base_10 [7, 3, 2] * to_base_10 [7, 1]) = [4, 6, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_product_in_base_9_l452_45267


namespace NUMINAMATH_CALUDE_new_oranges_added_l452_45290

/-- Calculates the number of new oranges added to a bin -/
def new_oranges (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - thrown_away)

/-- Proves that the number of new oranges added is 28 -/
theorem new_oranges_added : new_oranges 5 2 31 = 28 := by
  sorry

end NUMINAMATH_CALUDE_new_oranges_added_l452_45290


namespace NUMINAMATH_CALUDE_unoccupied_volume_correct_l452_45256

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of an ice cube -/
structure IceCubeDimensions where
  side : ℕ

/-- Calculates the unoccupied volume in a tank given its dimensions, water depth, and ice cubes -/
def unoccupiedVolume (tank : TankDimensions) (waterDepth : ℕ) (iceCube : IceCubeDimensions) (numIceCubes : ℕ) : ℕ :=
  let tankVolume := tank.length * tank.width * tank.height
  let waterVolume := tank.length * tank.width * waterDepth
  let iceCubeVolume := iceCube.side * iceCube.side * iceCube.side
  let totalIceVolume := numIceCubes * iceCubeVolume
  tankVolume - (waterVolume + totalIceVolume)

/-- Theorem stating the unoccupied volume in the tank under given conditions -/
theorem unoccupied_volume_correct :
  let tank : TankDimensions := ⟨12, 12, 15⟩
  let waterDepth : ℕ := 7
  let iceCube : IceCubeDimensions := ⟨3⟩
  let numIceCubes : ℕ := 15
  unoccupiedVolume tank waterDepth iceCube numIceCubes = 747 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_volume_correct_l452_45256


namespace NUMINAMATH_CALUDE_triangle_perimeter_l452_45289

/-- Given a triangle with inradius 2.5 cm and area 45 cm², its perimeter is 36 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 45 → A = r * (p / 2) → p = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l452_45289


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l452_45209

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The probability of 4 randomly selected chords from 8 points on a circle forming a convex quadrilateral -/
theorem convex_quadrilateral_probability :
  (n.choose k : ℚ) / (total_chords.choose k : ℚ) = 2 / 585 :=
sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l452_45209


namespace NUMINAMATH_CALUDE_f_minimum_l452_45234

/-- The polynomial f(x) = x^2 + 6x + 10 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 10

/-- The point where f(x) reaches its minimum -/
def min_point : ℝ := -3

theorem f_minimum :
  ∀ x : ℝ, f x ≥ f min_point := by sorry

end NUMINAMATH_CALUDE_f_minimum_l452_45234


namespace NUMINAMATH_CALUDE_perfect_square_condition_l452_45218

theorem perfect_square_condition (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - (m+1)*x + 1 = k^2) → (m = 1 ∨ m = -3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l452_45218


namespace NUMINAMATH_CALUDE_g_of_5_l452_45231

def g (x : ℝ) : ℝ := 3*x^5 - 15*x^4 + 30*x^3 - 45*x^2 + 24*x + 50

theorem g_of_5 : g 5 = 2795 := by sorry

end NUMINAMATH_CALUDE_g_of_5_l452_45231


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l452_45206

/-- The set T of points in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≤ 5) ∨
               (5 = y - 2 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≥ x + 3)}

/-- The common endpoint of the three rays -/
def common_endpoint : ℝ × ℝ := (2, 7)

/-- The three rays that form set T -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 7}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 7 ∧ p.1 ≤ 2}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 5 ∧ p.1 ≤ 2}

/-- Theorem stating that T consists of three rays with a common endpoint -/
theorem T_is_three_rays_with_common_endpoint :
  T = ray1 ∪ ray2 ∪ ray3 ∧
  common_endpoint ∈ ray1 ∧
  common_endpoint ∈ ray2 ∧
  common_endpoint ∈ ray3 :=
sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l452_45206


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l452_45293

theorem integer_pairs_satisfying_equation :
  ∀ x y : ℤ, y ≥ 0 → (x^2 + 2*x*y + Nat.factorial y.toNat = 131) ↔ ((x = 1 ∧ y = 5) ∨ (x = -11 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l452_45293


namespace NUMINAMATH_CALUDE_smartphone_cost_smartphone_cost_proof_l452_45280

theorem smartphone_cost (initial_savings : ℕ) (saving_months : ℕ) (weeks_per_month : ℕ) (weekly_savings : ℕ) : ℕ :=
  let total_weeks := saving_months * weeks_per_month
  let total_savings := weekly_savings * total_weeks
  initial_savings + total_savings

#check smartphone_cost 40 2 4 15 = 160

theorem smartphone_cost_proof :
  smartphone_cost 40 2 4 15 = 160 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_cost_smartphone_cost_proof_l452_45280


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l452_45215

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x ≤ 0}

-- State the theorem
theorem quadratic_inequality_theorem 
  (a b c : ℝ) 
  (h : solution_set a b c = {x : ℝ | x ≤ -1 ∨ x ≥ 3}) :
  (a + b + c > 0) ∧ 
  (4*a - 2*b + c < 0) ∧ 
  ({x : ℝ | c*x^2 - b*x + a < 0} = {x : ℝ | -1/3 < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l452_45215


namespace NUMINAMATH_CALUDE_arrangements_theorem_l452_45238

/-- The number of people in the row -/
def n : ℕ := 6

/-- The number of different arrangements where both person A and person B
    are on the same side of person C -/
def arrangements_same_side (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem arrangements_theorem :
  arrangements_same_side n = 480 :=
sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l452_45238


namespace NUMINAMATH_CALUDE_train_length_proof_l452_45292

theorem train_length_proof (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) 
  (h1 : bridge_length = 800)
  (h2 : bridge_time = 45)
  (h3 : post_time = 15) :
  ∃ train_length : ℝ, train_length = 400 ∧ 
  train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l452_45292


namespace NUMINAMATH_CALUDE_sequence_properties_l452_45254

/-- Definition of sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- Definition of S_n as the sum of first n terms of a_n -/
def S (n : ℕ) : ℝ := sorry

/-- Definition of sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- Definition of sequence c_n -/
def c (n : ℕ) : ℝ := a n * b n

/-- Definition of T_n as the sum of first n terms of c_n -/
def T (n : ℕ) : ℝ := sorry

/-- Main theorem -/
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = (S n + 2) / 2) ∧ 
  (b 1 = 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n - b (n + 1) + 2 = 0) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2^n) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2*n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = (2*n - 3) * 2^(n+1) + 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l452_45254
