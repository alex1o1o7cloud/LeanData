import Mathlib

namespace NUMINAMATH_CALUDE_ordinary_day_probability_l1360_136065

/-- Probability of shark appearance on any given day -/
def P_shark_appearance : ℚ := 1 / 30

/-- Probability of system detecting a shark when present -/
def P_detection_given_shark : ℚ := 3 / 4

/-- Probability of false alarm given no shark -/
def P_false_alarm_given_no_shark : ℚ := 10 * P_shark_appearance

/-- Theorem: The probability of an "ordinary" day (no sharks and no false alarms) is 29/45 -/
theorem ordinary_day_probability : 
  let P_no_shark : ℚ := 1 - P_shark_appearance
  let P_no_alarm_given_no_shark : ℚ := 1 - P_false_alarm_given_no_shark
  P_no_shark * P_no_alarm_given_no_shark = 29 / 45 := by
  sorry

end NUMINAMATH_CALUDE_ordinary_day_probability_l1360_136065


namespace NUMINAMATH_CALUDE_mark_deck_project_cost_l1360_136057

/-- The total cost of Mark's deck project -/
def deck_project_cost (length width : ℝ) (cost_A cost_B cost_sealant : ℝ) 
  (percent_A : ℝ) (tax_rate : ℝ) : ℝ :=
let total_area := length * width
let area_A := percent_A * total_area
let area_B := (1 - percent_A) * total_area
let cost_materials := cost_A * area_A + cost_B * area_B
let cost_sealant_total := cost_sealant * total_area
let subtotal := cost_materials + cost_sealant_total
subtotal * (1 + tax_rate)

/-- Theorem stating the total cost of Mark's deck project -/
theorem mark_deck_project_cost :
  deck_project_cost 30 40 3 5 1 0.6 0.07 = 6163.20 := by
  sorry


end NUMINAMATH_CALUDE_mark_deck_project_cost_l1360_136057


namespace NUMINAMATH_CALUDE_work_left_fraction_l1360_136020

theorem work_left_fraction (days_A days_B days_together : ℕ) 
  (h1 : days_A = 15)
  (h2 : days_B = 20)
  (h3 : days_together = 6) : 
  1 - (days_together : ℚ) * (1 / days_A + 1 / days_B) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l1360_136020


namespace NUMINAMATH_CALUDE_inequality_proof_l1360_136094

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1360_136094


namespace NUMINAMATH_CALUDE_total_triangles_is_18_l1360_136061

/-- Represents a figure with different types of triangles -/
structure TriangleFigure where
  smallest : Nat
  medium : Nat
  largest : Nat

/-- Calculates the total number of triangles in a TriangleFigure -/
def totalTriangles (figure : TriangleFigure) : Nat :=
  figure.smallest + figure.medium + figure.largest

/-- The given figure with 6 smallest, 7 medium, and 5 largest triangles -/
def givenFigure : TriangleFigure :=
  { smallest := 6, medium := 7, largest := 5 }

/-- Theorem stating that the total number of triangles in the given figure is 18 -/
theorem total_triangles_is_18 : totalTriangles givenFigure = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_18_l1360_136061


namespace NUMINAMATH_CALUDE_number_of_complementary_sets_l1360_136085

/-- Represents a card in the deck -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3
  size : Fin 3

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- Checks if a set of three cards is complementary -/
def isComplementary (c1 c2 c3 : Card) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementarySets : Finset (Finset Card) := sorry

theorem number_of_complementary_sets :
  Finset.card complementarySets = 4536 := by sorry

end NUMINAMATH_CALUDE_number_of_complementary_sets_l1360_136085


namespace NUMINAMATH_CALUDE_cookie_distribution_l1360_136024

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) : 
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1360_136024


namespace NUMINAMATH_CALUDE_point_outside_circle_if_line_intersects_l1360_136063

/-- A line intersects a circle at two distinct points if and only if
    the distance from the center of the circle to the line is less than the radius -/
axiom line_intersects_circle_iff_distance_lt_radius {a b : ℝ} :
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ≠ (x₂, y₂) ∧
    a * x₁ + b * y₁ = 1 ∧ x₁^2 + y₁^2 = 1 ∧
    a * x₂ + b * y₂ = 1 ∧ x₂^2 + y₂^2 = 1) ↔
  (1 / (a^2 + b^2).sqrt < 1)

theorem point_outside_circle_if_line_intersects
  (a b : ℝ)
  (h_intersect : ∃ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ≠ (x₂, y₂) ∧
    a * x₁ + b * y₁ = 1 ∧ x₁^2 + y₁^2 = 1 ∧
    a * x₂ + b * y₂ = 1 ∧ x₂^2 + y₂^2 = 1) :
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_if_line_intersects_l1360_136063


namespace NUMINAMATH_CALUDE_factorization_equality_l1360_136053

theorem factorization_equality (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1360_136053


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1360_136044

theorem rectangular_to_polar_conversion :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := -Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0
                then 2 * Real.pi - Real.arctan ((-y) / x)
                else 0  -- This else case is just a placeholder
  (r = Real.sqrt 6 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1360_136044


namespace NUMINAMATH_CALUDE_bottle_filling_proportion_l1360_136028

/-- Given two bottles with capacities of 4 and 8 cups, and a total of 8 cups of milk,
    prove that the proportion of capacity each bottle should be filled to is 2/3,
    when the 8-cup bottle contains 5.333333333333333 cups of milk. -/
theorem bottle_filling_proportion :
  let total_milk : ℚ := 8
  let bottle1_capacity : ℚ := 4
  let bottle2_capacity : ℚ := 8
  let milk_in_bottle2 : ℚ := 5.333333333333333
  let proportion : ℚ := milk_in_bottle2 / bottle2_capacity
  proportion = 2/3 ∧ 
  bottle1_capacity * proportion + bottle2_capacity * proportion = total_milk :=
by sorry

end NUMINAMATH_CALUDE_bottle_filling_proportion_l1360_136028


namespace NUMINAMATH_CALUDE_prob_kings_or_aces_l1360_136056

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Probability of drawing three kings -/
def probThreeKings : ℚ :=
  (KingsInDeck / StandardDeck) * ((KingsInDeck - 1) / (StandardDeck - 1)) * ((KingsInDeck - 2) / (StandardDeck - 2))

/-- Probability of drawing exactly two aces -/
def probTwoAces : ℚ :=
  3 * (AcesInDeck / StandardDeck) * ((AcesInDeck - 1) / (StandardDeck - 1)) * ((StandardDeck - AcesInDeck) / (StandardDeck - 2))

/-- Probability of drawing three aces -/
def probThreeAces : ℚ :=
  (AcesInDeck / StandardDeck) * ((AcesInDeck - 1) / (StandardDeck - 1)) * ((AcesInDeck - 2) / (StandardDeck - 2))

/-- The probability of drawing either three kings or at least 2 aces when selecting 3 cards from a standard 52-card deck -/
theorem prob_kings_or_aces : probThreeKings + probTwoAces + probThreeAces = 43 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_prob_kings_or_aces_l1360_136056


namespace NUMINAMATH_CALUDE_joans_remaining_practice_time_l1360_136078

/-- Given Joan's music practice schedule, calculate the remaining time for finger exercises. -/
theorem joans_remaining_practice_time :
  let total_time : ℕ := 2 * 60  -- 2 hours in minutes
  let piano_time : ℕ := 30
  let writing_time : ℕ := 25
  let reading_time : ℕ := 38
  let used_time : ℕ := piano_time + writing_time + reading_time
  total_time - used_time = 27 := by
  sorry

#check joans_remaining_practice_time

end NUMINAMATH_CALUDE_joans_remaining_practice_time_l1360_136078


namespace NUMINAMATH_CALUDE_dongwi_festival_cases_l1360_136049

/-- The number of cases in which Dongwi can go to play at the festival. -/
def num_cases (boys_schools : ℕ) (girls_schools : ℕ) : ℕ :=
  boys_schools + girls_schools

/-- Theorem stating that the number of cases for Dongwi to go to play is 7. -/
theorem dongwi_festival_cases :
  let boys_schools := 4
  let girls_schools := 3
  num_cases boys_schools girls_schools = 7 := by
  sorry

end NUMINAMATH_CALUDE_dongwi_festival_cases_l1360_136049


namespace NUMINAMATH_CALUDE_parallelogram_area_is_37_l1360_136089

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : Fin 2 → ℤ) : ℕ :=
  (v 0 * w 1 - v 1 * w 0).natAbs

/-- Vectors v and w -/
def v : Fin 2 → ℤ := ![7, -5]
def w : Fin 2 → ℤ := ![13, -4]

/-- Theorem: The area of the parallelogram formed by v and w is 37 -/
theorem parallelogram_area_is_37 : parallelogramArea v w = 37 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_37_l1360_136089


namespace NUMINAMATH_CALUDE_x_squared_in_set_l1360_136034

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l1360_136034


namespace NUMINAMATH_CALUDE_expression_factorization_l1360_136077

theorem expression_factorization (b : ℝ) : 
  (8 * b^3 - 104 * b^2 + 9) - (9 * b^3 - 2 * b^2 + 9) = -b^2 * (b + 102) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1360_136077


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_two_plus_i_l1360_136045

theorem imaginary_part_of_i_times_two_plus_i (i : ℂ) : 
  (i * i = -1) → Complex.im (i * (2 + i)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_two_plus_i_l1360_136045


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1360_136016

/-- Given a line segment with midpoint (5, -8) and one endpoint at (7, 2),
    the sum of the coordinates of the other endpoint is -15. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (5 = (x + 7) / 2) →
    (-8 = (y + 2) / 2) →
    x + y = -15 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1360_136016


namespace NUMINAMATH_CALUDE_binary_representation_sum_of_exponents_l1360_136099

theorem binary_representation_sum_of_exponents (n : ℕ) (h : n = 2023) :
  (Nat.digits 2 n).sum = 48 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_sum_of_exponents_l1360_136099


namespace NUMINAMATH_CALUDE_richard_solves_1099_problems_l1360_136003

/-- The number of problems Richard solves in 2013 --/
def problems_solved_2013 : ℕ :=
  let days_in_2013 : ℕ := 365
  let problems_per_week : ℕ := 2 + 1 + 2 + 1 + 2 + 5 + 7
  let full_weeks : ℕ := days_in_2013 / 7
  let extra_day : ℕ := days_in_2013 % 7
  let normal_tuesday_problems : ℕ := 1
  let special_tuesday_problems : ℕ := 60
  full_weeks * problems_per_week + extra_day * normal_tuesday_problems + 
    (special_tuesday_problems - normal_tuesday_problems)

theorem richard_solves_1099_problems : problems_solved_2013 = 1099 := by
  sorry

end NUMINAMATH_CALUDE_richard_solves_1099_problems_l1360_136003


namespace NUMINAMATH_CALUDE_walking_speed_is_10_l1360_136001

/-- Represents the walking speed of person A in km/h -/
def walking_speed : ℝ := 10

/-- Represents the cycling speed of person B in km/h -/
def cycling_speed : ℝ := 20

/-- Represents the time difference in hours between when A starts walking and B starts cycling -/
def time_difference : ℝ := 6

/-- Represents the distance in km at which B catches up with A -/
def catch_up_distance : ℝ := 120

theorem walking_speed_is_10 : 
  walking_speed = 10 ∧ 
  cycling_speed = 20 ∧
  time_difference = 6 ∧
  catch_up_distance = 120 ∧
  ∃ t : ℝ, t > time_difference ∧ 
        walking_speed * t = catch_up_distance ∧ 
        cycling_speed * (t - time_difference) = catch_up_distance :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_is_10_l1360_136001


namespace NUMINAMATH_CALUDE_daughter_age_in_three_years_l1360_136079

/-- Given that 5 years ago a mother was twice as old as her daughter,
    and the mother is 41 years old now, prove that the daughter
    will be 26 years old in 3 years. -/
theorem daughter_age_in_three_years
  (mother_age_now : ℕ)
  (h1 : mother_age_now = 41)
  (h2 : mother_age_now - 5 = 2 * ((mother_age_now - 5) / 2)) :
  ((mother_age_now - 5) / 2) + 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_daughter_age_in_three_years_l1360_136079


namespace NUMINAMATH_CALUDE_quadratic_root_l1360_136080

theorem quadratic_root (b : ℝ) : 
  (1 : ℝ) ^ 2 + b * 1 + 2 = 0 → ∃ x : ℝ, x ≠ 1 ∧ x ^ 2 + b * x + 2 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l1360_136080


namespace NUMINAMATH_CALUDE_max_rectangles_equals_black_squares_l1360_136012

/-- Represents a figure that can be cut into squares and rectangles -/
structure Figure where
  shape : Set (ℕ × ℕ)  -- Set of coordinates representing the shape

/-- Counts the number of black squares when coloring the middle diagonal -/
def count_black_squares (f : Figure) : ℕ :=
  sorry

/-- Represents the specific figure given in the problem -/
def given_figure : Figure :=
  { shape := sorry }

/-- The maximum number of 1×2 rectangles that can be obtained -/
def max_rectangles (f : Figure) : ℕ :=
  sorry

theorem max_rectangles_equals_black_squares :
  max_rectangles given_figure = count_black_squares given_figure ∧
  count_black_squares given_figure = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_equals_black_squares_l1360_136012


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1360_136093

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 4, -2; 0, 3, 1; 5, -1, 3]
  Matrix.det A = 70 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1360_136093


namespace NUMINAMATH_CALUDE_g_formula_l1360_136002

noncomputable def g (a : ℝ) : ℝ :=
  let m := Real.exp (Real.log 2 * min a 2)
  let n := Real.exp (Real.log 2 * max (-2) a)
  n - m

theorem g_formula (a : ℝ) (ha : a ≥ 0) :
  g a = if a ≤ 2 then -3 else 1 - Real.exp (Real.log 2 * a) := by
  sorry

end NUMINAMATH_CALUDE_g_formula_l1360_136002


namespace NUMINAMATH_CALUDE_zoo_field_trip_l1360_136031

/-- Calculates the number of individuals left at the zoo after a field trip --/
theorem zoo_field_trip (initial_fifth_grade : ℕ) (merged_fifth_grade : ℕ) 
  (initial_chaperones : ℕ) (teachers : ℕ) (third_grade : ℕ) 
  (additional_chaperones : ℕ) (fifth_grade_left : ℕ) (third_grade_left : ℕ) 
  (chaperones_left : ℕ) : 
  initial_fifth_grade = 10 →
  merged_fifth_grade = 12 →
  initial_chaperones = 5 →
  teachers = 2 →
  third_grade = 15 →
  additional_chaperones = 3 →
  fifth_grade_left = 10 →
  third_grade_left = 6 →
  chaperones_left = 2 →
  initial_fifth_grade + merged_fifth_grade + initial_chaperones + teachers + 
    third_grade + additional_chaperones - 
    (fifth_grade_left + third_grade_left + chaperones_left) = 29 := by
  sorry


end NUMINAMATH_CALUDE_zoo_field_trip_l1360_136031


namespace NUMINAMATH_CALUDE_largest_value_l1360_136051

theorem largest_value : 
  max (5 * Real.sqrt 2 - 7) (max (7 - 5 * Real.sqrt 2) (max |4/4 - 4/4| 0.1)) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l1360_136051


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_union_equals_A_iff_l1360_136035

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1) - Real.sqrt (x + 2)}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3*x + a = 0}

-- Statement 1: When a = 2, A ∩ B = {2}
theorem intersection_when_a_is_two :
  A ∩ B 2 = {2} := by sorry

-- Statement 2: A ∪ B = A if and only if a ∈ (2, +∞)
theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_union_equals_A_iff_l1360_136035


namespace NUMINAMATH_CALUDE_rectangle_width_l1360_136083

/-- Given a rectangle with perimeter 16 cm and width 2 cm longer than length, prove its width is 5 cm -/
theorem rectangle_width (length width : ℝ) : 
  (2 * (length + width) = 16) →  -- Perimeter is 16 cm
  (width = length + 2) →         -- Width is 2 cm longer than length
  width = 5 := by               -- Prove width is 5 cm
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1360_136083


namespace NUMINAMATH_CALUDE_cube_monotone_l1360_136043

theorem cube_monotone (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_monotone_l1360_136043


namespace NUMINAMATH_CALUDE_circle_centers_distance_l1360_136060

-- Define the circles and their properties
structure CirclePair where
  r : ℝ  -- radius of the smaller circle
  R : ℝ  -- radius of the larger circle
  common_chord : ℝ  -- length of the common chord

-- Define the theorem
theorem circle_centers_distance (c : CirclePair) :
  (c.r > 0) →  -- ensure positive radius
  (c.common_chord = c.r * Real.sqrt 2) →  -- common chord is side of square in smaller circle
  (c.R = c.r * Real.sqrt 2) →  -- radius of larger circle
  (∃ d : ℝ, (d = (c.r * (Real.sqrt 6 + Real.sqrt 2)) / 2) ∨
            (d = (c.r * (Real.sqrt 6 - Real.sqrt 2)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l1360_136060


namespace NUMINAMATH_CALUDE_unique_pair_l1360_136030

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_pair : 
  ∃! (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    b > a ∧ 
    is_prime (b - a) ∧ 
    (a + b) % 10 = 3 ∧ 
    ∃ k : ℕ, a * b = k * k ∧
    a = 4 ∧
    b = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_l1360_136030


namespace NUMINAMATH_CALUDE_star_two_neg_three_l1360_136042

-- Define the ★ operation
def star (a b : ℤ) : ℤ := a * b^3 - 2*b + 2

-- Theorem statement
theorem star_two_neg_three : star 2 (-3) = -46 := by
  sorry

end NUMINAMATH_CALUDE_star_two_neg_three_l1360_136042


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1360_136073

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1360_136073


namespace NUMINAMATH_CALUDE_equation_solution_l1360_136062

theorem equation_solution (x : ℝ) : (4 * x + 2) / (5 * x - 5) = 3 / 4 → x = -23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1360_136062


namespace NUMINAMATH_CALUDE_periodic_function_2009_l1360_136017

/-- A function satisfying the given functional equation -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) * (1 - f x) = 1 + f x

theorem periodic_function_2009 (f : ℝ → ℝ) 
  (h1 : PeriodicFunction f) 
  (h2 : f 5 = 2 + Real.sqrt 3) : 
  f 2009 = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_2009_l1360_136017


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l1360_136009

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ :=
  4 * s.side

/-- Theorem stating the perimeter of the large rectangle -/
theorem large_rectangle_perimeter 
  (square : Square)
  (small_rect : Rectangle)
  (h1 : square.perimeter = 24)
  (h2 : small_rect.perimeter = 16)
  (h3 : small_rect.length = square.side)
  (h4 : small_rect.width + square.side = small_rect.length) :
  let large_rect := Rectangle.mk (square.side + 2 * small_rect.length) (small_rect.width + square.side)
  large_rect.perimeter = 52 := by
  sorry


end NUMINAMATH_CALUDE_large_rectangle_perimeter_l1360_136009


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1360_136007

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 2 + a 8 = 180 → a 3 + a 4 + a 5 + a 6 + a 7 = 450 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1360_136007


namespace NUMINAMATH_CALUDE_correct_statements_l1360_136010

theorem correct_statements (x : ℝ) : 
  (x ≥ 0 → x^2 ≥ x) ∧ 
  (x^2 ≥ 0 → abs x ≥ 0) ∧ 
  (x ≤ -1 → x^2 ≥ abs x) := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_l1360_136010


namespace NUMINAMATH_CALUDE_opposite_sign_square_root_l1360_136006

theorem opposite_sign_square_root (a b : ℝ) : 
  (|2*a - 4| + Real.sqrt (3*b + 12) = 0) → 
  Real.sqrt (2*a - 3*b) = 4 ∨ Real.sqrt (2*a - 3*b) = -4 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_square_root_l1360_136006


namespace NUMINAMATH_CALUDE_shaded_area_in_grid_l1360_136097

/-- The area of a shape in a 3x3 grid formed by a 3x1 rectangle with one 1x1 square removed -/
theorem shaded_area_in_grid (grid_size : Nat) (square_side_length : ℝ) 
  (h1 : grid_size = 3) 
  (h2 : square_side_length = 1) : ℝ := by
  sorry

#check shaded_area_in_grid

end NUMINAMATH_CALUDE_shaded_area_in_grid_l1360_136097


namespace NUMINAMATH_CALUDE_sara_flowers_l1360_136072

theorem sara_flowers (yellow_flowers : ℕ) (num_bouquets : ℕ) (red_flowers : ℕ) :
  yellow_flowers = 24 →
  num_bouquets = 8 →
  yellow_flowers % num_bouquets = 0 →
  red_flowers = yellow_flowers →
  red_flowers = 24 := by
sorry

end NUMINAMATH_CALUDE_sara_flowers_l1360_136072


namespace NUMINAMATH_CALUDE_no_real_roots_k_value_l1360_136038

theorem no_real_roots_k_value (k : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → k / (x - 1) + 3 ≠ x / (1 - x)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_k_value_l1360_136038


namespace NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l1360_136075

theorem coefficient_m5n5_in_expansion : ∀ m n : ℕ,
  (Nat.choose 10 5 : ℕ) = 252 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l1360_136075


namespace NUMINAMATH_CALUDE_total_count_is_900_l1360_136032

/-- Represents the count of type A components -/
def a : ℕ := 400

/-- Represents the count of type B components -/
def b : ℕ := 300

/-- Represents the count of type C components -/
def c : ℕ := 200

/-- Represents the total sample size -/
def sample_size : ℕ := 45

/-- Represents the number of type C components sampled -/
def c_sampled : ℕ := 10

/-- Represents the total count of all components -/
def total_count : ℕ := a + b + c

/-- Theorem stating that the total count of all components is 900 -/
theorem total_count_is_900 : total_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_count_is_900_l1360_136032


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l1360_136039

/-- Given an ellipse and a line intersecting it, proves the relationship between the slopes of the intersecting line and the line connecting the origin to the midpoint of the intersection points. -/
theorem ellipse_line_intersection_slope_product (k1 k2 : ℝ) 
  (h1 : k1 ≠ 0) 
  (h2 : ∃ (P1 P2 P : ℝ × ℝ), 
    (P1.1^2 + 2*P1.2^2 = 2) ∧ 
    (P2.1^2 + 2*P2.2^2 = 2) ∧ 
    (P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2)) ∧ 
    (k1 = (P2.2 - P1.2)/(P2.1 - P1.1)) ∧ 
    (k2 = P.2/P.1)) : 
  k1 * k2 = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l1360_136039


namespace NUMINAMATH_CALUDE_range_of_m_l1360_136087

-- Define the propositions p and q
def p (m : ℝ) : Prop := 4 < m ∧ m < 10

def q (m : ℝ) : Prop := 8 < m ∧ m < 12

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1360_136087


namespace NUMINAMATH_CALUDE_total_crayons_l1360_136055

theorem total_crayons (billy jane mike sue : ℕ) 
  (h1 : billy = 62) 
  (h2 : jane = 52) 
  (h3 : mike = 78) 
  (h4 : sue = 97) : 
  billy + jane + mike + sue = 289 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1360_136055


namespace NUMINAMATH_CALUDE_probability_four_ones_eight_dice_l1360_136069

theorem probability_four_ones_eight_dice : 
  let n : ℕ := 8  -- number of dice
  let s : ℕ := 8  -- number of sides on each die
  let k : ℕ := 4  -- number of dice showing 1
  Nat.choose n k * (1 / s) ^ k * ((s - 1) / s) ^ (n - k) = 168070 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_eight_dice_l1360_136069


namespace NUMINAMATH_CALUDE_shoes_theorem_l1360_136011

/-- Given an initial number of shoe pairs and a number of lost individual shoes,
    calculate the maximum number of complete pairs remaining. -/
def maxRemainingPairs (initialPairs : ℕ) (lostShoes : ℕ) : ℕ :=
  initialPairs - lostShoes

/-- Theorem: Given 26 initial pairs of shoes and losing 9 individual shoes,
    the maximum number of complete pairs remaining is 17. -/
theorem shoes_theorem :
  maxRemainingPairs 26 9 = 17 := by
  sorry

#eval maxRemainingPairs 26 9

end NUMINAMATH_CALUDE_shoes_theorem_l1360_136011


namespace NUMINAMATH_CALUDE_new_person_age_l1360_136046

theorem new_person_age (T : ℕ) : 
  T > 0 →  -- Ensure total age is positive
  (T / 10 : ℚ) - ((T - 48 + 18) / 10 : ℚ) = 3 →
  18 = 18 := by sorry

end NUMINAMATH_CALUDE_new_person_age_l1360_136046


namespace NUMINAMATH_CALUDE_zoo_feeding_sequences_l1360_136058

def number_of_animal_pairs : ℕ := 5

def alternating_feeding_sequences (n : ℕ) : ℕ :=
  (Nat.factorial n) * (Nat.factorial n)

theorem zoo_feeding_sequences :
  alternating_feeding_sequences number_of_animal_pairs = 14400 :=
by sorry

end NUMINAMATH_CALUDE_zoo_feeding_sequences_l1360_136058


namespace NUMINAMATH_CALUDE_probability_all_co_captains_value_l1360_136033

def team_sizes : List Nat := [6, 9, 10]
def co_captains_per_team : Nat := 3
def num_teams : Nat := 3

def probability_all_co_captains : ℚ :=
  (1 : ℚ) / num_teams *
  (team_sizes.map (λ n => (co_captains_per_team : ℚ) / (n * (n - 1) * (n - 2)))).sum

theorem probability_all_co_captains_value :
  probability_all_co_captains = 59 / 2520 := by
  sorry

#eval probability_all_co_captains

end NUMINAMATH_CALUDE_probability_all_co_captains_value_l1360_136033


namespace NUMINAMATH_CALUDE_max_correct_answers_for_given_test_l1360_136059

/-- Represents a multiple choice test with scoring system -/
structure MCTest where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (test : MCTest) : ℕ :=
  sorry

/-- Theorem stating the maximum number of correct answers for the given test -/
theorem max_correct_answers_for_given_test :
  let test : MCTest := {
    total_questions := 60,
    correct_points := 3,
    incorrect_points := -2,
    total_score := 126
  }
  max_correct_answers test = 49 := by sorry

end NUMINAMATH_CALUDE_max_correct_answers_for_given_test_l1360_136059


namespace NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l1360_136067

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  (r.top_right.1 - r.bottom_left.1) * (r.top_right.2 - r.bottom_left.2)

/-- Checks if a rectangle is inscribed in another rectangle -/
def is_inscribed (inner outer : Rectangle) : Prop :=
  inner.bottom_left.1 ≥ outer.bottom_left.1 ∧
  inner.bottom_left.2 ≥ outer.bottom_left.2 ∧
  inner.top_right.1 ≤ outer.top_right.1 ∧
  inner.top_right.2 ≤ outer.top_right.2

/-- Checks if two rectangles share a vertex on the given side -/
def share_vertex_on_side (r1 r2 outer : Rectangle) (side : ℝ) : Prop :=
  (r1.bottom_left.1 = side ∨ r1.top_right.1 = side) ∧
  (r2.bottom_left.1 = side ∨ r2.top_right.1 = side) ∧
  ∃ y, (r1.bottom_left.2 = y ∨ r1.top_right.2 = y) ∧
       (r2.bottom_left.2 = y ∨ r2.top_right.2 = y)

theorem inscribed_rectangles_area_sum (outer r1 r2 : Rectangle) :
  is_inscribed r1 outer →
  is_inscribed r2 outer →
  share_vertex_on_side r1 r2 outer outer.bottom_left.1 →
  area r1 + area r2 = area outer := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l1360_136067


namespace NUMINAMATH_CALUDE_sum_areas_eighteen_disks_l1360_136086

/-- The sum of areas of 18 congruent disks arranged on a unit circle --/
theorem sum_areas_eighteen_disks : ℝ := by
  -- Define the number of disks
  let n : ℕ := 18

  -- Define the radius of the large circle
  let R : ℝ := 1

  -- Define the central angle for each disk
  let central_angle : ℝ := 2 * Real.pi / n

  -- Define the radius of each small disk
  let r : ℝ := Real.tan (central_angle / 2)

  -- Define the area of a single disk
  let single_disk_area : ℝ := Real.pi * r^2

  -- Define the sum of areas of all disks
  let total_area : ℝ := n * single_disk_area

  -- The theorem statement
  have : total_area = 18 * Real.pi * (Real.tan (Real.pi / 18))^2 := by sorry

  -- Return the result
  exact total_area


end NUMINAMATH_CALUDE_sum_areas_eighteen_disks_l1360_136086


namespace NUMINAMATH_CALUDE_bella_steps_theorem_l1360_136068

/-- The number of feet in a mile -/
def feet_per_mile : ℕ := 5280

/-- The distance between the two houses in miles -/
def distance_miles : ℕ := 3

/-- The length of Bella's step in feet -/
def step_length : ℕ := 3

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℕ := 4

/-- The number of steps Bella takes when they meet -/
def steps_taken : ℕ := 1056

theorem bella_steps_theorem :
  let total_distance_feet := distance_miles * feet_per_mile
  let combined_speed_ratio := speed_ratio + 1
  let bella_distance := total_distance_feet / combined_speed_ratio
  bella_distance / step_length = steps_taken := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_theorem_l1360_136068


namespace NUMINAMATH_CALUDE_thirteen_power_mod_thirtyseven_l1360_136047

theorem thirteen_power_mod_thirtyseven (a : ℕ+) (h : 3 ∣ a.val) :
  (13 : ℤ)^(a.val) ≡ 1 [ZMOD 37] := by
  sorry

end NUMINAMATH_CALUDE_thirteen_power_mod_thirtyseven_l1360_136047


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l1360_136066

theorem fixed_point_power_function (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x ^ α) →
  f 2 = Real.sqrt 2 / 2 →
  f 9 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l1360_136066


namespace NUMINAMATH_CALUDE_bug_path_tiles_l1360_136084

def width : ℕ := 15
def length : ℕ := 25
def total_tiles : ℕ := 375

theorem bug_path_tiles : 
  width + length - Nat.gcd width length = 35 := by sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l1360_136084


namespace NUMINAMATH_CALUDE_circle_radius_l1360_136090

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y = 0

-- Define the theorem
theorem circle_radius : 
  ∀ m : ℝ, 
  (∃ M N : ℝ × ℝ, 
    circle_equation M.1 M.2 m ∧ 
    circle_equation N.1 N.2 m ∧ 
    (∃ k : ℝ, symmetry_line ((M.1 + N.1)/2) ((M.2 + N.2)/2))) →
  (∃ center : ℝ × ℝ, ∀ x y : ℝ, 
    circle_equation x y m ↔ (x - center.1)^2 + (y - center.2)^2 = 3^2) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l1360_136090


namespace NUMINAMATH_CALUDE_cannot_cut_square_l1360_136064

theorem cannot_cut_square (rectangle_area : ℝ) (square_area : ℝ) 
  (h_rectangle_area : rectangle_area = 582) 
  (h_square_area : square_area = 400) : ¬ ∃ (l w : ℝ), 
  l * w = rectangle_area ∧ 
  l / w = 3 / 2 ∧ 
  w ≥ Real.sqrt square_area := by
sorry

end NUMINAMATH_CALUDE_cannot_cut_square_l1360_136064


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1360_136029

theorem inscribed_cube_surface_area (r : ℝ) (h : 4 * π * r^2 = π) :
  6 * (1 / (r * Real.sqrt 3))^2 = 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1360_136029


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1360_136070

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 4) (hb : b = 9) (hc : c = 36) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1360_136070


namespace NUMINAMATH_CALUDE_largest_and_smallest_A_l1360_136036

/-- A function that moves the last digit of a number to the first position -/
def moveLastDigitToFirst (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let restOfDigits := n / 10
  lastDigit * 10^8 + restOfDigits

/-- Theorem stating the largest and smallest A values -/
theorem largest_and_smallest_A :
  ∀ B : ℕ,
  (B > 22222222) →
  (Nat.gcd B 18 = 1) →
  (∃ A : ℕ, A = moveLastDigitToFirst B) →
  (∃ A_max A_min : ℕ,
    (A_max = moveLastDigitToFirst B → A_max ≤ 999999998) ∧
    (A_min = moveLastDigitToFirst B → A_min ≥ 122222224) ∧
    (∃ B_max B_min : ℕ,
      B_max > 22222222 ∧
      Nat.gcd B_max 18 = 1 ∧
      moveLastDigitToFirst B_max = 999999998 ∧
      B_min > 22222222 ∧
      Nat.gcd B_min 18 = 1 ∧
      moveLastDigitToFirst B_min = 122222224)) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_and_smallest_A_l1360_136036


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1360_136004

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) (h2 : M ∩ N r = N r) :
  r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1360_136004


namespace NUMINAMATH_CALUDE_squares_in_3x3_lattice_l1360_136014

/-- A point in a 2D lattice -/
structure LatticePoint where
  x : ℕ
  y : ℕ

/-- A square lattice -/
structure SquareLattice where
  size : ℕ
  points : List LatticePoint

/-- A square formed by four lattice points -/
structure LatticeSquare where
  vertices : List LatticePoint

/-- Function to check if four points form a valid square in the lattice -/
def is_valid_square (l : SquareLattice) (s : LatticeSquare) : Prop :=
  sorry

/-- Function to count the number of valid squares in a lattice -/
def count_squares (l : SquareLattice) : ℕ :=
  sorry

/-- Theorem: The number of squares in a 3x3 square lattice is 5 -/
theorem squares_in_3x3_lattice :
  ∀ (l : SquareLattice), l.size = 3 → count_squares l = 5 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_3x3_lattice_l1360_136014


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l1360_136019

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 168 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l1360_136019


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1360_136040

theorem simplify_sqrt_expression :
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1360_136040


namespace NUMINAMATH_CALUDE_phantom_needs_43_more_l1360_136041

/-- The amount of money Phantom's mom gave him initially -/
def initial_amount : ℕ := 50

/-- The cost of one black printer ink -/
def black_ink_cost : ℕ := 11

/-- The number of black printer inks Phantom wants to buy -/
def black_ink_count : ℕ := 2

/-- The cost of one red printer ink -/
def red_ink_cost : ℕ := 15

/-- The number of red printer inks Phantom wants to buy -/
def red_ink_count : ℕ := 3

/-- The cost of one yellow printer ink -/
def yellow_ink_cost : ℕ := 13

/-- The number of yellow printer inks Phantom wants to buy -/
def yellow_ink_count : ℕ := 2

/-- The additional amount Phantom needs to ask his mom -/
def additional_amount : ℕ := 43

theorem phantom_needs_43_more :
  (black_ink_cost * black_ink_count +
   red_ink_cost * red_ink_count +
   yellow_ink_cost * yellow_ink_count) - initial_amount = additional_amount := by
  sorry

end NUMINAMATH_CALUDE_phantom_needs_43_more_l1360_136041


namespace NUMINAMATH_CALUDE_specific_committee_selection_l1360_136027

/-- The number of ways to choose a committee with a specific person included -/
def committee_selection (n m k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 462 ways to choose a 6-person committee from 12 people with one specific person included -/
theorem specific_committee_selection :
  committee_selection 12 6 1 = 462 := by
  sorry

end NUMINAMATH_CALUDE_specific_committee_selection_l1360_136027


namespace NUMINAMATH_CALUDE_future_cup_defense_l1360_136050

/-- Represents the defensive statistics of a class --/
structure DefensiveStats where
  avgGoalsConceded : ℝ
  stdDevGoalsConceded : ℝ

/-- Determines if one class has better average defensive performance than another --/
def betterAverageDefense (a b : DefensiveStats) : Prop :=
  a.avgGoalsConceded > b.avgGoalsConceded

/-- Determines if one class has less stable defensive performance than another --/
def lessStableDefense (a b : DefensiveStats) : Prop :=
  a.stdDevGoalsConceded > b.stdDevGoalsConceded

/-- Determines if a class has relatively consistent defensive performance --/
def consistentDefense (a : DefensiveStats) : Prop :=
  a.stdDevGoalsConceded < 0.5

theorem future_cup_defense 
  (classA classB : DefensiveStats)
  (hA : classA.avgGoalsConceded = 1.9 ∧ classA.stdDevGoalsConceded = 0.3)
  (hB : classB.avgGoalsConceded = 1.3 ∧ classB.stdDevGoalsConceded = 1.2) :
  betterAverageDefense classA classB ∧ 
  lessStableDefense classB classA ∧ 
  consistentDefense classA := by
  sorry

end NUMINAMATH_CALUDE_future_cup_defense_l1360_136050


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1360_136008

theorem quadratic_equation_solution : 
  ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1360_136008


namespace NUMINAMATH_CALUDE_b_work_time_l1360_136022

/-- Represents the time in days it takes for a person to complete a task alone. -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents the rate at which a person completes a task, as a fraction of the task per day. -/
def workRate (wt : WorkTime) : ℚ := 1 / wt.days

/-- The combined work rate of multiple people working together. -/
def combinedWorkRate (rates : List ℚ) : ℚ := rates.sum

theorem b_work_time (a_time : WorkTime) (c_time : WorkTime) (abc_time : WorkTime) 
  (ha : a_time.days = 8)
  (hc : c_time.days = 24)
  (habc : abc_time.days = 4) :
  ∃ (b_time : WorkTime), b_time.days = 12 := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l1360_136022


namespace NUMINAMATH_CALUDE_shirts_not_washed_l1360_136074

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 39 → long_sleeve = 47 → washed = 20 → 
  short_sleeve + long_sleeve - washed = 66 := by
sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l1360_136074


namespace NUMINAMATH_CALUDE_dinner_savings_l1360_136091

theorem dinner_savings (total_savings : ℝ) (individual_savings : ℝ) : 
  total_savings > 0 →
  individual_savings > 0 →
  total_savings = 2 * individual_savings →
  (3/4) * total_savings + 2 * (6 * 1.5 + 1) = total_savings →
  individual_savings = 40 := by
sorry

end NUMINAMATH_CALUDE_dinner_savings_l1360_136091


namespace NUMINAMATH_CALUDE_sum_of_roots_theorem_l1360_136026

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 4

-- State the theorem
theorem sum_of_roots_theorem (a b : ℝ) 
  (h1 : f a = 14) 
  (h2 : f b = -14) : 
  a + b = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_theorem_l1360_136026


namespace NUMINAMATH_CALUDE_carnival_activity_order_l1360_136052

/-- Represents an activity at the school carnival -/
inductive Activity
  | Dodgeball
  | MagicShow
  | PettingZoo
  | FacePainting

/-- Returns the popularity of an activity as a fraction -/
def popularity (a : Activity) : Rat :=
  match a with
  | Activity.Dodgeball => 3 / 8
  | Activity.MagicShow => 9 / 24
  | Activity.PettingZoo => 1 / 3
  | Activity.FacePainting => 5 / 12

/-- Checks if one activity is more popular than another -/
def morePopularThan (a b : Activity) : Prop :=
  popularity a > popularity b

theorem carnival_activity_order :
  morePopularThan Activity.FacePainting Activity.Dodgeball ∧
  morePopularThan Activity.Dodgeball Activity.MagicShow ∧
  morePopularThan Activity.MagicShow Activity.PettingZoo :=
by sorry

end NUMINAMATH_CALUDE_carnival_activity_order_l1360_136052


namespace NUMINAMATH_CALUDE_annie_mike_toy_ratio_l1360_136015

/-- Represents the number of toys each person has -/
structure ToyCount where
  annie : ℕ
  mike : ℕ
  tom : ℕ

/-- Given the conditions of the problem, proves that the ratio of Annie's toys to Mike's toys is 4:1 -/
theorem annie_mike_toy_ratio 
  (tc : ToyCount) 
  (mike_toys : tc.mike = 6)
  (annie_multiple : ∃ k : ℕ, tc.annie = k * tc.mike)
  (annie_less_than_tom : tc.annie = tc.tom - 2)
  (total_toys : tc.annie + tc.mike + tc.tom = 56) :
  tc.annie / tc.mike = 4 := by
  sorry

#check annie_mike_toy_ratio

end NUMINAMATH_CALUDE_annie_mike_toy_ratio_l1360_136015


namespace NUMINAMATH_CALUDE_base_b_divisibility_l1360_136082

theorem base_b_divisibility (b : ℤ) : b ∈ ({3, 4, 5, 6, 8} : Set ℤ) →
  (b * (2 * b^2 - b - 1)) % 4 ≠ 0 ↔ b = 3 ∨ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_b_divisibility_l1360_136082


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1360_136054

theorem unique_four_digit_number : ∃! x : ℕ,
  1000 ≤ x ∧ x ≤ 9999 ∧
  x % 7 = 0 ∧
  x % 29 = 0 ∧
  (19 * x) % 37 = 3 ∧
  x = 5075 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1360_136054


namespace NUMINAMATH_CALUDE_papaya_problem_l1360_136081

/-- The number of fruits that turned yellow on Friday -/
def friday_yellow : ℕ := 2

theorem papaya_problem (initial_green : ℕ) (final_green : ℕ) :
  initial_green = 14 →
  final_green = 8 →
  initial_green - final_green = friday_yellow + 2 * friday_yellow →
  friday_yellow = 2 := by
  sorry

#check papaya_problem

end NUMINAMATH_CALUDE_papaya_problem_l1360_136081


namespace NUMINAMATH_CALUDE_percentage_problem_l1360_136098

theorem percentage_problem (x : ℝ) (p : ℝ) 
  (h1 : (p / 100) * x = 400)
  (h2 : (120 / 100) * x = 2400) : 
  p = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1360_136098


namespace NUMINAMATH_CALUDE_price_reduction_achieves_profit_l1360_136096

/-- Represents the store's sales and pricing data -/
structure StoreSales where
  initial_cost : ℝ
  initial_price : ℝ
  january_sales : ℝ
  march_sales : ℝ
  sales_increase_per_yuan : ℝ
  desired_profit : ℝ

/-- Calculates the required price reduction to achieve the desired profit -/
def calculate_price_reduction (s : StoreSales) : ℝ :=
  sorry

/-- Theorem stating that the calculated price reduction achieves the desired profit -/
theorem price_reduction_achieves_profit (s : StoreSales) 
  (h1 : s.initial_cost = 25)
  (h2 : s.initial_price = 40)
  (h3 : s.january_sales = 256)
  (h4 : s.march_sales = 400)
  (h5 : s.sales_increase_per_yuan = 5)
  (h6 : s.desired_profit = 4250) :
  let y := calculate_price_reduction s
  (s.initial_price - y - s.initial_cost) * (s.march_sales + s.sales_increase_per_yuan * y) = s.desired_profit :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_profit_l1360_136096


namespace NUMINAMATH_CALUDE_quadratic_solution_l1360_136021

theorem quadratic_solution (x : ℝ) : 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) ↔ 
  2 * x^2 - 7 * x - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1360_136021


namespace NUMINAMATH_CALUDE_base6_to_base10_12345_l1360_136071

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- The list representation of 12345 in base 6 --/
def number : List Nat := [5, 4, 3, 2, 1]

theorem base6_to_base10_12345 :
  base6ToBase10 number = 1865 := by
  sorry

#eval base6ToBase10 number

end NUMINAMATH_CALUDE_base6_to_base10_12345_l1360_136071


namespace NUMINAMATH_CALUDE_mary_baking_cake_l1360_136023

/-- Given a recipe that requires a certain amount of flour and an amount already added,
    calculate the remaining amount to be added. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Prove that for a recipe requiring 7 cups of flour, with 2 cups already added,
    the remaining amount to be added is 5 cups. -/
theorem mary_baking_cake :
  remaining_flour 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_baking_cake_l1360_136023


namespace NUMINAMATH_CALUDE_triangle_vector_equality_l1360_136095

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define vectors m and n
def m (t : Triangle) : ℝ × ℝ := t.B - t.C
def n (t : Triangle) : ℝ × ℝ := t.D - t.C

-- State the theorem
theorem triangle_vector_equality (t : Triangle) 
  (h1 : t.D.1 = t.A.1 + (2/3) * (t.B.1 - t.A.1) ∧ t.D.2 = t.A.2 + (2/3) * (t.B.2 - t.A.2)) :
  t.A - t.C = -1/2 * (m t) + 3/2 * (n t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_equality_l1360_136095


namespace NUMINAMATH_CALUDE_singh_gain_l1360_136048

/-- Represents the game with three players and their monetary amounts -/
structure Game where
  initial_amount : ℚ
  ashtikar_final : ℚ
  singh_final : ℚ
  bhatia_final : ℚ

/-- Defines the conditions of the game -/
def game_conditions (g : Game) : Prop :=
  g.initial_amount = 70 ∧
  g.ashtikar_final / g.singh_final = 1 / 2 ∧
  g.singh_final / g.bhatia_final = 4 / 1 ∧
  g.ashtikar_final + g.singh_final + g.bhatia_final = 3 * g.initial_amount

/-- Theorem stating Singh's gain -/
theorem singh_gain (g : Game) (h : game_conditions g) : 
  g.singh_final - g.initial_amount = 50 := by
  sorry


end NUMINAMATH_CALUDE_singh_gain_l1360_136048


namespace NUMINAMATH_CALUDE_book_cost_l1360_136013

theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) : 
  (7 * (cost_of_three / 3) : ℝ) = 105 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l1360_136013


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1360_136037

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 3 = 10 →
  a 1 * a 3 = 16 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1360_136037


namespace NUMINAMATH_CALUDE_star_four_three_l1360_136076

def star (x y : ℝ) : ℝ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l1360_136076


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l1360_136088

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 10 = 0) (h2 : n ≥ 30) : ∃ (blue red green yellow : ℕ),
  blue = n / 2 ∧
  red = n / 5 ∧
  green = 8 ∧
  yellow = n - (blue + red + green) ∧
  yellow ≥ 1 ∧
  ∀ m : ℕ, m < n → ¬(∃ (b r g y : ℕ),
    b = m / 2 ∧
    r = m / 5 ∧
    g = 8 ∧
    y = m - (b + r + g) ∧
    y ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l1360_136088


namespace NUMINAMATH_CALUDE_monkey_climb_height_l1360_136005

/-- The height of the tree that the monkey climbs -/
def tree_height : ℕ := 22

/-- The distance the monkey climbs up each hour -/
def climb_distance : ℕ := 3

/-- The distance the monkey slips back each hour -/
def slip_distance : ℕ := 2

/-- The total time it takes for the monkey to reach the top of the tree -/
def total_time : ℕ := 20

/-- Theorem stating that the height of the tree is 22 ft -/
theorem monkey_climb_height :
  tree_height = (total_time - 1) * (climb_distance - slip_distance) + climb_distance :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_height_l1360_136005


namespace NUMINAMATH_CALUDE_pr_length_l1360_136092

-- Define the triangles and their side lengths
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

def PQR : Triangle := { side1 := 30, side2 := 18, side3 := 22.5 }
def STU : Triangle := { side1 := 24, side2 := 18, side3 := 18 }

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.side1 / t2.side1 = t1.side2 / t2.side2 ∧
  t1.side1 / t2.side1 = t1.side3 / t2.side3

-- Theorem statement
theorem pr_length :
  similar PQR STU → PQR.side3 = 22.5 :=
by
  sorry

end NUMINAMATH_CALUDE_pr_length_l1360_136092


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1360_136025

/-- Given a cubic equation with distinct real roots between 0 and 1, 
    prove that the sum of the reciprocals of one minus each root equals 2/3 -/
theorem cubic_root_sum (a b c : ℝ) : 
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) →
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) →
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  1/(1-a) + 1/(1-b) + 1/(1-c) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1360_136025


namespace NUMINAMATH_CALUDE_school_sections_l1360_136018

theorem school_sections (num_boys num_girls : ℕ) 
  (h_boys : num_boys = 408) 
  (h_girls : num_girls = 312) : 
  (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l1360_136018


namespace NUMINAMATH_CALUDE_telescope_cost_l1360_136000

theorem telescope_cost (joan karl : ℕ) 
  (h1 : joan + karl = 400)
  (h2 : 2 * joan = karl + 74) : 
  joan = 158 := by
sorry

end NUMINAMATH_CALUDE_telescope_cost_l1360_136000
