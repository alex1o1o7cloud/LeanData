import Mathlib

namespace NUMINAMATH_CALUDE_drama_club_revenue_l3893_389342

theorem drama_club_revenue : 
  let total_tickets : ℕ := 1500
  let adult_price : ℕ := 12
  let student_price : ℕ := 6
  let student_tickets : ℕ := 300
  let adult_tickets : ℕ := total_tickets - student_tickets
  let total_revenue : ℕ := adult_tickets * adult_price + student_tickets * student_price
  total_revenue = 16200 := by
sorry

end NUMINAMATH_CALUDE_drama_club_revenue_l3893_389342


namespace NUMINAMATH_CALUDE_circle_condition_l3893_389339

theorem circle_condition (m : ℝ) :
  (∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔
  (m < 1/4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l3893_389339


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3893_389376

theorem triangle_perimeter (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 15) (h_side_ratio : a = b / 2) : 
  a + b + c = 15 + 9 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3893_389376


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l3893_389373

/-- The center of a circle satisfying given conditions -/
theorem circle_center_coordinates : ∃ (x y : ℝ),
  (x - 2 * y = 0) ∧
  (3 * x + 4 * y = 10) ∧
  (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l3893_389373


namespace NUMINAMATH_CALUDE_ball_ratio_proof_l3893_389312

/-- Proves that the ratio of blue balls to red balls is 16:5 given the initial conditions --/
theorem ball_ratio_proof (initial_red : ℕ) (lost_red : ℕ) (yellow : ℕ) (total : ℕ) :
  initial_red = 16 →
  lost_red = 6 →
  yellow = 32 →
  total = 74 →
  ∃ (blue : ℕ), blue * 5 = (initial_red - lost_red) * 16 ∧ 
                blue + (initial_red - lost_red) + yellow = total :=
by sorry

end NUMINAMATH_CALUDE_ball_ratio_proof_l3893_389312


namespace NUMINAMATH_CALUDE_parallelogram_area_and_perimeter_l3893_389387

/-- Represents a parallelogram EFGH -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  side : ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Calculate the perimeter of a parallelogram with all sides equal -/
def perimeter (p : Parallelogram) : ℝ := 4 * p.side

/-- Theorem about the area and perimeter of a specific parallelogram -/
theorem parallelogram_area_and_perimeter :
  ∀ (p : Parallelogram),
  p.base = 6 → p.height = 3 → p.side = 5 →
  area p = 18 ∧ perimeter p = 20 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_and_perimeter_l3893_389387


namespace NUMINAMATH_CALUDE_contractor_absent_days_l3893_389316

/-- Proves the number of absent days for a contractor under specific conditions -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_received : ℚ) : 
  total_days = 30 → 
  daily_pay = 25 → 
  daily_fine = 7.5 → 
  total_received = 620 → 
  ∃ (days_worked : ℕ) (days_absent : ℕ), 
    days_worked + days_absent = total_days ∧ 
    (daily_pay * days_worked : ℚ) - (daily_fine * days_absent : ℚ) = total_received ∧ 
    days_absent = 8 := by
  sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l3893_389316


namespace NUMINAMATH_CALUDE_ratio_of_300_to_2_l3893_389326

theorem ratio_of_300_to_2 : 
  let certain_number := 300
  300 / 2 = 150 := by sorry

end NUMINAMATH_CALUDE_ratio_of_300_to_2_l3893_389326


namespace NUMINAMATH_CALUDE_square_cut_impossible_l3893_389382

/-- Proves that a square with perimeter 40 cannot be cut into two identical rectangles with perimeter 20 each -/
theorem square_cut_impossible (square_perimeter : ℝ) (rect_perimeter : ℝ) : 
  square_perimeter = 40 → rect_perimeter = 20 → 
  ¬ ∃ (square_side rect_length rect_width : ℝ),
    (square_side * 4 = square_perimeter) ∧ 
    (rect_length + rect_width = square_side) ∧
    (2 * (rect_length + rect_width) = rect_perimeter) :=
by
  sorry

#check square_cut_impossible

end NUMINAMATH_CALUDE_square_cut_impossible_l3893_389382


namespace NUMINAMATH_CALUDE_probability_same_gender_is_four_ninths_l3893_389303

/-- Represents a school with a specific number of male and female teachers -/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- Calculates the total number of teachers in a school -/
def School.total_teachers (s : School) : ℕ := s.male_teachers + s.female_teachers

/-- Calculates the number of ways to select two teachers of the same gender -/
def same_gender_selections (s1 s2 : School) : ℕ :=
  s1.male_teachers * s2.male_teachers + s1.female_teachers * s2.female_teachers

/-- Calculates the total number of ways to select one teacher from each school -/
def total_selections (s1 s2 : School) : ℕ :=
  s1.total_teachers * s2.total_teachers

/-- The probability of selecting two teachers of the same gender -/
def probability_same_gender (s1 s2 : School) : ℚ :=
  (same_gender_selections s1 s2 : ℚ) / (total_selections s1 s2 : ℚ)

theorem probability_same_gender_is_four_ninths :
  let school_A : School := { male_teachers := 2, female_teachers := 1 }
  let school_B : School := { male_teachers := 1, female_teachers := 2 }
  probability_same_gender school_A school_B = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_gender_is_four_ninths_l3893_389303


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3893_389374

/-- An arithmetic sequence {a_n} with a_1 + a_9 = 10 and a_2 = -1 has a common difference of 2. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 + a 9 = 10 →                     -- given condition
  a 2 = -1 →                           -- given condition
  a 2 - a 1 = 2 :=                     -- conclusion: common difference is 2
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3893_389374


namespace NUMINAMATH_CALUDE_oil_temperature_increase_rate_l3893_389365

def oil_temperature (t : ℕ) : ℝ :=
  if t = 0 then 10
  else if t = 10 then 30
  else if t = 20 then 50
  else if t = 30 then 70
  else if t = 40 then 90
  else 0  -- undefined for other values

theorem oil_temperature_increase_rate :
  ∀ t : ℕ, t < 40 →
    oil_temperature (t + 10) - oil_temperature t = 20 :=
sorry

end NUMINAMATH_CALUDE_oil_temperature_increase_rate_l3893_389365


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l3893_389328

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l3893_389328


namespace NUMINAMATH_CALUDE_shane_current_age_l3893_389377

/-- Given that twenty years ago Shane was 2 times older than Garret is now,
    and Garret is currently 12 years old, prove that Shane is 44 years old now. -/
theorem shane_current_age :
  (∀ (shane_age_now garret_age_now : ℕ),
    garret_age_now = 12 →
    shane_age_now - 20 = 2 * garret_age_now →
    shane_age_now = 44) :=
by sorry

end NUMINAMATH_CALUDE_shane_current_age_l3893_389377


namespace NUMINAMATH_CALUDE_warehouse_loading_theorem_l3893_389315

/-- Represents the warehouse loading problem -/
def warehouseLoading (crateCapacity : ℕ) (numCrates : ℕ) 
                     (nailBags : ℕ) (nailWeight : ℕ) 
                     (hammerBags : ℕ) (hammerWeight : ℕ) 
                     (plankBags : ℕ) (plankWeight : ℕ) : Prop :=
  let totalWeight := nailBags * nailWeight + hammerBags * hammerWeight + plankBags * plankWeight
  let totalCapacity := crateCapacity * numCrates
  totalWeight - totalCapacity = 80

/-- Theorem stating the weight to be left out in the warehouse loading problem -/
theorem warehouse_loading_theorem : 
  warehouseLoading 20 15 4 5 12 5 10 30 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_loading_theorem_l3893_389315


namespace NUMINAMATH_CALUDE_largest_n_for_square_sum_equality_l3893_389352

theorem largest_n_for_square_sum_equality : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), i ≤ n → j ≤ n → (x + i)^2 + (y i)^2 = (x + j)^2 + (y j)^2) ∧
  (∀ (m : ℕ), m > n → ¬∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), i ≤ m → j ≤ m → (x + i)^2 + (y i)^2 = (x + j)^2 + (y j)^2) ∧
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_square_sum_equality_l3893_389352


namespace NUMINAMATH_CALUDE_longest_leg_of_smallest_triangle_l3893_389319

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse_def : hypotenuse = 2 * shorter_leg
  longer_leg_def : longer_leg = shorter_leg * Real.sqrt 3

/-- Represents a sequence of three 30-60-90 triangles -/
structure TriangleSequence where
  largest : Triangle30_60_90
  middle : Triangle30_60_90
  smallest : Triangle30_60_90
  sequence_property : 
    largest.longer_leg = middle.hypotenuse ∧
    middle.longer_leg = smallest.hypotenuse

theorem longest_leg_of_smallest_triangle 
  (seq : TriangleSequence) 
  (h : seq.largest.hypotenuse = 16) : 
  seq.smallest.longer_leg = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_leg_of_smallest_triangle_l3893_389319


namespace NUMINAMATH_CALUDE_line_translation_theorem_l3893_389301

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - l.slope * dx + dy }

theorem line_translation_theorem :
  let original_line : Line := { slope := 2, intercept := -3 }
  let translated_line := translate original_line 2 3
  translated_line = { slope := 2, intercept := -4 } := by sorry

end NUMINAMATH_CALUDE_line_translation_theorem_l3893_389301


namespace NUMINAMATH_CALUDE_angle_complement_l3893_389394

/-- 
Given an angle x and its complement y, prove that y equals 90° minus x.
-/
theorem angle_complement (x y : ℝ) (h : x + y = 90) : y = 90 - x := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_l3893_389394


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3893_389357

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if a = 1, C = 60°, and c = √3, then A = π/6 -/
theorem triangle_angle_calculation (a b c A B C : Real) : 
  a = 1 → C = π / 3 → c = Real.sqrt 3 → A = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3893_389357


namespace NUMINAMATH_CALUDE_book_organization_time_l3893_389369

theorem book_organization_time (time_A time_B joint_time : ℝ) 
  (h1 : time_A = 6)
  (h2 : time_B = 8)
  (h3 : joint_time = 2)
  (h4 : joint_time * (1 / time_A + 1 / time_B) + 1 / time_A * remaining_time = 1) :
  remaining_time = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_book_organization_time_l3893_389369


namespace NUMINAMATH_CALUDE_inequality_proof_l3893_389341

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3893_389341


namespace NUMINAMATH_CALUDE_same_color_left_neighbor_l3893_389385

/-- The number of children in the circle. -/
def total_children : ℕ := 150

/-- The number of children in blue jackets who have a left neighbor in a red jacket. -/
def blue_with_red_left : ℕ := 12

/-- Theorem stating the number of children with a left neighbor wearing a jacket of the same color. -/
theorem same_color_left_neighbor :
  total_children - 2 * blue_with_red_left = 126 := by
  sorry

end NUMINAMATH_CALUDE_same_color_left_neighbor_l3893_389385


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3893_389378

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x - 2*y)^5 = a*(x + 2*y)^5 + a₁*(x + 2*y)^4*y + a₂*(x + 2*y)^3*y^2 + 
                            a₃*(x + 2*y)^2*y^3 + a₄*(x + 2*y)*y^4 + a₅*y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3893_389378


namespace NUMINAMATH_CALUDE_f_properties_l3893_389397

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3) + 1

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 5 * Real.pi / 12 → f x₁ < f x₂) ∧
  (∀ x₁ x₂ x₃, x₁ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               x₂ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               x₃ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               f x₁ + f x₃ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3893_389397


namespace NUMINAMATH_CALUDE_exists_number_with_property_l3893_389390

-- Define small numbers
def isSmall (n : ℕ) : Prop := n ≤ 150

-- Define the property we're looking for
def hasProperty (N : ℕ) : Prop :=
  ∃ (a b : ℕ), isSmall a ∧ isSmall b ∧ b = a + 1 ∧
  ¬(N % a = 0) ∧ ¬(N % b = 0) ∧
  ∀ k, isSmall k → k ≠ a → k ≠ b → N % k = 0

-- Theorem statement
theorem exists_number_with_property :
  ∃ N : ℕ, hasProperty N :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_property_l3893_389390


namespace NUMINAMATH_CALUDE_prob_diff_games_l3893_389308

/-- The probability of getting heads on a single coin flip -/
def p_heads : ℚ := 3/5

/-- The probability of getting tails on a single coin flip -/
def p_tails : ℚ := 2/5

/-- The probability of winning Game A -/
def p_win_game_a : ℚ := p_heads^4 + p_tails^4

/-- The probability of winning Game B -/
def p_win_game_b : ℚ := (p_heads^2 + p_tails^2) * (p_heads^3 + p_tails^3)

theorem prob_diff_games : p_win_game_a - p_win_game_b = 6/625 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_games_l3893_389308


namespace NUMINAMATH_CALUDE_remainder_division_l3893_389354

theorem remainder_division (y k : ℤ) (h : y = 264 * k + 42) : y ≡ 20 [ZMOD 22] := by
  sorry

end NUMINAMATH_CALUDE_remainder_division_l3893_389354


namespace NUMINAMATH_CALUDE_range_of_a_l3893_389302

/-- The function g(x) = ax + 2 -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, g a x₁ = f x₀) →
  a ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3893_389302


namespace NUMINAMATH_CALUDE_original_number_l3893_389372

theorem original_number (w : ℝ) : (1.125 * w) - (0.75 * w) = 30 → w = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3893_389372


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_from_one_vertex_l3893_389309

/-- A dodecagon is a polygon with 12 sides. -/
def Dodecagon : Nat := 12

/-- The number of diagonals that can be drawn from one vertex of a polygon with n sides. -/
def diagonalsFromOneVertex (n : Nat) : Nat := n - 3

theorem dodecagon_diagonals_from_one_vertex :
  diagonalsFromOneVertex Dodecagon = 9 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_from_one_vertex_l3893_389309


namespace NUMINAMATH_CALUDE_quadratic_propositions_l3893_389317

/-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions -/
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 4*m*x + 1 = 0

/-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀² - 2x₀ - 1 > 0 -/
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, m*x₀^2 - 2*x₀ - 1 > 0

theorem quadratic_propositions (m : ℝ) :
  (p m ↔ (m ≥ 1/2 ∨ m ≤ -1/2)) ∧
  (q m ↔ m > -1) ∧
  ((p m ↔ ¬q m) → (-1 < m ∧ m < 1/2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_propositions_l3893_389317


namespace NUMINAMATH_CALUDE_six_people_arrangement_l3893_389388

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of positions where person A can stand (not at ends) -/
def validPositionsForA (n : ℕ) : ℕ := n - 2

/-- The number of ways to arrange the remaining people after placing A -/
def remainingArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The total number of valid arrangements -/
def validArrangements (n : ℕ) : ℕ :=
  (validPositionsForA n) * (remainingArrangements n)

theorem six_people_arrangement :
  validArrangements 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l3893_389388


namespace NUMINAMATH_CALUDE_initial_bananas_count_l3893_389320

/-- The number of bananas in each package -/
def package_size : ℕ := 13

/-- The number of bananas added to the pile -/
def bananas_added : ℕ := 7

/-- The total number of bananas after adding -/
def total_bananas : ℕ := 9

/-- The initial number of bananas on the desk -/
def initial_bananas : ℕ := total_bananas - bananas_added

theorem initial_bananas_count : initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l3893_389320


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean60_l3893_389335

-- Define the set of two-digit positive integers
def TwoDigitPositiveInt : Set ℕ := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the mean of x and y
def meanIs60 (x y : ℕ) : Prop := (x + y) / 2 = 60

-- Theorem statement
theorem max_ratio_two_digit_mean60 :
  ∃ (x y : ℕ), x ∈ TwoDigitPositiveInt ∧ y ∈ TwoDigitPositiveInt ∧ meanIs60 x y ∧
  ∀ (a b : ℕ), a ∈ TwoDigitPositiveInt → b ∈ TwoDigitPositiveInt → meanIs60 a b →
  (a : ℚ) / b ≤ 33 / 7 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean60_l3893_389335


namespace NUMINAMATH_CALUDE_pythagorean_triple_sequence_l3893_389351

theorem pythagorean_triple_sequence (k : ℕ+) :
  ∃ (c : ℕ), (k * (2 * k - 2))^2 + (2 * k - 1)^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_sequence_l3893_389351


namespace NUMINAMATH_CALUDE_tan_squared_f_equals_neg_cos_double_l3893_389348

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)))

-- State the theorem
theorem tan_squared_f_equals_neg_cos_double (t : ℝ) 
  (h1 : 0 ≤ t) (h2 : t ≤ π/2) : f (Real.tan t ^ 2) = -Real.cos (2 * t) :=
by
  sorry


end NUMINAMATH_CALUDE_tan_squared_f_equals_neg_cos_double_l3893_389348


namespace NUMINAMATH_CALUDE_f_inequality_l3893_389384

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem f_inequality (a : ℝ) (h : a ≤ -2) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → |f a x₁ - f a x₂| ≥ 4 * |x₁ - x₂| := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3893_389384


namespace NUMINAMATH_CALUDE_division_problem_l3893_389310

theorem division_problem (n : ℤ) : 
  (n / 20 = 15) ∧ (n % 20 = 6) → n = 306 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3893_389310


namespace NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l3893_389324

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = (-a/b)x - (c/b) -/
theorem slope_intercept_form {a b c : ℝ} (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a/b) * x - (c/b) :=
sorry

theorem parallel_lines_a_value :
  (∀ x y : ℝ, a * x + 4 * y + 1 = 0 ↔ 2 * x + y - 2 = 0) → a = 8 :=
sorry

end NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l3893_389324


namespace NUMINAMATH_CALUDE_tangent_line_property_l3893_389336

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def hasTangentAt (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

def tangentLineEquation (f : ℝ → ℝ) : Prop :=
  ∃ (y : ℝ), 2 + 2 * y + 1 = 0 ∧ f 2 = y

-- State the theorem
theorem tangent_line_property (f : ℝ → ℝ) 
  (h1 : hasTangentAt f) 
  (h2 : tangentLineEquation f) : 
  f 2 - 2 * (deriv f 2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_property_l3893_389336


namespace NUMINAMATH_CALUDE_circle_and_symmetry_line_l3893_389349

-- Define the center of the circle
def C : ℝ × ℝ := (-1, 0)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 3 = 0

-- Define the symmetry line
def symmetry_line (m x y : ℝ) : Prop := m * x + y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Theorem statement
theorem circle_and_symmetry_line :
  ∃ (r : ℝ), r > 0 ∧
  (∀ x y : ℝ, (x - C.1)^2 + (y - C.2)^2 = r^2 →
    (∃ x' y' : ℝ, tangent_line x' y' ∧ (x' - C.1)^2 + (y' - C.2)^2 = r^2)) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    ∃ m : ℝ, symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂ ∧ x₁ ≠ x₂) →
  (∀ x y : ℝ, circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = 4) ∧
  (∃! m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂ ∧ x₁ ≠ x₂ ∧ m = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_and_symmetry_line_l3893_389349


namespace NUMINAMATH_CALUDE_sin_graph_shift_l3893_389380

theorem sin_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x - π / 10)) = 3 * Real.sin (2 * x - π / 5) := by sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l3893_389380


namespace NUMINAMATH_CALUDE_sufficient_condition_inequality_l3893_389360

theorem sufficient_condition_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_inequality_l3893_389360


namespace NUMINAMATH_CALUDE_coefficient_a6_l3893_389356

theorem coefficient_a6 (x a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  x^2 + x^7 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 →
  a₆ = -7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a6_l3893_389356


namespace NUMINAMATH_CALUDE_saved_amount_l3893_389362

theorem saved_amount (x : ℕ) : (3 * x - 42)^2 = 2241 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_saved_amount_l3893_389362


namespace NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l3893_389311

def selling_price : ℝ := 250
def cost_price : ℝ := 208.33

theorem profit_percentage_is_20_percent :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l3893_389311


namespace NUMINAMATH_CALUDE_only_statement_4_implies_negation_l3893_389350

theorem only_statement_4_implies_negation (p q : Prop) :
  -- Define the four statements
  let s1 := p ∨ q
  let s2 := p ∧ ¬q
  let s3 := ¬p ∨ q
  let s4 := ¬p ∧ ¬q
  -- Define the negation of "p or q is true"
  let neg_p_or_q := ¬(p ∨ q)
  -- The theorem: only s4 implies neg_p_or_q
  (s1 → neg_p_or_q) = False ∧
  (s2 → neg_p_or_q) = False ∧
  (s3 → neg_p_or_q) = False ∧
  (s4 → neg_p_or_q) = True :=
by
  sorry

#check only_statement_4_implies_negation

end NUMINAMATH_CALUDE_only_statement_4_implies_negation_l3893_389350


namespace NUMINAMATH_CALUDE_max_value_inequality_l3893_389399

theorem max_value_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_condition : a + b + c + d ≤ 4) :
  (2*a^2 + a^2*b)^(1/4) + (2*b^2 + b^2*c)^(1/4) + 
  (2*c^2 + c^2*d)^(1/4) + (2*d^2 + d^2*a)^(1/4) ≤ 4 * 3^(1/4) :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3893_389399


namespace NUMINAMATH_CALUDE_prime_odd_sum_l3893_389392

theorem prime_odd_sum (a b : ℕ) : 
  Prime a → Odd b → a^2 + b = 2001 → a + b = 1999 := by sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l3893_389392


namespace NUMINAMATH_CALUDE_parallel_lines_coefficient_l3893_389383

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (3*a - 1)*x - a*y - 1 = 0) → 
  (a = 0 ∨ a = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_coefficient_l3893_389383


namespace NUMINAMATH_CALUDE_mary_chewing_gums_l3893_389343

theorem mary_chewing_gums (total sam sue : ℕ) (h1 : total = 30) (h2 : sam = 10) (h3 : sue = 15) :
  total - (sam + sue) = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_chewing_gums_l3893_389343


namespace NUMINAMATH_CALUDE_chocolate_box_count_l3893_389364

def chocolate_problem (total_bars : ℕ) : Prop :=
  let bar_cost : ℕ := 3
  let unsold_bars : ℕ := 4
  let revenue : ℕ := 9
  (total_bars - unsold_bars) * bar_cost = revenue

theorem chocolate_box_count : ∃ (n : ℕ), chocolate_problem n ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_count_l3893_389364


namespace NUMINAMATH_CALUDE_abs_z_equals_5_sqrt_2_l3893_389367

theorem abs_z_equals_5_sqrt_2 (z : ℂ) (h : z^2 = -48 + 14*I) : Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_5_sqrt_2_l3893_389367


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3893_389358

/-- A 30-60-90 triangle with shortest side length 2 -/
structure Triangle30_60_90 where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  is_30_60_90 : True  -- Placeholder for the triangle's angle properties
  de_length : dist D E = 2

/-- A circle tangent to coordinate axes and parts of the triangle -/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  triangle : Triangle30_60_90
  tangent_to_axes : True  -- Placeholder for tangency to coordinate axes
  tangent_to_leg : True   -- Placeholder for tangency to one leg of the triangle
  tangent_to_hypotenuse : True  -- Placeholder for tangency to hypotenuse

/-- The main theorem stating the radius of the tangent circle -/
theorem tangent_circle_radius (c : TangentCircle) : c.r = (5 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3893_389358


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_line_through_two_points_l3893_389375

-- Define a line with coefficients A, B, and C
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Theorem 1: Line does not pass through third quadrant
theorem line_not_in_third_quadrant (l : Line) 
  (h1 : l.A * l.B < 0) (h2 : l.B * l.C < 0) : 
  ∀ (p : Point), p.x < 0 ∧ p.y < 0 → l.A * p.x - l.B * p.y - l.C ≠ 0 :=
sorry

-- Theorem 2: Line equation through two distinct points
theorem line_through_two_points (p1 p2 : Point) (h : p1 ≠ p2) :
  ∀ (p : Point), (p2.x - p1.x) * (p.y - p1.y) = (p2.y - p1.y) * (p.x - p1.x) ↔
  ∃ (t : ℝ), p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y) :=
sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_line_through_two_points_l3893_389375


namespace NUMINAMATH_CALUDE_rosa_flowers_total_l3893_389322

theorem rosa_flowers_total (initial_flowers : Float) (additional_flowers : Float) :
  initial_flowers = 67.0 →
  additional_flowers = 90.0 →
  initial_flowers + additional_flowers = 157.0 := by
sorry

end NUMINAMATH_CALUDE_rosa_flowers_total_l3893_389322


namespace NUMINAMATH_CALUDE_june_population_calculation_l3893_389345

/-- Represents the fish population model in the reservoir --/
structure FishPopulation where
  june_population : ℕ
  tagged_fish : ℕ
  october_sample : ℕ
  tagged_in_sample : ℕ

/-- Calculates the number of fish in the reservoir on June 1 --/
def calculate_june_population (model : FishPopulation) : ℕ :=
  let remaining_tagged := model.tagged_fish * 7 / 10  -- 70% of tagged fish remain
  let october_old_fish := model.october_sample / 2    -- 50% of October fish are old
  (remaining_tagged * october_old_fish) / model.tagged_in_sample

/-- Theorem stating the correct number of fish in June based on the given model --/
theorem june_population_calculation (model : FishPopulation) :
  model.tagged_fish = 100 →
  model.october_sample = 90 →
  model.tagged_in_sample = 4 →
  calculate_june_population model = 1125 :=
by
  sorry

#eval calculate_june_population ⟨1125, 100, 90, 4⟩

end NUMINAMATH_CALUDE_june_population_calculation_l3893_389345


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3893_389391

theorem imaginary_part_of_reciprocal (i : ℂ) (h : i^2 = -1) :
  Complex.im (1 / (i - 2)) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3893_389391


namespace NUMINAMATH_CALUDE_divisors_of_500_l3893_389366

theorem divisors_of_500 : Finset.card (Nat.divisors 500) = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_500_l3893_389366


namespace NUMINAMATH_CALUDE_range_of_a_l3893_389368

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- Define the property of f not being monotonic
def not_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ (f x < f y ∧ f y > f z ∨ f x > f y ∧ f y < f z)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  not_monotonic (f a) ↔ a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3893_389368


namespace NUMINAMATH_CALUDE_chosen_numbers_sum_l3893_389395

theorem chosen_numbers_sum (S : Finset ℕ) : 
  S.card = 5 ∧ 
  S ⊆ Finset.range 9 ∧ 
  S.sum id = ((Finset.range 9).sum id - S.sum id) / 2 → 
  S.sum id = 15 := by sorry

end NUMINAMATH_CALUDE_chosen_numbers_sum_l3893_389395


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3893_389340

theorem purely_imaginary_complex_number (x : ℝ) : 
  (Complex.I * (x + 1) : ℂ).re = 0 ∧ (Complex.I * (x + 1) : ℂ).im ≠ 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3893_389340


namespace NUMINAMATH_CALUDE_sweet_potato_price_is_correct_l3893_389314

/-- The price of each sweet potato in Alice's grocery order --/
def sweet_potato_price : ℚ :=
  let minimum_spend : ℚ := 35
  let chicken_price : ℚ := 6 * (3/2)
  let lettuce_price : ℚ := 3
  let tomato_price : ℚ := 5/2
  let broccoli_price : ℚ := 2 * 2
  let sprouts_price : ℚ := 5/2
  let sweet_potato_count : ℕ := 4
  let additional_spend : ℚ := 11
  let total_without_potatoes : ℚ := chicken_price + lettuce_price + tomato_price + broccoli_price + sprouts_price
  let potato_total : ℚ := minimum_spend - additional_spend - total_without_potatoes
  potato_total / sweet_potato_count

theorem sweet_potato_price_is_correct : sweet_potato_price = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potato_price_is_correct_l3893_389314


namespace NUMINAMATH_CALUDE_inequality_proof_l3893_389325

theorem inequality_proof (x y : ℝ) : 
  5 * x^2 + y^2 + 1 ≥ 4 * x * y + 2 * x ∧ 
  (5 * x^2 + y^2 + 1 = 4 * x * y + 2 * x ↔ x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3893_389325


namespace NUMINAMATH_CALUDE_cube_difference_l3893_389330

theorem cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3893_389330


namespace NUMINAMATH_CALUDE_one_fourth_greater_than_one_fifth_of_successor_l3893_389337

theorem one_fourth_greater_than_one_fifth_of_successor :
  let N : ℝ := 24.000000000000004
  (1/4 : ℝ) * N - (1/5 : ℝ) * (N + 1) = 1.000000000000000 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_greater_than_one_fifth_of_successor_l3893_389337


namespace NUMINAMATH_CALUDE_a_2017_equals_2_l3893_389321

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 2 * n - 1

-- Define the sequence a_n
def a (n : ℕ) : ℕ := S n - S (n - 1)

-- Theorem statement
theorem a_2017_equals_2 : a 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_2017_equals_2_l3893_389321


namespace NUMINAMATH_CALUDE_unique_intersection_implies_a_value_l3893_389371

/-- Given a line y = 2a and a function y = |x-a| - 1 in the Cartesian coordinate system,
    if they have only one intersection point, then a = -1/2 --/
theorem unique_intersection_implies_a_value (a : ℝ) :
  (∃! p : ℝ × ℝ, p.2 = 2*a ∧ p.2 = |p.1 - a| - 1) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_a_value_l3893_389371


namespace NUMINAMATH_CALUDE_new_bucket_capacity_l3893_389304

/-- Represents the capacity of a water tank in liters. -/
def TankCapacity : ℝ := 22 * 13.5

/-- Proves that given a tank that can be filled by either 22 buckets of 13.5 liters each
    or 33 buckets of equal capacity, the capacity of each of the 33 buckets is 9 liters. -/
theorem new_bucket_capacity : 
  ∀ (new_capacity : ℝ), 
  (33 * new_capacity = TankCapacity) → 
  new_capacity = 9 := by
sorry

end NUMINAMATH_CALUDE_new_bucket_capacity_l3893_389304


namespace NUMINAMATH_CALUDE_complex_power_eight_l3893_389393

theorem complex_power_eight (a b : ℝ) (h : (a : ℂ) + Complex.I = 1 - b * Complex.I) : 
  (a + b * Complex.I : ℂ) ^ 8 = 16 := by sorry

end NUMINAMATH_CALUDE_complex_power_eight_l3893_389393


namespace NUMINAMATH_CALUDE_smallest_sticker_count_l3893_389353

theorem smallest_sticker_count (N : ℕ) : 
  N > 1 → 
  (∃ x y z : ℕ, N = 3 * x + 1 ∧ N = 5 * y + 1 ∧ N = 11 * z + 1) → 
  N ≥ 166 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sticker_count_l3893_389353


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3893_389344

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 50.4 → x + (1/4 * x) = 90 - (30/100 * 90) := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3893_389344


namespace NUMINAMATH_CALUDE_min_draw_theorem_l3893_389398

/-- Represents the colors of the balls in the bag -/
inductive BallColor
  | Red
  | White
  | Yellow

/-- Represents the bag of balls -/
structure BallBag where
  red : Nat
  white : Nat
  yellow : Nat

/-- The minimum number of balls to draw to guarantee two different colors -/
def minDrawDifferentColors (bag : BallBag) : Nat :=
  bag.red + 1

/-- The minimum number of balls to draw to guarantee two yellow balls -/
def minDrawTwoYellow (bag : BallBag) : Nat :=
  bag.red + bag.white + 2

/-- Theorem stating the minimum number of balls to draw for different scenarios -/
theorem min_draw_theorem (bag : BallBag) 
  (h_red : bag.red = 10) 
  (h_white : bag.white = 10) 
  (h_yellow : bag.yellow = 10) : 
  minDrawDifferentColors bag = 11 ∧ minDrawTwoYellow bag = 22 := by
  sorry

#check min_draw_theorem

end NUMINAMATH_CALUDE_min_draw_theorem_l3893_389398


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3893_389323

theorem rectangular_plot_breadth : 
  ∀ (length breadth area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 867 →
  breadth = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3893_389323


namespace NUMINAMATH_CALUDE_unique_two_digit_sum_diff_product_l3893_389361

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_sum_diff_product (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    n = 10 * x + y ∧
    1 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    n = (x + y) * (y - x)

theorem unique_two_digit_sum_diff_product :
  ∃! n : ℕ, is_two_digit n ∧ digits_sum_diff_product n ∧ n = 48 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_sum_diff_product_l3893_389361


namespace NUMINAMATH_CALUDE_problem1_problem2_l3893_389333

-- Problem 1
def M : ℝ × ℝ := (3, 2)
def N : ℝ × ℝ := (4, -1)

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = 0

def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem problem1 (P : ℝ × ℝ) :
  on_x_axis P ∧ is_right_angle M P N → P = (2, 0) ∨ P = (5, 0) :=
sorry

-- Problem 2
def A : ℝ × ℝ := (7, -4)
def B : ℝ × ℝ := (-5, 6)

def perpendicular_bisector (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  6 * x - 5 * y - 1 = 0

theorem problem2 :
  perpendicular_bisector A B = λ x y => 6 * x - 5 * y - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_problem1_problem2_l3893_389333


namespace NUMINAMATH_CALUDE_max_surface_area_inscribed_sphere_l3893_389346

/-- The maximum surface area of an inscribed sphere in a right triangular prism --/
theorem max_surface_area_inscribed_sphere (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 25) :
  ∃ (r : ℝ), r > 0 ∧ 
    r = (5/2) * (Real.sqrt 2 - 1) ∧
    4 * π * r^2 = 25 * (3 - 3 * Real.sqrt 2) * π ∧
    ∀ (r' : ℝ), r' > 0 → r' * (a + b + 5) ≤ a * b → 4 * π * r'^2 ≤ 25 * (3 - 3 * Real.sqrt 2) * π :=
by sorry

end NUMINAMATH_CALUDE_max_surface_area_inscribed_sphere_l3893_389346


namespace NUMINAMATH_CALUDE_work_done_equals_21_l3893_389389

def force : ℝ × ℝ := (5, 2)
def point_A : ℝ × ℝ := (-1, 3)
def point_B : ℝ × ℝ := (2, 6)

theorem work_done_equals_21 : 
  let displacement := (point_B.1 - point_A.1, point_B.2 - point_A.2)
  (force.1 * displacement.1 + force.2 * displacement.2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_work_done_equals_21_l3893_389389


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3893_389355

/-- An arithmetic sequence with first three terms a-1, a+1, and 2a+3 has the general formula 2n - 3 for its nth term. -/
theorem arithmetic_sequence_formula (a : ℝ) : 
  (∃ (seq : ℕ → ℝ), 
    seq 1 = a - 1 ∧ 
    seq 2 = a + 1 ∧ 
    seq 3 = 2*a + 3 ∧ 
    ∀ n m : ℕ, seq (n + 1) - seq n = seq (m + 1) - seq m) → 
  (∃ (seq : ℕ → ℝ), 
    seq 1 = a - 1 ∧ 
    seq 2 = a + 1 ∧ 
    seq 3 = 2*a + 3 ∧ 
    ∀ n : ℕ, seq n = 2*n - 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3893_389355


namespace NUMINAMATH_CALUDE_no_solution_implies_a_range_l3893_389338

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ∈ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_range_l3893_389338


namespace NUMINAMATH_CALUDE_largest_common_divisor_660_483_l3893_389370

theorem largest_common_divisor_660_483 : Nat.gcd 660 483 = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_660_483_l3893_389370


namespace NUMINAMATH_CALUDE_tenth_term_is_37_l3893_389307

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 3 + a 5 = 26) ∧
  (a 1 + a 2 + a 3 + a 4 = 28)

/-- The 10th term of the arithmetic sequence is 37 -/
theorem tenth_term_is_37 (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 10 = 37 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_37_l3893_389307


namespace NUMINAMATH_CALUDE_smallest_distance_between_circle_and_ellipse_l3893_389332

theorem smallest_distance_between_circle_and_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let ellipse := {p : ℝ × ℝ | ((p.1 - 2)^2 / 9) + (p.2^2 / 16) = 1}
  ∃ (d : ℝ), d = (Real.sqrt 35 - 2) / 2 ∧
    (∀ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ circle → p₂ ∈ ellipse →
      Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) ≥ d) ∧
    (∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ circle ∧ p₂ ∈ ellipse ∧
      Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = d) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circle_and_ellipse_l3893_389332


namespace NUMINAMATH_CALUDE_max_value_fraction_l3893_389306

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (2*a + b)^2 = 1 + 6*a*b) :
  (a * b) / (2*a + b + 1) ≤ 1/6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (2*a₀ + b₀)^2 = 1 + 6*a₀*b₀ ∧ (a₀ * b₀) / (2*a₀ + b₀ + 1) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3893_389306


namespace NUMINAMATH_CALUDE_probability_of_a_l3893_389396

theorem probability_of_a (p_a p_b : ℝ) (h_pb : p_b = 2/5)
  (h_independent : p_a * p_b = 0.22857142857142856) :
  p_a = 0.5714285714285714 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_a_l3893_389396


namespace NUMINAMATH_CALUDE_february_highest_percentage_l3893_389359

-- Define the months
inductive Month
| January
| February
| March
| April
| May

-- Define the sales data for each month
def sales_data (m : Month) : (Nat × Nat × Nat) :=
  match m with
  | Month.January => (5, 4, 6)
  | Month.February => (6, 5, 7)
  | Month.March => (5, 5, 8)
  | Month.April => (4, 6, 7)
  | Month.May => (3, 4, 5)

-- Calculate the percentage difference
def percentage_difference (m : Month) : Rat :=
  let (d, b, f) := sales_data m
  let c := d + b
  (c - f : Rat) / f * 100

-- Theorem statement
theorem february_highest_percentage :
  ∀ m : Month, m ≠ Month.February →
  percentage_difference Month.February ≥ percentage_difference m :=
by sorry

end NUMINAMATH_CALUDE_february_highest_percentage_l3893_389359


namespace NUMINAMATH_CALUDE_shared_circles_existence_l3893_389381

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to check if a point is on a circle
def isPointOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a function to check if a circle is the circumcircle of a triangle
def isCircumcircle (c : Circle) (t : Triangle) : Prop :=
  isPointOnCircle t.A c ∧ isPointOnCircle t.B c ∧ isPointOnCircle t.C c

-- Define a function to check if a circle is the inscribed circle of a triangle
def isInscribedCircle (c : Circle) (t : Triangle) : Prop :=
  -- This is a simplified condition; in reality, it would involve more complex geometric relationships
  true

-- The main theorem
theorem shared_circles_existence 
  (ABC : Triangle) 
  (O : Circle) 
  (I : Circle) 
  (h1 : isCircumcircle O ABC) 
  (h2 : isInscribedCircle I ABC) 
  (D : ℝ × ℝ) 
  (h3 : isPointOnCircle D O) : 
  ∃ (DEF : Triangle), isCircumcircle O DEF ∧ isInscribedCircle I DEF :=
sorry

end NUMINAMATH_CALUDE_shared_circles_existence_l3893_389381


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3893_389318

/-- Calculates the total number of handshakes in a basketball game scenario -/
theorem basketball_handshakes (team_size : ℕ) (referee_count : ℕ) : team_size = 7 ∧ referee_count = 3 →
  team_size * team_size + 2 * team_size * referee_count = 91 := by
  sorry

#check basketball_handshakes

end NUMINAMATH_CALUDE_basketball_handshakes_l3893_389318


namespace NUMINAMATH_CALUDE_hamlet_47_impossible_hamlet_41_possible_hamlet_59_possible_hamlet_61_possible_hamlet_66_possible_l3893_389329

/-- Represents the total number of animals and people in Hamlet -/
def hamlet_total (h c : ℕ) : ℕ := 13 * h + 5 * c

/-- Theorem stating that 47 cannot be expressed as a hamlet total -/
theorem hamlet_47_impossible : ¬ ∃ (h c : ℕ), hamlet_total h c = 47 := by sorry

/-- Theorem stating that 41 can be expressed as a hamlet total -/
theorem hamlet_41_possible : ∃ (h c : ℕ), hamlet_total h c = 41 := by sorry

/-- Theorem stating that 59 can be expressed as a hamlet total -/
theorem hamlet_59_possible : ∃ (h c : ℕ), hamlet_total h c = 59 := by sorry

/-- Theorem stating that 61 can be expressed as a hamlet total -/
theorem hamlet_61_possible : ∃ (h c : ℕ), hamlet_total h c = 61 := by sorry

/-- Theorem stating that 66 can be expressed as a hamlet total -/
theorem hamlet_66_possible : ∃ (h c : ℕ), hamlet_total h c = 66 := by sorry

end NUMINAMATH_CALUDE_hamlet_47_impossible_hamlet_41_possible_hamlet_59_possible_hamlet_61_possible_hamlet_66_possible_l3893_389329


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l3893_389327

theorem coffee_decaf_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (additional_purchase : ℝ) 
  (final_decaf_percent : ℝ) :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  additional_purchase = 100 →
  final_decaf_percent = 26 →
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := initial_stock * (initial_decaf_percent / 100)
  let final_decaf := total_stock * (final_decaf_percent / 100)
  let additional_decaf := final_decaf - initial_decaf
  (additional_decaf / additional_purchase) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l3893_389327


namespace NUMINAMATH_CALUDE_subtract_fractions_l3893_389313

theorem subtract_fractions : (7 : ℚ) / 9 - (5 : ℚ) / 6 = (-1 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l3893_389313


namespace NUMINAMATH_CALUDE_inequality_proof_l3893_389331

theorem inequality_proof (k m n : ℕ) (hk : k > 0) (hm : m > 0) (hn : n > 0) 
  (hkm : k ≠ m) (hkn : k ≠ n) (hmn : m ≠ n) : 
  (k - 1 / k) * (m - 1 / m) * (n - 1 / n) ≤ k * m * n - (k + m + n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3893_389331


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3893_389300

/-- An arithmetic sequence with given first term, second term, and last term -/
structure ArithmeticSequence where
  first_term : ℕ
  second_term : ℕ
  last_term : ℕ

/-- The number of terms in an arithmetic sequence -/
def num_terms (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℕ :=
  seq.second_term - seq.first_term

theorem arithmetic_sequence_length :
  let seq := ArithmeticSequence.mk 13 19 127
  num_terms seq = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3893_389300


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3893_389363

theorem arithmetic_sequence_ratio (a d : ℝ) : 
  (a + d) + (a + 3*d) = 6*a ∧ 
  a + 2*d = 10 →
  a / (a + 3*d) = 1/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3893_389363


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3893_389386

/-- Given an equilateral triangle with side length a inscribed in a circle,
    the area of a square inscribed in the same circle is 2a^2/3 -/
theorem inscribed_square_area (a : ℝ) (ha : a > 0) :
  ∃ (R : ℝ), R > 0 ∧
  ∃ (s : ℝ), s > 0 ∧
  (a = R * Real.sqrt 3) ∧ 
  (s = R * Real.sqrt 2) ∧
  (s^2 = 2 * a^2 / 3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3893_389386


namespace NUMINAMATH_CALUDE_boat_against_stream_distance_l3893_389347

/-- Proves the distance traveled against the stream given boat speed and along-stream distance --/
theorem boat_against_stream_distance
  (boat_speed : ℝ)
  (along_stream_distance : ℝ)
  (h1 : boat_speed = 8)
  (h2 : along_stream_distance = 11)
  : (2 * boat_speed - along_stream_distance) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_against_stream_distance_l3893_389347


namespace NUMINAMATH_CALUDE_expansion_coefficient_equality_l3893_389379

theorem expansion_coefficient_equality (n : ℕ+) : 
  (8 * (Nat.choose n 3)) = (8 * 2 * (Nat.choose n 1)) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_equality_l3893_389379


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3893_389334

theorem other_root_of_quadratic (m : ℝ) : 
  (1 : ℝ) ^ 2 - 3 * (1 : ℝ) + m = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 - 3 * x + m = 0 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3893_389334


namespace NUMINAMATH_CALUDE_inequality_theorem_l3893_389305

theorem inequality_theorem (p q r : ℝ) 
  (h_order : p < q)
  (h_inequality : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) :
  p + 2*q + 3*r = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3893_389305
