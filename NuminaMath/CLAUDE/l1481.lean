import Mathlib

namespace NUMINAMATH_CALUDE_video_cassettes_in_second_set_l1481_148163

-- Define the cost of a video cassette
def video_cassette_cost : ℕ := 300

-- Define the equations from the problem
def equation1 (audio_cost video_count : ℕ) : Prop :=
  5 * audio_cost + video_count * video_cassette_cost = 1350

def equation2 (audio_cost : ℕ) : Prop :=
  7 * audio_cost + 3 * video_cassette_cost = 1110

-- Theorem to prove
theorem video_cassettes_in_second_set :
  ∃ (audio_cost video_count : ℕ),
    equation1 audio_cost video_count ∧
    equation2 audio_cost →
    3 = 3 :=
sorry

end NUMINAMATH_CALUDE_video_cassettes_in_second_set_l1481_148163


namespace NUMINAMATH_CALUDE_chameleons_multiple_colors_l1481_148198

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The initial state of chameleons on the island -/
def initial_state : ChameleonState :=
  { red := 155, blue := 49, green := 96 }

/-- Defines the color change rule for chameleons -/
def color_change_rule (state : ChameleonState) : ChameleonState → Prop :=
  λ new_state =>
    (new_state.red + new_state.blue + new_state.green = state.red + state.blue + state.green) ∧
    (new_state.red - new_state.blue) % 3 = (state.red - state.blue) % 3 ∧
    (new_state.blue - new_state.green) % 3 = (state.blue - state.green) % 3 ∧
    (new_state.red - new_state.green) % 3 = (state.red - state.green) % 3

/-- Theorem stating that it's impossible for all chameleons to be the same color -/
theorem chameleons_multiple_colors (final_state : ChameleonState) :
  color_change_rule initial_state final_state →
  ¬(final_state.red = 0 ∧ final_state.blue = 0) ∧
  ¬(final_state.red = 0 ∧ final_state.green = 0) ∧
  ¬(final_state.blue = 0 ∧ final_state.green = 0) :=
by sorry

end NUMINAMATH_CALUDE_chameleons_multiple_colors_l1481_148198


namespace NUMINAMATH_CALUDE_sin_15_cos_15_half_l1481_148145

theorem sin_15_cos_15_half : 2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_half_l1481_148145


namespace NUMINAMATH_CALUDE_integral_sin_cos_identity_l1481_148126

theorem integral_sin_cos_identity : 
  ∫ x in (0)..(π / 2), (Real.sin (Real.sin x))^2 + (Real.cos (Real.cos x))^2 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cos_identity_l1481_148126


namespace NUMINAMATH_CALUDE_max_smaller_cuboids_l1481_148120

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the smaller cuboid -/
def smallCuboid : CuboidDimensions :=
  { length := 6, width := 4, height := 3 }

/-- The dimensions of the larger cuboid -/
def largeCuboid : CuboidDimensions :=
  { length := 18, width := 15, height := 2 }

/-- Theorem stating the maximum number of whole smaller cuboids that can be formed -/
theorem max_smaller_cuboids :
  (cuboidVolume largeCuboid) / (cuboidVolume smallCuboid) = 7 :=
sorry

end NUMINAMATH_CALUDE_max_smaller_cuboids_l1481_148120


namespace NUMINAMATH_CALUDE_percent_relation_l1481_148127

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (2/5) * (x + y)) :
  y = (1/9) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1481_148127


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1481_148121

/-- The perimeter of an equilateral triangle with side length 8 is 24. -/
theorem equilateral_triangle_perimeter (side : ℝ) (h : side = 8) : 
  3 * side = 24 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1481_148121


namespace NUMINAMATH_CALUDE_fraction_difference_l1481_148172

def fractions : List ℚ := [2/3, 3/4, 4/5, 5/7, 7/10, 11/13, 14/19]

theorem fraction_difference : 
  (List.maximum fractions).get! - (List.minimum fractions).get! = 11/13 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l1481_148172


namespace NUMINAMATH_CALUDE_min_value_theorem_l1481_148143

theorem min_value_theorem (x y : ℝ) 
  (h1 : x > 1/6) 
  (h2 : y > 0) 
  (h3 : x + y = 1/3) : 
  (∀ a b : ℝ, a > 1/6 ∧ b > 0 ∧ a + b = 1/3 → 
    1/(6*a - 1) + 6/b ≥ 1/(6*x - 1) + 6/y) ∧ 
  1/(6*x - 1) + 6/y = 49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1481_148143


namespace NUMINAMATH_CALUDE_larger_number_problem_l1481_148117

theorem larger_number_problem (x y : ℝ) 
  (h1 : 5 * y = 7 * x) 
  (h2 : y - x = 10) : 
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1481_148117


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l1481_148178

theorem salary_reduction_percentage (x : ℝ) : 
  (100 - x) * (1 + 53.84615384615385 / 100) = 100 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l1481_148178


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1481_148176

theorem trigonometric_identity : 
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / 
  Real.cos (17 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1481_148176


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1481_148169

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k) 
  (h2 : 10^2 * 2^4 = k) (h3 : y = Real.sqrt 8) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1481_148169


namespace NUMINAMATH_CALUDE_netGainDifference_l1481_148131

/-- Represents a job candidate with their associated costs and revenue --/
structure Candidate where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from a candidate --/
def netGain (c : Candidate) : ℕ :=
  c.revenue - c.salary - (c.trainingMonths * c.trainingCostPerMonth) - (c.salary * c.hiringBonusPercent / 100)

/-- The two candidates as described in the problem --/
def candidate1 : Candidate :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

def candidate2 : Candidate :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two candidates --/
theorem netGainDifference : netGain candidate1 - netGain candidate2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_netGainDifference_l1481_148131


namespace NUMINAMATH_CALUDE_prob_two_nondefective_pens_l1481_148104

/-- Given a box of 8 pens with 2 defective pens, the probability of selecting 2 non-defective pens at random is 15/28. -/
theorem prob_two_nondefective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : defective_pens = 2) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 15 / 28 := by
  sorry

#check prob_two_nondefective_pens

end NUMINAMATH_CALUDE_prob_two_nondefective_pens_l1481_148104


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_2_l1481_148112

/-- The function f(x) = -2x^2 + ax + 1 is decreasing on (1/2, +∞) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 1/2 < x ∧ x < y → (-2*x^2 + a*x + 1) > (-2*y^2 + a*y + 1)

/-- If f(x) = -2x^2 + ax + 1 is decreasing on (1/2, +∞), then a ≤ 2 -/
theorem decreasing_function_implies_a_leq_2 :
  ∀ a : ℝ, is_decreasing_on_interval a → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_2_l1481_148112


namespace NUMINAMATH_CALUDE_sand_dune_probability_l1481_148193

/-- The probability that a sand dune remains -/
def P_remain : ℚ := 1 / 3

/-- The probability that a blown-out sand dune has a treasure -/
def P_treasure : ℚ := 1 / 5

/-- The probability that a sand dune has lucky coupons -/
def P_coupons : ℚ := 2 / 3

/-- The probability that a dune is formed in the evening -/
def P_evening : ℚ := 70 / 100

/-- The probability that a dune is formed in the morning -/
def P_morning : ℚ := 1 - P_evening

/-- The combined probability that a blown-out sand dune contains both the treasure and lucky coupons -/
def P_combined : ℚ := P_treasure * P_morning * P_coupons

theorem sand_dune_probability : P_combined = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sand_dune_probability_l1481_148193


namespace NUMINAMATH_CALUDE_set_a_equals_set_b_l1481_148192

/-- A positive integer that is not a perfect square -/
structure NonSquare (a : ℕ) : Prop where
  pos : 0 < a
  not_square : ∀ n : ℕ, n^2 ≠ a

/-- The equation k = (x^2 - a) / (x^2 - y^2) has a solution in ℤ^2 -/
def HasSolution (k a : ℕ) : Prop :=
  ∃ x y : ℤ, k = (x^2 - a) / (x^2 - y^2)

/-- The set of positive integers k for which the equation has a solution with x > √a -/
def SetA (a : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ ∃ x y : ℤ, x^2 > a ∧ HasSolution k a}

/-- The set of positive integers k for which the equation has a solution with 0 ≤ x < √a -/
def SetB (a : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ ∃ x y : ℤ, 0 ≤ x^2 ∧ x^2 < a ∧ HasSolution k a}

/-- The main theorem: Set A equals Set B for any non-square positive integer a -/
theorem set_a_equals_set_b (a : ℕ) (h : NonSquare a) : SetA a = SetB a := by
  sorry

end NUMINAMATH_CALUDE_set_a_equals_set_b_l1481_148192


namespace NUMINAMATH_CALUDE_find_y_value_l1481_148161

theorem find_y_value (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^10) (h2 : x = 12) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1481_148161


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_positive_a_nonpositive_discriminant_l1481_148135

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The solution set of a quadratic inequality ax^2 + bx + c < 0 -/
def solutionSet (a b c : ℝ) : Set ℝ := {x : ℝ | a*x^2 + b*x + c < 0}

theorem empty_solution_set_implies_positive_a_nonpositive_discriminant
  (a b c : ℝ) (h_a_nonzero : a ≠ 0) :
  IsEmpty (solutionSet a b c) → a > 0 ∧ discriminant a b c ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_positive_a_nonpositive_discriminant_l1481_148135


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1481_148159

/-- Given a train of length 240 m crossing a platform of equal length in 27 s,
    its speed is approximately 64 km/h. -/
theorem train_speed_calculation (train_length platform_length : ℝ)
  (crossing_time : ℝ) (h1 : train_length = 240)
  (h2 : platform_length = train_length) (h3 : crossing_time = 27) :
  ∃ (speed : ℝ), abs (speed - 64) < 0.5 ∧ speed = (train_length + platform_length) / crossing_time * 3.6 :=
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1481_148159


namespace NUMINAMATH_CALUDE_inequality_proof_l1481_148111

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3 * x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1481_148111


namespace NUMINAMATH_CALUDE_largest_last_digit_l1481_148177

/-- A string of digits satisfying the given conditions -/
structure DigitString where
  digits : Fin 2050 → Nat
  first_digit_is_two : digits 0 = 2
  divisibility_condition : ∀ i : Fin 2049, 
    (digits i * 10 + digits (i + 1)) % 17 = 0 ∨ 
    (digits i * 10 + digits (i + 1)) % 29 = 0

/-- The theorem stating that the largest possible last digit is 8 -/
theorem largest_last_digit (s : DigitString) : 
  s.digits 2049 ≤ 8 ∧ ∃ s : DigitString, s.digits 2049 = 8 := by
  sorry


end NUMINAMATH_CALUDE_largest_last_digit_l1481_148177


namespace NUMINAMATH_CALUDE_altitude_segment_theorem_l1481_148132

-- Define the triangle and its properties
structure AcuteTriangle where
  -- We don't need to explicitly define the vertices, just the properties we need
  altitude1_segment1 : ℝ
  altitude1_segment2 : ℝ
  altitude2_segment1 : ℝ
  altitude2_segment2 : ℝ
  acute : Bool
  h_acute : acute = true
  h_altitude1 : altitude1_segment1 = 6 ∧ altitude1_segment2 = 4
  h_altitude2 : altitude2_segment1 = 3

-- State the theorem
theorem altitude_segment_theorem (t : AcuteTriangle) : t.altitude2_segment2 = 31/3 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_theorem_l1481_148132


namespace NUMINAMATH_CALUDE_triangle_perimeter_when_area_equals_four_inradius_l1481_148124

/-- Given a triangle with an inscribed circle, if the area of the triangle is numerically equal to
    four times the radius of the inscribed circle, then the perimeter of the triangle is 8. -/
theorem triangle_perimeter_when_area_equals_four_inradius (A r s p : ℝ) :
  A > 0 → r > 0 → s > 0 → p > 0 →
  A = r * s →  -- Area formula using inradius and semiperimeter
  A = 4 * r →  -- Given condition
  p = 2 * s →  -- Perimeter is twice the semiperimeter
  p = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_when_area_equals_four_inradius_l1481_148124


namespace NUMINAMATH_CALUDE_problem_statement_l1481_148147

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1481_148147


namespace NUMINAMATH_CALUDE_inequality_implication_l1481_148195

theorem inequality_implication (m n : ℝ) : -m/2 < -n/6 → 3*m > n := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1481_148195


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_difference_l1481_148106

/-- Given an ellipse with semi-major axis a and semi-minor axis b = √96,
    and a point P on the ellipse such that |PF₁| : |PF₂| : |OF₂| = 8 : 6 : 5,
    prove that |PF₁| - |PF₂| = 4 -/
theorem ellipse_focal_distance_difference 
  (a : ℝ) 
  (h_a : a > 4 * Real.sqrt 6) 
  (P : ℝ × ℝ) 
  (h_P : (P.1 / a)^2 + P.2^2 / 96 = 1) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_foci : ∃ (k : ℝ), k > 0 ∧ dist P F₁ = 8*k ∧ dist P F₂ = 6*k ∧ dist (0, 0) F₂ = 5*k) :
  dist P F₁ - dist P F₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_focal_distance_difference_l1481_148106


namespace NUMINAMATH_CALUDE_beth_book_collection_l1481_148110

theorem beth_book_collection (novels_percent : Real) (graphic_novels : Nat) (comic_books_percent : Real) :
  novels_percent = 0.65 →
  comic_books_percent = 0.2 →
  graphic_novels = 18 →
  ∃ (total_books : Nat), 
    (novels_percent + comic_books_percent + (graphic_novels : Real) / total_books) = 1 ∧
    total_books = 120 := by
  sorry

end NUMINAMATH_CALUDE_beth_book_collection_l1481_148110


namespace NUMINAMATH_CALUDE_hundredthDigitOf7Over33_l1481_148141

-- Define the fraction
def f : ℚ := 7 / 33

-- Define a function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem hundredthDigitOf7Over33 : nthDigitAfterDecimal f 100 = 1 := by sorry

end NUMINAMATH_CALUDE_hundredthDigitOf7Over33_l1481_148141


namespace NUMINAMATH_CALUDE_exam_problem_l1481_148174

/-- Proves that given the conditions of the exam problem, the number of students is 56 -/
theorem exam_problem (N : ℕ) (T : ℕ) : 
  T = 80 * N →                        -- The total marks equal 80 times the number of students
  (T - 160) / (N - 8) = 90 →          -- After excluding 8 students, the new average is 90
  N = 56 :=                           -- The number of students is 56
by
  sorry

#check exam_problem

end NUMINAMATH_CALUDE_exam_problem_l1481_148174


namespace NUMINAMATH_CALUDE_book_pages_theorem_l1481_148119

theorem book_pages_theorem (total_pages : ℚ) (read_pages : ℚ) 
  (h1 : read_pages = 3 / 7 * total_pages) : 
  ∃ (remaining_pages : ℚ),
    remaining_pages = 4 / 7 * total_pages ∧ 
    read_pages = 3 / 4 * remaining_pages := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l1481_148119


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l1481_148166

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem binary_addition_subtraction :
  let a := binary_to_decimal [true, true, false, true, true]
  let b := binary_to_decimal [true, false, true, true]
  let c := binary_to_decimal [true, true, true, false, false]
  let d := binary_to_decimal [true, false, true, false, true]
  let e := binary_to_decimal [true, false, false, true]
  let result := binary_to_decimal [true, true, true, true, false]
  a + b - c + d - e = result :=
by sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l1481_148166


namespace NUMINAMATH_CALUDE_michael_water_left_l1481_148154

/-- The amount of water Michael has left after giving some away -/
def water_left (initial : ℚ) (given_away : ℚ) : ℚ :=
  initial - given_away

/-- Theorem stating that Michael has 17/7 gallons left -/
theorem michael_water_left :
  water_left 5 (18/7) = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_michael_water_left_l1481_148154


namespace NUMINAMATH_CALUDE_total_people_needed_l1481_148146

def people_per_car : ℕ := 5

def people_per_truck (people_per_car : ℕ) : ℕ := 2 * people_per_car

def people_for_cars (num_cars : ℕ) (people_per_car : ℕ) : ℕ :=
  num_cars * people_per_car

def people_for_trucks (num_trucks : ℕ) (people_per_truck : ℕ) : ℕ :=
  num_trucks * people_per_truck

theorem total_people_needed (num_cars num_trucks : ℕ) :
  people_for_cars num_cars people_per_car +
  people_for_trucks num_trucks (people_per_truck people_per_car) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_total_people_needed_l1481_148146


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l1481_148140

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬(is_pythagorean_triple 3 4 5) ∧
  ¬(is_pythagorean_triple 3 4 7) ∧
  ¬(is_pythagorean_triple 0 1 1) ∧
  is_pythagorean_triple 9 12 15 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l1481_148140


namespace NUMINAMATH_CALUDE_triangle_internal_region_l1481_148183

-- Define the three lines that form the triangle
def line1 (x y : ℝ) : Prop := x + 2*y = 2
def line2 (x y : ℝ) : Prop := 2*x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the internal region of the triangle
def internal_region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2*y < 2 ∧ 2*x + y > 2

-- Theorem statement
theorem triangle_internal_region :
  ∀ x y : ℝ, 
    (∃ ε > 0, line1 (x + ε) y ∨ line2 (x + ε) y ∨ line3 (x + ε) y) →
    (∃ ε > 0, line1 (x - ε) y ∨ line2 (x - ε) y ∨ line3 (x - ε) y) →
    (∃ ε > 0, line1 x (y + ε) ∨ line2 x (y + ε) ∨ line3 x (y + ε)) →
    (∃ ε > 0, line1 x (y - ε) ∨ line2 x (y - ε) ∨ line3 x (y - ε)) →
    internal_region x y :=
sorry

end NUMINAMATH_CALUDE_triangle_internal_region_l1481_148183


namespace NUMINAMATH_CALUDE_ten_digit_divisible_by_11_exists_l1481_148114

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000 ∧ n < 10000000000) ∧
  (∀ d : Fin 10, ∃! p : Fin 10, (n / (10 ^ p.val) % 10) = d) ∧
  n % 11 = 0

theorem ten_digit_divisible_by_11_exists : ∃ n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_ten_digit_divisible_by_11_exists_l1481_148114


namespace NUMINAMATH_CALUDE_four_numbers_in_interval_l1481_148115

theorem four_numbers_in_interval (a b c d : Real) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < π / 2 →
  ∃ x y, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧
         (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
         x ≠ y ∧
         |x - y| < π / 6 :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_in_interval_l1481_148115


namespace NUMINAMATH_CALUDE_circle_radius_three_inches_l1481_148100

theorem circle_radius_three_inches (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2)) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_three_inches_l1481_148100


namespace NUMINAMATH_CALUDE_calculate_expression_l1481_148187

theorem calculate_expression : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1481_148187


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l1481_148116

theorem ordered_pairs_count : 
  ∃! (pairs : List (ℤ × ℕ)), 
    (∀ (x : ℤ) (y : ℕ), (x, y) ∈ pairs ↔ 
      (∃ (m : ℕ), y = m^2 ∧ y = (x - 90)^2 - 4907)) ∧ 
    pairs.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l1481_148116


namespace NUMINAMATH_CALUDE_simplify_expression_l1481_148194

theorem simplify_expression (a b c : ℝ) :
  (18 * a + 72 * b + 30 * c) + (15 * a + 40 * b - 20 * c) - (12 * a + 60 * b + 25 * c) = 21 * a + 52 * b - 15 * c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1481_148194


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l1481_148148

/-- Profit function given selling price -/
def profit (x : ℝ) : ℝ := (x - 40) * (1000 - 10 * x)

/-- The selling price that maximizes profit -/
def optimal_price : ℝ := 70

theorem profit_maximized_at_optimal_price :
  ∀ x : ℝ, profit x ≤ profit optimal_price :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l1481_148148


namespace NUMINAMATH_CALUDE_negative_reciprocal_positive_l1481_148164

theorem negative_reciprocal_positive (x : ℝ) (h : x < 0) : -x⁻¹ > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_positive_l1481_148164


namespace NUMINAMATH_CALUDE_rectangle_area_l1481_148179

-- Define the radius of the inscribed circle
def circle_radius : ℝ := 7

-- Define the ratio of length to width
def length_width_ratio : ℝ := 2

-- Theorem statement
theorem rectangle_area (width : ℝ) (length : ℝ) 
  (h1 : width = 2 * circle_radius) 
  (h2 : length = length_width_ratio * width) : 
  width * length = 392 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l1481_148179


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1481_148153

theorem cone_lateral_surface_angle (r h : ℝ) (h_positive : r > 0 ∧ h > 0) :
  (π * r * (r + (r^2 + h^2).sqrt) = 3 * π * r^2) →
  (2 * π * r / (r^2 + h^2).sqrt : ℝ) = π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1481_148153


namespace NUMINAMATH_CALUDE_marble_distribution_l1481_148197

theorem marble_distribution (x : ℚ) 
  (h1 : (5 * x + 2) + 2 * x + 4 * x = 88) : x = 86 / 11 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1481_148197


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1481_148173

theorem modulus_of_complex_number (x : ℝ) (i : ℂ) : 
  i * i = -1 →
  (∃ (y : ℝ), (x + i) * (2 + i) = y * i) →
  Complex.abs (2 * x - i) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1481_148173


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l1481_148151

theorem unique_prime_in_range : ∃! n : ℕ, 
  70 ≤ n ∧ n ≤ 90 ∧ 
  Nat.gcd n 15 = 5 ∧ 
  Nat.Prime n ∧
  n = 85 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l1481_148151


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l1481_148160

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l1481_148160


namespace NUMINAMATH_CALUDE_waiter_tip_earnings_l1481_148105

theorem waiter_tip_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : 
  total_customers = 10 →
  non_tipping_customers = 5 →
  tip_amount = 3 →
  (total_customers - non_tipping_customers) * tip_amount = 15 := by
sorry

end NUMINAMATH_CALUDE_waiter_tip_earnings_l1481_148105


namespace NUMINAMATH_CALUDE_simplify_expression_l1481_148167

theorem simplify_expression (m n : ℝ) : (8*m - 7*n) - 2*(m - 3*n) = 6*m - n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1481_148167


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l1481_148155

/-- The system of linear equations has infinitely many solutions when m = ±1 -/
theorem infinitely_many_solutions (m : ℝ) : 
  (∃ (x y : ℝ), m * x + y = m^2 ∧ x + m * y = m) ∧ 
  (∀ (a b : ℝ), ∃ (x y : ℝ), a * x + b * y = m^2 ∧ x + m * y = m) ↔ 
  m = 1 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l1481_148155


namespace NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_for_x_gt_2_l1481_148199

theorem x_gt_1_necessary_not_sufficient_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_for_x_gt_2_l1481_148199


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1481_148190

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y, m * x + (m + 2) * y - 1 = 0 ∧ (m - 1) * x + m * y = 0 → 
    (m * (m - 1) + (m + 2) * m = 0 ∨ m = 0)) → 
  (m = 0 ∨ m = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1481_148190


namespace NUMINAMATH_CALUDE_harry_bid_difference_l1481_148149

/-- Represents the auction process and calculates the difference between Harry's final bid and the third bidder's bid. -/
def auctionBidDifference (startingBid : ℕ) (harryFirstIncrement : ℕ) (harryFinalBid : ℕ) : ℕ :=
  let harryFirstBid := startingBid + harryFirstIncrement
  let secondBid := harryFirstBid * 2
  let thirdBid := secondBid + harryFirstIncrement * 3
  harryFinalBid - thirdBid

/-- Theorem stating that given the specific auction conditions, Harry's final bid exceeds the third bidder's bid by $2400. -/
theorem harry_bid_difference :
  auctionBidDifference 300 200 4000 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_harry_bid_difference_l1481_148149


namespace NUMINAMATH_CALUDE_inverse_of_A_l1481_148165

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; 4, -2]

theorem inverse_of_A : 
  (A⁻¹) = !![(-1 : ℝ), (3/2 : ℝ); (-2 : ℝ), (5/2 : ℝ)] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1481_148165


namespace NUMINAMATH_CALUDE_function_always_negative_m_range_l1481_148162

theorem function_always_negative_m_range
  (f : ℝ → ℝ)
  (m : ℝ)
  (h1 : ∀ x, f x = m * x^2 - m * x - 1)
  (h2 : ∀ x, f x < 0) :
  -4 < m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_function_always_negative_m_range_l1481_148162


namespace NUMINAMATH_CALUDE_marble_arrangement_l1481_148181

def arrange_marbles (n : ℕ) (restricted_pairs : ℕ) : ℕ :=
  n.factorial - restricted_pairs * (n - 1).factorial

theorem marble_arrangement :
  arrange_marbles 5 1 = 72 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_l1481_148181


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1481_148136

theorem fraction_equation_solution (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ y ≠ x) :
  1/x - 1/y = 1/z → z = x*y/(y-x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1481_148136


namespace NUMINAMATH_CALUDE_hammer_order_sequence_l1481_148156

theorem hammer_order_sequence (sequence : ℕ → ℕ) : 
  sequence 1 = 3 →  -- June (1st month)
  sequence 3 = 6 →  -- August (3rd month)
  sequence 4 = 9 →  -- September (4th month)
  sequence 5 = 13 → -- October (5th month)
  sequence 2 = 6    -- July (2nd month)
:= by sorry

end NUMINAMATH_CALUDE_hammer_order_sequence_l1481_148156


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l1481_148171

/-- Given a function f(x) = (1/3)x^3 + x^2 - ax + 3a that is monotonically increasing
    in the interval [1, 2], prove that a ≤ 3 -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => (1/3) * x^3 + x^2 - a*x + 3*a)) →
  a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l1481_148171


namespace NUMINAMATH_CALUDE_circle_area_theorem_l1481_148133

theorem circle_area_theorem (r : ℝ) (A : ℝ) (h : r > 0) :
  8 * (1 / A) = r^2 → A = 2 * Real.sqrt (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l1481_148133


namespace NUMINAMATH_CALUDE_p_false_and_q_true_l1481_148191

-- Define proposition p
def p : Prop := ∀ x > 0, 3^x > 1

-- Define proposition q
def q : Prop := ∀ a, (a < -2 → ∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
                    (∃ b, b ≥ -2 ∧ ∃ x ∈ Set.Icc (-1) 2, b * x + 3 = 0)

-- Theorem stating that p is false and q is true
theorem p_false_and_q_true : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_p_false_and_q_true_l1481_148191


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_l1481_148188

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  thousands_nonzero : thousands > 0
  all_digits : thousands < 10 ∧ hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- Checks if a four-digit number satisfies the given conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  n.thousands = 2 ∧
  n.hundreds % 2 = 0 ∧
  n.units = n.thousands + n.hundreds + n.tens

theorem count_satisfying_numbers :
  (∃ (s : Finset FourDigitNumber),
    (∀ n ∈ s, satisfiesConditions n) ∧
    s.card = 16 ∧
    (∀ n : FourDigitNumber, satisfiesConditions n → n ∈ s)) := by
  sorry

#check count_satisfying_numbers

end NUMINAMATH_CALUDE_count_satisfying_numbers_l1481_148188


namespace NUMINAMATH_CALUDE_quadratic_function_constraint_l1481_148170

theorem quadratic_function_constraint (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, a' * x^2 + b * x + c ≤ 1) →
  (7 * b + 5 * c ≤ -6 ∧ ∃ b' c', 7 * b' + 5 * c' = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_constraint_l1481_148170


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1481_148175

theorem sin_150_degrees : Real.sin (150 * Real.pi / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1481_148175


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1481_148125

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {2,5,8}
def B : Set ℕ := {1,3,5,7}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {1,3,7} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1481_148125


namespace NUMINAMATH_CALUDE_eight_ninths_position_l1481_148138

/-- Represents a fraction as a pair of natural numbers -/
def Fraction := ℕ × ℕ

/-- The sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → Fraction := sorry

/-- The sum of numerator and denominator of a fraction -/
def sum_of_parts (f : Fraction) : ℕ := f.1 + f.2

/-- The position of a fraction in the sequence -/
def position_in_sequence (f : Fraction) : ℕ := sorry

/-- The main theorem: 8/9 is at position 128 in the sequence -/
theorem eight_ninths_position :
  position_in_sequence (8, 9) = 128 := by sorry

end NUMINAMATH_CALUDE_eight_ninths_position_l1481_148138


namespace NUMINAMATH_CALUDE_slope_product_l1481_148196

/-- Given two lines L₁ and L₂ with equations y = mx and y = nx respectively,
    where L₁ makes three times as large an angle with the horizontal as L₂,
    L₁ has 5 times the slope of L₂, and L₁ is not horizontal,
    prove that mn = 5/7. -/
theorem slope_product (m n : ℝ) : 
  m ≠ 0 →  -- L₁ is not horizontal
  (∃ θ₁ θ₂ : ℝ, 
    θ₁ = 3 * θ₂ ∧  -- L₁ makes three times as large an angle with the horizontal as L₂
    m = Real.tan θ₁ ∧ 
    n = Real.tan θ₂ ∧
    m = 5 * n) →  -- L₁ has 5 times the slope of L₂
  m * n = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_l1481_148196


namespace NUMINAMATH_CALUDE_pizza_solution_l1481_148113

/-- Represents the number of pizza slices with different topping combinations -/
structure PizzaToppings where
  total : ℕ
  ham : ℕ
  pineapple : ℕ
  jalapeno : ℕ
  all_three : ℕ
  ham_only : ℕ
  pineapple_only : ℕ
  jalapeno_only : ℕ
  ham_pineapple : ℕ
  ham_jalapeno : ℕ
  pineapple_jalapeno : ℕ

/-- The pizza topping problem -/
def pizza_problem (p : PizzaToppings) : Prop :=
  p.total = 24 ∧
  p.ham = 15 ∧
  p.pineapple = 10 ∧
  p.jalapeno = 14 ∧
  p.all_three = p.jalapeno_only ∧
  p.total = p.ham_only + p.pineapple_only + p.jalapeno_only + 
            p.ham_pineapple + p.ham_jalapeno + p.pineapple_jalapeno + p.all_three ∧
  p.ham = p.ham_only + p.ham_pineapple + p.ham_jalapeno + p.all_three ∧
  p.pineapple = p.pineapple_only + p.ham_pineapple + p.pineapple_jalapeno + p.all_three ∧
  p.jalapeno = p.jalapeno_only + p.ham_jalapeno + p.pineapple_jalapeno + p.all_three

theorem pizza_solution (p : PizzaToppings) (h : pizza_problem p) : p.all_three = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_solution_l1481_148113


namespace NUMINAMATH_CALUDE_square_root_problem_l1481_148150

theorem square_root_problem (a b c : ℝ) : 
  (a - 4)^(1/3) = 1 →
  (3 * a - b - 2)^(1/2) = 3 →
  c = ⌊Real.sqrt 13⌋ →
  (2 * a - 3 * b + c)^(1/2) = 1 ∨ (2 * a - 3 * b + c)^(1/2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1481_148150


namespace NUMINAMATH_CALUDE_archibald_apple_eating_l1481_148158

theorem archibald_apple_eating (apples_per_day_first_two_weeks : ℕ) 
  (apples_per_day_last_two_weeks : ℕ) (total_weeks : ℕ) (average_apples_per_week : ℕ) :
  apples_per_day_first_two_weeks = 1 →
  apples_per_day_last_two_weeks = 3 →
  total_weeks = 7 →
  average_apples_per_week = 10 →
  ∃ (weeks_same_as_first_two : ℕ),
    weeks_same_as_first_two = 2 ∧
    (2 * 7 * apples_per_day_first_two_weeks) + 
    (weeks_same_as_first_two * 7 * apples_per_day_first_two_weeks) + 
    (2 * 7 * apples_per_day_last_two_weeks) = 
    total_weeks * average_apples_per_week :=
by sorry

end NUMINAMATH_CALUDE_archibald_apple_eating_l1481_148158


namespace NUMINAMATH_CALUDE_infinitely_many_n_squared_plus_one_divides_factorial_l1481_148128

/-- The set of positive integers n for which n^2 + 1 divides n! is infinite -/
theorem infinitely_many_n_squared_plus_one_divides_factorial :
  Set.Infinite {n : ℕ+ | (n^2 + 1) ∣ n!} := by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_squared_plus_one_divides_factorial_l1481_148128


namespace NUMINAMATH_CALUDE_frog_hop_probability_l1481_148118

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Defines whether a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop in one of the four cardinal directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, wrapping around if necessary -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x - 1, p.y⟩
  | Direction.Down => ⟨p.x + 1, p.y⟩
  | Direction.Left => ⟨p.x, p.y - 1⟩
  | Direction.Right => ⟨p.x, p.y + 1⟩

/-- The probability of ending on an edge after three hops -/
def probEndOnEdge (start : Position) : ℚ :=
  sorry

theorem frog_hop_probability :
  probEndOnEdge ⟨1, 1⟩ = 37 / 64 := by sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l1481_148118


namespace NUMINAMATH_CALUDE_yoongis_class_size_l1481_148123

theorem yoongis_class_size :
  ∀ (students_a students_b students_both : ℕ),
    students_a = 18 →
    students_b = 24 →
    students_both = 7 →
    students_a + students_b - students_both = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_yoongis_class_size_l1481_148123


namespace NUMINAMATH_CALUDE_product_of_odds_over_sum_of_squares_l1481_148186

theorem product_of_odds_over_sum_of_squares : 
  (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_odds_over_sum_of_squares_l1481_148186


namespace NUMINAMATH_CALUDE_probability_at_least_four_same_l1481_148134

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a specific number on a fair die -/
def prob_single : ℚ := 1 / num_sides

/-- The probability that at least four out of five fair six-sided dice show the same value -/
def prob_at_least_four_same : ℚ := 13 / 648

/-- Theorem stating that the probability of at least four out of five fair six-sided dice 
    showing the same value is 13/648 -/
theorem probability_at_least_four_same : 
  prob_at_least_four_same = (1 / num_sides^4) + (5 * (1 / num_sides^3) * (5 / 6)) :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_four_same_l1481_148134


namespace NUMINAMATH_CALUDE_auto_shop_discount_l1481_148130

theorem auto_shop_discount (part_cost : ℕ) (num_parts : ℕ) (total_discount : ℕ) : 
  part_cost = 80 → num_parts = 7 → total_discount = 121 → 
  part_cost * num_parts - total_discount = 439 := by
  sorry

end NUMINAMATH_CALUDE_auto_shop_discount_l1481_148130


namespace NUMINAMATH_CALUDE_function_properties_l1481_148182

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_symmetry : ∀ x, f (4 - x) = f x) : 
  (∀ x, f (x + 8) = f x) ∧ (f 2019 + f 2020 + f 2021 = 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1481_148182


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l1481_148101

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.sin A = 3 / 5) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l1481_148101


namespace NUMINAMATH_CALUDE_candy_store_sampling_theorem_l1481_148185

/-- The percentage of customers who sample candy but are not caught -/
def uncaught_samplers (total_samplers caught_samplers : ℝ) : ℝ :=
  total_samplers - caught_samplers

theorem candy_store_sampling_theorem 
  (total_samplers : ℝ) 
  (caught_samplers : ℝ) 
  (h1 : caught_samplers = 22)
  (h2 : total_samplers = 23.913043478260867) :
  uncaught_samplers total_samplers caught_samplers = 1.913043478260867 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_sampling_theorem_l1481_148185


namespace NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l1481_148189

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def condition (t : Triangle) : Prop :=
  (t.a * Real.cos t.B + t.b * Real.cos t.A) / t.c = 2 * Real.cos t.C

theorem angle_C_measure (t : Triangle) (h : condition t) : t.C = π / 3 := by
  sorry

theorem side_c_length (t : Triangle) 
  (h1 : (1 / 2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3)
  (h2 : t.a + t.b = 6)
  (h3 : t.C = π / 3) : t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l1481_148189


namespace NUMINAMATH_CALUDE_least_sum_m_n_l1481_148109

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m + n) 330 = 1) ∧ 
  (∃ (k : ℕ), m^(m : ℕ) = k * n^(n : ℕ)) ∧ 
  (¬∃ (l : ℕ), m = l * n) ∧
  (∀ (p q : ℕ+), 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ (k : ℕ), p^(p : ℕ) = k * q^(q : ℕ)) → 
    (¬∃ (l : ℕ), p = l * q) → 
    (m + n ≤ p + q)) ∧
  (m + n = 182) := by
sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l1481_148109


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_is_lower_bound_l1481_148157

theorem min_value_of_function (x : ℝ) (h : x > 1) : 
  2 * x + 2 / (x - 1) ≥ 6 := by
  sorry

theorem min_value_is_lower_bound (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, x > 1 ∧ 2 * x + 2 / (x - 1) < 6 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_is_lower_bound_l1481_148157


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1481_148168

def f (x : ℝ) : ℝ := -3 * (x - 2)^2 + 12

theorem quadratic_function_proof :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 4) ∧
  (∀ x ∈ Set.Icc (-1) 5, f x ≤ 12) ∧
  (∃ x ∈ Set.Icc (-1) 5, f x = 12) →
  ∀ x, f x = -3 * (x - 2)^2 + 12 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1481_148168


namespace NUMINAMATH_CALUDE_no_linear_term_iff_m_eq_six_l1481_148142

def expansion (x m : ℝ) : ℝ := 2 * x^2 + (m - 6) * x - 3 * m

theorem no_linear_term_iff_m_eq_six :
  ∀ m : ℝ, (∀ x : ℝ, expansion x m = 2 * x^2 - 3 * m) ↔ m = 6 := by sorry

end NUMINAMATH_CALUDE_no_linear_term_iff_m_eq_six_l1481_148142


namespace NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l1481_148144

theorem consecutive_integers_fourth_power_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 12 * (3 * x + 3) - 24 →
  x^4 + (x + 1)^4 + (x + 2)^4 = 98 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l1481_148144


namespace NUMINAMATH_CALUDE_min_value_expression_l1481_148184

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a^2 + b^2 + 4/a^2 + b/a ≥ m) ∧
             (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ c^2 + d^2 + 4/c^2 + d/c = m) ∧
             m = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1481_148184


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1481_148103

/-- Given a rectangle with width 10 m and area 150 square meters, 
    if its length is increased such that the new area is 1 (1/3) times the original area, 
    then the new perimeter of the rectangle is 60 meters. -/
theorem rectangle_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) : 
  width = 10 →
  original_area = 150 →
  new_area = original_area * (4/3) →
  let original_length := original_area / width
  let new_length := new_area / width
  2 * (new_length + width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1481_148103


namespace NUMINAMATH_CALUDE_points_form_circle_l1481_148180

theorem points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) →
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_points_form_circle_l1481_148180


namespace NUMINAMATH_CALUDE_angle_triple_complement_l1481_148122

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l1481_148122


namespace NUMINAMATH_CALUDE_ratio_calculation_l1481_148139

theorem ratio_calculation (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) 
  (hw : w ≠ 0) : 
  x * z / (y * w) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1481_148139


namespace NUMINAMATH_CALUDE_ten_person_meeting_handshakes_l1481_148107

/-- The number of handshakes in a meeting of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that in a meeting of 10 people, where each person
    shakes hands exactly once with every other person, the total number
    of handshakes is 45. -/
theorem ten_person_meeting_handshakes :
  handshakes 10 = 45 := by sorry

end NUMINAMATH_CALUDE_ten_person_meeting_handshakes_l1481_148107


namespace NUMINAMATH_CALUDE_rotate_point_A_about_C_l1481_148137

/-- Rotates a point 180 degrees about a center point -/
def rotate180 (point center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

theorem rotate_point_A_about_C : 
  let A : ℝ × ℝ := (-4, 1)
  let C : ℝ × ℝ := (-1, 1)
  rotate180 A C = (2, 1) := by sorry

end NUMINAMATH_CALUDE_rotate_point_A_about_C_l1481_148137


namespace NUMINAMATH_CALUDE_camera_price_difference_l1481_148129

/-- The list price of Camera Y in dollars -/
def list_price : ℚ := 52.99

/-- The discount amount at Best Deals in dollars -/
def best_deals_discount : ℚ := 12

/-- The discount percentage at Market Value -/
def market_value_discount_percent : ℚ := 20

/-- The sale price at Best Deals in dollars -/
def best_deals_price : ℚ := list_price - best_deals_discount

/-- The sale price at Market Value in dollars -/
def market_value_price : ℚ := list_price * (1 - market_value_discount_percent / 100)

/-- The price difference between Market Value and Best Deals in cents -/
def price_difference_cents : ℤ := 
  ⌊(market_value_price - best_deals_price) * 100⌋

theorem camera_price_difference : price_difference_cents = 140 := by
  sorry

end NUMINAMATH_CALUDE_camera_price_difference_l1481_148129


namespace NUMINAMATH_CALUDE_solve_jump_rope_problem_l1481_148152

def jump_rope_problem (cindy_time betsy_time tina_time : ℝ) : Prop :=
  cindy_time = 12 ∧
  tina_time = 3 * betsy_time ∧
  tina_time = cindy_time + 6 ∧
  betsy_time / cindy_time = 1 / 2

theorem solve_jump_rope_problem :
  ∃ (betsy_time tina_time : ℝ),
    jump_rope_problem 12 betsy_time tina_time :=
by
  sorry

end NUMINAMATH_CALUDE_solve_jump_rope_problem_l1481_148152


namespace NUMINAMATH_CALUDE_find_unknown_number_l1481_148108

theorem find_unknown_number (y : ℝ) : 
  (17.28 / 12) / (y * 0.2) = 2 → y = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l1481_148108


namespace NUMINAMATH_CALUDE_marble_weight_problem_l1481_148102

theorem marble_weight_problem (piece1 piece2 total : ℝ) 
  (h1 : piece1 = 0.3333333333333333)
  (h2 : piece2 = 0.3333333333333333)
  (h3 : total = 0.75) :
  total - (piece1 + piece2) = 0.08333333333333337 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_problem_l1481_148102
