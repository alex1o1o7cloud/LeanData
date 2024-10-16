import Mathlib

namespace NUMINAMATH_CALUDE_irrational_power_congruence_l1099_109976

theorem irrational_power_congruence :
  ∀ (k : ℕ), k ≥ 2 →
  ∃ (r : ℝ), Irrational r ∧
    ∀ (m : ℕ), (⌊r^m⌋ : ℤ) ≡ -1 [ZMOD k] :=
sorry

end NUMINAMATH_CALUDE_irrational_power_congruence_l1099_109976


namespace NUMINAMATH_CALUDE_range_of_a_l1099_109942

-- Define the conditions
def p (x a : ℝ) : Prop := |x - a| < 4
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 > 0

-- Define the theorem
theorem range_of_a :
  ∃ (a_min a_max : ℝ),
    (a_min = -1 ∧ a_max = 6) ∧
    (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ a_min ≤ a ∧ a ≤ a_max) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1099_109942


namespace NUMINAMATH_CALUDE_right_triangle_perpendicular_bisector_l1099_109937

theorem right_triangle_perpendicular_bisector 
  (A B C D : ℝ × ℝ) 
  (right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 75)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 100)
  (D_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2))
  (AD_perp_BC : (D.1 - A.1) * (C.1 - B.1) + (D.2 - A.2) * (C.2 - B.2) = 0) :
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 45 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perpendicular_bisector_l1099_109937


namespace NUMINAMATH_CALUDE_square_side_length_l1099_109965

theorem square_side_length (m n : ℝ) :
  let area := 9*m^2 + 24*m*n + 16*n^2
  ∃ (side : ℝ), side ≥ 0 ∧ side^2 = area ∧ side = |3*m + 4*n| :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l1099_109965


namespace NUMINAMATH_CALUDE_existence_of_uncuttable_rectangle_l1099_109952

/-- A rectangle with natural number side lengths -/
structure Rectangle where
  length : ℕ+
  width : ℕ+

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- A predicate that checks if two numbers are almost equal -/
def almost_equal (a b : ℕ) : Prop := a = b ∨ a = b + 1 ∨ a = b - 1

/-- A predicate that checks if a rectangle can be cut out from another rectangle -/
def can_cut_out (small big : Rectangle) : Prop :=
  small.length ≤ big.length ∧ small.width ≤ big.width ∨
  small.length ≤ big.width ∧ small.width ≤ big.length

theorem existence_of_uncuttable_rectangle :
  ∃ (r : Rectangle), ¬∃ (s : Rectangle), 
    can_cut_out s r ∧ almost_equal (area s) ((area r) / 2) :=
sorry

end NUMINAMATH_CALUDE_existence_of_uncuttable_rectangle_l1099_109952


namespace NUMINAMATH_CALUDE_fred_final_balloons_l1099_109989

def fred_balloons : ℕ → Prop
| n => ∃ (initial given received distributed : ℕ),
  initial = 1457 ∧
  given = 341 ∧
  received = 225 ∧
  distributed = ((initial - given + received) / 2) ∧
  n = initial - given + received - distributed

theorem fred_final_balloons : fred_balloons 671 := by
  sorry

end NUMINAMATH_CALUDE_fred_final_balloons_l1099_109989


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1099_109921

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ y = -(3/4) * x

/-- Theorem: The asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1099_109921


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1099_109995

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/4, -3/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := y + 3 = -9 * x

theorem intersection_point_is_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point := by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1099_109995


namespace NUMINAMATH_CALUDE_exponent_division_l1099_109992

theorem exponent_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1099_109992


namespace NUMINAMATH_CALUDE_middle_card_is_six_l1099_109919

theorem middle_card_is_six (a b c : ℕ) : 
  0 < a → 0 < b → 0 < c →
  a < b → b < c →
  a + b + c = 15 →
  (∀ x y z, x < y ∧ y < z ∧ x + y + z = 15 → x ≠ 3 ∨ (y ≠ 4 ∧ y ≠ 5)) →
  (∀ x y z, x < y ∧ y < z ∧ x + y + z = 15 → z ≠ 12 ∧ z ≠ 11 ∧ z ≠ 7) →
  (∃ p q, p < b ∧ b < q ∧ p + b + q = 15 ∧ (p ≠ a ∨ q ≠ c)) →
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_middle_card_is_six_l1099_109919


namespace NUMINAMATH_CALUDE_cloth_sale_gain_percentage_l1099_109925

/-- Calculates the gain percentage given the profit amount and total amount sold -/
def gainPercentage (profitAmount : ℕ) (totalAmount : ℕ) : ℚ :=
  (profitAmount : ℚ) / (totalAmount : ℚ) * 100

/-- Theorem: The gain percentage is 40% when the profit is 10 and the total amount sold is 25 -/
theorem cloth_sale_gain_percentage :
  gainPercentage 10 25 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_gain_percentage_l1099_109925


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l1099_109914

/-- Given three lines that intersect at the same point, find the value of p -/
theorem intersection_of_three_lines (p : ℝ) : 
  (∃ x y : ℝ, y = 3*x - 6 ∧ y = -4*x + 8 ∧ y = 7*x + p) → p = -14 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l1099_109914


namespace NUMINAMATH_CALUDE_team_games_count_l1099_109979

/-- Proves that a team playing under specific win conditions played 120 games in total -/
theorem team_games_count (first_games : ℕ) (first_win_rate : ℚ) (remaining_win_rate : ℚ) (total_win_rate : ℚ) : 
  first_games = 30 →
  first_win_rate = 2/5 →
  remaining_win_rate = 4/5 →
  total_win_rate = 7/10 →
  ∃ (total_games : ℕ), 
    total_games = 120 ∧
    (first_win_rate * first_games + remaining_win_rate * (total_games - first_games) : ℚ) = total_win_rate * total_games :=
by sorry

end NUMINAMATH_CALUDE_team_games_count_l1099_109979


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l1099_109938

/-- Represents the number of students in each year and the sample size -/
structure SchoolData where
  total_students : ℕ
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  sample_size : ℕ

/-- Represents the number of students to be sampled from each year -/
structure SampleAllocation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the correct sample allocation for stratified sampling -/
def stratifiedSample (data : SchoolData) : SampleAllocation :=
  { first_year := data.sample_size * data.first_year / data.total_students,
    second_year := data.sample_size * data.second_year / data.total_students,
    third_year := data.sample_size * data.third_year / data.total_students }

/-- Theorem stating that the stratified sampling allocation is correct -/
theorem stratified_sample_correct (data : SchoolData)
  (h1 : data.total_students = 2700)
  (h2 : data.first_year = 900)
  (h3 : data.second_year = 1200)
  (h4 : data.third_year = 600)
  (h5 : data.sample_size = 135) :
  stratifiedSample data = { first_year := 45, second_year := 60, third_year := 30 } :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_correct_l1099_109938


namespace NUMINAMATH_CALUDE_initial_books_borrowed_l1099_109978

/-- Represents the number of books Mary has at each stage --/
def books_count (initial : ℕ) : ℕ → ℕ
| 0 => initial  -- Initial number of books
| 1 => initial - 3 + 5  -- After first library visit
| 2 => initial - 3 + 5 - 2 + 7  -- After second library visit
| _ => 0  -- We don't need values beyond stage 2

/-- The theorem stating the initial number of books Mary borrowed --/
theorem initial_books_borrowed :
  ∃ (initial : ℕ), books_count initial 2 = 12 ∧ initial = 5 := by
  sorry


end NUMINAMATH_CALUDE_initial_books_borrowed_l1099_109978


namespace NUMINAMATH_CALUDE_claire_crafting_hours_l1099_109923

def total_hours : ℕ := 24
def cleaning_hours : ℕ := 4
def cooking_hours : ℕ := 2
def sleeping_hours : ℕ := 8

def remaining_hours : ℕ := total_hours - (cleaning_hours + cooking_hours + sleeping_hours)

theorem claire_crafting_hours : remaining_hours / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_claire_crafting_hours_l1099_109923


namespace NUMINAMATH_CALUDE_chicken_distribution_problem_l1099_109944

/-- The multiple of Skylar's chickens that Quentin has 25 more than -/
def chicken_multiple (total colten skylar quentin : ℕ) : ℕ :=
  (quentin - 25) / skylar

/-- Proof of the chicken distribution problem -/
theorem chicken_distribution_problem (total colten skylar quentin : ℕ) 
  (h1 : total = 383)
  (h2 : colten = 37)
  (h3 : skylar = 3 * colten - 4)
  (h4 : quentin + skylar + colten = total)
  (h5 : ∃ m : ℕ, quentin = m * skylar + 25) :
  chicken_multiple total colten skylar quentin = 2 := by
  sorry

#eval chicken_multiple 383 37 107 239

end NUMINAMATH_CALUDE_chicken_distribution_problem_l1099_109944


namespace NUMINAMATH_CALUDE_complex_calculation_proof_l1099_109973

theorem complex_calculation_proof :
  let expr1 := (1) - (3^3) * ((-1/3)^2) - 24 * (3/4 - 1/6 + 3/8)
  let expr2 := (2) - (1^100) - (3/4) / (((-2)^2) * ((-1/4)^2) - 1/2)
  (expr1 = -26) ∧ (expr2 = 2) := by
sorry

end NUMINAMATH_CALUDE_complex_calculation_proof_l1099_109973


namespace NUMINAMATH_CALUDE_rectangle_length_l1099_109961

/-- Proves that a rectangle with length 2 cm more than width and perimeter 20 cm has length 6 cm -/
theorem rectangle_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = width + 2 →
  perimeter = 2 * length + 2 * width →
  perimeter = 20 →
  length = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l1099_109961


namespace NUMINAMATH_CALUDE_biography_increase_l1099_109943

theorem biography_increase (B : ℝ) (b n : ℝ) 
  (h1 : b = 0.20 * B)  -- Initial biographies are 20% of total
  (h2 : b + n = 0.32 * (B + n))  -- After purchase, biographies are 32% of new total
  : (n / b) * 100 = 1500 / 17 := by
  sorry

end NUMINAMATH_CALUDE_biography_increase_l1099_109943


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1099_109947

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1099_109947


namespace NUMINAMATH_CALUDE_complex_power_six_l1099_109934

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l1099_109934


namespace NUMINAMATH_CALUDE_positive_real_product_and_sum_squares_l1099_109963

theorem positive_real_product_and_sum_squares (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2 * m * n) : 
  m * n ≥ 1 ∧ m^2 + n^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_product_and_sum_squares_l1099_109963


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1099_109968

theorem simplify_fraction_product : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1099_109968


namespace NUMINAMATH_CALUDE_factor_equality_l1099_109972

theorem factor_equality (x y : ℝ) : 9*x^2 - y^2 - 4*y - 4 = (3*x + y + 2)*(3*x - y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_equality_l1099_109972


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1099_109983

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1099_109983


namespace NUMINAMATH_CALUDE_tv_weight_difference_is_150_l1099_109903

/-- Calculates the weight difference between two TVs in pounds -/
def tv_weight_difference (l1 w1 l2 w2 : ℕ) (weight_per_sq_inch : ℚ) (oz_per_pound : ℕ) : ℚ :=
  let area1 := l1 * w1
  let area2 := l2 * w2
  let area_diff := max area1 area2 - min area1 area2
  let weight_diff_oz := (area_diff : ℚ) * weight_per_sq_inch
  weight_diff_oz / (oz_per_pound : ℚ)

theorem tv_weight_difference_is_150 :
  tv_weight_difference 48 100 70 60 (4 / 1) 16 = 150 := by
  sorry

end NUMINAMATH_CALUDE_tv_weight_difference_is_150_l1099_109903


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l1099_109990

theorem complex_number_opposite_parts (m : ℝ) : 
  let z : ℂ := (1 - m * I) / (1 - 2 * I)
  (∃ (a : ℝ), z = a - a * I) → m = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l1099_109990


namespace NUMINAMATH_CALUDE_line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular_l1099_109936

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel n α → perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular_l1099_109936


namespace NUMINAMATH_CALUDE_same_digit_sum_in_arithmetic_progression_l1099_109907

-- Define an arithmetic progression of natural numbers
def arithmeticProgression (a d : ℕ) : ℕ → ℕ := λ n => a + n * d

-- Define the sum of digits function
def sumOfDigits : ℕ → ℕ := sorry

theorem same_digit_sum_in_arithmetic_progression (a d : ℕ) :
  ∃ (k l : ℕ), k ≠ l ∧ sumOfDigits (arithmeticProgression a d k) = sumOfDigits (arithmeticProgression a d l) := by
  sorry

end NUMINAMATH_CALUDE_same_digit_sum_in_arithmetic_progression_l1099_109907


namespace NUMINAMATH_CALUDE_vector_collinearity_implies_ratio_l1099_109929

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 5)

-- Define collinearity for 2D vectors
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- State the theorem
theorem vector_collinearity_implies_ratio (m n : ℝ) (h_n : n ≠ 0) :
  collinear ((m * a.1 - n * b.1, m * a.2 - n * b.2) : ℝ × ℝ) (a.1 + 2 * b.1, a.2 + 2 * b.2) →
  m / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_implies_ratio_l1099_109929


namespace NUMINAMATH_CALUDE_fraction_of_25_problem_l1099_109922

theorem fraction_of_25_problem : ∃ x : ℚ, 
  x * 25 = 80 / 100 * 40 - 12 ∧ 
  x = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_25_problem_l1099_109922


namespace NUMINAMATH_CALUDE_x_minus_y_equals_40_l1099_109926

theorem x_minus_y_equals_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_40_l1099_109926


namespace NUMINAMATH_CALUDE_triangle_side_length_l1099_109999

/-- Prove that in a triangle ABC where angles A, B, C form an arithmetic sequence,
    if A = 75° and b = √3, then a = (√6 + √2) / 2. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  (B - A = C - B) → 
  -- A = 75°
  (A = 75 * π / 180) →
  -- b = √3
  (b = Real.sqrt 3) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Sum of angles in a triangle is π
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion: a = (√6 + √2) / 2
  a = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
    sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1099_109999


namespace NUMINAMATH_CALUDE_factorization_3x2_minus_12y2_l1099_109916

theorem factorization_3x2_minus_12y2 (x y : ℝ) :
  3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x2_minus_12y2_l1099_109916


namespace NUMINAMATH_CALUDE_equal_perimeter_interior_tiles_l1099_109956

/-- Represents a rectangular room with dimensions m × n -/
structure Room where
  m : ℕ
  n : ℕ
  h : m ≤ n

/-- The number of tiles on the perimeter of the room -/
def perimeterTiles (r : Room) : ℕ := 2 * r.m + 2 * r.n - 4

/-- The number of tiles in the interior of the room -/
def interiorTiles (r : Room) : ℕ := r.m * r.n - perimeterTiles r

/-- Predicate to check if a room has equal number of perimeter and interior tiles -/
def hasEqualTiles (r : Room) : Prop := perimeterTiles r = interiorTiles r

/-- The theorem stating that (5,12) and (6,8) are the only solutions -/
theorem equal_perimeter_interior_tiles :
  ∀ r : Room, hasEqualTiles r ↔ (r.m = 5 ∧ r.n = 12) ∨ (r.m = 6 ∧ r.n = 8) := by sorry

end NUMINAMATH_CALUDE_equal_perimeter_interior_tiles_l1099_109956


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_2023_l1099_109996

theorem absolute_value_of_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_2023_l1099_109996


namespace NUMINAMATH_CALUDE_larger_number_proof_l1099_109982

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1099_109982


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l1099_109918

theorem chocolate_gain_percent :
  ∀ (C S : ℝ),
  C > 0 →
  S > 0 →
  24 * C = 16 * S →
  (S - C) / C * 100 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l1099_109918


namespace NUMINAMATH_CALUDE_rational_sum_and_power_integers_l1099_109981

theorem rational_sum_and_power_integers (n : ℕ) : 
  (Odd n) ↔ 
  (∃ (a b : ℚ), 
    0 < a ∧ 0 < b ∧ 
    ¬(∃ (i : ℤ), a = i) ∧ ¬(∃ (j : ℤ), b = j) ∧
    (∃ (k : ℤ), (a + b : ℚ) = k) ∧ 
    (∃ (l : ℤ), (a^n + b^n : ℚ) = l)) :=
sorry

end NUMINAMATH_CALUDE_rational_sum_and_power_integers_l1099_109981


namespace NUMINAMATH_CALUDE_nancy_antacid_intake_l1099_109994

/-- Represents the number of antacids Nancy takes per day for different food types -/
structure AntacidIntake where
  indian : ℕ
  mexican : ℕ
  other : ℝ

/-- Represents Nancy's weekly food consumption -/
structure WeeklyConsumption where
  indian_days : ℕ
  mexican_days : ℕ

/-- Calculates Nancy's monthly antacid intake based on her eating habits -/
def monthly_intake (intake : AntacidIntake) (consumption : WeeklyConsumption) : ℝ :=
  4 * (intake.indian * consumption.indian_days + intake.mexican * consumption.mexican_days) +
  intake.other * (30 - 4 * (consumption.indian_days + consumption.mexican_days))

/-- Theorem stating Nancy's antacid intake for non-Indian and non-Mexican food days -/
theorem nancy_antacid_intake (intake : AntacidIntake) (consumption : WeeklyConsumption) :
  intake.indian = 3 →
  intake.mexican = 2 →
  consumption.indian_days = 3 →
  consumption.mexican_days = 2 →
  monthly_intake intake consumption = 60 →
  intake.other = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_nancy_antacid_intake_l1099_109994


namespace NUMINAMATH_CALUDE_special_right_triangle_legs_lengths_l1099_109954

/-- A right triangle with a point on the hypotenuse equidistant from both legs -/
structure SpecialRightTriangle where
  /-- Length of the first segment of the divided hypotenuse -/
  segment1 : ℝ
  /-- Length of the second segment of the divided hypotenuse -/
  segment2 : ℝ
  /-- The point divides the hypotenuse into the given segments -/
  hypotenuse_division : segment1 + segment2 = 70
  /-- The segments are positive -/
  segment1_pos : segment1 > 0
  segment2_pos : segment2 > 0

/-- The lengths of the legs of the special right triangle -/
def legs_lengths (t : SpecialRightTriangle) : ℝ × ℝ :=
  (42, 56)

/-- Theorem stating that the legs of the special right triangle have lengths 42 and 56 -/
theorem special_right_triangle_legs_lengths (t : SpecialRightTriangle)
    (h1 : t.segment1 = 30) (h2 : t.segment2 = 40) :
    legs_lengths t = (42, 56) := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_legs_lengths_l1099_109954


namespace NUMINAMATH_CALUDE_range_of_x_l1099_109998

theorem range_of_x (x : ℝ) : 
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → |3*a + b| + |a - b| ≥ |a| * (|x - 1| + |x + 1|)) 
  ↔ x ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1099_109998


namespace NUMINAMATH_CALUDE_equation_equality_l1099_109970

theorem equation_equality (a b : ℝ) : -0.25 * a * b + (1/4) * a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1099_109970


namespace NUMINAMATH_CALUDE_forum_total_posts_per_day_l1099_109927

/-- Represents a question and answer forum --/
structure Forum where
  members : ℕ
  questionsPerHour : ℕ
  answerRatio : ℕ

/-- Calculates the total number of questions and answers posted in a day --/
def totalPostsPerDay (f : Forum) : ℕ :=
  let questionsPerDay := f.members * (f.questionsPerHour * 24)
  let answersPerDay := f.members * (f.questionsPerHour * f.answerRatio * 24)
  questionsPerDay + answersPerDay

/-- Theorem stating the total number of posts per day for the given forum --/
theorem forum_total_posts_per_day :
  ∃ (f : Forum), f.members = 200 ∧ f.questionsPerHour = 3 ∧ f.answerRatio = 3 ∧
  totalPostsPerDay f = 57600 :=
by
  sorry

end NUMINAMATH_CALUDE_forum_total_posts_per_day_l1099_109927


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l1099_109932

theorem root_reciprocal_sum (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  1/(a-1) + 1/(b-1) + 1/(c-1) = -1 := by
sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l1099_109932


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l1099_109906

theorem arithmetic_sequence_sum_product (a b c : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (a + b + c = 12) →  -- sum condition
  (a * b * c = 48) →  -- product condition
  ((a = 2 ∧ b = 4 ∧ c = 6) ∨ (a = 6 ∧ b = 4 ∧ c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l1099_109906


namespace NUMINAMATH_CALUDE_quadratic_ratio_l1099_109987

/-- Given a quadratic polynomial x^2 + 1500x + 1800, prove that when written in the form (x+a)^2 + d,
    the ratio d/a equals -560700/750. -/
theorem quadratic_ratio (x : ℝ) :
  ∃ (a d : ℝ), x^2 + 1500*x + 1800 = (x + a)^2 + d ∧ d / a = -560700 / 750 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l1099_109987


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_in_equilateral_pyramid_l1099_109930

/-- A pyramid with an equilateral triangular base and equilateral triangular lateral faces -/
structure EquilateralPyramid where
  base_side_length : ℝ
  lateral_face_is_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  side_length : ℝ
  base_on_pyramid_base : Bool
  top_edges_on_lateral_faces : Bool

/-- The volume of the inscribed cube in the given pyramid -/
def inscribed_cube_volume (p : EquilateralPyramid) (c : InscribedCube) : ℝ :=
  c.side_length ^ 3

theorem inscribed_cube_volume_in_equilateral_pyramid 
  (p : EquilateralPyramid) 
  (c : InscribedCube) 
  (h1 : p.base_side_length = 2)
  (h2 : p.lateral_face_is_equilateral = true)
  (h3 : c.base_on_pyramid_base = true)
  (h4 : c.top_edges_on_lateral_faces = true) :
  inscribed_cube_volume p c = 3 * Real.sqrt 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_in_equilateral_pyramid_l1099_109930


namespace NUMINAMATH_CALUDE_expression_simplification_l1099_109949

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1099_109949


namespace NUMINAMATH_CALUDE_log_equation_solution_l1099_109908

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 4 + 2 * (Real.log x / Real.log 8) = 7 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1099_109908


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1099_109928

theorem x_minus_y_value (x y : ℝ) 
  (eq1 : 3015 * x + 3020 * y = 3025)
  (eq2 : 3018 * x + 3024 * y = 3030) : 
  x - y = 11.1167 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1099_109928


namespace NUMINAMATH_CALUDE_rectangle_difference_l1099_109993

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  breadth : ℕ

/-- The perimeter of the rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.breadth)

/-- The area of the rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.breadth

/-- The difference between length and breadth -/
def Rectangle.difference (r : Rectangle) : ℕ := r.length - r.breadth

theorem rectangle_difference (r : Rectangle) :
  r.perimeter = 266 ∧ r.area = 4290 → r.difference = 23 := by
  sorry

#eval Rectangle.difference { length := 78, breadth := 55 }

end NUMINAMATH_CALUDE_rectangle_difference_l1099_109993


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l1099_109988

/-- The cost price of one metre of cloth given the selling price, quantity, and profit per metre -/
theorem cost_price_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℚ) 
  (profit_per_metre : ℚ) 
  (h1 : total_metres = 85)
  (h2 : total_selling_price = 8925)
  (h3 : profit_per_metre = 15) :
  (total_selling_price - total_metres * profit_per_metre) / total_metres = 90 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l1099_109988


namespace NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l1099_109920

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l1099_109920


namespace NUMINAMATH_CALUDE_emir_needs_two_more_dollars_l1099_109986

/-- The amount of additional money Emir needs to buy three books --/
def additional_money_needed (dictionary_cost cookbook_cost dinosaur_book_cost savings : ℕ) : ℕ :=
  (dictionary_cost + cookbook_cost + dinosaur_book_cost) - savings

/-- Theorem: Emir needs $2 more to buy all three books --/
theorem emir_needs_two_more_dollars : 
  additional_money_needed 5 5 11 19 = 2 := by
  sorry

end NUMINAMATH_CALUDE_emir_needs_two_more_dollars_l1099_109986


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1099_109951

-- Define the set of all functions
variable (F : Type)

-- Define the property of being a logarithmic function
variable (isLogarithmic : F → Prop)

-- Define the property of being a monotonic function
variable (isMonotonic : F → Prop)

-- The theorem to prove
theorem negation_of_universal_proposition :
  (¬ ∀ f : F, isLogarithmic f → isMonotonic f) ↔ 
  (∃ f : F, isLogarithmic f ∧ ¬isMonotonic f) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1099_109951


namespace NUMINAMATH_CALUDE_total_celestial_bodies_l1099_109933

-- Define the number of planets
def num_planets : ℕ := 20

-- Define the ratio of solar systems to planets
def solar_system_ratio : ℕ := 8

-- Theorem: The total number of solar systems and planets is 180
theorem total_celestial_bodies : 
  num_planets * (solar_system_ratio + 1) = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_celestial_bodies_l1099_109933


namespace NUMINAMATH_CALUDE_chris_age_l1099_109939

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Five years ago, Chris was the same age as Amy is now
  ages.chris - 5 = ages.amy ∧
  -- In 4 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 4 = (3/4) * (ages.amy + 4)

/-- The theorem to be proved -/
theorem chris_age (ages : Ages) :
  problem_conditions ages → ages.chris = 15.55 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l1099_109939


namespace NUMINAMATH_CALUDE_area_of_triangle_AOC_l1099_109904

/-- Given three collinear points A, B, and C in a Cartesian coordinate system with origin O,
    where OA = (-2, m), OB = (n, 1), OC = (5, -1), OA ⊥ OB,
    G is the centroid of triangle OAC, and OB = (3/2) * OG,
    prove that the area of triangle AOC is 13/2. -/
theorem area_of_triangle_AOC (m n : ℝ) (A B C G : ℝ × ℝ) :
  A.1 = -2 ∧ A.2 = m →
  B.1 = n ∧ B.2 = 1 →
  C = (5, -1) →
  A.1 * B.1 + A.2 * B.2 = 0 →  -- OA ⊥ OB
  G = ((0 + A.1 + C.1) / 3, (0 + A.2 + C.2) / 3) →  -- G is centroid of OAC
  B = (3/2 : ℝ) • G →  -- OB = (3/2) * OG
  (A.1 - C.1) * (B.2 - A.2) = (B.1 - A.1) * (A.2 - C.2) →  -- A, B, C are collinear
  abs ((A.1 * C.2 - C.1 * A.2) / 2) = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AOC_l1099_109904


namespace NUMINAMATH_CALUDE_scientific_notation_of_169200000000_l1099_109935

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to convert to scientific notation -/
def number : ℝ := 169200000000

/-- Theorem stating that the scientific notation of 169200000000 is 1.692 × 10^11 -/
theorem scientific_notation_of_169200000000 :
  toScientificNotation number = ScientificNotation.mk 1.692 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_169200000000_l1099_109935


namespace NUMINAMATH_CALUDE_cube_volumes_sum_l1099_109975

theorem cube_volumes_sum (a b c : ℕ) (h : 6 * (a^2 + b^2 + c^2) = 564) :
  a^3 + b^3 + c^3 = 764 ∨ a^3 + b^3 + c^3 = 586 :=
by sorry

end NUMINAMATH_CALUDE_cube_volumes_sum_l1099_109975


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l1099_109960

/-- Defines the equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 6)^2 + (y + 4)^2) = 12

/-- Theorem stating that the equation describes an ellipse --/
theorem conic_is_ellipse : ∃ (a b x₀ y₀ : ℝ), 
  (∀ x y : ℝ, conic_equation x y ↔ 
    ((x - x₀) / a)^2 + ((y - y₀) / b)^2 = 1) ∧ 
  a > 0 ∧ b > 0 ∧ a ≠ b :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l1099_109960


namespace NUMINAMATH_CALUDE_ellipse_parabola_triangle_area_l1099_109980

/-- Definition of the ellipse C₁ -/
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

/-- Definition of the parabola C₂ -/
def parabola (x y : ℝ) : Prop := x^2 = 8 * y

/-- The focus F of the parabola, which is also the vertex of the ellipse -/
def F : ℝ × ℝ := (0, 2)

/-- Definition of a point being on the ellipse -/
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

/-- Definition of two vectors being orthogonal -/
def orthogonal (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

/-- Definition of a line being tangent to the parabola -/
def tangent_to_parabola (P Q : ℝ × ℝ) : Prop :=
  ∃ k m : ℝ, (∀ x y : ℝ, y = k * x + m → (x^2 = 8 * y ↔ x = P.1 ∧ y = P.2))

theorem ellipse_parabola_triangle_area :
  ∀ P Q : ℝ × ℝ,
  on_ellipse P → on_ellipse Q →
  orthogonal (P.1 - F.1, P.2 - F.2) (Q.1 - F.1, Q.2 - F.2) →
  tangent_to_parabola P Q →
  P ≠ F → Q ≠ F → P ≠ Q →
  abs ((P.1 - F.1) * (Q.2 - F.2) - (P.2 - F.2) * (Q.1 - F.1)) / 2 = 18 * Real.sqrt 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_triangle_area_l1099_109980


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1099_109985

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1099_109985


namespace NUMINAMATH_CALUDE_team_E_not_played_B_l1099_109902

/-- Represents a team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- The number of matches played by each team -/
def matches_played (t : Team) : ℕ :=
  match t with
  | Team.A => 5
  | Team.B => 4
  | Team.C => 3
  | Team.D => 2
  | Team.E => 1
  | Team.F => 5  -- Implied from the problem context

/-- The total number of teams in the tournament -/
def total_teams : ℕ := 6

/-- The total number of matches each team should play in a round-robin tournament -/
def total_matches_per_team : ℕ := total_teams - 1

/-- Predicate to check if a team has played against team B -/
def has_played_against_B (t : Team) : Prop :=
  t ≠ Team.B ∧ (matches_played Team.B + matches_played t > total_matches_per_team)

theorem team_E_not_played_B :
  ∀ t : Team, t ≠ Team.B → (¬has_played_against_B t ↔ t = Team.E) :=
sorry

end NUMINAMATH_CALUDE_team_E_not_played_B_l1099_109902


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1099_109941

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 48) :
  x^2 + 4*x*y + 4*y^2 + 3*z^2 ≥ 144 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 48 ∧ x^2 + 4*x*y + 4*y^2 + 3*z^2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1099_109941


namespace NUMINAMATH_CALUDE_guitar_sales_proof_l1099_109984

/-- Calculates the total amount earned from selling guitars -/
def total_guitar_sales (total_guitars : ℕ) (electric_guitars : ℕ) (electric_price : ℕ) (acoustic_price : ℕ) : ℕ :=
  let acoustic_guitars := total_guitars - electric_guitars
  electric_guitars * electric_price + acoustic_guitars * acoustic_price

/-- Proves that the total amount earned from selling 9 guitars, 
    consisting of 4 electric guitars at $479 each and 5 acoustic guitars at $339 each, is $3611 -/
theorem guitar_sales_proof : 
  total_guitar_sales 9 4 479 339 = 3611 := by
  sorry

#eval total_guitar_sales 9 4 479 339

end NUMINAMATH_CALUDE_guitar_sales_proof_l1099_109984


namespace NUMINAMATH_CALUDE_power_function_through_point_l1099_109955

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f (1/2) = 8) : 
  f 2 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1099_109955


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1099_109977

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 25| + |x - 21| = |2*x - 46| + |x - 17| ∧ x = 67/3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1099_109977


namespace NUMINAMATH_CALUDE_function_characterization_l1099_109905

/-- Euler's totient function -/
noncomputable def φ : ℕ+ → ℕ+ :=
  sorry

/-- The property that the function f satisfies -/
def satisfies_property (f : ℕ+ → ℕ+) : Prop :=
  ∀ (m n : ℕ+), m ≥ n → f (m * φ (n^3)) = f m * φ (n^3)

/-- The main theorem -/
theorem function_characterization :
  ∀ (f : ℕ+ → ℕ+), satisfies_property f →
  ∃ (b : ℕ+), ∀ (n : ℕ+), f n = b * n :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l1099_109905


namespace NUMINAMATH_CALUDE_distance_to_origin_l1099_109900

theorem distance_to_origin (x y z : ℝ) :
  (|x| = 2 ∧ |y| = 2 ∧ |z| = 2) →
  Real.sqrt (x^2 + y^2 + z^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1099_109900


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1099_109915

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1099_109915


namespace NUMINAMATH_CALUDE_count_integers_correct_l1099_109958

/-- Count of three-digit positive integers starting with 2 and greater than 217 -/
def count_integers : ℕ := 82

/-- The smallest three-digit integer starting with 2 and greater than 217 -/
def min_integer : ℕ := 218

/-- The largest three-digit integer starting with 2 -/
def max_integer : ℕ := 299

/-- Theorem stating that the count of integers is correct -/
theorem count_integers_correct :
  count_integers = max_integer - min_integer + 1 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_correct_l1099_109958


namespace NUMINAMATH_CALUDE_tangent_line_at_one_f_greater_than_one_l1099_109912

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.exp x * (Real.log x + 1 / x)

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m : ℝ), ∀ (x y : ℝ), y = m * (x - 1) + f 1 ↔ Real.exp x - y = 0 :=
sorry

-- Theorem for the magnitude comparison
theorem f_greater_than_one :
  ∀ (x : ℝ), x > 0 → f x > 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_f_greater_than_one_l1099_109912


namespace NUMINAMATH_CALUDE_range_of_a_l1099_109909

-- Define the set of real numbers x in [1,2]
def X : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- Define the set of real numbers y in [2,3]
def Y : Set ℝ := { y | 2 ≤ y ∧ y ≤ 3 }

-- State the theorem
theorem range_of_a (x : ℝ) (y : ℝ) (h1 : x ∈ X) (h2 : y ∈ Y) :
  ∃ a : ℝ, (∀ (x' : ℝ) (y' : ℝ), x' ∈ X → y' ∈ Y → x'*y' ≤ a*x'^2 + 2*y'^2) ∧
            (a ≥ -1) ∧
            (∀ b : ℝ, b > a → ∃ (x' : ℝ) (y' : ℝ), x' ∈ X ∧ y' ∈ Y ∧ x'*y' > b*x'^2 + 2*y'^2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1099_109909


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_pentagon_l1099_109964

/-- The sum of interior angles of a regular polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A regular pentagon has 5 sides -/
def regular_pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a regular pentagon is 540 degrees -/
theorem sum_interior_angles_regular_pentagon :
  sum_interior_angles regular_pentagon_sides = 540 := by
  sorry


end NUMINAMATH_CALUDE_sum_interior_angles_regular_pentagon_l1099_109964


namespace NUMINAMATH_CALUDE_matrix_square_zero_implication_l1099_109969

theorem matrix_square_zero_implication (n : ℕ) (M N : Matrix (Fin n) (Fin n) ℝ) 
  (h : (M * N)^2 = 0) :
  (n = 2 → (N * M)^2 = 0) ∧ 
  (n ≥ 3 → ∃ (M' N' : Matrix (Fin n) (Fin n) ℝ), (M' * N')^2 = 0 ∧ (N' * M')^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_matrix_square_zero_implication_l1099_109969


namespace NUMINAMATH_CALUDE_total_height_increase_two_centuries_l1099_109924

/-- Represents the increase in height (in meters) per decade for a specific species of plants -/
def height_increase_per_decade : ℕ := 90

/-- Represents the number of decades in 2 centuries -/
def decades_in_two_centuries : ℕ := 20

/-- Theorem stating that the total increase in height over 2 centuries is 1800 meters -/
theorem total_height_increase_two_centuries : 
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_height_increase_two_centuries_l1099_109924


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_quadratic_form_components_l1099_109950

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (x + 1)^2 + (x - 2) * (x + 2) = 1 ↔ 2 * x^2 + 2 * x - 4 = 0 :=
by sorry

-- Definitions for the components of the quadratic equation
def quadratic_term (x : ℝ) : ℝ := 2 * x^2
def quadratic_coefficient : ℝ := 2
def linear_term (x : ℝ) : ℝ := 2 * x
def linear_coefficient : ℝ := 2
def constant_term : ℝ := -4

-- Theorem stating that the transformed equation is in the general form of a quadratic equation
theorem quadratic_form_components (x : ℝ) :
  2 * x^2 + 2 * x - 4 = quadratic_term x + linear_term x + constant_term :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_quadratic_form_components_l1099_109950


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1099_109913

/-- The area of the region between a circle circumscribing two externally tangent circles and those two circles -/
theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) : 
  let R := r2 + (r1 + r2) / 2
  π * R^2 - π * r1^2 - π * r2^2 = 49.25 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1099_109913


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l1099_109945

/-- Given two types of candy mixed to produce a mixture selling at a certain price,
    calculate the total amount of mixture produced. -/
theorem candy_mixture_problem (x : ℝ) : 
  x > 0 ∧ 
  3.50 * x + 4.30 * 6.25 = 4.00 * (x + 6.25) → 
  x + 6.25 = 10 := by
  sorry

#check candy_mixture_problem

end NUMINAMATH_CALUDE_candy_mixture_problem_l1099_109945


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l1099_109946

theorem intersection_empty_implies_a_nonnegative 
  (A : Set ℝ) (B : Set ℝ) (a : ℝ) 
  (h1 : A = {x : ℝ | x - a > 0})
  (h2 : B = {x : ℝ | x ≤ 0})
  (h3 : A ∩ B = ∅) :
  a ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l1099_109946


namespace NUMINAMATH_CALUDE_three_heads_probability_l1099_109940

/-- A fair coin has a probability of 1/2 for heads on a single flip -/
def fair_coin_prob : ℚ := 1/2

/-- The probability of getting three heads in three flips of a fair coin -/
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

/-- Theorem: The probability of getting three heads in three flips of a fair coin is 1/8 -/
theorem three_heads_probability : three_heads_prob = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_probability_l1099_109940


namespace NUMINAMATH_CALUDE_rotate_minus_six_minus_three_i_l1099_109967

/-- Rotate a complex number by 180 degrees counter-clockwise around the origin -/
def rotate180 (z : ℂ) : ℂ := -z

/-- The theorem stating that rotating -6 - 3i by 180 degrees results in 6 + 3i -/
theorem rotate_minus_six_minus_three_i :
  rotate180 (-6 - 3*I) = (6 + 3*I) := by
  sorry

end NUMINAMATH_CALUDE_rotate_minus_six_minus_three_i_l1099_109967


namespace NUMINAMATH_CALUDE_monomial_count_l1099_109962

/-- An algebraic expression is a monomial if it consists of a single term. -/
def isMonomial (expr : String) : Bool := sorry

/-- The set of given algebraic expressions. -/
def expressions : List String := [
  "3a^2 + b",
  "-2",
  "3xy^3/5",
  "a^2b/3 + 1",
  "a^2 - 3b^2",
  "2abc"
]

/-- Counts the number of monomials in a list of expressions. -/
def countMonomials (exprs : List String) : Nat :=
  exprs.filter isMonomial |>.length

theorem monomial_count :
  countMonomials expressions = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l1099_109962


namespace NUMINAMATH_CALUDE_number_of_petunia_flats_l1099_109997

/-- The number of petunias per flat of petunias -/
def petunias_per_flat : ℕ := 8

/-- The amount of fertilizer needed for each petunia in ounces -/
def fertilizer_per_petunia : ℕ := 8

/-- The number of flats of roses -/
def rose_flats : ℕ := 3

/-- The number of roses per flat of roses -/
def roses_per_flat : ℕ := 6

/-- The amount of fertilizer needed for each rose in ounces -/
def fertilizer_per_rose : ℕ := 3

/-- The number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- The amount of fertilizer needed for each Venus flytrap in ounces -/
def fertilizer_per_venus_flytrap : ℕ := 2

/-- The total amount of fertilizer needed in ounces -/
def total_fertilizer : ℕ := 314

/-- The theorem stating that the number of flats of petunias is 4 -/
theorem number_of_petunia_flats : 
  ∃ (P : ℕ), P * (petunias_per_flat * fertilizer_per_petunia) + 
             (rose_flats * roses_per_flat * fertilizer_per_rose) + 
             (venus_flytraps * fertilizer_per_venus_flytrap) = total_fertilizer ∧ 
             P = 4 :=
by sorry

end NUMINAMATH_CALUDE_number_of_petunia_flats_l1099_109997


namespace NUMINAMATH_CALUDE_unique_prime_square_equation_l1099_109971

theorem unique_prime_square_equation : 
  ∃! p : ℕ, Prime p ∧ ∃ k : ℕ, 2 * p^4 - 7 * p^2 + 1 = k^2 := by sorry

end NUMINAMATH_CALUDE_unique_prime_square_equation_l1099_109971


namespace NUMINAMATH_CALUDE_new_student_weight_l1099_109957

theorem new_student_weight (n : ℕ) (w_avg : ℝ) (w_new_avg : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new_avg = 27.5 →
  (n : ℝ) * w_avg + (n + 1) * w_new_avg - n * w_avg = 13 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l1099_109957


namespace NUMINAMATH_CALUDE_vector_equation_and_parallelism_l1099_109966

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem vector_equation_and_parallelism :
  (a = (5/9 : ℝ) • b + (8/9 : ℝ) • c) ∧
  (∃ (t : ℝ), t • (a + (-16/13 : ℝ) • c) = 2 • b - a) :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_and_parallelism_l1099_109966


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1099_109911

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1099_109911


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1099_109931

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔
  (∀ x : ℝ, x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1099_109931


namespace NUMINAMATH_CALUDE_fourth_root_of_sqrt_fraction_l1099_109959

theorem fourth_root_of_sqrt_fraction : 
  (32 / 10000 : ℝ)^(1/4 * 1/2) = (2 : ℝ)^(1/8) / (5 : ℝ)^(1/2) := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_sqrt_fraction_l1099_109959


namespace NUMINAMATH_CALUDE_some_ai_in_machines_l1099_109917

-- Define the sets
variable (Robot : Type) -- Set of all robots
variable (Machine : Type) -- Set of all machines
variable (AdvancedAI : Type) -- Set of all advanced AI systems

-- Define the relations
variable (has_ai : Robot → AdvancedAI → Prop) -- Relation: robot has advanced AI
variable (is_machine : Robot → Machine → Prop) -- Relation: robot is a machine

-- State the theorem
theorem some_ai_in_machines 
  (h1 : ∀ (r : Robot), ∃ (ai : AdvancedAI), has_ai r ai) -- All robots have advanced AI
  (h2 : ∃ (r : Robot) (m : Machine), is_machine r m) -- Some robots are machines
  : ∃ (ai : AdvancedAI) (m : Machine), 
    ∃ (r : Robot), has_ai r ai ∧ is_machine r m :=
by sorry

end NUMINAMATH_CALUDE_some_ai_in_machines_l1099_109917


namespace NUMINAMATH_CALUDE_michael_has_eight_robots_l1099_109901

/-- The number of animal robots Michael has -/
def michaels_robots : ℕ := sorry

/-- The number of animal robots Tom has -/
def toms_robots : ℕ := 16

/-- Tom has twice as many animal robots as Michael -/
axiom twice_as_many : toms_robots = 2 * michaels_robots

theorem michael_has_eight_robots : michaels_robots = 8 := by sorry

end NUMINAMATH_CALUDE_michael_has_eight_robots_l1099_109901


namespace NUMINAMATH_CALUDE_carpet_fit_theorem_l1099_109948

theorem carpet_fit_theorem (carpet_width carpet_length room1_width room2_width room_length : ℕ) :
  carpet_width = 25 ∧ 
  carpet_length = 50 ∧ 
  room1_width = 38 ∧ 
  room2_width = 50 ∧ 
  room_length > 0 →
  (carpet_width^2 + carpet_length^2 = room1_width^2 + room_length^2) ∧
  (carpet_width^2 + carpet_length^2 = room2_width^2 + room_length^2) :=
by sorry

#check carpet_fit_theorem

end NUMINAMATH_CALUDE_carpet_fit_theorem_l1099_109948


namespace NUMINAMATH_CALUDE_satellite_orbits_in_week_l1099_109910

/-- The number of orbits a satellite completes in one week -/
def orbits_in_week (hours_per_orbit : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) : ℕ :=
  (days_per_week * hours_per_day) / hours_per_orbit

/-- Theorem: A satellite orbiting Earth once every 7 hours completes 24 orbits in one week -/
theorem satellite_orbits_in_week :
  orbits_in_week 7 7 24 = 24 := by
  sorry

#eval orbits_in_week 7 7 24

end NUMINAMATH_CALUDE_satellite_orbits_in_week_l1099_109910


namespace NUMINAMATH_CALUDE_max_bracelet_earnings_l1099_109991

theorem max_bracelet_earnings :
  let total_bracelets : ℕ := 235
  let bracelets_per_bag : ℕ := 10
  let price_per_bag : ℕ := 3000
  let full_bags : ℕ := total_bracelets / bracelets_per_bag
  let max_earnings : ℕ := full_bags * price_per_bag
  max_earnings = 69000 := by
  sorry

end NUMINAMATH_CALUDE_max_bracelet_earnings_l1099_109991


namespace NUMINAMATH_CALUDE_three_color_theorem_l1099_109974

theorem three_color_theorem : ∃ f : ℕ → Fin 3,
  (∀ n : ℕ, ∀ x y : ℕ, 2^n ≤ x ∧ x < 2^(n+1) ∧ 2^n ≤ y ∧ y < 2^(n+1) → f x = f y) ∧
  (∀ x y z : ℕ, f x = f y ∧ f y = f z ∧ x + y = z^2 → x = 2 ∧ y = 2 ∧ z = 2) :=
sorry

end NUMINAMATH_CALUDE_three_color_theorem_l1099_109974


namespace NUMINAMATH_CALUDE_license_plate_combinations_l1099_109953

def number_of_letter_combinations : ℕ :=
  (Nat.choose 26 2) * 24 * (5 * 4 * 3 / (2 * 2))

def number_of_digit_combinations : ℕ := 10 * 9 * 8

theorem license_plate_combinations :
  number_of_letter_combinations * number_of_digit_combinations = 5644800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l1099_109953
