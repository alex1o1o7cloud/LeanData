import Mathlib

namespace NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l329_32962

theorem integral_x_squared_plus_sin_x : ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l329_32962


namespace NUMINAMATH_CALUDE_speed_ratio_walking_l329_32946

/-- Theorem: Ratio of speeds when two people walk towards each other and in the same direction -/
theorem speed_ratio_walking (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b > a) : ∃ (v₁ v₂ : ℝ),
  v₁ > 0 ∧ v₂ > 0 ∧ 
  (∃ (S : ℝ), S > 0 ∧ S = a * (v₁ + v₂) ∧ S = b * (v₁ - v₂)) ∧
  v₂ / v₁ = (a + b) / (b - a) :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_walking_l329_32946


namespace NUMINAMATH_CALUDE_coefficient_x3_equals_negative_30_l329_32953

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (1-2x)(1-x)^5
def coefficient_x3 : ℤ :=
  -1 * (-1) * binomial 5 3 + (-2) * binomial 5 2

-- Theorem statement
theorem coefficient_x3_equals_negative_30 : coefficient_x3 = -30 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_equals_negative_30_l329_32953


namespace NUMINAMATH_CALUDE_shaded_area_of_square_pattern_l329_32971

/-- Given a square with side length a, this theorem proves that the area of the shaded region
    formed by connecting vertices to midpoints of opposite sides in a pattern is (3/5) * a^2. -/
theorem shaded_area_of_square_pattern (a : ℝ) (h : a > 0) : ℝ :=
  let square_area := a^2
  let shaded_area := (3/5) * square_area
  shaded_area

#check shaded_area_of_square_pattern

end NUMINAMATH_CALUDE_shaded_area_of_square_pattern_l329_32971


namespace NUMINAMATH_CALUDE_chapters_read_l329_32948

theorem chapters_read (num_books : ℕ) (chapters_per_book : ℕ) (total_chapters : ℕ) : 
  num_books = 10 → chapters_per_book = 24 → total_chapters = num_books * chapters_per_book →
  total_chapters = 240 :=
by sorry

end NUMINAMATH_CALUDE_chapters_read_l329_32948


namespace NUMINAMATH_CALUDE_max_distance_line_equation_l329_32932

/-- The line of maximum distance from the origin passing through (2, 3) -/
def max_distance_line (x y : ℝ) : Prop :=
  2 * x + 3 * y - 13 = 0

/-- The point through which the line passes -/
def point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the line of maximum distance from the origin
    passing through (2, 3) has the equation 2x + 3y - 13 = 0 -/
theorem max_distance_line_equation :
  ∀ x y : ℝ, (x, y) ∈ ({p : ℝ × ℝ | p.1 * point.2 + p.2 * point.1 = point.1 * point.2} : Set (ℝ × ℝ)) →
  max_distance_line x y :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_equation_l329_32932


namespace NUMINAMATH_CALUDE_cannot_determine_heavier_l329_32992

variable (M P O : ℝ)

def mandarin_lighter_than_pear := M < P
def orange_heavier_than_mandarin := O > M

theorem cannot_determine_heavier (h1 : mandarin_lighter_than_pear M P) 
  (h2 : orange_heavier_than_mandarin O M) : 
  ¬(∀ x y : ℝ, (x < y) ∨ (y < x)) :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_heavier_l329_32992


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l329_32954

-- Define the inverse proportionality relationship
def inverse_proportional (y x : ℝ) := ∃ k : ℝ, y = k / (x + 2)

-- Define the theorem
theorem inverse_proportion_problem (y x : ℝ) 
  (h1 : inverse_proportional y x) 
  (h2 : y = 3 ∧ x = -1) :
  (∀ x, y = 3 / (x + 2)) ∧ 
  (x = 0 → y = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l329_32954


namespace NUMINAMATH_CALUDE_snack_expenditure_l329_32900

theorem snack_expenditure (initial_amount : ℕ) (computer_accessories : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 48)
  (h2 : computer_accessories = 12)
  (h3 : remaining_amount = initial_amount / 2 + 4) :
  initial_amount - computer_accessories - remaining_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_snack_expenditure_l329_32900


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l329_32960

-- Define the matrices
def matrix1 (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def matrix2 (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^3, a^2*b, a^2*c],
    ![a*b^2, b^3, b^2*c],
    ![a*c^2, b*c^2, c^3]]

-- Theorem statement
theorem matrix_product_is_zero (a b c d e f : ℝ) :
  matrix1 d e f * matrix2 a b c = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l329_32960


namespace NUMINAMATH_CALUDE_outstanding_student_allocation_schemes_l329_32947

theorem outstanding_student_allocation_schemes :
  let total_slots : ℕ := 7
  let num_schools : ℕ := 5
  let min_slots_for_two_schools : ℕ := 2
  let remaining_slots : ℕ := total_slots - 2 * min_slots_for_two_schools
  Nat.choose (remaining_slots + num_schools - 1) (num_schools - 1) = Nat.choose total_slots (total_slots - remaining_slots) := by
  sorry

end NUMINAMATH_CALUDE_outstanding_student_allocation_schemes_l329_32947


namespace NUMINAMATH_CALUDE_unique_base_conversion_l329_32937

def base_conversion (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem unique_base_conversion : ∃! x : Nat,
  x < 1000 ∧
  x ≥ 100 ∧
  let digits := [x / 100, (x / 10) % 10, x % 10]
  base_conversion digits 20 = 2 * base_conversion digits 13 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_conversion_l329_32937


namespace NUMINAMATH_CALUDE_range_for_two_roots_roots_for_negative_integer_k_l329_32921

/-- The quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ :=
  x^2 + (2*k + 1)*x + k^2 - 1

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  (2*k + 1)^2 - 4*(k^2 - 1)

/-- Theorem stating the range of k for which the equation has two distinct real roots -/
theorem range_for_two_roots :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) ↔ k > -5/4 :=
sorry

/-- Theorem stating the roots when k is a negative integer satisfying the range condition -/
theorem roots_for_negative_integer_k :
  ∀ k : ℤ, k < 0 → k > -5/4 → quadratic (↑k) 0 = 0 ∧ quadratic (↑k) 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_range_for_two_roots_roots_for_negative_integer_k_l329_32921


namespace NUMINAMATH_CALUDE_jame_card_tearing_l329_32997

/-- The number of cards Jame can tear at a time -/
def cards_per_tear : ℕ := 30

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of times Jame tears cards per week -/
def tears_per_week : ℕ := 3

/-- The number of decks Jame buys -/
def decks_bought : ℕ := 18

/-- The number of weeks Jame can tear cards -/
def weeks_of_tearing : ℕ := 11

theorem jame_card_tearing :
  (cards_per_tear * tears_per_week) * weeks_of_tearing ≤ cards_per_deck * decks_bought ∧
  (cards_per_tear * tears_per_week) * (weeks_of_tearing + 1) > cards_per_deck * decks_bought :=
by sorry

end NUMINAMATH_CALUDE_jame_card_tearing_l329_32997


namespace NUMINAMATH_CALUDE_unique_true_proposition_l329_32976

theorem unique_true_proposition :
  (¬ ∀ x : ℝ, x^2 + 3 < 0) ∧
  (¬ ∀ x : ℕ, x^2 ≥ 1) ∧
  (∃ x : ℤ, x^5 < 1) ∧
  (¬ ∃ x : ℚ, x^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_true_proposition_l329_32976


namespace NUMINAMATH_CALUDE_train_distance_difference_l329_32958

theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 20)
  (h2 : v2 = 25)
  (h3 : total_distance = 675) :
  let t := total_distance / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  |d2 - d1| = 75 := by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l329_32958


namespace NUMINAMATH_CALUDE_largest_number_l329_32973

theorem largest_number (a b c d : ℝ) 
  (h : a + 5 = b^2 - 1 ∧ a + 5 = c^2 + 3 ∧ a + 5 = d - 4) : 
  d > a ∧ d > b ∧ d > c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l329_32973


namespace NUMINAMATH_CALUDE_power_of_81_equals_9_l329_32999

theorem power_of_81_equals_9 : (81 : ℝ) ^ (0.25 : ℝ) * (81 : ℝ) ^ (0.20 : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_equals_9_l329_32999


namespace NUMINAMATH_CALUDE_square_of_negative_product_l329_32981

theorem square_of_negative_product (a b : ℝ) : (-3 * a * b^2)^2 = 9 * a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l329_32981


namespace NUMINAMATH_CALUDE_secret_eggs_count_l329_32904

/-- Given a jar with candy and secret eggs, calculate the number of secret eggs. -/
theorem secret_eggs_count (candy : ℝ) (total : ℕ) (h1 : candy = 3409.0) (h2 : total = 3554) :
  ↑total - candy = 145 :=
by sorry

end NUMINAMATH_CALUDE_secret_eggs_count_l329_32904


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l329_32935

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = 1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 5)

-- Define what it means for a circle to be tangent to the y-axis
def tangent_to_y_axis (equation : (ℝ → ℝ → Prop)) : Prop :=
  ∃ y : ℝ, equation 0 y ∧ ∀ x ≠ 0, ¬equation x y

-- Theorem statement
theorem circle_tangent_to_y_axis :
  tangent_to_y_axis circle_equation ∧
  ∀ x y : ℝ, circle_equation x y → (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l329_32935


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l329_32924

theorem algebraic_expression_value (a b : ℝ) : 
  (a * 2^3 + b * 2 - 7 = -19) → 
  (a * (-2)^3 + b * (-2) - 7 = 5) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l329_32924


namespace NUMINAMATH_CALUDE_rectangle_max_area_l329_32931

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (2 * x + 2 * y = 60) → x * y ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l329_32931


namespace NUMINAMATH_CALUDE_point_on_intersection_line_l329_32902

-- Define the sets and points
variable (α β m n l : Set Point)
variable (P : Point)

-- State the theorem
theorem point_on_intersection_line
  (h1 : α ∩ β = l)
  (h2 : m ⊆ α)
  (h3 : n ⊆ β)
  (h4 : m ∩ n = {P}) :
  P ∈ l := by
sorry

end NUMINAMATH_CALUDE_point_on_intersection_line_l329_32902


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l329_32956

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem unique_three_digit_number :
  ∃! n : ℕ, is_three_digit n ∧
            units_digit n = 4 ∧
            hundreds_digit n = 5 ∧
            tens_digit n % 2 = 0 ∧
            n % 8 = 0 ∧
            n = 544 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l329_32956


namespace NUMINAMATH_CALUDE_radish_basket_difference_l329_32963

theorem radish_basket_difference (total : ℕ) (first_basket : ℕ) : 
  total = 88 → first_basket = 37 → total - first_basket - first_basket = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_radish_basket_difference_l329_32963


namespace NUMINAMATH_CALUDE_jack_needs_five_rocks_l329_32907

-- Define the weights and rock weight
def jack_weight : ℕ := 60
def anna_weight : ℕ := 40
def rock_weight : ℕ := 4

-- Define the function to calculate the number of rocks
def num_rocks (jack_w anna_w rock_w : ℕ) : ℕ :=
  (jack_w - anna_w) / rock_w

-- Theorem statement
theorem jack_needs_five_rocks :
  num_rocks jack_weight anna_weight rock_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_jack_needs_five_rocks_l329_32907


namespace NUMINAMATH_CALUDE_log_equation_implies_y_value_l329_32944

-- Define a positive real number type for the base of logarithms
def PositiveReal := {x : ℝ | x > 0}

-- Define the logarithm function
noncomputable def log (base : PositiveReal) (x : PositiveReal) : ℝ := Real.log x / Real.log base.val

-- The main theorem
theorem log_equation_implies_y_value 
  (a b c x : PositiveReal) 
  (p q r y : ℝ) 
  (base : PositiveReal)
  (h1 : log base a / p = log base b / q)
  (h2 : log base b / q = log base c / r)
  (h3 : log base c / r = log base x)
  (h4 : x.val ≠ 1)
  (h5 : b.val^2 / (a.val * c.val) = x.val^y) :
  y = 2*q - p - r := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_y_value_l329_32944


namespace NUMINAMATH_CALUDE_triangle_sequence_properties_l329_32998

/-- Isosceles triangle with perimeter 2s -/
structure IsoscelesTriangle (s : ℝ) :=
  (base : ℝ)
  (leg : ℝ)
  (perimeter_eq : base + 2 * leg = 2 * s)
  (isosceles : leg ≥ base / 2)

/-- Sequence of isosceles triangles -/
def triangle_sequence (s : ℝ) : ℕ → IsoscelesTriangle s
| 0 => ⟨2, 49, sorry, sorry⟩
| (n + 1) => ⟨(triangle_sequence s n).leg, sorry, sorry, sorry⟩

/-- Angle between the legs of triangle i -/
def angle (s : ℝ) (i : ℕ) : ℝ := sorry

theorem triangle_sequence_properties (s : ℝ) :
  (∀ j : ℕ, angle s (2 * j) < angle s (2 * (j + 1))) ∧
  (∀ j : ℕ, angle s (2 * j + 1) > angle s (2 * (j + 1) + 1)) ∧
  (abs (angle s 11 - Real.pi / 3) < Real.pi / 180) :=
sorry

end NUMINAMATH_CALUDE_triangle_sequence_properties_l329_32998


namespace NUMINAMATH_CALUDE_leading_zeros_in_decimal_representation_l329_32949

theorem leading_zeros_in_decimal_representation (n : ℕ) (m : ℕ) :
  (∃ k : ℕ, (1 : ℚ) / (2^7 * 5^3) = (k : ℚ) / 10^n ∧ 
   k ≠ 0 ∧ k < 10^m) → n - m = 5 := by
  sorry

end NUMINAMATH_CALUDE_leading_zeros_in_decimal_representation_l329_32949


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l329_32923

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A number is divisible by both 3 and 5 if it's divisible by 15. -/
def divisible_by_3_and_5 (n : ℕ) : Prop := n % 15 = 0

theorem smallest_perfect_square_divisible_by_3_and_5 :
  (∀ n : ℕ, n > 0 → is_perfect_square n → divisible_by_3_and_5 n → n ≥ 225) ∧
  (is_perfect_square 225 ∧ divisible_by_3_and_5 225) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l329_32923


namespace NUMINAMATH_CALUDE_grunters_win_probability_l329_32964

/-- The number of games played -/
def n : ℕ := 6

/-- The number of games to be won -/
def k : ℕ := 5

/-- The probability of winning a single game -/
def p : ℚ := 4/5

/-- The probability of winning exactly k out of n games -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem grunters_win_probability :
  binomial_probability n k p = 6144/15625 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l329_32964


namespace NUMINAMATH_CALUDE_nadia_hannah_walk_l329_32974

/-- The total distance walked by Nadia and Hannah -/
def total_distance (nadia_distance : ℝ) (hannah_distance : ℝ) : ℝ :=
  nadia_distance + hannah_distance

/-- Theorem: Given Nadia walked 18 km and twice as far as Hannah, their total distance is 27 km -/
theorem nadia_hannah_walk :
  let nadia_distance : ℝ := 18
  let hannah_distance : ℝ := nadia_distance / 2
  total_distance nadia_distance hannah_distance = 27 := by
sorry

end NUMINAMATH_CALUDE_nadia_hannah_walk_l329_32974


namespace NUMINAMATH_CALUDE_giannas_savings_l329_32911

/-- Gianna's savings calculation --/
theorem giannas_savings (daily_savings : ℕ) (days_in_year : ℕ) (total_savings : ℕ) :
  daily_savings = 39 →
  days_in_year = 365 →
  total_savings = daily_savings * days_in_year →
  total_savings = 14235 := by
  sorry

end NUMINAMATH_CALUDE_giannas_savings_l329_32911


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l329_32901

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number z -/
def z : ℂ := i * (2 - i)

/-- A complex number is in the first quadrant if its real part is positive and its imaginary part is positive -/
def is_in_first_quadrant (c : ℂ) : Prop := 0 < c.re ∧ 0 < c.im

/-- Theorem: z is in the first quadrant -/
theorem z_in_first_quadrant : is_in_first_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l329_32901


namespace NUMINAMATH_CALUDE_P_in_second_quadrant_l329_32918

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The given point P -/
def P : Point :=
  { x := -2, y := 3 }

/-- Theorem: Point P is in the second quadrant -/
theorem P_in_second_quadrant : secondQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_second_quadrant_l329_32918


namespace NUMINAMATH_CALUDE_min_value_theorem_l329_32975

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 2*y)) + (y / x) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l329_32975


namespace NUMINAMATH_CALUDE_knitting_rate_theorem_l329_32941

/-- The number of days it takes A to knit a pair of socks -/
def A_days : ℝ := 3

/-- The number of days it takes A and B together to knit two pairs of socks -/
def AB_days : ℝ := 4

/-- The number of days it takes B to knit a pair of socks -/
def B_days : ℝ := 6

/-- Theorem stating that given A's knitting rate and the combined rate of A and B,
    B's individual knitting rate can be determined -/
theorem knitting_rate_theorem :
  (1 / A_days + 1 / B_days) * AB_days = 2 :=
sorry

end NUMINAMATH_CALUDE_knitting_rate_theorem_l329_32941


namespace NUMINAMATH_CALUDE_total_blocks_l329_32910

theorem total_blocks (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 18)
  (h2 : yellow = red + 7)
  (h3 : blue = red + 14) :
  red + yellow + blue = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_l329_32910


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l329_32930

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (0 : ℝ) (1/2), 4^x + x - a ≤ 3/2) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l329_32930


namespace NUMINAMATH_CALUDE_sum_edges_vertices_faces_l329_32984

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

/-- The sum of edges, vertices, and faces in a rectangular prism is 26 -/
theorem sum_edges_vertices_faces (p : RectangularPrism) :
  num_edges p + num_vertices p + num_faces p = 26 := by
  sorry

#check sum_edges_vertices_faces

end NUMINAMATH_CALUDE_sum_edges_vertices_faces_l329_32984


namespace NUMINAMATH_CALUDE_sum_of_inverse_G_power_three_l329_32915

def G : ℕ → ℚ
  | 0 => 0
  | 1 => 8/3
  | (n+2) => 3 * G (n+1) - (1/2) * G n

theorem sum_of_inverse_G_power_three : ∑' n, 1 / G (3^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_inverse_G_power_three_l329_32915


namespace NUMINAMATH_CALUDE_perfect_square_condition_l329_32980

theorem perfect_square_condition (Z K : ℤ) : 
  (50 < Z ∧ Z < 5000) →
  K > 1 →
  Z = K * K^2 →
  (∃ n : ℤ, Z = n^2) ↔ (K = 4 ∨ K = 9 ∨ K = 16) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l329_32980


namespace NUMINAMATH_CALUDE_translate_quadratic_l329_32945

/-- Represents a quadratic function of the form y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal and vertical translation to a quadratic function -/
def translate (f : QuadraticFunction) (h : ℝ) (v : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := -2 * f.a * h + f.b
  , c := f.a * h^2 - f.b * h + f.c + v }

theorem translate_quadratic :
  let f : QuadraticFunction := { a := 2, b := 0, c := 0 }
  let g : QuadraticFunction := translate f (-1) 3
  g = { a := 2, b := 4, c := 5 } :=
by sorry

end NUMINAMATH_CALUDE_translate_quadratic_l329_32945


namespace NUMINAMATH_CALUDE_remainder_17_power_100_mod_7_l329_32959

theorem remainder_17_power_100_mod_7 : 17^100 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_100_mod_7_l329_32959


namespace NUMINAMATH_CALUDE_intersection_slope_l329_32994

/-- Given two lines that intersect at a point, find the slope of one line -/
theorem intersection_slope (k : ℝ) : 
  (∃ (x y : ℝ), y = 2*x + 3 ∧ y = k*x + 4 ∧ x = 1 ∧ y = 5) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l329_32994


namespace NUMINAMATH_CALUDE_least_y_solution_l329_32943

-- Define the function we're trying to solve
def f (y : ℝ) := y + y^2

-- State the theorem
theorem least_y_solution :
  ∃ y : ℝ, y > 2 ∧ f y = 360 ∧ ∃ ε > 0, |y - 18.79| < ε :=
sorry

end NUMINAMATH_CALUDE_least_y_solution_l329_32943


namespace NUMINAMATH_CALUDE_triangle_height_ratio_l329_32990

theorem triangle_height_ratio (a b c h₁ h₂ h₃ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧
  (a : ℝ) / 3 = (b : ℝ) / 4 ∧ (b : ℝ) / 4 = (c : ℝ) / 5 ∧
  a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ →
  (h₁ : ℝ) / 20 = (h₂ : ℝ) / 15 ∧ (h₂ : ℝ) / 15 = (h₃ : ℝ) / 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_ratio_l329_32990


namespace NUMINAMATH_CALUDE_set_union_condition_implies_m_geq_two_l329_32905

theorem set_union_condition_implies_m_geq_two (m : ℝ) :
  let A : Set ℝ := {x | x ≥ 2}
  let B : Set ℝ := {x | x ≥ m}
  A ∪ B = A → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_set_union_condition_implies_m_geq_two_l329_32905


namespace NUMINAMATH_CALUDE_factorization_identity_l329_32928

theorem factorization_identity (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l329_32928


namespace NUMINAMATH_CALUDE_pauls_vertical_distance_l329_32972

/-- The total vertical distance traveled by Paul in a week -/
def total_vertical_distance (floor : ℕ) (trips_per_day : ℕ) (days : ℕ) (story_height : ℕ) : ℕ :=
  floor * story_height * trips_per_day * 2 * days

/-- Theorem stating the total vertical distance Paul travels in a week -/
theorem pauls_vertical_distance :
  total_vertical_distance 5 3 7 10 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_pauls_vertical_distance_l329_32972


namespace NUMINAMATH_CALUDE_garland_arrangements_correct_l329_32934

/-- The number of ways to arrange 6 blue, 7 red, and 9 white light bulbs in a garland,
    such that no two white light bulbs are consecutive -/
def garland_arrangements : ℕ :=
  Nat.choose 13 6 * Nat.choose 14 9

/-- Theorem stating that the number of garland arrangements is correct -/
theorem garland_arrangements_correct :
  garland_arrangements = 3435432 := by sorry

end NUMINAMATH_CALUDE_garland_arrangements_correct_l329_32934


namespace NUMINAMATH_CALUDE_correct_time_exists_l329_32939

/-- Represents the position of a watch hand on the face of the watch -/
def HandPosition := ℝ

/-- Represents the angle of rotation for the watch dial -/
def DialRotation := ℝ

/-- Represents a point in time within a 24-hour period -/
def TimePoint := ℝ

/-- A watch with fixed hour and minute hands -/
structure Watch where
  hourHand : HandPosition
  minuteHand : HandPosition

/-- Calculates the correct angle between hour and minute hands for a given time -/
noncomputable def correctAngle (t : TimePoint) : ℝ :=
  sorry

/-- Calculates the actual angle between hour and minute hands for a given watch and dial rotation -/
noncomputable def actualAngle (w : Watch) (r : DialRotation) : ℝ :=
  sorry

/-- States that for any watch with fixed hands, there exists a dial rotation
    such that the watch shows the correct time at least once in a 24-hour period -/
theorem correct_time_exists (w : Watch) :
  ∃ r : DialRotation, ∃ t : TimePoint, actualAngle w r = correctAngle t :=
sorry

end NUMINAMATH_CALUDE_correct_time_exists_l329_32939


namespace NUMINAMATH_CALUDE_honey_work_days_l329_32988

/-- Proves that Honey worked for 20 days given her daily earnings and total spent and saved amounts. -/
theorem honey_work_days (daily_earnings : ℕ) (total_spent : ℕ) (total_saved : ℕ) :
  daily_earnings = 80 →
  total_spent = 1360 →
  total_saved = 240 →
  (total_spent + total_saved) / daily_earnings = 20 :=
by sorry

end NUMINAMATH_CALUDE_honey_work_days_l329_32988


namespace NUMINAMATH_CALUDE_hikers_count_l329_32955

theorem hikers_count (total : ℕ) (difference : ℕ) (hikers bike_riders : ℕ) 
  (h1 : total = hikers + bike_riders)
  (h2 : hikers = bike_riders + difference)
  (h3 : total = 676)
  (h4 : difference = 178) :
  hikers = 427 := by
  sorry

end NUMINAMATH_CALUDE_hikers_count_l329_32955


namespace NUMINAMATH_CALUDE_exactly_two_in_favor_l329_32938

def probability_in_favor : ℝ := 0.6

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_in_favor :
  binomial_probability 4 2 probability_in_favor = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_in_favor_l329_32938


namespace NUMINAMATH_CALUDE_teresa_jogging_time_l329_32926

-- Define the constants
def distance : ℝ := 45  -- kilometers
def speed : ℝ := 7      -- kilometers per hour
def break_time : ℝ := 0.5  -- hours (30 minutes)

-- Define the theorem
theorem teresa_jogging_time :
  let jogging_time := distance / speed
  let total_time := jogging_time + break_time
  total_time = 6.93 :=
by
  sorry


end NUMINAMATH_CALUDE_teresa_jogging_time_l329_32926


namespace NUMINAMATH_CALUDE_root_product_sum_l329_32929

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (Real.sqrt 1008) * x₁^3 - 2016 * x₁^2 + 5 * x₁ + 2 = 0 ∧
  (Real.sqrt 1008) * x₂^3 - 2016 * x₂^2 + 5 * x₂ + 2 = 0 ∧
  (Real.sqrt 1008) * x₃^3 - 2016 * x₃^2 + 5 * x₃ + 2 = 0 →
  x₂ * (x₁ + x₃) = 1010 / 1008 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l329_32929


namespace NUMINAMATH_CALUDE_root_sum_cubes_l329_32979

-- Define the equation
def equation (x : ℝ) : Prop := (x - (8 : ℝ)^(1/3)) * (x - (27 : ℝ)^(1/3)) * (x - (64 : ℝ)^(1/3)) = 1

-- Define the roots
def roots (u v w : ℝ) : Prop := equation u ∧ equation v ∧ equation w ∧ u ≠ v ∧ u ≠ w ∧ v ≠ w

-- Theorem statement
theorem root_sum_cubes (u v w : ℝ) : roots u v w → u^3 + v^3 + w^3 = 102 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_cubes_l329_32979


namespace NUMINAMATH_CALUDE_estimate_student_population_l329_32936

theorem estimate_student_population (n : ℕ) 
  (h1 : n > 0) 
  (h2 : 80 ≤ n) 
  (h3 : 100 ≤ n) : 
  (80 : ℝ) / n * 100 = 20 → n = 400 := by
  sorry

end NUMINAMATH_CALUDE_estimate_student_population_l329_32936


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l329_32942

/-- Given a triangle ABC where:
    - The sides opposite to angles A, B, C are a, b, c respectively
    - A = π/3
    - a = √3
    - b = 1
    Prove that C = π/2, i.e., the triangle is a right triangle -/
theorem triangle_abc_is_right_triangle (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  A = π / 3 →
  a / Real.sin A = b / Real.sin B →
  A + B + C = π →
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l329_32942


namespace NUMINAMATH_CALUDE_part_one_part_two_l329_32950

/-- Given expressions for A and B -/
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1

/-- Theorem for part 1 -/
theorem part_one (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 := by sorry

/-- Theorem for part 2 -/
theorem part_two (b : ℝ) :
  (∀ a : ℝ, A a b + 2 * B a b = A 0 b + 2 * B 0 b) → b = 2/5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l329_32950


namespace NUMINAMATH_CALUDE_min_value_sum_sqrt_ratios_equality_condition_l329_32986

theorem min_value_sum_sqrt_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 / b^2) + Real.sqrt (b^2 / c^2) + Real.sqrt (c^2 / a^2) ≥ 3 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 / b^2) + Real.sqrt (b^2 / c^2) + Real.sqrt (c^2 / a^2) = 3 ↔ a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_sqrt_ratios_equality_condition_l329_32986


namespace NUMINAMATH_CALUDE_complementary_angles_difference_theorem_l329_32961

def complementary_angles_difference (a b : ℝ) : Prop :=
  a + b = 90 ∧ a / b = 5 / 3 → |a - b| = 22.5

theorem complementary_angles_difference_theorem :
  ∀ a b : ℝ, complementary_angles_difference a b :=
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_theorem_l329_32961


namespace NUMINAMATH_CALUDE_composition_equation_solution_l329_32957

/-- Given functions f and g, prove that if f(g(a)) = 4, then a = 3/4 -/
theorem composition_equation_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = (2*x - 1) / 3 + 2)
  (hg : ∀ x, g x = 5 - 2*x)
  (h : f (g a) = 4) : 
  a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l329_32957


namespace NUMINAMATH_CALUDE_not_right_triangle_sides_l329_32914

theorem not_right_triangle_sides (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 4) (h3 : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_sides_l329_32914


namespace NUMINAMATH_CALUDE_remainder_of_sum_l329_32919

theorem remainder_of_sum (d : ℕ) (h1 : 242 % d = 8) (h2 : 698 % d = 9) (h3 : d = 13) :
  (242 + 698) % d = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l329_32919


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l329_32966

theorem exactly_one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 30 - 6 * n > 18 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l329_32966


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l329_32913

theorem square_sum_equals_sixteen (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) :
  x^2 + y^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l329_32913


namespace NUMINAMATH_CALUDE_klinked_from_connectivity_and_edges_l329_32978

/-- A graph is k-linked if for any k pairs of vertices (s₁, t₁), ..., (sₖ, tₖ),
    there exist k vertex-disjoint paths P₁, ..., Pₖ such that Pᵢ connects sᵢ to tᵢ. -/
def IsKLinked (G : SimpleGraph α) (k : ℕ) : Prop := sorry

/-- A graph is k-connected if it remains connected after removing any k-1 vertices. -/
def IsKConnected (G : SimpleGraph α) (k : ℕ) : Prop := sorry

/-- The number of edges in a graph. -/
def NumEdges (G : SimpleGraph α) : ℕ := sorry

theorem klinked_from_connectivity_and_edges
  {α : Type*} (G : SimpleGraph α) (k : ℕ) :
  IsKConnected G (2 * k) →
  NumEdges G ≥ 8 * k →
  IsKLinked G k :=
sorry

end NUMINAMATH_CALUDE_klinked_from_connectivity_and_edges_l329_32978


namespace NUMINAMATH_CALUDE_parabola_and_tangent_line_l329_32993

/-- Parabola with vertex at origin and focus on positive y-axis -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_focus_pos : focus.2 > 0
  h_focus_eq : focus = (0, p)

/-- Line through focus intersecting parabola -/
structure IntersectingLine (para : Parabola) where
  a : ℝ × ℝ
  b : ℝ × ℝ
  h_on_parabola_a : a.1^2 = 2 * para.p * a.2
  h_on_parabola_b : b.1^2 = 2 * para.p * b.2
  h_through_focus : ∃ t : ℝ, (1 - t) • a + t • b = para.focus

/-- Line with y-intercept 6 intersecting parabola -/
structure TangentLine (para : Parabola) where
  m : ℝ
  p : ℝ × ℝ
  q : ℝ × ℝ
  r : ℝ × ℝ
  h_on_parabola_p : p.1^2 = 2 * para.p * p.2
  h_on_parabola_q : q.1^2 = 2 * para.p * q.2
  h_on_line_p : p.2 = m * p.1 + 6
  h_on_line_q : q.2 = m * q.1 + 6
  h_r_on_directrix : r.2 = -para.p
  h_qfr_collinear : ∃ t : ℝ, (1 - t) • q + t • r = para.focus
  h_pr_tangent : (p.2 - r.2) / (p.1 - r.1) = p.1 / (2 * para.p)

theorem parabola_and_tangent_line (para : Parabola) 
  (line : IntersectingLine para) 
  (tline : TangentLine para) :
  (∀ (t : ℝ), (1 - t) • line.a + t • line.b - (0, 3) = (0, 1)) →
  (‖line.a - line.b‖ = 8) →
  (para.p = 2 ∧ (tline.m = 1/2 ∨ tline.m = -1/2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_tangent_line_l329_32993


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l329_32909

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 9 ∣ n ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ 3 ∣ m ∧ 4 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  n = 108 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l329_32909


namespace NUMINAMATH_CALUDE_percentage_relation_l329_32983

theorem percentage_relation (x y z : ℝ) :
  y = 0.3 * z →
  x = 0.36 * z →
  x = y * 1.2 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l329_32983


namespace NUMINAMATH_CALUDE_unique_real_root_l329_32987

theorem unique_real_root : 
  (∃ x : ℝ, x^2 + 3 = 0) = false ∧ 
  (∃ x : ℝ, x^3 + 3 = 0) = true ∧ 
  (∃ x : ℝ, |1 / (x^2 - 3)| = 0) = false ∧ 
  (∃ x : ℝ, |x| + 3 = 0) = false :=
by sorry

end NUMINAMATH_CALUDE_unique_real_root_l329_32987


namespace NUMINAMATH_CALUDE_tan_product_30_degrees_l329_32927

theorem tan_product_30_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_30_degrees_l329_32927


namespace NUMINAMATH_CALUDE_opposite_sign_expression_value_l329_32967

theorem opposite_sign_expression_value (a b : ℝ) :
  (|a + 2| = 0 ∧ (b - 5/2)^2 = 0) →
  (2*a + 3*b) * (2*b - 3*a) = 26 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_expression_value_l329_32967


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l329_32977

/-- Calculates the time spent shopping, performing tasks, and traveling between sections --/
theorem shopping_time_calculation (total_trip_time waiting_times break_time browsing_times walking_time_per_trip num_sections : ℕ) :
  total_trip_time = 165 ∧
  waiting_times = 5 + 10 + 8 + 15 + 20 ∧
  break_time = 10 ∧
  browsing_times = 12 + 7 + 10 ∧
  walking_time_per_trip = 2 ∧ -- Rounded up from 1.5
  num_sections = 8 →
  total_trip_time - (waiting_times + break_time + browsing_times + walking_time_per_trip * (num_sections - 1)) = 86 :=
by sorry

end NUMINAMATH_CALUDE_shopping_time_calculation_l329_32977


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l329_32995

theorem ceiling_floor_calculation : ⌈(15 / 8) * (-35 / 4)⌉ - ⌊(15 / 8) * ⌊(-35 / 4) + (1 / 4)⌋⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l329_32995


namespace NUMINAMATH_CALUDE_second_number_value_l329_32969

theorem second_number_value (A B C : ℚ) : 
  A + B + C = 98 → 
  A / B = 2 / 3 → 
  B / C = 5 / 8 → 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l329_32969


namespace NUMINAMATH_CALUDE_probability_sum_less_than_ten_l329_32917

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) := Finset.product (Finset.range sides) (Finset.range sides)

/-- The favorable outcomes (sum less than 10) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 < 10)

/-- The probability of the sum being less than 10 when rolling two fair six-sided dice -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_sum_less_than_ten : probability = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_ten_l329_32917


namespace NUMINAMATH_CALUDE_congruence_intercepts_sum_l329_32996

theorem congruence_intercepts_sum (x₀ y₀ : ℕ) : 
  (0 ≤ x₀ ∧ x₀ < 40) → 
  (0 ≤ y₀ ∧ y₀ < 40) → 
  (5 * x₀ ≡ -2 [ZMOD 40]) → 
  (3 * y₀ ≡ 2 [ZMOD 40]) → 
  x₀ + y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_intercepts_sum_l329_32996


namespace NUMINAMATH_CALUDE_homework_situations_l329_32952

/-- The number of teachers who have assigned homework -/
def num_teachers : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of possible homework situations for all students -/
def total_situations : ℕ := num_teachers ^ num_students

theorem homework_situations :
  total_situations = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_homework_situations_l329_32952


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l329_32922

theorem reciprocal_of_negative_2022 : ((-2022)⁻¹ : ℚ) = -1 / 2022 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l329_32922


namespace NUMINAMATH_CALUDE_decoration_sets_count_l329_32908

/-- Represents a decoration set with balloons and ribbons -/
structure DecorationSet where
  balloons : ℕ
  ribbons : ℕ

/-- The cost of a decoration set -/
def cost (set : DecorationSet) : ℕ := 4 * set.balloons + 6 * set.ribbons

/-- Predicate for valid decoration sets -/
def isValid (set : DecorationSet) : Prop :=
  cost set = 120 ∧ Even set.balloons

theorem decoration_sets_count :
  ∃! (sets : Finset DecorationSet), 
    (∀ s ∈ sets, isValid s) ∧ 
    (∀ s, isValid s → s ∈ sets) ∧
    Finset.card sets = 2 := by
  sorry

end NUMINAMATH_CALUDE_decoration_sets_count_l329_32908


namespace NUMINAMATH_CALUDE_samara_friends_average_alligators_l329_32906

/-- Given a group of people searching for alligators, calculate the average number
    of alligators seen by friends, given the total number seen, the number seen by
    one person, and the number of friends. -/
def average_alligators_seen_by_friends 
  (total_alligators : ℕ) 
  (alligators_seen_by_one : ℕ) 
  (num_friends : ℕ) : ℚ :=
  (total_alligators - alligators_seen_by_one) / num_friends

/-- Prove that given the specific values from the problem, 
    the average number of alligators seen by each friend is 10. -/
theorem samara_friends_average_alligators :
  average_alligators_seen_by_friends 50 20 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_samara_friends_average_alligators_l329_32906


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l329_32970

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The GDP value of Anning City in the first quarter of 2023 -/
def gdp_value : ℕ := 17580000000

/-- The scientific notation representation of the GDP value -/
def gdp_scientific : ScientificNotation :=
  { coefficient := 1.758
    exponent := 10
    is_valid := by sorry }

/-- Theorem stating that the GDP value is correctly represented in scientific notation -/
theorem gdp_scientific_notation_correct :
  (gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent) = gdp_value := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l329_32970


namespace NUMINAMATH_CALUDE_complex_multiplication_l329_32985

theorem complex_multiplication (i : ℂ) : i * i = -1 → (2 + 3*i) * (3 - 2*i) = 12 + 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l329_32985


namespace NUMINAMATH_CALUDE_shopping_spree_remaining_amount_l329_32991

def initial_amount : ℝ := 78

def kite_price_euro : ℝ := 6
def euro_to_usd : ℝ := 1.2

def frisbee_price_pound : ℝ := 7
def pound_to_usd : ℝ := 1.4

def roller_skates_price : ℝ := 15
def roller_skates_discount : ℝ := 0.125

def lego_set_price : ℝ := 25
def lego_set_discount : ℝ := 0.15

def puzzle_price : ℝ := 12
def puzzle_tax : ℝ := 0.075

def remaining_amount : ℝ := initial_amount - 
  (kite_price_euro * euro_to_usd +
   frisbee_price_pound * pound_to_usd +
   roller_skates_price * (1 - roller_skates_discount) +
   lego_set_price * (1 - lego_set_discount) +
   puzzle_price * (1 + puzzle_tax))

theorem shopping_spree_remaining_amount : 
  remaining_amount = 13.725 := by sorry

end NUMINAMATH_CALUDE_shopping_spree_remaining_amount_l329_32991


namespace NUMINAMATH_CALUDE_complex_absolute_value_l329_32912

theorem complex_absolute_value (ω : ℂ) (h : ω = 7 + 4 * Complex.I) :
  Complex.abs (ω^2 + 10*ω + 88) = Real.sqrt 313 * 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l329_32912


namespace NUMINAMATH_CALUDE_complement_union_theorem_l329_32951

def I : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1, 2}
def B : Set Nat := {2, 3}

theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l329_32951


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l329_32903

theorem arithmetic_calculations : 
  ((-8) + 10 - 2 + (-1) = -1) ∧ 
  (12 - 7 * (-4) + 8 / (-2) = 36) ∧ 
  ((1/2 + 1/3 - 1/6) / (-1/18) = -12) ∧ 
  (-1^4 - (1 + 0.5) * (1/3) * (-4)^2 = -33/32) := by
  sorry

#eval (-8) + 10 - 2 + (-1)
#eval 12 - 7 * (-4) + 8 / (-2)
#eval (1/2 + 1/3 - 1/6) / (-1/18)
#eval -1^4 - (1 + 0.5) * (1/3) * (-4)^2

end NUMINAMATH_CALUDE_arithmetic_calculations_l329_32903


namespace NUMINAMATH_CALUDE_max_value_of_expression_l329_32989

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 2 → x + y^3 + z^4 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l329_32989


namespace NUMINAMATH_CALUDE_machine_does_not_require_repair_no_repair_needed_l329_32925

/-- Represents the nominal portion weight in grams -/
def nominal_weight : ℝ := 390

/-- Represents the greatest deviation from the mean among preserved measurements in grams -/
def max_deviation : ℝ := 39

/-- Represents the threshold for requiring repair in grams -/
def repair_threshold : ℝ := 39

/-- Condition: The greatest deviation does not exceed 10% of the nominal weight -/
axiom max_deviation_condition : max_deviation ≤ 0.1 * nominal_weight

/-- Condition: All deviations are no more than the maximum deviation -/
axiom all_deviations_bounded (deviation : ℝ) : deviation ≤ max_deviation

/-- Condition: The standard deviation does not exceed the greatest deviation -/
axiom standard_deviation_bounded (σ : ℝ) : σ ≤ max_deviation

/-- Theorem: The standard deviation is no more than the repair threshold -/
theorem machine_does_not_require_repair (σ : ℝ) : 
  σ ≤ repair_threshold :=
sorry

/-- Corollary: The machine does not require repair -/
theorem no_repair_needed : 
  ∃ (σ : ℝ), σ ≤ repair_threshold :=
sorry

end NUMINAMATH_CALUDE_machine_does_not_require_repair_no_repair_needed_l329_32925


namespace NUMINAMATH_CALUDE_mixture_composition_l329_32916

-- Define the initial mixture
def initial_mixture : ℝ := 90

-- Define the initial milk to water ratio
def milk_water_ratio : ℚ := 2 / 1

-- Define the amount of water evaporated
def water_evaporated : ℝ := 10

-- Define the relation between liquid L and milk
def liquid_L_milk_ratio : ℚ := 1 / 3

-- Define the relation between milk and water after additions
def final_milk_water_ratio : ℚ := 2 / 1

-- Theorem to prove
theorem mixture_composition :
  let initial_milk := initial_mixture * (milk_water_ratio / (1 + milk_water_ratio))
  let initial_water := initial_mixture * (1 / (1 + milk_water_ratio))
  let remaining_water := initial_water - water_evaporated
  let liquid_L := initial_milk * liquid_L_milk_ratio
  let final_milk := initial_milk
  let final_water := remaining_water
  (liquid_L = 20) ∧ (final_milk / final_water = 3 / 1) := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l329_32916


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l329_32933

/-- Given a right triangle with legs x and y, if rotating about one leg produces a cone of volume 1000π cm³
    and rotating about the other leg produces a cone of volume 2250π cm³, 
    then the hypotenuse is approximately 39.08 cm. -/
theorem right_triangle_hypotenuse (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) :
  (1/3 * π * y^2 * x = 1000 * π) →
  (1/3 * π * x^2 * y = 2250 * π) →
  abs (Real.sqrt (x^2 + y^2) - 39.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l329_32933


namespace NUMINAMATH_CALUDE_division_problem_l329_32982

/-- Given the conditions of the division problem, prove the values of the divisors -/
theorem division_problem (D₁ D₂ : ℕ) : 
  1526 = 34 * D₁ + 18 → 
  34 * D₂ + 52 = 421 → 
  D₁ = 44 ∧ D₂ = 11 := by
  sorry

#check division_problem

end NUMINAMATH_CALUDE_division_problem_l329_32982


namespace NUMINAMATH_CALUDE_octal_131_equals_binary_1011001_l329_32940

-- Define octal_to_decimal function
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

-- Define decimal_to_binary function
def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

-- Theorem statement
theorem octal_131_equals_binary_1011001 :
  decimal_to_binary (octal_to_decimal 131) = [1, 0, 1, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_octal_131_equals_binary_1011001_l329_32940


namespace NUMINAMATH_CALUDE_mother_twice_bob_age_year_l329_32965

def bob_age_2010 : ℕ := 10
def mother_age_2010 : ℕ := 5 * bob_age_2010

def year_mother_twice_bob_age : ℕ :=
  2010 + (mother_age_2010 - 2 * bob_age_2010)

theorem mother_twice_bob_age_year :
  year_mother_twice_bob_age = 2040 := by
  sorry

end NUMINAMATH_CALUDE_mother_twice_bob_age_year_l329_32965


namespace NUMINAMATH_CALUDE_photographer_application_choices_l329_32920

theorem photographer_application_choices :
  let n : ℕ := 5  -- Total number of pre-selected photos
  let k₁ : ℕ := 3 -- First option for number of photos to include
  let k₂ : ℕ := 4 -- Second option for number of photos to include
  (Nat.choose n k₁) + (Nat.choose n k₂) = 15 := by
  sorry

end NUMINAMATH_CALUDE_photographer_application_choices_l329_32920


namespace NUMINAMATH_CALUDE_books_together_l329_32968

/-- The number of books Tim and Mike have together -/
def total_books (tim_books mike_books : ℕ) : ℕ := tim_books + mike_books

/-- Theorem: Tim and Mike have 42 books together -/
theorem books_together : total_books 22 20 = 42 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l329_32968
