import Mathlib

namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1292_129281

/-- Given a circle with equation x^2 + y^2 + 4x - 2y - 4 = 0, 
    its center coordinates are (-2, 1) and its radius is 3 -/
theorem circle_center_and_radius : 
  ∃ (x y : ℝ), x^2 + y^2 + 4*x - 2*y - 4 = 0 → 
  ∃ (h k r : ℝ), h = -2 ∧ k = 1 ∧ r = 3 ∧
  ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1292_129281


namespace NUMINAMATH_CALUDE_rice_mixture_price_l1292_129256

/-- Given two types of rice with different weights and prices, 
    prove that the price of the second type can be determined 
    from the average price of the mixture. -/
theorem rice_mixture_price 
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (avg_price : ℝ)
  (h1 : weight1 = 8)
  (h2 : price1 = 16)
  (h3 : weight2 = 4)
  (h4 : avg_price = 18)
  (h5 : (weight1 * price1 + weight2 * price2) / (weight1 + weight2) = avg_price) :
  price2 = 22 := by
sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l1292_129256


namespace NUMINAMATH_CALUDE_unique_number_doubled_plus_thirteen_l1292_129297

theorem unique_number_doubled_plus_thirteen : ∃! x : ℝ, 2 * x + 13 = 89 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_doubled_plus_thirteen_l1292_129297


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1292_129299

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * (60 * 60)

theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 2 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1292_129299


namespace NUMINAMATH_CALUDE_max_cube_volume_in_tetrahedron_l1292_129240

/-- Regular tetrahedron with edge length 2 -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq : edge_length = 2

/-- Cube placed inside the tetrahedron -/
structure InsideCube where
  side_length : ℝ
  bottom_face_parallel : Prop
  top_vertices_touch : Prop

/-- The maximum volume of the cube inside the tetrahedron -/
def max_cube_volume (t : RegularTetrahedron) (c : InsideCube) : ℝ :=
  c.side_length ^ 3

/-- Theorem stating the maximum volume of the cube -/
theorem max_cube_volume_in_tetrahedron (t : RegularTetrahedron) (c : InsideCube) :
  max_cube_volume t c = 8 * Real.sqrt 3 / 243 :=
sorry

end NUMINAMATH_CALUDE_max_cube_volume_in_tetrahedron_l1292_129240


namespace NUMINAMATH_CALUDE_percentage_problem_l1292_129251

theorem percentage_problem (x : ℝ) :
  (0.15 * 0.30 * 0.50 * x = 117) → (x = 5200) :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1292_129251


namespace NUMINAMATH_CALUDE_range_of_m_l1292_129261

theorem range_of_m (m : ℝ) : 
  (∃ x₀ ∈ Set.Icc 1 2, x₀^2 - m*x₀ + 4 > 0) ↔ m < 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1292_129261


namespace NUMINAMATH_CALUDE_total_clothes_washed_l1292_129203

/-- Represents the number of clothes a person has -/
structure ClothesCount where
  whiteShirts : ℕ
  coloredShirts : ℕ
  shorts : ℕ
  pants : ℕ

/-- Calculates the total number of clothes for a person -/
def totalClothes (c : ClothesCount) : ℕ :=
  c.whiteShirts + c.coloredShirts + c.shorts + c.pants

/-- Cally's clothes count -/
def cally : ClothesCount :=
  { whiteShirts := 10
    coloredShirts := 5
    shorts := 7
    pants := 6 }

/-- Danny's clothes count -/
def danny : ClothesCount :=
  { whiteShirts := 6
    coloredShirts := 8
    shorts := 10
    pants := 6 }

/-- Theorem stating that the total number of clothes washed by Cally and Danny is 58 -/
theorem total_clothes_washed : totalClothes cally + totalClothes danny = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_clothes_washed_l1292_129203


namespace NUMINAMATH_CALUDE_complex_exponential_identity_l1292_129270

theorem complex_exponential_identity (n : ℕ) (hn : n > 0 ∧ n ≤ 500) (t : ℝ) :
  (Complex.exp (Complex.I * t))^n = Complex.exp (Complex.I * (n * t)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_identity_l1292_129270


namespace NUMINAMATH_CALUDE_solution_set_l1292_129275

def satisfies_equations (x y : ℝ) : Prop :=
  y^2 - y*x^2 = 0 ∧ x^5 + x^4 = 0

theorem solution_set :
  ∀ x y : ℝ, satisfies_equations x y ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l1292_129275


namespace NUMINAMATH_CALUDE_parallel_lines_circle_chord_l1292_129215

/-- Given three equally spaced parallel lines intersecting a circle, creating chords of lengths 38, 38, and 34, the distance between two adjacent parallel lines is 6. -/
theorem parallel_lines_circle_chord (r : ℝ) : 
  let chord1 : ℝ := 38
  let chord2 : ℝ := 38
  let chord3 : ℝ := 34
  let d : ℝ := 6
  38 * r^2 = 722 + (19/4) * d^2 ∧ 
  34 * r^2 = 578 + (153/4) * d^2 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_chord_l1292_129215


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l1292_129233

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l1292_129233


namespace NUMINAMATH_CALUDE_set_A_definition_l1292_129242

def U : Set ℝ := {x | x > 1}

theorem set_A_definition (A : Set ℝ) (h1 : A ⊆ U) (h2 : (U \ A) = {x | x > 9}) : 
  A = {x | 1 < x ∧ x ≤ 9} := by
  sorry

end NUMINAMATH_CALUDE_set_A_definition_l1292_129242


namespace NUMINAMATH_CALUDE_no_solution_exists_l1292_129214

theorem no_solution_exists : ¬ ∃ (a b : ℝ), a^2 + 3*b^2 + 2 = 3*a*b := by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1292_129214


namespace NUMINAMATH_CALUDE_paige_recycled_amount_l1292_129220

/-- The number of pounds recycled per point earned -/
def pounds_per_point : ℕ := 4

/-- The number of pounds recycled by Paige's friends -/
def friends_recycled : ℕ := 2

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_recycled : ℕ := 14

theorem paige_recycled_amount :
  paige_recycled = total_points * pounds_per_point - friends_recycled := by
  sorry

end NUMINAMATH_CALUDE_paige_recycled_amount_l1292_129220


namespace NUMINAMATH_CALUDE_andrew_final_stickers_l1292_129221

def total_stickers : ℕ := 1500
def ratio_sum : ℕ := 5

def initial_shares (i : Fin 3) : ℕ := 
  if i = 0 ∨ i = 1 then total_stickers / ratio_sum else 3 * (total_stickers / ratio_sum)

theorem andrew_final_stickers : 
  initial_shares 1 + (2/3 : ℚ) * initial_shares 2 = 900 := by sorry

end NUMINAMATH_CALUDE_andrew_final_stickers_l1292_129221


namespace NUMINAMATH_CALUDE_unsatisfactory_fraction_is_8_25_l1292_129286

/-- Represents the grades in a class -/
structure GradeDistribution where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  f : Nat

/-- The grade distribution for the given class -/
def classGrades : GradeDistribution :=
  { a := 6, b := 5, c := 4, d := 2, f := 8 }

/-- The total number of students in the class -/
def totalStudents (grades : GradeDistribution) : Nat :=
  grades.a + grades.b + grades.c + grades.d + grades.f

/-- The number of students with unsatisfactory grades -/
def unsatisfactoryGrades (grades : GradeDistribution) : Nat :=
  grades.f

/-- Theorem: The fraction of unsatisfactory grades is 8/25 -/
theorem unsatisfactory_fraction_is_8_25 :
  (unsatisfactoryGrades classGrades : Rat) / (totalStudents classGrades) = 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_unsatisfactory_fraction_is_8_25_l1292_129286


namespace NUMINAMATH_CALUDE_tangent_parabola_circle_l1292_129222

/-- Theorem: Tangent Line to Parabola Touching Circle -/
theorem tangent_parabola_circle (r : ℝ) (hr : r > 0) :
  ∃ (x y : ℝ),
    -- Point P(x, y) lies on the parabola
    y = (1/4) * x^2 ∧
    -- Point P(x, y) lies on the circle
    (x - 1)^2 + (y - 2)^2 = r^2 ∧
    -- The tangent line to the parabola at P touches the circle
    ∃ (m : ℝ),
      -- m is the slope of the tangent line to the parabola at P
      m = (1/2) * x ∧
      -- The tangent line touches the circle (perpendicular to radius)
      m * ((y - 2) / (x - 1)) = -1
  → r = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_parabola_circle_l1292_129222


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1292_129219

/-- A polynomial of the form x^2 + bx + c is a perfect square trinomial if and only if
    there exists a real number m such that b = 2m and c = m^2 -/
def is_perfect_square_trinomial (b c : ℝ) : Prop :=
  ∃ m : ℝ, b = 2 * m ∧ c = m^2

/-- The main theorem: x^2 + (a-1)x + 9 is a perfect square trinomial iff a = 7 or a = -5 -/
theorem perfect_square_condition (a : ℝ) :
  is_perfect_square_trinomial (a - 1) 9 ↔ a = 7 ∨ a = -5 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_condition_l1292_129219


namespace NUMINAMATH_CALUDE_tangent_line_at_2_max_value_on_interval_l1292_129209

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement for the tangent line
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ x y, y = f x → (x = 2 → y = m * (x - 2) + f 2) ∧
  (9 * x - y - 15 = 0 ↔ y = m * (x - 2) + f 2) :=
sorry

-- Statement for the maximum value
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_max_value_on_interval_l1292_129209


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1292_129246

theorem absolute_value_equation_solution : 
  ∃! x : ℝ, |x - 30| + |x - 25| = |2*x - 50| + 5 ∧ x = 32.5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1292_129246


namespace NUMINAMATH_CALUDE_initial_balloons_eq_sum_l1292_129259

/-- The number of green balloons Fred initially had -/
def initial_balloons : ℕ := sorry

/-- The number of green balloons Fred gave to Sandy -/
def balloons_given : ℕ := 221

/-- The number of green balloons Fred has left -/
def balloons_left : ℕ := 488

/-- Theorem stating that the initial number of balloons is equal to the sum of balloons given away and balloons left -/
theorem initial_balloons_eq_sum : initial_balloons = balloons_given + balloons_left := by sorry

end NUMINAMATH_CALUDE_initial_balloons_eq_sum_l1292_129259


namespace NUMINAMATH_CALUDE_total_students_in_line_l1292_129263

/-- The number of students in a line, given specific positions of Hoseok and Yoongi -/
def number_of_students (left_of_hoseok : ℕ) (between_hoseok_yoongi : ℕ) (right_of_yoongi : ℕ) : ℕ :=
  left_of_hoseok + 1 + between_hoseok_yoongi + 1 + right_of_yoongi

/-- Theorem stating that the total number of students in the line is 22 -/
theorem total_students_in_line : 
  number_of_students 9 5 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_students_in_line_l1292_129263


namespace NUMINAMATH_CALUDE_abs_g_one_equals_31_l1292_129278

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that the absolute value of g at specific points is 24 -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  |g (-1)| = 24 ∧ |g 0| = 24 ∧ |g 2| = 24 ∧ |g 4| = 24 ∧ |g 5| = 24 ∧ |g 8| = 24

/-- The main theorem -/
theorem abs_g_one_equals_31 (g : ThirdDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 1| = 31 := by
  sorry

end NUMINAMATH_CALUDE_abs_g_one_equals_31_l1292_129278


namespace NUMINAMATH_CALUDE_remainder_3m_mod_5_l1292_129204

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3m_mod_5_l1292_129204


namespace NUMINAMATH_CALUDE_reflect_P_x_axis_l1292_129231

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -2)

/-- Theorem: Reflecting P(3,-2) across the x-axis results in (3,2) -/
theorem reflect_P_x_axis : reflect_x P = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_x_axis_l1292_129231


namespace NUMINAMATH_CALUDE_expression_evaluates_to_one_l1292_129229

theorem expression_evaluates_to_one :
  (100^2 - 7^2) / (70^2 - 11^2) * ((70 - 11) * (70 + 11)) / ((100 - 7) * (100 + 7)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluates_to_one_l1292_129229


namespace NUMINAMATH_CALUDE_tie_cost_l1292_129216

theorem tie_cost (pants_cost shirt_cost paid change : ℕ) 
  (h1 : pants_cost = 140)
  (h2 : shirt_cost = 43)
  (h3 : paid = 200)
  (h4 : change = 2) :
  paid - change - (pants_cost + shirt_cost) = 15 := by
sorry

end NUMINAMATH_CALUDE_tie_cost_l1292_129216


namespace NUMINAMATH_CALUDE_tan_120_degrees_l1292_129247

theorem tan_120_degrees : Real.tan (120 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_120_degrees_l1292_129247


namespace NUMINAMATH_CALUDE_population_scientific_notation_l1292_129288

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_scientific_notation :
  toScientificNotation (141260 * 1000000) =
    ScientificNotation.mk 1.4126 5 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l1292_129288


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1292_129260

theorem square_area_from_diagonal (d : ℝ) (h : d = 7) : 
  (d^2 / 2) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1292_129260


namespace NUMINAMATH_CALUDE_spade_then_king_probability_l1292_129287

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of decks shuffled together -/
def num_decks : ℕ := 2

/-- The total number of cards after shuffling -/
def total_cards : ℕ := standard_deck_size * num_decks

/-- The number of spades in a standard deck -/
def spades_per_deck : ℕ := 13

/-- The number of kings in a standard deck -/
def kings_per_deck : ℕ := 4

/-- The probability of drawing a spade as the first card and a king as the second card -/
theorem spade_then_king_probability : 
  (spades_per_deck * num_decks) / total_cards * 
  (kings_per_deck * num_decks) / (total_cards - 1) = 103 / 5356 := by
  sorry

end NUMINAMATH_CALUDE_spade_then_king_probability_l1292_129287


namespace NUMINAMATH_CALUDE_rectangle_shading_l1292_129296

theorem rectangle_shading (total_rectangles : ℕ) 
  (h1 : total_rectangles = 12) : 
  (2 : ℚ) / 3 * (3 : ℚ) / 4 * total_rectangles = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shading_l1292_129296


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1292_129262

/-- Represents a triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- Checks if the given EvenTriangle satisfies the triangle inequality -/
def is_valid (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of a valid EvenTriangle is 18 -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), is_valid t ∧ perimeter t = 18 ∧
  ∀ (t' : EvenTriangle), is_valid t' → perimeter t' ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1292_129262


namespace NUMINAMATH_CALUDE_a5_equals_6_l1292_129276

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

/-- The theorem stating that a5 = 6 in the given arithmetic sequence -/
theorem a5_equals_6 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a5_equals_6_l1292_129276


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1292_129284

theorem no_positive_integer_solution :
  ¬ ∃ (n : ℕ+) (p : ℕ), Nat.Prime p ∧ n.val^2 - 45*n.val + 520 = p := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1292_129284


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l1292_129285

theorem polynomial_roots_product (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℤ, x^3 + a*x^2 + b*x + 6*a = (x - r)^2 * (x - s)) ∧ 
   r ≠ s) → 
  |a * b| = 546 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l1292_129285


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_7560_l1292_129237

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_7560 :
  largest_perfect_square_factor 7560 = 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_7560_l1292_129237


namespace NUMINAMATH_CALUDE_inequality_proof_l1292_129206

theorem inequality_proof (a b c : ℝ) 
  (ha : a = (1/6) * Real.log 8)
  (hb : b = (1/2) * Real.log 5)
  (hc : c = Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1292_129206


namespace NUMINAMATH_CALUDE_toy_store_problem_l1292_129248

/-- Toy Store Problem -/
theorem toy_store_problem 
  (first_batch_cost second_batch_cost : ℝ)
  (quantity_ratio : ℝ)
  (cost_increase : ℝ)
  (min_profit : ℝ)
  (h1 : first_batch_cost = 2500)
  (h2 : second_batch_cost = 4500)
  (h3 : quantity_ratio = 1.5)
  (h4 : cost_increase = 10)
  (h5 : min_profit = 1750) :
  ∃ (first_batch_cost_per_set min_selling_price : ℝ),
    first_batch_cost_per_set = 50 ∧
    min_selling_price = 70 ∧
    (quantity_ratio * first_batch_cost / first_batch_cost_per_set) * min_selling_price +
    (first_batch_cost / first_batch_cost_per_set) * min_selling_price -
    first_batch_cost - second_batch_cost ≥ min_profit :=
by sorry

end NUMINAMATH_CALUDE_toy_store_problem_l1292_129248


namespace NUMINAMATH_CALUDE_pyramid_height_l1292_129200

/-- The height of a triangular pyramid with a right-angled base and equal lateral edges -/
theorem pyramid_height (a b l : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 0 < l) :
  let h := (1 / 2 : ℝ) * Real.sqrt (4 * l^2 - a^2 - b^2)
  ∃ (h : ℝ), h > 0 ∧ h = (1 / 2 : ℝ) * Real.sqrt (4 * l^2 - a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_l1292_129200


namespace NUMINAMATH_CALUDE_norm_equals_5_sqrt_5_l1292_129210

def vector : Fin 2 → ℝ
  | 0 => 3
  | 1 => 1

theorem norm_equals_5_sqrt_5 (k : ℝ) : 
  ∃ (v : Fin 2 → ℝ), v 0 = -5 ∧ v 1 = 6 ∧
  (‖(k • vector - v)‖ = 5 * Real.sqrt 5) ↔ 
  (k = (-9 + Real.sqrt 721) / 10 ∨ k = (-9 - Real.sqrt 721) / 10) :=
by sorry

end NUMINAMATH_CALUDE_norm_equals_5_sqrt_5_l1292_129210


namespace NUMINAMATH_CALUDE_factor_expression_l1292_129235

theorem factor_expression (x y : ℝ) : 3 * x^2 - 75 * y^2 = 3 * (x + 5*y) * (x - 5*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1292_129235


namespace NUMINAMATH_CALUDE_rational_expression_iff_perfect_square_l1292_129223

theorem rational_expression_iff_perfect_square (x : ℝ) :
  ∃ (q : ℚ), x + Real.sqrt (x^2 + 9) - 1 / (x + Real.sqrt (x^2 + 9)) = q ↔ 
  ∃ (n : ℕ), x^2 + 9 = n^2 := by
sorry

end NUMINAMATH_CALUDE_rational_expression_iff_perfect_square_l1292_129223


namespace NUMINAMATH_CALUDE_z_to_12_equals_one_l1292_129258

theorem z_to_12_equals_one :
  let z : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  z^12 = 1 := by sorry

end NUMINAMATH_CALUDE_z_to_12_equals_one_l1292_129258


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l1292_129274

theorem lcm_gcd_product (a b : ℕ) (ha : a = 28) (hb : b = 45) :
  (Nat.lcm a b) * (Nat.gcd a b) = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l1292_129274


namespace NUMINAMATH_CALUDE_inequality_proof_l1292_129211

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1292_129211


namespace NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l1292_129264

/-- Represents the regression line equation for pig iron cost -/
def regression_line (x : ℝ) : ℝ := 256 + 3 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_relationship (x : ℝ) :
  regression_line (x + 1) - regression_line x = 3 := by
  sorry

end NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l1292_129264


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_l1292_129207

/-- The number of pens given to Sharon -/
def pens_to_sharon (initial : ℕ) (from_mike : ℕ) (final : ℕ) : ℕ :=
  2 * (initial + from_mike) - final

theorem pens_given_to_sharon :
  pens_to_sharon 5 20 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_l1292_129207


namespace NUMINAMATH_CALUDE_light_blocks_count_is_twenty_l1292_129249

/-- Represents a tower with light colored blocks -/
structure LightTower where
  central_column_height : ℕ
  outer_columns_count : ℕ
  outer_column_height : ℕ

/-- Calculates the total number of light colored blocks in the tower -/
def total_light_blocks (tower : LightTower) : ℕ :=
  tower.central_column_height + tower.outer_columns_count * tower.outer_column_height

/-- Theorem stating that the total number of light colored blocks in the specific tower is 20 -/
theorem light_blocks_count_is_twenty :
  ∃ (tower : LightTower),
    tower.central_column_height = 4 ∧
    tower.outer_columns_count = 8 ∧
    tower.outer_column_height = 2 ∧
    total_light_blocks tower = 20 := by
  sorry


end NUMINAMATH_CALUDE_light_blocks_count_is_twenty_l1292_129249


namespace NUMINAMATH_CALUDE_sector_angle_l1292_129245

/-- Given a circle with radius 12 meters and a sector with area 45.25714285714286 square meters,
    the central angle of the sector is 36 degrees. -/
theorem sector_angle (r : ℝ) (area : ℝ) (h1 : r = 12) (h2 : area = 45.25714285714286) :
  (area / (π * r^2)) * 360 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1292_129245


namespace NUMINAMATH_CALUDE_det_of_matrix_l1292_129294

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 6]

theorem det_of_matrix : Matrix.det matrix = 36 := by sorry

end NUMINAMATH_CALUDE_det_of_matrix_l1292_129294


namespace NUMINAMATH_CALUDE_inverse_function_range_l1292_129201

def is_inverse_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) * f (a - x) = 1

theorem inverse_function_range 
  (f : ℝ → ℝ) 
  (h0 : is_inverse_function f 0)
  (h1 : is_inverse_function f 1)
  (h_range : ∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 1 2) :
  ∀ x ∈ Set.Icc (-2016) 2016, f x ∈ Set.Icc (1/2) 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_range_l1292_129201


namespace NUMINAMATH_CALUDE_binomial_identity_l1292_129252

theorem binomial_identity (n k : ℕ) (hn : n > 1) (hk : k > 1) (hkn : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identity_l1292_129252


namespace NUMINAMATH_CALUDE_simplify_fraction_l1292_129253

theorem simplify_fraction : 15 * (18 / 11) * (-42 / 45) = -23 - (1 / 11) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1292_129253


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1292_129265

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (r₁ + r₂ = 12 ∧ |r₁ - r₂| = 10) ↔ (a = 1 ∧ b = -12 ∧ c = 11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1292_129265


namespace NUMINAMATH_CALUDE_revenue_decrease_l1292_129225

theorem revenue_decrease (R : ℝ) (h1 : R > 0) : 
  let projected_revenue := 1.4 * R
  let actual_revenue := 0.5 * projected_revenue
  let percent_decrease := (R - actual_revenue) / R * 100
  percent_decrease = 30 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l1292_129225


namespace NUMINAMATH_CALUDE_f_properties_l1292_129238

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def f_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≠ Real.pi * ↑(Int.floor (x / Real.pi)) →
         y ≠ Real.pi * ↑(Int.floor (y / Real.pi)) →
         f (x - y) = (f x * f y + 1) / (f y - f x)

theorem f_properties (f : ℝ → ℝ) 
  (h_eq : f_equation f)
  (h_f1 : f 1 = 1)
  (h_pos : ∀ x, 0 < x → x < 2 → f x > 0) :
  is_odd f ∧ 
  f 2 = 0 ∧ 
  f 3 = -1 ∧
  (∀ x, 2 ≤ x → x ≤ 3 → f x ≤ 0) ∧
  (∀ x, 2 ≤ x → x ≤ 3 → f x ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1292_129238


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l1292_129243

theorem coefficient_x_squared (p q : Polynomial ℤ) : 
  p = X^3 - 4*X^2 + 6*X - 2 →
  q = 3*X^2 - 2*X + 5 →
  (p * q).coeff 2 = -38 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l1292_129243


namespace NUMINAMATH_CALUDE_partnership_investment_l1292_129218

theorem partnership_investment (b c total_profit a_profit : ℕ) 
  (hb : b = 4200)
  (hc : c = 10500)
  (htotal : total_profit = 14200)
  (ha_profit : a_profit = 4260) :
  ∃ a : ℕ, a = 6600 ∧ a_profit / total_profit = a / (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_partnership_investment_l1292_129218


namespace NUMINAMATH_CALUDE_point_on_graph_l1292_129272

/-- The function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- The point we're checking -/
def point : ℝ × ℝ := (1, -1)

/-- Theorem: The point (1, -1) lies on the graph of f(x) = -2x + 1 -/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l1292_129272


namespace NUMINAMATH_CALUDE_games_attended_l1292_129208

theorem games_attended (total : ℕ) (missed : ℕ) (attended : ℕ) : 
  total = 12 → missed = 7 → attended = total - missed → attended = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_attended_l1292_129208


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1292_129267

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 466 →
  boys = 127 →
  girls > boys →
  total = girls + boys →
  girls - boys = 212 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1292_129267


namespace NUMINAMATH_CALUDE_restricted_choose_equals_44_l1292_129257

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 cooks from 10 people with a restriction -/
def restrictedChoose : ℕ :=
  choose 10 2 - choose 2 2

theorem restricted_choose_equals_44 : restrictedChoose = 44 := by sorry

end NUMINAMATH_CALUDE_restricted_choose_equals_44_l1292_129257


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l1292_129268

theorem largest_power_dividing_factorial_squared (p : ℕ) (hp : Prime p) :
  (∃ k : ℕ, (p ^ k : ℕ) ∣ (p^2).factorial ∧ 
   ∀ m : ℕ, (p ^ m : ℕ) ∣ (p^2).factorial → m ≤ k) ↔ 
  (∃ k : ℕ, k = p + 1 ∧ (p ^ k : ℕ) ∣ (p^2).factorial ∧ 
   ∀ m : ℕ, (p ^ m : ℕ) ∣ (p^2).factorial → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l1292_129268


namespace NUMINAMATH_CALUDE_vacation_cost_division_l1292_129266

theorem vacation_cost_division (total_cost : ℝ) (cost_difference : ℝ) : 
  total_cost = 480 →
  (total_cost / 4 = total_cost / 6 + cost_difference) →
  cost_difference = 40 →
  6 = (total_cost / (total_cost / 4 - cost_difference)) := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l1292_129266


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1292_129205

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
    F > 0 ∧
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = 
      (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = 13 ∧ B = 9 ∧ C = -3 ∧ D = -2 ∧ E = 165 ∧ F = 51 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1292_129205


namespace NUMINAMATH_CALUDE_chord_length_is_sqrt_34_l1292_129277

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + m = 0
def l (x y : ℝ) : Prop := x + y = 0

-- Define external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ x y, C₁ x y ∧ C₂ x y m ∧ (x - 0)^2 + (y - 0)^2 = (2 + Real.sqrt (25 - m))^2

-- Theorem statement
theorem chord_length_is_sqrt_34 (m : ℝ) :
  externally_tangent m →
  ∃ x₁ y₁ x₂ y₂,
    C₂ x₁ y₁ m ∧ C₂ x₂ y₂ m ∧
    l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 34 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_is_sqrt_34_l1292_129277


namespace NUMINAMATH_CALUDE_jump_ratio_l1292_129255

def hattie_first_round : ℕ := 180
def lorelei_first_round : ℕ := (3 * hattie_first_round) / 4
def total_jumps : ℕ := 605

def hattie_second_round : ℕ := (total_jumps - hattie_first_round - lorelei_first_round - 50) / 2

theorem jump_ratio : 
  (hattie_second_round : ℚ) / hattie_first_round = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_jump_ratio_l1292_129255


namespace NUMINAMATH_CALUDE_polynomial_division_l1292_129273

theorem polynomial_division (x y : ℝ) (hx : x ≠ 0) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1292_129273


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1292_129271

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1292_129271


namespace NUMINAMATH_CALUDE_solution_value_l1292_129227

theorem solution_value (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x + 2 * x = 3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1292_129227


namespace NUMINAMATH_CALUDE_geometric_sequence_term_number_l1292_129212

theorem geometric_sequence_term_number (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * (1/2))  -- geometric sequence with q = 1/2
  → a 1 = 1/2                         -- a₁ = 1/2
  → (∃ n : ℕ, a n = 1/32)             -- aₙ = 1/32 for some n
  → (∃ n : ℕ, a n = 1/32 ∧ n = 5) :=  -- prove that this n is 5
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_number_l1292_129212


namespace NUMINAMATH_CALUDE_workers_count_l1292_129202

theorem workers_count (total_work : ℕ) : ∃ (workers : ℕ),
  (workers * 65 = total_work) ∧
  ((workers + 10) * 55 = total_work) ∧
  (workers = 55) := by
sorry

end NUMINAMATH_CALUDE_workers_count_l1292_129202


namespace NUMINAMATH_CALUDE_temperature_theorem_l1292_129295

def temperature_problem (temp_ny temp_miami temp_sd temp_phoenix : ℝ) : Prop :=
  temp_ny = 80 ∧
  temp_miami = temp_ny + 10 ∧
  temp_sd = temp_miami + 25 ∧
  temp_phoenix = temp_sd * 1.15 ∧
  (temp_ny + temp_miami + temp_sd + temp_phoenix) / 4 = 104.3125

theorem temperature_theorem :
  ∃ temp_ny temp_miami temp_sd temp_phoenix : ℝ,
    temperature_problem temp_ny temp_miami temp_sd temp_phoenix := by
  sorry

end NUMINAMATH_CALUDE_temperature_theorem_l1292_129295


namespace NUMINAMATH_CALUDE_marker_cost_l1292_129236

theorem marker_cost (total_students : ℕ) (buyers : ℕ) (markers_per_student : ℕ) (marker_cost : ℕ) :
  total_students = 24 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  markers_per_student > 1 →
  marker_cost > markers_per_student →
  buyers * marker_cost * markers_per_student = 924 →
  marker_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_marker_cost_l1292_129236


namespace NUMINAMATH_CALUDE_gruia_puzzle_solution_l1292_129280

/-- The number of Gruis (girls) -/
def num_gruis : ℕ := sorry

/-- The number of gruias (pears) -/
def num_gruias : ℕ := sorry

/-- When each Gruia receives one gruia, there is one gruia left over -/
axiom condition1 : num_gruias = num_gruis + 1

/-- When each Gruia receives two gruias, there is a shortage of two gruias -/
axiom condition2 : num_gruias = 2 * num_gruis - 2

theorem gruia_puzzle_solution : num_gruis = 3 ∧ num_gruias = 4 := by sorry

end NUMINAMATH_CALUDE_gruia_puzzle_solution_l1292_129280


namespace NUMINAMATH_CALUDE_min_value_of_cubic_function_l1292_129250

/-- Given a function f(x) = 2x^3 - 6x^2 + a, where a is a constant,
    prove that if the maximum value of f(x) on the interval [-2, 2] is 3,
    then the minimum value of f(x) on [-2, 2] is -37. -/
theorem min_value_of_cubic_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^3 - 6 * x^2 + a
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, f x = -37) ∧ (∀ x ∈ Set.Icc (-2) 2, f x ≥ -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_cubic_function_l1292_129250


namespace NUMINAMATH_CALUDE_sum_first_60_eq_1830_l1292_129290

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 60 natural numbers is 1830 -/
theorem sum_first_60_eq_1830 : sum_first_n 60 = 1830 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_60_eq_1830_l1292_129290


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l1292_129291

theorem polygon_interior_exterior_angles_equality (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 360) → 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l1292_129291


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_81_l1292_129292

theorem fraction_of_powers_equals_81 : (75000 ^ 4) / (25000 ^ 4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_81_l1292_129292


namespace NUMINAMATH_CALUDE_prob_hit_both_l1292_129293

variable (p : ℝ)

-- Define the probability of hitting a single basket in 6 throws
def prob_hit_single (p : ℝ) : ℝ := 1 - (1 - p)^6

-- Define the probability of hitting at least one of two baskets in 6 throws
def prob_hit_at_least_one (p : ℝ) : ℝ := 1 - (1 - 2*p)^6

-- State the theorem
theorem prob_hit_both (hp : 0 ≤ p ∧ p ≤ 1/2) :
  prob_hit_single p + prob_hit_single p - prob_hit_at_least_one p = 1 - 2*(1 - p)^6 + (1 - 2*p)^6 := by
  sorry

end NUMINAMATH_CALUDE_prob_hit_both_l1292_129293


namespace NUMINAMATH_CALUDE_certain_number_proof_l1292_129224

theorem certain_number_proof : ∃ x : ℕ, x * 12 = 173 * 240 ∧ x = 3460 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1292_129224


namespace NUMINAMATH_CALUDE_parabola_f_value_l1292_129289

-- Define the parabola equation
def parabola (d e f y : ℝ) : ℝ := d * y^2 + e * y + f

-- Theorem statement
theorem parabola_f_value :
  ∀ d e f : ℝ,
  -- Vertex condition
  (∀ y : ℝ, parabola d e f (-3) = 2) ∧
  -- Point (7, 0) condition
  parabola d e f 0 = 7 →
  -- Conclusion: f = 7
  f = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_f_value_l1292_129289


namespace NUMINAMATH_CALUDE_investment_years_equals_three_l1292_129232

/-- Calculates the number of years for which a principal is invested, given the interest rate,
    principal amount, and the difference between the principal and interest. -/
def calculate_investment_years (rate : ℚ) (principal : ℚ) (principal_minus_interest : ℚ) : ℚ :=
  (principal - principal_minus_interest) / (principal * rate / 100)

theorem investment_years_equals_three :
  let rate : ℚ := 12
  let principal : ℚ := 9200
  let principal_minus_interest : ℚ := 5888
  calculate_investment_years rate principal principal_minus_interest = 3 := by
  sorry

end NUMINAMATH_CALUDE_investment_years_equals_three_l1292_129232


namespace NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l1292_129241

-- Definition of opposite numbers
def are_opposite (a b : ℝ) : Prop := (abs a = abs b) ∧ (a = -b)

-- Theorem to prove
theorem three_and_negative_three_are_opposite : are_opposite 3 (-3) := by
  sorry

end NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l1292_129241


namespace NUMINAMATH_CALUDE_bus_performance_analysis_l1292_129213

structure BusCompany where
  name : String
  onTime : ℕ
  notOnTime : ℕ

def totalBuses (company : BusCompany) : ℕ := company.onTime + company.notOnTime

def onTimeProbability (company : BusCompany) : ℚ :=
  company.onTime / totalBuses company

def kSquared (companyA companyB : BusCompany) : ℚ :=
  let n := totalBuses companyA + totalBuses companyB
  let a := companyA.onTime
  let b := companyA.notOnTime
  let c := companyB.onTime
  let d := companyB.notOnTime
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def companyA : BusCompany := ⟨"A", 240, 20⟩
def companyB : BusCompany := ⟨"B", 210, 30⟩

theorem bus_performance_analysis :
  (onTimeProbability companyA = 12/13) ∧ 
  (onTimeProbability companyB = 7/8) ∧ 
  (kSquared companyA companyB > 2706/1000) := by
  sorry

end NUMINAMATH_CALUDE_bus_performance_analysis_l1292_129213


namespace NUMINAMATH_CALUDE_odd_prime_sqrt_integer_l1292_129282

theorem odd_prime_sqrt_integer (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) 
  (h_pos : k > 0) (h_sqrt : ∃ n : ℕ, n > 0 ∧ n^2 = k^2 - p*k) : 
  k = (p + 1)^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_sqrt_integer_l1292_129282


namespace NUMINAMATH_CALUDE_relationship_between_variables_l1292_129244

-- Define variables
variable (a b c d : ℝ)
variable (x y q z : ℝ)

-- Define the theorem
theorem relationship_between_variables 
  (h1 : a^(3*x) = c^(2*q)) 
  (h2 : c^(2*q) = b)
  (h3 : c^(4*y) = a^(5*z))
  (h4 : a^(5*z) = d) :
  5*q*z = 6*x*y := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_variables_l1292_129244


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l1292_129279

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i : Nat, i < arr.length - 1 →
    (10 * arr[i]! + arr[i+1]!) % 7 = 0

theorem no_valid_arrangement :
  ¬∃ arr : List Nat, arr.toFinset = {1, 2, 3, 4, 5, 6, 8, 9} ∧ is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l1292_129279


namespace NUMINAMATH_CALUDE_outfits_count_l1292_129230

/-- The number of shirts available. -/
def num_shirts : ℕ := 5

/-- The number of pairs of pants available. -/
def num_pants : ℕ := 3

/-- The number of ties available. -/
def num_ties : ℕ := 2

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * num_ties

/-- Theorem stating that the total number of possible outfits is 30. -/
theorem outfits_count : total_outfits = 30 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1292_129230


namespace NUMINAMATH_CALUDE_abby_damon_weight_l1292_129269

theorem abby_damon_weight 
  (a b c d : ℝ)  -- Weights of Abby, Bart, Cindy, and Damon
  (h1 : a + b = 280)  -- Abby and Bart's combined weight
  (h2 : b + c = 255)  -- Bart and Cindy's combined weight
  (h3 : c + d = 290)  -- Cindy and Damon's combined weight
  : a + d = 315 := by
  sorry

end NUMINAMATH_CALUDE_abby_damon_weight_l1292_129269


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1292_129217

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 10) 
  (h_term : a 10 = 6) : 
  ∃ d : ℚ, d = 1/4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1292_129217


namespace NUMINAMATH_CALUDE_smallest_sum_of_product_l1292_129239

theorem smallest_sum_of_product (a b c d e : ℕ+) : 
  a * b * c * d * e = Nat.factorial 12 → 
  (∀ w x y z v : ℕ+, w * x * y * z * v = Nat.factorial 12 → 
    a + b + c + d + e ≤ w + x + y + z + v) →
  a + b + c + d + e = 501 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_product_l1292_129239


namespace NUMINAMATH_CALUDE_min_value_theorem_l1292_129298

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 14 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 14 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1292_129298


namespace NUMINAMATH_CALUDE_xyz_value_l1292_129234

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 80 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1292_129234


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_for_greater_than_two_l1292_129226

theorem necessary_but_not_sufficient_condition_for_greater_than_two (a : ℝ) :
  (a ≥ 2 → a > 2 → True) ∧ ¬(a ≥ 2 → a > 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_for_greater_than_two_l1292_129226


namespace NUMINAMATH_CALUDE_binomial_product_and_evaluation_l1292_129254

theorem binomial_product_and_evaluation :
  ∀ x : ℝ,
  (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 ∧
  (8 * (-1)^2 - 18 * (-1) - 18) = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_and_evaluation_l1292_129254


namespace NUMINAMATH_CALUDE_max_power_under_500_l1292_129283

theorem max_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 2) (h_less_500 : a^b < 500) :
  ∃ (a_max b_max : ℕ),
    a_max > 0 ∧ b_max > 2 ∧ a_max^b_max < 500 ∧
    ∀ (x y : ℕ), x > 0 → y > 2 → x^y < 500 → x^y ≤ a_max^b_max ∧
    a_max + b_max = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_power_under_500_l1292_129283


namespace NUMINAMATH_CALUDE_cats_in_sacks_l1292_129228

theorem cats_in_sacks (cat_prices sack_prices : Finset ℕ) : 
  cat_prices.card = 20 →
  sack_prices.card = 20 →
  (∀ p ∈ cat_prices, 1200 ≤ p ∧ p ≤ 1500) →
  (∀ p ∈ sack_prices, 10 ≤ p ∧ p ≤ 100) →
  cat_prices.toList.Nodup →
  sack_prices.toList.Nodup →
  ∃ (c1 c2 : ℕ) (s1 s2 : ℕ),
    c1 ∈ cat_prices ∧ 
    c2 ∈ cat_prices ∧ 
    s1 ∈ sack_prices ∧ 
    s2 ∈ sack_prices ∧
    c1 ≠ c2 ∧ 
    s1 ≠ s2 ∧ 
    c1 + s1 = c2 + s2 :=
by sorry

end NUMINAMATH_CALUDE_cats_in_sacks_l1292_129228
