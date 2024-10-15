import Mathlib

namespace NUMINAMATH_CALUDE_special_square_divisions_l521_52109

/-- Represents a 5x5 square with a 3x3 center and 1x3 rectangles on each side -/
structure SpecialSquare :=
  (size : Nat)
  (center_size : Nat)
  (side_rectangle_size : Nat)
  (h_size : size = 5)
  (h_center : center_size = 3)
  (h_side : side_rectangle_size = 3)

/-- Counts the number of ways to divide the SpecialSquare into 1x3 rectangles -/
def count_divisions (square : SpecialSquare) : Nat :=
  2

/-- Theorem stating that the number of ways to divide the SpecialSquare into 1x3 rectangles is 2 -/
theorem special_square_divisions (square : SpecialSquare) :
  count_divisions square = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_square_divisions_l521_52109


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l521_52144

/-- The ratio of the area of a rectangle to the area of a triangle -/
theorem rectangle_triangle_area_ratio 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ) 
  (triangle_area : ℝ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 4)
  (h3 : triangle_area = 60) :
  (rectangle_length * rectangle_width) / triangle_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l521_52144


namespace NUMINAMATH_CALUDE_apple_boxes_l521_52145

theorem apple_boxes (apples_per_crate : ℕ) (crates : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) :
  apples_per_crate = 42 →
  crates = 12 →
  rotten_apples = 4 →
  apples_per_box = 10 →
  (crates * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_l521_52145


namespace NUMINAMATH_CALUDE_square_circle_union_area_l521_52150

theorem square_circle_union_area (s : Real) (r : Real) :
  s = 12 ∧ r = 12 →
  (s^2) + (π * r^2) - (π * r^2 / 4) = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l521_52150


namespace NUMINAMATH_CALUDE_jia_incorrect_questions_l521_52128

-- Define the type for questions
inductive Question
| Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7

-- Define a person's answers
def Answers := Question → Bool

-- Define the correct answers
def correct_answers : Answers := sorry

-- Define Jia's answers
def jia_answers : Answers := sorry

-- Define Yi's answers
def yi_answers : Answers := sorry

-- Define Bing's answers
def bing_answers : Answers := sorry

-- Function to count correct answers
def count_correct (answers : Answers) : Nat := sorry

-- Theorem stating the problem conditions and the conclusion to be proved
theorem jia_incorrect_questions :
  (count_correct jia_answers = 5) →
  (count_correct yi_answers = 5) →
  (count_correct bing_answers = 5) →
  (jia_answers Question.Q1 ≠ correct_answers Question.Q1) ∧
  (jia_answers Question.Q3 ≠ correct_answers Question.Q3) :=
by sorry

end NUMINAMATH_CALUDE_jia_incorrect_questions_l521_52128


namespace NUMINAMATH_CALUDE_power_eight_mod_eleven_l521_52149

theorem power_eight_mod_eleven : 8^2030 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_eight_mod_eleven_l521_52149


namespace NUMINAMATH_CALUDE_no_valid_schedule_for_100_l521_52112

/-- Represents a duty schedule for militia members -/
structure DutySchedule (n : ℕ) where
  nights : Set (Fin n × Fin n × Fin n)
  all_pairs_once : ∀ i j, i < j → ∃! k, (i, j, k) ∈ nights ∨ (i, k, j) ∈ nights ∨ (j, i, k) ∈ nights ∨ (j, k, i) ∈ nights ∨ (k, i, j) ∈ nights ∨ (k, j, i) ∈ nights

/-- Theorem stating the impossibility of creating a valid duty schedule for 100 militia members -/
theorem no_valid_schedule_for_100 : ¬∃ (schedule : DutySchedule 100), True := by
  sorry

end NUMINAMATH_CALUDE_no_valid_schedule_for_100_l521_52112


namespace NUMINAMATH_CALUDE_donation_growth_rate_l521_52118

theorem donation_growth_rate 
  (initial_donation : ℝ) 
  (third_day_donation : ℝ) 
  (h1 : initial_donation = 10000)
  (h2 : third_day_donation = 12100) :
  ∃ (rate : ℝ), 
    initial_donation * (1 + rate)^2 = third_day_donation ∧ 
    rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_donation_growth_rate_l521_52118


namespace NUMINAMATH_CALUDE_pyramid_edges_count_l521_52193

/-- A prism is a polyhedron with two congruent bases and rectangular faces connecting corresponding edges of the bases. -/
structure Prism where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  sum_property : vertices + faces + edges = 50
  euler_formula : vertices - edges + faces = 2

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a point (apex). -/
structure Pyramid where
  base_edges : ℕ

/-- Given a prism, construct a pyramid with the same base shape. -/
def pyramid_from_prism (p : Prism) : Pyramid :=
  { base_edges := (p.edges / 3) }

theorem pyramid_edges_count (p : Prism) : 
  (pyramid_from_prism p).base_edges * 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edges_count_l521_52193


namespace NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l521_52115

/-- A color type with three possible values -/
inductive Color
  | One
  | Two
  | Three

/-- A point on the integer plane -/
structure Point where
  x : Int
  y : Int

/-- A coloring of the integer plane -/
def Coloring := Point → Color

/-- Predicate for a right triangle -/
def is_right_triangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- Main theorem -/
theorem exists_tricolor_right_triangle (c : Coloring) 
  (h1 : ∃ p : Point, c p = Color.One)
  (h2 : ∃ p : Point, c p = Color.Two)
  (h3 : ∃ p : Point, c p = Color.Three) :
  ∃ p1 p2 p3 : Point, 
    is_right_triangle p1 p2 p3 ∧ 
    c p1 ≠ c p2 ∧ c p2 ≠ c p3 ∧ c p3 ≠ c p1 :=
sorry

end NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l521_52115


namespace NUMINAMATH_CALUDE_crayon_box_problem_l521_52173

theorem crayon_box_problem (C R B G Y P U : ℝ) : 
  R + B + G + Y + P + U = C →
  R = 12 →
  B = 8 →
  G = (3/4) * B →
  Y = 0.15 * C →
  P = U →
  P = 0.425 * C - 13 := by
sorry

end NUMINAMATH_CALUDE_crayon_box_problem_l521_52173


namespace NUMINAMATH_CALUDE_function_divisibility_l521_52139

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem function_divisibility 
  (f : ℤ → ℕ+) 
  (h : ∀ (m n : ℤ), is_divisible (f (m - n)) (f m - f n)) :
  ∀ (m n : ℤ), f m ≤ f n → is_divisible (f m) (f n) :=
sorry

end NUMINAMATH_CALUDE_function_divisibility_l521_52139


namespace NUMINAMATH_CALUDE_women_in_room_l521_52151

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  (2 * (initial_women - 3)) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l521_52151


namespace NUMINAMATH_CALUDE_projection_equality_l521_52105

/-- Given two vectors in R^2 that project to the same vector, 
    prove that the projection is (16/5, 8/5) -/
theorem projection_equality (v : ℝ × ℝ) :
  let a : ℝ × ℝ := (5, -2)
  let b : ℝ × ℝ := (2, 4)
  let proj (x : ℝ × ℝ) := 
    let dot_prod := x.1 * v.1 + x.2 * v.2
    let v_norm_sq := v.1 * v.1 + v.2 * v.2
    ((dot_prod / v_norm_sq) * v.1, (dot_prod / v_norm_sq) * v.2)
  proj a = proj b → proj a = (16/5, 8/5) :=
by
  sorry

#check projection_equality

end NUMINAMATH_CALUDE_projection_equality_l521_52105


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l521_52133

/-- Given a geometric sequence {a_n} with common ratio q (q ≠ 1),
    if a_1, a_3, a_2 form an arithmetic sequence, then q = -1/2 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1)
    (h2 : ∀ n, a (n + 1) = a n * q)  -- geometric sequence condition
    (h3 : 2 * a 3 = a 1 + a 2)       -- arithmetic sequence condition
    : q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l521_52133


namespace NUMINAMATH_CALUDE_power_of_256_l521_52157

theorem power_of_256 : (256 : ℝ) ^ (4/5 : ℝ) = 64 := by
  have h1 : 256 = 2^8 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_256_l521_52157


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l521_52140

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

-- Define vectors A, B, and D
def A (k : ℝ) := 2 • a + k • b
def B := a + b
def D := a - 2 • b

-- Define collinearity
def collinear (x y z : V) : Prop := ∃ (t : ℝ), y - x = t • (z - x)

-- Theorem statement
theorem collinear_points_k_value
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hnc : ¬ ∃ (r : ℝ), a = r • b)
  (hcol : collinear V (A V a b k) (B V a b) (D V a b)) :
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l521_52140


namespace NUMINAMATH_CALUDE_complex_circle_equation_l521_52106

theorem complex_circle_equation (z : ℂ) (h : Complex.abs (z - 1) = 5) :
  ∃ (x y : ℝ), z = Complex.mk x y ∧ 
  -4 ≤ x ∧ x ≤ 6 ∧ 
  (y = Real.sqrt (25 - (x - 1)^2) ∨ y = -Real.sqrt (25 - (x - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_complex_circle_equation_l521_52106


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l521_52124

/-- A point in the Cartesian plane lies in the fourth quadrant if and only if
    its x-coordinate is positive and its y-coordinate is negative. -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point (8, -3) lies in the fourth quadrant of the Cartesian coordinate system. -/
theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant 8 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l521_52124


namespace NUMINAMATH_CALUDE_common_tangent_sum_l521_52159

-- Define the parabolas
def P₁ (x y : ℚ) : Prop := y = x^2 + 51/50
def P₂ (x y : ℚ) : Prop := x = y^2 + 19/2

-- Define the tangent line
def TangentLine (a b c : ℕ) (x y : ℚ) : Prop := a * x + b * y = c

-- Define the property of being a common tangent to both parabolas
def CommonTangent (a b c : ℕ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℚ), 
    P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧ 
    TangentLine a b c x₁ y₁ ∧ 
    TangentLine a b c x₂ y₂

-- The main theorem
theorem common_tangent_sum :
  ∀ (a b c : ℕ), 
    a > 0 → b > 0 → c > 0 →
    Nat.gcd a (Nat.gcd b c) = 1 →
    CommonTangent a b c →
    (a : ℤ) + b + c = 37 := by sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l521_52159


namespace NUMINAMATH_CALUDE_calculation_proof_l521_52160

theorem calculation_proof :
  2 / (-1/4) - |(-Real.sqrt 18)| + (1/5)⁻¹ = -3 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l521_52160


namespace NUMINAMATH_CALUDE_abs_neg_one_third_eq_one_third_l521_52162

theorem abs_neg_one_third_eq_one_third : |(-1/3 : ℚ)| = 1/3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_eq_one_third_l521_52162


namespace NUMINAMATH_CALUDE_cube_ending_in_eight_and_nine_l521_52195

theorem cube_ending_in_eight_and_nine :
  ∀ a b : ℕ,
  (10 ≤ a ∧ a < 100) →
  (10 ≤ b ∧ b < 100) →
  (1000 ≤ a^3 ∧ a^3 < 10000) →
  (1000 ≤ b^3 ∧ b^3 < 10000) →
  a^3 % 10 = 8 →
  b^3 % 10 = 9 →
  a = 12 ∧ b = 19 :=
by sorry

end NUMINAMATH_CALUDE_cube_ending_in_eight_and_nine_l521_52195


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l521_52127

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 4 * x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + y' = 1 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l521_52127


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l521_52169

theorem trigonometric_equation_solution (x : ℝ) : 
  Real.cos (7 * x) + Real.sin (8 * x) = Real.cos (3 * x) - Real.sin (2 * x) → 
  (∃ n : ℤ, x = n * Real.pi / 5) ∨ 
  (∃ k : ℤ, x = Real.pi / 2 * (4 * k - 1)) ∨ 
  (∃ l : ℤ, x = Real.pi / 10 * (4 * l + 1)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l521_52169


namespace NUMINAMATH_CALUDE_diameter_in_scientific_notation_l521_52191

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

theorem diameter_in_scientific_notation :
  scientific_notation 0.0000077 7.7 (-6) :=
sorry

end NUMINAMATH_CALUDE_diameter_in_scientific_notation_l521_52191


namespace NUMINAMATH_CALUDE_angle_bisector_c_value_l521_52184

/-- Triangle with vertices A, B, C in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle bisector of a triangle -/
def angleBisector (t : Triangle) (v : ℝ × ℝ) (l : LineEquation) : Prop :=
  -- This is a placeholder for the actual definition of an angle bisector
  True

theorem angle_bisector_c_value (t : Triangle) (l : LineEquation) :
  t.A = (-2, 3) →
  t.B = (-6, -8) →
  t.C = (4, -1) →
  l.a = 5 →
  l.b = 4 →
  angleBisector t t.B l →
  l.c + 5 = -155/7 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_c_value_l521_52184


namespace NUMINAMATH_CALUDE_accessory_time_is_ten_l521_52153

/-- Represents the production details of a doll factory --/
structure DollFactory where
  num_dolls : ℕ
  time_per_doll : ℕ
  total_time : ℕ
  shoes_per_doll : ℕ
  bags_per_doll : ℕ
  cosmetics_per_doll : ℕ
  hats_per_doll : ℕ

/-- Calculates the time taken to make each accessory --/
def time_per_accessory (factory : DollFactory) : ℕ :=
  let total_accessories := factory.num_dolls * (factory.shoes_per_doll + factory.bags_per_doll + 
                           factory.cosmetics_per_doll + factory.hats_per_doll)
  let time_for_dolls := factory.num_dolls * factory.time_per_doll
  let time_for_accessories := factory.total_time - time_for_dolls
  time_for_accessories / total_accessories

/-- Theorem stating that the time to make each accessory is 10 seconds --/
theorem accessory_time_is_ten (factory : DollFactory) 
  (h1 : factory.num_dolls = 12000)
  (h2 : factory.time_per_doll = 45)
  (h3 : factory.total_time = 1860000)
  (h4 : factory.shoes_per_doll = 2)
  (h5 : factory.bags_per_doll = 3)
  (h6 : factory.cosmetics_per_doll = 1)
  (h7 : factory.hats_per_doll = 5) :
  time_per_accessory factory = 10 := by
  sorry

#eval time_per_accessory { 
  num_dolls := 12000, 
  time_per_doll := 45, 
  total_time := 1860000, 
  shoes_per_doll := 2, 
  bags_per_doll := 3, 
  cosmetics_per_doll := 1, 
  hats_per_doll := 5 
}

end NUMINAMATH_CALUDE_accessory_time_is_ten_l521_52153


namespace NUMINAMATH_CALUDE_average_of_five_integers_l521_52113

theorem average_of_five_integers (k m r s t : ℕ) : 
  k < m → m < r → r < s → s < t → 
  t = 42 → 
  r ≤ 17 → 
  (k + m + r + s + t : ℚ) / 5 = 266 / 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_five_integers_l521_52113


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l521_52142

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let d₁ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  let d₂ := (Q.1 - R.1)^2 + (Q.2 - R.2)^2
  let d₃ := (R.1 - P.1)^2 + (R.2 - P.2)^2
  d₁ = d₂ ∧ d₂ = d₃

-- Define being on the same branch of the hyperbola
def same_branch (P Q R : ℝ × ℝ) : Prop :=
  (P.1 > 0 ∧ Q.1 > 0 ∧ R.1 > 0) ∨ (P.1 < 0 ∧ Q.1 < 0 ∧ R.1 < 0)

-- Main theorem
theorem hyperbola_equilateral_triangle :
  ∀ P Q R : ℝ × ℝ,
  hyperbola P.1 P.2 →
  hyperbola Q.1 Q.2 →
  hyperbola R.1 R.2 →
  is_equilateral_triangle P Q R →
  (¬ same_branch P Q R) ∧
  (P = (-1, -1) →
   Q.1 > 0 →
   R.1 > 0 →
   Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧
   R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l521_52142


namespace NUMINAMATH_CALUDE_cubic_factorization_l521_52197

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m - 2)*(m + 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l521_52197


namespace NUMINAMATH_CALUDE_balloon_difference_is_one_l521_52100

/-- The number of balloons Jake has more than Allan -/
def balloon_difference (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) : ℕ :=
  (jake_initial_balloons + jake_bought_balloons) - allan_balloons

/-- Theorem stating the difference in balloons between Jake and Allan -/
theorem balloon_difference_is_one :
  balloon_difference 6 3 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_is_one_l521_52100


namespace NUMINAMATH_CALUDE_inequality_proof_l521_52102

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) ≥
  a^2 + b^2 + c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l521_52102


namespace NUMINAMATH_CALUDE_expected_sum_of_marbles_l521_52156

/-- The set of marble numbers -/
def marbleNumbers : Finset ℕ := {2, 3, 4, 5, 6, 7}

/-- The sum of two different elements from the set -/
def pairSum (a b : ℕ) : ℕ := a + b

/-- The set of all possible pairs of different marbles -/
def marblePairs : Finset (ℕ × ℕ) :=
  (marbleNumbers.product marbleNumbers).filter (fun p => p.1 < p.2)

/-- The expected value of the sum of two randomly drawn marbles -/
def expectedSum : ℚ :=
  (marblePairs.sum (fun p => pairSum p.1 p.2)) / marblePairs.card

theorem expected_sum_of_marbles :
  expectedSum = 145 / 15 := by sorry

end NUMINAMATH_CALUDE_expected_sum_of_marbles_l521_52156


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l521_52103

theorem geometric_sequence_fifth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : a * r^3 = 324) : a * r^4 = 324 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l521_52103


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l521_52143

/-- Given a square with side length 12 and a rectangle with the same perimeter and width 6,
    prove that the length of the rectangle is 18. -/
theorem rectangle_length_from_square_wire (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l521_52143


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l521_52122

/-- The number of complete books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem: Robert can read 2 complete 360-page books in 8 hours at a rate of 120 pages per hour -/
theorem robert_reading_capacity :
  books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l521_52122


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l521_52178

/-- Two parallel lines with a specified distance between them -/
structure ParallelLines where
  -- First line equation: 3x - y + 3 = 0
  l₁ : ℝ → ℝ → Prop
  l₁_def : l₁ = fun x y ↦ 3 * x - y + 3 = 0
  -- Second line equation: 3x - y + C = 0
  l₂ : ℝ → ℝ → Prop
  C : ℝ
  l₂_def : l₂ = fun x y ↦ 3 * x - y + C = 0
  -- Distance between the lines is √10
  distance : ℝ
  distance_def : distance = Real.sqrt 10

/-- The main theorem stating the possible values of C -/
theorem parallel_lines_distance (pl : ParallelLines) : pl.C = 13 ∨ pl.C = -7 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_distance_l521_52178


namespace NUMINAMATH_CALUDE_alex_has_sixty_shells_l521_52108

/-- The number of seashells in a dozen -/
def dozen : ℕ := 12

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := 2 * dozen

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := kyle_shells / 3

/-- The number of seashells Alex unearthed -/
def alex_shells : ℕ := 3 * leigh_shells + mimi_shells / 2

/-- Theorem stating that Alex had 60 seashells -/
theorem alex_has_sixty_shells : alex_shells = 60 := by
  sorry

end NUMINAMATH_CALUDE_alex_has_sixty_shells_l521_52108


namespace NUMINAMATH_CALUDE_p_or_q_true_not_imply_p_and_q_true_l521_52194

theorem p_or_q_true_not_imply_p_and_q_true (p q : Prop) : 
  (p ∨ q) → ¬(p ∧ q → True) :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_true_not_imply_p_and_q_true_l521_52194


namespace NUMINAMATH_CALUDE_joyce_apples_l521_52167

/-- Proves that if Joyce starts with 75 apples and gives 52 to Larry, she ends up with 23 apples -/
theorem joyce_apples : ∀ (initial_apples given_apples remaining_apples : ℕ),
  initial_apples = 75 →
  given_apples = 52 →
  remaining_apples = initial_apples - given_apples →
  remaining_apples = 23 := by
  sorry


end NUMINAMATH_CALUDE_joyce_apples_l521_52167


namespace NUMINAMATH_CALUDE_berts_profit_l521_52179

/-- Calculates the profit for a single item --/
def itemProfit (salesPrice : ℚ) (taxRate : ℚ) : ℚ :=
  salesPrice - (salesPrice * taxRate) - (salesPrice - 10)

/-- Calculates the total profit from the sale --/
def totalProfit (barrelPrice : ℚ) (toolsPrice : ℚ) (fertilizerPrice : ℚ) 
  (barrelTaxRate : ℚ) (toolsTaxRate : ℚ) (fertilizerTaxRate : ℚ) : ℚ :=
  itemProfit barrelPrice barrelTaxRate + 
  itemProfit toolsPrice toolsTaxRate + 
  itemProfit fertilizerPrice fertilizerTaxRate

/-- Theorem stating that Bert's total profit is $14.90 --/
theorem berts_profit : 
  totalProfit 90 50 30 (10/100) (5/100) (12/100) = 149/10 :=
by sorry

end NUMINAMATH_CALUDE_berts_profit_l521_52179


namespace NUMINAMATH_CALUDE_intersection_of_H_and_G_l521_52196

def H : Set ℕ := {2, 3, 4}
def G : Set ℕ := {1, 3}

theorem intersection_of_H_and_G : H ∩ G = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_H_and_G_l521_52196


namespace NUMINAMATH_CALUDE_proposition_relationship_l521_52123

theorem proposition_relationship (p q : Prop) : 
  (¬p ∨ ¬q → ¬p ∧ ¬q) ∧ 
  ∃ (p q : Prop), (¬p ∧ ¬q) ∧ ¬(¬p ∨ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l521_52123


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l521_52182

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (a - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (a - 1) * y^2 - 2 * y + 1 = 0) → 
  (a < 2 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l521_52182


namespace NUMINAMATH_CALUDE_apple_difference_l521_52129

theorem apple_difference (total : ℕ) (red : ℕ) (h1 : total = 44) (h2 : red = 16) :
  total > red → total - red - red = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l521_52129


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l521_52130

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Returns the four-digit number formed by removing the digit at position i -/
def removeDigit (n : FiveDigitNumber) (i : Fin 5) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem unique_five_digit_number :
  ∃! (n : FiveDigitNumber),
    ∃ (i : Fin 5),
      n.val + removeDigit n i = 54321 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l521_52130


namespace NUMINAMATH_CALUDE_max_p_value_l521_52198

/-- Given a function f(x) = e^x and real numbers m, n, p satisfying certain conditions,
    the maximum value of p is 2ln(2) - ln(3). -/
theorem max_p_value (f : ℝ → ℝ) (m n p : ℝ) 
    (h1 : ∀ x, f x = Real.exp x)
    (h2 : f (m + n) = f m + f n)
    (h3 : f (m + n + p) = f m + f n + f p) :
    p ≤ 2 * Real.log 2 - Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_max_p_value_l521_52198


namespace NUMINAMATH_CALUDE_regular_star_n_value_l521_52134

/-- Represents an n-pointed regular star diagram -/
structure RegularStar where
  n : ℕ
  edge_length : ℝ
  angle_A : ℝ
  angle_B : ℝ

/-- The properties of the regular star diagram -/
def is_valid_regular_star (star : RegularStar) : Prop :=
  star.n > 0 ∧
  star.edge_length > 0 ∧
  star.angle_A > 0 ∧
  star.angle_B > 0 ∧
  star.angle_A = (5 / 14) * star.angle_B ∧
  star.n * (star.angle_A + star.angle_B) = 360

theorem regular_star_n_value (star : RegularStar) 
  (h : is_valid_regular_star star) : star.n = 133 := by
  sorry

#check regular_star_n_value

end NUMINAMATH_CALUDE_regular_star_n_value_l521_52134


namespace NUMINAMATH_CALUDE_square_binomial_constant_l521_52192

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l521_52192


namespace NUMINAMATH_CALUDE_work_completion_time_l521_52187

/-- Work rates and completion times for a team project -/
theorem work_completion_time 
  (man_rate : ℚ) 
  (woman_rate : ℚ) 
  (girl_rate : ℚ) 
  (team_rate : ℚ) 
  (h1 : man_rate = 1/6) 
  (h2 : woman_rate = 1/18) 
  (h3 : girl_rate = 1/12) 
  (h4 : team_rate = 1/3) 
  (h5 : man_rate + woman_rate + girl_rate + (team_rate - man_rate - woman_rate - girl_rate) = team_rate) : 
  (1 / ((team_rate - man_rate - woman_rate - girl_rate) + 2 * girl_rate)) = 36/7 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l521_52187


namespace NUMINAMATH_CALUDE_rectangular_frame_properties_l521_52190

/-- Calculates the total length of wire needed for a rectangular frame --/
def total_wire_length (a b c : ℕ) : ℕ := 4 * (a + b + c)

/-- Calculates the total area of paper needed to cover a rectangular frame --/
def total_paper_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

theorem rectangular_frame_properties :
  total_wire_length 3 4 5 = 48 ∧ total_paper_area 3 4 5 = 94 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_frame_properties_l521_52190


namespace NUMINAMATH_CALUDE_fraction_simplification_l521_52148

theorem fraction_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : y + 1/x ≠ 0) :
  (x + 1/y) / (y + 1/x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l521_52148


namespace NUMINAMATH_CALUDE_lg_2_plus_lg_5_equals_1_l521_52126

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_2_plus_lg_5_equals_1 : lg 2 + lg 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_2_plus_lg_5_equals_1_l521_52126


namespace NUMINAMATH_CALUDE_alice_twice_bob_age_l521_52141

theorem alice_twice_bob_age (alice_age bob_age : ℕ) : 
  alice_age = bob_age + 10 →
  alice_age + 5 = 19 →
  ∃ (years : ℕ), (alice_age + years = 2 * (bob_age + years)) ∧ years = 6 :=
by sorry

end NUMINAMATH_CALUDE_alice_twice_bob_age_l521_52141


namespace NUMINAMATH_CALUDE_division_problem_l521_52186

theorem division_problem (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient →
  divisor = 5 * remainder →
  remainder = 46 →
  divisor * quotient + remainder = 5336 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l521_52186


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l521_52147

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -6
  let c : ℝ := 1
  let x₁ : ℝ := (3 + Real.sqrt 7) / 2
  let x₂ : ℝ := (3 - Real.sqrt 7) / 2
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l521_52147


namespace NUMINAMATH_CALUDE_B_power_difference_l521_52175

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_difference : 
  B^30 - B^29 = !![2, 4; 0, 1] := by sorry

end NUMINAMATH_CALUDE_B_power_difference_l521_52175


namespace NUMINAMATH_CALUDE_discount_saves_money_savings_amount_l521_52125

/-- Represents the ticket pricing strategy for a park -/
structure TicketStrategy where
  regular_price : ℕ  -- Regular price per ticket
  discount_rate : ℚ  -- Discount rate for group tickets
  discount_threshold : ℕ  -- Minimum number of people for group discount

/-- Calculates the total cost for a given number of tickets -/
def total_cost (strategy : TicketStrategy) (num_tickets : ℕ) : ℚ :=
  if num_tickets ≥ strategy.discount_threshold
  then (strategy.regular_price * num_tickets * (1 - strategy.discount_rate))
  else (strategy.regular_price * num_tickets)

/-- Theorem: Purchasing 25 tickets with discount is cheaper than 23 without discount -/
theorem discount_saves_money (strategy : TicketStrategy) 
  (h1 : strategy.regular_price = 10)
  (h2 : strategy.discount_rate = 1/5)
  (h3 : strategy.discount_threshold = 25) :
  total_cost strategy 25 < total_cost strategy 23 ∧ 
  total_cost strategy 23 - total_cost strategy 25 = 30 :=
by sorry

/-- Corollary: The savings amount to exactly 30 yuan -/
theorem savings_amount (strategy : TicketStrategy)
  (h1 : strategy.regular_price = 10)
  (h2 : strategy.discount_rate = 1/5)
  (h3 : strategy.discount_threshold = 25) :
  total_cost strategy 23 - total_cost strategy 25 = 30 :=
by sorry

end NUMINAMATH_CALUDE_discount_saves_money_savings_amount_l521_52125


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l521_52189

/-- Represents the Chemistry Club and its officer selection process -/
structure ChemistryClub where
  totalMembers : Nat
  aliceAndBobCondition : Bool
  ronaldCondition : Bool

/-- Calculates the number of ways to select officers -/
def selectOfficers (club : ChemistryClub) : Nat :=
  let withoutAliceBob := (club.totalMembers - 3) * (club.totalMembers - 4) * (club.totalMembers - 5)
  let withAliceBob := 6
  withoutAliceBob + withAliceBob

/-- The main theorem stating the number of ways to select officers -/
theorem officer_selection_theorem (club : ChemistryClub) 
  (h1 : club.totalMembers = 25)
  (h2 : club.aliceAndBobCondition = true)
  (h3 : club.ronaldCondition = true) :
  selectOfficers club = 9246 := by
  sorry

#eval selectOfficers { totalMembers := 25, aliceAndBobCondition := true, ronaldCondition := true }

end NUMINAMATH_CALUDE_officer_selection_theorem_l521_52189


namespace NUMINAMATH_CALUDE_towel_purchase_cost_is_correct_l521_52170

/-- Calculates the total cost of Bailey's towel purchase --/
def towel_purchase_cost : ℝ :=
  let guest_price := 40
  let master_price := 50
  let hand_price := 30
  let kitchen_price := 20
  
  let guest_discount := 0.15
  let master_discount := 0.20
  let hand_discount := 0.15
  let kitchen_discount := 0.10
  
  let sales_tax := 0.08
  
  let guest_discounted := guest_price * (1 - guest_discount)
  let master_discounted := master_price * (1 - master_discount)
  let hand_discounted := hand_price * (1 - hand_discount)
  let kitchen_discounted := kitchen_price * (1 - kitchen_discount)
  
  let total_before_tax := 
    2 * guest_discounted + 
    4 * master_discounted + 
    3 * hand_discounted + 
    5 * kitchen_discounted
  
  total_before_tax * (1 + sales_tax)

/-- Theorem stating that the total cost of Bailey's towel purchase is $426.06 --/
theorem towel_purchase_cost_is_correct : 
  towel_purchase_cost = 426.06 := by sorry

end NUMINAMATH_CALUDE_towel_purchase_cost_is_correct_l521_52170


namespace NUMINAMATH_CALUDE_complex_power_sum_l521_52165

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 1 + z := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l521_52165


namespace NUMINAMATH_CALUDE_product_increase_by_three_times_l521_52131

theorem product_increase_by_three_times : 
  ∃ (a b c d : ℕ), (a + 1) * (b + 1) * (c + 1) * (d + 1) = 3 * (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_product_increase_by_three_times_l521_52131


namespace NUMINAMATH_CALUDE_divisors_of_x_15_minus_1_l521_52158

theorem divisors_of_x_15_minus_1 :
  ∀ k : ℕ, k ≤ 14 →
    ∃ p : Polynomial ℤ, (Polynomial.degree p = k) ∧ (p ∣ (X ^ 15 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_x_15_minus_1_l521_52158


namespace NUMINAMATH_CALUDE_angle_at_point_l521_52180

theorem angle_at_point (x : ℝ) : 
  (170 : ℝ) + 3 * x = 360 → x = 190 / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_at_point_l521_52180


namespace NUMINAMATH_CALUDE_set_equality_l521_52183

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l521_52183


namespace NUMINAMATH_CALUDE_negative_three_is_rational_l521_52137

theorem negative_three_is_rational : ℚ :=
  sorry

end NUMINAMATH_CALUDE_negative_three_is_rational_l521_52137


namespace NUMINAMATH_CALUDE_positive_solution_equation_l521_52111

theorem positive_solution_equation (x : ℝ) :
  x = 20 + Real.sqrt 409 →
  x > 0 ∧
  (1 / 3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_positive_solution_equation_l521_52111


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l521_52104

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if there are at least six consecutive nonprime numbers before n, false otherwise -/
def hasSixConsecutiveNonprimes (n : ℕ) : Prop := sorry

theorem smallest_prime_after_six_nonprimes : 
  ∃ (k : ℕ), 
    isPrime k ∧ 
    hasSixConsecutiveNonprimes k ∧ 
    (∀ (m : ℕ), m < k → ¬(isPrime m ∧ hasSixConsecutiveNonprimes m)) ∧
    k = 97 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l521_52104


namespace NUMINAMATH_CALUDE_fill_water_tank_days_l521_52114

/-- Represents the number of days needed to fill a water tank -/
def days_to_fill_tank (tank_capacity : ℕ) (daily_collection : ℕ) : ℕ :=
  (tank_capacity * 1000 + daily_collection - 1) / daily_collection

/-- Theorem stating that it takes 206 days to fill the water tank -/
theorem fill_water_tank_days : days_to_fill_tank 350 1700 = 206 := by
  sorry

end NUMINAMATH_CALUDE_fill_water_tank_days_l521_52114


namespace NUMINAMATH_CALUDE_remaining_typing_orders_l521_52168

/-- The number of letters in total -/
def totalLetters : ℕ := 10

/-- The label of the letter that has been typed by midday -/
def typedLetter : ℕ := 9

/-- The number of different orders for typing the remaining letters -/
def typingOrders : ℕ := 1280

/-- 
Theorem: Given 10 letters labeled from 1 to 10, where letter 9 has been typed by midday,
the number of different orders for typing the remaining letters is 1280.
-/
theorem remaining_typing_orders :
  (totalLetters = 10) →
  (typedLetter = 9) →
  (typingOrders = 1280) :=
by sorry

end NUMINAMATH_CALUDE_remaining_typing_orders_l521_52168


namespace NUMINAMATH_CALUDE_intersection_point_l521_52132

/-- The quadratic function f(x) = x^2 - 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Theorem: The point (2,0) is the only intersection point of y = x^2 - 4x + 4 with the x-axis -/
theorem intersection_point : 
  (∃! x : ℝ, f x = 0) ∧ (f 2 = 0) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l521_52132


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l521_52138

/-- Represents the price reduction equation for a medicine that undergoes two
    successive price reductions of the same percentage. -/
theorem medicine_price_reduction (x : ℝ) : 
  (58 : ℝ) * (1 - x)^2 = 43 ↔ 
  (∃ (initial_price final_price : ℝ),
    initial_price = 58 ∧
    final_price = 43 ∧
    final_price = initial_price * (1 - x)^2) :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l521_52138


namespace NUMINAMATH_CALUDE_count_eight_in_product_l521_52146

/-- The number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The product of 987654321 and 9 -/
def product : ℕ := 987654321 * 9

/-- Theorem: The number of occurrences of the digit 8 in the product of 987654321 and 9 is 9 -/
theorem count_eight_in_product : countDigit product 8 = 9 := by sorry

end NUMINAMATH_CALUDE_count_eight_in_product_l521_52146


namespace NUMINAMATH_CALUDE_girls_in_class_l521_52155

theorem girls_in_class (num_boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  num_boys = 16 →
  ratio_boys = 4 →
  ratio_girls = 5 →
  (num_boys * ratio_girls) / ratio_boys = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l521_52155


namespace NUMINAMATH_CALUDE_tangent_line_at_A_l521_52172

/-- The function f(x) = -x^3 + 3x --/
def f (x : ℝ) := -x^3 + 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) := -3*x^2 + 3

/-- Point A --/
def A : ℝ × ℝ := (2, -2)

/-- Equation of a line passing through A with slope m --/
def line_eq (m : ℝ) (x : ℝ) : ℝ := m*(x - A.1) + A.2

/-- Theorem: The tangent line to f(x) at A is either y = -2 or 9x + y - 16 = 0 --/
theorem tangent_line_at_A : 
  (∃ x y, line_eq (f' A.1) x = y ∧ 9*x + y - 16 = 0) ∨
  (∀ x, line_eq (f' A.1) x = -2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_A_l521_52172


namespace NUMINAMATH_CALUDE_intersection_equidistant_l521_52154

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the condition AB = CD
def equal_sides (q : Quadrilateral) : Prop :=
  dist q.A q.B = dist q.C q.D

-- Define the intersection point O of diagonals AC and BD
def intersection_point (q : Quadrilateral) : ℝ × ℝ :=
  sorry

-- Define a line passing through O
structure Line :=
  (slope : ℝ)
  (point : ℝ × ℝ)

-- Define the intersection points of a line with the quadrilateral sides
def intersection_points (q : Quadrilateral) (l : Line) :
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

-- Define the intersection of a line with BD
def intersection_with_diagonal (q : Quadrilateral) (l : Line) : ℝ × ℝ :=
  sorry

-- Main theorem
theorem intersection_equidistant (q : Quadrilateral) (l1 l2 : Line)
  (h : equal_sides q) :
  let O := intersection_point q
  let I := intersection_with_diagonal q l1
  let J := intersection_with_diagonal q l2
  dist O I = dist O J :=
sorry

end NUMINAMATH_CALUDE_intersection_equidistant_l521_52154


namespace NUMINAMATH_CALUDE_johns_running_speed_l521_52181

/-- John's running problem -/
theorem johns_running_speed
  (speed_with_dog : ℝ)
  (time_with_dog : ℝ)
  (time_alone : ℝ)
  (total_distance : ℝ)
  (h1 : speed_with_dog = 6)
  (h2 : time_with_dog = 0.5)
  (h3 : time_alone = 0.5)
  (h4 : total_distance = 5)
  (h5 : speed_with_dog * time_with_dog + speed_alone * time_alone = total_distance) :
  speed_alone = 4 := by
  sorry


end NUMINAMATH_CALUDE_johns_running_speed_l521_52181


namespace NUMINAMATH_CALUDE_parabola_translation_up_2_l521_52176

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (k : ℝ) : Parabola where
  f := λ x => p.f x + k

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola where
  f := λ x => x^2

theorem parabola_translation_up_2 :
  (translate_vertical standard_parabola 2).f = λ x => x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_up_2_l521_52176


namespace NUMINAMATH_CALUDE_asymptote_sum_l521_52101

/-- 
Given a rational function y = x / (x³ + Ax² + Bx + C) where A, B, C are integers,
if the graph has vertical asymptotes at x = -3, 0, and 2,
then A + B + C = -5
-/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 2 → 
    ∃ y : ℝ, y = x / (x^3 + A*x^2 + B*x + C)) →
  A + B + C = -5 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l521_52101


namespace NUMINAMATH_CALUDE_ratio_of_Δy_to_Δx_l521_52119

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the two points on the curve
def point1 : ℝ × ℝ := (1, 2)
def point2 (Δx : ℝ) : ℝ × ℝ := (1 + Δx, f (1 + Δx))

-- Define Δy
def Δy (Δx : ℝ) : ℝ := (point2 Δx).2 - point1.2

-- Theorem statement
theorem ratio_of_Δy_to_Δx (Δx : ℝ) (h : Δx ≠ 0) :
  Δy Δx / Δx = Δx + 2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_of_Δy_to_Δx_l521_52119


namespace NUMINAMATH_CALUDE_f_property_l521_52163

def f (x : ℝ) : ℝ := x * |x|

theorem f_property : ∀ x : ℝ, f (Real.sqrt 2 * x) = 2 * f x := by
  sorry

end NUMINAMATH_CALUDE_f_property_l521_52163


namespace NUMINAMATH_CALUDE_product_equality_implies_n_equals_six_l521_52116

theorem product_equality_implies_n_equals_six (n : ℕ) : 
  2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_implies_n_equals_six_l521_52116


namespace NUMINAMATH_CALUDE_car_bike_speed_ratio_l521_52174

/-- Proves that the ratio of the average speed of a car to the average speed of a bike is 1.8 -/
theorem car_bike_speed_ratio :
  let tractor_distance : ℝ := 575
  let tractor_time : ℝ := 25
  let car_distance : ℝ := 331.2
  let car_time : ℝ := 4
  let tractor_speed : ℝ := tractor_distance / tractor_time
  let bike_speed : ℝ := 2 * tractor_speed
  let car_speed : ℝ := car_distance / car_time
  car_speed / bike_speed = 1.8 := by
sorry


end NUMINAMATH_CALUDE_car_bike_speed_ratio_l521_52174


namespace NUMINAMATH_CALUDE_shooter_probabilities_l521_52136

-- Define the probability of hitting the target on a single shot
def p_hit : ℝ := 0.9

-- Define the number of shots
def n_shots : ℕ := 4

-- Statement 1: Probability of hitting the target on the third shot
def statement1 : Prop := p_hit = 0.9

-- Statement 2: Probability of hitting the target exactly three times
def statement2 : Prop := Nat.choose n_shots 3 * p_hit^3 * (1 - p_hit) = p_hit^3 * (1 - p_hit)

-- Statement 3: Probability of hitting the target at least once
def statement3 : Prop := 1 - (1 - p_hit)^n_shots = 1 - (1 - 0.9)^4

theorem shooter_probabilities :
  statement1 ∧ ¬statement2 ∧ statement3 :=
sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l521_52136


namespace NUMINAMATH_CALUDE_multiply_monomials_l521_52110

theorem multiply_monomials (x : ℝ) : 2*x * 5*x^2 = 10*x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_monomials_l521_52110


namespace NUMINAMATH_CALUDE_king_ace_probability_l521_52121

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (suit_count : ∀ s, (cards.filter (λ c => c.1 = s)).card = 13)
  (rank_count : ∀ r, (cards.filter (λ c => c.2 = r)).card = 4)

/-- The probability of drawing a King first and an Ace second from a standard deck. -/
def king_ace_prob (d : Deck) : ℚ :=
  4 / 663

/-- Theorem stating that the probability of drawing a King first and an Ace second
    from a standard deck is 4/663. -/
theorem king_ace_probability (d : Deck) :
  king_ace_prob d = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_king_ace_probability_l521_52121


namespace NUMINAMATH_CALUDE_line_passes_through_points_l521_52166

/-- Given a line y = (1/2)x + c passing through points (b+4, 5) and (-2, 2),
    prove that c = 3 -/
theorem line_passes_through_points (b : ℝ) :
  ∃ c : ℝ, (5 : ℝ) = (1/2 : ℝ) * (b + 4) + c ∧ (2 : ℝ) = (1/2 : ℝ) * (-2) + c ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_points_l521_52166


namespace NUMINAMATH_CALUDE_library_book_redistribution_l521_52177

theorem library_book_redistribution (total_books : Nat) (initial_stack : Nat) (new_stack : Nat)
    (h1 : total_books = 1452)
    (h2 : initial_stack = 42)
    (h3 : new_stack = 43) :
  total_books % new_stack = 33 := by
  sorry

end NUMINAMATH_CALUDE_library_book_redistribution_l521_52177


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l521_52185

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 - a 5 + a 9 - a 13 + a 17 = 117 →
  a 3 + a 15 = 234 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l521_52185


namespace NUMINAMATH_CALUDE_evaluate_expression_l521_52117

theorem evaluate_expression (c : ℕ) (h : c = 3) : (c^c - c*(c-1)^c)^c = 27 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l521_52117


namespace NUMINAMATH_CALUDE_notebook_savings_proof_l521_52120

/-- Calculates the savings on a notebook purchase with discounts -/
def calculateSavings (originalPrice : ℝ) (quantity : ℕ) (saleDiscount : ℝ) (volumeDiscount : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - saleDiscount)
  let finalPrice := discountedPrice * (1 - volumeDiscount)
  quantity * (originalPrice - finalPrice)

/-- Proves that the savings on the notebook purchase is $7.84 -/
theorem notebook_savings_proof :
  calculateSavings 3 8 0.25 0.1 = 7.84 := by
  sorry

#eval calculateSavings 3 8 0.25 0.1

end NUMINAMATH_CALUDE_notebook_savings_proof_l521_52120


namespace NUMINAMATH_CALUDE_class_overlap_difference_l521_52152

theorem class_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119)
  (h4 : geometry ≤ total)
  (h5 : biology ≤ total) :
  min geometry biology - max 0 (geometry + biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_class_overlap_difference_l521_52152


namespace NUMINAMATH_CALUDE_new_light_wattage_l521_52171

/-- Given a light with a rating of 60 watts, a new light with 12% higher wattage will have 67.2 watts. -/
theorem new_light_wattage :
  let original_wattage : ℝ := 60
  let increase_percentage : ℝ := 12
  let new_wattage : ℝ := original_wattage * (1 + increase_percentage / 100)
  new_wattage = 67.2 := by
  sorry

end NUMINAMATH_CALUDE_new_light_wattage_l521_52171


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_b_value_l521_52164

/-- A cubic polynomial with coefficient b -/
def cubic (x b : ℂ) : ℂ := x^3 - 9*x^2 + 33*x + b

/-- Predicate to check if three complex numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℂ) : Prop := b - a = c - b

/-- Theorem stating that if the roots of the cubic form an arithmetic progression
    and at least one root is non-real, then b = -15 -/
theorem cubic_roots_arithmetic_progression_b_value (b : ℝ) :
  (∃ (r₁ r₂ r₃ : ℂ), 
    (∀ x, cubic x b = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ 
    isArithmeticProgression r₁ r₂ r₃ ∧
    (r₁.im ≠ 0 ∨ r₂.im ≠ 0 ∨ r₃.im ≠ 0)) →
  b = -15 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_b_value_l521_52164


namespace NUMINAMATH_CALUDE_willy_stuffed_animals_l521_52161

/-- The number of stuffed animals Willy's mom gave him for his birthday -/
def moms_gift : ℕ := 2

/-- Willy's initial number of stuffed animals -/
def initial_count : ℕ := 10

/-- The factor by which Willy's dad increases his stuffed animal count -/
def dad_factor : ℕ := 3

/-- The total number of stuffed animals Willy has at the end -/
def final_count : ℕ := 48

theorem willy_stuffed_animals :
  initial_count + moms_gift + dad_factor * (initial_count + moms_gift) = final_count :=
by sorry

end NUMINAMATH_CALUDE_willy_stuffed_animals_l521_52161


namespace NUMINAMATH_CALUDE_distinct_parenthesizations_l521_52199

-- Define a function to represent exponentiation
def exp (a : ℕ) (b : ℕ) : ℕ := a ^ b

-- Define the five possible parenthesizations
def p1 : ℕ := exp 3 (exp 3 (exp 3 3))
def p2 : ℕ := exp 3 ((exp 3 3) ^ 3)
def p3 : ℕ := ((exp 3 3) ^ 3) ^ 3
def p4 : ℕ := (exp 3 (exp 3 3)) ^ 3
def p5 : ℕ := (exp 3 3) ^ (exp 3 3)

-- Theorem stating that there are exactly 5 distinct values
theorem distinct_parenthesizations :
  ∃! (s : Finset ℕ), s = {p1, p2, p3, p4, p5} ∧ s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_distinct_parenthesizations_l521_52199


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l521_52135

/-- Calculates the sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in an n-row triangle -/
def num_rods (n : ℕ) : ℕ := 3 * sum_of_naturals n

/-- Calculates the number of connectors in an n-row triangle -/
def num_connectors (n : ℕ) : ℕ := sum_of_naturals (n + 1)

/-- Calculates the total number of pieces in an n-row triangle -/
def total_pieces (n : ℕ) : ℕ := num_rods n + num_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 231 := by
  sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l521_52135


namespace NUMINAMATH_CALUDE_not_mysterious_consecutive_odd_squares_diff_l521_52107

/-- A positive integer that can be expressed as the difference of squares of two consecutive even numbers. -/
def MysteriousNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2 ∧ n > 0

/-- The difference of squares of two consecutive odd numbers. -/
def ConsecutiveOddSquaresDiff (k : ℤ) : ℤ :=
  (2*k + 1)^2 - (2*k - 1)^2

theorem not_mysterious_consecutive_odd_squares_diff :
  ∀ k : ℤ, ¬(MysteriousNumber (ConsecutiveOddSquaresDiff k).natAbs) :=
by sorry

end NUMINAMATH_CALUDE_not_mysterious_consecutive_odd_squares_diff_l521_52107


namespace NUMINAMATH_CALUDE_grapes_cost_proof_l521_52188

/-- The amount Alyssa paid for grapes -/
def grapes_cost : ℝ := 12.08

/-- The amount Alyssa paid for cherries -/
def cherries_cost : ℝ := 9.85

/-- The total amount Alyssa spent -/
def total_cost : ℝ := 21.93

/-- Theorem: Given the total cost and the cost of cherries, prove that the cost of grapes is correct -/
theorem grapes_cost_proof : grapes_cost = total_cost - cherries_cost := by
  sorry

end NUMINAMATH_CALUDE_grapes_cost_proof_l521_52188
