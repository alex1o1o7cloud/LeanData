import Mathlib

namespace NUMINAMATH_CALUDE_coin_flip_probability_l2922_292206

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of four coins -/
structure FourCoins :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)

/-- The total number of possible outcomes when flipping four coins -/
def totalOutcomes : ℕ := 16

/-- The number of favorable outcomes (penny heads, nickel heads, dime tails) -/
def favorableOutcomes : ℕ := 2

/-- The probability of the desired outcome -/
def desiredProbability : ℚ := 1 / 8

theorem coin_flip_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = desiredProbability := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2922_292206


namespace NUMINAMATH_CALUDE_next_simultaneous_ring_l2922_292204

def library_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def hospital_interval : ℕ := 30

theorem next_simultaneous_ring (t : ℕ) : 
  t = lcm (lcm library_interval fire_station_interval) hospital_interval → 
  t = 360 :=
by sorry

end NUMINAMATH_CALUDE_next_simultaneous_ring_l2922_292204


namespace NUMINAMATH_CALUDE_rational_irrational_relations_l2922_292208

theorem rational_irrational_relations (m n : ℚ) :
  (((m - 3) * Real.sqrt 6 + n - 3 = 0) → Real.sqrt (m * n) = 3 ∨ Real.sqrt (m * n) = -3) ∧
  ((∃ x : ℝ, m^2 = x ∧ n^2 = x ∧ (2 + Real.sqrt 3) * m - (1 - Real.sqrt 3) * n = 5) → 
   ∃ x : ℝ, m^2 = x ∧ n^2 = x ∧ x = 25/9) :=
by sorry

end NUMINAMATH_CALUDE_rational_irrational_relations_l2922_292208


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l2922_292275

theorem number_exceeds_fraction (N : ℚ) (F : ℚ) : 
  N = 24 → N = F + 15 → F = 3/8 := by sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l2922_292275


namespace NUMINAMATH_CALUDE_simplification_fraction_l2922_292279

theorem simplification_fraction (k : ℤ) : 
  let simplified := (6 * k + 18) / 6
  ∃ (a b : ℤ), simplified = a * k + b ∧ a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplification_fraction_l2922_292279


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2922_292291

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating the general term of the arithmetic sequence -/
theorem arithmetic_sequence_general_term 
  (seq : ArithmeticSequence) 
  (sum10 : seq.S 10 = 10) 
  (sum20 : seq.S 20 = 220) : 
  ∀ n, seq.a n = 2 * n - 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2922_292291


namespace NUMINAMATH_CALUDE_projectile_max_height_l2922_292229

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

-- Theorem statement
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2922_292229


namespace NUMINAMATH_CALUDE_card_value_decrease_l2922_292288

/-- Proves that if a value decreases by x% in the first year and 10% in the second year, 
    and the total decrease over two years is 37%, then x = 30. -/
theorem card_value_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 1 - 0.37 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_card_value_decrease_l2922_292288


namespace NUMINAMATH_CALUDE_halloween_houses_per_hour_l2922_292284

theorem halloween_houses_per_hour 
  (num_children : ℕ) 
  (num_hours : ℕ) 
  (treats_per_child_per_house : ℕ) 
  (total_treats : ℕ) 
  (h1 : num_children = 3)
  (h2 : num_hours = 4)
  (h3 : treats_per_child_per_house = 3)
  (h4 : total_treats = 180) :
  total_treats / (num_children * num_hours * treats_per_child_per_house) = 5 := by
  sorry

end NUMINAMATH_CALUDE_halloween_houses_per_hour_l2922_292284


namespace NUMINAMATH_CALUDE_integer_equation_solution_l2922_292205

theorem integer_equation_solution (x y : ℤ) (h : x^2 + 2 = 3*x + 75*y) :
  ∃ t : ℤ, x = 75*t + 1 ∨ x = 75*t + 2 ∨ x = 75*t + 26 ∨ x = 75*t - 23 :=
sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l2922_292205


namespace NUMINAMATH_CALUDE_fenced_rectangle_fence_length_l2922_292256

/-- A rectangular region with a fence on three sides and a wall on the fourth -/
structure FencedRectangle where
  short_side : ℝ
  long_side : ℝ
  area : ℝ
  fence_length : ℝ

/-- Properties of the fenced rectangle -/
def is_valid_fenced_rectangle (r : FencedRectangle) : Prop :=
  r.long_side = 2 * r.short_side ∧
  r.area = r.short_side * r.long_side ∧
  r.fence_length = 2 * r.short_side + r.long_side

theorem fenced_rectangle_fence_length 
  (r : FencedRectangle) 
  (h : is_valid_fenced_rectangle r) 
  (area_eq : r.area = 200) : 
  r.fence_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_fenced_rectangle_fence_length_l2922_292256


namespace NUMINAMATH_CALUDE_morse_high_school_students_l2922_292246

/-- The number of seniors at Morse High School -/
def num_seniors : ℕ := 300

/-- The percentage of seniors with cars -/
def senior_car_percentage : ℚ := 40 / 100

/-- The percentage of seniors with motorcycles -/
def senior_motorcycle_percentage : ℚ := 5 / 100

/-- The percentage of lower grade students with cars -/
def lower_car_percentage : ℚ := 10 / 100

/-- The percentage of lower grade students with motorcycles -/
def lower_motorcycle_percentage : ℚ := 3 / 100

/-- The percentage of all students with either a car or a motorcycle -/
def total_vehicle_percentage : ℚ := 20 / 100

/-- The number of students in the lower grades -/
def num_lower_grades : ℕ := 1071

theorem morse_high_school_students :
  ∃ (total_students : ℕ),
    (num_seniors + num_lower_grades = total_students) ∧
    (↑num_seniors * senior_car_percentage + 
     ↑num_seniors * senior_motorcycle_percentage +
     ↑num_lower_grades * lower_car_percentage + 
     ↑num_lower_grades * lower_motorcycle_percentage : ℚ) = 
    ↑total_students * total_vehicle_percentage :=
by sorry

end NUMINAMATH_CALUDE_morse_high_school_students_l2922_292246


namespace NUMINAMATH_CALUDE_stations_between_cities_l2922_292289

theorem stations_between_cities (n : ℕ) : 
  (((n + 2) * (n + 1)) / 2 = 132) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_stations_between_cities_l2922_292289


namespace NUMINAMATH_CALUDE_store_prices_l2922_292235

def price_X : ℝ := 80 * (1 + 0.12)
def price_Y : ℝ := price_X * (1 - 0.15)
def price_Z : ℝ := price_Y * (1 + 0.25)

theorem store_prices :
  price_X = 89.6 ∧ price_Y = 76.16 ∧ price_Z = 95.20 := by
  sorry

end NUMINAMATH_CALUDE_store_prices_l2922_292235


namespace NUMINAMATH_CALUDE_graph_translation_l2922_292212

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define k as a positive real number
variable (k : ℝ)
variable (h : k > 0)

-- State the theorem
theorem graph_translation (x y : ℝ) : 
  y = f (x + k) ↔ y = f ((x + k) - k) :=
sorry

end NUMINAMATH_CALUDE_graph_translation_l2922_292212


namespace NUMINAMATH_CALUDE_chessboard_division_exists_l2922_292297

-- Define a chessboard piece
structure ChessboardPiece where
  total_squares : ℕ
  black_squares : ℕ

-- Define a chessboard division
structure ChessboardDivision where
  piece1 : ChessboardPiece
  piece2 : ChessboardPiece

-- Define the property of being a valid chessboard division
def is_valid_division (d : ChessboardDivision) : Prop :=
  d.piece1.total_squares + d.piece2.total_squares = 64 ∧
  d.piece1.total_squares = d.piece2.total_squares + 4 ∧
  d.piece2.black_squares = d.piece1.black_squares + 4 ∧
  d.piece1.black_squares + d.piece2.black_squares = 32

-- Theorem statement
theorem chessboard_division_exists : ∃ d : ChessboardDivision, is_valid_division d :=
sorry

end NUMINAMATH_CALUDE_chessboard_division_exists_l2922_292297


namespace NUMINAMATH_CALUDE_people_joined_line_l2922_292267

theorem people_joined_line (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 30 → left = 10 → final = 25 → final - (initial - left) = 5 := by
  sorry

end NUMINAMATH_CALUDE_people_joined_line_l2922_292267


namespace NUMINAMATH_CALUDE_smaller_rectangle_perimeter_l2922_292257

/-- Given a rectangle with dimensions a × b that is divided into a smaller rectangle 
    with dimensions c × b and two squares with side length c, 
    the perimeter of the smaller rectangle is 2(c + b). -/
theorem smaller_rectangle_perimeter 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a = 3 * c) : 
  2 * (c + b) = 2 * c + 2 * b := by
sorry

end NUMINAMATH_CALUDE_smaller_rectangle_perimeter_l2922_292257


namespace NUMINAMATH_CALUDE_min_value_a_l2922_292224

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) : 
  a ≥ 4 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1/(x) + (4 - ε)/y) < 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l2922_292224


namespace NUMINAMATH_CALUDE_harry_seed_purchase_cost_l2922_292251

/-- The cost of a garden seed purchase --/
def garden_seed_cost (pumpkin_price tomato_price chili_price : ℚ) 
  (pumpkin_qty tomato_qty chili_qty : ℕ) : ℚ :=
  pumpkin_price * pumpkin_qty + tomato_price * tomato_qty + chili_price * chili_qty

/-- Theorem stating the total cost of Harry's seed purchase --/
theorem harry_seed_purchase_cost : 
  garden_seed_cost 2.5 1.5 0.9 3 4 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_harry_seed_purchase_cost_l2922_292251


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l2922_292222

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 2 * y - 4 = 0

-- Define the parabola equations
def parabola_equation_1 (x y : ℝ) : Prop := y^2 = 16 * x
def parabola_equation_2 (x y : ℝ) : Prop := x^2 = -8 * y

-- Define a parabola type
structure Parabola where
  focus : ℝ × ℝ
  is_on_line : line_equation focus.1 focus.2

-- Theorem statement
theorem parabola_standard_equation (p : Parabola) :
  (∃ x y : ℝ, parabola_equation_1 x y) ∨ (∃ x y : ℝ, parabola_equation_2 x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l2922_292222


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l2922_292276

theorem complex_on_real_axis (a : ℝ) : Complex.im ((1 + Complex.I) * (a + Complex.I)) = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l2922_292276


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2922_292247

-- Define the inequality
def inequality (x m : ℝ) : Prop :=
  |x + 1| + |x - 2| + m - 7 > 0

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  {x : ℝ | inequality x m}

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  solution_set m = Set.univ → m > 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2922_292247


namespace NUMINAMATH_CALUDE_g_expression_l2922_292271

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g implicitly using its relationship with f
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l2922_292271


namespace NUMINAMATH_CALUDE_trig_identity_l2922_292220

theorem trig_identity (α β : ℝ) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin α ^ 2 * Real.sin β ^ 2 + Real.cos α ^ 2 * Real.cos β ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2922_292220


namespace NUMINAMATH_CALUDE_fourth_side_length_l2922_292261

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- Condition that the quadrilateral is inscribed in the circle -/
  inscribed : True -- This is a placeholder for the actual condition

/-- Theorem stating that given the specific conditions, the fourth side has length 500 -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h_radius : q.radius = 300 * Real.sqrt 2)
  (h_side1 : q.side1 = 300)
  (h_side2 : q.side2 = 300)
  (h_side3 : q.side3 = 400) :
  q.side4 = 500 := by
  sorry


end NUMINAMATH_CALUDE_fourth_side_length_l2922_292261


namespace NUMINAMATH_CALUDE_y_value_proof_l2922_292281

theorem y_value_proof (y : ℝ) (h : 9 / y^2 = 3 * y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2922_292281


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2922_292248

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I.im * (a + 2 * Complex.I)).re = 0 ∧ 
  (Complex.I.im * (a + 2 * Complex.I)).im ≠ 0 → 
  a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2922_292248


namespace NUMINAMATH_CALUDE_evaluate_expression_l2922_292263

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2922_292263


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2922_292228

theorem system_of_equations_solution :
  (∀ p q : ℚ, p + q = 4 ∧ 2 * p - q = 5 → p = 3 ∧ q = 1) ∧
  (∀ v t : ℚ, 2 * v + t = 3 ∧ 3 * v - 2 * t = 3 → v = 9 / 7 ∧ t = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2922_292228


namespace NUMINAMATH_CALUDE_geometric_sum_15_l2922_292278

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_15 (a : ℕ → ℤ) :
  geometric_sequence a →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) = a n * (-2)) →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_15_l2922_292278


namespace NUMINAMATH_CALUDE_stating_count_sequences_l2922_292277

/-- 
Given positive integers n and k where 1 ≤ k < n, T(n, k) represents the number of 
sequences of k positive integers that sum to n.
-/
def T (n k : ℕ) : ℕ := sorry

/-- 
Theorem stating that T(n, k) is equal to (n-1) choose (k-1) for 1 ≤ k < n.
-/
theorem count_sequences (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  T n k = Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_stating_count_sequences_l2922_292277


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2922_292233

/-- 
Given a quadratic equation ax^2 + 6bx + 9c = 0 with zero discriminant,
prove that a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) 
  (h_quad : ∀ x, a * x^2 + 6 * b * x + 9 * c = 0)
  (h_discr : (6 * b)^2 - 4 * a * (9 * c) = 0) :
  ∃ r : ℝ, b = a * r ∧ c = b * r :=
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2922_292233


namespace NUMINAMATH_CALUDE_power_sum_equality_l2922_292283

theorem power_sum_equality : (-1 : ℤ) ^ 53 + 3 ^ (2^3 + 5^2 - 7^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2922_292283


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2922_292298

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 + 12 * x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2922_292298


namespace NUMINAMATH_CALUDE_triangle_properties_l2922_292240

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 7 →
  b = 2 →
  A = π / 3 →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  c = 3 ∧
  Real.sin B = Real.sqrt 21 / 7 ∧
  π * (a / (2 * Real.sin A))^2 = 7 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2922_292240


namespace NUMINAMATH_CALUDE_slope_of_l₃_l2922_292259

-- Define the lines and points
def l₁ (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def l₂ (y : ℝ) : Prop := y = 2
def A : ℝ × ℝ := (0, -3)

-- Define the properties of the lines and points
axiom l₁_through_A : l₁ A.1 A.2
axiom l₂_meets_l₁ : ∃ B : ℝ × ℝ, l₁ B.1 B.2 ∧ l₂ B.2
axiom l₃_positive_slope : ∃ m : ℝ, m > 0 ∧ ∀ x y : ℝ, y - A.2 = m * (x - A.1)
axiom l₃_through_A : ∀ x y : ℝ, y - A.2 = (y - A.2) / (x - A.1) * (x - A.1)
axiom l₃_meets_l₂ : ∃ C : ℝ × ℝ, l₂ C.2 ∧ C.2 - A.2 = (C.2 - A.2) / (C.1 - A.1) * (C.1 - A.1)

-- Define the area of triangle ABC
axiom triangle_area : ∃ B C : ℝ × ℝ, 
  l₁ B.1 B.2 ∧ l₂ B.2 ∧ l₂ C.2 ∧ C.2 - A.2 = (C.2 - A.2) / (C.1 - A.1) * (C.1 - A.1) ∧
  1/2 * |(B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)| = 10

-- Theorem statement
theorem slope_of_l₃ : 
  ∃ m : ℝ, m = 5/4 ∧ ∀ x y : ℝ, y - A.2 = m * (x - A.1) :=
sorry

end NUMINAMATH_CALUDE_slope_of_l₃_l2922_292259


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achievable_l2922_292227

theorem quadratic_minimum (x : ℝ) : 7 * x^2 - 28 * x + 1425 ≥ 1397 :=
sorry

theorem quadratic_minimum_achievable : ∃ x : ℝ, 7 * x^2 - 28 * x + 1425 = 1397 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achievable_l2922_292227


namespace NUMINAMATH_CALUDE_water_displaced_by_sphere_l2922_292221

/-- The volume of water displaced by a completely submerged sphere -/
theorem water_displaced_by_sphere (diameter : ℝ) (volume_displaced : ℝ) :
  diameter = 8 →
  volume_displaced = (4/3) * Real.pi * (diameter/2)^3 →
  volume_displaced = (256/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_by_sphere_l2922_292221


namespace NUMINAMATH_CALUDE_solution_set_y_geq_4_min_value_reciprocal_sum_l2922_292285

-- Define the quadratic function
def y (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Part 1
theorem solution_set_y_geq_4 (a b : ℝ) :
  (∀ x : ℝ, y a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, y a b x ≥ 4 ↔ x = 1) :=
sorry

-- Part 2
theorem min_value_reciprocal_sum (a b : ℝ) :
  a > 0 →
  b > 0 →
  y a b 1 = 2 →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → y a' b' 1 = 2 → 1/a' + 4/b' ≥ 1/a + 4/b) →
  1/a + 4/b = 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_y_geq_4_min_value_reciprocal_sum_l2922_292285


namespace NUMINAMATH_CALUDE_intersection_product_equality_l2922_292207

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary operations and relations
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (on_arc : Point → Point → Point → Circle → Prop)
variable (meets_at : Point → Point → Circle → Point → Prop)
variable (intersect_at : Point → Point → Point → Point → Point → Prop)
variable (length : Point → Point → ℝ)

-- Define the given points and circles
variable (O₁ O₂ : Circle)
variable (A B R T C D Q P E F : Point)

-- State the theorem
theorem intersection_product_equality
  (h1 : intersect O₁ O₂ A B)
  (h2 : on_arc A B R O₁)
  (h3 : on_arc A B T O₂)
  (h4 : meets_at A R O₂ C)
  (h5 : meets_at B R O₂ D)
  (h6 : meets_at A T O₁ Q)
  (h7 : meets_at B T O₁ P)
  (h8 : intersect_at P R T D E)
  (h9 : intersect_at Q R T C F) :
  length A E * length B T * length B R = length B F * length A T * length A R :=
sorry

end NUMINAMATH_CALUDE_intersection_product_equality_l2922_292207


namespace NUMINAMATH_CALUDE_student_A_received_A_grade_l2922_292293

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the grade levels
inductive Grade : Type
| A : Grade
| B : Grade
| C : Grade

-- Define a function to represent the actual grades received
def actual_grade : Student → Grade := sorry

-- Define a function to represent the correctness of predictions
def prediction_correct : Student → Prop := sorry

-- Theorem statement
theorem student_A_received_A_grade :
  -- Only one student received an A grade
  (∃! s : Student, actual_grade s = Grade.A) →
  -- A's prediction: C can only get a B or C
  (actual_grade Student.C ≠ Grade.A) →
  -- B's prediction: B will get an A
  (actual_grade Student.B = Grade.A) →
  -- C's prediction: C agrees with A's prediction
  (actual_grade Student.C ≠ Grade.A) →
  -- Only one prediction was inaccurate
  (∃! s : Student, ¬prediction_correct s) →
  -- Student A received an A grade
  actual_grade Student.A = Grade.A :=
sorry

end NUMINAMATH_CALUDE_student_A_received_A_grade_l2922_292293


namespace NUMINAMATH_CALUDE_roses_in_garden_l2922_292244

/-- Proves that the number of roses in the garden before cutting is equal to
    the final number of roses in the vase minus the initial number of roses in the vase. -/
theorem roses_in_garden (initial_vase : ℕ) (cut_from_garden : ℕ) (final_vase : ℕ)
  (h1 : initial_vase = 7)
  (h2 : cut_from_garden = 13)
  (h3 : final_vase = 20)
  (h4 : final_vase = initial_vase + cut_from_garden) :
  cut_from_garden = final_vase - initial_vase :=
by sorry

end NUMINAMATH_CALUDE_roses_in_garden_l2922_292244


namespace NUMINAMATH_CALUDE_cosine_sine_sum_zero_l2922_292274

theorem cosine_sine_sum_zero : 
  Real.cos (-11 * π / 6) + Real.sin (11 * π / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_zero_l2922_292274


namespace NUMINAMATH_CALUDE_horner_method_v2_l2922_292250

def f (x : ℝ) : ℝ := 2*x^5 - x^4 + 2*x^2 + 5*x + 3

def horner_v2 (x v0 v1 : ℝ) : ℝ := v1 * x

theorem horner_method_v2 (x v0 v1 : ℝ) (hx : x = 3) (hv0 : v0 = 2) (hv1 : v1 = 5) :
  horner_v2 x v0 v1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2922_292250


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2922_292201

theorem max_sum_arithmetic_sequence (x y : ℝ) (h : x^2 + y^2 = 4) :
  (∃ (z : ℝ), (3/4) * (x + 3*y) ≤ z) ∧ (∀ (z : ℝ), (3/4) * (x + 3*y) ≤ z → 3 * Real.sqrt 10 / 2 ≤ z) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2922_292201


namespace NUMINAMATH_CALUDE_vinegar_percentage_second_brand_l2922_292299

/-- Calculates the vinegar percentage in the second brand of Italian dressing -/
theorem vinegar_percentage_second_brand 
  (total_volume : ℝ) 
  (desired_vinegar_percentage : ℝ) 
  (first_brand_volume : ℝ) 
  (second_brand_volume : ℝ) 
  (first_brand_vinegar_percentage : ℝ)
  (h1 : total_volume = 320)
  (h2 : desired_vinegar_percentage = 11)
  (h3 : first_brand_volume = 128)
  (h4 : second_brand_volume = 128)
  (h5 : first_brand_vinegar_percentage = 8) :
  ∃ (second_brand_vinegar_percentage : ℝ),
    second_brand_vinegar_percentage = 19.5 ∧
    (first_brand_volume * first_brand_vinegar_percentage / 100 + 
     second_brand_volume * second_brand_vinegar_percentage / 100) / total_volume * 100 = 
    desired_vinegar_percentage :=
by sorry

end NUMINAMATH_CALUDE_vinegar_percentage_second_brand_l2922_292299


namespace NUMINAMATH_CALUDE_purse_percentage_l2922_292296

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of pennies in Samantha's purse -/
def num_pennies : ℕ := 2

/-- The number of nickels in Samantha's purse -/
def num_nickels : ℕ := 3

/-- The number of dimes in Samantha's purse -/
def num_dimes : ℕ := 1

/-- The number of quarters in Samantha's purse -/
def num_quarters : ℕ := 2

/-- The total value of coins in Samantha's purse in cents -/
def total_cents : ℕ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

/-- The percentage of one dollar in Samantha's purse -/
theorem purse_percentage : (total_cents : ℚ) / 100 = 77 / 100 := by
  sorry

end NUMINAMATH_CALUDE_purse_percentage_l2922_292296


namespace NUMINAMATH_CALUDE_special_cubic_e_value_l2922_292231

/-- A cubic polynomial with specific properties -/
structure SpecialCubic where
  d : ℝ
  e : ℝ
  zeros_mean_prod : (- d / 9) = 2 * (-4)
  coeff_sum_y_intercept : 3 + d + e + 12 = 12

/-- The value of e in the special cubic polynomial is -75 -/
theorem special_cubic_e_value (p : SpecialCubic) : p.e = -75 := by
  sorry

end NUMINAMATH_CALUDE_special_cubic_e_value_l2922_292231


namespace NUMINAMATH_CALUDE_garden_transformation_l2922_292249

/-- Represents a rectangular garden --/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden --/
structure SquareGarden where
  side : ℝ

/-- Calculates the perimeter of a rectangular garden --/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.width)

/-- Calculates the area of a rectangular garden --/
def areaRectangular (garden : RectangularGarden) : ℝ :=
  garden.length * garden.width

/-- Calculates the area of a square garden --/
def areaSquare (garden : SquareGarden) : ℝ :=
  garden.side * garden.side

/-- Theorem: Changing a 60x20 rectangular garden to a square with the same perimeter
    results in a 40x40 square garden and increases the area by 400 square feet --/
theorem garden_transformation (original : RectangularGarden) 
    (h1 : original.length = 60)
    (h2 : original.width = 20) :
    ∃ (new : SquareGarden),
      perimeter original = 4 * new.side ∧
      new.side = 40 ∧
      areaSquare new - areaRectangular original = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_transformation_l2922_292249


namespace NUMINAMATH_CALUDE_scientific_notation_digits_l2922_292241

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log n 10).succ

/-- Conversion from scientific notation to standard form -/
def scientific_to_standard (mantissa : ℚ) (exponent : ℤ) : ℚ :=
  mantissa * (10 : ℚ) ^ exponent

theorem scientific_notation_digits :
  let mantissa : ℚ := 721 / 100
  let exponent : ℤ := 11
  let standard_form := scientific_to_standard mantissa exponent
  num_digits (Nat.floor standard_form) = 12 := by
sorry

end NUMINAMATH_CALUDE_scientific_notation_digits_l2922_292241


namespace NUMINAMATH_CALUDE_octal_724_equals_468_l2922_292203

/-- Converts an octal number represented as a list of digits to its decimal equivalent. -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the number -/
def octal_number : List Nat := [4, 2, 7]

theorem octal_724_equals_468 :
  octal_to_decimal octal_number = 468 := by
  sorry

end NUMINAMATH_CALUDE_octal_724_equals_468_l2922_292203


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l2922_292260

/-- Proves that given an initial monthly salary of $6000 and total earnings of $259200 after 3 years,
    with a salary increase occurring after 1 year, the percentage increase in salary is 30%. -/
theorem salary_increase_percentage 
  (initial_salary : ℝ) 
  (total_earnings : ℝ) 
  (increase_percentage : ℝ) :
  initial_salary = 6000 →
  total_earnings = 259200 →
  total_earnings = 12 * initial_salary + 24 * (initial_salary + initial_salary * increase_percentage / 100) →
  increase_percentage = 30 := by
  sorry

#check salary_increase_percentage

end NUMINAMATH_CALUDE_salary_increase_percentage_l2922_292260


namespace NUMINAMATH_CALUDE_probability_two_shirts_one_shorts_one_socks_l2922_292236

def num_shirts : ℕ := 3
def num_shorts : ℕ := 7
def num_socks : ℕ := 4
def num_selected : ℕ := 4

def total_articles : ℕ := num_shirts + num_shorts + num_socks

def favorable_outcomes : ℕ := (num_shirts.choose 2) * (num_shorts.choose 1) * (num_socks.choose 1)
def total_outcomes : ℕ := total_articles.choose num_selected

theorem probability_two_shirts_one_shorts_one_socks :
  (favorable_outcomes : ℚ) / total_outcomes = 84 / 1001 :=
sorry

end NUMINAMATH_CALUDE_probability_two_shirts_one_shorts_one_socks_l2922_292236


namespace NUMINAMATH_CALUDE_chloe_min_score_l2922_292272

/-- The minimum score needed on the fifth test to achieve a given average -/
def min_score_for_average (test1 test2 test3 test4 : ℚ) (required_avg : ℚ) : ℚ :=
  5 * required_avg - (test1 + test2 + test3 + test4)

/-- Proof that Chloe needs at least 86% on her fifth test -/
theorem chloe_min_score :
  let test1 : ℚ := 84
  let test2 : ℚ := 87
  let test3 : ℚ := 78
  let test4 : ℚ := 90
  let required_avg : ℚ := 85
  min_score_for_average test1 test2 test3 test4 required_avg = 86 := by
  sorry

#eval min_score_for_average 84 87 78 90 85

end NUMINAMATH_CALUDE_chloe_min_score_l2922_292272


namespace NUMINAMATH_CALUDE_expression_value_l2922_292253

theorem expression_value (n m : ℤ) (h : m = 2 * n^2 + n + 1) :
  8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2922_292253


namespace NUMINAMATH_CALUDE_light_bulb_probability_l2922_292243

/-- The probability of selecting a light bulb from Factory A and it passing the quality test -/
theorem light_bulb_probability (p_A : ℝ) (p_B : ℝ) (pass_A : ℝ) (pass_B : ℝ) 
  (h1 : p_A = 0.7) 
  (h2 : p_B = 0.3) 
  (h3 : p_A + p_B = 1) 
  (h4 : pass_A = 0.95) 
  (h5 : pass_B = 0.8) :
  p_A * pass_A = 0.665 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l2922_292243


namespace NUMINAMATH_CALUDE_division_preserves_inequality_l2922_292234

theorem division_preserves_inequality (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_preserves_inequality_l2922_292234


namespace NUMINAMATH_CALUDE_average_weight_of_a_and_b_l2922_292239

theorem average_weight_of_a_and_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 47 →
  b = 39 →
  (a + b) / 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_a_and_b_l2922_292239


namespace NUMINAMATH_CALUDE_pizza_piece_volume_l2922_292273

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/3 →
  diameter = 18 →
  num_pieces = 18 →
  (π * (diameter/2)^2 * thickness) / num_pieces = 3*π/2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_volume_l2922_292273


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l2922_292270

theorem smallest_five_digit_mod_9 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n % 9 = 5 → n ≥ 10000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l2922_292270


namespace NUMINAMATH_CALUDE_good_time_more_prevalent_l2922_292216

-- Define the clock hands
structure ClockHand where
  angle : ℝ
  speed : ℝ  -- angular speed in radians per hour

-- Define the clock
structure Clock where
  hour : ClockHand
  minute : ClockHand
  second : ClockHand

-- Define good time
def isGoodTime (c : Clock) : Prop :=
  ∃ (d : ℝ), (c.hour.angle - d) * (c.minute.angle - d) ≥ 0 ∧
             (c.hour.angle - d) * (c.second.angle - d) ≥ 0 ∧
             (c.minute.angle - d) * (c.second.angle - d) ≥ 0

-- Define the duration of good time in a day
def goodTimeDuration : ℝ :=
  sorry

-- Define the duration of bad time in a day
def badTimeDuration : ℝ :=
  sorry

-- The theorem to prove
theorem good_time_more_prevalent : goodTimeDuration > badTimeDuration := by
  sorry

end NUMINAMATH_CALUDE_good_time_more_prevalent_l2922_292216


namespace NUMINAMATH_CALUDE_fraction_equality_l2922_292254

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a - b) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2922_292254


namespace NUMINAMATH_CALUDE_tangent_line_m_values_l2922_292280

/-- The equation of a line that may be tangent to a circle -/
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (m : ℝ) : Prop := 
  ∃ x y : ℝ, line_equation x y m ∧ circle_equation x y

/-- Theorem stating the possible values of m when the line is tangent to the circle -/
theorem tangent_line_m_values :
  ∀ m : ℝ, is_tangent m → m = -3 ∨ m = -13 := by sorry

end NUMINAMATH_CALUDE_tangent_line_m_values_l2922_292280


namespace NUMINAMATH_CALUDE_original_number_proof_l2922_292268

theorem original_number_proof (x : ℝ) : x * 1.4 = 1680 ↔ x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2922_292268


namespace NUMINAMATH_CALUDE_min_value_w_l2922_292209

/-- The minimum value of w = 2x^2 + 3y^2 + 8x - 5y + 30 is 26.25 -/
theorem min_value_w :
  (∀ x y : ℝ, 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 ≥ 26.25) ∧
  (∃ x y : ℝ, 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 = 26.25) := by
  sorry

end NUMINAMATH_CALUDE_min_value_w_l2922_292209


namespace NUMINAMATH_CALUDE_product_of_solutions_l2922_292200

theorem product_of_solutions (x : ℝ) : 
  (45 = -x^2 - 4*x) → (∃ α β : ℝ, α * β = -45 ∧ 45 = -α^2 - 4*α ∧ 45 = -β^2 - 4*β) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2922_292200


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l2922_292237

/-- Calculates the total number of people who can sit in a bus with the given seating arrangement. -/
theorem bus_seating_capacity
  (left_seats : ℕ)
  (right_seats_difference : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : right_seats_difference = 3)
  (h3 : people_per_seat = 3)
  (h4 : back_seat_capacity = 10) :
  left_seats * people_per_seat +
  (left_seats - right_seats_difference) * people_per_seat +
  back_seat_capacity = 91 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l2922_292237


namespace NUMINAMATH_CALUDE_coin_problem_l2922_292252

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

theorem coin_problem (p d n q : ℕ) : 
  p + n + d + q = 12 →  -- Total number of coins
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 →  -- At least one of each type
  q = 2 * d →  -- Twice as many quarters as dimes
  p * penny + n * nickel + d * dime + q * quarter = 128 →  -- Total value in cents
  n = 3 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l2922_292252


namespace NUMINAMATH_CALUDE_all_statements_imply_target_l2922_292266

theorem all_statements_imply_target (p q r : Prop) :
  ((¬p ∧ ¬r ∧ q) → ((p ∧ q) → ¬r)) ∧
  ((p ∧ ¬r ∧ ¬q) → ((p ∧ q) → ¬r)) ∧
  ((¬p ∧ r ∧ q) → ((p ∧ q) → ¬r)) ∧
  ((p ∧ r ∧ ¬q) → ((p ∧ q) → ¬r)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_imply_target_l2922_292266


namespace NUMINAMATH_CALUDE_sequence_formulas_l2922_292202

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = 1

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, b (n + 1) = q * b n

theorem sequence_formulas
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a235 : a 2 + a 3 = a 5)
  (h_a4b12 : a 4 = 4 * b 1 - b 2)
  (h_b3a35 : b 3 = a 3 + a 5) :
  (∀ n, a n = n) ∧ (∀ n, b n = 2^n) :=
sorry

end NUMINAMATH_CALUDE_sequence_formulas_l2922_292202


namespace NUMINAMATH_CALUDE_triangle_centroid_incenter_relation_l2922_292286

open Real

-- Define a structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define functions to calculate centroid and incenter
def centroid (t : Triangle) : ℝ × ℝ := sorry

def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a function to calculate squared distance between two points
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_centroid_incenter_relation :
  ∃ k : ℝ, ∀ t : Triangle, ∀ P : ℝ × ℝ,
    let G := centroid t
    let I := incenter t
    dist_squared P t.A + dist_squared P t.B + dist_squared P t.C + dist_squared P I =
    k * (dist_squared P G + dist_squared G t.A + dist_squared G t.B + dist_squared G t.C + dist_squared G I) :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_incenter_relation_l2922_292286


namespace NUMINAMATH_CALUDE_lot_worth_l2922_292213

/-- Given a lot where a man owns half and sells a tenth of his share for $460, 
    prove that the worth of the entire lot is $9200. -/
theorem lot_worth (man_share : ℚ) (sold_fraction : ℚ) (sold_amount : ℕ) :
  man_share = 1/2 →
  sold_fraction = 1/10 →
  sold_amount = 460 →
  (sold_amount / sold_fraction) / man_share = 9200 := by
  sorry

end NUMINAMATH_CALUDE_lot_worth_l2922_292213


namespace NUMINAMATH_CALUDE_g_zero_iff_a_eq_seven_fifths_l2922_292223

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(a) = 0 if and only if a = 7/5 -/
theorem g_zero_iff_a_eq_seven_fifths :
  ∀ a : ℝ, g a = 0 ↔ a = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_iff_a_eq_seven_fifths_l2922_292223


namespace NUMINAMATH_CALUDE_y_derivative_l2922_292292

noncomputable def y (x : ℝ) : ℝ := (1/4) * Real.log (abs (Real.tanh (x/2))) - (1/4) * Real.log ((3 + Real.cosh x) / Real.sinh x)

theorem y_derivative (x : ℝ) : deriv y x = 1 / (2 * Real.sinh x) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l2922_292292


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2922_292262

/-- Given a hyperbola with imaginary axis length 2 and focal distance 2√3,
    prove that the equation of its asymptotes is y = ±(√2/2)x -/
theorem hyperbola_asymptotes 
  (b : ℝ) 
  (c : ℝ) 
  (h1 : b = 1)  -- half of the imaginary axis length
  (h2 : c = Real.sqrt 3)  -- half of the focal distance
  : ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ 
    (∀ (x y : ℝ), (y = k * x ∨ y = -k * x) ↔ 
      (x^2 / (c^2 - b^2) - y^2 / b^2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2922_292262


namespace NUMINAMATH_CALUDE_line_through_points_l2922_292219

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given distinct vectors a and b, and a scalar k, 
    prove that k*a + (1/2)*b lies on the line through a and b 
    if and only if k = 1/2 -/
theorem line_through_points (a b : V) (k : ℝ) 
    (h_distinct : a ≠ b) : 
    (∃ t : ℝ, k • a + (1/2) • b = a + t • (b - a)) ↔ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2922_292219


namespace NUMINAMATH_CALUDE_stationery_solution_l2922_292210

/-- Represents a pack of stationery -/
structure StationeryPack where
  sheets : ℕ
  envelopes : ℕ

/-- The problem setup -/
def stationeryProblem (pack : StationeryPack) : Prop :=
  ∃ (jack_leftover_sheets tom_leftover_envelopes : ℕ),
    -- Jack uses all envelopes and has 90 sheets left
    pack.sheets - 2 * pack.envelopes = jack_leftover_sheets ∧
    jack_leftover_sheets = 90 ∧
    -- Tom uses all sheets and has 30 envelopes left
    pack.sheets = 4 * (pack.envelopes - tom_leftover_envelopes) ∧
    tom_leftover_envelopes = 30

/-- The theorem to prove -/
theorem stationery_solution :
  ∃ (pack : StationeryPack),
    stationeryProblem pack ∧
    pack.sheets = 120 ∧
    pack.envelopes = 30 := by
  sorry

end NUMINAMATH_CALUDE_stationery_solution_l2922_292210


namespace NUMINAMATH_CALUDE_wendy_albums_l2922_292258

theorem wendy_albums (total_pictures : ℕ) (first_album : ℕ) (pictures_per_album : ℕ) 
  (h1 : total_pictures = 79)
  (h2 : first_album = 44)
  (h3 : pictures_per_album = 7) :
  (total_pictures - first_album) / pictures_per_album = 5 :=
by sorry

end NUMINAMATH_CALUDE_wendy_albums_l2922_292258


namespace NUMINAMATH_CALUDE_min_value_sum_product_l2922_292269

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l2922_292269


namespace NUMINAMATH_CALUDE_eccentricity_of_conic_l2922_292287

/-- The conic section defined by the equation 6x^2 + 4xy + 9y^2 = 20 -/
def conic_section (x y : ℝ) : Prop :=
  6 * x^2 + 4 * x * y + 9 * y^2 = 20

/-- The eccentricity of a conic section -/
def eccentricity (c : (ℝ → ℝ → Prop)) : ℝ := sorry

/-- Theorem: The eccentricity of the given conic section is √2/2 -/
theorem eccentricity_of_conic : eccentricity conic_section = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_of_conic_l2922_292287


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_from_medians_l2922_292217

/-- A right triangle with specific median lengths has a hypotenuse of 3√51 -/
theorem right_triangle_hypotenuse_from_medians 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (median1 : (b^2 + (a/2)^2) = 7^2) 
  (median2 : (a^2 + (b/2)^2) = (3*Real.sqrt 13)^2) : 
  c = 3 * Real.sqrt 51 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuse_from_medians_l2922_292217


namespace NUMINAMATH_CALUDE_negation_equivalence_l2922_292215

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) > x^2) ↔ 
  (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2922_292215


namespace NUMINAMATH_CALUDE_relationship_l2922_292238

-- Define the real numbers a, b, and c
variable (a b c : ℝ)

-- Define the conditions
axiom eq_a : 2 * a^3 + a = 2
axiom eq_b : b * Real.log b / Real.log 2 = 1
axiom eq_c : c * Real.log c / Real.log 5 = 1

-- State the theorem to be proved
theorem relationship : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_l2922_292238


namespace NUMINAMATH_CALUDE_ice_cream_stacking_permutations_l2922_292294

theorem ice_cream_stacking_permutations : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_stacking_permutations_l2922_292294


namespace NUMINAMATH_CALUDE_factorial_divisibility_power_of_two_l2922_292232

theorem factorial_divisibility_power_of_two (n : ℕ) : 
  (∃ k : ℕ, n = 2^k) ↔ (n.factorial % 2^(n-1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_power_of_two_l2922_292232


namespace NUMINAMATH_CALUDE_mean_motorcycles_rainy_days_l2922_292211

def sunny_car_counts : List ℝ := [30, 14, 14, 21, 25]
def sunny_motorcycle_counts : List ℝ := [5, 2, 4, 1, 3]
def rainy_car_counts : List ℝ := [40, 20, 17, 31, 30]
def rainy_motorcycle_counts : List ℝ := [2, 1, 1, 0, 2]

theorem mean_motorcycles_rainy_days :
  (rainy_motorcycle_counts.sum / rainy_motorcycle_counts.length : ℝ) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_motorcycles_rainy_days_l2922_292211


namespace NUMINAMATH_CALUDE_new_arithmetic_mean_l2922_292295

def original_count : ℕ := 60
def original_mean : ℝ := 45
def removed_numbers : List ℝ := [48, 58, 62]

theorem new_arithmetic_mean :
  let original_sum : ℝ := original_count * original_mean
  let removed_sum : ℝ := removed_numbers.sum
  let new_count : ℕ := original_count - removed_numbers.length
  let new_sum : ℝ := original_sum - removed_sum
  new_sum / new_count = 44.42 := by sorry

end NUMINAMATH_CALUDE_new_arithmetic_mean_l2922_292295


namespace NUMINAMATH_CALUDE_cosine_function_properties_l2922_292282

theorem cosine_function_properties (a b c d : ℝ) (ha : a > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (a = 4) →
  (2 * Real.pi / b = Real.pi / 2) →
  (b = 4 ∧ ∀ c₁ c₂, ∃ b', 
    (∀ x, ∃ y, y = a * Real.cos (b' * x + c₁) + d) ∧
    (∀ x, ∃ y, y = a * Real.cos (b' * x + c₂) + d) ∧
    (2 * Real.pi / b' = Real.pi / 2)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l2922_292282


namespace NUMINAMATH_CALUDE_min_voters_to_win_is_24_l2922_292242

/-- Represents the voting structure and outcome of a giraffe beauty contest. -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  (total_voters_eq : total_voters = num_districts * sections_per_district * voters_per_section)
  (num_districts_eq : num_districts = 5)
  (sections_per_district_eq : sections_per_district = 7)
  (voters_per_section_eq : voters_per_section = 3)

/-- Calculates the minimum number of voters required to win the contest. -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win := contest.sections_per_district / 2 + 1
  let voters_to_win_section := contest.voters_per_section / 2 + 1
  districts_to_win * sections_to_win * voters_to_win_section

/-- Theorem stating that the minimum number of voters required to win the contest is 24. -/
theorem min_voters_to_win_is_24 (contest : GiraffeContest) :
  min_voters_to_win contest = 24 := by
  sorry

#eval min_voters_to_win {
  total_voters := 105,
  num_districts := 5,
  sections_per_district := 7,
  voters_per_section := 3,
  total_voters_eq := rfl,
  num_districts_eq := rfl,
  sections_per_district_eq := rfl,
  voters_per_section_eq := rfl
}

end NUMINAMATH_CALUDE_min_voters_to_win_is_24_l2922_292242


namespace NUMINAMATH_CALUDE_book_length_l2922_292214

theorem book_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → area = 6 → area = length * width → length = 2 := by
sorry

end NUMINAMATH_CALUDE_book_length_l2922_292214


namespace NUMINAMATH_CALUDE_jasons_correct_answers_l2922_292226

theorem jasons_correct_answers
  (total_problems : ℕ)
  (points_for_correct : ℕ)
  (points_for_incorrect : ℕ)
  (final_score : ℕ)
  (h1 : total_problems = 12)
  (h2 : points_for_correct = 4)
  (h3 : points_for_incorrect = 1)
  (h4 : final_score = 33) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_problems ∧
    points_for_correct * correct_answers -
    points_for_incorrect * (total_problems - correct_answers) = final_score ∧
    correct_answers = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_jasons_correct_answers_l2922_292226


namespace NUMINAMATH_CALUDE_product_of_four_numbers_l2922_292225

theorem product_of_four_numbers (a b c d : ℝ) : 
  ((a + b + c + d) / 4 = 7.1) →
  (2.5 * a = b - 1.2) →
  (b - 1.2 = c + 4.8) →
  (c + 4.8 = 0.25 * d) →
  (a * b * c * d = 49.6) := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l2922_292225


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2922_292218

theorem chocolate_distribution (total_bars : ℕ) (num_people : ℕ) 
  (h1 : total_bars = 60) 
  (h2 : num_people = 5) : 
  let bars_per_person := total_bars / num_people
  let person1_final := bars_per_person - bars_per_person / 2
  let person2_final := bars_per_person + 2
  let person3_final := bars_per_person - 2
  let person4_final := bars_per_person
  person2_final + person3_final + person4_final = 36 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2922_292218


namespace NUMINAMATH_CALUDE_inequality_solution_l2922_292245

theorem inequality_solution (x : ℝ) : (x + 2) / (x + 4) ≤ 3 ↔ -5 < x ∧ x < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2922_292245


namespace NUMINAMATH_CALUDE_angle_trigonometry_l2922_292255

theorem angle_trigonometry (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α * Real.tan β = 13/7) (h4 : Real.sin (α - β) = Real.sqrt 5 / 3) :
  Real.cos (α - β) = 2/3 ∧ Real.cos (α + β) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l2922_292255


namespace NUMINAMATH_CALUDE_chocolate_candy_price_difference_l2922_292265

/-- Proves the difference in cost between a discounted chocolate and a taxed candy bar --/
theorem chocolate_candy_price_difference 
  (initial_money : ℝ)
  (chocolate_price gum_price candy_price soda_price : ℝ)
  (chocolate_discount gum_candy_tax : ℝ) :
  initial_money = 20 →
  chocolate_price = 7 →
  gum_price = 3 →
  candy_price = 2 →
  soda_price = 1.5 →
  chocolate_discount = 0.15 →
  gum_candy_tax = 0.08 →
  chocolate_price * (1 - chocolate_discount) - (candy_price * (1 + gum_candy_tax)) = 3.95 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_candy_price_difference_l2922_292265


namespace NUMINAMATH_CALUDE_grid_size_for_2017_colored_squares_l2922_292290

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ

/-- The number of colored squares on the two longest diagonals of a square grid -/
def coloredSquares (grid : SquareGrid) : ℕ := 2 * grid.size - 1

theorem grid_size_for_2017_colored_squares :
  ∃ (grid : SquareGrid), coloredSquares grid = 2017 ∧ grid.size = 1009 :=
sorry

end NUMINAMATH_CALUDE_grid_size_for_2017_colored_squares_l2922_292290


namespace NUMINAMATH_CALUDE_ceiling_painting_fraction_l2922_292264

def total_ceilings : ℕ := 28
def first_week_ceilings : ℕ := 12
def remaining_ceilings : ℕ := 13

theorem ceiling_painting_fraction :
  (total_ceilings - first_week_ceilings - remaining_ceilings) / first_week_ceilings = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_painting_fraction_l2922_292264


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l2922_292230

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 4*a

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x, |f a x| ≤ 2) ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l2922_292230
