import Mathlib

namespace NUMINAMATH_CALUDE_five_variable_inequality_two_is_smallest_constant_l2934_293440

theorem five_variable_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 :=
by sorry

theorem two_is_smallest_constant :
  ∀ ε > 0, ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
    Real.sqrt (e / (a + b + c + d)) < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_five_variable_inequality_two_is_smallest_constant_l2934_293440


namespace NUMINAMATH_CALUDE_jacks_total_money_l2934_293471

/-- Calculates the total amount of money in dollars given an amount in dollars and euros, with a fixed exchange rate. -/
def total_money_in_dollars (dollars : ℕ) (euros : ℕ) (exchange_rate : ℕ) : ℕ :=
  dollars + euros * exchange_rate

/-- Theorem stating that Jack's total money in dollars is 117 given the problem conditions. -/
theorem jacks_total_money :
  total_money_in_dollars 45 36 2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_jacks_total_money_l2934_293471


namespace NUMINAMATH_CALUDE_animals_in_field_l2934_293485

/-- The number of animals running through a field -/
def total_animals (dog : ℕ) (cats : ℕ) (rabbits_per_cat : ℕ) (hares_per_rabbit : ℕ) : ℕ :=
  dog + cats + (cats * rabbits_per_cat) + (cats * rabbits_per_cat * hares_per_rabbit)

/-- Theorem stating the total number of animals in the field -/
theorem animals_in_field : total_animals 1 4 2 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_animals_in_field_l2934_293485


namespace NUMINAMATH_CALUDE_julia_spent_114_l2934_293454

/-- The total amount Julia spent on food for her animals -/
def total_spent (weekly_total : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) (rabbit_food_cost : ℕ) : ℕ :=
  let parrot_food_cost := weekly_total - rabbit_food_cost
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost

/-- Proof that Julia spent $114 on food for her animals -/
theorem julia_spent_114 :
  total_spent 30 5 3 12 = 114 := by
  sorry

end NUMINAMATH_CALUDE_julia_spent_114_l2934_293454


namespace NUMINAMATH_CALUDE_circles_intersect_l2934_293453

theorem circles_intersect : ∃ (x y : ℝ), 
  ((x + 1)^2 + (y + 2)^2 = 4) ∧ ((x - 1)^2 + (y + 1)^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l2934_293453


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l2934_293402

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l2934_293402


namespace NUMINAMATH_CALUDE_chicken_problem_l2934_293423

theorem chicken_problem (total chickens_colten : ℕ) 
  (h_total : total = 383)
  (h_colten : chickens_colten = 37) : 
  ∃ (chickens_skylar chickens_quentin : ℕ),
    chickens_skylar = 3 * chickens_colten - 4 ∧
    chickens_quentin = 2 * chickens_skylar + 32 ∧
    chickens_quentin + chickens_skylar + chickens_colten = total :=
by
  sorry

#check chicken_problem

end NUMINAMATH_CALUDE_chicken_problem_l2934_293423


namespace NUMINAMATH_CALUDE_orange_cost_solution_l2934_293487

/-- Calculates the cost of an orange given the initial quantities, apple cost, and final earnings -/
def orange_cost (initial_apples initial_oranges : ℕ) (apple_cost : ℚ) 
  (final_apples final_oranges : ℕ) (total_earnings : ℚ) : ℚ :=
  let apples_sold := initial_apples - final_apples
  let oranges_sold := initial_oranges - final_oranges
  let apple_earnings := apples_sold * apple_cost
  let orange_earnings := total_earnings - apple_earnings
  orange_earnings / oranges_sold

theorem orange_cost_solution :
  orange_cost 50 40 (4/5) 10 6 49 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_solution_l2934_293487


namespace NUMINAMATH_CALUDE_abs_square_not_always_equal_to_value_l2934_293401

theorem abs_square_not_always_equal_to_value : ¬ ∀ a : ℝ, |a^2| = a := by
  sorry

end NUMINAMATH_CALUDE_abs_square_not_always_equal_to_value_l2934_293401


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2934_293429

theorem sqrt_fraction_simplification :
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2934_293429


namespace NUMINAMATH_CALUDE_solution_characterization_l2934_293458

def satisfies_equation (a b c d : ℝ) : Prop :=
  a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b)

def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(k, 0, 0, 0) | k : ℝ} ∪
  {(0, k, 0, 0) | k : ℝ} ∪
  {(0, 0, k, 0) | k : ℝ} ∪
  {(0, 0, 0, k) | k : ℝ} ∪
  {(k, k, k, k) | k : ℝ} ∪
  {(k, -k, k, -k) | k : ℝ} ∪
  {(k, k*(-1 + Real.sqrt 2), -k, k*(1 - Real.sqrt 2)) | k : ℝ} ∪
  {(k, k*(-1 - Real.sqrt 2), -k, k*(1 + Real.sqrt 2)) | k : ℝ}

theorem solution_characterization :
  ∀ (a b c d : ℝ), satisfies_equation a b c d ↔ (a, b, c, d) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l2934_293458


namespace NUMINAMATH_CALUDE_quadratic_always_two_roots_l2934_293432

theorem quadratic_always_two_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (∀ x : ℝ, x^2 - m*x + m - 2 = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_two_roots_l2934_293432


namespace NUMINAMATH_CALUDE_tan_problem_l2934_293459

theorem tan_problem (α : Real) (h : Real.tan (α + π/3) = 2) :
  (Real.sin (α + 4*π/3) + Real.cos (2*π/3 - α)) /
  (Real.cos (π/6 - α) - Real.sin (α + 5*π/6)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_problem_l2934_293459


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2934_293465

theorem complex_fraction_simplification :
  let z : ℂ := (5 - 3*I) / (2 - 3*I)
  z = -19/5 - 9/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2934_293465


namespace NUMINAMATH_CALUDE_cubic_polynomial_negative_one_bound_l2934_293431

/-- A polynomial of degree 3 with three distinct positive roots -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  roots : Fin 3 → ℝ
  roots_positive : ∀ i, roots i > 0
  roots_distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  is_root : ∀ i, (roots i)^3 + a*(roots i)^2 + b*(roots i) - 1 = 0

/-- The polynomial P(x) = x^3 + ax^2 + bx - 1 -/
def P (poly : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + poly.a * x^2 + poly.b * x - 1

theorem cubic_polynomial_negative_one_bound (poly : CubicPolynomial) : P poly (-1) < -8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_negative_one_bound_l2934_293431


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2934_293406

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a b c : ℕ → ℝ) :
  ArithmeticSequence a ∧ ArithmeticSequence b ∧ ArithmeticSequence c →
  a 1 + b 1 + c 1 = 0 →
  a 2 + b 2 + c 2 = 1 →
  a 2015 + b 2015 + c 2015 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2934_293406


namespace NUMINAMATH_CALUDE_area_ratio_equilateral_triangle_extension_l2934_293482

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Extends a side of a triangle by a given factor -/
def extendSide (t : Triangle) (vertex : ℝ × ℝ) (factor : ℝ) : ℝ × ℝ := sorry

theorem area_ratio_equilateral_triangle_extension
  (ABC : Triangle)
  (h_equilateral : ABC.A.1^2 + ABC.A.2^2 = ABC.B.1^2 + ABC.B.2^2 ∧
                   ABC.B.1^2 + ABC.B.2^2 = ABC.C.1^2 + ABC.C.2^2 ∧
                   ABC.C.1^2 + ABC.C.2^2 = ABC.A.1^2 + ABC.A.2^2)
  (B' : ℝ × ℝ)
  (C' : ℝ × ℝ)
  (A' : ℝ × ℝ)
  (h_BB' : B' = extendSide ABC ABC.B 2)
  (h_CC' : C' = extendSide ABC ABC.C 3)
  (h_AA' : A' = extendSide ABC ABC.A 4)
  : area (Triangle.mk A' B' C') / area ABC = 42 := by sorry

end NUMINAMATH_CALUDE_area_ratio_equilateral_triangle_extension_l2934_293482


namespace NUMINAMATH_CALUDE_perpendicular_planes_counterexample_l2934_293444

/-- A type representing a plane in 3D space -/
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

/-- Perpendicularity relation between planes -/
def perpendicular (p q : Plane) : Prop :=
  ∃ (k : ℝ), p.normal = k • q.normal

theorem perpendicular_planes_counterexample :
  ∃ (α β γ : Plane),
    α ≠ β ∧ β ≠ γ ∧ α ≠ γ ∧
    perpendicular α β ∧
    perpendicular β γ ∧
    ¬ perpendicular α γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_counterexample_l2934_293444


namespace NUMINAMATH_CALUDE_root_relationship_l2934_293455

-- Define the first polynomial equation
def f (x : ℝ) : ℝ := x^3 - 6*x^2 - 39*x - 10

-- Define the second polynomial equation
def g (x : ℝ) : ℝ := x^3 + x^2 - 20*x - 50

-- State the theorem
theorem root_relationship :
  (∃ (x y : ℝ), f x = 0 ∧ g y = 0 ∧ x = 2*y) →
  f 10 = 0 ∧ g 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l2934_293455


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2934_293438

/-- Given an arithmetic sequence {a_n} where a₂ = 3a₅ - 6, prove that S₉ = 27 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- sum formula for arithmetic sequence
  a 2 = 3 * a 5 - 6 →                   -- given condition
  S 9 = 27 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2934_293438


namespace NUMINAMATH_CALUDE_inscribed_polygon_division_l2934_293405

-- Define a polygon inscribed around a circle
structure InscribedPolygon where
  vertices : List (ℝ × ℝ)
  center : ℝ × ℝ
  radius : ℝ
  is_inscribed : ∀ v ∈ vertices, dist center v = radius

-- Define a line passing through a point
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define the area of a polygon
def area (p : InscribedPolygon) : ℝ := sorry

-- Define the perimeter of a polygon
def perimeter (p : InscribedPolygon) : ℝ := sorry

-- Define the two parts of a polygon divided by a line
def divided_parts (p : InscribedPolygon) (l : Line) : (InscribedPolygon × InscribedPolygon) := sorry

theorem inscribed_polygon_division (p : InscribedPolygon) (l : Line) 
  (h : l.point = p.center) : 
  let (p1, p2) := divided_parts p l
  (area p1 = area p2) ∧ (perimeter p1 = perimeter p2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polygon_division_l2934_293405


namespace NUMINAMATH_CALUDE_time_ratio_in_countries_l2934_293412

/- Given conditions -/
def total_trip_duration : ℕ := 10
def time_in_first_country : ℕ := 2

/- Theorem to prove -/
theorem time_ratio_in_countries :
  (total_trip_duration - time_in_first_country) / time_in_first_country = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_ratio_in_countries_l2934_293412


namespace NUMINAMATH_CALUDE_fractional_equation_root_l2934_293421

theorem fractional_equation_root (x m : ℝ) : 
  (∃ x, x / (x - 3) - 2 = (m - 1) / (x - 3) ∧ x ≠ 3) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l2934_293421


namespace NUMINAMATH_CALUDE_broken_seashells_l2934_293467

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h1 : total = 7) (h2 : unbroken = 3) :
  total - unbroken = 4 := by
  sorry

end NUMINAMATH_CALUDE_broken_seashells_l2934_293467


namespace NUMINAMATH_CALUDE_wire_shapes_area_difference_l2934_293475

theorem wire_shapes_area_difference :
  let wire_length : ℝ := 52
  let square_side : ℝ := wire_length / 4
  let rect_width : ℝ := 15
  let rect_length : ℝ := (wire_length / 2) - rect_width
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_width * rect_length
  square_area - rect_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_wire_shapes_area_difference_l2934_293475


namespace NUMINAMATH_CALUDE_sequence_general_term_l2934_293452

/-- Given a sequence {a_n} with sum of first n terms S_n = (3(3^n + 1)) / 2,
    prove that a_n = 3^n for n ≥ 2 -/
theorem sequence_general_term (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_sum : ∀ k, S k = (3 * (3^k + 1)) / 2) 
    (h_def : ∀ k, k ≥ 2 → a k = S k - S (k-1)) :
  ∀ m, m ≥ 2 → a m = 3^m :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2934_293452


namespace NUMINAMATH_CALUDE_triangle_problem_l2934_293403

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a * sin (2 * B) = Real.sqrt 3 * b * sin A →
  cos A = 1 / 3 →
  B = π / 6 ∧ sin C = (2 * Real.sqrt 6 + 1) / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2934_293403


namespace NUMINAMATH_CALUDE_rain_probability_l2934_293443

theorem rain_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2934_293443


namespace NUMINAMATH_CALUDE_cos_double_angle_for_point_l2934_293495

/-- Given a point P(-1, 2) on the terminal side of angle α, prove that cos(2α) = -3/5 -/
theorem cos_double_angle_for_point (α : ℝ) : 
  let P : ℝ × ℝ := (-1, 2)
  (P.1 = -1 ∧ P.2 = 2) → -- P has coordinates (-1, 2)
  (P.1 = -1 * Real.sqrt 5 * Real.cos α ∧ P.2 = 2 * Real.sqrt 5 * Real.sin α) → -- P is on the terminal side of angle α
  Real.cos (2 * α) = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_for_point_l2934_293495


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l2934_293435

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℝ)
  (h1 : sugar / flour = 5 / 6)
  (h2 : flour / baking_soda = 10)
  (h3 : flour / (baking_soda + 60) = 8) :
  sugar = 2000 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l2934_293435


namespace NUMINAMATH_CALUDE_qr_equals_b_l2934_293456

-- Define the curve
def curve (c : ℝ) (x y : ℝ) : Prop := y / c = Real.cosh (x / c)

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the theorem
theorem qr_equals_b (a b c : ℝ) (P Q R : Point) : 
  curve c P.x P.y →  -- P is on the curve
  curve c Q.x Q.y →  -- Q is on the curve
  P = Point.mk a b →  -- P has coordinates (a, b)
  Q = Point.mk 0 c →  -- Q has coordinates (0, c)
  R.y = 0 →  -- R is on the x-axis
  (∃ k : ℝ, R.x = k * Real.sinh (a / c)) →  -- R.x is proportional to sinh(a/c)
  (Q.y - R.y) / (Q.x - R.x) = -1 / Real.sinh (a / c) →  -- QR is parallel to normal at P
  Real.sqrt ((R.x - Q.x)^2 + (R.y - Q.y)^2) = b  -- Distance QR equals b
  := by sorry

end NUMINAMATH_CALUDE_qr_equals_b_l2934_293456


namespace NUMINAMATH_CALUDE_number_puzzle_l2934_293437

theorem number_puzzle (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2934_293437


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2934_293424

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2934_293424


namespace NUMINAMATH_CALUDE_hard_hats_remaining_is_51_l2934_293474

/-- Calculates the remaining hard hats after transactions --/
def remaining_hard_hats (pink_initial green_initial yellow_initial : ℕ) 
  (carl_pink_taken john_pink_taken : ℕ) : ℕ :=
  let john_green_taken := 2 * john_pink_taken
  let pink_after_taken := pink_initial - carl_pink_taken - john_pink_taken
  let green_after_taken := green_initial - john_green_taken
  let carl_pink_returned := carl_pink_taken / 2
  let john_pink_returned := john_pink_taken / 3
  let john_green_returned := john_green_taken / 3
  let pink_final := pink_after_taken + carl_pink_returned + john_pink_returned
  let green_final := green_after_taken + john_green_returned
  pink_final + green_final + yellow_initial

/-- Theorem stating that the total number of hard hats remaining is 51 --/
theorem hard_hats_remaining_is_51 : 
  remaining_hard_hats 26 15 24 4 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_hard_hats_remaining_is_51_l2934_293474


namespace NUMINAMATH_CALUDE_hexagon_division_l2934_293420

/- Define a hexagon -/
def Hexagon : Type := Unit

/- Define a legal point in the hexagon -/
inductive LegalPoint : Type
| vertex : LegalPoint
| intersection : LegalPoint → LegalPoint → LegalPoint

/- Define a legal triangle in the hexagon -/
structure LegalTriangle :=
(p1 p2 p3 : LegalPoint)

/- Define a division of the hexagon -/
def Division := List LegalTriangle

/- The main theorem to prove -/
theorem hexagon_division (n : Nat) (h : n ≥ 6) : 
  ∃ (d : Division), d.length = n := by sorry

end NUMINAMATH_CALUDE_hexagon_division_l2934_293420


namespace NUMINAMATH_CALUDE_candy_sharing_l2934_293460

theorem candy_sharing (hugh tommy melany : ℕ) (h1 : hugh = 8) (h2 : tommy = 6) (h3 : melany = 7) :
  (hugh + tommy + melany) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_l2934_293460


namespace NUMINAMATH_CALUDE_square_symmetry_count_l2934_293418

/-- Represents the symmetry operations on a square -/
inductive SquareSymmetry
| reflect : SquareSymmetry
| rotate : SquareSymmetry

/-- Represents a sequence of symmetry operations -/
def SymmetrySequence := List SquareSymmetry

/-- Checks if a sequence of symmetry operations results in the identity transformation -/
def is_identity (seq : SymmetrySequence) : Prop :=
  sorry

/-- Counts the number of valid symmetry sequences of a given length -/
def count_valid_sequences (n : Nat) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem square_symmetry_count :
  count_valid_sequences 2016 % 100000 = 20000 :=
sorry

end NUMINAMATH_CALUDE_square_symmetry_count_l2934_293418


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l2934_293489

theorem average_of_four_numbers (n : ℝ) :
  (3 + 16 + 33 + (n + 1)) / 4 = 20 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l2934_293489


namespace NUMINAMATH_CALUDE_num_valid_assignments_is_72_l2934_293411

/-- Represents a valid assignment of doctors to positions -/
structure DoctorAssignment where
  assignments : Fin 5 → Fin 4
  all_positions_filled : ∀ p : Fin 4, ∃ d : Fin 5, assignments d = p
  first_two_different : assignments 0 ≠ assignments 1

/-- The number of valid doctor assignments -/
def num_valid_assignments : ℕ := sorry

/-- Theorem stating that the number of valid assignments is 72 -/
theorem num_valid_assignments_is_72 : num_valid_assignments = 72 := by sorry

end NUMINAMATH_CALUDE_num_valid_assignments_is_72_l2934_293411


namespace NUMINAMATH_CALUDE_cut_triangular_prism_has_27_edges_l2934_293416

/-- Represents a triangular prism with corners cut off -/
structure CutTriangularPrism where
  /-- The number of vertices in the original triangular prism -/
  original_vertices : Nat
  /-- The number of edges in the original triangular prism -/
  original_edges : Nat
  /-- The number of new edges created by each corner cut -/
  new_edges_per_cut : Nat
  /-- Assertion that the cuts remove each corner entirely -/
  corners_removed : Prop
  /-- Assertion that the cuts do not intersect elsewhere on the prism -/
  cuts_dont_intersect : Prop

/-- The number of edges in a triangular prism with corners cut off -/
def num_edges_after_cuts (prism : CutTriangularPrism) : Nat :=
  prism.original_edges + prism.original_vertices * prism.new_edges_per_cut

/-- Theorem stating that a triangular prism with corners cut off has 27 edges -/
theorem cut_triangular_prism_has_27_edges (prism : CutTriangularPrism)
  (h1 : prism.original_vertices = 6)
  (h2 : prism.original_edges = 9)
  (h3 : prism.new_edges_per_cut = 3)
  (h4 : prism.corners_removed)
  (h5 : prism.cuts_dont_intersect) :
  num_edges_after_cuts prism = 27 := by
  sorry


end NUMINAMATH_CALUDE_cut_triangular_prism_has_27_edges_l2934_293416


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2934_293433

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

/-- Given two circles with radii 2 and 3, and the distance between their centers is 5,
    prove that they are externally tangent -/
theorem circles_externally_tangent :
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  let d : ℝ := 5
  externally_tangent r1 r2 d := by
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l2934_293433


namespace NUMINAMATH_CALUDE_belle_treat_cost_l2934_293447

/-- The cost of feeding Belle treats for a week -/
def weekly_cost : ℚ := 21

/-- The number of dog biscuits Belle eats daily -/
def daily_biscuits : ℕ := 4

/-- The number of rawhide bones Belle eats daily -/
def daily_bones : ℕ := 2

/-- The cost of each rawhide bone in dollars -/
def bone_cost : ℚ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The cost of each dog biscuit in dollars -/
def biscuit_cost : ℚ := 1/4

theorem belle_treat_cost : 
  weekly_cost = days_in_week * (daily_biscuits * biscuit_cost + daily_bones * bone_cost) :=
by sorry

end NUMINAMATH_CALUDE_belle_treat_cost_l2934_293447


namespace NUMINAMATH_CALUDE_probability_all_genuine_given_equal_weight_l2934_293497

/-- Represents the total number of coins -/
def total_coins : ℕ := 12

/-- Represents the number of genuine coins -/
def genuine_coins : ℕ := 9

/-- Represents the number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- Event A: All 4 selected coins are genuine -/
def event_A : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) :=
  sorry

/-- Event B: The combined weight of the first pair equals the combined weight of the second pair -/
def event_B : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) :=
  sorry

/-- The probability measure on the sample space -/
def P : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) → ℚ :=
  sorry

/-- Theorem stating the conditional probability of A given B -/
theorem probability_all_genuine_given_equal_weight :
    P (event_A ∩ event_B) / P event_B = 84 / 113 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_genuine_given_equal_weight_l2934_293497


namespace NUMINAMATH_CALUDE_exactly_four_triples_l2934_293413

/-- The number of ordered triples (a, b, c) of positive integers satisfying the given LCM conditions -/
def count_triples : ℕ := 4

/-- Predicate to check if a triple (a, b, c) satisfies the LCM conditions -/
def satisfies_conditions (a b c : ℕ+) : Prop :=
  Nat.lcm a b = 90 ∧ Nat.lcm a c = 980 ∧ Nat.lcm b c = 630

/-- The main theorem stating that there are exactly 4 triples satisfying the conditions -/
theorem exactly_four_triples :
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = count_triples ∧
    ∀ t, t ∈ s ↔ satisfies_conditions t.1 t.2.1 t.2.2) :=
sorry

end NUMINAMATH_CALUDE_exactly_four_triples_l2934_293413


namespace NUMINAMATH_CALUDE_mean_calculation_l2934_293425

theorem mean_calculation (x : ℝ) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 →
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 :=
by sorry

end NUMINAMATH_CALUDE_mean_calculation_l2934_293425


namespace NUMINAMATH_CALUDE_optimal_triangle_game_l2934_293436

open Real

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- A point inside a triangle -/
def pointInside (t : Triangle) (X : ℝ × ℝ) : Prop := sorry

/-- The sum of areas of three triangles formed by connecting a point to three pairs of points on the sides of the original triangle -/
def sumOfAreas (t : Triangle) (X : ℝ × ℝ) : ℝ := sorry

theorem optimal_triangle_game (t : Triangle) (h : t.area = 1) :
  ∃ (X : ℝ × ℝ), pointInside t X ∧ sumOfAreas t X = 1/3 ∧
  ∀ (Y : ℝ × ℝ), pointInside t Y → sumOfAreas t Y ≥ 1/3 := by sorry

end NUMINAMATH_CALUDE_optimal_triangle_game_l2934_293436


namespace NUMINAMATH_CALUDE_odd_function_representation_l2934_293434

def f (x : ℝ) : ℝ := x * (abs x - 2)

theorem odd_function_representation (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x ≥ 0, f x = x^2 - 2*x) →  -- definition for x ≥ 0
  (∀ x, f x = x * (abs x - 2)) :=  -- claim to prove
by
  sorry

end NUMINAMATH_CALUDE_odd_function_representation_l2934_293434


namespace NUMINAMATH_CALUDE_probability_at_most_one_defective_is_five_sevenths_l2934_293470

def total_products : ℕ := 8
def defective_products : ℕ := 3
def drawn_products : ℕ := 3

def probability_at_most_one_defective : ℚ :=
  (Nat.choose (total_products - defective_products) drawn_products +
   Nat.choose (total_products - defective_products) (drawn_products - 1) * 
   Nat.choose defective_products 1) /
  Nat.choose total_products drawn_products

theorem probability_at_most_one_defective_is_five_sevenths :
  probability_at_most_one_defective = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_most_one_defective_is_five_sevenths_l2934_293470


namespace NUMINAMATH_CALUDE_baseball_card_value_l2934_293464

/-- The value of a baseball card after four years of depreciation --/
def card_value (initial_value : ℝ) (year1_decrease year2_decrease year3_decrease year4_decrease : ℝ) : ℝ :=
  initial_value * (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease) * (1 - year4_decrease)

/-- Theorem stating the final value of the baseball card after four years of depreciation --/
theorem baseball_card_value : 
  card_value 100 0.10 0.12 0.08 0.05 = 69.2208 := by
  sorry


end NUMINAMATH_CALUDE_baseball_card_value_l2934_293464


namespace NUMINAMATH_CALUDE_koi_added_per_day_proof_l2934_293491

/-- The number of koi fish added per day to the tank -/
def koi_added_per_day : ℕ := 2

/-- The initial total number of fish in the tank -/
def initial_total_fish : ℕ := 280

/-- The number of goldfish added per day -/
def goldfish_added_per_day : ℕ := 5

/-- The number of days in 3 weeks -/
def days_in_three_weeks : ℕ := 21

/-- The final number of goldfish in the tank -/
def final_goldfish : ℕ := 200

/-- The final number of koi fish in the tank -/
def final_koi : ℕ := 227

theorem koi_added_per_day_proof :
  koi_added_per_day = 2 :=
by sorry

end NUMINAMATH_CALUDE_koi_added_per_day_proof_l2934_293491


namespace NUMINAMATH_CALUDE_total_surveys_completed_l2934_293441

def regular_rate : ℚ := 10
def cellphone_rate : ℚ := regular_rate * (1 + 30 / 100)
def cellphone_surveys : ℕ := 60
def total_earnings : ℚ := 1180

theorem total_surveys_completed :
  ∃ (regular_surveys : ℕ),
    (regular_surveys : ℚ) * regular_rate + 
    (cellphone_surveys : ℚ) * cellphone_rate = total_earnings ∧
    regular_surveys + cellphone_surveys = 100 :=
by sorry

end NUMINAMATH_CALUDE_total_surveys_completed_l2934_293441


namespace NUMINAMATH_CALUDE_gcf_of_75_and_125_l2934_293415

theorem gcf_of_75_and_125 : Nat.gcd 75 125 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_125_l2934_293415


namespace NUMINAMATH_CALUDE_glendas_average_speed_l2934_293473

/-- Calculates the average speed given initial and final odometer readings and total time -/
def average_speed (initial_reading : ℕ) (final_reading : ℕ) (total_time : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

/-- Theorem: Glenda's average speed is 55 miles per hour -/
theorem glendas_average_speed :
  let initial_reading := 1221
  let final_reading := 1881
  let total_time := 12
  average_speed initial_reading final_reading total_time = 55 := by
  sorry

end NUMINAMATH_CALUDE_glendas_average_speed_l2934_293473


namespace NUMINAMATH_CALUDE_textbook_cost_proof_l2934_293483

/-- Represents the cost of textbooks and proves the cost of each sale textbook --/
theorem textbook_cost_proof (sale_books : ℕ) (online_books : ℕ) (bookstore_books : ℕ)
  (online_total : ℚ) (total_spent : ℚ) :
  sale_books = 5 →
  online_books = 2 →
  bookstore_books = 3 →
  online_total = 40 →
  total_spent = 210 →
  (sale_books * (total_spent - online_total - 3 * online_total) / sale_books : ℚ) = 10 := by
  sorry

#check textbook_cost_proof

end NUMINAMATH_CALUDE_textbook_cost_proof_l2934_293483


namespace NUMINAMATH_CALUDE_trivia_team_groups_l2934_293477

theorem trivia_team_groups 
  (total_students : ℕ) 
  (not_picked : ℕ) 
  (students_per_group : ℕ) 
  (h1 : total_students = 36) 
  (h2 : not_picked = 9) 
  (h3 : students_per_group = 9) : 
  (total_students - not_picked) / students_per_group = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l2934_293477


namespace NUMINAMATH_CALUDE_range_of_t_l2934_293469

/-- A function f(x) = x^2 - 2tx + 1 that is decreasing on (-∞, 1] -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t (t : ℝ) : 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → f t x > f t y) →  -- f is decreasing on (-∞, 1]
  (∀ x₁ ∈ Set.Icc 0 (t+1), ∀ x₂ ∈ Set.Icc 0 (t+1), |f t x₁ - f t x₂| ≤ 2) →  -- |f(x₁) - f(x₂)| ≤ 2
  t ∈ Set.Icc 1 (Real.sqrt 2) :=  -- t ∈ [1, √2]
sorry

end NUMINAMATH_CALUDE_range_of_t_l2934_293469


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_exactly_16_divisors_210_smallest_positive_integer_with_16_divisors_l2934_293404

def number_of_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).card

theorem smallest_with_16_divisors : 
  ∀ n : ℕ, n > 0 → number_of_divisors n = 16 → n ≥ 210 :=
by
  sorry

theorem exactly_16_divisors_210 : number_of_divisors 210 = 16 :=
by
  sorry

theorem smallest_positive_integer_with_16_divisors : 
  ∀ n : ℕ, n > 0 → number_of_divisors n = 16 → n ≥ 210 ∧ number_of_divisors 210 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_exactly_16_divisors_210_smallest_positive_integer_with_16_divisors_l2934_293404


namespace NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l2934_293488

theorem definite_integral_sin_plus_one : ∫ x in (-1)..(1), (Real.sin x + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l2934_293488


namespace NUMINAMATH_CALUDE_largest_prime_divisor_101010101_base5_l2934_293461

theorem largest_prime_divisor_101010101_base5 :
  let n : ℕ := 5^8 + 5^6 + 5^4 + 5^2 + 1
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ (∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) ∧ p = 601 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_101010101_base5_l2934_293461


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_and_circle3_properties_l2934_293409

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 13
def circle3 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 13

-- Define the line equations
def line1 (x y : ℝ) : Prop := x - 2*y - 2 = 0
def line2 (x y : ℝ) : Prop := 2*x + 3*y - 10 = 0

-- Theorem for the first circle
theorem circle1_properties :
  (∀ x y, circle1 x y → line1 x y) ∧
  circle1 0 4 ∧
  circle1 4 6 := by sorry

-- Theorem for the second and third circles
theorem circle2_and_circle3_properties :
  (∀ x y, (circle2 x y ∨ circle3 x y) → (x - 2)^2 + (y - 2)^2 = 13) ∧
  (∃ x y, (circle2 x y ∨ circle3 x y) ∧ line2 x y ∧ x = 2 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_and_circle3_properties_l2934_293409


namespace NUMINAMATH_CALUDE_sum_of_coefficients_in_factorization_l2934_293466

theorem sum_of_coefficients_in_factorization (x y : ℝ) : 
  ∃ (a b c d e f : ℤ), 
    (8 * x^8 - 243 * y^8 = (a * x^2 + b * y^2) * (c * x^2 + d * y^2) * (e * x^4 + f * y^4)) ∧
    (a + b + c + d + e + f = 17) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_in_factorization_l2934_293466


namespace NUMINAMATH_CALUDE_doughnuts_left_l2934_293426

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of doughnuts in the box -/
def boxDozens : ℕ := 2

/-- The number of doughnuts eaten -/
def eatenDoughnuts : ℕ := 8

/-- Theorem: Given a box with 2 dozen doughnuts and 8 doughnuts eaten, 
    the number of doughnuts left is 16 -/
theorem doughnuts_left : 
  boxDozens * dozen - eatenDoughnuts = 16 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_left_l2934_293426


namespace NUMINAMATH_CALUDE_tracis_road_trip_l2934_293430

theorem tracis_road_trip (D : ℝ) : 
  (1/3 : ℝ) * D + (1/4 : ℝ) * (2/3 : ℝ) * D + 300 = D → D = 600 :=
by sorry

end NUMINAMATH_CALUDE_tracis_road_trip_l2934_293430


namespace NUMINAMATH_CALUDE_euler_totient_even_bound_l2934_293408

theorem euler_totient_even_bound (n : ℕ) (h : Even n) (h_pos : n > 0) : 
  (Finset.filter (fun x => Nat.gcd n x = 1) (Finset.range n)).card ≤ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_even_bound_l2934_293408


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_18_mod_25_l2934_293493

theorem largest_five_digit_congruent_18_mod_25 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≤ 99999 ∧ n % 25 = 18 → 
    n ≤ 99993 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_18_mod_25_l2934_293493


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2934_293457

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^8 + 5 * x^6 - 3 * x + 9) + 5 * (x^7 - 3 * x^3 + 2 * x^2 - 4)

theorem sum_of_coefficients : polynomial 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2934_293457


namespace NUMINAMATH_CALUDE_veranda_area_l2934_293417

/-- The area of a veranda surrounding a rectangular room. -/
theorem veranda_area (room_length room_width veranda_length_side veranda_width_side : ℝ)
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_length_side = 2.5)
  (h4 : veranda_width_side = 3) :
  (room_length + 2 * veranda_length_side) * (room_width + 2 * veranda_width_side) - 
  room_length * room_width = 204 := by
  sorry

end NUMINAMATH_CALUDE_veranda_area_l2934_293417


namespace NUMINAMATH_CALUDE_vector_magnitude_l2934_293446

theorem vector_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a - b = (Real.sqrt 3, Real.sqrt 2) →
  ‖a + 2 • b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2934_293446


namespace NUMINAMATH_CALUDE_carnival_tickets_l2934_293422

theorem carnival_tickets (tickets : ℕ) (extra : ℕ) : 
  let F := Nat.minFac (tickets + extra)
  F ∣ (tickets + extra) ∧ ¬(F ∣ tickets) →
  F = 3 :=
by
  sorry

#check carnival_tickets 865 8

end NUMINAMATH_CALUDE_carnival_tickets_l2934_293422


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2934_293490

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2934_293490


namespace NUMINAMATH_CALUDE_inequality_proof_l2934_293419

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2934_293419


namespace NUMINAMATH_CALUDE_y_squared_eq_three_x_squared_plus_one_l2934_293442

/-- Sequence x defined recursively -/
def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

/-- Sequence y defined recursively -/
def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

/-- Main theorem: For all natural numbers n, y(n)² = 3x(n)² + 1 -/
theorem y_squared_eq_three_x_squared_plus_one (n : ℕ) : (y n)^2 = 3*(x n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_eq_three_x_squared_plus_one_l2934_293442


namespace NUMINAMATH_CALUDE_bhavan_score_percentage_l2934_293427

theorem bhavan_score_percentage (max_score : ℝ) (amar_percent : ℝ) (chetan_percent : ℝ) (average_mark : ℝ) :
  max_score = 900 →
  amar_percent = 64 →
  chetan_percent = 44 →
  average_mark = 432 →
  ∃ bhavan_percent : ℝ,
    bhavan_percent = 36 ∧
    3 * average_mark = (amar_percent / 100 * max_score) + (bhavan_percent / 100 * max_score) + (chetan_percent / 100 * max_score) :=
by sorry

end NUMINAMATH_CALUDE_bhavan_score_percentage_l2934_293427


namespace NUMINAMATH_CALUDE_average_weight_solution_l2934_293439

def average_weight_problem (d e f : ℝ) : Prop :=
  (d + e + f) / 3 = 42 ∧
  (e + f) / 2 = 41 ∧
  e = 26 →
  (d + e) / 2 = 35

theorem average_weight_solution :
  ∀ d e f : ℝ, average_weight_problem d e f :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_solution_l2934_293439


namespace NUMINAMATH_CALUDE_no_infinite_sqrt_sequence_l2934_293468

theorem no_infinite_sqrt_sequence :
  ¬ (∃ (a : ℕ → ℕ+), ∀ (n : ℕ), n ≥ 1 → (a (n + 2)).val = Int.sqrt ((a (n + 1)).val) + (a n).val) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_sqrt_sequence_l2934_293468


namespace NUMINAMATH_CALUDE_incorrect_operation_l2934_293486

theorem incorrect_operation (x y : ℝ) : -2*x*(x - y) ≠ -2*x^2 - 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_l2934_293486


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l2934_293451

/-- A natural number is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Four consecutive natural numbers are all composite. -/
def FourConsecutiveComposites (n : ℕ) : Prop :=
  IsComposite n ∧ IsComposite (n + 1) ∧ IsComposite (n + 2) ∧ IsComposite (n + 3)

/-- The sum of four consecutive natural numbers starting from n. -/
def SumFourConsecutive (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3)

theorem smallest_sum_four_consecutive_composites :
  (∃ n : ℕ, FourConsecutiveComposites n) ∧
  (∀ m : ℕ, FourConsecutiveComposites m → SumFourConsecutive m ≥ 102) ∧
  (∃ k : ℕ, FourConsecutiveComposites k ∧ SumFourConsecutive k = 102) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l2934_293451


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l2934_293499

/-- Represents the number of triangles of each color in one half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

/-- Theorem stating that given the conditions, 7 white pairs must coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 6 ∧ 
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 2 →
  pairs.white_white = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l2934_293499


namespace NUMINAMATH_CALUDE_zeros_in_decimal_representation_l2934_293445

theorem zeros_in_decimal_representation (n : ℕ) : 
  (∃ k : ℕ, (1 : ℚ) / (25^10 : ℚ) = (1 : ℚ) / (10^k : ℚ)) ∧ 
  (∀ m : ℕ, m < 20 → (1 : ℚ) / (25^10 : ℚ) < (1 : ℚ) / (10^m : ℚ)) ∧
  (1 : ℚ) / (25^10 : ℚ) ≥ (1 : ℚ) / (10^20 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_zeros_in_decimal_representation_l2934_293445


namespace NUMINAMATH_CALUDE_max_value_theorem_l2934_293407

theorem max_value_theorem (p q r s : ℝ) (h : p^2 + q^2 + r^2 - s^2 + 4 = 0) :
  ∃ (M : ℝ), M = -2 * Real.sqrt 2 ∧ ∀ (p' q' r' s' : ℝ), 
    p'^2 + q'^2 + r'^2 - s'^2 + 4 = 0 → 
    3*p' + 2*q' + r' - 4*abs s' ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2934_293407


namespace NUMINAMATH_CALUDE_school_construction_problem_l2934_293472

/-- School construction problem -/
theorem school_construction_problem
  (total_area : ℝ)
  (demolition_cost : ℝ)
  (construction_cost : ℝ)
  (actual_demolition_ratio : ℝ)
  (actual_construction_ratio : ℝ)
  (greening_cost : ℝ)
  (h1 : total_area = 7200)
  (h2 : demolition_cost = 80)
  (h3 : construction_cost = 700)
  (h4 : actual_demolition_ratio = 1.1)
  (h5 : actual_construction_ratio = 0.8)
  (h6 : greening_cost = 200) :
  ∃ (planned_demolition planned_construction greening_area : ℝ),
    planned_demolition + planned_construction = total_area ∧
    actual_demolition_ratio * planned_demolition + actual_construction_ratio * planned_construction = total_area ∧
    planned_demolition = 4800 ∧
    planned_construction = 2400 ∧
    greening_area = 1488 ∧
    greening_area * greening_cost = 
      (planned_demolition * demolition_cost + planned_construction * construction_cost) -
      (actual_demolition_ratio * planned_demolition * demolition_cost + 
       actual_construction_ratio * planned_construction * construction_cost) :=
by sorry

end NUMINAMATH_CALUDE_school_construction_problem_l2934_293472


namespace NUMINAMATH_CALUDE_max_m_value_l2934_293450

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Theorem statement
theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 := by
  sorry

end NUMINAMATH_CALUDE_max_m_value_l2934_293450


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l2934_293448

/-- Given a function f(x) = 4x + a/x where x > 0 and a > 0,
    if f takes its minimum value at x = 3, then a = 36 -/
theorem minimum_value_implies_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, x > 0 → ∃ f : ℝ → ℝ, f x = 4*x + a/x) →
  (∃ f : ℝ → ℝ, ∀ x : ℝ, x > 0 → f x ≥ f 3) →
  a = 36 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l2934_293448


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l2934_293484

/-- The area between two concentric circles -/
theorem area_between_concentric_circles 
  (R : ℝ) -- Radius of the outer circle
  (c : ℝ) -- Length of the chord
  (h1 : R = 12) -- Given radius of outer circle
  (h2 : c = 20) -- Given length of chord
  (h3 : c ≤ 2 * R) -- Chord cannot be longer than diameter
  : ∃ (r : ℝ), 0 < r ∧ r < R ∧ π * (R^2 - r^2) = 100 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l2934_293484


namespace NUMINAMATH_CALUDE_sticker_pages_l2934_293400

theorem sticker_pages (stickers_per_page : ℕ) (remaining_stickers : ℕ) : 
  (stickers_per_page = 20 ∧ remaining_stickers = 220) → 
  ∃ (initial_pages : ℕ), 
    initial_pages * stickers_per_page - stickers_per_page = remaining_stickers ∧ 
    initial_pages = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sticker_pages_l2934_293400


namespace NUMINAMATH_CALUDE_counsel_probability_l2934_293496

def CANOE : Finset Char := {'C', 'A', 'N', 'O', 'E'}
def SHRUB : Finset Char := {'S', 'H', 'R', 'U', 'B'}
def FLOW : Finset Char := {'F', 'L', 'O', 'W'}
def COUNSEL : Finset Char := {'C', 'O', 'U', 'N', 'S', 'E', 'L'}

def prob_CANOE : ℚ := 1 / (CANOE.card.choose 2)
def prob_SHRUB : ℚ := 3 / (SHRUB.card.choose 3)
def prob_FLOW : ℚ := 1 / (FLOW.card.choose 4)

theorem counsel_probability :
  prob_CANOE * prob_SHRUB * prob_FLOW = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_counsel_probability_l2934_293496


namespace NUMINAMATH_CALUDE_power_of_17_mod_26_l2934_293478

theorem power_of_17_mod_26 : 17^1999 % 26 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_of_17_mod_26_l2934_293478


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l2934_293492

theorem smallest_four_digit_mod_9 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n ≡ 8 [MOD 9] → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l2934_293492


namespace NUMINAMATH_CALUDE_base_2_representation_of_75_l2934_293449

theorem base_2_representation_of_75 :
  ∃ (a b c d e f g : ℕ),
    a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 ∧
    75 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_75_l2934_293449


namespace NUMINAMATH_CALUDE_correct_minutes_for_ninth_day_l2934_293498

/-- The number of minutes Julia needs to read on the 9th day to achieve the target average -/
def minutes_to_read_on_ninth_day (days_reading_80_min : ℕ) (days_reading_100_min : ℕ) (target_average : ℕ) (total_days : ℕ) : ℕ :=
  let total_minutes_read := days_reading_80_min * 80 + days_reading_100_min * 100
  let target_total_minutes := total_days * target_average
  target_total_minutes - total_minutes_read

/-- Theorem stating the correct number of minutes Julia needs to read on the 9th day -/
theorem correct_minutes_for_ninth_day :
  minutes_to_read_on_ninth_day 6 2 95 9 = 175 := by
  sorry

end NUMINAMATH_CALUDE_correct_minutes_for_ninth_day_l2934_293498


namespace NUMINAMATH_CALUDE_car_rental_cost_l2934_293476

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total cost of renting a car for 5 days at $30 per day and driving 500 miles at $0.25 per mile is $275. -/
theorem car_rental_cost : rental_cost 30 (1/4) 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_car_rental_cost_l2934_293476


namespace NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l2934_293481

/-- The number of faces on each cubic die -/
def dice_faces : ℕ := 6

/-- The number of ways to obtain a sum of 7 -/
def favorable_outcomes : ℕ := 6

/-- The probability of obtaining a sum of 7 when throwing two cubic dice -/
def probability_sum_seven : ℚ := favorable_outcomes / (dice_faces * dice_faces)

theorem probability_sum_seven_is_one_sixth : 
  probability_sum_seven = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l2934_293481


namespace NUMINAMATH_CALUDE_student_count_bound_l2934_293494

theorem student_count_bound (N M k ℓ : ℕ) (h1 : M = k * N / 100) 
  (h2 : 100 * (M + 1) = ℓ * (N + 3)) (h3 : ℓ < 100) : N ≤ 197 := by
  sorry

end NUMINAMATH_CALUDE_student_count_bound_l2934_293494


namespace NUMINAMATH_CALUDE_exponent_simplification_l2934_293480

theorem exponent_simplification :
  3^6 * 6^6 * 3^12 * 6^12 = 18^18 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2934_293480


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l2934_293410

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let rectangle1_height : ℝ := 6
  let rectangle1_width : ℝ := 1
  let rectangle2_height : ℝ := 2
  let rectangle2_width : ℝ := 4
  let semicircle_diameter : ℝ := 2
  
  let volume_cylinder1 : ℝ := π * rectangle1_height^2 * rectangle1_width
  let volume_cylinder2 : ℝ := π * rectangle2_height^2 * rectangle2_width
  let volume_hemisphere : ℝ := (2/3) * π * (semicircle_diameter/2)^3
  
  let total_volume : ℝ := volume_cylinder1 + volume_cylinder2 + volume_hemisphere
  
  total_volume = 52 * (2/3) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l2934_293410


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2934_293462

/-- The equation 3^(3x^3 - 9x^2 + 15x - 5) = 1 has exactly one real solution. -/
theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (3 : ℝ) ^ (3 * x^3 - 9 * x^2 + 15 * x - 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2934_293462


namespace NUMINAMATH_CALUDE_average_difference_l2934_293428

theorem average_difference (x : ℝ) : 
  (10 + 30 + 50) / 3 = (20 + 40 + x) / 3 + 8 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2934_293428


namespace NUMINAMATH_CALUDE_product_mod_eight_l2934_293463

theorem product_mod_eight : (55 * 57) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eight_l2934_293463


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2934_293479

theorem triangle_angle_calculation (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A = 60 →           -- Angle A is 60°
  C = 2 * B →        -- Angle C is twice Angle B
  C = 80 :=          -- Conclusion: Angle C is 80°
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2934_293479


namespace NUMINAMATH_CALUDE_rain_given_east_wind_l2934_293414

/-- Given that:
    1. The probability of an east wind in April is 8/30
    2. The probability of both an east wind and rain in April is 7/30
    Prove that the probability of rain in April given an east wind is 7/8 -/
theorem rain_given_east_wind (p_east : ℚ) (p_east_and_rain : ℚ) 
  (h1 : p_east = 8/30) (h2 : p_east_and_rain = 7/30) :
  p_east_and_rain / p_east = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_rain_given_east_wind_l2934_293414
