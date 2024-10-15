import Mathlib

namespace NUMINAMATH_CALUDE_exterior_angle_of_right_triangle_l444_44434

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define a right triangle
structure RightTriangle extends Triangle :=
  (right_angle : C = 90)

-- Theorem statement
theorem exterior_angle_of_right_triangle (t : RightTriangle) :
  180 - t.C = 90 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_of_right_triangle_l444_44434


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l444_44464

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l444_44464


namespace NUMINAMATH_CALUDE_rectangle_area_l444_44468

/-- A rectangle with diagonal length x and length three times its width has area (3/10) * x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w : ℝ, 
  w > 0 ∧ 
  w^2 + (3*w)^2 = x^2 ∧ 
  w * (3*w) = (3/10) * x^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l444_44468


namespace NUMINAMATH_CALUDE_triangle_longest_side_l444_44413

theorem triangle_longest_side :
  ∀ x : ℝ,
  let side1 := 7
  let side2 := x + 4
  let side3 := 2*x + 1
  (side1 + side2 + side3 = 36) →
  (∃ longest : ℝ, longest = max side1 (max side2 side3) ∧ longest = 17) :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l444_44413


namespace NUMINAMATH_CALUDE_complex_equation_solution_l444_44495

theorem complex_equation_solution (c d x : ℂ) (i : ℂ) : 
  c * d = x - 5 * i → 
  Complex.abs c = 3 →
  Complex.abs d = Real.sqrt 50 →
  x = 5 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l444_44495


namespace NUMINAMATH_CALUDE_distance_to_focus_l444_44499

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (x : ℝ) : 
  let P : ℝ × ℝ := (x, (1/4) * x^2)
  let parabola := {(x, y) : ℝ × ℝ | y = (1/4) * x^2}
  P ∈ parabola → P.2 = 4 → ∃ F : ℝ × ℝ, F.2 = 1/4 ∧ dist P F = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l444_44499


namespace NUMINAMATH_CALUDE_non_overlapping_area_l444_44491

/-- Rectangle ABCD with side lengths 4 and 6 -/
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)
  (h_AB : AB = 4)
  (h_BC : BC = 6)

/-- The fold that makes B and D coincide -/
structure Fold (rect : Rectangle) :=
  (E : ℝ × ℝ)  -- Point E on the crease
  (F : ℝ × ℝ)  -- Point F on the crease
  (h_coincide : E.1 + F.1 = rect.AB ∧ E.2 + F.2 = rect.BC)  -- B and D coincide after folding

/-- The theorem stating the area of the non-overlapping part -/
theorem non_overlapping_area (rect : Rectangle) (fold : Fold rect) :
  ∃ (area : ℝ), area = 20 / 3 ∧ area = 2 * (1 / 2 * rect.AB * (rect.BC - fold.E.2)) :=
sorry

end NUMINAMATH_CALUDE_non_overlapping_area_l444_44491


namespace NUMINAMATH_CALUDE_orthogonal_matrix_sum_of_squares_l444_44404

theorem orthogonal_matrix_sum_of_squares (p q r s : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]
  (B.transpose = B⁻¹) → (p = s) → p^2 + q^2 + r^2 + s^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_orthogonal_matrix_sum_of_squares_l444_44404


namespace NUMINAMATH_CALUDE_range_of_slope_intersecting_line_l444_44430

/-- Given two points P and Q, and a line l that intersects the extension of PQ,
    prove the range of values for the slope of l. -/
theorem range_of_slope_intersecting_line (P Q : ℝ × ℝ) (m : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  ∃ (x y : ℝ), x + m * y + m = 0 ∧ 
    (∃ (t : ℝ), x = -1 + 3 * t ∧ y = 1 + t) →
  -3 < m ∧ m < -2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_slope_intersecting_line_l444_44430


namespace NUMINAMATH_CALUDE_beans_remaining_fraction_l444_44485

/-- The fraction of beans remaining in a jar after some have been removed -/
theorem beans_remaining_fraction (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (h1 : jar_weight = 0.1 * (jar_weight + full_beans_weight))
  (h2 : ∃ remaining_beans : ℝ, jar_weight + remaining_beans = 0.6 * (jar_weight + full_beans_weight)) :
  ∃ remaining_beans : ℝ, remaining_beans / full_beans_weight = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_beans_remaining_fraction_l444_44485


namespace NUMINAMATH_CALUDE_repetend_of_three_thirteenths_l444_44414

/-- The decimal representation of 3/13 has a 6-digit repetend of 230769 -/
theorem repetend_of_three_thirteenths : ∃ (n : ℕ), 
  (3 : ℚ) / 13 = (230769 : ℚ) / 999999 + n / (999999 * 13) := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_three_thirteenths_l444_44414


namespace NUMINAMATH_CALUDE_nine_knights_among_travelers_total_travelers_is_sixteen_l444_44472

/-- A traveler can be either a knight or a liar -/
inductive TravelerType
  | Knight
  | Liar

/-- Represents a room in the hotel -/
structure Room where
  knights : Nat
  liars : Nat

/-- Represents the hotel with three rooms -/
structure Hotel where
  room1 : Room
  room2 : Room
  room3 : Room

def total_travelers : Nat := 16

/-- Vasily, who makes contradictory statements -/
def vasily : TravelerType := TravelerType.Liar

/-- The theorem stating that there must be 9 knights among the 16 travelers -/
theorem nine_knights_among_travelers (h : Hotel) : 
  h.room1.knights + h.room2.knights + h.room3.knights = 9 :=
by
  sorry

/-- The theorem stating that the total number of travelers is 16 -/
theorem total_travelers_is_sixteen (h : Hotel) :
  h.room1.knights + h.room1.liars + 
  h.room2.knights + h.room2.liars + 
  h.room3.knights + h.room3.liars = total_travelers :=
by
  sorry

end NUMINAMATH_CALUDE_nine_knights_among_travelers_total_travelers_is_sixteen_l444_44472


namespace NUMINAMATH_CALUDE_lindas_savings_l444_44463

theorem lindas_savings (savings : ℝ) : 
  (2 / 3 : ℝ) * savings + 250 = savings → savings = 750 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l444_44463


namespace NUMINAMATH_CALUDE_max_product_sum_1988_l444_44469

theorem max_product_sum_1988 (sequence : List Nat) : 
  (sequence.sum = 1988) → (sequence.all (· > 0)) →
  (sequence.prod ≤ 2 * 3^662) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_1988_l444_44469


namespace NUMINAMATH_CALUDE_xiaoming_pencil_theorem_l444_44442

/-- Represents the number of pencils and amount spent in Xiaoming's purchases -/
structure PencilPurchase where
  x : ℕ  -- number of pencils in first purchase
  y : ℕ  -- amount spent in first purchase in yuan

/-- Determines if a PencilPurchase satisfies the problem conditions -/
def satisfiesConditions (p : PencilPurchase) : Prop :=
  ∃ (price : ℚ), 
    price = p.y / p.x ∧  -- initial price per pencil
    (4 : ℚ) / 5 * price * (p.x + 10) = 4  -- condition after price drop

/-- The theorem stating the possible total numbers of pencils bought -/
theorem xiaoming_pencil_theorem (p : PencilPurchase) :
  satisfiesConditions p → (p.x + (p.x + 10) = 40 ∨ p.x + (p.x + 10) = 90) :=
by
  sorry

#check xiaoming_pencil_theorem

end NUMINAMATH_CALUDE_xiaoming_pencil_theorem_l444_44442


namespace NUMINAMATH_CALUDE_polynomial_complex_roots_bounds_l444_44444

-- Define the polynomial P(x)
def P (x : ℂ) : ℂ := x^3 + x^2 - x + 2

-- State the theorem
theorem polynomial_complex_roots_bounds :
  ∀ r : ℝ, (∃ z : ℂ, z.im ≠ 0 ∧ P z = r) ↔ (3 < r ∧ r < 49/27) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_complex_roots_bounds_l444_44444


namespace NUMINAMATH_CALUDE_tourist_base_cottages_l444_44447

theorem tourist_base_cottages :
  ∀ (x : ℕ) (n : ℕ+),
    (2 * x) + x + (n : ℕ) * x ≥ 70 →
    3 * ((n : ℕ) * x) = 2 * x + 25 →
    (2 * x) + x + (n : ℕ) * x = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_tourist_base_cottages_l444_44447


namespace NUMINAMATH_CALUDE_tangent_length_circle_tangent_length_l444_44409

/-- The length of a tangent to a circle from an external point -/
theorem tangent_length (r d l : ℝ) (hr : r > 0) (hd : d > r) : 
  r = 36 → d = 85 → l = 77 → l^2 = d^2 - r^2 := by
  sorry

/-- The main theorem stating the length of the tangent -/
theorem circle_tangent_length : 
  ∃ (l : ℝ), l = 77 ∧ l^2 = 85^2 - 36^2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_circle_tangent_length_l444_44409


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l444_44493

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon is a polygon with 7 sides -/
def is_heptagon (n : ℕ) : Prop := n = 7

theorem heptagon_diagonals (n : ℕ) (h : is_heptagon n) : num_diagonals n = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l444_44493


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_monotonic_decreasing_on_open_interval_l444_44496

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 2) ↔ (f_derivative x < 0) :=
by sorry

-- Main theorem
theorem f_monotonic_decreasing_on_open_interval :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_monotonic_decreasing_on_open_interval_l444_44496


namespace NUMINAMATH_CALUDE_incorrect_calculation_l444_44448

theorem incorrect_calculation (x : ℝ) : (-3 * x)^2 ≠ 6 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l444_44448


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_min_value_three_iff_l444_44407

-- Define the function f
def f (x a : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - a)

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 5} = {x : ℝ | x ≤ -2 ∨ x ≥ 4/3} := by sorry

-- Theorem for part (2)
theorem min_value_three_iff :
  (∃ x : ℝ, f x a = 3) ∧ (∀ x : ℝ, f x a ≥ 3) ↔ a = 2 ∨ a = -4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_min_value_three_iff_l444_44407


namespace NUMINAMATH_CALUDE_snickers_for_nintendo_switch_l444_44429

def snickers_needed (total_points_needed : ℕ) (chocolate_bunnies_sold : ℕ) (points_per_bunny : ℕ) (points_per_snickers : ℕ) : ℕ :=
  let points_from_bunnies := chocolate_bunnies_sold * points_per_bunny
  let remaining_points := total_points_needed - points_from_bunnies
  remaining_points / points_per_snickers

theorem snickers_for_nintendo_switch : 
  snickers_needed 2000 8 100 25 = 48 := by
  sorry

end NUMINAMATH_CALUDE_snickers_for_nintendo_switch_l444_44429


namespace NUMINAMATH_CALUDE_intersection_implies_equality_l444_44416

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem intersection_implies_equality (a b c d : ℝ) 
  (h1 : f a b 1 = 1) 
  (h2 : g c d 1 = 1) : 
  a^5 + d^6 = c^6 - b^5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_equality_l444_44416


namespace NUMINAMATH_CALUDE_jamie_max_correct_answers_l444_44482

theorem jamie_max_correct_answers
  (total_questions : ℕ)
  (correct_points : ℤ)
  (blank_points : ℤ)
  (incorrect_points : ℤ)
  (total_score : ℤ)
  (h1 : total_questions = 60)
  (h2 : correct_points = 5)
  (h3 : blank_points = 0)
  (h4 : incorrect_points = -2)
  (h5 : total_score = 150) :
  ∃ (x : ℕ), x ≤ 38 ∧
    ∀ (y : ℕ), y > 38 →
      ¬∃ (blank incorrect : ℕ),
        y + blank + incorrect = total_questions ∧
        y * correct_points + blank * blank_points + incorrect * incorrect_points = total_score :=
by sorry

end NUMINAMATH_CALUDE_jamie_max_correct_answers_l444_44482


namespace NUMINAMATH_CALUDE_squirrel_nut_distance_l444_44419

theorem squirrel_nut_distance (total_time : ℝ) (speed_without_nut : ℝ) (speed_with_nut : ℝ) 
  (h1 : total_time = 1200)
  (h2 : speed_without_nut = 5)
  (h3 : speed_with_nut = 3) :
  ∃ x : ℝ, x = 2250 ∧ x / speed_without_nut + x / speed_with_nut = total_time :=
by sorry

end NUMINAMATH_CALUDE_squirrel_nut_distance_l444_44419


namespace NUMINAMATH_CALUDE_sin_plus_cos_eq_neg_one_solution_set_l444_44486

theorem sin_plus_cos_eq_neg_one_solution_set :
  {x : ℝ | Real.sin x + Real.cos x = -1} =
  {x : ℝ | ∃ n : ℤ, x = (2*n - 1)*π ∨ x = 2*n*π - π/2} := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_eq_neg_one_solution_set_l444_44486


namespace NUMINAMATH_CALUDE_triangle_properties_l444_44476

-- Define the points in the complex plane
def A : ℂ := 1
def B : ℂ := -Complex.I
def C : ℂ := -1 + 2 * Complex.I

-- Define the vectors
def AB : ℂ := B - A
def AC : ℂ := C - A
def BC : ℂ := C - B

-- Theorem statement
theorem triangle_properties :
  (AB.re = -1 ∧ AB.im = -1) ∧
  (AC.re = -2 ∧ AC.im = 2) ∧
  (BC.re = -1 ∧ BC.im = 3) ∧
  (AB.re * AC.re + AB.im * AC.im = 0) := by
  sorry

-- The last condition (AB.re * AC.re + AB.im * AC.im = 0) checks if AB and AC are perpendicular,
-- which implies that the triangle is right-angled.

end NUMINAMATH_CALUDE_triangle_properties_l444_44476


namespace NUMINAMATH_CALUDE_merchant_articles_l444_44465

/-- Represents the number of articles a merchant has -/
def N : ℕ := 20

/-- Represents the cost price of each article -/
def CP : ℝ := 1

/-- Represents the selling price of each article -/
def SP : ℝ := 1.25 * CP

theorem merchant_articles :
  (N * CP = 16 * SP) ∧ (SP = 1.25 * CP) → N = 20 := by
  sorry

end NUMINAMATH_CALUDE_merchant_articles_l444_44465


namespace NUMINAMATH_CALUDE_complex_equation_solution_l444_44420

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (z + 2) * (1 + i^3) = 2

-- Theorem statement
theorem complex_equation_solution :
  ∃ z : ℂ, equation z ∧ z = -1 + i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l444_44420


namespace NUMINAMATH_CALUDE_cable_intersections_6_8_l444_44470

/-- The number of pairwise intersections of cables connecting houses across a street -/
def cable_intersections (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose m 2

/-- Theorem stating the number of cable intersections for 6 houses on one side and 8 on the other -/
theorem cable_intersections_6_8 :
  cable_intersections 6 8 = 420 := by
  sorry

end NUMINAMATH_CALUDE_cable_intersections_6_8_l444_44470


namespace NUMINAMATH_CALUDE_division_remainder_proof_l444_44406

theorem division_remainder_proof (L S : ℕ) (h1 : L - S = 1395) (h2 : L = 1656) 
  (h3 : ∃ q r, L = S * q + r ∧ q = 6 ∧ r < S) : 
  ∃ r, L = S * 6 + r ∧ r = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l444_44406


namespace NUMINAMATH_CALUDE_a_value_range_l444_44439

/-- Proposition p: For any x, ax^2 + ax + 1 > 0 always holds true -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The equation x^2 - x + a = 0 has real roots -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The range of values for a satisfying the given conditions -/
def a_range (a : ℝ) : Prop := a < 0 ∨ (1/4 < a ∧ a < 4)

theorem a_value_range :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a_range a :=
by sorry

end NUMINAMATH_CALUDE_a_value_range_l444_44439


namespace NUMINAMATH_CALUDE_total_beanie_babies_l444_44445

theorem total_beanie_babies (lori_beanie_babies sydney_beanie_babies : ℕ) :
  lori_beanie_babies = 300 →
  lori_beanie_babies = 15 * sydney_beanie_babies →
  lori_beanie_babies + sydney_beanie_babies = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_beanie_babies_l444_44445


namespace NUMINAMATH_CALUDE_inequality_solution_max_value_condition_l444_44408

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem inequality_solution (x : ℝ) :
  f 2 x > 1 ↔ x < -3/2 ∨ x > 1 := by sorry

-- Part 2
theorem max_value_condition (a : ℝ) :
  (∃ x, f a x = 17/8 ∧ ∀ y, f a y ≤ 17/8) →
  (a = -2 ∨ a = -1/8) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_max_value_condition_l444_44408


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_count_l444_44477

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of distinct points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

theorem convex_quadrilaterals_count :
  binomial num_points vertices_per_quadrilateral = 495 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilaterals_count_l444_44477


namespace NUMINAMATH_CALUDE_cipher_solution_l444_44432

/-- Represents a mapping from letters to digits -/
def Cipher := Char → Nat

/-- The condition that each letter represents a unique digit -/
def is_valid_cipher (c : Cipher) : Prop :=
  ∀ x y : Char, c x = c y → x = y

/-- The value of a word under a given cipher -/
def word_value (c : Cipher) (w : String) : Nat :=
  w.foldl (λ acc d => 10 * acc + c d) 0

/-- The main theorem -/
theorem cipher_solution (c : Cipher) 
  (h1 : is_valid_cipher c)
  (h2 : word_value c "СЕКРЕТ" - word_value c "ОТКРЫТ" = 20010)
  (h3 : c 'Т' = 9) :
  word_value c "СЕК" = 392 ∧ c 'О' = 2 :=
sorry

end NUMINAMATH_CALUDE_cipher_solution_l444_44432


namespace NUMINAMATH_CALUDE_hyperbola_equation_l444_44421

/-- Given a hyperbola passing through the point (2√2, 1) with one asymptote equation y = 1/2x,
    its standard equation is x²/4 - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ k : ℝ, x^2 / 4 - y^2 = k ∧ (2 * Real.sqrt 2)^2 / 4 - 1^2 = k) ∧
  (∃ m : ℝ, y = 1/2 * x + m) →
  x^2 / 4 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l444_44421


namespace NUMINAMATH_CALUDE_socks_price_l444_44446

/-- Given the prices of jeans, t-shirt, and socks, where:
  1. The jeans cost twice as much as the t-shirt
  2. The t-shirt costs $10 more than the socks
  3. The jeans cost $30
  Prove that the socks cost $5 -/
theorem socks_price (jeans t_shirt socks : ℕ) 
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) : 
  socks = 5 := by
sorry

end NUMINAMATH_CALUDE_socks_price_l444_44446


namespace NUMINAMATH_CALUDE_digging_hours_calculation_l444_44436

/-- Calculates the initial working hours per day given the conditions of the digging problem. -/
theorem digging_hours_calculation 
  (initial_men : ℕ) 
  (initial_depth : ℝ) 
  (new_depth : ℝ) 
  (new_hours : ℝ) 
  (extra_men : ℕ) 
  (h : initial_men = 63)
  (i : initial_depth = 30)
  (n : new_depth = 50)
  (w : new_hours = 6)
  (e : extra_men = 77) :
  ∃ (initial_hours : ℝ), 
    initial_hours = 8 ∧ 
    (initial_men : ℝ) * initial_hours * initial_depth = 
    ((initial_men : ℝ) + extra_men) * new_hours * new_depth := by
  sorry

#check digging_hours_calculation

end NUMINAMATH_CALUDE_digging_hours_calculation_l444_44436


namespace NUMINAMATH_CALUDE_odd_painted_faces_6_4_2_l444_44403

/-- Represents a 3D rectangular block of cubes -/
structure Block :=
  (length : Nat) (width : Nat) (height : Nat)

/-- Counts the number of cubes with an odd number of painted faces in a block -/
def oddPaintedFaces (b : Block) : Nat :=
  sorry

/-- The main theorem: In a 6x4x2 block, 16 cubes have an odd number of painted faces -/
theorem odd_painted_faces_6_4_2 : 
  oddPaintedFaces (Block.mk 6 4 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_odd_painted_faces_6_4_2_l444_44403


namespace NUMINAMATH_CALUDE_kingsleys_friends_l444_44415

theorem kingsleys_friends (chairs_per_trip : ℕ) (total_trips : ℕ) (total_chairs : ℕ) :
  chairs_per_trip = 5 →
  total_trips = 10 →
  total_chairs = 250 →
  (total_chairs / (chairs_per_trip * total_trips)) - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_kingsleys_friends_l444_44415


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l444_44433

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/3)
  (h_S : S = 18)
  (h_sum : S = a / (1 - r))
  (h_convergence : abs r < 1) :
  a = 12 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l444_44433


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l444_44459

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 210 →
  8 * a = b - 11 →
  8 * a = c + 11 →
  a * b * c = 4173.75 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l444_44459


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l444_44427

/-- Represents a figure composed of unit squares -/
structure UnitSquareFigure where
  rows : Nat
  columns : Nat
  extra_column : Nat

/-- Calculates the perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : Nat :=
  sorry

/-- The specific figure described in the problem -/
def specific_figure : UnitSquareFigure :=
  { rows := 3, columns := 4, extra_column := 2 }

theorem specific_figure_perimeter :
  perimeter specific_figure = 13 := by sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l444_44427


namespace NUMINAMATH_CALUDE_same_root_implies_a_value_l444_44454

theorem same_root_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x - a = 0 ∧ x^2 + a*x - 2 = 0) → (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_same_root_implies_a_value_l444_44454


namespace NUMINAMATH_CALUDE_jessica_watermelons_l444_44440

/-- The number of watermelons Jessica initially grew -/
def initial_watermelons : ℕ := 35

/-- The number of watermelons eaten by rabbits -/
def eaten_watermelons : ℕ := 27

/-- The number of carrots Jessica grew (not used in the proof, but included for completeness) -/
def carrots : ℕ := 30

theorem jessica_watermelons : initial_watermelons - eaten_watermelons = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_watermelons_l444_44440


namespace NUMINAMATH_CALUDE_problem_statement_l444_44411

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x

theorem problem_statement (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2) (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l444_44411


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l444_44497

theorem largest_integer_satisfying_inequality : 
  ∀ n : ℕ+, n^200 < 3^500 ↔ n ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l444_44497


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l444_44483

theorem fraction_zero_implies_x_negative_three (x : ℝ) : 
  (x + 3) / (x - 4) = 0 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l444_44483


namespace NUMINAMATH_CALUDE_intersection_area_bound_l444_44437

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- Define a function to reflect a triangle about a point
def reflectTriangle (t : Triangle) (p : ℝ × ℝ) : Triangle := sorry

-- Define a function to calculate the area of the intersection polygon
noncomputable def intersectionArea (t1 t2 : Triangle) : ℝ := sorry

-- Theorem statement
theorem intersection_area_bound (ABC : Triangle) (P : ℝ × ℝ) :
  intersectionArea ABC (reflectTriangle ABC P) ≤ (2/3) * triangleArea ABC := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_bound_l444_44437


namespace NUMINAMATH_CALUDE_nate_running_distance_l444_44462

/-- The total distance Nate ran in meters -/
def total_distance (field_length : ℝ) (additional_distance : ℝ) : ℝ :=
  4 * field_length + additional_distance

/-- Theorem stating the total distance Nate ran -/
theorem nate_running_distance :
  let field_length : ℝ := 168
  let additional_distance : ℝ := 500
  total_distance field_length additional_distance = 1172 := by
  sorry

end NUMINAMATH_CALUDE_nate_running_distance_l444_44462


namespace NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l444_44449

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 128 ways to distribute 8 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_8_balls_3_boxes : distribute_balls 8 3 = 128 := by
  sorry

end NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l444_44449


namespace NUMINAMATH_CALUDE_min_value_of_function_l444_44400

theorem min_value_of_function (x : ℝ) (h : x > 0) : 4 * x + 1 / x^2 ≥ 5 ∧ ∃ y > 0, 4 * y + 1 / y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l444_44400


namespace NUMINAMATH_CALUDE_mcdonalds_fries_cost_l444_44475

/-- The cost of one pack of fries at McDonald's -/
def fries_cost : ℝ := 2

/-- The cost of a burger at McDonald's -/
def burger_cost : ℝ := 5

/-- The cost of a salad at McDonald's -/
def salad_cost (f : ℝ) : ℝ := 3 * f

/-- The total cost of the meal at McDonald's -/
def total_cost (f : ℝ) : ℝ := salad_cost f + burger_cost + 2 * f

theorem mcdonalds_fries_cost :
  fries_cost = 2 ∧ total_cost fries_cost = 15 :=
sorry

end NUMINAMATH_CALUDE_mcdonalds_fries_cost_l444_44475


namespace NUMINAMATH_CALUDE_xy_plus_one_eq_x_plus_y_l444_44453

theorem xy_plus_one_eq_x_plus_y (x y : ℝ) :
  x * y + 1 = x + y ↔ x = 1 ∨ y = 1 := by
sorry

end NUMINAMATH_CALUDE_xy_plus_one_eq_x_plus_y_l444_44453


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l444_44456

/-- Represents a cross country meet with two teams of 6 runners each. -/
structure CrossCountryMeet where
  runners_per_team : Nat
  total_runners : Nat
  min_score : Nat
  max_score : Nat

/-- Calculates the number of possible winning scores in a cross country meet. -/
def possible_winning_scores (meet : CrossCountryMeet) : Nat :=
  meet.max_score - meet.min_score + 1

/-- Theorem stating the number of possible winning scores in the given cross country meet setup. -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.runners_per_team = 6 ∧
    meet.total_runners = 12 ∧
    meet.min_score = 21 ∧
    meet.max_score = 39 ∧
    possible_winning_scores meet = 19 := by
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l444_44456


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l444_44450

theorem quadratic_real_root_condition (a b c : ℝ) : 
  (∃ x : ℝ, (a^2 + b^2 + c^2) * x^2 + 2*(a - b + c) * x + 3 = 0) →
  (a = c ∧ a = -b) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l444_44450


namespace NUMINAMATH_CALUDE_area_of_ω_l444_44492

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (7, 15)
def B : ℝ × ℝ := (15, 9)

-- State that A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- State that the tangent lines intersect at a point on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : circle_area ω = 6525 * Real.pi / 244 := sorry

end NUMINAMATH_CALUDE_area_of_ω_l444_44492


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l444_44443

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 4 * x - 25
def line2 (x y : ℝ) : Prop := 2 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 125 / 6 := by sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l444_44443


namespace NUMINAMATH_CALUDE_pencil_sale_l444_44458

theorem pencil_sale (x : ℕ) : 
  (2 * x) + (6 * 3) + (2 * 1) = 24 → x = 2 := by sorry

end NUMINAMATH_CALUDE_pencil_sale_l444_44458


namespace NUMINAMATH_CALUDE_adidas_to_skechers_ratio_l444_44402

/-- Proves the ratio of Adidas to Skechers sneakers spending is 1:5 --/
theorem adidas_to_skechers_ratio
  (total_spent : ℕ)
  (nike_to_adidas_ratio : ℕ)
  (adidas_cost : ℕ)
  (clothes_cost : ℕ)
  (h1 : total_spent = 8000)
  (h2 : nike_to_adidas_ratio = 3)
  (h3 : adidas_cost = 600)
  (h4 : clothes_cost = 2600) :
  (adidas_cost : ℚ) / (total_spent - clothes_cost - nike_to_adidas_ratio * adidas_cost - adidas_cost) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_adidas_to_skechers_ratio_l444_44402


namespace NUMINAMATH_CALUDE_total_books_count_l444_44431

def initial_books : ℕ := 35
def bought_books : ℕ := 21

theorem total_books_count : initial_books + bought_books = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l444_44431


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l444_44479

theorem margin_in_terms_of_selling_price 
  (n : ℝ) (C S M : ℝ) 
  (h1 : M = (2/n) * C) 
  (h2 : S = C + M) : 
  M = (2/(n+2)) * S := by
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l444_44479


namespace NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l444_44423

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a frustum -/
def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude

/-- Theorem: The altitude of the small cone cut off from a specific frustum is 18 -/
theorem specific_frustum_small_cone_altitude :
  let f : Frustum := { altitude := 18, lower_base_area := 400 * Real.pi, upper_base_area := 100 * Real.pi }
  small_cone_altitude f = 18 := by sorry

end NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l444_44423


namespace NUMINAMATH_CALUDE_total_age_difference_l444_44441

-- Define the ages of A, B, and C as natural numbers
variable (A B C : ℕ)

-- Define the condition that C is 15 years younger than A
def age_difference : Prop := C = A - 15

-- Define the difference in total ages
def age_sum_difference : ℕ := (A + B) - (B + C)

-- Theorem statement
theorem total_age_difference (h : age_difference A C) : age_sum_difference A B C = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_age_difference_l444_44441


namespace NUMINAMATH_CALUDE_vector_on_line_and_parallel_l444_44474

def line_x (t : ℝ) : ℝ := 5 * t + 3
def line_y (t : ℝ) : ℝ := t + 3

def vector_a : ℝ := 18
def vector_b : ℝ := 6

def parallel_vector_x : ℝ := 3
def parallel_vector_y : ℝ := 1

theorem vector_on_line_and_parallel :
  (∃ t : ℝ, line_x t = vector_a ∧ line_y t = vector_b) ∧
  (∃ k : ℝ, vector_a = k * parallel_vector_x ∧ vector_b = k * parallel_vector_y) :=
sorry

end NUMINAMATH_CALUDE_vector_on_line_and_parallel_l444_44474


namespace NUMINAMATH_CALUDE_cellphone_surveys_count_l444_44428

/-- Represents the weekly survey data for a worker --/
structure SurveyData where
  regularRate : ℕ
  totalSurveys : ℕ
  cellphoneRateIncrease : ℚ
  totalEarnings : ℕ

/-- Calculates the number of cellphone surveys given the survey data --/
def calculateCellphoneSurveys (data : SurveyData) : ℕ :=
  sorry

/-- Theorem stating that the number of cellphone surveys is 50 for the given data --/
theorem cellphone_surveys_count (data : SurveyData) 
  (h1 : data.regularRate = 30)
  (h2 : data.totalSurveys = 100)
  (h3 : data.cellphoneRateIncrease = 1/5)
  (h4 : data.totalEarnings = 3300) :
  calculateCellphoneSurveys data = 50 := by
  sorry

end NUMINAMATH_CALUDE_cellphone_surveys_count_l444_44428


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_two_thirds_l444_44435

theorem tan_alpha_3_implies_fraction_eq_two_thirds (α : Real) (h : Real.tan α = 3) :
  1 / (Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_two_thirds_l444_44435


namespace NUMINAMATH_CALUDE_solve_equation_l444_44488

theorem solve_equation (x : ℝ) : 
  ((2*x + 8) + (7*x + 3) + (3*x + 9)) / 3 = 5*x^2 - 8*x + 2 → 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l444_44488


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l444_44494

theorem sum_of_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) :
  a^10 + b^10 = 123 := by sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l444_44494


namespace NUMINAMATH_CALUDE_triangle_inequalities_l444_44460

/-- Triangle properties and inequalities -/
theorem triangle_inequalities (a b c S h_a h_b h_c r_a r_b r_c r : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0)
  (h_area : S = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))))
  (h_altitude : h_a = 2 * S / a ∧ h_b = 2 * S / b ∧ h_c = 2 * S / c)
  (h_excircle : (r_a * r_b * r_c)^2 = S^4 / r^2) :
  (S^3 ≤ (Real.sqrt 3 / 4)^3 * (a * b * c)^2) ∧
  ((h_a * h_b * h_c)^(1/3) ≤ 3^(1/4) * Real.sqrt S) ∧
  (3^(1/4) * Real.sqrt S ≤ (r_a * r_b * r_c)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l444_44460


namespace NUMINAMATH_CALUDE_hiker_distance_theorem_l444_44417

/-- Calculates the total distance walked by a hiker over three days given specific conditions -/
def total_distance_walked (day1_distance : ℕ) (day1_speed : ℕ) (day2_speed_increase : ℕ) (day3_speed : ℕ) (day3_hours : ℕ) : ℕ :=
  let day1_hours : ℕ := day1_distance / day1_speed
  let day2_hours : ℕ := day1_hours - 1
  let day2_speed : ℕ := day1_speed + day2_speed_increase
  let day2_distance : ℕ := day2_speed * day2_hours
  let day3_distance : ℕ := day3_speed * day3_hours
  day1_distance + day2_distance + day3_distance

/-- Theorem stating that the total distance walked is 53 miles given the specific conditions -/
theorem hiker_distance_theorem :
  total_distance_walked 18 3 1 5 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_theorem_l444_44417


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l444_44438

theorem pirate_treasure_division (S a b c d e : ℚ) : 
  a = (S - a) / 2 →
  b = (S - b) / 3 →
  c = (S - c) / 4 →
  d = (S - d) / 5 →
  e = 90 →
  S = a + b + c + d + e →
  S = 1800 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_division_l444_44438


namespace NUMINAMATH_CALUDE_power_product_squared_l444_44461

theorem power_product_squared (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := by sorry

end NUMINAMATH_CALUDE_power_product_squared_l444_44461


namespace NUMINAMATH_CALUDE_power_sum_equality_l444_44466

theorem power_sum_equality : (-1)^43 + 2^(2^3 + 5^2 - 7^2) = -(65535 / 65536) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l444_44466


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l444_44480

theorem simplify_and_evaluate (a : ℝ) (h : a = 2) : 
  (1 - 1 / (a + 1)) / (a / (a^2 - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l444_44480


namespace NUMINAMATH_CALUDE_parabola_tangent_and_intersection_l444_44425

/-- Parabola in the first quadrant -/
structure Parabola where
  n : ℝ
  pos_n : n > 0

/-- Point on the parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = c.n * x
  first_quadrant : x > 0 ∧ y > 0

/-- Line with slope and y-intercept -/
structure Line where
  m : ℝ
  b : ℝ

/-- Theorem about parabola tangent and intersection properties -/
theorem parabola_tangent_and_intersection
  (c : Parabola)
  (p : ParabolaPoint c)
  (h1 : p.x = 2)
  (h2 : (p.x + c.n / 4)^2 + p.y^2 = (5/2)^2) -- Distance from P to focus is 5/2
  (l₂ : Line)
  (h3 : l₂.m ≠ 0) :
  -- 1. The tangent at P intersects x-axis at (-2, 0)
  ∃ (q : ℝ × ℝ), q = (-2, 0) ∧
    (∃ (k : ℝ), k * (q.1 - p.x) + p.y = q.2 ∧ 
      ∀ (x y : ℝ), y^2 = c.n * x → (y - p.y) = k * (x - p.x) → x = q.1 ∧ y = q.2) ∧
  -- 2. If slopes of PA, PE, PB form arithmetic sequence, l₂ passes through (2, 0)
  (∀ (a b e : ℝ × ℝ),
    (a.2)^2 = c.n * a.1 ∧ (b.2)^2 = c.n * b.1 ∧ -- A and B on parabola
    a.1 = l₂.m * a.2 + l₂.b ∧ b.1 = l₂.m * b.2 + l₂.b ∧ -- A and B on l₂
    e.1 = -2 ∧ e.2 = -(l₂.b + 2) / l₂.m → -- E on l₁
    (((a.2 - p.y) / (a.1 - p.x) + (b.2 - p.y) / (b.1 - p.x)) / 2 = (e.2 - p.y) / (e.1 - p.x)) →
    l₂.b = 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_and_intersection_l444_44425


namespace NUMINAMATH_CALUDE_complex_product_equality_l444_44467

theorem complex_product_equality : (3 + 4*Complex.I) * (2 - 3*Complex.I) * (1 + 2*Complex.I) = 20 + 35*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equality_l444_44467


namespace NUMINAMATH_CALUDE_student_assignment_count_l444_44412

/-- The number of ways to assign students to internship positions -/
def assignment_count (n_students : ℕ) (n_positions : ℕ) : ℕ :=
  (n_students.choose 2) * (n_positions.factorial)

/-- Theorem: There are 36 ways to assign 4 students to 3 internship positions -/
theorem student_assignment_count :
  assignment_count 4 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_student_assignment_count_l444_44412


namespace NUMINAMATH_CALUDE_largest_solution_sum_l444_44422

noncomputable def f (x : ℝ) : ℝ := 
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20)

theorem largest_solution_sum (n : ℝ) (p q r : ℕ+) :
  (∀ x : ℝ, f x = x^2 - 13*x - 6 → x ≤ n) ∧
  f n = n^2 - 13*n - 6 ∧
  n = p + Real.sqrt (q + Real.sqrt r) →
  p + q + r = 309 := by sorry

end NUMINAMATH_CALUDE_largest_solution_sum_l444_44422


namespace NUMINAMATH_CALUDE_long_jump_solution_l444_44455

/-- Represents the long jump problem with given conditions -/
def LongJumpProblem (initial_avg : ℝ) (second_jump : ℝ) (second_avg : ℝ) (final_avg : ℝ) : Prop :=
  ∃ (n : ℕ) (third_jump : ℝ),
    -- Initial condition
    initial_avg = 3.80
    -- Second jump condition
    ∧ second_jump = 3.99
    -- New average after second jump
    ∧ second_avg = 3.81
    -- Final average after third jump
    ∧ final_avg = 3.82
    -- Relationship between jumps and averages
    ∧ (initial_avg * n + second_jump) / (n + 1) = second_avg
    ∧ (initial_avg * n + second_jump + third_jump) / (n + 2) = final_avg
    -- The third jump is the solution
    ∧ third_jump = 4.01

/-- Theorem stating the solution to the long jump problem -/
theorem long_jump_solution :
  LongJumpProblem 3.80 3.99 3.81 3.82 :=
by
  sorry

#check long_jump_solution

end NUMINAMATH_CALUDE_long_jump_solution_l444_44455


namespace NUMINAMATH_CALUDE_rocket_soaring_time_l444_44489

/-- Proves that the soaring time of a rocket is 12 seconds given specific conditions -/
theorem rocket_soaring_time :
  let soaring_speed : ℝ := 150
  let plummet_distance : ℝ := 600
  let plummet_time : ℝ := 3
  let average_speed : ℝ := 160
  let soaring_time : ℝ := 12

  (soaring_speed * soaring_time + plummet_distance) / (soaring_time + plummet_time) = average_speed :=
by
  sorry


end NUMINAMATH_CALUDE_rocket_soaring_time_l444_44489


namespace NUMINAMATH_CALUDE_repeating_decimal_simplest_form_sum_of_numerator_and_denominator_l444_44478

def repeating_decimal : ℚ := 24/99

theorem repeating_decimal_simplest_form : 
  repeating_decimal = 8/33 := by sorry

theorem sum_of_numerator_and_denominator : 
  (Nat.gcd 8 33 = 1) ∧ (8 + 33 = 41) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_simplest_form_sum_of_numerator_and_denominator_l444_44478


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l444_44405

/-- Given a curve in polar coordinates r = p * sin(5θ), 
    this theorem states its equivalent form in Cartesian coordinates. -/
theorem polar_to_cartesian (p : ℝ) (x y : ℝ) :
  (∃ (θ : ℝ), x = (p * Real.sin (5 * θ)) * Real.cos θ ∧
               y = (p * Real.sin (5 * θ)) * Real.sin θ) ↔
  x^6 - 5*p*x^4*y + 10*p*x^2*y^3 + y^6 + 3*x^4*y^2 - p*y^5 + 3*x^2*y^4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l444_44405


namespace NUMINAMATH_CALUDE_garage_sale_items_l444_44490

theorem garage_sale_items (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card = 43 ∧ radio_price ∈ prices ∧ 
  (prices.filter (λ x => x > radio_price)).card = 8 ∧
  (prices.filter (λ x => x < radio_price)).card = 34 →
  prices.card = 43 :=
by sorry

end NUMINAMATH_CALUDE_garage_sale_items_l444_44490


namespace NUMINAMATH_CALUDE_expression_value_l444_44451

theorem expression_value : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l444_44451


namespace NUMINAMATH_CALUDE_remainder_3m_mod_5_l444_44401

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3m_mod_5_l444_44401


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l444_44487

theorem square_difference_of_integers (a b : ℕ+) 
  (sum_eq : a + b = 70) 
  (diff_eq : a - b = 20) : 
  a ^ 2 - b ^ 2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l444_44487


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l444_44418

/-- Proves that the percentage in excess for the first side is 12% given the conditions of the problem -/
theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) :
  (L * (1 + x / 100) * (W * 0.95) = L * W * 1.064) →
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l444_44418


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l444_44481

theorem gcd_of_three_numbers : Nat.gcd 10711 (Nat.gcd 15809 28041) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l444_44481


namespace NUMINAMATH_CALUDE_lighthouse_model_height_l444_44424

def original_height : ℝ := 60
def original_base_height : ℝ := 12
def original_base_volume : ℝ := 150000
def model_base_volume : ℝ := 0.15

theorem lighthouse_model_height :
  let scale_factor := (model_base_volume / original_base_volume) ^ (1/3)
  let model_height := original_height * scale_factor
  model_height * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_lighthouse_model_height_l444_44424


namespace NUMINAMATH_CALUDE_correct_quadratic_not_in_options_l444_44410

theorem correct_quadratic_not_in_options : ∀ b c : ℝ,
  (∃ x y : ℝ, x + y = 10 ∧ x * y = c) →  -- From the first student's roots
  (∃ u v : ℝ, u + v = b ∧ u * v = -10) →  -- From the second student's roots
  (b = -10 ∧ c = -10) →  -- Derived from the conditions
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x^2 - 10*x - 10 = 0) →
  (x^2 + b*x + c ≠ x^2 - 9*x + 10) ∧
  (x^2 + b*x + c ≠ x^2 + 9*x + 10) ∧
  (x^2 + b*x + c ≠ x^2 - 9*x + 12) ∧
  (x^2 + b*x + c ≠ x^2 + 10*x - 21) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_not_in_options_l444_44410


namespace NUMINAMATH_CALUDE_commission_percentage_problem_l444_44484

/-- Calculates the commission percentage given the commission amount and total sales. -/
def commission_percentage (commission : ℚ) (total_sales : ℚ) : ℚ :=
  (commission / total_sales) * 100

/-- Theorem stating that for the given commission and sales values, the commission percentage is 4%. -/
theorem commission_percentage_problem :
  let commission : ℚ := 25/2  -- Rs. 12.50
  let total_sales : ℚ := 625/2  -- Rs. 312.5
  commission_percentage commission total_sales = 4 := by
  sorry

end NUMINAMATH_CALUDE_commission_percentage_problem_l444_44484


namespace NUMINAMATH_CALUDE_female_students_count_l444_44426

theorem female_students_count (total : ℕ) (ways : ℕ) (f : ℕ) : 
  total = 8 → 
  ways = 30 → 
  (total - f) * (total - f - 1) * f = 2 * ways → 
  f = 3 := by
sorry

end NUMINAMATH_CALUDE_female_students_count_l444_44426


namespace NUMINAMATH_CALUDE_min_decimal_digits_l444_44498

def fraction : ℚ := 987654321 / (2^30 * 5^6)

theorem min_decimal_digits (n : ℕ) : n = 30 ↔ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, fraction * 10^m ≠ k) ∧ 
  (∃ k : ℕ, fraction * 10^n = k) :=
sorry

end NUMINAMATH_CALUDE_min_decimal_digits_l444_44498


namespace NUMINAMATH_CALUDE_zero_derivative_not_always_extremum_l444_44473

/-- A function f: ℝ → ℝ is differentiable -/
def DifferentiableFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f

/-- x₀ is an extremum point of f if it's either a local maximum or minimum -/
def IsExtremumPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  IsLocalMax f x₀ ∨ IsLocalMin f x₀

/-- The statement that if f'(x₀) = 0, then x₀ is an extremum point of f -/
def ZeroDerivativeImpliesExtremum (f : ℝ → ℝ) : Prop :=
  ∀ x₀ : ℝ, DifferentiableAt ℝ f x₀ → deriv f x₀ = 0 → IsExtremumPoint f x₀

theorem zero_derivative_not_always_extremum :
  ¬ (∀ f : ℝ → ℝ, DifferentiableFunction f → ZeroDerivativeImpliesExtremum f) :=
by sorry

end NUMINAMATH_CALUDE_zero_derivative_not_always_extremum_l444_44473


namespace NUMINAMATH_CALUDE_total_stuffed_animals_l444_44452

/-- 
Given:
- x: initial number of stuffed animals
- y: additional stuffed animals from mom
- z: factor of increase from dad's gift

Prove: The total number of stuffed animals is (x + y) * (1 + z)
-/
theorem total_stuffed_animals (x y : ℕ) (z : ℝ) :
  (x + y : ℝ) * (1 + z) = x + y + z * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_total_stuffed_animals_l444_44452


namespace NUMINAMATH_CALUDE_parabola_max_value_l444_44471

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Whether a parabola opens downwards -/
def opens_downwards (p : Parabola) : Prop := p.a < 0

/-- The maximum value of a parabola -/
def max_value (p : Parabola) : ℝ := sorry

theorem parabola_max_value (p : Parabola) 
  (h1 : vertex p = (-3, 2)) 
  (h2 : opens_downwards p) : 
  max_value p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_max_value_l444_44471


namespace NUMINAMATH_CALUDE_quarters_left_l444_44457

/-- Given that Adam started with 88 quarters and spent 9 quarters at the arcade,
    prove that he had 79 quarters left. -/
theorem quarters_left (initial_quarters spent_quarters : ℕ) 
  (h1 : initial_quarters = 88)
  (h2 : spent_quarters = 9) :
  initial_quarters - spent_quarters = 79 := by
  sorry

end NUMINAMATH_CALUDE_quarters_left_l444_44457
