import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l2189_218912

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  2 * (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - Real.sqrt 2) + 6 = 5 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2189_218912


namespace NUMINAMATH_CALUDE_unique_factor_solution_l2189_218976

theorem unique_factor_solution (A B C D : ℕ+) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 72 →
  C * D = 72 →
  A + B = C - D →
  A = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_factor_solution_l2189_218976


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l2189_218925

/-- Proves that if a shopkeeper sells an article with a 4% discount and earns a 20% profit,
    then the profit percentage without discount would be 25%. -/
theorem shopkeeper_profit (cost_price : ℝ) (cost_price_pos : 0 < cost_price) :
  let discount_rate : ℝ := 0.04
  let profit_rate_with_discount : ℝ := 0.20
  let selling_price_with_discount : ℝ := cost_price * (1 + profit_rate_with_discount)
  let marked_price : ℝ := selling_price_with_discount / (1 - discount_rate)
  let profit_rate_without_discount : ℝ := (marked_price - cost_price) / cost_price
  profit_rate_without_discount = 0.25 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l2189_218925


namespace NUMINAMATH_CALUDE_handshake_count_l2189_218978

theorem handshake_count (n : ℕ) (couples : ℕ) (extra_exemptions : ℕ) : 
  n = 2 * couples → 
  n ≥ 2 →
  extra_exemptions ≤ n - 2 →
  (n * (n - 2) - extra_exemptions) / 2 = 57 :=
by
  sorry

#check handshake_count 12 6 2

end NUMINAMATH_CALUDE_handshake_count_l2189_218978


namespace NUMINAMATH_CALUDE_sandbox_cost_l2189_218931

/-- Calculates the cost of filling an L-shaped sandbox with sand -/
theorem sandbox_cost (short_length short_width short_depth long_length long_width long_depth sand_cost discount_threshold discount_rate : ℝ) :
  let short_volume := short_length * short_width * short_depth
  let long_volume := long_length * long_width * long_depth
  let total_volume := short_volume + long_volume
  let base_cost := total_volume * sand_cost
  let discounted_cost := if total_volume > discount_threshold then base_cost * (1 - discount_rate) else base_cost
  short_length = 3 ∧ 
  short_width = 2 ∧ 
  short_depth = 2 ∧ 
  long_length = 5 ∧ 
  long_width = 2 ∧ 
  long_depth = 2 ∧ 
  sand_cost = 3 ∧ 
  discount_threshold = 20 ∧ 
  discount_rate = 0.1 →
  discounted_cost = 86.4 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_cost_l2189_218931


namespace NUMINAMATH_CALUDE_exponent_division_l2189_218984

theorem exponent_division (a : ℝ) (m n : ℕ) (h : a ≠ 0) : a^m / a^n = a^(m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2189_218984


namespace NUMINAMATH_CALUDE_wendy_recycling_l2189_218986

/-- Given that Wendy earns 5 points per bag recycled, had 11 bags in total, 
    and earned 45 points, prove that she did not recycle 2 bags. -/
theorem wendy_recycling (points_per_bag : ℕ) (total_bags : ℕ) (total_points : ℕ) 
  (h1 : points_per_bag = 5)
  (h2 : total_bags = 11)
  (h3 : total_points = 45) :
  total_bags - (total_points / points_per_bag) = 2 := by
  sorry


end NUMINAMATH_CALUDE_wendy_recycling_l2189_218986


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2189_218953

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := Real.exp x

/-- A point on the curve where the tangent line passes through -/
noncomputable def x₀ : ℝ := 1

/-- The slope of the tangent line -/
noncomputable def k : ℝ := f' x₀

theorem tangent_line_slope : k = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2189_218953


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2189_218946

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 4 < x ∧ x < 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2189_218946


namespace NUMINAMATH_CALUDE_fourth_term_is_2016_l2189_218921

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  second_term : a 2 = 606
  sum_first_four : a 1 + a 2 + a 3 + a 4 = 3834

/-- The fourth term of the arithmetic sequence is 2016 -/
theorem fourth_term_is_2016 (seq : ArithmeticSequence) : seq.a 4 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_2016_l2189_218921


namespace NUMINAMATH_CALUDE_soccer_campers_l2189_218962

theorem soccer_campers (total : ℕ) (basketball : ℕ) (football : ℕ) 
  (h1 : total = 88) 
  (h2 : basketball = 24) 
  (h3 : football = 32) : 
  total - (basketball + football) = 32 := by
  sorry

end NUMINAMATH_CALUDE_soccer_campers_l2189_218962


namespace NUMINAMATH_CALUDE_julia_short_amount_l2189_218947

def rock_price : ℝ := 5
def pop_price : ℝ := 10
def dance_price : ℝ := 3
def country_price : ℝ := 7
def discount_rate : ℝ := 0.1
def julia_money : ℝ := 75

def rock_quantity : ℕ := 3
def pop_quantity : ℕ := 4
def dance_quantity : ℕ := 2
def country_quantity : ℕ := 4

def discount_threshold : ℕ := 3

def genre_cost (price : ℝ) (quantity : ℕ) : ℝ := price * quantity

def apply_discount (cost : ℝ) (quantity : ℕ) : ℝ :=
  if quantity ≥ discount_threshold then cost * (1 - discount_rate) else cost

theorem julia_short_amount : 
  let rock_cost := apply_discount (genre_cost rock_price rock_quantity) rock_quantity
  let pop_cost := apply_discount (genre_cost pop_price pop_quantity) pop_quantity
  let dance_cost := apply_discount (genre_cost dance_price dance_quantity) dance_quantity
  let country_cost := apply_discount (genre_cost country_price country_quantity) country_quantity
  let total_cost := rock_cost + pop_cost + dance_cost + country_cost
  total_cost - julia_money = 7.2 := by sorry

end NUMINAMATH_CALUDE_julia_short_amount_l2189_218947


namespace NUMINAMATH_CALUDE_tangent_circles_theorem_l2189_218954

/-- Two circles in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane -/
def Point : Type := ℝ × ℝ

/-- Predicate to check if two circles are tangent -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on the common tangent of two circles -/
def on_common_tangent (p : Point) (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if the common tangent is perpendicular to the line joining the centers -/
def perpendicular_to_center_line (p : Point) (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a circle is tangent to another circle -/
def is_tangent_to (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem tangent_circles_theorem 
  (c1 c2 : Circle) 
  (p : Point) 
  (h1 : are_tangent c1 c2)
  (h2 : on_common_tangent p c1 c2)
  (h3 : perpendicular_to_center_line p c1 c2) :
  ∃! (s1 s2 : Circle), 
    s1 ≠ s2 ∧ 
    is_tangent_to s1 c1 ∧ 
    is_tangent_to s1 c2 ∧ 
    point_on_circle p s1 ∧
    is_tangent_to s2 c1 ∧ 
    is_tangent_to s2 c2 ∧ 
    point_on_circle p s2 :=
  sorry

end NUMINAMATH_CALUDE_tangent_circles_theorem_l2189_218954


namespace NUMINAMATH_CALUDE_line_slope_l2189_218933

/-- The slope of the line given by the equation x/4 + y/3 = 1 is -3/4 -/
theorem line_slope (x y : ℝ) :
  x / 4 + y / 3 = 1 → (∃ b : ℝ, y = -(3/4) * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2189_218933


namespace NUMINAMATH_CALUDE_rhombus_area_l2189_218952

/-- The area of a rhombus with sides of length 3 cm and one internal angle of 45 degrees is (9√2)/2 square centimeters. -/
theorem rhombus_area (s : ℝ) (angle : ℝ) (h1 : s = 3) (h2 : angle = 45 * π / 180) :
  let area := s * s * Real.sin angle
  area = 9 * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l2189_218952


namespace NUMINAMATH_CALUDE_wood_piece_weight_relation_l2189_218981

/-- Represents a square piece of wood -/
structure WoodPiece where
  sideLength : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square wood pieces -/
theorem wood_piece_weight_relation 
  (piece1 piece2 : WoodPiece)
  (h1 : piece1.sideLength = 4)
  (h2 : piece1.weight = 16)
  (h3 : piece2.sideLength = 6)
  : piece2.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_wood_piece_weight_relation_l2189_218981


namespace NUMINAMATH_CALUDE_price_difference_shirt_sweater_l2189_218950

theorem price_difference_shirt_sweater : 
  ∀ (shirt_price sweater_price : ℝ),
    shirt_price = 36.46 →
    shirt_price < sweater_price →
    shirt_price + sweater_price = 80.34 →
    sweater_price - shirt_price = 7.42 := by
sorry

end NUMINAMATH_CALUDE_price_difference_shirt_sweater_l2189_218950


namespace NUMINAMATH_CALUDE_four_good_numbers_l2189_218996

/-- A real number k is a "good number" if the equation (x^2 - 1)(kx^2 - 6x - 8) = 0 
    has exactly three distinct real roots. -/
def is_good_number (k : ℝ) : Prop :=
  ∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    ∀ x : ℝ, (x^2 - 1) * (k * x^2 - 6 * x - 8) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃

/-- There are exactly 4 "good numbers". -/
theorem four_good_numbers : ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ k : ℝ, k ∈ s ↔ is_good_number k :=
  sorry

end NUMINAMATH_CALUDE_four_good_numbers_l2189_218996


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2189_218992

/-- Theorem: For a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has equation y = 3x, then b = 3. -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) :
  (∃ x y : ℝ, x^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = 3 * x ∧ x^2 - y^2 / b^2 = 1) →
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2189_218992


namespace NUMINAMATH_CALUDE_lindas_coins_value_l2189_218987

theorem lindas_coins_value :
  ∀ (n d q : ℕ),
  n + d + q = 30 →
  10 * n + 25 * d + 5 * q = 5 * n + 10 * d + 25 * q + 150 →
  5 * n + 10 * d + 25 * q = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_lindas_coins_value_l2189_218987


namespace NUMINAMATH_CALUDE_difference_of_squares_identity_l2189_218928

theorem difference_of_squares_identity (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_identity_l2189_218928


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficients_l2189_218971

theorem quadratic_inequality_coefficients 
  (a b : ℝ) 
  (h1 : Set.Ioo (-2 : ℝ) (-1/4 : ℝ) = {x : ℝ | 5 - x > 7 * |x + 1|})
  (h2 : Set.Ioo (-2 : ℝ) (-1/4 : ℝ) = {x : ℝ | a * x^2 + b * x - 2 > 0}) :
  a = -4 ∧ b = -9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficients_l2189_218971


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_one_l2189_218977

/-- The equation of a line passing through (-1, -1) with slope 1 is y = x -/
theorem line_equation_through_point_with_slope_one :
  ∀ (x y : ℝ), (y + 1 = 1 * (x + 1)) ↔ (y = x) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_one_l2189_218977


namespace NUMINAMATH_CALUDE_sphere_radius_l2189_218991

theorem sphere_radius (d h : ℝ) (h1 : d = 30) (h2 : h = 10) : 
  ∃ r : ℝ, r^2 = d^2 / 4 + h^2 ∧ r = 5 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_l2189_218991


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2189_218905

/-- Represents the side lengths of squares in the rectangle -/
structure SquareSides where
  smallest : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  largest : ℝ

/-- The rectangle composed of eight squares -/
structure Rectangle where
  sides : SquareSides
  length : ℝ
  width : ℝ

/-- Theorem stating the perimeter of the rectangle -/
theorem rectangle_perimeter (rect : Rectangle) 
  (h1 : rect.sides.smallest = 1)
  (h2 : rect.sides.a = 4)
  (h3 : rect.sides.b = 5)
  (h4 : rect.sides.c = 5)
  (h5 : rect.sides.largest = 14)
  (h6 : rect.length = rect.sides.largest + rect.sides.b)
  (h7 : rect.width = rect.sides.largest) : 
  2 * (rect.length + rect.width) = 66 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2189_218905


namespace NUMINAMATH_CALUDE_unique_prime_sum_10002_l2189_218909

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_sum_10002 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10002 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10002_l2189_218909


namespace NUMINAMATH_CALUDE_triangle_area_l2189_218917

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 2 →
  A = π / 4 →
  B = π / 3 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2189_218917


namespace NUMINAMATH_CALUDE_michael_and_anna_ages_l2189_218969

theorem michael_and_anna_ages :
  ∀ (michael anna : ℕ),
  michael = anna + 8 →
  michael + 12 = 3 * (anna - 6) →
  michael + anna = 46 :=
by sorry

end NUMINAMATH_CALUDE_michael_and_anna_ages_l2189_218969


namespace NUMINAMATH_CALUDE_triangle_FIL_area_l2189_218940

-- Define the triangle and squares
structure Triangle :=
  (F I L : ℝ × ℝ)

structure Square :=
  (area : ℝ)

-- Define the problem setup
def triangle_FIL : Triangle := sorry
def square_GQOP : Square := ⟨10⟩
def square_HJNO : Square := ⟨90⟩
def square_RKMN : Square := ⟨40⟩

-- Function to check if squares are on triangle sides
def squares_on_triangle_sides (t : Triangle) (s1 s2 s3 : Square) : Prop := sorry

-- Function to calculate triangle area
def triangle_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_FIL_area :
  squares_on_triangle_sides triangle_FIL square_GQOP square_HJNO square_RKMN →
  triangle_area triangle_FIL = 220.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_FIL_area_l2189_218940


namespace NUMINAMATH_CALUDE_average_problem_k_problem_point_problem_quadratic_problem_l2189_218936

-- Question 1
theorem average_problem (p q r t : ℝ) :
  (p + q + r) / 3 = 12 ∧ (p + q + r + t + 2*t) / 5 = 15 → t = 13 := by sorry

-- Question 2
theorem k_problem (k s : ℝ) :
  k^4 + 1/k^4 = 14 ∧ s = k^2 + 1/k^2 → s = 4 := by sorry

-- Question 3
theorem point_problem (a b s : ℝ) :
  let M : ℝ × ℝ := (1, 2)
  let N : ℝ × ℝ := (11, 7)
  let P : ℝ × ℝ := (a, b)
  P.1 = (1 * N.1 + s * M.1) / (1 + s) ∧
  P.2 = (1 * N.2 + s * M.2) / (1 + s) ∧
  s = 4 → a = 3 := by sorry

-- Question 4
theorem quadratic_problem (a c : ℝ) :
  a = 3 ∧ (∃ x : ℝ, a * x^2 + 12 * x + c = 0 ∧
    ∀ y : ℝ, y ≠ x → a * y^2 + 12 * y + c ≠ 0) → c = 12 := by sorry

end NUMINAMATH_CALUDE_average_problem_k_problem_point_problem_quadratic_problem_l2189_218936


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l2189_218913

theorem sum_of_coefficients_equals_one (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (1 - 2*x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l2189_218913


namespace NUMINAMATH_CALUDE_vasya_fool_count_l2189_218901

/-- Represents the number of times a player was left as the "fool" -/
structure FoolCount where
  count : ℕ
  positive : count > 0

/-- The game "Fool" with four players -/
structure FoolGame where
  misha : FoolCount
  petya : FoolCount
  kolya : FoolCount
  vasya : FoolCount
  total_games : misha.count + petya.count + kolya.count + vasya.count = 16
  misha_most : misha.count > petya.count ∧ misha.count > kolya.count ∧ misha.count > vasya.count
  petya_kolya_sum : petya.count + kolya.count = 9

theorem vasya_fool_count (game : FoolGame) : game.vasya.count = 1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_fool_count_l2189_218901


namespace NUMINAMATH_CALUDE_midpoint_movement_l2189_218920

/-- Given two points A and B on a Cartesian plane, their midpoint, and their new positions after moving,
    prove that the new midpoint and its distance from the original midpoint are as calculated. -/
theorem midpoint_movement (a b c d m n : ℝ) :
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (c, d)
  let M : ℝ × ℝ := (m, n)
  let A' : ℝ × ℝ := (a + 3, b + 5)
  let B' : ℝ × ℝ := (c - 4, d - 6)
  M = ((a + c) / 2, (b + d) / 2) →
  let M' : ℝ × ℝ := ((A'.1 + B'.1) / 2, (A'.2 + B'.2) / 2)
  M' = (m - 0.5, n - 0.5) ∧
  Real.sqrt ((M'.1 - M.1)^2 + (M'.2 - M.2)^2) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_movement_l2189_218920


namespace NUMINAMATH_CALUDE_part_one_part_two_l2189_218903

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a / Real.cos t.A = (3 * t.c - 2 * t.b) / Real.cos t.B

-- Theorem for part (1)
theorem part_one (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.b = Real.sqrt 5 * Real.sin t.B) : 
  t.a = 5/3 := by
sorry

-- Theorem for part (2)
theorem part_two (t : Triangle) 
  (h1 : triangle_condition t)
  (h2 : t.a = Real.sqrt 6)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 5 / 2) :
  t.b + t.c = 4 := by
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2189_218903


namespace NUMINAMATH_CALUDE_tax_rate_percentage_l2189_218914

/-- Given a tax rate of $82 per $100.00, prove that it is equivalent to 82% -/
theorem tax_rate_percentage : 
  let tax_amount : ℚ := 82
  let base_amount : ℚ := 100
  (tax_amount / base_amount) * 100 = 82 := by sorry

end NUMINAMATH_CALUDE_tax_rate_percentage_l2189_218914


namespace NUMINAMATH_CALUDE_power_product_equality_l2189_218922

theorem power_product_equality (a : ℝ) : 4 * a^2 * a = 4 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2189_218922


namespace NUMINAMATH_CALUDE_number_equality_l2189_218923

theorem number_equality : ∃ y : ℝ, 0.4 * y = (1/3) * 45 ∧ y = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2189_218923


namespace NUMINAMATH_CALUDE_sara_quarters_l2189_218966

/-- The number of quarters Sara has after receiving more from her dad -/
def total_quarters (initial : ℕ) (received : ℕ) : ℕ := initial + received

/-- Theorem stating that Sara now has 70 quarters -/
theorem sara_quarters : total_quarters 21 49 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l2189_218966


namespace NUMINAMATH_CALUDE_admission_ways_correct_l2189_218900

/-- The number of ways to assign three students to exactly two colleges out of 23 colleges -/
def admission_ways : ℕ := 1518

/-- The number of colleges recruiting students -/
def num_colleges : ℕ := 23

/-- The number of students to be admitted -/
def num_students : ℕ := 3

/-- The number of colleges each student is admitted to -/
def colleges_per_student : ℕ := 2

theorem admission_ways_correct : 
  admission_ways = (num_students.choose 1) * (colleges_per_student.choose colleges_per_student) * (num_colleges.choose colleges_per_student) :=
by sorry

end NUMINAMATH_CALUDE_admission_ways_correct_l2189_218900


namespace NUMINAMATH_CALUDE_initial_maple_trees_l2189_218975

/-- The number of maple trees in the park after planting -/
def total_trees : ℕ := 64

/-- The number of maple trees planted today -/
def planted_trees : ℕ := 11

/-- The initial number of maple trees in the park -/
def initial_trees : ℕ := total_trees - planted_trees

theorem initial_maple_trees : initial_trees = 53 := by
  sorry

end NUMINAMATH_CALUDE_initial_maple_trees_l2189_218975


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l2189_218979

theorem students_playing_both_sports (total : ℕ) (hockey : ℕ) (basketball : ℕ) (neither : ℕ) :
  total = 50 →
  hockey = 30 →
  basketball = 35 →
  neither = 10 →
  hockey + basketball - (total - neither) = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l2189_218979


namespace NUMINAMATH_CALUDE_tom_pennies_l2189_218930

/-- Represents the number of coins of each type --/
structure CoinCounts where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents given a CoinCounts --/
def totalValueInCents (coins : CoinCounts) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- The main theorem --/
theorem tom_pennies (coins : CoinCounts) 
    (h1 : coins.quarters = 10)
    (h2 : coins.dimes = 3)
    (h3 : coins.nickels = 4)
    (h4 : totalValueInCents coins = 500) :
    coins.pennies = 200 := by
  sorry


end NUMINAMATH_CALUDE_tom_pennies_l2189_218930


namespace NUMINAMATH_CALUDE_geometric_sequence_divisibility_l2189_218907

theorem geometric_sequence_divisibility (a₁ a₂ : ℚ) (n : ℕ) : 
  a₁ = 5/8 → a₂ = 25 → 
  (∃ k : ℕ, k > 0 ∧ (a₂/a₁)^(k-1) * a₁ % 2000000 = 0) →
  (∀ m : ℕ, m > 0 ∧ m < n → (a₂/a₁)^(m-1) * a₁ % 2000000 ≠ 0) →
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_divisibility_l2189_218907


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2189_218906

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (-1 < x ∧ x < 1) ↔ (2*x - a < 1 ∧ x - 2*b > 3)) → 
  (a + 1) * (b - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2189_218906


namespace NUMINAMATH_CALUDE_slope_range_l2189_218951

-- Define the circle F
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory of point P
def trajectory_P (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x < 0 ∧ y = 0)

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the condition for points A and B
def condition_AB (xA yA xB yB : ℝ) : Prop :=
  xA * xB + yA * yB = -4 ∧
  4 * Real.sqrt 6 ≤ Real.sqrt ((xB - xA)^2 + (yB - yA)^2) ∧
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2) ≤ 4 * Real.sqrt 30

-- Theorem statement
theorem slope_range (k : ℝ) :
  (∃ m xA yA xB yB,
    circle_F 1 0 ∧
    trajectory_P xA yA ∧ trajectory_P xB yB ∧
    line_l k m xA yA ∧ line_l k m xB yB ∧
    condition_AB xA yA xB yB ∧
    xA > 0 ∧ xB > 0 ∧ xA ≠ xB) →
  (k ∈ Set.Icc (-1) (-1/2) ∨ k ∈ Set.Icc (1/2) 1) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l2189_218951


namespace NUMINAMATH_CALUDE_only_81_satisfies_l2189_218989

/-- A function that returns true if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that swaps the digits of a two-digit number -/
def swapDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The main theorem -/
theorem only_81_satisfies : ∃! n : ℕ, isTwoDigit n ∧ (swapDigits n)^2 = 4 * n :=
  sorry

end NUMINAMATH_CALUDE_only_81_satisfies_l2189_218989


namespace NUMINAMATH_CALUDE_opposite_numbers_quotient_l2189_218958

theorem opposite_numbers_quotient (p q : ℝ) (h1 : p ≠ 0) (h2 : p + q = 0) : |q| / p = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_quotient_l2189_218958


namespace NUMINAMATH_CALUDE_constant_function_solution_l2189_218943

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

theorem constant_function_solution
  (f : ℝ → ℝ)
  (hf : FunctionalEquation f)
  (hnz : ∃ x, f x ≠ 0) :
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k := by
  sorry

end NUMINAMATH_CALUDE_constant_function_solution_l2189_218943


namespace NUMINAMATH_CALUDE_two_colonies_growth_time_l2189_218964

/-- Represents the number of days it takes for a colony to reach the habitat's limit -/
def daysToLimit : ℕ := 21

/-- Represents the daily growth factor of a bacteria colony -/
def growthFactor : ℕ := 2

/-- Represents the number of initial colonies -/
def initialColonies : ℕ := 2

theorem two_colonies_growth_time (daysToLimit : ℕ) (growthFactor : ℕ) (initialColonies : ℕ) :
  daysToLimit = 21 ∧ growthFactor = 2 ∧ initialColonies = 2 →
  daysToLimit = 21 :=
by sorry

end NUMINAMATH_CALUDE_two_colonies_growth_time_l2189_218964


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_negative_one_l2189_218941

theorem fraction_meaningful_iff_not_negative_one (x : ℝ) :
  (∃ y : ℝ, y = (x - 1) / (x + 1)) ↔ x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_negative_one_l2189_218941


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2189_218902

-- Define the hyperbola C
structure Hyperbola where
  -- The equation of the hyperbola in the form ax² + by² = c
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions of the problem
def hyperbola_conditions (C : Hyperbola) : Prop :=
  -- Center at origin (implied by the standard form)
  -- Asymptote y = √2x
  C.a / C.b = -2 ∧
  -- Point P(2√2, -√2) lies on C
  C.a * (2 * Real.sqrt 2)^2 + C.b * (-Real.sqrt 2)^2 = C.c

-- The theorem to prove
theorem hyperbola_equation (C : Hyperbola) :
  hyperbola_conditions C →
  C.a = 1/7 ∧ C.b = -1/14 ∧ C.c = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2189_218902


namespace NUMINAMATH_CALUDE_gym_budget_problem_l2189_218932

/-- Proves that given a budget that allows for the purchase of 10 softballs at $9 each after a 20% increase,
    the original budget would allow for the purchase of 15 dodgeballs at $5 each. -/
theorem gym_budget_problem (original_budget : ℝ) 
  (h1 : original_budget * 1.2 = 10 * 9) 
  (h2 : original_budget > 0) : 
  original_budget / 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_gym_budget_problem_l2189_218932


namespace NUMINAMATH_CALUDE_pool_filling_time_pool_filling_time_is_50_hours_l2189_218904

/-- The time required to fill a swimming pool given the hose flow rate, water cost, and total cost to fill the pool. -/
theorem pool_filling_time 
  (hose_flow_rate : ℝ) 
  (water_cost_per_ten_gallons : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let cost_per_gallon := water_cost_per_ten_gallons / 10
  let total_gallons := total_cost / cost_per_gallon
  total_gallons / hose_flow_rate

/-- The time to fill the pool is 50 hours. -/
theorem pool_filling_time_is_50_hours 
  (hose_flow_rate : ℝ) 
  (water_cost_per_ten_gallons : ℝ) 
  (total_cost : ℝ) 
  (h1 : hose_flow_rate = 100)
  (h2 : water_cost_per_ten_gallons = 1)
  (h3 : total_cost = 5) : 
  pool_filling_time hose_flow_rate water_cost_per_ten_gallons total_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_pool_filling_time_is_50_hours_l2189_218904


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2189_218998

theorem sum_of_squares_and_square_of_sum : (3 + 5 + 7)^2 + (3^2 + 5^2 + 7^2) = 308 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2189_218998


namespace NUMINAMATH_CALUDE_revenue_change_l2189_218985

theorem revenue_change (revenue_1995 : ℝ) : 
  let revenue_1996 := revenue_1995 * 1.2
  let revenue_1997 := revenue_1996 * 0.8
  (revenue_1995 - revenue_1997) / revenue_1995 * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_revenue_change_l2189_218985


namespace NUMINAMATH_CALUDE_f_2a_equals_7_l2189_218919

def f (x : ℝ) : ℝ := 2 * x + 2 - x

theorem f_2a_equals_7 (a : ℝ) (h : f a = 3) : f (2 * a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_2a_equals_7_l2189_218919


namespace NUMINAMATH_CALUDE_cylinder_height_comparison_l2189_218908

/-- Theorem: Comparing cylinder heights with equal volumes and different radii -/
theorem cylinder_height_comparison (r₁ h₁ r₂ h₂ : ℝ) 
  (volume_eq : r₁ ^ 2 * h₁ = r₂ ^ 2 * h₂)
  (radius_relation : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
sorry

end NUMINAMATH_CALUDE_cylinder_height_comparison_l2189_218908


namespace NUMINAMATH_CALUDE_right_triangle_area_l2189_218911

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) (h5 : a^2 = 36) (h6 : b^2 = 64) : (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2189_218911


namespace NUMINAMATH_CALUDE_average_temperature_l2189_218961

/-- The average temperature for Monday, Tuesday, Wednesday, and Thursday given the conditions -/
theorem average_temperature (t w th : ℝ) : 
  (t + w + th + 33) / 4 = 46 →
  (41 + t + w + th) / 4 = 48 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_l2189_218961


namespace NUMINAMATH_CALUDE_angle_of_inclination_range_l2189_218965

theorem angle_of_inclination_range (θ : Real) (x y : Real) :
  x - y * Real.sin θ + 1 = 0 →
  ∃ α, α ∈ Set.Icc (π/4) (3*π/4) ∧
       (α = π/2 ∨ Real.tan α = 1 / Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_range_l2189_218965


namespace NUMINAMATH_CALUDE_last_three_digits_of_11_pow_210_l2189_218926

theorem last_three_digits_of_11_pow_210 : 11^210 ≡ 601 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_11_pow_210_l2189_218926


namespace NUMINAMATH_CALUDE_calculate_individual_tip_l2189_218967

/-- Calculates the individual tip amount for a group dining out -/
theorem calculate_individual_tip (julie_order : ℚ) (letitia_order : ℚ) (anton_order : ℚ) 
  (tip_rate : ℚ) (h1 : julie_order = 10) (h2 : letitia_order = 20) (h3 : anton_order = 30) 
  (h4 : tip_rate = 0.2) : 
  (julie_order + letitia_order + anton_order) * tip_rate / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_individual_tip_l2189_218967


namespace NUMINAMATH_CALUDE_last_two_digits_of_17_to_17_l2189_218915

theorem last_two_digits_of_17_to_17 : 17^17 ≡ 77 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_17_to_17_l2189_218915


namespace NUMINAMATH_CALUDE_pen_pencil_cost_total_cost_is_13_l2189_218974

/-- The total cost of a pen and a pencil, where the pen costs $9 more than the pencil and the pencil costs $2. -/
theorem pen_pencil_cost : ℕ → ℕ → ℕ
  | pencil_cost, pen_extra_cost =>
    let pen_cost := pencil_cost + pen_extra_cost
    pencil_cost + pen_cost

/-- Proof that the total cost of a pen and a pencil is $13, given the conditions. -/
theorem total_cost_is_13 : pen_pencil_cost 2 9 = 13 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_total_cost_is_13_l2189_218974


namespace NUMINAMATH_CALUDE_amy_total_score_l2189_218973

/-- Calculates the total score for Amy's video game performance --/
def total_score (treasure_points enemy_points : ℕ)
                (level1_treasures level1_enemies : ℕ)
                (level2_enemies : ℕ) : ℕ :=
  let level1_score := treasure_points * level1_treasures + enemy_points * level1_enemies
  let level2_score := enemy_points * level2_enemies * 2
  level1_score + level2_score

/-- Theorem stating that Amy's total score is 154 points --/
theorem amy_total_score :
  total_score 4 10 6 3 5 = 154 :=
by sorry

end NUMINAMATH_CALUDE_amy_total_score_l2189_218973


namespace NUMINAMATH_CALUDE_existence_of_property_P_one_third_l2189_218972

-- Define the property P(m) for a function f on an interval
def has_property_P (f : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  ∃ x₀ ∈ D, f x₀ = f (x₀ + m) ∧ x₀ + m ∈ D

-- Theorem statement
theorem existence_of_property_P_one_third
  (f : ℝ → ℝ) (h_cont : Continuous f) (h_eq : f 0 = f 2) :
  ∃ x₀ ∈ Set.Icc 0 (5/3), f x₀ = f (x₀ + 1/3) :=
by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_existence_of_property_P_one_third_l2189_218972


namespace NUMINAMATH_CALUDE_problem_solution_l2189_218937

def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}
def B : Set ℝ := {-1, 1}

theorem problem_solution (a b : ℝ) :
  (B ⊆ A a b → a = -1) ∧
  (A a b ∩ B ≠ ∅ → a^2 - b^2 + 2*a = -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2189_218937


namespace NUMINAMATH_CALUDE_sum_of_numerator_and_denominator_l2189_218983

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.4̅5̅ -/
def decimal : ℚ := RepeatingDecimal 4 5

/-- The fraction representation of 0.4̅5̅ in lowest terms -/
def fraction : ℚ := 5 / 11

theorem sum_of_numerator_and_denominator : 
  decimal = fraction ∧ fraction.num + fraction.den = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_numerator_and_denominator_l2189_218983


namespace NUMINAMATH_CALUDE_division_problem_l2189_218957

theorem division_problem : ∃ (n : ℕ), n = 12401 ∧ n / 163 = 76 ∧ n % 163 = 13 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2189_218957


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l2189_218988

/-- The maximum value of x+y given x^2 + y^2 = 100 and xy = 36 is 2√43 -/
theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) (h2 : x * y = 36) : 
  x + y ≤ 2 * Real.sqrt 43 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l2189_218988


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_diff_sum_of_solutions_eq_eight_l2189_218982

theorem sum_of_solutions_squared_diff (a c : ℝ) :
  (∀ x : ℝ, (x - a)^2 = c) → 
  (∃ x₁ x₂ : ℝ, (x₁ - a)^2 = c ∧ (x₂ - a)^2 = c ∧ x₁ + x₂ = 2 * a) :=
by sorry

-- The specific problem
theorem sum_of_solutions_eq_eight :
  (∃ x₁ x₂ : ℝ, (x₁ - 4)^2 = 49 ∧ (x₂ - 4)^2 = 49 ∧ x₁ + x₂ = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_diff_sum_of_solutions_eq_eight_l2189_218982


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l2189_218939

/-- Given a cylinder with a rectangular front view of area 6,
    prove that its lateral area is 6π. -/
theorem cylinder_lateral_area (h : ℝ) (h_pos : h > 0) : 
  let d := 6 / h
  let lateral_area := π * d * h
  lateral_area = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l2189_218939


namespace NUMINAMATH_CALUDE_investment_value_l2189_218956

/-- Proves that the value of the larger investment is $1500 given the specified conditions. -/
theorem investment_value (x : ℝ) : 
  (0.07 * 500 + 0.27 * x = 0.22 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_l2189_218956


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l2189_218916

theorem sine_cosine_relation (θ : ℝ) (h : Real.cos (3 * Real.pi / 14 - θ) = 1 / 3) :
  Real.sin (2 * Real.pi / 7 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l2189_218916


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2189_218924

/-- The number of distinct arrangements of beads on a necklace with specific properties. -/
def necklaceArrangements (n : ℕ) : ℕ :=
  Nat.factorial n / 2

/-- Theorem stating that the number of distinct arrangements of 8 beads
    on a necklace with a fixed pendant and reflectional symmetry is 8! / 2. -/
theorem eight_bead_necklace_arrangements :
  necklaceArrangements 8 = 20160 := by
  sorry

#eval necklaceArrangements 8

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2189_218924


namespace NUMINAMATH_CALUDE_total_participants_is_280_l2189_218963

/-- The number of students who participated in at least one competition -/
def total_participants (math physics chem math_physics math_chem phys_chem all_three : ℕ) : ℕ :=
  math + physics + chem - math_physics - math_chem - phys_chem + all_three

/-- Theorem stating that the total number of participants is 280 given the conditions -/
theorem total_participants_is_280 :
  total_participants 203 179 165 143 116 97 89 = 280 := by
  sorry

#eval total_participants 203 179 165 143 116 97 89

end NUMINAMATH_CALUDE_total_participants_is_280_l2189_218963


namespace NUMINAMATH_CALUDE_max_sequence_length_is_17_l2189_218934

/-- The maximum length of a sequence satisfying the given conditions -/
def max_sequence_length : ℕ := 17

/-- A sequence of integers from 1 to 4 -/
def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ i, i ≤ k → 1 ≤ a i ∧ a i ≤ 4

/-- The uniqueness condition for consecutive pairs in the sequence -/
def unique_pairs (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ i j, i < k → j < k → a i = a j → a (i + 1) = a (j + 1) → i = j

/-- The main theorem stating that 17 is the maximum length of a valid sequence with unique pairs -/
theorem max_sequence_length_is_17 :
  ∀ k : ℕ, (∃ a : ℕ → ℕ, valid_sequence a k ∧ unique_pairs a k) →
  k ≤ max_sequence_length :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_is_17_l2189_218934


namespace NUMINAMATH_CALUDE_train_average_speed_l2189_218927

/-- Proves that the average speed of a train is 22.5 kmph, given specific travel conditions. -/
theorem train_average_speed 
  (x : ℝ) 
  (h₁ : x > 0)  -- Ensuring x is positive for meaningful distance
  (speed₁ : ℝ) (speed₂ : ℝ)
  (h₂ : speed₁ = 30) -- First speed in kmph
  (h₃ : speed₂ = 20) -- Second speed in kmph
  (distance₁ : ℝ) (distance₂ : ℝ)
  (h₄ : distance₁ = x) -- First distance
  (h₅ : distance₂ = 2 * x) -- Second distance
  (total_distance : ℝ)
  (h₆ : total_distance = distance₁ + distance₂) -- Total distance
  : 
  (total_distance / ((distance₁ / speed₁) + (distance₂ / speed₂))) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l2189_218927


namespace NUMINAMATH_CALUDE_mersenne_divisibility_l2189_218990

theorem mersenne_divisibility (n : ℕ+) :
  (∃ m : ℕ+, (2^n.val - 1) ∣ (m.val^2 + 81)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end NUMINAMATH_CALUDE_mersenne_divisibility_l2189_218990


namespace NUMINAMATH_CALUDE_curtis_farm_egg_laying_hens_l2189_218995

/-- The number of egg-laying hens on Mr. Curtis's farm -/
def egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) : ℕ :=
  total_chickens - roosters - non_laying_hens

/-- Theorem stating the number of egg-laying hens on Mr. Curtis's farm -/
theorem curtis_farm_egg_laying_hens :
  egg_laying_hens 325 28 20 = 277 := by
  sorry

end NUMINAMATH_CALUDE_curtis_farm_egg_laying_hens_l2189_218995


namespace NUMINAMATH_CALUDE_larry_cards_larry_cards_proof_l2189_218948

/-- If Larry initially has 67 cards and Dennis takes 9 cards away, 
    then Larry will have 58 cards remaining. -/
theorem larry_cards : ℕ → ℕ → ℕ → Prop :=
  fun initial_cards cards_taken remaining_cards =>
    initial_cards = 67 ∧ 
    cards_taken = 9 ∧ 
    remaining_cards = initial_cards - cards_taken →
    remaining_cards = 58

-- The proof would go here
theorem larry_cards_proof : larry_cards 67 9 58 := by
  sorry

end NUMINAMATH_CALUDE_larry_cards_larry_cards_proof_l2189_218948


namespace NUMINAMATH_CALUDE_root_product_one_l2189_218960

theorem root_product_one (b c : ℝ) (hb : b > 0) (hc : c > 0) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^2 + 2*b*x₁ + c = 0) ∧ 
    (x₂^2 + 2*b*x₂ + c = 0) ∧ 
    (x₃^2 + 2*c*x₃ + b = 0) ∧ 
    (x₄^2 + 2*c*x₄ + b = 0) ∧ 
    (x₁ * x₂ * x₃ * x₄ = 1)) → 
  b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_product_one_l2189_218960


namespace NUMINAMATH_CALUDE_range_of_f_l2189_218935

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc (-1 : ℝ) 2 = Set.Icc 3 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2189_218935


namespace NUMINAMATH_CALUDE_division_of_fractions_l2189_218944

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l2189_218944


namespace NUMINAMATH_CALUDE_largest_valid_factor_of_130000_l2189_218955

/-- A function that checks if a natural number contains the digit 0 or 5 --/
def containsZeroOrFive (n : ℕ) : Prop := sorry

/-- The largest factor of 130000 that does not contain the digit 0 or 5 --/
def largestValidFactor : ℕ := 26

theorem largest_valid_factor_of_130000 :
  (largestValidFactor ∣ 130000) ∧ 
  ¬containsZeroOrFive largestValidFactor ∧
  ∀ k : ℕ, k > largestValidFactor → (k ∣ 130000) → containsZeroOrFive k := by sorry

end NUMINAMATH_CALUDE_largest_valid_factor_of_130000_l2189_218955


namespace NUMINAMATH_CALUDE_lives_per_player_l2189_218999

/-- Given 8 friends playing a video game with a total of 64 lives,
    prove that each friend has 8 lives. -/
theorem lives_per_player (num_friends : ℕ) (total_lives : ℕ) :
  num_friends = 8 →
  total_lives = 64 →
  total_lives / num_friends = 8 :=
by sorry

end NUMINAMATH_CALUDE_lives_per_player_l2189_218999


namespace NUMINAMATH_CALUDE_grapes_pineapple_cost_l2189_218910

/-- Represents the cost of fruit items --/
structure FruitCosts where
  oranges : ℝ
  grapes : ℝ
  pineapple : ℝ
  strawberries : ℝ

/-- The total cost of all fruits is $24 --/
def total_cost (fc : FruitCosts) : Prop :=
  fc.oranges + fc.grapes + fc.pineapple + fc.strawberries = 24

/-- The box of strawberries costs twice as much as the bag of oranges --/
def strawberry_orange_relation (fc : FruitCosts) : Prop :=
  fc.strawberries = 2 * fc.oranges

/-- The price of pineapple equals the price of oranges minus the price of grapes --/
def pineapple_relation (fc : FruitCosts) : Prop :=
  fc.pineapple = fc.oranges - fc.grapes

/-- The main theorem: Given the conditions, the cost of grapes and pineapple together is $6 --/
theorem grapes_pineapple_cost (fc : FruitCosts) 
  (h1 : total_cost fc) 
  (h2 : strawberry_orange_relation fc) 
  (h3 : pineapple_relation fc) : 
  fc.grapes + fc.pineapple = 6 := by
  sorry

end NUMINAMATH_CALUDE_grapes_pineapple_cost_l2189_218910


namespace NUMINAMATH_CALUDE_arithmetic_has_three_term_correlation_geometric_has_three_term_correlation_l2189_218918

def has_three_term_correlation (a : ℕ → ℝ) : Prop :=
  ∃ A B : ℝ, A * B ≠ 0 ∧ ∀ n : ℕ, a (n + 2) = A * a (n + 1) + B * a n

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem arithmetic_has_three_term_correlation :
  ∀ a : ℕ → ℝ, arithmetic_sequence a → has_three_term_correlation a :=
sorry

theorem geometric_has_three_term_correlation :
  ∀ a : ℕ → ℝ, geometric_sequence a → has_three_term_correlation a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_has_three_term_correlation_geometric_has_three_term_correlation_l2189_218918


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l2189_218994

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem preimage_of_3_1 : f⁻¹' {(3, 1)} = {(2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l2189_218994


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2189_218968

theorem sum_of_coefficients (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) →
  a + b + c + d + e + f + g + h + j + k = 92 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2189_218968


namespace NUMINAMATH_CALUDE_zeros_of_f_range_of_b_l2189_218997

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

-- Part 1
theorem zeros_of_f (b : ℝ) :
  f b 0 = f b 4 → (∃ x : ℝ, f b x = 0) ∧ (∀ x : ℝ, f b x = 0 → x = 3 ∨ x = 1) :=
sorry

-- Part 2
theorem range_of_b :
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ f b x = 0 ∧ f b y = 0) → b > 4 :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_range_of_b_l2189_218997


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2189_218980

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → interior_angle = 144 → (n - 2) * 180 = n * interior_angle → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2189_218980


namespace NUMINAMATH_CALUDE_lunch_cost_calculation_l2189_218945

/-- Calculates the total cost of lunches for a field trip --/
theorem lunch_cost_calculation (total_lunches : ℕ) 
  (vegetarian_lunches : ℕ) (gluten_free_lunches : ℕ) (both_veg_gf : ℕ)
  (regular_cost : ℕ) (special_cost : ℕ) (both_cost : ℕ) : 
  total_lunches = 44 ∧ 
  vegetarian_lunches = 10 ∧ 
  gluten_free_lunches = 5 ∧ 
  both_veg_gf = 2 ∧
  regular_cost = 7 ∧ 
  special_cost = 8 ∧ 
  both_cost = 9 → 
  (both_veg_gf * both_cost + 
   (vegetarian_lunches - both_veg_gf) * special_cost + 
   (gluten_free_lunches - both_veg_gf) * special_cost + 
   (total_lunches - vegetarian_lunches - gluten_free_lunches + both_veg_gf) * regular_cost) = 323 := by
  sorry


end NUMINAMATH_CALUDE_lunch_cost_calculation_l2189_218945


namespace NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l2189_218949

theorem regular_polygon_with_108_degree_interior_angles (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 / n = 108) → 
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l2189_218949


namespace NUMINAMATH_CALUDE_like_terms_proof_l2189_218993

/-- Given that -3x^(m-1)y^3 and 4xy^(m+n) are like terms, prove that m = 2 and n = 1 -/
theorem like_terms_proof (m n : ℤ) : 
  (∀ x y : ℝ, ∃ k : ℝ, -3 * x^(m-1) * y^3 = k * 4 * x * y^(m+n)) → 
  m = 2 ∧ n = 1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_proof_l2189_218993


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2189_218959

def U : Finset ℤ := {0, -1, -2, -3, -4}
def M : Finset ℤ := {0, -1, -2}

theorem complement_of_M_in_U : U \ M = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2189_218959


namespace NUMINAMATH_CALUDE_platform_length_calculation_platform_length_proof_l2189_218942

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  platform_length

/-- Proves that the platform length is approximately 190.08 m given the specified conditions --/
theorem platform_length_proof 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 90) 
  (h2 : train_speed_kmph = 56) 
  (h3 : crossing_time = 18) : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (platform_length_calculation train_length train_speed_kmph crossing_time - 190.08) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_platform_length_proof_l2189_218942


namespace NUMINAMATH_CALUDE_largest_k_inequality_l2189_218970

theorem largest_k_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ k : ℝ, k > 174960 → ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) < k * a * b * c * d^3) ∧
  (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 →
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) ≥ 174960 * a * b * c * d^3) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l2189_218970


namespace NUMINAMATH_CALUDE_fred_basketball_games_l2189_218938

def games_this_year : ℕ := 36
def total_games : ℕ := 47

def games_last_year : ℕ := total_games - games_this_year

theorem fred_basketball_games : games_last_year = 11 := by
  sorry

end NUMINAMATH_CALUDE_fred_basketball_games_l2189_218938


namespace NUMINAMATH_CALUDE_square_difference_equality_l2189_218929

theorem square_difference_equality : 535^2 - 465^2 = 70000 := by sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2189_218929
