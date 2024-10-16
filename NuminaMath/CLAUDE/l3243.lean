import Mathlib

namespace NUMINAMATH_CALUDE_equal_quotient_remainder_divisible_by_seven_l3243_324380

theorem equal_quotient_remainder_divisible_by_seven :
  {n : ℕ | ∃ (q : ℕ), n = 7 * q + q ∧ q < 7} = {8, 16, 24, 32, 40, 48} := by
  sorry

end NUMINAMATH_CALUDE_equal_quotient_remainder_divisible_by_seven_l3243_324380


namespace NUMINAMATH_CALUDE_extra_fruits_count_l3243_324363

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 75

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 35

/-- The number of oranges ordered by the cafeteria -/
def oranges : ℕ := 40

/-- The number of bananas ordered by the cafeteria -/
def bananas : ℕ := 20

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 17

/-- The total number of fruits ordered by the cafeteria -/
def total_fruits : ℕ := red_apples + green_apples + oranges + bananas

/-- The number of extra fruits the cafeteria ended up with -/
def extra_fruits : ℕ := total_fruits - students_wanting_fruit

theorem extra_fruits_count : extra_fruits = 153 := by
  sorry

end NUMINAMATH_CALUDE_extra_fruits_count_l3243_324363


namespace NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l3243_324399

theorem right_triangle_and_multiplicative_inverse :
  (35^2 + 312^2 = 313^2) ∧ 
  (520 * 2026 % 4231 = 1) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l3243_324399


namespace NUMINAMATH_CALUDE_boys_in_class_l3243_324353

/-- Given a class with 10 girls, prove that if there are 780 ways to select 1 girl and 2 boys
    when choosing 3 students at random, then the number of boys in the class is 13. -/
theorem boys_in_class (num_girls : ℕ) (num_ways : ℕ) : 
  num_girls = 10 →
  num_ways = 780 →
  (∃ num_boys : ℕ, 
    num_ways = (num_girls.choose 1) * (num_boys.choose 2) ∧
    num_boys = 13) :=
by sorry

end NUMINAMATH_CALUDE_boys_in_class_l3243_324353


namespace NUMINAMATH_CALUDE_shaded_area_between_triangles_l3243_324355

/-- The area of the shaded region between two back-to-back isosceles triangles -/
theorem shaded_area_between_triangles (b h x₀ : ℝ) :
  b > 0 → h > 0 →
  let x₁ := x₀ - b / 2
  let x₂ := x₀ + b / 2
  let y := h
  (x₂ - x₁) * y = 280 :=
by
  sorry

#check shaded_area_between_triangles 12 10 10

end NUMINAMATH_CALUDE_shaded_area_between_triangles_l3243_324355


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3243_324379

theorem triangle_inequality_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a^2 + 2*b*c)/(b^2 + c^2) + (b^2 + 2*a*c)/(c^2 + a^2) + (c^2 + 2*a*b)/(a^2 + b^2) > 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3243_324379


namespace NUMINAMATH_CALUDE_tetrahedron_stripe_probability_l3243_324337

/-- Represents the orientation of a stripe on a face of a tetrahedron -/
inductive StripeOrientation
  | First
  | Second
  | Third

/-- Represents the configuration of stripes on a tetrahedron -/
def TetrahedronStripes := Fin 4 → StripeOrientation

/-- Predicate to check if a given configuration of stripes forms a continuous stripe around the tetrahedron -/
def isContinuousStripe (config : TetrahedronStripes) : Prop := sorry

/-- The total number of possible stripe configurations -/
def totalConfigurations : ℕ := 3^4

/-- The number of configurations that form a continuous stripe -/
def favorableConfigurations : ℕ := 18

/-- Theorem stating the probability of a continuous stripe encircling the tetrahedron -/
theorem tetrahedron_stripe_probability :
  (favorableConfigurations : ℚ) / totalConfigurations = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_stripe_probability_l3243_324337


namespace NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l3243_324321

theorem square_difference_fifty_fortynine : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l3243_324321


namespace NUMINAMATH_CALUDE_smaller_number_l3243_324394

theorem smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0) 
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : 
  min u v = 6 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_l3243_324394


namespace NUMINAMATH_CALUDE_intersecting_circles_chord_length_l3243_324314

/-- Given two circles with radii 10 and 7, whose centers are 15 units apart,
    and a point P where the circles intersect, if a line is drawn through P
    such that QP = PR, then QP^2 = 10800/35. -/
theorem intersecting_circles_chord_length 
  (O₁ O₂ P Q R : ℝ × ℝ) -- Points in 2D plane
  (h₁ : dist O₁ O₂ = 15) -- Centers are 15 units apart
  (h₂ : dist O₁ P = 10) -- Radius of first circle
  (h₃ : dist O₂ P = 7)  -- Radius of second circle
  (h₄ : dist Q P = dist P R) -- QP = PR
  : (dist Q P)^2 = 10800/35 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_chord_length_l3243_324314


namespace NUMINAMATH_CALUDE_parabola_intersecting_line_slope_l3243_324344

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point M
def point_M : ℝ × ℝ := (-1, 1)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through a point with a given slope
def line_through_point (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = k * (x - p.1)

-- Define the condition for a right angle
def is_right_angle (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

-- Main theorem
theorem parabola_intersecting_line_slope :
  ∀ (k : ℝ) (a b : ℝ × ℝ),
    (∀ x y, parabola x y ↔ (x, y) = a ∨ (x, y) = b) →
    (∀ x y, line_through_point focus k x y ↔ (x, y) = a ∨ (x, y) = b) →
    is_right_angle point_M a b →
    k = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersecting_line_slope_l3243_324344


namespace NUMINAMATH_CALUDE_min_value_expression_l3243_324369

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 3) :
  a^2 + 4*a*b + 8*b^2 + 10*b*c + 3*c^2 ≥ 27 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 3 ∧
    a₀^2 + 4*a₀*b₀ + 8*b₀^2 + 10*b₀*c₀ + 3*c₀^2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3243_324369


namespace NUMINAMATH_CALUDE_lcm_of_180_and_504_l3243_324366

theorem lcm_of_180_and_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_180_and_504_l3243_324366


namespace NUMINAMATH_CALUDE_square_difference_of_sums_l3243_324306

theorem square_difference_of_sums (a b : ℝ) :
  a = Real.sqrt 3 + Real.sqrt 2 →
  b = Real.sqrt 3 - Real.sqrt 2 →
  a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sums_l3243_324306


namespace NUMINAMATH_CALUDE_cubes_form_name_l3243_324317

/-- Represents a cube with letters on its faces -/
structure Cube where
  faces : Fin 6 → Char

/-- Represents the visible face of a cube -/
inductive VisibleFace
  | front
  | right

/-- Returns the letter on the visible face of a cube -/
def visibleLetter (c : Cube) (f : VisibleFace) : Char :=
  match f with
  | VisibleFace.front => c.faces 0
  | VisibleFace.right => c.faces 1

/-- Represents the arrangement of four cubes -/
structure CubeArrangement where
  cubes : Fin 4 → Cube
  visibleFaces : Fin 4 → VisibleFace

/-- The name formed by the visible letters in the cube arrangement -/
def formName (arr : CubeArrangement) : String :=
  String.mk (List.ofFn fun i => visibleLetter (arr.cubes i) (arr.visibleFaces i))

/-- The theorem stating that the given cube arrangement forms the name "Ника" -/
theorem cubes_form_name (arr : CubeArrangement) 
  (h1 : visibleLetter (arr.cubes 0) (arr.visibleFaces 0) = 'Н')
  (h2 : visibleLetter (arr.cubes 1) (arr.visibleFaces 1) = 'И')
  (h3 : visibleLetter (arr.cubes 2) (arr.visibleFaces 2) = 'К')
  (h4 : visibleLetter (arr.cubes 3) (arr.visibleFaces 3) = 'А') :
  formName arr = "Ника" := by
  sorry


end NUMINAMATH_CALUDE_cubes_form_name_l3243_324317


namespace NUMINAMATH_CALUDE_function_value_theorem_l3243_324398

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2 where f(-5) = m, prove that f(5) = -m + 4 -/
theorem function_value_theorem (a b c m : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^5 + c * x^3 + 2
  f (-5) = m → f 5 = -m + 4 := by
sorry

end NUMINAMATH_CALUDE_function_value_theorem_l3243_324398


namespace NUMINAMATH_CALUDE_product_expansion_l3243_324319

theorem product_expansion (y : ℝ) : 4 * (y - 3) * (y + 2) = 4 * y^2 - 4 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3243_324319


namespace NUMINAMATH_CALUDE_cats_remaining_l3243_324305

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 15 → house = 49 → sold = 19 → siamese + house - sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l3243_324305


namespace NUMINAMATH_CALUDE_room_width_is_15_l3243_324373

/-- Represents the dimensions and features of a room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  door_area : ℝ
  window_area : ℝ
  window_count : ℕ
  whitewash_cost_per_sqft : ℝ

/-- Calculates the total cost of whitewashing the room -/
def whitewash_cost (r : Room) : ℝ :=
  let wall_area := 2 * (r.length * r.height + r.width * r.height)
  let paintable_area := wall_area - r.door_area - r.window_count * r.window_area
  paintable_area * r.whitewash_cost_per_sqft

/-- Theorem stating that given the room specifications, the width is 15 feet -/
theorem room_width_is_15 (r : Room) 
    (h1 : r.length = 25)
    (h2 : r.height = 12)
    (h3 : r.door_area = 18)
    (h4 : r.window_area = 12)
    (h5 : r.window_count = 3)
    (h6 : r.whitewash_cost_per_sqft = 2)
    (h7 : whitewash_cost r = 1812) :
    r.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_15_l3243_324373


namespace NUMINAMATH_CALUDE_topsoil_cost_calculation_l3243_324333

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The amount of topsoil in cubic yards -/
def topsoil_amount : ℝ := 8

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := topsoil_amount * cubic_yards_to_cubic_feet * topsoil_cost_per_cubic_foot

theorem topsoil_cost_calculation : total_cost = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_calculation_l3243_324333


namespace NUMINAMATH_CALUDE_men_count_in_alternating_arrangement_l3243_324396

/-- Represents the number of arrangements for a given number of men and women -/
def arrangements (men : ℕ) (women : ℕ) : ℕ := sorry

/-- Represents whether men and women are alternating in an arrangement -/
def isAlternating (men : ℕ) (women : ℕ) : Prop := sorry

theorem men_count_in_alternating_arrangement :
  ∀ (men : ℕ),
  (women : ℕ) → women = 2 →
  isAlternating men women →
  arrangements men women = 12 →
  men = 4 := by sorry

end NUMINAMATH_CALUDE_men_count_in_alternating_arrangement_l3243_324396


namespace NUMINAMATH_CALUDE_sum_f_positive_l3243_324342

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l3243_324342


namespace NUMINAMATH_CALUDE_min_transport_cost_l3243_324371

/-- Represents the number of machines at each location --/
structure LocationMachines where
  a : ℕ
  b : ℕ

/-- Represents the number of machines needed in each area --/
structure AreaNeeds where
  a : ℕ
  b : ℕ

/-- Represents the transportation costs between locations and areas --/
structure TransportCosts where
  a_to_a : ℕ
  a_to_b : ℕ
  b_to_a : ℕ
  b_to_b : ℕ

/-- Calculates the total transportation cost --/
def totalCost (x : ℕ) (lm : LocationMachines) (an : AreaNeeds) (tc : TransportCosts) : ℕ :=
  tc.b_to_a * x + tc.b_to_b * (lm.b - x) + tc.a_to_a * (an.a - x) + tc.a_to_b * (lm.a - (an.a - x))

/-- The main theorem stating the minimum transportation cost --/
theorem min_transport_cost (lm : LocationMachines) (an : AreaNeeds) (tc : TransportCosts) :
  lm.a = 12 ∧ lm.b = 6 ∧ an.a = 10 ∧ an.b = 8 ∧
  tc.a_to_a = 400 ∧ tc.a_to_b = 800 ∧ tc.b_to_a = 300 ∧ tc.b_to_b = 500 →
  (∀ x : ℕ, x ≤ 6 → totalCost x lm an tc ≥ 8600) ∧
  (∃ x : ℕ, x ≤ 6 ∧ totalCost x lm an tc = 8600) :=
sorry

end NUMINAMATH_CALUDE_min_transport_cost_l3243_324371


namespace NUMINAMATH_CALUDE_factors_of_8_to_15_l3243_324348

/-- The number of positive factors of 8^15 is 46 -/
theorem factors_of_8_to_15 : Nat.card (Nat.divisors (8^15)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_8_to_15_l3243_324348


namespace NUMINAMATH_CALUDE_power_plus_one_prime_l3243_324358

theorem power_plus_one_prime (a n : ℕ) (ha : a > 1) (hprime : Nat.Prime (a^n + 1)) :
  Even a ∧ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_plus_one_prime_l3243_324358


namespace NUMINAMATH_CALUDE_jessica_seashells_l3243_324378

/-- The number of seashells Jessica has left after giving some away -/
def seashells_left (found : ℝ) (given_away : ℝ) : ℝ :=
  found - given_away

/-- Theorem: Jessica is left with 2.25 seashells -/
theorem jessica_seashells :
  seashells_left 8.5 6.25 = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l3243_324378


namespace NUMINAMATH_CALUDE_smallest_special_number_l3243_324392

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_special_number :
  ∀ n : ℕ,
    is_two_digit n →
    n % 6 = 0 →
    n % 3 = 0 →
    is_perfect_square (digit_product n) →
    30 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l3243_324392


namespace NUMINAMATH_CALUDE_complex_product_equality_l3243_324356

theorem complex_product_equality (x : ℂ) (h : x = Complex.exp (Complex.I * π / 9)) : 
  (2*x + x^3) * (2*x^3 + x^9) * (2*x^6 + x^18) * (2*x^9 + x^27) * (2*x^12 + x^36) * (2*x^15 + x^45) = 549 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equality_l3243_324356


namespace NUMINAMATH_CALUDE_book_sale_result_l3243_324303

/-- Represents the book sale scenario -/
structure BookSale where
  initial_fiction : ℕ
  initial_nonfiction : ℕ
  fiction_sold : ℕ
  fiction_remaining : ℕ
  total_earnings : ℕ
  fiction_price : ℕ
  nonfiction_price : ℕ

/-- Theorem stating the results of the book sale -/
theorem book_sale_result (sale : BookSale)
  (h1 : sale.fiction_sold = 137)
  (h2 : sale.fiction_remaining = 105)
  (h3 : sale.total_earnings = 685)
  (h4 : sale.fiction_price = 3)
  (h5 : sale.nonfiction_price = 5)
  (h6 : sale.initial_fiction = sale.fiction_sold + sale.fiction_remaining) :
  sale.initial_fiction = 242 ∧
  (sale.total_earnings - sale.fiction_sold * sale.fiction_price) / sale.nonfiction_price = 54 := by
  sorry


end NUMINAMATH_CALUDE_book_sale_result_l3243_324303


namespace NUMINAMATH_CALUDE_integer_set_equivalence_l3243_324364

theorem integer_set_equivalence (a : ℝ) : 
  (a ≤ 1 ∧ (Set.range (fun n : ℤ => (n : ℝ)) ∩ Set.Icc a (2 - a)).ncard = 3) ↔ 
  -1 < a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_integer_set_equivalence_l3243_324364


namespace NUMINAMATH_CALUDE_existence_of_midpoint_with_odd_double_coordinates_l3243_324387

/-- A point in the xy-plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A sequence of 1993 distinct points with the required properties -/
def PointSequence : Type :=
  { ps : Fin 1993 → IntPoint //
    (∀ i j, i ≠ j → ps i ≠ ps j) ∧  -- points are distinct
    (∀ i : Fin 1992, ∀ p : IntPoint,
      p ≠ ps i ∧ p ≠ ps (i + 1) →
      ¬∃ (t : ℚ), 0 < t ∧ t < 1 ∧
        p.x = (1 - t) * (ps i).x + t * (ps (i + 1)).x ∧
        p.y = (1 - t) * (ps i).y + t * (ps (i + 1)).y) }

theorem existence_of_midpoint_with_odd_double_coordinates (ps : PointSequence) :
    ∃ i : Fin 1992, ∃ qx qy : ℚ,
      (2 * qx).num % 2 = 1 ∧
      (2 * qy).num % 2 = 1 ∧
      qx = ((ps.val i).x + (ps.val (i + 1)).x) / 2 ∧
      qy = ((ps.val i).y + (ps.val (i + 1)).y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_midpoint_with_odd_double_coordinates_l3243_324387


namespace NUMINAMATH_CALUDE_circle_and_tangents_l3243_324377

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line tangent to the circle
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 4 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 3)

-- Define the two possible tangent lines through P
def tangent_line1 (x : ℝ) : Prop := x = 2
def tangent_line2 (x y : ℝ) : Prop := 5 * x - 12 * y + 26 = 0

theorem circle_and_tangents :
  -- The circle is tangent to the given line
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line x y) ∧
  -- The circle passes through only one point of the line
  (∀ (x y : ℝ), circle_equation x y → tangent_line x y → 
    ∀ (x' y' : ℝ), x' ≠ x ∨ y' ≠ y → circle_equation x' y' → ¬tangent_line x' y') ∧
  -- The two tangent lines pass through P and are tangent to the circle
  (tangent_line1 point_P.1 ∨ tangent_line2 point_P.1 point_P.2) ∧
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line1 x) ∧
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line2 x y) ∧
  -- There are no other tangent lines through P
  (∀ (f : ℝ → ℝ), f point_P.1 = point_P.2 → 
    (∃ (x y : ℝ), circle_equation x y ∧ y = f x) →
    (∀ x, f x = point_P.2 + (x - point_P.1) * 5 / 12 ∨ f x = point_P.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l3243_324377


namespace NUMINAMATH_CALUDE_length_MN_circle_P_equation_l3243_324313

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the intersection points M and N
def intersection_points (M N : ℝ × ℝ) : Prop :=
  line_l M.1 M.2 ∧ circle_C M.1 M.2 ∧
  line_l N.1 N.2 ∧ circle_C N.1 N.2 ∧
  M ≠ N

-- Theorem for the length of MN
theorem length_MN (M N : ℝ × ℝ) (h : intersection_points M N) :
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 :=
sorry

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem for the equation of circle P
theorem circle_P_equation (M N : ℝ × ℝ) (h : intersection_points M N) :
  ∀ x y : ℝ, circle_P x y ↔ 
    ((x - (M.1 + N.1) / 2)^2 + (y - (M.2 + N.2) / 2)^2 = 
     ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_length_MN_circle_P_equation_l3243_324313


namespace NUMINAMATH_CALUDE_smallest_n_value_l3243_324323

/-- Represents a rectangular block made of 1-cm cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in the block -/
def Block.totalCubes (b : Block) : ℕ := b.length * b.width * b.height

/-- Calculates the number of invisible cubes when three faces are visible -/
def Block.invisibleCubes (b : Block) : ℕ := (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- Theorem stating the smallest possible value of N -/
theorem smallest_n_value (b : Block) (h : b.invisibleCubes = 300) :
  ∃ (min_b : Block), min_b.invisibleCubes = 300 ∧
    min_b.totalCubes ≤ b.totalCubes ∧
    min_b.totalCubes = 468 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3243_324323


namespace NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l3243_324393

theorem impossibility_of_simultaneous_inequalities (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1/4 ∧ b * (1 - c) > 1/4 ∧ c * (1 - a) > 1/4) := by
sorry

end NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l3243_324393


namespace NUMINAMATH_CALUDE_square_area_ratio_l3243_324307

theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (3 * y)^2 / (9 * y)^2 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3243_324307


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l3243_324383

/-- Parabola tangent intersection theorem -/
theorem parabola_tangent_intersection
  (t₁ t₂ : ℝ) (h : t₁ ≠ t₂) :
  let parabola := fun x : ℝ => x^2 / 4
  let tangent₁ := fun x : ℝ => t₁ * x - t₁^2
  let tangent₂ := fun x : ℝ => t₂ * x - t₂^2
  let intersection_x := t₁ + t₂
  let intersection_y := t₁ * t₂
  (parabola (2 * t₁) = t₁^2) ∧
  (parabola (2 * t₂) = t₂^2) ∧
  (tangent₁ intersection_x = intersection_y) ∧
  (tangent₂ intersection_x = intersection_y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l3243_324383


namespace NUMINAMATH_CALUDE_triangle_properties_l3243_324386

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 2 * t.b * Real.cos t.A ∧
  t.b + t.c = Real.sqrt 10 ∧
  t.a = 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 3 ∧ (1 / 2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3243_324386


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l3243_324395

theorem salary_increase_percentage (initial_salary final_salary : ℝ) 
  (increase_percentage decrease_percentage : ℝ) :
  initial_salary = 5000 →
  final_salary = 5225 →
  decrease_percentage = 5 →
  final_salary = initial_salary * (1 + increase_percentage / 100) * (1 - decrease_percentage / 100) →
  increase_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l3243_324395


namespace NUMINAMATH_CALUDE_reflection_theorem_l3243_324310

def P : ℝ × ℝ := (1, 2)

-- Reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Reflection across origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem reflection_theorem :
  reflect_x P = (1, -2) ∧ reflect_origin P = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_theorem_l3243_324310


namespace NUMINAMATH_CALUDE_max_freshmen_is_eight_l3243_324374

/-- Represents the relation of knowing each other among freshmen. -/
def Knows (n : ℕ) := Fin n → Fin n → Prop

/-- The property that any 3 people include at least 2 who know each other. -/
def AnyThreeHaveTwoKnown (n : ℕ) (knows : Knows n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- The property that any 4 people include at least 2 who do not know each other. -/
def AnyFourHaveTwoUnknown (n : ℕ) (knows : Knows n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The theorem stating that the maximum number of freshmen satisfying the conditions is 8. -/
theorem max_freshmen_is_eight :
  ∀ n : ℕ, (∃ knows : Knows n, AnyThreeHaveTwoKnown n knows ∧ AnyFourHaveTwoUnknown n knows) →
    n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_freshmen_is_eight_l3243_324374


namespace NUMINAMATH_CALUDE_minimum_m_in_range_l3243_324343

/-- Represents a sequence of five consecutive integers -/
structure FiveConsecutiveIntegers where
  m : ℕ  -- The middle integer
  h1 : m > 2  -- Ensures all integers are positive

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- Checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k * k

/-- The main theorem -/
theorem minimum_m_in_range (seq : FiveConsecutiveIntegers) :
  (isPerfectSquare (3 * seq.m)) →
  (isPerfectCube (5 * seq.m)) →
  (∃ min_m : ℕ, 
    (∀ m : ℕ, m < min_m → ¬(isPerfectSquare (3 * m) ∧ isPerfectCube (5 * m))) ∧
    600 < min_m ∧
    min_m ≤ 800) :=
sorry

end NUMINAMATH_CALUDE_minimum_m_in_range_l3243_324343


namespace NUMINAMATH_CALUDE_x_values_l3243_324311

theorem x_values (x n : ℕ) (h1 : x = 2^n - 32) 
  (h2 : (Nat.factors x).card = 3) 
  (h3 : 3 ∈ Nat.factors x) : 
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_values_l3243_324311


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3243_324309

/-- The slope angle of the line x - y + 1 = 0 is 45 degrees -/
theorem slope_angle_of_line (x y : ℝ) : 
  x - y + 1 = 0 → Real.arctan 1 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3243_324309


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3243_324385

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144

-- Define the distance between foci
def distance_between_foci (eq : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_distance :
  distance_between_foci ellipse_equation = 2 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3243_324385


namespace NUMINAMATH_CALUDE_ratio_expression_value_l3243_324375

theorem ratio_expression_value (A B C : ℚ) (h : A/B = 3/2 ∧ B/C = 2/6) :
  (4*A - 3*B) / (5*C + 2*A) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l3243_324375


namespace NUMINAMATH_CALUDE_fence_painting_problem_l3243_324354

/-- Given a fence of 360 square feet to be painted by three people in the ratio 3:5:2,
    prove that the person with the smallest share paints 72 square feet. -/
theorem fence_painting_problem (total_area : ℝ) (ratio_a ratio_b ratio_c : ℕ) :
  total_area = 360 →
  ratio_a = 3 →
  ratio_b = 5 →
  ratio_c = 2 →
  (ratio_a + ratio_b + ratio_c : ℝ) * (total_area / (ratio_a + ratio_b + ratio_c : ℝ) * ratio_c) = 72 :=
by sorry

end NUMINAMATH_CALUDE_fence_painting_problem_l3243_324354


namespace NUMINAMATH_CALUDE_water_needed_for_punch_l3243_324382

/-- Represents the recipe ratios and calculates the required amount of water -/
def water_needed (lemon_juice : ℝ) : ℝ :=
  let sugar := 3 * lemon_juice
  let water := 3 * sugar
  water

/-- Proves that 36 cups of water are needed given the recipe ratios and 4 cups of lemon juice -/
theorem water_needed_for_punch : water_needed 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_punch_l3243_324382


namespace NUMINAMATH_CALUDE_coefficient_of_y_l3243_324368

theorem coefficient_of_y (b : ℝ) : 
  (5 * (2 : ℝ)^2 - b * 2 + 55 = 59) → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l3243_324368


namespace NUMINAMATH_CALUDE_inequality_solution_l3243_324350

theorem inequality_solution (x : ℝ) :
  x ∈ Set.Icc (-3 : ℝ) 3 ∧ 
  x ≠ -5/3 ∧ 
  (4*x^2 + 2) / (5 + 3*x) ≥ 1 ↔ 
  x ∈ Set.Icc (-3 : ℝ) (-3/4) ∪ Set.Icc 1 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3243_324350


namespace NUMINAMATH_CALUDE_minimize_quadratic_l3243_324335

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The theorem states that x = 3 minimizes the quadratic function f(x) = 3x^2 - 18x + 7 -/
theorem minimize_quadratic :
  ∃ (x_min : ℝ), x_min = 3 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
sorry

end NUMINAMATH_CALUDE_minimize_quadratic_l3243_324335


namespace NUMINAMATH_CALUDE_cube_difference_1234567_l3243_324336

theorem cube_difference_1234567 : ∃ a b : ℤ, a^3 - b^3 = 1234567 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_1234567_l3243_324336


namespace NUMINAMATH_CALUDE_space_diagonals_specific_polyhedron_l3243_324312

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of space diagonals in the specific polyhedron -/
theorem space_diagonals_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by
  sorry

end NUMINAMATH_CALUDE_space_diagonals_specific_polyhedron_l3243_324312


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l3243_324334

/-- Given a shopkeeper selling cloth with a total selling price, loss per metre, and cost price per metre,
    prove that the number of metres sold is as calculated. -/
theorem cloth_sale_calculation
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (cost_price_per_metre : ℕ)
  (h1 : total_selling_price = 36000)
  (h2 : loss_per_metre = 10)
  (h3 : cost_price_per_metre = 70) :
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 600 := by
  sorry

#check cloth_sale_calculation

end NUMINAMATH_CALUDE_cloth_sale_calculation_l3243_324334


namespace NUMINAMATH_CALUDE_tunnel_length_l3243_324331

/-- The length of a tunnel given train parameters -/
theorem tunnel_length (train_length : ℝ) (time_diff : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  time_diff = 4 →
  train_speed = 30 →
  train_length = train_speed * time_diff / 60 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_tunnel_length_l3243_324331


namespace NUMINAMATH_CALUDE_fold_cut_result_l3243_324332

/-- Represents the possible number of parts after cutting a folded square --/
inductive CutResult
  | OppositeMiddle : CutResult
  | AdjacentMiddle : CutResult

/-- Represents the dimensions of the original rectangle --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the result of folding and cutting the rectangle --/
def fold_and_cut (rect : Rectangle) (cut : CutResult) : Set ℕ :=
  match cut with
  | CutResult.OppositeMiddle => {11, 13}
  | CutResult.AdjacentMiddle => {31, 36, 37, 43}

/-- Theorem stating the result of folding and cutting the specific rectangle --/
theorem fold_cut_result (rect : Rectangle) (h1 : rect.width = 10) (h2 : rect.height = 12) :
  (fold_and_cut rect CutResult.OppositeMiddle = {11, 13}) ∧
  (fold_and_cut rect CutResult.AdjacentMiddle = {31, 36, 37, 43}) := by
  sorry

#check fold_cut_result

end NUMINAMATH_CALUDE_fold_cut_result_l3243_324332


namespace NUMINAMATH_CALUDE_candy_packing_problem_l3243_324397

theorem candy_packing_problem (n : ℕ) : 
  n % 10 = 6 ∧ 
  n % 15 = 11 ∧ 
  200 ≤ n ∧ n ≤ 250 → 
  n = 206 ∨ n = 236 := by
sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l3243_324397


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3243_324301

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₀ + a₁ + a₂ + a₃ = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3243_324301


namespace NUMINAMATH_CALUDE_divisibility_implies_five_divisor_l3243_324324

theorem divisibility_implies_five_divisor (n : ℕ) : 
  n > 1 → (6^n - 1) % n = 0 → n % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_five_divisor_l3243_324324


namespace NUMINAMATH_CALUDE_trapezoid_area_division_l3243_324370

/-- Represents a trapezoid with a diagonal and a parallel line -/
structure Trapezoid where
  /-- The ratio in which the diagonal divides the area -/
  diagonal_ratio : Rat
  /-- The ratio in which the parallel line divides the area -/
  parallel_line_ratio : Rat

/-- Theorem about area division in a specific trapezoid -/
theorem trapezoid_area_division (T : Trapezoid) 
  (h : T.diagonal_ratio = 3 / 7) : 
  T.parallel_line_ratio = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_division_l3243_324370


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3243_324339

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + Real.sqrt 3 * z = 1) : 
  ∀ (a b c : ℝ), a^2 + b^2 + c^2 ≥ (1/8 : ℝ) ∧ 
  (∃ (x₀ y₀ z₀ : ℝ), x₀^2 + y₀^2 + z₀^2 = (1/8 : ℝ) ∧ x₀ + 2*y₀ + Real.sqrt 3 * z₀ = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3243_324339


namespace NUMINAMATH_CALUDE_area_of_triangle_FQH_area_of_triangle_FQH_proof_l3243_324330

-- Define the rectangle EFGH
structure Rectangle where
  EF : ℝ
  EH : ℝ

-- Define the trapezoid PRHG
structure Trapezoid where
  EP : ℝ
  area : ℝ

-- Define the problem setup
def problem (rect : Rectangle) (trap : Trapezoid) : Prop :=
  rect.EF = 16 ∧ 
  trap.EP = 8 ∧
  trap.area = 160

-- Theorem statement
theorem area_of_triangle_FQH (rect : Rectangle) (trap : Trapezoid) 
  (h : problem rect trap) : ℝ :=
  80

-- Proof
theorem area_of_triangle_FQH_proof (rect : Rectangle) (trap : Trapezoid) 
  (h : problem rect trap) : area_of_triangle_FQH rect trap h = 80 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_FQH_area_of_triangle_FQH_proof_l3243_324330


namespace NUMINAMATH_CALUDE_split_cost_12_cupcakes_at_1_50_l3243_324360

/-- The amount each person pays when two people buy cupcakes and split the cost evenly -/
def split_cost (num_cupcakes : ℕ) (price_per_cupcake : ℚ) : ℚ :=
  (num_cupcakes : ℚ) * price_per_cupcake / 2

/-- Theorem: When two people buy 12 cupcakes at $1.50 each and split the cost evenly, each person pays $9.00 -/
theorem split_cost_12_cupcakes_at_1_50 :
  split_cost 12 (3/2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_split_cost_12_cupcakes_at_1_50_l3243_324360


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l3243_324329

theorem fraction_sum_equation (x : ℝ) : 
  (7 / (x - 2) + x / (2 - x) = 4) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l3243_324329


namespace NUMINAMATH_CALUDE_range_of_m_l3243_324327

/-- A function f(x) = x^2 - 2x + m where x is a real number -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

/-- The theorem stating the range of m given the conditions -/
theorem range_of_m (m : ℝ) : 
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0) → 
  (∀ x, f m (1 - x) ≥ -1) → 
  m ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3243_324327


namespace NUMINAMATH_CALUDE_inequality_problem_l3243_324349

theorem inequality_problem (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr : p * r > q * r) :
  ¬(-p > -q) ∧ ¬(-p > q) ∧ ¬(1 > -q/p) ∧ ¬(1 < q/p) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l3243_324349


namespace NUMINAMATH_CALUDE_gavin_dreams_total_l3243_324384

/-- The number of dreams Gavin has per day this year -/
def dreams_per_day : ℕ := 4

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- The number of dreams Gavin had this year -/
def dreams_this_year : ℕ := dreams_per_day * days_per_year

/-- The number of dreams Gavin had last year -/
def dreams_last_year : ℕ := 2 * dreams_this_year

/-- The total number of dreams Gavin had in two years -/
def total_dreams : ℕ := dreams_this_year + dreams_last_year

theorem gavin_dreams_total : total_dreams = 4380 := by
  sorry

end NUMINAMATH_CALUDE_gavin_dreams_total_l3243_324384


namespace NUMINAMATH_CALUDE_abcd_over_hife_value_l3243_324346

theorem abcd_over_hife_value (a b c d e f g h i : ℝ) 
  (hab : a / b = 1 / 3)
  (hbc : b / c = 2)
  (hcd : c / d = 1 / 2)
  (hde : d / e = 3)
  (hef : e / f = 1 / 10)
  (hfg : f / g = 3 / 4)
  (hgh : g / h = 1 / 5)
  (hhi : h / i = 5)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0) :
  a * b * c * d / (h * i * f * e) = 432 / 25 := by
  sorry

end NUMINAMATH_CALUDE_abcd_over_hife_value_l3243_324346


namespace NUMINAMATH_CALUDE_rocky_first_round_knockouts_l3243_324328

def total_fights : ℕ := 190
def knockout_percentage : ℚ := 1/2
def first_round_knockout_percentage : ℚ := 1/5

theorem rocky_first_round_knockouts :
  (total_fights : ℚ) * knockout_percentage * first_round_knockout_percentage = 19 := by
  sorry

end NUMINAMATH_CALUDE_rocky_first_round_knockouts_l3243_324328


namespace NUMINAMATH_CALUDE_bus_speed_problem_l3243_324316

theorem bus_speed_problem (bus_length : ℝ) (fast_bus_speed : ℝ) (passing_time : ℝ) :
  bus_length = 3125 →
  fast_bus_speed = 40 →
  passing_time = 50/3600 →
  ∃ (slow_bus_speed : ℝ),
    slow_bus_speed = (2 * bus_length / 1000) / passing_time - fast_bus_speed ∧
    slow_bus_speed = 410 :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l3243_324316


namespace NUMINAMATH_CALUDE_complement_of_A_l3243_324300

-- Define the universal set U
def U : Set ℝ := {x | x < 4}

-- Define set A
def A : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_of_A : 
  (U \ A) = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3243_324300


namespace NUMINAMATH_CALUDE_reflected_light_is_two_thirds_l3243_324361

/-- A mirror that reflects half the light shined on it back and passes the other half onward -/
structure FiftyPercentMirror :=
  (reflect : ℝ → ℝ)
  (pass : ℝ → ℝ)
  (reflect_half : ∀ x, reflect x = x / 2)
  (pass_half : ∀ x, pass x = x / 2)

/-- Two parallel fifty percent mirrors -/
structure TwoParallelMirrors :=
  (mirror1 : FiftyPercentMirror)
  (mirror2 : FiftyPercentMirror)

/-- The fraction of light reflected back to the left by two parallel fifty percent mirrors -/
def reflected_light (mirrors : TwoParallelMirrors) (initial_light : ℝ) : ℝ :=
  sorry

/-- Theorem: The total fraction of light reflected back to the left by two parallel "fifty percent mirrors" is 2/3 when light is shined from the left -/
theorem reflected_light_is_two_thirds (mirrors : TwoParallelMirrors) (initial_light : ℝ) :
  reflected_light mirrors initial_light = 2/3 * initial_light :=
sorry

end NUMINAMATH_CALUDE_reflected_light_is_two_thirds_l3243_324361


namespace NUMINAMATH_CALUDE_problem_solution_l3243_324351

noncomputable def equation (x a : ℝ) : Prop := Real.arctan (x / 2) + Real.arctan (2 - x) = a

theorem problem_solution :
  (∀ x : ℝ, equation x (π / 4) → Real.arccos (x / 2) = 2*π/3 ∨ Real.arccos (x / 2) = 0) ∧
  (∀ a : ℝ, (∃ x : ℝ, equation x a) → a ∈ Set.Icc (Real.arctan (1 / (-2 * Real.sqrt 10 - 6))) (Real.arctan (1 / (2 * Real.sqrt 10 - 6)))) ∧
  (∀ a : ℝ, (∃ α β : ℝ, α ≠ β ∧ α ∈ Set.Icc 5 15 ∧ β ∈ Set.Icc 5 15 ∧ equation α a ∧ equation β a) →
    (∀ γ δ : ℝ, γ ≠ δ ∧ γ ∈ Set.Icc 5 15 ∧ δ ∈ Set.Icc 5 15 ∧ equation γ a ∧ equation δ a → γ + δ ≤ 19)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3243_324351


namespace NUMINAMATH_CALUDE_smallest_multiple_of_three_l3243_324388

def cards : List ℕ := [1, 2, 6]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : ℕ), a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧ n = 10 * a + b

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3 * k

theorem smallest_multiple_of_three :
  ∃ (n : ℕ), is_valid_number n ∧ is_multiple_of_three n ∧
  ∀ (m : ℕ), is_valid_number m → is_multiple_of_three m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_three_l3243_324388


namespace NUMINAMATH_CALUDE_problem_statement_l3243_324304

theorem problem_statement :
  -- Part 1
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 1/a + 1/b ≥ 4) ∧
  -- Part 2
  (∃ min : ℝ, min = 1/14 ∧
    ∀ x y z : ℝ, x + 2*y + 3*z = 1 → x^2 + y^2 + z^2 ≥ min ∧
    ∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + 3*z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = min) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3243_324304


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l3243_324345

theorem reporters_covering_local_politics
  (total_reporters : ℕ)
  (h1 : total_reporters > 0)
  (politics_not_local : Real)
  (h2 : politics_not_local = 0.4)
  (not_politics : Real)
  (h3 : not_politics = 0.7) :
  (1 - politics_not_local) * (1 - not_politics) * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l3243_324345


namespace NUMINAMATH_CALUDE_sum_of_f_values_l3243_324325

/-- Function f defined on rational numbers -/
def f (a b c : ℚ) : ℚ := a^2 + 2*b*c

/-- Theorem stating that the sum of specific f values equals 10000 -/
theorem sum_of_f_values : f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l3243_324325


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3243_324341

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 3 ∧ x₂ = -2) ∧ 
  ((2 * x₁ - 1)^2 - 25 = 0) ∧ 
  ((2 * x₂ - 1)^2 - 25 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3243_324341


namespace NUMINAMATH_CALUDE_total_seedlings_l3243_324389

/-- Given that each packet contains 7 seeds and there are 60 packets,
    prove that the total number of seedlings is 420. -/
theorem total_seedlings (seeds_per_packet : ℕ) (num_packets : ℕ) 
  (h1 : seeds_per_packet = 7) 
  (h2 : num_packets = 60) : 
  seeds_per_packet * num_packets = 420 := by
sorry

end NUMINAMATH_CALUDE_total_seedlings_l3243_324389


namespace NUMINAMATH_CALUDE_orange_harvest_sacks_l3243_324318

/-- Proves that harvesting 38 sacks per day for 49 days results in 1862 sacks total. -/
theorem orange_harvest_sacks (daily_harvest : ℕ) (days : ℕ) (total_sacks : ℕ) 
  (h1 : daily_harvest = 38)
  (h2 : days = 49)
  (h3 : total_sacks = 1862) :
  daily_harvest * days = total_sacks :=
by sorry

end NUMINAMATH_CALUDE_orange_harvest_sacks_l3243_324318


namespace NUMINAMATH_CALUDE_max_value_of_function_l3243_324322

theorem max_value_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 9 ≤ 6) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 9 = 6) →
  a = 3 ∨ a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3243_324322


namespace NUMINAMATH_CALUDE_susan_decade_fraction_l3243_324376

/-- Represents the collection of quarters Susan has -/
structure QuarterCollection where
  total : ℕ
  decade_count : ℕ

/-- The fraction of quarters representing states that joined the union in a specific decade -/
def decade_fraction (c : QuarterCollection) : ℚ :=
  c.decade_count / c.total

/-- Susan's collection of quarters -/
def susan_collection : QuarterCollection :=
  { total := 22, decade_count := 7 }

theorem susan_decade_fraction :
  decade_fraction susan_collection = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_susan_decade_fraction_l3243_324376


namespace NUMINAMATH_CALUDE_quartic_polynomial_e_value_l3243_324315

/-- A polynomial of degree 4 with integer coefficients -/
structure QuarticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  e : ℤ

/-- The sum of coefficients of the polynomial -/
def QuarticPolynomial.sumCoeffs (p : QuarticPolynomial) : ℤ :=
  p.a + p.b + p.c + p.e

/-- Predicate for a polynomial having all negative integer roots -/
def hasAllNegativeIntegerRoots (p : QuarticPolynomial) : Prop :=
  ∃ (s₁ s₂ s₃ s₄ : ℕ+), 
    p.a = s₁ + s₂ + s₃ + s₄ ∧
    p.b = s₁*s₂ + s₁*s₃ + s₁*s₄ + s₂*s₃ + s₂*s₄ + s₃*s₄ ∧
    p.c = s₁*s₂*s₃ + s₁*s₂*s₄ + s₁*s₃*s₄ + s₂*s₃*s₄ ∧
    p.e = s₁*s₂*s₃*s₄

theorem quartic_polynomial_e_value (p : QuarticPolynomial) 
  (h1 : hasAllNegativeIntegerRoots p) 
  (h2 : p.sumCoeffs = 2023) : 
  p.e = 1540 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_e_value_l3243_324315


namespace NUMINAMATH_CALUDE_sum_abcd_l3243_324365

theorem sum_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ 
       b + 3 = c + 4 ∧ 
       c + 4 = d + 5 ∧ 
       d + 5 = a + b + c + d + 15) : 
  a + b + c + d = -46/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_l3243_324365


namespace NUMINAMATH_CALUDE_no_real_solutions_l3243_324352

theorem no_real_solutions : ∀ x : ℝ, (x^2000 / 2001 + 2 * Real.sqrt 3 * x^2 - 2 * Real.sqrt 5 * x + Real.sqrt 3) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3243_324352


namespace NUMINAMATH_CALUDE_root_sum_l3243_324381

theorem root_sum (n m : ℝ) (h1 : n ≠ 0) (h2 : n^2 + m*n + 3*n = 0) : m + n = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_l3243_324381


namespace NUMINAMATH_CALUDE_abc_product_values_l3243_324340

theorem abc_product_values (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5) (eq2 : b + 1/c = 2) (eq3 : c + 1/a = 8/3) :
  a * b * c = 1 ∨ a * b * c = 37/3 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_values_l3243_324340


namespace NUMINAMATH_CALUDE_car_y_time_is_one_third_correct_graph_is_c_l3243_324357

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The scenario of two cars traveling the same distance -/
def TwoCarScenario (x y : Car) : Prop :=
  x.distance = y.distance ∧ y.speed = 3 * x.speed

/-- Theorem: In the given scenario, Car Y's time is one-third of Car X's time -/
theorem car_y_time_is_one_third (x y : Car) 
  (h : TwoCarScenario x y) : y.time = x.time / 3 := by
  sorry

/-- Theorem: The correct graph representation matches option C -/
theorem correct_graph_is_c (x y : Car) 
  (h : TwoCarScenario x y) : 
  (x.speed = y.speed / 3 ∧ x.time = y.time * 3) := by
  sorry

end NUMINAMATH_CALUDE_car_y_time_is_one_third_correct_graph_is_c_l3243_324357


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3243_324308

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x, x^2 - 18*x + 72 = (x - a)*(x - b)) : 
  2*b - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3243_324308


namespace NUMINAMATH_CALUDE_greatest_plants_per_row_l3243_324372

theorem greatest_plants_per_row (sunflowers corn tomatoes : ℕ) 
  (h1 : sunflowers = 45)
  (h2 : corn = 81)
  (h3 : tomatoes = 63) :
  Nat.gcd sunflowers (Nat.gcd corn tomatoes) = 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_plants_per_row_l3243_324372


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_product_l3243_324367

/-- Given a triangle ABC with angles α, β, and γ, 
    the sum of the tangents of these angles equals the product of their tangents. -/
theorem triangle_tangent_sum_product (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_product_l3243_324367


namespace NUMINAMATH_CALUDE_ababab_divisible_by_13_l3243_324347

theorem ababab_divisible_by_13 (a b : Nat) (h1 : a < 10) (h2 : b < 10) :
  ∃ k : Nat, 100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b = 13 * k := by
  sorry

end NUMINAMATH_CALUDE_ababab_divisible_by_13_l3243_324347


namespace NUMINAMATH_CALUDE_car_speed_problem_l3243_324362

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_fraction : ℝ) 
  (h1 : distance = 720)
  (h2 : original_time = 8)
  (h3 : new_time_fraction = 5/8) :
  let new_time := new_time_fraction * original_time
  let new_speed := distance / new_time
  new_speed = 144 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3243_324362


namespace NUMINAMATH_CALUDE_combined_work_time_l3243_324359

theorem combined_work_time (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a = 21 → b = 6 → c = 12 → (1 / (1/a + 1/b + 1/c) : ℝ) = 84/25 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_time_l3243_324359


namespace NUMINAMATH_CALUDE_gabriel_forgotten_days_l3243_324390

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Gabriel took his capsules -/
def days_capsules_taken : ℕ := 28

/-- The number of days Gabriel forgot to take his capsules -/
def days_forgotten : ℕ := days_in_july - days_capsules_taken

theorem gabriel_forgotten_days :
  days_forgotten = 3 := by sorry

end NUMINAMATH_CALUDE_gabriel_forgotten_days_l3243_324390


namespace NUMINAMATH_CALUDE_janes_drawing_paper_l3243_324338

/-- The number of old, brown sheets of drawing paper Jane has. -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has. -/
def yellow_sheets : ℕ := 27

/-- The total number of sheets of drawing paper Jane has. -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem janes_drawing_paper : total_sheets = 55 := by
  sorry

end NUMINAMATH_CALUDE_janes_drawing_paper_l3243_324338


namespace NUMINAMATH_CALUDE_problem_solution_l3243_324320

theorem problem_solution (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3243_324320


namespace NUMINAMATH_CALUDE_yearly_income_is_130_l3243_324391

/-- Calculates the yearly simple interest income given principal and rate -/
def simple_interest (principal : ℕ) (rate : ℕ) : ℕ :=
  principal * rate / 100

/-- Proves that the yearly annual income is 130 given the specified conditions -/
theorem yearly_income_is_130 (total : ℕ) (part1 : ℕ) (rate1 : ℕ) (rate2 : ℕ) 
  (h1 : total = 2500)
  (h2 : part1 = 2000)
  (h3 : rate1 = 5)
  (h4 : rate2 = 6) :
  simple_interest part1 rate1 + simple_interest (total - part1) rate2 = 130 := by
  sorry

#eval simple_interest 2000 5 + simple_interest 500 6

end NUMINAMATH_CALUDE_yearly_income_is_130_l3243_324391


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_both_zero_l3243_324326

theorem sqrt_sum_zero_implies_both_zero (x y : ℝ) :
  Real.sqrt x + Real.sqrt y = 0 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_both_zero_l3243_324326


namespace NUMINAMATH_CALUDE_range_of_a_l3243_324302

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 + x - 2 > 0
def condition_q (x a : ℝ) : Prop := x > a

-- Define the sufficient but not necessary relationship
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬(q x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, 
    (sufficient_not_necessary (condition_p) (condition_q a)) → 
    a ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3243_324302
