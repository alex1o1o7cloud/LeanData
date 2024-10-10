import Mathlib

namespace intersection_chord_length_l2140_214038

/-- The line C in the Cartesian plane -/
def line_C (x y : ℝ) : Prop := x - y - 1 = 0

/-- The circle P in the Cartesian plane -/
def circle_P (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- The theorem stating that the length of the chord formed by the intersection
    of line C and circle P is √2 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line_C A.1 A.2 ∧ line_C B.1 B.2 ∧
    circle_P A.1 A.2 ∧ circle_P B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
sorry

end intersection_chord_length_l2140_214038


namespace hannah_dessert_cost_l2140_214099

def county_fair_problem (initial_amount : ℝ) (amount_left : ℝ) : Prop :=
  let total_spent := initial_amount - amount_left
  let rides_cost := initial_amount / 2
  let dessert_cost := total_spent - rides_cost
  dessert_cost = 5

theorem hannah_dessert_cost :
  county_fair_problem 30 10 := by
  sorry

end hannah_dessert_cost_l2140_214099


namespace sin_360_degrees_equals_zero_l2140_214034

theorem sin_360_degrees_equals_zero : Real.sin (2 * Real.pi) = 0 := by
  sorry

end sin_360_degrees_equals_zero_l2140_214034


namespace matrix_addition_and_scalar_multiplication_l2140_214031

theorem matrix_addition_and_scalar_multiplication :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-2, 1; 0, 3]
  A + 3 • B = !![-2, 0; 2, 14] := by sorry

end matrix_addition_and_scalar_multiplication_l2140_214031


namespace circle_area_outside_triangle_l2140_214049

/-- Given a right triangle ABC with ∠BAC = 90° and AB = 6, and a circle tangent to AB at X and AC at Y 
    with points diametrically opposite X and Y lying on BC, the area of the portion of the circle 
    that lies outside the triangle is 18π - 18. -/
theorem circle_area_outside_triangle (A B C X Y : ℝ × ℝ) (r : ℝ) : 
  -- Triangle ABC is a right triangle with ∠BAC = 90°
  (A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 6 ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = 6) →
  -- Circle is tangent to AB at X and AC at Y
  (X.1 = r ∧ X.2 = 0 ∧ Y.1 = 0 ∧ Y.2 = r) →
  -- Points diametrically opposite X and Y lie on BC
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 →
  -- The area of the portion of the circle outside the triangle
  π * r^2 - (B.1 * C.2 / 2) = 18 * π - 18 := by
sorry

end circle_area_outside_triangle_l2140_214049


namespace parallel_lines_condition_l2140_214042

-- Define the lines
def line1 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 1 = 0
def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define parallelism
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂, f x₁ y₁ → f x₂ y₂ → g x₁ y₁ → g x₂ y₂ → 
    (y₂ - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x₂ - x₁)

-- Theorem statement
theorem parallel_lines_condition (a : ℝ) :
  parallel (line1 a) (line2 a) → a = -1 ∨ a = 2 := by
  sorry

end parallel_lines_condition_l2140_214042


namespace polynomial_has_negative_root_l2140_214032

theorem polynomial_has_negative_root : ∃ x : ℝ, x < 0 ∧ x^7 + 2*x^5 + 5*x^3 - x + 12 = 0 := by
  sorry

end polynomial_has_negative_root_l2140_214032


namespace fraction_relation_l2140_214066

theorem fraction_relation (p r t u : ℚ) 
  (h1 : p / r = 8)
  (h2 : t / r = 5)
  (h3 : t / u = 2 / 3) :
  u / p = 15 / 16 := by
sorry

end fraction_relation_l2140_214066


namespace exists_unvisited_planet_l2140_214087

/-- A type representing a planet in the solar system -/
structure Planet where
  id : ℕ

/-- A function that returns the closest planet to a given planet -/
def closest_planet (planets : Finset Planet) : Planet → Planet :=
  sorry

theorem exists_unvisited_planet (n : ℕ) (h : n ≥ 1) :
  ∀ (planets : Finset Planet),
    Finset.card planets = 2 * n + 1 →
    (∀ p q : Planet, p ∈ planets → q ∈ planets → p ≠ q → 
      closest_planet planets p ≠ closest_planet planets q) →
    ∃ p : Planet, p ∈ planets ∧ 
      ∀ q : Planet, q ∈ planets → closest_planet planets q ≠ p :=
sorry

end exists_unvisited_planet_l2140_214087


namespace real_solutions_iff_a_in_range_l2140_214016

/-- Given a system of equations with real parameter a, 
    prove that real solutions exist if and only if 1 ≤ a ≤ 2 -/
theorem real_solutions_iff_a_in_range (a : ℝ) :
  (∃ x y : ℝ, x + y = a * (Real.sqrt x - Real.sqrt y) ∧
               x^2 + y^2 = a * (Real.sqrt x - Real.sqrt y)^2) ↔
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end real_solutions_iff_a_in_range_l2140_214016


namespace staircase_extension_l2140_214094

/-- Calculates the number of additional toothpicks needed to extend a staircase -/
def additional_toothpicks (initial_steps : ℕ) (final_steps : ℕ) (initial_toothpicks : ℕ) (increase_rate : ℕ) : ℕ :=
  sorry

/-- Theorem: Given a 4-step staircase with 28 toothpicks and an increase rate of 3,
    33 additional toothpicks are needed to build a 6-step staircase -/
theorem staircase_extension :
  additional_toothpicks 4 6 28 3 = 33 :=
sorry

end staircase_extension_l2140_214094


namespace f_decreasing_on_interval_l2140_214082

-- Define the function f(x) = x³ - x² - x
def f (x : ℝ) := x^3 - x^2 - x

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1/3 : ℝ) 1, 
    ∀ y ∈ Set.Ioo (-1/3 : ℝ) 1, 
      x < y → f x > f y :=
by sorry

end f_decreasing_on_interval_l2140_214082


namespace cyrus_day4_pages_l2140_214012

/-- Represents the number of pages written on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Represents the book writing problem --/
structure BookWritingProblem where
  totalPages : ℕ
  pagesWritten : DailyPages
  remainingPages : ℕ

/-- The specific instance of the book writing problem --/
def cyrusProblem : BookWritingProblem where
  totalPages := 500
  pagesWritten := {
    day1 := 25,
    day2 := 50,
    day3 := 100,
    day4 := 10  -- This is what we want to prove
  }
  remainingPages := 315

/-- Theorem stating that Cyrus wrote 10 pages on day 4 --/
theorem cyrus_day4_pages : 
  cyrusProblem.pagesWritten.day4 = 10 ∧
  cyrusProblem.pagesWritten.day2 = 2 * cyrusProblem.pagesWritten.day1 ∧
  cyrusProblem.pagesWritten.day3 = 2 * cyrusProblem.pagesWritten.day2 ∧
  cyrusProblem.totalPages = 
    cyrusProblem.pagesWritten.day1 + 
    cyrusProblem.pagesWritten.day2 + 
    cyrusProblem.pagesWritten.day3 + 
    cyrusProblem.pagesWritten.day4 + 
    cyrusProblem.remainingPages := by
  sorry

end cyrus_day4_pages_l2140_214012


namespace xiao_hua_first_place_l2140_214065

def fish_counts : List Nat := [23, 20, 15, 18, 13]

def xiao_hua_count : Nat := 20

def min_additional_fish (counts : List Nat) (xiao_hua : Nat) : Nat :=
  match counts.maximum? with
  | none => 0
  | some max_count => max_count - xiao_hua + 1

theorem xiao_hua_first_place (counts : List Nat) (xiao_hua : Nat) :
  counts = fish_counts ∧ xiao_hua = xiao_hua_count →
  min_additional_fish counts xiao_hua = 4 :=
by sorry

end xiao_hua_first_place_l2140_214065


namespace distance_to_line_l2140_214060

/-- The point from which we're measuring the distance -/
def P : ℝ × ℝ × ℝ := (0, 1, 5)

/-- The point on the line -/
def Q : ℝ → ℝ × ℝ × ℝ := λ t => (4 + 3*t, 5 - t, 6 + 2*t)

/-- The direction vector of the line -/
def v : ℝ × ℝ × ℝ := (3, -1, 2)

/-- The distance from a point to a line -/
def distanceToLine (P : ℝ × ℝ × ℝ) (Q : ℝ → ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ := 
  sorry

theorem distance_to_line : 
  distanceToLine P Q v = Real.sqrt 1262 / 7 := by sorry

end distance_to_line_l2140_214060


namespace roots_relation_l2140_214088

-- Define the original quadratic equation
def original_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the original equation
def root1 (a b c : ℝ) : ℝ := sorry
def root2 (a b c : ℝ) : ℝ := sorry

-- Define the new quadratic equation
def new_equation (a b c y : ℝ) : Prop := a^2 * y^2 + a * (b - c) * y - b * c = 0

-- State the theorem
theorem roots_relation (a b c : ℝ) (ha : a ≠ 0) :
  (∃ y1 y2 : ℝ, new_equation a b c y1 ∧ new_equation a b c y2 ∧
    y1 = root1 a b c + root2 a b c ∧
    y2 = root1 a b c * root2 a b c) :=
sorry

end roots_relation_l2140_214088


namespace dart_probability_l2140_214033

/-- Represents an equilateral triangle divided into regions -/
structure DividedTriangle where
  total_regions : ℕ
  shaded_regions : ℕ
  h_positive : 0 < total_regions
  h_shaded_le_total : shaded_regions ≤ total_regions

/-- The probability of a dart landing in a shaded region -/
def shaded_probability (triangle : DividedTriangle) : ℚ :=
  triangle.shaded_regions / triangle.total_regions

/-- The specific triangle described in the problem -/
def problem_triangle : DividedTriangle where
  total_regions := 6
  shaded_regions := 3
  h_positive := by norm_num
  h_shaded_le_total := by norm_num

theorem dart_probability :
  shaded_probability problem_triangle = 1/2 := by sorry

end dart_probability_l2140_214033


namespace wire_shapes_l2140_214080

/-- Given a wire of length 28 cm, prove properties about shapes formed from it -/
theorem wire_shapes (wire_length : ℝ) (h_wire : wire_length = 28) :
  let square_side : ℝ := wire_length / 4
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := wire_length / 2 - rectangle_length
  (square_side = 7 ∧ rectangle_width = 2) := by
  sorry

#check wire_shapes

end wire_shapes_l2140_214080


namespace polynomial_divisibility_implies_root_l2140_214055

theorem polynomial_divisibility_implies_root (r : ℝ) : 
  (∃ (p : ℝ → ℝ), (∀ x, 9 * x^3 - 6 * x^2 - 48 * x + 54 = (x - r)^2 * p x)) → 
  r = 4/3 := by
sorry

end polynomial_divisibility_implies_root_l2140_214055


namespace pentagon_angle_measure_l2140_214037

theorem pentagon_angle_measure :
  ∀ (a b c d e : ℝ),
  a + b + c + d + e = 540 →
  a = 111 →
  b = 113 →
  c = 92 →
  d = 128 →
  e = 96 :=
by
  sorry

end pentagon_angle_measure_l2140_214037


namespace not_parabola_l2140_214010

/-- The equation x² + ky² = 1 cannot represent a parabola for any real k -/
theorem not_parabola (k : ℝ) : 
  ¬ ∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    (x^2 + k*y^2 = 1) ↔ (a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0 ∧ b^2 = 4*a*c) :=
sorry

end not_parabola_l2140_214010


namespace express_w_in_terms_of_abc_l2140_214015

/-- Given distinct real numbers and a system of equations, prove the expression for w -/
theorem express_w_in_terms_of_abc (a b c w : ℝ) (x y z : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ w ≠ a ∧ w ≠ b ∧ w ≠ c ∧ 157 ≠ w ∧ 157 ≠ a ∧ 157 ≠ b ∧ 157 ≠ c)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  (a*b + a*c + b*c = 0 → w = -a*b/(a+b)) ∧ 
  (a*b + a*c + b*c ≠ 0 → w = -a*b*c/(a*b + a*c + b*c)) :=
sorry

end express_w_in_terms_of_abc_l2140_214015


namespace symmetric_points_mn_l2140_214096

/-- Given two points P and Q that are symmetric about the origin, prove that mn = -2 --/
theorem symmetric_points_mn (m n : ℝ) : 
  (m - n = -3 ∧ 1 = -(m + n)) → m * n = -2 := by
  sorry

end symmetric_points_mn_l2140_214096


namespace rectangular_prism_width_l2140_214081

theorem rectangular_prism_width (l w h : ℕ) : 
  l * w * h = 128 → 
  w = 2 * l → 
  w = 2 * h → 
  w + 2 = 10 :=
by sorry

end rectangular_prism_width_l2140_214081


namespace inequality_solution_l2140_214014

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 48 -/
theorem inequality_solution (a b c : ℝ) : 
  (∀ x, ((x - a) * (x - b)) / (x - c) ≥ 0 ↔ (x < -6 ∨ (20 ≤ x ∧ x ≤ 23))) →
  a < b →
  a + 2*b + 3*c = 48 := by
sorry

end inequality_solution_l2140_214014


namespace eight_cubic_polynomials_l2140_214056

/-- A polynomial function of degree at most 3 -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The condition that f(x) f(-x) = f(x^3) for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that there are exactly 8 cubic polynomials satisfying the condition -/
theorem eight_cubic_polynomials :
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔ SatisfiesCondition (CubicPolynomial a b c d)) ∧
    Finset.card s = 8 := by
  sorry


end eight_cubic_polynomials_l2140_214056


namespace salt_solution_mixture_l2140_214044

/-- Given a mixture of water and salt solution, calculate the volume of salt solution needed --/
theorem salt_solution_mixture (x : ℝ) : 
  (1 : ℝ) + x > 0 →  -- Total volume is positive
  0.6 * x = 0.2 * (1 + x) → -- Salt conservation equation
  x = 0.5 := by
sorry


end salt_solution_mixture_l2140_214044


namespace integral_of_exponential_l2140_214013

theorem integral_of_exponential (x : ℝ) :
  let f : ℝ → ℝ := λ x => (3^(7*x - 1/9)) / (7 * Real.log 3)
  (deriv f) x = 3^(7*x - 1/9) := by
  sorry

end integral_of_exponential_l2140_214013


namespace alligator_journey_time_l2140_214045

theorem alligator_journey_time (initial_time : ℝ) (return_time : ℝ) : 
  initial_time = 4 →
  return_time = initial_time + 2 * Real.sqrt initial_time →
  initial_time + return_time = 12 := by
  sorry

end alligator_journey_time_l2140_214045


namespace principal_mistake_l2140_214072

theorem principal_mistake : ¬∃ (x y : ℕ), 2 * x = 2 * y + 11 := by
  sorry

end principal_mistake_l2140_214072


namespace gideon_age_proof_l2140_214002

/-- The number of years in a century -/
def years_in_century : ℕ := 100

/-- Gideon's current age -/
def gideon_age : ℕ := 45

/-- The number of marbles Gideon has -/
def gideon_marbles : ℕ := years_in_century

/-- Gideon's age five years from now -/
def gideon_future_age : ℕ := gideon_age + 5

theorem gideon_age_proof :
  gideon_age = 45 ∧
  gideon_marbles = years_in_century ∧
  gideon_future_age = 2 * (gideon_marbles / 4) :=
by sorry

end gideon_age_proof_l2140_214002


namespace isosceles_trapezoid_diagonal_squared_l2140_214052

/-- An isosceles trapezoid with bases a and b, lateral side c, and diagonal d -/
structure IsoscelesTrapezoid (a b c d : ℝ) : Prop where
  bases_positive : 0 < a ∧ 0 < b
  lateral_positive : 0 < c
  diagonal_positive : 0 < d
  is_isosceles : true  -- This is a placeholder for the isosceles property

/-- The diagonal of an isosceles trapezoid satisfies d^2 = ab + c^2 -/
theorem isosceles_trapezoid_diagonal_squared 
  (a b c d : ℝ) (trap : IsoscelesTrapezoid a b c d) : 
  d^2 = a * b + c^2 := by
  sorry

end isosceles_trapezoid_diagonal_squared_l2140_214052


namespace min_distance_sum_l2140_214058

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 7 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := (-4, 0)

-- Define the fixed point A
def point_A : ℝ × ℝ := (1, 4)

-- Define a point on the right branch of the hyperbola
def is_on_right_branch (P : ℝ × ℝ) : Prop :=
  is_on_hyperbola P.1 P.2 ∧ P.1 > 0

-- Theorem statement
theorem min_distance_sum (P : ℝ × ℝ) (h : is_on_right_branch P) :
  dist P left_focus + dist P point_A ≥ 11 :=
sorry

end min_distance_sum_l2140_214058


namespace fraction_integer_values_fraction_values_l2140_214093

theorem fraction_integer_values (n : ℕ) : 
  (∃ k : ℤ, (8 * n + 157 : ℤ) / (4 * n + 7) = k) ↔ (n = 1 ∨ n = 34) :=
by sorry

theorem fraction_values (n : ℕ) :
  n = 1 → (8 * n + 157 : ℤ) / (4 * n + 7) = 15 ∧
  n = 34 → (8 * n + 157 : ℤ) / (4 * n + 7) = 3 :=
by sorry

end fraction_integer_values_fraction_values_l2140_214093


namespace root_product_zero_l2140_214079

theorem root_product_zero (x₁ x₂ x₃ : ℝ) :
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (Real.sqrt 2025) * x₁^3 - 4050 * x₁^2 - 4 = 0 ∧
  (Real.sqrt 2025) * x₂^3 - 4050 * x₂^2 - 4 = 0 ∧
  (Real.sqrt 2025) * x₃^3 - 4050 * x₃^2 - 4 = 0 →
  x₂ * (x₁ + x₃) = 0 := by
sorry

end root_product_zero_l2140_214079


namespace third_card_value_l2140_214018

def sum_of_permutations (a b x : ℕ) : ℕ :=
  100000 * a + 10000 * b + 100 * x +
  100000 * a + 10000 * x + b +
  100000 * b + 10000 * a + x +
  100000 * b + 10000 * x + a +
  100000 * x + 10000 * b + a +
  100000 * x + 10000 * a + b

theorem third_card_value (x : ℕ) :
  x < 100 →
  sum_of_permutations 18 75 x = 2606058 →
  x = 36 := by
sorry

end third_card_value_l2140_214018


namespace arithmetic_sequence_ratio_l2140_214026

def arithmeticSum (a1 : ℚ) (d : ℚ) (an : ℚ) : ℚ :=
  let n := (an - a1) / d + 1
  n * (a1 + an) / 2

theorem arithmetic_sequence_ratio : 
  let numerator := arithmeticSum 3 3 39
  let denominator := arithmeticSum 4 4 64
  numerator / denominator = 1 / 2 := by sorry

end arithmetic_sequence_ratio_l2140_214026


namespace barley_percentage_is_80_percent_l2140_214050

/-- Represents the percentage of land that is cleared -/
def cleared_percentage : ℝ := 0.9

/-- Represents the percentage of cleared land planted with potato -/
def potato_percentage : ℝ := 0.1

/-- Represents the area of cleared land planted with tomato in acres -/
def tomato_area : ℝ := 90

/-- Represents the approximate total land area in acres -/
def total_land : ℝ := 1000

/-- Theorem stating that the percentage of cleared land planted with barley is 80% -/
theorem barley_percentage_is_80_percent :
  let cleared_land := cleared_percentage * total_land
  let barley_percentage := 1 - potato_percentage - (tomato_area / cleared_land)
  barley_percentage = 0.8 := by sorry

end barley_percentage_is_80_percent_l2140_214050


namespace exists_triangle_area_not_greater_than_two_l2140_214019

/-- A lattice point in a 2D coordinate system -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Checks if a lattice point is within the 5x5 grid centered at the origin -/
def isWithinGrid (p : LatticePoint) : Prop :=
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Calculates the area of a triangle formed by three lattice points -/
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)| / 2

/-- Main theorem statement -/
theorem exists_triangle_area_not_greater_than_two 
  (points : Fin 6 → LatticePoint)
  (h_within_grid : ∀ i, isWithinGrid (points i))
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j → j ≠ k → i ≠ k → triangleArea (points i) (points j) (points k) ≤ 2 := by
  sorry

end exists_triangle_area_not_greater_than_two_l2140_214019


namespace new_mean_after_removal_l2140_214005

def original_mean : ℝ := 42
def original_count : ℕ := 65
def removed_score : ℝ := 50
def removed_count : ℕ := 6

theorem new_mean_after_removal :
  let original_sum := original_mean * original_count
  let removed_sum := removed_score * removed_count
  let new_sum := original_sum - removed_sum
  let new_count := original_count - removed_count
  let new_mean := new_sum / new_count
  ∃ ε > 0, |new_mean - 41.2| < ε :=
by
  sorry

end new_mean_after_removal_l2140_214005


namespace circle_properties_l2140_214035

-- Define the circle C: (x-2)²+y²=1
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point P(m,n) on the circle C
def P (m n : ℝ) : Prop := C m n

-- Theorem statement
theorem circle_properties :
  (∃ (m₀ n₀ : ℝ), P m₀ n₀ ∧ ∀ (m n : ℝ), P m n → |n / m| ≤ |n₀ / m₀| ∧ |n₀ / m₀| = Real.sqrt 3 / 3) ∧
  (∃ (m₁ n₁ : ℝ), P m₁ n₁ ∧ ∀ (m n : ℝ), P m n → m^2 + n^2 ≤ m₁^2 + n₁^2 ∧ m₁^2 + n₁^2 = 9) :=
by sorry

end circle_properties_l2140_214035


namespace unique_positive_solution_l2140_214059

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ x - 4 = 21 * (1/x) := by
  sorry

end unique_positive_solution_l2140_214059


namespace base8_addition_subtraction_l2140_214084

/-- Converts a base 8 number to base 10 --/
def base8ToBase10 (x : ℕ) : ℕ :=
  let ones := x % 10
  let eights := x / 10
  8 * eights + ones

/-- Converts a base 10 number to base 8 --/
def base10ToBase8 (x : ℕ) : ℕ :=
  let quotient := x / 8
  let remainder := x % 8
  10 * quotient + remainder

theorem base8_addition_subtraction :
  base10ToBase8 ((base8ToBase10 10 + base8ToBase10 26) - base8ToBase10 13) = 23 := by
  sorry

end base8_addition_subtraction_l2140_214084


namespace equation_true_when_x_is_three_l2140_214009

theorem equation_true_when_x_is_three : ∀ x : ℝ, x = 3 → 3 * x - 1 = 8 := by
  sorry

end equation_true_when_x_is_three_l2140_214009


namespace hall_dimensions_l2140_214041

/-- Given a rectangular hall with width half of its length and area 800 sq. m,
    prove that the difference between length and width is 20 meters. -/
theorem hall_dimensions (length width : ℝ) : 
  width = length / 2 →
  length * width = 800 →
  length - width = 20 :=
by sorry

end hall_dimensions_l2140_214041


namespace power_fraction_equality_l2140_214021

theorem power_fraction_equality : (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end power_fraction_equality_l2140_214021


namespace page_lines_increase_percentage_correct_increase_percentage_l2140_214025

theorem page_lines_increase_percentage : ℕ → ℝ → Prop :=
  fun original_lines increase_percentage =>
    let new_lines : ℕ := original_lines + 80
    new_lines = 240 →
    (increase_percentage * original_lines : ℝ) = 80 * 100

theorem correct_increase_percentage : 
  ∃ (original_lines : ℕ), page_lines_increase_percentage original_lines 50 := by
  sorry

end page_lines_increase_percentage_correct_increase_percentage_l2140_214025


namespace triangle_properties_l2140_214054

open Real

theorem triangle_properties (A B C a b c : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) (h5 : cos (2 * A) - 3 * cos (B + C) = 1) (h6 : a > 0 ∧ b > 0 ∧ c > 0) :
  -- Part 1
  A = π / 3 ∧
  -- Part 2
  (∃ S : Real, S = 5 * Real.sqrt 3 ∧ b = 5 → sin B * sin C = 5 / 7) ∧
  -- Part 3
  (a = 1 → ∃ l : Real, l = a + b + c ∧ 2 < l ∧ l ≤ 3) :=
by sorry

end triangle_properties_l2140_214054


namespace quadratic_vertex_coordinates_l2140_214027

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- Define the vertex coordinates
def vertex : ℝ × ℝ := (-1, 8)

-- Theorem statement
theorem quadratic_vertex_coordinates :
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end quadratic_vertex_coordinates_l2140_214027


namespace fourth_person_height_l2140_214053

/-- Heights of four people in increasing order -/
def Heights := Fin 4 → ℕ

/-- The condition that heights are in increasing order -/
def increasing_heights (h : Heights) : Prop :=
  ∀ i j, i < j → h i < h j

/-- The condition for the differences between heights -/
def height_differences (h : Heights) : Prop :=
  h 1 - h 0 = 2 ∧ h 2 - h 1 = 2 ∧ h 3 - h 2 = 6

/-- The condition for the average height -/
def average_height (h : Heights) : Prop :=
  (h 0 + h 1 + h 2 + h 3) / 4 = 79

theorem fourth_person_height (h : Heights) 
  (inc : increasing_heights h) 
  (diff : height_differences h) 
  (avg : average_height h) : 
  h 3 = 85 := by
  sorry

end fourth_person_height_l2140_214053


namespace min_expression_le_one_l2140_214048

theorem min_expression_le_one (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end min_expression_le_one_l2140_214048


namespace correct_replacement_l2140_214020

/-- Represents a digit in the addition problem -/
inductive Digit
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents whether a digit is correct or potentially incorrect -/
inductive DigitStatus
| correct
| potentiallyIncorrect

/-- Function to get the status of a digit -/
def digitStatus (d : Digit) : DigitStatus :=
  match d with
  | Digit.zero | Digit.one | Digit.three | Digit.four | Digit.five | Digit.six | Digit.eight => DigitStatus.correct
  | Digit.two | Digit.seven | Digit.nine => DigitStatus.potentiallyIncorrect

/-- Function to check if replacing a digit corrects the addition -/
def replacementCorrects (d : Digit) (replacement : Digit) : Prop :=
  match d, replacement with
  | Digit.two, Digit.six => True
  | _, _ => False

/-- Theorem stating that replacing 2 with 6 corrects the addition -/
theorem correct_replacement :
  ∃ (d : Digit) (replacement : Digit),
    digitStatus d = DigitStatus.potentiallyIncorrect ∧
    digitStatus replacement = DigitStatus.correct ∧
    replacementCorrects d replacement :=
by sorry

end correct_replacement_l2140_214020


namespace annie_completion_time_correct_l2140_214004

/-- Dan's time to complete the job alone -/
def dan_time : ℝ := 15

/-- Annie's time to complete the job alone -/
def annie_time : ℝ := 3.6

/-- Time Dan works before stopping -/
def dan_work_time : ℝ := 6

/-- Time Annie takes to finish the job after Dan stops -/
def annie_finish_time : ℝ := 6

/-- The theorem stating that Annie's time to complete the job alone is correct -/
theorem annie_completion_time_correct :
  (dan_work_time / dan_time) + (annie_finish_time / annie_time) = 1 := by
  sorry

end annie_completion_time_correct_l2140_214004


namespace percentage_loss_l2140_214024

theorem percentage_loss (cost_price selling_price : ℝ) 
  (h1 : cost_price = 1400)
  (h2 : selling_price = 1050) : 
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end percentage_loss_l2140_214024


namespace lines_intersect_at_point_l2140_214047

/-- Represents a 2D point -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form -/
structure ParametricLine where
  p : Point  -- Starting point
  v : Point  -- Direction vector

/-- The first line -/
def line1 : ParametricLine :=
  { p := { x := 1, y := 4 },
    v := { x := -2, y := 3 } }

/-- The second line -/
def line2 : ParametricLine :=
  { p := { x := 5, y := 2 },
    v := { x := 1, y := 6 } }

/-- A point on a parametric line -/
def pointOnLine (l : ParametricLine) (t : ℚ) : Point :=
  { x := l.p.x + t * l.v.x,
    y := l.p.y + t * l.v.y }

/-- The proposed intersection point -/
def intersectionPoint : Point :=
  { x := 21 / 5,
    y := -4 / 5 }

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry

end lines_intersect_at_point_l2140_214047


namespace power_of_fraction_cube_l2140_214064

theorem power_of_fraction_cube (x : ℝ) : ((1/2) * x^3)^2 = (1/4) * x^6 := by sorry

end power_of_fraction_cube_l2140_214064


namespace simplify_and_rationalize_l2140_214097

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l2140_214097


namespace infinitely_many_non_representable_l2140_214086

def is_representable (m : ℕ) : Prop :=
  ∃ (n p : ℕ), p.Prime ∧ m = n^2 + p

theorem infinitely_many_non_representable :
  ∀ k : ℕ, ∃ m : ℕ, m > k ∧ ¬ is_representable m :=
sorry

end infinitely_many_non_representable_l2140_214086


namespace triangle_abc_area_l2140_214006

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))

theorem triangle_abc_area :
  ∀ (A B C : ℝ) (a b c : ℝ),
  (Real.sin A - Real.sin B) * (Real.sin A + Real.sin B) = Real.sin A * Real.sin C - Real.sin C^2 →
  c = 2*a ∧ c = 2 * Real.sqrt 2 →
  triangle_area a b c = Real.sqrt 3 := by
  sorry

end triangle_abc_area_l2140_214006


namespace line_equation_from_slope_and_point_l2140_214003

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with slope 3 passing through (1, -2), its equation is 3x - y - 5 = 0 -/
theorem line_equation_from_slope_and_point :
  ∀ (l : Line),
  l.slope = 3 ∧ l.point = (1, -2) →
  ∃ (eq : LineEquation),
  eq.a = 3 ∧ eq.b = -1 ∧ eq.c = -5 ∧
  ∀ (x y : ℝ), eq.a * x + eq.b * y + eq.c = 0 ↔ y = l.slope * (x - l.point.1) + l.point.2 :=
by sorry

end line_equation_from_slope_and_point_l2140_214003


namespace class_president_election_l2140_214075

theorem class_president_election (total_votes : ℕ) 
  (emily_votes : ℕ) (fiona_votes : ℕ) : 
  emily_votes = total_votes / 4 →
  fiona_votes = total_votes / 3 →
  emily_votes + fiona_votes = 77 →
  total_votes = 132 := by
sorry

end class_president_election_l2140_214075


namespace max_value_problem_l2140_214078

theorem max_value_problem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 9 / 8 := by
  sorry

end max_value_problem_l2140_214078


namespace rectangle_area_is_eight_l2140_214043

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangle on a 2D grid -/
structure GridRectangle where
  bottomLeft : GridPoint
  topRight : GridPoint

/-- Calculates the area of a grid rectangle -/
def gridRectangleArea (rect : GridRectangle) : ℤ :=
  (rect.topRight.x - rect.bottomLeft.x) * (rect.topRight.y - rect.bottomLeft.y)

theorem rectangle_area_is_eight :
  let rect : GridRectangle := {
    bottomLeft := { x := 0, y := 0 },
    topRight := { x := 4, y := 2 }
  }
  gridRectangleArea rect = 8 := by
  sorry

end rectangle_area_is_eight_l2140_214043


namespace absolute_value_equation_l2140_214036

theorem absolute_value_equation : ∀ x : ℝ, 
  (abs x) * (abs (-25) - abs 5) = 40 ↔ x = 2 ∨ x = -2 := by
  sorry

end absolute_value_equation_l2140_214036


namespace banana_production_ratio_l2140_214068

/-- The ratio of banana production between Jakies Island and a nearby island -/
theorem banana_production_ratio :
  ∀ (jakies_multiple : ℕ) (nearby_production : ℕ) (total_production : ℕ),
  nearby_production = 9000 →
  total_production = 99000 →
  total_production = nearby_production + jakies_multiple * nearby_production →
  (jakies_multiple * nearby_production) / nearby_production = 10 :=
by
  sorry

#check banana_production_ratio

end banana_production_ratio_l2140_214068


namespace friends_receiving_pens_l2140_214061

/-- The number of friends Kendra and Tony will give pens to -/
def num_friends (kendra_packs tony_packs kendra_pens_per_pack tony_pens_per_pack kept_pens : ℕ) : ℕ :=
  kendra_packs * kendra_pens_per_pack + tony_packs * tony_pens_per_pack - 2 * kept_pens

/-- Theorem stating the number of friends Kendra and Tony will give pens to -/
theorem friends_receiving_pens :
  num_friends 7 5 4 6 3 = 52 := by
  sorry

end friends_receiving_pens_l2140_214061


namespace line_slope_problem_l2140_214029

/-- Given m > 0 and points (m, 4) and (2, m) lie on a line with slope m^2, prove m = 2 -/
theorem line_slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end line_slope_problem_l2140_214029


namespace combined_work_time_l2140_214067

def team_A_time : ℝ := 15
def team_B_time : ℝ := 30

theorem combined_work_time :
  1 / (1 / team_A_time + 1 / team_B_time) = 10 := by
  sorry

end combined_work_time_l2140_214067


namespace smallest_interesting_number_l2140_214085

theorem smallest_interesting_number : 
  ∃ (n : ℕ), n = 1800 ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), 2 * m = k ^ 2) ∨ ¬(∃ (l : ℕ), 15 * m = l ^ 3)) ∧
  (∃ (k : ℕ), 2 * n = k ^ 2) ∧
  (∃ (l : ℕ), 15 * n = l ^ 3) := by
sorry

end smallest_interesting_number_l2140_214085


namespace tylers_eggs_l2140_214011

/-- Given a recipe for 4 people requiring 2 eggs, prove that if Tyler needs to buy 1 more egg
    to make a cake for 8 people, then Tyler has 3 eggs in the fridge. -/
theorem tylers_eggs (recipe_eggs : ℕ) (people : ℕ) (scale_factor : ℕ) (eggs_to_buy : ℕ) : 
  recipe_eggs = 2 →
  people = 8 →
  scale_factor = people / 4 →
  eggs_to_buy = 1 →
  recipe_eggs * scale_factor - eggs_to_buy = 3 := by
  sorry

end tylers_eggs_l2140_214011


namespace number_divided_by_seven_l2140_214070

theorem number_divided_by_seven (x : ℝ) : x / 7 = 5 / 14 → x = 5 / 2 := by
  sorry

end number_divided_by_seven_l2140_214070


namespace degenerate_ellipse_c_l2140_214076

/-- The equation of a potentially degenerate ellipse -/
def ellipse_equation (x y c : ℝ) : Prop :=
  2 * x^2 + y^2 + 8 * x - 10 * y + c = 0

/-- A degenerate ellipse is represented by a single point -/
def is_degenerate_ellipse (c : ℝ) : Prop :=
  ∃! (x y : ℝ), ellipse_equation x y c

/-- The value of c for which the ellipse is degenerate -/
theorem degenerate_ellipse_c : 
  ∃! c : ℝ, is_degenerate_ellipse c ∧ c = 33 :=
sorry

end degenerate_ellipse_c_l2140_214076


namespace no_solution_for_sock_problem_l2140_214063

theorem no_solution_for_sock_problem :
  ¬∃ (n m : ℕ), n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1) : ℚ) = 1/2 := by
sorry

end no_solution_for_sock_problem_l2140_214063


namespace allen_reading_time_l2140_214077

/-- Calculates the number of days required to finish reading a book -/
def days_to_finish (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Proves that Allen took 12 days to finish reading the book -/
theorem allen_reading_time : days_to_finish 120 10 = 12 := by
  sorry

end allen_reading_time_l2140_214077


namespace polygon_sides_l2140_214040

theorem polygon_sides (n : ℕ) (h1 : n > 2) : 
  (140 + 145 * (n - 1) = 180 * (n - 2)) → n = 10 := by
  sorry

end polygon_sides_l2140_214040


namespace inequality_proof_l2140_214092

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end inequality_proof_l2140_214092


namespace polynomial_properties_l2140_214001

theorem polynomial_properties :
  (∀ x : ℝ, x^2 + 2*x - 3 = (x-1)*(x+3)) ∧
  (∀ x : ℝ, x^2 + 4*x + 5 ≥ 1) := by
sorry

end polynomial_properties_l2140_214001


namespace min_distinct_values_with_unique_mode_l2140_214095

theorem min_distinct_values_with_unique_mode (list_size : ℕ) (mode_frequency : ℕ) 
  (h1 : list_size = 3000)
  (h2 : mode_frequency = 15) :
  (∃ (distinct_values : ℕ), 
    distinct_values ≥ 215 ∧ 
    distinct_values * (mode_frequency - 1) + mode_frequency ≥ list_size ∧
    ∀ (n : ℕ), n < 215 → n * (mode_frequency - 1) + mode_frequency < list_size) :=
by sorry

end min_distinct_values_with_unique_mode_l2140_214095


namespace car_speed_problem_l2140_214071

theorem car_speed_problem (v : ℝ) (h1 : v > 0) : 
  (60 / v - 60 / (v + 20) = 0.5) → v = 40 := by
  sorry

end car_speed_problem_l2140_214071


namespace bus_seat_capacity_l2140_214057

theorem bus_seat_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let back_seat_capacity : ℕ := 12
  let total_capacity : ℕ := 93
  let seat_capacity : ℕ := (total_capacity - back_seat_capacity) / (left_seats + right_seats)
  seat_capacity = 3 := by sorry

end bus_seat_capacity_l2140_214057


namespace set_A_properties_l2140_214028

-- Define the set A
def A : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Theorem statement
theorem set_A_properties :
  (∀ n : ℤ, (2*n + 1) ∈ A) ∧
  (∀ k : ℤ, (4*k - 2) ∉ A) ∧
  (∀ a b : ℤ, a ∈ A → b ∈ A → (a * b) ∈ A) := by
  sorry

end set_A_properties_l2140_214028


namespace red_balls_count_l2140_214023

/-- Given a bag of balls with some red and some white balls, prove the number of red balls. -/
theorem red_balls_count (total_balls : ℕ) (red_prob : ℝ) (h_total : total_balls = 50) (h_prob : red_prob = 0.7) :
  ⌊(total_balls : ℝ) * red_prob⌋ = 35 := by
  sorry

end red_balls_count_l2140_214023


namespace EF_length_l2140_214022

-- Define the segment AB and points C, D, E, F
def AB : ℝ := 26
def AC : ℝ := 1
def AD : ℝ := 8

-- Define the semicircle with diameter AB
def semicircle (x y : ℝ) : Prop :=
  x ≥ 0 ∧ x ≤ AB ∧ y ≥ 0 ∧ x * (AB - x) = y^2

-- Define the perpendicularity condition
def perpendicular (x y : ℝ) : Prop :=
  semicircle x y ∧ (x = AC ∨ x = AD)

-- Theorem statement
theorem EF_length :
  ∃ (xE yE xF yF : ℝ),
    perpendicular xE yE ∧
    perpendicular xF yF ∧
    xE = AC ∧
    xF = AD ∧
    (yF - yE)^2 + (xF - xE)^2 = (7 * Real.sqrt 2)^2 :=
sorry

end EF_length_l2140_214022


namespace max_sum_of_vertex_products_l2140_214073

/-- Represents the set of numbers that can be assigned to cube faces -/
def CubeNumbers : Finset ℕ := {0, 1, 2, 3, 8, 9}

/-- A function that assigns numbers to cube faces -/
def FaceAssignment := Fin 6 → ℕ

/-- Predicate to check if a face assignment is valid -/
def ValidAssignment (f : FaceAssignment) : Prop :=
  (∀ i : Fin 6, f i ∈ CubeNumbers) ∧ (∀ i j : Fin 6, i ≠ j → f i ≠ f j)

/-- Calculate the product at a vertex given three face numbers -/
def VertexProduct (a b c : ℕ) : ℕ := a * b * c

/-- Calculate the sum of all vertex products for a given face assignment -/
def SumOfVertexProducts (f : FaceAssignment) : ℕ :=
  VertexProduct (f 0) (f 1) (f 2) +
  VertexProduct (f 0) (f 1) (f 3) +
  VertexProduct (f 0) (f 2) (f 4) +
  VertexProduct (f 0) (f 3) (f 4) +
  VertexProduct (f 1) (f 2) (f 5) +
  VertexProduct (f 1) (f 3) (f 5) +
  VertexProduct (f 2) (f 4) (f 5) +
  VertexProduct (f 3) (f 4) (f 5)

/-- The main theorem stating that the maximum sum of vertex products is 405 -/
theorem max_sum_of_vertex_products :
  ∃ (f : FaceAssignment), ValidAssignment f ∧
  SumOfVertexProducts f = 405 ∧
  ∀ (g : FaceAssignment), ValidAssignment g → SumOfVertexProducts g ≤ 405 := by
  sorry

end max_sum_of_vertex_products_l2140_214073


namespace alpha_value_l2140_214039

-- Define the triangle and point S
variable (P Q R S : Point)

-- Define the angles
variable (α β γ δ : ℝ)

-- Define the conditions
variable (triangle_PQR : Triangle P Q R)
variable (S_interior : InteriorPoint S triangle_PQR)
variable (QSP_bisected : AngleBisector S Q (Angle P S Q))
variable (delta_exterior : ExteriorAngle Q triangle_PQR δ)

-- Given angle values
variable (beta_value : β = 100)
variable (gamma_value : γ = 30)
variable (delta_value : δ = 150)

-- Theorem statement
theorem alpha_value : α = 215 := by sorry

end alpha_value_l2140_214039


namespace max_value_sqrt_product_max_value_achievable_l2140_214046

theorem max_value_sqrt_product (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1/2) : 
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1 / Real.sqrt 2 + 1 / 2 :=
by sorry

theorem max_value_achievable : 
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1/2 ∧
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) = 1 / Real.sqrt 2 + 1 / 2 :=
by sorry

end max_value_sqrt_product_max_value_achievable_l2140_214046


namespace min_sum_tangents_l2140_214090

theorem min_sum_tangents (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a = 2 * b * Real.sin C →  -- Given condition
  8 ≤ Real.tan A + Real.tan B + Real.tan C ∧
  (∃ (A' B' C' : Real), 0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π ∧
    Real.tan A' + Real.tan B' + Real.tan C' = 8) :=
by sorry

end min_sum_tangents_l2140_214090


namespace unique_number_satisfying_conditions_l2140_214062

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_nonzero (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def product_divisible_by_1000 (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (n * (10 * a + b) * a) % 1000 = 0

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_three_digit n ∧
             digits_nonzero n ∧
             product_divisible_by_1000 n ∧
             n = 875 := by sorry

end unique_number_satisfying_conditions_l2140_214062


namespace nadia_hannah_distance_ratio_l2140_214089

/-- Proves the ratio of Nadia's distance to Hannah's distance -/
theorem nadia_hannah_distance_ratio :
  ∀ (nadia_distance hannah_distance : ℕ) (k : ℕ),
    nadia_distance = 18 →
    nadia_distance + hannah_distance = 27 →
    nadia_distance = k * hannah_distance →
    nadia_distance / hannah_distance = 2 := by
  sorry

end nadia_hannah_distance_ratio_l2140_214089


namespace right_triangle_perimeter_l2140_214030

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h_area : area = 180) (h_leg : leg = 30) :
  ∃ (perimeter : ℝ), perimeter = 42 + 2 * Real.sqrt 261 := by
  sorry

end right_triangle_perimeter_l2140_214030


namespace heather_initial_blocks_l2140_214098

/-- The number of blocks Heather shared with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ended up with -/
def remaining_blocks : ℕ := 45

/-- The initial number of blocks Heather had -/
def initial_blocks : ℕ := shared_blocks + remaining_blocks

theorem heather_initial_blocks : initial_blocks = 86 := by
  sorry

end heather_initial_blocks_l2140_214098


namespace collinear_points_b_value_l2140_214017

theorem collinear_points_b_value :
  ∀ b : ℚ,
  let p1 : ℚ × ℚ := (4, -6)
  let p2 : ℚ × ℚ := (b + 3, 4)
  let p3 : ℚ × ℚ := (3*b + 4, 3)
  (p1.2 - p2.2) * (p2.1 - p3.1) = (p2.2 - p3.2) * (p1.1 - p2.1) →
  b = -3/7 := by
sorry

end collinear_points_b_value_l2140_214017


namespace convex_regular_polygon_integer_angles_l2140_214074

/-- The number of positive integers n ≥ 3 such that 360 is divisible by n -/
def count_divisors : Nat :=
  (Finset.filter (fun n => n ≥ 3 ∧ 360 % n = 0) (Finset.range 361)).card

/-- Theorem stating that there are exactly 22 positive integers n ≥ 3 
    such that 360 is divisible by n -/
theorem convex_regular_polygon_integer_angles : count_divisors = 22 := by
  sorry

end convex_regular_polygon_integer_angles_l2140_214074


namespace range_of_m_l2140_214007

theorem range_of_m (m : ℝ) : 
  let p := (1^2 + 1^2 - 2*m*1 + 2*m*1 + 2*m^2 - 4 < 0)
  let q := ∀ (x y : ℝ), m*x - y + 1 + 2*m = 0 → (x > 0 → y ≥ 0)
  (p ∨ q) ∧ ¬(p ∧ q) → ((-1 < m ∧ m < 0) ∨ m ≥ 1) :=
by sorry

end range_of_m_l2140_214007


namespace grading_orders_mod_100_l2140_214083

/-- The number of students --/
def num_students : ℕ := 40

/-- The number of problems per student --/
def problems_per_student : ℕ := 3

/-- The number of different grading orders --/
def N : ℕ := 2 * 3^(num_students - 2)

/-- Theorem stating the result of N modulo 100 --/
theorem grading_orders_mod_100 : N % 100 = 78 := by
  sorry

end grading_orders_mod_100_l2140_214083


namespace A_subset_B_l2140_214069

/-- Set A is defined as {x | x(x-1) < 0} -/
def A : Set ℝ := {x | x * (x - 1) < 0}

/-- Set B is defined as {y | y = x^2 for some real x} -/
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

/-- Theorem: A is a subset of B -/
theorem A_subset_B : A ⊆ B := by sorry

end A_subset_B_l2140_214069


namespace sqrt_29_minus_1_between_4_and_5_l2140_214051

theorem sqrt_29_minus_1_between_4_and_5 :
  let a : ℝ := Real.sqrt 29 - 1
  4 < a ∧ a < 5 := by sorry

end sqrt_29_minus_1_between_4_and_5_l2140_214051


namespace complement_of_A_in_U_l2140_214000

-- Define the universal set U
def U : Set ℕ := {x | 1 < x ∧ x < 5}

-- Define set A
def A : Set ℕ := {2, 3}

-- State the theorem
theorem complement_of_A_in_U :
  (U \ A) = {4} := by sorry

end complement_of_A_in_U_l2140_214000


namespace function_inequality_implies_k_range_l2140_214008

open Real

theorem function_inequality_implies_k_range (k : ℝ) : k > 0 → 
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
    (exp 2 * x₁ / exp x₁) / k ≤ (exp 2 * x₂^2 + 1) / (x₂ * (k + 1))) → 
  k ≥ 1 := by
sorry

end function_inequality_implies_k_range_l2140_214008


namespace smallest_part_of_proportional_division_l2140_214091

theorem smallest_part_of_proportional_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 120)
  (h_prop : a + b + c = 15)
  (h_a : a = 3)
  (h_b : b = 5)
  (h_c : c = 7) :
  min (total * a / (a + b + c)) (min (total * b / (a + b + c)) (total * c / (a + b + c))) = 24 :=
by sorry

end smallest_part_of_proportional_division_l2140_214091
