import Mathlib

namespace NUMINAMATH_CALUDE_fish_in_pond_l3827_382765

/-- Approximates the total number of fish in a pond based on a tag-and-recapture method. -/
def approximate_fish_count (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (initial_tagged * second_catch) / tagged_in_second

/-- The approximate number of fish in the pond given the tag-and-recapture data. -/
theorem fish_in_pond (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ)
    (h1 : initial_tagged = 50)
    (h2 : second_catch = 50)
    (h3 : tagged_in_second = 10) :
  approximate_fish_count initial_tagged second_catch tagged_in_second = 250 := by
  sorry

#eval approximate_fish_count 50 50 10

end NUMINAMATH_CALUDE_fish_in_pond_l3827_382765


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_rectangle_l3827_382753

/-- A rectangle containing an equilateral triangle -/
structure TriangleInRectangle where
  /-- The measure of one angle between the rectangle and triangle sides -/
  x : ℝ
  /-- The measure of the other angle between the rectangle and triangle sides -/
  y : ℝ
  /-- The rectangle has right angles -/
  rectangle_right_angles : x + y + 60 + 90 + 90 = 540
  /-- The inner triangle is equilateral -/
  equilateral_triangle : True

/-- The sum of angles x and y in a rectangle containing an equilateral triangle is 60° -/
theorem angle_sum_in_triangle_rectangle (t : TriangleInRectangle) : t.x + t.y = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_rectangle_l3827_382753


namespace NUMINAMATH_CALUDE_distance_between_trees_l3827_382727

theorem distance_between_trees (total_length : ℝ) (num_trees : ℕ) :
  total_length = 600 →
  num_trees = 26 →
  (total_length / (num_trees - 1 : ℝ)) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3827_382727


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3827_382711

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 1

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  Complex.re z > 0 ∧ Complex.im z < 0

-- State the theorem
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, z_condition z ∧ in_fourth_quadrant z := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3827_382711


namespace NUMINAMATH_CALUDE_jaylen_kristin_bell_pepper_ratio_l3827_382795

/-- Prove that the ratio of Jaylen's bell peppers to Kristin's bell peppers is 2:1 -/
theorem jaylen_kristin_bell_pepper_ratio :
  let jaylen_carrots : ℕ := 5
  let jaylen_cucumbers : ℕ := 2
  let kristin_bell_peppers : ℕ := 2
  let kristin_green_beans : ℕ := 20
  let jaylen_green_beans : ℕ := kristin_green_beans / 2 - 3
  let jaylen_total_vegetables : ℕ := 18
  let jaylen_bell_peppers : ℕ := jaylen_total_vegetables - (jaylen_carrots + jaylen_cucumbers + jaylen_green_beans)
  
  (jaylen_bell_peppers : ℚ) / kristin_bell_peppers = 2 := by
  sorry


end NUMINAMATH_CALUDE_jaylen_kristin_bell_pepper_ratio_l3827_382795


namespace NUMINAMATH_CALUDE_cost_of_stationery_l3827_382786

/-- Given the cost of erasers, pens, and markers satisfying certain conditions,
    prove that the total cost of 3 erasers, 4 pens, and 6 markers is 520 rubles. -/
theorem cost_of_stationery (E P M : ℕ) 
    (h1 : E + 3 * P + 2 * M = 240)
    (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := by
  sorry


end NUMINAMATH_CALUDE_cost_of_stationery_l3827_382786


namespace NUMINAMATH_CALUDE_min_value_z_l3827_382793

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 6 * x - 4 * y + 3 * x^3 + 15 ≥ 8.2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l3827_382793


namespace NUMINAMATH_CALUDE_unique_solution_l3827_382782

theorem unique_solution (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a*b - 9) : a = 3 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3827_382782


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3827_382707

theorem smallest_number_with_given_remainders : ∃ (b : ℕ), b > 0 ∧
  b % 4 = 2 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧
  ∀ (n : ℕ), n > 0 ∧ n % 4 = 2 ∧ n % 3 = 2 ∧ n % 5 = 3 → b ≤ n :=
by
  use 38
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3827_382707


namespace NUMINAMATH_CALUDE_circle_not_proportional_line_directly_proportional_hyperbola_inversely_proportional_line_through_origin_directly_proportional_another_line_through_origin_directly_proportional_l3827_382761

/-- A relation between x and y is directly proportional if it can be expressed as y = kx for some constant k ≠ 0 -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- A relation between x and y is inversely proportional if it can be expressed as xy = k for some constant k ≠ 0 -/
def InverselyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, x ≠ 0 → f x * x = k

/-- The main theorem stating that x^2 + y^2 = 16 is neither directly nor inversely proportional -/
theorem circle_not_proportional :
  ¬ (DirectlyProportional (fun x => Real.sqrt (16 - x^2)) ∨
     InverselyProportional (fun x => Real.sqrt (16 - x^2))) :=
sorry

/-- 2x + 3y = 6 describes y as directly proportional to x -/
theorem line_directly_proportional :
  DirectlyProportional (fun x => (6 - 2*x) / 3) ∨
  InverselyProportional (fun x => (6 - 2*x) / 3) :=
sorry

/-- xy = 5 describes y as inversely proportional to x -/
theorem hyperbola_inversely_proportional :
  DirectlyProportional (fun x => 5 / x) ∨
  InverselyProportional (fun x => 5 / x) :=
sorry

/-- x = 7y describes y as directly proportional to x -/
theorem line_through_origin_directly_proportional :
  DirectlyProportional (fun x => x / 7) ∨
  InverselyProportional (fun x => x / 7) :=
sorry

/-- x/y = 2 describes y as directly proportional to x -/
theorem another_line_through_origin_directly_proportional :
  DirectlyProportional (fun x => x / 2) ∨
  InverselyProportional (fun x => x / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_not_proportional_line_directly_proportional_hyperbola_inversely_proportional_line_through_origin_directly_proportional_another_line_through_origin_directly_proportional_l3827_382761


namespace NUMINAMATH_CALUDE_wall_height_proof_l3827_382754

/-- Proves that the height of each wall is 2 meters given the painting conditions --/
theorem wall_height_proof (num_walls : ℕ) (wall_width : ℝ) (paint_rate : ℝ) 
  (total_time : ℝ) (spare_time : ℝ) :
  num_walls = 5 →
  wall_width = 3 →
  paint_rate = 1 / 10 →
  total_time = 10 →
  spare_time = 5 →
  ∃ (wall_height : ℝ), 
    wall_height = 2 ∧ 
    (total_time - spare_time) * 60 * paint_rate = num_walls * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_wall_height_proof_l3827_382754


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l3827_382784

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property that the line joining the midpoints of the diagonals is half the difference of the bases -/
def midpoint_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

/-- The theorem to prove -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 120)
  (h2 : t.midpoint_segment = 7)
  (h3 : midpoint_property t) : 
  t.shorter_base = 106 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l3827_382784


namespace NUMINAMATH_CALUDE_triangle_count_segment_count_l3827_382758

/-- Represents a convex polygon divided into triangles -/
structure TriangulatedPolygon where
  p : ℕ  -- number of triangles
  n : ℕ  -- number of vertices on the boundary
  m : ℕ  -- number of vertices inside

/-- The number of triangles in a triangulated polygon satisfies p = n + 2m - 2 -/
theorem triangle_count (poly : TriangulatedPolygon) :
  poly.p = poly.n + 2 * poly.m - 2 := by sorry

/-- The number of segments that are sides of the resulting triangles is 2n + 3m - 3 -/
theorem segment_count (poly : TriangulatedPolygon) :
  2 * poly.n + 3 * poly.m - 3 = poly.p + poly.n + poly.m - 1 := by sorry

end NUMINAMATH_CALUDE_triangle_count_segment_count_l3827_382758


namespace NUMINAMATH_CALUDE_measure_15_minutes_with_7_and_11_l3827_382752

/-- Represents an hourglass that measures a specific duration. -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses. -/
structure MeasurementState where
  time_elapsed : ℕ
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Checks if it's possible to measure the target time with given hourglasses. -/
def can_measure_time (target : ℕ) (h1 h2 : Hourglass) : Prop :=
  ∃ (state : MeasurementState), state.time_elapsed = target ∧
    state.hourglass1 = h1 ∧ state.hourglass2 = h2

/-- Theorem stating that 15 minutes can be measured using 7-minute and 11-minute hourglasses. -/
theorem measure_15_minutes_with_7_and_11 :
  can_measure_time 15 (Hourglass.mk 7) (Hourglass.mk 11) := by
  sorry


end NUMINAMATH_CALUDE_measure_15_minutes_with_7_and_11_l3827_382752


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l3827_382773

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Theorem for part (1)
theorem range_of_x (a : ℝ) (h_a : a > 0) (h_a1 : a = 1) :
  (∃ x : ℝ, p x a ∨ q x) → (∃ x : ℝ, 1 < x ∧ x < 3) :=
sorry

-- Theorem for part (2)
theorem range_of_a :
  (∃ a : ℝ, (a > 0) ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) →
  (∃ a : ℝ, 1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l3827_382773


namespace NUMINAMATH_CALUDE_football_team_members_l3827_382775

theorem football_team_members :
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 5 = 1 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  n = 251 := by
sorry

end NUMINAMATH_CALUDE_football_team_members_l3827_382775


namespace NUMINAMATH_CALUDE_percentage_relationship_l3827_382746

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.4117647058823529)) :
  y = x * (1 + 0.7) := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l3827_382746


namespace NUMINAMATH_CALUDE_line_triangle_area_theorem_l3827_382748

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a line forms a triangle with the coordinate axes -/
def formsTriangle (l : Line) : Prop :=
  l.b ≠ 0 ∧ l.b / l.m < 0

/-- Calculates the area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : Line) : ℝ :=
  abs (l.b * (l.b / l.m)) / 2

/-- The main theorem -/
theorem line_triangle_area_theorem (k : ℝ) :
  let l : Line := { m := -2, b := k }
  formsTriangle l ∧ triangleArea l = 4 → k = 4 ∨ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_triangle_area_theorem_l3827_382748


namespace NUMINAMATH_CALUDE_mel_age_is_21_l3827_382756

/-- Katherine's age in years -/
def katherine_age : ℕ := 24

/-- The age difference between Katherine and Mel in years -/
def age_difference : ℕ := 3

/-- Mel's age in years -/
def mel_age : ℕ := katherine_age - age_difference

theorem mel_age_is_21 : mel_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_mel_age_is_21_l3827_382756


namespace NUMINAMATH_CALUDE_visit_neither_country_l3827_382766

theorem visit_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 60 → iceland = 35 → norway = 23 → both = 31 →
  total - (iceland + norway - both) = 33 := by
sorry

end NUMINAMATH_CALUDE_visit_neither_country_l3827_382766


namespace NUMINAMATH_CALUDE_existence_of_rationals_l3827_382794

theorem existence_of_rationals (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_rationals_l3827_382794


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3827_382776

theorem inequality_equivalence (x : ℝ) : 
  |((7-x)/4)| < 3 ↔ 2 < x ∧ x < 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3827_382776


namespace NUMINAMATH_CALUDE_quadratic_symmetry_and_value_l3827_382751

/-- A quadratic function with symmetry around x = 5.5 and p(0) = -4 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry_and_value (a b c : ℝ) :
  (∀ x, p a b c (5.5 - x) = p a b c (5.5 + x)) →  -- Symmetry around x = 5.5
  p a b c 0 = -4 →                                -- p(0) = -4
  p a b c 11 = -4 :=                              -- Conclusion: p(11) = -4
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_and_value_l3827_382751


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3827_382757

theorem proof_by_contradiction_assumption (a b : ℝ) : 
  (a ≤ b → False) → a > b :=
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3827_382757


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l3827_382780

theorem sugar_solution_replacement (W : ℝ) (x : ℝ) : 
  (W > 0) → 
  (0 ≤ x) → (x ≤ 1) →
  ((1 - x) * (0.22 * W) + x * (0.74 * W) = 0.35 * W) ↔ 
  (x = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_sugar_solution_replacement_l3827_382780


namespace NUMINAMATH_CALUDE_sphere_in_dihedral_angle_l3827_382790

/-- Given a sphere of unit radius with its center on the edge of a dihedral angle α,
    the radius r of a new sphere whose volume equals the volume of the part of the given sphere
    that lies inside the dihedral angle is r = ∛(α / (2π)). -/
theorem sphere_in_dihedral_angle (α : Real) (h : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (r : Real), r = (α / (2 * Real.pi)) ^ (1/3) ∧
  (4/3 * Real.pi * r^3) = (α / (2 * Real.pi)) * (4/3 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_dihedral_angle_l3827_382790


namespace NUMINAMATH_CALUDE_andrew_game_preparation_time_l3827_382760

/-- Represents the time in minutes required to prepare each type of game -/
structure GamePreparationTime where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the number of games of each type to be prepared -/
structure GameCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total preparation time for all games -/
def totalPreparationTime (prep : GamePreparationTime) (counts : GameCounts) : ℕ :=
  prep.typeA * counts.typeA + prep.typeB * counts.typeB + prep.typeC * counts.typeC

/-- Theorem: Given the specific game preparation times and counts, the total preparation time is 350 minutes -/
theorem andrew_game_preparation_time :
  let prep : GamePreparationTime := { typeA := 15, typeB := 25, typeC := 30 }
  let counts : GameCounts := { typeA := 5, typeB := 5, typeC := 5 }
  totalPreparationTime prep counts = 350 := by
  sorry

end NUMINAMATH_CALUDE_andrew_game_preparation_time_l3827_382760


namespace NUMINAMATH_CALUDE_binomial_20_choose_7_l3827_382715

theorem binomial_20_choose_7 : Nat.choose 20 7 = 5536 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_choose_7_l3827_382715


namespace NUMINAMATH_CALUDE_dinner_cakes_l3827_382755

def total_cakes : ℕ := 15
def lunch_cakes : ℕ := 6

theorem dinner_cakes : total_cakes - lunch_cakes = 9 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_l3827_382755


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3827_382718

/-- 
Given two points A and B that are symmetric with respect to the origin,
prove that the sum of their x and y coordinates is -2.
-/
theorem symmetric_points_sum (m n : ℝ) : 
  (3 : ℝ) = -(-m) → n = -(5 : ℝ) → m + n = -2 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3827_382718


namespace NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l3827_382736

/-- A positive three-digit palindrome is a natural number between 100 and 999 (inclusive) 
    that reads the same backwards as forwards. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

/-- The main theorem stating the existence of two positive three-digit palindromes 
    with the given product and sum. -/
theorem palindrome_product_sum_theorem : 
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧ 
                IsPositiveThreeDigitPalindrome b ∧ 
                a * b = 436995 ∧ 
                a + b = 1332 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l3827_382736


namespace NUMINAMATH_CALUDE_first_quarter_time_proportion_l3827_382735

/-- Represents the proportion of time spent traveling the first quarter of a distance
    when the speed for that quarter is 4 times the speed for the remaining distance -/
def time_proportion_first_quarter : ℚ := 1 / 13

/-- Proves that the proportion of time spent traveling the first quarter of the distance
    is 1/13 of the total time, given the specified speed conditions -/
theorem first_quarter_time_proportion 
  (D : ℝ) -- Total distance
  (V : ℝ) -- Speed for the remaining three-quarters of the distance
  (h1 : D > 0) -- Distance is positive
  (h2 : V > 0) -- Speed is positive
  : (D / (16 * V)) / ((D / (16 * V)) + (3 * D / (4 * V))) = time_proportion_first_quarter :=
sorry

end NUMINAMATH_CALUDE_first_quarter_time_proportion_l3827_382735


namespace NUMINAMATH_CALUDE_parabola_through_point_standard_form_l3827_382714

/-- A parabola is defined by its equation and the point it passes through. -/
structure Parabola where
  /-- The point that the parabola passes through -/
  point : ℝ × ℝ
  /-- The equation of the parabola, represented as a function -/
  equation : (ℝ × ℝ) → Prop

/-- The standard form of a parabola's equation -/
inductive StandardForm
  | VerticalAxis (p : ℝ) : StandardForm  -- y² = -2px
  | HorizontalAxis (p : ℝ) : StandardForm  -- x² = 2py

/-- Theorem: If a parabola passes through the point (-2, 3), then its standard equation
    must be either y² = -9/2x or x² = 4/3y -/
theorem parabola_through_point_standard_form (P : Parabola) 
    (h : P.point = (-2, 3)) :
    (∃ (sf : StandardForm), 
      (sf = StandardForm.VerticalAxis (9/4) ∨ 
       sf = StandardForm.HorizontalAxis (2/3)) ∧
      (∀ (x y : ℝ), P.equation (x, y) ↔ 
        (sf = StandardForm.VerticalAxis (9/4) → y^2 = -9/2 * x) ∧
        (sf = StandardForm.HorizontalAxis (2/3) → x^2 = 4/3 * y))) :=
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_standard_form_l3827_382714


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3827_382724

theorem condition_necessary_not_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3827_382724


namespace NUMINAMATH_CALUDE_binomial_coefficient_x6_in_expansion_1_plus_x_8_l3827_382777

theorem binomial_coefficient_x6_in_expansion_1_plus_x_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^(8 - k) * 1^k) = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x6_in_expansion_1_plus_x_8_l3827_382777


namespace NUMINAMATH_CALUDE_inserted_numbers_sequence_l3827_382799

theorem inserted_numbers_sequence (x y : ℝ) : 
  (2 < x ∧ x < y ∧ y < 20) ∧ 
  (x^2 = 2*y) ∧ 
  (2*y = x + 20) →
  (x + y = 4 ∨ x + y = 35/2) :=
by sorry

end NUMINAMATH_CALUDE_inserted_numbers_sequence_l3827_382799


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3827_382774

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3827_382774


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3827_382783

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let area_small : ℝ := π * r₁^2  -- area of smaller circle
  let area_large : ℝ := π * r₂^2  -- area of larger circle
  let area_between : ℝ := area_large - area_small  -- area between circles
  (area_between / area_small) = 8 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3827_382783


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l3827_382725

/-- The slope angle of the tangent line to y = x^3 forming an isosceles triangle -/
theorem tangent_slope_angle (x₀ : ℝ) : 
  let B : ℝ × ℝ := (x₀, x₀^3)
  let slope : ℝ := 3 * x₀^2
  let A : ℝ × ℝ := ((2/3) * x₀, 0)
  (x₀^4 = 1/3) →  -- This ensures OAB is isosceles
  (slope = Real.sqrt 3) →
  Real.arctan slope = π/3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l3827_382725


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3827_382787

theorem greatest_integer_b_for_quadratic_range : 
  ∃ (b : ℤ), b = 9 ∧ 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -4) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + c*x + 20 = -4) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3827_382787


namespace NUMINAMATH_CALUDE_fifth_number_correct_l3827_382726

/-- The function that generates the 5th number on the n-th row of the array -/
def fifthNumber (n : ℕ) : ℚ :=
  (n - 1) * (n - 2) * (n - 3) * (3 * n + 8) / 24

/-- The theorem stating that for n > 5, the 5th number on the n-th row
    of the given array is equal to (n-1)(n-2)(n-3)(3n + 8) / 24 -/
theorem fifth_number_correct (n : ℕ) (h : n > 5) :
  fifthNumber n = (n - 1) * (n - 2) * (n - 3) * (3 * n + 8) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_correct_l3827_382726


namespace NUMINAMATH_CALUDE_trebled_result_l3827_382701

theorem trebled_result (x : ℕ) (h : x = 7) : 3 * ((2 * x) + 9) = 69 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_l3827_382701


namespace NUMINAMATH_CALUDE_no_double_application_increment_l3827_382769

theorem no_double_application_increment (f : ℕ → ℕ) : ∃ n : ℕ, n > 0 ∧ f (f n) ≠ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_increment_l3827_382769


namespace NUMINAMATH_CALUDE_transistors_2010_l3827_382791

/-- Moore's law: Number of transistors doubles every 18 months -/
def moores_law_doubling_period : ℕ := 18

/-- Number of transistors in 1995 -/
def transistors_1995 : ℕ := 2500000

/-- Calculate the number of transistors after a given number of months -/
def transistors_after (initial_transistors : ℕ) (months : ℕ) : ℕ :=
  initial_transistors * 2^(months / moores_law_doubling_period)

/-- Theorem: Number of transistors in 2010 according to Moore's law -/
theorem transistors_2010 :
  transistors_after transistors_1995 ((2010 - 1995) * 12) = 2560000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_2010_l3827_382791


namespace NUMINAMATH_CALUDE_intersection_perpendicular_bisector_l3827_382744

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem intersection_perpendicular_bisector :
  ∀ A B : ℝ × ℝ,
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B →
  ∀ x y : ℝ,
  perp_bisector x y ↔
  (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_bisector_l3827_382744


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3827_382713

theorem complex_magnitude_equation (x : ℝ) : 
  (x > 0 ∧ Complex.abs (-3 + x * Complex.I) = 5 * Real.sqrt 5) → x = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3827_382713


namespace NUMINAMATH_CALUDE_min_intersection_cardinality_l3827_382745

-- Define the cardinality function
def card (S : Set α) : ℕ := sorry

-- Define the power set function
def powerset (S : Set α) : Set (Set α) := sorry

theorem min_intersection_cardinality 
  (A B C D : Set α) 
  (h1 : card A = 150) 
  (h2 : card B = 150) 
  (h3 : card D = 102) 
  (h4 : card (powerset A) + card (powerset B) + card (powerset C) + card (powerset D) = 
        card (powerset (A ∪ B ∪ C ∪ D)))
  (h5 : card (powerset A) + card (powerset B) + card (powerset C) + card (powerset D) = 2^152) :
  card (A ∩ B ∩ C ∩ D) ≥ 99 ∧ ∃ (A' B' C' D' : Set α), 
    card A' = 150 ∧ 
    card B' = 150 ∧ 
    card D' = 102 ∧ 
    card (powerset A') + card (powerset B') + card (powerset C') + card (powerset D') = 
      card (powerset (A' ∪ B' ∪ C' ∪ D')) ∧
    card (powerset A') + card (powerset B') + card (powerset C') + card (powerset D') = 2^152 ∧
    card (A' ∩ B' ∩ C' ∩ D') = 99 :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_cardinality_l3827_382745


namespace NUMINAMATH_CALUDE_plum_jelly_sales_l3827_382778

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the conditions for jelly sales -/
def validJellySales (sales : JellySales) : Prop :=
  sales.grape = 2 * sales.strawberry ∧
  sales.raspberry = 2 * sales.plum ∧
  sales.raspberry = sales.grape / 3 ∧
  sales.strawberry = 18

/-- Theorem stating that given the conditions, 6 jars of plum jelly were sold -/
theorem plum_jelly_sales (sales : JellySales) (h : validJellySales sales) : sales.plum = 6 := by
  sorry

end NUMINAMATH_CALUDE_plum_jelly_sales_l3827_382778


namespace NUMINAMATH_CALUDE_G_1000_units_digit_l3827_382798

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 3^(3^n) + 2

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem G_1000_units_digit :
  unitsDigit (G 1000) = 5 := by
  sorry

end NUMINAMATH_CALUDE_G_1000_units_digit_l3827_382798


namespace NUMINAMATH_CALUDE_intersection_M_N_l3827_382772

-- Define the sets M and N
def M : Set ℝ := {x | 1 + x ≥ 0}
def N : Set ℝ := {x | (4 : ℝ) / (1 - x) > 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3827_382772


namespace NUMINAMATH_CALUDE_sameTotalHeadsProbability_eq_565_2048_l3827_382706

/-- Represents the probability distribution of flipping four coins, 
    where three are fair and one has 5/8 probability of heads -/
def coinFlipDistribution : List ℚ :=
  [3/64, 14/64, 24/64, 18/64, 5/64]

/-- The probability of two people getting the same number of heads 
    when each flips four coins (three fair, one biased) -/
def sameTotalHeadsProbability : ℚ :=
  (coinFlipDistribution.map (λ x => x^2)).sum

theorem sameTotalHeadsProbability_eq_565_2048 :
  sameTotalHeadsProbability = 565/2048 := by
  sorry

end NUMINAMATH_CALUDE_sameTotalHeadsProbability_eq_565_2048_l3827_382706


namespace NUMINAMATH_CALUDE_pipeline_theorem_l3827_382732

/-- Represents the pipeline construction problem -/
structure PipelineConstruction where
  total_length : ℝ
  daily_increase : ℝ
  days_ahead : ℝ
  actual_daily_length : ℝ

/-- The equation describing the pipeline construction problem -/
def pipeline_equation (p : PipelineConstruction) : Prop :=
  p.total_length / (p.actual_daily_length - p.daily_increase) -
  p.total_length / p.actual_daily_length = p.days_ahead

/-- Theorem stating that the equation holds for the given parameters -/
theorem pipeline_theorem (p : PipelineConstruction)
  (h1 : p.total_length = 4000)
  (h2 : p.daily_increase = 10)
  (h3 : p.days_ahead = 20) :
  pipeline_equation p :=
sorry

end NUMINAMATH_CALUDE_pipeline_theorem_l3827_382732


namespace NUMINAMATH_CALUDE_minimize_segment_expression_l3827_382788

/-- Given a line segment AB of length a, the point C that minimizes AC^2 + 3CB^2 is at 3a/4 from A -/
theorem minimize_segment_expression (a : ℝ) (h : a > 0) :
  ∃ c : ℝ, c = 3*a/4 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ a → 
      x^2 + 3*(a-x)^2 ≥ c^2 + 3*(a-c)^2 :=
by sorry


end NUMINAMATH_CALUDE_minimize_segment_expression_l3827_382788


namespace NUMINAMATH_CALUDE_cake_slices_l3827_382739

/-- The cost of ingredients and number of slices eaten by Laura's mother and the dog --/
structure CakeData where
  flour_cost : ℝ
  sugar_cost : ℝ
  butter_cost : ℝ
  eggs_cost : ℝ
  mother_slices : ℕ
  dog_cost : ℝ

/-- The total number of slices Laura cut the cake into --/
def total_slices (data : CakeData) : ℕ :=
  sorry

/-- Theorem stating that the total number of slices is 6 --/
theorem cake_slices (data : CakeData) 
  (h1 : data.flour_cost = 4)
  (h2 : data.sugar_cost = 2)
  (h3 : data.butter_cost = 2.5)
  (h4 : data.eggs_cost = 0.5)
  (h5 : data.mother_slices = 2)
  (h6 : data.dog_cost = 6) :
  total_slices data = 6 :=
sorry

end NUMINAMATH_CALUDE_cake_slices_l3827_382739


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3827_382709

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 7 = 45 →                     -- first given condition
  a 2 + a 5 + a 8 = 29 →                     -- second given condition
  a 3 + a 6 + a 9 = 13 :=                    -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3827_382709


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3827_382717

theorem quadratic_transformation (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*(m+1)*x + 16 = (x-4)^2) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3827_382717


namespace NUMINAMATH_CALUDE_log_comparison_l3827_382720

theorem log_comparison : 
  Real.log 6 / Real.log 3 > Real.log 10 / Real.log 5 ∧ 
  Real.log 10 / Real.log 5 > Real.log 14 / Real.log 7 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l3827_382720


namespace NUMINAMATH_CALUDE_correct_average_after_errors_l3827_382792

theorem correct_average_after_errors (n : ℕ) (initial_avg : ℚ) 
  (error1 : ℚ) (error2 : ℚ) (error3 : ℚ) : 
  n = 15 → 
  initial_avg = 24 → 
  error1 = 65 - 45 → 
  error2 = 42 - 28 → 
  error3 = 75 - 55 → 
  (n : ℚ) * initial_avg + error1 + error2 + error3 = n * (27.6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_errors_l3827_382792


namespace NUMINAMATH_CALUDE_expense_representation_l3827_382703

-- Define income and expense as real numbers
def income : ℝ := 5
def expense : ℝ := 5

-- Define the representation of income
def income_representation : ℝ := 5

-- Theorem to prove
theorem expense_representation : 
  income_representation = income → -expense = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_expense_representation_l3827_382703


namespace NUMINAMATH_CALUDE_john_total_spent_l3827_382768

/-- Calculates the total amount spent by John in USD -/
def total_spent (umbrella_price : ℝ) (raincoat_price : ℝ) (bag_price : ℝ)
                (umbrella_count : ℕ) (raincoat_count : ℕ) (bag_count : ℕ)
                (umbrella_raincoat_discount : ℝ) (bag_discount : ℝ)
                (refund_percentage : ℝ) (initial_conversion_rate : ℝ)
                (refund_conversion_rate : ℝ) : ℝ :=
  sorry

theorem john_total_spent :
  total_spent 8 15 25 2 3 1 0.1 0.05 0.8 1.15 1.17 = 77.81 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spent_l3827_382768


namespace NUMINAMATH_CALUDE_cat_walking_distance_l3827_382749

/-- The distance a cat walks given resistance time, walking rate, and total time -/
theorem cat_walking_distance (resistance_time walking_rate total_time : ℕ) : 
  resistance_time = 20 →
  walking_rate = 8 →
  total_time = 28 →
  (total_time - resistance_time) * walking_rate = 64 := by
  sorry

end NUMINAMATH_CALUDE_cat_walking_distance_l3827_382749


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3827_382771

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3827_382771


namespace NUMINAMATH_CALUDE_find_decrease_rate_village_x_decrease_rate_l3827_382789

/-- Represents the population change in two villages over time -/
def village_population_equality (x_initial : ℕ) (y_initial : ℕ) (y_growth_rate : ℕ) (years : ℕ) (x_decrease_rate : ℕ) : Prop :=
  x_initial - years * x_decrease_rate = y_initial + years * y_growth_rate

/-- Theorem stating the condition for equal populations after a given time -/
theorem find_decrease_rate (x_initial y_initial y_growth_rate years : ℕ) :
  ∃ (x_decrease_rate : ℕ),
    village_population_equality x_initial y_initial y_growth_rate years x_decrease_rate ∧
    x_decrease_rate = (x_initial - y_initial - years * y_growth_rate) / years :=
by
  sorry

/-- Application of the theorem to the specific problem -/
theorem village_x_decrease_rate :
  ∃ (x_decrease_rate : ℕ),
    village_population_equality 76000 42000 800 17 x_decrease_rate ∧
    x_decrease_rate = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_find_decrease_rate_village_x_decrease_rate_l3827_382789


namespace NUMINAMATH_CALUDE_circle_properties_l3827_382797

/-- The equation of a circle in the form ax² + by² + cx + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

def circle_equation : CircleEquation :=
  { a := 1, b := 1, c := -2, d := 4, e := 3 }

theorem circle_properties :
  ∃ (props : CircleProperties),
    props.center = (1, -2) ∧ 
    props.radius = Real.sqrt 2 ∧
    ∀ (x y : ℝ),
      (circle_equation.a * x^2 + circle_equation.b * y^2 + 
       circle_equation.c * x + circle_equation.d * y + 
       circle_equation.e = 0) ↔
      ((x - props.center.1)^2 + (y - props.center.2)^2 = props.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3827_382797


namespace NUMINAMATH_CALUDE_minor_arc_probability_l3827_382779

/-- The probability that the length of the minor arc is less than 1 on a circle
    with circumference 3, given a fixed point A and a randomly selected point B. -/
theorem minor_arc_probability (circle_circumference : ℝ) (arc_length : ℝ) :
  circle_circumference = 3 →
  arc_length = 1 →
  (2 * arc_length) / circle_circumference = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_minor_arc_probability_l3827_382779


namespace NUMINAMATH_CALUDE_fraction_problem_l3827_382731

theorem fraction_problem (x : ℝ) (h : (5 / 9) * x = 60) : (1 / 4) * x = 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3827_382731


namespace NUMINAMATH_CALUDE_c_condition_l3827_382770

theorem c_condition (a b c : ℝ) (h1 : a < b) (h2 : a * c > b * c) : c < 0 := by
  sorry

end NUMINAMATH_CALUDE_c_condition_l3827_382770


namespace NUMINAMATH_CALUDE_inequality_proof_l3827_382719

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 3 * Real.sqrt 2) ∧
  (2 * (a^3 + b^3 + c^3) ≥ a*b + b*c + c*a - 3*a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3827_382719


namespace NUMINAMATH_CALUDE_meal_with_tip_l3827_382733

/-- Calculates the total amount spent on a meal including tip -/
theorem meal_with_tip (lunch_cost : ℝ) (tip_percentage : ℝ) : 
  lunch_cost = 50.50 → tip_percentage = 20 → lunch_cost * (1 + tip_percentage / 100) = 60.60 := by
  sorry

end NUMINAMATH_CALUDE_meal_with_tip_l3827_382733


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l3827_382747

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (70/31, 135/31)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8*x - 3*y = 5

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 5*x + 2*y = 20

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l3827_382747


namespace NUMINAMATH_CALUDE_solution_pairs_l3827_382743

/-- Sum of factorials from 1 to k -/
def sumFactorials (k : ℕ) : ℕ :=
  (List.range k).map Nat.factorial |>.sum

/-- Sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The set of pairs (k, n) that satisfy the equation -/
def solutionSet : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ sumFactorials p.1 = sumIntegers p.2}

theorem solution_pairs : solutionSet = {(1, 1), (2, 2), (5, 17)} := by
  sorry


end NUMINAMATH_CALUDE_solution_pairs_l3827_382743


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3827_382704

-- Define the function f(x)
def f (x : ℝ) : ℝ := 6 - 12*x + x^3

-- Define the interval
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem max_min_values_of_f :
  (∃ x ∈ interval, f x = 22 ∧ ∀ y ∈ interval, f y ≤ 22) ∧
  (∃ x ∈ interval, f x = -5 ∧ ∀ y ∈ interval, f y ≥ -5) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3827_382704


namespace NUMINAMATH_CALUDE_interest_calculation_l3827_382764

theorem interest_calculation (initial_investment second_investment second_interest : ℝ) 
  (h1 : initial_investment = 5000)
  (h2 : second_investment = 20000)
  (h3 : second_interest = 1000)
  (h4 : second_interest = second_investment * (second_interest / second_investment))
  (h5 : initial_investment > 0)
  (h6 : second_investment > 0) :
  initial_investment * (second_interest / second_investment) = 250 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l3827_382764


namespace NUMINAMATH_CALUDE_xy_sum_l3827_382702

theorem xy_sum (x y : ℕ) : 
  0 < x ∧ x < 20 ∧ 0 < y ∧ y < 20 ∧ x + y + x * y = 95 → x + y = 18 ∨ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l3827_382702


namespace NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l3827_382741

/-- Defines a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- Defines the condition b^2 = ac -/
def condition_b_squared_eq_ac (a b c : ℝ) : Prop :=
  b^2 = a * c

/-- Theorem stating that "a, b, c form a geometric sequence" is sufficient 
    but not necessary for "b^2 = ac" -/
theorem geometric_sequence_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_sequence a b c → condition_b_squared_eq_ac a b c) ∧
  ¬(∀ a b c : ℝ, condition_b_squared_eq_ac a b c → is_geometric_sequence a b c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l3827_382741


namespace NUMINAMATH_CALUDE_dog_escape_ways_l3827_382781

def base_7_to_10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₀ * 7^0 + d₁ * 7^1 + d₂ * 7^2

theorem dog_escape_ways : base_7_to_10 2 3 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_escape_ways_l3827_382781


namespace NUMINAMATH_CALUDE_flatbread_division_l3827_382734

-- Define a planar region
def PlanarRegion : Type := Set (ℝ × ℝ)

-- Define the area of a planar region
noncomputable def area (R : PlanarRegion) : ℝ := sorry

-- Define a line in 2D space
def Line : Type := Set (ℝ × ℝ)

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the division of a planar region by two lines
def divide (R : PlanarRegion) (l1 l2 : Line) : List PlanarRegion := sorry

-- Theorem statement
theorem flatbread_division (R : PlanarRegion) (P : ℝ) (h : area R = P) :
  ∃ (l1 l2 : Line), perpendicular l1 l2 ∧ 
    ∀ (part : PlanarRegion), part ∈ divide R l1 l2 → area part = P / 4 := by
  sorry

end NUMINAMATH_CALUDE_flatbread_division_l3827_382734


namespace NUMINAMATH_CALUDE_min_trucks_required_l3827_382740

/-- Represents the weight capacity of a truck in tons -/
def truck_capacity : ℝ := 3

/-- Represents the total weight of all boxes in tons -/
def total_weight : ℝ := 10

/-- Represents the maximum weight of a single box in tons -/
def max_box_weight : ℝ := 1

/-- The minimum number of trucks required -/
def min_trucks : ℕ := 5

/-- Theorem stating that the minimum number of trucks required is 5 -/
theorem min_trucks_required :
  ∀ (box_weights : List ℝ),
    (box_weights.sum = total_weight) →
    (∀ w ∈ box_weights, w ≤ max_box_weight) →
    (∀ n : ℕ, n < min_trucks → n * truck_capacity < total_weight) →
    (min_trucks * truck_capacity ≥ total_weight) :=
by sorry

end NUMINAMATH_CALUDE_min_trucks_required_l3827_382740


namespace NUMINAMATH_CALUDE_pizza_coworkers_l3827_382708

theorem pizza_coworkers (num_pizzas : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) :
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  slices_per_person = 2 →
  (num_pizzas * slices_per_pizza) / slices_per_person = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coworkers_l3827_382708


namespace NUMINAMATH_CALUDE_discount_savings_l3827_382712

/-- Given a purchase with a 10% discount, calculate the amount saved -/
theorem discount_savings (purchase_price : ℝ) (discount_rate : ℝ) (savings : ℝ) : 
  purchase_price = 100 →
  discount_rate = 0.1 →
  savings = purchase_price * discount_rate →
  savings = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_savings_l3827_382712


namespace NUMINAMATH_CALUDE_julia_tag_difference_l3827_382759

theorem julia_tag_difference (x y : ℕ) (hx : x = 45) (hy : y = 28) : x - y = 17 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_difference_l3827_382759


namespace NUMINAMATH_CALUDE_article_cost_l3827_382750

theorem article_cost (cost : ℝ) (selling_price : ℝ) : 
  selling_price = 1.25 * cost →
  (0.8 * cost + 0.3 * (0.8 * cost) = selling_price - 8.4) →
  cost = 40 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l3827_382750


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3827_382700

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (1, 2) (x, -2) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3827_382700


namespace NUMINAMATH_CALUDE_wall_cleaning_time_l3827_382729

/-- Represents the cleaning rate in minutes per section -/
def cleaning_rate (time_spent : ℕ) (sections_cleaned : ℕ) : ℚ :=
  (time_spent : ℚ) / sections_cleaned

/-- Calculates the remaining time to clean the wall -/
def remaining_time (total_sections : ℕ) (cleaned_sections : ℕ) (rate : ℚ) : ℚ :=
  ((total_sections - cleaned_sections) : ℚ) * rate

/-- Theorem stating the remaining time to clean the wall -/
theorem wall_cleaning_time (total_sections : ℕ) (cleaned_sections : ℕ) (time_spent : ℕ) :
  total_sections = 18 ∧ cleaned_sections = 3 ∧ time_spent = 33 →
  remaining_time total_sections cleaned_sections (cleaning_rate time_spent cleaned_sections) = 165 := by
  sorry

end NUMINAMATH_CALUDE_wall_cleaning_time_l3827_382729


namespace NUMINAMATH_CALUDE_cannot_determine_f_triple_prime_l3827_382763

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- State the theorem
theorem cannot_determine_f_triple_prime (a b c : ℝ) :
  (∃ x, f a b c x = a * x^4 + b * x^2 + c) →
  ((12 * a + 2 * b) = 2) →
  ¬ (∃! y, (24 * a * (-1) = y)) :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_f_triple_prime_l3827_382763


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l3827_382767

/-- Pizza sharing problem -/
theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_cost : ℚ := 12
  let mushroom_cost : ℚ := 3
  let pepperoni_cost : ℚ := 5
  let bob_slices : ℕ := 8
  let anne_slices : ℕ := 3
  let total_cost : ℚ := plain_cost + mushroom_cost + pepperoni_cost
  let bob_cost : ℚ := total_cost - (anne_slices : ℚ) * (plain_cost / total_slices)
  let anne_cost : ℚ := (anne_slices : ℚ) * (plain_cost / total_slices)
  bob_cost - anne_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l3827_382767


namespace NUMINAMATH_CALUDE_second_number_is_90_l3827_382730

theorem second_number_is_90 (a b c : ℚ) : 
  a + b + c = 330 → 
  a = 2 * b → 
  c = (1/3) * a → 
  b = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_90_l3827_382730


namespace NUMINAMATH_CALUDE_exists_distinct_subaveraging_equal_entries_imply_infinite_equal_pairs_l3827_382796

-- Define a subaveraging sequence
def IsSubaveraging (s : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, s n = (s (n - 1) + s (n + 1)) / 4

-- Part (a): Existence of a subaveraging sequence with all distinct entries
theorem exists_distinct_subaveraging :
  ∃ s : ℤ → ℝ, IsSubaveraging s ∧ (∀ m n : ℤ, m ≠ n → s m ≠ s n) :=
sorry

-- Part (b): If two entries are equal, infinitely many pairs are equal
theorem equal_entries_imply_infinite_equal_pairs
  (s : ℤ → ℝ) (h : IsSubaveraging s) :
  (∃ m n : ℤ, m ≠ n ∧ s m = s n) →
  (∀ k : ℕ, ∃ i j : ℤ, i ≠ j ∧ s i = s j ∧ |i - j| > k) :=
sorry

end NUMINAMATH_CALUDE_exists_distinct_subaveraging_equal_entries_imply_infinite_equal_pairs_l3827_382796


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3827_382723

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2 * x - 1|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem range_of_a :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≤ |2 * x + 1|) → a ∈ Set.Icc (-1 : ℝ) (5/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3827_382723


namespace NUMINAMATH_CALUDE_comic_book_frames_l3827_382737

theorem comic_book_frames (frames_per_page : ℝ) (pages : ℝ) 
  (h1 : frames_per_page = 143.0) 
  (h2 : pages = 11.0) : 
  frames_per_page * pages = 1573.0 := by
sorry

end NUMINAMATH_CALUDE_comic_book_frames_l3827_382737


namespace NUMINAMATH_CALUDE_part_one_part_two_range_of_m_l3827_382742

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
sorry

-- Part 2
theorem part_two :
  ∀ x : ℝ, f 1 x + f 1 (x + 5) ≥ 5 :=
sorry

-- Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f 1 x + f 1 (x + 5) ≥ m) ↔ m ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_range_of_m_l3827_382742


namespace NUMINAMATH_CALUDE_smaller_integer_proof_l3827_382738

theorem smaller_integer_proof (x y : ℤ) : 
  y = 5 * x + 2 →  -- One integer is 2 more than 5 times the other
  y - x = 26 →     -- The difference between the two integers is 26
  x = 6            -- The smaller integer is 6
:= by sorry

end NUMINAMATH_CALUDE_smaller_integer_proof_l3827_382738


namespace NUMINAMATH_CALUDE_power_two_99_mod_7_l3827_382705

theorem power_two_99_mod_7 : 2^99 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_99_mod_7_l3827_382705


namespace NUMINAMATH_CALUDE_probability_six_even_numbers_l3827_382716

def integers_range : Set ℤ := {x | -9 ≤ x ∧ x ≤ 9}

def even_numbers (S : Set ℤ) : Set ℤ := {x ∈ S | x % 2 = 0}

def total_count : ℕ := Finset.card (Finset.range 19)

def even_count (S : Set ℤ) : ℕ := Finset.card (Finset.filter (λ x => x % 2 = 0) (Finset.range 19))

theorem probability_six_even_numbers :
  let S := integers_range
  let n := total_count
  let k := even_count S
  (k.choose 6 : ℚ) / (n.choose 6 : ℚ) = 1 / 76 := by sorry

end NUMINAMATH_CALUDE_probability_six_even_numbers_l3827_382716


namespace NUMINAMATH_CALUDE_banana_permutations_l3827_382710

/-- The number of distinct permutations of a sequence with repeated elements -/
def distinctPermutations (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations :
  distinctPermutations 6 [3, 2, 1] = 60 := by
  sorry

#eval distinctPermutations 6 [3, 2, 1]

end NUMINAMATH_CALUDE_banana_permutations_l3827_382710


namespace NUMINAMATH_CALUDE_expected_socks_removed_theorem_l3827_382785

/-- The expected number of socks removed to get both favorite socks -/
def expected_socks_removed (n : ℕ) : ℚ :=
  2 * (n + 1) / 3

/-- Theorem stating the expected number of socks removed to get both favorite socks -/
theorem expected_socks_removed_theorem (n : ℕ) (h : n ≥ 2) :
  expected_socks_removed n = 2 * (n + 1) / 3 := by
  sorry

#check expected_socks_removed_theorem

end NUMINAMATH_CALUDE_expected_socks_removed_theorem_l3827_382785


namespace NUMINAMATH_CALUDE_nancy_total_games_l3827_382722

/-- The total number of football games Nancy would attend over three months -/
def total_games (this_month next_month last_month : ℕ) : ℕ :=
  this_month + next_month + last_month

/-- Theorem: Nancy would attend 24 games in total -/
theorem nancy_total_games : 
  total_games 9 7 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l3827_382722


namespace NUMINAMATH_CALUDE_min_abs_z_l3827_382762

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*I) + Complex.abs (z - 2) = 10) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 10 / Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l3827_382762


namespace NUMINAMATH_CALUDE_square_diagonal_side_perimeter_l3827_382728

theorem square_diagonal_side_perimeter :
  ∀ (d s p : ℝ),
  d = 2 * Real.sqrt 2 →  -- diagonal is 2√2 inches
  d = s * Real.sqrt 2 →  -- relation between diagonal and side in a square
  s = 2 ∧                -- side length is 2 inches
  p = 4 * s              -- perimeter is 4 times the side length
  := by sorry

end NUMINAMATH_CALUDE_square_diagonal_side_perimeter_l3827_382728


namespace NUMINAMATH_CALUDE_trapezoid_xy_relation_l3827_382721

-- Define the trapezoid and its properties
structure Trapezoid where
  x : ℝ
  y : ℝ
  h : ℝ
  AC : ℝ
  BD : ℝ
  AB : ℝ
  CD : ℝ
  h_def : h = 5 * x * y
  area_relation : (1/2) * AC * BD = (15*Real.sqrt 3)/(36) * AB * CD
  xy_constraint : x^2 + y^2 = 1

-- State the theorem
theorem trapezoid_xy_relation (t : Trapezoid) : 5 * t.x * t.y = 4 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_xy_relation_l3827_382721
