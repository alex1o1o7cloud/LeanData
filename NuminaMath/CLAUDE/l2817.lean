import Mathlib

namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2817_281766

/-- Two vectors a and b in ℝ² are collinear if there exists a scalar k such that b = k * a -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

/-- Given two vectors a = (1, -2) and b = (-2, x) in ℝ², 
    if a and b are collinear, then x = 4 -/
theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-2, x)
  collinear a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2817_281766


namespace NUMINAMATH_CALUDE_book_ratio_problem_l2817_281714

theorem book_ratio_problem (lit sci : ℕ) (h : lit * 5 = sci * 8) : 
  (lit - sci : ℚ) / sci = 3 / 5 ∧ (lit - sci : ℚ) / lit = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_problem_l2817_281714


namespace NUMINAMATH_CALUDE_calculate_expression_solve_inequalities_l2817_281792

-- Part 1
theorem calculate_expression : 
  |3 - Real.pi| - (-2)⁻¹ + 4 * Real.cos (60 * π / 180) = Real.pi - 1/2 := by sorry

-- Part 2
theorem solve_inequalities (x : ℝ) : 
  (5*x - 1 > 3*(x + 1) ∧ 1 + 2*x ≥ x - 1) ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_inequalities_l2817_281792


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2817_281759

/-- Given a rectangle ABCD and a square JKLM, if the rectangle shares 40% of its area with the square,
    and the square shares 25% of its area with the rectangle, then the ratio of the length (AB) to 
    the width (AD) of the rectangle is 15.625. -/
theorem rectangle_square_overlap_ratio : 
  ∀ (AB AD s : ℝ), 
  AB > 0 → AD > 0 → s > 0 →
  0.4 * AB * AD = 0.25 * s^2 →
  AB / AD = 15.625 := by
    sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2817_281759


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2817_281761

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  side : ℝ

/-- Defines the locus equation for a point P relative to an isosceles triangle -/
def locusEquation (P : Point) (triangle : IsoscelesTriangle) (k : ℝ) : Prop :=
  3 * P.x^2 + 2 * P.y^2 - 2 * triangle.height * P.y + triangle.height^2 + 
  2 * (triangle.base / 2)^2 = k * triangle.side^2

/-- States that the locus of points satisfying the equation forms an ellipse -/
theorem locus_is_ellipse (triangle : IsoscelesTriangle) (k : ℝ) 
    (h_k : k > 1) (h_side : triangle.side^2 = (triangle.base / 2)^2 + triangle.height^2) :
  ∃ (center : Point) (a b : ℝ), ∀ (P : Point),
    locusEquation P triangle k ↔ 
    (P.x - center.x)^2 / a^2 + (P.y - center.y)^2 / b^2 = 1 :=
  sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l2817_281761


namespace NUMINAMATH_CALUDE_cube_cross_section_theorem_l2817_281778

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a polygon in 3D space -/
structure Polygon where
  vertices : List Point3D

def isPerpendicularTo (p : Plane) (v : Point3D) : Prop := sorry

def intersectsAllFaces (p : Plane) (c : Cube) : Prop := sorry

def crossSectionPolygon (p : Plane) (c : Cube) : Polygon := sorry

def perimeter (poly : Polygon) : ℝ := sorry

def area (poly : Polygon) : ℝ := sorry

theorem cube_cross_section_theorem (c : Cube) (p : Plane) (ac' : Point3D) :
  isPerpendicularTo p ac' →
  intersectsAllFaces p c →
  (∃ l : ℝ, ∀ α : Plane, isPerpendicularTo α ac' → intersectsAllFaces α c →
    perimeter (crossSectionPolygon α c) = l) ∧
  (¬∃ s : ℝ, ∀ α : Plane, isPerpendicularTo α ac' → intersectsAllFaces α c →
    area (crossSectionPolygon α c) = s) := by
  sorry

end NUMINAMATH_CALUDE_cube_cross_section_theorem_l2817_281778


namespace NUMINAMATH_CALUDE_smallest_coin_collection_l2817_281777

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def proper_factors (n : ℕ) : Finset ℕ :=
  (Nat.divisors n).filter (λ x => x > 1 ∧ x < n)

theorem smallest_coin_collection :
  ∃ (n : ℕ), n > 0 ∧ num_factors n = 13 ∧ (proper_factors n).card ≥ 11 ∧
  ∀ (m : ℕ), m > 0 → num_factors m = 13 → (proper_factors m).card ≥ 11 → n ≤ m :=
by
  use 4096
  sorry

end NUMINAMATH_CALUDE_smallest_coin_collection_l2817_281777


namespace NUMINAMATH_CALUDE_problem_solution_l2817_281735

theorem problem_solution (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (3 * a + a * b - b) + 4 * a / b = 64) :
  a = 8 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2817_281735


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2817_281755

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Vector addition -/
def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → vec_add a b = (6, 3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2817_281755


namespace NUMINAMATH_CALUDE_sticks_form_triangle_l2817_281768

-- Define the lengths of the sticks
def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := 5

-- Define the triangle inequality theorem
def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Theorem statement
theorem sticks_form_triangle : triangle_inequality a b c := by
  sorry

end NUMINAMATH_CALUDE_sticks_form_triangle_l2817_281768


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2817_281779

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

theorem base_conversion_subtraction :
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_6_number := [6, 5, 1]  -- 156 in base 6 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 193 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2817_281779


namespace NUMINAMATH_CALUDE_trip_cost_calculation_l2817_281742

theorem trip_cost_calculation (original_price discount : ℕ) (num_people : ℕ) : 
  original_price = 147 → 
  discount = 14 → 
  num_people = 2 → 
  (original_price - discount) * num_people = 266 := by
sorry

end NUMINAMATH_CALUDE_trip_cost_calculation_l2817_281742


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2817_281726

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 300 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 7 + a 8 + a 9 = 300

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  a 2 + a 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2817_281726


namespace NUMINAMATH_CALUDE_sandy_final_position_l2817_281701

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define Sandy's walk
def sandy_walk (start : Point) : Point :=
  let p1 : Point := ⟨start.x, start.y - 20⟩  -- 20 meters south
  let p2 : Point := ⟨p1.x + 20, p1.y⟩        -- 20 meters east
  let p3 : Point := ⟨p2.x, p2.y + 20⟩        -- 20 meters north
  let p4 : Point := ⟨p3.x + 10, p3.y⟩        -- 10 meters east
  p4

-- Theorem stating that Sandy ends up 10 meters east of her starting point
theorem sandy_final_position (start : Point) : 
  (sandy_walk start).x - start.x = 10 ∧ (sandy_walk start).y = start.y :=
by sorry

end NUMINAMATH_CALUDE_sandy_final_position_l2817_281701


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2817_281794

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 1024 ways to put 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes :
  distribute_balls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2817_281794


namespace NUMINAMATH_CALUDE_sarah_connor_wage_ratio_l2817_281725

def connors_hourly_wage : ℝ := 7.20
def sarahs_daily_wage : ℝ := 288
def work_hours : ℕ := 8

theorem sarah_connor_wage_ratio :
  (sarahs_daily_wage / work_hours) / connors_hourly_wage = 5 := by sorry

end NUMINAMATH_CALUDE_sarah_connor_wage_ratio_l2817_281725


namespace NUMINAMATH_CALUDE_largest_k_for_positive_root_l2817_281720

/-- The equation has at least one positive root -/
def has_positive_root (k : ℤ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ 3 * x * (2 * k * x - 5) - 2 * x^2 + 8 = 0

/-- 1 is the largest integer value of k such that the equation has at least one positive root -/
theorem largest_k_for_positive_root :
  (has_positive_root 1) ∧ 
  (∀ k : ℤ, k > 1 → ¬(has_positive_root k)) :=
sorry

end NUMINAMATH_CALUDE_largest_k_for_positive_root_l2817_281720


namespace NUMINAMATH_CALUDE_water_displaced_squared_5ft_l2817_281730

/-- The volume of water displaced by a fully submerged cube -/
def water_displaced (cube_side : ℝ) : ℝ := cube_side ^ 3

/-- The square of the volume of water displaced by a fully submerged cube -/
def water_displaced_squared (cube_side : ℝ) : ℝ := (water_displaced cube_side) ^ 2

/-- Theorem: The square of the volume of water displaced by a fully submerged cube
    with side length 5 feet is equal to 15625 (cubic feet)^2 -/
theorem water_displaced_squared_5ft :
  water_displaced_squared 5 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_squared_5ft_l2817_281730


namespace NUMINAMATH_CALUDE_xy_equation_implications_l2817_281713

theorem xy_equation_implications (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_xy_equation_implications_l2817_281713


namespace NUMINAMATH_CALUDE_paint_room_time_l2817_281782

/-- Andy's painting rate in rooms per hour -/
def andy_rate : ℚ := 1 / 4

/-- Bob's painting rate in rooms per hour -/
def bob_rate : ℚ := 1 / 6

/-- The combined painting rate of Andy and Bob in rooms per hour -/
def combined_rate : ℚ := andy_rate + bob_rate

/-- The time taken to paint the room, including the lunch break -/
def t : ℚ := 22 / 5

theorem paint_room_time :
  (combined_rate * (t - 2) = 1) ∧ (combined_rate = 5 / 12) := by
  sorry

end NUMINAMATH_CALUDE_paint_room_time_l2817_281782


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2817_281781

/-- The asymptotic lines of a hyperbola with equation x^2 - y^2/9 = 1 are y = ±3x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - y^2/9 = 1) → (∃ k : ℝ, k = 3 ∨ k = -3) → (y = k*x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2817_281781


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2817_281723

theorem imaginary_part_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z := i^2 / (1 - i)
  (z.im : ℝ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2817_281723


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2817_281774

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2817_281774


namespace NUMINAMATH_CALUDE_min_value_expression_l2817_281705

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 2) :
  (x + 40 * y + 4) / (3 * x * y) ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2817_281705


namespace NUMINAMATH_CALUDE_mass_of_Fe2CO3_3_l2817_281757

/-- The molar mass of iron in g/mol -/
def molar_mass_Fe : ℝ := 55.845

/-- The molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.011

/-- The molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 15.999

/-- The number of moles of Fe2(CO3)3 -/
def num_moles : ℝ := 12

/-- The molar mass of Fe2(CO3)3 in g/mol -/
def molar_mass_Fe2CO3_3 : ℝ :=
  2 * molar_mass_Fe + 3 * molar_mass_C + 9 * molar_mass_O

/-- The mass of Fe2(CO3)3 in grams -/
def mass_Fe2CO3_3 : ℝ := num_moles * molar_mass_Fe2CO3_3

theorem mass_of_Fe2CO3_3 : mass_Fe2CO3_3 = 3500.568 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_Fe2CO3_3_l2817_281757


namespace NUMINAMATH_CALUDE_radius_ratio_in_regular_hexagonal_pyramid_l2817_281783

/-- A regular hexagonal pyramid with a circumscribed sphere and an inscribed sphere. -/
structure RegularHexagonalPyramid where
  /-- The radius of the circumscribed sphere -/
  R_c : ℝ
  /-- The radius of the inscribed sphere -/
  R_i : ℝ
  /-- The center of the circumscribed sphere lies on the surface of the inscribed sphere -/
  center_on_surface : R_c = R_i + R_i

/-- The ratio of the radius of the circumscribed sphere to the radius of the inscribed sphere
    in a regular hexagonal pyramid where the center of the circumscribed sphere lies on
    the surface of the inscribed sphere is equal to 1 + √(7/3). -/
theorem radius_ratio_in_regular_hexagonal_pyramid (p : RegularHexagonalPyramid) :
  p.R_c / p.R_i = 1 + Real.sqrt (7/3) := by
  sorry

end NUMINAMATH_CALUDE_radius_ratio_in_regular_hexagonal_pyramid_l2817_281783


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2817_281734

-- Define the sets M and N
def M : Set Nat := {1, 2}
def N : Set Nat := {2, 3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2817_281734


namespace NUMINAMATH_CALUDE_collinear_points_determine_a_l2817_281739

/-- Given three points A(1,-1), B(a,3), and C(4,5) that are collinear,
    prove that a = 3. -/
theorem collinear_points_determine_a (a : ℝ) :
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (a, 3)
  let C : ℝ × ℝ := (4, 5)
  (∃ (t : ℝ), B.1 = A.1 + t * (C.1 - A.1) ∧ B.2 = A.2 + t * (C.2 - A.2)) →
  a = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_collinear_points_determine_a_l2817_281739


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l2817_281700

open Real

theorem min_translation_for_symmetry :
  let f (x m : ℝ) := Real.sqrt 3 * cos (x + m) + sin (x + m)
  ∃ (min_m : ℝ), min_m > 0 ∧
    (∀ (m : ℝ), m > 0 → 
      (∀ (x : ℝ), f x m = f (-x) m) → m ≥ min_m) ∧
    (∀ (x : ℝ), f x min_m = f (-x) min_m) ∧
    min_m = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l2817_281700


namespace NUMINAMATH_CALUDE_girl_scouts_expenses_l2817_281704

def total_earnings : ℝ := 30

def pool_entry_cost : ℝ :=
  5 * 3.5 + 3 * 2 + 2 * 1

def transportation_cost : ℝ :=
  6 * 1.5 + 4 * 0.75

def snack_cost : ℝ :=
  3 * 3 + 4 * 2.5 + 3 * 2

def total_expenses : ℝ :=
  pool_entry_cost + transportation_cost + snack_cost

theorem girl_scouts_expenses (h : total_expenses > total_earnings) :
  total_expenses - total_earnings = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_girl_scouts_expenses_l2817_281704


namespace NUMINAMATH_CALUDE_biker_journey_time_l2817_281785

/-- Given a biker's journey between two towns, prove the time taken for the first half. -/
theorem biker_journey_time (total_distance : ℝ) (initial_speed : ℝ) (speed_increase : ℝ) (second_half_time : ℝ) :
  total_distance = 140 →
  initial_speed = 14 →
  speed_increase = 2 →
  second_half_time = 7/3 →
  (total_distance / 2) / initial_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_biker_journey_time_l2817_281785


namespace NUMINAMATH_CALUDE_triangle_properties_l2817_281765

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (t : Triangle)
  (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c)
  (h2 : t.a = Real.sqrt 2)
  (h3 : Real.sin t.B * Real.sin t.C = (Real.sin t.A)^2) :
  t.A = π/3 ∧ (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2817_281765


namespace NUMINAMATH_CALUDE_taxi_theorem_l2817_281775

def taxi_distances : List ℤ := [9, -3, -5, 4, -8, 6]

def fuel_consumption : ℚ := 0.08
def gasoline_price : ℚ := 6
def starting_price : ℚ := 6
def additional_charge : ℚ := 1.5
def starting_distance : ℕ := 3

def total_distance (distances : List ℤ) : ℕ :=
  (distances.map (Int.natAbs)).sum

def fuel_cost (distance : ℕ) : ℚ :=
  distance * fuel_consumption * gasoline_price

def segment_income (distance : ℕ) : ℚ :=
  if distance ≤ starting_distance then
    starting_price
  else
    starting_price + (distance - starting_distance) * additional_charge

def total_income (distances : List ℤ) : ℚ :=
  (distances.map (Int.natAbs)).map segment_income |>.sum

def net_income (distances : List ℤ) : ℚ :=
  total_income distances - fuel_cost (total_distance distances)

theorem taxi_theorem :
  total_distance taxi_distances = 35 ∧
  fuel_cost (total_distance taxi_distances) = 16.8 ∧
  net_income taxi_distances = 44.7 := by
  sorry

end NUMINAMATH_CALUDE_taxi_theorem_l2817_281775


namespace NUMINAMATH_CALUDE_random_subset_is_sample_l2817_281786

/-- Represents a population of elements -/
structure Population (α : Type) where
  elements : Finset α
  size : ℕ
  size_eq : elements.card = size

/-- Represents a sample taken from a population -/
structure Sample (α : Type) where
  elements : Finset α
  size : ℕ
  size_eq : elements.card = size

/-- Defines what it means for a sample to be from a population -/
def is_sample_of {α : Type} (s : Sample α) (p : Population α) : Prop :=
  s.elements ⊆ p.elements ∧ s.size < p.size

/-- The theorem statement -/
theorem random_subset_is_sample 
  {α : Type} (p : Population α) (s : Sample α) 
  (h_p_size : p.size = 50000) 
  (h_s_size : s.size = 2000) 
  (h_subset : s.elements ⊆ p.elements) : 
  is_sample_of s p := by
  sorry


end NUMINAMATH_CALUDE_random_subset_is_sample_l2817_281786


namespace NUMINAMATH_CALUDE_lisas_eggs_per_child_l2817_281788

theorem lisas_eggs_per_child (breakfasts_per_year : ℕ) (num_children : ℕ) 
  (husband_eggs : ℕ) (self_eggs : ℕ) (total_eggs : ℕ) :
  breakfasts_per_year = 260 →
  num_children = 4 →
  husband_eggs = 3 →
  self_eggs = 2 →
  total_eggs = 3380 →
  ∃ (eggs_per_child : ℕ), 
    eggs_per_child = 2 ∧
    total_eggs = breakfasts_per_year * (num_children * eggs_per_child + husband_eggs + self_eggs) :=
by sorry

end NUMINAMATH_CALUDE_lisas_eggs_per_child_l2817_281788


namespace NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l2817_281708

/-- Represents a mathematical problem that may or may not require conditional statements --/
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| PiecewiseFunction

/-- Determines if a problem requires conditional statements --/
def requiresConditionalStatements (p : Problem) : Bool :=
  match p with
  | Problem.OppositeNumber => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.PiecewiseFunction => true

/-- The list of all problems --/
def allProblems : List Problem :=
  [Problem.OppositeNumber, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.PiecewiseFunction]

/-- Theorem stating that the number of problems not requiring conditional statements is 2 --/
theorem two_problems_without_conditional_statements :
  (allProblems.filter (fun p => ¬(requiresConditionalStatements p))).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l2817_281708


namespace NUMINAMATH_CALUDE_robin_total_pieces_l2817_281712

/-- The number of gum packages Robin has -/
def gum_packages : ℕ := 21

/-- The number of candy packages Robin has -/
def candy_packages : ℕ := 45

/-- The number of mint packages Robin has -/
def mint_packages : ℕ := 30

/-- The number of gum pieces in each gum package -/
def gum_pieces_per_package : ℕ := 9

/-- The number of candy pieces in each candy package -/
def candy_pieces_per_package : ℕ := 12

/-- The number of mint pieces in each mint package -/
def mint_pieces_per_package : ℕ := 8

/-- The total number of pieces Robin has -/
def total_pieces : ℕ := gum_packages * gum_pieces_per_package + 
                        candy_packages * candy_pieces_per_package + 
                        mint_packages * mint_pieces_per_package

theorem robin_total_pieces : total_pieces = 969 := by
  sorry

end NUMINAMATH_CALUDE_robin_total_pieces_l2817_281712


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2817_281771

structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  isIsosceles : side > 0

def medianDividesDifference (t : IsoscelesTriangle) (diff : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = t.base + 2 * t.side ∧ |x - y| = diff

theorem isosceles_triangle_side_length 
  (t : IsoscelesTriangle) 
  (h_base : t.base = 7) 
  (h_median : medianDividesDifference t 3) : 
  t.side = 4 ∨ t.side = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2817_281771


namespace NUMINAMATH_CALUDE_stating_valid_orderings_count_l2817_281789

/-- 
Given a positive integer n, this function returns the number of ways to order 
integers from 1 to n, where except for the first integer, every integer differs 
by 1 from some integer to its left.
-/
def validOrderings (n : ℕ) : ℕ :=
  2^(n-1)

/-- 
Theorem stating that the number of valid orderings of integers from 1 to n 
is equal to 2^(n-1), where a valid ordering is one in which, except for the 
first integer, every integer differs by 1 from some integer to its left.
-/
theorem valid_orderings_count (n : ℕ) (h : n > 0) : 
  validOrderings n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_stating_valid_orderings_count_l2817_281789


namespace NUMINAMATH_CALUDE_honey_savings_l2817_281738

/-- Calculates the savings given daily earnings, number of days worked, and total spent -/
def calculate_savings (daily_earnings : ℕ) (days_worked : ℕ) (total_spent : ℕ) : ℕ :=
  daily_earnings * days_worked - total_spent

/-- Proves that given the problem conditions, Honey's savings are $240 -/
theorem honey_savings :
  let daily_earnings : ℕ := 80
  let days_worked : ℕ := 20
  let total_spent : ℕ := 1360
  calculate_savings daily_earnings days_worked total_spent = 240 := by
sorry

#eval calculate_savings 80 20 1360

end NUMINAMATH_CALUDE_honey_savings_l2817_281738


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_spheres_l2817_281717

theorem surface_area_ratio_of_spheres (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_spheres_l2817_281717


namespace NUMINAMATH_CALUDE_cos_240_degrees_l2817_281784

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l2817_281784


namespace NUMINAMATH_CALUDE_festival_selection_probability_l2817_281760

-- Define the number of festivals
def total_festivals : ℕ := 5

-- Define the number of festivals to be selected
def selected_festivals : ℕ := 2

-- Define the number of specific festivals we're interested in
def specific_festivals : ℕ := 2

-- Define the probability of selecting at least one of the specific festivals
def probability : ℚ := 0.7

-- Theorem statement
theorem festival_selection_probability :
  1 - (Nat.choose (total_festivals - specific_festivals) selected_festivals) / 
      (Nat.choose total_festivals selected_festivals) = probability := by
  sorry

end NUMINAMATH_CALUDE_festival_selection_probability_l2817_281760


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l2817_281758

def A : Set ℝ := {x | x^2 + x - 12 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem intersection_implies_m_value :
  ∃ m : ℝ, A ∩ B m = {3} → m = -1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l2817_281758


namespace NUMINAMATH_CALUDE_optimal_shopping_solution_l2817_281736

/-- Represents the shopping problem with discounts --/
structure ShoppingProblem where
  budget : ℕ
  jacket_price : ℕ
  tshirt_price : ℕ
  jeans_price : ℕ

/-- Calculates the cost of jackets with the buy 2 get 1 free discount --/
def jacket_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 3 * 2 + n % 3) * price

/-- Calculates the cost of t-shirts with the buy 3 get 1 free discount --/
def tshirt_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 4 * 3 + n % 4) * price

/-- Calculates the cost of jeans with the 50% discount on every other pair --/
def jeans_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 2 * 3 + n % 2) * (price / 2)

/-- Represents the optimal shopping solution --/
structure ShoppingSolution where
  jackets : ℕ
  tshirts : ℕ
  jeans : ℕ
  total_spent : ℕ
  remaining : ℕ

/-- Theorem stating the optimal solution for the shopping problem --/
theorem optimal_shopping_solution (p : ShoppingProblem)
    (h : p = { budget := 400, jacket_price := 50, tshirt_price := 25, jeans_price := 40 }) :
    ∃ (s : ShoppingSolution),
      s.jackets = 4 ∧
      s.tshirts = 12 ∧
      s.jeans = 3 ∧
      s.total_spent = 380 ∧
      s.remaining = 20 ∧
      jacket_cost s.jackets p.jacket_price +
      tshirt_cost s.tshirts p.tshirt_price +
      jeans_cost s.jeans p.jeans_price = s.total_spent ∧
      s.total_spent + s.remaining = p.budget ∧
      ∀ (s' : ShoppingSolution),
        jacket_cost s'.jackets p.jacket_price +
        tshirt_cost s'.tshirts p.tshirt_price +
        jeans_cost s'.jeans p.jeans_price ≤ p.budget →
        s'.jackets + s'.tshirts + s'.jeans ≤ s.jackets + s.tshirts + s.jeans :=
by sorry

end NUMINAMATH_CALUDE_optimal_shopping_solution_l2817_281736


namespace NUMINAMATH_CALUDE_selling_price_a_is_1600_l2817_281749

/-- Represents the sales and pricing information for bicycle types A and B --/
structure BikeData where
  lastYearTotalSalesA : ℕ
  priceDecreaseA : ℕ
  salesDecreasePercentage : ℚ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ

/-- Calculates the selling price of type A bikes this year --/
def calculateSellingPriceA (data : BikeData) : ℕ :=
  sorry

/-- Theorem stating that the selling price of type A bikes this year is 1600 yuan --/
theorem selling_price_a_is_1600 (data : BikeData) 
  (h1 : data.lastYearTotalSalesA = 50000)
  (h2 : data.priceDecreaseA = 400)
  (h3 : data.salesDecreasePercentage = 1/5)
  (h4 : data.purchasePriceA = 1100)
  (h5 : data.purchasePriceB = 1400)
  (h6 : data.sellingPriceB = 2000) :
  calculateSellingPriceA data = 1600 :=
sorry

end NUMINAMATH_CALUDE_selling_price_a_is_1600_l2817_281749


namespace NUMINAMATH_CALUDE_parabola_translation_l2817_281715

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk (-2) 0 0
  let translated := translate original 1 (-3)
  translated = Parabola.mk (-2) 4 (-3) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2817_281715


namespace NUMINAMATH_CALUDE_trig_identity_l2817_281776

theorem trig_identity (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.tan (θ + π/3) = 1/2) : 
  Real.sin θ + Real.sqrt 3 * Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2817_281776


namespace NUMINAMATH_CALUDE_direct_proportion_increases_l2817_281741

theorem direct_proportion_increases (x₁ x₂ : ℝ) (h : x₁ < x₂) : 2 * x₁ < 2 * x₂ := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_increases_l2817_281741


namespace NUMINAMATH_CALUDE_log_equation_holds_l2817_281716

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 49 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l2817_281716


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2817_281703

theorem smallest_sum_of_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ r : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (∃ q : ℚ, C = B * q ∧ D = C * q) →  -- B, C, D form a geometric sequence
  C = (7 * B) / 4 →  -- C/B = 7/4
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (∃ r : ℤ, C' - B' = B' - A') → 
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) → 
    C' = (7 * B') / 4 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2817_281703


namespace NUMINAMATH_CALUDE_car_sales_prediction_l2817_281791

theorem car_sales_prediction (sports_cars : ℕ) (sedans : ℕ) (other_cars : ℕ) : 
  sports_cars = 35 →
  5 * sedans = 8 * sports_cars →
  sedans = 2 * other_cars →
  other_cars = 28 := by
sorry

end NUMINAMATH_CALUDE_car_sales_prediction_l2817_281791


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2817_281796

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : math = 80)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2817_281796


namespace NUMINAMATH_CALUDE_part_one_part_two_l2817_281797

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part 1
theorem part_one :
  A ∪ B 1 = Set.Icc (-1) 3 ∧
  (Set.univ \ B 1) = {x | x < 0 ∨ x > 3} :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (Set.univ \ A) ∩ B a = ∅ ↔ (0 ≤ a ∧ a ≤ 1) ∨ a < -2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2817_281797


namespace NUMINAMATH_CALUDE_muffin_distribution_l2817_281780

theorem muffin_distribution (total_students : ℕ) (absent_students : ℕ) (extra_muffins : ℕ) : 
  total_students = 400 →
  absent_students = 180 →
  extra_muffins = 36 →
  (total_students * ((total_students - absent_students) * extra_muffins + total_students * (total_students - absent_students))) / 
  ((total_students - absent_students) * total_students) = 80 := by
sorry

end NUMINAMATH_CALUDE_muffin_distribution_l2817_281780


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2817_281731

/-- A geometric sequence with first term a₁ and common ratio q. -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h_geom : ∃ (a₁ q : ℝ), ∀ n, a n = geometric_sequence a₁ q n)
  (h_a₁ : a 1 = 2) (h_a₄ : a 4 = 16) :
  ∃ q, ∀ n, a n = geometric_sequence 2 q n ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2817_281731


namespace NUMINAMATH_CALUDE_extreme_value_M_inequality_condition_l2817_281746

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := x + 1/x + a * (1/x)

noncomputable def M (x : ℝ) : ℝ := x - log x + 2/x

theorem extreme_value_M :
  (∀ x > 0, M x ≥ 3 - log 2) ∧
  (M 2 = 3 - log 2) ∧
  (∀ b : ℝ, ∃ x > 0, M x > b) :=
sorry

theorem inequality_condition (m : ℝ) :
  (∀ x > 0, 1 / (x + 1/x) ≤ 1 / (2 + m * (log x)^2)) ↔ 0 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_M_inequality_condition_l2817_281746


namespace NUMINAMATH_CALUDE_exactly_two_valid_multiplications_l2817_281750

def is_valid_multiplication (a b : ℕ) : Prop :=
  100 ≤ a ∧ a < 1000 ∧  -- a is a three-digit number
  a / 100 = 1 ∧  -- a starts with 1
  1 ≤ b ∧ b < 10 ∧  -- b is a single-digit number
  1000 ≤ a * b ∧ a * b < 10000 ∧  -- product is four digits
  (a * (b % 10) / 100 = 1)  -- third row starts with '100'

theorem exactly_two_valid_multiplications :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ a ∈ s, ∃ b, is_valid_multiplication a b :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_multiplications_l2817_281750


namespace NUMINAMATH_CALUDE_blood_expiration_time_l2817_281744

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  hle24 : hours < 24
  mle59 : minutes < 60

/-- Represents a date with a day and a time -/
structure Date where
  day : Nat
  time : TimeOfDay

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def secondsToTime (seconds : Nat) : TimeOfDay :=
  let totalMinutes := seconds / 60
  let hours := totalMinutes / 60
  let minutes := totalMinutes % 60
  ⟨hours % 24, minutes, by sorry, by sorry⟩

def addTimeToDate (d : Date) (seconds : Nat) : Date :=
  let newTime := secondsToTime ((d.time.hours * 60 + d.time.minutes) * 60 + seconds)
  ⟨d.day + (if newTime.hours < d.time.hours then 1 else 0), newTime⟩

theorem blood_expiration_time :
  let donationDate : Date := ⟨1, ⟨12, 0, by sorry, by sorry⟩⟩
  let expirationSeconds := factorial 8
  let expirationDate := addTimeToDate donationDate expirationSeconds
  expirationDate = ⟨1, ⟨23, 13, by sorry, by sorry⟩⟩ :=
by sorry

end NUMINAMATH_CALUDE_blood_expiration_time_l2817_281744


namespace NUMINAMATH_CALUDE_simplify_fraction_l2817_281754

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2817_281754


namespace NUMINAMATH_CALUDE_tangent_lines_count_l2817_281767

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Counts the number of lines tangent to two circles -/
def count_tangent_lines (c1 c2 : Circle) : ℕ :=
  sorry

theorem tangent_lines_count 
  (A B : ℝ × ℝ)
  (C_A : Circle)
  (C_B : Circle)
  (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 7)
  (h_C_A_center : C_A.center = A)
  (h_C_B_center : C_B.center = B)
  (h_C_A_radius : C_A.radius = 3)
  (h_C_B_radius : C_B.radius = 4) :
  count_tangent_lines C_A C_B = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l2817_281767


namespace NUMINAMATH_CALUDE_sqrt_comparison_quadratic_inequality_solution_l2817_281787

-- Part 1
theorem sqrt_comparison : Real.sqrt 7 + Real.sqrt 10 > Real.sqrt 3 + Real.sqrt 14 := by sorry

-- Part 2
theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x, -1/2 * x^2 + 2*x > m*x ↔ 0 < x ∧ x < 2) → m = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_comparison_quadratic_inequality_solution_l2817_281787


namespace NUMINAMATH_CALUDE_tangent_sum_inequality_l2817_281727

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π
  
-- Define the perimeter and inradius
def perimeter (t : AcuteTriangle) : Real := sorry
def inradius (t : AcuteTriangle) : Real := sorry

-- State the theorem
theorem tangent_sum_inequality (t : AcuteTriangle) :
  Real.tan t.A + Real.tan t.B + Real.tan t.C ≥ perimeter t / (2 * inradius t) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_inequality_l2817_281727


namespace NUMINAMATH_CALUDE_min_triangle_area_l2817_281733

-- Define the triangle and square
structure Triangle :=
  (X Y Z : ℝ × ℝ)

structure Square :=
  (side : ℝ)
  (area : ℝ)

-- Define the properties
def is_acute_angled (t : Triangle) : Prop := sorry

def square_inscribed (t : Triangle) (s : Square) : Prop := sorry

-- Theorem statement
theorem min_triangle_area 
  (t : Triangle) 
  (s : Square) 
  (h_acute : is_acute_angled t) 
  (h_inscribed : square_inscribed t s) 
  (h_area : s.area = 2017) : 
  ∃ (min_area : ℝ), min_area = 2017/2 ∧ 
  ∀ (actual_area : ℝ), actual_area ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_min_triangle_area_l2817_281733


namespace NUMINAMATH_CALUDE_equality_of_polynomials_l2817_281790

theorem equality_of_polynomials (a b c : ℝ) :
  (∀ x : ℝ, (x^2 + a*x - 3)*(x + 1) = x^3 + b*x^2 + c*x - 3) →
  b - c = 4 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_polynomials_l2817_281790


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2817_281793

theorem polynomial_factorization (x : ℝ) :
  9 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (3 * x^2 + 52 * x + 231) * (3 * x^2 + 56 * x + 231) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2817_281793


namespace NUMINAMATH_CALUDE_lawrence_county_camp_kids_l2817_281747

/-- The number of kids going to camp during summer break in Lawrence county --/
def kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) : ℕ :=
  total_kids - kids_at_home

/-- Theorem stating the number of kids going to camp in Lawrence county --/
theorem lawrence_county_camp_kids :
  kids_at_camp 313473 274865 = 38608 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_camp_kids_l2817_281747


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l2817_281719

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) : 
  (4 * π * r^2) / (4 * π * R^2) = 4 / 9 → 
  ((4 / 3) * π * r^3) / ((4 / 3) * π * R^3) = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l2817_281719


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l2817_281753

/-- The square of the side length of an equilateral triangle inscribed in a specific circle -/
theorem equilateral_triangle_side_length_squared (x y : ℝ) : 
  x^2 + y^2 = 16 →  -- Circle equation
  (0 : ℝ)^2 + 4^2 = 16 →  -- One vertex at (0, 4)
  (∃ a b : ℝ, a^2 + b^2 = 16 ∧ (0 - a)^2 + (4 - b)^2 = a^2 + (4 - b)^2) →  -- Triangle inscribed in circle
  (∃ c : ℝ, c^2 + (-3)^2 = 16) →  -- Altitude on y-axis (implied by y = -3 for other vertices)
  (0 : ℝ)^2 + 7^2 = 49 :=  -- Square of side length is 49
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l2817_281753


namespace NUMINAMATH_CALUDE_gcd_g_50_52_eq_one_l2817_281769

/-- The function g(x) = x^2 - 3x + 2023 -/
def g (x : ℤ) : ℤ := x^2 - 3*x + 2023

/-- Theorem: The greatest common divisor of g(50) and g(52) is 1 -/
theorem gcd_g_50_52_eq_one : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_50_52_eq_one_l2817_281769


namespace NUMINAMATH_CALUDE_fourth_number_ninth_row_l2817_281764

/-- Represents the lattice structure with the given pattern -/
def lattice_sequence (row : ℕ) (position : ℕ) : ℕ :=
  8 * (row - 1) + position

/-- The problem statement -/
theorem fourth_number_ninth_row :
  lattice_sequence 9 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_ninth_row_l2817_281764


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l2817_281756

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_half : reciprocal (-1/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l2817_281756


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l2817_281711

theorem snooker_ticket_difference :
  ∀ (vip_tickets gen_tickets : ℕ),
    vip_tickets + gen_tickets = 320 →
    40 * vip_tickets + 15 * gen_tickets = 7500 →
    gen_tickets - vip_tickets = 104 := by
  sorry

end NUMINAMATH_CALUDE_snooker_ticket_difference_l2817_281711


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2817_281748

/-- The length of the major axis of an ellipse C, given specific conditions -/
theorem ellipse_major_axis_length : 
  ∀ (m : ℝ) (x y : ℝ → ℝ),
    (m > 0) →
    (∀ t, 2 * (x t) - (y t) + 4 = 0) →
    (∀ t, (x t)^2 / m + (y t)^2 / 2 = 1) →
    (∃ t₀, (x t₀, y t₀) = (-2, 0) ∨ (x t₀, y t₀) = (0, 4)) →
    ∃ a b : ℝ, a^2 = m ∧ b^2 = 2 ∧ 2 * max a b = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2817_281748


namespace NUMINAMATH_CALUDE_area_triangle_ABC_l2817_281729

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line m
def line_m : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Define the circles A, B, and C
def circle_A : Circle := ⟨(-7, 3), 3⟩
def circle_B : Circle := ⟨(0, -4), 4⟩
def circle_C : Circle := ⟨(9, 5), 5⟩

-- Define the tangent points
def point_A' : ℝ × ℝ := (-7, 0)
def point_B' : ℝ × ℝ := (0, 0)
def point_C' : ℝ × ℝ := (9, 0)

-- Define the properties of the circles and their arrangement
axiom tangent_to_m : 
  circle_A.center.2 = circle_A.radius ∧
  circle_B.center.2 = -circle_B.radius ∧
  circle_C.center.2 = circle_C.radius

axiom external_tangency :
  (circle_A.center.1 - circle_B.center.1)^2 + (circle_A.center.2 - circle_B.center.2)^2 
    = (circle_A.radius + circle_B.radius)^2 ∧
  (circle_C.center.1 - circle_B.center.1)^2 + (circle_C.center.2 - circle_B.center.2)^2 
    = (circle_C.radius + circle_B.radius)^2

axiom B'_between_A'_C' :
  point_A'.1 < point_B'.1 ∧ point_B'.1 < point_C'.1

-- Theorem to prove
theorem area_triangle_ABC : 
  let A := circle_A.center
  let B := circle_B.center
  let C := circle_C.center
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 63 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_l2817_281729


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2817_281752

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2817_281752


namespace NUMINAMATH_CALUDE_books_per_shelf_l2817_281718

theorem books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves : ℕ) : 
  total_books = 14 → books_taken = 2 → shelves = 4 → 
  (total_books - books_taken) / shelves = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2817_281718


namespace NUMINAMATH_CALUDE_third_podcast_length_l2817_281721

/-- Given a 6-hour drive and podcast lengths, prove the third podcast's length --/
theorem third_podcast_length 
  (total_drive_time : ℕ) 
  (first_podcast : ℕ) 
  (second_podcast : ℕ) 
  (fourth_podcast : ℕ) 
  (fifth_podcast : ℕ) 
  (h1 : total_drive_time = 6 * 60) 
  (h2 : first_podcast = 45) 
  (h3 : second_podcast = 2 * first_podcast) 
  (h4 : fourth_podcast = 60) 
  (h5 : fifth_podcast = 60) : 
  ∃ (third_podcast : ℕ), 
    third_podcast = total_drive_time - (first_podcast + second_podcast + fourth_podcast + fifth_podcast) ∧ 
    third_podcast = 105 := by
  sorry

end NUMINAMATH_CALUDE_third_podcast_length_l2817_281721


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_l2817_281702

theorem numerator_increase_percentage (original_fraction : ℚ) 
  (denominator_decrease : ℚ) (new_fraction : ℚ) : 
  original_fraction = 3/4 →
  denominator_decrease = 8/100 →
  new_fraction = 15/16 →
  ∃ numerator_increase : ℚ, 
    (original_fraction * (1 + numerator_increase)) / (1 - denominator_decrease) = new_fraction ∧
    numerator_increase = 15/100 := by
  sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_l2817_281702


namespace NUMINAMATH_CALUDE_magician_earnings_l2817_281724

-- Define the problem parameters
def initial_decks : ℕ := 20
def final_decks : ℕ := 5
def full_price : ℚ := 7
def discount_percentage : ℚ := 20 / 100

-- Define the number of decks sold at full price and discounted price
def full_price_sales : ℕ := 7
def discounted_sales : ℕ := 8

-- Calculate the discounted price
def discounted_price : ℚ := full_price * (1 - discount_percentage)

-- Calculate the total earnings
def total_earnings : ℚ := 
  (full_price_sales : ℚ) * full_price + 
  (discounted_sales : ℚ) * discounted_price

-- Theorem statement
theorem magician_earnings : 
  initial_decks - final_decks = full_price_sales + discounted_sales ∧ 
  total_earnings = 93.8 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l2817_281724


namespace NUMINAMATH_CALUDE_average_income_l2817_281709

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    calculate the average monthly income of the remaining pair. -/
theorem average_income (p q r : ℕ) : 
  (p + q) / 2 = 2050 →
  (p + r) / 2 = 6200 →
  p = 3000 →
  (q + r) / 2 = 5250 := by
  sorry


end NUMINAMATH_CALUDE_average_income_l2817_281709


namespace NUMINAMATH_CALUDE_cricketer_score_percentage_l2817_281745

/-- A cricketer's score breakdown and calculation of runs made by running between wickets --/
theorem cricketer_score_percentage (total_score : ℕ) (boundaries : ℕ) (sixes : ℕ)
  (singles : ℕ) (twos : ℕ) (threes : ℕ) :
  total_score = 138 →
  boundaries = 12 →
  sixes = 2 →
  singles = 25 →
  twos = 7 →
  threes = 3 →
  (((singles * 1 + twos * 2 + threes * 3) : ℚ) / total_score) * 100 = 48 / 138 * 100 := by
  sorry

#eval (48 : ℚ) / 138 * 100

end NUMINAMATH_CALUDE_cricketer_score_percentage_l2817_281745


namespace NUMINAMATH_CALUDE_compound_interest_doubling_l2817_281763

/-- The annual interest rate as a decimal -/
def r : ℝ := 0.15

/-- The compound interest factor for one year -/
def factor : ℝ := 1 + r

/-- The number of years we're proving about -/
def years : ℕ := 5

theorem compound_interest_doubling :
  (∀ n : ℕ, n < years → factor ^ n ≤ 2) ∧
  factor ^ years > 2 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_doubling_l2817_281763


namespace NUMINAMATH_CALUDE_second_train_length_proof_l2817_281710

/-- The length of the second train given the conditions of the problem -/
def second_train_length : ℝ := 199.9760019198464

/-- The length of the first train in meters -/
def first_train_length : ℝ := 100

/-- The speed of the first train in kilometers per hour -/
def first_train_speed : ℝ := 42

/-- The speed of the second train in kilometers per hour -/
def second_train_speed : ℝ := 30

/-- The time it takes for the trains to clear each other in seconds -/
def clearing_time : ℝ := 14.998800095992321

theorem second_train_length_proof :
  second_train_length = 
    (first_train_speed + second_train_speed) * (1000 / 3600) * clearing_time - first_train_length := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_proof_l2817_281710


namespace NUMINAMATH_CALUDE_duck_count_proof_l2817_281740

/-- The number of mallard ducks initially at the park -/
def initial_ducks : ℕ := sorry

/-- The number of geese initially at the park -/
def initial_geese : ℕ := 2 * initial_ducks - 10

/-- The number of ducks after the small flock arrives -/
def ducks_after_arrival : ℕ := initial_ducks + 4

/-- The number of geese after some leave -/
def geese_after_leaving : ℕ := initial_geese - 10

theorem duck_count_proof : 
  initial_ducks = 25 ∧ 
  geese_after_leaving = ducks_after_arrival + 1 :=
sorry

end NUMINAMATH_CALUDE_duck_count_proof_l2817_281740


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2817_281728

-- Problem 1
theorem problem_1 : -1.5 + 1.4 - (-3.6) - 4.3 + (-5.2) = -6 := by
  sorry

-- Problem 2
theorem problem_2 : 17 - 2^3 / (-2) * 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2817_281728


namespace NUMINAMATH_CALUDE_sandy_has_144_marbles_l2817_281751

def dozen : ℕ := 12

def jessica_marbles : ℕ := 3 * dozen

def sandy_marbles : ℕ := 4 * jessica_marbles

theorem sandy_has_144_marbles : sandy_marbles = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_has_144_marbles_l2817_281751


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2817_281799

/-- Proves that given a journey of 120 miles in 75 minutes, where the average speed
    for the first 25 minutes is 75 mph and for the next 25 minutes is 80 mph,
    the average speed for the last 25 minutes is 133 mph. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
    (speed1 : ℝ) (speed2 : ℝ) (h1 : total_distance = 120) 
    (h2 : total_time = 75 / 60) (h3 : speed1 = 75) (h4 : speed2 = 80) : 
    (3 * total_distance / total_time - speed1 - speed2 : ℝ) = 133 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l2817_281799


namespace NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l2817_281706

theorem rectangular_solid_on_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l2817_281706


namespace NUMINAMATH_CALUDE_green_marbles_fraction_l2817_281773

theorem green_marbles_fraction (total : ℚ) (h1 : total > 0) : 
  let blue : ℚ := 2/3 * total
  let red : ℚ := 1/6 * total
  let green : ℚ := total - blue - red
  let new_total : ℚ := total + blue
  (green / new_total) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_fraction_l2817_281773


namespace NUMINAMATH_CALUDE_expression_evaluation_l2817_281770

theorem expression_evaluation : 
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2817_281770


namespace NUMINAMATH_CALUDE_percentage_problem_l2817_281795

theorem percentage_problem (P : ℝ) : P = 35 ↔ (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2817_281795


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2817_281772

-- Define the quadratic function f
def f (x : ℝ) : ℝ := sorry

-- Define the conditions for f
axiom f_zero : f 0 = 0
axiom f_recurrence (x : ℝ) : f (x + 1) = f x + x + 1

-- Define the minimum value function g
def g (t : ℝ) : ℝ := sorry

-- Theorem to prove
theorem quadratic_function_properties :
  -- Part 1: Expression for f(x)
  (∀ x, f x = (1/2) * x^2 + (1/2) * x) ∧
  -- Part 2: Expression for g(t)
  (∀ t, g t = if t ≤ -3/2 then (1/2) * t^2 + (3/2) * t + 1
              else if t < -1/2 then -1/8
              else (1/2) * t^2 + (1/2) * t) ∧
  -- Part 3: Range of m
  (∀ m, (∀ t, g t + m ≥ 0) ↔ m ≥ 1/8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2817_281772


namespace NUMINAMATH_CALUDE_grain_milling_theorem_l2817_281798

/-- The amount of grain needed to be milled, in pounds -/
def grain_amount : ℚ := 111 + 1/9

/-- The milling fee percentage -/
def milling_fee_percent : ℚ := 1/10

/-- The amount of flour remaining after paying the fee, in pounds -/
def remaining_flour : ℚ := 100

theorem grain_milling_theorem :
  (1 - milling_fee_percent) * grain_amount = remaining_flour :=
by sorry

end NUMINAMATH_CALUDE_grain_milling_theorem_l2817_281798


namespace NUMINAMATH_CALUDE_max_ratio_in_triangle_l2817_281732

/-- Given a triangle OAB where O is the origin, A is the point (4,3), and B is the point (x,0) with x > 0,
    this theorem states that the maximum value of the ratio x/l(x) is 5/3,
    where l(x) is the length of line segment AB. -/
theorem max_ratio_in_triangle (x : ℝ) (hx : x > 0) : 
  let A : ℝ × ℝ := (4, 3)
  let B : ℝ × ℝ := (x, 0)
  let l : ℝ → ℝ := fun x => Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (∀ y > 0, y / l y ≤ x / l x) → x / l x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_max_ratio_in_triangle_l2817_281732


namespace NUMINAMATH_CALUDE_probability_two_girls_l2817_281722

theorem probability_two_girls (total_members : ℕ) (girl_members : ℕ) : 
  total_members = 12 → 
  girl_members = 7 → 
  (Nat.choose girl_members 2 : ℚ) / (Nat.choose total_members 2 : ℚ) = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l2817_281722


namespace NUMINAMATH_CALUDE_max_mineral_value_l2817_281743

/-- Represents a type of mineral with its weight and value --/
structure Mineral where
  weight : ℕ
  value : ℕ

/-- The problem setup --/
def mineral_problem : Prop :=
  ∃ (j k l : Mineral) (x y z : ℕ),
    j.weight = 6 ∧ j.value = 17 ∧
    k.weight = 3 ∧ k.value = 9 ∧
    l.weight = 2 ∧ l.value = 5 ∧
    x * j.weight + y * k.weight + z * l.weight ≤ 20 ∧
    ∀ (a b c : ℕ),
      a * j.weight + b * k.weight + c * l.weight ≤ 20 →
      a * j.value + b * k.value + c * l.value ≤ x * j.value + y * k.value + z * l.value ∧
      x * j.value + y * k.value + z * l.value = 60

theorem max_mineral_value : mineral_problem := by sorry

end NUMINAMATH_CALUDE_max_mineral_value_l2817_281743


namespace NUMINAMATH_CALUDE_average_cost_is_14_cents_l2817_281762

/-- Calculates the average cost per pencil in cents, rounded to the nearest whole number -/
def average_cost_per_pencil (num_pencils : ℕ) (catalog_price shipping_cost discount : ℚ) : ℕ :=
  let total_cost_cents := (catalog_price + shipping_cost - discount) * 100
  let average_cost_cents := total_cost_cents / num_pencils
  (average_cost_cents + 1/2).floor.toNat

/-- Proves that the average cost per pencil is 14 cents given the specified conditions -/
theorem average_cost_is_14_cents :
  average_cost_per_pencil 150 15 7.5 1.5 = 14 := by
  sorry

#eval average_cost_per_pencil 150 15 7.5 1.5

end NUMINAMATH_CALUDE_average_cost_is_14_cents_l2817_281762


namespace NUMINAMATH_CALUDE_stock_sold_percentage_l2817_281737

/-- Given the cash realized, brokerage rate, and total amount including brokerage,
    prove that the percentage of stock sold is 100% -/
theorem stock_sold_percentage
  (cash_realized : ℝ)
  (brokerage_rate : ℝ)
  (total_amount : ℝ)
  (h1 : cash_realized = 106.25)
  (h2 : brokerage_rate = 1 / 4 / 100)
  (h3 : total_amount = 106) :
  let sale_amount := cash_realized + (cash_realized * brokerage_rate)
  let stock_percentage := sale_amount / sale_amount * 100
  stock_percentage = 100 := by sorry

end NUMINAMATH_CALUDE_stock_sold_percentage_l2817_281737


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2817_281707

theorem tan_alpha_value (α : ℝ) 
  (h : (Real.sin (α + Real.pi) + Real.cos (Real.pi - α)) / 
       (Real.sin (Real.pi / 2 - α) + Real.sin (2 * Real.pi - α)) = 5) : 
  Real.tan α = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2817_281707
