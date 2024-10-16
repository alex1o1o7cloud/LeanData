import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_vertices_l524_52404

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 5
def g (x : ℝ) : ℝ := x^2 + 6*x + 13

-- Define the vertices of the two parabolas
def vertex_f : ℝ × ℝ := (2, f 2)
def vertex_g : ℝ × ℝ := (-3, g (-3))

-- State the theorem
theorem distance_between_vertices : 
  Real.sqrt ((vertex_f.1 - vertex_g.1)^2 + (vertex_f.2 - vertex_g.2)^2) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l524_52404


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l524_52485

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_of_integers 10 30 + count_even_integers 10 30 = 431 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l524_52485


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_and_point_l524_52434

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0

-- Define the point (1, -2)
def point : ℝ × ℝ := (1, -2)

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem equation_represents_two_lines_and_point :
  ∀ x y : ℝ, equation x y ↔ (x = point.1 ∧ y = point.2) ∨ line1 x y ∨ line2 x y :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_and_point_l524_52434


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l524_52470

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The angle at vertex A of a triangle -/
def angleA (t : Triangle) : ℝ := sorry

/-- Check if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- The region G formed by points P inside the triangle satisfying PA ≤ PB and PA ≤ PC -/
def regionG (t : Triangle) : Set Point :=
  {p : Point | isInside p t ∧ distance p t.A ≤ distance p t.B ∧ distance p t.A ≤ distance p t.C}

/-- The area of region G -/
def areaG (t : Triangle) : ℝ := sorry

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_is_equilateral (t : Triangle) :
  isAcute t →
  angleA t = π / 3 →
  areaG t = (1 / 3) * triangleArea t →
  isEquilateral t := by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l524_52470


namespace NUMINAMATH_CALUDE_paint_per_statue_l524_52496

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 7 / 16)
  (h2 : num_statues = 7) :
  total_paint / num_statues = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l524_52496


namespace NUMINAMATH_CALUDE_red_mushrooms_with_spots_l524_52410

/-- Represents the number of mushrooms gathered by Bill and Ted -/
structure MushroomGathering where
  red : ℕ
  brown : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the fraction of red mushrooms with white spots -/
def fraction_red_with_spots (g : MushroomGathering) (total_spotted : ℕ) : ℚ :=
  (total_spotted - g.brown - g.blue / 2) / g.red

/-- The main theorem stating the fraction of red mushrooms with white spots -/
theorem red_mushrooms_with_spots :
  let g := MushroomGathering.mk 12 6 14 6
  let total_spotted := 17
  fraction_red_with_spots g total_spotted = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_red_mushrooms_with_spots_l524_52410


namespace NUMINAMATH_CALUDE_intersection_sum_l524_52407

/-- Given two lines y = mx + 4 and y = 3x + b intersecting at (6, 10), prove b + m = -7 -/
theorem intersection_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + 4 ↔ y = 3 * x + b) → 
  (6 : ℝ) * m + 4 = 10 → 
  3 * 6 + b = 10 → 
  b + m = -7 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l524_52407


namespace NUMINAMATH_CALUDE_sequence_is_geometric_from_second_term_l524_52461

def is_geometric_from_second_term (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n + 1) = r * a n

theorem sequence_is_geometric_from_second_term
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : S 1 = 1)
  (h2 : S 2 = 2)
  (h3 : ∀ n : ℕ, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0)
  (h4 : ∀ n : ℕ, S (n + 1) - S n = a (n + 1))
  : is_geometric_from_second_term a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_geometric_from_second_term_l524_52461


namespace NUMINAMATH_CALUDE_range_of_m_l524_52493

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 15 / 2) * Real.sin (Real.pi * x)

theorem range_of_m (x₀ : ℝ) (h₁ : x₀ ∈ Set.Ioo (-1) 1)
  (h₂ : ∀ x : ℝ, f x ≤ f x₀)
  (h₃ : ∃ m : ℝ, x₀^2 + (f x₀)^2 < m^2) :
  ∃ m : ℝ, m ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l524_52493


namespace NUMINAMATH_CALUDE_sin_equality_iff_side_equality_l524_52432

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (angle_sum : A + B + C = π)

-- State the theorem
theorem sin_equality_iff_side_equality (t : Triangle) : 
  Real.sin t.A = Real.sin t.B ↔ t.a = t.b :=
sorry

end NUMINAMATH_CALUDE_sin_equality_iff_side_equality_l524_52432


namespace NUMINAMATH_CALUDE_basketball_tryouts_l524_52444

theorem basketball_tryouts (girls : ℕ) (called_back : ℕ) (not_selected : ℕ) :
  girls = 17 → called_back = 10 → not_selected = 39 →
  ∃ (boys : ℕ), girls + boys = called_back + not_selected ∧ boys = 32 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l524_52444


namespace NUMINAMATH_CALUDE_photo_perimeter_is_23_l524_52425

/-- Represents a rectangular photograph with a border -/
structure BorderedPhoto where
  width : ℝ
  length : ℝ
  borderWidth : ℝ

/-- Calculates the total area of a bordered photograph -/
def totalArea (photo : BorderedPhoto) : ℝ :=
  (photo.width + 2 * photo.borderWidth) * (photo.length + 2 * photo.borderWidth)

/-- Calculates the perimeter of the photograph without the border -/
def photoPerimeter (photo : BorderedPhoto) : ℝ :=
  2 * (photo.width + photo.length)

theorem photo_perimeter_is_23 (photo : BorderedPhoto) (m : ℝ) :
  photo.borderWidth = 2 →
  totalArea photo = m →
  totalArea { photo with borderWidth := 4 } = m + 94 →
  photoPerimeter photo = 23 := by
  sorry

end NUMINAMATH_CALUDE_photo_perimeter_is_23_l524_52425


namespace NUMINAMATH_CALUDE_origin_on_circle_A_l524_52413

def circle_A : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 5^2}

theorem origin_on_circle_A : (0, 0) ∈ circle_A := by
  sorry

end NUMINAMATH_CALUDE_origin_on_circle_A_l524_52413


namespace NUMINAMATH_CALUDE_all_nat_gt2_as_fib_sum_l524_52478

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define a function to check if a number is in the Fibonacci sequence
def isFib (n : ℕ) : Prop :=
  ∃ k, fib k = n

-- Define a function to represent a number as a sum of distinct Fibonacci numbers
def representAsFibSum (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ x ∈ S, isFib x) ∧ (S.sum id = n)

-- The main theorem
theorem all_nat_gt2_as_fib_sum :
  ∀ n : ℕ, n > 2 → representAsFibSum n :=
by
  sorry


end NUMINAMATH_CALUDE_all_nat_gt2_as_fib_sum_l524_52478


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_reciprocal_squares_l524_52462

theorem cubic_roots_sum_of_reciprocal_squares :
  ∀ a b c : ℝ,
  (a^3 - 12*a^2 + 14*a + 3 = 0) →
  (b^3 - 12*b^2 + 14*b + 3 = 0) →
  (c^3 - 12*c^2 + 14*c + 3 = 0) →
  (a + b + c = 12) →
  (a*b + b*c + c*a = 14) →
  (a*b*c = -3) →
  (1/a^2 + 1/b^2 + 1/c^2 = 268/9) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_reciprocal_squares_l524_52462


namespace NUMINAMATH_CALUDE_weight_difference_l524_52465

-- Define the weights as natural numbers
def sam_weight : ℕ := 105
def peter_weight : ℕ := 65

-- Define Tyler's weight based on Peter's weight
def tyler_weight : ℕ := 2 * peter_weight

-- Theorem to prove
theorem weight_difference : tyler_weight - sam_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l524_52465


namespace NUMINAMATH_CALUDE_bad_carrots_count_l524_52497

theorem bad_carrots_count (haley_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : mom_carrots = 38)
  (h3 : good_carrots = 64) :
  haley_carrots + mom_carrots - good_carrots = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l524_52497


namespace NUMINAMATH_CALUDE_simons_school_students_l524_52487

def total_students : ℕ := 2500

theorem simons_school_students (linas_students : ℕ) 
  (h1 : linas_students * 5 = total_students) : 
  linas_students * 4 = 2000 := by
  sorry

#check simons_school_students

end NUMINAMATH_CALUDE_simons_school_students_l524_52487


namespace NUMINAMATH_CALUDE_polynomial_remainder_l524_52416

/-- Given a polynomial Q(x) where Q(17) = 41 and Q(93) = 13, 
    the remainder when Q(x) is divided by (x - 17)(x - 93) is -7/19*x + 900/19 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 17 = 41) (h2 : Q 93 = 13) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 17) * (x - 93) * R x + (-7/19 * x + 900/19) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l524_52416


namespace NUMINAMATH_CALUDE_sphere_surface_area_and_volume_l524_52492

/-- Given a sphere with diameter 18 inches, prove its surface area and volume -/
theorem sphere_surface_area_and_volume :
  let diameter : ℝ := 18
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  let volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  surface_area = 324 * Real.pi ∧ volume = 972 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_sphere_surface_area_and_volume_l524_52492


namespace NUMINAMATH_CALUDE_smallest_divisible_by_11_ending_in_9_l524_52498

def is_smallest_divisible_by_11_ending_in_9 (n : ℕ) : Prop :=
  n > 0 ∧ 
  n % 10 = 9 ∧ 
  n % 11 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n

theorem smallest_divisible_by_11_ending_in_9 : 
  is_smallest_divisible_by_11_ending_in_9 99 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_11_ending_in_9_l524_52498


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l524_52467

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 5 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)) / 6)) →
  (0 ≤ n) ∧ (∀ m : ℤ, m < n → m + 5 ≥ 3 * ((m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) / 6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l524_52467


namespace NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l524_52414

/-- A function that checks if a number is composite -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

/-- The property we want to prove for 211 -/
def IsLargestNonSum30AndComposite (m : ℕ) : Prop :=
  (∀ n > m, ∃ k c : ℕ, n = 30 * k + c ∧ k > 0 ∧ IsComposite c) ∧
  (¬∃ k c : ℕ, m = 30 * k + c ∧ k > 0 ∧ IsComposite c)

/-- The main theorem -/
theorem largest_non_sum_30_and_composite :
  IsLargestNonSum30AndComposite 211 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l524_52414


namespace NUMINAMATH_CALUDE_mathematical_induction_l524_52435

theorem mathematical_induction (P : ℕ → Prop) (base : ℕ) 
  (base_case : P base)
  (inductive_step : ∀ k : ℕ, k ≥ base → P k → P (k + 1)) :
  ∀ n : ℕ, n ≥ base → P n :=
by
  sorry


end NUMINAMATH_CALUDE_mathematical_induction_l524_52435


namespace NUMINAMATH_CALUDE_evaluate_expression_l524_52412

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/2) (hy : y = 3/4) (hz : z = -6) (hw : w = 2) : 
  x^2 * y^4 * z * w = -243/256 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l524_52412


namespace NUMINAMATH_CALUDE_initial_group_size_l524_52468

theorem initial_group_size (initial_avg : ℝ) (new_people : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_people = 20 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ x : ℕ, x = 20 ∧
    (x : ℝ) * initial_avg + (new_people : ℝ) * new_avg = (x + new_people : ℝ) * final_avg :=
by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l524_52468


namespace NUMINAMATH_CALUDE_all_numbers_equal_l524_52437

/-- Represents a 10x10 table of real numbers -/
def Table := Fin 10 → Fin 10 → ℝ

/-- Predicate to check if a number is underlined in its row -/
def is_underlined_in_row (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, t i j ≥ t i k

/-- Predicate to check if a number is underlined in its column -/
def is_underlined_in_col (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, t i j ≤ t k j

/-- Predicate to check if a number is underlined exactly twice -/
def is_underlined_twice (t : Table) (i j : Fin 10) : Prop :=
  is_underlined_in_row t i j ∧ is_underlined_in_col t i j

theorem all_numbers_equal (t : Table) 
  (h : ∀ i j : Fin 10, is_underlined_in_row t i j ∨ is_underlined_in_col t i j → is_underlined_twice t i j) :
  ∀ i j k l : Fin 10, t i j = t k l :=
sorry

end NUMINAMATH_CALUDE_all_numbers_equal_l524_52437


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l524_52401

/-- Given a round trip with outbound speed of 96 mph and return speed of 88 mph,
    prove that the average speed for the entire trip is (2 * 96 * 88) / (96 + 88) mph. -/
theorem average_speed_round_trip (outbound_speed return_speed : ℝ) 
  (h1 : outbound_speed = 96) 
  (h2 : return_speed = 88) : 
  (2 * outbound_speed * return_speed) / (outbound_speed + return_speed) = 
  (2 * 96 * 88) / (96 + 88) :=
by sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l524_52401


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l524_52442

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

def is_perfect_square (x : ℚ) : Prop := ∃ y : ℚ, x = y^2

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 1 ∧
    is_perfect_cube (2002 * n) ∧
    is_perfect_square (n / 2002 : ℚ) ∧
    (∀ m : ℕ, m ≥ 1 →
      is_perfect_cube (2002 * m) →
      is_perfect_square (m / 2002 : ℚ) →
      n ≤ m) ∧
    n = 2002^5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l524_52442


namespace NUMINAMATH_CALUDE_handshake_remainder_l524_52475

/-- The number of ways 8 people can shake hands, where each person shakes hands with exactly 2 others -/
def M : ℕ := sorry

/-- The group size -/
def group_size : ℕ := 8

/-- The number of handshakes per person -/
def handshakes_per_person : ℕ := 2

theorem handshake_remainder : M ≡ 355 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_remainder_l524_52475


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l524_52471

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 * I + 1) / (1 - I) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l524_52471


namespace NUMINAMATH_CALUDE_tylers_age_l524_52491

theorem tylers_age (tyler clay : ℕ) 
  (h1 : tyler = 3 * clay + 1) 
  (h2 : tyler + clay = 21) : 
  tyler = 16 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l524_52491


namespace NUMINAMATH_CALUDE_min_value_z_l524_52494

theorem min_value_z (x y : ℝ) (h1 : y ≥ x + 2) (h2 : x + y ≤ 6) (h3 : x ≥ 1) :
  ∃ (z : ℝ), z = 2 * |x - 2| + |y| ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = 2 * |x - 2| + |y| → w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l524_52494


namespace NUMINAMATH_CALUDE_triangle_t_range_l524_52449

theorem triangle_t_range (a b c : ℝ) (A B C : ℝ) (t : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a * c = (1/4) * b^2 →
  Real.sin A + Real.sin C = t * Real.sin B →
  0 < B → B < Real.pi/2 →
  ∃ (t_min t_max : ℝ), t_min = Real.sqrt 6 / 2 ∧ t_max = Real.sqrt 2 ∧ t_min < t ∧ t < t_max :=
by sorry

end NUMINAMATH_CALUDE_triangle_t_range_l524_52449


namespace NUMINAMATH_CALUDE_curve_and_tangent_line_l524_52489

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define the line l1
def l1 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the property of point P
def P_property (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 3)^2 + y^2 = 4 * ((x - 3)^2 + y^2)

-- Define the minimization condition
def min_distance (Q M : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (mx, my) := M
  l1 qx qy ∧ C mx my ∧
  ∀ M' : ℝ × ℝ, C M'.1 M'.2 → (qx - mx)^2 + (qy - my)^2 ≤ (qx - M'.1)^2 + (qy - M'.2)^2

-- State the theorem
theorem curve_and_tangent_line :
  (∀ P : ℝ × ℝ, P_property P → C P.1 P.2) ∧
  (∀ Q M : ℝ × ℝ, min_distance Q M → (M.1 = 1 ∨ M.2 = -4)) :=
sorry

end NUMINAMATH_CALUDE_curve_and_tangent_line_l524_52489


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l524_52464

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem quadrilateral_diagonal_theorem (ABCD : Quadrilateral) (O : Point) :
  is_convex ABCD →
  distance ABCD.A ABCD.B = 10 →
  distance ABCD.C ABCD.D = 15 →
  distance ABCD.A ABCD.C = 20 →
  O = intersection ABCD.A ABCD.C ABCD.B ABCD.D →
  triangle_area ABCD.A O ABCD.D = triangle_area ABCD.B O ABCD.C →
  distance ABCD.A O = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l524_52464


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_l524_52481

theorem floor_sqrt_sum (a b c : ℝ) : 
  |a| = 4 → b^2 = 9 → c^3 = -8 → a > b → b > c → 
  ⌊Real.sqrt (a + b + c)⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_l524_52481


namespace NUMINAMATH_CALUDE_unique_positive_p_for_geometric_progression_l524_52480

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- The theorem states that 4 is the only positive real number p such that -p-12, 2√p, and p-5 form a geometric progression. -/
theorem unique_positive_p_for_geometric_progression :
  ∃! p : ℝ, p > 0 ∧ IsGeometricProgression (-p - 12) (2 * Real.sqrt p) (p - 5) :=
by
  sorry

#check unique_positive_p_for_geometric_progression

end NUMINAMATH_CALUDE_unique_positive_p_for_geometric_progression_l524_52480


namespace NUMINAMATH_CALUDE_sum_of_five_cubes_l524_52406

theorem sum_of_five_cubes (n : ℤ) : ∃ a b c d e : ℤ, n = a^3 + b^3 + c^3 + d^3 + e^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_cubes_l524_52406


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l524_52420

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 2*x + 3 > 0} = Set.Ioo (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l524_52420


namespace NUMINAMATH_CALUDE_part1_part2_l524_52447

-- Define the inequality function
def f (k x : ℝ) : ℝ := (k^2 - 2*k - 3)*x^2 - (k + 1)*x - 1

-- Define the solution set M
def M (k : ℝ) : Set ℝ := {x : ℝ | f k x < 0}

-- Part 1: Range of positive integer k when 1 ∈ M
theorem part1 : 
  (∀ k : ℕ+, 1 ∈ M k ↔ k ∈ ({1, 2, 3, 4} : Set ℕ+)) :=
sorry

-- Part 2: Range of real k when M = ℝ
theorem part2 : 
  (∀ k : ℝ, M k = Set.univ ↔ k ∈ Set.Icc (-1) (11/5)) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l524_52447


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l524_52418

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  b = (5/4) * a →          -- ratio of second to first angle is 5:4
  c = (9/4) * a →          -- ratio of third to first angle is 9:4
  a + b + c = 180 →        -- sum of angles in a triangle is 180°
  a = 40 :=                -- smallest angle is 40°
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l524_52418


namespace NUMINAMATH_CALUDE_remainder_sum_modulo_l524_52446

theorem remainder_sum_modulo (p q : ℤ) 
  (hp : p % 98 = 84) 
  (hq : q % 126 = 117) : 
  (p + q) % 42 = 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_modulo_l524_52446


namespace NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l524_52417

theorem square_difference_of_quadratic_solutions : 
  ∀ Φ φ : ℝ, Φ ≠ φ → Φ^2 = Φ + 2 → φ^2 = φ + 2 → (Φ - φ)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l524_52417


namespace NUMINAMATH_CALUDE_test_question_points_l524_52460

theorem test_question_points :
  ∀ (total_points total_questions two_point_questions : ℕ) 
    (other_question_points : ℚ),
  total_points = 100 →
  total_questions = 40 →
  two_point_questions = 30 →
  total_points = 2 * two_point_questions + (total_questions - two_point_questions) * other_question_points →
  other_question_points = 4 := by
sorry

end NUMINAMATH_CALUDE_test_question_points_l524_52460


namespace NUMINAMATH_CALUDE_average_of_four_data_points_l524_52440

theorem average_of_four_data_points
  (n : ℕ)
  (total_average : ℚ)
  (one_data_point : ℚ)
  (h1 : n = 5)
  (h2 : total_average = 81)
  (h3 : one_data_point = 85) :
  (n : ℚ) * total_average - one_data_point = (n - 1 : ℚ) * 80 :=
by sorry

end NUMINAMATH_CALUDE_average_of_four_data_points_l524_52440


namespace NUMINAMATH_CALUDE_magnitude_of_3_minus_i_l524_52472

/-- Given a complex number z = 3 - i, prove that its magnitude |z| is equal to √10 -/
theorem magnitude_of_3_minus_i :
  let z : ℂ := 3 - I
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_minus_i_l524_52472


namespace NUMINAMATH_CALUDE_parabola_intercept_minimum_l524_52403

/-- Parabola defined by x^2 = 8y -/
def Parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- Line with slope k passing through point (x, y) -/
def Line (k x y : ℝ) : Prop := y = k*x + 2

/-- Length of line segment intercepted by the parabola for a line with slope k -/
def InterceptLength (k : ℝ) : ℝ := 8*k^2 + 8

/-- The condition given in the problem relating k1 and k2 -/
def SlopeCondition (k1 k2 : ℝ) : Prop := 1/k1^2 + 4/k2^2 = 1

theorem parabola_intercept_minimum :
  ∀ k1 k2 : ℝ, 
  SlopeCondition k1 k2 →
  InterceptLength k1 + InterceptLength k2 ≥ 88 :=
sorry

end NUMINAMATH_CALUDE_parabola_intercept_minimum_l524_52403


namespace NUMINAMATH_CALUDE_problem_solution_l524_52499

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + 2*x - 3

-- Define the set M
def M : Set ℝ := {x | f x ≤ -1}

-- Theorem statement
theorem problem_solution :
  (M = {x : ℝ | x ≤ 0}) ∧
  (∀ x ∈ M, x * (f x)^2 - x^2 * (f x) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l524_52499


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l524_52443

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : y - 2 * x = 5) : 
  8 * x^3 * y - 8 * x^2 * y^2 + 2 * x * y^3 = -100 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l524_52443


namespace NUMINAMATH_CALUDE_problem_statement_l524_52424

theorem problem_statement (x y : ℝ) (h : x + y = 1) :
  (x^2 + 3*y^2 ≥ 3/4) ∧
  (x*y > 0 → ∀ a : ℝ, a ≤ 5/2 → 1/x + 1/y ≥ |a - 2| + |a + 1|) ∧
  (∀ a : ℝ, (x*y > 0 → 1/x + 1/y ≥ |a - 2| + |a + 1|) ↔ a ≤ 5/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l524_52424


namespace NUMINAMATH_CALUDE_min_bilingual_students_l524_52402

theorem min_bilingual_students (total : ℕ) (hindi : ℕ) (english : ℕ) 
  (h_total : total = 40)
  (h_hindi : hindi = 30)
  (h_english : english = 20) :
  ∃ (both : ℕ), both ≥ hindi + english - total ∧ 
  (∀ (x : ℕ), x ≥ hindi + english - total → x ≥ both) :=
by sorry

end NUMINAMATH_CALUDE_min_bilingual_students_l524_52402


namespace NUMINAMATH_CALUDE_nth_term_is_4021_l524_52408

/-- An arithmetic sequence with given first three terms -/
structure ArithmeticSequence (x : ℝ) where
  first_term : ℝ := 3 * x - 4
  second_term : ℝ := 6 * x - 17
  third_term : ℝ := 4 * x + 5
  is_arithmetic : second_term - first_term = third_term - second_term

/-- The nth term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence x) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

theorem nth_term_is_4021 (x : ℝ) (seq : ArithmeticSequence x) :
  ∃ n : ℕ, nth_term seq n = 4021 ∧ n = 502 := by
  sorry

end NUMINAMATH_CALUDE_nth_term_is_4021_l524_52408


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l524_52488

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) :
  n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l524_52488


namespace NUMINAMATH_CALUDE_square_area_74_l524_52455

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the square
def Square (p q : Point) :=
  {s : Set Point | ∃ (a b : ℝ), s = {(x, y) | min p.1 q.1 ≤ x ∧ x ≤ max p.1 q.1 ∧ min p.2 q.2 ≤ y ∧ y ≤ max p.2 q.2}}

-- Calculate the area of the square
def area (p q : Point) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem square_area_74 :
  let p : Point := (-2, -1)
  let q : Point := (3, 6)
  area p q = 74 := by
sorry

end NUMINAMATH_CALUDE_square_area_74_l524_52455


namespace NUMINAMATH_CALUDE_ten_times_average_sum_positions_elida_length_adrianna_length_l524_52486

/-- Represents the alphabetical position of a letter (A=1, B=2, ..., Z=26) -/
def alphabeticalPosition (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

/-- The name Elida -/
def elida : String := "ELIDA"

/-- The name Adrianna -/
def adrianna : String := "ADRIANNA"

/-- Sum of alphabetical positions of letters in a name -/
def sumAlphabeticalPositions (name : String) : ℕ :=
  name.toList.map alphabeticalPosition |>.sum

/-- Theorem stating that 10 times the average of the sum of alphabetical positions
    in both names is 465 -/
theorem ten_times_average_sum_positions : 
  (10 : ℚ) * ((sumAlphabeticalPositions elida + sumAlphabeticalPositions adrianna) / 2) = 465 := by
  sorry

/-- Elida has 5 letters -/
theorem elida_length : elida.length = 5 := by sorry

/-- Adrianna has 2 less than twice the number of letters Elida has -/
theorem adrianna_length : adrianna.length = 2 * elida.length - 2 := by sorry

end NUMINAMATH_CALUDE_ten_times_average_sum_positions_elida_length_adrianna_length_l524_52486


namespace NUMINAMATH_CALUDE_sum_of_roots_l524_52482

theorem sum_of_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 - a*x + 3*b = 0 ↔ x = a ∨ x = b) : 
  a + b = a :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l524_52482


namespace NUMINAMATH_CALUDE_alternative_basis_l524_52463

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given that {a, b, c} is a basis in space, prove that {c, a+b, a-b} is also a basis. -/
theorem alternative_basis
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![c, a+b, a-b] ∧
  Submodule.span ℝ {c, a+b, a-b} = ⊤ :=
sorry

end NUMINAMATH_CALUDE_alternative_basis_l524_52463


namespace NUMINAMATH_CALUDE_find_number_l524_52430

theorem find_number : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l524_52430


namespace NUMINAMATH_CALUDE_at_least_one_composite_l524_52477

theorem at_least_one_composite (a b c k : ℕ) 
  (ha : a ≥ 3) (hb : b ≥ 3) (hc : c ≥ 3) 
  (heq : a * b * c = k^2 + 1) : 
  ¬(Nat.Prime (a - 1) ∧ Nat.Prime (b - 1) ∧ Nat.Prime (c - 1)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_composite_l524_52477


namespace NUMINAMATH_CALUDE_largest_four_digit_prime_product_l524_52428

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_four_digit_prime_product :
  ∀ (n x y z : ℕ),
    n = x * y * z * (10 * x + y) →
    x < 20 ∧ y < 20 ∧ z < 20 →
    is_prime x ∧ is_prime y ∧ is_prime z →
    is_prime (10 * x + y) →
    x ≠ y ∧ x ≠ z ∧ y ≠ z →
    n ≤ 25058 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_prime_product_l524_52428


namespace NUMINAMATH_CALUDE_variance_transformation_l524_52469

variable {n : ℕ}
variable (a : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

def transformed_sample (a : Fin n → ℝ) : Fin n → ℝ := 
  fun i => 3 * a i + (if i.val = n - 1 then 2 else 1)

theorem variance_transformation (h : variance a = 3) : 
  variance (transformed_sample a) = 27 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l524_52469


namespace NUMINAMATH_CALUDE_triangle_inequality_l524_52415

/-- Given points P, Q, R, S on a line with PQ = a, PR = b, PS = c,
    if PQ and RS can be rotated to form a non-degenerate triangle,
    then a < c/2 and b < a + c/2 -/
theorem triangle_inequality (a b c : ℝ) 
  (h_order : 0 < a ∧ a < b ∧ b < c)
  (h_triangle : 2*b > c ∧ c > a ∧ c > b - a) :
  a < c/2 ∧ b < a + c/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l524_52415


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l524_52484

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 4 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l524_52484


namespace NUMINAMATH_CALUDE_function_value_range_bounds_are_tight_l524_52421

theorem function_value_range (x : ℝ) : 
  ∃ (y : ℝ), y = Real.sin x - Real.cos (x + π/6) ∧ 
  -Real.sqrt 3 ≤ y ∧ y ≤ Real.sqrt 3 :=
by sorry

theorem bounds_are_tight : 
  (∃ (x : ℝ), Real.sin x - Real.cos (x + π/6) = -Real.sqrt 3) ∧
  (∃ (x : ℝ), Real.sin x - Real.cos (x + π/6) = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_function_value_range_bounds_are_tight_l524_52421


namespace NUMINAMATH_CALUDE_point_on_line_l524_52479

/-- A point on a line in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines a line y = mx + c -/
structure Line2D where
  m : ℝ
  c : ℝ

/-- Theorem: If a point P(a,b) lies on the line y = -3x - 4, then b + 3a + 4 = 0 -/
theorem point_on_line (P : Point2D) (L : Line2D) 
  (h1 : L.m = -3)
  (h2 : L.c = -4)
  (h3 : P.y = L.m * P.x + L.c) :
  P.y + 3 * P.x + 4 = 0 := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l524_52479


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l524_52454

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3*a - 4) * (6*b - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l524_52454


namespace NUMINAMATH_CALUDE_distance_to_point_l524_52400

theorem distance_to_point : Real.sqrt (9^2 + (-40)^2) = 41 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l524_52400


namespace NUMINAMATH_CALUDE_max_y_over_x_l524_52422

theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) : 
  ∃ (max : ℝ), (∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 3 → y' / x' ≤ max) ∧ max = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_y_over_x_l524_52422


namespace NUMINAMATH_CALUDE_sean_julie_sum_ratio_l524_52445

/-- The sum of even integers from 2 to 600, inclusive -/
def sean_sum : ℕ := 2 * (300 * 301) / 2

/-- The sum of integers from 1 to 300, inclusive -/
def julie_sum : ℕ := (300 * 301) / 2

/-- Theorem stating that Sean's sum divided by Julie's sum equals 2 -/
theorem sean_julie_sum_ratio :
  (sean_sum : ℚ) / (julie_sum : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_sean_julie_sum_ratio_l524_52445


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l524_52476

theorem arccos_one_half_equals_pi_third : 
  Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l524_52476


namespace NUMINAMATH_CALUDE_sin_negative_45_degrees_l524_52451

theorem sin_negative_45_degrees :
  Real.sin (-(45 * π / 180)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_45_degrees_l524_52451


namespace NUMINAMATH_CALUDE_cl35_properties_neutron_calculation_electron_proton_equality_l524_52426

/-- Represents an atom with its atomic properties -/
structure Atom where
  protons : ℕ
  mass_number : ℕ
  neutrons : ℕ
  electrons : ℕ

/-- Cl-35 atom -/
def cl35 : Atom :=
  { protons := 17,
    mass_number := 35,
    neutrons := 35 - 17,
    electrons := 17 }

/-- Theorem stating the properties of Cl-35 -/
theorem cl35_properties :
  cl35.protons = 17 ∧
  cl35.mass_number = 35 ∧
  cl35.neutrons = 18 ∧
  cl35.electrons = 17 := by
  sorry

/-- Theorem stating the relationship between neutrons, mass number, and protons -/
theorem neutron_calculation (a : Atom) :
  a.neutrons = a.mass_number - a.protons := by
  sorry

/-- Theorem stating the relationship between electrons and protons -/
theorem electron_proton_equality (a : Atom) :
  a.electrons = a.protons := by
  sorry

end NUMINAMATH_CALUDE_cl35_properties_neutron_calculation_electron_proton_equality_l524_52426


namespace NUMINAMATH_CALUDE_river_current_speed_l524_52490

/-- Represents the speed of a motorboat in various conditions -/
structure MotorboatSpeed where
  still : ℝ  -- Speed in still water
  current : ℝ  -- River current speed
  wind : ℝ  -- Wind speed (positive for tailwind, negative for headwind)

/-- Calculates the effective speed of the motorboat -/
def effectiveSpeed (s : MotorboatSpeed) : ℝ := s.still + s.current + s.wind

/-- Theorem: River current speed is 1 mile per hour -/
theorem river_current_speed 
  (distance : ℝ) 
  (downstream_time upstream_time : ℝ) 
  (h : distance = 24 ∧ downstream_time = 4 ∧ upstream_time = 6) 
  (s : MotorboatSpeed) 
  (h_downstream : effectiveSpeed { still := s.still, current := s.current, wind := -s.wind } * downstream_time = distance) 
  (h_upstream : effectiveSpeed { still := s.still, current := -s.current, wind := s.wind } * upstream_time = distance) :
  s.current = 1 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l524_52490


namespace NUMINAMATH_CALUDE_equation_solution_l524_52458

theorem equation_solution (M : ℚ) : 
  (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M → M = 1003 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l524_52458


namespace NUMINAMATH_CALUDE_hockey_team_size_l524_52483

/-- Calculates the total number of players on a hockey team given specific conditions -/
theorem hockey_team_size 
  (percent_boys : ℚ)
  (num_junior_girls : ℕ)
  (h1 : percent_boys = 60 / 100)
  (h2 : num_junior_girls = 10) : 
  (2 * num_junior_girls : ℚ) / (1 - percent_boys) = 50 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_size_l524_52483


namespace NUMINAMATH_CALUDE_lizzies_group_difference_l524_52473

theorem lizzies_group_difference (total : ℕ) (lizzies_group : ℕ) : 
  total = 91 → lizzies_group = 54 → lizzies_group > (total - lizzies_group) → 
  lizzies_group - (total - lizzies_group) = 17 := by
sorry

end NUMINAMATH_CALUDE_lizzies_group_difference_l524_52473


namespace NUMINAMATH_CALUDE_expression_evaluation_l524_52452

theorem expression_evaluation :
  let x : ℤ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l524_52452


namespace NUMINAMATH_CALUDE_cards_traded_is_35_l524_52429

/-- The total number of cards traded between Padma and Robert -/
def total_cards_traded (padma_initial : ℕ) (robert_initial : ℕ) 
  (padma_first_trade : ℕ) (robert_first_trade : ℕ) 
  (padma_second_trade : ℕ) (robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + robert_first_trade + padma_second_trade + robert_second_trade

/-- Theorem stating the total number of cards traded is 35 -/
theorem cards_traded_is_35 : 
  total_cards_traded 75 88 2 10 15 8 = 35 := by
  sorry


end NUMINAMATH_CALUDE_cards_traded_is_35_l524_52429


namespace NUMINAMATH_CALUDE_pamphlet_cost_is_correct_l524_52409

/-- The cost of one pamphlet in dollars -/
def pamphlet_cost : ℝ := 1.11

/-- Condition 1: Nine copies cost less than $10.00 -/
axiom condition1 : 9 * pamphlet_cost < 10

/-- Condition 2: Ten copies cost more than $11.00 -/
axiom condition2 : 10 * pamphlet_cost > 11

/-- Theorem: The cost of one pamphlet is $1.11 -/
theorem pamphlet_cost_is_correct : pamphlet_cost = 1.11 := by
  sorry


end NUMINAMATH_CALUDE_pamphlet_cost_is_correct_l524_52409


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_minus_one_l524_52453

theorem gcd_of_powers_of_two_minus_one :
  Nat.gcd (2^2100 - 1) (2^1950 - 1) = 2^150 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_minus_one_l524_52453


namespace NUMINAMATH_CALUDE_problem_statement_l524_52459

theorem problem_statement :
  (∀ x : ℝ, x^2 - x ≥ x - 1) ∧
  (∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6) ∧
  (∀ x : ℝ, x > 2 → Real.sqrt (x^2 + 1) + 4 / Real.sqrt (x^2 + 1) ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l524_52459


namespace NUMINAMATH_CALUDE_robie_chocolates_l524_52450

theorem robie_chocolates (initial_bags : ℕ) : 
  (initial_bags - 2 + 3 = 4) → initial_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolates_l524_52450


namespace NUMINAMATH_CALUDE_hat_wearers_count_l524_52438

theorem hat_wearers_count (total_people adults children : ℕ)
  (adult_women adult_men : ℕ)
  (women_hat_percentage men_hat_percentage children_hat_percentage : ℚ) :
  total_people = adults + children →
  adults = adult_women + adult_men →
  adult_women = adult_men →
  women_hat_percentage = 25 / 100 →
  men_hat_percentage = 12 / 100 →
  children_hat_percentage = 10 / 100 →
  adults = 1800 →
  children = 200 →
  (adult_women * women_hat_percentage).floor +
  (adult_men * men_hat_percentage).floor +
  (children * children_hat_percentage).floor = 353 := by
sorry

end NUMINAMATH_CALUDE_hat_wearers_count_l524_52438


namespace NUMINAMATH_CALUDE_horner_first_step_value_l524_52405

/-- Horner's Method first step for polynomial evaluation -/
def horner_first_step (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  a 4 * x + a 3

/-- Polynomial coefficients -/
def f_coeff : ℕ → ℝ
  | 4 => 3
  | 3 => 0
  | 2 => 2
  | 1 => 1
  | 0 => 4
  | _ => 0

theorem horner_first_step_value :
  horner_first_step f_coeff 10 = 30 := by sorry

end NUMINAMATH_CALUDE_horner_first_step_value_l524_52405


namespace NUMINAMATH_CALUDE_four_positive_integers_sum_l524_52419

theorem four_positive_integers_sum (a b c d : ℕ+) 
  (sum1 : a + b + c = 6)
  (sum2 : a + b + d = 7)
  (sum3 : a + c + d = 8)
  (sum4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_positive_integers_sum_l524_52419


namespace NUMINAMATH_CALUDE_smoothies_from_fifteen_bananas_l524_52433

/-- The number of smoothies Caroline can make from a given number of bananas. -/
def smoothies_from_bananas (bananas : ℕ) : ℕ :=
  (9 * bananas) / 3

/-- Theorem stating that Caroline can make 45 smoothies from 15 bananas. -/
theorem smoothies_from_fifteen_bananas :
  smoothies_from_bananas 15 = 45 := by
  sorry

#eval smoothies_from_bananas 15

end NUMINAMATH_CALUDE_smoothies_from_fifteen_bananas_l524_52433


namespace NUMINAMATH_CALUDE_no_real_roots_l524_52441

theorem no_real_roots : ∀ x : ℝ, x^2 + x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l524_52441


namespace NUMINAMATH_CALUDE_probability_centrally_symmetric_shape_l524_52448

/-- Represents the shapes on the cards -/
inductive Shape
  | Circle
  | Rectangle
  | EquilateralTriangle
  | RegularPentagon

/-- Determines if a shape is centrally symmetric -/
def isCentrallySymmetric (s : Shape) : Bool :=
  match s with
  | Shape.Circle => true
  | Shape.Rectangle => true
  | Shape.EquilateralTriangle => false
  | Shape.RegularPentagon => false

/-- The set of all shapes -/
def allShapes : List Shape :=
  [Shape.Circle, Shape.Rectangle, Shape.EquilateralTriangle, Shape.RegularPentagon]

/-- Theorem: The probability of randomly selecting a centrally symmetric shape is 1/2 -/
theorem probability_centrally_symmetric_shape :
  (allShapes.filter isCentrallySymmetric).length / allShapes.length = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_centrally_symmetric_shape_l524_52448


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l524_52423

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure that the next person must sit next to someone -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats - 2) / 4 + 1

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l524_52423


namespace NUMINAMATH_CALUDE_one_root_implies_a_range_l524_52474

-- Define the function f(x) = 2x³ - 3x² + a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

-- Define the property of having only one root in [-2, 2]
def has_one_root_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Icc (-2) 2 ∧ f a x = 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 1 28

-- State the theorem
theorem one_root_implies_a_range :
  ∀ a : ℝ, has_one_root_in_interval a → a_range a := by
  sorry

end NUMINAMATH_CALUDE_one_root_implies_a_range_l524_52474


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l524_52466

theorem perfect_square_trinomial (a b t : ℝ) : 
  (∃ k : ℝ, a^2 + (2*t - 1)*a*b + 4*b^2 = (k*a + 2*b)^2) → 
  (t = 5/2 ∨ t = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l524_52466


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l524_52457

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 4*a^2 + 6*a - 8 = 0) ∧ 
  (b^3 - 4*b^2 + 6*b - 8 = 0) ∧ 
  (c^3 - 4*c^2 + 6*c - 8 = 0) →
  ((a + 3)^3 - 13*(a + 3)^2 + 57*(a + 3) - 89 = 0) ∧
  ((b + 3)^3 - 13*(b + 3)^2 + 57*(b + 3) - 89 = 0) ∧
  ((c + 3)^3 - 13*(c + 3)^2 + 57*(c + 3) - 89 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l524_52457


namespace NUMINAMATH_CALUDE_sum_of_coefficients_fifth_power_one_plus_sqrt_two_l524_52411

theorem sum_of_coefficients_fifth_power_one_plus_sqrt_two (a b : ℚ) : 
  (1 + Real.sqrt 2) ^ 5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_fifth_power_one_plus_sqrt_two_l524_52411


namespace NUMINAMATH_CALUDE_water_jars_count_l524_52427

/-- Given 35 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars,
    the total number of water-filled jars is 60. -/
theorem water_jars_count (total_volume : ℚ) (jar_sizes : Fin 3 → ℚ) :
  total_volume = 35 →
  jar_sizes 0 = 1/4 →
  jar_sizes 1 = 1/2 →
  jar_sizes 2 = 1 →
  (∃ (x : ℕ), total_volume = x * (jar_sizes 0 + jar_sizes 1 + jar_sizes 2)) →
  (∃ (x : ℕ), 3 * x = 60) :=
by sorry

end NUMINAMATH_CALUDE_water_jars_count_l524_52427


namespace NUMINAMATH_CALUDE_solution_range_l524_52456

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop :=
  3 * x - (a * x + 1) / 2 < 4 * x / 3

-- State the theorem
theorem solution_range (a : ℝ) : 
  (inequality 3 a) → a > 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_range_l524_52456


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l524_52495

/-- The total number of dogwood trees after planting -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating that the total number of dogwood trees after planting is 100 -/
theorem dogwood_tree_count :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l524_52495


namespace NUMINAMATH_CALUDE_average_of_first_21_multiples_of_6_l524_52439

/-- The average of the first n multiples of a number -/
def averageOfMultiples (n : ℕ) (x : ℕ) : ℚ :=
  (n * x * (n + 1)) / (2 * n)

/-- Theorem: The average of the first 21 multiples of 6 is 66 -/
theorem average_of_first_21_multiples_of_6 :
  averageOfMultiples 21 6 = 66 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_21_multiples_of_6_l524_52439


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l524_52436

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a (n + 1) < a n) →  -- decreasing sequence
  a 2 * a 4 * a 6 = 45 →
  a 2 + a 4 + a 6 = 15 →
  ∃ d : ℝ, d < 0 ∧ ∀ n, a n = a 1 + (n - 1) * d ∧ a n = -2 * n + 13 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l524_52436


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l524_52431

theorem complex_fraction_simplification :
  (7 + 8 * Complex.I) / (3 - 4 * Complex.I) = -11/25 + 52/25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l524_52431
