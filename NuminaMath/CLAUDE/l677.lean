import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l677_67757

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ -8) 
  (h2 : a ≠ 1) 
  (h3 : a ≠ -1) : 
  (9 / (a + 8) - (a^(1/3) + 2) / (a^(2/3) - 2*a^(1/3) + 4)) * 
  ((a^(4/3) + 8*a^(1/3)) / (1 - a^(2/3))) + 
  (5 - a^(2/3)) / (1 + a^(1/3)) = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l677_67757


namespace NUMINAMATH_CALUDE_chord_division_theorem_l677_67708

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an annulus -/
structure Annulus where
  inner : Circle
  outer : Circle

/-- Theorem: Given an annulus and a point inside it, there exists a chord passing through the point
    that divides the chord in a given ratio -/
theorem chord_division_theorem (A : Annulus) (P : Point) (m n : ℝ) 
    (h_concentric : A.inner.center = A.outer.center)
    (h_inside : (P.x - A.inner.center.x)^2 + (P.y - A.inner.center.y)^2 > A.inner.radius^2 ∧ 
                (P.x - A.outer.center.x)^2 + (P.y - A.outer.center.y)^2 < A.outer.radius^2)
    (h_positive : m > 0 ∧ n > 0) :
  ∃ (A₁ A₂ : Point),
    -- A₁ is on the inner circle
    (A₁.x - A.inner.center.x)^2 + (A₁.y - A.inner.center.y)^2 = A.inner.radius^2 ∧
    -- A₂ is on the outer circle
    (A₂.x - A.outer.center.x)^2 + (A₂.y - A.outer.center.y)^2 = A.outer.radius^2 ∧
    -- P is on the line segment A₁A₂
    ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P.x = A₁.x + t * (A₂.x - A₁.x) ∧ P.y = A₁.y + t * (A₂.y - A₁.y) ∧
    -- The ratio A₁P:PA₂ is m:n
    t / (1 - t) = m / n :=
by sorry

end NUMINAMATH_CALUDE_chord_division_theorem_l677_67708


namespace NUMINAMATH_CALUDE_min_value_of_w_min_value_achievable_l677_67739

theorem min_value_of_w (x y : ℝ) : 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27 ≥ 81/4 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27 = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_w_min_value_achievable_l677_67739


namespace NUMINAMATH_CALUDE_alex_shirts_l677_67741

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3) 
  (h2 : ben = joe + 8) 
  (h3 : ben = 15) : 
  alex = 4 := by
sorry

end NUMINAMATH_CALUDE_alex_shirts_l677_67741


namespace NUMINAMATH_CALUDE_pyramid_cases_l677_67722

/-- The sum of the first n triangular numbers -/
def sum_triangular (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2)) / 6

/-- The pyramid has 6 levels -/
def pyramid_levels : ℕ := 6

theorem pyramid_cases : sum_triangular pyramid_levels = 56 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cases_l677_67722


namespace NUMINAMATH_CALUDE_tank_capacity_l677_67792

/-- The capacity of a tank with specific leak and inlet properties -/
theorem tank_capacity 
  (leak_empty_time : ℝ) 
  (inlet_rate : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : leak_empty_time = 6) 
  (h2 : inlet_rate = 2.5) 
  (h3 : combined_empty_time = 8) : 
  ∃ C : ℝ, C = 3600 / 7 ∧ 
    C / leak_empty_time - inlet_rate * 60 = C / combined_empty_time :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l677_67792


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l677_67784

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c^2 + 9 * c - 21 = 0) → 
  (3 * d^2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l677_67784


namespace NUMINAMATH_CALUDE_negative_two_x_squared_cubed_l677_67768

theorem negative_two_x_squared_cubed (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_x_squared_cubed_l677_67768


namespace NUMINAMATH_CALUDE_bicycle_speed_l677_67759

/-- Given a journey with two modes of transport (on foot and by bicycle), 
    calculate the speed of the bicycle. -/
theorem bicycle_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (foot_distance : ℝ) 
  (foot_speed : ℝ) 
  (h1 : total_distance = 80) 
  (h2 : total_time = 7) 
  (h3 : foot_distance = 32) 
  (h4 : foot_speed = 8) :
  (total_distance - foot_distance) / (total_time - foot_distance / foot_speed) = 16 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_speed_l677_67759


namespace NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l677_67726

theorem infinite_solutions_diophantine_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → 
      x > 2008 ∧ y > 2008 ∧ z > 2008 ∧ 
      x^2 + y^2 + z^2 - x*y*z + 10 = 0) ∧
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l677_67726


namespace NUMINAMATH_CALUDE_average_and_square_difference_l677_67743

theorem average_and_square_difference (y : ℝ) : 
  (45 + y) / 2 = 50 → (y - 45)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_average_and_square_difference_l677_67743


namespace NUMINAMATH_CALUDE_fraction_simplification_l677_67719

theorem fraction_simplification : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 6) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l677_67719


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l677_67767

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + 3*I) / (-3 + 2*I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l677_67767


namespace NUMINAMATH_CALUDE_intersection_distance_l677_67734

/-- The distance between the intersection points of a circle and a line --/
theorem intersection_distance (x y : ℝ) : 
  x^2 + y^2 = 25 → 
  y = x + 3 → 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = 25 ∧ 
    y₁ = x₁ + 3 ∧ 
    x₂^2 + y₂^2 = 25 ∧ 
    y₂ = x₂ + 3 ∧ 
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 72 := by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_intersection_distance_l677_67734


namespace NUMINAMATH_CALUDE_min_value_of_xy_l677_67733

theorem min_value_of_xy (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h_geom : (Real.log x) * (Real.log y) = 1/4) : 
  ∀ z, x * y ≥ z → z ≤ Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_xy_l677_67733


namespace NUMINAMATH_CALUDE_min_hours_theorem_min_hours_sufficient_less_hours_insufficient_l677_67775

/-- Represents the minimum number of hours required for all friends to know all news -/
def min_hours (N : ℕ) : ℕ :=
  if N = 64 then 6
  else if N = 55 then 7
  else if N = 100 then 7
  else 0  -- undefined for other values of N

/-- The theorem stating the minimum number of hours for specific N values -/
theorem min_hours_theorem :
  (min_hours 64 = 6) ∧ (min_hours 55 = 7) ∧ (min_hours 100 = 7) := by
  sorry

/-- Helper function to calculate the maximum number of friends who can know a piece of news after h hours -/
def max_friends_knowing (h : ℕ) : ℕ := 2^h

/-- Theorem stating that the minimum hours is sufficient for all friends to know all news -/
theorem min_hours_sufficient (N : ℕ) (h : ℕ) (h_eq : h = min_hours N) :
  max_friends_knowing h ≥ N := by
  sorry

/-- Theorem stating that one less hour is insufficient for all friends to know all news -/
theorem less_hours_insufficient (N : ℕ) (h : ℕ) (h_eq : h = min_hours N) :
  max_friends_knowing (h - 1) < N := by
  sorry

end NUMINAMATH_CALUDE_min_hours_theorem_min_hours_sufficient_less_hours_insufficient_l677_67775


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l677_67712

/-- A quadratic function of the form f(x) = x^2 + c*x + d -/
def f (c d : ℝ) (x : ℝ) : ℝ := x^2 + c*x + d

/-- The theorem stating the uniqueness of c and d for the given condition -/
theorem quadratic_function_uniqueness :
  ∀ c d : ℝ,
  (∀ x : ℝ, (f c d (f c d x + 2*x)) / (f c d x) = 2*x^2 + 1984*x + 2024) →
  c = 1982 ∧ d = 21 := by
  sorry

#check quadratic_function_uniqueness

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l677_67712


namespace NUMINAMATH_CALUDE_derivative_of_f_l677_67744

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  (f ∘ Real.cos ∘ (fun x => 2 * x)) x = 1 - 2 * (Real.sin x) ^ 2 → 
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l677_67744


namespace NUMINAMATH_CALUDE_hyperbola_ac_range_l677_67707

-- Define the hyperbola and its properties
structure Hyperbola where
  focal_distance : ℝ
  a : ℝ
  left_focus : ℝ × ℝ
  right_focus : ℝ × ℝ
  point_on_right_branch : ℝ × ℝ

-- Define the theorem
theorem hyperbola_ac_range (h : Hyperbola) : 
  h.focal_distance = 4 → 
  h.a < 2 →
  let A := h.left_focus
  let B := h.right_focus
  let C := h.point_on_right_branch
  (dist C B - dist C A = 2 * h.a) →
  (dist A C + dist B C + dist A B = 10) →
  (3 < dist A C) ∧ (dist A C < 5) := by
  sorry

-- Note: dist is assumed to be a function that calculates the distance between two points

end NUMINAMATH_CALUDE_hyperbola_ac_range_l677_67707


namespace NUMINAMATH_CALUDE_factorization_3x_squared_minus_9x_l677_67727

theorem factorization_3x_squared_minus_9x (x : ℝ) : 3 * x^2 - 9 * x = 3 * x * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x_squared_minus_9x_l677_67727


namespace NUMINAMATH_CALUDE_readers_overlap_l677_67736

theorem readers_overlap (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : literary = 550) : 
  science_fiction + literary - total = 150 := by
  sorry

end NUMINAMATH_CALUDE_readers_overlap_l677_67736


namespace NUMINAMATH_CALUDE_morning_campers_l677_67795

theorem morning_campers (total : ℕ) (afternoon : ℕ) (morning : ℕ) : 
  total = 60 → afternoon = 7 → morning = total - afternoon → morning = 53 := by
sorry

end NUMINAMATH_CALUDE_morning_campers_l677_67795


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l677_67704

theorem cubic_equation_sum (r s t : ℝ) : 
  r^3 - 7*r^2 + 11*r = 13 →
  s^3 - 7*s^2 + 11*s = 13 →
  t^3 - 7*t^2 + 11*t = 13 →
  (r+s)/t + (s+t)/r + (t+r)/s = 38/13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l677_67704


namespace NUMINAMATH_CALUDE_v_2008_equals_3703_l677_67776

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ := sorry

/-- The 2008th term of the sequence v_n is 3703 -/
theorem v_2008_equals_3703 : v 2008 = 3703 := by sorry

end NUMINAMATH_CALUDE_v_2008_equals_3703_l677_67776


namespace NUMINAMATH_CALUDE_BE_length_l677_67738

-- Define the points
variable (A B C D E F G H : Point)

-- Define the square ABCD
def is_square (A B C D : Point) : Prop := sorry

-- Define that E is on the extension of BC
def on_extension (E B C : Point) : Prop := sorry

-- Define the square AEFG
def is_square_AEFG (A E F G : Point) : Prop := sorry

-- Define that A and G are on the same side of BE
def same_side (A G B E : Point) : Prop := sorry

-- Define that H is on the extension of BD and intersects AF
def intersects_extension (H B D A F : Point) : Prop := sorry

-- Define the lengths
def length (P Q : Point) : ℝ := sorry

-- State the theorem
theorem BE_length 
  (h1 : is_square A B C D)
  (h2 : on_extension E B C)
  (h3 : is_square_AEFG A E F G)
  (h4 : same_side A G B E)
  (h5 : intersects_extension H B D A F)
  (h6 : length H D = Real.sqrt 2)
  (h7 : length F H = 5 * Real.sqrt 2) :
  length B E = 8 := by sorry

end NUMINAMATH_CALUDE_BE_length_l677_67738


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l677_67731

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property (b : ℕ → ℝ) :
  is_geometric_sequence b →
  b 9 = (3 + 5) / 2 →
  b 1 * b 17 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l677_67731


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l677_67753

theorem quadratic_equation_roots (k : ℝ) (a b : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + k = 0 ↔ x = a ∨ x = b) →
  (a*b + 2*a + 2*b = 1) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l677_67753


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l677_67797

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l677_67797


namespace NUMINAMATH_CALUDE_number_problem_l677_67772

theorem number_problem : ∃ (x : ℝ), x = 40 ∧ 0.8 * x > (4/5 * 15 + 20) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l677_67772


namespace NUMINAMATH_CALUDE_expand_and_simplify_l677_67755

theorem expand_and_simplify (x y : ℝ) : (x + 2*y) * (2*x - 3*y) = 2*x^2 + x*y - 6*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l677_67755


namespace NUMINAMATH_CALUDE_divisibility_property_l677_67725

theorem divisibility_property (a b : ℤ) : (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l677_67725


namespace NUMINAMATH_CALUDE_box_height_l677_67764

/-- Calculates the height of a box with given specifications -/
theorem box_height (internal_volume : ℕ) (external_side_length : ℕ) : 
  internal_volume = 6912 ∧ 
  external_side_length = 26 → 
  (external_side_length - 2)^2 * 12 = internal_volume := by
  sorry

#check box_height

end NUMINAMATH_CALUDE_box_height_l677_67764


namespace NUMINAMATH_CALUDE_minesweeper_configurations_l677_67785

def valid_configuration (A B C D E : ℕ) : Prop :=
  A + B = 2 ∧ B + C + D = 1 ∧ D + E = 2

def count_configurations : ℕ := sorry

theorem minesweeper_configurations :
  count_configurations = 4545 := by sorry

end NUMINAMATH_CALUDE_minesweeper_configurations_l677_67785


namespace NUMINAMATH_CALUDE_fort_blocks_count_l677_67774

/-- Represents the dimensions of a rectangular structure -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular structure given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the specifications of the fort -/
structure FortSpecs where
  outerDimensions : Dimensions
  wallThickness : ℕ
  floorThickness : ℕ

/-- Calculates the inner dimensions of the fort given its specifications -/
def innerDimensions (specs : FortSpecs) : Dimensions :=
  { length := specs.outerDimensions.length - 2 * specs.wallThickness,
    width := specs.outerDimensions.width - 2 * specs.wallThickness,
    height := specs.outerDimensions.height - specs.floorThickness }

/-- Calculates the number of blocks needed for the fort -/
def blocksNeeded (specs : FortSpecs) : ℕ :=
  volume specs.outerDimensions - volume (innerDimensions specs)

theorem fort_blocks_count : 
  let fortSpecs : FortSpecs := 
    { outerDimensions := { length := 20, width := 15, height := 8 },
      wallThickness := 2,
      floorThickness := 1 }
  blocksNeeded fortSpecs = 1168 := by sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l677_67774


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l677_67718

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fourth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_3 : a 3 = 2) 
  (h_5 : a 5 = 16) : 
  a 4 = 4 * Real.sqrt 2 ∨ a 4 = -4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l677_67718


namespace NUMINAMATH_CALUDE_linear_function_positive_sum_product_inequality_l677_67756

-- Define the linear function
def f (k h : ℝ) (x : ℝ) : ℝ := k * x + h

-- Theorem for the first part
theorem linear_function_positive (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n) 
  (hfm : f k h m > 0) (hfn : f k h n > 0) :
  ∀ x, m < x ∧ x < n → f k h x > 0 := by sorry

-- Theorem for the second part
theorem sum_product_inequality (a b c : ℝ) 
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) :
  a * b + b * c + c * a > -1 := by sorry

end NUMINAMATH_CALUDE_linear_function_positive_sum_product_inequality_l677_67756


namespace NUMINAMATH_CALUDE_bus_passengers_after_three_stops_l677_67777

theorem bus_passengers_after_three_stops : 
  let initial_passengers := 0
  let first_stop_on := 7
  let second_stop_off := 3
  let second_stop_on := 5
  let third_stop_off := 2
  let third_stop_on := 4
  
  let after_first_stop := initial_passengers + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  
  after_third_stop = 11 := by sorry

end NUMINAMATH_CALUDE_bus_passengers_after_three_stops_l677_67777


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l677_67763

theorem rectangle_area_problem (w : ℝ) (L L' A : ℝ) : 
  w = 10 →                      -- Width is 10 m
  A = L * w →                   -- Original area
  L' * w = (4/3) * A →          -- New area is 1 1/3 times original
  2 * L' + 2 * w = 60 →         -- New perimeter is 60 m
  A = 150                       -- Original area is 150 square meters
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l677_67763


namespace NUMINAMATH_CALUDE_regular_polygon_with_405_diagonals_has_30_sides_l677_67721

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 405 diagonals has 30 sides -/
theorem regular_polygon_with_405_diagonals_has_30_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 405 → n = 30 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_405_diagonals_has_30_sides_l677_67721


namespace NUMINAMATH_CALUDE_unique_intersection_l677_67799

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- State the theorem
theorem unique_intersection :
  ∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l677_67799


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l677_67789

theorem smallest_integer_with_remainder_two (n : ℕ) : 
  (n > 1) →
  (n % 3 = 2) →
  (n % 5 = 2) →
  (n % 7 = 2) →
  (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 5 = 2 ∧ m % 7 = 2 → m ≥ n) →
  n = 107 :=
by
  sorry

#check smallest_integer_with_remainder_two

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l677_67789


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_l677_67724

theorem triangle_inequality_condition (m : ℝ) :
  (m > 0) →
  (∀ (x y : ℝ), x > 0 → y > 0 →
    (x + y + m * Real.sqrt (x * y) > Real.sqrt (x^2 + y^2 + x * y) ∧
     x + y + Real.sqrt (x^2 + y^2 + x * y) > m * Real.sqrt (x * y) ∧
     m * Real.sqrt (x * y) + Real.sqrt (x^2 + y^2 + x * y) > x + y)) ↔
  (m > 2 - Real.sqrt 3 ∧ m < 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_condition_l677_67724


namespace NUMINAMATH_CALUDE_shannon_stones_l677_67791

/-- The number of heart-shaped stones Shannon wants in each bracelet -/
def stones_per_bracelet : ℕ := 8

/-- The number of bracelets Shannon can make -/
def number_of_bracelets : ℕ := 6

/-- The total number of heart-shaped stones Shannon brought -/
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem shannon_stones : total_stones = 48 := by
  sorry

end NUMINAMATH_CALUDE_shannon_stones_l677_67791


namespace NUMINAMATH_CALUDE_unique_hyperbolas_l677_67771

/-- Binomial coefficient function -/
def binomial (m n : ℕ) : ℕ := Nat.choose m n

/-- The set of binomial coefficients for 1 ≤ n ≤ m ≤ 5 -/
def binomial_set : Finset ℕ :=
  Finset.filter (λ x => x > 1) $
    Finset.image (λ (m, n) => binomial m n) $
      Finset.filter (λ (m, n) => 1 ≤ n ∧ n ≤ m ∧ m ≤ 5) $
        Finset.product (Finset.range 6) (Finset.range 6)

theorem unique_hyperbolas : Finset.card binomial_set = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_hyperbolas_l677_67771


namespace NUMINAMATH_CALUDE_scoop_size_l677_67746

/-- Given the total amount of ingredients and the total number of scoops, 
    calculate the size of each scoop. -/
theorem scoop_size (total_cups : ℚ) (total_scoops : ℕ) 
  (h1 : total_cups = 5) 
  (h2 : total_scoops = 15) : 
  total_cups / total_scoops = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_scoop_size_l677_67746


namespace NUMINAMATH_CALUDE_sector_area_l677_67735

theorem sector_area (θ : Real) (s : Real) (A : Real) :
  θ = 2 ∧ s = 4 → A = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l677_67735


namespace NUMINAMATH_CALUDE_pentagon_area_l677_67782

/-- The area of a specific pentagon -/
theorem pentagon_area : 
  ∀ (pentagon_sides : List ℝ) 
    (trapezoid_bases : List ℝ) 
    (trapezoid_height : ℝ) 
    (triangle_base : ℝ) 
    (triangle_height : ℝ),
  pentagon_sides = [18, 25, 30, 28, 25] →
  trapezoid_bases = [25, 28] →
  trapezoid_height = 30 →
  triangle_base = 18 →
  triangle_height = 24 →
  (1/2 * (trapezoid_bases.sum) * trapezoid_height) + (1/2 * triangle_base * triangle_height) = 1011 := by
sorry


end NUMINAMATH_CALUDE_pentagon_area_l677_67782


namespace NUMINAMATH_CALUDE_min_value_problem_l677_67798

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + 3 * b = 1) :
  (2 / a) + (3 / b) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l677_67798


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l677_67742

theorem cos_two_theta_value (θ : ℝ) 
  (h1 : 3 * Real.sin (2 * θ) = 4 * Real.tan θ) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.cos (2 * θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l677_67742


namespace NUMINAMATH_CALUDE_polynomial_roots_problem_l677_67750

theorem polynomial_roots_problem (c d : ℤ) (h1 : c ≠ 0) (h2 : d ≠ 0) 
  (h3 : ∃ p q : ℤ, (X - p)^2 * (X - q) = X^3 + c*X^2 + d*X + 12*c) : 
  |c * d| = 192 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_problem_l677_67750


namespace NUMINAMATH_CALUDE_list_number_property_l677_67788

theorem list_number_property (L : List ℝ) (n : ℝ) :
  L.length = 21 →
  L.Nodup →
  n ∈ L →
  n = 0.2 * L.sum →
  n = 5 * ((L.sum - n) / 20) :=
by sorry

end NUMINAMATH_CALUDE_list_number_property_l677_67788


namespace NUMINAMATH_CALUDE_face_value_of_shares_l677_67751

/-- Theorem: Face value of shares given investment and dividend information -/
theorem face_value_of_shares 
  (investment : ℝ) 
  (premium_rate : ℝ) 
  (dividend_rate : ℝ) 
  (dividend_amount : ℝ) 
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_amount = 720) :
  ∃ (face_value : ℝ), 
    face_value = 12000 ∧ 
    investment = face_value * (1 + premium_rate) ∧
    dividend_amount = face_value * dividend_rate :=
by sorry

end NUMINAMATH_CALUDE_face_value_of_shares_l677_67751


namespace NUMINAMATH_CALUDE_ending_number_divisible_by_eleven_l677_67766

theorem ending_number_divisible_by_eleven (start : Nat) (count : Nat) : 
  start ≥ 29 →
  start % 11 = 0 →
  count = 5 →
  ∀ k, k ∈ Finset.range count → (start + k * 11) % 11 = 0 →
  start + (count - 1) * 11 = 77 :=
by sorry

end NUMINAMATH_CALUDE_ending_number_divisible_by_eleven_l677_67766


namespace NUMINAMATH_CALUDE_molecular_weight_of_BaSO4_l677_67713

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.327

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of moles of BaSO4 -/
def moles_BaSO4 : ℝ := 3

/-- The molecular weight of BaSO4 in g/mol -/
def molecular_weight_BaSO4 : ℝ := atomic_weight_Ba + atomic_weight_S + 4 * atomic_weight_O

/-- The total weight of the given moles of BaSO4 in grams -/
def total_weight_BaSO4 : ℝ := moles_BaSO4 * molecular_weight_BaSO4

theorem molecular_weight_of_BaSO4 :
  total_weight_BaSO4 = 700.164 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_of_BaSO4_l677_67713


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l677_67701

theorem quadratic_roots_relation (a b c d : ℝ) (h : a ≠ 0 ∧ c ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ c * (x/2007)^2 + d * (x/2007) + a = 0) →
  b^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l677_67701


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l677_67728

theorem solve_exponential_equation :
  ∃ x : ℝ, 3^(3*x) = Real.sqrt 81 ∧ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l677_67728


namespace NUMINAMATH_CALUDE_average_problem_l677_67747

theorem average_problem (c d e : ℝ) : 
  (4 + 6 + 9 + c + d + e) / 6 = 20 → (c + d + e) / 3 = 101 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l677_67747


namespace NUMINAMATH_CALUDE_sin_two_x_value_l677_67790

theorem sin_two_x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 6) : 
  Real.sin (2 * x) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_x_value_l677_67790


namespace NUMINAMATH_CALUDE_line_relationship_l677_67706

-- Define the concept of lines in 3D space
structure Line3D where
  -- This is a placeholder definition. In a real implementation, 
  -- we might represent a line using a point and a direction vector.
  id : ℕ

-- Define the relationships between lines
def are_skew (l1 l2 : Line3D) : Prop := sorry
def are_parallel (l1 l2 : Line3D) : Prop := sorry
def are_intersecting (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem line_relationship (a b c : Line3D) 
  (h1 : are_skew a b) (h2 : are_parallel a c) : 
  are_intersecting b c ∨ are_skew b c := by sorry

end NUMINAMATH_CALUDE_line_relationship_l677_67706


namespace NUMINAMATH_CALUDE_sum_of_digits_1948_base9_l677_67714

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of a list of natural numbers -/
def sum (l : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_1948_base9 :
  sum (toBase9 1948) = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_1948_base9_l677_67714


namespace NUMINAMATH_CALUDE_scale_division_l677_67745

/-- Given a scale of length 80 inches divided into 5 equal parts, 
    prove that the length of each part is 16 inches. -/
theorem scale_division (total_length : ℕ) (num_parts : ℕ) (part_length : ℕ) 
  (h1 : total_length = 80) 
  (h2 : num_parts = 5) 
  (h3 : part_length * num_parts = total_length) : 
  part_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l677_67745


namespace NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l677_67794

/-- Given a rectangle with side lengths a and b, and a triangle constructed on its diagonal
    as base with area equal to the rectangle's area, the altitude of the triangle is
    (2 * a * b) / sqrt(a^2 + b^2). -/
theorem triangle_altitude_on_rectangle_diagonal 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ (h : ℝ), h = (2 * a * b) / Real.sqrt (a^2 + b^2) ∧ 
  h * Real.sqrt (a^2 + b^2) / 2 = a * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l677_67794


namespace NUMINAMATH_CALUDE_fraction_inequality_l677_67720

theorem fraction_inequality (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) ≤ 1) ↔ (x < 1 ∨ x ≥ 2) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l677_67720


namespace NUMINAMATH_CALUDE_antonio_age_is_51_months_l677_67709

/- Define Isabella's current age in months -/
def isabella_age_months : ℕ := 10 * 12 - 18

/- Define the relationship between Isabella's and Antonio's ages -/
def antonio_age_months : ℕ := isabella_age_months / 2

/- Theorem stating Antonio's age in months -/
theorem antonio_age_is_51_months : antonio_age_months = 51 := by
  sorry

end NUMINAMATH_CALUDE_antonio_age_is_51_months_l677_67709


namespace NUMINAMATH_CALUDE_power_multiplication_equality_l677_67700

theorem power_multiplication_equality (m : ℝ) : m^2 * (-m)^4 = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equality_l677_67700


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l677_67787

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 8

-- Define the points and foci
variable (P Q F₁ F₂ : ℝ × ℝ)

-- Define the chord passing through left focus
def chord_through_left_focus : Prop := 
  (∃ t : ℝ, P = F₁ + t • (Q - F₁)) ∨ (∃ t : ℝ, Q = F₁ + t • (P - F₁))

-- Define the length of PQ
def PQ_length : Prop := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 7

-- Define F₂ as the right focus
def right_focus (F₂ : ℝ × ℝ) : Prop :=
  F₂.1 > 0 ∧ F₂.1^2 - F₂.2^2 = 8

-- Theorem statement
theorem hyperbola_triangle_perimeter 
  (h_hyperbola_P : hyperbola P.1 P.2)
  (h_hyperbola_Q : hyperbola Q.1 Q.2)
  (h_chord : chord_through_left_focus P Q F₁)
  (h_PQ_length : PQ_length P Q)
  (h_right_focus : right_focus F₂) :
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) +
  Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) +
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) =
  14 + 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l677_67787


namespace NUMINAMATH_CALUDE_mobile_phone_costs_and_schemes_l677_67748

/-- Given the cost equations for mobile phones, this theorem proves the costs of each type
    and the number of valid purchasing schemes. -/
theorem mobile_phone_costs_and_schemes :
  ∃ (cost_A cost_B : ℕ) (num_schemes : ℕ),
    -- Cost equations
    (2 * cost_A + 3 * cost_B = 7400) ∧
    (3 * cost_A + 5 * cost_B = 11700) ∧
    -- Costs of phones
    (cost_A = 1900) ∧
    (cost_B = 1200) ∧
    -- Number of valid purchasing schemes
    (num_schemes = 9) ∧
    -- Definition of valid purchasing schemes
    (∀ m : ℕ, 
      (12 ≤ m ∧ m ≤ 20) ↔ 
      (44400 ≤ 1900*m + 1200*(30-m) ∧ 1900*m + 1200*(30-m) ≤ 50000)) := by
  sorry


end NUMINAMATH_CALUDE_mobile_phone_costs_and_schemes_l677_67748


namespace NUMINAMATH_CALUDE_four_Z_one_equals_five_l677_67780

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^2 - 3*a*b + b^2

-- Theorem statement
theorem four_Z_one_equals_five : Z 4 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_one_equals_five_l677_67780


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_squared_minus_one_l677_67740

theorem imaginary_part_of_z_squared_minus_one (z : ℂ) :
  z = 1 + Complex.I →
  Complex.im ((z + 1) * (z - 1)) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_squared_minus_one_l677_67740


namespace NUMINAMATH_CALUDE_carmelas_initial_money_l677_67705

/-- Proves that Carmela's initial amount of money is $7 given the problem conditions --/
theorem carmelas_initial_money :
  ∀ x : ℕ,
  (∃ (final_amount : ℕ),
    -- Carmela's final amount after giving $1 to each of 4 cousins
    x - 4 = final_amount ∧
    -- Each cousin's final amount after receiving $1
    2 + 1 = final_amount) →
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_carmelas_initial_money_l677_67705


namespace NUMINAMATH_CALUDE_digit_selection_theorem_l677_67702

/-- The number of digits available for selection -/
def n : ℕ := 10

/-- The number of digits to be selected -/
def k : ℕ := 4

/-- Function to calculate the number of permutations without repetition -/
def permutations_without_repetition (n k : ℕ) : ℕ := sorry

/-- Function to calculate the number of four-digit numbers without repetition -/
def four_digit_numbers_without_repetition (n k : ℕ) : ℕ := sorry

/-- Function to calculate the number of even four-digit numbers greater than 3000 without repetition -/
def even_four_digit_numbers_gt_3000_without_repetition (n k : ℕ) : ℕ := sorry

theorem digit_selection_theorem :
  permutations_without_repetition n k = 5040 ∧
  four_digit_numbers_without_repetition n k = 4356 ∧
  even_four_digit_numbers_gt_3000_without_repetition n k = 1792 := by
  sorry

end NUMINAMATH_CALUDE_digit_selection_theorem_l677_67702


namespace NUMINAMATH_CALUDE_opposite_of_neg_one_half_l677_67703

/-- The opposite of a rational number -/
def opposite (x : ℚ) : ℚ := -x

/-- The property that defines the opposite of a number -/
def is_opposite (x y : ℚ) : Prop := x + y = 0

theorem opposite_of_neg_one_half :
  is_opposite (-1/2 : ℚ) (1/2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_neg_one_half_l677_67703


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l677_67760

theorem coefficient_x_squared_in_expansion : 
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (2^k) * if k = 2 then 1 else 0) = 180 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l677_67760


namespace NUMINAMATH_CALUDE_cube_root_1600_l677_67779

theorem cube_root_1600 (c d : ℕ+) (h1 : (1600 : ℝ)^(1/3) = c * d^(1/3)) 
  (h2 : ∀ (c' d' : ℕ+), (1600 : ℝ)^(1/3) = c' * d'^(1/3) → d ≤ d') : 
  c + d = 29 := by
sorry

end NUMINAMATH_CALUDE_cube_root_1600_l677_67779


namespace NUMINAMATH_CALUDE_carnival_spending_theorem_l677_67754

def carnival_spending (bumper_car_rides : ℕ) (space_shuttle_rides : ℕ) (ferris_wheel_rides : ℕ)
  (bumper_car_cost : ℕ) (space_shuttle_cost : ℕ) (ferris_wheel_cost : ℕ) : ℕ :=
  bumper_car_rides * bumper_car_cost +
  space_shuttle_rides * space_shuttle_cost +
  2 * ferris_wheel_rides * ferris_wheel_cost

theorem carnival_spending_theorem :
  carnival_spending 2 4 3 2 4 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_carnival_spending_theorem_l677_67754


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l677_67716

-- Define the operation
def determinant (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0

-- State the theorem
theorem complex_in_first_quadrant :
  ∃ z : ℂ, determinant z (1 + Complex.I) 2 1 = 0 ∧ is_in_first_quadrant z :=
sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l677_67716


namespace NUMINAMATH_CALUDE_ball_color_probability_l677_67769

/-- The number of balls -/
def n : ℕ := 8

/-- The probability of a ball being painted black or white -/
def p : ℚ := 1/2

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k successes in n independent trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

theorem ball_color_probability :
  binomial_probability n (n/2) p = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_ball_color_probability_l677_67769


namespace NUMINAMATH_CALUDE_triple_counted_number_l677_67723

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n % 10) % 5 = 0

def sum_valid_numbers : ℕ := sorry

theorem triple_counted_number (triple_counted : ℕ) 
  (h1 : is_valid_number triple_counted)
  (h2 : sum_valid_numbers + 2 * triple_counted = 1035) :
  triple_counted = 45 := by sorry

end NUMINAMATH_CALUDE_triple_counted_number_l677_67723


namespace NUMINAMATH_CALUDE_milk_production_theorem_l677_67770

/-- Represents the milk production scenario -/
structure MilkProduction where
  initial_cows : ℕ
  initial_days : ℕ
  initial_gallons : ℕ
  max_daily_per_cow : ℕ
  available_cows : ℕ
  target_days : ℕ

/-- Calculates the total milk production given the scenario -/
def total_milk_production (mp : MilkProduction) : ℕ :=
  let daily_rate_per_cow := mp.initial_gallons / (mp.initial_cows * mp.initial_days)
  let actual_rate := min daily_rate_per_cow mp.max_daily_per_cow
  mp.available_cows * actual_rate * mp.target_days

/-- Theorem stating that the total milk production is 96 gallons -/
theorem milk_production_theorem (mp : MilkProduction) 
  (h1 : mp.initial_cows = 10)
  (h2 : mp.initial_days = 5)
  (h3 : mp.initial_gallons = 40)
  (h4 : mp.max_daily_per_cow = 2)
  (h5 : mp.available_cows = 15)
  (h6 : mp.target_days = 8) :
  total_milk_production mp = 96 := by
  sorry

end NUMINAMATH_CALUDE_milk_production_theorem_l677_67770


namespace NUMINAMATH_CALUDE_expression_value_l677_67762

theorem expression_value : 
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l677_67762


namespace NUMINAMATH_CALUDE_ball_attendance_l677_67711

theorem ball_attendance :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by sorry

end NUMINAMATH_CALUDE_ball_attendance_l677_67711


namespace NUMINAMATH_CALUDE_rectangle_sides_theorem_l677_67796

/-- A pair of positive integers representing the sides of a rectangle --/
structure RectangleSides where
  x : ℕ+
  y : ℕ+

/-- The set of rectangle sides that satisfy the perimeter-area equality condition --/
def validRectangleSides : Set RectangleSides :=
  { sides | (2 * sides.x.val + 2 * sides.y.val : ℕ) = sides.x.val * sides.y.val }

/-- The theorem stating that only three specific pairs of sides satisfy the conditions --/
theorem rectangle_sides_theorem :
  validRectangleSides = {⟨3, 6⟩, ⟨6, 3⟩, ⟨4, 4⟩} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_theorem_l677_67796


namespace NUMINAMATH_CALUDE_car_speed_l677_67758

/-- Given a car that travels 325 miles in 5 hours, its speed is 65 miles per hour -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 325) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : speed = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l677_67758


namespace NUMINAMATH_CALUDE_parabola_vertices_distance_l677_67749

/-- Given an equation representing portions of two parabolas, 
    this theorem states the distance between their vertices. -/
theorem parabola_vertices_distance : 
  ∃ (f g : ℝ → ℝ),
    (∀ x y : ℝ, (Real.sqrt (x^2 + y^2) + |y + 2| = 4) ↔ 
      ((y ≥ -2 ∧ y = f x) ∨ (y < -2 ∧ y = g x))) →
    ∃ (v1 v2 : ℝ × ℝ),
      (v1.1 = 0 ∧ v1.2 = f 0) ∧
      (v2.1 = 0 ∧ v2.2 = g 0) ∧
      Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 58 / 11 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertices_distance_l677_67749


namespace NUMINAMATH_CALUDE_m_range_l677_67778

-- Define the condition on x
def X := { x : ℝ | x ≤ -1 }

-- Define the inequality condition
def inequality (m : ℝ) : Prop :=
  ∀ x ∈ X, (m^2 - m) * 4^x - 2^x < 0

-- Theorem statement
theorem m_range :
  ∀ m : ℝ, inequality m ↔ -1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_m_range_l677_67778


namespace NUMINAMATH_CALUDE_smallest_union_size_l677_67715

theorem smallest_union_size (A B : Finset ℕ) 
  (hA : A.card = 30)
  (hB : B.card = 20)
  (hInter : (A ∩ B).card ≥ 10) :
  (A ∪ B).card ≥ 40 ∧ ∃ (C D : Finset ℕ), C.card = 30 ∧ D.card = 20 ∧ (C ∩ D).card ≥ 10 ∧ (C ∪ D).card = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_union_size_l677_67715


namespace NUMINAMATH_CALUDE_sequence_a_4_equals_zero_l677_67786

theorem sequence_a_4_equals_zero :
  let a : ℕ+ → ℤ := fun n => n.val^2 - 3*n.val - 4
  a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_4_equals_zero_l677_67786


namespace NUMINAMATH_CALUDE_platform_length_l677_67730

/-- The length of a platform given train speed and crossing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) : 
  train_speed = 72 →
  platform_time = 30 →
  man_time = 19 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 220 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l677_67730


namespace NUMINAMATH_CALUDE_event_probability_l677_67710

theorem event_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  (3 * p * (1 - p)^2 = 9/64) :=
by
  sorry

end NUMINAMATH_CALUDE_event_probability_l677_67710


namespace NUMINAMATH_CALUDE_tangent_sum_l677_67781

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem tangent_sum (A B : ℝ) 
  (h1 : tg A + tg B = 2) 
  (h2 : ctg A + ctg B = 3) : 
  tg (A + B) = 6 := by sorry

end NUMINAMATH_CALUDE_tangent_sum_l677_67781


namespace NUMINAMATH_CALUDE_time_to_find_worm_l677_67765

/-- Given Kevin's toad feeding scenario, prove the time to find each worm. -/
theorem time_to_find_worm (num_toads : ℕ) (worms_per_toad : ℕ) (total_hours : ℕ) :
  num_toads = 8 →
  worms_per_toad = 3 →
  total_hours = 6 →
  (total_hours * 60) / (num_toads * worms_per_toad) = 15 :=
by sorry

end NUMINAMATH_CALUDE_time_to_find_worm_l677_67765


namespace NUMINAMATH_CALUDE_range_of_x_l677_67793

def p (x : ℝ) := 1 / (x - 2) < 0
def q (x : ℝ) := x^2 - 4*x - 5 < 0

theorem range_of_x (x : ℝ) :
  (p x ∨ q x) ∧ ¬(p x ∧ q x) →
  x ∈ Set.Iic (-1) ∪ Set.Ico 3 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l677_67793


namespace NUMINAMATH_CALUDE_max_value_not_one_l677_67729

theorem max_value_not_one :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x + π/4)
  let g : ℝ → ℝ := λ x ↦ Real.cos (x - π/4)
  let y : ℝ → ℝ := λ x ↦ f x * g x
  ∃ M : ℝ, (∀ x, y x ≤ M) ∧ M < 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_not_one_l677_67729


namespace NUMINAMATH_CALUDE_bubble_gum_count_l677_67732

/-- The cost of a single piece of bubble gum in cents -/
def cost_per_piece : ℕ := 18

/-- The total cost of all pieces of bubble gum in cents -/
def total_cost : ℕ := 2448

/-- The number of pieces of bubble gum -/
def num_pieces : ℕ := total_cost / cost_per_piece

theorem bubble_gum_count : num_pieces = 136 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_count_l677_67732


namespace NUMINAMATH_CALUDE_runners_meet_time_l677_67773

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.minutes + m
  { hours := t.hours + totalMinutes / 60
    minutes := totalMinutes % 60 }

theorem runners_meet_time (startTime : Time) (lapTime1 lapTime2 lapTime3 : Nat) : 
  startTime.hours = 7 ∧ startTime.minutes = 45 ∧
  lapTime1 = 5 ∧ lapTime2 = 8 ∧ lapTime3 = 10 →
  let meetTime := addMinutes startTime (Nat.lcm lapTime1 (Nat.lcm lapTime2 lapTime3))
  meetTime.hours = 8 ∧ meetTime.minutes = 25 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_time_l677_67773


namespace NUMINAMATH_CALUDE_betty_oranges_l677_67761

/-- The number of boxes Betty has -/
def num_boxes : ℕ := 3

/-- The number of oranges in each box -/
def oranges_per_box : ℕ := 8

/-- The total number of oranges Betty has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem betty_oranges : total_oranges = 24 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l677_67761


namespace NUMINAMATH_CALUDE_parabola_vertex_l677_67737

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 2)^2 - 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -5)

/-- Theorem: The vertex of the parabola y = -2(x-2)^2 - 5 is at the point (2, -5) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l677_67737


namespace NUMINAMATH_CALUDE_minimal_distance_point_l677_67783

/-- Given points A, B, and C in the xy-plane, prove that the value of m that minimizes 
    the sum of distances AC + CB is -7/5 when C is constrained to the y-axis. -/
theorem minimal_distance_point (A B C : ℝ × ℝ) : 
  A = (-2, -3) → 
  B = (3, 1) → 
  C.1 = 0 →
  (∀ m' : ℝ, dist A C + dist C B ≤ dist A (0, m') + dist (0, m') B) →
  C.2 = -7/5 := by
sorry

end NUMINAMATH_CALUDE_minimal_distance_point_l677_67783


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l677_67717

theorem consecutive_numbers_product (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + d = 109) → (b * c = 2970) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l677_67717


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_fibonacci_factorial_series_l677_67752

def last_two_digits (n : ℕ) : ℕ := n % 100

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

def factorial_ends_in_zeros (n : ℕ) : Prop := n > 10 → last_two_digits (n.factorial) = 0

theorem sum_of_last_two_digits_fibonacci_factorial_series :
  factorial_ends_in_zeros 11 →
  (fibonacci_factorial_series.map (λ n => last_two_digits n.factorial)).sum = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_fibonacci_factorial_series_l677_67752
