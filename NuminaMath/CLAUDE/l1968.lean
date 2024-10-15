import Mathlib

namespace NUMINAMATH_CALUDE_money_ratio_proof_l1968_196863

/-- Proves that the ratio of Nataly's money to Raquel's money is 3:1 given the problem conditions -/
theorem money_ratio_proof (tom nataly raquel : ℚ) : 
  tom = (1 / 4) * nataly →  -- Tom has 1/4 as much money as Nataly
  nataly = raquel * (nataly / raquel) →  -- Nataly has a certain multiple of Raquel's money
  tom + raquel + nataly = 190 →  -- Total money is $190
  raquel = 40 →  -- Raquel has $40
  nataly / raquel = 3 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l1968_196863


namespace NUMINAMATH_CALUDE_trig_sum_equals_one_l1968_196859

theorem trig_sum_equals_one : 4 * Real.cos (Real.pi / 3) + 8 * Real.sin (Real.pi / 6) - 5 * Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_one_l1968_196859


namespace NUMINAMATH_CALUDE_sock_pair_count_l1968_196898

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: The number of ways to choose a pair of socks with different colors
    from 4 white, 4 brown, and 2 blue socks is 32 -/
theorem sock_pair_count :
  different_color_pairs 4 4 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l1968_196898


namespace NUMINAMATH_CALUDE_smallest_number_property_l1968_196823

/-- The smallest positive integer that is not prime, not a square, and has no prime factor less than 60 -/
def smallest_number : ℕ := 290977

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop := sorry

/-- A function that returns the smallest prime factor of a number -/
def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem smallest_number_property : 
  ¬ is_prime smallest_number ∧ 
  ¬ is_square smallest_number ∧ 
  smallest_prime_factor smallest_number > 59 ∧
  ∀ m : ℕ, m < smallest_number → 
    is_prime m ∨ is_square m ∨ smallest_prime_factor m ≤ 59 := by sorry

end NUMINAMATH_CALUDE_smallest_number_property_l1968_196823


namespace NUMINAMATH_CALUDE_fourth_to_second_quadrant_l1968_196850

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if a point is in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that if P(a,b) is in the fourth quadrant, then Q(-a,-b) is in the second quadrant -/
theorem fourth_to_second_quadrant (p : Point) :
  is_in_fourth_quadrant p → is_in_second_quadrant (Point.mk (-p.x) (-p.y)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_to_second_quadrant_l1968_196850


namespace NUMINAMATH_CALUDE_fourth_player_wins_probability_l1968_196828

def roll_probability : ℚ := 1 / 6

def other_roll_probability : ℚ := 1 - roll_probability

def num_players : ℕ := 4

def first_cycle_probability : ℚ := (other_roll_probability ^ (num_players - 1)) * roll_probability

def cycle_continuation_probability : ℚ := other_roll_probability ^ num_players

theorem fourth_player_wins_probability :
  let a := first_cycle_probability
  let r := cycle_continuation_probability
  (a / (1 - r)) = 125 / 671 := by sorry

end NUMINAMATH_CALUDE_fourth_player_wins_probability_l1968_196828


namespace NUMINAMATH_CALUDE_calf_grazing_area_increase_l1968_196817

/-- The additional area a calf can graze when its rope is increased from 10 m to 35 m -/
theorem calf_grazing_area_increase : 
  let initial_radius : ℝ := 10
  let increased_radius : ℝ := 35
  let additional_area := π * increased_radius^2 - π * initial_radius^2
  additional_area = 1125 * π := by
  sorry

end NUMINAMATH_CALUDE_calf_grazing_area_increase_l1968_196817


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1968_196805

theorem expand_and_simplify (y : ℝ) : -3 * (y - 4) * (y + 9) = -3 * y^2 - 15 * y + 108 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1968_196805


namespace NUMINAMATH_CALUDE_max_xy_value_l1968_196892

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) :
  ∃ (max_xy : ℝ), max_xy = 25.92 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 6 * x' + 8 * y' = 72 → x' = 2 * y' → x' * y' ≤ max_xy :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l1968_196892


namespace NUMINAMATH_CALUDE_long_track_five_times_short_track_l1968_196841

/-- Represents the lengths of the short and long tracks -/
structure TrackLengths where
  short : ℝ
  long : ℝ

/-- Represents the training schedule for a week -/
structure WeekSchedule where
  days : ℕ
  longTracksPerDay : ℕ
  shortTracksPerDay : ℕ

/-- Calculates the total distance run in a week -/
def totalDistance (t : TrackLengths) (w : WeekSchedule) : ℝ :=
  w.days * (w.longTracksPerDay * t.long + w.shortTracksPerDay * t.short)

theorem long_track_five_times_short_track 
  (t : TrackLengths) 
  (w1 w2 : WeekSchedule) 
  (h1 : w1.days = 6 ∧ w1.longTracksPerDay = 1 ∧ w1.shortTracksPerDay = 2)
  (h2 : w2.days = 7 ∧ w2.longTracksPerDay = 1 ∧ w2.shortTracksPerDay = 1)
  (h3 : totalDistance t w1 = 5000)
  (h4 : totalDistance t w1 = totalDistance t w2) :
  t.long = 5 * t.short := by
  sorry

end NUMINAMATH_CALUDE_long_track_five_times_short_track_l1968_196841


namespace NUMINAMATH_CALUDE_cosine_range_in_triangle_l1968_196804

theorem cosine_range_in_triangle (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_AB : dist A B = 3)
  (h_AC : dist A C = 2)
  (h_BC : dist B C > Real.sqrt 2) :
  ∃ (cosA : ℝ), cosA = (dist A B)^2 + (dist A C)^2 - (dist B C)^2 / (2 * dist A B * dist A C) ∧ 
  -1 < cosA ∧ cosA < 11/12 :=
sorry

end NUMINAMATH_CALUDE_cosine_range_in_triangle_l1968_196804


namespace NUMINAMATH_CALUDE_projection_squared_magnitude_l1968_196858

-- Define the 3D Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define point A
def A : Point3D := ⟨3, 7, -4⟩

-- Define point B as the projection of A onto the xOz plane
def B : Point3D := ⟨A.x, 0, A.z⟩

-- Define the squared magnitude of a vector
def squaredMagnitude (p : Point3D) : ℝ :=
  p.x^2 + p.y^2 + p.z^2

-- Theorem statement
theorem projection_squared_magnitude :
  squaredMagnitude B = 25 := by sorry

end NUMINAMATH_CALUDE_projection_squared_magnitude_l1968_196858


namespace NUMINAMATH_CALUDE_regular_tetrahedron_unordered_pairs_l1968_196819

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The number of edges in a regular tetrahedron -/
  num_edges : ℕ
  /-- The property that any two edges determine the same plane -/
  edges_same_plane : Unit

/-- The number of unordered pairs of edges in a regular tetrahedron -/
def num_unordered_pairs (t : RegularTetrahedron) : ℕ :=
  (t.num_edges * (t.num_edges - 1)) / 2

/-- Theorem stating that the number of unordered pairs of edges in a regular tetrahedron is 15 -/
theorem regular_tetrahedron_unordered_pairs :
  ∀ t : RegularTetrahedron, t.num_edges = 6 → num_unordered_pairs t = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_regular_tetrahedron_unordered_pairs_l1968_196819


namespace NUMINAMATH_CALUDE_arrangements_combinations_ratio_l1968_196845

/-- Number of arrangements of n items taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - r))

/-- Number of combinations of n items taken r at a time -/
def C (n : ℕ) (r : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem arrangements_combinations_ratio : (A 7 2) / (C 10 2) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_combinations_ratio_l1968_196845


namespace NUMINAMATH_CALUDE_total_soccer_balls_l1968_196844

/-- Given the following conditions:
  - The school purchased 10 boxes of soccer balls
  - Each box contains 8 packages
  - Each package has 13 soccer balls
  Prove that the total number of soccer balls purchased is 1040 -/
theorem total_soccer_balls (num_boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ)
  (h1 : num_boxes = 10)
  (h2 : packages_per_box = 8)
  (h3 : balls_per_package = 13) :
  num_boxes * packages_per_box * balls_per_package = 1040 := by
  sorry

end NUMINAMATH_CALUDE_total_soccer_balls_l1968_196844


namespace NUMINAMATH_CALUDE_fraction_simplest_form_l1968_196861

theorem fraction_simplest_form (x y : ℝ) : 
  ¬∃ (a b : ℝ), (x - y) / (x^2 + y^2) = a / b ∧ (a ≠ x - y ∨ b ≠ x^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplest_form_l1968_196861


namespace NUMINAMATH_CALUDE_uncovered_area_calculation_l1968_196867

theorem uncovered_area_calculation (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  large_square_side^2 - 2 * small_square_side^2 = 68 :=
by sorry

end NUMINAMATH_CALUDE_uncovered_area_calculation_l1968_196867


namespace NUMINAMATH_CALUDE_scooter_price_calculation_l1968_196891

/-- Calculates the selling price of a scooter given its purchase price, repair costs, and gain percentage. -/
def scooter_selling_price (purchase_price repair_costs : ℚ) (gain_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_costs
  let gain := gain_percent / 100
  let profit := total_cost * gain
  total_cost + profit

/-- Theorem stating that the selling price of the scooter is $5800 given the specified conditions. -/
theorem scooter_price_calculation :
  scooter_selling_price 4700 800 (5454545454545454 / 100000000000000) = 5800 := by
  sorry

end NUMINAMATH_CALUDE_scooter_price_calculation_l1968_196891


namespace NUMINAMATH_CALUDE_oil_leak_during_work_l1968_196813

/-- The amount of oil leaked while engineers were working, given the total amount leaked and the amount leaked before they started. -/
theorem oil_leak_during_work (total_leak : ℕ) (pre_work_leak : ℕ) 
  (h1 : total_leak = 11687)
  (h2 : pre_work_leak = 6522) :
  total_leak - pre_work_leak = 5165 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_during_work_l1968_196813


namespace NUMINAMATH_CALUDE_ginos_brown_bears_l1968_196888

theorem ginos_brown_bears :
  ∀ (total white black brown : ℕ),
    total = 66 →
    white = 24 →
    black = 27 →
    total = white + black + brown →
    brown = 15 := by
  sorry

end NUMINAMATH_CALUDE_ginos_brown_bears_l1968_196888


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l1968_196808

/-- Represents a 9x9x9 cube composed of 3x3x3 subcubes -/
structure LargeCube where
  subcubes : Fin 3 → Fin 3 → Fin 3 → Unit

/-- Represents the modified structure after removing center cubes and facial units -/
structure ModifiedCube where
  remaining_subcubes : Fin 20 → Unit
  removed_centers : Unit
  removed_facial_units : Unit

/-- Calculates the surface area of the modified cube structure -/
def surface_area (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 1056 square units -/
theorem modified_cube_surface_area :
  ∀ (cube : LargeCube),
  ∃ (modified : ModifiedCube),
  surface_area modified = 1056 :=
sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l1968_196808


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1968_196836

theorem right_triangle_third_side (x y z : ℝ) : 
  (x > 0 ∧ y > 0 ∧ z > 0) →  -- positive sides
  (x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2) →  -- right triangle condition
  (|x - 4| + Real.sqrt (y - 3) = 0) →  -- given equation
  (z = 5 ∨ z = Real.sqrt 7) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1968_196836


namespace NUMINAMATH_CALUDE_minimum_school_payment_l1968_196899

/-- The minimum amount a school should pay for cinema tickets -/
theorem minimum_school_payment
  (individual_price : ℝ)
  (group_price : ℝ)
  (group_size : ℕ)
  (student_discount : ℝ)
  (num_students : ℕ)
  (h1 : individual_price = 6)
  (h2 : group_price = 40)
  (h3 : group_size = 10)
  (h4 : student_discount = 0.1)
  (h5 : num_students = 1258) :
  ∃ (min_payment : ℝ),
    min_payment = 4536 ∧
    min_payment ≤ (↑(num_students / group_size) * group_price * (1 - student_discount)) + 
                  (↑(num_students % group_size) * individual_price * (1 - student_discount)) :=
by
  sorry

#eval 1258 / 10 * 40 * 0.9

end NUMINAMATH_CALUDE_minimum_school_payment_l1968_196899


namespace NUMINAMATH_CALUDE_male_students_count_l1968_196811

/-- Calculates the total number of male students in first and second year -/
def total_male_students (total_first_year : ℕ) (female_first_year : ℕ) (male_second_year : ℕ) : ℕ :=
  (total_first_year - female_first_year) + male_second_year

/-- Proves that the total number of male students in first and second year is 620 -/
theorem male_students_count : 
  total_male_students 695 329 254 = 620 := by
  sorry

end NUMINAMATH_CALUDE_male_students_count_l1968_196811


namespace NUMINAMATH_CALUDE_product_of_three_terms_l1968_196814

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem product_of_three_terms
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 5 = 4) :
  a 4 * a 5 * a 6 = 64 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_terms_l1968_196814


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1968_196818

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp a α → perp b α → parallel a b := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1968_196818


namespace NUMINAMATH_CALUDE_point_direction_form_equation_l1968_196895

/-- The point-direction form equation of a line with direction vector (2, -3) passing through the point (1, 0) -/
theorem point_direction_form_equation (x y : ℝ) : 
  let direction_vector : ℝ × ℝ := (2, -3)
  let point : ℝ × ℝ := (1, 0)
  let line_equation := (x - point.1) / direction_vector.1 = y / direction_vector.2
  line_equation = ((x - 1) / 2 = y / (-3))
  := by sorry

end NUMINAMATH_CALUDE_point_direction_form_equation_l1968_196895


namespace NUMINAMATH_CALUDE_expansion_sum_l1968_196846

theorem expansion_sum (d : ℝ) (h : d ≠ 0) :
  let expansion := (15*d + 21 + 17*d^2) * (3*d + 4)
  ∃ (a b c e : ℝ), expansion = a*d^3 + b*d^2 + c*d + e ∧ a + b + c + e = 371 := by
  sorry

end NUMINAMATH_CALUDE_expansion_sum_l1968_196846


namespace NUMINAMATH_CALUDE_max_necklaces_is_five_l1968_196831

/-- Represents the number of beads of each color required for a single necklace -/
structure NecklacePattern where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the total number of beads available for each color -/
structure AvailableBeads where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Calculates the maximum number of complete necklaces that can be made -/
def maxNecklaces (pattern : NecklacePattern) (available : AvailableBeads) : ℕ :=
  min (available.green / pattern.green)
      (min (available.white / pattern.white)
           (available.orange / pattern.orange))

/-- Theorem stating that given the specific bead counts, the maximum number of necklaces is 5 -/
theorem max_necklaces_is_five :
  let pattern := NecklacePattern.mk 9 6 3
  let available := AvailableBeads.mk 45 45 45
  maxNecklaces pattern available = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_necklaces_is_five_l1968_196831


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_is_integer_l1968_196812

theorem greatest_integer_fraction_is_integer : 
  ∀ y : ℤ, y > 12 → ¬(∃ k : ℤ, (y^2 - 3*y + 4) / (y - 4) = k) ∧ 
  ∃ k : ℤ, (12^2 - 3*12 + 4) / (12 - 4) = k := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_is_integer_l1968_196812


namespace NUMINAMATH_CALUDE_church_cookies_total_l1968_196802

/-- Calculates the total number of cookies baked by church members -/
theorem church_cookies_total (members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ) : 
  members = 100 → sheets_per_member = 10 → cookies_per_sheet = 16 → 
  members * sheets_per_member * cookies_per_sheet = 16000 := by
  sorry

end NUMINAMATH_CALUDE_church_cookies_total_l1968_196802


namespace NUMINAMATH_CALUDE_inequality_condition_l1968_196879

theorem inequality_condition (a b : ℝ) (h : a * Real.sqrt a > b * Real.sqrt b) : a > b ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l1968_196879


namespace NUMINAMATH_CALUDE_special_heptagon_perturbation_l1968_196896

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A heptagon represented by its vertices -/
structure Heptagon :=
  (vertices : Fin 7 → Point)

/-- Predicate to check if a heptagon is convex -/
def is_convex (h : Heptagon) : Prop := sorry

/-- Predicate to check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Point × Point) (p : Point) : Prop := sorry

/-- Predicate to check if a heptagon is special -/
def is_special (h : Heptagon) : Prop :=
  ∃ (i j k : Fin 7) (p : Point),
    i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    intersect_at_point
      (h.vertices i, h.vertices ((i + 3) % 7))
      (h.vertices j, h.vertices ((j + 3) % 7))
      (h.vertices k, h.vertices ((k + 3) % 7))
      p

/-- Definition of a small perturbation -/
def small_perturbation (h1 h2 : Heptagon) (ε : ℝ) : Prop :=
  ∃ (i : Fin 7),
    ∀ (j : Fin 7),
      if i = j then
        (h1.vertices j).x - ε < (h2.vertices j).x ∧ (h2.vertices j).x < (h1.vertices j).x + ε ∧
        (h1.vertices j).y - ε < (h2.vertices j).y ∧ (h2.vertices j).y < (h1.vertices j).y + ε
      else
        h1.vertices j = h2.vertices j

/-- The main theorem -/
theorem special_heptagon_perturbation (h : Heptagon) (hconv : is_convex h) (hspec : is_special h) :
  ∃ (h' : Heptagon) (ε : ℝ), ε > 0 ∧ small_perturbation h h' ε ∧ is_convex h' ∧ ¬is_special h' :=
sorry

end NUMINAMATH_CALUDE_special_heptagon_perturbation_l1968_196896


namespace NUMINAMATH_CALUDE_expand_expression_l1968_196803

theorem expand_expression (y : ℝ) : 12 * (3 * y + 7) = 36 * y + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1968_196803


namespace NUMINAMATH_CALUDE_at_most_two_rational_points_l1968_196843

/-- A point in the 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- A circle in the 2D plane -/
structure Circle where
  center_x : ℚ
  center_y : ℝ
  radius : ℝ

/-- A point is on a circle if it satisfies the circle equation -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center_x)^2 + (p.y - c.center_y)^2 = c.radius^2

/-- The main theorem: there are at most two rational points on a circle with irrational y-coordinate of the center -/
theorem at_most_two_rational_points (c : Circle) 
    (h : Irrational c.center_y) :
    ∃ (p1 p2 : Point), ∀ (p : Point), 
      p.onCircle c → p = p1 ∨ p = p2 := by
  sorry

end NUMINAMATH_CALUDE_at_most_two_rational_points_l1968_196843


namespace NUMINAMATH_CALUDE_reseating_arrangements_l1968_196873

/-- Number of ways to reseat n people in n+2 seats with restrictions -/
def T : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| n + 3 => T (n + 2) + T (n + 1)

/-- There are 8 seats in total -/
def total_seats : ℕ := 8

/-- There are 6 people to be seated -/
def num_people : ℕ := 6

theorem reseating_arrangements :
  T num_people = 13 :=
sorry

end NUMINAMATH_CALUDE_reseating_arrangements_l1968_196873


namespace NUMINAMATH_CALUDE_club_membership_count_l1968_196882

theorem club_membership_count :
  let tennis : ℕ := 138
  let baseball : ℕ := 255
  let both : ℕ := 94
  let neither : ℕ := 11
  tennis + baseball - both + neither = 310 :=
by sorry

end NUMINAMATH_CALUDE_club_membership_count_l1968_196882


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l1968_196857

theorem equation_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) + 2 * x :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l1968_196857


namespace NUMINAMATH_CALUDE_min_value_of_f_l1968_196826

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1968_196826


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l1968_196860

/-- Given that M(4,4) is the midpoint of AB and A has coordinates (8,4),
    prove that the sum of the coordinates of B is 4. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (4, 4) →
  A = (8, 4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B.1 + B.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l1968_196860


namespace NUMINAMATH_CALUDE_susan_vacation_pay_missed_l1968_196820

/-- Calculate the pay missed during Susan's vacation --/
theorem susan_vacation_pay_missed
  (vacation_length : ℕ) -- Length of vacation in weeks
  (work_days_per_week : ℕ) -- Number of work days per week
  (paid_vacation_days : ℕ) -- Number of paid vacation days
  (hourly_rate : ℚ) -- Hourly pay rate
  (hours_per_day : ℕ) -- Number of work hours per day
  (h1 : vacation_length = 2)
  (h2 : work_days_per_week = 5)
  (h3 : paid_vacation_days = 6)
  (h4 : hourly_rate = 15)
  (h5 : hours_per_day = 8) :
  (vacation_length * work_days_per_week - paid_vacation_days) * (hourly_rate * hours_per_day) = 480 :=
by sorry

end NUMINAMATH_CALUDE_susan_vacation_pay_missed_l1968_196820


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1968_196856

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1968_196856


namespace NUMINAMATH_CALUDE_range_of_m_l1968_196833

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem range_of_m (m : ℝ) : (A ∩ B m = B m) → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1968_196833


namespace NUMINAMATH_CALUDE_scrap_iron_average_l1968_196838

theorem scrap_iron_average (total_friends : Nat) (total_average : ℝ) (ivan_amount : ℝ) :
  total_friends = 5 →
  total_average = 55 →
  ivan_amount = 43 →
  let total_amount := total_friends * total_average
  let remaining_amount := total_amount - ivan_amount
  let remaining_friends := total_friends - 1
  (remaining_amount / remaining_friends : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_scrap_iron_average_l1968_196838


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1968_196829

open Set

def U : Set Nat := {0, 1, 2, 3, 4}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1968_196829


namespace NUMINAMATH_CALUDE_sum_of_integers_l1968_196853

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 120) : 
  x.val + y.val = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1968_196853


namespace NUMINAMATH_CALUDE_no_zeros_implies_a_less_than_negative_one_l1968_196871

theorem no_zeros_implies_a_less_than_negative_one (a : ℝ) :
  (∀ x : ℝ, 4^x - 2^(x+1) - a ≠ 0) → a < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_zeros_implies_a_less_than_negative_one_l1968_196871


namespace NUMINAMATH_CALUDE_package_cost_theorem_l1968_196852

/-- The cost of a 12-roll package of paper towels -/
def package_cost : ℝ := 9

/-- The cost of one roll sold individually -/
def individual_roll_cost : ℝ := 1

/-- The percent savings per roll for the 12-roll package -/
def percent_savings : ℝ := 0.25

/-- The number of rolls in a package -/
def rolls_per_package : ℕ := 12

theorem package_cost_theorem : 
  package_cost = rolls_per_package * (individual_roll_cost * (1 - percent_savings)) :=
by sorry

end NUMINAMATH_CALUDE_package_cost_theorem_l1968_196852


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1968_196834

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem ellipse_and_line_intersection
  (C : Ellipse)
  (h_point : C.a^2 * (6 / C.a^2 + 1 / C.b^2) = C.a^2) -- Point (√6, 1) lies on the ellipse
  (h_focus : C.a^2 - C.b^2 = 4) -- Left focus is at (-2, 0)
  (m : ℝ)
  (h_distinct : ∃ (A B : Point), A ≠ B ∧
    C.a^2 * ((A.x^2 / C.a^2) + (A.y^2 / C.b^2)) = C.a^2 ∧
    C.a^2 * ((B.x^2 / C.a^2) + (B.y^2 / C.b^2)) = C.a^2 ∧
    A.y = A.x + m ∧ B.y = B.x + m)
  (h_midpoint : ∃ (M : Point), M.x^2 + M.y^2 = 1 ∧
    ∃ (A B : Point), A ≠ B ∧
      C.a^2 * ((A.x^2 / C.a^2) + (A.y^2 / C.b^2)) = C.a^2 ∧
      C.a^2 * ((B.x^2 / C.a^2) + (B.y^2 / C.b^2)) = C.a^2 ∧
      A.y = A.x + m ∧ B.y = B.x + m ∧
      M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) :
  C.a = 2 * Real.sqrt 2 ∧ C.b = 2 ∧ m = 3 * Real.sqrt 5 / 5 ∨ m = -3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1968_196834


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1968_196864

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 2 (-3) 4 →
  point = Point.mk (-1) 2 →
  ∃ (result_line : Line),
    result_line.perpendicular given_line ∧
    point.liesOn result_line ∧
    result_line = Line.mk 3 2 (-1) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l1968_196864


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_solution_set_correct_l1968_196877

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The vectors a and b as functions of x -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x - 1, 2)

/-- Theorem stating the conditions for a and b to be parallel -/
theorem parallel_vectors_condition :
  ∀ x : ℝ, are_parallel (a x) (b x) ↔ x = 2 ∨ x = -1 :=
by
  sorry

/-- The solution set for x -/
def solution_set : Set ℝ := {2, -1}

/-- Theorem stating that the solution set is correct -/
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ are_parallel (a x) (b x) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_solution_set_correct_l1968_196877


namespace NUMINAMATH_CALUDE_third_number_in_list_l1968_196810

theorem third_number_in_list (a b c d e : ℕ) : 
  a = 60 → 
  e = 300 → 
  a * b * c = 810000 → 
  b * c * d = 2430000 → 
  c * d * e = 8100000 → 
  c = 150 := by
sorry

end NUMINAMATH_CALUDE_third_number_in_list_l1968_196810


namespace NUMINAMATH_CALUDE_table_tennis_tournament_l1968_196875

theorem table_tennis_tournament (x : ℕ) :
  let sixth_graders := 2 * x
  let seventh_graders := x
  let total_participants := sixth_graders + seventh_graders
  let total_matches := total_participants * (total_participants - 1) / 2
  let matches_between_grades := sixth_graders * seventh_graders
  let matches_among_sixth := sixth_graders * (sixth_graders - 1) / 2
  let matches_among_seventh := seventh_graders * (seventh_graders - 1) / 2
  let matches_won_by_sixth := matches_among_sixth + matches_between_grades / 2
  let matches_won_by_seventh := matches_among_seventh + matches_between_grades / 2
  matches_won_by_seventh = (matches_won_by_sixth * 14) / 10 →
  total_participants = 9 :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l1968_196875


namespace NUMINAMATH_CALUDE_function_properties_l1968_196854

/-- Given that y+6 is directly proportional to x+1 and when x=3, y=2 -/
def proportional_function (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y + 6 = k * (x + 1) ∧ 2 + 6 = k * (3 + 1)

theorem function_properties :
  ∀ x y m : ℝ,
  proportional_function x y →
  (y = 2*x - 4 ∧
   (proportional_function m (-2) → m = 1) ∧
   ¬proportional_function 1 (-3)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1968_196854


namespace NUMINAMATH_CALUDE_double_average_l1968_196840

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 25) (h2 : original_avg = 70) :
  let total_marks := n * original_avg
  let doubled_marks := 2 * total_marks
  let new_avg := doubled_marks / n
  new_avg = 140 := by
sorry

end NUMINAMATH_CALUDE_double_average_l1968_196840


namespace NUMINAMATH_CALUDE_fraction_equality_l1968_196830

theorem fraction_equality (x y z : ℝ) (h : (x - y) / (z - y) = -10) :
  (x - z) / (y - z) = 11 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1968_196830


namespace NUMINAMATH_CALUDE_factorization_equality_l1968_196890

theorem factorization_equality (a b c d : ℝ) :
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 = 
  (a - b) * (b - c) * (c - d) * (d - a) * (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1968_196890


namespace NUMINAMATH_CALUDE_interior_point_distance_l1968_196855

-- Define the rectangle and point
def Rectangle (E F G H : ℝ × ℝ) : Prop := sorry

def InteriorPoint (P : ℝ × ℝ) (E F G H : ℝ × ℝ) : Prop := 
  Rectangle E F G H ∧ sorry

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem interior_point_distance 
  (E F G H P : ℝ × ℝ) 
  (h_rect : Rectangle E F G H)
  (h_interior : InteriorPoint P E F G H)
  (h_PE : distance P E = 5)
  (h_PH : distance P H = 12)
  (h_PG : distance P G = 13) :
  distance P F = 12 := by
  sorry

end NUMINAMATH_CALUDE_interior_point_distance_l1968_196855


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l1968_196889

theorem x_minus_y_equals_three 
  (h1 : 3 * x - 5 * y = 5) 
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l1968_196889


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l1968_196816

theorem negation_of_exists_lt_is_forall_ge :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l1968_196816


namespace NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l1968_196847

theorem smallest_n_for_irreducible_fractions : 
  ∃ (n : ℕ), n = 35 ∧ 
  (∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + k + 2) = 1) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, 7 ≤ k ∧ k ≤ 31 ∧ Nat.gcd k (m + k + 2) ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l1968_196847


namespace NUMINAMATH_CALUDE_museum_discount_percentage_l1968_196886

/-- Represents the discount percentage for people 18 years old or younger -/
def discount_percentage : ℝ := 30

/-- Represents the regular ticket cost -/
def regular_ticket_cost : ℝ := 10

/-- Represents Dorothy's initial amount of money -/
def dorothy_initial_money : ℝ := 70

/-- Represents Dorothy's remaining money after the trip -/
def dorothy_remaining_money : ℝ := 26

/-- Represents the number of people in Dorothy's family -/
def family_size : ℕ := 5

/-- Represents the number of adults (paying full price) in Dorothy's family -/
def num_adults : ℕ := 3

/-- Represents the number of children (eligible for discount) in Dorothy's family -/
def num_children : ℕ := 2

theorem museum_discount_percentage :
  let total_spent := dorothy_initial_money - dorothy_remaining_money
  let adult_cost := num_adults * regular_ticket_cost
  let children_cost := total_spent - adult_cost
  let discounted_ticket_cost := regular_ticket_cost * (1 - discount_percentage / 100)
  children_cost = num_children * discounted_ticket_cost :=
by sorry

#check museum_discount_percentage

end NUMINAMATH_CALUDE_museum_discount_percentage_l1968_196886


namespace NUMINAMATH_CALUDE_max_x5_value_l1968_196897

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h_eq : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) :
  x₅ ≤ 5 := by
  sorry

#check max_x5_value

end NUMINAMATH_CALUDE_max_x5_value_l1968_196897


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1968_196880

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y + 2*x - 4*y + a = 0

-- Define the midpoint M
def midpoint_M (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

-- Define the chord length
def chord_length (l : ℝ) : Prop :=
  l = 2 * Real.sqrt 7

-- Main theorem
theorem circle_intersection_theorem (a : ℝ) :
  (∃ x y : ℝ, circle_C a x y ∧ midpoint_M x y) →
  (a < 3 ∧
   ∃ k b : ℝ, k = 1 ∧ b = 1 ∧ ∀ x y : ℝ, y = k*x + b) ∧
  (∀ l : ℝ, chord_length l →
    ∀ x y : ℝ, circle_C a x y ↔ (x+1)^2 + (y-2)^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1968_196880


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1968_196837

theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ = Real.log x₀ ∧ k = 1 / x₀) → k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1968_196837


namespace NUMINAMATH_CALUDE_projection_line_equation_l1968_196872

/-- The line l passing through a point P that is the projection of the origin onto l -/
structure ProjectionLine where
  -- The coordinates of point P
  px : ℝ
  py : ℝ
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- P is on the line
  point_on_line : a * px + b * py + c = 0
  -- P is the projection of the origin onto the line
  is_projection : a * px + b * py = 0

/-- The equation of the line l given the projection point P(-2, 1) -/
theorem projection_line_equation (l : ProjectionLine) 
  (h1 : l.px = -2) 
  (h2 : l.py = 1) : 
  l.a = 2 ∧ l.b = -1 ∧ l.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_line_equation_l1968_196872


namespace NUMINAMATH_CALUDE_fruit_difference_l1968_196862

theorem fruit_difference (total : ℕ) (apples : ℕ) : 
  total = 913 → apples = 514 → apples - (total - apples) = 115 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_l1968_196862


namespace NUMINAMATH_CALUDE_heartsuit_nested_equals_fourteen_l1968_196827

-- Define the ⊛ operation for positive real numbers
def heartsuit (x y : ℝ) : ℝ := x + 2 * y

-- State the theorem
theorem heartsuit_nested_equals_fourteen :
  heartsuit 2 (heartsuit 2 2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_nested_equals_fourteen_l1968_196827


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l1968_196800

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Fluffy in the 4-dog group and Nipper in the 6-dog group -/
def dog_grouping_ways : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4
  let group2_size : ℕ := 6
  let group3_size : ℕ := 2
  let remaining_dogs : ℕ := total_dogs - 2  -- Fluffy and Nipper are already placed
  Nat.choose remaining_dogs (group1_size - 1) * Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1)

theorem dog_grouping_theorem : dog_grouping_ways = 2520 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l1968_196800


namespace NUMINAMATH_CALUDE_juice_drink_cost_l1968_196809

theorem juice_drink_cost (initial_amount : ℕ) (pizza_cost : ℕ) (pizza_quantity : ℕ) 
  (juice_quantity : ℕ) (return_amount : ℕ) : 
  initial_amount = 50 → 
  pizza_cost = 12 → 
  pizza_quantity = 2 → 
  juice_quantity = 2 → 
  return_amount = 22 → 
  (initial_amount - return_amount - pizza_cost * pizza_quantity) / juice_quantity = 2 :=
by sorry

end NUMINAMATH_CALUDE_juice_drink_cost_l1968_196809


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1968_196868

theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (-2, m) →
  (∃ (k : ℝ), a = k • b) →
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1968_196868


namespace NUMINAMATH_CALUDE_adam_students_count_l1968_196824

/-- The number of students Adam teaches per year (except for the first year) -/
def studentsPerYear : ℕ := 50

/-- The number of students Adam teaches in the first year -/
def studentsFirstYear : ℕ := 40

/-- The total number of years Adam teaches -/
def totalYears : ℕ := 10

/-- The total number of students Adam teaches over the given period -/
def totalStudents : ℕ := studentsFirstYear + studentsPerYear * (totalYears - 1)

theorem adam_students_count : totalStudents = 490 := by
  sorry

end NUMINAMATH_CALUDE_adam_students_count_l1968_196824


namespace NUMINAMATH_CALUDE_insect_eggs_l1968_196885

def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

theorem insect_eggs : base_6_to_10 2 5 3 = 105 := by sorry

end NUMINAMATH_CALUDE_insect_eggs_l1968_196885


namespace NUMINAMATH_CALUDE_issac_utensils_count_l1968_196821

/-- The total number of writing utensils bought by Issac -/
def total_utensils (num_pens : ℕ) (num_pencils : ℕ) : ℕ :=
  num_pens + num_pencils

/-- Theorem stating the total number of writing utensils Issac bought -/
theorem issac_utensils_count :
  ∀ (num_pens : ℕ) (num_pencils : ℕ),
    num_pens = 16 →
    num_pencils = 5 * num_pens + 12 →
    total_utensils num_pens num_pencils = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_issac_utensils_count_l1968_196821


namespace NUMINAMATH_CALUDE_cube_edges_sum_l1968_196866

/-- Given a cube-shaped toy made up of 27 small cubes, with the total length of all edges
    of the large cube being 82.8 cm, prove that the sum of the length of one edge of the
    large cube and one edge of a small cube is 9.2 cm. -/
theorem cube_edges_sum (total_edge_length : ℝ) (num_small_cubes : ℕ) :
  total_edge_length = 82.8 ∧ num_small_cubes = 27 →
  ∃ (large_edge small_edge : ℝ),
    large_edge = total_edge_length / 12 ∧
    small_edge = large_edge / 3 ∧
    large_edge + small_edge = 9.2 :=
by sorry

end NUMINAMATH_CALUDE_cube_edges_sum_l1968_196866


namespace NUMINAMATH_CALUDE_no_infinite_arithmetic_progression_in_squares_l1968_196815

theorem no_infinite_arithmetic_progression_in_squares :
  ¬ ∃ (a d : ℕ) (f : ℕ → ℕ),
    (∀ n, f n < f (n + 1)) ∧
    (∀ n, ∃ k, f n = k^2) ∧
    (∀ n, f (n + 1) - f n = d) :=
sorry

end NUMINAMATH_CALUDE_no_infinite_arithmetic_progression_in_squares_l1968_196815


namespace NUMINAMATH_CALUDE_book_selling_price_l1968_196839

/-- Calculates the selling price of a book given its cost price and profit percentage. -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating that a book with a cost price of $60 and a profit percentage of 30% has a selling price of $78. -/
theorem book_selling_price :
  selling_price 60 30 = 78 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_l1968_196839


namespace NUMINAMATH_CALUDE_expand_expression_l1968_196883

theorem expand_expression (x : ℝ) : 20 * (3 * x + 7 - 2 * x^2) = 60 * x + 140 - 40 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1968_196883


namespace NUMINAMATH_CALUDE_function_inequality_l1968_196884

/-- Given functions f and g, prove that if f(x) ≥ g(x) - exp(x) for all x ≥ 1, then a ≥ 1/(2*exp(1)) -/
theorem function_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → a * x - Real.exp x ≥ Real.log x / x - Real.exp x) →
  a ≥ 1 / (2 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1968_196884


namespace NUMINAMATH_CALUDE_fraction_exceeding_by_30_l1968_196848

theorem fraction_exceeding_by_30 (x : ℚ) : 
  48 = 48 * x + 30 → x = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exceeding_by_30_l1968_196848


namespace NUMINAMATH_CALUDE_bad_carrots_l1968_196869

/-- Given the number of carrots picked by Carol and her mother, and the number of good carrots,
    calculate the number of bad carrots. -/
theorem bad_carrots (carol_carrots mother_carrots good_carrots : ℕ) : 
  carol_carrots = 29 → mother_carrots = 16 → good_carrots = 38 →
  carol_carrots + mother_carrots - good_carrots = 7 := by
  sorry

#check bad_carrots

end NUMINAMATH_CALUDE_bad_carrots_l1968_196869


namespace NUMINAMATH_CALUDE_max_area_and_front_wall_length_l1968_196870

/-- The material cost function for the house -/
def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

/-- The constraint on the material cost -/
def cost_constraint (x y : ℝ) : Prop := material_cost x y ≤ 32000

/-- The area of the house -/
def house_area (x y : ℝ) : ℝ := x * y

/-- Theorem stating the maximum area and corresponding front wall length -/
theorem max_area_and_front_wall_length :
  ∃ (x y : ℝ), 
    cost_constraint x y ∧ 
    ∀ (x' y' : ℝ), cost_constraint x' y' → house_area x' y' ≤ house_area x y ∧
    house_area x y = 100 ∧
    x = 20 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_area_and_front_wall_length_l1968_196870


namespace NUMINAMATH_CALUDE_trig_identity_l1968_196807

theorem trig_identity (α : ℝ) : 
  1 - Real.cos (2 * α - π) + Real.cos (4 * α - 2 * π) = 
  4 * Real.cos (2 * α) * Real.cos (π / 6 + α) * Real.cos (π / 6 - α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1968_196807


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1968_196825

theorem power_fraction_simplification :
  (3^1024 + 5 * 3^1022) / (3^1024 - 3^1022) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1968_196825


namespace NUMINAMATH_CALUDE_min_m_value_l1968_196865

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x ^ 2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_m_value (m : ℝ) :
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x ≤ m) →
  m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_m_value_l1968_196865


namespace NUMINAMATH_CALUDE_bacteria_urea_phenol_red_l1968_196842

/-- Represents the color of the phenol red indicator -/
inductive IndicatorColor
| Blue
| Red
| Black
| Brown

/-- Represents the pH level of the medium -/
inductive pHLevel
| Acidic
| Neutral
| Alkaline

/-- Represents a culture medium -/
structure CultureMedium where
  nitrogenSource : String
  indicator : String
  pH : pHLevel

/-- Represents the bacterial culture -/
structure BacterialCulture where
  medium : CultureMedium
  bacteriaPresent : Bool

/-- Function to determine the color of phenol red based on pH -/
def phenolRedColor (pH : pHLevel) : IndicatorColor :=
  match pH with
  | pHLevel.Alkaline => IndicatorColor.Red
  | _ => IndicatorColor.Blue  -- Simplified for this problem

/-- Main theorem to prove -/
theorem bacteria_urea_phenol_red 
  (culture : BacterialCulture)
  (h1 : culture.medium.nitrogenSource = "urea")
  (h2 : culture.medium.indicator = "phenol red")
  (h3 : culture.bacteriaPresent = true) :
  phenolRedColor culture.medium.pH = IndicatorColor.Red :=
sorry

end NUMINAMATH_CALUDE_bacteria_urea_phenol_red_l1968_196842


namespace NUMINAMATH_CALUDE_changhyeok_snacks_l1968_196851

theorem changhyeok_snacks :
  ∀ (s d : ℕ),
  s + d = 12 →
  1000 * s + 1300 * d = 15000 →
  s = 2 := by
sorry

end NUMINAMATH_CALUDE_changhyeok_snacks_l1968_196851


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l1968_196893

/-- A cubic function f(x) = (a/3)x^3 + bx^2 + cx + d is monotonically increasing on ℝ 
    if and only if its derivative is non-negative for all x ∈ ℝ -/
def monotonically_increasing (a b c : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0

/-- The theorem stating the minimum value of (a + 2b + 3c)/(b - a) 
    for a monotonically increasing cubic function with a < b -/
theorem min_value_cubic_function (a b c : ℝ) 
    (h1 : a < b) 
    (h2 : monotonically_increasing a b c) : 
  (a + 2*b + 3*c) / (b - a) ≥ 8 + 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l1968_196893


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_2000_l1968_196806

theorem modular_inverse_13_mod_2000 : ∃ x : ℤ, 0 ≤ x ∧ x < 2000 ∧ (13 * x) % 2000 = 1 :=
by
  use 1077
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_2000_l1968_196806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_term_l1968_196832

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_specific_term
  (seq : ArithmeticSequence)
  (m : ℕ)
  (h1 : seq.S (m - 2) = -4)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 2) = 12) :
  seq.a m = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_term_l1968_196832


namespace NUMINAMATH_CALUDE_ounces_per_cup_ounces_per_cup_is_eight_l1968_196849

/-- The number of ounces in a cup, given Cassie's water consumption habits -/
theorem ounces_per_cup : ℕ :=
  let cups_per_day : ℕ := 12
  let bottle_capacity : ℕ := 16
  let refills_per_day : ℕ := 6
  (refills_per_day * bottle_capacity) / cups_per_day

/-- Proof that the number of ounces in a cup is 8 -/
theorem ounces_per_cup_is_eight : ounces_per_cup = 8 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_cup_ounces_per_cup_is_eight_l1968_196849


namespace NUMINAMATH_CALUDE_product_of_five_terms_l1968_196894

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_five_terms
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a3 : a 3 = -1) :
  a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_terms_l1968_196894


namespace NUMINAMATH_CALUDE_min_value_expression_l1968_196874

/-- Given that x₁ and x₂ are the roots of the equations x + exp x = 3 and x + log x = 3 respectively,
    and x₁ + x₂ = a + b where a and b are positive real numbers,
    prove that the minimum value of (7b² + 1) / (ab) is 2. -/
theorem min_value_expression (x₁ x₂ a b : ℝ) : 
  (∃ (x : ℝ), x + Real.exp x = 3 ∧ x = x₁) →
  (∃ (x : ℝ), x + Real.log x = 3 ∧ x = x₂) →
  x₁ + x₂ = a + b →
  a > 0 →
  b > 0 →
  (∀ c d : ℝ, c > 0 → d > 0 → c + d = a + b → (7 * d^2 + 1) / (c * d) ≥ 2) ∧
  (∃ e f : ℝ, e > 0 ∧ f > 0 ∧ e + f = a + b ∧ (7 * f^2 + 1) / (e * f) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1968_196874


namespace NUMINAMATH_CALUDE_simplify_expression_l1968_196878

theorem simplify_expression (x : ℝ) (h : x ≠ -1) :
  (x - 1 - 8 / (x + 1)) / ((x + 3) / (x + 1)) = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1968_196878


namespace NUMINAMATH_CALUDE_game_ends_in_45_rounds_l1968_196887

/-- Represents the state of the game with token counts for each player -/
structure GameState where
  playerA : ℕ
  playerB : ℕ
  playerC : ℕ

/-- Applies one round of the game rules to the current state -/
def applyRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (initialState : GameState) : ℕ :=
  sorry

theorem game_ends_in_45_rounds :
  let initialState : GameState := ⟨18, 16, 15⟩
  countRounds initialState = 45 := by
  sorry

end NUMINAMATH_CALUDE_game_ends_in_45_rounds_l1968_196887


namespace NUMINAMATH_CALUDE_inequality_preservation_l1968_196822

theorem inequality_preservation (a b : ℝ) : a < b → 1 - a > 1 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1968_196822


namespace NUMINAMATH_CALUDE_classroom_students_count_l1968_196835

theorem classroom_students_count :
  ∃! n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by sorry

end NUMINAMATH_CALUDE_classroom_students_count_l1968_196835


namespace NUMINAMATH_CALUDE_triangle_altitude_l1968_196876

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 960 ∧ base = 48 ∧ area = (1/2) * base * altitude →
  altitude = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l1968_196876


namespace NUMINAMATH_CALUDE_smallest_positive_constant_inequality_l1968_196881

theorem smallest_positive_constant_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (∃ c : ℝ, c > 0 ∧ ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 →
    Real.sqrt (x * y * z) + c * Real.sqrt (|x - y|) ≥ (x + y + z) / 3) ∧
  (∀ c : ℝ, c > 0 ∧ (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 →
    Real.sqrt (x * y * z) + c * Real.sqrt (|x - y|) ≥ (x + y + z) / 3) → c ≥ 1) ∧
  (Real.sqrt (x * y * z) + Real.sqrt (|x - y|) ≥ (x + y + z) / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_constant_inequality_l1968_196881


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l1968_196801

/-- The quadratic function y = x^2 - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- The shifted quadratic function y = x^2 - 4x - 3 + a -/
def f_shifted (x a : ℝ) : ℝ := f x + a

theorem quadratic_shift_theorem :
  /- The value of a that makes the parabola pass through (0,1) is 4 -/
  (∃ a : ℝ, f_shifted 0 a = 1 ∧ a = 4) ∧
  /- The values of a that make the parabola intersect the coordinate axes at exactly 2 points are 3 and 7 -/
  (∃ a₁ a₂ : ℝ, 
    ((f_shifted 0 a₁ = 0 ∨ (∃ x : ℝ, x ≠ 0 ∧ f_shifted x a₁ = 0)) ∧
     (∃! x : ℝ, f_shifted x a₁ = 0)) ∧
    ((f_shifted 0 a₂ = 0 ∨ (∃ x : ℝ, x ≠ 0 ∧ f_shifted x a₂ = 0)) ∧
     (∃! x : ℝ, f_shifted x a₂ = 0)) ∧
    a₁ = 3 ∧ a₂ = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l1968_196801
