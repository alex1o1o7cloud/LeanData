import Mathlib

namespace NUMINAMATH_CALUDE_car_speed_comparison_l3919_391974

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 3 / (1 / u + 2 / v)
  let y := (2 * u + v) / 3
  x ≤ y := by
sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l3919_391974


namespace NUMINAMATH_CALUDE_vector_square_difference_l3919_391905

theorem vector_square_difference (a b : ℝ × ℝ) (h1 : a + b = (-3, 6)) (h2 : a - b = (-3, 2)) :
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by sorry

end NUMINAMATH_CALUDE_vector_square_difference_l3919_391905


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l3919_391939

/-- A tetrahedron with an inscribed sphere --/
structure Tetrahedron :=
  (r : ℝ) -- radius of inscribed sphere
  (a b : ℝ) -- lengths of a pair of opposite edges
  (r_pos : r > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Theorem: The radius of the inscribed sphere is less than ab/(2(a+b)) --/
theorem inscribed_sphere_radius_bound (t : Tetrahedron) : t.r < (t.a * t.b) / (2 * (t.a + t.b)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l3919_391939


namespace NUMINAMATH_CALUDE_triangle_count_in_specific_rectangle_l3919_391967

/-- Represents a rectangle divided by vertical and horizontal lines -/
structure DividedRectangle where
  vertical_divisions : ℕ
  horizontal_divisions : ℕ

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  let small_rectangles := r.vertical_divisions * r.horizontal_divisions
  let smallest_triangles := small_rectangles * 4
  let isosceles_by_width := small_rectangles
  let large_right_triangles := small_rectangles * 2
  let largest_isosceles := r.horizontal_divisions
  smallest_triangles + isosceles_by_width + large_right_triangles + largest_isosceles

/-- Theorem stating that a rectangle divided by 3 vertical and 2 horizontal lines contains 50 triangles -/
theorem triangle_count_in_specific_rectangle :
  let r : DividedRectangle := ⟨3, 2⟩
  count_triangles r = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_in_specific_rectangle_l3919_391967


namespace NUMINAMATH_CALUDE_max_marks_proof_l3919_391945

/-- Given a maximum mark M, calculate the passing mark as 60% of M -/
def passing_mark (M : ℝ) : ℝ := 0.6 * M

/-- The maximum marks for an exam -/
def M : ℝ := 300

/-- The marks obtained by the student -/
def obtained_marks : ℝ := 160

/-- The number of marks by which the student failed -/
def failed_by : ℝ := 20

theorem max_marks_proof :
  passing_mark M = obtained_marks + failed_by :=
sorry

end NUMINAMATH_CALUDE_max_marks_proof_l3919_391945


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l3919_391929

/-- Given that N(5,9) is the midpoint of line segment CD and C has coordinates (11,5),
    prove that the sum of the coordinates of point D is 12. -/
theorem sum_coordinates_of_D (C D N : ℝ × ℝ) : 
  C = (11, 5) → 
  N = (5, 9) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l3919_391929


namespace NUMINAMATH_CALUDE_power_of_i_2023_l3919_391962

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_2023 : i^2023 = -i := by sorry

end NUMINAMATH_CALUDE_power_of_i_2023_l3919_391962


namespace NUMINAMATH_CALUDE_vector_equality_l3919_391907

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, and D in a vector space,
    DA + CD - CB = BA -/
theorem vector_equality (A B C D : V) : D - A + (C - D) - (C - B) = B - A := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l3919_391907


namespace NUMINAMATH_CALUDE_train_crossing_platforms_l3919_391918

/-- A train crosses two platforms of different lengths. -/
theorem train_crossing_platforms
  (train_length : ℝ)
  (platform1_length : ℝ)
  (platform2_length : ℝ)
  (time1 : ℝ)
  (h1 : train_length = 350)
  (h2 : platform1_length = 100)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  : (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
  sorry

#check train_crossing_platforms

end NUMINAMATH_CALUDE_train_crossing_platforms_l3919_391918


namespace NUMINAMATH_CALUDE_percentage_theorem_l3919_391998

theorem percentage_theorem (y x z : ℝ) (h : y * x^2 + 3 * z - 6 > 0) :
  ((2 * (y * x^2 + 3 * z - 6)) / 5 + (3 * (y * x^2 + 3 * z - 6)) / 10) / (y * x^2 + 3 * z - 6) * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_theorem_l3919_391998


namespace NUMINAMATH_CALUDE_cube_distance_theorem_l3919_391942

/-- Represents a cube with a specific configuration above a plane -/
structure CubeAbovePlane where
  side_length : ℝ
  adjacent_heights : Fin 3 → ℝ
  distance_numerator : ℕ
  distance_denominator : ℕ

/-- The specific cube configuration given in the problem -/
def problem_cube : CubeAbovePlane :=
  { side_length := 12
  , adjacent_heights := ![13, 14, 16]
  , distance_numerator := 9
  , distance_denominator := 1 }

/-- Theorem stating the properties of the cube's distance from the plane -/
theorem cube_distance_theorem (cube : CubeAbovePlane) 
  (h_side : cube.side_length = 12)
  (h_heights : cube.adjacent_heights = ![13, 14, 16])
  (h_distance : ∃ (p q u : ℕ), p + q + u < 1200 ∧ 
    (cube.distance_numerator : ℝ) / cube.distance_denominator = p - Real.sqrt q) :
  cube.distance_numerator = 9 ∧ cube.distance_denominator = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_distance_theorem_l3919_391942


namespace NUMINAMATH_CALUDE_sector_properties_l3919_391937

/-- Represents a circular sector --/
structure Sector where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the perimeter of a sector --/
def sectorPerimeter (s : Sector) : ℝ :=
  2 * s.radius + s.radius * s.centralAngle

/-- Calculates the area of a sector --/
def sectorArea (s : Sector) : ℝ :=
  0.5 * s.radius * s.radius * s.centralAngle

theorem sector_properties :
  ∃ (s : Sector),
    sectorPerimeter s = 8 ∧
    (s.centralAngle = 2 → sectorArea s = 4) ∧
    (∀ (t : Sector), sectorPerimeter t = 8 → sectorArea t ≤ 4) ∧
    (sectorArea s = 4 ∧ s.centralAngle = 2) := by
  sorry

end NUMINAMATH_CALUDE_sector_properties_l3919_391937


namespace NUMINAMATH_CALUDE_min_balls_to_guarantee_color_l3919_391955

theorem min_balls_to_guarantee_color (red green yellow blue white black : ℕ) 
  (h_red : red = 30) (h_green : green = 25) (h_yellow : yellow = 22) 
  (h_blue : blue = 15) (h_white : white = 12) (h_black : black = 10) : 
  ∃ (n : ℕ), n = 95 ∧ 
  (∀ m : ℕ, m < n → 
    ∃ (r g y b w k : ℕ), r < 20 ∧ g < 20 ∧ y < 20 ∧ b < 20 ∧ w < 20 ∧ k < 20 ∧
    r + g + y + b + w + k = m ∧
    r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black) ∧
  (∀ (r g y b w k : ℕ), r + g + y + b + w + k = n →
    r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black →
    r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20 ∨ k ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_to_guarantee_color_l3919_391955


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l3919_391976

def f (n : ℤ) : ℤ := -2 * n + 3

theorem function_satisfies_equation :
  ∀ a b : ℤ, f (a + b) + f (a^2 + b^2) = f a * f b + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l3919_391976


namespace NUMINAMATH_CALUDE_triangle_side_product_range_l3919_391933

theorem triangle_side_product_range (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) 
  (h3 : y < x + 1) : 
  let t := max (1/x) (max (x/y) y) * min (1/x) (min (x/y) y)
  1 ≤ t ∧ t < (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_product_range_l3919_391933


namespace NUMINAMATH_CALUDE_distinct_lines_theorem_l3919_391963

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to determine if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Function to create a line from two points -/
def line_from_points (p q : Point) : Line :=
  { a := q.y - p.y,
    b := p.x - q.x,
    c := p.y * q.x - p.x * q.y }

/-- Function to check if two lines are distinct -/
def distinct_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a ∨ l1.a * l2.c ≠ l1.c * l2.a ∨ l1.b * l2.c ≠ l1.c * l2.b

/-- Theorem: For n points on a plane, not all collinear, there are at least n distinct lines -/
theorem distinct_lines_theorem (n : ℕ) (points : Fin n → Point) 
  (h : ∃ i j k : Fin n, ¬collinear (points i) (points j) (points k)) :
  ∃ (lines : Fin n → Line), ∀ i j : Fin n, i ≠ j → distinct_lines (lines i) (lines j) :=
sorry

end NUMINAMATH_CALUDE_distinct_lines_theorem_l3919_391963


namespace NUMINAMATH_CALUDE_equation_solution_l3919_391913

theorem equation_solution : ∃! x : ℝ, 0.05 * x + 0.12 * (30 + x) = 15.84 ∧ x = 72 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3919_391913


namespace NUMINAMATH_CALUDE_max_square_cookies_l3919_391986

theorem max_square_cookies (length width : ℕ) (h1 : length = 24) (h2 : width = 18) :
  let cookie_size := Nat.gcd length width
  (length / cookie_size) * (width / cookie_size) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_square_cookies_l3919_391986


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l3919_391961

theorem smallest_number_in_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 29 →                 -- Median is 29
  c = b + 7 →              -- Largest number is 7 more than median
  a < b ∧ b < c →          -- Ensuring order: a < b < c
  a = 25 :=                -- Smallest number is 25
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l3919_391961


namespace NUMINAMATH_CALUDE_triangle_radius_equations_l3919_391941

/-- Given a triangle ABC with angles 2α, 2β, and 2γ, prove two equations involving inradius, exradii, and side lengths. -/
theorem triangle_radius_equations (R α β γ : ℝ) (r r_a r_b r_c a b c : ℝ) 
  (h_r : r = 4 * R * Real.sin α * Real.sin β * Real.sin γ)
  (h_ra : r_a = 4 * R * Real.sin α * Real.cos β * Real.cos γ)
  (h_rb : r_b = 4 * R * Real.cos α * Real.sin β * Real.cos γ)
  (h_rc : r_c = 4 * R * Real.cos α * Real.cos β * Real.sin γ)
  (h_a : a = 4 * R * Real.sin α * Real.cos α)
  (h_bc : b + c = 4 * R * Real.sin (β + γ) * Real.cos (β - γ)) :
  (a * (b + c) = (r + r_a) * (4 * R + r - r_a)) ∧ 
  (a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_radius_equations_l3919_391941


namespace NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_coefficients_l3919_391932

-- Definition of a double root equation
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

-- Theorem 1
theorem first_equation_is_double_root :
  is_double_root_equation 1 (-3) 2 :=
sorry

-- Theorem 2
theorem second_equation_coefficients (a b : ℝ) :
  is_double_root_equation a b (-6) ∧ (a * 2^2 + b * 2 - 6 = 0) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_coefficients_l3919_391932


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l3919_391971

/-- Represents the length of a marathon in miles and yards -/
structure Marathon :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a distance in miles and yards -/
structure Distance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_length : Marathon :=
  { miles := 30, yards := 520 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 8

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), y < yards_per_mile ∧
    Distance.yards (
      { miles := m
      , yards := y
      } : Distance
    ) = 640 ∧
    Distance.miles (
      { miles := m
      , yards := y
      } : Distance
    ) * yards_per_mile + y =
    num_marathons * (marathon_length.miles * yards_per_mile + marathon_length.yards) :=
by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l3919_391971


namespace NUMINAMATH_CALUDE_friends_with_oranges_l3919_391970

theorem friends_with_oranges (total_friends : ℕ) (friends_with_pears : ℕ) : 
  total_friends = 15 → friends_with_pears = 9 → total_friends - friends_with_pears = 6 := by
  sorry

end NUMINAMATH_CALUDE_friends_with_oranges_l3919_391970


namespace NUMINAMATH_CALUDE_prob_king_hearts_or_spade_l3919_391927

-- Define the total number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of spades in the deck
def num_spades : ℕ := 13

-- Define the probability of drawing the King of Hearts
def prob_king_hearts : ℚ := 1 / total_cards

-- Define the probability of drawing a Spade
def prob_spade : ℚ := num_spades / total_cards

-- Theorem to prove
theorem prob_king_hearts_or_spade :
  prob_king_hearts + prob_spade = 7 / 26 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_hearts_or_spade_l3919_391927


namespace NUMINAMATH_CALUDE_salary_increase_l3919_391975

theorem salary_increase (starting_salary current_salary : ℝ) 
  (h1 : starting_salary = 80000)
  (h2 : current_salary = 134400)
  (h3 : current_salary = 1.2 * (starting_salary * 1.4)) :
  starting_salary * 1.4 = starting_salary + 0.4 * starting_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_l3919_391975


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3919_391992

theorem remainder_444_power_444_mod_13 : 444^444 ≡ 1 [MOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3919_391992


namespace NUMINAMATH_CALUDE_seats_taken_l3919_391909

theorem seats_taken (rows : ℕ) (chairs_per_row : ℕ) (unoccupied : ℕ) : 
  rows = 40 → chairs_per_row = 20 → unoccupied = 10 → 
  rows * chairs_per_row - unoccupied = 790 := by
  sorry

end NUMINAMATH_CALUDE_seats_taken_l3919_391909


namespace NUMINAMATH_CALUDE_sallys_initial_cards_l3919_391908

/-- Proves that Sally initially had 27 Pokemon cards given the problem conditions --/
theorem sallys_initial_cards (x : ℕ) : x + 20 = 41 + 6 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sallys_initial_cards_l3919_391908


namespace NUMINAMATH_CALUDE_volume_conversion_m_to_dm_volume_conversion_mL_to_L_volume_conversion_cm_to_dm_l3919_391988

-- Define conversion factors
def m_to_dm : ℝ := 10
def L_to_mL : ℝ := 1000
def dm_to_cm : ℝ := 10

-- Theorem statements
theorem volume_conversion_m_to_dm : 
  20 * (m_to_dm ^ 3) = 20000 := by sorry

theorem volume_conversion_mL_to_L : 
  15 / L_to_mL = 0.015 := by sorry

theorem volume_conversion_cm_to_dm : 
  1200 / (dm_to_cm ^ 3) = 1.2 := by sorry

end NUMINAMATH_CALUDE_volume_conversion_m_to_dm_volume_conversion_mL_to_L_volume_conversion_cm_to_dm_l3919_391988


namespace NUMINAMATH_CALUDE_union_when_m_is_neg_one_subset_iff_m_leq_neg_two_disjoint_iff_m_geq_zero_l3919_391902

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Statement 1
theorem union_when_m_is_neg_one : 
  A ∪ B (-1) = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Statement 2
theorem subset_iff_m_leq_neg_two :
  ∀ m : ℝ, A ⊆ B m ↔ m ≤ -2 := by sorry

-- Statement 3
theorem disjoint_iff_m_geq_zero :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_union_when_m_is_neg_one_subset_iff_m_leq_neg_two_disjoint_iff_m_geq_zero_l3919_391902


namespace NUMINAMATH_CALUDE_new_person_weight_l3919_391996

/-- Given two people, where one weighs 65 kg, if replacing that person with a new person
    increases the average weight by 4.5 kg, then the new person weighs 74 kg. -/
theorem new_person_weight (initial_weight : ℝ) : 
  let total_initial_weight := initial_weight + 65
  let new_average_weight := (total_initial_weight / 2) + 4.5
  let new_total_weight := new_average_weight * 2
  new_total_weight - initial_weight = 74 := by
sorry


end NUMINAMATH_CALUDE_new_person_weight_l3919_391996


namespace NUMINAMATH_CALUDE_negation_equivalence_l3919_391950

/-- The original proposition p -/
def p : Prop := ∀ x : ℝ, x^2 + x - 6 ≤ 0

/-- The proposed negation of p -/
def q : Prop := ∃ x : ℝ, x^2 + x - 6 > 0

/-- Theorem stating that q is the negation of p -/
theorem negation_equivalence : ¬p ↔ q := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3919_391950


namespace NUMINAMATH_CALUDE_line_symmetry_l3919_391915

-- Define the lines
def line_l (x y : ℝ) : Prop := 3 * x - y + 3 = 0
def line_1 (x y : ℝ) : Prop := x - y - 2 = 0
def line_2 (x y : ℝ) : Prop := 7 * x + y + 22 = 0

-- Define symmetry with respect to line_l
def symmetric_wrt_l (x y x' y' : ℝ) : Prop :=
  -- The product of the slopes of PP' and line_l is -1
  ((y' - y) / (x' - x)) * 3 = -1 ∧
  -- The midpoint of PP' lies on line_l
  3 * ((x + x') / 2) - ((y + y') / 2) + 3 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ x y x' y' : ℝ,
    line_1 x y ∧ line_2 x' y' →
    symmetric_wrt_l x y x' y' :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l3919_391915


namespace NUMINAMATH_CALUDE_twenty_fifth_digit_sum_l3919_391953

/-- The decimal representation of 1/9 -/
def decimal_1_9 : ℚ := 1/9

/-- The decimal representation of 1/11 -/
def decimal_1_11 : ℚ := 1/11

/-- The sum of the decimal representations of 1/9 and 1/11 -/
def sum_decimals : ℚ := decimal_1_9 + decimal_1_11

/-- The 25th digit after the decimal point in a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem twenty_fifth_digit_sum :
  nth_digit_after_decimal sum_decimals 25 = 2 := by sorry

end NUMINAMATH_CALUDE_twenty_fifth_digit_sum_l3919_391953


namespace NUMINAMATH_CALUDE_opposite_of_two_l3919_391989

theorem opposite_of_two : -(2 : ℝ) = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_l3919_391989


namespace NUMINAMATH_CALUDE_three_numbers_product_l3919_391990

theorem three_numbers_product (x y z : ℤ) : 
  x + y + z = 165 ∧ 
  7 * x = y - 9 ∧ 
  7 * x = z + 9 → 
  x * y * z = 64328 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_product_l3919_391990


namespace NUMINAMATH_CALUDE_log_ride_cost_l3919_391924

/-- The cost of a single log ride, given the following conditions:
  * Dolly wants to ride the Ferris wheel twice
  * Dolly wants to ride the roller coaster three times
  * Dolly wants to ride the log ride seven times
  * The Ferris wheel costs 2 tickets per ride
  * The roller coaster costs 5 tickets per ride
  * Dolly has 20 tickets
  * Dolly needs to buy 6 more tickets
-/
theorem log_ride_cost : ℕ := by
  sorry

#check log_ride_cost

end NUMINAMATH_CALUDE_log_ride_cost_l3919_391924


namespace NUMINAMATH_CALUDE_octal_74532_to_decimal_l3919_391947

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

theorem octal_74532_to_decimal :
  octal_to_decimal [2, 3, 5, 4, 7] = 31066 := by
  sorry

end NUMINAMATH_CALUDE_octal_74532_to_decimal_l3919_391947


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3919_391921

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 6}
def B : Set ℤ := {1, 4, 5}

theorem intersection_complement_equals : A ∩ (U \ B) = {3, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3919_391921


namespace NUMINAMATH_CALUDE_f_has_minimum_at_6_l3919_391978

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 32

/-- Theorem stating that f has a minimum at x = 6 -/
theorem f_has_minimum_at_6 : 
  ∀ x : ℝ, f x ≥ f 6 := by sorry

end NUMINAMATH_CALUDE_f_has_minimum_at_6_l3919_391978


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l3919_391916

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.3 : ℝ)⌉ = 27 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l3919_391916


namespace NUMINAMATH_CALUDE_team_total_score_l3919_391938

def team_score (connor_score : ℕ) (amy_score : ℕ) (jason_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score

theorem team_total_score :
  ∀ (connor_score amy_score jason_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    team_score connor_score amy_score jason_score = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_team_total_score_l3919_391938


namespace NUMINAMATH_CALUDE_mittens_count_l3919_391914

theorem mittens_count (original_plugs current_plugs mittens : ℕ) : 
  mittens = original_plugs - 20 →
  current_plugs = original_plugs + 30 →
  400 = 2 * current_plugs →
  mittens = 150 := by
  sorry

end NUMINAMATH_CALUDE_mittens_count_l3919_391914


namespace NUMINAMATH_CALUDE_f_increasing_neg_f_max_neg_l3919_391985

/-- An odd function that is increasing on [3, 7] with minimum value 5 -/
def f : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f is increasing on [3, 7] -/
axiom f_increasing_pos : ∀ x y, 3 ≤ x ∧ x < y ∧ y ≤ 7 → f x < f y

/-- The minimum value of f on [3, 7] is 5 -/
axiom f_min_pos : ∃ x₀, 3 ≤ x₀ ∧ x₀ ≤ 7 ∧ f x₀ = 5 ∧ ∀ x, 3 ≤ x ∧ x ≤ 7 → f x₀ ≤ f x

/-- f is increasing on [-7, -3] -/
theorem f_increasing_neg : ∀ x y, -7 ≤ x ∧ x < y ∧ y ≤ -3 → f x < f y :=
sorry

/-- The maximum value of f on [-7, -3] is -5 -/
theorem f_max_neg : ∃ x₀, -7 ≤ x₀ ∧ x₀ ≤ -3 ∧ f x₀ = -5 ∧ ∀ x, -7 ≤ x ∧ x ≤ -3 → f x ≤ f x₀ :=
sorry

end NUMINAMATH_CALUDE_f_increasing_neg_f_max_neg_l3919_391985


namespace NUMINAMATH_CALUDE_tire_price_calculation_l3919_391906

theorem tire_price_calculation (num_tires sale_price total_savings : ℕ) 
  (h1 : num_tires = 4)
  (h2 : sale_price = 75)
  (h3 : total_savings = 36)
  : sale_price + total_savings / num_tires = 84 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_calculation_l3919_391906


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l3919_391960

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 3*a|

-- Theorem 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 1 ∨ a = -1 :=
sorry

-- Theorem 2
theorem inequality_implies_m_range (m : ℝ) :
  (∀ x, ∃ a ∈ Set.Icc (-1) 1, m^2 - |m| - f x a < 0) → -2 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l3919_391960


namespace NUMINAMATH_CALUDE_translation_theorem_l3919_391912

/-- The original function f(x) = 2x^2 - 2x -/
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x

/-- The transformed function g(x) = 2x^2 - 10x - 9 -/
def g (x : ℝ) : ℝ := 2 * x^2 - 10 * x - 9

/-- Theorem stating that g is the result of translating f 2 units right and 3 units down -/
theorem translation_theorem : ∀ x : ℝ, g x = f (x - 2) - 3 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l3919_391912


namespace NUMINAMATH_CALUDE_square_overlap_area_l3919_391956

theorem square_overlap_area (β : Real) (h1 : 0 < β) (h2 : β < Real.pi / 2) (h3 : Real.cos β = 3/5) :
  let square_side : Real := 2
  let overlap_area : Real := 
    2 * (square_side * (1 - Real.tan (β/2)) / (1 + Real.tan (β/2))) * square_side / 2
  overlap_area = 4/3 := by sorry

end NUMINAMATH_CALUDE_square_overlap_area_l3919_391956


namespace NUMINAMATH_CALUDE_cake_recipe_proof_l3919_391949

def baking_problem (total_flour sugar_needed flour_added : ℕ) : Prop :=
  total_flour - flour_added - sugar_needed = 5

theorem cake_recipe_proof :
  baking_problem 10 3 2 := by sorry

end NUMINAMATH_CALUDE_cake_recipe_proof_l3919_391949


namespace NUMINAMATH_CALUDE_least_bench_sections_l3919_391977

theorem least_bench_sections (M : ℕ) : M > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k < M → ¬(120 ∣ 8*k ∧ 120 ∣ 12*k ∧ 120 ∣ 10*k)) ∧ 
  (120 ∣ 8*M ∧ 120 ∣ 12*M ∧ 120 ∣ 10*M) → M = 15 := by
  sorry

end NUMINAMATH_CALUDE_least_bench_sections_l3919_391977


namespace NUMINAMATH_CALUDE_susie_bob_ratio_l3919_391935

-- Define the number of slices for each pizza size
def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8

-- Define the number of pizzas George purchased
def small_pizzas_bought : ℕ := 3
def large_pizzas_bought : ℕ := 2

-- Define the number of pieces eaten by each person
def george_pieces : ℕ := 3
def bob_pieces : ℕ := george_pieces + 1
def bill_fred_mark_pieces : ℕ := 3 * 3

-- Define the number of slices left over
def leftover_slices : ℕ := 10

-- Calculate the total number of slices
def total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought

-- Define Susie's pieces as a function of the other variables
def susie_pieces : ℕ := total_slices - leftover_slices - (george_pieces + bob_pieces + bill_fred_mark_pieces)

-- Theorem to prove
theorem susie_bob_ratio :
  susie_pieces * 2 = bob_pieces := by sorry

end NUMINAMATH_CALUDE_susie_bob_ratio_l3919_391935


namespace NUMINAMATH_CALUDE_equation_solution_l3919_391966

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3919_391966


namespace NUMINAMATH_CALUDE_main_project_time_l3919_391969

def total_days : ℕ := 4
def hours_per_day : ℝ := 8
def time_on_smaller_tasks : ℝ := 9
def time_on_naps : ℝ := 13.5

theorem main_project_time :
  total_days * hours_per_day - time_on_smaller_tasks - time_on_naps = 9.5 := by
sorry

end NUMINAMATH_CALUDE_main_project_time_l3919_391969


namespace NUMINAMATH_CALUDE_xy_value_l3919_391931

theorem xy_value (x y : ℝ) 
  (h1 : x + y = 2) 
  (h2 : x^2 * y^3 + y^2 * x^3 = 32) : 
  x * y = -8 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3919_391931


namespace NUMINAMATH_CALUDE_sum_squares_ge_product_sum_l3919_391948

theorem sum_squares_ge_product_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ x₁ * (x₂ + x₃ + x₄ + x₅) := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_ge_product_sum_l3919_391948


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l3919_391917

theorem lawn_mowing_earnings (total_lawns : ℕ) (forgotten_lawns : ℕ) (total_earned : ℕ) :
  total_lawns = 12 →
  forgotten_lawns = 8 →
  total_earned = 36 →
  (total_earned : ℚ) / ((total_lawns - forgotten_lawns) : ℚ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l3919_391917


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l3919_391930

/-- Given a sinusoidal function with specific properties, prove its exact form and the set of x-values where it equals 1. -/
theorem sinusoidal_function_properties (A ω φ : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h2 : A > 0)
  (h3 : ω > 0)
  (h4 : |φ| < π)
  (h5 : f (π/8) = 2)
  (h6 : f (5*π/8) = -2) :
  (∀ x, f x = 2 * Real.sin (2*x + π/4)) ∧ 
  (∀ x, f x = 1 ↔ ∃ k : ℤ, x = -π/24 + k*π ∨ x = 7*π/24 + k*π) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l3919_391930


namespace NUMINAMATH_CALUDE_ratio_HC_JE_l3919_391999

-- Define the points
variable (A B C D E F G H J K : ℝ × ℝ)

-- Define the conditions
axiom points_on_line : ∃ (t : ℝ), A = (0, 0) ∧ B = (1, 0) ∧ C = (3, 0) ∧ D = (4, 0) ∧ E = (5, 0) ∧ F = (7, 0)
axiom G_off_line : G.2 ≠ 0
axiom H_on_GD : ∃ (t : ℝ), H = G + t • (D - G)
axiom J_on_GF : ∃ (t : ℝ), J = G + t • (F - G)
axiom K_on_GB : ∃ (t : ℝ), K = G + t • (B - G)
axiom parallel_lines : ∃ (k : ℝ), 
  H - C = k • (G - A) ∧ 
  J - E = k • (G - A) ∧ 
  K - B = k • (G - A)

-- Define the theorem
theorem ratio_HC_JE : 
  (H.1 - C.1) / (J.1 - E.1) = 7/8 :=
sorry

end NUMINAMATH_CALUDE_ratio_HC_JE_l3919_391999


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3919_391991

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 3, 4}

theorem complement_M_intersect_N : (U \ M) ∩ N = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3919_391991


namespace NUMINAMATH_CALUDE_skylar_age_l3919_391943

/-- Represents the age when Skylar started donating -/
def starting_age : ℕ := 17

/-- Represents the annual donation amount in thousands -/
def annual_donation : ℕ := 8

/-- Represents the total amount donated in thousands -/
def total_donated : ℕ := 440

/-- Calculates Skylar's current age -/
def current_age : ℕ := starting_age + (total_donated / annual_donation)

/-- Proves that Skylar's current age is 72 years -/
theorem skylar_age : current_age = 72 := by
  sorry

end NUMINAMATH_CALUDE_skylar_age_l3919_391943


namespace NUMINAMATH_CALUDE_union_of_sets_l3919_391980

/-- Given sets M and N, prove that their union is equal to the set of all x between -1 and 5 inclusive -/
theorem union_of_sets (M N : Set ℝ) (hM : M = {x : ℝ | -1 ≤ x ∧ x < 3}) (hN : N = {x : ℝ | 2 < x ∧ x ≤ 5}) :
  M ∪ N = {x : ℝ | -1 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3919_391980


namespace NUMINAMATH_CALUDE_extrema_of_quadratic_form_l3919_391923

theorem extrema_of_quadratic_form (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  1 ≤ x^2 + 2*y^2 + 3*z^2 ∧ x^2 + 2*y^2 + 3*z^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_extrema_of_quadratic_form_l3919_391923


namespace NUMINAMATH_CALUDE_pyramid_with_base_six_has_56_apples_l3919_391972

/-- Calculates the number of apples in a triangular layer -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a triangular pyramid stack of apples -/
structure ApplePyramid where
  base_side : ℕ
  inv_base_side_pos : 0 < base_side

/-- Calculates the total number of apples in the pyramid -/
def total_apples (pyramid : ApplePyramid) : ℕ :=
  (List.range pyramid.base_side).map triangular_number |>.sum

/-- The theorem stating that a pyramid with base side 6 contains 56 apples -/
theorem pyramid_with_base_six_has_56_apples :
  ∃ (pyramid : ApplePyramid), pyramid.base_side = 6 ∧ total_apples pyramid = 56 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_with_base_six_has_56_apples_l3919_391972


namespace NUMINAMATH_CALUDE_triangle_ratio_l3919_391965

theorem triangle_ratio (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * Real.sin A * Real.sin B + a * (Real.cos B)^2 = 2 * c →
  a / c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3919_391965


namespace NUMINAMATH_CALUDE_friend_initial_savings_l3919_391944

/-- Proves that given the conditions of the savings problem, the friend's initial amount is $210 --/
theorem friend_initial_savings (your_initial : ℕ) (your_weekly : ℕ) (friend_weekly : ℕ) (weeks : ℕ) 
  (h1 : your_initial = 160)
  (h2 : your_weekly = 7)
  (h3 : friend_weekly = 5)
  (h4 : weeks = 25)
  (h5 : your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks) :
  friend_initial = 210 := by
  sorry

#check friend_initial_savings

end NUMINAMATH_CALUDE_friend_initial_savings_l3919_391944


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3919_391903

def original_price : ℝ := 77.95
def sale_price : ℝ := 59.95

theorem price_decrease_percentage :
  let difference := original_price - sale_price
  let percentage_decrease := (difference / original_price) * 100
  ∃ ε > 0, abs (percentage_decrease - 23.08) < ε :=
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3919_391903


namespace NUMINAMATH_CALUDE_newspaper_cost_theorem_l3919_391920

/-- The cost of a weekday newspaper -/
def weekday_cost : ℚ := 1/2

/-- The cost of a Sunday newspaper -/
def sunday_cost : ℚ := 2

/-- The number of weekday newspapers bought per week -/
def weekday_papers_per_week : ℕ := 3

/-- The number of weeks -/
def num_weeks : ℕ := 8

/-- The total cost of newspapers over the given number of weeks -/
def total_cost : ℚ := num_weeks * (weekday_papers_per_week * weekday_cost + sunday_cost)

theorem newspaper_cost_theorem : total_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_theorem_l3919_391920


namespace NUMINAMATH_CALUDE_lucy_current_age_l3919_391982

/-- Lucy's current age -/
def lucy_age : ℕ := sorry

/-- Lovely's current age -/
def lovely_age : ℕ := sorry

/-- Lucy's age was three times Lovely's age 5 years ago -/
axiom past_age_relation : lucy_age - 5 = 3 * (lovely_age - 5)

/-- Lucy's age will be twice Lovely's age 10 years from now -/
axiom future_age_relation : lucy_age + 10 = 2 * (lovely_age + 10)

/-- Lucy's current age is 50 -/
theorem lucy_current_age : lucy_age = 50 := by sorry

end NUMINAMATH_CALUDE_lucy_current_age_l3919_391982


namespace NUMINAMATH_CALUDE_trig_identity_l3919_391911

theorem trig_identity (α : Real) 
  (h : Real.sin α - Real.cos α = -7/5) : 
  (Real.sin α * Real.cos α = -12/25) ∧ 
  ((Real.tan α = -3/4) ∨ (Real.tan α = -4/3)) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3919_391911


namespace NUMINAMATH_CALUDE_x_power_five_minus_twenty_seven_x_squared_l3919_391983

theorem x_power_five_minus_twenty_seven_x_squared (x : ℝ) (h : x^3 - 3*x = 5) :
  x^5 - 27*x^2 = -22*x^2 + 9*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_x_power_five_minus_twenty_seven_x_squared_l3919_391983


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l3919_391952

theorem walnut_trees_planted (initial_trees final_trees : ℕ) 
  (h1 : initial_trees = 22)
  (h2 : final_trees = 55) :
  final_trees - initial_trees = 33 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l3919_391952


namespace NUMINAMATH_CALUDE_watch_cost_price_proof_l3919_391934

/-- The cost price of a watch satisfying certain selling conditions -/
def watch_cost_price : ℝ := 875

/-- The selling price of the watch at a loss -/
def selling_price_loss : ℝ := watch_cost_price * (1 - 0.12)

/-- The selling price of the watch at a gain -/
def selling_price_gain : ℝ := watch_cost_price * (1 + 0.04)

/-- Theorem stating the cost price of the watch given the selling conditions -/
theorem watch_cost_price_proof :
  (selling_price_loss = watch_cost_price * (1 - 0.12)) ∧
  (selling_price_gain = watch_cost_price * (1 + 0.04)) ∧
  (selling_price_gain - selling_price_loss = 140) →
  watch_cost_price = 875 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_proof_l3919_391934


namespace NUMINAMATH_CALUDE_exponent_division_l3919_391940

theorem exponent_division (x : ℝ) (hx : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3919_391940


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_mans_upstream_speed_is_twelve_l3919_391994

/-- Calculates the speed of a man rowing upstream given his speed in still water and his speed downstream. -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that for a given man's speed in still water and downstream, 
    his upstream speed is equal to twice his still water speed minus his downstream speed. -/
theorem upstream_speed_calculation 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still > 0)
  (h2 : speed_downstream > speed_still) :
  speed_upstream speed_still speed_downstream = 2 * speed_still - speed_downstream :=
by sorry

/-- The speed of the man rowing upstream in the given problem. -/
def mans_upstream_speed : ℝ := speed_upstream 25 38

/-- Theorem proving that the man's upstream speed in the given problem is 12 km/h. -/
theorem mans_upstream_speed_is_twelve : 
  mans_upstream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_upstream_speed_calculation_mans_upstream_speed_is_twelve_l3919_391994


namespace NUMINAMATH_CALUDE_l_companion_properties_l3919_391958

/-- Definition of an l-companion function -/
def is_l_companion (f : ℝ → ℝ) (l : ℝ) : Prop :=
  l ≠ 0 ∧ Continuous f ∧ ∀ x : ℝ, f (x + l) + l * f x = 0

theorem l_companion_properties (f : ℝ → ℝ) (l : ℝ) (h : is_l_companion f l) :
  (∀ c : ℝ, is_l_companion (λ _ => c) l → c = 0) ∧
  ¬ is_l_companion (λ x => x) l ∧
  ¬ is_l_companion (λ x => x^2) l ∧
  ∃ x : ℝ, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_l_companion_properties_l3919_391958


namespace NUMINAMATH_CALUDE_compare_fractions_l3919_391946

theorem compare_fractions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / (1 + x + y) < x / (1 + x) + y / (1 + y) := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l3919_391946


namespace NUMINAMATH_CALUDE_power_sum_problem_l3919_391993

theorem power_sum_problem : ∃ (x y n : ℕ+), 
  (x * y = 6) ∧ 
  (x ^ n.val + y ^ n.val = 35) ∧ 
  (∀ m : ℕ+, m < n → x ^ m.val + y ^ m.val ≠ 35) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l3919_391993


namespace NUMINAMATH_CALUDE_M_subset_N_l3919_391954

/-- Set M definition -/
def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}

/-- Set N definition -/
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

/-- Theorem stating that M is a subset of N -/
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3919_391954


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_nonpositive_l3919_391979

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem f_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_nonpositive_l3919_391979


namespace NUMINAMATH_CALUDE_sum_range_l3919_391951

theorem sum_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) 
  (h : a^2 - a + b^2 - b + a*b = 0) : 
  1 < a + b ∧ a + b < 4/3 := by
sorry

end NUMINAMATH_CALUDE_sum_range_l3919_391951


namespace NUMINAMATH_CALUDE_square_circumscribed_circle_radius_l3919_391984

/-- Given a square with perimeter x and circumscribed circle radius y, prove that y = (√2 / 8) * x -/
theorem square_circumscribed_circle_radius (x y : ℝ) 
  (h_perimeter : x > 0) -- Ensure positive perimeter
  (h_square : ∃ (s : ℝ), s > 0 ∧ 4 * s = x) -- Existence of square side length
  (h_circumscribed : y > 0) -- Ensure positive radius
  : y = (Real.sqrt 2 / 8) * x := by
  sorry

end NUMINAMATH_CALUDE_square_circumscribed_circle_radius_l3919_391984


namespace NUMINAMATH_CALUDE_leaf_fall_problem_l3919_391926

/-- The rate of leaves falling per hour in the second and third hour -/
def leaf_fall_rate (first_hour : ℕ) (average : ℚ) : ℚ :=
  (3 * average - first_hour) / 2

theorem leaf_fall_problem (first_hour : ℕ) (average : ℚ) :
  first_hour = 7 →
  average = 5 →
  leaf_fall_rate first_hour average = 4 := by
sorry

end NUMINAMATH_CALUDE_leaf_fall_problem_l3919_391926


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l3919_391987

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  55 * p^9 * q^2 = 165 * p^8 * q^3 → 
  p = 3/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l3919_391987


namespace NUMINAMATH_CALUDE_gift_wrapping_l3919_391910

theorem gift_wrapping (total_rolls total_gifts first_roll_gifts second_roll_gifts : ℕ) :
  total_rolls = 3 →
  total_gifts = 12 →
  first_roll_gifts = 3 →
  second_roll_gifts = 5 →
  total_gifts = first_roll_gifts + second_roll_gifts + (total_gifts - (first_roll_gifts + second_roll_gifts)) →
  (total_gifts - (first_roll_gifts + second_roll_gifts)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_l3919_391910


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3919_391919

theorem cube_root_simplification : 
  (54880000 : ℝ)^(1/3) = 140 * 20^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3919_391919


namespace NUMINAMATH_CALUDE_initial_geese_count_l3919_391936

theorem initial_geese_count (initial_count : ℕ) : 
  (initial_count / 2 + 4 = 12) → initial_count = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_geese_count_l3919_391936


namespace NUMINAMATH_CALUDE_full_price_revenue_l3919_391957

/-- Represents a concert ticket sale scenario -/
structure ConcertSale where
  fullPrice : ℕ  -- Number of full-price tickets
  discountPrice : ℕ  -- Number of discount-price tickets
  price : ℕ  -- Price of a full-price ticket in dollars

/-- Conditions for a valid concert sale -/
def isValidSale (sale : ConcertSale) : Prop :=
  sale.fullPrice + sale.discountPrice = 200 ∧
  sale.fullPrice * sale.price + sale.discountPrice * (sale.price / 3) = 3000

/-- Theorem stating the revenue from full-price tickets -/
theorem full_price_revenue (sale : ConcertSale) 
  (h : isValidSale sale) : sale.fullPrice * sale.price = 1500 := by
  sorry


end NUMINAMATH_CALUDE_full_price_revenue_l3919_391957


namespace NUMINAMATH_CALUDE_jerrys_shelf_books_l3919_391900

/-- The number of books on Jerry's shelf -/
def books : ℕ := 9

/-- The initial number of action figures -/
def initial_figures : ℕ := 5

/-- The number of action figures added -/
def added_figures : ℕ := 7

/-- The difference between action figures and books -/
def figure_book_difference : ℕ := 3

theorem jerrys_shelf_books :
  books = initial_figures + added_figures - figure_book_difference := by
  sorry

end NUMINAMATH_CALUDE_jerrys_shelf_books_l3919_391900


namespace NUMINAMATH_CALUDE_shared_foci_hyperbola_ellipse_l3919_391997

theorem shared_foci_hyperbola_ellipse (a : ℝ) : 
  (∀ x y : ℝ, x^2 / (a + 1) - y^2 = 1 ↔ x^2 / 4 + y^2 / a^2 = 1) →
  a + 1 > 0 →
  4 > a^2 →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_shared_foci_hyperbola_ellipse_l3919_391997


namespace NUMINAMATH_CALUDE_birch_not_adjacent_probability_l3919_391925

def maple_count : ℕ := 3
def oak_count : ℕ := 4
def birch_count : ℕ := 5

def total_trees : ℕ := maple_count + oak_count + birch_count

def total_arrangements : ℕ := Nat.factorial total_trees / (Nat.factorial maple_count * Nat.factorial oak_count * Nat.factorial birch_count)

def favorable_arrangements : ℕ := (Nat.choose (maple_count + oak_count + 1) birch_count) * (Nat.factorial (maple_count + oak_count))

theorem birch_not_adjacent_probability : 
  (favorable_arrangements : ℚ) / total_arrangements = 7 / 99 := by sorry

end NUMINAMATH_CALUDE_birch_not_adjacent_probability_l3919_391925


namespace NUMINAMATH_CALUDE_milk_drinking_l3919_391973

theorem milk_drinking (total_milk : ℚ) (drunk_fraction : ℚ) : 
  total_milk = 1/4 → drunk_fraction = 3/4 → drunk_fraction * total_milk = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_milk_drinking_l3919_391973


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l3919_391995

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101₂ -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

/-- Theorem stating that 110101₂ equals 53 in decimal -/
theorem binary_110101_equals_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l3919_391995


namespace NUMINAMATH_CALUDE_intersection_A_B_l3919_391981

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3919_391981


namespace NUMINAMATH_CALUDE_find_divisor_l3919_391959

theorem find_divisor (N : ℕ) (D : ℕ) (h1 : N = 269 * D) (h2 : N % 67 = 1) : D = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3919_391959


namespace NUMINAMATH_CALUDE_infinite_prime_factors_and_non_factors_l3919_391901

def sequence_a : ℕ → ℕ
  | 0 => 4
  | n + 1 => sequence_a n * (sequence_a n - 1)

def prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

def prime_factors_of_sequence : Set ℕ :=
  ⋃ n, prime_factors (sequence_a n)

def primes_not_dividing_sequence : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ n, ¬(p ∣ sequence_a n)}

theorem infinite_prime_factors_and_non_factors :
  (Set.Infinite prime_factors_of_sequence) ∧
  (Set.Infinite primes_not_dividing_sequence) :=
sorry

end NUMINAMATH_CALUDE_infinite_prime_factors_and_non_factors_l3919_391901


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3919_391904

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  c = 65 →                 -- Hypotenuse length
  a < b →                  -- a is the shorter leg
  a = 39 :=                -- Shorter leg length
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3919_391904


namespace NUMINAMATH_CALUDE_village_new_average_age_l3919_391964

/-- Represents the population data of a village --/
structure VillagePopulation where
  men_ratio : ℚ
  women_ratio : ℚ
  men_increase : ℚ
  men_avg_age : ℚ
  women_avg_age : ℚ

/-- Calculates the new average age of the population after men's population increase --/
def new_average_age (v : VillagePopulation) : ℚ :=
  let new_men_ratio := v.men_ratio * (1 + v.men_increase)
  let total_population := new_men_ratio + v.women_ratio
  let total_age := new_men_ratio * v.men_avg_age + v.women_ratio * v.women_avg_age
  total_age / total_population

/-- Theorem stating that the new average age is approximately 37.3 years --/
theorem village_new_average_age :
  let v : VillagePopulation := {
    men_ratio := 3,
    women_ratio := 4,
    men_increase := 1/10,
    men_avg_age := 40,
    women_avg_age := 35
  }
  ∃ ε > 0, |new_average_age v - 37.3| < ε :=
sorry

end NUMINAMATH_CALUDE_village_new_average_age_l3919_391964


namespace NUMINAMATH_CALUDE_chinese_chess_tournament_l3919_391928

-- Define the winning relation
def Wins (n : ℕ) : (ℕ → ℕ → Prop) := sorry

-- Main theorem
theorem chinese_chess_tournament (n : ℕ) (h : n ≥ 2) :
  ∃ (P : ℕ → ℕ → ℕ),
    (∀ i j i' j', i ≤ n ∧ j ≤ n ∧ i' ≤ n ∧ j' ≤ n ∧ (i, j) ≠ (i', j') → P i j ≠ P i' j') ∧ 
    (∀ i j, i ≤ n ∧ j ≤ n → P i j ≤ 2*n^2) ∧
    (∀ i j i' j', i < i' ∧ i ≤ n ∧ j ≤ n ∧ i' ≤ n ∧ j' ≤ n → Wins n (P i j) (P i' j')) :=
by
  sorry

-- Transitive property of winning
axiom wins_trans (n : ℕ) : ∀ a b c, Wins n a b → Wins n b c → Wins n a c

-- Maximum number of draws
axiom max_draws (n : ℕ) : ∃ (draw_count : ℕ), draw_count ≤ n^3/16 ∧ 
  ∀ a b, a ≤ 2*n^2 ∧ b ≤ 2*n^2 ∧ a ≠ b → (Wins n a b ∨ Wins n b a ∨ (¬Wins n a b ∧ ¬Wins n b a))

end NUMINAMATH_CALUDE_chinese_chess_tournament_l3919_391928


namespace NUMINAMATH_CALUDE_birds_remaining_after_week_l3919_391968

/-- Calculates the number of birds remaining after a week given initial counts and daily losses. -/
def birdsRemaining (initialChickens initialTurkeys initialGuineaFowls : ℕ)
  (oddDayLossChickens oddDayLossTurkeys oddDayLossGuineaFowls : ℕ)
  (evenDayLossChickens evenDayLossTurkeys evenDayLossGuineaFowls : ℕ) : ℕ :=
  let oddDays := 4
  let evenDays := 3
  let remainingChickens := initialChickens - (oddDays * oddDayLossChickens + evenDays * evenDayLossChickens)
  let remainingTurkeys := initialTurkeys - (oddDays * oddDayLossTurkeys + evenDays * evenDayLossTurkeys)
  let remainingGuineaFowls := initialGuineaFowls - (oddDays * oddDayLossGuineaFowls + evenDays * evenDayLossGuineaFowls)
  remainingChickens + remainingTurkeys + remainingGuineaFowls

/-- Theorem stating that given the initial bird counts and daily losses, 379 birds remain after a week. -/
theorem birds_remaining_after_week :
  birdsRemaining 300 200 80 20 8 5 15 5 3 = 379 := by sorry

end NUMINAMATH_CALUDE_birds_remaining_after_week_l3919_391968


namespace NUMINAMATH_CALUDE_final_sum_after_transformations_l3919_391922

/-- Given two numbers x and y with sum T, prove that after transformations, 
    the sum of the resulting numbers is 4T + 22 -/
theorem final_sum_after_transformations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 4) + 2 * (y + 5) = 4 * T + 22 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformations_l3919_391922
