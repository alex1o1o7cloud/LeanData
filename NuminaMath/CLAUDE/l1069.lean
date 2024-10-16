import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l1069_106941

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ a * x^2 - 1 ≥ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1069_106941


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1069_106939

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 4 * I
  let z₂ : ℂ := 2 - 4 * I
  z₁ / z₂ - z₂ / z₁ = -8/5 + 16/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1069_106939


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1069_106950

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the condition c * sin(A) = √3 * a * cos(C), prove that C = π/3. -/
theorem triangle_angle_proof (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_condition : c * Real.sin A = Real.sqrt 3 * a * Real.cos C) : 
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1069_106950


namespace NUMINAMATH_CALUDE_books_remaining_l1069_106948

theorem books_remaining (initial_books : ℕ) (donating_people : ℕ) (books_per_donation : ℕ) (borrowed_books : ℕ) :
  initial_books = 500 →
  donating_people = 10 →
  books_per_donation = 8 →
  borrowed_books = 220 →
  initial_books + donating_people * books_per_donation - borrowed_books = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_books_remaining_l1069_106948


namespace NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l1069_106972

/-- The number of ways to choose 2 items from 10 --/
def choose_2_from_10 : ℕ := 45

/-- The number of rectangles on a 10x10 chessboard --/
def num_rectangles : ℕ := choose_2_from_10 * choose_2_from_10

/-- The sum of squares from 1^2 to 10^2 --/
def sum_squares : ℕ := (10 * 11 * 21) / 6

/-- The number of squares on a 10x10 chessboard --/
def num_squares : ℕ := sum_squares

/-- The ratio of squares to rectangles on a 10x10 chessboard is 7/37 --/
theorem squares_to_rectangles_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = 7 / 37 := by sorry

end NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l1069_106972


namespace NUMINAMATH_CALUDE_profit_increase_l1069_106949

theorem profit_increase (cost_price selling_price : ℝ) (a : ℝ) 
  (h1 : selling_price - cost_price = cost_price * (a / 100))
  (h2 : selling_price - (cost_price * 0.95) = (cost_price * 0.95) * ((a + 15) / 100)) :
  a = 185 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l1069_106949


namespace NUMINAMATH_CALUDE_superhero_advantage_l1069_106902

/-- Superhero's speed in miles per minute -/
def superhero_speed : ℚ := 10 / 4

/-- Supervillain's speed in miles per hour -/
def supervillain_speed : ℚ := 100

/-- Minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem superhero_advantage : 
  (superhero_speed * minutes_per_hour) - supervillain_speed = 50 := by sorry

end NUMINAMATH_CALUDE_superhero_advantage_l1069_106902


namespace NUMINAMATH_CALUDE_mike_initial_cards_l1069_106979

/-- The number of baseball cards Mike has initially -/
def initial_cards : ℕ := sorry

/-- The number of baseball cards Sam gave to Mike -/
def cards_from_sam : ℕ := 13

/-- The total number of baseball cards Mike has after receiving cards from Sam -/
def total_cards : ℕ := 100

/-- Theorem stating that Mike initially had 87 baseball cards -/
theorem mike_initial_cards : initial_cards = 87 := by
  sorry

end NUMINAMATH_CALUDE_mike_initial_cards_l1069_106979


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1069_106903

theorem unique_solution_factorial_equation : 
  ∃! (n : ℕ), (Nat.factorial (n + 2) - Nat.factorial (n + 1) - Nat.factorial n = n^2 + n^4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1069_106903


namespace NUMINAMATH_CALUDE_womens_average_age_l1069_106971

theorem womens_average_age (n : ℕ) (A : ℝ) (age_increase : ℝ) (man1_age man2_age : ℕ) :
  n = 8 ∧ age_increase = 2 ∧ man1_age = 20 ∧ man2_age = 28 →
  ∃ W1 W2 : ℝ,
    W1 + W2 = n * (A + age_increase) - (n * A - man1_age - man2_age) ∧
    (W1 + W2) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_womens_average_age_l1069_106971


namespace NUMINAMATH_CALUDE_polynomial_division_l1069_106913

-- Define the dividend polynomial
def dividend (z : ℚ) : ℚ := 4*z^5 - 9*z^4 + 7*z^3 - 12*z^2 + 8*z - 3

-- Define the divisor polynomial
def divisor (z : ℚ) : ℚ := 2*z + 3

-- Define the quotient polynomial
def quotient (z : ℚ) : ℚ := 2*z^4 - 5*z^3 + 4*z^2 - (5/2)*z + 3/4

-- State the theorem
theorem polynomial_division :
  ∀ z : ℚ, dividend z / divisor z = quotient z :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l1069_106913


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l1069_106995

theorem lcm_gcf_ratio : (Nat.lcm 240 540) / (Nat.gcd 240 540) = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l1069_106995


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1069_106998

/-- Proves that the speed of a boat in still water is 13 km/hr given the conditions -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 68)
  (h3 : downstream_time = 4)
  : ∃ (boat_speed : ℝ), boat_speed = 13 ∧ 
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1069_106998


namespace NUMINAMATH_CALUDE_this_year_sales_calculation_l1069_106989

def last_year_sales : ℝ := 320
def percent_increase : ℝ := 0.25

theorem this_year_sales_calculation :
  last_year_sales * (1 + percent_increase) = 400 := by
  sorry

end NUMINAMATH_CALUDE_this_year_sales_calculation_l1069_106989


namespace NUMINAMATH_CALUDE_triangle_area_tripled_sides_l1069_106999

/-- Given a triangle, prove that tripling its sides multiplies its area by 9 -/
theorem triangle_area_tripled_sides (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let s' := ((3 * a) + (3 * b) + (3 * c)) / 2
  let area' := Real.sqrt (s' * (s' - 3 * a) * (s' - 3 * b) * (s' - 3 * c))
  area' = 9 * area := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_tripled_sides_l1069_106999


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l1069_106964

/-- Given a parallelogram with area 128 sq m and base 8 m, prove the ratio of altitude to base is 2 -/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
  area = 128 ∧ base = 8 ∧ area = base * altitude →
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l1069_106964


namespace NUMINAMATH_CALUDE_complex_polynomial_equality_l1069_106951

theorem complex_polynomial_equality (z : ℂ) (h : z = 2 - I) :
  z^6 - 3*z^5 + z^4 + 5*z^3 + 2 = (z^2 - 4*z + 5)*(z^4 + z^3) + 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_polynomial_equality_l1069_106951


namespace NUMINAMATH_CALUDE_square_equality_l1069_106926

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l1069_106926


namespace NUMINAMATH_CALUDE_sin_a_n_bound_l1069_106945

theorem sin_a_n_bound (a : ℕ → ℝ) :
  (a 1 = π / 3) →
  (∀ n, 0 < a n ∧ a n < π / 3) →
  (∀ n ≥ 2, Real.sin (a (n + 1)) ≤ (1 / 3) * Real.sin (3 * a n)) →
  ∀ n, Real.sin (a n) < 1 / Real.sqrt n :=
by sorry

end NUMINAMATH_CALUDE_sin_a_n_bound_l1069_106945


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1069_106923

/-- Given vectors a and b, where b and b-a are collinear, prove |a+b| = 3√5/2 -/
theorem vector_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∃ (k : ℝ), b = k • (b - a)) →
  ‖a + b‖ = 3 * Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1069_106923


namespace NUMINAMATH_CALUDE_grandmother_pill_duration_l1069_106937

/-- Calculates the duration in months for a given pill supply -/
def pillDuration (pillSupply : ℕ) (pillFraction : ℚ) (daysPerDose : ℕ) (daysPerMonth : ℕ) : ℚ :=
  (pillSupply : ℚ) * daysPerDose / pillFraction / daysPerMonth

theorem grandmother_pill_duration :
  pillDuration 60 (1/3) 3 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_pill_duration_l1069_106937


namespace NUMINAMATH_CALUDE_perpendicular_planes_not_necessarily_parallel_l1069_106966

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- Two planes are perpendicular -/
def perpendicular (p1 p2 : Plane3D) : Prop := sorry

/-- Two planes are parallel -/
def parallel (p1 p2 : Plane3D) : Prop := sorry

/-- The statement that if two planes are perpendicular to a third plane, they are parallel to each other is false -/
theorem perpendicular_planes_not_necessarily_parallel (α β γ : Plane3D) :
  ¬(∀ α β γ : Plane3D, perpendicular α β → perpendicular β γ → parallel α γ) := by
  sorry

#check perpendicular_planes_not_necessarily_parallel

end NUMINAMATH_CALUDE_perpendicular_planes_not_necessarily_parallel_l1069_106966


namespace NUMINAMATH_CALUDE_max_value_of_product_l1069_106928

/-- Given real numbers x and y that satisfy x + y = 1, 
    the maximum value of (x^3 + 1)(y^3 + 1) is 4. -/
theorem max_value_of_product (x y : ℝ) (h : x + y = 1) :
  ∃ M : ℝ, M = 4 ∧ ∀ x y : ℝ, x + y = 1 → (x^3 + 1) * (y^3 + 1) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_product_l1069_106928


namespace NUMINAMATH_CALUDE_count_nonadjacent_permutations_l1069_106962

/-- The number of permutations of n distinct elements where two specific elements are not adjacent -/
def nonadjacent_permutations (n : ℕ) : ℕ :=
  (n - 2) * Nat.factorial (n - 1)

/-- Theorem stating that the number of permutations of n distinct elements 
    where two specific elements are not adjacent is (n-2)(n-1)! -/
theorem count_nonadjacent_permutations (n : ℕ) (h : n ≥ 2) :
  nonadjacent_permutations n = (n - 2) * Nat.factorial (n - 1) := by
  sorry

#check count_nonadjacent_permutations

end NUMINAMATH_CALUDE_count_nonadjacent_permutations_l1069_106962


namespace NUMINAMATH_CALUDE_spaghetti_tortellini_ratio_l1069_106970

theorem spaghetti_tortellini_ratio : 
  ∀ (total_students : ℕ) 
    (spaghetti_students tortellini_students : ℕ) 
    (grade_levels : ℕ),
  total_students = 800 →
  spaghetti_students = 300 →
  tortellini_students = 120 →
  grade_levels = 4 →
  (spaghetti_students / grade_levels) / (tortellini_students / grade_levels) = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_spaghetti_tortellini_ratio_l1069_106970


namespace NUMINAMATH_CALUDE_percent_of_percent_l1069_106956

theorem percent_of_percent (y : ℝ) : 0.21 * y = 0.3 * (0.7 * y) := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1069_106956


namespace NUMINAMATH_CALUDE_min_operations_for_2006_l1069_106912

/-- The minimal number of operations needed to calculate x^2006 -/
def min_operations : ℕ := 17

/-- A function that represents the number of operations needed to calculate x^n given x -/
noncomputable def operations (n : ℕ) : ℕ := sorry

/-- The theorem stating that the minimal number of operations to calculate x^2006 is 17 -/
theorem min_operations_for_2006 : operations 2006 = min_operations := by sorry

end NUMINAMATH_CALUDE_min_operations_for_2006_l1069_106912


namespace NUMINAMATH_CALUDE_adults_in_sleeper_class_l1069_106922

def total_passengers : ℕ := 320
def adult_percentage : ℚ := 75 / 100
def sleeper_adult_percentage : ℚ := 15 / 100

theorem adults_in_sleeper_class : 
  ⌊(total_passengers : ℚ) * adult_percentage * sleeper_adult_percentage⌋ = 36 := by
  sorry

end NUMINAMATH_CALUDE_adults_in_sleeper_class_l1069_106922


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l1069_106957

theorem shaded_area_fraction (x : ℝ) (h_x : x > 0) : 
  let y := (5 * Real.sqrt 2 / 2) * x
  let z := Real.sqrt 2 * x
  let large_square_side := 2 * y + z
  let small_square_area := x^2
  let shaded_area := 24 * small_square_area
  let large_square_area := large_square_side^2
  shaded_area / large_square_area = 1/3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l1069_106957


namespace NUMINAMATH_CALUDE_triangle_properties_l1069_106980

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (1/2 : ℝ) * Real.cos (2 * A) = Real.cos A ^ 2 - Real.cos A →
  a = 3 →
  Real.sin B = 2 * Real.sin C →
  A = π / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1069_106980


namespace NUMINAMATH_CALUDE_problem_solution_l1069_106992

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 10}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

theorem problem_solution (a : ℝ) (h : a > 0) :
  ((A ∩ B a = ∅) → a ≥ 9) ∧
  ((∀ x, (¬p x → q a x) ∧ (∃ y, q a y ∧ p y)) → (a ≤ 3)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1069_106992


namespace NUMINAMATH_CALUDE_negation_existence_statement_l1069_106934

theorem negation_existence_statement :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 > 0) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_statement_l1069_106934


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1069_106978

theorem initial_money_calculation (initial_money : ℚ) : 
  (initial_money * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 400) → 
  initial_money = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1069_106978


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l1069_106910

theorem contrapositive_real_roots (m : ℝ) :
  (¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ↔
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l1069_106910


namespace NUMINAMATH_CALUDE_mat_equation_solution_l1069_106904

theorem mat_equation_solution :
  ∃! x : ℝ, (589 + x) + (544 - x) + 80 * x = 2013 := by
  sorry

end NUMINAMATH_CALUDE_mat_equation_solution_l1069_106904


namespace NUMINAMATH_CALUDE_multiply_72519_9999_l1069_106933

theorem multiply_72519_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_9999_l1069_106933


namespace NUMINAMATH_CALUDE_tan_22_5_deg_sum_l1069_106965

theorem tan_22_5_deg_sum (a b c d : ℕ+) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - (d : ℝ)) :
  a + b + c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_sum_l1069_106965


namespace NUMINAMATH_CALUDE_vector_operation_l1069_106994

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, -1)

theorem vector_operation :
  (1/3 : ℝ) • a - (4/3 : ℝ) • b = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l1069_106994


namespace NUMINAMATH_CALUDE_bus_stop_time_l1069_106991

/-- The time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 54)
  (h2 : speed_with_stops = 45) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l1069_106991


namespace NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l1069_106919

theorem cubic_root_sum_inverse_squares (a b c : ℝ) : 
  a^3 - 8*a^2 + 6*a - 3 = 0 →
  b^3 - 8*b^2 + 6*b - 3 = 0 →
  c^3 - 8*c^2 + 6*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l1069_106919


namespace NUMINAMATH_CALUDE_buoy_distance_is_24_l1069_106924

/-- The distance between two consecutive buoys in the ocean -/
def buoy_distance (d1 d2 : ℝ) : ℝ := d2 - d1

/-- Theorem: The distance between two consecutive buoys is 24 meters -/
theorem buoy_distance_is_24 :
  let d1 := 72 -- distance of first buoy from beach
  let d2 := 96 -- distance of second buoy from beach
  buoy_distance d1 d2 = 24 := by sorry

end NUMINAMATH_CALUDE_buoy_distance_is_24_l1069_106924


namespace NUMINAMATH_CALUDE_min_cards_l1069_106946

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem min_cards : ∃ (n a b c d e : ℕ),
  n = 63 ∧
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧
  (n - a) % 5 = 0 ∧
  (n - a - b) % 3 = 0 ∧
  (n - a - b - c) % 2 = 0 ∧
  n - a - b - c - d = e ∧
  ∀ (m : ℕ), m < n →
    ¬(∃ (a' b' c' d' e' : ℕ),
      is_prime a' ∧ is_prime b' ∧ is_prime c' ∧ is_prime d' ∧ is_prime e' ∧
      a' > b' ∧ b' > c' ∧ c' > d' ∧ d' > e' ∧
      (m - a') % 5 = 0 ∧
      (m - a' - b') % 3 = 0 ∧
      (m - a' - b' - c') % 2 = 0 ∧
      m - a' - b' - c' - d' = e') :=
by sorry

end NUMINAMATH_CALUDE_min_cards_l1069_106946


namespace NUMINAMATH_CALUDE_min_value_a2_plus_b2_l1069_106935

theorem min_value_a2_plus_b2 (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) :
  a^2 + b^2 ≥ (Real.sqrt 5 + 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a2_plus_b2_l1069_106935


namespace NUMINAMATH_CALUDE_smaller_cuboid_height_l1069_106961

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- The original large cuboid -/
def original : Cuboid := { length := 18, width := 15, height := 2 }

/-- The smaller cuboid with unknown height -/
def smaller (h : ℝ) : Cuboid := { length := 5, width := 6, height := h }

/-- The number of smaller cuboids that can be formed -/
def num_smaller : ℕ := 6

/-- Theorem: The height of each smaller cuboid is 3 meters -/
theorem smaller_cuboid_height :
  ∃ h : ℝ, volume original = num_smaller * volume (smaller h) ∧ h = 3 := by
  sorry


end NUMINAMATH_CALUDE_smaller_cuboid_height_l1069_106961


namespace NUMINAMATH_CALUDE_boxes_with_neither_l1069_106947

theorem boxes_with_neither (total_boxes : ℕ) (marker_boxes : ℕ) (crayon_boxes : ℕ) (both_boxes : ℕ) 
  (h1 : total_boxes = 15)
  (h2 : marker_boxes = 9)
  (h3 : crayon_boxes = 5)
  (h4 : both_boxes = 4) :
  total_boxes - (marker_boxes + crayon_boxes - both_boxes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l1069_106947


namespace NUMINAMATH_CALUDE_valid_lineups_count_l1069_106920

/-- The number of players in the team -/
def total_players : ℕ := 15

/-- The number of players in a starting lineup -/
def lineup_size : ℕ := 6

/-- The number of players who refuse to play together -/
def refusing_players : ℕ := 3

/-- Calculates the number of valid lineups -/
def valid_lineups : ℕ := 
  Nat.choose total_players lineup_size - Nat.choose (total_players - refusing_players) (lineup_size - refusing_players)

theorem valid_lineups_count : valid_lineups = 4785 := by sorry

end NUMINAMATH_CALUDE_valid_lineups_count_l1069_106920


namespace NUMINAMATH_CALUDE_eliminate_denominators_l1069_106900

theorem eliminate_denominators (x : ℝ) :
  (x - 1) / 3 = 4 - (2 * x + 1) / 2 ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l1069_106900


namespace NUMINAMATH_CALUDE_valentines_day_theorem_l1069_106944

/-- The number of valentines given on Valentine's Day -/
def valentines_given (male_students female_students : ℕ) : ℕ :=
  male_students * female_students

/-- The total number of students -/
def total_students (male_students female_students : ℕ) : ℕ :=
  male_students + female_students

/-- Theorem stating the number of valentines given -/
theorem valentines_day_theorem (male_students female_students : ℕ) :
  valentines_given male_students female_students = 
  total_students male_students female_students + 22 →
  valentines_given male_students female_students = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_valentines_day_theorem_l1069_106944


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1069_106981

/-- Given a hyperbola with equation x²/4 - y²/m² = 1 where m > 0,
    and one of its asymptotes is 5x - 2y = 0, prove that m = 5. -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) : 
  (∃ x y : ℝ, x^2/4 - y^2/m^2 = 1 ∧ 5*x - 2*y = 0) → m = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1069_106981


namespace NUMINAMATH_CALUDE_sherry_catch_train_prob_l1069_106993

def train_arrival_prob : ℝ := 0.75
def notice_train_prob : ℝ := 0.25
def time_frame : ℕ := 5

theorem sherry_catch_train_prob :
  1 - (1 - train_arrival_prob * notice_train_prob)^time_frame =
  1 - (0.8125 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_sherry_catch_train_prob_l1069_106993


namespace NUMINAMATH_CALUDE_susan_remaining_money_l1069_106905

def susan_fair_spending (initial_amount food_cost : ℕ) : ℕ :=
  let game_cost := 3 * food_cost
  let total_spent := food_cost + game_cost
  initial_amount - total_spent

theorem susan_remaining_money :
  susan_fair_spending 90 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_remaining_money_l1069_106905


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1069_106925

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 4 / 5 →  -- angles are in ratio 4:5
  a = 80 :=  -- smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1069_106925


namespace NUMINAMATH_CALUDE_johnson_martinez_tie_l1069_106997

/-- Represents the months of the baseball season --/
inductive Month
| Mar
| Apr
| May
| Jun
| Jul
| Aug
| Sep

/-- Calculates the cumulative home runs for a player --/
def cumulativeHomeRuns (monthlyData : List Nat) : List Nat :=
  List.scanl (· + ·) 0 monthlyData

/-- Checks if two lists are equal up to a certain index --/
def equalUpTo (l1 l2 : List Nat) (index : Nat) : Bool :=
  (l1.take index) = (l2.take index)

/-- Finds the first index where two lists become equal --/
def firstEqualIndex (l1 l2 : List Nat) : Option Nat :=
  (List.range l1.length).find? (fun i => l1[i]! = l2[i]!)

theorem johnson_martinez_tie (johnsonData martinezData : List Nat) 
    (h1 : johnsonData = [3, 8, 15, 12, 5, 7, 14])
    (h2 : martinezData = [0, 3, 9, 20, 7, 12, 13]) : 
    firstEqualIndex 
      (cumulativeHomeRuns johnsonData) 
      (cumulativeHomeRuns martinezData) = some 6 := by
  sorry

#check johnson_martinez_tie

end NUMINAMATH_CALUDE_johnson_martinez_tie_l1069_106997


namespace NUMINAMATH_CALUDE_expression_evaluation_l1069_106955

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1069_106955


namespace NUMINAMATH_CALUDE_smallest_top_block_exists_l1069_106973

/-- Represents a block in the pyramid --/
structure Block where
  layer : Nat
  value : Nat

/-- Represents the pyramid structure --/
structure Pyramid where
  blocks : List Block
  layer1 : List Nat
  layer2 : List Nat
  layer3 : List Nat
  layer4 : Nat

/-- Check if a pyramid configuration is valid --/
def isValidPyramid (p : Pyramid) : Prop :=
  p.blocks.length = 54 ∧
  p.layer1.length = 30 ∧
  p.layer2.length = 15 ∧
  p.layer3.length = 8 ∧
  ∀ n ∈ p.layer1, 1 ≤ n ∧ n ≤ 30

/-- Calculate the value of a block in an upper layer --/
def calculateBlockValue (below : List Nat) : Nat :=
  below.sum

/-- The main theorem --/
theorem smallest_top_block_exists (p : Pyramid) :
  isValidPyramid p →
  ∃ (minTop : Nat), 
    p.layer4 = minTop ∧
    ∀ (p' : Pyramid), isValidPyramid p' → p'.layer4 ≥ minTop := by
  sorry


end NUMINAMATH_CALUDE_smallest_top_block_exists_l1069_106973


namespace NUMINAMATH_CALUDE_number_sum_proof_l1069_106915

theorem number_sum_proof : ∃ x : ℤ, x + 15 = 96 ∧ x = 81 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_proof_l1069_106915


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1069_106996

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, 2 * a * x^2 - 2 * x + 3 < 0 ↔ 2 < x ∧ x < b) →
  (∀ x, 3 * x^2 + 2 * x + 2 * a < 0 ↔ -1/2 < x ∧ x < -1/6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1069_106996


namespace NUMINAMATH_CALUDE_perpendicular_and_minimum_points_l1069_106974

-- Define the vectors
def OA : Fin 2 → ℝ := ![1, 7]
def OB : Fin 2 → ℝ := ![5, 1]
def OP : Fin 2 → ℝ := ![2, 1]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := v 0 * w 0 + v 1 * w 1

-- Define the function for OQ based on parameter t
def OQ (t : ℝ) : Fin 2 → ℝ := ![2*t, t]

-- Define QA as a function of t
def QA (t : ℝ) : Fin 2 → ℝ := ![1 - 2*t, 7 - t]

-- Define QB as a function of t
def QB (t : ℝ) : Fin 2 → ℝ := ![5 - 2*t, 1 - t]

theorem perpendicular_and_minimum_points :
  (∃ t : ℝ, dot_product (QA t) OP = 0 ∧ OQ t = ![18/5, 9/5]) ∧
  (∃ t : ℝ, ∀ s : ℝ, dot_product OA (QB t) ≤ dot_product OA (QB s) ∧ OQ t = ![4, 2]) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_and_minimum_points_l1069_106974


namespace NUMINAMATH_CALUDE_square_difference_ratio_l1069_106909

theorem square_difference_ratio : 
  (1632^2 - 1629^2) / (1635^2 - 1626^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l1069_106909


namespace NUMINAMATH_CALUDE_money_left_after_transactions_l1069_106917

def initial_money : ℕ := 50 * 10 + 24 * 25 + 40 * 5 + 75

def candy_cost : ℕ := 6 * 85
def lollipop_cost : ℕ := 3 * 50
def chips_cost : ℕ := 4 * 95
def soda_cost : ℕ := 2 * 125

def total_cost : ℕ := candy_cost + lollipop_cost + chips_cost + soda_cost

theorem money_left_after_transactions : 
  initial_money - total_cost = 85 := by
sorry

end NUMINAMATH_CALUDE_money_left_after_transactions_l1069_106917


namespace NUMINAMATH_CALUDE_parabola_coefficients_l1069_106984

/-- A parabola with vertex (4, 3), vertical axis of symmetry, passing through (2, 1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a * 4^2 + b * 4 + c = 3
  symmetry : b = -2 * a * 4
  point : a * 2^2 + b * 2 + c = 1

/-- The coefficients of the parabola are (-1/2, 4, -5) -/
theorem parabola_coefficients (p : Parabola) : p.a = -1/2 ∧ p.b = 4 ∧ p.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l1069_106984


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1069_106969

theorem inequality_system_integer_solutions (x : ℤ) : 
  (2 * (1 - x) ≤ 4 ∧ x - 4 < (x - 8) / 3) ↔ (x = -1 ∨ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1069_106969


namespace NUMINAMATH_CALUDE_jason_initial_cards_l1069_106931

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l1069_106931


namespace NUMINAMATH_CALUDE_sum_and_product_of_radical_conjugates_l1069_106959

theorem sum_and_product_of_radical_conjugates (a b : ℝ) : 
  ((a + Real.sqrt b) + (a - Real.sqrt b) = -6) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 9) →
  (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_of_radical_conjugates_l1069_106959


namespace NUMINAMATH_CALUDE_warriors_truth_count_l1069_106952

theorem warriors_truth_count :
  ∀ (total_warriors : ℕ) 
    (sword_yes spear_yes axe_yes bow_yes : ℕ),
  total_warriors = 33 →
  sword_yes = 13 →
  spear_yes = 15 →
  axe_yes = 20 →
  bow_yes = 27 →
  ∃ (truth_tellers : ℕ),
    truth_tellers = 12 ∧
    truth_tellers + (total_warriors - truth_tellers) * 3 = 
      sword_yes + spear_yes + axe_yes + bow_yes :=
by sorry

end NUMINAMATH_CALUDE_warriors_truth_count_l1069_106952


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1069_106958

theorem inequality_solution_set (m : ℝ) :
  {x : ℝ | x^2 - (2*m - 1)*x + m^2 - m > 0} = {x : ℝ | x < m - 1 ∨ x > m} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1069_106958


namespace NUMINAMATH_CALUDE_marias_green_beans_l1069_106988

/-- Given Maria's vegetable cutting preferences and the number of potatoes,
    calculate the number of green beans she needs to cut. -/
theorem marias_green_beans (potatoes : ℕ) : potatoes = 2 → 8 = (potatoes * 6 * 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_marias_green_beans_l1069_106988


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_forty_l1069_106986

/-- If x is 20 percent greater than 40, then x equals 48. -/
theorem twenty_percent_greater_than_forty (x : ℝ) : x = 40 * (1 + 0.2) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_forty_l1069_106986


namespace NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l1069_106936

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- Theorem statement
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-1, -1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l1069_106936


namespace NUMINAMATH_CALUDE_largest_after_removal_l1069_106938

/-- Represents the original sequence of digits --/
def original_sequence : List Nat := sorry

/-- Represents the sequence after removing 100 digits --/
def removed_sequence : List Nat := sorry

/-- The number of digits to be removed --/
def digits_to_remove : Nat := 100

/-- Function to convert a list of digits to a natural number --/
def list_to_number (l : List Nat) : Nat := sorry

/-- Function to check if a list of digits is a valid removal from the original sequence --/
def is_valid_removal (l : List Nat) : Prop := sorry

/-- Theorem stating that the removed_sequence is the largest possible after removing 100 digits --/
theorem largest_after_removal :
  (list_to_number removed_sequence = list_to_number original_sequence - digits_to_remove) ∧
  is_valid_removal removed_sequence ∧
  ∀ (other_sequence : List Nat),
    is_valid_removal other_sequence →
    list_to_number other_sequence ≤ list_to_number removed_sequence :=
sorry

end NUMINAMATH_CALUDE_largest_after_removal_l1069_106938


namespace NUMINAMATH_CALUDE_symmetry_of_f_l1069_106927

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
axiom functional_equation : ∀ x : ℝ, f (x + 5) = f (9 - x)

-- State the theorem to be proved
theorem symmetry_of_f : 
  (∀ x : ℝ, f (7 + x) = f (7 - x)) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_f_l1069_106927


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1069_106960

theorem roots_of_polynomial : ∃ (a b c d : ℂ),
  (a = 2 ∧ b = -2 ∧ c = 2*I ∧ d = -2*I) ∧
  (∀ x : ℂ, x^4 + 4*x^3 - 2*x^2 - 20*x + 24 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1069_106960


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l1069_106921

theorem sequence_gcd_property :
  (¬∃(a : ℕ → ℕ), ∀i j, i < j → Nat.gcd (a i + j) (a j + i) = 1) ∧
  (∀p, Prime p ∧ Odd p → ∃(a : ℕ → ℕ), ∀i j, i < j → ¬(p ∣ Nat.gcd (a i + j) (a j + i))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l1069_106921


namespace NUMINAMATH_CALUDE_missing_fraction_problem_l1069_106940

theorem missing_fraction_problem (sum : ℚ) (f1 f2 f3 f4 f5 f6 f7 : ℚ) : 
  sum = 45/100 →
  f1 = 1/3 →
  f2 = 1/2 →
  f3 = -5/6 →
  f4 = 1/4 →
  f5 = -9/20 →
  f6 = -9/20 →
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = sum →
  f7 = 11/10 := by
sorry

end NUMINAMATH_CALUDE_missing_fraction_problem_l1069_106940


namespace NUMINAMATH_CALUDE_regression_line_equation_l1069_106906

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation of the form y = mx + b -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Given a regression line with slope 1.2 passing through (4,5), prove its equation is ŷ = 1.2x + 0.2 -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point)
  (h1 : slope = 1.2)
  (h2 : center = ⟨4, 5⟩)
  : ∃ (eq : LinearEquation), 
    eq.slope = slope ∧ 
    eq.intercept = 0.2 ∧ 
    center.y = eq.slope * center.x + eq.intercept := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l1069_106906


namespace NUMINAMATH_CALUDE_function_properties_l1069_106942

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Statement of the theorem
theorem function_properties :
  -- 1. The tangent line to y = f(x) at x = 1 is y = x - 1
  (∀ x, (f x - f 1) = (x - 1) * (Real.log 1 + 1)) ∧
  -- 2. There are exactly 2 lines tangent to y = f(x) passing through (1, -1)
  (∃! a b : ℝ, a ≠ b ∧ 
    (∀ x, f x = (Real.log a + 1) * (x - a) + f a) ∧
    (Real.log a + 1) * (1 - a) + f a = -1 ∧
    (∀ x, f x = (Real.log b + 1) * (x - b) + f b) ∧
    (Real.log b + 1) * (1 - b) + f b = -1) ∧
  -- 3. f(x) has a local minimum and no local maximum
  (∃ c : ℝ, ∀ x, x > 0 → x ≠ c → f x > f c) ∧
  (¬ ∃ d : ℝ, ∀ x, x > 0 → x ≠ d → f x < f d) ∧
  -- 4. The equation f(x) = 1 does not have two distinct solutions
  ¬ (∃ x y : ℝ, x ≠ y ∧ f x = 1 ∧ f y = 1) :=
by sorry


end NUMINAMATH_CALUDE_function_properties_l1069_106942


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1069_106918

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.sin (-20/3 * Real.pi)) / Real.tan (11/3 * Real.pi) - 
  Real.cos (13/4 * Real.pi) * Real.tan (-35/4 * Real.pi) = 
  (Real.sqrt 2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1069_106918


namespace NUMINAMATH_CALUDE_total_cost_is_3200_cents_l1069_106982

/-- Represents the number of shirt boxes that can be wrapped with one roll of paper -/
def shirt_boxes_per_roll : ℕ := 5

/-- Represents the number of XL boxes that can be wrapped with one roll of paper -/
def xl_boxes_per_roll : ℕ := 3

/-- Represents the number of shirt boxes Harold needs to wrap -/
def total_shirt_boxes : ℕ := 20

/-- Represents the number of XL boxes Harold needs to wrap -/
def total_xl_boxes : ℕ := 12

/-- Represents the cost of one roll of wrapping paper in cents -/
def cost_per_roll : ℕ := 400

/-- Theorem stating that the total cost for Harold to wrap all boxes is $32.00 -/
theorem total_cost_is_3200_cents : 
  (((total_shirt_boxes + shirt_boxes_per_roll - 1) / shirt_boxes_per_roll) + 
   ((total_xl_boxes + xl_boxes_per_roll - 1) / xl_boxes_per_roll)) * 
  cost_per_roll = 3200 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_3200_cents_l1069_106982


namespace NUMINAMATH_CALUDE_problem_I_problem_II_l1069_106914

theorem problem_I (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin (π - α) - 2 * Real.cos (-α)) / (3 * Real.cos (π/2 - α) - 5 * Real.cos (π + α)) = 5/7 := by
  sorry

theorem problem_II (x : Real) (h1 : Real.sin x + Real.cos x = 1/5) (h2 : 0 < x) (h3 : x < π) :
  Real.sin x = 4/5 ∧ Real.cos x = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_I_problem_II_l1069_106914


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l1069_106983

/-- Represents a participant in the chess tournament -/
structure Participant :=
  (id : Nat)

/-- Represents the results of the chess tournament -/
structure TournamentResult :=
  (participants : Finset Participant)
  (white_wins : Participant → Nat)
  (black_wins : Participant → Nat)

/-- Defines the "no weaker than" relation between two participants -/
def no_weaker_than (result : TournamentResult) (a b : Participant) : Prop :=
  result.white_wins a ≥ result.white_wins b ∧ result.black_wins a ≥ result.black_wins b

theorem chess_tournament_theorem :
  ∀ (result : TournamentResult),
    result.participants.card = 20 →
    (∀ p q : Participant, p ∈ result.participants → q ∈ result.participants → p ≠ q →
      result.white_wins p + result.white_wins q = 1 ∧
      result.black_wins p + result.black_wins q = 1) →
    ∃ a b : Participant, a ∈ result.participants ∧ b ∈ result.participants ∧ a ≠ b ∧
      no_weaker_than result a b := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_theorem_l1069_106983


namespace NUMINAMATH_CALUDE_sum_of_twenty_terms_l1069_106943

/-- Given a sequence of non-zero terms {aₙ}, where Sₙ is the sum of the first n terms,
    prove that S₂₀ = 210 under the given conditions. -/
theorem sum_of_twenty_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n ≠ 0) →
  (∀ n, S n = (a n * a (n + 1)) / 2) →
  a 1 = 1 →
  S 20 = 210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_twenty_terms_l1069_106943


namespace NUMINAMATH_CALUDE_equilateral_triangles_congruence_l1069_106968

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Two triangles are congruent if all their corresponding sides are equal -/
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side = t2.side

theorem equilateral_triangles_congruence (t1 t2 : EquilateralTriangle) :
  congruent t1 t2 ↔ t1.side = t2.side :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangles_congruence_l1069_106968


namespace NUMINAMATH_CALUDE_max_digit_sum_for_reciprocal_decimal_l1069_106932

/-- Given digits a, b, c forming a decimal 0.abc that equals 1/y for some integer y between 1 and 12,
    the sum a + b + c is at most 8. -/
theorem max_digit_sum_for_reciprocal_decimal (a b c y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (0 < y ∧ y ≤ 12) →            -- 0 < y ≤ 12
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →  -- 0.abc = 1/y
  a + b + c ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_for_reciprocal_decimal_l1069_106932


namespace NUMINAMATH_CALUDE_expr_D_not_complete_square_expr_A_is_complete_square_expr_B_is_complete_square_expr_C_is_complete_square_l1069_106911

-- Define the expressions
def expr_A (x : ℝ) := x^2 - 2*x + 1
def expr_B (x : ℝ) := 1 - 2*x + x^2
def expr_C (a b : ℝ) := a^2 + b^2 - 2*a*b
def expr_D (x : ℝ) := 4*x^2 + 4*x - 1

-- Define what it means for an expression to be factored as a complete square
def is_complete_square (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * (x - b)^2

-- Theorem stating that expr_D cannot be factored as a complete square
theorem expr_D_not_complete_square :
  ¬ is_complete_square expr_D :=
sorry

-- Theorems stating that the other expressions can be factored as complete squares
theorem expr_A_is_complete_square :
  is_complete_square expr_A :=
sorry

theorem expr_B_is_complete_square :
  is_complete_square expr_B :=
sorry

theorem expr_C_is_complete_square :
  ∃ (f : ℝ → ℝ → ℝ), ∀ a b, expr_C a b = f a b ∧ is_complete_square (f a) :=
sorry

end NUMINAMATH_CALUDE_expr_D_not_complete_square_expr_A_is_complete_square_expr_B_is_complete_square_expr_C_is_complete_square_l1069_106911


namespace NUMINAMATH_CALUDE_antonella_coins_l1069_106953

theorem antonella_coins (num_coins : ℕ) (loonie_value toonie_value : ℚ) 
  (frappuccino_cost remaining_money : ℚ) :
  num_coins = 10 →
  loonie_value = 1 →
  toonie_value = 2 →
  frappuccino_cost = 3 →
  remaining_money = 11 →
  ∃ (num_loonies num_toonies : ℕ),
    num_loonies + num_toonies = num_coins ∧
    num_loonies * loonie_value + num_toonies * toonie_value = 
      remaining_money + frappuccino_cost ∧
    num_toonies = 4 :=
by sorry

end NUMINAMATH_CALUDE_antonella_coins_l1069_106953


namespace NUMINAMATH_CALUDE_students_without_A_l1069_106901

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (science : ℕ)
  (history_math : ℕ) (history_science : ℕ) (math_science : ℕ) (all_three : ℕ) :
  total = 45 →
  history = 11 →
  math = 16 →
  science = 9 →
  history_math = 5 →
  history_science = 3 →
  math_science = 4 →
  all_three = 2 →
  total - (history + math + science - history_math - history_science - math_science + all_three) = 19 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l1069_106901


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l1069_106967

theorem log_2_base_10_bounds :
  (2^9 = 512) →
  (2^14 = 16384) →
  (10^3 = 1000) →
  (10^4 = 10000) →
  (2/7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l1069_106967


namespace NUMINAMATH_CALUDE_stations_visited_l1069_106916

theorem stations_visited (total_nails : ℕ) (nails_per_station : ℕ) (h1 : total_nails = 140) (h2 : nails_per_station = 7) :
  total_nails / nails_per_station = 20 := by
sorry

end NUMINAMATH_CALUDE_stations_visited_l1069_106916


namespace NUMINAMATH_CALUDE_quadratic_roots_and_graph_point_l1069_106985

theorem quadratic_roots_and_graph_point (a b c : ℝ) (x : ℝ) 
  (h1 : a ≠ 0)
  (h2 : Real.tan x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h3 : Real.tan (π/4 - x) = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  : a * 1^2 + b * 1 - c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_graph_point_l1069_106985


namespace NUMINAMATH_CALUDE_min_filtration_cycles_l1069_106975

theorem min_filtration_cycles (initial_conc : ℝ) (reduction_rate : ℝ) (target_conc : ℝ) : 
  initial_conc = 225 →
  reduction_rate = 1/3 →
  target_conc = 7.5 →
  (∃ n : ℕ, (initial_conc * (1 - reduction_rate)^n ≤ target_conc ∧ 
             ∀ m : ℕ, m < n → initial_conc * (1 - reduction_rate)^m > target_conc)) →
  (∃ n : ℕ, n = 9 ∧ initial_conc * (1 - reduction_rate)^n ≤ target_conc ∧ 
             ∀ m : ℕ, m < n → initial_conc * (1 - reduction_rate)^m > target_conc) :=
by sorry

end NUMINAMATH_CALUDE_min_filtration_cycles_l1069_106975


namespace NUMINAMATH_CALUDE_problem_solution_l1069_106907

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem problem_solution :
  let a₁ := 5
  let d := 3
  let aₙ := 38
  let n := (aₙ - a₁) / d + 1
  let a := arithmetic_sum a₁ d n
  let b := sum_of_digits a
  let c := b ^ 2
  let d := c / 3
  d = 75 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1069_106907


namespace NUMINAMATH_CALUDE_remainder_h_x_10_divided_by_h_x_l1069_106963

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

-- State the theorem
theorem remainder_h_x_10_divided_by_h_x : 
  ∃ (q : ℝ → ℝ), h (x^10) = q x * h x + 7 := by sorry

end NUMINAMATH_CALUDE_remainder_h_x_10_divided_by_h_x_l1069_106963


namespace NUMINAMATH_CALUDE_triangle_problem_l1069_106987

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0)
  (h2 : t.c = 2)
  (h3 : t.a + t.b = t.a * t.b) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1069_106987


namespace NUMINAMATH_CALUDE_first_number_is_30_l1069_106990

def fibonacci_like_sequence (a₁ a₂ : ℤ) : ℕ → ℤ
  | 0 => a₁
  | 1 => a₂
  | (n+2) => fibonacci_like_sequence a₁ a₂ n + fibonacci_like_sequence a₁ a₂ (n+1)

theorem first_number_is_30 (a₁ a₂ : ℤ) :
  fibonacci_like_sequence a₁ a₂ 6 = 5 ∧
  fibonacci_like_sequence a₁ a₂ 7 = 14 ∧
  fibonacci_like_sequence a₁ a₂ 8 = 33 →
  a₁ = 30 := by
sorry

end NUMINAMATH_CALUDE_first_number_is_30_l1069_106990


namespace NUMINAMATH_CALUDE_factorization_equality_l1069_106929

theorem factorization_equality (m a b : ℝ) : 3*m*a^2 - 6*m*a*b + 3*m*b^2 = 3*m*(a-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1069_106929


namespace NUMINAMATH_CALUDE_ln_ratio_monotone_l1069_106954

open Real

theorem ln_ratio_monotone (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1) :
  (log a) / a < (log b) / b ∧ (log b) / b < (log c) / c :=
by sorry

end NUMINAMATH_CALUDE_ln_ratio_monotone_l1069_106954


namespace NUMINAMATH_CALUDE_garden_perimeter_is_800_l1069_106976

/-- The perimeter of a rectangular garden with given length and breadth -/
def garden_perimeter (length breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem: The perimeter of a rectangular garden with length 300 m and breadth 100 m is 800 m -/
theorem garden_perimeter_is_800 :
  garden_perimeter 300 100 = 800 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_800_l1069_106976


namespace NUMINAMATH_CALUDE_sum_equals_80790_l1069_106977

theorem sum_equals_80790 : 30 + 80000 + 700 + 60 = 80790 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_80790_l1069_106977


namespace NUMINAMATH_CALUDE_library_books_count_l1069_106908

/-- The number of bookshelves in the library -/
def num_bookshelves : ℕ := 28

/-- The number of floors in each bookshelf -/
def floors_per_bookshelf : ℕ := 6

/-- The number of books left on a floor after taking two books -/
def books_left_after_taking_two : ℕ := 20

/-- The total number of books in the library -/
def total_books : ℕ := num_bookshelves * floors_per_bookshelf * (books_left_after_taking_two + 2)

theorem library_books_count : total_books = 3696 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1069_106908


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1069_106930

/-- The equation of a line passing through the intersection of two lines and perpendicular to a third line -/
theorem perpendicular_line_equation (a b c d e f g h i j : ℝ) :
  let l₁ : ℝ × ℝ → Prop := λ p => a * p.1 + b * p.2 = 0
  let l₂ : ℝ × ℝ → Prop := λ p => c * p.1 + d * p.2 + e = 0
  let l₃ : ℝ × ℝ → Prop := λ p => f * p.1 + g * p.2 + h = 0
  let l₄ : ℝ × ℝ → Prop := λ p => i * p.1 + j * p.2 + 5 = 0
  (∃! p, l₁ p ∧ l₂ p) →  -- l₁ and l₂ intersect at a unique point
  (∀ p q : ℝ × ℝ, l₃ p ∧ l₃ q → (p.1 - q.1) * (f * (p.1 - q.1) + g * (p.2 - q.2)) + (p.2 - q.2) * (g * (p.1 - q.1) - f * (p.2 - q.2)) = 0) →  -- l₄ is perpendicular to l₃
  (a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = -2 ∧ f = 2 ∧ g = 1 ∧ h = 3 ∧ i = 1 ∧ j = -2) →
  ∀ p, l₁ p ∧ l₂ p → l₄ p  -- The point of intersection of l₁ and l₂ satisfies l₄
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1069_106930
