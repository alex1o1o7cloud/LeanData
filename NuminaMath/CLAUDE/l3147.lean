import Mathlib

namespace NUMINAMATH_CALUDE_min_white_pairs_problem_solution_l3147_314705

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Nat)

/-- Calculates the total number of adjacent cell pairs in a square grid -/
def total_pairs (g : Grid) : Nat :=
  2 * g.size * (g.size - 1)

/-- Calculates the maximum number of central cell pairs -/
def max_central_pairs (g : Grid) : Nat :=
  ((g.size - 2) * (g.size - 2)) / 2

/-- Theorem: Given an 8x8 grid with 20 black cells, the minimum number of pairs of adjacent white cells is 34 -/
theorem min_white_pairs (g : Grid) (h1 : g.size = 8) (h2 : g.black_cells = 20) :
  total_pairs g - (60 + min g.black_cells (max_central_pairs g)) = 34 := by
  sorry

/-- Main theorem stating the result for the specific problem -/
theorem problem_solution : 
  ∃ (g : Grid), g.size = 8 ∧ g.black_cells = 20 ∧ 
  (total_pairs g - (60 + min g.black_cells (max_central_pairs g)) = 34) := by
  sorry

end NUMINAMATH_CALUDE_min_white_pairs_problem_solution_l3147_314705


namespace NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l3147_314796

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus_of_parabola (x y : ℝ) :
  x^2 = 2*y →  -- Parabola equation
  y = 3 →      -- Ordinate of point P
  (y + 1/4) = 7/2  -- Distance to focus
  := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l3147_314796


namespace NUMINAMATH_CALUDE_max_intersections_l3147_314759

/-- Given 15 points on the positive x-axis and 10 points on the positive y-axis,
    with segments connecting each point on the x-axis to each point on the y-axis,
    the maximum number of intersection points in the interior of the first quadrant is 4725. -/
theorem max_intersections (x_points y_points : ℕ) (h1 : x_points = 15) (h2 : y_points = 10) :
  (x_points.choose 2) * (y_points.choose 2) = 4725 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_l3147_314759


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3147_314750

def U : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Finset ℕ := {0, 1, 3, 5, 8}
def B : Finset ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7, 9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3147_314750


namespace NUMINAMATH_CALUDE_hexagon_segment_length_l3147_314740

/-- A regular hexagon with side length 2 inscribed in a circle -/
structure RegularHexagon :=
  (side_length : ℝ)
  (inscribed_in_circle : Bool)
  (h_side_length : side_length = 2)
  (h_inscribed : inscribed_in_circle = true)

/-- A segment connecting a vertex to the midpoint of the opposite side -/
def opposite_midpoint_segment (h : RegularHexagon) : ℝ → ℝ := sorry

/-- The total length of all segments connecting vertices to opposite midpoints -/
def total_segment_length (h : RegularHexagon) : ℝ :=
  6 * opposite_midpoint_segment h 1

theorem hexagon_segment_length (h : RegularHexagon) :
  total_segment_length h = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_segment_length_l3147_314740


namespace NUMINAMATH_CALUDE_system_solution_l3147_314720

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 7 * y = -14) ∧ 
    (5 * x + 3 * y = -13) ∧ 
    (x = -133 / 47) ∧ 
    (y = 18 / 47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3147_314720


namespace NUMINAMATH_CALUDE_k_value_l3147_314749

theorem k_value (a b c k : ℝ) 
  (h1 : 2 * a / (b + c) = k) 
  (h2 : 2 * b / (a + c) = k) 
  (h3 : 2 * c / (a + b) = k) : 
  k = 1 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3147_314749


namespace NUMINAMATH_CALUDE_circle_equation_theta_range_l3147_314709

theorem circle_equation_theta_range :
  ∀ (x y θ : ℝ),
  (x^2 + y^2 + x + Real.sqrt 3 * y + Real.tan θ = 0) →
  (-π/2 < θ ∧ θ < π/2) →
  (∃ (c : ℝ × ℝ) (r : ℝ), ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ↔ 
    p.1^2 + p.2^2 + p.1 + Real.sqrt 3 * p.2 + Real.tan θ = 0) →
  -π/2 < θ ∧ θ < π/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_theta_range_l3147_314709


namespace NUMINAMATH_CALUDE_sheets_in_box_l3147_314733

/-- The number of sheets needed per printer -/
def sheets_per_printer : ℕ := 7

/-- The number of printers that can be filled -/
def num_printers : ℕ := 31

/-- The total number of sheets in the box -/
def total_sheets : ℕ := sheets_per_printer * num_printers

theorem sheets_in_box : total_sheets = 217 := by
  sorry

end NUMINAMATH_CALUDE_sheets_in_box_l3147_314733


namespace NUMINAMATH_CALUDE_function_value_when_previous_is_one_l3147_314772

theorem function_value_when_previous_is_one 
  (f : ℤ → ℤ) 
  (h1 : ∀ n : ℤ, f n = f (n - 1) - n) 
  (h2 : f 4 = 12) :
  ∀ n : ℤ, f (n - 1) = 1 → f n = 7 := by
sorry

end NUMINAMATH_CALUDE_function_value_when_previous_is_one_l3147_314772


namespace NUMINAMATH_CALUDE_bird_eggs_problem_l3147_314743

theorem bird_eggs_problem (total_eggs : ℕ) 
  (eggs_per_nest_tree1 : ℕ) (nests_in_tree1 : ℕ) 
  (eggs_in_front_yard : ℕ) (eggs_in_tree2 : ℕ) : 
  total_eggs = 17 →
  eggs_per_nest_tree1 = 5 →
  nests_in_tree1 = 2 →
  eggs_in_front_yard = 4 →
  total_eggs = nests_in_tree1 * eggs_per_nest_tree1 + eggs_in_front_yard + eggs_in_tree2 →
  eggs_in_tree2 = 3 := by
sorry

end NUMINAMATH_CALUDE_bird_eggs_problem_l3147_314743


namespace NUMINAMATH_CALUDE_base_b_problem_l3147_314773

theorem base_b_problem (b : ℕ) : 
  b > 1 ∧ 
  (2 * b + 5 < b^2) ∧ 
  (5 * b + 2 < b^2) ∧ 
  (5 * b + 2 = 2 * (2 * b + 5)) → 
  b = 8 := by sorry

end NUMINAMATH_CALUDE_base_b_problem_l3147_314773


namespace NUMINAMATH_CALUDE_distinct_distances_lower_bound_l3147_314730

/-- Given n points on a plane, where n ≥ 2, the number of distinct distances k
    between these points satisfies k ≥ √(n - 3/4) - 1/2. -/
theorem distinct_distances_lower_bound (n : ℕ) (k : ℕ) (h : n ≥ 2) :
  k ≥ Real.sqrt (n - 3/4) - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_distinct_distances_lower_bound_l3147_314730


namespace NUMINAMATH_CALUDE_roots_of_equation_l3147_314736

theorem roots_of_equation : ∃ (x₁ x₂ : ℝ), 
  (∀ x : ℝ, (x - 3)^2 = 3 - x ↔ x = x₁ ∨ x = x₂) ∧ 
  x₁ = 3 ∧ x₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3147_314736


namespace NUMINAMATH_CALUDE_pencil_packs_l3147_314708

theorem pencil_packs (pencils_per_pack : ℕ) (pencils_per_row : ℕ) (total_rows : ℕ) : 
  pencils_per_pack = 4 →
  pencils_per_row = 2 →
  total_rows = 70 →
  (total_rows * pencils_per_row) / pencils_per_pack = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_packs_l3147_314708


namespace NUMINAMATH_CALUDE_parallelogram_base_l3147_314790

/-- The area of a parallelogram -/
def area_parallelogram (base height : ℝ) : ℝ := base * height

/-- Theorem: Given a parallelogram with height 36 cm and area 1728 cm², its base is 48 cm -/
theorem parallelogram_base (height area : ℝ) (h1 : height = 36) (h2 : area = 1728) :
  ∃ base : ℝ, area_parallelogram base height = area ∧ base = 48 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3147_314790


namespace NUMINAMATH_CALUDE_range_of_2sin_squared_l3147_314762

theorem range_of_2sin_squared (x : ℝ) : 0 ≤ 2 * (Real.sin x)^2 ∧ 2 * (Real.sin x)^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2sin_squared_l3147_314762


namespace NUMINAMATH_CALUDE_cubic_inequality_l3147_314788

theorem cubic_inequality (x : ℝ) : 
  x^3 - 12*x^2 + 36*x + 8 > 0 ↔ x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3147_314788


namespace NUMINAMATH_CALUDE_paperbacks_count_l3147_314724

/-- The number of books on the shelf -/
def total_books : ℕ := 8

/-- The number of hardback books on the shelf -/
def hardbacks : ℕ := 6

/-- The number of possible selections of 3 books that include at least one paperback -/
def selections_with_paperback : ℕ := 36

/-- The number of paperbacks on the shelf -/
def paperbacks : ℕ := total_books - hardbacks

/-- Theorem stating that the number of paperbacks is 2 -/
theorem paperbacks_count : paperbacks = 2 := by sorry

end NUMINAMATH_CALUDE_paperbacks_count_l3147_314724


namespace NUMINAMATH_CALUDE_tan_theta_value_l3147_314758

theorem tan_theta_value (θ : ℝ) (h : Real.tan (π / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l3147_314758


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l3147_314776

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- Semi-major axis length -/
  a : ℝ
  /-- Semi-minor axis length -/
  b : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- The axes of symmetry are the coordinate axes -/
  axes_are_coordinate_axes : True
  /-- One endpoint of minor axis and two foci form equilateral triangle -/
  equilateral_triangle : b / c = Real.sqrt 3
  /-- Foci are on the y-axis -/
  foci_on_y_axis : True
  /-- Relation between a and c -/
  a_minus_c : a - c = Real.sqrt 3
  /-- Pythagorean theorem for ellipse -/
  ellipse_relation : a^2 = b^2 + c^2

/-- The equation of the special ellipse -/
def ellipse_equation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, y^2 / 12 + x^2 / 9 = 1 ↔ y^2 / e.a^2 + x^2 / e.b^2 = 1

/-- The main theorem about the special ellipse -/
theorem special_ellipse_equation (e : SpecialEllipse) : ellipse_equation e := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l3147_314776


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3147_314723

/-- Definition of a hyperbola with equation x^2 - y^2/4 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- Definition of the asymptotes y = ±2x -/
def asymptotes (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

/-- Theorem stating that the given hyperbola has the specified asymptotes -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ 0 ∧ asymptotes x' y') :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3147_314723


namespace NUMINAMATH_CALUDE_balls_remaining_l3147_314742

/-- The number of baskets --/
def num_baskets : ℕ := 5

/-- The number of tennis balls in each basket --/
def tennis_balls_per_basket : ℕ := 15

/-- The number of soccer balls in each basket --/
def soccer_balls_per_basket : ℕ := 5

/-- The number of students who removed 8 balls each --/
def students_eight : ℕ := 3

/-- The number of students who removed 10 balls each --/
def students_ten : ℕ := 2

/-- The number of balls removed by each student in the first group --/
def balls_removed_eight : ℕ := 8

/-- The number of balls removed by each student in the second group --/
def balls_removed_ten : ℕ := 10

/-- Theorem: The number of balls remaining in the baskets is 56 --/
theorem balls_remaining :
  (num_baskets * (tennis_balls_per_basket + soccer_balls_per_basket)) -
  (students_eight * balls_removed_eight + students_ten * balls_removed_ten) = 56 := by
  sorry

end NUMINAMATH_CALUDE_balls_remaining_l3147_314742


namespace NUMINAMATH_CALUDE_maria_furniture_assembly_l3147_314765

/-- Given the number of chairs, tables, and total assembly time, 
    calculate the time spent on each piece of furniture. -/
def time_per_piece (chairs : ℕ) (tables : ℕ) (total_time : ℕ) : ℚ :=
  (total_time : ℚ) / (chairs + tables : ℚ)

/-- Theorem stating that for 2 chairs, 2 tables, and 32 minutes total time,
    the time per piece is 8 minutes. -/
theorem maria_furniture_assembly : 
  time_per_piece 2 2 32 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maria_furniture_assembly_l3147_314765


namespace NUMINAMATH_CALUDE_priyas_trip_l3147_314721

/-- Priya's trip between towns X, Y, and Z -/
theorem priyas_trip (time_x_to_z : ℝ) (speed_x_to_z : ℝ) (time_z_to_y : ℝ) :
  time_x_to_z = 5 →
  speed_x_to_z = 50 →
  time_z_to_y = 2.0833333333333335 →
  let distance_x_to_z := time_x_to_z * speed_x_to_z
  let distance_z_to_y := distance_x_to_z / 2
  let speed_z_to_y := distance_z_to_y / time_z_to_y
  speed_z_to_y = 60 := by
sorry


end NUMINAMATH_CALUDE_priyas_trip_l3147_314721


namespace NUMINAMATH_CALUDE_octagon_arc_length_l3147_314719

/-- The arc length intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (side_length : ℝ) (h : side_length = 4) :
  let radius : ℝ := side_length
  let circumference : ℝ := 2 * π * radius
  let central_angle : ℝ := π / 4  -- 45 degrees in radians
  let arc_length : ℝ := (central_angle / (2 * π)) * circumference
  arc_length = π :=
by sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l3147_314719


namespace NUMINAMATH_CALUDE_intersection_equation_l3147_314771

theorem intersection_equation (a b : ℝ) (hb : b ≠ 0) :
  ∃ m n : ℤ, (m : ℝ)^3 - a*(m : ℝ)^2 - b*(m : ℝ) = a*(m : ℝ) + b ∧
             (m : ℝ)^3 - a*(m : ℝ)^2 - b*(m : ℝ) = (n : ℝ) →
  2*a - b + 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_equation_l3147_314771


namespace NUMINAMATH_CALUDE_fruit_basket_strawberries_l3147_314751

def fruit_basket (num_strawberries : ℕ) : Prop :=
  let banana_cost : ℕ := 1
  let apple_cost : ℕ := 2
  let avocado_cost : ℕ := 3
  let strawberry_dozen_cost : ℕ := 4
  let half_grape_bunch_cost : ℕ := 2
  let total_cost : ℕ := 28
  let num_bananas : ℕ := 4
  let num_apples : ℕ := 3
  let num_avocados : ℕ := 2
  banana_cost * num_bananas +
  apple_cost * num_apples +
  avocado_cost * num_avocados +
  strawberry_dozen_cost * (num_strawberries / 12) +
  half_grape_bunch_cost * 2 = total_cost

theorem fruit_basket_strawberries : 
  ∃ (n : ℕ), fruit_basket n ∧ n = 24 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_strawberries_l3147_314751


namespace NUMINAMATH_CALUDE_qizhi_median_is_65_l3147_314795

/-- Represents the homework duration data for a group of students -/
structure HomeworkData where
  durations : List Nat
  counts : List Nat
  total_students : Nat

/-- Calculates the median of a dataset given its HomeworkData -/
def calculate_median (data : HomeworkData) : Rat :=
  sorry

/-- The specific homework data for the problem -/
def qizhi_data : HomeworkData :=
  { durations := [50, 60, 70, 80],
    counts := [14, 11, 10, 15],
    total_students := 50 }

/-- Theorem stating that the median of the given homework data is 65 minutes -/
theorem qizhi_median_is_65 : calculate_median qizhi_data = 65 := by
  sorry

end NUMINAMATH_CALUDE_qizhi_median_is_65_l3147_314795


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l3147_314760

theorem cube_sum_of_roots (a b c : ℂ) : 
  (5 * a^3 + 2003 * a + 3005 = 0) → 
  (5 * b^3 + 2003 * b + 3005 = 0) → 
  (5 * c^3 + 2003 * c + 3005 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l3147_314760


namespace NUMINAMATH_CALUDE_max_female_students_theorem_min_group_size_theorem_l3147_314757

/-- Represents the composition of a study group --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

/-- The maximum number of female students when there are 4 teachers --/
def max_female_students_with_4_teachers : ℕ := 6

/-- The minimum number of people in a valid study group --/
def min_group_size : ℕ := 12

/-- Theorem: The maximum number of female students is 6 when there are 4 teachers --/
theorem max_female_students_theorem :
  ∀ g : StudyGroup, is_valid_group g → g.teachers = 4 → g.female_students ≤ max_female_students_with_4_teachers :=
sorry

/-- Theorem: The minimum number of people in a valid study group is 12 --/
theorem min_group_size_theorem :
  ∀ g : StudyGroup, is_valid_group g → g.male_students + g.female_students + g.teachers ≥ min_group_size :=
sorry

end NUMINAMATH_CALUDE_max_female_students_theorem_min_group_size_theorem_l3147_314757


namespace NUMINAMATH_CALUDE_a_value_proof_l3147_314732

theorem a_value_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l3147_314732


namespace NUMINAMATH_CALUDE_problem_solution_l3147_314783

theorem problem_solution :
  ∀ (m x : ℝ),
    (m = 1 → (((x - 3*m) * (x - m) < 0 ∧ |x - 3| ≤ 1) ↔ (2 ≤ x ∧ x < 3))) ∧
    (m > 0 → ((∀ x, |x - 3| ≤ 1 → (x - 3*m) * (x - m) < 0) ∧
              (∃ x, (x - 3*m) * (x - m) < 0 ∧ |x - 3| > 1)) ↔
             (4/3 < m ∧ m < 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3147_314783


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3147_314768

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + count → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, consecutive_nonprimes n 6 ∧ 
           is_prime (n + 6) ∧ 
           ∀ m : ℕ, m < n → ¬(consecutive_nonprimes m 6 ∧ is_prime (m + 6)) ∧
           n + 6 = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3147_314768


namespace NUMINAMATH_CALUDE_square_difference_equality_l3147_314715

theorem square_difference_equality (m n : ℝ) :
  9 * m^2 - (m - 2*n)^2 = 4 * (2*m - n) * (m + n) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3147_314715


namespace NUMINAMATH_CALUDE_johns_money_to_father_l3147_314747

def initial_amount : ℚ := 200
def fraction_to_mother : ℚ := 3/8
def amount_left : ℚ := 65

theorem johns_money_to_father :
  (initial_amount - fraction_to_mother * initial_amount - amount_left) / initial_amount = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_to_father_l3147_314747


namespace NUMINAMATH_CALUDE_stock_market_investment_l3147_314778

theorem stock_market_investment (P : ℝ) (x : ℝ) (h : P > 0) :
  (P + x / 100 * P) * (1 - 30 / 100) = P * (1 + 4.999999999999982 / 100) →
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_stock_market_investment_l3147_314778


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_3_l3147_314746

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ) + 1

theorem sin_2alpha_plus_pi_3 (ω φ α : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ π/2 →
  (∀ x : ℝ, f ω φ (x + π/ω) = f ω φ x) →
  f ω φ (π/3) = 2 →
  f ω φ α = 8/5 →
  π/3 < α ∧ α < 5*π/6 →
  sin (2*α + π/3) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_3_l3147_314746


namespace NUMINAMATH_CALUDE_symmetry_of_f_2x_l3147_314769

def center_of_symmetry (f : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ k : ℤ, p = (k * Real.pi / 2 - Real.pi / 8, 0)}

theorem symmetry_of_f_2x (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (-x) = 3 * Real.cos x - Real.sin x) :
  center_of_symmetry (fun x ↦ f (2 * x)) = 
    {p : ℝ × ℝ | ∃ k : ℤ, p = (k * Real.pi / 2 - Real.pi / 8, 0)} := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_f_2x_l3147_314769


namespace NUMINAMATH_CALUDE_original_pay_before_tax_l3147_314739

/-- Given a 10% tax rate and a take-home pay of $585, prove that the original pay before tax deduction is $650. -/
theorem original_pay_before_tax (tax_rate : ℝ) (take_home_pay : ℝ) (original_pay : ℝ) :
  tax_rate = 0.1 →
  take_home_pay = 585 →
  original_pay * (1 - tax_rate) = take_home_pay →
  original_pay = 650 :=
by sorry

end NUMINAMATH_CALUDE_original_pay_before_tax_l3147_314739


namespace NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3147_314798

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
by sorry

theorem min_choir_members : 
  ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3147_314798


namespace NUMINAMATH_CALUDE_cavalier_projection_triangle_area_l3147_314791

/-- Given a right-angled triangle represented in an oblique cavalier projection
    with a hypotenuse of √2a, prove that its area is √2a² -/
theorem cavalier_projection_triangle_area (a : ℝ) (h : a > 0) :
  let leg1 := Real.sqrt 2 * a
  let leg2 := 2 * a
  (1 / 2) * leg1 * leg2 = Real.sqrt 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_cavalier_projection_triangle_area_l3147_314791


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3147_314797

def geometricSequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence :
  let a₁ : ℚ := 27
  let r : ℚ := 1/6
  geometricSequence a₁ r 15 = 1/14155776 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3147_314797


namespace NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_1_l3147_314767

theorem a_gt_1_sufficient_not_necessary_for_a_sq_gt_1 :
  (∀ a : ℝ, a > 1 → a^2 > 1) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_1_l3147_314767


namespace NUMINAMATH_CALUDE_puppy_sleep_duration_l3147_314793

theorem puppy_sleep_duration (connor_sleep : ℕ) (luke_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  luke_sleep = connor_sleep + 2 →
  puppy_sleep = 2 * luke_sleep →
  puppy_sleep = 16 := by
sorry

end NUMINAMATH_CALUDE_puppy_sleep_duration_l3147_314793


namespace NUMINAMATH_CALUDE_amanda_notebooks_l3147_314789

theorem amanda_notebooks (initial : ℕ) : 
  initial + 6 - 2 = 14 → initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l3147_314789


namespace NUMINAMATH_CALUDE_peggy_total_dolls_l3147_314711

def initial_dolls : ℕ := 6
def grandmother_gift : ℕ := 30
def additional_dolls : ℕ := grandmother_gift / 2

theorem peggy_total_dolls :
  initial_dolls + grandmother_gift + additional_dolls = 51 := by
  sorry

end NUMINAMATH_CALUDE_peggy_total_dolls_l3147_314711


namespace NUMINAMATH_CALUDE_smallest_block_volume_l3147_314794

/-- A rectangular block made of 1-cm cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of cubes in the block -/
def Block.volume (b : Block) : ℕ := b.length * b.width * b.height

/-- The number of cubes not visible when three faces are shown -/
def Block.hiddenCubes (b : Block) : ℕ := (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- One dimension is at least 5 -/
def Block.hasLargeDimension (b : Block) : Prop :=
  b.length ≥ 5 ∨ b.width ≥ 5 ∨ b.height ≥ 5

theorem smallest_block_volume (b : Block) :
  b.hiddenCubes = 252 →
  b.hasLargeDimension →
  ∀ b' : Block, b'.hiddenCubes = 252 → b'.hasLargeDimension → b.volume ≤ b'.volume →
  b.volume = 280 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l3147_314794


namespace NUMINAMATH_CALUDE_complex_subtraction_reciprocal_l3147_314704

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_subtraction_reciprocal : i - (1 : ℂ) / i = 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_reciprocal_l3147_314704


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l3147_314780

theorem greatest_integer_solution (x : ℤ) : (7 - 3 * x > 20) ↔ x ≤ -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l3147_314780


namespace NUMINAMATH_CALUDE_milk_problem_solution_l3147_314761

/-- Calculates the final amount of milk in a storage tank given initial amount,
    pumping out rate and duration, and adding rate and duration. -/
def final_milk_amount (initial : ℝ) (pump_rate : ℝ) (pump_hours : ℝ) 
                      (add_rate : ℝ) (add_hours : ℝ) : ℝ :=
  initial - pump_rate * pump_hours + add_rate * add_hours

/-- Theorem stating that given the specific conditions from the problem,
    the final amount of milk in the storage tank is 28,980 gallons. -/
theorem milk_problem_solution :
  final_milk_amount 30000 2880 4 1500 7 = 28980 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_solution_l3147_314761


namespace NUMINAMATH_CALUDE_inequality_proof_l3147_314729

theorem inequality_proof (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (hrs : r + s = 1) :
  r^r * s^s + r^s * s^r ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3147_314729


namespace NUMINAMATH_CALUDE_boys_circle_distance_l3147_314707

/-- The least total distance traveled by 8 boys on a circle -/
theorem boys_circle_distance (n : ℕ) (r : ℝ) (h_n : n = 8) (h_r : r = 30) :
  let chord_length := 2 * r * Real.sqrt ((2 : ℝ) + Real.sqrt 2) / 2
  let non_adjacent_count := n - 3
  let total_distance := n * non_adjacent_count * chord_length
  total_distance = 1200 * Real.sqrt ((2 : ℝ) + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_boys_circle_distance_l3147_314707


namespace NUMINAMATH_CALUDE_equation_solution_l3147_314712

theorem equation_solution :
  ∃! x : ℝ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 63 / 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3147_314712


namespace NUMINAMATH_CALUDE_f_lower_bound_solution_set_range_l3147_314775

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: f(x) ≥ 1 for all x
theorem f_lower_bound : ∀ x : ℝ, f x ≥ 1 := by sorry

-- Define the set of x values that satisfy the equation
def solution_set : Set ℝ := {x | ∃ a : ℝ, f x = (a^2 + 2) / Real.sqrt (a^2 + 1)}

-- Theorem 2: The solution set is equal to (-∞, 1/2] ∪ [5/2, +∞)
theorem solution_set_range : solution_set = Set.Iic (1/2) ∪ Set.Ici (5/2) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_solution_set_range_l3147_314775


namespace NUMINAMATH_CALUDE_caramel_apple_cost_is_25_l3147_314710

/-- The cost of an ice cream cone in cents -/
def ice_cream_cost : ℕ := 15

/-- The additional cost of a caramel apple compared to an ice cream cone in cents -/
def apple_additional_cost : ℕ := 10

/-- The cost of a caramel apple in cents -/
def caramel_apple_cost : ℕ := ice_cream_cost + apple_additional_cost

/-- Theorem: The cost of a caramel apple is 25 cents -/
theorem caramel_apple_cost_is_25 : caramel_apple_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_caramel_apple_cost_is_25_l3147_314710


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l3147_314722

-- Define the two original circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 3 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 3 = 0

-- Define the line on which the center of the new circle should lie
def center_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the new circle
def new_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 16*y - 3 = 0

-- Theorem statement
theorem circle_satisfies_conditions :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → new_circle x y) ∧
    (∃ (h k : ℝ), center_line h k ∧ 
      ∀ (x y : ℝ), new_circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 12*h + 16*k + 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l3147_314722


namespace NUMINAMATH_CALUDE_expression_equals_24_l3147_314727

theorem expression_equals_24 : 
  2012 * ((3.75 * 1.3 + 3 / 2.6666666666666665) / ((1+3+5+7+9) * 20 + 3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_24_l3147_314727


namespace NUMINAMATH_CALUDE_cookies_left_after_three_days_l3147_314764

/-- Calculates the number of cookies left after a specified number of days -/
def cookies_left (initial_cookies : ℕ) (first_day_consumption : ℕ) (julie_daily : ℕ) (matt_daily : ℕ) (days : ℕ) : ℕ :=
  initial_cookies - (first_day_consumption + (julie_daily + matt_daily) * days)

/-- Theorem stating the number of cookies left after 3 days -/
theorem cookies_left_after_three_days : 
  cookies_left 32 9 2 3 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_three_days_l3147_314764


namespace NUMINAMATH_CALUDE_age_difference_proof_l3147_314714

theorem age_difference_proof (younger_age elder_age : ℕ) 
  (h1 : elder_age > younger_age)
  (h2 : elder_age - 4 = 5 * (younger_age - 4))
  (h3 : younger_age = 29)
  (h4 : elder_age = 49) :
  elder_age - younger_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3147_314714


namespace NUMINAMATH_CALUDE_annual_subscription_cost_is_96_l3147_314731

/-- The cost of a monthly newspaper subscription in dollars. -/
def monthly_cost : ℝ := 10

/-- The discount rate for an annual subscription. -/
def discount_rate : ℝ := 0.2

/-- The number of months in a year. -/
def months_per_year : ℕ := 12

/-- The cost of an annual newspaper subscription with a discount. -/
def annual_subscription_cost : ℝ :=
  monthly_cost * months_per_year * (1 - discount_rate)

/-- Theorem stating that the annual subscription cost is $96. -/
theorem annual_subscription_cost_is_96 :
  annual_subscription_cost = 96 := by
  sorry


end NUMINAMATH_CALUDE_annual_subscription_cost_is_96_l3147_314731


namespace NUMINAMATH_CALUDE_perfect_square_digit_sum_l3147_314718

def is_valid_number (N : ℕ) : Prop :=
  ∃ k : ℕ,
    10000 ≤ N ∧ N < 100000 ∧
    N = 10000 * k + 1000 * (k + 1) + 100 * (k + 2) + 10 * (3 * k) + (k + 3)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem perfect_square_digit_sum :
  ∀ N : ℕ, is_valid_number N →
    ∃ m : ℕ, m * m = N → sum_of_digits m = 15 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_digit_sum_l3147_314718


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l3147_314787

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (abs (x₁ + 3) = 15 ∧ abs (x₂ + 3) = 15) ∧ 
  x₁ ≠ x₂ ∧
  abs (x₁ - x₂) = 30 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l3147_314787


namespace NUMINAMATH_CALUDE_gcd_7524_16083_l3147_314755

theorem gcd_7524_16083 : Nat.gcd 7524 16083 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7524_16083_l3147_314755


namespace NUMINAMATH_CALUDE_all_ciphers_are_good_l3147_314726

/-- Represents a cipher where each letter is replaced by a word. -/
structure Cipher where
  encode : Char → List Char
  decode : List Char → Option Char
  encode_length : ∀ c, (encode c).length ≤ 10

/-- A word is a list of characters. -/
def Word := List Char

/-- Encrypts a word using the given cipher. -/
def encrypt (cipher : Cipher) (word : Word) : Word :=
  word.bind cipher.encode

/-- A cipher is good if any encrypted word can be uniquely decrypted. -/
def is_good_cipher (cipher : Cipher) : Prop :=
  ∀ (w : Word), w.length ≤ 10000 → 
    ∃! (original : Word), encrypt cipher original = w

/-- Main theorem: Any cipher satisfying the given conditions is good. -/
theorem all_ciphers_are_good (cipher : Cipher) : is_good_cipher cipher := by
  sorry

end NUMINAMATH_CALUDE_all_ciphers_are_good_l3147_314726


namespace NUMINAMATH_CALUDE_complex_transformation_l3147_314713

/-- The result of applying a 60° counter-clockwise rotation followed by a dilation 
    with scale factor 2 to the complex number -4 + 3i -/
theorem complex_transformation : 
  let z : ℂ := -4 + 3 * Complex.I
  let rotation : ℂ := Complex.exp (Complex.I * Real.pi / 3)
  let dilation : ℝ := 2
  (dilation * rotation * z) = (-4 - 3 * Real.sqrt 3) + (3 - 4 * Real.sqrt 3) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_transformation_l3147_314713


namespace NUMINAMATH_CALUDE_cake_distribution_l3147_314779

theorem cake_distribution (n : ℕ) (initial_cakes : ℕ) : 
  n = 5 →
  initial_cakes = 2 * (n * (n - 1)) →
  initial_cakes = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_distribution_l3147_314779


namespace NUMINAMATH_CALUDE_afternoon_pear_sales_l3147_314770

/-- Given a salesman who sold pears in the morning and afternoon, this theorem proves
    that if he sold twice as much in the afternoon as in the morning, and the total
    amount sold was 480 kilograms, then he sold 320 kilograms in the afternoon. -/
theorem afternoon_pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 480 →
  afternoon_sales = 320 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_pear_sales_l3147_314770


namespace NUMINAMATH_CALUDE_peters_age_fraction_l3147_314745

/-- Proves that Peter's current age is 1/2 of his mother's age -/
theorem peters_age_fraction (peter_age harriet_age mother_age : ℕ) : 
  harriet_age = 13 →
  mother_age = 60 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  peter_age = mother_age / 2 := by
sorry

end NUMINAMATH_CALUDE_peters_age_fraction_l3147_314745


namespace NUMINAMATH_CALUDE_least_difference_l3147_314706

theorem least_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 5 ∧
  Even x ∧
  Nat.Prime y.toNat ∧ Odd y ∧
  Odd z ∧ z % 3 = 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∀ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧
    y' - x' > 5 ∧
    Even x' ∧
    Nat.Prime y'.toNat ∧ Odd y' ∧
    Odd z' ∧ z' % 3 = 0 ∧
    x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' →
    z - x ≤ z' - x' ∧
    z - x = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_difference_l3147_314706


namespace NUMINAMATH_CALUDE_negation_equivalence_l3147_314777

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3147_314777


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_plus_one_is_perfect_square_l3147_314781

theorem product_of_four_consecutive_integers_plus_one_is_perfect_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_plus_one_is_perfect_square_l3147_314781


namespace NUMINAMATH_CALUDE_stationery_cost_is_18300_l3147_314744

/-- Calculates the total amount paid for stationery given the number of pencil boxes,
    pencils per box, and the costs of pens and pencils. -/
def total_stationery_cost (pencil_boxes : ℕ) (pencils_per_box : ℕ) 
                          (pen_cost : ℕ) (pencil_cost : ℕ) : ℕ :=
  let total_pencils := pencil_boxes * pencils_per_box
  let total_pens := 2 * total_pencils + 300
  total_pens * pen_cost + total_pencils * pencil_cost

/-- Proves that the total amount paid for the stationery is $18300 -/
theorem stationery_cost_is_18300 :
  total_stationery_cost 15 80 5 4 = 18300 := by
  sorry

#eval total_stationery_cost 15 80 5 4

end NUMINAMATH_CALUDE_stationery_cost_is_18300_l3147_314744


namespace NUMINAMATH_CALUDE_stating_table_tennis_sequences_count_l3147_314703

/-- Represents the number of possible game sequences in a table tennis match --/
def table_tennis_sequences : ℕ := 20

/-- 
Theorem stating that the number of possible game sequences in a table tennis match,
where the first player to win three games wins the match, is exactly 20.
--/
theorem table_tennis_sequences_count : table_tennis_sequences = 20 := by
  sorry

end NUMINAMATH_CALUDE_stating_table_tennis_sequences_count_l3147_314703


namespace NUMINAMATH_CALUDE_striped_cube_loop_probability_l3147_314784

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  /-- Each face has a stripe from midpoint to midpoint of opposite edges -/
  faces : Fin 6 → Bool
  /-- For any two opposing faces, one stripe must be perpendicular to the other -/
  opposing_perpendicular : ∀ i : Fin 3, faces i ≠ faces (i + 3)

/-- Predicate to check if a given striped cube forms a valid loop -/
def forms_loop (cube : StripedCube) : Prop :=
  ∃ i : Fin 3, cube.faces i = cube.faces (i + 3) ∧
    (cube.faces ((i + 1) % 3) ≠ cube.faces ((i + 4) % 3)) ∧
    (cube.faces ((i + 2) % 3) ≠ cube.faces ((i + 5) % 3))

/-- The total number of valid striped cube configurations -/
def total_configurations : ℕ := 64

/-- The number of striped cube configurations that form a loop -/
def loop_configurations : ℕ := 6

/-- Theorem stating the probability of a striped cube forming a loop -/
theorem striped_cube_loop_probability :
  (loop_configurations : ℚ) / total_configurations = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_striped_cube_loop_probability_l3147_314784


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3147_314774

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → b = 10 → (a + b + c) / 3 = a + 20 → (a + b + c) / 3 = c - 10 → 
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3147_314774


namespace NUMINAMATH_CALUDE_investment_percentage_l3147_314738

theorem investment_percentage (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) 
  (h1 : initial_investment = 1400)
  (h2 : initial_rate = 0.05)
  (h3 : additional_investment = 700)
  (h4 : additional_rate = 0.08) : 
  (initial_investment * initial_rate + additional_investment * additional_rate) / 
  (initial_investment + additional_investment) = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l3147_314738


namespace NUMINAMATH_CALUDE_factorization_equality_l3147_314741

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3147_314741


namespace NUMINAMATH_CALUDE_penny_stack_more_valuable_l3147_314766

/-- Represents a stack of coins -/
structure CoinStack :=
  (onePence : ℕ)
  (twoPence : ℕ)
  (fivePence : ℕ)

/-- Calculates the height of a coin stack in millimeters -/
def stackHeight (stack : CoinStack) : ℚ :=
  1.6 * stack.onePence + 2.05 * stack.twoPence + 1.75 * stack.fivePence

/-- Calculates the value of a coin stack in pence -/
def stackValue (stack : CoinStack) : ℕ :=
  stack.onePence + 2 * stack.twoPence + 5 * stack.fivePence

/-- Checks if a stack is valid according to the problem constraints -/
def isValidStack (stack : CoinStack) : Prop :=
  stackHeight stack = stackValue stack ∧ 
  (stack.onePence > 0 ∨ stack.twoPence > 0 ∨ stack.fivePence > 0)

/-- Joe's optimal stack using only 1p and 5p coins -/
def joesStack : CoinStack :=
  ⟨65, 0, 12⟩

/-- Penny's optimal stack using only 2p and 5p coins -/
def pennysStack : CoinStack :=
  ⟨0, 65, 1⟩

theorem penny_stack_more_valuable :
  isValidStack joesStack ∧
  isValidStack pennysStack ∧
  stackValue pennysStack > stackValue joesStack :=
sorry

end NUMINAMATH_CALUDE_penny_stack_more_valuable_l3147_314766


namespace NUMINAMATH_CALUDE_count_negative_numbers_l3147_314786

theorem count_negative_numbers : let numbers := [-(-3), |-2|, (-2)^3, -3^2]
  ∃ (negative_count : ℕ), negative_count = (numbers.filter (λ x => x < 0)).length ∧ negative_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l3147_314786


namespace NUMINAMATH_CALUDE_range_of_t_l3147_314737

def A : Set ℝ := {x | x^2 - 3*x + 2 ≥ 0}
def B (t : ℝ) : Set ℝ := {x | x ≥ t}

theorem range_of_t (t : ℝ) : A ∪ B t = A → t ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l3147_314737


namespace NUMINAMATH_CALUDE_ashok_subjects_l3147_314734

theorem ashok_subjects (average_all : ℝ) (average_five : ℝ) (sixth_subject : ℝ) 
  (h1 : average_all = 72)
  (h2 : average_five = 74)
  (h3 : sixth_subject = 62) :
  ∃ n : ℕ, n = 6 ∧ n * average_all = 5 * average_five + sixth_subject :=
by
  sorry

end NUMINAMATH_CALUDE_ashok_subjects_l3147_314734


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3147_314700

def f (x : ℝ) : ℝ := -x * abs x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3147_314700


namespace NUMINAMATH_CALUDE_dress_price_calculation_l3147_314735

theorem dress_price_calculation (discount_percent : ℝ) (final_price : ℝ) : 
  discount_percent = 30 → final_price = 35 → (100 - discount_percent) / 100 * 50 = final_price := by
  sorry

end NUMINAMATH_CALUDE_dress_price_calculation_l3147_314735


namespace NUMINAMATH_CALUDE_smallest_clock_equivalent_is_nine_l3147_314748

/-- A number is clock equivalent to its square if it's congruent to its square modulo 12 -/
def IsClockEquivalent (n : ℕ) : Prop := n ≡ n^2 [MOD 12]

/-- The smallest number greater than 5 that is clock equivalent to its square -/
def SmallestClockEquivalent : ℕ := 9

theorem smallest_clock_equivalent_is_nine :
  IsClockEquivalent SmallestClockEquivalent ∧
  ∀ n : ℕ, 5 < n ∧ n < SmallestClockEquivalent → ¬IsClockEquivalent n :=
by sorry

end NUMINAMATH_CALUDE_smallest_clock_equivalent_is_nine_l3147_314748


namespace NUMINAMATH_CALUDE_function_f_property_l3147_314792

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_f_property : 
  (∀ x, f x + 2 * f (27 - x) = x) → f 11 = 7 := by sorry

end NUMINAMATH_CALUDE_function_f_property_l3147_314792


namespace NUMINAMATH_CALUDE_probability_of_drawing_parts_l3147_314763

def total_parts : ℕ := 10
def drawn_parts : ℕ := 6

def prob_draw_one (n m k : ℕ) : ℚ :=
  (n.choose k) / (m.choose k)

def prob_draw_two (n m k : ℕ) : ℚ :=
  ((n-2).choose (k-2)) / (m.choose k)

theorem probability_of_drawing_parts :
  (prob_draw_one (total_parts - 1) total_parts drawn_parts = 3/5) ∧
  (prob_draw_two (total_parts - 2) total_parts drawn_parts = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_parts_l3147_314763


namespace NUMINAMATH_CALUDE_limit_inequality_l3147_314701

theorem limit_inequality : 12.37 * (3/2 - 1/3) > Real.cos (π/10) := by
  sorry

end NUMINAMATH_CALUDE_limit_inequality_l3147_314701


namespace NUMINAMATH_CALUDE_unique_base7_digit_l3147_314753

/-- Converts a base 7 number of the form 52x4₇ to base 10 --/
def base7ToBase10 (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

/-- The set of valid digits in base 7 --/
def base7Digits : Set ℕ := {0, 1, 2, 3, 4, 5, 6}

theorem unique_base7_digit : 
  ∃! x : ℕ, x ∈ base7Digits ∧ isDivisibleBy19 (base7ToBase10 x) := by sorry

end NUMINAMATH_CALUDE_unique_base7_digit_l3147_314753


namespace NUMINAMATH_CALUDE_odd_function_implies_k_equals_two_inequality_range_minimum_value_of_g_l3147_314785

noncomputable section

variable (a : ℝ) (k : ℝ)

def f (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem odd_function_implies_k_equals_two
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x, f a k x = -f a k (-x)) :
  k = 2 := by sorry

theorem inequality_range
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a 2 1 < 0) :
  (∀ t, (∀ x, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ -3 < t ∧ t < 5) := by sorry

theorem minimum_value_of_g
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a 2 1 = 3/2) :
  (∃ x_min : ℝ, x_min ∈ Set.Ici 1 ∧
    ∀ x, x ∈ Set.Ici 1 →
      a^(2*x) + a^(-2*x) - 2 * f a 2 x ≥ a^(2*x_min) + a^(-2*x_min) - 2 * f a 2 x_min) ∧
  (∃ x_0 : ℝ, x_0 ∈ Set.Ici 1 ∧ a^(2*x_0) + a^(-2*x_0) - 2 * f a 2 x_0 = 5/4) := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_k_equals_two_inequality_range_minimum_value_of_g_l3147_314785


namespace NUMINAMATH_CALUDE_max_odd_integers_l3147_314754

/-- Given a list of six positive integers, returns true if their product is even -/
def productIsEven (nums : List Nat) : Prop :=
  nums.length = 6 ∧ nums.all (· > 0) ∧ (nums.prod % 2 = 0)

/-- Given a list of six positive integers, returns true if their sum is odd -/
def sumIsOdd (nums : List Nat) : Prop :=
  nums.length = 6 ∧ nums.all (· > 0) ∧ (nums.sum % 2 = 1)

/-- Returns the count of odd numbers in a list -/
def oddCount (nums : List Nat) : Nat :=
  nums.filter (· % 2 = 1) |>.length

theorem max_odd_integers (nums : List Nat) 
  (h1 : productIsEven nums) (h2 : sumIsOdd nums) : 
  oddCount nums ≤ 5 ∧ ∃ (nums' : List Nat), productIsEven nums' ∧ sumIsOdd nums' ∧ oddCount nums' = 5 :=
sorry

end NUMINAMATH_CALUDE_max_odd_integers_l3147_314754


namespace NUMINAMATH_CALUDE_investment_average_rate_l3147_314752

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) : 
  total = 5500 ∧ 
  rate1 = 0.03 ∧ 
  rate2 = 0.07 ∧ 
  (∃ x : ℝ, x > 0 ∧ x < total ∧ rate1 * (total - x) = rate2 * x) →
  (rate1 * (total - (rate2 * total) / (rate1 + rate2)) + rate2 * ((rate2 * total) / (rate1 + rate2))) / total = 0.042 := by
  sorry

#check investment_average_rate

end NUMINAMATH_CALUDE_investment_average_rate_l3147_314752


namespace NUMINAMATH_CALUDE_third_digit_even_l3147_314717

theorem third_digit_even (n : ℤ) : ∃ k : ℤ, (10*n + 5)^2 = 1000*k + 200*m + 25 ∧ m % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_third_digit_even_l3147_314717


namespace NUMINAMATH_CALUDE_total_time_is_541_l3147_314799

-- Define the structure for a cupcake batch
structure CupcakeBatch where
  name : String
  bakeTime : ℕ
  iceTime : ℕ
  decorateTimePerCupcake : ℕ

-- Define the number of cupcakes per batch
def cupcakesPerBatch : ℕ := 6

-- Define the batches
def chocolateBatch : CupcakeBatch := ⟨"Chocolate", 18, 25, 10⟩
def vanillaBatch : CupcakeBatch := ⟨"Vanilla", 20, 30, 15⟩
def redVelvetBatch : CupcakeBatch := ⟨"Red Velvet", 22, 28, 12⟩
def lemonBatch : CupcakeBatch := ⟨"Lemon", 24, 32, 20⟩

-- Define the list of all batches
def allBatches : List CupcakeBatch := [chocolateBatch, vanillaBatch, redVelvetBatch, lemonBatch]

-- Calculate the total time for a single batch
def batchTotalTime (batch : CupcakeBatch) : ℕ :=
  batch.bakeTime + batch.iceTime + (batch.decorateTimePerCupcake * cupcakesPerBatch)

-- Theorem: The total time to make, ice, and decorate all cupcakes is 541 minutes
theorem total_time_is_541 : (allBatches.map batchTotalTime).sum = 541 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_541_l3147_314799


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l3147_314782

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear -/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(0,2), B(3,0), and C(m,1-m) are collinear, then m = -9 -/
theorem collinear_points_m_value :
  ∀ m : ℝ, are_collinear (0, 2) (3, 0) (m, 1 - m) → m = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l3147_314782


namespace NUMINAMATH_CALUDE_strawberry_fraction_remaining_l3147_314716

theorem strawberry_fraction_remaining 
  (num_hedgehogs : ℕ) 
  (num_baskets : ℕ) 
  (strawberries_per_basket : ℕ) 
  (strawberries_eaten_per_hedgehog : ℕ) : 
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  strawberries_eaten_per_hedgehog = 1050 →
  (num_baskets * strawberries_per_basket - num_hedgehogs * strawberries_eaten_per_hedgehog : ℚ) / 
  (num_baskets * strawberries_per_basket) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_fraction_remaining_l3147_314716


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l3147_314728

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : parallelogramArea 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l3147_314728


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3147_314702

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3147_314702


namespace NUMINAMATH_CALUDE_solution_bounded_l3147_314756

open Real

/-- A function satisfying the differential equation y'' + e^x y = 0 is bounded -/
theorem solution_bounded (f : ℝ → ℝ) (hf : ∀ x, (deriv^[2] f) x + exp x * f x = 0) :
  ∃ M, ∀ x, |f x| ≤ M :=
sorry

end NUMINAMATH_CALUDE_solution_bounded_l3147_314756


namespace NUMINAMATH_CALUDE_square_sum_of_reciprocal_sum_and_sum_l3147_314725

theorem square_sum_of_reciprocal_sum_and_sum (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) (h2 : x + y = 5) : x^2 + y^2 = 35/2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_reciprocal_sum_and_sum_l3147_314725
