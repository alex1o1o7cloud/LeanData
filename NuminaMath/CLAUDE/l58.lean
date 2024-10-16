import Mathlib

namespace NUMINAMATH_CALUDE_committee_selection_count_l58_5824

/-- The number of ways to choose a committee of size k from n people -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the club -/
def club_size : ℕ := 30

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem: The number of ways to choose a 5-person committee from a 30-person club is 142506 -/
theorem committee_selection_count : choose club_size committee_size = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l58_5824


namespace NUMINAMATH_CALUDE_janeth_balloons_left_l58_5820

/-- Calculates the number of balloons left after some burst -/
def balloons_left (round_bags : ℕ) (round_per_bag : ℕ) (long_bags : ℕ) (long_per_bag : ℕ) (burst : ℕ) : ℕ :=
  round_bags * round_per_bag + long_bags * long_per_bag - burst

/-- Proves that the number of balloons left is 215 given the specified conditions -/
theorem janeth_balloons_left : 
  balloons_left 5 20 4 30 5 = 215 := by
  sorry

end NUMINAMATH_CALUDE_janeth_balloons_left_l58_5820


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l58_5893

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (r s : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, r x * s x = k

theorem inverse_variation_problem (r s : ℝ → ℝ) 
  (h1 : VaryInversely r s)
  (h2 : r 1 = 1500)
  (h3 : s 1 = 0.4)
  (h4 : r 2 = 3000) :
  s 2 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l58_5893


namespace NUMINAMATH_CALUDE_table_length_is_77_l58_5833

/-- The length of the rectangular table -/
def table_length : ℕ := 77

/-- The width of the rectangular table -/
def table_width : ℕ := 80

/-- The height of each sheet of paper -/
def sheet_height : ℕ := 5

/-- The width of each sheet of paper -/
def sheet_width : ℕ := 8

/-- The horizontal and vertical increment for each subsequent sheet -/
def increment : ℕ := 1

theorem table_length_is_77 :
  ∃ (n : ℕ), 
    table_length = sheet_height + n * increment ∧
    table_width = sheet_width + n * increment ∧
    table_width - table_length = sheet_width - sheet_height := by
  sorry

end NUMINAMATH_CALUDE_table_length_is_77_l58_5833


namespace NUMINAMATH_CALUDE_train_length_is_140_meters_l58_5813

-- Define the given conditions
def train_speed : Real := 45 -- km/hr
def bridge_crossing_time : Real := 30 -- seconds
def bridge_length : Real := 235 -- meters

-- Define the theorem
theorem train_length_is_140_meters :
  let speed_mps := train_speed * 1000 / 3600 -- Convert km/hr to m/s
  let total_distance := speed_mps * bridge_crossing_time
  let train_length := total_distance - bridge_length
  train_length = 140 := by sorry

end NUMINAMATH_CALUDE_train_length_is_140_meters_l58_5813


namespace NUMINAMATH_CALUDE_largest_package_size_l58_5898

theorem largest_package_size (john_markers alice_markers : ℕ) 
  (h1 : john_markers = 36) (h2 : alice_markers = 60) : 
  Nat.gcd john_markers alice_markers = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l58_5898


namespace NUMINAMATH_CALUDE_hyperbola_properties_l58_5899

/-- Definition of hyperbola C -/
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

/-- Right focus of hyperbola C -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Directrix of hyperbola C -/
def directrix (x : ℝ) : Prop := x = 3/2

/-- Standard equation of hyperbola C -/
def standard_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- Asymptotes of hyperbola C -/
def asymptotes (x y : ℝ) : Prop := y = x / Real.sqrt 3 ∨ y = -x / Real.sqrt 3

/-- Equation of hyperbola sharing asymptotes and passing through (√3, 2) -/
def shared_asymptotes_equation (x y : ℝ) : Prop := y^2 / 3 - x^2 / 9 = 1

theorem hyperbola_properties :
  ∀ (x y a b : ℝ),
  hyperbola_C x y a b →
  (∃ (c : ℝ), right_focus = (c, 0)) →
  (∃ (d : ℝ), directrix d) →
  standard_equation x y ∧
  asymptotes x y ∧
  shared_asymptotes_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l58_5899


namespace NUMINAMATH_CALUDE_seed_germination_problem_l58_5809

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.25 * x + 0.35 * 200) / (x + 200) = 0.28999999999999996 → 
  x = 300 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l58_5809


namespace NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l58_5830

theorem three_digit_difference_divisible_by_nine :
  ∀ (a b c : ℕ), 
    a ≤ 9 → b ≤ 9 → c ≤ 9 → a ≠ 0 →
    ∃ (k : ℤ), (100 * a + 10 * b + c) - (a + b + c) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l58_5830


namespace NUMINAMATH_CALUDE_coplanar_vectors_lambda_l58_5877

/-- Given three vectors a, b, and c in ℝ³, if they are coplanar and have specific coordinates,
    then the third coordinate of c equals 9. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) : 
  a = (2, -1, 3) → b = (-1, 4, -2) → c.1 = 7 → c.2.1 = 7 →
  (∃ (m n : ℝ), c = m • a + n • b) →
  c.2.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_coplanar_vectors_lambda_l58_5877


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l58_5863

theorem simplify_and_evaluate : 
  let x : ℚ := -4
  let y : ℚ := 1/2
  (x + 2*y)^2 - x*(x + 3*y) - 4*y^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l58_5863


namespace NUMINAMATH_CALUDE_max_students_distribution_l58_5840

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1204) (h2 : pencils = 840) :
  Nat.gcd pens pencils = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l58_5840


namespace NUMINAMATH_CALUDE_biker_journey_west_distance_l58_5878

/-- Represents the journey of a biker -/
structure BikerJourney where
  west : ℝ
  north1 : ℝ
  east : ℝ
  north2 : ℝ
  straightLineDistance : ℝ

/-- Theorem stating the distance traveled west given specific journey parameters -/
theorem biker_journey_west_distance (journey : BikerJourney) 
  (h1 : journey.north1 = 5)
  (h2 : journey.east = 4)
  (h3 : journey.north2 = 15)
  (h4 : journey.straightLineDistance = 20.396078054371138) :
  journey.west = 8 := by
  sorry

end NUMINAMATH_CALUDE_biker_journey_west_distance_l58_5878


namespace NUMINAMATH_CALUDE_functional_equation_solution_l58_5815

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = x * f(1) -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = x * f 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l58_5815


namespace NUMINAMATH_CALUDE_library_fiction_percentage_l58_5828

theorem library_fiction_percentage 
  (total_volumes : ℕ) 
  (fiction_percentage : ℚ)
  (transfer_fraction : ℚ)
  (fiction_transfer_fraction : ℚ)
  (h1 : total_volumes = 18360)
  (h2 : fiction_percentage = 30 / 100)
  (h3 : transfer_fraction = 1 / 3)
  (h4 : fiction_transfer_fraction = 1 / 5) :
  let original_fiction := (fiction_percentage * total_volumes : ℚ)
  let transferred_volumes := (transfer_fraction * total_volumes : ℚ)
  let transferred_fiction := (fiction_transfer_fraction * transferred_volumes : ℚ)
  let remaining_fiction := original_fiction - transferred_fiction
  let remaining_volumes := total_volumes - transferred_volumes
  (remaining_fiction / remaining_volumes) * 100 = 35 := by
sorry


end NUMINAMATH_CALUDE_library_fiction_percentage_l58_5828


namespace NUMINAMATH_CALUDE_locus_and_tangent_lines_l58_5881

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point M on the ellipse
def M : ℝ × ℝ := sorry

-- Define point N as the projection of M on x = 3
def N : ℝ × ℝ := (3, M.2)

-- Define point P
def P : ℝ × ℝ := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector from O to a point
def vector_to (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - O.1, p.2 - O.2)

-- Define the locus E
def E (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point A
def A : ℝ × ℝ := (1, 4)

-- Define the tangent line equations
def tangent_line_1 (x y : ℝ) : Prop := x = 1
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y - 19 = 0

theorem locus_and_tangent_lines :
  ellipse M.1 M.2 ∧
  N = (3, M.2) ∧
  vector_to P = vector_add (vector_to M) (vector_to N) →
  (∀ x y, E x y ↔ (∃ m n, ellipse m n ∧ x = m + 3 ∧ y = 2*n)) ∧
  (∀ x y, (tangent_line_1 x y ∨ tangent_line_2 x y) ↔
    (E x y ∧ (x - A.1)^2 + (y - A.2)^2 = ((x - 3)^2 + y^2))) :=
sorry

end NUMINAMATH_CALUDE_locus_and_tangent_lines_l58_5881


namespace NUMINAMATH_CALUDE_cards_lost_l58_5819

def initial_cards : ℕ := 88
def remaining_cards : ℕ := 18

theorem cards_lost : initial_cards - remaining_cards = 70 := by
  sorry

end NUMINAMATH_CALUDE_cards_lost_l58_5819


namespace NUMINAMATH_CALUDE_student_distribution_l58_5874

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with at least one object in each box -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 college students -/
def num_students : ℕ := 5

/-- There are 3 factories -/
def num_factories : ℕ := 3

/-- The theorem stating that there are 150 ways to distribute 5 students among 3 factories
    with at least one student in each factory -/
theorem student_distribution : distribute num_students num_factories = 150 := by sorry

end NUMINAMATH_CALUDE_student_distribution_l58_5874


namespace NUMINAMATH_CALUDE_base4_sum_equals_2133_l58_5879

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Converts a base 4 number to its decimal representation --/
def to_decimal (n : Base4) : ℕ := sorry

/-- Converts a decimal number to its base 4 representation --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Adds two base 4 numbers --/
def base4_add (a b : Base4) : Base4 := sorry

theorem base4_sum_equals_2133 :
  let a := to_base4 2
  let b := to_base4 (4 + 3)
  let c := to_base4 (16 + 12 + 2)
  let d := to_base4 (256 + 192 + 0)
  base4_add (base4_add (base4_add a b) c) d = to_base4 (512 + 48 + 12 + 3) := by
  sorry

end NUMINAMATH_CALUDE_base4_sum_equals_2133_l58_5879


namespace NUMINAMATH_CALUDE_intersection_M_N_l58_5823

def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}

def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l58_5823


namespace NUMINAMATH_CALUDE_min_semi_focal_distance_l58_5847

/-- The minimum semi-focal distance of a hyperbola satisfying certain conditions -/
theorem min_semi_focal_distance (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → c^2 = a^2 + b^2) →
  (a * b / Real.sqrt (a^2 + b^2) = c / 3 + 1) →
  c ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_semi_focal_distance_l58_5847


namespace NUMINAMATH_CALUDE_gcd_2024_2048_l58_5876

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2024_2048_l58_5876


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l58_5871

theorem pizza_toppings_combinations (n m : ℕ) (h1 : n = 7) (h2 : m = 4) : 
  Nat.choose n m = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l58_5871


namespace NUMINAMATH_CALUDE_rectangular_enclosure_properties_l58_5829

/-- Represents the area of a rectangular enclosure with perimeter 32 meters and side length x -/
def area (x : ℝ) : ℝ := -x^2 + 16*x

/-- Theorem stating the properties of the rectangular enclosure -/
theorem rectangular_enclosure_properties :
  ∀ x : ℝ, 0 < x → x < 16 →
  (∀ y : ℝ, y = area x → 
    (y = 60 → (x = 6 ∨ x = 10)) ∧
    (y ≤ 64) ∧
    (y = 64 ↔ x = 8)) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_enclosure_properties_l58_5829


namespace NUMINAMATH_CALUDE_china_population_scientific_notation_l58_5822

/-- Represents the population of China in millions -/
def china_population : ℝ := 1412.60

/-- The scientific notation representation of the population -/
def scientific_notation : ℝ := 1.4126 * (10 ^ 5)

/-- Theorem stating that the scientific notation representation is correct -/
theorem china_population_scientific_notation :
  china_population = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_china_population_scientific_notation_l58_5822


namespace NUMINAMATH_CALUDE_work_completion_men_difference_l58_5842

theorem work_completion_men_difference (work : ℕ) : 
  ∀ (m n : ℕ), 
    m = 20 → 
    m * 10 = n * 20 → 
    m - n = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_men_difference_l58_5842


namespace NUMINAMATH_CALUDE_jesse_carpet_problem_l58_5857

/-- Given a room with length and width, and some carpet already available,
    calculate the additional carpet needed to cover the whole floor. -/
def additional_carpet_needed (length width available_carpet : ℝ) : ℝ :=
  length * width - available_carpet

/-- Theorem: Given a room that is 4 feet long and 20 feet wide, with 18 square feet
    of carpet already available, the additional carpet needed is 62 square feet. -/
theorem jesse_carpet_problem :
  additional_carpet_needed 4 20 18 = 62 := by
  sorry

end NUMINAMATH_CALUDE_jesse_carpet_problem_l58_5857


namespace NUMINAMATH_CALUDE_fish_count_theorem_l58_5848

def is_valid_fish_count (t : ℕ) : Prop :=
  (t > 10 ∧ t > 15 ∧ t ≤ 18) ∨
  (t > 10 ∧ t ≤ 15 ∧ t > 18) ∨
  (t ≤ 10 ∧ t > 15 ∧ t > 18)

theorem fish_count_theorem :
  ∀ t : ℕ, is_valid_fish_count t ↔ (t = 16 ∨ t = 17 ∨ t = 18) :=
by sorry

end NUMINAMATH_CALUDE_fish_count_theorem_l58_5848


namespace NUMINAMATH_CALUDE_lcm_inequality_l58_5861

theorem lcm_inequality (k m n : ℕ) : 
  (Nat.lcm k m) * (Nat.lcm m n) * (Nat.lcm n k) ≥ (Nat.lcm (Nat.lcm k m) n)^2 := by
sorry

end NUMINAMATH_CALUDE_lcm_inequality_l58_5861


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l58_5827

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b^2 = a^2 - 2*b*c →
  A = 2*π/3 →
  C = π/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l58_5827


namespace NUMINAMATH_CALUDE_barn_painted_area_l58_5811

/-- Calculates the total area to be painted in a barn with given dimensions and conditions -/
def total_painted_area (length width height : ℝ) (window_side : ℝ) (num_windows : ℕ) : ℝ :=
  let long_wall_area := length * height
  let wide_wall_area := width * height
  let ceiling_area := length * width
  let window_area := window_side * window_side * num_windows
  let total_wall_area := 2 * (2 * long_wall_area + 2 * wide_wall_area - window_area)
  total_wall_area + ceiling_area

/-- The total area to be painted in the barn is 796 square yards -/
theorem barn_painted_area :
  total_painted_area 12 15 6 2 2 = 796 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l58_5811


namespace NUMINAMATH_CALUDE_always_possible_scatter_plot_l58_5801

/-- Represents statistical data for two variables -/
structure TwoVariableData where
  -- We don't need to specify the internal structure of the data
  -- as the problem doesn't provide details about it

/-- Represents a scatter plot -/
structure ScatterPlot where
  -- We don't need to specify the internal structure of the scatter plot
  -- as the problem doesn't provide details about it

/-- States that it's always possible to create a scatter plot from two-variable data -/
theorem always_possible_scatter_plot (data : TwoVariableData) : 
  ∃ (plot : ScatterPlot), true :=
sorry

end NUMINAMATH_CALUDE_always_possible_scatter_plot_l58_5801


namespace NUMINAMATH_CALUDE_rachel_plant_arrangement_l58_5892

def num_arrangements (n : ℕ) : ℕ :=
  let all_under_one := 2  -- All plants under one white lamp or one red lamp
  let all_same_color := 2 * (n.choose 2)  -- All plants under lamps of the same color
  let diff_colors := (n.choose 2) + 2 * (n.choose 1)  -- Plants under lamps of different colors
  all_under_one + all_same_color + diff_colors

theorem rachel_plant_arrangement :
  num_arrangements 4 = 28 :=
by sorry

end NUMINAMATH_CALUDE_rachel_plant_arrangement_l58_5892


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l58_5817

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {2, 3, 4}

-- Define set B
def B : Finset Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l58_5817


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l58_5890

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 9)
  (h3 : ∀ n, a (n + 1) = a n + d)
  (h4 : (a 4) ^ 2 = (a 1) * (a 8)) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l58_5890


namespace NUMINAMATH_CALUDE_reservoir_capacity_l58_5808

theorem reservoir_capacity : 
  ∀ (C : ℝ), 
  (C / 3 + 150 = 3 * C / 4) → 
  C = 360 := by
sorry

end NUMINAMATH_CALUDE_reservoir_capacity_l58_5808


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l58_5831

theorem sin_cos_sum_equals_half : 
  Real.sin (17 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (167 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l58_5831


namespace NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l58_5862

/-- Theorem: Equal volumes of modified cylinders -/
theorem equal_volumes_of_modified_cylinders :
  let initial_radius : ℝ := 5
  let initial_height : ℝ := 10
  let radius_increase : ℝ := 4
  let volume1 := π * (initial_radius + radius_increase)^2 * initial_height
  let volume2 (x : ℝ) := π * initial_radius^2 * (initial_height + x)
  ∀ x : ℝ, volume1 = volume2 x ↔ x = 112 / 5 :=
by sorry

end NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l58_5862


namespace NUMINAMATH_CALUDE_circle_division_sum_l58_5864

/-- The sum of numbers on a circle after n steps of division -/
def circleSum (n : ℕ) : ℕ :=
  2 * 3^n

/-- The process of dividing the circle and summing numbers -/
def divideAndSum : ℕ → ℕ
  | 0 => 2  -- Initial sum: 1 + 1
  | n + 1 => 3 * divideAndSum n

theorem circle_division_sum (n : ℕ) :
  divideAndSum n = circleSum n := by
  sorry

end NUMINAMATH_CALUDE_circle_division_sum_l58_5864


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l58_5891

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 + x^2/m = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, hyperbola_equation x y m ↔ asymptote_equation x y) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l58_5891


namespace NUMINAMATH_CALUDE_blue_markers_count_l58_5887

theorem blue_markers_count (total_markers red_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : red_markers = 2315) :
  total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_CALUDE_blue_markers_count_l58_5887


namespace NUMINAMATH_CALUDE_f_properties_l58_5865

noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then x + 2
  else if x = 0 then 0
  else x - 2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f x < 2 ↔ x < 4) := by sorry

end NUMINAMATH_CALUDE_f_properties_l58_5865


namespace NUMINAMATH_CALUDE_frame_sales_ratio_l58_5870

/-- Given:
  - Dorothy sells glass frames at half the price of Jemma
  - Jemma sells glass frames at 5 dollars each
  - Jemma sold 400 frames
  - They made 2500 dollars together in total
Prove that the ratio of frames Jemma sold to frames Dorothy sold is 2:1 -/
theorem frame_sales_ratio (jemma_price : ℚ) (jemma_sold : ℕ) (total_revenue : ℚ) 
    (h1 : jemma_price = 5)
    (h2 : jemma_sold = 400)
    (h3 : total_revenue = 2500) : 
  ∃ (dorothy_sold : ℕ), jemma_sold = 2 * dorothy_sold := by
  sorry

#check frame_sales_ratio

end NUMINAMATH_CALUDE_frame_sales_ratio_l58_5870


namespace NUMINAMATH_CALUDE_unique_m_value_l58_5826

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + m - 1 = 0}

theorem unique_m_value : ∃! m : ℝ, A ∪ B m = A := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l58_5826


namespace NUMINAMATH_CALUDE_hamburgers_count_l58_5866

/-- The number of hamburgers initially made -/
def initial_hamburgers : ℝ := 9.0

/-- The number of additional hamburgers made -/
def additional_hamburgers : ℝ := 3.0

/-- The total number of hamburgers made -/
def total_hamburgers : ℝ := initial_hamburgers + additional_hamburgers

theorem hamburgers_count : total_hamburgers = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_count_l58_5866


namespace NUMINAMATH_CALUDE_total_rainfall_three_days_l58_5883

/-- Calculates the total rainfall over three days given specific conditions --/
theorem total_rainfall_three_days 
  (monday_hours : ℕ) 
  (monday_rate : ℕ) 
  (tuesday_hours : ℕ) 
  (tuesday_rate : ℕ) 
  (wednesday_hours : ℕ) 
  (h_monday : monday_hours = 7 ∧ monday_rate = 1)
  (h_tuesday : tuesday_hours = 4 ∧ tuesday_rate = 2)
  (h_wednesday : wednesday_hours = 2)
  (h_wednesday_rate : wednesday_hours * (2 * tuesday_rate) = 8) :
  monday_hours * monday_rate + 
  tuesday_hours * tuesday_rate + 
  wednesday_hours * (2 * tuesday_rate) = 23 := by
sorry


end NUMINAMATH_CALUDE_total_rainfall_three_days_l58_5883


namespace NUMINAMATH_CALUDE_solution_to_equation_l58_5884

theorem solution_to_equation (y : ℝ) (h1 : y ≠ 3) (h2 : y ≠ 3/2) :
  (y^2 - 11*y + 24)/(y - 3) + (2*y^2 + 7*y - 18)/(2*y - 3) = -10 ↔ y = -4 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l58_5884


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_expression_l58_5807

def is_integer_expression (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.factorial (n^3 - 1)) = k * (Nat.factorial n)^(n^2)

theorem unique_integer_satisfying_expression :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ is_integer_expression n :=
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_expression_l58_5807


namespace NUMINAMATH_CALUDE_point_C_in_fourth_quadrant_l58_5839

/-- A point in the 2D Cartesian coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point we want to prove is in the fourth quadrant -/
def point_C : Point :=
  { x := 1, y := -2 }

/-- Theorem: point_C is in the fourth quadrant -/
theorem point_C_in_fourth_quadrant : is_in_fourth_quadrant point_C := by
  sorry

end NUMINAMATH_CALUDE_point_C_in_fourth_quadrant_l58_5839


namespace NUMINAMATH_CALUDE_first_ring_at_three_am_l58_5844

/-- A clock that rings at regular intervals throughout the day -/
structure RingingClock where
  ring_interval : ℕ  -- Interval between rings in hours
  rings_per_day : ℕ  -- Number of times the clock rings in a day

/-- The time of day in hours (0 to 23) -/
def Time := Fin 24

/-- Calculate the time of the first ring for a given clock -/
def first_ring_time (clock : RingingClock) : Time :=
  ⟨clock.ring_interval, by sorry⟩

theorem first_ring_at_three_am 
  (clock : RingingClock) 
  (h1 : clock.ring_interval = 3) 
  (h2 : clock.rings_per_day = 8) : 
  first_ring_time clock = ⟨3, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_first_ring_at_three_am_l58_5844


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l58_5860

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 7 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧ c^2 = 9 - 7

-- Define a point on the ellipse
def point_on_ellipse (A : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2

-- Define the angle condition
def angle_condition (A F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, Real.cos θ = Real.sqrt 2 / 2 ∧
    (A.1 - F₁.1) * (F₂.1 - A.1) + (A.2 - F₁.2) * (F₂.2 - A.2) =
    Real.cos θ * Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) *
                  Real.sqrt ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2)

-- Theorem statement
theorem ellipse_triangle_area
  (F₁ F₂ A : ℝ × ℝ)
  (h_foci : foci F₁ F₂)
  (h_point : point_on_ellipse A)
  (h_angle : angle_condition A F₁ F₂) :
  ∃ area : ℝ, area = 7 * Real.sqrt 5 / 2 ∧
    area = 1/2 * Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) *
                 Real.sqrt ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2) *
                 Real.sin (45 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l58_5860


namespace NUMINAMATH_CALUDE_school_fee_calculation_l58_5894

def mother_contribution : ℕ := 2 * 100 + 1 * 50 + 5 * 20 + 3 * 10 + 4 * 5
def father_contribution : ℕ := 3 * 100 + 4 * 50 + 2 * 20 + 1 * 10 + 6 * 5

theorem school_fee_calculation : mother_contribution + father_contribution = 980 := by
  sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l58_5894


namespace NUMINAMATH_CALUDE_hyperbola_equation_l58_5846

/-- Given a hyperbola with center at the origin and foci at (-√5, 0) and (√5, 0),
    prove that its equation is x²/4 - y² = 1 when a point P on the hyperbola
    forms a right-angled triangle PF₁F₂ with area 1. -/
theorem hyperbola_equation (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  F₁ = (-Real.sqrt 5, 0) →
  F₂ = (Real.sqrt 5, 0) →
  (∃ x y : ℝ, P = (x, y) ∧ x^2/4 - y^2 = 1) →
  (P.1 - F₁.1) * (P.2 - F₁.2) + (P.1 - F₂.1) * (P.2 - F₂.2) = 0 →
  abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = 1 →
  ∀ x y : ℝ, P = (x, y) → x^2/4 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l58_5846


namespace NUMINAMATH_CALUDE_rhombus_area_from_diagonals_l58_5867

/-- The area of a rhombus given its diagonals -/
theorem rhombus_area_from_diagonals (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  (1 / 2 : ℝ) * d1 * d2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_diagonals_l58_5867


namespace NUMINAMATH_CALUDE_not_perfect_square_l58_5852

theorem not_perfect_square : 
  (∃ x : ℕ, 6^3024 = x^2) ∧
  (∀ y : ℕ, 7^3025 ≠ y^2) ∧
  (∃ z : ℕ, 8^3026 = z^2) ∧
  (∃ w : ℕ, 9^3027 = w^2) ∧
  (∃ v : ℕ, 10^3028 = v^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l58_5852


namespace NUMINAMATH_CALUDE_quadratic_sequence_exists_smallest_n_for_specific_sequence_l58_5882

/-- A sequence is quadratic if the absolute difference between consecutive terms is the square of their index. -/
def IsQuadraticSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≥ 1 ∧ i ≤ n → |a i - a (i-1)| = i^2

theorem quadratic_sequence_exists (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ IsQuadraticSequence a n :=
sorry

theorem smallest_n_for_specific_sequence :
  (∃ (a : ℕ → ℤ), a 0 = 0 ∧ a 19 = 1996 ∧ IsQuadraticSequence a 19) ∧
  (∀ n : ℕ, n < 19 → ¬∃ (a : ℕ → ℤ), a 0 = 0 ∧ a n = 1996 ∧ IsQuadraticSequence a n) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_exists_smallest_n_for_specific_sequence_l58_5882


namespace NUMINAMATH_CALUDE_inconsistent_age_sum_l58_5853

theorem inconsistent_age_sum (total_students : ℕ) (class_avg_age : ℝ)
  (group1_size group2_size group3_size unknown_size : ℕ)
  (group1_avg_age group2_avg_age group3_avg_age : ℝ)
  (unknown_sum_age : ℝ) :
  total_students = 25 →
  class_avg_age = 18 →
  group1_size = 8 →
  group2_size = 10 →
  group3_size = 5 →
  unknown_size = 2 →
  group1_avg_age = 16 →
  group2_avg_age = 20 →
  group3_avg_age = 17 →
  unknown_sum_age = 35 →
  total_students = group1_size + group2_size + group3_size + unknown_size →
  ¬(class_avg_age * total_students =
    group1_avg_age * group1_size + group2_avg_age * group2_size +
    group3_avg_age * group3_size + unknown_sum_age) :=
by sorry

end NUMINAMATH_CALUDE_inconsistent_age_sum_l58_5853


namespace NUMINAMATH_CALUDE_hyperbola_range_l58_5872

-- Define the equation
def equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + (2 - m) * y^2 = 1

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  m ≠ 0 ∧ m ≠ 2 ∧ m * (2 - m) < 0

-- Theorem statement
theorem hyperbola_range (m : ℝ) :
  is_hyperbola m ↔ m < 0 ∨ m > 2 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_range_l58_5872


namespace NUMINAMATH_CALUDE_mikes_ride_length_mikes_ride_length_proof_l58_5845

/-- Proves that Mike's ride was 35 miles long given the taxi fare conditions -/
theorem mikes_ride_length : ℝ → Prop :=
  fun T =>
    let mike_start : ℝ := 3
    let mike_per_mile : ℝ := 0.3
    let mike_surcharge : ℝ := 1.5
    let annie_start : ℝ := 3.5
    let annie_per_mile : ℝ := 0.25
    let annie_toll : ℝ := 5
    let annie_surcharge : ℝ := 2
    let annie_miles : ℝ := 18
    ∀ M : ℝ,
      (mike_start + mike_per_mile * M + mike_surcharge = T) ∧
      (annie_start + annie_per_mile * annie_miles + annie_toll + annie_surcharge = T) →
      M = 35

/-- Proof of the theorem -/
theorem mikes_ride_length_proof : ∀ T : ℝ, mikes_ride_length T :=
  fun T => by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_mikes_ride_length_mikes_ride_length_proof_l58_5845


namespace NUMINAMATH_CALUDE_triangle_side_range_l58_5832

theorem triangle_side_range (p : ℝ) : 
  (∃ r s : ℝ, r * s = 4 * 26 ∧ 
              r^2 + p*r + 1 = 0 ∧ 
              s^2 + p*s + 1 = 0 ∧ 
              r > 0 ∧ s > 0 ∧
              r + s > 2 ∧ r + 2 > s ∧ s + 2 > r) →
  -2 * Real.sqrt 2 < p ∧ p < -2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l58_5832


namespace NUMINAMATH_CALUDE_negation_of_existence_equals_forall_not_equal_l58_5859

theorem negation_of_existence_equals_forall_not_equal (x : ℝ) :
  ¬(∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, x > 0 → Real.log x ≠ x - 1 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_equals_forall_not_equal_l58_5859


namespace NUMINAMATH_CALUDE_triangle_DEF_angle_F_l58_5816

theorem triangle_DEF_angle_F (D E F : Real) : 
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = Real.pi →
  2 * Real.sin D + 3 * Real.cos E = 3 →
  3 * Real.sin E + 5 * Real.cos D = 4 →
  Real.sin F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_DEF_angle_F_l58_5816


namespace NUMINAMATH_CALUDE_value_of_lg_ta_ratio_l58_5806

-- Define the necessary functions
noncomputable def sn (x : ℝ) : ℝ := Real.sin x
noncomputable def si (x : ℝ) : ℝ := Real.sin x
noncomputable def ta (x : ℝ) : ℝ := Real.tan x
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem value_of_lg_ta_ratio (α β : ℝ) 
  (h1 : sn (α + β) = 1/2) 
  (h2 : si (α - β) = 1/3) : 
  lg (5 * (ta α / ta β)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_lg_ta_ratio_l58_5806


namespace NUMINAMATH_CALUDE_roots_sum_relation_l58_5896

theorem roots_sum_relation (a b c d : ℝ) : 
  (∀ x, x^2 + c*x + d = 0 ↔ x = a ∨ x = b) → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_relation_l58_5896


namespace NUMINAMATH_CALUDE_book_cost_l58_5849

/-- If three identical books cost $45, then seven of these books cost $105. -/
theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) :
  7 * (cost_of_three / 3) = 105 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l58_5849


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l58_5851

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x^2 - 2*x - 8 ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l58_5851


namespace NUMINAMATH_CALUDE_super_ball_distance_l58_5875

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  -- Definition of total distance calculation
  sorry

/-- Theorem stating the total distance traveled by the ball -/
theorem super_ball_distance :
  let initialHeight : ℝ := 150
  let reboundRatio : ℝ := 2/3
  let bounces : ℕ := 5
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |totalDistance initialHeight reboundRatio bounces - 591.67| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_super_ball_distance_l58_5875


namespace NUMINAMATH_CALUDE_inequality_proof_l58_5885

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_condition : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l58_5885


namespace NUMINAMATH_CALUDE_intersection_max_difference_zero_l58_5834

-- Define the polynomial functions
def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

-- State the theorem
theorem intersection_max_difference_zero :
  (∀ x : ℝ, f x = g x → x = -1) →  -- Given condition: x = -1 is the only intersection
  (∃ x : ℝ, f x = g x) →           -- Ensure at least one intersection exists
  (∀ x y : ℝ, f x = g x ∧ f y = g y → |f x - f y| = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_max_difference_zero_l58_5834


namespace NUMINAMATH_CALUDE_expansion_theorem_l58_5800

-- Define the sum of binomial coefficients
def sum_binomial_coeff (m : ℝ) (n : ℕ) : ℝ := 2^n

-- Define the coefficient of x in the expansion
def coeff_x (m : ℝ) (n : ℕ) : ℝ := (n.choose 2) * m^2

theorem expansion_theorem (m : ℝ) (n : ℕ) (h_m : m > 0) 
  (h_sum : sum_binomial_coeff m n = 256)
  (h_coeff : coeff_x m n = 112) :
  n = 8 ∧ m = 2 ∧ 
  (Nat.choose 8 4 * 2^4 - Nat.choose 8 2 * 2^2 : ℝ) = 1008 :=
sorry

end NUMINAMATH_CALUDE_expansion_theorem_l58_5800


namespace NUMINAMATH_CALUDE_horner_method_correct_l58_5888

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 4x^2 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x - 4

theorem horner_method_correct :
  f 3 = horner [2, 3, 4, 5, -4] 3 ∧ horner [2, 3, 4, 5, -4] 3 = 290 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_correct_l58_5888


namespace NUMINAMATH_CALUDE_library_sunday_visitors_l58_5810

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_sunday_visitors
  (total_days : ℕ) 
  (non_sunday_visitors : ℕ) 
  (overall_average : ℕ) 
  (h1 : total_days = 30)
  (h2 : non_sunday_visitors = 240)
  (h3 : overall_average = 285) :
  let sundays : ℕ := total_days / 7 + 1
  let non_sundays : ℕ := total_days - sundays
  let sunday_visitors : ℕ := (overall_average * total_days - non_sunday_visitors * non_sundays) / sundays
  sunday_visitors = 510 := by
sorry

end NUMINAMATH_CALUDE_library_sunday_visitors_l58_5810


namespace NUMINAMATH_CALUDE_range_of_g_l58_5814

theorem range_of_g : ∀ x : ℝ, 
  (3/4 : ℝ) ≤ (Real.cos x)^4 + (Real.sin x)^2 ∧ 
  (Real.cos x)^4 + (Real.sin x)^2 ≤ 1 ∧
  ∃ y z : ℝ, (Real.cos y)^4 + (Real.sin y)^2 = (3/4 : ℝ) ∧
             (Real.cos z)^4 + (Real.sin z)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l58_5814


namespace NUMINAMATH_CALUDE_cube_surface_area_l58_5818

/-- Given a cube with volume 125 cubic cm, its surface area is 150 square cm. -/
theorem cube_surface_area (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 125 → 
  side_length ^ 3 = volume →
  surface_area = 6 * side_length ^ 2 →
  surface_area = 150 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l58_5818


namespace NUMINAMATH_CALUDE_solution_y_l58_5802

-- Define the function G
def G (a b c d : ℕ) : ℕ := a^b + c * d

-- Define the theorem
theorem solution_y : ∃ y : ℕ, G 3 y 6 15 = 300 ∧ 
  ∀ z : ℕ, G 3 z 6 15 = 300 → y = z :=
by
  sorry

end NUMINAMATH_CALUDE_solution_y_l58_5802


namespace NUMINAMATH_CALUDE_number_representation_and_addition_l58_5858

theorem number_representation_and_addition :
  (4090000 = 409 * 10000) ∧ (800000 + 5000 + 20 + 4 = 805024) := by
  sorry

end NUMINAMATH_CALUDE_number_representation_and_addition_l58_5858


namespace NUMINAMATH_CALUDE_quadratic_sum_real_roots_l58_5886

/-- A quadratic polynomial with positive leading coefficient and real roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  real_roots : b^2 - 4*a*c ≥ 0

/-- The sum of two QuadraticPolynomials -/
def add_poly (P Q : QuadraticPolynomial) : QuadraticPolynomial :=
  { a := P.a + Q.a,
    b := P.b + Q.b,
    c := P.c + Q.c,
    a_pos := by 
      apply add_pos P.a_pos Q.a_pos
    real_roots := sorry }

/-- Two QuadraticPolynomials have a common root -/
def have_common_root (P Q : QuadraticPolynomial) : Prop :=
  ∃ x : ℝ, P.a * x^2 + P.b * x + P.c = 0 ∧ Q.a * x^2 + Q.b * x + Q.c = 0

theorem quadratic_sum_real_roots (P₁ P₂ P₃ : QuadraticPolynomial)
  (h₁₂ : have_common_root P₁ P₂)
  (h₂₃ : have_common_root P₂ P₃)
  (h₁₃ : have_common_root P₁ P₃) :
  ∃ x : ℝ, (P₁.a + P₂.a + P₃.a) * x^2 + (P₁.b + P₂.b + P₃.b) * x + (P₁.c + P₂.c + P₃.c) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_real_roots_l58_5886


namespace NUMINAMATH_CALUDE_davids_weighted_average_l58_5835

-- Define the marks and weightages
def english_marks : ℝ := 96
def math_marks : ℝ := 95
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 97
def biology_marks : ℝ := 95

def english_weight : ℝ := 0.1
def math_weight : ℝ := 0.2
def physics_weight : ℝ := 0.3
def chemistry_weight : ℝ := 0.2
def biology_weight : ℝ := 0.2

-- Define the weighted average calculation
def weighted_average : ℝ :=
  english_marks * english_weight +
  math_marks * math_weight +
  physics_marks * physics_weight +
  chemistry_marks * chemistry_weight +
  biology_marks * biology_weight

-- Theorem statement
theorem davids_weighted_average :
  weighted_average = 91.6 := by sorry

end NUMINAMATH_CALUDE_davids_weighted_average_l58_5835


namespace NUMINAMATH_CALUDE_a_66_mod_55_l58_5837

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a_66 is congruent to 51 modulo 55 -/
theorem a_66_mod_55 : a 66 ≡ 51 [ZMOD 55] := by
  sorry

end NUMINAMATH_CALUDE_a_66_mod_55_l58_5837


namespace NUMINAMATH_CALUDE_survey_c_count_l58_5821

theorem survey_c_count (total_population : ℕ) (sample_size : ℕ) (first_number : ℕ) 
  (survey_c_start : ℕ) (survey_c_end : ℕ) : 
  total_population = 600 →
  sample_size = 50 →
  first_number = 3 →
  survey_c_start = 496 →
  survey_c_end = 600 →
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ sample_size ∧ 
    survey_c_start ≤ first_number + (total_population / sample_size) * (n - 1) ∧
    first_number + (total_population / sample_size) * (n - 1) ≤ survey_c_end) →
  (Finset.filter (λ n : ℕ => 
    n ≥ 1 ∧ n ≤ sample_size ∧ 
    survey_c_start ≤ first_number + (total_population / sample_size) * (n - 1) ∧
    first_number + (total_population / sample_size) * (n - 1) ≤ survey_c_end) 
    (Finset.range (sample_size + 1))).card = 8 :=
by sorry

end NUMINAMATH_CALUDE_survey_c_count_l58_5821


namespace NUMINAMATH_CALUDE_blueberry_earnings_relationship_l58_5895

/-- Represents the relationship between blueberry picking amount and earnings --/
def blueberry_earnings (x : ℝ) : ℝ × ℝ :=
  let y₁ := 60 + 30 * 0.6 * x
  let y₂ := 10 * 30 + 30 * 0.5 * (x - 10)
  (y₁, y₂)

/-- Theorem stating the relationship between y₁, y₂, and x when x > 10 --/
theorem blueberry_earnings_relationship (x : ℝ) (h : x > 10) :
  let (y₁, y₂) := blueberry_earnings x
  y₁ = 60 + 18 * x ∧ y₂ = 150 + 15 * x :=
by sorry

end NUMINAMATH_CALUDE_blueberry_earnings_relationship_l58_5895


namespace NUMINAMATH_CALUDE_sin_300_cos_0_l58_5803

theorem sin_300_cos_0 : Real.sin (300 * π / 180) * Real.cos 0 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_cos_0_l58_5803


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_equilateral_triangle_side_length_proof_l58_5850

/-- The length of one side of an equilateral triangle whose perimeter equals 
    the perimeter of a 125 cm × 115 cm rectangle is 160 cm. -/
theorem equilateral_triangle_side_length : ℝ → Prop :=
  λ side_length : ℝ =>
    let rectangle_width : ℝ := 125
    let rectangle_length : ℝ := 115
    let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_length)
    let triangle_perimeter : ℝ := 3 * side_length
    (triangle_perimeter = rectangle_perimeter) → (side_length = 160)

theorem equilateral_triangle_side_length_proof : 
  equilateral_triangle_side_length 160 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_equilateral_triangle_side_length_proof_l58_5850


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l58_5868

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 1 ∧ (4 / a₀) + (9 / b₀) = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l58_5868


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l58_5836

theorem consecutive_odd_integers_multiplier (x : ℤ) (m : ℚ) : 
  x + 4 = 15 →  -- Third integer is 15
  (∀ k : ℤ, x + 2*k ∈ {n : ℤ | n % 2 = 1}) →  -- All three are odd integers
  x * m = 2 * (x + 4) + 3 →  -- First integer times m equals 3 more than twice the third
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l58_5836


namespace NUMINAMATH_CALUDE_egg_count_problem_l58_5897

/-- Calculates the final number of eggs for a family given initial count and various changes --/
def final_egg_count (initial : ℕ) (mother_used : ℕ) (father_used : ℕ) 
  (chicken1_laid : ℕ) (chicken2_laid : ℕ) (chicken3_laid : ℕ) (child_took : ℕ) : ℕ :=
  initial - mother_used - father_used + chicken1_laid + chicken2_laid + chicken3_laid - child_took

/-- Theorem stating that given the specific values in the problem, the final egg count is 19 --/
theorem egg_count_problem : 
  final_egg_count 20 5 3 4 3 2 2 = 19 := by sorry

end NUMINAMATH_CALUDE_egg_count_problem_l58_5897


namespace NUMINAMATH_CALUDE_diplomats_speaking_french_l58_5843

theorem diplomats_speaking_french (T : ℕ) (F R B : ℕ) : 
  T = 70 →
  R = 38 →
  B = 7 →
  (T - F - R + B : ℤ) = 14 →
  F = 25 :=
by sorry

end NUMINAMATH_CALUDE_diplomats_speaking_french_l58_5843


namespace NUMINAMATH_CALUDE_floor_product_equals_45_l58_5804

theorem floor_product_equals_45 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 45 ↔ x ∈ Set.Ico 7.5 (46 / 6) :=
sorry

end NUMINAMATH_CALUDE_floor_product_equals_45_l58_5804


namespace NUMINAMATH_CALUDE_homework_problem_ratio_l58_5889

theorem homework_problem_ratio : 
  ∀ (total_problems : ℕ) 
    (martha_problems : ℕ) 
    (angela_problems : ℕ) 
    (jenna_problems : ℕ),
  total_problems = 20 →
  martha_problems = 2 →
  angela_problems = 9 →
  jenna_problems + martha_problems + (jenna_problems / 2) + angela_problems = total_problems →
  (jenna_problems : ℚ) / martha_problems = 3 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_ratio_l58_5889


namespace NUMINAMATH_CALUDE_count_numbers_divisible_by_291_l58_5825

theorem count_numbers_divisible_by_291 :
  let max_k : ℕ := 291000
  let is_valid : ℕ → Prop := λ k => k ≤ max_k ∧ (k^2 - 1) % 291 = 0
  (Finset.filter is_valid (Finset.range (max_k + 1))).card = 4000 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_divisible_by_291_l58_5825


namespace NUMINAMATH_CALUDE_binary_sum_exp_eq_four_l58_5838

/-- B(n) is the number of ones in the base two expression for the positive integer n -/
def B (n : ℕ+) : ℕ := sorry

/-- The infinite sum of B(n)/(n(n+1)) for n from 1 to infinity -/
noncomputable def infiniteSum : ℝ := sorry

theorem binary_sum_exp_eq_four :
  Real.exp infiniteSum = 4 := by sorry

end NUMINAMATH_CALUDE_binary_sum_exp_eq_four_l58_5838


namespace NUMINAMATH_CALUDE_triangle_circles_tangency_l58_5873

theorem triangle_circles_tangency (DE DF EF : ℝ) (R S : ℝ) :
  DE = 120 →
  DF = 120 →
  EF = 70 →
  R = 20 →
  S > 0 →
  S + R > EF / 2 →
  S < DE - R →
  (S + R)^2 + (S - R)^2 = ((130 - 4*S) / 3)^2 →
  S = 55 - 5 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circles_tangency_l58_5873


namespace NUMINAMATH_CALUDE_gold_ratio_l58_5841

theorem gold_ratio (total_gold : ℕ) (greg_gold : ℕ) (h1 : total_gold = 100) (h2 : greg_gold = 20) :
  greg_gold / (total_gold - greg_gold) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_gold_ratio_l58_5841


namespace NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l58_5812

theorem sum_of_real_roots_of_quartic (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^4 - 6*x^2 - 2*x - 1
  ∃ (r₁ r₂ : ℝ), (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l58_5812


namespace NUMINAMATH_CALUDE_percentage_calculation_l58_5855

theorem percentage_calculation (x : ℝ) : 
  (0.08 : ℝ) * x = (0.6 : ℝ) * ((0.3 : ℝ) * x) - (0.1 : ℝ) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l58_5855


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l58_5856

/-- A regular icosahedron -/
structure Icosahedron where
  vertices : Finset (Fin 12)
  edges : Finset (Fin 30)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  5 / 11

/-- Theorem: The probability of randomly selecting two vertices that form an edge
    in a regular icosahedron is 5/11 -/
theorem icosahedron_edge_probability (i : Icosahedron) :
  edge_probability i = 5 / 11 := by
  sorry


end NUMINAMATH_CALUDE_icosahedron_edge_probability_l58_5856


namespace NUMINAMATH_CALUDE_p_fourth_minus_one_divisible_by_ten_l58_5880

theorem p_fourth_minus_one_divisible_by_ten (p : ℕ) (hp : Prime p) (hp_not_two : p ≠ 2) (hp_not_five : p ≠ 5) :
  10 ∣ (p^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_p_fourth_minus_one_divisible_by_ten_l58_5880


namespace NUMINAMATH_CALUDE_sum_of_subtraction_equation_l58_5854

theorem sum_of_subtraction_equation :
  ∀ A B : ℕ,
    A ≠ B →
    A < 10 →
    B < 10 →
    (80 + A) - (10 * B + 2) = 45 →
    A + B = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_subtraction_equation_l58_5854


namespace NUMINAMATH_CALUDE_lisa_candy_consumption_l58_5869

/-- The number of candies Lisa eats on other days of the week -/
def candies_on_other_days (total_candies : ℕ) (candies_on_mon_wed : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  let total_days := days_per_week * num_weeks
  let mon_wed_days := 2 * num_weeks
  let other_days := total_days - mon_wed_days
  let candies_on_mon_wed_total := candies_on_mon_wed * mon_wed_days
  let remaining_candies := total_candies - candies_on_mon_wed_total
  (remaining_candies : ℚ) / other_days

theorem lisa_candy_consumption :
  candies_on_other_days 36 2 7 4 = 1 := by sorry

end NUMINAMATH_CALUDE_lisa_candy_consumption_l58_5869


namespace NUMINAMATH_CALUDE_prob_12th_last_value_l58_5805

/-- Probability of getting a different roll on a four-sided die -/
def p_different : ℚ := 3 / 4

/-- Probability of getting the same roll on a four-sided die -/
def p_same : ℚ := 1 / 4

/-- Number of rolls before the final roll -/
def n : ℕ := 11

/-- Probability of the 12th roll being the last roll -/
def prob_12th_last : ℚ := p_different ^ n * p_same

theorem prob_12th_last_value : 
  prob_12th_last = (3 ^ 10 : ℚ) / (4 ^ 11 : ℚ) := by sorry

end NUMINAMATH_CALUDE_prob_12th_last_value_l58_5805
