import Mathlib

namespace NUMINAMATH_CALUDE_number_of_trucks_l3841_384127

/-- The number of trucks used in transportation -/
def x : ℕ := sorry

/-- The total profit from Qingxi to Shenzhen in yuan -/
def total_profit : ℕ := 11560

/-- The profit per truck from Qingxi to Guangzhou in yuan -/
def profit_qingxi_guangzhou : ℕ := 480

/-- The initial profit per truck from Guangzhou to Shenzhen in yuan -/
def initial_profit_guangzhou_shenzhen : ℕ := 520

/-- The decrease in profit for each additional truck in yuan -/
def profit_decrease : ℕ := 20

/-- The profit from Guangzhou to Shenzhen as a function of the number of trucks -/
def profit_guangzhou_shenzhen (n : ℕ) : ℤ :=
  initial_profit_guangzhou_shenzhen * n - profit_decrease * (n - 1)

theorem number_of_trucks : x = 10 := by
  have h1 : profit_qingxi_guangzhou * x + profit_guangzhou_shenzhen x = total_profit := by sorry
  sorry

end NUMINAMATH_CALUDE_number_of_trucks_l3841_384127


namespace NUMINAMATH_CALUDE_philosophers_more_numerous_than_mathematicians_l3841_384140

theorem philosophers_more_numerous_than_mathematicians 
  (x : ℕ) -- x represents the number of people who are both mathematicians and philosophers
  (h_positive : x > 0) -- assumption that at least one person belongs to either group
  : 9 * x > 7 * x := by
  sorry

end NUMINAMATH_CALUDE_philosophers_more_numerous_than_mathematicians_l3841_384140


namespace NUMINAMATH_CALUDE_integer_solution_2017_l3841_384157

theorem integer_solution_2017 (x y z : ℤ) : 
  x + y + z + x*y + y*z + z*x + x*y*z = 2017 ↔ 
  ((x = 0 ∧ y = 1 ∧ z = 1008) ∨
   (x = 0 ∧ y = 1008 ∧ z = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = 1008) ∨
   (x = 1 ∧ y = 1008 ∧ z = 0) ∨
   (x = 1008 ∧ y = 0 ∧ z = 1) ∨
   (x = 1008 ∧ y = 1 ∧ z = 0)) :=
by sorry

#check integer_solution_2017

end NUMINAMATH_CALUDE_integer_solution_2017_l3841_384157


namespace NUMINAMATH_CALUDE_min_abs_z_plus_one_l3841_384150

theorem min_abs_z_plus_one (z : ℂ) (h : Complex.abs (z^2 + 1) = Complex.abs (z * (z + Complex.I))) :
  ∃ (w : ℂ), ∀ (z : ℂ), Complex.abs (z^2 + 1) = Complex.abs (z * (z + Complex.I)) →
    Complex.abs (w + 1) ≤ Complex.abs (z + 1) ∧ Complex.abs (w + 1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_one_l3841_384150


namespace NUMINAMATH_CALUDE_complex_modulus_l3841_384137

theorem complex_modulus (a b : ℝ) :
  (1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3841_384137


namespace NUMINAMATH_CALUDE_x_value_proof_l3841_384146

theorem x_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 * x^2 + 16 * x * y = 2 * x^3 + 4 * x^2 * y) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3841_384146


namespace NUMINAMATH_CALUDE_intersection_point_on_circle_l3841_384195

theorem intersection_point_on_circle (m : ℝ) :
  ∃ (x y r : ℝ),
    r > 0 ∧
    m * x + y + 2 * m = 0 ∧
    x - m * y + 2 * m = 0 ∧
    (x - 2)^2 + (y - 4)^2 = r^2 →
    2 * Real.sqrt 2 ≤ r ∧ r ≤ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_on_circle_l3841_384195


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3841_384165

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  a_eq : a = 6
  b_eq : b = 8
  c_eq : c = 10

/-- Square inscribed in the triangle with one vertex at the right angle -/
def inscribed_square_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the triangle with one side along the hypotenuse -/
def inscribed_square_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ y / t.c = (6/5 * y + 8/5 * y) / (t.a + t.b)

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_right_angle t x) (hy : inscribed_square_hypotenuse t y) : 
  x / y = 111 / 175 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3841_384165


namespace NUMINAMATH_CALUDE_neighbor_house_height_l3841_384122

/-- Given three houses where one is 80 feet tall, another is 70 feet tall,
    and the 80-foot house is 3 feet shorter than the average height of all three houses,
    prove that the height of the third house must be 99 feet. -/
theorem neighbor_house_height (h1 h2 h3 : ℝ) : 
  h1 = 80 → h2 = 70 → h1 = (h1 + h2 + h3) / 3 - 3 → h3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_house_height_l3841_384122


namespace NUMINAMATH_CALUDE_road_repair_fractions_l3841_384100

theorem road_repair_fractions (road_length : ℝ) (first_week_fraction second_week_fraction : ℚ) :
  road_length = 1500 →
  first_week_fraction = 5 / 17 →
  second_week_fraction = 4 / 17 →
  (first_week_fraction + second_week_fraction = 9 / 17) ∧
  (1 - (first_week_fraction + second_week_fraction) = 8 / 17) := by
  sorry

end NUMINAMATH_CALUDE_road_repair_fractions_l3841_384100


namespace NUMINAMATH_CALUDE_complement_of_M_l3841_384111

/-- The complement of set M in the real numbers -/
theorem complement_of_M (x : ℝ) :
  x ∈ (Set.univ : Set ℝ) \ {x : ℝ | x^2 - 4 ≤ 0} ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l3841_384111


namespace NUMINAMATH_CALUDE_cube_packing_surface_area_l3841_384102

/-- A rectangular box that can fit cubic products. -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The surface area of a box in square centimeters. -/
def surfaceArea (b : Box) : ℕ :=
  2 * (b.length * b.width + b.length * b.height + b.width * b.height)

/-- The volume of a box in cubic centimeters. -/
def volume (b : Box) : ℕ :=
  b.length * b.width * b.height

theorem cube_packing_surface_area :
  ∃ (b : Box), volume b = 12 ∧ (surfaceArea b = 40 ∨ surfaceArea b = 38 ∨ surfaceArea b = 32) := by
  sorry


end NUMINAMATH_CALUDE_cube_packing_surface_area_l3841_384102


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3841_384189

theorem polynomial_division_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (S : ℝ → ℝ),
    (∀ x, x^50 = (x^2 - 5*x + 6) * Q x + S x) ∧
    (∃ a b : ℝ, ∀ x, S x = a * x + b) ∧
    S x = (3^50 - 2^50) * x + (4^50 - 6^50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3841_384189


namespace NUMINAMATH_CALUDE_equation_implication_l3841_384176

theorem equation_implication (x y : ℝ) 
  (h1 : x^2 - 3*x*y + 2*y^2 + x - y = 0)
  (h2 : x^2 - 2*x*y + y^2 - 5*x + 2*y = 0) :
  x*y - 12*x + 15*y = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_implication_l3841_384176


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3841_384186

def solution_set : Set ℚ := {16/23, 17/23, 18/23, 19/23, 20/23, 21/23, 22/23, 1}

theorem floor_equation_solution (x : ℚ) :
  (⌊(20 : ℚ) * x + 23⌋ = 20 + 23 * x) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3841_384186


namespace NUMINAMATH_CALUDE_two_a_minus_three_b_value_l3841_384164

theorem two_a_minus_three_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 3) (h3 : b < a) :
  2 * a - 3 * b = 13 ∨ 2 * a - 3 * b = 5 :=
by sorry

end NUMINAMATH_CALUDE_two_a_minus_three_b_value_l3841_384164


namespace NUMINAMATH_CALUDE_calculation_proof_fractional_equation_solution_l3841_384119

-- Problem 1
theorem calculation_proof : (-2)^2 - Real.rpow 64 (1/3) + (-3)^0 - (1/3)^0 = 0 := by sorry

-- Problem 2
theorem fractional_equation_solution :
  let x : ℚ := 3/4
  (x / (x + 1) = 5 / (2*x + 2) - 1) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_fractional_equation_solution_l3841_384119


namespace NUMINAMATH_CALUDE_fifth_term_value_l3841_384145

theorem fifth_term_value (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : ∀ n, S n = 2 * n^2 + 3 * n - 1)
  (h2 : a 5 = S 5 - S 4) : 
  a 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3841_384145


namespace NUMINAMATH_CALUDE_restaurant_group_kids_l3841_384112

/-- Proves that in a group of 12 people, where adult meals cost $3 each and kids eat free,
    if the total cost is $15, then the number of kids in the group is 7. -/
theorem restaurant_group_kids (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 12)
  (h2 : adult_meal_cost = 3)
  (h3 : total_cost = 15) :
  total_people - (total_cost / adult_meal_cost) = 7 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_group_kids_l3841_384112


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l3841_384129

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeparallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular m α)
  (h3 : perpendicular n β) :
  planeparallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l3841_384129


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3841_384125

theorem circumscribed_sphere_surface_area (cube_volume : ℝ) (h : cube_volume = 27) :
  let cube_side := cube_volume ^ (1/3)
  let sphere_diameter := cube_side * Real.sqrt 3
  let sphere_radius := sphere_diameter / 2
  4 * Real.pi * sphere_radius ^ 2 = 27 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3841_384125


namespace NUMINAMATH_CALUDE_range_of_a_l3841_384185

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3841_384185


namespace NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l3841_384170

/-- Represents a square divided into a 5x5 grid -/
structure GridSquare where
  side_length : ℝ
  small_square_count : ℕ
  small_square_count_eq : small_square_count = 5

/-- Represents a triangle within the GridSquare -/
structure Triangle where
  grid : GridSquare
  covered_squares : ℝ
  covered_squares_eq : covered_squares = 3.5

theorem triangle_to_square_area_ratio 
  (grid : GridSquare) 
  (triangle : Triangle) 
  (h_triangle : triangle.grid = grid) :
  (triangle.covered_squares * (grid.side_length / grid.small_square_count)^2) / 
  (grid.side_length^2) = 7 / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l3841_384170


namespace NUMINAMATH_CALUDE_rocky_training_totals_l3841_384172

/-- Rocky's training schedule over three days -/
structure TrainingSchedule where
  initial_distance : ℝ
  initial_elevation : ℝ
  day2_distance_multiplier : ℝ
  day2_elevation_multiplier : ℝ
  day3_distance_multiplier : ℝ
  day3_elevation_multiplier : ℝ

/-- Calculate total distance and elevation gain over three days -/
def calculate_totals (schedule : TrainingSchedule) : ℝ × ℝ :=
  let day1_distance := schedule.initial_distance
  let day1_elevation := schedule.initial_elevation
  let day2_distance := day1_distance * schedule.day2_distance_multiplier
  let day2_elevation := day1_elevation * schedule.day2_elevation_multiplier
  let day3_distance := day2_distance * schedule.day3_distance_multiplier
  let day3_elevation := day2_elevation * schedule.day3_elevation_multiplier
  (day1_distance + day2_distance + day3_distance,
   day1_elevation + day2_elevation + day3_elevation)

/-- Theorem stating the total distance and elevation gain for Rocky's training -/
theorem rocky_training_totals :
  let schedule := TrainingSchedule.mk 4 100 2 1.5 4 2
  calculate_totals schedule = (44, 550) := by
  sorry

#eval calculate_totals (TrainingSchedule.mk 4 100 2 1.5 4 2)

end NUMINAMATH_CALUDE_rocky_training_totals_l3841_384172


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inv_x_l3841_384162

theorem max_value_of_x_plus_inv_x (x : ℝ) (h : 15 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 17 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inv_x_l3841_384162


namespace NUMINAMATH_CALUDE_solid_yellow_marbles_percentage_l3841_384193

theorem solid_yellow_marbles_percentage
  (total_marbles : ℝ)
  (solid_color_percentage : ℝ)
  (solid_color_not_yellow_percentage : ℝ)
  (h1 : solid_color_percentage = 90)
  (h2 : solid_color_not_yellow_percentage = 85)
  : (solid_color_percentage - solid_color_not_yellow_percentage) * total_marbles / 100 = 5 * total_marbles / 100 :=
by sorry

end NUMINAMATH_CALUDE_solid_yellow_marbles_percentage_l3841_384193


namespace NUMINAMATH_CALUDE_largest_proportional_part_l3841_384143

theorem largest_proportional_part (total : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a / b = 2 ∧ a / c = 3 →
  max (total * a / (a + b + c)) (max (total * b / (a + b + c)) (total * c / (a + b + c))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_proportional_part_l3841_384143


namespace NUMINAMATH_CALUDE_largest_consecutive_even_l3841_384184

theorem largest_consecutive_even : 
  ∀ (x : ℕ), 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) → 
  (x + 14 = 60) := by
sorry

end NUMINAMATH_CALUDE_largest_consecutive_even_l3841_384184


namespace NUMINAMATH_CALUDE_strawberry_pies_l3841_384142

def christine_strawberries : ℕ := 10
def rachel_strawberries : ℕ := 2 * christine_strawberries
def strawberries_per_pie : ℕ := 3

theorem strawberry_pies : 
  (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_pies_l3841_384142


namespace NUMINAMATH_CALUDE_program_flowchart_unique_start_end_l3841_384132

/-- Represents a chart with start and end points -/
structure Chart where
  start_points : ℕ
  end_points : ℕ

/-- Definition of a general flowchart -/
def is_flowchart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points ≥ 1

/-- Definition of a program flowchart -/
def is_program_flowchart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points = 1

/-- Definition of a structure chart (assumed equivalent to process chart) -/
def is_structure_chart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points ≥ 1

/-- Theorem stating that a program flowchart has exactly one start point and one end point -/
theorem program_flowchart_unique_start_end :
  ∀ c : Chart, is_program_flowchart c → c.start_points = 1 ∧ c.end_points = 1 := by
  sorry


end NUMINAMATH_CALUDE_program_flowchart_unique_start_end_l3841_384132


namespace NUMINAMATH_CALUDE_circle_chords_with_equal_sums_l3841_384174

/-- Given 2^500 points on a circle labeled 1 to 2^500, there exist 100 pairwise disjoint chords
    such that the sums of the labels at their endpoints are all equal. -/
theorem circle_chords_with_equal_sums :
  ∀ (labeling : Fin (2^500) → Fin (2^500)),
  ∃ (chords : Finset (Fin (2^500) × Fin (2^500))),
    (chords.card = 100) ∧
    (∀ (c1 c2 : Fin (2^500) × Fin (2^500)), c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
      (c1.1 ≠ c2.1 ∧ c1.1 ≠ c2.2 ∧ c1.2 ≠ c2.1 ∧ c1.2 ≠ c2.2)) ∧
    (∃ (sum : Nat), ∀ (c : Fin (2^500) × Fin (2^500)), c ∈ chords → 
      (labeling c.1).val + (labeling c.2).val = sum) :=
by sorry

end NUMINAMATH_CALUDE_circle_chords_with_equal_sums_l3841_384174


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3841_384108

def Q (n : ℕ) : ℚ := (2^(n-1) : ℚ) / (n.factorial * (2*n + 1))

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → k < 10 → Q k ≥ 1/5000 ∧ Q 10 < 1/5000 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3841_384108


namespace NUMINAMATH_CALUDE_conic_is_ellipse_iff_l3841_384169

/-- A conic section represented by the equation x^2 + 9y^2 - 6x + 27y = k --/
def conic (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 27*y = k

/-- Predicate for a non-degenerate ellipse --/
def is_nondegenerate_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧ 
    ∀ x y, f x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem conic_is_ellipse_iff (k : ℝ) :
  is_nondegenerate_ellipse (conic k) ↔ k > -117/4 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_iff_l3841_384169


namespace NUMINAMATH_CALUDE_population_reaches_target_in_2095_l3841_384133

/-- The initial population of the island -/
def initial_population : ℕ := 450

/-- The year when the population count starts -/
def initial_year : ℕ := 2020

/-- The number of years it takes for the population to triple -/
def tripling_period : ℕ := 25

/-- The target population we want to reach or exceed -/
def target_population : ℕ := 10800

/-- Function to calculate the population after a given number of periods -/
def population_after_periods (periods : ℕ) : ℕ :=
  initial_population * (3 ^ periods)

/-- Function to calculate the year after a given number of periods -/
def year_after_periods (periods : ℕ) : ℕ :=
  initial_year + (periods * tripling_period)

/-- Theorem stating that 2095 is the closest year to when the population reaches or exceeds the target -/
theorem population_reaches_target_in_2095 :
  ∃ (n : ℕ), 
    (population_after_periods n ≥ target_population) ∧
    (population_after_periods (n - 1) < target_population) ∧
    (year_after_periods n = 2095) :=
  sorry

end NUMINAMATH_CALUDE_population_reaches_target_in_2095_l3841_384133


namespace NUMINAMATH_CALUDE_g_negative_six_l3841_384194

def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

theorem g_negative_six (h : g 6 = 12) : g (-6) = -28 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_six_l3841_384194


namespace NUMINAMATH_CALUDE_percent_of_sixty_l3841_384126

theorem percent_of_sixty : (25 : ℚ) / 100 * 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_sixty_l3841_384126


namespace NUMINAMATH_CALUDE_peter_mowing_time_l3841_384155

/-- The time it takes Nancy to mow the yard alone (in hours) -/
def nancy_time : ℝ := 3

/-- The time it takes Nancy and Peter together to mow the yard (in hours) -/
def combined_time : ℝ := 1.71428571429

/-- The time it takes Peter to mow the yard alone (in hours) -/
def peter_time : ℝ := 4

/-- Theorem stating that given Nancy's time and the combined time, Peter's individual time is approximately 4 hours -/
theorem peter_mowing_time (ε : ℝ) (h_ε : ε > 0) :
  ∃ (t : ℝ), abs (t - peter_time) < ε ∧ 
  1 / nancy_time + 1 / t = 1 / combined_time :=
sorry


end NUMINAMATH_CALUDE_peter_mowing_time_l3841_384155


namespace NUMINAMATH_CALUDE_cubic_local_max_l3841_384118

/-- Given a cubic function with a local maximum, prove the product of two coefficients -/
theorem cubic_local_max (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧ 
  (f 1 = -3) →
  a * b = 9 := by
sorry

end NUMINAMATH_CALUDE_cubic_local_max_l3841_384118


namespace NUMINAMATH_CALUDE_dan_buys_five_dozens_l3841_384177

/-- The number of golf balls in one dozen -/
def balls_per_dozen : ℕ := 12

/-- The total number of golf balls purchased -/
def total_balls : ℕ := 132

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The number of golf balls Chris buys -/
def chris_balls : ℕ := 48

/-- Theorem stating that Dan buys 5 dozens of golf balls -/
theorem dan_buys_five_dozens :
  (total_balls - gus_dozens * balls_per_dozen - chris_balls) / balls_per_dozen = 5 :=
sorry

end NUMINAMATH_CALUDE_dan_buys_five_dozens_l3841_384177


namespace NUMINAMATH_CALUDE_min_sum_squares_l3841_384187

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3841_384187


namespace NUMINAMATH_CALUDE_geometric_sequence_50th_term_l3841_384197

/-- The 50th term of a geometric sequence with first term 8 and second term -16 -/
theorem geometric_sequence_50th_term :
  let a₁ : ℝ := 8
  let a₂ : ℝ := -16
  let r : ℝ := a₂ / a₁
  let aₙ (n : ℕ) : ℝ := a₁ * r^(n - 1)
  aₙ 50 = -8 * 2^49 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_50th_term_l3841_384197


namespace NUMINAMATH_CALUDE_normal_symmetry_l3841_384153

/-- A random variable with normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a normal random variable is less than or equal to a given value -/
def normalCDF (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

theorem normal_symmetry (X : NormalRandomVariable) (a : ℝ) :
  normalCDF X 0 = 1 - normalCDF X (a - 2) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_normal_symmetry_l3841_384153


namespace NUMINAMATH_CALUDE_point_lies_on_graph_l3841_384130

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define a point lying on the graph of a function
def LiesOnGraph (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Theorem statement
theorem point_lies_on_graph (f : ℝ → ℝ) (a : ℝ) 
  (h : EvenFunction f) : LiesOnGraph f (-a) (f a) := by
  sorry

end NUMINAMATH_CALUDE_point_lies_on_graph_l3841_384130


namespace NUMINAMATH_CALUDE_tan_five_pi_fourth_l3841_384178

theorem tan_five_pi_fourth : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourth_l3841_384178


namespace NUMINAMATH_CALUDE_population_growth_l3841_384159

/-- Given an initial population and two consecutive percentage increases,
    calculate the final population after both increases. -/
def final_population (initial : ℕ) (increase1 : ℚ) (increase2 : ℚ) : ℚ :=
  initial * (1 + increase1) * (1 + increase2)

/-- Theorem stating that the population after two years of growth is 1320. -/
theorem population_growth : final_population 1000 (1/10) (1/5) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l3841_384159


namespace NUMINAMATH_CALUDE_cylinder_cut_surface_area_l3841_384180

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- Represents the area of a flat surface created by cutting the cylinder -/
def cutSurfaceArea (c : RightCircularCylinder) (arcAngle : ℝ) : ℝ :=
  sorry

theorem cylinder_cut_surface_area :
  let c : RightCircularCylinder := { radius := 8, height := 10 }
  let arcAngle : ℝ := π / 2  -- 90 degrees in radians
  cutSurfaceArea c arcAngle = 40 * π - 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cut_surface_area_l3841_384180


namespace NUMINAMATH_CALUDE_convex_nonagon_diagonals_l3841_384135

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem convex_nonagon_diagonals : 
  nonagon_diagonals = 27 := by sorry

end NUMINAMATH_CALUDE_convex_nonagon_diagonals_l3841_384135


namespace NUMINAMATH_CALUDE_sequence_e_is_perfect_cube_l3841_384138

def sequence_a (n : ℕ) : ℕ := n

def sequence_b (n : ℕ) : ℕ :=
  if sequence_a n % 3 ≠ 0 then sequence_a n else 0

def sequence_c (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_b

def sequence_d (n : ℕ) : ℕ :=
  if sequence_c n % 3 ≠ 0 then sequence_c n else 0

def sequence_e (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_d

theorem sequence_e_is_perfect_cube (n : ℕ) :
  sequence_e n = ((n + 2) / 3)^3 := by sorry

end NUMINAMATH_CALUDE_sequence_e_is_perfect_cube_l3841_384138


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3841_384131

theorem fraction_subtraction : (18 : ℚ) / 42 - (3 : ℚ) / 8 = (3 : ℚ) / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3841_384131


namespace NUMINAMATH_CALUDE_ellipse_parameter_range_l3841_384161

/-- The equation of an ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (2 + m) + y^2 / (1 - m) = 1

/-- Conditions for the equation to represent an ellipse with foci on the x-axis -/
def is_valid_ellipse (m : ℝ) : Prop :=
  2 + m > 0 ∧ 1 - m > 0 ∧ 2 + m > 1 - m

/-- The range of m for which the equation represents a valid ellipse -/
theorem ellipse_parameter_range :
  ∀ m : ℝ, is_valid_ellipse m ↔ -1/2 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parameter_range_l3841_384161


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l3841_384154

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := fun (x y : ℝ) => x^2 - 6*x + y^2 - 8*y - 15 = 0
  let circle2 := fun (x y : ℝ) => x^2 + 10*x + y^2 + 12*y + 21 = 0
  ∃ d : ℝ, d = 2 * Real.sqrt 41 - Real.sqrt 97 ∧
    ∀ p q : ℝ × ℝ, circle1 p.1 p.2 → circle2 q.1 q.2 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l3841_384154


namespace NUMINAMATH_CALUDE_stating_inspection_probability_theorem_l3841_384179

/-- Represents the total number of items -/
def total_items : ℕ := 5

/-- Represents the number of defective items -/
def defective_items : ℕ := 2

/-- Represents the number of good items -/
def good_items : ℕ := 3

/-- Represents the number of inspections after which we want to calculate the probability -/
def target_inspections : ℕ := 4

/-- Represents the probability of the inspection stopping after exactly the target number of inspections -/
noncomputable def inspection_probability : ℚ := 3/5

/-- 
Theorem stating that the probability of the inspection stopping after exactly 
the target number of inspections is equal to the calculated probability
-/
theorem inspection_probability_theorem : 
  let p := inspection_probability
  p = (1 : ℚ) - (defective_items.choose 2 / total_items.choose 2) - 
      ((good_items.choose 3 + defective_items.choose 1 * good_items.choose 1 * (total_items - 3).choose 1) / total_items.choose 3) :=
by sorry

end NUMINAMATH_CALUDE_stating_inspection_probability_theorem_l3841_384179


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3841_384106

open Real

theorem least_positive_t_for_geometric_progression (α : ℝ) (h : 0 < α ∧ α < π / 2) :
  ∃ t : ℝ, t > 0 ∧
  (∀ r : ℝ, r > 0 →
    (arcsin (sin α) = r * α ∧
     arcsin (sin (3 * α)) = r^2 * α ∧
     arcsin (sin (5 * α)) = r^3 * α ∧
     arcsin (sin (t * α)) = r^4 * α)) ∧
  (∀ s : ℝ, s > 0 →
    (∃ r : ℝ, r > 0 ∧
      arcsin (sin α) = r * α ∧
      arcsin (sin (3 * α)) = r^2 * α ∧
      arcsin (sin (5 * α)) = r^3 * α ∧
      arcsin (sin (s * α)) = r^4 * α) →
    t ≤ s) ∧
  t = 3 * (π - 5 * α) / (π - 3 * α) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3841_384106


namespace NUMINAMATH_CALUDE_tangent_lines_parallel_to_given_line_l3841_384123

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- The slope of the line parallel to 4x - y = 1 -/
def m : ℝ := 4

theorem tangent_lines_parallel_to_given_line :
  ∃ (a b : ℝ), 
    (f' a = m) ∧ 
    (b = f a) ∧ 
    ((4*x - y = 0) ∨ (4*x - y - 4 = 0)) ∧
    (∀ x y : ℝ, y - b = m * (x - a) → y = f x) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_parallel_to_given_line_l3841_384123


namespace NUMINAMATH_CALUDE_nancy_home_economics_marks_l3841_384158

/-- Calculates the marks in Home Economics given the marks in other subjects and the average --/
def calculate_home_economics_marks (american_literature : ℕ) (history : ℕ) (physical_education : ℕ) (art : ℕ) (average : ℕ) : ℕ :=
  5 * average - (american_literature + history + physical_education + art)

/-- Theorem stating that Nancy's marks in Home Economics is 52 --/
theorem nancy_home_economics_marks :
  calculate_home_economics_marks 66 75 68 89 70 = 52 := by
  sorry

#eval calculate_home_economics_marks 66 75 68 89 70

end NUMINAMATH_CALUDE_nancy_home_economics_marks_l3841_384158


namespace NUMINAMATH_CALUDE_baker_pastries_cakes_difference_l3841_384110

theorem baker_pastries_cakes_difference (cakes pastries : ℕ) 
  (h1 : cakes = 19) 
  (h2 : pastries = 131) : 
  pastries - cakes = 112 := by
sorry

end NUMINAMATH_CALUDE_baker_pastries_cakes_difference_l3841_384110


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l3841_384124

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 900 →
  length - width = 45 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l3841_384124


namespace NUMINAMATH_CALUDE_survey_result_l3841_384188

theorem survey_result (total : ℕ) (migraines insomnia anxiety : ℕ)
  (migraines_insomnia migraines_anxiety insomnia_anxiety : ℕ)
  (all_three : ℕ) :
  total = 150 →
  migraines = 90 →
  insomnia = 60 →
  anxiety = 30 →
  migraines_insomnia = 20 →
  migraines_anxiety = 10 →
  insomnia_anxiety = 15 →
  all_three = 5 →
  total - (migraines + insomnia + anxiety - migraines_insomnia - migraines_anxiety - insomnia_anxiety + all_three) = 40 := by
  sorry

#check survey_result

end NUMINAMATH_CALUDE_survey_result_l3841_384188


namespace NUMINAMATH_CALUDE_allison_video_upload_ratio_l3841_384136

/-- Represents the problem of calculating the ratio of days Allison uploaded videos at her initial pace to the total days in June. -/
theorem allison_video_upload_ratio :
  ∀ (x y : ℕ), 
    x + y = 30 →  -- Total days in June
    10 * x + 20 * y = 450 →  -- Total video hours uploaded
    (x : ℚ) / 30 = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_allison_video_upload_ratio_l3841_384136


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3841_384144

theorem tan_alpha_value (α : Real) (h : Real.tan (π/4 - α) = 1/5) : Real.tan α = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3841_384144


namespace NUMINAMATH_CALUDE_algorithm_finite_results_l3841_384160

-- Define the properties of an algorithm
structure Algorithm where
  steps : ℕ
  inputs : ℕ
  deterministic : Bool
  unique_meaning : Bool
  definite : Bool
  finite : Bool
  orderly : Bool
  non_unique : Bool
  universal : Bool

-- Define the theorem
theorem algorithm_finite_results (a : Algorithm) : 
  a.definite → ¬(∃ (results : ℕ → Prop), (∀ n : ℕ, results n) ∧ (∀ m n : ℕ, m ≠ n → results m ≠ results n)) :=
by sorry

end NUMINAMATH_CALUDE_algorithm_finite_results_l3841_384160


namespace NUMINAMATH_CALUDE_nice_people_count_l3841_384149

/-- Represents the proportion of nice people for each name --/
def nice_proportion (name : String) : ℚ :=
  match name with
  | "Barry" => 1
  | "Kevin" => 1/2
  | "Julie" => 3/4
  | "Joe" => 1/10
  | _ => 0

/-- Represents the number of people with each name in the crowd --/
def crowd_count (name : String) : ℕ :=
  match name with
  | "Barry" => 24
  | "Kevin" => 20
  | "Julie" => 80
  | "Joe" => 50
  | _ => 0

/-- Calculates the number of nice people for a given name --/
def nice_count (name : String) : ℕ :=
  (nice_proportion name * crowd_count name).num.toNat

/-- The total number of nice people in the crowd --/
def total_nice_people : ℕ :=
  nice_count "Barry" + nice_count "Kevin" + nice_count "Julie" + nice_count "Joe"

/-- Theorem stating that the total number of nice people in the crowd is 99 --/
theorem nice_people_count : total_nice_people = 99 := by
  sorry

end NUMINAMATH_CALUDE_nice_people_count_l3841_384149


namespace NUMINAMATH_CALUDE_inequality_implies_k_range_l3841_384139

theorem inequality_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_k_range_l3841_384139


namespace NUMINAMATH_CALUDE_train_speed_ratio_l3841_384196

/-- Prove that the ratio of the speeds of two trains is 2:1 given specific conditions --/
theorem train_speed_ratio :
  let train_length : ℝ := 150  -- Length of each train in meters
  let crossing_time : ℝ := 8   -- Time taken to cross in seconds
  let faster_speed : ℝ := 90   -- Speed of faster train in km/h

  let total_distance : ℝ := 2 * train_length
  let relative_speed : ℝ := total_distance / crossing_time
  let faster_speed_ms : ℝ := faster_speed * 1000 / 3600
  let slower_speed_ms : ℝ := relative_speed - faster_speed_ms

  (faster_speed_ms / slower_speed_ms : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l3841_384196


namespace NUMINAMATH_CALUDE_q_investment_time_l3841_384141

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  time_p : ℚ

/-- Calculates the investment time for partner Q given the partnership data -/
def calculate_time_q (data : PartnershipData) : ℚ :=
  (data.profit_ratio_q * data.investment_ratio_p * data.time_p) / (data.profit_ratio_p * data.investment_ratio_q)

/-- Theorem stating that given the problem conditions, Q's investment time is 20 months -/
theorem q_investment_time (data : PartnershipData)
  (h1 : data.investment_ratio_p = 7)
  (h2 : data.investment_ratio_q = 5)
  (h3 : data.profit_ratio_p = 7)
  (h4 : data.profit_ratio_q = 10)
  (h5 : data.time_p = 10) :
  calculate_time_q data = 20 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_time_l3841_384141


namespace NUMINAMATH_CALUDE_bug_probability_after_six_steps_l3841_384116

/-- Represents a vertex of the tetrahedron -/
inductive Vertex : Type
| A : Vertex
| B : Vertex
| C : Vertex
| D : Vertex

/-- The probability of the bug being at a given vertex after n steps -/
def prob_at_vertex (v : Vertex) (n : ℕ) : ℚ :=
  sorry

/-- The probability of the bug choosing a non-opposite vertex -/
def prob_non_opposite : ℚ := 1/2

/-- The probability of the bug choosing the opposite vertex -/
def prob_opposite : ℚ := 1/6

/-- The edge length of the tetrahedron -/
def edge_length : ℝ := 1

theorem bug_probability_after_six_steps :
  prob_at_vertex Vertex.A 6 = 53/324 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_after_six_steps_l3841_384116


namespace NUMINAMATH_CALUDE_cindy_same_color_prob_l3841_384175

/-- Represents the number of marbles of each color in the box -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (mc : MarbleCount) : ℕ := mc.red + mc.green + mc.yellow

/-- Represents the number of marbles drawn by each person -/
structure DrawCounts where
  alice : ℕ
  bob : ℕ
  cindy : ℕ

/-- Calculates the probability of Cindy getting 3 marbles of the same color -/
noncomputable def probCindySameColor (mc : MarbleCount) (dc : DrawCounts) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem cindy_same_color_prob :
  let initial_marbles : MarbleCount := ⟨2, 2, 4⟩
  let draw_counts : DrawCounts := ⟨2, 3, 3⟩
  probCindySameColor initial_marbles draw_counts = 13 / 140 :=
sorry

end NUMINAMATH_CALUDE_cindy_same_color_prob_l3841_384175


namespace NUMINAMATH_CALUDE_circular_segment_area_l3841_384105

theorem circular_segment_area (r a : ℝ) (hr : r > 0) (ha : 0 < a ∧ a < 2*r) :
  let segment_area := r^2 * Real.arcsin (a / (2*r)) - (a/4) * Real.sqrt (4*r^2 - a^2)
  segment_area = r^2 * Real.arcsin (a / (2*r)) - (a/4) * Real.sqrt (4*r^2 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_circular_segment_area_l3841_384105


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3841_384198

/-- Given two vectors a and b in R², prove that if they are parallel,
    then the magnitude of a + 2b is 3√5. -/
theorem parallel_vectors_magnitude (t : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, t]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  ‖(a + 2 • b)‖ = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3841_384198


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l3841_384128

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y + Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y + Real.cos y / Real.sin x = 4) :
  Real.tan x / Real.tan y + Real.tan y / Real.tan x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l3841_384128


namespace NUMINAMATH_CALUDE_tangent_line_power_function_l3841_384190

theorem tangent_line_power_function (n : ℝ) :
  (2 : ℝ) ^ n = 8 →
  let f := λ x : ℝ => x ^ n
  let f' := λ x : ℝ => n * x ^ (n - 1)
  let tangent_slope := f' 2
  let tangent_eq := λ x y : ℝ => tangent_slope * (x - 2) = y - 8
  tangent_eq = λ x y : ℝ => 12 * x - y - 16 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_power_function_l3841_384190


namespace NUMINAMATH_CALUDE_sum4_equivalence_l3841_384163

-- Define the type for a die
def Die := Fin 6

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : Nat := d1.val + d2.val + 2

-- Define the event where the sum is 4
def sumIs4 (d1 d2 : Die) : Prop := diceSum d1 d2 = 4

-- Define the event where one die is 3 and the other is 1
def oneThreeOneOne (d1 d2 : Die) : Prop :=
  (d1.val = 2 ∧ d2.val = 0) ∨ (d1.val = 0 ∧ d2.val = 2)

-- Define the event where both dice show 2
def bothTwo (d1 d2 : Die) : Prop := d1.val = 1 ∧ d2.val = 1

-- Theorem stating the equivalence
theorem sum4_equivalence (d1 d2 : Die) :
  sumIs4 d1 d2 ↔ oneThreeOneOne d1 d2 ∨ bothTwo d1 d2 := by
  sorry


end NUMINAMATH_CALUDE_sum4_equivalence_l3841_384163


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3841_384114

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 →
  g = -a - c - e →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) + (g + h * Complex.I) = -2 * Complex.I →
  d + f + h = -4 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3841_384114


namespace NUMINAMATH_CALUDE_coin_distribution_rotations_l3841_384173

/-- Represents the coin distribution problem on a round table. -/
structure CoinDistribution where
  n : ℕ  -- number of sectors and players
  m : ℕ  -- number of rotations
  h_n_ge_4 : n ≥ 4

  /-- Player 1 received 74 fewer coins than player 4 -/
  h_player1_4 : ∃ (c1 c4 : ℕ), c4 - c1 = 74

  /-- Player 2 received 50 fewer coins than player 3 -/
  h_player2_3 : ∃ (c2 c3 : ℕ), c3 - c2 = 50

  /-- Player 4 received 3 coins twice as often as 2 coins -/
  h_player4_3_2 : ∃ (t2 t3 : ℕ), t3 = 2 * t2

  /-- Player 4 received 3 coins half as often as 1 coin -/
  h_player4_3_1 : ∃ (t1 t3 : ℕ), t3 = t1 / 2

/-- The number of rotations in the coin distribution problem is 69. -/
theorem coin_distribution_rotations (cd : CoinDistribution) : cd.m = 69 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_rotations_l3841_384173


namespace NUMINAMATH_CALUDE_chocolate_bar_weight_l3841_384115

/-- Proves that given a 2-kilogram box containing 16 chocolate bars, 
    each chocolate bar weighs 125 grams. -/
theorem chocolate_bar_weight :
  let box_weight_kg : ℕ := 2
  let bars_per_box : ℕ := 16
  let grams_per_kg : ℕ := 1000
  let box_weight_g : ℕ := box_weight_kg * grams_per_kg
  let bar_weight_g : ℕ := box_weight_g / bars_per_box
  bar_weight_g = 125 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_weight_l3841_384115


namespace NUMINAMATH_CALUDE_camille_bird_count_l3841_384104

/-- The number of birds Camille saw while bird watching -/
def total_birds (cardinals robins blue_jays sparrows : ℕ) : ℕ :=
  cardinals + robins + blue_jays + sparrows

/-- Theorem stating the total number of birds Camille saw -/
theorem camille_bird_count :
  ∃ (cardinals robins blue_jays sparrows : ℕ),
    cardinals = 3 ∧
    robins = 4 * cardinals ∧
    blue_jays = 2 * cardinals ∧
    sparrows = 3 * cardinals + 1 ∧
    total_birds cardinals robins blue_jays sparrows = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_camille_bird_count_l3841_384104


namespace NUMINAMATH_CALUDE_remainder_scaling_l3841_384183

theorem remainder_scaling (a b : ℕ) (c r : ℕ) (h : a = b * c + r) (hr : r = 7) :
  ∃ (c' : ℕ), 10 * a = 10 * b * c' + 70 :=
sorry

end NUMINAMATH_CALUDE_remainder_scaling_l3841_384183


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l3841_384147

theorem solution_implies_m_value (m : ℚ) :
  (∀ x : ℚ, (m - 2) * x = 5 * (x + 1) → x = 2) →
  m = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l3841_384147


namespace NUMINAMATH_CALUDE_banana_arrangements_l3841_384152

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- Theorem: The number of unique arrangements of "BANANA" is 60 -/
theorem banana_arrangements :
  uniqueArrangements 6 [3, 2, 1] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3841_384152


namespace NUMINAMATH_CALUDE_rancher_unique_solution_l3841_384192

/-- Represents the solution to the rancher's problem -/
structure RancherSolution where
  steers : ℕ
  cows : ℕ

/-- Checks if a given solution satisfies all conditions of the rancher's problem -/
def is_valid_solution (s : RancherSolution) : Prop :=
  s.steers > 0 ∧ 
  s.cows > 0 ∧ 
  30 * s.steers + 25 * s.cows = 1200

/-- Theorem stating that (5, 42) is the only valid solution to the rancher's problem -/
theorem rancher_unique_solution : 
  ∀ s : RancherSolution, is_valid_solution s ↔ s.steers = 5 ∧ s.cows = 42 := by
  sorry

#check rancher_unique_solution

end NUMINAMATH_CALUDE_rancher_unique_solution_l3841_384192


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_l3841_384168

/-- A triangle with sides 5, 12, and k is obtuse -/
def isObtuse (k : ℕ) : Prop :=
  (k > 5 ∧ k > 12 ∧ k^2 > 5^2 + 12^2) ∨
  (12 > 5 ∧ 12 > k ∧ 12^2 > 5^2 + k^2) ∨
  (5 > 12 ∧ 5 > k ∧ 5^2 > 12^2 + k^2)

/-- The triangle with sides 5, 12, and k is valid (satisfies triangle inequality) -/
def isValidTriangle (k : ℕ) : Prop :=
  k + 5 > 12 ∧ k + 12 > 5 ∧ 5 + 12 > k

theorem obtuse_triangle_count :
  ∃! (s : Finset ℕ), (∀ k ∈ s, k > 0 ∧ isValidTriangle k ∧ isObtuse k) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_l3841_384168


namespace NUMINAMATH_CALUDE_total_parents_is_fourteen_l3841_384134

/-- Represents the field trip to the zoo --/
structure FieldTrip where
  fifth_graders : ℕ
  sixth_graders : ℕ
  seventh_graders : ℕ
  teachers : ℕ
  buses : ℕ
  seats_per_bus : ℕ

/-- Calculates the total number of parents on the field trip --/
def total_parents (trip : FieldTrip) : ℕ :=
  trip.buses * trip.seats_per_bus - (trip.fifth_graders + trip.sixth_graders + trip.seventh_graders + trip.teachers)

/-- Theorem stating that the total number of parents on the trip is 14 --/
theorem total_parents_is_fourteen (trip : FieldTrip) 
  (h1 : trip.fifth_graders = 109)
  (h2 : trip.sixth_graders = 115)
  (h3 : trip.seventh_graders = 118)
  (h4 : trip.teachers = 4)
  (h5 : trip.buses = 5)
  (h6 : trip.seats_per_bus = 72) :
  total_parents trip = 14 := by
  sorry

#eval total_parents { fifth_graders := 109, sixth_graders := 115, seventh_graders := 118, teachers := 4, buses := 5, seats_per_bus := 72 }

end NUMINAMATH_CALUDE_total_parents_is_fourteen_l3841_384134


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_twice_sqrt_a_l3841_384171

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

theorem roots_sum_greater_than_twice_sqrt_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > Real.exp 1) 
  (hx₁ : f a x₁ = 0) 
  (hx₂ : f a x₂ = 0) 
  (hx_dist : x₁ ≠ x₂) : 
  x₁ + x₂ > 2 * Real.sqrt a := by
sorry

end NUMINAMATH_CALUDE_roots_sum_greater_than_twice_sqrt_a_l3841_384171


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l3841_384121

/-- Represents the number of aquariums of each type -/
structure AquariumCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the number of animals in each type of aquarium -/
structure AquariumAnimals where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total number of saltwater animals -/
def totalSaltwaterAnimals (counts : AquariumCounts) (animals : AquariumAnimals) : ℕ :=
  counts.typeA * animals.typeA + counts.typeB * animals.typeB + counts.typeC * animals.typeC

/-- Tyler's aquarium setup -/
def tylerAquariums : AquariumCounts :=
  { typeA := 10
    typeB := 14
    typeC := 6 }

/-- Number of animals in each type of Tyler's aquariums -/
def tylerAnimals : AquariumAnimals :=
  { typeA := 12 * 4  -- 12 corals with 4 animals each
    typeB := 18 + 10 -- 18 large fish and 10 small fish
    typeC := 25 + 20 -- 25 invertebrates and 20 small fish
  }

theorem tyler_saltwater_animals :
  totalSaltwaterAnimals tylerAquariums tylerAnimals = 1142 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l3841_384121


namespace NUMINAMATH_CALUDE_sin_minus_cos_eq_one_solution_set_l3841_384148

theorem sin_minus_cos_eq_one_solution_set :
  {x : ℝ | Real.sin (x / 2) - Real.cos (x / 2) = 1} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 ∨ x = k * Real.pi + Real.pi / 2} := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_eq_one_solution_set_l3841_384148


namespace NUMINAMATH_CALUDE_mary_books_checked_out_l3841_384107

/-- Calculates the number of books Mary has checked out after a series of transactions. -/
def books_checked_out (initial : ℕ) (first_return : ℕ) (first_checkout : ℕ) (second_return : ℕ) (second_checkout : ℕ) : ℕ :=
  initial - first_return + first_checkout - second_return + second_checkout

/-- Proves that Mary has 12 books checked out after the given transactions. -/
theorem mary_books_checked_out : 
  books_checked_out 5 3 5 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_books_checked_out_l3841_384107


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l3841_384182

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℝ := 10.0

/-- The number of yellow bouncy ball packs Maggie bought -/
def yellow_packs : ℝ := 8.0

/-- The number of green bouncy ball packs Maggie gave away -/
def green_packs_given : ℝ := 4.0

/-- The number of green bouncy ball packs Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℝ := yellow_packs * balls_per_pack + green_packs_bought * balls_per_pack - green_packs_given * balls_per_pack

theorem maggie_bouncy_balls : total_balls = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l3841_384182


namespace NUMINAMATH_CALUDE_class_test_percentages_l3841_384156

theorem class_test_percentages
  (percent_first : ℝ)
  (percent_second : ℝ)
  (percent_both : ℝ)
  (h1 : percent_first = 75)
  (h2 : percent_second = 35)
  (h3 : percent_both = 30) :
  100 - (percent_first + percent_second - percent_both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_percentages_l3841_384156


namespace NUMINAMATH_CALUDE_division_of_hundred_by_quarter_l3841_384167

theorem division_of_hundred_by_quarter : (100 : ℝ) / 0.25 = 400 := by
  sorry

end NUMINAMATH_CALUDE_division_of_hundred_by_quarter_l3841_384167


namespace NUMINAMATH_CALUDE_ram_ravi_selection_probability_l3841_384191

theorem ram_ravi_selection_probability :
  let p_ram : ℝ := 6/7
  let p_both : ℝ := 0.17142857142857143
  let p_ravi : ℝ := p_both / p_ram
  p_ravi = 0.2 := by sorry

end NUMINAMATH_CALUDE_ram_ravi_selection_probability_l3841_384191


namespace NUMINAMATH_CALUDE_fish_population_calculation_l3841_384103

/-- Calculates the number of fish in a lake on May 1 based on sampling data --/
theorem fish_population_calculation (tagged_may : ℕ) (caught_sept : ℕ) (tagged_sept : ℕ)
  (death_rate : ℚ) (new_fish_rate : ℚ) 
  (h1 : tagged_may = 60)
  (h2 : caught_sept = 70)
  (h3 : tagged_sept = 3)
  (h4 : death_rate = 1/4)
  (h5 : new_fish_rate = 2/5)
  (h6 : tagged_sept ≤ caught_sept) :
  ∃ (fish_may : ℕ), fish_may = 840 := by
  sorry

end NUMINAMATH_CALUDE_fish_population_calculation_l3841_384103


namespace NUMINAMATH_CALUDE_infinite_intersection_of_roots_l3841_384120

/-- The sequence S(x) defined as {⌊nx⌋ | n ∈ ℕ₊} -/
def S (x : ℝ) : Set ℕ := {n : ℕ | ∃ (m : ℕ+), n = ⌊m * x⌋}

/-- The polynomial f(x) = x³ - 10x² + 29x - 25 -/
def f (x : ℝ) : ℝ := x^3 - 10*x^2 + 29*x - 25

theorem infinite_intersection_of_roots :
  ∃ (α β : ℝ), α ≠ β ∧ f α = 0 ∧ f β = 0 ∧ Set.Infinite (S α ∩ S β) := by
  sorry

end NUMINAMATH_CALUDE_infinite_intersection_of_roots_l3841_384120


namespace NUMINAMATH_CALUDE_rhombus_properties_l3841_384166

/-- Properties of a rhombus with given area and one diagonal --/
theorem rhombus_properties (area : ℝ) (d1 : ℝ) (d2 : ℝ) (θ : ℝ) 
  (h1 : area = 432)
  (h2 : d1 = 36)
  (h3 : area = (d1 * d2) / 2)
  (h4 : θ = 2 * Real.arccos (2 / 3)) :
  d2 = 24 ∧ θ = 2 * Real.arccos (2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_rhombus_properties_l3841_384166


namespace NUMINAMATH_CALUDE_rope_cutting_l3841_384181

/-- Proves that a 200-meter rope cut into equal parts, with half given away and the rest subdivided,
    results in 25-meter pieces if and only if it was initially cut into 8 parts. -/
theorem rope_cutting (total_length : ℕ) (final_piece_length : ℕ) (initial_parts : ℕ) : 
  total_length = 200 ∧ 
  final_piece_length = 25 ∧
  (initial_parts : ℚ) * final_piece_length = total_length ∧
  (initial_parts / 2 : ℚ) * 2 * final_piece_length = total_length →
  initial_parts = 8 :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_l3841_384181


namespace NUMINAMATH_CALUDE_information_spread_time_l3841_384117

theorem information_spread_time (population : ℕ) (h : population = 1000000) :
  ∃ n : ℕ, n ≥ 19 ∧ 2^(n+1) - 1 ≥ population :=
by sorry

end NUMINAMATH_CALUDE_information_spread_time_l3841_384117


namespace NUMINAMATH_CALUDE_stating_average_enter_exit_time_l3841_384151

/-- Represents the speed of the car in miles per minute -/
def car_speed : ℚ := 5/4

/-- Represents the speed of the storm in miles per minute -/
def storm_speed : ℚ := 1/2

/-- Represents the radius of the storm in miles -/
def storm_radius : ℚ := 51

/-- Represents the initial y-coordinate of the storm center in miles -/
def initial_storm_y : ℚ := 110

/-- 
Theorem stating that the average time at which the car enters and exits the storm is 880/29 minutes
-/
theorem average_enter_exit_time : 
  let car_pos (t : ℚ) := (car_speed * t, 0)
  let storm_center (t : ℚ) := (0, initial_storm_y - storm_speed * t)
  let distance (t : ℚ) := 
    ((car_pos t).1 - (storm_center t).1)^2 + ((car_pos t).2 - (storm_center t).2)^2
  ∃ t₁ t₂,
    distance t₁ = storm_radius^2 ∧ 
    distance t₂ = storm_radius^2 ∧ 
    t₁ < t₂ ∧
    (t₁ + t₂) / 2 = 880 / 29 :=
sorry

end NUMINAMATH_CALUDE_stating_average_enter_exit_time_l3841_384151


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3841_384113

theorem right_triangle_hypotenuse : 
  ∀ (hypotenuse : ℝ), 
  hypotenuse > 0 →
  (hypotenuse - 1)^2 + 7^2 = hypotenuse^2 →
  hypotenuse = 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3841_384113


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l3841_384101

/-- Represents a city with its number of sales outlets -/
structure City where
  name : String
  outlets : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SystematicSampling
  | SimpleRandomSampling

/-- Represents an investigation with its requirements -/
structure Investigation where
  id : ℕ
  totalOutlets : ℕ
  sampleSize : ℕ
  cities : List City

/-- Determines the most appropriate sampling method for an investigation -/
def mostAppropriateMethod (inv : Investigation) : SamplingMethod := sorry

/-- The main theorem stating the appropriate sampling methods for the given investigations -/
theorem appropriate_sampling_methods 
  (cityA : City)
  (cityB : City)
  (cityC : City)
  (cityD : City)
  (inv1 : Investigation)
  (inv2 : Investigation)
  (h1 : cityA.outlets = 150)
  (h2 : cityB.outlets = 120)
  (h3 : cityC.outlets = 190)
  (h4 : cityD.outlets = 140)
  (h5 : inv1.totalOutlets = 600)
  (h6 : inv1.sampleSize = 100)
  (h7 : inv1.cities = [cityA, cityB, cityC, cityD])
  (h8 : inv2.totalOutlets = 20)
  (h9 : inv2.sampleSize = 8)
  (h10 : inv2.cities = [cityC]) :
  (mostAppropriateMethod inv1 = SamplingMethod.StratifiedSampling) ∧ 
  (mostAppropriateMethod inv2 = SamplingMethod.SimpleRandomSampling) := by
  sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l3841_384101


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3841_384199

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := -4 * x^2 + 5

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3841_384199


namespace NUMINAMATH_CALUDE_modified_chessboard_cannot_be_tiled_l3841_384109

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (black_squares : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the properties of a standard 8x8 chessboard with opposite corners removed -/
def standard_modified_chessboard : ModifiedChessboard :=
  { size := 8,
    total_squares := 62,
    white_squares := 32,
    black_squares := 30 }

/-- Defines the properties of a 1x2 domino -/
def standard_domino : Domino :=
  { length := 1,
    width := 2 }

/-- Checks if a chessboard can be tiled with dominoes -/
def can_be_tiled (board : ModifiedChessboard) (tile : Domino) : Prop :=
  board.white_squares = board.black_squares

/-- Theorem stating that the modified 8x8 chessboard cannot be tiled with 1x2 dominoes -/
theorem modified_chessboard_cannot_be_tiled :
  ¬(can_be_tiled standard_modified_chessboard standard_domino) :=
by
  sorry


end NUMINAMATH_CALUDE_modified_chessboard_cannot_be_tiled_l3841_384109
