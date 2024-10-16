import Mathlib

namespace NUMINAMATH_CALUDE_fourth_power_sqrt_equals_256_l588_58827

theorem fourth_power_sqrt_equals_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sqrt_equals_256_l588_58827


namespace NUMINAMATH_CALUDE_student_count_equality_l588_58850

/-- Proves that the number of students in class A equals the number in class C,
    given the average ages of each class and the overall average age. -/
theorem student_count_equality (a b c : ℕ) : 
  (14 * a + 13 * b + 12 * c : ℝ) / (a + b + c : ℝ) = 13 → a = c := by
  sorry

end NUMINAMATH_CALUDE_student_count_equality_l588_58850


namespace NUMINAMATH_CALUDE_zoo_lion_cubs_l588_58885

theorem zoo_lion_cubs (initial_animals : ℕ) (gorillas_sent : ℕ) (hippo_adopted : ℕ) (giraffes_adopted : ℕ) 
  (rhinos_added : ℕ) (crocodiles_added : ℕ) (final_animals : ℕ) :
  initial_animals = 150 →
  gorillas_sent = 12 →
  hippo_adopted = 1 →
  giraffes_adopted = 8 →
  rhinos_added = 4 →
  crocodiles_added = 5 →
  final_animals = 260 →
  ∃ (cubs : ℕ), 
    final_animals = initial_animals - gorillas_sent + hippo_adopted + giraffes_adopted + 
                    rhinos_added + crocodiles_added + cubs + 3 * cubs ∧
    cubs = 26 :=
by sorry

end NUMINAMATH_CALUDE_zoo_lion_cubs_l588_58885


namespace NUMINAMATH_CALUDE_problem_polygon_area_l588_58856

-- Define a point on a 2D grid
structure GridPoint where
  x : Int
  y : Int

-- Define a polygon as a list of grid points
def Polygon := List GridPoint

-- Function to calculate the area of a polygon given its vertices
def polygonArea (p : Polygon) : ℚ :=
  sorry

-- Define the specific polygon from the problem
def problemPolygon : Polygon := [
  ⟨0, 0⟩, ⟨20, 0⟩, ⟨30, 10⟩, ⟨30, 0⟩, ⟨40, 0⟩, ⟨40, 10⟩,
  ⟨40, 20⟩, ⟨30, 30⟩, ⟨20, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨0, 10⟩
]

-- Theorem statement
theorem problem_polygon_area :
  polygonArea problemPolygon = 15/2 := by sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l588_58856


namespace NUMINAMATH_CALUDE_range_of_m_l588_58835

def A : Set ℝ := {x | (x - 4) / (x + 3) ≤ 0}

def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 1}

theorem range_of_m : 
  ∀ m : ℝ, (A ∩ B m = B m) ↔ m ∈ Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l588_58835


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l588_58811

theorem rectangular_plot_length 
  (metallic_cost : ℝ) 
  (wooden_cost : ℝ) 
  (gate_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : metallic_cost = 26.5)
  (h2 : wooden_cost = 22)
  (h3 : gate_cost = 240)
  (h4 : total_cost = 5600) :
  ∃ (breadth length : ℝ),
    length = breadth + 14 ∧ 
    (2 * length + breadth) * metallic_cost + breadth * wooden_cost + gate_cost = total_cost ∧
    length = 59.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l588_58811


namespace NUMINAMATH_CALUDE_no_obtuse_tetrahedron_l588_58829

/-- Definition of a tetrahedron with obtuse angles -/
def ObtuseTetrahedron :=
  {t : Set (ℝ × ℝ × ℝ) | 
    (∃ v₁ v₂ v₃ v₄, t = {v₁, v₂, v₃, v₄}) ∧ 
    (∀ v ∈ t, ∀ α β γ, 
      (α + β + γ = 360) ∧ 
      (90 < α) ∧ (α < 180) ∧ 
      (90 < β) ∧ (β < 180) ∧ 
      (90 < γ) ∧ (γ < 180))}

/-- Theorem stating that an obtuse tetrahedron does not exist -/
theorem no_obtuse_tetrahedron : ¬ ∃ t : Set (ℝ × ℝ × ℝ), t ∈ ObtuseTetrahedron := by
  sorry

end NUMINAMATH_CALUDE_no_obtuse_tetrahedron_l588_58829


namespace NUMINAMATH_CALUDE_system_solution_l588_58864

theorem system_solution (x y z : ℝ) : 
  (x * (y^2 + z) = z * (z + x*y)) ∧ 
  (y * (z^2 + x) = x * (x + y*z)) ∧ 
  (z * (x^2 + y) = y * (y + x*z)) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l588_58864


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l588_58894

theorem regular_polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 45) :
  (360 / exterior_angle : ℝ) = 8 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l588_58894


namespace NUMINAMATH_CALUDE_A_disjoint_B_iff_l588_58839

/-- The set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

/-- The set B defined by the linear inequalities involving m -/
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

/-- Theorem stating the condition for A and B to be disjoint -/
theorem A_disjoint_B_iff (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_A_disjoint_B_iff_l588_58839


namespace NUMINAMATH_CALUDE_waldo_puzzles_per_book_l588_58887

theorem waldo_puzzles_per_book 
  (num_books : ℕ) 
  (minutes_per_puzzle : ℕ) 
  (total_minutes : ℕ) 
  (h1 : num_books = 15)
  (h2 : minutes_per_puzzle = 3)
  (h3 : total_minutes = 1350) :
  total_minutes / minutes_per_puzzle / num_books = 30 := by
  sorry

end NUMINAMATH_CALUDE_waldo_puzzles_per_book_l588_58887


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l588_58820

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l588_58820


namespace NUMINAMATH_CALUDE_component_scrap_probability_l588_58823

/-- The probability of a component passing the first inspection -/
def p_pass_first : ℝ := 0.8

/-- The probability of a component passing the second inspection, given it failed the first -/
def p_pass_second : ℝ := 0.9

/-- The probability of a component being scrapped -/
def p_scrapped : ℝ := (1 - p_pass_first) * (1 - p_pass_second)

theorem component_scrap_probability :
  p_scrapped = 0.02 :=
sorry

end NUMINAMATH_CALUDE_component_scrap_probability_l588_58823


namespace NUMINAMATH_CALUDE_congruence_problem_l588_58821

theorem congruence_problem (x : ℤ) : 
  (3 * x + 8) % 17 = 3 → (2 * x + 14) % 17 = 5 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l588_58821


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l588_58808

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + 2 * a 2 = 4 →
  a 4 ^ 2 = 4 * a 3 * a 7 →
  a 5 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l588_58808


namespace NUMINAMATH_CALUDE_whack_a_mole_tickets_value_l588_58845

/-- The number of tickets Ned won playing 'skee ball' -/
def skee_ball_tickets : ℕ := 19

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 9

/-- The number of candies Ned could buy -/
def candies_bought : ℕ := 5

/-- The number of tickets Ned won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := candy_cost * candies_bought - skee_ball_tickets

theorem whack_a_mole_tickets_value : whack_a_mole_tickets = 26 := by
  sorry

end NUMINAMATH_CALUDE_whack_a_mole_tickets_value_l588_58845


namespace NUMINAMATH_CALUDE_dress_price_difference_l588_58825

theorem dress_price_difference : 
  let original_price := 78.2 / 0.85
  let discounted_price := 78.2
  let final_price := discounted_price * 1.25
  final_price - original_price = 5.75 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l588_58825


namespace NUMINAMATH_CALUDE_bees_count_second_day_l588_58875

theorem bees_count_second_day (first_day_count : ℕ) (second_day_multiplier : ℕ) :
  first_day_count = 144 →
  second_day_multiplier = 3 →
  first_day_count * second_day_multiplier = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_bees_count_second_day_l588_58875


namespace NUMINAMATH_CALUDE_child_workers_count_l588_58833

/-- Represents the number of child workers employed by the contractor. -/
def num_child_workers : ℕ := 5

/-- Represents the number of male workers employed by the contractor. -/
def num_male_workers : ℕ := 20

/-- Represents the number of female workers employed by the contractor. -/
def num_female_workers : ℕ := 15

/-- Represents the daily wage of a male worker in rupees. -/
def male_wage : ℕ := 25

/-- Represents the daily wage of a female worker in rupees. -/
def female_wage : ℕ := 20

/-- Represents the daily wage of a child worker in rupees. -/
def child_wage : ℕ := 8

/-- Represents the average daily wage paid by the contractor in rupees. -/
def average_wage : ℕ := 21

/-- Theorem stating that the number of child workers is 5, given the conditions. -/
theorem child_workers_count :
  (num_male_workers * male_wage + num_female_workers * female_wage + num_child_workers * child_wage) / 
  (num_male_workers + num_female_workers + num_child_workers) = average_wage := by
  sorry

end NUMINAMATH_CALUDE_child_workers_count_l588_58833


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l588_58895

/-- The sum of the lengths of all sides of a rectangle with sides 9 cm and 11 cm is 40 cm. -/
theorem rectangle_perimeter (length width : ℝ) (h1 : length = 9) (h2 : width = 11) :
  2 * (length + width) = 40 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l588_58895


namespace NUMINAMATH_CALUDE_quadratic_function_domain_range_l588_58879

theorem quadratic_function_domain_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 m, ∃ y ∈ Set.Icc (-6) (-2), y = x^2 - 4*x - 2) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, y = x^2 - 4*x - 2) →
  m ∈ Set.Icc 2 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_domain_range_l588_58879


namespace NUMINAMATH_CALUDE_triangle_altitude_sum_square_l588_58873

theorem triangle_altitude_sum_square (a b c : ℕ) : 
  (∃ (h_a h_b h_c : ℝ), h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ 
    h_a = (2 * (a * h_a / 2)) / a ∧ 
    h_b = (2 * (b * h_b / 2)) / b ∧ 
    h_c = (2 * (c * h_c / 2)) / c ∧ 
    h_a = h_b + h_c) → 
  ∃ (k : ℚ), a^2 + b^2 + c^2 = k^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_sum_square_l588_58873


namespace NUMINAMATH_CALUDE_unique_divisor_square_equality_l588_58838

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- Theorem: The only positive integer n that satisfies n = [d(n)]^2 is 1 -/
theorem unique_divisor_square_equality :
  ∀ n : ℕ+, n.val = (num_divisors n)^2 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_square_equality_l588_58838


namespace NUMINAMATH_CALUDE_regression_change_l588_58863

/-- Represents a simple linear regression equation -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The change in the dependent variable when the independent variable increases by one unit -/
def change_in_y (reg : LinearRegression) : ℝ :=
  reg.intercept - reg.slope * (reg.intercept + 1) - (reg.intercept - reg.slope * reg.intercept)

theorem regression_change (reg : LinearRegression) 
  (h : reg.intercept = 2 ∧ reg.slope = 3) : 
  change_in_y reg = -3 := by
  sorry

#eval change_in_y { intercept := 2, slope := 3 }

end NUMINAMATH_CALUDE_regression_change_l588_58863


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l588_58801

theorem cube_root_equation_solution :
  ∃ (r s : ℕ) (x : ℝ),
    (x^(1/3) + (28 - x)^(1/3) = 2) ∧
    (∀ y : ℝ, y^(1/3) + (28 - y)^(1/3) = 2 → y ≤ x) ∧
    (x = r - Real.sqrt s) ∧
    (r + s = 188) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l588_58801


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l588_58851

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2010 :
  units_digit (factorial_sum 2010) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l588_58851


namespace NUMINAMATH_CALUDE_firewood_per_log_l588_58886

/-- Calculates the number of pieces of firewood per log -/
def piecesPerLog (totalPieces : ℕ) (totalTrees : ℕ) (logsPerTree : ℕ) : ℚ :=
  totalPieces / (totalTrees * logsPerTree)

theorem firewood_per_log :
  piecesPerLog 500 25 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_firewood_per_log_l588_58886


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l588_58858

noncomputable def f (x : ℝ) : ℝ := x - Real.exp x

theorem tangent_parallel_to_x_axis :
  ∃ (p : ℝ × ℝ), 
    (∀ x : ℝ, (p.2 = f p.1) ∧ 
    (HasDerivAt f 0 p.1)) →
    p = (0, -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l588_58858


namespace NUMINAMATH_CALUDE_g_2_4_neg1_eq_neg7_div_3_l588_58877

/-- The function g as defined in the problem -/
def g (a b c : ℚ) : ℚ := (a + b - c) / (a - b + c)

/-- Theorem stating that g(2, 4, -1) = -7/3 -/
theorem g_2_4_neg1_eq_neg7_div_3 : g 2 4 (-1) = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_g_2_4_neg1_eq_neg7_div_3_l588_58877


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_for_specific_configuration_l588_58862

/-- Configuration of three congruent circles -/
structure CircleConfiguration where
  radius : ℝ
  passes_through_centers : Prop

/-- Triangle formed by midpoints of arcs -/
structure MidpointTriangle where
  config : CircleConfiguration
  area : ℝ

/-- The main theorem -/
theorem midpoint_triangle_area_for_specific_configuration :
  ∀ (config : CircleConfiguration) (triangle : MidpointTriangle),
    config.radius = 2 ∧
    config.passes_through_centers ∧
    triangle.config = config →
    ∃ (a b : ℕ),
      triangle.area = Real.sqrt 3 ∧
      triangle.area = Real.sqrt a - b ∧
      100 * a + b = 300 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_for_specific_configuration_l588_58862


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l588_58831

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 4}

-- Define set N
def N : Set Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equals_set :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l588_58831


namespace NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l588_58868

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem largest_odd_digit_multiple_of_11 :
  ∀ n : ℕ,
    n < 10000 →
    has_only_odd_digits n →
    is_divisible_by_11 n →
    n ≤ 9559 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l588_58868


namespace NUMINAMATH_CALUDE_calvin_prevents_hobbes_win_l588_58830

/-- Represents a position on the integer lattice -/
structure Position where
  x : ℤ
  y : ℤ

/-- The game state -/
structure GameState where
  position : Position
  chosenIntegers : Set ℤ

/-- Calvin's strategy function -/
def calvinsStrategy (state : GameState) : Position := sorry

/-- Theorem stating Calvin can always prevent Hobbes from winning -/
theorem calvin_prevents_hobbes_win :
  ∀ (state : GameState),
  let newPos := calvinsStrategy state
  ∀ a b : ℤ,
    a ∉ state.chosenIntegers →
    b ∉ state.chosenIntegers →
    a ≠ (newPos.x - state.position.x) →
    b ≠ (newPos.y - state.position.y) →
    Position.mk (newPos.x + a) (newPos.y + b) ≠ Position.mk 0 0 :=
by sorry

end NUMINAMATH_CALUDE_calvin_prevents_hobbes_win_l588_58830


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l588_58890

theorem cube_volume_surface_area (x : ℝ) :
  (∃ s : ℝ, s > 0 ∧ s^3 = 27*x ∧ 6*s^2 = 3*x) → x = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l588_58890


namespace NUMINAMATH_CALUDE_equation_solution_l588_58834

theorem equation_solution :
  ∃ (a b c d : ℝ), 
    2 * a^2 + b^2 + 2 * c^2 + 2 = 3 * d + Real.sqrt (2 * a + b + 2 * c - 3 * d) ∧
    d = 2/3 ∧ a = 1/2 ∧ b = 1 ∧ c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l588_58834


namespace NUMINAMATH_CALUDE_car_endpoint_locus_l588_58836

-- Define the car's properties
structure Car where
  r₁ : ℝ
  r₂ : ℝ
  start : ℝ × ℝ
  s : ℝ
  α : ℝ

-- Define the circles W₁ and W₂
def W₁ (car : Car) (α₂ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (center : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = (2*(car.r₁ - car.r₂)*Real.sin α₂)^2}

def W₂ (car : Car) (α₁ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (center : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = (2*(car.r₁ - car.r₂)*Real.sin α₁)^2}

-- State the theorem
theorem car_endpoint_locus (car : Car) (α₁ α₂ : ℝ) 
  (h₁ : car.r₁ > car.r₂)
  (h₂ : car.α < 2 * Real.pi)
  (h₃ : α₁ + α₂ = car.α)
  (h₄ : car.r₁ * α₁ + car.r₂ * α₂ = car.s) :
  ∃ (endpoints : Set (ℝ × ℝ)), endpoints = W₁ car α₂ ∩ W₂ car α₁ := by
  sorry

end NUMINAMATH_CALUDE_car_endpoint_locus_l588_58836


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l588_58857

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l588_58857


namespace NUMINAMATH_CALUDE_sick_animals_count_l588_58812

/-- The number of chickens at Stacy's farm -/
def num_chickens : ℕ := 26

/-- The number of piglets at Stacy's farm -/
def num_piglets : ℕ := 40

/-- The number of goats at Stacy's farm -/
def num_goats : ℕ := 34

/-- The fraction of animals that get sick -/
def sick_fraction : ℚ := 1/2

/-- The total number of sick animals -/
def total_sick_animals : ℕ := (num_chickens + num_piglets + num_goats) / 2

theorem sick_animals_count : total_sick_animals = 50 := by
  sorry

end NUMINAMATH_CALUDE_sick_animals_count_l588_58812


namespace NUMINAMATH_CALUDE_locus_of_G_is_parabola_l588_58841

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Finds the intersection point of two lines -/
noncomputable def lineIntersection (l1 l2 : Line) : Point :=
  { x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b),
    y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b) }

/-- Theorem: Locus of point G forms a parabola -/
theorem locus_of_G_is_parabola (abc : RightTriangle) (d : Point) (ad ce ab : Line) :
  ∀ (e : Point), pointOnLine e ad →
  let f := lineIntersection ce ab
  let bc := Line.mk (abc.B.y - abc.C.y) (abc.C.x - abc.B.x) (abc.B.x * abc.C.y - abc.C.x * abc.B.y)
  let perpF := Line.mk bc.b (-bc.a) (-bc.b * f.x + bc.a * f.y)
  let be := Line.mk (e.y - abc.B.y) (abc.B.x - e.x) (e.x * abc.B.y - abc.B.x * e.y)
  let g := lineIntersection perpF be
  ∃ (a b : ℝ), g.y = (a / (b^2)) * (g.x - b)^2 := by
    sorry

end NUMINAMATH_CALUDE_locus_of_G_is_parabola_l588_58841


namespace NUMINAMATH_CALUDE_find_number_l588_58853

theorem find_number (x : ℝ) : x^2 * 15^2 / 356 = 51.193820224719104 → x = 9 ∨ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l588_58853


namespace NUMINAMATH_CALUDE_faire_percentage_calculation_dirk_faire_percentage_l588_58854

/-- Calculates the percentage of revenue given to the faire for Dirk's amulet sales --/
theorem faire_percentage_calculation (days : Nat) (amulets_per_day : Nat) 
  (selling_price : Nat) (cost_price : Nat) (final_profit : Nat) : ℚ :=
  let total_amulets := days * amulets_per_day
  let revenue := total_amulets * selling_price
  let total_cost := total_amulets * cost_price
  let profit_before_fee := revenue - total_cost
  let faire_fee := profit_before_fee - final_profit
  (faire_fee : ℚ) / revenue * 100

/-- Proves that Dirk gave 10% of his revenue to the faire --/
theorem dirk_faire_percentage : 
  faire_percentage_calculation 2 25 40 30 300 = 10 := by
  sorry

end NUMINAMATH_CALUDE_faire_percentage_calculation_dirk_faire_percentage_l588_58854


namespace NUMINAMATH_CALUDE_largest_n_for_product_2210_l588_58816

/-- An arithmetic sequence with integer terms -/
def ArithmeticSeq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_2210 :
  ∀ a b : ℕ → ℕ,
  ArithmeticSeq a → ArithmeticSeq b →
  a 1 = 1 → b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 2210) →
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 2210) → m ≤ 170) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2210_l588_58816


namespace NUMINAMATH_CALUDE_divisibility_by_441_l588_58861

theorem divisibility_by_441 (a b : ℕ) (h : 21 ∣ (a^2 + b^2)) : 441 ∣ (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_441_l588_58861


namespace NUMINAMATH_CALUDE_odd_function_property_l588_58804

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f (-1) = 2) :
  f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l588_58804


namespace NUMINAMATH_CALUDE_compound_propositions_l588_58892

-- Define the propositions p and q
def p : Prop := ∃ x : ℝ, x > 2 ∧ x > 1
def q : Prop := ∀ a b : ℝ, a > b → (1 / a) < (1 / b)

-- Define that p is sufficient but not necessary for x > 1
axiom p_sufficient : p → ∃ x : ℝ, x > 1
axiom p_not_necessary : ∃ x : ℝ, x > 1 ∧ ¬(x > 2)

-- Theorem stating that p ∧ ¬q is true, while other compounds are false
theorem compound_propositions :
  (p ∧ ¬q) ∧ ¬(p ∧ q) ∧ ¬(¬p ∨ q) ∧ ¬(¬p ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_compound_propositions_l588_58892


namespace NUMINAMATH_CALUDE_population_doubling_time_l588_58897

/-- The number of years required for a population to double given birth and death rates -/
theorem population_doubling_time (birth_rate death_rate : ℚ) : 
  birth_rate = 39.4 ∧ death_rate = 19.4 → 
  (70 : ℚ) / ((birth_rate - death_rate) / 10) = 35 := by
  sorry

end NUMINAMATH_CALUDE_population_doubling_time_l588_58897


namespace NUMINAMATH_CALUDE_congruence_problem_l588_58844

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) :
  ∃! n : ℤ, 200 ≤ n ∧ n ≤ 251 ∧ a - b ≡ n [ZMOD 60] ∧ n = 248 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l588_58844


namespace NUMINAMATH_CALUDE_x_equals_two_l588_58815

theorem x_equals_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 6 * x^3 + 12 * x * y^2 = 3 * x^2 * y + 3 * x^4) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_two_l588_58815


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l588_58884

/-- The sum of a geometric series with first term 1/4, common ratio 3/4, and 6 terms -/
def geometric_sum : ℚ :=
  let a : ℚ := 1/4
  let r : ℚ := 3/4
  let n : ℕ := 6
  a * (1 - r^n) / (1 - r)

/-- Theorem stating that the geometric sum equals 3367/4096 -/
theorem kevin_kangaroo_hops : geometric_sum = 3367/4096 := by
  sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l588_58884


namespace NUMINAMATH_CALUDE_square_area_theorem_l588_58800

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (topLeft : Point)
  (bottomRight : Point)

/-- Represents a square -/
structure Square :=
  (topLeft : Point)
  (sideLength : ℝ)

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.bottomRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomRight.y)

/-- Theorem: If a square is divided into four rectangles of equal area and MN = 3,
    then the area of the square is 64 -/
theorem square_area_theorem (s : Square) 
  (r1 r2 r3 r4 : Rectangle) 
  (h1 : rectangleArea r1 = rectangleArea r2)
  (h2 : rectangleArea r2 = rectangleArea r3)
  (h3 : rectangleArea r3 = rectangleArea r4)
  (h4 : r1.topLeft = s.topLeft)
  (h5 : r4.bottomRight.x = s.topLeft.x + s.sideLength)
  (h6 : r4.bottomRight.y = s.topLeft.y - s.sideLength)
  (h7 : r1.bottomRight.x - r1.topLeft.x = 3) : 
  s.sideLength * s.sideLength = 64 :=
sorry

end NUMINAMATH_CALUDE_square_area_theorem_l588_58800


namespace NUMINAMATH_CALUDE_microwave_sales_calculation_toaster_sales_calculation_l588_58817

/-- Represents the relationship between number of items sold and their cost --/
structure SalesCostRelation where
  items : ℕ  -- number of items sold
  cost : ℕ   -- cost of each item in dollars
  constant : ℕ -- the constant of proportionality

/-- Given a SalesCostRelation and a new cost, calculate the new number of items --/
def calculate_new_sales (scr : SalesCostRelation) (new_cost : ℕ) : ℚ :=
  scr.constant / new_cost

theorem microwave_sales_calculation 
  (microwave_initial : SalesCostRelation)
  (h_microwave_initial : microwave_initial.items = 10 ∧ microwave_initial.cost = 400)
  (h_microwave_constant : microwave_initial.constant = microwave_initial.items * microwave_initial.cost) :
  calculate_new_sales microwave_initial 800 = 5 := by sorry

theorem toaster_sales_calculation
  (toaster_initial : SalesCostRelation)
  (h_toaster_initial : toaster_initial.items = 6 ∧ toaster_initial.cost = 600)
  (h_toaster_constant : toaster_initial.constant = toaster_initial.items * toaster_initial.cost) :
  Int.floor (calculate_new_sales toaster_initial 1000) = 4 := by sorry

end NUMINAMATH_CALUDE_microwave_sales_calculation_toaster_sales_calculation_l588_58817


namespace NUMINAMATH_CALUDE_exists_motion_with_one_stationary_point_l588_58828

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A rigid body in 3D space -/
structure RigidBody where
  points : Set Point3D

/-- A motion of a rigid body -/
def Motion := RigidBody → ℝ → RigidBody

/-- A point is stationary under a motion if its position doesn't change over time -/
def IsStationary (p : Point3D) (m : Motion) (b : RigidBody) : Prop :=
  ∀ t : ℝ, p ∈ (m b t).points → p ∈ b.points

/-- A motion has exactly one stationary point -/
def HasExactlyOneStationaryPoint (m : Motion) (b : RigidBody) : Prop :=
  ∃! p : Point3D, IsStationary p m b ∧ p ∈ b.points

/-- Theorem: There exists a motion for a rigid body where exactly one point remains stationary -/
theorem exists_motion_with_one_stationary_point :
  ∃ (b : RigidBody) (m : Motion), HasExactlyOneStationaryPoint m b :=
sorry

end NUMINAMATH_CALUDE_exists_motion_with_one_stationary_point_l588_58828


namespace NUMINAMATH_CALUDE_inequalities_for_positive_sum_two_l588_58869

theorem inequalities_for_positive_sum_two (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_sum_two_l588_58869


namespace NUMINAMATH_CALUDE_student_arrangements_l588_58859

def num_male_students : ℕ := 4
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students

def arrangements_female_together : ℕ := sorry

def arrangements_no_adjacent_females : ℕ := sorry

def arrangements_ordered_females : ℕ := sorry

theorem student_arrangements :
  (arrangements_female_together = 720) ∧
  (arrangements_no_adjacent_females = 1440) ∧
  (arrangements_ordered_females = 840) := by sorry

end NUMINAMATH_CALUDE_student_arrangements_l588_58859


namespace NUMINAMATH_CALUDE_expected_disease_cases_l588_58846

theorem expected_disease_cases (total_sample : ℕ) (disease_proportion : ℚ) :
  total_sample = 300 →
  disease_proportion = 1 / 4 →
  (total_sample : ℚ) * disease_proportion = 75 := by
  sorry

end NUMINAMATH_CALUDE_expected_disease_cases_l588_58846


namespace NUMINAMATH_CALUDE_experiment_sequences_l588_58883

/-- The number of procedures in the experiment -/
def num_procedures : ℕ := 5

/-- Represents the possible positions for procedure A -/
inductive ProcedureAPosition
| First
| Last

/-- Represents a pair of adjacent procedures (C and D) -/
structure AdjacentPair where
  first : Fin num_procedures
  second : Fin num_procedures
  adjacent : first.val + 1 = second.val

/-- The total number of possible sequences in the experiment -/
def num_sequences : ℕ := 24

/-- Theorem stating the number of possible sequences in the experiment -/
theorem experiment_sequences :
  ∀ (a_pos : ProcedureAPosition) (cd_pair : AdjacentPair),
  num_sequences = 24 :=
sorry

end NUMINAMATH_CALUDE_experiment_sequences_l588_58883


namespace NUMINAMATH_CALUDE_common_intersection_point_l588_58870

-- Define a type for points in a plane
variable {Point : Type}

-- Define a type for half-planes
variable {HalfPlane : Type}

-- Define a function to check if a point is in a half-plane
variable (in_half_plane : Point → HalfPlane → Prop)

-- Define a set of half-planes
variable {S : Set HalfPlane}

-- Theorem statement
theorem common_intersection_point 
  (h : ∀ (a b c : HalfPlane), a ∈ S → b ∈ S → c ∈ S → 
    ∃ (p : Point), in_half_plane p a ∧ in_half_plane p b ∧ in_half_plane p c) :
  ∃ (p : Point), ∀ (h : HalfPlane), h ∈ S → in_half_plane p h :=
sorry

end NUMINAMATH_CALUDE_common_intersection_point_l588_58870


namespace NUMINAMATH_CALUDE_largest_value_l588_58832

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > max (1/2) (max (a^2 + b^2) (2*a*b)) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l588_58832


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l588_58802

/-- Calculates the total cost of John's purchase of soap and shampoo with discount and tax --/
theorem johns_purchase_cost :
  let soap_bars : ℕ := 20
  let soap_weight : ℝ := 1.5
  let soap_price : ℝ := 0.5
  let shampoo_bottles : ℕ := 15
  let shampoo_weight : ℝ := 2.2
  let shampoo_price : ℝ := 0.8
  let soap_discount : ℝ := 0.1
  let sales_tax : ℝ := 0.05
  
  let soap_cost := soap_bars * soap_weight * soap_price
  let discounted_soap_cost := soap_cost * (1 - soap_discount)
  let shampoo_cost := shampoo_bottles * shampoo_weight * shampoo_price
  let total_before_tax := discounted_soap_cost + shampoo_cost
  let tax_amount := total_before_tax * sales_tax
  let total_cost := total_before_tax + tax_amount

  total_cost = 41.90 := by sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l588_58802


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l588_58880

theorem lattice_points_on_hyperbola : 
  ∃! (s : Finset (ℤ × ℤ)), s.card = 4 ∧ ∀ (x y : ℤ), (x, y) ∈ s ↔ x^2 - y^2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l588_58880


namespace NUMINAMATH_CALUDE_S_100_equals_10100_l588_58818

/-- The number of integers in the solution set for x^2 - x < 2nx -/
def a (n : ℕ+) : ℕ := 2 * n

/-- The sum of the first n terms of the sequence {a_n} -/
def S (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that S_100 equals 10100 -/
theorem S_100_equals_10100 : S 100 = 10100 := by sorry

end NUMINAMATH_CALUDE_S_100_equals_10100_l588_58818


namespace NUMINAMATH_CALUDE_selection_ways_l588_58806

/-- The number of students in the group -/
def num_students : ℕ := 5

/-- The number of positions to be filled (representative and vice-president) -/
def num_positions : ℕ := 2

/-- Theorem: The number of ways to select one representative and one vice-president
    from a group of 5 students is equal to 20 -/
theorem selection_ways : (num_students * (num_students - 1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l588_58806


namespace NUMINAMATH_CALUDE_csc_negative_330_degrees_l588_58810

-- Define the cosecant function
noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

-- State the theorem
theorem csc_negative_330_degrees : csc ((-330 : Real) * Real.pi / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_csc_negative_330_degrees_l588_58810


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l588_58803

/-- A geometric sequence {a_n} with a_1 = 1 and a_5 = 9 has a_3 = 3 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 5 / a 1)^(1/4)) →  -- Geometric sequence condition
  a 1 = 1 →
  a 5 = 9 →
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l588_58803


namespace NUMINAMATH_CALUDE_percent_of_y_l588_58849

theorem percent_of_y (y : ℝ) (h : y > 0) : ((9 * y) / 20 + (3 * y) / 10) / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l588_58849


namespace NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l588_58805

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ ≥ 3 * Real.sqrt 2 :=
by sorry

theorem equality_condition (θ : Real) (h : θ = π / 4) :
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l588_58805


namespace NUMINAMATH_CALUDE_grandfather_wins_l588_58871

/-- The number of games played -/
def total_games : ℕ := 12

/-- Points scored by grandfather for each win -/
def grandfather_points : ℕ := 1

/-- Points scored by grandson for each win -/
def grandson_points : ℕ := 3

/-- Theorem stating the number of games won by the grandfather -/
theorem grandfather_wins (x : ℕ) 
  (h1 : x ≤ total_games)
  (h2 : x * grandfather_points = (total_games - x) * grandson_points) : 
  x = 9 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_wins_l588_58871


namespace NUMINAMATH_CALUDE_inverse_proposition_l588_58865

theorem inverse_proposition :
  (∀ x a b : ℝ, x ≥ a^2 + b^2 → x ≥ 2*a*b) →
  (∀ x a b : ℝ, x ≥ 2*a*b → x ≥ a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l588_58865


namespace NUMINAMATH_CALUDE_min_sum_squares_l588_58888

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (min : ℝ), min = 10 ∧ x^2 + y^2 + z^2 ≥ min ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l588_58888


namespace NUMINAMATH_CALUDE_total_swimming_time_l588_58822

/-- Represents the swimming times for various events -/
structure SwimmingTimes where
  freestyle : ℕ
  backstroke : ℕ
  butterfly : ℕ
  breaststroke : ℕ
  sidestroke : ℕ
  individual_medley : ℕ

/-- Calculates the total time for all events -/
def total_time (times : SwimmingTimes) : ℕ :=
  times.freestyle + times.backstroke + times.butterfly + 
  times.breaststroke + times.sidestroke + times.individual_medley

/-- Theorem stating the total time for all events -/
theorem total_swimming_time :
  ∀ (times : SwimmingTimes),
    times.freestyle = 48 →
    times.backstroke = times.freestyle + 4 + 2 →
    times.butterfly = times.backstroke + 3 + 3 →
    times.breaststroke = times.butterfly + 2 - 1 →
    times.sidestroke = times.butterfly + 5 + 4 →
    times.individual_medley = times.breaststroke + 6 + 3 →
    total_time times = 362 := by
  sorry

#eval total_time { freestyle := 48, backstroke := 54, butterfly := 60, 
                   breaststroke := 61, sidestroke := 69, individual_medley := 70 }

end NUMINAMATH_CALUDE_total_swimming_time_l588_58822


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l588_58893

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), n > 0 ∧ n = 6 ∧
  (∃ (p : ℕ), p < n ∧ Prime p ∧ Odd p ∧ (n^2 - n + 4) % p = 0) ∧
  (∃ (q : ℕ), q < n ∧ Prime q ∧ (n^2 - n + 4) % q ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    ¬((∃ (p : ℕ), p < m ∧ Prime p ∧ Odd p ∧ (m^2 - m + 4) % p = 0) ∧
      (∃ (q : ℕ), q < m ∧ Prime q ∧ (m^2 - m + 4) % q ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l588_58893


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l588_58813

theorem parabola_y_intercepts :
  let f (y : ℝ) := 3 * y^2 - 6 * y + 1
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ f y₁ = 0 ∧ f y₂ = 0 ∧ ∀ y, f y = 0 → y = y₁ ∨ y = y₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l588_58813


namespace NUMINAMATH_CALUDE_relay_race_average_time_l588_58867

/-- Calculates the average time for a leg of a relay race given the times of two runners. -/
def average_leg_time (y_time z_time : ℕ) : ℚ :=
  (y_time + z_time : ℚ) / 2

/-- Theorem stating that for the given runner times, the average leg time is 42 seconds. -/
theorem relay_race_average_time :
  average_leg_time 58 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_average_time_l588_58867


namespace NUMINAMATH_CALUDE_addition_subtraction_proof_l588_58847

theorem addition_subtraction_proof : 987 + 113 - 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_proof_l588_58847


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l588_58882

theorem sum_of_squares_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) ≥ 1/2 ∧
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1/2 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l588_58882


namespace NUMINAMATH_CALUDE_summer_break_length_l588_58899

/-- Represents the summer break reading scenario --/
structure SummerReading where
  deshaun_books : ℕ
  avg_pages_per_book : ℕ
  second_person_percentage : ℚ
  second_person_daily_pages : ℕ

/-- Calculates the number of days in the summer break --/
def summer_break_days (sr : SummerReading) : ℚ :=
  (sr.deshaun_books * sr.avg_pages_per_book * sr.second_person_percentage) / sr.second_person_daily_pages

/-- Theorem stating that the summer break is 80 days long --/
theorem summer_break_length (sr : SummerReading) 
  (h1 : sr.deshaun_books = 60)
  (h2 : sr.avg_pages_per_book = 320)
  (h3 : sr.second_person_percentage = 3/4)
  (h4 : sr.second_person_daily_pages = 180) :
  summer_break_days sr = 80 := by
  sorry

#eval summer_break_days { 
  deshaun_books := 60, 
  avg_pages_per_book := 320, 
  second_person_percentage := 3/4, 
  second_person_daily_pages := 180 
}

end NUMINAMATH_CALUDE_summer_break_length_l588_58899


namespace NUMINAMATH_CALUDE_bob_wins_for_S_l588_58860

/-- A set of lattice points in the Cartesian plane -/
def LatticeSet := Set (ℤ × ℤ)

/-- The set S defined by m and n -/
def S (m n : ℕ) : LatticeSet :=
  {p : ℤ × ℤ | m ≤ p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 ≤ n}

/-- Count of points on a line -/
def LineCount := ℕ

/-- Information provided by Alice: counts of points on horizontal, vertical, and diagonal lines -/
structure AliceInfo :=
  (horizontal : ℤ → LineCount)
  (vertical : ℤ → LineCount)
  (diagonalPos : ℤ → LineCount)  -- y = x + k
  (diagonalNeg : ℤ → LineCount)  -- y = -x + k

/-- Generate AliceInfo from a given set -/
def getAliceInfo (s : LatticeSet) : AliceInfo :=
  sorry

/-- Bob's winning condition -/
def BobCanWin (s : LatticeSet) : Prop :=
  ∀ t : LatticeSet, getAliceInfo s = getAliceInfo t → s = t

/-- Main theorem -/
theorem bob_wins_for_S (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  BobCanWin (S m n) :=
sorry

end NUMINAMATH_CALUDE_bob_wins_for_S_l588_58860


namespace NUMINAMATH_CALUDE_M_mod_100_l588_58891

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

def M : ℕ := trailingZeros (factorial 50)

theorem M_mod_100 : M % 100 = 12 := by sorry

end NUMINAMATH_CALUDE_M_mod_100_l588_58891


namespace NUMINAMATH_CALUDE_initial_speed_proof_l588_58826

/-- Proves that given the conditions of the journey, the initial speed must be 60 mph -/
theorem initial_speed_proof (v : ℝ) : 
  (v * 3 + 85 * 2) / 5 = 70 → v = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_proof_l588_58826


namespace NUMINAMATH_CALUDE_inequality_equivalence_l588_58814

theorem inequality_equivalence (x : ℝ) : 
  (|(x^2 - 9) / 3| < 3) ↔ (-Real.sqrt 18 < x ∧ x < Real.sqrt 18) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l588_58814


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_l588_58837

theorem sum_of_multiples_of_6_and_9 (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 3 ∣ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_l588_58837


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l588_58896

theorem hemisphere_surface_area (base_area : Real) (h : base_area = 225 * Real.pi) :
  let radius : Real := (base_area / Real.pi).sqrt
  let curved_surface_area : Real := 2 * Real.pi * radius^2
  let total_surface_area : Real := curved_surface_area + base_area
  total_surface_area = 675 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l588_58896


namespace NUMINAMATH_CALUDE_phil_card_loss_ratio_l588_58819

/-- Represents the number of baseball cards Phil has in various states --/
structure PhilCards where
  weekly_purchase : ℕ
  weeks_in_year : ℕ
  cards_left : ℕ

/-- Calculate the ratio of cards lost to total cards before the fire --/
def lost_to_total_ratio (p : PhilCards) : Rat :=
  let total := p.weekly_purchase * p.weeks_in_year
  let lost := total - p.cards_left
  lost / total

/-- Theorem stating the ratio of cards lost to total cards is 1:2 --/
theorem phil_card_loss_ratio :
  ∀ (p : PhilCards),
    p.weekly_purchase = 20 ∧
    p.weeks_in_year = 52 ∧
    p.cards_left = 520 →
    lost_to_total_ratio p = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_phil_card_loss_ratio_l588_58819


namespace NUMINAMATH_CALUDE_f_properties_l588_58809

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem f_properties :
  -- 1. f(x) is increasing on (-∞, -1) and (1, +∞)
  (∀ x y, (x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 1 ∧ y > 1))) → f x < f y) ∧
  -- 2. f(x) is decreasing on (-1, 1)
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- 3. The maximum value of f(x) on [-3, 2] is 2
  (∀ x, -3 ≤ x ∧ x ≤ 2 → f x ≤ 2) ∧
  (∃ x, -3 ≤ x ∧ x ≤ 2 ∧ f x = 2) ∧
  -- 4. The minimum value of f(x) on [-3, 2] is -18
  (∀ x, -3 ≤ x ∧ x ≤ 2 → f x ≥ -18) ∧
  (∃ x, -3 ≤ x ∧ x ≤ 2 ∧ f x = -18) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l588_58809


namespace NUMINAMATH_CALUDE_cans_recycled_from_64_l588_58898

def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 4 then 0
  else (initial_cans / 4) + recycle_cans (initial_cans / 4)

theorem cans_recycled_from_64 :
  recycle_cans 64 = 21 :=
sorry

end NUMINAMATH_CALUDE_cans_recycled_from_64_l588_58898


namespace NUMINAMATH_CALUDE_dans_stickers_l588_58881

theorem dans_stickers (bob_stickers : ℕ) (tom_stickers : ℕ) (dan_stickers : ℕ)
  (h1 : bob_stickers = 12)
  (h2 : tom_stickers = 3 * bob_stickers)
  (h3 : dan_stickers = 2 * tom_stickers) :
  dan_stickers = 72 := by
sorry

end NUMINAMATH_CALUDE_dans_stickers_l588_58881


namespace NUMINAMATH_CALUDE_second_derivative_at_pi_over_six_l588_58848

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem second_derivative_at_pi_over_six :
  (deriv^[2] f) (π / 6) = -(1 - Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_second_derivative_at_pi_over_six_l588_58848


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_120_l588_58866

theorem largest_multiple_of_8_less_than_120 :
  ∃ n : ℕ, n * 8 = 112 ∧ 
  112 < 120 ∧
  ∀ m : ℕ, m * 8 < 120 → m * 8 ≤ 112 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_120_l588_58866


namespace NUMINAMATH_CALUDE_trombone_players_l588_58824

/-- Represents the number of players for each instrument in an orchestra -/
structure Orchestra where
  total : Nat
  drummer : Nat
  trumpet : Nat
  frenchHorn : Nat
  violin : Nat
  cello : Nat
  contrabass : Nat
  clarinet : Nat
  flute : Nat
  maestro : Nat

/-- Theorem stating the number of trombone players in the orchestra -/
theorem trombone_players (o : Orchestra)
  (h1 : o.total = 21)
  (h2 : o.drummer = 1)
  (h3 : o.trumpet = 2)
  (h4 : o.frenchHorn = 1)
  (h5 : o.violin = 3)
  (h6 : o.cello = 1)
  (h7 : o.contrabass = 1)
  (h8 : o.clarinet = 3)
  (h9 : o.flute = 4)
  (h10 : o.maestro = 1) :
  o.total - (o.drummer + o.trumpet + o.frenchHorn + o.violin + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro) = 4 := by
  sorry


end NUMINAMATH_CALUDE_trombone_players_l588_58824


namespace NUMINAMATH_CALUDE_max_product_digits_l588_58874

theorem max_product_digits : ∀ a b : ℕ,
  10000 ≤ a ∧ a < 100000 →
  1000 ≤ b ∧ b < 10000 →
  a * b < 1000000000 := by
sorry

end NUMINAMATH_CALUDE_max_product_digits_l588_58874


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l588_58872

/-- The problem setup for the interest rate calculation --/
structure InterestProblem where
  total_sum : ℝ
  second_part : ℝ
  first_part : ℝ
  first_rate : ℝ
  first_time : ℝ
  second_time : ℝ
  second_rate : ℝ

/-- The interest rate calculation theorem --/
theorem interest_rate_calculation (p : InterestProblem)
  (h1 : p.total_sum = 2769)
  (h2 : p.second_part = 1704)
  (h3 : p.first_part = p.total_sum - p.second_part)
  (h4 : p.first_rate = 3 / 100)
  (h5 : p.first_time = 8)
  (h6 : p.second_time = 3)
  (h7 : p.first_part * p.first_rate * p.first_time = p.second_part * p.second_rate * p.second_time) :
  p.second_rate = 5 / 100 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l588_58872


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_l588_58878

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 6*x^3 + 9*x^2 + 6*x - 14

-- Theorem statement
theorem sum_of_abs_roots :
  ∃ (r₁ r₂ r₃ r₄ : ℝ),
    (∀ x : ℝ, p x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    |r₁| + |r₂| + |r₃| + |r₄| = 3 + Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_l588_58878


namespace NUMINAMATH_CALUDE_ambiguous_decomposition_l588_58876

def M : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 4 * k - 3}

def isSimple (n : ℕ) : Prop :=
  n ∈ M ∧ ∀ a b : ℕ, a ∈ M → b ∈ M → a * b = n → (a = 1 ∨ b = 1)

theorem ambiguous_decomposition : ∃ n : ℕ,
  n ∈ M ∧ (∃ a b c d : ℕ,
    isSimple a ∧ isSimple b ∧ isSimple c ∧ isSimple d ∧
    a * b = n ∧ c * d = n ∧ (a ≠ c ∨ b ≠ d)) :=
sorry

end NUMINAMATH_CALUDE_ambiguous_decomposition_l588_58876


namespace NUMINAMATH_CALUDE_circle_center_trajectory_equation_l588_58852

/-- The trajectory of the center of a circle passing through (4,0) and 
    intersecting the y-axis with a chord of length 8 -/
def circle_center_trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1 - 16}

/-- The fixed point through which the circle passes -/
def fixed_point : ℝ × ℝ := (4, 0)

/-- The length of the chord cut by the circle on the y-axis -/
def chord_length : ℝ := 8

/-- Theorem stating that the trajectory of the circle's center satisfies the given equation -/
theorem circle_center_trajectory_equation :
  ∀ (x y : ℝ), (x, y) ∈ circle_center_trajectory ↔ y^2 = 8*x - 16 :=
sorry

end NUMINAMATH_CALUDE_circle_center_trajectory_equation_l588_58852


namespace NUMINAMATH_CALUDE_jacket_purchase_price_l588_58842

/-- Calculates the purchase price of a jacket given the conditions of the problem -/
def calculate_purchase_price (selling_price : ℝ) : ℝ :=
  0.6 * selling_price

/-- Calculates the discounted price of a jacket given the selling price -/
def calculate_discounted_price (selling_price : ℝ) : ℝ :=
  0.8 * selling_price

/-- Calculates the gross profit given the selling price -/
def calculate_gross_profit (selling_price : ℝ) : ℝ :=
  calculate_discounted_price selling_price - calculate_purchase_price selling_price

theorem jacket_purchase_price :
  ∃ (selling_price : ℝ),
    calculate_gross_profit selling_price = 16 ∧
    calculate_purchase_price selling_price = 48 :=
by sorry

end NUMINAMATH_CALUDE_jacket_purchase_price_l588_58842


namespace NUMINAMATH_CALUDE_kylie_piggy_bank_coins_l588_58855

/-- The number of coins Kylie got from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie got from her brother -/
def brother_coins : ℕ := 13

/-- The number of coins Kylie got from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie gave to her friend Laura -/
def coins_given_away : ℕ := 21

/-- The number of coins Kylie was left with -/
def coins_left : ℕ := 15

/-- Theorem stating that the number of coins Kylie got from her piggy bank is 15 -/
theorem kylie_piggy_bank_coins :
  piggy_bank_coins = coins_left + coins_given_away - brother_coins - father_coins :=
by sorry

end NUMINAMATH_CALUDE_kylie_piggy_bank_coins_l588_58855


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l588_58889

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem bouncing_ball_distance :
  let initialHeight : ℝ := 80
  let reboundRatio : ℝ := 2/3
  let bounces : ℕ := 3
  totalDistance initialHeight reboundRatio bounces = 257.78 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l588_58889


namespace NUMINAMATH_CALUDE_product_of_roots_l588_58843

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = 24 → ∃ y : ℝ, (x + 2) * (x - 3) = 24 ∧ (y + 2) * (y - 3) = 24 ∧ x * y = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l588_58843


namespace NUMINAMATH_CALUDE_prob_two_black_one_red_standard_deck_l588_58840

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 26,
    black_cards := 26 }

/-- The probability of drawing two black cards followed by a red card -/
def prob_two_black_one_red (d : Deck) : ℚ :=
  (d.black_cards * (d.black_cards - 1) * d.red_cards) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability for a standard deck -/
theorem prob_two_black_one_red_standard_deck :
  prob_two_black_one_red standard_deck = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_black_one_red_standard_deck_l588_58840


namespace NUMINAMATH_CALUDE_max_min_sum_of_2x_minus_3y_l588_58807

theorem max_min_sum_of_2x_minus_3y : 
  ∀ x y : ℝ, 3 ≤ x → x ≤ 5 → 4 ≤ y → y ≤ 6 → 
  (∃ (max min : ℝ), 
    (∀ z : ℝ, z = 2*x - 3*y → z ≤ max) ∧
    (∃ x' y' : ℝ, 3 ≤ x' ∧ x' ≤ 5 ∧ 4 ≤ y' ∧ y' ≤ 6 ∧ 2*x' - 3*y' = max) ∧
    (∀ z : ℝ, z = 2*x - 3*y → min ≤ z) ∧
    (∃ x' y' : ℝ, 3 ≤ x' ∧ x' ≤ 5 ∧ 4 ≤ y' ∧ y' ≤ 6 ∧ 2*x' - 3*y' = min) ∧
    max + min = -14) :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_of_2x_minus_3y_l588_58807
