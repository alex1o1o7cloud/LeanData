import Mathlib

namespace NUMINAMATH_CALUDE_product_of_r_values_l3327_332748

theorem product_of_r_values : ∃ (r₁ r₂ : ℝ),
  (∀ r : ℝ, (∃! x : ℝ, (1 : ℝ) / (3 * x) = (r - x) / 6) ↔ (r = r₁ ∨ r = r₂)) ∧
  r₁ * r₂ = -8 :=
sorry

end NUMINAMATH_CALUDE_product_of_r_values_l3327_332748


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l3327_332795

/-- A moving circle in a plane passing through (-2, 0) and tangent to x = 2 -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_A : center.1 ^ 2 + center.2 ^ 2 = (-2 - center.1) ^ 2 + center.2 ^ 2
  tangent_to_line : (2 - center.1) ^ 2 + center.2 ^ 2 = (2 - (-2)) ^ 2

/-- The trajectory of the center of the moving circle -/
def trajectory_equation (x y : ℝ) : Prop :=
  y ^ 2 = -8 * x

theorem moving_circle_trajectory :
  ∀ (c : MovingCircle), trajectory_equation c.center.1 c.center.2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l3327_332795


namespace NUMINAMATH_CALUDE_initial_cookies_count_l3327_332716

def cookies_eaten : ℕ := 15
def cookies_left : ℕ := 78

theorem initial_cookies_count : 
  cookies_eaten + cookies_left = 93 := by sorry

end NUMINAMATH_CALUDE_initial_cookies_count_l3327_332716


namespace NUMINAMATH_CALUDE_lights_remaining_on_l3327_332704

def total_lights : ℕ := 2013

def lights_on_after_switches (n : ℕ) : ℕ :=
  n - (n / 2 + n / 3 + n / 5 - n / 6 - n / 10 - n / 15 + n / 30)

theorem lights_remaining_on :
  lights_on_after_switches total_lights = 1006 := by
  sorry

end NUMINAMATH_CALUDE_lights_remaining_on_l3327_332704


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l3327_332721

theorem quadratic_root_m_value : ∀ m : ℝ,
  ((-1 : ℝ)^2 + m * (-1) - 1 = 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l3327_332721


namespace NUMINAMATH_CALUDE_min_distance_hyperbola_circle_l3327_332709

theorem min_distance_hyperbola_circle (a b c d : ℝ) 
  (h1 : a * b = 1) (h2 : c^2 + d^2 = 1) : 
  ∃ (min : ℝ), min = 3 - 2 * Real.sqrt 2 ∧ 
  ∀ (x y z w : ℝ), x * y = 1 → z^2 + w^2 = 1 → 
  (x - z)^2 + (y - w)^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_distance_hyperbola_circle_l3327_332709


namespace NUMINAMATH_CALUDE_outer_wheel_speed_double_radii_difference_outer_wheel_circumference_l3327_332787

/-- Represents a car driving in a circle -/
structure CircularDrivingCar where
  inner_radius : ℝ
  outer_radius : ℝ
  wheel_distance : ℝ

/-- The properties of the car as described in the problem -/
def problem_car : CircularDrivingCar :=
  { inner_radius := 1.5,  -- This value is derived from the solution, not given directly
    outer_radius := 3,    -- This value is derived from the solution, not given directly
    wheel_distance := 1.5 }

/-- The theorem stating the relationship between the outer and inner wheel speeds -/
theorem outer_wheel_speed_double (car : CircularDrivingCar) :
  car.outer_radius = 2 * car.inner_radius :=
sorry

/-- The theorem stating the relationship between the radii and the wheel distance -/
theorem radii_difference (car : CircularDrivingCar) :
  car.outer_radius - car.inner_radius = car.wheel_distance :=
sorry

/-- The main theorem to prove -/
theorem outer_wheel_circumference (car : CircularDrivingCar) :
  2 * π * car.outer_radius = π * 6 :=
sorry

end NUMINAMATH_CALUDE_outer_wheel_speed_double_radii_difference_outer_wheel_circumference_l3327_332787


namespace NUMINAMATH_CALUDE_final_value_after_four_iterations_l3327_332743

def iterate_operation (x : ℕ) (s : ℕ) : ℕ := s * x + 1

def final_value (x : ℕ) (initial_s : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => initial_s
  | n + 1 => iterate_operation x (final_value x initial_s n)

theorem final_value_after_four_iterations :
  final_value 2 0 4 = 15 := by sorry

end NUMINAMATH_CALUDE_final_value_after_four_iterations_l3327_332743


namespace NUMINAMATH_CALUDE_cos_eight_degrees_l3327_332747

theorem cos_eight_degrees (m : ℝ) (h : Real.sin (74 * π / 180) = m) :
  Real.cos (8 * π / 180) = Real.sqrt ((1 + m) / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_degrees_l3327_332747


namespace NUMINAMATH_CALUDE_problem_solution_l3327_332784

theorem problem_solution : 2^2 + (-3)^2 - 1^2 + 4*2*(-3) = -12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3327_332784


namespace NUMINAMATH_CALUDE_water_amount_equals_sugar_amount_l3327_332780

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  flour : ℚ
  water : ℚ
  sugar : ℚ

/-- The original recipe ratio -/
def original_ratio : RecipeRatio := ⟨10, 6, 3⟩

/-- The new recipe ratio -/
def new_ratio : RecipeRatio := 
  let flour_water_doubled := original_ratio.flour / original_ratio.water * 2
  let flour_sugar_halved := original_ratio.flour / original_ratio.sugar / 2
  ⟨
    flour_water_doubled * original_ratio.water,
    original_ratio.water,
    flour_sugar_halved * original_ratio.sugar
  ⟩

/-- Amount of sugar in the new recipe -/
def sugar_amount : ℚ := 4

theorem water_amount_equals_sugar_amount : 
  (new_ratio.water / new_ratio.sugar) * sugar_amount = sugar_amount := by
  sorry

end NUMINAMATH_CALUDE_water_amount_equals_sugar_amount_l3327_332780


namespace NUMINAMATH_CALUDE_initial_pizzas_count_l3327_332726

/-- The number of pizzas returned by customers. -/
def returned_pizzas : ℕ := 6

/-- The number of pizzas successfully served to customers. -/
def served_pizzas : ℕ := 3

/-- The total number of pizzas initially served by the restaurant. -/
def total_pizzas : ℕ := returned_pizzas + served_pizzas

theorem initial_pizzas_count : total_pizzas = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_pizzas_count_l3327_332726


namespace NUMINAMATH_CALUDE_certain_number_is_80_l3327_332755

theorem certain_number_is_80 : 
  ∃ x : ℝ, (70 : ℝ) = 0.6 * x + 22 ∧ x = 80 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_80_l3327_332755


namespace NUMINAMATH_CALUDE_budget_allocation_l3327_332717

def budget : ℝ := 1000

def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def clothes_percentage : ℝ := 0.05

def coursework_percentage : ℝ :=
  1 - (food_percentage + accommodation_percentage + entertainment_percentage + transportation_percentage + clothes_percentage)

def combined_percentage : ℝ :=
  entertainment_percentage + transportation_percentage + coursework_percentage

def combined_amount : ℝ :=
  combined_percentage * budget

theorem budget_allocation :
  combined_percentage = 0.50 ∧ combined_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l3327_332717


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l3327_332720

/-- Given two lines that intersect at (3,3), prove that the sum of their y-intercepts is 4 -/
theorem intersection_y_intercept_sum (c d : ℝ) : 
  (3 = (1/3)*3 + c) → (3 = (1/3)*3 + d) → c + d = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l3327_332720


namespace NUMINAMATH_CALUDE_range_of_m_l3327_332723

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 4

-- Define proposition P
def P (m : ℝ) : Prop := ∀ x₁ x₂, 2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

-- Define proposition Q
def Q (m : ℝ) : Prop := ∀ x, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ≤ 1 ∨ (2 < m ∧ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3327_332723


namespace NUMINAMATH_CALUDE_sqrt_product_exists_l3327_332738

theorem sqrt_product_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ∃ x : ℝ, x^2 = a * b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_exists_l3327_332738


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3327_332794

theorem smallest_n_congruence : ∃! n : ℕ+, n.val = 20 ∧ 
  (∀ m : ℕ+, m.val < n.val → ¬(5 * m.val ≡ 1826 [ZMOD 26])) ∧
  (5 * n.val ≡ 1826 [ZMOD 26]) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3327_332794


namespace NUMINAMATH_CALUDE_problem_solution_l3327_332766

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 7) : 
  (x + y)^2 - x*y = 1183/36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3327_332766


namespace NUMINAMATH_CALUDE_triangle_area_inequalities_l3327_332764

/-- Given two triangles and a third triangle constructed from their sides, 
    the area of the third triangle is greater than or equal to 
    both the geometric and arithmetic means of the areas of the original triangles. -/
theorem triangle_area_inequalities 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h₂ : 0 < a₂ ∧ 0 < b₂ ∧ 0 < c₂)
  (h₃ : a₁ + b₁ > c₁ ∧ a₁ + c₁ > b₁ ∧ b₁ + c₁ > a₁)
  (h₄ : a₂ + b₂ > c₂ ∧ a₂ + c₂ > b₂ ∧ b₂ + c₂ > a₂)
  (a : ℝ) (ha : a = Real.sqrt ((a₁^2 + a₂^2) / 2))
  (b : ℝ) (hb : b = Real.sqrt ((b₁^2 + b₂^2) / 2))
  (c : ℝ) (hc : c = Real.sqrt ((c₁^2 + c₂^2) / 2))
  (h₅ : a + b > c ∧ a + c > b ∧ b + c > a)
  (S₁ : ℝ) (hS₁ : S₁ = Real.sqrt (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)))
  (S₂ : ℝ) (hS₂ : S₂ = Real.sqrt (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)))
  (S : ℝ)  (hS : S = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (s₁ : ℝ) (hs₁ : s₁ = (a₁ + b₁ + c₁) / 2)
  (s₂ : ℝ) (hs₂ : s₂ = (a₂ + b₂ + c₂) / 2)
  (s : ℝ)  (hs : s = (a + b + c) / 2) :
  S ≥ Real.sqrt (S₁ * S₂) ∧ S ≥ (S₁ + S₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequalities_l3327_332764


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3327_332707

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ x : ℕ, p x) ↔ (∀ x : ℕ, ¬ p x) := by sorry

theorem negation_of_proposition : 
  (¬ ∃ x : ℕ, x^2 ≥ x) ↔ (∀ x : ℕ, x^2 < x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3327_332707


namespace NUMINAMATH_CALUDE_tan_390_deg_l3327_332788

/-- Proves that the tangent of 390 degrees is equal to √3/3 -/
theorem tan_390_deg : Real.tan (390 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_390_deg_l3327_332788


namespace NUMINAMATH_CALUDE_fraction_simplification_l3327_332729

theorem fraction_simplification (a b m : ℝ) (h : a + b ≠ 0) :
  (m * a) / (a + b) + (m * b) / (a + b) = m := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3327_332729


namespace NUMINAMATH_CALUDE_isosceles_triangle_square_equal_area_l3327_332760

/-- 
Given an isosceles triangle with base s and height h, and a square with side length s,
if their areas are equal, then the height of the triangle is twice the side length of the square.
-/
theorem isosceles_triangle_square_equal_area (s h : ℝ) (s_pos : s > 0) :
  (1 / 2) * s * h = s^2 → h = 2 * s := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_square_equal_area_l3327_332760


namespace NUMINAMATH_CALUDE_dartboard_angle_for_quarter_probability_l3327_332769

/-- The central angle (in degrees) of a sector in a circular dartboard,
    given the probability of a dart landing in that sector -/
def central_angle (probability : ℝ) : ℝ :=
  360 * probability

/-- Theorem: If the probability of a dart landing in a particular region
    of a circular dartboard is 1/4, then the central angle of that region is 90 degrees -/
theorem dartboard_angle_for_quarter_probability :
  central_angle (1/4 : ℝ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_angle_for_quarter_probability_l3327_332769


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3327_332798

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 7 * x - 10 = 0 ↔ x = -2 ∨ x = 5/2) ∧ 
  (7^2 - 4 * k * (-10) > 0) ↔ 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3327_332798


namespace NUMINAMATH_CALUDE_bug_total_distance_l3327_332762

def bug_crawl (start : ℝ) (p1 p2 p3 : ℝ) : ℝ :=
  |p1 - start| + |p2 - p1| + |p3 - p2|

theorem bug_total_distance : bug_crawl 3 (-4) 6 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_bug_total_distance_l3327_332762


namespace NUMINAMATH_CALUDE_average_study_time_difference_l3327_332765

def asha_times : List ℝ := [40, 60, 50, 70, 30, 55, 45]
def sasha_times : List ℝ := [50, 70, 40, 100, 10, 55, 0]

theorem average_study_time_difference :
  (List.sum (List.zipWith (·-·) sasha_times asha_times)) / asha_times.length = -25 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l3327_332765


namespace NUMINAMATH_CALUDE_apps_added_minus_deleted_l3327_332712

theorem apps_added_minus_deleted (initial_apps added_apps final_apps : ℕ) :
  initial_apps = 115 →
  added_apps = 235 →
  final_apps = 178 →
  added_apps - (initial_apps + added_apps - final_apps) = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_apps_added_minus_deleted_l3327_332712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3327_332719

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a3 : a 3 = 12)
  (h_a6 : a 6 = 27) :
  a 10 = 47 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3327_332719


namespace NUMINAMATH_CALUDE_equation_solution_l3327_332797

theorem equation_solution (y : ℝ) : y + 81 / (y - 3) = -12 ↔ y = -6 ∨ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3327_332797


namespace NUMINAMATH_CALUDE_special_ellipse_intersecting_line_l3327_332773

/-- An ellipse with its upper vertex and left focus on a given line --/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 + y^2 / b^2 = 1
  vertex_focus_line : ℝ → ℝ → Prop := fun x y => x - y + 2 = 0

/-- A line intersecting the ellipse --/
structure IntersectingLine (E : SpecialEllipse) where
  l : ℝ → ℝ → Prop
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P : E.equation P.1 P.2 ∧ l P.1 P.2
  h_Q : E.equation Q.1 Q.2 ∧ l Q.1 Q.2
  h_midpoint : (P.1 + Q.1) / 2 = -1 ∧ (P.2 + Q.2) / 2 = 1

/-- The main theorem --/
theorem special_ellipse_intersecting_line 
  (E : SpecialEllipse) 
  (h_E : E.a^2 = 8 ∧ E.b^2 = 4) 
  (l : IntersectingLine E) : 
  l.l = fun x y => x - 2*y + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_intersecting_line_l3327_332773


namespace NUMINAMATH_CALUDE_legs_fraction_of_height_l3327_332710

/-- Represents the height measurements of a person --/
structure PersonHeight where
  total : ℝ
  head : ℝ
  restOfBody : ℝ

/-- Theorem stating the fraction of total height occupied by legs --/
theorem legs_fraction_of_height (p : PersonHeight) 
  (h_total : p.total = 60)
  (h_head : p.head = 1/4 * p.total)
  (h_rest : p.restOfBody = 25) :
  (p.total - p.head - p.restOfBody) / p.total = 1/3 := by
  sorry

#check legs_fraction_of_height

end NUMINAMATH_CALUDE_legs_fraction_of_height_l3327_332710


namespace NUMINAMATH_CALUDE_x_12_equals_439_l3327_332789

theorem x_12_equals_439 (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^12 = 439 := by
  sorry

end NUMINAMATH_CALUDE_x_12_equals_439_l3327_332789


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3327_332781

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    a * x₁^2 + (a + 2) * x₁ + 9 * a = 0 ∧
    a * x₂^2 + (a + 2) * x₂ + 9 * a = 0 ∧
    x₁ < 2 ∧ 2 < x₂) →
  -4/15 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3327_332781


namespace NUMINAMATH_CALUDE_pyramid_min_faces_l3327_332742

/-- A pyramid is a three-dimensional polyhedron with a polygonal base and triangular faces meeting at a common point (apex). -/
structure Pyramid where
  faces : ℕ

/-- Theorem: The number of faces in any pyramid is at least 4. -/
theorem pyramid_min_faces (p : Pyramid) : p.faces ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_min_faces_l3327_332742


namespace NUMINAMATH_CALUDE_point_in_region_l3327_332725

theorem point_in_region (m : ℝ) :
  m^2 - 3*m + 2 > 0 ↔ m < 1 ∨ m > 2 := by sorry

end NUMINAMATH_CALUDE_point_in_region_l3327_332725


namespace NUMINAMATH_CALUDE_min_value_theorem_l3327_332732

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 5) :
  9/x + 4/y + 25/z ≥ 20 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 5 ∧ 9/x' + 4/y' + 25/z' = 20 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3327_332732


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3327_332711

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 9)
  (h3 : crayons = 5)
  (h4 : both = 4) :
  total - (markers + crayons - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3327_332711


namespace NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l3327_332792

-- Define the concept of corresponding angles
def corresponding_angles (l1 l2 t : Line) (a1 a2 : Angle) : Prop :=
  -- We don't provide a specific definition, as it's not given in the problem
  sorry

-- Define the theorem
theorem corresponding_angles_not_always_equal :
  ¬ ∀ (l1 l2 t : Line) (a1 a2 : Angle),
    corresponding_angles l1 l2 t a1 a2 → a1 = a2 :=
by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l3327_332792


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3327_332746

/-- The speed of the current downstream in kilometers per hour -/
def current_speed : ℝ := 3

/-- The time taken to cover the distance downstream in seconds -/
def time_downstream : ℝ := 9.390553103577801

/-- The distance covered downstream in meters -/
def distance_downstream : ℝ := 60

/-- The speed at which the man can row in still water in kilometers per hour -/
def rowing_speed : ℝ := 20

theorem man_rowing_speed :
  rowing_speed = 
    (distance_downstream / 1000) / (time_downstream / 3600) - current_speed :=
by sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l3327_332746


namespace NUMINAMATH_CALUDE_ghost_paths_count_l3327_332786

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways a ghost can enter and exit the mansion -/
def ghost_paths : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that there are exactly 56 ways for a ghost to enter and exit the mansion -/
theorem ghost_paths_count : ghost_paths = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_paths_count_l3327_332786


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l3327_332776

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (8, 6)

theorem vectors_orthogonal : v1.1 * v2.1 + v1.2 * v2.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l3327_332776


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3327_332775

/-- The sum of the first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (m : ℕ) : ℕ := 5 * m - 20

theorem largest_of_five_consecutive_even : 
  ∃ m : ℕ, sum_first_n_even 30 = sum_five_consecutive_even m ∧ m = 190 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3327_332775


namespace NUMINAMATH_CALUDE_no_solution_greater_than_two_l3327_332791

theorem no_solution_greater_than_two (n : ℕ) (h : n > 2) :
  ¬ (3^(n-1) + 5^(n-1) ∣ 3^n + 5^n) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_greater_than_two_l3327_332791


namespace NUMINAMATH_CALUDE_elberta_money_l3327_332770

theorem elberta_money (granny_smith anjou elberta : ℕ) : 
  granny_smith = 120 →
  anjou = granny_smith / 4 →
  elberta = anjou + 5 →
  elberta = 35 := by
sorry

end NUMINAMATH_CALUDE_elberta_money_l3327_332770


namespace NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l3327_332737

theorem max_pairs_with_distinct_sums (n : ℕ) (hn : n = 2009) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 803 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs → p2 ∈ pairs → p1 ≠ p2 →
      p1.1 ≠ p2.1 ∧ p1.1 ≠ p2.2 ∧ p1.2 ≠ p2.1 ∧ p1.2 ≠ p2.2) ∧
    (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs → p2 ∈ pairs → p1 ≠ p2 →
      p1.1 + p1.2 ≠ p2.1 + p2.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (pairs' : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs' → p2 ∈ pairs' → p1 ≠ p2 →
        p1.1 ≠ p2.1 ∧ p1.1 ≠ p2.2 ∧ p1.2 ≠ p2.1 ∧ p1.2 ≠ p2.2) →
      (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs' → p2 ∈ pairs' → p1 ≠ p2 →
        p1.1 + p1.2 ≠ p2.1 + p2.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ n) →
      pairs'.card ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l3327_332737


namespace NUMINAMATH_CALUDE_equation_solutions_l3327_332736

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 - 9 = 0 ↔ x = -4 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - 12*x - 4 = 0 ↔ x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10) ∧
  (∀ x : ℝ, 3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3327_332736


namespace NUMINAMATH_CALUDE_classroom_area_less_than_hectare_l3327_332758

-- Define the area of 1 hectare in square meters
def hectare_area : ℝ := 10000

-- Define the typical area of a classroom in square meters
def typical_classroom_area : ℝ := 60

-- Theorem stating that a typical classroom area is much less than a hectare
theorem classroom_area_less_than_hectare :
  typical_classroom_area < hectare_area ∧ typical_classroom_area / hectare_area < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_classroom_area_less_than_hectare_l3327_332758


namespace NUMINAMATH_CALUDE_normal_binomial_properties_l3327_332740

/-- A random variable with normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ

/-- A random variable with binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ

/-- The probability that X is less than or equal to x -/
noncomputable def P (X : NormalRV) (x : ℝ) : ℝ := sorry

/-- The expected value of a normal random variable -/
noncomputable def E_normal (X : NormalRV) : ℝ := X.μ

/-- The expected value of a binomial random variable -/
noncomputable def E_binomial (Y : BinomialRV) : ℝ := Y.n * Y.p

/-- The variance of a binomial random variable -/
noncomputable def D_binomial (Y : BinomialRV) : ℝ := Y.n * Y.p * (1 - Y.p)

/-- The main theorem -/
theorem normal_binomial_properties (X : NormalRV) (Y : BinomialRV) 
    (h1 : P X 2 = 0.5)
    (h2 : E_binomial Y = E_normal X)
    (h3 : Y.n = 3) :
  X.μ = 2 ∧ Y.p = 2/3 ∧ 9 * D_binomial Y = 6 := by
  sorry

end NUMINAMATH_CALUDE_normal_binomial_properties_l3327_332740


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l3327_332771

theorem unique_congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 10389 [ZMOD 12] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l3327_332771


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l3327_332779

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) where
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (n : ℕ) (X : BinomialDistribution n) (k : ℕ) : ℝ :=
  (n.choose k) * X.p^k * (1 - X.p)^(n - k)

theorem binomial_probability_problem (X : BinomialDistribution 4) 
  (h3 : X.p < 1/2) 
  (h4 : binomialPMF 4 X 2 = 8/27) : 
  binomialPMF 4 X 1 = 32/81 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l3327_332779


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l3327_332796

/-- Given a rectangle ACDE with AC = 48 and AE = 30, where point B divides AC in ratio 1:3
    and point F divides AE in ratio 2:3, the area of quadrilateral ABDF is equal to 468. -/
theorem area_of_quadrilateral (AC AE : ℝ) (B F : ℝ) : 
  AC = 48 → 
  AE = 30 → 
  B / AC = 1 / 4 → 
  F / AE = 2 / 5 → 
  (AC * AE) - (3 * AC * AE / 4) - (3 * AC * AE / 5) = 468 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l3327_332796


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3327_332759

theorem quadratic_roots_problem (x₁ x₂ k : ℝ) : 
  (x₁^2 - 3*x₁ + k = 0) →
  (x₂^2 - 3*x₂ + k = 0) →
  (x₁ = 2*x₂) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3327_332759


namespace NUMINAMATH_CALUDE_division_of_decimals_l3327_332749

theorem division_of_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3327_332749


namespace NUMINAMATH_CALUDE_basketball_court_measurements_l3327_332728

theorem basketball_court_measurements :
  ∃! (A B C D E F : ℝ),
    A - B = C ∧
    D = 2 * (A + B) ∧
    E = A * B ∧
    F = 3 ∧
    A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0 ∧ F > 0 ∧
    ({A, B, C, D, E, F} : Set ℝ) = {86, 13, 420, 15, 28, 3} ∧
    A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 :=
by sorry

end NUMINAMATH_CALUDE_basketball_court_measurements_l3327_332728


namespace NUMINAMATH_CALUDE_new_cards_count_l3327_332739

def cards_per_page : ℕ := 3
def pages_used : ℕ := 6
def old_cards : ℕ := 10

theorem new_cards_count :
  (cards_per_page * pages_used) - old_cards = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_new_cards_count_l3327_332739


namespace NUMINAMATH_CALUDE_cubic_inequality_l3327_332774

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3327_332774


namespace NUMINAMATH_CALUDE_balls_in_boxes_count_l3327_332777

def num_balls : ℕ := 6
def num_boxes : ℕ := 3

theorem balls_in_boxes_count : 
  (num_boxes : ℕ) ^ (num_balls : ℕ) = 729 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_count_l3327_332777


namespace NUMINAMATH_CALUDE_max_min_on_interval_l3327_332793

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the interval
def interval : Set ℝ := Set.Icc 1 3

-- State the theorem
theorem max_min_on_interval :
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ interval → 1 ≤ f x ∧ f x ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l3327_332793


namespace NUMINAMATH_CALUDE_test_questions_count_l3327_332756

theorem test_questions_count (score : ℤ) (correct : ℤ) (incorrect : ℤ) :
  score = correct - 2 * incorrect →
  score = 61 →
  correct = 87 →
  correct + incorrect = 100 := by
sorry

end NUMINAMATH_CALUDE_test_questions_count_l3327_332756


namespace NUMINAMATH_CALUDE_log_xy_value_l3327_332733

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 3) :
  Real.log (x * y) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l3327_332733


namespace NUMINAMATH_CALUDE_inequality_proof_l3327_332783

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  x + 1 / (2 * y) > y + 1 / x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3327_332783


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_root_sum_power_6_l3327_332745

theorem smallest_integer_greater_than_root_sum_power_6 :
  ∃ n : ℕ, n = 970 ∧ (∀ m : ℤ, m > (Real.sqrt 3 + Real.sqrt 2)^6 → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_root_sum_power_6_l3327_332745


namespace NUMINAMATH_CALUDE_mushroom_collectors_problem_l3327_332722

theorem mushroom_collectors_problem :
  ∃ (n m : ℕ),
    n > 0 ∧ m > 0 ∧
    6 + 13 * (n - 1) = 5 + 10 * (m - 1) ∧
    100 < 6 + 13 * (n - 1) ∧
    6 + 13 * (n - 1) < 200 ∧
    n = 14 ∧ m = 18 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_collectors_problem_l3327_332722


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l3327_332751

/-- If f(x) = √(mx² - 6mx + m + 8) has a domain of ℝ, then m ∈ [0, 1] -/
theorem function_domain_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, mx^2 - 6*m*x + m + 8 ≥ 0) → 0 ≤ m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l3327_332751


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3327_332706

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 36 = 0) ↔ m = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3327_332706


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3327_332772

theorem quadratic_solution_sum (m n : ℝ) (h1 : m ≠ 0) 
  (h2 : m * 1^2 + n * 1 - 2022 = 0) : m + n + 1 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3327_332772


namespace NUMINAMATH_CALUDE_equation_solutions_l3327_332714

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, 64 * (x + 1)^3 = -125 ↔ x = -9/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3327_332714


namespace NUMINAMATH_CALUDE_beth_crayons_packs_l3327_332768

/-- Given the number of crayons per pack, extra crayons, and total crayons,
    calculates the number of packs of crayons. -/
def number_of_packs (crayons_per_pack : ℕ) (extra_crayons : ℕ) (total_crayons : ℕ) : ℕ :=
  (total_crayons - extra_crayons) / crayons_per_pack

theorem beth_crayons_packs :
  let crayons_per_pack : ℕ := 10
  let extra_crayons : ℕ := 6
  let total_crayons : ℕ := 40
  number_of_packs crayons_per_pack extra_crayons total_crayons = 3 := by
sorry

end NUMINAMATH_CALUDE_beth_crayons_packs_l3327_332768


namespace NUMINAMATH_CALUDE_fraction_simplification_l3327_332713

theorem fraction_simplification : (4 * 5) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3327_332713


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3327_332708

-- Define the set of real numbers that satisfy the inequality
def solution_set : Set ℝ := {x | x ≠ 0 ∧ (1 / x < x)}

-- Theorem statement
theorem inequality_solution_set : 
  solution_set = {x | -1 < x ∧ x < 0} ∪ {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3327_332708


namespace NUMINAMATH_CALUDE_unique_fraction_exists_l3327_332702

def is_relatively_prime (x y : ℕ+) : Prop := Nat.gcd x.val y.val = 1

theorem unique_fraction_exists : ∃! (x y : ℕ+), 
  is_relatively_prime x y ∧ 
  (x.val + 1 : ℚ) / (y.val + 1) = 1.2 * (x.val : ℚ) / y.val := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_exists_l3327_332702


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3327_332703

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0 → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3327_332703


namespace NUMINAMATH_CALUDE_marble_arrangement_remainder_l3327_332744

def green_marbles : ℕ := 7

-- m is the maximum number of red marbles
def red_marbles (m : ℕ) : Prop := 
  m = 19 ∧ ∀ k, k > m → ¬∃ (arr : List (Fin 2)), 
    arr.length = green_marbles + k ∧ 
    (arr.countP (λ i => arr[i]? = arr[i+1]?)) = 
    (arr.countP (λ i => arr[i]? ≠ arr[i+1]?))

-- N is the number of valid arrangements
def valid_arrangements (m : ℕ) : ℕ := Nat.choose (m + green_marbles) green_marbles

theorem marble_arrangement_remainder (m : ℕ) : 
  red_marbles m → valid_arrangements m % 1000 = 388 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_remainder_l3327_332744


namespace NUMINAMATH_CALUDE_equation_solution_l3327_332705

theorem equation_solution (x : ℝ) : x ≠ 2 → (-x^2 = (4*x + 2) / (x - 2)) ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3327_332705


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l3327_332767

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  (a + b + c) / 3 = 20 →
  max a (max b c) = 24 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l3327_332767


namespace NUMINAMATH_CALUDE_hexagon_walk_distance_l3327_332785

theorem hexagon_walk_distance (side_length : ℝ) (walk_distance : ℝ) (end_distance : ℝ) : 
  side_length = 3 →
  walk_distance = 11 →
  end_distance = 2 * Real.sqrt 3 →
  ∃ (x y : ℝ), x^2 + y^2 = end_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_walk_distance_l3327_332785


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l3327_332799

/-- Given a line y = kx + b tangent to the curve y = x³ + ax + 1 at the point (2, 3), prove that b = -15 -/
theorem tangent_line_b_value (k a : ℝ) : 
  (3 = 2 * k + b) →  -- Line equation at (2, 3)
  (3 = 8 + 2 * a + 1) →  -- Curve equation at (2, 3)
  (k = 3 * 2^2 + a) →  -- Slope of the tangent line equals derivative of the curve at x = 2
  (b = -15) := by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l3327_332799


namespace NUMINAMATH_CALUDE_star_commutative_star_associative_star_identity_star_not_distributive_l3327_332718

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Associativity
theorem star_associative : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := by sorry

-- Identity element
theorem star_identity : ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x := by sorry

-- Not distributive over addition
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_associative_star_identity_star_not_distributive_l3327_332718


namespace NUMINAMATH_CALUDE_cookie_baking_problem_l3327_332701

theorem cookie_baking_problem (x : ℚ) : 
  x > 0 → 
  x + x/2 + (3*x/2 - 4) = 92 → 
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_cookie_baking_problem_l3327_332701


namespace NUMINAMATH_CALUDE_current_speed_l3327_332778

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 22)
  (h2 : speed_against_current = 12) :
  ∃ (current_speed : ℝ), current_speed = 5 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l3327_332778


namespace NUMINAMATH_CALUDE_book_length_calculation_l3327_332724

theorem book_length_calculation (B₁ B₂ : ℕ) : 
  (2 : ℚ) / 3 * B₁ - (1 : ℚ) / 3 * B₁ = 90 →
  (3 : ℚ) / 4 * B₂ - (1 : ℚ) / 4 * B₂ = 120 →
  B₁ + B₂ = 510 := by
sorry

end NUMINAMATH_CALUDE_book_length_calculation_l3327_332724


namespace NUMINAMATH_CALUDE_fraction_problem_l3327_332763

theorem fraction_problem (f : ℚ) : f * 10 + 6 = 11 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3327_332763


namespace NUMINAMATH_CALUDE_fifth_sample_number_l3327_332727

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (first_sample : ℕ) (group : ℕ) : ℕ :=
  first_sample + (total / sample_size) * (group - 1)

/-- Theorem: In a systematic sampling of 100 samples from 2000 items, 
    if the first sample is numbered 11, then the fifth sample will be numbered 91 -/
theorem fifth_sample_number :
  systematic_sample 2000 100 11 5 = 91 := by
  sorry

end NUMINAMATH_CALUDE_fifth_sample_number_l3327_332727


namespace NUMINAMATH_CALUDE_water_breadth_in_cistern_l3327_332741

/-- Represents a rectangular cistern with water --/
structure WaterCistern where
  length : ℝ
  width : ℝ
  wetSurfaceArea : ℝ
  breadth : ℝ

/-- Theorem stating the correct breadth of water in the cistern --/
theorem water_breadth_in_cistern (c : WaterCistern)
  (h_length : c.length = 7)
  (h_width : c.width = 5)
  (h_wetArea : c.wetSurfaceArea = 68.6)
  (h_breadth_calc : c.breadth = (c.wetSurfaceArea - c.length * c.width) / (2 * (c.length + c.width))) :
  c.breadth = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_water_breadth_in_cistern_l3327_332741


namespace NUMINAMATH_CALUDE_alexei_weekly_loss_l3327_332750

/-- Given the weight loss information for Aleesia and Alexei, calculate Alexei's weekly weight loss. -/
theorem alexei_weekly_loss 
  (aleesia_weekly_loss : ℝ) 
  (aleesia_weeks : ℕ) 
  (alexei_weeks : ℕ) 
  (total_loss : ℝ) 
  (h1 : aleesia_weekly_loss = 1.5)
  (h2 : aleesia_weeks = 10)
  (h3 : alexei_weeks = 8)
  (h4 : total_loss = 35)
  : (total_loss - aleesia_weekly_loss * aleesia_weeks) / alexei_weeks = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_alexei_weekly_loss_l3327_332750


namespace NUMINAMATH_CALUDE_range_of_m_l3327_332700

/-- A quadratic function f(x) = ax^2 - 2ax + c -/
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + c

/-- The statement that f is monotonically decreasing on [0,1] -/
def is_monotone_decreasing (a c : ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a c x > f a c y

/-- The main theorem -/
theorem range_of_m (a c : ℝ) :
  is_monotone_decreasing a c →
  (∃ m, f a c m ≤ f a c 0) →
  ∃ m, 0 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3327_332700


namespace NUMINAMATH_CALUDE_unique_sine_solution_l3327_332790

theorem unique_sine_solution : ∃! x : Real, 0 ≤ x ∧ x < Real.pi ∧ Real.sin x = -0.45 := by
  sorry

end NUMINAMATH_CALUDE_unique_sine_solution_l3327_332790


namespace NUMINAMATH_CALUDE_eliot_account_balance_l3327_332761

/-- Represents the bank account balances of Al and Eliot -/
structure BankAccounts where
  al : ℝ
  eliot : ℝ

/-- The conditions of the problem -/
def satisfiesConditions (accounts : BankAccounts) : Prop :=
  accounts.al > accounts.eliot ∧
  accounts.al - accounts.eliot = (1 / 12) * (accounts.al + accounts.eliot) ∧
  1.10 * accounts.al - 1.15 * accounts.eliot = 22

/-- The theorem stating Eliot's account balance -/
theorem eliot_account_balance :
  ∀ accounts : BankAccounts, satisfiesConditions accounts → accounts.eliot = 146.67 := by
  sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l3327_332761


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l3327_332730

/-- Given a triangle with vertices at (7, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 24 square units, then x = -48/√58. -/
theorem third_vertex_coordinates (x : ℝ) :
  x < 0 →
  (1/2 : ℝ) * |x| * 3 * Real.sqrt 58 = 24 →
  x = -48 / Real.sqrt 58 := by
sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l3327_332730


namespace NUMINAMATH_CALUDE_proton_origin_probability_proton_max_probability_at_six_l3327_332757

/-- Represents the probability of a proton being at a specific position after n moves --/
def ProtonProbability (n : ℕ) (position : ℤ) : ℚ :=
  sorry

/-- The probability of the proton being at the origin after 4 moves --/
theorem proton_origin_probability : ProtonProbability 4 0 = 3/8 :=
  sorry

/-- The number of moves that maximizes the probability of the proton being at position 6 --/
def MaxProbabilityMoves : Finset ℕ :=
  sorry

/-- The probability of the proton being at position 6 is maximized when the number of moves is either 34 or 36 --/
theorem proton_max_probability_at_six :
  MaxProbabilityMoves = {34, 36} :=
  sorry

end NUMINAMATH_CALUDE_proton_origin_probability_proton_max_probability_at_six_l3327_332757


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3327_332753

theorem inequality_equivalence (y : ℝ) : 
  (3/10 : ℝ) + |2*y - 1/5| < 7/10 ↔ -1/10 < y ∧ y < 3/10 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3327_332753


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l3327_332735

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_increasing_function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_incr : increasing_on f (Set.Ici 0)) : 
  f (-2) < f (-3) ∧ f (-3) < f π := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l3327_332735


namespace NUMINAMATH_CALUDE_ellipse_and_triangle_area_l3327_332752

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the inscribed circle
def inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the parabola E
def parabola_E (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m ∧ 0 ≤ m ∧ m ≤ 1

-- State the theorem
theorem ellipse_and_triangle_area :
  ∀ (a b p m : ℝ) (x y : ℝ),
  ellipse_C a b x y →
  inscribed_circle x y →
  parabola_E p x y →
  line_l m x y →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola_E p x₁ y₁ ∧ 
    parabola_E p x₂ y₂ ∧ 
    line_l m x₁ y₁ ∧ 
    line_l m x₂ y₂ ∧ 
    x₁ ≠ x₂) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  (∃ (S : ℝ), S = (32 * Real.sqrt 6) / 9 ∧
    ∀ (S' : ℝ), S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_triangle_area_l3327_332752


namespace NUMINAMATH_CALUDE_reconstruct_diagonals_l3327_332782

/-- Represents a convex polygon with labeled vertices -/
structure LabeledPolygon where
  vertices : Finset ℕ
  labels : vertices → ℕ

/-- Represents a set of non-intersecting diagonals in a polygon -/
def Diagonals (p : LabeledPolygon) := Finset (Finset ℕ)

/-- Checks if a set of diagonals divides a polygon into triangles -/
def divides_into_triangles (p : LabeledPolygon) (d : Diagonals p) : Prop := sorry

/-- Checks if a set of diagonals matches the vertex labels -/
def matches_labels (p : LabeledPolygon) (d : Diagonals p) : Prop := sorry

/-- Main theorem: For any labeled convex polygon, there exists a unique set of diagonals
    that divides it into triangles and matches the labels -/
theorem reconstruct_diagonals (p : LabeledPolygon) : 
  ∃! d : Diagonals p, divides_into_triangles p d ∧ matches_labels p d := by sorry

end NUMINAMATH_CALUDE_reconstruct_diagonals_l3327_332782


namespace NUMINAMATH_CALUDE_sum_squares_bounds_l3327_332754

theorem sum_squares_bounds (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y = 10) :
  50 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 100 ∧
  (∃ a b : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ a + b = 10 ∧ a^2 + b^2 = 50) ∧
  (∃ c d : ℝ, c ≥ 0 ∧ d ≥ 0 ∧ c + d = 10 ∧ c^2 + d^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_bounds_l3327_332754


namespace NUMINAMATH_CALUDE_quadratic_sum_l3327_332731

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (3 * x^2 + 9 * x - 81 = a * (x + b)^2 + c) ∧ (a + b + c = -83.25) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3327_332731


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3327_332715

theorem smallest_integer_solution (x : ℤ) : x^2 - x = 24 → x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3327_332715


namespace NUMINAMATH_CALUDE_gcd_1230_990_l3327_332734

theorem gcd_1230_990 : Nat.gcd 1230 990 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1230_990_l3327_332734
