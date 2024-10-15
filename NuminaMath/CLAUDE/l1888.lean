import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_line_equation_l1888_188877

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The y-axis in 2D space -/
def yAxis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- Symmetry of a line with respect to the y-axis -/
def symmetricLine (l : Line) : Line :=
  { slope := -l.slope, intercept := l.intercept }

/-- The original line y = 2x + 1 -/
def originalLine : Line :=
  { slope := 2, intercept := 1 }

theorem symmetric_line_equation :
  symmetricLine originalLine = { slope := -2, intercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l1888_188877


namespace NUMINAMATH_CALUDE_calculate_expression_l1888_188834

theorem calculate_expression : 2 * (-3)^2 - 4 * (-3) - 15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1888_188834


namespace NUMINAMATH_CALUDE_initials_count_l1888_188864

/-- The number of letters available (A through H) -/
def num_letters : ℕ := 8

/-- The length of each set of initials -/
def set_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through H -/
theorem initials_count : (num_letters ^ set_length : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_initials_count_l1888_188864


namespace NUMINAMATH_CALUDE_product_of_exponents_l1888_188809

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^4 = 90 → 2^r + 44 = 76 → 5^3 + 6^s = 1421 → p * r * s = 40 := by
sorry

end NUMINAMATH_CALUDE_product_of_exponents_l1888_188809


namespace NUMINAMATH_CALUDE_pizza_theorem_l1888_188872

/-- Represents a pizza with given topping distributions -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  olive_slices : ℕ
  all_toppings_slices : ℕ

/-- Conditions for a valid pizza configuration -/
def is_valid_pizza (p : Pizza) : Prop :=
  p.total_slices = 20 ∧
  p.pepperoni_slices = 12 ∧
  p.mushroom_slices = 14 ∧
  p.olive_slices = 12 ∧
  p.all_toppings_slices ≤ p.total_slices ∧
  p.all_toppings_slices ≤ p.pepperoni_slices ∧
  p.all_toppings_slices ≤ p.mushroom_slices ∧
  p.all_toppings_slices ≤ p.olive_slices

theorem pizza_theorem (p : Pizza) (h : is_valid_pizza p) : p.all_toppings_slices = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l1888_188872


namespace NUMINAMATH_CALUDE_equation_roots_problem_l1888_188884

/-- Given two equations with specific root conditions, prove the value of 100c + d -/
theorem equation_roots_problem (c d : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    ∀ (x : ℝ), (x + c) * (x + d) * (x + 15) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∀ (x : ℝ), x ≠ -4 → (x + c) * (x + d) * (x + 15) ≠ 0) →
  (∃! (x : ℝ), (x + 3*c) * (x + 4) * (x + 9) = 0 ∧ (x + d) * (x + 15) ≠ 0) →
  100 * c + d = -291 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_problem_l1888_188884


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1888_188833

/-- Right isosceles triangle with legs of length 2 -/
structure RightIsoscelesTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle : Q.1 = 0 ∧ Q.2 = 0 ∧ P.2 = 0 ∧ R.1 = 0
  is_isosceles : P.1 = 2 ∧ R.2 = 2

/-- Circle tangent to triangle hypotenuse and coordinate axes -/
structure TangentCircle where
  S : ℝ × ℝ
  radius : ℝ
  tangent_to_hypotenuse : (S.1 - 2)^2 + (S.2 - 2)^2 = 8
  tangent_to_axes : S.1 = radius ∧ S.2 = radius

/-- The radius of the tangent circle is 4 -/
theorem tangent_circle_radius (t : RightIsoscelesTriangle) (c : TangentCircle) :
  c.radius = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l1888_188833


namespace NUMINAMATH_CALUDE_car_transfer_equation_l1888_188835

theorem car_transfer_equation (x : ℕ) : 
  (100 - x = 68 + x) ↔ 
  (∃ (team_a team_b : ℕ), 
    team_a = 100 ∧ 
    team_b = 68 ∧ 
    team_a - x = team_b + x) :=
sorry

end NUMINAMATH_CALUDE_car_transfer_equation_l1888_188835


namespace NUMINAMATH_CALUDE_complement_of_S_in_U_l1888_188887

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the set S
def S : Set Nat := {1, 2, 3, 4}

-- Theorem statement
theorem complement_of_S_in_U : 
  (U \ S) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_S_in_U_l1888_188887


namespace NUMINAMATH_CALUDE_chenny_friends_l1888_188862

/-- The number of friends Chenny has -/
def num_friends (initial_candies : ℕ) (bought_candies : ℕ) (candies_per_friend : ℕ) : ℕ :=
  (initial_candies + bought_candies) / candies_per_friend

/-- Proof that Chenny has 7 friends -/
theorem chenny_friends : num_friends 10 4 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chenny_friends_l1888_188862


namespace NUMINAMATH_CALUDE_junk_mail_for_block_l1888_188870

/-- Given a block with houses and junk mail distribution, calculate the total junk mail for the block. -/
def total_junk_mail (num_houses : ℕ) (pieces_per_house : ℕ) : ℕ :=
  num_houses * pieces_per_house

/-- Theorem: The total junk mail for a block with 6 houses, each receiving 4 pieces, is 24. -/
theorem junk_mail_for_block :
  total_junk_mail 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_for_block_l1888_188870


namespace NUMINAMATH_CALUDE_three_number_sum_l1888_188879

theorem three_number_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 10) 
  (h4 : (a + b + c) / 3 = a + 20) (h5 : (a + b + c) / 3 = c - 30) : 
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l1888_188879


namespace NUMINAMATH_CALUDE_rectangular_field_shortcut_l1888_188816

theorem rectangular_field_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_shortcut_l1888_188816


namespace NUMINAMATH_CALUDE_doughnuts_given_away_l1888_188873

theorem doughnuts_given_away (total_doughnuts : ℕ) (small_boxes_sold : ℕ) (large_boxes_sold : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : small_boxes_sold = 20)
  (h3 : large_boxes_sold = 10) :
  total_doughnuts - (small_boxes_sold * 6 + large_boxes_sold * 12) = 60 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_given_away_l1888_188873


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l1888_188854

/-- The number of yellow Easter eggs -/
def yellow_eggs : ℕ := 16

/-- The number of green Easter eggs -/
def green_eggs : ℕ := 28

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 4

theorem max_eggs_per_basket :
  eggs_per_basket = 4 ∧
  yellow_eggs % eggs_per_basket = 0 ∧
  green_eggs % eggs_per_basket = 0 ∧
  eggs_per_basket ≥ 2 ∧
  ∀ n : ℕ, n > eggs_per_basket →
    (yellow_eggs % n ≠ 0 ∨ green_eggs % n ≠ 0 ∨ n < 2) :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l1888_188854


namespace NUMINAMATH_CALUDE_distance_between_vertices_l1888_188855

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 2| = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 2.5)
def vertex2 : ℝ × ℝ := (0, -1.5)

-- Theorem statement
theorem distance_between_vertices : 
  ∀ (v1 v2 : ℝ × ℝ), 
  (∀ x y, parabola_equation x y → (x = v1.1 ∧ y = v1.2) ∨ (x = v2.1 ∧ y = v2.2)) →
  v1 = vertex1 ∧ v2 = vertex2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l1888_188855


namespace NUMINAMATH_CALUDE_triangle_intersecting_circle_theorem_l1888_188876

/-- Given a triangle ABC with sides a, b, c, and points A₁, B₁, C₁ on its sides satisfying certain ratios,
    if the circumcircle of A₁B₁C₁ intersects the sides of ABC at segments of lengths x, y, z,
    then x/a^(n-1) + y/b^(n-1) + z/c^(n-1) = 0 -/
theorem triangle_intersecting_circle_theorem
  (a b c : ℝ) (n : ℕ) (x y z : ℝ)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_AB₁ : ∃ B₁ : ℝ, B₁ > 0 ∧ B₁ < c ∧ B₁ / (c - B₁) = (c^n) / (a^n))
  (h_BC₁ : ∃ C₁ : ℝ, C₁ > 0 ∧ C₁ < a ∧ C₁ / (a - C₁) = (a^n) / (b^n))
  (h_CA₁ : ∃ A₁ : ℝ, A₁ > 0 ∧ A₁ < b ∧ A₁ / (b - A₁) = (b^n) / (c^n))
  (h_intersect : ∃ (A₁ B₁ C₁ : ℝ), 
    (B₁ > 0 ∧ B₁ < c ∧ B₁ / (c - B₁) = (c^n) / (a^n)) ∧
    (C₁ > 0 ∧ C₁ < a ∧ C₁ / (a - C₁) = (a^n) / (b^n)) ∧
    (A₁ > 0 ∧ A₁ < b ∧ A₁ / (b - A₁) = (b^n) / (c^n)) ∧
    (∃ (x' y' z' : ℝ), x' * x = (B₁ * (c - B₁)) ∧ 
                       y' * y = (C₁ * (a - C₁)) ∧ 
                       z' * z = (A₁ * (b - A₁))))
  : x / (a^(n-1)) + y / (b^(n-1)) + z / (c^(n-1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_intersecting_circle_theorem_l1888_188876


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1888_188807

theorem condition_sufficient_not_necessary :
  (∃ x y : ℝ, x = 1 ∧ y = -1 → x * y = -1) ∧
  ¬(∀ x y : ℝ, x * y = -1 → x = 1 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1888_188807


namespace NUMINAMATH_CALUDE_dress_shoes_count_l1888_188829

theorem dress_shoes_count (polished_percent : ℚ) (remaining : ℕ) : 
  polished_percent = 45/100 → remaining = 11 → (1 - polished_percent) * (2 * remaining / (1 - polished_percent)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dress_shoes_count_l1888_188829


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1888_188878

/-- If an arc of 60° on circle X has the same length as an arc of 40° on circle Y,
    then the ratio of the area of circle X to the area of circle Y is 4/9. -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) :
  (60 / 360) * (2 * Real.pi * X) = (40 / 360) * (2 * Real.pi * Y) →
  (Real.pi * X^2) / (Real.pi * Y^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1888_188878


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_l1888_188843

theorem smallest_k_with_remainder (k : ℕ) : k = 534 ↔ 
  (k > 2) ∧ 
  (k % 19 = 2) ∧ 
  (k % 7 = 2) ∧ 
  (k % 4 = 2) ∧ 
  (∀ m : ℕ, m > 2 ∧ m % 19 = 2 ∧ m % 7 = 2 ∧ m % 4 = 2 → k ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_l1888_188843


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1888_188880

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_angle : a^2 + b^2 = c^2) : 
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1888_188880


namespace NUMINAMATH_CALUDE_red_balls_count_l1888_188885

theorem red_balls_count (total : Nat) (white green yellow purple : Nat) (prob : Real) :
  total = 100 ∧ 
  white = 50 ∧ 
  green = 20 ∧ 
  yellow = 10 ∧ 
  purple = 3 ∧ 
  prob = 0.8 ∧ 
  prob = (white + green + yellow : Real) / total →
  total - (white + green + yellow + purple) = 17 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1888_188885


namespace NUMINAMATH_CALUDE_line_through_points_l1888_188819

/-- Given a line y = ax + b passing through points (3, 7) and (9/2, 13), prove that a - b = 9 -/
theorem line_through_points (a b : ℝ) : 
  (7 = a * 3 + b) → (13 = a * (9/2) + b) → a - b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1888_188819


namespace NUMINAMATH_CALUDE_possible_student_totals_l1888_188811

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : Nat
  groups_with_13 : Nat
  total_students : Nat

/-- Checks if the distribution is valid according to the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating the possible total numbers of students -/
theorem possible_student_totals :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

#check possible_student_totals

end NUMINAMATH_CALUDE_possible_student_totals_l1888_188811


namespace NUMINAMATH_CALUDE_paint_time_problem_l1888_188866

theorem paint_time_problem (anthony_time : ℝ) (combined_time : ℝ) (first_person_time : ℝ) : 
  anthony_time = 5 →
  combined_time = 20 / 7 →
  (1 / first_person_time + 1 / anthony_time) * combined_time = 2 →
  first_person_time = 2 := by
sorry

end NUMINAMATH_CALUDE_paint_time_problem_l1888_188866


namespace NUMINAMATH_CALUDE_diorama_time_proof_l1888_188804

/-- Proves that the total time spent on a diorama is 67 minutes, given the specified conditions. -/
theorem diorama_time_proof (planning_time building_time : ℕ) : 
  building_time = 3 * planning_time - 5 →
  building_time = 49 →
  planning_time + building_time = 67 := by
  sorry

#check diorama_time_proof

end NUMINAMATH_CALUDE_diorama_time_proof_l1888_188804


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1888_188846

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1888_188846


namespace NUMINAMATH_CALUDE_library_visitors_average_l1888_188859

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 26
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

theorem library_visitors_average :
  averageVisitorsPerDay 500 140 = 188 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l1888_188859


namespace NUMINAMATH_CALUDE_point_P_conditions_l1888_188898

def point_P (m : ℝ) : ℝ × ℝ := (3*m - 6, m + 1)

def point_A : ℝ × ℝ := (-1, 2)

theorem point_P_conditions (m : ℝ) :
  (∃ m, point_P m = (-9, 0) ∧ (point_P m).2 = 0) ∧
  (∃ m, point_P m = (-1, 8/3) ∧ (point_P m).1 = (point_A).1) :=
by sorry

end NUMINAMATH_CALUDE_point_P_conditions_l1888_188898


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_16_l1888_188818

/-- Given two points A and B in a 2D plane, where:
  - A is at the origin (0, 0)
  - B is on the line y = 6
  - The slope of segment AB is 3/5
  Prove that the sum of the x- and y-coordinates of B is 16. -/
theorem sum_of_coordinates_is_16 (B : ℝ × ℝ) : 
  B.2 = 6 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 5 → 
  B.1 + B.2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_16_l1888_188818


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l1888_188806

/-- Represents a point in 3D Cartesian coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in 3D Cartesian coordinates -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

/-- Theorem: The point symmetric to (-1, 2, 1) with respect to xOz plane is (-1, -2, 1) -/
theorem symmetric_point_xoz :
  let original := Point3D.mk (-1) 2 1
  symmetricPointXOZ original = Point3D.mk (-1) (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l1888_188806


namespace NUMINAMATH_CALUDE_project_completion_time_l1888_188801

/-- The time (in days) it takes for person A to complete the project alone -/
def time_A : ℝ := 20

/-- The time (in days) it takes for person B to complete the project alone -/
def time_B : ℝ := 30

/-- The number of days before project completion that A quits -/
def quit_time : ℝ := 15

/-- The total time to complete the project when A and B work together, with A quitting early -/
def total_time : ℝ := 36

theorem project_completion_time :
  (1 / time_A + 1 / time_B) * (total_time - quit_time) + (1 / time_B) * quit_time = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l1888_188801


namespace NUMINAMATH_CALUDE_tangent_sum_identity_l1888_188841

theorem tangent_sum_identity : 
  Real.sqrt 3 * Real.tan (12 * π / 180) + 
  Real.sqrt 3 * Real.tan (18 * π / 180) + 
  Real.tan (12 * π / 180) * Real.tan (18 * π / 180) = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_sum_identity_l1888_188841


namespace NUMINAMATH_CALUDE_max_crayfish_revenue_l1888_188850

/-- The revenue function for selling crayfish -/
def revenue (x : ℝ) : ℝ := (32 - x) * (x - 4.5)

/-- The theorem stating the maximum revenue and number of crayfish sold -/
theorem max_crayfish_revenue :
  ∃ (x : ℕ), x ≤ 32 ∧ 
  revenue (32 - x : ℝ) = 189 ∧
  ∀ (y : ℕ), y ≤ 32 → revenue (32 - y : ℝ) ≤ 189 ∧
  x = 14 :=
sorry

end NUMINAMATH_CALUDE_max_crayfish_revenue_l1888_188850


namespace NUMINAMATH_CALUDE_value_of_M_l1888_188899

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1504) ∧ (M = 2105.6) := by sorry

end NUMINAMATH_CALUDE_value_of_M_l1888_188899


namespace NUMINAMATH_CALUDE_xy_value_l1888_188838

theorem xy_value (x y : ℝ) 
  (h : (x / (1 - Complex.I)) - (y / (1 - 2 * Complex.I)) = (5 : ℝ) / (1 - 3 * Complex.I)) : 
  x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1888_188838


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l1888_188875

-- Define the function f(x) = |x| - 1
def f (x : ℝ) : ℝ := |x| - 1

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l1888_188875


namespace NUMINAMATH_CALUDE_product_simplification_l1888_188871

theorem product_simplification :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l1888_188871


namespace NUMINAMATH_CALUDE_mandy_pieces_l1888_188857

def chocolate_distribution (total : Nat) (n : Nat) : Nat :=
  if n = 0 then
    total
  else
    chocolate_distribution (total / 2) (n - 1)

theorem mandy_pieces : chocolate_distribution 60 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_mandy_pieces_l1888_188857


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths_l1888_188894

theorem unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths : 
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 20 * k) ∧ 
    (9 : ℝ) < n.val ^ (1/3 : ℝ) ∧ 
    n.val ^ (1/3 : ℝ) < (91/10 : ℝ) ∧
    n = 740 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths_l1888_188894


namespace NUMINAMATH_CALUDE_base_conversion_1729_l1888_188845

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_1729 :
  toBase5 1729 = [2, 3, 4, 0, 4] ∧ fromBase5 [2, 3, 4, 0, 4] = 1729 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l1888_188845


namespace NUMINAMATH_CALUDE_existence_of_integer_representation_l1888_188813

theorem existence_of_integer_representation (n : ℤ) :
  ∃ (a b : ℤ), n = ⌊(a : ℝ) * Real.sqrt 2⌋ + ⌊(b : ℝ) * Real.sqrt 3⌋ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integer_representation_l1888_188813


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_side_equation_l1888_188822

/-- Given a triangle with sides a, b, and c satisfying a² + bc = b² + ac, prove it's isosceles --/
theorem isosceles_triangle_from_side_equation 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_equation : a^2 + b*c = b^2 + a*c) : 
  a = b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_side_equation_l1888_188822


namespace NUMINAMATH_CALUDE_jack_ernie_income_ratio_l1888_188867

theorem jack_ernie_income_ratio :
  ∀ (ernie_prev ernie_curr jack_curr : ℝ),
    ernie_curr = (4/5) * ernie_prev →
    ernie_curr + jack_curr = 16800 →
    ernie_prev = 6000 →
    jack_curr / ernie_prev = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_ernie_income_ratio_l1888_188867


namespace NUMINAMATH_CALUDE_least_reducible_n_l1888_188810

def is_reducible (a b : Int) : Prop :=
  Int.gcd a b > 1

def fraction_numerator (n : Int) : Int :=
  2*n - 26

def fraction_denominator (n : Int) : Int :=
  10*n + 12

theorem least_reducible_n :
  (∀ k : Nat, k > 0 ∧ k < 49 → ¬(is_reducible (fraction_numerator k) (fraction_denominator k))) ∧
  (is_reducible (fraction_numerator 49) (fraction_denominator 49)) :=
sorry

end NUMINAMATH_CALUDE_least_reducible_n_l1888_188810


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1888_188882

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h_quad : IsQuadratic f) 
  (h_f0 : f 0 = 1) 
  (h_fx1 : ∀ x, f (x + 1) = f x + x + 1) : 
  ∀ x, f x = (1/2) * x^2 + (1/2) * x + 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1888_188882


namespace NUMINAMATH_CALUDE_number_increase_l1888_188820

theorem number_increase (n : ℕ) (m : ℕ) (increase : ℕ) : n = 18 → m = 12 → increase = m * n - n := by
  sorry

end NUMINAMATH_CALUDE_number_increase_l1888_188820


namespace NUMINAMATH_CALUDE_box_surface_area_l1888_188865

/-- Calculates the surface area of the interior of a box formed by cutting out square corners from a rectangular sheet and folding the sides. -/
def interior_surface_area (sheet_length sheet_width corner_size : ℕ) : ℕ :=
  let remaining_area := sheet_length * sheet_width - 4 * (corner_size * corner_size)
  remaining_area

/-- The surface area of the interior of a box formed by cutting out 8-unit squares from the corners of a 40x50 unit sheet and folding the sides is 1744 square units. -/
theorem box_surface_area : interior_surface_area 40 50 8 = 1744 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l1888_188865


namespace NUMINAMATH_CALUDE_book_original_price_l1888_188897

/-- Given a book sold for $78 with a 30% profit, prove that the original price was $60 -/
theorem book_original_price (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 78 → profit_percentage = 30 → 
  ∃ (original_price : ℝ), 
    original_price = 60 ∧ 
    selling_price = original_price * (1 + profit_percentage / 100) := by
  sorry

#check book_original_price

end NUMINAMATH_CALUDE_book_original_price_l1888_188897


namespace NUMINAMATH_CALUDE_no_fixed_point_function_l1888_188888

-- Define the types for our polynomials
variable (p q h : ℝ → ℝ)

-- Define m and n as natural numbers
variable (m n : ℕ)

-- Define the descending property for p
def IsDescending (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem no_fixed_point_function
  (hp : IsDescending p)
  (hpqh : ∀ x, p (q (n * x + m) + h x) = n * (q (p x) + h x) + m) :
  ¬ ∃ f : ℝ → ℝ, ∀ x, f (q (p x) + h x) = f x ^ 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_point_function_l1888_188888


namespace NUMINAMATH_CALUDE_problem_solution_l1888_188830

theorem problem_solution (x a : ℝ) :
  (a > 0) →
  (∀ x, (x^2 - 4*x + 3 < 0 ∧ x^2 - x - 12 ≤ 0 ∧ x^2 + 2*x - 8 > 0) → (2 < x ∧ x < 3)) ∧
  ((∀ x, (x^2 - 4*a*x + 3*a^2 ≥ 0) → (x^2 - x - 12 > 0 ∨ x^2 + 2*x - 8 ≤ 0)) ∧
   (∃ x, (x^2 - x - 12 > 0 ∨ x^2 + 2*x - 8 ≤ 0) ∧ x^2 - 4*a*x + 3*a^2 < 0) →
   (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1888_188830


namespace NUMINAMATH_CALUDE_distribution_equivalence_l1888_188827

/-- The number of ways to distribute n indistinguishable objects among k recipients,
    with each recipient receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribution_equivalence :
  distribute 10 7 = choose 9 6 := by sorry

end NUMINAMATH_CALUDE_distribution_equivalence_l1888_188827


namespace NUMINAMATH_CALUDE_total_medicine_boxes_l1888_188840

def vitamins : ℕ := 472
def supplements : ℕ := 288

theorem total_medicine_boxes : vitamins + supplements = 760 := by
  sorry

end NUMINAMATH_CALUDE_total_medicine_boxes_l1888_188840


namespace NUMINAMATH_CALUDE_circle_area_through_point_l1888_188863

/-- The area of a circle with center P(2, 5) passing through point Q(6, -1) is 52π. -/
theorem circle_area_through_point (P Q : ℝ × ℝ) : P = (2, 5) → Q = (6, -1) → 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 52 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_point_l1888_188863


namespace NUMINAMATH_CALUDE_x_squared_minus_y_equals_three_l1888_188821

theorem x_squared_minus_y_equals_three (x y : ℝ) :
  |x + 1| + (2 * x - y)^2 = 0 → x^2 - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_equals_three_l1888_188821


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l1888_188861

theorem mean_of_combined_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 53 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l1888_188861


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1888_188803

theorem polynomial_expansion (x : ℝ) :
  (3 * x^2 - 4 * x + 3) * (-2 * x^2 + 3 * x - 4) =
  -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1888_188803


namespace NUMINAMATH_CALUDE_max_a_for_real_roots_l1888_188856

theorem max_a_for_real_roots : ∃ (a_max : ℤ), 
  (∀ a : ℤ, (∃ x : ℝ, (a + 1 : ℝ) * x^2 - 2*x + 3 = 0) → a ≤ a_max) ∧ 
  (∃ x : ℝ, (a_max + 1 : ℝ) * x^2 - 2*x + 3 = 0) ∧ 
  a_max = -2 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_real_roots_l1888_188856


namespace NUMINAMATH_CALUDE_cone_base_radius_l1888_188831

/-- Given a cone whose lateral surface is a sector of a circle with a central angle of 216° 
    and a radius of 15 cm, the radius of the base of the cone is 9 cm. -/
theorem cone_base_radius (central_angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) : 
  central_angle = 216 * (π / 180) →  -- Convert 216° to radians
  sector_radius = 15 →
  base_radius = sector_radius * (central_angle / (2 * π)) →
  base_radius = 9 := by
sorry


end NUMINAMATH_CALUDE_cone_base_radius_l1888_188831


namespace NUMINAMATH_CALUDE_solve_star_equation_l1888_188896

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem solve_star_equation (x : ℝ) : star 3 x = 15 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1888_188896


namespace NUMINAMATH_CALUDE_tape_overlap_division_l1888_188800

/-- Given 5 pieces of tape, each 2.7 meters long, with an overlap of 0.3 meters between pieces,
    when divided into 6 equal parts, each part is 2.05 meters long. -/
theorem tape_overlap_division (n : ℕ) (piece_length overlap_length : ℝ) (h1 : n = 5) 
    (h2 : piece_length = 2.7) (h3 : overlap_length = 0.3) : 
  (n * piece_length - (n - 1) * overlap_length) / 6 = 2.05 := by
  sorry

#check tape_overlap_division

end NUMINAMATH_CALUDE_tape_overlap_division_l1888_188800


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1888_188842

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 25 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1888_188842


namespace NUMINAMATH_CALUDE_octagon_area_in_square_l1888_188844

-- Define the square's perimeter
def square_perimeter : ℝ := 72

-- Define the number of parts each side is divided into
def parts_per_side : ℕ := 3

-- Theorem statement
theorem octagon_area_in_square (square_perimeter : ℝ) (parts_per_side : ℕ) :
  square_perimeter = 72 ∧ parts_per_side = 3 →
  let side_length := square_perimeter / 4
  let segment_length := side_length / parts_per_side
  let triangle_area := 1/2 * segment_length * segment_length
  let total_removed_area := 4 * triangle_area
  let square_area := side_length * side_length
  square_area - total_removed_area = 252 :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_in_square_l1888_188844


namespace NUMINAMATH_CALUDE_intersection_count_theorem_l1888_188851

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters

/-- Represents the number of intersection points between two lines -/
def intersectionCount (l1 l2 : Line3D) : ℕ := sorry

/-- Represents if two lines are skew -/
def areSkew (l1 l2 : Line3D) : Prop := sorry

/-- Represents if two lines are parallel -/
def areParallel (l1 l2 : Line3D) : Prop := sorry

/-- Represents if a line is perpendicular to two other lines -/
def isCommonPerpendicular (l l1 l2 : Line3D) : Prop := sorry

theorem intersection_count_theorem 
  (a b EF l : Line3D) 
  (h1 : isCommonPerpendicular EF a b) 
  (h2 : areSkew a b) 
  (h3 : areParallel l EF) : 
  (intersectionCount l a + intersectionCount l b = 0) ∨ 
  (intersectionCount l a + intersectionCount l b = 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_theorem_l1888_188851


namespace NUMINAMATH_CALUDE_angela_has_eight_more_l1888_188886

/-- The number of marbles each person has -/
structure MarbleCount where
  albert : ℕ
  angela : ℕ
  allison : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.albert = 3 * m.angela ∧
  m.angela > m.allison ∧
  m.allison = 28 ∧
  m.albert + m.allison = 136

/-- The theorem stating that Angela has 8 more marbles than Allison -/
theorem angela_has_eight_more (m : MarbleCount) 
  (h : marble_problem m) : m.angela - m.allison = 8 := by
  sorry

end NUMINAMATH_CALUDE_angela_has_eight_more_l1888_188886


namespace NUMINAMATH_CALUDE_late_fisherman_arrival_day_l1888_188839

/-- Represents the day of the week when the late fisherman arrived -/
inductive ArrivalDay
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Calculates the number of days the late fisherman fished -/
def daysLateArrivalFished (d : ArrivalDay) : Nat :=
  match d with
  | .Monday => 5
  | .Tuesday => 4
  | .Wednesday => 3
  | .Thursday => 2
  | .Friday => 1

theorem late_fisherman_arrival_day :
  ∃ (n : Nat) (d : ArrivalDay),
    n > 0 ∧
    50 * n + 10 * (daysLateArrivalFished d) = 370 ∧
    d = ArrivalDay.Thursday :=
by sorry

end NUMINAMATH_CALUDE_late_fisherman_arrival_day_l1888_188839


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1888_188808

theorem trigonometric_identity (a b : ℝ) : 
  (∀ x : ℝ, 2 * (Real.cos (x + b / 2))^2 - 2 * Real.sin (a * x - π / 2) * Real.cos (a * x - π / 2) = 1) ↔ 
  ((a = 1 ∧ ∃ k : ℤ, b = -3 * π / 2 + 2 * ↑k * π) ∨ 
   (a = -1 ∧ ∃ k : ℤ, b = 3 * π / 2 + 2 * ↑k * π)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1888_188808


namespace NUMINAMATH_CALUDE_fraction_equality_l1888_188848

theorem fraction_equality (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (1 / a + 1 / b = 4 / (a + b)) → (a / b + b / a = 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1888_188848


namespace NUMINAMATH_CALUDE_min_value_abc_l1888_188817

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ a₀ + 3 * b₀ + 9 * c₀ = 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l1888_188817


namespace NUMINAMATH_CALUDE_negative_a_cubed_div_squared_l1888_188802

theorem negative_a_cubed_div_squared (a : ℝ) : (-a)^3 / (-a)^2 = -a := by sorry

end NUMINAMATH_CALUDE_negative_a_cubed_div_squared_l1888_188802


namespace NUMINAMATH_CALUDE_equation_solutions_l1888_188836

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 8*x + 6 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x - 1)^2 = 3*x - 3
  let sol1 : Set ℝ := {4 + Real.sqrt 10, 4 - Real.sqrt 10}
  let sol2 : Set ℝ := {1, 4}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1888_188836


namespace NUMINAMATH_CALUDE_total_marks_is_530_l1888_188874

/-- Calculates the total marks scored by Amaya in all subjects given the following conditions:
  * Amaya scored 20 marks fewer in Maths than in Arts
  * She got 10 marks more in Social Studies than in Music
  * She scored 70 in Music
  * She scored 1/10 less in Maths than in Arts
-/
def totalMarks (musicScore : ℕ) : ℕ :=
  let socialStudiesScore := musicScore + 10
  let artsScore := 200
  let mathsScore := artsScore - 20
  musicScore + socialStudiesScore + artsScore + mathsScore

theorem total_marks_is_530 : totalMarks 70 = 530 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_is_530_l1888_188874


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_b_geq_f_a_l1888_188889

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 2*a| + |x - a|

-- Theorem for part I
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x > 3} = {x : ℝ | x < 0 ∨ x > 3} := by sorry

-- Theorem for part II
theorem f_b_geq_f_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f a b ≥ f a a ∧
  (f a b = f a a ↔ (2*a - b) * (b - a) ≥ 0) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_b_geq_f_a_l1888_188889


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l1888_188828

theorem opposite_sides_line_range (a : ℝ) : 
  (((3 : ℝ) * 3 - 2 * 1 + a > 0 ∧ (3 : ℝ) * (-4) - 2 * 6 + a < 0) ∨
   ((3 : ℝ) * 3 - 2 * 1 + a < 0 ∧ (3 : ℝ) * (-4) - 2 * 6 + a > 0)) →
  -7 < a ∧ a < 24 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l1888_188828


namespace NUMINAMATH_CALUDE_biffs_drinks_and_snacks_cost_l1888_188837

/-- Represents Biff's expenses and earnings during his bus trip -/
structure BusTrip where
  ticket_cost : ℝ
  headphones_cost : ℝ
  online_rate : ℝ
  wifi_rate : ℝ
  trip_duration : ℝ

/-- Calculates the amount Biff spent on drinks and snacks -/
def drinks_and_snacks_cost (trip : BusTrip) : ℝ :=
  (trip.online_rate - trip.wifi_rate) * trip.trip_duration - 
  (trip.ticket_cost + trip.headphones_cost)

/-- Theorem stating that Biff's expenses on drinks and snacks equal $3 -/
theorem biffs_drinks_and_snacks_cost :
  let trip := BusTrip.mk 11 16 12 2 3
  drinks_and_snacks_cost trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_biffs_drinks_and_snacks_cost_l1888_188837


namespace NUMINAMATH_CALUDE_coefficient_a2_l1888_188847

theorem coefficient_a2 (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, a₀ + a₁ * (2 * x - 1) + a₂ * (2 * x - 1)^2 + a₃ * (2 * x - 1)^3 + a₄ * (2 * x - 1)^4 = x^4) →
  a₂ = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_l1888_188847


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l1888_188823

def lennon_current_age : ℕ := 8
def ophelia_current_age : ℕ := 38
def years_passed : ℕ := 2

def lennon_future_age : ℕ := lennon_current_age + years_passed
def ophelia_future_age : ℕ := ophelia_current_age + years_passed

theorem age_ratio_in_two_years :
  ophelia_future_age / lennon_future_age = 4 ∧ ophelia_future_age % lennon_future_age = 0 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l1888_188823


namespace NUMINAMATH_CALUDE_lcm_18_24_l1888_188893

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1888_188893


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1888_188895

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a < 0)
  (h2 : quadratic_function a b c 2 = 0)
  (h3 : quadratic_function a b c (-1) = 0) :
  {x : ℝ | quadratic_function a b c x ≥ 0} = Set.Icc (-1) 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1888_188895


namespace NUMINAMATH_CALUDE_ratio_of_a_to_b_l1888_188883

theorem ratio_of_a_to_b (a b : ℚ) (h : (6*a - 5*b) / (8*a - 3*b) = 2/7) : 
  a/b = 29/26 := by sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_b_l1888_188883


namespace NUMINAMATH_CALUDE_no_solution_exponential_equation_l1888_188805

theorem no_solution_exponential_equation :
  ¬∃ y : ℝ, (16 : ℝ)^(3*y - 6) = (64 : ℝ)^(2*y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exponential_equation_l1888_188805


namespace NUMINAMATH_CALUDE_poly5_with_negative_integer_roots_l1888_188815

/-- A polynomial of degree 5 with integer coefficients -/
structure Poly5 where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ
  t : ℤ

/-- The polynomial function corresponding to a Poly5 -/
def poly5_func (g : Poly5) : ℝ → ℝ :=
  λ x => x^5 + g.p * x^4 + g.q * x^3 + g.r * x^2 + g.s * x + g.t

/-- Predicate stating that all roots of a polynomial are negative integers -/
def all_roots_negative_integers (g : Poly5) : Prop :=
  ∀ x : ℝ, poly5_func g x = 0 → (∃ n : ℤ, x = -n ∧ n > 0)

theorem poly5_with_negative_integer_roots
  (g : Poly5)
  (h1 : all_roots_negative_integers g)
  (h2 : g.p + g.q + g.r + g.s + g.t = 3024) :
  g.t = 1600 := by
  sorry

end NUMINAMATH_CALUDE_poly5_with_negative_integer_roots_l1888_188815


namespace NUMINAMATH_CALUDE_boat_speed_proof_l1888_188891

/-- The speed of the boat in standing water -/
def boat_speed : ℝ := 9

/-- The speed of the stream -/
def stream_speed : ℝ := 6

/-- The distance traveled in one direction -/
def distance : ℝ := 170

/-- The total time taken for the round trip -/
def total_time : ℝ := 68

theorem boat_speed_proof :
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = total_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_proof_l1888_188891


namespace NUMINAMATH_CALUDE_company_workforce_after_hiring_l1888_188860

theorem company_workforce_after_hiring 
  (initial_female_percentage : Real)
  (additional_male_workers : Nat)
  (new_female_percentage : Real) :
  initial_female_percentage = 0.60 →
  additional_male_workers = 22 →
  new_female_percentage = 0.55 →
  (initial_female_percentage * (264 - additional_male_workers)) / 264 = new_female_percentage :=
by sorry

end NUMINAMATH_CALUDE_company_workforce_after_hiring_l1888_188860


namespace NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l1888_188869

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  sorry -- Definition of a square

-- Define the property of having four equal interior angles
def has_four_equal_angles (q : Quadrilateral) : Prop :=
  sorry -- Definition of having four equal interior angles

theorem equal_angles_necessary_not_sufficient :
  (∀ q : Quadrilateral, is_square q → has_four_equal_angles q) ∧
  (∃ q : Quadrilateral, has_four_equal_angles q ∧ ¬is_square q) :=
sorry

end NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l1888_188869


namespace NUMINAMATH_CALUDE_total_days_2004_to_2008_l1888_188814

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDaysInRange (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |> List.sum

theorem total_days_2004_to_2008 :
  totalDaysInRange 2004 2008 = 1827 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2004_to_2008_l1888_188814


namespace NUMINAMATH_CALUDE_fraction_equality_l1888_188832

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1888_188832


namespace NUMINAMATH_CALUDE_plane_division_by_lines_l1888_188826

/-- The number of regions created by n non-parallel lines in a plane --/
def num_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of infinite regions created by n non-parallel lines in a plane --/
def num_infinite_regions (n : ℕ) : ℕ := 2 * n

theorem plane_division_by_lines (n : ℕ) (h : n = 20) :
  num_regions n = 211 ∧ num_regions n - num_infinite_regions n = 171 :=
sorry

end NUMINAMATH_CALUDE_plane_division_by_lines_l1888_188826


namespace NUMINAMATH_CALUDE_petrol_expense_l1888_188858

def monthly_expenses (rent milk groceries education misc petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + misc + petrol

def savings_percentage : ℚ := 1/10

theorem petrol_expense (rent milk groceries education misc savings : ℕ) 
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : misc = 6100)
  (h6 : savings = 2400)
  : ∃ (petrol total_salary : ℕ),
    (savings_percentage * total_salary = savings) ∧
    (monthly_expenses rent milk groceries education misc petrol + savings = total_salary) ∧
    (petrol = 2000) := by
  sorry

end NUMINAMATH_CALUDE_petrol_expense_l1888_188858


namespace NUMINAMATH_CALUDE_victoria_rice_packets_l1888_188824

def rice_packets (initial_balance : ℕ) (rice_cost : ℕ) (wheat_flour_packets : ℕ) (wheat_flour_cost : ℕ) (soda_cost : ℕ) (remaining_balance : ℕ) : ℕ :=
  (initial_balance - (wheat_flour_packets * wheat_flour_cost + soda_cost + remaining_balance)) / rice_cost

theorem victoria_rice_packets :
  rice_packets 500 20 3 25 150 235 = 2 := by
sorry

end NUMINAMATH_CALUDE_victoria_rice_packets_l1888_188824


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l1888_188853

theorem arithmetic_geometric_mean_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_mean : (x + y) / 2 = 3 * Real.sqrt (x * y)) : 
  ∃ (n : ℤ), ∀ (m : ℤ), |x / y - n| ≤ |x / y - m| ∧ n = 34 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l1888_188853


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1888_188852

/-- Given a hyperbola with equation x²/a² - y²/16 = 1 where a > 0,
    if one of its asymptotes has equation 2x - y = 0, then a = 2 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 16 = 1 ∧ 2*x - y = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1888_188852


namespace NUMINAMATH_CALUDE_sum_of_complex_magnitudes_l1888_188849

theorem sum_of_complex_magnitudes : 
  Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) + Complex.abs (6 - 8*I) = 2 * Real.sqrt 34 + 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_magnitudes_l1888_188849


namespace NUMINAMATH_CALUDE_harriett_quarters_l1888_188892

/-- Represents the number of coins of each type found by Harriett --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents for a given coin count --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- The coin count found by Harriett --/
def harriettCoins : CoinCount := {
  quarters := 10,  -- This is what we want to prove
  dimes := 3,
  nickels := 3,
  pennies := 5
}

theorem harriett_quarters : 
  harriettCoins.quarters = 10 ∧ totalValue harriettCoins = 300 := by
  sorry

end NUMINAMATH_CALUDE_harriett_quarters_l1888_188892


namespace NUMINAMATH_CALUDE_f_geq_one_for_a_eq_two_g_min_value_g_min_value_exists_l1888_188825

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x + 2/a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + f a (-x)

-- Theorem for part (1)
theorem f_geq_one_for_a_eq_two (x : ℝ) : f 2 x ≥ 1 := by sorry

-- Theorem for part (2)
theorem g_min_value (a : ℝ) : ∀ x : ℝ, g a x ≥ 4 * Real.sqrt 2 := by sorry

-- Theorem for existence of x that achieves the minimum value
theorem g_min_value_exists (a : ℝ) : ∃ x : ℝ, g a x = 4 * Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_f_geq_one_for_a_eq_two_g_min_value_g_min_value_exists_l1888_188825


namespace NUMINAMATH_CALUDE_equipment_marked_price_marked_price_approx_58_82_l1888_188868

/-- The marked price of equipment given specific buying and selling conditions --/
theorem equipment_marked_price (original_price : ℝ) (buying_discount : ℝ) 
  (desired_gain : ℝ) (selling_discount : ℝ) : ℝ :=
  let cost_price := original_price * (1 - buying_discount)
  let selling_price := cost_price * (1 + desired_gain)
  selling_price / (1 - selling_discount)

/-- The marked price of equipment is approximately 58.82 given the specific conditions --/
theorem marked_price_approx_58_82 : 
  ∃ ε > 0, |equipment_marked_price 50 0.2 0.25 0.15 - 58.82| < ε :=
sorry

end NUMINAMATH_CALUDE_equipment_marked_price_marked_price_approx_58_82_l1888_188868


namespace NUMINAMATH_CALUDE_cube_difference_l1888_188890

theorem cube_difference (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) :
  a^3 - b^3 = 486 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l1888_188890


namespace NUMINAMATH_CALUDE_fish_remaining_l1888_188812

theorem fish_remaining (initial : ℝ) (given_away : ℝ) (remaining : ℝ) : 
  initial = 47.0 → given_away = 22.0 → remaining = initial - given_away → remaining = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_fish_remaining_l1888_188812


namespace NUMINAMATH_CALUDE_two_digit_number_system_l1888_188881

theorem two_digit_number_system (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 →
  (10 * x + y) - 3 * (x + y) = 13 →
  (10 * x + y) % (x + y) = 6 →
  (10 * x + y) / (x + y) = 4 →
  (10 * x + y - 3 * (x + y) = 13 ∧ 10 * x + y - 6 = 4 * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_system_l1888_188881
