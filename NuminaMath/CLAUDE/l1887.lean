import Mathlib

namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_twelve_l1887_188736

def repeating_decimal : ℚ := 356 / 999

theorem product_of_repeating_decimal_and_twelve :
  repeating_decimal * 12 = 1424 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_twelve_l1887_188736


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1887_188779

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 10 = 9) ∧ 
  (∀ m : ℕ, m > 0 → m % 2 = 1 → m % 3 = 2 → m % 10 = 9 → m ≥ n) ∧
  (n = 59) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1887_188779


namespace NUMINAMATH_CALUDE_quadratic_as_binomial_square_l1887_188789

theorem quadratic_as_binomial_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_as_binomial_square_l1887_188789


namespace NUMINAMATH_CALUDE_cloth_square_cutting_l1887_188783

/-- Proves that a 29 cm by 40 cm cloth can be cut into at most 280 squares of 4 square centimeters each. -/
theorem cloth_square_cutting (cloth_width : ℕ) (cloth_length : ℕ) 
  (square_area : ℕ) (max_squares : ℕ) : 
  cloth_width = 29 → 
  cloth_length = 40 → 
  square_area = 4 → 
  max_squares = 280 → 
  (cloth_width / 2) * (cloth_length / 2) ≤ max_squares :=
by
  sorry

#check cloth_square_cutting

end NUMINAMATH_CALUDE_cloth_square_cutting_l1887_188783


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l1887_188782

def m : ℕ := 2021^2 + 3^2021

theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ := 2021^2 + 3^2021) :
  (m^2 + 3^m) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l1887_188782


namespace NUMINAMATH_CALUDE_eastward_fish_caught_fraction_l1887_188714

/-- Given the following conditions:
  - 1800 fish swim westward
  - 3200 fish swim eastward
  - 500 fish swim north
  - Fishers catch 3/4 of the fish that swam westward
  - There are 2870 fish left in the sea
Prove that the fraction of eastward-swimming fish caught by fishers is 2/5 -/
theorem eastward_fish_caught_fraction :
  let total_fish : ℕ := 1800 + 3200 + 500
  let westward_fish : ℕ := 1800
  let eastward_fish : ℕ := 3200
  let northward_fish : ℕ := 500
  let westward_caught_fraction : ℚ := 3 / 4
  let remaining_fish : ℕ := 2870
  let eastward_caught_fraction : ℚ := 2 / 5
  (total_fish : ℚ) - (westward_caught_fraction * westward_fish + eastward_caught_fraction * eastward_fish) = remaining_fish :=
by
  sorry

#check eastward_fish_caught_fraction

end NUMINAMATH_CALUDE_eastward_fish_caught_fraction_l1887_188714


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1887_188799

theorem quadratic_minimum : 
  (∃ (y : ℝ), y^2 - 6*y + 5 = -4) ∧ 
  (∀ (y : ℝ), y^2 - 6*y + 5 ≥ -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1887_188799


namespace NUMINAMATH_CALUDE_last_two_digits_same_l1887_188711

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = a n + 54) ∨ (a (n + 1) = a n + 77)

theorem last_two_digits_same (a : ℕ → ℕ) (h : sequence_property a) :
  ∃ k : ℕ, a k % 100 = a (k + 1) % 100 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_same_l1887_188711


namespace NUMINAMATH_CALUDE_circle_radius_in_square_configuration_l1887_188788

/-- A configuration of five congruent circles packed inside a unit square,
    where one circle is centered at the center of the square and
    the other four are tangent to the central circle and two adjacent sides of the square. -/
structure CircleConfiguration where
  radius : ℝ
  is_unit_square : ℝ
  circle_count : ℕ
  central_circle_exists : Bool
  external_circles_tangent : Bool

/-- The radius of each circle in the described configuration is √2 / (4 + 2√2) -/
theorem circle_radius_in_square_configuration (config : CircleConfiguration) 
  (h1 : config.is_unit_square = 1)
  (h2 : config.circle_count = 5)
  (h3 : config.central_circle_exists = true)
  (h4 : config.external_circles_tangent = true) :
  config.radius = Real.sqrt 2 / (4 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_square_configuration_l1887_188788


namespace NUMINAMATH_CALUDE_parallel_plane_count_l1887_188760

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Enum representing the possible number of parallel planes -/
inductive ParallelPlaneCount
  | Zero
  | One
  | Infinite

/-- Function to determine the number of parallel planes -/
def countParallelPlanes (l1 l2 : Line3D) : ParallelPlaneCount :=
  sorry

/-- Theorem stating that the number of parallel planes is either zero, one, or infinite -/
theorem parallel_plane_count (l1 l2 : Line3D) :
  ∃ (count : ParallelPlaneCount), countParallelPlanes l1 l2 = count :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_count_l1887_188760


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l1887_188759

/-- The total height of a sculpture and its base -/
def total_height (sculpture_height_m : ℝ) (base_height_cm : ℝ) : ℝ :=
  sculpture_height_m * 100 + base_height_cm

/-- Theorem stating that a 0.88m sculpture on a 20cm base is 108cm tall -/
theorem sculpture_and_base_height : 
  total_height 0.88 20 = 108 := by sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l1887_188759


namespace NUMINAMATH_CALUDE_semicircle_radius_with_inscribed_circles_l1887_188735

/-- The radius of a semicircle that inscribes two externally touching circles -/
theorem semicircle_radius_with_inscribed_circles 
  (r₁ r₂ R : ℝ) 
  (h₁ : r₁ = Real.sqrt 19)
  (h₂ : r₂ = Real.sqrt 76)
  (h_touch : r₁ + r₂ = R - r₁ + R - r₂) 
  (h_inscribed : R^2 = (R - r₁)^2 + r₁^2 ∧ R^2 = (R - r₂)^2 + r₂^2) :
  R = 4 * Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_semicircle_radius_with_inscribed_circles_l1887_188735


namespace NUMINAMATH_CALUDE_is_center_of_hyperbola_l1887_188754

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 1001 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ (x y : ℝ),
  hyperbola_equation x y ↔
  hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) :=
sorry

end NUMINAMATH_CALUDE_is_center_of_hyperbola_l1887_188754


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_median_area_l1887_188745

theorem right_isosceles_triangle_median_area (h : ℝ) :
  h > 0 →
  let leg := h / Real.sqrt 2
  let area := (1 / 2) * leg * leg
  let median_area := area / 2
  (h = 16) → median_area = 32 := by sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_median_area_l1887_188745


namespace NUMINAMATH_CALUDE_min_colors_is_three_l1887_188702

/-- Represents a coloring of a 5x5 grid -/
def Coloring := Fin 5 → Fin 5 → ℕ

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Fin 5 × Fin 5) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x3 - x1) * (y2 - y1) = (x2 - x1) * (y3 - y1)

/-- Checks if a coloring is valid (no three same-colored points are collinear) -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ p1 p2 p3 : Fin 5 × Fin 5,
    collinear p1 p2 p3 →
    (c p1.1 p1.2 = c p2.1 p2.2 ∧ c p2.1 p2.2 = c p3.1 p3.2) →
    p1 = p2 ∨ p2 = p3 ∨ p3 = p1

/-- The main theorem: the minimum number of colors for a valid coloring is 3 -/
theorem min_colors_is_three :
  (∃ (c : Coloring), valid_coloring c ∧ (∀ i j, c i j < 3)) ∧
  (∀ (c : Coloring), valid_coloring c → ∃ i j, c i j ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_colors_is_three_l1887_188702


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1887_188776

theorem quadratic_equation_real_roots (k : ℕ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1887_188776


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1887_188770

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  q = -3 →                         -- given common ratio
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1887_188770


namespace NUMINAMATH_CALUDE_sum_mod_seven_l1887_188790

theorem sum_mod_seven : (1001 + 1002 + 1003 + 1004 + 1005) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l1887_188790


namespace NUMINAMATH_CALUDE_symmetric_hexagon_relationship_l1887_188703

/-- A hexagon that is both inscribed and circumscribed, and symmetric about the perpendicular bisector of one of its sides. -/
structure SymmetricHexagon where
  R : ℝ  -- radius of circumscribed circle
  r : ℝ  -- radius of inscribed circle
  c : ℝ  -- distance between centers of circles
  R_pos : 0 < R
  r_pos : 0 < r
  c_pos : 0 < c
  inscribed : True  -- represents that the hexagon is inscribed
  circumscribed : True  -- represents that the hexagon is circumscribed
  symmetric : True  -- represents that the hexagon is symmetric about the perpendicular bisector of one of its sides

/-- The relationship between R, r, and c for a symmetric hexagon -/
theorem symmetric_hexagon_relationship (h : SymmetricHexagon) :
  3 * (h.R^2 - h.c^2)^4 - 4 * h.r^2 * (h.R^2 - h.c^2)^2 * (h.R^2 + h.c^2) - 16 * h.R^2 * h.c^2 * h.r^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_hexagon_relationship_l1887_188703


namespace NUMINAMATH_CALUDE_max_k_value_l1887_188744

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^2 * (x^2/y^2 + y^2/x^2) + 2*k * (x/y + y/x)) :
  k ≤ (-1 + Real.sqrt 56) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1887_188744


namespace NUMINAMATH_CALUDE_number_calculation_l1887_188739

theorem number_calculation (n : ℝ) : (0.1 * 0.2 * 0.35 * 0.4 * n = 84) → n = 300000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1887_188739


namespace NUMINAMATH_CALUDE_sarah_fish_difference_l1887_188763

/-- The number of fish each person has -/
structure FishCounts where
  billy : ℕ
  tony : ℕ
  sarah : ℕ
  bobby : ℕ

/-- The conditions of the problem -/
def fish_problem (fc : FishCounts) : Prop :=
  fc.billy = 10 ∧
  fc.tony = 3 * fc.billy ∧
  fc.bobby = 2 * fc.sarah ∧
  fc.sarah > fc.tony ∧
  fc.billy + fc.tony + fc.sarah + fc.bobby = 145

theorem sarah_fish_difference (fc : FishCounts) :
  fish_problem fc → fc.sarah - fc.tony = 5 := by
  sorry

end NUMINAMATH_CALUDE_sarah_fish_difference_l1887_188763


namespace NUMINAMATH_CALUDE_angle_B_is_60_degrees_l1887_188795

-- Define a structure for a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem angle_B_is_60_degrees (t : Triangle) 
  (h1 : t.B = 2 * t.A)
  (h2 : t.C = 3 * t.A)
  (h3 : t.A + t.B + t.C = 180) : 
  t.B = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_60_degrees_l1887_188795


namespace NUMINAMATH_CALUDE_total_apples_bought_l1887_188762

theorem total_apples_bought (num_men num_women : ℕ) (apples_per_man : ℕ) (extra_apples_per_woman : ℕ) : 
  num_men = 2 → 
  num_women = 3 → 
  apples_per_man = 30 → 
  extra_apples_per_woman = 20 →
  num_men * apples_per_man + num_women * (apples_per_man + extra_apples_per_woman) = 210 := by
  sorry

#check total_apples_bought

end NUMINAMATH_CALUDE_total_apples_bought_l1887_188762


namespace NUMINAMATH_CALUDE_mixture_ratio_change_l1887_188758

def initial_ratio : ℚ := 3 / 2
def initial_total : ℚ := 20
def added_water : ℚ := 10

def milk : ℚ := initial_total * (initial_ratio / (1 + initial_ratio))
def water : ℚ := initial_total * (1 / (1 + initial_ratio))

def new_water : ℚ := water + added_water
def new_ratio : ℚ := milk / new_water

theorem mixture_ratio_change :
  new_ratio = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_mixture_ratio_change_l1887_188758


namespace NUMINAMATH_CALUDE_inequality_solution_l1887_188743

theorem inequality_solution (x : ℝ) : 
  (x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0 ↔ 
  x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ x > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1887_188743


namespace NUMINAMATH_CALUDE_stair_climbing_time_l1887_188791

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The time taken to climb stairs -/
theorem stair_climbing_time : arithmetic_sum 15 8 7 = 273 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l1887_188791


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1887_188753

theorem arithmetic_evaluation : 12 / 4 - 3 - 6 + 3 * 5 = 9 := by sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1887_188753


namespace NUMINAMATH_CALUDE_simplify_fraction_l1887_188761

theorem simplify_fraction : 5 * (14 / 3) * (21 / -70) = -35 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1887_188761


namespace NUMINAMATH_CALUDE_new_person_weight_l1887_188798

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 35 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1887_188798


namespace NUMINAMATH_CALUDE_inequality_proof_l1887_188793

theorem inequality_proof (x y z : ℤ) :
  (x^2 + y^2*z^2) * (y^2 + x^2*z^2) * (z^2 + x^2*y^2) ≥ 8*x*y^2*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1887_188793


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1887_188708

theorem quadratic_form_equivalence :
  let f (x : ℝ) := 2 * x^2 - 8 * x + 3
  let g (x : ℝ) := 2 * (x - 2)^2 - 5
  ∀ x, f x = g x := by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1887_188708


namespace NUMINAMATH_CALUDE_oil_added_to_mixture_l1887_188721

/-- Proves that the amount of oil added to mixture A is 2 kilograms -/
theorem oil_added_to_mixture (mixture_a_weight : ℝ) (oil_percentage : ℝ) (material_b_percentage : ℝ)
  (added_mixture_a : ℝ) (final_material_b_percentage : ℝ) :
  mixture_a_weight = 8 →
  oil_percentage = 0.2 →
  material_b_percentage = 0.8 →
  added_mixture_a = 6 →
  final_material_b_percentage = 0.7 →
  ∃ (x : ℝ),
    x = 2 ∧
    (material_b_percentage * mixture_a_weight + material_b_percentage * added_mixture_a) =
      final_material_b_percentage * (mixture_a_weight + x + added_mixture_a) :=
by sorry

end NUMINAMATH_CALUDE_oil_added_to_mixture_l1887_188721


namespace NUMINAMATH_CALUDE_may_red_yarns_l1887_188757

/-- The number of scarves May can knit using one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May will be able to make -/
def total_scarves : ℕ := 36

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

theorem may_red_yarns : 
  scarves_per_yarn * (blue_yarns + yellow_yarns + red_yarns) = total_scarves := by
  sorry

end NUMINAMATH_CALUDE_may_red_yarns_l1887_188757


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1887_188781

theorem quadratic_roots_problem (m : ℝ) (α β : ℝ) :
  (α > 0 ∧ β > 0) ∧ 
  (α^2 + (2*m - 1)*α + m^2 = 0) ∧ 
  (β^2 + (2*m - 1)*β + m^2 = 0) →
  ((m ≤ 1/4 ∧ m ≠ 0) ∧
   (α^2 + β^2 = 49 → m = -4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1887_188781


namespace NUMINAMATH_CALUDE_sector_central_angle_l1887_188751

/-- Given a sector with radius R and area 2R^2, 
    the radian measure of its central angle is 4. -/
theorem sector_central_angle (R : ℝ) (h : R > 0) :
  let area := 2 * R^2
  let angle := (2 * area) / R^2
  angle = 4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1887_188751


namespace NUMINAMATH_CALUDE_meaningful_expression_l1887_188706

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1887_188706


namespace NUMINAMATH_CALUDE_axis_of_symmetry_for_quadratic_with_roots_1_and_5_l1887_188710

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem axis_of_symmetry_for_quadratic_with_roots_1_and_5 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, quadratic a b c x = 0 ↔ x = 1 ∨ x = 5) →
  (∃ k, ∀ x, quadratic a b c (k + x) = quadratic a b c (k - x)) ∧
  (∀ k, (∀ x, quadratic a b c (k + x) = quadratic a b c (k - x)) → k = 3) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_for_quadratic_with_roots_1_and_5_l1887_188710


namespace NUMINAMATH_CALUDE_cube_root_2450_l1887_188731

theorem cube_root_2450 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (2450 : ℝ)^(1/3) = a * b^(1/3) ∧ 
  (∀ (c d : ℕ), c > 0 → d > 0 → (2450 : ℝ)^(1/3) = c * d^(1/3) → d ≥ b) ∧
  a = 35 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_2450_l1887_188731


namespace NUMINAMATH_CALUDE_two_players_percentage_of_goals_l1887_188796

def total_goals : ℕ := 300
def player_goals : ℕ := 30
def num_players : ℕ := 2

theorem two_players_percentage_of_goals :
  (player_goals * num_players : ℚ) / total_goals * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_players_percentage_of_goals_l1887_188796


namespace NUMINAMATH_CALUDE_rhombus_diagonal_sum_squares_l1887_188723

/-- A rhombus with side length 2 has the sum of squares of its diagonals equal to 16 -/
theorem rhombus_diagonal_sum_squares (d₁ d₂ : ℝ) : 
  d₁ > 0 → d₂ > 0 → (d₁ / 2) ^ 2 + (d₂ / 2) ^ 2 = 2 ^ 2 → d₁ ^ 2 + d₂ ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_sum_squares_l1887_188723


namespace NUMINAMATH_CALUDE_equation_solutions_l1887_188725

theorem equation_solutions :
  let f : ℝ → ℝ → ℝ := λ x y => y^4 + 4*y^2*x - 11*y^2 + 4*x*y - 8*y + 8*x^2 - 40*x + 52
  ∀ x y : ℝ, f x y = 0 ↔ (x = 1 ∧ y = 2) ∨ (x = 5/2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1887_188725


namespace NUMINAMATH_CALUDE_jenny_sweets_problem_l1887_188707

theorem jenny_sweets_problem : ∃ n : ℕ+, 
  5 ∣ n ∧ 6 ∣ n ∧ ¬(12 ∣ n) ∧ n = 90 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sweets_problem_l1887_188707


namespace NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l1887_188733

/-- Represents a seating arrangement in a hockey arena --/
structure ArenaSeating where
  total_students : ℕ
  seats_per_row : ℕ
  max_students_per_school : ℕ
  same_row_constraint : Bool

/-- Calculates the minimum number of rows required for the given seating arrangement --/
def min_rows_required (seating : ArenaSeating) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rows required for the given problem --/
theorem min_rows_for_hockey_arena :
  let seating := ArenaSeating.mk 2016 168 40 true
  min_rows_required seating = 15 :=
sorry

end NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l1887_188733


namespace NUMINAMATH_CALUDE_max_cone_volume_in_sphere_l1887_188712

/-- The maximum volume of a cone formed by a circular section of a sphere --/
theorem max_cone_volume_in_sphere (R : ℝ) (h : R = 9) : 
  ∃ (V : ℝ), V = 54 * Real.sqrt 3 * Real.pi ∧ 
  ∀ (r h : ℝ), r^2 + h^2 = R^2 → 
  (1/3 : ℝ) * Real.pi * r^2 * h ≤ V := by
  sorry

end NUMINAMATH_CALUDE_max_cone_volume_in_sphere_l1887_188712


namespace NUMINAMATH_CALUDE_invalid_atomic_number_difference_l1887_188738

/-- Represents a period in the periodic table -/
inductive Period
| Second
| Third
| Fourth
| Fifth
| Sixth

/-- Represents an element in the periodic table -/
structure Element where
  atomicNumber : ℕ
  period : Period

/-- The difference in atomic numbers between elements in groups VIA and IA in the same period -/
def atomicNumberDifference (p : Period) : ℕ :=
  match p with
  | Period.Second => 5
  | Period.Third => 5
  | Period.Fourth => 15
  | Period.Fifth => 15
  | Period.Sixth => 29

theorem invalid_atomic_number_difference (X Y : Element) 
  (h1 : X.period = Y.period)
  (h2 : Y.atomicNumber = X.atomicNumber + atomicNumberDifference X.period) :
  Y.atomicNumber - X.atomicNumber ≠ 9 := by
  sorry

#check invalid_atomic_number_difference

end NUMINAMATH_CALUDE_invalid_atomic_number_difference_l1887_188738


namespace NUMINAMATH_CALUDE_train_length_l1887_188748

/-- Calculates the length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 5 → speed_kmh * (1000 / 3600) * time_s = 100 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1887_188748


namespace NUMINAMATH_CALUDE_expression_evaluation_l1887_188780

theorem expression_evaluation (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1887_188780


namespace NUMINAMATH_CALUDE_ticket_price_uniqueness_l1887_188720

theorem ticket_price_uniqueness :
  ∃! x : ℕ, x > 0 ∧ 
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * x = 60 ∧ b * x = 90 ∧ c * x = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_ticket_price_uniqueness_l1887_188720


namespace NUMINAMATH_CALUDE_quadratic_rewrite_proof_l1887_188792

theorem quadratic_rewrite_proof :
  ∃ (a b c : ℚ), 
    (∀ k, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧
    (c + 3 * b = -49 / 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_proof_l1887_188792


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1887_188717

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3*y - 10 → y ≥ 6 ∧ 6 < 3*6 - 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1887_188717


namespace NUMINAMATH_CALUDE_constant_is_arithmetic_l1887_188785

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem constant_is_arithmetic :
  ∀ a : ℕ → ℝ, is_constant_sequence a → is_arithmetic_sequence a :=
by
  sorry

end NUMINAMATH_CALUDE_constant_is_arithmetic_l1887_188785


namespace NUMINAMATH_CALUDE_factorization_equality_l1887_188777

theorem factorization_equality (x y : ℝ) : -4 * x^2 + y^2 = (y - 2*x) * (y + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1887_188777


namespace NUMINAMATH_CALUDE_jake_peaches_l1887_188773

/-- Given the number of peaches each person has, prove Jake has 17 peaches -/
theorem jake_peaches (jill steven jake : ℕ) 
  (h1 : jake + 6 = steven)
  (h2 : steven = jill + 18)
  (h3 : jill = 5) : 
  jake = 17 := by
sorry

end NUMINAMATH_CALUDE_jake_peaches_l1887_188773


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1887_188765

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1887_188765


namespace NUMINAMATH_CALUDE_special_numbers_count_l1887_188772

/-- Sum of digits of a positive integer -/
def heartsuit (n : ℕ+) : ℕ :=
  sorry

/-- Counts the number of three-digit positive integers x such that heartsuit(heartsuit(x)) = 5 -/
def count_special_numbers : ℕ :=
  sorry

/-- Theorem stating that there are exactly 60 three-digit positive integers x 
    such that heartsuit(heartsuit(x)) = 5 -/
theorem special_numbers_count : count_special_numbers = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_count_l1887_188772


namespace NUMINAMATH_CALUDE_last_digit_of_max_value_l1887_188742

/-- Represents the operation of replacing two numbers with their product plus one -/
def combine (a b : ℕ) : ℕ := a * b + 1

/-- The maximum value after performing the combine operation 127 times on 128 ones -/
def max_final_value : ℕ := sorry

/-- The problem statement -/
theorem last_digit_of_max_value :
  (max_final_value % 10) = 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_max_value_l1887_188742


namespace NUMINAMATH_CALUDE_area_of_composite_rectangle_l1887_188778

/-- The area of a rectangle formed by four identical smaller rectangles --/
theorem area_of_composite_rectangle (short_side : ℝ) : 
  short_side = 7 →
  (2 * short_side) * (2 * short_side) = 392 := by
  sorry

end NUMINAMATH_CALUDE_area_of_composite_rectangle_l1887_188778


namespace NUMINAMATH_CALUDE_quadratic_value_l1887_188750

/-- A quadratic function with specific properties -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 2)^2 + 7

theorem quadratic_value (a : ℝ) :
  (∀ x, f a x ≤ 7) →  -- Maximum value condition
  (f a 2 = 7) →       -- Maximum occurs at x = 2
  (f a 0 = -7) →      -- Passes through (0, -7)
  (a < 0) →           -- Implied by maximum condition
  (f a 5 = -24.5) :=  -- The value at x = 5
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l1887_188750


namespace NUMINAMATH_CALUDE_sticker_redistribution_l1887_188755

/-- Represents the number of stickers each person has -/
structure StickerCount where
  noah : ℕ
  emma : ℕ
  liam : ℕ

/-- Represents the initial distribution of stickers -/
def initial_distribution (n : ℕ) : StickerCount :=
  { noah := n
  , emma := 3 * n
  , liam := 12 * n }

/-- The fraction of stickers Liam should give to Emma -/
def fraction_to_emma : ℚ := 7 / 36

theorem sticker_redistribution (n : ℕ) :
  let init := initial_distribution n
  let total := init.noah + init.emma + init.liam
  let each_should_have := total / 3
  fraction_to_emma = (each_should_have - init.emma) / init.liam := by
  sorry

end NUMINAMATH_CALUDE_sticker_redistribution_l1887_188755


namespace NUMINAMATH_CALUDE_chord_intersection_parameter_l1887_188701

/-- Given a line and a circle, prove that the parameter a equals 1 when they intersect to form a chord of length √2. -/
theorem chord_intersection_parameter (a : ℝ) : a > 0 → ∃ (x y : ℝ),
  (x + y + a = 0) ∧ (x^2 + y^2 = a) ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), (x1 + y1 + a = 0) ∧ (x2 + y2 + a = 0) ∧
                        (x1^2 + y1^2 = a) ∧ (x2^2 + y2^2 = a) ∧
                        ((x1 - x2)^2 + (y1 - y2)^2 = 2)) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_parameter_l1887_188701


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1887_188705

/-- Arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Geometric sequence -/
def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem arithmetic_geometric_inequality (b₁ q : ℝ) (m : ℕ) 
  (h₁ : b₁ > 0) 
  (h₂ : m > 0) 
  (h₃ : 1 < q) 
  (h₄ : q < (2 : ℝ)^(1 / m)) :
  ∃ d : ℝ, ∀ n : ℕ, 2 ≤ n ∧ n ≤ m + 1 → 
    |arithmetic_sequence b₁ d n - geometric_sequence b₁ q n| ≤ b₁ ∧
    b₁ * (q^m - 2) / m ≤ d ∧ d ≤ b₁ * q^m / m :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1887_188705


namespace NUMINAMATH_CALUDE_abs_negative_seven_l1887_188797

theorem abs_negative_seven : |(-7 : ℤ)| = 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_seven_l1887_188797


namespace NUMINAMATH_CALUDE_linear_case_quadratic_case_l1887_188709

-- Define the coefficient of x^2
def coeff_x2 (k : ℝ) : ℝ := k^2 - 1

-- Define the coefficient of x
def coeff_x (k : ℝ) : ℝ := 2*(k + 1)

-- Define the constant term
def const_term (k : ℝ) : ℝ := 3*(k - 1)

-- Theorem for the linear case
theorem linear_case (k : ℝ) : 
  (coeff_x2 k = 0 ∧ coeff_x k ≠ 0) ↔ k = 1 := by sorry

-- Theorem for the quadratic case
theorem quadratic_case (k : ℝ) :
  coeff_x2 k ≠ 0 ↔ k ≠ 1 ∧ k ≠ -1 := by sorry

end NUMINAMATH_CALUDE_linear_case_quadratic_case_l1887_188709


namespace NUMINAMATH_CALUDE_harkamal_payment_l1887_188787

def grapes_qty : ℝ := 8
def grapes_price : ℝ := 80
def mangoes_qty : ℝ := 9
def mangoes_price : ℝ := 55
def apples_qty : ℝ := 6
def apples_price : ℝ := 120
def oranges_qty : ℝ := 4
def oranges_price : ℝ := 75
def apple_discount : ℝ := 0.1
def sales_tax : ℝ := 0.05

def total_cost : ℝ :=
  grapes_qty * grapes_price +
  mangoes_qty * mangoes_price +
  apples_qty * apples_price * (1 - apple_discount) +
  oranges_qty * oranges_price

def final_cost : ℝ := total_cost * (1 + sales_tax)

theorem harkamal_payment : final_cost = 2187.15 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l1887_188787


namespace NUMINAMATH_CALUDE_rectangle_area_l1887_188732

/-- Given a rectangle with width 4 inches and perimeter 30 inches, prove its area is 44 square inches -/
theorem rectangle_area (width : ℝ) (perimeter : ℝ) : 
  width = 4 → perimeter = 30 → width * ((perimeter / 2) - width) = 44 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1887_188732


namespace NUMINAMATH_CALUDE_inequality_proof_l1887_188724

theorem inequality_proof :
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + b) ≤ (1 / 4) * (1 / a + 1 / b)) ∧
  (∀ x₁ x₂ x₃ : ℝ, x₁ > 0 → x₂ > 0 → x₃ > 0 → 1 / x₁ + 1 / x₂ + 1 / x₃ = 1 →
    (x₁ + x₂ + x₃) / (x₁ * x₃ + x₃ * x₂) + 
    (x₁ + x₂ + x₃) / (x₁ * x₂ + x₃ * x₁) + 
    (x₁ + x₂ + x₃) / (x₂ * x₁ + x₃ * x₂) ≤ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1887_188724


namespace NUMINAMATH_CALUDE_find_n_l1887_188719

theorem find_n (a b c n : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_n : 0 < n)
  (eq1 : (a + b) / a = 3)
  (eq2 : (b + c) / b = 4)
  (eq3 : (c + a) / c = n) :
  n = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_find_n_l1887_188719


namespace NUMINAMATH_CALUDE_isosceles_at_second_iteration_l1887_188734

/-- Represents a triangle with angles α, β, and γ -/
structure Triangle where
  α : Real
  β : Real
  γ : Real

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { α := t.β, β := t.α, γ := 90 }

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  t.α = t.β ∨ t.β = t.γ ∨ t.γ = t.α

/-- The initial triangle A₀B₀C₀ -/
def A₀B₀C₀ : Triangle :=
  { α := 30, β := 60, γ := 90 }

/-- Generates the nth triangle in the sequence -/
def nthTriangle (n : Nat) : Triangle :=
  match n with
  | 0 => A₀B₀C₀
  | n + 1 => nextTriangle (nthTriangle n)

theorem isosceles_at_second_iteration :
  ∃ n : Nat, n > 0 ∧ isIsosceles (nthTriangle n) ∧ ∀ m : Nat, 0 < m ∧ m < n → ¬isIsosceles (nthTriangle m) :=
  sorry

end NUMINAMATH_CALUDE_isosceles_at_second_iteration_l1887_188734


namespace NUMINAMATH_CALUDE_octal_subtraction_correct_l1887_188746

/-- Represents a number in base 8 -/
def OctalNumber := List Nat

/-- Converts a list of digits in base 8 to a natural number -/
def octal_to_nat (x : OctalNumber) : Nat :=
  x.foldr (fun digit acc => acc * 8 + digit) 0

/-- Subtracts two octal numbers -/
def octal_subtract (x y : OctalNumber) : OctalNumber :=
  sorry -- Implementation of octal subtraction

theorem octal_subtraction_correct :
  octal_subtract [7, 3, 2, 4] [3, 6, 5, 7] = [4, 4, 4, 5] :=
by sorry

end NUMINAMATH_CALUDE_octal_subtraction_correct_l1887_188746


namespace NUMINAMATH_CALUDE_jones_clothing_count_l1887_188747

def pants_count : ℕ := 40
def shirts_per_pants : ℕ := 6
def ties_per_pants : ℕ := 5
def socks_per_shirt : ℕ := 3

def total_clothing : ℕ := 
  pants_count + 
  (pants_count * shirts_per_pants) + 
  (pants_count * ties_per_pants) + 
  (pants_count * shirts_per_pants * socks_per_shirt)

theorem jones_clothing_count : total_clothing = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jones_clothing_count_l1887_188747


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l1887_188722

theorem fraction_sum_proof : 
  (1 / 12 : ℚ) + (2 / 12 : ℚ) + (3 / 12 : ℚ) + (4 / 12 : ℚ) + (5 / 12 : ℚ) + 
  (6 / 12 : ℚ) + (7 / 12 : ℚ) + (8 / 12 : ℚ) + (9 / 12 : ℚ) + (65 / 12 : ℚ) + 
  (3 / 4 : ℚ) = 119 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l1887_188722


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1887_188771

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Bᶜ) = {x | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1887_188771


namespace NUMINAMATH_CALUDE_line_curve_hyperbola_l1887_188784

variable (a b : ℝ)

theorem line_curve_hyperbola (h1 : a ≠ 0) (h2 : b ≠ 0) :
  ∃ (x y : ℝ), (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) →
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧ ∀ (x y : ℝ), x^2 / A - y^2 / B = 1 :=
sorry

end NUMINAMATH_CALUDE_line_curve_hyperbola_l1887_188784


namespace NUMINAMATH_CALUDE_grandfather_age_ratio_l1887_188774

/-- Given the current ages of Xiao Hong and her grandfather, 
    prove the ratio of their ages last year -/
theorem grandfather_age_ratio (xiao_hong_age grandfather_age : ℕ) 
  (h1 : xiao_hong_age = 8) 
  (h2 : grandfather_age = 64) : 
  (grandfather_age - 1) / (xiao_hong_age - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_age_ratio_l1887_188774


namespace NUMINAMATH_CALUDE_david_pushups_count_l1887_188794

/-- The number of push-ups done by Zachary -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups done by David -/
def david_pushups : ℕ := 3 * zachary_pushups

theorem david_pushups_count : david_pushups = 57 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_count_l1887_188794


namespace NUMINAMATH_CALUDE_particle_average_velocity_l1887_188740

/-- Given a particle with motion law s = t^2 + 3, its average velocity 
    during the time interval (3, 3+Δx) is equal to 6 + Δx. -/
theorem particle_average_velocity (Δx : ℝ) : 
  let s (t : ℝ) := t^2 + 3
  ((s (3 + Δx) - s 3) / Δx) = 6 + Δx :=
sorry

end NUMINAMATH_CALUDE_particle_average_velocity_l1887_188740


namespace NUMINAMATH_CALUDE_number_equation_l1887_188730

theorem number_equation (x : ℚ) : (x + 20 / 90) * 90 = 4520 ↔ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1887_188730


namespace NUMINAMATH_CALUDE_x35x_divisible_by_18_l1887_188727

def is_single_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 350 + x

theorem x35x_divisible_by_18 : 
  ∃! (x : ℕ), is_single_digit x ∧ (four_digit_number x) % 18 = 0 ∧ x = 8 :=
sorry

end NUMINAMATH_CALUDE_x35x_divisible_by_18_l1887_188727


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1887_188769

/-- Given a sphere whose surface area increases by 4π cm² when cut in half,
    prove that its original surface area was 8π cm². -/
theorem sphere_surface_area (R : ℝ) (h : 2 * Real.pi * R^2 = 4 * Real.pi) :
  4 * Real.pi * R^2 = 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1887_188769


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1887_188713

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1887_188713


namespace NUMINAMATH_CALUDE_initial_water_percentage_is_70_l1887_188764

/-- The percentage of liquid X in solution Y -/
def liquid_x_percentage : ℝ := 30

/-- The initial mass of solution Y in kg -/
def initial_mass : ℝ := 8

/-- The mass of water that evaporates in kg -/
def evaporated_water : ℝ := 3

/-- The mass of solution Y added after evaporation in kg -/
def added_solution : ℝ := 3

/-- The percentage of liquid X in the new solution -/
def new_liquid_x_percentage : ℝ := 41.25

/-- The initial percentage of water in solution Y -/
def initial_water_percentage : ℝ := 100 - liquid_x_percentage

theorem initial_water_percentage_is_70 :
  initial_water_percentage = 70 :=
sorry

end NUMINAMATH_CALUDE_initial_water_percentage_is_70_l1887_188764


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l1887_188728

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def point_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_lines_sum (a b c : ℝ) : 
  perpendicular (-a/4) (2/5) →
  point_on_line 1 c a 4 (-2) →
  point_on_line 1 c 2 (-5) b →
  a + b + c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l1887_188728


namespace NUMINAMATH_CALUDE_sequence_properties_l1887_188729

def a : ℕ → ℕ
  | n => if n % 2 = 1 then n else 2 * 3^((n / 2) - 1)

def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

theorem sequence_properties :
  (∀ k : ℕ, a (2 * k + 1) = a 1 + k * (a 3 - a 1)) ∧
  (∀ k : ℕ, k > 0 → a (2 * k) = a 2 * (a 4 / a 2) ^ (k - 1)) ∧
  (S 5 = 2 * a 4 + a 5) ∧
  (a 9 = a 3 + a 4) →
  (∀ n : ℕ, n > 0 → a n = if n % 2 = 1 then n else 2 * 3^((n / 2) - 1)) ∧
  (∀ m : ℕ, m > 0 → (a m * a (m + 1) = a (m + 2)) ↔ m = 2) ∧
  (∀ m : ℕ, m > 0 → (∃ k : ℕ, k > 0 ∧ S (2 * m) / S (2 * m - 1) = a k) ↔ (m = 1 ∨ m = 2)) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l1887_188729


namespace NUMINAMATH_CALUDE_solve_baseball_card_problem_l1887_188716

def baseball_card_problem (initial_cards : ℕ) (final_cards : ℕ) : Prop :=
  ∃ (cards_to_peter : ℕ),
    let cards_after_maria := initial_cards - (initial_cards + 1) / 2
    let cards_before_paul := cards_after_maria - cards_to_peter
    3 * cards_before_paul = final_cards ∧
    cards_to_peter = 1

theorem solve_baseball_card_problem :
  baseball_card_problem 15 18 :=
sorry

end NUMINAMATH_CALUDE_solve_baseball_card_problem_l1887_188716


namespace NUMINAMATH_CALUDE_remaining_money_l1887_188700

def initial_amount : ℚ := 50
def shirt_cost : ℚ := 7.85
def meal_cost : ℚ := 15.49
def magazine_cost : ℚ := 6.13
def debt_payment : ℚ := 3.27
def cd_cost : ℚ := 11.75

theorem remaining_money :
  initial_amount - (shirt_cost + meal_cost + magazine_cost + debt_payment + cd_cost) = 5.51 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l1887_188700


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1887_188786

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1887_188786


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1887_188768

-- Define the displacement function
def s (t : ℝ) : ℝ := 100 * t - 5 * t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := 100 - 10 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  ∀ t : ℝ, 0 < t → t < 20 → v 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1887_188768


namespace NUMINAMATH_CALUDE_debby_yoyo_tickets_debby_yoyo_tickets_proof_l1887_188752

/-- Theorem: Debby's yoyo ticket expenditure --/
theorem debby_yoyo_tickets : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun hat_tickets stuffed_animal_tickets total_tickets yoyo_tickets =>
    hat_tickets = 2 ∧ 
    stuffed_animal_tickets = 10 ∧ 
    total_tickets = 14 ∧ 
    yoyo_tickets + hat_tickets + stuffed_animal_tickets = total_tickets →
    yoyo_tickets = 2

/-- Proof of the theorem --/
theorem debby_yoyo_tickets_proof : 
  debby_yoyo_tickets 2 10 14 2 := by
  sorry

end NUMINAMATH_CALUDE_debby_yoyo_tickets_debby_yoyo_tickets_proof_l1887_188752


namespace NUMINAMATH_CALUDE_john_received_120_l1887_188737

def grandpa_amount : ℕ := 30

def grandma_amount : ℕ := 3 * grandpa_amount

def total_amount : ℕ := grandpa_amount + grandma_amount

theorem john_received_120 : total_amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_received_120_l1887_188737


namespace NUMINAMATH_CALUDE_four_integer_sum_l1887_188718

theorem four_integer_sum (a b c d : ℕ) : 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 →
  a * b * c * d = 14400 →
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧
  Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1 →
  a + b + c + d = 98 := by
sorry

end NUMINAMATH_CALUDE_four_integer_sum_l1887_188718


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l1887_188775

/-- Represents the number of possible line-ups for a basketball team -/
def number_of_lineups (total_players : ℕ) (centers : ℕ) (right_forwards : ℕ) (left_forwards : ℕ) (right_guards : ℕ) (flexible_guards : ℕ) : ℕ :=
  let guard_combinations := flexible_guards * flexible_guards
  guard_combinations * centers * right_forwards * left_forwards

/-- Theorem stating the number of possible line-ups for the given team composition -/
theorem basketball_lineup_count :
  number_of_lineups 10 2 2 2 1 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l1887_188775


namespace NUMINAMATH_CALUDE_students_wanting_fruit_l1887_188749

theorem students_wanting_fruit (red_apples green_apples extra_apples : ℕ) :
  red_apples = 43 →
  green_apples = 32 →
  extra_apples = 73 →
  (red_apples + green_apples + extra_apples) - (red_apples + green_apples) = extra_apples :=
by sorry

end NUMINAMATH_CALUDE_students_wanting_fruit_l1887_188749


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l1887_188766

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

/-- Theorem stating the total number of elementary students in Lansing -/
theorem lansing_elementary_students : total_students = 6175 := by
  sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l1887_188766


namespace NUMINAMATH_CALUDE_rectangle_width_l1887_188704

theorem rectangle_width (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) : 
  square_side = 700 →
  rect_length = 400 →
  4 * square_side = 2 * (2 * rect_length + 2 * rect_width) →
  rect_width = 300 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1887_188704


namespace NUMINAMATH_CALUDE_box_volume_constraint_l1887_188726

theorem box_volume_constraint (x : ℕ+) : 
  (∃! x, (2 * x + 6 : ℝ) * ((x : ℝ)^3 - 8) * ((x : ℝ)^2 + 4) < 1200) := by
  sorry

end NUMINAMATH_CALUDE_box_volume_constraint_l1887_188726


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_is_two_l1887_188715

/-- A geometric sequence with specified properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a_5_eq_2 : a 5 = 2
  a_6_a_8_eq_8 : a 6 * a 8 = 8

/-- The ratio of differences in a geometric sequence with specific properties is 2 -/
theorem geometric_sequence_ratio_is_two (seq : GeometricSequence) :
  (seq.a 2018 - seq.a 2016) / (seq.a 2014 - seq.a 2012) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_is_two_l1887_188715


namespace NUMINAMATH_CALUDE_boys_without_notebooks_l1887_188756

theorem boys_without_notebooks
  (total_boys : ℕ)
  (total_with_notebooks : ℕ)
  (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24)
  (h2 : total_with_notebooks = 30)
  (h3 : girls_with_notebooks = 18) :
  total_boys - (total_with_notebooks - girls_with_notebooks) = 12 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_notebooks_l1887_188756


namespace NUMINAMATH_CALUDE_line_equation_proof_l1887_188767

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0

-- Define points A, B, M, and N
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry
def point_M : ℝ × ℝ := sorry
def point_N : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem line_equation_proof :
  -- Conditions
  (ellipse point_A.1 point_A.2) ∧
  (ellipse point_B.1 point_B.2) ∧
  (point_A.1 > 0 ∧ point_A.2 > 0) ∧
  (point_B.1 > 0 ∧ point_B.2 > 0) ∧
  (point_M.2 = 0) ∧
  (point_N.1 = 0) ∧
  (distance point_M point_A = distance point_N point_B) ∧
  (distance point_M point_N = 2 * Real.sqrt 3) →
  -- Conclusion
  ∀ x y : ℝ, line_l x y ↔ (x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1887_188767


namespace NUMINAMATH_CALUDE_table_tennis_tournament_l1887_188741

theorem table_tennis_tournament (n : ℕ) (total_matches : ℕ) (withdrawn_players : ℕ) 
  (matches_per_withdrawn : ℕ) (h1 : n = 13) (h2 : total_matches = 50) 
  (h3 : withdrawn_players = 3) (h4 : matches_per_withdrawn = 2) : 
  (n.choose 2) - ((n - withdrawn_players).choose 2) - 
  (withdrawn_players * matches_per_withdrawn) = 1 := by
sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l1887_188741
