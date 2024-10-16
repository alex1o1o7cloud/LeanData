import Mathlib

namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3707_370747

theorem smallest_x_for_perfect_cube (x : ℕ) : x = 36 ↔ 
  (x > 0 ∧ ∃ y : ℕ, 1152 * x = y^3 ∧ ∀ z < x, z > 0 → ¬∃ w : ℕ, 1152 * z = w^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3707_370747


namespace NUMINAMATH_CALUDE_speed_difference_proof_l3707_370724

/-- Proves the speed difference between two vehicles given their travel conditions -/
theorem speed_difference_proof (base_speed : ℝ) (time : ℝ) (total_distance : ℝ) :
  base_speed = 44 →
  time = 4 →
  total_distance = 384 →
  ∃ (speed_diff : ℝ),
    speed_diff > 0 ∧
    total_distance = base_speed * time + (base_speed + speed_diff) * time ∧
    speed_diff = 8 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_proof_l3707_370724


namespace NUMINAMATH_CALUDE_initial_games_count_l3707_370719

theorem initial_games_count (sold : ℕ) (added : ℕ) (final : ℕ) : 
  sold = 68 → added = 47 → final = 74 → 
  ∃ initial : ℕ, initial - sold + added = final ∧ initial = 95 := by
sorry

end NUMINAMATH_CALUDE_initial_games_count_l3707_370719


namespace NUMINAMATH_CALUDE_max_square_plots_l3707_370773

/-- Represents the dimensions of the park -/
structure ParkDimensions where
  width : ℕ
  length : ℕ

/-- Represents the constraints for the park division -/
structure ParkConstraints where
  dimensions : ParkDimensions
  pathwayMaterial : ℕ

/-- Calculates the number of square plots given the number of plots along the width -/
def calculatePlots (n : ℕ) : ℕ := n * (2 * n)

/-- Calculates the total length of pathways given the number of plots along the width -/
def calculatePathwayLength (n : ℕ) : ℕ := 120 * n - 90

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (constraints : ParkConstraints) 
  (h1 : constraints.dimensions.width = 30)
  (h2 : constraints.dimensions.length = 60)
  (h3 : constraints.pathwayMaterial = 2010) :
  ∃ (n : ℕ), calculatePlots n = 578 ∧ 
             calculatePathwayLength n ≤ constraints.pathwayMaterial ∧
             ∀ (m : ℕ), m > n → calculatePathwayLength m > constraints.pathwayMaterial :=
  by sorry


end NUMINAMATH_CALUDE_max_square_plots_l3707_370773


namespace NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_l3707_370757

theorem log_sqrt12_1728sqrt12 : Real.log (1728 * Real.sqrt 12) / Real.log (Real.sqrt 12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_l3707_370757


namespace NUMINAMATH_CALUDE_least_multiple_squared_l3707_370727

theorem least_multiple_squared (X Y : ℕ) : 
  (∃ Y, 3456^2 * X = 6789^2 * Y) ∧ 
  (∀ Z, Z < X → ¬∃ W, 3456^2 * Z = 6789^2 * W) →
  X = 290521 := by
sorry

end NUMINAMATH_CALUDE_least_multiple_squared_l3707_370727


namespace NUMINAMATH_CALUDE_coaches_next_meeting_l3707_370702

/-- The number of days between Ella's coaching sessions -/
def ella_days : ℕ := 5

/-- The number of days between Felix's coaching sessions -/
def felix_days : ℕ := 9

/-- The number of days between Greta's coaching sessions -/
def greta_days : ℕ := 8

/-- The number of days between Harry's coaching sessions -/
def harry_days : ℕ := 11

/-- The number of days until all coaches work together again -/
def days_until_next_meeting : ℕ := 3960

theorem coaches_next_meeting :
  Nat.lcm ella_days (Nat.lcm felix_days (Nat.lcm greta_days harry_days)) = days_until_next_meeting := by
  sorry

end NUMINAMATH_CALUDE_coaches_next_meeting_l3707_370702


namespace NUMINAMATH_CALUDE_specific_cyclic_quadrilateral_radii_l3707_370734

/-- A cyclic quadrilateral with given side lengths --/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  cyclic : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The radius of the circumscribed circle of a cyclic quadrilateral --/
def circumradius (q : CyclicQuadrilateral) : ℝ := sorry

/-- The radius of the inscribed circle of a cyclic quadrilateral --/
def inradius (q : CyclicQuadrilateral) : ℝ := sorry

/-- Theorem about the radii of circumscribed and inscribed circles for a specific cyclic quadrilateral --/
theorem specific_cyclic_quadrilateral_radii :
  ∃ (q : CyclicQuadrilateral),
    q.a = 36 ∧ q.b = 91 ∧ q.c = 315 ∧ q.d = 260 ∧
    circumradius q = 162.5 ∧
    inradius q = 140 / 3 := by sorry

end NUMINAMATH_CALUDE_specific_cyclic_quadrilateral_radii_l3707_370734


namespace NUMINAMATH_CALUDE_athul_rowing_time_l3707_370778

theorem athul_rowing_time (upstream_distance : ℝ) (downstream_distance : ℝ) (stream_speed : ℝ) :
  upstream_distance = 16 →
  downstream_distance = 24 →
  stream_speed = 1 →
  ∃ (rowing_speed : ℝ),
    rowing_speed > stream_speed ∧
    (upstream_distance / (rowing_speed - stream_speed) = downstream_distance / (rowing_speed + stream_speed)) ∧
    (upstream_distance / (rowing_speed - stream_speed) = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_athul_rowing_time_l3707_370778


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3707_370788

theorem arithmetic_evaluation : 5 + 12 / 3 - 3^2 + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3707_370788


namespace NUMINAMATH_CALUDE_symmetric_point_origin_symmetric_point_negative_two_five_l3707_370714

def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem symmetric_point_origin (x y : ℝ) : 
  symmetric_point x y = (-x, -y) := by sorry

theorem symmetric_point_negative_two_five : 
  symmetric_point (-2) 5 = (2, -5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_origin_symmetric_point_negative_two_five_l3707_370714


namespace NUMINAMATH_CALUDE_non_intercept_line_conditions_l3707_370772

/-- A line that cannot be converted to intercept form -/
def NonInterceptLine (m : ℝ) : Prop :=
  ∃ (x y : ℝ), m * (x + y - 1) + (3 * y - 4 * x + 5) = 0 ∧
  ((m - 4 = 0) ∨ (m + 3 = 0) ∨ (-m + 5 = 0))

/-- The theorem stating the conditions for a line that cannot be converted to intercept form -/
theorem non_intercept_line_conditions :
  ∀ m : ℝ, NonInterceptLine m ↔ (m = 4 ∨ m = -3 ∨ m = 5) :=
by sorry

end NUMINAMATH_CALUDE_non_intercept_line_conditions_l3707_370772


namespace NUMINAMATH_CALUDE_g_prime_symmetry_l3707_370791

open Function Real

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- Assume f' is the derivative of f and g' is the derivative of g
variable (hf : ∀ x, HasDerivAt f (f' x) x)
variable (hg : ∀ x, HasDerivAt g (g' x) x)

-- Define the conditions
variable (h1 : ∀ x, f x + g' x = 5)
variable (h2 : ∀ x, f (2 - x) - g' (2 + x) = 5)
variable (h3 : Odd g)

-- State the theorem
theorem g_prime_symmetry (x : ℝ) : g' (8 - x) = g' x := sorry

end NUMINAMATH_CALUDE_g_prime_symmetry_l3707_370791


namespace NUMINAMATH_CALUDE_sum_distinct_prime_factors_156000_l3707_370784

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_distinct_prime_factors_156000 :
  sum_of_distinct_prime_factors 156000 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_factors_156000_l3707_370784


namespace NUMINAMATH_CALUDE_minimum_toddlers_l3707_370730

theorem minimum_toddlers (total_teeth : ℕ) (max_pair_teeth : ℕ) (h1 : total_teeth = 90) (h2 : max_pair_teeth = 9) :
  ∃ (n : ℕ), n ≥ 23 ∧
  (∀ (m : ℕ), m < n →
    ¬∃ (teeth_distribution : Fin m → ℕ),
      (∀ i j : Fin m, i ≠ j → teeth_distribution i + teeth_distribution j ≤ max_pair_teeth) ∧
      (Finset.sum (Finset.univ : Finset (Fin m)) teeth_distribution = total_teeth)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_toddlers_l3707_370730


namespace NUMINAMATH_CALUDE_exists_cube_with_2014_prime_points_l3707_370708

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Predicate to check if a number is prime -/
def isPrime (n : ℤ) : Prop := sorry

/-- Predicate to check if a point is in the first octant -/
def isFirstOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y > 0 ∧ p.z > 0

/-- Predicate to check if a point has all prime coordinates -/
def isPrimePoint (p : Point3D) : Prop :=
  isPrime p.x ∧ isPrime p.y ∧ isPrime p.z

/-- Definition of a cube in 3D space -/
structure Cube where
  corner : Point3D
  edgeLength : ℤ

/-- Predicate to check if a point is inside a cube -/
def isInsideCube (p : Point3D) (c : Cube) : Prop :=
  c.corner.x ≤ p.x ∧ p.x < c.corner.x + c.edgeLength ∧
  c.corner.y ≤ p.y ∧ p.y < c.corner.y + c.edgeLength ∧
  c.corner.z ≤ p.z ∧ p.z < c.corner.z + c.edgeLength

/-- The main theorem to be proved -/
theorem exists_cube_with_2014_prime_points :
  ∃ (c : Cube), c.edgeLength = 2014 ∧
    isFirstOctant c.corner ∧
    (∃ (points : Finset Point3D),
      points.card = 2014 ∧
      (∀ p ∈ points, isPrimePoint p ∧ isInsideCube p c) ∧
      (∀ p : Point3D, isPrimePoint p ∧ isInsideCube p c → p ∈ points)) :=
sorry

end NUMINAMATH_CALUDE_exists_cube_with_2014_prime_points_l3707_370708


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3707_370795

/-- Condition p: 0 < x < 2 -/
def p (x : ℝ) : Prop := 0 < x ∧ x < 2

/-- Condition q: -1 < x < 3 -/
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

/-- p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3707_370795


namespace NUMINAMATH_CALUDE_vector_problem_l3707_370751

theorem vector_problem (α β : ℝ) (a b c : ℝ × ℝ) :
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  c = (1, 2) →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 2 / 2 →
  ∃ (k : ℝ), a = k • c →
  0 < β →
  β < α →
  α < Real.pi / 2 →
  Real.cos (α - β) = Real.sqrt 2 / 2 ∧ Real.cos β = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3707_370751


namespace NUMINAMATH_CALUDE_acyclic_orientations_not_div_three_l3707_370712

/-- A bipartite graph representing airline connections between Russian and Ukrainian cities -/
structure AirlineGraph where
  vertices : Type
  edges : Set (vertices × vertices)
  is_bipartite : ∃ (A B : Set vertices), A ∪ B = univ ∧ A ∩ B = ∅ ∧
    ∀ e ∈ edges, (e.1 ∈ A ∧ e.2 ∈ B) ∨ (e.1 ∈ B ∧ e.2 ∈ A)

/-- The number of acyclic orientations of a graph -/
def num_acyclic_orientations (G : AirlineGraph) : ℕ :=
  sorry

/-- Theorem: The number of acyclic orientations of the airline graph is not divisible by 3 -/
theorem acyclic_orientations_not_div_three (G : AirlineGraph) :
  ¬(3 ∣ num_acyclic_orientations G) :=
sorry

end NUMINAMATH_CALUDE_acyclic_orientations_not_div_three_l3707_370712


namespace NUMINAMATH_CALUDE_cube_volume_l3707_370728

theorem cube_volume (a : ℤ) : 
  (∃ (x y : ℤ), x = a + 2 ∧ y = a - 2 ∧ 
    x * a * y = a^3 - 16 ∧
    2 * (x + a) = 2 * (a + a) + 4) →
  a^3 = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_l3707_370728


namespace NUMINAMATH_CALUDE_parallel_vector_implies_zero_y_coordinate_l3707_370723

/-- Given vectors a and b in R², if b - a is parallel to a, then the y-coordinate of b is 0 -/
theorem parallel_vector_implies_zero_y_coordinate (m n : ℝ) :
  let a : Fin 2 → ℝ := ![1, 0]
  let b : Fin 2 → ℝ := ![m, n]
  (∃ (k : ℝ), (b - a) = k • a) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_implies_zero_y_coordinate_l3707_370723


namespace NUMINAMATH_CALUDE_negation_equivalence_l3707_370789

theorem negation_equivalence (x : ℝ) :
  ¬(x = 0 ∨ x = 1 → x^2 - x = 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3707_370789


namespace NUMINAMATH_CALUDE_triple_equation_solution_l3707_370722

theorem triple_equation_solution :
  ∀ (a b c : ℝ), 
    ((2*a+1)^2 - 4*b = 5 ∧ 
     (2*b+1)^2 - 4*c = 5 ∧ 
     (2*c+1)^2 - 4*a = 5) ↔ 
    ((a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = -1 ∧ b = -1 ∧ c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_triple_equation_solution_l3707_370722


namespace NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l3707_370764

theorem geometric_mean_of_sqrt2_plus_minus_one :
  let a := Real.sqrt 2 + 1
  let b := Real.sqrt 2 - 1
  ∃ x : ℝ, x^2 = a * b ∧ (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l3707_370764


namespace NUMINAMATH_CALUDE_function_value_at_negative_m_l3707_370774

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the theorem
theorem function_value_at_negative_m (a b m : ℝ) :
  f a b m = 6 → f a b (-m) = -4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_m_l3707_370774


namespace NUMINAMATH_CALUDE_alexa_reading_progress_l3707_370703

def pages_left_to_read (total_pages first_day_pages second_day_pages : ℕ) : ℕ :=
  total_pages - (first_day_pages + second_day_pages)

theorem alexa_reading_progress :
  pages_left_to_read 95 18 58 = 19 := by
  sorry

end NUMINAMATH_CALUDE_alexa_reading_progress_l3707_370703


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3707_370769

theorem simplify_sqrt_expression :
  (Real.sqrt 600 / Real.sqrt 75) - (Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3707_370769


namespace NUMINAMATH_CALUDE_three_unit_fractions_sum_to_one_l3707_370799

theorem three_unit_fractions_sum_to_one :
  ∀ a b c : ℕ+,
    a ≠ b → b ≠ c → a ≠ c →
    (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ = 1 →
    ({a, b, c} : Set ℕ+) = {2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_three_unit_fractions_sum_to_one_l3707_370799


namespace NUMINAMATH_CALUDE_ellipse_and_line_equations_l3707_370725

noncomputable section

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def c : ℝ := Real.sqrt 3

-- Define the perimeter of triangle MF₁F₂
def triangle_perimeter : ℝ := 4 + 2 * Real.sqrt 3

-- Define point P
def P : ℝ × ℝ := (0, 2)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem ellipse_and_line_equations :
  ∃ (a b : ℝ),
    -- Conditions
    ellipse_C a b (-c) 0 ∧
    (∀ x y, ellipse_C a b x y → 
      ∃ (m : ℝ × ℝ), Real.sqrt ((x - (-c))^2 + y^2) + Real.sqrt ((x - c)^2 + y^2) = triangle_perimeter) ∧
    -- Conclusions
    (a = 2 ∧ b = 1) ∧
    (∃ (k : ℝ), k = 2 ∨ k = -2) ∧
    (∀ k, k = 2 ∨ k = -2 →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        ellipse_C 2 1 x₁ y₁ ∧
        ellipse_C 2 1 x₂ y₂ ∧
        y₁ = k * x₁ - 2 ∧
        y₂ = k * x₂ - 2 ∧
        perpendicular x₁ y₁ x₂ y₂) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equations_l3707_370725


namespace NUMINAMATH_CALUDE_units_digit_of_k97_l3707_370731

-- Define the modified Lucas sequence
def modifiedLucas : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem units_digit_of_k97 : unitsDigit (modifiedLucas 97) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k97_l3707_370731


namespace NUMINAMATH_CALUDE_triangle_interior_angle_mean_l3707_370737

theorem triangle_interior_angle_mean :
  let num_angles : ℕ := 3
  let angle_sum : ℝ := 180
  (angle_sum / num_angles : ℝ) = 60 := by sorry

end NUMINAMATH_CALUDE_triangle_interior_angle_mean_l3707_370737


namespace NUMINAMATH_CALUDE_oil_transfer_height_l3707_370715

/-- Given a cone with base radius 9 cm and height 27 cm, when its volume is transferred to a cylinder with base radius 18 cm, the height of the liquid in the cylinder is 2.25 cm. -/
theorem oil_transfer_height :
  let cone_radius : ℝ := 9
  let cone_height : ℝ := 27
  let cylinder_radius : ℝ := 18
  let cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height
  let cylinder_height : ℝ := cone_volume / (Real.pi * cylinder_radius^2)
  cylinder_height = 2.25
  := by sorry

end NUMINAMATH_CALUDE_oil_transfer_height_l3707_370715


namespace NUMINAMATH_CALUDE_third_month_sale_l3707_370756

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sixth_month_sale : ℕ := 4791
def first_month_sale : ℕ := 6635
def second_month_sale : ℕ := 6927
def fourth_month_sale : ℕ := 7230
def fifth_month_sale : ℕ := 6562

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := first_month_sale + second_month_sale + fourth_month_sale + fifth_month_sale + sixth_month_sale
  total_sales - known_sales = 14085 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l3707_370756


namespace NUMINAMATH_CALUDE_cole_total_students_l3707_370705

/-- The number of students in Ms. Cole's math classes -/
structure ColeMathClasses where
  sixth_level : ℕ
  fourth_level : ℕ
  seventh_level : ℕ

/-- The conditions for Ms. Cole's math classes -/
def cole_math_class_conditions (c : ColeMathClasses) : Prop :=
  c.sixth_level = 40 ∧
  c.fourth_level = 4 * c.sixth_level ∧
  c.seventh_level = 2 * c.fourth_level

/-- The theorem stating the total number of students Ms. Cole teaches -/
theorem cole_total_students (c : ColeMathClasses) 
  (h : cole_math_class_conditions c) : 
  c.sixth_level + c.fourth_level + c.seventh_level = 520 := by
  sorry


end NUMINAMATH_CALUDE_cole_total_students_l3707_370705


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3707_370786

theorem complex_magnitude_fourth_power : 
  Complex.abs ((4 + 2 * Real.sqrt 2 * Complex.I) ^ 4) = 576 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3707_370786


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l3707_370749

/-- Given a cloth sale scenario, prove the cost price per meter. -/
theorem cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 78)
  (h2 : total_selling_price = 6788)
  (h3 : profit_per_meter = 29) :
  (total_selling_price - profit_per_meter * total_length) / total_length = 58 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l3707_370749


namespace NUMINAMATH_CALUDE_even_sum_of_even_sum_of_squares_l3707_370721

theorem even_sum_of_even_sum_of_squares (n m : ℤ) (h : Even (n^2 + m^2)) : Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_of_even_sum_of_squares_l3707_370721


namespace NUMINAMATH_CALUDE_cube_greater_than_l3707_370752

theorem cube_greater_than (x y : ℝ) (h : x > y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_greater_than_l3707_370752


namespace NUMINAMATH_CALUDE_difference_between_half_and_sixth_l3707_370718

theorem difference_between_half_and_sixth (x y : ℚ) : x = 1/2 → y = 1/6 → x - y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_half_and_sixth_l3707_370718


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3707_370738

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 - Complex.I → z = -1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3707_370738


namespace NUMINAMATH_CALUDE_class_average_score_l3707_370704

theorem class_average_score (scores : List ℝ) (avg_others : ℝ) : 
  scores.length = 4 →
  scores = [90, 85, 88, 80] →
  avg_others = 82 →
  let total_students : ℕ := 30
  let sum_scores : ℝ := scores.sum + (total_students - 4 : ℕ) * avg_others
  sum_scores / total_students = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l3707_370704


namespace NUMINAMATH_CALUDE_wendy_facial_products_l3707_370736

/-- The number of minutes Wendy waits between applying each facial product -/
def wait_time : ℕ := 5

/-- The number of minutes Wendy spends on make-up -/
def makeup_time : ℕ := 30

/-- The total number of minutes for Wendy's full face routine -/
def total_time : ℕ := 55

/-- The number of facial products Wendy uses -/
def num_products : ℕ := 6

theorem wendy_facial_products :
  wait_time * (num_products - 1) + makeup_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_wendy_facial_products_l3707_370736


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3707_370701

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = 2x + 3 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote: y = -2x - 1 -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through this point -/
  point : ℝ × ℝ
  /-- The first asymptote has the form y = 2x + 3 -/
  h₁ : ∀ x, asymptote1 x = 2 * x + 3
  /-- The second asymptote has the form y = -2x - 1 -/
  h₂ : ∀ x, asymptote2 x = -2 * x - 1
  /-- The point (4, 5) lies on the hyperbola -/
  h₃ : point = (4, 5)

/-- The distance between the foci of the hyperbola is 6√2 -/
theorem hyperbola_foci_distance (h : Hyperbola) : 
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3707_370701


namespace NUMINAMATH_CALUDE_equal_exchange_ways_l3707_370748

/-- Represents the number of ways to exchange money -/
def exchange_ways (n a b : ℕ) (use_blue : Bool) : ℕ :=
  sorry

/-- The main theorem stating that the number of ways to exchange is equal for both scenarios -/
theorem equal_exchange_ways (n a b : ℕ) :
  exchange_ways n a b true = exchange_ways n a b false :=
sorry

end NUMINAMATH_CALUDE_equal_exchange_ways_l3707_370748


namespace NUMINAMATH_CALUDE_m_greater_than_one_l3707_370733

theorem m_greater_than_one (m : ℝ) : (∀ x : ℝ, |x| ≤ 1 → x < m) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_one_l3707_370733


namespace NUMINAMATH_CALUDE_range_of_h_sign_and_inequality_smallest_upper_bound_l3707_370765

-- Define sets A and M_n
def A : Set (ℝ → ℝ) := {f | ∃ k, ∀ x > 0, f x < k}
def M (n : ℕ) : Set (ℝ → ℝ) := {f | ∀ x y, 0 < x ∧ x < y → (f x / x^n) < (f y / y^n)}

-- Statement 1
theorem range_of_h (h : ℝ) :
  (fun x => x^3 + h) ∈ M 1 ↔ h ≤ 0 :=
sorry

-- Statement 2
theorem sign_and_inequality (f : ℝ → ℝ) (a b d : ℝ) 
  (hf : f ∈ M 1) (hab : 0 < a ∧ a < b) (hd : f a = d ∧ f b = d) :
  d < 0 ∧ f (a + b) > 2 * d :=
sorry

-- Statement 3
theorem smallest_upper_bound (m : ℝ) :
  (∀ f ∈ A ∩ M 2, ∀ x > 0, f x < m) ↔ m ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_h_sign_and_inequality_smallest_upper_bound_l3707_370765


namespace NUMINAMATH_CALUDE_rotation_implies_equilateral_l3707_370743

-- Define the triangle
variable (A₁ A₂ A₃ : ℝ × ℝ)

-- Define the rotation function
def rotate (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the sequence of rotations
def rotate_sequence (n : ℕ) (P₀ : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define equilateral triangle
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  sorry

theorem rotation_implies_equilateral 
  (P₀ : ℝ × ℝ) 
  (h : rotate_sequence 1986 P₀ = P₀) : 
  is_equilateral A₁ A₂ A₃ :=
sorry

end NUMINAMATH_CALUDE_rotation_implies_equilateral_l3707_370743


namespace NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l3707_370763

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if two planes are parallel -/
def planesParallel (plane1 plane2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ plane1.a = k * plane2.a ∧ plane1.b = k * plane2.b ∧ plane1.c = k * plane2.c

theorem plane_equation_satisfies_conditions : 
  let plane := Plane.mk 24 20 (-16) (-20)
  let point1 := Point3D.mk 2 (-1) 3
  let point2 := Point3D.mk 1 3 (-4)
  let parallelPlane := Plane.mk 3 (-4) 1 (-5)
  pointOnPlane plane point1 ∧ 
  pointOnPlane plane point2 ∧ 
  planesParallel plane parallelPlane :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l3707_370763


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l3707_370741

theorem sum_of_cubes_divisibility (n : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3 * n * (n^2 + 2) = 3 * k₁ ∧ 3 * n * (n^2 + 2) = 9 * k₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l3707_370741


namespace NUMINAMATH_CALUDE_no_y_intercepts_l3707_370796

theorem no_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - 5 * y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l3707_370796


namespace NUMINAMATH_CALUDE_problem_1_l3707_370798

theorem problem_1 : (1) - 2 + 8 - (-30) = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3707_370798


namespace NUMINAMATH_CALUDE_ten_team_league_max_points_l3707_370742

/-- Represents a football league with n teams -/
structure FootballLeague where
  n : ℕ
  points_per_win : ℕ
  points_per_draw : ℕ
  points_per_loss : ℕ

/-- The maximum possible points for each team in the league -/
def max_points_per_team (league : FootballLeague) : ℕ :=
  sorry

/-- Theorem stating that in a 10-team league with 3 points for a win, 
    1 for a draw, and 0 for a loss, the maximum points per team is 13 -/
theorem ten_team_league_max_points :
  let league := FootballLeague.mk 10 3 1 0
  max_points_per_team league = 13 :=
sorry

end NUMINAMATH_CALUDE_ten_team_league_max_points_l3707_370742


namespace NUMINAMATH_CALUDE_fish_ratio_l3707_370766

/-- The number of fish in Billy's aquarium -/
def billy_fish : ℕ := 10

/-- The number of fish in Tony's aquarium -/
def tony_fish : ℕ := billy_fish * 3

/-- The number of fish in Sarah's aquarium -/
def sarah_fish : ℕ := tony_fish + 5

/-- The number of fish in Bobby's aquarium -/
def bobby_fish : ℕ := sarah_fish * 2

/-- The total number of fish in all aquariums -/
def total_fish : ℕ := 145

theorem fish_ratio : 
  billy_fish = 10 ∧ 
  tony_fish = billy_fish * 3 ∧ 
  sarah_fish = tony_fish + 5 ∧ 
  bobby_fish = sarah_fish * 2 ∧ 
  bobby_fish + sarah_fish + tony_fish + billy_fish = total_fish → 
  tony_fish / billy_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_fish_ratio_l3707_370766


namespace NUMINAMATH_CALUDE_doctor_lawyer_engineer_ratio_l3707_370760

-- Define the number of doctors, lawyers, and engineers
variable (d l e : ℕ)

-- Define the average ages
def avg_all : ℚ := 45
def avg_doctors : ℕ := 40
def avg_lawyers : ℕ := 55
def avg_engineers : ℕ := 35

-- State the theorem
theorem doctor_lawyer_engineer_ratio :
  (avg_all : ℚ) * (d + l + e : ℚ) = avg_doctors * d + avg_lawyers * l + avg_engineers * e →
  l = d + 2 * e :=
by sorry

end NUMINAMATH_CALUDE_doctor_lawyer_engineer_ratio_l3707_370760


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3707_370732

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + 3*m = 0

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  base_positive : base > 0
  side_positive : side > 0
  triangle_inequality : 2 * side > base

-- Theorem statement
theorem isosceles_triangle_perimeter : ∃ (m : ℝ) (t : IsoscelesTriangle),
  equation m 2 ∧ 
  (equation m t.base ∨ equation m t.side) ∧
  (t.base = 2 ∨ t.side = 2) ∧
  t.base + 2 * t.side = 14 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3707_370732


namespace NUMINAMATH_CALUDE_second_sample_is_26_l3707_370711

/-- Systematic sampling function -/
def systematicSample (totalSchools : ℕ) (sampleSize : ℕ) (startPoint : ℕ) : ℕ → ℕ :=
  fun i => (startPoint + (i - 1) * (totalSchools / sampleSize) - 1) % totalSchools + 1

/-- Theorem: In the given systematic sampling scenario, the second selected school number is 26 -/
theorem second_sample_is_26 :
  let totalSchools := 400
  let sampleSize := 20
  let startPoint := 6
  let secondSampleIndex := 2
  systematicSample totalSchools sampleSize startPoint secondSampleIndex = 26 := by
  sorry

end NUMINAMATH_CALUDE_second_sample_is_26_l3707_370711


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3707_370750

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3707_370750


namespace NUMINAMATH_CALUDE_range_of_expression_l3707_370739

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- State the theorem
theorem range_of_expression (t : AcuteTriangle) 
  (h : t.b * Real.cos t.A - t.a * Real.cos t.B = t.a) :
  2 < Real.sqrt 3 * Real.sin t.B + 2 * Real.sin t.A ^ 2 ∧ 
  Real.sqrt 3 * Real.sin t.B + 2 * Real.sin t.A ^ 2 < Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3707_370739


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3707_370744

theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a + b = 105 →
  b = a + 40 →
  max a (max b c) = 75 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3707_370744


namespace NUMINAMATH_CALUDE_unit_conversions_l3707_370735

-- Define conversion factors
def meters_to_decimeters : ℝ → ℝ := (· * 10)
def minutes_to_seconds : ℝ → ℝ := (· * 60)

-- Theorem to prove the conversions
theorem unit_conversions :
  (meters_to_decimeters 2 = 20) ∧
  (minutes_to_seconds 2 = 120) ∧
  (minutes_to_seconds (600 / 60) = 10) := by
  sorry

end NUMINAMATH_CALUDE_unit_conversions_l3707_370735


namespace NUMINAMATH_CALUDE_barycentric_geometry_l3707_370758

/-- Barycentric coordinates in a triangle --/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Definition of a line in barycentric coordinates --/
def is_line (f : BarycentricCoord → ℝ) : Prop :=
  ∃ u v w : ℝ, ∀ p : BarycentricCoord, f p = u * p.α + v * p.β + w * p.γ

/-- Definition of a circle in barycentric coordinates --/
def is_circle (f : BarycentricCoord → ℝ) : Prop :=
  ∃ a b c u v w : ℝ, ∀ p : BarycentricCoord,
    f p = -a^2 * p.β * p.γ - b^2 * p.γ * p.α - c^2 * p.α * p.β + 
          (u * p.α + v * p.β + w * p.γ) * (p.α + p.β + p.γ)

theorem barycentric_geometry :
  ∀ A : BarycentricCoord,
  (∃ f : BarycentricCoord → ℝ, is_line f ∧ ∀ p : BarycentricCoord, f p = p.β * w - p.γ * v) ∧
  (∃ g : BarycentricCoord → ℝ, is_line g) ∧
  (∃ h : BarycentricCoord → ℝ, is_circle h) :=
by sorry

end NUMINAMATH_CALUDE_barycentric_geometry_l3707_370758


namespace NUMINAMATH_CALUDE_min_difference_in_sample_l3707_370762

theorem min_difference_in_sample (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  c = 12 →
  (a + b + c + d + e) / 5 = 10 →
  e - a ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_in_sample_l3707_370762


namespace NUMINAMATH_CALUDE_box_interior_area_l3707_370746

/-- Calculates the surface area of the interior of a box formed from a rectangular sheet of cardboard
    with square corners cut out and edges folded upwards. -/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  (sheet_length - 2 * corner_size) * (sheet_width - 2 * corner_size)

/-- Theorem stating that the interior surface area of the box formed from a 35x50 sheet
    with 7-unit corners cut out is 756 square units. -/
theorem box_interior_area :
  interior_surface_area 35 50 7 = 756 := by
  sorry

end NUMINAMATH_CALUDE_box_interior_area_l3707_370746


namespace NUMINAMATH_CALUDE_geordie_commute_cost_l3707_370785

/-- Represents the cost calculation for Geordie's weekly commute -/
def weekly_commute_cost (car_toll : ℚ) (motorcycle_toll : ℚ) (mpg : ℚ) (distance : ℚ) (gas_price : ℚ) (car_trips : ℕ) (motorcycle_trips : ℕ) : ℚ :=
  let total_toll := car_toll * car_trips + motorcycle_toll * motorcycle_trips
  let total_miles := (distance * 2) * (car_trips + motorcycle_trips)
  let total_gas_cost := (total_miles / mpg) * gas_price
  total_toll + total_gas_cost

/-- Theorem stating that Geordie's weekly commute cost is $66.50 -/
theorem geordie_commute_cost :
  weekly_commute_cost 12.5 7 35 14 3.75 3 2 = 66.5 := by
  sorry

end NUMINAMATH_CALUDE_geordie_commute_cost_l3707_370785


namespace NUMINAMATH_CALUDE_a_correct_S_correct_l3707_370792

/-- The number of different selection methods for two non-empty subsets A and B of {1,2,3,...,n}
    where the smallest number in B is greater than the largest number in A. -/
def a (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 0
  else if n = 2 then 1
  else n * 2^(n-1) - 2^n + 1

/-- The sum of the first n terms of the sequence a_n. -/
def S (n : ℕ) : ℕ := (n - 3) * 2^n + n + 3

theorem a_correct (n : ℕ) : a n = n * 2^(n-1) - 2^n + 1 := by sorry

theorem S_correct (n : ℕ) : S n = (n - 3) * 2^n + n + 3 := by sorry

end NUMINAMATH_CALUDE_a_correct_S_correct_l3707_370792


namespace NUMINAMATH_CALUDE_inequality_property_l3707_370781

theorem inequality_property (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3707_370781


namespace NUMINAMATH_CALUDE_celine_erasers_l3707_370716

/-- Proves that Celine collected 10 erasers given the conditions of the problem -/
theorem celine_erasers (gabriel : ℕ) (celine : ℕ) (julian : ℕ) : 
  celine = 2 * gabriel → 
  julian = 2 * celine → 
  gabriel + celine + julian = 35 → 
  celine = 10 := by
sorry

end NUMINAMATH_CALUDE_celine_erasers_l3707_370716


namespace NUMINAMATH_CALUDE_gcd_binomial_coefficients_l3707_370797

theorem gcd_binomial_coefficients (n : ℕ+) :
  (∃ (p : ℕ) (k : ℕ+), Nat.Prime p ∧ n = p^(k:ℕ)) ↔
  (∃ m : ℕ, m > 1 ∧ (∀ i : Fin (n-1), m ∣ Nat.choose n i.val.succ)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_coefficients_l3707_370797


namespace NUMINAMATH_CALUDE_third_angle_is_70_l3707_370793

-- Define a triangle type
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the sum of angles in a triangle
def sum_of_angles (t : Triangle) : Real :=
  t.angle1 + t.angle2 + t.angle3

-- Theorem statement
theorem third_angle_is_70 (t : Triangle) 
  (h1 : t.angle1 = 50)
  (h2 : t.angle2 = 60)
  (h3 : sum_of_angles t = 180) : 
  t.angle3 = 70 := by
sorry


end NUMINAMATH_CALUDE_third_angle_is_70_l3707_370793


namespace NUMINAMATH_CALUDE_zoes_overall_accuracy_l3707_370753

/-- Represents the problem of calculating Zoe's overall accuracy rate -/
theorem zoes_overall_accuracy 
  (x : ℝ) -- Total number of problems
  (h_positive : x > 0) -- Ensure x is positive
  (h_chloe_indep : ℝ) -- Chloe's independent accuracy rate
  (h_chloe_indep_val : h_chloe_indep = 0.8) -- Chloe's independent accuracy is 80%
  (h_overall : ℝ) -- Overall accuracy rate for all problems
  (h_overall_val : h_overall = 0.88) -- Overall accuracy is 88%
  (h_zoe_indep : ℝ) -- Zoe's independent accuracy rate
  (h_zoe_indep_val : h_zoe_indep = 0.9) -- Zoe's independent accuracy is 90%
  : ∃ (y : ℝ), -- y represents the accuracy rate of problems solved together
    (0.5 * x * h_chloe_indep + 0.5 * x * y) / x = h_overall ∧ 
    (0.5 * x * h_zoe_indep + 0.5 * x * y) / x = 0.93 := by
  sorry

end NUMINAMATH_CALUDE_zoes_overall_accuracy_l3707_370753


namespace NUMINAMATH_CALUDE_price_reduction_equation_l3707_370767

/-- Represents the price reduction scenario for a certain type of chip -/
theorem price_reduction_equation (initial_price : ℝ) (final_price : ℝ) (x : ℝ) :
  initial_price = 400 →
  final_price = 144 →
  0 < x →
  x < 1 →
  initial_price * (1 - x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l3707_370767


namespace NUMINAMATH_CALUDE_absolute_value_and_exponent_simplification_l3707_370713

theorem absolute_value_and_exponent_simplification :
  |(-3 : ℝ)| + (3 - Real.sqrt 3) ^ (0 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponent_simplification_l3707_370713


namespace NUMINAMATH_CALUDE_matt_profit_l3707_370768

/-- Represents a baseball card collection --/
structure CardCollection where
  count : ℕ
  value : ℕ

/-- Calculates the total value of a card collection --/
def totalValue (c : CardCollection) : ℕ := c.count * c.value

/-- Represents a trade transaction --/
structure Trade where
  givenCards : List CardCollection
  receivedCards : List CardCollection

/-- Calculates the profit from a trade --/
def tradeProfitᵢ (t : Trade) : ℤ :=
  (t.receivedCards.map totalValue).sum - (t.givenCards.map totalValue).sum

/-- The initial card collection --/
def initialCollection : CardCollection := ⟨8, 6⟩

/-- The four trades Matt made --/
def trades : List Trade := [
  ⟨[⟨2, 6⟩], [⟨3, 2⟩, ⟨1, 9⟩]⟩,
  ⟨[⟨1, 2⟩, ⟨1, 6⟩], [⟨2, 5⟩, ⟨1, 8⟩]⟩,
  ⟨[⟨1, 5⟩, ⟨1, 9⟩], [⟨3, 3⟩, ⟨1, 10⟩, ⟨1, 1⟩]⟩,
  ⟨[⟨2, 3⟩, ⟨1, 8⟩], [⟨2, 7⟩, ⟨1, 4⟩]⟩
]

/-- Calculates the total profit from all trades --/
def totalProfit : ℤ := (trades.map tradeProfitᵢ).sum

theorem matt_profit : totalProfit = 23 := by
  sorry

end NUMINAMATH_CALUDE_matt_profit_l3707_370768


namespace NUMINAMATH_CALUDE_coating_time_for_given_problem_l3707_370771

/-- Represents the properties of the sphere coating problem -/
structure SphereCoating where
  copper_sphere_diameter : ℝ
  silver_layer_thickness : ℝ
  hydrogen_production : ℝ
  hydrogen_silver_ratio : ℝ
  silver_density : ℝ

/-- Calculates the time required for coating the sphere -/
noncomputable def coating_time (sc : SphereCoating) : ℝ :=
  sorry

/-- Theorem stating the coating time for the given problem -/
theorem coating_time_for_given_problem :
  let sc : SphereCoating := {
    copper_sphere_diameter := 3,
    silver_layer_thickness := 0.05,
    hydrogen_production := 11.11,
    hydrogen_silver_ratio := 1 / 108,
    silver_density := 10.5
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |coating_time sc - 987| < ε :=
sorry

end NUMINAMATH_CALUDE_coating_time_for_given_problem_l3707_370771


namespace NUMINAMATH_CALUDE_max_abs_z_plus_4_l3707_370755

theorem max_abs_z_plus_4 (z : ℂ) (h : Complex.abs (z + 3 * Complex.I) = 5) :
  ∃ (max_val : ℝ), max_val = 10 ∧ ∀ (w : ℂ), Complex.abs (w + 3 * Complex.I) = 5 → Complex.abs (w + 4) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_plus_4_l3707_370755


namespace NUMINAMATH_CALUDE_valentines_day_cards_l3707_370745

theorem valentines_day_cards (total_students : ℕ) (card_cost : ℚ) (total_money : ℚ) 
  (spend_percentage : ℚ) (h1 : total_students = 30) (h2 : card_cost = 2) 
  (h3 : total_money = 40) (h4 : spend_percentage = 0.9) : 
  (((total_money * spend_percentage) / card_cost) / total_students) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_cards_l3707_370745


namespace NUMINAMATH_CALUDE_apollonius_circle_tangency_locus_l3707_370729

/-- Apollonius circle associated with segment AB -/
structure ApolloniusCircle (A B : ℝ × ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  divides_ratio : ℝ → ℝ → Prop

/-- Point of tangency from A to the Apollonius circle -/
def tangency_point (A B : ℝ × ℝ) (circle : ApolloniusCircle A B) : ℝ × ℝ := sorry

/-- Line perpendicular to AB at point B -/
def perpendicular_line_at_B (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem apollonius_circle_tangency_locus 
  (A B : ℝ × ℝ) 
  (p : ℝ) 
  (h_p : p > 1) 
  (circle : ApolloniusCircle A B) 
  (h_circle : circle.divides_ratio p 1) :
  tangency_point A B circle ∈ perpendicular_line_at_B A B :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_tangency_locus_l3707_370729


namespace NUMINAMATH_CALUDE_exhibits_permutation_l3707_370787

theorem exhibits_permutation : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_exhibits_permutation_l3707_370787


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3707_370700

theorem solve_linear_equation (m : ℝ) (x : ℝ) : 
  (m * x + 1 = 2) → (x = -1) → (m = -1) := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3707_370700


namespace NUMINAMATH_CALUDE_power_negative_cube_squared_l3707_370790

theorem power_negative_cube_squared (a : ℝ) (n : ℤ) : (-a^(3*n))^2 = a^(6*n) := by
  sorry

end NUMINAMATH_CALUDE_power_negative_cube_squared_l3707_370790


namespace NUMINAMATH_CALUDE_sequence_properties_and_sum_l3707_370709

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem sequence_properties_and_sum (a b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 1 = 2 →
  (∃ r : ℝ, r = 2 ∧ ∀ n : ℕ, b (n + 1) = r * b n) →
  a 2 + b 3 = 7 →
  a 4 + b 5 = 21 →
  (∀ n : ℕ, c n = a n / b n) →
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, b n = 2^(n - 1)) ∧
  (∀ n : ℕ, S n = 6 - (n + 3) / 2^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_and_sum_l3707_370709


namespace NUMINAMATH_CALUDE_library_problem_l3707_370717

/-- Calculates the number of students helped on the first day given the total books,
    books per student, and students helped on subsequent days. -/
def students_helped_first_day (total_books : ℕ) (books_per_student : ℕ) 
    (students_day2 : ℕ) (students_day3 : ℕ) (students_day4 : ℕ) : ℕ :=
  (total_books - (students_day2 + students_day3 + students_day4) * books_per_student) / books_per_student

/-- Theorem stating that given the conditions in the problem, 
    the number of students helped on the first day is 4. -/
theorem library_problem : 
  students_helped_first_day 120 5 5 6 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_library_problem_l3707_370717


namespace NUMINAMATH_CALUDE_distribution_theorem_l3707_370761

/-- The number of ways to distribute 5 students into 3 groups (A, B, C),
    where group A has at least 2 students and groups B and C each have at least 1 student. -/
def distribution_schemes : ℕ := 80

/-- The total number of students -/
def total_students : ℕ := 5

/-- The number of groups -/
def num_groups : ℕ := 3

/-- The minimum number of students in group A -/
def min_group_a : ℕ := 2

/-- The minimum number of students in groups B and C -/
def min_group_bc : ℕ := 1

theorem distribution_theorem :
  (∀ (scheme : Fin total_students → Fin num_groups),
    (∃ (a b c : Finset (Fin total_students)),
      a.card ≥ min_group_a ∧
      b.card ≥ min_group_bc ∧
      c.card ≥ min_group_bc ∧
      a ∪ b ∪ c = Finset.univ ∧
      a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅)) →
  (Fintype.card {scheme : Fin total_students → Fin num_groups |
    ∃ (a b c : Finset (Fin total_students)),
      a.card ≥ min_group_a ∧
      b.card ≥ min_group_bc ∧
      c.card ≥ min_group_bc ∧
      a ∪ b ∪ c = Finset.univ ∧
      a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅}) = distribution_schemes :=
by sorry

end NUMINAMATH_CALUDE_distribution_theorem_l3707_370761


namespace NUMINAMATH_CALUDE_problem_solving_probability_l3707_370794

theorem problem_solving_probability 
  (kyle_prob : ℚ) 
  (david_prob : ℚ) 
  (catherine_prob : ℚ) 
  (h1 : kyle_prob = 1/3) 
  (h2 : david_prob = 2/7) 
  (h3 : catherine_prob = 5/9) : 
  kyle_prob * catherine_prob * (1 - david_prob) = 25/189 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l3707_370794


namespace NUMINAMATH_CALUDE_passing_percentage_problem_l3707_370770

/-- The passing percentage problem -/
theorem passing_percentage_problem (mike_score : ℕ) (shortfall : ℕ) (max_marks : ℕ) 
  (h1 : mike_score = 212)
  (h2 : shortfall = 25)
  (h3 : max_marks = 790) :
  let passing_marks : ℕ := mike_score + shortfall
  let passing_percentage : ℚ := (passing_marks : ℚ) / max_marks * 100
  ∃ ε > 0, abs (passing_percentage - 30) < ε := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_problem_l3707_370770


namespace NUMINAMATH_CALUDE_function_inequality_l3707_370740

theorem function_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : Real.exp (a + 1) = a + 4) (h2 : Real.log (b + 3) = b) :
  let f := fun x => Real.exp x + (a - b) * x
  f (2/3) < f 0 ∧ f 0 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3707_370740


namespace NUMINAMATH_CALUDE_sequence_sum_l3707_370779

def is_six_digit_number (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def sequence_property (x : ℕ → ℕ) : Prop :=
  is_six_digit_number (x 1) ∧
  ∀ n : ℕ, n ≥ 1 → Nat.Prime (x (n + 1)) ∧ (x (n + 1) ∣ x n + 1)

theorem sequence_sum (x : ℕ → ℕ) (h : sequence_property x) : x 19 + x 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3707_370779


namespace NUMINAMATH_CALUDE_football_season_games_l3707_370726

/-- Calculates the total number of football games in a season -/
def total_games (months : ℕ) (games_per_month : ℕ) : ℕ :=
  months * games_per_month

theorem football_season_games :
  let season_length : ℕ := 17
  let games_per_month : ℕ := 19
  total_games season_length games_per_month = 323 := by
sorry

end NUMINAMATH_CALUDE_football_season_games_l3707_370726


namespace NUMINAMATH_CALUDE_siblings_combined_age_l3707_370775

/-- The combined age of five siblings -/
def combined_age (aaron_age henry_sister_age henry_age alice_age eric_age : ℕ) : ℕ :=
  aaron_age + henry_sister_age + henry_age + alice_age + eric_age

theorem siblings_combined_age :
  ∀ (aaron_age henry_sister_age henry_age alice_age eric_age : ℕ),
    aaron_age = 15 →
    henry_sister_age = 3 * aaron_age →
    henry_age = 4 * henry_sister_age →
    alice_age = aaron_age - 2 →
    eric_age = henry_sister_age + alice_age →
    combined_age aaron_age henry_sister_age henry_age alice_age eric_age = 311 :=
by
  sorry

end NUMINAMATH_CALUDE_siblings_combined_age_l3707_370775


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3707_370777

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (10, y), (-2, -1), (10, -1)]
  let length : ℝ := 10 - (-2)
  let height : ℝ := y - (-1)
  let area : ℝ := length * height
  area = 108 → y = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3707_370777


namespace NUMINAMATH_CALUDE_fuel_consumption_analysis_l3707_370782

/-- Represents the fuel consumption data for a sedan --/
structure FuelData where
  initial_fuel : ℝ
  distance : ℝ
  remaining_fuel : ℝ

/-- Theorem about fuel consumption of a sedan --/
theorem fuel_consumption_analysis 
  (data : List FuelData)
  (h1 : data.length ≥ 2)
  (h2 : data[0].distance = 0 ∧ data[0].remaining_fuel = 50)
  (h3 : data[1].distance = 100 ∧ data[1].remaining_fuel = 42)
  (h4 : ∀ d ∈ data, d.initial_fuel = 50)
  (h5 : ∀ d ∈ data, d.remaining_fuel = d.initial_fuel - 0.08 * d.distance) :
  (∀ d ∈ data, d.initial_fuel = 50) ∧ 
  (∀ d ∈ data, d.remaining_fuel = -0.08 * d.distance + 50) := by
  sorry


end NUMINAMATH_CALUDE_fuel_consumption_analysis_l3707_370782


namespace NUMINAMATH_CALUDE_eight_stairs_climb_ways_l3707_370707

-- Define the function for the number of ways to climb n stairs
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | m + 4 => climbStairs (m + 2) + climbStairs (m + 1)

-- Theorem stating that there are 4 ways to climb 8 stairs
theorem eight_stairs_climb_ways : climbStairs 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eight_stairs_climb_ways_l3707_370707


namespace NUMINAMATH_CALUDE_sarah_new_shirts_l3707_370776

/-- Given that Sarah initially had 9 shirts and now has a total of 17 shirts,
    prove that she bought 8 new shirts. -/
theorem sarah_new_shirts (initial_shirts : ℕ) (total_shirts : ℕ) (new_shirts : ℕ) :
  initial_shirts = 9 →
  total_shirts = 17 →
  new_shirts = total_shirts - initial_shirts →
  new_shirts = 8 := by
  sorry

end NUMINAMATH_CALUDE_sarah_new_shirts_l3707_370776


namespace NUMINAMATH_CALUDE_loan_interest_rate_calculation_l3707_370720

/-- The interest rate for the second part of a loan, given specific conditions -/
theorem loan_interest_rate_calculation (total : ℝ) (second_part : ℝ) : 
  total = 2743 →
  second_part = 1688 →
  let first_part := total - second_part
  let interest_rate_first := 0.03
  let time_first := 8
  let time_second := 3
  let interest_first := first_part * interest_rate_first * time_first
  let interest_second := second_part * time_second
  ∃ (r : ℝ), interest_first = r * interest_second ∧ 
             r ≥ 0.0499 ∧ r ≤ 0.05 := by
  sorry

#check loan_interest_rate_calculation

end NUMINAMATH_CALUDE_loan_interest_rate_calculation_l3707_370720


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1135_l3707_370783

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (13 ^ i)) 0

/-- The value of digit C in base 13 -/
def C : Nat := 12

/-- The theorem to prove -/
theorem sum_of_bases_equals_1135 :
  base9ToBase10 [1, 6, 3] + base13ToBase10 [5, C, 4] = 1135 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1135_l3707_370783


namespace NUMINAMATH_CALUDE_fraction_order_l3707_370759

theorem fraction_order : 
  let f1 := 21 / 16
  let f2 := 25 / 19
  let f3 := 23 / 17
  let f4 := 27 / 20
  f1 < f2 ∧ f2 < f4 ∧ f4 < f3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l3707_370759


namespace NUMINAMATH_CALUDE_proportion_and_equation_imply_c_value_l3707_370706

theorem proportion_and_equation_imply_c_value 
  (a b c : ℝ) 
  (h1 : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 7*k) 
  (h2 : a - b + 3 = c - 2*b) : 
  c = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_and_equation_imply_c_value_l3707_370706


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3707_370710

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = (1/2 : ℂ) * (1 + Complex.I)) :
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3707_370710


namespace NUMINAMATH_CALUDE_ellipse_constant_slope_l3707_370754

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : 4/a^2 + 1/b^2 = 1

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2/E.a^2 + y^2/E.b^2 = 1

/-- The theorem statement -/
theorem ellipse_constant_slope (E : Ellipse) 
  (h_bisector : ∀ (P Q : PointOnEllipse E), 
    (∃ (k : ℝ), k * (P.x - 2) = P.y - 1 ∧ k * (Q.x - 2) = -(Q.y - 1))) :
  ∀ (P Q : PointOnEllipse E), (Q.y - P.y) / (Q.x - P.x) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_slope_l3707_370754


namespace NUMINAMATH_CALUDE_food_distribution_l3707_370780

/-- The number of days the food initially lasts -/
def initial_days : ℝ := 45

/-- The initial number of men in the camp -/
def initial_men : ℕ := 40

/-- The number of days the food lasts after additional men join -/
def final_days : ℝ := 32.73

/-- The number of additional men who joined the camp -/
def additional_men : ℕ := 15

theorem food_distribution (total_food : ℝ) :
  total_food = initial_men * initial_days ∧
  total_food = (initial_men + additional_men) * final_days :=
sorry

#check food_distribution

end NUMINAMATH_CALUDE_food_distribution_l3707_370780
