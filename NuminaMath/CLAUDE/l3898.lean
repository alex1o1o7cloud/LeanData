import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3898_389861

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1,
    given that one of its asymptotes passes through the point (2, √21) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0) :
  (∃ (x y : ℝ), x = 2 ∧ y = Real.sqrt 21 ∧ y = (b / a) * x) →
  Real.sqrt (1 + (b / a)^2) = 5/2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3898_389861


namespace NUMINAMATH_CALUDE_cone_rotation_ratio_l3898_389895

theorem cone_rotation_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  2 * π * Real.sqrt (r^2 + h^2) = 20 * π * r →
  h / r = Real.sqrt 399 := by
sorry

end NUMINAMATH_CALUDE_cone_rotation_ratio_l3898_389895


namespace NUMINAMATH_CALUDE_total_boys_in_three_sections_l3898_389803

theorem total_boys_in_three_sections (section1_total : ℕ) (section2_total : ℕ) (section3_total : ℕ)
  (section1_girls_ratio : ℚ) (section2_boys_ratio : ℚ) (section3_boys_ratio : ℚ) :
  section1_total = 160 →
  section2_total = 200 →
  section3_total = 240 →
  section1_girls_ratio = 1/4 →
  section2_boys_ratio = 3/5 →
  section3_boys_ratio = 7/12 →
  (section1_total - section1_total * section1_girls_ratio) +
  (section2_total * section2_boys_ratio) +
  (section3_total * section3_boys_ratio) = 380 := by
sorry

end NUMINAMATH_CALUDE_total_boys_in_three_sections_l3898_389803


namespace NUMINAMATH_CALUDE_line_segment_has_measurable_length_l3898_389880

-- Define the characteristics of geometric objects
structure GeometricObject where
  has_endpoints : Bool
  is_infinite : Bool

-- Define specific geometric objects
def line : GeometricObject :=
  { has_endpoints := false, is_infinite := true }

def ray : GeometricObject :=
  { has_endpoints := true, is_infinite := true }

def line_segment : GeometricObject :=
  { has_endpoints := true, is_infinite := false }

-- Define a property for having measurable length
def has_measurable_length (obj : GeometricObject) : Prop :=
  obj.has_endpoints ∧ ¬obj.is_infinite

-- Theorem statement
theorem line_segment_has_measurable_length :
  has_measurable_length line_segment ∧
  ¬has_measurable_length line ∧
  ¬has_measurable_length ray :=
sorry

end NUMINAMATH_CALUDE_line_segment_has_measurable_length_l3898_389880


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3898_389805

theorem complex_equation_solution (z : ℂ) :
  z / (1 - 2 * Complex.I) = Complex.I → z = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3898_389805


namespace NUMINAMATH_CALUDE_sin_2x_equiv_cos_2x_shifted_l3898_389844

theorem sin_2x_equiv_cos_2x_shifted (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_equiv_cos_2x_shifted_l3898_389844


namespace NUMINAMATH_CALUDE_f_of_4_equals_82_l3898_389882

-- Define a monotonic function f
def monotonic_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f y ≤ f x

-- State the theorem
theorem f_of_4_equals_82
  (f : ℝ → ℝ)
  (h_monotonic : monotonic_function f)
  (h_condition : ∀ x : ℝ, f (f x - 3^x) = 4) :
  f 4 = 82 := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_equals_82_l3898_389882


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l3898_389899

/-- Given a total amount of clothes and a number of washing machines, 
    calculate the amount of clothes one washing machine can wash per day. -/
def clothes_per_machine (total_clothes : ℕ) (num_machines : ℕ) : ℕ :=
  total_clothes / num_machines

/-- Theorem stating that for 200 pounds of clothes and 8 machines, 
    each machine can wash 25 pounds per day. -/
theorem washing_machine_capacity : clothes_per_machine 200 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l3898_389899


namespace NUMINAMATH_CALUDE_custom_cartesian_product_of_A_and_B_l3898_389858

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2*x - x^2)}
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define the custom cartesian product
def customCartesianProduct (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- Theorem statement
theorem custom_cartesian_product_of_A_and_B :
  customCartesianProduct A B = Set.Icc 0 1 ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_custom_cartesian_product_of_A_and_B_l3898_389858


namespace NUMINAMATH_CALUDE_number_equation_solution_l3898_389800

theorem number_equation_solution : 
  ∃ n : ℚ, (3/4 : ℚ) * n - (8/5 : ℚ) * n + 63 = 12 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3898_389800


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3898_389808

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, 1 + 1996 * m + 1998 * n = m * n ↔
    (m = 1999 ∧ n = 1997^2 + 1996) ∨
    (m = 3995 ∧ n = 3993) ∨
    (m = 1997^2 + 1998 ∧ n = 1997) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3898_389808


namespace NUMINAMATH_CALUDE_geometric_sequence_consecutive_terms_l3898_389864

theorem geometric_sequence_consecutive_terms (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = x * r ∧ (3*x + 3) = (2*x + 2) * r) → 
  x = 1 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_consecutive_terms_l3898_389864


namespace NUMINAMATH_CALUDE_number_divided_by_004_l3898_389862

theorem number_divided_by_004 :
  ∃ x : ℝ, x / 0.04 = 500.90000000000003 ∧ x = 20.036 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_004_l3898_389862


namespace NUMINAMATH_CALUDE_point_division_theorem_l3898_389870

/-- Given points A and B, if there exists a point C on the line y=x that divides AB in the ratio 2:1, then the y-coordinate of B is 4. -/
theorem point_division_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, a)
  ∃ (C : ℝ × ℝ), 
    (C.1 = C.2) ∧  -- C is on the line y = x
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B) ∧  -- C is on line segment AB
    (C - A = 2 • (B - C))  -- AC = 2CB
    → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_division_theorem_l3898_389870


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3898_389821

/-- The common ratio of the geometric series 2/3 + 4/9 + 8/27 + ... is 2/3 -/
theorem geometric_series_common_ratio : 
  let a : ℕ → ℚ := fun n => (2 / 3) * (2 / 3)^n
  ∀ n : ℕ, a (n + 1) / a n = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3898_389821


namespace NUMINAMATH_CALUDE_hero_qin_equivalence_l3898_389867

theorem hero_qin_equivalence (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c)) =
  Real.sqrt ((1 / 4) * (a^2 * b^2 - ((a^2 + b^2 + c^2) / 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_hero_qin_equivalence_l3898_389867


namespace NUMINAMATH_CALUDE_ab_divisible_by_twelve_l3898_389863

/-- A primitive Pythagorean triple -/
structure PrimitivePythagoreanTriple :=
  (a b c : ℕ)
  (primitive : Nat.gcd a (Nat.gcd b c) = 1)
  (pythagorean : a^2 + b^2 = c^2)

/-- Theorem: For any primitive Pythagorean triple (a, b, c), ab is divisible by 12 -/
theorem ab_divisible_by_twelve (t : PrimitivePythagoreanTriple) : 
  12 ∣ (t.a * t.b) := by
  sorry

end NUMINAMATH_CALUDE_ab_divisible_by_twelve_l3898_389863


namespace NUMINAMATH_CALUDE_ellipse_equation_l3898_389893

theorem ellipse_equation (x y : ℝ) :
  let a : ℝ := 4
  let b : ℝ := Real.sqrt 7
  let ε : ℝ := 0.75
  let passes_through : Prop := (-3)^2 / a^2 + 1.75^2 / b^2 = 1
  let eccentricity : Prop := ε = Real.sqrt (a^2 - b^2) / a
  passes_through ∧ eccentricity →
  x^2 / 16 + y^2 / 7 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3898_389893


namespace NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_y_greater_than_one_ninth_l3898_389804

theorem sqrt_less_than_3y_iff_y_greater_than_one_ninth (y : ℝ) (h : y > 0) :
  Real.sqrt y < 3 * y ↔ y > 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_y_greater_than_one_ninth_l3898_389804


namespace NUMINAMATH_CALUDE_max_plus_shapes_in_square_l3898_389872

theorem max_plus_shapes_in_square (side_length : ℕ) (l_shape_area : ℕ) (plus_shape_area : ℕ) 
  (h_side : side_length = 7)
  (h_l : l_shape_area = 3)
  (h_plus : plus_shape_area = 5) :
  ∃ (num_l num_plus : ℕ),
    num_l * l_shape_area + num_plus * plus_shape_area = side_length ^ 2 ∧
    num_l ≥ 4 ∧
    ∀ (other_num_l other_num_plus : ℕ),
      other_num_l * l_shape_area + other_num_plus * plus_shape_area = side_length ^ 2 →
      other_num_l ≥ 4 →
      other_num_plus ≤ num_plus :=
by sorry

end NUMINAMATH_CALUDE_max_plus_shapes_in_square_l3898_389872


namespace NUMINAMATH_CALUDE_toy_position_l3898_389879

theorem toy_position (total_toys : ℕ) (position_from_right : ℕ) (position_from_left : ℕ) :
  total_toys = 19 →
  position_from_right = 8 →
  position_from_left = total_toys - (position_from_right - 1) →
  position_from_left = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_position_l3898_389879


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3898_389853

def shirt_price : ℝ := 50
def pants_price : ℝ := 40
def shoes_price : ℝ := 60
def shirt_discount : ℝ := 0.2
def shoes_discount : ℝ := 0.5
def sales_tax : ℝ := 0.08

def total_cost : ℝ :=
  let shirt_cost := 6 * shirt_price * (1 - shirt_discount)
  let pants_cost := 2 * pants_price
  let shoes_cost := 2 * shoes_price + shoes_price * (1 - shoes_discount)
  let subtotal := shirt_cost + pants_cost + shoes_cost
  subtotal * (1 + sales_tax)

theorem total_cost_calculation :
  total_cost = 507.60 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3898_389853


namespace NUMINAMATH_CALUDE_gcd_of_repeated_digit_ints_l3898_389837

/-- Represents a four-digit positive integer -/
def FourDigitInt := {n : ℕ // 1000 ≤ n ∧ n < 10000}

/-- Constructs an eight-digit integer by repeating a four-digit integer -/
def repeatFourDigits (n : FourDigitInt) : ℕ :=
  10000 * n.val + n.val

/-- The set of all eight-digit integers formed by repeating a four-digit integer -/
def RepeatedDigitInts : Set ℕ :=
  {m | ∃ n : FourDigitInt, m = repeatFourDigits n}

/-- Theorem stating that 10001 is the greatest common divisor of all eight-digit integers
    formed by repeating a four-digit integer -/
theorem gcd_of_repeated_digit_ints :
  ∃ d : ℕ, d > 0 ∧ (∀ m ∈ RepeatedDigitInts, d ∣ m) ∧
  (∀ d' : ℕ, d' > 0 → (∀ m ∈ RepeatedDigitInts, d' ∣ m) → d' ≤ d) ∧
  d = 10001 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_digit_ints_l3898_389837


namespace NUMINAMATH_CALUDE_large_planks_count_l3898_389868

theorem large_planks_count (nails_per_plank : ℕ) (additional_nails : ℕ) (total_nails : ℕ) :
  nails_per_plank = 17 →
  additional_nails = 8 →
  total_nails = 229 →
  ∃ (x : ℕ), x * nails_per_plank + additional_nails = total_nails ∧ x = 13 :=
by sorry

end NUMINAMATH_CALUDE_large_planks_count_l3898_389868


namespace NUMINAMATH_CALUDE_parallelogram_area_l3898_389814

def v1 : Fin 3 → ℝ := ![4, -1, 3]
def v2 : Fin 3 → ℝ := ![-2, 5, -1]

theorem parallelogram_area : 
  Real.sqrt ((v1 1 * v2 2 - v1 2 * v2 1)^2 + 
             (v1 2 * v2 0 - v1 0 * v2 2)^2 + 
             (v1 0 * v2 1 - v1 1 * v2 0)^2) = Real.sqrt 684 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3898_389814


namespace NUMINAMATH_CALUDE_normal_force_wooden_blocks_l3898_389832

/-- The normal force from a system of wooden blocks to a table -/
theorem normal_force_wooden_blocks
  (M m : ℝ)  -- Masses of the larger block and smaller cubes
  (α β : ℝ)  -- Angles of the sides of the larger block
  (hM : M > 0)  -- Mass of larger block is positive
  (hm : m > 0)  -- Mass of smaller cubes is positive
  (hα : 0 < α ∧ α < π/2)  -- α is between 0 and π/2
  (hβ : 0 < β ∧ β < π/2)  -- β is between 0 and π/2
  (g : ℝ)  -- Gravitational acceleration
  (hg : g > 0)  -- Gravitational acceleration is positive
  : ℝ :=
  M * g + m * g * (Real.cos α ^ 2 + Real.cos β ^ 2)

#check normal_force_wooden_blocks

end NUMINAMATH_CALUDE_normal_force_wooden_blocks_l3898_389832


namespace NUMINAMATH_CALUDE_complex_equation_difference_l3898_389859

theorem complex_equation_difference (a b : ℝ) :
  (a : ℂ) + b * Complex.I = (1 + 2 * Complex.I) * (3 - Complex.I) + (1 + Complex.I) / (1 - Complex.I) →
  a - b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_difference_l3898_389859


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3898_389833

/-- Geometric sequence with a_3 = 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 3 = 1

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h : geometric_sequence a) 
  (h_prod : a 6 * a 8 = 64) : a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3898_389833


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3898_389856

/-- Represents a right triangle with angles 30°, 60°, and 90° -/
structure Triangle30_60_90 where
  shortSide : ℝ
  longSide : ℝ
  hypotenuse : ℝ
  angle30 : Real
  angle60 : Real
  angle90 : Real

/-- Represents a circle tangent to coordinate axes and triangle hypotenuse -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a 30-60-90 triangle with shortest side 2, 
    the radius of a circle tangent to coordinate axes and hypotenuse is 1 + 2√3 -/
theorem tangent_circle_radius 
  (t : Triangle30_60_90) 
  (c : TangentCircle) 
  (h1 : t.shortSide = 2) 
  (h2 : c.center.1 > 0 ∧ c.center.2 > 0) 
  (h3 : c.radius = c.center.1 ∧ c.radius = c.center.2) 
  (h4 : ∃ (x y : ℝ), x^2 + y^2 = t.hypotenuse^2 ∧ 
                     (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :
  c.radius = 1 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3898_389856


namespace NUMINAMATH_CALUDE_inequality_solution_l3898_389836

theorem inequality_solution :
  ∀ x : ℕ+, (2 * x + 9 ≥ 3 * (x + 2)) ↔ (x = 1 ∨ x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3898_389836


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3898_389894

theorem linear_equation_solution (a : ℝ) : 
  (a * 1 + (-2) = 3) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3898_389894


namespace NUMINAMATH_CALUDE_no_divisibility_by_1955_l3898_389886

theorem no_divisibility_by_1955 : ∀ n : ℤ, ¬(1955 ∣ (n^2 + n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_divisibility_by_1955_l3898_389886


namespace NUMINAMATH_CALUDE_ribbon_length_reduction_l3898_389875

theorem ribbon_length_reduction (original_length new_length : ℝ) : 
  (11 : ℝ) / 7 = original_length / new_length →
  new_length = 35 →
  original_length = 55 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_length_reduction_l3898_389875


namespace NUMINAMATH_CALUDE_other_car_speed_l3898_389816

/-- Given two cars traveling in the same direction, prove that if:
    1. The red car travels at a constant speed of 30 mph
    2. The red car is initially 20 miles ahead of the other car
    3. The other car overtakes the red car in 1 hour
    Then the speed of the other car is 50 mph -/
theorem other_car_speed (red_speed : ℝ) (initial_distance : ℝ) (overtake_time : ℝ) :
  red_speed = 30 →
  initial_distance = 20 →
  overtake_time = 1 →
  (red_speed * overtake_time + initial_distance) / overtake_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_other_car_speed_l3898_389816


namespace NUMINAMATH_CALUDE_common_chord_length_l3898_389869

theorem common_chord_length (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0) ∧
  (∀ x y : ℝ, x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0 → y = 1/a) →
  a = 1 :=
sorry


end NUMINAMATH_CALUDE_common_chord_length_l3898_389869


namespace NUMINAMATH_CALUDE_sunflower_count_l3898_389889

theorem sunflower_count (total_flowers : ℕ) (other_flowers : ℕ) 
  (h1 : total_flowers = 160) 
  (h2 : other_flowers = 40) : 
  total_flowers - other_flowers = 120 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_count_l3898_389889


namespace NUMINAMATH_CALUDE_cook_selection_theorem_l3898_389820

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 cooks from a group of 10 people,
    where one specific person must always be selected. -/
def cookSelectionWays : ℕ := choose 9 1

theorem cook_selection_theorem :
  cookSelectionWays = 9 := by sorry

end NUMINAMATH_CALUDE_cook_selection_theorem_l3898_389820


namespace NUMINAMATH_CALUDE_drawings_on_last_page_is_sixty_l3898_389881

/-- Represents the problem of rearranging drawings in notebooks --/
structure NotebookProblem where
  initial_notebooks : ℕ
  pages_per_notebook : ℕ
  initial_drawings_per_page : ℕ
  new_drawings_per_page : ℕ
  filled_notebooks : ℕ
  filled_pages_in_last_notebook : ℕ

/-- Calculate the number of drawings on the last page of the partially filled notebook --/
def drawings_on_last_page (p : NotebookProblem) : ℕ :=
  let total_drawings := p.initial_notebooks * p.pages_per_notebook * p.initial_drawings_per_page
  let filled_pages := p.filled_notebooks * p.pages_per_notebook + p.filled_pages_in_last_notebook
  let drawings_on_filled_pages := filled_pages * p.new_drawings_per_page
  total_drawings - drawings_on_filled_pages

/-- The main theorem stating that for the given problem, there are 60 drawings on the last page --/
theorem drawings_on_last_page_is_sixty :
  let p : NotebookProblem := {
    initial_notebooks := 5,
    pages_per_notebook := 60,
    initial_drawings_per_page := 8,
    new_drawings_per_page := 12,
    filled_notebooks := 3,
    filled_pages_in_last_notebook := 45
  }
  drawings_on_last_page p = 60 := by
  sorry


end NUMINAMATH_CALUDE_drawings_on_last_page_is_sixty_l3898_389881


namespace NUMINAMATH_CALUDE_initial_shells_count_l3898_389851

/-- The number of shells Ed found at the beach -/
def ed_shells : ℕ := 13

/-- The number of shells Jacob found at the beach -/
def jacob_shells : ℕ := ed_shells + 2

/-- The total number of shells after collecting -/
def total_shells : ℕ := 30

/-- The initial number of shells in the collection -/
def initial_shells : ℕ := total_shells - (ed_shells + jacob_shells)

theorem initial_shells_count : initial_shells = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_shells_count_l3898_389851


namespace NUMINAMATH_CALUDE_pi_estimation_l3898_389855

theorem pi_estimation (total_points : ℕ) (obtuse_points : ℕ) : 
  total_points = 120 → obtuse_points = 34 → 
  (obtuse_points : ℝ) / (total_points : ℝ) = π / 4 - 1 / 2 → 
  π = 47 / 15 := by
sorry

end NUMINAMATH_CALUDE_pi_estimation_l3898_389855


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3898_389888

theorem greatest_x_quadratic_inequality :
  ∀ x : ℝ, -x^2 + 11*x - 28 ≥ 0 → x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3898_389888


namespace NUMINAMATH_CALUDE_triangle_problem_l3898_389854

theorem triangle_problem (A B C : ℝ) (m n : ℝ × ℝ) (AC : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  m = (Real.cos (A + π / 3), Real.sin (A + π / 3)) →
  n = (Real.cos B, Real.sin B) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  Real.cos B = 3 / 5 →
  AC = 8 →
  A - B = π / 6 ∧ Real.sqrt ((4 * Real.sqrt 3 + 3) ^ 2) = 4 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3898_389854


namespace NUMINAMATH_CALUDE_probability_five_successes_in_seven_trials_l3898_389898

/-- The probability of getting exactly 5 successes in 7 trials with a success probability of 3/4 -/
theorem probability_five_successes_in_seven_trials :
  let n : ℕ := 7  -- number of trials
  let k : ℕ := 5  -- number of successes
  let p : ℚ := 3/4  -- probability of success on each trial
  Nat.choose n k * p^k * (1 - p)^(n - k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_successes_in_seven_trials_l3898_389898


namespace NUMINAMATH_CALUDE_cosine_sum_identity_l3898_389897

theorem cosine_sum_identity : 
  Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + 
  Real.sin (80 * π / 180) * Real.sin (20 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_identity_l3898_389897


namespace NUMINAMATH_CALUDE_rogers_coins_l3898_389884

theorem rogers_coins (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 3)
  (h2 : dime_piles = 3)
  (h3 : coins_per_pile = 7) :
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 42 :=
by sorry

end NUMINAMATH_CALUDE_rogers_coins_l3898_389884


namespace NUMINAMATH_CALUDE_product_of_terms_l3898_389819

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1) ^ 2 - 8 * (a 1) + 1 = 0 →
  (a 13) ^ 2 - 8 * (a 13) + 1 = 0 →
  a 5 * a 7 * a 9 = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_of_terms_l3898_389819


namespace NUMINAMATH_CALUDE_average_weight_increase_l3898_389840

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 98.6 →
  (new_weight - old_weight) / initial_count = 4.2 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3898_389840


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_60_l3898_389865

theorem product_of_five_consecutive_integers_divisible_by_60 (n : ℤ) : 
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_60_l3898_389865


namespace NUMINAMATH_CALUDE_two_painter_time_l3898_389812

/-- The time taken for two painters to complete a wall together, given their individual rates -/
theorem two_painter_time (harish_rate ganpat_rate : ℝ) (harish_time ganpat_time : ℝ) :
  harish_rate = 1 / harish_time →
  ganpat_rate = 1 / ganpat_time →
  harish_time = 3 →
  ganpat_time = 6 →
  1 / (harish_rate + ganpat_rate) = 2 := by
  sorry

#check two_painter_time

end NUMINAMATH_CALUDE_two_painter_time_l3898_389812


namespace NUMINAMATH_CALUDE_floor_product_eq_42_l3898_389824

theorem floor_product_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 :=
sorry

end NUMINAMATH_CALUDE_floor_product_eq_42_l3898_389824


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l3898_389876

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_overs : ℕ
  first_run_rate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_overs
  let runs_in_first_overs := game.first_run_rate * game.first_overs
  let remaining_runs := game.target - runs_in_first_overs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_overs = 10)
  (h3 : game.first_run_rate = 3.6)
  (h4 : game.target = 282) :
  required_run_rate game = 6.15 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l3898_389876


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_l3898_389823

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] where
  F₁ : α
  F₂ : α

/-- A point P in the plane satisfying the locus condition -/
structure LocusPoint (α : Type*) [NormedAddCommGroup α] (FP : FixedPoints α) where
  P : α
  k : ℝ
  h_positive : k > 0
  h_less : k < ‖FP.F₁ - FP.F₂‖
  h_condition : ‖P - FP.F₁‖ - ‖P - FP.F₂‖ = k

/-- Definition of a hyperbola -/
def IsHyperbola (α : Type*) [NormedAddCommGroup α] (S : Set α) (FP : FixedPoints α) :=
  ∃ k : ℝ, k > 0 ∧ k < ‖FP.F₁ - FP.F₂‖ ∧
    S = {P | ‖P - FP.F₁‖ - ‖P - FP.F₂‖ = k ∨ ‖P - FP.F₂‖ - ‖P - FP.F₁‖ = k}

/-- The main theorem: The locus of points satisfying the given condition forms a hyperbola -/
theorem locus_is_hyperbola {α : Type*} [NormedAddCommGroup α] (FP : FixedPoints α) :
  IsHyperbola α {P | ∃ LP : LocusPoint α FP, LP.P = P} FP :=
sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_l3898_389823


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_range_of_a_l3898_389818

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorems to be proved
theorem union_A_B : A ∪ B = {x | 3 ≤ x ∧ x ≤ 10} := by sorry

theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x ≤ 10} := by sorry

theorem range_of_a (a : ℝ) (h : (A ∩ C a).Nonempty) : a > 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_range_of_a_l3898_389818


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3898_389839

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3898_389839


namespace NUMINAMATH_CALUDE_odd_function_property_l3898_389825

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Main theorem -/
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : IsOdd f) 
  (h_even : IsEven (fun x ↦ f (x + 2))) 
  (h_f_neg_one : f (-1) = -1) : 
  f 2017 + f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3898_389825


namespace NUMINAMATH_CALUDE_dice_probability_l3898_389848

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def dice_outcome := ℕ × ℕ

def favorable_outcome (outcome : dice_outcome) : Prop :=
  is_prime outcome.1 ∧ is_perfect_square outcome.2

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 6

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l3898_389848


namespace NUMINAMATH_CALUDE_heptagon_side_sum_l3898_389852

/-- Represents a polygon with 7 vertices --/
structure Heptagon :=
  (A B C D E F G : ℝ × ℝ)

/-- Calculates the area of a polygon --/
def area (p : Heptagon) : ℝ := sorry

/-- Calculates the distance between two points --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem heptagon_side_sum (p : Heptagon) :
  area p = 120 ∧
  distance p.A p.B = 10 ∧
  distance p.B p.C = 15 ∧
  distance p.G p.A = 7 →
  distance p.D p.E + distance p.E p.F = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_side_sum_l3898_389852


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l3898_389883

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 24) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l3898_389883


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l3898_389892

theorem largest_multiple_of_11_under_100 : ∃ n : ℕ, n * 11 = 99 ∧ n * 11 < 100 ∧ ∀ m : ℕ, m * 11 < 100 → m * 11 ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l3898_389892


namespace NUMINAMATH_CALUDE_max_sum_of_factors_of_24_l3898_389810

theorem max_sum_of_factors_of_24 : 
  ∀ (a b : ℕ), a * b = 24 → a + b ≤ 25 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_of_24_l3898_389810


namespace NUMINAMATH_CALUDE_total_commission_proof_l3898_389845

def commission_rate : ℝ := 0.02

def house_prices : List ℝ := [157000, 499000, 125000]

theorem total_commission_proof :
  (house_prices.map (· * commission_rate)).sum = 15620 := by
  sorry

end NUMINAMATH_CALUDE_total_commission_proof_l3898_389845


namespace NUMINAMATH_CALUDE_compare_expressions_min_value_expression_l3898_389829

variable (m n : ℝ)

/-- Part 1: Compare m² + n and mn + m when m > n > 1 -/
theorem compare_expressions (hm : m > 0) (hn : n > 0) (hmn : m > n) (hn1 : n > 1) :
  m^2 + n > m*n + m := by sorry

/-- Part 2: Find the minimum value of 2/m + 1/n when m + 2n = 1 -/
theorem min_value_expression (hm : m > 0) (hn : n > 0) (hmn : m > n) (hsum : m + 2*n = 1) :
  ∃ (min_val : ℝ), min_val = 8 ∧ ∀ x, x = 2/m + 1/n → x ≥ min_val := by sorry

end NUMINAMATH_CALUDE_compare_expressions_min_value_expression_l3898_389829


namespace NUMINAMATH_CALUDE_small_circle_radius_l3898_389801

/-- Given a large circle with radius 10 meters containing three smaller circles
    that touch each other and are aligned horizontally across its center,
    prove that the radius of each smaller circle is 10/3 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : R = 10 →
  3 * (2 * r) = 2 * R →
  r = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3898_389801


namespace NUMINAMATH_CALUDE_wheel_probability_l3898_389873

theorem wheel_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 2/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/10 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l3898_389873


namespace NUMINAMATH_CALUDE_mikes_total_spending_l3898_389847

/-- Represents Mike's shopping expenses -/
structure ShoppingExpenses where
  food : ℝ
  wallet : ℝ
  shirt : ℝ

/-- Calculates the total spending given Mike's shopping expenses -/
def totalSpending (expenses : ShoppingExpenses) : ℝ :=
  expenses.food + expenses.wallet + expenses.shirt

/-- Theorem stating Mike's total spending given the problem conditions -/
theorem mikes_total_spending :
  ∀ (expenses : ShoppingExpenses),
    expenses.food = 30 →
    expenses.wallet = expenses.food + 60 →
    expenses.shirt = expenses.wallet / 3 →
    totalSpending expenses = 150 := by
  sorry


end NUMINAMATH_CALUDE_mikes_total_spending_l3898_389847


namespace NUMINAMATH_CALUDE_problem_statement_l3898_389891

theorem problem_statement (a b k : ℕ+) (h : (a.val^2 - 1 - b.val^2) / (a.val * b.val - 1) = k.val) : k = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3898_389891


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l3898_389874

theorem triangle_with_angle_ratio_1_2_3_is_right_angled (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l3898_389874


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l3898_389834

theorem no_real_sqrt_negative_quadratic :
  ∀ x : ℝ, ¬ ∃ y : ℝ, y ^ 2 = -(x ^ 2 + 2 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l3898_389834


namespace NUMINAMATH_CALUDE_common_ratio_is_two_l3898_389843

/-- An increasing geometric sequence with specific conditions -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  is_increasing : q > 1
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q
  a2_eq_2 : a 2 = 2
  a4_minus_a3_eq_4 : a 4 - a 3 = 4

/-- The common ratio of the increasing geometric sequence is 2 -/
theorem common_ratio_is_two (seq : IncreasingGeometricSequence) : seq.q = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l3898_389843


namespace NUMINAMATH_CALUDE_bamboo_problem_l3898_389822

theorem bamboo_problem (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 + a 3 + a 4 = 3 →   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →         -- sum of last 3 terms
  a 5 = 67 / 66 := by
sorry

end NUMINAMATH_CALUDE_bamboo_problem_l3898_389822


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3898_389831

/-- Represents a school with a given number of students -/
structure School where
  students : ℕ

/-- Calculates the number of students to be sampled from a school in a stratified sample -/
def stratifiedSampleSize (school : School) (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (school.students * sampleSize) / totalStudents

theorem stratified_sample_theorem (schoolA schoolB schoolC : School) 
    (h1 : schoolA.students = 3600)
    (h2 : schoolB.students = 5400)
    (h3 : schoolC.students = 1800)
    (totalSampleSize : ℕ)
    (h4 : totalSampleSize = 90) :
  let totalStudents := schoolA.students + schoolB.students + schoolC.students
  (stratifiedSampleSize schoolA totalStudents totalSampleSize = 30) ∧
  (stratifiedSampleSize schoolB totalStudents totalSampleSize = 45) ∧
  (stratifiedSampleSize schoolC totalStudents totalSampleSize = 15) := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_theorem_l3898_389831


namespace NUMINAMATH_CALUDE_morning_afternoon_emails_l3898_389885

theorem morning_afternoon_emails (morning_emails afternoon_emails : ℕ) 
  (h1 : morning_emails = 5)
  (h2 : afternoon_emails = 8) :
  morning_emails + afternoon_emails = 13 := by
sorry

end NUMINAMATH_CALUDE_morning_afternoon_emails_l3898_389885


namespace NUMINAMATH_CALUDE_max_factors_power_function_l3898_389849

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- b^n where b and n are positive integers less than or equal to 10 -/
def power_function (b n : ℕ) : ℕ := 
  if b ≤ 10 ∧ n ≤ 10 ∧ b > 0 ∧ n > 0 then b^n else 0

theorem max_factors_power_function :
  ∃ b n : ℕ, b ≤ 10 ∧ n ≤ 10 ∧ b > 0 ∧ n > 0 ∧
    num_factors (power_function b n) = 31 ∧
    ∀ b' n' : ℕ, b' ≤ 10 → n' ≤ 10 → b' > 0 → n' > 0 →
      num_factors (power_function b' n') ≤ 31 :=
sorry

end NUMINAMATH_CALUDE_max_factors_power_function_l3898_389849


namespace NUMINAMATH_CALUDE_unique_students_in_musical_groups_l3898_389890

/-- The number of unique students in four musical groups -/
theorem unique_students_in_musical_groups 
  (orchestra : Nat) (band : Nat) (choir : Nat) (jazz : Nat)
  (orchestra_band : Nat) (orchestra_choir : Nat) (band_choir : Nat)
  (band_jazz : Nat) (orchestra_jazz : Nat) (choir_jazz : Nat)
  (orchestra_band_choir : Nat) (all_four : Nat)
  (h1 : orchestra = 25)
  (h2 : band = 40)
  (h3 : choir = 30)
  (h4 : jazz = 15)
  (h5 : orchestra_band = 5)
  (h6 : orchestra_choir = 6)
  (h7 : band_choir = 4)
  (h8 : band_jazz = 3)
  (h9 : orchestra_jazz = 2)
  (h10 : choir_jazz = 4)
  (h11 : orchestra_band_choir = 3)
  (h12 : all_four = 1) :
  orchestra + band + choir + jazz
  - orchestra_band - orchestra_choir - band_choir
  - band_jazz - orchestra_jazz - choir_jazz
  + orchestra_band_choir + all_four = 90 :=
by sorry

end NUMINAMATH_CALUDE_unique_students_in_musical_groups_l3898_389890


namespace NUMINAMATH_CALUDE_factorial_product_not_perfect_power_l3898_389866

-- Define the factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the product of factorials from 1 to n
def factorial_product (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * factorial (i + 1)) 1

-- Define a function to check if a number is a perfect power greater than 1
def is_perfect_power (n : ℕ) : Prop :=
  ∃ (base exponent : ℕ), base > 1 ∧ exponent > 1 ∧ base ^ exponent = n

-- State the theorem
theorem factorial_product_not_perfect_power :
  ¬ (is_perfect_power (factorial_product 2022)) :=
sorry

end NUMINAMATH_CALUDE_factorial_product_not_perfect_power_l3898_389866


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3898_389857

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 19 * x + k = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 - 19 * y + k = 0 ∧ y = 16/3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3898_389857


namespace NUMINAMATH_CALUDE_ninety_six_configurations_l3898_389827

/-- Represents a configuration of numbers in the grid -/
def Configuration := Fin 6 → Fin 6

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 6) : Prop :=
  sorry

/-- Checks if a configuration is valid according to the rules -/
def valid_configuration (c : Configuration) : Prop :=
  ∀ p1 p2 : Fin 6, adjacent p1 p2 → abs (c p1 - c p2) ≠ 3

/-- The total number of valid configurations -/
def total_valid_configurations : ℕ :=
  sorry

/-- Main theorem: There are 96 valid configurations -/
theorem ninety_six_configurations : total_valid_configurations = 96 :=
  sorry

end NUMINAMATH_CALUDE_ninety_six_configurations_l3898_389827


namespace NUMINAMATH_CALUDE_consequences_of_only_some_A_are_B_l3898_389811

-- Define sets A and B
variable (A B : Set α)

-- Define the premise "Only some A are B"
def only_some_A_are_B : Prop := ∃ x ∈ A, x ∈ B ∧ ∃ y ∈ A, y ∉ B

-- Theorem stating the consequences
theorem consequences_of_only_some_A_are_B (h : only_some_A_are_B A B) :
  (¬ ∀ x ∈ A, x ∈ B) ∧
  (∃ x ∈ A, x ∉ B) ∧
  (∃ x ∈ B, x ∈ A) ∧
  (∃ x ∈ A, x ∈ B) ∧
  (∃ x ∈ A, x ∉ B) :=
by sorry

end NUMINAMATH_CALUDE_consequences_of_only_some_A_are_B_l3898_389811


namespace NUMINAMATH_CALUDE_log_problem_l3898_389850

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_problem (x : ℝ) (h : log 8 (3 * x) = 3) :
  log x 125 = 3 / (9 * log 5 2 - log 5 3) := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l3898_389850


namespace NUMINAMATH_CALUDE_solve_for_c_l3898_389860

theorem solve_for_c (m a b c : ℝ) (h : m = (c * b * a) / (a - c)) :
  c = (m * a) / (m + b * a) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l3898_389860


namespace NUMINAMATH_CALUDE_popcorn_per_serving_l3898_389830

/-- The number of pieces of popcorn Jared can eat -/
def jared_popcorn : ℕ := 90

/-- The number of pieces of popcorn each of Jared's friends can eat -/
def friend_popcorn : ℕ := 60

/-- The number of Jared's friends -/
def num_friends : ℕ := 3

/-- The number of servings Jared should order -/
def num_servings : ℕ := 9

/-- Theorem stating that the number of pieces of popcorn in a serving is 30 -/
theorem popcorn_per_serving : 
  (jared_popcorn + num_friends * friend_popcorn) / num_servings = 30 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_per_serving_l3898_389830


namespace NUMINAMATH_CALUDE_count_juggling_sequences_l3898_389838

/-- The number of juggling sequences of length n with exactly 1 ball -/
def jugglingSequences (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: The number of juggling sequences of length n with exactly 1 ball is 2^n - 1 -/
theorem count_juggling_sequences (n : ℕ) : 
  jugglingSequences n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_count_juggling_sequences_l3898_389838


namespace NUMINAMATH_CALUDE_max_volume_rectangular_prism_l3898_389828

/-- Represents a right prism with a rectangular base -/
structure RectangularPrism where
  a : ℝ  -- length of the base
  b : ℝ  -- width of the base
  h : ℝ  -- height of the prism

/-- The sum of areas of two lateral faces and the base face is 32 -/
def area_constraint (p : RectangularPrism) : Prop :=
  p.a * p.h + p.b * p.h + p.a * p.b = 32

/-- The volume of the prism -/
def volume (p : RectangularPrism) : ℝ :=
  p.a * p.b * p.h

/-- Theorem stating the maximum volume of the prism -/
theorem max_volume_rectangular_prism :
  ∀ p : RectangularPrism, area_constraint p →
  volume p ≤ (128 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_rectangular_prism_l3898_389828


namespace NUMINAMATH_CALUDE_sum_difference_is_50_l3898_389896

def sam_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_20 (x : ℕ) : ℕ :=
  20 * ((x + 10) / 20)

def alex_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_20 (List.range n))

theorem sum_difference_is_50 :
  sam_sum 100 - alex_sum 100 = 50 := by
  sorry

#eval sam_sum 100 - alex_sum 100

end NUMINAMATH_CALUDE_sum_difference_is_50_l3898_389896


namespace NUMINAMATH_CALUDE_prob_at_least_one_to_museum_l3898_389877

/-- The probability that at least one of two independent events occurs -/
def prob_at_least_one (p₁ p₂ : ℝ) : ℝ := 1 - (1 - p₁) * (1 - p₂)

/-- The probability that at least one of two people goes to the museum -/
theorem prob_at_least_one_to_museum (p_a p_b : ℝ) 
  (h_a : p_a = 0.8) 
  (h_b : p_b = 0.7) : 
  prob_at_least_one p_a p_b = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_to_museum_l3898_389877


namespace NUMINAMATH_CALUDE_projection_periodicity_l3898_389841

/-- Regular n-gon with vertices A₁, A₂, ..., Aₙ -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Point on a side of the n-gon -/
structure PointOnSide (n : ℕ) where
  ngon : RegularNGon n
  side : Fin n
  point : ℝ × ℝ

/-- Projection function that maps Mᵢ to Mᵢ₊₁ -/
def project (n : ℕ) (m : PointOnSide n) : PointOnSide n :=
  sorry

/-- The k-th projection of a point -/
def kthProjection (n k : ℕ) (m : PointOnSide n) : PointOnSide n :=
  sorry

theorem projection_periodicity (n : ℕ) (m : PointOnSide n) :
  (n = 4 → kthProjection n 13 m = m) ∧
  (n = 6 → kthProjection n 13 m = m) ∧
  (n = 10 → kthProjection n 11 m = m) :=
sorry

end NUMINAMATH_CALUDE_projection_periodicity_l3898_389841


namespace NUMINAMATH_CALUDE_fernanda_savings_after_payments_l3898_389806

/-- Calculates the total amount in Fernanda's savings account after receiving payments from debtors -/
theorem fernanda_savings_after_payments (aryan_debt kyro_debt : ℚ) 
  (h1 : aryan_debt = 1200)
  (h2 : aryan_debt = 2 * kyro_debt)
  (h3 : aryan_payment = 0.6 * aryan_debt)
  (h4 : kyro_payment = 0.8 * kyro_debt)
  (h5 : initial_savings = 300) :
  initial_savings + aryan_payment + kyro_payment = 1500 := by
  sorry

end NUMINAMATH_CALUDE_fernanda_savings_after_payments_l3898_389806


namespace NUMINAMATH_CALUDE_possible_distances_andrey_gleb_l3898_389817

/-- Represents the position of a house on a straight street. -/
structure HousePosition where
  position : ℝ

/-- Represents the configuration of houses on the street. -/
structure StreetConfiguration where
  andrey : HousePosition
  borya : HousePosition
  vova : HousePosition
  gleb : HousePosition

/-- The distance between two house positions. -/
def distance (a b : HousePosition) : ℝ :=
  |a.position - b.position|

/-- Theorem stating the possible distances between Andrey's and Gleb's houses. -/
theorem possible_distances_andrey_gleb (config : StreetConfiguration) :
  (distance config.andrey config.borya = 600) →
  (distance config.vova config.gleb = 600) →
  (distance config.andrey config.gleb = 3 * distance config.borya config.vova) →
  (distance config.andrey config.gleb = 900 ∨ distance config.andrey config.gleb = 1800) :=
by sorry

end NUMINAMATH_CALUDE_possible_distances_andrey_gleb_l3898_389817


namespace NUMINAMATH_CALUDE_baseball_league_games_played_l3898_389842

/-- Represents a baseball league with the given parameters -/
structure BaseballLeague where
  num_teams : ℕ
  games_per_week : ℕ
  season_length_months : ℕ
  
/-- Calculates the total number of games played in a season -/
def total_games_played (league : BaseballLeague) : ℕ :=
  (league.num_teams * league.games_per_week * league.season_length_months * 4) / 2

/-- Theorem stating the total number of games played in the given league configuration -/
theorem baseball_league_games_played :
  ∃ (league : BaseballLeague),
    league.num_teams = 10 ∧
    league.games_per_week = 5 ∧
    league.season_length_months = 6 ∧
    total_games_played league = 600 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_played_l3898_389842


namespace NUMINAMATH_CALUDE_units_digit_of_42_pow_5_plus_27_pow_5_l3898_389826

theorem units_digit_of_42_pow_5_plus_27_pow_5 : (42^5 + 27^5) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_42_pow_5_plus_27_pow_5_l3898_389826


namespace NUMINAMATH_CALUDE_mary_balloon_count_l3898_389835

/-- The number of black balloons Nancy has -/
def nancy_balloons : ℕ := 7

/-- The factor by which Mary's balloons exceed Nancy's -/
def mary_factor : ℕ := 4

/-- The number of black balloons Mary has -/
def mary_balloons : ℕ := nancy_balloons * mary_factor

theorem mary_balloon_count : mary_balloons = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_balloon_count_l3898_389835


namespace NUMINAMATH_CALUDE_proposition_p_and_its_negation_l3898_389846

theorem proposition_p_and_its_negation :
  (∃ x : ℝ, x^2 - x = 0) ∧
  (¬(∃ x : ℝ, x^2 - x = 0) ↔ (∀ x : ℝ, x^2 - x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_and_its_negation_l3898_389846


namespace NUMINAMATH_CALUDE_centroid_of_equal_areas_l3898_389878

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Check if a point is inside a triangle -/
def isInside (M : Point) (T : Triangle) : Prop :=
  sorry

/-- Calculate the area of a triangle -/
def triangleArea (A B C : Point) : ℝ :=
  sorry

/-- Check if three triangles have equal areas -/
def equalAreas (T1 T2 T3 : Triangle) : Prop :=
  triangleArea T1.A T1.B T1.C = triangleArea T2.A T2.B T2.C ∧
  triangleArea T2.A T2.B T2.C = triangleArea T3.A T3.B T3.C

/-- Check if a point is the centroid of a triangle -/
def isCentroid (M : Point) (T : Triangle) : Prop :=
  sorry

theorem centroid_of_equal_areas (ABC : Triangle) (M : Point) 
  (h1 : isInside M ABC)
  (h2 : equalAreas (Triangle.mk M ABC.A ABC.B) (Triangle.mk M ABC.A ABC.C) (Triangle.mk M ABC.B ABC.C)) :
  isCentroid M ABC :=
sorry

end NUMINAMATH_CALUDE_centroid_of_equal_areas_l3898_389878


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l3898_389887

theorem greatest_prime_factor_of_4_pow_17_minus_2_pow_29 : 
  ∃ (p : ℕ), p.Prime ∧ p = 31 ∧ 
  (∀ q : ℕ, q.Prime → q ∣ (4^17 - 2^29) → q ≤ p) :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l3898_389887


namespace NUMINAMATH_CALUDE_nth_root_two_inequality_l3898_389871

theorem nth_root_two_inequality (n : ℕ) (h : n ≥ 2) :
  (2 : ℝ) ^ (1 / n) - 1 ≤ Real.sqrt (2 / (n * (n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_two_inequality_l3898_389871


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l3898_389813

/-- Given that points (-4, y₁), (-1, y₂), and (5/3, y₃) lie on the graph of y = -x² - 4x + 5,
    prove that y₂ > y₁ > y₃ -/
theorem parabola_y_relationship (y₁ y₂ y₃ : ℝ) : 
  y₁ = -(-4)^2 - 4*(-4) + 5 →
  y₂ = -(-1)^2 - 4*(-1) + 5 →
  y₃ = -(5/3)^2 - 4*(5/3) + 5 →
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_parabola_y_relationship_l3898_389813


namespace NUMINAMATH_CALUDE_discounted_three_books_cost_l3898_389807

/-- The cost of two identical books without discount -/
def two_books_cost : ℝ := 36

/-- The discount rate applied to each book -/
def discount_rate : ℝ := 0.1

/-- The number of books to purchase after discount -/
def num_books_after_discount : ℕ := 3

/-- Theorem stating the total cost of three books after applying a 10% discount -/
theorem discounted_three_books_cost :
  let original_price := two_books_cost / 2
  let discounted_price := original_price * (1 - discount_rate)
  discounted_price * num_books_after_discount = 48.60 := by
  sorry

end NUMINAMATH_CALUDE_discounted_three_books_cost_l3898_389807


namespace NUMINAMATH_CALUDE_central_angle_for_given_arc_central_angle_proof_l3898_389815

/-- Given a circle with radius 100mm and an arc length of 300mm,
    the central angle corresponding to this arc is 3 radians. -/
theorem central_angle_for_given_arc : ℝ → ℝ → ℝ → Prop :=
  λ radius arc_length angle =>
    radius = 100 ∧ arc_length = 300 → angle = 3

/-- The theorem proof -/
theorem central_angle_proof :
  ∃ (angle : ℝ), central_angle_for_given_arc 100 300 angle :=
by
  sorry

end NUMINAMATH_CALUDE_central_angle_for_given_arc_central_angle_proof_l3898_389815


namespace NUMINAMATH_CALUDE_square_equals_self_only_zero_and_one_l3898_389809

theorem square_equals_self_only_zero_and_one :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_square_equals_self_only_zero_and_one_l3898_389809


namespace NUMINAMATH_CALUDE_badminton_equipment_purchase_l3898_389802

/-- Represents the cost of purchasing badminton equipment from Store A -/
def cost_store_a (x : ℝ) : ℝ := 1760 + 40 * x

/-- Represents the cost of purchasing badminton equipment from Store B -/
def cost_store_b (x : ℝ) : ℝ := 1920 + 32 * x

theorem badminton_equipment_purchase (x : ℝ) (h : x > 16) :
  (x > 20 → cost_store_b x < cost_store_a x) ∧
  (x < 20 → cost_store_a x < cost_store_b x) := by
  sorry

#check badminton_equipment_purchase

end NUMINAMATH_CALUDE_badminton_equipment_purchase_l3898_389802
