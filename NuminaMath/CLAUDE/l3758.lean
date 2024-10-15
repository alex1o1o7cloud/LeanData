import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l3758_375897

/-- The area of a rectangle with a hole -/
def rectangle_with_hole_area (x : ℝ) : ℝ :=
  (2*x + 8) * (x + 6) - (3*x - 4) * (x + 1)

/-- Theorem: The area of the rectangle with a hole is equal to -x^2 + 21x + 52 -/
theorem rectangle_with_hole_area_formula (x : ℝ) :
  rectangle_with_hole_area x = -x^2 + 21*x + 52 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l3758_375897


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l3758_375821

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 8 → 
  (10 * x + y) - (10 * y + x) = 72 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l3758_375821


namespace NUMINAMATH_CALUDE_tomato_seed_planting_l3758_375880

theorem tomato_seed_planting (mike_morning mike_afternoon ted_morning ted_afternoon total : ℕ) : 
  mike_morning = 50 →
  ted_morning = 2 * mike_morning →
  mike_afternoon = 60 →
  ted_afternoon < mike_afternoon →
  total = mike_morning + ted_morning + mike_afternoon + ted_afternoon →
  total = 250 →
  mike_afternoon - ted_afternoon = 20 := by
sorry

end NUMINAMATH_CALUDE_tomato_seed_planting_l3758_375880


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_56_minus_1_l3758_375822

theorem divisors_of_2_pow_56_minus_1 :
  ∃ (a b : ℕ), 95 < a ∧ a < 105 ∧ 95 < b ∧ b < 105 ∧
  a ≠ b ∧
  (2^56 - 1) % a = 0 ∧ (2^56 - 1) % b = 0 ∧
  (∀ c : ℕ, 95 < c ∧ c < 105 → (2^56 - 1) % c = 0 → c = a ∨ c = b) ∧
  a = 101 ∧ b = 127 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_56_minus_1_l3758_375822


namespace NUMINAMATH_CALUDE_inspection_ratio_l3758_375832

theorem inspection_ratio (j n : ℝ) (hj : j > 0) (hn : n > 0) : 
  0.005 * j + 0.007 * n = 0.0075 * (j + n) → n / j = 5 := by sorry

end NUMINAMATH_CALUDE_inspection_ratio_l3758_375832


namespace NUMINAMATH_CALUDE_blue_balls_unchanged_jungkook_blue_balls_l3758_375874

/-- Represents the number of balls of each color Jungkook has -/
structure BallCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Jungkook's initial ball count -/
def initial_count : BallCount :=
  { red := 5, blue := 4, yellow := 3 }

/-- Yoon-gi gives Jungkook a yellow ball -/
def give_yellow_ball (count : BallCount) : BallCount :=
  { count with yellow := count.yellow + 1 }

/-- The number of blue balls remains unchanged after receiving a yellow ball -/
theorem blue_balls_unchanged (count : BallCount) :
  (give_yellow_ball count).blue = count.blue :=
by sorry

/-- Jungkook has 4 blue balls after receiving a yellow ball from Yoon-gi -/
theorem jungkook_blue_balls :
  (give_yellow_ball initial_count).blue = 4 :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_unchanged_jungkook_blue_balls_l3758_375874


namespace NUMINAMATH_CALUDE_tile_border_ratio_l3758_375816

/-- Represents the arrangement of tiles in a square garden -/
structure TileArrangement where
  n : ℕ               -- Number of tiles along one side of the garden
  s : ℝ               -- Side length of each tile in meters
  d : ℝ               -- Width of the border around each tile in meters
  h_positive_s : 0 < s
  h_positive_d : 0 < d

/-- The theorem stating the ratio of border width to tile side length -/
theorem tile_border_ratio (arr : TileArrangement) (h_n : arr.n = 30) 
  (h_coverage : (arr.n^2 * arr.s^2) / ((arr.n * arr.s + 2 * arr.n * arr.d)^2) = 0.81) :
  arr.d / arr.s = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l3758_375816


namespace NUMINAMATH_CALUDE_equation_real_solution_l3758_375888

theorem equation_real_solution (x : ℝ) :
  (∀ y : ℝ, ∃ z : ℝ, x^2 + y^2 + z^2 + 2*x*y*z = 1) ↔ (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equation_real_solution_l3758_375888


namespace NUMINAMATH_CALUDE_parabola_properties_l3758_375801

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the parabola at a given x -/
def Parabola.eval (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A parabola that passes through the given points -/
def specificParabola : Parabola :=
  { a := sorry
    b := sorry
    c := sorry }

theorem parabola_properties :
  let p := specificParabola
  -- The parabola passes through the given points
  (p.eval (-2) = 0) ∧
  (p.eval (-1) = 4) ∧
  (p.eval 0 = 6) ∧
  (p.eval 1 = 6) →
  -- 1. The parabola opens downwards
  (p.a < 0) ∧
  -- 2. The axis of symmetry is x = 1/2
  (- p.b / (2 * p.a) = 1/2) ∧
  -- 3. The maximum value of the function is 25/4
  (p.c - p.b^2 / (4 * p.a) = 25/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3758_375801


namespace NUMINAMATH_CALUDE_bug_meeting_point_l3758_375863

/-- Triangle with sides a, b, c and point S on perimeter --/
structure Triangle (a b c : ℝ) where
  S : ℝ
  h1 : 0 < a ∧ 0 < b ∧ 0 < c
  h2 : a + b > c ∧ b + c > a ∧ c + a > b
  h3 : 0 ≤ S ∧ S ≤ a + b + c

/-- The length of QS in the triangle --/
def qsLength (t : Triangle 7 8 9) : ℝ :=
  5

theorem bug_meeting_point (t : Triangle 7 8 9) : qsLength t = 5 := by
  sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l3758_375863


namespace NUMINAMATH_CALUDE_minimum_values_l3758_375877

theorem minimum_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 + y^2 ≥ 1/2) ∧ (1/x + 1/y + 1/(x*y) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_minimum_values_l3758_375877


namespace NUMINAMATH_CALUDE_max_distance_sum_l3758_375875

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a line passing through F₁
def line_through_F₁ (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x + 1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ ellipse x y ∧ line_through_F₁ m x y}

-- Statement of the theorem
theorem max_distance_sum :
  ∀ (m : ℝ), ∃ (A B : ℝ × ℝ),
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    A ≠ B ∧
    (∀ (A' B' : ℝ × ℝ),
      A' ∈ intersection_points m →
      B' ∈ intersection_points m →
      A' ≠ B' →
      dist A' F₂ + dist B' F₂ ≤ 5) ∧
    (∃ (m' : ℝ), ∃ (A' B' : ℝ × ℝ),
      A' ∈ intersection_points m' ∧
      B' ∈ intersection_points m' ∧
      A' ≠ B' ∧
      dist A' F₂ + dist B' F₂ = 5) :=
sorry


end NUMINAMATH_CALUDE_max_distance_sum_l3758_375875


namespace NUMINAMATH_CALUDE_circle_radius_in_isosceles_triangle_l3758_375898

theorem circle_radius_in_isosceles_triangle (a b c : Real) (r_p r_q : Real) : 
  a = 60 → b = 60 → c = 40 → r_p = 12 →
  -- Triangle ABC is isosceles with AB = AC = 60 and BC = 40
  -- Circle P has radius r_p = 12 and is tangent to AC and BC
  -- Circle Q is externally tangent to P and tangent to AB and BC
  -- No point of circle Q lies outside of triangle ABC
  r_q = 36 - 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_isosceles_triangle_l3758_375898


namespace NUMINAMATH_CALUDE_simplify_expression_l3758_375831

theorem simplify_expression (x : ℝ) : (3*x + 20) + (200*x + 45) = 203*x + 65 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3758_375831


namespace NUMINAMATH_CALUDE_range_of_a_l3758_375868

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) → 
  1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3758_375868


namespace NUMINAMATH_CALUDE_even_function_quadratic_behavior_l3758_375867

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- State the theorem
theorem even_function_quadratic_behavior :
  ∀ m : ℝ, (∀ x : ℝ, f m x = f m (-x)) →
  (m = 0 ∧
   ∃ c : ℝ, c ∈ (Set.Ioo (-4) 2) ∧
   (∀ x y : ℝ, x ∈ (Set.Ioo (-4) c) → y ∈ (Set.Ioo (-4) c) → x < y → f m x < f m y) ∧
   (∀ x y : ℝ, x ∈ (Set.Ioo c 2) → y ∈ (Set.Ioo c 2) → x < y → f m x > f m y)) :=
by sorry

end NUMINAMATH_CALUDE_even_function_quadratic_behavior_l3758_375867


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l3758_375805

/-- Given a cone whose lateral surface development has a central angle of α radians,
    the vertex angle of its axial section is equal to 2 * arcsin(α / (2π)). -/
theorem cone_vertex_angle (α : ℝ) (h : 0 < α ∧ α < 2 * Real.pi) :
  let vertex_angle := 2 * Real.arcsin (α / (2 * Real.pi))
  vertex_angle = 2 * Real.arcsin (α / (2 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l3758_375805


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l3758_375899

theorem jellybean_box_capacity 
  (bert_capacity : ℕ)
  (bert_volume : ℝ)
  (lisa_volume : ℝ)
  (h1 : bert_capacity = 150)
  (h2 : lisa_volume = 24 * bert_volume)
  (h3 : ∀ (c : ℝ) (v : ℝ), c / v = bert_capacity / bert_volume → c = (v / bert_volume) * bert_capacity)
  : (lisa_volume / bert_volume) * bert_capacity = 3600 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_box_capacity_l3758_375899


namespace NUMINAMATH_CALUDE_grass_stains_count_l3758_375845

theorem grass_stains_count (grass_stain_time marinara_stain_time total_time : ℕ) 
  (marinara_stain_count : ℕ) (h1 : grass_stain_time = 4) 
  (h2 : marinara_stain_time = 7) (h3 : marinara_stain_count = 1) 
  (h4 : total_time = 19) : 
  ∃ (grass_stain_count : ℕ), 
    grass_stain_count * grass_stain_time + 
    marinara_stain_count * marinara_stain_time = total_time ∧ 
    grass_stain_count = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_grass_stains_count_l3758_375845


namespace NUMINAMATH_CALUDE_fraction_equality_l3758_375829

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 3 / 5) 
  (h2 : r / t = 8 / 9) : 
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3758_375829


namespace NUMINAMATH_CALUDE_carlos_class_size_l3758_375870

theorem carlos_class_size :
  ∃! b : ℕ, 80 < b ∧ b < 150 ∧
  ∃ k₁ : ℕ, b = 3 * k₁ - 2 ∧
  ∃ k₂ : ℕ, b = 4 * k₂ - 3 ∧
  ∃ k₃ : ℕ, b = 5 * k₃ - 4 ∧
  b = 121 := by
sorry

end NUMINAMATH_CALUDE_carlos_class_size_l3758_375870


namespace NUMINAMATH_CALUDE_total_amount_proof_l3758_375891

def coffee_maker_price : ℝ := 70
def blender_price : ℝ := 100
def coffee_maker_discount : ℝ := 0.2
def blender_discount : ℝ := 0.15
def num_coffee_makers : ℕ := 2

def total_price : ℝ :=
  (num_coffee_makers : ℝ) * coffee_maker_price * (1 - coffee_maker_discount) +
  blender_price * (1 - blender_discount)

theorem total_amount_proof :
  total_price = 197 := by sorry

end NUMINAMATH_CALUDE_total_amount_proof_l3758_375891


namespace NUMINAMATH_CALUDE_reduction_equivalence_l3758_375826

def operation (seq : Vector ℤ 8) : Vector ℤ 8 :=
  Vector.ofFn (λ i => |seq.get i - seq.get ((i + 1) % 8)|)

def all_equal (seq : Vector ℤ 8) : Prop :=
  ∀ i j, seq.get i = seq.get j

def all_zero (seq : Vector ℤ 8) : Prop :=
  ∀ i, seq.get i = 0

def reduces_to_equal (init : Vector ℤ 8) : Prop :=
  ∃ n : ℕ, all_equal (n.iterate operation init)

def reduces_to_zero (init : Vector ℤ 8) : Prop :=
  ∃ n : ℕ, all_zero (n.iterate operation init)

theorem reduction_equivalence (init : Vector ℤ 8) :
  reduces_to_equal init ↔ reduces_to_zero init :=
sorry

end NUMINAMATH_CALUDE_reduction_equivalence_l3758_375826


namespace NUMINAMATH_CALUDE_campground_distance_l3758_375869

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance : 
  distance_to_campground 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l3758_375869


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3758_375819

/-- Proves that for a hyperbola with given parameters, the sum of h, k, a, and b equals 6 + 2√10 -/
theorem hyperbola_sum (h k : ℝ) (focus_y vertex_y : ℝ) : 
  h = 1 → 
  k = 2 → 
  focus_y = 9 → 
  vertex_y = -1 → 
  let a := |k - vertex_y|
  let c := |k - focus_y|
  let b := Real.sqrt (c^2 - a^2)
  h + k + a + b = 6 + 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3758_375819


namespace NUMINAMATH_CALUDE_min_value_of_f_l3758_375886

def f (x : ℝ) := -2 * x + 5

theorem min_value_of_f :
  ∀ x ∈ Set.Icc 2 4, f x ≥ f 4 ∧ f 4 = -3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3758_375886


namespace NUMINAMATH_CALUDE_discount_difference_l3758_375824

def original_bill : ℝ := 15000

def single_discount_rate : ℝ := 0.3
def first_successive_discount_rate : ℝ := 0.25
def second_successive_discount_rate : ℝ := 0.06

def single_discount_amount : ℝ := original_bill * (1 - single_discount_rate)
def successive_discount_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)

theorem discount_difference :
  successive_discount_amount - single_discount_amount = 75 := by sorry

end NUMINAMATH_CALUDE_discount_difference_l3758_375824


namespace NUMINAMATH_CALUDE_total_pencils_l3758_375844

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) :
  jessica_pencils = 8 →
  sandy_pencils = 8 →
  jason_pencils = 8 →
  jessica_pencils + sandy_pencils + jason_pencils = 24 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l3758_375844


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l3758_375871

/-- The parabola function -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + 2*m - 1

theorem parabola_intersection_range (m : ℝ) :
  (∃ x₁ x₂, x₁ < 2 ∧ x₂ > 2 ∧ f m x₁ = 0 ∧ f m x₂ = 0) →  -- Intersects x-axis at two points
  (f m 0 < -1/2) →  -- Intersects y-axis below (0, -1/2)
  1/6 < m ∧ m < 1/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l3758_375871


namespace NUMINAMATH_CALUDE_greatest_integer_abs_inequality_l3758_375834

theorem greatest_integer_abs_inequality :
  (∃ (x : ℤ), ∀ (y : ℤ), |3*y - 2| ≤ 21 → y ≤ x) ∧
  (∀ (x : ℤ), |3*x - 2| ≤ 21 → x ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_abs_inequality_l3758_375834


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3758_375815

theorem sphere_volume_ratio (r R : ℝ) (h : R = 4 * r) :
  (4 / 3 * Real.pi * R^3) / (4 / 3 * Real.pi * r^3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3758_375815


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3758_375846

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 10 - 5 * Complex.I) :
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3758_375846


namespace NUMINAMATH_CALUDE_circular_triangle_angle_sum_l3758_375884

/-- Represents a circular triangle --/
structure CircularTriangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  r_a : ℝ  -- radius of arc forming side a
  r_b : ℝ  -- radius of arc forming side b
  r_c : ℝ  -- radius of arc forming side c
  s_a : Int  -- sign of side a (1 or -1)
  s_b : Int  -- sign of side b (1 or -1)
  s_c : Int  -- sign of side c (1 or -1)

/-- The theorem about the sum of angles in a circular triangle --/
theorem circular_triangle_angle_sum (t : CircularTriangle) :
  t.A + t.B + t.C - (t.s_a : ℝ) * (t.a / t.r_a) - (t.s_b : ℝ) * (t.b / t.r_b) - (t.s_c : ℝ) * (t.c / t.r_c) = π :=
by sorry

end NUMINAMATH_CALUDE_circular_triangle_angle_sum_l3758_375884


namespace NUMINAMATH_CALUDE_line_l_line_l_l3758_375856

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

/-- The equation of line l' that passes through (-1, 3) and is parallel to l -/
def line_l'_parallel (x y : ℝ) : Prop := 3 * x + 4 * y - 9 = 0

/-- The equation of line l' that is symmetric to l about the y-axis -/
def line_l'_symmetric (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

/-- Point (-1, 3) -/
def point : ℝ × ℝ := (-1, 3)

theorem line_l'_parallel_correct :
  (∀ x y, line_l'_parallel x y ↔ (∃ k, y - point.2 = k * (x - point.1) ∧
    ∀ x₁ y₁ x₂ y₂, line_l x₁ y₁ → line_l x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = k)) ∧
  line_l'_parallel point.1 point.2 :=
sorry

theorem line_l'_symmetric_correct :
  ∀ x y, line_l'_symmetric x y ↔ line_l (-x) y :=
sorry

end NUMINAMATH_CALUDE_line_l_line_l_l3758_375856


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3758_375825

/-- A quadratic function f(x) = x^2 + ax + b with specific properties -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

theorem quadratic_function_properties (a b : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), QuadraticFunction a b y ≥ QuadraticFunction a b x ∧ QuadraticFunction a b x = 2) →
  (∀ (x : ℝ), QuadraticFunction a b (2 - x) = QuadraticFunction a b x) →
  (∃ (m n : ℝ), m < n ∧
    (∀ (x : ℝ), m ≤ x ∧ x ≤ n → QuadraticFunction a b x ≤ 6) ∧
    (∃ (x : ℝ), m ≤ x ∧ x ≤ n ∧ QuadraticFunction a b x = 6)) →
  (∃ (m n : ℝ), n - m = 4 ∧
    ∀ (m' n' : ℝ), (∀ (x : ℝ), m' ≤ x ∧ x ≤ n' → QuadraticFunction a b x ≤ 6) →
    (∃ (x : ℝ), m' ≤ x ∧ x ≤ n' ∧ QuadraticFunction a b x = 6) →
    n' - m' ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3758_375825


namespace NUMINAMATH_CALUDE_total_coins_is_21_l3758_375876

/-- Represents the coin distribution pattern between Pete and Paul -/
def coin_distribution (x : ℕ) : Prop :=
  ∃ (paul_coins : ℕ) (pete_coins : ℕ),
    paul_coins = x ∧
    pete_coins = 6 * x ∧
    pete_coins = x * (x + 1) * (x + 2) / 6

/-- The total number of coins is 21 -/
theorem total_coins_is_21 : ∃ (x : ℕ), coin_distribution x ∧ x + 6 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_21_l3758_375876


namespace NUMINAMATH_CALUDE_james_cattle_profit_l3758_375838

def cattle_profit (num_cattle : ℕ) (purchase_price : ℕ) (feeding_cost_percentage : ℕ) 
                  (weight_per_cattle : ℕ) (selling_price_per_pound : ℕ) : ℕ :=
  let feeding_cost := purchase_price * feeding_cost_percentage / 100
  let total_cost := purchase_price + feeding_cost
  let selling_price_per_cattle := weight_per_cattle * selling_price_per_pound
  let total_selling_price := num_cattle * selling_price_per_cattle
  total_selling_price - total_cost

theorem james_cattle_profit :
  cattle_profit 100 40000 20 1000 2 = 112000 := by
  sorry

end NUMINAMATH_CALUDE_james_cattle_profit_l3758_375838


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l3758_375881

theorem merchant_profit_percentage (C S : ℝ) (h : C > 0) :
  20 * C = 15 * S →
  (S - C) / C * 100 = 100/3 :=
by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l3758_375881


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l3758_375804

/-- The number of ways to distribute 10 colored balls into two boxes -/
def distribute_balls : ℕ :=
  Nat.choose 10 4

/-- The total number of balls -/
def total_balls : ℕ := 10

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of green balls -/
def green_balls : ℕ := 2

/-- The capacity of the smaller box -/
def small_box_capacity : ℕ := 4

/-- The capacity of the larger box -/
def large_box_capacity : ℕ := 6

theorem ball_distribution_theorem :
  distribute_balls = 210 ∧
  total_balls = red_balls + white_balls + green_balls ∧
  total_balls = small_box_capacity + large_box_capacity :=
sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l3758_375804


namespace NUMINAMATH_CALUDE_man_speed_against_current_l3758_375811

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that for the given conditions, 
    the man's speed against the current is 14 km/h. -/
theorem man_speed_against_current :
  speed_against_current 20 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l3758_375811


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3758_375818

/-- A parabola with equation y = x^2 - 4x + c has its vertex on the x-axis if and only if c = 4 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + c = 0 ∧ ∀ y : ℝ, y^2 - 4*y + c ≥ x^2 - 4*x + c) ↔ c = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3758_375818


namespace NUMINAMATH_CALUDE_stating_auntie_em_parking_probability_l3758_375833

/-- The number of parking spaces in the lot -/
def total_spaces : ℕ := 20

/-- The number of cars that arrive before Auntie Em -/
def cars_before : ℕ := 15

/-- The number of spaces Auntie Em's SUV requires -/
def suv_spaces : ℕ := 2

/-- The probability that Auntie Em can park her SUV -/
def prob_auntie_em_can_park : ℚ := 232 / 323

/-- 
Theorem stating that the probability of Auntie Em being able to park her SUV
is equal to 232/323, given the conditions of the parking lot problem.
-/
theorem auntie_em_parking_probability :
  let remaining_spaces := total_spaces - cars_before
  let total_arrangements := Nat.choose total_spaces cars_before
  let unfavorable_arrangements := Nat.choose (remaining_spaces + cars_before - 1) (remaining_spaces - 1)
  (1 : ℚ) - (unfavorable_arrangements : ℚ) / (total_arrangements : ℚ) = prob_auntie_em_can_park :=
by sorry

end NUMINAMATH_CALUDE_stating_auntie_em_parking_probability_l3758_375833


namespace NUMINAMATH_CALUDE_min_sort_steps_l3758_375810

/-- Represents the color of a cow -/
inductive Color
| Purple
| White

/-- A configuration of cows -/
def Configuration (n : ℕ) := Fin (2 * n) → Color

/-- A valid swap operation on a configuration -/
def ValidSwap (n : ℕ) (c : Configuration n) (i j : ℕ) : Prop :=
  i < j ∧ j ≤ 2 * n ∧ j - i = 2 * n - j

/-- The number of steps required to sort a configuration -/
def SortSteps (n : ℕ) (c : Configuration n) : ℕ := sorry

/-- The theorem stating that n steps are always sufficient and sometimes necessary -/
theorem min_sort_steps (n : ℕ) :
  (∀ c : Configuration n, SortSteps n c ≤ n) ∧
  (∃ c : Configuration n, SortSteps n c = n) := by sorry

end NUMINAMATH_CALUDE_min_sort_steps_l3758_375810


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3758_375808

theorem right_triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (pythagorean : a^2 + b^2 = c^2) : (a + b) / Real.sqrt 2 ≤ c := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3758_375808


namespace NUMINAMATH_CALUDE_cafeteria_apples_l3758_375895

/-- The cafeteria problem -/
theorem cafeteria_apples (initial : ℕ) (used : ℕ) (bought : ℕ) :
  initial = 38 → used = 20 → bought = 28 → initial - used + bought = 46 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l3758_375895


namespace NUMINAMATH_CALUDE_felix_lifting_capacity_l3758_375847

/-- Felix's lifting capacity problem -/
theorem felix_lifting_capacity 
  (felix_lift_ratio : ℝ) 
  (brother_weight_ratio : ℝ) 
  (brother_lift_ratio : ℝ) 
  (brother_lift_weight : ℝ) 
  (h1 : felix_lift_ratio = 1.5)
  (h2 : brother_weight_ratio = 2)
  (h3 : brother_lift_ratio = 3)
  (h4 : brother_lift_weight = 600) :
  felix_lift_ratio * (brother_lift_weight / brother_lift_ratio / brother_weight_ratio) = 150 := by
  sorry


end NUMINAMATH_CALUDE_felix_lifting_capacity_l3758_375847


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3758_375864

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < s.sample_size ∧ n = s.first_element + k * s.interval

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 52)
  (h_sample : s.sample_size = 4)
  (h_5 : s.contains 5)
  (h_31 : s.contains 31)
  (h_44 : s.contains 44)
  : s.contains 18 := by
  sorry

#check systematic_sample_fourth_element

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3758_375864


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3758_375860

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 ∧ b = 8 ∧ c^2 - 14*c + 40 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3758_375860


namespace NUMINAMATH_CALUDE_quadratic_polynomials_theorem_l3758_375865

theorem quadratic_polynomials_theorem (a b c d : ℝ) : 
  let f (x : ℝ) := x^2 + a*x + b
  let g (x : ℝ) := x^2 + c*x + d
  (∀ x, f x ≠ g x) →  -- f and g are distinct
  g (-a/2) = 0 →  -- x-coordinate of vertex of f is a root of g
  f (-c/2) = 0 →  -- x-coordinate of vertex of g is a root of f
  f 50 = -50 ∧ g 50 = -50 →  -- f and g intersect at (50, -50)
  (∃ x₁ x₂, ∀ x, f x ≥ f x₁ ∧ g x ≥ g x₂ ∧ f x₁ = g x₂) →  -- minimum value of f is the same as g
  a + c = -200 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_theorem_l3758_375865


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l3758_375809

theorem quadratic_roots_existence : ∃ (p q : ℝ), 
  ((p - 1)^2 - 4*q > 0) ∧ 
  ((p + 1)^2 - 4*q > 0) ∧ 
  (p^2 - 4*q < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l3758_375809


namespace NUMINAMATH_CALUDE_linear_function_intersection_l3758_375853

/-- The linear function that intersects with two given lines -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_intersection : 
  ∃ k b : ℝ, 
    (linear_function k b 4 = -4 + 6) ∧ 
    (linear_function k b (1 + 1) = 1) ∧
    (∀ x : ℝ, linear_function k b x = (1/2) * x) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l3758_375853


namespace NUMINAMATH_CALUDE_blue_shells_count_l3758_375896

theorem blue_shells_count (total purple pink yellow orange : ℕ) 
  (h_total : total = 65)
  (h_purple : purple = 13)
  (h_pink : pink = 8)
  (h_yellow : yellow = 18)
  (h_orange : orange = 14) :
  total - (purple + pink + yellow + orange) = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_shells_count_l3758_375896


namespace NUMINAMATH_CALUDE_volume_cone_from_right_triangle_l3758_375820

/-- The volume of a cone formed by rotating a right triangle around its hypotenuse -/
theorem volume_cone_from_right_triangle (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let v := (1 / 3) * π * r^2 * h
  h = 2 ∧ r = 1 → v = (2 * π) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_cone_from_right_triangle_l3758_375820


namespace NUMINAMATH_CALUDE_necessary_condition_abs_l3758_375836

theorem necessary_condition_abs (x y : ℝ) (hx : x > 0) : x > |y| → x > y := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_abs_l3758_375836


namespace NUMINAMATH_CALUDE_num_perfect_square_factors_is_525_l3758_375835

/-- The number of positive perfect square factors of 2^8 * 3^9 * 5^12 * 7^4 -/
def num_perfect_square_factors : ℕ := 525

/-- The exponents of prime factors in the given product -/
def prime_exponents : List ℕ := [8, 9, 12, 4]

/-- Counts the number of even numbers (including 0) up to and including a given number -/
def count_even_numbers_up_to (n : ℕ) : ℕ :=
  (n / 2) + 1

/-- Theorem: The number of positive perfect square factors of 2^8 * 3^9 * 5^12 * 7^4 is 525 -/
theorem num_perfect_square_factors_is_525 :
  num_perfect_square_factors = (prime_exponents.map count_even_numbers_up_to).prod :=
sorry

end NUMINAMATH_CALUDE_num_perfect_square_factors_is_525_l3758_375835


namespace NUMINAMATH_CALUDE_floor_multiple_implies_integer_l3758_375849

theorem floor_multiple_implies_integer (r : ℝ) : 
  r ≥ 1 →
  (∀ (m n : ℕ+), n.val % m.val = 0 → (⌊n.val * r⌋ : ℤ) % (⌊m.val * r⌋ : ℤ) = 0) →
  ∃ (k : ℤ), r = k := by
  sorry

end NUMINAMATH_CALUDE_floor_multiple_implies_integer_l3758_375849


namespace NUMINAMATH_CALUDE_smallest_three_digit_perfect_square_append_l3758_375830

theorem smallest_three_digit_perfect_square_append : ∃ (n : ℕ), 
  (n = 183) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬(∃ k : ℕ, 1000 * m + (m + 1) = k^2)) ∧
  (∃ k : ℕ, 1000 * n + (n + 1) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_perfect_square_append_l3758_375830


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l3758_375859

/-- Represents the peg game with n holes and k pegs. -/
structure PegGame where
  n : ℕ
  k : ℕ
  h1 : 1 ≤ k
  h2 : k < n

/-- Predicate that determines if Alice has a winning strategy. -/
def alice_wins (game : PegGame) : Prop :=
  ¬(Even game.n ∧ Even game.k)

/-- The main theorem about Alice's winning strategy in the peg game. -/
theorem alice_winning_strategy (game : PegGame) :
  alice_wins game ↔
  (∃ (strategy : Unit), 
    (∀ (bob_move : Unit), ∃ (alice_move : Unit), 
      -- Alice can always make a move that leads to a winning position
      true)) := by sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l3758_375859


namespace NUMINAMATH_CALUDE_two_number_problem_l3758_375839

theorem two_number_problem :
  ∃ (x y : ℝ), 38 + 2 * x = 124 ∧ x + 3 * y = 47 ∧ x = 43 ∧ y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l3758_375839


namespace NUMINAMATH_CALUDE_sand_bags_problem_l3758_375813

/-- Given that each bag has a capacity of 65 pounds and 12 bags are needed,
    prove that the total pounds of sand is 780. -/
theorem sand_bags_problem (bag_capacity : ℕ) (num_bags : ℕ) 
    (h1 : bag_capacity = 65) (h2 : num_bags = 12) : 
    bag_capacity * num_bags = 780 := by
  sorry

end NUMINAMATH_CALUDE_sand_bags_problem_l3758_375813


namespace NUMINAMATH_CALUDE_number_less_than_abs_is_negative_l3758_375843

theorem number_less_than_abs_is_negative (x : ℝ) : x < |x| → x < 0 := by
  sorry

end NUMINAMATH_CALUDE_number_less_than_abs_is_negative_l3758_375843


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l3758_375841

theorem solution_implies_a_value (x a : ℝ) : x = 1 ∧ 2 * x - a = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l3758_375841


namespace NUMINAMATH_CALUDE_sequence_value_l3758_375823

theorem sequence_value (n : ℕ) (a : ℕ → ℕ) : 
  (∀ k, a k = 3 * k + 4) → a n = 13 → n = 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_value_l3758_375823


namespace NUMINAMATH_CALUDE_integer_root_characterization_l3758_375855

def polynomial (x b : ℤ) : ℤ := x^4 + 4*x^3 + 4*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_characterization (b : ℤ) :
  has_integer_root b ↔ b ∈ ({-38, -21, -2, 10, 13, 34} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_root_characterization_l3758_375855


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3758_375878

theorem sum_of_four_numbers (a b c d : ℕ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a > d)
  (h2 : a * b = c * d)
  (h3 : a + b + c + d = a * c) :
  a + b + c + d = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3758_375878


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l3758_375852

theorem no_solution_for_inequality :
  ¬∃ (x : ℝ), x > 0 ∧ x * Real.sqrt (10 - x) + Real.sqrt (10 * x - x^3) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l3758_375852


namespace NUMINAMATH_CALUDE_kenny_trumpet_practice_l3758_375857

/-- Given Kenny's activities and their durations, prove that he practiced trumpet for 40 hours. -/
theorem kenny_trumpet_practice (x y z w : ℕ) : 
  let basketball : ℕ := 10
  let running : ℕ := 2 * basketball
  let trumpet : ℕ := 2 * running
  let other_activities : ℕ := x + y + z + w
  other_activities = basketball + running + trumpet - 5
  → trumpet = 40 := by
sorry

end NUMINAMATH_CALUDE_kenny_trumpet_practice_l3758_375857


namespace NUMINAMATH_CALUDE_circle_radius_with_tangent_l3758_375893

/-- The radius of a circle with equation x^2 + y^2 = 25 and a tangent at y = 5 is 5 -/
theorem circle_radius_with_tangent (x y : ℝ) :
  x^2 + y^2 = 25 → ∃ (x₀ : ℝ), x₀^2 + 5^2 = 25 → 
  Real.sqrt ((0 - x₀)^2 + (5 - 0)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_with_tangent_l3758_375893


namespace NUMINAMATH_CALUDE_dandelion_seed_percentage_l3758_375812

/-- Represents the number of sunflowers Carla has -/
def num_sunflowers : ℕ := 6

/-- Represents the number of dandelions Carla has -/
def num_dandelions : ℕ := 8

/-- Represents the number of seeds per sunflower -/
def seeds_per_sunflower : ℕ := 9

/-- Represents the number of seeds per dandelion -/
def seeds_per_dandelion : ℕ := 12

/-- Calculates the total number of seeds from sunflowers -/
def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower

/-- Calculates the total number of seeds from dandelions -/
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion

/-- Calculates the total number of seeds -/
def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds

/-- Theorem: The percentage of seeds from dandelions is 64% -/
theorem dandelion_seed_percentage : 
  (total_dandelion_seeds : ℚ) / (total_seeds : ℚ) * 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_percentage_l3758_375812


namespace NUMINAMATH_CALUDE_quadratic_transformations_integer_roots_l3758_375854

/-- 
Given a quadratic equation x^2 + px + q = 0, where p and q are integers,
this function returns true if the equation has integer roots.
-/
def has_integer_roots (p q : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x ≠ y

/-- 
This theorem states that there exist initial integer values for p and q
such that the quadratic equation x^2 + px + q = 0 and its nine transformations
(where p and q are increased by 1 each time) all have integer roots.
-/
theorem quadratic_transformations_integer_roots :
  ∃ p q : ℤ, 
    (∀ i : ℕ, i ≤ 9 → has_integer_roots (p + i) (q + i)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformations_integer_roots_l3758_375854


namespace NUMINAMATH_CALUDE_final_water_percentage_l3758_375889

/-- Calculates the final percentage of water in a mixture after adding water -/
theorem final_water_percentage
  (initial_mixture : ℝ)
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (h_initial_mixture : initial_mixture = 50)
  (h_initial_water_percentage : initial_water_percentage = 10)
  (h_added_water : added_water = 25) :
  let initial_water := initial_mixture * (initial_water_percentage / 100)
  let final_water := initial_water + added_water
  let final_mixture := initial_mixture + added_water
  (final_water / final_mixture) * 100 = 40 := by
sorry


end NUMINAMATH_CALUDE_final_water_percentage_l3758_375889


namespace NUMINAMATH_CALUDE_zoo_penguins_l3758_375842

theorem zoo_penguins (penguins : ℕ) (polar_bears : ℕ) : 
  polar_bears = 2 * penguins → 
  penguins + polar_bears = 63 → 
  penguins = 21 := by
sorry

end NUMINAMATH_CALUDE_zoo_penguins_l3758_375842


namespace NUMINAMATH_CALUDE_existence_of_common_element_l3758_375872

theorem existence_of_common_element (ε : ℝ) (h_ε_pos : 0 < ε) (h_ε_bound : ε < 1/2) :
  ∃ m : ℕ+, ∀ x : ℝ, ∃ i : ℕ+, ∃ k : ℤ, i.val ≤ m.val ∧ |i.val • x - k| ≤ ε :=
sorry

end NUMINAMATH_CALUDE_existence_of_common_element_l3758_375872


namespace NUMINAMATH_CALUDE_dolls_count_l3758_375866

/-- The number of dolls Jane has -/
def jane_dolls : ℕ := 13

/-- The difference between Jill's and Jane's dolls -/
def doll_difference : ℕ := 6

/-- The total number of dolls Jane and Jill have together -/
def total_dolls : ℕ := jane_dolls + (jane_dolls + doll_difference)

theorem dolls_count : total_dolls = 32 := by sorry

end NUMINAMATH_CALUDE_dolls_count_l3758_375866


namespace NUMINAMATH_CALUDE_power_of_power_l3758_375892

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3758_375892


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3758_375858

/-- Given an isosceles triangle with two equal sides of 15 cm and a base of 24 cm,
    prove that a similar triangle with a base of 60 cm has a perimeter of 135 cm. -/
theorem similar_triangle_perimeter 
  (original_equal_sides : ℝ)
  (original_base : ℝ)
  (similar_base : ℝ)
  (h_isosceles : original_equal_sides = 15)
  (h_original_base : original_base = 24)
  (h_similar_base : similar_base = 60) :
  let scale_factor := similar_base / original_base
  let similar_equal_sides := original_equal_sides * scale_factor
  similar_equal_sides * 2 + similar_base = 135 :=
by
  sorry

#check similar_triangle_perimeter

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3758_375858


namespace NUMINAMATH_CALUDE_sum_of_central_angles_is_360_l3758_375807

/-- A circle with an inscribed pentagon -/
structure PentagonInCircle where
  /-- The circle -/
  circle : Set ℝ × Set ℝ
  /-- The inscribed pentagon -/
  pentagon : Set (ℝ × ℝ)
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The vertices of the pentagon -/
  vertices : Fin 5 → ℝ × ℝ
  /-- The lines from vertices to center -/
  lines : Fin 5 → Set (ℝ × ℝ)

/-- The sum of angles at the center formed by lines from pentagon vertices to circle center -/
def sumOfCentralAngles (p : PentagonInCircle) : ℝ := sorry

/-- Theorem: The sum of central angles in a pentagon inscribed in a circle is 360° -/
theorem sum_of_central_angles_is_360 (p : PentagonInCircle) : 
  sumOfCentralAngles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_of_central_angles_is_360_l3758_375807


namespace NUMINAMATH_CALUDE_f_composition_at_one_l3758_375828

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 2^x

theorem f_composition_at_one :
  f (f 1) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_one_l3758_375828


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l3758_375887

theorem quadratic_equation_transformation (a b c : ℝ) : 
  (∀ x, a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) →
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l3758_375887


namespace NUMINAMATH_CALUDE_bike_ride_time_l3758_375882

/-- Given a constant speed where 2 miles are covered in 8 minutes, 
    prove that the time required to cover 5 miles is 20 minutes. -/
theorem bike_ride_time (speed : ℝ) (distance_to_julia : ℝ) (time_to_julia : ℝ) 
  (distance_to_bernard : ℝ) : 
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  speed = distance_to_julia / time_to_julia →
  distance_to_bernard / speed = 20 := by
  sorry

#check bike_ride_time

end NUMINAMATH_CALUDE_bike_ride_time_l3758_375882


namespace NUMINAMATH_CALUDE_ann_initial_blocks_l3758_375861

/-- Given that Ann finds 44 blocks and ends with 53 blocks, prove that she initially had 9 blocks. -/
theorem ann_initial_blocks (found : ℕ) (final : ℕ) (h1 : found = 44) (h2 : final = 53) :
  final - found = 9 := by sorry

end NUMINAMATH_CALUDE_ann_initial_blocks_l3758_375861


namespace NUMINAMATH_CALUDE_milk_tea_sales_distribution_l3758_375894

/-- Represents the sales distribution of milk tea flavors -/
structure MilkTeaSales where
  total : ℕ
  winterMelon : ℕ
  okinawa : ℕ
  chocolate : ℕ
  thai : ℕ
  taro : ℕ

/-- Conditions for the milk tea sales problem -/
def salesConditions (s : MilkTeaSales) : Prop :=
  s.total = 100 ∧
  s.winterMelon = (35 * s.total) / 100 ∧
  s.okinawa = s.total / 4 ∧
  s.taro = 12 ∧
  3 * s.chocolate = 7 * s.thai ∧
  s.chocolate + s.thai = s.total - s.winterMelon - s.okinawa - s.taro

/-- Theorem stating the correct distribution of milk tea sales -/
theorem milk_tea_sales_distribution :
  ∃ (s : MilkTeaSales),
    salesConditions s ∧
    s.winterMelon = 35 ∧
    s.okinawa = 25 ∧
    s.chocolate = 8 ∧
    s.thai = 20 ∧
    s.taro = 12 ∧
    s.winterMelon + s.okinawa + s.chocolate + s.thai + s.taro = s.total :=
by
  sorry

end NUMINAMATH_CALUDE_milk_tea_sales_distribution_l3758_375894


namespace NUMINAMATH_CALUDE_ad_difference_l3758_375803

/-- Represents the number of ads on each web page -/
structure WebPageAds where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the problem -/
def adConditions (w : WebPageAds) : Prop :=
  w.first = 12 ∧
  w.second = 2 * w.first ∧
  w.third > w.second ∧
  w.fourth = (3 * w.second) / 4 ∧
  (2 * (w.first + w.second + w.third + w.fourth)) / 3 = 68

theorem ad_difference (w : WebPageAds) (h : adConditions w) : 
  w.third - w.second = 24 := by
sorry

end NUMINAMATH_CALUDE_ad_difference_l3758_375803


namespace NUMINAMATH_CALUDE_dannys_bottle_caps_l3758_375848

/-- Calculates the total number of bottle caps in Danny's collection -/
def total_bottle_caps (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Danny's total bottle caps is 55 -/
theorem dannys_bottle_caps :
  total_bottle_caps 37 18 = 55 := by
  sorry

end NUMINAMATH_CALUDE_dannys_bottle_caps_l3758_375848


namespace NUMINAMATH_CALUDE_break_even_books_l3758_375862

/-- Represents the fixed cost of making books -/
def fixed_cost : ℝ := 50000

/-- Represents the marketing cost per book -/
def marketing_cost_per_book : ℝ := 4

/-- Represents the selling price per book -/
def selling_price_per_book : ℝ := 9

/-- Calculates the total cost for a given number of books -/
def total_cost (num_books : ℝ) : ℝ :=
  fixed_cost + marketing_cost_per_book * num_books

/-- Calculates the revenue for a given number of books -/
def revenue (num_books : ℝ) : ℝ :=
  selling_price_per_book * num_books

/-- Theorem: The number of books needed to break even is 10000 -/
theorem break_even_books : 
  ∃ (x : ℝ), x = 10000 ∧ total_cost x = revenue x :=
by sorry

end NUMINAMATH_CALUDE_break_even_books_l3758_375862


namespace NUMINAMATH_CALUDE_dogs_liking_no_food_l3758_375890

def total_dogs : ℕ := 80
def watermelon_dogs : ℕ := 18
def salmon_dogs : ℕ := 58
def chicken_dogs : ℕ := 16
def watermelon_and_salmon : ℕ := 7
def chicken_and_salmon : ℕ := 6
def chicken_and_watermelon : ℕ := 4
def all_three : ℕ := 3

theorem dogs_liking_no_food : 
  total_dogs - (watermelon_dogs + salmon_dogs + chicken_dogs
              - watermelon_and_salmon - chicken_and_salmon - chicken_and_watermelon
              + all_three) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dogs_liking_no_food_l3758_375890


namespace NUMINAMATH_CALUDE_min_value_of_f_l3758_375885

def f (x : ℕ) : ℤ := 3 * x^2 - 12 * x + 800

theorem min_value_of_f :
  ∀ x : ℕ, f x ≥ 788 ∧ ∃ x₀ : ℕ, f x₀ = 788 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3758_375885


namespace NUMINAMATH_CALUDE_median_in_80_84_interval_l3758_375814

/-- Represents the score intervals --/
inductive ScoreInterval
  | interval_65_69
  | interval_70_74
  | interval_75_79
  | interval_80_84
  | interval_85_89
  | interval_90_94

/-- The number of students in each score interval --/
def studentCount (interval : ScoreInterval) : Nat :=
  match interval with
  | .interval_65_69 => 6
  | .interval_70_74 => 10
  | .interval_75_79 => 25
  | .interval_80_84 => 30
  | .interval_85_89 => 20
  | .interval_90_94 => 10

/-- The total number of students --/
def totalStudents : Nat := 101

/-- The position of the median in the dataset --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score is in the 80-84 interval --/
theorem median_in_80_84_interval :
  ∃ k, k ≤ medianPosition ∧
       k > (studentCount ScoreInterval.interval_90_94 +
            studentCount ScoreInterval.interval_85_89) ∧
       k ≤ (studentCount ScoreInterval.interval_90_94 +
            studentCount ScoreInterval.interval_85_89 +
            studentCount ScoreInterval.interval_80_84) :=
  sorry

end NUMINAMATH_CALUDE_median_in_80_84_interval_l3758_375814


namespace NUMINAMATH_CALUDE_inequality_proof_l3758_375851

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  a / Real.sqrt (a^2 + 1) + b / Real.sqrt (b^2 + 1) + c / Real.sqrt (c^2 + 1) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3758_375851


namespace NUMINAMATH_CALUDE_bottle_capacity_proof_l3758_375879

theorem bottle_capacity_proof (x : ℚ) : 
  (16/3 : ℚ) / 8 * x + 16/3 = 8 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_capacity_proof_l3758_375879


namespace NUMINAMATH_CALUDE_tobias_swims_3000_meters_l3758_375800

/-- The number of meters Tobias swims in 3 hours with regular pauses -/
def tobias_swim_distance : ℕ :=
  let total_time : ℕ := 3 * 60  -- 3 hours in minutes
  let swim_pause_cycle : ℕ := 25 + 5  -- 25 min swim + 5 min pause
  let num_cycles : ℕ := total_time / swim_pause_cycle
  let total_swim_time : ℕ := num_cycles * 25  -- Total swimming time in minutes
  let meters_per_5min : ℕ := 100  -- Swims 100 meters every 5 minutes
  total_swim_time / 5 * meters_per_5min

/-- Theorem stating that Tobias swims 3000 meters -/
theorem tobias_swims_3000_meters : tobias_swim_distance = 3000 := by
  sorry

#eval tobias_swim_distance  -- This should output 3000

end NUMINAMATH_CALUDE_tobias_swims_3000_meters_l3758_375800


namespace NUMINAMATH_CALUDE_transformation_identity_l3758_375806

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotation 180° about y-axis -/
def rotateY180 (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- Reflection through yz-plane -/
def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Rotation 90° about z-axis -/
def rotateZ90 (p : Point3D) : Point3D :=
  { x := p.y, y := -p.x, z := p.z }

/-- Reflection through xz-plane -/
def reflectXZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- Reflection through xy-plane -/
def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- The sequence of transformations -/
def transformSequence (p : Point3D) : Point3D :=
  reflectXY (reflectXZ (rotateZ90 (reflectYZ (rotateY180 p))))

theorem transformation_identity :
  transformSequence { x := 2, y := 2, z := 2 } = { x := 2, y := 2, z := 2 } := by
  sorry

end NUMINAMATH_CALUDE_transformation_identity_l3758_375806


namespace NUMINAMATH_CALUDE_three_number_average_l3758_375837

theorem three_number_average (a b c : ℝ) 
  (h1 : (a + b) / 2 = 26.5)
  (h2 : (b + c) / 2 = 34.5)
  (h3 : (a + c) / 2 = 29)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a + b + c) / 3 = 30 := by
sorry

end NUMINAMATH_CALUDE_three_number_average_l3758_375837


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3758_375827

theorem cubic_equation_solution (p q : ℝ) :
  ∃ x : ℝ, x^3 + p*x + q = 0 ∧
  x = -(Real.rpow ((q/2) + Real.sqrt ((q^2/4) + (p^3/27))) (1/3)) -
      (Real.rpow ((q/2) - Real.sqrt ((q^2/4) + (p^3/27))) (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3758_375827


namespace NUMINAMATH_CALUDE_smallest_non_factor_non_prime_l3758_375817

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_non_factor_non_prime : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    ¬(factorial 30 % n = 0) ∧ 
    ¬(Nat.Prime n) ∧
    (∀ m : ℕ, m > 0 ∧ m < n → 
      (factorial 30 % m = 0) ∨ (Nat.Prime m)) ∧
    n = 961 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_factor_non_prime_l3758_375817


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3758_375873

/-- An angle is in the third quadrant if it's between 180° and 270° (or equivalent in radians) -/
def is_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

/-- An angle is in the second or fourth quadrant if it's between 90° and 180° or between 270° and 360° (or equivalent in radians) -/
def is_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, (k * 2 * Real.pi + Real.pi / 2 < α ∧ α < k * 2 * Real.pi + Real.pi) ∨
           (k * 2 * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * 2 * Real.pi)

theorem half_angle_quadrant (α : Real) :
  is_third_quadrant α → is_second_or_fourth_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3758_375873


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3758_375883

/-- A quadratic function with vertex (3, 5) passing through (-2, -20) has a = -1 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (3, 5) = (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c) →  -- Condition 2
  a * (-2)^2 + b * (-2) + c = -20 →  -- Condition 3
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3758_375883


namespace NUMINAMATH_CALUDE_village_assistants_selection_l3758_375840

theorem village_assistants_selection (n : ℕ) (k : ℕ) (h_n : n = 10) (h_k : k = 3) :
  (Nat.choose n k) - (Nat.choose (n - 3) k) - 
  (2 * (Nat.choose (n - 2) (k - 1)) - (Nat.choose (n - 3) (k - 2))) = 49 := by
  sorry

end NUMINAMATH_CALUDE_village_assistants_selection_l3758_375840


namespace NUMINAMATH_CALUDE_tan_half_product_l3758_375802

theorem tan_half_product (a b : ℝ) :
  5 * (Real.cos a + Real.cos b) + 4 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 3 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -3) :=
by sorry

end NUMINAMATH_CALUDE_tan_half_product_l3758_375802


namespace NUMINAMATH_CALUDE_sequence_a_properties_l3758_375850

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * sequence_a (n + 1) - sequence_a n

theorem sequence_a_properties :
  (∀ n : ℕ, ∃ k : ℤ, sequence_a n = k) ∧
  (∀ n : ℕ, 3 ∣ sequence_a n ↔ 3 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_properties_l3758_375850
