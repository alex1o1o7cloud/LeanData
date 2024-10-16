import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l507_50721

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 
  (a + b + c)^2 / (a * b * (a + b) + b * c * (b + c) + c * a * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l507_50721


namespace NUMINAMATH_CALUDE_min_students_with_blue_shirt_and_red_shoes_l507_50708

theorem min_students_with_blue_shirt_and_red_shoes
  (n : ℕ)  -- Total number of students
  (blue_shirt : ℕ)  -- Number of students wearing blue shirts
  (red_shoes : ℕ)  -- Number of students wearing red shoes
  (h1 : blue_shirt = n * 3 / 7)  -- 3/7 of students wear blue shirts
  (h2 : red_shoes = n * 4 / 9)  -- 4/9 of students wear red shoes
  : ∃ (both : ℕ), both ≥ 8 ∧ blue_shirt + red_shoes - both = n :=
sorry

end NUMINAMATH_CALUDE_min_students_with_blue_shirt_and_red_shoes_l507_50708


namespace NUMINAMATH_CALUDE_is_vertex_of_parabola_l507_50761

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := -2 * x^2 - 20 * x - 50

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-5, 0)

/-- Theorem stating that the given point is the vertex of the parabola -/
theorem is_vertex_of_parabola :
  let (m, n) := vertex
  ∀ x : ℝ, parabola_equation x ≤ parabola_equation m :=
by sorry

end NUMINAMATH_CALUDE_is_vertex_of_parabola_l507_50761


namespace NUMINAMATH_CALUDE_towel_set_price_l507_50725

/-- The price of towel sets for guest and master bathrooms -/
theorem towel_set_price (guest_sets master_sets : ℕ) (master_price : ℝ) 
  (discount : ℝ) (total_spent : ℝ) (h1 : guest_sets = 2) 
  (h2 : master_sets = 4) (h3 : master_price = 50) 
  (h4 : discount = 0.2) (h5 : total_spent = 224) : 
  ∃ (guest_price : ℝ), guest_price = 40 ∧ 
  (1 - discount) * (guest_sets * guest_price + master_sets * master_price) = total_spent :=
by
  sorry

#check towel_set_price

end NUMINAMATH_CALUDE_towel_set_price_l507_50725


namespace NUMINAMATH_CALUDE_four_fives_equal_100_l507_50705

/-- An arithmetic expression using fives -/
inductive FiveExpr
  | Const : FiveExpr
  | Add : FiveExpr → FiveExpr → FiveExpr
  | Sub : FiveExpr → FiveExpr → FiveExpr
  | Mul : FiveExpr → FiveExpr → FiveExpr

/-- Evaluate a FiveExpr to an integer -/
def eval : FiveExpr → Int
  | FiveExpr.Const => 5
  | FiveExpr.Add a b => eval a + eval b
  | FiveExpr.Sub a b => eval a - eval b
  | FiveExpr.Mul a b => eval a * eval b

/-- Count the number of fives in a FiveExpr -/
def countFives : FiveExpr → Nat
  | FiveExpr.Const => 1
  | FiveExpr.Add a b => countFives a + countFives b
  | FiveExpr.Sub a b => countFives a + countFives b
  | FiveExpr.Mul a b => countFives a + countFives b

/-- Theorem: There exists an arithmetic expression using exactly four fives that equals 100 -/
theorem four_fives_equal_100 : ∃ e : FiveExpr, countFives e = 4 ∧ eval e = 100 := by
  sorry


end NUMINAMATH_CALUDE_four_fives_equal_100_l507_50705


namespace NUMINAMATH_CALUDE_dj_snake_engagement_treats_value_l507_50701

/-- The total value of treats received by DJ Snake on his engagement day -/
def total_value (hotel_nights : ℕ) (hotel_price_per_night : ℕ) (car_value : ℕ) : ℕ :=
  hotel_nights * hotel_price_per_night + car_value + 4 * car_value

/-- Theorem stating the total value of treats received by DJ Snake on his engagement day -/
theorem dj_snake_engagement_treats_value :
  total_value 2 4000 30000 = 158000 := by
  sorry

end NUMINAMATH_CALUDE_dj_snake_engagement_treats_value_l507_50701


namespace NUMINAMATH_CALUDE_work_completion_time_l507_50757

theorem work_completion_time 
  (x : ℝ) 
  (hx : x > 0) 
  (h_combined : 1/x + 1/8 = 3/16) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l507_50757


namespace NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l507_50743

theorem sphere_radius_when_area_equals_volume :
  ∀ R : ℝ,
  R > 0 →
  (4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) →
  R = 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l507_50743


namespace NUMINAMATH_CALUDE_intersecting_lines_sum_l507_50762

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  m : ℝ
  b : ℝ
  intersect_x : ℝ
  intersect_y : ℝ
  eq1 : intersect_y = 2 * m * intersect_x + 5
  eq2 : intersect_y = 4 * intersect_x + b

/-- The sum of b and m for two intersecting lines -/
def sum_b_m (lines : IntersectingLines) : ℝ :=
  lines.b + lines.m

/-- Theorem: For two lines y = 2mx + 5 and y = 4x + b intersecting at (4, 17), b + m = 2.5 -/
theorem intersecting_lines_sum (lines : IntersectingLines)
    (h1 : lines.intersect_x = 4)
    (h2 : lines.intersect_y = 17) :
    sum_b_m lines = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_intersecting_lines_sum_l507_50762


namespace NUMINAMATH_CALUDE_chocolates_not_in_box_l507_50707

theorem chocolates_not_in_box (initial_chocolates : ℕ) (initial_boxes : ℕ) 
  (additional_chocolates : ℕ) (additional_boxes : ℕ) :
  initial_chocolates = 50 →
  initial_boxes = 3 →
  additional_chocolates = 25 →
  additional_boxes = 2 →
  ∃ (chocolates_per_box : ℕ),
    chocolates_per_box * (initial_boxes + additional_boxes) = initial_chocolates + additional_chocolates →
    initial_chocolates - (chocolates_per_box * initial_boxes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_not_in_box_l507_50707


namespace NUMINAMATH_CALUDE_open_box_volume_formula_l507_50796

/-- Represents the volume of an open box constructed from a rectangular metal sheet. -/
def boxVolume (sheetLength sheetWidth x : ℝ) : ℝ :=
  (sheetLength - 2*x) * (sheetWidth - 2*x) * x

theorem open_box_volume_formula (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 10) : 
  boxVolume 30 20 x = 600*x - 100*x^2 + 4*x^3 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_formula_l507_50796


namespace NUMINAMATH_CALUDE_fraction_inequality_l507_50703

theorem fraction_inequality (a b : ℝ) :
  ((b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0)) → (1 / a < 1 / b) ∧
  (a > 0 ∧ 0 > b) → ¬(1 / a < 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l507_50703


namespace NUMINAMATH_CALUDE_xy_value_l507_50750

theorem xy_value (x y : ℝ) (h1 : x + y = 2) (h2 : x^2 * y^3 + y^2 * x^3 = 32) : x * y = 2^(5/3) := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l507_50750


namespace NUMINAMATH_CALUDE_function_identification_l507_50728

theorem function_identification (f : ℝ → ℝ) (b : ℝ) 
  (h1 : ∀ x, f (3 * x) = 3 * x^2 + b) 
  (h2 : f 1 = 0) : 
  ∀ x, f x = (1/3) * x^2 - (1/3) := by
sorry

end NUMINAMATH_CALUDE_function_identification_l507_50728


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l507_50782

theorem quadratic_equation_solution :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x^2 - 4*x + 9 = 25 ↔ (x = a ∨ x = b)) ∧
    a ≥ b ∧
    3*a + 2*b = 10 + 2*Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l507_50782


namespace NUMINAMATH_CALUDE_binomial_coefficient_formula_l507_50704

theorem binomial_coefficient_formula (n k : ℕ) (h : k ≤ n) :
  Nat.choose n k = n.factorial / ((n - k).factorial * k.factorial) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_formula_l507_50704


namespace NUMINAMATH_CALUDE_brown_paint_red_pigment_weight_l507_50780

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : Real
  red : Real
  yellow : Real

/-- Represents the weight of each paint in the mixture -/
structure MixtureWeights where
  maroon : Real
  green : Real

theorem brown_paint_red_pigment_weight
  (maroon : PaintMixture)
  (green : PaintMixture)
  (weights : MixtureWeights)
  (h_maroon_comp : maroon.blue = 0.5 ∧ maroon.red = 0.5 ∧ maroon.yellow = 0)
  (h_green_comp : green.blue = 0.3 ∧ green.red = 0 ∧ green.yellow = 0.7)
  (h_total_weight : weights.maroon + weights.green = 10)
  (h_brown_blue : weights.maroon * maroon.blue + weights.green * green.blue = 4) :
  weights.maroon * maroon.red = 2.5 := by
  sorry

#check brown_paint_red_pigment_weight

end NUMINAMATH_CALUDE_brown_paint_red_pigment_weight_l507_50780


namespace NUMINAMATH_CALUDE_line_equation_correct_l507_50771

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is parallel to a line --/
def vectorParallelToLine (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.y = l.b * v.x

/-- The main theorem --/
theorem line_equation_correct (l : Line2D) (p : Point2D) (v : Vector2D) : 
  l.a = 1 ∧ l.b = 2 ∧ l.c = -1 ∧
  p.x = 1 ∧ p.y = 0 ∧
  v.x = 2 ∧ v.y = -1 →
  pointOnLine l p ∧ vectorParallelToLine l v := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l507_50771


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l507_50712

/-- A circle passing through three given points -/
def circle_through_points (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2 + 4 * p.1 - 2 * p.2) = 0}

/-- The three given points -/
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (0, 2)

/-- Theorem stating that the defined circle passes through the given points -/
theorem circle_passes_through_points :
  A ∈ circle_through_points A B C ∧
  B ∈ circle_through_points A B C ∧
  C ∈ circle_through_points A B C :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l507_50712


namespace NUMINAMATH_CALUDE_solution_set1_solution_set2_l507_50775

-- Part 1
def system1 (x : ℝ) : Prop :=
  3 * x - (x - 2) ≥ 6 ∧ x + 1 > (4 * x - 1) / 3

theorem solution_set1 : 
  ∀ x : ℝ, system1 x ↔ 1 ≤ x ∧ x < 4 := by sorry

-- Part 2
def system2 (x : ℝ) : Prop :=
  2 * x + 1 > 0 ∧ x > 2 * x - 5

def is_positive_integer (x : ℝ) : Prop :=
  ∃ n : ℕ, x = n ∧ n > 0

theorem solution_set2 :
  {x : ℝ | system2 x ∧ is_positive_integer x} = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_solution_set1_solution_set2_l507_50775


namespace NUMINAMATH_CALUDE_greatest_divisor_of_exponential_sum_l507_50745

theorem greatest_divisor_of_exponential_sum :
  ∃ (x : ℕ), x > 0 ∧
  (∀ (y : ℕ), y > 0 → (7^y + 12*y - 1) % x = 0) ∧
  (∀ (z : ℕ), z > x → ∃ (w : ℕ), w > 0 ∧ (7^w + 12*w - 1) % z ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_exponential_sum_l507_50745


namespace NUMINAMATH_CALUDE_subset_perfect_square_sum_l507_50772

theorem subset_perfect_square_sum (n : ℕ) (hn : n ≥ 3) (S : Finset ℕ) 
  (hS : S ⊆ Finset.range n) (hSsize : S.card > n/2 + 1) : 
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ∃ (k : ℕ), (a*b)^2 + (b*c)^2 + (c*a)^2 = k^2 := by
sorry

end NUMINAMATH_CALUDE_subset_perfect_square_sum_l507_50772


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l507_50790

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  x^4 + 4*x^2 + 20*x + 1 = (x^2 - 2*x + 7) * q + r ∧
  r.degree < (x^2 - 2*x + 7).degree ∧
  r = 8*x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l507_50790


namespace NUMINAMATH_CALUDE_new_person_weight_l507_50767

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 6 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l507_50767


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l507_50724

theorem complex_number_quadrant (z : ℂ) (h : (1 - Complex.I) / z = 4 + 2 * Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z < 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l507_50724


namespace NUMINAMATH_CALUDE_composition_equation_solution_l507_50718

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 5 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 7 * x + 6
  ∃ x : ℝ, δ (φ x) = -4 ∧ x = -43/35 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l507_50718


namespace NUMINAMATH_CALUDE_ages_sum_l507_50752

theorem ages_sum (a b c : ℕ) : 
  a = 20 + 2 * (b + c) →
  a^2 = 1980 + 3 * (b + c)^2 →
  a + b + c = 68 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l507_50752


namespace NUMINAMATH_CALUDE_sector_central_angle_l507_50737

theorem sector_central_angle (arc_length radius : ℝ) (h1 : arc_length = 2 * Real.pi) (h2 : radius = 2) :
  arc_length / radius = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l507_50737


namespace NUMINAMATH_CALUDE_symmetric_function_inequality_l507_50749

/-- A function that is symmetric about x = 1 -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (2 - x)

/-- The derivative condition for x < 1 -/
def DerivativeCondition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, x < 1 → 2 * f x + (x - 1) * f' x < 0

theorem symmetric_function_inequality
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_symmetric : SymmetricAboutOne f)
  (h_derivative : DerivativeCondition f f') :
  {x : ℝ | (x + 1)^2 * f (x + 2) > f 2} = Set.Ioo (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_inequality_l507_50749


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l507_50716

def f (x : ℝ) := x^3 + 3*x^2 - 9*x + 1

theorem tangent_line_and_extrema :
  ∃ (y : ℝ → ℝ),
    (∀ x, y x = -9*x + 1) ∧
    (∀ x, x ∈ [-1, 2] → f x ≤ 12) ∧
    (∀ x, x ∈ [-1, 2] → f x ≥ -4) ∧
    (∃ x₁ ∈ [-1, 2], f x₁ = 12) ∧
    (∃ x₂ ∈ [-1, 2], f x₂ = -4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l507_50716


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l507_50776

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l507_50776


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l507_50773

theorem polar_to_rectangular (ρ : ℝ) (θ : ℝ) :
  ρ = 2 ∧ θ = π / 6 →
  ∃ x y : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ x = Real.sqrt 3 ∧ y = 1 := by
sorry


end NUMINAMATH_CALUDE_polar_to_rectangular_l507_50773


namespace NUMINAMATH_CALUDE_power_sixteen_div_sixteen_squared_l507_50778

theorem power_sixteen_div_sixteen_squared : 2^16 / 16^2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_div_sixteen_squared_l507_50778


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l507_50777

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l507_50777


namespace NUMINAMATH_CALUDE_negation_equivalence_l507_50787

theorem negation_equivalence : 
  (¬ ∃ (x : ℝ), x^2 - x + 1 ≤ 0) ↔ (∀ (x : ℝ), x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l507_50787


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l507_50733

/-- The area of a circle circumscribed around an isosceles triangle -/
theorem circumscribed_circle_area (base lateral : ℝ) (h_base : base = 24) (h_lateral : lateral = 13) :
  let height : ℝ := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let triangle_area : ℝ := (base * height) / 2
  let radius : ℝ := (base * lateral ^ 2) / (4 * triangle_area)
  let circle_area : ℝ := π * radius ^ 2
  circle_area = 285.61 * π :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l507_50733


namespace NUMINAMATH_CALUDE_square_difference_l507_50791

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l507_50791


namespace NUMINAMATH_CALUDE_x_y_z_order_l507_50764

-- Define the constants
noncomputable def x : ℝ := Real.exp (3⁻¹ * Real.log 3)
noncomputable def y : ℝ := Real.exp (6⁻¹ * Real.log 7)
noncomputable def z : ℝ := 7 ^ (1/7 : ℝ)

-- State the theorem
theorem x_y_z_order : z < y ∧ y < x := by
  sorry

end NUMINAMATH_CALUDE_x_y_z_order_l507_50764


namespace NUMINAMATH_CALUDE_typist_salary_l507_50736

theorem typist_salary (x : ℝ) : 
  (x * 1.1 * 0.95 = 1045) → x = 1000 := by sorry

end NUMINAMATH_CALUDE_typist_salary_l507_50736


namespace NUMINAMATH_CALUDE_determinant_value_l507_50783

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_value (m : ℝ) (h : m^2 - 2*m - 3 = 0) : 
  determinant (m^2) (m-3) (1-2*m) (m-2) = 9 := by sorry

end NUMINAMATH_CALUDE_determinant_value_l507_50783


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l507_50798

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (2*k - 6)*x + k - 3 > 0) ↔ (3 < k ∧ k < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l507_50798


namespace NUMINAMATH_CALUDE_exists_plane_parallel_to_skew_lines_l507_50799

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A plane is parallel to a line if the line's direction is perpendicular to the plane's normal -/
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- There exists a plane parallel to both skew lines -/
theorem exists_plane_parallel_to_skew_lines (a b : Line3D) (h : are_skew a b) :
  ∃ (α : Plane3D), plane_parallel_to_line α a ∧ plane_parallel_to_line α b := by
  sorry

end NUMINAMATH_CALUDE_exists_plane_parallel_to_skew_lines_l507_50799


namespace NUMINAMATH_CALUDE_prob_same_suit_is_one_seventeenth_l507_50731

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function that returns the suit of a card given its index in the deck -/
def cardSuit (card : Fin 52) : Suit :=
  sorry

/-- The probability of drawing two cards of the same suit from a standard deck -/
def probabilitySameSuit : ℚ :=
  1 / 17

/-- Theorem stating that the probability of drawing two cards of the same suit is 1/17 -/
theorem prob_same_suit_is_one_seventeenth :
  probabilitySameSuit = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_suit_is_one_seventeenth_l507_50731


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l507_50793

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l507_50793


namespace NUMINAMATH_CALUDE_calculate_upstream_speed_l507_50788

/-- The speed of a man rowing in a river -/
structure RowerSpeed where
  stillWater : ℝ
  downstream : ℝ
  upstream : ℝ

/-- Theorem: Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem calculate_upstream_speed (s : RowerSpeed) 
  (h1 : s.stillWater = 40)
  (h2 : s.downstream = 45) :
  s.upstream = 35 := by
  sorry

end NUMINAMATH_CALUDE_calculate_upstream_speed_l507_50788


namespace NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l507_50765

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x * (1 - 2*x) * (1 - 3*x) + y * (1 - 2*y) * (1 - 3*y) + z * (1 - 2*z) * (1 - 3*z) ≥ 0 :=
sorry

theorem equality_conditions (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x * (1 - 2*x) * (1 - 3*x) + y * (1 - 2*y) * (1 - 3*y) + z * (1 - 2*z) * (1 - 3*z) = 0 ↔
  ((x = 0 ∧ y = 1/2 ∧ z = 1/2) ∨
   (y = 0 ∧ z = 1/2 ∧ x = 1/2) ∨
   (z = 0 ∧ x = 1/2 ∧ y = 1/2) ∨
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3)) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l507_50765


namespace NUMINAMATH_CALUDE_spinner_sector_areas_l507_50753

/-- Represents a circular spinner with WIN and BONUS sectors -/
structure Spinner where
  radius : ℝ
  win_prob : ℝ
  bonus_prob : ℝ

/-- Calculates the area of a sector given its probability and the total area -/
def sector_area (prob : ℝ) (total_area : ℝ) : ℝ := prob * total_area

/-- Theorem stating the areas of WIN and BONUS sectors for a specific spinner -/
theorem spinner_sector_areas (s : Spinner) 
  (h_radius : s.radius = 15)
  (h_win_prob : s.win_prob = 1/3)
  (h_bonus_prob : s.bonus_prob = 1/4) :
  let total_area := π * s.radius^2
  sector_area s.win_prob total_area = 75 * π ∧ 
  sector_area s.bonus_prob total_area = 56.25 * π := by
  sorry


end NUMINAMATH_CALUDE_spinner_sector_areas_l507_50753


namespace NUMINAMATH_CALUDE_first_method_is_simple_random_second_method_is_systematic_l507_50742

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a student in the survey --/
structure Student where
  id : Nat
  deriving Repr

/-- Represents the survey setup --/
structure Survey where
  totalStudents : Nat
  selectedStudents : Nat
  selectionCriteria : Student → Bool

/-- Determines the sampling method based on the survey setup --/
def determineSamplingMethod (s : Survey) : SamplingMethod :=
  sorry

/-- The first survey method --/
def firstSurvey : Survey :=
  { totalStudents := 200
  , selectedStudents := 20
  , selectionCriteria := λ _ => true }

/-- The second survey method --/
def secondSurvey : Survey :=
  { totalStudents := 200
  , selectedStudents := 20
  , selectionCriteria := λ student => student.id % 10 = 2 }

/-- Theorem stating that the first method is simple random sampling --/
theorem first_method_is_simple_random :
  determineSamplingMethod firstSurvey = SamplingMethod.SimpleRandom :=
  sorry

/-- Theorem stating that the second method is systematic sampling --/
theorem second_method_is_systematic :
  determineSamplingMethod secondSurvey = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_first_method_is_simple_random_second_method_is_systematic_l507_50742


namespace NUMINAMATH_CALUDE_sine_derivative_2007_l507_50797

open Real

theorem sine_derivative_2007 (f : ℝ → ℝ) (x : ℝ) :
  f = sin →
  (∀ n : ℕ, deriv^[n] f = fun x ↦ f (x + n * (π / 2))) →
  deriv^[2007] f = fun x ↦ -cos x := by
  sorry

end NUMINAMATH_CALUDE_sine_derivative_2007_l507_50797


namespace NUMINAMATH_CALUDE_number_division_problem_l507_50754

theorem number_division_problem (N : ℝ) (x : ℝ) : 
  ((N - 34) / 10 = 2) → ((N - 5) / x = 7) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l507_50754


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l507_50740

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 58 = (X - 3) * q + 19 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l507_50740


namespace NUMINAMATH_CALUDE_unique_modular_solution_l507_50710

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1453 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l507_50710


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l507_50758

theorem rectangular_solid_volume (x y z : ℝ) 
  (h1 : x * y = 15)  -- Area of side face
  (h2 : y * z = 10)  -- Area of front face
  (h3 : x * z = 6)   -- Area of bottom face
  : x * y * z = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l507_50758


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l507_50781

theorem subtraction_of_fractions : (8 : ℚ) / 15 - (11 : ℚ) / 20 = -1 / 60 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l507_50781


namespace NUMINAMATH_CALUDE_yard_length_with_26_trees_32m_apart_l507_50795

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1 : ℝ) * distanceBetweenTrees

theorem yard_length_with_26_trees_32m_apart :
  yardLength 26 32 = 800 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_26_trees_32m_apart_l507_50795


namespace NUMINAMATH_CALUDE_total_stickers_is_182_l507_50755

/-- Represents a folder with its properties -/
structure Folder :=
  (color : String)
  (sheets : Nat)
  (starsPerSheet : Nat)
  (circlesPerSheet : Nat)
  (sharedCircles : Nat := 0)

/-- Calculates the total number of stickers in a folder -/
def totalStickersInFolder (f : Folder) : Nat :=
  f.sheets * (f.starsPerSheet + f.circlesPerSheet) + f.sharedCircles

/-- The list of folders with their properties -/
def folders : List Folder := [
  { color := "red", sheets := 10, starsPerSheet := 3, circlesPerSheet := 4 },
  { color := "green", sheets := 8, starsPerSheet := 5, circlesPerSheet := 1 },
  { color := "blue", sheets := 6, starsPerSheet := 2, circlesPerSheet := 1 },
  { color := "yellow", sheets := 4, starsPerSheet := 4, circlesPerSheet := 4 },
  { color := "purple", sheets := 2, starsPerSheet := 6, circlesPerSheet := 0, sharedCircles := 2 }
]

/-- The main theorem stating that the total number of stickers is 182 -/
theorem total_stickers_is_182 : 
  (folders.map totalStickersInFolder).sum = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_is_182_l507_50755


namespace NUMINAMATH_CALUDE_one_third_of_recipe_flour_l507_50744

-- Define the original amount of flour in the recipe
def original_flour : ℚ := 16/3

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/3

-- Define the result we want to prove
def result : ℚ := 16/9

-- Theorem statement
theorem one_third_of_recipe_flour :
  recipe_fraction * original_flour = result := by sorry

end NUMINAMATH_CALUDE_one_third_of_recipe_flour_l507_50744


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l507_50746

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem cube_of_square_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 2 ^ 3 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l507_50746


namespace NUMINAMATH_CALUDE_minimum_score_needed_l507_50711

def current_scores : List ℕ := [90, 80, 70, 60, 85]
def num_current_tests : ℕ := current_scores.length
def sum_current_scores : ℕ := current_scores.sum
def current_average : ℚ := sum_current_scores / num_current_tests
def target_increase : ℚ := 3
def num_total_tests : ℕ := num_current_tests + 1

theorem minimum_score_needed (min_score : ℕ) : 
  (sum_current_scores + min_score) / num_total_tests ≥ current_average + target_increase ∧
  ∀ (score : ℕ), score < min_score → 
    (sum_current_scores + score) / num_total_tests < current_average + target_increase →
  min_score = 95 := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_needed_l507_50711


namespace NUMINAMATH_CALUDE_card_digits_problem_l507_50729

theorem card_digits_problem (a b c : ℕ) : 
  0 < a → a < b → b < c → c < 10 →
  (999 * c + 90 * b - 990 * a) + 
  (100 * c + 9 * b - 99 * a) + 
  (10 * c + b - 10 * a) + 
  (c - a) = 9090 →
  a = 1 ∧ b = 2 ∧ c = 9 := by
sorry

end NUMINAMATH_CALUDE_card_digits_problem_l507_50729


namespace NUMINAMATH_CALUDE_polynomial_factorization_l507_50779

theorem polynomial_factorization (x : ℝ) :
  x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l507_50779


namespace NUMINAMATH_CALUDE_product_and_power_constraint_l507_50719

theorem product_and_power_constraint (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1)
  (h_product : a * b * c = 10)
  (h_power : a^(Real.log a) * b^(Real.log b) * c^(Real.log c) ≥ 10) :
  (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) :=
by sorry

end NUMINAMATH_CALUDE_product_and_power_constraint_l507_50719


namespace NUMINAMATH_CALUDE_bag_original_price_l507_50726

/-- Given a bag sold for $120 after a 20% discount, prove its original price was $150 -/
theorem bag_original_price (discounted_price : ℝ) (discount_rate : ℝ) : 
  discounted_price = 120 → 
  discount_rate = 0.2 → 
  discounted_price = (1 - discount_rate) * 150 := by
sorry

end NUMINAMATH_CALUDE_bag_original_price_l507_50726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l507_50751

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a4 : a 4 = 9)
  (h_a8 : a 8 = -(a 9)) :
  a 1 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l507_50751


namespace NUMINAMATH_CALUDE_tracksuit_discount_problem_l507_50727

/-- Given a tracksuit with an original price, if it is discounted by 20% and the discount amount is 30 yuan, then the actual amount spent is 120 yuan. -/
theorem tracksuit_discount_problem (original_price : ℝ) : 
  (original_price - original_price * 0.8 = 30) → 
  (original_price * 0.8 = 120) := by
sorry

end NUMINAMATH_CALUDE_tracksuit_discount_problem_l507_50727


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l507_50739

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l507_50739


namespace NUMINAMATH_CALUDE_decimal_255_to_octal_l507_50735

-- Define a function to convert decimal to octal
def decimal_to_octal (n : ℕ) : List ℕ :=
  sorry

-- Theorem statement
theorem decimal_255_to_octal :
  decimal_to_octal 255 = [3, 7, 7] := by
  sorry

end NUMINAMATH_CALUDE_decimal_255_to_octal_l507_50735


namespace NUMINAMATH_CALUDE_not_all_face_sums_different_not_all_face_sums_different_b_l507_50734

/-- Represents the possible values that can be assigned to a vertex of the cube -/
inductive VertexValue
  | Zero
  | One

/-- Represents a cube with values assigned to its vertices -/
structure Cube :=
  (vertices : Fin 8 → VertexValue)

/-- Calculates the sum of values on a face of the cube -/
def faceSum (c : Cube) (face : Fin 6) : Nat :=
  sorry

/-- Theorem stating that it's impossible for all face sums to be different -/
theorem not_all_face_sums_different (c : Cube) : 
  ¬(∀ (i j : Fin 6), i ≠ j → faceSum c i ≠ faceSum c j) :=
sorry

/-- Represents the possible values that can be assigned to a vertex of the cube (for part b) -/
inductive VertexValueB
  | NegOne
  | PosOne

/-- Represents a cube with values assigned to its vertices (for part b) -/
structure CubeB :=
  (vertices : Fin 8 → VertexValueB)

/-- Calculates the sum of values on a face of the cube (for part b) -/
def faceSumB (c : CubeB) (face : Fin 6) : Int :=
  sorry

/-- Theorem stating that it's impossible for all face sums to be different (for part b) -/
theorem not_all_face_sums_different_b (c : CubeB) : 
  ¬(∀ (i j : Fin 6), i ≠ j → faceSumB c i ≠ faceSumB c j) :=
sorry

end NUMINAMATH_CALUDE_not_all_face_sums_different_not_all_face_sums_different_b_l507_50734


namespace NUMINAMATH_CALUDE_sequence_problem_l507_50763

/-- Given a sequence {aₙ}, prove that a₁₉ = 1/16 under specific conditions -/
theorem sequence_problem (a : ℕ → ℚ) : 
  (a 4 = 1) → 
  (a 6 = 1/3) → 
  (∃ d : ℚ, ∀ n : ℕ, 1/(a (n+1)) - 1/(a n) = d) → 
  (a 19 = 1/16) := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l507_50763


namespace NUMINAMATH_CALUDE_toms_floor_replacement_cost_l507_50715

/-- The total cost to replace a floor given the room dimensions, removal cost, and new flooring cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_toms_floor_replacement_cost_l507_50715


namespace NUMINAMATH_CALUDE_prob_rolling_doubles_l507_50760

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice being rolled -/
def numDice : ℕ := 3

/-- The number of favorable outcomes (rolling the same number on all dice) -/
def favorableOutcomes : ℕ := numSides

/-- The total number of possible outcomes when rolling the dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The probability of rolling doubles with three six-sided dice -/
theorem prob_rolling_doubles : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_rolling_doubles_l507_50760


namespace NUMINAMATH_CALUDE_binomial_15_choose_3_l507_50784

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_choose_3_l507_50784


namespace NUMINAMATH_CALUDE_number_puzzle_l507_50768

theorem number_puzzle : ∃! x : ℚ, x / 5 + 6 = x / 4 - 6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l507_50768


namespace NUMINAMATH_CALUDE_multiplication_difference_l507_50738

theorem multiplication_difference : 
  let correct_number : ℕ := 134
  let correct_multiplier : ℕ := 43
  let incorrect_multiplier : ℕ := 34
  (correct_number * correct_multiplier) - (correct_number * incorrect_multiplier) = 1206 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_difference_l507_50738


namespace NUMINAMATH_CALUDE_transaction_result_l507_50717

theorem transaction_result (car_sale_price motorcycle_sale_price : ℝ)
  (car_loss_percent motorcycle_gain_percent : ℝ)
  (h1 : car_sale_price = 18000)
  (h2 : motorcycle_sale_price = 10000)
  (h3 : car_loss_percent = 10)
  (h4 : motorcycle_gain_percent = 25) :
  car_sale_price + motorcycle_sale_price =
  (car_sale_price / (100 - car_loss_percent) * 100) +
  (motorcycle_sale_price / (100 + motorcycle_gain_percent) * 100) :=
by sorry

end NUMINAMATH_CALUDE_transaction_result_l507_50717


namespace NUMINAMATH_CALUDE_abc_product_l507_50759

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 160) (h2 : b * (c + a) = 168) (h3 : c * (a + b) = 180) :
  a * b * c = 772 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l507_50759


namespace NUMINAMATH_CALUDE_cylinder_volume_l507_50756

theorem cylinder_volume (r h : ℝ) (lateral_area : ℝ) : 
  r = 3 → 
  lateral_area = 12 * Real.pi → 
  lateral_area = 2 * Real.pi * r * h →
  r * r * h * Real.pi = 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l507_50756


namespace NUMINAMATH_CALUDE_qinghai_lake_travel_solution_l507_50720

/-- Represents the travel plans and costs for two teams visiting Qinghai Lake. -/
structure TravelPlan where
  distanceA : ℕ  -- Distance for Team A in km
  distanceB : ℕ  -- Distance for Team B in km
  daysA : ℕ      -- Number of days for Team A
  daysB : ℕ      -- Number of days for Team B
  costA : ℕ      -- Daily cost per person for Team A in yuan
  costB : ℕ      -- Daily cost per person for Team B in yuan
  peopleA : ℕ    -- Number of people in Team A
  peopleB : ℕ    -- Number of people in Team B
  m : ℕ          -- Additional people joining Team A

/-- The theorem stating the solution to the Qinghai Lake travel problem. -/
theorem qinghai_lake_travel_solution (plan : TravelPlan) : 
  plan.distanceA = 2700 ∧ 
  plan.distanceB = 1800 ∧
  plan.distanceA / plan.daysA = 2 * (plan.distanceB / plan.daysB) ∧
  plan.daysA + 1 = plan.daysB ∧
  plan.costA = 200 ∧
  plan.costB = 150 ∧
  plan.peopleA = 10 ∧
  plan.peopleB = 8 ∧
  (plan.costA - 30) * (plan.peopleA + plan.m) * plan.daysA + plan.costB * plan.peopleB * plan.daysB = 
    (plan.costA * plan.peopleA * plan.daysA + plan.costB * plan.peopleB * plan.daysB) * 120 / 100 →
  plan.daysA = 3 ∧ plan.daysB = 4 ∧ plan.m = 6 := by
  sorry

end NUMINAMATH_CALUDE_qinghai_lake_travel_solution_l507_50720


namespace NUMINAMATH_CALUDE_amusement_park_probabilities_l507_50700

/-- Amusement park problem -/
theorem amusement_park_probabilities
  (p_A1 : ℝ)
  (p_B1 : ℝ)
  (p_A2_given_A1 : ℝ)
  (p_A2_given_B1 : ℝ)
  (h1 : p_A1 = 0.3)
  (h2 : p_B1 = 0.7)
  (h3 : p_A2_given_A1 = 0.7)
  (h4 : p_A2_given_B1 = 0.6)
  (h5 : p_A1 + p_B1 = 1) :
  let p_A2 := p_A1 * p_A2_given_A1 + p_B1 * p_A2_given_B1
  let p_B1_given_A2 := (p_B1 * p_A2_given_B1) / p_A2
  ∃ (ε : ℝ), abs (p_A2 - 0.63) < ε ∧ abs (p_B1_given_A2 - (2/3)) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_amusement_park_probabilities_l507_50700


namespace NUMINAMATH_CALUDE_unique_solution_cube_system_l507_50730

theorem unique_solution_cube_system :
  ∃! (x y z : ℝ), y^2 = 4*x^3 + x - 4 ∧
                   z^2 = 4*y^3 + y - 4 ∧
                   x^2 = 4*z^3 + z - 4 ∧
                   x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_system_l507_50730


namespace NUMINAMATH_CALUDE_parabola_focus_l507_50792

/-- Represents a parabola with equation y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun y x => y^2 = 8*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (2, 0)

/-- Theorem stating that the focus of the parabola y^2 = 8x is (2, 0) -/
theorem parabola_focus (p : Parabola) : focus p = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l507_50792


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l507_50714

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7cm, one side is 3cm
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 17 :=  -- Perimeter is 17cm
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l507_50714


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l507_50766

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 30) :
  Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2)) = 2 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l507_50766


namespace NUMINAMATH_CALUDE_alyssa_cherries_cost_l507_50747

/-- The amount Alyssa paid for cherries -/
def cherries_cost (total_spent grapes_cost : ℚ) : ℚ :=
  total_spent - grapes_cost

/-- Proof that Alyssa paid $9.85 for cherries -/
theorem alyssa_cherries_cost :
  let total_spent : ℚ := 21.93
  let grapes_cost : ℚ := 12.08
  cherries_cost total_spent grapes_cost = 9.85 := by
  sorry

#eval cherries_cost 21.93 12.08

end NUMINAMATH_CALUDE_alyssa_cherries_cost_l507_50747


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l507_50722

theorem complex_sum_theorem : 
  let Z₁ : ℂ := (1 - Complex.I) / (1 + Complex.I)
  let Z₂ : ℂ := (3 - Complex.I) * Complex.I
  Z₁ + Z₂ = 1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l507_50722


namespace NUMINAMATH_CALUDE_divisibility_of_fourth_power_sum_l507_50789

theorem divisibility_of_fourth_power_sum (a b c n : ℤ) 
  (h1 : n ∣ (a + b + c)) 
  (h2 : n ∣ (a^2 + b^2 + c^2)) : 
  n ∣ (a^4 + b^4 + c^4) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_fourth_power_sum_l507_50789


namespace NUMINAMATH_CALUDE_f_composition_value_l507_50785

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem f_composition_value : f (f 2) = 78652 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l507_50785


namespace NUMINAMATH_CALUDE_least_2310_divisors_form_l507_50769

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is not divisible by 10 -/
def not_div_by_ten (m : ℕ) : Prop := ¬(10 ∣ m)

/-- The least positive integer with exactly 2310 distinct positive divisors -/
def least_with_2310_divisors : ℕ := sorry

theorem least_2310_divisors_form :
  ∃ (m k : ℕ), 
    least_with_2310_divisors = m * 10^k ∧ 
    not_div_by_ten m ∧ 
    m + k = 10 := by sorry

end NUMINAMATH_CALUDE_least_2310_divisors_form_l507_50769


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_l507_50748

-- Define a monic polynomial of degree 2
def monicQuadratic (b c : ℝ) : ℝ → ℝ := λ x => x^2 + b*x + c

-- State the theorem
theorem unique_monic_quadratic (g : ℝ → ℝ) :
  (∃ b c : ℝ, ∀ x, g x = monicQuadratic b c x) →  -- g is a monic quadratic polynomial
  g 0 = 6 →                                       -- g(0) = 6
  g 1 = 8 →                                       -- g(1) = 8
  ∀ x, g x = x^2 + x + 6 :=                       -- Conclusion: g(x) = x^2 + x + 6
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_unique_monic_quadratic_l507_50748


namespace NUMINAMATH_CALUDE_x_is_irrational_l507_50709

/-- Representation of the digits of 1987^k -/
def digits (k : ℕ) : List ℕ :=
  sorry

/-- Construct the number x as described in the problem -/
def x : ℝ :=
  sorry

/-- Theorem stating that x is irrational -/
theorem x_is_irrational : Irrational x := by
  sorry

end NUMINAMATH_CALUDE_x_is_irrational_l507_50709


namespace NUMINAMATH_CALUDE_binomial_divisibility_l507_50770

theorem binomial_divisibility (k : ℕ) (h : k ≥ 2) :
  ∃ m : ℤ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = m * 2^(3*k) ∧
  ¬∃ n : ℤ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = n * 2^(3*k+1) :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l507_50770


namespace NUMINAMATH_CALUDE_percentage_increase_l507_50706

theorem percentage_increase (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 60)
  (h2 : new_earnings = 78) : 
  (new_earnings - original_earnings) / original_earnings * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l507_50706


namespace NUMINAMATH_CALUDE_terms_are_like_l507_50774

-- Define a structure for algebraic terms
structure AlgebraicTerm where
  coefficient : ℤ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define a function to check if two terms are like terms
def are_like_terms (t1 t2 : AlgebraicTerm) : Prop :=
  t1.x_exponent = t2.x_exponent ∧ t1.y_exponent = t2.y_exponent

-- Define the two terms we want to compare
def term1 : AlgebraicTerm := { coefficient := -4, x_exponent := 1, y_exponent := 2 }
def term2 : AlgebraicTerm := { coefficient := 4, x_exponent := 1, y_exponent := 2 }

-- State the theorem
theorem terms_are_like : are_like_terms term1 term2 := by
  sorry

end NUMINAMATH_CALUDE_terms_are_like_l507_50774


namespace NUMINAMATH_CALUDE_cookie_recipe_total_cups_l507_50794

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)

/-- Calculates the total cups of ingredients given a ratio and amount of sugar -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem: Given the specified ratio and sugar amount, the total cups is 40 -/
theorem cookie_recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 5, 3⟩
  totalCups ratio 12 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cookie_recipe_total_cups_l507_50794


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l507_50732

theorem tan_sum_simplification : 
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l507_50732


namespace NUMINAMATH_CALUDE_contractor_payment_proof_l507_50723

/-- Calculates the total amount received by a contractor given the contract details and attendance. -/
def contractorPayment (totalDays duration : ℕ) (dailyWage dailyFine : ℚ) (absentDays : ℕ) : ℚ :=
  let workingDays := duration - absentDays
  let earnings := workingDays * dailyWage
  let fines := absentDays * dailyFine
  earnings - fines

/-- Proves that the contractor receives Rs. 490 under the given conditions. -/
theorem contractor_payment_proof :
  contractorPayment 30 30 25 (7.5 : ℚ) 8 = 490 := by
  sorry

#eval contractorPayment 30 30 25 (7.5 : ℚ) 8

end NUMINAMATH_CALUDE_contractor_payment_proof_l507_50723


namespace NUMINAMATH_CALUDE_negation_of_implication_l507_50741

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l507_50741


namespace NUMINAMATH_CALUDE_product_equals_32_l507_50702

theorem product_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l507_50702


namespace NUMINAMATH_CALUDE_car_rental_problem_l507_50713

/-- Represents a car rental company -/
structure Company where
  totalCars : Nat
  baseRent : ℝ
  rentIncrease : ℝ
  maintenanceFee : ℝ → ℝ

/-- Calculates the profit for a company given the number of cars rented -/
def profit (c : Company) (rented : ℝ) : ℝ :=
  (c.baseRent + (c.totalCars - rented) * c.rentIncrease) * rented - c.maintenanceFee rented

/-- Company A as described in the problem -/
def companyA : Company :=
  { totalCars := 50
  , baseRent := 3000
  , rentIncrease := 50
  , maintenanceFee := λ x => 200 * x }

/-- Company B as described in the problem -/
def companyB : Company :=
  { totalCars := 50
  , baseRent := 3500
  , rentIncrease := 0
  , maintenanceFee := λ _ => 1850 }

theorem car_rental_problem :
  (profit companyA 10 = 48000) ∧
  (∃ x : ℝ, x = 37 ∧ profit companyA x = profit companyB x) ∧
  (∃ max : ℝ, max = 33150 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ 50 → |profit companyA x - profit companyB x| ≤ max) ∧
  (∀ a : ℝ, 50 < a ∧ a < 150 ↔
    (let f := λ x => profit companyA x - a * x - profit companyB x
     ∃ max : ℝ, max = f 17 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 50 → f x ≤ max ∧ f x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_car_rental_problem_l507_50713


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l507_50786

theorem simplify_cube_roots : 
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 28 ^ (1/3) * 4 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l507_50786
