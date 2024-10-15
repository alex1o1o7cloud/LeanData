import Mathlib

namespace NUMINAMATH_CALUDE_complex_multiplication_l1118_111807

theorem complex_multiplication (i : ℂ) : i * i = -1 → (3 + i) * (1 - 2*i) = 5 - 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1118_111807


namespace NUMINAMATH_CALUDE_bake_sale_solution_l1118_111812

def bake_sale_problem (brownie_price : ℚ) (brownie_count : ℕ) 
                      (lemon_square_price : ℚ) (lemon_square_count : ℕ)
                      (total_goal : ℚ) (cookie_count : ℕ) : Prop :=
  let current_total : ℚ := brownie_price * brownie_count + lemon_square_price * lemon_square_count
  let remaining_goal : ℚ := total_goal - current_total
  let cookie_price : ℚ := remaining_goal / cookie_count
  cookie_price = 4

theorem bake_sale_solution :
  bake_sale_problem 3 4 2 5 50 7 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_solution_l1118_111812


namespace NUMINAMATH_CALUDE_expression_value_l1118_111837

theorem expression_value (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1118_111837


namespace NUMINAMATH_CALUDE_turtle_count_relationship_lonely_island_turtle_count_l1118_111800

/-- The number of turtles on Happy Island -/
def happy_turtles : ℕ := 60

/-- The number of turtles on Lonely Island -/
def lonely_turtles : ℕ := 25

/-- Theorem stating the relationship between turtles on Happy and Lonely Islands -/
theorem turtle_count_relationship : happy_turtles = 2 * lonely_turtles + 10 := by
  sorry

/-- Theorem proving the number of turtles on Lonely Island -/
theorem lonely_island_turtle_count : lonely_turtles = 25 := by
  sorry

end NUMINAMATH_CALUDE_turtle_count_relationship_lonely_island_turtle_count_l1118_111800


namespace NUMINAMATH_CALUDE_family_eating_habits_l1118_111876

theorem family_eating_habits (only_veg only_nonveg total_veg : ℕ) 
  (h1 : only_veg = 16)
  (h2 : only_nonveg = 9)
  (h3 : total_veg = 28) :
  total_veg - only_veg = 12 := by
  sorry

end NUMINAMATH_CALUDE_family_eating_habits_l1118_111876


namespace NUMINAMATH_CALUDE_no_odd_sided_cross_section_polyhedron_l1118_111824

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields here

/-- A polygon -/
structure Polygon where
  sides : ℕ

/-- Represents a cross-section of a polyhedron with a plane -/
def cross_section (p : ConvexPolyhedron) (plane : Plane) : Polygon :=
  sorry

/-- Predicate to check if a plane passes through a vertex of the polyhedron -/
def passes_through_vertex (p : ConvexPolyhedron) (plane : Plane) : Prop :=
  sorry

/-- Main theorem: No such convex polyhedron exists -/
theorem no_odd_sided_cross_section_polyhedron :
  ¬ ∃ (p : ConvexPolyhedron),
    (∀ (plane : Plane),
      ¬passes_through_vertex p plane →
      (cross_section p plane).sides % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_no_odd_sided_cross_section_polyhedron_l1118_111824


namespace NUMINAMATH_CALUDE_expression_evaluation_l1118_111893

theorem expression_evaluation : 
  let a := 12
  let b := 14
  let c := 18
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = 44 := by
sorry


end NUMINAMATH_CALUDE_expression_evaluation_l1118_111893


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_l1118_111821

theorem angle_sum_in_triangle (A B C : ℝ) : 
  -- Triangle ABC exists
  -- Sum of angles A and B is 80°
  (A + B = 80) →
  -- Sum of all angles in a triangle is 180°
  (A + B + C = 180) →
  -- Angle C measures 100°
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_l1118_111821


namespace NUMINAMATH_CALUDE_power_function_property_l1118_111886

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

theorem power_function_property (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 → f 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l1118_111886


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_over_4_l1118_111839

theorem cos_alpha_plus_5pi_over_4 (α : ℝ) (h : Real.sin (α - π/4) = 1/3) :
  Real.cos (α + 5*π/4) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_over_4_l1118_111839


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1118_111851

theorem trigonometric_identity (x : ℝ) :
  Real.cos (4 * x) * Real.cos (π + 2 * x) - Real.sin (2 * x) * Real.cos (π / 2 - 4 * x) = 
  Real.sqrt 2 / 2 * Real.sin (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1118_111851


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1118_111816

/-- Proves that given a person walking at 3/5 of their usual speed and missing the bus by 5 minutes, their usual time to catch the bus is 7.5 minutes. -/
theorem usual_time_to_catch_bus : ∀ (usual_speed : ℝ) (usual_time : ℝ),
  usual_time > 0 →
  usual_speed > 0 →
  (3/5 * usual_speed) * (usual_time + 5) = usual_speed * usual_time →
  usual_time = 7.5 := by
sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1118_111816


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1118_111803

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p ^ 2 - 5 * p - 14 = 0) →
  (3 * q ^ 2 - 5 * q - 14 = 0) →
  p ≠ q →
  (3 * p ^ 2 - 3 * q ^ 2) / (p - q) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1118_111803


namespace NUMINAMATH_CALUDE_part_one_part_two_l1118_111841

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 1|

-- Part 1
theorem part_one : 
  (∀ x : ℝ, f 2 x < 5 ↔ x ∈ Set.Ioo (-2) 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4 - |a - 1|) ↔ 
    a ∈ Set.Iic (-2) ∪ Set.Ici 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1118_111841


namespace NUMINAMATH_CALUDE_probability_of_two_red_balls_l1118_111894

def total_balls : ℕ := 7 + 5 + 4

def red_balls : ℕ := 7

def balls_picked : ℕ := 2

def probability_both_red : ℚ := 175 / 1000

theorem probability_of_two_red_balls :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked : ℚ) = probability_both_red :=
by sorry

end NUMINAMATH_CALUDE_probability_of_two_red_balls_l1118_111894


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1118_111868

def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B : Set ℝ := {-1, 2, 3, 6}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1118_111868


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l1118_111882

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, y)
  collinear a b → y = -6 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l1118_111882


namespace NUMINAMATH_CALUDE_least_k_inequality_l1118_111818

theorem least_k_inequality (a b c : ℝ) : 
  ∃ (k : ℝ), k = 8 ∧ (∀ (x : ℝ), x ≥ k → 
    (2*a/(a-b))^2 + (2*b/(b-c))^2 + (2*c/(c-a))^2 + x ≥ 
    4*((2*a/(a-b)) + (2*b/(b-c)) + (2*c/(c-a)))) ∧
  (∀ (y : ℝ), y < k → 
    ∃ (a' b' c' : ℝ), (2*a'/(a'-b'))^2 + (2*b'/(b'-c'))^2 + (2*c'/(c'-a'))^2 + y < 
    4*((2*a'/(a'-b')) + (2*b'/(b'-c')) + (2*c'/(c'-a')))) :=
sorry

end NUMINAMATH_CALUDE_least_k_inequality_l1118_111818


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1118_111896

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1118_111896


namespace NUMINAMATH_CALUDE_expression_evaluation_l1118_111867

theorem expression_evaluation (m n : ℤ) (hm : m = -1) (hn : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1118_111867


namespace NUMINAMATH_CALUDE_lead_percentage_in_mixture_l1118_111899

/-- Proves that the percentage of lead in a mixture is 25% given the specified conditions -/
theorem lead_percentage_in_mixture
  (cobalt_percent : Real)
  (copper_percent : Real)
  (lead_weight : Real)
  (copper_weight : Real)
  (h1 : cobalt_percent = 0.15)
  (h2 : copper_percent = 0.60)
  (h3 : lead_weight = 5)
  (h4 : copper_weight = 12)
  : (lead_weight / (copper_weight / copper_percent)) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_lead_percentage_in_mixture_l1118_111899


namespace NUMINAMATH_CALUDE_simplify_expression_l1118_111897

theorem simplify_expression : (8 * (10 ^ 12)) / (4 * (10 ^ 4)) = 200000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1118_111897


namespace NUMINAMATH_CALUDE_parabola_chord_intersection_l1118_111883

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a parabola in the form y^2 = 2px -/
structure Parabola where
  p : ℝ

def Parabola.contains (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * p.p * pt.x

def Line.contains (l : Line) (pt : Point) : Prop :=
  pt.y = l.m * pt.x + l.b

def perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

theorem parabola_chord_intersection (p : Parabola) (m : Point) (d e : Point) :
  p.p = 2 →
  p.contains m →
  m.y = 4 →
  ∃ (l_md l_me l_de : Line),
    l_md.contains m ∧ l_md.contains d ∧
    l_me.contains m ∧ l_me.contains e ∧
    l_de.contains d ∧ l_de.contains e ∧
    perpendicular l_md l_me →
    l_de.contains (Point.mk 8 (-4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_intersection_l1118_111883


namespace NUMINAMATH_CALUDE_sum_of_valid_a_values_l1118_111808

theorem sum_of_valid_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∃ x : ℝ, x + 1 > (x - 1) / 3 ∧ x + a < 3) ∧ 
    (∃ y : ℤ, y > 0 ∧ y ≠ 2 ∧ (y - a) / (y - 2) + 1 = 1 / (y - 2))) ∧
  (∀ a : ℤ, 
    ((∃ x : ℝ, x + 1 > (x - 1) / 3 ∧ x + a < 3) ∧ 
     (∃ y : ℤ, y > 0 ∧ y ≠ 2 ∧ (y - a) / (y - 2) + 1 = 1 / (y - 2))) → 
    a ∈ S) ∧
  S.sum id = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_values_l1118_111808


namespace NUMINAMATH_CALUDE_cut_cube_edges_l1118_111820

/-- A cube with one corner cut off, creating a new triangular face -/
structure CutCube where
  /-- The number of edges in the original cube -/
  original_edges : ℕ
  /-- The number of new edges created by the cut -/
  new_edges : ℕ
  /-- The cut creates a triangular face -/
  triangular_face : Bool

/-- The total number of edges in the cut cube -/
def CutCube.total_edges (c : CutCube) : ℕ := c.original_edges + c.new_edges

/-- Theorem stating that a cube with one corner cut off has 15 edges -/
theorem cut_cube_edges :
  ∀ (c : CutCube),
  c.original_edges = 12 ∧ c.new_edges = 3 ∧ c.triangular_face = true →
  c.total_edges = 15 := by
  sorry

end NUMINAMATH_CALUDE_cut_cube_edges_l1118_111820


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1118_111861

theorem sum_of_five_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1118_111861


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l1118_111860

theorem fraction_addition_simplification :
  5 / 462 + 23 / 42 = 43 / 77 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l1118_111860


namespace NUMINAMATH_CALUDE_eight_lines_theorem_l1118_111809

/-- Represents a collection of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_concurrent : Bool

/-- Calculates the number of regions formed by a given line configuration -/
def num_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem: Eight non-parallel, non-concurrent lines divide a plane into 37 regions -/
theorem eight_lines_theorem (config : LineConfiguration) :
  config.num_lines = 8 ∧ config.no_parallel ∧ config.no_concurrent →
  num_regions config = 37 :=
by sorry

end NUMINAMATH_CALUDE_eight_lines_theorem_l1118_111809


namespace NUMINAMATH_CALUDE_inverse_function_equality_l1118_111842

/-- Given a function f(x) = 2 / (ax + b) where a and b are nonzero constants,
    prove that if f^(-1)(x) = 2, then b = 1 - 2a. -/
theorem inverse_function_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f := fun x : ℝ => 2 / (a * x + b)
  (∃ x, f x = 2) → b = 1 - 2 * a :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_equality_l1118_111842


namespace NUMINAMATH_CALUDE_divisibility_of_sum_l1118_111855

theorem divisibility_of_sum (a b c d x : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (x^2 - (a+b)*x + a*b) * (x^2 - (c+d)*x + c*d) = 9 →
  ∃ k : ℤ, a + b + c + d = 4 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_l1118_111855


namespace NUMINAMATH_CALUDE_sarah_savings_l1118_111891

def savings_schedule : List ℕ := [5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20]

theorem sarah_savings : (savings_schedule.sum = 140) := by
  sorry

end NUMINAMATH_CALUDE_sarah_savings_l1118_111891


namespace NUMINAMATH_CALUDE_smallest_angle_is_27_l1118_111844

/-- Represents the properties of a circle divided into sectors --/
structure CircleSectors where
  num_sectors : ℕ
  angle_sum : ℕ
  is_arithmetic_sequence : Bool
  all_angles_integer : Bool

/-- Finds the smallest possible sector angle given the circle properties --/
def smallest_sector_angle (circle : CircleSectors) : ℕ :=
  sorry

/-- Theorem stating that for a circle divided into 10 sectors with the given properties,
    the smallest possible sector angle is 27 degrees --/
theorem smallest_angle_is_27 :
  ∀ (circle : CircleSectors),
    circle.num_sectors = 10 ∧
    circle.angle_sum = 360 ∧
    circle.is_arithmetic_sequence = true ∧
    circle.all_angles_integer = true →
    smallest_sector_angle circle = 27 :=
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_27_l1118_111844


namespace NUMINAMATH_CALUDE_x_minus_y_equals_nine_l1118_111834

theorem x_minus_y_equals_nine
  (x y : ℕ)
  (h1 : 3^x * 4^y = 19683)
  (h2 : x = 9) :
  x - y = 9 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_nine_l1118_111834


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l1118_111859

theorem incorrect_average_calculation (n : Nat) (incorrect_num correct_num : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 75 ∧ 
  correct_avg = 51 →
  ∃ (S : ℚ), 
    (S + correct_num) / n = correct_avg ∧
    (S + incorrect_num) / n = 46 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l1118_111859


namespace NUMINAMATH_CALUDE_coefficient_x3_is_correct_l1118_111858

/-- The coefficient of x^3 in the expansion of (x^2 - 2x)(1 + x)^6 -/
def coefficient_x3 : ℤ := -24

/-- The expansion of (x^2 - 2x)(1 + x)^6 -/
def expansion (x : ℚ) : ℚ := (x^2 - 2*x) * (1 + x)^6

theorem coefficient_x3_is_correct :
  (∃ f : ℚ → ℚ, ∀ x, expansion x = f x + coefficient_x3 * x^3 + x^4 * f x) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_correct_l1118_111858


namespace NUMINAMATH_CALUDE_translation_preserves_vector_translation_problem_l1118_111865

/-- A translation in 2D space -/
structure Translation (α : Type*) [Add α] :=
  (dx dy : α)

/-- Apply a translation to a point -/
def apply_translation {α : Type*} [Add α] (t : Translation α) (p : α × α) : α × α :=
  (p.1 + t.dx, p.2 + t.dy)

theorem translation_preserves_vector {α : Type*} [AddCommGroup α] 
  (t : Translation α) (a b c d : α × α) :
  apply_translation t a = c →
  apply_translation t b = d →
  c.1 - a.1 = d.1 - b.1 ∧ c.2 - a.2 = d.2 - b.2 :=
sorry

/-- The main theorem to prove -/
theorem translation_problem :
  ∃ (t : Translation ℤ),
    apply_translation t (-1, 4) = (3, 6) ∧
    apply_translation t (-3, 2) = (1, 4) :=
sorry

end NUMINAMATH_CALUDE_translation_preserves_vector_translation_problem_l1118_111865


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l1118_111828

theorem intersection_point_k_value :
  ∀ (k : ℝ),
  (∃ (y : ℝ), -3 * (-6) + 2 * y = k ∧ 0.75 * (-6) + y = 16) →
  k = 59 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l1118_111828


namespace NUMINAMATH_CALUDE_janet_lives_l1118_111869

theorem janet_lives (initial_lives lost_lives gained_lives : ℕ) :
  initial_lives ≥ lost_lives →
  initial_lives - lost_lives + gained_lives =
    initial_lives + gained_lives - lost_lives :=
by
  sorry

#check janet_lives 38 16 32

end NUMINAMATH_CALUDE_janet_lives_l1118_111869


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l1118_111853

theorem quadratic_equation_positive_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^2 - 2*x₁ + m + 1 = 0 ∧ x₂^2 - 2*x₂ + m + 1 = 0) ↔ 
  (-1 < m ∧ m ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l1118_111853


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_condition_l1118_111823

theorem sine_cosine_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ c > Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_condition_l1118_111823


namespace NUMINAMATH_CALUDE_ribbon_cutting_l1118_111881

-- Define the lengths of the two ribbons
def ribbon1_length : ℕ := 28
def ribbon2_length : ℕ := 16

-- Define the function to calculate the maximum length of shorter ribbons
def max_short_ribbon_length (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the function to calculate the total number of shorter ribbons
def total_short_ribbons (a b c : ℕ) : ℕ := (a + b) / c

-- Theorem statement
theorem ribbon_cutting :
  (max_short_ribbon_length ribbon1_length ribbon2_length = 4) ∧
  (total_short_ribbons ribbon1_length ribbon2_length (max_short_ribbon_length ribbon1_length ribbon2_length) = 11) :=
by sorry

end NUMINAMATH_CALUDE_ribbon_cutting_l1118_111881


namespace NUMINAMATH_CALUDE_no_charming_seven_digit_number_l1118_111845

/-- A function that checks if a list of digits forms a charming number -/
def is_charming (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = Finset.range 7 ∧
  (∀ k : Nat, k ∈ Finset.range 7 → 
    (digits.take k).foldl (fun acc d => acc * 10 + d) 0 % k = 0) ∧
  digits.getLast? = some 7

/-- Theorem stating that no charming 7-digit number exists -/
theorem no_charming_seven_digit_number : 
  ¬ ∃ (digits : List Nat), is_charming digits := by
  sorry

end NUMINAMATH_CALUDE_no_charming_seven_digit_number_l1118_111845


namespace NUMINAMATH_CALUDE_existence_implies_range_l1118_111871

theorem existence_implies_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x - a) < 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_range_l1118_111871


namespace NUMINAMATH_CALUDE_stamps_needed_l1118_111833

/-- The weight of one piece of paper in ounces -/
def paper_weight : ℚ := 1/5

/-- The number of pieces of paper used -/
def num_papers : ℕ := 8

/-- The weight of the envelope in ounces -/
def envelope_weight : ℚ := 2/5

/-- The number of stamps needed per ounce -/
def stamps_per_ounce : ℕ := 1

/-- The theorem stating the number of stamps needed for Jessica's letter -/
theorem stamps_needed : 
  ⌈(num_papers * paper_weight + envelope_weight) * stamps_per_ounce⌉ = 2 := by
  sorry

end NUMINAMATH_CALUDE_stamps_needed_l1118_111833


namespace NUMINAMATH_CALUDE_min_value_of_f_l1118_111854

/-- The function f(x) = 5x^2 + 10x + 20 -/
def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

/-- The minimum value of f(x) is 15 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 15 ∧ ∀ x, f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1118_111854


namespace NUMINAMATH_CALUDE_remainder_of_2543_base12_div_9_l1118_111846

/-- Converts a base-12 digit to its decimal value -/
def base12ToDecimal (digit : Nat) : Nat :=
  if digit < 12 then digit else 0

/-- Converts a base-12 number to its decimal equivalent -/
def convertBase12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun digit acc => acc * 12 + base12ToDecimal digit) 0

/-- The base-12 representation of 2543 -/
def base12Number : List Nat := [2, 5, 4, 3]

theorem remainder_of_2543_base12_div_9 :
  (convertBase12ToDecimal base12Number) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2543_base12_div_9_l1118_111846


namespace NUMINAMATH_CALUDE_mechanic_worked_five_and_half_hours_l1118_111870

/-- Calculates the number of hours a mechanic worked given the total cost, part costs, labor rate, and break time. -/
def mechanic_work_hours (total_cost parts_cost labor_rate_per_minute break_minutes : ℚ) : ℚ :=
  ((total_cost - parts_cost) / labor_rate_per_minute - break_minutes) / 60

/-- Proves that the mechanic worked 5.5 hours given the problem conditions. -/
theorem mechanic_worked_five_and_half_hours :
  let total_cost : ℚ := 220
  let parts_cost : ℚ := 2 * 20
  let labor_rate_per_minute : ℚ := 0.5
  let break_minutes : ℚ := 30
  mechanic_work_hours total_cost parts_cost labor_rate_per_minute break_minutes = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_mechanic_worked_five_and_half_hours_l1118_111870


namespace NUMINAMATH_CALUDE_krista_savings_exceed_target_l1118_111890

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- The first day Krista deposits money -/
def initialDeposit : ℚ := 3

/-- The ratio by which Krista increases her deposit each day -/
def depositRatio : ℚ := 3

/-- The amount Krista wants to exceed in cents -/
def targetAmount : ℚ := 2000

theorem krista_savings_exceed_target :
  (∀ k < 7, geometricSum initialDeposit depositRatio k ≤ targetAmount) ∧
  geometricSum initialDeposit depositRatio 7 > targetAmount :=
sorry

end NUMINAMATH_CALUDE_krista_savings_exceed_target_l1118_111890


namespace NUMINAMATH_CALUDE_min_value_expression_l1118_111806

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c) ≥ 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1118_111806


namespace NUMINAMATH_CALUDE_snack_bar_employees_l1118_111898

theorem snack_bar_employees (total : ℕ) (buffet dining : ℕ) (two_restaurants all_restaurants : ℕ) : 
  total = 39 →
  buffet = 17 →
  dining = 18 →
  two_restaurants = 4 →
  all_restaurants = 2 →
  ∃ (snack_bar : ℕ), 
    snack_bar = total - (buffet + dining - two_restaurants - all_restaurants) := by
  sorry

end NUMINAMATH_CALUDE_snack_bar_employees_l1118_111898


namespace NUMINAMATH_CALUDE_shopping_with_refund_l1118_111895

/-- Calculates the remaining money after shopping with a partial refund --/
theorem shopping_with_refund 
  (initial_amount : ℕ) 
  (sweater_cost t_shirt_cost shoes_cost : ℕ) 
  (refund_percentage : ℚ) : 
  initial_amount = 74 →
  sweater_cost = 9 →
  t_shirt_cost = 11 →
  shoes_cost = 30 →
  refund_percentage = 90 / 100 →
  initial_amount - (sweater_cost + t_shirt_cost + (shoes_cost * (1 - refund_percentage))) = 51 := by
  sorry

end NUMINAMATH_CALUDE_shopping_with_refund_l1118_111895


namespace NUMINAMATH_CALUDE_sum_of_squares_l1118_111822

theorem sum_of_squares (a b c : ℝ) 
  (sum_condition : a + b + c = 5)
  (product_sum_condition : a * b + b * c + a * c = 5) :
  a^2 + b^2 + c^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1118_111822


namespace NUMINAMATH_CALUDE_jen_bird_count_l1118_111840

/-- The number of ducks Jen has -/
def num_ducks : ℕ := 150

/-- The number of chickens Jen has -/
def num_chickens : ℕ := (num_ducks - 10) / 4

/-- The total number of birds Jen has -/
def total_birds : ℕ := num_ducks + num_chickens

theorem jen_bird_count : total_birds = 185 := by
  sorry

end NUMINAMATH_CALUDE_jen_bird_count_l1118_111840


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l1118_111875

/-- Given a total of 344 coins consisting of 20 paise and 25 paise coins,
    with a total value of Rs. 71, prove that the number of 20 paise coins is 300. -/
theorem twenty_paise_coins_count :
  ∀ (x y : ℕ),
  x + y = 344 →
  20 * x + 25 * y = 7100 →
  x = 300 :=
by sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l1118_111875


namespace NUMINAMATH_CALUDE_min_sum_squares_l1118_111848

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ a^2 + b^2 + c^2 ≥ m ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ x^2 + y^2 + z^2 = m :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1118_111848


namespace NUMINAMATH_CALUDE_candy_chocolate_difference_l1118_111888

theorem candy_chocolate_difference (initial_candy : ℕ) (additional_candy : ℕ) (chocolate : ℕ) :
  initial_candy = 38 →
  additional_candy = 36 →
  chocolate = 16 →
  (initial_candy + additional_candy) - chocolate = 58 := by
  sorry

end NUMINAMATH_CALUDE_candy_chocolate_difference_l1118_111888


namespace NUMINAMATH_CALUDE_part_one_part_two_l1118_111847

-- Define combinatorial and permutation functions
def C (n k : ℕ) : ℕ := Nat.choose n k
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Part 1: Prove that C₁₀⁴ - C₇³A₃³ = 0
theorem part_one : C 10 4 - C 7 3 * A 3 3 = 0 := by sorry

-- Part 2: Prove that the solution to 3A₈ˣ = 4A₉ˣ⁻¹ is x = 6
theorem part_two : ∃ (x : ℕ), 3 * A 8 x = 4 * A 9 (x - 1) ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1118_111847


namespace NUMINAMATH_CALUDE_fourth_side_length_l1118_111802

/-- A quadrilateral inscribed in a circle with three known side lengths -/
structure InscribedQuadrilateral where
  -- Three known side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- The fourth side length
  d : ℝ
  -- Condition that the quadrilateral is inscribed in a circle
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  -- Condition that the areas of triangles ABC and ACD are equal
  equal_areas : a * b = c * d

/-- Theorem stating the possible values of the fourth side length -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h1 : q.a = 5 ∨ q.b = 5 ∨ q.c = 5)
  (h2 : q.a = 8 ∨ q.b = 8 ∨ q.c = 8)
  (h3 : q.a = 10 ∨ q.b = 10 ∨ q.c = 10) :
  q.d = 4 ∨ q.d = 6.25 ∨ q.d = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l1118_111802


namespace NUMINAMATH_CALUDE_area_rectangle_circumscribing_right_triangle_l1118_111836

/-- The area of a rectangle circumscribing a right triangle with legs of length 5 and 6 is 30. -/
theorem area_rectangle_circumscribing_right_triangle : 
  ∀ (A B C D E : ℝ × ℝ),
    -- Right triangle ABC
    (B.1 - A.1) * (C.2 - B.2) = (C.1 - B.1) * (B.2 - A.2) →
    -- AB = 5
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 →
    -- BC = 6
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 →
    -- Rectangle ADEC circumscribes triangle ABC
    A.1 = D.1 ∧ A.2 = D.2 ∧
    C.1 = E.1 ∧ C.2 = E.2 ∧
    D.2 = E.2 ∧ A.1 = C.1 →
    -- Area of rectangle ADEC is 30
    (E.1 - D.1) * (E.2 - D.2) = 30 := by
  sorry


end NUMINAMATH_CALUDE_area_rectangle_circumscribing_right_triangle_l1118_111836


namespace NUMINAMATH_CALUDE_figure_b_cannot_be_assembled_l1118_111814

-- Define the basic rhombus
structure Rhombus :=
  (color1 : String)
  (color2 : String)

-- Define the operation of rotation
def rotate (r : Rhombus) : Rhombus := r

-- Define the larger figures
inductive LargerFigure
  | A
  | B
  | C
  | D

-- Define a function to check if a larger figure can be assembled
def can_assemble (figure : LargerFigure) (r : Rhombus) : Prop :=
  match figure with
  | LargerFigure.A => True
  | LargerFigure.B => False
  | LargerFigure.C => True
  | LargerFigure.D => True

-- Theorem statement
theorem figure_b_cannot_be_assembled (r : Rhombus) :
  ¬(can_assemble LargerFigure.B r) ∧
  (can_assemble LargerFigure.A r) ∧
  (can_assemble LargerFigure.C r) ∧
  (can_assemble LargerFigure.D r) :=
sorry

end NUMINAMATH_CALUDE_figure_b_cannot_be_assembled_l1118_111814


namespace NUMINAMATH_CALUDE_optimalPlan_is_most_cost_effective_l1118_111873

/-- Represents a vehicle type with its capacity and cost -/
structure VehicleType where
  peopleCapacity : ℕ
  luggageCapacity : ℕ
  cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

def totalStudents : ℕ := 290
def totalLuggage : ℕ := 100
def totalVehicles : ℕ := 8

def typeA : VehicleType := ⟨40, 10, 2000⟩
def typeB : VehicleType := ⟨30, 20, 1800⟩

def isValidPlan (plan : RentalPlan) : Prop :=
  plan.typeA + plan.typeB = totalVehicles ∧
  plan.typeA * typeA.peopleCapacity + plan.typeB * typeB.peopleCapacity ≥ totalStudents ∧
  plan.typeA * typeA.luggageCapacity + plan.typeB * typeB.luggageCapacity ≥ totalLuggage

def planCost (plan : RentalPlan) : ℕ :=
  plan.typeA * typeA.cost + plan.typeB * typeB.cost

def optimalPlan : RentalPlan := ⟨5, 3⟩

theorem optimalPlan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → planCost optimalPlan ≤ planCost plan :=
sorry

end NUMINAMATH_CALUDE_optimalPlan_is_most_cost_effective_l1118_111873


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1118_111872

/-- The maximum value of y = -x^2 + 4x + 3, where x is a real number, is 7. -/
theorem max_value_quadratic :
  ∃ (y_max : ℝ), y_max = 7 ∧ ∀ (x : ℝ), -x^2 + 4*x + 3 ≤ y_max :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1118_111872


namespace NUMINAMATH_CALUDE_sum_factorials_mod_5_l1118_111892

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials : ℕ := 
  (factorial 1) + (factorial 2) + (factorial 3) + (factorial 4) + 
  (factorial 5) + (factorial 6) + (factorial 7) + (factorial 8) + 
  (factorial 9) + (factorial 10)

theorem sum_factorials_mod_5 : sum_factorials % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_5_l1118_111892


namespace NUMINAMATH_CALUDE_olivia_trip_length_l1118_111810

theorem olivia_trip_length :
  ∀ (total_length : ℚ),
    (1 / 4 : ℚ) * total_length + 30 + (1 / 6 : ℚ) * total_length = total_length →
    total_length = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_olivia_trip_length_l1118_111810


namespace NUMINAMATH_CALUDE_log_equation_solution_l1118_111813

theorem log_equation_solution :
  ∃ t : ℝ, t > 0 ∧ 4 * (Real.log t / Real.log 3) = Real.log (4 * t) / Real.log 3 → t = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1118_111813


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1118_111825

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(a,1) and point B(5,b) are symmetric with respect to the origin O, prove that a + b = -6 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1118_111825


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_l1118_111827

/-- Given a rectangular field with width 4 m and area 12 m², 
    prove that its diagonal is 5 m. -/
theorem rectangular_field_diagonal : 
  ∀ (w l d : ℝ), 
    w = 4 → 
    w * l = 12 → 
    d ^ 2 = w ^ 2 + l ^ 2 → 
    d = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_l1118_111827


namespace NUMINAMATH_CALUDE_homework_problem_l1118_111884

theorem homework_problem (p t : ℕ) (h1 : p > 15) (h2 : p * t = (2*p - 6) * (t - 3)) : p * t = 126 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l1118_111884


namespace NUMINAMATH_CALUDE_zeros_in_nine_nines_squared_l1118_111805

/-- The number of zeros in the square of a number consisting of n repeated 9s -/
def zeros_in_square (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- The number 999,999,999 -/
def nine_nines : ℕ := 999999999

theorem zeros_in_nine_nines_squared :
  zeros_in_square 9 = 8 :=
sorry

#check zeros_in_nine_nines_squared

end NUMINAMATH_CALUDE_zeros_in_nine_nines_squared_l1118_111805


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l1118_111843

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- State the theorem
theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : min (fib n / fib (n-1)) (fib (n+1) / fib n) < a / b ∧ 
            a / b < max (fib n / fib (n-1)) (fib (n+1) / fib n)) : 
  b ≥ fib (n+1) := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_inequality_l1118_111843


namespace NUMINAMATH_CALUDE_value_calculation_l1118_111838

theorem value_calculation (number : ℝ) (value : ℝ) : 
  number = 8 → 
  value = 0.75 * number + 2 → 
  value = 8 := by
sorry

end NUMINAMATH_CALUDE_value_calculation_l1118_111838


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_l1118_111819

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryRepresentation := List Nat

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : Nat) : BinaryRepresentation :=
  sorry

/-- Checks if a given BinaryRepresentation is valid (contains only 0s and 1s) -/
def isValidBinary (b : BinaryRepresentation) : Prop :=
  sorry

/-- Converts a binary representation back to decimal -/
def binaryToDecimal (b : BinaryRepresentation) : Nat :=
  sorry

theorem decimal_93_to_binary :
  let binary : BinaryRepresentation := [1, 0, 1, 1, 1, 0, 1]
  isValidBinary binary ∧
  binaryToDecimal binary = 93 ∧
  decimalToBinary 93 = binary :=
by sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_l1118_111819


namespace NUMINAMATH_CALUDE_min_value_expression_l1118_111874

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1118_111874


namespace NUMINAMATH_CALUDE_river_objects_l1118_111801

/-- The number of objects Bill tossed into the river -/
def bill_objects (ted_sticks ted_rocks bill_sticks bill_rocks : ℕ) : ℕ :=
  bill_sticks + bill_rocks

/-- The problem statement -/
theorem river_objects 
  (ted_sticks ted_rocks : ℕ) 
  (h1 : ted_sticks = 10)
  (h2 : ted_rocks = 10)
  (h3 : ∃ bill_sticks : ℕ, bill_sticks = ted_sticks + 6)
  (h4 : ∃ bill_rocks : ℕ, ted_rocks = 2 * bill_rocks) :
  ∃ bill_sticks bill_rocks : ℕ, 
    bill_objects ted_sticks ted_rocks bill_sticks bill_rocks = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_river_objects_l1118_111801


namespace NUMINAMATH_CALUDE_total_shirts_is_ten_l1118_111857

/-- Represents the total number of shirts sold by the retailer -/
def total_shirts : ℕ := 10

/-- Represents the number of initially sold shirts -/
def initial_shirts : ℕ := 3

/-- Represents the prices of the initially sold shirts -/
def initial_prices : List ℝ := [20, 22, 25]

/-- Represents the desired overall average price -/
def desired_average : ℝ := 20

/-- Represents the minimum average price of the remaining shirts -/
def min_remaining_average : ℝ := 19

/-- Theorem stating that the total number of shirts is 10 given the conditions -/
theorem total_shirts_is_ten :
  total_shirts = initial_shirts + (total_shirts - initial_shirts) ∧
  (List.sum initial_prices + min_remaining_average * (total_shirts - initial_shirts)) / total_shirts > desired_average :=
by sorry

end NUMINAMATH_CALUDE_total_shirts_is_ten_l1118_111857


namespace NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l1118_111866

/-- A six-digit integer formed by repeating a positive three-digit integer -/
def repeatedDigitNumber (m : ℕ) : Prop :=
  100 ≤ m ∧ m < 1000 ∧ ∃ n : ℕ, n = 1001 * m

/-- The greatest common divisor of all six-digit integers formed by repeating a positive three-digit integer is 1001 -/
theorem gcd_repeated_digit_numbers :
  ∃ d : ℕ, d > 0 ∧ (∀ n : ℕ, repeatedDigitNumber n → d ∣ n) ∧
  ∀ k : ℕ, k > 0 → (∀ n : ℕ, repeatedDigitNumber n → k ∣ n) → k ∣ d :=
by sorry

end NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l1118_111866


namespace NUMINAMATH_CALUDE_lara_overtakes_darla_l1118_111887

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- The speed ratio of Lara to Darla -/
def speed_ratio : ℝ := 1.2

/-- The number of laps completed by Lara when she first overtakes Darla -/
def laps_completed : ℝ := 6

/-- Theorem stating that Lara completes 6 laps when she first overtakes Darla -/
theorem lara_overtakes_darla :
  ∃ (t : ℝ), t > 0 ∧ speed_ratio * t * track_length = t * track_length + track_length ∧
  laps_completed = speed_ratio * t * track_length / track_length :=
sorry

end NUMINAMATH_CALUDE_lara_overtakes_darla_l1118_111887


namespace NUMINAMATH_CALUDE_sum_of_ages_l1118_111849

/-- Given information about the ages of Nacho, Divya, and Samantha, prove that the sum of their current ages is 80 years. -/
theorem sum_of_ages (nacho divya samantha : ℕ) : 
  (divya = 5) →
  (nacho + 5 = 3 * (divya + 5)) →
  (samantha = 2 * nacho) →
  (nacho + divya + samantha = 80) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1118_111849


namespace NUMINAMATH_CALUDE_exactly_one_solves_l1118_111829

theorem exactly_one_solves (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2)
  (h_B : p_B = 1/3)
  (h_C : p_C = 1/4)
  (h_independent : True) -- Representing the independence condition
  : p_A * (1 - p_B) * (1 - p_C) + 
    (1 - p_A) * p_B * (1 - p_C) + 
    (1 - p_A) * (1 - p_B) * p_C = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_solves_l1118_111829


namespace NUMINAMATH_CALUDE_expression_evaluation_l1118_111826

theorem expression_evaluation (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1118_111826


namespace NUMINAMATH_CALUDE_complex_number_problem_l1118_111804

theorem complex_number_problem (α β : ℂ) :
  (α + β).re > 0 →
  (Complex.I * (α - 3 * β)).re > 0 →
  β = 2 + 3 * Complex.I →
  α = 6 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1118_111804


namespace NUMINAMATH_CALUDE_painting_selection_theorem_l1118_111862

/-- The number of traditional Chinese paintings -/
def traditional_paintings : Nat := 6

/-- The number of oil paintings -/
def oil_paintings : Nat := 4

/-- The number of watercolor paintings -/
def watercolor_paintings : Nat := 5

/-- The number of ways to select one painting from each type -/
def select_one_each : Nat := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to select two paintings of different types -/
def select_two_different : Nat :=
  traditional_paintings * oil_paintings +
  traditional_paintings * watercolor_paintings +
  oil_paintings * watercolor_paintings

theorem painting_selection_theorem :
  select_one_each = 120 ∧ select_two_different = 74 := by
  sorry

end NUMINAMATH_CALUDE_painting_selection_theorem_l1118_111862


namespace NUMINAMATH_CALUDE_hexagon_trapezoid_height_l1118_111856

/-- Given a 9 × 16 rectangle cut into two congruent hexagons that can form a larger rectangle
    with width 12, prove that the height of the internal trapezoid in one hexagon is 12. -/
theorem hexagon_trapezoid_height (original_width : ℝ) (original_height : ℝ)
  (resultant_width : ℝ) (y : ℝ) :
  original_width = 16 ∧ original_height = 9 ∧ resultant_width = 12 →
  y = 12 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_trapezoid_height_l1118_111856


namespace NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1118_111815

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The chord length of a circle intercepted by a line -/
def chordLength (c : Circle) (l : Line) : ℝ := sorry

/-- The given circle x^2 + y^2 - 4y = 0 -/
def givenCircle : Circle :=
  { center := (0, 2),
    radius := 2 }

/-- The line passing through the origin with slope 1 -/
def givenLine : Line :=
  { slope := 1,
    yIntercept := 0 }

/-- Theorem: The chord length of the given circle intercepted by the given line is 2√2 -/
theorem chord_length_is_2_sqrt_2 :
  chordLength givenCircle givenLine = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1118_111815


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l1118_111830

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0
def line2 (x y : ℝ) : Prop := x + y + 2 = 0
def line3 (x y : ℝ) : Prop := 3 * x + y - 1 = 0
def result_line (x y : ℝ) : Prop := 3 * x + y = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallel lines
def parallel_lines (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

theorem line_through_intersection_and_parallel :
  (∃ (x y : ℝ), intersection_point x y ∧ result_line x y) ∧
  parallel_lines line3 result_line :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l1118_111830


namespace NUMINAMATH_CALUDE_arccos_neg_sqrt3_div2_l1118_111885

theorem arccos_neg_sqrt3_div2 :
  Real.arccos (-Real.sqrt 3 / 2) = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_sqrt3_div2_l1118_111885


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l1118_111864

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) :
  Nat.Prime p → 1 ≤ k → k < p →
  ∃ m : ℕ, Nat.choose p k = m * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l1118_111864


namespace NUMINAMATH_CALUDE_classroom_seating_arrangements_l1118_111863

/-- Represents a seating arrangement of students in a classroom -/
structure SeatingArrangement where
  rows : Nat
  cols : Nat
  boys : Nat
  girls : Nat

/-- Calculates the number of valid seating arrangements -/
def validArrangements (s : SeatingArrangement) : Nat :=
  2 * (Nat.factorial s.boys) * (Nat.factorial s.girls)

/-- Theorem stating the number of valid seating arrangements
    for the given classroom configuration -/
theorem classroom_seating_arrangements :
  let s : SeatingArrangement := {
    rows := 5,
    cols := 6,
    boys := 15,
    girls := 15
  }
  validArrangements s = 2 * (Nat.factorial 15) * (Nat.factorial 15) :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_seating_arrangements_l1118_111863


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1118_111877

theorem partial_fraction_decomposition (x A B C : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  (6 * x) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2 ↔ 
  A = 3 ∧ B = -3 ∧ C = -6 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1118_111877


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l1118_111835

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_height : ℝ
  painting_width : ℝ
  frame_side_width : ℝ

/-- Calculates the framed dimensions of the painting -/
def framed_dimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.frame_side_width, 
   fp.painting_height + 6 * fp.frame_side_width)

/-- Calculates the area of the framed painting -/
def framed_area (fp : FramedPainting) : ℝ :=
  let (w, h) := framed_dimensions fp
  w * h

/-- Theorem stating the ratio of smaller to larger dimension of the framed painting -/
theorem framed_painting_ratio 
  (fp : FramedPainting)
  (h1 : fp.painting_height = 30)
  (h2 : fp.painting_width = 20)
  (h3 : framed_area fp = fp.painting_height * fp.painting_width) :
  let (w, h) := framed_dimensions fp
  min w h / max w h = 4 / 7 := by
    sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l1118_111835


namespace NUMINAMATH_CALUDE_line_x_intercept_l1118_111832

/-- The x-intercept of a line passing through (2, -2) and (6, 10) is 8/3 -/
theorem line_x_intercept :
  let p1 : ℝ × ℝ := (2, -2)
  let p2 : ℝ × ℝ := (6, 10)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let x_intercept : ℝ := -b / m
  x_intercept = 8/3 := by
sorry


end NUMINAMATH_CALUDE_line_x_intercept_l1118_111832


namespace NUMINAMATH_CALUDE_geralds_bag_contains_40_apples_l1118_111880

/-- The number of bags Pam has -/
def pams_bags : ℕ := 10

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- The number of apples in each of Gerald's bags -/
def geralds_bag_apples : ℕ := pams_total_apples / (3 * pams_bags)

/-- Theorem stating that each of Gerald's bags contains 40 apples -/
theorem geralds_bag_contains_40_apples : geralds_bag_apples = 40 := by
  sorry

end NUMINAMATH_CALUDE_geralds_bag_contains_40_apples_l1118_111880


namespace NUMINAMATH_CALUDE_m_range_proof_l1118_111852

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a < b ∧ (m + 1 = a) ∧ (3 - m = b)

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 2*m + 3 ≠ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 3

-- Theorem statement
theorem m_range_proof (m : ℝ) : (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_proof_l1118_111852


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1118_111817

theorem min_value_of_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (∀ x y : ℝ, x > y ∧ x * y = 1 → (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2) ∧
  ∃ x y : ℝ, x > y ∧ x * y = 1 ∧ (x^2 + y^2) / (x - y) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1118_111817


namespace NUMINAMATH_CALUDE_table_runner_coverage_l1118_111889

theorem table_runner_coverage (
  total_runner_area : ℝ)
  (table_area : ℝ)
  (two_layer_area : ℝ)
  (three_layer_area : ℝ)
  (h1 : total_runner_area = 212)
  (h2 : table_area = 175)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 24)
  : ∃ (coverage_percentage : ℝ),
    abs (coverage_percentage - 52.57) < 0.01 ∧
    coverage_percentage = (total_runner_area - 2 * two_layer_area - 3 * three_layer_area) / table_area * 100 := by
  sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l1118_111889


namespace NUMINAMATH_CALUDE_fruit_seller_gain_percent_l1118_111831

theorem fruit_seller_gain_percent (cost_price selling_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : selling_price > 0)
  (h3 : 150 * selling_price - 150 * cost_price = 30 * selling_price) :
  (((150 * selling_price - 150 * cost_price) / (150 * cost_price)) * 100 = 25) :=
by sorry

end NUMINAMATH_CALUDE_fruit_seller_gain_percent_l1118_111831


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1118_111811

theorem weight_of_new_person
  (n : ℕ) -- number of persons
  (old_weight : ℝ) -- weight of the person being replaced
  (avg_increase : ℝ) -- increase in average weight
  (h1 : n = 15) -- there are 15 persons
  (h2 : old_weight = 80) -- the replaced person weighs 80 kg
  (h3 : avg_increase = 3.2) -- average weight increases by 3.2 kg
  : ∃ (new_weight : ℝ), new_weight = n * avg_increase + old_weight :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1118_111811


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1118_111879

theorem quadratic_minimum (c : ℝ) : 
  (1/3 : ℝ) * c^2 + 6*c + 4 ≥ (1/3 : ℝ) * (-9)^2 + 6*(-9) + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1118_111879


namespace NUMINAMATH_CALUDE_shirt_store_profit_optimization_l1118_111878

/-- Represents the daily profit function for a shirt store -/
def daily_profit (x : ℝ) : ℝ := (20 + 2*x) * (40 - x)

/-- Represents the price reduction that achieves a specific daily profit -/
def price_reduction_for_profit (target_profit : ℝ) : ℝ :=
  20 -- The actual value should be solved from the equation, but we're using the known result

/-- Represents the price reduction that maximizes daily profit -/
def optimal_price_reduction : ℝ := 15

/-- The maximum daily profit achieved at the optimal price reduction -/
def max_daily_profit : ℝ := 1250

theorem shirt_store_profit_optimization :
  (daily_profit (price_reduction_for_profit 1200) = 1200) ∧
  (∀ x : ℝ, daily_profit x ≤ max_daily_profit) ∧
  (daily_profit optimal_price_reduction = max_daily_profit) := by
  sorry


end NUMINAMATH_CALUDE_shirt_store_profit_optimization_l1118_111878


namespace NUMINAMATH_CALUDE_smallest_three_star_three_star_divisibility_l1118_111850

/-- A three-star number is a positive three-digit integer that is the product of three distinct prime numbers. -/
def is_three_star (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ p q r : ℕ, 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    n = p * q * r

/-- The smallest three-star number is 102. -/
theorem smallest_three_star : 
  is_three_star 102 ∧ ∀ n, is_three_star n → 102 ≤ n :=
sorry

/-- Every three-star number is divisible by 2, 3, or 5. -/
theorem three_star_divisibility (n : ℕ) :
  is_three_star n → (2 ∣ n) ∨ (3 ∣ n) ∨ (5 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_star_three_star_divisibility_l1118_111850
