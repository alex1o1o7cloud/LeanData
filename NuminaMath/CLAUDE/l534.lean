import Mathlib

namespace NUMINAMATH_CALUDE_ratio_x_to_y_l534_53480

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 3/5) : 
  x / y = 16/15 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l534_53480


namespace NUMINAMATH_CALUDE_max_product_sum_300_l534_53426

theorem max_product_sum_300 : 
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ 
  (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l534_53426


namespace NUMINAMATH_CALUDE_average_fuel_efficiency_l534_53429

/-- Calculates the average fuel efficiency for a trip with multiple segments and different vehicles. -/
theorem average_fuel_efficiency 
  (total_distance : ℝ)
  (sedan_distance : ℝ)
  (truck_distance : ℝ)
  (detour_distance : ℝ)
  (sedan_efficiency : ℝ)
  (truck_efficiency : ℝ)
  (detour_efficiency : ℝ)
  (h1 : total_distance = sedan_distance + truck_distance + detour_distance)
  (h2 : sedan_distance = 150)
  (h3 : truck_distance = 150)
  (h4 : detour_distance = 50)
  (h5 : sedan_efficiency = 25)
  (h6 : truck_efficiency = 15)
  (h7 : detour_efficiency = 10) :
  ∃ (ε : ℝ), abs (total_distance / (sedan_distance / sedan_efficiency + 
                                    truck_distance / truck_efficiency + 
                                    detour_distance / detour_efficiency) - 16.67) < ε :=
by sorry

end NUMINAMATH_CALUDE_average_fuel_efficiency_l534_53429


namespace NUMINAMATH_CALUDE_alok_mixed_veg_order_l534_53408

/-- Represents the order and payment details of Alok's meal --/
structure MealOrder where
  chapatis : ℕ
  rice_plates : ℕ
  ice_cream_cups : ℕ
  chapati_cost : ℕ
  rice_cost : ℕ
  mixed_veg_cost : ℕ
  total_paid : ℕ

/-- Calculates the number of mixed vegetable plates ordered --/
def mixed_veg_plates (order : MealOrder) : ℕ :=
  ((order.total_paid - (order.chapatis * order.chapati_cost + order.rice_plates * order.rice_cost)) / order.mixed_veg_cost)

/-- Theorem stating that Alok ordered 9 plates of mixed vegetable --/
theorem alok_mixed_veg_order : 
  ∀ (order : MealOrder), 
    order.chapatis = 16 ∧ 
    order.rice_plates = 5 ∧ 
    order.ice_cream_cups = 6 ∧
    order.chapati_cost = 6 ∧ 
    order.rice_cost = 45 ∧ 
    order.mixed_veg_cost = 70 ∧ 
    order.total_paid = 961 → 
    mixed_veg_plates order = 9 := by
  sorry


end NUMINAMATH_CALUDE_alok_mixed_veg_order_l534_53408


namespace NUMINAMATH_CALUDE_sum_of_integers_l534_53483

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l534_53483


namespace NUMINAMATH_CALUDE_duplicated_chromosome_configuration_l534_53484

/-- Represents a duplicated chromosome -/
structure DuplicatedChromosome where
  centromeres : ℕ
  chromatids : ℕ
  dna_molecules : ℕ

/-- The correct configuration of a duplicated chromosome -/
def correct_configuration : DuplicatedChromosome :=
  { centromeres := 1
  , chromatids := 2
  , dna_molecules := 2 }

/-- Theorem stating that a duplicated chromosome has the correct configuration -/
theorem duplicated_chromosome_configuration :
  ∀ (dc : DuplicatedChromosome), dc = correct_configuration :=
by sorry

end NUMINAMATH_CALUDE_duplicated_chromosome_configuration_l534_53484


namespace NUMINAMATH_CALUDE_uncovered_area_l534_53416

/-- The area not covered by a smaller square and a right triangle within a larger square -/
theorem uncovered_area (larger_side small_side triangle_base triangle_height : ℝ) 
  (h1 : larger_side = 10)
  (h2 : small_side = 4)
  (h3 : triangle_base = 3)
  (h4 : triangle_height = 3)
  : larger_side ^ 2 - (small_side ^ 2 + (triangle_base * triangle_height) / 2) = 79.5 := by
  sorry

#check uncovered_area

end NUMINAMATH_CALUDE_uncovered_area_l534_53416


namespace NUMINAMATH_CALUDE_sum_of_coefficients_of_expanded_f_l534_53435

-- Define the polynomial expression
def f (c : ℝ) : ℝ := 2 * (c - 2) * (c^2 + c * (4 - c))

-- Define the sum of coefficients function
def sumOfCoefficients (p : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem sum_of_coefficients_of_expanded_f :
  sumOfCoefficients f = -8 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_of_expanded_f_l534_53435


namespace NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_66_l534_53417

def fine_per_mph : ℝ := 16
def total_fine : ℝ := 256
def speed_limit : ℝ := 50

theorem jeds_speed : ℝ :=
  speed_limit + total_fine / fine_per_mph

theorem jeds_speed_is_66 : jeds_speed = 66 := by sorry

end NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_66_l534_53417


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l534_53494

theorem opposite_of_negative_five : -((-5) : ℝ) = (5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l534_53494


namespace NUMINAMATH_CALUDE_cone_base_circumference_l534_53476

/-- The circumference of the base of a right circular cone with volume 24π cubic centimeters and height 6 cm is 4√3π cm. -/
theorem cone_base_circumference :
  ∀ (V h r : ℝ),
  V = 24 * Real.pi ∧
  h = 6 ∧
  V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l534_53476


namespace NUMINAMATH_CALUDE_prism_volume_l534_53477

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (x y z : ℝ) 
  (h₁ : x * y = 20)  -- side face area
  (h₂ : y * z = 12)  -- front face area
  (h₃ : x * z = 8)   -- bottom face area
  : x * y * z = 8 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l534_53477


namespace NUMINAMATH_CALUDE_sqrt_5x_plus_y_squared_l534_53458

theorem sqrt_5x_plus_y_squared (x y : ℝ) 
  (h : Real.sqrt (x - 1) + (3 * x + y - 1)^2 = 0) : 
  Real.sqrt (5 * x + y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5x_plus_y_squared_l534_53458


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l534_53439

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x - 3) * (x + 1) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 12)) →
  p = -8 ∧ q = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l534_53439


namespace NUMINAMATH_CALUDE_student_difference_l534_53457

/-- Given that the sum of students in grades 1 and 2 is 30 more than the sum of students in grades 2 and 5,
    prove that the difference between the number of students in grade 1 and grade 5 is 30. -/
theorem student_difference (g1 g2 g5 : ℕ) (h : g1 + g2 = g2 + g5 + 30) : g1 - g5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_difference_l534_53457


namespace NUMINAMATH_CALUDE_nested_average_equals_seven_ninths_l534_53427

def average2 (a b : ℚ) : ℚ := (a + b) / 2

def average3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_equals_seven_ninths :
  average3 (average3 2 2 0) (average2 0 2) 0 = 7/9 := by sorry

end NUMINAMATH_CALUDE_nested_average_equals_seven_ninths_l534_53427


namespace NUMINAMATH_CALUDE_powers_of_two_in_arithmetic_sequence_l534_53441

theorem powers_of_two_in_arithmetic_sequence (k : ℕ) :
  (∃ n : ℕ, 2^k = 6*n + 8) ↔ (k > 1 ∧ k % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_powers_of_two_in_arithmetic_sequence_l534_53441


namespace NUMINAMATH_CALUDE_interest_calculation_l534_53400

/-- Represents the interest calculation problem -/
theorem interest_calculation (x y z : ℝ) 
  (h1 : x * y / 100 * 2 = 800)  -- Simple interest condition
  (h2 : x * ((1 + y / 100)^2 - 1) = 820)  -- Compound interest condition
  : x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l534_53400


namespace NUMINAMATH_CALUDE_carls_garden_area_l534_53495

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_separation : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the garden --/
def garden_area (g : Garden) : ℕ :=
  let longer_side_posts := 2 * g.shorter_side_posts
  let shorter_side_length := (g.shorter_side_posts - 1) * g.post_separation
  let longer_side_length := (longer_side_posts - 1) * g.post_separation
  shorter_side_length * longer_side_length

/-- Theorem stating that Carl's garden has an area of 900 square yards --/
theorem carls_garden_area :
  ∀ (g : Garden),
    g.total_posts = 26 ∧
    g.post_separation = 5 ∧
    g.shorter_side_posts = 5 →
    garden_area g = 900 := by
  sorry

end NUMINAMATH_CALUDE_carls_garden_area_l534_53495


namespace NUMINAMATH_CALUDE_triangle_rotation_l534_53453

/-- Triangle OAB with given properties and rotation of OA --/
theorem triangle_rotation (A : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (5, 0)
  let angle_ABO : ℝ := π / 2  -- 90°
  let angle_AOB : ℝ := π / 6  -- 30°
  let rotation_angle : ℝ := 2 * π / 3  -- 120°
  A.1 > 0 ∧ A.2 > 0 →  -- A is in the first quadrant
  (A.1 - O.1) * (B.2 - O.2) = (B.1 - O.1) * (A.2 - O.2) →  -- ABO is a right angle
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 
    Real.cos angle_AOB * Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) * Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2) →
  let rotated_A : ℝ × ℝ := (
    A.1 * Real.cos rotation_angle - A.2 * Real.sin rotation_angle,
    A.1 * Real.sin rotation_angle + A.2 * Real.cos rotation_angle
  )
  rotated_A = (-5/2 * (1 + Real.sqrt 3), 5/2 * (Real.sqrt 3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rotation_l534_53453


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l534_53456

/-- Represents a dodecahedron -/
structure Dodecahedron where
  /-- The number of faces in a dodecahedron -/
  faces : ℕ
  /-- The number of vertices in a dodecahedron -/
  vertices : ℕ
  /-- The number of faces meeting at each vertex -/
  faces_per_vertex : ℕ
  /-- Assertion that the dodecahedron has 12 faces -/
  faces_eq : faces = 12
  /-- Assertion that the dodecahedron has 20 vertices -/
  vertices_eq : vertices = 20
  /-- Assertion that 3 faces meet at each vertex -/
  faces_per_vertex_eq : faces_per_vertex = 3

/-- Calculates the number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - d.faces_per_vertex - 1)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : 
  interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l534_53456


namespace NUMINAMATH_CALUDE_vector_expression_not_equal_PQ_l534_53496

variable {V : Type*} [AddCommGroup V]
variable (A B P Q : V)

theorem vector_expression_not_equal_PQ :
  A - B + B - P - (A - Q) ≠ P - Q :=
sorry

end NUMINAMATH_CALUDE_vector_expression_not_equal_PQ_l534_53496


namespace NUMINAMATH_CALUDE_odd_function_properties_l534_53432

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * x + m) / (x^2 + 1)

theorem odd_function_properties :
  ∃ (m : ℝ),
    (∀ x, f m x = -f m (-x)) ∧
    (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f m x < f m y) ∧
    (∀ x y, 1 ≤ x ∧ x < y → f m x > f m y) ∧
    (∀ x y, 0 ≤ x ∧ 0 ≤ y → f m x - f m y ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l534_53432


namespace NUMINAMATH_CALUDE_prob_at_most_one_red_l534_53420

/-- The probability of drawing at most 1 red ball from a bag of 8 balls (3 red, 2 white, 3 black) when randomly selecting 3 balls. -/
theorem prob_at_most_one_red (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ) 
  (h_total : total = 8)
  (h_red : red = 3)
  (h_white : white = 2)
  (h_black : black = 3)
  (h_sum : red + white + black = total)
  (draw : ℕ)
  (h_draw : draw = 3) :
  (Nat.choose (total - red) draw + Nat.choose red 1 * Nat.choose (total - red) (draw - 1)) / Nat.choose total draw = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_red_l534_53420


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l534_53404

theorem arithmetic_mean_of_three_numbers (a b c : ℝ) (h : a = 14 ∧ b = 22 ∧ c = 36) :
  (a + b + c) / 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l534_53404


namespace NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l534_53440

/-- Proves that the difference between Alberto's and Bjorn's biking distances is 10 miles -/
theorem alberto_bjorn_distance_difference : 
  ∀ (alberto_distance bjorn_distance : ℕ), 
  alberto_distance = 75 → 
  bjorn_distance = 65 → 
  alberto_distance - bjorn_distance = 10 := by
sorry

end NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l534_53440


namespace NUMINAMATH_CALUDE_solution_equations_solution_inequalities_l534_53409

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def equation2 (x y : ℝ) : Prop := 5 * x + 2 * y = 6

-- Define the system of inequalities
def inequality1 (x : ℝ) : Prop := 3 * x + 6 > 0
def inequality2 (x : ℝ) : Prop := x - 2 < -x

-- Theorem for the system of equations
theorem solution_equations :
  ∃! (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 1 ∧ y = 1/2 := by sorry

-- Theorem for the system of inequalities
theorem solution_inequalities :
  ∀ x : ℝ, inequality1 x ∧ inequality2 x ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_equations_solution_inequalities_l534_53409


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l534_53490

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l534_53490


namespace NUMINAMATH_CALUDE_min_value_expression1_min_value_expression2_min_value_expression3_l534_53411

/-- The minimum value of x^2 + y^2 + xy + x + y for real x and y is -1/3 -/
theorem min_value_expression1 :
  ∃ (m : ℝ), m = -1/3 ∧ ∀ (x y : ℝ), x^2 + y^2 + x*y + x + y ≥ m :=
sorry

/-- The minimum value of x^2 + y^2 + z^2 + xy + yz + zx + x + y + z for real x, y, and z is -3/8 -/
theorem min_value_expression2 :
  ∃ (m : ℝ), m = -3/8 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + x*y + y*z + z*x + x + y + z ≥ m :=
sorry

/-- The minimum value of x^2 + y^2 + z^2 + r^2 + xy + xz + xr + yz + yr + zr + x + y + z + r for real x, y, z, and r is -2/5 -/
theorem min_value_expression3 :
  ∃ (m : ℝ), m = -2/5 ∧ ∀ (x y z r : ℝ),
    x^2 + y^2 + z^2 + r^2 + x*y + x*z + x*r + y*z + y*r + z*r + x + y + z + r ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression1_min_value_expression2_min_value_expression3_l534_53411


namespace NUMINAMATH_CALUDE_sum_of_divisors_2i3j_l534_53478

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j is 960, then i + j = 5 -/
theorem sum_of_divisors_2i3j (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 960 → i + j = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_2i3j_l534_53478


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l534_53442

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Check if a line passes through a point -/
def passesThrough (l : Line) (p : Point) : Prop := sorry

theorem tangent_line_to_circle (c : Circle) (p : Point) :
  c.center = Point.mk 2 0 →
  c.radius = 2 →
  p = Point.mk 4 5 →
  ∀ l : Line, (isTangent l c ∧ passesThrough l p) ↔ 
    (l = Line.mk 21 (-20) 16 ∨ l = Line.mk 1 0 (-4)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l534_53442


namespace NUMINAMATH_CALUDE_inequality_range_l534_53499

theorem inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l534_53499


namespace NUMINAMATH_CALUDE_product_of_ratios_l534_53462

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2004)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1002 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l534_53462


namespace NUMINAMATH_CALUDE_base_angle_measure_l534_53405

-- Define an isosceles triangle
structure IsoscelesTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180
  isosceles : angle2 = angle3

-- Theorem statement
theorem base_angle_measure (t : IsoscelesTriangle) (h : t.angle1 = 80 ∨ t.angle2 = 80) :
  t.angle2 = 50 ∨ t.angle2 = 80 := by
  sorry


end NUMINAMATH_CALUDE_base_angle_measure_l534_53405


namespace NUMINAMATH_CALUDE_equation_solution_l534_53464

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -3 ∧ x₂ = 2/3 ∧
  (∀ x : ℝ, 3*x*(x+3) = 2*(x+3) ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l534_53464


namespace NUMINAMATH_CALUDE_refrigerator_savings_l534_53463

/- Define the parameters of the problem -/
def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def num_installments : ℕ := 30
def installment_amount : ℕ := 300

/- Define the total amount paid in installments -/
def total_installments : ℕ := num_installments * installment_amount + deposit

/- Theorem statement -/
theorem refrigerator_savings : total_installments - cash_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l534_53463


namespace NUMINAMATH_CALUDE_female_rainbow_count_l534_53402

/-- Represents the number of trout in a fishery -/
structure Fishery where
  female_speckled : ℕ
  male_speckled : ℕ
  female_rainbow : ℕ
  male_rainbow : ℕ

/-- The conditions of the fishery problem -/
def fishery_conditions (f : Fishery) : Prop :=
  f.female_speckled + f.male_speckled = 645 ∧
  f.male_speckled = 2 * f.female_speckled + 45 ∧
  4 * f.male_rainbow = 3 * f.female_speckled ∧
  3 * (f.female_speckled + f.male_speckled + f.female_rainbow + f.male_rainbow) = 20 * f.male_rainbow

/-- The theorem stating that under the given conditions, there are 205 female rainbow trout -/
theorem female_rainbow_count (f : Fishery) :
  fishery_conditions f → f.female_rainbow = 205 := by
  sorry


end NUMINAMATH_CALUDE_female_rainbow_count_l534_53402


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l534_53452

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 7}

-- Define the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l534_53452


namespace NUMINAMATH_CALUDE_fourth_circle_radius_is_p_l534_53468

-- Define the right triangle
structure RightTriangle :=
  (a b c : ℝ)
  (right_angle : a^2 + b^2 = c^2)
  (perimeter : a + b + c = 2 * p)

-- Define the circles
structure Circles (t : RightTriangle) :=
  (r1 r2 r3 : ℝ)
  (externally_tangent : t.a = r2 + r3 ∧ t.b = r1 + r3 ∧ t.c = r1 + r2)
  (fourth_circle_radius : ℝ)
  (internally_tangent : 
    t.a = fourth_circle_radius - r3 + (fourth_circle_radius - r2) ∧
    t.b = fourth_circle_radius - r1 + (fourth_circle_radius - r3) ∧
    t.c = fourth_circle_radius - r1 + (fourth_circle_radius - r2))

-- The theorem to prove
theorem fourth_circle_radius_is_p (t : RightTriangle) (c : Circles t) : 
  c.fourth_circle_radius = p :=
sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_is_p_l534_53468


namespace NUMINAMATH_CALUDE_eight_digit_increasing_integers_mod_1000_l534_53482

theorem eight_digit_increasing_integers_mod_1000 : 
  (Nat.choose 17 8) % 1000 = 310 := by sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_integers_mod_1000_l534_53482


namespace NUMINAMATH_CALUDE_seymours_venus_flytraps_l534_53486

/-- Represents the plant shop inventory and fertilizer requirements --/
structure PlantShop where
  petunia_flats : ℕ
  petunias_per_flat : ℕ
  rose_flats : ℕ
  roses_per_flat : ℕ
  fertilizer_per_petunia : ℕ
  fertilizer_per_rose : ℕ
  fertilizer_per_venus_flytrap : ℕ
  total_fertilizer : ℕ

/-- Calculates the number of Venus flytraps in the shop --/
def venus_flytraps (shop : PlantShop) : ℕ :=
  let petunia_fertilizer := shop.petunia_flats * shop.petunias_per_flat * shop.fertilizer_per_petunia
  let rose_fertilizer := shop.rose_flats * shop.roses_per_flat * shop.fertilizer_per_rose
  let remaining_fertilizer := shop.total_fertilizer - petunia_fertilizer - rose_fertilizer
  remaining_fertilizer / shop.fertilizer_per_venus_flytrap

/-- Theorem stating that Seymour's shop has 2 Venus flytraps --/
theorem seymours_venus_flytraps :
  let shop : PlantShop := {
    petunia_flats := 4,
    petunias_per_flat := 8,
    rose_flats := 3,
    roses_per_flat := 6,
    fertilizer_per_petunia := 8,
    fertilizer_per_rose := 3,
    fertilizer_per_venus_flytrap := 2,
    total_fertilizer := 314
  }
  venus_flytraps shop = 2 := by
  sorry

end NUMINAMATH_CALUDE_seymours_venus_flytraps_l534_53486


namespace NUMINAMATH_CALUDE_problem_solution_l534_53406

theorem problem_solution (t : ℝ) (x y : ℝ) 
    (h1 : x = 3 - 2*t) 
    (h2 : y = 3*t + 6) 
    (h3 : x = 0) : 
  y = 21/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l534_53406


namespace NUMINAMATH_CALUDE_line_parallel_to_skew_line_l534_53466

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relations between lines
variable (parallel skew intersecting : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_to_skew_line
  (l1 l2 l3 : Line)
  (h1 : skew l1 l2)
  (h2 : parallel l3 l1) :
  intersecting l3 l2 ∨ skew l3 l2 :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_skew_line_l534_53466


namespace NUMINAMATH_CALUDE_prob_three_two_digit_dice_l534_53479

-- Define the number of dice
def num_dice : ℕ := 6

-- Define the number of sides on each die
def num_sides : ℕ := 12

-- Define the number of two-digit outcomes on a single die
def two_digit_outcomes : ℕ := 3

-- Define the probability of rolling a two-digit number on a single die
def prob_two_digit : ℚ := two_digit_outcomes / num_sides

-- Define the probability of rolling a one-digit number on a single die
def prob_one_digit : ℚ := 1 - prob_two_digit

-- Define the number of dice we want to show two-digit numbers
def target_two_digit : ℕ := 3

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- State the theorem
theorem prob_three_two_digit_dice :
  (binomial num_dice target_two_digit : ℚ) * prob_two_digit ^ target_two_digit * prob_one_digit ^ (num_dice - target_two_digit) = 135 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_two_digit_dice_l534_53479


namespace NUMINAMATH_CALUDE_range_of_m_unbounded_below_m_characterization_of_m_range_l534_53460

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 5/2 :=
by sorry

theorem unbounded_below_m : ∀ (k : ℝ), ∃ (m : ℝ), m < k ∧ (A ∪ B m = A) :=
by sorry

theorem characterization_of_m_range : 
  ∀ (m : ℝ), (A ∪ B m = A) ↔ m ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_unbounded_below_m_characterization_of_m_range_l534_53460


namespace NUMINAMATH_CALUDE_vector_orthogonality_l534_53412

theorem vector_orthogonality (a b c : ℝ × ℝ) (x : ℝ) : 
  a = (1, 2) → b = (1, 0) → c = (3, 4) → 
  (b.1 + x * a.1, b.2 + x * a.2) • c = 0 → 
  x = -3/11 := by
  sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l534_53412


namespace NUMINAMATH_CALUDE_kelly_has_8_students_l534_53491

/-- Represents the number of students in Kelly's class -/
def num_students : ℕ := sorry

/-- Represents the number of construction paper pieces needed per student -/
def paper_per_student : ℕ := 3

/-- Represents the number of glue bottles Kelly bought -/
def glue_bottles : ℕ := 6

/-- Represents the number of additional construction paper pieces Kelly bought -/
def additional_paper : ℕ := 5

/-- Represents the total number of supplies Kelly has left -/
def total_supplies : ℕ := 20

/-- Theorem stating that Kelly has 8 students given the conditions -/
theorem kelly_has_8_students :
  (((num_students * paper_per_student + glue_bottles) / 2 + additional_paper) = total_supplies) →
  num_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_kelly_has_8_students_l534_53491


namespace NUMINAMATH_CALUDE_positive_real_inequality_l534_53421

theorem positive_real_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a - b) * (a - c) / (2 * a^2 + (b + c)^2) +
  (b - c) * (b - a) / (2 * b^2 + (c + a)^2) +
  (c - a) * (c - b) / (2 * c^2 + (a + b)^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l534_53421


namespace NUMINAMATH_CALUDE_quadratic_roots_l534_53438

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l534_53438


namespace NUMINAMATH_CALUDE_penalty_kick_probability_l534_53487

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem penalty_kick_probability :
  let n : ℕ := 5
  let k : ℕ := 3
  let p : ℝ := 0.05
  abs (binomial_probability n k p - 0.00113) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kick_probability_l534_53487


namespace NUMINAMATH_CALUDE_weekday_hourly_brew_l534_53430

/-- Represents a coffee shop's brewing schedule and output -/
structure CoffeeShop where
  weekdayHourlyBrew : ℕ
  weekendTotalBrew : ℕ
  dailyHours : ℕ
  weeklyTotalBrew : ℕ

/-- Theorem stating the number of coffee cups brewed per hour on a weekday -/
theorem weekday_hourly_brew (shop : CoffeeShop) 
  (h1 : shop.dailyHours = 5)
  (h2 : shop.weekendTotalBrew = 120)
  (h3 : shop.weeklyTotalBrew = 370) :
  shop.weekdayHourlyBrew = 10 := by
  sorry

end NUMINAMATH_CALUDE_weekday_hourly_brew_l534_53430


namespace NUMINAMATH_CALUDE_circles_intersection_l534_53414

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ → ℝ × ℝ := λ m ↦ (m, -1)

-- Define the line equation
def line_equation (x y c : ℝ) : Prop := x - y + c = 0

-- Theorem statement
theorem circles_intersection (m c : ℝ) 
  (h1 : ∃ (center1 center2 : ℝ × ℝ), 
    line_equation center1.1 center1.2 c ∧ 
    line_equation center2.1 center2.2 c) : 
  m + c = 3 := by
sorry

end NUMINAMATH_CALUDE_circles_intersection_l534_53414


namespace NUMINAMATH_CALUDE_opposite_sides_of_line_l534_53451

theorem opposite_sides_of_line (m : ℝ) : 
  (∀ (x y : ℝ), 2*x + y - 2 = 0 → (2*(-2) + m - 2) * (2*m + 4 - 2) < 0) → 
  -1 < m ∧ m < 6 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_of_line_l534_53451


namespace NUMINAMATH_CALUDE_female_salmon_count_l534_53415

theorem female_salmon_count (male_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : total_salmon = 971639) :
  total_salmon - male_salmon = 259378 := by
  sorry

end NUMINAMATH_CALUDE_female_salmon_count_l534_53415


namespace NUMINAMATH_CALUDE_dan_youngest_l534_53419

def ages (a b c d e : ℕ) : Prop :=
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105

theorem dan_youngest (a b c d e : ℕ) (h : ages a b c d e) : 
  d < a ∧ d < b ∧ d < c ∧ d < e := by
  sorry

end NUMINAMATH_CALUDE_dan_youngest_l534_53419


namespace NUMINAMATH_CALUDE_university_applications_l534_53437

theorem university_applications (n : ℕ) (s : Fin 5 → ℕ) : 
  (∀ i, s i ≥ n / 2) → 
  ∃ i j, i ≠ j ∧ (s i).min (s j) ≥ n / 5 := by
  sorry


end NUMINAMATH_CALUDE_university_applications_l534_53437


namespace NUMINAMATH_CALUDE_calculation_proof_l534_53481

theorem calculation_proof : 
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l534_53481


namespace NUMINAMATH_CALUDE_symmetrical_line_equation_l534_53428

/-- Given two lines in the plane, this function returns the equation of a line symmetrical to the first line with respect to the second line. -/
def symmetricalLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line with equation 2x - y - 2 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  fun x y ↦ 2 * x - y - 2 = 0

/-- The line with equation x + y - 4 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  fun x y ↦ x + y - 4 = 0

/-- The theorem stating that the symmetrical line has the equation x - 2y + 2 = 0 -/
theorem symmetrical_line_equation :
  symmetricalLine line1 line2 = fun x y ↦ x - 2 * y + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetrical_line_equation_l534_53428


namespace NUMINAMATH_CALUDE_system_solution_l534_53461

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x = -10 - 3 * y) ∧ 
  (4 * x = 6 * y - 38) ∧ 
  (x = -29/9) ∧ 
  (y = -113/27) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l534_53461


namespace NUMINAMATH_CALUDE_greater_l_conference_teams_l534_53471

/-- The number of teams in the GREATER L conference -/
def n : ℕ := sorry

/-- The total number of games played in the season -/
def total_games : ℕ := 90

/-- The formula for the total number of games when each team plays every other team twice -/
def games_formula (x : ℕ) : ℕ := x * (x - 1)

theorem greater_l_conference_teams :
  n = 10 ∧ games_formula n = total_games := by sorry

end NUMINAMATH_CALUDE_greater_l_conference_teams_l534_53471


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l534_53470

theorem polar_to_rectangular (r θ : ℝ) (h : r = 7 ∧ θ = π/3) :
  (r * Real.cos θ, r * Real.sin θ) = (3.5, 7 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l534_53470


namespace NUMINAMATH_CALUDE_point_above_line_l534_53424

theorem point_above_line (a : ℝ) : 
  (2*a - (-1) + 1 < 0) ↔ (a < -1) := by sorry

end NUMINAMATH_CALUDE_point_above_line_l534_53424


namespace NUMINAMATH_CALUDE_line_slope_l534_53446

/-- Given a line passing through points (1,2) and (4,2+√3), its slope is √3/3 -/
theorem line_slope : ∃ (k : ℝ), k = (2 + Real.sqrt 3 - 2) / (4 - 1) ∧ k = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l534_53446


namespace NUMINAMATH_CALUDE_smallest_b_value_l534_53447

theorem smallest_b_value (a b : ℕ) : 
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (2 * a - b = x^2) ∧ 
    (a - 2 * b = y^2) ∧ 
    (a + b = z^2)) →
  (∀ b' : ℕ, b' < b → 
    ¬(∃ a' x' y' z' : ℕ, x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' ∧
      (2 * a' - b' = x'^2) ∧ 
      (a' - 2 * b' = y'^2) ∧ 
      (a' + b' = z'^2))) →
  b = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l534_53447


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l534_53433

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l534_53433


namespace NUMINAMATH_CALUDE_new_person_weight_l534_53465

/-- Given a group of 8 people where one person weighing 66 kg is replaced by a new person,
    if the average weight of the group increases by 2.5 kg,
    then the weight of the new person is 86 kg. -/
theorem new_person_weight (initial_group_size : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_group_size = 8 →
  weight_increase = 2.5 →
  replaced_weight = 66 →
  (initial_group_size : ℝ) * weight_increase + replaced_weight = 86 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l534_53465


namespace NUMINAMATH_CALUDE_complex_sixth_root_of_negative_eight_l534_53449

theorem complex_sixth_root_of_negative_eight :
  {z : ℂ | z^6 = -8} = {Complex.I * Real.rpow 2 (1/3), -Complex.I * Real.rpow 2 (1/3)} := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_root_of_negative_eight_l534_53449


namespace NUMINAMATH_CALUDE_exact_two_support_probability_l534_53410

/-- The probability of a voter supporting the law -/
def p_support : ℝ := 0.6

/-- The probability of a voter not supporting the law -/
def p_oppose : ℝ := 1 - p_support

/-- The number of voters selected -/
def n : ℕ := 5

/-- The number of voters supporting the law in our target scenario -/
def k : ℕ := 2

/-- The binomial coefficient for choosing k items from n items -/
def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k out of n voters supporting the law -/
def prob_exact_support (n k : ℕ) (p : ℝ) : ℝ :=
  (binom_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exact_two_support_probability :
  prob_exact_support n k p_support = 0.2304 := by sorry

end NUMINAMATH_CALUDE_exact_two_support_probability_l534_53410


namespace NUMINAMATH_CALUDE_intersecting_circles_sum_l534_53488

/-- Given two circles intersecting at points A and B, with their centers on a line,
    prove that m+2c equals 26 -/
theorem intersecting_circles_sum (m c : ℝ) : 
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (-6, m)
  let center_line (x y : ℝ) := x - y + c = 0
  -- Assume the centers of both circles lie on the line x - y + c = 0
  -- Assume A and B are intersection points of the two circles
  m + 2*c = 26 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_sum_l534_53488


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l534_53418

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l534_53418


namespace NUMINAMATH_CALUDE_min_plane_spotlights_theorem_min_space_spotlights_theorem_l534_53444

/-- A spotlight that illuminates a 90° plane angle --/
structure PlaneSpotlight where
  angle : ℝ
  angle_eq : angle = 90

/-- A spotlight that illuminates a trihedral angle with all plane angles of 90° --/
structure SpaceSpotlight where
  angle : ℝ
  angle_eq : angle = 90

/-- The minimum number of spotlights required to illuminate the entire plane --/
def min_plane_spotlights : ℕ := 4

/-- The minimum number of spotlights required to illuminate the entire space --/
def min_space_spotlights : ℕ := 8

/-- Theorem stating the minimum number of spotlights required for full plane illumination --/
theorem min_plane_spotlights_theorem (s : PlaneSpotlight) :
  min_plane_spotlights = 4 := by sorry

/-- Theorem stating the minimum number of spotlights required for full space illumination --/
theorem min_space_spotlights_theorem (s : SpaceSpotlight) :
  min_space_spotlights = 8 := by sorry

end NUMINAMATH_CALUDE_min_plane_spotlights_theorem_min_space_spotlights_theorem_l534_53444


namespace NUMINAMATH_CALUDE_seven_power_minus_two_power_l534_53473

theorem seven_power_minus_two_power : 
  ∀ x y : ℕ+, 7^(x.val) - 3 * 2^(y.val) = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_seven_power_minus_two_power_l534_53473


namespace NUMINAMATH_CALUDE_college_students_count_l534_53448

/-- Calculates the total number of students in a college given the ratio of boys to girls and the number of girls -/
def total_students (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : ℕ :=
  let num_boys := boys_ratio * num_girls / girls_ratio
  num_boys + num_girls

/-- Proves that in a college with a boys to girls ratio of 8:4 and 200 girls, the total number of students is 600 -/
theorem college_students_count : total_students 8 4 200 = 600 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l534_53448


namespace NUMINAMATH_CALUDE_additional_grazing_area_l534_53489

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 12^2 = 385 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l534_53489


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l534_53403

/-- The number of rectangles in a 4x4 grid -/
def num_rectangles_4x4 : ℕ := 36

/-- The number of ways to choose 2 items from 4 -/
def choose_2_from_4 : ℕ := 6

/-- Theorem: The number of rectangles in a 4x4 grid is 36 -/
theorem rectangles_in_4x4_grid :
  num_rectangles_4x4 = choose_2_from_4 * choose_2_from_4 :=
by sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l534_53403


namespace NUMINAMATH_CALUDE_exist_four_numbers_perfect_squares_l534_53455

theorem exist_four_numbers_perfect_squares :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exist_four_numbers_perfect_squares_l534_53455


namespace NUMINAMATH_CALUDE_inequality_proof_l534_53431

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h : (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2*a + b + c) * (2*b + c + d) * (2*c + d + a) * (2*d + a + b) * (a*b*c*d)^2 ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l534_53431


namespace NUMINAMATH_CALUDE_average_fuel_efficiency_l534_53497

/-- Calculate the average fuel efficiency for a round trip with two different vehicles -/
theorem average_fuel_efficiency
  (total_distance : ℝ)
  (distance_first_leg : ℝ)
  (efficiency_first_vehicle : ℝ)
  (efficiency_second_vehicle : ℝ)
  (h1 : total_distance = 300)
  (h2 : distance_first_leg = total_distance / 2)
  (h3 : efficiency_first_vehicle = 50)
  (h4 : efficiency_second_vehicle = 25) :
  (total_distance) / ((distance_first_leg / efficiency_first_vehicle) + 
  (distance_first_leg / efficiency_second_vehicle)) = 33 := by
  sorry

#check average_fuel_efficiency

end NUMINAMATH_CALUDE_average_fuel_efficiency_l534_53497


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l534_53472

/-- Given a geometric sequence {aₙ}, prove that if a₇ × a₉ = 4 and a₄ = 1, then a₁₂ = 16 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∃ (r : ℝ), ∀ n, a (n + 1) = a n * r) →  -- {aₙ} is a geometric sequence
  a 7 * a 9 = 4 →                         -- a₇ × a₉ = 4
  a 4 = 1 →                               -- a₄ = 1
  a 12 = 16 :=                            -- a₁₂ = 16
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l534_53472


namespace NUMINAMATH_CALUDE_maggies_total_earnings_l534_53434

/-- Calculates Maggie's earnings from selling magazine subscriptions -/
def maggies_earnings (price_per_subscription : ℕ) 
  (parents_subscriptions : ℕ) 
  (grandfather_subscriptions : ℕ) 
  (nextdoor_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := 
    parents_subscriptions + 
    grandfather_subscriptions + 
    nextdoor_subscriptions + 
    (2 * nextdoor_subscriptions)
  price_per_subscription * total_subscriptions

theorem maggies_total_earnings : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_maggies_total_earnings_l534_53434


namespace NUMINAMATH_CALUDE_product_of_common_ratios_l534_53498

/-- Given two nonconstant geometric sequences with different common ratios
    satisfying a specific equation, prove that the product of their common ratios is 9. -/
theorem product_of_common_ratios (x p r : ℝ) (hx : x ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  3 * x * p^2 - 4 * x * r^2 = 5 * (3 * x * p - 4 * x * r) →
  p * r = 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_common_ratios_l534_53498


namespace NUMINAMATH_CALUDE_condition_relationship_l534_53474

theorem condition_relationship :
  (∀ x : ℝ, (x - 1) / (x + 2) ≥ 0 → (x - 1) * (x + 2) ≥ 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) ≥ 0 ∧ ¬((x - 1) / (x + 2) ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l534_53474


namespace NUMINAMATH_CALUDE_equation_roots_l534_53459

theorem equation_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    (0.5 : ℝ)^(x^2 - m*x + 0.5*m - 1.5) = (Real.sqrt 8)^(m - 1) ∧
    (0.5 : ℝ)^(y^2 - m*y + 0.5*m - 1.5) = (Real.sqrt 8)^(m - 1))
  ↔ (m < 2 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l534_53459


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l534_53436

/-- Perimeter of a trapezoid EFGH with given properties -/
theorem trapezoid_perimeter (EF GH EG FH : ℝ) (h1 : EF = 40) (h2 : GH = 20) 
  (h3 : EG = 30) (h4 : FH = 45) : 
  EF + GH + Real.sqrt (EF ^ 2 - EG ^ 2) + Real.sqrt (FH ^ 2 - GH ^ 2) = 60 + 10 * Real.sqrt 7 + 5 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l534_53436


namespace NUMINAMATH_CALUDE_haley_marble_distribution_l534_53485

/-- Represents the number of marbles each boy receives when Haley's marbles are distributed equally -/
def marbles_per_boy (total_marbles : ℕ) (num_boys : ℕ) : ℕ :=
  total_marbles / num_boys

/-- Theorem stating that when 20 marbles are distributed equally between 2 boys, each boy receives 10 marbles -/
theorem haley_marble_distribution :
  marbles_per_boy 20 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_haley_marble_distribution_l534_53485


namespace NUMINAMATH_CALUDE_money_calculation_l534_53423

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def total_amount (n_50 : ℕ) (n_500 : ℕ) : ℕ :=
  50 * n_50 + 500 * n_500

/-- Proves that given 72 total notes with 57 being 50 rupee notes, the total amount is 10350 rupees -/
theorem money_calculation : total_amount 57 (72 - 57) = 10350 := by
  sorry

end NUMINAMATH_CALUDE_money_calculation_l534_53423


namespace NUMINAMATH_CALUDE_equation_solution_l534_53475

theorem equation_solution (a b : ℝ) :
  (a + b - 1)^2 = a^2 + b^2 - 1 ↔ a = 1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l534_53475


namespace NUMINAMATH_CALUDE_correct_equation_l534_53443

theorem correct_equation : 4 - 4 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l534_53443


namespace NUMINAMATH_CALUDE_school_choir_robe_cost_l534_53469

/-- Calculates the total cost of buying additional robes for a school choir, including discount and sales tax. -/
theorem school_choir_robe_cost
  (total_robes_needed : ℕ)
  (robes_owned : ℕ)
  (cost_per_robe : ℚ)
  (discount_rate : ℚ)
  (discount_threshold : ℕ)
  (sales_tax_rate : ℚ)
  (h1 : total_robes_needed = 30)
  (h2 : robes_owned = 12)
  (h3 : cost_per_robe = 2)
  (h4 : discount_rate = 15 / 100)
  (h5 : discount_threshold = 10)
  (h6 : sales_tax_rate = 8 / 100)
  : ∃ (final_cost : ℚ), final_cost = 3305 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_school_choir_robe_cost_l534_53469


namespace NUMINAMATH_CALUDE_tech_club_enrollment_l534_53492

theorem tech_club_enrollment (total : ℕ) (cs : ℕ) (robotics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 90)
  (h3 : robotics = 70)
  (h4 : both = 20) :
  total - (cs + robotics - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tech_club_enrollment_l534_53492


namespace NUMINAMATH_CALUDE_smaller_number_proof_l534_53401

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : 
  min x y = 4 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l534_53401


namespace NUMINAMATH_CALUDE_complex_equation_sum_squares_l534_53422

theorem complex_equation_sum_squares (a b : ℝ) :
  (a + Complex.I) / Complex.I = b + Complex.I * Real.sqrt 2 →
  a^2 + b^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_squares_l534_53422


namespace NUMINAMATH_CALUDE_absolute_value_equation_a_l534_53413

theorem absolute_value_equation_a (x : ℝ) : |x - 5| = 2 ↔ x = 3 ∨ x = 7 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_a_l534_53413


namespace NUMINAMATH_CALUDE_evaluate_expression_l534_53445

theorem evaluate_expression : 6 - 5 * (10 - (2 + 1)^2) * 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l534_53445


namespace NUMINAMATH_CALUDE_constant_function_l534_53450

variable (f : ℝ → ℝ)

theorem constant_function
  (h1 : Continuous f')
  (h2 : f 0 = 0)
  (h3 : ∀ x, |f' x| ≤ |f x|) :
  ∃ c, ∀ x, f x = c :=
sorry

end NUMINAMATH_CALUDE_constant_function_l534_53450


namespace NUMINAMATH_CALUDE_cannot_reach_target_l534_53493

/-- Represents a positive integer as a list of digits (most significant digit first) -/
def Digits := List Nat

/-- The starting number -/
def startNum : Digits := [1]

/-- The target 100-digit number -/
def targetNum : Digits := List.replicate 98 2 ++ [5, 2, 2, 2, 1]

/-- Checks if a number is valid (non-zero first digit) -/
def isValidNumber (d : Digits) : Prop := d.head? ≠ some 0

/-- Represents the operation of multiplying by 5 -/
def multiplyBy5 (d : Digits) : Digits := sorry

/-- Represents the operation of rearranging digits -/
def rearrangeDigits (d : Digits) : Digits := sorry

/-- Represents a sequence of operations -/
inductive Operation
| Multiply
| Rearrange

def applyOperation (op : Operation) (d : Digits) : Digits :=
  match op with
  | Operation.Multiply => multiplyBy5 d
  | Operation.Rearrange => rearrangeDigits d

/-- Theorem stating the impossibility of reaching the target number -/
theorem cannot_reach_target : 
  ∀ (ops : List Operation), 
    let finalNum := ops.foldl (λ acc op => applyOperation op acc) startNum
    isValidNumber finalNum → finalNum ≠ targetNum :=
by sorry

end NUMINAMATH_CALUDE_cannot_reach_target_l534_53493


namespace NUMINAMATH_CALUDE_triangle_sine_relation_l534_53467

theorem triangle_sine_relation (A B C : ℝ) (h : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) :
  Real.sin (A + π / 4) = Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_relation_l534_53467


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l534_53425

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_condition : a 2 + a 4 = 10
  geometric_condition : (a 2) ^ 2 = a 1 * a 5
  arithmetic_property : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  seq.a 1 = 1 ∧ ∀ n : ℕ, seq.a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l534_53425


namespace NUMINAMATH_CALUDE_sum_m_n_equals_34_l534_53454

theorem sum_m_n_equals_34 (m n : ℕ+) (p : ℚ) : 
  m + 15 < n + 5 →
  (m + (m + 5) + (m + 15) + (n + 5) + (n + 6) + (2 * n - 1)) / 6 = p →
  ((m + 15) + (n + 5)) / 2 = p →
  m + n = 34 := by sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_34_l534_53454


namespace NUMINAMATH_CALUDE_tan_product_identity_l534_53407

theorem tan_product_identity : (1 + Real.tan (3 * π / 180)) * (1 + Real.tan (42 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_identity_l534_53407
