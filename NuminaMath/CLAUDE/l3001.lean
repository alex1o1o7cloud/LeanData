import Mathlib

namespace NUMINAMATH_CALUDE_number_difference_l3001_300100

theorem number_difference (a b : ℕ) : 
  b = 10 * a + 5 →
  a + b = 22500 →
  b - a = 18410 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3001_300100


namespace NUMINAMATH_CALUDE_equation_solution_l3001_300178

theorem equation_solution : ∃! x : ℝ, (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ∧ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3001_300178


namespace NUMINAMATH_CALUDE_percentage_of_12_to_80_l3001_300117

theorem percentage_of_12_to_80 : ∀ (x : ℝ), x = 12 ∧ (x / 80) * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_of_12_to_80_l3001_300117


namespace NUMINAMATH_CALUDE_arrange_seven_white_five_black_l3001_300167

/-- The number of ways to arrange white and black balls with constraints -/
def arrangeBalls (white black : ℕ) : ℕ :=
  Nat.choose (white + 1) black

/-- Theorem stating the number of ways to arrange 7 white balls and 5 black balls -/
theorem arrange_seven_white_five_black :
  arrangeBalls 7 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_arrange_seven_white_five_black_l3001_300167


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l3001_300118

theorem greatest_integer_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5*x - 4 < 3 - 2*x := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l3001_300118


namespace NUMINAMATH_CALUDE_triangles_with_integer_sides_not_exceeding_two_l3001_300126

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_sides_not_exceeding_two (a b c : ℕ) : Prop :=
  a ≤ 2 ∧ b ≤ 2 ∧ c ≤ 2

theorem triangles_with_integer_sides_not_exceeding_two :
  ∃! (S : Set (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ 
      is_valid_triangle a b c ∧ 
      triangle_sides_not_exceeding_two a b c) ∧
    S = {(1, 1, 1), (2, 2, 1), (2, 2, 2)} :=
sorry

end NUMINAMATH_CALUDE_triangles_with_integer_sides_not_exceeding_two_l3001_300126


namespace NUMINAMATH_CALUDE_single_digit_sum_l3001_300110

/-- Given two different single-digit numbers A and B where AB × 6 = BBB, 
    prove that A + B = 11 -/
theorem single_digit_sum (A B : ℕ) : 
  A ≠ B ∧ 
  A < 10 ∧ 
  B < 10 ∧ 
  (10 * A + B) * 6 = 100 * B + 10 * B + B → 
  A + B = 11 := by
sorry

end NUMINAMATH_CALUDE_single_digit_sum_l3001_300110


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3001_300102

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x < 0 → x^2019 < x^2018)) ↔ 
  (∃ x : ℝ, x < 0 ∧ x^2019 ≥ x^2018) := by
sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3001_300102


namespace NUMINAMATH_CALUDE_inequality_solution_l3001_300146

theorem inequality_solution (x : ℝ) : (3*x + 7)/5 + 1 > x ↔ x < 6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3001_300146


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l3001_300172

-- Define the first inequality
def inequality1 (x : ℝ) : Prop := (5 * (x - 1)) / 6 - 1 < (x + 2) / 3

-- Define the second system of inequalities
def inequality2 (x : ℝ) : Prop := 3 * x - 2 ≤ x + 6 ∧ (5 * x + 3) / 2 > x

-- Theorem for the first inequality
theorem solution_inequality1 : 
  {x : ℕ | inequality1 x} = {1, 2, 3, 4} :=
sorry

-- Theorem for the second system of inequalities
theorem solution_inequality2 : 
  {x : ℝ | inequality2 x} = {x : ℝ | -1 < x ∧ x ≤ 4} :=
sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l3001_300172


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3001_300174

theorem lcm_gcd_product (a b : ℕ) (ha : a = 11) (hb : b = 12) :
  Nat.lcm a b * Nat.gcd a b = 132 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3001_300174


namespace NUMINAMATH_CALUDE_base_addition_theorem_l3001_300150

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

theorem base_addition_theorem :
  let base13_number := [3, 5, 7]
  let base14_number := [4, 12, 13]
  (base_to_decimal base13_number 13) + (base_to_decimal base14_number 14) = 1544 := by
  sorry

end NUMINAMATH_CALUDE_base_addition_theorem_l3001_300150


namespace NUMINAMATH_CALUDE_total_average_marks_specific_classes_l3001_300194

/-- The total average marks of students in three classes -/
def total_average_marks (class1_students : ℕ) (class1_avg : ℚ)
                        (class2_students : ℕ) (class2_avg : ℚ)
                        (class3_students : ℕ) (class3_avg : ℚ) : ℚ :=
  (class1_avg * class1_students + class2_avg * class2_students + class3_avg * class3_students) /
  (class1_students + class2_students + class3_students)

/-- Theorem stating the total average marks of students in three specific classes -/
theorem total_average_marks_specific_classes :
  total_average_marks 47 52 33 68 40 75 = 7688 / 120 :=
by sorry

end NUMINAMATH_CALUDE_total_average_marks_specific_classes_l3001_300194


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l3001_300129

def A (a : ℝ) : Set ℝ := {1, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem subset_implies_a_value (a : ℝ) (h : B a ⊆ A a) : a = -1 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l3001_300129


namespace NUMINAMATH_CALUDE_jesse_room_area_l3001_300123

/-- The length of Jesse's room in feet -/
def room_length : ℝ := 12

/-- The width of Jesse's room in feet -/
def room_width : ℝ := 8

/-- The area of Jesse's room floor in square feet -/
def room_area : ℝ := room_length * room_width

theorem jesse_room_area : room_area = 96 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l3001_300123


namespace NUMINAMATH_CALUDE_four_points_same_inradius_congruent_triangles_l3001_300163

-- Define a structure for a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a structure for a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a function to calculate the inradius of a triangle
noncomputable def inradius (t : Triangle) : ℝ := sorry

-- Define a predicate for triangle congruence
def is_congruent (t1 t2 : Triangle) : Prop := sorry

-- Main theorem
theorem four_points_same_inradius_congruent_triangles 
  (A B C D : Point) 
  (h_same_inradius : ∃ r : ℝ, 
    inradius (Triangle.mk A B C) = r ∧
    inradius (Triangle.mk A B D) = r ∧
    inradius (Triangle.mk A C D) = r ∧
    inradius (Triangle.mk B C D) = r) :
  is_congruent (Triangle.mk A B C) (Triangle.mk A B D) ∧
  is_congruent (Triangle.mk A B C) (Triangle.mk A C D) ∧
  is_congruent (Triangle.mk A B C) (Triangle.mk B C D) :=
sorry

end NUMINAMATH_CALUDE_four_points_same_inradius_congruent_triangles_l3001_300163


namespace NUMINAMATH_CALUDE_roots_sum_absolute_values_l3001_300147

theorem roots_sum_absolute_values (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2013*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  abs a + abs b + abs c = 94 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_absolute_values_l3001_300147


namespace NUMINAMATH_CALUDE_treasure_chest_gems_l3001_300132

theorem treasure_chest_gems (diamonds : ℕ) (rubies : ℕ) 
    (h1 : diamonds = 45) 
    (h2 : rubies = 5110) : 
  diamonds + rubies = 5155 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_gems_l3001_300132


namespace NUMINAMATH_CALUDE_smallest_nut_count_l3001_300160

def nut_division (N : ℕ) (i : ℕ) : ℕ :=
  match i with
  | 0 => N
  | i + 1 => (nut_division N i - 1) / 5

theorem smallest_nut_count :
  ∀ N : ℕ, (∀ i : ℕ, i ≤ 5 → nut_division N i % 5 = 1) ↔ N ≥ 15621 :=
sorry

end NUMINAMATH_CALUDE_smallest_nut_count_l3001_300160


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l3001_300104

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem complex_determinant_equation :
  ∃ (z : ℂ), det z Complex.I 1 Complex.I = 1 + Complex.I ∧ z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l3001_300104


namespace NUMINAMATH_CALUDE_paint_intensity_problem_l3001_300131

/-- Given an original paint intensity of 50%, a new paint intensity of 30%,
    and 2/3 of the original paint replaced, prove that the intensity of
    the added paint solution is 20%. -/
theorem paint_intensity_problem (original_intensity new_intensity fraction_replaced : ℚ)
    (h1 : original_intensity = 50/100)
    (h2 : new_intensity = 30/100)
    (h3 : fraction_replaced = 2/3) :
    let added_intensity := (new_intensity - original_intensity * (1 - fraction_replaced)) / fraction_replaced
    added_intensity = 20/100 := by
  sorry

end NUMINAMATH_CALUDE_paint_intensity_problem_l3001_300131


namespace NUMINAMATH_CALUDE_proposition_3_proposition_4_l3001_300162

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Line → Prop)
variable (linePlaneParallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Axioms
axiom distinct_lines : m ≠ n
axiom non_coincident_planes : α ≠ β

-- Theorem for proposition 3
theorem proposition_3 
  (h1 : perpendicular m α)
  (h2 : perpendicular n β)
  (h3 : lineParallel m n) :
  parallel α β :=
sorry

-- Theorem for proposition 4
theorem proposition_4
  (h1 : skew m n)
  (h2 : linePlaneParallel m α)
  (h3 : linePlaneParallel m β)
  (h4 : linePlaneParallel n α)
  (h5 : linePlaneParallel n β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_proposition_3_proposition_4_l3001_300162


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequalities_l3001_300106

theorem arithmetic_geometric_mean_inequalities
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequalities_l3001_300106


namespace NUMINAMATH_CALUDE_ordering_of_abc_l3001_300149

theorem ordering_of_abc : 
  let a : ℝ := (1.7 : ℝ) ^ (0.9 : ℝ)
  let b : ℝ := (0.9 : ℝ) ^ (1.7 : ℝ)
  let c : ℝ := 1
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ordering_of_abc_l3001_300149


namespace NUMINAMATH_CALUDE_first_consecutive_shot_probability_value_l3001_300107

/-- The probability of making a shot -/
def shot_probability : ℚ := 2/3

/-- The number of attempts before the first consecutive shot -/
def attempts : ℕ := 6

/-- The probability of making the first consecutive shot on the 7th attempt -/
def first_consecutive_shot_probability : ℚ :=
  (1 - shot_probability)^attempts * shot_probability^2

theorem first_consecutive_shot_probability_value :
  first_consecutive_shot_probability = 8/729 := by
  sorry

end NUMINAMATH_CALUDE_first_consecutive_shot_probability_value_l3001_300107


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_2024_l3001_300175

/-- Given a quadratic equation with roots satisfying specific conditions, 
    the maximum value of the sum of their reciprocals raised to the 2024th power is 2. -/
theorem max_reciprocal_sum_2024 (s p r₁ r₂ : ℝ) : 
  (r₁^2 - s*r₁ + p = 0) →
  (r₂^2 - s*r₂ + p = 0) →
  (∀ (n : ℕ), n ≤ 2023 → r₁^n + r₂^n = s) →
  (∃ (max : ℝ), max = 2 ∧ 
    ∀ (s' p' r₁' r₂' : ℝ), 
      (r₁'^2 - s'*r₁' + p' = 0) →
      (r₂'^2 - s'*r₂' + p' = 0) →
      (∀ (n : ℕ), n ≤ 2023 → r₁'^n + r₂'^n = s') →
      1/r₁'^2024 + 1/r₂'^2024 ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_2024_l3001_300175


namespace NUMINAMATH_CALUDE_mars_mission_cost_share_l3001_300182

-- Define the total cost in billions of dollars
def total_cost : ℕ := 25

-- Define the number of people sharing the cost in millions
def num_people : ℕ := 200

-- Define the conversion factor from billions to millions
def billion_to_million : ℕ := 1000

-- Theorem statement
theorem mars_mission_cost_share :
  (total_cost * billion_to_million) / num_people = 125 := by
sorry

end NUMINAMATH_CALUDE_mars_mission_cost_share_l3001_300182


namespace NUMINAMATH_CALUDE_car_distance_theorem_l3001_300122

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc i => acc + (initialSpeed + i * speedIncrease)) 0

/-- Proves that a car traveling for 12 hours, starting at 45 km/h and increasing
    speed by 2 km/h each hour, travels a total of 672 km. -/
theorem car_distance_theorem :
  totalDistance 45 2 12 = 672 := by
  sorry

#eval totalDistance 45 2 12

end NUMINAMATH_CALUDE_car_distance_theorem_l3001_300122


namespace NUMINAMATH_CALUDE_decimal_sum_difference_l3001_300183

theorem decimal_sum_difference : (0.5 : ℚ) - 0.03 + 0.007 = 0.477 := by sorry

end NUMINAMATH_CALUDE_decimal_sum_difference_l3001_300183


namespace NUMINAMATH_CALUDE_equation_solution_l3001_300141

theorem equation_solution : 
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3001_300141


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3001_300154

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3001_300154


namespace NUMINAMATH_CALUDE_pencil_count_l3001_300192

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 5 →
  pencils = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3001_300192


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l3001_300187

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15 ∧ |x₂ + 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l3001_300187


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3001_300115

theorem units_digit_of_product (n : ℕ) : (3^1001 * 7^1002 * 13^1003) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3001_300115


namespace NUMINAMATH_CALUDE_square_equation_solution_l3001_300128

theorem square_equation_solution : ∃! (N : ℕ), N > 0 ∧ 36^2 * 60^2 = 30^2 * N^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3001_300128


namespace NUMINAMATH_CALUDE_new_students_count_l3001_300190

theorem new_students_count (initial_students : Nat) (left_students : Nat) (final_students : Nat) :
  initial_students = 11 →
  left_students = 6 →
  final_students = 47 →
  final_students - (initial_students - left_students) = 42 :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l3001_300190


namespace NUMINAMATH_CALUDE_power_of_five_l3001_300153

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l3001_300153


namespace NUMINAMATH_CALUDE_cubic_function_property_l3001_300127

/-- A cubic function g(x) = ax^3 + bx^2 + cx + d with g(0) = 3 and g(1) = 5 satisfies a + 2b + c + 3d = 0 -/
theorem cubic_function_property (a b c d : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d
  (g 0 = 3) → (g 1 = 5) → (a + 2*b + c + 3*d = 0) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3001_300127


namespace NUMINAMATH_CALUDE_equal_roots_implies_a_equals_negative_one_l3001_300180

/-- The quadratic equation with parameter a -/
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x * (x + 1) + a * x

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ := (1 + a)^2

theorem equal_roots_implies_a_equals_negative_one :
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ 
    ∀ y : ℝ, quadratic_equation a y = 0 → y = x) →
  discriminant a = 0 →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_implies_a_equals_negative_one_l3001_300180


namespace NUMINAMATH_CALUDE_maria_carrots_l3001_300130

def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  (initial - thrown_out) + new_picked

theorem maria_carrots (initial thrown_out new_picked : ℕ) 
  (h1 : initial ≥ thrown_out) : 
  total_carrots initial thrown_out new_picked = initial - thrown_out + new_picked :=
by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_l3001_300130


namespace NUMINAMATH_CALUDE_NQ_passes_through_fixed_point_l3001_300142

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line l
def line (k p : ℝ) (x y : ℝ) : Prop := y = k*(x + p/2)

-- Define the intersection points M and N
def intersection_points (p k : ℝ) (M N : ℝ × ℝ) : Prop :=
  parabola p M.1 M.2 ∧ parabola p N.1 N.2 ∧
  line k p M.1 M.2 ∧ line k p N.1 N.2 ∧
  M ≠ N

-- Define the chord length condition
def chord_length_condition (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16*15

-- Define the third intersection point Q
def third_intersection (p : ℝ) (M N Q : ℝ × ℝ) : Prop :=
  parabola p Q.1 Q.2 ∧ Q ≠ M ∧ Q ≠ N

-- Define point B
def point_B : ℝ × ℝ := (1, -1)

-- Define the condition that MQ passes through B
def MQ_through_B (M Q : ℝ × ℝ) : Prop :=
  (point_B.2 - M.2) * (Q.1 - M.1) = (Q.2 - M.2) * (point_B.1 - M.1)

-- Theorem statement
theorem NQ_passes_through_fixed_point (p k : ℝ) (M N Q : ℝ × ℝ) :
  p > 0 →
  k = 1/2 →
  intersection_points p k M N →
  chord_length_condition p M N →
  third_intersection p M N Q →
  MQ_through_B M Q →
  ∃ (fixed_point : ℝ × ℝ), fixed_point = (1, -4) ∧
    (fixed_point.2 - N.2) * (Q.1 - N.1) = (Q.2 - N.2) * (fixed_point.1 - N.1) :=
sorry

end NUMINAMATH_CALUDE_NQ_passes_through_fixed_point_l3001_300142


namespace NUMINAMATH_CALUDE_starting_elevation_l3001_300191

/-- Calculates the starting elevation of a person climbing a hill --/
theorem starting_elevation (final_elevation horizontal_distance : ℝ) : 
  final_elevation = 1450 ∧ 
  horizontal_distance = 2700 → 
  final_elevation - (horizontal_distance / 2) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_starting_elevation_l3001_300191


namespace NUMINAMATH_CALUDE_rectangle_circle_tangency_l3001_300109

theorem rectangle_circle_tangency (r : ℝ) (a b : ℝ) : 
  r = 6 →                             -- Circle radius is 6 cm
  a ≥ b →                             -- a is the longer side, b is the shorter side
  b = 2 * r →                         -- Circle is tangent to shorter side
  a * b = 3 * (π * r^2) →             -- Rectangle area is triple the circle area
  b = 12 :=                           -- Shorter side length is 12 cm
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangency_l3001_300109


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3001_300138

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity 
  (h : Hyperbola a b) 
  (F A B P O : Point) -- F is the right focus, O is the origin
  (right_branch : Point → Prop) -- Predicate for points on the right branch
  (on_hyperbola : Point → Prop) -- Predicate for points on the hyperbola
  (line_through : Point → Point → Point → Prop) -- Predicate for collinear points
  (symmetric_to : Point → Point → Point → Prop) -- Predicate for point symmetry
  (perpendicular : Point → Point → Point → Point → Prop) -- Predicate for perpendicular lines
  (h_right_focus : F.x > 0 ∧ F.y = 0)
  (h_AB_on_C : on_hyperbola A ∧ on_hyperbola B)
  (h_AB_right : right_branch A ∧ right_branch B)
  (h_line_FAB : line_through F A B)
  (h_A_symmetric : symmetric_to A O P)
  (h_PF_perp_AB : perpendicular P F A B)
  (h_BF_3AF : distance B F = 3 * distance A F)
  : eccentricity h = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3001_300138


namespace NUMINAMATH_CALUDE_square_perimeter_from_circle_l3001_300195

theorem square_perimeter_from_circle (circle_perimeter : ℝ) : 
  circle_perimeter = 52.5 → 
  ∃ (square_perimeter : ℝ), square_perimeter = 210 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_circle_l3001_300195


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3001_300179

theorem cube_root_equation_solution :
  ∃ (a b c : ℕ+),
    (2 * (7^(1/3) + 6^(1/3))^(1/2) : ℝ) = a^(1/3) - b^(1/3) + c^(1/3) ∧
    a + b + c = 42 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3001_300179


namespace NUMINAMATH_CALUDE_choose_three_roles_from_eight_people_l3001_300124

def number_of_people : ℕ := 8
def number_of_roles : ℕ := 3

theorem choose_three_roles_from_eight_people : 
  (number_of_people * (number_of_people - 1) * (number_of_people - 2) = 336) := by
  sorry

end NUMINAMATH_CALUDE_choose_three_roles_from_eight_people_l3001_300124


namespace NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l3001_300157

/-- Represents the days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a 30-day month -/
def Month := Fin 30

/-- Returns the day of the week for a given day in the month, given the starting day -/
def dayOfWeek (startDay : DayOfWeek) (day : Month) : DayOfWeek :=
  sorry

/-- Counts the number of occurrences of a specific day in the month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem stating that there are no starting days that result in equal Tuesdays and Fridays -/
theorem no_equal_tuesdays_fridays :
  ∀ startDay : DayOfWeek,
    countDayOccurrences startDay DayOfWeek.Tuesday ≠
    countDayOccurrences startDay DayOfWeek.Friday :=
  sorry

end NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l3001_300157


namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l3001_300165

theorem three_digit_divisibility_by_seven :
  ∃ (start : ℕ), 
    (100 ≤ start) ∧ 
    (start + 127 ≤ 999) ∧ 
    (∀ k : ℕ, k < 128 → (start + k) % 7 = (start % 7)) ∧
    (start % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l3001_300165


namespace NUMINAMATH_CALUDE_symmetric_points_of_M_l3001_300108

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Original point M -/
def M : Point3D := ⟨1, -2, 3⟩

/-- Symmetric point with respect to xy-plane -/
def symmetricXY (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

/-- Symmetric point with respect to z-axis -/
def symmetricZ (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, p.z⟩

theorem symmetric_points_of_M :
  (symmetricXY M = ⟨1, -2, -3⟩) ∧ (symmetricZ M = ⟨-1, 2, 3⟩) := by sorry

end NUMINAMATH_CALUDE_symmetric_points_of_M_l3001_300108


namespace NUMINAMATH_CALUDE_valid_choices_count_l3001_300197

/-- The number of objects placed along a circle -/
def n : ℕ := 32

/-- The number of objects to be chosen -/
def k : ℕ := 3

/-- The number of ways to choose k objects from n objects -/
def total_ways : ℕ := n.choose k

/-- The number of pairs of adjacent objects -/
def adjacent_pairs : ℕ := n

/-- The number of pairs of diametrically opposite objects -/
def opposite_pairs : ℕ := n / 2

/-- The number of remaining objects after choosing two adjacent or opposite objects -/
def remaining_objects : ℕ := n - 4

/-- The theorem stating the number of valid ways to choose objects -/
theorem valid_choices_count : 
  total_ways - adjacent_pairs * remaining_objects - opposite_pairs * remaining_objects + n = 3648 := by
  sorry

end NUMINAMATH_CALUDE_valid_choices_count_l3001_300197


namespace NUMINAMATH_CALUDE_power_equality_l3001_300177

theorem power_equality : 32^5 * 4^5 = 2^35 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3001_300177


namespace NUMINAMATH_CALUDE_xyz_value_l3001_300168

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 20 * Real.rpow 2 (1/3))
  (hxz : x * z = 35 * Real.rpow 2 (1/3))
  (hyz : y * z = 14 * Real.rpow 2 (1/3)) :
  x * y * z = 140 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3001_300168


namespace NUMINAMATH_CALUDE_cookies_left_to_take_home_l3001_300196

def initial_cookies : ℕ := 120
def dozen : ℕ := 12
def morning_sales : ℕ := 3 * dozen
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16

theorem cookies_left_to_take_home : 
  initial_cookies - morning_sales - lunch_sales - afternoon_sales = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_to_take_home_l3001_300196


namespace NUMINAMATH_CALUDE_total_bill_sum_l3001_300139

-- Define the variables for each person's bill
variable (alice_bill : ℝ) (bob_bill : ℝ) (charlie_bill : ℝ)

-- Define the conditions
axiom alice_tip : 0.15 * alice_bill = 3
axiom bob_tip : 0.25 * bob_bill = 5
axiom charlie_tip : 0.20 * charlie_bill = 4

-- Theorem statement
theorem total_bill_sum :
  alice_bill + bob_bill + charlie_bill = 60 :=
sorry

end NUMINAMATH_CALUDE_total_bill_sum_l3001_300139


namespace NUMINAMATH_CALUDE_unique_prime_cube_l3001_300161

theorem unique_prime_cube : ∃! n : ℕ, ∃ p : ℕ,
  Prime p ∧ n = 2 * p + 1 ∧ ∃ m : ℕ, n = m^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_cube_l3001_300161


namespace NUMINAMATH_CALUDE_square_perimeter_is_48_l3001_300173

-- Define a square with side length 12
def square_side_length : ℝ := 12

-- Define the perimeter of a square
def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: The perimeter of the square with side length 12 cm is 48 cm
theorem square_perimeter_is_48 : 
  square_perimeter square_side_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_is_48_l3001_300173


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l3001_300164

/-- The number of ways to choose k items from n items -/
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of ways to arrange k items in k distinct positions -/
def arrangements (k : ℕ) : ℕ := Nat.factorial k

theorem flower_arrangement_count :
  let n : ℕ := 5  -- Total number of flower types
  let k : ℕ := 2  -- Number of flowers to pick
  (combinations n k) * (arrangements k) = 20 := by
  sorry

#eval (combinations 5 2) * (arrangements 2)  -- Should output 20

end NUMINAMATH_CALUDE_flower_arrangement_count_l3001_300164


namespace NUMINAMATH_CALUDE_second_player_wins_123_l3001_300112

/-- A game where players color points on a circle. -/
structure ColorGame where
  num_points : ℕ
  first_player : Bool
  
/-- The result of the game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- Determine the winner of the color game. -/
def winner (game : ColorGame) : GameResult :=
  if game.num_points % 2 = 1 then GameResult.SecondPlayerWins
  else GameResult.FirstPlayerWins

/-- The main theorem stating that the second player wins in a game with 123 points. -/
theorem second_player_wins_123 :
  ∀ (game : ColorGame), game.num_points = 123 → winner game = GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_second_player_wins_123_l3001_300112


namespace NUMINAMATH_CALUDE_marble_problem_l3001_300116

def initial_red_marbles : ℕ := 33
def initial_green_marbles : ℕ := 22

theorem marble_problem :
  (initial_red_marbles : ℚ) / initial_green_marbles = 3 / 2 ∧
  (initial_red_marbles - 18 : ℚ) / (initial_green_marbles + 15) = 2 / 5 :=
by
  sorry

#check marble_problem

end NUMINAMATH_CALUDE_marble_problem_l3001_300116


namespace NUMINAMATH_CALUDE_simplify_expression_l3001_300145

theorem simplify_expression (y : ℝ) : 5*y + 8*y + 2*y + 7 = 15*y + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3001_300145


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3001_300133

theorem pasta_preference_ratio :
  let total_students : ℕ := 1000
  let lasagna_pref : ℕ := 300
  let manicotti_pref : ℕ := 200
  let ravioli_pref : ℕ := 150
  let spaghetti_pref : ℕ := 270
  let fettuccine_pref : ℕ := 80
  (spaghetti_pref : ℚ) / (manicotti_pref : ℚ) = 27 / 20 :=
by sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3001_300133


namespace NUMINAMATH_CALUDE_loan_interest_calculation_l3001_300155

/-- Calculates simple interest given principal, rate, and time --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem: The simple interest on a loan of $1200 at 3% for 3 years is $108 --/
theorem loan_interest_calculation :
  let principal : ℝ := 1200
  let rate : ℝ := 0.03
  let time : ℝ := 3
  simple_interest principal rate time = 108 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_calculation_l3001_300155


namespace NUMINAMATH_CALUDE_erased_number_l3001_300199

theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) 
  (h2 : (a - 4) + (a - 3) + (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) + (a + 4) - (a + b) = 1703) : 
  a + b = 214 := by
sorry

end NUMINAMATH_CALUDE_erased_number_l3001_300199


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3001_300151

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x ≤ -1}

-- Theorem statement
theorem intersection_A_complement_B : 
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3001_300151


namespace NUMINAMATH_CALUDE_class_cans_collection_l3001_300188

/-- Calculates the total number of cans collected by a class given specific conditions -/
def totalCansCollected (totalStudents : ℕ) (cansPerHalf : ℕ) (nonCollectingStudents : ℕ) 
  (remainingStudents : ℕ) (cansPerRemaining : ℕ) : ℕ :=
  let halfStudents := totalStudents / 2
  let cansFromHalf := halfStudents * cansPerHalf
  let cansFromRemaining := remainingStudents * cansPerRemaining
  cansFromHalf + cansFromRemaining

/-- Theorem stating that under given conditions, the class collects 232 cans in total -/
theorem class_cans_collection : 
  totalCansCollected 30 12 2 13 4 = 232 := by
  sorry

end NUMINAMATH_CALUDE_class_cans_collection_l3001_300188


namespace NUMINAMATH_CALUDE_max_min_difference_l3001_300152

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * abs (x - a)

-- State the theorem
theorem max_min_difference (a : ℝ) (h : a ≥ 2) :
  ∃ M m : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ M ∧ m ≤ f a x) ∧ M - m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l3001_300152


namespace NUMINAMATH_CALUDE_sin_690_degrees_l3001_300101

theorem sin_690_degrees : Real.sin (690 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l3001_300101


namespace NUMINAMATH_CALUDE_stock_price_return_l3001_300140

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_increase := initial_price * 1.25
  let price_after_decrease := price_after_increase * 0.8
  price_after_decrease = initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_return_l3001_300140


namespace NUMINAMATH_CALUDE_new_drive_usage_percentage_l3001_300148

def initial_free_space : ℝ := 324
def initial_used_space : ℝ := 850
def initial_document_size : ℝ := 180
def initial_photo_size : ℝ := 380
def initial_video_size : ℝ := 290
def document_compression_ratio : ℝ := 0.05
def photo_compression_ratio : ℝ := 0.12
def video_compression_ratio : ℝ := 0.20
def deleted_photo_size : ℝ := 65.9
def deleted_video_size : ℝ := 98.1
def added_document_size : ℝ := 20.4
def added_photo_size : ℝ := 37.6
def new_drive_size : ℝ := 1500

theorem new_drive_usage_percentage (ε : ℝ) (hε : ε > 0) :
  ∃ (percentage : ℝ),
    abs (percentage - 43.56) < ε ∧
    percentage = 
      (((initial_document_size + added_document_size) * (1 - document_compression_ratio) +
        (initial_photo_size - deleted_photo_size + added_photo_size) * (1 - photo_compression_ratio) +
        (initial_video_size - deleted_video_size) * (1 - video_compression_ratio)) /
       new_drive_size) * 100 :=
by sorry

end NUMINAMATH_CALUDE_new_drive_usage_percentage_l3001_300148


namespace NUMINAMATH_CALUDE_alcohol_percentage_problem_l3001_300170

/-- Proves that the initial alcohol percentage is 30% given the conditions of the problem -/
theorem alcohol_percentage_problem (initial_volume : ℝ) (added_alcohol : ℝ) (final_percentage : ℝ) :
  initial_volume = 6 →
  added_alcohol = 2.4 →
  final_percentage = 50 →
  (∃ initial_percentage : ℝ,
    initial_percentage * initial_volume / 100 + added_alcohol = 
    final_percentage * (initial_volume + added_alcohol) / 100 ∧
    initial_percentage = 30) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_problem_l3001_300170


namespace NUMINAMATH_CALUDE_comic_book_ratio_l3001_300113

/-- Given the initial number of comic books, the number bought, and the final number,
    prove that the ratio of comic books sold to initial comic books is 1/2. -/
theorem comic_book_ratio 
  (initial : ℕ) (bought : ℕ) (final : ℕ) 
  (h1 : initial = 14) 
  (h2 : bought = 6) 
  (h3 : final = 13) : 
  (initial - (final - bought)) / initial = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_comic_book_ratio_l3001_300113


namespace NUMINAMATH_CALUDE_chair_production_theorem_l3001_300156

/-- Represents the chair production scenario in a furniture factory -/
structure ChairProduction where
  workers : ℕ
  individual_rate : ℕ
  total_time : ℕ
  total_chairs : ℕ

/-- Calculates the frequency of producing an additional chair as a group -/
def group_chair_frequency (cp : ChairProduction) : ℚ :=
  cp.total_time / (cp.total_chairs - cp.workers * cp.individual_rate * cp.total_time)

/-- Theorem stating the group chair frequency for the given scenario -/
theorem chair_production_theorem (cp : ChairProduction) 
  (h1 : cp.workers = 3)
  (h2 : cp.individual_rate = 4)
  (h3 : cp.total_time = 6)
  (h4 : cp.total_chairs = 73) :
  group_chair_frequency cp = 6 := by
  sorry

#eval group_chair_frequency ⟨3, 4, 6, 73⟩

end NUMINAMATH_CALUDE_chair_production_theorem_l3001_300156


namespace NUMINAMATH_CALUDE_product_of_sums_is_even_l3001_300136

/-- A card with two numbers -/
structure Card where
  front : Nat
  back : Nat

/-- The set of all cards -/
def deck : Finset Card := sorry

/-- The theorem to prove -/
theorem product_of_sums_is_even :
  (∀ c ∈ deck, c.front ∈ Finset.range 100 ∧ c.back ∈ Finset.range 100) →
  deck.card = 99 →
  (Finset.range 100).card = deck.card + 1 →
  (∀ n ∈ Finset.range 100, (deck.filter (λ c => c.front = n)).card +
    (deck.filter (λ c => c.back = n)).card = 1) →
  Even ((deck.prod (λ c => c.front + c.back))) :=
by sorry

end NUMINAMATH_CALUDE_product_of_sums_is_even_l3001_300136


namespace NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l3001_300198

theorem smallest_angle_in_345_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  b = (4/3) * a →
  c = (5/3) * a →
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l3001_300198


namespace NUMINAMATH_CALUDE_max_value_theorem_l3001_300125

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 2*a*b*Real.sqrt 3 + 2*a*c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3001_300125


namespace NUMINAMATH_CALUDE_fruit_count_l3001_300144

/-- Given:
  1. If each bag contains 5 oranges and 7 apples, after packing all the apples, there will be 1 orange left.
  2. If each bag contains 9 oranges and 7 apples, after packing all the oranges, there will be 21 apples left.
Prove that the total number of oranges and apples is 85. -/
theorem fruit_count (oranges apples : ℕ) 
  (h1 : ∃ m : ℕ, oranges = 5 * m + 1 ∧ apples = 7 * m)
  (h2 : ∃ n : ℕ, oranges = 9 * n ∧ apples = 7 * n + 21) :
  oranges + apples = 85 := by
sorry

end NUMINAMATH_CALUDE_fruit_count_l3001_300144


namespace NUMINAMATH_CALUDE_coffee_mix_price_l3001_300135

theorem coffee_mix_price (P : ℝ) : 
  let price_second : ℝ := 2.45
  let total_weight : ℝ := 18
  let mix_price : ℝ := 2.30
  let weight_each : ℝ := 9
  (weight_each * P + weight_each * price_second = total_weight * mix_price) →
  P = 2.15 := by
sorry

end NUMINAMATH_CALUDE_coffee_mix_price_l3001_300135


namespace NUMINAMATH_CALUDE_vector_subtraction_l3001_300186

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := fun m ↦ (4, m)

theorem vector_subtraction (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (a.1 - (b m).1, a.2 - (b m).2) = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3001_300186


namespace NUMINAMATH_CALUDE_constant_a_value_l3001_300193

theorem constant_a_value (x y : ℝ) (a : ℝ) 
  (h1 : (a * x + 4 * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = 5 / 2) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_a_value_l3001_300193


namespace NUMINAMATH_CALUDE_ball_distribution_l3001_300159

theorem ball_distribution (n : ℕ) (k : ℕ) : 
  (∃ x y z : ℕ, x + y + z = n ∧ x ≥ 1 ∧ y ≥ 2 ∧ z ≥ 3) →
  (Nat.choose (n - 6 + k - 1) (k - 1) = 15) →
  (k = 3 ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_l3001_300159


namespace NUMINAMATH_CALUDE_last_two_digits_of_nine_to_2008_l3001_300169

theorem last_two_digits_of_nine_to_2008 : 9^2008 % 100 = 21 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_nine_to_2008_l3001_300169


namespace NUMINAMATH_CALUDE_chord_bisector_line_equation_l3001_300166

/-- Given an ellipse and a point inside it, this theorem proves the equation of the line
    on which the chord bisected by the point lies. -/
theorem chord_bisector_line_equation (x y : ℝ) :
  (x^2 / 16 + y^2 / 4 = 1) →  -- Ellipse equation
  (3^2 / 16 + 1^2 / 4 < 1) →  -- Point P(3,1) is inside the ellipse
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x^2 / 16 + y^2 / 4 = 1) ∧ 
    ((x + 3) / 2 = 3 ∧ (y + 1) / 2 = 1) → 
    y = m * x + b ∧ 
    3 * x + 4 * y - 13 = 0 :=
by sorry

end NUMINAMATH_CALUDE_chord_bisector_line_equation_l3001_300166


namespace NUMINAMATH_CALUDE_marble_jar_theorem_l3001_300181

/-- Represents a jar of marbles with orange, purple, and yellow colors. -/
structure MarbleJar where
  orange : ℝ
  purple : ℝ
  yellow : ℝ

/-- The total number of marbles in the jar. -/
def MarbleJar.total (jar : MarbleJar) : ℝ :=
  jar.orange + jar.purple + jar.yellow

/-- A jar satisfying the given conditions. -/
def specialJar : MarbleJar :=
  { orange := 0,  -- placeholder values
    purple := 0,
    yellow := 0 }

theorem marble_jar_theorem (jar : MarbleJar) :
  jar.purple + jar.yellow = 7 →
  jar.orange + jar.yellow = 5 →
  jar.orange + jar.purple = 9 →
  jar.total = 10.5 := by
  sorry

#check marble_jar_theorem

end NUMINAMATH_CALUDE_marble_jar_theorem_l3001_300181


namespace NUMINAMATH_CALUDE_airplane_seat_ratio_l3001_300184

theorem airplane_seat_ratio :
  ∀ (total_seats coach_seats first_class_seats k : ℕ),
    total_seats = 387 →
    coach_seats = 310 →
    coach_seats = k * first_class_seats + 2 →
    first_class_seats + coach_seats = total_seats →
    (coach_seats - 2) / first_class_seats = 4 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seat_ratio_l3001_300184


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3001_300111

structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)
  (PQ QR RS SP : ℝ)
  (PR : ℤ)

theorem quadrilateral_diagonal_length 
  (quad : Quadrilateral) 
  (h1 : quad.PQ = 7)
  (h2 : quad.QR = 15)
  (h3 : quad.RS = 7)
  (h4 : quad.SP = 8) :
  9 ≤ quad.PR ∧ quad.PR ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3001_300111


namespace NUMINAMATH_CALUDE_employees_abroad_l3001_300105

theorem employees_abroad (total : ℕ) (fraction : ℚ) (abroad : ℕ) : 
  total = 450 → fraction = 0.06 → abroad = (total : ℚ) * fraction → abroad = 27 := by
sorry

end NUMINAMATH_CALUDE_employees_abroad_l3001_300105


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_2_a_value_when_union_equals_A_l3001_300185

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a+1)*x + a = 0}

-- Part 1
theorem intersection_complement_when_a_2 :
  A ∩ (Set.univ \ B 2) = {-3} := by sorry

-- Part 2
theorem a_value_when_union_equals_A :
  ∀ a : ℝ, A ∪ B a = A → a = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_2_a_value_when_union_equals_A_l3001_300185


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3001_300137

theorem polynomial_simplification (x : ℝ) :
  (x + 1)^4 - 4*(x + 1)^3 + 6*(x + 1)^2 - 4*(x + 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3001_300137


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3001_300119

/-- A line passing through a point and perpendicular to another line -/
def perpendicular_line (x₀ y₀ a b c : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ 
    (∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (a * x + b * y + c = 0)) ∧
    k * (a / b) = -1

theorem perpendicular_line_equation :
  perpendicular_line 1 3 2 (-5) 1 →
  ∀ x y : ℝ, 5 * x + 2 * y - 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3001_300119


namespace NUMINAMATH_CALUDE_bottle_caps_problem_l3001_300171

theorem bottle_caps_problem (katherine_initial : ℕ) (hippopotamus_eaten : ℕ) : 
  katherine_initial = 34 →
  hippopotamus_eaten = 8 →
  (katherine_initial / 2 : ℕ) - hippopotamus_eaten = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_problem_l3001_300171


namespace NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l3001_300189

theorem not_prime_n4_plus_n2_plus_1 (n : ℤ) (h : n ≥ 2) :
  ¬(Nat.Prime (n^4 + n^2 + 1).natAbs) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l3001_300189


namespace NUMINAMATH_CALUDE_roots_expression_equals_one_l3001_300121

theorem roots_expression_equals_one (α β γ δ : ℝ) : 
  (α^2 - 2*α + 1 = 0) → 
  (β^2 - 2*β + 1 = 0) → 
  (γ^2 - 3*γ + 1 = 0) → 
  (δ^2 - 3*δ + 1 = 0) → 
  (α - γ)^2 * (β - δ)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_expression_equals_one_l3001_300121


namespace NUMINAMATH_CALUDE_rick_ironing_theorem_l3001_300103

/-- The number of dress shirts Rick can iron in one hour -/
def shirts_per_hour : ℕ := 4

/-- The number of dress pants Rick can iron in one hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick has ironed -/
def total_pieces : ℕ := shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing_theorem : total_pieces = 27 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironing_theorem_l3001_300103


namespace NUMINAMATH_CALUDE_broken_stick_pairing_probability_l3001_300114

/-- The number of sticks --/
def n : ℕ := 5

/-- The probability of pairing each long part with a short part when rearranging broken sticks --/
theorem broken_stick_pairing_probability :
  (2^n : ℚ) / (Nat.choose (2*n) n : ℚ) = 8/63 := by sorry

end NUMINAMATH_CALUDE_broken_stick_pairing_probability_l3001_300114


namespace NUMINAMATH_CALUDE_shirt_cost_l3001_300120

theorem shirt_cost (initial_amount : ℕ) (socks_cost : ℕ) (amount_left : ℕ) :
  initial_amount = 100 →
  socks_cost = 11 →
  amount_left = 65 →
  initial_amount - amount_left - socks_cost = 24 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l3001_300120


namespace NUMINAMATH_CALUDE_flour_added_l3001_300134

/-- Given that Mary already put in 8 cups of flour and the recipe requires 10 cups in total,
    prove that she added 2 more cups of flour. -/
theorem flour_added (initial_flour : ℕ) (total_flour : ℕ) (h1 : initial_flour = 8) (h2 : total_flour = 10) :
  total_flour - initial_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_added_l3001_300134


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l3001_300158

theorem quadratic_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l3001_300158


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l3001_300176

theorem amusement_park_tickets :
  ∀ (a b c : ℕ),
  a + b + c = 85 →
  7 * a + 4 * b + 2 * c = 500 →
  a = b + 31 →
  a = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l3001_300176


namespace NUMINAMATH_CALUDE_consecutive_points_segment_length_l3001_300143

/-- Given five consecutive points on a straight line, prove the length of a specific segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Define points as real numbers representing their positions on the line
  (consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Ensure points are consecutive
  (bc_cd : c - b = 2 * (d - c)) -- bc = 2 cd
  (de_length : e - d = 4) -- de = 4
  (ab_length : b - a = 5) -- ab = 5
  (ae_length : e - a = 18) -- ae = 18
  : c - a = 11 := by -- Prove that ac = 11
  sorry

end NUMINAMATH_CALUDE_consecutive_points_segment_length_l3001_300143
