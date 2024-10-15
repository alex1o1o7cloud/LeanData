import Mathlib

namespace NUMINAMATH_CALUDE_direction_525_to_527_l617_61710

/-- Represents the directions of movement -/
inductive Direction
| Right
| Up
| Left
| Down
| Diagonal

/-- Defines the cyclic pattern of directions -/
def directionPattern : Fin 5 → Direction
| 0 => Direction.Right
| 1 => Direction.Up
| 2 => Direction.Left
| 3 => Direction.Down
| 4 => Direction.Diagonal

/-- Returns the direction for a given point number -/
def directionAtPoint (n : Nat) : Direction :=
  directionPattern (n % 5)

/-- Theorem: The sequence of directions from point 525 to 527 is Right, Up -/
theorem direction_525_to_527 :
  (directionAtPoint 525, directionAtPoint 526) = (Direction.Right, Direction.Up) := by
  sorry

#check direction_525_to_527

end NUMINAMATH_CALUDE_direction_525_to_527_l617_61710


namespace NUMINAMATH_CALUDE_solution_set_inequality_l617_61774

theorem solution_set_inequality (x : ℝ) : x^2 - 3*x > 0 ↔ x < 0 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l617_61774


namespace NUMINAMATH_CALUDE_range_of_a_l617_61727

-- Define the sets A and B
def A : Set ℝ := {x | 4 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ 2*a - 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = A → 3 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l617_61727


namespace NUMINAMATH_CALUDE_operations_to_zero_l617_61734

/-- The initial value before operations begin -/
def initial_value : ℕ := 2100

/-- The amount subtracted in each operation -/
def subtract_amount : ℕ := 50

/-- The amount added in each operation -/
def add_amount : ℕ := 20

/-- The effective change per operation -/
def effective_change : ℤ := (subtract_amount : ℤ) - (add_amount : ℤ)

/-- The number of operations needed to reach 0 -/
def num_operations : ℕ := initial_value / (effective_change.natAbs)

theorem operations_to_zero : num_operations = 70 := by
  sorry

end NUMINAMATH_CALUDE_operations_to_zero_l617_61734


namespace NUMINAMATH_CALUDE_ribbon_triangle_to_pentagon_l617_61758

theorem ribbon_triangle_to_pentagon (triangle_side : ℝ) (pentagon_side : ℝ) : 
  triangle_side = 20 / 9 → pentagon_side = (3 * triangle_side) / 5 → pentagon_side = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_triangle_to_pentagon_l617_61758


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l617_61794

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set condition
def solution_set (a b c : ℝ) : Set ℝ := {x : ℝ | x ≤ -2 ∨ x ≥ 3}

-- Define the theorem
theorem quadratic_inequalities 
  (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x ≤ 0) :
  (a < 0) ∧ 
  ({x : ℝ | a * x + c > 0} = {x : ℝ | x < 6}) ∧
  ({x : ℝ | c * x^2 + b * x + a < 0} = {x : ℝ | -1/2 < x ∧ x < 1/3}) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l617_61794


namespace NUMINAMATH_CALUDE_medication_mixture_volume_l617_61730

/-- Given a mixture of two medications A and B, where:
    - Medication A contains 40% pain killer
    - Medication B contains 20% pain killer
    - The patient receives exactly 215 milliliters of pain killer daily
    - There are 425 milliliters of medication B in the mixture
    Prove that the total volume of the mixture given to the patient daily is 750 milliliters. -/
theorem medication_mixture_volume :
  let medication_a_percentage : ℝ := 0.40
  let medication_b_percentage : ℝ := 0.20
  let total_painkiller : ℝ := 215
  let medication_b_volume : ℝ := 425
  let total_mixture_volume : ℝ := (total_painkiller - medication_b_percentage * medication_b_volume) / 
                                  (medication_a_percentage - medication_b_percentage) + medication_b_volume
  total_mixture_volume = 750 :=
by sorry

end NUMINAMATH_CALUDE_medication_mixture_volume_l617_61730


namespace NUMINAMATH_CALUDE_divisor_problem_l617_61756

theorem divisor_problem (d : ℕ) (z : ℤ) 
  (h1 : d > 0)
  (h2 : ∃ k : ℤ, (z + 3) / d = k)
  (h3 : ∃ m : ℤ, z = m * d + 6) :
  d = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l617_61756


namespace NUMINAMATH_CALUDE_angle_FHP_equals_angle_BAC_l617_61763

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define that ABC is an acute triangle
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define that BC > CA
def BC_greater_than_CA (A B C : ℝ × ℝ) : Prop := sorry

-- Define the circumcenter O
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the orthocenter H
def orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the foot F of the altitude from C to AB
def altitude_foot (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the point P
def point_P (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_FHP_equals_angle_BAC
  (h1 : is_acute_triangle A B C)
  (h2 : BC_greater_than_CA A B C)
  (O : ℝ × ℝ) (hO : O = circumcenter A B C)
  (H : ℝ × ℝ) (hH : H = orthocenter A B C)
  (F : ℝ × ℝ) (hF : F = altitude_foot A B C)
  (P : ℝ × ℝ) (hP : P = point_P A B C) :
  angle (F - H) (P - H) = angle (B - A) (C - A) :=
sorry

end NUMINAMATH_CALUDE_angle_FHP_equals_angle_BAC_l617_61763


namespace NUMINAMATH_CALUDE_taehyungs_calculation_l617_61767

theorem taehyungs_calculation : ∃ x : ℝ, (x / 5 = 30) ∧ (8 * x = 1200) := by
  sorry

end NUMINAMATH_CALUDE_taehyungs_calculation_l617_61767


namespace NUMINAMATH_CALUDE_subtract_sum_digits_100_times_is_zero_l617_61713

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Computes the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Performs one iteration of subtracting the sum of digits -/
def subtract_sum_of_digits (n : ThreeDigitNumber) : ℕ := 
  n.value - sum_of_digits n.value

/-- Performs the subtraction process n times -/
def iterate_subtraction (n : ThreeDigitNumber) (iterations : ℕ) : ℕ := sorry

/-- Theorem: After 100 iterations of subtracting the sum of digits from any three-digit number, the result is zero -/
theorem subtract_sum_digits_100_times_is_zero (n : ThreeDigitNumber) : 
  iterate_subtraction n 100 = 0 := by sorry

end NUMINAMATH_CALUDE_subtract_sum_digits_100_times_is_zero_l617_61713


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l617_61789

theorem no_integer_solutions_for_equation : 
  ¬ ∃ (x y : ℤ), (2 : ℝ) ^ (2 * x) - (3 : ℝ) ^ (2 * y) = 35 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l617_61789


namespace NUMINAMATH_CALUDE_mean_temperature_is_79_9_l617_61786

def temperatures : List ℝ := [75, 74, 76, 77, 80, 81, 83, 85, 83, 85]

theorem mean_temperature_is_79_9 :
  (temperatures.sum / temperatures.length : ℝ) = 79.9 := by
sorry

end NUMINAMATH_CALUDE_mean_temperature_is_79_9_l617_61786


namespace NUMINAMATH_CALUDE_total_students_l617_61790

theorem total_students (scavenger_hunting : ℕ) (skiing : ℕ) : 
  scavenger_hunting = 4000 → 
  skiing = 2 * scavenger_hunting → 
  scavenger_hunting + skiing = 12000 := by
sorry

end NUMINAMATH_CALUDE_total_students_l617_61790


namespace NUMINAMATH_CALUDE_road_trip_distance_l617_61739

theorem road_trip_distance (D : ℝ) : 
  (D / 3 + (D - D / 3) / 4 + 300 = D) → D = 600 := by sorry

end NUMINAMATH_CALUDE_road_trip_distance_l617_61739


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_even_integers_divisible_by_three_l617_61783

theorem sum_of_three_consecutive_even_integers_divisible_by_three (n : ℤ) (h : Even n) :
  ∃ k : ℤ, n + (n + 2) + (n + 4) = 3 * k :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_even_integers_divisible_by_three_l617_61783


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l617_61723

theorem repeating_decimal_problem (n : ℕ) :
  n < 1000 ∧
  (∃ (a b c d e f : ℕ), (1 : ℚ) / n = (a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) / 999999) ∧
  (∃ (w x y z : ℕ), (1 : ℚ) / (n + 5) = (w * 1000 + x * 100 + y * 10 + z) / 9999) →
  151 ≤ n ∧ n ≤ 300 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l617_61723


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l617_61744

/-- Represents the height reached by a monkey with a specific climbing pattern -/
def monkey_climb_height (climb_rate : ℕ) (slip_rate : ℕ) (total_time : ℕ) : ℕ :=
  let full_cycles := total_time / 2
  let remainder := total_time % 2
  full_cycles * (climb_rate - slip_rate) + remainder * climb_rate

/-- Theorem stating that given the specific climbing pattern and time, the monkey reaches 60 meters -/
theorem monkey_climb_theorem (climb_rate slip_rate total_time : ℕ) 
  (h_climb : climb_rate = 6)
  (h_slip : slip_rate = 3)
  (h_time : total_time = 37) :
  monkey_climb_height climb_rate slip_rate total_time = 60 := by
  sorry

#eval monkey_climb_height 6 3 37

end NUMINAMATH_CALUDE_monkey_climb_theorem_l617_61744


namespace NUMINAMATH_CALUDE_invisible_dots_count_l617_61750

/-- The sum of numbers on a single six-sided die -/
def die_sum : Nat := 21

/-- The total number of dots on four dice -/
def total_dots : Nat := 4 * die_sum

/-- The sum of visible numbers on the dice -/
def visible_sum : Nat := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6

/-- The number of dots not visible on the dice -/
def invisible_dots : Nat := total_dots - visible_sum

theorem invisible_dots_count : invisible_dots = 54 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l617_61750


namespace NUMINAMATH_CALUDE_octahedron_inscribed_in_cube_l617_61751

/-- A cube in 3D space -/
structure Cube where
  edge_length : ℝ
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- An octahedron in 3D space -/
structure Octahedron where
  vertices : Fin 6 → ℝ × ℝ × ℝ

/-- Predicate to check if a point lies on an edge of a cube -/
def point_on_cube_edge (c : Cube) (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (i j : Fin 8), i ≠ j ∧ 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
      p = ((1 - t) • (c.vertices i) + t • (c.vertices j))

/-- Theorem stating that an octahedron can be inscribed in a cube 
    with its vertices on the cube's edges -/
theorem octahedron_inscribed_in_cube : 
  ∃ (c : Cube) (o : Octahedron), 
    ∀ (i : Fin 6), point_on_cube_edge c (o.vertices i) :=
  sorry

end NUMINAMATH_CALUDE_octahedron_inscribed_in_cube_l617_61751


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l617_61708

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l617_61708


namespace NUMINAMATH_CALUDE_inequality_proof_l617_61770

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) + 
  (2*b + c + a)^2 / (2*b^2 + (c + a)^2) + 
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l617_61770


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l617_61728

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x| - 2 - |-1| = 2) ↔ (x = 5 ∨ x = -5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l617_61728


namespace NUMINAMATH_CALUDE_final_weight_is_16_l617_61779

/-- The weight of the box after each step of adding ingredients --/
structure BoxWeight where
  initial : ℝ
  afterBrownies : ℝ
  afterMoreJellyBeans : ℝ
  final : ℝ

/-- The process of creating the care package --/
def createCarePackage : BoxWeight :=
  { initial := 2,
    afterBrownies := 2 * 3,
    afterMoreJellyBeans := 2 * 3 + 2,
    final := (2 * 3 + 2) * 2 }

/-- The theorem stating the final weight of the care package --/
theorem final_weight_is_16 :
  (createCarePackage.final : ℝ) = 16 := by sorry

end NUMINAMATH_CALUDE_final_weight_is_16_l617_61779


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_2023_l617_61778

/-- A function that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The theorem stating that there are no two-digit factors of 2023 -/
theorem no_two_digit_factors_of_2023 : 
  ¬ ∃ (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ a * b = 2023 := by
  sorry

#check no_two_digit_factors_of_2023

end NUMINAMATH_CALUDE_no_two_digit_factors_of_2023_l617_61778


namespace NUMINAMATH_CALUDE_roots_have_nonzero_imag_l617_61726

/-- The complex number i such that i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation 5z^2 + 2iz - m = 0 -/
def equation (z : ℂ) (m : ℝ) : Prop := 5 * z^2 + 2 * i * z - m = 0

/-- A complex number has a non-zero imaginary part -/
def has_nonzero_imag (z : ℂ) : Prop := z.im ≠ 0

/-- Both roots of the equation have non-zero imaginary parts for all real m -/
theorem roots_have_nonzero_imag :
  ∀ m : ℝ, ∀ z : ℂ, equation z m → has_nonzero_imag z :=
sorry

end NUMINAMATH_CALUDE_roots_have_nonzero_imag_l617_61726


namespace NUMINAMATH_CALUDE_utility_graph_non_planar_l617_61752

/-- A graph representing the connection of houses to utilities -/
structure UtilityGraph where
  houses : Finset (Fin 3)
  utilities : Finset (Fin 3)
  connections : Set (Fin 3 × Fin 3)

/-- The utility graph is complete bipartite -/
def is_complete_bipartite (g : UtilityGraph) : Prop :=
  ∀ h ∈ g.houses, ∀ u ∈ g.utilities, (h, u) ∈ g.connections

/-- A graph is planar if it can be drawn on a plane without edge crossings -/
def is_planar (g : UtilityGraph) : Prop :=
  sorry -- Definition of planarity

/-- The theorem stating that the utility graph is non-planar -/
theorem utility_graph_non_planar (g : UtilityGraph) 
  (h1 : g.houses.card = 3) 
  (h2 : g.utilities.card = 3) 
  (h3 : is_complete_bipartite g) : 
  ¬ is_planar g :=
sorry

end NUMINAMATH_CALUDE_utility_graph_non_planar_l617_61752


namespace NUMINAMATH_CALUDE_inequality_proof_l617_61797

theorem inequality_proof (A B C a b c r : ℝ) 
  (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  (A + a + B + b) / (A + a + B + b + c + r) + 
  (B + b + C + c) / (B + b + C + c + a + r) > 
  (C + c + A + a) / (C + c + A + a + b + r) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l617_61797


namespace NUMINAMATH_CALUDE_onion_basket_change_l617_61757

theorem onion_basket_change (initial : ℝ) : 
  let added_by_sara := 4.5
  let removed_by_sally := 5.25
  let added_by_fred := 9.75
  (initial + added_by_sara - removed_by_sally + added_by_fred) - initial = 9 :=
by sorry

end NUMINAMATH_CALUDE_onion_basket_change_l617_61757


namespace NUMINAMATH_CALUDE_buratino_number_problem_l617_61761

theorem buratino_number_problem (x : ℚ) : 4 * x + 15 = 15 * x + 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_buratino_number_problem_l617_61761


namespace NUMINAMATH_CALUDE_rose_sale_earnings_l617_61712

theorem rose_sale_earnings :
  ∀ (price : ℕ) (initial : ℕ) (remaining : ℕ),
    price = 7 →
    initial = 9 →
    remaining = 4 →
    (initial - remaining) * price = 35 :=
by sorry

end NUMINAMATH_CALUDE_rose_sale_earnings_l617_61712


namespace NUMINAMATH_CALUDE_min_surface_area_large_solid_l617_61771

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surface_area (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the smaller rectangular solids -/
def small_solid : Dimensions :=
  { length := 3, width := 4, height := 5 }

/-- The number of smaller rectangular solids -/
def num_small_solids : ℕ := 24

/-- Theorem stating the minimum surface area of the large rectangular solid -/
theorem min_surface_area_large_solid :
  ∃ (d : Dimensions), surface_area d = 788 ∧
  (∀ (d' : Dimensions), surface_area d' ≥ surface_area d) := by
  sorry


end NUMINAMATH_CALUDE_min_surface_area_large_solid_l617_61771


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainder_one_l617_61748

theorem smallest_positive_integer_with_remainder_one : ∃ m : ℕ,
  m > 1 ∧
  m % 3 = 1 ∧
  m % 5 = 1 ∧
  m % 7 = 1 ∧
  (∀ n : ℕ, n > 1 → n % 3 = 1 → n % 5 = 1 → n % 7 = 1 → m ≤ n) ∧
  m = 106 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainder_one_l617_61748


namespace NUMINAMATH_CALUDE_first_expression_value_l617_61795

theorem first_expression_value (a : ℝ) (E : ℝ) : 
  a = 26 → (E + (3 * a - 8)) / 2 = 69 → E = 68 := by sorry

end NUMINAMATH_CALUDE_first_expression_value_l617_61795


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l617_61746

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y r : ℝ) : Prop := (x+4)^2 + (y-3)^2 = r^2

-- Define the condition of external tangency
def externally_tangent (r : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y r ∧
  (∀ (x' y' : ℝ), circle1 x' y' → circle2 x' y' r → (x = x' ∧ y = y'))

-- Theorem statement
theorem tangent_circles_radius :
  ∀ r : ℝ, externally_tangent r → (r = 3 ∨ r = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l617_61746


namespace NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_one_l617_61703

theorem sufficient_condition_for_product_greater_than_one :
  ∀ (a b : ℝ), a > 1 ∧ b > 1 → a * b > 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_one_l617_61703


namespace NUMINAMATH_CALUDE_f_equals_f_inv_at_two_point_five_l617_61759

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 5 * x - 3

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (5 + Real.sqrt (8 * x + 49)) / 4

-- Theorem statement
theorem f_equals_f_inv_at_two_point_five :
  f 2.5 = f_inv 2.5 := by sorry

end NUMINAMATH_CALUDE_f_equals_f_inv_at_two_point_five_l617_61759


namespace NUMINAMATH_CALUDE_quadrilateral_relation_l617_61736

/-- Given four points A, B, C, D in a plane satisfying certain conditions,
    prove that CD = 12 / AB -/
theorem quadrilateral_relation (A B C D : ℝ × ℝ) :
  (∀ (t : ℝ), ‖A - D‖ = 2 ∧ ‖B - C‖ = 2) →
  (∀ (t : ℝ), ‖A - C‖ = 4 ∧ ‖B - D‖ = 4) →
  (∃ (P : ℝ × ℝ), ∃ (s t : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ t ∧ t ≤ 1 ∧
    P = (1 - s) • A + s • C ∧ P = (1 - t) • B + t • D) →
  ‖C - D‖ = 12 / ‖A - B‖ :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_relation_l617_61736


namespace NUMINAMATH_CALUDE_rat_digging_difference_l617_61784

/-- The distance dug by the large rat after n days -/
def large_rat_distance (n : ℕ) : ℚ := 2^n - 1

/-- The distance dug by the small rat after n days -/
def small_rat_distance (n : ℕ) : ℚ := 2 - 1 / 2^(n-1)

/-- The difference in distance dug between the large and small rat after n days -/
def distance_difference (n : ℕ) : ℚ := large_rat_distance n - small_rat_distance n

theorem rat_digging_difference :
  distance_difference 5 = 29 / 16 := by sorry

end NUMINAMATH_CALUDE_rat_digging_difference_l617_61784


namespace NUMINAMATH_CALUDE_salary_change_l617_61753

theorem salary_change (original_salary : ℝ) (h : original_salary > 0) :
  let increased_salary := original_salary * 1.15
  let final_salary := increased_salary * 0.85
  let net_change := (final_salary - original_salary) / original_salary
  net_change = -0.0225 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l617_61753


namespace NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l617_61780

theorem mrs_hilt_pizza_slices : ∀ (num_pizzas slices_per_pizza : ℕ),
  num_pizzas = 2 →
  slices_per_pizza = 8 →
  num_pizzas * slices_per_pizza = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l617_61780


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_l617_61715

theorem tangent_slope_at_pi (f : ℝ → ℝ) (h : f = λ x => 2*x + Real.sin x) :
  HasDerivAt f 1 π := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_l617_61715


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_main_theorem_l617_61775

/-- Theorem: Speed of a boat in still water
Given:
- The rate of current is 4 km/hr
- The boat travels downstream for 44 minutes
- The distance travelled downstream is 33.733333333333334 km
Prove: The speed of the boat in still water is 42.09090909090909 km/hr
-/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (travel_time_minutes : ℝ) 
  (distance_downstream : ℝ) : ℝ :=
  let travel_time_hours := travel_time_minutes / 60
  let downstream_speed := (distance_downstream / travel_time_hours) - current_speed
  downstream_speed

/-- Main theorem application -/
theorem main_theorem : 
  boat_speed_in_still_water 4 44 33.733333333333334 = 42.09090909090909 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_main_theorem_l617_61775


namespace NUMINAMATH_CALUDE_total_age_l617_61792

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 6 years old
Prove that the total of their ages is 17 years. -/
theorem total_age (a b c : ℕ) : 
  b = 6 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 17 := by
sorry

end NUMINAMATH_CALUDE_total_age_l617_61792


namespace NUMINAMATH_CALUDE_nancy_soap_purchase_l617_61777

/-- Calculates the total number of soap bars Nancy bought from three brands. -/
def total_soap_bars (brand_a_packs : ℕ) (brand_a_bars_per_pack : ℕ)
                    (brand_b_packs : ℕ) (brand_b_bars_per_pack : ℕ)
                    (brand_c_packs : ℕ) (brand_c_bars_per_pack : ℕ)
                    (brand_c_free_pack_bars : ℕ) : ℕ :=
  brand_a_packs * brand_a_bars_per_pack +
  brand_b_packs * brand_b_bars_per_pack +
  brand_c_packs * brand_c_bars_per_pack +
  brand_c_free_pack_bars

theorem nancy_soap_purchase :
  total_soap_bars 4 3 3 5 2 6 4 = 43 := by
  sorry

end NUMINAMATH_CALUDE_nancy_soap_purchase_l617_61777


namespace NUMINAMATH_CALUDE_gails_wallet_total_l617_61764

/-- Represents the count of bills of a specific denomination in Gail's wallet. -/
structure BillCount where
  fives : Nat
  tens : Nat
  twenties : Nat

/-- Calculates the total amount of money in Gail's wallet given the bill counts. -/
def totalMoney (bills : BillCount) : Nat :=
  5 * bills.fives + 10 * bills.tens + 20 * bills.twenties

/-- Theorem stating that the total amount of money in Gail's wallet is $100. -/
theorem gails_wallet_total :
  ∃ (bills : BillCount),
    bills.fives = 4 ∧
    bills.tens = 2 ∧
    bills.twenties = 3 ∧
    totalMoney bills = 100 := by
  sorry

end NUMINAMATH_CALUDE_gails_wallet_total_l617_61764


namespace NUMINAMATH_CALUDE_kira_song_memory_space_l617_61776

/-- Calculates the total memory space occupied by downloaded songs -/
def total_memory_space (morning_songs : ℕ) (afternoon_songs : ℕ) (night_songs : ℕ) (song_size : ℕ) : ℕ :=
  (morning_songs + afternoon_songs + night_songs) * song_size

/-- Proves that the total memory space occupied by Kira's downloaded songs is 140 MB -/
theorem kira_song_memory_space :
  total_memory_space 10 15 3 5 = 140 := by
sorry

end NUMINAMATH_CALUDE_kira_song_memory_space_l617_61776


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l617_61700

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 7 * x₁ + 2 = 0) → 
  (5 * x₂^2 - 7 * x₂ + 2 = 0) → 
  (x₁^2 + x₂^2 = 29/25) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l617_61700


namespace NUMINAMATH_CALUDE_intersection_equals_closed_interval_l617_61716

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ -1}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the closed interval [-1, 3]
def closedInterval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_equals_closed_interval : M ∩ N = closedInterval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_closed_interval_l617_61716


namespace NUMINAMATH_CALUDE_candy_count_l617_61729

/-- The number of candy pieces caught by Tabitha, Stan, Julie, and Carlos -/
def total_candy (tabitha stan julie carlos : ℕ) : ℕ :=
  tabitha + stan + julie + carlos

/-- Theorem stating the total number of candy pieces caught by the friends -/
theorem candy_count : ∃ (julie carlos : ℕ),
  let tabitha := 22
  let stan := 13
  julie = tabitha / 2 ∧
  carlos = stan * 2 ∧
  total_candy tabitha stan julie carlos = 72 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l617_61729


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l617_61733

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l617_61733


namespace NUMINAMATH_CALUDE_system_solution_ratio_l617_61714

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  d ≠ 0 →
  8 * x - 6 * y = c →
  10 * y - 15 * x = d →
  c / d = -2 / 5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l617_61714


namespace NUMINAMATH_CALUDE_drape_cost_calculation_l617_61701

/-- The cost of window treatments for a house with the given conditions. -/
def window_treatment_cost (num_windows : ℕ) (sheer_cost drape_cost total_cost : ℚ) : Prop :=
  num_windows * (sheer_cost + drape_cost) = total_cost

/-- The theorem stating the cost of a pair of drapes given the conditions. -/
theorem drape_cost_calculation :
  ∃ (drape_cost : ℚ),
    window_treatment_cost 3 40 drape_cost 300 ∧ drape_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_drape_cost_calculation_l617_61701


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l617_61766

/-- Given that Kayla and Kylie picked a total of 200 apples, and Kayla picked 40 apples,
    prove that the ratio of apples Kayla picked to apples Kylie picked is 1:4. -/
theorem apple_picking_ratio :
  ∀ (total_apples kayla_apples : ℕ),
    total_apples = 200 →
    kayla_apples = 40 →
    ∃ (kylie_apples : ℕ),
      kylie_apples = total_apples - kayla_apples ∧
      kayla_apples * 4 = kylie_apples :=
by
  sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l617_61766


namespace NUMINAMATH_CALUDE_limit_cubic_fraction_l617_61732

theorem limit_cubic_fraction : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((x^3 - 1) / (x - 1)) - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_cubic_fraction_l617_61732


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l617_61741

theorem fractional_equation_solution :
  ∀ x : ℚ, x ≠ 0 → x ≠ 1 → (3 / (x - 1) = 1 / x) ↔ (x = -1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l617_61741


namespace NUMINAMATH_CALUDE_three_hundredth_non_square_l617_61781

/-- The count of perfect squares less than or equal to a given number -/
def countSquaresUpTo (n : ℕ) : ℕ := (n.sqrt : ℕ)

/-- The nth term of the sequence of non-square positive integers -/
def nthNonSquare (n : ℕ) : ℕ :=
  n + countSquaresUpTo n

theorem three_hundredth_non_square : nthNonSquare 300 = 318 := by
  sorry

end NUMINAMATH_CALUDE_three_hundredth_non_square_l617_61781


namespace NUMINAMATH_CALUDE_spinner_final_direction_l617_61760

/-- Represents the four cardinal directions --/
inductive Direction
| North
| East
| South
| West

/-- Represents a rotation of the spinner --/
structure Rotation :=
  (revolutions : ℚ)
  (clockwise : Bool)

/-- Calculates the final direction after applying a sequence of rotations --/
def finalDirection (initial : Direction) (rotations : List Rotation) : Direction :=
  sorry

/-- The sequence of rotations described in the problem --/
def problemRotations : List Rotation :=
  [⟨7/2, true⟩, ⟨21/4, false⟩, ⟨1/2, true⟩]

theorem spinner_final_direction :
  finalDirection Direction.North problemRotations = Direction.West :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l617_61760


namespace NUMINAMATH_CALUDE_solution_set_characterization_l617_61719

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  b : ℤ
  c : ℤ

/-- The set of all quadratic polynomials satisfying the given conditions -/
def SolutionSet : Set QuadraticPolynomial :=
  { p | ∃ (r₁ r₂ : ℤ), 
    p.b = -(r₁ + r₂) ∧ 
    p.c = r₁ * r₂ ∧ 
    1 + p.b + p.c = 10 }

/-- The theorem stating the equivalence of the solution set and the given polynomials -/
theorem solution_set_characterization :
  SolutionSet = 
    { ⟨-13, 22⟩, ⟨-9, 18⟩, ⟨9, 0⟩, ⟨5, 4⟩ } :=
by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l617_61719


namespace NUMINAMATH_CALUDE_shoe_picking_probability_l617_61709

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def blue_pairs : ℕ := 4
def green_pairs : ℕ := 3

def total_shoes : ℕ := 2 * total_pairs

theorem shoe_picking_probability :
  let black_shoes : ℕ := 2 * black_pairs
  let blue_shoes : ℕ := 2 * blue_pairs
  let green_shoes : ℕ := 2 * green_pairs
  let prob_black := (black_shoes : ℚ) / total_shoes * (black_pairs : ℚ) / (total_shoes - 1)
  let prob_blue := (blue_shoes : ℚ) / total_shoes * (blue_pairs : ℚ) / (total_shoes - 1)
  let prob_green := (green_shoes : ℚ) / total_shoes * (green_pairs : ℚ) / (total_shoes - 1)
  prob_black + prob_blue + prob_green = 89 / 435 :=
by sorry

end NUMINAMATH_CALUDE_shoe_picking_probability_l617_61709


namespace NUMINAMATH_CALUDE_same_solution_value_of_c_l617_61722

theorem same_solution_value_of_c : 
  ∀ x c : ℚ, (3 * x + 4 = 2 ∧ c * x - 15 = 0) → c = -45/2 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_value_of_c_l617_61722


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l617_61738

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (2 - x) = 1 / (2 - x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l617_61738


namespace NUMINAMATH_CALUDE_exam_score_theorem_l617_61755

/-- Proves that given the exam conditions, the number of correctly answered questions is 34 -/
theorem exam_score_theorem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) : 
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 110 →
  ∃ (correct_answers : ℕ),
    correct_answers = 34 ∧
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score :=
by sorry

end NUMINAMATH_CALUDE_exam_score_theorem_l617_61755


namespace NUMINAMATH_CALUDE_A_power_50_l617_61718

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![5, 2; -16, -6]

theorem A_power_50 : A^50 = !![301, 100; -800, -249] := by
  sorry

end NUMINAMATH_CALUDE_A_power_50_l617_61718


namespace NUMINAMATH_CALUDE_bicycle_price_reduction_l617_61731

theorem bicycle_price_reduction (initial_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  initial_price = 200 →
  discount1 = 0.3 →
  discount2 = 0.4 →
  discount3 = 0.1 →
  initial_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 75.60 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_reduction_l617_61731


namespace NUMINAMATH_CALUDE_min_value_y_squared_plus_nine_y_plus_eightyone_over_y_cubed_l617_61749

theorem min_value_y_squared_plus_nine_y_plus_eightyone_over_y_cubed 
  (y : ℝ) (h : y > 0) : y^2 + 9*y + 81/y^3 ≥ 39 ∧ 
  (y^2 + 9*y + 81/y^3 = 39 ↔ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_y_squared_plus_nine_y_plus_eightyone_over_y_cubed_l617_61749


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l617_61754

def geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = 2 * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 3 = 2) : 
  a 5 + a 7 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l617_61754


namespace NUMINAMATH_CALUDE_budget_allocation_theorem_l617_61707

def budget_allocation (microphotonics home_electronics food_additives industrial_lubricants basic_astrophysics_degrees : ℝ) : Prop :=
  let total_degrees : ℝ := 360
  let total_percentage : ℝ := 100
  let basic_astrophysics_percentage : ℝ := (basic_astrophysics_degrees / total_degrees) * total_percentage
  let known_percentage : ℝ := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage
  let gmo_percentage : ℝ := total_percentage - known_percentage
  gmo_percentage = 19

theorem budget_allocation_theorem :
  budget_allocation 14 24 15 8 72 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_theorem_l617_61707


namespace NUMINAMATH_CALUDE_vector_sum_max_min_l617_61735

/-- Given plane vectors a, b, and c satisfying certain conditions, 
    prove that the sum of the maximum and minimum values of |c| is √7 -/
theorem vector_sum_max_min (a b c : ℝ × ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 1) → 
  (a • (a - 2 • b) = 0) → 
  ((c - 2 • a) • (c - b) = 0) →
  (Real.sqrt ((max (‖c‖) (‖c‖)) ^ 2 + (min (‖c‖) (‖c‖)) ^ 2) = Real.sqrt 7) := by
  sorry

#check vector_sum_max_min

end NUMINAMATH_CALUDE_vector_sum_max_min_l617_61735


namespace NUMINAMATH_CALUDE_black_ball_prob_compare_l617_61705

-- Define the number of balls in each box
def box_a_red : ℕ := 40
def box_a_black : ℕ := 10
def box_b_red : ℕ := 60
def box_b_black : ℕ := 40
def box_b_white : ℕ := 50

-- Define the total number of balls in each box
def total_a : ℕ := box_a_red + box_a_black
def total_b : ℕ := box_b_red + box_b_black + box_b_white

-- Define the probabilities of drawing a black ball from each box
def prob_a : ℚ := box_a_black / total_a
def prob_b : ℚ := box_b_black / total_b

-- Theorem statement
theorem black_ball_prob_compare : prob_b > prob_a := by
  sorry

end NUMINAMATH_CALUDE_black_ball_prob_compare_l617_61705


namespace NUMINAMATH_CALUDE_order_of_abc_l617_61769

theorem order_of_abc : 
  let a := (Real.exp 0.6)⁻¹
  let b := 0.4
  let c := (Real.log 1.4) / 1.4
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l617_61769


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l617_61799

theorem complex_fraction_simplification :
  let numerator := (11^4 + 484) * (23^4 + 484) * (35^4 + 484) * (47^4 + 484) * (59^4 + 484)
  let denominator := (5^4 + 484) * (17^4 + 484) * (29^4 + 484) * (41^4 + 484) * (53^4 + 484)
  ∀ x : ℕ, x^4 + 484 = (x^2 - 22*x + 22) * (x^2 + 22*x + 22) →
  (numerator / denominator : ℚ) = 3867 / 7 := by
sorry

#eval (3867 : ℚ) / 7

end NUMINAMATH_CALUDE_complex_fraction_simplification_l617_61799


namespace NUMINAMATH_CALUDE_night_ride_ratio_l617_61706

def ferris_wheel_total : ℕ := 13
def roller_coaster_total : ℕ := 9
def ferris_wheel_day : ℕ := 7
def roller_coaster_day : ℕ := 4

theorem night_ride_ratio :
  (ferris_wheel_total - ferris_wheel_day) * 5 = (roller_coaster_total - roller_coaster_day) * 6 := by
  sorry

end NUMINAMATH_CALUDE_night_ride_ratio_l617_61706


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_l617_61702

theorem sqrt_two_times_sqrt_three : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_l617_61702


namespace NUMINAMATH_CALUDE_meeting_chair_rows_l617_61798

/-- Calculates the number of rows of meeting chairs given the initial water amount,
    cup capacity, chairs per row, and remaining water. -/
theorem meeting_chair_rows
  (initial_water : ℕ)  -- Initial water in gallons
  (cup_capacity : ℕ)   -- Cup capacity in ounces
  (chairs_per_row : ℕ) -- Number of chairs per row
  (water_left : ℕ)     -- Water left after filling cups in ounces
  (h1 : initial_water = 3)
  (h2 : cup_capacity = 6)
  (h3 : chairs_per_row = 10)
  (h4 : water_left = 84)
  : ℕ := by
  sorry

#check meeting_chair_rows

end NUMINAMATH_CALUDE_meeting_chair_rows_l617_61798


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_720_l617_61762

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_720 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≤ 99999 ∧ 
    digit_product n = 720 → 
    n ≤ 98521 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_720_l617_61762


namespace NUMINAMATH_CALUDE_shuffle_32_cards_l617_61793

/-- The number of ways to shuffle a deck of cards -/
def shuffleWays (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of ways to shuffle a deck of 32 cards is 32! -/
theorem shuffle_32_cards : shuffleWays 32 = Nat.factorial 32 := by
  sorry

end NUMINAMATH_CALUDE_shuffle_32_cards_l617_61793


namespace NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l617_61796

/-- The revenue function for the bookstore -/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The optimal price that maximizes revenue -/
def optimal_price : ℝ := 18.75

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℝ, p ≤ 30 → revenue p ≤ revenue optimal_price := by
  sorry


end NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l617_61796


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l617_61747

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people on each side of Cara -/
def people_per_side : ℕ := 3

/-- The number of potential neighbors for Cara -/
def potential_neighbors : ℕ := 2 * people_per_side - 2

/-- The number of people in each pair next to Cara -/
def pair_size : ℕ := 2

theorem cara_seating_arrangements :
  Nat.choose potential_neighbors pair_size = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l617_61747


namespace NUMINAMATH_CALUDE_zacks_marbles_l617_61725

theorem zacks_marbles (initial_marbles : ℕ) : 
  (initial_marbles - 5) % 3 = 0 → 
  initial_marbles = 3 * 20 + 5 → 
  initial_marbles = 65 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l617_61725


namespace NUMINAMATH_CALUDE_second_side_length_l617_61768

/-- A triangle with a perimeter of 55 centimeters and two sides measuring 5 and 30 centimeters. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter_eq : side1 + side2 + side3 = 55
  side1_eq : side1 = 5
  side3_eq : side3 = 30

/-- The second side of the triangle measures 20 centimeters. -/
theorem second_side_length (t : Triangle) : t.side2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_side_length_l617_61768


namespace NUMINAMATH_CALUDE_misses_both_mutually_exclusive_not_contradictory_l617_61785

-- Define the sample space for two shots
inductive ShotOutcome
  | Miss
  | Hit

-- Define the event of hitting exactly once
def hits_exactly_once (outcome : ShotOutcome × ShotOutcome) : Prop :=
  (outcome.1 = ShotOutcome.Hit ∧ outcome.2 = ShotOutcome.Miss) ∨
  (outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Hit)

-- Define the event of missing both times
def misses_both_times (outcome : ShotOutcome × ShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

-- Theorem statement
theorem misses_both_mutually_exclusive_not_contradictory :
  (∀ outcome : ShotOutcome × ShotOutcome, ¬(hits_exactly_once outcome ∧ misses_both_times outcome)) ∧
  (∃ outcome : ShotOutcome × ShotOutcome, hits_exactly_once outcome ∨ misses_both_times outcome) :=
sorry

end NUMINAMATH_CALUDE_misses_both_mutually_exclusive_not_contradictory_l617_61785


namespace NUMINAMATH_CALUDE_solve_equation_l617_61717

theorem solve_equation (y : ℤ) (h : 7 - y = 13) : y = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l617_61717


namespace NUMINAMATH_CALUDE_dance_group_average_age_l617_61742

/-- Calculates the average age of a dance group given the number and average ages of children and adults. -/
theorem dance_group_average_age 
  (num_children : ℕ) 
  (num_adults : ℕ) 
  (avg_age_children : ℚ) 
  (avg_age_adults : ℚ) 
  (h1 : num_children = 8) 
  (h2 : num_adults = 12) 
  (h3 : avg_age_children = 12) 
  (h4 : avg_age_adults = 40) :
  let total_members := num_children + num_adults
  let total_age := num_children * avg_age_children + num_adults * avg_age_adults
  total_age / total_members = 288 / 10 := by
  sorry

end NUMINAMATH_CALUDE_dance_group_average_age_l617_61742


namespace NUMINAMATH_CALUDE_g_of_two_l617_61720

/-- Given a function g: ℝ → ℝ that satisfies g(x) - 2g(1/x) = 3^x for all x ≠ 0,
    prove that g(2) = -3 - (4√3)/9 -/
theorem g_of_two (g : ℝ → ℝ) (h : ∀ x ≠ 0, g x - 2 * g (1/x) = 3^x) :
  g 2 = -3 - (4 * Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_l617_61720


namespace NUMINAMATH_CALUDE_james_muffins_l617_61788

theorem james_muffins (arthur_muffins : ℕ) (james_multiplier : ℕ) 
  (h1 : arthur_muffins = 115)
  (h2 : james_multiplier = 12) :
  arthur_muffins * james_multiplier = 1380 :=
by sorry

end NUMINAMATH_CALUDE_james_muffins_l617_61788


namespace NUMINAMATH_CALUDE_last_locker_opened_is_511_l617_61724

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction of the student's movement -/
inductive Direction
| Forward
| Backward

/-- Represents the student's position and movement direction -/
structure StudentPosition :=
  (position : Nat)
  (direction : Direction)

/-- Represents the state of all lockers -/
def LockerSystem := Fin 512 → LockerState

/-- The process of opening lockers according to the described pattern -/
def openLockers : LockerSystem → StudentPosition → LockerSystem
  | lockers, _ => sorry  -- Implementation details omitted

/-- Checks if all lockers are open -/
def allLockersOpen : LockerSystem → Bool
  | _ => sorry  -- Implementation details omitted

/-- Finds the number of the last closed locker -/
def lastClosedLocker : LockerSystem → Option Nat
  | _ => sorry  -- Implementation details omitted

/-- The main theorem stating that the last locker opened is 511 -/
theorem last_locker_opened_is_511 :
  let initial_lockers : LockerSystem := fun _ => LockerState.Closed
  let initial_position : StudentPosition := ⟨0, Direction.Forward⟩
  let final_lockers := openLockers initial_lockers initial_position
  allLockersOpen final_lockers ∧ lastClosedLocker final_lockers = some 511 :=
sorry


end NUMINAMATH_CALUDE_last_locker_opened_is_511_l617_61724


namespace NUMINAMATH_CALUDE_cody_initial_tickets_l617_61740

def initial_tickets : ℕ → Prop
  | t => (t - 25 + 6 = 30)

theorem cody_initial_tickets : ∃ t : ℕ, initial_tickets t ∧ t = 49 := by
  sorry

end NUMINAMATH_CALUDE_cody_initial_tickets_l617_61740


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l617_61787

/-- A polynomial x^2 + bx + c has exactly one real root if and only if its discriminant is zero -/
def has_one_real_root (b c : ℝ) : Prop :=
  b^2 - 4*c = 0

/-- The theorem statement -/
theorem unique_root_quadratic (b c : ℝ) 
  (h1 : has_one_real_root b c)
  (h2 : b = c^2 + 1) : 
  c = 1 ∨ c = -1 :=
sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l617_61787


namespace NUMINAMATH_CALUDE_salary_increase_after_reduction_l617_61704

theorem salary_increase_after_reduction : ∀ (original_salary : ℝ),
  original_salary > 0 →
  let reduced_salary := original_salary * (1 - 0.25)
  let increase_factor := (1 + 1/3)
  reduced_salary * increase_factor = original_salary :=
by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_reduction_l617_61704


namespace NUMINAMATH_CALUDE_onions_on_shelf_l617_61782

def remaining_onions (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

theorem onions_on_shelf : remaining_onions 98 65 = 33 := by
  sorry

end NUMINAMATH_CALUDE_onions_on_shelf_l617_61782


namespace NUMINAMATH_CALUDE_halloween_candy_count_l617_61737

/-- Calculates Haley's final candy count after Halloween -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem: Given Haley's initial candy count, the amount she ate, and the amount she received,
    her final candy count is equal to 35. -/
theorem halloween_candy_count :
  final_candy_count 33 17 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l617_61737


namespace NUMINAMATH_CALUDE_max_hands_in_dance_l617_61711

/-- Represents a Martian participating in the dance --/
structure Martian :=
  (hands : Nat)
  (hands_le_three : hands ≤ 3)

/-- Represents the dance configuration --/
structure DanceConfiguration :=
  (participants : List Martian)
  (participant_count_le_seven : participants.length ≤ 7)

/-- Calculates the total number of hands in a dance configuration --/
def total_hands (config : DanceConfiguration) : Nat :=
  config.participants.foldl (λ sum martian => sum + martian.hands) 0

/-- Theorem: The maximum number of hands involved in the dance is 20 --/
theorem max_hands_in_dance :
  ∃ (config : DanceConfiguration),
    (∀ (other_config : DanceConfiguration),
      total_hands other_config ≤ total_hands config) ∧
    total_hands config = 20 ∧
    total_hands config % 2 = 0 :=
  sorry

end NUMINAMATH_CALUDE_max_hands_in_dance_l617_61711


namespace NUMINAMATH_CALUDE_cloth_selling_price_l617_61773

/-- Calculates the total selling price of cloth given the quantity, cost price, and loss per metre. -/
def totalSellingPrice (quantity : ℕ) (costPrice lossPerMetre : ℚ) : ℚ :=
  quantity * (costPrice - lossPerMetre)

/-- Proves that the total selling price for 500 metres of cloth with a cost price of 41 and a loss of 5 per metre is 18000. -/
theorem cloth_selling_price :
  totalSellingPrice 500 41 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l617_61773


namespace NUMINAMATH_CALUDE_ann_age_is_36_l617_61745

/-- Represents the ages of Ann and Barbara -/
structure Ages where
  ann : ℕ
  barbara : ℕ

/-- The condition that the sum of their ages is 72 -/
def sum_of_ages (ages : Ages) : Prop :=
  ages.ann + ages.barbara = 72

/-- The complex relationship between their ages as described in the problem -/
def age_relationship (ages : Ages) : Prop :=
  ages.barbara = ages.ann - (ages.barbara - (ages.ann - (ages.ann / 3)))

/-- The theorem stating that given the conditions, Ann's age is 36 -/
theorem ann_age_is_36 (ages : Ages) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relationship ages) : 
  ages.ann = 36 := by
  sorry

end NUMINAMATH_CALUDE_ann_age_is_36_l617_61745


namespace NUMINAMATH_CALUDE_division_simplification_l617_61721

theorem division_simplification : (180 : ℚ) / (12 + 9 * 3 - 4) = 36 / 7 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l617_61721


namespace NUMINAMATH_CALUDE_average_gas_mileage_l617_61743

/-- Calculates the average gas mileage for a trip with electric and gas cars -/
theorem average_gas_mileage 
  (total_distance : ℝ) 
  (electric_distance : ℝ) 
  (rented_distance : ℝ) 
  (electric_efficiency : ℝ) 
  (rented_efficiency : ℝ) 
  (h1 : total_distance = 400)
  (h2 : electric_distance = 300)
  (h3 : rented_distance = 100)
  (h4 : electric_efficiency = 50)
  (h5 : rented_efficiency = 25)
  (h6 : total_distance = electric_distance + rented_distance) :
  (total_distance / (electric_distance / electric_efficiency + rented_distance / rented_efficiency)) = 40 := by
  sorry

#check average_gas_mileage

end NUMINAMATH_CALUDE_average_gas_mileage_l617_61743


namespace NUMINAMATH_CALUDE_all_false_if_some_false_l617_61772

-- Define the universe of quadrilaterals
variable (Q : Type)

-- Define property A
variable (A : Q → Prop)

-- Theorem statement
theorem all_false_if_some_false :
  (¬ ∃ x : Q, A x) → ¬ (∀ x : Q, A x) := by
  sorry

end NUMINAMATH_CALUDE_all_false_if_some_false_l617_61772


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l617_61765

-- Define the piecewise function
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then 2*a*x + 1
  else if x ≥ -1 then 3*x - c
  else 3*x + b

-- State the theorem
theorem continuous_piecewise_function_sum (a b c : ℝ) :
  (Continuous (f a b c)) →
  a + b - c = 5*a - 4 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l617_61765


namespace NUMINAMATH_CALUDE_inequality_holds_iff_in_interval_l617_61791

/-- For fixed positive real numbers a and b, the inequality 
    (1/√x) + (1/√(a+b-x)) < (1/√a) + (1/√b) 
    holds if and only if x is in the open interval (min(a, b), max(a, b)) -/
theorem inequality_holds_iff_in_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, (1 / Real.sqrt x + 1 / Real.sqrt (a + b - x) < 1 / Real.sqrt a + 1 / Real.sqrt b) ↔ 
    (min a b < x ∧ x < max a b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_in_interval_l617_61791
