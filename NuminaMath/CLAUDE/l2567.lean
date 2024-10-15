import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2567_256746

theorem equation_solution :
  ∃ x : ℝ, (3 * x + 9 = 0) ∧ (x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2567_256746


namespace NUMINAMATH_CALUDE_point_not_on_line_l2567_256734

theorem point_not_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  ¬ (∃ (x y : ℝ), y = m * x + b ∧ x = 0 ∧ y = -2023) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l2567_256734


namespace NUMINAMATH_CALUDE_inequality_not_always_correct_l2567_256765

theorem inequality_not_always_correct 
  (x y z : ℝ) (k : ℤ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z ≠ 0) (hk : k ≠ 0) :
  ¬ (∀ (x y z : ℝ) (k : ℤ), x > 0 → y > 0 → x > y → z ≠ 0 → k ≠ 0 → x / z^k > y / z^k) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_correct_l2567_256765


namespace NUMINAMATH_CALUDE_waiter_problem_l2567_256729

/-- The number of customers who left the waiter's section -/
def customers_left : ℕ := 14

/-- The number of people at each remaining table -/
def people_per_table : ℕ := 4

/-- The number of tables in the waiter's section -/
def number_of_tables : ℕ := 2

/-- The initial number of customers in the waiter's section -/
def initial_customers : ℕ := 22

theorem waiter_problem :
  initial_customers = customers_left + (number_of_tables * people_per_table) :=
sorry

end NUMINAMATH_CALUDE_waiter_problem_l2567_256729


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l2567_256756

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfDigitFactorials (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def hasDigit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ n.digits 10

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = sumOfDigitFactorials n ∧ hasDigit n 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l2567_256756


namespace NUMINAMATH_CALUDE_total_money_l2567_256733

theorem total_money (r p q : ℕ) (h1 : r = 1600) (h2 : r = (2 * (p + q)) / 3) : 
  p + q + r = 4000 := by
sorry

end NUMINAMATH_CALUDE_total_money_l2567_256733


namespace NUMINAMATH_CALUDE_fraction_equality_l2567_256715

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 2) : (a - b) / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2567_256715


namespace NUMINAMATH_CALUDE_equation_solutions_l2567_256761

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 10 ∧ x₂ = 3 - Real.sqrt 10 ∧
    x₁^2 - 6*x₁ = 1 ∧ x₂^2 - 6*x₂ = 1) ∧
  (∃ x₃ x₄ : ℝ, x₃ = 2/3 ∧ x₄ = -4 ∧
    (x₃ - 3)^2 = (2*x₃ + 1)^2 ∧ (x₄ - 3)^2 = (2*x₄ + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2567_256761


namespace NUMINAMATH_CALUDE_betty_oranges_l2567_256799

theorem betty_oranges (emily sandra betty : ℕ) 
  (h1 : emily = 7 * sandra) 
  (h2 : sandra = 3 * betty) 
  (h3 : emily = 252) : 
  betty = 12 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l2567_256799


namespace NUMINAMATH_CALUDE_calculation_proof_l2567_256728

theorem calculation_proof :
  (- (1 : ℤ)^4 + 16 / (-2 : ℤ)^3 * |-3 - 1| = -3) ∧
  (∀ a b : ℝ, -2 * (a^2 * b - 1/4 * a * b^2 + 1/2 * a^3) - (-2 * a^2 * b + 3 * a * b^2) = -5/2 * a * b^2 - a^3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2567_256728


namespace NUMINAMATH_CALUDE_problem_solution_l2567_256775

theorem problem_solution : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2567_256775


namespace NUMINAMATH_CALUDE_cayley_hamilton_for_A_l2567_256780

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 2, 3],
    ![2, 1, 2],
    ![3, 2, 1]]

theorem cayley_hamilton_for_A :
  A^3 + (-8 : ℤ) • A^2 + (-2 : ℤ) • A + (-8 : ℤ) • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cayley_hamilton_for_A_l2567_256780


namespace NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l2567_256721

theorem not_divisible_by_1000_power_minus_1 (m : ℕ) : ¬(1000^m - 1 ∣ 1978^m - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l2567_256721


namespace NUMINAMATH_CALUDE_no_integer_points_on_circle_l2567_256774

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, (x - 3)^2 + (3*x + 1)^2 > 16 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_points_on_circle_l2567_256774


namespace NUMINAMATH_CALUDE_product_sum_and_reciprocals_bound_l2567_256776

theorem product_sum_and_reciprocals_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 ∧
  ∀ ε > 0, ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    (a' + b' + c') * (1/a' + 1/b' + 1/c') < 9 + ε :=
by sorry

end NUMINAMATH_CALUDE_product_sum_and_reciprocals_bound_l2567_256776


namespace NUMINAMATH_CALUDE_smallest_b_value_l2567_256716

theorem smallest_b_value (a c d : ℤ) (x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, x^4 + a*x^3 + (x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄)*x^2 + c*x + d = 0 → x > 0) →
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 →
  d = x₁ * x₂ * x₃ * x₄ →
  x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄ ≥ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2567_256716


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_sequence_l2567_256784

theorem smallest_sum_arithmetic_geometric_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →
  (∃ r : ℚ, C - B = B - A ∧ C = B * r ∧ D = C * r) →
  C = (5 : ℚ) / 3 * B →
  A + B + C + D ≥ 52 ∧ (∃ A' B' C' D' : ℤ, 
    A' > 0 ∧ B' > 0 ∧ C' > 0 ∧
    (∃ r' : ℚ, C' - B' = B' - A' ∧ C' = B' * r' ∧ D' = C' * r') ∧
    C' = (5 : ℚ) / 3 * B' ∧
    A' + B' + C' + D' = 52) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_sequence_l2567_256784


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l2567_256720

/-- The number of regular soda bottles in the grocery store. -/
def regular_soda : ℕ := 67

/-- The number of diet soda bottles in the grocery store. -/
def diet_soda : ℕ := 9

/-- The difference between the number of regular soda bottles and diet soda bottles. -/
def soda_difference : ℕ := regular_soda - diet_soda

theorem soda_bottle_difference : soda_difference = 58 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l2567_256720


namespace NUMINAMATH_CALUDE_min_value_of_sum_equality_condition_l2567_256788

theorem min_value_of_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b ≥ 8 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b = 8 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_equality_condition_l2567_256788


namespace NUMINAMATH_CALUDE_complex_magnitude_three_fourths_minus_five_sixths_i_l2567_256724

theorem complex_magnitude_three_fourths_minus_five_sixths_i :
  Complex.abs (3/4 - Complex.I * 5/6) = Real.sqrt 181 / 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_three_fourths_minus_five_sixths_i_l2567_256724


namespace NUMINAMATH_CALUDE_equation_implies_equilateral_l2567_256783

/-- A triangle with sides a, b, c and opposite angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The equation that the triangle satisfies -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.a * Real.cos t.α + t.b * Real.cos t.β + t.c * Real.cos t.γ) /
  (t.a * Real.sin t.β + t.b * Real.sin t.γ + t.c * Real.sin t.α) =
  perimeter t / (9 * circumradius t)

/-- A triangle is equilateral if all its sides are equal -/
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

/-- The main theorem -/
theorem equation_implies_equilateral (t : Triangle) :
  satisfies_equation t → is_equilateral t :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_equilateral_l2567_256783


namespace NUMINAMATH_CALUDE_f_range_theorem_l2567_256727

def f (x m : ℝ) : ℝ := |x + 1| + |x - m|

theorem f_range_theorem :
  (∀ m : ℝ, (∀ x : ℝ, f x m ≥ 3) ↔ m ∈ Set.Ici 2 ∪ Set.Iic (-4)) ∧
  (∀ m : ℝ, (∃ x : ℝ, f m m - 2*m ≥ x^2 - x) ↔ m ∈ Set.Iic (5/4)) := by
  sorry

end NUMINAMATH_CALUDE_f_range_theorem_l2567_256727


namespace NUMINAMATH_CALUDE_sum_equals_two_thirds_l2567_256786

theorem sum_equals_two_thirds :
  let original_sum := (1:ℚ)/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
  let removed_terms := 1/12 + 1/15
  let remaining_sum := original_sum - removed_terms
  remaining_sum = 2/3 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_two_thirds_l2567_256786


namespace NUMINAMATH_CALUDE_division_sum_theorem_l2567_256751

theorem division_sum_theorem (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 65)
  (h_divisor : divisor = 24)
  (h_remainder : remainder = 5) :
  quotient * divisor + remainder = 1565 :=
by sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l2567_256751


namespace NUMINAMATH_CALUDE_art_project_marker_distribution_l2567_256791

/-- Proves that each student in the last group receives 5 markers given the conditions of the art project. -/
theorem art_project_marker_distribution :
  let total_students : ℕ := 68
  let total_groups : ℕ := 5
  let total_marker_boxes : ℕ := 48
  let markers_per_box : ℕ := 6
  let group1_students : ℕ := 12
  let group1_markers_per_student : ℕ := 2
  let group2_students : ℕ := 20
  let group2_markers_per_student : ℕ := 3
  let group3_students : ℕ := 15
  let group3_markers_per_student : ℕ := 5
  let group4_students : ℕ := 8
  let group4_markers_per_student : ℕ := 8
  let total_markers : ℕ := total_marker_boxes * markers_per_box
  let used_markers : ℕ := group1_students * group1_markers_per_student +
                          group2_students * group2_markers_per_student +
                          group3_students * group3_markers_per_student +
                          group4_students * group4_markers_per_student
  let remaining_markers : ℕ := total_markers - used_markers
  let last_group_students : ℕ := total_students - (group1_students + group2_students + group3_students + group4_students)
  remaining_markers / last_group_students = 5 :=
by sorry

end NUMINAMATH_CALUDE_art_project_marker_distribution_l2567_256791


namespace NUMINAMATH_CALUDE_append_two_digit_numbers_formula_l2567_256752

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Appends one two-digit number after another -/
def append_two_digit_numbers (n1 n2 : TwoDigitNumber) : Nat :=
  1000 * n1.tens + 100 * n1.units + 10 * n2.tens + n2.units

/-- Theorem: Appending two two-digit numbers results in the expected formula -/
theorem append_two_digit_numbers_formula (n1 n2 : TwoDigitNumber) :
  append_two_digit_numbers n1 n2 = 1000 * n1.tens + 100 * n1.units + 10 * n2.tens + n2.units :=
by sorry

end NUMINAMATH_CALUDE_append_two_digit_numbers_formula_l2567_256752


namespace NUMINAMATH_CALUDE_store_pricing_l2567_256703

-- Define variables for the prices of individual items
variable (p n e : ℝ)

-- Define the equations based on the given conditions
def equation1 : Prop := 10 * p + 12 * n + 6 * e = 5.50
def equation2 : Prop := 6 * p + 4 * n + 3 * e = 2.40

-- Define the final cost calculation
def final_cost : ℝ := 20 * p + 15 * n + 9 * e

-- Theorem statement
theorem store_pricing (h1 : equation1 p n e) (h2 : equation2 p n e) : 
  final_cost p n e = 8.95 := by
  sorry


end NUMINAMATH_CALUDE_store_pricing_l2567_256703


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2567_256738

def α : ℝ → ℝ := λ x ↦ 4 * x + 9
def β : ℝ → ℝ := λ x ↦ 9 * x + 6

theorem composition_equation_solution :
  ∃! x : ℝ, α (β x) = 8 ∧ x = -25/36 := by sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2567_256738


namespace NUMINAMATH_CALUDE_min_sum_of_integers_l2567_256785

theorem min_sum_of_integers (m n : ℕ) : 
  m < n → 
  m > 0 → 
  n > 0 → 
  m * n = (m - 20) * (n + 23) → 
  ∀ k l : ℕ, k < l → k > 0 → l > 0 → k * l = (k - 20) * (l + 23) → m + n ≤ k + l →
  m + n = 321 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_integers_l2567_256785


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l2567_256782

theorem smallest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → 1013 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l2567_256782


namespace NUMINAMATH_CALUDE_percentage_equality_l2567_256764

theorem percentage_equality (x y : ℝ) (h1 : 2 * x = 0.5 * y) (h2 : x = 16) : y = 64 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2567_256764


namespace NUMINAMATH_CALUDE_johns_pace_l2567_256778

/-- Given the conditions of a race between John and Steve, prove that John's pace during his final push was 178 / 42.5 m/s. -/
theorem johns_pace (john_initial_behind : ℝ) (steve_speed : ℝ) (john_final_ahead : ℝ) (push_duration : ℝ) :
  john_initial_behind = 15 →
  steve_speed = 3.8 →
  john_final_ahead = 2 →
  push_duration = 42.5 →
  (john_initial_behind + john_final_ahead + steve_speed * push_duration) / push_duration = 178 / 42.5 := by
  sorry

#eval (178 : ℚ) / 42.5

end NUMINAMATH_CALUDE_johns_pace_l2567_256778


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_l2567_256709

theorem largest_consecutive_sum (n : ℕ) : n = 14141 ↔ 
  (∀ k : ℕ, k ≤ n → (k * (k + 1)) / 2 ≤ 100000000) ∧
  (∀ m : ℕ, m > n → (m * (m + 1)) / 2 > 100000000) := by
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_l2567_256709


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_l2567_256753

theorem sqrt_meaningful_iff (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_l2567_256753


namespace NUMINAMATH_CALUDE_whistle_search_bound_l2567_256710

/-- Represents a football field -/
structure FootballField where
  length : ℝ
  width : ℝ

/-- Represents the position of an object on the field -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the referee's search process -/
def search (field : FootballField) (start : Position) (whistle : Position) : ℕ :=
  sorry

/-- Theorem stating the upper bound on the number of steps needed to find the whistle -/
theorem whistle_search_bound 
  (field : FootballField)
  (start : Position)
  (whistle : Position)
  (h_field_size : field.length = 100 ∧ field.width = 70)
  (h_start_corner : start.x = 0 ∧ start.y = 0)
  (h_whistle_on_field : whistle.x ≥ 0 ∧ whistle.x ≤ field.length ∧ whistle.y ≥ 0 ∧ whistle.y ≤ field.width)
  (d : ℝ)
  (h_initial_distance : d = Real.sqrt ((whistle.x - start.x)^2 + (whistle.y - start.y)^2)) :
  (search field start whistle) ≤ ⌊Real.sqrt 2 * (d + 1)⌋ + 4 :=
sorry

end NUMINAMATH_CALUDE_whistle_search_bound_l2567_256710


namespace NUMINAMATH_CALUDE_circumscribed_polygon_has_triangle_l2567_256749

/-- A polygon circumscribed about a circle. -/
structure CircumscribedPolygon where
  /-- The number of sides in the polygon. -/
  n : ℕ
  /-- The lengths of the sides of the polygon. -/
  sides : Fin n → ℝ
  /-- All side lengths are positive. -/
  sides_pos : ∀ i, 0 < sides i

/-- Theorem: In any polygon circumscribed about a circle, 
    there exist three sides that can form a triangle. -/
theorem circumscribed_polygon_has_triangle (P : CircumscribedPolygon) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    P.sides i + P.sides j > P.sides k ∧
    P.sides j + P.sides k > P.sides i ∧
    P.sides k + P.sides i > P.sides j :=
sorry

end NUMINAMATH_CALUDE_circumscribed_polygon_has_triangle_l2567_256749


namespace NUMINAMATH_CALUDE_female_officers_count_l2567_256793

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 204 →
  female_on_duty_ratio = 1/2 →
  female_ratio = 17/100 →
  ∃ (total_female : ℕ), total_female = 600 ∧ 
    (female_ratio * total_female : ℚ) = (female_on_duty_ratio * total_on_duty : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2567_256793


namespace NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l2567_256714

def A : Set ℕ := {n : ℕ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a^2 + 2*b^2}

theorem prime_square_in_A_implies_prime_in_A (p : ℕ) (hp : Nat.Prime p) (hp2 : p^2 ∈ A) : p ∈ A := by
  sorry

end NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l2567_256714


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2567_256722

theorem regular_polygon_exterior_angle (n : ℕ) :
  (360 / n : ℝ) = 72 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2567_256722


namespace NUMINAMATH_CALUDE_combination_equation_solution_l2567_256718

theorem combination_equation_solution (n : ℕ+) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l2567_256718


namespace NUMINAMATH_CALUDE_sand_pit_fill_theorem_l2567_256725

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents a sand pit with its dimensions and current fill level -/
structure SandPit where
  dimensions : PrismDimensions
  fillLevel : ℝ  -- Represents the fraction of the pit that is filled (0 to 1)

/-- Calculates the additional sand volume needed to fill the pit completely -/
def additionalSandNeeded (pit : SandPit) : ℝ :=
  (1 - pit.fillLevel) * prismVolume pit.dimensions

theorem sand_pit_fill_theorem (pit : SandPit) 
    (h1 : pit.dimensions.length = 10)
    (h2 : pit.dimensions.width = 2)
    (h3 : pit.dimensions.height = 0.5)
    (h4 : pit.fillLevel = 0.5) :
    additionalSandNeeded pit = 5 := by
  sorry

#eval additionalSandNeeded {
  dimensions := { length := 10, width := 2, height := 0.5 },
  fillLevel := 0.5
}

end NUMINAMATH_CALUDE_sand_pit_fill_theorem_l2567_256725


namespace NUMINAMATH_CALUDE_even_monotone_decreasing_inequality_l2567_256763

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

-- State the theorem
theorem even_monotone_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_mono : monotone_decreasing_on_pos f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_decreasing_inequality_l2567_256763


namespace NUMINAMATH_CALUDE_candy_bar_earnings_difference_l2567_256745

/-- The problem of calculating the difference in earnings between Tina and Marvin from selling candy bars. -/
theorem candy_bar_earnings_difference : 
  let candy_bar_price : ℕ := 2
  let marvin_sales : ℕ := 35
  let tina_sales : ℕ := 3 * marvin_sales
  let marvin_earnings : ℕ := candy_bar_price * marvin_sales
  let tina_earnings : ℕ := candy_bar_price * tina_sales
  tina_earnings - marvin_earnings = 140 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_earnings_difference_l2567_256745


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2567_256744

theorem complex_fraction_equals_i : 
  let i : ℂ := Complex.I
  (1 + i^2017) / (1 - i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2567_256744


namespace NUMINAMATH_CALUDE_circle_center_and_point_check_l2567_256760

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 11 = 0

def center : ℝ × ℝ := (3, -1)

def point : ℝ × ℝ := (5, -1)

theorem circle_center_and_point_check :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = 21) ∧
  ¬ circle_equation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_point_check_l2567_256760


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2567_256740

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 2003 * a + 3005 = 0) →
  (5 * b^3 + 2003 * b + 3005 = 0) →
  (5 * c^3 + 2003 * c + 3005 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2567_256740


namespace NUMINAMATH_CALUDE_square_sum_equals_25_l2567_256767

theorem square_sum_equals_25 (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) :
  x^2 + y^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_25_l2567_256767


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2567_256701

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 → 
  n ∈ Finset.range 1982 → 
  (n^2 - m*n - m^2)^2 = 1 → 
  m^2 + n^2 ≤ 3524578 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2567_256701


namespace NUMINAMATH_CALUDE_expression_evaluation_l2567_256712

theorem expression_evaluation : 
  Real.sqrt 3 * Real.cos (30 * π / 180) + (3 - π)^0 - 2 * Real.tan (45 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2567_256712


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l2567_256708

/-- Circle1 is defined by the equation x^2 - 6x + y^2 + 10y + 9 = 0 -/
def Circle1 (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 10*y + 9 = 0

/-- Circle2 is defined by the equation x^2 + 4x + y^2 - 8y + 4 = 0 -/
def Circle2 (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 8*y + 4 = 0

/-- The shortest distance between Circle1 and Circle2 is √106 - 9 -/
theorem shortest_distance_between_circles :
  ∃ (d : ℝ), d = Real.sqrt 106 - 9 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    Circle1 x₁ y₁ → Circle2 x₂ y₂ →
    d ≤ Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l2567_256708


namespace NUMINAMATH_CALUDE_man_speed_man_speed_result_l2567_256735

/-- Calculates the speed of a man given a train passing him in the opposite direction -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * (3600 / 1000)
  man_speed_kmh

/-- The speed of the man is approximately 6 km/h -/
theorem man_speed_result :
  ∃ ε > 0, |man_speed 200 60 10.909090909090908 - 6| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_result_l2567_256735


namespace NUMINAMATH_CALUDE_last_three_digits_proof_l2567_256796

theorem last_three_digits_proof : 
  (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 3 % 1000 = 976 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_proof_l2567_256796


namespace NUMINAMATH_CALUDE_ivan_number_properties_l2567_256779

def sum_of_digits (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

theorem ivan_number_properties (n : ℕ) (h : n > 0) :
  let x := (sum_of_digits n)^2
  (num_digits n ≤ 3 → x < 730) ∧
  (num_digits n = 4 → x < n) ∧
  (num_digits n ≥ 5 → x < n) ∧
  (∀ m : ℕ, m > 0 → (sum_of_digits x)^2 = m → (m = 1 ∨ m = 81)) :=
by sorry

#check ivan_number_properties

end NUMINAMATH_CALUDE_ivan_number_properties_l2567_256779


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l2567_256739

/-- Given a farm with sheep and horses, calculate the daily horse food requirement per horse -/
theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (h_sheep_count : sheep_count = 32) 
  (h_total_horse_food : total_horse_food = 12880) 
  (h_ratio : sheep_count * 7 = 32 * 4) : 
  total_horse_food / (sheep_count * 7 / 4) = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l2567_256739


namespace NUMINAMATH_CALUDE_intersection_M_N_l2567_256706

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2567_256706


namespace NUMINAMATH_CALUDE_number_difference_problem_l2567_256773

theorem number_difference_problem : ∃ (a b : ℕ), 
  a + b = 25650 ∧ 
  a % 100 = 0 ∧ 
  a / 100 = b ∧ 
  a - b = 25146 := by
sorry

end NUMINAMATH_CALUDE_number_difference_problem_l2567_256773


namespace NUMINAMATH_CALUDE_fourth_week_sales_l2567_256762

def chocolate_sales (week1 week2 week3 week4 week5 : ℕ) : Prop :=
  let total := week1 + week2 + week3 + week4 + week5
  (total : ℚ) / 5 = 71

theorem fourth_week_sales :
  ∀ week4 : ℕ,
  chocolate_sales 75 67 75 week4 68 →
  week4 = 70 := by
sorry

end NUMINAMATH_CALUDE_fourth_week_sales_l2567_256762


namespace NUMINAMATH_CALUDE_carnival_tickets_l2567_256741

/-- The total number of tickets bought by a group of friends at a carnival. -/
def total_tickets (num_friends : ℕ) (tickets_per_friend : ℕ) : ℕ :=
  num_friends * tickets_per_friend

/-- Theorem stating that 6 friends buying 39 tickets each results in 234 total tickets. -/
theorem carnival_tickets : total_tickets 6 39 = 234 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l2567_256741


namespace NUMINAMATH_CALUDE_number_of_candles_l2567_256795

def candle_weight : ℕ := 9
def total_weight : ℕ := 63

theorem number_of_candles : (total_weight / candle_weight = 7) := by
  sorry

end NUMINAMATH_CALUDE_number_of_candles_l2567_256795


namespace NUMINAMATH_CALUDE_stifel_conjecture_counterexample_l2567_256747

theorem stifel_conjecture_counterexample : ∃ n : ℕ, ¬ Nat.Prime (2^(2*n + 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_stifel_conjecture_counterexample_l2567_256747


namespace NUMINAMATH_CALUDE_frac_two_thirds_is_quadratic_radical_l2567_256723

def is_quadratic_radical (x : ℝ) : Prop := x ≥ 0

theorem frac_two_thirds_is_quadratic_radical :
  is_quadratic_radical (2/3) :=
by sorry

end NUMINAMATH_CALUDE_frac_two_thirds_is_quadratic_radical_l2567_256723


namespace NUMINAMATH_CALUDE_max_togs_value_l2567_256757

def tag_price : ℕ := 3
def tig_price : ℕ := 4
def tog_price : ℕ := 8
def total_budget : ℕ := 100

def max_togs (x y z : ℕ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧
  x * tag_price + y * tig_price + z * tog_price = total_budget ∧
  ∀ (a b c : ℕ), a ≥ 1 → b ≥ 1 → c ≥ 1 →
    a * tag_price + b * tig_price + c * tog_price = total_budget →
    c ≤ z

theorem max_togs_value : ∃ (x y : ℕ), max_togs x y 11 := by
  sorry

end NUMINAMATH_CALUDE_max_togs_value_l2567_256757


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2567_256792

theorem cubic_equation_solution : ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2567_256792


namespace NUMINAMATH_CALUDE_circuit_length_difference_l2567_256755

/-- The length of the small circuit in meters -/
def small_circuit_length : ℕ := 400

/-- The number of laps Jana runs -/
def jana_laps : ℕ := 3

/-- The number of laps Father runs -/
def father_laps : ℕ := 4

/-- The total distance Jana runs in meters -/
def jana_distance : ℕ := small_circuit_length * jana_laps

/-- The total distance Father runs in meters -/
def father_distance : ℕ := 2 * jana_distance

/-- The length of the large circuit in meters -/
def large_circuit_length : ℕ := father_distance / father_laps

theorem circuit_length_difference :
  large_circuit_length - small_circuit_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_circuit_length_difference_l2567_256755


namespace NUMINAMATH_CALUDE_counterexample_exists_l2567_256769

theorem counterexample_exists : ∃ (x y : ℝ), x > y ∧ x^2 ≤ y^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2567_256769


namespace NUMINAMATH_CALUDE_multiple_of_twenty_day_after_power_of_three_l2567_256717

-- Part 1
theorem multiple_of_twenty (n : ℕ+) : ∃ k : ℤ, 4 * 6^n.val + 5^(n.val + 1) - 9 = 20 * k := by sorry

-- Part 2
theorem day_after_power_of_three : (3^100 % 7 : ℕ) + 1 = 5 := by sorry

end NUMINAMATH_CALUDE_multiple_of_twenty_day_after_power_of_three_l2567_256717


namespace NUMINAMATH_CALUDE_appliance_pricing_l2567_256711

/-- Represents the cost price of an electrical appliance in yuan -/
def cost_price : ℝ := sorry

/-- The markup percentage as a decimal -/
def markup : ℝ := 0.30

/-- The discount percentage as a decimal -/
def discount : ℝ := 0.20

/-- The final selling price in yuan -/
def selling_price : ℝ := 2080

theorem appliance_pricing :
  cost_price * (1 + markup) * (1 - discount) = selling_price := by sorry

end NUMINAMATH_CALUDE_appliance_pricing_l2567_256711


namespace NUMINAMATH_CALUDE_maria_trip_portion_l2567_256758

theorem maria_trip_portion (total_distance : ℝ) (first_stop_fraction : ℝ) (remaining_distance : ℝ)
  (h1 : total_distance = 560)
  (h2 : first_stop_fraction = 1 / 2)
  (h3 : remaining_distance = 210) :
  (total_distance * (1 - first_stop_fraction) - remaining_distance) / (total_distance * (1 - first_stop_fraction)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_maria_trip_portion_l2567_256758


namespace NUMINAMATH_CALUDE_first_part_count_l2567_256766

theorem first_part_count (total_count : Nat) (total_avg : Nat) (first_avg : Nat) (last_avg : Nat) (thirteenth_result : Nat) :
  total_count = 25 →
  total_avg = 18 →
  first_avg = 10 →
  last_avg = 20 →
  thirteenth_result = 90 →
  ∃ n : Nat, n = 14 ∧ 
    n * first_avg + thirteenth_result + (total_count - n) * last_avg = total_count * total_avg :=
by sorry

end NUMINAMATH_CALUDE_first_part_count_l2567_256766


namespace NUMINAMATH_CALUDE_power_calculation_l2567_256777

theorem power_calculation : 27^3 * 9^2 / 3^15 = (1 : ℚ) / 9 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l2567_256777


namespace NUMINAMATH_CALUDE_parallel_vectors_implies_y_eq_neg_four_l2567_256731

/-- Two vectors in ℝ² -/
def a : Fin 2 → ℝ := ![1, 2]
def b (y : ℝ) : Fin 2 → ℝ := ![-2, y]

/-- Parallel vectors in ℝ² have proportional coordinates -/
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ i, v i = k * u i

/-- If a and b are parallel plane vectors, then y = -4 -/
theorem parallel_vectors_implies_y_eq_neg_four :
  parallel a (b y) → y = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_implies_y_eq_neg_four_l2567_256731


namespace NUMINAMATH_CALUDE_ant_path_impossibility_l2567_256759

/-- Represents a vertex of a cube --/
inductive Vertex
| V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8

/-- Represents the label of a vertex (+1 or -1) --/
def vertexLabel (v : Vertex) : Int :=
  match v with
  | Vertex.V1 | Vertex.V3 | Vertex.V6 | Vertex.V8 => 1
  | Vertex.V2 | Vertex.V4 | Vertex.V5 | Vertex.V7 => -1

/-- Represents a path of an ant on the cube --/
def AntPath := List Vertex

/-- Checks if the path is valid (no backtracking) --/
def isValidPath (path : AntPath) : Prop :=
  sorry

/-- Counts the number of visits to each vertex --/
def countVisits (path : AntPath) : Vertex → Nat :=
  sorry

/-- The main theorem to prove --/
theorem ant_path_impossibility :
  ¬ ∃ (path : AntPath),
    isValidPath path ∧
    (∃ (v : Vertex),
      countVisits path v = 25 ∧
      ∀ (w : Vertex), w ≠ v → countVisits path w = 20) :=
sorry

end NUMINAMATH_CALUDE_ant_path_impossibility_l2567_256759


namespace NUMINAMATH_CALUDE_arrangement_counts_l2567_256772

def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def total_people : ℕ := number_of_boys + number_of_girls

theorem arrangement_counts :
  (∃ (arrange_four : ℕ) (arrange_two_rows : ℕ) (girls_together : ℕ) 
      (boys_not_adjacent : ℕ) (a_not_ends : ℕ) (a_not_left_b_not_right : ℕ) 
      (a_b_c_order : ℕ),
    arrange_four = 120 ∧
    arrange_two_rows = 120 ∧
    girls_together = 36 ∧
    boys_not_adjacent = 72 ∧
    a_not_ends = 72 ∧
    a_not_left_b_not_right = 78 ∧
    a_b_c_order = 20) :=
by
  sorry


end NUMINAMATH_CALUDE_arrangement_counts_l2567_256772


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2567_256705

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (a^2 - 2*a) + (a^2 - a - 2)*I
  (z.re = 0) → (a = 0 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2567_256705


namespace NUMINAMATH_CALUDE_square_root_three_squared_l2567_256790

theorem square_root_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_three_squared_l2567_256790


namespace NUMINAMATH_CALUDE_largest_circle_area_l2567_256707

theorem largest_circle_area (playground_area : Real) (π : Real) : 
  playground_area = 400 → π = 3.1 → 
  (π * (Real.sqrt playground_area / 2)^2 : Real) = 310 := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_area_l2567_256707


namespace NUMINAMATH_CALUDE_equation_solution_l2567_256768

theorem equation_solution (x : ℚ) :
  x ≠ 2/3 →
  ((3*x + 2) / (3*x^2 + 4*x - 4) = 3*x / (3*x - 2)) ↔ (x = 1/3 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2567_256768


namespace NUMINAMATH_CALUDE_expression_evaluation_l2567_256732

theorem expression_evaluation :
  let x : ℚ := -3
  let y : ℚ := 1/5
  (2*x + y)^2 - (x + 2*y)*(x - 2*y) - (3*x - y)*(x - 5*y) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2567_256732


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l2567_256771

/-- The number of students in total -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students out of 5 -/
theorem probability_of_selecting_A_and_B :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students : ℚ) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l2567_256771


namespace NUMINAMATH_CALUDE_other_number_proof_l2567_256748

/-- Given two positive integers with known HCF, LCM, and one of the numbers, prove the value of the other number -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 176 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2567_256748


namespace NUMINAMATH_CALUDE_cleos_marbles_eq_15_l2567_256713

/-- The number of marbles Cleo has on the third day -/
def cleos_marbles : ℕ :=
  let initial_marbles : ℕ := 30
  let marbles_taken_day2 : ℕ := (3 * initial_marbles) / 5
  let marbles_each_day2 : ℕ := marbles_taken_day2 / 2
  let marbles_remaining_day2 : ℕ := initial_marbles - marbles_taken_day2
  let marbles_taken_day3 : ℕ := marbles_remaining_day2 / 2
  marbles_each_day2 + marbles_taken_day3

theorem cleos_marbles_eq_15 : cleos_marbles = 15 := by
  sorry

end NUMINAMATH_CALUDE_cleos_marbles_eq_15_l2567_256713


namespace NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l2567_256702

theorem no_solutions_lcm_gcd_equation : 
  ¬∃ (n : ℕ+), Nat.lcm n 120 = Nat.gcd n 120 + 360 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l2567_256702


namespace NUMINAMATH_CALUDE_division_value_proof_l2567_256743

theorem division_value_proof (x : ℝ) : (2.25 / x) * 12 = 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_value_proof_l2567_256743


namespace NUMINAMATH_CALUDE_sport_water_amount_l2567_256737

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 4 * standard_ratio.flavoring / standard_ratio.corn_syrup,
    water := 2 * standard_ratio.water / standard_ratio.flavoring }

/-- Amount of corn syrup in the sport formulation bottle (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- Theorem stating the amount of water in the sport formulation bottle -/
theorem sport_water_amount :
  (sport_ratio.water / sport_ratio.corn_syrup) * sport_corn_syrup = 105 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l2567_256737


namespace NUMINAMATH_CALUDE_exists_k_for_circle_through_E_l2567_256704

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line equation -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

/-- The fixed point E -/
def point_E : ℝ × ℝ := (-1, 0)

/-- Predicate to check if a circle with CD as diameter passes through E -/
def circle_passes_through_E (C D : ℝ × ℝ) : Prop :=
  let (x1, y1) := C
  let (x2, y2) := D
  y1 / (x1 + 1) * y2 / (x2 + 1) = -1

/-- The main theorem -/
theorem exists_k_for_circle_through_E :
  ∃ k : ℝ, k ≠ 0 ∧ k = 7/6 ∧
  ∃ C D : ℝ × ℝ,
    ellipse C.1 C.2 ∧
    ellipse D.1 D.2 ∧
    line k C.1 C.2 ∧
    line k D.1 D.2 ∧
    circle_passes_through_E C D :=
sorry

end NUMINAMATH_CALUDE_exists_k_for_circle_through_E_l2567_256704


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l2567_256726

theorem dormitory_to_city_distance :
  ∀ D : ℝ,
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 4 = D →
  D = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l2567_256726


namespace NUMINAMATH_CALUDE_taco_castle_parking_lot_l2567_256770

theorem taco_castle_parking_lot : 
  ∀ (volkswagen ford toyota dodge : ℕ),
    volkswagen = 5 →
    toyota = 2 * volkswagen →
    ford = 2 * toyota →
    3 * ford = dodge →
    dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_taco_castle_parking_lot_l2567_256770


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2567_256798

-- Define the sets A and S
def A : Set ℝ := {x | -7 ≤ 2*x - 5 ∧ 2*x - 5 ≤ 9}
def S (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2*k - 1}

-- Statement 1
theorem subset_condition (k : ℝ) : 
  (S k).Nonempty ∧ S k ⊆ A ↔ 2 ≤ k ∧ k ≤ 4 := by sorry

-- Statement 2
theorem disjoint_condition (k : ℝ) : 
  A ∩ S k = ∅ ↔ k < 2 ∨ k > 6 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2567_256798


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2567_256730

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h : a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) ≤ 1) : 
  1 / (b + c + 1) + 1 / (c + a + 1) + 1 / (a + b + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2567_256730


namespace NUMINAMATH_CALUDE_long_jump_ratio_l2567_256742

/-- Given the conditions of a long jump event, prove the ratio of Margarita's jump to Ricciana's jump -/
theorem long_jump_ratio (ricciana_run : ℕ) (ricciana_jump : ℕ) (margarita_run : ℕ) (total_difference : ℕ) :
  ricciana_run = 20 →
  ricciana_jump = 4 →
  margarita_run = 18 →
  total_difference = 1 →
  (margarita_run + (ricciana_run + ricciana_jump + total_difference - margarita_run)) / ricciana_jump = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_ratio_l2567_256742


namespace NUMINAMATH_CALUDE_sum_of_edge_lengths_pyramid_volume_l2567_256797

-- Define the pyramid
def pyramid_base_side : ℝ := 8
def pyramid_height : ℝ := 15

-- Theorem for the sum of edge lengths
theorem sum_of_edge_lengths :
  let diagonal := pyramid_base_side * Real.sqrt 2
  let slant_edge := Real.sqrt (pyramid_height^2 + (diagonal / 2)^2)
  4 * pyramid_base_side + 4 * slant_edge = 32 + 4 * Real.sqrt 257 :=
sorry

-- Theorem for the volume
theorem pyramid_volume :
  (1 / 3) * pyramid_base_side^2 * pyramid_height = 320 :=
sorry

end NUMINAMATH_CALUDE_sum_of_edge_lengths_pyramid_volume_l2567_256797


namespace NUMINAMATH_CALUDE_isosceles_base_angle_l2567_256794

-- Define an isosceles triangle with a 30° vertex angle
def IsoscelesTriangle (α β γ : ℝ) : Prop :=
  α = 30 ∧ β = γ ∧ α + β + γ = 180

-- Theorem: In an isosceles triangle with a 30° vertex angle, each base angle is 75°
theorem isosceles_base_angle (α β γ : ℝ) (h : IsoscelesTriangle α β γ) : β = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_base_angle_l2567_256794


namespace NUMINAMATH_CALUDE_equilateral_triangle_quadratic_ac_l2567_256700

/-- A quadratic function f(x) = ax^2 + c whose graph intersects the coordinate axes 
    at the vertices of an equilateral triangle. -/
structure EquilateralTriangleQuadratic where
  a : ℝ
  c : ℝ
  is_equilateral : ∀ (x y : ℝ), y = a * x^2 + c → 
    (x = 0 ∨ y = 0) → 
    -- The three intersection points form an equilateral triangle
    ∃ (p q r : ℝ × ℝ), 
      (p.1 = 0 ∨ p.2 = 0) ∧ 
      (q.1 = 0 ∨ q.2 = 0) ∧ 
      (r.1 = 0 ∨ r.2 = 0) ∧
      (p.2 = a * p.1^2 + c) ∧
      (q.2 = a * q.1^2 + c) ∧
      (r.2 = a * r.1^2 + c) ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = (q.1 - r.1)^2 + (q.2 - r.2)^2 ∧
      (q.1 - r.1)^2 + (q.2 - r.2)^2 = (r.1 - p.1)^2 + (r.2 - p.2)^2

/-- The product of a and c for an EquilateralTriangleQuadratic is -3. -/
theorem equilateral_triangle_quadratic_ac (f : EquilateralTriangleQuadratic) : 
  f.a * f.c = -3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_quadratic_ac_l2567_256700


namespace NUMINAMATH_CALUDE_friction_negative_work_on_slope_l2567_256719

/-- A slope-block system where a block slides down a slope -/
structure SlopeBlockSystem where
  M : ℝ  -- Mass of the slope
  m : ℝ  -- Mass of the block
  μ : ℝ  -- Coefficient of friction between block and slope
  θ : ℝ  -- Angle of the slope
  g : ℝ  -- Acceleration due to gravity

/-- The horizontal surface is smooth -/
def is_smooth_surface (system : SlopeBlockSystem) : Prop :=
  sorry

/-- The block is released from rest at the top of the slope -/
def block_released_from_rest (system : SlopeBlockSystem) : Prop :=
  sorry

/-- The friction force does negative work on the slope -/
def friction_does_negative_work (system : SlopeBlockSystem) : Prop :=
  sorry

/-- Main theorem: The friction force of the block on the slope does negative work on the slope -/
theorem friction_negative_work_on_slope (system : SlopeBlockSystem) 
  (h1 : system.M > 0) 
  (h2 : system.m > 0) 
  (h3 : system.μ > 0) 
  (h4 : system.θ > 0) 
  (h5 : system.g > 0) 
  (h6 : is_smooth_surface system) 
  (h7 : block_released_from_rest system) : 
  friction_does_negative_work system :=
sorry

end NUMINAMATH_CALUDE_friction_negative_work_on_slope_l2567_256719


namespace NUMINAMATH_CALUDE_kody_age_proof_l2567_256736

/-- Kody's current age -/
def kody_age : ℕ := 32

/-- Mohamed's current age -/
def mohamed_age : ℕ := 60

/-- The time difference between now and the past reference point -/
def years_passed : ℕ := 4

theorem kody_age_proof :
  (∃ (kody_past mohamed_past : ℕ),
    kody_past = mohamed_past / 2 ∧
    kody_past + years_passed = kody_age ∧
    mohamed_past + years_passed = mohamed_age) ∧
  mohamed_age = 2 * 30 →
  kody_age = 32 := by sorry

end NUMINAMATH_CALUDE_kody_age_proof_l2567_256736


namespace NUMINAMATH_CALUDE_stating_third_number_formula_l2567_256781

/-- 
Given a triangular array of positive odd numbers arranged as follows:
1
3  5
7  9  11
13 15 17 19
...
This function returns the third number from the left in the nth row.
-/
def thirdNumberInRow (n : ℕ) : ℕ :=
  n^2 - n + 5

/-- 
Theorem stating that for n ≥ 3, the third number from the left 
in the nth row of the described triangular array is n^2 - n + 5.
-/
theorem third_number_formula (n : ℕ) (h : n ≥ 3) : 
  thirdNumberInRow n = n^2 - n + 5 := by
  sorry

end NUMINAMATH_CALUDE_stating_third_number_formula_l2567_256781


namespace NUMINAMATH_CALUDE_symmetric_decreasing_property_l2567_256789

/-- A function f: ℝ → ℝ that is decreasing on (4, +∞) and symmetric about x = 4 -/
def SymmetricDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 4 ∧ y > x → f y < f x) ∧
  (∀ x, f (4 + x) = f (4 - x))

/-- Given a symmetric decreasing function f, prove that f(3) > f(6) -/
theorem symmetric_decreasing_property (f : ℝ → ℝ) 
  (h : SymmetricDecreasingFunction f) : f 3 > f 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_decreasing_property_l2567_256789


namespace NUMINAMATH_CALUDE_specific_plate_probability_l2567_256750

/-- The set of vowels used in Mathlandia license plates -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- The set of non-vowels used in Mathlandia license plates -/
def nonVowels : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z', '1'}

/-- The set of digits used in Mathlandia license plates -/
def digits : Finset Char := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

/-- A license plate in Mathlandia -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char
  fifth : Char

/-- The probability of a specific license plate occurring in Mathlandia -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (vowels.card * vowels.card * nonVowels.card * (nonVowels.card - 1) * digits.card)

/-- The specific license plate "AIE19" -/
def specificPlate : LicensePlate := ⟨'A', 'I', 'E', '1', '9'⟩

theorem specific_plate_probability :
  licensePlateProbability specificPlate = 1 / 105000 :=
sorry

end NUMINAMATH_CALUDE_specific_plate_probability_l2567_256750


namespace NUMINAMATH_CALUDE_tomato_harvest_ratio_l2567_256787

/-- Proves that the ratio of tomatoes harvested on Wednesday to Thursday is 2:1 --/
theorem tomato_harvest_ratio :
  ∀ (thursday_harvest : ℕ),
  400 + thursday_harvest + (700 + 700) = 2000 →
  (400 : ℚ) / thursday_harvest = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_harvest_ratio_l2567_256787


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_ratio_l2567_256754

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Given two regular tetrahedra where one is inscribed inside the other
    such that its vertices are at the midpoints of the edges of the larger tetrahedron,
    the ratio of their volumes is 1/8 -/
theorem inscribed_tetrahedron_volume_ratio
  (large : RegularTetrahedron) (small : RegularTetrahedron)
  (h : small.sideLength = large.sideLength / 2) :
  (small.sideLength ^ 3) / (large.sideLength ^ 3) = 1 / 8 := by
  sorry

#check inscribed_tetrahedron_volume_ratio

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_ratio_l2567_256754
