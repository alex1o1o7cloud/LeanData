import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l7_705

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 16 * x₁ - 18 = 0) → 
  (10 * x₂^2 + 16 * x₂ - 18 = 0) → 
  x₁^2 + x₂^2 = 244 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l7_705


namespace NUMINAMATH_CALUDE_guest_bedroom_area_l7_751

/-- Proves that the area of each guest bedroom is 200 sq ft given the specified conditions --/
theorem guest_bedroom_area
  (total_rent : ℝ)
  (rent_rate : ℝ)
  (master_area : ℝ)
  (common_area : ℝ)
  (h1 : total_rent = 3000)
  (h2 : rent_rate = 2)
  (h3 : master_area = 500)
  (h4 : common_area = 600)
  : ∃ (guest_bedroom_area : ℝ),
    guest_bedroom_area = 200 ∧
    total_rent / rent_rate = master_area + common_area + 2 * guest_bedroom_area :=
by sorry

end NUMINAMATH_CALUDE_guest_bedroom_area_l7_751


namespace NUMINAMATH_CALUDE_equation_solution_l7_764

theorem equation_solution : ∃ (y₁ y₂ y₃ : ℂ),
  y₁ = -Real.sqrt 3 ∧
  y₂ = -Real.sqrt 3 + Complex.I ∧
  y₃ = -Real.sqrt 3 - Complex.I ∧
  (∀ y : ℂ, (y^3 + 3*y^2*(Real.sqrt 3) + 9*y + 3*(Real.sqrt 3)) + (y + Real.sqrt 3) = 0 ↔ 
    y = y₁ ∨ y = y₂ ∨ y = y₃) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l7_764


namespace NUMINAMATH_CALUDE_tangent_parallel_range_l7_733

open Real

/-- The function f(x) = x(m - e^(-2x)) --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * (m - Real.exp (-2 * x))

/-- The derivative of f with respect to x --/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m - (1 - 2*x) * Real.exp (-2 * x)

/-- Theorem stating the range of m for which there exist two distinct points
    on the curve y = f(x) where the tangent lines are parallel to y = x --/
theorem tangent_parallel_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv m x₁ = 1 ∧ f_deriv m x₂ = 1) ↔ 
  (1 - Real.exp (-2) < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_range_l7_733


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l7_752

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) :
  y = (4 - 2 * x) / 3 := by
sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l7_752


namespace NUMINAMATH_CALUDE_inverse_composition_equals_one_third_l7_780

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x - 2) / 3

-- Theorem statement
theorem inverse_composition_equals_one_third :
  g_inv (g_inv 11) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_one_third_l7_780


namespace NUMINAMATH_CALUDE_marble_redistribution_l7_787

/-- Represents the number of marbles each person has -/
structure Marbles :=
  (dilan : ℕ)
  (martha : ℕ)
  (phillip : ℕ)
  (veronica : ℕ)

/-- The theorem statement -/
theorem marble_redistribution (initial : Marbles) (final : Marbles) :
  initial.dilan = 14 →
  initial.martha = 20 →
  initial.veronica = 7 →
  final.dilan = 15 →
  final.martha = 15 →
  final.phillip = 15 →
  final.veronica = 15 →
  initial.dilan + initial.martha + initial.phillip + initial.veronica =
  final.dilan + final.martha + final.phillip + final.veronica →
  initial.phillip = 19 := by
  sorry

end NUMINAMATH_CALUDE_marble_redistribution_l7_787


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l7_758

/-- Represents a hyperbola with equation x²/m - y²/6 = 1 -/
structure Hyperbola where
  m : ℝ
  eq : ∀ x y : ℝ, x^2 / m - y^2 / 6 = 1

/-- The focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ := 6

theorem hyperbola_m_value (h : Hyperbola) (hf : focal_distance h = 6) : h.m = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l7_758


namespace NUMINAMATH_CALUDE_intersection_A_B_zero_union_A_B_equals_A_l7_711

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: Intersection of A and B when a = 0
theorem intersection_A_B_zero : A ∩ B 0 = {x | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: Range of a for which A ∪ B = A
theorem union_A_B_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_zero_union_A_B_equals_A_l7_711


namespace NUMINAMATH_CALUDE_min_c_value_l7_759

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < c) (h3 : b = c - 1)
  (h4 : ∃! p : ℝ × ℝ, p.1^2 + p.2 = 2003 ∧ 
    p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|) :
  c.val ≥ 1006 ∧ ∃ (a' b' : ℕ+), a' < b' ∧ b' < 1006 ∧ b' = 1005 ∧
    ∃! p : ℝ × ℝ, p.1^2 + p.2 = 2003 ∧ 
      p.2 = |p.1 - a'.val| + |p.1 - b'.val| + |p.1 - 1006| := by
  sorry

end NUMINAMATH_CALUDE_min_c_value_l7_759


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l7_781

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (totalCans : ℕ) (canCapacity : SoupCan) (childrenFed : ℕ) : ℕ :=
  let cansUsedForChildren := (childrenFed + canCapacity.children - 1) / canCapacity.children
  let remainingCans := totalCans - cansUsedForChildren
  remainingCans * canCapacity.adults

/-- Theorem stating that given 10 cans of soup, where each can feeds 4 adults or 6 children,
    if 30 children are fed, the remaining soup can feed 20 adults -/
theorem soup_feeding_theorem (totalCans : ℕ) (canCapacity : SoupCan) (childrenFed : ℕ) :
  totalCans = 10 →
  canCapacity.adults = 4 →
  canCapacity.children = 6 →
  childrenFed = 30 →
  remainingAdults totalCans canCapacity childrenFed = 20 := by
  sorry

end NUMINAMATH_CALUDE_soup_feeding_theorem_l7_781


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l7_767

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a 1 + (n - 1) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (a 4 + 4)^2 = (a 2 + 2) * (a 6 + 6) → d = -1 := by
  sorry

#check arithmetic_sequence_common_difference

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l7_767


namespace NUMINAMATH_CALUDE_tan_135_degrees_l7_776

/-- Tangent of 135 degrees is -1 -/
theorem tan_135_degrees : Real.tan (135 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_degrees_l7_776


namespace NUMINAMATH_CALUDE_mistaken_addition_l7_720

theorem mistaken_addition (N : ℤ) : (41 - N = 12) → (41 + N = 70) := by
  sorry

end NUMINAMATH_CALUDE_mistaken_addition_l7_720


namespace NUMINAMATH_CALUDE_a_share_l7_726

/-- Represents the share of money for each person -/
structure Share where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The theorem stating A's share given the conditions -/
theorem a_share (s : Share) 
  (h1 : s.a = 5 * s.d ∧ s.b = 2 * s.d ∧ s.c = 4 * s.d) 
  (h2 : s.c = s.d + 500) : 
  s.a = 2500 := by
  sorry

end NUMINAMATH_CALUDE_a_share_l7_726


namespace NUMINAMATH_CALUDE_sin_2x_value_l7_756

theorem sin_2x_value (x : ℝ) : 
  (Real.cos (4 * π / 5) * Real.cos (7 * π / 15) - Real.sin (9 * π / 5) * Real.sin (7 * π / 15) = 
   Real.cos (x + π / 2) * Real.cos x + 2 / 3) → 
  Real.sin (2 * x) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l7_756


namespace NUMINAMATH_CALUDE_percentage_problem_l7_794

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.20 * 1000 - 30 → x = 680 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l7_794


namespace NUMINAMATH_CALUDE_store_earnings_proof_l7_755

def store_earnings : ℕ := by
  let graphics_cards := 10 * 600
  let hard_drives := 14 * 80
  let cpus := 8 * 200
  let ram := 4 * 60
  let power_supply_units := 12 * 90
  let monitors := 6 * 250
  let keyboards := 18 * 40
  let mice := 24 * 20
  exact graphics_cards + hard_drives + cpus + ram + power_supply_units + monitors + keyboards + mice

theorem store_earnings_proof : store_earnings = 12740 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_proof_l7_755


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l7_722

/-- An isosceles triangle with sides a, b, and c, where b = c -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : b = c
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, t.a = 3 → t.b = 6 → perimeter t = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l7_722


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l7_731

theorem simplify_radical_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l7_731


namespace NUMINAMATH_CALUDE_emma_skateboard_time_l7_716

/-- The time taken for Emma to skateboard along a looping path on a highway --/
theorem emma_skateboard_time : ∀ (highway_length highway_width emma_speed : ℝ),
  highway_length = 2 * 5280 →
  highway_width = 50 →
  emma_speed = 4 →
  ∃ (time : ℝ), time = π / 2 ∧ time * emma_speed = 2 * π :=
by
  sorry

end NUMINAMATH_CALUDE_emma_skateboard_time_l7_716


namespace NUMINAMATH_CALUDE_marble_combination_count_l7_735

def num_marbles_per_color : ℕ := 2
def num_colors : ℕ := 4
def total_marbles : ℕ := num_marbles_per_color * num_colors

def choose_two_same_color : ℕ := num_colors * (num_marbles_per_color.choose 2)
def choose_two_diff_colors : ℕ := (num_colors.choose 2) * num_marbles_per_color * num_marbles_per_color

theorem marble_combination_count :
  choose_two_same_color + choose_two_diff_colors = 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_combination_count_l7_735


namespace NUMINAMATH_CALUDE_training_trip_duration_l7_772

/-- The number of supervisors --/
def n : ℕ := 15

/-- The number of supervisors overseeing the pool each day --/
def k : ℕ := 3

/-- The number of ways to choose 2 supervisors from n supervisors --/
def total_pairs : ℕ := n.choose 2

/-- The number of pairs formed each day --/
def pairs_per_day : ℕ := k.choose 2

/-- The number of days required for the training trip --/
def days : ℕ := total_pairs / pairs_per_day

theorem training_trip_duration :
  n = 15 → k = 3 → days = 35 := by sorry

end NUMINAMATH_CALUDE_training_trip_duration_l7_772


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_eq_6_l7_791

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (a b c d e f : ℝ), a ≠ 0 ∧ d ≠ 0 ∧ 
    (∀ x y, a*x + b*y + c = 0 ↔ y = m₁*x + (c/b)) ∧
    (∀ x y, d*x + e*y + f = 0 ↔ y = m₂*x + (f/e)))

/-- The theorem to be proved -/
theorem perpendicular_lines_m_eq_6 :
  ∀ (m : ℝ), (∀ x y, x - 2*y - 3 = 0 ↔ y = (1/2)*x - 3/2) →
             (∀ x y, m*x + 3*y - 6 = 0 ↔ y = (-m/3)*x + 2) →
             (∃ (m₁ m₂ : ℝ), m₁ * m₂ = -1 ∧ 
               (∀ x y, x - 2*y - 3 = 0 ↔ y = m₁*x - 3/2) ∧
               (∀ x y, m*x + 3*y - 6 = 0 ↔ y = m₂*x + 2)) →
             m = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_eq_6_l7_791


namespace NUMINAMATH_CALUDE_sticker_theorem_l7_773

def sticker_problem (initial_stickers : ℕ) (stickers_per_friend : ℕ) (num_friends : ℕ) 
  (remaining_stickers : ℕ) (justin_diff : ℕ) : Prop :=
  let total_to_friends := stickers_per_friend * num_friends
  let total_given_away := initial_stickers - remaining_stickers
  let mandy_and_justin := total_given_away - total_to_friends
  let mandy_stickers := (mandy_and_justin + justin_diff) / 2
  mandy_stickers - total_to_friends = 2

theorem sticker_theorem : 
  sticker_problem 72 4 3 42 10 := by sorry

end NUMINAMATH_CALUDE_sticker_theorem_l7_773


namespace NUMINAMATH_CALUDE_light_ray_reflection_l7_734

/-- A light ray reflection problem -/
theorem light_ray_reflection 
  (M : ℝ × ℝ) 
  (N : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) : 
  M = (2, 6) → 
  N = (-3, 4) → 
  (∀ x y, l x y ↔ x - y + 3 = 0) → 
  ∃ A B C : ℝ, 
    (∀ x y, A * x + B * y + C = 0 ↔ 
      (∃ K : ℝ × ℝ, 
        -- K is symmetric to M with respect to l
        (K.1 - M.1) / (K.2 - M.2) = -1 ∧ 
        l ((K.1 + M.1) / 2) ((K.2 + M.2) / 2) ∧
        -- N lies on the line through K
        (N.2 - K.2) / (N.1 - K.1) = (y - K.2) / (x - K.1))) ∧
    A = 1 ∧ B = -6 ∧ C = 27 :=
by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_l7_734


namespace NUMINAMATH_CALUDE_parallel_lines_parallelograms_l7_736

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of parallelograms formed by intersecting sets of parallel lines -/
def parallelograms_count (set1 : ℕ) (set2 : ℕ) : ℕ :=
  (choose_two set1) * (choose_two set2)

theorem parallel_lines_parallelograms :
  parallelograms_count 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_parallelograms_l7_736


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l7_707

/-- 
Given a point on a number line that:
1. Starts at position -2
2. Moves 8 units to the right
3. Moves 4 units to the left
This theorem proves that the final position of the point is 2.
-/
theorem point_movement_on_number_line : 
  let start_position : ℤ := -2
  let right_movement : ℤ := 8
  let left_movement : ℤ := 4
  let final_position := start_position + right_movement - left_movement
  final_position = 2 := by sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l7_707


namespace NUMINAMATH_CALUDE_thirty_six_hundredths_decimal_l7_786

theorem thirty_six_hundredths_decimal : (36 : ℚ) / 100 = 0.36 := by sorry

end NUMINAMATH_CALUDE_thirty_six_hundredths_decimal_l7_786


namespace NUMINAMATH_CALUDE_change_in_cubic_expression_l7_757

theorem change_in_cubic_expression (x a : ℝ) (ha : a > 0) :
  abs ((x + a)^3 - 3*(x + a) - (x^3 - 3*x)) = 3*a*x^2 + 3*a^2*x + a^3 - 3*a ∧
  abs ((x - a)^3 - 3*(x - a) - (x^3 - 3*x)) = 3*a*x^2 + 3*a^2*x + a^3 - 3*a :=
by sorry

end NUMINAMATH_CALUDE_change_in_cubic_expression_l7_757


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_sqrt_152_is_solution_sqrt_152_is_smallest_solution_l7_746

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 23) → x ≥ Real.sqrt 152 :=
by sorry

theorem sqrt_152_is_solution :
  ⌊(Real.sqrt 152)^2⌋ - ⌊Real.sqrt 152⌋^2 = 23 :=
by sorry

theorem sqrt_152_is_smallest_solution :
  ∀ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 23) → x ≥ Real.sqrt 152 ∧
  ⌊(Real.sqrt 152)^2⌋ - ⌊Real.sqrt 152⌋^2 = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_sqrt_152_is_solution_sqrt_152_is_smallest_solution_l7_746


namespace NUMINAMATH_CALUDE_circle_product_theorem_l7_796

/-- A circular permutation of five elements -/
def CircularPerm (α : Type) := Fin 5 → α

/-- The condition for the first part of the problem -/
def FirstCondition (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
  a + b + c + d + e = 1 ∧
  ∀ π : CircularPerm ℝ, π 0 = a ∧ π 1 = b ∧ π 2 = c ∧ π 3 = d ∧ π 4 = e →
    ∃ i : Fin 5, π i * π ((i + 1) % 5) ≥ 1/9

/-- The condition for the second part of the problem -/
def SecondCondition (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
  a + b + c + d + e = 1

/-- The theorem statement combining both parts of the problem -/
theorem circle_product_theorem :
  (∃ a b c d e : ℝ, FirstCondition a b c d e) ∧
  (∀ a b c d e : ℝ, SecondCondition a b c d e →
    ∃ π : CircularPerm ℝ, π 0 = a ∧ π 1 = b ∧ π 2 = c ∧ π 3 = d ∧ π 4 = e ∧
      ∀ i : Fin 5, π i * π ((i + 1) % 5) ≤ 1/9) :=
by sorry

end NUMINAMATH_CALUDE_circle_product_theorem_l7_796


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l7_779

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem: The center coordinates of the circle x^2 + y^2 + 2x - 4y = 0 are (-1, 2) -/
theorem circle_center_coordinates :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l7_779


namespace NUMINAMATH_CALUDE_factorization_proof_l7_715

theorem factorization_proof (x : ℝ) : 
  (2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4)) ∧ 
  (x^2 - 14 * x + 49 = (x - 7)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l7_715


namespace NUMINAMATH_CALUDE_equal_distance_to_axes_l7_728

theorem equal_distance_to_axes (m : ℝ) : 
  let M : ℝ × ℝ := (-3*m - 1, -2*m)
  (|M.1| = |M.2|) ↔ (m = -1/5 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_equal_distance_to_axes_l7_728


namespace NUMINAMATH_CALUDE_line_parameterization_values_l7_789

/-- A line parameterized by a point and a direction vector -/
structure ParametricLine (α : Type*) [Field α] where
  point : α × α
  direction : α × α

/-- The equation of a line in slope-intercept form -/
structure LineEquation (α : Type*) [Field α] where
  slope : α
  intercept : α

/-- Check if a point lies on a line given by slope-intercept equation -/
def LineEquation.contains_point {α : Type*} [Field α] (l : LineEquation α) (p : α × α) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- Check if a parametric line is equivalent to a line equation -/
def parametric_line_equiv_equation {α : Type*} [Field α] 
  (pl : ParametricLine α) (le : LineEquation α) : Prop :=
  ∀ t : α, le.contains_point (pl.point.1 + t * pl.direction.1, pl.point.2 + t * pl.direction.2)

theorem line_parameterization_values 
  (l : LineEquation ℝ) 
  (pl : ParametricLine ℝ) 
  (h_equiv : parametric_line_equiv_equation pl l) 
  (h_slope : l.slope = 2) 
  (h_intercept : l.intercept = -7) 
  (h_point : pl.point = (s, 2)) 
  (h_direction : pl.direction = (3, m)) : 
  s = 9/2 ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_values_l7_789


namespace NUMINAMATH_CALUDE_problem_solution_l7_738

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem problem_solution :
  (∀ m : ℝ, m = 4 → A ∪ B m = {x | -2 ≤ x ∧ x ≤ 7}) ∧
  (∀ m : ℝ, (B m ∩ A = B m) ↔ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l7_738


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l7_784

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 20)
  (h2 : Complex.abs (z + 2 * w) = 10)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l7_784


namespace NUMINAMATH_CALUDE_min_value_function_l7_775

theorem min_value_function (x : ℝ) (hx : x > 1) : 
  let m := (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2)
  let y := 2 * m * x + 3 / (x - 1) + 1
  y ≥ 2 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_function_l7_775


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l7_754

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 7b₃ is -9/14. -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) : 
  b₁ = 2 → 
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = 2 * r' ∧ b₃' = b₂' * r') → 
    3 * b₂ + 7 * b₃ ≤ 3 * b₂' + 7 * b₃') → 
  3 * b₂ + 7 * b₃ = -9/14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l7_754


namespace NUMINAMATH_CALUDE_unique_solution_l7_774

-- Define the range of numbers
def valid_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 50

-- Define primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the conditions of the problem
structure DrawResult where
  alice : ℕ
  bob : ℕ
  alice_valid : valid_number alice
  bob_valid : valid_number bob
  alice_uncertain : ∀ n, valid_number n → n ≠ alice → (n < alice ∨ n > alice)
  bob_certain : bob < alice ∨ bob > alice
  bob_prime : is_prime bob
  product_multiple_of_10 : (alice * bob) % 10 = 0
  perfect_square : ∃ k : ℕ, 100 * bob + alice = k * k

-- Theorem statement
theorem unique_solution (d : DrawResult) : d.alice = 29 ∧ d.bob = 5 ∧ d.alice + d.bob = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l7_774


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l7_769

theorem last_digit_of_one_over_two_to_twelve (n : ℕ) : 
  n = 12 → (1 : ℚ) / (2 ^ n) * 10^n % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l7_769


namespace NUMINAMATH_CALUDE_twelve_balloons_floated_away_l7_706

/-- Calculates the number of balloons that floated away -/
def balloons_floated_away (initial_count : ℕ) (given_away : ℕ) (grabbed : ℕ) (final_count : ℕ) : ℕ :=
  initial_count - given_away + grabbed - final_count

/-- Proves that 12 balloons floated away given the problem conditions -/
theorem twelve_balloons_floated_away :
  balloons_floated_away 50 10 11 39 = 12 := by
  sorry

#eval balloons_floated_away 50 10 11 39

end NUMINAMATH_CALUDE_twelve_balloons_floated_away_l7_706


namespace NUMINAMATH_CALUDE_f_symmetry_l7_744

/-- A cubic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

/-- Theorem: If f(-2) = 0, then f(2) = 2 -/
theorem f_symmetry (a b : ℝ) (h : f a b (-2) = 0) : f a b 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l7_744


namespace NUMINAMATH_CALUDE_largest_valid_number_l7_743

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 = 1 ∨ n / 100 = 7 ∨ n / 100 = 0) ∧
  ((n / 10) % 10 = 1 ∨ (n / 10) % 10 = 7 ∨ (n / 10) % 10 = 0) ∧
  (n % 10 = 1 ∨ n % 10 = 7 ∨ n % 10 = 0) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 710 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l7_743


namespace NUMINAMATH_CALUDE_human_genome_project_satisfies_conditions_l7_710

/-- Represents a scientific plan --/
structure ScientificPlan where
  name : String
  launchYear : Nat
  participatingCountries : List String
  isMajorPlan : Bool

/-- The Human Genome Project --/
def humanGenomeProject : ScientificPlan := {
  name := "Human Genome Project",
  launchYear := 1990,
  participatingCountries := ["United States", "United Kingdom", "France", "Germany", "Japan", "China"],
  isMajorPlan := true
}

/-- The Manhattan Project --/
def manhattanProject : ScientificPlan := {
  name := "Manhattan Project",
  launchYear := 1942,
  participatingCountries := ["United States", "United Kingdom", "Canada"],
  isMajorPlan := true
}

/-- The Apollo Program --/
def apolloProgram : ScientificPlan := {
  name := "Apollo Program",
  launchYear := 1961,
  participatingCountries := ["United States"],
  isMajorPlan := true
}

/-- The set of "Three Major Scientific Plans" --/
def threeMajorPlans : List ScientificPlan := [humanGenomeProject, manhattanProject, apolloProgram]

/-- Theorem stating that the Human Genome Project satisfies all conditions --/
theorem human_genome_project_satisfies_conditions :
  humanGenomeProject.launchYear = 1990 ∧
  humanGenomeProject.participatingCountries = ["United States", "United Kingdom", "France", "Germany", "Japan", "China"] ∧
  humanGenomeProject ∈ threeMajorPlans := by
  sorry


end NUMINAMATH_CALUDE_human_genome_project_satisfies_conditions_l7_710


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l7_793

theorem greatest_integer_radius (A : ℝ) (h : A < 90 * Real.pi) : 
  ∀ r : ℕ, r * r * Real.pi ≤ A → r ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l7_793


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l7_778

def total_spent (robert_pens julia_pens_multiplier dorothy_pens_divisor : ℕ)
                (pen_cost : ℚ)
                (robert_pencils julia_pencils_difference dorothy_pencils_multiplier : ℕ)
                (pencil_cost : ℚ)
                (dorothy_notebooks julia_notebooks_addition robert_notebooks_divisor : ℕ)
                (notebook_cost : ℚ) : ℚ :=
  let julia_pens := julia_pens_multiplier * robert_pens
  let dorothy_pens := julia_pens / dorothy_pens_divisor
  let total_pens := robert_pens + julia_pens + dorothy_pens
  let pens_cost := (total_pens : ℚ) * pen_cost

  let julia_pencils := robert_pencils - julia_pencils_difference
  let dorothy_pencils := dorothy_pencils_multiplier * julia_pencils
  let total_pencils := robert_pencils + julia_pencils + dorothy_pencils
  let pencils_cost := (total_pencils : ℚ) * pencil_cost

  let julia_notebooks := dorothy_notebooks + julia_notebooks_addition
  let robert_notebooks := julia_notebooks / robert_notebooks_divisor
  let total_notebooks := dorothy_notebooks + julia_notebooks + robert_notebooks
  let notebooks_cost := (total_notebooks : ℚ) * notebook_cost

  pens_cost + pencils_cost + notebooks_cost

theorem total_spent_is_correct :
  total_spent 4 3 2 (3/2) 12 5 2 (3/4) 3 1 2 4 = 93.75 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l7_778


namespace NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l7_762

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem sum_of_3rd_4th_5th_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_common_ratio : q = 2)
  (h_sum_first_3 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l7_762


namespace NUMINAMATH_CALUDE_photo_exhibition_total_l7_727

/-- Represents the number of photographs in various categories -/
structure PhotoExhibition where
  octavia_total : ℕ  -- Total photos taken by Octavia
  jack_octavia : ℕ   -- Photos taken by Octavia and framed by Jack
  jack_others : ℕ    -- Photos taken by others and framed by Jack

/-- Theorem stating the total number of photos either framed by Jack or taken by Octavia -/
theorem photo_exhibition_total (e : PhotoExhibition) 
  (h1 : e.octavia_total = 36)
  (h2 : e.jack_octavia = 24)
  (h3 : e.jack_others = 12) : 
  e.octavia_total + e.jack_others = 48 := by
  sorry


end NUMINAMATH_CALUDE_photo_exhibition_total_l7_727


namespace NUMINAMATH_CALUDE_modular_equivalence_problem_l7_745

theorem modular_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -315 ≡ n [ZMOD 23] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_modular_equivalence_problem_l7_745


namespace NUMINAMATH_CALUDE_inscribed_triangle_relation_l7_713

-- Define a triangle inscribed in a unit circle
structure InscribedTriangle where
  a : Real
  b : Real
  c : Real
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi
  side_a : a = 2 * Real.sin (α / 2)
  side_b : b = 2 * Real.sin (β / 2)
  side_c : c = 2 * Real.sin (γ / 2)

-- Theorem statement
theorem inscribed_triangle_relation (t : InscribedTriangle) :
  t.a^2 + t.b^2 + t.c^2 = 8 + 4 * Real.cos t.α * Real.cos t.β * Real.cos t.γ := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_relation_l7_713


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l7_714

theorem quadratic_equation_properties : ∃ (x y : ℝ),
  x^2 + 1984513*x + 3154891 = 0 ∧
  y^2 + 1984513*y + 3154891 = 0 ∧
  x ≠ y ∧
  (∀ z : ℤ, z^2 + 1984513*z + 3154891 ≠ 0) ∧
  x ≤ 0 ∧
  y ≤ 0 ∧
  1/x + 1/y ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l7_714


namespace NUMINAMATH_CALUDE_binomial_identities_l7_753

theorem binomial_identities (n k : ℕ) (h : k ≤ n) :
  (n.factorial = n.choose k * k.factorial * (n - k).factorial) ∧
  (n.choose k = (n - 1).choose k + (n - 1).choose (k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identities_l7_753


namespace NUMINAMATH_CALUDE_tank_capacity_l7_721

theorem tank_capacity (T : ℚ) 
  (h1 : 2 / 5 * T + 15 + 20 = T - 25) : T = 100 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l7_721


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l7_750

-- Define the constants
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours_worked : ℕ := 44

-- Define the functions
def overtime_rate : ℝ := regular_rate * (1 + overtime_rate_increase)

def calculate_compensation (hours : ℕ) : ℝ :=
  if hours ≤ regular_hours then
    hours * regular_rate
  else
    regular_hours * regular_rate + (hours - regular_hours) * overtime_rate

-- Theorem to prove
theorem bus_driver_compensation :
  calculate_compensation total_hours_worked = 752 :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l7_750


namespace NUMINAMATH_CALUDE_broken_line_length_bound_l7_798

/-- A broken line is represented as a list of points -/
def BrokenLine := List ℝ × ℝ

/-- A rectangle is represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Predicate to check if a broken line is inside a rectangle -/
def isInside (bl : BrokenLine) (rect : Rectangle) : Prop := sorry

/-- Predicate to check if every line parallel to the sides of the rectangle
    intersects the broken line at most once -/
def intersectsAtMostOnce (bl : BrokenLine) (rect : Rectangle) : Prop := sorry

/-- Function to calculate the length of a broken line -/
def length (bl : BrokenLine) : ℝ := sorry

/-- Theorem: If a broken line is inside a rectangle and every line parallel to the sides
    of the rectangle intersects the broken line at most once, then the length of the
    broken line is less than the sum of the lengths of two adjacent sides of the rectangle -/
theorem broken_line_length_bound (bl : BrokenLine) (rect : Rectangle) :
  isInside bl rect →
  intersectsAtMostOnce bl rect →
  length bl < rect.width + rect.height := by
  sorry

end NUMINAMATH_CALUDE_broken_line_length_bound_l7_798


namespace NUMINAMATH_CALUDE_correct_seat_notation_l7_790

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (seatNum : ℕ)

/-- Defines the notation for a seat -/
def seatNotation (s : Seat) : ℕ × ℕ := (s.row, s.seatNum)

theorem correct_seat_notation :
  let example_seat := Seat.mk 10 3
  let target_seat := Seat.mk 6 16
  (seatNotation example_seat = (10, 3)) →
  (seatNotation target_seat = (6, 16)) := by
  sorry

end NUMINAMATH_CALUDE_correct_seat_notation_l7_790


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l7_719

/-- An arithmetic progression with a_3 = 10 -/
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d ∧ a 3 = 10

/-- The sum of a_1, a_2, and a_6 in the arithmetic progression -/
def sum_terms (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 6

theorem arithmetic_progression_sum (a : ℕ → ℝ) :
  arithmetic_progression a → sum_terms a = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l7_719


namespace NUMINAMATH_CALUDE_sum_of_n_terms_l7_724

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- S_1, S_3, and S_2 form an arithmetic sequence -/
def S_arithmetic (S : ℕ → ℝ) : Prop :=
  S 3 - S 2 = S 2 - S 1

/-- a_1 - a_3 = 3 -/
def a_difference (a : ℕ → ℝ) : Prop :=
  a 1 - a 3 = 3

/-- Theorem: Sum of first n terms of the sequence -/
theorem sum_of_n_terms
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a S)
  (h2 : S_arithmetic S)
  (h3 : a_difference a) :
  ∀ n, S n = (8/3) * (1 - (-1/2)^n) :=
sorry

end NUMINAMATH_CALUDE_sum_of_n_terms_l7_724


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l7_730

theorem triangle_cosine_sum (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (b + c = 12) →  -- Given condition
  (C = 2 * π / 3) →  -- 120° in radians
  (Real.sin B = 5 * Real.sqrt 3 / 14) →
  (Real.cos A + Real.cos B = 12 / 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l7_730


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_l7_717

def adult_price : ℝ := 10

def grandparent_discount : ℝ := 0.2
def child_discount : ℝ := 0.6

def grandparent_price : ℝ := adult_price * (1 - grandparent_discount)
def child_price : ℝ := adult_price * (1 - child_discount)

def total_cost : ℝ := 2 * grandparent_price + adult_price + child_price

theorem zoo_ticket_cost : total_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_l7_717


namespace NUMINAMATH_CALUDE_only_non_algorithm_l7_768

/-- A process is a description of a task or method. -/
structure Process where
  description : String

/-- An algorithm is a process that has a sequence of defined steps. -/
structure Algorithm extends Process where
  has_defined_steps : Bool

/-- The property of having defined steps for a process. -/
def has_defined_steps (p : Process) : Prop :=
  ∃ (a : Algorithm), a.description = p.description

/-- The list of processes to be evaluated. -/
def processes : List Process :=
  [{ description := "The process of solving the equation 2x-6=0 involves moving terms and making the coefficient 1" },
   { description := "To get from Jinan to Vancouver, one must first take a train to Beijing, then transfer to a plane" },
   { description := "Solving the equation 2x^2+x-1=0" },
   { description := "Using the formula S=πr^2 to calculate the area of a circle with radius 3 involves computing π×3^2" }]

/-- The theorem stating that "Solving the equation 2x^2+x-1=0" is the only process without defined steps. -/
theorem only_non_algorithm :
  ∃! (p : Process), p ∈ processes ∧ ¬(has_defined_steps p) ∧
    p.description = "Solving the equation 2x^2+x-1=0" :=
  sorry

end NUMINAMATH_CALUDE_only_non_algorithm_l7_768


namespace NUMINAMATH_CALUDE_all_or_none_triangular_l7_788

/-- A polynomial of degree 4 -/
structure Polynomial4 where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ

/-- Evaluate the polynomial at a given x -/
def eval (poly : Polynomial4) (x : ℝ) : ℝ :=
  x^4 + poly.p * x^3 + poly.q * x^2 + poly.r * x + poly.s

/-- Represents four points on a horizontal line intersecting the curve -/
structure FourPoints where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  h₁ : x₁ < x₂
  h₂ : x₂ < x₃
  h₃ : x₃ < x₄

/-- Check if three lengths can form a triangle -/
def isTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of four points is triangular -/
def isTriangular (pts : FourPoints) : Prop :=
  isTriangle (pts.x₂ - pts.x₁) (pts.x₃ - pts.x₁) (pts.x₄ - pts.x₁)

/-- The main theorem -/
theorem all_or_none_triangular (poly : Polynomial4) :
  (∀ y : ℝ, ∀ pts : FourPoints, eval poly pts.x₁ = y ∧ eval poly pts.x₂ = y ∧
    eval poly pts.x₃ = y ∧ eval poly pts.x₄ = y → isTriangular pts) ∨
  (∀ y : ℝ, ∀ pts : FourPoints, eval poly pts.x₁ = y ∧ eval poly pts.x₂ = y ∧
    eval poly pts.x₃ = y ∧ eval poly pts.x₄ = y → ¬isTriangular pts) :=
sorry

end NUMINAMATH_CALUDE_all_or_none_triangular_l7_788


namespace NUMINAMATH_CALUDE_min_integer_value_is_seven_l7_704

def expression (parentheses : List (Nat × Nat)) : ℚ :=
  let nums := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  -- Define a function to evaluate the expression based on parentheses placement
  sorry

def is_valid_parentheses (parentheses : List (Nat × Nat)) : Prop :=
  -- Define a predicate to check if the parentheses placement is valid
  sorry

theorem min_integer_value_is_seven :
  ∃ (parentheses : List (Nat × Nat)),
    is_valid_parentheses parentheses ∧
    (expression parentheses).num = 7 ∧
    (expression parentheses).den = 1 ∧
    (∀ (other_parentheses : List (Nat × Nat)),
      is_valid_parentheses other_parentheses →
      (expression other_parentheses).num ≥ 7 ∨ (expression other_parentheses).den ≠ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_min_integer_value_is_seven_l7_704


namespace NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l7_763

theorem fourth_power_nested_sqrt : 
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 2)))^4 = 7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l7_763


namespace NUMINAMATH_CALUDE_determine_a_l7_703

-- Define positive integers a, b, c, and d
variable (a b c d : ℕ+)

-- Define the main theorem
theorem determine_a :
  (18^a.val * 9^(4*a.val - 1) * 27^c.val = 2^6 * 3^b.val * 7^d.val) →
  (a.val * c.val : ℚ) = 4 / (2*b.val + d.val) →
  b.val^2 - 4*a.val*c.val = d.val →
  a = 6 := by
  sorry


end NUMINAMATH_CALUDE_determine_a_l7_703


namespace NUMINAMATH_CALUDE_class_size_problem_l7_718

theorem class_size_problem :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 29 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l7_718


namespace NUMINAMATH_CALUDE_expression_evaluation_l7_771

theorem expression_evaluation :
  (5^1003 + 6^1002)^2 - (5^1003 - 6^1002)^2 = 600 * 30^1002 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l7_771


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l7_777

theorem certain_fraction_proof (x y : ℚ) : 
  x / y ≠ 0 → -- Ensure division by y is valid
  (x / y) / (1 / 5) = (3 / 4) / (2 / 5) →
  x / y = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l7_777


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l7_701

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_divisors_even (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → Even d

def count_prime_divisors (n : ℕ) : ℕ := (n.divisors.filter Nat.Prime).card

def count_composite_divisors (n : ℕ) : ℕ := (n.divisors.filter (λ d => ¬Nat.Prime d ∧ d ≠ 1)).card

theorem unique_number_with_properties : 
  ∃! n : ℕ, is_four_digit n ∧ 
            all_divisors_even n ∧ 
            count_prime_divisors n = 3 ∧ 
            count_composite_divisors n = 39 ∧
            n = 6336 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l7_701


namespace NUMINAMATH_CALUDE_reduced_price_is_six_l7_729

/-- Represents the price of apples and the quantity that can be purchased -/
structure ApplePricing where
  originalPrice : ℚ
  quantityBefore : ℚ
  quantityAfter : ℚ

/-- Calculates the reduced price per dozen apples -/
def reducedPricePerDozen (ap : ApplePricing) : ℚ :=
  6

/-- Theorem stating the reduced price per dozen apples is 6 rupees -/
theorem reduced_price_is_six (ap : ApplePricing) 
  (h1 : ap.quantityAfter = ap.quantityBefore + 50)
  (h2 : ap.quantityBefore * ap.originalPrice = 50)
  (h3 : ap.quantityAfter * (ap.originalPrice / 2) = 50) : 
  reducedPricePerDozen ap = 6 := by
  sorry

#check reduced_price_is_six

end NUMINAMATH_CALUDE_reduced_price_is_six_l7_729


namespace NUMINAMATH_CALUDE_total_population_proof_l7_732

def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

def greenville_population : ℕ := springfield_population - population_difference
def oakville_population : ℕ := 2 * population_difference

def total_population : ℕ := springfield_population + greenville_population + oakville_population

theorem total_population_proof : total_population = 1084972 := by
  sorry

end NUMINAMATH_CALUDE_total_population_proof_l7_732


namespace NUMINAMATH_CALUDE_calculation_result_l7_723

theorem calculation_result : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l7_723


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l7_765

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / (1 - Complex.I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l7_765


namespace NUMINAMATH_CALUDE_fraction_sum_division_simplification_l7_740

theorem fraction_sum_division_simplification :
  (3 : ℚ) / 7 + 5 / 8 + 1 / 3 / ((5 : ℚ) / 12 + 2 / 9) = 2097 / 966 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_division_simplification_l7_740


namespace NUMINAMATH_CALUDE_integral_of_special_function_l7_737

theorem integral_of_special_function (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = x^3 + x^2 * (deriv f 1)) : 
  ∫ x in (0:ℝ)..(2:ℝ), f x = -4 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_special_function_l7_737


namespace NUMINAMATH_CALUDE_electronic_components_production_ahead_of_schedule_l7_797

theorem electronic_components_production_ahead_of_schedule 
  (total_components : ℕ) 
  (planned_days : ℕ) 
  (additional_daily_production : ℕ) : 
  total_components = 15000 → 
  planned_days = 30 → 
  additional_daily_production = 250 → 
  (planned_days - (total_components / ((total_components / planned_days) + additional_daily_production))) = 10 := by
sorry

end NUMINAMATH_CALUDE_electronic_components_production_ahead_of_schedule_l7_797


namespace NUMINAMATH_CALUDE_probability_two_red_balls_probability_two_red_balls_is_5_22_l7_725

/-- The probability of picking two red balls from a bag containing 6 red balls, 4 blue balls, and 2 green balls when 2 balls are picked at random -/
theorem probability_two_red_balls : ℚ :=
  let total_balls : ℕ := 6 + 4 + 2
  let red_balls : ℕ := 6
  let prob_first_red : ℚ := red_balls / total_balls
  let prob_second_red : ℚ := (red_balls - 1) / (total_balls - 1)
  prob_first_red * prob_second_red

/-- Proof that the probability of picking two red balls is 5/22 -/
theorem probability_two_red_balls_is_5_22 : 
  probability_two_red_balls = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_probability_two_red_balls_is_5_22_l7_725


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l7_760

theorem complex_fraction_simplification :
  (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l7_760


namespace NUMINAMATH_CALUDE_shortest_chord_length_for_given_circle_and_point_l7_742

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the shortest chord length for a given circle and point -/
def shortestChordLength (c : Circle) (p : Point) : ℝ := sorry

/-- The main theorem -/
theorem shortest_chord_length_for_given_circle_and_point :
  let c : Circle := { equation := λ x y => x^2 + y^2 - 2*x - 3 }
  let p : Point := { x := 2, y := 1 }
  shortestChordLength c p = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_shortest_chord_length_for_given_circle_and_point_l7_742


namespace NUMINAMATH_CALUDE_probability_same_color_plates_l7_741

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def green_plates : ℕ := 3

def total_plates : ℕ := red_plates + blue_plates + green_plates

def same_color_combinations : ℕ := (
  Nat.choose red_plates 3 +
  Nat.choose blue_plates 3 +
  Nat.choose green_plates 3
)

def total_combinations : ℕ := Nat.choose total_plates 3

theorem probability_same_color_plates :
  (same_color_combinations : ℚ) / total_combinations = 31 / 364 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_plates_l7_741


namespace NUMINAMATH_CALUDE_max_value_range_l7_700

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) * (x - a)

-- State the theorem
theorem max_value_range (a : ℝ) :
  (∀ x, f_derivative a x = a * (x + 1) * (x - a)) →
  (∃ x₀, ∀ x, f x ≤ f x₀) →
  (∀ x, x < a → f_derivative a x > 0) →
  (∀ x, x > a → f_derivative a x < 0) →
  a ∈ Set.Ioo (-1 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_range_l7_700


namespace NUMINAMATH_CALUDE_power_of_difference_squared_l7_712

theorem power_of_difference_squared : (3^2 - 3)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_of_difference_squared_l7_712


namespace NUMINAMATH_CALUDE_mango_jelly_dishes_l7_748

theorem mango_jelly_dishes (total_dishes : ℕ) 
  (mango_salsa_dishes : ℕ) (fresh_mango_dishes : ℕ) 
  (oliver_pickout_dishes : ℕ) (oliver_left_dishes : ℕ) :
  total_dishes = 36 →
  mango_salsa_dishes = 3 →
  fresh_mango_dishes = total_dishes / 6 →
  oliver_pickout_dishes = 2 →
  oliver_left_dishes = 28 →
  total_dishes - oliver_left_dishes - (mango_salsa_dishes + (fresh_mango_dishes - oliver_pickout_dishes)) = 1 :=
by
  sorry

#check mango_jelly_dishes

end NUMINAMATH_CALUDE_mango_jelly_dishes_l7_748


namespace NUMINAMATH_CALUDE_smallest_twin_egg_number_l7_708

def is_twin_egg_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧
  ((n / 100) % 10 = (n / 10) % 10)

def exchange_digits (n : ℕ) : ℕ :=
  (n % 100) * 100 + (n / 100)

def F (m : ℕ) : ℚ :=
  (m - exchange_digits m : ℚ) / 11

theorem smallest_twin_egg_number :
  ∃ (m : ℕ),
    is_twin_egg_number m ∧
    (m / 1000 ≠ (m / 100) % 10) ∧
    ∃ (k : ℕ), F m / 54 = (k : ℚ) ^ 2 ∧
    ∀ (n : ℕ),
      is_twin_egg_number n ∧
      (n / 1000 ≠ (n / 100) % 10) ∧
      (∃ (j : ℕ), F n / 54 = (j : ℚ) ^ 2) →
      m ≤ n ∧
    m = 7117 :=
by sorry


end NUMINAMATH_CALUDE_smallest_twin_egg_number_l7_708


namespace NUMINAMATH_CALUDE_fruit_prices_l7_783

/-- Fruit prices problem -/
theorem fruit_prices (total_cost apple_cost orange_cost banana_cost : ℚ) : 
  total_cost = 7.84 ∧ 
  orange_cost = apple_cost + 0.28 ∧ 
  banana_cost = apple_cost - 0.15 ∧ 
  3 * apple_cost + 7 * orange_cost + 5 * banana_cost = total_cost →
  apple_cost = 0.442 ∧ orange_cost = 0.722 ∧ banana_cost = 0.292 := by
sorry

#eval (0.442 : ℚ) + 0.28 -- Should output 0.722
#eval (0.442 : ℚ) - 0.15 -- Should output 0.292
#eval 3 * (0.442 : ℚ) + 7 * 0.722 + 5 * 0.292 -- Should output 7.84

end NUMINAMATH_CALUDE_fruit_prices_l7_783


namespace NUMINAMATH_CALUDE_erica_earnings_l7_761

def fish_price : ℕ := 20
def past_four_months_catch : ℕ := 80
def monthly_maintenance : ℕ := 50
def fuel_cost_per_kg : ℕ := 2
def num_months : ℕ := 5

def total_catch : ℕ := past_four_months_catch * 3

def total_income : ℕ := total_catch * fish_price

def total_maintenance_cost : ℕ := monthly_maintenance * num_months

def total_fuel_cost : ℕ := fuel_cost_per_kg * total_catch

def total_cost : ℕ := total_maintenance_cost + total_fuel_cost

def net_income : ℤ := total_income - total_cost

theorem erica_earnings : net_income = 4070 := by
  sorry

end NUMINAMATH_CALUDE_erica_earnings_l7_761


namespace NUMINAMATH_CALUDE_star_example_l7_795

-- Define the star operation
def star (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem star_example : star (5/9) (6/4) = 135/2 := by sorry

end NUMINAMATH_CALUDE_star_example_l7_795


namespace NUMINAMATH_CALUDE_score_difference_is_negative_1_75_l7_747

def score_distribution : List (Float × Float) := [
  (0.15, 80),
  (0.40, 90),
  (0.25, 95),
  (0.20, 100)
]

def median (dist : List (Float × Float)) : Float :=
  90  -- The median is 90 as per the problem description

def mean (dist : List (Float × Float)) : Float :=
  dist.foldr (λ (p, s) acc => acc + p * s) 0

theorem score_difference_is_negative_1_75 :
  median score_distribution - mean score_distribution = -1.75 := by
  sorry

#eval median score_distribution - mean score_distribution

end NUMINAMATH_CALUDE_score_difference_is_negative_1_75_l7_747


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l7_799

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 8*x + k = (x + a)^2) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l7_799


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l7_782

theorem product_from_hcf_lcm (a b : ℕ+) (h_hcf : Nat.gcd a b = 14) (h_lcm : Nat.lcm a b = 183) :
  a * b = 2562 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l7_782


namespace NUMINAMATH_CALUDE_sector_area_l7_770

/-- Given a circular sector with central angle 60° and arc length 4, its area is 24/π -/
theorem sector_area (r : ℝ) : 
  (π / 3 : ℝ) = 4 / r →   -- Central angle in radians = Arc length / radius
  (1 / 2) * r^2 * (π / 3) = 24 / π := by
sorry

end NUMINAMATH_CALUDE_sector_area_l7_770


namespace NUMINAMATH_CALUDE_parabolas_intersect_l7_749

/-- Parabola function -/
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- Theorem: All parabolas with p + q = 2019 intersect at (1, 2020) -/
theorem parabolas_intersect (p q : ℝ) (h : p + q = 2019) : parabola p q 1 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_intersect_l7_749


namespace NUMINAMATH_CALUDE_red_pens_count_l7_702

/-- Proves the number of red pens initially in a jar --/
theorem red_pens_count (initial_blue : ℕ) (initial_black : ℕ) (removed_blue : ℕ) (removed_black : ℕ) (remaining_total : ℕ) : 
  initial_blue = 9 →
  initial_black = 21 →
  removed_blue = 4 →
  removed_black = 7 →
  remaining_total = 25 →
  ∃ (initial_red : ℕ), 
    initial_red = 6 ∧
    initial_blue + initial_black + initial_red = 
    remaining_total + removed_blue + removed_black :=
by sorry

end NUMINAMATH_CALUDE_red_pens_count_l7_702


namespace NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l7_785

theorem stuffed_animal_sales_difference (thor jake quincy : ℕ) 
  (h1 : jake = thor + 10)
  (h2 : quincy = 10 * thor)
  (h3 : quincy = 200) :
  quincy - jake = 170 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l7_785


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l7_766

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Define the line y = -x + 2
def line_intercept (x y : ℝ) : Prop := y = -x + 2

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 3/2

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem circle_and_line_theorem :
  -- Circle C passes through (1, √3)
  circle_C 1 (Real.sqrt 3) →
  -- The center of C is on the line y = x
  ∃ a : ℝ, line_y_eq_x a a ∧ ∀ x y : ℝ, circle_C x y ↔ (x - a)^2 + (y - a)^2 = 4 →
  -- The chord intercepted by y = -x + 2 has length 2√2
  ∃ x1 y1 x2 y2 : ℝ, 
    line_intercept x1 y1 ∧ line_intercept x2 y2 ∧ 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = 8 →
  -- Line l passes through (3/2, 0) and intersects C at P and Q
  ∃ m xP yP xQ yQ : ℝ,
    line_l m (3/2) 0 ∧ 
    circle_C xP yP ∧ circle_C xQ yQ ∧
    line_l m xP yP ∧ line_l m xQ yQ ∧
    -- OP · OQ = -2
    dot_product xP yP xQ yQ = -2 →
  -- Conclusion 1: Equation of circle C
  (∀ x y : ℝ, circle_C x y ↔ x^2 + y^2 = 4) ∧
  -- Conclusion 2: Equation of line l
  (m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2) ∧
  (∀ x y : ℝ, line_l m x y ↔ 2*x + m*y - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l7_766


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_420_l7_709

theorem distinct_prime_factors_of_420 : Nat.card (Nat.factors 420).toFinset = 4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_420_l7_709


namespace NUMINAMATH_CALUDE_unique_four_letter_product_l7_739

def letter_value (c : Char) : ℕ :=
  c.toNat - 'A'.toNat + 1

def four_letter_product (s : String) : ℕ :=
  if s.length = 4 then
    s.foldl (fun acc c => acc * letter_value c) 1
  else
    0

theorem unique_four_letter_product : ∀ s : String,
  s.length = 4 ∧ s ≠ "MNOQ" ∧ four_letter_product s = four_letter_product "MNOQ" →
  s = "NOQZ" :=
sorry

end NUMINAMATH_CALUDE_unique_four_letter_product_l7_739


namespace NUMINAMATH_CALUDE_calculator_sale_loss_l7_792

theorem calculator_sale_loss : 
  ∀ (x y : ℝ),
    x * (1 + 0.2) = 60 →
    y * (1 - 0.2) = 60 →
    60 + 60 - (x + y) = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_calculator_sale_loss_l7_792
