import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l4071_407149

theorem equation_solutions :
  (∀ x : ℝ, 25 * x^2 = 36 ↔ x = 6/5 ∨ x = -6/5) ∧
  (∀ x : ℝ, (1/3) * (x + 2)^3 - 9 = 0 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4071_407149


namespace NUMINAMATH_CALUDE_original_fraction_l4071_407153

theorem original_fraction (n : ℚ) : 
  (n + 1) / (n + 6) = 7 / 12 → n / (n + 5) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l4071_407153


namespace NUMINAMATH_CALUDE_max_b_value_l4071_407166

theorem max_b_value (a b : ℤ) (h : (127 : ℚ) / a - (16 : ℚ) / b = 1) : b ≤ 2016 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l4071_407166


namespace NUMINAMATH_CALUDE_smallest_block_volume_l4071_407138

theorem smallest_block_volume (a b c : ℕ) (h : (a - 1) * (b - 1) * (c - 1) = 143) :
  a * b * c ≥ 336 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l4071_407138


namespace NUMINAMATH_CALUDE_polyhedron_vertices_l4071_407155

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for polyhedra states that V - E + F = 2, where V is the number of vertices,
    E is the number of edges, and F is the number of faces. -/
axiom eulers_formula (p : Polyhedron) : p.vertices - p.edges + p.faces = 2

/-- The theorem states that a polyhedron with 21 edges and 9 faces has 14 vertices. -/
theorem polyhedron_vertices (p : Polyhedron) (h1 : p.edges = 21) (h2 : p.faces = 9) : 
  p.vertices = 14 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_vertices_l4071_407155


namespace NUMINAMATH_CALUDE_puzzle_solution_l4071_407171

/-- Given a permutation of the digits 1 to 6, prove that it satisfies the given conditions
    and corresponds to the number 132465 --/
theorem puzzle_solution (E U L S R T : Nat) : 
  ({E, U, L, S, R, T} : Finset Nat) = {1, 2, 3, 4, 5, 6} →
  E + U + L = 6 →
  S + R + U + T = 18 →
  U * T = 15 →
  S * L = 8 →
  E * 100000 + U * 10000 + L * 1000 + S * 100 + R * 10 + T = 132465 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l4071_407171


namespace NUMINAMATH_CALUDE_two_triangles_exist_l4071_407165

/-- A triangle with a given angle, height, and circumradius. -/
structure SpecialTriangle where
  /-- One of the angles of the triangle -/
  angle : ℝ
  /-- The height corresponding to one side of the triangle -/
  height : ℝ
  /-- The radius of the circumcircle -/
  circumradius : ℝ
  /-- The angle is positive and less than π -/
  angle_pos : 0 < angle
  angle_lt_pi : angle < π
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The circumradius is positive -/
  circumradius_pos : 0 < circumradius

/-- There exist two distinct triangles satisfying the given conditions -/
theorem two_triangles_exist (α m r : ℝ) 
  (h_α_pos : 0 < α) (h_α_lt_pi : α < π) 
  (h_m_pos : 0 < m) (h_r_pos : 0 < r) : 
  ∃ (t1 t2 : SpecialTriangle), t1 ≠ t2 ∧ 
    t1.angle = α ∧ t1.height = m ∧ t1.circumradius = r ∧
    t2.angle = α ∧ t2.height = m ∧ t2.circumradius = r := by
  sorry

end NUMINAMATH_CALUDE_two_triangles_exist_l4071_407165


namespace NUMINAMATH_CALUDE_set_equality_proof_l4071_407198

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}

def B (a : ℝ) : Set ℝ := {x | x < a}

def C : Set ℝ := {2, 3}

theorem set_equality_proof : 
  ∀ a : ℝ, (A ∪ B a = A) ↔ (a ∈ C) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_proof_l4071_407198


namespace NUMINAMATH_CALUDE_lasagna_pieces_needed_l4071_407139

/-- Represents the amount of lasagna each person eats relative to Manny's portion --/
structure LasagnaPortion where
  manny : ℚ
  lisa : ℚ
  raphael : ℚ
  aaron : ℚ
  kai : ℚ
  priya : ℚ

/-- Calculates the total number of lasagna pieces needed --/
def totalPieces (portions : LasagnaPortion) : ℚ :=
  portions.manny + portions.lisa + portions.kai + portions.priya

/-- The specific portions for each person based on the problem conditions --/
def givenPortions : LasagnaPortion :=
  { manny := 1
  , lisa := 2 + 1/2
  , raphael := 1/2
  , aaron := 0
  , kai := 2
  , priya := 1/3 }

theorem lasagna_pieces_needed : 
  ∃ n : ℕ, n > 0 ∧ n = ⌈totalPieces givenPortions⌉ ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_pieces_needed_l4071_407139


namespace NUMINAMATH_CALUDE_pyramid_sculpture_surface_area_l4071_407187

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat

/-- Calculates the painted surface area of a cube sculpture -/
def painted_surface_area (sculpture : CubeSculpture) : Nat :=
  sorry

/-- The specific sculpture described in the problem -/
def pyramid_sculpture : CubeSculpture :=
  { top_layer := 1
  , middle_layer := 5
  , bottom_layer := 13 }

theorem pyramid_sculpture_surface_area :
  painted_surface_area pyramid_sculpture = 31 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_sculpture_surface_area_l4071_407187


namespace NUMINAMATH_CALUDE_image_of_4_neg2_l4071_407183

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (xy, x+y) -/
def f : ℝ × ℝ → ℝ × ℝ := λ (x, y) => (x * y, x + y)

/-- The theorem stating that the image of (4, -2) under f is (-8, 2) -/
theorem image_of_4_neg2 : f (4, -2) = (-8, 2) := by sorry

end NUMINAMATH_CALUDE_image_of_4_neg2_l4071_407183


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_f_prime_l4071_407163

def f (x : ℝ) : ℝ := (1 - 2*x)^10

theorem coefficient_x_squared_in_f_prime : 
  ∃ (g : ℝ → ℝ), (∀ x, deriv f x = g x) ∧ 
  (∃ (a b c : ℝ), ∀ x, g x = a*x^2 + b*x + c) ∧
  (∃ (a b c : ℝ), (∀ x, g x = a*x^2 + b*x + c) ∧ a = -2880) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_f_prime_l4071_407163


namespace NUMINAMATH_CALUDE_bus_car_ratio_l4071_407101

theorem bus_car_ratio (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 65 →
  num_buses = num_cars - 60 →
  (num_buses : ℚ) / (num_cars : ℚ) = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_bus_car_ratio_l4071_407101


namespace NUMINAMATH_CALUDE_system_solution_l4071_407142

theorem system_solution (x y : ℝ) :
  (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7 ∧
  (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7 →
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4071_407142


namespace NUMINAMATH_CALUDE_inequality_proof_l4071_407122

theorem inequality_proof (a b : ℝ) (ha : |a| < 2) (hb : |b| < 2) : 2*|a + b| < |4 + a*b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4071_407122


namespace NUMINAMATH_CALUDE_kerosene_mixture_l4071_407180

theorem kerosene_mixture (x : ℝ) : 
  (((6 * (x / 100)) + (4 * 0.3)) / 10 = 0.27) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_mixture_l4071_407180


namespace NUMINAMATH_CALUDE_min_S_value_l4071_407176

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_S_value (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_constraint : x^2 + y^2 + z^2 = 1) :
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x'^2 + y'^2 + z'^2 = 1 → 
    S x y z ≤ S x' y' z') →
  x = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_min_S_value_l4071_407176


namespace NUMINAMATH_CALUDE_mikes_money_duration_l4071_407159

/-- The number of weeks Mike's money will last given his earnings and weekly spending. -/
def weeks_money_lasts (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Mike's money will last 8 weeks given his earnings and spending. -/
theorem mikes_money_duration :
  weeks_money_lasts 14 26 5 = 8 :=
by sorry

end NUMINAMATH_CALUDE_mikes_money_duration_l4071_407159


namespace NUMINAMATH_CALUDE_difference_of_squares_503_497_l4071_407131

theorem difference_of_squares_503_497 : 503^2 - 497^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_503_497_l4071_407131


namespace NUMINAMATH_CALUDE_ratio_change_l4071_407184

theorem ratio_change (x y : ℤ) (n : ℤ) : 
  y = 48 → x / y = 1 / 4 → (x + n) / y = 1 / 2 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_l4071_407184


namespace NUMINAMATH_CALUDE_modular_arithmetic_equivalence_l4071_407196

theorem modular_arithmetic_equivalence : 144 * 20 - 17^2 + 5 ≡ 4 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_equivalence_l4071_407196


namespace NUMINAMATH_CALUDE_nested_expression_value_l4071_407110

theorem nested_expression_value : 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))) = 1364 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l4071_407110


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4071_407199

def p (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 6*x - 18

theorem polynomial_divisibility :
  (∃ q : ℝ → ℝ, p = fun x ↦ (x - 3) * q x) ∧
  (∃ r : ℝ → ℝ, p = fun x ↦ (2*x^2 + 6) * r x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4071_407199


namespace NUMINAMATH_CALUDE_square_diagonal_l4071_407150

theorem square_diagonal (A : ℝ) (h : A = 200) : 
  ∃ d : ℝ, d^2 = 2 * A ∧ d = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l4071_407150


namespace NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_line_l_not_in_fourth_quadrant_min_area_of_triangle_AOB_l4071_407121

-- Define the line l: kx - y + 1 + 2k = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-2, 1)

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Define the negative half of x-axis and positive half of y-axis
def neg_x_axis (x : ℝ) : Prop := x < 0
def pos_y_axis (y : ℝ) : Prop := y > 0

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the area of a triangle given three points
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statements
theorem line_l_passes_through_fixed_point :
  ∀ k : ℝ, line_l k (fixed_point.1) (fixed_point.2) := by sorry

theorem line_l_not_in_fourth_quadrant :
  ∀ k x y : ℝ, line_l k x y → ¬(fourth_quadrant x y) → k ≥ 0 := by sorry

theorem min_area_of_triangle_AOB :
  ∀ k x y : ℝ,
  line_l k x y →
  neg_x_axis x →
  pos_y_axis y →
  let a := (x, 0)
  let b := (0, y)
  triangle_area a origin b ≥ 4 ∧
  (triangle_area a origin b = 4 ↔ line_l (1/2) x y) := by sorry

end NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_line_l_not_in_fourth_quadrant_min_area_of_triangle_AOB_l4071_407121


namespace NUMINAMATH_CALUDE_problem_solution_l4071_407162

theorem problem_solution (c d : ℚ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4071_407162


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l4071_407114

theorem unique_solution_for_exponential_equation :
  ∀ n m : ℕ+, 5^(n : ℕ) = 6*(m : ℕ)^2 + 1 ↔ n = 2 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l4071_407114


namespace NUMINAMATH_CALUDE_increasing_quadratic_coefficient_range_l4071_407112

def f (m : ℝ) (x : ℝ) := 3 * x^2 + m * x + 2

theorem increasing_quadratic_coefficient_range (m : ℝ) :
  (∀ x ≥ 1, ∀ y > x, f m y > f m x) →
  m ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_coefficient_range_l4071_407112


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4071_407146

/-- Given vectors a and b in R², if a is perpendicular to (t*a + b), then t = -5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, -1))
  (h2 : b = (6, -4))
  (h3 : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) :
  t = -5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4071_407146


namespace NUMINAMATH_CALUDE_remainder_proof_l4071_407178

theorem remainder_proof : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l4071_407178


namespace NUMINAMATH_CALUDE_weight_replaced_is_75_l4071_407185

/-- The weight of the replaced person in a group, given the following conditions:
  * There are 7 persons initially
  * The average weight increases by 3.5 kg when a new person replaces one of them
  * The weight of the new person is 99.5 kg
-/
def weight_of_replaced_person (num_persons : ℕ) (avg_weight_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - (num_persons * avg_weight_increase)

/-- Theorem stating that the weight of the replaced person is 75 kg -/
theorem weight_replaced_is_75 :
  weight_of_replaced_person 7 3.5 99.5 = 75 := by sorry

end NUMINAMATH_CALUDE_weight_replaced_is_75_l4071_407185


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l4071_407169

/-- Proves that given a boat with speed 36 kmph in still water and a stream with speed 12 kmph,
    if the boat covers 80 km downstream in the same time as it covers a certain distance upstream,
    then that upstream distance is 40 km. -/
theorem boat_upstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (h1 : boat_speed = 36)
  (h2 : stream_speed = 12)
  (h3 : downstream_distance = 80)
  (h4 : downstream_distance / (boat_speed + stream_speed) =
        upstream_distance / (boat_speed - stream_speed)) :
  upstream_distance = 40 :=
sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l4071_407169


namespace NUMINAMATH_CALUDE_sin_squared_sum_l4071_407177

theorem sin_squared_sum (α : ℝ) : 
  Real.sin α ^ 2 + Real.sin (α + Real.pi / 3) ^ 2 + Real.sin (α + 2 * Real.pi / 3) ^ 2 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_l4071_407177


namespace NUMINAMATH_CALUDE_unique_n_congruence_l4071_407120

theorem unique_n_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_n_congruence_l4071_407120


namespace NUMINAMATH_CALUDE_prob_at_least_one_second_class_l4071_407109

/-- The probability of selecting at least one second-class item when randomly choosing 3 items
    from a set of 10 items, where 6 are first-class and 4 are second-class. -/
theorem prob_at_least_one_second_class (total : Nat) (first_class : Nat) (second_class : Nat) (selected : Nat)
    (h1 : total = 10)
    (h2 : first_class = 6)
    (h3 : second_class = 4)
    (h4 : selected = 3)
    (h5 : total = first_class + second_class) :
    (1 : ℚ) - (Nat.choose first_class selected : ℚ) / (Nat.choose total selected : ℚ) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_second_class_l4071_407109


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l4071_407193

theorem mean_proportional_problem (x : ℝ) : 
  (156 : ℝ) = Real.sqrt (234 * x) → x = 104 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l4071_407193


namespace NUMINAMATH_CALUDE_twenty_two_students_remain_l4071_407174

/-- The number of remaining students after some leave early -/
def remaining_students (total_groups : ℕ) (students_per_group : ℕ) (students_who_left : ℕ) : ℕ :=
  total_groups * students_per_group - students_who_left

/-- Theorem stating that given 3 groups of 8 students with 2 leaving early, 22 students remain -/
theorem twenty_two_students_remain :
  remaining_students 3 8 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_twenty_two_students_remain_l4071_407174


namespace NUMINAMATH_CALUDE_max_value_of_f_l4071_407125

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4071_407125


namespace NUMINAMATH_CALUDE_sin_690_degrees_l4071_407170

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l4071_407170


namespace NUMINAMATH_CALUDE_inequality_proof_l4071_407175

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b)/(a + b) + b^2 * (b - c)/(b + c) + c^2 * (c - a)/(c + a) ≥ 0 ∧
  (a^2 * (a - b)/(a + b) + b^2 * (b - c)/(b + c) + c^2 * (c - a)/(c + a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4071_407175


namespace NUMINAMATH_CALUDE_binary_1110011_is_115_l4071_407179

def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1110011_is_115 :
  binary_to_decimal [true, true, false, false, true, true, true] = 115 := by
  sorry

end NUMINAMATH_CALUDE_binary_1110011_is_115_l4071_407179


namespace NUMINAMATH_CALUDE_stopping_time_maximizes_distance_l4071_407161

/-- The distance function representing the distance traveled by a car after braking. -/
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

/-- The time at which the distance function reaches its maximum value. -/
def stopping_time : ℝ := 3

/-- Theorem stating that the stopping time maximizes the distance function. -/
theorem stopping_time_maximizes_distance :
  ∀ t : ℝ, S t ≤ S stopping_time :=
sorry

end NUMINAMATH_CALUDE_stopping_time_maximizes_distance_l4071_407161


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4071_407106

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = 5) : z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4071_407106


namespace NUMINAMATH_CALUDE_wednesday_profit_l4071_407173

/-- The profit made by a beadshop over three days -/
def BeadshopProfit (total : ℝ) (monday : ℝ) (tuesday : ℝ) (wednesday : ℝ) : Prop :=
  total = 1200 ∧
  monday = (1/3) * total ∧
  tuesday = (1/4) * total ∧
  wednesday = total - monday - tuesday

/-- The profit made on Wednesday is $500 -/
theorem wednesday_profit (total monday tuesday wednesday : ℝ) :
  BeadshopProfit total monday tuesday wednesday →
  wednesday = 500 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_profit_l4071_407173


namespace NUMINAMATH_CALUDE_square_minus_a_nonpositive_implies_a_geq_four_l4071_407195

theorem square_minus_a_nonpositive_implies_a_geq_four :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_a_nonpositive_implies_a_geq_four_l4071_407195


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l4071_407137

/-- The first term of the geometric series -/
def a₁ : ℚ := 7/8

/-- The second term of the geometric series -/
def a₂ : ℚ := -14/27

/-- The third term of the geometric series -/
def a₃ : ℚ := 28/81

/-- The common ratio of the geometric series -/
def r : ℚ := -2/3

/-- Theorem stating that the given series is geometric with common ratio r -/
theorem geometric_series_common_ratio :
  a₂ = a₁ * r ∧ a₃ = a₂ * r := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l4071_407137


namespace NUMINAMATH_CALUDE_total_socks_l4071_407130

def sock_problem (red_socks black_socks white_socks : ℕ) : Prop :=
  red_socks = 40 ∧
  black_socks = red_socks / 2 ∧
  white_socks = 2 * (red_socks + black_socks)

theorem total_socks (red_socks black_socks white_socks : ℕ) 
  (h : sock_problem red_socks black_socks white_socks) : 
  red_socks + black_socks + white_socks = 180 :=
by sorry

end NUMINAMATH_CALUDE_total_socks_l4071_407130


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4071_407168

/-- Given a quadratic function f(x) = ax^2 + bx + c with specific conditions,
    prove properties about its coefficients, roots, and values. -/
theorem quadratic_function_properties
  (a b c : ℝ) (m₁ m₂ : ℝ)
  (h_order : a > b ∧ b > c)
  (h_points : a^2 + (a * m₁^2 + b * m₁ + c + a * m₂^2 + b * m₂ + c) * a +
              (a * m₁^2 + b * m₁ + c) * (a * m₂^2 + b * m₂ + c) = 0)
  (h_root : a + b + c = 0) :
  (b ≥ 0) ∧
  (2 ≤ |1 - c/a| ∧ |1 - c/a| < 3) ∧
  (max (a * (m₁ + 3)^2 + b * (m₁ + 3) + c) (a * (m₂ + 3)^2 + b * (m₂ + 3) + c) > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4071_407168


namespace NUMINAMATH_CALUDE_right_triangle_set_l4071_407136

theorem right_triangle_set : ∀ (a b c : ℝ),
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = 3) ∨
   (a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 12) ∨
   (a = 5 ∧ b = 11 ∧ c = 13)) →
  (a^2 + b^2 = c^2 ↔ (a = 3 ∧ b = 4 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_set_l4071_407136


namespace NUMINAMATH_CALUDE_fifth_number_13th_row_is_715_l4071_407123

/-- The fifth number in the 13th row of Pascal's triangle -/
def fifth_number_13th_row : ℕ :=
  Nat.choose 13 4

/-- Theorem stating that the fifth number in the 13th row of Pascal's triangle is 715 -/
theorem fifth_number_13th_row_is_715 : fifth_number_13th_row = 715 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_13th_row_is_715_l4071_407123


namespace NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l4071_407186

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l4071_407186


namespace NUMINAMATH_CALUDE_compare_absolute_values_l4071_407194

theorem compare_absolute_values (m n : ℝ) 
  (h1 : m * n < 0) 
  (h2 : m + n < 0) 
  (h3 : n > 0) : 
  |m| > |n| := by
  sorry

end NUMINAMATH_CALUDE_compare_absolute_values_l4071_407194


namespace NUMINAMATH_CALUDE_at_least_one_third_l4071_407160

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l4071_407160


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l4071_407129

theorem quadratic_is_perfect_square (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 26 * x + 9 = (r * x + s)^2) → a = 169 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l4071_407129


namespace NUMINAMATH_CALUDE_power_of_product_l4071_407102

theorem power_of_product (x y : ℝ) : (-2 * x * y^3)^2 = 4 * x^2 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4071_407102


namespace NUMINAMATH_CALUDE_linear_function_max_value_l4071_407151

theorem linear_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → a * x - a + 2 ≤ 7) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 4 ∧ a * x - a + 2 = 7) →
  a = 5/3 ∨ a = -5/2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_max_value_l4071_407151


namespace NUMINAMATH_CALUDE_power_of_product_l4071_407191

theorem power_of_product (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4071_407191


namespace NUMINAMATH_CALUDE_roja_speed_calculation_l4071_407144

/-- Roja's speed in km/hr -/
def rojaSpeed : ℝ := 8

/-- Pooja's speed in km/hr -/
def poojaSpeed : ℝ := 3

/-- Time elapsed in hours -/
def timeElapsed : ℝ := 4

/-- Distance between Roja and Pooja after the elapsed time in km -/
def distanceBetween : ℝ := 44

theorem roja_speed_calculation :
  rojaSpeed = 8 ∧
  poojaSpeed = 3 ∧
  timeElapsed = 4 ∧
  distanceBetween = 44 ∧
  distanceBetween = (rojaSpeed + poojaSpeed) * timeElapsed :=
by sorry

end NUMINAMATH_CALUDE_roja_speed_calculation_l4071_407144


namespace NUMINAMATH_CALUDE_ice_cream_sales_l4071_407192

def daily_sales : List ℝ := [100, 92, 109, 96, 0, 96, 105]

theorem ice_cream_sales (x : ℝ) :
  let sales := daily_sales.set 4 x
  sales.length = 7 ∧ 
  sales.sum / sales.length = 100.1 →
  x = 102.7 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l4071_407192


namespace NUMINAMATH_CALUDE_complex_power_of_four_l4071_407105

theorem complex_power_of_four : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_power_of_four_l4071_407105


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4071_407167

theorem sum_of_coefficients (a c : ℚ) : 
  (3 : ℚ) ∈ {x | a * x^2 - 6 * x + c = 0} →
  (1/3 : ℚ) ∈ {x | a * x^2 - 6 * x + c = 0} →
  a + c = 18/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4071_407167


namespace NUMINAMATH_CALUDE_range_of_a_l4071_407124

-- Define the condition that x^2 > 1 is necessary but not sufficient for x < a
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x : ℝ, x < a → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a)

-- Theorem stating the range of values for a
theorem range_of_a (a : ℝ) : 
  necessary_not_sufficient a ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4071_407124


namespace NUMINAMATH_CALUDE_water_consumption_correct_l4071_407134

/-- Water consumption per person per year in cubic meters for different regions -/
structure WaterConsumption where
  west : ℝ
  nonWest : ℝ
  russia : ℝ

/-- Given water consumption data -/
def givenData : WaterConsumption :=
  { west := 21428
    nonWest := 26848.55
    russia := 302790.13 }

/-- Theorem stating that the given water consumption data is correct -/
theorem water_consumption_correct (data : WaterConsumption) :
  data.west = givenData.west ∧
  data.nonWest = givenData.nonWest ∧
  data.russia = givenData.russia :=
by sorry

#check water_consumption_correct

end NUMINAMATH_CALUDE_water_consumption_correct_l4071_407134


namespace NUMINAMATH_CALUDE_no_partition_sum_product_l4071_407132

theorem no_partition_sum_product (x y : ℕ) : 
  x ∈ Finset.range 15 → 
  y ∈ Finset.range 15 → 
  x ≠ y → 
  x * y ≠ 120 - x - y := by
sorry

end NUMINAMATH_CALUDE_no_partition_sum_product_l4071_407132


namespace NUMINAMATH_CALUDE_cubic_expression_zero_l4071_407135

theorem cubic_expression_zero (x : ℝ) (h : x^2 + 3*x - 3 = 0) : 
  x^3 + 2*x^2 - 6*x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_zero_l4071_407135


namespace NUMINAMATH_CALUDE_louise_yellow_pencils_l4071_407145

/-- Proves the number of yellow pencils Louise has --/
theorem louise_yellow_pencils :
  let box_capacity : ℕ := 20
  let red_pencils : ℕ := 20
  let blue_pencils : ℕ := 2 * red_pencils
  let green_pencils : ℕ := red_pencils + blue_pencils
  let total_boxes : ℕ := 8
  let total_capacity : ℕ := total_boxes * box_capacity
  let other_pencils : ℕ := red_pencils + blue_pencils + green_pencils
  let yellow_pencils : ℕ := total_capacity - other_pencils
  yellow_pencils = 40 := by
  sorry

end NUMINAMATH_CALUDE_louise_yellow_pencils_l4071_407145


namespace NUMINAMATH_CALUDE_wire_around_square_field_l4071_407197

/-- Proves that a wire of length 15840 m goes around a square field of area 69696 m^2 exactly 15 times -/
theorem wire_around_square_field (field_area : ℝ) (wire_length : ℝ) : 
  field_area = 69696 → wire_length = 15840 → 
  (wire_length / (4 * Real.sqrt field_area) : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_wire_around_square_field_l4071_407197


namespace NUMINAMATH_CALUDE_correct_average_l4071_407148

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 46 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 50 := by
sorry

end NUMINAMATH_CALUDE_correct_average_l4071_407148


namespace NUMINAMATH_CALUDE_town_distance_interval_l4071_407128

def distance_to_town (d : ℝ) : Prop :=
  (¬ (d ≥ 8)) ∧ (¬ (d ≤ 7)) ∧ (¬ (d ≤ 6)) ∧ (d ≠ 5)

theorem town_distance_interval :
  ∀ d : ℝ, distance_to_town d → (7 < d ∧ d < 8) :=
by sorry

end NUMINAMATH_CALUDE_town_distance_interval_l4071_407128


namespace NUMINAMATH_CALUDE_hundred_three_square_partitions_l4071_407118

/-- A function that returns the number of ways to write a given number as the sum of three positive perfect squares, where the order doesn't matter. -/
def count_three_square_partitions (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there is exactly one way to write 100 as the sum of three positive perfect squares, where the order doesn't matter. -/
theorem hundred_three_square_partitions : count_three_square_partitions 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hundred_three_square_partitions_l4071_407118


namespace NUMINAMATH_CALUDE_triangle_problem_l4071_407141

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C. -/
theorem triangle_problem (a b c A B C : Real) :
  -- Conditions
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c) →
  (a = 1) →
  (b * c * Real.cos A = 3) →
  -- Conclusions
  (A = π / 6) ∧ (b + c = Real.sqrt 3 + 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l4071_407141


namespace NUMINAMATH_CALUDE_problem1_l4071_407147

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_problem1_l4071_407147


namespace NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l4071_407188

theorem complex_roots_isosceles_triangle (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  Complex.abs z₁ = Complex.abs (2*z₂) → 
  a^2 / b = 4.5 := by sorry

end NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l4071_407188


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l4071_407143

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (70 * π / 180) + Real.tan (80 * π / 180)) / Real.cos (30 * π / 180) =
  (1 + Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) / (Real.cos (20 * π / 180) * Real.cos (70 * π / 180) * Real.cos (30 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l4071_407143


namespace NUMINAMATH_CALUDE_annes_walking_time_l4071_407108

/-- Anne's walking problem -/
theorem annes_walking_time (distance : ℝ) (speed : ℝ) (time : ℝ) 
  (h1 : distance = 6)
  (h2 : speed = 2)
  (h3 : distance = speed * time) : 
  time = 3 := by
  sorry

end NUMINAMATH_CALUDE_annes_walking_time_l4071_407108


namespace NUMINAMATH_CALUDE_cat_weight_sum_l4071_407104

/-- The combined weight of three cats -/
def combined_weight (w1 w2 w3 : ℕ) : ℕ := w1 + w2 + w3

/-- Theorem: The combined weight of cats weighing 2, 7, and 4 pounds is 13 pounds -/
theorem cat_weight_sum : combined_weight 2 7 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_sum_l4071_407104


namespace NUMINAMATH_CALUDE_seventh_term_is_eight_l4071_407103

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 2 ∧
  a 3 + a 4 = 9

/-- Theorem: For an arithmetic sequence satisfying the given conditions, the 7th term is 8 -/
theorem seventh_term_is_eight (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_eight_l4071_407103


namespace NUMINAMATH_CALUDE_problem_solution_l4071_407127

theorem problem_solution (x : ℝ) : (0.5 * x - (1/3) * x = 110) → x = 660 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4071_407127


namespace NUMINAMATH_CALUDE_cone_height_ratio_l4071_407115

theorem cone_height_ratio (original_circumference : ℝ) (original_height : ℝ) (new_volume : ℝ) :
  original_circumference = 20 * Real.pi →
  original_height = 40 →
  new_volume = 800 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3) * Real.pi * (original_circumference / (2 * Real.pi))^2 * new_height = new_volume ∧
    new_height / original_height = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l4071_407115


namespace NUMINAMATH_CALUDE_sqrt_square_nine_l4071_407181

theorem sqrt_square_nine : Real.sqrt 9 ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_nine_l4071_407181


namespace NUMINAMATH_CALUDE_vector_problem_l4071_407126

def vector_a : Fin 2 → ℝ := ![3, -4]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![2, x]
def vector_c (y : ℝ) : Fin 2 → ℝ := ![2, y]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i, v i = k * u i

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem vector_problem (x y : ℝ) 
  (h1 : parallel vector_a (vector_b x))
  (h2 : perpendicular vector_a (vector_c y)) :
  (x = -8/3 ∧ y = 3/2) ∧ 
  perpendicular (vector_b (-8/3)) (vector_c (3/2)) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l4071_407126


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l4071_407182

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and a point on the line x - √2 y - 5 = 0 is √3/3 -/
theorem min_distance_ellipse_line : 
  ∃ (d : ℝ), d = Real.sqrt 3 / 3 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.1^2 / 8 + P.2^2 / 4 = 1) → 
    (Q.1 - Real.sqrt 2 * Q.2 - 5 = 0) → 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d ∧
    ∃ (P₀ Q₀ : ℝ × ℝ), 
      (P₀.1^2 / 8 + P₀.2^2 / 4 = 1) ∧
      (Q₀.1 - Real.sqrt 2 * Q₀.2 - 5 = 0) ∧
      Real.sqrt ((P₀.1 - Q₀.1)^2 + (P₀.2 - Q₀.2)^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l4071_407182


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l4071_407117

def euler_family_children : ℕ := 7
def girls_aged_8 : ℕ := 4
def boys_aged_11 : ℕ := 2
def girl_aged_16 : ℕ := 1

def total_age : ℕ := girls_aged_8 * 8 + boys_aged_11 * 11 + girl_aged_16 * 16

theorem euler_family_mean_age :
  (total_age : ℚ) / euler_family_children = 10 := by sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l4071_407117


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l4071_407190

def is_valid_selection (s : Finset Nat) : Prop :=
  s.card = 6 ∧ s ⊆ Finset.range 8

def Q (s : Finset Nat) : Nat :=
  s.prod id

theorem largest_certain_divisor :
  ∀ (s : Finset Nat), is_valid_selection s →
  (2 ∣ Q s) ∧ 
  ∀ (n : Nat), n > 2 → (∃ (t : Finset Nat), is_valid_selection t ∧ ¬(n ∣ Q t)) :=
by sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l4071_407190


namespace NUMINAMATH_CALUDE_prob_at_least_one_odd_is_nine_tenths_l4071_407140

def numbers : Finset ℕ := {1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Bool := n % 2 = 1

def prob_at_least_one_odd : ℚ :=
  1 - (Finset.filter (λ n => ¬(is_odd n)) numbers).card.choose 2 / numbers.card.choose 2

theorem prob_at_least_one_odd_is_nine_tenths :
  prob_at_least_one_odd = 9/10 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_odd_is_nine_tenths_l4071_407140


namespace NUMINAMATH_CALUDE_households_with_both_count_l4071_407116

/-- Represents the distribution of car and bike ownership in a neighborhood -/
structure Neighborhood where
  total : ℕ
  neither : ℕ
  with_car : ℕ
  only_bike : ℕ

/-- Calculates the number of households with both a car and a bike -/
def households_with_both (n : Neighborhood) : ℕ :=
  n.with_car - n.only_bike

/-- Theorem stating the number of households with both a car and a bike -/
theorem households_with_both_count (n : Neighborhood) 
  (h1 : n.total = 90)
  (h2 : n.neither = 11)
  (h3 : n.with_car = 44)
  (h4 : n.only_bike = 35)
  (h5 : n.total = n.neither + n.with_car + n.only_bike) :
  households_with_both n = 9 := by
  sorry

#eval households_with_both { total := 90, neither := 11, with_car := 44, only_bike := 35 }

end NUMINAMATH_CALUDE_households_with_both_count_l4071_407116


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4071_407113

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m = 0) ∧ (m^2 - 5*m + 6 ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4071_407113


namespace NUMINAMATH_CALUDE_budget_food_percentage_l4071_407154

theorem budget_food_percentage (total_budget : ℝ) (accommodation_percent : ℝ) (entertainment_percent : ℝ) (coursework_materials : ℝ) :
  total_budget = 1000 →
  accommodation_percent = 15 →
  entertainment_percent = 25 →
  coursework_materials = 300 →
  (total_budget - (total_budget * accommodation_percent / 100 + total_budget * entertainment_percent / 100 + coursework_materials)) / total_budget * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_budget_food_percentage_l4071_407154


namespace NUMINAMATH_CALUDE_largest_sum_largest_sum_proof_l4071_407158

theorem largest_sum : ℝ → ℝ → ℝ → Prop :=
  fun A B C => 
    let A := 2010 / 2009 + 2010 / 2011
    let B := 2010 / 2011 + 2012 / 2011
    let C := 2011 / 2010 + 2011 / 2012 + 1 / 2011
    C > A ∧ C > B

-- The proof is omitted
theorem largest_sum_proof : largest_sum (2010 / 2009 + 2010 / 2011) (2010 / 2011 + 2012 / 2011) (2011 / 2010 + 2011 / 2012 + 1 / 2011) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_largest_sum_proof_l4071_407158


namespace NUMINAMATH_CALUDE_race_length_l4071_407172

/-- The race between Nicky and Cristina -/
def race (cristina_speed nicky_speed : ℝ) (head_start catch_up_time : ℝ) : Prop :=
  let nicky_distance := nicky_speed * catch_up_time
  let cristina_time := catch_up_time - head_start
  let cristina_distance := cristina_speed * cristina_time
  nicky_distance = cristina_distance ∧ nicky_distance = 90

/-- The race length is 90 meters -/
theorem race_length :
  race 5 3 12 30 :=
by
  sorry

end NUMINAMATH_CALUDE_race_length_l4071_407172


namespace NUMINAMATH_CALUDE_blue_line_length_is_correct_l4071_407152

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.67

/-- The difference in length between the white and blue lines in inches -/
def length_difference : ℝ := 4.33

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := white_line_length - length_difference

theorem blue_line_length_is_correct : blue_line_length = 3.34 := by
  sorry

end NUMINAMATH_CALUDE_blue_line_length_is_correct_l4071_407152


namespace NUMINAMATH_CALUDE_dad_borrowed_quarters_l4071_407164

/-- The number of quarters borrowed by Sara's dad -/
def quarters_borrowed (initial_quarters current_quarters : ℕ) : ℕ :=
  initial_quarters - current_quarters

/-- Proof that Sara's dad borrowed 271 quarters -/
theorem dad_borrowed_quarters : quarters_borrowed 783 512 = 271 := by
  sorry

end NUMINAMATH_CALUDE_dad_borrowed_quarters_l4071_407164


namespace NUMINAMATH_CALUDE_books_given_away_l4071_407100

theorem books_given_away (original_books : Real) (books_left : Nat) : 
  original_books = 54.0 → books_left = 31 → original_books - books_left = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_given_away_l4071_407100


namespace NUMINAMATH_CALUDE_square_plate_nails_l4071_407157

/-- The number of nails on each side of the square -/
def nails_per_side : ℕ := 25

/-- The total number of unique nails used to fix the square plate -/
def total_nails : ℕ := nails_per_side * 4 - 4

theorem square_plate_nails :
  total_nails = 96 :=
by sorry

end NUMINAMATH_CALUDE_square_plate_nails_l4071_407157


namespace NUMINAMATH_CALUDE_no_perfect_cube_in_range_l4071_407156

theorem no_perfect_cube_in_range : 
  ¬ ∃ (n : ℤ), 4 ≤ n ∧ n ≤ 11 ∧ ∃ (k : ℤ), n^2 + 3*n + 2 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cube_in_range_l4071_407156


namespace NUMINAMATH_CALUDE_point_location_implies_coordinate_signs_l4071_407133

/-- A point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is to the right of the y-axis -/
def isRightOfYAxis (p : Point) : Prop := p.x > 0

/-- Predicate to check if a point is below the x-axis -/
def isBelowXAxis (p : Point) : Prop := p.y < 0

/-- Theorem stating that if a point is to the right of the y-axis and below the x-axis,
    then its x-coordinate is positive and y-coordinate is negative -/
theorem point_location_implies_coordinate_signs (p : Point) :
  isRightOfYAxis p → isBelowXAxis p → p.x > 0 ∧ p.y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_location_implies_coordinate_signs_l4071_407133


namespace NUMINAMATH_CALUDE_fence_perimeter_is_177_l4071_407119

/-- Calculates the outer perimeter of a rectangular fence with specified conditions -/
def fence_perimeter (num_posts : ℕ) (post_width : ℚ) (gap_width : ℚ) : ℚ :=
  let width_posts := num_posts / 4
  let length_posts := width_posts * 2
  let width := (width_posts - 1) * gap_width + width_posts * post_width
  let length := (length_posts - 1) * gap_width + length_posts * post_width
  2 * (width + length)

/-- The outer perimeter of the fence is 177 feet -/
theorem fence_perimeter_is_177 :
  fence_perimeter 36 (1/2) 3 = 177 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_177_l4071_407119


namespace NUMINAMATH_CALUDE_two_hearts_three_different_probability_l4071_407189

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (hearts : Nat)
  (other_suits : Nat)
  (cards_eq : cards = 52)
  (hearts_eq : hearts = 13)
  (other_suits_eq : other_suits = 39)

/-- The probability of the specified event -/
def probability_two_hearts_three_different (d : Deck) : ℚ :=
  135 / 1024

/-- Theorem statement -/
theorem two_hearts_three_different_probability (d : Deck) :
  probability_two_hearts_three_different d = 135 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_two_hearts_three_different_probability_l4071_407189


namespace NUMINAMATH_CALUDE_system_solutions_correct_l4071_407107

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℚ, y = 2 * x ∧ 3 * y + 2 * x = 8 ∧ x = 1 ∧ y = 2) ∧
  -- System 2
  (∃ x y : ℚ, x - 3 * y = -2 ∧ 2 * x + 3 * y = 3 ∧ x = 1/3 ∧ y = 7/9) := by
  sorry

#check system_solutions_correct

end NUMINAMATH_CALUDE_system_solutions_correct_l4071_407107


namespace NUMINAMATH_CALUDE_fraction_equality_l4071_407111

def f (x : ℕ) : ℚ := (x^4 + 400 : ℚ)

def numerator : ℚ := f 15 * f 27 * f 39 * f 51 * f 63
def denominator : ℚ := f 5 * f 17 * f 29 * f 41 * f 53

theorem fraction_equality : numerator / denominator = 4115 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4071_407111
