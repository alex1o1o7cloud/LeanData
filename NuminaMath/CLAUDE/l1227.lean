import Mathlib

namespace NUMINAMATH_CALUDE_gcd_sum_lcm_l1227_122710

theorem gcd_sum_lcm (a b : ℤ) : Nat.gcd (a + b).natAbs (Nat.lcm a.natAbs b.natAbs) = Nat.gcd a.natAbs b.natAbs := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_lcm_l1227_122710


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1227_122769

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ ∀ n, ¬p n :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, 3^n > 2018) ↔ (∀ n : ℕ, 3^n ≤ 2018) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1227_122769


namespace NUMINAMATH_CALUDE_optimal_dimensions_maximize_volume_unique_maximum_volume_l1227_122739

/-- Represents the volume of a rectangular frame as a function of its width. -/
def volume (x : ℝ) : ℝ := 2 * x^2 * (4.5 - 3*x)

/-- The maximum volume of the rectangular frame. -/
def max_volume : ℝ := 3

/-- The width that maximizes the volume. -/
def optimal_width : ℝ := 1

/-- The length that maximizes the volume. -/
def optimal_length : ℝ := 2

/-- The height that maximizes the volume. -/
def optimal_height : ℝ := 1.5

/-- Theorem stating that the given dimensions maximize the volume of the rectangular frame. -/
theorem optimal_dimensions_maximize_volume :
  (∀ x, 0 < x → x < 3/2 → volume x ≤ max_volume) ∧
  volume optimal_width = max_volume ∧
  optimal_length = 2 * optimal_width ∧
  optimal_height = 4.5 - 3 * optimal_width :=
sorry

/-- Theorem stating that the maximum volume is unique. -/
theorem unique_maximum_volume :
  ∀ x, 0 < x → x < 3/2 → volume x = max_volume → x = optimal_width :=
sorry

end NUMINAMATH_CALUDE_optimal_dimensions_maximize_volume_unique_maximum_volume_l1227_122739


namespace NUMINAMATH_CALUDE_max_perpendicular_pairs_l1227_122747

/-- A line in a plane -/
structure Line

/-- A perpendicular pair of lines -/
structure PerpendicularPair (Line : Type) where
  line1 : Line
  line2 : Line

/-- A configuration of lines in a plane -/
structure PlaneConfiguration where
  lines : Finset Line
  perpendicular_pairs : Finset (PerpendicularPair Line)
  line_count : lines.card = 20

/-- The theorem stating the maximum number of perpendicular pairs -/
theorem max_perpendicular_pairs (config : PlaneConfiguration) :
  ∃ (max_config : PlaneConfiguration), 
    ∀ (c : PlaneConfiguration), c.perpendicular_pairs.card ≤ max_config.perpendicular_pairs.card ∧
    max_config.perpendicular_pairs.card = 100 :=
  sorry

end NUMINAMATH_CALUDE_max_perpendicular_pairs_l1227_122747


namespace NUMINAMATH_CALUDE_right_triangle_sine_roots_l1227_122738

theorem right_triangle_sine_roots (A B C : Real) (p q : Real) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  C = Real.pi / 2 →
  A + B + C = Real.pi →
  (∀ x, x^2 + p*x + q = 0 ↔ x = Real.sin A ∨ x = Real.sin B) →
  (p^2 - 2*q = 1 ∧ -Real.sqrt 2 ≤ p ∧ p < -1 ∧ 0 < q ∧ q ≤ 1/2) ∧
  (∀ x, x^2 + p*x + q = 0 → (x = Real.sin A ∨ x = Real.sin B)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sine_roots_l1227_122738


namespace NUMINAMATH_CALUDE_cube_sum_simplification_l1227_122799

theorem cube_sum_simplification (a b c : ℕ) (ha : a = 43) (hb : b = 26) (hc : c = 17) :
  (a^3 + c^3) / (a^3 + b^3) = (a + c) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_simplification_l1227_122799


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1227_122729

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 7 ∧ 
  (∃ n : ℕ, c = 2 * n + 1) ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1227_122729


namespace NUMINAMATH_CALUDE_roots_difference_is_one_l1227_122722

-- Define the polynomial
def f (x : ℝ) : ℝ := 64 * x^3 - 144 * x^2 + 92 * x - 15

-- Define the roots
def roots : Set ℝ := {x : ℝ | f x = 0}

-- Define the arithmetic progression property
def is_arithmetic_progression (s : Set ℝ) : Prop :=
  ∃ (a d : ℝ), s = {a - d, a, a + d}

-- Theorem statement
theorem roots_difference_is_one :
  is_arithmetic_progression roots →
  ∃ (r₁ r₂ r₃ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₃ ∈ roots ∧
  r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ - r₁ = 1 :=
sorry

end NUMINAMATH_CALUDE_roots_difference_is_one_l1227_122722


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1227_122764

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n with a_3 = 2 and a_5 = 8, prove that a_7 = 32 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a3 : a 3 = 2) 
    (h_a5 : a 5 = 8) : 
  a 7 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1227_122764


namespace NUMINAMATH_CALUDE_initial_drawer_pencils_count_l1227_122796

/-- The number of pencils initially in the drawer -/
def initial_drawer_pencils : ℕ := sorry

/-- The number of pencils initially on the desk -/
def initial_desk_pencils : ℕ := 19

/-- The number of pencils added to the desk -/
def added_desk_pencils : ℕ := 16

/-- The total number of pencils after the addition -/
def total_pencils : ℕ := 78

theorem initial_drawer_pencils_count : 
  initial_drawer_pencils = 43 :=
by sorry

end NUMINAMATH_CALUDE_initial_drawer_pencils_count_l1227_122796


namespace NUMINAMATH_CALUDE_g_is_odd_and_f_negative_two_l1227_122718

/-- The function f(x) -/
noncomputable def f (x m n : ℝ) : ℝ := (2^x - 2^(-x)) * m + (x^3 + x) * n + x^2 - 1

/-- The function g(x) -/
noncomputable def g (x m n : ℝ) : ℝ := (2^x - 2^(-x)) * m + (x^3 + x) * n

theorem g_is_odd_and_f_negative_two (m n : ℝ) :
  (∀ x, g (-x) m n = -g x m n) ∧ (f 2 m n = 8 → f (-2) m n = -2) :=
sorry

end NUMINAMATH_CALUDE_g_is_odd_and_f_negative_two_l1227_122718


namespace NUMINAMATH_CALUDE_students_interested_in_both_sports_and_music_l1227_122702

/-- Given a class with the following properties:
  * There are 55 students in total
  * 43 students are sports enthusiasts
  * 34 students are music enthusiasts
  * 4 students are neither interested in sports nor music
  Prove that 26 students are interested in both sports and music -/
theorem students_interested_in_both_sports_and_music 
  (total : ℕ) (sports : ℕ) (music : ℕ) (neither : ℕ) 
  (h_total : total = 55)
  (h_sports : sports = 43)
  (h_music : music = 34)
  (h_neither : neither = 4) :
  sports + music - (total - neither) = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_interested_in_both_sports_and_music_l1227_122702


namespace NUMINAMATH_CALUDE_translation_of_point_l1227_122787

/-- Given a point A with coordinates (-2, 3) in a Cartesian coordinate system,
    prove that translating it 3 units right and 5 units down results in
    point B with coordinates (1, -2). -/
theorem translation_of_point (A B : ℝ × ℝ) :
  A = (-2, 3) →
  B.1 = A.1 + 3 →
  B.2 = A.2 - 5 →
  B = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_point_l1227_122787


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1227_122713

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) + x * y = f x * f y) →
  (∀ x : ℝ, f x = 1 - x) ∨ (∀ x : ℝ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1227_122713


namespace NUMINAMATH_CALUDE_larger_interior_angle_measure_l1227_122723

/-- A circular monument consisting of congruent isosceles trapezoids. -/
structure CircularMonument where
  /-- The number of trapezoids in the monument. -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees. -/
  larger_interior_angle : ℝ

/-- The properties of the circular monument. -/
def monument_properties (m : CircularMonument) : Prop :=
  m.num_trapezoids = 12 ∧
  m.larger_interior_angle > 0 ∧
  m.larger_interior_angle < 180

/-- Theorem stating the measure of the larger interior angle in the monument. -/
theorem larger_interior_angle_measure (m : CircularMonument) 
  (h : monument_properties m) : m.larger_interior_angle = 97.5 := by
  sorry

#check larger_interior_angle_measure

end NUMINAMATH_CALUDE_larger_interior_angle_measure_l1227_122723


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l1227_122745

theorem min_value_sum_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (m : ℝ), m = 16 - 2 * Real.sqrt 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l1227_122745


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1227_122774

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) : (n - 3 = 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1227_122774


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1227_122733

/-- The line equation kx - y - 3k + 3 = 0 passes through the point (3,3) for all values of k. -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (3 : ℝ) * k - 3 - 3 * k + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1227_122733


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2891_l1227_122797

theorem cubic_equation_solutions (n : ℕ+) :
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n ∧
                (y-x)^3 - 3*(y-x)*(-x)^2 + (-x)^3 = n ∧
                (-y)^3 - 3*(-y)*(x-y)^2 + (x-y)^3 = n) :=
sorry

theorem no_solution_for_2891 :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2891 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2891_l1227_122797


namespace NUMINAMATH_CALUDE_orange_harvest_l1227_122703

theorem orange_harvest (days : ℕ) (total_sacks : ℕ) (sacks_per_day : ℕ) : 
  days = 6 → total_sacks = 498 → sacks_per_day * days = total_sacks → sacks_per_day = 83 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_l1227_122703


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l1227_122766

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l1227_122766


namespace NUMINAMATH_CALUDE_movie_ticket_price_l1227_122793

/-- The price of a movie ticket and nachos, where the nachos cost half the ticket price and the total is $24. -/
def MovieTheaterVisit : Type :=
  {ticket : ℚ // ∃ (nachos : ℚ), nachos = ticket / 2 ∧ ticket + nachos = 24}

/-- Theorem stating that the price of the movie ticket is $16. -/
theorem movie_ticket_price (visit : MovieTheaterVisit) : visit.val = 16 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_price_l1227_122793


namespace NUMINAMATH_CALUDE_factorization_3a_squared_minus_3_l1227_122700

theorem factorization_3a_squared_minus_3 (a : ℝ) : 3 * a^2 - 3 = 3 * (a - 1) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3a_squared_minus_3_l1227_122700


namespace NUMINAMATH_CALUDE_circle_triangle_perpendiculars_l1227_122754

-- Define the basic structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Define the triangle
structure Triangle := (A B C : Point)

-- Define the intersection points
structure IntersectionPoints := 
  (A₁ A₂ B₁ B₂ C₁ C₂ : Point)

-- Define a function to check if three lines are concurrent
def are_concurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define a function to create a perpendicular line
def perpendicular_at (l : Line) (p : Point) : Line := sorry

-- Main theorem
theorem circle_triangle_perpendiculars 
  (triangle : Triangle) 
  (circle : Circle) 
  (intersections : IntersectionPoints) : 
  are_concurrent 
    (perpendicular_at (Line.mk 0 1 0) intersections.A₁)
    (perpendicular_at (Line.mk 1 0 0) intersections.B₁)
    (perpendicular_at (Line.mk 1 1 0) intersections.C₁) →
  are_concurrent 
    (perpendicular_at (Line.mk 0 1 0) intersections.A₂)
    (perpendicular_at (Line.mk 1 0 0) intersections.B₂)
    (perpendicular_at (Line.mk 1 1 0) intersections.C₂) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_perpendiculars_l1227_122754


namespace NUMINAMATH_CALUDE_total_amount_earned_l1227_122775

/-- The total amount earned from selling rackets given the average price per pair and the number of pairs sold. -/
theorem total_amount_earned (avg_price : ℝ) (num_pairs : ℕ) : avg_price = 9.8 → num_pairs = 50 → avg_price * (num_pairs : ℝ) = 490 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_earned_l1227_122775


namespace NUMINAMATH_CALUDE_parabola_properties_l1227_122736

/-- A parabola with coefficient a < 0 intersecting x-axis at (-3,0) and (1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  h_root1 : a * (-3)^2 + b * (-3) + c = 0
  h_root2 : a * 1^2 + b * 1 + c = 0

theorem parabola_properties (p : Parabola) :
  (p.b^2 - 4 * p.a * p.c > 0) ∧ (3 * p.b + 2 * p.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1227_122736


namespace NUMINAMATH_CALUDE_line_slopes_problem_l1227_122720

theorem line_slopes_problem (k₁ k₂ b : ℝ) : 
  (2 * k₁^2 - 3 * k₁ - b = 0) → 
  (2 * k₂^2 - 3 * k₂ - b = 0) → 
  ((k₁ * k₂ = -1) → b = 2) ∧ 
  ((k₁ = k₂) → b = -9/8) := by
sorry

end NUMINAMATH_CALUDE_line_slopes_problem_l1227_122720


namespace NUMINAMATH_CALUDE_second_quadrant_angle_ratio_l1227_122724

theorem second_quadrant_angle_ratio (x : Real) : 
  (π/2 < x) ∧ (x < π) →  -- x is in the second quadrant
  (Real.tan x)^2 + 3*(Real.tan x) - 4 = 0 → 
  (Real.sin x + Real.cos x) / (2*(Real.sin x) - Real.cos x) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_second_quadrant_angle_ratio_l1227_122724


namespace NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_parallel_line_to_intersecting_planes_l1227_122708

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (in_plane : Line → Plane → Prop)

-- Theorem for proposition 2
theorem parallel_lines_in_parallel_planes
  (α β γ : Plane) (m n : Line) :
  parallel_plane α β →
  intersect α γ m →
  intersect β γ n →
  parallel m n :=
sorry

-- Theorem for proposition 4
theorem parallel_line_to_intersecting_planes
  (α β : Plane) (m n : Line) :
  intersect α β m →
  parallel m n →
  ¬in_plane n α →
  ¬in_plane n β →
  parallel_line_plane n α ∧ parallel_line_plane n β :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_parallel_line_to_intersecting_planes_l1227_122708


namespace NUMINAMATH_CALUDE_H_composition_equals_neg_one_l1227_122780

/-- The function H defined as H(x) = x^2 - 2x - 1 -/
def H (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Theorem stating that H(H(H(H(H(2))))) = -1 -/
theorem H_composition_equals_neg_one : H (H (H (H (H 2)))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_H_composition_equals_neg_one_l1227_122780


namespace NUMINAMATH_CALUDE_vector_magnitude_l1227_122761

theorem vector_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖b‖ = 5)
  (h2 : ‖2 • a + b‖ = 5 * Real.sqrt 3)
  (h3 : ‖a - b‖ = 5 * Real.sqrt 2) :
  ‖a‖ = 5 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1227_122761


namespace NUMINAMATH_CALUDE_division_problem_l1227_122706

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 1375 → 
  divisor = 66 → 
  remainder = 55 → 
  dividend = divisor * quotient + remainder → 
  quotient = 20 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1227_122706


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1227_122751

/-- The smallest area of a right triangle with one side 6 and another side x < 6 -/
theorem smallest_right_triangle_area :
  ∀ x : ℝ, x < 6 →
  (5 * Real.sqrt 11) / 2 ≤ min (3 * x) ((x * Real.sqrt (36 - x^2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1227_122751


namespace NUMINAMATH_CALUDE_complex_sum_equals_eleven_l1227_122765

/-- Given complex numbers a and b, prove that a + 3b = 11 -/
theorem complex_sum_equals_eleven (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + I) :
  a + 3*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_eleven_l1227_122765


namespace NUMINAMATH_CALUDE_largest_prime_factor_largest_prime_factor_of_expression_l1227_122725

theorem largest_prime_factor (n : ℕ) : ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p :=
  sorry

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (18^3 + 12^4 - 6^5) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (18^3 + 12^4 - 6^5) → q ≤ p ∧ p = 23 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_largest_prime_factor_of_expression_l1227_122725


namespace NUMINAMATH_CALUDE_max_value_quadratic_constraint_l1227_122788

theorem max_value_quadratic_constraint (x y z w : ℝ) 
  (h : 9*x^2 + 4*y^2 + 25*z^2 + 16*w^2 = 4) : 
  (∃ (a b c d : ℝ), 9*a^2 + 4*b^2 + 25*c^2 + 16*d^2 = 4 ∧ 
  2*a + 3*b + 5*c - 4*d = 6*Real.sqrt 6) ∧ 
  (∀ (x y z w : ℝ), 9*x^2 + 4*y^2 + 25*z^2 + 16*w^2 = 4 → 
  2*x + 3*y + 5*z - 4*w ≤ 6*Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_constraint_l1227_122788


namespace NUMINAMATH_CALUDE_new_tax_rate_is_28_percent_l1227_122742

/-- Calculates the new tax rate given the initial conditions --/
def calculate_new_tax_rate (initial_rate : ℚ) (income : ℚ) (savings : ℚ) : ℚ :=
  100 * (initial_rate * income - savings) / income

/-- Proves that the new tax rate is 28% given the initial conditions --/
theorem new_tax_rate_is_28_percent :
  let initial_rate : ℚ := 42 / 100
  let income : ℚ := 34500
  let savings : ℚ := 4830
  calculate_new_tax_rate initial_rate income savings = 28 := by
  sorry

#eval calculate_new_tax_rate (42/100) 34500 4830

end NUMINAMATH_CALUDE_new_tax_rate_is_28_percent_l1227_122742


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l1227_122717

theorem final_sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l1227_122717


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1227_122719

/-- A quadratic function passing through three given points -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_proof (a b c : ℝ) :
  (quadratic_function a b c 1 = 5) ∧
  (quadratic_function a b c 0 = 3) ∧
  (quadratic_function a b c (-1) = -3) →
  (∀ x, quadratic_function a b c x = -2 * x^2 + 4 * x + 3) ∧
  (∃ x y, x = 1 ∧ y = 5 ∧ ∀ t, quadratic_function a b c t ≤ quadratic_function a b c x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1227_122719


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l1227_122716

noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_between_curves :
  ∃ (min_val : ℝ), min_val = (1/3 : ℝ) + (1/3 : ℝ) * Real.log 3 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l1227_122716


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1227_122779

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, (x = a ∧ y = b) ∨ (x = b ∧ y = a) → x * y = k

theorem inverse_proportion_problem (a b : ℝ) :
  InverselyProportional a b →
  (∃ a₀ b₀ : ℝ, a₀ + b₀ = 60 ∧ a₀ = 3 * b₀ ∧ InverselyProportional a₀ b₀) →
  (a = -12 → b = -225/4) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1227_122779


namespace NUMINAMATH_CALUDE_second_largest_of_5_8_4_l1227_122785

def second_largest (a b c : ℕ) : ℕ :=
  if a ≥ b ∧ b ≥ c then b
  else if a ≥ c ∧ c ≥ b then c
  else if b ≥ a ∧ a ≥ c then a
  else if b ≥ c ∧ c ≥ a then c
  else if c ≥ a ∧ a ≥ b then a
  else b

theorem second_largest_of_5_8_4 : second_largest 5 8 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_of_5_8_4_l1227_122785


namespace NUMINAMATH_CALUDE_towel_sets_cost_l1227_122731

def guest_sets : ℕ := 2
def master_sets : ℕ := 4
def guest_price : ℚ := 40
def master_price : ℚ := 50
def discount_rate : ℚ := 0.2

def total_cost : ℚ := guest_sets * guest_price + master_sets * master_price
def discount_amount : ℚ := total_cost * discount_rate
def final_cost : ℚ := total_cost - discount_amount

theorem towel_sets_cost : final_cost = 224 := by
  sorry

end NUMINAMATH_CALUDE_towel_sets_cost_l1227_122731


namespace NUMINAMATH_CALUDE_technician_salary_l1227_122792

theorem technician_salary (total_workers : Nat) (technicians : Nat) (avg_salary : ℕ) 
  (non_tech_avg : ℕ) (h1 : total_workers = 12) (h2 : technicians = 6) 
  (h3 : avg_salary = 9000) (h4 : non_tech_avg = 6000) : 
  (total_workers * avg_salary - (total_workers - technicians) * non_tech_avg) / technicians = 12000 :=
sorry

end NUMINAMATH_CALUDE_technician_salary_l1227_122792


namespace NUMINAMATH_CALUDE_max_different_ages_is_17_l1227_122767

/-- Represents the problem of finding the maximum number of different ages within a range --/
def MaxDifferentAges (average : ℕ) (stdDev : ℕ) : ℕ :=
  (average + stdDev) - (average - stdDev) + 1

/-- Theorem stating that for the given conditions, the maximum number of different ages is 17 --/
theorem max_different_ages_is_17 (average : ℕ) (stdDev : ℕ)
    (h_average : average = 20)
    (h_stdDev : stdDev = 8) :
    MaxDifferentAges average stdDev = 17 := by
  sorry

#eval MaxDifferentAges 20 8  -- Should output 17

end NUMINAMATH_CALUDE_max_different_ages_is_17_l1227_122767


namespace NUMINAMATH_CALUDE_flagstaff_height_is_correct_l1227_122727

/-- The height of the flagstaff in meters -/
def flagstaff_height : ℝ := 17.5

/-- The length of the flagstaff's shadow in meters -/
def flagstaff_shadow : ℝ := 40.25

/-- The height of the building in meters -/
def building_height : ℝ := 12.5

/-- The length of the building's shadow in meters -/
def building_shadow : ℝ := 28.75

/-- Theorem stating that the calculated flagstaff height is correct -/
theorem flagstaff_height_is_correct :
  flagstaff_height = (building_height * flagstaff_shadow) / building_shadow :=
by sorry

end NUMINAMATH_CALUDE_flagstaff_height_is_correct_l1227_122727


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l1227_122721

theorem revenue_change_after_price_and_quantity_change 
  (P Q : ℝ) (P_new Q_new R R_new : ℝ) 
  (h1 : P_new = 0.8 * P) 
  (h2 : Q_new = 1.6 * Q) 
  (h3 : R = P * Q) 
  (h4 : R_new = P_new * Q_new) : 
  R_new = 1.28 * R := by sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l1227_122721


namespace NUMINAMATH_CALUDE_train_crossing_time_l1227_122709

/-- Proves that the time taken for a train to cross a bridge is 20 seconds -/
theorem train_crossing_time (bridge_length : ℝ) (train_length : ℝ) (train_speed : ℝ) :
  bridge_length = 180 →
  train_length = 120 →
  train_speed = 15 →
  (bridge_length + train_length) / train_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1227_122709


namespace NUMINAMATH_CALUDE_equation_with_integer_roots_l1227_122791

/-- Given that the equation (x-a)(x-8) - 1 = 0 has two integer roots, prove that a = 8 -/
theorem equation_with_integer_roots (a : ℤ) :
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ - a) * (x₁ - 8) - 1 = 0 ∧ (x₂ - a) * (x₂ - 8) - 1 = 0) →
  a = 8 := by
  sorry


end NUMINAMATH_CALUDE_equation_with_integer_roots_l1227_122791


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1227_122715

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  (∃ k : ℕ, k > 12 ∧ (∀ m : ℕ, m > 0 → k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) →
  False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1227_122715


namespace NUMINAMATH_CALUDE_winning_strategy_extends_l1227_122762

/-- Represents the winning player for a given game state -/
inductive Winner : Type
  | Player1 : Winner
  | Player2 : Winner

/-- Represents the game state -/
structure GameState :=
  (t : ℕ)  -- Current number on the blackboard
  (a : ℕ)  -- First subtraction option
  (b : ℕ)  -- Second subtraction option

/-- Determines the winner of the game given a game state -/
def winningPlayer (state : GameState) : Winner :=
  sorry

/-- Theorem stating that if Player 1 wins for x, they also win for x + 2005k -/
theorem winning_strategy_extends (x k a b : ℕ) :
  (1 ≤ x) →
  (x ≤ 2005) →
  (0 < a) →
  (0 < b) →
  (a + b = 2005) →
  (winningPlayer { t := x, a := a, b := b } = Winner.Player1) →
  (winningPlayer { t := x + 2005 * k, a := a, b := b } = Winner.Player1) :=
by
  sorry

end NUMINAMATH_CALUDE_winning_strategy_extends_l1227_122762


namespace NUMINAMATH_CALUDE_janine_read_150_pages_l1227_122748

/-- The number of pages Janine read in two months -/
def pages_read_in_two_months (books_last_month : ℕ) (books_this_month_factor : ℕ) (pages_per_book : ℕ) : ℕ :=
  (books_last_month + books_last_month * books_this_month_factor) * pages_per_book

/-- Theorem stating that Janine read 150 pages in two months -/
theorem janine_read_150_pages :
  pages_read_in_two_months 5 2 10 = 150 := by
  sorry

#eval pages_read_in_two_months 5 2 10

end NUMINAMATH_CALUDE_janine_read_150_pages_l1227_122748


namespace NUMINAMATH_CALUDE_baseball_cards_per_pack_l1227_122704

theorem baseball_cards_per_pack : 
  ∀ (num_people : ℕ) (cards_per_person : ℕ) (total_packs : ℕ),
    num_people = 4 →
    cards_per_person = 540 →
    total_packs = 108 →
    (num_people * cards_per_person) / total_packs = 20 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_per_pack_l1227_122704


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1227_122782

/-- Given a geometric sequence {a_n} where a_2010 = 8a_2007, prove that the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (h : a 2010 = 8 * a 2007) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1227_122782


namespace NUMINAMATH_CALUDE_discount_order_difference_l1227_122776

/-- The difference in final price when applying discounts in different orders -/
theorem discount_order_difference : 
  let original_price : ℚ := 25
  let flat_discount : ℚ := 4
  let percentage_discount : ℚ := 0.2
  let price_flat_then_percent : ℚ := (original_price - flat_discount) * (1 - percentage_discount)
  let price_percent_then_flat : ℚ := (original_price * (1 - percentage_discount)) - flat_discount
  (price_flat_then_percent - price_percent_then_flat) * 100 = 80
  := by sorry

end NUMINAMATH_CALUDE_discount_order_difference_l1227_122776


namespace NUMINAMATH_CALUDE_complex_real_condition_l1227_122760

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (z.im = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1227_122760


namespace NUMINAMATH_CALUDE_min_sum_squares_l1227_122714

def S : Finset Int := {-6, -4, -3, -1, 1, 3, 5, 8}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 5) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1227_122714


namespace NUMINAMATH_CALUDE_percentage_same_grade_l1227_122728

def total_students : ℕ := 50

def same_grade_A : ℕ := 3
def same_grade_B : ℕ := 6
def same_grade_C : ℕ := 8
def same_grade_D : ℕ := 2
def same_grade_F : ℕ := 1

def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D + same_grade_F

theorem percentage_same_grade : 
  (total_same_grade : ℚ) / total_students * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_same_grade_l1227_122728


namespace NUMINAMATH_CALUDE_remaining_pencils_l1227_122790

/-- The number of pencils remaining in a drawer after some are taken. -/
def pencils_remaining (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem stating that 12 pencils remain in the drawer. -/
theorem remaining_pencils :
  pencils_remaining 34 22 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pencils_l1227_122790


namespace NUMINAMATH_CALUDE_universal_set_determination_l1227_122770

universe u

theorem universal_set_determination (U : Set ℕ) (A : Set ℕ) (h1 : A = {1, 3, 5})
  (h2 : Set.compl A = {2, 4, 6}) : U = {1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_universal_set_determination_l1227_122770


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l1227_122746

theorem smallest_positive_largest_negative_smallest_abs_rational :
  ∃ (x y : ℤ) (z : ℚ),
    (∀ n : ℤ, n > 0 → x ≤ n) ∧
    (∀ n : ℤ, n < 0 → y ≥ n) ∧
    (∀ q : ℚ, |z| ≤ |q|) ∧
    2 * x + 3 * y + 4 * z = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l1227_122746


namespace NUMINAMATH_CALUDE_function_value_at_five_l1227_122701

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3g(1-x) = 4x^2 - 1 for all x,
    prove that g(5) = 11.25 -/
theorem function_value_at_five (g : ℝ → ℝ) 
    (h : ∀ x, g x + 3 * g (1 - x) = 4 * x^2 - 1) : 
    g 5 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_five_l1227_122701


namespace NUMINAMATH_CALUDE_expression_simplification_l1227_122730

theorem expression_simplification (a : ℤ) (h : a = 2020) : 
  (a^4 - 3*a^3*(a+1) + 4*a^2*(a+1)^2 - (a+1)^4 + 1) / (a*(a+1)) = a^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1227_122730


namespace NUMINAMATH_CALUDE_household_spending_theorem_l1227_122784

/-- The number of households that did not spend at least $150 per month on electricity, natural gas, or water -/
def x : ℕ := 46

/-- The total number of households surveyed -/
def total_households : ℕ := 500

/-- Households spending ≥$150 on both electricity and gas -/
def both_elec_gas : ℕ := 160

/-- Households spending ≥$150 on electricity but not gas -/
def elec_not_gas : ℕ := 75

/-- Households spending ≥$150 on gas but not electricity -/
def gas_not_elec : ℕ := 80

theorem household_spending_theorem :
  x + 3 * x + both_elec_gas + elec_not_gas + gas_not_elec = total_households :=
sorry

end NUMINAMATH_CALUDE_household_spending_theorem_l1227_122784


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1227_122712

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_constraint : 6 * a + 5 * b = 45) :
  a * b ≤ 135 / 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 6 * a₀ + 5 * b₀ = 45 ∧ a₀ * b₀ = 135 / 8 :=
by sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1227_122712


namespace NUMINAMATH_CALUDE_power_five_mod_150_l1227_122755

theorem power_five_mod_150 : 5^2023 % 150 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_150_l1227_122755


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l1227_122734

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Theorem for part II
theorem range_of_m (h : ∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) :
  -1/2 < m ∧ m < 5/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l1227_122734


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l1227_122758

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := 12.14

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The total amount Sandy spent on clothes -/
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

/-- Theorem stating that the total amount Sandy spent on clothes is $33.56 -/
theorem sandy_clothes_cost : total_cost = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l1227_122758


namespace NUMINAMATH_CALUDE_inequality_proof_l1227_122756

theorem inequality_proof (x y z t : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ t ≥ 0) 
  (h_sum : x + y + z + t = 4) : 
  Real.sqrt (x^2 + t^2) + Real.sqrt (z^2 + 1) + Real.sqrt (z^2 + t^2) + 
  Real.sqrt (y^2 + x^2) + Real.sqrt (y^2 + 64) ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1227_122756


namespace NUMINAMATH_CALUDE_larger_number_puzzle_l1227_122781

theorem larger_number_puzzle (x y : ℕ) : 
  x * y = 18 → x + y = 13 → max x y = 9 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_puzzle_l1227_122781


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1227_122744

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 5 → (∀ k : ℕ, k > 5 ∧ k < b → ¬∃ n : ℕ, 4 * k + 5 = n^2) → 
  ∃ n : ℕ, 4 * b + 5 = n^2 → b = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1227_122744


namespace NUMINAMATH_CALUDE_exist_numbers_with_digit_sum_property_l1227_122773

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- Theorem stating the existence of numbers satisfying the given conditions -/
theorem exist_numbers_with_digit_sum_property : 
  ∃ (a b c : ℕ), 
    S (a + b) < 5 ∧ 
    S (a + c) < 5 ∧ 
    S (b + c) < 5 ∧ 
    S (a + b + c) > 50 := by
  sorry

end NUMINAMATH_CALUDE_exist_numbers_with_digit_sum_property_l1227_122773


namespace NUMINAMATH_CALUDE_sin_cos_product_given_tan_l1227_122705

theorem sin_cos_product_given_tan (θ : Real) (h : Real.tan θ = 2) :
  Real.sin θ * Real.cos θ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_given_tan_l1227_122705


namespace NUMINAMATH_CALUDE_complex_power_difference_l1227_122740

theorem complex_power_difference (x : ℂ) : 
  x - (1 / x) = 3 * Complex.I → x^3375 - (1 / x^3375) = -18 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1227_122740


namespace NUMINAMATH_CALUDE_road_trip_days_l1227_122726

/-- Proves that the number of days of the road trip is 3, given the driving hours of Jade and Krista and the total hours driven. -/
theorem road_trip_days (jade_hours krista_hours total_hours : ℕ) 
  (h1 : jade_hours = 8)
  (h2 : krista_hours = 6)
  (h3 : total_hours = 42) :
  (total_hours : ℚ) / (jade_hours + krista_hours : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_days_l1227_122726


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l1227_122737

/-- Given y = x^2 + 1/x^2, prove that x^6 + x^4 - 5x^3 + x^2 + 1 = 0 is equivalent to x^3(y+1) + y = -5x^3 -/
theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x^2 + 1/x^2) :
  x^6 + x^4 - 5*x^3 + x^2 + 1 = 0 ↔ x^3*(y+1) + y = -5*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l1227_122737


namespace NUMINAMATH_CALUDE_total_grapes_is_83_l1227_122750

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_grapes_is_83_l1227_122750


namespace NUMINAMATH_CALUDE_sequence_range_l1227_122707

theorem sequence_range (a : ℝ) : 
  (∀ n : ℕ+, (fun n => if n < 6 then (1/2 - a) * n + 1 else a^(n - 5)) n > 
             (fun n => if n < 6 then (1/2 - a) * n + 1 else a^(n - 5)) (n + 1)) → 
  (1/2 < a ∧ a < 7/12) := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l1227_122707


namespace NUMINAMATH_CALUDE_robin_gum_count_l1227_122777

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 9

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 135 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l1227_122777


namespace NUMINAMATH_CALUDE_cookies_eaten_l1227_122735

theorem cookies_eaten (charlie_cookies : ℕ) (father_cookies : ℕ) (mother_cookies : ℕ)
  (h1 : charlie_cookies = 15)
  (h2 : father_cookies = 10)
  (h3 : mother_cookies = 5) :
  charlie_cookies + father_cookies + mother_cookies = 30 := by
sorry

end NUMINAMATH_CALUDE_cookies_eaten_l1227_122735


namespace NUMINAMATH_CALUDE_magnitude_relationship_l1227_122741

theorem magnitude_relationship
  (a b c d : ℝ)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_positive : d > 0)
  (x : ℝ)
  (h_x : x = Real.sqrt (a * b) + Real.sqrt (c * d))
  (y : ℝ)
  (h_y : y = Real.sqrt (a * c) + Real.sqrt (b * d))
  (z : ℝ)
  (h_z : z = Real.sqrt (a * d) + Real.sqrt (b * c)) :
  x > y ∧ y > z :=
by sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l1227_122741


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1227_122732

theorem right_triangle_shorter_leg :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →
  c = 65 →
  a ≤ b →
  a = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1227_122732


namespace NUMINAMATH_CALUDE_sally_buttons_count_l1227_122798

/-- The number of buttons Sally needs for all shirts -/
def total_buttons (monday_shirts tuesday_shirts wednesday_shirts buttons_per_shirt : ℕ) : ℕ :=
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt

/-- Theorem stating that Sally needs 45 buttons for all shirts -/
theorem sally_buttons_count : total_buttons 4 3 2 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_count_l1227_122798


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l1227_122749

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (∀ (N : ℕ), N > 0 ∧ 
    N % 7 = 6 ∧
    N % 8 = 7 ∧
    N % 9 = 8 ∧
    N % 10 = 9 ∧
    N % 11 = 10 ∧
    N % 12 = 11 → M ≤ N) ∧
  M = 27719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l1227_122749


namespace NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_implies_perpendicular_l1227_122789

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop := sorry

/-- Theorem: If a line is parallel to a plane and another line is perpendicular to the same plane, 
    then the two lines are perpendicular to each other -/
theorem line_parallel_plane_perpendicular_implies_perpendicular 
  (m n : Line3D) (α : Plane3D) 
  (h1 : parallel m α) 
  (h2 : perpendicular_line_plane n α) : 
  perpendicular_lines m n := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_implies_perpendicular_l1227_122789


namespace NUMINAMATH_CALUDE_carton_width_l1227_122759

/-- Represents the dimensions of a rectangular carton -/
structure CartonDimensions where
  length : ℝ
  width : ℝ

/-- Given a carton with dimensions 25 inches by 60 inches, its width is 25 inches -/
theorem carton_width (c : CartonDimensions) 
  (h1 : c.length = 60) 
  (h2 : c.width = 25) : 
  c.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_carton_width_l1227_122759


namespace NUMINAMATH_CALUDE_tangent_ratio_inequality_l1227_122752

theorem tangent_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  Real.tan α / α < Real.tan β / β := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_inequality_l1227_122752


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1227_122743

theorem cubic_polynomials_common_roots :
  ∃! (c d : ℝ), ∃ (r s : ℝ),
    r ≠ s ∧
    (r^3 + c*r^2 + 17*r + 10 = 0) ∧
    (r^3 + d*r^2 + 22*r + 14 = 0) ∧
    (s^3 + c*s^2 + 17*s + 10 = 0) ∧
    (s^3 + d*s^2 + 22*s + 14 = 0) ∧
    (∀ (x : ℝ), x ≠ r ∧ x ≠ s →
      (x^3 + c*x^2 + 17*x + 10 ≠ 0) ∨
      (x^3 + d*x^2 + 22*x + 14 ≠ 0)) ∧
    c = 8 ∧
    d = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1227_122743


namespace NUMINAMATH_CALUDE_power_calculation_l1227_122771

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1227_122771


namespace NUMINAMATH_CALUDE_largest_package_size_l1227_122795

def markers_elliot : ℕ := 60
def markers_tara : ℕ := 36
def markers_sam : ℕ := 90

theorem largest_package_size : ∃ (n : ℕ), n > 0 ∧ 
  markers_elliot % n = 0 ∧ 
  markers_tara % n = 0 ∧ 
  markers_sam % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (markers_elliot % m = 0 ∧ 
     markers_tara % m = 0 ∧ 
     markers_sam % m = 0) → False :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1227_122795


namespace NUMINAMATH_CALUDE_expression_simplification_l1227_122783

theorem expression_simplification (y : ℝ) : 3*y + 4*y^2 + 2 - (8 - 3*y - 4*y^2) = 8*y^2 + 6*y - 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1227_122783


namespace NUMINAMATH_CALUDE_first_three_consecutive_fives_l1227_122794

/-- The sequence of digits formed by concatenating natural numbers -/
def digitSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => sorry  -- Definition of the sequence

/-- The function that returns the digit at a given position in the sequence -/
def digitAt (position : ℕ) : ℕ := sorry

/-- Theorem stating the positions of the first occurrence of three consecutive '5' digits -/
theorem first_three_consecutive_fives :
  ∃ (start : ℕ), start = 100 ∧
    digitAt start = 5 ∧
    digitAt (start + 1) = 5 ∧
    digitAt (start + 2) = 5 ∧
    (∀ (pos : ℕ), pos < start → ¬(digitAt pos = 5 ∧ digitAt (pos + 1) = 5 ∧ digitAt (pos + 2) = 5)) :=
  sorry


end NUMINAMATH_CALUDE_first_three_consecutive_fives_l1227_122794


namespace NUMINAMATH_CALUDE_certain_number_proof_l1227_122753

theorem certain_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = 127 * k + 6) →
  (∃ m : ℕ, 2037 = 127 * m + 5) →
  (∀ d : ℕ, d > 127 → (n % d ≠ 6 ∨ 2037 % d ≠ 5)) →
  n = 2038 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1227_122753


namespace NUMINAMATH_CALUDE_supplement_double_complement_30_l1227_122711

def original_angle : ℝ := 30

-- Define complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define double
def double (x : ℝ) : ℝ := 2 * x

-- Define supplement
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_double_complement_30 : 
  supplement (double (complement original_angle)) = 60 := by sorry

end NUMINAMATH_CALUDE_supplement_double_complement_30_l1227_122711


namespace NUMINAMATH_CALUDE_marys_speed_l1227_122768

theorem marys_speed (mary_hill_length ann_hill_length ann_speed time_difference : ℝ) 
  (h1 : mary_hill_length = 630)
  (h2 : ann_hill_length = 800)
  (h3 : ann_speed = 40)
  (h4 : time_difference = 13)
  (h5 : ann_hill_length / ann_speed = mary_hill_length / mary_speed + time_difference) :
  mary_speed = 90 :=
by
  sorry

#check marys_speed

end NUMINAMATH_CALUDE_marys_speed_l1227_122768


namespace NUMINAMATH_CALUDE_zoo_zebra_count_l1227_122786

theorem zoo_zebra_count :
  ∀ (penguins zebras tigers zookeepers : ℕ),
    penguins = 30 →
    tigers = 8 →
    zookeepers = 12 →
    (penguins + zebras + tigers + zookeepers) = 
      (2 * penguins + 4 * zebras + 4 * tigers + 2 * zookeepers) - 132 →
    zebras = 22 := by
  sorry

end NUMINAMATH_CALUDE_zoo_zebra_count_l1227_122786


namespace NUMINAMATH_CALUDE_intersection_A_B_values_a_b_l1227_122772

-- Define sets A and B
def A : Set ℝ := {x | 4 - x^2 > 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 1} := by sorry

-- Define the quadratic inequality
def quadratic_inequality (a b : ℝ) : Set ℝ := {x | 2*x^2 + a*x + b < 0}

-- Theorem for the values of a and b
theorem values_a_b : 
  ∃ a b : ℝ, quadratic_inequality a b = B ∧ a = 4 ∧ b = -6 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_values_a_b_l1227_122772


namespace NUMINAMATH_CALUDE_largest_square_area_l1227_122778

theorem largest_square_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a^2 + b^2 + c^2 = 450 →  -- sum of square areas
  a^2 = 100 →  -- area of square on AB
  c^2 = 225 :=  -- area of largest square (on BC)
by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l1227_122778


namespace NUMINAMATH_CALUDE_concept_laws_theorem_l1227_122757

/-- Probability of M laws being included in the Concept -/
def prob_M_laws_included (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- Expected number of laws included in the Concept -/
def expected_laws_included (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the probability of M laws being included and the expected number of laws -/
theorem concept_laws_theorem (K N M : ℕ) (p : ℝ) 
    (hK : K > 0) (hN : N > 0) (hM : M ≤ K) (hp : 0 ≤ p ∧ p ≤ 1) :
  prob_M_laws_included K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) ∧
  expected_laws_included K N p = K * (1 - (1 - p)^N) := by
  sorry

end NUMINAMATH_CALUDE_concept_laws_theorem_l1227_122757


namespace NUMINAMATH_CALUDE_exists_square_between_consecutive_prime_sums_l1227_122763

-- Define S_n as the sum of the first n prime numbers
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_square_between_consecutive_prime_sums : 
  ∃ k : ℕ, S 2023 < k^2 ∧ k^2 < S 2024 := by sorry

end NUMINAMATH_CALUDE_exists_square_between_consecutive_prime_sums_l1227_122763
